"""Export a checkpoint to standard formats so other tools can run it.

* Hugging Face (`--format hf`): safetensors weights + config.json + tokenizer.json.
  The v2 architecture (RMSNorm, SwiGLU, RoPE, GQA, no biases, tied embeddings) is exactly
  Llama's, so the model loads with `transformers.AutoModelForCausalLM`; with QK-norm it is
  exported as Qwen3 (Llama + per-head q/k RMSNorm).
* GGUF (`--format gguf`): the single-file format of llama.cpp, Ollama and LM Studio.
  The tokenizer is byte-level BPE with the GPT-2 pre-tokenizer ("gpt-2" in llama.cpp),
  so tokenization is identical when the checkpoint uses the "gpt2" split pattern.

    python -m llm.export --ckpt models/corpus-base.pt --format hf --out export/hf
    python -m llm.export --ckpt models/corpus-chat.pt --format gguf --out export/oz-chat.gguf
    # Ollama: see export/Modelfile written next to the .gguf

Models using options outside these architectures (value residual, U-Net skips, MoE,
logit soft-cap, learned positions, LayerNorm, ReLU/GELU MLP, biases) cannot be exported.
"""
import argparse
import json
import os

import torch

from .checkpoint import load_checkpoint
from .tokenizer import SPLIT_PATTERNS, BPETokenizer


def bytes_to_unicode():
    """GPT-2's reversible byte -> printable unicode character mapping (used by byte-level
    BPE vocab files so every token is a valid string)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


def check_exportable(config):
    bad = []
    if config.pos_emb != "rope":
        bad.append("learned positions")
    if config.norm != "rms":
        bad.append("LayerNorm")
    if config.mlp != "swiglu":
        bad.append(f"{config.mlp} MLP")
    for flag in ("bias", "value_residual", "unet_skips"):
        if getattr(config, flag):
            bad.append(flag)
    if config.n_experts:
        bad.append("mixture of experts")
    if config.logit_softcap:
        bad.append("logit soft-cap")
    if bad:
        raise ValueError("cannot export a model with: " + ", ".join(bad))


def token_strings(tokenizer):
    """Vocabulary as byte-level unicode strings, indexed by token id, plus merges."""
    assert isinstance(tokenizer, BPETokenizer), "export needs a BPE tokenizer"
    b2u = bytes_to_unicode()
    enc = lambda bs: "".join(b2u[b] for b in bs)  # noqa: E731
    vocab = [enc(tokenizer.vocab[i]) for i in range(256 + len(tokenizer.merges))]
    vocab += list(tokenizer.special_tokens)
    merges = [(enc(tokenizer.vocab[a]), enc(tokenizer.vocab[b])) for a, b in tokenizer.merges]
    return vocab, merges


def split_weights(model):
    """Yield (layer, name, tensor) in Llama naming; q/k/v split out of the fused qkv."""
    c = model.config
    hd = c.n_embd // c.n_head
    for i, blk in enumerate(model.blocks):
        q, k, v = blk.attn.qkv.weight.detach().split([c.n_head * hd, c.n_kv_head * hd, c.n_kv_head * hd], dim=0)
        yield i, "q", q
        yield i, "k", k
        yield i, "v", v
        yield i, "o", blk.attn.proj.weight.detach()
        yield i, "attn_norm", blk.norm1.weight.detach()
        yield i, "ffn_norm", blk.norm2.weight.detach()
        yield i, "gate", blk.mlp.gate.weight.detach()
        yield i, "up", blk.mlp.up.weight.detach()
        yield i, "down", blk.mlp.down.weight.detach()
        if c.qk_norm:
            yield i, "q_norm", blk.attn.q_norm.weight.detach()
            yield i, "k_norm", blk.attn.k_norm.weight.detach()


# ----------------------------------------------------------------------------- Hugging Face
HF_NAMES = {"q": "self_attn.q_proj.weight", "k": "self_attn.k_proj.weight", "v": "self_attn.v_proj.weight",
            "o": "self_attn.o_proj.weight", "attn_norm": "input_layernorm.weight",
            "ffn_norm": "post_attention_layernorm.weight", "gate": "mlp.gate_proj.weight",
            "up": "mlp.up_proj.weight", "down": "mlp.down_proj.weight",
            "q_norm": "self_attn.q_norm.weight", "k_norm": "self_attn.k_norm.weight"}


def export_hf(model, tokenizer, out_dir):
    from safetensors.torch import save_file
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    c = model.config
    os.makedirs(out_dir, exist_ok=True)
    tensors = {"model.embed_tokens.weight": model.tok_emb.weight.detach(),
               "model.norm.weight": model.norm_f.weight.detach()}
    if not c.tie_weights:
        tensors["lm_head.weight"] = model.lm_head.weight.detach()
    for i, name, t in split_weights(model):
        tensors[f"model.layers.{i}.{HF_NAMES[name]}"] = t
    save_file({k: v.contiguous().float() for k, v in tensors.items()}, os.path.join(out_dir, "model.safetensors"))

    eot = tokenizer.special_tokens["<|endoftext|>"]
    arch = "Qwen3ForCausalLM" if c.qk_norm else "LlamaForCausalLM"
    config = {
        "architectures": [arch], "model_type": "qwen3" if c.qk_norm else "llama",
        "vocab_size": c.vocab_size, "hidden_size": c.n_embd, "intermediate_size": model.blocks[0].mlp.up.out_features,
        "num_hidden_layers": c.n_layer, "num_attention_heads": c.n_head, "num_key_value_heads": c.n_kv_head,
        "head_dim": c.n_embd // c.n_head, "max_position_embeddings": c.block_size, "rms_norm_eps": 1e-6,
        "rope_theta": c.rope_theta, "hidden_act": "silu", "tie_word_embeddings": c.tie_weights,
        "attention_bias": False, "mlp_bias": False, "bos_token_id": eot, "eos_token_id": eot,
        "torch_dtype": "float32",
    }
    if c.rope_scaling:
        config["rope_scaling"] = {"rope_type": "linear" if c.rope_scaling == "linear" else "dynamic",
                                  "factor": c.rope_factor}
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    vocab, merges = token_strings(tokenizer)
    tok = Tokenizer(models.BPE(vocab={s: i for i, s in enumerate(vocab)}, merges=merges, ignore_merges=False))
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(SPLIT_PATTERNS[tokenizer.pattern]), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(list(tokenizer.special_tokens))
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    with open(os.path.join(out_dir, "tokenizer_config.json"), "w") as f:
        json.dump({"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|endoftext|>",
                   "eos_token": "<|endoftext|>", "model_max_length": c.block_size,
                   "chat_template": "{% for m in messages %}{% if m['role'] == 'user' %}<|user|>{{ m['content'] }}"
                                    "{% else %}<|assistant|>{{ m['content'] }}<|endoftext|>{% endif %}{% endfor %}"
                                    "{% if add_generation_prompt %}<|assistant|>{% endif %}"}, f, indent=2)
    return out_dir


# ----------------------------------------------------------------------------- GGUF
GGUF_NAMES = {"q": "attn_q", "k": "attn_k", "v": "attn_v", "o": "attn_output", "attn_norm": "attn_norm",
              "ffn_norm": "ffn_norm", "gate": "ffn_gate", "up": "ffn_up", "down": "ffn_down",
              "q_norm": "attn_q_norm", "k_norm": "attn_k_norm"}


def permute_for_llama_cpp(w, n_head):
    """llama.cpp's "llama" architecture rotates interleaved pairs (x0,x1),(x2,x3)…, while this
    model (like HF) rotates the two halves; reorder q/k output rows to match (same as
    llama.cpp's convert_hf_to_gguf.py)."""
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def export_gguf(model, tokenizer, path, name="llm-from-scratch"):
    import gguf
    import numpy as np
    c = model.config
    arch = "qwen3" if c.qk_norm else "llama"
    w = gguf.GGUFWriter(path, arch)
    w.add_name(name)
    w.add_context_length(c.block_size)
    w.add_embedding_length(c.n_embd)
    w.add_block_count(c.n_layer)
    w.add_feed_forward_length(model.blocks[0].mlp.up.out_features)
    w.add_head_count(c.n_head)
    w.add_head_count_kv(c.n_kv_head)
    w.add_rope_dimension_count(c.n_embd // c.n_head)
    w.add_rope_freq_base(c.rope_theta)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_vocab_size(c.vocab_size)
    w.add_file_type(gguf.LlamaFileType.ALL_F32)
    if c.rope_scaling:
        w.add_rope_scaling_type(gguf.RopeScalingType.LINEAR if c.rope_scaling == "linear" else gguf.RopeScalingType.YARN)
        w.add_rope_scaling_factor(c.rope_factor)

    vocab, merges = token_strings(tokenizer)
    n_normal = 256 + len(tokenizer.merges)
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2" if tokenizer.pattern == "gpt2" else "default")
    w.add_token_list(vocab)
    w.add_token_types([gguf.TokenType.NORMAL] * n_normal + [gguf.TokenType.CONTROL] * len(tokenizer.special_tokens))
    w.add_token_merges([f"{a} {b}" for a, b in merges])
    eot = tokenizer.special_tokens["<|endoftext|>"]
    w.add_bos_token_id(eot)
    w.add_eos_token_id(eot)
    w.add_add_bos_token(False)
    w.add_chat_template("{% for m in messages %}{% if m['role'] == 'user' %}<|user|>{{ m['content'] }}"
                        "{% else %}<|assistant|>{{ m['content'] }}<|endoftext|>{% endif %}{% endfor %}"
                        "{% if add_generation_prompt %}<|assistant|>{% endif %}")

    def add(name, t):
        w.add_tensor(name, t.float().contiguous().numpy().astype(np.float32))

    add("token_embd.weight", model.tok_emb.weight.detach())
    add("output_norm.weight", model.norm_f.weight.detach())
    if not c.tie_weights:
        add("output.weight", model.lm_head.weight.detach())
    for i, kind, t in split_weights(model):
        if arch == "llama" and kind == "q":
            t = permute_for_llama_cpp(t, c.n_head)
        elif arch == "llama" and kind == "k":
            t = permute_for_llama_cpp(t, c.n_kv_head)
        add(f"blk.{i}.{GGUF_NAMES[kind]}.weight", t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    modelfile = os.path.join(os.path.dirname(os.path.abspath(path)), "Modelfile")
    with open(modelfile, "w") as f:
        f.write(f"FROM ./{os.path.basename(path)}\n"
                'TEMPLATE """{{ if .Prompt }}<|user|>{{ .Prompt }}<|assistant|>{{ end }}{{ .Response }}"""\n'
                'PARAMETER stop "<|endoftext|>"\nPARAMETER stop "<|user|>"\nPARAMETER temperature 0.7\n'
                f"PARAMETER num_ctx {c.block_size}\n")
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--format", choices=["hf", "gguf"], required=True)
    p.add_argument("--out", required=True, help="directory (hf) or .gguf file")
    args = p.parse_args(argv)
    model, tokenizer, _ = load_checkpoint(args.ckpt, "cpu")
    check_exportable(model.config)
    if args.format == "hf":
        export_hf(model, tokenizer, args.out)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        export_gguf(model, tokenizer, args.out)
    print(f"exported {args.ckpt} -> {args.out}")


if __name__ == "__main__":
    main()
