"""Retrieval, DPO, chat evaluation, generation options and export."""
import json

import pytest
import torch

from llm import dpo, export, generate
from llm.checkpoint import load_checkpoint, save_checkpoint
from llm.model import GPT, GPTConfig
from llm.retrieval import BM25, chunk_text, rag_prompt
from llm.tokenizer import BPETokenizer

BOOK = ("Dorothy lived in Kansas with Uncle Henry and Aunt Em.\n\n"
        "The Scarecrow wanted brains, so he asked the Wizard for some.\n\n"
        "The Gargoyles were made of wood and feared every noise.\n\n"
        "Ozma lived in a palace in the Emerald City and ruled the Land of Oz.\n\n") * 5


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    torch.manual_seed(0)
    tok = BPETokenizer.train(BOOK, 330, pattern="gpt2")
    model = GPT(GPTConfig(vocab_size=tok.vocab_size, block_size=96, n_layer=2, n_head=2, n_embd=32, dropout=0.0))
    path = tmp_path_factory.mktemp("m") / "m.pt"
    save_checkpoint(path, model, tok, iter=0, args={})
    return path


def test_chunking_and_bm25():
    chunks = chunk_text(BOOK, max_chars=120)
    assert all(len(c) <= 120 for c in chunks)
    index = BM25(chunks)
    top, score, _ = index.search("What did the Gargoyles fear?", 1)[0]
    assert "Gargoyles" in top and score > 0
    assert index.search("xylophone", 3) == []
    assert rag_prompt("Q?", ["a", "b"]) == "Context: a\n\nb\n\nQuestion: Q?"


def test_bm25_save_load(tmp_path):
    index = BM25(chunk_text(BOOK, 120), ["book"] * 20)
    index.save(tmp_path / "i.json")
    loaded = BM25.load(tmp_path / "i.json")
    assert loaded.search("Ozma palace", 1)[0][0] == index.search("Ozma palace", 1)[0][0]


def test_dpo_improves_preference_accuracy(ckpt, tmp_path):
    pairs = [{"prompt": f"Who lives in the palace {i}?", "chosen": "Ozma lives in the palace.",
              "rejected": "The Gargoyles feared noise."} for i in range(16)]
    data = tmp_path / "pairs.jsonl"
    data.write_text("\n".join(json.dumps(p) for p in pairs))
    result = dpo.main(["--ckpt", str(ckpt), "--data", str(data), "--val_data", str(data), "--out_dir",
                       str(tmp_path / "dpo"), "--epochs", "3", "--batch_size", "8", "--lr", "3e-3"])
    assert result["val_accuracy"] == 1.0
    load_checkpoint(tmp_path / "dpo" / "best.pt")


def test_generate_options(ckpt, tmp_path, capsys):
    index = BM25(chunk_text(BOOK, 120))
    index.save(tmp_path / "idx.json")
    generate.main(["--ckpt", str(ckpt), "--prompt", "Dorothy", "--num_samples", "3", "--max_new_tokens", "5"])
    assert capsys.readouterr().out.count("--- sample") == 3
    generate.main(["--ckpt", str(ckpt), "--chat", "--rag", "--index", str(tmp_path / "idx.json"),
                   "--prompt", "Who is Ozma?", "--max_new_tokens", "5"])
    generate.main(["--ckpt", str(ckpt), "--prompt", "Oz", "--int8", "--context", "192", "--max_new_tokens", "5"])


def test_int8_model_is_close(ckpt):
    model, tok, _ = load_checkpoint(ckpt)
    model.eval()
    q = generate.quantize_int8(load_checkpoint(ckpt)[0].eval())
    x = torch.tensor([tok.encode(BOOK[:200])[:40]])
    a, _ = model(x, x)
    b, _ = q(x, x)
    assert (a - b).abs().max() < 0.1 * a.abs().max()


def test_export_hf_matches(ckpt, tmp_path):
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("tokenizers")
    export.main(["--ckpt", str(ckpt), "--format", "hf", "--out", str(tmp_path / "hf")])
    model, tok, _ = load_checkpoint(ckpt)
    model.eval()
    hf = transformers.AutoModelForCausalLM.from_pretrained(tmp_path / "hf").eval()
    ht = transformers.PreTrainedTokenizerFast(tokenizer_file=str(tmp_path / "hf" / "tokenizer.json"))
    s = "Dorothy said: \"Héllo 42!\"<|endoftext|>Ozma"
    assert ht.encode(s) == tok.encode(s)
    x = torch.tensor([tok.encode(BOOK)[:60]])
    with torch.no_grad():
        assert torch.allclose(model(x, x)[0], hf(x).logits, atol=1e-4)


def test_export_gguf_writes_file(ckpt, tmp_path):
    gguf = pytest.importorskip("gguf")
    out = tmp_path / "m.gguf"
    export.main(["--ckpt", str(ckpt), "--format", "gguf", "--out", str(out)])
    reader = gguf.GGUFReader(str(out))
    names = {t.name for t in reader.tensors}
    assert "token_embd.weight" in names and "blk.1.ffn_down.weight" in names
    assert (tmp_path / "Modelfile").exists()


def test_export_rejects_unsupported():
    with pytest.raises(ValueError):
        export.check_exportable(GPTConfig(value_residual=True))


def test_permute_is_a_permutation():
    w = torch.arange(8 * 3).float().view(8, 3)
    p = export.permute_for_llama_cpp(w, 2)
    assert sorted(p.flatten().tolist()) == sorted(w.flatten().tolist())
    assert not torch.equal(p, w)
