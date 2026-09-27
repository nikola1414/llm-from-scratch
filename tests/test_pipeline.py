"""End-to-end: prepare -> train -> resume -> evaluate -> finetune -> generate."""
import json

import torch

from llm import evaluate, finetune, generate, prepare, train
from llm.checkpoint import load_checkpoint

TEXT = ("Dorothy met the Wizard in the glass city. Zeb and Jim followed them. "
        "Eureka the kitten watched the nine tiny piglets.\n") * 60


def run(module, args, monkeypatch):
    monkeypatch.setattr("sys.argv", ["x"] + args)
    return module.main()


def test_end_to_end(tmp_path, monkeypatch, capsys):
    src = tmp_path / "book.txt"
    src.write_text(TEXT)
    data = tmp_path / "data"
    run(prepare, ["--input", str(src), "--out_dir", str(data), "--vocab_size", "300"], monkeypatch)
    meta = json.loads((data / "meta.json").read_text())
    assert meta["train_tokens"] > 0 and meta["val_chars"] > 0

    out = tmp_path / "run"
    common = ["--data_dir", str(data), "--out_dir", str(out), "--n_layer", "2", "--n_head", "2",
              "--n_embd", "32", "--block_size", "32", "--batch_size", "4", "--eval_interval", "20",
              "--eval_iters", "2", "--warmup_iters", "5"]
    result = train.main(common + ["--max_iters", "40"])
    assert result["val_bpc"] > 0
    log = (out / "log.csv").read_text().splitlines()
    assert log[0].startswith("iter") and len(log) >= 3

    # resume keeps the architecture and continues the iteration count
    train.main(["--out_dir", str(out), "--resume", "--max_iters", "50", "--eval_interval", "20", "--eval_iters", "2"])
    _, _, ckpt = load_checkpoint(out / "last.pt")
    assert ckpt["iter"] == 49

    # modern recipe: Muon + EMA + architecture options
    result2 = train.main(["--data_dir", str(data), "--out_dir", str(tmp_path / "run2"), "--n_layer", "2",
                          "--n_head", "2", "--n_embd", "32", "--block_size", "32", "--batch_size", "4",
                          "--eval_interval", "20", "--eval_iters", "2", "--warmup_iters", "5", "--max_iters", "40",
                          "--optimizer", "muon", "--ema", "0.9", "--qk_norm", "--logit_softcap", "30",
                          "--value_residual", "--unet_skips", "--n_experts", "2", "--moe_top_k", "1"])
    assert result2["val_bpc"] > 0
    _, _, ck2 = load_checkpoint(tmp_path / "run2" / "last.pt")
    assert "ema_model" in ck2 and len(ck2["optimizer"]) == 2
    train.main(["--out_dir", str(tmp_path / "run2"), "--resume", "--max_iters", "45", "--eval_interval", "20",
                "--eval_iters", "2"])

    res = evaluate.main(["--ckpt", str(out / "best.pt")])
    assert res["perplexity"] > 1

    sft = tmp_path / "sft.jsonl"
    sft.write_text("\n".join(json.dumps({"prompt": f"Who is {n}?", "response": f"{n} is a friend."})
                             for n in ["Zeb", "Jim", "Eureka", "Dorothy"] * 4))
    finetune.main(["--ckpt", str(out / "best.pt"), "--data", str(sft), "--out_dir", str(tmp_path / "sft"),
                   "--epochs", "2", "--batch_size", "4"])
    assert (tmp_path / "sft" / "best.pt").exists()

    generate.main(["--ckpt", str(tmp_path / "sft" / "best.pt"), "--chat", "--prompt", "Who is Zeb?",
                   "--max_new_tokens", "10", "--seed", "0"])
    generate.main(["--ckpt", str(out / "best.pt"), "--prompt", "Dorothy", "--max_new_tokens", "10"])
    assert "Dorothy" in capsys.readouterr().out


def test_checkpoint_is_safe_to_load(tmp_path):
    """Checkpoints must load with weights_only=True (no pickled code)."""
    from llm.checkpoint import save_checkpoint
    from llm.model import GPT, GPTConfig
    from llm.tokenizer import CharTokenizer
    model = GPT(GPTConfig(vocab_size=10, block_size=8, n_layer=1, n_head=2, n_embd=16))
    save_checkpoint(tmp_path / "c.pt", model, CharTokenizer("abcdefg"), iter=0)
    torch.load(tmp_path / "c.pt", weights_only=True)


def test_bpe_dropout_prepare(tmp_path, monkeypatch):
    src = tmp_path / "book.txt"
    src.write_text(TEXT)
    run(prepare, ["--input", str(src), "--val_input", str(src), "--out_dir", str(tmp_path / "d"),
                  "--vocab_size", "300", "--bpe_dropout", "0.2", "--dropout_copies", "3"], monkeypatch)
    meta = json.loads((tmp_path / "d" / "meta.json").read_text())
    assert meta["train_copies"] == 3
    assert meta["train_chars"] == 3 * meta["val_chars"]
    assert meta["train_tokens"] > 3 * meta["val_tokens"]      # dropout -> more tokens per copy


def test_ddp_two_processes(tmp_path, monkeypatch):
    """DistributedDataParallel on CPU (gloo) with two processes."""
    import subprocess
    import sys
    src = tmp_path / "book.txt"
    src.write_text(TEXT)
    run(prepare, ["--input", str(src), "--out_dir", str(tmp_path / "d"), "--vocab_size", "300"], monkeypatch)
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node", "2", "-m", "llm.train",
           "--data_dir", str(tmp_path / "d"), "--out_dir", str(tmp_path / "ddp"), "--n_layer", "1", "--n_head", "2",
           "--n_embd", "16", "--block_size", "16", "--batch_size", "2", "--max_iters", "6", "--eval_interval", "5",
           "--eval_iters", "1", "--warmup_iters", "1", "--grad_accum", "2"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env={**__import__("os").environ,
                                                                                  "OMP_NUM_THREADS": "1"})
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "x2" in proc.stdout and (tmp_path / "ddp" / "result.json").exists()
