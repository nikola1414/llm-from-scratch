import math

import pytest
import torch

from llm.model import GPT, GPTConfig, apply_rope, rope_cache

VARIANTS = {
    "modern": dict(),
    "gqa": dict(n_kv_head=1),
    "v1-like": dict(pos_emb="learned", norm="layer", mlp="relu", tie_weights=False, bias=True),
    "gelu": dict(mlp="gelu"),
}


def make(**kw):
    torch.manual_seed(0)
    cfg = dict(vocab_size=50, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    cfg.update(kw)
    return GPT(GPTConfig(**cfg)).eval()


@pytest.mark.parametrize("variant", VARIANTS)
def test_shapes_and_initial_loss(variant):
    model = make(**VARIANTS[variant])
    x = torch.randint(0, 50, (3, 16))
    y = torch.randint(0, 50, (3, 16))   # random targets (with tied weights, y == x would score above chance)
    logits, loss = model(x, y)
    assert logits.shape == (3, 16, 50)
    # at init the model is close to uniform: loss ~ ln(vocab)
    assert abs(loss.item() - math.log(50)) < 0.5
    logits, loss = model(x)
    assert logits.shape == (3, 1, 50) and loss is None


@pytest.mark.parametrize("variant", VARIANTS)
def test_causality(variant):
    """Changing a future token must not change the logits of earlier positions."""
    model = make(**VARIANTS[variant])
    x = torch.randint(0, 50, (1, 16))
    y = x.clone()
    y[0, 10] = (y[0, 10] + 1) % 50
    a, _ = model(x, x)
    b, _ = model(y, y)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:])


@pytest.mark.parametrize("variant", VARIANTS)
def test_kv_cache_matches_full_forward(variant):
    model = make(**VARIANTS[variant])
    x = torch.randint(0, 50, (2, 16))
    full, _ = model(x, x)
    caches = [{} for _ in model.blocks]
    model(x[:, :6], kv_caches=caches)
    for t in range(6, 16):
        step, _ = model(x[:, t:t + 1], kv_caches=caches, start_pos=t)
        assert torch.allclose(step[:, -1], full[:, t], atol=1e-4)


def test_ignore_index():
    model = make()
    x = torch.randint(0, 50, (2, 16))
    y = x.clone()
    y[:, :8] = -1
    _, masked = model(x, y)
    logits, _ = model(x, x)
    manual = torch.nn.functional.cross_entropy(logits[:, 8:].reshape(-1, 50), x[:, 8:].reshape(-1))
    assert torch.allclose(masked, manual, atol=1e-5)


def test_weight_tying():
    m = make()
    assert m.lm_head.weight.data_ptr() == m.tok_emb.weight.data_ptr()
    m2 = make(tie_weights=False)
    assert m2.lm_head.weight.data_ptr() != m2.tok_emb.weight.data_ptr()


def test_rope_preserves_norm_and_is_relative():
    cos, sin = rope_cache(32, 8, 10000.0)
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 1, 8)
    rot = lambda v, p: apply_rope(v, cos[p:p + 1], sin[p:p + 1])
    assert torch.allclose(rot(q, 5).norm(), q.norm(), atol=1e-5)
    # q·k depends only on the distance between positions
    d1 = (rot(q, 3) * rot(k, 1)).sum()
    d2 = (rot(q, 12) * rot(k, 10)).sum()
    assert torch.allclose(d1, d2, atol=1e-4)


def test_generation_beyond_block_size_and_sampling_options():
    model = make()
    x = torch.zeros(1, 1, dtype=torch.long)
    out = model.generate_all(x, 40, top_k=5, top_p=0.9, repetition_penalty=1.3)
    assert out.shape == (1, 41)
    greedy_cached = model.generate_all(x, 10, temperature=0)
    greedy_nocache = model.generate_all(x, 10, temperature=0, use_cache=False)
    assert torch.equal(greedy_cached, greedy_nocache)


def test_stop_ids():
    model = make()
    x = torch.zeros(1, 1, dtype=torch.long)
    first = model.generate_all(x, 1, temperature=0)[0, -1].item()
    out = model.generate_all(x, 20, temperature=0, stop_ids={first})
    assert out.shape[1] == 2


def test_overfits_single_batch():
    model = make(dropout=0.0).train()
    x = torch.randint(0, 50, (4, 16))
    opt = model.configure_optimizer(0.0, 3e-3)
    for _ in range(150):
        _, loss = model(x[:, :-1], x[:, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.5
