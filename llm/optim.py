"""Muon optimizer (Keller Jordan et al., 2024) — MomentUm Orthogonalized by Newton-Schulz.

For each 2-D weight matrix, Muon takes the (Nesterov) momentum of the gradient and
replaces it by the nearest semi-orthogonal matrix (all singular values ≈ 1), computed
with a few Newton-Schulz iterations. The update therefore changes the weight equally
in every direction instead of being dominated by a few large singular directions. It is
used for the hidden matrices of the transformer; embeddings, the output layer and
1-D parameters (norm gains) stay on AdamW.
"""
import torch


@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5):
    """Approximately orthogonalise G (UΣVᵀ -> ≈UVᵀ) with a quintic Newton-Schulz iteration
    whose coefficients maximise the slope at zero (so tiny singular values grow fast)."""
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() if G.is_cuda else G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    X = X / (X.norm() + 1e-7)                     # spectral norm <= 1
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(group["momentum"]).add_(g)
                g = g.add(buf, alpha=group["momentum"]) if group["nesterov"] else buf
                update = zeropower_via_newtonschulz5(g, group["ns_steps"])
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                # scale so the update RMS is comparable for tall and wide matrices
                scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                p.add_(update, alpha=-group["lr"] * scale)
        return loss
