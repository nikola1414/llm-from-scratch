# 03 — Gradient descent, the training loop and optimizers

## Gradient descent

Training minimises a loss `L(θ)`. The gradient `∇L` points in the direction of steepest
increase, so each step moves the parameters a little the other way:

```
θ ← θ − η · ∇L(θ)          η = learning rate
```

Too large an `η` overshoots/diverges; too small learns slowly. **Stochastic** gradient
descent estimates `∇L` from a random mini-batch rather than the whole dataset.

## The loop

```python
for step in range(max_iters):
    if step % eval_interval == 0:
        print(estimate_loss())            # averaged, no_grad, model.eval()
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)          # forward
    optimizer.zero_grad(set_to_none=True) # gradients accumulate by default -> clear them
    loss.backward()                       # back-propagation (chain rule via autograd)
    optimizer.step()                      # update θ
```

`set_to_none=True` frees the gradient tensors instead of filling them with zeros — slightly
faster and less memory.

## Optimizers overview

| Optimizer | Update idea | Typical application |
|---|---|---|
| **SGD** | `θ -= η g` | convex problems, sometimes large-batch vision |
| **SGD + momentum** | accumulate velocity `v = βv + g`, step with `v` | CNNs (ResNet training) |
| **Adagrad** | divide by √(sum of squared grads) → rare features get bigger steps | sparse features / NLP bag-of-words |
| **RMSprop** | like Adagrad but with an exponential moving average | RNNs, reinforcement learning |
| **Adam** | momentum (1st moment) + RMSprop (2nd moment) + bias correction | default for most deep nets |
| **AdamW** | Adam with *decoupled* weight decay `θ -= η λ θ` | transformers / LLMs (GPT, Llama) |

We use `torch.optim.AdamW(model.parameters(), lr=3e-4)`.

## Train vs eval mode

* `model.train()` — dropout active.
* `model.eval()` — dropout disabled (and BatchNorm uses running stats).
* `@torch.no_grad()` — do not record operations for autograd during evaluation/generation.

## Loss reporting

Cross-entropy of a uniform guess over `V` classes is `ln V` (≈ 4.4 for 80 characters), so
the first reported loss should be close to that. Report the *average* over many batches
for both train and val; if train keeps falling while val rises, the model is overfitting.
