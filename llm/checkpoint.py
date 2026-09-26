"""Safe checkpoints: tensors + plain Python data only, loadable with weights_only=True
(unlike pickling whole objects, loading cannot execute arbitrary code)."""
import torch

from .model import GPT, GPTConfig
from .tokenizer import Tokenizer


def save_checkpoint(path, model, tokenizer, optimizer=None, **extra):
    ckpt = {
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "tokenizer": tokenizer.to_dict(),
        **extra,
    }
    if optimizer is not None:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)


def load_checkpoint(path, device="cpu"):
    """Returns (model, tokenizer, checkpoint_dict)."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)
    return model, Tokenizer.from_dict(ckpt["tokenizer"]), ckpt
