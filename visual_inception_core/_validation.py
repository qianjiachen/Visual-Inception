"""Input checks shared by the public numerical API."""
import math
from numbers import Real
import torch


def tensor(value, name):
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{name} must be a torch.Tensor")
    if value.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must use float32 or float64")
    if value.numel() == 0 or not torch.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return value


def compatible(reference, *values):
    if any(x.dtype != reference.dtype or x.device != reference.device for x in values):
        raise ValueError("Tensors must share dtype and device")


def embeddings(value, name):
    tensor(value, name)
    if value.ndim != 2 or value.shape[1] == 0:
        raise ValueError(f"{name} must have shape [N, D]")
    norms = value.norm(dim=-1)
    if not torch.isfinite(norms).all() or (norms <= 1e-8).any():
        raise ValueError(f"{name} must contain finite, nonzero embedding vectors")
    return value


def image(value, name):
    tensor(value, name)
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3:
        raise ValueError(f"{name} must have shape [1, 3, H, W]")
    if value.min() < 0 or value.max() > 1:
        raise ValueError(f"{name} pixels must be in [0, 1]")
    return value


def real(value, name, minimum=None, maximum=None, strict_min=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if minimum is not None and (value <= minimum if strict_min else value < minimum):
        raise ValueError(f"Invalid lower bound for {name}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Invalid upper bound for {name}")
    return float(value)


def integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value
