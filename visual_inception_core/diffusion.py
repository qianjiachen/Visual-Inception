"""Forward noising and deterministic DDIM (eta=0) kernels, not full System 1."""
import torch
from . import _validation as check


def _alpha(value, reference, name):
    if not isinstance(value, torch.Tensor):
        check.real(value, name, 0, 1, strict_min=True)
        value = reference.new_tensor(value)
    check.tensor(value, name)
    check.compatible(reference, value)
    if value.ndim != 0 or not 0 < value.item() <= 1:
        raise ValueError(f"{name} must be a scalar cumulative alpha in (0,1]")
    return value


def forward_noise(clean, noise, alpha):
    """sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*epsilon."""
    check.tensor(clean, "clean")
    check.tensor(noise, "noise")
    check.compatible(clean, noise)
    if clean.shape != noise.shape:
        raise ValueError("clean and noise must have equal shapes")
    alpha = _alpha(alpha, clean, "alpha")
    return alpha.sqrt() * clean + (1 - alpha).sqrt() * noise


def ddim_step(sample, model_output, alpha, alpha_prev, prediction_type="epsilon"):
    """Return (previous_sample, predicted_clean), without clipping or noise.

    alpha is cumulative alpha_bar_t, and alpha_prev is alpha_bar_s for s < t.
    Require 0 < alpha < 1 and alpha <= alpha_prev <= 1. This step assumes the
    model output's prediction type is explicitly matched to its scheduler.
    """
    check.tensor(sample, "sample")
    check.tensor(model_output, "model_output")
    check.compatible(sample, model_output)
    if sample.shape != model_output.shape:
        raise ValueError("sample and model_output must have equal shapes")
    alpha = _alpha(alpha, sample, "alpha")
    alpha_prev = _alpha(alpha_prev, sample, "alpha_prev")
    if alpha >= 1 or alpha_prev < alpha:
        raise ValueError("Require alpha < 1 and alpha_prev >= alpha")
    if prediction_type == "epsilon":
        eps = model_output
        clean = (sample - (1 - alpha).sqrt() * eps) / alpha.sqrt()
    elif prediction_type == "v_prediction":
        clean = alpha.sqrt() * sample - (1 - alpha).sqrt() * model_output
        eps = alpha.sqrt() * model_output + (1 - alpha).sqrt() * sample
    elif prediction_type == "sample":
        clean = model_output
        eps = (sample - alpha.sqrt() * clean) / (1 - alpha).sqrt()
    else:
        raise ValueError("prediction_type must be epsilon, v_prediction, or sample")
    previous = alpha_prev.sqrt() * clean + (1 - alpha_prev).sqrt() * eps
    if not torch.isfinite(previous).all() or not torch.isfinite(clean).all():
        raise FloatingPointError("Nonfinite DDIM output")
    return previous, clean
