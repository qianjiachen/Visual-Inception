"""Single-encoder, zero-start, sign-gradient descent with L-infinity projection."""
from dataclasses import dataclass
import random
from collections.abc import Callable
import torch
from . import _validation as check
from .losses import total_loss


@dataclass(frozen=True)
class PGDConfig:
    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 100
    batch_size: int = 16
    temperature: float = 0.07
    lambda_semantic: float = 1.0
    lambda_perceptual: float = 0.1
    seed: int = 42

    def __post_init__(self):
        check.real(self.epsilon, "epsilon", 0, 1, strict_min=True)
        check.real(self.alpha, "alpha", 0, self.epsilon, strict_min=True)
        check.real(self.temperature, "temperature", 0, strict_min=True)
        check.real(self.lambda_semantic, "lambda_semantic", 0)
        check.real(self.lambda_perceptual, "lambda_perceptual", 0)
        check.integer(self.steps, "steps")
        check.integer(self.batch_size, "batch_size")
        check.integer(self.seed, "seed", 0)


@dataclass(frozen=True)
class PGDResult:
    image: torch.Tensor
    best_loss: float
    perturbation_linf: float
    history: tuple[dict, ...]


def optimize_image(image, image_encoder: Callable, target, queries, memories,
                   config=None, lpips_model=None):
    """Optimize one [1,3,H,W] image using precomputed same-space embeddings.

    The image encoder must be deterministic, differentiable, and preserve tensor
    dtype/device. It includes the real model's differentiable preprocessing.
    This API does not construct query sources or implement an encoder ensemble.
    """
    config = PGDConfig() if config is None else config
    if not isinstance(config, PGDConfig) or not callable(image_encoder):
        raise ValueError("Provide PGDConfig and a callable image encoder")
    check.image(image, "image")
    for name, value in (("target", target), ("queries", queries), ("memories", memories)):
        check.embeddings(value, name)
    check.compatible(image, target, queries, memories)
    if target.shape[0] != 1 or len({x.shape[1] for x in (target, queries, memories)}) != 1:
        raise ValueError("Target [1,D], query [Q,D], memory [M,D] must share one space")
    if config.lambda_perceptual and not callable(lpips_model):
        raise ValueError("Default objective requires a supplied real LPIPS model")
    original = image.detach().clone()
    target, queries, memories = (x.detach() for x in (target, queries, memories))
    rng = random.Random(config.seed)

    def objective(candidate, batch):
        embedding = image_encoder(candidate)
        check.embeddings(embedding, "image embedding")
        if torch.is_grad_enabled() and candidate.requires_grad:
            if not embedding.requires_grad:
                raise ValueError("Image encoder must retain an input gradient")
            derivative = torch.autograd.grad(embedding.sum(), candidate,
                                             allow_unused=True, retain_graph=True)[0]
            if derivative is None or not torch.isfinite(derivative).all():
                raise ValueError("Image encoder must have a finite gradient connected to pixels")
        return total_loss(
            embedding, target, batch, memories, original, candidate, lpips_model,
            config.temperature, config.lambda_semantic, config.lambda_perceptual,
        )

    with torch.no_grad():
        initial, parts = objective(original, queries)
    best_loss = float(initial)
    best_image = original.clone()
    history = [dict(step=0, validation_loss=best_loss, **parts)]
    current = original.clone()
    for step in range(1, config.steps + 1):
        current = current.detach().requires_grad_(True)
        indices = rng.sample(range(len(queries)), min(config.batch_size, len(queries)))
        loss, _ = objective(current, queries[indices])
        if not loss.requires_grad:
            raise ValueError("The objective must retain a gradient to image pixels")
        gradient = torch.autograd.grad(loss, current, allow_unused=True)[0]
        if gradient is None:
            raise ValueError("The objective is disconnected from image pixels")
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("Nonfinite image gradient")
        with torch.no_grad():
            proposal = current - config.alpha * gradient.sign()
            delta = (proposal - original).clamp(-config.epsilon, config.epsilon)
            current = (original + delta).clamp(0, 1)
            candidate_loss, parts = objective(current, queries)
            value = float(candidate_loss)
            if value < best_loss:
                best_loss, best_image = value, current.clone()
        history.append(dict(step=step, batch_loss_before_step=float(loss.detach()),
                            validation_loss=value, **parts))
    return PGDResult(best_image.detach(), best_loss,
                     float((best_image - original).abs().max()), tuple(history))
