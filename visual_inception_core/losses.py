"""Single-encoder objective. See ALGORITHMS.md for the denominator convention."""
from collections.abc import Callable
import torch
import torch.nn.functional as F
from . import _validation as check


def retrieval_loss(adv, queries, memories, temperature=0.07):
    """Mean negative log probability, including adv itself in the denominator.

    adv: [1,D]; queries: [Q,D]; memories: [M,D], all in one embedding space.
    memories contains the existing surrogate items, excluding adv itself.
    """
    for name, value in (("adv", adv), ("queries", queries), ("memories", memories)):
        check.embeddings(value, name)
    check.compatible(adv, queries, memories)
    check.real(temperature, "temperature", 0, strict_min=True)
    if adv.shape[0] != 1 or len({x.shape[1] for x in (adv, queries, memories)}) != 1:
        raise ValueError("Require one adversarial image and matching embedding dimensions")
    adv, queries, memories = (F.normalize(x, dim=-1) for x in (adv, queries, memories))
    logits = torch.cat((adv @ queries.T, memories @ queries.T), dim=0) / temperature
    if not torch.isfinite(logits).all():
        raise FloatingPointError("Nonfinite retrieval logits")
    return -F.log_softmax(logits, dim=0)[0].mean()


def semantic_loss(adv, target):
    """1 - cosine for one image and one target embedding, both [1,D]."""
    check.embeddings(adv, "adv")
    check.embeddings(target, "target")
    check.compatible(adv, target)
    if adv.shape != target.shape or adv.shape[0] != 1:
        raise ValueError("adv and target must both have shape [1,D]")
    return 1 - F.cosine_similarity(adv, target, dim=-1).mean()


def perceptual_loss(original, perturbed, lpips_model: Callable):
    """Call the supplied LPIPS model with pixels mapped to [-1,1].

    The caller supplies a genuine LPIPS implementation in eval mode with frozen
    weights. This function never loads weights or replaces LPIPS with a proxy.
    """
    check.image(original, "original")
    check.image(perturbed, "perturbed")
    check.compatible(original, perturbed)
    if original.shape != perturbed.shape or not callable(lpips_model):
        raise ValueError("Provide equal image shapes and a callable LPIPS model")
    distance = lpips_model(original * 2 - 1, perturbed * 2 - 1)
    check.tensor(distance, "LPIPS output")
    check.compatible(original, distance)
    distance = distance.mean()
    if torch.is_grad_enabled() and perturbed.requires_grad:
        if not distance.requires_grad:
            raise ValueError("LPIPS must retain an input gradient")
        derivative = torch.autograd.grad(distance, perturbed, allow_unused=True, retain_graph=True)[0]
        if derivative is None or not torch.isfinite(derivative).all():
            raise ValueError("LPIPS must have a finite gradient connected to perturbed pixels")
    return distance


def total_loss(adv, target, queries, memories, original=None, perturbed=None,
               lpips_model=None, temperature=0.07, lambda_semantic=1.0,
               lambda_perceptual=0.1):
    """Return the differentiable objective and detached scalar components.

    lambda_perceptual=0 explicitly selects a reduced objective, not the paper's
    full perceptually regularized objective.
    """
    check.real(lambda_semantic, "lambda_semantic", 0)
    check.real(lambda_perceptual, "lambda_perceptual", 0)
    ret = retrieval_loss(adv, queries, memories, temperature)
    sem = semantic_loss(adv, target)
    per = adv.new_zeros(())
    if lambda_perceptual:
        if original is None or perturbed is None or lpips_model is None:
            raise ValueError("Enabled perceptual term requires images and a real LPIPS model")
        check.compatible(adv, original, perturbed)
        per = perceptual_loss(original, perturbed, lpips_model)
    total = ret + lambda_semantic * sem + lambda_perceptual * per
    if not torch.isfinite(total):
        raise FloatingPointError("Nonfinite objective")
    return total, {
        "total": total.detach().item(), "retrieval": ret.detach().item(),
        "semantic": sem.detach().item(), "perceptual": per.detach().item(),
    }
