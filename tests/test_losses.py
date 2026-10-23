import math
import pytest
import torch
from visual_inception_core import retrieval_loss, semantic_loss, perceptual_loss, total_loss


def vectors():
    return (torch.tensor([[1., 0.]], dtype=torch.float64),
            torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64),
            torch.tensor([[0., 1.], [-1., 0.]], dtype=torch.float64))


def test_retrieval_matches_explicit_scalar_partition():
    adv, queries, memories = vectors()
    temperature = .5
    expected = (math.log(math.exp(2) + 1 + math.exp(-2)) - 2
                + math.log(1 + math.exp(2) + 1)) / 2
    assert retrieval_loss(adv, queries, memories, temperature).item() == pytest.approx(expected)


def test_retrieval_gradient_matches_finite_differences():
    adv = torch.tensor([[.8, .3]], dtype=torch.float64, requires_grad=True)
    _, queries, memories = vectors()
    assert torch.autograd.gradcheck(lambda x: retrieval_loss(x, queries, memories), (adv,))


def test_retrieval_normalizes_and_is_stable_at_default_temperature():
    adv, queries, memories = vectors()
    a = retrieval_loss(adv, queries, memories)
    b = retrieval_loss(adv * 3, queries * 9, memories * 2)
    torch.testing.assert_close(a, b)
    assert torch.isfinite(a) and a >= 0


@pytest.mark.parametrize("target,expected", [([1., 0.], 0), ([0., 1.], 1), ([-1., 0.], 2)])
def test_semantic_angles(target, expected):
    adv, _, _ = vectors()
    assert semantic_loss(adv, torch.tensor([target], dtype=adv.dtype)).item() == expected


def test_total_weights_and_lpips_input_contract():
    adv, queries, memories = vectors()
    original = torch.zeros(1, 3, 2, 2, dtype=adv.dtype)
    perturbed = torch.full_like(original, .25).requires_grad_(True)
    calls = []

    def synthetic_distance(a, b):
        # Tests scaling and gradient plumbing only. This is not an LPIPS model.
        calls.append((a.detach().clone(), b.detach().clone()))
        return (a - b).square().mean().reshape(1, 1, 1, 1)

    total, parts = total_loss(adv, queries[1:2], queries, memories, original, perturbed,
                              synthetic_distance, lambda_semantic=2, lambda_perceptual=.3)
    assert parts["perceptual"] == pytest.approx(.25)
    assert parts["total"] == pytest.approx(parts["retrieval"] + 2 + .3 * .25)
    assert torch.autograd.grad(total, perturbed)[0].abs().sum() > 0
    assert torch.all(calls[0][0] == -1) and torch.all(calls[0][1] == -.5)


def test_missing_lpips_is_not_silently_substituted():
    adv, queries, memories = vectors()
    with pytest.raises(ValueError, match="real LPIPS"):
        total_loss(adv, adv, queries, memories)
    total, parts = total_loss(adv, adv, queries, memories, lambda_perceptual=0)
    assert parts["perceptual"] == 0 and torch.isfinite(total)


@pytest.mark.parametrize("kind", ["empty", "zero", "nan", "wrong_dim", "dtype", "temperature"])
def test_retrieval_rejects_invalid_inputs(kind):
    adv, queries, memories = vectors()
    temperature = .07
    if kind == "empty":
        memories = memories[:0]
    elif kind == "zero":
        queries = torch.zeros_like(queries)
    elif kind == "nan":
        adv[0, 0] = float("nan")
    elif kind == "wrong_dim":
        memories = torch.ones(2, 3, dtype=adv.dtype)
    elif kind == "dtype":
        memories = memories.float()
    else:
        temperature = 0
    with pytest.raises(ValueError):
        retrieval_loss(adv, queries, memories, temperature)


def test_detached_lpips_is_rejected():
    original = torch.zeros(1, 3, 2, 2)
    perturbed = original.clone().requires_grad_(True)
    with pytest.raises(ValueError, match="gradient"):
        perceptual_loss(original, perturbed, lambda a, b: (a - b).square().mean().detach())
