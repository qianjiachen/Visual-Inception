from dataclasses import replace
import pytest
import torch
from visual_inception_core import PGDConfig, optimize_image, total_loss


def fixture(dtype=torch.float32):
    image = torch.full((1, 3, 2, 2), .5, dtype=dtype)

    def encoder(x):
        means = x.mean(dim=(2, 3))
        return torch.stack((means[:, 0] + .2, means[:, 1] + .1), dim=1)

    target = torch.tensor([[0., 1.]], dtype=dtype)
    queries = torch.tensor([[0., 1.], [.1, .9], [.2, .8]], dtype=dtype)
    memories = torch.tensor([[1., 0.], [.8, .2]], dtype=dtype)
    return image, encoder, target, queries, memories


def test_one_step_matches_manual_sign_descent_and_keeps_final_update():
    image, encoder, target, queries, memories = fixture(torch.float64)
    config = PGDConfig(steps=1, batch_size=3, lambda_perceptual=0)
    variable = image.clone().requires_grad_(True)
    loss, _ = total_loss(encoder(variable), target, queries, memories, lambda_perceptual=0)
    gradient = torch.autograd.grad(loss, variable)[0]
    proposed = variable.detach() - config.alpha * gradient.sign()
    expected = (image + (proposed - image).clamp(-config.epsilon, config.epsilon)).clamp(0, 1)
    result = optimize_image(image, encoder, target, queries, memories, config)
    torch.testing.assert_close(result.image, expected)
    assert result.best_loss < loss.item()
    assert result.best_loss == result.history[-1]["validation_loss"]


@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_100_steps_respect_pixel_budget_and_best_checkpoint(seed, dtype):
    image, encoder, target, queries, memories = fixture(dtype)
    original = image.clone()
    query_copy, memory_copy = queries.clone(), memories.clone()
    config = PGDConfig(seed=seed, batch_size=2, lambda_perceptual=0)
    result = optimize_image(image, encoder, target, queries, memories, config)
    assert len(result.history) == 101
    assert result.perturbation_linf <= config.epsilon + 1e-6
    assert result.image.min() >= 0 and result.image.max() <= 1
    assert result.best_loss == min(row["validation_loss"] for row in result.history)
    final_loss, _ = total_loss(encoder(result.image), target, queries, memories, lambda_perceptual=0)
    assert final_loss.item() == pytest.approx(result.best_loss)
    assert not result.image.requires_grad
    torch.testing.assert_close(image, original)
    torch.testing.assert_close(queries, query_copy)
    torch.testing.assert_close(memories, memory_copy)


@pytest.mark.parametrize("pixel", [0., 1.])
def test_projection_at_image_boundaries(pixel):
    image, encoder, target, queries, memories = fixture()
    image.fill_(pixel)
    result = optimize_image(image, encoder, target, queries, memories,
                             PGDConfig(steps=4, lambda_perceptual=0))
    assert result.image.min() >= 0 and result.image.max() <= 1
    assert result.perturbation_linf <= 8 / 255 + 1e-6


def test_fixed_seed_reproducibility():
    args = fixture()
    config = PGDConfig(steps=8, batch_size=1, lambda_perceptual=0)
    first, second = (optimize_image(*args, config) for _ in range(2))
    assert first.history == second.history
    torch.testing.assert_close(first.image, second.image, rtol=0, atol=0)


def test_best_checkpoint_can_remain_initial_image():
    image, encoder, _, queries, memories = fixture()

    def flat_encoder(x):
        return torch.tensor([[1., 1.]]) + x.mean() * 0

    target = torch.tensor([[1., 1.]])
    result = optimize_image(image, flat_encoder, target, queries, memories,
                             PGDConfig(steps=3, lambda_perceptual=0))
    torch.testing.assert_close(result.image, image)
    assert result.perturbation_linf == 0


def test_default_perceptual_term_requires_explicit_model():
    with pytest.raises(ValueError, match="LPIPS"):
        optimize_image(*fixture())


@pytest.mark.parametrize("kind", ["detached", "unrelated_leaf"])
def test_disconnected_image_encoder_is_rejected(kind):
    image, encoder, target, queries, memories = fixture()

    def broken(x):
        value = encoder(x).detach()
        return value.requires_grad_(True) if kind == "unrelated_leaf" else value

    with pytest.raises(ValueError, match="gradient"):
        optimize_image(image, broken, target, queries, memories,
                       PGDConfig(steps=1, lambda_perceptual=0))


@pytest.mark.parametrize("settings", [
    {"epsilon": 0}, {"alpha": 1}, {"steps": 0}, {"steps": 1.5},
    {"batch_size": 0}, {"temperature": float("nan")},
    {"lambda_semantic": -1}, {"lambda_perceptual": -1}, {"seed": True},
])
def test_invalid_configuration(settings):
    with pytest.raises(ValueError):
        replace(PGDConfig(), **settings)
