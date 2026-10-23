import pytest
import torch
from visual_inception_core import forward_noise, ddim_step


@pytest.mark.parametrize("prediction_type", ["epsilon", "v_prediction", "sample"])
@pytest.mark.parametrize("alpha_prev", [.6, 1.])
def test_ddim_recovers_analytic_clean_and_previous_sample(prediction_type, alpha_prev):
    clean = torch.tensor([.4, -.2, 1.1], dtype=torch.float64)
    eps = torch.tensor([.2, .7, -.8], dtype=clean.dtype)
    alpha = clean.new_tensor(.25)
    previous_alpha = clean.new_tensor(alpha_prev)
    sample = forward_noise(clean, eps, alpha)
    model_output = {"epsilon": eps, "sample": clean,
                    "v_prediction": alpha.sqrt() * eps - (1 - alpha).sqrt() * clean}[prediction_type]
    previous, recovered = ddim_step(sample, model_output, alpha, previous_alpha, prediction_type)
    torch.testing.assert_close(recovered, clean)
    torch.testing.assert_close(previous, forward_noise(clean, eps, previous_alpha))


def test_forward_noise_and_ddim_gradient():
    sample = torch.tensor([.3, .7], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor([.1, -.2], dtype=sample.dtype)
    assert torch.autograd.gradcheck(lambda x: ddim_step(x, eps, .3, .8)[0], (sample,))


@pytest.mark.parametrize("alpha,alpha_prev", [(0, .5), (-.1, .5), (1, 1), (.8, .5), (.5, 1.1), (.5, float("nan"))])
def test_invalid_diffusion_schedule(alpha, alpha_prev):
    with pytest.raises(ValueError):
        ddim_step(torch.ones(2), torch.zeros(2), alpha, alpha_prev)


def test_unknown_prediction_type_rejected():
    with pytest.raises(ValueError):
        ddim_step(torch.ones(2), torch.zeros(2), .2, .8, "unknown")


def test_no_implicit_shape_broadcasting():
    with pytest.raises(ValueError):
        forward_noise(torch.ones(2), torch.ones(1), .5)


def test_forward_noise_identity_at_alpha_one():
    value = torch.tensor([2., -3.])
    torch.testing.assert_close(forward_noise(value, torch.randn_like(value), 1), value)
