"""Offline API demonstration with synthetic tensors, not a model experiment."""
import json
import torch
from visual_inception_core import (
    PGDConfig, optimize_image, reasoning_divergence, decide_memory,
    forward_noise, ddim_step,
)


def main():
    torch.set_num_threads(1)
    image = torch.full((1, 3, 4, 4), .5)

    def toy_encoder(x):
        means = x.mean(dim=(2, 3))
        return torch.stack((means[:, 0] + .2, means[:, 1] + .1), dim=1)

    target = torch.tensor([[0., 1.]])
    queries = torch.tensor([[0., 1.], [.1, .9]])
    memories = torch.tensor([[1., 0.], [.8, .2]])
    # Deliberately reduced objective: no LPIPS weights are supplied in this demo.
    result = optimize_image(image, toy_encoder, target, queries, memories,
                            PGDConfig(lambda_perceptual=0))
    rds = reasoning_divergence(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]))
    decision = decide_memory(rds, [3, 4, 5])
    clean, noise = torch.tensor([.4]), torch.tensor([.2])
    sample = forward_noise(clean, noise, .25)
    _, recovered = ddim_step(sample, noise, .25, 1.)
    print(json.dumps({
        "scope": "synthetic tensor demonstration; not paper results",
        "pgd_steps": len(result.history) - 1,
        "initial_loss": result.history[0]["validation_loss"],
        "best_loss": result.best_loss,
        "linf": result.perturbation_linf,
        "rds": rds,
        "flagged": decision.flagged,
        "ddim_recovered_clean": recovered.item(),
    }, indent=2))


if __name__ == "__main__":
    main()
