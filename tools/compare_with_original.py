"""Optional offline cross-check against a separately supplied original checkout.

Run from the core source root with:
  python -B -m tools.compare_with_original --source-root PATH --output PATH
This is synthetic numerical evidence, never a benchmark/model experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import socket
import sys
import torch
from visual_inception_core import (
    retrieval_loss, semantic_loss, perceptual_loss, PGDConfig, optimize_image,
    reasoning_divergence, decide_memory, ddim_step,
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manuscript", type=Path)
    parser.add_argument("--paper-pdf", type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    paths = ["src/attack/loss.py", "src/attack/visual_inception.py",
             "src/attack/samplers.py", "src/defense/system1.py",
             "src/defense/system2.py", "LICENSE"]
    before = {path: sha256(root / path) for path in paths}

    def blocked(*unused_args, **unused_kwargs):
        raise AssertionError("Network access is forbidden in synthetic comparison")

    socket.create_connection = blocked
    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    sys.path.insert(0, str(root))
    from src.attack.loss import LossCalculator
    from src.attack.visual_inception import VisualInceptionAttack
    from src.attack.samplers import FixedQuerySampler
    from src.defense.system1 import System1Sanitizer
    from src.defense.system2 import System2Verifier

    torch.set_num_threads(1)
    original_loss = LossCalculator(lambda_perceptual=0)
    errors = {"retrieval_value": 0., "retrieval_gradient": 0., "semantic_value": 0.,
              "ddim_previous": 0., "ddim_clean": 0.}
    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        adv = torch.randn(1, 5, generator=generator, dtype=torch.float64, requires_grad=True)
        target = torch.randn(1, 5, generator=generator, dtype=torch.float64)
        queries = torch.randn(7, 5, generator=generator, dtype=torch.float64)
        memories = torch.randn(8, 5, generator=generator, dtype=torch.float64)
        old = original_loss.compute_retrieval_loss(adv, queries, memories)
        new = retrieval_loss(adv, queries, memories)
        torch.testing.assert_close(new, old, rtol=1e-12, atol=1e-12)
        old_grad = torch.autograd.grad(old, adv)[0]
        new_grad = torch.autograd.grad(new, adv)[0]
        torch.testing.assert_close(new_grad, old_grad, rtol=1e-12, atol=1e-12)
        old_sem = original_loss.compute_semantic_loss(adv, target)
        new_sem = semantic_loss(adv, target)
        torch.testing.assert_close(old_sem, new_sem, rtol=1e-12, atol=1e-12)
        for name, error in (
            ("retrieval_value", (new - old).abs().item()),
            ("retrieval_gradient", (new_grad - old_grad).abs().max().item()),
            ("semantic_value", (new_sem - old_sem).abs().item()),
        ):
            errors[name] = max(errors[name], error)

        sample = torch.randn(2, 3, generator=generator, dtype=torch.float64)
        output = torch.randn(2, 3, generator=generator, dtype=torch.float64)
        alpha, alpha_prev = sample.new_tensor(.25), sample.new_tensor(.8)
        for mode in ("epsilon", "v_prediction", "sample"):
            old_pair = System1Sanitizer.ddim_step(sample, output, alpha, alpha_prev, mode)
            new_pair = ddim_step(sample, output, alpha, alpha_prev, mode)
            for name, left, right in zip(("ddim_previous", "ddim_clean"), new_pair, old_pair):
                torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
                errors[name] = max(errors[name], (left - right).abs().max().item())

    # A synthetic perceptual callable checks scaling/plumbing, not real LPIPS.
    def synthetic_distance(a, b):
        return (a - b).square().mean().reshape(1, 1, 1, 1)

    original_image = torch.zeros(1, 3, 2, 2)
    changed_image = torch.full_like(original_image, .25).requires_grad_(True)
    old_per = LossCalculator(perceptual_model=synthetic_distance).compute_perceptual_loss(
        original_image, changed_image)
    new_per = perceptual_loss(original_image, changed_image, synthetic_distance)
    torch.testing.assert_close(old_per, new_per)

    image = torch.full((1, 3, 2, 2), .5)
    target = torch.tensor([[0., 1.]])
    queries = torch.tensor([[0., 1.], [.1, .9], [.2, .8]])
    memories = torch.tensor([[1., 0.], [.8, .2]])

    def toy_encoder(x):
        means = x.mean(dim=(2, 3))
        return torch.stack((means[:, 0] + .2, means[:, 1] + .1), dim=1)

    class ToyManager:
        available_encoders = ["toy"]

        def get_embedding_dim(self, name):
            return 2

        def encode_image(self, value, name, require_grad=False):
            return toy_encoder(value)

        def encode_text(self, texts, name):
            if isinstance(texts, str):
                return target
            return torch.cat([queries[int(text[1:]):int(text[1:]) + 1] for text in texts])

    attack = VisualInceptionAttack(ToyManager(), use_ensemble=False, lambda_perceptual=0)
    old_image, old_info = attack.optimize_perturbation(
        image, "target", memories, FixedQuerySampler(["q0", "q1", "q2"], batch_size=2))
    new_result = optimize_image(image, toy_encoder, target, queries, memories,
                                PGDConfig(batch_size=2, lambda_perceptual=0))
    torch.testing.assert_close(new_result.image.squeeze(0), old_image, rtol=0, atol=0)
    assert new_result.best_loss == old_info["best_loss"]
    assert [r["validation_loss"] for r in new_result.history[1:]] == [
        r["validation_loss"] for r in old_info["history"]]

    class PlanEncoder:
        def encode_text(self, texts):
            return torch.tensor([[1., 0.], [0., 1.]])

    verifier = System2Verifier(encoder=PlanEncoder())
    old_rds = verifier.compute_rds("original", "counterfactual")
    new_rds = reasoning_divergence(torch.tensor([[1., 0.]]), torch.tensor([[0., 1.]]))
    assert new_rds == old_rds == 1.
    assert decide_memory(new_rds, [3, 4, 5]).rds_flagged == verifier.should_flag(old_rds)
    assert not verifier.should_flag(.7) and not decide_memory(.7, [3, 4, 5]).flagged

    class Judge:
        def __init__(self):
            self.scores = iter([2, 5, 1])

        def generate(self, *unused_args, **unused_kwargs):
            return json.dumps({"score": next(self.scores)})

    old_score = verifier.validate_coherence({}, "synthetic query", llm_judge=Judge())
    assert old_score == decide_memory(.1, [2, 5, 1]).median_score == 2
    after = {path: sha256(root / path) for path in paths}
    assert before == after
    materials = {}
    for label, path in (("manuscript", args.manuscript), ("published_paper", args.paper_pdf)):
        if path is not None:
            materials[label] = {"filename": path.name, "sha256": sha256(path)}

    report = {
        "status": "passed", "scope": "synthetic numerical comparisons only",
        "python": platform.python_version(), "torch": torch.__version__,
        "source_sha256": before, "source_files_unchanged": before == after,
        "reference_materials": materials,
        "comparisons": {
            "random_loss_and_gradient_cases": 20, "ddim_cases": 60,
            "max_absolute_error": errors,
            "pgd_single_encoder_steps": 100,
            "pgd_image_exactly_equal": True, "pgd_validation_history_exactly_equal": True,
            "pgd_linf": new_result.perturbation_linf,
            "lpips_interface_only_synthetic_model": True,
            "rds_and_threshold_boundary": True, "three_vote_median": True,
        },
        "not_verified": ["real_encoders", "real_LPIPS_weights", "SBERT_weights", "real_judge",
                         "full_diffusion_pipeline", "full_System2", "paper_benchmark_results"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report["comparisons"], indent=2))


if __name__ == "__main__":
    main()
