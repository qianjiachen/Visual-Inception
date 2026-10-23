# Visual Inception — Paper Code

[中文说明](README.zh-CN.md)

This repository contains the paper publication code for *Visual Inception*. It provides the single-encoder objective and bounded PGD attack, RDS and three-call relevance voting rules, and forward-noising and deterministic DDIM single-step updates.

Jiachen Qian, *Visual Inception: Compromising Long-term Planning in Agentic Recommenders via Multimodal Memory Poisoning*, ACL 2026: [paper page](https://aclanthology.org/2026.acl-long.954/) · [PDF](https://aclanthology.org/2026.acl-long.954.pdf).

Version `0.1.0` is the checked publication-code release of the selected components. See [RELEASE_NOTES.md](RELEASE_NOTES.md) for the version contents and the implementation conventions below.

## Included scope

| Area | Included | Paper basis and scope |
| --- | --- | --- |
| Objectives | Retrieval log-softmax, target semantic cosine loss, and an interface for a real LPIPS model | §3.2; the retrieval denominator convention is stated below |
| Optimization | Single-encoder, zero-start, bounded sign-gradient descent; 100 steps by default | Appendix A.2 step count, step size, perturbation budget, and loss weights |
| Verification | RDS from paired plan embeddings and a majority decision over three relevance scores | §4.2; the Agent, SBERT model, and LLM judge are not included |
| Diffusion numerics | Forward noising and one deterministic DDIM step (`eta=0`) with three prediction types | Forward equation in §4.1; DDIM is the documented implementation choice |

The release is limited to the components above. Cross-encoder integration, hard-negative recipes, query-source construction, real encoder preprocessing, memory stores, the complete Agent, the full System 1/2 pipelines, metric evaluators, third-party baselines, datasets, and model weights are outside this version. Cite the paper for the reported experiments; the synthetic tests in this package validate the released numerical code.

## Installation and usage

Requirements: Python ≥ 3.10 and PyTorch ≥ 2.2. The tested environment is recorded in [VALIDATION.md](VALIDATION.md); other versions were not exhaustively tested.

```bash
python -m pip install .
python -m pip install pytest
python -m pytest -q
python -m examples.toy_demo
```

Installing dependencies may access a package index. The algorithms and demo do not make network requests, download models, or call APIs. If the dependencies are already installed, tests and the demo can be run from this directory without installing the package.

The demo uses synthetic tensors and a toy encoder and explicitly sets `lambda_perceptual=0`, so it runs a reduced objective without a perceptual term. The default configuration retains the paper value `lambda_perceptual=0.1` and requires the caller to provide a real LPIPS model; it raises an error when that model is absent.

## Implementation conventions

1. The retrieval denominator contains the adversarial image and the existing surrogate memories, with the adversarial image appearing once. The paper does not make the set-membership convention fully explicit; this release uses the positive-class softmax definition stated in [ALGORITHMS.md](ALGORITHMS.md).
2. The caller provides query embeddings. Each step samples up to 16 indices without replacement from the fixed query pool. The paper's 100 queries and its 50%/30%/20% source construction are not implemented here; an arbitrary input matrix does not automatically satisfy that protocol.
3. PGD starts from the original image, evaluates the initial point and every update with the same full query pool, and returns the image with the lowest full-pool objective. Initialization and checkpoint selection are documented implementation choices where the paper does not fully specify them.
4. The optimizer accepts one encoder. The image, target, query, and memory embeddings must come from the same embedding space; matching tensor shapes cannot establish model or preprocessing compatibility.
5. RDS inputs must be paired Agent plans encoded by the same SBERT configuration, with and without the specified memory. Relevance inputs must be the outputs of three real judge calls. This package computes the scores and rules but does not generate those inputs.

The formulas, input contracts, and pseudocode are in [ALGORITHMS.md](ALGORITHMS.md). The paper mapping and version boundary are in [SCOPE_AND_ALIGNMENT.md](SCOPE_AND_ALIGNMENT.md).

## Validation and license

Tests cover the formulas, gradients, projection constraint, checkpoint selection, threshold boundaries, and analytic DDIM relations. [VALIDATION.md](VALIDATION.md) records the independent tests, numerical comparisons with selected original paths, and installation checks outside the original source tree. Real models and the paper experiments are outside this numerical validation.

The code is released under the [MIT License](LICENSE). Third-party models, datasets, and the paper remain subject to their respective licenses. The source mapping is documented in [SOURCE_ORIGINS.md](SOURCE_ORIGINS.md).

## Citation

Please cite the paper when using this code. BibTeX is available in [references.bib](references.bib), and software citation metadata is provided in [CITATION.cff](CITATION.cff).
