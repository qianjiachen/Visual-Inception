"""Numerical scores and decisions only; no agent, SBERT or judge is invoked."""
from dataclasses import dataclass
from statistics import median
import torch.nn.functional as F
from . import _validation as check


@dataclass(frozen=True)
class VerificationDecision:
    rds: float
    median_score: int
    rds_flagged: bool
    coherence_flagged: bool
    flagged: bool


def reasoning_divergence(original, counterfactual):
    """1-cos for the [1,D] SBERT embeddings of paired agent plans."""
    check.embeddings(original, "original")
    check.embeddings(counterfactual, "counterfactual")
    check.compatible(original, counterfactual)
    if original.shape != counterfactual.shape or original.shape[0] != 1:
        raise ValueError("Provide one equal-dimensional embedding per plan, [1,D]")
    cosine = F.cosine_similarity(original, counterfactual, dim=-1).clamp(-1, 1)
    return float((1 - cosine).item())


def decide_memory(rds, judge_scores, theta_anomaly=0.7, coherence_threshold=2):
    """Flag if RDS > threshold OR at least two of exactly three scores <= 2.

    judge_scores are actual integer scores from three independent judge calls.
    Equality at the RDS threshold is not a flag; equality at coherence is a flag.
    """
    check.real(rds, "rds", 0, 2)
    check.real(theta_anomaly, "theta_anomaly", 0, 2, strict_min=True)
    if theta_anomaly >= 2:
        raise ValueError("theta_anomaly must be < 2")
    check.integer(coherence_threshold, "coherence_threshold")
    if coherence_threshold > 5:
        raise ValueError("coherence_threshold must be in [1,5]")
    scores = tuple(judge_scores)
    if len(scores) != 3 or any(type(x) is not int or not 1 <= x <= 5 for x in scores):
        raise ValueError("Exactly three integer judge scores in [1,5] are required")
    score = int(median(scores))
    pivot, incoherent = rds > theta_anomaly, score <= coherence_threshold
    return VerificationDecision(float(rds), score, pivot, incoherent, pivot or incoherent)
