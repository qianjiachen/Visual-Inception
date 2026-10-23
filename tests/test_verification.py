import itertools
import math
import pytest
import torch
from visual_inception_core import reasoning_divergence, decide_memory


@pytest.mark.parametrize("second,expected", [([1., 0.], 0), ([0., 1.], 1), ([-1., 0.], 2)])
def test_rds_angles(second, expected):
    first = torch.tensor([[1., 0.]])
    assert reasoning_divergence(first, torch.tensor([second])) == expected


def test_strict_rds_boundary_and_inclusive_coherence_boundary():
    assert not decide_memory(.7, [3, 4, 5]).flagged
    assert decide_memory(math.nextafter(.7, math.inf), [3, 4, 5]).rds_flagged
    assert decide_memory(.1, [2, 2, 5]).coherence_flagged
    assert not decide_memory(.1, [2, 3, 5]).coherence_flagged


def test_median_decision_equals_majority_for_all_125_score_triples():
    for scores in itertools.product(range(1, 6), repeat=3):
        decision = decide_memory(.1, scores)
        expected = sum(score <= 2 for score in scores) >= 2
        assert decision.coherence_flagged == expected
        assert decision.flagged == expected


@pytest.mark.parametrize("scores", [[1, 2], [1, 2, 3, 4], [True, 2, 5], [1., 2, 3], [0, 2, 3], [1, 2, 6]])
def test_invalid_judge_scores(scores):
    with pytest.raises(ValueError):
        decide_memory(.1, scores)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 2.1])
def test_invalid_rds(value):
    with pytest.raises(ValueError):
        decide_memory(value, [3, 4, 5])


def test_zero_embedding_is_not_a_meaningful_plan():
    with pytest.raises(ValueError):
        reasoning_divergence(torch.zeros(1, 2), torch.ones(1, 2))
