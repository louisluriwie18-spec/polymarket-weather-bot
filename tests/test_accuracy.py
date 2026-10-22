import pytest
from accuracy_report import generate_accuracy_report


def test_brier_score_logic():
    # Helper to calculate Brier score
    def calc_brier(predictions, outcomes):
        return sum((p - o) ** 2 for p, o in zip(predictions, outcomes)) / len(predictions)

    # Perfect predictions
    assert calc_brier([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0.0

    # Completely wrong predictions
    assert calc_brier([0.0, 1.0, 0.0], [1.0, 0.0, 1.0]) == 1.0

    # 50/50 baseline
    assert calc_brier([0.5, 0.5], [1.0, 0.0]) == 0.25

    # Skill score: Model Brier 0.15 vs Market Brier 0.20
    model_brier = 0.15
    mkt_brier = 0.20
    skill_score = 1.0 - (model_brier / mkt_brier)
    assert abs(skill_score - 0.25) < 1e-4  # +25% skill over market
