from __future__ import annotations

import math
from pathlib import Path

import pytest

from common.config_io import validate_target_sample_range


def _common() -> dict:
    return {
        "target": {
            "sample_range": {
                "x": [-0.5, -0.4],
                "y": [-0.1, 0.1],
                "z": [-0.0125, -0.0125],
            }
        }
    }


def test_target_contract_accepts_inclusive_trained_bounds():
    assert validate_target_sample_range(
        (-0.5, 0.1, -0.0125), _common(), tag="test"
    ) == (-0.5, 0.1, -0.0125)


@pytest.mark.parametrize(
    "target",
    [
        (-0.500001, 0.0, -0.0125),
        (-0.45, 0.100001, -0.0125),
        (-0.45, 0.0, -0.0124),
        (-0.45, 0.0, math.nan),
        (-0.45, 0.0),
    ],
)
def test_target_contract_rejects_out_of_distribution_or_invalid_values(target):
    with pytest.raises(ValueError):
        validate_target_sample_range(target, _common(), tag="test")


def test_sim_rehearsal_uses_shared_target_contract():
    source = (Path(__file__).parents[1] / "Phase3/SimDeploy_Rehearsal.py").read_text(
        encoding="utf-8"
    )
    assert "validate_target_sample_range(" in source
