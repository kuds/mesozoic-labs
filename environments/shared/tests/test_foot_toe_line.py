"""The held-offset toe-line rule (``scripts/foot_toe_line.py``), on synthetic levels.

The rule's simulation is a hand run (its record is pinned to the plant in
``test_gait_morphology.py``); what is tested here is the reduction that
turns a record's episodes into a line, so a record can be re-checked
without re-rolling it.
"""

from __future__ import annotations

import pytest

from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.shared.scripts.foot_toe_line import (
    RESOLUTION_M,
    SEEDS,
    START_DEG,
    STEP_DEG,
    ankle_onset,
    level_median,
    record_path,
    toe_line_from_onsets,
    toe_line_of_record,
)


def _episodes(values, falls=()):
    return [{"seed": seed, "fell": seed in falls, "other_foot_cop_m": value} for seed, value in zip(SEEDS, values)]


def _level(deg, values, falls=()):
    return {"deg": deg, "episodes": _episodes(values, falls)}


def test_the_rule_is_fixed():
    assert SEEDS == (3042, 3043, 3044, 3045, 3046)
    assert (START_DEG, STEP_DEG, RESOLUTION_M) == (-2.5, 0.5, 0.001)
    assert record_path(ModelId("tyrannosaurus_rex", "anatomical")) == (
        ENVIRONMENTS_ROOT / "tyrannosaurus_rex" / "anatomical" / "data" / "foot_toe_line.json"
    )


def test_a_level_is_the_median_of_its_standing_episodes():
    level = _episodes([0.20, 0.21, 0.30, 0.22, 0.23], falls={3044})
    assert level_median(level) == pytest.approx(0.215)  # the fallen 0.30 is left out
    assert level_median(_episodes([0.2] * 5, falls=set(SEEDS))) is None


def test_the_onset_is_the_first_level_with_a_fall_and_the_line_the_smaller_onset_floored():
    right = [_level(-2.5, [0.18] * 5), _level(-3.0, [0.19] * 5), _level(-3.5, [0.2148] * 5, falls={3042})]
    left = [_level(-2.5, [0.18] * 5), _level(-3.0, [0.21344] * 5, falls={3045, 3046})]
    assert ankle_onset(right)["deg"] == -3.5 and ankle_onset(left)["deg"] == -3.0
    record = {"ankles": {"r": {"levels": right}, "l": {"levels": left}}}
    # Floored, never rounded: 213.44 mm -> 0.213 m (nearest-5 mm would read 0.215).
    assert toe_line_of_record(record) == 0.213
    assert toe_line_from_onsets([0.2139999, 0.25]) == 0.213
    assert toe_line_from_onsets([0.213]) == 0.213  # an exact millimetre stays put


def test_a_grid_with_no_fall_or_no_standing_episode_at_the_onset_gives_no_line():
    standing = [_level(-2.5, [0.18] * 5)]
    with pytest.raises(ValueError, match="did not fall"):
        toe_line_of_record({"ankles": {"r": {"levels": standing}}})
    all_fell = [_level(-2.5, [0.18] * 5, falls=set(SEEDS))]
    with pytest.raises(ValueError, match="no standing episode"):
        toe_line_of_record({"ankles": {"r": {"levels": all_fell}}})
