"""Every command-free stage reproduces the rollouts captured before the live walk (decision D-D62).

The live walk adds reward code that runs while the command is exactly zero,
and a command-free stage's command is exactly zero too.  Each stance and
recovery must therefore roll out exactly as it did on the tree of decision
D-D61 (``none_stage_rollout_golden.py``, ``fixtures/none_stage_rollout_golden.json``):

* **Exactly**, in the numerics environment the fixture was captured in:
  every reward, flag, info key and value with its type, observation and
  reset, through the per-rollout, per-component and per-block digests, the
  episodes and the reward sums; elsewhere this skips, naming the difference.
* **Portably**, on any machine with the fixture's MuJoCo release and numpy
  major version: each rollout's first steps within the pilot golden's
  tolerance, and the task digest, command mode and info keys exactly.
* The golden covers every stance and recovery the stage manifests declare,
  each of them ``command_mode = "none"``, and was captured on a committed
  tree.
* The digests name what moved and where: a one-ulp reward change and a
  changed info value type are each reported in their component and block.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import numpy as np
import pytest

from environments.shared import config
from environments.shared.config import load_stage_config
from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids
from environments.shared.stage_manifest import load_stage_manifest
from environments.shared.tests import none_stage_rollout_golden as golden

GOLDEN = golden.load_fixture()
CAPTURE_IDS = [golden.capture_id(*spec) for spec in golden.CAPTURE_SPECS]
#: The cheapest stage to roll (25 info keys, about 0.3 s a rollout).  Its zero-action statue falls on its 97th step at
#: seed 42, so the tampered step 130 below lies in the episode after a fall and a reset.
PROBE = (ModelId("allosaurus_fragilis", "anatomical"), "stance")
#: The commit of decision D-D61 the fixture was captured on.
SOURCE_COMMIT = "857b6dd"
#: The construction path a rollout takes, unpatched (each tampered rollout below patches ``config.build_env`` anew).
BUILD_ENV = config.build_env


@functools.lru_cache(maxsize=None)
def _capture(model_id: ModelId, stage: str) -> dict[str, Any]:
    """One stage's capture on this tree, shared by the exact and the portable replay."""
    return golden.capture(model_id, stage)


def _stored(model_id: ModelId, stage: str) -> dict[str, Any]:
    stored = golden.stored_capture(GOLDEN, model_id, stage)
    assert stored is not None, f"the fixture has no capture of {model_id} {stage}"
    return stored


@pytest.mark.parametrize("spec", golden.CAPTURE_SPECS, ids=CAPTURE_IDS)
def test_every_command_free_stage_reproduces_its_golden_rollouts_exactly(spec):
    """Byte for byte: the task digest, the info keys and every rollout's digests, episodes, reward sum and rows."""
    if mismatch := golden.numerics_mismatch(GOLDEN):
        pytest.skip(f"the golden's numerics environment differs here in {mismatch}; its digests are exact only there")
    expected, actual = _stored(*spec), _capture(*spec)
    differences = golden.exact_differences(expected, actual)
    assert not differences, "\n".join(differences)
    assert golden.dumps(actual) == golden.dumps(expected)


@pytest.mark.parametrize("spec", golden.CAPTURE_SPECS, ids=CAPTURE_IDS)
def test_every_command_free_stage_reproduces_its_first_steps_on_any_machine(spec):
    """The portable companion: the first steps' rewards, flags, info key order and values within the tolerance."""
    if mismatch := golden.portable_mismatch(GOLDEN):
        pytest.skip(f"the golden's {mismatch} differ here; a step moves beyond the portable tolerance")
    differences = golden.portable_differences(_stored(*spec), _capture(*spec))
    assert not differences, "\n".join(differences[:20])


def test_the_golden_covers_every_command_free_stage():
    """Each model's stance and every recovery the manifests declare, each committed with no [commands] table."""
    declared = {
        (model_id, entry.id)
        for model_id in model_ids()
        for entry in load_stage_manifest(model_id).stages
        if entry.id in {"stance", "recovery"}
    }
    assert set(golden.CAPTURE_SPECS) == declared
    assert len(golden.CAPTURE_SPECS) == len(declared)
    for model_id, stage in golden.CAPTURE_SPECS:
        env_kwargs = load_stage_config(model_id, stage)["env_kwargs"]
        assert env_kwargs.get("command_mode", "none") == "none", (model_id, stage)
        assert "command_config" not in env_kwargs, (model_id, stage)
        assert _stored(model_id, stage)["command_mode"] == "none"


def test_the_golden_was_captured_on_the_committed_tree_of_decision_d_d61():
    assert GOLDEN["schema"] == golden.FIXTURE_SCHEMA
    assert GOLDEN["source_commit"] == SOURCE_COMMIT and GOLDEN["source_modified_files"] == []
    assert GOLDEN["seeds"] == list(golden.SEEDS) and GOLDEN["steps"] == golden.STEPS
    assert GOLDEN["action_kinds"] == list(golden.ACTION_KINDS)
    assert (GOLDEN["action_seed"], GOLDEN["action_amplitude"]) == (golden.ACTION_SEED, golden.ACTION_AMPLITUDE)
    assert (GOLDEN["block_steps"], GOLDEN["portable_steps"]) == (golden.BLOCK_STEPS, golden.PORTABLE_STEPS)
    assert GOLDEN["portable_prefix"] == list(golden.PORTABLE_PREFIX)
    assert GOLDEN["components"] == list(golden.COMPONENTS)
    assert GOLDEN["spec"] == [{**model_id.record(), "stage": stage} for model_id, stage in golden.CAPTURE_SPECS]
    blocks = math.ceil(golden.STEPS / golden.BLOCK_STEPS)
    for model_id, stage in golden.CAPTURE_SPECS:
        stored = _stored(model_id, stage)
        assert (stored["species"], stored["embodiment"], stored["stage"]) == (
            model_id.species,
            model_id.embodiment,
            stage,
        )
        assert set(stored["rollouts"]) == set(golden.ACTION_KINDS)
        for kind, by_seed in stored["rollouts"].items():
            assert set(by_seed) == {str(seed) for seed in golden.SEEDS}, (model_id, stage, kind)
            for seed, record in by_seed.items():
                assert (record["seed"], record["action"]) == (int(seed), kind)
                assert set(record["component_sha256"]) == set(golden.COMPONENTS)
                assert len(record["block_sha256"]) == blocks
                assert len(record["portable"]) == golden.PORTABLE_STEPS
                assert sum(episode["length"] for episode in record["episodes"]) == golden.STEPS
                assert {episode["ended_by"] for episode in record["episodes"][:-1]} <= {"terminated", "truncated"}
    episodes = [
        episode
        for model_id, stage in golden.CAPTURE_SPECS
        for by_seed in _stored(model_id, stage)["rollouts"].values()
        for record in by_seed.values()
        for episode in record["episodes"]
    ]
    assert any(episode["ended_by"] == "terminated" for episode in episodes), "no rollout covers a fall and its reset"


def _probe_rollout(monkeypatch: pytest.MonkeyPatch, tamper=None) -> dict[str, Any]:
    """The probe stage's zero-action seed-42 rollout, as a one-rollout capture, with *tamper* applied to each step."""

    def build(model_id: ModelId, stage: str, **overrides: Any) -> Any:
        env = BUILD_ENV(model_id, stage, **overrides)
        if tamper is not None:
            step, counter = env.step, iter(range(golden.STEPS))

            def tampered(action: Any) -> Any:
                return tamper(next(counter), *step(action))

            env.step = tampered
        return env

    monkeypatch.setattr(config, "build_env", build)
    record = golden.rollout(*PROBE, golden.SEEDS[0], "zero")
    record.pop("info_keys")
    record.pop("command_mode")
    return {"rollouts": {"zero": {str(golden.SEEDS[0]): record}}}


def test_the_digests_name_what_moved_and_where(monkeypatch):
    """A one-ulp reward change at step 130 is a reward move in block 5; one info value's type flipped between Python's
    float and numpy's float64, its value unchanged, at step 3 is an info move in block 0 and nothing else; an unchanged
    re-roll is no difference."""
    baseline = _probe_rollout(monkeypatch)
    assert golden.exact_differences(baseline, _probe_rollout(monkeypatch)) == []

    def nudge_reward(index, observation, reward, terminated, truncated, info):
        if index == 130:
            reward = type(reward)(math.nextafter(reward, math.inf))
        return observation, reward, terminated, truncated, info

    moved = golden.exact_differences(baseline, _probe_rollout(monkeypatch, nudge_reward))
    assert moved[0].endswith("reward moved, first in block 5 (steps 125-149)"), moved
    assert not any("episodes" in line or "portable" in line for line in moved), moved

    def retype_info(index, observation, reward, terminated, truncated, info):
        if index == 3:
            key, value = next((key, value) for key, value in info.items() if isinstance(value, float))
            info = {**info, key: np.float64(value) if type(value) is float else float(value)}
        return observation, reward, terminated, truncated, info

    moved = golden.exact_differences(baseline, _probe_rollout(monkeypatch, retype_info))
    assert moved == [f"zero seed {golden.SEEDS[0]}: info moved, first in block 0 (steps 0-24)"], moved


def test_the_portable_comparison_tolerates_the_last_bits_and_nothing_more():
    """A last-bit float change passes; a relative 1e-7 change, a type change, a flipped flag and a moved info key
    order are each refused."""
    stored = _stored(*PROBE)
    row = stored["rollouts"]["zero"][str(golden.SEEDS[0])]["portable"][0]
    keys = stored["info_keys"]
    prefix = len(golden.PORTABLE_PREFIX)
    column = next(
        prefix + index for index, value in enumerate(row[prefix:]) if isinstance(value, float) and abs(value) > 1e-3
    )

    def changed(index: int, value: Any) -> list[Any]:
        return [value if position == index else item for position, item in enumerate(row)]

    assert golden.row_differences(row, list(row), keys) == []
    assert golden.row_differences(row, changed(column, math.nextafter(row[column], math.inf)), keys) == []
    (refused,) = golden.row_differences(row, changed(column, row[column] * (1 + 1e-7)), keys)
    assert refused.startswith(f"{keys[column - prefix]}:"), refused
    assert golden.row_differences(row, changed(column, str(row[column])), keys)
    assert golden.row_differences(row, changed(golden.PORTABLE_PREFIX.index("terminated"), True), keys)
    keys_column = golden.PORTABLE_PREFIX.index("info_keys_sha256")
    assert golden.row_differences(row, changed(keys_column, "0" * 16), keys) == [
        f"the info key order moved ({len(row)} columns, expected {len(row)})"
    ]
