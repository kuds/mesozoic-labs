"""The gait protocol identity covers measurement code only and is checked before training."""

from __future__ import annotations

import copy
import shutil
from types import SimpleNamespace

import pytest

from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.curriculum.gait_gate import GAIT_GATE_KIND, provisional_gait_criteria
from environments.shared.gait import identity, preflight
from environments.shared.gait.identity import (
    current_implementation_hashes,
    measurement_protocol,
    protocol_sha256,
    stage_measurement_protocol,
)
from environments.shared.gait.preflight import GaitPreflightError, check_gait_stage, check_gait_stages
from environments.shared.gait.types import GaitProtocol
from environments.shared.paths import REPOSITORY_ROOT
from environments.trex.envs.trex_env import TRexEnv

SPECIES = SimpleNamespace(species="trex", env_class=TRexEnv)
MEASUREMENT_SOURCES = {
    "environments/shared/gait/types.py",
    "environments/shared/gait/events.py",
    "environments/shared/gait/labels.py",
    "environments/shared/gait/metrics.py",
    "environments/shared/gait/morphology.py",
    "environments/shared/gait/recorder.py",
    "environments/shared/curriculum/gait_gate.py",
}


def test_protocol_hash_covers_only_the_measurement_and_classification_code():
    hashes = current_implementation_hashes()
    assert set(hashes) == MEASUREMENT_SOURCES
    for unrelated in ("reporting/gates.py", "curriculum/gate_schema.py", "curriculum/recovery_gate.py"):
        assert f"environments/shared/{unrelated}" not in hashes
    payload = measurement_protocol(
        "trex",
        GaitProtocol(),
        settle_s=1.0,
        direction_xy=(1.0, 0.0),
        horizon=1000,
        physics_dt_s=0.002,
        control_dt_s=0.01,
        episodes=40,
        seed_start=PUBLICATION_SEED_START,
    )
    assert payload["measurement_version"] == identity.MEASUREMENT_VERSION
    # Runtime versions are recorded beside a report, never hashed.
    assert "mujoco_version" not in payload["sampling"]


def _tree(root, *, edit=None, crlf=None):
    """A copy of the hashed sources (plus the unrelated shared readers) under *root*."""
    shared = root / "environments" / "shared"
    for relative in (*MEASUREMENT_SOURCES, "environments/shared/reporting/gates.py"):
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPOSITORY_ROOT / relative, target)
    if edit is not None:
        with (root / edit).open("a", encoding="utf-8") as handle:
            handle.write("# an unrelated comment\n")
    if crlf is not None:
        path = root / crlf
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    return shared


@pytest.mark.parametrize(
    "edit,moves",
    [
        ("environments/shared/reporting/gates.py", False),  # a comment in the shared reader revokes nothing
        ("environments/shared/gait/metrics.py", True),
        ("environments/shared/curriculum/gait_gate.py", True),
    ],
)
def test_only_measurement_edits_move_the_planned_hash(tmp_path, monkeypatch, edit, moves):
    def hashes(root, **changes):
        shared = _tree(root, **changes)
        monkeypatch.setattr(identity, "REPOSITORY_ROOT", root)
        monkeypatch.setattr(identity, "SHARED_ROOT", shared)
        return current_implementation_hashes()

    base = hashes(tmp_path / "base")
    assert (hashes(tmp_path / "edited", edit=edit) != base) is moves


def test_a_crlf_checkout_is_the_same_implementation(tmp_path, monkeypatch):
    original = current_implementation_hashes()
    shared = _tree(tmp_path, crlf="environments/shared/gait/metrics.py")
    assert b"\r\n" in (shared / "gait" / "metrics.py").read_bytes()
    monkeypatch.setattr(identity, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(identity, "SHARED_ROOT", shared)
    assert current_implementation_hashes() == original


def _gait_stage(**updates):
    """A T. rex locomotion config declaring the gait gate with this build's planned digest."""
    config = {"env_kwargs": {"max_episode_steps": 300, "reset_noise_scale": 0.01}}
    planned = protocol_sha256(
        stage_measurement_protocol(SPECIES, config, episodes=40, seed_start=PUBLICATION_SEED_START)
    )
    config["curriculum_kwargs"] = {
        **provisional_gait_criteria("biped_walk"),
        "gate_kind": GAIT_GATE_KIND,
        "gate_schema_version": 1,
        "gait_profile": "biped_walk",
        "measurement_protocol_sha256": planned,
        "min_eval_episodes": 40,
        "gait_panel_seed_start": PUBLICATION_SEED_START,
        "min_gait_success_lcb": 0.8,
        "min_episode_forward_vel": 0.5,
        "min_episode_duration_s": 1.5,
        **updates,
    }
    return config


def test_current_declaration_passes_and_other_kinds_are_not_checked():
    config = _gait_stage()
    assert check_gait_stage(SPECIES, 2, config) == config["curriculum_kwargs"]["measurement_protocol_sha256"]
    assert check_gait_stage(SPECIES, 2, {"curriculum_kwargs": {"gate_kind": "reward_and_length/v1"}}) is None


def test_stale_declared_digest_is_refused_with_the_replanning_command():
    config = _gait_stage(measurement_protocol_sha256="sha256:" + "0" * 64)
    with pytest.raises(GaitPreflightError, match="--protocol-only --episodes 40 --seed 3042"):
        check_gait_stage(SPECIES, 2, config)


def test_malformed_gate_is_refused():
    with pytest.raises(GaitPreflightError, match="cannot be planned"):
        check_gait_stage(SPECIES, 2, _gait_stage(min_eval_episodes=40.0))


@pytest.mark.parametrize(
    "seed,n_envs,match",
    [
        (3040, 4, "training environment"),
        (2050, 4, "checkpoint-selection"),
        (1040, 4, "replay video"),  # replay_seed(1040, 2) = 3042
        (848, 1, "replay video"),  # replay_seed(848, "locomotion") = 3042
    ],
)
def test_panel_overlapping_the_run_seeds_is_refused_before_training(seed, n_envs, match):
    with pytest.raises(GaitPreflightError, match=match):
        check_gait_stage(SPECIES, 2, _gait_stage(), training_seed=seed, n_envs=n_envs)
    assert check_gait_stage(SPECIES, 2, _gait_stage(), training_seed=42, n_envs=4)


def test_check_gait_stages_names_every_failure():
    stale = _gait_stage(measurement_protocol_sha256="sha256:" + "0" * 64)
    with pytest.raises(GaitPreflightError) as error:
        check_gait_stages(SPECIES, {1: stale, 2: copy.deepcopy(stale)}, [1, 2])
    assert "stage 1" in str(error.value) and "stage 2" in str(error.value)


def test_train_curriculum_refuses_a_stale_gait_declaration_before_writing(tmp_path):
    pytest.importorskip("stable_baselines3")
    from environments.shared.config import load_all_stages
    from environments.shared.species_registry import get_species_config
    from environments.shared.train_base import train_curriculum

    configs = load_all_stages("trex")
    configs[2] = dict(
        configs[2], curriculum_kwargs=_gait_stage(measurement_protocol_sha256="sha256:" + "0" * 64)["curriculum_kwargs"]
    )
    output = tmp_path / "run"
    with pytest.raises(GaitPreflightError, match="declares measurement_protocol_sha256"):
        train_curriculum(get_species_config("trex"), configs, output_dir=str(output), target=2)
    assert not output.exists()


def test_committed_stages_check_cleanly(capsys):
    assert preflight.main(["--check"]) == 0  # every trainable species, aliases resolved once
    assert f"No committed stage declares {GAIT_GATE_KIND}" in capsys.readouterr().out
    assert preflight.main(["--check", "dibo", "trex"]) == 0


def test_check_fails_on_a_stale_committed_declaration(monkeypatch, capsys):
    from environments.shared import config as config_module

    stale = _gait_stage(measurement_protocol_sha256="sha256:" + "0" * 64)
    monkeypatch.setattr(config_module, "load_all_stages", lambda species: {2: stale})
    assert preflight.main(["--check", "trex"]) == 1
    assert "FAIL: trex stage 2 declares measurement_protocol_sha256" in capsys.readouterr().err


def test_check_requires_the_flag_and_known_species(capsys):
    with pytest.raises(SystemExit) as error:
        preflight.main([])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        preflight.main(["--check", "unicorn"])
    assert error.value.code == 2
