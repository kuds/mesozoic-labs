"""The recovery-profile assembler (environments/shared/scripts/assemble_recovery_profile.py).

Each committed Compsognathus profile re-assembles byte for byte from the
measurement it names (the CI form of ``--check``); assembling writes the
measurement artifact and the profile; a measurement that cannot be the model's
profile on this tree is refused; and the assembler is no recorded source of a
measurement, so adding or editing it cannot invalidate one.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pytest

from environments.shared.model_id import ModelId
from environments.shared.paths import REPOSITORY_ROOT
from environments.shared.scripts import assemble_recovery_profile as tool

ANATOMICAL = ModelId("compsognathus_longipes", "anatomical")
ROBOT = ModelId("compsognathus_longipes", "robot")
PROFILED = (ANATOMICAL, ROBOT)
ASSEMBLER = "environments/shared/scripts/assemble_recovery_profile.py"
#: Recorded sources of the committed measurements edited since they were taken, each pinned to the sha256 of
#: the explained edit, and why the edit leaves the measured task as it was (the assembler reports such a source
#: and does not refuse it).  Any other drift fails, and so does any later edit of a listed source: re-explain it
#: here with its new hash, or re-measure.
EDITED_SINCE_MEASUREMENT = {
    # Decision D-D60 added the live-command code (the tracking reward, metrics and dwell, set_direction and the
    # independent command stream) to BaseDinoEnv; none of it runs under the recovery stage's command_mode "none",
    # whose task fingerprint, reset stream and digests are unchanged.  Decision D-D62 added the live walk's rules
    # (the command's forward reference and drift anchor, the stand block, the idle gate, set_command_config), each
    # keyed on a live command: under "none" the recovery's rollouts replay byte for byte
    # (test_none_stage_rollout_golden.py) and its task_sha256 is unchanged.  2026-11-24: a comment-only edit, no
    # behavior change (the is_success comment in _step_body now says no in-training reader collects it since
    # decision D-D59).
    "environments/shared/base_env.py": "c56fff1d42a74c21d32f53e90e649ed2ec2cf2fb67559ea3cd5b346251c74469",
    # Decision D-D62: the species reward reads BaseDinoEnv's forward reference and drift anchor, which under the
    # recovery's command_mode "none" are the spawn arrays it read before (the same golden proves it).
    "environments/compsognathus_longipes/base_env.py": "303d40f3aa13f5cf65159eb180061b7214f941eff4d9c47c531d1b4e8ee12ba8",
}


def _committed(model_id: ModelId) -> tuple[str, dict, bytes]:
    """The committed profile's text and JSON, and its uncompressed measurement."""
    text = tool.profile_path(model_id).read_text(encoding="utf-8")
    profile = json.loads(text)
    raw = tool.read_measurement(REPOSITORY_ROOT / profile["provenance"]["measurement_artifact"])
    return text, profile, raw


def test_the_committed_profiles_are_the_two_compsognathus_longipes_models():
    assert tool._committed_models(None) == list(PROFILED)


@pytest.mark.parametrize("model_id", PROFILED, ids=str)
def test_each_committed_profile_reassembles_byte_for_byte(model_id):
    text, profile, raw = _committed(model_id)
    assembled = tool.assemble_recovery_profile(
        model_id,
        raw,
        measurement_artifact=profile["provenance"]["measurement_artifact"],
        calibration_id=profile["calibration_id"],
    )
    assert tool.render_profile(assembled) == text
    assert "restamp_history" not in profile


def test_the_check_cli_passes_on_the_committed_profiles(capsys):
    # The CI form of `python -m environments.shared.scripts.assemble_recovery_profile --check`.
    assert tool.main(["--check"]) == 0
    assert capsys.readouterr().out == (
        "Recovery profiles are current (compsognathus_longipes/anatomical, compsognathus_longipes/robot)\n"
    )


def test_check_finds_a_hand_edit_and_keeps_a_restamp_history(tmp_path, capsys):
    root = tmp_path / "configs"
    for model_id in PROFILED:
        target = tool.profile_path(model_id, root)
        target.parent.mkdir(parents=True)
        shutil.copy(tool.profile_path(model_id), target)
    robot = tool.profile_path(ROBOT, root)
    profile = json.loads(robot.read_text(encoding="utf-8"))
    # Only a restamp writes the history; --check carries it over.
    profile["restamp_history"] = [{"reason": "an interface-only revision"}]
    robot.write_text(tool.render_profile(profile), encoding="utf-8")
    assert tool.main(["--check", "--configs-root", str(root)]) == 0
    capsys.readouterr()
    profile["safe_set"]["tilt_max_rad"] = 0.09
    robot.write_text(tool.render_profile(profile), encoding="utf-8")
    assert tool.main(["--check", "--configs-root", str(root)]) == 1
    assert capsys.readouterr().out == (
        "compsognathus_longipes/robot: the committed profile differs from the one its measurement assembles\n"
    )


def test_assembling_writes_the_artifact_and_the_profile(tmp_path):
    text, profile, raw = _committed(ROBOT)
    artifact = Path(profile["provenance"]["measurement_artifact"]).name
    measurement = tmp_path / "recovery_robot.json"
    measurement.write_bytes(raw)
    repository, configs = tmp_path / "repo", tmp_path / "configs"
    path, written, drifted = tool.write_recovery_profile(
        ROBOT,
        measurement,
        artifact=artifact,
        calibration_id=profile["calibration_id"],
        configs_root=configs,
        repository_root=repository,
    )
    assert path == tool.profile_path(ROBOT, configs) and path.read_text(encoding="utf-8") == text
    assert written == repository / profile["provenance"]["measurement_artifact"]
    compressed = written.read_bytes()
    assert gzip.decompress(compressed) == raw and compressed[4:8] == b"\0\0\0\0"  # gzip mtime 0
    # The committed measurement's recorded sources are this tree's, but for the explained edits.
    assert set(drifted) <= set(EDITED_SINCE_MEASUREMENT), drifted
    for source in drifted:
        assert (
            hashlib.sha256((REPOSITORY_ROOT / source).read_bytes()).hexdigest() == EDITED_SINCE_MEASUREMENT[source]
        ), f"{source} changed since its edit was explained; re-explain it in EDITED_SINCE_MEASUREMENT or re-measure"
    # The same measurement again is a no-op on the artifact; another measurement under its name is refused.
    tool.write_recovery_profile(
        ROBOT, measurement, artifact=artifact, calibration_id="x", configs_root=configs, repository_root=repository
    )
    assert written.read_bytes() == compressed
    other = tmp_path / "other.json"
    other.write_bytes(raw.replace(b'"schema_version":2', b'"schema_version":2 '))
    with pytest.raises(tool.AssemblyError, match="already holds another measurement"):
        tool.write_recovery_profile(
            ROBOT, other, artifact=artifact, calibration_id="x", configs_root=configs, repository_root=repository
        )
    with pytest.raises(tool.AssemblyError, match="bare file name ending in .json.gz"):
        tool.write_recovery_profile(ROBOT, measurement, artifact="data/x.json.gz", calibration_id="x")


def test_the_cli_assembles_from_the_committed_artifact_in_place(tmp_path, capsys):
    """``--measurement`` may be the committed ``.json.gz`` itself: nothing is re-written there."""
    text, profile, _ = _committed(ANATOMICAL)
    artifact = REPOSITORY_ROOT / profile["provenance"]["measurement_artifact"]
    before = artifact.read_bytes()
    argv = [
        "--species",
        "compsognathus_longipes",
        "--embodiment",
        "anatomical",
        "--measurement",
        str(artifact),
        "--artifact",
        artifact.name,
        "--calibration-id",
        profile["calibration_id"],
        "--configs-root",
        str(tmp_path),
    ]
    assert tool.main(argv) == 0
    assert tool.profile_path(ANATOMICAL, tmp_path).read_text(encoding="utf-8") == text
    assert artifact.read_bytes() == before
    assert capsys.readouterr().out.startswith("Wrote ")


def _tampered(raw: bytes, edit) -> bytes:
    evidence = json.loads(raw)
    edit(evidence)
    return json.dumps(evidence).encode()


def _set_selected(evidence: dict) -> None:
    evidence["selected_capture_velocity_multiple"] = 1.3


def _set_schema(evidence: dict) -> None:
    evidence["schema_version"] = 1


def _set_species(evidence: dict) -> None:
    evidence["embodiment"] = "anatomical"


def _set_mjcf_hash(evidence: dict) -> None:
    path = "environments/compsognathus_longipes/robot/assets/compsognathus_longipes_robot.xml"
    evidence["provenance"]["source_sha256"][path] = "0" * 64


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (_set_selected, "selected multiple 1.3, but its selection panels give 1.2"),
        (_set_schema, "measurement schema_version 1"),
        (_set_species, "the measurement is of compsognathus_longipes/anatomical"),
        (_set_mjcf_hash, "was not taken on the current environments/compsognathus_longipes/robot/assets"),
    ],
    ids=["selection rule", "schema", "model", "plant"],
)
def test_a_measurement_that_cannot_be_the_profile_is_refused(edit, message):
    _, profile, raw = _committed(ROBOT)
    with pytest.raises(tool.AssemblyError, match=message):
        tool.assemble_recovery_profile(
            ROBOT,
            _tampered(raw, edit),
            measurement_artifact=profile["provenance"]["measurement_artifact"],
            calibration_id=profile["calibration_id"],
        )


def test_a_selection_the_recovery_stage_does_not_run_is_refused(monkeypatch):
    from environments.shared import config

    _, profile, raw = _committed(ROBOT)
    real = config.load_stage_config

    def moved(model_id, stage):
        loaded = real(model_id, stage)
        return {**loaded, "env_kwargs": {**loaded["env_kwargs"], "perturbation_capture_velocity_multiple": 1.5}}

    monkeypatch.setattr(config, "load_stage_config", moved)
    with pytest.raises(tool.AssemblyError, match=r"multiple is 1.5; the measurement selected 1.2 \(set it in recovery"):
        tool.assemble_recovery_profile(
            ROBOT,
            raw,
            measurement_artifact=profile["provenance"]["measurement_artifact"],
            calibration_id=profile["calibration_id"],
        )


@pytest.mark.parametrize("model_id", PROFILED, ids=str)
def test_the_assembler_is_no_recorded_source(model_id):
    """``calibrate_recovery`` records its own source list in every measurement: the assembler is not on it, so
    neither the committed measurements nor a later one depend on its bytes."""
    from environments.shared.scripts.calibrate_recovery import _source_paths

    _, _, raw = _committed(model_id)
    assert ASSEMBLER not in _source_paths(model_id)
    assert ASSEMBLER not in json.loads(raw)["provenance"]["source_sha256"]
