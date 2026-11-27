"""The walk warm-start noise-tolerance scan (``scripts/warm_start_noise_scan.py``): report only.

The grid, the stand counts, the answer and the refusals before anything loads
are pure and run everywhere.  The rollouts use tiny real PPO stance parents
(:func:`tiny_parent`), saved as a run's stance stage directory holds its
handoff pair and stamped with the model's current plant identity, so the
selector, the loader's plant checks and the walk env run as on a real parent.
The smoke runs cover a T. rex, the Compsognathus robot (a 0.01 s and a 0.02 s
control step, two different plants) and a critic-phase, mirror-KL T. rex
parent (the kind of the T. rex's certified stance) on two seeds, one noise
stream and a short horizon.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NamedTuple, cast

import numpy as np
import pytest

from environments.shared.model_id import ModelId
from environments.shared.scripts import warm_start_noise_scan as scan
from environments.shared.stage_manifest import StageManifest, load_stage_manifest, load_stage_manifest_file

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
#: Walking pilot 3's parent: its largest learned log_std (tail_1_yaw) and its smallest (a knee).
PILOT3_PARENT_LOG_STD = (-2.2006, -4.96, -4.37, -2.96)
SMOKE_SEEDS = "3042,3043"
SMOKE_HORIZON = 12
TYRANNOSAURUS_REX_ARGS = ["--species", "tyrannosaurus_rex", "--embodiment", "anatomical"]
#: A panel that keeps a refusal test short even if the refusal regressed.
SHORT = ["--seeds", "3042", "--streams", "1", "--horizon", "2", "--log-std", "-2"]


def _row(value: float, stood: int, episodes: int = 120) -> dict[str, Any]:
    return {"log_std": value, "std": float(np.exp(value)), "stood": stood, "episodes": episodes}


def _reference(stood: int, episodes: int) -> dict[str, Any]:
    return {"stood": stood, "episodes": episodes}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------------------------------- the grid


def test_the_default_grid_is_centred_on_the_parents_largest_learned_log_std_on_the_step_lattice():
    grid = scan.default_grid(PILOT3_PARENT_LOG_STD)
    # -2.2006 rounds to -2.25; the first pass runs -0.5 below it to +1.0 above it in 0.25 steps.
    assert scan.grid_reference(PILOT3_PARENT_LOG_STD) == pytest.approx(-2.2006)
    assert scan.grid_centre(PILOT3_PARENT_LOG_STD) == -2.25
    assert grid == tuple(-2.75 + 0.25 * index for index in range(7))
    # Pilot 3's own grid is the first pass's middle, its answer -2.0 included.
    assert {-2.5, -2.25, -2.0, -1.75, -1.5} <= set(grid)


def test_a_parent_on_the_lattice_is_its_own_centre_and_the_grid_moves_with_the_parent():
    assert scan.default_grid([-2.0, -3.0])[2] == -2.0
    assert scan.default_grid([-1.0])[0] == -1.5
    assert scan.default_grid([-1.0])[-1] == 0.0
    # The centre is on the 0.125 lattice, so a parent between two 0.25 steps keeps its own centre.
    assert scan.default_grid([-2.13])[2] == -2.125
    assert all(value / scan.REFINE_STEP == round(value / scan.REFINE_STEP) for value in scan.default_grid([-2.13]))
    with pytest.raises(scan.NoiseScanError, match="finite"):
        scan.default_grid([float("nan")])


def test_the_refinement_halves_the_gap_above_the_first_passs_answer():
    rows = [_row(-2.5, 120), _row(-2.25, 120), _row(-2.0, 120), _row(-1.75, 91), _row(-1.5, 9)]
    # Pilot 3's first pass: -2.0 stood and -1.75 fell, so the refinement is the -1.875 pilot 3 added.
    assert scan.refinement(rows) == -1.875
    # A value below the answer that fell does not move it: the refinement follows the literal answer.
    assert scan.refinement([_row(-2.5, 119), _row(-2.25, 120), _row(-2.0, 100)]) == -2.125


def test_there_is_no_refinement_when_the_edge_lies_outside_the_first_pass():
    # Nothing stood: the edge is below the grid.
    assert scan.refinement([_row(-2.5, 100), _row(-2.25, 50)]) is None
    # The largest value stood: the edge is above the grid.
    assert scan.refinement([_row(-2.5, 120), _row(-2.25, 120)]) is None
    # The next value up is already one refinement step away.
    assert scan.refinement([_row(-2.0, 120), _row(-1.875, 100)]) is None


def test_explicit_values_are_sorted_and_a_repeat_is_refused():
    assert scan.checked_grid([-1.5, -2.0, -1.75]) == (-2.0, -1.75, -1.5)
    with pytest.raises(scan.NoiseScanError, match="repeats"):
        scan.checked_grid([-2.0, -2.0])
    with pytest.raises(scan.NoiseScanError, match="finite"):
        scan.checked_grid([float("inf")])


# ---------------------------------------------------------------------------------------------------- the summary


def _episode(stream: int, seed: int, *, stood: bool, reason: str | None = None, lifts: int | None = 3) -> dict:
    return {
        "seed": seed,
        "stream": stream,
        "length": 10 if stood else 4,
        "stood": stood,
        "termination_reason": None if stood else reason,
        "return": 1.0,
        "lift_offs": lifts,
        "forward_displacement_m": -0.1 if stood else 0.5,
        "nonfinite": False,
        "clipped_fraction": 0.0,
    }


def test_the_stand_count_is_per_stream_and_the_readouts_read_the_standing_episodes_only():
    by_stream = [
        [_episode(0, 3042, stood=True), _episode(0, 3043, stood=False, reason="fallen")],
        [_episode(1, 3042, stood=True, lifts=0), _episode(1, 3043, stood=False, reason="head_contact")],
        [_episode(2, 3042, stood=False, reason="fallen"), _episode(2, 3043, stood=True)],
    ]
    row = scan.summarize_setting(-2.0, by_stream)
    assert (row["label"], row["kind"]) == ("-2", "grid")
    assert row["stood_per_stream"] == [1, 1, 1]
    assert (row["stood"], row["episodes"], row["stand_rate"]) == (3, 6, 0.5)
    assert row["ends"] == {"fallen": 2, "head_contact": 1}
    assert row["lift_offs_per_standing_episode"] == pytest.approx(2.0)
    assert row["standing_episodes_with_lift_off"] == 2
    assert row["forward_displacement_mean_m"] == pytest.approx(-0.1)
    assert row["mean_length"] == pytest.approx(7.0)
    assert row["std"] == pytest.approx(np.exp(-2.0)) and row["nonfinite_episodes"] == 0


def test_the_reference_rows_carry_their_std_the_mean_none_and_the_parents_own_its_range():
    by_stream = [[_episode(0, 3042, stood=True)]]
    mean = scan.summarize_setting(scan.MEAN, by_stream)
    assert (mean["label"], mean["kind"], mean["log_std"], mean["std"]) == ("mean", "mean", None, 0.0)
    own = scan.summarize_setting(scan.OWN, by_stream, parent_log_std=[-3.0, -2.0])
    assert (own["label"], own["kind"], own["log_std"], own["std"]) == ("own", "own", None, None)
    assert own["std_range"] == pytest.approx([np.exp(-3.0), np.exp(-2.0)])
    assert scan._std_text(own) == f"{np.exp(-3.0):.3f}-{np.exp(-2.0):.3f}"
    assert scan._std_text(mean) == "0.000"


def test_a_nonfinite_episode_is_counted_and_kept_out_of_the_means_and_the_json_writes_first(tmp_path):
    bad = {**_episode(0, 3043, stood=False, reason="nonfinite_state"), "return": None, "nonfinite": True}
    bad["forward_displacement_m"] = None
    row = scan.summarize_setting(-1.0, [[_episode(0, 3042, stood=True), bad]])
    assert row["nonfinite_episodes"] == 1 and row["mean_return"] == pytest.approx(1.0)
    json.dumps(row, allow_nan=False)
    # A record the JSON cannot hold leaves neither file: the JSON is written before the table.
    with pytest.raises(ValueError):
        scan.write_scan(tmp_path / "out", {"settings": [{"return": float("nan")}]}, "table\n")
    assert not (tmp_path / "out" / scan.SCAN_TEXT).exists()


def test_the_answer_is_the_largest_value_at_which_every_episode_stood():
    rows = [_row(-2.25, 120), _row(-2.0, 120), _row(-1.875, 116), _row(-1.75, 91)]
    answer = scan.largest_standing(rows[::-1])  # order does not matter
    assert answer["log_std"] == -2.0 and answer["below_first_failure_log_std"] == -2.0
    assert (answer["next_log_std"], answer["next_stood"]) == (-1.875, 116)
    assert (answer["first_failing_log_std"], answer["first_failing_stood"]) == (-1.875, 116)
    assert not answer["edge_above_grid"] and answer["failing_below"] == []
    line = scan.recommendation_line(
        answer,
        walk_config="configs/x/locomotion.toml",
        references={scan.MEAN: _reference(40, 40), scan.OWN: _reference(120, 120)},
    )
    assert "-2 (std 0.135, 120/120)" in line and "-1.875, stood 116/120" in line
    assert "mean action, without noise, stood 40/40, and its own learned log_std 120/120" in line
    assert "maintainer's decision" in line and "sets nothing" in line
    assert "shortened" not in line


def test_a_value_standing_above_a_failure_is_the_answer_and_the_value_below_the_first_failure_is_named():
    # docs/NEXT_STEPS.md's "largest value at which 120/120 stand" is -1.75; -1.875 below it fell by one episode.
    answer = scan.largest_standing([_row(-2.0, 120), _row(-1.875, 119), _row(-1.75, 120), _row(-1.5, 60)])
    assert answer["log_std"] == -1.75 and (answer["next_log_std"], answer["next_stood"]) == (-1.5, 60)
    assert answer["below_first_failure_log_std"] == -2.0 and answer["failing_below"] == [-1.875]
    line = scan.recommendation_line(answer, walk_config="c")
    assert "is -1.75" in line and "-1.875 below it did not stand every episode" in line
    assert "the largest value with every smaller grid value standing too is -2" in line


def test_the_edges_no_value_standing_and_every_value_standing():
    none = scan.largest_standing([_row(-2.0, 100), _row(-1.5, 9)])
    assert none["log_std"] is None and none["first_failing_log_std"] == -2.0
    assert none["below_first_failure_log_std"] is None
    assert "no grid value stood every episode" in scan.recommendation_line(none, walk_config="c")
    every = scan.largest_standing([_row(-2.0, 120), _row(-1.5, 120)])
    assert every["log_std"] == -1.5 and every["edge_above_grid"] and every["next_log_std"] is None
    assert "edge lies above the grid" in scan.recommendation_line(every, walk_config="c")
    lowest = scan.largest_standing([_row(-2.0, 119), _row(-1.5, 120)])
    assert lowest["log_std"] == -1.5 and lowest["below_first_failure_log_std"] is None
    assert "no grid value stood every episode with every smaller value" in scan.recommendation_line(
        lowest, walk_config="c"
    )


def test_when_no_grid_value_stands_the_line_says_whether_the_mean_action_stood():
    none = scan.largest_standing([_row(-2.0, 0, 2), _row(-1.0, 0, 2)])
    stood = scan.recommendation_line(
        none, walk_config="c", references={scan.MEAN: _reference(2, 2), scan.OWN: _reference(1, 2)}
    )
    assert "the edge lies below the grid (scan lower values with --log-std)" in stood
    fell = scan.recommendation_line(
        none, walk_config="c", references={scan.MEAN: _reference(0, 2), scan.OWN: _reference(0, 2)}
    )
    assert "falls in this env without noise too" in fell and "scan lower values" not in fell


def test_a_shortened_panel_is_named_in_the_line():
    assert scan._protocol_note(seeds=scan.DEFAULT_SEEDS, streams=3, horizon=1000, episode_steps=1000) is None
    note = scan._protocol_note(seeds=(3042, 3043), streams=1, horizon=30, episode_steps=1000)
    assert note is not None and "2 seeds x 1 stream to 30 of the walk's 1000 steps" in note
    every = scan.largest_standing([_row(-1.5, 2, 2)])
    assert note in scan.recommendation_line(every, walk_config="c", protocol=note)


def test_seeds_print_as_a_range_only_when_consecutive():
    assert scan.seeds_text(range(3042, 3082)) == "3042-3081"
    assert scan.seeds_text([3042, 3050]) == "3042, 3050"
    assert scan.seeds_text([3042]) == "3042"


def test_lift_offs_are_debounced_edges_after_the_spawn_grace():
    # One foot: a lift-off at step 1 (inside a 3-step grace), a one-step lift at step 5, a lift-off at step 8.
    foot = np.array([1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1], dtype=bool)
    down = np.stack([foot, np.ones_like(foot)], axis=1)
    assert scan.count_lift_offs(down, debounce_steps=1, grace_steps=0) == 3
    assert scan.count_lift_offs(down, debounce_steps=1, grace_steps=3) == 2
    # The 20 ms debounce at a 0.01 s step (2 steps) fills the one-step lift at step 5.
    assert scan.count_lift_offs(down, debounce_steps=2, grace_steps=3) == 1


def test_the_lift_off_definition_says_when_the_20_ms_debounce_is_one_step():
    assert "no debounce at all" in scan.lift_off_definition(debounce_steps=1, grace_steps=5, control_dt=0.02)
    two = scan.lift_off_definition(debounce_steps=2, grace_steps=10, control_dt=0.01)
    assert "2 control steps at 0.01 s" in two and "no debounce" not in two and "(10 steps)" in two


# ---------------------------------------------------------------------------------------------------- the stages


def _manifest(directory: Path, body: str) -> StageManifest:
    directory.mkdir(parents=True, exist_ok=True)
    for name in re.findall(r'^config = "([^"]+)"', body, flags=re.MULTILINE):
        (directory / name).write_text("", encoding="utf-8")
    (directory / "stages.toml").write_text(body, encoding="utf-8")
    return load_stage_manifest_file(directory / "stages.toml")


_HEAD = """schema = "mesozoic.stage-manifest/v2"
[[stages]]
id = "stance"
config = "stance.toml"
legacy_number = 1
recipe = "stance"
deliverable = true
[[stages]]
id = "recovery"
config = "recovery.toml"
warm_start_from = "stance"
recipe = "recover"
deliverable = true
"""
_NO_WALK = _HEAD
_TWO_WALKS = (
    _HEAD
    + """[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2
warm_start_from = "stance"
recipe = "walk"
deliverable = true
[[stages]]
id = "trot"
config = "trot.toml"
warm_start_from = "stance"
recipe = "walk"
deliverable = true
"""
)
_UNLABELLED_LOCOMOTION = (
    _HEAD
    + """[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2
warm_start_from = "stance"
deliverable = true
"""
)
_WALK_FROM_RECOVERY = (
    _HEAD
    + """[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2
warm_start_from = "recovery"
recipe = "walk"
deliverable = true
"""
)


@pytest.mark.parametrize(
    "model_id",
    [
        TYRANNOSAURUS_REX,
        ModelId("velociraptor_mongoliensis", "anatomical"),
        ModelId("compsognathus_longipes", "anatomical"),
        COMPSOGNATHUS_LONGIPES_ROBOT,
        ModelId("brachiosaurus_altithorax", "anatomical"),
        ModelId("dibothrosuchus_elaphros", "anatomical"),
        ModelId("allosaurus_fragilis", "anatomical"),
    ],
    ids=str,
)
def test_every_model_walks_from_its_stance_into_locomotion(model_id):
    manifest = load_stage_manifest(model_id)
    assert scan.resolve_walk_stage(manifest, "stance").id == "locomotion"
    parent, walk = scan.resolve_stages(manifest, parent=None, walk_stage=None)
    assert (parent.id, walk.id) == ("stance", "locomotion")


def test_a_model_with_no_walk_stage_is_refused(tmp_path):
    manifest = _manifest(tmp_path / "no_walk", _NO_WALK)
    with pytest.raises(scan.NoiseScanError, match="declares no walk stage that warm-starts from 'stance'"):
        scan.resolve_walk_stage(manifest, "stance")


def test_two_walk_children_are_refused_and_an_unlabelled_locomotion_child_is_the_walk(tmp_path):
    with pytest.raises(scan.NoiseScanError, match="more than one walk stage"):
        scan.resolve_walk_stage(_manifest(tmp_path / "two", _TWO_WALKS), "stance")
    assert scan.resolve_walk_stage(_manifest(tmp_path / "bare", _UNLABELLED_LOCOMOTION), "stance").id == "locomotion"


def test_a_named_stage_off_the_parents_edge_is_refused_with_the_way_to_scan_it():
    manifest = load_stage_manifest(TYRANNOSAURUS_REX)
    assert scan.resolve_walk_stage(manifest, "stance", "recovery").id == "recovery"
    with pytest.raises(scan.NoiseScanError, match="warm-starts from 'locomotion', not from 'stance'") as refused:
        scan.resolve_walk_stage(manifest, "stance", "behavior")
    assert "point --stage-dir at a 'locomotion' stage directory" in str(refused.value)
    assert "'locomotion'" in str(refused.value).split("they are")[1]


def test_a_model_pair_takes_its_parent_from_the_walk_stages_declared_edge(tmp_path):
    manifest = _manifest(tmp_path / "from_recovery", _WALK_FROM_RECOVERY)
    parent, walk = scan.resolve_stages(manifest, parent=None, walk_stage="locomotion")
    assert (parent.id, walk.id) == ("recovery", "locomotion")
    # Without --walk-stage the parent is the stance, whose children hold no walk here.
    with pytest.raises(scan.NoiseScanError, match="declares no walk stage"):
        scan.resolve_stages(manifest, parent=None, walk_stage=None)
    with pytest.raises(scan.NoiseScanError, match="'stance' is a root.*\\['recovery', 'locomotion'\\]"):
        scan.resolve_stages(manifest, parent=None, walk_stage="stance")


def test_the_command_line_refuses_a_model_with_no_walk_stage_before_loading_anything(tmp_path, monkeypatch):
    from environments.shared import stage_manifest

    configs = tmp_path / "configs"
    _manifest(TYRANNOSAURUS_REX.path(configs), _NO_WALK)
    monkeypatch.setattr(stage_manifest, "_CONFIGS_DIR", configs)
    with pytest.raises(SystemExit, match="declares no walk stage"):
        scan.main(
            [
                *TYRANNOSAURUS_REX_ARGS,
                "--model", str(tmp_path / "missing.zip"), "--vecnorm", str(tmp_path / "missing.pkl"),
                "--out-dir", str(tmp_path / "out"),
            ]
        )  # fmt: skip
    assert not (tmp_path / "out").exists()


def test_an_out_dir_inside_a_run_or_the_parents_stage_directory_is_refused_before_loading(tmp_path):
    run = tmp_path / "run"
    models = run / "01_stance" / "models"
    models.mkdir(parents=True)
    pair = ["--model", str(models / "robust_best_model.zip")]
    pair += ["--vecnorm", str(models / "robust_best_model_vecnorm.pkl")]
    with pytest.raises(SystemExit, match="inside the stage directory"):
        scan.main([*TYRANNOSAURUS_REX_ARGS, *pair, "--out-dir", str(run / "01_stance" / "scan")])
    (run / "provenance.json").write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="inside the run directory"):
        scan.main([*TYRANNOSAURUS_REX_ARGS, *pair, "--out-dir", str(run / "reports" / "scan")])
    assert not (run / "reports").exists() and not (run / "01_stance" / "scan").exists()


def test_an_out_dir_holding_an_earlier_scan_is_refused_before_loading(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / scan.SCAN_JSON).write_text("{}", encoding="utf-8")
    pair = ["--model", str(tmp_path / "missing.zip"), "--vecnorm", str(tmp_path / "missing.pkl")]
    with pytest.raises(SystemExit, match="already holds warm_start_noise_scan.json"):
        scan.main([*TYRANNOSAURUS_REX_ARGS, *pair, "--out-dir", str(out)])
    assert (out / scan.SCAN_JSON).read_text(encoding="utf-8") == "{}"


def test_a_stage_directory_without_a_handoff_pair_is_refused(tmp_path):
    (tmp_path / "01_stance" / "models").mkdir(parents=True)
    (tmp_path / "01_stance" / "models" / "robust_best_model.zip").write_bytes(b"")
    with pytest.raises(SystemExit, match="holds no complete handoff pair"):
        scan.main(
            [*TYRANNOSAURUS_REX_ARGS, "--stage-dir", str(tmp_path / "01_stance"), "--out-dir", str(tmp_path / "out")]
        )


def test_a_stage_directory_not_named_as_a_stage_is_refused_with_the_model_pair_alternative(tmp_path):
    models = tmp_path / "my_parent" / "models"
    models.mkdir(parents=True)
    (models / "robust_best_model.zip").write_bytes(b"")
    (models / "robust_best_model_vecnorm.pkl").write_bytes(b"")
    with pytest.raises(SystemExit, match="is not named as a stage directory") as refused:
        scan.main(
            [*TYRANNOSAURUS_REX_ARGS, "--stage-dir", str(tmp_path / "my_parent"), "--out-dir", str(tmp_path / "out")]
        )
    assert "--model and --vecnorm" in str(refused.value)


def test_a_missing_checkpoint_or_statistics_file_is_refused_by_name(tmp_path):
    present = tmp_path / "present_vecnorm.pkl"
    present.write_bytes(b"")
    out = ["--out-dir", str(tmp_path / "out")]
    with pytest.raises(SystemExit, match="no such file: .*nowhere.zip") as refused:
        scan.main(
            [*TYRANNOSAURUS_REX_ARGS, "--model", str(tmp_path / "nowhere"), "--vecnorm", str(present), *SHORT, *out]
        )
    assert "--stage-dir" in str(refused.value)
    model = tmp_path / "model.zip"
    model.write_bytes(b"")
    with pytest.raises(SystemExit, match="no such file: .*missing_vecnorm.pkl"):
        scan.main(
            [
                *TYRANNOSAURUS_REX_ARGS,
                "--model",
                str(model),
                "--vecnorm",
                str(tmp_path / "missing_vecnorm.pkl"),
                *SHORT,
                *out,
            ]
        )


def test_a_checkpoint_stem_resolves_to_its_zip(tmp_path):
    (tmp_path / "parent.zip").write_bytes(b"")
    (tmp_path / "parent_vecnorm.pkl").write_bytes(b"")
    assert scan.checkpoint_files(tmp_path / "parent", tmp_path / "parent_vecnorm.pkl") == (
        tmp_path / "parent.zip",
        tmp_path / "parent_vecnorm.pkl",
    )


def test_a_horizon_beyond_the_walks_episode_is_refused(tmp_path):
    pair = ["--model", str(tmp_path / "missing.zip"), "--vecnorm", str(tmp_path / "missing.pkl")]
    with pytest.raises(SystemExit, match="--horizon 1001 is longer than .*locomotion.toml's episode \\(1000 steps\\)"):
        scan.main([*TYRANNOSAURUS_REX_ARGS, *pair, "--horizon", "1001", "--out-dir", str(tmp_path / "out")])


def test_a_walk_stage_with_a_live_command_is_refused(tmp_path, monkeypatch):
    from environments.shared import config as config_module

    real = config_module.load_stage_config

    def live_command(model_id: Any, stage: Any, **kwargs: Any) -> dict[str, Any]:
        loaded = real(model_id, stage, **kwargs)
        return {**loaded, "env_kwargs": {**loaded["env_kwargs"], "command_mode": "heading"}}

    monkeypatch.setattr(config_module, "load_stage_config", live_command)
    pair = ["--model", str(tmp_path / "missing.zip"), "--vecnorm", str(tmp_path / "missing.pkl")]
    with pytest.raises(SystemExit, match="trains under command_mode 'heading'"):
        scan.main([*TYRANNOSAURUS_REX_ARGS, *pair, "--out-dir", str(tmp_path / "out")])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--model", "m.zip"], "--model needs --vecnorm"),
        (["--stage-dir", "01_stance", "--vecnorm", "v.pkl"], "--vecnorm goes with --model"),
        (["--stage-dir", "01_stance", "--seeds", "3042,x"], "--seeds must be comma-separated whole numbers"),
        (["--stage-dir", "01_stance", "--seeds", ","], "--seeds names no seed"),
        (["--stage-dir", "01_stance", "--seeds", "3042,3042"], "--seeds repeats a seed"),
        (["--stage-dir", "01_stance", "--streams", "0"], "--streams must be at least 1"),
        (["--stage-dir", "01_stance", "--horizon", "0"], "--horizon must be at least 1"),
    ],
    ids=[
        "model-without-vecnorm",
        "stage-dir-with-vecnorm",
        "seeds-word",
        "seeds-empty",
        "seeds-repeat",
        "streams",
        "horizon",
    ],
)
def test_the_arguments_are_checked_before_anything_else(argv, message, capsys):
    with pytest.raises(SystemExit) as refused:
        scan.main([*TYRANNOSAURUS_REX_ARGS, *argv])
    assert refused.value.code == 2 and message in capsys.readouterr().err


def test_a_policy_without_a_state_independent_log_std_is_refused():
    torch = pytest.importorskip("torch")

    gsde = SimpleNamespace(policy=SimpleNamespace(log_std=torch.nn.Parameter(torch.zeros(3, 2)), use_sde=True))
    with pytest.raises(scan.NoiseScanError, match="has none \\(gSDE\\)"):
        scan.policy_log_std(cast(Any, gsde))
    sac = SimpleNamespace(policy=SimpleNamespace(actor=object()))
    with pytest.raises(scan.NoiseScanError, match="SimpleNamespace has none"):
        scan.policy_log_std(cast(Any, sac))


# ---------------------------------------------------------------------------------------------------- tiny parents


class TinyParent(NamedTuple):
    stage_dir: Path
    model: Path
    vecnorm: Path
    log_std: np.ndarray


def tiny_parent(run_dir: Path, model_id: ModelId, *, critic_phase_mirror_kl: bool = False, stamp: bool = True):
    """A tiny real PPO stance parent, its handoff pair in ``<run_dir>/01_stance/models``, plant-stamped.

    Its ``log_std`` differs per actuator (from -3.0 to -2.2), so the parent's own row and the grid's reference
    are not the uniform values the grid writes.  *critic_phase_mirror_kl* builds it as the T. rex's certified
    stance was trained: a critic-phase policy and statistics (``CRITIC_PHASE = True``) under mirror-KL PPO, with
    mirror-symmetric ``log_std`` pairs.  ``stamp=False`` leaves out the plant identity (a pre-contract parent).
    """
    pytest.importorskip("mujoco")
    pytest.importorskip("stable_baselines3")
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.config import load_all_stages, load_stage_config
    from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
    from environments.shared.species_registry import env_class, get_species_config

    identity = current_plant_identity(model_id)
    kwargs: dict[str, Any] = {"n_steps": 16, "batch_size": 8, "seed": 7, "device": "cpu"}
    if critic_phase_mirror_kl:
        from environments.shared import critic_phase_ppo as CPP
        from environments.shared import mirror_kl_ppo as MK
        from environments.shared import symmetry as SY
        from environments.shared.train_base import create_vec_env

        species, configs = get_species_config(model_id), load_all_stages(model_id)
        normalizer = create_vec_env(
            species, configs, 1, 1, seed=3, algorithm="ppo", gamma=0.98, critic_phase_gamma=0.98
        )
        mirror_map = SY.MirrorMap.load(model_id, identity)
        model = PPO(env=normalizer, **CPP.new_model_kwargs(normalizer.critic_phase, {"net_arch": [16, 16]}), **kwargs)
        model = MK.enable_mirror_kl(model, SY.SymmetryConfig(), mirror_map)
    else:
        env_kwargs = load_stage_config(model_id, "stance")["env_kwargs"]
        cls = env_class(model_id)
        normalizer = VecNormalize(DummyVecEnv([lambda: cls(**env_kwargs)]))
        model = PPO("MlpPolicy", normalizer, policy_kwargs={"net_arch": [16, 16]}, **kwargs)
    models = run_dir / "01_stance" / "models"
    models.mkdir(parents=True)
    try:
        nu = int(cast(Any, model.action_space).shape[0])
        normalizer.reset()
        for _ in range(20):
            normalizer.step(np.zeros((1, nu), dtype=np.float32))
        log_std = np.linspace(-3.0, -2.2, nu)
        if critic_phase_mirror_kl:
            perm = mirror_map.action.perm
            log_std = (log_std + log_std[perm]) / 2.0
        with torch.no_grad():
            model.policy.log_std.copy_(torch.as_tensor(log_std, dtype=torch.float32))
        if stamp:
            for artifact in (model, normalizer):
                attach_plant_identity(artifact, identity)
        model.save(models / "robust_best_model.zip")
        normalizer.save(str(models / "robust_best_model_vecnorm.pkl"))
    finally:
        normalizer.close()
    learned = np.asarray(np.float32(log_std), dtype=np.float64)
    return TinyParent(
        run_dir / "01_stance", models / "robust_best_model.zip", models / "robust_best_model_vecnorm.pkl", learned
    )


@pytest.fixture(scope="module")
def tyrannosaurus_rex_parent(tmp_path_factory):
    return tiny_parent(tmp_path_factory.mktemp("tyrannosaurus-rex-run"), TYRANNOSAURUS_REX)


@pytest.fixture(scope="module")
def compsognathus_longipes_robot_parent(tmp_path_factory):
    return tiny_parent(tmp_path_factory.mktemp("compsognathus-longipes-robot-run"), COMPSOGNATHUS_LONGIPES_ROBOT)


@pytest.fixture(scope="module")
def tyrannosaurus_rex_critic_phase_parent(tmp_path_factory):
    run = tmp_path_factory.mktemp("tyrannosaurus-rex-critic-phase-run")
    return tiny_parent(run, TYRANNOSAURUS_REX, critic_phase_mirror_kl=True)


def _spec(parent: TinyParent, model_id: ModelId, *, horizon: int, seeds=(3042,)) -> Any:
    from environments.shared.config import load_stage_config

    config = load_stage_config(model_id, "locomotion")
    return scan.ScanSpec(
        model_id=model_id,
        model_path=str(parent.model),
        vecnorm_path=str(parent.vecnorm),
        walk_stage="locomotion",
        reward_weights=scan.warm_start_reward_weights(config, parent_id="stance", load_path=str(parent.model)),
        horizon=horizon,
        seeds=tuple(seeds),
    )


# ---------------------------------------------------------------------------------------------------- the sampling


@pytest.mark.parametrize("setting", [-1.0, scan.OWN, scan.MEAN], ids=["grid value", "own", "mean"])
def test_actions_are_sampled_from_the_policys_distribution_at_the_set_log_std(
    tyrannosaurus_rex_parent, monkeypatch, setting
):
    import torch

    context = scan.open_context(_spec(tyrannosaurus_rex_parent, TYRANNOSAURUS_REX, horizon=6))
    try:
        log_std = scan.policy_log_std(context.policy.model)
        before = log_std.detach().clone()
        read: list[tuple[np.ndarray, np.ndarray]] = []
        sent: list[np.ndarray] = []
        real_distribution = scan.action_mean_and_std

        def spy_distribution(model: Any, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            mean, std = real_distribution(model, observation)
            read.append((mean, std))
            return mean, std

        env = context.env
        real_step = env.step

        def spy_step(action: np.ndarray) -> Any:
            sent.append(np.asarray(action, dtype=np.float64).copy())
            return real_step(action)

        monkeypatch.setattr(scan, "action_mean_and_std", spy_distribution)
        monkeypatch.setattr(env, "step", spy_step)
        (episode,) = scan.roll_setting(context, setting, stream=1)
        learned = np.exp(tyrannosaurus_rex_parent.log_std)
        expected_std = learned if isinstance(setting, str) else np.exp(np.float32(setting))
        rng = scan.noise_rng(3042, 1)
        low, high = env.action_space.low, env.action_space.high
        assert len(sent) == len(read) == episode["length"]
        for (mean, std), action in zip(read, sent):
            assert std == pytest.approx(expected_std, rel=1e-6)
            noise = 0.0 if setting == scan.MEAN else std * rng.standard_normal(mean.shape[0])
            assert np.array_equal(action, np.clip(mean + noise, low, high))
        # The parameter is the loaded one again, bit for bit.
        assert torch.equal(log_std, before)
        assert np.allclose(before.double().numpy(), tyrannosaurus_rex_parent.log_std)
    finally:
        context.close()


def test_the_log_std_is_restored_when_an_episode_fails(tyrannosaurus_rex_parent, monkeypatch):
    import torch

    context = scan.open_context(_spec(tyrannosaurus_rex_parent, TYRANNOSAURUS_REX, horizon=6))
    try:
        before = scan.policy_log_std(context.policy.model).detach().clone()

        def failing_step(action: np.ndarray) -> Any:
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(context.env, "step", failing_step)
        with pytest.raises(RuntimeError, match="simulated failure"):
            scan.roll_setting(context, -0.5, stream=0)
        assert torch.equal(scan.policy_log_std(context.policy.model), before)
    finally:
        context.close()


def test_a_nonfinite_reward_is_recorded_as_none_not_nan(tyrannosaurus_rex_parent, monkeypatch):
    context = scan.open_context(_spec(tyrannosaurus_rex_parent, TYRANNOSAURUS_REX, horizon=3))
    try:
        real_step = context.env.step

        def nan_step(action: np.ndarray) -> Any:
            obs, _reward, terminated, truncated, info = real_step(action)
            return obs, float("nan"), terminated, truncated, info

        monkeypatch.setattr(context.env, "step", nan_step)
        (episode,) = scan.roll_setting(context, -1.0, stream=0)
    finally:
        context.close()
    assert episode["return"] is None and episode["nonfinite"]
    json.dumps(episode, allow_nan=False)


def test_a_different_stream_draws_different_noise_and_the_same_stream_repeats():
    first = scan.noise_rng(3042, 0).standard_normal(4)
    assert np.array_equal(first, scan.noise_rng(3042, 0).standard_normal(4))
    assert not np.array_equal(first, scan.noise_rng(3042, 1).standard_normal(4))
    assert not np.array_equal(first, scan.noise_rng(3043, 0).standard_normal(4))


# ---------------------------------------------------------------------------------------------------- smoke runs


def _smoke(parent: TinyParent, model_id: ModelId, out_dir: Path, *extra: str) -> dict[str, Any]:
    before = (_sha256(parent.model), _sha256(parent.vecnorm))
    code = scan.main(
        [
            "--species", model_id.species, "--embodiment", model_id.embodiment,
            "--stage-dir", str(parent.stage_dir), "--seeds", SMOKE_SEEDS, "--streams", "1",
            "--horizon", str(SMOKE_HORIZON), "--log-std", "-2.0", "-1.0", "--out-dir", str(out_dir), *extra,
        ]
    )  # fmt: skip
    assert code == 0
    assert (_sha256(parent.model), _sha256(parent.vecnorm)) == before
    return cast(dict[str, Any], json.loads((out_dir / scan.SCAN_JSON).read_text(encoding="utf-8")))


@pytest.mark.parametrize(
    ("fixture", "model_id", "debounce_steps", "policy_class"),
    [
        ("tyrannosaurus_rex_parent", TYRANNOSAURUS_REX, 2, "ActorCriticPolicy"),
        ("compsognathus_longipes_robot_parent", COMPSOGNATHUS_LONGIPES_ROBOT, 1, "ActorCriticPolicy"),
        ("tyrannosaurus_rex_critic_phase_parent", TYRANNOSAURUS_REX, 2, "CriticPhaseActorCriticPolicy"),
    ],
    ids=["tyrannosaurus_rex-anatomical", "compsognathus_longipes-robot", "tyrannosaurus_rex-critic-phase-mirror-kl"],
)
def test_a_smoke_scan_writes_the_table_and_the_json(
    fixture, model_id, debounce_steps, policy_class, request, tmp_path, capsys
):
    parent = request.getfixturevalue(fixture)
    report = _smoke(parent, model_id, tmp_path / "out")
    printed = capsys.readouterr().out
    assert report["schema"] == scan.SCAN_SCHEMA
    assert (report["species"], report["embodiment"]) == (model_id.species, model_id.embodiment)
    assert (report["parent_stage"], report["walk_stage"], report["handoff"]) == (
        "stance",
        "locomotion",
        "robust_best_model",
    )
    assert report["walk_config"] == f"configs/{model_id.species}/{model_id.embodiment}/locomotion.toml"
    assert report["declared_warm_start_log_std"] is None
    assert report["checkpoint_plant_validated"] is True
    assert report["policy"]["policy_class"] == policy_class
    assert report["horizon"] == SMOKE_HORIZON and report["seeds"] == [3042, 3043] and report["streams"] == 1
    assert report["episode_steps"] == 1000 and report["full_protocol"] is False
    assert report["lift_off_debounce_steps"] == debounce_steps
    assert f"{debounce_steps} control step" in report["definitions"]["lift_offs"]
    assert report["parent"]["log_std"] == pytest.approx(parent.log_std.tolist())
    assert "forward_vel_weight" in report["reward_weights"]
    assert report["definitions"]["stands"] == scan.STANDS_DEFINITION
    assert [row["label"] for row in report["settings"]] == ["mean", "own", "-2", "-1"]
    assert [row["kind"] for row in report["settings"]] == ["mean", "own", "grid", "grid"]
    for row in report["settings"]:
        records = row["episode_records"]
        assert [record["seed"] for record in records] == [3042, 3043]
        assert row["stood"] == sum(record["stood"] for record in records) == row["stood_per_stream"][0]
        for record in records:
            assert record["stood"] == (record["length"] == SMOKE_HORIZON and record["termination_reason"] is None)
    assert report["largest_all_standing"] == scan.largest_standing(report["settings"][2:])
    text = (tmp_path / "out" / scan.SCAN_TEXT).read_text(encoding="utf-8")
    assert text in printed and report["recommendation"] in text
    assert "maintainer's decision" in report["recommendation"]
    assert "a shortened panel: 2 seeds x 1 stream to 12 of the walk's 1000 steps" in report["recommendation"]
    assert "mean action, without noise, stood" in report["recommendation"]


def test_a_model_pair_given_as_a_stem_with_its_walk_stage_scans_from_the_declared_parent(
    tyrannosaurus_rex_parent, tmp_path
):
    stem = str(tyrannosaurus_rex_parent.model)[: -len(".zip")]
    code = scan.main(
        [
            *TYRANNOSAURUS_REX_ARGS, "--model", stem, "--vecnorm", str(tyrannosaurus_rex_parent.vecnorm), "--walk-stage", "locomotion",
            *SHORT, "--out-dir", str(tmp_path / "out"),
        ]
    )  # fmt: skip
    report = json.loads((tmp_path / "out" / scan.SCAN_JSON).read_text(encoding="utf-8"))
    assert code == 0 and (report["parent_stage"], report["walk_stage"], report["handoff"]) == (
        "stance",
        "locomotion",
        None,
    )
    assert report["checkpoint"] == str(tyrannosaurus_rex_parent.model)


def test_the_scan_is_the_same_across_processes(tyrannosaurus_rex_parent, tmp_path):
    serial = _smoke(tyrannosaurus_rex_parent, TYRANNOSAURUS_REX, tmp_path / "serial")
    parallel = _smoke(tyrannosaurus_rex_parent, TYRANNOSAURUS_REX, tmp_path / "parallel", "--jobs", "2")
    assert parallel["settings"] == serial["settings"]
    # A finished scan leaves no progress file, in either mode.
    assert not (tmp_path / "serial" / scan.SCAN_PROGRESS).exists()
    assert not (tmp_path / "parallel" / scan.SCAN_PROGRESS).exists()


def _fake_rolls(monkeypatch, stands, *, interrupt_at: float | None = None) -> list[Any]:
    """Replace the rollouts with records that stand when ``stands(setting)``; returns the settings rolled, per stream."""
    rolled: list[Any] = []

    def roll_setting(context, setting, stream):
        if interrupt_at is not None and setting == interrupt_at:
            raise KeyboardInterrupt
        rolled.append(setting)
        stood = bool(stands(setting))
        return [_episode(stream, seed, stood=stood, reason=None if stood else "fallen") for seed in context.seeds]

    monkeypatch.setattr(scan, "roll_setting", roll_setting)
    return rolled


def _default_grid_argv(parent: TinyParent, out_dir: Path) -> list[str]:
    return [
        *TYRANNOSAURUS_REX_ARGS, "--stage-dir", str(parent.stage_dir), "--seeds", SMOKE_SEEDS, "--streams", "1",
        "--horizon", "2", "--out-dir", str(out_dir),
    ]  # fmt: skip


def test_the_default_grid_rolls_a_first_pass_then_one_refinement_above_its_answer(
    tyrannosaurus_rex_parent, tmp_path, monkeypatch
):
    first = scan.default_grid(tyrannosaurus_rex_parent.log_std)
    assert -2.0 in first and -1.75 in first
    rolled = _fake_rolls(monkeypatch, lambda setting: isinstance(setting, str) or setting <= -2.0)
    out = tmp_path / "out"
    assert scan.main(_default_grid_argv(tyrannosaurus_rex_parent, out)) == 0
    report = json.loads((out / scan.SCAN_JSON).read_text(encoding="utf-8"))
    # Pilot 3's pattern: -2.0 stood and -1.75 fell, so the second pass rolls -1.875 alone.
    assert rolled == [scan.MEAN, scan.OWN, *first, -1.875]
    assert report["grid"] == sorted([*first, -1.875])
    assert [row["label"] for row in report["settings"]] == [
        "mean",
        "own",
        *[scan.setting_label(value) for value in sorted([*first, -1.875])],
    ]
    answer = report["largest_all_standing"]
    assert (answer["log_std"], answer["next_log_std"], answer["next_stood"]) == (-2.0, -1.875, 0)
    assert "then -1.875, 0.125 above -2, the largest of those at which every episode stood" in report["grid_rule"]
    assert report["complete"] is True and "pending" not in report
    assert not (out / scan.SCAN_PROGRESS).exists()


def test_the_default_grid_adds_no_refinement_when_the_edge_lies_above_it(
    tyrannosaurus_rex_parent, tmp_path, monkeypatch
):
    first = scan.default_grid(tyrannosaurus_rex_parent.log_std)
    rolled = _fake_rolls(monkeypatch, lambda setting: True)
    out = tmp_path / "out"
    assert scan.main(_default_grid_argv(tyrannosaurus_rex_parent, out)) == 0
    report = json.loads((out / scan.SCAN_JSON).read_text(encoding="utf-8"))
    assert rolled == [scan.MEAN, scan.OWN, *first] and report["grid"] == list(first)
    assert "no refinement: the largest value stood every episode" in report["grid_rule"]
    assert report["largest_all_standing"]["edge_above_grid"] is True


def test_an_explicit_grid_is_rolled_as_given_without_a_refinement(tyrannosaurus_rex_parent, tmp_path, monkeypatch):
    rolled = _fake_rolls(monkeypatch, lambda setting: isinstance(setting, str) or setting <= -2.0)
    argv = _default_grid_argv(tyrannosaurus_rex_parent, tmp_path / "out")
    assert scan.main([*argv, "--log-std", "-2.0", "-1.5"]) == 0
    assert rolled == [scan.MEAN, scan.OWN, -2.0, -1.5]


def test_an_interrupted_scan_keeps_the_settings_it_finished_and_its_out_dir_is_then_refused(
    tyrannosaurus_rex_parent, tmp_path, monkeypatch
):
    first = scan.default_grid(tyrannosaurus_rex_parent.log_std)
    _fake_rolls(monkeypatch, lambda setting: True, interrupt_at=first[2])
    out = tmp_path / "out"
    with pytest.raises(KeyboardInterrupt):
        scan.main(_default_grid_argv(tyrannosaurus_rex_parent, out))
    progress = json.loads((out / scan.SCAN_PROGRESS).read_text(encoding="utf-8"))
    assert progress["schema"] == scan.SCAN_SCHEMA and progress["complete"] is False
    assert [row["label"] for row in progress["settings"]] == [
        "mean",
        "own",
        *[scan.setting_label(value) for value in first[:2]],
    ]
    assert progress["pending"] == [scan.setting_label(value) for value in first[2:]]
    assert "largest_all_standing" not in progress
    assert not (out / scan.SCAN_JSON).exists() and not (out / scan.SCAN_TEXT).exists()
    # The interrupted rows are never replaced by a second scan into the same directory.
    with pytest.raises(SystemExit, match="holds the rows an interrupted scan finished"):
        scan.main(_default_grid_argv(tyrannosaurus_rex_parent, out))
    assert json.loads((out / scan.SCAN_PROGRESS).read_text(encoding="utf-8")) == progress


def test_a_parent_from_another_plant_is_refused(compsognathus_longipes_robot_parent, tmp_path):
    with pytest.raises(SystemExit) as refused:
        scan.main(
            [
                "--species", "compsognathus_longipes", "--embodiment", "anatomical",
                "--stage-dir", str(compsognathus_longipes_robot_parent.stage_dir), *SHORT,
                "--out-dir", str(tmp_path / "out"),
            ]
        )  # fmt: skip
    assert "plant" in str(refused.value).lower()
    assert not (tmp_path / "out").exists()


def test_a_pre_contract_parent_is_refused_unless_the_legacy_plant_is_allowed(tmp_path):
    parent = tiny_parent(tmp_path / "legacy-run", TYRANNOSAURUS_REX, stamp=False)
    argv = [*TYRANNOSAURUS_REX_ARGS, "--stage-dir", str(parent.stage_dir), *SHORT]
    with pytest.raises(SystemExit) as refused:
        scan.main([*argv, "--out-dir", str(tmp_path / "refused")])
    assert "plant" in str(refused.value).lower() and not (tmp_path / "refused").exists()
    assert scan.main([*argv, "--allow-legacy-plant", "--out-dir", str(tmp_path / "out")]) == 0
    report = json.loads((tmp_path / "out" / scan.SCAN_JSON).read_text(encoding="utf-8"))
    assert report["checkpoint_plant_validated"] is False


def test_a_parent_recording_another_stage_is_refused_as_training_refuses_it(
    tyrannosaurus_rex_parent, tmp_path, monkeypatch
):
    from environments.shared import task_fingerprint

    recorded = {**TYRANNOSAURUS_REX.record(), "stage": 2}
    monkeypatch.setattr(task_fingerprint, "read_checkpoint_task_fingerprint", lambda path: recorded)
    with pytest.raises(SystemExit, match="must come from the node's declared parent"):
        scan.main(
            [
                *TYRANNOSAURUS_REX_ARGS,
                "--stage-dir",
                str(tyrannosaurus_rex_parent.stage_dir),
                *SHORT,
                "--out-dir",
                str(tmp_path / "out"),
            ]
        )


def test_a_pair_that_changes_on_disk_during_the_scan_is_refused(tyrannosaurus_rex_parent, tmp_path, monkeypatch):
    run = tmp_path / "copied-run"
    shutil.copytree(tyrannosaurus_rex_parent.stage_dir, run / "01_stance")
    vecnorm = run / "01_stance" / "models" / "robust_best_model_vecnorm.pkl"
    real = scan.run_scan

    def touching_run_scan(*args: Any, **kwargs: Any) -> Any:
        results = real(*args, **kwargs)
        with vecnorm.open("ab") as handle:
            handle.write(b"\0")
        return results

    monkeypatch.setattr(scan, "run_scan", touching_run_scan)
    with pytest.raises(SystemExit, match="the checkpoint pair changed on disk during the scan"):
        scan.main(
            [*TYRANNOSAURUS_REX_ARGS, "--stage-dir", str(run / "01_stance"), *SHORT, "--out-dir", str(tmp_path / "out")]
        )
    # The progress file the scan wrote is removed with the refusal: those rows measure no one checkpoint.
    assert not any((tmp_path / "out").iterdir())
