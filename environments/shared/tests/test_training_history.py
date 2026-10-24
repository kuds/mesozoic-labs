"""The training-history report (``reporting/training_history.py``, ``scripts/plot_training_history.py``).

Small synthetic ledgers only: the reductions (counts, the rails' over-bar
counts, the band counts, per-foot means, left-minus-right offsets, training
windows), the saturation flag and alarm rule, tolerance of missing files and
columns, the refusal to write inside a run directory, the per-actuator std
reader on a tiny checkpoint zip and the plant check behind its names, the
checkpoint table, and that every page renders on the Agg canvas with its bars,
band edges, marks and count strips.
"""

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from environments.shared.reporting import training_history as th
from environments.shared.scripts import plot_training_history as script

pytest.importorskip("matplotlib")

FEET = ("r", "l")
STEP = 2048
#: The columns a T. rex row carries that the report reads (a subset of the real 63).
COLUMNS = [
    "evaluation",
    "timesteps",
    "episode",
    "env",
    "return",
    "length",
    "full_horizon",
    "cause",
    "screen_v2_clean",
    "screen_v2_reasons",
    *[f"{foot}_{suffix}" for foot in FEET for suffix in th.FOOT_SUFFIXES],
    *th.BODY_COLUMNS,
    "leg_q_l_minus_r_hip_pitch_deg",
    "leg_q_l_r_mean_hip_pitch_deg",
    "leg_q_l_minus_r_knee_deg",
    "leg_q_l_minus_r_ankle_deg",
]


def _episode(evaluation: int, episode: int, *, fell: bool = False, saturated: bool = False, **values: Any) -> dict:
    row: dict[str, Any] = {column: 0.0 for column in COLUMNS}
    reasons = (["horizon"] if fell else []) + (["max_actuator_saturation_fraction: 0.31 > 0.1"] if saturated else [])
    row.update(
        evaluation=evaluation,
        timesteps=(evaluation + 1) * STEP,
        episode=episode,
        env=0,
        length=400 if fell else 1000,
        full_horizon=0 if fell else 1,
        cause="nosedive" if fell else "truncated",
        screen_v2_clean=0 if (fell or saturated) else 1,
        screen_v2_reasons=";".join(reasons),
    )
    row["return"] = 100.0 + episode
    row.update(values)
    return row


def _write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    columns = columns or list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: ("nan" if isinstance(v, float) and math.isnan(v) else v) for key, v in row.items()})


def _stage(tmp_path: Path, *, saturated: list[int] | None = None, name: str = "run_a") -> Path:
    """A stage directory of 10-episode evaluations; evaluation k has ``saturated[k]`` saturation-flagged episodes."""
    saturated = saturated if saturated is not None else [0, 6, 7, 0]
    stage = tmp_path / "runs" / name / "01_stance"
    rows = []
    for evaluation, count in enumerate(saturated):
        for episode in range(10):
            rows.append(
                _episode(
                    evaluation,
                    episode,
                    fell=episode == 9 and evaluation == 0,
                    saturated=episode < count,
                    r_foot_cop_fore_aft=0.1 * evaluation + (0.02 if episode % 2 else -0.02),
                    l_foot_cop_fore_aft=-0.05,
                    r_sole_cop_fore_aft=0.3,
                    l_sole_cop_fore_aft=0.25,
                    leg_q_l_minus_r_ankle_deg=float(episode),
                    max_sole_tilt_deg=0.5 + 0.1 * episode,
                )
            )
    _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows, COLUMNS)
    train = [
        {"timesteps": step, "env": 0, "length": 1000, "return": 1.0, "cause": cause}
        for step, cause in [
            (100, "nosedive"),
            (2048, "truncated"),
            (3000, "truncated"),
            (4096, "fallen"),
            (9000, "truncated"),
        ]
    ]
    _write_csv(stage / th.TRAIN_EPISODES_FILENAME, train)
    config = {
        "species": "tyrannosaurus_rex",
        "embodiment": "anatomical",
        "run": {"seed": 42},
        "curriculum": {
            "max_sole_cop_fore_aft": 0.8,
            "max_sole_tilt_deg": 2.0,
            "max_sole_corner_lift_m": 0.004,
            "max_actuator_saturation_fraction": 0.1,
        },
        "reward_weights": {
            "stance_cop_band_weight": 0.3,
            "stance_cop_band_lower": -0.6,
            "stance_cop_band_upper": 0.4,
            "stance_cop_band_pad_lower": -0.65,
            "stance_cop_band_pad_upper": 0.65,
        },
    }
    (stage / th.STAGE_CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")
    return stage


class TestReductions:
    def test_counts_per_evaluation(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        assert run.steps.tolist() == [STEP, 2 * STEP, 3 * STEP, 4 * STEP]
        assert run.series("episodes").tolist() == [10, 10, 10, 10]
        assert run.series("falls").tolist() == [1, 0, 0, 0]
        # Clean: not fallen and not saturated.
        assert run.series("clean").tolist() == [9, 4, 3, 10]
        assert run.series("saturated").tolist() == [0, 6, 7, 0]
        assert run.series("mean_return").tolist() == pytest.approx([104.5] * 4)

    def test_per_foot_means_and_spread_over_full_horizon_episodes(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        assert run.feet == ["r", "l"]
        # Evaluation 0 drops its fallen episode 9 (odd: +0.02), leaving five -0.02 and four +0.02.
        mean0 = (5 * -0.02 + 4 * 0.02) / 9
        assert run.column("r_foot_cop_fore_aft")[0] == pytest.approx(mean0)
        assert run.column("r_foot_cop_fore_aft")[1:].tolist() == pytest.approx([0.1, 0.2, 0.3])
        assert run.column("r_foot_cop_fore_aft", "std")[1] == pytest.approx(0.02)
        assert run.column("l_foot_cop_fore_aft").tolist() == pytest.approx([-0.05] * 4)
        assert run.column("r_sole_cop_fore_aft", "n")[0] == 9
        assert run.column("max_sole_tilt_deg", "max")[0] == pytest.approx(0.5 + 0.1 * 8)
        assert run.column("max_sole_tilt_deg", "max")[1] == pytest.approx(0.5 + 0.1 * 9)

    def test_left_minus_right_differences(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        assert run.lr == {
            "hip_pitch": "leg_q_l_minus_r_hip_pitch_deg",
            "knee": "leg_q_l_minus_r_knee_deg",
            "ankle": "leg_q_l_minus_r_ankle_deg",
        }
        assert run.column("leg_q_l_minus_r_ankle_deg")[1] == pytest.approx(4.5)
        assert run.column("leg_q_l_minus_r_ankle_deg", "std")[1] == pytest.approx(np.std(np.arange(10.0)))
        assert run.column("leg_q_l_minus_r_ankle_deg")[0] == pytest.approx(4.0)  # episode 9 fell
        # The mirror-even mean is not a left-minus-right column, and hip roll is not recorded at all.
        assert "leg_q_l_r_mean_hip_pitch_deg" not in run.evaluations["columns"]
        assert not any("hip_roll" in column for column in run.evaluations["columns"])
        assert any("Hip roll" in note for note in th.NOT_RECORDED)

    def test_training_falls_per_window(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        training = run.training
        assert training is not None and training["window_timesteps"] == th.TRAINING_WINDOW_TIMESTEPS
        header = ["timesteps", "cause"]
        rows = [{"timesteps": str(s), "cause": c} for s, c in [(10, "nosedive"), (10, "truncated"), (25, "fallen")]]
        windows = th.reduce_training(header, rows, window=10)
        # The last window ends at its last episode (25), not at its nominal edge (30).
        assert windows["window_end"].tolist() == [10, 20, 25] and windows["last_episode_step"] == 25
        assert windows["episodes"].tolist() == [2, 0, 1]
        assert windows["falls"].tolist() == [1, 0, 1]
        assert windows["fall_fraction"][0] == 0.5 and math.isnan(windows["fall_fraction"][1])

    def test_the_last_training_window_does_not_stretch_the_run(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        # train_episodes.csv ends at step 9,000, inside the first 262,144-step window.
        assert run.training["window_end"].tolist() == [9000] and run.last_step == 9000

    def test_full_horizon_falls_back_to_cause(self, tmp_path):
        stage = tmp_path / "cause_only" / "01_stance"
        rows = [
            {"timesteps": STEP, "cause": cause, "r_foot_cop_fore_aft": value, "l_foot_cop_fore_aft": 0.0}
            for cause, value in [("truncated", 0.2), ("truncated", 0.4), ("nosedive", 0.9)]
        ]
        _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows)
        run = th.load_run_history(stage, read_std=False)
        assert run.series("falls").tolist() == [1] and run.series("full_horizon").tolist() == [2]
        assert run.column("r_foot_cop_fore_aft").tolist() == pytest.approx([0.3])  # the fall left out

    def test_clean_and_saturated_count_the_judged_episodes_only(self, tmp_path):
        stage = tmp_path / "judged" / "01_stance"
        rows = [
            _episode(0, 0),
            _episode(0, 1, saturated=True),
            {**_episode(0, 2, saturated=True), "screen_v2_clean": math.nan},  # not judged: neither clean nor flagged
        ]
        _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows, COLUMNS)
        run = th.load_run_history(stage, read_std=False)
        assert run.series("episodes").tolist() == [3]
        assert run.series("clean").tolist() == [1] and run.series("saturated").tolist() == [1]

    def test_reason_keys_take_the_name_before_a_colon(self):
        assert th.reason_keys("horizon; max_actuator_saturation_fraction: 0.3 > 0.1;") == {
            "horizon",
            "max_actuator_saturation_fraction",
        }
        assert th.reason_keys("") == set() and th.reason_keys(None) == set()

    def test_bars_and_band_from_stage_config(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        assert run.bars["max_sole_cop_fore_aft"] == 0.8 and run.bars["max_actuator_saturation_fraction"] == 0.1
        assert run.bars["max_foot_cop_fore_aft"] is None
        assert run.band_active and run.band["stance_cop_band_upper"] == 0.4
        assert run.model_id is not None and run.seed == 42 and run.label == "run_a"


class TestRailsAndBand:
    def _ledger(self, tmp_path: Path, name: str = "rail", band: bool = True) -> Path:
        """One evaluation of 10: every signed pad mean near 0, three episodes over the 0.8 pad rail (one fell)."""
        stage = _stage(tmp_path, saturated=[0], name=name)
        rows = []
        for episode in range(10):
            sign = 1.0 if episode % 2 else -1.0
            rail = {0: 0.95, 1: 0.9, 2: 0.85, 3: 0.81}.get(episode, 0.3)
            rows.append(
                _episode(
                    0,
                    episode,
                    fell=episode == 3,
                    r_sole_cop_fore_aft=sign * 0.5,
                    l_sole_cop_fore_aft=-sign * 0.5,
                    max_sole_cop_fore_aft=rail,
                    r_foot_cop_fore_aft=0.6 if episode == 4 else (-0.7 if episode == 5 else 0.0),
                    l_foot_cop_fore_aft=math.nan if episode == 6 else 0.1,
                    max_sole_tilt_deg=2.5 if episode == 7 else 1.0,
                    max_sole_corner_lift_m=0.005 if episode in (7, 8) else 0.001,
                )
            )
        _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows, COLUMNS)
        if not band:
            config = json.loads((stage / th.STAGE_CONFIG_FILENAME).read_text())
            del config["reward_weights"]
            (stage / th.STAGE_CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")
        return stage

    def test_over_bar_counts_read_each_episode_not_the_signed_mean(self, tmp_path):
        run = th.load_run_history(self._ledger(tmp_path), read_std=False)
        # The signed per-foot means cancel to about 0 ...
        assert abs(run.column("r_sole_cop_fore_aft")[0]) < 0.12 and abs(run.column("l_sole_cop_fore_aft")[0]) < 0.12
        # ... while three full-horizon episodes fail the 0.8 rail (episode 3 also does, but fell).
        assert run.series("max_sole_cop_fore_aft_over_bar").tolist() == [3]
        assert run.series("max_sole_cop_fore_aft_over_bar_fraction")[0] == pytest.approx(3 / 9)
        assert run.column("max_sole_cop_fore_aft", "max")[0] == pytest.approx(0.95)
        assert run.series("max_sole_tilt_deg_over_bar").tolist() == [1]
        assert run.series("max_sole_corner_lift_m_over_bar").tolist() == [2]
        # No stage declares max_foot_cop_fore_aft's bar, so it has no count.
        assert run.series("max_foot_cop_fore_aft_over_bar") is None and run.bars["max_foot_cop_fore_aft"] is None
        row = th.checkpoint_rows(run, [STEP])[0]
        assert row["max_sole_cop_fore_aft_over_bar"] == 3 and row["max_sole_cop_fore_aft_worst"] == pytest.approx(0.95)

    def test_band_counts_use_the_run_s_own_edges(self, tmp_path):
        run = th.load_run_history(self._ledger(tmp_path), read_std=False)
        # Episode 4 (+0.6 > +0.4) and 5 (-0.7 < -0.6) leave [-0.6, +0.4]; episode 6's unloaded left foot is not outside.
        assert run.series("foot_cop_outside_band").tolist() == [2]
        assert run.series("sole_cop_outside_pad_band").tolist() == [0]
        assert run.band_edges == {"lower": -0.6, "upper": 0.4, "pad_lower": -0.65, "pad_upper": 0.65}
        assert "own band" in run.band_edges_source

    def test_a_run_without_a_band_is_counted_against_the_reference(self, tmp_path):
        banded = th.load_run_history(self._ledger(tmp_path, "a"), read_std=False)
        control = th.load_run_history(self._ledger(tmp_path, "b", band=False), read_std=False)
        assert control.series("foot_cop_outside_band") is None and not control.band_edges
        assert th.reference_band([banded, control]) == banded.band_edges
        th.apply_reference_band([banded, control])
        assert control.series("foot_cop_outside_band").tolist() == [2]
        assert "reference" in control.band_edges_source and "own band" in banded.band_edges_source
        assert not control.band_active


class TestSaturationAlarm:
    def test_flags_and_alarms(self):
        steps = list(range(1, 10))
        result = th.saturation_alarms(steps, [0, 3, 5, 4, 5, 6, math.nan, 5, 5])
        assert result["first_saturated_episode_at"] == 2
        assert result["flagged"] == [3, 5, 6, 8, 9]
        assert result["first_flagged_at"] == 3
        # 3 stands alone; 5-6 and 8-9 are runs (the NaN at 7 ends the first); each alarms at its second evaluation.
        assert [r["alarm_at"] for r in result["runs"]] == [6, 9]
        assert result["first_alarm_at"] == 6 and result["alarm"]

    def test_no_alarm_without_two_in_a_row(self):
        result = th.saturation_alarms([1, 2, 3, 4], [5, 0, 9, 4])
        assert result["flagged"] == [1, 3] and not result["alarm"] and result["first_alarm_at"] is None

    def test_an_evaluation_size_other_than_ten_is_said(self):
        assert "note" not in th.saturation_alarms([1, 2], [5, 5], episodes=[10, 10])
        result = th.saturation_alarms([1, 2], [5, 5], episodes=[40, 40])
        assert result["episodes_per_evaluation"] == [40] and "40" in result["note"] and result["alarm"]

    def test_the_rule_reads_the_ledger(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path, saturated=[0, 3, 5, 6, 0, 5]), read_std=False)
        assert run.saturation["first_saturated_episode_at"] == 2 * STEP
        assert run.saturation["flagged"] == [3 * STEP, 4 * STEP, 6 * STEP]
        assert run.saturation["first_alarm_at"] == 4 * STEP


class TestTolerance:
    def test_missing_columns_are_absent_series(self, tmp_path):
        stage = tmp_path / "bare" / "01_stance"
        rows = [
            {"evaluation": e, "timesteps": (e + 1) * STEP, "episode": i, "return": 1.0}
            for e in range(2)
            for i in range(3)
        ]
        _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows)
        run = th.load_run_history(stage, read_std=False)
        assert run.steps.tolist() == [STEP, 2 * STEP]
        assert run.series("mean_return").tolist() == [1.0, 1.0]
        assert run.series("clean") is None and run.series("falls") is None and run.series("saturated") is None
        assert run.evaluations["columns"] == {} and run.feet == [] and run.lr == {}
        assert run.saturation == {"computed": False, "why": "no screen_v2_reasons column"}
        assert f"{th.EVAL_EPISODES_FILENAME}: column screen_v2_reasons" in run.absent
        assert th.STAGE_CONFIG_FILENAME in run.absent and th.TRAIN_EPISODES_FILENAME in run.absent

    def test_missing_files_render(self, tmp_path):
        stage = tmp_path / "empty" / "01_stance"
        stage.mkdir(parents=True)
        run = th.load_run_history(stage, read_std=False)
        assert not run.steps.size and run.training is None and run.errors == []
        written = th.write_report([run], tmp_path / "out", title="empty", marks=["final"], dpi=30)
        assert written["errors"] == []
        assert len(written["pngs"]) == len(th.PAGES) + 1
        document = json.loads((tmp_path / "out" / th.JSON_FILENAME).read_text(), parse_constant=_no_constant)
        assert document["runs"][0]["evaluations"]["timesteps"] == []

    def test_an_unreadable_stage_config_is_no_record(self, tmp_path):
        stage = _stage(tmp_path)
        (stage / th.STAGE_CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
        run = th.load_run_history(stage, read_std=False)
        assert run.model_id is None and run.band is None and run.bars["max_sole_tilt_deg"] is None


def _no_constant(name: str) -> None:
    raise AssertionError(f"the JSON holds {name}")


class TestPolicyStd:
    def _zip(self, path: Path, *, log_std: Any = None, steps: int | None = None) -> None:
        torch = pytest.importorskip("torch")
        state = {"action_net.bias": torch.zeros(3)}
        if log_std is not None:
            state["log_std"] = torch.tensor(log_std, dtype=torch.float32)
        buffer = io.BytesIO()
        torch.save(state, buffer)
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("policy.pth", buffer.getvalue())
            archive.writestr("data", json.dumps({"num_timesteps": steps} if steps is not None else {}))
            archive.writestr("policy.optimizer.pth", b"not read")

    def test_reads_exp_log_std_at_each_checkpoint(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        self._zip(models / "stage1_4096_steps.zip", log_std=[0.0, -1.0, -2.0])
        self._zip(models / "best_model.zip", log_std=[-0.5, -0.5, -0.5], steps=1024)
        self._zip(models / "stage1_final.zip", log_std=[-3.0, -3.0, -3.0], steps=8192)
        self._zip(models / "robust_best_model.zip", steps=2048)  # no log_std (as SAC)
        (models / "stage1_vecnormalize_4096_steps.pkl").write_bytes(b"never opened")
        result = th.read_policy_std(models)
        assert result["available"] == 4 and result["read"] == 3
        points = [(p["kind"], p["timesteps"], p["step_source"]) for p in result["checkpoints"]]
        assert points == [
            ("best", 1024, "zip data (JSON num_timesteps)"),
            ("periodic", 4096, "file name"),
            ("final", 8192, "zip data (JSON num_timesteps)"),
        ]
        assert result["checkpoints"][1]["std"] == pytest.approx(np.exp([0.0, -1.0, -2.0]), rel=1e-6)
        assert any("robust_best_model.zip" in error and "log_std" in error for error in result["errors"])

    def test_spacing_and_budget(self, tmp_path):
        models = tmp_path / "models"
        models.mkdir()
        for k in range(1, 6):
            self._zip(models / f"stage1_{k * 1000}_steps.zip", log_std=[0.0])
        assert [p["timesteps"] for p in th.read_policy_std(models, max_periodic=3)["checkpoints"]] == [1000, 3000, 5000]
        stopped = th.read_policy_std(models, budget_s=-1.0)
        assert stopped["read"] == 0 and "stopped" in stopped["skipped"]

    def test_names_and_kinds(self, tmp_path):
        assert th.checkpoint_kind(Path("stage1_123_steps.zip")) == ("periodic", 123)
        assert th.checkpoint_kind(Path("robust_best_model.zip")) == ("robust_best", None)
        assert th.checkpoint_kind(Path("stage1_final.zip")) == ("final", None)
        assert th.checkpoint_kind(Path("stage1_vecnormalize_123_steps.pkl")) is None
        assert th.read_policy_std(tmp_path / "absent")["skipped"] == "no models/ directory"

    def test_actuator_names_follow_the_model(self):
        pytest.importorskip("mujoco")
        from environments.shared.model_id import ModelId

        names, source = th.actuator_names(ModelId("tyrannosaurus_rex", "anatomical"), 15)
        assert names is not None and "r_hip_roll_act" in names and names[-1] == "tail_3_pitch_act", source
        assert "unchecked" in source
        assert th.actuator_names(ModelId("tyrannosaurus_rex", "anatomical"), 3)[0] is None
        assert th.actuator_names(None, 15) == (None, "no model recorded")

    def test_the_names_need_the_run_s_physics(self, tmp_path):
        from environments.shared.file_io import read_json_object
        from environments.shared.model_id import ModelId
        from environments.shared.paths import REPOSITORY_ROOT

        model = ModelId("tyrannosaurus_rex", "anatomical")
        manifest = read_json_object(REPOSITORY_ROOT / "configs" / "plant_manifest.generated.json")
        physics = manifest["plants"]["tyrannosaurus_rex"]["anatomical"]["physics"]
        stage = tmp_path / "plant" / "01_stance"
        stage.mkdir(parents=True)
        assert th.plant_check(stage, model)["status"] == "unchecked"
        identity = {"physics_sha256": physics["sha256"], "physics_revision": physics["revision"]}
        (stage / th.PLANT_IDENTITY_FILENAME).write_text(json.dumps(identity), encoding="utf-8")
        assert th.plant_check(stage, model)["status"] == "match"
        identity["physics_sha256"] = "sha256:" + "0" * 64
        (stage / th.PLANT_IDENTITY_FILENAME).write_text(json.dumps(identity), encoding="utf-8")
        plant = th.plant_check(stage, model)
        assert plant["status"] == "differs"
        names, source = th.actuator_names(model, 15, plant=plant)
        assert names is None and "labelled by index" in source

    def test_left_equals_right_and_the_joins(self):
        pairs = th._pairs(["r_hip_act", "l_hip_act", "tail_act"])
        assert pairs == {"hip": (0, 1), "tail": (2, None)}
        assert th._left_equals_right(np.asarray([[0.1, 0.1, 0.3], [0.2, 0.2, 0.3]]), pairs) is True
        assert th._left_equals_right(np.asarray([[0.1, 0.11, 0.3]]), pairs) is False
        points = [
            {"kind": "robust_best", "timesteps": 900},
            {"kind": "periodic", "timesteps": 4000},
            {"kind": "periodic", "timesteps": 5000},
            {"kind": "periodic", "timesteps": 6000},
            {"kind": "periodic", "timesteps": 8000},
            {"kind": "final", "timesteps": 8500},
        ]
        solid, dotted = th._std_segments(points)
        # Solid only between periodic checkpoints at most 1.5 spacings (1,000) apart.
        assert solid == [(1, 2), (2, 3)] and dotted == [(0, 1), (3, 4), (4, 5)]


class TestRefusal:
    def test_an_out_inside_the_run_is_refused(self, tmp_path):
        stage = _stage(tmp_path)
        assert script.refusal(stage / "figures", [stage]) is not None
        assert script.refusal(stage.parent / "report", [stage]) is not None
        assert script.refusal(tmp_path / "reports" / "a", [stage]) is None

    def test_the_provenance_ancestor_is_the_run(self, tmp_path):
        run_dir = tmp_path / "drive" / "run_x"
        stage = run_dir / "nested" / "01_stance"
        stage.mkdir(parents=True)
        (run_dir / "provenance.json").write_text("{}", encoding="utf-8")
        assert script.refusal(run_dir / "elsewhere", [stage]) is not None
        assert script.refusal(tmp_path / "drive" / "reports", [stage]) is None

    def test_duplicate_labels_are_refused(self, tmp_path, capsys):
        first = _stage(tmp_path / "x", name="same")
        second = _stage(tmp_path / "y", name="same")
        out = tmp_path / "reports" / "dup"
        assert script.main([str(first), str(second), "--out", str(out), "--no-std"]) == 1
        assert "share a label" in capsys.readouterr().err and not out.exists()
        with pytest.raises(SystemExit):
            script.main([str(first), str(second), "--out", str(out), "--no-std", "--labels", "a", "a"])

    def test_main_writes_nothing_when_refused(self, tmp_path, capsys):
        stage = _stage(tmp_path)
        out = stage.parent / "history"
        assert script.main([str(stage), "--out", str(out), "--no-std"]) == 2
        assert not out.exists()
        assert "refusing" in capsys.readouterr().err


class TestRender:
    def test_one_run_end_to_end(self, tmp_path, capsys):
        stage = _stage(tmp_path)
        out = tmp_path / "reports" / "one"
        status = script.main(
            [str(stage), "--out", str(out), "--no-std", "--dpi", "30", "--mark-steps", "4096", "final"]
        )
        assert status == 0, capsys.readouterr().err
        pngs = sorted(out.glob("*.png"))
        assert [p.stem for p in pngs] == [spec.name for spec in th.PAGES] + ["10_policy_std"]
        assert all(p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n" for p in pngs)
        document = json.loads((out / th.JSON_FILENAME).read_text(), parse_constant=_no_constant)
        assert document["schema"] == th.SCHEMA and document["errors"] == []
        record = document["runs"][0]
        assert record["saturation"]["first_alarm_at"] == 3 * STEP
        assert record["evaluations"]["clean"] == [9, 4, 3, 10]
        assert record["bars"]["max_sole_tilt_deg"] == 2.0 and record["band_active"] is True
        assert document["marks"]["resolved"] == [4096]
        assert record["checkpoint_table"][0]["at"] == "J 4,096" and record["checkpoint_table"][0]["saturated"] == 6
        text = (out / th.HTML_FILENAME).read_text(encoding="utf-8")
        assert text.count("data:image/png;base64,") == len(pngs) and "Hip roll" in text
        assert not list(stage.parent.rglob("*.png")) and not list(stage.parent.rglob("*.html"))

    def test_comparison_pages(self, tmp_path):
        runs = [
            th.load_run_history(_stage(tmp_path, name="a"), label="a", group="band", read_std=False),
            th.load_run_history(
                _stage(tmp_path, saturated=[0, 0, 0, 0], name="b"), label="b", group="control", read_std=False
            ),
        ]
        pages, errors = th.render_pages(runs, title="a against b", marks=[STEP], dpi=30)
        assert errors == [] and len(pages) == len(th.PAGES) + 1
        assert all(page.png[:4] == b"\x89PNG" for page in pages)
        assert "colour per seed" in pages[0].caption and "10 episodes per evaluation" in pages[0].caption

    @staticmethod
    def _texts(axis) -> list[str]:
        return [text.get_text() for text in axis.texts]

    def test_one_run_panels_draw_their_bars_band_and_marks(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        context = th._make_context([run], marks=[2 * STEP], smooth=5)
        spec = {page.name: page for page in th.PAGES}
        figure = th._page_figure(spec["04_pad_cop"], context, title="t", dpi=30, width=12.0)
        signed, rail, over, pad_band = figure.axes[:4]
        # The pad band's edges on the signed panel, the rail's bar on the rail's own panel only.
        assert any("pad band edge +0.65" in text for text in self._texts(signed))
        assert not any("bar" in text for text in self._texts(signed))
        assert any("max_sole_cop_fore_aft bar 0.8" in text for text in self._texts(rail))
        assert 0.8 in [line.get_ydata()[0] for line in rail.lines if line.get_linestyle() == "--"]
        for axis in (signed, rail, over, pad_band):
            assert not any("not recorded" in text for text in self._texts(axis))
        assert "J 4k" in self._texts(signed)
        # The alarm (third evaluation) is a dotted vertical line; J a solid one.
        xs = {round(float(line.get_xdata()[0]) * 1e6) for line in signed.lines if len(line.get_xdata()) == 2}
        assert {2 * STEP, 3 * STEP} <= xs
        roll = th._page_figure(spec["07_sole_roll"], context, title="t", dpi=30, width=12.0)
        assert any("bar 4 mm (0.004 m)" in text for text in self._texts(roll.axes[1]))
        falls = th._page_figure(spec["01_outcomes"], context, title="t", dpi=30, width=12.0).axes[1]
        assert "1" in self._texts(falls) and falls.get_ylim()[1] >= 3

    def test_comparison_counts_are_unsmoothed_strips(self, tmp_path):
        runs = [
            th.load_run_history(_stage(tmp_path, name="a"), label="a", group="band", read_std=False),
            th.load_run_history(
                _stage(tmp_path, saturated=[0, 0, 0, 0], name="b"), label="b", group="control", read_std=False
            ),
        ]
        context = th._make_context(runs, marks=[STEP], smooth=5)
        figure = th._page_figure(th.PAGES[1], context, title="t", dpi=30, width=12.0)
        strip = figure.axes[0]
        # One cell per evaluation per run, coloured by the raw count; the flagged evaluations outlined.
        cells = [patch for patch in strip.patches if patch.get_fill()]
        assert len(cells) == 8
        outlined = [patch for patch in strip.patches if not patch.get_fill()]
        assert len(outlined) == 2  # run a's evaluations 2 and 3 (6 and 7 flagged episodes)
        assert "a" in self._texts(strip) and "b" in self._texts(strip)

    def test_at_most_eight_runs(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        with pytest.raises(ValueError, match="at most 8"):
            th.render_pages([run] * 9, title="t", marks=[])

    def test_final_mark_resolves_per_run(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        assert th.resolve_marks([4096, "final"], [run]) == [4096]
        run.checkpoint_steps["final"] = 9000
        assert th.resolve_marks(["final", 4096], [run]) == [4096, 9000]


def test_the_comparison_smoothing_keeps_the_length_and_skips_nan():
    values = np.asarray([1.0, math.nan, 3.0, 5.0])
    smoothed = th._smooth(values, 5)
    assert smoothed.shape == values.shape
    assert smoothed.tolist() == pytest.approx([2.0, 3.0, 3.0, 4.0])
    unsmoothed = th._smooth(values, 1)
    assert unsmoothed[0] == 1.0 and math.isnan(unsmoothed[1]) and unsmoothed[2:].tolist() == [3.0, 5.0]
    assert math.isnan(th._smooth(np.asarray([math.nan]), 3)[0])


def test_colours_follow_the_seed_and_line_styles_the_group(tmp_path):
    runs = [
        th.load_run_history(_stage(tmp_path, name=name), label=name, group=group, read_std=False)
        for name, group in (("band_a", "band"), ("control_a", "control"))
    ]
    runs[1].seed = 42  # both seed 42 (the synthetic stage_config's)
    context = th._make_context(runs, marks=[], smooth=5)
    assert context.colours[0] == context.colours[1] and context.styles[0] != context.styles[1]
    runs[1].group = "band"  # the same (seed, group) twice: colour per run instead
    assert len(set(th._make_context(runs, marks=[], smooth=5).colours)) == 2


def test_the_final_mark_row_reads_the_last_evaluation_and_the_final_std(tmp_path):
    run = th.load_run_history(_stage(tmp_path), read_std=False)
    run.checkpoint_steps["final"] = 9000
    run.actuators = ["r_hip_pitch_act", "l_hip_pitch_act", "tail_3_pitch_act"]
    run.policy_std["checkpoints"] = [
        {"name": "stage1_8192_steps.zip", "kind": "periodic", "timesteps": 8192, "std": np.asarray([0.5, 0.5, 0.2])},
        {"name": "stage1_final.zip", "kind": "final", "timesteps": 9000, "std": np.asarray([0.4, 0.4, 0.1])},
    ]
    rows = th.checkpoint_rows(run, th.resolve_marks([4 * STEP, "final"], [run]))
    final = next(row for row in rows if row["at"] == "J 9,000 (final)")
    assert final["evaluation_timesteps"] == 4 * STEP and final["note"] == "evaluation at 8,192 (none at this step)"
    assert final["std_checkpoint"] == "stage1_final.zip" and final["std_tail_3_pitch"] == pytest.approx(0.1)
    at_last = next(row for row in rows if row["at"] == "J 8,192")
    assert at_last["std_checkpoint"] == "stage1_8192_steps.zip" and "note" not in at_last
    assert [row["at"] for row in rows][-1] == "last evaluation"
