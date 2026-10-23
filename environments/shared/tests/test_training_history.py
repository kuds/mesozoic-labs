"""The training-history report (``reporting/training_history.py``, ``scripts/plot_training_history.py``).

Small synthetic ledgers only: the reductions, the saturation flag and alarm
rule, tolerance of missing files and columns, the refusal to write inside a run
directory, the per-actuator std reader on a tiny checkpoint zip, and that every
page renders on the Agg canvas.
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
        assert windows["window_end"].tolist() == [10, 20, 30]
        assert windows["episodes"].tolist() == [2, 0, 1]
        assert windows["falls"].tolist() == [1, 0, 1]
        assert windows["fall_fraction"][0] == 0.5 and math.isnan(windows["fall_fraction"][1])

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
        assert th.actuator_names(ModelId("tyrannosaurus_rex", "anatomical"), 3)[0] is None
        assert th.actuator_names(None, 15) == (None, "no model recorded")


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
    assert th._smooth(values, 1) is not None and math.isnan(th._smooth(np.asarray([math.nan]), 3)[0])
