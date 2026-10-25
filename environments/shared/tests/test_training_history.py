"""The training-history report (``reporting/training_history.py``, ``scripts/plot_training_history.py``).

Small synthetic ledgers only: the reductions (counts, the rails' over-bar
counts, the band counts, per-foot means, left-minus-right offsets, training
windows), the saturation flag and alarm rule, the flagged episodes by the
actuator they name and the alarms labelled with it, tolerance of missing files and
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
#: The columns a T. rex row carries that the report reads (a subset of the real 68).
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
    "saturated_actuator",
    "saturated_actuator_fraction",
    "saturated_actuator_positive_share",
    "saturated_actuator_ties",
    "actuators_saturated",
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
        saturated_actuator="tail_3_pitch_act" if saturated else "",
        saturated_actuator_fraction=0.31 if saturated else 0.0,
        saturated_actuator_positive_share=1.0 if saturated else math.nan,
        actuators_saturated=1 if saturated else 0,
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


class TestNamedActuators:
    """The flagged episodes by the actuator each names (``saturated_actuator``) and its sign, per evaluation."""

    def _ledger(self, tmp_path: Path, name: str = "named") -> Path:
        stage = _stage(tmp_path, saturated=[0, 6, 7, 0], name=name)
        rows = list(csv.DictReader((stage / th.EVAL_EPISODES_FILENAME).open(newline="", encoding="utf-8")))
        for row in rows:
            evaluation, episode = int(row["evaluation"]), int(row["episode"])
            if evaluation == 2 and episode in (0, 1):
                row.update(saturated_actuator="r_ankle_act", saturated_actuator_positive_share="0.0")
            if evaluation == 2 and episode == 2:
                row.update(saturated_actuator_positive_share="0.25")  # bang-bang: both signs
            if evaluation == 2 and episode == 3:
                row.update(saturated_actuator="", saturated_actuator_fraction="nan")  # an unmeasured window
            if evaluation == 2 and episode == 4:
                row.update(saturated_actuator="", saturated_actuator_fraction="")  # a row from before the column
            if evaluation == 3 and episode == 0:
                row.update(saturated_actuator="l_knee_act", saturated_actuator_fraction="0.05")  # under the bar
        _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows, COLUMNS)
        return stage

    def test_flagged_episodes_by_actuator_and_sign_per_evaluation(self, tmp_path):
        run = th.load_run_history(self._ledger(tmp_path), read_std=False)
        assert run.evaluations["flagged_actuators"] == [
            {},
            {"tail_3_pitch_act +": 6},
            {
                "r_ankle_act -": 2,
                "tail_3_pitch_act +": 2,
                "(not recorded)": 1,
                "(unmeasured)": 1,
                "tail_3_pitch_act +/-": 1,
            },
            {},  # an episode naming an actuator under the bar is not flagged
        ]
        assert sum(run.evaluations["flagged_actuators"][2].values()) == run.series("saturated")[2] == 7
        assert th.saturation_sign(1.0) == "+" and th.saturation_sign(0.0) == "-" and th.saturation_sign(0.5) == "+/-"
        assert th.most_named({"b +": 2, "a -": 2, "c +": 1}) == "a -" and th.most_named({}) is None
        # The two numeric columns are reduced; the name and its share are read per flagged episode, as text.
        named = list(th.SATURATED_ACTUATOR_COLUMNS)
        assert th.reduced_columns(named) == ["saturated_actuator_fraction", "actuators_saturated"]
        # The fraction is reduced like a rail: over the full-horizon episodes, mean and worst.
        assert run.column("saturated_actuator_fraction")[1] == pytest.approx(0.6 * 0.31)
        assert run.column("saturated_actuator_fraction", "max")[1] == pytest.approx(0.31)

    def test_the_most_named_actuator_is_counted_over_its_signs(self):
        # The tail split over + and +/- (10) outnumbers the ankle of one sign (6): named, with the sign most carry.
        split = {"r_ankle_act -": 6, "tail_3_pitch_act +": 4, "tail_3_pitch_act +/-": 4}
        assert th.most_named(split) == "tail_3_pitch_act +/-"
        assert th.most_named({"tail_3_pitch_act +": 15, "tail_3_pitch_act +/-": 14, "r_ankle_act -": 16}) == (
            "tail_3_pitch_act +"
        )
        assert th.most_named({"a +": 2, "a -": 2}) == "a +/-"  # the signs tied
        # A named actuator before one naming none on a tie, whichever sorts first.
        assert th.most_named({"tail_3_pitch_act +": 1, th.NOT_RECORDED_ACTUATOR: 1}) == "tail_3_pitch_act +"
        assert th.most_named({th.UNMEASURED_ACTUATOR: 2, "z_act -": 2}) == "z_act -"
        assert th.most_named({th.UNMEASURED_ACTUATOR: 2, th.NOT_RECORDED_ACTUATOR: 1}) == th.UNMEASURED_ACTUATOR
        assert th.most_named({"a +": 0}) is None
        # A label only the tie of signs gives still has a colour for the cell it is drawn in, after the counted ones.
        tied = th.RunHistory(stage_dir=Path("t"), label="t")
        tied.evaluations["flagged_actuators"] = [{"a +": 1, "a -": 1}, None]
        assert list(th._named_colours([tied])) == ["a +", "a -", "a +/-"]

    def test_the_alarm_names_the_actuator_its_flagged_episodes_name_most(self, tmp_path):
        run = th.load_run_history(self._ledger(tmp_path), read_std=False)
        (alarm,) = run.saturation["runs"]
        assert alarm["alarm_at"] == 3 * STEP and alarm["named"] == "tail_3_pitch_act +"
        assert alarm["actuators"] == {
            "tail_3_pitch_act +": 8,
            "r_ankle_act -": 2,
            "(not recorded)": 1,
            "(unmeasured)": 1,
            "tail_3_pitch_act +/-": 1,
        }

    def test_the_finished_models_read_their_own_evaluation(self, tmp_path):
        run = th.load_run_history(self._ledger(tmp_path), read_std=False)
        run.checkpoint_steps.update(robust_best=2 * STEP, best=3 * STEP, final=9000)
        rows = {row["at"]: row for row in th.checkpoint_rows(run, th.resolve_marks(["final"], [run]))}
        assert rows["robust_best"]["flagged_actuators"] == {"tail_3_pitch_act +": 6}
        assert rows["best"]["flagged_actuators"]["r_ankle_act -"] == 2
        assert (
            rows["J 9,000 (final)"]["flagged_actuators"] == {}
            and rows["J 9,000 (final)"]["evaluation_timesteps"] == 4 * STEP
        )
        assert rows["robust_best"]["saturated_actuator_fraction_worst"] == pytest.approx(0.31)

    def test_the_json_html_and_saturation_page_carry_it(self, tmp_path):
        stage = self._ledger(tmp_path)
        out = tmp_path / "reports" / "named"
        assert script.main([str(stage), "--out", str(out), "--no-std", "--dpi", "30", "--mark-steps", "final"]) == 0
        document = json.loads((out / th.JSON_FILENAME).read_text(), parse_constant=_no_constant)
        record = document["runs"][0]
        assert record["evaluations"]["flagged_actuators"][1] == {"tail_3_pitch_act +": 6}
        assert record["saturation"]["runs"][0]["named"] == "tail_3_pitch_act +"
        assert record["checkpoint_table"][-1]["flagged_actuators"] == {}
        assert "flagged_actuators" in document["rules"]
        assert record["evaluations"]["columns"]["saturated_actuator_fraction"]["max"][1] == pytest.approx(0.31)
        text = (out / th.HTML_FILENAME).read_text(encoding="utf-8")
        assert "flagged by actuator" in text and "(tail_3_pitch_act +)" in text
        run = th.load_run_history(stage, read_std=False)
        context = th._make_context([run], marks=[], smooth=5)
        spec = {page.name: page for page in th.PAGES}["02_saturation"]
        figure = th._page_figure(spec, context, title="t", dpi=30, width=12.0)
        saturated, named, fraction = figure.axes[:3]
        # Stacked raw counts per evaluation: the bars of one evaluation sum to its flagged episodes.
        heights = {}
        for patch in named.patches:
            key = round(patch.get_x() + patch.get_width() / 2, 6)
            heights[key] = heights.get(key, 0.0) + patch.get_height()
        assert sorted(heights.values()) == [6.0, 7.0]
        legend = [text.get_text() for text in named.get_legend().get_texts()]
        assert legend[0] == "tail_3_pitch_act + (8)" and "r_ankle_act - (2)" in legend
        assert any("(tail_3_pitch_act +)" in text.get_text() for text in saturated.texts)  # the alarm, labelled
        assert any("max_actuator_saturation_fraction bar 0.1" in text.get_text() for text in fraction.texts)
        caption = th.render_pages([run], title="t", marks=[], dpi=30)[0][1].caption
        assert "saturated_actuator" in caption and "records no" not in caption

    def test_a_comparison_colours_each_cell_by_the_actuator_most_named(self, tmp_path):
        from matplotlib.colors import to_rgba

        other = _stage(tmp_path, saturated=[0, 0, 5, 0], name="b")
        rows = list(csv.DictReader((other / th.EVAL_EPISODES_FILENAME).open(newline="", encoding="utf-8")))
        for row in rows:
            if row["saturated_actuator"]:
                row.update(saturated_actuator="r_ankle_act", saturated_actuator_positive_share="0.0")
        _write_csv(other / th.EVAL_EPISODES_FILENAME, rows, COLUMNS)
        runs = [
            th.load_run_history(self._ledger(tmp_path, "a"), label="a", read_std=False),
            th.load_run_history(other, label="b", read_std=False),
        ]
        context = th._make_context(runs, marks=[], smooth=5)
        assert list(context.named_colours)[:2] == ["tail_3_pitch_act +", "r_ankle_act -"]
        figure = th._page_figure(th.PAGES[1], context, title="t", dpi=30, width=12.0)
        strip = figure.axes[1]
        colours = [tuple(patch.get_facecolor()) for patch in strip.patches]
        tail, ankle = (to_rgba(context.named_colours[label]) for label in ("tail_3_pitch_act +", "r_ankle_act -"))
        # a: evaluation 1, then evaluation 2 (the tail in three episodes, + and +/-, before the ankle in two); b: 2.
        assert colours == [tail, tail, ankle]
        labels = [text.get_text() for text in figure.legends[0].get_texts()]
        assert "flagged: tail_3_pitch_act +" in labels and "flagged: r_ankle_act -" in labels
        # The key lists only the colours a cell is drawn in: a's one +/- episode never colours one.
        assert not any("+/-" in label or "(unmeasured)" in label for label in labels)

    def test_an_older_ledger_is_an_absent_series(self, tmp_path):
        stage = _stage(tmp_path, name="old")
        older = [column for column in COLUMNS if column not in th.SATURATED_ACTUATOR_COLUMNS]
        rows = list(csv.DictReader((stage / th.EVAL_EPISODES_FILENAME).open(newline="", encoding="utf-8")))
        _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows, older)
        run = th.load_run_history(stage, read_std=False)
        assert "flagged_actuators" not in run.evaluations and run.series("saturated").tolist() == [0, 6, 7, 0]
        assert "named" not in run.saturation["runs"][0] and run.column("saturated_actuator_fraction") is None
        assert f"{th.EVAL_EPISODES_FILENAME}: column saturated_actuator" in run.absent
        assert "flagged_actuators" not in th.checkpoint_rows(run, [STEP])[0]
        pages, errors = th.render_pages([run], title="t", marks=[], dpi=30)
        assert errors == [] and "this ledger records no saturated_actuator" in pages[1].caption
        figure = th._page_figure(
            th.PAGES[1], th._make_context([run], marks=[], smooth=5), title="t", dpi=30, width=12.0
        )
        assert any(
            "not recorded in these files (saturated_actuator)" in text.get_text() for text in figure.axes[1].texts
        )


def _probe(stage: Path, *, rollouts: int = 8, older: bool = False) -> Path:
    """A ``policy_probe.npz`` of *rollouts* rows of 2,048 buffer rows, the first two from before the rollout arrays.

    tail_3_pitch_act (14): samples clipped at + rising to 0.3 and its mean beyond + to 0.5; r_knee_act (5): 2% of
    samples clipped at - from the fourth rollout; l_knee_act (9): 0.4% at + (under the naming level).
    """
    nu = 15
    steps = np.arange(1, rollouts + 1, dtype=np.int64) * STEP
    arrays: dict[str, Any] = {
        "update_timesteps": steps,
        "std_per_actuator": np.full((rollouts, nu), 0.2, dtype=np.float32),
    }
    if not older:
        zeros = np.zeros((rollouts, nu), dtype=np.float32)
        clip_high, clip_low, mu_high, mu_low = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
        clip_high[:, 14] = np.linspace(0.0, 0.3, rollouts)
        mu_high[:, 14] = np.linspace(0.0, 0.5, rollouts)
        clip_low[3:, 5] = 0.02
        clip_high[:, 9] = 0.004
        mean = np.full((rollouts, nu), 0.1, dtype=np.float32)
        rows = np.full(rollouts, STEP, dtype=np.int64)
        arrays.update(
            rollout_rows=rows,
            rollout_action_clip_high_share=clip_high,
            rollout_action_clip_low_share=clip_low,
            rollout_action_saturated_share=clip_high + clip_low,
            rollout_mu_beyond_high_share=mu_high,
            rollout_mu_beyond_low_share=mu_low,
            rollout_mu_mean=mean,
            rollout_mu_q50=mean,
            rollout_mu_q95=mean + 0.5,
        )
        rows[:2] = 0  # a resumed session's rows from before the arrays: NaN
        for key in th.ROLLOUT_ARRAYS:
            arrays[key][:2] = np.nan
    path = stage / th.POLICY_PROBE_FILENAME
    np.savez(path, **arrays)
    return path


class TestRolloutShares:
    def test_the_shares_are_read_and_summarised_per_actuator(self, tmp_path):
        stage = _stage(tmp_path)
        _probe(stage)
        run = th.load_run_history(stage, read_std=False)
        assert run.rollout_shares is not None and run.rollout_actuators[14] == "tail_3_pitch_act"
        assert run.last_step == 8 * STEP  # the probe's last rollout is past the last evaluation
        summary = th.rollout_summary(run)
        assert summary["rollouts"] == 8 and summary["rollouts_measured"] == 6
        assert (summary["first_timesteps"], summary["last_timesteps"], summary["rows_per_rollout"]) == (
            3 * STEP,
            8 * STEP,
            [STEP],
        )
        assert summary["named"] == {
            "action_clip": ["tail_3_pitch_act +", "r_knee_act -"],
            "mu_beyond": ["tail_3_pitch_act +"],
        }
        tail = summary["per_actuator"]["tail_3_pitch_act"]
        assert tail["max"]["action_clip_high_share"] == pytest.approx(0.3)
        assert tail["last"]["mu_beyond_high_share"] == pytest.approx(0.5)
        assert tail["last"]["mu_q95"] == pytest.approx(0.6) and "mu_q95" not in tail["max"]
        # The first rollout at the level: 0.3 * 2 / 7 = 0.086 at the third (the first two are NaN).
        assert tail["first_at_level"] == {"action_clip": 3 * STEP, "mu_beyond": 3 * STEP}
        assert summary["per_actuator"]["r_knee_act"]["first_at_level"] == {"action_clip": 4 * STEP, "mu_beyond": None}
        assert summary["per_actuator"]["l_knee_act"]["max"]["action_clip_high_share"] == pytest.approx(0.004)

    def test_the_named_series_and_the_rest(self):
        shares = {
            "rollout_action_clip_high_share": np.array([[0.0, 0.2, 0.001], [np.nan, np.nan, np.nan]]),
            "rollout_action_clip_low_share": np.array([[0.05, 0.0, 0.003], [np.nan, np.nan, np.nan]]),
        }
        named, others, count = th.rollout_series(shares, ["a", "b", "c"], "rollout_action_clip")
        assert [label for label, _ in named] == ["b +", "a -"]
        assert others is not None and others[0] == pytest.approx(0.003) and math.isnan(others[1]) and count == 3
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(th, "ROLLOUT_NAMED_SERIES", 1)
            named, others, count = th.rollout_series(shares, ["a", "b", "c"], "rollout_action_clip")
        assert [label for label, _ in named] == ["b +"] and others[0] == pytest.approx(0.05) and count == 3

    def test_the_saturation_page_draws_them_unsmoothed_and_names_the_actuators(self, tmp_path):
        stage = _stage(tmp_path)
        _probe(stage)
        run = th.load_run_history(stage, read_std=False)
        context = th._make_context([run], marks=[], smooth=5)
        figure = th._page_figure(th.PAGES[1], context, title="t", dpi=30, width=12.0)
        clipped, beyond = figure.axes[4:6]
        legend = [text.get_text() for text in clipped.get_legend().get_texts()]
        assert legend == ["tail_3_pitch_act +", "r_knee_act -", "the rest, largest"]
        tail = clipped.lines[0]
        assert tail.get_linestyle() == "-" and clipped.lines[1].get_linestyle() == "--"
        # The tail keeps the colour its flagged evaluation episodes are drawn in above; the knee takes a free one.
        assert tail.get_color() == context.named_colours["tail_3_pitch_act +"]
        assert clipped.lines[1].get_color() not in (tail.get_color(), th._MUTED)
        assert any("named from 0.01" in text.get_text() for text in clipped.texts)
        # The first two rollouts were not measured (a resume's rows from an older file): said on the panel.
        assert "not measured before 6k" in [text.get_text() for text in clipped.texts]
        # Every rollout's raw value, never smoothed.
        assert np.allclose(tail.get_ydata()[2:], np.linspace(0.0, 0.3, 8)[2:]) and np.isnan(tail.get_ydata()[0])
        assert np.allclose(clipped.lines[2].get_ydata()[2:], 0.004)
        assert [text.get_text() for text in beyond.get_legend().get_texts()][0] == "tail_3_pitch_act +"
        caption = th.render_pages([run], title="t", marks=[], dpi=30)[0][1].caption
        assert "policy_probe.npz" in caption and "panels are absent" not in caption
        assert "(this run: some rollouts not measured, said on the panel" in caption

    def test_an_actuator_only_the_rollouts_name_takes_no_flagged_actuators_colour(self, tmp_path):
        stage = _stage(tmp_path)
        # Evaluation 2's seven flagged episodes name l_hip_pitch_act at -, so it ranks first and the tail second.
        hip = {"saturated_actuator": "l_hip_pitch_act", "saturated_actuator_positive_share": 0.0}
        rows = [
            _episode(
                evaluation, episode, saturated=episode < count, **(hip if evaluation == 2 and episode < count else {})
            )
            for evaluation, count in enumerate([0, 6, 7, 0])
            for episode in range(10)
        ]
        _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows, COLUMNS)
        _probe(stage)
        run = th.load_run_history(stage, read_std=False)
        context = th._make_context([run], marks=[], smooth=5)
        assert list(context.named_colours) == ["l_hip_pitch_act -", "tail_3_pitch_act +"]
        clipped = th._page_figure(th.PAGES[1], context, title="t", dpi=30, width=12.0).axes[4]
        tail, knee = clipped.lines[:2]
        assert (tail.get_label(), knee.get_label()) == ("tail_3_pitch_act +", "r_knee_act -")
        assert tail.get_color() == context.named_colours["tail_3_pitch_act +"]
        # The knee is flagged nowhere: neither the hip's colour above nor the tail's, nor grey.
        assert knee.get_color() not in (*context.named_colours.values(), th._MUTED)

    def test_with_every_named_colour_flagged_the_rollouts_still_draw_distinct_colours(self, tmp_path):
        stage = _stage(tmp_path)
        context = th._make_context([th.load_run_history(stage, read_std=False)], marks=[], smooth=5)
        context.named_colours = {f"flagged_{i} +": colour for i, colour in enumerate(th._NAMED_SERIES)}
        colours = th._rollout_colours(context, ["flagged_2 -", "knee +", "knee -", "ankle +"])
        assert colours["flagged_2"] == th._NAMED_SERIES[2]
        assert len(set(colours.values())) == 3 and th._MUTED not in colours.values()

    def test_a_comparison_draws_each_runs_largest_share(self, tmp_path):
        first = _stage(tmp_path, name="a")
        _probe(first)
        second = _stage(tmp_path, name="b")
        runs = [th.load_run_history(stage, label=stage.parent.name, read_std=False) for stage in (first, second)]
        context = th._make_context(runs, marks=[], smooth=5)
        figure = th._page_figure(th.PAGES[1], context, title="t", dpi=30, width=12.0)
        clipped = figure.axes[4]
        drawn = [line for line in clipped.lines if line.get_label() == "_nolegend_" and len(line.get_xdata()) == 8]
        assert len(drawn) == 1
        assert np.allclose(
            drawn[0].get_ydata()[2:], np.maximum(np.linspace(0.0, 0.3, 8), [0, 0, 0, 0.02, 0.02, 0.02, 0.02, 0.02])[2:]
        )
        texts = [text.get_text() for text in clipped.texts]
        assert "a (tail_3_pitch_act)" in texts and "not recorded: b" in texts and "a: not measured before 6k" in texts
        caption = th.render_pages(runs, title="t", marks=[], dpi=30)[0][1].caption
        assert "(b: no policy_probe.npz rollout_* arrays" in caption

    def test_a_comparisons_label_names_the_actuator_its_line_traces(self, tmp_path):
        runs = []
        for name in ("a", "b"):
            stage = _stage(tmp_path, name=name)
            _probe(stage)
            runs.append(th.load_run_history(stage, label=name, read_std=False))
        shares = runs[0].rollout_shares
        assert shares is not None
        high, low = (np.zeros_like(shares[key]) for key in th.ROLLOUT_SHARES["rollout_action_clip"])
        high[:, 0], high[:, 1], low[:, 0] = 0.04, 0.06, 0.04  # neck_pitch_act: 0.08 summed; neck_yaw_act's + peaks
        shares["rollout_action_clip_high_share"], shares["rollout_action_clip_low_share"] = high, low
        assert th.rollout_series(shares, runs[0].rollout_actuators, "rollout_action_clip")[0][0][0] == "neck_yaw_act +"
        context = th._make_context(runs, marks=[], smooth=5)
        clipped = th._page_figure(th.PAGES[1], context, title="t", dpi=30, width=12.0).axes[4]
        assert np.allclose(clipped.lines[0].get_ydata(), 0.08)
        texts = [text.get_text() for text in clipped.texts]
        assert "a (neck_pitch_act)" in texts and "b (tail_3_pitch_act)" in texts

    def test_an_older_probe_file_or_none_is_an_absent_series(self, tmp_path):
        stage = _stage(tmp_path)
        run = th.load_run_history(stage, read_std=False)
        assert run.rollout_shares is None and "rollout clip shares: no policy_probe.npz" in run.absent
        _probe(stage, older=True)
        run = th.load_run_history(stage, read_std=False)
        assert run.rollout_shares is None and run.errors == []
        assert th.rollout_summary(run) == {
            "recorded": False,
            "why": "policy_probe.npz has no rollout_* arrays (written before them)",
        }
        figure = th._page_figure(
            th.PAGES[1], th._make_context([run], marks=[], smooth=5), title="t", dpi=30, width=12.0
        )
        assert any("policy_probe.npz rollout_action_clip_*" in text.get_text() for text in figure.axes[4].texts)
        (stage / th.POLICY_PROBE_FILENAME).write_bytes(b"not a zip")
        run = th.load_run_history(stage, read_std=False)
        assert run.rollout_shares is None and run.errors[0].startswith("policy_probe.npz: ")

    def test_the_json_carries_the_summary_and_the_rule(self, tmp_path):
        stage = _stage(tmp_path)
        _probe(stage)
        out = tmp_path / "reports" / "rollout"
        assert script.main([str(stage), "--out", str(out), "--no-std", "--dpi", "30"]) == 0
        document = json.loads((out / th.JSON_FILENAME).read_text(), parse_constant=_no_constant)
        record = document["runs"][0]["rollout_shares"]
        assert record["recorded"] is True and record["named"]["mu_beyond"] == ["tail_3_pitch_act +"]
        assert record["per_actuator"]["neck_pitch_act"]["last"]["action_clip_low_share"] == 0.0
        assert "rollout_shares" in document["rules"] and document["errors"] == []


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
        assert len(written["pngs"]) == len(th.page_names())
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


POSTURE = [*th.POSTURE_COLUMNS, *(f"{foot}_{suffix}" for foot in FEET for suffix in th.HIP_ROLL_SUFFIXES)]


def _posture_stage(
    tmp_path: Path,
    *,
    name: str = "posed",
    head_target: float | None = 0.9,
    constant: dict[str, float] | None = None,
    saturated: list[int] | None = None,
) -> Path:
    """:func:`_stage` with the head posture and hip roll columns appended, as the ledger writes them (last).

    Evaluation k: the head tip at 1.2 - 0.2 k m (episode i 0.01 i higher), its window minimum 0.05 m lower, the neck
    at 10 k degrees and the head at -2 degrees; the left hip roll 1 + k degrees, the right 0.5 degrees, each
    episode's max |roll| 2 degrees.  *constant*: columns given one value on every episode instead (nan: a model
    without that site or hinge); *saturated*: as :func:`_stage`'s.
    """
    stage = _stage(tmp_path, name=name, saturated=saturated)
    rows = list(csv.DictReader((stage / th.EVAL_EPISODES_FILENAME).open(newline="", encoding="utf-8")))
    for row in rows:
        evaluation, episode = int(row["evaluation"]), int(row["episode"])
        head = 1.2 - 0.2 * evaluation + 0.01 * episode
        row.update(
            head_tip_height_mean_m=head,
            head_tip_height_min_m=head - 0.05,
            neck_pitch_mean_deg=10.0 * evaluation,
            head_pitch_mean_deg=-2.0,
            r_hip_roll_mean_deg=0.5,
            r_hip_roll_max_abs_deg=2.0,
            l_hip_roll_mean_deg=1.0 + evaluation,
            l_hip_roll_max_abs_deg=2.0,
        )
        row.update(constant or {})
    _write_csv(stage / th.EVAL_EPISODES_FILENAME, rows, COLUMNS + POSTURE)
    config = json.loads((stage / th.STAGE_CONFIG_FILENAME).read_text(encoding="utf-8"))
    if head_target is not None:
        config["reward_weights"]["head_clearance_target"] = head_target
    (stage / th.STAGE_CONFIG_FILENAME).write_text(json.dumps(config), encoding="utf-8")
    return stage


class TestHeadPostureAndHipRoll:
    def test_the_columns_reduce_over_the_full_horizon_episodes(self, tmp_path):
        run = th.load_run_history(_posture_stage(tmp_path), read_std=False)
        assert run.hip_roll_feet == ["r", "l"] and run.head_target == 0.9
        # Evaluation 0 drops its fallen episode 9: the mean of episodes 0..8, and the lowest minimum is episode 0's.
        assert run.column("head_tip_height_mean_m")[0] == pytest.approx(1.2 + 0.01 * 4)
        assert run.column("head_tip_height_mean_m")[1] == pytest.approx(1.0 + 0.01 * 4.5)
        assert run.column("head_tip_height_min_m", "min")[1] == pytest.approx(1.0 - 0.05)
        assert run.column("neck_pitch_mean_deg").tolist() == pytest.approx([0.0, 10.0, 20.0, 30.0])
        assert run.column("r_hip_roll_max_abs_deg", "max").tolist() == pytest.approx([2.0] * 4)
        # The left-minus-right hip roll: per episode, the left leg's mean minus the right's (both + abduction).
        assert run.lr["hip_roll"] == "hip_roll_l_minus_r_deg"
        assert run.column("hip_roll_l_minus_r_deg").tolist() == pytest.approx([0.5, 1.5, 2.5, 3.5])
        assert not any("hip roll" in entry or "head_tip" in entry for entry in run.absent)
        row = th.checkpoint_rows(run, [2 * STEP])[0]
        assert row["head_tip_height_mean_m"] == pytest.approx(1.045) and row["l_minus_r_hip_roll_deg"] == 1.5
        assert row["head_tip_height_min_m_worst"] == pytest.approx(0.95) and row["neck_pitch_mean_deg"] == 10.0

    def test_the_pairs_follow_the_side_letter(self):
        assert th.hip_roll_pairs(["fr", "fl", "rr", "rl"]) == [("fl", "fr"), ("rl", "rr")]
        assert th.hip_roll_pairs(["r", "l"]) == [("l", "r")] and th.hip_roll_pairs(["l"]) == []
        header = [f"{foot}_hip_roll_mean_deg" for foot in ("fr", "fl", "rr", "rl")]
        assert th.hip_roll_labels(header) == ["fr", "fl", "rr", "rl"]
        assert th.lr_columns(header) == {"hip_roll": "hip_roll_fl_minus_fr_deg"}
        # No posture column reads as a foot's centre of pressure or a left-right offset of the ledger.
        assert th.foot_labels(POSTURE) == [] and set(th.lr_columns(POSTURE)) == {"hip_roll"}

    def test_the_pages_draw_them_with_the_reward_target_and_zero_lines(self, tmp_path):
        run = th.load_run_history(_posture_stage(tmp_path), read_std=False)
        context = th._make_context([run], marks=[], smooth=5)
        spec = {page.name: page for page in th.POSTURE_PAGES}
        head = th._page_figure(spec["11_head_posture"], context, title="t", dpi=30, width=12.0)
        mean, lowest, neck, _head_pitch = head.axes[:4]
        assert any("head_clearance reward target 0.9 m" in text.get_text() for text in mean.texts)
        assert lowest.get_legend() is not None  # the worst episode's minimum, dashed
        assert 0.0 in [line.get_ydata()[0] for line in neck.lines if len(line.get_ydata()) == 2]
        hips = th._page_figure(spec["12_hip_roll"], context, title="t", dpi=30, width=12.0).axes[0]
        assert [text.get_text() for text in hips.get_legend().get_texts()][:2] == ["foot r", "foot l"]
        asymmetry = {page.name: page for page in th.PAGES}["09_asymmetry"]
        roll = th._page_figure(asymmetry, context, title="t", dpi=30, width=12.0).axes[3]
        assert roll.get_title(loc="left").startswith("Hip roll: left minus right (each leg + abduction)")
        assert not any("not recorded" in text.get_text() for text in roll.texts)

    def test_an_older_ledger_renders_them_as_labelled_absent_series(self, tmp_path):
        run = th.load_run_history(_stage(tmp_path), read_std=False)
        assert run.hip_roll_feet == [] and "hip_roll" not in run.lr and run.head_target is None
        assert f"{th.EVAL_EPISODES_FILENAME}: column head_tip_height_mean_m" in run.absent
        assert f"{th.EVAL_EPISODES_FILENAME}: column l_hip_roll_mean_deg" in run.absent
        assert any("left-minus-right hip roll" in entry for entry in run.absent)
        pages, errors = th.render_pages([run], title="t", marks=[], dpi=30)
        assert errors == [] and [page.name for page in pages] == th.page_names()
        captions = {page.name: page.caption for page in pages}
        assert (
            "this ledger records none" in captions["11_head_posture"] and "none recorded" in captions["11_head_posture"]
        )
        context = th._make_context([run], marks=[], smooth=5)
        for spec in th.POSTURE_PAGES:
            figure = th._page_figure(spec, context, title="t", dpi=30, width=12.0)
            assert all(
                any("not recorded" in t.get_text() for t in axis.texts) for axis in figure.axes[: len(spec.quantities)]
            )
        asymmetry = th._page_figure(th.PAGES[8], context, title="t", dpi=30, width=12.0).axes[3]
        assert any("hip_roll_mean_deg" in text.get_text() for text in asymmetry.texts)

    def test_a_comparison_of_an_older_and_a_newer_ledger(self, tmp_path):
        runs = [
            th.load_run_history(_stage(tmp_path, name="old"), label="old", read_std=False),
            th.load_run_history(_posture_stage(tmp_path, name="new"), label="new", read_std=False),
        ]
        pages, errors = th.render_pages(runs, title="t", marks=[], dpi=30)
        caption = {page.name: page for page in pages}["12_hip_roll"].caption
        assert errors == [] and "(old records none" in caption and "so its series are absent" in caption

    def test_a_column_the_model_cannot_measure_is_drawn_as_absent_with_its_reason(self, tmp_path):
        """A model without a neck or head pitch hinge writes nan there on every episode: the panel says why, the
        captions name the columns, and the JSON's absent lists them, rather than an empty panel with a legend."""
        nan = math.nan
        run = th.load_run_history(
            _posture_stage(tmp_path, constant={"neck_pitch_mean_deg": nan, "head_pitch_mean_deg": nan}),
            read_std=False,
        )
        assert run.posture_unmeasured == ["neck_pitch_mean_deg", "head_pitch_mean_deg"]
        assert f"{th.EVAL_EPISODES_FILENAME}: column neck_pitch_mean_deg is nan on every episode" in " ".join(
            run.absent
        )
        context = th._make_context([run], marks=[], smooth=5)
        head = th._page_figure(th.POSTURE_PAGES[0], context, title="t", dpi=30, width=12.0)
        mean, _lowest, neck, pitch = head.axes[:4]
        for axis, column in ((neck, "neck_pitch_mean_deg"), (pitch, "head_pitch_mean_deg")):
            texts = [text.get_text() for text in axis.texts]
            assert any(f"nan on every episode ({column}): the model has no such site or hinge" in t for t in texts)
            assert axis.get_legend() is None and not [line for line in axis.lines if line.get_label() == "mean"]
        assert not any("nan on every episode" in text.get_text() for text in mean.texts)
        pages, errors = th.render_pages([run], title="t", marks=[], dpi=30)
        caption = {page.name: page for page in pages}["11_head_posture"].caption
        assert (
            errors == []
            and "this ledger reads nan on every episode in neck_pitch_mean_deg, head_pitch_mean_deg" in (caption)
        )
        # Beside a run that measures it, the column draws that run's line and no end label for the other.
        other = th.load_run_history(_posture_stage(tmp_path, name="measured"), label="measured", read_std=False)
        run.label = "unmeasured"
        neck = th._page_figure(
            th.POSTURE_PAGES[0], th._make_context([run, other], marks=[], smooth=5), title="t", dpi=30, width=12.0
        ).axes[2]
        labels = [text.get_text() for text in neck.texts]
        assert "measured" in labels and "unmeasured" not in labels

    def test_a_near_constant_series_keeps_a_readable_span_without_an_offset(self, tmp_path):
        run = th.load_run_history(
            _posture_stage(tmp_path, constant={"head_tip_height_mean_m": 0.325, "neck_pitch_mean_deg": 12.0}),
            read_std=False,
        )
        figure = th._page_figure(
            th.POSTURE_PAGES[0], th._make_context([run], marks=[], smooth=5), title="t", dpi=30, width=12.0
        )
        mean, _lowest, neck, _pitch = figure.axes[:4]
        low, high = mean.get_ylim()
        assert high - low >= 0.05 - 1e-12 and low < 0.325 < high
        assert neck.get_ylim()[1] - neck.get_ylim()[0] >= 1.0 - 1e-12
        assert mean.yaxis.get_major_formatter().get_useOffset() is False


def _judge_file(directory: Path, name: str, *, joint: int, own: int, seed: int = 42, **extra: Any) -> None:
    """A judge file as the stance judges write one (the keys the readout reads), joint/own falls of 20."""
    record = {
        "schema": "stance5r-judge/v1",
        "mode": "full",
        "smoke": False,
        "noise_run": True,
        "seed": seed,
        "steps": name[len("judge_") : -len(".json")],
        "noise_matched_control_joint": {"n": 20, "falls": joint, "space": "joint"},
        "noise_matched_own": {"n": 20, "falls": own, "space": "policy"},
        **extra,
    }
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(record), encoding="utf-8")


class TestJudgeK3:
    def _judged(self, tmp_path: Path) -> Path:
        judged = tmp_path / "judged" / "run_a"
        _judge_file(judged, "judge_4194304.json", joint=2, own=0)
        _judge_file(judged, "judge_6291456.json", joint=3, own=0)
        _judge_file(judged, "judge_final.json", joint=1, own=1, checks={"checkpoint_num_timesteps": 12_001_280})
        _judge_file(judged, "judge_8388608.json", joint=0, own=0, smoke=True)
        _judge_file(judged, "judge_2097152.json", joint=0, own=0, seed=70)
        _judge_file(judged, "judge_best.json", joint=0, own=0)
        _judge_file(judged, "judge_quick_4718592.json", joint=0, own=0)
        return judged

    def test_the_readout_reads_each_judged_checkpoint_against_k3s_bars(self, tmp_path):
        k3 = th.read_judge_k3(self._judged(tmp_path), seed=42, final_step=None)
        assert [(p["timesteps"], p["joint_falls"], p["own_falls"], p["k3_slot"]) for p in k3["checkpoints"]] == [
            (4194304, 2, 0, False),
            (6291456, 3, 0, True),
            (12_001_280, 1, 1, True),
        ]
        six, final = k3["checkpoints"][1:]
        assert (six["joint_within_bar"], six["own_within_bar"]) == (False, True)
        assert (final["joint_within_bar"], final["own_within_bar"]) == (True, True)
        assert "joint_within_bar" not in k3["checkpoints"][0]  # K3 does not read 4,194,304
        assert {entry["file"] for entry in k3["skipped"]} == {"judge_2097152.json", "judge_8388608.json"}
        assert set(k3["not_read"]) == {"judge_best.json", "judge_quick_4718592.json"}
        assert k3["criterion"] == th.K3_CRITERION and "never a verdict" in k3["readout"]
        assert "why" not in k3 and "note" not in k3

    def test_only_the_stance_judges_files_are_read_and_each_is_named_by_its_judge(self, tmp_path):
        """stance3 to stance5 and the replication fix the same K3: their files are read and named by their own judge;
        a file of any other schema is skipped, since its K3 need not be the one drawn."""
        judged = tmp_path / "judged"
        _judge_file(judged, "judge_6291456.json", joint=0, own=0, schema="stance5-judge/v1")
        _judge_file(judged, "judge_final.json", joint=1, own=0, schema="stance4-judge/v1")
        _judge_file(judged, "judge_8388608.json", joint=0, own=0, schema="r10-judge/v1")
        k3 = th.read_judge_k3(judged, seed=42, final_step=12_001_280)
        assert [(p["file"], p["judge"]) for p in k3["checkpoints"]] == [
            ("judge_6291456.json", "the stance5 judge"),
            ("judge_final.json", "the stance4 judge"),
        ]
        assert k3["schemas"] == ["stance4-judge/v1", "stance5-judge/v1"]
        assert [entry["file"] for entry in k3["skipped"]] == ["judge_8388608.json"]
        assert "judge schema 'r10-judge/v1': its K3 is not the one drawn here" in k3["skipped"][0]["why"]
        run = th.load_run_history(_stage(tmp_path), read_std=False, judged=judged)
        panel = th._page_figure(
            th.PAGES[0], th._make_context([run], marks=[], smooth=5), title="t", dpi=30, width=12.0
        ).axes[4]
        title = panel.get_title(loc="left").replace("\n", " ")
        assert "(files of the stance4 judge and the stance5 judge)" in title
        assert "against the stance5 replication judge's K3 criterion" in title

    def test_within_bar_flags_are_withheld_where_a_panel_is_not_k3s(self, tmp_path):
        """K3's bars are set on 20 episodes on spawns 3042-3061, as the scorers require: a panel of another size or
        spawns, or whose falls disagree with its episodes, keeps its counts but gives no within-bar flag."""
        judged = tmp_path / "judged"
        detail = [{"seed": seed, "reached_horizon": seed != 3042} for seed in range(3042, 3062)]
        _judge_file(
            judged, "judge_6291456.json", joint=0, own=0, noise_matched_own={"n": 10, "falls": 0, "space": "policy"}
        )
        _judge_file(
            judged,
            "judge_final.json",
            joint=0,
            own=0,
            noise_matched_control_joint={"n": 20, "falls": 0, "seeds": [3042, 3061], "episodes_detail": detail},
        )
        _judge_file(
            judged,
            "judge_4194304.json",
            joint=1,
            own=0,
            noise_matched_control_joint={"n": 20, "falls": 1, "seeds": [3042, 3061], "episodes_detail": detail},
        )
        six, final = (p for p in th.read_judge_k3(judged, final_step=12_001_280)["checkpoints"] if p["k3_slot"])
        assert six["joint_within_bar"] is None and six["own_within_bar"] is None
        assert "noise_matched_own: n 10, not K3's 20" in six["within_bar_withheld"]
        assert final["joint_within_bar"] is None and "falls 0, but 1 of its episodes" in final["within_bar_withheld"]
        early = th.read_judge_k3(judged, final_step=12_001_280)["checkpoints"][0]
        assert early["timesteps"] == 4194304 and "panel_issues" not in early  # recorded falls agree with its episodes
        _judge_file(
            judged, "judge_final.json", joint=0, own=0, noise_matched_own={"n": 20, "falls": 0, "seeds": [3062, 3081]}
        )
        final = th.read_judge_k3(judged, final_step=12_001_280)["checkpoints"][-1]
        assert "spawns [3062, 3081], not K3's 3042-3061" in final["within_bar_withheld"]
        record = th.run_record(th.load_run_history(_stage(tmp_path), read_std=False, judged=judged), [])
        assert "withheld: noise_matched_own" in th._k3_table(record)

    def test_the_final_slot_without_a_recorded_step_sits_at_the_runs_final(self, tmp_path):
        judged = tmp_path / "judged"
        _judge_file(judged, "judge_final.json", joint=0, own=0)
        assert th.read_judge_k3(judged, final_step=12_001_280)["checkpoints"][0]["timesteps"] == 12_001_280
        assert th.read_judge_k3(judged)["checkpoints"][0]["timesteps"] is None
        assert "no such directory" in th.read_judge_k3(tmp_path / "nowhere")["why"]

    def test_the_outcomes_page_draws_it_only_when_judge_files_are_named(self, tmp_path):
        stage = _stage(tmp_path)
        plain = th.load_run_history(stage, read_std=False)
        judged = th.load_run_history(stage, read_std=False, judged=self._judged(tmp_path))
        outcomes = th.PAGES[0]
        assert len(th._panels(outcomes, th._make_context([plain], marks=[], smooth=5))) == 4
        figure = th._page_figure(
            outcomes, th._make_context([judged], marks=[], smooth=5), title="t", dpi=30, width=12.0
        )
        panel = figure.axes[4]
        assert panel.get_title(loc="left").startswith("Judge K3 readout, never a verdict")
        legend = [text.get_text() for text in panel.get_legend().get_texts()]
        assert legend == [
            "joint-space noise falls (K3)",
            "joint-space noise falls (report only)",
            "own-sigma falls (K3)",
            "own-sigma falls (report only)",
        ]
        texts = [text.get_text() for text in panel.texts]
        assert "3/0" in texts and "1/1" in texts and "2/0" not in texts  # counts written at K3's checkpoints only
        # One label for both bars (two overprint once the falls axis spans more than a few falls), and the counts
        # boxed over the marks and that label.
        assert [text for text in texts if text.startswith("K3 bar")] == [
            "K3 bars: joint <= 2 (dashed), own <= 1 (dotted)"
        ]
        label = next(text for text in panel.texts if text.get_text().startswith("K3 bars"))
        counts = [text for text in panel.texts if text.get_text() in ("3/0", "1/1")]
        assert all(text.get_bbox_patch() is not None and text.get_zorder() > label.get_zorder() for text in counts)
        caption = th.render_pages([judged], title="t", marks=[], dpi=30)[0][0].caption
        assert "readout of the stance5 replication judge's K3 criterion" in caption and "never a verdict" in caption

    def test_a_comparison_draws_one_strip_cell_per_judged_checkpoint(self, tmp_path):
        runs = [
            th.load_run_history(_stage(tmp_path, name="a"), label="a", read_std=False, judged=self._judged(tmp_path)),
            th.load_run_history(_stage(tmp_path, name="b"), label="b", read_std=False),
        ]
        figure = th._page_figure(th.PAGES[0], th._make_context(runs, marks=[], smooth=5), title="t", dpi=30, width=12.0)
        strip = figure.axes[4]
        assert len([patch for patch in strip.patches if patch.get_fill()]) == 3
        assert len([patch for patch in strip.patches if not patch.get_fill()]) == 2  # K3's two checkpoints
        texts = [text.get_text() for text in strip.texts]
        assert {"2/0", "3/0", "1/1", "no judge files"} <= set(texts)
        # The final checkpoint's cell (K3's) is drawn whole, not cut at the axes' right edge.
        right = max(patch.get_x() + patch.get_width() for patch in strip.patches)
        assert right <= strip.get_xlim()[1]

    def test_the_json_html_and_table_carry_it(self, tmp_path, capsys):
        stage = _stage(tmp_path)
        out = tmp_path / "reports" / "k3"
        judged = self._judged(tmp_path)
        argv = [str(stage), "--out", str(out), "--no-std", "--dpi", "30", "--mark-steps", str(4194304)]
        assert script.main([*argv, "--judged", str(judged)]) == 0, capsys.readouterr().err
        document = json.loads((out / th.JSON_FILENAME).read_text(), parse_constant=_no_constant)
        record = document["runs"][0]
        assert [p["joint_falls"] for p in record["judge_k3"]["checkpoints"]] == [2, 3, 1]
        assert "judge_k3" in document["rules"] and record["judge_k3"]["source"] == str(judged)
        text = (out / th.HTML_FILENAME).read_text(encoding="utf-8")
        assert "judge K3 readout" in text and "never a verdict" in text and "judge_6291456.json" in text
        # The stage-end report (no judge files) and a run given "-" draw none.
        assert script.main([*argv, "--judged", "-"]) == 0
        document = json.loads((out / th.JSON_FILENAME).read_text(), parse_constant=_no_constant)
        assert document["runs"][0]["judge_k3"] is None
        with pytest.raises(SystemExit):
            script.main([*argv, "--judged", str(judged), str(judged)])

    @pytest.mark.parametrize("group", ["new", "stance5"])
    def test_the_readout_of_the_runs_the_repo_records_agrees_with_their_scored_k3(self, tmp_path, group):
        """The replication's score of record (results/scores/score_final.json.gz) and the K3 mechanism readouts
        (k3readouts/out) hold each judge file's recorded falls: the readout of files carrying them reproduces the
        scorer's K3 values at 6,291,456 and the final, and its within-bar flags agree with the scorer's K3 pass."""
        import gzip

        from environments.shared.paths import REPOSITORY_ROOT

        root = REPOSITORY_ROOT / "docs" / "investigations" / "stance_2026_10" / "stance5" / "replication"
        with gzip.open(root / "results" / "scores" / "score_final.json.gz", "rt", encoding="utf-8") as handle:
            score = json.load(handle)
        for run_id, run in score[group].items():
            judged = tmp_path / run_id
            for cell in run["cells"].values():
                if not cell:
                    continue
                name = Path(cell["path"]).name
                extra = {"checks": {"checkpoint_num_timesteps": 12_001_280}} if name == "judge_final.json" else {}
                _judge_file(
                    judged,
                    name,
                    joint=cell["noise_joint_falls_recorded"],
                    own=cell["noise_own_falls_recorded"],
                    seed=run["seed"],
                    **extra,
                )
            k3 = th.read_judge_k3(judged, seed=run["seed"])
            slots = {
                ("6.3M" if p["timesteps"] == th.K3_SLOT_STEPS else "final"): p
                for p in k3["checkpoints"]
                if p["k3_slot"]
            }
            recorded = run["criteria"]["K3"]
            assert {slot: p["joint_falls"] for slot, p in slots.items()} == recorded["joint_falls"], run_id
            assert {slot: p["own_falls"] for slot, p in slots.items()} == recorded["own_falls"], run_id
            within = all(p["joint_within_bar"] and p["own_within_bar"] for p in slots.values())
            assert within == recorded["pass"], run_id
            for slot, steps in (("6.3M", "6291456"), ("final", "final")):
                readout = root / "k3readouts" / "out" / run_id / f"k3r_{steps}.json"
                for candidate in sorted(readout.parent.glob(f"k3r_{steps}*.json")) if readout.parent.is_dir() else []:
                    full = json.loads(candidate.read_text(encoding="utf-8"))["conditions"]["full"]
                    assert full["judge_falls"] == slots[slot]["joint_falls"], (run_id, candidate.name)


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
        assert [p.stem for p in pngs] == th.page_names()
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
        assert errors == [] and [page.name for page in pages] == th.page_names()
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

    def test_a_comparisons_run_labels_sit_at_their_lines_ends(self, tmp_path):
        """Each run's name sits at its line's last value under the panel's final view limits (the lines only request
        an autoscale; reading the limits before it is applied put the labels where earlier limits would). No alarm
        and no mark: their vertical lines happen to apply the autoscale first."""
        runs = [
            th.load_run_history(
                _posture_stage(tmp_path, name=name, constant={"neck_pitch_mean_deg": neck}, saturated=[0, 0, 0, 0]),
                label=name,
                read_std=False,
            )
            for name, neck in (("low", 12.0), ("high", 33.0))
        ]
        figure = th._page_figure(
            th.POSTURE_PAGES[0], th._make_context(runs, marks=[], smooth=5), title="t", dpi=30, width=12.0
        )
        neck = figure.axes[2]
        figure.canvas.draw()
        low, high = neck.get_ylim()
        labels = {text.get_text(): text.get_position()[1] for text in neck.texts}
        for name, value in (("low", 12.0), ("high", 33.0)):
            assert abs(labels[name] - (value - low) / (high - low)) < 0.1, (name, labels, (low, high))

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
