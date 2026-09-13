"""Tests for environments.shared.scripts.backfill_gate_verdict — re-deriving a verdict from evidence.

Decision D-A6: a stage directory judged before Phase A carries no
``gate_verdict.json``; the tool re-derives one from the evidence the
directory already holds, refusing every missing input.  Decision D-A22: the
verdict records the gate configuration it was judged under — the block
``stage_config.json`` recorded by default (``--gate recorded``), the
checkout's current block under ``--gate current`` — so reuse rule 7 can
compare it.  A pre-D-A22 verdict (no ``gate_sha256``) is re-derived with
``--force``.

SB3-free: the handoff pair is a pair of opaque files (the tool hashes them,
it never loads them) and the evidence is a hand-written
``evaluation_selected.csv`` bound to that checkpoint's digest.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import pytest

from environments.shared.config import load_all_stages
from environments.shared.curriculum.gate_schema import gate_config_sha256, gate_config_view
from environments.shared.result_bundle import GATE_VERDICT_FILENAME, read_gate_verdict, sha256_file
from environments.shared.scripts.backfill_gate_verdict import (
    BACKFILL_JUDGED_BY,
    SELECTED_EVIDENCE_CSV,
    BackfillError,
    backfill_gate_verdict,
    main,
)

TASK = "sha256:" + "5" * 64

#: The block the fixture directory recorded as its ``[curriculum]``: a
#: ``reward_and_length/v1`` gate the evidence below passes, plus schedule
#: and collapse keys that are not the gate.
RECORDED_CURRICULUM: dict[str, Any] = {
    "gate_kind": "reward_and_length/v1",
    "gate_schema_version": 1,
    "timesteps": 1_000,
    "min_avg_reward": 100,
    "min_avg_episode_length": 500,
    "required_consecutive": 3,
    "collapse_patience": 10,
}


def _stage_dir(tmp_path: Path, *, curriculum: dict[str, Any] | None = None, evidence: bool = True) -> Path:
    """A velociraptor stance directory as a pre-Phase-A run left it: config, handoff pair, evidence."""
    stage_dir = tmp_path / "20260801_120000" / "stage1"
    models = stage_dir / "models"
    models.mkdir(parents=True)
    checkpoint = models / "best_model.zip"
    checkpoint.write_bytes(b"policy weights")
    normalization = models / "best_model_vecnorm.pkl"
    normalization.write_bytes(b"normalisation statistics")
    (stage_dir / "stage_config.json").write_text(
        json.dumps(
            {
                "species": "velociraptor",
                "stage": 1,
                "name": "Balance",
                "description": "Stand",
                "reward_weights": {"forward_vel_weight": 0.0},
                "curriculum": dict(curriculum if curriculum is not None else RECORDED_CURRICULUM),
                "task_fingerprint": {"schema": "mesozoic.task-fingerprint/v2", "task_sha256": TASK},
                "run": {"seed": 42, "timesteps": 1_000},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if evidence:
        with (stage_dir / SELECTED_EVIDENCE_CSV).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "episode",
                    "reward",
                    "length",
                    "mean_forward_velocity",
                    "distance_traveled",
                    "task_success",
                    "checkpoint_sha256",
                    "normalization_sha256",
                ],
            )
            writer.writeheader()
            for episode, reward in enumerate((180.0, 220.0, 200.0)):
                writer.writerow(
                    {
                        "episode": episode,
                        "reward": reward,
                        "length": 1000,
                        "mean_forward_velocity": 0.0,
                        "distance_traveled": 0.0,
                        "task_success": "False",
                        "checkpoint_sha256": sha256_file(checkpoint),
                        "normalization_sha256": sha256_file(normalization),
                    }
                )
    return stage_dir


#: The trex stance block as it was declared on the day the certified run
#: ``20260810_145546`` trained: the rail sat at 1940.0 before commit
#: 8d26376 moved it to 2100.0 the same day.
STANCE_CURRICULUM_AT_1940: dict[str, Any] = {
    "gate_kind": "stance_quality/v1",
    "gate_schema_version": 1,
    "timesteps": 11_000_000,
    "min_full_horizon_fraction": 0.95,
    "max_unsupported_duty": 0.02,
    "max_unsupported_duty_ucb": 0.02,
    "settle_steps": 200,
    "min_eval_episodes": 40,
    "min_avg_reward": 1940.0,
    "required_consecutive": 3,
}


def _stance_report(curriculum: dict[str, Any], *, mean_reward: float = 1950.0, thresholds: bool = True) -> dict:
    """A ``stance_gate_report.json`` as ``reporting.stance_report`` writes it: a PASS scored under *curriculum*."""
    report: dict[str, Any] = {
        "schema": "mesozoic.stance-gate-report/v2",
        "species": "trex",
        "stage": 1,
        "gate_kind": "stance_quality/v1",
        "episodes": 40,
        "passed": True,
        "failures": [],
        "metrics": {
            "reward_mean": mean_reward,
            "full_horizon_fraction": 1.0,
            "mean_unsupported_duty": 0.001,
            "unsupported_duty_ucb": 0.002,
            "n_duty_episodes": 40,
        },
    }
    if thresholds:
        report["thresholds"] = {
            key: curriculum[key]
            for key in (
                "min_full_horizon_fraction",
                "max_unsupported_duty",
                "max_unsupported_duty_ucb",
                "min_avg_reward",
                "min_eval_episodes",
            )
        }
    return report


def _stance_stage_dir(tmp_path: Path, *, recorded: dict[str, Any], report: dict[str, Any]) -> Path:
    """A trex 01_stance directory as the certified run left it: config, handoff pair, stance report."""
    stage_dir = tmp_path / "20260810_145546" / "01_stance"
    models = stage_dir / "models"
    models.mkdir(parents=True)
    (models / "robust_best_model.zip").write_bytes(b"policy weights")
    (models / "robust_best_model_vecnorm.pkl").write_bytes(b"normalisation statistics")
    (stage_dir / "stage_config.json").write_text(
        json.dumps(
            {
                "species": "trex",
                "stage": 1,
                "name": "Balance",
                "description": "Stand",
                "reward_weights": {"forward_vel_weight": 0.0},
                "curriculum": dict(recorded),
                "task_fingerprint": {"schema": "mesozoic.task-fingerprint/v2", "task_sha256": TASK},
                "run": {"seed": 42, "timesteps": 11_000_000},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (stage_dir / "stance_gate_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return stage_dir


def _phase_a_verdict(stage_dir: Path) -> dict[str, Any]:
    """Backfill once, then strip the D-A22 fields: the file a pre-D-A22 backfill left behind."""
    backfill_gate_verdict(stage_dir)
    path = stage_dir / GATE_VERDICT_FILENAME
    phase_a = {
        k: v for k, v in json.loads(path.read_text(encoding="utf-8")).items() if k not in {"gate", "gate_sha256"}
    }
    path.write_text(json.dumps(phase_a, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return phase_a


class TestBackfill:
    def test_a_reward_gate_is_re_derived_from_hash_bound_evidence_and_records_the_recorded_gate(self, tmp_path, caplog):
        stage_dir = _stage_dir(tmp_path)
        with caplog.at_level(logging.INFO):
            written = backfill_gate_verdict(stage_dir)

        assert written == stage_dir / GATE_VERDICT_FILENAME
        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None
        assert verdict["passed"] is True and verdict["failures"] == []
        assert (verdict["species"], verdict["stage"], verdict["stage_id"]) == ("velociraptor", 1, "stance")
        assert verdict["gate_kind"] == "reward_and_length/v1" and verdict["gate_schema_version"] == 1
        assert verdict["judged_by"] == BACKFILL_JUDGED_BY
        assert verdict["checkpoint"] == "models/best_model.zip"
        assert verdict["checkpoint_sha256"] == sha256_file(stage_dir / "models" / "best_model.zip")
        assert verdict["normalization_sha256"] == sha256_file(stage_dir / "models" / "best_model_vecnorm.pkl")
        assert verdict["task_sha256"] == TASK
        assert verdict["stage_result"]["best_model_reward"] == pytest.approx(200.0)
        assert verdict["stage_result"]["gate_passed"] is True
        # D-A22: the verdict digests what the directory actually ran under —
        # the recorded block's gate keys only (schedule and collapse excluded).
        assert verdict["gate"] == gate_config_view(RECORDED_CURRICULUM)
        assert verdict["gate"]["thresholds"] == {
            "min_avg_episode_length": 500,
            "min_avg_reward": 100,
            "required_consecutive": 3,
        }
        assert verdict["gate_sha256"] == gate_config_sha256(gate_config_view(RECORDED_CURRICULUM))
        assert "judged under the recorded gate configuration" in caplog.text

    def test_an_existing_verdict_is_never_overwritten_without_force(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        phase_a = _phase_a_verdict(stage_dir)
        assert "gate_sha256" not in phase_a and read_gate_verdict(stage_dir) == phase_a

        with pytest.raises(BackfillError, match="already exists; pass --force"):
            backfill_gate_verdict(stage_dir)
        assert read_gate_verdict(stage_dir) == phase_a

    def test_force_re_derives_a_pre_d_a22_verdict_with_its_gate_digest(self, tmp_path):
        """The re-judge path for every pre-D-A22 directory: ``--force`` re-derives it, and the new
        file carries the gate it was judged under, which reuse rule 7 needs."""
        stage_dir = _stage_dir(tmp_path)
        phase_a = _phase_a_verdict(stage_dir)

        backfill_gate_verdict(stage_dir, force=True)

        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None
        assert verdict["gate_sha256"] == gate_config_sha256(gate_config_view(RECORDED_CURRICULUM))
        assert verdict["gate"] == gate_config_view(RECORDED_CURRICULUM)
        assert {k: v for k, v in verdict.items() if k not in {"gate", "gate_sha256", "judged_at"}} == {
            k: v for k, v in phase_a.items() if k != "judged_at"
        }

    def test_gate_current_judges_and_digests_the_checkouts_block(self, tmp_path, caplog):
        """``--gate current``: the re-judge-after-a-threshold-edit path.  The verdict is judged
        under the checkout's velociraptor stance block (whose floors this evidence does not clear)
        and records THAT gate — not the block the directory trained under."""
        stage_dir = _stage_dir(tmp_path)
        current = load_all_stages("velociraptor")[1]["curriculum_kwargs"]
        assert current["gate_kind"] == "reward_and_length/v1"
        assert gate_config_sha256(gate_config_view(current)) != gate_config_sha256(
            gate_config_view(RECORDED_CURRICULUM)
        )

        with caplog.at_level(logging.INFO):
            backfill_gate_verdict(stage_dir, gate="current")

        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None
        assert verdict["gate"] == gate_config_view(current)
        assert verdict["gate_sha256"] == gate_config_sha256(gate_config_view(current))
        assert verdict["passed"] is False
        assert any("best model reward 200.00 <" in failure for failure in verdict["failures"])
        assert "under the checkout's current velociraptor 'stance' gate" in caplog.text
        # The recorded block on disk is untouched: only the verdict names the gate judged under.
        recorded = json.loads((stage_dir / "stage_config.json").read_text(encoding="utf-8"))["curriculum"]
        assert recorded == RECORDED_CURRICULUM
        # Judged under the recorded gate again, it passes and records that gate.
        backfill_gate_verdict(stage_dir, force=True, gate="recorded")
        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None and verdict["passed"] is True
        assert verdict["gate_sha256"] == gate_config_sha256(gate_config_view(RECORDED_CURRICULUM))

    def test_an_unknown_gate_source_is_a_caller_error(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        with pytest.raises(ValueError, match="gate must be one of"):
            backfill_gate_verdict(stage_dir, gate="edited")
        assert read_gate_verdict(stage_dir) is None

    def test_gate_current_fails_closed_when_the_checkout_has_no_gate_for_the_stage(self, tmp_path, monkeypatch):
        """No current block is a refusal, never a fallback to the recorded one."""
        from environments.shared import config as config_module

        stage_dir = _stage_dir(tmp_path)
        monkeypatch.setattr(config_module, "load_all_stages", lambda species: {2: {"curriculum_kwargs": {}}})
        with pytest.raises(BackfillError, match="declares no stage 1"):
            backfill_gate_verdict(stage_dir, gate="current")
        monkeypatch.setattr(config_module, "load_all_stages", lambda species: {1: {"curriculum_kwargs": {}}})
        with pytest.raises(BackfillError, match="declares no \\[curriculum\\] block"):
            backfill_gate_verdict(stage_dir, gate="current")

        def broken(species):
            raise RuntimeError("no such species config")

        monkeypatch.setattr(config_module, "load_all_stages", broken)
        with pytest.raises(BackfillError, match="cannot be loaded: no such species config"):
            backfill_gate_verdict(stage_dir, gate="current")
        assert read_gate_verdict(stage_dir) is None

    def test_a_recovery_gate_is_refused(self, tmp_path):
        stage_dir = _stage_dir(
            tmp_path,
            curriculum={
                "gate_kind": "recovery_quality/v1",
                "gate_schema_version": 1,
                "min_recovery_success_lcb": 0.3,
                "recovery_t_recover_steps": 100,
                "recovery_dwell_steps": 50,
            },
        )
        with pytest.raises(BackfillError, match="never persisted"):
            backfill_gate_verdict(stage_dir, species="trex", stage="recovery")
        assert read_gate_verdict(stage_dir) is None

    def test_missing_evidence_is_refused(self, tmp_path):
        stage_dir = _stage_dir(tmp_path, evidence=False)
        with pytest.raises(BackfillError, match="neither evaluation_selected.csv nor evaluations.npz"):
            backfill_gate_verdict(stage_dir)
        assert read_gate_verdict(stage_dir) is None

    def test_evidence_bound_to_another_checkpoint_is_refused(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        (stage_dir / "models" / "best_model.zip").write_bytes(b"rewritten after the evidence")
        with pytest.raises(BackfillError, match="not bound to the selected handoff checkpoint"):
            backfill_gate_verdict(stage_dir)
        assert read_gate_verdict(stage_dir) is None

    def test_a_missing_curriculum_block_is_refused(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        config_path = stage_dir / "stage_config.json"
        record = json.loads(config_path.read_text(encoding="utf-8"))
        del record["curriculum"]
        config_path.write_text(json.dumps(record), encoding="utf-8")
        with pytest.raises(BackfillError, match="records no curriculum block"):
            backfill_gate_verdict(stage_dir)
        # The loaded-config spelling of the block is read too.
        record["curriculum_kwargs"] = RECORDED_CURRICULUM
        config_path.write_text(json.dumps(record), encoding="utf-8")
        assert read_gate_verdict(backfill_gate_verdict(stage_dir).parent)["gate"] == gate_config_view(
            RECORDED_CURRICULUM
        )


class TestStanceReports:
    """A stance verdict is read off the report's ``passed`` — the judge re-derives nothing — so the
    report certifies only the thresholds it scored.  The rail moved 1940.0 -> 2100.0 the day the
    certified run trained; a report scored under 1940.0 must never be minted as a pass under 2100.0."""

    def test_a_report_scored_under_the_recorded_gate_backfills_under_it(self, tmp_path):
        stage_dir = _stance_stage_dir(
            tmp_path, recorded=STANCE_CURRICULUM_AT_1940, report=_stance_report(STANCE_CURRICULUM_AT_1940)
        )
        backfill_gate_verdict(stage_dir)
        verdict = read_gate_verdict(stage_dir)
        assert verdict is not None and verdict["passed"] is True and verdict["gate_kind"] == "stance_quality/v1"
        assert verdict["gate"] == gate_config_view(STANCE_CURRICULUM_AT_1940)
        assert verdict["gate"]["thresholds"]["min_avg_reward"] == 1940.0
        assert verdict["gate_sha256"] == gate_config_sha256(gate_config_view(STANCE_CURRICULUM_AT_1940))

    def test_gate_current_refuses_a_report_scored_under_another_rail(self, tmp_path):
        """The one real-world use of ``--gate current`` on a stance directory: refused, never a false
        certificate under the moved rail (the panel's 1950 would fail the 2100 rail)."""
        current = load_all_stages("trex")[1]["curriculum_kwargs"]
        assert current["gate_kind"] == "stance_quality/v1" and current["min_avg_reward"] == 2100.0
        stage_dir = _stance_stage_dir(
            tmp_path, recorded=STANCE_CURRICULUM_AT_1940, report=_stance_report(STANCE_CURRICULUM_AT_1940)
        )
        with pytest.raises(BackfillError) as excinfo:
            backfill_gate_verdict(stage_dir, gate="current", force=True)
        message = str(excinfo.value)
        assert "min_avg_reward: report scored at 1940.0, judging under 2100.0 now" in message
        assert "notebook JUDGE branch" in message and "current 'stance' block" in message
        assert read_gate_verdict(stage_dir) is None
        # The same evidence still backfills under the gate it was scored under.
        backfill_gate_verdict(stage_dir, gate="recorded")
        assert read_gate_verdict(stage_dir)["gate_sha256"] == gate_config_sha256(
            gate_config_view(STANCE_CURRICULUM_AT_1940)
        )
        # And under the current gate once the report itself was scored under it.
        (stage_dir / "stance_gate_report.json").write_text(
            json.dumps(_stance_report(current, mean_reward=2200.0)) + "\n", encoding="utf-8"
        )
        backfill_gate_verdict(stage_dir, gate="current", force=True)
        assert read_gate_verdict(stage_dir)["gate_sha256"] == gate_config_sha256(gate_config_view(current))

    def test_a_report_that_disagrees_with_the_recorded_block_is_refused_under_the_default_too(self, tmp_path):
        """A stale report beside an edited stage_config.json proves nothing about the recorded gate either."""
        recorded = {**STANCE_CURRICULUM_AT_1940, "max_unsupported_duty_ucb": 0.01}
        stage_dir = _stance_stage_dir(tmp_path, recorded=recorded, report=_stance_report(STANCE_CURRICULUM_AT_1940))
        with pytest.raises(
            BackfillError, match="max_unsupported_duty_ucb: report scored at 0.02, judging under 0.01 now"
        ):
            backfill_gate_verdict(stage_dir)
        assert read_gate_verdict(stage_dir) is None

    def test_a_report_without_thresholds_is_refused(self, tmp_path):
        stage_dir = _stance_stage_dir(
            tmp_path,
            recorded=STANCE_CURRICULUM_AT_1940,
            report=_stance_report(STANCE_CURRICULUM_AT_1940, thresholds=False),
        )
        with pytest.raises(BackfillError, match="records no thresholds"):
            backfill_gate_verdict(stage_dir)
        assert read_gate_verdict(stage_dir) is None

    def test_a_judged_block_without_a_complete_stance_gate_is_refused(self, tmp_path):
        recorded = {k: v for k, v in STANCE_CURRICULUM_AT_1940.items() if k != "max_unsupported_duty_ucb"}
        stage_dir = _stance_stage_dir(tmp_path, recorded=recorded, report=_stance_report(STANCE_CURRICULUM_AT_1940))
        with pytest.raises(BackfillError, match="does not declare a complete stance_quality/v1 gate"):
            backfill_gate_verdict(stage_dir)
        assert read_gate_verdict(stage_dir) is None


class TestCommandLine:
    def test_the_default_judges_under_the_recorded_gate(self, tmp_path, capsys):
        stage_dir = _stage_dir(tmp_path)
        assert main([str(stage_dir)]) == 0
        assert capsys.readouterr().out.strip() == str(stage_dir / GATE_VERDICT_FILENAME)
        assert read_gate_verdict(stage_dir)["gate_sha256"] == gate_config_sha256(gate_config_view(RECORDED_CURRICULUM))

    def test_force_and_gate_current_are_forwarded(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        _phase_a_verdict(stage_dir)
        assert main([str(stage_dir)]) == 1
        assert "gate_sha256" not in read_gate_verdict(stage_dir)
        assert main([str(stage_dir), "--force", "--gate", "current"]) == 0
        current = load_all_stages("velociraptor")[1]["curriculum_kwargs"]
        assert read_gate_verdict(stage_dir)["gate_sha256"] == gate_config_sha256(gate_config_view(current))

    def test_an_unknown_gate_source_is_rejected_by_the_parser(self, tmp_path):
        stage_dir = _stage_dir(tmp_path)
        with pytest.raises(SystemExit):
            main([str(stage_dir), "--gate", "edited"])
        assert read_gate_verdict(stage_dir) is None
