"""Guards that stop a tool, a notebook cell or a resume from writing destructively into a run or stage directory.

* A judged node is never trained further in place (decision D-A20 as amended 2026-11-24, after D-D16):
  ``config.refuse_occupied_stage_dir`` refuses a same-stage resume into a directory holding
  ``gate_verdict.json``, so ``train --load ... --load-mode resume_same_stage`` cannot rewrite the final pair and
  the handoff pair a verdict hashes; the notebook's manual-node cell refuses such a directory before it writes.
* A complete result bundle is immutable (``docs/RESULT_BUNDLES.md``): ``backfill_gate_verdict``,
  ``widen_checkpoint`` and ``harnesses/freeze_recovery_gate`` refuse to write into one, ``--force`` or
  ``--replace`` or not; the freeze harness never re-freezes a judged node's resolution either.
* A report tool never writes into a run (``result_bundle.refuse_output_inside_run``): ``stance_gate_report``'s
  ``--out-dir`` is refused inside a run directory or the checkpoint's stage directory, as ``episode_trace_report``'s,
  and so is ``harnesses/recovery_offdist_panel``'s ``--out``.
* In Colab without Drive the run lives only on the VM's disk, so the runtime is never released automatically
  (``notebook_runtime.disconnect_runtime``) and the storage cell warns. The release reads the storage cell's
  resolved ``RUNS_ON_DRIVE``, never the ``USE_GOOGLE_DRIVE`` knob, which can change after the run was placed.

New names are imported inside the tests, so each test fails on its own against code that predates its guard.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from environments.shared.model_id import ModelId

from .notebook_cells import code_cell

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "sb3_training.ipynb"
MANUAL_CELL_MARKER = "# ===== MANUAL SINGLE NODE"
STORAGE_CELL_MARKER = "# Storage Configuration"
CONFIG_CELL_MARKER = "AUTO_DISCONNECT = True"
FINAL_CELL_MARKER = 'f"Training finished — behavior {BEHAVIOR!r}'

TYRANNOSAURUS_REX = ModelId("tyrannosaurus_rex", "anatomical")
VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")


def _digests(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _write_verdict(stage_dir: Path, model_id: ModelId, *, passed: bool = True) -> None:
    """A ``gate_verdict.json`` the reader admits (gate-verdict/v2 names its model whole, decision D-D33)."""
    from environments.shared.result_bundle import GATE_VERDICT_SCHEMA

    stage_dir.mkdir(parents=True, exist_ok=True)
    (stage_dir / "gate_verdict.json").write_text(
        json.dumps(
            {
                "schema": GATE_VERDICT_SCHEMA,
                **model_id.record(),
                "passed": passed,
                "failures": [] if passed else ["too slow"],
            }
        )
    )


def _write_manifest(run_dir: Path, status: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifact_manifest.json").write_text(json.dumps({"status": status, "files": []}))


# ── a judged node is never trained further in place ─────────────────────────


class TestAJudgedStageIsNeverResumedInPlace:
    """D-A20 as amended 2026-11-24: ``gate_verdict.json`` ends a stage for every load mode, the resume included."""

    @pytest.mark.parametrize("with_config", [True, False])
    @pytest.mark.parametrize("task_load_mode", ["resume_same_stage", "initialize_next_stage", None])
    def test_every_load_mode_is_refused(self, tmp_path, task_load_mode, with_config):
        from environments.shared.config import StageDirectoryOccupiedError, refuse_occupied_stage_dir

        if with_config:
            (tmp_path / "stage_config.json").write_text("{}")
        (tmp_path / "gate_verdict.json").write_text("{}")

        with pytest.raises(StageDirectoryOccupiedError) as excinfo:
            refuse_occupied_stage_dir(tmp_path, task_load_mode=task_load_mode)

        message = str(excinfo.value)
        assert str(tmp_path) in message and "gate_verdict.json" in message and repr(task_load_mode) in message
        assert "never trained further in place" in message
        # The way forward is a new attempt in a fresh run directory, never the resume it just refused.
        assert "--output-dir" in message and "RUN_ID" in message and "TRUNK_FROM" in message
        assert "RESUME cell" not in message and "resume this stage" not in message

    def test_an_unjudged_stage_still_resumes(self, tmp_path):
        """An interrupted node (a config, no verdict) is what the resume is for."""
        from environments.shared.config import refuse_occupied_stage_dir

        (tmp_path / "stage_config.json").write_text("{}")
        assert refuse_occupied_stage_dir(tmp_path, task_load_mode="resume_same_stage") is None

    def test_train_refuses_the_cli_resume_into_a_judged_directory_before_it_writes(self, tmp_path, monkeypatch):
        """``train --load <periodic> --load-mode resume_same_stage --output-dir <judged stage>`` (the CLI's default
        load mode) used to pass the guard and re-save the stage config, then rewrite the final pair."""
        from environments.shared import config as config_module
        from environments.shared import task_fingerprint, train_base
        from environments.shared.config import StageDirectoryOccupiedError, load_all_stages

        class ConfigReached(RuntimeError):
            """The guard passed: train() went on to write the stage config."""

        def save_config(*args, **kwargs):
            raise ConfigReached()

        monkeypatch.setattr(train_base, "current_plant_identity", lambda model_id: SimpleNamespace(to_dict=dict))
        monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
        monkeypatch.setattr(task_fingerprint, "read_checkpoint_task_fingerprint", lambda path: None)
        monkeypatch.setattr(task_fingerprint, "validate_declared_parent", lambda *a, **k: None)
        monkeypatch.setattr("environments.shared.policy_loading._ensure_sb3", lambda: {})
        monkeypatch.setattr(config_module, "save_stage_config", save_config)
        judged = tmp_path / "run" / "01_stance"
        judged.mkdir(parents=True)
        (judged / "stage_config.json").write_text("{}")
        _write_verdict(judged, VELOCIRAPTOR_MONGOLIENSIS)
        before = _digests(judged)

        with pytest.raises(StageDirectoryOccupiedError, match="never trained further in place"):
            train_base.train(
                SimpleNamespace(model_id=VELOCIRAPTOR_MONGOLIENSIS, env_class=object),
                load_all_stages(VELOCIRAPTOR_MONGOLIENSIS),
                1,
                total_timesteps=1,
                load_path=str(judged / "models" / "stage1_1000_steps.zip"),
                task_load_mode="resume_same_stage",
                output_dir=str(judged),
                use_tensorboard=False,
                verbose=0,
            )
        assert _digests(judged) == before and not (judged / "models").exists()


class TestTheManualCellRefusesAJudgedNode:
    """The manual-node cell (section 7) refuses a target stage directory holding a gate verdict before anything is
    written, under every ``MANUAL_LOAD_MODE``: with ``resume_same_stage`` the library guard used to let it through,
    and ``generate_stage_artifacts`` then rewrote the verdict (a PASS could become a FAIL) and the final pair."""

    @staticmethod
    def _run(tmp_path, *, load_path, load_mode, calls):
        from environments.shared.config import load_all_stages
        from environments.shared.stage_manifest import load_stage_manifest, stage_dirname

        def train_stage(**kwargs):
            calls.append(("train_stage", kwargs))
            return None, "handoff", "final", tmp_path / stage_dirname(TYRANNOSAURUS_REX, kwargs["stage"]), None, {}

        def generate_stage_artifacts(**kwargs):
            calls.append(("generate_stage_artifacts", kwargs))
            return {"publication_gate_passed": True, "gate_failures": []}

        namespace = {
            "Path": Path,
            "MODEL_ID": TYRANNOSAURUS_REX,
            "MANIFEST": load_stage_manifest(TYRANNOSAURUS_REX),
            "STAGE_CONFIGS": load_all_stages(TYRANNOSAURUS_REX),
            "RUN_DIR": tmp_path,
            "RUN_LABEL": "",
            "SPECIES_CFG": None,
            "ALGORITHM": "ppo",
            "SEED": 42,
            "NODE_RESULTS": {},
            "completed_stages": [],
            "stage_dirname": stage_dirname,
            "node_budget": lambda stage: 1_000,
            "train_stage": train_stage,
            "generate_stage_artifacts": generate_stage_artifacts,
            "display_stage_videos": lambda stage_dir: calls.append(("display_stage_videos", stage_dir)),
            "chain_results": lambda: [],
            "write_training_summary": lambda *a, **k: calls.append(("write_training_summary", a)),
            "save_run_bundle": lambda *a, **k: calls.append(("save_run_bundle", a)),
        }
        src = code_cell(NOTEBOOK_PATH, MANUAL_CELL_MARKER)
        knobs = {
            "MANUAL_NODE = None": 'MANUAL_NODE = "stance"',
            "MANUAL_LOAD_PATH = None": f"MANUAL_LOAD_PATH = {load_path!r}",
            "MANUAL_VECNORM_PATH = None": f"MANUAL_VECNORM_PATH = {load_path and load_path + '_vecnorm.pkl'!r}",
            'MANUAL_LOAD_MODE = "initialize_next_stage"': f"MANUAL_LOAD_MODE = {load_mode!r}",
        }
        for old, new in knobs.items():
            assert src.count(old) == 1, old
            src = src.replace(old, new, 1)
        exec(compile(src, "sb3_manual", "exec"), namespace)

    @pytest.mark.parametrize("passed", [True, False])
    @pytest.mark.parametrize(
        ("load", "load_mode"),
        [
            ("own_periodic", "resume_same_stage"),
            ("parent_handoff", "initialize_next_stage"),
            (None, "initialize_next_stage"),
        ],
    )
    def test_a_judged_node_is_refused_before_anything_is_written(self, tmp_path, load, load_mode, passed):
        stage_dir = tmp_path / "01_stance"
        _write_verdict(stage_dir, TYRANNOSAURUS_REX, passed=passed)
        (stage_dir / "stage_config.json").write_text("{}")
        before = _digests(tmp_path)
        load_path = None if load is None else str(stage_dir / "models" / f"{load}_1000_steps")
        calls: list = []

        with pytest.raises(RuntimeError) as excinfo:
            self._run(tmp_path, load_path=load_path, load_mode=load_mode, calls=calls)

        message = str(excinfo.value)
        assert f"The manual node 'stance' already holds a gate verdict in {stage_dir}" in message
        assert "never trained further in place" in message and "a fresh RUN_ID" in message
        assert calls == [], "nothing is trained, judged, shown or published"
        assert _digests(tmp_path) == before

    def test_an_unjudged_node_still_trains(self, tmp_path):
        """The refusal reads the verdict, not the directory: an interrupted node resumes as before."""
        stage_dir = tmp_path / "01_stance"
        stage_dir.mkdir()
        (stage_dir / "stage_config.json").write_text("{}")
        calls: list = []

        self._run(
            tmp_path,
            load_path=str(stage_dir / "models" / "stage1_1000_steps"),
            load_mode="resume_same_stage",
            calls=calls,
        )

        [train] = [kwargs for name, kwargs in calls if name == "train_stage"]
        assert train["task_load_mode"] == "resume_same_stage" and train["run_dir"] == tmp_path

    def test_the_refusal_precedes_the_freeze_and_the_training(self):
        src = code_cell(NOTEBOOK_PATH, MANUAL_CELL_MARKER)
        tree = ast.parse(src)
        [judged] = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "read_gate_verdict(_manual_dir) is not None"
        ]
        assert isinstance(judged.body[0], ast.Raise)
        later = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and ast.unparse(node.func) in ("freeze_recovery_gate", "validate_recovery_resolution", "train_stage")
        ]
        assert later and judged.lineno < min(later)


# ── no tool writes into a finished run ──────────────────────────────────────


class TestBackfillRefusesACompleteRun:
    """``backfill_gate_verdict`` refuses a stage directory whose run holds a complete bundle, ``--force`` or not:
    a verdict added or rewritten there stopped the manifest verifying (canonical-conflict, no promotion)."""

    @staticmethod
    def _stage(tmp_path, *, status, verdict):
        run = tmp_path / "20260101_000000"
        stage_dir = run / "01_stance"
        stage_dir.mkdir(parents=True)
        (stage_dir / "stage_config.json").write_text(json.dumps({**TYRANNOSAURUS_REX.record(), "stage": "stance"}))
        if verdict:
            _write_verdict(stage_dir, TYRANNOSAURUS_REX, passed=True)
        if status is not None:
            _write_manifest(run, status)
        return run, stage_dir

    @pytest.mark.parametrize("verdict", [True, False])
    @pytest.mark.parametrize("force", [True, False])
    def test_a_complete_run_is_refused_before_anything_is_read_or_written(self, tmp_path, force, verdict):
        from environments.shared.scripts.backfill_gate_verdict import BackfillError, backfill_gate_verdict

        run, stage_dir = self._stage(tmp_path, status="complete", verdict=verdict)
        before = _digests(run)

        with pytest.raises(BackfillError) as excinfo:
            backfill_gate_verdict(stage_dir, force=force)

        message = str(excinfo.value)
        assert f"run {run.resolve()} holds a complete result bundle" in message
        assert "immutable" in message and "--force does not apply" in message
        assert _digests(run) == before

    def test_the_command_line_exits_1(self, tmp_path, caplog):
        from environments.shared.scripts.backfill_gate_verdict import main

        run, stage_dir = self._stage(tmp_path, status="complete", verdict=True)
        before = _digests(run)
        assert main([str(stage_dir), "--force"]) == 1
        assert "complete result bundle" in caplog.text
        assert _digests(run) == before

    @pytest.mark.parametrize("status", [None, "partial", "failed"])
    def test_a_run_that_is_not_complete_passes_this_guard(self, tmp_path, status):
        """The next check decides: here, the missing evidence."""
        from environments.shared.scripts.backfill_gate_verdict import BackfillError, backfill_gate_verdict

        _, stage_dir = self._stage(tmp_path, status=status, verdict=False)
        with pytest.raises(BackfillError) as excinfo:
            backfill_gate_verdict(stage_dir)
        assert "complete result bundle" not in str(excinfo.value)

    def test_an_unreadable_manifest_is_refused(self, tmp_path):
        from environments.shared.scripts.backfill_gate_verdict import BackfillError, backfill_gate_verdict

        run, stage_dir = self._stage(tmp_path, status=None, verdict=False)
        (run / "artifact_manifest.json").write_text("{not json")
        with pytest.raises(BackfillError, match="bundle status of run .* is unknown.*Restore a readable"):
            backfill_gate_verdict(stage_dir, force=True)
        assert not (stage_dir / "gate_verdict.json").exists()


class TestReportOutputStaysOutsideRuns:
    """``result_bundle.refuse_output_inside_run``: the guard ``stance_gate_report`` and ``episode_trace_report``
    share (it was ``episode_trace.refuse_run_directory_output``, which checked only the checkpoint's run)."""

    @staticmethod
    def _run(tmp_path, *, provenance=True):
        run = tmp_path / "20260101_000000"
        models = run / "01_stance" / "models"
        models.mkdir(parents=True)
        if provenance:
            (run / "provenance.json").write_text("{}")
        return run, models / "robust_best_model.zip", models / "robust_best_model_vecnorm.pkl"

    def test_any_run_directory_is_refused(self, tmp_path):
        from environments.shared.result_bundle import ResultBundleError, refuse_output_inside_run, run_directory

        run, model, vecnorm = self._run(tmp_path)
        assert run_directory(run / "01_stance" / "audit") == run.resolve()
        for out_dir in (run, run / "01_stance", run / "01_stance" / "fresh_audit", run / "elsewhere"):
            with pytest.raises(ResultBundleError, match="inside the run directory"):
                refuse_output_inside_run(out_dir)  # no checkpoint named: the zero-action reference
            with pytest.raises(ResultBundleError, match="inside the run directory"):
                refuse_output_inside_run(out_dir, [model, vecnorm])
        refuse_output_inside_run(tmp_path / "scratch" / "fresh_audit", [model, vecnorm, None])

    def test_the_checkpoints_stage_directory_is_refused_without_provenance(self, tmp_path):
        """A run older than ``provenance.json``: the stage directory, by the layout, still holds its evidence."""
        from environments.shared.result_bundle import ResultBundleError, refuse_output_inside_run

        run, model, vecnorm = self._run(tmp_path, provenance=False)
        for source in (model, vecnorm):
            with pytest.raises(ResultBundleError, match="inside the stage directory"):
                refuse_output_inside_run(run / "01_stance", [source])
        refuse_output_inside_run(run / "sibling", [model, vecnorm])

    def test_episode_trace_keeps_its_error_type(self, tmp_path):
        from environments.shared.reporting.episode_trace import EpisodeTraceError, refuse_run_directory_output

        run, model, _ = self._run(tmp_path, provenance=False)
        with pytest.raises(EpisodeTraceError, match="inside the stage directory"):
            refuse_run_directory_output(run / "01_stance" / "trace", [model])


class TestStanceGateReportOutDir:
    """``stance_gate_report.py --out-dir`` inside a run is refused before anything is loaded or rolled: the
    fresh-seed audit's ``stance_gate_report.json`` written into the stage directory replaced the stage's own gate
    evidence, and any file added to a published run stops its manifest verifying (NEXT_STEPS §3)."""

    class Loaded(RuntimeError):
        """The guard passed: the tool went on to load the stage config."""

    @pytest.fixture
    def tool(self, monkeypatch):
        from environments.shared.scripts import stance_gate_report

        def load_stage_config(*args, **kwargs):
            raise self.Loaded()

        monkeypatch.setattr(stance_gate_report, "load_stage_config", load_stage_config)
        return stance_gate_report

    @staticmethod
    def _main(tool, monkeypatch, out_dir, *checkpoint):
        """The tool as the command line runs it (``sys.argv``)."""
        argv = ["--species", "tyrannosaurus_rex", "--embodiment", "anatomical", *checkpoint, "--out-dir", str(out_dir)]
        monkeypatch.setattr(sys, "argv", ["stance_gate_report.py", *argv])
        return tool.main()

    @pytest.mark.parametrize("zero_action", [False, True])
    def test_an_out_dir_inside_a_run_is_refused(self, tool, tmp_path, monkeypatch, zero_action):
        run = tmp_path / "20260101_000000"
        models = run / "01_stance" / "models"
        models.mkdir(parents=True)
        (run / "provenance.json").write_text("{}")
        checkpoint = (
            ["--zero-action"]
            if zero_action
            else ["--model", str(models / "robust_best_model.zip"), "--vecnorm", str(models / "x_vecnorm.pkl")]
        )
        out_dir = run / "01_stance" / "fresh_audit"
        before = _digests(run)

        with pytest.raises(SystemExit) as excinfo:
            self._main(tool, monkeypatch, out_dir, *checkpoint)

        assert "inside the run directory" in str(excinfo.value.code)
        assert _digests(run) == before and not out_dir.exists()

    def test_the_stage_directory_of_a_run_without_provenance_is_refused(self, tool, tmp_path, monkeypatch):
        models = tmp_path / "stage1" / "models"
        models.mkdir(parents=True)
        model = ["--model", str(models / "robust_best_model.zip"), "--vecnorm", str(models / "x_vecnorm.pkl")]
        with pytest.raises(SystemExit) as excinfo:
            self._main(tool, monkeypatch, tmp_path / "stage1", *model)
        assert "inside the stage directory" in str(excinfo.value.code)

    def test_a_scratch_directory_passes_the_guard(self, tool, tmp_path, monkeypatch):
        run = tmp_path / "20260101_000000"
        models = run / "01_stance" / "models"
        models.mkdir(parents=True)
        (run / "provenance.json").write_text("{}")
        model = ["--model", str(models / "robust_best_model.zip"), "--vecnorm", str(models / "x_vecnorm.pkl")]
        with pytest.raises(self.Loaded):
            self._main(tool, monkeypatch, tmp_path / "scratch" / "fresh_audit", *model)


@pytest.mark.parametrize(
    ("manifest", "refusal"),
    [
        ("complete", "holds a complete result bundle, which is immutable"),
        # A manifest that cannot be read has an unknown status: never treated as writable.
        (None, r"bundle status of .* is unknown, so nothing is written there: .*Restore a readable"),
    ],
)
def test_widen_refuses_a_target_in_a_complete_run(tmp_path, manifest, refusal):
    """A widened root is written into a fresh run of its own; a complete run's manifest never listed it."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("torch")
    from environments.shared.scripts.widen_checkpoint import WidenError, widen_checkpoint
    from environments.shared.stage_manifest import stage_dirname

    from .test_widen_checkpoint import MODEL_ID, STAGE, build_narrow_parent

    parent = build_narrow_parent(tmp_path / "parent", "ppo")
    published = tmp_path / "logs" / "20260102_000000"
    if manifest is None:
        published.mkdir(parents=True)
        (published / "artifact_manifest.json").write_text("{not json")
    else:
        _write_manifest(published, manifest)
    before = _digests(published)
    target = published / stage_dirname(MODEL_ID, STAGE)

    with pytest.raises(WidenError, match=refusal):
        widen_checkpoint(model_id=MODEL_ID, stage=STAGE, target_stage_dir=target, parent_stage_dir=parent["stage_dir"])

    assert not target.exists() and _digests(published) == before


class _Rolled(RuntimeError):
    """The guard passed: the harness went on to derive the task and roll a panel."""


def _roll(*args, **kwargs):
    raise _Rolled()


class TestFreezeRecoveryGateRefusesAJudgedOrPublishedStage:
    """``harnesses/freeze_recovery_gate.py --stage-dir ... --replace`` rewrote the ``gate_resolution.json`` a judged
    recovery verdict was paired against, or wrote into a complete run. Both are refused before anything is rolled,
    ``--replace`` or not."""

    @pytest.fixture
    def producer(self, monkeypatch):
        from environments.shared.harnesses import freeze_recovery_gate as producer

        monkeypatch.setattr(producer, "stage_task_fingerprint", _roll)
        return producer

    @staticmethod
    def _stage(tmp_path, *, status=None, verdict=False, resolution=True):
        run = tmp_path / "20260101_000000"
        stage_dir = run / "02_recovery"
        stage_dir.mkdir(parents=True)
        if resolution:
            (stage_dir / "gate_resolution.json").write_text("{}")
        if verdict:
            _write_verdict(stage_dir, TYRANNOSAURUS_REX)
        if status is not None:
            _write_manifest(run, status)
        return run, stage_dir

    @pytest.mark.parametrize("replace", [True, False])
    @pytest.mark.parametrize("resolution", [True, False])
    def test_a_judged_node_is_never_re_frozen(self, producer, tmp_path, replace, resolution):
        run, stage_dir = self._stage(tmp_path, verdict=True, resolution=resolution)
        before = _digests(run)
        with pytest.raises(ValueError, match="holds gate_verdict.json: .* never re-frozen"):
            producer.freeze_recovery_gate(stage_dir, model_id=TYRANNOSAURUS_REX, replace=replace)
        assert _digests(run) == before

    @pytest.mark.parametrize("replace", [True, False])
    def test_a_complete_run_is_refused(self, producer, tmp_path, replace):
        run, stage_dir = self._stage(tmp_path, status="complete")
        before = _digests(run)
        with pytest.raises(ValueError, match="holds a complete result bundle"):
            producer.freeze_recovery_gate(stage_dir, model_id=TYRANNOSAURUS_REX, replace=replace)
        assert _digests(run) == before

    def test_an_unreadable_manifest_is_refused(self, producer, tmp_path):
        run, stage_dir = self._stage(tmp_path, resolution=False)
        (run / "artifact_manifest.json").write_text("{not json")
        with pytest.raises(ValueError, match="bundle status of run .* is unknown.*Restore a readable"):
            producer.freeze_recovery_gate(stage_dir, model_id=TYRANNOSAURUS_REX)
        assert not (stage_dir / "gate_resolution.json").exists()

    @pytest.mark.parametrize("status", [None, "partial", "failed"])
    def test_an_unjudged_stage_of_an_open_run_still_freezes(self, producer, tmp_path, status):
        _, stage_dir = self._stage(tmp_path, status=status)
        with pytest.raises(_Rolled):
            producer.freeze_recovery_gate(stage_dir, model_id=TYRANNOSAURUS_REX, replace=True)


class TestRecoveryOffdistPanelOut:
    """``harnesses/recovery_offdist_panel.py --out`` takes the report tools' guard: no JSON inside a run."""

    @pytest.fixture
    def panel(self, monkeypatch):
        from environments.shared.harnesses import recovery_offdist_panel as panel

        monkeypatch.setattr(panel, "build_env", _roll)
        return panel

    @staticmethod
    def _main(panel, monkeypatch, out, *extra):
        monkeypatch.setattr(sys, "argv", ["recovery_offdist_panel.py", "--controller", "statue", "--out", str(out)])
        sys.argv.extend(extra)
        return panel.main()

    def test_an_out_inside_a_run_is_refused_before_anything_is_rolled(self, panel, tmp_path, monkeypatch, capsys):
        run = tmp_path / "20260101_000000"
        (run / "02_recovery").mkdir(parents=True)
        (run / "provenance.json").write_text("{}")
        out = run / "02_recovery" / "panel.json"
        with pytest.raises(SystemExit) as excinfo:
            self._main(panel, monkeypatch, out)
        assert excinfo.value.code == 2 and "inside the run directory" in capsys.readouterr().err
        assert not out.exists()

    def test_the_policy_stage_directory_is_refused_without_provenance(self, panel, tmp_path, monkeypatch, capsys):
        models = tmp_path / "stage1" / "models"
        models.mkdir(parents=True)
        checkpoint = ["--policy-zip", str(models / "robust_best_model.zip"), "--vecnorm", str(models / "x.pkl")]
        with pytest.raises(SystemExit):
            self._main(panel, monkeypatch, tmp_path / "stage1" / "panel.json", *checkpoint)
        assert "inside the stage directory" in capsys.readouterr().err

    def test_a_scratch_out_passes_the_guard(self, panel, tmp_path, monkeypatch):
        with pytest.raises(_Rolled):
            self._main(panel, monkeypatch, tmp_path / "scratch" / "panel.json")


# ── Colab without Drive keeps its runtime ───────────────────────────────────


@pytest.fixture
def colab(monkeypatch):
    """A fake ``google.colab`` whose ``drive`` and ``runtime`` record their calls; ``time.sleep`` only records."""
    calls: list[str] = []
    module = types.ModuleType("google.colab")
    module.drive = types.SimpleNamespace(flush_and_unmount=lambda: calls.append("flush"))
    module.runtime = types.SimpleNamespace(unassign=lambda: calls.append("unassign"))
    monkeypatch.setitem(sys.modules, "google.colab", module)
    monkeypatch.setattr(time, "sleep", lambda seconds: calls.append(f"sleep {seconds}"))
    return calls


class TestColabWithoutDriveKeepsItsRuntime:
    def test_disconnect_keeps_the_runtime_and_says_why(self, colab, capsys):
        from environments.shared.notebook_runtime import disconnect_runtime

        disconnect_runtime("Training finished.", in_colab=True, auto=True, flush_drive=False)

        assert colab == [], "never unassigned: that would delete the run on the VM's disk"
        out = capsys.readouterr().out
        assert "Auto-disconnect skipped: the storage cell kept a run of this runtime on its own disk" in out
        assert "RUNS_ON_DRIVE is False" in out and "Download the run directory" in out

    def test_a_gate_failure_halt_raises_without_releasing_it(self, colab, capsys):
        from environments.shared.notebook_runtime import halt

        with pytest.raises(RuntimeError, match="failed its curriculum gate"):
            halt("stance failed its curriculum gate", in_colab=True, auto=True, flush_drive=False)
        assert colab == []
        assert "RUNS_ON_DRIVE is False" in capsys.readouterr().out

    def test_with_drive_it_still_flushes_and_releases(self, colab):
        from environments.shared.notebook_runtime import disconnect_runtime

        disconnect_runtime("Training finished.", in_colab=True, auto=True, flush_drive=True)
        assert colab == ["flush", "sleep 5", "unassign"]

    @staticmethod
    def _storage_cell(tmp_path, monkeypatch, *, in_colab, **earlier):
        """Execute the storage cell with Drive off, its provenance write recorded instead of run (*earlier*: names
        an earlier pass left in the namespace)."""
        from environments.shared import plant_contract, result_bundle

        identity = {"species": "tyrannosaurus_rex", "embodiment": "anatomical"}
        monkeypatch.setattr(
            plant_contract, "current_plant_identity", lambda model_id: SimpleNamespace(to_dict=lambda: identity)
        )
        monkeypatch.setattr(result_bundle, "initialize_result_bundle", lambda directory, **kw: directory / "p.json")
        from datetime import datetime

        namespace = {
            "Path": Path,
            "datetime": datetime,
            "repo_root": tmp_path,
            "IN_COLAB": in_colab,
            "USE_GOOGLE_DRIVE": False,
            "SPECIES": "tyrannosaurus_rex",
            "EMBODIMENT": "anatomical",
            "MODEL_ID": TYRANNOSAURUS_REX,
            "ALGORITHM": "ppo",
            "SEED": 42,
            "N_ENVS": 1,
            "QUICK_TEST": False,
            "RUN_KIND": "certify",
            "TRUNK_FROM": "",
            "RUN_ID": "",
            **earlier,
        }
        exec(compile(code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER), "sb3_storage", "exec"), namespace)
        return namespace

    def test_the_storage_cell_warns_that_the_run_lives_on_the_vm(self, tmp_path, monkeypatch, capsys):
        namespace = self._storage_cell(tmp_path, monkeypatch, in_colab=True)
        out = capsys.readouterr().out
        assert namespace["LOG_BASE"] == tmp_path / "logs" and namespace["RUNS_ON_DRIVE"] is False
        assert "WARNING: USE_GOOGLE_DRIVE is False" in out and "ONLY on this Colab" in out
        assert str(tmp_path / "logs") in out and "Download the run directory" in out
        assert "AUTO_DISCONNECT does not" in out

    def test_outside_colab_there_is_no_warning(self, tmp_path, monkeypatch, capsys):
        namespace = self._storage_cell(tmp_path, monkeypatch, in_colab=False)
        assert "WARNING" not in capsys.readouterr().out
        assert namespace["RUNS_ON_DRIVE"] is False  # nothing to flush; disconnect_runtime is a no-op there anyway

    def test_a_knob_flipped_after_the_storage_cell_keeps_the_runtime(self, tmp_path, monkeypatch, colab, capsys):
        """The run went to the VM's disk; the user then sets USE_GOOGLE_DRIVE = True and re-runs only the
        configuration cell. The final cell used to pass the knob, flush a Drive never mounted, and unassign."""
        from environments.shared.notebook_runtime import disconnect_runtime

        namespace = self._storage_cell(tmp_path, monkeypatch, in_colab=True)
        namespace.update(
            USE_GOOGLE_DRIVE=True,
            AUTO_DISCONNECT=True,
            BEHAVIOR="stance",
            TARGET_NODE=SimpleNamespace(id="stance"),
            disconnect_runtime=disconnect_runtime,
        )
        capsys.readouterr()

        exec(compile(code_cell(NOTEBOOK_PATH, FINAL_CELL_MARKER), "sb3_final", "exec"), namespace)

        assert colab == [], "the run on the VM's disk outlives the final cell"
        assert "RUNS_ON_DRIVE is False" in capsys.readouterr().out

    @staticmethod
    def _resolve(**names):
        """Run the storage cell's RUNS_ON_DRIVE statement and the note after it, alone, in a namespace of *names*."""
        tree = ast.parse(code_cell(NOTEBOOK_PATH, STORAGE_CELL_MARKER))
        [at] = [
            index
            for index, node in enumerate(tree.body)
            if isinstance(node, ast.Assign) and [ast.unparse(target) for target in node.targets] == ["RUNS_ON_DRIVE"]
        ]
        module = ast.Module(body=tree.body[at : at + 2], type_ignores=[])
        namespace = dict(names)
        exec(compile(module, "sb3_storage_resolve", "exec"), namespace)
        return namespace["RUNS_ON_DRIVE"]

    @pytest.mark.parametrize(
        ("use_drive", "in_colab", "earlier", "on_drive"),
        [
            (True, True, {}, True),
            (True, True, {"RUNS_ON_DRIVE": True}, True),
            (False, True, {}, False),
            (True, False, {}, False),
            # A later pass that mounts Drive leaves the earlier run on the VM's disk: the runtime is still kept.
            (True, True, {"RUNS_ON_DRIVE": False}, False),
        ],
    )
    def test_runs_on_drive_is_false_once_a_run_was_kept_on_the_vm(self, use_drive, in_colab, earlier, on_drive, capsys):
        resolved = self._resolve(USE_GOOGLE_DRIVE=use_drive, IN_COLAB=in_colab, **earlier)
        assert resolved is on_drive
        note = "an earlier pass of this cell kept a run on this VM's disk" in capsys.readouterr().out
        assert note == (use_drive and in_colab and not on_drive)

    def test_later_cells_read_the_resolved_value_never_the_knob(self):
        """As RUN_ID (``test_later_cells_read_the_resolved_run_id_never_the_knob``): only the configuration cell binds
        USE_GOOGLE_DRIVE and only it and the storage cell read it; only the storage cell binds RUNS_ON_DRIVE, and
        every release passes it."""
        from .notebook_cells import code_cell_sources

        def names(tree, context):
            return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, context)}

        sources = code_cell_sources(NOTEBOOK_PATH)
        config = [i for i, src in enumerate(sources) if CONFIG_CELL_MARKER in src]
        storage = [i for i, src in enumerate(sources) if STORAGE_CELL_MARKER in src]
        assert len(config) == len(storage) == 1
        releases = 0
        for index, src in enumerate(sources):
            tree = ast.parse(src)
            if index not in config:
                assert "USE_GOOGLE_DRIVE" not in names(tree, ast.Store), f"code cell {index} rebinds the knob"
            if index not in config + storage:
                assert "USE_GOOGLE_DRIVE" not in names(tree, ast.Load), f"code cell {index} reads USE_GOOGLE_DRIVE"
            if index not in storage:
                assert "RUNS_ON_DRIVE" not in names(tree, ast.Store), f"code cell {index} rebinds RUNS_ON_DRIVE"
            for call in ast.walk(tree):
                if isinstance(call, ast.Call) and ast.unparse(call.func) in ("halt", "disconnect_runtime"):
                    [flush] = [keyword for keyword in call.keywords if keyword.arg == "flush_drive"]
                    assert ast.unparse(flush.value) == "RUNS_ON_DRIVE", f"code cell {index}"
                    releases += 1
        assert releases == 2, "the chain loop's gate-failure halt and the final cell"
