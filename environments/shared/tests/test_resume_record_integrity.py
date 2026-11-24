"""A same-stage resume continues the record it resumes, or is refused before it writes anything.

Three findings on the same-stage resume path (the SB3 notebook's RESUME cell, its section 5 and
``train_base.train``'s ``resume_same_stage`` load), each pinned by tests that fail on the code before them:

* a SAC node was resumed in place on a fresh step counter, so its first evaluation rewrote the node's
  evaluation history, ledgers and best models from the new session alone, and the node was judged on that
  tail: the RESUME cell refuses ``ALGORITHM`` other than ``"ppo"``, and the library refuses such a load, or
  a curated PPO checkpoint's, into a directory holding the stage's training records
  (``_refuse_non_continuation_resume``);
* a resume rewrote ``diagnostics.npz`` from the new session's rows: a PPO continuation now seeds
  ``DiagnosticsCallback`` from the file (``seed_prior_diagnostics``), and the first rewrite of an existing
  file copies it aside as ``diagnostics.prior_<UTC>.npz``;
* a resume rebuilt ``stage_config.json`` from the new session (its commit, library version, remaining
  budget and label) and accepted a changed ``[ppo]`` or shaping recipe: the library refuses a resume whose
  ``hyperparameters_sha256`` or label differs from the record's (``config.refuse_resume_recipe_change``),
  the re-saved record keeps the first session's identity and load lineage and lists each resume that
  reached training under ``run.resume_sessions`` (a checkpoint trained on another task is refused before
  anything is written), and the RESUME cell warns when ``BEHAVIOR`` is ``"auto"``, when the run's
  ``provenance.json`` records another target deliverable, and when the checkout is at another commit.
"""

from __future__ import annotations

import copy
import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import numpy as np
import pytest

from environments.shared.config import (
    hyperparameters_sha256,
    load_all_stages,
    record_resume_session,
    save_stage_config,
)
from environments.shared.diagnostics import DiagnosticsCallback
from environments.shared.model_id import ModelId
from environments.shared.stage_manifest import load_stage_manifest, stage_dirname, stage_label

from .test_sb3_notebook_pins import RESUME_CELL_MARKER, _all_cell_sources, _cell, _define_node_budget

VELOCIRAPTOR_MONGOLIENSIS = ModelId("velociraptor_mongoliensis", "anatomical")
COMPSOGNATHUS_LONGIPES_ROBOT = ModelId("compsognathus_longipes", "robot")
COMPSOGNATHUS_LONGIPES = ModelId("compsognathus_longipes", "anatomical")


def _sb3_style_zip(path: Path, data: dict | None = None) -> Path:
    """An SB3 checkpoint archive's shape: a JSON ``data`` member beside the weights."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps(data or {}))
        archive.writestr("policy.pth", b"weights")
    return path


def _listing(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes() for path in sorted(directory.rglob("*")) if path.is_file()
    }


class _ConfigReached(RuntimeError):
    """``train_base.train`` got past its refusals to the stage-config save."""


def _train_to_the_config_save(
    monkeypatch, stage_dir: Path, *, algorithm: str = "ppo", configs=None, task: dict | None = None, **kwargs
) -> None:
    """``train_base.train`` on the velociraptor stance, stopped at its stage-config save (raising
    :class:`_ConfigReached`): the declared-parent, D-A20 and resume refusals before that save run for real.
    *task* is the task fingerprint this session derives (``{}`` by default)."""
    from environments.shared import config as config_module
    from environments.shared import task_fingerprint, train_base

    def reached(*args, **kwargs):
        raise _ConfigReached()

    monkeypatch.setattr(train_base, "current_plant_identity", lambda model_id: SimpleNamespace(to_dict=dict))
    monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: dict(task or {}))
    monkeypatch.setattr("environments.shared.policy_loading._ensure_sb3", lambda: {})
    monkeypatch.setattr(config_module, "save_stage_config", reached)
    train_base.train(
        cast(Any, SimpleNamespace(model_id=VELOCIRAPTOR_MONGOLIENSIS, env_class=object)),
        configs if configs is not None else load_all_stages(VELOCIRAPTOR_MONGOLIENSIS),
        1,
        total_timesteps=kwargs.pop("total_timesteps", 1_000),
        output_dir=str(stage_dir),
        use_tensorboard=False,
        verbose=0,
        algorithm=algorithm,
        **kwargs,
    )


class _BodyReached(RuntimeError):
    """The stubbed ``_train_stage_body`` stopped where its *learns* argument says."""


def _train_into_a_stubbed_body(monkeypatch, stage_dir: Path, *, learns: bool, **kwargs) -> None:
    """``train_base.train`` on the velociraptor stance with its refusals and stage-config save run for real and
    ``_train_stage_body`` stubbed: it raises :class:`_BodyReached` before the model loads, or (*learns*) once it
    has called ``before_learn``, as a session interrupted in ``learn()`` would have."""
    from environments.shared import task_fingerprint, train_base

    def body(*args, before_learn=None, **kwargs):
        if learns and before_learn is not None:
            before_learn()
        raise _BodyReached()

    monkeypatch.setattr(train_base, "current_plant_identity", lambda model_id: SimpleNamespace(to_dict=dict))
    monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
    monkeypatch.setattr("environments.shared.policy_loading._ensure_sb3", lambda: {})
    monkeypatch.setattr(train_base, "_train_stage_body", body)
    with pytest.raises(_BodyReached):
        train_base.train(
            cast(Any, SimpleNamespace(model_id=VELOCIRAPTOR_MONGOLIENSIS, env_class=object)),
            load_all_stages(VELOCIRAPTOR_MONGOLIENSIS),
            1,
            total_timesteps=kwargs.pop("total_timesteps", 600),
            output_dir=str(stage_dir),
            use_tensorboard=False,
            verbose=0,
            **kwargs,
        )


# --------------------------------------------------------------------------------------------- item 12: SAC in place
#: Each training record a stage directory can hold, as a path relative to it.
TRAINING_RECORDS = (
    "evaluations.npz",
    "eval_episodes.csv",
    "train_episodes.csv",
    "models/stage1_300000_steps.zip",
    "models/best_model.zip",
    "models/robust_best_model.zip",
    "models/stage1_final.zip",
)


class TestOnlyAContinuationResumesInPlace:
    """Only a PPO load of a periodic checkpoint continues a run; a SAC same-stage load, or a curated PPO
    checkpoint's, into the interrupted directory would rewrite the node's records from a fresh step counter, so
    it is refused before anything is written."""

    @staticmethod
    def _interrupted_stage(tmp_path: Path, record: str) -> tuple[Path, Path]:
        stage_dir = tmp_path / "run" / "01_stance"
        save_stage_config(
            stage_dir, 1, load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1], "SAC", model_id=VELOCIRAPTOR_MONGOLIENSIS
        )
        target = stage_dir / record
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"record")
        checkpoint = _sb3_style_zip(tmp_path / "elsewhere" / "stage1_200000_steps.zip")
        return stage_dir, checkpoint

    @pytest.mark.parametrize("record", TRAINING_RECORDS)
    def test_the_library_refuses_a_sac_resume_over_each_training_record(self, tmp_path, monkeypatch, record):
        from environments.shared.config import ResumeRefusedError

        stage_dir, checkpoint = self._interrupted_stage(tmp_path, record)
        before = _listing(stage_dir)
        with pytest.raises(ResumeRefusedError) as refused:
            _train_to_the_config_save(
                monkeypatch, stage_dir, algorithm="sac", load_path=str(checkpoint), task_load_mode="resume_same_stage"
            )
        message = str(refused.value)
        assert record in message and "SAC" in message and "replay buffer" in message
        assert "fresh run directory" in message and "Nothing was written" in message
        assert _listing(stage_dir) == before, "the refusal writes nothing"

    def test_a_sac_load_into_a_fresh_or_empty_directory_is_allowed(self, tmp_path, monkeypatch):
        """The CLI's default resume mints a fresh directory: nothing there to overwrite, so it trains."""
        checkpoint = _sb3_style_zip(tmp_path / "interrupted" / "models" / "stage1_200000_steps.zip")
        empty = tmp_path / "empty"
        # Another stage's checkpoint, a K3 file and an unrelated zip are not this stage's training records.
        for name in ("models/stage2_100_steps.zip", "models/notes.zip", "k3_panel.csv"):
            (empty / name).parent.mkdir(parents=True, exist_ok=True)
            (empty / name).write_bytes(b"")
        for stage_dir in (tmp_path / "fresh", empty):
            with pytest.raises(_ConfigReached):
                _train_to_the_config_save(
                    monkeypatch,
                    stage_dir,
                    algorithm="sac",
                    load_path=str(checkpoint),
                    task_load_mode="resume_same_stage",
                )

    @pytest.mark.parametrize("record", TRAINING_RECORDS)
    def test_ppo_still_continues_over_its_records(self, tmp_path, monkeypatch, record):
        stage_dir, checkpoint = self._interrupted_stage(tmp_path, record)
        # The record was saved as SAC above; resave it as PPO, the algorithm this resume trains.
        save_stage_config(
            stage_dir, 1, load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1], "PPO", model_id=VELOCIRAPTOR_MONGOLIENSIS
        )
        with pytest.raises(_ConfigReached):
            _train_to_the_config_save(
                monkeypatch, stage_dir, load_path=str(checkpoint), task_load_mode="resume_same_stage"
            )

    @pytest.mark.parametrize("curated", ["best_model.zip", "robust_best_model.zip", "stage1_final.zip"])
    @pytest.mark.parametrize("record", ["evaluations.npz", "train_episodes.csv", "models/stage1_300000_steps.zip"])
    def test_a_curated_ppo_load_over_the_records_is_refused_too(self, tmp_path, monkeypatch, curated, record):
        """A curated checkpoint carries no cumulative step count, so a PPO load of one restarts the counter and
        seeds nothing, exactly as SAC does."""
        from environments.shared.config import ResumeRefusedError

        stage_dir, _ = self._interrupted_stage(tmp_path, record)
        save_stage_config(
            stage_dir, 1, load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1], "PPO", model_id=VELOCIRAPTOR_MONGOLIENSIS
        )
        checkpoint = _sb3_style_zip(stage_dir / "models" / curated)
        before = _listing(stage_dir)
        with pytest.raises(ResumeRefusedError) as refused:
            _train_to_the_config_save(
                monkeypatch, stage_dir, load_path=str(checkpoint), task_load_mode="resume_same_stage"
            )
        message = str(refused.value)
        assert f"this --load of {curated} under resume_same_stage cannot continue them" in message
        assert "stage1_<steps>_steps.zip" in message and record in message and "Nothing was written" in message
        assert _listing(stage_dir) == before

    def test_a_curated_ppo_load_into_a_fresh_directory_is_allowed(self, tmp_path, monkeypatch):
        checkpoint = _sb3_style_zip(tmp_path / "interrupted" / "models" / "best_model.zip")
        with pytest.raises(_ConfigReached):
            _train_to_the_config_save(
                monkeypatch, tmp_path / "fresh", load_path=str(checkpoint), task_load_mode="resume_same_stage"
            )

    def test_the_resume_cell_refuses_sac_before_it_trains(self, tmp_path):
        with pytest.raises(RuntimeError, match=r"ALGORITHM='sac': an interrupted SAC node cannot be continued") as no:
            _run_resume_cell(tmp_path, algorithm="sac")
        message = str(no.value)
        assert "fresh RUN_ID" in message and "TRUNK_FROM naming this run" in message
        assert _run_resume_cell.calls == [], "nothing trains"


# ------------------------------------------------------------------------------------------ item 2: diagnostics.npz
def _diagnostics_callback(log_dir: Path) -> DiagnosticsCallback:
    callback = DiagnosticsCallback(log_dir=str(log_dir), verbose=0)
    model = MagicMock()
    model.logger = MagicMock()
    model.get_env.return_value = MagicMock(spec=[])
    del model.rollout_buffer
    callback.init_callback(model)
    return callback


def _rollout(callback: DiagnosticsCallback, timesteps: int, height: float, terminations: dict[str, int]) -> None:
    """One rollout of two steps ending at *timesteps*, an algorithm metric and *terminations* recorded."""
    callback.num_timesteps = timesteps
    callback.model.logger.name_to_value = {"train/approx_kl": timesteps / 1000}
    for _ in range(2):
        callback.locals = {"infos": [{"pelvis_height": height, "reward_alive": 1.0}], "dones": np.asarray([True])}
        callback._on_step()
    callback._rollout_terminations = Counter(terminations)
    callback._on_rollout_end()


def _prior_session(stage_dir: Path) -> bytes:
    """A first session's diagnostics.npz: rollouts at 100, 200 and 300 steps."""
    first = _diagnostics_callback(stage_dir)
    _rollout(first, 100, 0.5, {"fallen": 1})
    _rollout(first, 200, 0.6, {})
    _rollout(first, 300, 0.7, {"fallen": 2, "skull_low": 1})
    first._on_training_end()
    return (stage_dir / "diagnostics.npz").read_bytes()


class TestDiagnosticsContinueAcrossAResume:
    def test_a_seeded_session_rewrites_the_whole_history_up_to_the_resumed_step(self, tmp_path):
        prior = _prior_session(tmp_path)
        resumed = _diagnostics_callback(tmp_path)
        assert resumed.seed_prior_diagnostics(tmp_path / "diagnostics.npz", max_timesteps=200) == 2
        _rollout(resumed, 300, 0.9, {"skull_low": 1})
        resumed._on_training_end()
        data = np.load(tmp_path / "diagnostics.npz")
        # The rows after the resumed step (the old 300) belonged to the discarded policy; the new 300 replaces them.
        assert data["timesteps"].tolist() == [100, 200, 300]
        assert data["pelvis_height"].tolist() == pytest.approx([0.5, 0.6, 0.9])
        assert data["reward_alive"].tolist() == pytest.approx([1.0, 1.0, 1.0])
        assert data["rollout_episodes"].tolist() == [2, 2, 2]
        assert data["term_timesteps"].tolist() == [100, 300]
        assert data["term_fallen"].tolist() == pytest.approx([1.0, 0.0])
        assert data["term_skull_low"].tolist() == pytest.approx([0.0, 1.0])
        assert data["rollout_term_count_fallen"].tolist() == [1, 0]
        assert data["algo_timesteps"].tolist() == [100, 200, 300]
        assert data["algo_approx_kl"].tolist() == pytest.approx([0.1, 0.2, 0.3])
        # The file it replaced is set aside whole, the dropped row included.
        [aside] = sorted(tmp_path.glob("diagnostics.prior_*.npz"))
        assert aside.read_bytes() == prior

    def test_a_series_the_prior_file_lacks_reads_nan_on_the_seeded_rows(self, tmp_path):
        _prior_session(tmp_path)
        with np.load(tmp_path / "diagnostics.npz") as data:
            arrays = {key: data[key] for key in data.files if key not in ("pelvis_height", "rollout_term_count_fallen")}
        np.savez(tmp_path / "diagnostics.npz", **arrays)
        resumed = _diagnostics_callback(tmp_path)
        resumed.seed_prior_diagnostics(tmp_path / "diagnostics.npz")
        _rollout(resumed, 400, 0.9, {"fallen": 1})
        resumed._on_training_end()
        data = np.load(tmp_path / "diagnostics.npz")
        assert data["timesteps"].tolist() == [100, 200, 300, 400]
        assert np.isnan(data["pelvis_height"][:3]).all() and data["pelvis_height"][3] == pytest.approx(0.9)
        # A count the prior file never recorded is unknown on its rows, not zero.
        counts = data["rollout_term_count_fallen"]
        assert np.isnan(counts[:2]).all() and counts[2] == 1

    def test_an_unseeded_rewrite_copies_the_prior_file_aside_once(self, tmp_path):
        """A resume the history cannot seed (a curated --load, a SAC load) still loses no row."""
        prior = _prior_session(tmp_path)
        fresh = _diagnostics_callback(tmp_path)
        _rollout(fresh, 100, 0.9, {})
        _rollout(fresh, 200, 0.9, {})
        fresh._on_training_end()
        [aside] = sorted(tmp_path.glob("diagnostics.prior_*.npz"))
        assert aside.read_bytes() == prior
        assert np.load(tmp_path / "diagnostics.npz")["timesteps"].tolist() == [100, 200]

    def test_a_fresh_directory_gets_no_copy(self, tmp_path):
        _prior_session(tmp_path)
        assert not list(tmp_path.glob("diagnostics.prior_*.npz"))

    def test_a_failed_copy_leaves_the_prior_file_in_place(self, tmp_path, monkeypatch):
        from environments.shared import file_io

        prior = _prior_session(tmp_path)

        def refuse(src, dst):
            raise OSError("the mount went away")

        monkeypatch.setattr(file_io, "atomic_copy", refuse)
        fresh = _diagnostics_callback(tmp_path)
        _rollout(fresh, 100, 0.9, {})
        assert (tmp_path / "diagnostics.npz").read_bytes() == prior
        monkeypatch.undo()
        fresh._on_training_end()
        [aside] = sorted(tmp_path.glob("diagnostics.prior_*.npz"))
        assert aside.read_bytes() == prior

    def test_an_unreadable_prior_file_seeds_nothing(self, tmp_path):
        (tmp_path / "diagnostics.npz").write_bytes(b"not an npz")
        resumed = _diagnostics_callback(tmp_path)
        assert resumed.seed_prior_diagnostics(tmp_path / "diagnostics.npz", max_timesteps=10) == 0
        assert resumed.seed_prior_diagnostics(tmp_path / "missing.npz") == 0
        assert resumed._history_timesteps == []

    def test_a_resume_after_a_resume_seeds_the_rows_it_back_filled(self, tmp_path):
        """A seeded session writes its back-filled counts as NaN; the next resume seeds that file again."""
        first = _diagnostics_callback(tmp_path)
        first.record_rollout_counts = False  # a session that recorded no counts (code before 2026-10-18)
        _rollout(first, 100, 0.5, {"fallen": 1})
        _rollout(first, 200, 0.6, {"fallen": 1})
        first._on_training_end()
        second = _diagnostics_callback(tmp_path)
        assert second.seed_prior_diagnostics(tmp_path / "diagnostics.npz", max_timesteps=200) == 2
        _rollout(second, 300, 0.7, {"fallen": 2})
        second._on_training_end()
        with np.load(tmp_path / "diagnostics.npz") as data:
            assert np.isnan(data["rollout_episodes"][:2]).all() and data["rollout_episodes"][2] == 2
            assert np.isnan(data["rollout_term_count_fallen"][:2]).all()
        third = _diagnostics_callback(tmp_path)
        assert third.seed_prior_diagnostics(tmp_path / "diagnostics.npz", max_timesteps=300) == 3
        _rollout(third, 400, 0.8, {"fallen": 1})
        third._on_training_end()
        with np.load(tmp_path / "diagnostics.npz") as data:
            assert data["timesteps"].tolist() == [100, 200, 300, 400]
            episodes = data["rollout_episodes"]
            assert np.isnan(episodes[:2]).all() and episodes[2:].tolist() == [2, 2]
            counts = data["rollout_term_count_fallen"]
            assert np.isnan(counts[:2]).all() and counts[2:].tolist() == [2, 1]
        # A fully counted file still seeds integers.
        fourth = _diagnostics_callback(tmp_path / "counted")
        (tmp_path / "counted").mkdir()
        _prior_session(tmp_path / "counted")
        fourth.seed_prior_diagnostics(tmp_path / "counted" / "diagnostics.npz")
        assert fourth._history_rollout_episodes == [2, 2, 2]
        assert all(isinstance(value, int) for value in fourth._history_rollout_episodes)

    def test_a_file_whose_clock_is_not_a_finite_number_seeds_nothing(self, tmp_path):
        """A malformed prior file never sinks the resume nor leaves the callback half seeded."""
        _prior_session(tmp_path)
        with np.load(tmp_path / "diagnostics.npz") as data:
            arrays = {key: data[key] for key in data.files}
        arrays["term_timesteps"] = np.array([100.0, np.nan])
        np.savez(tmp_path / "diagnostics.npz", **arrays)
        resumed = _diagnostics_callback(tmp_path)
        assert resumed.seed_prior_diagnostics(tmp_path / "diagnostics.npz") == 0
        assert resumed._history_timesteps == [] and resumed._history_term_timesteps == []
        arrays["term_timesteps"] = np.array([100, 300])
        arrays["rollout_episodes"] = np.array(["two", "two", "two"])
        np.savez(tmp_path / "diagnostics.npz", **arrays)
        assert resumed.seed_prior_diagnostics(tmp_path / "diagnostics.npz") == 3
        assert np.isnan(resumed._history_rollout_episodes).all()

    def test_seed_resume_eval_state_seeds_diagnostics_whatever_the_evaluation_record_holds(self, tmp_path):
        """The diagnostics file is seeded beside the record, even when there is no evaluations.npz to seed."""
        from environments.shared.curriculum.checkpoints import seed_resume_eval_state

        _prior_session(tmp_path)
        resumed = _diagnostics_callback(tmp_path)
        eval_callback = SimpleNamespace(n_eval_episodes=1, best_mean_reward=-np.inf)
        assert seed_resume_eval_state(eval_callback, [resumed], tmp_path / "evaluations.npz", max_timesteps=200) == 0
        assert resumed._history_timesteps == [100, 200]

    def test_the_prior_copy_is_no_stage_reader_input(self, tmp_path):
        """The figures and the run comparison read ``diagnostics.npz`` by its name alone."""
        source = (Path(__file__).resolve().parents[1] / "visualization.py").read_text(encoding="utf-8")
        compare = (Path(__file__).resolve().parents[1] / "scripts" / "compare_run_diagnostics.py").read_text(
            encoding="utf-8"
        )
        for text in (source, compare):
            assert '"diagnostics.npz"' in text and "diagnostics*" not in text and "*.npz" not in text


# ----------------------------------------------------------------------------------- item 3: recipe, label, identity
class TestAResumeKeepsTheRecipe:
    @staticmethod
    def _recorded(tmp_path: Path, config: dict, *, label: str | None = None) -> tuple[Path, Path]:
        stage_dir = tmp_path / "run" / "01_stance"
        save_stage_config(
            stage_dir, 1, config, "PPO", extra={"timesteps": 1_000}, label=label, model_id=VELOCIRAPTOR_MONGOLIENSIS
        )
        checkpoint = _sb3_style_zip(stage_dir / "models" / "stage1_400_steps.zip")
        return stage_dir, checkpoint

    def test_the_digest_never_covers_the_budget(self):
        """So a resume that trains the rest of the budget, or a QUICK_TEST or --timesteps budget, digests the same."""
        config = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1]
        edited = copy.deepcopy(config)
        edited["curriculum_kwargs"]["timesteps"] = 50_000
        assert hyperparameters_sha256(edited, "PPO") == hyperparameters_sha256(config, "PPO")

    @pytest.mark.parametrize(
        ("table", "key", "named"),
        [("ppo_kwargs", "learning_rate", "ppo.learning_rate"), ("ppo_kwargs", "critic_phase", "ppo.critic_phase")],
    )
    def test_a_changed_recipe_is_refused_naming_the_keys_before_anything_is_written(
        self, tmp_path, monkeypatch, table, key, named
    ):
        from environments.shared.config import ResumeRefusedError

        configs = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)
        stage_dir, checkpoint = self._recorded(tmp_path, configs[1])
        before = _listing(stage_dir)
        edited = copy.deepcopy(configs)
        edited[1][table][key] = 1e-6 if key == "learning_rate" else True
        with pytest.raises(ResumeRefusedError) as refused:
            _train_to_the_config_save(
                monkeypatch, stage_dir, configs=edited, load_path=str(checkpoint), task_load_mode="resume_same_stage"
            )
        message = str(refused.value)
        recorded = json.loads((stage_dir / "stage_config.json").read_text())
        assert named in message and recorded["run"]["hyperparameters_sha256"] in message
        assert f"pin REPO_REF to the commit the stage recorded ({recorded['git_commit']})" in message
        assert "CRITIC_PHASE" in message and "fresh RUN_ID" in message
        assert "A commit from before 2026-11-24 resumes without the trainer's resume protections" in message
        assert _listing(stage_dir) == before

    @pytest.mark.parametrize("commit", ["a" * 40, None])
    def test_a_digest_its_own_blocks_do_not_explain_names_the_only_remedies(self, tmp_path, commit):
        """Restoring the recipe cannot clear a record whose digest its own blocks do not give: the recorded
        commit's digest definition, or a fresh RUN_ID, can."""
        from environments.shared.config import ResumeRefusedError, refuse_resume_recipe_change

        config = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1]
        stage_dir, _ = self._recorded(tmp_path, config)
        record = json.loads((stage_dir / "stage_config.json").read_text())
        record["run"]["hyperparameters_sha256"] = "sha256:" + "0" * 64
        record["git_commit"] = commit
        (stage_dir / "stage_config.json").write_text(json.dumps(record))
        with pytest.raises(ResumeRefusedError) as refused:
            refuse_resume_recipe_change(stage_dir, config, "PPO", task_load_mode="resume_same_stage")
        message = str(refused.value)
        assert "the recorded algorithm and shaping blocks match this session's" in message
        assert "Restoring the recipe cannot clear this" in message and "fresh RUN_ID" in message
        assert "CRITIC_PHASE" not in message
        if commit is None:
            assert "pin REPO_REF to the commit the interrupted session ran at (the record names none)" in message
        else:
            assert f"pin REPO_REF to the commit the stage recorded ({commit})" in message

    def test_a_checkpoint_trained_on_another_task_is_refused_before_anything_is_written(self, tmp_path, monkeypatch):
        """The task fingerprint is read off the archive before the stage directory is written: an env or reward
        edit no longer rewrites the record (its reward weights and task fingerprint) before the load refuses."""
        from environments.shared.task_fingerprint import MODEL_TASK_ATTRIBUTE, TaskFingerprintError

        configs = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)
        stage_dir, _ = self._recorded(tmp_path, configs[1])
        recorded_task = {"task_sha256": "sha256:" + "a" * 64, "env": {"energy_penalty_weight": 0.02}}
        checkpoint = _sb3_style_zip(
            stage_dir / "models" / "stage1_400_steps.zip", {MODEL_TASK_ATTRIBUTE: recorded_task}
        )
        before = _listing(stage_dir)
        current = {"task_sha256": "sha256:" + "b" * 64, "env": {"energy_penalty_weight": 0.52}}
        with pytest.raises(TaskFingerprintError, match=r"differing sections: \['env'\]"):
            _train_to_the_config_save(
                monkeypatch, stage_dir, load_path=str(checkpoint), task_load_mode="resume_same_stage", task=current
            )
        assert _listing(stage_dir) == before
        # The same task trains on (to the save), and a checkpoint that records none is left to the load's valve.
        with pytest.raises(_ConfigReached):
            _train_to_the_config_save(
                monkeypatch,
                stage_dir,
                load_path=str(checkpoint),
                task_load_mode="resume_same_stage",
                task={**current, "task_sha256": recorded_task["task_sha256"]},
            )
        bare = _sb3_style_zip(stage_dir / "models" / "stage1_200_steps.zip")
        with pytest.raises(_ConfigReached):
            _train_to_the_config_save(
                monkeypatch, stage_dir, load_path=str(bare), task_load_mode="resume_same_stage", task=current
            )

    def test_a_shaping_edit_is_refused_too(self, tmp_path):
        from environments.shared.config import ResumeRefusedError, refuse_resume_recipe_change

        config = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1]
        stage_dir, _ = self._recorded(tmp_path, config)
        edited = copy.deepcopy(config)
        edited["curriculum_kwargs"]["warmup_timesteps"] = 12_345
        with pytest.raises(ResumeRefusedError, match=r"shaping\.warmup_timesteps"):
            refuse_resume_recipe_change(stage_dir, edited, "PPO", task_load_mode="resume_same_stage")

    def test_the_same_recipe_on_the_remaining_budget_trains(self, tmp_path, monkeypatch):
        configs = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)
        stage_dir, checkpoint = self._recorded(tmp_path, configs[1])
        with pytest.raises(_ConfigReached):
            _train_to_the_config_save(
                monkeypatch,
                stage_dir,
                load_path=str(checkpoint),
                task_load_mode="resume_same_stage",
                total_timesteps=600,
            )

    def test_only_a_same_stage_resume_over_a_record_with_a_digest_is_checked(self, tmp_path):
        from environments.shared.config import refuse_resume_recipe_change

        config = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1]
        stage_dir, _ = self._recorded(tmp_path, config)
        edited = copy.deepcopy(config)
        edited["ppo_kwargs"]["learning_rate"] = 1e-6
        for mode in (None, "initialize_next_stage"):
            refuse_resume_recipe_change(stage_dir, edited, "PPO", task_load_mode=mode)
        refuse_resume_recipe_change(tmp_path / "fresh", edited, "PPO", task_load_mode="resume_same_stage")
        record = json.loads((stage_dir / "stage_config.json").read_text())
        del record["run"]["hyperparameters_sha256"]  # a record from before decision D-A21
        (stage_dir / "stage_config.json").write_text(json.dumps(record))
        refuse_resume_recipe_change(stage_dir, edited, "PPO", task_load_mode="resume_same_stage")
        (stage_dir / "stage_config.json").write_text("{not json")
        refuse_resume_recipe_change(stage_dir, edited, "PPO", task_load_mode="resume_same_stage")

    @pytest.mark.parametrize(
        ("recorded", "given", "refused"),
        [
            ("lr-sweep-a", None, False),
            ("lr-sweep-a", "", False),
            ("lr-sweep-a", " lr-sweep-a ", False),
            ("lr-sweep-a", "lr-sweep-b", True),
            (None, "lr-sweep-b", True),
            (None, None, False),
        ],
    )
    def test_the_label_is_kept_or_refused(self, tmp_path, recorded, given, refused):
        from environments.shared.config import ResumeRefusedError, refuse_resume_recipe_change

        config = load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1]
        stage_dir, _ = self._recorded(tmp_path, config, label=recorded)
        if not refused:
            refuse_resume_recipe_change(stage_dir, config, "PPO", task_load_mode="resume_same_stage", label=given)
            return
        with pytest.raises(ResumeRefusedError) as no:
            refuse_resume_recipe_change(stage_dir, config, "PPO", task_load_mode="resume_same_stage", label=given)
        held = f"the run label {recorded!r}" if recorded else "no run label"
        assert held in str(no.value) and repr(given) in str(no.value) and 'RUN_LABEL = ""' in str(no.value)


class TestAResumeKeepsTheFirstSessionsIdentity:
    CONFIG = {"name": "t", "env_kwargs": {}, "ppo_kwargs": {"n_steps": 64}, "sac_kwargs": {}, "curriculum_kwargs": {}}

    @staticmethod
    def _session(monkeypatch, stage_dir: Path, commit: str, version: str, **kwargs) -> dict[str, Any]:
        """One session's record as ``train_base.train`` writes it: the save, then (a resume over a record)
        its session appended once the model has loaded."""
        from environments.shared import config as config_module

        monkeypatch.setattr(config_module, "get_git_commit", lambda: commit)
        monkeypatch.setattr(config_module, "get_library_version", lambda: version)
        kwargs.setdefault("extra", {"seed": 42, "n_envs": 4, "timesteps": 1_000})
        continues = kwargs.get("load_mode") == "resume_same_stage" and (stage_dir / "stage_config.json").is_file()
        out = save_stage_config(
            stage_dir,
            1,
            TestAResumeKeepsTheFirstSessionsIdentity.CONFIG,
            "PPO",
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
            **kwargs,
        )
        if continues:
            record_resume_session(stage_dir, load_path=kwargs["load_path"], timesteps=kwargs["extra"]["timesteps"])
        data: dict[str, Any] = json.loads(out.read_text())
        return data

    def test_each_resume_keeps_the_first_identity_and_appends_its_session(self, tmp_path, monkeypatch):
        from environments.shared.result_bundle import sha256_file

        stage_dir = tmp_path / "01_stance"
        first = self._session(monkeypatch, stage_dir, "a" * 40, "0.4.0.dev0", label="lr-sweep-a")
        periodic = _sb3_style_zip(stage_dir / "models" / "stage1_400_steps.zip")
        resumed = self._session(
            monkeypatch,
            stage_dir,
            "b" * 40,
            "0.4.1.dev0",
            extra={"seed": 42, "n_envs": 4, "timesteps": 600},
            load_path=str(periodic),
            load_mode="resume_same_stage",
        )
        assert (resumed["git_commit"], resumed["library_version"]) == (first["git_commit"], first["library_version"])
        run = resumed["run"]
        assert run["timesteps"] == 1_000 and run["label"] == "lr-sweep-a"
        assert run["hyperparameters_sha256"] == first["run"]["hyperparameters_sha256"]
        [session] = run["resume_sessions"]
        assert session["git_commit"] == "b" * 40 and session["library_version"] == "0.4.1.dev0"
        assert session["timesteps"] == 600
        assert session["resume_load_path"] == str(periodic)
        assert session["resume_checkpoint_sha256"] == sha256_file(periodic)
        assert session["resumed_at"].endswith("+00:00")
        # A root trained from scratch records its first resume's load under the lineage keys, as before.
        assert run["load_path"] == str(periodic) and run["load_mode"] == "resume_same_stage"
        assert "resume_load_path" not in run

        later = _sb3_style_zip(stage_dir / "models" / "stage1_800_steps.zip")
        again = self._session(
            monkeypatch,
            stage_dir,
            "c" * 40,
            "0.4.1.dev0",
            extra={"seed": 42, "n_envs": 4, "timesteps": 200},
            load_path=str(later),
            load_mode="resume_same_stage",
        )
        assert again["git_commit"] == "a" * 40 and again["run"]["timesteps"] == 1_000
        assert [s["git_commit"] for s in again["run"]["resume_sessions"]] == ["b" * 40, "c" * 40]
        assert [s["timesteps"] for s in again["run"]["resume_sessions"]] == [600, 200]
        # A later resume keeps the lineage the record holds and records its own load apart.
        assert again["run"]["load_path"] == str(periodic)
        assert again["run"]["resume_load_path"] == str(later)
        assert again["run"]["resume_checkpoint_sha256"] == sha256_file(later)

    def test_the_save_alone_appends_no_session(self, tmp_path, monkeypatch):
        """``save_stage_config`` keeps the identity and the earlier sessions; this session is appended by
        ``record_resume_session``, once the model has loaded."""
        stage_dir = tmp_path / "01_stance"
        self._session(monkeypatch, stage_dir, "a" * 40, "0.4.0.dev0")
        periodic = _sb3_style_zip(stage_dir / "models" / "stage1_400_steps.zip")
        self._session(
            monkeypatch, stage_dir, "b" * 40, "0.4.0.dev0", load_path=str(periodic), load_mode="resume_same_stage"
        )
        saved = json.loads(
            save_stage_config(
                stage_dir,
                1,
                self.CONFIG,
                "PPO",
                extra={"seed": 42, "n_envs": 4, "timesteps": 300},
                model_id=VELOCIRAPTOR_MONGOLIENSIS,
                load_path=str(periodic),
                load_mode="resume_same_stage",
            ).read_text()
        )
        assert [s["git_commit"] for s in saved["run"]["resume_sessions"]] == ["b" * 40]
        assert saved["git_commit"] == "a" * 40 and saved["run"]["timesteps"] == 1_000
        with pytest.raises(FileNotFoundError, match="no readable stage_config.json"):
            record_resume_session(tmp_path / "nowhere", load_path=str(periodic), timesteps=1)

    def test_train_lists_a_session_only_once_its_model_has_loaded(self, tmp_path, monkeypatch):
        """A resume that fails after the save (a load refused at model load, an environment that will not
        build) trained nothing, so it lists no session; one that reaches ``learn()`` does, interrupted or not."""
        from environments.shared import config as config_module

        stage_dir = tmp_path / "run" / "01_stance"
        save_stage_config(
            stage_dir,
            1,
            load_all_stages(VELOCIRAPTOR_MONGOLIENSIS)[1],
            "PPO",
            extra={"timesteps": 1_000},
            model_id=VELOCIRAPTOR_MONGOLIENSIS,
        )
        periodic = _sb3_style_zip(stage_dir / "models" / "stage1_400_steps.zip")
        monkeypatch.setattr(config_module, "get_git_commit", lambda: "d" * 40)
        _train_into_a_stubbed_body(monkeypatch, stage_dir, learns=False, load_path=str(periodic))
        record = json.loads((stage_dir / "stage_config.json").read_text())
        assert "resume_sessions" not in record["run"] and record["run"]["timesteps"] == 1_000
        _train_into_a_stubbed_body(monkeypatch, stage_dir, learns=True, load_path=str(periodic))
        [session] = json.loads((stage_dir / "stage_config.json").read_text())["run"]["resume_sessions"]
        assert session["git_commit"] == "d" * 40 and session["timesteps"] == 600
        # A resume into a fresh directory is its own first session: nothing to append to.
        fresh = tmp_path / "fresh" / "01_stance"
        _train_into_a_stubbed_body(monkeypatch, fresh, learns=True, load_path=str(periodic))
        assert "resume_sessions" not in json.loads((fresh / "stage_config.json").read_text())["run"]

    def test_a_root_s_first_load_survives_its_resumes(self, tmp_path, monkeypatch):
        """A root whose first session loaded another run's checkpoint keeps that load under the lineage keys,
        so rule (d) still sees a screen's weights after an in-place resume."""
        from environments.shared.run_exclusions import EXCLUDED_RUN_IDS, excluded_stage_reason

        screen = _sb3_style_zip(tmp_path / sorted(EXCLUDED_RUN_IDS)[0] / "01_stance" / "models" / "stage1_final.zip")
        stage_dir = tmp_path / "mine" / "01_stance"
        first = self._session(
            monkeypatch, stage_dir, "a" * 40, "0.4.0.dev0", load_path=str(screen), load_mode="resume_same_stage"
        )
        assert "resume_sessions" not in first["run"]
        assert (excluded_stage_reason(stage_dir) or "").startswith("01_stance/stage_config.json run.load_path:")
        own = _sb3_style_zip(stage_dir / "models" / "stage1_400_steps.zip")
        resumed = self._session(
            monkeypatch, stage_dir, "b" * 40, "0.4.0.dev0", load_path=str(own), load_mode="resume_same_stage"
        )
        lineage = ("load_path", "load_mode", "parent_checkpoint_sha256")
        assert {key: resumed["run"][key] for key in lineage} == {key: first["run"][key] for key in lineage}
        assert resumed["run"]["resume_load_path"] == str(own)
        assert "run.load_path" in (excluded_stage_reason(stage_dir) or "")

    def test_an_entered_stage_keeps_its_edge_its_identity_and_its_sessions(self, tmp_path, monkeypatch):
        from environments.shared.config import LOAD_LINEAGE_KEYS

        stage_dir = tmp_path / "02_locomotion"
        parent = _sb3_style_zip(tmp_path / "stance_final.zip")
        entered = self._session(
            monkeypatch, stage_dir, "a" * 40, "0.4.0.dev0", load_path=str(parent), load_mode="initialize_next_stage"
        )
        periodic = _sb3_style_zip(stage_dir / "models" / "stage2_400_steps.zip")
        resumed = self._session(
            monkeypatch, stage_dir, "b" * 40, "0.4.0.dev0", load_path=str(periodic), load_mode="resume_same_stage"
        )
        edge = {key: entered["run"][key] for key in LOAD_LINEAGE_KEYS if key in entered["run"]}
        assert {key: resumed["run"][key] for key in edge} == edge
        assert resumed["run"]["resume_load_path"] == str(periodic)
        assert resumed["git_commit"] == "a" * 40
        assert resumed["run"]["resume_sessions"][0]["resume_load_path"] == str(periodic)

    def test_a_resume_into_a_fresh_directory_is_its_own_first_session(self, tmp_path, monkeypatch):
        periodic = _sb3_style_zip(tmp_path / "interrupted" / "models" / "stage1_400_steps.zip")
        record = self._session(
            monkeypatch,
            tmp_path / "fresh",
            "b" * 40,
            "0.4.0.dev0",
            load_path=str(periodic),
            load_mode="resume_same_stage",
        )
        assert record["git_commit"] == "b" * 40 and record["run"]["timesteps"] == 1_000
        assert "resume_sessions" not in record["run"]

    def test_the_duration_record_keeps_the_sessions(self, tmp_path, monkeypatch):
        from environments.shared.config import read_stage_duration, record_stage_duration

        stage_dir = tmp_path / "01_stance"
        self._session(monkeypatch, stage_dir, "a" * 40, "0.4.0.dev0")
        periodic = _sb3_style_zip(stage_dir / "models" / "stage1_400_steps.zip")
        self._session(
            monkeypatch, stage_dir, "b" * 40, "0.4.0.dev0", load_path=str(periodic), load_mode="resume_same_stage"
        )
        record_stage_duration(stage_dir, 12.5)
        record = json.loads((stage_dir / "stage_config.json").read_text())
        assert read_stage_duration(stage_dir) == 12.5 and len(record["run"]["resume_sessions"]) == 1
        # And the session append keeps the duration.
        record_resume_session(stage_dir, load_path=str(periodic), timesteps=100)
        record = json.loads((stage_dir / "stage_config.json").read_text())
        assert read_stage_duration(stage_dir) == 12.5 and len(record["run"]["resume_sessions"]) == 2

    def test_the_exclusion_rules_read_every_resume_session(self, tmp_path, monkeypatch):
        """Rule (b) once read the resume's commit at the top level, which now keeps the first session's; it reads
        each resume session's too. Rule (d) reads each session's continued-from checkpoint."""
        from environments.shared.run_exclusions import (
            EXCLUDED_CHECKPOINT_SHA256,
            EXCLUDED_COMMITS,
            excluded_stage_config_reason,
            excluded_stage_reason,
        )

        screen_commit = next(iter(EXCLUDED_COMMITS))
        stage_dir = tmp_path / "01_stance"
        self._session(monkeypatch, stage_dir, "a" * 40, "0.4.0.dev0")
        periodic = _sb3_style_zip(stage_dir / "models" / "stage1_400_steps.zip")
        record = self._session(
            monkeypatch, stage_dir, screen_commit, "0.4.0.dev0", load_path=str(periodic), load_mode="resume_same_stage"
        )
        assert record["git_commit"] == "a" * 40
        reason = excluded_stage_config_reason(record)
        assert reason is not None and reason.startswith("run.resume_sessions[0]: it was trained at commit")
        assert "run.resume_sessions[0]" in (excluded_stage_reason(stage_dir) or "")

        record = self._session(monkeypatch, tmp_path / "other", "a" * 40, "0.4.0.dev0")
        record["run"]["resume_sessions"] = [
            {"git_commit": "a" * 40, "resume_checkpoint_sha256": next(iter(EXCLUDED_CHECKPOINT_SHA256))}
        ]
        reason = excluded_stage_config_reason(record)
        assert reason is not None and reason.startswith("run.resume_sessions[0].resume_checkpoint_sha256:")
        assert excluded_stage_config_reason({"git_commit": "a" * 40, "run": {"resume_sessions": "x"}}) is None


# ------------------------------------------------------------------------------------------- the RESUME cell, executed
def _write_pair(zip_path: Path, vecnorm_path: Path) -> None:
    """An intact checkpoint pair as the RESUME cell checks one: SB3's outer members and an unpickling sidecar."""
    _sb3_style_zip(zip_path)
    vecnorm_path.write_bytes(pickle.dumps({}))


def _run_resume_cell(
    tmp_path: Path,
    *,
    behavior: str = "walk",
    resume_stage: Any = "locomotion",
    reference: Any = 2,
    algorithm: str = "ppo",
    steps: int = 100_000,
) -> list[dict]:
    """Execute the RESUME cell on the compsognathus robot's interrupted *reference* (one intact periodic pair at
    *steps*), with ``RUN_DIR`` at *tmp_path*; the ``train_stage`` calls are kept in ``_run_resume_cell.calls``."""
    models = tmp_path / stage_dirname(COMPSOGNATHUS_LONGIPES_ROBOT, reference) / "models"
    label = stage_label(reference)
    _write_pair(models / f"{label}_{steps}_steps.zip", models / f"{label}_vecnormalize_{steps}_steps.pkl")
    manifest = load_stage_manifest(COMPSOGNATHUS_LONGIPES_ROBOT)
    target = manifest.deliverables[-1] if behavior == "auto" else manifest.resolve_behavior(behavior)
    calls: list[dict] = []
    _run_resume_cell.calls = calls  # type: ignore[attr-defined]

    def train_stage(save_freq=100_000, **kwargs):  # the notebook's default checkpoint cadence
        calls.append(kwargs)
        return (None,) * 6

    namespace = {
        "SPECIES": COMPSOGNATHUS_LONGIPES_ROBOT.species,
        "EMBODIMENT": COMPSOGNATHUS_LONGIPES_ROBOT.embodiment,
        "MODEL_ID": COMPSOGNATHUS_LONGIPES_ROBOT,
        "BEHAVIOR": behavior,
        "MANIFEST": manifest,
        "CHAIN": manifest.chain_for(target.id),
        "STAGE_CONFIGS": load_all_stages(COMPSOGNATHUS_LONGIPES_ROBOT),
        "QUICK_TEST": False,
        "RUN_DIR": tmp_path,
        "LOG_BASE": tmp_path / "logs",
        "ALGORITHM": algorithm,
        "TRUNK_DIR": None,
        "RUN_LABEL": "",
        "RETRAIN_NODE": None,
        "train_stage": train_stage,
    }
    _define_node_budget(namespace)
    src = _cell(RESUME_CELL_MARKER).replace("RESUME_STAGE = None", f"RESUME_STAGE = {resume_stage!r}", 1)
    exec(compile(src, "sb3_resume", "exec"), namespace)
    return calls


class TestTheResumeCellWarns:
    """Warned, not refused: what the session after the resume trains toward, and under which code."""

    def test_an_ordinary_resume_warns_about_nothing(self, tmp_path, capsys):
        [call] = _run_resume_cell(tmp_path)
        assert call["task_load_mode"] == "resume_same_stage"
        assert "WARNING" not in capsys.readouterr().out

    def test_behavior_auto_is_warned(self, tmp_path, capsys):
        [_] = _run_resume_cell(tmp_path, behavior="auto")
        out = capsys.readouterr().out
        assert "WARNING: BEHAVIOR is 'auto'" in out and "last deliverable, 'behavior'" in out
        # An action, not a note: the training below would scroll it past before the chain loop retargets the run.
        assert "interrupt the run before the chain loop (section 6) starts" in out
        assert "set BEHAVIOR as the interrupted session had it and Run all again" in out

    def test_a_recorded_target_other_than_behaviors_is_warned(self, tmp_path, capsys):
        (tmp_path / "provenance.json").write_text(json.dumps({"target_deliverable": "3"}))
        [_] = _run_resume_cell(tmp_path, behavior="walk")
        out = capsys.readouterr().out
        assert "records the target deliverable '3'" in out and "BEHAVIOR='walk' resolves to '2' ('locomotion')" in out
        assert "interrupt the run before the chain loop (section 6) starts" in out
        capsys.readouterr()
        (tmp_path / "provenance.json").write_text(json.dumps({"target_deliverable": "2"}))
        [_] = _run_resume_cell(tmp_path, behavior="walk")
        assert "WARNING" not in capsys.readouterr().out

    @pytest.mark.parametrize("checkout", ["f" * 40, "e" * 40])
    def test_a_checkout_at_another_commit_than_the_node_recorded_is_warned(
        self, tmp_path, capsys, monkeypatch, checkout
    ):
        from environments.shared import config as config_module

        stage_dir = tmp_path / stage_dirname(COMPSOGNATHUS_LONGIPES_ROBOT, 2)
        stage_dir.mkdir(parents=True)
        (stage_dir / "stage_config.json").write_text(json.dumps({"git_commit": "e" * 40}))
        monkeypatch.setattr(config_module, "get_git_commit", lambda: checkout)
        [_] = _run_resume_cell(tmp_path)
        out = capsys.readouterr().out
        if checkout == "e" * 40:
            assert "WARNING" not in out
            return
        assert f"recorded git_commit {'e' * 40} in its stage_config.json" in out
        assert f"this checkout is at {'f' * 40}" in out and f"pin REPO_REF (section 1) to {'e' * 40}" in out
        assert "a commit from before 2026-11-24 resumes without the trainer's record protections" in out

    def test_the_section_5_prose_states_the_rules(self):
        every = _all_cell_sources()
        prose = every[every.index(_cell(RESUME_CELL_MARKER)) - 1]
        for phrase in (
            "Only a PPO node is resumed",
            'refuses `ALGORITHM` other than `"ppo"`',
            "`hyperparameters_sha256`",
            # The step list itself names the recipe knobs a refusal would otherwise send the reader back for.
            "1. Set `REPO_REF` (section 1) to the `git_commit` the node's `stage_config.json` records",
            "`RUN_LABEL` empty (the resume keeps the recorded label)",
            "`CRITIC_PHASE` (with any other recipe knob)",
            "A commit from before 2026-11-24 resumes without these protections",
            "interrupt the run before the chain\nloop starts",
            "load lineage",
            '`BEHAVIOR` is `"auto"`',
            "`provenance.json` records another target deliverable",
            "`run.resume_sessions`",
            "`diagnostics.prior_<UTC>.npz`",
        ):
            assert phrase in prose, phrase


# -------------------------------------------------------------------------------------------- two real train() sessions
@pytest.fixture
def one_torch_thread():
    """One torch thread while the sessions train: small matmuls slow down sharply on a shared CPU."""
    torch = pytest.importorskip("torch")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_two_real_sessions_continue_the_diagnostics_and_keep_the_first_identity(
    tmp_path, monkeypatch, one_torch_thread
):
    """The real entry point twice over the Compsognathus stance: the second session seeds the first's
    diagnostics.npz up to the resumed checkpoint and sets the file it replaces aside, and the re-saved
    stage_config.json keeps the first session's identity; a SAC resume, a changed recipe or a changed task in
    that directory is refused before anything is written, and a resume that fails at model load lists no
    session."""
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("mujoco")
    from environments.shared import config as config_module
    from environments.shared import train_base
    from environments.shared.config import ResumeRefusedError
    from environments.shared.result_bundle import sha256_file
    from environments.shared.species_registry import get_species_config
    from environments.shared.task_fingerprint import TaskFingerprintError

    configs = copy.deepcopy(load_all_stages(COMPSOGNATHUS_LONGIPES))
    configs[1]["ppo_kwargs"].update(n_steps=256, batch_size=64, n_epochs=1)
    out = tmp_path / "01_stance"
    common: dict[str, Any] = {
        "species_cfg": get_species_config(COMPSOGNATHUS_LONGIPES),
        "stage_configs": configs,
        "stage": 1,
        "n_envs": 1,
        "seed": 11,
        "eval_freq": 10**9,
        "save_freq": 512,
        "verbose": 0,
        "use_tensorboard": False,
        "output_dir": str(out),
        "report_metrics": False,
        "save_on_interrupt": False,
    }

    monkeypatch.setattr(config_module, "get_git_commit", lambda: "1" * 40)
    train_base.train(total_timesteps=1024, label="first", **common)
    first_diagnostics = (out / "diagnostics.npz").read_bytes()
    with np.load(out / "diagnostics.npz") as data:
        first_rows = data["timesteps"].tolist()
        first_height = data["pelvis_height"].tolist()
    assert first_rows == [256, 512, 768, 1024]
    for stale in (out / "models").glob("stage1_final*"):
        stale.unlink()  # the interruption: the session never reached its final save
    checkpoint = out / "models" / "stage1_512_steps.zip"

    before = _listing(out)
    with pytest.raises(ResumeRefusedError, match="SAC --load under resume_same_stage cannot continue"):
        train_base.train(total_timesteps=512, load_path=str(checkpoint), **{**common, "algorithm": "sac"})
    edited = copy.deepcopy(configs)
    edited[1]["ppo_kwargs"]["learning_rate"] = 1e-6
    with pytest.raises(ResumeRefusedError, match=r"ppo\.learning_rate"):
        train_base.train(total_timesteps=512, load_path=str(checkpoint), **{**common, "stage_configs": edited})
    reward = copy.deepcopy(configs)
    reward[1]["env_kwargs"]["energy_penalty_weight"] = float(reward[1]["env_kwargs"]["energy_penalty_weight"]) + 0.5
    with pytest.raises(TaskFingerprintError, match="differing sections"):
        train_base.train(total_timesteps=512, load_path=str(checkpoint), **{**common, "stage_configs": reward})
    assert _listing(out) == before, "no refusal wrote anything"

    monkeypatch.setattr(config_module, "get_git_commit", lambda: "2" * 40)

    def refused_at_load(*args, **kwargs):
        raise RuntimeError("the load failed after the save")

    with monkeypatch.context() as failing:
        failing.setattr(train_base, "_create_or_load_model", refused_at_load)
        with pytest.raises(RuntimeError, match="the load failed after the save"):
            train_base.train(total_timesteps=512, load_path=str(checkpoint), **common)
    assert "resume_sessions" not in json.loads((out / "stage_config.json").read_text())["run"]

    train_base.train(total_timesteps=512, load_path=str(checkpoint), **common)

    with np.load(out / "diagnostics.npz") as data:
        assert data["timesteps"].tolist() == [256, 512, 768, 1024]
        assert data["pelvis_height"].tolist()[:2] == pytest.approx(first_height[:2])
    [aside] = sorted(out.glob("diagnostics.prior_*.npz"))
    assert aside.read_bytes() == first_diagnostics
    record = json.loads((out / "stage_config.json").read_text())
    assert record["git_commit"] == "1" * 40
    run = record["run"]
    assert run["timesteps"] == 1024 and run["label"] == "first"
    [session] = run["resume_sessions"]
    assert session["git_commit"] == "2" * 40 and session["timesteps"] == 512
    assert session["resume_load_path"] == str(checkpoint)
    assert session["resume_checkpoint_sha256"] == sha256_file(checkpoint)
