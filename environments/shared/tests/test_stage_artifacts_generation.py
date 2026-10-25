"""Tests for ``generate_stage_artifacts`` on a trainer-shaped stage directory."""

import contextlib
from unittest.mock import MagicMock, patch

import pytest

from environments.shared.model_id import ModelId
from environments.shared.species_names import model_ids


@pytest.mark.parametrize(
    "model_id,stage",
    [(model_id, 1) for model_id in model_ids()]
    + [
        (model_id, "recovery")
        for model_id in (
            ModelId("tyrannosaurus_rex", "anatomical"),
            ModelId("compsognathus_longipes", "anatomical"),
            ModelId("compsognathus_longipes", "robot"),
        )
    ],
    ids=lambda value: f"{value.species}-{value.embodiment}" if isinstance(value, ModelId) else str(value),
)
def test_stance_replay_producer_enables_the_same_outputs_and_matched_checkpoints(tmp_path, model_id, stage):
    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
    from environments.shared.reporting.stage_artifacts import _record_stage_replays
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import stage_label

    models = tmp_path / "models"
    models.mkdir()
    final = f"{stage_label(stage)}_final"
    for name in ("best_model", "robust_best_model", final):
        (models / f"{name}.zip").write_bytes(b"model")
        (models / f"{name}_vecnorm.pkl").write_bytes(b"normalization")
    model = MagicMock()
    attach_plant_identity(model, current_plant_identity(model_id))
    algorithm = MagicMock()
    algorithm.load.return_value = model
    replays = tmp_path / "replays"
    with (
        patch("environments.shared.policy_loading._ensure_sb3", return_value={"PPO": algorithm, "SAC": algorithm}),
        patch("environments.shared.evaluation.record_stage_video") as video,
    ):
        _record_stage_replays(
            species_cfg=get_species_config(model_id),
            stage_config=load_stage_config(model_id, stage),
            stage=stage,
            algorithm="ppo",
            stage_dir=tmp_path,
            replays_out=replays,
            model_dir=models,
            seed=42,
            stage_results={},
            allow_legacy_plant=False,
        )
    calls = [call.kwargs for call in video.call_args_list]
    assert [call["label"] for call in calls] == ["selected", "final"]
    for call, name in zip(calls, ("robust_best_model", final)):
        assert call["vecnorm_path"] == str(models / f"{name}_vecnorm.pkl")
        assert call["output_dir"] == replays
        assert call["collect_stance_diagnostics"] is True
        assert set(call["camera_views"]) == {"side", "front"}
        # Other species retain their own camera distance, rather than the
        # 3.4 m framing that makes the small Compsognathus barely visible.
        if model_id.species not in ("tyrannosaurus_rex", "brachiosaurus_altithorax"):
            assert all("distance" not in preset for preset in call["camera_views"].values())
        if model_id.species == "brachiosaurus_altithorax":
            assert call["camera_views"]["front"]["distance"] > call["env_class"]._camera_distance


class TestGenerateTrialArtifacts:
    """Test generate_stage_artifacts produces stage summary and videos.

    This tests the shared function the training notebook uses to
    produce its stage artifacts.
    """

    def test_writes_stage_summary_and_records_videos(self, tmp_path):
        """After a stage, stage_summary.txt is written and videos are attempted."""
        import numpy as np

        from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
        from environments.shared.reporting import generate_stage_artifacts

        # Setup directory structure matching what train() produces
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        (model_dir / "best_model.zip").write_bytes(b"fake")
        (model_dir / "best_model_vecnorm.pkl").write_bytes(b"fake")
        (model_dir / "stage1_final.zip").write_bytes(b"fake")
        (model_dir / "stage1_final_vecnorm.pkl").write_bytes(b"fake")

        # Create evaluations.npz
        rewards = np.array([[10.0, 12.0], [20.0, 22.0]])
        lengths = np.array([[100, 110], [200, 210]])
        timesteps = np.array([50000, 100000])
        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=rewards,
            ep_lengths=lengths,
            timesteps=timesteps,
        )

        mock_species_cfg = MagicMock()
        mock_species_cfg.model_id = ModelId("velociraptor_mongoliensis", "anatomical")
        mock_species_cfg.env_class = MagicMock

        stage_config = {
            "name": "Balance",
            "description": "Stand up",
            "env_kwargs": {"sim_dt": 0.01},
            "ppo_kwargs": {},
        }

        fake_model = MagicMock()
        attach_plant_identity(
            fake_model,
            current_plant_identity(ModelId("velociraptor_mongoliensis", "anatomical"), verify_generated=False),
        )
        mock_algorithm = MagicMock()
        mock_algorithm.load.return_value = fake_model
        mock_sb3 = {"PPO": mock_algorithm, "SAC": mock_algorithm}

        with (
            patch(
                "environments.shared.policy_loading._ensure_sb3",
                return_value=mock_sb3,
            ),
            patch(
                "environments.shared.evaluation.record_stage_video",
            ) as mock_video,
        ):
            results = generate_stage_artifacts(
                species_cfg=mock_species_cfg,
                stage_config=stage_config,
                stage=1,
                algorithm="ppo",
                stage_dir=tmp_path,
                seed=42,
                timesteps=100_000,
            )

        # stage_summary.txt should exist
        summary = tmp_path / "stage_summary.txt"
        assert summary.exists()
        text = summary.read_text()
        assert "Stance" in text
        assert "Velociraptor" in text

        # Videos should be recorded for the selected and final checkpoints.
        # "selected", not "best": the replay comes from the same selector that
        # decides the next-stage handoff and that evaluation_selected.csv is
        # evidence for, so the two describe one policy.
        assert mock_video.call_count == 2
        labels = [call.kwargs["label"] for call in mock_video.call_args_list]
        assert "selected" in labels
        assert "final" in labels

        # Returned results should have best eval metrics from evaluations.npz
        assert results["best_eval_reward"] == 21.0
        assert results["best_eval_timestep"] == 100000

        # Training graphs should be generated when matplotlib is available,
        # into stage_dir/figures/ rather than loose in the stage root.
        try:
            import matplotlib  # noqa: F401

            from environments.shared.reporting import stage_layout

            figures = stage_layout.figures_dir(tmp_path)
            assert (figures / "training_curves.png").exists()
            assert (figures / "locomotion_health.png").exists()
            assert (figures / "behavioral_metrics.png").exists()
            assert not list(tmp_path.glob("*.png")), "figures must not be loose in the stage root"
        except ImportError:
            pass  # graphs are skipped gracefully without matplotlib


def test_generate_stage_artifacts_takes_sim_dt_from_the_nodes_env(tmp_path):
    """Without precomputed results, sim time uses a bare env of this node's task."""
    from environments.shared.reporting import stage_artifacts

    class ProbeEnv:
        dt = 0.02
        created_with: dict = {}
        closed = False

        def __init__(self, **kwargs):
            ProbeEnv.created_with = kwargs

        def close(self):
            ProbeEnv.closed = True

    class Stop(Exception):
        pass

    captured = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        raise Stop

    species_cfg = MagicMock()
    species_cfg.model_id = ModelId("compsognathus_longipes", "anatomical")
    species_cfg.env_class = ProbeEnv
    stage_config = {"name": "Locomotion", "description": "Walk", "env_kwargs": {"alive_bonus": 0.5}}
    with patch.object(stage_artifacts, "build_stage_results_from_eval_data", side_effect=fake_build):
        with pytest.raises(Stop):
            stage_artifacts.generate_stage_artifacts(
                species_cfg=species_cfg,
                stage_config=stage_config,
                stage=2,
                algorithm="ppo",
                stage_dir=tmp_path,
                seed=42,
                timesteps=1,
            )
    assert captured["sim_dt"] == 0.02
    assert ProbeEnv.created_with == {"alive_bonus": 0.5}
    assert ProbeEnv.closed


@pytest.mark.parametrize(
    "model_id",
    [ModelId("compsognathus_longipes", "anatomical"), ModelId("compsognathus_longipes", "robot")],
    ids=lambda m: f"{m.species}-{m.embodiment}",
)
def test_the_compsognathus_longipes_pair_steps_at_twenty_milliseconds(model_id):
    """The premise of the probe: a 0.01 s default halves this pair's printed sim time."""
    from environments.shared.species_registry import env_class

    env = env_class(model_id)()
    try:
        assert env.dt == pytest.approx(0.02)
    finally:
        env.close()


@pytest.mark.parametrize("figure_fails", [False, True], ids=["drawn", "raises"])
def test_the_gate_progress_figure_is_drawn_in_its_own_guard(tmp_path, figure_fails):
    """``figures/gate_progress.png`` beside the other figures; a failure drawing it costs that file only."""
    pytest.importorskip("matplotlib")
    import numpy as np

    from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
    from environments.shared.reporting import generate_stage_artifacts, stage_layout

    model_id = ModelId("velociraptor_mongoliensis", "anatomical")
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name in ("best_model.zip", "best_model_vecnorm.pkl", "stage1_final.zip", "stage1_final_vecnorm.pkl"):
        (model_dir / name).write_bytes(b"fake")
    np.savez(
        tmp_path / "evaluations.npz",
        results=np.array([[10.0, 12.0], [20.0, 22.0]]),
        ep_lengths=np.array([[100, 110], [200, 210]]),
        timesteps=np.array([50000, 100000]),
    )
    np.savez(
        tmp_path / "gate_progress.npz",
        timesteps=np.array([50000, 100000]),
        mean_reward=np.array([11.0, 21.0]),
        full_horizon_fraction=np.array([0.0, 0.0]),
    )
    species_cfg = MagicMock()
    species_cfg.model_id = model_id
    species_cfg.env_class = MagicMock
    fake_model = MagicMock()
    attach_plant_identity(fake_model, current_plant_identity(model_id, verify_generated=False))
    algorithm = MagicMock()
    algorithm.load.return_value = fake_model

    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("environments.shared.policy_loading._ensure_sb3", return_value={"PPO": algorithm, "SAC": algorithm})
        )
        stack.enter_context(patch("environments.shared.evaluation.record_stage_video"))
        if figure_fails:
            stack.enter_context(
                patch(
                    "environments.shared.visualization.plot_gate_progress",
                    side_effect=RuntimeError("a figure went wrong"),
                )
            )
        generate_stage_artifacts(
            species_cfg=species_cfg,
            stage_config={
                "name": "Balance",
                "description": "Stand up",
                "env_kwargs": {"sim_dt": 0.01},
                "ppo_kwargs": {},
            },
            stage=1,
            algorithm="ppo",
            stage_dir=tmp_path,
            seed=42,
            timesteps=100_000,
        )
    figures = stage_layout.figures_dir(tmp_path)
    for name in ("training_curves.png", "locomotion_health.png", "behavioral_metrics.png"):
        assert (figures / name).exists(), name
    assert (figures / "gate_progress.png").exists() is not figure_fails


# ---------------------------------------------------------------------------------------------- training history

#: An eval_episodes.csv header as an early ledger wrote it (no foot, balance or asymmetry columns).
_EVAL_LEDGER_COLUMNS = (
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
)


def _write_ledgers(stage_dir, *, train: bool = True) -> None:
    """Two 3-episode evaluations in eval_episodes.csv and, with *train*, a few training episodes."""
    import csv

    with (stage_dir / "eval_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(_EVAL_LEDGER_COLUMNS)
        for evaluation, timesteps in enumerate((50_000, 100_000)):
            for episode in range(3):
                fell = evaluation == 0 and episode == 2
                writer.writerow(
                    [
                        evaluation,
                        timesteps,
                        episode,
                        0,
                        10.0 * (evaluation + 1) + episode,
                        400 if fell else 1000,
                        0 if fell else 1,
                        "fallen" if fell else "truncated",
                        0 if fell else 1,
                        "horizon" if fell else "",
                    ]
                )
    if train:
        with (stage_dir / "train_episodes.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timesteps", "env", "length", "return", "cause"])
            for timesteps, cause in ((20_000, "fallen"), (60_000, "truncated"), (90_000, "truncated")):
                writer.writerow([timesteps, 0, 1000, 1.0, cause])


def _judge_velociraptor_mongoliensis_stage(stage_dir, *, patches=(), generate_graphs: bool = True) -> dict:
    """``generate_stage_artifacts`` on a Velociraptor stage-1 directory (fake checkpoints, stubbed replays)."""
    import numpy as np

    from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
    from environments.shared.reporting import generate_stage_artifacts

    model_id = ModelId("velociraptor_mongoliensis", "anatomical")
    model_dir = stage_dir / "models"
    model_dir.mkdir(parents=True)
    for name in ("best_model.zip", "best_model_vecnorm.pkl", "stage1_final.zip", "stage1_final_vecnorm.pkl"):
        (model_dir / name).write_bytes(b"fake")
    np.savez(
        stage_dir / "evaluations.npz",
        results=np.array([[10.0, 12.0], [20.0, 22.0]]),
        ep_lengths=np.array([[100, 110], [200, 210]]),
        timesteps=np.array([50000, 100000]),
    )
    np.savez(
        stage_dir / "gate_progress.npz",
        timesteps=np.array([50000, 100000]),
        mean_reward=np.array([11.0, 21.0]),
        full_horizon_fraction=np.array([0.0, 0.0]),
    )
    species_cfg = MagicMock()
    species_cfg.model_id = model_id
    species_cfg.env_class = MagicMock
    fake_model = MagicMock()
    attach_plant_identity(fake_model, current_plant_identity(model_id, verify_generated=False))
    algorithm = MagicMock()
    algorithm.load.return_value = fake_model
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch("environments.shared.policy_loading._ensure_sb3", return_value={"PPO": algorithm, "SAC": algorithm})
        )
        stack.enter_context(patch("environments.shared.evaluation.record_stage_video"))
        for patcher in patches:
            stack.enter_context(patcher)
        return generate_stage_artifacts(
            species_cfg=species_cfg,
            stage_config={
                "name": "Balance",
                "description": "Stand up",
                "env_kwargs": {"sim_dt": 0.01},
                "ppo_kwargs": {},
            },
            stage=1,
            algorithm="ppo",
            stage_dir=stage_dir,
            seed=42,
            timesteps=100_000,
            generate_graphs=generate_graphs,
        )


def _listing(stage_dir) -> list[str]:
    return sorted(str(path.relative_to(stage_dir)) for path in stage_dir.rglob("*") if path.is_file())


def test_the_training_history_is_drawn_at_stage_end_and_its_failure_costs_it_only(tmp_path, caplog):
    """``figures/training_history/`` from the stage's ledgers, after the verdict, beside the other figures.

    Three judgements of the same stage: without the report (the step patched out), with it, and with it raising
    part-way. The returned results, the verdict, the summary and every other file are the same in all three; the
    raising one publishes none of the report and logs one warning.
    """
    pytest.importorskip("matplotlib")
    import json
    import logging
    import re

    from environments.shared.reporting import stage_artifacts, stage_layout, training_history

    def raise_part_way(runs, out_dir, **kwargs):
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "01_outcomes.png").write_bytes(b"half a page")
        raise RuntimeError("a page went wrong")

    variants = {
        "without": [patch.object(stage_artifacts, "_write_training_history", return_value=None)],
        "drawn": [],
        "raises": [patch.object(training_history, "write_report", side_effect=raise_part_way)],
    }
    results, listings, summaries, verdicts = {}, {}, {}, {}
    for name, patches in variants.items():
        stage_dir = tmp_path / name / "01_stance"
        stage_dir.mkdir(parents=True)
        _write_ledgers(stage_dir)
        with caplog.at_level(logging.WARNING, logger=stage_artifacts.logger.name):
            caplog.clear()
            returned = _judge_velociraptor_mongoliensis_stage(stage_dir, patches=patches)
            warnings = [record.getMessage() for record in caplog.records if record.levelno >= logging.WARNING]
        # Each variant judges its own directory, so its path is the one difference allowed.
        results[name] = json.dumps(returned, sort_keys=True, default=str).replace(str(stage_dir), "<stage>")
        listings[name] = _listing(stage_dir)
        summary = (stage_dir / "stage_summary.txt").read_text(encoding="utf-8").replace(str(stage_dir), "<stage>")
        summaries[name] = re.sub(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d", "<written at>", summary)
        verdict = json.loads((stage_dir / "gate_verdict.json").read_text(encoding="utf-8"))
        verdict.pop("judged_at")
        verdicts[name] = json.dumps(verdict, sort_keys=True).replace(str(stage_dir), "<stage>")
        failed = [message for message in warnings if "training history failed" in message]
        assert len(failed) == (name == "raises"), warnings

    history = stage_layout.training_history_dir(tmp_path / "drawn" / "01_stance")
    pages = [f"{name}.png" for name in training_history.page_names()]
    assert sorted(path.name for path in history.iterdir()) == sorted(
        pages + [training_history.HTML_FILENAME, training_history.JSON_FILENAME]
    )
    document = json.loads((history / training_history.JSON_FILENAME).read_text(encoding="utf-8"))
    assert document["schema"] == training_history.SCHEMA and document["errors"] == []
    assert document["runs"][0]["evaluations"]["timesteps"] == [50_000, 100_000]
    assert document["runs"][0]["evaluations"]["falls"] == [1, 0]
    prefix = f"{stage_layout.FIGURES_DIRNAME}/{stage_layout.TRAINING_HISTORY_DIRNAME}/"
    assert listings["drawn"] == sorted(listings["without"] + [prefix + path.name for path in history.iterdir()])
    assert listings["raises"] == listings["without"]
    assert "figures/gate_progress.png" in listings["raises"]
    assert results["drawn"] == results["without"] == results["raises"]
    assert summaries["drawn"] == summaries["without"] == summaries["raises"]
    assert verdicts["drawn"] == verdicts["without"] == verdicts["raises"]


def test_old_runs_and_missing_ledgers_are_fine(tmp_path, caplog):
    """A stage without either ledger (a run from before them) gets no report; one ledger is enough for one."""
    pytest.importorskip("matplotlib")
    import json
    import logging

    from environments.shared.reporting import stage_artifacts, stage_layout, training_history

    old = tmp_path / "old" / "stage1"
    old.mkdir(parents=True)
    with caplog.at_level(logging.INFO, logger=stage_artifacts.logger.name):
        assert stage_artifacts._write_training_history(old, tmp_path / "old_staging") is None
    assert not (tmp_path / "old_staging").exists()
    assert any("neither eval_episodes.csv nor train_episodes.csv" in r.getMessage() for r in caplog.records)

    early = tmp_path / "early" / "01_stance"
    early.mkdir(parents=True)
    _write_ledgers(early, train=False)
    staging = tmp_path / "early_staging"
    staged = stage_artifacts._write_training_history(early, staging)
    assert staged == stage_layout.training_history_dir(staging)
    document = json.loads((staged / training_history.JSON_FILENAME).read_text(encoding="utf-8"))
    run = document["runs"][0]
    assert document["errors"] == []
    assert "train_episodes.csv" in run["absent"] and "models/" in run["absent"]
    assert "eval_episodes.csv: column r_foot_cop_fore_aft" in run["absent"]
    assert not list(early.rglob("training_history*")), "the stage directory itself is only read"

    # No graphs asked for, no report either: it is drawn inside the figures' switch.
    stage_dir = tmp_path / "no_graphs" / "01_stance"
    stage_dir.mkdir(parents=True)
    _write_ledgers(stage_dir)
    with patch.object(stage_artifacts, "_write_training_history") as drawn:
        _judge_velociraptor_mongoliensis_stage(stage_dir, generate_graphs=False)
    drawn.assert_not_called()
    assert not stage_layout.figures_dir(stage_dir).exists()


def test_the_training_history_leaves_the_global_rng_streams_untouched(tmp_path):
    """The probes after it load policies and roll episodes; the std reads draw nothing they would see."""
    pytest.importorskip("matplotlib")
    torch = pytest.importorskip("torch")
    import io
    import json
    import random
    import zipfile

    import numpy as np

    from environments.shared.reporting import stage_artifacts, training_history

    stage_dir = tmp_path / "run" / "01_stance"
    (stage_dir / "models").mkdir(parents=True)
    _write_ledgers(stage_dir)
    for name, steps in (("stage1_50000_steps.zip", None), ("stage1_final.zip", 100_000)):
        buffer = io.BytesIO()
        torch.save({"log_std": torch.tensor([0.0, -1.0])}, buffer)
        with zipfile.ZipFile(stage_dir / "models" / name, "w") as archive:
            archive.writestr("policy.pth", buffer.getvalue())
            archive.writestr("data", json.dumps({"num_timesteps": steps} if steps else {}))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    before = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    staged = stage_artifacts._write_training_history(stage_dir, tmp_path / "staging")
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    document = json.loads((staged / training_history.JSON_FILENAME).read_text(encoding="utf-8"))
    assert document["runs"][0]["policy_std"]["read"] == 2
    assert before[0] == after[0]
    assert all(np.array_equal(a, b) for a, b in zip(before[1][1:], after[1][1:])) and before[1][0] == after[1][0]
    assert torch.equal(before[2], after[2])


def test_the_training_history_follows_the_verdict_and_precedes_the_replays_and_probes():
    """The ordering pin: drawn after the gate progress figure, after the verdict and summary, before the replays."""
    import inspect

    from environments.shared.reporting import stage_artifacts

    source = inspect.getsource(stage_artifacts.generate_stage_artifacts)
    assert (
        source.index("_apply_stage_gate(")
        < source.index("write_stage_summary(")
        < source.index("staged_artifacts(")
        < source.index("plot_gate_progress(")
        < source.index("_write_training_history(")
        < source.index("_record_stage_replays(")
        < source.index("_run_stance_probes(")
    )
    helper = inspect.getsource(stage_artifacts._write_training_history)
    # The library API, not the script: the script refuses an --out inside a run.
    assert "write_report(" in helper and "plot_training_history" not in helper.split('"""')[-1]
