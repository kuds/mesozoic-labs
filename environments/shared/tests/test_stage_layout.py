"""Tests for the stage artifact layout and its local-staging publish."""

from __future__ import annotations

from pathlib import Path

import pytest

from environments.shared.reporting import stage_layout


def _touch(path: Path, content: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


class TestDirectoryAccessors:
    def test_dirs_are_not_created_unless_asked(self, tmp_path):
        assert stage_layout.figures_dir(tmp_path) == tmp_path / "figures"
        assert stage_layout.replays_dir(tmp_path) == tmp_path / "replays"
        assert not (tmp_path / "figures").exists()
        assert not (tmp_path / "replays").exists()

    def test_create_makes_the_directory(self, tmp_path):
        assert stage_layout.figures_dir(tmp_path, create=True).is_dir()
        assert stage_layout.replays_dir(tmp_path, create=True).is_dir()


class TestFindFigure:
    def test_prefers_the_nested_form(self, tmp_path):
        _touch(tmp_path / "training_curves.png")
        nested = _touch(tmp_path / "figures" / "training_curves.png")
        assert stage_layout.find_figure(tmp_path, "training_curves.png") == nested

    def test_falls_back_to_the_legacy_flat_form(self, tmp_path):
        legacy = _touch(tmp_path / "training_curves.png")
        assert stage_layout.find_figure(tmp_path, "training_curves.png") == legacy

    def test_returns_none_when_the_figure_was_never_produced(self, tmp_path):
        assert stage_layout.find_figure(tmp_path, "training_curves.png") is None


class TestIterFigures:
    def test_finds_nested_figures(self, tmp_path):
        _touch(tmp_path / "figures" / "training_curves.png")
        _touch(tmp_path / "figures" / "foot_contacts.png")
        names = {p.name for p in stage_layout.iter_figures(tmp_path)}
        assert names == {"training_curves.png", "foot_contacts.png"}

    def test_finds_legacy_flat_figures(self, tmp_path):
        # The layout the ~35 existing runs on Drive use.
        for name in stage_layout.FIGURE_NAMES:
            _touch(tmp_path / name)
        names = {p.name for p in stage_layout.iter_figures(tmp_path)}
        assert names == set(stage_layout.FIGURE_NAMES)

    def test_a_nested_figure_shadows_its_legacy_twin(self, tmp_path):
        _touch(tmp_path / "training_curves.png")
        _touch(tmp_path / "figures" / "training_curves.png")
        found = list(stage_layout.iter_figures(tmp_path))
        assert len(found) == 1
        assert found[0].parent.name == "figures"


class TestIterReplayFiles:
    def test_finds_nested_videos_and_stance_csvs(self, tmp_path):
        _touch(tmp_path / "replays" / "trex_ppo_stage1_best.mp4")
        _touch(tmp_path / "replays" / "trex_ppo_stage1_best_side.mp4")
        _touch(tmp_path / "replays" / "trex_ppo_stage1_best_stance.csv")
        names = {p.name for p in stage_layout.iter_replay_files(tmp_path)}
        assert names == {
            "trex_ppo_stage1_best.mp4",
            "trex_ppo_stage1_best_side.mp4",
            "trex_ppo_stage1_best_stance.csv",
        }

    def test_legacy_flat_layout_excludes_the_publication_evidence_csvs(self, tmp_path):
        # A legacy stage root holds the replays beside evaluation_*.csv. Those
        # are the result-bundle evidence files, not replay artifacts; sweeping
        # them in here would upload them twice and imply they moved.
        _touch(tmp_path / "trex_ppo_stage1_best.mp4")
        _touch(tmp_path / "trex_ppo_stage1_best_stance.csv")
        _touch(tmp_path / "evaluation_selected.csv")
        _touch(tmp_path / "evaluation_final.csv")
        names = {p.name for p in stage_layout.iter_replay_files(tmp_path)}
        assert names == {"trex_ppo_stage1_best.mp4", "trex_ppo_stage1_best_stance.csv"}

    def test_a_nested_replay_shadows_its_legacy_twin(self, tmp_path):
        _touch(tmp_path / "trex_ppo_stage1_best.mp4")
        _touch(tmp_path / "replays" / "trex_ppo_stage1_best.mp4")
        found = list(stage_layout.iter_replay_files(tmp_path))
        assert len(found) == 1
        assert found[0].parent.name == "replays"


class TestPublishTree:
    def test_preserves_relative_paths(self, tmp_path):
        source = tmp_path / "staging"
        destination = tmp_path / "stage1"
        _touch(source / "figures" / "training_curves.png", b"png-bytes")
        _touch(source / "replays" / "best.mp4", b"mp4-bytes")

        count, total = stage_layout.publish_tree(source, destination)

        assert count == 2
        assert total == len(b"png-bytes") + len(b"mp4-bytes")
        assert (destination / "figures" / "training_curves.png").read_bytes() == b"png-bytes"
        assert (destination / "replays" / "best.mp4").read_bytes() == b"mp4-bytes"

    def test_missing_source_is_not_an_error(self, tmp_path):
        assert stage_layout.publish_tree(tmp_path / "nope", tmp_path / "out") == (0, 0)

    def test_one_unreadable_artifact_does_not_sink_the_rest(self, tmp_path, monkeypatch):
        source = tmp_path / "staging"
        destination = tmp_path / "stage1"
        _touch(source / "figures" / "a.png", b"aa")
        _touch(source / "figures" / "b.png", b"bb")

        from environments.shared import file_io

        real_copy = file_io.atomic_copy

        def flaky_copy(src, dst):
            if Path(src).name == "a.png":
                raise OSError("simulated Drive hiccup")
            real_copy(src, dst)

        monkeypatch.setattr(file_io, "atomic_copy", flaky_copy)

        count, _ = stage_layout.publish_tree(source, destination)

        assert count == 1
        assert not (destination / "figures" / "a.png").exists()
        assert (destination / "figures" / "b.png").read_bytes() == b"bb"


class TestStagedArtifacts:
    def test_writes_land_in_the_stage_dir_on_exit(self, tmp_path):
        stage_dir = tmp_path / "stage1"
        stage_dir.mkdir()

        with stage_layout.staged_artifacts(stage_dir) as staging:
            assert staging != stage_dir, "artifacts must render off the Drive mount"
            _touch(stage_layout.figures_dir(staging, create=True) / "training_curves.png", b"fig")
            # Nothing is published until the block exits.
            assert not (stage_dir / "figures").exists()

        assert (stage_dir / "figures" / "training_curves.png").read_bytes() == b"fig"

    def test_the_scratch_directory_is_removed(self, tmp_path):
        stage_dir = tmp_path / "stage1"
        with stage_layout.staged_artifacts(stage_dir) as staging:
            captured = staging
        assert not captured.exists()

    def test_artifacts_that_rendered_survive_a_later_failure(self, tmp_path):
        # generate_stage_artifacts records the best replay, then the final one.
        # A plant mismatch on the second must not discard the first.
        stage_dir = tmp_path / "stage1"
        with pytest.raises(RuntimeError):
            with stage_layout.staged_artifacts(stage_dir) as staging:
                _touch(stage_layout.replays_dir(staging, create=True) / "best.mp4", b"ok")
                raise RuntimeError("plant identity mismatch on the final checkpoint")
        assert (stage_dir / "replays" / "best.mp4").read_bytes() == b"ok"

    def test_disabled_writes_straight_through(self, tmp_path):
        stage_dir = tmp_path / "stage1"
        with stage_layout.staged_artifacts(stage_dir, enabled=False) as staging:
            assert staging == stage_dir
            _touch(stage_layout.figures_dir(staging, create=True) / "training_curves.png", b"fig")
        assert (stage_dir / "figures" / "training_curves.png").read_bytes() == b"fig"

    def test_a_publish_failure_does_not_propagate(self, tmp_path, monkeypatch):
        # Losing the diagnostics must never cost a completed run its result.
        monkeypatch.setattr(
            stage_layout,
            "publish_tree",
            lambda *a, **k: (_ for _ in ()).throw(OSError("Drive unmounted")),
        )
        with stage_layout.staged_artifacts(tmp_path / "stage1") as staging:
            _touch(staging / "figures" / "x.png")


class TestGenerateStageArtifactsLayout:
    """End-to-end: the artifact generator writes the nested layout."""

    def _run(self, tmp_path, video_side_effect=None):
        from unittest.mock import MagicMock, patch

        import numpy as np

        from environments.shared.plant_contract import attach_plant_identity, current_plant_identity
        from environments.shared.reporting import generate_stage_artifacts

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        for name in ("best_model.zip", "best_model_vecnorm.pkl", "stage1_final.zip", "stage1_final_vecnorm.pkl"):
            (model_dir / name).write_bytes(b"fake")
        np.savez(
            str(tmp_path / "evaluations.npz"),
            results=np.array([[10.0, 12.0], [20.0, 22.0]]),
            ep_lengths=np.array([[100, 110], [200, 210]]),
            timesteps=np.array([50000, 100000]),
        )

        species_cfg = MagicMock()
        species_cfg.species = "velociraptor"
        species_cfg.env_class = MagicMock

        fake_model = MagicMock()
        attach_plant_identity(fake_model, current_plant_identity("velociraptor", verify_generated=False))
        algorithm = MagicMock()
        algorithm.load.return_value = fake_model

        captured: list[Path] = []

        def fake_record(*args, **kwargs):
            output_dir = Path(kwargs["output_dir"])
            output_dir.mkdir(parents=True, exist_ok=True)
            label = kwargs["label"]
            captured.append(output_dir)
            # Fail before writing, the way a recorder that rejects the
            # checkpoint's plant identity does: no partial file is produced.
            if video_side_effect is not None:
                video_side_effect(label)
            (output_dir / f"velociraptor_ppo_stage1_{label}.mp4").write_bytes(b"mp4")

        with (
            patch("environments.shared.train_base._ensure_sb3", return_value={"PPO": algorithm, "SAC": algorithm}),
            patch("environments.shared.evaluation.record_stage_video", side_effect=fake_record),
        ):
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
        return captured

    def test_replays_and_figures_land_in_their_subdirectories(self, tmp_path):
        pytest.importorskip("matplotlib")
        captured = self._run(tmp_path)

        # Replays are rendered off the stage directory, then published into it.
        assert captured and all(d != tmp_path for d in captured), "replays must render off the Drive mount"
        assert sorted(p.name for p in (tmp_path / "replays").iterdir()) == [
            "velociraptor_ppo_stage1_best.mp4",
            "velociraptor_ppo_stage1_final.mp4",
        ]
        assert (tmp_path / "figures" / "training_curves.png").is_file()

        # Nothing renderable is left loose in the stage root.
        assert not list(tmp_path.glob("*.mp4"))
        assert not list(tmp_path.glob("*.png"))

        # The unchanged artifacts stay exactly where their readers expect them.
        assert (tmp_path / "stage_summary.txt").is_file()
        assert (tmp_path / "evaluations.npz").is_file()

    def test_the_best_replay_survives_a_failure_on_the_final_one(self, tmp_path):
        def blow_up_on_final(label):
            if label == "final":
                raise RuntimeError("plant identity mismatch")

        self._run(tmp_path, video_side_effect=blow_up_on_final)
        published = sorted(p.name for p in (tmp_path / "replays").iterdir())
        assert published == ["velociraptor_ppo_stage1_best.mp4"]


class TestGcsUploadScope:
    """The GCS upload resolves replays through the layout, videos only."""

    def _uploaded(self, tmp_path, monkeypatch):
        from environments.shared import config

        calls: list[str] = []
        monkeypatch.setattr(
            config,
            "_upload_to_gcs",
            lambda src, bucket, key, **kw: calls.append(key),
        )
        (tmp_path / "plant_identity.json").write_text("{}")
        config.upload_curriculum_artifacts(tmp_path, "trex", "ppo", bucket="b")
        return calls

    def test_uploads_nested_videos_at_their_relative_path(self, tmp_path, monkeypatch):
        stage = tmp_path / "stage1"
        _touch(stage / "replays" / "trex_ppo_stage1_best.mp4")
        keys = self._uploaded(tmp_path, monkeypatch)
        assert "training/trex/{}/stage1/replays/trex_ppo_stage1_best.mp4".format(tmp_path.name) in keys

    def test_still_uploads_a_legacy_flat_video(self, tmp_path, monkeypatch):
        stage = tmp_path / "stage1"
        _touch(stage / "trex_ppo_stage1_best.mp4")
        keys = self._uploaded(tmp_path, monkeypatch)
        assert "training/trex/{}/stage1/trex_ppo_stage1_best.mp4".format(tmp_path.name) in keys

    def test_does_not_start_uploading_figures_or_stance_csvs(self, tmp_path, monkeypatch):
        # Nesting must not quietly enlarge what lands in someone's bucket:
        # a stance CSV is ~1.7 MB and neither it nor the figures ever
        # uploaded before.
        stage = tmp_path / "stage1"
        _touch(stage / "replays" / "trex_ppo_stage1_best.mp4")
        _touch(stage / "replays" / "trex_ppo_stage1_best_stance.csv")
        _touch(stage / "figures" / "training_curves.png")
        keys = self._uploaded(tmp_path, monkeypatch)
        assert not [k for k in keys if k.endswith(".png") or k.endswith("_stance.csv")]
