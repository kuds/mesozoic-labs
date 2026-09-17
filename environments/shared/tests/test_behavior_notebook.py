"""Execute notebook behavior routing and its shared CLI handoff without a training budget."""

from __future__ import annotations

import ast
import json
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("stable_baselines3")

from environments.shared import behavior_notebook as notebook
from environments.shared.species_names import species_display_names

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "sb3_training.ipynb"
SPECIES_NAMES = species_display_names(backend="stable-baselines3")
CONFIG_MARKER = "# ===== SPECIES SELECTION ====="
STORAGE_MARKER = "# ===== BEHAVIOR STORAGE AND RUN PLAN ====="
RUN_MARKER = "# ===== RUN DIRECTION OR TERRAIN BEHAVIOR ====="
DISPLAY_MARKER = "# ===== DISPLAY SAVED BEHAVIOR EVIDENCE ====="
CANONICAL_GUARD = 'if not globals().get("COMMAND_TERRAIN_BEHAVIOR", False):\n'


def _code_cells() -> list[str]:
    return [
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK_PATH.read_text())["cells"]
        if cell["cell_type"] == "code"
    ]


def _cell(marker: str) -> str:
    return next(source for source in _code_cells() if marker in source)


def _edited_cell(source: str, **values: Any) -> types.CodeType:
    """Apply the same literal setting edits an operator makes in Colab."""
    tree = ast.parse(source)
    found = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in values and name not in found:
                node.value = ast.Constant(values[name])
                found.add(name)
    assert found == set(values), (found, values)
    return compile(ast.fix_missing_locations(tree), "sb3_training.ipynb", "exec")


@pytest.fixture
def behavior_files(tmp_path):
    root = tmp_path / "checkout"
    for species in SPECIES_NAMES:
        config_dir = root / "configs" / species / "behaviors"
        config_dir.mkdir(parents=True)
        for name in notebook.BEHAVIOR_RECIPES.values():
            (config_dir / name).write_bytes((REPO_ROOT / "configs" / species / "behaviors" / name).read_bytes())
    model = root / "source model.zip"
    stats = root / "source normalization.pkl"
    model.write_bytes(b"source-model-placeholder")
    stats.write_bytes(b"source-normalizer-placeholder")
    return root, model, stats


def _plan(behavior_files, **overrides):
    root, model, stats = behavior_files
    values = dict(
        repo_root=root,
        log_base=root / "logs",
        species="trex",
        algorithm="ppo",
        behavior="follow_direction",
        checkpoint=str(model),
        vecnormalize=str(stats),
        seed=42,
        run_id="behavior-test",
    )
    values.update(overrides)
    return notebook.build_notebook_behavior_plan(**values)


def _behavior_namespace(behavior_files, **overrides):
    root, model, stats = behavior_files
    namespace: dict[str, Any] = {"Path": Path, "repo_root": root, "IN_COLAB": False}
    values = dict(
        SPECIES="Tyrannosaurus Rex",
        BEHAVIOR="follow_direction",
        BEHAVIOR_CHECKPOINT=str(model),
        BEHAVIOR_VECNORMALIZE=str(stats),
        USE_GOOGLE_DRIVE=False,
        AUTO_DISCONNECT=False,
    )
    values.update(overrides)
    exec(_edited_cell(_cell(CONFIG_MARKER), **values), namespace)
    return namespace


@pytest.mark.parametrize("species", SPECIES_NAMES)
@pytest.mark.parametrize("behavior", notebook.BEHAVIOR_RECIPES)
def test_each_dropdown_behavior_resolves_an_executable_recipe(behavior_files, species, behavior):
    from environments.shared.train_behaviors import read_recipe

    plan = _plan(behavior_files, species=species, behavior=behavior)
    recipe, commands, terrain, _ = read_recipe(plan.recipe_path, species=plan.species)
    assert recipe["behavior"]["timesteps"] > 0
    assert recipe["behavior"]["species"] == plan.species == species
    assert recipe["behavior"]["name"] == behavior
    assert plan.argv()[plan.argv().index("--species") + 1] == species
    assert plan.argv()[plan.argv().index("--recipe") + 1] == str(plan.recipe_path)
    assert plan.output_dir.parts[-5:-3] == (species, "ppo")
    assert commands is not None
    assert (terrain is None) == (behavior in {"follow_direction", "follow_direction_speed"})
    assert plan.output_dir.parts[-3:] == ("behaviors", behavior, "behavior-test")
    assert not plan.output_dir.exists()


@pytest.mark.parametrize(
    "override,match",
    [
        ({"species": "not-a-species"}, "Unknown species"),
        ({"algorithm": "sac"}, "ppo"),
        ({"behavior": "unknown"}, "Unknown direction or terrain behavior"),
        ({"load_mode": "legacy"}, "prepare, resume, or adapt"),
        ({"checkpoint": ""}, "both BEHAVIOR"),
        ({"vecnormalize": " "}, "both BEHAVIOR"),
        ({"trunk_from": "previous"}, "Clear TRUNK"),
        ({"widen_from": "previous"}, "Clear TRUNK"),
        ({"retrain_from": "stance"}, "Clear TRUNK"),
    ],
)
def test_behavior_selection_refuses_ambiguous_or_incompatible_source(override, match):
    values = dict(
        species="trex",
        algorithm="ppo",
        behavior="follow_direction",
        checkpoint="a.zip",
        vecnormalize="a.pkl",
        load_mode="prepare",
    )
    values.update(override)
    with pytest.raises(ValueError, match=match):
        notebook.validate_behavior_selection(**values)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"seed": -1}, "BEHAVIOR_SEED"),
        ({"seed": True}, "BEHAVIOR_SEED"),
        ({"seed": 2**32}, "BEHAVIOR_SEED"),
        ({"steps": 1.5}, "BEHAVIOR_STEPS"),
        ({"eval_episodes": -1}, "BEHAVIOR_EVAL_EPISODES"),
        ({"eval_episodes": 0}, "at least one"),
        ({"video_fps": float("nan")}, "BEHAVIOR_VIDEO_FPS"),
        ({"video_fps": 0}, "BEHAVIOR_VIDEO_FPS"),
        ({"run_id": "../escape"}, "directory name"),
        ({"run_id": "a\\b"}, "directory name"),
        ({"checkpoint": "missing.zip"}, "source file not found"),
    ],
)
def test_plan_refuses_invalid_inputs_before_creating_outputs(behavior_files, override, match):
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        _plan(behavior_files, **override)
    assert not (behavior_files[0] / "logs").exists()


@pytest.mark.parametrize("mode", ["prepare", "resume", "adapt"])
def test_modes_use_the_shared_runner_and_preserve_default_remaining_budget(behavior_files, mode):
    plan = _plan(behavior_files, load_mode=mode)
    args = plan.argv()
    assert "--steps" not in args
    assert ("--resume" in args) == (mode == "resume")
    assert ("--adapt" in args) == (mode == "adapt")
    assert args[args.index("--checkpoint") + 1] == str(behavior_files[1])
    assert args[args.index("--vecnormalize") + 1] == str(behavior_files[2])
    assert "--record-video" in args


@pytest.mark.parametrize("mode,eval_only", [("prepare", False), ("resume", False), ("prepare", True)])
def test_automatic_source_is_resolved_by_runner_after_identity_is_known_without_early_outputs(
    behavior_files, mode, eval_only
):
    plan = _plan(behavior_files, checkpoint="", vecnormalize="", load_mode=mode, eval_only=eval_only)
    assert plan.auto_source
    assert plan.checkpoint_path is None and plan.vecnormalize_path is None
    assert plan.certified_library == behavior_files[0] / "certified"
    args = plan.argv()
    assert "--auto-source" in args and "--publish-certified" in args
    assert "--checkpoint" not in args and "--vecnormalize" not in args
    assert args[args.index("--comparison-episodes") + 1] == "50"
    assert not plan.output_dir.exists() and not plan.certified_library.exists()


@pytest.mark.parametrize("options", [{"source_selection": "manual"}, {"load_mode": "adapt"}])
def test_manual_selection_and_cross_behavior_adaptation_require_an_explicit_pair(behavior_files, options):
    with pytest.raises(ValueError, match="both BEHAVIOR_CHECKPOINT"):
        _plan(behavior_files, checkpoint="", vecnormalize="", **options)


def test_explicit_pair_overrides_auto_and_library_publication_can_be_disabled(behavior_files):
    plan = _plan(behavior_files, source_selection="auto", publish_certified=False)
    assert not plan.auto_source
    assert "--auto-source" not in plan.argv()
    assert "--publish-certified" not in plan.argv()
    assert "--comparison-episodes" not in plan.argv()
    assert plan.checkpoint_path == behavior_files[1]


@pytest.mark.parametrize("count", [2, 50, 100])
def test_comparison_panel_size_is_an_explicit_independent_control(behavior_files, count):
    plan = _plan(behavior_files, comparison_episodes=count, eval_episodes=5)
    args = plan.argv()
    assert args[args.index("--comparison-episodes") + 1] == str(count)
    assert args[args.index("--eval-episodes") + 1] == "5"


@pytest.mark.parametrize("count", [0, 1, -1, True, 1.5])
def test_comparison_panel_size_refuses_invalid_values_before_creating_outputs(behavior_files, count):
    with pytest.raises(ValueError, match="CERTIFIED_COMPARISON_EPISODES"):
        _plan(behavior_files, comparison_episodes=count)
    assert not (behavior_files[0] / "logs").exists()


def test_quick_test_explicit_steps_and_eval_only_have_clear_budget_semantics(behavior_files):
    assert _plan(behavior_files, quick_test=True).steps == 4096
    assert _plan(behavior_files, quick_test=True, steps=12).steps == 12
    assert _plan(behavior_files, quick_test=True, eval_only=True).steps is None
    args = _plan(behavior_files, eval_only=True, steps=0, record_video=False, eval_episodes=0).argv()
    assert args[args.index("--steps") + 1] == "0"
    assert "--eval-only" in args
    assert "--record-video" not in args


def test_quick_test_saves_why_certification_was_skipped_without_relaxing_requirements(behavior_files, monkeypatch):
    from environments.shared import train_behaviors

    plan = _plan(behavior_files, quick_test=True)
    assert not plan.publish_certified
    assert "--publish-certified" not in plan.argv()
    assert "QUICK_TEST" in plan.certification_skip_reason

    def run(args):
        plan.output_dir.mkdir(parents=True)
        (plan.output_dir / "run.json").write_text(
            json.dumps({"schema": "mesozoic.behavior-run/v1", "canonical_certification": False, "status": "complete"})
        )

    monkeypatch.setattr(train_behaviors, "main", run)
    report = notebook.run_notebook_behavior(plan)
    saved = json.loads((plan.output_dir / "run.json").read_text())
    assert report["certification_skip_reason"] == saved["certification_skip_reason"] == plan.certification_skip_reason
    assert "gates were not relaxed" in saved["certification_skip_reason"]


def test_fresh_seed_default_and_explicit_replay_seed(behavior_files, monkeypatch):
    seeds = iter((101, 202))
    monkeypatch.setattr(notebook.secrets, "randbelow", lambda bound: next(seeds))
    first = _plan(behavior_files, seed=None, run_id="")
    second = _plan(behavior_files, seed=None, run_id="")
    replay = _plan(behavior_files, seed=first.seed)
    assert (first.seed, second.seed, replay.seed) == (101, 202, 101)
    assert first.output_dir != second.output_dir


def test_identical_storage_rerun_retains_plan_and_changed_selection_mints_seed(behavior_files, monkeypatch):
    seeds = iter((101, 202, 303, 404, 505))
    monkeypatch.setattr(notebook.secrets, "randbelow", lambda bound: next(seeds))
    namespace = _behavior_namespace(behavior_files)
    source = _cell(STORAGE_MARKER)
    exec(source, namespace)
    first = namespace["BEHAVIOR_PLAN"]
    exec(source, namespace)
    assert namespace["BEHAVIOR_PLAN"] == first
    for key, value, expected_seed in (
        ("BEHAVIOR_RUN_ID", "new-session", 202),
        ("BEHAVIOR", "mixed_terrain", 303),
        ("BEHAVIOR_LOAD_MODE", "resume", 404),
        ("SPECIES", "velociraptor", 505),
    ):
        namespace[key] = value
        exec(source, namespace)
        assert namespace["BEHAVIOR_PLAN"].seed == expected_seed
    namespace.update(BEHAVIOR_SEED=818, BEHAVIOR_RUN_ID="explicit-replay")
    exec(source, namespace)
    assert namespace["BEHAVIOR_PLAN"].seed == 818


def test_colab_storage_mounts_drive_and_uses_a_separate_behavior_tree(behavior_files, monkeypatch):
    calls = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda location: calls.append(location))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = _behavior_namespace(behavior_files, USE_GOOGLE_DRIVE=True, BEHAVIOR_SEED=72)
    namespace["IN_COLAB"] = True
    exec(_cell(STORAGE_MARKER), namespace)
    plan = namespace["BEHAVIOR_PLAN"]
    assert calls == ["/content/drive"]
    assert str(plan.output_dir).startswith("/content/drive/MyDrive/mesozoic-labs/logs/trex/ppo/behaviors/")
    assert "RUN_DIR" not in namespace and "PLANT_IDENTITY" not in namespace
    assert namespace["CHAIN"] == []
    assert plan.certified_library == Path("/content/drive/MyDrive/mesozoic-labs/certified")


def test_notebook_defaults_to_automatic_complete_bundle_selection(behavior_files):
    namespace = _behavior_namespace(behavior_files, BEHAVIOR_CHECKPOINT="", BEHAVIOR_VECNORMALIZE="")
    assert namespace["SOURCE_SELECTION"] == "auto"
    assert namespace["PUBLISH_CERTIFIED"] is False
    assert namespace["CERTIFIED_COMPARISON_EPISODES"] == 50
    exec(_cell(STORAGE_MARKER), namespace)
    plan = namespace["BEHAVIOR_PLAN"]
    assert plan.auto_source and plan.checkpoint_path is None
    assert plan.certified_library == behavior_files[0] / "certified"
    assert not plan.output_dir.exists()


def test_source_library_and_comparison_changes_make_a_new_behavior_plan(behavior_files, monkeypatch):
    seeds = iter((101, 202, 303, 404))
    monkeypatch.setattr(notebook.secrets, "randbelow", lambda bound: next(seeds))
    namespace = _behavior_namespace(behavior_files)
    source = _cell(STORAGE_MARKER)
    exec(source, namespace)
    for key, value, seed in (
        ("CERTIFIED_LIBRARY_ROOT", str(behavior_files[0] / "other-library"), 202),
        ("CERTIFIED_COMPARISON_EPISODES", 100, 303),
        ("PUBLISH_CERTIFIED", True, 404),
    ):
        namespace[key] = value
        exec(source, namespace)
        assert namespace["BEHAVIOR_PLAN"].seed == seed


@pytest.mark.parametrize("schema", ["mesozoic.behavior-run/v1", "mesozoic.behavior-pilot-run/v1"])
def test_runner_receives_exact_plan_and_returns_diagnostic_manifest(behavior_files, monkeypatch, schema):
    from environments.shared import train_behaviors

    plan = _plan(behavior_files, load_mode="adapt", steps=1234, seed=999)
    calls = []
    expected = {"schema": schema, "canonical_certification": False, "status": "complete"}

    def fake_main(args):
        calls.append(args)
        plan.output_dir.mkdir(parents=True)
        (plan.output_dir / "run.json").write_text(json.dumps(expected))

    monkeypatch.setattr(train_behaviors, "main", fake_main)
    assert notebook.run_notebook_behavior(plan) == expected
    assert calls == [plan.argv()]


@pytest.mark.parametrize("mode", ["resume", "adapt"])
def test_notebook_uses_actual_runner_bundle_refusal(behavior_files, mode):
    plan = _plan(behavior_files, load_mode=mode)
    with pytest.raises(ValueError, match="bundle.json"):
        notebook.run_notebook_behavior(plan)
    assert not plan.output_dir.exists()


def test_saved_video_and_matching_maps_are_displayed_without_changing_fps(tmp_path, monkeypatch):
    import IPython.display as ipy

    received = []
    replay = {
        key: str(tmp_path / name)
        for key, name in (
            ("video", "replay.mp4"),
            ("full_map", "terrain_full_map.png"),
            ("local_map", "terrain_local_map.png"),
        )
    }
    for filename in replay.values():
        Path(filename).write_bytes(b"saved-media")
    report = {
        "status": "complete",
        "run_seed": 24,
        "training": {"actual_additional_steps": 4096},
        "evaluation": {
            "full_horizon_count": 1,
            "episode_count": 1,
            "fall_count": 0,
            "episodes": [{"episode": 0, "episode_seed": 25, "replay": replay}],
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(report))
    monkeypatch.setattr(ipy, "Video", lambda **kwargs: ("video", kwargs))
    monkeypatch.setattr(ipy, "Image", lambda **kwargs: ("image", kwargs))
    monkeypatch.setattr(ipy, "display", received.append)
    notebook.display_notebook_behavior(tmp_path)
    assert received == [
        ("video", {"filename": replay["video"], "embed": True}),
        ("image", {"filename": replay["full_map"]}),
        ("image", {"filename": replay["local_map"]}),
    ]


def _saved_replay_run(root, *, path_style="absolute"):
    """Write the production layout with distinct plane/heightfield episode media."""
    root.mkdir()
    episodes, assets = [], []
    recorded_root = root if path_style == "absolute" else Path("../behavior-runs") / root.name
    for episode, prefix in enumerate(("flat_plane", "terrain")):
        seed = 25 + episode
        directory = Path("replays") / f"episode_{episode:03d}_seed_{seed}"
        (root / directory).mkdir(parents=True)
        files = {
            "video": "replay.mp4",
            "full_map": f"{prefix}_full_map.png",
            "local_map": f"{prefix}_local_map.png",
        }
        for key, name in files.items():
            relative = directory / name
            payload = f"episode-{episode}-{key}".encode()
            (root / relative).write_bytes(payload)
            assets.append((relative, payload))
        manifest = {
            "schema": "mesozoic.behavior-replay/v1",
            "episode": episode,
            "episode_seed": seed,
            "files": {key: {"path": name} for key, name in files.items()},
        }
        (root / directory / "manifest.json").write_text(json.dumps(manifest))
        replay = {
            "episode": episode,
            "episode_seed": seed,
            "directory": str(recorded_root / directory),
            "manifest": str(recorded_root / directory / "manifest.json"),
            **{key: str(recorded_root / directory / name) for key, name in files.items()},
        }
        episodes.append({"episode": episode, "episode_seed": seed, "replay": replay})
    report = {
        "status": "complete",
        "run_seed": 24,
        "evaluation": {"full_horizon_count": 2, "episode_count": 2, "fall_count": 0, "episodes": episodes},
    }
    (root / "run.json").write_text(json.dumps(report))
    (root / "replays" / "index.json").write_text(
        json.dumps({"schema": "mesozoic.behavior-replay-index/v1", "episodes": [e["replay"] for e in episodes]})
    )
    return assets


def _capture_saved_media(monkeypatch):
    import IPython.display as ipy

    received = []

    def read_media(**kwargs):
        path = Path(kwargs["filename"])
        return path.resolve(), path.read_bytes()

    monkeypatch.setattr(ipy, "Video", read_media)
    monkeypatch.setattr(ipy, "Image", read_media)
    monkeypatch.setattr(ipy, "display", received.append)
    return received


def test_copied_run_uses_its_own_matching_episode_media_even_if_original_exists(tmp_path, monkeypatch):
    original, copied = tmp_path / "original", tmp_path / "downloaded copy"
    assets = _saved_replay_run(original)
    shutil.copytree(original, copied)
    received = _capture_saved_media(monkeypatch)
    notebook.display_notebook_behavior(copied)
    assert received == [(copied / relative, payload) for relative, payload in assets]
    assert all((original / relative).is_file() for relative, _ in assets)


@pytest.mark.parametrize("path_style", ["absolute", "legacy_relative"])
def test_moved_run_displays_after_working_directory_changes(tmp_path, monkeypatch, path_style):
    original, moved = tmp_path / "original", tmp_path / "renamed"
    assets = _saved_replay_run(original, path_style=path_style)
    original.rename(moved)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    received = _capture_saved_media(monkeypatch)
    notebook.display_notebook_behavior(Path("..") / moved.name)
    assert received == [(moved / relative, payload) for relative, payload in assets]


def test_copied_run_missing_map_does_not_fall_back_to_original_media(tmp_path, monkeypatch):
    original, copied = tmp_path / "original", tmp_path / "copied"
    assets = _saved_replay_run(original)
    shutil.copytree(original, copied)
    missing = assets[2][0]
    (copied / missing).unlink()
    received = _capture_saved_media(monkeypatch)
    with pytest.raises(FileNotFoundError, match="flat_plane_local_map.png") as error:
        notebook.display_notebook_behavior(copied)
    assert str(copied) in str(error.value)
    assert (original / missing).is_file()
    assert received == []


@pytest.mark.parametrize("unsafe_path", ["/outside.png", "../outside.png", "nested/../../outside.png"])
def test_replay_manifest_files_must_stay_inside_saved_episode(tmp_path, monkeypatch, unsafe_path):
    root = tmp_path / "saved"
    _saved_replay_run(root)
    manifest_path = root / "replays" / "episode_000_seed_25" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["local_map"]["path"] = unsafe_path
    manifest_path.write_text(json.dumps(manifest))
    received = _capture_saved_media(monkeypatch)
    with pytest.raises(ValueError):
        notebook.display_notebook_behavior(root)
    assert received == []


@pytest.mark.parametrize("evaluation_present", [False, True])
def test_saved_run_without_replays_still_displays_summary(tmp_path, monkeypatch, capsys, evaluation_present):
    report = {"status": "complete", "run_seed": 24}
    if evaluation_present:
        report["evaluation"] = {
            "full_horizon_count": 1,
            "episode_count": 1,
            "fall_count": 0,
            "episodes": [{"episode": 0, "episode_seed": 25}],
        }
    (tmp_path / "run.json").write_text(json.dumps(report))
    received = _capture_saved_media(monkeypatch)
    notebook.display_notebook_behavior(tmp_path)
    assert "Behavior: complete" in capsys.readouterr().out
    assert received == []


def test_saved_behavior_certification_displays_provisional_status_and_recommendation_reason(
    tmp_path, monkeypatch, capsys
):
    report = {
        "status": "complete",
        "run_seed": 24,
        "certification": {
            "passed": True,
            "failures": [],
            "publication": {
                "status": "provisional",
                "version": "v000002",
                "distinct_seeds": 2,
                "required_seeds": 3,
                "decision_reason": "More independent training seeds are required; recommendation unchanged.",
            },
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(report))
    _capture_saved_media(monkeypatch)
    notebook.display_notebook_behavior(tmp_path)
    output = capsys.readouterr().out
    assert "Behavior certification: passed" in output
    assert "Certified library: provisional" in output
    assert "recommendation unchanged" in output
    assert "Distinct training seeds: 2/3" in output


def test_saved_terrain_summary_distinguishes_missing_families_from_passing_results(tmp_path, monkeypatch, capsys):
    report = {
        "status": "complete",
        "run_seed": 24,
        "evaluation": {
            "full_horizon_count": 1,
            "episode_count": 2,
            "fall_count": 1,
            "terrain_coverage": {
                "enabled_families": ["flat", "sloped", "bumps", "depressions", "mixed"],
                "evaluated_families": ["flat", "bumps"],
                "missing_families": ["sloped", "depressions", "mixed"],
                "complete": False,
            },
            "by_terrain_family": {
                "flat": {
                    "episode_count": 1,
                    "full_horizon_count": 1,
                    "fall_count": 0,
                    "tracking_fraction": 0.85,
                    "eligible_event_settle_fraction": 0.5,
                },
                "bumps": {
                    "episode_count": 1,
                    "full_horizon_count": 0,
                    "fall_count": 1,
                    "tracking_fraction": 0.25,
                    "eligible_event_settle_fraction": None,
                },
                "sloped": {"episode_count": 0},
                "depressions": {"episode_count": 0},
                "mixed": {"episode_count": 0},
            },
        },
    }
    (tmp_path / "run.json").write_text(json.dumps(report))
    received = _capture_saved_media(monkeypatch)
    notebook.display_notebook_behavior(tmp_path)
    output = capsys.readouterr().out
    assert "Terrain coverage: incomplete (2/5 families)" in output
    assert "Not evaluated: sloped, depressions, mixed" in output
    assert "flat: 1 episodes; full horizon 1/1; falls 0; tracking 85.0%; commands settled 50.0%" in output
    assert "bumps: 1 episodes; full horizon 0/1; falls 1; tracking 25.0%; commands settled n/a" in output
    assert "mixed: not evaluated (0 episodes)" in output
    assert received == []


def test_canonical_default_and_freeform_stage_resolution_remain_available():
    from environments.shared.config import load_all_stages
    from environments.shared.stage_manifest import load_stage_manifest

    namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
    exec(_cell(CONFIG_MARKER), namespace)
    assert namespace["BEHAVIOR"] == "hunt"
    assert namespace["COMMAND_TERRAIN_BEHAVIOR"] is False
    assert namespace["SPECIES"] == "velociraptor"
    assert namespace["N_ENVS"] == 4 and namespace["SEED"] == 42
    namespace["BEHAVIOR"] = "stance"
    exec(_cell("EnvClass = SPECIES_CFG.env_class"), namespace)
    assert namespace["TARGET_NODE"].id == "stance"
    assert [node.id for node in namespace["CHAIN"]] == ["stance"]


def test_dropdown_has_free_input_and_valid_json_annotations():
    config = _cell(CONFIG_MARKER)
    behavior_line = next(line for line in config.splitlines() if line.startswith("BEHAVIOR ="))
    options, trailing = json.JSONDecoder().raw_decode(behavior_line.split("# @param ", 1)[1])
    assert set(notebook.BEHAVIOR_RECIPES) <= set(options)
    assert json.loads(behavior_line.split("# @param ", 1)[1][trailing:].strip()) == {"allow-input": True}
    for source in (_cell("REPO_REF ="), config):
        for line in source.splitlines():
            if "# @param {" in line:
                assert "type" in json.loads(line.split("# @param ", 1)[1])


def test_every_guarded_canonical_cell_is_inert_during_behaviors():
    guarded = [source for source in _code_cells() if source.startswith(CANONICAL_GUARD)]
    assert len(guarded) == 12
    for source in guarded:
        namespace = {"COMMAND_TERRAIN_BEHAVIOR": True}
        exec(source, namespace)
        assert set(namespace) == {"COMMAND_TERRAIN_BEHAVIOR", "__builtins__"}


def test_chain_cell_clears_stale_canonical_chain_before_loop():
    namespace = {"COMMAND_TERRAIN_BEHAVIOR": True, "CHAIN": [object()]}
    exec(_cell("# ===== BEHAVIOR CHAIN LOOP ====="), namespace)
    assert namespace["CHAIN"] == []
    assert "NODE" not in namespace


@pytest.mark.parametrize("species", SPECIES_NAMES)
@pytest.mark.parametrize(
    "behavior", ["combined_mixed_terrain", "difficult_terrain", "follow_direction_difficult_terrain"]
)
def test_all_local_notebook_code_cells_route_a_behavior_without_canonical_artifacts(
    behavior_files, monkeypatch, species, behavior
):
    """Only long-running behavior training/display are replaced; all cells execute."""
    from environments.shared import train_behaviors

    monkeypatch.chdir(REPO_ROOT)
    root, model, stats = behavior_files
    recorded_args = []
    displays = []

    def fake_main(args):
        recorded_args.append(args)
        output = Path(args[args.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "run.json").write_text(
            json.dumps(
                {
                    "schema": "mesozoic.behavior-run/v1",
                    "canonical_certification": False,
                    "status": "complete",
                    "run_seed": 747,
                }
            )
        )

    monkeypatch.setattr(train_behaviors, "main", fake_main)
    monkeypatch.setattr(notebook, "display_notebook_behavior", displays.append)
    namespace: dict[str, Any] = {}
    for source in _code_cells():
        if "REPO_REF =" in source:
            code = _edited_cell(source, IN_COLAB=False)
        elif CONFIG_MARKER in source:
            code = _edited_cell(
                source,
                SPECIES=SPECIES_NAMES[species],
                BEHAVIOR=behavior,
                BEHAVIOR_CHECKPOINT=str(model),
                BEHAVIOR_VECNORMALIZE=str(stats),
                BEHAVIOR_SEED=747,
                QUICK_TEST=True,
                USE_GOOGLE_DRIVE=False,
                AUTO_DISCONNECT=False,
            )
        else:
            code = compile(source, "sb3_training.ipynb", "exec")
        exec(code, namespace)
        if "Repo root:" in source:
            namespace["repo_root"] = root
    assert namespace["BEHAVIOR_PLAN"].species == species
    assert len(recorded_args) == 1
    assert recorded_args[0] == namespace["BEHAVIOR_PLAN"].argv()
    assert displays == [namespace["BEHAVIOR_PLAN"].output_dir]
    assert namespace["BEHAVIOR_RESULT"]["status"] == "complete"
    assert namespace["completed_stages"] == [] and namespace["NODE_HANDOFF"] == {}
    assert "RUN_DIR" not in namespace and "PLANT_IDENTITY" not in namespace
    assert not list(root.rglob("result_bundle.json"))
    assert not list(root.rglob("gate_verdict.json"))
    Path(namespace["_preflight_zip"]).unlink()
    Path(namespace["_preflight_zip"]).parent.rmdir()


@pytest.mark.parametrize(
    "dirty,loaded,expected", [(True, False, "local edits"), (False, True, "Restart"), (False, False, None)]
)
def test_colab_ref_change_is_explicit_and_preserves_edits(tmp_path, monkeypatch, dirty, loaded, expected):
    """Execute the notebook's actual Git setup block with controlled Git replies."""
    import subprocess

    checkout = tmp_path / "checkout"
    (checkout / ".git").mkdir(parents=True)
    source = _cell("REPO_REF =")
    tree = ast.parse(source)
    colab_block = next(node for node in tree.body if isinstance(node, ast.If))
    start = next(
        i for i, node in enumerate(colab_block.body) if isinstance(node, ast.Import) and node.names[0].name == "pathlib"
    )
    body = colab_block.body[start:]
    for node in body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "repo_dir":
            node.value = ast.Call(
                func=ast.Name(id="Path", ctx=ast.Load()), args=[ast.Constant(str(checkout))], keywords=[]
            )
    commands = []

    def fake_output(command, **kwargs):
        if command[1:3] == ["rev-parse", "FETCH_HEAD^{commit}"]:
            return "new-commit\n"
        if command[1:3] == ["rev-parse", "HEAD"]:
            return "old-commit\n"
        if command[1:3] == ["status", "--porcelain"]:
            return " M notebook.ipynb\n" if dirty else ""
        raise AssertionError(command)

    monkeypatch.setattr(subprocess, "check_output", fake_output)
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: commands.append(command))
    namespace = {
        "Path": Path,
        "REPO_REF": "codex/direction-and-random-terrain",
        "sys": types.SimpleNamespace(modules={"environments": object()} if loaded else {}),
        "importlib": types.SimpleNamespace(util=types.SimpleNamespace(find_spec=lambda name: object())),
    }
    code = compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), "setup", "exec")
    if expected:
        with pytest.raises(RuntimeError, match=expected):
            exec(code, namespace)
        assert not any(command[1] == "checkout" for command in commands)
    else:
        exec(code, namespace)
        assert ["git", "checkout", "--detach", "new-commit"] in commands
    assert commands[0] == ["git", "fetch", "origin", "codex/direction-and-random-terrain"]
    assert not any("--force" in command or "reset" in command or "clean" in command for command in commands)


@pytest.mark.parametrize("species", SPECIES_NAMES)
def test_helper_resolves_species_display_names_to_stable_paths(behavior_files, species):
    plan = _plan(behavior_files, species=SPECIES_NAMES[species])
    assert plan.species == species
    assert plan.recipe_path.parent.parent.name == species


@pytest.mark.parametrize("field,value", [("species", "velociraptor"), ("name", "mixed_terrain")])
def test_recipe_metadata_cannot_route_a_different_species_or_behavior(behavior_files, field, value):
    root, _, _ = behavior_files
    recipe = root / "configs" / "trex" / "behaviors" / "follow_direction.toml"
    original = 'species = "trex"' if field == "species" else 'name = "follow_direction"'
    contents = recipe.read_text()
    assert original in contents
    recipe.write_text(contents.replace(original, f'{field} = "{value}"'))
    with pytest.raises(ValueError, match="does not match selected species/behavior"):
        _plan(behavior_files)
    assert not (root / "logs").exists()


def test_notebook_has_supported_controls_without_pilot_or_trex_only_label():
    text = NOTEBOOK_PATH.read_text()
    assert "Direction and terrain behaviors (all species, PPO)" in text
    assert "PILOT_" not in text
    assert "T-Rex PPO only" not in text
    assert "Before PR 540" not in text
