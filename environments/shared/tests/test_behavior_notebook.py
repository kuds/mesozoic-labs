"""Execute notebook pilot routing and its shared CLI handoff without a training budget."""

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

REPO_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "sb3_training.ipynb"
CONFIG_MARKER = "# ===== SPECIES SELECTION ====="
STORAGE_MARKER = "# ===== PILOT STORAGE AND RUN PLAN ====="
RUN_MARKER = "# ===== RUN DIRECTION OR TERRAIN PILOT ====="
DISPLAY_MARKER = "# ===== DISPLAY SAVED PILOT EVIDENCE ====="
CANONICAL_GUARD = 'if not globals().get("BEHAVIOR_PILOT", False):\n'


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
def pilot_files(tmp_path):
    root = tmp_path / "checkout"
    config_dir = root / "configs" / "trex" / "behavior_pilots"
    config_dir.mkdir(parents=True)
    for name in notebook.PILOT_RECIPES.values():
        (config_dir / name).write_bytes((REPO_ROOT / "configs/trex/behavior_pilots" / name).read_bytes())
    model = root / "source model.zip"
    stats = root / "source normalization.pkl"
    model.write_bytes(b"source-model-placeholder")
    stats.write_bytes(b"source-normalizer-placeholder")
    return root, model, stats


def _plan(pilot_files, **overrides):
    root, model, stats = pilot_files
    values = dict(
        repo_root=root,
        log_base=root / "logs",
        species="trex",
        algorithm="ppo",
        behavior="follow_direction",
        checkpoint=str(model),
        vecnormalize=str(stats),
        seed=42,
        run_id="pilot-test",
    )
    values.update(overrides)
    return notebook.build_notebook_pilot_plan(**values)


def _pilot_namespace(pilot_files, **overrides):
    root, model, stats = pilot_files
    namespace: dict[str, Any] = {"Path": Path, "repo_root": root, "IN_COLAB": False}
    values = dict(
        SPECIES="Tyrannosaurus Rex",
        BEHAVIOR="follow_direction",
        PILOT_CHECKPOINT=str(model),
        PILOT_VECNORMALIZE=str(stats),
        USE_GOOGLE_DRIVE=False,
        AUTO_DISCONNECT=False,
    )
    values.update(overrides)
    exec(_edited_cell(_cell(CONFIG_MARKER), **values), namespace)
    return namespace


@pytest.mark.parametrize("behavior", notebook.PILOT_RECIPES)
def test_each_dropdown_pilot_resolves_an_executable_recipe(pilot_files, behavior):
    from environments.trex.scripts.train_behaviors import read_recipe

    plan = _plan(pilot_files, behavior=behavior)
    recipe, commands, terrain, _ = read_recipe(plan.recipe_path)
    assert recipe["pilot"]["timesteps"] > 0
    assert commands is not None
    assert (terrain is None) == (behavior in {"follow_direction", "follow_direction_speed"})
    assert plan.output_dir.parts[-3:] == ("behavior_pilots", behavior, "pilot-test")
    assert not plan.output_dir.exists()


@pytest.mark.parametrize(
    "override,match",
    [
        ({"species": "velociraptor"}, "Tyrannosaurus"),
        ({"algorithm": "sac"}, "ppo"),
        ({"behavior": "unknown"}, "Unknown behavior"),
        ({"load_mode": "legacy"}, "prepare, resume, or adapt"),
        ({"checkpoint": ""}, "both PILOT"),
        ({"vecnormalize": " "}, "both PILOT"),
        ({"trunk_from": "previous"}, "Clear TRUNK"),
        ({"widen_from": "previous"}, "Clear TRUNK"),
        ({"retrain_from": "stance"}, "Clear TRUNK"),
    ],
)
def test_pilot_selection_refuses_ambiguous_or_incompatible_source(override, match):
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
        notebook.validate_pilot_selection(**values)


@pytest.mark.parametrize(
    "override,match",
    [
        ({"seed": -1}, "PILOT_SEED"),
        ({"seed": True}, "PILOT_SEED"),
        ({"seed": 2**32}, "PILOT_SEED"),
        ({"steps": 1.5}, "PILOT_STEPS"),
        ({"eval_episodes": -1}, "PILOT_EVAL_EPISODES"),
        ({"eval_episodes": 0}, "at least one"),
        ({"video_fps": float("nan")}, "PILOT_VIDEO_FPS"),
        ({"video_fps": 0}, "PILOT_VIDEO_FPS"),
        ({"run_id": "../escape"}, "directory name"),
        ({"run_id": "a\\b"}, "directory name"),
        ({"checkpoint": "missing.zip"}, "source file not found"),
    ],
)
def test_plan_refuses_invalid_inputs_before_creating_outputs(pilot_files, override, match):
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        _plan(pilot_files, **override)
    assert not (pilot_files[0] / "logs").exists()


@pytest.mark.parametrize("mode", ["prepare", "resume", "adapt"])
def test_modes_use_the_shared_runner_and_preserve_default_remaining_budget(pilot_files, mode):
    plan = _plan(pilot_files, load_mode=mode)
    args = plan.argv()
    assert "--steps" not in args
    assert ("--resume" in args) == (mode == "resume")
    assert ("--adapt" in args) == (mode == "adapt")
    assert args[args.index("--checkpoint") + 1] == str(pilot_files[1])
    assert args[args.index("--vecnormalize") + 1] == str(pilot_files[2])
    assert "--record-video" in args


def test_quick_test_explicit_steps_and_eval_only_have_clear_budget_semantics(pilot_files):
    assert _plan(pilot_files, quick_test=True).steps == 4096
    assert _plan(pilot_files, quick_test=True, steps=12).steps == 12
    assert _plan(pilot_files, quick_test=True, eval_only=True).steps is None
    args = _plan(pilot_files, eval_only=True, steps=0, record_video=False, eval_episodes=0).argv()
    assert args[args.index("--steps") + 1] == "0"
    assert "--eval-only" in args
    assert "--record-video" not in args


def test_fresh_seed_default_and_explicit_replay_seed(pilot_files, monkeypatch):
    seeds = iter((101, 202))
    monkeypatch.setattr(notebook.secrets, "randbelow", lambda bound: next(seeds))
    first = _plan(pilot_files, seed=None, run_id="")
    second = _plan(pilot_files, seed=None, run_id="")
    replay = _plan(pilot_files, seed=first.seed)
    assert (first.seed, second.seed, replay.seed) == (101, 202, 101)
    assert first.output_dir != second.output_dir


def test_identical_storage_rerun_retains_plan_and_changed_selection_mints_seed(pilot_files, monkeypatch):
    seeds = iter((101, 202, 303, 404))
    monkeypatch.setattr(notebook.secrets, "randbelow", lambda bound: next(seeds))
    namespace = _pilot_namespace(pilot_files)
    source = _cell(STORAGE_MARKER)
    exec(source, namespace)
    first = namespace["PILOT_PLAN"]
    exec(source, namespace)
    assert namespace["PILOT_PLAN"] == first
    for key, value, expected_seed in (
        ("PILOT_RUN_ID", "new-session", 202),
        ("BEHAVIOR", "mixed_terrain", 303),
        ("PILOT_LOAD_MODE", "resume", 404),
    ):
        namespace[key] = value
        exec(source, namespace)
        assert namespace["PILOT_PLAN"].seed == expected_seed
    namespace.update(PILOT_SEED=818, PILOT_RUN_ID="explicit-replay")
    exec(source, namespace)
    assert namespace["PILOT_PLAN"].seed == 818


def test_colab_storage_mounts_drive_and_uses_a_separate_pilot_tree(pilot_files, monkeypatch):
    calls = []
    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=lambda location: calls.append(location))
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    namespace = _pilot_namespace(pilot_files, USE_GOOGLE_DRIVE=True, PILOT_SEED=72)
    namespace["IN_COLAB"] = True
    exec(_cell(STORAGE_MARKER), namespace)
    plan = namespace["PILOT_PLAN"]
    assert calls == ["/content/drive"]
    assert str(plan.output_dir).startswith("/content/drive/MyDrive/mesozoic-labs/logs/trex/ppo/behavior_pilots/")
    assert "RUN_DIR" not in namespace and "PLANT_IDENTITY" not in namespace
    assert namespace["CHAIN"] == []


def test_runner_receives_exact_plan_and_returns_diagnostic_manifest(pilot_files, monkeypatch):
    from environments.trex.scripts import train_behaviors

    plan = _plan(pilot_files, load_mode="adapt", steps=1234, seed=999)
    calls = []
    expected = {"schema": "mesozoic.behavior-pilot-run/v1", "canonical_certification": False, "status": "complete"}

    def fake_main(args):
        calls.append(args)
        plan.output_dir.mkdir(parents=True)
        (plan.output_dir / "run.json").write_text(json.dumps(expected))

    monkeypatch.setattr(train_behaviors, "main", fake_main)
    assert notebook.run_notebook_pilot(plan) == expected
    assert calls == [plan.argv()]


@pytest.mark.parametrize("mode", ["resume", "adapt"])
def test_notebook_uses_actual_runner_bundle_refusal(pilot_files, mode):
    plan = _plan(pilot_files, load_mode=mode)
    with pytest.raises(ValueError, match="bundle.json"):
        notebook.run_notebook_pilot(plan)
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
    notebook.display_notebook_pilot(tmp_path)
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
    notebook.display_notebook_pilot(copied)
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
    notebook.display_notebook_pilot(Path("..") / moved.name)
    assert received == [(moved / relative, payload) for relative, payload in assets]


def test_copied_run_missing_map_does_not_fall_back_to_original_media(tmp_path, monkeypatch):
    original, copied = tmp_path / "original", tmp_path / "copied"
    assets = _saved_replay_run(original)
    shutil.copytree(original, copied)
    missing = assets[2][0]
    (copied / missing).unlink()
    received = _capture_saved_media(monkeypatch)
    with pytest.raises(FileNotFoundError, match="flat_plane_local_map.png") as error:
        notebook.display_notebook_pilot(copied)
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
        notebook.display_notebook_pilot(root)
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
    notebook.display_notebook_pilot(tmp_path)
    assert "Behavior pilot: complete" in capsys.readouterr().out
    assert received == []


def test_canonical_default_and_freeform_stage_resolution_remain_available():
    from environments.shared.config import load_all_stages
    from environments.shared.stage_manifest import load_stage_manifest

    namespace = {"load_all_stages": load_all_stages, "load_stage_manifest": load_stage_manifest}
    exec(_cell(CONFIG_MARKER), namespace)
    assert namespace["BEHAVIOR"] == "hunt"
    assert namespace["BEHAVIOR_PILOT"] is False
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
    assert set(notebook.PILOT_RECIPES) <= set(options)
    assert json.loads(behavior_line.split("# @param ", 1)[1][trailing:].strip()) == {"allow-input": True}
    for source in (_cell("REPO_REF ="), config):
        for line in source.splitlines():
            if "# @param {" in line:
                assert "type" in json.loads(line.split("# @param ", 1)[1])


def test_every_guarded_canonical_cell_is_inert_during_pilots():
    guarded = [source for source in _code_cells() if source.startswith(CANONICAL_GUARD)]
    assert len(guarded) == 12
    for source in guarded:
        namespace = {"BEHAVIOR_PILOT": True}
        exec(source, namespace)
        assert set(namespace) == {"BEHAVIOR_PILOT", "__builtins__"}


def test_chain_cell_clears_stale_canonical_chain_before_loop():
    namespace = {"BEHAVIOR_PILOT": True, "CHAIN": [object()]}
    exec(_cell("# ===== BEHAVIOR CHAIN LOOP ====="), namespace)
    assert namespace["CHAIN"] == []
    assert "NODE" not in namespace


def test_all_local_notebook_code_cells_route_a_pilot_without_canonical_artifacts(pilot_files, monkeypatch):
    """Only long-running pilot training/display are replaced; all cells execute."""
    from environments.trex.scripts import train_behaviors

    monkeypatch.chdir(REPO_ROOT)
    root, model, stats = pilot_files
    recorded_args = []
    displays = []

    def fake_main(args):
        recorded_args.append(args)
        output = Path(args[args.index("--output") + 1])
        output.mkdir(parents=True)
        (output / "run.json").write_text(
            json.dumps(
                {
                    "schema": "mesozoic.behavior-pilot-run/v1",
                    "canonical_certification": False,
                    "status": "complete",
                    "run_seed": 747,
                }
            )
        )

    monkeypatch.setattr(train_behaviors, "main", fake_main)
    monkeypatch.setattr(notebook, "display_notebook_pilot", displays.append)
    namespace: dict[str, Any] = {}
    for source in _code_cells():
        if "REPO_REF =" in source:
            code = _edited_cell(source, IN_COLAB=False)
        elif CONFIG_MARKER in source:
            code = _edited_cell(
                source,
                SPECIES="Tyrannosaurus Rex",
                BEHAVIOR="mixed_terrain",
                PILOT_CHECKPOINT=str(model),
                PILOT_VECNORMALIZE=str(stats),
                PILOT_SEED=747,
                QUICK_TEST=True,
                USE_GOOGLE_DRIVE=False,
                AUTO_DISCONNECT=False,
            )
        else:
            code = compile(source, "sb3_training.ipynb", "exec")
        exec(code, namespace)
        if "Repo root:" in source:
            namespace["repo_root"] = root
    assert len(recorded_args) == 1
    assert recorded_args[0] == namespace["PILOT_PLAN"].argv()
    assert displays == [namespace["PILOT_PLAN"].output_dir]
    assert namespace["PILOT_RESULT"]["status"] == "complete"
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
