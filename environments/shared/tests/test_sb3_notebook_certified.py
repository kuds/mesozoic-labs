"""Execute canonical notebook selection and publication branches without training."""

from __future__ import annotations

import ast
import json
import shutil
import sys
import types
from datetime import datetime
from pathlib import Path

import pytest

from environments.shared import ancestors, config, result_bundle, task_fingerprint
from environments.shared.stage_manifest import load_stage_manifest

NOTEBOOK = Path(__file__).resolve().parents[3] / "notebooks/sb3_training.ipynb"


def _chain_source():
    return next(
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if "# ===== BEHAVIOR CHAIN LOOP =====" in "".join(cell["source"])
    )


@pytest.mark.parametrize("custom_library", [False, True])
def test_canonical_storage_initializes_certified_library_before_training(tmp_path, monkeypatch, custom_library):
    """Execute the storage cell that initializes the chain's shared-library path."""
    from environments.shared import plant_contract

    source = next(
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if cell["cell_type"] == "code" and "# Storage Configuration" in "".join(cell["source"])
    )
    repository = tmp_path / "checkout"
    repository.mkdir()
    library_root = str(tmp_path / "custom" / ".." / "shared-certified") if custom_library else ""
    identity = {"species": "trex", "test_identity": True}
    provenance_calls = []
    monkeypatch.setattr(
        plant_contract, "current_plant_identity", lambda species: types.SimpleNamespace(to_dict=lambda: identity)
    )

    def initialize(directory, **kwargs):
        provenance_calls.append((directory, kwargs))
        return directory / "provenance.json"

    monkeypatch.setattr(result_bundle, "initialize_result_bundle", initialize)
    namespace = {
        "Path": Path,
        "datetime": datetime,
        "repo_root": repository,
        "IN_COLAB": False,
        "USE_GOOGLE_DRIVE": False,
        "COMMAND_TERRAIN_BEHAVIOR": False,
        "CERTIFIED_LIBRARY_ROOT": library_root,
        "SOURCE_SELECTION": "auto",
        "PUBLISH_CERTIFIED": True,
        "SPECIES": "trex",
        "ALGORITHM": "ppo",
        "SEED": 42,
        "N_ENVS": 1,
        "TRUNK_FROM": "",
    }
    exec(compile(source, "sb3_storage", "exec"), namespace)
    expected = Path(library_root).resolve() if custom_library else repository / "certified"
    assert namespace["CERTIFIED_LIBRARY"] == expected
    assert namespace["RUN_DIR"].is_dir()
    assert namespace["RUN_DIR"].is_relative_to(repository / "logs")
    assert namespace["TRUNK_DIR"] is None
    assert provenance_calls[0][1]["plant_identity"] == identity
    original_run = namespace["RUN_DIR"]
    exec(compile(source, "sb3_storage_rerun", "exec"), namespace)
    assert namespace["CERTIFIED_LIBRARY"] == expected
    assert namespace["RUN_DIR"] == original_run


def _session(
    tmp_path,
    monkeypatch,
    *,
    manual=False,
    failed=False,
    quick=False,
    retrain="",
    target="behavior",
    publication_error=None,
):
    """Real notebook control flow; bounded adapters replace training and gates."""
    manifest = load_stage_manifest("trex")
    chain = manifest.chain_for(target)
    run = tmp_path / "new-run"
    run.mkdir()
    source = tmp_path / "original-run"
    source.mkdir()
    (source / "provenance.json").write_text('{"run_id":"original-run"}')
    events = []
    source_ancestors = {}
    for entry in chain:
        stage_dir = source / f"stage{entry.reference}"
        (stage_dir / "models").mkdir(parents=True)
        (stage_dir / "videos").mkdir()
        (stage_dir / "stage_config.json").write_text("{}")
        (stage_dir / "videos" / "replay.mp4").write_bytes(b"complete-source-video")
        (stage_dir / "terrain_map.png").write_bytes(b"complete-source-map")
        model = stage_dir / "models" / "handoff.zip"
        normalizer = stage_dir / "models" / "handoff.pkl"
        model.write_bytes(entry.id.encode())
        normalizer.write_bytes(b"normalization")
        source_ancestors[entry.id] = types.SimpleNamespace(
            stage_id=entry.id,
            stage_dir=stage_dir,
            source_run_dir=source,
            model_zip=model,
            model_stem=str(model.with_suffix("")),
            normalization_path=normalizer,
            model_sha256=result_bundle.sha256_file(model),
            normalization_sha256=result_bundle.sha256_file(normalizer),
            run_id="original-run",
            handoff_name="handoff",
            verdict={},
        )

    def copied(ancestor):
        destination = run / "certified_inputs" / ancestor.stage_id / "version"
        shutil.copytree(source, destination)
        values = dict(vars(ancestor))
        for name in ("source_run_dir", "stage_dir", "model_zip", "normalization_path"):
            values[name] = destination / getattr(ancestor, name).relative_to(source)
        values["model_stem"] = str(values["model_zip"].with_suffix(""))
        return types.SimpleNamespace(**values)

    def find(candidate, **kwargs):
        events.append(("find", kwargs["entry"].id, candidate))
        if manual and candidate == source:
            return source_ancestors[kwargs["entry"].id]
        raise ancestors.AncestorReuseError("no certified stage in this run")

    def copy_manual(ancestor, destination):
        assert destination == run
        events.append(("manual_copy", ancestor.stage_id))
        return copied(ancestor)

    def publish(library, **kwargs):
        events.append(("publish", kwargs["entry"].id, kwargs))
        if publication_error is not None:
            error_type = (
                adapter.CanonicalLibraryError if publication_error == "canonical" else ancestors.AncestorReuseError
            )
            raise error_type("evidence changed before publication")
        return {"status": "failed" if failed else "eligible", "decision_reason": "recorded", "version": "v1"}

    adapter = types.ModuleType("environments.shared.certified_canonical")
    adapter.CanonicalLibraryError = type("CanonicalLibraryError", (RuntimeError,), {})
    adapter.copy_canonical_ancestor = copy_manual
    adapter.publish_canonical_stage = publish
    monkeypatch.setitem(sys.modules, adapter.__name__, adapter)
    monkeypatch.setattr(ancestors, "find_certified_ancestor", find)
    monkeypatch.setattr(
        ancestors, "record_ancestor", lambda run, ancestor: events.append(("record", ancestor.stage_id, ancestor))
    )
    monkeypatch.setattr(config, "hyperparameter_diff", lambda *args: [])
    monkeypatch.setattr(result_bundle, "read_gate_verdict", lambda stage_dir: None)
    monkeypatch.setattr(
        task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {"task_sha256": f"task-{kwargs['stage']}"}
    )

    def train(**kwargs):
        entry = manifest.resolve(kwargs["stage"])
        events.append(("train", entry.id, kwargs))
        directory = run / f"stage{entry.reference}"
        (directory / "models").mkdir(parents=True)
        stem = directory / "models" / "final"
        Path(f"{stem}.zip").write_bytes(b"new-trained-model")
        norm = directory / "models" / "final.pkl"
        norm.write_bytes(b"new-normalizer")
        return object(), str(stem), f"{stem}.zip", directory, str(norm), {}

    def artifacts(**kwargs):
        entry = manifest.resolve(kwargs["stage"])
        events.append(("artifacts", entry.id))
        return {"publication_gate_passed": not failed, "gate_failures": ["failed test gate"] if failed else []}

    namespace = {
        "Path": Path,
        "SPECIES": "trex",
        "SPECIES_CFG": object(),
        "ALGORITHM": "ppo",
        "BEHAVIOR": target,
        "CHAIN": chain,
        "MANIFEST": manifest,
        "TARGET_NODE": manifest.by_id(target),
        "RETRAIN_NODE": manifest.by_id(retrain) if retrain else None,
        "RETRAIN_FROM": retrain,
        "TRUNK_DIR": source if manual else None,
        "CERTIFIED_LIBRARY": tmp_path / "certified",
        "PUBLISH_CERTIFIED": True,
        "CERTIFIED_COMPARISON_EPISODES": 77,
        "RUN_DIR": run,
        "RUN_LABEL": "",
        "QUICK_TEST": quick,
        "SEED": 42,
        "PLANT_IDENTITY": types.SimpleNamespace(to_dict=lambda: {"plant": "current"}),
        "STAGE_CONFIGS": {
            entry.reference: {"name": entry.id, "curriculum_kwargs": {"timesteps": 100}} for entry in chain
        },
        "NODE_HANDOFF": {},
        "NODE_RESULTS": {},
        "completed_stages": [],
        "stage_dirname": lambda species, stage: f"stage{stage}",
        "stage_label": lambda stage: f"stage{stage}",
        "train_stage": train,
        "generate_stage_artifacts": artifacts,
        "display_stage_videos": lambda *args: None,
        "plot_training_curves": lambda *args: None,
        "plot_diagnostics_graphs": lambda *args: None,
        "write_training_summary": lambda *args: events.append(("summary",)),
        "save_run_bundle": lambda *args, **kwargs: events.append(("bundle",)),
        "chain_results": lambda: [],
        "disconnect_runtime": lambda message: events.append(("disconnect", message)),
    }
    return namespace, events, source_ancestors


def test_manual_trunk_takes_priority_and_copies_its_complete_artifacts(tmp_path, monkeypatch):
    namespace, events, _ = _session(tmp_path, monkeypatch, manual=True)
    exec(_chain_source(), namespace)
    assert [event[1] for event in events if event[0] == "manual_copy"] == ["stance", "locomotion"]
    assert Path(namespace["NODE_HANDOFF"]["locomotion"]["model"]).is_relative_to(namespace["RUN_DIR"])


def test_no_trunk_trains_every_node(tmp_path, monkeypatch):
    """Without a TRUNK_DIR (no pinned run, nothing selected under "auto") the loop consults only RUN_DIR."""
    namespace, events, _ = _session(tmp_path, monkeypatch)
    exec(_chain_source(), namespace)
    assert [event[1] for event in events if event[0] == "train"] == ["stance", "locomotion", "behavior"]
    assert {event[2] for event in events if event[0] == "find"} == {namespace["RUN_DIR"]}


def test_failed_target_is_recorded_after_artifacts_and_before_gate_disconnect(tmp_path, monkeypatch):
    # The ancestors come from a pinned trunk (D-A25 moved automatic selection out of the loop),
    # so the target is the node that trains and fails.
    namespace, events, _ = _session(tmp_path, monkeypatch, failed=True, manual=True)
    with pytest.raises(RuntimeError, match="failed its curriculum gate"):
        exec(_chain_source(), namespace)
    tags = [event[0] for event in events]
    assert tags.index("artifacts") < tags.index("bundle") < tags.index("publish") < tags.index("disconnect")
    assert "behavior" not in namespace["NODE_HANDOFF"]
    report = json.loads((namespace["RUN_DIR"] / "certified_publications.json").read_text())
    assert report["behavior"]["status"] == "failed"


def test_canonical_quick_test_keeps_gate_and_skips_promotion_benchmark(tmp_path, monkeypatch):
    namespace, events, _ = _session(tmp_path, monkeypatch, quick=True, target="stance")
    exec(_chain_source(), namespace)
    assert not any(event[0] == "auto" for event in events)
    publication = next(event[2] for event in events if event[0] == "publish")
    assert publication["benchmark"] is False
    assert namespace["NODE_RESULTS"]["stance"]["publication_gate_passed"]


@pytest.mark.parametrize("error", ["canonical", "ancestor"])
def test_publication_evidence_refusal_disconnects_after_saving_artifacts(tmp_path, monkeypatch, error):
    namespace, events, _ = _session(tmp_path, monkeypatch, target="stance", publication_error=error)
    with pytest.raises(RuntimeError, match="evidence changed before publication"):
        exec(_chain_source(), namespace)
    tags = [event[0] for event in events]
    assert tags.index("artifacts") < tags.index("bundle") < tags.index("publish") < tags.index("disconnect")
    assert "Certified library publication refused" in events[-1][1]
    assert "stance" not in namespace["NODE_HANDOFF"]


@pytest.mark.parametrize("mode", [None, "initialize_next_stage", "resume_same_stage"])
def test_training_origin_is_stamped_before_learning_with_the_exact_parent_pair(tmp_path, monkeypatch, mode):
    """Execute the actual notebook stamp block and verify its API inputs/order."""
    from environments.shared import certified_canonical

    source = next(
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if "def train_stage(" in "".join(cell["source"])
    )
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "train_stage"
    )
    create = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "model"
    )
    end = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "loaded_steps"
    )
    stamp_statements = function.body[create + 1 : end]
    learning = next(
        node for node in ast.walk(function) if isinstance(node, ast.Call) and ast.unparse(node.func) == "model.learn"
    )
    assert stamp_statements[-1].end_lineno < learning.lineno
    pair = tmp_path / "selected"
    pair.mkdir()
    model_path, normalizer_path = pair / "model.zip", pair / "vecnormalize.pkl"
    model_path.write_bytes(b"selected-parent-model")
    normalizer_path.write_bytes(b"selected-parent-normalizer")
    cfg = tmp_path / "stage_config.json"
    cfg.write_text('{"run":{"seed":99}}')
    received = []
    origin = {"version": "test", "seed": 42 if mode == "resume_same_stage" else 99}

    def stamp(model, **kwargs):
        received.append(kwargs)
        return origin

    monkeypatch.setattr(certified_canonical, "stamp_canonical_training", stamp)
    namespace = {
        "Path": Path,
        "model": object(),
        "load_path": str(model_path.with_suffix("")) if mode else None,
        "vecnorm_path": str(normalizer_path) if mode else None,
        "task_load_mode": mode or "resume_same_stage",
        "entering_from_parent": mode == "initialize_next_stage",
        "SEED": 99,
        "task_fingerprint": {"task_sha256": "task"},
        "cfg_path": cfg,
    }
    code = compile(
        ast.fix_missing_locations(ast.Module(body=stamp_statements, type_ignores=[])), "notebook-stamp", "exec"
    )
    exec(code, namespace)
    assert received == [
        {
            "seed": 99,
            "task_sha256": "task",
            "load_mode": mode,
            "parent_checkpoint_sha256": result_bundle.sha256_file(model_path)
            if mode == "initialize_next_stage"
            else None,
            "parent_normalization_sha256": result_bundle.sha256_file(normalizer_path)
            if mode == "initialize_next_stage"
            else None,
        }
    ]
    saved = json.loads(cfg.read_text())
    assert saved["run"]["seed"] == 99  # Requested run settings retain their history.
    assert saved["run"]["canonical_training_origin"] == origin


def test_policy_initialization_receives_recorded_seed_before_origin_is_stamped(tmp_path):
    source = next(
        "".join(cell["source"])
        for cell in json.loads(NOTEBOOK.read_text())["cells"]
        if "def train_stage(" in "".join(cell["source"])
    )
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "train_stage"
    )
    preparation = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "_prepare_alg_kwargs"
    )
    construction = next(
        i
        for i, node in enumerate(function.body)
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "model"
    )
    received = []

    def create(sb3, algorithm, kwargs, train_env, load_path, **options):
        received.append(dict(kwargs))
        return object()

    namespace = {
        "config": {},
        "ALGORITHM": "ppo",
        "VERBOSE": 0,
        "stage_dir": tmp_path,
        "SEED": 8675309,
        "load_path": None,
        "sb3": {},
        "train_env": object(),
        "PLANT_IDENTITY": object(),
        "task_fingerprint": {"task_sha256": "task"},
        "task_load_mode": "resume_same_stage",
        "_prepare_alg_kwargs": lambda *args, **kwargs: ({"learning_rate": 0.001}, None, None),
        "_create_or_load_model": create,
    }
    block = ast.Module(body=function.body[preparation : construction + 1], type_ignores=[])
    exec(compile(ast.fix_missing_locations(block), "notebook-construction", "exec"), namespace)
    assert received == [{"learning_rate": 0.001, "seed": 8675309}]
