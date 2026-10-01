"""Pins for ``policy_loading.load_sb3_model``: the one way an SB3 archive is opened.

An SB3 archive's ``learning_rate`` / ``lr_schedule`` / ``clip_range`` members
are cloudpickled; a closure (the repository's ``linear_schedule`` before the
loader PR, SB3's own ``constant_fn`` lambdas before 2.7) is pickled BY VALUE
with its code object, and ``PPO.load`` calls it while rebuilding the policy's
optimizer. Bytecode compiled by one Python minor version executed by another
segfaults the interpreter (3.12 <-> 3.13 reproduced both ways with torch
held constant), which is what killed the Colab kernel twice inside the widen
tool's self-verification on 2026-09-19 (KNOWN_ISSUES, "SB3 archives are
bound to the interpreter that saved them").

The fixtures under ``fixtures/sb3_archives/`` are such closure-bearing
archives saved under Python 3.12 and 3.13 (``make_fixtures.py`` regenerates
them; ``manifest.json`` records the saving interpreter, the members expected
to carry bytecode and the deterministic actions the fresh model produced),
so whichever interpreter runs this suite finds at least one foreign archive.
Pins that need SB3 skip without it; the inspection, rule and AST pins run on
a bare install.

The pins at the end hold the SB3 import helper and the VecNormalize sidecar
resolver that moved here from ``train_base`` (cleanup CU-8b): one definition
each, every caller importing them from this module at call time so one patch
reaches all of them, ``train_base``'s old names bound to the same objects and
read by no other code or test, and a bare ``policy_loading`` that resolves a
sidecar without importing SB3, torch or ``train_base``.
"""

from __future__ import annotations

import ast
import base64
import functools
import json
import logging
import math
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

from environments.shared import policy_loading
from environments.shared.curriculum.schedules import (
    CosineSchedule,
    LinearSchedule,
    _ConstantSchedule,
    schedule_members_from_hyperparameters,
)
from environments.shared.policy_loading import (
    INFERENCE_SCHEDULE_DEFAULTS,
    SCHEDULE_MEMBERS,
    PolicyLoadError,
    SB3ArchiveInspection,
    inspect_sb3_archive,
    load_sb3_model,
    schedule_custom_objects,
)

from .notebook_cells import code_cell_sources, code_cells, strip_magics

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "sb3_archives"
MANIFEST: dict[str, dict] = json.loads((FIXTURES / "manifest.json").read_text(encoding="utf-8"))
RUNNING = tuple(sys.version_info[:2])
#: The interpreters the bare-load segfault was reproduced on, both ways.
VERIFIED_CRASHING_PAIR = {(3, 12), (3, 13)}
FIXTURE_NAMES = sorted(MANIFEST)
FOREIGN_FIXTURES = [name for name in FIXTURE_NAMES if tuple(MANIFEST[name]["python_minor"]) != RUNNING]
NATIVE_FIXTURES = [name for name in FIXTURE_NAMES if tuple(MANIFEST[name]["python_minor"]) == RUNNING]


def _legacy_linear_schedule(initial_lr: float, final_lr: float):
    """The closure ``train_base.linear_schedule`` returned before the loader PR (what the fixtures embed)."""

    def schedule(progress_remaining: float) -> float:
        return final_lr + progress_remaining * (initial_lr - final_lr)

    return schedule


# ── the fixtures and the inspector ───────────────────────────────────────────


def test_the_fixture_set_covers_both_verified_interpreters_and_both_algorithms():
    saved = {tuple(entry["python_minor"]) for entry in MANIFEST.values()}
    assert VERIFIED_CRASHING_PAIR <= saved, "one closure-bearing fixture per interpreter of the reproduced pair"
    for minor in VERIFIED_CRASHING_PAIR:
        algorithms = {entry["algorithm"] for entry in MANIFEST.values() if tuple(entry["python_minor"]) == minor}
        assert algorithms == {"ppo", "sac"}
    for name in FIXTURE_NAMES:
        assert (FIXTURES / name).is_file(), name


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_inspector_reads_the_saving_interpreter_and_the_bytecode_members_without_unpickling(name):
    entry = MANIFEST[name]
    inspection = inspect_sb3_archive(FIXTURES / name)
    assert inspection.saved_python == tuple(entry["python_minor"])
    assert inspection.saved_python_text == f"{entry['python_minor'][0]}.{entry['python_minor'][1]}"
    assert inspection.cross_interpreter is (inspection.saved_python != RUNNING)
    assert sorted(inspection.bytecode_members) == entry["bytecode_members"]
    # Only the schedule members carry code; every other pickled member (spaces,
    # arrays, deques, class references) is code-free.
    assert inspection.bytecode_members <= set(SCHEDULE_MEMBERS[entry["algorithm"]])
    assert inspection.bytecode_members <= inspection.serialized_members
    assert {"observation_space", "action_space", "policy_class"} <= inspection.serialized_members
    # The path may be given without its .zip suffix, as SB3 allows.
    assert inspect_sb3_archive(str(FIXTURES / name)[: -len(".zip")]).bytecode_members == inspection.bytecode_members


def test_the_inspector_refuses_what_it_cannot_read(tmp_path):
    with pytest.raises(PolicyLoadError):
        inspect_sb3_archive(tmp_path / "missing.zip")
    corrupt = tmp_path / "corrupt.zip"
    corrupt.write_bytes(b"not a zip")
    with pytest.raises(PolicyLoadError):
        inspect_sb3_archive(corrupt)
    with zipfile.ZipFile(tmp_path / "no_data.zip", "w") as archive:
        archive.writestr("policy.pth", b"")
    with pytest.raises(PolicyLoadError):
        inspect_sb3_archive(tmp_path / "no_data.zip")


def test_pickle_globals_flags_functions_pickled_by_value_and_nothing_else():
    cloudpickle = pytest.importorskip("cloudpickle")
    by_value = {
        "closure": _legacy_linear_schedule(1.0, 0.1),
        "lambda": (lambda x: x),
        "dict holding a lambda": {"activation_fn": (lambda x: x)},
    }
    by_reference = {
        "module-level function": math.cos,
        "schedule class instance": LinearSchedule(1.0, 0.1),
        "constant schedule": _ConstantSchedule(0.2),
        "plain data": {"a": 1, "b": [1.0, 2.0], "c": None},
    }
    for label, value in by_value.items():
        assert policy_loading._carries_bytecode(policy_loading._pickle_globals(cloudpickle.dumps(value))), label
    for label, value in by_reference.items():
        globals_ = policy_loading._pickle_globals(cloudpickle.dumps(value))
        assert globals_ is not None and not policy_loading._carries_bytecode(globals_), label
    assert policy_loading._pickle_globals(b"\x80\x05not a pickle") is None
    assert policy_loading._carries_bytecode(None) is True


def test_pickle_globals_follows_memoized_module_names():
    """A second by-value function re-uses the memoized ``cloudpickle.cloudpickle`` string through a memo get.

    Protocol 4+ pushes a string once and fetches it back with ``BINGET`` afterwards, so a walker that only pairs
    fresh string pushes would miss every ``STACK_GLOBAL`` after the first. The pair must be found on both
    references, and a stream whose FIRST cloudpickle reference is an unlisted reducer is still flagged by the
    module backstop.
    """
    cloudpickle = pytest.importorskip("cloudpickle")
    pickletools = pytest.importorskip("pickletools")
    payload = cloudpickle.dumps(((lambda x: x), (lambda y: y + 1), {"nested": (lambda z: z)}))
    ops = [opcode.name for opcode, _arg, _pos in pickletools.genops(payload)]
    assert ops.count("STACK_GLOBAL") >= 3 and any(name in ("BINGET", "LONG_BINGET") for name in ops), (
        "the payload exercises memo gets between STACK_GLOBAL pairs"
    )
    globals_ = policy_loading._pickle_globals(payload)
    assert globals_ is not None
    assert ("cloudpickle.cloudpickle", "_make_function") in globals_
    assert policy_loading._carries_bytecode(globals_)
    assert policy_loading._carries_bytecode({("cloudpickle.cloudpickle", "_some_future_reducer")})
    assert not policy_loading._carries_bytecode({("environments.shared.curriculum.schedules", "LinearSchedule")})


# ── the schedule classes ─────────────────────────────────────────────────────


def test_the_schedule_classes_reproduce_the_legacy_closures_and_pickle_by_reference():
    cloudpickle = pytest.importorskip("cloudpickle")
    for progress in (1.0, 0.75, 0.5, 0.25, 0.0):
        assert LinearSchedule(3e-4, 1e-5)(progress) == pytest.approx(_legacy_linear_schedule(3e-4, 1e-5)(progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
        assert CosineSchedule(3e-4, 1e-5)(progress) == pytest.approx(1e-5 + cosine_decay * (3e-4 - 1e-5))
    for schedule in (LinearSchedule(3e-4, 1e-5), CosineSchedule(3e-4, 1e-5), _ConstantSchedule(0.2)):
        payload = cloudpickle.dumps(schedule)
        assert not policy_loading._carries_bytecode(policy_loading._pickle_globals(payload))
        assert cloudpickle.loads(payload)(0.5) == schedule(0.5)
        assert repr(schedule).startswith(type(schedule).__name__)


def test_train_base_factories_return_the_classes():
    from environments.shared.train_base import cosine_schedule, linear_schedule

    assert isinstance(linear_schedule(3e-4, 1e-5), LinearSchedule)
    assert isinstance(cosine_schedule(3e-4, 1e-5), CosineSchedule)


def test_schedule_members_from_hyperparameters_mirrors_prepare_alg_kwargs(tmp_path):
    from environments.shared.config import load_stage_config
    from environments.shared.train_base import _prepare_alg_kwargs

    config = load_stage_config("trex", "stance")
    block = config["ppo_kwargs"]
    assert block.get("learning_rate_end") is not None, "the trex stance trains under an LR decay (the Drive parents)"
    prepared, _, _ = _prepare_alg_kwargs(dict(config), "ppo", 0, tmp_path, False)
    members = schedule_members_from_hyperparameters("ppo", block)
    assert isinstance(members["learning_rate"], LinearSchedule)
    assert (members["learning_rate"].initial, members["learning_rate"].final) == (
        prepared["learning_rate"].initial,
        prepared["learning_rate"].final,
    )
    assert members["clip_range"] == prepared["clip_range"]
    assert "clip_range_vf" not in members or members["clip_range_vf"] == prepared.get("clip_range_vf")
    # The mapping itself: an end value makes a decay, cosine by name, SAC has no clip range.
    assert schedule_members_from_hyperparameters("ppo", {"learning_rate": 1e-3}) == {"learning_rate": 1e-3}
    cosine = schedule_members_from_hyperparameters(
        "ppo", {"learning_rate": 1e-3, "learning_rate_end": 1e-4, "lr_schedule": "cosine", "clip_range": 0.2}
    )
    assert isinstance(cosine["learning_rate"], CosineSchedule) and cosine["clip_range"] == 0.2
    clipped = schedule_members_from_hyperparameters(
        "PPO", {"clip_range": 0.2, "clip_range_end": 0.1, "clip_range_vf": 10.0}
    )
    assert isinstance(clipped["clip_range"], LinearSchedule) and clipped["clip_range_vf"] == 10.0
    assert schedule_members_from_hyperparameters("sac", {"learning_rate": 1e-3, "clip_range": 0.2}) == {
        "learning_rate": 1e-3
    }
    # The trainer applies no decay mapping to a [sac] table; neither does the helper.
    assert schedule_members_from_hyperparameters("sac", {"learning_rate": 1e-3, "learning_rate_end": 1e-4}) == {
        "learning_rate": 1e-3
    }
    assert schedule_members_from_hyperparameters("sac", None) == {}


# ── the replacement rule ─────────────────────────────────────────────────────


def _inspection(bytecode: set[str], *, saved=(3, 12), serialized: "set[str] | None" = None) -> SB3ArchiveInspection:
    return SB3ArchiveInspection(
        path=Path("archive.zip"),
        saved_python=saved,
        serialized_members=frozenset(serialized if serialized is not None else bytecode | {"observation_space"}),
        bytecode_members=frozenset(bytecode),
    )


def test_schedule_custom_objects_replaces_exactly_the_bytecode_members():
    # An unreadable archive: every schedule member of the algorithm, with the inference defaults.
    custom = schedule_custom_objects(None, "ppo")
    assert set(custom) == set(SCHEDULE_MEMBERS["ppo"])
    assert custom["learning_rate"] == INFERENCE_SCHEDULE_DEFAULTS["learning_rate"] == 0.0
    assert isinstance(custom["lr_schedule"], _ConstantSchedule) and custom["lr_schedule"](1.0) == 0.0
    assert custom["clip_range"] == 0.2 and custom["clip_range_vf"] is None
    assert set(schedule_custom_objects(None, "sac")) == {"learning_rate", "lr_schedule"}
    # An unknown algorithm (a test double) gets the union.
    assert set(schedule_custom_objects(None, None)) == set(SCHEDULE_MEMBERS["ppo"])
    # A readable archive without bytecode is left entirely alone: a float learning
    # rate or SB3's own by-reference schedule classes load exactly as before.
    assert schedule_custom_objects(_inspection(set()), "ppo") == {}
    # Only the members that embed bytecode are supplied.
    custom = schedule_custom_objects(_inspection({"learning_rate", "lr_schedule"}), "ppo")
    assert set(custom) == {"learning_rate", "lr_schedule"}


def test_schedule_custom_objects_prefers_the_caller_then_the_training_keywords():
    decay = LinearSchedule(3e-4, 1e-5)
    clip = LinearSchedule(0.2, 0.1)
    inspection = _inspection({"learning_rate", "lr_schedule", "clip_range"})
    custom = schedule_custom_objects(inspection, "ppo", training_kwargs={"learning_rate": decay, "clip_range": clip})
    assert custom["learning_rate"] is decay
    assert custom["lr_schedule"] is decay, "lr_schedule follows the learning rate it is rebuilt from"
    assert custom["clip_range"] is clip
    # A float learning rate gives a constant lr_schedule of the same value.
    custom = schedule_custom_objects(inspection, "ppo", training_kwargs={"learning_rate": 3e-5})
    assert custom["learning_rate"] == 3e-5 and custom["lr_schedule"](0.3) == 3e-5
    # The caller's custom_objects win over everything and are passed through untouched.
    sentinel = _ConstantSchedule(0.5)
    custom = schedule_custom_objects(
        inspection,
        "ppo",
        custom_objects={"learning_rate": 1.0, "lr_schedule": sentinel, "extra": 7},
        training_kwargs={"learning_rate": decay},
    )
    assert custom["learning_rate"] == 1.0 and custom["lr_schedule"] is sentinel and custom["extra"] == 7
    assert custom["clip_range"] == 0.2


def test_load_sb3_model_reports_a_failed_load_from_an_unreadable_path_through_sb3(tmp_path):
    pytest.importorskip("stable_baselines3")
    with pytest.raises((FileNotFoundError, ValueError, OSError)):
        load_sb3_model(tmp_path / "missing.zip", algorithm="ppo", device="cpu")
    with pytest.raises(PolicyLoadError):
        load_sb3_model(tmp_path / "missing.zip", algorithm="a2c")


# ── loading the fixtures ─────────────────────────────────────────────────────


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_the_loader_reproduces_the_recorded_actions_whatever_interpreter_saved_the_archive(name):
    pytest.importorskip("stable_baselines3")
    np = pytest.importorskip("numpy")
    entry = MANIFEST[name]
    observations = np.asarray(entry["observations"], dtype=np.float32)
    expected = np.asarray(entry["actions"], dtype=np.float64)
    for algorithm in (None, entry["algorithm"]):
        model = load_sb3_model(FIXTURES / name, algorithm=algorithm, device="cpu")
        actions, _ = model.predict(observations, deterministic=True)
        assert np.allclose(np.asarray(actions, dtype=np.float64).ravel(), expected, atol=1e-6), name
        # The bytecode members were supplied, never unpickled: inference placeholders.
        assert model.learning_rate == INFERENCE_SCHEDULE_DEFAULTS["learning_rate"]
        assert model.lr_schedule(1.0) == 0.0
        if entry["algorithm"] == "ppo":
            assert model.clip_range(1.0) == pytest.approx(0.2)
        # ... and the optimizer keeps the learning rate the archive saved (set_parameters restores it).
        optimizer = model.policy.optimizer if entry["algorithm"] == "ppo" else model.actor.optimizer
        assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5)


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_training_keywords_are_applied_over_the_archive_and_become_the_model_schedules(name):
    pytest.importorskip("stable_baselines3")
    entry = MANIFEST[name]
    decay = LinearSchedule(3e-4, 1e-5)
    keywords = {"learning_rate": decay}
    if entry["algorithm"] == "ppo":
        keywords["clip_range"] = LinearSchedule(0.2, 0.1)
    model = load_sb3_model(FIXTURES / name, algorithm=entry["algorithm"], device="cpu", **keywords)
    assert model.learning_rate is decay
    assert model.lr_schedule(1.0) == pytest.approx(3e-4) and model.lr_schedule(0.0) == pytest.approx(1e-5)
    if entry["algorithm"] == "ppo":
        assert model.clip_range(1.0) == pytest.approx(0.2) and model.clip_range(0.0) == pytest.approx(0.1)


def test_a_model_re_saved_through_the_loader_carries_no_bytecode(tmp_path):
    pytest.importorskip("stable_baselines3")
    for name in FIXTURE_NAMES:
        entry = MANIFEST[name]
        keywords = {"learning_rate": LinearSchedule(3e-4, 1e-5)}
        if entry["algorithm"] == "ppo":
            keywords["clip_range"] = LinearSchedule(0.2, 0.1)
        model = load_sb3_model(FIXTURES / name, algorithm=entry["algorithm"], device="cpu", **keywords)
        out = tmp_path / f"resaved_{name}"
        model.save(str(out))
        resaved = inspect_sb3_archive(out)
        assert resaved.bytecode_members == frozenset(), name
        assert resaved.saved_python == RUNNING


@pytest.mark.skipif(not FOREIGN_FIXTURES, reason="every fixture was saved by this interpreter")
@pytest.mark.skipif(
    RUNNING not in VERIFIED_CRASHING_PAIR,
    reason="the bare-load crash is undefined behaviour verified on 3.12 <-> 3.13; other interpreters may fail differently",
)
def test_a_bare_load_of_a_foreign_archive_dies_where_the_loader_survives():
    pytest.importorskip("stable_baselines3")
    name = next(n for n in FOREIGN_FIXTURES if MANIFEST[n]["algorithm"] == "ppo")
    archive = FIXTURES / name
    bare = (
        "import sys; sys.path.insert(0, sys.argv[1]); from stable_baselines3 import PPO; "
        "PPO.load(sys.argv[2], device='cpu'); print('loaded')"
    )
    through_loader = (
        "import sys; sys.path.insert(0, sys.argv[1]); from environments.shared.policy_loading import load_sb3_model; "
        "load_sb3_model(sys.argv[2], device='cpu'); print('loaded')"
    )
    env = {"MUJOCO_GL": "", "PYTHONWARNINGS": "ignore"}
    import os

    env = {**os.environ, **env}
    dead = subprocess.run(
        [sys.executable, "-c", bare, str(REPO_ROOT), str(archive)], capture_output=True, text=True, timeout=600, env=env
    )
    alive = subprocess.run(
        [sys.executable, "-c", through_loader, str(REPO_ROOT), str(archive)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
    )
    assert alive.returncode == 0 and "loaded" in alive.stdout, alive.stderr[-2000:]
    assert dead.returncode != 0, (
        f"the bare PPO.load of {name} survived under Python {RUNNING}:\n{dead.stdout}\n{dead.stderr[-2000:]}"
    )


def _with_extra_member(source: Path, target: Path, member: str, payload: bytes) -> None:
    with zipfile.ZipFile(source) as archive, zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for item in archive.infolist():
            content = archive.read(item.filename)
            if item.filename == "data":
                data = json.loads(content)
                data[member] = {":type:": "<class 'function'>", ":serialized:": base64.b64encode(payload).decode()}
                content = json.dumps(data, indent=4).encode()
            out.writestr(item, content)


def test_bytecode_outside_the_schedule_members_is_refused_across_interpreters_only(tmp_path):
    pytest.importorskip("stable_baselines3")
    cloudpickle = pytest.importorskip("cloudpickle")
    payload = cloudpickle.dumps(lambda progress: 0.5 * progress)
    if FOREIGN_FIXTURES:
        name = FOREIGN_FIXTURES[0]
        foreign = tmp_path / f"foreign_{name}"
        _with_extra_member(FIXTURES / name, foreign, "mesozoic_test_callable", payload)
        assert "mesozoic_test_callable" in inspect_sb3_archive(foreign).bytecode_members
        with pytest.raises(PolicyLoadError, match="mesozoic_test_callable"):
            load_sb3_model(foreign, algorithm=MANIFEST[name]["algorithm"], device="cpu")
        # Naming it in custom_objects is the documented way through.
        model = load_sb3_model(
            foreign,
            algorithm=MANIFEST[name]["algorithm"],
            device="cpu",
            custom_objects={"mesozoic_test_callable": _ConstantSchedule(0.0)},
        )
        assert isinstance(model.mesozoic_test_callable, _ConstantSchedule)
    if NATIVE_FIXTURES:
        name = NATIVE_FIXTURES[0]
        native = tmp_path / f"native_{name}"
        _with_extra_member(FIXTURES / name, native, "mesozoic_test_callable", payload)
        # Same interpreter: the bytecode is this Python's own, so it loads.
        model = load_sb3_model(native, algorithm=MANIFEST[name]["algorithm"], device="cpu")
        assert model.mesozoic_test_callable(2.0) == 1.0


def test_the_widen_tool_restates_a_parents_schedules_from_its_recorded_hyperparameters(tmp_path):
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file

    from environments.shared.scripts import widen_checkpoint as widen

    name = next(n for n in FIXTURE_NAMES if MANIFEST[n]["algorithm"] == "ppo")
    stage_dir = tmp_path / "01_stance"
    stage_dir.mkdir()
    (stage_dir / "stage_config.json").write_text(
        json.dumps(
            {
                "hyperparameters": {"learning_rate": 3e-5, "learning_rate_end": 1e-5, "clip_range": 0.2},
                "run": {"seed": 44, "n_envs": 4, "timesteps": 11_001_856},
            }
        )
    )
    parent = widen._ParentSource(
        model_zip=FIXTURES / name,
        vecnorm_pkl=tmp_path / "unused.pkl",
        handoff_name="robust_best_model",
        declared_algorithm="ppo",
        seed=44,
        n_envs=4,
        timesteps=11_001_856,
        duration_seconds=None,
        run_id="20260815_205206",
        stage_dir=stage_dir,
    )
    current_block = {"learning_rate": 9e-5, "learning_rate_end": 2e-5, "clip_range": 0.15}
    objects, source = widen._parent_schedule_objects(parent, "ppo", current_block)
    assert source == "parent_stage_config"
    assert set(objects) == {"learning_rate", "lr_schedule", "clip_range"}
    assert isinstance(objects["learning_rate"], LinearSchedule)
    assert (objects["learning_rate"].initial, objects["learning_rate"].final) == (3e-5, 1e-5)
    assert objects["lr_schedule"] is objects["learning_rate"]
    assert objects["clip_range"] == 0.2
    # Read through SB3's serializer with those objects and written back: the copy is bytecode-free
    # and loads bare on any interpreter (only its spaces, arrays and class references are pickled).
    data, params, pytorch_variables = load_from_zip_file(str(FIXTURES / name), device="cpu", custom_objects=objects)
    out = tmp_path / "widened.zip"
    save_to_zip_file(str(out), data=data, params=params, pytorch_variables=pytorch_variables)
    assert inspect_sb3_archive(out).bytecode_members == frozenset()
    # The explicit --model/--vecnorm form (or a parent recorded before the hyperparameters block existed) falls
    # back to the CURRENT stage config's algorithm block, never to an inference placeholder.
    bare = widen._ParentSource(**{**parent.__dict__, "stage_dir": None})
    objects, source = widen._parent_schedule_objects(bare, "ppo", current_block)
    assert source == "current_stage_config"
    assert isinstance(objects["learning_rate"], LinearSchedule)
    assert (objects["learning_rate"].initial, objects["learning_rate"].final) == (9e-5, 2e-5)
    assert objects["clip_range"] == 0.15
    (stage_dir / "stage_config.json").write_text(json.dumps({"run": {"seed": 44, "n_envs": 4, "timesteps": 1}}))
    objects, source = widen._parent_schedule_objects(parent, "ppo", current_block)
    assert source == "current_stage_config" and objects["clip_range"] == 0.15
    # Nothing to re-state when the archive stores its schedules by reference.
    resaved = tmp_path / "by_reference.zip"
    model = load_sb3_model(FIXTURES / name, algorithm="ppo", device="cpu", learning_rate=LinearSchedule(1e-3, 1e-4))
    model.save(str(resaved))
    clean = widen._ParentSource(**{**parent.__dict__, "model_zip": resaved})
    assert widen._parent_schedule_objects(clean, "ppo", current_block) == ({}, None)


# ── the pin: no bare algorithm-class load anywhere but the loader ───────────

#: Receivers whose ``.load(...)`` is an SB3 algorithm load: the classes themselves (also as attributes, as in
#: ``sb3.PPO.load``), ``cls`` / ``klass`` inside a classmethod, and every ``alg_cls`` / ``AlgoClass`` /
#: ``_alg_cls_rank`` / ``ppo_cls`` / ``model_class`` style alias, plus any name bound by ``from stable_baselines3
#: import PPO as <alias>`` (``sb3["PPO"].load`` is a Subscript, matched below; ``sb3["VecNormalize"].load`` is a
#: plain pickle of statistics and stays where it is).
_ALGORITHM_RECEIVER_RE = re.compile(
    r"^(ppo|sac|cls|klass|algorithm|algo)$|(alg|algo|algorithm|model|policy)_?(cls|class)|algoclass|(ppo|sac)_(cls|class)",
    re.IGNORECASE,
)


def _algorithm_aliases(tree: ast.AST) -> set[str]:
    """Names bound to PPO / SAC by ``from stable_baselines3 import PPO as <alias>`` (and plain imports)."""
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("stable_baselines3"):
            for alias in node.names:
                if alias.name in ("PPO", "SAC"):
                    aliases.add(alias.asname or alias.name)
    return aliases


def _bare_algorithm_loads(tree: ast.AST) -> list[str]:
    aliases = _algorithm_aliases(tree)
    hits: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "load"):
            continue
        receiver = node.func.value
        subscript_algorithm = (
            isinstance(receiver, ast.Subscript)
            and isinstance(receiver.slice, ast.Constant)
            and receiver.slice.value in ("PPO", "SAC")
        )
        bare = (
            (
                isinstance(receiver, ast.Name)
                and (receiver.id in aliases or _ALGORITHM_RECEIVER_RE.search(receiver.id) is not None)
            )
            or (
                isinstance(receiver, ast.Attribute)
                and (receiver.attr in ("PPO", "SAC") or _ALGORITHM_RECEIVER_RE.search(receiver.attr) is not None)
            )
            or subscript_algorithm
            or isinstance(receiver, ast.IfExp)
            or (isinstance(receiver, ast.Call) and getattr(receiver.func, "id", "") == "_checkpoint_algorithm")
        )
        if bare:
            hits.append(f"{node.lineno}: {ast.unparse(node)[:80]}")
    return hits


def test_the_bare_load_detector_catches_the_alias_and_attribute_forms():
    caught = [
        "PPO.load(p)",
        "SAC.load(p, device='cpu')",
        "alg_cls.load(p, env=e)",
        "AlgoClass.load(p)",
        "_alg_cls_rank.load(p)",
        "ppo_cls.load(p)",
        "model_class.load(p)",
        "self.algorithm_cls.load(p)",
        "sb3.PPO.load(p)",
        "stable_baselines3.SAC.load(p)",
        "sb3['PPO'].load(p)",
        "(PPO if a else SAC).load(p)",
        "_checkpoint_algorithm(p).load(p)",
        "from stable_baselines3 import PPO as _Alias\n_Alias.load(p)",
        "klass.load(p)",
    ]
    ignored = [
        "VecNormalize.load(p, venv)",
        "sb3['VecNormalize'].load(p, venv)",
        "json.load(f)",
        "pickle.load(f)",
        "torch.load(f)",
        "np.load(f)",
        "load_sb3_model(p, algorithm=alg_cls)",
        "tomllib.load(f)",
    ]
    for snippet in caught:
        assert _bare_algorithm_loads(ast.parse(snippet)), snippet
    for snippet in ignored:
        assert not _bare_algorithm_loads(ast.parse(snippet)), snippet


def _python_sources() -> list[tuple[str, str]]:
    sources = []
    for path in sorted((REPO_ROOT / "environments").rglob("*.py")):
        if "tests" in path.parts or path.name == "policy_loading.py":
            continue
        sources.append((str(path.relative_to(REPO_ROOT)), path.read_text(encoding="utf-8")))
    return sources


def test_every_sb3_archive_load_goes_through_the_loader():
    """``PPO.load`` / ``SAC.load`` / ``alg_cls.load`` appear nowhere but in ``load_sb3_model`` itself."""
    offenders: dict[str, list[str]] = {}
    for name, source in _python_sources():
        hits = _bare_algorithm_loads(ast.parse(source))
        if hits:
            offenders[name] = hits
    for path in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")):
        for index, source in code_cells(path):
            hits = _bare_algorithm_loads(ast.parse(strip_magics(source)))
            if hits:
                offenders[f"{path.relative_to(REPO_ROOT)}[cell {index}]"] = hits
    assert not offenders, f"bare SB3 archive loads outside policy_loading.load_sb3_model: {offenders}"
    # The loader itself is where the one real call lives.
    loader_source = (REPO_ROOT / "environments/shared/policy_loading.py").read_text(encoding="utf-8")
    assert len(_bare_algorithm_loads(ast.parse(loader_source))) == 1


def test_the_known_load_sites_call_the_loader():
    """The sites the incident review listed name ``load_sb3_model``."""
    expected = {
        "environments/shared/train_base.py": 3,
        "environments/shared/evaluation.py": 1,
        "environments/shared/reporting/stage_artifacts.py": 5,
        "environments/shared/harnesses/freeze_recovery_gate.py": 1,
        "environments/shared/scripts/widen_checkpoint.py": 1,
        "environments/shared/behavior_checkpoint.py": 1,
    }
    for name, count in expected.items():
        source = (REPO_ROOT / name).read_text(encoding="utf-8")
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_sb3_model"
        ]
        assert len(calls) == count, (name, len(calls))
    cells = code_cell_sources(REPO_ROOT / "notebooks/sb3_training.ipynb")
    assert sum(cell.count("load_sb3_model(") for cell in cells) == 1, "the preflight (evaluation loads in the library)"


# ── CU-8b: the SB3 import helper and the sidecar resolver live here ──────────
#
# ``_ensure_sb3``, ``_resolve_vecnorm_sidecar`` and ``_PERIODIC_CHECKPOINT_RE``
# moved here from ``train_base``, which keeps its old names bound to the same
# objects for importers that predate the move. Every caller imports them from
# this module at call time, so one patch here reaches every caller. A patch of
# the ``train_base`` name reaches none, and with SB3 installed a test written
# that way can pass without exercising what it claims (54 did when the library
# moved and the tests still patched the train_base module).

MOVED_NAMES = ("_ensure_sb3", "_resolve_vecnorm_sidecar", "_PERIODIC_CHECKPOINT_RE")
POLICY_LOADING = "environments/shared/policy_loading.py"
PERIODIC_PATTERN = r"(.+)_(\d+)_steps$"
SB3_MISSING_MESSAGE = "stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]"
SB3_NAMES = {
    "PPO",
    "SAC",
    "CallbackList",
    "CheckpointCallback",
    "EvalCallback",
    "Monitor",
    "set_random_seed",
    "DummyVecEnv",
    "SubprocVecEnv",
    "VecNormalize",
}
#: Every bare-name read of a moved name outside this module. Each sits in a
#: function that imports the name from ``policy_loading`` itself.
CALL_TIME_READS = {
    "environments/shared/train_base.py": {
        "_ensure_sb3": 5,
        "_resolve_vecnorm_sidecar": 1,
        "_PERIODIC_CHECKPOINT_RE": 1,
    },
    "environments/shared/evaluation.py": {"_ensure_sb3": 2, "_resolve_vecnorm_sidecar": 1},
    "environments/shared/reporting/stage_artifacts.py": {"_ensure_sb3": 2},
    "environments/shared/cli.py": {"_PERIODIC_CHECKPOINT_RE": 1},
}
#: The only bindings of a moved name made outside a function body (at import
#: or class-definition time) outside this module: ``train_base``'s
#: compatibility aliases, which no library code reads.
IMPORT_TIME_BINDINGS = {
    "environments/shared/train_base.py": [
        "_PERIODIC_CHECKPOINT_RE = _policy_loading._PERIODIC_CHECKPOINT_RE",
        "_ensure_sb3 = _policy_loading._ensure_sb3",
        "_resolve_vecnorm_sidecar = _policy_loading._resolve_vecnorm_sidecar",
    ],
}
#: The one test allowed to read the old ``train_base`` names.
ALIAS_TEST = "test_train_base_keeps_its_old_names_bound_to_the_moved_objects"
_TRAIN_BASE_PATH = re.compile(r"(?:\w+\.)*train_base\.(?:" + "|".join(MOVED_NAMES) + r")")


@functools.lru_cache(maxsize=None)
def _parsed_sources(*, tests: bool) -> list[tuple[str, ast.Module]]:
    """Every Python module of the repository, either the test ones or the rest (with every notebook code cell).

    Cached: the pins below only read the trees.
    """
    paths = [path for root in ("environments", "configs", "docs") for path in sorted((REPO_ROOT / root).rglob("*.py"))]
    paths += sorted(REPO_ROOT.glob("*.py"))
    sources = [
        (path.relative_to(REPO_ROOT).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
        for path in paths
        if ("tests" in path.relative_to(REPO_ROOT).parts or path.name == "conftest.py") == tests
    ]
    if not tests:
        for path in sorted((REPO_ROOT / "notebooks").glob("*.ipynb")):
            for index, source in code_cells(path):
                sources.append(
                    (f"{path.relative_to(REPO_ROOT).as_posix()}[cell {index}]", ast.parse(strip_magics(source)))
                )
    return sources


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    return ""


def _train_base_reads(tree: ast.AST) -> list[str]:
    """Reads of a moved name THROUGH ``train_base``: imports, attributes, patch targets and patch strings."""
    modules = {"train_base"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules |= {alias.asname for alias in node.names if alias.asname and alias.name.endswith("train_base")}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "train_base":
            hits += [f"from {node.module} import {alias.name}" for alias in node.names if alias.name in MOVED_NAMES]
        elif isinstance(node, ast.Attribute) and node.attr in MOVED_NAMES:
            if _dotted(node.value).split(".")[-1] in modules:
                hits.append(_dotted(node))
        elif isinstance(node, ast.Call) and len(node.args) >= 2:
            target, name = node.args[0], node.args[1]
            if isinstance(name, ast.Constant) and name.value in MOVED_NAMES:
                if _dotted(target).split(".")[-1] in modules:
                    hits.append(f"{_dotted(node.func)}({_dotted(target)}, {name.value!r})")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and _TRAIN_BASE_PATH.fullmatch(node.value):
            hits.append(node.value)
    return hits


def _bare_reads(tree: ast.AST) -> list[tuple[str, list[ast.AST]]]:
    """Each bare-name read of a moved name, with the functions enclosing it (innermost last)."""
    reads: list[tuple[str, list[ast.AST]]] = []

    def visit(node: ast.AST, stack: list[ast.AST]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load) and child.id in MOVED_NAMES:
                reads.append((child.id, stack))
            visit(child, [*stack, child] if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else stack)

    visit(tree, [])
    return reads


def _imported_from_policy_loading(function: ast.AST) -> set[str]:
    return {
        alias.name
        for node in ast.walk(function)
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "policy_loading"
        for alias in node.names
        if alias.asname is None
    }


def _import_time_bindings(tree: ast.AST) -> list[str]:
    """Each moved name taken from ``policy_loading`` outside a function body, as its statement.

    An import of a moved name (aliased or not) or an attribute read of the
    module (``policy_loading`` itself or a name an import bound to it) made at
    import or class-definition time keeps the object it saw, so a later patch
    here never reaches it. Everything of a ``def`` or ``lambda`` but its body
    counts as outside: decorators and default arguments are evaluated when the
    definition runs, not when the function is called.
    """
    modules = {"policy_loading"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules |= {
                alias.asname for alias in node.names if alias.asname and alias.name.split(".")[-1] == "policy_loading"
            }
    hits: list[str] = []

    def visit(node: ast.AST, statement: ast.stmt | None) -> None:
        if isinstance(node, ast.stmt):
            statement = node
        if isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[-1] == "policy_loading":
            if any(alias.name in MOVED_NAMES for alias in node.names):
                hits.append(ast.unparse(node))
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and node.attr in MOVED_NAMES
            and _dotted(node.value).split(".")[-1] in modules
        ):
            if statement is None or hasattr(statement, "body"):
                hits.append(_dotted(node))
            else:
                hits.append(ast.unparse(statement))
        call_time: set[int] = set()
        if isinstance(node, ast.Lambda):
            call_time = {id(node.body)}
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            call_time = {id(child) for child in node.body}
        for child in ast.iter_child_nodes(node):
            if id(child) not in call_time:
                visit(child, statement)

    visit(tree, None)
    return hits


def test_each_moved_name_is_defined_once_here():
    """One definition each, in this module; ``train_base`` only binds its old names to them."""
    definitions: dict[str, list[str]] = {}
    literals = []
    for name, tree in _parsed_sources(tests=False):
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.lstrip("_") in {
                "ensure_sb3",
                "resolve_vecnorm_sidecar",
            }:
                definitions.setdefault(node.name, []).append(name)
            elif isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(node.value, ast.Call):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and target.id.lstrip("_") == "PERIODIC_CHECKPOINT_RE":
                        definitions.setdefault(target.id, []).append(name)
            elif isinstance(node, ast.Constant) and node.value in (PERIODIC_PATTERN, SB3_MISSING_MESSAGE):
                literals.append((name, node.value))
    assert definitions == {moved: [POLICY_LOADING] for moved in MOVED_NAMES}
    assert sorted(literals) == sorted([(POLICY_LOADING, PERIODIC_PATTERN), (POLICY_LOADING, SB3_MISSING_MESSAGE)])


def test_no_library_module_script_or_notebook_reads_the_old_train_base_names():
    offenders = {name: hits for name, tree in _parsed_sources(tests=False) if (hits := _train_base_reads(tree))}
    assert not offenders


def test_every_caller_imports_the_moved_names_from_here_at_call_time():
    """One patch point: every read outside this module looks the name up here when its function runs.

    A binding made outside a function body -- an import of a moved name, under
    its own name or another, or an attribute read of this module -- keeps the
    object it saw and would ignore a later patch here. ``train_base``'s three
    compatibility aliases are the only such bindings, and no library code reads
    them.
    """
    reads: dict[str, dict[str, int]] = {}
    unpatchable = []
    bindings: dict[str, list[str]] = {}
    for name, tree in _parsed_sources(tests=False):
        if name == POLICY_LOADING:
            continue
        for moved, stack in _bare_reads(tree):
            reads.setdefault(name, {}).setdefault(moved, 0)
            reads[name][moved] += 1
            if not any(moved in _imported_from_policy_loading(function) for function in stack):
                unpatchable.append((name, moved, getattr(stack[-1], "name", "<module>") if stack else "<module>"))
        if hits := _import_time_bindings(tree):
            bindings[name] = sorted(hits)
    assert not unpatchable
    assert reads == CALL_TIME_READS
    assert bindings == IMPORT_TIME_BINDINGS


def test_the_import_time_binding_detector_catches_each_form():
    caught = [
        "from .policy_loading import _ensure_sb3 as _x",
        "from environments.shared.policy_loading import _resolve_vecnorm_sidecar",
        "from . import policy_loading as _pl\n_y = _pl._ensure_sb3",
        "class C:\n    f = policy_loading._ensure_sb3",
        "import environments.shared.policy_loading as pl\nPATTERN = pl._PERIODIC_CHECKPOINT_RE",
        "environments.shared.policy_loading._ensure_sb3()",
        "def f(helper=policy_loading._ensure_sb3):\n    return helper()",
        "g = lambda helper=policy_loading._ensure_sb3: helper()",
    ]
    ignored = [
        "def f():\n    from .policy_loading import _ensure_sb3 as e\n    return e()",
        "def f():\n    return policy_loading._ensure_sb3()",
        "class C:\n    def f(self):\n        return policy_loading._resolve_vecnorm_sidecar(self.path)",
        "g = lambda: policy_loading._ensure_sb3()",
        "from .policy_loading import load_sb3_model",
        "loader = policy_loading.load_sb3_model",
    ]
    for snippet in caught:
        assert _import_time_bindings(ast.parse(snippet)), snippet
    for snippet in ignored:
        assert not _import_time_bindings(ast.parse(snippet)), snippet


def test_train_base_keeps_its_old_names_bound_to_the_moved_objects():
    """Notebooks saved before PR-14c import ``_ensure_sb3`` from ``train_base``."""
    from environments.shared import train_base

    assert train_base._ensure_sb3 is policy_loading._ensure_sb3
    assert train_base._resolve_vecnorm_sidecar is policy_loading._resolve_vecnorm_sidecar
    assert train_base._PERIODIC_CHECKPOINT_RE is policy_loading._PERIODIC_CHECKPOINT_RE


def test_no_test_patches_or_reads_the_old_train_base_names():
    """A patch of the ``train_base`` name reaches no caller; only the alias test above may read it."""
    offenders = {}
    for name, tree in _parsed_sources(tests=True):
        if name == "environments/shared/tests/test_policy_loading.py":
            tree = ast.Module(
                body=[node for node in tree.body if getattr(node, "name", None) != ALIAS_TEST], type_ignores=[]
            )
        if hits := _train_base_reads(tree):
            offenders[name] = hits
    assert not offenders


def test_the_train_base_read_detector_catches_each_form():
    caught = [
        "from environments.shared.train_base import _ensure_sb3",
        "from .train_base import _resolve_vecnorm_sidecar",
        "train_base._PERIODIC_CHECKPOINT_RE.match(s)",
        "environments.shared.train_base._ensure_sb3()",
        "import environments.shared.train_base as tb\ntb._ensure_sb3()",
        "monkeypatch.setattr(train_base, '_ensure_sb3', f)",
        "patch.object(train_base, '_resolve_vecnorm_sidecar')",
        "patch('environments.shared.train_base._ensure_sb3')",
        "monkeypatch.setattr('environments.shared.train_base._ensure_sb3', f)",
    ]
    ignored = [
        "from environments.shared.policy_loading import _ensure_sb3",
        "policy_loading._ensure_sb3()",
        "monkeypatch.setattr(policy_loading, '_ensure_sb3', f)",
        "patch('environments.shared.policy_loading._ensure_sb3')",
        "train_base.create_vec_env(cfg)",
        "'''Prose naming train_base._ensure_sb3 is not a patch target.'''",
    ]
    for snippet in caught:
        assert _train_base_reads(ast.parse(snippet)), snippet
    for snippet in ignored:
        assert not _train_base_reads(ast.parse(snippet)), snippet


class _Reached(BaseException):
    """A ``BaseException``, so the stage-artifact guards' ``except Exception`` cannot swallow it."""


def _ensure_sb3_callers(monkeypatch, tmp_path) -> dict[str, Callable[[], object]]:
    """The nine library callers of ``_ensure_sb3``, each called just far enough to reach it."""
    from environments.shared import evaluation, task_fingerprint, train_base
    from environments.shared.config import load_all_stages
    from environments.shared.reporting import stage_artifacts

    monkeypatch.setattr(train_base, "current_plant_identity", lambda species: SimpleNamespace(to_dict=dict))
    monkeypatch.setattr(task_fingerprint, "derive_stage_task_fingerprint", lambda **kwargs: {})
    monkeypatch.setitem(sys.modules, "mediapy", SimpleNamespace())
    species_cfg: Any = SimpleNamespace(species="trex", env_class=object, success_keys=["bite_success"])
    models = tmp_path / "models"
    models.mkdir()
    (models / "robust_best_model.zip").write_bytes(b"weights")
    (models / "robust_best_model_vecnorm.pkl").write_bytes(b"stats")
    task_success = {"env_kwargs": {}, "curriculum_kwargs": {"gate_kind": "task_success/v1", "min_eval_episodes": 30}}
    return {
        "make_env": lambda: train_base.make_env(species_cfg, {}, 1, 0),
        "create_vec_env": lambda: train_base.create_vec_env(species_cfg, {}, 1, 1),
        "train": lambda: train_base.train(
            species_cfg,
            load_all_stages("trex"),
            1,
            total_timesteps=1,
            output_dir=str(tmp_path),
            use_tensorboard=False,
            verbose=0,
        ),
        "_post_training_eval_panels": lambda: train_base._post_training_eval_panels(
            species_cfg, None, None, tmp_path, "ppo", None, quality_episodes=1, velocity_episodes=1
        ),
        "train_curriculum": lambda: train_base.train_curriculum(species_cfg, {}),
        "record_stage_video": lambda: evaluation.record_stage_video(None, object, {}, 1, tmp_path),
        "evaluate": lambda: evaluation.evaluate(species_cfg, {1: {"env_kwargs": {}}}, str(tmp_path / "m.zip")),
        "_write_task_success_evidence": lambda: stage_artifacts._write_task_success_evidence(
            species_cfg=species_cfg,
            stage=3,
            stage_config=task_success,
            stage_dir=tmp_path,
            model_dir=models,
            algorithm="ppo",
        ),
        "_record_stage_replays": lambda: stage_artifacts._record_stage_replays(
            species_cfg=species_cfg,
            stage_config={"env_kwargs": {}},
            stage=1,
            algorithm="ppo",
            stage_dir=tmp_path,
            replays_out=tmp_path,
            model_dir=models,
            seed=0,
            stage_results={},
            allow_legacy_plant=False,
        ),
    }


@pytest.mark.parametrize(
    "caller",
    [
        "make_env",
        "create_vec_env",
        "train",
        "_post_training_eval_panels",
        "train_curriculum",
        "record_stage_video",
        "evaluate",
        "_write_task_success_evidence",
        "_record_stage_replays",
    ],
)
def test_one_patch_of_ensure_sb3_reaches_every_caller(caller, monkeypatch, tmp_path):
    """Each caller's OWN call reaches the patch: the spy records the function that called it.

    Each parametrised name is the name of the function that calls the helper.
    With SB3 installed, a caller whose own lookup went elsewhere (the
    ``train_base`` alias, say) gets the real classes and carries on, and could
    reach the spy through a caller further down (``create_vec_env`` through
    ``make_env``, ``train`` and ``_write_task_success_evidence`` through
    ``create_vec_env``).
    """
    callers = _ensure_sb3_callers(monkeypatch, tmp_path)
    assert len(callers) == 9

    def reached():
        raise _Reached(sys._getframe(1).f_code.co_name)

    monkeypatch.setattr(policy_loading, "_ensure_sb3", reached)
    with pytest.raises(_Reached) as excinfo:
        callers[caller]()
    assert excinfo.value.args == (caller,)


@pytest.mark.parametrize("caller", ["resolve_vecnorm_path", "evaluate", "_load_vecnorm_into_envs"])
def test_one_patch_of_the_resolver_reaches_every_caller(caller, monkeypatch, tmp_path):
    from environments.shared import evaluation, plant_contract, train_base

    def reached(load_path):
        raise _Reached(load_path)

    model_path = str(tmp_path / "stage1_5_steps.zip")
    monkeypatch.setattr(policy_loading, "_resolve_vecnorm_sidecar", reached)
    monkeypatch.setattr(policy_loading, "_ensure_sb3", dict)
    monkeypatch.setattr(plant_contract, "current_plant_identity", lambda species: None)
    callers = {
        "resolve_vecnorm_path": lambda: policy_loading.resolve_vecnorm_path(model_path, None, False),
        "evaluate": lambda: evaluation.evaluate(
            SimpleNamespace(species="trex"), {1: {"env_kwargs": {}}}, model_path, stage=1
        ),
        "_load_vecnorm_into_envs": lambda: train_base._load_vecnorm_into_envs(
            model_path, None, None, task_load_mode="resume_same_stage"
        ),
    }
    with pytest.raises(_Reached) as excinfo:
        callers[caller]()
    assert excinfo.value.args == (model_path,)


def test_ensure_sb3_without_sb3_logs_the_install_hint_and_exits_1(monkeypatch, caplog):
    """A pure move: the same log line and ``sys.exit(1)``, not an exception the artifact guards would swallow."""
    monkeypatch.setitem(sys.modules, "stable_baselines3", None)
    with caplog.at_level(logging.ERROR, logger=policy_loading.__name__), pytest.raises(SystemExit) as excinfo:
        policy_loading._ensure_sb3()
    assert excinfo.value.code == 1
    assert [record.getMessage() for record in caplog.records if record.name == policy_loading.__name__] == [
        SB3_MISSING_MESSAGE
    ]


def test_ensure_sb3_returns_the_ten_sb3_names():
    sb3 = pytest.importorskip("stable_baselines3")
    names = policy_loading._ensure_sb3()
    assert set(names) == SB3_NAMES
    assert names["PPO"] is sb3.PPO and names["SAC"] is sb3.SAC


def test_policy_loading_resolves_a_sidecar_without_sb3_torch_or_train_base(tmp_path):
    """``policy_loading`` names ``train_base`` in no import, and works without SB3 or torch.

    In a subprocess where importing SB3 or torch fails as it does when they are
    not installed, ``policy_loading`` imports, resolves a periodic sidecar
    without loading ``train_base``, and ``_ensure_sb3`` then exits 1 with the
    install hint. Neither can be loaded there, so this pins that neither is
    needed; that neither is loaded when both are installed is the next test's.
    """
    tree = ast.parse((REPO_ROOT / POLICY_LOADING).read_text(encoding="utf-8"))
    imported = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imported += [
        f"{node.module or ''}.{alias.name}"
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    ]
    assert not [name for name in imported if "train_base" in name]

    sidecar = tmp_path / "stage1_vecnormalize_5_steps.pkl"
    sidecar.touch()
    code = f"""
import sys


class _Absent:
    \"\"\"SB3 and torch are not installed: importing either fails the way a missing package does.\"\"\"

    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in {{"stable_baselines3", "torch"}}:
            raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)
        return None


sys.meta_path.insert(0, _Absent())
from environments.shared import policy_loading

periodic = {str(tmp_path / "stage1_5_steps.zip")!r}
assert policy_loading._resolve_vecnorm_sidecar(periodic) == {str(sidecar)!r}
assert policy_loading.resolve_vecnorm_path(periodic, None, False) == {str(sidecar)!r}
loaded = [m for m in sys.modules if m.split(".")[0] in {{"stable_baselines3", "torch"}}]
loaded += [m for m in sys.modules if m == "environments.shared.train_base"]
assert not loaded, loaded
print("resolved without SB3, torch or train_base", flush=True)
policy_loading._ensure_sb3()
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300
    )
    assert "resolved without SB3, torch or train_base" in result.stdout, result.stderr
    assert result.returncode == 1, result.stderr
    assert SB3_MISSING_MESSAGE in result.stderr
    assert "Traceback" not in result.stderr


def test_policy_loading_resolves_a_sidecar_without_loading_an_installed_sb3_or_torch(tmp_path):
    """With SB3 and torch installed and nothing blocked, a bare resolve loads neither, nor ``train_base``.

    Before CU-8b the same call imported ``train_base``, and SB3 and torch with
    it. Skipped without SB3, where the cost cannot show; the SB3 integration
    step runs it.
    """
    pytest.importorskip("stable_baselines3")
    sidecar = tmp_path / "stage1_vecnormalize_5_steps.pkl"
    sidecar.touch()
    code = f"""
import importlib.util
import json
import sys

from environments.shared import policy_loading

assert policy_loading.resolve_vecnorm_path({str(tmp_path / "stage1_5_steps.zip")!r}, None, False) == {str(sidecar)!r}
loaded = sorted(
    name
    for name in sys.modules
    if name.split(".")[0].startswith(("stable_baselines3", "torch")) or name == "environments.shared.train_base"
)
installed = [name for name in ("stable_baselines3", "torch") if importlib.util.find_spec(name) is not None]
print(json.dumps({{"loaded": loaded, "installed": installed}}))
"""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(REPO_ROOT), os.environ.get("PYTHONPATH")]))}
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.splitlines()[-1]) == {"loaded": [], "installed": ["stable_baselines3", "torch"]}


def test_policy_loading_imports_only_the_standard_library_at_import_time():
    """The module docstring's promise: SB3 is imported inside the functions and no project module at import time.

    Every import that runs when the module is imported (at module level,
    including inside ``try``/``if``/``with`` blocks and class bodies) names a
    standard-library module; the imports inside the functions are the lazy ones.
    """
    tree = ast.parse((REPO_ROOT / POLICY_LOADING).read_text(encoding="utf-8"))
    imported: list[str] = []
    pending: list[ast.AST] = list(tree.body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if isinstance(node, ast.If) and _dotted(node.test).split(".")[-1] == "TYPE_CHECKING":
            pending += node.orelse
            continue
        if isinstance(node, ast.Import):
            imported += [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append("." * node.level + (node.module or ""))
        pending += ast.iter_child_nodes(node)
    assert imported
    assert [name for name in imported if name.split(".")[0] not in sys.stdlib_module_names] == []
