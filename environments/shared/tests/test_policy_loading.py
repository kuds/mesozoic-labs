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
"""

from __future__ import annotations

import ast
import base64
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

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
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for index, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            stripped = "\n".join(
                line if not line.lstrip().startswith(("!", "%")) else line[: len(line) - len(line.lstrip())] + "pass"
                for line in source.splitlines()
            )
            hits = _bare_algorithm_loads(ast.parse(stripped))
            if hits:
                offenders[f"{path.relative_to(REPO_ROOT)}[cell {index}]"] = hits
    assert not offenders, f"bare SB3 archive loads outside policy_loading.load_sb3_model: {offenders}"
    # The loader itself is where the one real call lives.
    loader_source = (REPO_ROOT / "environments/shared/policy_loading.py").read_text(encoding="utf-8")
    assert len(_bare_algorithm_loads(ast.parse(loader_source))) == 1


def test_the_known_load_sites_call_the_loader():
    """The sites the incident review listed, plus the sweep scripts, name ``load_sb3_model``."""
    expected = {
        "environments/shared/train_base.py": 3,
        "environments/shared/evaluation.py": 1,
        "environments/shared/reporting/stage_artifacts.py": 3,
        "environments/shared/harnesses/freeze_recovery_gate.py": 1,
        "environments/shared/scripts/widen_checkpoint.py": 1,
        "environments/shared/behavior_checkpoint.py": 1,
        "environments/shared/scripts/sweep/ray_tune.py": 3,
        "environments/shared/scripts/sweep/ray_orchestration.py": 3,
    }
    for name, count in expected.items():
        source = (REPO_ROOT / name).read_text(encoding="utf-8")
        calls = [
            node
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "load_sb3_model"
        ]
        assert len(calls) == count, (name, len(calls))
    notebook = json.loads((REPO_ROOT / "notebooks/sb3_training.ipynb").read_text(encoding="utf-8"))
    cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    assert sum(cell.count("load_sb3_model(") for cell in cells) >= 3, "the preflight and the two evaluation loads"
