"""Load an SB3 checkpoint together with the VecNormalize statistics it trained under.

Two loaders live here, with the SB3 import helper and the VecNormalize
sidecar resolver that the trainer, the evaluator and the stage artifacts
share. :func:`load_sb3_model` is the ONE way the repository
opens an SB3 model archive (``PPO.load`` / ``SAC.load`` are never called
bare outside this module; ``test_policy_loading`` pins it): it reads the
archive's ``data`` JSON without unpickling anything, supplies the model's
schedule members through SB3's ``custom_objects`` so their cloudpickled
bytecode is never executed, and refuses an archive whose other members
carry bytecode from another Python minor version. SB3 stores a model's
``learning_rate`` / ``lr_schedule`` / ``clip_range`` through cloudpickle,
which pickles a closure BY VALUE with its code object; ``PPO.load`` then
CALLS that closure (``_setup_model`` sets the optimizer's initial learning
rate from it), and bytecode compiled by Python 3.12 executed by Python
3.13 -- or the reverse -- segfaults the interpreter with no traceback. That
killed the Colab kernel twice on 2026-09-19 inside the widen tool's
self-verification (KNOWN_ISSUES, "SB3 archives are bound to the
interpreter that saved them"). The stage config, not the archive, is the
source of truth for the schedules: a training load passes them in and an
inference load never needs them.

:func:`load_sb3_checkpoint` is the one loading path the offline report
scripts and the stance gate report share.

Four copies of this block had drifted apart -- in how they found the
sidecar, in what they did when it was missing, and in whether they checked
the plant -- and every one of those differences changes *which policy* gets
scored: a policy evaluated on raw observations is a different policy, and
the report would blame the joints, the actions or the gate for a loading
mistake.  Keeping the block in one place is what makes the four reports
comparable.

Not the trainer's :func:`~environments.shared.curriculum.checkpoints.load_vecnorm_stats`,
which loads statistics INTO a live training wrapper, and not the
Monitor-wrapped evaluator in :mod:`~environments.shared.evaluation`: this is
the read-only, normalise-then-predict loader an offline rollout wants.

The two helpers moved here from ``train_base`` (cleanup CU-8b), which keeps
its old names bound to these same objects for importers that predate the
move. :func:`_ensure_sb3` is the one place ``train_base``, ``evaluation``
and ``reporting.stage_artifacts`` import SB3's classes, and
:func:`_resolve_vecnorm_sidecar` the one probe that finds a checkpoint's
sidecar under either naming convention (the curated ``<base>_vecnorm.pkl``
or SB3's periodic ``<prefix>_vecnormalize_<steps>_steps.pkl``, matched by
:data:`_PERIODIC_CHECKPOINT_RE`). Every caller imports them from this module
at call time, so one patch of the name here reaches every caller. The
resolver only finds a candidate: what a missing sidecar means stays with
each caller (:func:`resolve_vecnorm_path` raises :class:`PolicyLoadError`,
``evaluation.evaluate`` and ``train_base._load_vecnorm_into_envs`` raise
``FileNotFoundError``, each with its own escape hatch).

SB3 is imported inside the functions, per the repository's lazy-SB3
convention, so the module stays importable without it; no project module is
imported at import time, and ``train_base`` never is. The loaders raise
:class:`PolicyLoadError`, so a diagnostic inside an artifact guard cannot
sink a finished run. :func:`_ensure_sb3` alone exits the process (status 1,
after logging an install hint) when SB3 is missing, as it always has: its
callers are the training, evaluation and replay paths, which can do nothing
without SB3, and two of them run inside the stage-artifact guards, which
would swallow an exception as one more failed diagnostic instead of stopping
the run.
"""

from __future__ import annotations

import base64
import dataclasses
import io
import json
import logging
import pickletools
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

UNNORMALIZED_BANNER = "UNNORMALIZED EVAL — results are not comparable to training-time metrics"

#: The archive members SB3 stores as callables, per algorithm: every one may
#: hold a cloudpickled closure (the repository's LR / clip-range decays, or
#: the nested functions older SB3 releases built for a float value), and ``PPO.load`` calls
#: ``lr_schedule`` while building the policy. ``clip_range_vf`` is a float or
#: ``None`` in every stage TOML but SB3 accepts a schedule there too.
SCHEDULE_MEMBERS: dict[str, tuple[str, ...]] = {
    "ppo": ("learning_rate", "lr_schedule", "clip_range", "clip_range_vf"),
    "sac": ("learning_rate", "lr_schedule"),
}

#: SB3's constructor defaults for the schedule members an inference load
#: replaces when the archive stores them as bytecode. Inference never calls
#: them: ``set_parameters`` restores the optimizer's saved learning rate and
#: ``predict`` reads neither. A training load passes the stage config's own
#: values as keyword arguments, and SB3 applies those over the archive.
INFERENCE_SCHEDULE_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.0,
    "clip_range": 0.2,
    "clip_range_vf": None,
}

#: cloudpickle reducers that appear in a pickle stream only when a function,
#: class or code object was pickled BY VALUE (its bytecode is in the stream).
#: A function pickled by reference is a plain ``module.qualname`` global.
_BYTECODE_REDUCERS = frozenset(
    {
        ("cloudpickle.cloudpickle", "_make_function"),
        ("cloudpickle.cloudpickle", "_make_skeleton_class"),
        ("cloudpickle.cloudpickle", "_make_cell"),
        ("cloudpickle.cloudpickle", "_make_empty_cell"),
        ("cloudpickle.cloudpickle", "_builtin_type"),
        ("cloudpickle.cloudpickle_fast", "_make_function"),
        ("cloudpickle.cloudpickle_fast", "_make_skeleton_class"),
        ("cloudpickle", "_make_function"),
        ("types", "CodeType"),
    }
)
_SYSTEM_INFO_PYTHON_RE = re.compile(r"^-?\s*Python:\s*(\d+)\.(\d+)", re.MULTILINE)


class PolicyLoadError(RuntimeError):
    """A checkpoint could not be opened safely: unreadable metadata or statistics, or foreign bytecode.

    Raised by :func:`inspect_sb3_archive` for an unreadable archive, by
    :func:`load_sb3_model` when an archive saved by another Python minor
    version embeds bytecode outside its schedule members, and by
    :func:`load_sb3_checkpoint` when the normalisation statistics cannot be
    resolved or read.

    An ordinary exception (a ``RuntimeError`` subclass) rather than
    ``SystemExit`` on purpose: the stance gate report runs inside the
    training pipeline's artifact guard, which catches ``Exception`` so a
    diagnostic cannot sink a finished run, and ``SystemExit`` (a
    ``BaseException``) would sail straight through it. Each CLI converts
    this to ``SystemExit`` at its own boundary, which is where an exit
    status belongs. :func:`_ensure_sb3`'s exit on a missing SB3 is the one
    deliberate departure (see the module docstring).
    """


#: SB3's ``CheckpointCallback`` names its periodic checkpoints
#: ``{prefix}_{steps}_steps.zip`` (``CheckpointCallback._checkpoint_path``;
#: mirrored by ``curriculum.checkpoints._CHECKPOINT_KINDS``).
_PERIODIC_CHECKPOINT_RE = re.compile(r"(.+)_(\d+)_steps$")


def _ensure_sb3() -> dict[str, Any]:
    """Import SB3 or exit with a helpful error."""
    try:
        from stable_baselines3 import PPO, SAC
        from stable_baselines3.common.callbacks import (
            CallbackList,
            CheckpointCallback,
            EvalCallback,
        )
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.utils import set_random_seed
        from stable_baselines3.common.vec_env import (
            DummyVecEnv,
            SubprocVecEnv,
            VecNormalize,
        )

        return {
            "PPO": PPO,
            "SAC": SAC,
            "CallbackList": CallbackList,
            "CheckpointCallback": CheckpointCallback,
            "EvalCallback": EvalCallback,
            "Monitor": Monitor,
            "set_random_seed": set_random_seed,
            "DummyVecEnv": DummyVecEnv,
            "SubprocVecEnv": SubprocVecEnv,
            "VecNormalize": VecNormalize,
        }
    except ImportError:
        logger.error("stable-baselines3 not installed. Install with: pip install stable-baselines3[extra]")
        sys.exit(1)


def _resolve_vecnorm_sidecar(load_path: str) -> str:
    """Resolve the VecNormalize sidecar path for a checkpoint being loaded.

    Two sidecar naming conventions coexist: this repository's curated
    checkpoints (``best_model``, ``robust_best_model``, ``stage<N>_final``)
    save ``<base>_vecnorm.pkl``, while SB3's
    ``CheckpointCallback(save_vecnormalize=True)`` writes
    ``<prefix>_vecnormalize_<steps>_steps.pkl`` for its periodic
    ``<prefix>_<steps>_steps.zip``.  Probing only the curated name made a
    ``--load stage2_5000000_steps.zip`` resume warn and then train the loaded
    policy under fresh normalization statistics — silently (review F3).

    A ``load_path`` that already names a ``.pkl`` file is returned unchanged:
    ``train_curriculum`` hands the sidecar path itself, and appending
    ``_vecnorm.pkl`` to it would probe a file that cannot exist.

    Returns the first existing candidate; when none exists, the curated
    ``<base>_vecnorm.pkl`` name, so the caller's warning names the primary
    probe.
    """
    if load_path.endswith(".pkl"):
        return load_path
    base = load_path[:-4] if load_path.endswith(".zip") else load_path
    curated = base + "_vecnorm.pkl"
    if Path(curated).exists():
        return curated
    match = _PERIODIC_CHECKPOINT_RE.match(Path(base).name)
    if match:
        periodic = Path(base).parent / f"{match.group(1)}_vecnormalize_{match.group(2)}_steps.pkl"
        if periodic.exists():
            return str(periodic)
    return curated


def resolve_vecnorm_path(model_path: str, vecnorm_arg: str | None, allow_unnormalized: bool) -> str | None:
    """The VecNormalize sidecar to evaluate with, or ``None`` for a deliberately unnormalised run.

    An explicit *vecnorm_arg* (the CLI's ``--vecnorm``) wins.  Otherwise the
    trainer's own resolver, :func:`_resolve_vecnorm_sidecar`, probes both
    sidecar conventions -- the ``<stem>_vecnorm.pkl`` guess the report
    scripts used to make can never match SB3's periodic
    ``<prefix>_vecnormalize_<steps>_steps.pkl``, so every periodic
    checkpoint was silently scored on raw observations.  No
    sidecar is fatal unless *allow_unnormalized*: a policy evaluated
    unnormalised is a different policy, and the report would blame the
    policy for a loading mistake.
    """
    if vecnorm_arg is not None:
        return vecnorm_arg
    candidate = _resolve_vecnorm_sidecar(model_path)
    if Path(candidate).exists():
        return candidate
    if allow_unnormalized:
        return None
    raise PolicyLoadError(
        f"no VecNormalize sidecar found for {model_path} (probed {candidate}). A policy evaluated on "
        "unnormalised observations is a different policy; pass --vecnorm, or --allow-unnormalized "
        "to proceed deliberately."
    )


def _checkpoint_algorithm(model_path: str) -> Any:
    """Choose PPO or SAC from the checkpoint's JSON optimizer metadata.

    Report filenames such as ``robust_best_model.zip`` do not encode the
    algorithm. Inspect JSON before loading; trying PPO and falling back after
    arbitrary load errors can hide a corrupt checkpoint. Custom policy classes
    retain these algorithm attributes, so their module names are not required.
    """
    path = Path(model_path)
    if not path.is_file():
        path = Path(f"{model_path}.zip")
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise PolicyLoadError(f"cannot read SB3 checkpoint metadata from {model_path}: {exc}") from exc
    is_ppo = isinstance(data, dict) and "clip_range" in data and "n_epochs" in data
    is_sac = isinstance(data, dict) and "target_entropy" in data and "replay_buffer_class" in data
    if is_ppo == is_sac:
        raise PolicyLoadError(f"checkpoint {model_path} does not identify exactly one supported algorithm (PPO/SAC)")
    from stable_baselines3 import PPO, SAC

    return PPO if is_ppo else SAC


def _archive_path(model_path: "str | Path") -> Path:
    """The archive file for *model_path*, which SB3 lets callers give with or without ``.zip``."""
    path = Path(model_path)
    if not path.is_file() and not str(path).endswith(".zip"):
        path = Path(f"{model_path}.zip")
    return path


def _pickle_globals(payload: bytes) -> "set[tuple[str, str]] | None":
    """Every ``(module, name)`` global a pickle stream references, found without executing it.

    ``GLOBAL`` carries the pair itself. ``STACK_GLOBAL`` (protocol 4+) pops
    it from the stack, where each half was pushed by a string opcode -- or
    fetched from the memo (``BINGET`` and friends) when the same string,
    typically the ``cloudpickle.cloudpickle`` module name, appeared
    earlier in the stream -- so the walk tracks the memo for strings too.
    ``None`` when the stream cannot be walked: a member that cannot even be
    disassembled is treated as if it carried bytecode.
    """
    string_ops = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "STRING", "SHORT_BINSTRING", "BINSTRING"}
    put_ops = {"BINPUT", "LONG_BINPUT", "PUT"}
    get_ops = {"BINGET", "LONG_BINGET", "GET"}
    found: set[tuple[str, str]] = set()
    memo: dict[int, "str | None"] = {}
    # The two most recent values on the stack when both are strings (the
    # (module, name) pair STACK_GLOBAL consumes); any other value breaks it.
    recent: list[str] = []
    last: "str | None" = None
    try:
        for opcode, arg, _pos in pickletools.genops(io.BytesIO(payload)):
            name = opcode.name
            if name == "GLOBAL":
                module, _, attribute = str(arg).partition(" ")
                found.add((module, attribute))
                recent, last = [], None
            elif name == "STACK_GLOBAL":
                if len(recent) >= 2:
                    found.add((recent[-2], recent[-1]))
                recent, last = [], None
            elif name in string_ops:
                last = str(arg)
                recent.append(last)
            elif name == "MEMOIZE":
                memo[len(memo)] = last
            elif name in put_ops:
                memo[int(arg) if arg is not None else -1] = last
            elif name in get_ops:
                last = memo.get(int(arg) if arg is not None else -1)
                if last is None:
                    recent = []
                else:
                    recent.append(last)
            else:
                recent, last = [], None
    except Exception:  # noqa: BLE001 - a malformed stream is reported as unknown, never trusted
        return None
    return found


def _carries_bytecode(globals_: "set[tuple[str, str]] | None") -> bool:
    """Whether a pickle's globals show something pickled by value (its bytecode is in the stream).

    Any reference into cloudpickle's own reducers means a function, class,
    cell or code object was serialised by value rather than by import path;
    the named reducers are the ones seen in practice, the module test is
    the backstop for cloudpickle versions that reorganise them.
    """
    if globals_ is None:
        return True
    return bool(globals_ & _BYTECODE_REDUCERS) or any(module.split(".")[0] == "cloudpickle" for module, _ in globals_)


@dataclasses.dataclass(frozen=True)
class SB3ArchiveInspection:
    """What an SB3 archive's ``data`` member says without unpickling any of it."""

    path: Path
    #: ``(major, minor)`` of the Python that saved the archive (``system_info.txt``), or ``None``.
    saved_python: "tuple[int, int] | None"
    #: The ``data`` members stored as cloudpickle payloads (everything that is not plain JSON).
    serialized_members: frozenset[str]
    #: The serialized members whose payload embeds function / class / code bytecode.
    bytecode_members: frozenset[str]

    @property
    def saved_python_text(self) -> str:
        return "unknown" if self.saved_python is None else f"{self.saved_python[0]}.{self.saved_python[1]}"

    @property
    def cross_interpreter(self) -> "bool | None":
        """``True`` when the saving Python minor version differs from this one; ``None`` when unrecorded."""
        if self.saved_python is None:
            return None
        return self.saved_python != tuple(sys.version_info[:2])


def inspect_sb3_archive(model_path: "str | Path") -> SB3ArchiveInspection:
    """Read an SB3 archive's ``data`` JSON and ``system_info.txt`` without executing anything in them.

    Raises :class:`PolicyLoadError` when the archive or its ``data`` member
    cannot be read.
    """
    path = _archive_path(model_path)
    try:
        with zipfile.ZipFile(path) as archive:
            data = json.loads(archive.read("data"))
            names = set(archive.namelist())
            system_info = (
                archive.read("system_info.txt").decode("utf-8", "replace") if "system_info.txt" in names else ""
            )
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise PolicyLoadError(f"cannot read SB3 checkpoint metadata from {model_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise PolicyLoadError(f"SB3 checkpoint {model_path} holds no data mapping")
    match = _SYSTEM_INFO_PYTHON_RE.search(system_info)
    saved_python = (int(match.group(1)), int(match.group(2))) if match else None
    serialized: set[str] = set()
    bytecode: set[str] = set()
    for key, value in data.items():
        if not (isinstance(value, dict) and ":serialized:" in value):
            continue
        serialized.add(key)
        try:
            payload = base64.b64decode(str(value[":serialized:"]).encode())
        except (ValueError, TypeError):
            bytecode.add(key)
            continue
        if _carries_bytecode(_pickle_globals(payload)):
            bytecode.add(key)
    return SB3ArchiveInspection(
        path=path,
        saved_python=saved_python,
        serialized_members=frozenset(serialized),
        bytecode_members=frozenset(bytecode),
    )


def _algorithm_name(alg_cls: Any) -> "str | None":
    name = str(getattr(alg_cls, "__name__", "")).lower()
    return name if name in SCHEDULE_MEMBERS else None


def _resolve_algorithm(model_path: "str | Path", algorithm: Any) -> tuple[Any, "str | None"]:
    """``(class, "ppo" | "sac" | None)`` for *algorithm*: a name, an SB3 class (or a test double), or ``None`` to sniff."""
    if algorithm is None:
        alg_cls = _checkpoint_algorithm(str(model_path))
        return alg_cls, _algorithm_name(alg_cls)
    if isinstance(algorithm, str):
        name = algorithm.lower()
        if name not in SCHEDULE_MEMBERS:
            raise PolicyLoadError(f"unknown SB3 algorithm {algorithm!r}; expected ppo or sac")
        from stable_baselines3 import PPO, SAC

        return (SAC if name == "sac" else PPO), name
    return algorithm, _algorithm_name(algorithm)


def schedule_custom_objects(
    inspection: "SB3ArchiveInspection | None",
    algorithm: "str | None",
    *,
    custom_objects: "Mapping[str, Any] | None" = None,
    training_kwargs: "Mapping[str, Any] | None" = None,
) -> dict[str, Any]:
    """The ``custom_objects`` that keep an archive's schedule bytecode from ever being unpickled.

    A schedule member (:data:`SCHEDULE_MEMBERS`) is replaced when the
    archive stores it as bytecode -- or, when *inspection* is ``None``
    (the archive could not be read), unconditionally. The replacement is
    the caller's *custom_objects* entry, else the caller's training keyword
    (``learning_rate`` / ``clip_range`` / ``clip_range_vf`` from the stage
    config; ``lr_schedule`` follows ``learning_rate``), else the inference
    default. The values only need to be picklable by reference and never
    executed as foreign code: SB3 rebuilds ``lr_schedule`` and wraps
    ``clip_range`` in ``_setup_model``, and a training load's keywords are
    applied over the archive before that. A JSON-stored member (a float
    learning rate, a ``None`` ``clip_range_vf``) is left alone.
    """
    from environments.shared.curriculum.schedules import _ConstantSchedule

    custom: dict[str, Any] = dict(custom_objects or {})
    kwargs = dict(training_kwargs or {})
    members: tuple[str, ...]
    if algorithm is None:
        members = tuple(dict.fromkeys(member for group in SCHEDULE_MEMBERS.values() for member in group))
    else:
        members = SCHEDULE_MEMBERS[algorithm]
    for member in members:
        if member in custom:
            continue
        if inspection is not None and member not in inspection.bytecode_members:
            continue
        if member == "lr_schedule":
            learning_rate = custom.get(
                "learning_rate", kwargs.get("learning_rate", INFERENCE_SCHEDULE_DEFAULTS["learning_rate"])
            )
            custom[member] = learning_rate if callable(learning_rate) else _ConstantSchedule(float(learning_rate))
        elif member in kwargs:
            custom[member] = kwargs[member]
        else:
            custom[member] = INFERENCE_SCHEDULE_DEFAULTS[member]
    return custom


def load_sb3_model(
    model_path: "str | Path",
    *,
    algorithm: Any = None,
    env: Any = None,
    device: "str | None" = None,
    custom_objects: "Mapping[str, Any] | None" = None,
    **load_kwargs: Any,
) -> Any:
    """``PPO.load`` / ``SAC.load`` without executing any bytecode the archive carries.

    *algorithm* is ``"ppo"`` / ``"sac"``, the SB3 class itself (a caller's
    ``sb3["PPO"]`` or a test double is used as given), or ``None`` to read
    it off the archive's optimizer metadata. *env* and *device* go to SB3
    unchanged (``device`` is omitted when ``None`` so SB3's ``"auto"``
    applies, as it does for every training load today). Every other keyword
    is SB3's ``**kwargs`` -- the stage config's algorithm block on a
    training load -- and doubles as the source of the schedule replacements
    (:func:`schedule_custom_objects`).

    Fail-closed: when the archive was saved by another Python minor version
    and a member OTHER than the schedules embeds bytecode (a lambda in
    ``policy_kwargs``, say), the load is refused with a
    :class:`PolicyLoadError` naming the members, because SB3 would unpickle
    and may execute it. An archive whose ``data`` cannot be read at all is
    handed to SB3 with every schedule member replaced, so SB3 raises its own
    error for a missing or corrupt file exactly as before.
    """
    alg_cls, algorithm_name = _resolve_algorithm(model_path, algorithm)
    try:
        inspection: "SB3ArchiveInspection | None" = inspect_sb3_archive(model_path)
    except PolicyLoadError:
        inspection = None
    custom = schedule_custom_objects(
        inspection, algorithm_name, custom_objects=custom_objects, training_kwargs=load_kwargs
    )
    if inspection is not None:
        foreign = sorted(inspection.bytecode_members - set(custom))
        if foreign and inspection.cross_interpreter:
            raise PolicyLoadError(
                f"{inspection.path} was saved by Python {inspection.saved_python_text} and its "
                f"{', '.join(foreign)} member(s) embed cloudpickled bytecode that this Python "
                f"{sys.version_info[0]}.{sys.version_info[1]} would execute; refusing to load it. Re-save the "
                "archive under the saving interpreter with those members replaced, or pass custom_objects for them."
            )
        if foreign and inspection.cross_interpreter is None:
            logger.warning(
                "%s records no saving Python version and its %s member(s) embed bytecode; loading anyway",
                inspection.path,
                ", ".join(foreign),
            )
        replaced = sorted(set(custom) & inspection.bytecode_members)
        if replaced:
            logger.info(
                "%s: schedule member(s) %s stored as bytecode by Python %s are supplied, not unpickled",
                inspection.path,
                ", ".join(replaced),
                inspection.saved_python_text,
            )
    if device is not None:
        load_kwargs["device"] = device
    return alg_cls.load(model_path, env=env, custom_objects=custom, **load_kwargs)


def load_sb3_checkpoint(
    model_path: str,
    vecnorm_path: str | None,
    env_factory: Callable[[], Any],
    *,
    guess_sidecar: bool = True,
    allow_unnormalized: bool = False,
    plant_identity: Any = None,
    allow_legacy_plant: bool = False,
    reseed_command_slice: bool = False,
) -> tuple[Any, Any, str | None]:
    """Return ``(model, normalizer, resolved_vecnorm_path)`` for a saved SB3 checkpoint.

    :func:`load_sb3_model` on the CPU, the algorithm selected from the
    checkpoint's recorded optimizer metadata, then the statistics through ``VecNormalize.load``
    over a throwaway ``DummyVecEnv`` built from *env_factory* -- the loader
    needs a live env to rebuild the wrapper, and hand-unpickling reconstructs
    a partial object whose ``__setstate__`` expectations drift with the SB3
    version.  The wrapper comes back frozen (``training = False``,
    ``norm_reward = False``) so a rollout can call ``normalize_obs`` without
    moving the statistics.

    A *vecnorm_path* of ``None`` means: when *guess_sidecar*, probe the
    checkpoint's sidecar under both naming conventions; when that finds
    nothing, or when guessing is off, run unnormalised only if
    *allow_unnormalized*.  Fail-closed by default because the failure is
    silent -- a policy scored on raw observations is a different policy.
    ``normalizer`` and the resolved path are both ``None`` for an
    unnormalised run, and the caller prints :data:`UNNORMALIZED_BANNER`.

    With *plant_identity*, both artifacts are validated against it in the
    order they load -- the model right after the algorithm's ``load``, the statistics
    right after ``VecNormalize.load`` -- so a checkpoint from another plant
    is refused before anything is scored; *allow_legacy_plant* admits one
    that predates the contract.  An unreadable sidecar raises
    :class:`PolicyLoadError` naming the file; plant refusals propagate as
    ``PlantCompatibilityError``.

    *reseed_command_slice* resets the trailing command slice of the loaded
    ``obs_rms`` to mean 0 / var 1 (``command_frame.reseed_command_slice``).
    It is for a sidecar whose command slice was never live -- a
    ``command_mode = "none"`` parent scored under a live command, the same
    case ``load_vecnorm_stats`` reseeds when such a parent is loaded into a
    live node (BEHAVIOR_RECIPES_PLAN §4.6, invariant 8).  A checkpoint saved
    by a live-command node already carries the slice statistics it trained
    under; loading it with the flag would score the policy under a
    normaliser it never saw, so the default stays ``False`` and Phase D
    wires the flag from the stage config only for that never-live case.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = load_sb3_model(model_path, device="cpu")
    if plant_identity is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(model, plant_identity, artifact=model_path, allow_legacy=allow_legacy_plant)

    if vecnorm_path is None:
        if guess_sidecar:
            vecnorm_path = resolve_vecnorm_path(model_path, None, allow_unnormalized)
        elif not allow_unnormalized:
            raise PolicyLoadError(
                f"no VecNormalize statistics given for {model_path}. A policy evaluated on unnormalised "
                "observations is a different policy; name the checkpoint's sidecar, or allow an "
                "unnormalised run deliberately."
            )
    if vecnorm_path is None:
        return model, None, None

    try:
        normalizer = VecNormalize.load(vecnorm_path, DummyVecEnv([env_factory]))
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        # A truncated or text-mode-copied .pkl is the usual cause, and the raw
        # UnpicklingError/KeyError gives no hint that the file rather than the
        # code is at fault.
        raise PolicyLoadError(
            f"cannot read VecNormalize statistics from {vecnorm_path}: "
            f"{type(exc).__name__}: {exc}. Re-copy the file in binary mode. "
            "Running without --vecnorm would evaluate the policy on unnormalised "
            "observations — a different policy — so this is fatal, not a warning."
        ) from exc
    normalizer.training = False
    normalizer.norm_reward = False
    if reseed_command_slice:
        from environments.shared.command_frame import reseed_command_slice as _reseed_command_slice

        _reseed_command_slice(normalizer.obs_rms)

    if plant_identity is not None:
        from environments.shared.plant_contract import validate_model_plant

        validate_model_plant(normalizer, plant_identity, artifact=vecnorm_path, allow_legacy=allow_legacy_plant)

    return model, normalizer, vecnorm_path


# ── the SB3 notebook's archive-load preflight (cleanup CU-6) ─────────────────


def _root_handoff_archive(species: str, root_reference: "int | str", run_dir: "str | Path") -> "Path | None":
    """The chain root's handoff archive under *run_dir* (either stage-directory naming, newest first), or ``None``."""
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint
    from environments.shared.stage_manifest import stage_dir_candidates

    for name in stage_dir_candidates(species, root_reference):
        handoff = select_handoff_checkpoint(Path(run_dir) / name / "models")
        if handoff is not None:
            return Path(handoff[1] + ".zip")
    return None


def _save_throwaway_ppo(path: Path) -> None:
    """Save a minimal PPO model to *path* with this runtime (a 2-obs/1-action env, an 8-unit net); never trained."""
    import gymnasium as gym
    import numpy as np
    from stable_baselines3 import PPO

    class _PreflightEnv(gym.Env):
        """Tiny 2-obs/1-action env; exists only to construct a minimal PPO."""

        observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        def reset(self, *, seed: "int | None" = None, options: "dict[str, Any] | None" = None) -> Any:
            super().reset(seed=seed)
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action: Any) -> Any:
            return np.zeros(2, dtype=np.float32), 0.0, True, False, {}

    PPO("MlpPolicy", _PreflightEnv(), n_steps=32, batch_size=32, policy_kwargs={"net_arch": [8]}, device="cpu").save(
        str(path)
    )


def sb3_archive_load_preflight(
    species: str, root_reference: "int | str", *, trunk_dir: "str | Path | None"
) -> SB3ArchiveInspection:
    """Prove that this runtime loads a real SB3 archive through :func:`load_sb3_model`, before anything trains.

    The SB3 notebook's section-4 preflight: until cleanup CU-6 the body of the
    preflight cell, which runs right after the resolve cell (KNOWN_ISSUES, "SB3
    archives are bound to the interpreter that saved them"). The archive is
    the chain root's handoff in the trunk run *trunk_dir*: *root_reference*
    of *species*' manifest, its directory found under either naming
    (:func:`~environments.shared.stage_manifest.stage_dir_candidates`, newest
    first) and its handoff chosen as the next node would load it
    (:func:`~environments.shared.curriculum.checkpoints.select_handoff_checkpoint`).
    Without a trunk (``trunk_dir=None``), or when the trunk's root holds no
    complete handoff pair, it is a throwaway PPO this very runtime saves into
    a temporary directory, removed before this returns; a trunk without a
    pair is never refused.

    The archive is inspected; then a line naming it, the Python that saved it
    and its bytecode members is printed and flushed immediately before the
    one load, so a kernel death (an archive whose cloudpickled bytecode this
    interpreter cannot execute kills the process with no traceback) is
    attributable to the load; a "passed" line follows the load.

    Returns the loaded archive's :class:`SB3ArchiveInspection` (a throwaway's
    ``path`` no longer exists). Catches nothing and adds no refusal:
    :class:`PolicyLoadError` from :func:`inspect_sb3_archive` (an unreadable
    handoff) or :func:`load_sb3_model` (foreign bytecode outside the schedule
    members), SB3's own error when the load fails and ``ImportError`` without
    SB3 propagate, so a failure halts the notebook's Run all before any
    training. Trains nothing and writes nothing outside its temporary
    directory. The project modules, gymnasium, numpy and SB3 are imported
    when it is called, and ``train_base`` never is.
    """
    import tempfile

    archive: "Path | None" = None
    source = "a throwaway model saved by this runtime"
    if trunk_dir is not None:
        archive = _root_handoff_archive(species, root_reference, trunk_dir)
        if archive is not None:
            source = f"the trunk run's root handoff {archive}"
    with tempfile.TemporaryDirectory(prefix="sb3_load_preflight_") as scratch:
        if archive is None:
            archive = Path(scratch) / "preflight_ppo.zip"
            _save_throwaway_ppo(archive)
        inspection = inspect_sb3_archive(archive)
        print(
            f"SB3 archive load preflight: loading {source} (saved by Python "
            f"{inspection.saved_python_text}; this runtime is Python {sys.version_info[0]}.{sys.version_info[1]}; "
            f"bytecode members: {', '.join(sorted(inspection.bytecode_members)) or 'none'}) ... "
            "a kernel death HERE means this image cannot load SB3 archives",
            flush=True,
        )
        model = load_sb3_model(archive, device="cpu")
        del model
    print("SB3 archive load preflight passed: archives load back on this runtime through load_sb3_model.")
    return inspection
