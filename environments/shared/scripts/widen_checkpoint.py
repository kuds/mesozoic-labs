"""Widen a certified SB3 checkpoint pair from policy-interface revision r to r+k (1 <= k <= --max-revision-gap, default 1).

BEHAVIOR_RECIPES_PLAN §4.6 ("Widening instead of retraining", "Exact
transfer"; invariant 7; decisions D-C8, D-C9, D-C10, D-C12): Phase C appends
a 3-dim body-relative command segment at the END of every species'
observation, so a policy trained under the previous interface is the SAME
function under the new one once its first layers gain zero columns for the
new dims.  This tool maps one PPO or SAC checkpoint plus its VecNormalize
sidecar across that bump — or, under ``--max-revision-gap N``, across up
to N fingerprint-only bumps ending in it — and writes them into a
judge-ready stage directory; the policy is widened, never re-labelled, and
never re-trained:

* **Identity gate** (never hand-edited): the parent's recorded plant
  identity must be the current plant a bounded number of interface-only
  revisions behind — same species, physics digest, ``nq``/``nv``/``nu`` and
  ``action_dim``, ``1 <= current.policy_interface_revision -
  parent.policy_interface_revision <= max_revision_gap`` and
  ``observation_dim + COMMAND_WIDTH == current``.  The bound defaults to 1
  (fail closed: exactly r -> r+1); ``--max-revision-gap N`` (D-C17) admits a
  parent up to N revisions behind, which only a chain of fingerprint-only
  bumps can satisfy since every other field — the widths in particular —
  is still checked.  Opting in asserts, from ``plant_versions.toml``'s
  numbered notes, that no intermediate bump changed an observation layout,
  physics or action field.  Anything else is refused with the differing
  fields named, the measured gap and the bound.  A parent without an
  identity is refused unless ``--allow-legacy-plant`` (then the saved
  observation space is its width).
* **Archive rewrite** through SB3's own serializer: every 2-D tensor whose
  ``in_features`` equals the parent width gets the zero columns APPENDED
  (PPO ``mlp_extractor.policy_net.0`` / ``value_net.0``, SAC
  ``actor.latent_pi.0``); every tensor whose ``in_features`` equals
  ``parent width + action_dim`` (the SAC critics, which read the
  observation-action pair) gets them INSERTED after the observation block
  so the action columns shift intact — refused on a PPO archive.  The Adam
  ``exp_avg`` / ``exp_avg_sq`` moments of those tensors are padded
  identically (D-C10: unpadded moments load but crash on the first
  optimizer step; a stripped member fails ``load`` with ``exact_match``),
  matched to their parameters by order with shape assertions; any other
  optimizer moment still shaped like the pre-pad weight (RMSprop, SGD
  momentum) is refused rather than carried at the parent width.  The saved
  observation space becomes the ``Box(-inf, inf, (new_dim,), float32)``
  ``BaseDinoEnv`` builds, ``_last_obs`` / ``_last_original_obs`` are
  cleared, ``num_timesteps`` is kept (D-C12), the plant identity and the
  stage's task fingerprint are re-stamped, and ``mesozoic_widen_lineage``
  records the parent hashes and the revision gap crossed; the parent's
  ``mesozoic_task_lineage`` stays.
* **Sidecar rewrite**: ``obs_rms`` gains the reseeded command slice (mean 0
  / var 1 at the carried count — ``command_frame.pad_running_stats``, the
  reseed rule applied at birth), the observation space is widened, the
  identity re-stamped, and the wrapper is re-bound once to the real widened
  environment before it is saved frozen (``training = False``).
* **Output layout** (D-C9): ``models/<handoff>.zip`` + ``<handoff>_vecnorm
  .pkl`` under the parent's own handoff name (``robust_best_model`` or
  ``best_model`` — exactly one), byte-identical ``models/<stage_label>
  _final.*`` copies (the notebook JUDGE branch fires on that pair),
  ``stage_config.json`` whose run block carries the parent's ``seed`` /
  ``n_envs`` / ``timesteps`` / ``duration_seconds`` and the
  ``config.WIDEN_LINEAGE_KEYS`` (NEVER the ``LOAD_LINEAGE_KEYS`` — a widened
  node is a root, D-C8), ``plant_identity.json``, ``task_fingerprint.json``
  and ``widen_report.json``.  It never writes ``gate_verdict.json``,
  ``provenance.json``, ``gate_resolution.json``, ``evaluations.npz``,
  ``metrics.json`` or periodic checkpoints: the widened node is re-paneled
  before it certifies.
* **Self-verification** before returning: every padded column is exactly
  zero; over a seeded rollout of the real environment the widened policy's
  actions equal the parent's (``np.allclose(atol=1e-6, rtol=0)``) both with
  the command slice zero and with ``COMMAND_PROBE_VECTOR`` in it — the
  §4.6 exact-transfer probe, which holds only because the columns are zero;
  both artifacts validate against the current identity; the files' hashes
  are unchanged by the verification itself.  Any failure deletes the target
  directory and raises :class:`WidenError`.

Refusals (exit 1, nothing written): a parent ``gate_verdict.json`` that did
not pass; an algorithm that disagrees with the archive; an occupied or
non-empty target directory, one not named as the stage's directory
(``stage_dir_candidates``), or one inside the parent's run directory; an
explicit ``--model`` that is not a handoff checkpoint; a missing identity
without ``--allow-legacy-plant``; a missing or unstamped sidecar (or, under
the legacy allowance, one recording another species / width); any
identity-gate failure (a parent further behind than ``--max-revision-gap``
named with the gap and the bound); a ``--max-revision-gap`` below 1.  A
failure after the first write removes the target and every directory the
tool created above it.

Run: ``python -m environments.shared.scripts.widen_checkpoint --species trex
--stage stance --from-stage-dir <run>/01_stance --to-stage-dir
<newrun>/01_stance [--label L] [--parent-run-id ID] [--allow-legacy-plant]
[--max-revision-gap N]`` or the explicit-pair form ``--model <zip> --vecnorm
<pkl> --algorithm ppo|sac --seed S --n-envs N --timesteps T``.
``--max-revision-gap N`` (default 1, an integer >= 1) bounds how many
interface revisions behind the parent may be; the report records
``revision_gap`` (``null`` for an ``--allow-legacy-plant`` parent that
carries no identity) / ``max_revision_gap``.  Prints ``widen_report.json``.

Judging a widened root in the SB3 notebook (decision D-D14 removed the
notebook's widen cell and knobs, so this tool is the widen path): widen into a
NEW run id, one no run uses and the notebook has not opened yet, written as a
timestamp ``YYYYMMDD_HHMMSS`` like the ids the storage cell mints
(``TRUNK_FROM = "auto"`` breaks a coverage tie by the newest directory name,
so an id in another format would outrank every later run), with
``--to-stage-dir <LOG_BASE>/<species>/<algo>/<new run id>/<stage_dirname(species, root)>``
(``01_stance`` for every current root: the chain loop judges the
``<stage_label>_final`` pair in that directory only) and ``--label`` set to
the session's ``RUN_LABEL`` when it sets one.  On Colab, in three steps:

1. run the notebook's section 1 (it checks the repository out at
   ``/content/mesozoic-labs``), then a scratch cell
   ``from google.colab import drive; drive.mount("/content/drive")``, never
   the storage cell, which would mint a run directory and its provenance;
2. take ``LOG_BASE`` = ``/content/drive/MyDrive/mesozoic-labs/logs``, the
   storage cell's Drive log directory (``<repository>/logs`` locally);
3. run ``!cd /content/mesozoic-labs && python -m
   environments.shared.scripts.widen_checkpoint ... --to-stage-dir ...``.

Then run the notebook with ``RUN_ID = "<new run id>"``, ``SEED`` = the
parent's recorded ``run.seed`` (the storage cell refuses any other value
before it writes anything, D-C14) and ``TRUNK_FROM = ""`` (the resolve cell
refuses a trunk while the widened root has no verdict, D-C13; nothing below a
widened root is reusable from another run anyway).

Stable-Baselines3, torch and gymnasium are imported inside the functions so
the module stays importable on a bare install (the lint job).
"""

from __future__ import annotations

import argparse
import json
import logging
import numbers
import pickle
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from environments.shared.command_frame import COMMAND_PROBE_VECTOR, COMMAND_WIDTH, pad_running_stats

logger = logging.getLogger(__name__)

#: Archive attribute recording the parent the checkpoint was widened from.
MODEL_WIDEN_LINEAGE_ATTRIBUTE = "mesozoic_widen_lineage"
WIDEN_TOOL_VERSION = "widen_checkpoint/v1"
WIDEN_LINEAGE_SCHEMA = "mesozoic.widen-lineage/v1"
WIDEN_REPORT_SCHEMA = "mesozoic.widen-report/v1"
WIDEN_REPORT_FILENAME = "widen_report.json"
#: The backend string the task fingerprint is derived under — what
#: ``train_base`` and the recovery harness both use.
FINGERPRINT_BACKEND = "stable-baselines3"
ALGORITHMS = ("ppo", "sac")
#: The handoff names ``select_handoff_checkpoint`` recognises, in its order.
HANDOFF_NAMES = ("robust_best_model", "best_model")
#: How many interface revisions behind the parent may be by default (D-C17):
#: exactly one bump, r -> r+1; ``max_revision_gap`` widens the bound.
DEFAULT_MAX_REVISION_GAP = 1
#: The seeded verification rollout (plan §4.6 "Exact transfer").
VERIFICATION_ROLLOUT_SEED = 3042
VERIFICATION_ROLLOUT_STEPS = 200
#: Action-equality tolerance (amendment A13b): the padded columns are an
#: exact-zero pin; the actions differ only by summation order over a wider
#: first layer, so they are compared allclose and the measured delta recorded.
ACTION_DELTA_ATOL = 1e-6
#: Optimizer members per algorithm and the ``params['policy']`` key prefix
#: their state indices count along (SB3 2.9.0: ``policy.optimizer`` over the
#: whole PPO policy; SAC's ``actor.optimizer`` / ``critic.optimizer`` over
#: ``actor.*`` / ``critic.*`` — ``critic_target`` has no optimizer and
#: ``ent_coef_optimizer`` touches no observation).
_OPTIMIZER_MEMBERS: dict[str, dict[str, str]] = {
    "ppo": {"policy.optimizer": ""},
    "sac": {"actor.optimizer": "actor.", "critic.optimizer": "critic."},
}
_MOMENT_KEYS = ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
#: Files the tool must never leave behind (the judge, the bundle export and
#: the evidence audit each read them as claims about a run that trained).
FORBIDDEN_OUTPUT_FILES = (
    "gate_verdict.json",
    "evaluations.npz",
    "metrics.json",
    "provenance.json",
    "gate_resolution.json",
)


class WidenError(RuntimeError):
    """The pair cannot be widened, or the widened pair failed its own verification."""


@dataclass(frozen=True)
class WidenResult:
    """What :func:`widen_checkpoint` wrote and measured."""

    target_stage_dir: Path
    handoff_name: str
    algorithm: str
    model_zip: Path
    vecnorm_pkl: Path
    final_zip: Path
    final_vecnorm_pkl: Path
    stage_config_path: Path
    report_path: Path
    parent_checkpoint_sha256: str
    parent_normalization_sha256: str
    widened_checkpoint_sha256: str
    widened_normalization_sha256: str
    from_observation_dim: int
    to_observation_dim: int
    #: ``current - parent`` interface revisions crossed (``None`` for a legacy parent without an identity).
    revision_gap: int | None
    max_action_delta_zero_command: float
    max_action_delta_probe_command: float
    report: dict[str, Any] = field(repr=False)


@dataclass(frozen=True)
class _ParentSource:
    """The resolved parent pair and the run facts the widened root inherits."""

    model_zip: Path
    vecnorm_pkl: Path
    handoff_name: str
    declared_algorithm: str | None
    seed: Any
    n_envs: Any
    timesteps: Any
    duration_seconds: float | None
    run_id: str | None
    stage_dir: Path | None


def _load_json(path: Path, *, what: str) -> Any:
    if not path.is_file():
        raise WidenError(f"{what} is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WidenError(f"{what} is unreadable: {path}: {exc}") from exc


def sniff_archive_algorithm(model_zip: "str | Path") -> str:
    """``"ppo"`` / ``"sac"`` from the archive's ``data`` member, exactly as the recovery harness sniffs it."""
    import zipfile

    try:
        with zipfile.ZipFile(model_zip) as archive:
            data = json.loads(archive.read("data"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile) as exc:
        raise WidenError(f"cannot read SB3 checkpoint metadata from {model_zip}: {exc}") from exc
    if not isinstance(data, Mapping):
        raise WidenError(f"{model_zip} contains invalid SB3 checkpoint metadata")
    is_ppo = "clip_range" in data and "n_epochs" in data
    is_sac = "target_entropy" in data and "replay_buffer_class" in data
    if is_ppo == is_sac:
        raise WidenError(f"{model_zip} does not identify exactly one supported algorithm (PPO/SAC)")
    return "ppo" if is_ppo else "sac"


def _resolve_parent(
    *,
    parent_stage_dir: "str | Path | None",
    model_zip: "str | Path | None",
    vecnorm_pkl: "str | Path | None",
    algorithm: str | None,
    seed: Any,
    n_envs: Any,
    timesteps: Any,
    parent_run_id: str | None,
) -> _ParentSource:
    from environments.shared.config import read_stage_duration
    from environments.shared.result_bundle import GATE_VERDICT_FILENAME

    declared = algorithm.lower() if isinstance(algorithm, str) and algorithm else None
    if declared is not None and declared not in ALGORITHMS:
        raise WidenError(f"algorithm must be one of {ALGORITHMS}, not {algorithm!r}")
    run_id = parent_run_id.strip() if isinstance(parent_run_id, str) and parent_run_id.strip() else None

    if parent_stage_dir is not None:
        if model_zip is not None or vecnorm_pkl is not None:
            raise WidenError("give either a parent stage directory or the explicit --model/--vecnorm pair, not both")
        from environments.shared.ancestors import AncestorReuseError, run_id_for
        from environments.shared.curriculum.checkpoints import select_handoff_checkpoint

        stage_dir = Path(parent_stage_dir)
        if not stage_dir.is_dir():
            raise WidenError(f"{stage_dir} is not a stage directory")
        verdict_path = stage_dir / GATE_VERDICT_FILENAME
        if verdict_path.is_file():
            verdict = _load_json(verdict_path, what=GATE_VERDICT_FILENAME)
            passed = verdict.get("passed") if isinstance(verdict, Mapping) else None
            if passed is not True:
                raise WidenError(
                    f"{verdict_path} records passed={passed!r}: only a parent whose gate passed is widened "
                    "(a widened copy of a failed node would carry that node's weights under a fresh directory)"
                )
        handoff = select_handoff_checkpoint(stage_dir / "models")
        if handoff is None:
            raise WidenError(
                f"{stage_dir / 'models'} has no complete handoff pair (robust_best_model or best_model with its "
                "matched _vecnorm.pkl sidecar); a checkpoint without its normalization statistics is not widened"
            )
        handoff_name, model_stem, normalization = handoff
        record = _load_json(stage_dir / "stage_config.json", what="stage_config.json")
        if not isinstance(record, Mapping):
            raise WidenError(f"{stage_dir / 'stage_config.json'} must hold a JSON object")
        run_block = record.get("run")
        if not isinstance(run_block, Mapping):
            raise WidenError(f"{stage_dir / 'stage_config.json'} records no run block (seed / n_envs / timesteps)")
        missing = [key for key in ("seed", "n_envs", "timesteps") if run_block.get(key) is None]
        if missing:
            raise WidenError(
                f"{stage_dir / 'stage_config.json'} run block lacks {missing}; the widened root inherits the "
                "parent's run facts and cannot invent them"
            )
        recorded_algorithm = record.get("algorithm")
        if isinstance(recorded_algorithm, str) and recorded_algorithm:
            recorded_algorithm = recorded_algorithm.lower()
            if declared is not None and declared != recorded_algorithm:
                raise WidenError(
                    f"--algorithm {declared} disagrees with the {recorded_algorithm.upper()} the parent's "
                    "stage_config.json records"
                )
            declared = recorded_algorithm
        if run_id is None:
            try:
                run_id = run_id_for(stage_dir.parent)
            except AncestorReuseError as exc:
                raise WidenError(str(exc)) from exc
        return _ParentSource(
            model_zip=Path(model_stem + ".zip"),
            vecnorm_pkl=Path(normalization),
            handoff_name=handoff_name,
            declared_algorithm=declared,
            seed=run_block["seed"],
            n_envs=run_block["n_envs"],
            timesteps=run_block["timesteps"],
            duration_seconds=read_stage_duration(stage_dir),
            run_id=run_id,
            stage_dir=stage_dir,
        )

    if model_zip is None or vecnorm_pkl is None:
        raise WidenError("give a parent stage directory, or both --model and --vecnorm")
    absent = [
        name for name, value in (("--seed", seed), ("--n-envs", n_envs), ("--timesteps", timesteps)) if value is None
    ]
    if declared is None:
        absent.insert(0, "--algorithm")
    if absent:
        raise WidenError(
            f"the explicit-pair form requires {' '.join(absent)}: the widened root records the parent's run facts"
        )
    zip_path = Path(model_zip)
    if not zip_path.is_file() and not zip_path.name.endswith(".zip"):
        zip_path = zip_path.with_name(zip_path.name + ".zip")
    if not zip_path.is_file():
        raise WidenError(f"parent checkpoint not found: {model_zip}")
    pkl_path = Path(vecnorm_pkl)
    if not pkl_path.is_file():
        raise WidenError(
            f"parent VecNormalize sidecar not found: {pkl_path}; a checkpoint without its normalization "
            "statistics is not widened (the widened policy would be a different policy)"
        )
    if zip_path.stem not in HANDOFF_NAMES:
        raise WidenError(
            f"{zip_path.name} is not a handoff checkpoint ({' / '.join(HANDOFF_NAMES)}); the widened pair is "
            "written under the parent's own handoff name (D-C9), which cannot be guessed from another stem — "
            "pass the parent's robust_best_model / best_model pair"
        )
    return _ParentSource(
        model_zip=zip_path,
        vecnorm_pkl=pkl_path,
        handoff_name=zip_path.stem,
        declared_algorithm=declared,
        seed=seed,
        n_envs=n_envs,
        timesteps=timesteps,
        duration_seconds=None,
        run_id=run_id,
        stage_dir=None,
    )


def _check_max_revision_gap(max_revision_gap: Any) -> int:
    """*max_revision_gap* as a plain ``int >= 1``, or :class:`WidenError`.

    Any integral number except a bool is accepted (a ``numpy`` integer
    computed from a manifest included) and coerced to a Python ``int`` so
    the value the gate compares against is the one the JSON report records.
    """
    if (
        isinstance(max_revision_gap, bool)
        or not isinstance(max_revision_gap, numbers.Integral)
        or int(max_revision_gap) < 1
    ):
        raise WidenError(
            f"max_revision_gap must be an integer >= 1, not {max_revision_gap!r}: the tool widens forward across "
            f"at least one interface revision (the default {DEFAULT_MAX_REVISION_GAP} is exactly one, r -> r+1)"
        )
    return int(max_revision_gap)


def revision_gap_bound_text(max_revision_gap: int) -> str:
    """How far behind the current plant the gate admits a parent, in words (for refusal messages)."""
    if max_revision_gap == 1:
        return "one interface-only revision behind"
    return f"at most {max_revision_gap} interface-only revisions behind"


def identity_gate_errors(parent: Any, current: Any, *, max_revision_gap: int = DEFAULT_MAX_REVISION_GAP) -> list[str]:
    """The fields on which *parent* is not the *current* plant at most *max_revision_gap* interface-only revisions behind.

    The revision rule (D-C17) is ``1 <= current.policy_interface_revision -
    parent.policy_interface_revision <= max_revision_gap``; every other field
    (species, physics digest, ``nq``/``nv``/``nu``, ``observation_dim +
    COMMAND_WIDTH``, ``action_dim``) is checked regardless of the bound, so a
    gap above 1 can only be crossed when every intermediate bump was
    fingerprint-only.  Raises :class:`WidenError` for a bound below 1.
    """
    bound = _check_max_revision_gap(max_revision_gap)
    problems: list[str] = []
    if parent.species != current.species:
        problems.append(f"species: parent={parent.species!r}, current={current.species!r}")
    gap = int(current.policy_interface_revision) - int(parent.policy_interface_revision)
    if gap < 1:
        problems.append(
            f"policy_interface_revision: parent={parent.policy_interface_revision}, "
            f"current={current.policy_interface_revision} (gap {gap}: the tool widens forward only, "
            f"r -> r+k with 1 <= k <= max_revision_gap={bound})"
        )
    elif gap > bound:
        problems.append(
            f"policy_interface_revision: parent={parent.policy_interface_revision}, "
            f"current={current.policy_interface_revision} (gap {gap} exceeds max_revision_gap={bound}; "
            f"pass --max-revision-gap {gap} / max_revision_gap={gap} to cross {gap} bumps — opting in asserts, "
            "from plant_versions.toml's numbered notes, that every intermediate bump changed no observation "
            "layout, physics or action field; the gate still checks those fields)"
        )
    if parent.physics_sha256 != current.physics_sha256:
        problems.append(f"physics_sha256: parent={parent.physics_sha256!r}, current={current.physics_sha256!r}")
    for name in ("nq", "nv", "nu"):
        if getattr(parent, name) != getattr(current, name):
            problems.append(f"{name}: parent={getattr(parent, name)}, current={getattr(current, name)}")
    if parent.observation_dim + COMMAND_WIDTH != current.observation_dim:
        problems.append(
            f"observation_dim: parent={parent.observation_dim} + {COMMAND_WIDTH} != current={current.observation_dim}"
        )
    if parent.action_dim != current.action_dim:
        problems.append(f"action_dim: parent={parent.action_dim}, current={current.action_dim}")
    return problems


def _insert_zero_columns(tensor: Any, column: int, width: int = COMMAND_WIDTH) -> Any:
    """*tensor* (out, in) with *width* zero columns inserted before column *column*."""
    import torch

    zeros = torch.zeros((tensor.shape[0], width), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor[:, :column], zeros, tensor[:, column:]], dim=1).contiguous()


def _widen_params(
    params: dict[str, Any],
    *,
    algorithm: str,
    parent_obs: int,
    action_dim: int,
    artifact: str,
) -> tuple[dict[str, int], dict[str, int], dict[str, list[int]], dict[str, tuple[int, ...]]]:
    """Pad the observation columns of every first layer in ``params`` IN PLACE.

    Returns ``(padded, inserted_at, optimizer_members_padded, new_shapes)``:
    the column each padded tensor received its zero block at, the subset
    that are observation-action inputs (SAC critics), the optimizer state
    indices whose moments were padded per member, and the shape every
    padded tensor (and its moments) must now have — what :func:`_verify`
    asserts before it looks at the zero block.
    """
    policy = params.get("policy")
    if not isinstance(policy, dict) or not policy:
        raise WidenError(f"{artifact} carries no policy state_dict")
    old_shapes = {name: tuple(tensor.shape) for name, tensor in policy.items()}
    padded: dict[str, int] = {}
    inserted_at: dict[str, int] = {}
    for name, tensor in list(policy.items()):
        if getattr(tensor, "ndim", 0) != 2:
            continue
        in_features = int(tensor.shape[1])
        if in_features == parent_obs:
            observation_action = False
        elif in_features == parent_obs + action_dim:
            if algorithm != "sac":
                raise WidenError(
                    f"{artifact}: {name} has in_features {in_features} == observation_dim + action_dim, an "
                    f"observation-action input the {algorithm.upper()} architecture is not expected to have; "
                    "refusing to guess where the command columns belong"
                )
            observation_action = True
        else:
            continue
        if not name.endswith(".0.weight"):
            raise WidenError(
                f"{artifact}: {name} has in_features {in_features}, the parent observation width, but it is not "
                "a first layer (unexpected architecture); refusing to pad a hidden layer"
            )
        policy[name] = _insert_zero_columns(tensor, parent_obs)
        padded[name] = parent_obs
        if observation_action:
            inserted_at[name] = parent_obs
    if not padded:
        raise WidenError(f"{artifact}: no policy tensor consumes a {parent_obs}-dim observation; nothing to widen")
    new_shapes = {name: tuple(policy[name].shape) for name in padded}

    members_padded: dict[str, list[int]] = {}
    for member, prefix in _OPTIMIZER_MEMBERS[algorithm].items():
        state_dict = params.get(member)
        if not isinstance(state_dict, dict):
            raise WidenError(
                f"{artifact} carries no {member!r} member: SB3 restores every optimizer with exact_match, so an "
                "archive without it cannot be loaded, let alone widened"
            )
        names = [name for name in policy if name.startswith(prefix)]
        listed = [index for group in state_dict.get("param_groups") or [] for index in group.get("params", [])]
        if len(listed) != len(names):
            raise WidenError(
                f"{artifact}: {member} lists {len(listed)} parameters but the policy holds {len(names)} under "
                f"{prefix or 'the root'!r}; the optimizer state cannot be aligned to the weights by order"
            )
        for index, moments in (state_dict.get("state") or {}).items():
            position = int(index)
            if position < 0 or position >= len(names):
                raise WidenError(f"{artifact}: {member} state index {index!r} names no parameter")
            name = names[position]
            if not isinstance(moments, dict):
                raise WidenError(f"{artifact}: {member} state[{index}] is not an Adam moment record")
            moments_padded = False
            for key in _MOMENT_KEYS:
                moment = moments.get(key)
                if moment is None or not hasattr(moment, "shape"):
                    continue
                if tuple(moment.shape) != old_shapes[name]:
                    raise WidenError(
                        f"{artifact}: {member} state[{index}].{key} has shape {tuple(moment.shape)} but its "
                        f"parameter {name} has {old_shapes[name]}: the optimizer state does not belong to these "
                        "weights (stale moment), and padding it would corrupt the first update"
                    )
                if name in padded:
                    moments[key] = _insert_zero_columns(moment, padded[name])
                    moments_padded = True
            if name not in padded:
                continue
            # Only Adam's moments are padded; any other tensor still shaped
            # like the pre-pad weight (RMSprop ``square_avg``, SGD
            # ``momentum_buffer``, ...) would load and crash the first update
            # exactly as an unpadded Adam moment does (D-C10), so it refuses.
            for key, value in moments.items():
                if key in _MOMENT_KEYS or not hasattr(value, "shape"):
                    continue
                if tuple(value.shape) == old_shapes[name]:
                    raise WidenError(
                        f"{artifact}: {member} state[{index}].{key} is a {tuple(value.shape)} optimizer moment of "
                        f"{name} the tool does not pad (only Adam's {', '.join(_MOMENT_KEYS)} are); left at the "
                        "parent width it would crash the first update (D-C10)"
                    )
            if moments_padded:
                members_padded.setdefault(member, []).append(position)
    return padded, inserted_at, members_padded, new_shapes


def _widened_box(new_dim: int) -> Any:
    """Exactly the observation ``Box`` ``BaseDinoEnv`` builds (``check_for_correct_spaces`` compares by value)."""
    import gymnasium as gym

    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(new_dim,), dtype=np.float32)


def _load_widened_archive(
    parent_zip: Path,
    *,
    algorithm: str,
    parent_obs: int,
    action_dim: int,
    new_dim: int,
    custom_objects: "Mapping[str, Any] | None" = None,
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    """The parent archive read through SB3's serializer with its parameters widened.

    Returns ``(data, params, pytorch_variables, rewrite)`` ready for
    :func:`_save_archive`; *rewrite* names the padded tensors and columns,
    the inserted-at map, the optimizer members padded, ``num_timesteps``
    and the widened shape of every padded tensor. *custom_objects* replace
    ``data`` members before they are deserialized (SB3's ``custom_objects``):
    the parent's schedule members when they are cloudpickled bytecode, so
    the widened archive carries none of the saving interpreter's code
    (:func:`~environments.shared.policy_loading.schedule_custom_objects`).
    """
    from stable_baselines3.common.save_util import load_from_zip_file

    artifact = str(parent_zip)
    try:
        data, params, pytorch_variables = load_from_zip_file(
            artifact, device="cpu", custom_objects=dict(custom_objects) if custom_objects else None
        )
    except Exception as exc:  # noqa: BLE001 - any serializer failure is a refusal, never a partial write
        raise WidenError(f"cannot read {artifact} through the SB3 serializer: {type(exc).__name__}: {exc}") from exc
    if not isinstance(data, dict) or not params:
        raise WidenError(f"{artifact} holds no SB3 data / params members")
    saved_space = data.get("observation_space")
    saved_shape = tuple(getattr(saved_space, "shape", ()) or ())
    if saved_shape != (parent_obs,):
        raise WidenError(
            f"{artifact} saved an observation space of shape {saved_shape}, not the parent width ({parent_obs},) "
            "its plant identity records"
        )
    action_shape = tuple(getattr(data.get("action_space"), "shape", ()) or ())
    if action_shape != (action_dim,):
        raise WidenError(f"{artifact} saved an action space of shape {action_shape}, not ({action_dim},)")

    padded, inserted_at, members_padded, new_shapes = _widen_params(
        params, algorithm=algorithm, parent_obs=parent_obs, action_dim=action_dim, artifact=artifact
    )
    data["observation_space"] = _widened_box(new_dim)
    data["_last_obs"] = None
    data["_last_original_obs"] = None
    rewrite = {
        "padded_tensors": {name: list(range(column, column + COMMAND_WIDTH)) for name, column in padded.items()},
        "inserted_at": dict(inserted_at),
        "optimizer_members_padded": members_padded,
        "num_timesteps": int(data.get("num_timesteps") or 0),
        "widened_shapes": new_shapes,
    }
    return data, params, pytorch_variables, rewrite


def _parent_schedule_objects(
    parent: _ParentSource, algorithm: str, current_hyperparameters: "Mapping[str, Any] | None"
) -> "tuple[dict[str, Any], str | None]":
    """The schedule members to stamp into the widened archive in place of the parent's cloudpickled ones.

    Returns ``(replacements, source)``. The values are rebuilt through
    :func:`~environments.shared.curriculum.schedules.schedule_members_from_hyperparameters`
    from the parent stage's recorded ``"hyperparameters"`` block
    (``stage_config.json``; source ``"parent_stage_config"``) or, when the
    parent records none (a stage saved before the block existed, or the
    explicit ``--model`` / ``--vecnorm`` form), from *current_hyperparameters*,
    the current stage config's algorithm block that every child of the
    widened root trains under anyway (source ``"current_stage_config"``).
    Only members the archive stores as bytecode are replaced
    (:func:`~environments.shared.policy_loading.schedule_custom_objects`);
    ``source`` is ``None`` when nothing is.
    """
    from environments.shared.curriculum.schedules import schedule_members_from_hyperparameters
    from environments.shared.policy_loading import inspect_sb3_archive, schedule_custom_objects

    recorded: Any = None
    if parent.stage_dir is not None:
        record = _load_json(parent.stage_dir / "stage_config.json", what="stage_config.json")
        recorded = record.get("hyperparameters") if isinstance(record, Mapping) else None
    if isinstance(recorded, Mapping) and recorded:
        block: Mapping[str, Any] = recorded
        source = "parent_stage_config"
    else:
        block = current_hyperparameters or {}
        source = "current_stage_config"
    inspection = inspect_sb3_archive(parent.model_zip)
    replacements = schedule_custom_objects(
        inspection,
        algorithm,
        training_kwargs=schedule_members_from_hyperparameters(algorithm, block),
    )
    if not replacements:
        return {}, None
    logger.info(
        "%s stores %s as cloudpickled bytecode (saving Python: %s); the widened archive re-states them from the %s "
        "as %s",
        parent.model_zip,
        ", ".join(sorted(replacements)),
        inspection.saved_python_text,
        "parent's recorded hyperparameters" if source == "parent_stage_config" else "current stage config",
        {key: repr(value) for key, value in sorted(replacements.items())},
    )
    return replacements, source


def _save_archive(
    out_zip: Path, data: dict[str, Any], params: dict[str, Any], pytorch_variables: Any, attributes: Mapping[str, Any]
) -> None:
    """Stamp *attributes* into ``data`` and write the archive through SB3's serializer."""
    from stable_baselines3.common.save_util import save_to_zip_file

    for key, value in attributes.items():
        data[key] = value
    save_to_zip_file(str(out_zip), data=data, params=params, pytorch_variables=pytorch_variables)


def _widen_sidecar(
    parent_pkl: Path,
    out_pkl: Path,
    *,
    species: str,
    stage: "int | str",
    parent_obs: int,
    new_dim: int,
    current: Any,
    parent_identity: Any,
    allow_legacy_plant: bool,
) -> None:
    """Rewrite the parent VecNormalize sidecar into *out_pkl* (amendment A13a)."""
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from environments.shared.config import build_env
    from environments.shared.plant_contract import (
        MODEL_IDENTITY_ATTRIBUTE,
        PlantCompatibilityError,
        PlantIdentity,
        attach_plant_identity,
        validate_recorded_identity,
    )

    try:
        with open(parent_pkl, "rb") as handle:
            vecnorm = pickle.load(handle)
    except Exception as exc:  # noqa: BLE001 - the message matters more than the type
        raise WidenError(f"cannot read VecNormalize statistics from {parent_pkl}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(vecnorm, VecNormalize):
        raise WidenError(f"{parent_pkl} does not hold a VecNormalize wrapper ({type(vecnorm).__name__})")
    # The identity is read off the raw object's ``__dict__``: a VecNormalize
    # unpickled without ``set_venv`` has no ``class_attributes`` yet, so a
    # plain ``getattr`` miss recurses through ``VecEnvWrapper.__getattr__``
    # instead of raising AttributeError.
    recorded = vecnorm.__dict__.get(MODEL_IDENTITY_ATTRIBUTE)
    if recorded is not None and not isinstance(recorded, Mapping):
        raise WidenError(f"{parent_pkl} contains invalid plant identity metadata of type {type(recorded).__name__}")
    if parent_identity is not None:
        try:
            validate_recorded_identity(
                recorded, parent_identity, artifact=str(parent_pkl), allow_legacy=allow_legacy_plant
            )
        except PlantCompatibilityError as exc:
            raise WidenError(f"the sidecar does not record the parent archive's plant: {exc}") from exc
    elif recorded is not None:
        # A legacy archive gives the sidecar's identity nothing to validate
        # against; the species and the width it records must still be the
        # parent's.
        try:
            sidecar_identity = PlantIdentity.from_mapping(recorded)
        except PlantCompatibilityError as exc:
            raise WidenError(f"{parent_pkl}: {exc}") from exc
        if sidecar_identity.species != species or int(sidecar_identity.observation_dim) != parent_obs:
            raise WidenError(
                f"{parent_pkl} records a {sidecar_identity.species} plant of observation_dim "
                f"{sidecar_identity.observation_dim}, not the {species} parent's {parent_obs}; the sidecar "
                "does not belong to the archive being widened"
            )
    obs_rms = getattr(vecnorm, "obs_rms", None)
    mean = getattr(obs_rms, "mean", None)
    if isinstance(obs_rms, dict) or not isinstance(mean, np.ndarray):
        raise WidenError(f"{parent_pkl} holds no Box observation statistics to widen")
    if mean.shape != (parent_obs,):
        raise WidenError(f"{parent_pkl} normalises {mean.shape} observations, not the parent width ({parent_obs},)")
    vecnorm.obs_rms = pad_running_stats(obs_rms, COMMAND_WIDTH)
    vecnorm.observation_space = _widened_box(new_dim)
    attach_plant_identity(vecnorm, current)
    venv = DummyVecEnv([lambda: build_env(species, stage)])
    try:
        vecnorm.set_venv(venv)
        vecnorm.old_obs = np.zeros((vecnorm.num_envs, new_dim), dtype=np.float32)
        vecnorm.training = False
        vecnorm.save(str(out_pkl))
    finally:
        venv.close()


def _verify(
    *,
    species: str,
    stage: "int | str",
    algorithm: str,
    parent_zip: Path,
    parent_pkl: Path,
    widened_zip: Path,
    widened_pkl: Path,
    current: Any,
    padded_tensors: Mapping[str, list[int]],
    optimizer_members_padded: Mapping[str, list[int]],
    widened_shapes: Mapping[str, tuple[int, ...]],
) -> dict[str, Any]:
    """The self-verification pass (plan §4.6 pins; amendment A13b); read-only."""
    import torch
    from stable_baselines3.common.save_util import load_from_zip_file

    from environments.shared.config import build_env
    from environments.shared.plant_contract import validate_model_plant
    from environments.shared.policy_loading import load_sb3_checkpoint, load_sb3_model

    # (a) the padded columns are exactly zero — a weight property, pinned
    # exact.  Each width is asserted BEFORE the block is sliced: a slice past
    # the end of a tensor that was never widened is empty, and an empty
    # block equals its zeros vacuously.
    _, params, _ = load_from_zip_file(str(widened_zip), device="cpu", load_data=False)
    max_padded_abs = 0.0
    for name, columns in padded_tensors.items():
        shape = tuple(params["policy"][name].shape)
        if shape != tuple(widened_shapes[name]):
            raise WidenError(f"widened {name} has shape {shape}, not the widened {tuple(widened_shapes[name])}")
        block = params["policy"][name][:, columns[0] : columns[-1] + 1]
        max_padded_abs = max(max_padded_abs, float(block.abs().max()) if block.numel() else 0.0)
        if not torch.equal(block, torch.zeros_like(block)):
            raise WidenError(f"widened {name} columns {columns} are not exactly zero (max |w| = {max_padded_abs})")
    prefixes = _OPTIMIZER_MEMBERS[algorithm]
    for member, indices in optimizer_members_padded.items():
        names = [name for name in params["policy"] if name.startswith(prefixes[member])]
        for index in indices:
            columns = padded_tensors[names[index]]
            expected = tuple(widened_shapes[names[index]])
            for key, moment in params[member]["state"][index].items():
                if getattr(moment, "ndim", 0) != 2:
                    continue
                if tuple(moment.shape) != expected:
                    raise WidenError(
                        f"widened {member} state[{index}].{key} has shape {tuple(moment.shape)}, not its "
                        f"parameter's widened {expected}; the first update would crash (D-C10)"
                    )
                block = moment[:, columns[0] : columns[-1] + 1]
                if not torch.equal(block, torch.zeros_like(block)):
                    raise WidenError(f"widened {member} state[{index}].{key} columns {columns} are not exactly zero")

    # (c) both widened artifacts validate against the current plant.
    parent_model = load_sb3_model(str(parent_zip), algorithm=algorithm, device="cpu")
    with open(parent_pkl, "rb") as handle:
        parent_norm = pickle.load(handle)
    model, normalizer, _ = load_sb3_checkpoint(
        str(widened_zip),
        str(widened_pkl),
        env_factory=lambda: build_env(species, stage),
        guess_sidecar=False,
        plant_identity=current,
    )
    if normalizer is None:
        raise WidenError(f"{widened_pkl} did not load as the widened normalizer")
    validate_model_plant(model, current, artifact=str(widened_zip))
    validate_model_plant(normalizer, current, artifact=str(widened_pkl))

    # (b) actions equal the parent's over a seeded rollout of the real env,
    # with the command slice zero and with the probe vector in it.
    env = build_env(species, stage)
    probe = np.asarray(COMMAND_PROBE_VECTOR, dtype=np.float32)
    delta_zero = 0.0
    delta_probe = 0.0
    try:
        if model.observation_space != env.observation_space:
            raise WidenError(
                f"the widened archive's observation space {model.observation_space} is not the environment's "
                f"{env.observation_space}"
            )
        obs, _ = env.reset(seed=VERIFICATION_ROLLOUT_SEED)
        for _ in range(VERIFICATION_ROLLOUT_STEPS):
            observation = np.asarray(obs, dtype=np.float32)
            parent_action, _ = parent_model.predict(
                parent_norm.normalize_obs(observation[:-COMMAND_WIDTH]), deterministic=True
            )
            widened_action, _ = model.predict(normalizer.normalize_obs(observation), deterministic=True)
            probed = observation.copy()
            probed[-COMMAND_WIDTH:] = probe
            probed_action, _ = model.predict(normalizer.normalize_obs(probed), deterministic=True)
            delta_zero = max(delta_zero, float(np.max(np.abs(widened_action - parent_action))))
            delta_probe = max(delta_probe, float(np.max(np.abs(probed_action - parent_action))))
            obs, _, terminated, truncated, _ = env.step(parent_action)
            if terminated or truncated:
                obs, _ = env.reset()
    finally:
        env.close()
        try:
            normalizer.close()
        except Exception:  # noqa: BLE001 - a throwaway venv that will not close is not a verification failure
            pass
    if not (delta_zero <= ACTION_DELTA_ATOL and delta_probe <= ACTION_DELTA_ATOL):
        raise WidenError(
            f"the widened policy's actions differ from the parent's over the seeded rollout "
            f"(max |delta| {delta_zero} on zero command, {delta_probe} on the probe command; tolerance "
            f"{ACTION_DELTA_ATOL}); exact transfer does not hold"
        )
    return {
        "padded_columns_exactly_zero": True,
        "max_padded_column_abs": max_padded_abs,
        "max_action_delta_zero_command": delta_zero,
        "max_action_delta_probe_command": delta_probe,
        "action_delta_atol": ACTION_DELTA_ATOL,
        "rollout": {"seed": VERIFICATION_ROLLOUT_SEED, "steps": VERIFICATION_ROLLOUT_STEPS, "deterministic": True},
        "plant_identity_validated": True,
    }


def _remove_partial_output(target: Path, created_ancestors: list[Path]) -> None:
    """Remove *target* and every ancestor the tool created, while each is still empty."""
    shutil.rmtree(target, ignore_errors=True)
    for path in created_ancestors:
        try:
            path.rmdir()
        except OSError:
            break


def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in ("mujoco", "stable_baselines3", "torch", "gymnasium"):
        try:
            module = __import__(name)
        except ImportError:
            versions[name] = None
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def widen_checkpoint(
    *,
    species: str,
    stage: "int | str",
    target_stage_dir: "str | Path",
    parent_stage_dir: "str | Path | None" = None,
    model_zip: "str | Path | None" = None,
    vecnorm_pkl: "str | Path | None" = None,
    algorithm: str | None = None,
    label: str | None = None,
    allow_legacy_plant: bool = False,
    parent_run_id: str | None = None,
    seed: Any = None,
    n_envs: Any = None,
    timesteps: Any = None,
    max_revision_gap: int = DEFAULT_MAX_REVISION_GAP,
) -> WidenResult:
    """Widen one SB3 checkpoint pair r -> r+k (``1 <= k <= max_revision_gap``) into *target_stage_dir*.

    The parent is *parent_stage_dir* (its ``select_handoff_checkpoint`` pair
    and the run facts its ``stage_config.json`` records) or the explicit
    *model_zip* / *vecnorm_pkl* pair, for which *algorithm*, *seed*,
    *n_envs* and *timesteps* are required.  *stage* is any reference the
    species' manifest resolves; the widened artifacts are stamped with that
    stage's CURRENT task fingerprint.  *max_revision_gap* (default 1, an
    ``int >= 1``; D-C17) bounds how many interface revisions behind the
    parent may be — see :func:`identity_gate_errors`.  Raises
    :class:`WidenError` on every refusal (a bound below 1 included) and on
    any self-verification failure (the target directory is removed first).
    See the module docstring.
    """
    from environments.shared.config import (
        WIDEN_LINEAGE_KEYS,
        StageDirectoryOccupiedError,
        get_git_commit,
        load_stage_config,
        refuse_occupied_stage_dir,
        save_stage_config,
    )
    from environments.shared.file_io import atomic_write_text
    from environments.shared.plant_contract import (
        MODEL_IDENTITY_ATTRIBUTE,
        PlantCompatibilityError,
        PlantContractError,
        PlantIdentity,
        current_plant_identity,
    )
    from environments.shared.result_bundle import sha256_file
    from environments.shared.species_registry import get_species_config
    from environments.shared.stage_manifest import (
        StageManifestError,
        load_stage_manifest,
        stage_dir_candidates,
        stage_label,
    )
    from environments.shared.task_fingerprint import (
        MODEL_TASK_ATTRIBUTE,
        derive_stage_task_fingerprint,
        read_checkpoint_attribute,
        read_checkpoint_task_fingerprint,
    )

    max_revision_gap = _check_max_revision_gap(max_revision_gap)
    try:
        entry = load_stage_manifest(species).resolve(stage)
    except (StageManifestError, TypeError, ValueError) as exc:
        raise WidenError(f"cannot resolve {species} stage {stage!r}: {exc}") from exc
    reference = entry.reference
    label_prefix = stage_label(reference)
    try:
        current = current_plant_identity(species)
    except PlantContractError as exc:
        raise WidenError(f"the current {species} plant identity is unavailable: {exc}") from exc

    parent = _resolve_parent(
        parent_stage_dir=parent_stage_dir,
        model_zip=model_zip,
        vecnorm_pkl=vecnorm_pkl,
        algorithm=algorithm,
        seed=seed,
        n_envs=n_envs,
        timesteps=timesteps,
        parent_run_id=parent_run_id,
    )
    sniffed = sniff_archive_algorithm(parent.model_zip)
    if parent.declared_algorithm is not None and parent.declared_algorithm != sniffed:
        raise WidenError(
            f"{parent.model_zip} is a {sniffed.upper()} archive, not the {parent.declared_algorithm.upper()} declared"
        )
    resolved_algorithm = sniffed

    raw_identity = read_checkpoint_attribute(parent.model_zip, MODEL_IDENTITY_ATTRIBUTE)
    parent_identity: Any = None
    if raw_identity is None:
        if not allow_legacy_plant:
            raise WidenError(
                f"{parent.model_zip} has no plant identity; it predates the plant-safety contract and its "
                "interface revision cannot be verified. Pass --allow-legacy-plant only for a deliberate widening "
                "of a historical checkpoint whose width is read from its saved observation space."
            )
        logger.warning("%s has no plant identity; widening under --allow-legacy-plant", parent.model_zip)
    else:
        if not isinstance(raw_identity, Mapping):
            raise WidenError(f"{parent.model_zip} contains invalid plant identity metadata")
        try:
            parent_identity = PlantIdentity.from_mapping(raw_identity)
        except PlantCompatibilityError as exc:
            raise WidenError(f"{parent.model_zip}: {exc}") from exc
        problems = identity_gate_errors(parent_identity, current, max_revision_gap=max_revision_gap)
        if problems:
            raise WidenError(
                f"{parent.model_zip} is not the current {species} plant "
                f"{revision_gap_bound_text(max_revision_gap)}:\n- " + "\n- ".join(problems)
            )
    revision_gap: int | None = (
        int(current.policy_interface_revision) - int(parent_identity.policy_interface_revision)
        if parent_identity is not None
        else None
    )
    new_dim = int(current.observation_dim)
    parent_obs = int(parent_identity.observation_dim) if parent_identity is not None else new_dim - COMMAND_WIDTH
    action_dim = int(current.action_dim)

    target = Path(target_stage_dir)
    # The widened node is located inside its run the way every reader
    # locates a stage (``stage_dir_candidates``): a target under any other
    # name would be written in full and then invisible to the reuse rule and
    # the chain loop.
    candidates = stage_dir_candidates(species, reference)
    if target.name not in candidates:
        raise WidenError(
            f"{target} is not a {species} {entry.id} stage directory (its name must be one of "
            f"{' / '.join(candidates)}, the names every reader locates the stage under)"
        )
    # A parent run is a published result bundle whose every file is a
    # declared artifact; never write the widened node inside it (nor swallow
    # the parent inside the target).
    parent_roots = (
        (parent.stage_dir.parent,)
        if parent.stage_dir is not None
        else (parent.model_zip.parent, parent.vecnorm_pkl.parent)
    )
    resolved_target = target.resolve()
    for root in parent_roots:
        resolved_root = root.resolve()
        if resolved_target.is_relative_to(resolved_root) or resolved_root.is_relative_to(resolved_target):
            raise WidenError(
                f"{target} lies inside the parent's directory {root} (or contains it); a widened root is written "
                "into a fresh run directory of its own, never beside the parent's certified pair"
            )
    try:
        refuse_occupied_stage_dir(target, task_load_mode=None)
    except StageDirectoryOccupiedError as exc:
        raise WidenError(str(exc)) from exc
    if target.exists():
        if not target.is_dir():
            raise WidenError(f"{target} exists and is not a directory")
        present = sorted(path.name for path in target.iterdir())
        if present:
            raise WidenError(
                f"{target} is not empty ({', '.join(present[:8])}{', ...' if len(present) > 8 else ''}); "
                "a widened root is written only into a fresh stage directory"
            )

    stage_config = load_stage_config(species, reference)
    task_fingerprint = derive_stage_task_fingerprint(
        species=species,
        stage=reference,
        backend=FINGERPRINT_BACKEND,
        env_kwargs=stage_config.get("env_kwargs", {}),
        plant_identity=current.to_dict(),
    )
    parent_task = read_checkpoint_task_fingerprint(parent.model_zip)
    parent_task_sha256 = parent_task.get("task_sha256") if parent_task else None
    parent_checkpoint_sha256 = sha256_file(parent.model_zip)
    parent_normalization_sha256 = sha256_file(parent.vecnorm_pkl)
    commit = get_git_commit()
    widened_by = f"{WIDEN_TOOL_VERSION}@{commit}"

    lineage: dict[str, Any] = {
        "schema": WIDEN_LINEAGE_SCHEMA,
        "tool": WIDEN_TOOL_VERSION,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "parent_normalization_sha256": parent_normalization_sha256,
        "parent_task_sha256": parent_task_sha256,
        "parent_plant_identity": parent_identity.to_dict() if parent_identity is not None else None,
        "parent_path": str(parent.model_zip),
        "from_observation_dim": parent_obs,
        "to_observation_dim": new_dim,
        "revision_gap": revision_gap,
        "padded_tensors": [],
        "inserted_at": {},
        "widened_at_commit": commit,
    }
    run_lineage: dict[str, Any] = {
        "widened_from_path": str(parent.model_zip),
        "widened_from_checkpoint_sha256": parent_checkpoint_sha256,
        "widened_from_normalization_sha256": parent_normalization_sha256,
        "widened_from_task_sha256": parent_task_sha256,
        "widened_from_policy_interface_sha256": (
            parent_identity.policy_interface_sha256 if parent_identity is not None else None
        ),
        "widened_from_policy_interface_revision": (
            parent_identity.policy_interface_revision if parent_identity is not None else None
        ),
        "widened_from_run_id": parent.run_id,
        "widened_by": widened_by,
    }
    assert tuple(run_lineage) == tuple(WIDEN_LINEAGE_KEYS)

    models_dir = target / "models"
    widened_zip = models_dir / f"{parent.handoff_name}.zip"
    widened_pkl = models_dir / f"{parent.handoff_name}_vecnorm.pkl"
    final_zip = models_dir / f"{label_prefix}_final.zip"
    final_pkl = models_dir / f"{label_prefix}_final_vecnorm.pkl"
    # Every directory above the target that the tool creates (a fresh run
    # directory, typically) is removed again on failure — deepest first,
    # only while still empty — so a refusal leaves exactly what it found.
    created_ancestors = [path for path in target.parents if not path.exists()]
    models_dir.mkdir(parents=True, exist_ok=True)

    try:
        # The archive first: its padded-tensor names complete the lineage
        # record stamped into it before it is written. The parent's schedule
        # members are re-stated from its recorded hyperparameters block (or
        # inference defaults) whenever the archive stores them as bytecode:
        # cloudpickled closures run only on the interpreter that saved them,
        # and a widened archive must load on whatever image judges it.
        schedule_objects, schedule_source = _parent_schedule_objects(
            parent, resolved_algorithm, stage_config.get(f"{resolved_algorithm}_kwargs")
        )
        data, params, pytorch_variables, rewrite = _load_widened_archive(
            parent.model_zip,
            algorithm=resolved_algorithm,
            parent_obs=parent_obs,
            action_dim=action_dim,
            new_dim=new_dim,
            custom_objects=schedule_objects,
        )
        lineage["padded_tensors"] = list(rewrite["padded_tensors"])
        lineage["inserted_at"] = dict(rewrite["inserted_at"])
        _save_archive(
            widened_zip,
            data,
            params,
            pytorch_variables,
            {
                MODEL_IDENTITY_ATTRIBUTE: current.to_dict(),
                MODEL_TASK_ATTRIBUTE: dict(task_fingerprint),
                MODEL_WIDEN_LINEAGE_ATTRIBUTE: lineage,
            },
        )
        del data, params, pytorch_variables
        _widen_sidecar(
            parent.vecnorm_pkl,
            widened_pkl,
            species=species,
            stage=reference,
            parent_obs=parent_obs,
            new_dim=new_dim,
            current=current,
            parent_identity=parent_identity,
            allow_legacy_plant=allow_legacy_plant,
        )
        shutil.copyfile(widened_zip, final_zip)
        shutil.copyfile(widened_pkl, final_pkl)
        widened_checkpoint_sha256 = sha256_file(widened_zip)
        widened_normalization_sha256 = sha256_file(widened_pkl)

        verification = _verify(
            species=species,
            stage=reference,
            algorithm=resolved_algorithm,
            parent_zip=parent.model_zip,
            parent_pkl=parent.vecnorm_pkl,
            widened_zip=widened_zip,
            widened_pkl=widened_pkl,
            current=current,
            padded_tensors=rewrite["padded_tensors"],
            optimizer_members_padded=rewrite["optimizer_members_padded"],
            widened_shapes=rewrite["widened_shapes"],
        )
        if (
            sha256_file(widened_zip) != widened_checkpoint_sha256
            or sha256_file(widened_pkl) != widened_normalization_sha256
        ):
            raise WidenError("the verification pass changed the widened files; the widened pair is not stable")
        if (
            sha256_file(final_zip) != widened_checkpoint_sha256
            or sha256_file(final_pkl) != widened_normalization_sha256
        ):
            raise WidenError("the <stage_label>_final copies are not byte-identical to the handoff pair")
        verification["hashes_stable_after_verification"] = True

        extra: dict[str, Any] = {"seed": parent.seed, "n_envs": parent.n_envs, "timesteps": parent.timesteps}
        if parent.duration_seconds is not None:
            extra["duration_seconds"] = parent.duration_seconds
        extra.update(run_lineage)
        stage_config_path = save_stage_config(
            target,
            reference,
            stage_config,
            resolved_algorithm.upper(),
            extra=extra,
            env_class=get_species_config(species).env_class,
            species=species,
            plant_identity=current,
            task_fingerprint=task_fingerprint,
            load_path=None,
            load_mode=None,
            parent_run_id=None,
            label=label,
        )
        report: dict[str, Any] = {
            "schema": WIDEN_REPORT_SCHEMA,
            "tool": WIDEN_TOOL_VERSION,
            "species": species,
            "stage": reference,
            "stage_id": entry.id,
            "algorithm": resolved_algorithm,
            "handoff_name": parent.handoff_name,
            "parent": {
                "stage_dir": str(parent.stage_dir) if parent.stage_dir is not None else None,
                "run_id": parent.run_id,
                "model_path": str(parent.model_zip),
                "normalization_path": str(parent.vecnorm_pkl),
                "checkpoint_sha256": parent_checkpoint_sha256,
                "normalization_sha256": parent_normalization_sha256,
                "task_sha256": parent_task_sha256,
                "policy_interface_revision": run_lineage["widened_from_policy_interface_revision"],
                "policy_interface_sha256": run_lineage["widened_from_policy_interface_sha256"],
                "legacy_plant": parent_identity is None,
            },
            "widened": {
                "model_path": str(widened_zip),
                "normalization_path": str(widened_pkl),
                "final_model_path": str(final_zip),
                "final_normalization_path": str(final_pkl),
                "checkpoint_sha256": widened_checkpoint_sha256,
                "normalization_sha256": widened_normalization_sha256,
                "task_sha256": task_fingerprint.get("task_sha256"),
                "policy_interface_revision": current.policy_interface_revision,
            },
            "from_observation_dim": parent_obs,
            "to_observation_dim": new_dim,
            "revision_gap": revision_gap,
            "max_revision_gap": max_revision_gap,
            "command_width": COMMAND_WIDTH,
            "padded_tensors": rewrite["padded_tensors"],
            "inserted_at": rewrite["inserted_at"],
            "optimizer_members_padded": rewrite["optimizer_members_padded"],
            "num_timesteps": rewrite["num_timesteps"],
            "widened_at_commit": commit,
            "widened_by": widened_by,
            # The parent's schedule members that were cloudpickled bytecode and
            # were re-stated so the widened archive carries none (empty when the
            # parent stored them by reference), and where the values came from:
            # the parent's recorded hyperparameters block, else the current
            # stage config's algorithm block.
            "schedule_members_restated": {key: repr(value) for key, value in sorted(schedule_objects.items())},
            "schedule_members_source": schedule_source,
            "versions": _library_versions(),
            **verification,
        }
        report_path = target / WIDEN_REPORT_FILENAME
        atomic_write_text(report_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
        leftovers = [name for name in FORBIDDEN_OUTPUT_FILES if (target / name).exists()]
        if leftovers:
            raise WidenError(f"{target} unexpectedly holds {leftovers}")
    except WidenError:
        _remove_partial_output(target, created_ancestors)
        raise
    except Exception as exc:  # noqa: BLE001 - every failure after the first write removes the partial directory
        _remove_partial_output(target, created_ancestors)
        raise WidenError(f"widening {parent.model_zip} failed: {type(exc).__name__}: {exc}") from exc

    logger.info(
        "Widened %s (%s, %d -> %d dims, revision gap %s) into %s; max action delta %.3g (zero command) / %.3g "
        "(probe command)",
        parent.model_zip,
        resolved_algorithm.upper(),
        parent_obs,
        new_dim,
        revision_gap,
        target,
        verification["max_action_delta_zero_command"],
        verification["max_action_delta_probe_command"],
    )
    return WidenResult(
        target_stage_dir=target,
        handoff_name=parent.handoff_name,
        algorithm=resolved_algorithm,
        model_zip=widened_zip,
        vecnorm_pkl=widened_pkl,
        final_zip=final_zip,
        final_vecnorm_pkl=final_pkl,
        stage_config_path=stage_config_path,
        report_path=report_path,
        parent_checkpoint_sha256=parent_checkpoint_sha256,
        parent_normalization_sha256=parent_normalization_sha256,
        widened_checkpoint_sha256=widened_checkpoint_sha256,
        widened_normalization_sha256=widened_normalization_sha256,
        from_observation_dim=parent_obs,
        to_observation_dim=new_dim,
        revision_gap=revision_gap,
        max_action_delta_zero_command=verification["max_action_delta_zero_command"],
        max_action_delta_probe_command=verification["max_action_delta_probe_command"],
        report=report,
    )


def main(argv: "list[str] | None" = None) -> int:
    from environments.shared.config import SPECIES_NAMES

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    parser.add_argument("--species", required=True, choices=SPECIES_NAMES)
    parser.add_argument("--stage", required=True, help="the stage reference (a legacy number or a stage id)")
    parser.add_argument(
        "--to-stage-dir",
        required=True,
        help=(
            "the fresh stage directory to write the widened pair into; for the SB3 notebook to judge it, "
            "<LOG_BASE>/<species>/<algo>/<new run id>/<stage_dirname(species, root)> (01_stance) in a run the "
            "notebook has not opened yet, <new run id> being a new timestamp id (YYYYMMDD_HHMMSS, the format the "
            'storage cell mints) that no run uses yet, then the notebook with RUN_ID = "<new run id>", SEED = the '
            'parent\'s run seed and TRUNK_FROM = "" (decision D-D14; the module docstring has the Colab steps)'
        ),
    )
    parser.add_argument(
        "--from-stage-dir", default=None, help="the parent stage directory (its handoff pair and run block)"
    )
    parser.add_argument("--model", default=None, help="explicit parent checkpoint .zip (with --vecnorm)")
    parser.add_argument("--vecnorm", default=None, help="explicit parent VecNormalize .pkl (with --model)")
    parser.add_argument(
        "--algorithm", default=None, choices=ALGORITHMS, help="required with --model; checked against the archive"
    )
    parser.add_argument("--seed", type=int, default=None, help="the parent run's seed (explicit-pair form)")
    parser.add_argument("--n-envs", type=int, default=None, help="the parent run's n_envs (explicit-pair form)")
    parser.add_argument("--timesteps", type=int, default=None, help="the parent run's timesteps (explicit-pair form)")
    parser.add_argument("--label", default=None, help="free text recorded as the run block's label")
    parser.add_argument(
        "--parent-run-id", default=None, help="override the parent run id recorded as widened_from_run_id"
    )
    parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="widen a parent that carries no plant identity (its width is read from its saved observation space)",
    )
    parser.add_argument(
        "--max-revision-gap",
        type=int,
        default=DEFAULT_MAX_REVISION_GAP,
        metavar="N",
        help=(
            "how many interface revisions behind the parent may be (default 1: exactly r -> r+1; D-C17). "
            "N > 1 asserts, from plant_versions.toml's numbered notes, that every intermediate bump was "
            "fingerprint-only; the gate still checks every width and physics field"
        ),
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.max_revision_gap < 1:
        logger.error(
            "Refusing to widen: --max-revision-gap must be >= 1, not %d (the default %d widens across exactly one "
            "interface revision, r -> r+1)",
            args.max_revision_gap,
            DEFAULT_MAX_REVISION_GAP,
        )
        return 1
    stage: "int | str" = int(args.stage) if args.stage.isdigit() else args.stage
    try:
        result = widen_checkpoint(
            species=args.species,
            stage=stage,
            target_stage_dir=args.to_stage_dir,
            parent_stage_dir=args.from_stage_dir,
            model_zip=args.model,
            vecnorm_pkl=args.vecnorm,
            algorithm=args.algorithm,
            label=args.label,
            allow_legacy_plant=args.allow_legacy_plant,
            parent_run_id=args.parent_run_id,
            seed=args.seed,
            n_envs=args.n_envs,
            timesteps=args.timesteps,
            max_revision_gap=args.max_revision_gap,
        )
    except WidenError as exc:
        logger.error("Refusing to widen: %s", exc)
        return 1
    print(json.dumps(result.report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
