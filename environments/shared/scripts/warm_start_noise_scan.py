"""Scan how much policy noise a stance parent tolerates in its walk stage's env, before a walk warm-starts from it.

Report only.  A walk stage (the manifest node with recipe ``walk``,
``locomotion`` on every committed model) warm-starts PPO from its parent's
handoff checkpoint and inherits the parent's learned action std.  Too little
noise and the walk explores nothing: the T. rex's walking pilot 1 read STILL
on 3 of 3 seeds from a parent of std 0.095.  Too much and the parent falls at
once: walking pilot 3's parent stood in 9 of 120 episodes at a uniform
``log_std`` of -1.5.  Pilot 3 chose its reset of -2.0 from a scan like this
one, on an r10 T. rex parent (``evidence/overnight/logstd/RESULT.md`` on the
unmerged branch ``claude/handoff-walking-pilot3``), and
``docs/NEXT_STEPS.md`` prescribes the same scan for the velociraptor before
its first walk.  This tool runs it for any of the seven models (the
maintainer's decision of 2026-11-26).  What it measures is a starting point
for ``[curriculum] warm_start_log_std``
(``train_base._apply_warm_start_log_std``), which no committed walk stage
declares.  It sets nothing: whether to declare a value is the maintainer's
decision.

Protocol:

* **Parent.**  ``--stage-dir`` names the parent's stage directory (for
  example ``RUN/01_stance``; its name says which stage it is), and the
  checkpoint pair is the stage's handoff, the pair
  ``curriculum.checkpoints.select_handoff_checkpoint`` picks in its
  ``models/`` directory (``robust_best_model``, else ``best_model``).  Or
  ``--model`` and ``--vecnorm`` name a pair directly (a checkpoint stem
  resolves to ``<stem>.zip``, as SB3 resolves it), and the parent is the
  stance, or the declared parent of ``--walk-stage``.  A missing file is
  refused by name.  The pair loads through
  ``policy_loading.load_sb3_checkpoint`` (through
  ``reporting.episode_trace.load_trace_policy``), checked against the
  model's current plant as it loads; ``--allow-legacy-plant`` admits a
  checkpoint that predates the plant contract, as training's flag does, and
  the JSON then records ``checkpoint_plant_validated`` false.  Its recorded
  task must be the walk's declared parent
  (``task_fingerprint.validate_declared_parent``, the refusal training
  makes).  The policy must be a diagonal Gaussian with one state-independent
  ``log_std`` per actuator (no gSDE, no SAC), the policy a warm-start reset
  can apply to.  A critic-phase or mirror-KL parent is such a policy.
* **Walk stage.**  By default this is the manifest node with recipe ``walk``
  that warm-starts from the parent (``locomotion``).  ``--walk-stage``
  names another child of the parent.  A model with no such node is refused,
  and so is a walk stage with a live ``command_mode``: training reseeds that
  stage's command statistics at the warm start, and this scan does not.
* **Env.**  The env is built from the walk stage's committed ``[env]`` keys
  through ``config.build_env``, the construction path the other report
  tools share, and checked against the current plant as ``make_env`` checks
  a training env (``validate_environment_plant``).  The forward-velocity
  weight is set to the reward ramp's start value
  (``train_base._stage_entry_shaping_callbacks``): that is the weight at
  the warm start.  The reward weight changes the return only, never a
  termination, except the allosaurus's course completion, which needs a
  weight above 0, as it is all along the ramp.  The parent's VecNormalize
  statistics are frozen: observations are normalized as
  ``VecNormalize.normalize_obs`` does and cast to float32, and nothing
  updates the statistics.
* **Noise settings.**  Two reference settings come first.  ``mean`` sends
  the policy's mean action with no noise (pilot 3's deterministic parent):
  it rolls the seeds once, since the noise streams do not apply.  ``own``
  samples at the parent's learned ``log_std``, unchanged, as a walk without
  a reset starts.  Then each value of the grid is written into every entry
  of the policy's ``log_std``, in memory only, and restored afterwards (an
  exception included).  The checkpoint files are only read.  The default
  grid has two passes, as pilot 3's scan had.  The first is centred on the
  parent's largest learned ``log_std``, rounded to the nearest multiple of
  0.125, and runs from 0.5 below it to 1.0 above it in steps of 0.25: 7
  values.  The second adds one value 0.125 above the largest first-pass
  value at which every episode stood, halving the gap to the next value up.
  It adds none when no first-pass value stood every episode, or when the
  largest did.  Every default value is a multiple of 0.125, a number a stage
  TOML can declare.  On pilot 3's parent (largest learned log_std -2.20,
  tail_1_yaw) the first pass is -2.75 to -1.25, which holds pilot 3's own
  grid (-2.5 to -1.5 in steps of 0.25); pilot 3's -2.0 stood and -1.75 fell,
  so the second pass adds -1.875, the value pilot 3 added.  ``--log-std V
  [V ...]`` replaces the grid, and is rolled as given, in one pass.
* **Episodes.**  Each noisy setting rolls the reset seeds 3042-3081
  (``--seed`` and ``--episodes``, or ``--seeds``) in three independent noise
  streams (``--streams``): 120 episodes per value.  In stream ``k`` the
  episode of seed ``s`` draws its noise from
  ``numpy.random.default_rng(SeedSequence([s, 20261126, k]))``.  Every
  setting uses the same draws, so the settings are compared on common random
  numbers, as pilot 3's numpy streams were.  Each step sends
  ``clip(mean + std * z, low, high)``, with the mean and std read off the
  policy's own action distribution (``policy.get_distribution``, which
  builds its std from ``log_std``) and the clip at the action bounds.  That
  is PPO's sampling in its rollouts.  The horizon is the walk stage's own
  episode length (``max_episode_steps``): 1000 control steps on six of the
  seven models (10 s at a 0.01 s control step, as pilot 3's protocol on
  the T. rex; 20 s on the two Compsognathus models at 0.02 s) and 3000 on
  the allosaurus.  ``--horizon N`` rolls fewer steps, and the
  recommendation line then says the panel was shortened.
* **Stands.**  An episode stands when it reaches the horizon and the env
  reported no termination on any step (``terminated`` from ``env.step``,
  with ``info["termination_reason"]`` naming the cause).  A success
  termination, such as ``target_reached``, also ends an episode early and
  counts as not standing; the table lists every cause.

Readouts per setting, all cheap:

* the stand count per stream and in total, the causes of the early ends,
  the mean episode length and the mean return (over the episodes whose
  return is finite: an episode ending in ``nonfinite_state``, which the
  allosaurus and both Compsognathus envs report, records ``None`` for a
  non-finite return or displacement, and the row counts it);
* lift-offs, over the standing episodes only.  A foot is down on a step
  when its touch force exceeds 0.1 N (``CONTACT_THRESHOLD_N``) on at least
  half of the step's substeps (the env's own per-substep touch block).  The
  down state is debounced over 20 ms (``gait.events.debounce``, the
  repository's gait rule, which is set in seconds: 2 steps at a control
  step of 0.01 s, but 1 step, which is no debounce at all, at the
  Compsognathus models' 0.02 s), and a lift-off is a down-to-up edge after
  the 0.1 s spawn grace (``SPAWN_GRACE_S``: the reset transient is the
  plant's, not the policy's).  The JSON records the debounce and the grace
  in steps.  These are the touch sensors, not the floor-truth contacts
  pilot 3's probe read, and lift-offs after the grace, not its swings (a
  lift-off followed by a touchdown) over the whole episode.  Pilot 3's
  independent check counted swings on the touch sensors: 7.3-7.8 per
  episode at -2.0 against the probe's 7.33, and 0.53 against 0.70 at -2.25,
  so the counts compare only roughly;
* the slide, over the standing episodes: the root's horizontal displacement
  from reset to the end, along the walk reward's own forward reference at
  reset (the prey or food direction; the root's heading where an env keeps
  none).  Pilot 3's parent slid backwards under noise, -0.226 m in 10 s at
  -2.0;
* the share of action entries the clip changed.

The answer is the largest grid value at which every episode stood, the
quantity ``docs/NEXT_STEPS.md`` names ("the largest value at which 120/120
episodes stand").  Near the edge the stand counts need not fall
monotonically with the noise, so the JSON also records the largest value
at which every episode stood with every smaller grid value standing too
(``below_first_failure_log_std``), and the recommendation line names both
when they differ.  The tool prints that one line, and it only reports the
measurement: the reference rows, the next value up, and a shortened panel.
Pilot 3's independent check found one fall in 160 fresh-seed episodes at its
-2.0, so 120 of 120 on seeds 3042-3081 is a measurement on those seeds, not a
guarantee.

Limits: each setting is a fixed policy with fixed statistics.  In training,
PPO moves the mean and ``log_std`` from the first update on, the warm-up's
entropy coefficient acts on ``log_std``, and the statistics update.  The
stand rates describe the warm start only.

Output, in ``--out-dir`` (default ``warm_start_noise_scan_<species>_<embodiment>``
in the current directory): ``warm_start_noise_scan.json``
(``mesozoic.warm-start-noise-scan/v1``) and ``warm_start_noise_scan.txt``,
the printed table.  The JSON holds the inputs and their sha256, the
parent's per-actuator ``log_std``, the policy's classes, the grid, the
definitions above, the summary per setting with every episode's record,
and the answer.  While the scan runs, ``warm_start_noise_scan.partial.json``
is rewritten as each setting finishes (``complete`` false, the rows so far
and the settings still ``pending`` in the current pass), so an interrupted
scan keeps what it measured; it is removed once the JSON and the table are
written, and when the pair changed on disk during the scan.  An ``--out-dir`` that already holds a scan, finished or
interrupted, is refused, so a second parent's scan never replaces the
first's; so is one inside a run
directory (the nearest ancestor holding ``provenance.json``), or inside the
stage directory of the checkpoint or its statistics
(``result_bundle.refuse_output_inside_run``): a result bundle's manifest
hashes every file under the run.  Each is refused before anything loads.
The tool changes no config, gate, reward, task, checkpoint or plant, and
writes nothing into any run.

Cost: one process steps the T. rex at about 1.5 ms per control step with
a [512, 256] policy (measured on a shared 4-CPU machine).  The default scan
is 40 ``mean`` episodes and at most 9 noisy settings (``own``, 7 first-pass
values and the refinement) x 120 episodes, at 1000 steps each: at most
1.12M steps, fewer where episodes fall, about 28 minutes of one CPU at that
rate.  A slower CPU, such as a Colab CPU runtime's, takes longer.  ``--jobs
N`` splits each pass's (setting, stream) pairs across N processes.  Each
process loads its own copy of the pair and gives torch one thread, and the
records are the same for any N.  The allosaurus's 3000-step episodes triple
the cost; ``--horizon`` shortens them.  Run the scan as a session of its
own, after the parent's run is certified and published: it needs no GPU,
and an 11M-step T. rex stance session leaves under two hours of Colab's
24-hour limit, which the run's own certificate and bundle need.

Usage::

    python -m environments.shared.scripts.warm_start_noise_scan \\
        --species velociraptor_mongoliensis --embodiment anatomical \\
        --stage-dir RUN/01_stance --jobs 4 --out-dir scratch/noise_scan_vel_r5

    python -m environments.shared.scripts.warm_start_noise_scan \\
        --species tyrannosaurus_rex --embodiment anatomical \\
        --model RUN/01_stance/models/robust_best_model.zip \\
        --vecnorm RUN/01_stance/models/robust_best_model_vecnorm.pkl \\
        --log-std -2.25 -2.0 -1.875 -1.75 --out-dir scratch/noise_scan_trex
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np

from environments.shared.constants import PUBLICATION_SEED_START
from environments.shared.model_id import ModelId
from environments.shared.species_names import add_model_arguments, parsed_model_id
from environments.shared.stage_manifest import StageEntry, StageManifest

#: Bumped when a field's meaning changes.
SCAN_SCHEMA = "mesozoic.warm-start-noise-scan/v1"
SCAN_JSON = "warm_start_noise_scan.json"
SCAN_TEXT = "warm_start_noise_scan.txt"
#: Rewritten as each setting finishes, and removed once the scan's JSON and table are written.
SCAN_PROGRESS = "warm_start_noise_scan.partial.json"
#: The default panel: seeds 3042-3081 in three noise streams (walking pilot 3's).
DEFAULT_EPISODES = 40
DEFAULT_STREAMS = 3
DEFAULT_SEEDS = tuple(range(PUBLICATION_SEED_START, PUBLICATION_SEED_START + DEFAULT_EPISODES))
#: The default grid's first pass: from GRID_BELOW under the parent's largest learned log_std to GRID_ABOVE over
#: it, in steps of GRID_STEP.
GRID_BELOW = 0.5
GRID_ABOVE = 1.0
GRID_STEP = 0.25
#: The default grid's lattice and its one refinement step: the centre is rounded to a multiple of it, so every
#: default value, the refinement included, is one a stage TOML can declare.
REFINE_STEP = 0.125
#: The middle word of every noise stream's seed sequence (the date of the decision that built the tool).
STREAM_ENTROPY = 20261126
#: The reference settings, rolled before the grid: the mean action without noise, and the parent's own log_std.
MEAN = "mean"
OWN = "own"
#: A setting: :data:`MEAN`, :data:`OWN`, or a grid value written into every entry of ``log_std``.
Setting: TypeAlias = "str | float"
#: The recipe label of the stage the scan measures.
WALK_RECIPE = "walk"

STANDS_DEFINITION = (
    "an episode stands when it reaches the horizon and env.step reported terminated=False on every step; "
    "info['termination_reason'] names the cause of an early end, and a success termination counts as not standing"
)
SLIDE_DEFINITION = (
    "standing episodes only: the root's horizontal displacement from reset to the end, along the walk reward's "
    "forward reference at reset (the prey or food direction, else the root's heading)"
)
SAMPLING_DEFINITION = (
    "clip(mean + std * z, low, high) per step, mean and std from policy.get_distribution under the set log_std; "
    "z from numpy default_rng(SeedSequence([seed, 20261126, stream])), the same draws for every setting; "
    "the mean setting sends clip(mean, low, high) and rolls each seed once"
)
ANSWER_RULE = (
    "the largest grid value at which every episode stood (docs/NEXT_STEPS.md: 'the largest value at which 120/120 "
    "episodes stand'); below_first_failure_log_std is the largest with every smaller grid value standing too"
)


def lift_off_definition(*, debounce_steps: int, grace_steps: int, control_dt: float) -> str:
    """The lift-off rule as this model's control step makes it, in words (the JSON and the table carry it)."""
    debounce = f"{debounce_steps} control step{'s' if debounce_steps != 1 else ''} at {control_dt:g} s"
    if debounce_steps <= 1:
        debounce += ", which is no debounce at all"
    return (
        "standing episodes only: a foot is down on a step when its touch force exceeds 0.1 N on at least half of the "
        f"step's substeps (the env's touch block); the down state is debounced over 20 ms ({debounce}; the gait "
        f"rule is set in seconds), and a lift-off is a down-to-up edge after the 0.1 s spawn grace ({grace_steps} "
        "steps)"
    )


class NoiseScanError(RuntimeError):
    """The scan cannot run for a reason the caller should see (the CLI turns it into an exit)."""


# ---------------------------------------------------------------------------------------------------- the grid


def grid_reference(parent_log_std: Sequence[float]) -> float:
    """The parent's largest learned ``log_std``, which the default grid is centred on."""
    values = np.asarray(parent_log_std, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise NoiseScanError(f"the parent's log_std must be finite numbers, got {values.tolist()}")
    return float(values.max())


def grid_centre(parent_log_std: Sequence[float]) -> float:
    """The reference rounded to the nearest multiple of :data:`REFINE_STEP`: the default grid's centre."""
    return math.floor(grid_reference(parent_log_std) / REFINE_STEP + 0.5) * REFINE_STEP


def default_grid(parent_log_std: Sequence[float]) -> tuple[float, ...]:
    """The first pass's 7 values: the centre, from :data:`GRID_BELOW` below it to :data:`GRID_ABOVE` above it."""
    centre = grid_centre(parent_log_std)
    below, above = round(GRID_BELOW / GRID_STEP), round(GRID_ABOVE / GRID_STEP)
    return tuple(centre + index * GRID_STEP for index in range(-below, above + 1))


def refinement(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """The default grid's second pass: one value :data:`REFINE_STEP` above the first pass's answer, or ``None``.

    *rows* are the first pass's grid summaries.  The value halves the gap
    between the largest value at which every episode stood and the next
    value up, as walking pilot 3 added -1.875 between its -2.0 and -1.75.
    There is none when no value stood every episode (the edge is below the
    grid), when the answer is the grid's largest value (the edge is above
    it), or when the next value up is already one step away.
    """
    answer = largest_standing(rows)
    if answer["log_std"] is None or answer["next_log_std"] is None:
        return None
    if answer["next_log_std"] - answer["log_std"] <= REFINE_STEP + 1e-9:
        return None
    return float(answer["log_std"]) + REFINE_STEP


def checked_grid(values: Sequence[float]) -> tuple[float, ...]:
    """``--log-std`` values, ascending; a repeat or a value that is not finite is refused."""
    grid = [float(value) for value in values]
    if not grid:
        raise NoiseScanError("--log-std names no value")
    if not all(math.isfinite(value) for value in grid):
        raise NoiseScanError(f"--log-std values must be finite, got {grid}")
    if len(set(grid)) != len(grid):
        raise NoiseScanError(f"--log-std repeats a value: {grid}")
    return tuple(sorted(grid))


# ---------------------------------------------------------------------------------------------------- the stages


def _owner(manifest: StageManifest) -> str:
    return str(manifest.model_id) if manifest.model_id is not None else "this stage manifest"


def resolve_walk_stage(manifest: StageManifest, parent_id: str, walk_stage: "int | str | None" = None) -> StageEntry:
    """The walk node that warm-starts from *parent_id*: *walk_stage* when given, else the manifest's ``walk`` child.

    The default is the one child of the parent with recipe ``walk``, else the
    one child with id ``locomotion``.  A manifest with none is refused, as is
    a named stage whose declared parent is another node.
    """
    owner = _owner(manifest)
    children = [entry for entry in manifest.stages if entry.warm_start_from == parent_id]
    if walk_stage is not None:
        entry = manifest.resolve(walk_stage)
        if entry.warm_start_from != parent_id:
            source = (
                f"point --stage-dir at a {entry.warm_start_from!r} stage directory, or "
                if entry.warm_start_from is not None
                else ""
            )
            raise NoiseScanError(
                f"{owner} stage {entry.id!r} warm-starts from {entry.warm_start_from!r}, not from {parent_id!r}: "
                f"the scan measures a warm start across the node's declared edge. To scan it, {source}name a stage "
                f"that warm-starts from {parent_id!r} with --walk-stage (they are {[child.id for child in children]})"
            )
        return entry
    walks = [entry for entry in children if entry.recipe == WALK_RECIPE] or [
        entry for entry in children if entry.id == "locomotion"
    ]
    if len(walks) != 1:
        raise NoiseScanError(
            f"{owner} declares {'no' if not walks else 'more than one'} walk stage that warm-starts from "
            f"{parent_id!r} (recipe {WALK_RECIPE!r}, or id 'locomotion'); the stages that warm-start from "
            f"{parent_id!r} are {[entry.id for entry in children]}. The scan measures a walk's warm start; name one "
            "with --walk-stage"
        )
    return walks[0]


def resolve_stages(
    manifest: StageManifest, *, parent: StageEntry | None, walk_stage: "int | str | None"
) -> tuple[StageEntry, StageEntry]:
    """``(parent, walk)``.  *parent* is the stage a ``--stage-dir`` denotes, or ``None`` for a ``--model`` pair.

    For a ``--model`` pair the parent is the declared parent of *walk_stage*
    when it is given (a root is refused), else the stance.
    """
    if parent is not None:
        return parent, resolve_walk_stage(manifest, parent.id, walk_stage)
    if walk_stage is not None:
        walk = manifest.resolve(walk_stage)
        if walk.warm_start_from is None:
            raise NoiseScanError(
                f"{_owner(manifest)} stage {walk.id!r} is a root: it warm-starts from nothing. Name the walk with "
                "--walk-stage; the stages that warm-start from another are "
                f"{[entry.id for entry in manifest.stages if entry.warm_start_from is not None]}"
            )
        return manifest.by_id(walk.warm_start_from), walk
    stance = manifest.by_id("stance")
    return stance, resolve_walk_stage(manifest, stance.id)


def parent_stage_of_directory(model_id: ModelId, manifest: StageManifest, stage_dir: Path) -> StageEntry:
    """The stage a stage directory's name denotes (``01_stance``, ``stage1``, ``stance``), or a refusal."""
    from environments.shared.stage_manifest import stage_ref_from_dirname

    reference = stage_ref_from_dirname(stage_dir.resolve().name, model_id=model_id)
    if reference is None:
        raise NoiseScanError(
            f"{stage_dir} is not named as a stage directory of {model_id} (e.g. 01_stance), so it does not say "
            "which stage the parent is. Name the pair with --model and --vecnorm instead (the parent is then the "
            "stance, or the declared parent of --walk-stage)"
        )
    return manifest.resolve(reference)


def handoff_pair(stage_dir: Path) -> tuple[str, Path, Path]:
    """``(name, checkpoint, statistics)``: the handoff ``select_handoff_checkpoint`` picks in ``<stage_dir>/models``."""
    from environments.shared.curriculum.checkpoints import select_handoff_checkpoint

    handoff = select_handoff_checkpoint(stage_dir / "models")
    if handoff is None:
        raise NoiseScanError(
            f"{stage_dir / 'models'} holds no complete handoff pair (robust_best_model or best_model, each with "
            "its _vecnorm.pkl); name a pair with --model and --vecnorm"
        )
    name, stem, vecnorm = handoff
    return name, Path(f"{stem}.zip"), Path(vecnorm)


def checkpoint_files(model: Path, vecnorm: Path) -> tuple[Path, Path]:
    """The pair as files: a checkpoint stem resolved to ``<stem>.zip`` as SB3 resolves it; a missing file refused."""
    model, vecnorm = Path(model), Path(vecnorm)
    if not model.is_file() and not model.name.endswith(".zip"):
        model = model.with_name(model.name + ".zip")
    missing = [str(path) for path in (model, vecnorm) if not path.is_file()]
    if missing:
        raise NoiseScanError(
            f"no such file: {', '.join(missing)}. Pass the parent's checkpoint (.zip, or its stem) with --model and "
            "its _vecnorm.pkl with --vecnorm, or the parent's stage directory with --stage-dir"
        )
    return model, vecnorm


def refuse_existing_scan(out_dir: Path) -> None:
    """Refuse an *out_dir* that already holds a scan, finished or interrupted: a scan must not replace another's."""
    present = [name for name in (SCAN_JSON, SCAN_TEXT, SCAN_PROGRESS) if (out_dir / name).exists()]
    if present:
        interrupted = (
            f" ({SCAN_PROGRESS} holds the rows an interrupted scan finished)" if SCAN_PROGRESS in present else ""
        )
        raise NoiseScanError(
            f"{out_dir} already holds {', '.join(present)} from an earlier scan{interrupted}; pass another --out-dir "
            "(one per parent checkpoint), or remove those files to scan again"
        )


def warm_start_reward_weights(config: Mapping[str, Any], *, parent_id: str, load_path: str) -> dict[str, float]:
    """The reward weights training sets at the warm start: each ramp's start value (stage-entry shaping)."""
    from environments.shared.curriculum import RewardRampCallback
    from environments.shared.train_base import _stage_entry_shaping_callbacks

    shaping = _stage_entry_shaping_callbacks(
        dict(config), task_load_mode="initialize_next_stage", parent_id=parent_id, load_path=load_path
    )
    return {
        callback.attr_name: float(callback.start_value)
        for callback in shaping
        if isinstance(callback, RewardRampCallback)
    }


# ---------------------------------------------------------------------------------------------------- the policy


def policy_log_std(model: Any) -> Any:
    """The policy's ``log_std`` parameter, refused unless it is one state-independent entry per actuator."""
    import torch

    policy = getattr(model, "policy", None)
    log_std = getattr(policy, "log_std", None)
    if not isinstance(log_std, torch.nn.Parameter) or log_std.dim() != 1 or getattr(policy, "use_sde", False):
        raise NoiseScanError(
            "the scan sets a Gaussian policy's state-independent log_std, as [curriculum] warm_start_log_std "
            f"does; {type(policy).__name__} has none" + (" (gSDE)" if getattr(policy, "use_sde", False) else "")
        )
    return log_std


@contextmanager
def log_std_set_to(model: Any, value: "float | None") -> Iterator[None]:
    """Every entry of the policy's ``log_std`` set to *value* in memory, restored on exit; ``None`` leaves it."""
    if value is None:
        yield
        return
    import torch

    log_std = policy_log_std(model)
    original = log_std.detach().clone()
    with torch.no_grad():
        log_std.fill_(float(value))
    try:
        yield
    finally:
        with torch.no_grad():
            log_std.copy_(original)


def action_mean_and_std(model: Any, observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The policy's action distribution for one normalized observation: its mean and std, as float64."""
    import torch

    policy = model.policy
    with torch.no_grad():
        tensor, _vectorized = policy.obs_to_tensor(observation)
        distribution = policy.get_distribution(tensor).distribution
        mean, std = distribution.mean, distribution.stddev
    return (
        np.asarray(mean.cpu().numpy(), dtype=np.float64).reshape(-1),
        np.asarray(std.cpu().numpy(), dtype=np.float64).reshape(-1),
    )


# ---------------------------------------------------------------------------------------------------- the rollouts


def noise_rng(seed: int, stream: int) -> np.random.Generator:
    """Noise stream *stream*'s generator for the episode of reset seed *seed*."""
    return np.random.default_rng(np.random.SeedSequence([int(seed), STREAM_ENTROPY, int(stream)]))


def setting_streams(setting: Setting, streams: int) -> int:
    """The noise streams a setting rolls: one for :data:`MEAN`, which draws no noise, else *streams*."""
    return 1 if setting == MEAN else streams


def setting_label(setting: Setting) -> str:
    return setting if isinstance(setting, str) else f"{setting:g}"


def _forward_reference(base: Any) -> np.ndarray:
    """The walk reward's forward direction at reset (prey or food), else the root's heading: a unit 2-vector."""
    for name in ("_initial_prey_dir_2d", "_initial_food_dir_2d"):
        reference = getattr(base, name, None)
        if reference is not None:
            break
    else:
        reference = base._quat_to_forward_2d(np.asarray(base.data.qpos[3:7], dtype=np.float64))
    vector = np.asarray(reference, dtype=np.float64).reshape(2)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 0.0 else np.array([1.0, 0.0])


def count_lift_offs(down: np.ndarray, *, debounce_steps: int, grace_steps: int) -> int:
    """The down-to-up edges of each foot's debounced down state (``down``: steps x feet) from *grace_steps* on."""
    from environments.shared.gait.events import debounce, falling_edges

    flags = np.asarray(down, dtype=bool)
    return sum(
        int(np.count_nonzero(falling_edges(debounce(flags[:, foot], debounce_steps)) >= grace_steps))
        for foot in range(flags.shape[1])
    )


def _finite(value: float) -> float | None:
    """*value*, or ``None`` when it is not finite (a ``nonfinite_state`` end): the JSON holds no NaN."""
    return value if math.isfinite(value) else None


@dataclass
class ScanContext:
    """One process's loaded pair and walk env."""

    policy: Any  # reporting.episode_trace.TracePolicy
    env: Any
    horizon: int
    seeds: tuple[int, ...]
    debounce_steps: int
    grace_steps: int
    control_dt: float

    def close(self) -> None:
        self.env.close()


def roll_episode(context: ScanContext, *, seed: int, stream: int, deterministic: bool = False) -> dict[str, Any]:
    """One episode from ``env.reset(seed=seed)`` under the policy's current ``log_std``: its record.

    *deterministic* sends the mean action and draws no noise (the :data:`MEAN` setting).
    """
    from environments.shared.gait.constants import CONTACT_THRESHOLD_N, DOWN_SUBSTEP_FRACTION

    env, model = context.env, context.policy.model
    base = env.unwrapped
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    rng = noise_rng(seed, stream)
    obs, _ = env.reset(seed=int(seed))
    start = np.asarray(base.data.qpos[0:2], dtype=np.float64).copy()
    forward = _forward_reference(base)
    total, clipped, length = 0.0, 0, 0
    terminated = False
    reason: str | None = None
    down: list[np.ndarray] = []
    while length < context.horizon:
        mean, std = action_mean_and_std(model, context.policy.observation(obs))
        sample = mean if deterministic else mean + std * rng.standard_normal(mean.shape[0])
        action = np.clip(sample, low, high)
        clipped += int(np.count_nonzero(action != sample))
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        length += 1
        block = getattr(base, "_substep_foot_forces", None)
        if block is not None:
            down.append(np.mean(np.asarray(block) > CONTACT_THRESHOLD_N, axis=0) >= DOWN_SUBSTEP_FRACTION)
        if terminated:
            reason = str(info.get("termination_reason", "terminated"))
            break
        if truncated:
            break
    stood = length >= context.horizon and not terminated
    lift_offs = (
        count_lift_offs(np.asarray(down), debounce_steps=context.debounce_steps, grace_steps=context.grace_steps)
        if down
        else None
    )
    end = np.asarray(base.data.qpos[0:2], dtype=np.float64)
    displacement = float(np.dot(end - start, forward))
    return {
        "seed": int(seed),
        "stream": int(stream),
        "length": length,
        "stood": bool(stood),
        "termination_reason": reason,
        "return": _finite(total),
        "lift_offs": lift_offs,
        "forward_displacement_m": _finite(displacement),
        "nonfinite": not (math.isfinite(total) and math.isfinite(displacement)),
        "clipped_fraction": clipped / float(length * low.shape[0]),
    }


def roll_setting(context: ScanContext, setting: Setting, stream: int) -> list[dict[str, Any]]:
    """Every seed's episode in one stream at one setting (:data:`MEAN`, :data:`OWN` or a grid value)."""
    value = None if isinstance(setting, str) else float(setting)
    with log_std_set_to(context.policy.model, value):
        return [
            roll_episode(context, seed=seed, stream=stream, deterministic=setting == MEAN) for seed in context.seeds
        ]


@dataclass(frozen=True)
class ScanSpec:
    """What every process needs to build the same :class:`ScanContext` (picklable, for ``--jobs``)."""

    model_id: ModelId
    model_path: str
    vecnorm_path: str
    walk_stage: "int | str"
    reward_weights: Mapping[str, float]
    horizon: int
    seeds: tuple[int, ...]
    allow_legacy_plant: bool = False


def open_context(spec: ScanSpec) -> ScanContext:
    """Load the pair (plant-checked, statistics frozen) and build the walk env (plant-checked, warm-start weights)."""
    from environments.shared.config import build_env
    from environments.shared.gait.constants import DEBOUNCE_S, SPAWN_GRACE_S
    from environments.shared.gait.events import steps_for
    from environments.shared.plant_contract import (
        PlantCompatibilityError,
        current_plant_identity,
        validate_environment_plant,
    )
    from environments.shared.reporting.episode_trace import EpisodeTraceError, load_trace_policy

    def walk_env() -> Any:
        return build_env(spec.model_id, spec.walk_stage)

    identity = current_plant_identity(spec.model_id)
    try:
        policy = load_trace_policy(
            spec.model_path,
            spec.vecnorm_path,
            walk_env,
            plant_identity=identity,
            allow_legacy_plant=spec.allow_legacy_plant,
        )
        policy_log_std(policy.model)
        env = walk_env()
    except (EpisodeTraceError, PlantCompatibilityError) as exc:
        raise NoiseScanError(str(exc)) from exc
    try:
        # As make_env checks a training env: the policy interface too, not only the compiled physics.
        validate_environment_plant(env, identity, artifact=f"{spec.model_id} warm-start noise scan environment")
        for name, value in spec.reward_weights.items():
            env.unwrapped.set_reward_weight(name, value)
        base = env.unwrapped
        control_dt = float(base.model.opt.timestep) * int(base.frame_skip)
    except PlantCompatibilityError as exc:
        env.close()
        raise NoiseScanError(str(exc)) from exc
    except BaseException:
        env.close()
        raise
    return ScanContext(
        policy,
        env,
        spec.horizon,
        tuple(spec.seeds),
        debounce_steps=steps_for(DEBOUNCE_S, control_dt),
        grace_steps=steps_for(SPAWN_GRACE_S, control_dt),
        control_dt=control_dt,
    )


@contextmanager
def one_torch_thread() -> Iterator[None]:
    """Torch on one thread for the scan, so a batch-one forward pass gives the same bits in every process."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


_WORKER: ScanContext | None = None


def _init_worker(spec: ScanSpec) -> None:
    """A ``--jobs`` process's start: torch on one thread, then its own :class:`ScanContext`."""
    import torch

    global _WORKER
    torch.set_num_threads(1)
    _WORKER = open_context(spec)


def _run_task(task: tuple[int, Setting, int]) -> tuple[int, int, list[dict[str, Any]]]:
    """One (setting, stream) pair in a ``--jobs`` process: ``(setting index, stream, episodes)``."""
    index, setting, stream = task
    assert _WORKER is not None, "the worker was not initialised"
    return index, stream, roll_setting(_WORKER, setting, stream)


def run_scan(
    spec: ScanSpec,
    settings: Sequence[Setting],
    *,
    streams: int,
    jobs: int = 1,
    context: ScanContext | None = None,
    on_setting: Callable[[int, list[list[dict[str, Any]]]], None] | None = None,
) -> list[list[list[dict[str, Any]]]]:
    """Every setting's episodes, ``[setting][stream][seed]``; in this process, or across *jobs* processes.

    :data:`MEAN` rolls one stream (:func:`setting_streams`).  *context* is
    this process's, reused when *jobs* is 1.  The records do not depend on
    *jobs*: each (setting, stream) pair resets its own episodes and draws its
    own noise.  *on_setting* is called in this process with a setting's index
    and episodes once its last stream is in, in the order of *settings*.
    """
    tasks = [
        (index, setting, stream)
        for index, setting in enumerate(settings)
        for stream in range(setting_streams(setting, streams))
    ]
    results: list[list[list[dict[str, Any]]]] = [
        [[] for _ in range(setting_streams(setting, streams))] for setting in settings
    ]
    remaining = [setting_streams(setting, streams) for setting in settings]

    def finished(index: int, stream: int, episodes: list[dict[str, Any]]) -> None:
        results[index][stream] = episodes
        remaining[index] -= 1
        if remaining[index] == 0 and on_setting is not None:
            on_setting(index, results[index])

    if jobs <= 1:
        own = context if context is not None else open_context(spec)
        try:
            with one_torch_thread():
                for index, setting, stream in tasks:
                    finished(index, stream, roll_setting(own, setting, stream))
        finally:
            if context is None:
                own.close()
        return results
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=min(jobs, len(tasks)),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
        initargs=(spec,),
    ) as pool:
        for index, stream, episodes in pool.map(_run_task, tasks):
            finished(index, stream, episodes)
    return results


# ---------------------------------------------------------------------------------------------------- the summary


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean(values)) if len(values) else None


def summarize_setting(
    setting: Setting,
    by_stream: Sequence[Sequence[Mapping[str, Any]]],
    *,
    parent_log_std: Sequence[float] | None = None,
) -> dict:
    """One setting's readouts over its streams (the module docstring defines each).

    *parent_log_std* gives :data:`OWN`'s std range (its std differs per actuator).
    """
    episodes = [episode for stream in by_stream for episode in stream]
    standing = [episode for episode in episodes if episode["stood"]]
    lifts = [episode["lift_offs"] for episode in standing if episode["lift_offs"] is not None]
    slide = [episode["forward_displacement_m"] for episode in standing if episode["forward_displacement_m"] is not None]
    returns = [episode["return"] for episode in episodes if episode["return"] is not None]
    ends = Counter(episode["termination_reason"] or "truncated" for episode in episodes if not episode["stood"])
    steps = sum(episode["length"] for episode in episodes)
    # Every standing episode counted its lift-offs, or none did (an env without a touch block).
    counted = len(lifts) == len(standing)
    value: float | None = None
    std: float | None = None
    std_range: list[float] | None = None
    if setting == MEAN:
        kind, std, std_range = MEAN, 0.0, [0.0, 0.0]
    elif setting == OWN:
        kind = OWN
        learned = np.exp(np.asarray(parent_log_std if parent_log_std is not None else [], dtype=np.float64))
        std_range = [float(learned.min()), float(learned.max())] if learned.size else None
    else:
        kind, value = "grid", float(setting)
        std = math.exp(value)
        std_range = [std, std]
    return {
        "label": setting_label(setting),
        "kind": kind,
        "log_std": value,
        "std": std,
        "std_range": std_range,
        "episodes": len(episodes),
        "stood": len(standing),
        "stand_rate": len(standing) / len(episodes) if episodes else None,
        "stood_per_stream": [sum(1 for episode in stream if episode["stood"]) for stream in by_stream],
        "episodes_per_stream": [len(stream) for stream in by_stream],
        "ends": dict(sorted(ends.items(), key=lambda item: (-item[1], item[0]))),
        "mean_length": _mean([episode["length"] for episode in episodes]),
        "mean_return": _mean(returns),
        "nonfinite_episodes": sum(1 for episode in episodes if episode.get("nonfinite")),
        "lift_offs_per_standing_episode": _mean(lifts) if counted else None,
        "standing_episodes_with_lift_off": sum(1 for count in lifts if count > 0) if counted else None,
        "forward_displacement_mean_m": _mean(slide),
        "forward_displacement_sd_m": float(np.std(slide)) if slide else None,
        "clipped_fraction": (
            sum(episode["clipped_fraction"] * episode["length"] for episode in episodes) / steps if steps else None
        ),
        "episode_records": [dict(episode) for episode in episodes],
    }


def largest_standing(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The largest grid value at which every episode stood, and the largest below the first failure.

    *rows* are the grid's summaries (never the reference settings).  The
    answer (``log_std``) is the largest value at which every episode stood,
    as ``docs/NEXT_STEPS.md`` defines it.  ``below_first_failure_log_std``
    is the largest value at which every episode stood with every smaller
    value standing too; the two differ only when a value below the answer
    fell (``failing_below``), which seed luck near the edge can do.
    """
    ordered = sorted(rows, key=lambda row: row["log_std"])
    standing = [index for index, row in enumerate(ordered) if row["stood"] == row["episodes"]]
    failing = [index for index, row in enumerate(ordered) if row["stood"] < row["episodes"]]
    best = ordered[standing[-1]] if standing else None
    above = ordered[standing[-1] + 1] if standing and standing[-1] + 1 < len(ordered) else None
    first = ordered[failing[0]] if failing else None
    monotone = (len(ordered) if not failing else failing[0]) - 1
    return {
        "rule": ANSWER_RULE,
        "log_std": None if best is None else best["log_std"],
        "std": None if best is None else best["std"],
        "episodes": None if best is None else best["episodes"],
        "edge_above_grid": best is not None and above is None,
        "next_log_std": None if above is None else above["log_std"],
        "next_stood": None if above is None else above["stood"],
        "next_episodes": None if above is None else above["episodes"],
        "first_failing_log_std": None if first is None else first["log_std"],
        "first_failing_stood": None if first is None else first["stood"],
        "first_failing_episodes": None if first is None else first["episodes"],
        "below_first_failure_log_std": None if monotone < 0 else ordered[monotone]["log_std"],
        "failing_below": [ordered[index]["log_std"] for index in failing if standing and index < standing[-1]],
    }


def ordered_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The reference rows (:data:`MEAN`, then :data:`OWN`), then the grid's rows by ascending ``log_std``."""
    references = [row for kind in (MEAN, OWN) for row in rows if row["kind"] == kind]
    grid = sorted((row for row in rows if row["kind"] == "grid"), key=lambda row: row["log_std"])
    return [dict(row) for row in (*references, *grid)]


def _protocol_note(*, seeds: Sequence[int], streams: int, horizon: int, episode_steps: int) -> str | None:
    """How the panel differs from the default protocol, in words; ``None`` for the default panel."""
    if tuple(seeds) == DEFAULT_SEEDS and streams == DEFAULT_STREAMS and horizon == episode_steps:
        return None
    return (
        f"a shortened panel: {len(seeds)} seed{'s' if len(seeds) != 1 else ''} x {streams} "
        f"stream{'s' if streams != 1 else ''} to {horizon} of the walk's {episode_steps} steps, not seeds "
        f"{DEFAULT_SEEDS[0]}-{DEFAULT_SEEDS[-1]} x {DEFAULT_STREAMS} streams to the full episode"
    )


def recommendation_line(
    answer: Mapping[str, Any],
    *,
    walk_config: str,
    references: Mapping[str, Mapping[str, Any]] | None = None,
    protocol: str | None = None,
) -> str:
    """The one line the scan recommends from: the measurement, and whose decision it is.

    *references* are the :data:`MEAN` and :data:`OWN` rows; *protocol* is
    :func:`_protocol_note`'s sentence for a shortened panel.
    """
    decision = (
        f"whether to declare [curriculum] warm_start_log_std in {walk_config} is the maintainer's decision; "
        "this scan sets nothing"
    )
    mean = (references or {}).get(MEAN)
    own = (references or {}).get(OWN)
    parts: list[str] = []
    if answer["log_std"] is None:
        parts.append(
            f"Measured: no grid value stood every episode (the smallest, {answer['first_failing_log_std']:g}, stood "
            f"{answer['first_failing_stood']}/{answer['first_failing_episodes']})"
        )
    else:
        found = (
            f"Measured: the largest grid log_std at which every episode stood is {answer['log_std']:g} "
            f"(std {answer['std']:.3f}, {answer['episodes']}/{answer['episodes']})"
        )
        if answer["edge_above_grid"]:
            found += "; it is the grid's largest value, so the edge lies above the grid (extend it with --log-std)"
        else:
            found += (
                f"; the next grid value, {answer['next_log_std']:g}, stood "
                f"{answer['next_stood']}/{answer['next_episodes']}"
            )
        parts.append(found)
        if answer["failing_below"]:
            below = answer["below_first_failure_log_std"]
            parts.append(
                "but " + ", ".join(f"{value:g}" for value in answer["failing_below"]) + " below it did not stand "
                f"every episode (the first, {answer['first_failing_log_std']:g}, stood "
                f"{answer['first_failing_stood']}/{answer['first_failing_episodes']}), so "
                + (
                    "no grid value stood every episode with every smaller value standing too"
                    if below is None
                    else f"the largest value with every smaller grid value standing too is {below:g}"
                )
            )
    if mean is not None and own is not None:
        parts.append(
            f"the parent's mean action, without noise, stood {mean['stood']}/{mean['episodes']}, and its own learned "
            f"log_std {own['stood']}/{own['episodes']}"
        )
        if answer["log_std"] is None:
            parts.append(
                "the mean action stood every episode, so the edge lies below the grid (scan lower values with "
                "--log-std)"
                if mean["stood"] == mean["episodes"]
                else "the parent falls in this env without noise too, so a lower grid cannot find a value at which "
                "every episode stands: check the parent and the walk env first"
            )
    if protocol is not None:
        parts.append(protocol)
    return "; ".join(parts) + f"; {decision}."


def _number(value: "float | None", spec: str) -> str:
    return "-" if value is None else format(value, spec)


def seeds_text(seeds: Sequence[int]) -> str:
    """``3042-3081`` for consecutive seeds, else every seed listed."""
    seeds = list(seeds)
    if len(seeds) > 1 and seeds == list(range(seeds[0], seeds[0] + len(seeds))):
        return f"{seeds[0]}-{seeds[-1]}"
    return ", ".join(str(seed) for seed in seeds)


def _std_text(row: Mapping[str, Any]) -> str:
    if row["std"] is not None:
        return format(row["std"], ".3f")
    if row.get("std_range"):
        low, high = row["std_range"]
        return f"{low:.3f}-{high:.3f}"
    return "-"


def render_scan(report: Mapping[str, Any]) -> str:
    """The printed report: the inputs, the definitions, one row per setting, the answer and the recommendation."""
    parent = report["parent"]
    log_std = parent["log_std"]
    names = parent["actuator_names"]
    largest = int(np.argmax(log_std))
    std = np.exp(np.asarray(log_std, dtype=np.float64))
    seeds = report["seeds"]
    handoff = f" (the stage's handoff, {report['handoff']})" if report["handoff"] else ""
    declared = report["declared_warm_start_log_std"]
    validated = "" if report["checkpoint_plant_validated"] else " (legacy plant admitted: not plant-validated)"
    lines = [
        f"Warm-start noise scan: {report['species']}/{report['embodiment']}, "
        f"{report['parent_stage']} -> {report['walk_stage']} ({report['walk_config']})",
        f"  checkpoint: {report['checkpoint']}{handoff}{validated}",
        f"  statistics: {report['vecnorm']} (frozen)",
        f"  policy: {report['policy']['algorithm']} / {report['policy']['policy_class']}",
        f"  parent log_std: largest {log_std[largest]:.3f} ({names[largest]}), mean {np.mean(log_std):.3f}, "
        f"smallest {min(log_std):.3f}; std mean {std.mean():.3f}, min {std.min():.3f}, max {std.max():.3f}",
        "  walk stage declares warm_start_log_std: "
        + ("none (the walk inherits the parent's learned log_std)" if declared is None else f"{declared:g}"),
        f"  grid: {report['grid_rule']}",
        f"  settings: {MEAN} = the mean action, no noise, each seed once; {OWN} = the parent's learned log_std; "
        "then the grid, every log_std entry set to the value",
        f"  episodes per noisy setting: {len(seeds)} seed{'s' if len(seeds) != 1 else ''} ({seeds_text(seeds)}) x "
        f"{report['streams']} noise stream{'s' if report['streams'] != 1 else ''}; horizon {report['horizon']} of "
        f"the walk's {report['episode_steps']} steps "
        f"({report['horizon'] * report['control_dt']:.1f} s); reward weights at the warm start: "
        f"{report['reward_weights'] or 'as configured'}",
        f"  stands: {report['definitions']['stands']}",
        f"  lift-offs: {report['definitions']['lift_offs']}",
        f"  slide: {report['definitions']['slide']}",
        f"  answer: {report['definitions']['answer']}",
        "",
        f"{'log_std':>8} {'std':>11} {'stood per stream':>18} {'total':>9} {'mean len':>8} {'return':>9} "
        f"{'lifts/ep':>8} {'w/ lift':>9} {'slide m (sd)':>15} {'clipped':>8}  ends",
    ]
    for row in report["settings"]:
        stood = "/".join(str(count) for count in row["stood_per_stream"])
        lifted = row["standing_episodes_with_lift_off"]
        with_lift = "-" if lifted is None else f"{lifted}/{row['stood']}"
        slide = (
            "-"
            if row["forward_displacement_mean_m"] is None
            else f"{row['forward_displacement_mean_m']:+.3f} ({row['forward_displacement_sd_m']:.3f})"
        )
        ends = ", ".join(f"{reason} {count}" for reason, count in row["ends"].items()) or "-"
        if row["nonfinite_episodes"]:
            ends += f" (non-finite return or displacement: {row['nonfinite_episodes']})"
        lines.append(
            f"{row['label']:>8} {_std_text(row):>11} {stood:>18} {row['stood']:>4}/{row['episodes']:<4} "
            f"{_number(row['mean_length'], '.1f'):>8} {_number(row['mean_return'], '.1f'):>9} "
            f"{_number(row['lift_offs_per_standing_episode'], '.2f'):>8} {with_lift:>9} {slide:>15} "
            f"{_number(row['clipped_fraction'], '.4f'):>8}  {ends}"
        )
    lines += ["", report["recommendation"]]
    return "\n".join(lines) + "\n"


def write_progress(out_dir: Path, report: Mapping[str, Any]) -> Path:
    """Write :data:`SCAN_PROGRESS`: the rows finished so far, so an interrupted scan keeps what it measured."""
    from environments.shared.file_io import atomic_write_json

    out_dir.mkdir(parents=True, exist_ok=True)
    return atomic_write_json(out_dir / SCAN_PROGRESS, report, allow_nan=False)


def write_scan(out_dir: Path, report: Mapping[str, Any], text: str) -> dict[str, Path]:
    """Write :data:`SCAN_JSON`, then :data:`SCAN_TEXT`, into *out_dir*, then remove :data:`SCAN_PROGRESS`.

    The JSON goes first: a write that fails leaves no table without its
    record, and the progress file stays until both are written.
    """
    from environments.shared.file_io import atomic_write_json, atomic_write_text

    out_dir.mkdir(parents=True, exist_ok=True)
    written = {SCAN_JSON: atomic_write_json(out_dir / SCAN_JSON, report, allow_nan=False)}
    atomic_write_text(out_dir / SCAN_TEXT, text)
    written[SCAN_TEXT] = out_dir / SCAN_TEXT
    (out_dir / SCAN_PROGRESS).unlink(missing_ok=True)
    return written


# ---------------------------------------------------------------------------------------------------- the CLI


def _seeds(parser: argparse.ArgumentParser, args: argparse.Namespace) -> tuple[int, ...]:
    """The reset seeds: ``--seeds`` as given, else ``--episodes`` consecutive seeds from ``--seed``."""
    if args.seeds is not None:
        try:
            seeds = [int(part) for part in args.seeds.split(",") if part.strip()]
        except ValueError:
            parser.error(f"--seeds must be comma-separated whole numbers, got {args.seeds!r}")
        if not seeds:
            parser.error("--seeds names no seed")
        if len(set(seeds)) != len(seeds):
            parser.error(f"--seeds repeats a seed: {args.seeds!r}")
        return tuple(seeds)
    if args.episodes < 1:
        parser.error(f"--episodes must be at least 1, got {args.episodes}")
    return tuple(args.seed + index for index in range(args.episodes))


def _stage_reference(text: str) -> "int | str":
    return int(text) if text.isdigit() else text


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--stage-dir", type=Path, help="The parent's stage directory: its handoff pair is scanned")
    source.add_argument("--model", type=Path, help="A parent checkpoint (.zip, or its stem), with --vecnorm")
    parser.add_argument("--vecnorm", type=Path, help="The --model checkpoint's VecNormalize .pkl")
    parser.add_argument(
        "--walk-stage",
        type=_stage_reference,
        help="The walk stage (default: the manifest's walk node that warm-starts from the parent, 'locomotion')",
    )
    parser.add_argument(
        "--log-std",
        type=float,
        nargs="+",
        metavar="V",
        help=f"The grid, rolled as given (default: the parent's largest learned log_std, rounded to "
        f"{REFINE_STEP:g}, -{GRID_BELOW:g} to +{GRID_ABOVE:g} in steps of {GRID_STEP:g}, then one value "
        f"{REFINE_STEP:g} above the largest that stood every episode)",
    )
    parser.add_argument("--seed", type=int, default=PUBLICATION_SEED_START, help="First reset seed (default 3042)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Seeds from --seed (default 40)")
    parser.add_argument("--seeds", help="Comma-separated reset seeds, instead of --seed and --episodes")
    parser.add_argument("--streams", type=int, default=DEFAULT_STREAMS, help="Noise streams per setting (default 3)")
    parser.add_argument("--horizon", type=int, help="Steps per episode (default: the walk stage's max_episode_steps)")
    parser.add_argument("--jobs", type=int, default=1, help="Processes to split the scan across (default 1)")
    parser.add_argument(
        "--allow-legacy-plant",
        action="store_true",
        help="scan a checkpoint that predates the plant contract; the JSON records checkpoint_plant_validated=false",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Where to write; refused if it already holds a scan "
        "(default: warm_start_noise_scan_<species>_<embodiment> here)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    model_id = parsed_model_id(parser, args)
    seeds = _seeds(parser, args)
    if args.model is not None and args.vecnorm is None:
        parser.error("--model needs --vecnorm: the scan runs on the parent's frozen statistics")
    if args.stage_dir is not None and args.vecnorm is not None:
        parser.error("--vecnorm goes with --model; --stage-dir scans its handoff pair")
    for name in ("streams", "jobs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name} must be at least 1, got {getattr(args, name)}")
    if args.horizon is not None and args.horizon < 1:
        parser.error(f"--horizon must be at least 1, got {args.horizon}")
    default_out = Path.cwd() / f"warm_start_noise_scan_{model_id.species}_{model_id.embodiment}"
    out_dir = args.out_dir if args.out_dir is not None else default_out
    try:
        report, text = _scan(args, model_id, seeds, out_dir)
    except NoiseScanError as exc:
        # The CLI boundary is where a diagnosable failure becomes an exit status.
        raise SystemExit(str(exc)) from exc
    print(text, end="")
    for path in write_scan(out_dir, report, text).values():
        print(f"written: {path}")
    return 0


def _scan(
    args: argparse.Namespace, model_id: ModelId, seeds: tuple[int, ...], out_dir: Path
) -> tuple[dict[str, Any], str]:
    """Resolve the inputs, refuse what the module docstring refuses, roll every setting and build the report."""
    from environments.shared.config import load_stage_config
    from environments.shared.curriculum.gate_schema import declared_warm_start_log_std
    from environments.shared.result_bundle import ResultBundleError, refuse_output_inside_run, sha256_file
    from environments.shared.stage_manifest import StageManifestError, load_stage_manifest
    from environments.shared.task_fingerprint import (
        TaskFingerprintError,
        read_checkpoint_task_fingerprint,
        validate_declared_parent,
    )

    # Every refusal before anything is loaded or rolled: a refused scan costs nothing.
    handoff: str | None = None
    if args.stage_dir is not None:
        handoff, model_path, vecnorm_path = handoff_pair(args.stage_dir)
    else:
        model_path, vecnorm_path = args.model, args.vecnorm
    try:
        refuse_output_inside_run(out_dir, [model_path, vecnorm_path])
        refuse_existing_scan(out_dir)
        manifest = load_stage_manifest(model_id)
        directory_stage = (
            parent_stage_of_directory(model_id, manifest, args.stage_dir) if args.stage_dir is not None else None
        )
        parent, walk = resolve_stages(manifest, parent=directory_stage, walk_stage=args.walk_stage)
    except (ResultBundleError, StageManifestError) as exc:
        raise NoiseScanError(str(exc)) from exc

    config = load_stage_config(model_id, walk.reference)
    walk_config = f"configs/{model_id.species}/{model_id.embodiment}/{walk.config_file}"
    env_kwargs = dict(config["env_kwargs"])
    command_mode = str(env_kwargs.get("command_mode", "none"))
    if command_mode != "none":
        raise NoiseScanError(
            f"{walk_config} trains under command_mode {command_mode!r}: a warm start into a live-command node "
            "reseeds the parent's command statistics (train_base._load_vecnorm_into_envs), which this scan does not"
        )
    episode_steps = int(env_kwargs.get("max_episode_steps", 1000))
    horizon = episode_steps if args.horizon is None else int(args.horizon)
    if horizon > episode_steps:
        raise NoiseScanError(
            f"--horizon {horizon} is longer than {walk_config}'s episode ({episode_steps} steps); pass at most "
            f"{episode_steps}, or leave --horizon out to roll the whole episode"
        )
    model_path, vecnorm_path = checkpoint_files(model_path, vecnorm_path)
    try:
        validate_declared_parent(
            read_checkpoint_task_fingerprint(model_path),
            declared_parent=parent.reference,
            model_id=model_id,
            child_stage=walk.reference,
            artifact=str(model_path),
        )
    except TaskFingerprintError as exc:
        raise NoiseScanError(str(exc)) from exc

    spec = ScanSpec(
        model_id=model_id,
        model_path=str(model_path),
        vecnorm_path=str(vecnorm_path),
        walk_stage=walk.reference,
        reward_weights=warm_start_reward_weights(config, parent_id=parent.id, load_path=str(model_path)),
        horizon=horizon,
        seeds=seeds,
        allow_legacy_plant=bool(args.allow_legacy_plant),
    )
    sha256 = {"checkpoint": sha256_file(model_path), "vecnorm": sha256_file(vecnorm_path)}
    protocol = _protocol_note(seeds=seeds, streams=int(args.streams), horizon=horizon, episode_steps=episode_steps)
    context = open_context(spec)
    try:
        trained = context.policy.model
        parent_log_std = policy_log_std(trained).detach().cpu().double().numpy().copy()
        policy_classes = {"algorithm": type(trained).__name__, "policy_class": type(trained.policy).__name__}
        model = context.env.unwrapped.model
        names = [str(model.actuator(index).name) for index in range(int(model.nu))]
        report: dict[str, Any] = {
            "schema": SCAN_SCHEMA,
            **model_id.record(),
            "complete": False,
            "parent_stage": parent.id,
            "walk_stage": walk.id,
            "walk_config": walk_config,
            "declared_warm_start_log_std": declared_warm_start_log_std(config["curriculum_kwargs"], stage=walk.id),
            "checkpoint": str(model_path),
            "vecnorm": str(vecnorm_path),
            "handoff": handoff,
            "sha256": sha256,
            "checkpoint_plant_validated": not args.allow_legacy_plant,
            "normalization": context.policy.normalization,
            "policy": policy_classes,
            "parent": {"actuator_names": names, "log_std": parent_log_std.tolist()},
            "grid_rule": None,
            "grid": [],
            "seeds": list(seeds),
            "streams": int(args.streams),
            "horizon": horizon,
            "episode_steps": episode_steps,
            "full_protocol": protocol is None,
            "control_dt": context.control_dt,
            "lift_off_debounce_steps": context.debounce_steps,
            "lift_off_grace_steps": context.grace_steps,
            "reward_weights": dict(spec.reward_weights),
            "definitions": {
                "stands": STANDS_DEFINITION,
                "sampling": SAMPLING_DEFINITION,
                "lift_offs": lift_off_definition(
                    debounce_steps=context.debounce_steps,
                    grace_steps=context.grace_steps,
                    control_dt=context.control_dt,
                ),
                "slide": SLIDE_DEFINITION,
                "answer": ANSWER_RULE,
            },
            "settings": [],
            "pending": [],
        }
        rows: list[dict[str, Any]] = []

        def roll(settings: Sequence[Setting]) -> None:
            """Roll one pass of *settings*, writing :data:`SCAN_PROGRESS` as each finishes."""

            def record(index: int, by_stream: list[list[dict[str, Any]]]) -> None:
                rows.append(summarize_setting(settings[index], by_stream, parent_log_std=parent_log_std.tolist()))
                report["settings"] = ordered_rows(rows)
                report["pending"] = [setting_label(setting) for setting in settings[index + 1 :]]
                write_progress(out_dir, report)

            report["pending"] = [setting_label(setting) for setting in settings]
            run_scan(spec, settings, streams=int(args.streams), jobs=args.jobs, context=context, on_setting=record)

        print(f"progress: {out_dir / SCAN_PROGRESS} (rewritten as each setting finishes)", flush=True)
        if args.log_std is None:
            first = default_grid(parent_log_std)
            report["grid"] = list(first)
            report["grid_rule"] = (
                f"the parent's largest learned log_std {grid_reference(parent_log_std):.4f} rounded to "
                f"{grid_centre(parent_log_std):g}, from {first[0]:g} to {first[-1]:g} in steps of {GRID_STEP:g}"
            )
            roll([MEAN, OWN, *first])
            first_answer = largest_standing([row for row in rows if row["kind"] == "grid"])
            extra = refinement([row for row in rows if row["kind"] == "grid"])
            if extra is not None:
                report["grid"] = sorted([*first, extra])
                report["grid_rule"] += (
                    f", then {extra:g}, {REFINE_STEP:g} above {first_answer['log_std']:g}, the largest of those at "
                    "which every episode stood"
                )
                roll([extra])
            elif first_answer["log_std"] is None:
                report["grid_rule"] += "; no refinement: no value stood every episode"
            else:
                report["grid_rule"] += "; no refinement: the largest value stood every episode, so the edge lies above"
        else:
            grid = checked_grid(args.log_std)
            report["grid"] = list(grid)
            report["grid_rule"] = "--log-std " + " ".join(f"{value:g}" for value in grid)
            roll([MEAN, OWN, *grid])
    finally:
        context.close()
    after = {"checkpoint": sha256_file(model_path), "vecnorm": sha256_file(vecnorm_path)}
    if after != sha256:
        # Rows rolled from a pair that changed under them measure no one checkpoint: none is kept.
        (out_dir / SCAN_PROGRESS).unlink(missing_ok=True)
        raise NoiseScanError(
            f"the checkpoint pair changed on disk during the scan: {sha256} -> {after}; nothing is kept (the progress "
            "file is removed): scan again once the pair is no longer being written"
        )

    del report["pending"]
    report["complete"] = True
    report["settings"] = ordered_rows(rows)
    answer = largest_standing([row for row in report["settings"] if row["kind"] == "grid"])
    report["largest_all_standing"] = answer
    report["recommendation"] = recommendation_line(
        answer,
        walk_config=walk_config,
        references={row["kind"]: row for row in rows if row["kind"] in (MEAN, OWN)},
        protocol=protocol,
    )
    return report, render_scan(report)


if __name__ == "__main__":
    raise SystemExit(main())
