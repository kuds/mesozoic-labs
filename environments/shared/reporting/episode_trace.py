"""Trace single episodes of a checkpoint step by step, per actuator: report-only.

The stance gate flags actuator saturation (``max_actuator_saturation_fraction``,
``|a| >= SATURATION_ABS`` over the post-settle window), and the stance report
reduces each actuator's action to a window mean and spread.  Neither says,
per step, what the policy asks for, what reaches the plant and what the joint
and motor do with it.  The T. rex stance habit of stance3 Run 2 and stance4
s42 (``tail_3_pitch_act`` commanded at +1, tail up;
``docs/investigations/stance_2026_10/stance4/satdiag/report.md``) took
scratch tools to measure: the raw Gaussian mean sat at +1.01 to +1.12 before
the clip, the joint 0.216 rad below its stop on 0.26 of its force limit, and
the command first reached 0.99 at a median of step 216.  This module measures
those quantities for any checkpoint.

Per episode, per step and per actuator (:data:`TRACE_FIELDS`):

* ``raw_mean`` -- the policy's Gaussian mean before the action clip:
  ``policy.get_distribution(obs).mode()`` under ``torch.no_grad()``, the
  forward pass the policy probe uses
  (:func:`~environments.shared.stance_diagnostics.policy_raw_mean`);
* ``sample`` -- with *stochastic_seed* only, the mean plus the policy's own
  noise, before the clip;
* ``command`` -- the action sent to ``env.step``;
* ``ctrl`` -- ``data.ctrl``, the servo target the env applied;
* ``joint_pos`` and ``joint_vel`` -- the driven joint's position and velocity
  (NaN for a transmission that is not a hinge or slide joint);
* ``force`` and ``force_frac`` -- ``data.actuator_force`` and its fraction of
  the force limit on that side
  (:func:`~environments.shared.gait.balance_telemetry.actuator_force_fraction`,
  NaN where the actuator has none).

Everything after the step is read after ``env.step``, so it is the state the
command produced (the force is the last substep's).

The default rollout is the stance judge's: the stage's own env built from its
config and checked against the model's current plant, the checkpoint and its
VecNormalize statistics loaded frozen through
:func:`~environments.shared.policy_loading.load_sb3_checkpoint` with no
sidecar guessing (as :mod:`~environments.shared.reporting.stance_report`
loads them), the observation normalized by the same formula and cast to
float32, ``model.predict(..., deterministic=True)`` (which clips to the
action bound) as the command, as float64, and ``env.reset(seed=seed)`` per
episode.  So the trace of seed ``3042 + i`` is episode ``i`` of the stance
panel, the v1 and v2 panels and the heading certificate's 0 degree cell
alike (``test_episode_trace.py`` checks it against ``run_panel``).  The
other headings of the certificate are not traced.

Two report-only modifications, recorded in the report and never the judge's:

* *stochastic_seed*: the command is ``clip(raw_mean + sigma z)`` with
  ``sigma = exp(log_std)``, the policy's own exploration noise, and ``z``
  from ``numpy.random.default_rng(stochastic_seed + i)`` in episode ``i`` --
  with the first reset seed as *stochastic_seed* that is the stance screens'
  ``noise_matched_own`` draw (``default_rng(episode seed)``);
* *holds*: one or more actuators' commands replaced by a constant
  (:class:`ActuatorHold`), for counterfactual probes such as holding
  ``tail_3_pitch_act`` at 0.  The raw mean is still the policy's.

The summary reduces each episode over the stance gate's post-settle window,
steps ``settle_steps`` to the episode's end (0-based, the step index of the
action, :mod:`~environments.shared.gait.stance_metrics`), at the gate's
:data:`~environments.shared.gait.constants.SATURATION_ABS`.  It changes no
gate, reward, observation, action, task, checkpoint selection or plant, and
it writes outside the run (:func:`refuse_run_directory_output`).

Deliberately not re-exported from :mod:`environments.shared.reporting`, like
:mod:`~environments.shared.reporting.stance_report`: the species registry it
imports pulls ``train_base`` at module level.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import PUBLICATION_SEED_START
from ..curriculum.stance_heading_gate import FROZEN_VECNORMALIZE_MODE
from ..file_io import atomic_write_csv, atomic_write_json
from ..gait.balance_telemetry import actuator_force_fraction, actuator_force_limits
from ..gait.constants import SATURATION_ABS
from ..model_id import ModelId
from ..policy_loading import PolicyLoadError, load_sb3_checkpoint
from ..species_names import model_display_name
from ..species_registry import env_class as model_env_class
from ..stage_manifest import stage_display_name
from ..stance_diagnostics import ActuatorLayout, policy_raw_mean

#: Bumped when the summary's field meanings change.
TRACE_SCHEMA = "mesozoic.episode-trace/v1"
#: The summary the writer puts beside the per-episode CSVs.
TRACE_SUMMARY_JSON = "episode_trace.json"
#: The second, looser threshold the summary counts at: the satdiag's ">= 0.9" (a command on its way to the rail).
NEAR_SATURATION_ABS = 0.9
#: The per-actuator quantities of a step, in column order (``<actuator>_<field>`` in the CSV).
TRACE_FIELDS: tuple[str, ...] = (
    "raw_mean",
    "sample",
    "command",
    "ctrl",
    "joint_pos",
    "joint_vel",
    "force",
    "force_frac",
)

NAN = float("nan")


class EpisodeTraceError(RuntimeError):
    """The trace could not be produced for a reason the caller should see (the CLI turns it into an exit)."""


@dataclass(frozen=True)
class ActuatorHold:
    """One actuator's command replaced by *value* on every step: a counterfactual probe, never the judge's rollout."""

    name: str
    value: float

    @classmethod
    def parse(cls, text: str) -> "ActuatorHold":
        """``NAME=VALUE``, as ``--hold`` takes it."""
        name, separator, value = text.partition("=")
        if not separator or not name.strip():
            raise ValueError(f"a hold is NAME=VALUE, got {text!r}")
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"a hold's value must be a number, got {text!r}") from None
        if not math.isfinite(number):
            raise ValueError(f"a hold's value must be finite, got {text!r}")
        return cls(name.strip(), number)

    def as_dict(self) -> dict[str, Any]:
        return {"actuator": self.name, "value": self.value}


@dataclass
class EpisodeTrace:
    """One traced episode: ``(T, nu)`` arrays per :data:`TRACE_FIELDS` entry, every actuator, and its outcome."""

    seed: int
    reward: np.ndarray  # (T,)
    values: dict[str, np.ndarray] = field(default_factory=dict)  # field -> (T, nu)
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None

    @property
    def length(self) -> int:
        return int(self.reward.shape[0])

    @property
    def episode_return(self) -> float:
        """The rewards summed in step order, as the stance report sums an episode's (so the two agree to the bit)."""
        total = 0.0
        for reward in self.reward.tolist():
            total += reward
        return total


@dataclass
class TracePolicy:
    """A loaded checkpoint and the observation it is given: the stance report's frozen normalization."""

    model: Any
    normalization: dict[str, Any]
    mean: np.ndarray | None = None
    var: np.ndarray | None = None
    epsilon: float = 0.0
    clip: float = 0.0

    def observation(self, obs: np.ndarray) -> np.ndarray:
        """What ``predict`` is given: ``stance_report._load_policy``'s formula, cast to float32, or *obs* unchanged."""
        if self.mean is None or self.var is None:
            return obs
        normalized: np.ndarray = np.clip((obs - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip, self.clip)
        return normalized.astype(np.float32)


def load_trace_policy(
    model_path: str,
    vecnorm_path: str | None,
    env_factory: Any,
    *,
    plant_identity: Any = None,
    allow_unnormalized: bool = False,
    allow_legacy_plant: bool = False,
) -> TracePolicy:
    """Load the checkpoint and its statistics as the stance report does: frozen, named, checked against the plant.

    No sidecar is guessed.  A missing *vecnorm_path* is refused unless
    *allow_unnormalized*: a policy given raw observations is a different
    policy, and a trace of it would describe that one.
    """
    if vecnorm_path is None and not allow_unnormalized:
        raise EpisodeTraceError(
            f"no VecNormalize statistics named for {model_path}: pass the checkpoint's matched .pkl, or allow an "
            "unnormalized trace explicitly (it then traces a different policy)"
        )
    try:
        model, normalizer, resolved = load_sb3_checkpoint(
            model_path,
            vecnorm_path,
            env_factory,
            guess_sidecar=False,
            allow_unnormalized=True,
            plant_identity=plant_identity,
            allow_legacy_plant=allow_legacy_plant,
        )
    except PolicyLoadError as exc:
        raise EpisodeTraceError(str(exc)) from exc
    policy = getattr(model, "policy", None)
    if policy is not None and hasattr(policy, "set_training_mode"):
        policy.set_training_mode(False)
    if normalizer is None:
        return TracePolicy(model, {"mode": "none"})
    obs_rms = normalizer.obs_rms
    if isinstance(obs_rms, dict):
        raise EpisodeTraceError(
            f"{resolved} holds per-key statistics for a Dict observation space (keys: {sorted(obs_rms)}), but this "
            "stage builds a flat Box observation."
        )
    epsilon = float(normalizer.epsilon)
    clip = float(normalizer.clip_obs)
    return TracePolicy(
        model,
        {"mode": FROZEN_VECNORMALIZE_MODE, "clip_obs": clip, "epsilon": epsilon},
        mean=np.asarray(obs_rms.mean, dtype=np.float64),
        var=np.asarray(obs_rms.var, dtype=np.float64),
        epsilon=epsilon,
        clip=clip,
    )


def _policy_std(model: Any) -> np.ndarray | None:
    """``exp(log_std)`` of a diagonal Gaussian policy, as float64, or ``None``."""
    log_std = getattr(getattr(model, "policy", None), "log_std", None)
    if log_std is None:
        return None
    try:
        values = log_std.detach().cpu().numpy()
    except AttributeError:
        return None
    return np.exp(np.asarray(values, dtype=np.float64).reshape(-1))


def trace_episode(
    env: Any,
    policy: TracePolicy,
    *,
    seed: int,
    layout: ActuatorLayout,
    holds: "dict[int, float] | None" = None,
    noise_rng: "np.random.Generator | None" = None,
) -> EpisodeTrace:
    """Roll one episode from ``env.reset(seed=seed)`` to its end, recording :data:`TRACE_FIELDS` at every step.

    Deterministic without *noise_rng*: the command is the stance judge's
    (``model.predict(..., deterministic=True)``, clipped by SB3, as float64).
    With it, the command is ``clip(raw_mean + exp(log_std) z, low, high)``,
    ``z = noise_rng.standard_normal(nu)`` per step.  *holds* then replaces
    the listed actuators' commands.
    """
    base = env.unwrapped
    model, data = base.model, base.data
    nu = int(model.nu)
    low = np.asarray(env.action_space.low, dtype=np.float64)
    high = np.asarray(env.action_space.high, dtype=np.float64)
    limits = actuator_force_limits(model)
    qpos_index = np.asarray(layout.qpos_index)
    dof_index = np.asarray(layout.dof_index)
    has_joint = qpos_index >= 0
    sigma = _policy_std(policy.model) if noise_rng is not None else None
    if noise_rng is not None and (sigma is None or sigma.shape != (nu,)):
        raise EpisodeTraceError("--stochastic needs a diagonal Gaussian policy with one log_std per actuator")
    rows: dict[str, list[np.ndarray]] = {name: [] for name in TRACE_FIELDS}
    rewards: list[float] = []
    obs, _ = env.reset(seed=seed)
    terminated = truncated = False
    info: dict[str, Any] = {}
    while True:
        obs_input = policy.observation(obs)
        raw = policy_raw_mean(policy.model, obs_input)
        raw = raw if raw is not None and raw.shape == (nu,) else np.full(nu, NAN)
        if noise_rng is None or sigma is None:
            sample = np.full(nu, NAN)
            action = np.asarray(policy.model.predict(obs_input, deterministic=True)[0], dtype=np.float64).ravel()
        else:
            if not np.all(np.isfinite(raw)):
                raise EpisodeTraceError("--stochastic needs the policy's Gaussian mean, and its forward pass failed")
            sample = raw + sigma * noise_rng.standard_normal(nu)
            action = np.clip(sample, low, high)
        for index, value in (holds or {}).items():
            action[index] = value
        obs, reward, terminated, truncated, info = env.step(action)
        force = np.asarray(data.actuator_force, dtype=np.float64).copy()
        qpos = np.asarray(data.qpos, dtype=np.float64)
        qvel = np.asarray(data.qvel, dtype=np.float64)
        rows["raw_mean"].append(raw)
        rows["sample"].append(sample)
        rows["command"].append(action.copy())
        rows["ctrl"].append(np.asarray(data.ctrl, dtype=np.float64).copy())
        rows["joint_pos"].append(np.where(has_joint, qpos[np.where(has_joint, qpos_index, 0)], NAN))
        rows["joint_vel"].append(np.where(has_joint, qvel[np.where(has_joint, dof_index, 0)], NAN))
        rows["force"].append(force)
        rows["force_frac"].append(actuator_force_fraction(force[None], limits)[0])
        rewards.append(float(reward))
        if terminated or truncated:
            break
    reason = info.get("termination_reason", "terminated" if terminated else "truncated")
    return EpisodeTrace(
        seed=int(seed),
        reward=np.asarray(rewards, dtype=np.float64),
        values={name: np.asarray(values, dtype=np.float64).reshape(-1, nu) for name, values in rows.items()},
        terminated=bool(terminated),
        truncated=bool(truncated),
        termination_reason=str(reason),
    )


def _finite(value: Any) -> float | None:
    """A finite float, or ``None`` (JSON's null) for NaN and infinities."""
    number = float(value)
    return number if math.isfinite(number) else None


def _window_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    return _finite(np.mean(finite)) if finite.size else None


def _first_step(mask: np.ndarray, start: int = 0) -> int | None:
    hits = np.flatnonzero(mask[start:])
    return int(hits[0]) + start if hits.size else None


def summarize_episode(
    trace: EpisodeTrace, *, names: Sequence[str], settle_steps: int, horizon: int, selected: Sequence[int]
) -> dict[str, Any]:
    """One episode's summary: its outcome, the gate's saturation over every actuator, and the *selected* actuators.

    The window is the stance gate's, steps ``settle_steps`` to the end
    (empty when the episode ends first: every window statistic is then
    ``None``).  ``max_actuator_saturation_fraction`` and
    ``mean_actuator_saturation_fraction`` are the gate's two
    (``stance_metrics._saturation_metrics``), over every actuator.
    """
    settle = int(settle_steps)
    window = slice(settle, trace.length)
    command = trace.values["command"]
    raw = trace.values["raw_mean"]
    window_command = np.abs(command[window])
    saturated = (window_command >= SATURATION_ABS).mean(axis=0) if window_command.shape[0] else None
    actuators: dict[str, Any] = {}
    for index in selected:
        column_command = command[:, index]
        column_raw = raw[:, index]
        window_raw = column_raw[window]
        finite_raw = column_raw[np.isfinite(column_raw)]
        steps = window_command.shape[0]
        actuators[names[index]] = {
            "index": int(index),
            "command_window_mean": _window_mean(column_command[window]),
            "raw_mean_window_mean": _window_mean(window_raw),
            "saturated_fraction": _finite(saturated[index]) if saturated is not None else None,
            "near_saturated_fraction": (
                _finite(np.mean(window_command[:, index] >= NEAR_SATURATION_ABS)) if steps else None
            ),
            "raw_mean_beyond_bound_fraction": (
                _finite(np.mean(np.abs(window_raw) > 1.0)) if steps and np.isfinite(window_raw).all() else None
            ),
            "raw_mean_max": _finite(finite_raw.max()) if finite_raw.size else None,
            "raw_mean_min": _finite(finite_raw.min()) if finite_raw.size else None,
            "first_saturated_step": _first_step(np.abs(column_command) >= SATURATION_ABS),
            "first_saturated_window_step": _first_step(np.abs(column_command) >= SATURATION_ABS, settle),
            "ctrl_window_mean": _window_mean(trace.values["ctrl"][window, index]),
            "joint_pos_window_mean": _window_mean(trace.values["joint_pos"][window, index]),
            "force_frac_window_mean": _window_mean(trace.values["force_frac"][window, index]),
            "force_frac_window_max": (
                _finite(np.nanmax(trace.values["force_frac"][window, index]))
                if steps and np.isfinite(trace.values["force_frac"][window, index]).any()
                else None
            ),
        }
    return {
        "seed": trace.seed,
        "length": trace.length,
        "return": trace.episode_return,
        "reached_horizon": trace.length >= horizon,
        "terminated": trace.terminated,
        "truncated": trace.truncated,
        "termination_reason": trace.termination_reason,
        "window_steps": max(trace.length - settle, 0),
        "max_actuator_saturation_fraction": _finite(saturated.max()) if saturated is not None else None,
        "mean_actuator_saturation_fraction": _finite(saturated.mean()) if saturated is not None else None,
        "actuators": actuators,
    }


def select_actuators(names: Sequence[str], patterns: Sequence[str]) -> tuple[int, ...]:
    """Indices whose name contains any of *patterns* (every actuator for none); a pattern matching none is refused."""
    if not patterns:
        return tuple(range(len(names)))
    unmatched = [pattern for pattern in patterns if not any(pattern in name for name in names)]
    if unmatched:
        raise EpisodeTraceError(f"no actuator name contains {unmatched}; the actuators are {list(names)}")
    return tuple(index for index, name in enumerate(names) if any(pattern in name for pattern in patterns))


def _hold_indices(
    names: Sequence[str], holds: Sequence[ActuatorHold], low: np.ndarray, high: np.ndarray
) -> dict[int, float]:
    """Each hold's actuator index; an unknown name, a value outside the action bound or a repeat is refused."""
    indices: dict[int, float] = {}
    for hold in holds:
        if hold.name not in names:
            raise EpisodeTraceError(f"--hold names no actuator: {hold.name!r}; the actuators are {list(names)}")
        index = list(names).index(hold.name)
        if index in indices:
            raise EpisodeTraceError(f"--hold names {hold.name!r} twice")
        if not low[index] <= hold.value <= high[index]:
            raise EpisodeTraceError(
                f"--hold {hold.name}={hold.value:g} lies outside the action bound [{low[index]:g}, {high[index]:g}]"
            )
        indices[index] = float(hold.value)
    return indices


def build_episode_trace(
    model_id: ModelId,
    stage: "int | str",
    *,
    stage_config: dict[str, Any],
    model_path: str,
    vecnorm_path: str | None,
    seeds: Sequence[int] = (PUBLICATION_SEED_START,),
    actuators: Sequence[str] = (),
    holds: Sequence[ActuatorHold] = (),
    stochastic_seed: int | None = None,
    allow_unnormalized: bool = False,
    allow_legacy_plant: bool = False,
) -> tuple[dict[str, Any], list[EpisodeTrace]]:
    """Trace one episode per seed and return ``(report, traces)``; the module docstring says what each holds.

    Checked before anything is rolled: the checkpoint and its statistics
    against the model's current plant (as they load), the rollout env against
    it, the actuator filter and the holds against the env's actuators.
    """
    from ..curriculum.stance_gate import StanceGateThresholds
    from ..plant_contract import current_plant_identity, validate_environment_plant

    if not seeds:
        raise ValueError("seeds must name at least one episode")
    curriculum = stage_config["curriculum_kwargs"]
    env_kwargs = dict(stage_config["env_kwargs"])
    horizon = int(env_kwargs.get("max_episode_steps", 1000))
    control_dt = float(env_kwargs.get("timestep", 0.002)) * int(env_kwargs.get("frame_skip", 5))
    # The lenient reader the stance report uses: the settle window of every stance gate kind, 0 where none is declared.
    settle_steps = StanceGateThresholds.from_curriculum(curriculum, require_criteria=False).settle_steps
    plant_identity = current_plant_identity(model_id)
    env_class = model_env_class(model_id)
    policy = load_trace_policy(
        model_path,
        vecnorm_path,
        lambda: env_class(**env_kwargs),
        plant_identity=plant_identity,
        allow_unnormalized=allow_unnormalized,
        allow_legacy_plant=allow_legacy_plant,
    )
    env = env_class(**env_kwargs)
    traces: list[EpisodeTrace] = []
    try:
        validate_environment_plant(env, plant_identity, artifact=f"{model_id} episode trace environment")
        layout = ActuatorLayout.from_model(env.unwrapped.model)
        selected = select_actuators(layout.names, actuators)
        hold_indices = _hold_indices(
            layout.names,
            holds,
            np.asarray(env.action_space.low, dtype=np.float64),
            np.asarray(env.action_space.high, dtype=np.float64),
        )
        for index, seed in enumerate(seeds):
            rng = None if stochastic_seed is None else np.random.default_rng(int(stochastic_seed) + index)
            traces.append(trace_episode(env, policy, seed=int(seed), layout=layout, holds=hold_indices, noise_rng=rng))
    finally:
        env.close()

    report = {
        "schema": TRACE_SCHEMA,
        **model_id.record(),
        "stage": stage,
        "gate_kind": curriculum.get("gate_kind"),
        "checkpoint": str(model_path),
        "vecnorm": None if vecnorm_path is None else str(vecnorm_path),
        "normalization": policy.normalization,
        "checkpoint_plant_validated": not allow_legacy_plant,
        # The judge's rollout only when both are absent: either one makes this a probe of a modified policy.
        "deterministic": stochastic_seed is None,
        "stochastic_seed": stochastic_seed,
        "holds": [hold.as_dict() for hold in holds],
        "seeds": [int(seed) for seed in seeds],
        "settle_steps": int(settle_steps),
        "horizon": horizon,
        "control_dt": control_dt,
        "saturation_abs": SATURATION_ABS,
        "near_saturation_abs": NEAR_SATURATION_ABS,
        "actuator_names": list(layout.names),
        "traced_actuators": [layout.names[index] for index in selected],
        "episodes": [
            summarize_episode(trace, names=layout.names, settle_steps=settle_steps, horizon=horizon, selected=selected)
            for trace in traces
        ],
    }
    return report, traces


def trace_filename(seed: int) -> str:
    """The per-episode CSV's name."""
    return f"episode_trace_seed{int(seed)}.csv"


def trace_rows(
    trace: EpisodeTrace, *, names: Sequence[str], selected: Sequence[int], settle_steps: int, control_dt: float
) -> tuple[list[str], list[dict[str, Any]]]:
    """``(fieldnames, rows)`` of one episode's CSV: one row per step, ``<actuator>_<field>`` per selected actuator.

    ``step`` is 0-based, the step index of the action (the replay CSV's
    ``step`` counts from 1); ``time_s`` is the time after the step and
    ``in_window`` whether the gate's window counts it.  A value that is not
    finite is an empty cell; the ``sample`` columns appear only in a
    stochastic trace.
    """
    fields = [name for name in TRACE_FIELDS if name != "sample" or np.isfinite(trace.values["sample"]).any()]
    fieldnames = ["step", "time_s", "in_window", "reward"]
    fieldnames += [f"{names[index]}_{name}" for index in selected for name in fields]
    rows: list[dict[str, Any]] = []
    for step in range(trace.length):
        row: dict[str, Any] = {
            "step": step,
            "time_s": (step + 1) * control_dt,
            "in_window": int(step >= settle_steps),
            "reward": float(trace.reward[step]),
        }
        for index in selected:
            for name in fields:
                value = float(trace.values[name][step, index])
                row[f"{names[index]}_{name}"] = value if math.isfinite(value) else ""
        rows.append(row)
    return fieldnames, rows


def run_directory(path: "str | Path") -> Path | None:
    """The run directory *path* lies in: its nearest ancestor (or itself) holding ``provenance.json``."""
    resolved = Path(path).resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "provenance.json").is_file():
            return candidate
    return None


def refuse_run_directory_output(out_dir: "str | Path", inputs: Sequence["str | Path | None"]) -> None:
    """Refuse an *out_dir* inside the run directory of any of *inputs* (as ``plot_gate_progress.py`` does).

    A result bundle's manifest hashes every file under the run directory
    (``result_bundle/manifest.py``) and verification rejects a file it did
    not list, so a trace written into a published run would make it fail.
    """
    target = Path(out_dir).resolve()
    for source in inputs:
        if source is None:
            continue
        run = run_directory(Path(source).parent)
        if run is not None and target.is_relative_to(run):
            raise EpisodeTraceError(
                f"refusing to write {target} inside the run directory {run}: the result bundle's manifest "
                "(result_bundle/manifest.py) hashes every file there, and verification rejects a file it did not "
                "list. Pass an output directory outside the run."
            )


def write_episode_trace(
    out_dir: "str | Path", report: dict[str, Any], traces: Sequence[EpisodeTrace]
) -> dict[str, Path]:
    """Write one CSV per episode (:func:`trace_filename`) and :data:`TRACE_SUMMARY_JSON`; returns them by name."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    names = report["actuator_names"]
    selected = [names.index(name) for name in report["traced_actuators"]]
    written: dict[str, Path] = {}
    for trace in traces:
        fieldnames, rows = trace_rows(
            trace,
            names=names,
            selected=selected,
            settle_steps=report["settle_steps"],
            control_dt=report["control_dt"],
        )
        name = trace_filename(trace.seed)
        written[name] = atomic_write_csv(destination / name, fieldnames, rows)
    written[TRACE_SUMMARY_JSON] = atomic_write_json(destination / TRACE_SUMMARY_JSON, report, allow_nan=False)
    return written


def _cell(value: Any, spec: str, width: int) -> str:
    """*value* formatted by *spec*, or ``-`` for ``None``, right-aligned in *width*."""
    return ("-" if value is None else format(value, spec)).rjust(width)


def render_episode_trace(report: dict[str, Any]) -> str:
    """The summary as text: a header, then per episode its outcome and one line per traced actuator."""
    model_id = ModelId.from_record(report)
    mode = (
        "deterministic"
        if report["deterministic"]
        else f"stochastic (policy noise, numpy default_rng({report['stochastic_seed']} + episode))"
    )
    lines = [
        f"Episode trace — {model_display_name(model_id)} / {stage_display_name(report['stage'])} (report only)",
        f"  checkpoint     {report['checkpoint']}",
        f"  normalization  {report['normalization']['mode']}",
        f"  rollout        {mode}",
    ]
    if report["holds"]:
        held = ", ".join(f"{hold['actuator']}={hold['value']:g}" for hold in report["holds"])
        lines.append(f"  held           {held} (a counterfactual probe, not the judge's rollout)")
    lines.append(
        f"  window         steps {report['settle_steps']}-end (0-based), saturated at |command| >= "
        f"{report['saturation_abs']}, near at >= {report['near_saturation_abs']}"
    )
    width = max([len(name) for name in report["traced_actuators"]] + [8])
    for episode in report["episodes"]:
        lines.append("")
        lines.append(
            f"seed {episode['seed']}: length {episode['length']}, return {episode['return']:.2f}, "
            f"{episode['termination_reason']}; gate saturation max {_cell(episode['max_actuator_saturation_fraction'], '.3f', 0)}"
        )
        lines.append(
            f"  {'actuator':<{width}}  {'cmd':>7}  {'raw':>7}  {'sat':>6}  {'>=0.9':>6}  {'raw>1':>6}  "
            f"{'rawmax':>7}  {'first':>6}  {'win1st':>6}  {'force':>6}"
        )
        for name, entry in episode["actuators"].items():
            cells = (
                _cell(entry["command_window_mean"], "+.3f", 7),
                _cell(entry["raw_mean_window_mean"], "+.3f", 7),
                _cell(entry["saturated_fraction"], ".3f", 6),
                _cell(entry["near_saturated_fraction"], ".3f", 6),
                _cell(entry["raw_mean_beyond_bound_fraction"], ".3f", 6),
                _cell(entry["raw_mean_max"], "+.3f", 7),
                _cell(entry["first_saturated_step"], "d", 6),
                _cell(entry["first_saturated_window_step"], "d", 6),
                _cell(entry["force_frac_window_mean"], ".3f", 6),
            )
            lines.append(f"  {name:<{width}}  " + "  ".join(cells))
    lines.append("")
    lines.append(
        "cmd/raw: window means of the command and of the raw Gaussian mean; sat/>=0.9/raw>1: shares of window steps; "
        "first/win1st: the first step at |command| >= saturation, anywhere and in the window; force: the window mean "
        "of |force| / its limit."
    )
    return "\n".join(lines) + "\n"
