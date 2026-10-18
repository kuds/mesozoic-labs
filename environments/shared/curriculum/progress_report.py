"""Write ``progress.json``, a small human-readable snapshot of a training stage, after every evaluation.

A stage's in-training record is spread over ``evaluations.npz``,
``gate_progress.npz``, ``diagnostics.npz``, TensorBoard and the training
output, and the first two are binary: reading a Colab run's posture mid-run
meant downloading them from Drive and opening them in numpy.  The T. rex
physics-r10 screen of 2026-10-16 was watched that way, and its two drifting
seeds were found by hand.  This callback writes one JSON file into the stage
directory, beside ``gate_progress.npz``, that a person can open from Drive:
the run's identity, its step count against its budget and projected finish,
the latest evaluation and how its episodes ended, the stage's posture
telemetry against the bars it declares, a few evaluations of history, the
checkpoints so far, and the advisories the run has raised.  Where the
evaluation callbacks record them, it also carries the in-training v2
screen (``evaluation.screen_v2``, never a verdict), the whole-body balance
and force telemetry (``balance``, ``forces``) and the achieved leg offsets
(``posture.leg_q_deg``); and from the training side, the policy probe's
latest update (``policy``) and the training episodes of the last 262,144
steps (``training_rollouts``).  A block whose source is absent is left out,
and the schema string changes only when a key is removed, renamed, or
changes meaning, units or type, never for an added one.

It is **report-only**.  It reads what the evaluation callbacks already
recorded (``StageAwareEvalCallback``'s evaluation history,
``StageGatePlateauCallback``'s latest ``gate_progress.npz`` row, through
``gate_progress_row``, the policy probe's ``latest()`` and the training
ledger's ``summary()``) and recomputes nothing; nothing reads the file back.
It is rewritten atomically (:func:`~environments.shared.file_io.atomic_write_json`,
a temporary file and ``os.replace``, safe on a Drive mount) at the start of
training, after every periodic evaluation and when training ends, and a
failure to write it is logged once and costs the file, never the run.  Its
schema is :data:`PROGRESS_SCHEMA`, documented in
``docs/TRAINING_ARTIFACT_PRESENTATION.md``.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ..model_id import ModelId
from . import sb3_compat
from .sb3_compat import BaseCallback

logger = logging.getLogger(__name__)

#: The file this callback writes into the stage directory.
PROGRESS_FILENAME = "progress.json"
#: The schema it records.
PROGRESS_SCHEMA = "mesozoic.training-progress/v1"
#: Evaluations kept in the ``history`` list, newest last.
HISTORY_EVALUATIONS = 5
#: Significant digits of the posture statistics (rewards keep six).
POSTURE_DIGITS = 4
#: Warnings kept in the ``warnings`` list, the newest; ``warnings_total`` counts them all.
WARNINGS_KEPT = 10

#: The rails whose ``gate_progress.npz`` statistics the ``posture`` block reports: ``(gate key, per-foot field of
#: the rail's own statistic, extra per-foot fields)``.
_POSTURE_RAILS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("max_sole_cop_fore_aft", "sole_cop_fore_aft_gate", ("sole_cop_fore_aft", "digit_share")),
    (
        "max_foot_cop_fore_aft",
        "foot_cop_fore_aft_gate",
        ("foot_cop_fore_aft", "toe_margin_nm", "heel_margin_nm"),
    ),
    ("max_sole_tilt_deg", "sole_tilt_deg", ()),
    ("max_sole_corner_lift_m", "sole_corner_lift_m", ()),
)
#: The ``gate_progress.npz`` prefix of the commanded leg DC keys (left-minus-right, and the mirror-even mean).
_LEG_DC_PREFIX = "leg_dc_"
#: The ``gate_progress.npz`` prefix of the achieved leg offsets (``gait.balance_telemetry.LEG_Q_PREFIX``).
_LEG_Q_PREFIX = "leg_q_"
#: The in-training v2 screen's history keys (``eval_diagnostics.SCREEN_KEY_PREFIX``).
_SCREEN_HISTORY_KEYS = ("screen_v2_clean_count", "screen_v2_clean_lcb")
#: The ``gate_progress.npz`` prefix of the termination-cause counts (``eval_diagnostics.TERMINATION_KEY_PREFIX``).
_TERMINATION_PREFIX = "termination_"
#: The ``training_rollouts`` entries that are counts (whole numbers); the rest are statistics.
_COUNT_KEYS = frozenset({"window_timesteps", "episodes", "falls", "episodes_total", "falls_total"})


def _iso(moment: datetime) -> str:
    """UTC, to the second, with a ``Z``: ``2026-10-16T20:20:01Z``."""
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _number(value: Any, digits: int = 6) -> float | int | None:
    """*value* as a JSON number kept to *digits* significant digits (an int stays an int); ``None`` when not finite."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return float(f"{number:.{digits}g}")


def _declared_bar(curriculum: Mapping[str, Any], key: str) -> float | None:
    value = curriculum.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return None
    return float(value)


def _feet_of(row: Mapping[str, float], field: str) -> list[str]:
    """The foot labels whose ``<label>_<field>`` key the row holds, in the row's (foot-major) order."""
    suffix = f"_{field}"
    return [key[: -len(suffix)] for key in row if key.endswith(suffix) and not key.startswith("max_")]


def posture_block(row: Mapping[str, float], curriculum: Mapping[str, Any]) -> dict[str, Any]:
    """The ``posture`` block of one ``gate_progress.npz`` row: each rail against its declared bar, per foot.

    Per rail present in the row (a model whose feet measure it): the bar the
    stage declares (``None`` when it declares none), the full-horizon mean of
    the episodes' own rail statistic (``<key>_mean``), the fraction of those
    episodes over the bar (``<key>_over_bar_fraction``, only with a bar) and
    per foot the full-horizon mean of the per-foot statistic (``gate``: the
    ``<label>_..._gate`` key, or the roll rail's own per-foot key, then a bare
    number), with the signed centre of pressure and the digit share or
    moment margins where the foot has them.  Then the left-minus-right leg
    DC per sagittal joint pair, in degrees.  Four significant digits.
    """
    out: dict[str, Any] = {}
    for key, per_foot_field, extras in _POSTURE_RAILS:
        if f"{key}_mean" not in row:
            continue
        bar = _declared_bar(curriculum, key)
        entry: dict[str, Any] = {"bar": bar, "mean": _number(row.get(f"{key}_mean"), POSTURE_DIGITS)}
        if bar is not None:
            entry["over_bar_fraction"] = _number(row.get(f"{key}_over_bar_fraction"), POSTURE_DIGITS)
        feet: dict[str, Any] = {}
        for label in _feet_of(row, per_foot_field):
            values = {"gate": _number(row.get(f"{label}_{per_foot_field}"), POSTURE_DIGITS)}
            for extra in extras:
                if f"{label}_{extra}" in row:
                    values[extra] = _number(row.get(f"{label}_{extra}"), POSTURE_DIGITS)
            feet[label] = values if len(values) > 1 else values["gate"]
        entry["per_foot"] = feet
        out[key] = entry
    for prefix, block in ((_LEG_DC_PREFIX, "leg_dc_deg"), (_LEG_Q_PREFIX, "leg_q_deg")):
        values = {
            key[len(prefix) :].removesuffix("_deg"): _number(value, POSTURE_DIGITS)
            for key, value in row.items()
            if key.startswith(prefix)
        }
        if values:
            out[block] = values
    return out


def balance_block(row: Mapping[str, float]) -> dict[str, Any]:
    """The ``balance`` block of one ``gate_progress.npz`` row: its balance scalars, four significant digits.

    The keys of ``gait.balance_telemetry.BALANCE_SCALAR_KEYS`` the row holds
    (the CoM and root pitch, the XcoM and its margins, Hof's split, the
    fall-only lead time's mean over the falls); the achieved leg offsets
    are ``posture.leg_q_deg``.  ``{}`` for a row without them.
    """
    from ..gait.balance_telemetry import BALANCE_SCALAR_KEYS

    return {key: _number(row[key], POSTURE_DIGITS) for key in BALANCE_SCALAR_KEYS if key in row}


def forces_block(row: Mapping[str, float]) -> dict[str, Any]:
    """The ``forces`` block of one row: the whole-animal force keys, then ``per_foot`` (``{}`` without them)."""
    from ..gait.balance_telemetry import ANKLE_FORCE_FIELD, FOOT_FORCE_FIELDS, FORCE_KEYS

    out: dict[str, Any] = {key: _number(row[key], POSTURE_DIGITS) for key in FORCE_KEYS if key in row}
    per_foot: dict[str, Any] = {}
    suffix = f"_{FOOT_FORCE_FIELDS[0]}"
    for label in [key[: -len(suffix)] for key in row if key.endswith(suffix)]:
        per_foot[label] = {
            field: _number(row[f"{label}_{field}"], POSTURE_DIGITS)
            for field in (ANKLE_FORCE_FIELD, *FOOT_FORCE_FIELDS)
            if f"{label}_{field}" in row
        }
    if per_foot:
        out["per_foot"] = per_foot
    return out


def screen_block(entry: Mapping[str, Any], curriculum: Mapping[str, Any]) -> dict[str, Any]:
    """``evaluation.screen_v2``: one evaluation's in-training v2 screen against the stage's bars.

    The stage's own ``stance_quality/v2`` bars at yaw 0 on the evaluation's
    episodes, without the statue-relative criteria (``skipped``):
    ``yaw0_bars_met`` is never a verdict, which only the post-stage
    certificate gives.
    """
    failing = {
        str(key): int(count) for key, count in dict(entry.get("criterion_failures") or {}).items() if int(count) > 0
    }
    out: dict[str, Any] = {
        "clean": entry.get("n_clean"),
        "episodes": entry.get("n_episodes"),
        "clean_lcb": _number(entry.get("clean_lcb"), POSTURE_DIGITS),
        "bar": _declared_bar(curriculum, "min_clean_stance_lcb"),
        "yaw0_bars_met": bool(entry.get("passed")),
    }
    if entry.get("hop_or_fall_episodes") is not None:
        out["hop_or_fall"] = int(entry["hop_or_fall_episodes"])
        out["hop_or_fall_bar"] = _declared_bar(curriculum, "max_hop_or_fall_episodes")
    out["failing"] = failing
    out["skipped"] = [str(key) for key in entry.get("skipped_criteria") or ()]
    return out


def termination_counts(row: Mapping[str, float]) -> dict[str, int]:
    """The row's episode count per termination cause, the causes no episode ended by left out."""
    out = {}
    for key, value in row.items():
        if key.startswith(_TERMINATION_PREFIX) and math.isfinite(value) and value > 0:
            out[key[len(_TERMINATION_PREFIX) :]] = int(value)
    return out


def _periodic_checkpoint_steps(model_dir: Path, name_prefix: str) -> list[int]:
    """Every periodic checkpoint's step count in *model_dir*, sorted, by its name (never its mtime)."""
    from ..policy_loading import _PERIODIC_CHECKPOINT_RE

    steps = []
    for path in model_dir.glob(f"{name_prefix}_*_steps.zip"):
        match = _PERIODIC_CHECKPOINT_RE.fullmatch(path.stem)
        if match is not None and match.group(1) == name_prefix:
            steps.append(int(match.group(2)))
    return sorted(steps)


def _first_evaluation_scoring(results: Sequence[Any], timesteps: Sequence[Any], best: float, score: Any) -> int | None:
    """The step of the first evaluation whose ``score(rewards)`` equals *best* (the one its checkpoint was saved at)."""
    if not math.isfinite(best):
        return None
    for rewards, step in zip(results, timesteps):
        if score(np.asarray(rewards, dtype=float)) == best:
            return int(step)
    return None


class TrainingProgressCallback(BaseCallback):  # type: ignore[misc]
    """Rewrite ``progress.json`` in the stage directory at the start, after each evaluation and at the end.

    Args:
        eval_callback: The ``StageAwareEvalCallback`` whose evaluation history
            (``evaluations_timesteps``, ``_results``, ``_length``) to read.
        plateau_callback: The ``StageGatePlateauCallback`` whose latest
            ``gate_progress.npz`` row (``gate_progress_row``) to read, and the
            horizon a fall is shorter than.  It runs earlier in the same
            callback list.
        stage_dir: The stage directory, where the file goes.
        model_dir: Its ``models/`` directory (the published checkpoints).
        stage: The stage reference the trainer was given.
        model_id: The model trained (``None``: unknown; the identity reads null).
        seed: The run's seed (``None``: not recorded).
        total_timesteps: The cumulative step count this session trains to.
        curriculum_kwargs: The stage's ``[curriculum]`` table: its declared bars.
        warning_sources: Callbacks whose ``progress_warnings`` lists to report.
        robust_callback: The ``RobustBestModelCallback``, for the robust best step.
        policy_probe: The ``PolicyProbeCallback`` whose ``latest()`` is the ``policy`` block (``None``: none).
        rollout_ledger: The ``TrainingEpisodeLedgerCallback`` whose ``summary()`` is the
            ``training_rollouts`` block (``None``: none).
    """

    #: Class-scope defaults so an instance built without ``__init__`` (as the
    #: tests do, to exercise the writer without stable-baselines3) has them.
    _last_seen_n_evals = 0
    _warned_unwritable = False
    _session_start: "tuple[float, datetime, int] | None" = None
    policy_probe: Any = None
    rollout_ledger: Any = None

    def __init__(
        self,
        eval_callback: Any,
        plateau_callback: Any,
        *,
        stage_dir: "str | Path",
        model_dir: "str | Path",
        stage: "int | str",
        model_id: ModelId | None,
        seed: int | None,
        total_timesteps: int | None,
        curriculum_kwargs: Mapping[str, Any],
        warning_sources: Sequence[Any] = (),
        robust_callback: Any = None,
        policy_probe: Any = None,
        rollout_ledger: Any = None,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for TrainingProgressCallback.")
        super().__init__(verbose)
        self.eval_callback = eval_callback
        self.plateau_callback = plateau_callback
        self.stage_dir = Path(stage_dir)
        self.model_dir = Path(model_dir)
        self.stage = stage
        self.model_id = model_id
        self.seed = None if seed is None else int(seed)
        self.total_timesteps = None if total_timesteps is None else int(total_timesteps)
        self.curriculum_kwargs = dict(curriculum_kwargs)
        self.warning_sources = list(warning_sources)
        self.robust_callback = robust_callback
        self.policy_probe = policy_probe
        self.rollout_ledger = rollout_ledger
        self._last_seen_n_evals = 0
        self._warned_unwritable = False
        self._session_start = None

    @property
    def path(self) -> Path:
        """Where the file is written: ``<stage_dir>/progress.json``."""
        return Path(self.stage_dir) / PROGRESS_FILENAME

    # ------------------------------------------------------------------ SB3 hooks
    def _on_training_start(self) -> None:
        self._session_start = (time.monotonic(), datetime.now(timezone.utc), int(self.num_timesteps))
        self.write("training")

    def _on_step(self) -> bool:
        results = getattr(self.eval_callback, "evaluations_results", None) or []
        if len(results) > self._last_seen_n_evals:
            self._last_seen_n_evals = len(results)
            self.write("training")
        return True

    def _on_training_end(self) -> None:
        self.write("ended")

    # ------------------------------------------------------------------ the record
    def write(self, status: str) -> None:
        """Rewrite the file with the current state; a failure is logged once and never raised."""
        try:
            from ..file_io import atomic_write_json

            atomic_write_json(self.path, self.record(status), allow_nan=False)
        except Exception:  # noqa: BLE001 - a report must not sink a run
            if not self.__dict__.get("_warned_unwritable", False):
                self._warned_unwritable = True
                logger.warning("Could not write %s", self.path, exc_info=True)

    def record(self, status: str) -> dict[str, Any]:
        """The ``progress.json`` object (:data:`PROGRESS_SCHEMA`): small, finite, JSON-ready."""
        now = datetime.now(timezone.utc)
        timesteps = int(getattr(self, "num_timesteps", 0) or 0)
        budget = self.total_timesteps
        record: dict[str, Any] = {
            "schema": PROGRESS_SCHEMA,
            "status": status,
            "written_at": _iso(now),
            **self._identity(),
            "seed": self.seed,
            "timesteps": timesteps,
            "budget_timesteps": budget,
            "fraction_done": _number(timesteps / budget, 4) if budget else None,
            **self._clock(now, timesteps),
            "zero_action_reward": _number(self._zero_action_reward()),
        }
        row = self._gate_row()
        record["evaluation"] = self._latest_evaluation(row)
        record["posture"] = posture_block(row, self.curriculum_kwargs) if row else {}
        # The balance and force telemetry (a block whose source is absent is omitted, never null-filled).
        for name, block in (("balance", balance_block), ("forces", forces_block)):
            values = block(row) if row else {}
            if values:
                record[name] = values
        # The training side: the policy probe's latest update and the recent training episodes.
        for name, values in (("policy", self._policy()), ("training_rollouts", self._training_rollouts(timesteps))):
            if values:
                record[name] = values
        record["history"] = self._history()
        record["checkpoints"] = self._checkpoints()
        warnings = self._warnings()
        record["warnings_total"] = len(warnings)
        record["warnings"] = warnings[-WARNINGS_KEPT:]
        return record

    def _identity(self) -> dict[str, Any]:
        model_id = self.model_id
        stage_id: str | None
        try:
            from ..stage_manifest import resolve_stage_key, stage_label

            stage_id = resolve_stage_key(model_id, self.stage).id if model_id is not None else stage_label(self.stage)
        except Exception:  # noqa: BLE001 - a report must not sink a run
            stage_id = str(self.stage)
        display = None
        if model_id is not None:
            try:
                from ..species_names import model_display_name

                display = model_display_name(model_id)
            except Exception:  # noqa: BLE001 - a report must not sink a run
                display = None
        run_dir = Path(self.stage_dir).parent
        return {
            "species": None if model_id is None else model_id.species,
            "embodiment": None if model_id is None else model_id.embodiment,
            "model": display,
            "stage": stage_id,
            "run_id": self._run_id(run_dir),
            "run_dir": run_dir.name,
            "stage_dir": Path(self.stage_dir).name,
        }

    @staticmethod
    def _run_id(run_dir: Path) -> str | None:
        """The ``run_id`` the run directory's ``provenance.json`` records, or ``None`` (a run without a bundle)."""
        from ..file_io import read_json_object

        value = (read_json_object(run_dir / "provenance.json") or {}).get("run_id")
        return value if isinstance(value, str) and value else None

    def _clock(self, now: datetime, timesteps: int) -> dict[str, Any]:
        start = self.__dict__.get("_session_start")
        if start is None:
            return {"session_started_at": None, "wall_clock_s": None, "steps_per_s": None, "projected_finish": None}
        started_monotonic, started_at, started_steps = start
        elapsed = max(time.monotonic() - started_monotonic, 0.0)
        rate = (timesteps - started_steps) / elapsed if elapsed > 0 and timesteps > started_steps else None
        finish = None
        if rate and self.total_timesteps is not None:
            remaining = max(self.total_timesteps - timesteps, 0)
            finish = _iso(now + timedelta(seconds=remaining / rate))
        return {
            "session_started_at": _iso(started_at),
            "wall_clock_s": _number(elapsed, 7),
            "steps_per_s": _number(rate, 4),
            "projected_finish": finish,
        }

    def _zero_action_reward(self) -> float | None:
        if self.model_id is None:
            return None
        cached = self.__dict__.get("_zero_action_cache")
        if cached is None:
            from .baseline_watch import read_zero_action_baseline

            try:
                value = read_zero_action_baseline(Path(self.stage_dir).parent, self.model_id)
            except Exception:  # noqa: BLE001 - a report must not sink a run
                value = None
            cached = self._zero_action_cache = (value,)
        return cached[0]

    def _gate_row(self) -> dict[str, float]:
        accessor = getattr(self.plateau_callback, "gate_progress_row", None)
        if not callable(accessor):
            return {}
        try:
            return dict(accessor())
        except Exception:  # noqa: BLE001 - a report must not sink a run
            return {}

    def _horizon(self) -> int | None:
        horizon = getattr(self.plateau_callback, "horizon", None)
        return int(horizon) if isinstance(horizon, (int, np.integer)) and horizon > 0 else None

    def _evaluation_at(self, index: int) -> dict[str, Any]:
        """One evaluation's summary from the evaluation history: rewards, lengths and falls."""
        source = self.eval_callback
        rewards = np.asarray(source.evaluations_results[index], dtype=float).reshape(-1)
        lengths = np.asarray(source.evaluations_length[index], dtype=float).reshape(-1)
        horizon = self._horizon()
        out: dict[str, Any] = {
            "timesteps": int(source.evaluations_timesteps[index]),
            "episodes": int(rewards.size),
            "mean_reward": _number(rewards.mean()) if rewards.size else None,
            "std_reward": _number(rewards.std()) if rewards.size else None,
            "mean_episode_length": _number(lengths.mean()) if lengths.size else None,
        }
        if horizon is not None and lengths.size:
            out["falls"] = int(np.count_nonzero(lengths < horizon))
        return out

    def _latest_evaluation(self, row: Mapping[str, float]) -> dict[str, Any] | None:
        source = self.eval_callback
        if not getattr(source, "evaluations_results", None):
            return None
        index = len(source.evaluations_results) - 1
        latest = self._evaluation_at(index)
        if row and int(row.get("timesteps", -1)) == latest["timesteps"]:
            latest["full_horizon_fraction"] = _number(row.get("full_horizon_fraction"))
            causes = termination_counts(row)
            if causes:
                latest["termination_causes"] = causes
        screens = getattr(source, "evaluations_screen_v2", None) or []
        entry = screens[index] if index < len(screens) else None
        if isinstance(entry, Mapping):
            latest["screen_v2"] = screen_block(entry, self.curriculum_kwargs)
        return latest

    def _history(self) -> list[dict[str, Any]]:
        source = self.eval_callback
        results = getattr(source, "evaluations_results", None) or []
        accessor = getattr(self.plateau_callback, "gate_progress_series", None)
        latest_row = self._gate_row()
        keys = [
            key
            for key in (
                "full_horizon_fraction",
                *(f"{rail}_{stat}" for rail, _f, _x in _POSTURE_RAILS for stat in ("mean", "over_bar_fraction")),
                *_SCREEN_HISTORY_KEYS,
            )
            if key in latest_row
        ]
        series = {key: accessor(key) for key in keys} if callable(accessor) else {}
        history = []
        for index in range(max(0, len(results) - HISTORY_EVALUATIONS), len(results)):
            entry = self._evaluation_at(index)
            for key, (steps, values) in series.items():
                for step, value in zip(steps, values):
                    if int(step) == entry["timesteps"]:
                        entry[key] = _number(value, POSTURE_DIGITS)
            history.append(entry)
        return history

    def _checkpoints(self) -> dict[str, Any]:
        from ..stage_manifest import stage_label

        try:
            periodic = _periodic_checkpoint_steps(Path(self.model_dir), stage_label(self.stage))
        except Exception:  # noqa: BLE001 - a report must not sink a run
            periodic = []
        source = self.eval_callback
        results = getattr(source, "evaluations_results", None) or []
        timesteps = getattr(source, "evaluations_timesteps", None) or []
        best = float(getattr(source, "best_mean_reward", -math.inf))
        out: dict[str, Any] = {
            "latest_periodic_timesteps": periodic[-1] if periodic else None,
            # Every periodic checkpoint kept: the newest few and, with the archive rule, one per interval.
            "periodic_timesteps": periodic,
            "best_model_timesteps": _first_evaluation_scoring(results, timesteps, best, lambda r: float(np.mean(r))),
            "best_mean_reward": _number(best),
        }
        robust = self.robust_callback
        if robust is not None:
            risk = float(getattr(robust, "risk_coef", 1.0))
            score = float(getattr(robust, "best_score", -math.inf))
            out["robust_best_model_timesteps"] = _first_evaluation_scoring(
                results, timesteps, score, lambda r: float(r.mean() - risk * r.std())
            )
            out["robust_best_score"] = _number(score)
        return out

    def _policy(self) -> dict[str, Any]:
        """``policy``: the probe's latest row (its ``policy_probe/*`` scalars and ``update_timesteps``)."""
        latest = getattr(self.policy_probe, "latest", None)
        if not callable(latest):
            return {}
        try:
            values = dict(latest())
        except Exception:  # noqa: BLE001 - a report must not sink a run
            return {}
        return {
            key: int(value) if key == "update_timesteps" else _number(value, POSTURE_DIGITS)
            for key, value in values.items()
        }

    def _training_rollouts(self, timesteps: int) -> dict[str, Any]:
        """``training_rollouts``: the training ledger's summary of the episodes that ended in its window."""
        summary = getattr(self.rollout_ledger, "summary", None)
        if not callable(summary):
            return {}
        try:
            values = dict(summary(timesteps))
        except Exception:  # noqa: BLE001 - a report must not sink a run
            return {}
        out: dict[str, Any] = {}
        for key, value in values.items():
            if key == "touch_shares":
                out[key] = {
                    str(label): {str(stem): _number(share, POSTURE_DIGITS) for stem, share in dict(shares).items()}
                    for label, shares in dict(value).items()
                }
            elif key in _COUNT_KEYS:
                out[key] = int(value)
            else:
                out[key] = _number(value, POSTURE_DIGITS)
        return out

    def _warnings(self) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        for source in self.warning_sources:
            for warning in getattr(source, "progress_warnings", None) or []:
                if isinstance(warning, Mapping):
                    warnings.append(dict(warning))
        return sorted(warnings, key=lambda warning: int(warning.get("timesteps", 0)))


def build_training_progress_callback(
    eval_callback: Any,
    plateau_callback: Any,
    *,
    stage_dir: "str | Path | None",
    model_dir: "str | Path",
    stage: "int | str",
    model_id: ModelId | None,
    seed: int | None,
    total_timesteps: int | None,
    stage_config: Mapping[str, Any],
    warning_sources: Sequence[Any] = (),
    robust_callback: Any = None,
    policy_probe: Any = None,
    rollout_ledger: Any = None,
    verbose: int = 0,
) -> TrainingProgressCallback | None:
    """Build the writer, or ``None`` without a stage directory to write into."""
    if stage_dir is None:
        return None
    return TrainingProgressCallback(
        eval_callback,
        plateau_callback,
        stage_dir=stage_dir,
        model_dir=model_dir,
        stage=stage,
        model_id=model_id,
        seed=seed,
        total_timesteps=total_timesteps,
        curriculum_kwargs=stage_config.get("curriculum_kwargs", {}) or {},
        warning_sources=[source for source in warning_sources if source is not None],
        robust_callback=robust_callback,
        policy_probe=policy_probe,
        rollout_ledger=rollout_ledger,
        verbose=verbose,
    )
