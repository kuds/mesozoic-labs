"""One CSV row per training episode: its posture, its touch shares and its limited force, report-only.

The evaluation episodes (``eval_episodes.csv``) are the deterministic
policy's.  The falls a stance run learns from are its training episodes:
15-17% of one T. rex seed's training episodes ended early at 4.7-5.2M steps
while its evaluations stood.  Nothing recorded them beyond a per-rollout
termination fraction.  :class:`TrainingEpisodeLedgerCallback` writes one row
per training episode to ``<stage_dir>/train_episodes.csv``:

* the outcome: ``timesteps`` (the run's step count when the episode ended),
  ``env``, ``length`` and ``return`` (Monitor's), and ``cause``
  (``truncated``, the env's ``termination_reason``, or ``terminated``);
* the posture over the post-settle window: ``com_fore_aft_mean_m`` (the
  whole animal's centre of mass ahead of the mean foot reference site along
  the heading, m) and ``root_pitch_mean_deg`` (the root's forward axis above
  horizontal, + nose up), and the same at the last reading before the end,
  ``com_fore_aft_last_m`` and ``root_pitch_last_deg``;
* ``actuator_force_limited_fraction``: the window fraction of steps on which
  some force-limited actuator is at 0.99 of its limit or more, the
  evaluation key's definition (``balance_telemetry.actuator_limited_fraction``);
* on a foot with a sole, one touch sensor on the sole's body and at least one
  on a hinged digit (the T. rex), each sensor's share of the foot's summed
  touch readings over the window, ``<label>_touch_share_pad``,
  ``<label>_touch_share_<stem>`` per digit sensor (``r_toe_d2_touch`` ->
  ``toe_d2``) and ``<label>_touch_share_digits``: the four-sensor proxy of the
  whole-foot centre of pressure (R^2 0.91 against floor truth).  No column
  ends in a foot rail's suffix (``_digit_share``): readers that enumerate
  those take such a key for a foot.

Per step it only copies raw ``mjData`` reads of each in-process env that is
not done (``subtree_com`` and ``xquat`` of the root, the feet's reference
sites, those touch sensors and ``actuator_force``); the heading frames and
the reductions run once per episode, vectorised, when the row closes.  When
an env reports ``done`` it has already reset, so its data is the next
episode's: the terminal state is never read, and the window is the
post-step readings past ``settle_steps``, the terminal one excluded.  A
SubprocVecEnv's envs live in other processes: their rows keep the outcome
and read NaN elsewhere.

Report-only: the callback reads the env and writes nothing to it, draws no
random number, and returns ``True`` from every step.  A failure costs the
telemetry of that env's episode (NaN, one warning), never the run.  The file
is rewritten atomically on the checkpoint cadence and at training end, and a
same-stage resume keeps the prior session's rows at or before the resumed
step (``seed_prior_record``).  TensorBoard gets per-rollout means under
``rollout_balance/`` and ``progress.json`` a summary of the last 262,144
steps (:meth:`TrainingEpisodeLedgerCallback.summary`).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .curriculum import sb3_compat
from .curriculum.sb3_compat import BaseCallback
from .episode_ledger import EpisodeLedger
from .model_id import ModelId

logger = logging.getLogger(__name__)

#: The file the callback writes into the stage directory.
TRAIN_LEDGER_FILENAME = "train_episodes.csv"
#: The columns every row has, in this order; the touch-share columns follow.
TRAIN_LEDGER_FIXED_COLUMNS: tuple[str, ...] = (
    "timesteps",
    "env",
    "length",
    "return",
    "cause",
    "com_fore_aft_mean_m",
    "com_fore_aft_last_m",
    "root_pitch_mean_deg",
    "root_pitch_last_deg",
    "actuator_force_limited_fraction",
)
#: The ``mjData`` columns: NaN when the env could not be read.
_MEASURED_COLUMNS: tuple[str, ...] = TRAIN_LEDGER_FIXED_COLUMNS[5:]
#: The ``progress.json`` summary's window: the episodes that ended in the last this many steps.
DEFAULT_WINDOW_TIMESTEPS = 262_144
#: The touch-share column infix, and the stem of the sole's own sensor.
TOUCH_SHARE_INFIX = "_touch_share_"
PAD_STEM = "pad"
DIGITS_STEM = "digits"

NAN = float("nan")


@dataclass(frozen=True)
class FootTouch:
    """One foot's touch-share sensors: the sole's, then each hinged digit's."""

    label: str
    #: ``pad``, then each digit sensor's stem (its name less the foot's label prefix and ``_touch``).
    stems: tuple[str, ...]
    #: Their ``sensordata`` addresses, in the same order.
    addresses: tuple[int, ...]

    @property
    def columns(self) -> tuple[str, ...]:
        """``<label>_touch_share_<stem>`` per sensor, then ``<label>_touch_share_digits``."""
        return tuple(f"{self.label}{TOUCH_SHARE_INFIX}{stem}" for stem in (*self.stems, DIGITS_STEM))


@dataclass(frozen=True)
class TouchLayout:
    """Which feet get touch-share columns, and from which sensors (:meth:`from_morphology`)."""

    feet: tuple[FootTouch, ...] = ()

    @classmethod
    def from_morphology(cls, env: Any, morphology: Any) -> TouchLayout:
        """Classify each foot's touch sensors by the body of the sensor's site.

        A sensor on the sole geom's body is the pad's; one on the body of a
        hinged digit (a support geom with ``Morphology.support_geom_mtp >= 0``)
        is a digit's; any other (a metatarsal sensor) is neither.  A foot
        gets columns only when it has a sole (``sole_geom >= 0``: indexing
        ``geom_bodyid[-1]`` would read the last geom's body), exactly one pad
        sensor and at least one digit sensor: on this repository's models,
        the T. rex only.
        """
        import mujoco

        model = getattr(env, "unwrapped", env).model
        touch = int(mujoco.mjtSensor.mjSENS_TOUCH)
        site = int(mujoco.mjtObj.mjOBJ_SITE)
        feet = []
        for index, foot in enumerate(morphology.feet):
            if foot.sole_geom < 0:
                continue
            sole_body = int(model.geom_bodyid[foot.sole_geom])
            digit_bodies = {
                int(model.geom_bodyid[geom])
                for geom, owner, joint in zip(
                    morphology.support_geom_ids, morphology.support_geom_foot, morphology.support_geom_mtp
                )
                if owner == index and joint >= 0
            }
            pads: list[tuple[str, int]] = []
            digits: list[tuple[str, int]] = []
            for name in foot.sensor_names:
                sensor = model.sensor(name)
                if (
                    int(model.sensor_type[sensor.id]) != touch
                    or int(model.sensor_objtype[sensor.id]) != site
                    or int(model.sensor_dim[sensor.id]) != 1
                ):
                    continue
                body = int(model.site_bodyid[int(model.sensor_objid[sensor.id])])
                address = int(model.sensor_adr[sensor.id])
                if body == sole_body:
                    pads.append((PAD_STEM, address))
                elif body in digit_bodies:
                    digits.append((_sensor_stem(str(name), foot.label), address))
            if len(pads) != 1 or not digits:
                continue
            stems, addresses = zip(*(pads + digits))
            feet.append(FootTouch(label=foot.label, stems=tuple(stems), addresses=tuple(addresses)))
        return cls(feet=tuple(feet))

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(column for foot in self.feet for column in foot.columns)

    @property
    def addresses(self) -> tuple[int, ...]:
        return tuple(address for foot in self.feet for address in foot.addresses)


def _sensor_stem(name: str, label: str) -> str:
    """``r_toe_d2_touch`` on foot ``r`` -> ``toe_d2``."""
    stem = name.removesuffix("_touch")
    return stem[len(label) + 1 :] if stem.startswith(f"{label}_") else stem


@dataclass(frozen=True)
class _EnvLayout:
    """What the per-step copy reads of one env, and how a row's columns split it."""

    root_body: int
    sites: np.ndarray  # (F,) the feet's reference sites
    touch: TouchLayout
    touch_addresses: np.ndarray  # the touch layout's sensordata addresses, as an index array
    force_limits: np.ndarray  # (nu, 2)

    @property
    def width(self) -> int:
        return 7 + 3 * len(self.sites) + len(self.touch.addresses) + int(self.force_limits.shape[0])


def episode_columns(readings: np.ndarray, layout: _EnvLayout, *, settle_steps: int) -> dict[str, float]:
    """The measured columns of one episode from its (L, width) post-step readings, the terminal one excluded.

    The window is the readings past ``settle_steps`` (reading ``k`` follows
    step ``k + 1``); ``_last`` is the last reading.  NaN where there is
    nothing to reduce.
    """
    from .gait.balance_telemetry import actuator_limited_fraction, heading_frames

    readings = np.asarray(readings, dtype=np.float64).reshape(-1, layout.width)
    out: dict[str, float] = {column: NAN for column in (*_MEASURED_COLUMNS, *layout.touch.columns)}
    if not readings.shape[0]:
        return out
    n_feet = len(layout.sites)
    com = readings[:, 0:3]
    forward, pitch = heading_frames(readings[:, 3:7])
    feet_point = readings[:, 7 : 7 + 3 * n_feet].reshape(-1, n_feet, 3).mean(axis=1)
    com_ahead = np.einsum("tj,tj->t", forward, (com - feet_point)[:, :2])
    start = 7 + 3 * n_feet
    n_touch = len(layout.touch.addresses)
    touch = readings[:, start : start + n_touch]
    force = readings[:, start + n_touch :]
    window = slice(int(settle_steps), None)
    out["com_fore_aft_last_m"] = float(com_ahead[-1])
    out["root_pitch_last_deg"] = float(pitch[-1])
    if readings.shape[0] > settle_steps:
        out["com_fore_aft_mean_m"] = float(com_ahead[window].mean())
        out["root_pitch_mean_deg"] = float(pitch[window].mean())
        out["actuator_force_limited_fraction"] = actuator_limited_fraction(force[window], layout.force_limits)
        sums = touch[window].sum(axis=0)
        offset = 0
        for foot in layout.touch.feet:
            foot_sums = sums[offset : offset + len(foot.addresses)]
            offset += len(foot.addresses)
            total = float(foot_sums.sum())
            if total > 0.0:
                for column, value in zip(foot.columns, (*foot_sums, foot_sums[1:].sum())):
                    out[column] = float(value) / total
    return out


def _number(value: Any) -> float:
    """A ledger cell as a float (a seeded row holds text); NaN for an empty or unreadable one."""
    if isinstance(value, bool):
        return NAN
    try:
        return float(value)
    except (TypeError, ValueError):
        return NAN


def _finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else NAN


class TrainingEpisodeLedgerCallback(BaseCallback):  # type: ignore[misc]
    """Write ``train_episodes.csv``: one row per training episode (see the module docstring).

    Args:
        model_id: The model trained (its feet, sites and touch sensors).
        settle_steps: The stage's settle window (``[curriculum] settle_steps``):
            the window means start after it.
        ledger_dir: The stage directory, where the file goes (``None``: kept in memory).
        flush_every_calls: The ``CheckpointCallback``'s ``save_freq`` (already
            divided by ``n_envs``): the file is rewritten on the calls a
            checkpoint is saved on, and at training end.
        window_timesteps: The :meth:`summary`'s window.
    """

    def __init__(
        self,
        *,
        model_id: ModelId,
        settle_steps: int = 0,
        ledger_dir: str | Path | None,
        flush_every_calls: int,
        window_timesteps: int = DEFAULT_WINDOW_TIMESTEPS,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for TrainingEpisodeLedgerCallback.")
        super().__init__(verbose)
        if int(flush_every_calls) < 1:
            raise ValueError("flush_every_calls must be at least 1")
        self.model_id = model_id
        self.settle_steps = int(settle_steps)
        self.flush_every_calls = int(flush_every_calls)
        self.window_timesteps = int(window_timesteps)
        path = None if ledger_dir is None else Path(ledger_dir) / TRAIN_LEDGER_FILENAME
        self._ledger = EpisodeLedger(path, fixed_columns=TRAIN_LEDGER_FIXED_COLUMNS)
        self._envs: list[Any] | None = None
        self._layouts: list[_EnvLayout | None] = []
        self._buffers: list[list[np.ndarray]] = []
        self._steps: list[int] = []
        self._failed: list[bool] = []
        self._readers: list[tuple[Any, int, np.ndarray, np.ndarray] | None] = []
        self._rollout_rows: list[dict[str, Any]] = []
        self._episodes_total = 0
        self._falls_total = 0
        self._warned: set[str] = set()

    # ------------------------------------------------------------------ helpers
    def _warn_once(self, key: str, message: str, *, exc_info: bool = False) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, exc_info=exc_info)

    @property
    def ledger(self) -> EpisodeLedger:
        """The rows held (for tests and the progress report)."""
        return self._ledger

    def _layout_of(self, env: Any) -> _EnvLayout:
        from .gait.balance_telemetry import actuator_force_limits
        from .gait.morphology import Morphology

        unwrapped = getattr(env, "unwrapped", env)
        morphology = Morphology.from_env(unwrapped, self.model_id)
        touch = TouchLayout.from_morphology(unwrapped, morphology)
        return _EnvLayout(
            root_body=int(morphology.root_body),
            sites=np.array([foot.primary_site for foot in morphology.feet], dtype=np.int64),
            touch=touch,
            touch_addresses=np.array(touch.addresses, dtype=np.int64),
            force_limits=actuator_force_limits(unwrapped.model),
        )

    # ------------------------------------------------------------------ SB3 hooks
    def _on_training_start(self) -> None:
        from .eval_diagnostics import _in_process_envs

        n_envs = int(getattr(self.training_env, "num_envs", 1) or 1)
        self._envs = _in_process_envs(self.training_env)
        if self._envs is None:
            self._warn_once(
                "subproc",
                "Training-episode ledger: the training envs are not in this process (a SubprocVecEnv); its rows "
                "keep each episode's outcome and read NaN for the posture, touch and force columns.",
            )
            self._layouts = [None] * n_envs
        else:
            self._layouts = []
            for index, env in enumerate(self._envs):
                try:
                    self._layouts.append(self._layout_of(env))
                except Exception:  # noqa: BLE001 - a report must not sink a run
                    self._warn_once(
                        "layout",
                        f"Training-episode ledger: could not describe training env {index}; its rows read NaN for "
                        "the posture, touch and force columns.",
                        exc_info=True,
                    )
                    self._layouts.append(None)
        self._buffers = [[] for _ in self._layouts]
        self._steps = [0] * len(self._layouts)
        # An env without a layout is never read: it counts as failed for the whole session.
        self._failed = [layout is None for layout in self._layouts]
        self._readers = [
            None
            if layout is None or self._envs is None
            else (
                getattr(self._envs[index], "unwrapped", self._envs[index]),
                layout.root_body,
                layout.sites,
                layout.touch_addresses,
            )
            for index, layout in enumerate(self._layouts)
        ]

    def _read(self, index: int) -> None:
        """Copy env *index*'s post-step reads into its episode buffer: copies only, no frame algebra.

        ``data`` is looked up on every call (never cached): an env that
        rebuilt its ``MjData`` is read from the new one.
        """
        reader = self._readers[index]
        assert reader is not None  # _on_step reads no env without one (it is marked failed)
        unwrapped, root, sites, addresses = reader
        data = unwrapped.data
        self._buffers[index].append(
            np.concatenate(
                (
                    data.subtree_com[root],
                    data.xquat[root],
                    data.site_xpos[sites].ravel(),
                    data.sensordata[addresses],
                    data.actuator_force,
                )
            )
        )

    def _close(self, index: int, info: Mapping[str, Any]) -> dict[str, Any]:
        """The row of env *index*'s episode that just ended, from its buffer and its last ``info``."""
        info = info if isinstance(info, Mapping) else {}
        episode = info.get("episode")
        buffer = self._buffers[index]
        length: Any = self._steps[index] + 1  # Monitor's count when there is no Monitor
        reward: Any = NAN
        if isinstance(episode, Mapping):
            length = int(episode.get("l", length))
            reward = float(episode.get("r", NAN))
        if info.get("TimeLimit.truncated"):
            cause = "truncated"
        else:
            reason = info.get("termination_reason")
            cause = str(reason) if reason else "terminated"
        row: dict[str, Any] = {
            "timesteps": int(self.num_timesteps),
            "env": index,
            "length": length,
            "return": reward,
            "cause": cause,
        }
        layout = self._layouts[index] if index < len(self._layouts) else None
        measured: dict[str, float] = {column: NAN for column in _MEASURED_COLUMNS}
        if layout is not None:
            measured.update({column: NAN for column in layout.touch.columns})
            if not self._failed[index] and buffer:
                try:
                    measured = episode_columns(np.stack(buffer), layout, settle_steps=self.settle_steps)
                except Exception:  # noqa: BLE001 - a report must not sink a run
                    self._warn_once(
                        "close",
                        "Training-episode ledger: could not reduce an episode; its row reads NaN for the posture, "
                        "touch and force columns.",
                        exc_info=True,
                    )
        row.update(measured)
        return row

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos") or []
        if dones is None:
            return True
        rows = []
        steps, failed = self._steps, self._failed
        flags = dones.tolist() if isinstance(dones, np.ndarray) else list(dones)
        for index, done in enumerate(flags[: len(steps)]):
            if done:
                info = infos[index] if index < len(infos) else {}
                try:
                    rows.append(self._close(index, info))
                except Exception:  # noqa: BLE001 - a report must not sink a run
                    self._warn_once("row", "Training-episode ledger: could not close a row.", exc_info=True)
                self._buffers[index] = []
                steps[index] = 0
                failed[index] = self._layouts[index] is None
                continue
            steps[index] += 1
            if failed[index]:
                continue
            try:
                self._read(index)
            except Exception:  # noqa: BLE001 - a report must not sink a run
                failed[index] = True
                self._buffers[index] = []
                self._warn_once(
                    "read",
                    f"Training-episode ledger: could not read training env {index}; that episode's row reads "
                    "NaN for the posture, touch and force columns.",
                    exc_info=True,
                )
        if rows:
            self._ledger.extend(rows)
            self._rollout_rows.extend(rows)
            self._episodes_total += len(rows)
            self._falls_total += sum(1 for row in rows if row["cause"] != "truncated")
        if self.n_calls % self.flush_every_calls == 0:
            self._ledger.write()
        return True

    def _on_rollout_end(self) -> None:
        rows, self._rollout_rows = self._rollout_rows, []
        try:
            falls = [row for row in rows if row["cause"] != "truncated"]
            scalars = {
                "episodes": len(rows),
                "falls": len(falls),
                "com_fore_aft_mean_m": _finite_mean([_number(row["com_fore_aft_mean_m"]) for row in rows]),
                "com_fore_aft_last_falls_m": _finite_mean([_number(row["com_fore_aft_last_m"]) for row in falls]),
                "force_limited_fraction_falls": _finite_mean(
                    [_number(row["actuator_force_limited_fraction"]) for row in falls]
                ),
            }
            for key, value in scalars.items():
                if math.isfinite(value):
                    # TensorBoard only: SB3's logger prints every recorded key at each dump.
                    self.logger.record(f"rollout_balance/{key}", value, exclude=("stdout", "log"))
        except Exception:  # noqa: BLE001 - a report must not sink a run
            self._warn_once("tensorboard", "Training-episode ledger: could not record rollout_balance/.", exc_info=True)

    def _on_training_end(self) -> None:
        self._ledger.write()

    # ------------------------------------------------------------------ resume and report
    def seed_prior_record(self, directory: str | Path, *, max_timesteps: int | None = None) -> int:
        """Keep the prior session's rows in *directory*'s ``train_episodes.csv`` at or before *max_timesteps*."""
        return self._ledger.seed_prior(Path(directory) / TRAIN_LEDGER_FILENAME, max_timesteps=max_timesteps)

    def summary(self, timesteps: int | None = None) -> dict[str, Any]:
        """``progress.json``'s ``training_rollouts``: the episodes that ended in the last :attr:`window_timesteps`.

        ``episodes``, ``falls`` (episodes that did not reach the time limit)
        and ``fall_fraction``; the mean ``com_fore_aft_mean_m`` over them,
        ``com_fore_aft_last_falls_m`` and ``force_limited_fraction_falls``
        over the falls; per foot with touch shares, the mean of each share
        (``touch_shares``); then this session's ``episodes_total`` and
        ``falls_total``.  Empty before the first row.
        """
        ledger = self._ledger
        if not len(ledger):
            return {}
        now = int(self.num_timesteps if timesteps is None else timesteps)
        steps = [_number(value) for value in ledger.column("timesteps")]
        first = len(steps)
        while first > 0 and steps[first - 1] > now - self.window_timesteps:
            first -= 1
        rows = range(first, len(steps))
        causes = ledger.column("cause")
        falls = [row for row in rows if str(causes[row]) != "truncated"]

        def mean_of(column: str, which: Sequence[int]) -> float:
            if column not in ledger.columns:
                return NAN
            values = ledger.column(column)
            return _finite_mean([_number(values[row]) for row in which])

        out: dict[str, Any] = {
            "window_timesteps": self.window_timesteps,
            "episodes": len(rows),
            "falls": len(falls),
            "fall_fraction": len(falls) / len(rows) if len(rows) else NAN,
            "com_fore_aft_mean_m": mean_of("com_fore_aft_mean_m", rows),
            "com_fore_aft_last_falls_m": mean_of("com_fore_aft_last_m", falls),
            "force_limited_fraction_falls": mean_of("actuator_force_limited_fraction", falls),
        }
        shares: dict[str, dict[str, float]] = {}
        for column in ledger.columns:
            if TOUCH_SHARE_INFIX in column:
                label, stem = column.split(TOUCH_SHARE_INFIX, 1)
                shares.setdefault(label, {})[stem] = mean_of(column, rows)
        if shares:
            out["touch_shares"] = shares
        out["episodes_total"] = self._episodes_total
        out["falls_total"] = self._falls_total
        return out


def build_training_ledger_callback(
    *,
    model_id: ModelId | None,
    stage_config: Mapping[str, Any],
    log_dir: str | Path | None,
    flush_every_calls: int,
    verbose: int = 0,
) -> TrainingEpisodeLedgerCallback | None:
    """The ledger for a stage, or ``None`` without a model (no feet to read) or a stage directory to write into."""
    if model_id is None or log_dir is None:
        return None
    curriculum = stage_config.get("curriculum_kwargs", {}) or {}
    return TrainingEpisodeLedgerCallback(
        model_id=model_id,
        settle_steps=int(curriculum.get("settle_steps", 0)),
        ledger_dir=log_dir,
        flush_every_calls=flush_every_calls,
        verbose=verbose,
    )
