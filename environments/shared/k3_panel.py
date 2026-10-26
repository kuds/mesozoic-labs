"""The in-run K3 noise panel: the stance5 replication judge's two K3 noise tests, rolled during training (report only).

K3, the stance judges' robustness criterion
(``docs/investigations/stance_2026_10/stance5/replication/criteria.txt``:
"joint-space noise falls at most 2 of 20, and own-sigma falls at most 1 of
20" at 6,291,456 and the final), is measured offline by the judge on saved
checkpoints, after a run.  :class:`K3PanelCallback` rolls the same two tests
during training, so a run's K3 history is in its own stage directory: at the
first step on or past every multiple of :data:`K3_PANEL_EVERY` and once when
training ends, on a CPU copy of the live policy and a copy of the live
normalizer's statistics, in an env of its own.  The final panel is rolled
after ``learn()`` returns and the final checkpoint pair is saved
(:meth:`K3PanelCallback.roll_final_panel`, which ``train_base`` calls), so
an interruption during it cannot cost the final pair.

The arithmetic is the judge's ``noise_matched`` (``judge_stance5r.py``,
unchanged; :func:`run_k3_panel`).  Per test and per episode ``i``: seed ``s
= seeds[i]`` (spawns 3042-3061), ``rng = numpy.random.default_rng(s)``,
``obs = env.reset(seed=s)``; each step the observation is normalized in numpy
(``clip((obs - mean) / sqrt(var + epsilon), -clip_obs, clip_obs)`` as
float32), the policy's mean action ``mu`` is read under ``torch.no_grad``
(float64), and

- own-sigma (``noise_matched_own``): ``a = clip(mu + exp(log_std) z)``, with
  ``exp(log_std)`` in float64 of the live float32 ``log_std``;
- joint-space (``noise_matched_control_joint``): ``a = clip(clip(mu) +
  sigma_joint z)``, with ``sigma_joint`` read at launch from a JSON file
  (:func:`read_joint_sigma`), recorded by its sha256;

``z = rng.standard_normal(nu)``, the clip the action space's bounds.  Own
first, then joint (the judge's order).  A fall is an episode shorter than the
horizon (the stage's ``max_episode_steps``).

Report only, and inert to training (the in-suite proof is
``tests/test_curriculum_balance_telemetry.py``):

- It is off unless the launch enables it (``train(..., k3_panel=
  K3PanelConfig(...))``, the notebook's ``K3_PANEL`` cell, the command line's
  ``--k3-panel``): no stage TOML key, so no stage-config, task or
  hyperparameter digest moves.
- No global random draw: each episode's own ``numpy.random.Generator`` and
  the panel env's own ``np_random``; the whole panel runs inside
  :class:`~environments.shared.policy_probe.GlobalRngGuard`, which restores
  numpy's legacy generator, ``random``, torch's CPU and CUDA generators on
  exit (an exception included) and is reported if any moved.
- The live policy is only read: its state dict is copied (to the CPU) into a
  twin built once per session (``type(policy)(**_get_constructor_parameters())``
  under the guard: its weight initialisation draws, and the guard restores
  them); never ``predict()`` (which sets the training mode), never a
  ``deepcopy`` (the mirror-symmetric PPO hooks ``log_std``).  The
  normalizer's ``obs_rms`` arrays are copied; nothing calls its
  ``normalize_obs``, ``step``, ``reset`` or ``training``.  No env of the run
  is stepped or reset.
- Its results go to ``k3_panel.csv`` (one row per episode) and
  ``k3_panel.json`` (``mesozoic.k3-panel/v1``) in the stage directory, and
  one log line per panel: never ``gate_progress.npz``, ``evaluations.npz``,
  ``eval_episodes.csv``, ``progress.json``, ``gate_verdict.json``,
  ``policy_probe.npz``, TensorBoard, a gate, the plateau, the collapse stop,
  the checkpoint selection or the handoff.  A failure is logged and recorded
  for that panel; training goes on.
- Synchronous: training waits while a panel runs (40 episodes).

The offline judge's K3 stays the K3 of record; this panel is a readout.  On a
same-stage resume (:meth:`K3PanelCallback.seed_prior_record`) the prior
session's panels at or before the resumed step are kept, the prior files are
copied aside before the first rewrite, and a resume exactly on a grid step
whose panel is missing rolls it at training start on the restored checkpoint
pair's statistics.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import math
import platform
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from .curriculum import sb3_compat
from .curriculum.sb3_compat import BaseCallback
from .file_io import read_json_object
from .model_id import ModelId

logger = logging.getLogger(__name__)

#: The JSON file's schema string.
K3_PANEL_SCHEMA = "mesozoic.k3-panel/v1"
#: The files the callback writes into the stage directory.
K3_PANEL_CSV_FILENAME = "k3_panel.csv"
K3_PANEL_JSON_FILENAME = "k3_panel.json"
#: A panel runs at the first step on or past every multiple of this (and once at training end, after the final save).
K3_PANEL_EVERY = 1_048_576
#: The judge's spawns (``NOISE_SEED``, ``NOISE_EPISODES``): episode ``i`` resets and draws its noise from ``3042 + i``.
K3_PANEL_SEEDS: tuple[int, ...] = tuple(range(3042, 3062))
#: The stage the launch option is for (the judge's test); another stage trains without a panel.
K3_PANEL_STAGE = "stance"
#: The two tests, in the judge's order, with the judge file's key for each.
K3_TESTS: tuple[tuple[str, str], ...] = (("own", "noise_matched_own"), ("joint", "noise_matched_control_joint"))
#: The criterion the panel reads out, and where it is written.
K3_CRITERIA_FILE = "docs/investigations/stance_2026_10/stance5/replication/criteria.txt"
#: ``k3_panel.csv``'s columns: one row per episode.
CSV_COLUMNS: tuple[str, ...] = (
    "panel",
    "timesteps",
    "kind",
    "test",
    "episode",
    "seed",
    "length",
    "fall",
    "termination_reason",
    "return",
    "sigma_sha256",
)
#: A panel's kinds: on the grid, at training end, or a missing grid panel rolled when a resumed session starts.
PANEL_KINDS: tuple[str, ...] = ("grid", "final", "resume")


def _sha256_float64(values: Any) -> str:
    """The sha256 of *values* as little-endian float64 bytes."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class K3PanelConfig:
    """The launch-time option that turns the panel on (``None`` anywhere means off).

    Args:
        sigma_file: A JSON file holding the joint-space noise sigma (one entry per actuator); ``None`` or ``""``
            runs the own-sigma test only and records why.
        sigma_key: Where the vector sits in the file (:func:`read_joint_sigma`).
        every: The step grid (:data:`K3_PANEL_EVERY`).
        seeds: The spawns, one episode per test each (:data:`K3_PANEL_SEEDS`).
        stage: The stage id the panel runs on (:data:`K3_PANEL_STAGE`).
    """

    sigma_file: str | None = None
    sigma_key: str | None = None
    every: int = K3_PANEL_EVERY
    seeds: tuple[int, ...] = K3_PANEL_SEEDS
    stage: str = K3_PANEL_STAGE

    def __post_init__(self) -> None:
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError(f"K3 panel: every must be a positive integer, not {self.every!r}")
        seeds = tuple(self.seeds)
        if not seeds or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise ValueError(f"K3 panel: seeds must be one or more integers, not {self.seeds!r}")
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "sigma_file", str(self.sigma_file) if self.sigma_file else None)
        object.__setattr__(self, "sigma_key", str(self.sigma_key) if self.sigma_key not in (None, "") else None)


@dataclass(frozen=True)
class JointSigma:
    """The joint-space sigma as read at launch, with its provenance; ``vector`` is ``None`` when ``problem`` says why."""

    vector: np.ndarray | None
    problem: str = ""
    file: str | None = None
    file_sha256: str | None = None
    key: str | None = None
    path_in_file: str | None = None

    @property
    def sha256(self) -> str | None:
        """The vector's sha256 (little-endian float64 bytes), or ``None``."""
        return None if self.vector is None else _sha256_float64(self.vector)

    def record(self) -> dict[str, Any]:
        """The JSON block ``k3_panel.json`` records."""
        return {
            "file": self.file,
            "file_sha256": self.file_sha256,
            "key": self.key,
            "path_in_file": self.path_in_file,
            "vector": None if self.vector is None else [float(value) for value in self.vector],
            "mean": None if self.vector is None else float(np.mean(self.vector)),
            "sha256": self.sha256,
            "problem": self.problem or None,
        }

    def checked(self, dimension: int) -> "JointSigma":
        """This sigma, or one with the vector withdrawn and the reason when it is not *dimension* long."""
        if self.vector is not None and self.vector.shape != (dimension,):
            return JointSigma(
                None,
                f"the sigma vector has {self.vector.size} entries, the action {dimension}",
                self.file,
                self.file_sha256,
                self.key,
                self.path_in_file,
            )
        return self


def _vector_at(node: Any) -> list[Any] | None:
    if isinstance(node, Mapping) and "vector" in node:
        node = node["vector"]
    return node if isinstance(node, list) else None


def read_joint_sigma(sigma_file: str | Path | None, sigma_key: str | None) -> JointSigma:
    """The joint-space sigma from *sigma_file* at *sigma_key*; ``vector`` ``None`` with the reason when unusable.

    *sigma_key* is a dotted path into the JSON object (``control_sigma.80``),
    or a bare key, looked up as ``source_vectors.<key>``, then
    ``control_sigma.<key>``, then ``<key>`` (the stance5 replication sidecar
    ``sigma_s70_s80.json``: ``42`` and ``60`` are those seeds' own control
    sigma, ``80`` the mean of the three).  The node is a list of numbers or an
    object whose ``vector`` is one; every entry must be finite and positive.
    The file's path and sha256, the resolved path and the vector's sha256 are
    recorded with it.
    """
    if not sigma_file:
        return JointSigma(None, "no joint-space sigma file was given at launch: the joint-space test does not run")
    path = Path(sigma_file)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return JointSigma(
            None, f"the sigma file cannot be read ({type(exc).__name__}: {exc})", str(path), None, sigma_key
        )
    digest = hashlib.sha256(raw).hexdigest()
    base = JointSigma(None, "", str(path), digest, sigma_key, None)
    # The repo's one lenient reader (``file_io.read_json_object``): a file that is not UTF-8, not
    # JSON or not an object reads ``None``, named here, never raised on.
    document = read_json_object(path)
    if document is None:
        return JointSigma(
            None,
            "the sigma file is not a JSON object (not UTF-8, not JSON, or not an object)",
            base.file,
            digest,
            sigma_key,
        )
    if not sigma_key:
        return JointSigma(None, "no sigma key was given at launch", base.file, digest, sigma_key)
    candidates = (
        [sigma_key] if "." in sigma_key else [f"source_vectors.{sigma_key}", f"control_sigma.{sigma_key}", sigma_key]
    )
    for candidate in candidates:
        node: Any = document
        for part in candidate.split("."):
            node = node.get(part) if isinstance(node, Mapping) else None
        values = _vector_at(node)
        if values is None:
            continue
        try:
            vector = np.asarray([float(value) for value in values], dtype=np.float64)
        except (TypeError, ValueError):
            return JointSigma(None, f"{candidate} holds a non-number", base.file, digest, sigma_key, candidate)
        if vector.size == 0 or not np.isfinite(vector).all() or not (vector > 0.0).all():
            return JointSigma(
                None, f"{candidate} is not finite and positive throughout", base.file, digest, sigma_key, candidate
            )
        return JointSigma(vector, "", base.file, digest, sigma_key, candidate)
    return JointSigma(
        None, f"no sigma vector at {' or '.join(candidates)} in the file", base.file, digest, sigma_key, None
    )


class ObsNormalization(NamedTuple):
    """A copy of VecNormalize's observation statistics: float64 ``mean`` and ``var``, ``epsilon``, ``clip_obs``."""

    mean: np.ndarray
    var: np.ndarray
    epsilon: float
    clip_obs: float

    def apply(self, observation: np.ndarray) -> np.ndarray:
        """VecNormalize's formula in numpy, as the judge applies it: the normalized observation as float32."""
        normalized = np.clip(
            (observation - self.mean) / np.sqrt(self.var + self.epsilon), -self.clip_obs, self.clip_obs
        )
        return np.asarray(normalized).astype(np.float32)


def copy_normalization(vec_normalize: Any) -> ObsNormalization | None:
    """Float64 copies of *vec_normalize*'s observation statistics, or ``None`` when it normalizes no observation.

    Only ``obs_rms.mean``, ``obs_rms.var``, ``epsilon`` and ``clip_obs`` are read.  A dict observation space's
    statistics (one per key) are refused: the judge's formula is the flat one.
    """
    if vec_normalize is None or not getattr(vec_normalize, "norm_obs", False):
        return None
    rms = getattr(vec_normalize, "obs_rms", None)
    if rms is None:
        return None
    if isinstance(rms, Mapping):
        raise ValueError("a dict observation's statistics: the panel normalizes a flat observation only")
    return ObsNormalization(
        np.array(rms.mean, dtype=np.float64, copy=True),
        np.array(rms.var, dtype=np.float64, copy=True),
        float(vec_normalize.epsilon),
        float(vec_normalize.clip_obs),
    )


def policy_mean_action(policy: Any, observation: np.ndarray) -> np.ndarray:
    """The policy's mean action for one normalized float32 *observation*, as float64, under ``torch.no_grad``.

    The judge's forward pass: the features, the actor's latent and ``action_net``, not ``predict()``.
    """
    import torch
    from stable_baselines3.common.policies import BaseModel

    with torch.no_grad():
        features = BaseModel.extract_features(policy, torch.as_tensor(observation[None]), policy.pi_features_extractor)
        mean = policy.action_net(policy.mlp_extractor.forward_actor(features))
    action: np.ndarray = mean.cpu().numpy()[0].astype(np.float64)
    return action


def own_sigma(policy: Any) -> np.ndarray:
    """``exp(log_std)`` in float64 of the policy's float32 ``log_std`` (the judge's own sigma)."""
    sigma: np.ndarray = np.exp(policy.log_std.detach().cpu().numpy().astype(np.float64))
    return sigma


def run_k3_panel(
    policy: Any,
    normalization: ObsNormalization | None,
    env_factory: Callable[[], Any],
    *,
    joint_sigma: np.ndarray | None,
    seeds: Sequence[int] = K3_PANEL_SEEDS,
    horizon: int,
    tests: Sequence[str] = ("own", "joint"),
) -> list[dict[str, Any]]:
    """The judge's K3 noise tests on *policy*: one row per episode, in the judge's order (own, then joint).

    *policy* is an SB3 Gaussian actor-critic policy (``log_std``, ``action_net``) on the CPU; *normalization*
    the observation statistics (``None``: raw observations); *env_factory* builds a fresh env (one per test,
    closed after it).  The joint-space test is left out when *joint_sigma* is ``None``.  A row holds
    ``test``, ``episode``, ``seed``, ``length``, ``fall`` (``length < horizon``), ``termination_reason``
    (the env's info, ``""`` at the time limit) and ``return`` (the sum of the rewards).  Draws only from each
    episode's ``numpy.random.default_rng(seed)`` and the env's own reset seed.
    """
    sigma_own = own_sigma(policy)
    rows: list[dict[str, Any]] = []
    for test in tests:
        if test not in ("own", "joint"):
            raise ValueError(f"unknown K3 test {test!r}")
        if test == "joint" and joint_sigma is None:
            continue
        sigma = sigma_own if test == "own" else np.asarray(joint_sigma, dtype=np.float64)
        if sigma.shape != sigma_own.shape:
            raise ValueError(f"the {test} sigma has shape {sigma.shape}, the policy's actions {sigma_own.shape}")
        env = env_factory()
        try:
            low = np.asarray(env.action_space.low, dtype=np.float64)
            high = np.asarray(env.action_space.high, dtype=np.float64)
            for episode, seed in enumerate(seeds):
                rng = np.random.default_rng(int(seed))
                observation, _info = env.reset(seed=int(seed))
                length, total = 0, 0.0
                info: Mapping[str, Any] = {}
                while True:
                    normalized = (
                        normalization.apply(observation)
                        if normalization is not None
                        else np.asarray(observation).astype(np.float32)
                    )
                    mu = policy_mean_action(policy, normalized)
                    if test == "joint":
                        command = np.clip(mu, low, high)
                        action = np.clip(command + sigma * rng.standard_normal(command.shape), low, high)
                    else:
                        action = np.clip(mu + sigma * rng.standard_normal(mu.shape), low, high)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total += float(reward)
                    length += 1
                    if terminated or truncated:
                        break
                rows.append(
                    {
                        "test": test,
                        "episode": episode,
                        "seed": int(seed),
                        "length": length,
                        "fall": int(length < int(horizon)),
                        "termination_reason": str(info.get("termination_reason", "")),
                        "return": total,
                    }
                )
        finally:
            env.close()
    return rows


def summarize_tests(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per test of *rows*: ``n``, ``falls``, ``fall_seeds`` and ``fall_lengths`` (the judge's own counts)."""
    out: dict[str, dict[str, Any]] = {}
    for test, _key in K3_TESTS:
        mine = [row for row in rows if row["test"] == test]
        if not mine:
            continue
        falls = [row for row in mine if int(row["fall"])]
        out[test] = {
            "n": len(mine),
            "falls": len(falls),
            "fall_seeds": [int(row["seed"]) for row in falls],
            "fall_lengths": [int(row["length"]) for row in falls],
        }
    return out


def state_dict_sha256(state: Mapping[str, Any]) -> str:
    """One sha256 over a state dict's tensors, in key order (each key's bytes, then the tensor's)."""
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(str(key).encode("utf-8"))
        digest.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


def _versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {"python": platform.python_version(), "numpy": np.__version__}
    for name in ("mujoco", "torch", "stable_baselines3"):
        try:
            module = __import__(name)
        except ImportError:
            found[name] = None
        else:
            found[name] = str(getattr(module, "__version__", None))
    return found


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_safe(value: Any) -> Any:
    """NaN and infinities as ``None``; numpy scalars and arrays as plain values."""
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return value


def panel_complete(panel: Mapping[str, Any], episodes: int) -> bool:
    """Whether a recorded panel ran in full: no failure, the own test's *episodes* episodes, and the joint test's
    or the reason it did not run."""
    tests = panel.get("tests")
    if panel.get("failure") or not isinstance(tests, Mapping):
        return False
    own = tests.get("own")
    joint = tests.get("joint")
    if not isinstance(own, Mapping) or own.get("n") != episodes:
        return False
    return isinstance(joint, Mapping) and (joint.get("n") == episodes or bool(joint.get("skipped")))


class K3PanelCallback(BaseCallback):  # type: ignore[misc]
    """Roll the K3 panel on the grid and, after the final save, at training end; write the two files.

    The files are ``k3_panel.csv`` and ``k3_panel.json``.  The grid panels
    run inside ``learn()`` (``_on_step``); the final one runs when
    ``train_base`` calls :meth:`roll_final_panel` after saving the final
    checkpoint pair.

    Args:
        config: The launch option (:class:`K3PanelConfig`).
        env_factory: Builds a fresh bare env of the stage (the stage's own ``env_kwargs``): no Monitor, no VecEnv,
            never ``make_env`` (which seeds the global generators).
        horizon: The stage's ``max_episode_steps``: an episode shorter than this is a fall.
        stage_dir: The stage directory the files go to (``None``: kept in memory).
        model_id: The model trained (recorded).
    """

    def __init__(
        self,
        *,
        config: K3PanelConfig,
        env_factory: Callable[[], Any],
        horizon: int,
        stage_dir: str | Path | None,
        model_id: ModelId | None = None,
        verbose: int = 0,
    ):
        if not sb3_compat._SB3_AVAILABLE:
            raise ImportError("stable-baselines3 is required for K3PanelCallback.")
        super().__init__(verbose)
        self.config = config
        self.env_factory = env_factory
        self.horizon = int(horizon)
        self.stage_dir = None if stage_dir is None else Path(stage_dir)
        self.model_id = model_id
        # Read at launch, before anything trains: the file may move or change later, and the record says which
        # bytes were read.
        self.sigma = read_joint_sigma(config.sigma_file, config.sigma_key)
        self._twin: Any = None
        self._unusable: str | None = None
        self._next = int(config.every)
        self._panels: list[dict[str, Any]] = []
        self._rows: list[dict[str, Any]] = []
        self._prior_panels: list[dict[str, Any]] = []
        self._prior_rows: list[dict[str, Any]] = []
        self._prior_sources: list[Path] = []
        self._prior_copied = False
        self._restored: ObsNormalization | None = None
        self._restored_note: str | None = None
        self._warned: set[str] = set()

    def _warn_once(self, key: str, message: str, *, exc_info: bool = False) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, exc_info=exc_info)

    # ------------------------------------------------------------------ the record
    @property
    def panels(self) -> list[dict[str, Any]]:
        """Every panel the JSON holds, the prior session's kept ones first."""
        return [*self._prior_panels, *self._panels]

    @property
    def rows(self) -> list[dict[str, Any]]:
        """Every CSV row, the prior session's kept ones first."""
        return [*self._prior_rows, *self._rows]

    def document(self) -> dict[str, Any]:
        """``k3_panel.json``: the panel's definition, the sigma read at launch, and every panel."""
        document: dict[str, Any] = _json_safe(
            {
                "schema": K3_PANEL_SCHEMA,
                "written_utc": _utc(),
                "report_only": (
                    "a readout of K3's two noise tests during training; no gate, verdict, checkpoint selection or "
                    "handoff reads it, and the offline judge's K3 stays the K3 of record"
                ),
                "model": None if self.model_id is None else self.model_id.record(),
                "spec": {
                    "judge": (
                        "the stance5 replication judge's noise_matched (judge_stance5r.py), arithmetic unchanged: "
                        "per episode seed s, rng = numpy.random.default_rng(s) and env.reset(seed=s); the "
                        "observation normalized as clip((obs - mean) / sqrt(var + epsilon), -clip_obs, clip_obs) "
                        "in float32; mu the policy's mean action under torch.no_grad, in float64; own: "
                        "a = clip(mu + exp(log_std) z); joint: a = clip(clip(mu) + sigma_joint z); "
                        "z = rng.standard_normal(nu), clipped to the action bounds"
                    ),
                    "criteria_file": K3_CRITERIA_FILE,
                    "tests": [{"test": test, "judge_key": key} for test, key in K3_TESTS],
                    "order": "own, then joint (the judge's)",
                    "seeds": list(self.config.seeds),
                    "episodes_per_test": len(self.config.seeds),
                    "horizon": self.horizon,
                    "fall": "an episode shorter than the horizon",
                    "every": self.config.every,
                    "stage": self.config.stage,
                    "forward": (
                        "a CPU copy of the live policy, built once per session, its parameters copied from the live "
                        "policy at each panel; one torch thread during the panel"
                    ),
                    "observation_statistics": (
                        "float64 copies of the training VecNormalize's obs_rms mean and var, its epsilon and "
                        "clip_obs, at the panel's step (the checkpoint pair saved at that step holds the same)"
                    ),
                    "joint_sigma": self.sigma.record(),
                },
                "session_problem": self._unusable,
                "panels": self.panels,
            }
        )
        return document

    def write(self) -> bool:
        """Rewrite both files atomically (the prior session's files copied aside first); ``True`` when written."""
        if self.stage_dir is None:
            return False
        try:
            from .file_io import atomic_copy, atomic_write_csv, atomic_write_json

            if not self._prior_copied:
                # The prior session's files (seed_prior_record), else any this directory already holds (a resume
                # whose record could not be seeded): copied aside before the first rewrite, never deleted.
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                here = [self.stage_dir / K3_PANEL_CSV_FILENAME, self.stage_dir / K3_PANEL_JSON_FILENAME]
                for source in self._prior_sources or here:
                    if source.is_file():
                        atomic_copy(source, self.stage_dir / f"k3_panel.prior_{stamp}{source.suffix}")
                self._prior_copied = True
            atomic_write_csv(self.stage_dir / K3_PANEL_CSV_FILENAME, CSV_COLUMNS, self.rows)
            atomic_write_json(self.stage_dir / K3_PANEL_JSON_FILENAME, self.document(), allow_nan=False)
        except Exception:  # noqa: BLE001 - a report must not sink a run
            self._warn_once("write", f"K3 panel: could not write its files in {self.stage_dir}.", exc_info=True)
            return False
        return True

    # ------------------------------------------------------------------ resume
    def seed_prior_record(self, directory: str | Path, *, max_timesteps: int | None = None) -> int:
        """Keep the prior session's panels (and rows) at or before *max_timesteps*; returns the panels kept.

        ``seed_resume_eval_state`` calls this before ``learn()``.  The prior
        files are copied aside (``k3_panel.prior_<UTC>.csv`` / ``.json`` in
        this session's stage directory) before this session first rewrites
        them: a superseded panel is moved aside, never deleted.
        """
        from .file_io import read_json_object

        folder = Path(directory)
        json_path, csv_path = folder / K3_PANEL_JSON_FILENAME, folder / K3_PANEL_CSV_FILENAME
        self._prior_sources = [path for path in (csv_path, json_path) if path.is_file()]
        record = read_json_object(json_path)
        if record is None:
            if json_path.exists():
                logger.warning("Resume: could not read %s; it starts with this session's panels.", json_path)
            return 0
        panels = [panel for panel in record.get("panels") or [] if isinstance(panel, Mapping)]

        def kept(step: Any) -> bool:
            try:
                value = int(step)
            except (TypeError, ValueError):
                return False
            return max_timesteps is None or value <= int(max_timesteps)

        self._prior_panels = [dict(panel) for panel in panels if kept(panel.get("timesteps"))]
        rows: list[dict[str, Any]] = []
        if csv_path.is_file():
            try:
                with csv_path.open(newline="", encoding="utf-8") as handle:
                    rows = [dict(row) for row in csv.DictReader(handle)]
            except Exception:  # noqa: BLE001 - a report must not sink a run
                logger.warning("Resume: could not read %s; its rows are left out.", csv_path, exc_info=True)
                rows = []
        numbers = {panel.get("panel") for panel in self._prior_panels}
        self._prior_rows = [
            {column: row.get(column, "") for column in CSV_COLUMNS}
            for row in rows
            if kept(row.get("timesteps")) and _whole(row.get("panel")) in numbers
        ]
        return len(self._prior_panels)

    def hold_restored_normalization(self, vec_normalize: Any) -> None:
        """Copy the restored checkpoint pair's statistics before ``learn()`` resets the training env.

        SB3's ``learn()`` resets the env before the first callback runs, and a
        training VecNormalize updates its statistics with those reset
        observations; a panel rolled at training start on a resumed grid step
        reads these copies instead, which equal the checkpoint pair's.
        """
        try:
            self._restored = copy_normalization(vec_normalize)
            self._restored_note = (
                "the restored checkpoint pair's statistics, copied before learn() reset the training env"
                if self._restored is not None
                else "unnormalized"
            )
        except Exception:  # noqa: BLE001 - a report must not sink a run
            self._warn_once("restored", "K3 panel: could not copy the restored statistics.", exc_info=True)
            self._restored, self._restored_note = None, None

    # ------------------------------------------------------------------ the hooks
    def _on_training_start(self) -> None:
        policy = getattr(self.model, "policy", None)
        if policy is None or not hasattr(policy, "log_std") or not hasattr(policy, "action_net"):
            self._unusable = f"{type(self.model).__name__}'s policy is not a Gaussian actor-critic (no log_std)"
            logger.info("K3 panel: %s; no panel this session.", self._unusable)
            self.write()
            return
        try:
            from .policy_probe import GlobalRngGuard

            # The twin's weight initialisation draws from torch's generator: an expected draw, restored by the guard.
            with GlobalRngGuard():
                twin = type(policy)(**policy._get_constructor_parameters())
            twin.set_training_mode(False)
            self._twin = twin
            nu = int(np.prod(self.model.action_space.shape or ()))
            self.sigma = self.sigma.checked(nu)
        except Exception as exc:  # noqa: BLE001 - a report must not sink a run
            self._unusable = f"the policy's CPU copy could not be built ({type(exc).__name__}: {exc})"
            self._warn_once("twin", f"K3 panel: {self._unusable}; no panel this session.", exc_info=True)
            self.write()
            return
        if self.sigma.problem:
            logger.info("K3 panel: %s; the own-sigma test runs alone.", self.sigma.problem)
        every = int(self.config.every)
        step = int(self.num_timesteps)
        self._next = (step // every + 1) * every
        if step > 0 and step % every == 0:
            episodes = len(self.config.seeds)
            if not any(
                _whole(panel.get("timesteps")) == step and panel_complete(panel, episodes)
                for panel in self._prior_panels
            ):
                self._panel("resume", grid_point=step, normalization=self._restored, note=self._restored_note)

    def _on_step(self) -> bool:
        if self._twin is not None and self.num_timesteps >= self._next:
            grid = self._next
            every = int(self.config.every)
            self._next = (int(self.num_timesteps) // every + 1) * every
            self._panel("grid", grid_point=grid)
        return True

    def _on_training_end(self) -> None:
        # Nothing here: inside learn() the final checkpoint pair is not saved yet, and train_base saves it before
        # anything evaluates the final weights.  train_base calls roll_final_panel() right after that save.
        return None

    def roll_final_panel(self) -> None:
        """Roll the panel at training end, after the final checkpoint pair is saved; never raises an ``Exception``.

        ``train_base`` calls this once ``learn()`` has returned (not after an
        interruption) and the stage's final pair (``<stage>_final.zip`` and
        its VecNormalize pickle) is on disk: the weights and statistics read
        are the pair's, and an interruption during the panel (about 85 s on a
        real stance policy) cannot cost the pair.  Nothing when this session
        runs no panel.
        """
        if self._twin is None or self.model is None:
            return
        self.num_timesteps = int(self.model.num_timesteps)
        self._panel("final", grid_point=None)

    # ------------------------------------------------------------------ one panel
    def _live_normalization(self) -> tuple[ObsNormalization | None, str]:
        normalization = copy_normalization(self.model.get_vec_normalize_env())
        return normalization, "the training VecNormalize's statistics at this step" if normalization else "unnormalized"

    def _panel(
        self,
        kind: str,
        *,
        grid_point: int | None,
        normalization: ObsNormalization | None = None,
        note: str | None = None,
    ) -> None:
        """Roll one panel, record it (or its failure) and rewrite the files; never raises."""
        import torch

        from .policy_probe import GlobalRngGuard

        number = 1 + max((_whole(panel.get("panel")) or 0 for panel in self.panels), default=-1)
        record: dict[str, Any] = {
            "panel": number,
            "timesteps": int(self.num_timesteps),
            "grid_point": grid_point,
            "kind": kind,
            "started_utc": _utc(),
            "failure": None,
        }
        rows: list[dict[str, Any]] = []
        started = time.monotonic()
        guard = GlobalRngGuard()
        try:
            with guard:
                threads = torch.get_num_threads()
                torch.set_num_threads(1)
                try:
                    live = self.model.policy.state_dict()
                    state = {key: value.detach().to("cpu", copy=True) for key, value in live.items()}
                    self._twin.load_state_dict(state)
                    if note is None:
                        normalization, note = self._live_normalization()
                    sigma_own = own_sigma(self._twin)
                    record.update(
                        policy_state_sha256=state_dict_sha256(state),
                        normalization=note,
                        obs_mean_sha256=None if normalization is None else _sha256_float64(normalization.mean),
                        obs_var_sha256=None if normalization is None else _sha256_float64(normalization.var),
                        epsilon=None if normalization is None else normalization.epsilon,
                        clip_obs=None if normalization is None else normalization.clip_obs,
                        own_sigma=[float(value) for value in sigma_own],
                        own_sigma_mean=float(sigma_own.mean()),
                        own_sigma_sha256=_sha256_float64(sigma_own),
                        joint_sigma_sha256=self.sigma.sha256,
                    )
                    rows = run_k3_panel(
                        self._twin,
                        normalization,
                        self.env_factory,
                        joint_sigma=self.sigma.vector,
                        seeds=self.config.seeds,
                        horizon=self.horizon,
                    )
                finally:
                    torch.set_num_threads(threads)
        except Exception as exc:  # noqa: BLE001 - a diagnostic must not sink a run
            record["failure"] = f"{type(exc).__name__}: {exc}"
            self._warn_once(f"panel-{kind}", f"K3 panel at {int(self.num_timesteps):,} failed.", exc_info=True)
            rows = []
        record["global_rng_moved"] = bool(guard.changed)
        if guard.changed:
            self._warn_once("rng", "K3 panel: a panel moved a global random generator; it was restored.")
        record["seconds"] = time.monotonic() - started
        record["versions"] = _versions()
        tests = summarize_tests(rows)
        if record["failure"] is None and "joint" not in tests:
            tests["joint"] = {"skipped": self.sigma.problem or "no joint-space sigma"}
        record["tests"] = tests
        for test, _key in K3_TESTS:
            block = tests.get(test) or {}
            record[f"{test}_falls"] = block.get("falls")
            record[f"{test}_episodes"] = block.get("n")
        sigma_for = {"own": record.get("own_sigma_sha256"), "joint": self.sigma.sha256}
        self._rows += [
            {
                "panel": number,
                "timesteps": record["timesteps"],
                "kind": kind,
                "test": row["test"],
                "episode": row["episode"],
                "seed": row["seed"],
                "length": row["length"],
                "fall": row["fall"],
                "termination_reason": row["termination_reason"],
                "return": repr(float(row["return"])),
                "sigma_sha256": sigma_for[row["test"]] or "",
            }
            for row in rows
        ]
        self._panels.append(record)
        self.write()
        if record["failure"] is None:
            joint = tests.get("joint", {})
            logger.info(
                "K3 panel at %s (%s): own-sigma falls %s/%s, joint-space falls %s, %.0f s; report only.",
                f"{record['timesteps']:,}",
                kind,
                tests.get("own", {}).get("falls"),
                tests.get("own", {}).get("n"),
                f"{joint['falls']}/{joint['n']}" if "n" in joint else f"not run ({joint.get('skipped')})",
                record["seconds"],
            )


def _whole(value: Any) -> int | None:
    """*value* as an int (a CSV cell or a JSON number), else ``None``."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return int(number) if math.isfinite(number) and number == int(number) else None


def build_k3_panel_callback(
    config: K3PanelConfig | None,
    *,
    model_id: ModelId | None,
    stage: "int | str",
    stage_config: Mapping[str, Any],
    env_factory: Callable[[], Any] | None,
    log_dir: str | Path | None,
    verbose: int = 0,
) -> K3PanelCallback | None:
    """The panel for a stage, or ``None``: not enabled at launch, no env factory or stage directory, or a stage
    other than the one the option names (``config.stage``, resolved through the model's stage manifest)."""
    if config is None or env_factory is None or log_dir is None:
        return None
    stage_id = str(stage)
    if model_id is not None:
        try:
            from .stage_manifest import load_stage_manifest

            stage_id = load_stage_manifest(model_id).resolve(stage).id
        except Exception:  # noqa: BLE001 - an unresolvable stage is not the panel's stage
            stage_id = str(stage)
    if stage_id != config.stage:
        logger.info("K3 panel: enabled for the %s stage; stage %s trains without it.", config.stage, stage_id)
        return None
    horizon = int(dict(stage_config.get("env_kwargs") or {}).get("max_episode_steps", 1000))
    return K3PanelCallback(
        config=config,
        env_factory=env_factory,
        horizon=horizon,
        stage_dir=log_dir,
        model_id=model_id,
        verbose=verbose,
    )
