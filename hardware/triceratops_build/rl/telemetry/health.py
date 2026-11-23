"""Training health of a Stable-Baselines3 on-policy run, one JSON line per rollout (robot-agnostic; decision D21).

Why: a long run can stall or collapse long before its evaluations show it. The learning algorithm's own
diagnostics (how far each update moved the policy, how random its actions still are, how well the value function
predicts returns), each reward term and the reasons episodes end say early why.

Each line of ``train_health.jsonl`` covers one rollout and the update that followed it:

- ``timesteps``: the model's step count at the end of that rollout; ``wall_s``: the wall time since the previous
  line (the rollout, its update, and any evaluations, checks or saves the other callbacks made in between);
- ``algo``: SB3's ``train/*`` values of that update (PPO: approx_kl, clip_fraction, entropy_loss, explained_variance,
  value_loss, policy_gradient_loss, loss, learning_rate, clip_range, std, n_updates), as SB3 recorded them;
- ``grad_norm``: the global L2 norm of the policy's gradients after the update's last minibatch (after PPO's
  clipping at ``max_grad_norm``, so it sits at that ceiling while clipping binds);
- ``action_std``: the mean of the policy's action standard deviations (Gaussian policies), when it has ``log_std``;
- ``rewards``: the per-step mean over the rollout and all envs of every numeric ``info`` value whose key starts with
  one of ``reward_prefixes``. The default, ``("reward_",)``, is the repository's convention for its species envs; an
  env with another convention says so (the 60 cm build passes ``("r_", "reward_")``). Choose prefixes that only
  reward terms carry: in the species envs ``r_`` means "right" (``r_foot_contact`` is a force);
- ``episodes``: episodes that ended in the rollout; ``endings``: their count per reason. The reason is the first
  non-empty value among ``termination_keys`` in the final ``info`` (default ``termination`` and
  ``termination_reason``, the 60 cm build's and the species envs' names); without one, ``time_limit`` when SB3's
  ``TimeLimit.truncated`` is set, else ``unlabelled``;
- ``obs_var``: the min, median and max of the VecNormalize observation variances, and ``ret_var``, when the training
  env is wrapped in VecNormalize (a collapsing or exploding input scale shows here first).

Reading only: nothing here draws a random number or changes the model, so training is the same with or without it.
A failed write is reported once and training goes on.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def grad_norm(parameters) -> float | None:
    """Global L2 norm over the ``.grad`` of ``parameters``; None before any gradient exists."""
    total, found = 0.0, False
    for p in parameters:
        g = getattr(p, "grad", None)
        if g is not None:
            total += float((g.detach() ** 2).sum().item())
            found = True
    return math.sqrt(total) if found else None


class TrainingHealth(BaseCallback):
    def __init__(self, path, reward_prefixes=("reward_",), termination_keys=("termination", "termination_reason"),
                 say=print, verbose: int = 0):
        super().__init__(verbose)
        self.path = Path(path)
        self.prefixes = tuple(reward_prefixes)
        self.termination_keys = tuple(termination_keys)
        self.say = say
        self._reset_rollout()
        self._pending = False  # a rollout has been collected whose line is not yet written
        self._write_failed = False
        self.rollouts = 0

    def _reset_rollout(self) -> None:
        self._sums = defaultdict(float)
        self._counts = defaultdict(int)
        self._endings: Counter = Counter()
        self._episodes = 0
        self._t0 = time.time()

    def _on_training_start(self) -> None:
        self._reset_rollout()
        # a session killed mid-write leaves a torn last line: end it, so this session's first line stands on its own
        try:
            if self.path.exists() and self.path.stat().st_size:
                with open(self.path, "rb") as f:
                    f.seek(-1, 2)
                    torn = f.read(1) != b"\n"
                if torn:
                    with open(self.path, "a", encoding="utf-8") as f:
                        f.write("\n")
        except OSError:
            pass

    def _ending(self, info: dict) -> str:
        for k in self.termination_keys:
            v = info.get(k)
            if v:
                return str(v)
        return "time_limit" if info.get("TimeLimit.truncated") else "unlabelled"

    # --- per step: reward terms and endings ---
    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or ()
        dones = self.locals.get("dones")
        for i, info in enumerate(infos):
            for k, v in info.items():
                if isinstance(k, str) and k.startswith(self.prefixes) and isinstance(
                        v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                    self._sums[k] += float(v)
                    self._counts[k] += 1
            if dones is not None and bool(dones[i]):
                self._episodes += 1
                self._endings[self._ending(info)] += 1
        return True

    def _on_rollout_end(self) -> None:
        self._pending = True
        self._steps_at_end = int(self.model.num_timesteps)

    # --- after the update: SB3 keeps train/* until the next dump, which follows the next rollout ---
    def _on_rollout_start(self) -> None:
        if self._pending:
            self._write()

    def _on_training_end(self) -> None:
        if self._pending:
            self._write()

    def _write(self) -> None:
        m = self.model
        algo = {k[len("train/"):]: _finite(v) for k, v in m.logger.name_to_value.items() if k.startswith("train/")}
        policy = m.policy
        log_std = getattr(policy, "log_std", None)
        rec = {"timesteps": self._steps_at_end, "rollout": self.rollouts, "time_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "wall_s": round(time.time() - self._t0, 3),
            "algo": algo, "grad_norm": _finite(grad_norm(policy.parameters())),
            "action_std": _finite(float(np.exp(log_std.detach().cpu().numpy()).mean())) if log_std is not None
            else None,
            "rewards": {k: _finite(self._sums[k] / self._counts[k]) for k in sorted(self._sums) if self._counts[k]},
            "episodes": self._episodes, "endings": dict(sorted(self._endings.items()))}
        vn = m.get_vec_normalize_env()
        if vn is not None and getattr(vn, "obs_rms", None) is not None and hasattr(vn.obs_rms, "var"):
            var = np.asarray(vn.obs_rms.var, dtype=np.float64)
            rec["obs_var"] = {"min": _finite(var.min()), "median": _finite(np.median(var)), "max": _finite(var.max())}
            if getattr(vn, "ret_rms", None) is not None:
                rec["ret_var"] = _finite(np.asarray(vn.ret_rms.var).mean())
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as exc:
            if not self._write_failed:
                self.say(f"WARNING: {self.path.name} could not be written ({exc!r}); training goes on")
                self._write_failed = True
        self.rollouts += 1
        self._pending = False
        self._reset_rollout()


def latest_branch(rows: list[dict], key: str) -> list[dict]:
    """The rows of the run as it stands: a resumed session restarts at its checkpoint's step count, so rows the
    killed session wrote past that point (its discarded branch) are dropped, and a later row replaces an earlier one
    at the same count. Rows without ``key`` are skipped. In step order."""
    kept: list[dict] = []
    for r in rows:
        n = r.get(key)
        if not isinstance(n, (int, float)):
            continue
        while kept and kept[-1][key] >= n:
            kept.pop()
        kept.append(r)
    return kept


def read_health(path) -> list[dict]:
    """The lines of a ``train_health.jsonl``; a torn line (a run killed mid-write) is skipped."""
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
