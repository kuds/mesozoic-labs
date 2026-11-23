"""Periodic checks during training: named conditions, a few deterministic episodes each (robot-agnostic; D21).

Why: a run that trains on terrain, pushes, slippery floors or turns is normally evaluated only on the flat floor
while it trains, so nobody sees when (or whether) those skills appear until the run ends. These checks run the
current policy at a few named points every ``every`` steps and append one JSON line per check to a file.

The robot supplies everything specific through one callable: ``run_point(name, spec, seed, act) -> dict`` runs one
episode at point ``spec`` with the policy ``act(obs) -> action`` and returns that episode's record. The record is
JSON-safe, with at least ``fell`` (bool); the charts also read ``forward_speed_mps`` when it is there. Points are
``{name: spec}``; a spec is whatever the robot's ``run_point`` understands (for the 60 cm build, a robustness-sweep
point). ``act`` takes one observation as the env returns it (a flat array; a dict observation needs its own
``act``) and applies the training VecNormalize statistics.

A point whose episode raises is recorded as ``{"error": ...}`` for that seed and reported once per check; training
goes on.

Not a verdict: these are progress readings on a few seeds. A pre-registered run is judged on its final checkpoint
with its own tools. Nothing here draws from the training's random streams: the policy acts deterministically on
the training VecNormalize statistics without updating them, and the episodes run in envs the robot builds apart
from the training envs.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.callbacks import BaseCallback


def first_multiple_above(n: int, every: int) -> int:
    """The smallest positive multiple of ``every`` above ``n``."""
    return (n // every + 1) * every


class PeriodicChecks(BaseCallback):
    def __init__(self, path, every: int, points: dict[str, Any],
                 run_point: Callable[[str, Any, int, Callable], dict], seeds=(0, 1), say=print,
                 summarise: Callable[[str, list[dict]], str] | None = None, verbose: int = 0):
        super().__init__(verbose)
        if every <= 0:
            raise ValueError("every must be positive")
        self.path, self.every, self.points = Path(path), int(every), dict(points)
        self.run_point, self.seeds, self.say, self.summarise = run_point, tuple(seeds), say, summarise
        self._write_failed = False

    def _on_training_start(self) -> None:
        self.next = first_multiple_above(self.model.num_timesteps, self.every)

    def _act(self, obs):
        vn = self.model.get_vec_normalize_env()
        o = vn.normalize_obs(obs[None, :]) if vn is not None else obs[None, :]
        a, _ = self.model.predict(o, deterministic=True)
        return a[0]

    def _on_step(self) -> bool:
        n = self.model.num_timesteps
        if n >= self.next:
            while self.next <= n:
                self.next += self.every
            self.check(n)
        return True

    def check(self, n: int) -> dict:
        t0 = time.time()
        out, errors = {}, []
        for name, spec in self.points.items():
            eps = []
            for s in self.seeds:
                try:
                    eps.append(self.run_point(name, spec, s, self._act))
                except Exception as exc:  # a check never stops training
                    eps.append({"error": repr(exc), "seed": s})
                    errors.append(f"{name} seed {s}: {exc!r}")
            out[name] = eps
        rec = {"steps": int(n), "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "wall_s": round(time.time() - t0, 2), "seeds": list(self.seeds), "points": out}
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except OSError as exc:
            if not self._write_failed:
                self.say(f"WARNING: {self.path.name} could not be written ({exc!r}); training goes on")
                self._write_failed = True
        parts = []
        for name, eps in out.items():
            ok = [e for e in eps if "error" not in e]
            try:
                parts.append(self.summarise(name, ok) if self.summarise else
                             f"{name} falls {sum(bool(e.get('fell')) for e in ok)}/{len(ok)}")
            except Exception as exc:
                parts.append(f"{name}: summary failed ({exc!r})")
        self.say(f"checks at {n} steps ({rec['wall_s']:.0f} s): " + "; ".join(parts))
        if errors:
            self.say(f"WARNING: checks at {n} steps: {len(errors)} episode(s) failed, training goes on; first: "
                     f"{errors[0]}")
        return rec


def read_checks(path) -> list[dict]:
    """The lines of a checks file; a torn last line is skipped."""
    rows = []
    p = Path(path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows
