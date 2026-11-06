"""Shared helpers of the pilot-4 setup/loading check (both trees import this; it imports no repository module).

Nothing here trains: rollouts step a plain gymnasium env with given or deterministic actions.
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

# prelaunch/setup copy: this lane's folder (the postfix/code original named postfix/code).
C = "/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/prelaunch/setup"
RUNS = "/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/runs/final"
SEEDS = (42, 52, 60)
EPISODE_SEED = 3042
EXTRA_EPISODE_SEEDS = (3043, 3044, 3045, 3046)
MAX_STEPS = 1000
#: The [env] keys pilot 4 adds to pilot 3's locomotion [env] (the build's five new keys, prey_lateral_range included).
COMMAND_KEYS = ("command_mode", "command_config")
PRICE_KEYS = ("walk_heading_price_weight", "walk_heading_price_scale_rad")
PREY_KEY = "prey_lateral_range"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_pair(seed: int) -> tuple[str, str]:
    base = f"{RUNS}/pilot3_walk_s{seed}/03_locomotion/models"
    return f"{base}/stage2_8000000_steps.zip", f"{base}/stage2_vecnormalize_8000000_steps.pkl"


def my_pair(seed: int) -> tuple[str, str]:
    # prelaunch/setup copy: the byte copies sit at the run's own relative path, as the runner's path map expects.
    base = f"{C}/pairs/pilot3_walk_s{seed}/03_locomotion/models"
    return f"{base}/stage2_8000000_steps.zip", f"{base}/stage2_vecnormalize_8000000_steps.pkl"


def jsonable(value):
    """A JSON-exact copy of an info value (floats keep their repr; arrays become lists)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value)
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()] if value.ndim else jsonable(value.item())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    return value if value is None or isinstance(value, str) else repr(value)


def same(a, b) -> bool:
    """Bitwise equality of two jsonable values (NaN equals NaN; -0.0 differs from 0.0)."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b and math.copysign(1.0, a) == math.copysign(1.0, b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def env_heading(env) -> float:
    """World yaw of the root body (the env's own _heading)."""
    return float(env._heading())


def replay(env, actions: np.ndarray, seed: int) -> dict:
    """Reset with *seed*, step *actions* in order until done; record everything bitwise-comparable."""
    obs, info0 = env.reset(seed=seed)
    rec = {"obs": [jsonable(obs)], "reward": [], "info": [], "terminated": [], "truncated": [],
           "qpos": [jsonable(env.data.qpos.copy())], "qvel": [jsonable(env.data.qvel.copy())],
           "heading": [env_heading(env)], "xy": [jsonable(env.data.qpos[0:2].copy())]}
    for a in actions:
        obs, r, te, tr, info = env.step(np.asarray(a))
        rec["obs"].append(jsonable(obs))
        rec["reward"].append(float(r))
        rec["info"].append(jsonable(info))
        rec["terminated"].append(bool(te))
        rec["truncated"].append(bool(tr))
        rec["qpos"].append(jsonable(env.data.qpos.copy()))
        rec["qvel"].append(jsonable(env.data.qvel.copy()))
        rec["heading"].append(env_heading(env))
        rec["xy"].append(jsonable(env.data.qpos[0:2].copy()))
        if te or tr:
            break
    return rec


def heading_metrics(headings, xy) -> dict:
    """Yaw change and weave of one episode from the root-yaw series (reset first) and the pelvis xy positions.

    yaw_change_deg: unwrapped end yaw minus start yaw.  swing_deg: max minus min of the unwrapped yaw (sees a weave that
    ends near its start).  detrended_swing_deg / detrended_rms_deg: the same after removing the least-squares line in
    time (the weave about the steady curve).  lateral_m / forward_m: the end position in the spawn heading's frame.
    """
    h = np.unwrap(np.asarray(headings, dtype=np.float64))
    rel = h - h[0]
    t = np.arange(len(rel), dtype=np.float64)
    coef = np.polyfit(t, rel, 1) if len(rel) > 2 else np.array([0.0, 0.0])
    resid = rel - np.polyval(coef, t)
    p = np.asarray(xy, dtype=np.float64)
    d = p[-1] - p[0]
    c, s = math.cos(h[0]), math.sin(h[0])
    return {
        "steps": int(len(rel) - 1),
        "yaw_change_deg": float(np.degrees(rel[-1])),
        "swing_deg": float(np.degrees(rel.max() - rel.min())),
        "max_abs_yaw_from_start_deg": float(np.degrees(np.max(np.abs(rel)))),
        "rms_yaw_from_start_deg": float(np.degrees(np.sqrt(np.mean(rel ** 2)))),
        "detrended_swing_deg": float(np.degrees(resid.max() - resid.min())),
        "detrended_rms_deg": float(np.degrees(np.sqrt(np.mean(resid ** 2)))),
        "forward_m": float(c * d[0] + s * d[1]),
        "lateral_m": float(-s * d[0] + c * d[1]),
    }


def dump(path: str, obj) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
