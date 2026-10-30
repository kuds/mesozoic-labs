"""Verifier's independent re-reduction of the policies lens's saved tracks (no simulation, no repo import).

python -I -B recompute.py LENS_NOTES_DIR OUT_JSON

Reads the lens's float32 tracks (run/, run_held/, run_null/, run_r9/ tracks/*.npz) and re-derives, with this file's
own numpy code (not stance_track_helpers), per episode over the window from 5 s: the CoM lateral offset from the
feet's centre in the pelvis-heading frame, the pelvis roll, pitch, fore-aft CoM offset, horizontal CoM speed RMS,
turn from spawn, the feet-centroid least-squares creep speed and its fore-aft component, and the PR's settle second.
It also re-reads the lens's episodes.jsonl so the two can be compared key by key.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

LENS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
WINDOWS = (2.0, 5.0, 9.0)


def ypr(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    yaw = np.arctan2(2 * (x * y + w * z), 1 - 2 * (y * y + z * z))
    pitch = np.arcsin(np.clip(2 * (x * z - w * y), -1, 1))  # + nose up (x axis elevation)
    roll = np.arctan2(2 * (y * z + w * x), 1 - 2 * (x * x + y * y))  # + left side up
    return np.stack([yaw, pitch, roll], axis=1)


def reduce_episode(t, com, quat, feet) -> dict:
    a = ypr(quat)
    yaw = np.unwrap(a[:, 0])
    fwd = np.stack([np.cos(a[:, 0]), np.sin(a[:, 0])], axis=1)
    left = np.stack([-np.sin(a[:, 0]), np.cos(a[:, 0])], axis=1)
    centre = feet[:, :, :2].mean(axis=1)
    off = com[:, :2] - centre
    lat = (off * left).sum(axis=1)
    fa = (off * fwd).sum(axis=1)
    vel = np.gradient(com, t, axis=0)
    speed = np.hypot(vel[:, 0], vel[:, 1])
    # PR settle rule: first whole second after which every second's RMS horizontal CoM speed stays < 1 mm/s
    rms = []
    for k in range(int(round(t[-1] - t[0]))):
        m = (t >= t[0] + k - 1e-9) & (t < t[0] + k + 1 - 1e-9)
        rms.append(math.sqrt(float(np.mean(speed[m] ** 2))))
    settle = next((k for k in range(len(rms)) if all(v < 1e-3 for v in rms[k:])), len(rms))
    out = {"settle_s": settle, "turn_from_spawn_deg": math.degrees(yaw[-1] - yaw[0])}
    for W in WINDOWS:
        w = t >= t[0] + W - 1e-9
        tw = t[w] - t[w].mean()
        X = np.vstack([tw, np.ones_like(tw)]).T
        slope = np.linalg.lstsq(X, centre[w], rcond=None)[0][0]
        f = fwd[w].mean(axis=0)
        f /= np.linalg.norm(f)
        out[str(W)] = {
            "com_lat_mm": 1000 * float(lat[w].mean()),
            "roll_deg": math.degrees(float(a[w, 2].mean())),
            "pitch_deg": math.degrees(float(a[w, 1].mean())),
            "com_fa_mm": 1000 * float(fa[w].mean()),
            "speed_rms_mmps": 1000 * math.sqrt(float(np.mean(speed[w] ** 2))),
            "creep_mmps": 1000 * float(np.linalg.norm(slope)),
            "creep_fa_mmps": 1000 * float(slope @ f),
            "turn_window_deg": math.degrees(float(yaw[w][-1] - yaw[w][0])),
        }
    return out


res: dict = {}
for run in ("run", "run_held", "run_null", "run_r9"):
    d = LENS / run / "tracks"
    res[run] = {}
    for npz in sorted(d.glob("*.npz")):
        z = np.load(npz)
        who = npz.stem
        res[run][who] = {}
        for i, seed in enumerate(z["seeds"]):
            t = z[f"t_{i}"].astype(np.float64)
            if len(t) < 1001:
                res[run][who][int(seed)] = None
                continue
            res[run][who][int(seed)] = reduce_episode(
                t, z[f"com_{i}"].astype(np.float64), z[f"quat_{i}"].astype(np.float64), z[f"feet_{i}"].astype(np.float64)
            )
            res[run][who][int(seed)]["cmd_from_2s"] = z[f"action_{i}"][200:].astype(np.float64).mean(axis=0).tolist()
# the lens's own in-run metrics, for a key-by-key comparison
lens = defaultdict(dict)
for run in ("run", "run_held", "run_null", "run_r9"):
    for line in open(LENS / run / "episodes.jsonl"):
        r = json.loads(line)
        lens[run].setdefault(r["who"], {})[int(r["seed"])] = {w: r["metrics"].get(w) for w in ("2.0", "5.0", "9.0")}
OUT.write_text(json.dumps({"mine": res, "lens": lens}))
print("episodes:", {run: sum(len(v) for v in res[run].values()) for run in res})
