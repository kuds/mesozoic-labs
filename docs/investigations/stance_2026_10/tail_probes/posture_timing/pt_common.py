"""Shared loaders for the posture-timing analysis (TEST 1).  Read-only on every source; no simulation.

Sources (all local copies; nothing fetched):
  S1F2  r10 stance6 F2 copies (other session's folder, read only)
  ECS   r10 entropy continuation screen runs (read only)
  R9    r9 stance5 / stance5r / stance4 ledgers (other session's folder), r9 policy probes (tailround/fetch)
  TS    r9 tail screen continuations (tailround/screen/runs)
"""
from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import numpy as np

S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
OUT = f"{S2}/com_revisit/notes/posture_timing"

F2 = f"{S1}/foot/stance6/fetch/F2"
PANELS = f"{S1}/foot/stance6/panels"
ECS_RUNS = f"{S2}/r10round/ecs/runs"
R9_RUNS = {
    "stance5_LTSB_s42": f"{S1}/foot/stance5/runs/stance5_LTSB_s42/01_stance",
    "stance5_LTSB_s52": f"{S1}/foot/stance5/runs/stance5_LTSB_s52/01_stance",
    "stance5_LTSB_s60": f"{S1}/foot/stance5/runs/stance5_LTSB_s60/01_stance",
    "stance5_LTSB_s70": f"{S1}/foot/stance5/replication/runs/stance5_LTSB_s70/01_stance",
    "stance5_LTSB_s80": f"{S1}/foot/stance5/replication/runs/stance5_LTSB_s80/01_stance",
    "stance4_LTS_s42": f"{S1}/foot/stance4/runs/stance4_LTS_s42/01_stance",
    "stance4_LTS_s52": f"{S1}/foot/stance4/runs/stance4_LTS_s52/01_stance",
    "stance4_LTS_s60": f"{S1}/foot/stance4/runs/stance4_LTS_s60/01_stance",
}
R9_PROBE = {k: f"{S2}/tailround/fetch/{k}/A/policy_probe.npz"
            for k in ("stance5_LTSB_s42", "stance5_LTSB_s52", "stance5_LTSB_s60", "stance5_LTSB_s70", "stance5_LTSB_s80")}
TS_RUNS = f"{S2}/tailround/screen/runs"

# Observation layout (probe/layout.txt, measured by the probe agent on the r10 code copy; the r9 tree's _get_obs
# concatenates the same segments in the same order, and both plants have nq 28, obs 63).
OBS = {"neck_pitch": 0, "neck_yaw": 1, "head_pitch": 2, "tail_1_pitch": 3, "tail_1_yaw": 4, "tail_2_pitch": 5,
       "tail_3_pitch": 6, "r_hip_pitch": 7, "r_hip_roll": 8, "r_knee": 9, "r_ankle": 10, "l_hip_pitch": 14,
       "l_hip_roll": 15, "l_knee": 16, "l_ankle": 17, "g_x": 42, "g_y": 43, "g_z": 44, "v_fwd": 48}
TAIL3 = 14  # action index of tail_3_pitch_act
NECK = 0    # action index of neck_pitch_act
HEAD = 2    # action index of head_pitch_act
R2D = 180.0 / math.pi


def f2_dir(run: str) -> str:
    return f"{F2}/stance6_r10LTSB_{run}/copies/01_stance"


def f2_file(run: str, name: str) -> str:
    return f"{f2_dir(run)}/{name}.copy1"


def fnum(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def read_csv(path: str) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def ledger_per_eval(rows: list[dict], cols: list[str]) -> dict:
    """Group eval_episodes rows by timesteps; per evaluation: n, mean of each numeric column (nan-aware)."""
    by = defaultdict(list)
    for r in rows:
        by[int(r["timesteps"])].append(r)
    ts = sorted(by)
    out = {"t": np.array(ts, dtype=np.int64), "n": np.array([len(by[t]) for t in ts])}
    for c in cols:
        vals = []
        for t in ts:
            xs = [fnum(r.get(c)) for r in by[t]]
            xs = [x for x in xs if not math.isnan(x)]
            vals.append(float(np.mean(xs)) if xs else float("nan"))
        out[c] = np.array(vals)
    out["_by"] = by
    return out


def tail_overbar_counts(by: dict, ts) -> np.ndarray:
    """Per evaluation: episodes whose ledger saturated_actuator is tail_3_pitch_act with fraction > 0.1 (M5's bar)."""
    out = []
    for t in ts:
        k = 0
        for r in by[t]:
            if r.get("saturated_actuator") == "tail_3_pitch_act" and fnum(r.get("saturated_actuator_fraction")) > 0.1:
                k += 1
        out.append(k)
    return np.array(out)


def any_overbar_counts(by: dict, ts, actuator: str | None = None) -> np.ndarray:
    out = []
    for t in ts:
        k = 0
        for r in by[t]:
            a = r.get("saturated_actuator")
            if a and fnum(r.get("saturated_actuator_fraction")) > 0.1 and (actuator is None or a == actuator):
                k += 1
        out.append(k)
    return np.array(out)


def r9_sat_reason_counts(by: dict, ts) -> np.ndarray:
    """r9 ledgers name no actuator: episodes listing max_actuator_saturation_fraction in screen_v2_reasons."""
    return np.array([sum(1 for r in by[t] if "max_actuator_saturation_fraction" in (r.get("screen_v2_reasons") or ""))
                     for t in ts])


def probe_batch_means(path: str) -> dict:
    """Per-rollout means of the training observations, recovered exactly from the running obs_rms.

    The normaliser is SymmetricRunningMeanStd (mirror_kl_ppo.py l100-l140): each update adds a batch and its mirror
    image, so sagittal channels (pitch joints, g_x, g_z) keep their batch mean while lateral channels cancel.  For
    a running mean m with count c, batch mean = (m_n c_n - m_{n-1} c_{n-1}) / (c_n - c_{n-1}).  The first row's
    batch includes the 8 reset observations (count 16,392).
    """
    d = np.load(path, allow_pickle=False)
    m = d["obs_rms_mean"].astype(np.float64)
    c = d["obs_rms_count"].astype(np.float64)
    t = d["update_timesteps"].astype(np.int64)
    prev_m = np.vstack([np.zeros((1, m.shape[1])), m[:-1]])
    prev_c = np.concatenate([[1e-4], c[:-1]])
    dc = c - prev_c
    bm = (m * c[:, None] - prev_m * prev_c[:, None]) / dc[:, None]
    out = {"t": t, "batch_mean": bm, "dc": dc}
    for k in d.files:
        if k.startswith("rollout_") or k in ("std_per_actuator", "ent_coef"):
            out[k] = d[k]
    return out


def pitch_from_gx(gx) -> np.ndarray:
    """Pelvis pitch (deg, + nose up, the ledger's root_pitch convention): g = -R[2,:], pitch = asin(R[2,0]) = -asin(g_x)."""
    return -np.degrees(np.arcsin(np.clip(gx, -1, 1)))


def bin_series(t, y, edges):
    """Median of y in [edges[i], edges[i+1])."""
    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t >= a) & (t < b) & ~np.isnan(y)
        out.append(float(np.median(y[sel])) if sel.any() else float("nan"))
    return np.array(out)


def train_episode_series(path: str) -> dict:
    """Training episodes (stochastic policy): full-horizon rows only (length 1000), CoM fore-aft and pitch means."""
    rows = read_csv(path)
    t, com, pit = [], [], []
    for r in rows:
        if int(float(r["length"])) < 1000:
            continue
        c, p = fnum(r["com_fore_aft_mean_m"]), fnum(r["root_pitch_mean_deg"])
        if math.isnan(c) or math.isnan(p):
            continue
        t.append(int(r["timesteps"]))
        com.append(c)
        pit.append(p)
    return {"t": np.array(t), "com": np.array(com), "pitch": np.array(pit), "n_rows": len(rows)}


def first_sustained(t, y, thr, above=True, k=3):
    """First t where y crosses thr and stays across it for k consecutive points (nan breaks a run)."""
    y = np.asarray(y, dtype=float)
    ok = (y >= thr) if above else (y <= thr)
    ok &= ~np.isnan(y)
    run = 0
    for i, v in enumerate(ok):
        run = run + 1 if v else 0
        if run >= k:
            return int(t[i - k + 1])
    return None


def smooth(y, w):
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y
    out = np.full_like(y, np.nan)
    for i in range(len(y)):
        lo = max(0, i - w + 1)
        seg = y[lo:i + 1]
        seg = seg[~np.isnan(seg)]
        if len(seg):
            out[i] = seg.mean()
    return out


def spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, (int, np.integer)):
        return f"{x:,}"
    if isinstance(x, float) and math.isnan(x):
        return "nan"
    return f"{x:+.{nd}f}"
