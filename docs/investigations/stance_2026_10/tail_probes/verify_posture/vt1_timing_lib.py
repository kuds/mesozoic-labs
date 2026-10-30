"""Loaders shared by vt1b_onset.py (copied from vt1_timing.py, my own code)."""
import csv
import math
from collections import defaultdict

import numpy as np

S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
ECS = f"{S2}/r10round/ecs/runs"
OUT = f"{S2}/com_revisit/notes/verify_posture"
R2D = 180.0 / math.pi


def f2(run, name):
    return f"{S1}/foot/stance6/fetch/F2/stance6_r10LTSB_{run}/copies/01_stance/{name}.copy1"


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def ledger(path):
    by = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            by[int(r["timesteps"])].append(r)
    ts = sorted(by)
    return ts, by


def lmean(by, t, col, full_only=True):
    xs = []
    for r in by[t]:
        if full_only and num(r.get("length")) < 1000:
            continue
        v = num(r.get(col))
        if not math.isnan(v):
            xs.append(v)
    return float(np.mean(xs)) if xs else float("nan")


def overbar(by, t, act):
    return sum(1 for r in by[t] if r.get("saturated_actuator") == act and num(r.get("saturated_actuator_fraction")) > 0.10)


def onsets(ts, by, act):
    cnt = [overbar(by, t, act) for t in ts]
    first = next((t for t, c in zip(ts, cnt) if c > 0), None)
    flagged = [t for t, c in zip(ts, cnt) if c >= 5]
    alarm = None
    for i in range(1, len(ts)):
        if cnt[i] >= 5 and cnt[i - 1] >= 5:
            alarm = ts[i]
            break
    return dict(first_overbar=first, first_flag=flagged[0] if flagged else None, first_alarm=alarm,
                n_overbar=int(sum(cnt)))


def probe(path):
    d = np.load(path, allow_pickle=False)
    m = d["obs_rms_mean"].astype(np.float64)
    c = d["obs_rms_count"].astype(np.float64)
    t = d["update_timesteps"].astype(np.int64)
    # my own recovery: explicit loop, initial running state (mean 0, count 1e-4 = SB3 RunningMeanStd epsilon)
    bm = np.empty_like(m)
    pm, pc = np.zeros(m.shape[1]), 1e-4
    for i in range(len(t)):
        bm[i] = (m[i] * c[i] - pm * pc) / (c[i] - pc)
        pm, pc = m[i], c[i]
    return dict(t=t, bm=bm, mu=d["rollout_mu_mean"].astype(np.float64), dc=np.diff(np.concatenate([[1e-4], c])))


def train_eps(path):
    t, com, pit = [], [], []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            if num(r["length"]) < 1000:
                continue
            t.append(int(r["timesteps"]))
            com.append(num(r["com_fore_aft_mean_m"]) * 1000)
            pit.append(num(r["root_pitch_mean_deg"]))
    return np.array(t), np.array(com), np.array(pit)


