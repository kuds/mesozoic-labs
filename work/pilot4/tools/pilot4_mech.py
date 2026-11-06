"""Walking pilot 4, report only: the gait mechanics behind the limp and the asymmetry (criteria_pilot4 v3, "Report
only", 2nd item).

Scratch, never merged; pinned by sha256 before launch. No outcome, label, bar or run decision reads anything this
tool writes; it prints numbers and pilot 3's 8M values beside them, no verdict. Every metric is the gait-report plan
lanes' definition (SP/work/gait_report/GAIT_REPORT_PLAN.txt sections 4.1-4.4, measured by the lanes' scripts, which
this tool reproduces line by line on the frozen probe's trace):

  from the probe's own row (gait_probe.analyze_episode, window from 100 steps, as judged; probe_rows_summary.py):
    lead_mean_m (fore-aft foot placement, left ahead +), floor_duty_asymmetry_L_minus_R (duty L-R; v3 cites -0.09 /
    -0.01 / -0.19), roll_mean_deg (+ = right side down; s60's 3.8-deg lean), jointlag_antiphase_index_ankle,
    yaw_change_deg.
  per touchdown and swing (survey_metrics/scripts/survey_analyze.py; window steps >= 100, the probe's floor truth
  debounced with the probe's own debounce/_events):
    landing peak = the largest substep floor normal force on the touching leg over the touchdown step and the next 4
    (50 ms), in body weights; median per side and SI = (L - R) / (0.5 (L + R)); p95 per side (hard landing);
    step length (fore-aft distance from the touching foot to the other along the travel direction over +-0.25 s);
    stance time; scuff = a swing (lift-off a to touchdown b) with any substep floor force above 0.1 N in steps
    [a + 2, b - 2), as the per-episode share (survey_analyze) and pooled over the panel (scuff_detail.py).
  per own-foot stride (gait_report/fix/scripts/stride_check.py; whole-episode contact, its own 2-step debounce,
    touchdowns from step 100): knee, ankle and hip-pitch range per stride (max - min, deg), median per side, ratio
    min/max (s42's knee 12.1 vs 28.0 deg).
  weave (gait_report/fix/scripts/ct_handbuilt.py): t_start (the first time from 1 s at which the centred 0.5-s mean
    horizontal pelvis speed reaches 90% of the median over the next 2 s) and the counter-turn of the 100-sample mean
    yaw from t_start: max over t1 < t2 of s (psi(t1) - psi(t2)), s the sign of the net turn (s60's 29-deg reversal).
  leg power (gait_report/plan/scripts/leg_power.py on survey_analyze's per-actuator mean |power| over window steps):
    per side the sum over hip pitch, hip roll, knee and ankle of each actuator's median over episodes; max/min.
  the ledger's asymmetry (the gait ledger page, training_history.gait_ledger, PR #659): over the last 6
    evaluations at or before the checkpoint, each evaluation's full-horizon episodes pooled, the median of the
    per-episode r_leg_work_abs_j / l_leg_work_abs_j and its max/min (2.06x / 1.97x / 1.80x at pilot 3's 8M), from
    the run copy's own eval_episodes.csv.
Aggregates: the median [min, max] over the panel's full-length episodes (not terminated, the horizon reached), as the
survey; n full of 40 printed.

    cd TREE && ISO_WT=TREE python -I TOOLS/iso.py TOOLS/pilot4_mech.py --cell OUT/s42/2000000 --json M/s42.json
    cd TREE && ISO_WT=TREE python -I TOOLS/iso.py TOOLS/pilot4_mech.py --parent 42 [--runs RUNS] --json M/p42.json
        [--episodes 40] [--validation]

--cell: a judged cell (pilot4_judge.py's OUT/s<seed>/<steps>; with --validation also pilot 3's cells, for checks
only); every re-rolled episode must equal probe.json's row (pilot3_video.py's rule; a difference is recorded and the
exit status is 1); the ledger is the cell's run copy's own RUN/03_locomotion/eval_episodes.csv. --parent SEED: that
seed's pilot-3 8M pair (sha256 checked against v3's table), edge-prepared in memory in a live-command tree (as
pilot4_mirror.py), its ledger the pilot-3 run copy's. Refusals as pilot4_mirror.py's (nothing written, exit 2): an
existing --json; a --json inside the tree, the cell's run copy or an input's run directory; local edits; another
head; `environments` from outside the tree; a changed pair; a non-panel probe.json or other env kwargs; an
off-schedule, validation-only (unless --validation) or pilot-3 (unless --validation) cell; a parent not v3's.
A part whose input cannot be read (no ledger, no full-length episode, no t_start) is reported as not run, with why.
Deterministic, CPU, no training of any kind.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCHEMA = "walking-pilot4-mech/v1"
SPECIES, EMBODIMENT, STAGE = "tyrannosaurus_rex", "anatomical", "locomotion"
PILOT4_CELL_SCHEMA = "walking-pilot4-judge/v1"
PILOT3_CELL_SCHEMA = "walking-pilot3-judge/v1"
PANEL_SEED_BASE = 3042
PANEL_EPISODES = 40
SETTLE = 100  # the lanes' window: control steps k >= 100
LANDING_STEPS = 5  # survey_analyze: fmax[i : i + 5] (the first 50 ms at dt 0.01)
SCUFF_EDGE_STEPS = 2  # survey_analyze / scuff_detail: [lift-off + 2, touchdown - 2)
TRAVEL_HALF_STEPS = 25  # survey_analyze travel_dir: +-0.25 s
MOVAVG_SAMPLES = 100  # ct_handbuilt: movavg(yaw[ks:], 100)
LEDGER_EVALUATIONS = 6  # training_history.GAIT_LEDGER_EVALUATIONS
LEG_JOINTS = ("hip_pitch", "hip_roll", "knee", "ankle")  # leg_power.py
ROM_JOINTS = ("knee", "ankle", "hip_pitch")  # stride_check.py
COMMAND_SLICE = slice(60, 63)
DEFAULT_RUNS = ("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/"
                "runs/final")
PARENTS = {
    42: ("sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
         "sha256:b9e6dee894e549ca4e844f7ef3c162e0f52609eacede7b1790e5d649e77d9cce"),
    52: ("sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
         "sha256:cf29635feb8c4040ea9a9bb08255b97de60673365dce107445c115aa2927f37a"),
    60: ("sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2",
         "sha256:56e58a536d08198fdafbde3310e05116ff65e7cd471f536016858f87f9c32dc9"),
}
#: Pilot 3's 8M values, printed beside (report only), each with its source (medians over the 40-episode panels).
PILOT3_REFERENCE = {
    "knee_rom_L_R_deg": {"source": "gait_report/fix/out/stride_check.txt", 42: (12.1, 28.0), 52: (28.7, 20.8),
                         60: (13.5, 27.9)},
    "knee_rom_ratio": {"source": "gait_report/fix/out/stride_check.txt", 42: 0.426, 52: 0.714, 60: 0.485},
    "landing_peak_bw_L_R_SI": {"source": "gait_report/survey_metrics/out/survey.txt", 42: (1.582, 3.107, -0.647),
                               52: (3.499, 2.422, 0.377), 60: (2.081, 1.963, 0.069)},
    "lead_mean_m": {"source": "gait_report/survey_metrics/out/probe_rows_summary.txt", 42: -0.243, 52: 0.090,
                    60: 0.126},
    "step_len_L_R_m": {"source": "gait_report/survey_metrics/out/survey.txt", 42: (0.339, 0.711), 52: (0.669, 0.574),
                       60: (0.491, 0.279)},
    "duty_L_minus_R": {"source": "run log line 1071 (probe.json floor duty asymmetry, median)", 42: -0.09,
                       52: -0.01, 60: -0.19},
    "roll_mean_deg": {"source": "gait_report/survey_metrics/out/survey.txt", 42: -0.258, 52: 1.298, 60: 3.774},
    "scuff_share_R_per_episode_median": {"source": "gait_report/survey_metrics/out/survey.txt scuff_frac_R",
                                         42: 0.0, 52: 0.556, 60: 0.0},
    "scuff_R_pooled_touched_of_swings": {"source": "gait_report/survey_metrics/out/scuff_detail.txt",
                                         42: (0, 854), 52: (384, 702), 60: (0, 972)},
    "counter_turn_deg": {"source": "gait_report/fix/out/ct_handbuilt.txt", 42: 0.99, 52: 3.90, 60: 28.95},
    "ankle_antiphase_index": {"source": "gait_report/survey_metrics/out/probe_rows_summary.txt", 42: 0.103,
                              52: 0.958, 60: 0.914},
    "leg_power_max_over_min": {"source": "gait_report/plan/out/leg_power.json", 42: 2.38, 52: 2.03, 60: 2.24},
    "ledger_leg_work_max_over_min": {"source": "gait_report/phase1/fix/pilot3_final/ledger_summary.txt",
                                     42: 2.05802, 52: 1.97467, 60: 1.79504},
}
REPORT_ONLY = ("report only (criteria_pilot4 v3 'Report only'): no outcome, label, bar, gate or run decision reads "
               "this; it certifies nothing")


# ---------------------------------------------------------------- pure helpers (unit-tested without the tree)


def sha256_file(path: "str | Path") -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def med(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else float("nan")


def si(a: float, b: float) -> float:
    """survey_analyze.si: (L - R) / (0.5 (L + R)), + = left larger; NaN when undefined."""
    return float((a - b) / (0.5 * (a + b))) if np.isfinite(a) and np.isfinite(b) and (a + b) != 0 else float("nan")


def moving_avg(x, n: int) -> np.ndarray:
    """survey_analyze / ct_handbuilt movavg: the n-sample boxcar, valid part only."""
    c = np.cumsum(np.insert(np.asarray(x, dtype=float), 0, 0.0))
    return (c[n:] - c[:-n]) / n


def counter_turn(psi) -> float:
    """ct_handbuilt.ct: max over t1 < t2 of s (psi(t1) - psi(t2)), s = +1 if psi ends at or above its start."""
    psi = np.asarray(psi, dtype=float)
    if psi.size < 2:
        return float("nan")
    s = 1.0 if psi[-1] >= psi[0] else -1.0
    return float(np.max(np.maximum.accumulate(s * psi) - s * psi))


def t_start(root_pos, dt: float) -> float:
    """ct_handbuilt.tstart: the first centred time >= 1 s at which the 0.5-s mean horizontal speed reaches 90% of the
    median speed over the next 2 s; NaN if none (or the 2 s run past the episode)."""
    v = np.linalg.norm(np.diff(np.asarray(root_pos, dtype=float)[:, :2], axis=0), axis=1) / dt
    n05, n2 = int(round(0.5 / dt)), int(round(2.0 / dt))
    vm = moving_avg(v, n05)
    for j in range(vm.size):
        tc = (j + n05 / 2) * dt
        if tc < 1.0:
            continue
        k = int(round(tc / dt))
        fut = v[k:k + n2]
        if fut.size < n2 // 2:
            return float("nan")
        if vm[j] >= 0.9 * np.median(fut):
            return tc
    return float("nan")


def debounce_blips_first(x, n: int = 2) -> np.ndarray:
    """stride_check.py's debounce: interior True runs shorter than n set False, then interior False runs set True."""
    x = np.asarray(x, dtype=bool).copy()
    for val in (True, False):
        i = 0
        while i < len(x):
            if x[i] == val:
                j = i
                while j < len(x) and x[j] == val:
                    j += 1
                if j - i < n and i > 0 and j < len(x):
                    x[i:j] = not val
                i = j
            else:
                i += 1
    return x


def rising(x) -> np.ndarray:
    x = np.asarray(x, dtype=bool)
    return np.flatnonzero(~x[:-1] & x[1:]) + 1


def debounce_gaps_first(b, min_run: int = 2) -> np.ndarray:
    """gait_probe.debounce (a copy for the unit tests; the tool calls the frozen probe's own)."""
    out = np.asarray(b, dtype=bool).copy()

    def runs(z):
        if z.size == 0:
            return []
        change = np.flatnonzero(np.diff(z.astype(np.int8))) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [z.size]])
        return [(bool(z[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]

    for value, start, length in runs(out):
        if not value and length < min_run and start > 0 and start + length < out.size:
            out[start:start + length] = True
    for value, start, length in runs(out):
        if value and length < min_run and start > 0 and start + length < out.size:
            out[start:start + length] = False
    return out


def events(b) -> tuple[np.ndarray, np.ndarray]:
    """gait_probe._events: (touchdowns, lift-offs) of a debounced contact array."""
    d = np.diff(np.asarray(b, dtype=np.int8))
    return np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1


def quat_yaw(q) -> float:
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def quat_euler(q) -> tuple[float, float, float]:
    w, x, y, z = (float(v) for v in q)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    return roll, pitch, math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def per_touchdown_metrics(E: dict, *, iL: int, iR: int, dt: float, bw: float, debounce=debounce_gaps_first,
                          evts=events) -> dict:
    """survey_analyze.episode_metrics' contact part, line for line, on one episode's trace arrays:
    root_pos, quat, foot_pos, foot_h (T+1 rows), floor_contact_substep_frac, ffoot_max, ffoot_mean (T rows)."""
    T = int(E["length"])
    W = np.arange(SETTLE, T)
    nW = W.size
    root = np.asarray(E["root_pos"], float)[1:][W]
    quat = np.asarray(E["quat"], float)[1:][W]
    pos = np.asarray(E["foot_pos"], float)[1:][W]
    fh = np.asarray(E["foot_h"], float)[1:][W]  # noqa: F841 - survey_analyze line; its use there (clearance, impulse) is not ported
    frac = np.asarray(E["floor_contact_substep_frac"], float)[W]
    fmax = np.asarray(E["ffoot_max"], float)[W]
    fmean = np.asarray(E["ffoot_mean"], float)[W]  # noqa: F841 - survey_analyze line; its use there (clearance, impulse) is not ported
    yaw = np.degrees(np.unwrap(np.array([quat_euler(q)[2] for q in quat])))
    psi = moving_avg(yaw, int(round(1.0 / dt)))
    h0 = np.array([math.cos(math.radians(psi[0])), math.sin(math.radians(psi[0]))])
    down = frac >= 0.5
    td, lo = {}, {}
    for f in range(2):
        td[f], lo[f] = evts(debounce(down[:, f]))

    def travel_dir(k):
        a, b = max(k - TRAVEL_HALF_STEPS, 0), min(k + TRAVEL_HALF_STEPS, nW - 1)
        v = root[b, :2] - root[a, :2]
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else h0

    per = {f: {k: [] for k in ("step_len", "land_peak_bw", "stance_s", "scuff")} for f in range(2)}
    for f in range(2):
        o = 1 - f
        for i in td[f]:
            dv = travel_dir(i)
            rel = pos[i, f, :2] - pos[i, o, :2]
            per[f]["step_len"].append(float(rel @ dv))
            per[f]["land_peak_bw"].append(float(fmax[i: i + LANDING_STEPS, f].max() / bw))
            nxt = lo[f][lo[f] > i]
            if nxt.size:
                per[f]["stance_s"].append((int(nxt[0]) - i) * dt)
        for a in lo[f]:
            nx = td[f][td[f] > a]
            if not nx.size:
                continue
            b = int(nx[0])
            inner = frac[a + SCUFF_EDGE_STEPS: b - SCUFF_EDGE_STEPS, f]
            per[f]["scuff"].append(float(np.any(inner > 0.0)) if inner.size else 0.0)
    r = {}
    for k in ("step_len", "land_peak_bw", "stance_s"):
        L, R = med(per[iL][k]), med(per[iR][k])
        r[f"{k}_L"], r[f"{k}_R"], r[f"{k}_SI"] = L, R, si(L, R)
    for side, fi in (("L", iL), ("R", iR)):
        lp = np.asarray(per[fi]["land_peak_bw"])
        r[f"land_peak_p95_bw_{side}"] = float(np.percentile(lp, 95)) if lp.size else float("nan")
        sc = per[fi]["scuff"]
        r[f"scuff_share_{side}"] = float(np.mean(sc)) if sc else float("nan")
        r[f"scuff_swings_{side}"] = len(sc)
        r[f"scuff_touched_{side}"] = int(sum(sc))
    return r


def stride_rom(frac, qj_deg: dict, *, iL: int, iR: int) -> dict:
    """stride_check.py's per-own-foot-stride range: whole-episode contact (frac >= 0.5) with its own debounce,
    touchdowns at or after step 100; per stride q[a:b].max() - q[a:b].min(); per side the median; ratio min/max.
    qj_deg[name] holds the state after each step (index k = after step k), in degrees."""
    frac = np.asarray(frac, dtype=float)
    down = np.stack([debounce_blips_first(frac[:, f] >= 0.5) for f in range(2)], axis=1)
    td = {f: rising(down[:, f]) for f in range(2)}
    td = {f: td[f][td[f] >= SETTLE] for f in range(2)}
    out = {}
    for jn in ROM_JOINTS:
        per = {}
        for side, f, pre in (("L", iL, "l_"), ("R", iR, "r_")):
            q = np.asarray(qj_deg[pre + jn], dtype=float)
            r = [q[a:b].max() - q[a:b].min() for a, b in zip(td[f][:-1], td[f][1:])]
            per[side] = float(np.median(r)) if r else float("nan")
        out[f"{jn}_rom_L_deg"], out[f"{jn}_rom_R_deg"] = per["L"], per["R"]
        hi = max(per.values())
        out[f"{jn}_rom_ratio"] = float(min(per.values()) / hi) if np.isfinite(hi) and hi > 0 else float("nan")
    return out


def weave(root_pos, quat, dt: float) -> dict:
    """ct_handbuilt.py: t_start and the counter-turn of the 100-sample mean yaw (deg) from t_start."""
    ts = t_start(root_pos, dt)
    if not np.isfinite(ts):
        return {"t_start_s": None, "counter_turn_deg": float("nan"), "why_not": "no t_start (never reached speed)"}
    yaw = np.degrees(np.unwrap(np.array([quat_yaw(q) for q in np.asarray(quat, dtype=float)])))
    ks = int(round(ts / dt))
    return {"t_start_s": ts, "counter_turn_deg": counter_turn(moving_avg(yaw[ks:], MOVAVG_SAMPLES))}


def read_ledger(path: "str | Path") -> tuple[list[str], list[dict]]:
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _number(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def ledger_asymmetry(header: list[str], rows: list[dict], *, through: int,
                     evaluations: int = LEDGER_EVALUATIONS) -> dict:
    """training_history.gait_ledger's ASYMMETRY metric: the last *evaluations* evaluations at or before *through*,
    each evaluation's full-horizon episodes (full_horizon == 1, else cause == 'truncated'), pooled; per episode
    r_leg_work_abs_j / l_leg_work_abs_j (NaN unless both finite, left > 0, right >= 0); median; max/min."""
    need = {"timesteps", "r_leg_work_abs_j", "l_leg_work_abs_j"}
    if not need <= set(header):
        return {"read": False, "why_not": f"the ledger lacks {sorted(need - set(header))}"}
    present = set(header)
    by_step: dict[int, list[dict]] = {}
    for row in rows:
        step = _number(row.get("timesteps"))
        if math.isfinite(step) and int(step) <= int(through):
            by_step.setdefault(int(step), []).append(row)
    steps = sorted(by_step)
    if not steps:
        return {"read": False, "why_not": f"no evaluation at or before {int(through):,}"}

    def full(row):
        if "full_horizon" in present:
            v = _number(row.get("full_horizon"))
            if math.isfinite(v):
                return v == 1.0
        if "cause" in present and (row.get("cause") or "").strip():
            return (row.get("cause") or "").strip() == "truncated"
        return None

    used = steps[-min(evaluations, len(steps)):]
    window = []
    for s in used:
        group = by_step[s]
        flags = [full(r) for r in group]
        known = [f for f in flags if f is not None]
        window += [r for r, f in zip(group, flags) if f] if known else list(group)
    right = np.asarray([_number(r.get("r_leg_work_abs_j")) for r in window], dtype=float)
    left = np.asarray([_number(r.get("l_leg_work_abs_j")) for r in window], dtype=float)
    ok = np.isfinite(right) & np.isfinite(left) & (left > 0.0) & (right >= 0.0)
    ratio = np.full(right.shape, np.nan)
    np.divide(right, left, out=ratio, where=ok)
    finite = ratio[np.isfinite(ratio)]
    if not finite.size:
        return {"read": False, "why_not": "no full-horizon episode with both legs' work in the window"}
    m = float(np.median(finite))
    return {"read": True, "evaluations": len(used), "from_timesteps": used[0], "to_timesteps": used[-1],
            "episodes": int(finite.size), "r_over_l_median": m, "max_over_min": max(m, 1.0 / m) if m > 0 else None,
            "larger_side": "right" if m > 1 else "left" if m < 1 else "neither",
            "right_larger": int((finite > 1.0).sum()), "left_larger": int((finite < 1.0).sum()),
            "r_leg_work_abs_j_median": med(right[ok]), "l_leg_work_abs_j_median": med(left[ok])}


def aggregate(values) -> "dict | None":
    v = np.asarray([x for x in values if x is not None], dtype=float)
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    return {"n": int(v.size), "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def leg_power(per_episode_power: list[dict], actuator_names: list[str]) -> dict:
    """leg_power.py: per side the sum over hip pitch, hip roll, knee, ankle of each actuator's median (over the
    full-length episodes) of its mean |power| over window steps; max/min."""
    if not per_episode_power:
        return {"read": False, "why_not": "no full-length episode"}
    meds = {n: med([e[n] for e in per_episode_power]) for n in actuator_names}
    side = {}
    for s in "rl":
        names = [f"{s}_{j}_act" for j in LEG_JOINTS]
        if any(n not in meds for n in names):
            return {"read": False, "why_not": f"actuators {names} not all present"}
        side[s] = float(sum(meds[n] for n in names))
    lo, hi = min(side.values()), max(side.values())
    return {"read": True, "R_W": side["r"], "L_W": side["l"], "max_over_min": hi / lo if lo > 0 else None,
            "larger_side": "right" if side["r"] > side["l"] else "left"}


def run_directory(path: "str | Path") -> "Path | None":
    resolved = Path(path).resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "provenance.json").is_file():
            return candidate
    return None


def output_problems(json_path: "str | Path", *, tree: "Path | None", inputs: list, run_copy: "str | None") -> list[str]:
    problems = []
    target = Path(json_path).resolve()
    if target.exists():
        problems.append(f"--json {json_path} exists (never overwritten; move it or name another file)")
    if tree is not None and target.is_relative_to(Path(tree).resolve()):
        problems.append(f"--json {target} is inside the tree {tree} (the tree stays clean)")
    if run_copy and target.is_relative_to(Path(run_copy).resolve()):
        problems.append(f"--json {target} is inside the cell's run copy {run_copy}")
    for source in inputs:
        if source is None:
            continue
        run = run_directory(Path(source).parent)
        if run is not None and target.is_relative_to(run):
            problems.append(f"--json {target} is inside the run directory {run} (a result bundle rejects files it did "
                            "not list)")
    return problems


def cell_problems(binding: dict, *, head: str, dirty: str, validation: bool) -> list[str]:
    problems = []
    schema = binding.get("schema")
    if schema not in (PILOT4_CELL_SCHEMA, PILOT3_CELL_SCHEMA):
        problems.append(f"binding.json schema {schema!r} is not a judge cell's")
    elif schema == PILOT3_CELL_SCHEMA and not validation:
        problems.append("the cell is pilot 3's (walking-pilot3-judge/v1); --validation reads it for a check only")
    if binding.get("head") != head:
        problems.append(f"the cell was judged at {binding.get('head')}, this tree is {head}")
    if dirty:
        problems.append("this tree has local edits (git status --porcelain --untracked-files=no is not empty)")
    if binding.get("off_schedule"):
        problems.append(f"the cell is off-schedule ({binding.get('off_schedule_reason')}): never read")
    if binding.get("validation_only") and not validation:
        problems.append("the cell is validation-only; --validation reads it for a check only")
    return problems


def parent_paths(runs: "str | Path", seed: int) -> tuple[Path, Path]:
    base = Path(runs) / f"pilot3_walk_s{seed}" / "03_locomotion" / "models"
    return base / "stage2_8000000_steps.zip", base / "stage2_vecnormalize_8000000_steps.pkl"


def parent_problems(seed: int, zip_path: Path, pkl_path: Path) -> list[str]:
    if seed not in PARENTS:
        return [f"--parent {seed} is not one of the scored seeds {sorted(PARENTS)}"]
    problems = []
    for path, want in zip((zip_path, pkl_path), PARENTS[seed]):
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif sha256_file(path) != want:
            problems.append(f"{path} is not s{seed}'s 8M parent ({want})")
    return problems


def jsafe(value):
    if isinstance(value, dict):
        return {str(k): jsafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsafe(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsafe(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------- the tree side


def install_probe_shims(embodiment: str = EMBODIMENT) -> None:
    """gait_probe_pilot.py's four shims (pilot3_video.py's install_probe_shims)."""
    from environments.shared import config, plant_contract, species_registry, stage_manifest
    from environments.shared.model_id import ModelId
    from environments.shared.species_names import require_model_id

    def mid(species):
        return species if isinstance(species, ModelId) else require_model_id(str(species), embodiment)

    _load_stage_config = config.load_stage_config
    _load_stage_manifest = stage_manifest.load_stage_manifest
    _current_plant_identity = plant_contract.current_plant_identity

    class _Factories(dict):
        def __missing__(self, species):
            return lambda: species_registry.get_species_config(mid(species))

    species_registry.SPECIES_FACTORIES = _Factories()  # type: ignore[attr-defined]
    config.load_stage_config = lambda species, stage, *a, **k: _load_stage_config(mid(species), stage, *a, **k)
    stage_manifest.load_stage_manifest = lambda species, *a, **k: _load_stage_manifest(mid(species), *a, **k)
    plant_contract.current_plant_identity = lambda species, *a, **k: _current_plant_identity(mid(species), *a, **k)


def load_probe(tree: Path):
    path = tree / "docs" / "investigations" / "gait_2026_09" / "gait_probe.py"
    spec = importlib.util.spec_from_file_location("gait_probe_frozen", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def git(tree: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=tree, text=True,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()


def main(argv=None) -> int:  # noqa: C901 - one linear procedure, sectioned
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cell", help="a judged cell OUT/s<seed>/<steps> (binding.json, probe.json)")
    src.add_argument("--parent", type=int, help="a seed: its pilot-3 8M pair (edge-prepared in a live-command tree)")
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="the folder holding pilot3_walk_s<seed>/ (--parent)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--episodes", type=int, default=PANEL_EPISODES)
    ap.add_argument("--validation", action="store_true", help="read pilot 3's or a validation-only cell (checks only)")
    a = ap.parse_args(argv)
    tree = Path.cwd().resolve()
    tool_sha = sha256_file(__file__)  # the bytes that run, read before anything else

    def refuse(problems: list[str]) -> int:
        print("pilot4_mech: refusing (nothing written): " + "; ".join(problems), flush=True)
        return 2

    early = [] if 1 <= a.episodes <= PANEL_EPISODES else [f"--episodes must be 1..{PANEL_EPISODES} (the panel)"]
    if Path(a.json).exists():
        early.append(f"--json {a.json} exists (never overwritten; move it or name another file)")
    if early:  # before anything is imported from the tree
        return refuse(early)
    import environments

    env_file = str(Path(environments.__file__).resolve())
    if not env_file.startswith(str(tree) + os.sep):
        return refuse([f"environments imported from {env_file}, not from the tree {tree} (run through iso.py)"])
    try:
        head, dirty = git(tree, "rev-parse", "HEAD"), git(tree, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        return refuse([f"the tree {tree} is not a readable git checkout: {exc}"])

    binding, rows, meta, run_copy = None, None, None, None
    if a.cell:
        cell = Path(a.cell).resolve()
        try:
            binding = json.loads((cell / "binding.json").read_text())
            doc = json.loads((cell / "probe.json").read_text())
        except (OSError, ValueError) as exc:
            return refuse([f"the cell {cell} cannot be read: {exc}"])
        meta, rows = doc["meta"], doc["episodes"]
        problems = cell_problems(binding, head=head, dirty=dirty, validation=a.validation)
        ckpt, vec = Path(binding["checkpoint"]), Path(binding["vecnorm"])
        for p, want in ((ckpt, binding.get("checkpoint_sha256")), (vec, binding.get("vecnorm_sha256"))):
            if not p.is_file() or sha256_file(p) != want:
                problems.append(f"{p} is missing or no longer the judged file ({want})")
        if meta.get("seed_scheme") != "panel" or int(meta.get("seed_base", -1)) != PANEL_SEED_BASE:
            problems.append(f"probe.json's seeds are {meta.get('seed_scheme')!r} from {meta.get('seed_base')}, not the "
                            f"panel from {PANEL_SEED_BASE}")
        if len(rows) < a.episodes:
            problems.append(f"probe.json holds {len(rows)} episodes, fewer than {a.episodes}")
        seed, steps = int(binding["seed"]), int(binding["steps"])
        run_copy = binding.get("run")
        ledger = Path(str(run_copy)) / "03_locomotion" / "eval_episodes.csv" if run_copy else None
    else:
        seed, steps = int(a.parent), 8_000_000
        ckpt, vec = parent_paths(a.runs, seed)
        problems = parent_problems(seed, ckpt, vec)
        if dirty:
            problems.append("this tree has local edits (git status --porcelain --untracked-files=no is not empty)")
        ledger = Path(a.runs) / f"pilot3_walk_s{seed}" / "03_locomotion" / "eval_episodes.csv"
    problems += output_problems(a.json, tree=tree, inputs=[ckpt, vec, ledger], run_copy=run_copy)
    if problems:
        return refuse(problems)

    install_probe_shims()
    probe, probe_path = load_probe(tree)
    import gymnasium as gym
    import mujoco
    import torch as th

    from environments.shared.config import load_stage_config
    from environments.shared.plant_contract import current_plant_identity, validate_environment_plant
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import get_species_config

    th.set_num_threads(1)
    mid = require_model_id(SPECIES, EMBODIMENT)
    stage_cfg = load_stage_config(mid, STAGE)
    env_kwargs = dict(stage_cfg["env_kwargs"])
    env_class = get_species_config(mid).env_class
    plant = current_plant_identity(mid)
    if meta is not None:
        from environments.shared.curriculum.stance_gate import StanceGateThresholds

        gate_settle = int(StanceGateThresholds.from_curriculum(stage_cfg.get("curriculum_kwargs", {}),
                                                               require_criteria=False).settle_steps)
        same = json.loads(json.dumps(probe._json_safe(env_kwargs))) == meta.get("env_kwargs")
        if not same or gate_settle != meta.get("settle_steps_gate"):
            return refuse(["this tree's locomotion env kwargs or gate settle differ from probe.json's"])
        settle = int(meta["settle_steps_gait_window"])
    else:
        gate_settle, settle = 0, SETTLE
    live = str(env_kwargs.get("command_mode", "none")) != "none"
    horizon = int(env_kwargs.get("max_episode_steps", 1000))

    def factory():
        return env_class(**env_kwargs)

    model, normalizer, _ = load_sb3_checkpoint(str(ckpt), str(vec), factory, guess_sidecar=False,
                                               allow_unnormalized=False, plant_identity=plant,
                                               allow_legacy_plant=False)
    prepared = {"edge_prepared": False}
    check = None
    if a.parent is not None and live:
        from environments.shared.command_frame import reseed_command_slice
        from environments.shared.policy_loading import assert_command_blind, neutralize_command_columns

        orig, orig_norm, _ = load_sb3_checkpoint(str(ckpt), str(vec), factory, guess_sidecar=False,
                                                 allow_unnormalized=False, plant_identity=plant,
                                                 allow_legacy_plant=False)
        reseed_command_slice(normalizer.obs_rms)
        assert_command_blind(model, normalizer, prepare=lambda: neutralize_command_columns(model, observation_dim=63))
        om, ov = np.asarray(orig_norm.obs_rms.mean, float), np.asarray(orig_norm.obs_rms.var, float)
        oe, oc = float(orig_norm.epsilon), float(orig_norm.clip_obs)

        def check(o):
            z = np.array(o, dtype=np.float64, copy=True)
            z[COMMAND_SLICE] = 0.0
            x = np.clip((z - om) / np.sqrt(ov + oe), -oc, oc).astype(np.float32)
            return np.asarray(orig.predict(x, deterministic=True)[0], dtype=np.float64)

        prepared = {"edge_prepared": True, "how": "command_frame.reseed_command_slice(obs_rms); policy_loading."
                    "assert_command_blind(prepare=neutralize_command_columns(observation_dim=63)), as recov.py"}
    mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float64)
    var = np.asarray(normalizer.obs_rms.var, dtype=np.float64)
    eps, clip = float(normalizer.epsilon), float(normalizer.clip_obs)
    max_dA = [0.0]

    def predict(obs):  # the probe's own predict
        x = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        act = np.asarray(model.predict(x, deterministic=True)[0])
        if check is not None:
            max_dA[0] = max(max_dA[0], float(np.max(np.abs(np.asarray(act, dtype=np.float64) - check(obs)))))
        return act

    base = factory()
    validate_environment_plant(base, plant, artifact="pilot4_mech env")
    rd = probe.RobotDescription(base)
    m = base.unwrapped.model
    act_names = [m.actuator(i).name for i in range(m.nu)]

    class Extra(gym.Wrapper):
        """Pass-through; a hook chained after the probe's own substep hook records, per substep, each foot's floor
        normal force (summed as the probe's hook sums it) and each actuator's force x velocity (read only)."""

        def __init__(self, env):
            super().__init__(env)
            self.fmax, self.apow = [], []

        def reset(self, **kw):
            self.fmax, self.apow = [], []
            return self.env.reset(**kw)

        def step(self, action):
            u = self.env.unwrapped
            d = u.data
            probe_hook = u._substep_probe_hook
            sub_f, sub_p = [], []
            f6 = np.zeros(6)

            def chained() -> None:
                if probe_hook is not None:
                    probe_hook()
                fl = np.zeros(rd.n_feet)
                for ci in range(d.ncon):
                    c = d.contact[ci]
                    g1, g2 = int(c.geom1), int(c.geom2)
                    f1, f2 = rd.geom_foot[g1], rd.geom_foot[g2]
                    if (f1 < 0 and f2 < 0) or not (rd.is_floor[g1] or rd.is_floor[g2]):
                        continue
                    mujoco.mj_contactForce(m, d, ci, f6)
                    fl[f1 if f1 >= 0 else f2] += float(f6[0])
                sub_f.append(fl)
                sub_p.append(np.asarray(d.actuator_force, dtype=float) * np.asarray(d.actuator_velocity, dtype=float))

            u._substep_probe_hook = chained
            try:
                out = self.env.step(action)
            finally:
                u._substep_probe_hook = probe_hook
            self.fmax.append(np.max(np.asarray(sub_f), axis=0))
            self.apow.append(np.mean(np.asarray(sub_p), axis=0))
            return out

    env = Extra(base)
    dt = float(base.unwrapped.dt)
    disp = list(rd.display)
    iL, iR = disp.index("L"), disp.index("R")
    hinge = list(rd.hinge_names)
    bw = float(rd.body_weight_n)
    episodes, matches, power_rows = [], [], []
    for i in range(a.episodes):
        rs = PANEL_SEED_BASE + i
        tr = probe.rollout_episode(env, predict, rd, rs)
        row = json.loads(json.dumps(probe._json_safe(probe.analyze_episode(tr, rd, dt, settle, gate_settle))))
        if rows is not None:
            diff = sorted(k for k in (set(row) | set(rows[i])) - {"episode", "reset_seed"}
                          if row.get(k, "<absent>") != rows[i].get(k, "<absent>"))
            matches.append({"episode": i, "matches_probe": not diff, "differing_keys": diff[:20]})
        T = int(tr["length"])
        full = bool(T == horizon and not tr["terminated"])
        ep = {"episode": i, "reset_seed": rs, "length": T, "terminated": bool(tr["terminated"]), "full": full}
        for k in ("lead_mean_m", "floor_duty_asymmetry_L_minus_R", "roll_mean_deg", "jointlag_antiphase_index_ankle",
                  "yaw_change_deg"):
            ep[k] = row.get(k)
        if T > SETTLE + 50:
            E = {"length": T, "root_pos": tr["root_pos"], "quat": tr["quat"], "foot_pos": tr["foot_pos"],
                 "foot_h": tr["foot_h"], "floor_contact_substep_frac": tr["floor_contact_substep_frac"],
                 "ffoot_max": np.asarray(env.fmax), "ffoot_mean": tr["floor_force_mean"]}
            ep.update(per_touchdown_metrics(E, iL=iL, iR=iR, dt=dt, bw=bw, debounce=probe.debounce,
                                            evts=probe._events))
            qdeg = {n: np.degrees(np.asarray(tr["qj"], float)[1:, j]) for j, n in enumerate(hinge)}
            ep.update(stride_rom(tr["floor_contact_substep_frac"], qdeg, iL=iL, iR=iR))
            ep.update(weave(tr["root_pos"], tr["quat"], dt))
            if full:
                ap_ = np.abs(np.asarray(env.apow)[SETTLE:T])
                power_rows.append({n: float(ap_[:, j].mean()) for j, n in enumerate(act_names)})
        episodes.append(ep)
        print(f"  ep {i:2d} reset {rs}: length {T} knee L/R {ep.get('knee_rom_L_deg', float('nan')):.1f}/"
              f"{ep.get('knee_rom_R_deg', float('nan')):.1f} landing L/R {ep.get('land_peak_bw_L', float('nan')):.2f}/"
              f"{ep.get('land_peak_bw_R', float('nan')):.2f} BW"
              + ("" if rows is None else ("  matches probe.json" if not matches[-1]["differing_keys"]
                                          else f"  DIFFERS {matches[-1]['differing_keys'][:4]}")), flush=True)
    env.close()
    full_eps = [e for e in episodes if e["full"]]
    keys = ["knee_rom_L_deg", "knee_rom_R_deg", "knee_rom_ratio", "ankle_rom_L_deg", "ankle_rom_R_deg",
            "ankle_rom_ratio", "hip_pitch_rom_L_deg", "hip_pitch_rom_R_deg", "hip_pitch_rom_ratio",
            "land_peak_bw_L", "land_peak_bw_R", "land_peak_bw_SI", "land_peak_p95_bw_L", "land_peak_p95_bw_R",
            "step_len_L", "step_len_R", "step_len_SI", "stance_s_L", "stance_s_R", "stance_s_SI",
            "lead_mean_m", "floor_duty_asymmetry_L_minus_R", "roll_mean_deg", "jointlag_antiphase_index_ankle",
            "yaw_change_deg", "scuff_share_L", "scuff_share_R", "t_start_s", "counter_turn_deg"]
    agg = {k: aggregate([e.get(k) for e in full_eps]) for k in keys}
    pooled = {}
    for side in ("L", "R"):
        sw = sum(int(e.get(f"scuff_swings_{side}", 0)) for e in full_eps)
        tc = sum(int(e.get(f"scuff_touched_{side}", 0)) for e in full_eps)
        pooled[side] = {"swings": sw, "touched": tc, "share": tc / sw if sw else None}
    try:
        header, lrows = read_ledger(ledger)
        ledger_part = ledger_asymmetry(header, lrows, through=steps)
        ledger_part.update(path=str(ledger), sha256=sha256_file(ledger))
    except OSError as exc:
        ledger_part = {"read": False, "why_not": f"cannot read {ledger}: {exc}"}
    all_match = all(x["matches_probe"] for x in matches) if matches else None
    record = {
        "schema": SCHEMA, "report_only": REPORT_ONLY, "tool_sha256": tool_sha, "tree": str(tree),
        "head": head, "environments_file": env_file, "probe": str(probe_path), "probe_sha256": sha256_file(probe_path),
        "mode": "cell" if a.cell else "parent", "cell": str(Path(a.cell).resolve()) if a.cell else None,
        "binding_schema": (binding or {}).get("schema"), "validation": bool(a.validation), "seed": seed,
        "steps": steps, "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt), "vecnorm": str(vec),
        "vecnorm_sha256": sha256_file(vec), "env_command_mode": str(env_kwargs.get("command_mode", "none")),
        "horizon": horizon, "settle_steps_gait_window": settle, "settle_steps_gate": gate_settle, **prepared,
        "edge_check_max_action_delta": max_dA[0] if check is not None else None,
        "body_weight_n": bw, "dt": dt, "panel_seed_base": PANEL_SEED_BASE, "episodes_rolled": a.episodes,
        "full_length_episodes": len(full_eps), "episodes_match_probe": matches, "all_episodes_match_probe": all_match,
        "aggregate_full_length": agg, "scuff_pooled": pooled,
        "leg_power": leg_power(power_rows, act_names), "ledger_asymmetry": ledger_part,
        "episodes": episodes, "pilot3_reference": PILOT3_REFERENCE,
    }
    out = Path(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsafe(record), indent=1))
    for line in describe(record):
        print(line, flush=True)
    print(f"pilot4_mech: wrote {out}", flush=True)
    return 0 if all_match in (None, True) else 1


def describe(r: dict) -> list[str]:
    seed, A, ref = r["seed"], r["aggregate_full_length"], r["pilot3_reference"]

    def v(key, fmt="{:.2f}"):
        d = A.get(key)
        return "n/a" if not d else fmt.format(d["median"])

    def p3(key):
        return ref.get(key, {}).get(seed, "n/a")

    lines = [f"MECHANICS (report only) s{seed} at {r['steps']:,} [{r['mode']}"
             f"{', edge-prepared' if r['edge_prepared'] else ''}; command_mode {r['env_command_mode']}]; "
             f"{r['full_length_episodes']} of {r['episodes_rolled']} episodes full length; medians",
             f"  knee range per stride L {v('knee_rom_L_deg', '{:.1f}')} / R {v('knee_rom_R_deg', '{:.1f}')} deg, "
             f"ratio {v('knee_rom_ratio', '{:.3f}')} (pilot 3 8M: {p3('knee_rom_L_R_deg')}, {p3('knee_rom_ratio')})",
             f"  ankle ratio {v('ankle_rom_ratio', '{:.3f}')}, hip pitch ratio {v('hip_pitch_rom_ratio', '{:.3f}')}; "
             f"ankle anti-phase index {v('jointlag_antiphase_index_ankle', '{:.3f}')} (pilot 3: "
             f"{p3('ankle_antiphase_index')})",
             f"  landing peak L {v('land_peak_bw_L', '{:.3f}')} / R {v('land_peak_bw_R', '{:.3f}')} BW, SI "
             f"{v('land_peak_bw_SI', '{:+.3f}')}; p95 L {v('land_peak_p95_bw_L')} / R {v('land_peak_p95_bw_R')} "
             f"(pilot 3: {p3('landing_peak_bw_L_R_SI')})",
             f"  fore-aft placement lead_mean_m {v('lead_mean_m', '{:+.3f}')} m (left ahead +); step L "
             f"{v('step_len_L', '{:.3f}')} / R {v('step_len_R', '{:.3f}')} m (pilot 3: {p3('lead_mean_m')}, "
             f"{p3('step_len_L_R_m')})",
             f"  duty L-R {v('floor_duty_asymmetry_L_minus_R', '{:+.3f}')} (pilot 3: {p3('duty_L_minus_R')}); stance "
             f"SI {v('stance_s_SI', '{:+.3f}')}; roll mean {v('roll_mean_deg', '{:+.2f}')} deg, + = right side down "
             f"(pilot 3: {p3('roll_mean_deg')})"]
    pooled = r["scuff_pooled"]
    lines.append(f"  scuff: right {pooled['R']['touched']}/{pooled['R']['swings']} swings pooled, per-episode median "
                 f"{v('scuff_share_R', '{:.3f}')}; left {pooled['L']['touched']}/{pooled['L']['swings']} (pilot 3: "
                 f"{p3('scuff_R_pooled_touched_of_swings')}, {p3('scuff_share_R_per_episode_median')})")
    lines.append(f"  weave counter-turn from t_start {v('counter_turn_deg', '{:.2f}')} deg (pilot 3: "
                 f"{p3('counter_turn_deg')})")
    lp = r["leg_power"]
    lines.append("  leg power " + (f"R {lp['R_W']:.1f} W / L {lp['L_W']:.1f} W, max/min {lp['max_over_min']:.2f} "
                                   f"({lp['larger_side']} larger)" if lp.get("read") else f"not run ({lp['why_not']})")
                 + f" (pilot 3: {p3('leg_power_max_over_min')})")
    la = r["ledger_asymmetry"]
    lines.append("  ledger leg work " + (
        f"r/l median {la['r_over_l_median']:.5f}, max/min {la['max_over_min']:.5f} ({la['larger_side']} larger in "
        f"{max(la['right_larger'], la['left_larger'])} of {la['episodes']}), evaluations {la['from_timesteps']:,} to "
        f"{la['to_timesteps']:,}" if la.get("read") else f"not run ({la['why_not']})")
        + f" (pilot 3: {p3('ledger_leg_work_max_over_min')})")
    if r["all_episodes_match_probe"] is not None:
        lines.append(f"  episodes equal probe.json's: {r['all_episodes_match_probe']}")
    if r["edge_check_max_action_delta"] is not None:
        lines.append(f"  edge-prepared walker vs the original on command-zeroed observations: max |dA| "
                     f"{r['edge_check_max_action_delta']:.3g}")
    lines.append(f"  {REPORT_ONLY}")
    return lines


if __name__ == "__main__":
    sys.exit(main())
