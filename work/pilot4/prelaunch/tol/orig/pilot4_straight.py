"""Walking pilot 4's straightness tool: the pre-registered STRAIGHT rule (criteria_pilot4 draft v3, "STRAIGHT").

Scratch, never merged; built, unit-tested, validated and pinned by sha256 before launch (v3 "TOOL PLAN",
"pilot4_straight.py (new)"). Deterministic rollouts and arithmetic only: nothing here trains (the training entry points
are blocked in the tree modes before any checkpoint is loaded).

THE RULE (v3 "STRAIGHT"; every bar a named constant below, every comparison inclusive: "<=")
  psi(t)   the unwrapped pelvis yaw from the probe's recorded quaternion (the probe's _quat_to_euler, the formula
           yaw_change_deg reads), states 0..T, on the branch that puts psi[0] within half a turn of psi_cmd.
  psi_cmd  the command state's desired_heading (pilot-4 env); checked constant over the episode and against every
           step's info walk_heading_error_rad = wrap(yaw(q_t) - psi_cmd). In a command-free env (pilot 3's, only under
           --validation) the spawn heading (env._heading() after the reset) stands in, and the record says so.
  m_k      the mean of psi over states k-50..k+50 (101 states), k = 151..T-50.
  (a)      on the full-length (T = 1000) panel episodes: |median of the probe's yaw_change_deg| <= 5.0 AND
           p95 |yaw_change_deg| <= 15.0.
  (b)      D = max_k |m_k - psi_cmd|, W0 = max over states 101..T of |psi - psi_cmd|, full-length episodes:
           p95 D <= 11.5 AND p95 W0 <= 18.5. Report only: R = max m_k - min m_k, the last mean's signed error m_{T-50}.
  (c)      the recovery panel: resets 3042-3061, each nudged once by +15 and once by -15 deg. The nudge: after state 650
           is recorded and step 651's action computed from its observation, and only if step 650 neither terminated
           nor truncated: the root quaternion q <- q_z(angle) x q, the root linear velocity qvel[0:3] (world frame)
           turned by the angle, qvel[3:6] left, the prey (data.mocap_pos[0]) turned by the angle about the vertical
           through the root (qpos[0:2]); no mj_forward, no observation rebuild; step 651's env.step integrates from the
           turned state and refreshes the command. COUNTS when the pelvis x at state 650 is above the prey x at state
           650 (before the turn); one that ends before the nudge, or has not passed, is left out (both counted).
           r = |m_950 - psi_cmd| (states 900-1000); r = inf when it ends before state 1000 or its pelvis moved less
           than 0.35 m (horizontal, states 650 -> 1000). Fewer than 30 counting: NOT READ. Pass: median r <= 5.0 AND
           p95 r <= 11.5. States 0..650 of every nudged episode must equal its panel episode's exactly, or the tool
           refuses. Report only: the share removed 1 - (e_nudged - e_twin) / angle per sign, the same nudge with the
           prey left in place, and the walker turned at the reset (the same mechanics at state 0, the prey left in
           place, read at m_500, states 450-550).
  Percentiles: numpy.percentile's default (linear); inf sorts last; any nan numpy returns is read as inf; inf fails.
  STRAIGHT = PASS only when (a), (b) and (c) are PASS (a NOT READ (c) is not a pass).

MODES (run through iso.py with the tree as the working directory; nothing is ever overwritten)
    cd TREE && ISO_WT=TREE MUJOCO_GL=disable OMP_NUM_THREADS=1 python -I TOOLS/iso.py TOOLS/pilot4_straight.py \\
        seed --seed 42 --run RUNS/pilot4_walk_s42 --cell OUT/s42/2000000 --json STRAIGHT/s42_2000000.json \\
        [--plot STRAIGHT/s42_2000000.png] [--traces STRAIGHT/s42_2000000.npz] [--validation]
    cd TREE && ISO_WT=TREE ... pilot4_straight.py baseline --seed 42 --checkpoint P3RUN/.../stage2_8000000_steps.zip \\
        --vecnorm P3RUN/.../stage2_vecnormalize_8000000_steps.pkl --json OUT.json [--plot PNG] [--traces NPZ] \\
        [--labels-from SCORER.py] [--zero-nudge-check] [--expect-head SHA] [--limit-resets N]
  seed      a judged cell (the judge's OUT/s<seed>/<steps>: binding.json, probe.json): re-rolls the 40 panel episodes
            through the probe's own rollout_episode / analyze_episode (pilot3_video.py's loader and shims) and uses an
            episode only if every re-rolled metric equals probe.json's; (a) reads probe.json's yaw_change_deg, and the
            psi-derived yaw must equal it to 1e-12 deg. Refused (exit 2, nothing written; every reason listed): an
            existing --json/--plot/--traces; another seed; another head; a dirty tree (git status --porcelain
            --untracked-files=no); an off-schedule cell; a validation-only cell without --validation; a cell from
            another run (binding "run" is not --run, the run folder is not pilot4_walk_s<seed>[_b], the pair is not in
            --run/03_locomotion/models/, or a sha256 no longer matches); a probe.json that is not the fixed panel
            (panel, 3042, 40 episodes, settle 100) or names another checkpoint, or whose own env-kwargs drift record
            (meta.env_kwargs_drift_vs_recorded: the run's recorded env kwargs against the tree's) is not empty (fix
            stage); this tree's locomotion env kwargs or
            gate settle differ from probe.json's; an env that is not the pilot-4 env (command_mode heading, prey
            [0, 0], price 0.2 / 0.5 rad), unless --validation. --validation reads a validation-only cell, a run named
            pilot<N>_walk_s<seed>[_b] and a command-free env, for checks only: the record says validation true.
  baseline  the entry baseline (v3 "Pre-launch checks"): a pilot-3 8M parent pair (its sha256s must be the seed's
            entry of PARENT_PAIRS) prepared in memory as the edge prepares it (command_frame.reseed_command_slice on
            the sidecar's obs_rms, policy_loading.assert_command_blind with neutralize_command_columns as its
            prepare), rolled in the committed pilot-4 env (refused otherwise) on the whole panel; every step's action
            is checked against the unprepared parent on the command-zeroed observation. --labels-from labels the rows
            with that scorer's own label_episode / seed_outcome (pilot 3's rule). The mechanics checks of the TOOL PLAN
            run on every nudged episode: the prefix, the state-651 observation (only obs[60:63] may differ beyond
            1e-3), the command-blind turned twin (heading at state 1000 = the twin's + angle within 1e-5 deg), and
            with --zero-nudge-check a 0-deg nudge per recovery reset that must reproduce its panel episode exactly.
            --limit-resets N (tests only) rolls panel episodes 0..N-1 and the first N recovery resets; such a record
            is partial and every part reads NOT READ.
Exit status: 0 done; 1 written, but an episode did not match probe.json or a mechanics check failed (the record
says which); 2 refused, nothing written; 3 an internal consistency check failed (psi's yaw against yaw_change_deg,
psi_cmd against info walk_heading_error_rad, a moving commanded heading), nothing written.

JSON (strict JSON; an infinite value is the string "inf"): schema, mode, validation, partial, tool and probe sha256s,
tree, head, environments_file, seed, steps, run, cell, checkpoint(+sha256), vecnorm(+sha256), constants (every bar),
env (the env keys read, pilot4_env), psi_cmd (source, values, the info check), panel (episodes: per episode T, full,
used, matches_probe, differing_keys, yaw_change_deg, yaw_from_psi_deg, D_deg, D_k, W0_deg, W0_state, R_deg,
last_mean_error_deg, crab_median_deg, end_cross_track_m, window_displacement_m, yaw_per_10m_deg, label), parts
(a, b, c: verdict PASS / FAIL / NOT READ with its values and counts, b with R and the last mean's error as report,
and, fix stage, report_crab_median_deg and report_end_cross_track_median_m, the medians over the same full-length
episodes of crab_median_deg and end_cross_track_m, also printed and written on the path panel of the look's figure;
straight PASS / FAIL), recovery (episodes: reset_seed, variant, angle_deg, T, reached_nudge, passed_prey_at_650,
pelvis_minus_prey_x_650_m, counting, moved_650_1000_m, m_read_error_deg, r_deg, r_reason, share_removed,
twin_m_read_error_deg, heading_minus_twin_at_1000_deg, obs651_channels_beyond_tol_vs_twin, obs60_63_at_651;
prey_in_place and turned_at_reset: summary and episodes), checks (prefix, obs651, turned_twin, zero_nudge;
baseline: action_delta_vs_blind), labels (baseline with --labels-from), files (plot, traces with sha256s).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA = "walking-pilot4-straight/v1"
SPECIES, EMBODIMENT, STAGE = "tyrannosaurus_rex", "anatomical", "locomotion"

# ---------------------------------------------------------------- the panel and the heading window (v3 "Terms")
HORIZON = 1000
PANEL_EPISODES = 40
PANEL_SEED_BASE = 3042
SETTLE = 100                  # the probe's gait window starts at 1 s: states 101..T
WINDOW_FIRST_STATE = 101      # yaw_change_deg = psi[T] - psi[101]; W0 over states 101..T
MEAN_HALF = 50                # m_k: the mean of psi over states k-50..k+50 (101 states)
K_FIRST = 151                 # k = 151..T-50, so every mean lies inside the gait window
YAW_MIN_LENGTH = 110          # the probe forms yaw_change_deg only with at least 10 window steps (T >= 110)
FULL_MIN_ON_WALKS = 36        # a WALKS seed has at most 4 falls, so at least 36 full-length episodes (printed only)

# ---------------------------------------------------------------- the bars (v3 Q1, Q20; maintainer decision (31))
A_MEDIAN_ABS_MAX_DEG = 5.0    # (a) |median of the signed yaw_change_deg|
A_P95_ABS_MAX_DEG = 15.0      # (a) p95 of |yaw_change_deg|
B_D_P95_MAX_DEG = 11.5        # (b) p95 of D
B_W0_P95_MAX_DEG = 18.5       # (b) p95 of W0, the zig-zag guard
C_MEDIAN_MAX_DEG = 5.0        # (c) median r
C_P95_MAX_DEG = 11.5          # (c) p95 r
C_MIN_COUNTING = 30           # (c) NOT READ below this many counting episodes
C_MIN_NET_MOVE_M = 0.35       # (c) r = inf below this net horizontal pelvis displacement, states 650 -> 1000
PCT = 95

# ---------------------------------------------------------------- the recovery panel (v3 (c))
RECOVERY_SEEDS = tuple(range(3042, 3062))
NUDGE_ANGLES_DEG = (15.0, -15.0)
NUDGE_STATE = 650             # the turn is applied after state 650, before step 651 integrates
READ_K = 950                  # r reads m_950: states 900..1000
RESET_TURN_STATE = 0          # report only: the same turn after state 0 (the reset), prey left in place
RESET_READ_K = 500            # report only: m_500, states 450..550
COMMAND_CHANNELS = (60, 61, 62)
OBS651_TOL = 1e-3             # the state-651 observation may differ from the twin's beyond this only in obs[60:63]
TWIN_HEADING_TOL_DEG = 1e-5   # command-blind walker: heading at state 1000 = the twin's + angle within this
YAW_REPRO_TOL_DEG = 1e-12     # psi[T] - psi[101] against the probe's yaw_change_deg
INFO_HEADING_TOL_RAD = 1e-9   # wrap(yaw(q_t) - psi_cmd) against info walk_heading_error_rad

# ---------------------------------------------------------------- the parents (v3 "Parent and entry")
PARENT_PAIRS = {
    42: ("sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
         "sha256:b9e6dee894e549ca4e844f7ef3c162e0f52609eacede7b1790e5d649e77d9cce"),
    52: ("sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
         "sha256:cf29635feb8c4040ea9a9bb08255b97de60673365dce107445c115aa2927f37a"),
    60: ("sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2",
         "sha256:56e58a536d08198fdafbde3310e05116ff65e7cd471f536016858f87f9c32dc9"),
}
PILOT4_ENV = {"command_mode": "heading", "prey_lateral_range": (0.0, 0.0), "walk_heading_price_weight": 0.2,
              "walk_heading_price_scale_rad": 0.5}
RUN_ID_RE = re.compile(r"^pilot4_walk_s(\d+)(_b)?$")
VALIDATION_RUN_RE = re.compile(r"^pilot(\d+)_walk_s(\d+)(_b)?$")
SKIP_KEYS = ("episode", "reset_seed")
INF = math.inf


# ================================================================ pure arithmetic (no tree needed)


def wrap_rad(a: float) -> float:
    """[-pi, pi), the tree's direction_commands.wrap_angle formula."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def wrap_deg(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def yaw_from_quat(q) -> float:
    """The probe's yaw (gait_probe._quat_to_euler, lines 243-248), (w, x, y, z) -> radians. The tree modes call the
    probe's own function; this copy serves the pure tests and is checked equal to it there."""
    w, x, y, z = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def unwrap_heading(raw_yaw_rad, psi_cmd_rad: float) -> np.ndarray:
    """psi (radians) over states 0..T: numpy.unwrap of the raw yaw, shifted by whole turns so that psi[0] lies within
    half a turn of psi_cmd (no shift, so no arithmetic, when it already does)."""
    psi = np.unwrap(np.asarray(raw_yaw_rad, dtype=np.float64))
    turns = round((float(psi[0]) - float(psi_cmd_rad)) / (2.0 * math.pi))
    if turns:
        psi = psi - turns * 2.0 * math.pi
    return psi


def one_s_mean(psi, k: int) -> float:
    """m_k: the mean of psi over states k-50..k+50."""
    if k - MEAN_HALF < 0 or k + MEAN_HALF >= len(psi):
        raise ValueError(f"m_{k} needs states {k - MEAN_HALF}..{k + MEAN_HALF}; the trace has 0..{len(psi) - 1}")
    return float(np.mean(np.asarray(psi, dtype=np.float64)[k - MEAN_HALF: k + MEAN_HALF + 1]))


def one_s_means(psi, T: int) -> tuple[np.ndarray, np.ndarray]:
    """(k, m_k) for k = 151..T-50."""
    ks = np.arange(K_FIRST, T - MEAN_HALF + 1)
    return ks, np.array([one_s_mean(psi, int(k)) for k in ks], dtype=np.float64)


def episode_metrics(psi_rad, psi_cmd_rad: float, T: int, root_xy=None, horizon: int = HORIZON) -> dict:
    """Per-episode heading metrics (degrees). D, W0, R and the last mean's error only for a full-length episode
    (T = horizon): a fall has no D. yaw_from_psi_deg = degrees(psi[T] - psi[101]) for T >= 110, the probe's own
    arithmetic for yaw_change_deg. With root_xy (states 0..T, metres), report only: the median 1-s crab (course of the
    pelvis displacement over states k-50..k+50 minus m_k) and the end cross-track (pelvis y at state T)."""
    psi_rad = np.asarray(psi_rad, dtype=np.float64)
    if psi_rad.shape != (T + 1,):
        raise ValueError(f"psi has shape {psi_rad.shape}, not ({T + 1},) for T = {T}")
    psi = np.degrees(psi_rad)
    cmd = math.degrees(float(psi_cmd_rad))
    out: dict = {"T": int(T), "full": int(T) == int(horizon), "psi_cmd_deg": cmd,
                 "yaw_from_psi_deg": (float(np.degrees(psi_rad[T] - psi_rad[WINDOW_FIRST_STATE]))
                                      if T >= YAW_MIN_LENGTH else None)}
    for key in ("D_deg", "D_k", "D_signed_deg", "W0_deg", "W0_state", "R_deg", "last_mean_error_deg", "last_mean_k",
                "crab_median_deg", "end_cross_track_m"):
        out[key] = None
    if not out["full"]:
        return out
    ks, m = one_s_means(psi, T)
    dev = m - cmd
    i = int(np.argmax(np.abs(dev)))
    e = psi[WINDOW_FIRST_STATE: T + 1] - cmd
    j = int(np.argmax(np.abs(e)))
    out.update(D_deg=float(abs(dev[i])), D_k=int(ks[i]), D_signed_deg=float(dev[i]), W0_deg=float(abs(e[j])),
               W0_state=WINDOW_FIRST_STATE + j, R_deg=float(m.max() - m.min()), last_mean_error_deg=float(dev[-1]),
               last_mean_k=int(ks[-1]))
    if root_xy is not None:
        xy = np.asarray(root_xy, dtype=np.float64)
        crabs = []
        for k, mk in zip(ks, m, strict=True):
            d = xy[int(k) + MEAN_HALF, :2] - xy[int(k) - MEAN_HALF, :2]
            crabs.append(wrap_deg(math.degrees(math.atan2(d[1], d[0])) - mk))
        out["crab_median_deg"] = float(np.median(crabs))
        out["end_cross_track_m"] = float(xy[T, 1])
    return out


def percentile(values, p: float) -> "float | None":
    """numpy.percentile's default (linear) on the values (inf allowed, sorting last); a nan numpy returns (both
    neighbours inf, or the upper one inf at an interpolation fraction of 0.5 or more, or at 0) reads inf. None for no
    values; a nan input is refused."""
    x = np.asarray(list(values), dtype=np.float64)
    if x.size == 0:
        return None
    if np.isnan(x).any():
        raise ValueError("a nan among the values")
    with np.errstate(invalid="ignore"):
        v = float(np.percentile(x, p))
    return INF if math.isnan(v) else v


def _le(value: "float | None", bar: float) -> bool:
    return value is not None and math.isfinite(value) and value <= bar


def part_a(yaws) -> dict:
    """(a) on the full-length episodes' signed yaw_change_deg."""
    yaws = [float(y) for y in yaws]
    if not yaws:
        return {"verdict": "NOT READ", "why": "no full-length panel episode", "n": 0}
    med = percentile(yaws, 50)
    p95 = percentile([abs(y) for y in yaws], PCT)
    ok = _le(abs(med), A_MEDIAN_ABS_MAX_DEG) and _le(p95, A_P95_ABS_MAX_DEG)
    return {"verdict": "PASS" if ok else "FAIL", "n": len(yaws), "median_signed_deg": med, "p95_abs_deg": p95,
            "min_deg": min(yaws), "max_deg": max(yaws),
            "bars": {"median_abs_max_deg": A_MEDIAN_ABS_MAX_DEG, "p95_abs_max_deg": A_P95_ABS_MAX_DEG}}


def part_b(D, W0, R=(), last=()) -> dict:
    """(b) on the full-length episodes' D and W0 (same episodes, same order); R and the last mean's error: report
    only."""
    D, W0 = [float(x) for x in D], [float(x) for x in W0]
    if len(D) != len(W0):
        raise ValueError("D and W0 must come from the same episodes")
    if not D:
        return {"verdict": "NOT READ", "why": "no full-length panel episode", "n": 0}
    pd, pw = percentile(D, PCT), percentile(W0, PCT)
    ok = _le(pd, B_D_P95_MAX_DEG) and _le(pw, B_W0_P95_MAX_DEG)
    gap = [w - d for d, w in zip(D, W0, strict=True)]
    out = {"verdict": "PASS" if ok else "FAIL", "n": len(D), "D_p95_deg": pd, "W0_p95_deg": pw,
           "D_median_deg": percentile(D, 50), "D_max_deg": max(D), "W0_median_deg": percentile(W0, 50),
           "W0_minus_D_min_deg": min(gap), "W0_minus_D_max_deg": max(gap),
           "bars": {"D_p95_max_deg": B_D_P95_MAX_DEG, "W0_p95_max_deg": B_W0_P95_MAX_DEG}}
    if R:
        out["report_R_median_deg"], out["report_R_p95_deg"] = percentile(R, 50), percentile(R, PCT)
    if last:
        out["report_last_mean_error_median_deg"] = percentile(last, 50)
    return out


def recovery_reading(ep: dict, twin: "dict | None", angle_deg: float, read_k: int = READ_K,
                     pass_rule: bool = True, movement: bool = True) -> dict:
    """One recovery episode. ep: T, turned (the turn was applied), psi_rad (states 0..T), psi_cmd_rad, root_xy
    (states 0..T), prey_x_at_turn (the prey's x at the turn state, before the turn). twin: the same reset's un-turned
    episode (T, psi_rad, psi_cmd_rad), for the share removed. With pass_rule the episode counts only when it reached the
    turn and its pelvis x at the turn state is above the prey's (v3 (c)); without (the reset variant) every episode
    counts. r = inf when it ends before the read window's last state, or (with movement) moved < 0.35 m."""
    T = int(ep["T"])
    last = read_k + MEAN_HALF
    s = NUDGE_STATE
    reached = bool(ep["turned"])
    xy = np.asarray(ep["root_xy"], dtype=np.float64)
    out = {"T": T, "angle_deg": float(angle_deg), "reached_nudge": reached, "passed_prey_at_650": None,
           "pelvis_minus_prey_x_650_m": None, "counting": False, "moved_650_1000_m": None, "m_read_error_deg": None,
           "r_deg": None, "r_reason": None, "share_removed": None, "twin_m_read_error_deg": None}
    if pass_rule:
        if reached:
            margin = float(xy[s, 0] - float(ep["prey_x_at_turn"]))
            out["pelvis_minus_prey_x_650_m"] = margin
            out["passed_prey_at_650"] = margin > 0.0
        out["counting"] = bool(reached and out["passed_prey_at_650"])
    else:
        out["counting"] = True
    if not out["counting"]:
        out["r_reason"] = "ended before the nudge" if not reached else "prey not yet passed at state 650"
        return out
    if pass_rule and T >= last:
        out["moved_650_1000_m"] = float(np.linalg.norm(xy[last, :2] - xy[s, :2]))
    if T < last:
        out["r_deg"], out["r_reason"] = INF, f"ended before state {last} (T = {T})"
    elif movement and pass_rule and out["moved_650_1000_m"] < C_MIN_NET_MOVE_M:
        out["r_deg"], out["r_reason"] = INF, f"pelvis moved {out['moved_650_1000_m']:.3f} m < {C_MIN_NET_MOVE_M} m"
    else:
        e = math.degrees(one_s_mean(np.asarray(ep["psi_rad"], dtype=np.float64), read_k) - float(ep["psi_cmd_rad"]))
        out["m_read_error_deg"] = e
        out["r_deg"] = abs(e)
    if twin is not None and int(twin["T"]) >= last:
        et = math.degrees(one_s_mean(np.asarray(twin["psi_rad"], dtype=np.float64), read_k)
                          - float(twin["psi_cmd_rad"]))
        out["twin_m_read_error_deg"] = et
        if out["m_read_error_deg"] is not None:
            out["share_removed"] = 1.0 - (out["m_read_error_deg"] - et) / float(angle_deg)
    return out


def part_c(readings: list, n_total: "int | None" = None) -> dict:
    """(c) over the recovery readings (both signs), with the counts v3 prints."""
    n_total = len(readings) if n_total is None else n_total
    counting = [x for x in readings if x["counting"]]
    ended = sum(1 for x in readings if not x["reached_nudge"])
    not_passed = sum(1 for x in readings if x["reached_nudge"] and not x["passed_prey_at_650"])
    out = {"n_episodes": n_total, "n_counting": len(counting), "left_out_ended_before_nudge": ended,
           "left_out_not_yet_passed": not_passed,
           "bars": {"median_max_deg": C_MEDIAN_MAX_DEG, "p95_max_deg": C_P95_MAX_DEG, "min_counting": C_MIN_COUNTING,
                    "min_net_move_m": C_MIN_NET_MOVE_M}}
    r = [x["r_deg"] for x in counting]
    out["fell_after_nudge"] = sum(1 for x in counting if x["r_reason"] and x["r_reason"].startswith("ended before"))
    out["moved_too_little"] = sum(1 for x in counting if x["r_reason"] and x["r_reason"].startswith("pelvis moved"))
    out["within_11_5"] = sum(1 for v in r if v <= C_P95_MAX_DEG)
    out["within_5"] = sum(1 for v in r if v <= C_MEDIAN_MAX_DEG)
    shares: dict = {}
    for sign, name in ((1, "plus"), (-1, "minus")):
        vals = [x["share_removed"] for x in counting if x["share_removed"] is not None and x["angle_deg"] * sign > 0]
        shares[name] = percentile(vals, 50)
    out["share_removed_median"] = shares
    if len(counting) < C_MIN_COUNTING:
        out.update(verdict="NOT READ", median_r_deg=None, p95_r_deg=None,
                   why=(f"prey not yet passed at 6.5 s in {not_passed} of {n_total}"
                        + (f"; {ended} ended before the nudge" if ended else "")
                        + f"; {len(counting)} counting, fewer than {C_MIN_COUNTING}"))
        return out
    med, p95 = percentile(r, 50), percentile(r, PCT)
    ok = _le(med, C_MEDIAN_MAX_DEG) and _le(p95, C_P95_MAX_DEG)
    out.update(verdict="PASS" if ok else "FAIL", median_r_deg=med, p95_r_deg=p95)
    return out


def variant_summary(readings: list) -> dict:
    """Report only: a variant's r and share (both signs pooled and per sign), with its counts."""
    counting = [x for x in readings if x["counting"]]
    r = [x["r_deg"] for x in counting]
    sh = [x["share_removed"] for x in counting if x["share_removed"] is not None]
    return {"n_episodes": len(readings), "n_counting": len(counting), "median_r_deg": percentile(r, 50),
            "p95_r_deg": percentile(r, PCT), "within_11_5": sum(1 for v in r if v <= C_P95_MAX_DEG),
            "infinite": sum(1 for v in r if not math.isfinite(v)), "share_removed_median_pooled": percentile(sh, 50),
            "share_removed_median_plus": percentile([x["share_removed"] for x in counting if x["share_removed"]
                                                     is not None and x["angle_deg"] > 0], 50),
            "share_removed_median_minus": percentile([x["share_removed"] for x in counting if x["share_removed"]
                                                      is not None and x["angle_deg"] < 0], 50)}


def straight_verdict(a: dict, b: dict, c: dict) -> str:
    return "PASS" if a.get("verdict") == b.get("verdict") == c.get("verdict") == "PASS" else "FAIL"


# ================================================================ the turn (v3 (c); no tree needed)


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a x b, (w, x, y, z)."""
    w1, x1, y1, z1 = (float(v) for v in a)
    w2, x2, y2, z2 = (float(v) for v in b)
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def quat_z(angle_rad: float) -> np.ndarray:
    return np.array([math.cos(angle_rad / 2.0), 0.0, 0.0, math.sin(angle_rad / 2.0)])


def turn_increment(x: float, y: float, angle_rad: float) -> tuple[float, float]:
    """R(angle) (x, y) - (x, y), written as an increment so that a 0-deg turn adds exactly zero (cos 0 - 1 = sin 0 = 0):
    the turned vector is v + increment, the rotation itself in exact arithmetic."""
    c1, s = math.cos(angle_rad) - 1.0, math.sin(angle_rad)
    return c1 * x - s * y, s * x + c1 * y


def apply_turn(data, angle_deg: float, turn_prey: bool) -> None:
    """The nudge on a MuJoCo MjData, in place: q <- q_z(angle) x q (qpos[3:7]); qvel[0:3] (world frame) turned about
    the vertical; qvel[3:6] (body frame) left; with turn_prey, data.mocap_pos[0] turned about the vertical through the
    root (qpos[0:2]). Nothing else: no mj_forward, no observation rebuild."""
    a = math.radians(float(angle_deg))
    data.qpos[3:7] = quat_mul(quat_z(a), np.array(data.qpos[3:7], dtype=np.float64))
    vx, vy = float(data.qvel[0]), float(data.qvel[1])
    dvx, dvy = turn_increment(vx, vy, a)
    data.qvel[0], data.qvel[1] = vx + dvx, vy + dvy
    if turn_prey:
        rx, ry = float(data.qpos[0]), float(data.qpos[1])
        px, py = float(data.mocap_pos[0][0]), float(data.mocap_pos[0][1])
        dx, dy = turn_increment(px - rx, py - ry, a)
        data.mocap_pos[0][0], data.mocap_pos[0][1] = px + dx, py + dy


# ================================================================ refusals (pure)


def sha256_file(path: "str | Path") -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def refuse_existing(paths) -> "str | None":
    """An output that already exists is refused (a record is never overwritten)."""
    hit = [str(p) for p in paths if p and Path(p).exists()]
    return f"{', '.join(hit)} exist(s) (never overwritten; move it or name another file)" if hit else None


def run_name_problem(run: "str | Path", seed: int, validation: bool) -> "str | None":
    name = Path(str(run)).name
    m = RUN_ID_RE.match(name)
    if m and int(m.group(1)) == seed:
        return None
    if validation:
        vm = VALIDATION_RUN_RE.match(name)
        if vm and int(vm.group(2)) == seed:
            return None
        return f"the run folder {name!r} is not pilot<N>_walk_s{seed}[_b] (even under --validation)"
    return f"the run folder {name!r} is not pilot4_walk_s{seed}[_b] (a cell from another run)"


def seed_problems(binding: dict, meta: dict, n_rows: int, *, seed: int, run: "str | Path", head: str, dirty: str,
                  validation: bool = False) -> list[str]:
    """Every reason the seed mode refuses a cell before rolling anything (each one listed); file sha256s are checked
    by the caller (seed_file_problems)."""
    problems = []
    if binding.get("seed") != seed:
        problems.append(f"the cell judged seed {binding.get('seed')}, not --seed {seed}")
    if binding.get("head") != head:
        problems.append(f"the cell was judged at {binding.get('head')}, this tree is {head}")
    if dirty:
        problems.append("this tree has local edits (git status --porcelain --untracked-files=no is not empty)")
    if binding.get("off_schedule"):
        problems.append(f"the cell is off-schedule ({binding.get('off_schedule_reason')}): never scored, never read")
    if binding.get("validation_only") and not validation:
        problems.append(f"the cell is validation-only (run budget {binding.get('expected_budget')}); --validation "
                        "reads it for a check, never for an outcome")
    brun = binding.get("run")
    if not brun or Path(str(brun)).resolve() != Path(str(run)).resolve():
        problems.append(f"the cell's run {brun!r} is not --run {str(run)!r} (a cell from another run)")
    why = run_name_problem(run, seed, validation)
    if why:
        problems.append(why)
    models = Path(str(run)).resolve() / "03_locomotion" / "models"
    for key in ("checkpoint", "vecnorm"):
        p = binding.get(key)
        if not p or Path(str(p)).resolve().parent != models:
            problems.append(f"the cell's {key} {p!r} is not in {models} (a cell from another run)")
    if meta.get("seed_scheme") != "panel" or meta.get("seed_base") != PANEL_SEED_BASE:
        problems.append(f"probe.json's panel is {meta.get('seed_scheme')} {meta.get('seed_base')}, not panel "
                        f"{PANEL_SEED_BASE}")
    if meta.get("episodes") != PANEL_EPISODES or n_rows != PANEL_EPISODES:
        problems.append(f"probe.json holds {n_rows} episodes (meta {meta.get('episodes')}), not {PANEL_EPISODES}")
    if meta.get("settle_steps_gait_window") != SETTLE:
        problems.append(f"probe.json's gait window starts at {meta.get('settle_steps_gait_window')}, not {SETTLE}")
    if meta.get("checkpoint_sha256") != binding.get("checkpoint_sha256"):
        problems.append("probe.json names another checkpoint than binding.json")
    drift = meta.get("env_kwargs_drift_vs_recorded")
    if drift:  # fix stage: the probe only warns when the run's recorded env kwargs differ from the tree's
        problems.append(f"probe.json's env-kwargs drift record is not empty: the run's recorded env kwargs differ from "
                        f"the env the cell was judged in ({json.dumps(drift, sort_keys=True, default=str)})")
    return problems


def seed_file_problems(binding: dict) -> list[str]:
    out = []
    for key in ("checkpoint", "vecnorm"):
        p, want = binding.get(key), binding.get(f"{key}_sha256")
        if not p or not Path(str(p)).is_file() or sha256_file(p) != want:
            out.append(f"{p} is missing or no longer the judged file ({want})")
    return out


def baseline_problems(seed: int, ckpt_sha: str, vec_sha: str, *, head: str, dirty: str,
                      expect_head: "str | None") -> list[str]:
    problems = []
    if seed not in PARENT_PAIRS:
        problems.append(f"--seed {seed} has no pilot-3 8M parent (seeds {sorted(PARENT_PAIRS)})")
    else:
        want_zip, want_pkl = PARENT_PAIRS[seed]
        if ckpt_sha != want_zip:
            problems.append(f"the checkpoint is {ckpt_sha}, not s{seed}'s 8M parent {want_zip}")
        if vec_sha != want_pkl:
            problems.append(f"the sidecar is {vec_sha}, not s{seed}'s 8M parent sidecar {want_pkl}")
    if dirty:
        problems.append("this tree has local edits (git status --porcelain --untracked-files=no is not empty)")
    if expect_head and head != expect_head:
        problems.append(f"this tree is at {head}, not --expect-head {expect_head}")
    return problems


def env_problems(env_facts: dict, validation: bool) -> list[str]:
    """The pilot-4 env (command_mode heading, prey [0, 0], price 0.2 / 0.5 rad), unless --validation."""
    if validation:
        return []
    bad = [f"{k} {env_facts.get(k)!r} (want {v!r})" for k, v in PILOT4_ENV.items() if env_facts.get(k) != v]
    return [f"not the pilot-4 env: {'; '.join(bad)}"] if bad else []


def compare_rows(rerolled: dict, recorded: dict) -> list[str]:
    """pilot3_video.compare_rows: the keys whose values differ (both JSON round-tripped), index keys skipped."""
    keys = (set(rerolled) | set(recorded)) - set(SKIP_KEYS)
    return sorted(k for k in keys if rerolled.get(k, "<absent>") != recorded.get(k, "<absent>"))


def jsonable(x):
    """Strict-JSON form: numpy scalars and arrays to Python; +-inf to the strings "inf" / "-inf"; nan refused."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        f = float(x)
        if math.isnan(f):
            raise ValueError("a nan in the record")
        return f if math.isfinite(f) else ("inf" if f > 0 else "-inf")
    return x


# ================================================================ the tree side


def install_no_training_guard() -> list[str]:
    """Block every training entry point before anything is loaded: SB3 learn()/train(), torch optimizer step(),
    backward. A tool run that reached one would raise before any update."""

    class TrainingBlocked(RuntimeError):
        pass

    def blocked(name):
        def _raise(*args, **kwargs):
            raise TrainingBlocked(f"pilot4_straight: training entry point blocked: {name}")
        return _raise

    import torch
    import torch.optim as optim

    patched = []
    for attr in dir(optim):
        cls = getattr(optim, attr)
        if isinstance(cls, type) and issubclass(cls, optim.Optimizer) and "step" in cls.__dict__:
            cls.step = blocked(f"torch.optim.{attr}.step")
            patched.append(f"torch.optim.{attr}.step")
    torch.Tensor.backward = blocked("torch.Tensor.backward")
    torch.autograd.backward = blocked("torch.autograd.backward")
    patched += ["torch.Tensor.backward", "torch.autograd.backward"]
    import stable_baselines3 as sb3
    from stable_baselines3.common import base_class, off_policy_algorithm, on_policy_algorithm

    classes = [base_class.BaseAlgorithm, on_policy_algorithm.OnPolicyAlgorithm, off_policy_algorithm.OffPolicyAlgorithm]
    classes += [getattr(sb3, n) for n in ("PPO", "SAC", "A2C", "TD3", "DDPG", "DQN") if hasattr(sb3, n)]
    for cls in classes:
        for method in ("learn", "train"):
            if method in cls.__dict__:
                setattr(cls, method, blocked(f"{cls.__name__}.{method}"))
                patched.append(f"{cls.__name__}.{method}")
    return patched


def install_probe_shims(embodiment: str = EMBODIMENT) -> None:
    """gait_probe_pilot.py's four shims (bare species string -> the model id), as pilot3_video.py installs them."""
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


class Context:
    """The env, loader and predict of one tool run (the probe's own, as pilot3_video.py builds them)."""


def build_context(tree: Path, ckpt: Path, vec: Path, *, prepare_parent: bool) -> Context:
    import gymnasium as gym

    install_probe_shims()
    probe, probe_path = load_probe(tree)
    from environments.shared.config import load_stage_config
    from environments.shared.curriculum.stance_gate import StanceGateThresholds
    from environments.shared.plant_contract import current_plant_identity, validate_environment_plant
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import SPECIES_FACTORIES

    ctx = Context()
    ctx.probe, ctx.probe_path = probe, probe_path
    stage_cfg = load_stage_config(SPECIES, STAGE)
    ctx.env_kwargs = dict(stage_cfg["env_kwargs"])
    ctx.gate_settle = int(StanceGateThresholds.from_curriculum(stage_cfg.get("curriculum_kwargs", {}),
                                                               require_criteria=False).settle_steps)
    env_class = SPECIES_FACTORIES[SPECIES]().env_class
    plant = current_plant_identity(SPECIES)

    def load():
        return load_sb3_checkpoint(str(ckpt), str(vec), lambda: env_class(**ctx.env_kwargs), guess_sidecar=False,
                                   allow_unnormalized=False, plant_identity=plant, allow_legacy_plant=False)

    model, normalizer, _ = load()
    base = env_class(**ctx.env_kwargs)
    validate_environment_plant(base, plant, artifact=f"{SPECIES} {STAGE} pilot4_straight environment")
    obs_dim = int(base.observation_space.shape[0])
    ctx.blind = None
    ctx.max_dA = [0.0]
    if prepare_parent:
        from environments.shared.command_frame import reseed_command_slice
        from environments.shared.policy_loading import assert_command_blind, neutralize_command_columns

        orig_model, orig_norm, _ = load()
        reseed_command_slice(normalizer.obs_rms)
        ctx.blind = assert_command_blind(model, normalizer,
                                         prepare=lambda: neutralize_command_columns(model, observation_dim=obs_dim))
        om = np.asarray(orig_norm.obs_rms.mean, dtype=np.float64)
        ov = np.asarray(orig_norm.obs_rms.var, dtype=np.float64)
        oe, oc = float(orig_norm.epsilon), float(orig_norm.clip_obs)

        def original(obs):
            x = np.clip((obs - om) / np.sqrt(ov + oe), -oc, oc).astype(np.float32)
            return np.asarray(orig_model.predict(x, deterministic=True)[0])
    obs_rms = normalizer.obs_rms
    mean = np.asarray(obs_rms.mean, dtype=np.float64)
    var = np.asarray(obs_rms.var, dtype=np.float64)
    eps, clip = float(normalizer.epsilon), float(normalizer.clip_obs)

    def predict(obs: np.ndarray) -> np.ndarray:  # the probe's own predict
        normalized = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
        action = np.asarray(model.predict(normalized, deterministic=True)[0])
        if prepare_parent:
            z = np.array(obs, dtype=np.float32).copy()
            z[COMMAND_CHANNELS[0]: COMMAND_CHANNELS[-1] + 1] = 0.0
            ctx.max_dA[0] = max(ctx.max_dA[0], float(np.max(np.abs(np.asarray(action, dtype=np.float64)
                                                                     - np.asarray(original(z), dtype=np.float64)))))
        return action

    class Recorder(gym.Wrapper):
        """Pass-through for the probe's rollout_episode. Records the observations, actions, the prey position, the
        command state's desired heading and info walk_heading_error_rad; when `turn` = (state, angle_deg, turn_prey)
        it applies the turn inside the step call that follows that state (the action already chosen), only if the
        previous step neither terminated nor truncated."""

        def __init__(self, env):
            super().__init__(env)
            self.turn = None
            self._clear()

        def _clear(self):
            self.obs, self.actions, self.prey, self.desired, self.info_err = [], [], [], [], []
            self.steps, self.done, self.turned_at, self.spawn_heading = 0, False, None, None

        def _note_state(self, obs):
            u = self.env.unwrapped
            self.obs.append(np.array(obs, copy=True))
            self.prey.append(np.array(u.data.mocap_pos[0], dtype=np.float64))
            st = getattr(u, "_command_state", None)
            self.desired.append(None if st is None else float(st.desired_heading))

        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            self._clear()
            u = self.env.unwrapped
            self.spawn_heading = float(u._heading())
            self._note_state(obs)
            return obs, info

        def step(self, action):
            u = self.env.unwrapped
            if self.turn is not None and self.steps == self.turn[0] and not self.done:
                apply_turn(u.data, self.turn[1], self.turn[2])
                self.turned_at = self.steps
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.steps += 1
            self.done = bool(terminated or truncated)
            self.actions.append(np.array(action, dtype=np.float64, copy=True))
            self._note_state(obs)
            v = info.get("walk_heading_error_rad")
            self.info_err.append(None if v is None else float(v))
            return obs, reward, terminated, truncated, info

    ctx.rec = Recorder(base)
    ctx.rd = probe.RobotDescription(base)
    ctx.dt = float(base.unwrapped.dt)
    ctx.settle = SETTLE
    ctx.predict = predict
    u = base.unwrapped
    ctx.env_facts = {"command_mode": getattr(u, "command_mode", None),
                     "prey_lateral_range": tuple(float(v) for v in getattr(u, "prey_lateral_range", ()) or ()),
                     "walk_heading_price_weight": getattr(u, "walk_heading_price_weight", None),
                     "walk_heading_price_scale_rad": getattr(u, "walk_heading_price_scale_rad", None),
                     "observation_dim": obs_dim}
    ctx.env_facts["pilot4_env"] = not env_problems(ctx.env_facts, False)
    return ctx


PREFIX_STATE_KEYS = ("quat", "root_pos", "foot_pos", "foot_h", "qj")
PREFIX_STEP_KEYS = ("reward", "floor_force_mean", "forces", "root_vel")


def roll(ctx: Context, reset_seed: int, turn=None) -> dict:
    """One episode through the probe's rollout_episode and analyze_episode; the arrays the tool reads are kept."""
    ctx.rec.turn = turn
    ctx.max_dA[0] = 0.0
    try:
        tr = ctx.probe.rollout_episode(ctx.rec, ctx.predict, ctx.rd, int(reset_seed))
    finally:
        ctx.rec.turn = None
    row = ctx.probe.analyze_episode(tr, ctx.rd, ctx.dt, ctx.settle, ctx.gate_settle)
    rec = ctx.rec
    T = int(tr["length"])
    ep = {"reset_seed": int(reset_seed), "T": T, "row": json.loads(json.dumps(ctx.probe._json_safe(row))),
          "obs": np.asarray(rec.obs), "actions": np.asarray(rec.actions), "prey": np.asarray(rec.prey),
          "desired": list(rec.desired), "info_err": list(rec.info_err), "spawn_heading": rec.spawn_heading,
          "turned_at": rec.turned_at, "turn": turn,
          "max_dA": ctx.max_dA[0] if ctx.blind is not None else None}
    for k in PREFIX_STATE_KEYS + PREFIX_STEP_KEYS:
        ep[k] = np.asarray(tr[k])
    ep["raw_yaw"] = np.array([ctx.probe._quat_to_euler(q)[2] for q in ep["quat"]], dtype=np.float64)
    return ep


def heading_of(ep: dict) -> dict:
    """psi_cmd (source and value), psi (radians) and the internal checks of one episode."""
    desired = ep["desired"]
    if all(d is None for d in desired):
        cmd, source = float(ep["spawn_heading"]), "spawn heading (env._heading() after the reset; no command state)"
        moving = False
    else:
        cmd, source = float(desired[0]), "command_state.desired_heading"
        moving = any(d is None or d != cmd for d in desired)
    psi = unwrap_heading(ep["raw_yaw"], cmd)
    info = [v for v in ep["info_err"] if v is not None]
    diff = None
    if info and len(info) == len(ep["info_err"]):
        diff = max(abs(wrap_rad(wrap_rad(ep["raw_yaw"][t + 1] - cmd) - v)) for t, v in enumerate(ep["info_err"]))
    return {"psi_cmd_rad": cmd, "source": source, "moving": moving, "psi_rad": psi, "info_max_abs_diff_rad": diff,
            "info_steps": len(info), "spawn_heading_rad": float(ep["spawn_heading"])}


def arrays_equal_prefix(a: dict, b: dict, state: int) -> list[str]:
    """The keys whose states 0..state (steps 1..state; actions 1..state+1) differ (numpy.array_equal, no tolerance)."""
    bad = []
    for k in PREFIX_STATE_KEYS + ("obs", "prey"):
        if not np.array_equal(np.asarray(a[k])[: state + 1], np.asarray(b[k])[: state + 1]):
            bad.append(k)
    for k in PREFIX_STEP_KEYS:
        if not np.array_equal(np.asarray(a[k])[:state], np.asarray(b[k])[:state]):
            bad.append(k)
    if not np.array_equal(np.asarray(a["actions"])[: state + 1], np.asarray(b["actions"])[: state + 1]):
        bad.append("actions")
    if [d for d in a["desired"][: state + 1]] != [d for d in b["desired"][: state + 1]]:
        bad.append("desired")
    return bad


def arrays_equal_full(a: dict, b: dict) -> list[str]:
    if a["T"] != b["T"]:
        return ["T"]
    bad = [k for k in PREFIX_STATE_KEYS + PREFIX_STEP_KEYS + ("obs", "prey", "actions")
           if not np.array_equal(np.asarray(a[k]), np.asarray(b[k]))]
    if a["desired"] != b["desired"] or a["info_err"] != b["info_err"]:
        bad.append("desired/info")
    if compare_rows(a["row"], b["row"]):
        bad.append("probe row")
    return bad


def make_plot(path: Path, title: str, panel: list, recovery: list, panel_note: "str | None" = None) -> None:
    """The straightness look's figure: top-down pelvis paths with the commanded line (the world x axis), and the
    heading error psi - psi_cmd against time, for every panel and recovery episode. panel_note (fix stage, report
    only): the panel's median crab and end cross-track, added to the path panel's title."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ink, ink2, grid, surface = "#0b0b0b", "#52514e", "#d9d8d4", "#fcfcfb"
    c_panel, c_plus, c_minus = "#2a78d6", "#eb6834", "#1baf7a"
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor=surface)
    for ax in axes.ravel():
        ax.set_facecolor(surface)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=9)
        ax.grid(True, color=grid, linewidth=0.6)
    ax = axes[0, 0]
    ax.axhline(0.0, color=ink2, linewidth=1.2, linestyle="--", label="commanded line (world x axis)")
    for i, ep in enumerate(panel):
        xy = ep["root_xy"]
        ax.plot(xy[:, 0], xy[:, 1], color=c_panel, linewidth=1.0, alpha=0.55, label="panel episode" if i == 0 else None)
        if not ep["full"]:
            ax.plot(xy[-1, 0], xy[-1, 1], marker="x", color=ink, markersize=8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"panel: {len(panel)} pelvis paths, top-down (x marks a fall)"
                 + (f"\n{panel_note}" if panel_note else ""), color=ink, fontsize=11, loc="left")
    ax.set_xlabel("world x (m)", color=ink2)
    ax.set_ylabel("world y (m)", color=ink2)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=ink2)
    ax = axes[0, 1]
    ax.axhline(0.0, color=ink2, linewidth=1.2, linestyle="--", label="commanded line (world x axis)")
    seen = set()
    s = NUDGE_STATE
    for ep in recovery:  # states 0..650 (equal for both signs) in neutral ink, the rest in the sign's colour
        xy = ep["root_xy"]
        col, lab = (c_plus, "nudged +15 deg") if ep["angle_deg"] > 0 else (c_minus, "nudged -15 deg")
        ax.plot(xy[: s + 1, 0], xy[: s + 1, 1], color=ink2, linewidth=0.8, alpha=0.35,
                label=None if "pre" in seen else "before the nudge")
        seen.add("pre")
        if len(xy) > s + 1:
            ax.plot(xy[s:, 0], xy[s:, 1], color=col, linewidth=1.0, alpha=0.6, label=None if lab in seen else lab)
            seen.add(lab)
            ax.plot(xy[s, 0], xy[s, 1], marker="o", color=ink2, markersize=3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"recovery panel: {len(recovery)} pelvis paths (dot: the nudge at 6.5 s)", color=ink, fontsize=11,
                 loc="left")
    ax.set_xlabel("world x (m)", color=ink2)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=ink2)
    ax = axes[1, 0]
    for y, style in ((0.0, "-"), (B_D_P95_MAX_DEG, ":"), (-B_D_P95_MAX_DEG, ":"), (B_W0_P95_MAX_DEG, "--"),
                     (-B_W0_P95_MAX_DEG, "--")):
        ax.axhline(y, color=ink2, linewidth=0.9, linestyle=style)
    for i, ep in enumerate(panel):
        t = np.arange(len(ep["err_deg"])) * 0.01
        ax.plot(t, ep["err_deg"], color=c_panel, linewidth=0.8, alpha=0.5, label="panel episode" if i == 0 else None)
    ax.axvline(WINDOW_FIRST_STATE * 0.01, color=ink2, linewidth=0.8)
    ax.set_title("panel: heading error psi - psi_cmd (dotted +-11.5, dashed +-18.5 deg)", color=ink, fontsize=11,
                 loc="left")
    ax.set_xlabel("time (s)", color=ink2)
    ax.set_ylabel("heading error (deg)", color=ink2)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=ink2)
    ax = axes[1, 1]
    for y, style in ((0.0, "-"), (C_MEDIAN_MAX_DEG, ":"), (-C_MEDIAN_MAX_DEG, ":")):
        ax.axhline(y, color=ink2, linewidth=0.9, linestyle=style)
    ax.axvspan((READ_K - MEAN_HALF) * 0.01, (READ_K + MEAN_HALF) * 0.01, color=grid, alpha=0.6, linewidth=0)
    ax.axvline(NUDGE_STATE * 0.01, color=ink2, linewidth=0.8)
    seen = set()
    for ep in recovery:
        col, lab = (c_plus, "nudged +15 deg") if ep["angle_deg"] > 0 else (c_minus, "nudged -15 deg")
        t = np.arange(len(ep["err_deg"])) * 0.01
        ax.plot(t[: s + 1], ep["err_deg"][: s + 1], color=ink2, linewidth=0.7, alpha=0.3,
                label=None if "pre" in seen else "before the nudge")
        seen.add("pre")
        if len(t) > s + 1:
            ax.plot(t[s:], ep["err_deg"][s:], color=col, linewidth=0.8, alpha=0.55,
                    label=None if lab in seen else lab)
            seen.add(lab)
    ax.set_title("recovery: heading error (nudge at 6.5 s; shaded: read window 9-10 s; dotted +-5 deg)", color=ink,
                 fontsize=11, loc="left")
    ax.set_xlabel("time (s)", color=ink2)
    ax.legend(loc="lower left", fontsize=8, frameon=False, labelcolor=ink2)
    fig.suptitle(title, color=ink, fontsize=13, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=surface)
    plt.close(fig)


def evaluate(ctx: Context, *, recorded_rows: "list | None", limit: "int | None", zero_nudge: bool,
             command_blind: bool) -> tuple[dict, int, list]:
    """Roll the panel and the recovery panel, read the parts. Returns (record body, exit status, plot data).
    Raises _Refuse on a prefix refusal (main: exit 2) and _Internal on an internal check failure (main: exit 3);
    nothing is written then."""
    n_panel = PANEL_EPISODES if limit is None else min(limit, PANEL_EPISODES)
    rec_seeds = RECOVERY_SEEDS if limit is None else RECOVERY_SEEDS[:limit]
    status = 0
    internal: list[str] = []
    panel, panel_out = [], []
    for i in range(n_panel):
        seed = PANEL_SEED_BASE + i
        ep = roll(ctx, seed)
        h = heading_of(ep)
        ep["h"] = h
        diff = compare_rows(ep["row"], recorded_rows[i]) if recorded_rows is not None else []
        yaw_probe = (recorded_rows[i] if recorded_rows is not None else ep["row"]).get("yaw_change_deg")
        mt = episode_metrics(h["psi_rad"], h["psi_cmd_rad"], ep["T"], ep["root_pos"][:, :2])
        yaw_diff = None
        if mt["yaw_from_psi_deg"] is not None:
            if yaw_probe is None:
                internal.append(f"panel ep {i}: probe.json has no yaw_change_deg although T = {ep['T']}")
            else:
                yaw_diff = abs(mt["yaw_from_psi_deg"] - float(yaw_probe))
                if yaw_diff > YAW_REPRO_TOL_DEG:
                    internal.append(f"panel ep {i}: psi's yaw {mt['yaw_from_psi_deg']!r} differs from yaw_change_deg "
                                    f"{yaw_probe!r} by {yaw_diff:.3e} deg")
        if h["moving"]:
            internal.append(f"panel ep {i}: the commanded heading moved during the episode")
        if h["info_max_abs_diff_rad"] is not None and h["info_max_abs_diff_rad"] > INFO_HEADING_TOL_RAD:
            internal.append(f"panel ep {i}: info walk_heading_error_rad differs from wrap(yaw - psi_cmd) by "
                            f"{h['info_max_abs_diff_rad']:.3e} rad")
        disp = ep["row"].get("window_displacement_m")
        ent = {"episode": i, "reset_seed": seed, "T": ep["T"], "full": mt["full"], "matches_probe": not diff,
               "differing_keys": diff[:20], "used": not diff, "yaw_change_deg": yaw_probe,
               "yaw_from_psi_deg": mt["yaw_from_psi_deg"], "yaw_abs_diff_deg": yaw_diff,
               "psi_cmd_deg": mt["psi_cmd_deg"], "psi_cmd_source": h["source"],
               "info_heading_max_abs_diff_rad": h["info_max_abs_diff_rad"],
               "spawn_heading_deg": math.degrees(h["spawn_heading_rad"]),
               **{k: mt[k] for k in ("D_deg", "D_k", "D_signed_deg", "W0_deg", "W0_state", "R_deg",
                                     "last_mean_error_deg", "crab_median_deg", "end_cross_track_m")},
               "window_displacement_m": disp,
               "yaw_per_10m_deg": (float(yaw_probe) / float(disp) * 10.0
                                   if yaw_probe is not None and disp not in (None, 0, 0.0) else None),
               "floor_gait_class": ep["row"].get("floor_gait_class"),
               "max_action_delta_vs_blind": ep["max_dA"] if command_blind else None}
        if diff:
            status = 1
        panel_out.append(ent)
        panel.append(ep)
        print(f"  panel ep {i:2d} (reset {seed}): T={ep['T']} yaw={yaw_probe} D={mt['D_deg']} W0={mt['W0_deg']} "
              + ("matches probe.json" if recorded_rows is not None and not diff else
                 (f"DIFFERS in {len(diff)} keys {diff[:6]}" if diff else "")), flush=True)
        if i >= len(RECOVERY_SEEDS) or seed not in rec_seeds:
            for k in ("obs", "foot_pos", "foot_h", "qj", "forces", "floor_force_mean", "root_vel", "reward",
                      "actions", "prey"):
                ep.pop(k, None)
    if internal:
        raise _Internal(internal)

    used_full = [e for e in panel_out if e["used"] and e["full"]]
    parts_a = part_a([e["yaw_change_deg"] for e in used_full])
    parts_b = part_b([e["D_deg"] for e in used_full], [e["W0_deg"] for e in used_full],
                     [e["R_deg"] for e in used_full], [e["last_mean_error_deg"] for e in used_full])
    # fix stage, report only (v3 "Course against heading"; the look is the only guard against a crab): the medians
    # over the same full-length episodes of the per-episode 1-s crab and the end cross-track
    parts_b["report_crab_median_deg"] = percentile([e["crab_median_deg"] for e in used_full], 50)
    parts_b["report_end_cross_track_median_m"] = percentile([e["end_cross_track_m"] for e in used_full], 50)
    parts_a["n_full_of_panel"] = f"{sum(e['full'] for e in panel_out)} of {len(panel_out)}"
    parts_a["note_36"] = (None if sum(e["full"] for e in panel_out) >= FULL_MIN_ON_WALKS or limit is not None
                          else f"fewer than {FULL_MIN_ON_WALKS} full-length episodes: the seed cannot be WALKS")

    by_seed = {ep["reset_seed"]: ep for ep in panel}
    rec_out, readings, pip, rst = [], [], [], []
    checks = {"prefix": {"episodes": 0, "equal": 0, "refused": []},
              "obs651": {"episodes": 0, "only_command_channels": 0, "channels_seen": [], "bad": []},
              "turned_twin": {"applies": command_blind, "episodes": 0, "max_abs_dev_deg": None, "bad": [],
                              "max_action_diff_after_651": None},
              "zero_nudge": {"run": zero_nudge, "episodes": 0, "equal": 0, "bad": []}}
    channels_seen: set = set()
    twin_dev, act_diff, zero_dA = [], [], []
    plot_rec = []
    for seed in rec_seeds:
        twin = by_seed[seed]
        twin_h = twin["h"]
        twin_view = {"T": twin["T"], "psi_rad": twin_h["psi_rad"], "psi_cmd_rad": twin_h["psi_cmd_rad"]}
        for variant, state, prey_turned, read_k, angles in (
                ("c", NUDGE_STATE, True, READ_K, NUDGE_ANGLES_DEG),
                ("prey_in_place", NUDGE_STATE, False, READ_K, NUDGE_ANGLES_DEG),
                ("turned_at_reset", RESET_TURN_STATE, False, RESET_READ_K, NUDGE_ANGLES_DEG),
                ("zero", NUDGE_STATE, True, READ_K, (0.0,) if zero_nudge else ())):
            for angle in angles:
                ep = roll(ctx, seed, (state, angle, prey_turned))
                h = heading_of(ep)
                if h["moving"]:
                    raise _Internal([f"recovery reset {seed} {variant} {angle:+g}: the commanded heading moved"])
                if h["info_max_abs_diff_rad"] is not None and h["info_max_abs_diff_rad"] > INFO_HEADING_TOL_RAD:
                    raise _Internal([f"recovery reset {seed} {variant} {angle:+g}: info heading error differs by "
                                     f"{h['info_max_abs_diff_rad']:.3e} rad"])
                if variant == "zero":
                    if command_blind:
                        zero_dA.append(ep["max_dA"])
                    bad = arrays_equal_full(ep, twin)
                    checks["zero_nudge"]["episodes"] += 1
                    checks["zero_nudge"]["equal"] += int(not bad)
                    if bad:
                        checks["zero_nudge"]["bad"].append({"reset_seed": seed, "keys": bad})
                        status = 1
                    print(f"  zero-deg nudge reset {seed}: {'reproduces the panel episode' if not bad else bad}",
                          flush=True)
                    continue
                turned = ep["turned_at"] is not None
                if variant != "turned_at_reset" and turned:
                    bad = arrays_equal_prefix(ep, twin, NUDGE_STATE)
                    checks["prefix"]["episodes"] += 1
                    checks["prefix"]["equal"] += int(not bad)
                    if bad:
                        checks["prefix"]["refused"].append({"reset_seed": seed, "variant": variant,
                                                            "angle_deg": angle, "keys": bad})
                ch651 = None
                if variant != "turned_at_reset" and turned and ep["T"] > NUDGE_STATE and twin["T"] > NUDGE_STATE:
                    d651 = np.abs(np.asarray(ep["obs"][NUDGE_STATE + 1], dtype=np.float64)
                                  - np.asarray(twin["obs"][NUDGE_STATE + 1], dtype=np.float64))
                    ch651 = [int(c) for c in np.flatnonzero(d651 > OBS651_TOL)]
                if variant == "c" and ch651 is not None:  # the prey-in-place variant shows the prey block too
                    ch = ch651
                    channels_seen.update(ch)
                    checks["obs651"]["episodes"] += 1
                    ok = set(ch) <= set(COMMAND_CHANNELS)
                    checks["obs651"]["only_command_channels"] += int(ok)
                    if not ok:
                        checks["obs651"]["bad"].append({"reset_seed": seed, "variant": variant, "angle_deg": angle,
                                                        "channels": ch})
                        status = 1
                minus_twin = None
                if variant != "turned_at_reset" and ep["T"] == HORIZON and twin["T"] == HORIZON:
                    minus_twin = float(np.degrees(h["psi_rad"])[HORIZON] - np.degrees(twin_h["psi_rad"])[HORIZON])
                if command_blind and variant == "c" and minus_twin is not None:
                    dev = minus_twin - angle
                    twin_dev.append(abs(dev))
                    act_diff.append(float(np.max(np.abs(ep["actions"][NUDGE_STATE + 1:]
                                                        - twin["actions"][NUDGE_STATE + 1:]))))
                    checks["turned_twin"]["episodes"] += 1
                    if abs(dev) > TWIN_HEADING_TOL_DEG:
                        checks["turned_twin"]["bad"].append({"reset_seed": seed, "angle_deg": angle, "dev_deg": dev})
                        status = 1
                view = {"T": ep["T"], "turned": turned, "psi_rad": h["psi_rad"], "psi_cmd_rad": h["psi_cmd_rad"],
                        "root_xy": ep["root_pos"][:, :2],
                        "prey_x_at_turn": float(ep["prey"][state][0]) if ep["T"] >= state else None}
                rd = recovery_reading(view, twin_view, angle, read_k=read_k, pass_rule=(variant != "turned_at_reset"),
                                      movement=(variant != "turned_at_reset"))
                rd.update(reset_seed=seed, variant=variant, psi_cmd_deg=math.degrees(h["psi_cmd_rad"]),
                          turned_at=ep["turned_at"], obs651_channels_beyond_tol_vs_twin=ch651,
                          heading_minus_twin_at_1000_deg=minus_twin,
                          obs60_63_at_651=(ep["obs"][NUDGE_STATE + 1][60:63].tolist()
                                           if variant != "turned_at_reset" and ep["T"] > NUDGE_STATE else None))
                if command_blind:
                    rd["max_action_delta_vs_blind"] = ep["max_dA"]
                {"c": readings, "prey_in_place": pip, "turned_at_reset": rst}[variant].append(rd)
                if variant == "c":
                    rec_out.append(rd)
                    plot_rec.append({"angle_deg": angle, "root_xy": ep["root_pos"][:, :2],
                                     "err_deg": np.degrees(h["psi_rad"] - h["psi_cmd_rad"]), "reset_seed": seed})
                print(f"  {variant} reset {seed} {angle:+g}: T={ep['T']} counting={rd['counting']} r={rd['r_deg']} "
                      f"share={rd['share_removed']}", flush=True)
    if checks["prefix"]["refused"]:
        raise _Refuse([f"states 0..{NUDGE_STATE} of a nudged episode differ from its panel episode's: "
                       f"{checks['prefix']['refused'][:4]}"])
    checks["obs651"]["channels_seen"] = sorted(channels_seen)
    if twin_dev:
        checks["turned_twin"]["max_abs_dev_deg"] = max(twin_dev)
        checks["turned_twin"]["max_action_diff_after_651"] = max(act_diff)
    parts_c = part_c(readings)
    if limit is not None:
        for p in (parts_a, parts_b, parts_c):
            p["verdict_if_complete"] = p.get("verdict")
            p["verdict"] = "NOT READ"
            p["why_partial"] = f"partial run (--limit-resets {limit}): no verdict"
    body = {"panel": {"episodes": panel_out, "all_rolled_match": all(e["matches_probe"] for e in panel_out),
                      "mismatched": [e["episode"] for e in panel_out if not e["matches_probe"]],
                      "n_full": sum(e["full"] for e in panel_out), "n_used_full": len(used_full),
                      "yaw_repro_max_abs_diff_deg": max([e["yaw_abs_diff_deg"] for e in panel_out
                                                         if e["yaw_abs_diff_deg"] is not None], default=None)},
            "parts": {"a": parts_a, "b": parts_b, "c": parts_c,
                      "straight": "NOT READ" if limit is not None else straight_verdict(parts_a, parts_b, parts_c)},
            "recovery": {"episodes": rec_out, "prey_in_place": {"summary": variant_summary(pip), "episodes": pip},
                         "turned_at_reset": {"summary": variant_summary(rst), "episodes": rst, "read_k": RESET_READ_K,
                                             "mechanics": "the (c) turn applied after state 0 (the action of step 1 "
                                                          "chosen), prey left in place, every episode counts"}},
            "checks": checks, "_rows": [ep["row"] for ep in panel],
            "_variant_dA": [x["max_action_delta_vs_blind"] for x in pip + rst if "max_action_delta_vs_blind" in x]
            + zero_dA}
    psi_cmds = sorted({e["psi_cmd_deg"] for e in panel_out})
    body["psi_cmd"] = {"source": sorted({e["psi_cmd_source"] for e in panel_out}), "values_deg": psi_cmds,
                       "info_check_max_abs_diff_rad": max([e["info_heading_max_abs_diff_rad"] for e in panel_out
                                                           if e["info_heading_max_abs_diff_rad"] is not None],
                                                          default=None),
                       "info_check_available": all(e["info_heading_max_abs_diff_rad"] is not None
                                                   for e in panel_out)}
    plot_panel = [{"root_xy": ep["root_pos"][:, :2], "full": ep["T"] == HORIZON,
                   "err_deg": np.degrees(ep["h"]["psi_rad"] - ep["h"]["psi_cmd_rad"]), "reset_seed": ep["reset_seed"]}
                  for ep in panel]
    return body, status, [plot_panel, plot_rec]


class _Refuse(Exception):
    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


class _Internal(Exception):
    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def constants() -> dict:
    return {k: globals()[k] for k in (
        "HORIZON", "PANEL_EPISODES", "PANEL_SEED_BASE", "SETTLE", "WINDOW_FIRST_STATE", "MEAN_HALF", "K_FIRST",
        "A_MEDIAN_ABS_MAX_DEG", "A_P95_ABS_MAX_DEG", "B_D_P95_MAX_DEG", "B_W0_P95_MAX_DEG", "C_MEDIAN_MAX_DEG",
        "C_P95_MAX_DEG", "C_MIN_COUNTING", "C_MIN_NET_MOVE_M", "PCT", "RECOVERY_SEEDS", "NUDGE_ANGLES_DEG",
        "NUDGE_STATE", "READ_K", "RESET_TURN_STATE", "RESET_READ_K", "OBS651_TOL", "TWIN_HEADING_TOL_DEG",
        "YAW_REPRO_TOL_DEG", "INFO_HEADING_TOL_RAD")}


def crab_note(b: dict) -> "str | None":
    """The path panel's report-only line: the median 1-s crab (course minus heading) and end cross-track over the
    full-length episodes (None when (b) read none)."""
    crab, cross = b.get("report_crab_median_deg"), b.get("report_end_cross_track_median_m")
    if crab is None or cross is None:
        return None
    return f"report only: median crab {crab:+.1f} deg (course minus heading), median end cross-track {cross:+.2f} m"


def write_outputs(a, record: dict, plots: list, title: str, episodes_for_traces: "list | None") -> None:
    files = {}
    if a.plot:
        mpl_dir = tempfile.mkdtemp(prefix="mpl_")
        os.environ.setdefault("MPLCONFIGDIR", mpl_dir)
        try:
            make_plot(Path(a.plot), title, plots[0], plots[1], panel_note=crab_note(record["parts"]["b"]))
        finally:
            shutil.rmtree(mpl_dir, ignore_errors=True)
        files["plot"] = {"path": str(a.plot), "sha256": sha256_file(a.plot)}
    if a.traces:
        arrays = {}
        for kind, eps in (("panel", plots[0]), ("recovery", plots[1])):
            for j, ep in enumerate(eps):
                tag = f"{kind}_{j:02d}_reset{ep['reset_seed']}"
                if "angle_deg" in ep:
                    tag += f"_{ep['angle_deg']:+g}"
                arrays[f"{tag}_err_deg"] = ep["err_deg"]
                arrays[f"{tag}_root_xy"] = ep["root_xy"]
        with open(a.traces, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        files["traces"] = {"path": str(a.traces), "sha256": sha256_file(a.traces)}
    record["files"] = files
    Path(a.json).write_text(json.dumps(jsonable(record), indent=1, allow_nan=False))


def summary_lines(record: dict) -> list[str]:
    p = record["parts"]
    a, b, c = p["a"], p["b"], p["c"]

    def f(x, nd=1):
        return "inf" if x == INF else ("-" if x is None else f"{x:.{nd}f}")

    lines = [f"(a) end heading: {a['verdict']} | n {a.get('n')} | median {f(a.get('median_signed_deg'))} "
             f"p95|yaw| {f(a.get('p95_abs_deg'))} (bars |median| <= {A_MEDIAN_ABS_MAX_DEG}, p95 <= "
             f"{A_P95_ABS_MAX_DEG})",
             f"(b) weave: {b['verdict']} | n {b.get('n')} | D p95 {f(b.get('D_p95_deg'))} W0 p95 "
             f"{f(b.get('W0_p95_deg'))}"
             f" (bars {B_D_P95_MAX_DEG}, {B_W0_P95_MAX_DEG}) | report: D median {f(b.get('D_median_deg'))}, R median "
             f"{f(b.get('report_R_median_deg'))} p95 {f(b.get('report_R_p95_deg'))}, W0-D "
             f"{f(b.get('W0_minus_D_min_deg'))}..{f(b.get('W0_minus_D_max_deg'))}, crab median "
             f"{f(b.get('report_crab_median_deg'))} deg, end cross-track median "
             f"{f(b.get('report_end_cross_track_median_m'), 2)} m",
             f"(c) recovery: {c['verdict']} | counting {c['n_counting']} of {c['n_episodes']} (ended before the nudge "
             f"{c['left_out_ended_before_nudge']}, not yet passed {c['left_out_not_yet_passed']}) | median r "
             f"{f(c.get('median_r_deg'))} p95 r {f(c.get('p95_r_deg'))} | within 11.5: {c['within_11_5']}, within 5: "
             f"{c['within_5']} | fell after {c['fell_after_nudge']}, moved < 0.35 m {c['moved_too_little']} | share "
             f"removed median +15 {f(c['share_removed_median']['plus'], 2)} -15 "
             f"{f(c['share_removed_median']['minus'], 2)}" + (f" | {c['why']}" if c.get("why") else ""),
             f"STRAIGHT: {p['straight']}"]
    for name in ("prey_in_place", "turned_at_reset"):
        s = record["recovery"][name]["summary"]
        lines.append(f"  report only, {name}: counting {s['n_counting']} of {s['n_episodes']} | median r "
                     f"{f(s['median_r_deg'])} p95 {f(s['p95_r_deg'])} | within 11.5 {s['within_11_5']} | share "
                     f"pooled {f(s['share_removed_median_pooled'], 2)} (+ {f(s['share_removed_median_plus'], 2)}, - "
                     f"{f(s['share_removed_median_minus'], 2)})")
    ch = record["checks"]
    lines.append(f"  checks: prefix {ch['prefix']['equal']}/{ch['prefix']['episodes']} equal; state-651 observation "
                 f"beyond {OBS651_TOL} only in obs[60:63] {ch['obs651']['only_command_channels']}/"
                 f"{ch['obs651']['episodes']} (channels seen {ch['obs651']['channels_seen']}); turned twin "
                 f"{ch['turned_twin']['episodes'] - len(ch['turned_twin']['bad'])}/{ch['turned_twin']['episodes']} "
                 f"(max dev {ch['turned_twin']['max_abs_dev_deg']}); zero nudge {ch['zero_nudge']['equal']}/"
                 f"{ch['zero_nudge']['episodes']}")
    return lines


def tree_identity() -> tuple[Path, str, str, str]:
    import environments

    tree = Path.cwd().resolve()
    env_file = str(Path(environments.__file__).resolve())
    if not env_file.startswith(str(tree) + os.sep):
        raise _Refuse([f"environments imported from {env_file}, not from the tree {tree} (run through iso.py)"])
    return tree, env_file, git(tree, "rev-parse", "HEAD"), git(tree, "status", "--porcelain", "--untracked-files=no")


def run_seed(a) -> int:
    cell = Path(a.cell).resolve()
    binding = json.loads((cell / "binding.json").read_text())
    doc = json.loads((cell / "probe.json").read_text())
    meta, rows = doc.get("meta", {}), doc.get("episodes", [])
    tree, env_file, head, dirty = tree_identity()
    problems = seed_problems(binding, meta, len(rows), seed=a.seed, run=a.run, head=head, dirty=dirty,
                             validation=a.validation)
    problems += seed_file_problems(binding)
    if problems:
        raise _Refuse(problems)
    patched = install_no_training_guard()
    ctx = build_context(tree, Path(binding["checkpoint"]), Path(binding["vecnorm"]), prepare_parent=False)
    same_kwargs = json.loads(json.dumps(ctx.probe._json_safe(ctx.env_kwargs))) == meta.get("env_kwargs")
    if not same_kwargs or ctx.gate_settle != meta.get("settle_steps_gate"):
        problems.append("this tree's locomotion env kwargs or gate settle differ from probe.json's")
    problems += env_problems(ctx.env_facts, a.validation)
    if problems:
        raise _Refuse(problems)
    print(f"pilot4_straight seed s{a.seed} {binding.get('steps')}: tree {tree} at {head}; environments {env_file}; "
          f"blocked {len(patched)} training entry points; env {ctx.env_facts}", flush=True)
    body, status, plots = evaluate(ctx, recorded_rows=rows, limit=None, zero_nudge=False, command_blind=False)
    body.pop("_rows", None)
    body.pop("_variant_dA", None)
    import torch

    record = {"schema": SCHEMA, "mode": "seed", "validation": bool(a.validation),
              "validation_only_cell": bool(binding.get("validation_only")), "partial": False,
              "tool_sha256": sha256_file(__file__), "probe": str(ctx.probe_path),
              "probe_sha256": sha256_file(ctx.probe_path), "tree": str(tree), "head": head,
              "environments_file": env_file, "torch_threads": int(torch.get_num_threads()),
              "training_entry_points_blocked": len(patched), "seed": a.seed, "steps": binding.get("steps"),
              "run": str(Path(a.run).resolve()), "run_name": Path(a.run).name, "cell": str(cell),
              "binding_schema": binding.get("schema"), "binding_sha256": sha256_file(cell / "binding.json"),
              "probe_json_sha256": sha256_file(cell / "probe.json"),
              "final_fallback": bool(binding.get("final_fallback")),
              "checkpoint": binding["checkpoint"], "checkpoint_sha256": binding["checkpoint_sha256"],
              "vecnorm": binding["vecnorm"], "vecnorm_sha256": binding["vecnorm_sha256"], "constants": constants(),
              "env": ctx.env_facts, **body}
    title = (f"s{a.seed} at {binding.get('steps')}: STRAIGHT {body['parts']['straight']} ((a) "
             f"{body['parts']['a']['verdict']}, (b) {body['parts']['b']['verdict']}, (c) "
             f"{body['parts']['c']['verdict']})"
             + ("  [validation]" if a.validation else ""))
    write_outputs(a, record, plots, title, None)
    print("\n".join(summary_lines(record)))
    print(f"wrote {a.json}", flush=True)
    return status


def run_baseline(a) -> int:
    tree, env_file, head, dirty = tree_identity()
    ckpt, vec = Path(a.checkpoint).resolve(), Path(a.vecnorm).resolve()
    if not ckpt.is_file() or not vec.is_file():
        raise _Refuse([f"{ckpt} or {vec} is not a file"])
    ckpt_sha, vec_sha = sha256_file(ckpt), sha256_file(vec)
    problems = baseline_problems(a.seed, ckpt_sha, vec_sha, head=head, dirty=dirty, expect_head=a.expect_head)
    if problems:
        raise _Refuse(problems)
    patched = install_no_training_guard()
    ctx = build_context(tree, ckpt, vec, prepare_parent=True)
    problems += env_problems(ctx.env_facts, False)
    if problems:
        raise _Refuse(problems)
    print(f"pilot4_straight baseline s{a.seed}: tree {tree} at {head}; environments {env_file}; blocked {len(patched)} "
          f"training entry points; env {ctx.env_facts}; assert_command_blind {ctx.blind}", flush=True)
    body, status, plots = evaluate(ctx, recorded_rows=None, limit=a.limit_resets, zero_nudge=a.zero_nudge_check,
                                   command_blind=True)
    dA = max([e["max_action_delta_vs_blind"] for e in body["panel"]["episodes"]]
             + [e["max_action_delta_vs_blind"] for e in body["recovery"]["episodes"]]
             + list(body.pop("_variant_dA", [])), default=None)
    body["checks"]["action_delta_vs_blind"] = {"max": dA, "rule": "the prepared walker's action minus the unprepared "
                                               "parent's on the command-zeroed observation, every step"}
    if dA is not None and dA != 0.0:
        status = 1
    labels = None
    if a.labels_from:
        spec = importlib.util.spec_from_file_location("straight_labeler", a.labels_from)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)
        # the scorer reads probe.json's rows: the re-rolled rows, JSON round-tripped, in the panel order
        rows = [e_row for e_row in body.pop("_rows", [])]
        labs = [mod.label_episode(r) for r in rows]
        counts = Counter(labs)
        labels = {"scorer": str(Path(a.labels_from).resolve()), "scorer_sha256": sha256_file(a.labels_from),
                  "labels": labs, "counts": {k: counts.get(k, 0) for k in ("WALK", "HOP", "STILL", "FALL", "OTHER")},
                  "seed_outcome": mod.seed_outcome(counts) if a.limit_resets is None else None}
        for e, lab in zip(body["panel"]["episodes"], labs, strict=True):
            e["label"] = lab
    body.pop("_rows", None)
    blind = {k: v for k, v in (ctx.blind or {}).items() if k != "prepared"}
    blind["prepared"] = list((ctx.blind or {}).get("prepared") or [])
    import torch

    record = {"schema": SCHEMA, "mode": "baseline", "validation": False, "partial": a.limit_resets is not None,
              "tool_sha256": sha256_file(__file__), "probe": str(ctx.probe_path),
              "probe_sha256": sha256_file(ctx.probe_path), "tree": str(tree), "head": head,
              "environments_file": env_file, "torch_threads": int(torch.get_num_threads()),
              "training_entry_points_blocked": len(patched), "seed": a.seed, "steps": 8_000_000,
              "checkpoint": str(ckpt), "checkpoint_sha256": ckpt_sha, "vecnorm": str(vec), "vecnorm_sha256": vec_sha,
              "preparation": {"reseed_command_slice": True, "assert_command_blind": blind},
              "constants": constants(), "env": ctx.env_facts, "labels": labels, **body}
    title = (f"entry baseline s{a.seed} (8M parent, edge-prepared, pilot-4 env): STRAIGHT {body['parts']['straight']}"
             f" ((a) {body['parts']['a']['verdict']}, (b) {body['parts']['b']['verdict']}, (c) "
             f"{body['parts']['c']['verdict']})" + ("  [partial]" if a.limit_resets is not None else ""))
    write_outputs(a, record, plots, title, None)
    print("\n".join(summary_lines(record)))
    if labels:
        print(f"labels by {labels['scorer']}: {labels['counts']} -> {labels['seed_outcome']}")
    print(f"wrote {a.json}", flush=True)
    return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    ps = sub.add_parser("seed", help="a judged cell (binding.json, probe.json)")
    ps.add_argument("--seed", type=int, required=True)
    ps.add_argument("--run", required=True, help="the run copy the cell's pair must come from")
    ps.add_argument("--cell", required=True, help="the judge's OUT/s<seed>/<steps>")
    ps.add_argument("--validation", action="store_true",
                    help="read a validation-only or another pilot's cell and a command-free env (checks only)")
    pb = sub.add_parser("baseline", help="the entry baseline: an 8M parent pair prepared as the edge prepares it")
    pb.add_argument("--seed", type=int, required=True)
    pb.add_argument("--checkpoint", required=True)
    pb.add_argument("--vecnorm", required=True)
    pb.add_argument("--labels-from", default=None, help="a scorer module (label_episode, seed_outcome) to label rows")
    pb.add_argument("--zero-nudge-check", action="store_true", help="also roll a 0-deg nudge per recovery reset")
    pb.add_argument("--expect-head", default=None)
    pb.add_argument("--limit-resets", type=int, default=None, help="tests only: a partial run, no verdict")
    for p in (ps, pb):
        p.add_argument("--json", required=True)
        p.add_argument("--plot", default=None, help="the straightness look's PNG")
        p.add_argument("--traces", default=None, help="an npz of every episode's heading error and pelvis path")
    a = ap.parse_args(argv)
    why = refuse_existing([a.json, a.plot, a.traces])
    if why:
        print(f"refusing (nothing written): {why}", flush=True)
        return 2
    if a.mode == "baseline" and a.limit_resets is not None and not 1 <= a.limit_resets <= len(RECOVERY_SEEDS):
        print(f"refusing (nothing written): --limit-resets must be 1..{len(RECOVERY_SEEDS)}", flush=True)
        return 2
    try:
        return run_seed(a) if a.mode == "seed" else run_baseline(a)
    except _Refuse as exc:
        print("refusing (nothing written): " + "; ".join(exc.reasons), flush=True)
        return 2
    except _Internal as exc:
        print("internal check failed (nothing written): " + "; ".join(exc.reasons), flush=True)
        return 3


if __name__ == "__main__":
    sys.exit(main())
