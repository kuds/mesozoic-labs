#!/usr/bin/env python
"""gait_probe.py -- reusable gait probe for Mesozoic Labs SB3 checkpoints (any species, any stage).

RECORD
  The hand-run probe of the 2026-09-28 gait audit (docs/investigations/GAIT_AUDIT_2026_09.md), frozen with
  that note: nothing imports it, and the gait plan's PR-G1 (docs/GAIT_QUALITY_PLAN_2026_09.md) ports it into
  the library.  Run it by hand from the repository root, as in USAGE below, e.g.
    PYTHONPATH=. MUJOCO_GL=osmesa python docs/investigations/gait_2026_09/gait_probe.py --help
    PYTHONPATH=. MUJOCO_GL=osmesa python docs/investigations/gait_2026_09/gait_probe.py \
        --species compsognathus --stage locomotion --zero-action --episodes 2 --out <outdir>

WHAT IT DOES
  Rolls a saved SB3 policy on the stage's OWN environment (``config.load_stage_config`` +
  ``species_registry``), with the checkpoint's VecNormalize statistics applied in eval mode
  (training=False, norm_reward=False; loaded by the repo's ``policy_loading.load_sb3_checkpoint``),
  deterministic actions, and the plant contract enforced (``plant_contract.current_plant_identity``).
  Per control step it records the env's own per-foot contact reading
  (``BaseDinoEnv._aggregated_foot_contact_forces()``: per-foot MIN over the frame_skip substeps of
  the SUM of every touch sensor of that foot -- T. rex pad+3 digits, brachiosaurus pad+meta), foot
  site kinematics, root-body pose, joint angles, reward terms and info, then computes gait metrics
  per episode and summarises them (mean/std/min/max) over the panel.  The robot is never
  hard-coded: feet come from ``env._foot_sensor_groups`` (arity 2 or 4), foot labels from the
  touch-sensor names (r/l or fr/fl/rr/rl), the root body from the free joint, L/R joint pairs
  from joint names (left_/right_/l_/r_/fl_/fr_/rl_/rr_ prefixes).

SEED SCHEMES (--seed-scheme)
  panel  : env.reset(seed=seed_base + i) for episode i -- the stance_quality/v1 report
           (stance_report._roll_episodes; 40 episodes, seeds 3042-3081) and the recovery panel.
  vecenv : the 30-episode selected/final evaluation of reporting.stage_artifacts
           (create_vec_env(seed) -> make_env: set_random_seed(seed), env.reset(seed=seed) once;
           eval_policy then calls VecEnv.reset() per episode AND DummyVecEnv auto-resets after
           every done, so each episode after the first skips one unseeded reset).  Reproduces
           evaluation_selected.csv / evaluation_final.csv of reward_and_length/v1 and
           task_success/v1 nodes (seed 3042 = constants.PUBLICATION_SEED_START).
  auto   : panel for stance_quality/v1 and recovery_quality/v1 gates, vecenv otherwise.

USAGE
  WT=<clean worktree of the repo>
  PYTHONPATH=$WT MUJOCO_GL=osmesa OMP_NUM_THREADS=1 python gait_probe.py \
      --species compsognathus_robot --stage locomotion \
      --checkpoint <node>/models/robust_best_model.zip \
      --vecnorm    <node>/models/robust_best_model_vecnorm.pkl \
      --episodes 30 --seed-base 3042 --seed-scheme auto \
      --run-id 20260924_031815 --out <outdir> \
      [--expected <node>/evaluation_selected.csv <node>/gate_verdict.json <node>/stance_gate_report.json] \
      [--stage-config-json <node>/stage_config.json]  # warn if main's TOML (+ constructor defaults)
                                    #   differs from the env kwargs the run recorded
      [--use-recorded-env-kwargs]   # roll with the recorded kwargs instead of main's TOML
      [--tag <suffix>]              # extra suffix on every output name (second panel, final ckpt, ...)
      [--settle-steps N]            # gait-analysis window starts after N control steps
                                    #   (default: stage's settle_steps, else 1 s of control steps)
      [--trace-episode median|<i>]  # which episode gets the npz trace + plots (default: median reward)
      [--plot-start 6.0 --plot-span 3.0]
      [--video <replay.mp4> --video-start 6.0 --video-span 2.0 --video-frames 12 --video-crop x0,y0,x1,y1]
                                    # contact sheet; repo replays hold one frame per control step, the
                                    #   label t=(i+1)*frame_dt is the probe's plot clock
      [--video-frame-dt <env.dt>]   # sim seconds per frame (default 1/fps); repo replays are encoded at
                                    #   50 fps, so pass 0.01 for the 0.01 s species
      [--allow-legacy-plant]        # only admits checkpoints that PREDATE the plant contract
  Contact-sheet only (no rollout):  gait_probe.py --video replay.mp4 --out <dir> [--run-id ..]
  Do-nothing reference (no checkpoint): add --zero-action instead of --checkpoint/--vecnorm.
  Stage may be a semantic id (stance, recovery, locomotion, behavior) or a legacy number (1, 2, 3).
  Exit codes: 0 ok, 3 plant contract / checkpoint refused (reported, never forced).

OUTPUTS (--out is a directory)
  <species>_<run_id>_<stage>.json           meta, sanity check, per-episode rows, summary, diagnosis
  <species>_<run_id>_<stage>_ep<i>_trace.npz per-step trace of the representative episode
  <species>_<run_id>_<stage>_footfall.png   footfall diagram (--plot-span s)
  <species>_<run_id>_<stage>_foreaft.png    fore-aft foot position relative to the root body
  <species>_<run_id>_<stage>_contact_sheet.png  (with --video)

DEFINITIONS (all at control-step resolution, dt = env.dt)
  contact        EVERY contact-dependent metric is computed twice, keys prefixed touch_ / floor_:
                 touch_  the env's own reading: foot f is down at step k iff the aggregated touch force
                         (per-foot MIN over substeps of the summed sensors) > 0.1 N -- the repo's
                         per-foot threshold (stance_diagnostics.derive_stance_info, metrics.py, MJX).
                         This is what the rewards and the stance gate see.
                 floor_  ground truth recorded through the env's own per-substep hook
                         (BaseDinoEnv._substep_probe_hook, the real step path): normal force of
                         contacts between a static floor geom and the foot's LIMB (subtree of the
                         highest ancestor of its touch-site bodies that holds no other foot's sites
                         and is not the free-joint root) > 0.1 N on at least half of the step's
                         substeps.  Non-floor contact force counts contacts with other limbs / the
                         trunk / tail (intra-limb contacts excluded).
                 A MuJoCo touch sensor sums ALL contact normal forces inside its site volume,
                 including non-floor contacts (one sole resting on the other, foot against shin), so
                 touch_ can report support the floor never gives: see phantom_support_fraction_*,
                 nonfloor_contact_mean_force_bw_*, foot_foot_contact_substep_frac.  Also reported:
                 floor_duty_strict_allsubsteps_* (repo-style MIN aggregation of the floor force),
                 floor_duty_loaded10bw_* (10 % body weight), env_support_flag_fraction_touch_full vs
                 _floor_counterfactual_full (the env's sum-over-feet support flag on touch vs floor
                 forces) and paid_in_floor_flight_<reward term> (reward paid while no foot is on
                 the floor).
  window         steps k >= settle (see --settle-steps); duty/support/kinematics use this window.
  debounce       touchdown/lift-off events use contact with runs shorter than 2 control steps
                 removed (1-step gaps filled first, then 1-step blips dropped); raw contact is
                 used for duty and support fractions.
  touchdown merge (bipeds, and each fore/hind pair of quadrupeds): an L and an R touchdown within
                 2 control steps are ONE simultaneous event 'B' (the hop signature).
                 alternation_index = consecutive event pairs that switch foot (L->R, R->L) / pairs;
                 same_foot_repeat_fraction = (L->L, R->R) / pairs; simultaneous_fraction = B/events.
                 (Unlike LocomotionMetrics.gait_symmetry, which compares contact periods and scores a
                 synchronous hop 1.0, and the alternation reward, which logs a simultaneous landing
                 as R then L.)
  stride         touchdown to next touchdown of the same foot; stride_frequency = 1/mean stride
                 duration (per foot, then averaged) -- never "any foot loaded" switches.
  step           interval between consecutive merged touchdown events.
  heading frame  body +X of the env's own quaternion (sensor at env._sensor_quat_start, as the
                 heading reward uses; root xquat if absent).  fore-aft x_rel = (foot - root)·fwd.
  lead           x_rel(left) - x_rel(right) (fore pair and hind pair for quadrupeds); lead swaps are
                 sign changes of the debounced sign; ~2 per stride in a walk, ~0 in a staggered scoot.
  foot travel    increments of the foot's primary touch site along the episode's travel direction
                 (unit vector of root displacement over the window); share = foot/all feet;
                 in_contact fraction = forward travel accrued while the foot was down at both ends
                 of the step (sliding) / its forward travel.
  slip           mean horizontal speed of the primary touch site while down (mj_objectVelocity).
  clearance      max over a swing of foot height (min z of its sites above ground) minus the
                 foot's median height while down.
  joint lag      for each L/R joint pair (mirror-sign corrected from the joint axes at the home
                 keyframe), the lag in [0, P) maximising corr(left[t], right[t+lag]) as a fraction
                 of the stride P (median floor_ stride, else touch_ stride, else hip_pitch FFT
                 period; resolution 1/P): walk/run ~0.5, hop/bound ~0.
  gait_class     heuristic label per contact source (touch_gait_class / floor_gait_class): bipeds
                 stand / stand (stepping in place) / slide / hop / walk / run / staggered scoot /
                 shuffle / irregular; quadrupeds as above.  Every number behind it is in the row.
  L (leg)        mean world-z of the hip_pitch joint anchors at the 'home' keyframe (hind hips for
                 quadrupeds); Froude = v^2/(g L), dimensionless stride = v * T_stride / L.
  pelvis         root-body z: amplitude = (p95 - p5)/2 of the linearly detrended z, dominant
                 frequency from a Hann-windowed FFT; roll/pitch std from the same quaternion.
  quadrupeds     limb phases = touchdown time after the left-hind (LH) touchdown / LH stride;
                 circular mean + concentration R; Hildebrand-style class from the RH phase
                 (0 -> hind pair synchronous: pronk/bound; 0.5 -> symmetric: pace/trot/walk by the
                 LF lateral phase and hind duty factor; other -> gallop).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROBE_VERSION = "gait_probe/v1"
REPO_CONTACT_THRESHOLD_N = 0.1  # stance_diagnostics.derive_stance_info / metrics.py / MJX
LOADED_FRACTION_OF_BW = 0.10
MIN_RUN_STEPS = 2  # debounce: ignore contact flicker shorter than 2 control steps
MERGE_STEPS = 2  # L/R touchdowns within 2 control steps = one simultaneous event


# ───────────────────────────── small numeric helpers ─────────────────────────────


def _runs(b: np.ndarray) -> list[tuple[bool, int, int]]:
    """Run-length encoding: [(value, start, length), ...]."""
    b = np.asarray(b, dtype=bool)
    if b.size == 0:
        return []
    change = np.flatnonzero(np.diff(b.astype(np.int8))) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [b.size]])
    return [(bool(b[s]), int(s), int(e - s)) for s, e in zip(starts, ends)]


def debounce(b: np.ndarray, min_run: int = MIN_RUN_STEPS) -> np.ndarray:
    """Remove interior runs shorter than ``min_run``: fill short gaps first, then drop short blips."""
    out = np.asarray(b, dtype=bool).copy()
    for value, start, length in _runs(out):
        if not value and length < min_run and start > 0 and start + length < out.size:
            out[start : start + length] = True
    for value, start, length in _runs(out):
        if value and length < min_run and start > 0 and start + length < out.size:
            out[start : start + length] = False
    return out


def _events(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(touchdown indices, lift-off indices) of a debounced contact array (k where state changed)."""
    b = np.asarray(b, dtype=np.int8)
    d = np.diff(b)
    return np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1


def _nan() -> float:
    return float("nan")


def _mean(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else _nan()


def _median(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if x.size else _nan()


def _circ_stats(phases) -> tuple[float, float]:
    """Circular mean (in [0,1)) and resultant length R of phases given as cycle fractions."""
    p = np.asarray(phases, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return _nan(), _nan()
    z = np.exp(2j * np.pi * p).mean()
    return float((np.angle(z) / (2 * np.pi)) % 1.0), float(abs(z))


def _circ_dist(a: float, b: float) -> float:
    d = abs((a - b) % 1.0)
    return min(d, 1.0 - d)


def _dominant_frequency(x: np.ndarray, dt: float, fmin: float = 0.2) -> tuple[float, float]:
    """(dominant frequency Hz, fraction of non-DC power in that bin±1) of a detrended signal."""
    x = np.asarray(x, dtype=float)
    if x.size < 16 or not np.all(np.isfinite(x)):
        return _nan(), _nan()
    t = np.arange(x.size)
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    if np.std(x) < 1e-12:
        return _nan(), _nan()
    spec = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    freqs = np.fft.rfftfreq(x.size, dt)
    ok = freqs >= fmin
    if not ok.any():
        return _nan(), _nan()
    idx = int(np.flatnonzero(ok)[np.argmax(spec[ok])])
    lo, hi = max(idx - 1, 1), min(idx + 2, spec.size)
    return float(freqs[idx]), float(spec[lo:hi].sum() / max(spec[1:].sum(), 1e-30))


def _quat_to_euler(quat) -> tuple[float, float, float]:
    """MuJoCo (w,x,y,z) -> (roll, pitch, yaw); same formula as stance_diagnostics._quat_to_euler."""
    w, x, y, z = (float(v) for v in quat)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def _fwd2d(quat) -> np.ndarray:
    """Body +X projected on the ground plane (reward_functions.quat_to_forward_2d)."""
    w, x, y, z = (float(v) for v in quat)
    v = np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)])
    n = np.linalg.norm(v)
    return v / n if n > 1e-6 else v


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


# ───────────────────────────── robot description from the env ─────────────────────────────

_SIDE_RE = re.compile(r"^(left|right|fl|fr|rl|rr|l|r)_(.+)$")
_SIDE_MAP = {
    "left": ("", "L"),
    "right": ("", "R"),
    "l": ("", "L"),
    "r": ("", "R"),
    "fl": ("fore", "L"),
    "fr": ("fore", "R"),
    "rl": ("hind", "L"),
    "rr": ("hind", "R"),
}
_DISPLAY = {"r": "R", "l": "L", "fr": "RF", "fl": "LF", "rr": "RH", "rl": "LH"}


class RobotDescription:
    """Everything the probe needs about a species, read from a live env (never hard-coded)."""

    def __init__(self, env: Any):
        import mujoco

        u = env.unwrapped
        m = u.model
        self.model = m
        groups = tuple(tuple(int(a) for a in g) for g in getattr(u, "_foot_sensor_groups", ()))
        if not groups:
            raise RuntimeError(f"{type(u).__name__} declares no _foot_sensor_groups; cannot probe gait")
        adr_to_sensor = {int(m.sensor_adr[i]): i for i in range(m.nsensor)}
        self.labels: list[str] = []
        self.sensor_names: list[list[str]] = []
        self.site_ids: list[list[int]] = []
        for gi, group in enumerate(groups):
            sids = [adr_to_sensor[a] for a in group]
            names = [m.sensor(s).name for s in sids]
            sites = []
            for s in sids:
                if int(m.sensor_objtype[s]) != int(mujoco.mjtObj.mjOBJ_SITE):
                    raise RuntimeError(f"foot sensor {m.sensor(s).name} is not attached to a site")
                sites.append(int(m.sensor_objid[s]))
            prefix = names[0].split("_")[0].lower()
            prefix = {"right": "r", "left": "l"}.get(prefix, prefix)
            self.labels.append(prefix if prefix in _DISPLAY else f"foot{gi}")
            self.sensor_names.append(names)
            self.site_ids.append(sites)
        self.n_feet = len(groups)
        self.display = [_DISPLAY.get(lbl, lbl) for lbl in self.labels]
        self.quadruped = self.n_feet == 4 and set(self.labels) == {"fr", "fl", "rr", "rl"}
        self.biped = self.n_feet == 2 and set(self.labels) == {"r", "l"}
        self.side = [lbl[-1].upper() if lbl[-1] in "rl" else "?" for lbl in self.labels]
        self.limb = [("fore" if lbl[0] == "f" else "hind") if len(lbl) == 2 else "" for lbl in self.labels]
        self.primary_site = [s[0] for s in self.site_ids]
        # ground-truth contact bookkeeping: a foot's "leg" = the subtree of the highest ancestor of its
        # touch-site bodies that contains no other foot's sites and is not the free-joint root, i.e. the
        # whole limb (so a toe the touch sensor does not cover still counts as that leg's support).
        free_bodies = {
            int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
        }

        def subtree(root_b: int) -> set[int]:
            out = {root_b}
            for b in range(root_b + 1, m.nbody):
                if int(m.body_parentid[b]) in out:
                    out.add(b)
            return out

        site_bodies = [{int(m.site_bodyid[s]) for s in sites} for sites in self.site_ids]
        self.foot_bodies: list[set[int]] = []
        for fi, bodies in enumerate(site_bodies):
            others = set().union(*[sb for fj, sb in enumerate(site_bodies) if fj != fi])
            tops = set()
            for b in bodies:
                while True:
                    parent = int(m.body_parentid[b])
                    if parent == 0 or parent in free_bodies or subtree(parent) & others:
                        break
                    b = parent
                tops.add(b)
            leg: set[int] = set()
            for b in tops:
                leg |= subtree(b)
            self.foot_bodies.append(leg)
        self.leg_top_bodies = [
            sorted({m.body(b).name for b in fb if int(m.body_parentid[b]) not in fb}) for fb in self.foot_bodies
        ]
        self.geom_foot = np.full(m.ngeom, -1, dtype=int)
        for fi, bodies in enumerate(self.foot_bodies):
            for g in range(m.ngeom):
                if int(m.geom_bodyid[g]) in bodies:
                    self.geom_foot[g] = fi
        floor = getattr(u, "_static_floor_geoms", None)
        floor_ids = list(floor()) if callable(floor) else []
        if not floor_ids and getattr(u, "floor_geom_id", None) is not None:
            floor_ids = [int(u.floor_geom_id)]
        self.is_floor = np.zeros(m.ngeom, dtype=bool)
        self.is_floor[[int(g) for g in floor_ids]] = True
        # root body = body of the (first) free joint
        free = [j for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        if not free:
            raise RuntimeError("model has no free joint")
        self.root_joint = free[0]
        self.root_body = int(m.jnt_bodyid[self.root_joint])
        self.root_name = m.body(self.root_body).name
        self.root_qposadr = int(m.jnt_qposadr[self.root_joint])
        self.root_dofadr = int(m.jnt_dofadr[self.root_joint])
        self.quat_sensor_start = getattr(u, "_sensor_quat_start", None)
        self.quat_source = (
            f"sensordata[{self.quat_sensor_start}:+4] (env heading quaternion)"
            if self.quat_sensor_start is not None
            else f"xquat[{self.root_name}]"
        )
        self.body_mass = float(m.body_subtreemass[self.root_body])
        self.gravity = abs(float(m.opt.gravity[2]))
        self.body_weight_n = self.body_mass * self.gravity
        self.env_contact_threshold = getattr(u, "_contact_threshold", None)
        # hinge joints and L/R pairs
        self.hinges = [
            j
            for j in range(m.njnt)
            if int(m.jnt_type[j]) in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
        ]
        self.hinge_names = [m.joint(j).name for j in self.hinges]
        self.hinge_qposadr = np.array([int(m.jnt_qposadr[j]) for j in self.hinges], dtype=int)
        parsed: dict[tuple[str, str, str], int] = {}
        for idx, name in enumerate(self.hinge_names):
            mt = _SIDE_RE.match(name)
            if mt:
                group, side = _SIDE_MAP[mt.group(1)]
                parsed[(group, side, mt.group(2))] = idx
        # home keyframe: hip heights and joint axes in the root frame (mirror signs)
        data = mujoco.MjData(m)
        key = 0
        try:
            key = int(m.key("home").id)
        except Exception:  # noqa: BLE001
            key = 0
        if m.nkey:
            mujoco.mj_resetDataKeyframe(m, data, key)
        mujoco.mj_forward(m, data)
        root_R = data.xmat[self.root_body].reshape(3, 3)
        mirror = np.diag([1.0, -1.0, 1.0])

        def axis_root(j: int) -> np.ndarray:
            return root_R.T @ data.xaxis[j]

        self.pairs: list[dict[str, Any]] = []
        keys = sorted({(g, rest) for (g, _s, rest) in parsed})
        for g, rest in keys:
            if (g, "L", rest) in parsed and (g, "R", rest) in parsed:
                li, ri = parsed[(g, "L", rest)], parsed[(g, "R", rest)]
                sign = float(np.sign(np.dot(axis_root(self.hinges[ri]), -mirror @ axis_root(self.hinges[li]))) or 1.0)
                self.pairs.append(
                    {"name": f"{g + '_' if g else ''}{rest}", "a": li, "b": ri, "sign_b": sign, "kind": "L_vs_R"}
                )
        if self.quadruped:
            for rest in sorted({rest for (g, _s, rest) in parsed if g}):
                for label, (ga, sa), (gb, sb) in (
                    ("ipsilateral_LF_vs_LH", ("fore", "L"), ("hind", "L")),
                    ("diagonal_RF_vs_LH", ("fore", "R"), ("hind", "L")),
                ):
                    if (ga, sa, rest) in parsed and (gb, sb, rest) in parsed:
                        ai, bi = parsed[(ga, sa, rest)], parsed[(gb, sb, rest)]
                        sign = float(np.sign(np.dot(axis_root(self.hinges[ai]), axis_root(self.hinges[bi]))) or 1.0)
                        self.pairs.append({"name": f"{label}_{rest}", "a": ai, "b": bi, "sign_b": sign, "kind": label})
        hips: dict[str, list[float]] = {"fore": [], "hind": [], "": []}
        for (g, _s, rest), idx in parsed.items():
            if rest == "hip_pitch":
                hips[g].append(float(data.xanchor[self.hinges[idx]][2]))
        if self.quadruped and hips["hind"]:
            self.leg_length = float(np.mean(hips["hind"]))
            self.leg_length_source = "mean world z of the hind hip_pitch joint anchors at the home keyframe"
        elif any(hips.values()):
            allh = [h for v in hips.values() for h in v]
            self.leg_length = float(np.mean(allh))
            self.leg_length_source = "mean world z of the hip_pitch joint anchors at the home keyframe"
        else:
            self.leg_length = float(data.xpos[self.root_body][2])
            self.leg_length_source = "root-body z at the home keyframe (no hip_pitch joints found)"
        self.hip_heights = {k or "all": v for k, v in hips.items() if v}
        self.home_root_z = float(data.xpos[self.root_body][2])

    def describe(self) -> dict[str, Any]:
        return {
            "feet": [
                {
                    "label": lbl,
                    "display": d,
                    "side": s,
                    "limb": lb,
                    "sensors": sn,
                    "sites": [self.model.site(i).name for i in si],
                }
                for lbl, d, s, lb, sn, si in zip(
                    self.labels, self.display, self.side, self.limb, self.sensor_names, self.site_ids
                )
            ],
            "leg_subtree_tops_ground_truth": self.leg_top_bodies,
            "leg_bodies_ground_truth": [[self.model.body(b).name for b in sorted(bs)] for bs in self.foot_bodies],
            "floor_geoms": [self.model.geom(int(g)).name for g in np.flatnonzero(self.is_floor)],
            "root_body": self.root_name,
            "quat_source": self.quat_source,
            "body_mass_kg": self.body_mass,
            "body_weight_n": self.body_weight_n,
            "env_contact_threshold_n": self.env_contact_threshold,
            "leg_length_m": self.leg_length,
            "leg_length_source": self.leg_length_source,
            "hip_heights_home_m": self.hip_heights,
            "joint_pairs": [
                {
                    "name": p["name"],
                    "a": self.hinge_names[p["a"]],
                    "b": self.hinge_names[p["b"]],
                    "sign_b": p["sign_b"],
                    "kind": p["kind"],
                }
                for p in self.pairs
            ],
        }


# ───────────────────────────── rollout ─────────────────────────────


def _record_state(u: Any, rd: RobotDescription, buf: dict[str, list]) -> None:
    d = u.data
    ground = getattr(u, "_ground_height_at", lambda xy: 0.0)
    pos = np.array([d.site_xpos[s] for s in rd.primary_site], dtype=float)
    heights = np.array(
        [min(float(d.site_xpos[s][2]) - float(ground(d.site_xpos[s][:2])) for s in sites) for sites in rd.site_ids]
    )
    buf["foot_pos"].append(pos)
    buf["foot_h"].append(heights)
    buf["root_pos"].append(np.array(d.xpos[rd.root_body], dtype=float))
    if rd.quat_sensor_start is not None:
        q = np.array(d.sensordata[rd.quat_sensor_start : rd.quat_sensor_start + 4], dtype=float)
    else:
        q = np.array(d.xquat[rd.root_body], dtype=float)
    buf["quat"].append(q)
    buf["qj"].append(np.array(d.qpos[rd.hinge_qposadr], dtype=float))


def rollout_episode(env: Any, predict: Any, rd: RobotDescription, reset_seed: "int | None") -> dict[str, Any]:
    """One deterministic episode; returns per-step arrays (state arrays have T+1 rows, index 0 = reset)."""
    import mujoco

    from environments.shared.stance_diagnostics import derive_stance_info

    u = env.unwrapped
    obs, _ = env.reset(seed=reset_seed) if reset_seed is not None else env.reset()
    reset_predict = getattr(predict, "reset", None)
    if callable(reset_predict):
        reset_predict()
    st: dict[str, list] = {k: [] for k in ("foot_pos", "foot_h", "root_pos", "quat", "qj")}
    _record_state(u, rd, st)
    forces, forces_inst, foot_vel, root_vel, rewards, fwd, stance = [], [], [], [], [], [], []
    floor_min, floor_mean, floor_frac, nonfloor_mean, footfoot_frac = [], [], [], [], []
    m, d = u.model, u.data
    F = rd.n_feet
    f6 = np.zeros(6)
    sub: list[tuple[np.ndarray, np.ndarray, bool]] = []

    def hook() -> None:
        # Ground truth per physics substep: floor normal force on each foot's bodies, normal force
        # from NON-floor contacts on the foot (self/foot-foot contact), and whether two feet touch.
        floor_n = np.zeros(F)
        other_n = np.zeros(F)
        foot_foot = False
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            f1, f2 = rd.geom_foot[g1], rd.geom_foot[g2]
            if f1 < 0 and f2 < 0:
                continue
            mujoco.mj_contactForce(m, d, i, f6)
            fn = float(f6[0])
            if rd.is_floor[g1] or rd.is_floor[g2]:
                floor_n[f1 if f1 >= 0 else f2] += fn
                continue
            if f1 == f2:
                continue  # intra-limb self contact
            if f1 >= 0:
                other_n[f1] += fn
            if f2 >= 0:
                other_n[f2] += fn
            if f1 >= 0 and f2 >= 0 and f1 != f2:
                foot_foot = True
        sub.append((floor_n, other_n, foot_foot))

    previous_hook = u._substep_probe_hook
    u._substep_probe_hook = hook
    comps: dict[str, list[float]] = {}
    vel6 = np.zeros(6)
    info: dict[str, Any] = {}
    steps = 0
    terminated = truncated = False
    while True:
        action = np.asarray(predict(obs), dtype=np.float64).ravel()
        sub.clear()
        try:
            obs, reward, terminated, truncated, info = env.step(action)
        except BaseException:
            u._substep_probe_hook = previous_hook
            raise
        fl = np.array([x[0] for x in sub])
        floor_min.append(fl.min(axis=0))
        floor_mean.append(fl.mean(axis=0))
        floor_frac.append((fl > REPO_CONTACT_THRESHOLD_N).mean(axis=0))
        nonfloor_mean.append(np.array([x[1] for x in sub]).mean(axis=0))
        footfoot_frac.append(float(np.mean([x[2] for x in sub])))
        forces.append(np.asarray(u._aggregated_foot_contact_forces(), dtype=float))
        forces_inst.append(np.asarray(u._foot_contact_forces(), dtype=float))
        fv = []
        for s in rd.primary_site:
            mujoco.mj_objectVelocity(u.model, u.data, mujoco.mjtObj.mjOBJ_SITE, s, vel6, 0)
            fv.append(vel6[3:6].copy())
        foot_vel.append(np.array(fv))
        root_vel.append(np.array(u.data.qvel[rd.root_dofadr : rd.root_dofadr + 3], dtype=float))
        _record_state(u, rd, st)
        rewards.append(float(reward))
        fwd.append(float(info.get("forward_vel", np.nan)))
        s = derive_stance_info(info)
        stance.append(
            (
                s.get("unsupported_duty", np.nan),
                s.get("bilateral_support_duty", np.nan),
                s.get("single_support_duty", np.nan),
            )
        )
        for key, value in info.items():
            if key.startswith("reward_") and key != "reward_total":
                try:
                    comps.setdefault(key, []).append(float(value))
                except (TypeError, ValueError):
                    pass
        steps += 1
        if terminated or truncated:
            break
    u._substep_probe_hook = previous_hook
    out: dict[str, Any] = {k: np.asarray(v) for k, v in st.items()}
    out.update(
        floor_force_min=np.asarray(floor_min),
        floor_force_mean=np.asarray(floor_mean),
        floor_contact_substep_frac=np.asarray(floor_frac),
        nonfloor_force_mean=np.asarray(nonfloor_mean),
        foot_foot_contact_frac=np.asarray(footfoot_frac),
        forces=np.asarray(forces),
        forces_inst=np.asarray(forces_inst),
        foot_vel=np.asarray(foot_vel),
        root_vel=np.asarray(root_vel),
        reward=np.asarray(rewards),
        forward_vel=np.asarray(fwd),
        stance_flags=np.asarray(stance, dtype=float),
        components={k: np.asarray(v) for k, v in comps.items()},
        length=steps,
        total_reward=float(np.sum(rewards)),
        terminated=bool(terminated),
        termination_reason=str(info.get("termination_reason", "terminated" if terminated else "truncated")),
        distance_traveled=float(info.get("distance_traveled", np.nan)),
        final_drift_distance=float(info.get("drift_distance", np.nan)),
    )
    return out


# ───────────────────────────── per-episode analysis ─────────────────────────────


def _pair_events(td: dict[int, np.ndarray], left: int, right: int) -> list[tuple[int, str]]:
    """Merge L/R touchdowns within MERGE_STEPS into a simultaneous 'B' event."""
    evs = sorted([(int(k), "L") for k in td[left]] + [(int(k), "R") for k in td[right]])
    merged: list[tuple[int, str]] = []
    i = 0
    while i < len(evs):
        if i + 1 < len(evs) and evs[i + 1][1] != evs[i][1] and evs[i + 1][0] - evs[i][0] <= MERGE_STEPS:
            merged.append((evs[i][0], "B"))
            i += 2
        else:
            merged.append(evs[i])
            i += 1
    return merged


def _alternation(events: list[tuple[int, str]]) -> dict[str, float]:
    seq = [e for _, e in events]
    n_pairs = max(len(seq) - 1, 0)
    alt = sum(1 for a, b in zip(seq, seq[1:]) if {a, b} == {"L", "R"})
    same = sum(1 for a, b in zip(seq, seq[1:]) if a == b and a in "LR")
    n_b = seq.count("B")
    return {
        "n_touchdown_events": float(len(seq)),
        "alternation_index": alt / n_pairs if n_pairs else _nan(),
        "same_foot_repeat_fraction": same / n_pairs if n_pairs else _nan(),
        "simultaneous_fraction": n_b / len(seq) if seq else _nan(),
        "pairs_involving_simultaneous_fraction": (n_pairs - alt - same) / n_pairs if n_pairs else _nan(),
    }


def _joint_lag(a: np.ndarray, b: np.ndarray, period: float) -> tuple[float, float, float]:
    """(lag fraction in [0,1), corr at best lag, corr at lag 0) for b following a."""
    if not np.isfinite(period) or period < 4 or a.size < 2 * period:
        return _nan(), _nan(), _nan()
    a = a - a.mean()
    b = b - b.mean()
    if a.std() < 1e-3 or b.std() < 1e-3:  # < 0.06 deg: no oscillation to phase
        return _nan(), _nan(), _nan()
    p = int(round(period))
    cs = []
    for k in range(p):
        x, y = a[: a.size - k], b[k:]
        cs.append(float(np.corrcoef(x, y)[0, 1]))
    k_best = int(np.argmax(cs))
    return k_best / p, cs[k_best], cs[0]


CONTACT_SOURCES = ("touch", "floor")


def contact_arrays(tr: dict[str, Any], W: np.ndarray) -> dict[str, np.ndarray]:
    """Per-source boolean foot-down arrays over the window.

    touch : the env's own reading -- aggregated (min over substeps) touch-sensor sum > 0.1 N.
            This is what the rewards and the stance gate see.  A touch volume also registers
            NON-floor contacts inside it (e.g. one sole resting on the other), so it can report
            support that the floor never provides.
    floor : ground truth -- normal force between the floor and the foot's bodies > 0.1 N on at
            least half of the control step's physics substeps.
    """
    return {
        "touch": tr["forces"][W] > REPO_CONTACT_THRESHOLD_N,
        "floor": tr["floor_contact_substep_frac"][W] >= 0.5,
    }


def _contact_block(
    row: dict[str, Any], p: str, raw: np.ndarray, rd: RobotDescription, dt: float, ctx: dict[str, Any]
) -> None:
    """All contact-dependent metrics for one contact source, keys prefixed with ``p``."""
    nW, F = raw.shape
    tw = nW * dt
    deb = np.stack([debounce(raw[:, f]) for f in range(F)], axis=1)
    ctx[p + "deb"] = deb
    for f in range(F):
        row[f"{p}duty_{rd.display[f]}"] = float(raw[:, f].mean())
    n_down = raw.sum(axis=1)
    for c in range(F + 1):
        row[f"{p}support_frac_{c}"] = float(np.mean(n_down == c))
    row[f"{p}flight_fraction"] = row[f"{p}support_frac_0"]
    if rd.biped:
        L, R = rd.labels.index("l"), rd.labels.index("r")
        row[f"{p}single_support_fraction"] = row[f"{p}support_frac_1"]
        row[f"{p}double_support_fraction"] = row[f"{p}support_frac_2"]
        dl, dr = row[f"{p}duty_L"], row[f"{p}duty_R"]
        row[f"{p}duty_asymmetry_L_minus_R"] = dl - dr
        row[f"{p}duty_asymmetry_norm"] = abs(dl - dr) / max((dl + dr) / 2, 1e-9)
        names = ["both", "left_only", "right_only", "none"]
        state = np.where(deb[:, L] & deb[:, R], 0, np.where(deb[:, L], 1, np.where(deb[:, R], 2, 3)))
        mat = np.zeros((4, 4), dtype=int)
        for a, b in zip(state[:-1], state[1:]):
            mat[a, b] += 1
        row[f"{p}transition_states"] = names
        row[f"{p}transition_counts"] = mat.tolist()
        for i, a in enumerate(names):
            off = mat[i].sum() - mat[i, i]
            for j, b in enumerate(names):
                if i != j:
                    row[f"{p}trans_{a}->{b}_per_s"] = mat[i, j] / tw
                    row[f"{p}trans_p_{a}->{b}"] = mat[i, j] / off if off else _nan()
    idx = {lbl: rd.labels.index(lbl) for lbl in rd.labels}
    if rd.quadruped:
        fore = (row[f"{p}duty_LF"] + row[f"{p}duty_RF"]) / 2
        hind = (row[f"{p}duty_LH"] + row[f"{p}duty_RH"]) / 2
        row[f"{p}duty_fore_mean"], row[f"{p}duty_hind_mean"] = fore, hind
        row[f"{p}duty_asymmetry_fore_minus_hind"] = fore - hind
        row[f"{p}duty_asymmetry_L_minus_R_fore"] = row[f"{p}duty_LF"] - row[f"{p}duty_RF"]
        row[f"{p}duty_asymmetry_L_minus_R_hind"] = row[f"{p}duty_LH"] - row[f"{p}duty_RH"]
        patt = ["+".join(rd.display[f] for f in range(F) if deb[k, f]) or "none" for k in range(nW)]
        pc = Counter(a + "->" + b for a, b in zip(patt[:-1], patt[1:]) if a != b)
        row[f"{p}top_support_pattern_transitions"] = [[k, v] for k, v in pc.most_common(10)]
        row[f"{p}top_support_patterns"] = [[k, v / nW] for k, v in Counter(patt).most_common(8)]
        for f in range(F):
            mat = np.zeros((2, 2), dtype=int)
            for a, b in zip(deb[:-1, f], deb[1:, f]):
                mat[int(a), int(b)] += 1
            row[f"{p}foot_transition_counts_{rd.display[f]}"] = mat.tolist()
    # events, strides, swings
    td: dict[int, np.ndarray] = {}
    lo: dict[int, np.ndarray] = {}
    stride_all: list[float] = []
    fh, fvel, fwd_inc = ctx["foot_h"], ctx["foot_vel"], ctx["fwd_inc"]
    for f in range(F):
        lab = rd.display[f]
        td[f], lo[f] = _events(deb[:, f])
        row[f"{p}touchdowns_{lab}"] = float(td[f].size)
        row[f"{p}swings_per_s_{lab}"] = lo[f].size / tw
        strides = np.diff(td[f]) * dt
        row[f"{p}stride_mean_s_{lab}"] = _mean(strides)
        stride_all.extend(strides.tolist())
        sw = [(a, int(b)) for a in lo[f] for b in td[f][td[f] > a][:1]]
        stc = [(a, int(b)) for a in td[f] for b in lo[f][lo[f] > a][:1]]
        row[f"{p}swing_mean_s_{lab}"] = _mean([(b - a) * dt for a, b in sw])
        row[f"{p}stance_mean_s_{lab}"] = _mean([(b - a) * dt for a, b in stc])
        down = raw[:, f]
        base = _median(fh[down, f]) if down.any() else _nan()
        clear = [float(fh[a:b, f].max() - base) for a, b in sw if b > a]
        row[f"{p}swing_clearance_mean_m_{lab}"] = _mean(clear)
        row[f"{p}swing_clearance_max_m_{lab}"] = float(np.max(clear)) if clear else _nan()
        vxy = np.linalg.norm(fvel[:, f, :2], axis=1)
        row[f"{p}slip_speed_in_contact_mps_{lab}"] = _mean(vxy[down])
        row[f"{p}foot_speed_off_ground_mps_{lab}"] = _mean(vxy[~down])
        both_down = down & np.concatenate([[False], down[:-1]])
        ft = row[f"forward_travel_m_{lab}"]
        row[f"{p}in_contact_forward_travel_fraction_{lab}"] = (
            float(np.clip(fwd_inc[both_down, f], 0, None).sum() / ft) if ft > 1e-9 else _nan()
        )
    per_foot_freq = [
        1.0 / row[f"{p}stride_mean_s_{rd.display[f]}"]
        for f in range(F)
        if np.isfinite(row[f"{p}stride_mean_s_{rd.display[f]}"])
    ]
    row[f"{p}stride_duration_mean_s"] = _mean(stride_all)
    row[f"{p}stride_duration_median_s"] = _median(stride_all)
    row[f"{p}stride_frequency_hz"] = _mean(per_foot_freq)
    row[f"{p}touchdowns_per_s_all_feet"] = sum(td[f].size for f in range(F)) / tw
    row[f"{p}in_contact_forward_travel_fraction_mean"] = _mean(
        [row[f"{p}in_contact_forward_travel_fraction_{rd.display[f]}"] for f in range(F)]
    )
    if rd.biped:
        evs = _pair_events(td, L, R)
        for k, v in _alternation(evs).items():
            row[f"{p}{k}"] = v
        times = np.array([k for k, _ in evs], dtype=float)
        row[f"{p}step_duration_mean_s"] = _mean(np.diff(times) * dt) if times.size > 1 else _nan()
        row[f"{p}step_duration_median_s"] = _median(np.diff(times) * dt) if times.size > 1 else _nan()
        row[f"{p}footfall_sequence_head"] = "".join(e for _, e in evs[:40])
    if rd.quadruped:
        for pair, (lf, rf) in {"fore": (idx["fl"], idx["fr"]), "hind": (idx["rl"], idx["rr"])}.items():
            for k, v in _alternation(_pair_events(td, lf, rf)).items():
                row[f"{p}{pair}_{k}"] = v
        allev = sorted((int(k), rd.display[f]) for f in range(F) for k in td[f])
        row[f"{p}footfall_sequence_head"] = " ".join(lbl for _, lbl in allev[:32])
        times = np.array(sorted({k for k, _ in allev}), dtype=float)
        row[f"{p}step_duration_mean_s"] = _mean(np.diff(times) * dt) if times.size > 1 else _nan()
        ref = td[idx["rl"]]
        phases: dict[str, list[float]] = {"RH": [], "LF": [], "RF": []}
        orders = []
        for t0, t1 in zip(ref[:-1], ref[1:]):
            order = [(0.0, "LH")]
            for lab, fi in (("RH", idx["rr"]), ("LF", idx["fl"]), ("RF", idx["fr"])):
                inside = td[fi][(td[fi] >= t0) & (td[fi] < t1)]
                ph = (inside[0] - t0) / (t1 - t0) if inside.size else np.nan
                phases[lab].append(ph)
                if np.isfinite(ph):
                    order.append((ph, lab))
            orders.append("-".join(lbl for _, lbl in sorted(order)))
        for lab, ph in phases.items():
            mu, rr = _circ_stats(ph)
            row[f"{p}phase_{lab}_rel_LH"] = mu
            row[f"{p}phase_R_{lab}_rel_LH"] = rr
        row[f"{p}footfall_order_mode"] = Counter(orders).most_common(1)[0][0] if orders else ""
        row[f"{p}footfall_order_mode_fraction"] = (
            Counter(orders).most_common(1)[0][1] / len(orders) if orders else _nan()
        )
    # lead swaps per stride, stride length
    sf_hz = row[f"{p}stride_frequency_hz"]
    for pfx in ctx["lead_prefixes"]:
        swaps = row[f"{pfx}lead_swaps_per_s"] * tw
        row[f"{p}{pfx}lead_swaps_per_stride"] = swaps / (tw * sf_hz) if np.isfinite(sf_hz) and sf_hz > 0 else _nan()
    v = ctx["speed"]
    sd = row[f"{p}stride_duration_mean_s"]
    row[f"{p}stride_length_m"] = v * sd if np.isfinite(sd) else _nan()
    row[f"{p}stride_length_over_L"] = row[f"{p}stride_length_m"] / rd.leg_length if np.isfinite(sd) else _nan()
    fz = row["pelvis_z_dominant_hz"]
    row[f"{p}pelvis_z_freq_over_stride_freq"] = (
        fz / sf_hz if np.isfinite(fz) and np.isfinite(sf_hz) and sf_hz > 0 else _nan()
    )
    row[f"{p}gait_class"] = (
        classify_biped(row, p) if rd.biped else (classify_quadruped(row, p) if rd.quadruped else "n/a")
    )


def analyze_episode(
    tr: dict[str, Any], rd: RobotDescription, dt: float, settle: int, gate_settle: int
) -> dict[str, Any]:
    T = int(tr["length"])
    F = rd.n_feet
    row: dict[str, Any] = {
        "reward": tr["total_reward"],
        "length": float(T),
        "mean_forward_velocity_full": _mean(tr["forward_vel"]),  # eval_policy's per-episode number
        "distance_traveled": tr["distance_traveled"],
        "terminated": float(tr["terminated"]),
        "termination_reason": tr["termination_reason"],
    }
    # gate-style stance duties (stance_report._roll_episodes: steps >= settle_steps, derive_stance_info)
    sf = tr["stance_flags"]
    if sf.size and np.isfinite(sf[:, 0]).any() and T > gate_settle:
        row["gate_unsupported_duty"] = _mean(sf[gate_settle:, 0])
        row["gate_bilateral_support_duty"] = _mean(sf[gate_settle:, 1])
        row["gate_single_support_duty"] = _mean(sf[gate_settle:, 2])
    for key, v in tr["components"].items():
        row[f"sum_{key}"] = float(np.sum(v))
    # which reward terms are paid while NO foot is on the floor (ground truth), whole episode
    all_steps = np.arange(T)
    floor_all = contact_arrays(tr, all_steps)["floor"]
    airborne = ~floor_all.any(axis=1)
    row["floor_flight_fraction_full_episode"] = float(airborne.mean()) if T else _nan()
    for key, v in tr["components"].items():
        row[f"paid_in_floor_flight_{key}"] = float(np.sum(v[airborne]))
    if rd.env_contact_threshold is not None:
        thr = float(rd.env_contact_threshold)
        # the env's support flag (sum over feet of the per-foot MIN over substeps > threshold) evaluated
        # on the touch sensors (what was paid) and on the true floor forces (counterfactual)
        row["env_support_flag_fraction_touch_full"] = float(np.mean(tr["forces"].sum(axis=1) > thr))
        row["env_support_flag_fraction_floor_counterfactual_full"] = float(
            np.mean(tr["floor_force_min"].sum(axis=1) > thr)
        )
    row["horizontal_drift_m"] = float(np.linalg.norm(tr["root_pos"][-1, :2] - tr["root_pos"][0, :2]))
    s0 = min(settle, max(T - 1, 0))
    W = np.arange(s0, T)
    nW = W.size
    row["window_steps"] = float(nW)
    if nW < 10:
        row["note"] = "episode shorter than the settle window; gait metrics skipped"
        return row
    tw = nW * dt
    touch = tr["forces"][W]
    pos = tr["foot_pos"][1:][W]
    pos_prev = tr["foot_pos"][:-1][W]
    root = tr["root_pos"][1:][W]
    root_prev = tr["root_pos"][:-1][W]
    quat = tr["quat"][1:][W]
    qj = tr["qj"][1:][W]
    floor_mean = tr["floor_force_mean"][W]
    nonfloor = tr["nonfloor_force_mean"][W]
    srcs = contact_arrays(tr, W)

    # loads and the touch-vs-floor audit
    if rd.env_contact_threshold is not None:
        row["env_support_flag_fraction"] = float(np.mean(touch.sum(axis=1) > float(rd.env_contact_threshold)))
    row["floor_force_total_mean_bw"] = float(floor_mean.sum(axis=1).mean() / rd.body_weight_n)
    row["touch_force_total_mean_bw"] = float(touch.sum(axis=1).mean() / rd.body_weight_n)
    row["foot_foot_contact_substep_frac"] = float(tr["foot_foot_contact_frac"][W].mean())
    ftot = floor_mean.sum(axis=1)
    for f in range(F):
        lab = rd.display[f]
        row[f"touch_mean_force_bw_{lab}"] = float(touch[:, f].mean() / rd.body_weight_n)
        row[f"floor_mean_force_bw_{lab}"] = float(floor_mean[:, f].mean() / rd.body_weight_n)
        row[f"nonfloor_contact_mean_force_bw_{lab}"] = float(nonfloor[:, f].mean() / rd.body_weight_n)
        row[f"floor_load_share_{lab}"] = float(
            np.mean(np.where(ftot > 1e-9, floor_mean[:, f] / np.maximum(ftot, 1e-9), 0.0))
        )
        row[f"floor_duty_strict_allsubsteps_{lab}"] = float(
            np.mean(tr["floor_force_min"][W][:, f] > REPO_CONTACT_THRESHOLD_N)
        )
        row[f"floor_duty_loaded10bw_{lab}"] = float(
            np.mean(floor_mean[:, f] > LOADED_FRACTION_OF_BW * rd.body_weight_n)
        )
        row[f"phantom_support_fraction_{lab}"] = float(np.mean(srcs["touch"][:, f] & ~srcs["floor"][:, f]))
        row[f"touch_missed_floor_fraction_{lab}"] = float(np.mean(~srcs["touch"][:, f] & srcs["floor"][:, f]))
        row[f"touch_floor_agreement_{lab}"] = float(np.mean(srcs["touch"][:, f] == srcs["floor"][:, f]))

    # heading-frame kinematics (contact-independent)
    fwd = np.array([_fwd2d(q) for q in quat])
    lat = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
    rel = pos[:, :, :2] - root[:, None, :2]
    x_rel = np.einsum("kfi,ki->kf", rel, fwd)
    y_rel = np.einsum("kfi,ki->kf", rel, lat)
    disp = root[-1, :2] - root_prev[0, :2]
    travel_dir = disp / np.linalg.norm(disp) if np.linalg.norm(disp) > 0.01 else _mean_dir(fwd)
    row["heading_alignment_mean"] = float(np.mean(fwd @ travel_dir))
    fwd_inc = np.stack([(pos[:, f, :2] - pos_prev[:, f, :2]) @ travel_dir for f in range(F)], axis=1)
    for f in range(F):
        lab = rd.display[f]
        d = pos[:, f, :2] - pos_prev[:, f, :2]
        row[f"x_rel_mean_m_{lab}"] = float(x_rel[:, f].mean())
        row[f"x_rel_excursion_m_{lab}"] = float(np.percentile(x_rel[:, f], 95) - np.percentile(x_rel[:, f], 5))
        row[f"y_rel_mean_m_{lab}"] = float(y_rel[:, f].mean())
        row[f"forward_travel_m_{lab}"] = float(np.clip(fwd_inc[:, f], 0, None).sum())
        row[f"backward_travel_m_{lab}"] = float(np.clip(-fwd_inc[:, f], 0, None).sum())
        row[f"path_length_m_{lab}"] = float(np.linalg.norm(d, axis=1).sum())
        row[f"foot_height_median_m_{lab}"] = _median(tr["foot_h"][1:][W][:, f])
        row[f"foot_height_p95_m_{lab}"] = float(np.percentile(tr["foot_h"][1:][W][:, f], 95))
    tot = sum(row[f"forward_travel_m_{rd.display[f]}"] for f in range(F))
    for f in range(F):
        row[f"forward_travel_share_{rd.display[f]}"] = (
            row[f"forward_travel_m_{rd.display[f]}"] / tot if tot > 1e-9 else _nan()
        )
    if rd.biped:
        lat_sep = y_rel[:, rd.labels.index("l")] - y_rel[:, rd.labels.index("r")]
        row["lateral_foot_separation_mean_m"] = float(lat_sep.mean())
        row["lateral_foot_separation_min_m"] = float(lat_sep.min())
    idx = {lbl: rd.labels.index(lbl) for lbl in rd.labels}
    pairs_lr: list[tuple[str, int, int]] = []
    if rd.biped:
        pairs_lr = [("", idx["l"], idx["r"])]
    elif rd.quadruped:
        pairs_lr = [("fore_", idx["fl"], idx["fr"]), ("hind_", idx["rl"], idx["rr"])]
    for pfx, lf, rf in pairs_lr:
        lead = x_rel[:, lf] - x_rel[:, rf]
        row[f"{pfx}left_leads_fraction"] = float(np.mean(lead > 0))
        row[f"{pfx}lead_mean_m"] = float(lead.mean())
        amp = float((np.percentile(lead, 95) - np.percentile(lead, 5)) / 2)
        row[f"{pfx}lead_amplitude_m"] = amp
        row[f"{pfx}stagger_index"] = abs(float(lead.mean())) / max(amp, 1e-9)
        sgn = debounce(lead > 0)
        row[f"{pfx}lead_swaps_per_s"] = int(np.count_nonzero(np.diff(sgn.astype(np.int8)))) / tw
    # pelvis / root body
    z = root[:, 2]
    tt = np.arange(nW)
    zd = z - np.polyval(np.polyfit(tt, z, 1), tt)
    row["pelvis_z_mean_m"] = float(z.mean())
    row["pelvis_z_amplitude_m"] = float((np.percentile(zd, 95) - np.percentile(zd, 5)) / 2)
    row["pelvis_z_std_m"] = float(zd.std())
    fz, pz = _dominant_frequency(z, dt)
    row["pelvis_z_dominant_hz"] = fz
    row["pelvis_z_dominant_power_frac"] = pz
    eul = np.array([_quat_to_euler(q) for q in quat])
    row["pitch_mean_deg"] = float(np.degrees(eul[:, 1].mean()))
    row["pitch_std_deg"] = float(np.degrees(eul[:, 1].std()))
    row["roll_mean_deg"] = float(np.degrees(eul[:, 0].mean()))
    row["roll_std_deg"] = float(np.degrees(eul[:, 0].std()))
    yaw = np.unwrap(eul[:, 2])
    row["yaw_change_deg"] = float(np.degrees(yaw[-1] - yaw[0]))
    row["window_displacement_m"] = float(np.linalg.norm(disp))
    v_info = _mean(tr["forward_vel"][W])
    v_disp = float(np.linalg.norm(disp) / tw)
    v = v_info if np.isfinite(v_info) else v_disp
    row["speed_forward_vel_mps"] = v_info
    row["speed_displacement_mps"] = v_disp
    row["froude"] = v * v / (rd.gravity * rd.leg_length)
    row["speed_over_sqrt_gL"] = v / math.sqrt(rd.gravity * rd.leg_length)
    # contact-dependent blocks, one per source
    ctx = {
        "foot_h": tr["foot_h"][1:][W],
        "foot_vel": tr["foot_vel"][W],
        "fwd_inc": fwd_inc,
        "lead_prefixes": [pfx for pfx, _a, _b in pairs_lr],
        "speed": v,
    }
    for src in CONTACT_SOURCES:
        _contact_block(row, f"{src}_", srcs[src], rd, dt, ctx)
    # joint symmetry: stride period from the ground-truth contacts, else the env's touch, else FFT
    period, period_src = _nan(), "none"
    for src in ("floor", "touch"):
        med = row.get(f"{src}_stride_duration_median_s", _nan())
        if np.isfinite(med):
            period, period_src = med / dt, f"{src} stride median"
            break
    if not np.isfinite(period):
        hp = [pp for pp in rd.pairs if pp["name"].endswith("hip_pitch") and pp["kind"] == "L_vs_R"]
        if hp:
            fq, _ = _dominant_frequency(qj[:, hp[0]["a"]], dt)
            if np.isfinite(fq) and fq > 0:
                period, period_src = 1.0 / (fq * dt), "hip_pitch FFT"
    row["joint_lag_period_steps"] = float(period) if np.isfinite(period) else _nan()
    row["joint_lag_period_source"] = period_src
    for pp in rd.pairs:
        a = qj[:, pp["a"]]
        b = pp["sign_b"] * qj[:, pp["b"]]
        frac, cbest, c0 = _joint_lag(a, b, period)
        name = pp["name"]
        row[f"jointlag_frac_{name}"] = frac
        row[f"jointlag_antiphase_index_{name}"] = 2 * min(frac, 1 - frac) if np.isfinite(frac) else _nan()
        row[f"jointlag_corr_best_{name}"] = cbest
        row[f"jointlag_corr_lag0_{name}"] = c0
        row[f"joint_std_deg_a_{name}"] = float(np.degrees(a.std()))
        row[f"joint_std_deg_b_{name}"] = float(np.degrees(b.std()))
        row[f"joint_mean_deg_a_{name}"] = float(np.degrees(a.mean()))
        row[f"joint_mean_deg_b_{name}"] = float(np.degrees(b.mean()))
        fq, pw = _dominant_frequency(a, dt)
        row[f"joint_dominant_hz_a_{name}"] = fq
    return row


def _mean_dir(fwd: np.ndarray) -> np.ndarray:
    v = fwd.mean(axis=0)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def classify_biped(r: dict[str, Any], p: str = "") -> str:
    """Heuristic label from one contact source (prefix p); the JSON keeps every number it uses."""
    tps = r.get(f"{p}touchdowns_per_s_all_feet", 0.0)
    v = r.get("speed_forward_vel_mps", _nan())
    v = r.get("speed_displacement_mps", 0.0) if not np.isfinite(v) else v
    if not np.isfinite(tps) or tps < 0.2:
        return "stand" if not np.isfinite(v) or abs(v) < 0.03 else "slide (moves without stepping)"
    if np.isfinite(v) and abs(v) < 0.03 and r.get("speed_displacement_mps", 0.0) < 0.03:
        return "stand (stepping / tapping in place)"
    sim = r.get(f"{p}simultaneous_fraction", _nan())
    alt = r.get(f"{p}alternation_index", _nan())
    swaps = r.get(f"{p}lead_swaps_per_stride", _nan())
    leadf = r.get("left_leads_fraction", _nan())
    flight = r.get(f"{p}flight_fraction", 0.0)
    double = r.get(f"{p}double_support_fraction", 0.0)
    slide = r.get(f"{p}in_contact_forward_travel_fraction_mean", _nan())
    if np.isfinite(sim) and sim >= 0.5:
        return "hop" if flight >= 0.05 else "two-footed shuffle-hop (synchronous, little flight)"
    if np.isfinite(alt) and alt >= 0.7 and np.isfinite(swaps) and swaps >= 1.4:
        return "run" if flight >= 0.05 and double < 0.05 else "walk"
    if np.isfinite(leadf) and (leadf >= 0.85 or leadf <= 0.15) and (not np.isfinite(swaps) or swaps < 1.0):
        return "staggered scoot (one foot always leads; skip/gallop-like)"
    if np.isfinite(slide) and slide >= 0.5:
        return "shuffle (forward travel mostly sliding in contact)"
    return "irregular / other"


def classify_quadruped(r: dict[str, Any], p: str = "") -> str:
    rh, lf, rf = (r.get(f"{p}phase_{k}_rel_LH", _nan()) for k in ("RH", "LF", "RF"))
    rs = [r.get(f"{p}phase_R_{k}_rel_LH", _nan()) for k in ("RH", "LF", "RF")]
    duty = r.get(f"{p}duty_hind_mean", _nan())
    tps = r.get(f"{p}touchdowns_per_s_all_feet", 0.0)
    if not np.isfinite(tps) or tps < 0.2:
        v = r.get("speed_forward_vel_mps", _nan())
        v = r.get("speed_displacement_mps", 0.0) if not np.isfinite(v) else v
        return "stand" if not np.isfinite(v) or abs(v) < 0.03 else "slide (moves without stepping)"
    v = r.get("speed_forward_vel_mps", _nan())
    v = r.get("speed_displacement_mps", 0.0) if not np.isfinite(v) else v
    if np.isfinite(v) and abs(v) < 0.03 and r.get("speed_displacement_mps", 0.0) < 0.03:
        return "stand (stepping / tapping in place)"
    if not all(np.isfinite([rh, lf, rf])):
        return "undetermined (too few LH strides / touchdowns)"
    prefix = "" if min(rs) >= 0.5 else "irregular "
    tol = 0.1
    if _circ_dist(rh, 0.0) <= tol:
        if _circ_dist(lf, 0.0) <= tol and _circ_dist(rf, 0.0) <= tol:
            return prefix + "pronk"
        if _circ_dist(lf, rf) <= tol:
            return prefix + "bound"
        return prefix + "half-bound"
    if _circ_dist(rh, 0.5) <= tol:
        if _circ_dist(lf, 0.0) <= tol:
            return prefix + "pace"
        if _circ_dist(lf, 0.5) <= tol:
            return prefix + "trot"
        walkrun = "walk" if np.isfinite(duty) and duty > 0.5 else "run"
        return prefix + ("lateral-sequence " if lf < 0.5 else "diagonal-sequence ") + walkrun
    return prefix + "gallop (asymmetric)"


# ───────────────────────────── summary / sanity ─────────────────────────────


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys: list[str] = []
    for r in rows:
        for k, v in r.items():
            if k not in keys and isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                keys.append(k)
    out: dict[str, Any] = {}
    for k in keys:
        vals = np.array([float(r.get(k, np.nan)) for r in rows], dtype=float)
        fin = vals[np.isfinite(vals)]
        out[k] = {
            "mean": float(fin.mean()) if fin.size else None,
            "std": float(fin.std()) if fin.size else None,
            "min": float(fin.min()) if fin.size else None,
            "max": float(fin.max()) if fin.size else None,
            "n": int(fin.size),
        }
    for k in sorted(
        {
            k
            for r in rows
            for k in r
            if k.endswith(("gait_class", "footfall_order_mode", "termination_reason", "joint_lag_period_source"))
        }
    ):
        labels = [r[k] for r in rows if k in r]
        if labels:
            out[k + "_counts"] = dict(Counter(labels).most_common())
    return out


def _read_expected(paths: list[str]) -> dict[str, Any]:
    exp: dict[str, Any] = {}
    for p in paths or []:
        path = Path(p)
        if path.suffix == ".csv":
            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            exp.setdefault("csv", []).append({"path": str(path), "rows": rows})
        elif path.suffix == ".json":
            data = json.loads(path.read_text())
            if "stage_result" in data:
                exp["gate_verdict"] = data
            elif "episode_evidence" in data or "metrics" in data:
                exp["stance_report"] = data
            else:
                exp.setdefault("other_json", []).append(data)
    return exp


def sanity_check(
    rows: list[dict[str, Any]],
    expected: dict[str, Any],
    scheme: str,
    seed_base: int,
    panel: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    n = len(rows)
    rew = np.array([r["reward"] for r in rows])
    ln = np.array([r["length"] for r in rows])
    fv = np.array([r["mean_forward_velocity_full"] for r in rows])
    dist = np.array([r["distance_traveled"] for r in rows])
    replay = {
        "episodes": n,
        "reward_mean": float(rew.mean()),
        "reward_std": float(rew.std()),
        "length_mean": float(ln.mean()),
        "forward_vel_mean": float(fv.mean()),
        "forward_vel_std": float(fv.std()),
        "distance_traveled_mean": float(dist.mean()),
    }
    out: dict[str, Any] = {"replayed": replay, "comparisons": []}

    def cmp(name: str, recorded: float, replayed: float, spread: "float | None" = None, nrec: int = 0) -> None:
        d = replayed - recorded
        se = None
        if spread is not None and nrec:
            se = math.sqrt(spread**2 / nrec + (spread**2 / max(n, 1)))
        out["comparisons"].append(
            {"quantity": name, "recorded": recorded, "replayed": replayed, "diff": d, "z_vs_se": d / se if se else None}
        )

    for c in expected.get("csv", []):
        rr = c["rows"]
        if not rr:
            continue
        recs = np.array([float(x["reward"]) for x in rr])
        lens = np.array([float(x["length"]) for x in rr])
        vels = np.array([float(x.get("mean_forward_velocity", "nan") or "nan") for x in rr])
        dsts = np.array([float(x.get("distance_traveled", "nan") or "nan") for x in rr])
        tag = Path(c["path"]).stem
        cmp(f"{tag}.reward_mean", float(recs.mean()), replay["reward_mean"], float(recs.std()), len(recs))
        cmp(f"{tag}.reward_std", float(recs.std()), replay["reward_std"])
        cmp(f"{tag}.length_mean", float(lens.mean()), replay["length_mean"])
        cmp(
            f"{tag}.forward_vel_mean",
            float(np.nanmean(vels)),
            replay["forward_vel_mean"],
            float(np.nanstd(vels)),
            len(vels),
        )
        cmp(
            f"{tag}.distance_mean",
            float(np.nanmean(dsts)),
            replay["distance_traveled_mean"],
            float(np.nanstd(dsts)),
            len(dsts),
        )
        seeds = {x.get("evaluation_seed") for x in rr}
        m = min(len(rr), n)
        per = [
            {
                "episode": i + 1,
                "recorded_reward": float(rr[i]["reward"]),
                "replayed_reward": float(rew[i]),
                "recorded_fwd": float(vels[i]),
                "replayed_fwd": float(fv[i]),
                "recorded_len": float(lens[i]),
                "replayed_len": float(ln[i]),
            }
            for i in range(m)
        ]
        diffs = np.array([abs(p["recorded_reward"] - p["replayed_reward"]) for p in per])
        out.setdefault("per_episode", {})[tag] = {
            "evaluation_seed_recorded": sorted(s for s in seeds if s),
            "scheme_used": scheme,
            "seed_base_used": seed_base,
            "max_abs_reward_diff": float(diffs.max()) if diffs.size else None,
            "n_exact_to_1e-3": int(np.sum(diffs < 1e-3)),
            "rows": per,
        }
    gv = expected.get("gate_verdict")
    if gv:
        sr = gv["stage_result"]
        for key, rep, spread_key in (
            ("best_model_reward", replay["reward_mean"], "best_model_std_reward"),
            ("best_model_length", replay["length_mean"], "best_model_std_length"),
            ("best_model_fwd_vel", replay["forward_vel_mean"], None),
            ("best_model_distance", replay["distance_traveled_mean"], None),
        ):
            if isinstance(sr.get(key), (int, float)):
                spread = sr.get(spread_key) if spread_key else None
                cmp(
                    f"gate_verdict.{key}",
                    float(sr[key]),
                    rep,
                    float(spread) if isinstance(spread, (int, float)) else None,
                    30,
                )
        out["gate_verdict_checkpoint"] = gv.get("checkpoint")
    stn = expected.get("stance_report")
    if stn:
        met = stn.get("metrics", {})
        panel = panel or {}
        gm = {
            "reward_mean": replay["reward_mean"],
            "episode_length_mean": replay["length_mean"],
            "full_horizon_fraction": panel.get("full_horizon_fraction", _nan()),
            "mean_unsupported_duty": panel.get("mean_unsupported_duty", _nan()),
            "unsupported_duty_ucb": panel.get("unsupported_duty_ucb", _nan()),
            "bilateral_support_duty": panel.get("bilateral_support_duty", _nan()),
            "single_support_duty": panel.get("single_support_duty", _nan()),
        }
        for key, rep in gm.items():
            if isinstance(met.get(key), (int, float)):
                cmp(f"stance_report.{key}", float(met[key]), rep)
        ev = stn.get("episode_evidence") or []
        if ev:
            m = min(len(ev), n)
            per = [
                {
                    "episode": i,
                    "recorded_seed": ev[i].get("seed"),
                    "replayed_seed": rows[i].get("reset_seed"),
                    "recorded_reward": ev[i].get("reward"),
                    "replayed_reward": float(rew[i]),
                    "recorded_len": ev[i].get("length"),
                    "replayed_len": float(ln[i]),
                    "recorded_unsupported_duty": ev[i].get("unsupported_duty"),
                    "replayed_unsupported_duty": rows[i].get("gate_unsupported_duty"),
                }
                for i in range(m)
            ]
            diffs = np.array(
                [
                    abs(float(p["recorded_reward"]) - p["replayed_reward"])
                    for p in per
                    if p["recorded_reward"] is not None
                ]
            )
            out.setdefault("per_episode", {})["stance_report"] = {
                "scheme_used": scheme,
                "seed_base_used": seed_base,
                "max_abs_reward_diff": float(diffs.max()) if diffs.size else None,
                "n_exact_to_1e-3": int(np.sum(diffs < 1e-3)),
                "rows": per,
            }
    return out


def gate_style_panel(rows: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    """stance_quality/v1 panel numbers computed by the repo's own reducer (full-horizon episodes)."""
    from environments.shared.curriculum.stance_gate import stance_panel_from_episode_duties

    lengths = [r["length"] for r in rows]
    duties = [r.get("gate_unsupported_duty") for r in rows]
    if all(d is None for d in duties):
        return {}
    panel = stance_panel_from_episode_duties(
        episode_lengths=lengths, episode_duties=duties, episode_rewards=[r["reward"] for r in rows], horizon=horizon
    )
    full = [r for r in rows if r["length"] >= horizon]
    return {
        "n_episodes": panel.n_episodes,
        "full_horizon_fraction": panel.full_horizon_fraction,
        "mean_unsupported_duty": panel.mean_unsupported_duty,
        "unsupported_duty_ucb": panel.unsupported_duty_ucb,
        "bilateral_support_duty": _mean([r.get("gate_bilateral_support_duty") for r in full]),
        "single_support_duty": _mean([r.get("gate_single_support_duty") for r in full]),
    }


# ───────────────────────────── plots / video ─────────────────────────────

_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # validated categorical slots 1-4 (dataviz reference)
_INK, _INK2, _GRID, _SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(_SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(_INK2)
    ax.tick_params(colors=_INK2, labelsize=9)
    ax.grid(axis="x", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_plots(
    tr: dict[str, Any], rd: RobotDescription, dt: float, start_s: float, span_s: float, stem: Path, title: str
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = int(tr["length"])
    k0 = int(round(start_s / dt))
    k1 = min(T, k0 + int(round(span_s / dt)))
    if k1 - k0 < 10:
        k0, k1 = max(0, T - int(round(span_s / dt))), T
    ks = np.arange(k0, k1)
    t = (ks + 1) * dt
    srcs = contact_arrays(tr, ks)
    F = rd.n_feet
    paths = []
    # ---- footfall diagram: per foot, ground truth (floor) and the env's touch reading ----
    rows_spec = []
    for f in range(F):
        rows_spec.append((f"{rd.display[f]} floor (truth)", srcs["floor"][:, f], _SERIES[f % 4], 1.0))
        rows_spec.append((f"{rd.display[f]} touch (env)", srcs["touch"][:, f], _SERIES[f % 4], 0.35))
    ff = tr["foot_foot_contact_frac"][ks] > 0
    rows_spec.append(("feet touching each other", ff, _INK2, 1.0))
    nrow = len(rows_spec)
    fig, ax = plt.subplots(figsize=(10, 1.3 + 0.42 * (nrow + 1)), dpi=130)
    fig.patch.set_facecolor(_SURF)
    _style_axes(ax)
    for i, (label, arr, color, alpha) in enumerate(rows_spec):
        y = nrow - i
        for value, s, n in _runs(arr):
            if value:
                ax.broken_barh(
                    [(t[s] - dt / 2, n * dt)],
                    (y - 0.34, 0.68),
                    facecolors=color,
                    alpha=alpha,
                    edgecolor=_SURF,
                    linewidth=1,
                )
        if "floor" in label:
            tdk, _ = _events(debounce(arr))
            ax.plot(
                t[tdk] - dt / 2, np.full(tdk.size, y + 0.46), marker="v", linestyle="none", color=_INK, markersize=4
            )
    n_down = srcs["floor"].sum(axis=1)
    ax.step(t, n_down / max(F, 1) * 0.7 - 0.35, where="mid", color=_INK, linewidth=1.2)
    ax.set_yticks(list(range(nrow, 0, -1)) + [0])
    ax.set_yticklabels([r[0] for r in rows_spec] + [f"feet on floor (0..{F})"], color=_INK, fontsize=9)
    ax.set_ylim(-0.6, nrow + 0.8)
    ax.set_xlim(t[0] - dt, t[-1] + dt)
    ax.set_xlabel("time (s)", color=_INK2)
    ax.set_title(
        f"{title}\nfloor (truth) = floor normal force on the foot > {REPO_CONTACT_THRESHOLD_N} N on >= half the step's substeps\n"
        f"touch (env) = the env's touch-sensor reading (min over substeps) > {REPO_CONTACT_THRESHOLD_N} N;  ▼ = debounced floor touchdown",
        color=_INK,
        fontsize=9,
        loc="left",
    )
    fig.tight_layout()
    p = stem.with_name(stem.name + "_footfall.png")
    fig.savefig(p, facecolor=_SURF)
    plt.close(fig)
    paths.append(str(p))
    # ---- fore-aft positions in the heading frame ----
    quat = tr["quat"][1:][ks]
    fwd = np.array([_fwd2d(q) for q in quat])
    rel = tr["foot_pos"][1:][ks][:, :, :2] - tr["root_pos"][1:][ks][:, None, :2]
    x_rel = np.einsum("kfi,ki->kf", rel, fwd)
    pairs = []
    if rd.biped:
        pairs = [("L - R", rd.labels.index("l"), rd.labels.index("r"))]
    elif rd.quadruped:
        pairs = [
            ("LF - RF", rd.labels.index("fl"), rd.labels.index("fr")),
            ("LH - RH", rd.labels.index("rl"), rd.labels.index("rr")),
        ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.6), dpi=130, sharex=True, gridspec_kw={"height_ratios": [2.2, 1, 1]})
    fig.patch.set_facecolor(_SURF)
    ax = axes[0]
    _style_axes(ax)
    ax.grid(axis="y", color=_GRID, linewidth=0.8)
    for f in range(F):
        ax.plot(t, x_rel[:, f] * 100, color=_SERIES[f % 4], linewidth=2, label=f"{rd.display[f]} foot")
        xs = np.where(srcs["floor"][:, f], x_rel[:, f] * 100, np.nan)
        ax.plot(t, xs, color=_SERIES[f % 4], linewidth=6, alpha=0.35, solid_capstyle="butt")
        ax.annotate(
            rd.display[f],
            (t[-1], x_rel[-1, f] * 100),
            xytext=(4, 0),
            textcoords="offset points",
            color=_INK,
            fontsize=9,
            va="center",
        )
    ax.axhline(0, color=_INK2, linewidth=0.8)
    ax.set_ylabel(f"fore-aft rel. {rd.root_name} (cm)\n+ = ahead", color=_INK2)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=F)
    ax.set_title(
        f"{title}\nfoot touch-site position along the heading (thick = on the floor)",
        color=_INK,
        fontsize=10,
        loc="left",
    )
    ax2 = axes[1]
    _style_axes(ax2)
    ax2.grid(axis="y", color=_GRID, linewidth=0.8)
    for i, (name, a, b) in enumerate(pairs):
        ax2.plot(
            t,
            (x_rel[:, a] - x_rel[:, b]) * 100,
            color=_INK if i == 0 else _INK2,
            linewidth=2,
            linestyle="-" if i == 0 else "--",
            label=f"lead {name}",
        )
    ax2.axhline(0, color=_INK2, linewidth=0.8)
    ax2.set_ylabel("lead (cm)\n+ = left ahead", color=_INK2)
    ax2.legend(loc="upper left", frameon=False, fontsize=9)
    ax3 = axes[2]
    _style_axes(ax3)
    ax3.grid(axis="y", color=_GRID, linewidth=0.8)
    fh = tr["foot_h"][1:][ks]
    for f in range(F):
        ax3.plot(t, fh[:, f] * 1000, color=_SERIES[f % 4], linewidth=2, label=f"{rd.display[f]} foot")
    ax3.set_ylabel("lowest touch-site\nheight (mm)", color=_INK2)
    ax3.set_xlabel("time (s)", color=_INK2)
    ax3.legend(loc="upper left", frameon=False, fontsize=9, ncol=F)
    fig.tight_layout()
    p = stem.with_name(stem.name + "_foreaft.png")
    fig.savefig(p, facecolor=_SURF)
    plt.close(fig)
    paths.append(str(p))
    return paths


def contact_sheet(
    video: str,
    out_png: Path,
    start_s: float,
    span_s: float,
    n_frames: int,
    cols: int,
    crop: "str | None",
    frame_dt: "float | None" = None,
) -> dict[str, Any]:
    import cv2

    cap = cv2.VideoCapture(video)
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 50.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # evaluation.record_stage_video writes ONE frame per control step but always encodes at fps=50
    # (mediapy.write_video(..., fps=50)), so 1/fps is the sim time per frame only when env.dt == 0.02; for the
    # 0.01 s species (trex, velociraptor, dibothrosuchus, brachiosaurus) that mapping is off by 2x.
    # --video-frame-dt gives the sim time per frame explicitly; the default keeps the original 1/fps behaviour.
    fdt = float(frame_dt) if frame_dt else 1.0 / fps
    idxs = np.linspace(start_s / fdt, (start_s + span_s) / fdt, n_frames).round().astype(int)
    idxs = np.clip(idxs, 0, max(total - 1, 0))
    tiles = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        if crop:
            x0, y0, x1, y1 = (float(v) for v in crop.split(","))
            h, w = frame.shape[:2]
            frame = frame[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]
        frame = frame.copy()
        # repo replays (evaluation.record_stage_video) append one frame AFTER each env.step, so frame i shows
        # the state after control step i+1, i.e. sim time (i+1)/fps -- the same clock as the probe's plots
        label = f"t={(i + 1) * fdt:5.2f}s f{i}"
        (tw_, th_), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (4, 4), (12 + tw_, 12 + th_), (0, 0, 0), -1)
        cv2.putText(frame, label, (8, 8 + th_), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(frame)
    cap.release()
    if not tiles:
        raise RuntimeError(f"could not read frames from {video}")
    h, w = tiles[0].shape[:2]
    rows = int(math.ceil(len(tiles) / cols))
    sheet = np.full((rows * h, cols * w, 3), 255, dtype=np.uint8)
    for j, tile in enumerate(tiles):
        r, c = divmod(j, cols)
        sheet[r * h : (r + 1) * h, c * w : (c + 1) * w] = tile
    cv2.imwrite(str(out_png), sheet)
    return {
        "video": video,
        "fps": fps,
        "frame_dt_s": fdt,
        "frames_total": total,
        "frame_indices": idxs.tolist(),
        "png": str(out_png),
    }


# ───────────────────────────── main ─────────────────────────────


def _pick_representative(rows: list[dict[str, Any]], spec: str) -> int:
    if spec != "median":
        return int(spec)
    rew = np.array([r["reward"] for r in rows])
    return int(np.argmin(np.abs(rew - np.median(rew))))


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--species")
    ap.add_argument("--stage", help="semantic id (stance/recovery/locomotion/behavior) or legacy number")
    ap.add_argument("--checkpoint", help="SB3 .zip")
    ap.add_argument("--vecnorm", help="VecNormalize .pkl paired with the checkpoint")
    ap.add_argument("--episodes", type=int, default=None, help="default: max(20, stage min_eval_episodes)")
    ap.add_argument("--seed-base", type=int, default=None, help="default: constants.PUBLICATION_SEED_START (3042)")
    ap.add_argument("--seed-scheme", choices=("auto", "panel", "vecenv"), default="auto")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--run-id", default="run")
    ap.add_argument("--settle-steps", type=int, default=None)
    ap.add_argument("--trace-episode", default="median")
    ap.add_argument("--plot-start", type=float, default=6.0)
    ap.add_argument("--plot-span", type=float, default=3.0)
    ap.add_argument(
        "--expected", nargs="*", default=[], help="evaluation_*.csv / gate_verdict.json / stance_gate_report.json"
    )
    ap.add_argument("--stage-config-json", default=None, help="the node's recorded stage_config.json (drift check)")
    ap.add_argument("--use-recorded-env-kwargs", action="store_true", help="roll with stage_config.json's env kwargs")
    ap.add_argument("--allow-legacy-plant", action="store_true")
    ap.add_argument("--video", default=None)
    ap.add_argument("--video-start", type=float, default=6.0)
    ap.add_argument("--video-span", type=float, default=2.0)
    ap.add_argument("--video-frames", type=int, default=12)
    ap.add_argument("--video-cols", type=int, default=4)
    ap.add_argument("--video-crop", default=None, help="x0,y0,x1,y1 fractions of the frame")
    ap.add_argument(
        "--video-frame-dt",
        type=float,
        default=None,
        help="sim seconds per video frame (repo replays: env.dt; default 1/container fps)",
    )
    ap.add_argument("--tag", default=None, help="extra suffix for output names (e.g. a second panel)")
    ap.add_argument("--zero-action", action="store_true", help="roll the zero action (home pose) instead of a policy")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    species_tag = args.species or "video"
    stage_tag = str(args.stage) if args.stage is not None else "na"
    stem = out_dir / f"{species_tag}_{args.run_id}_{stage_tag}{('_' + args.tag) if args.tag else ''}"

    video_info = None
    if args.video:
        video_info = contact_sheet(
            args.video,
            stem.with_name(stem.name + "_contact_sheet.png"),
            args.video_start,
            args.video_span,
            args.video_frames,
            args.video_cols,
            args.video_crop,
            args.video_frame_dt,
        )
        print(f"contact sheet: {video_info['png']} (frames {video_info['frame_indices']})")
    if not args.checkpoint and not args.zero_action:
        if video_info:
            return 0
        ap.error("--checkpoint (or --zero-action) is required unless only --video is given")
    if not (args.species and args.stage):
        ap.error("--species and --stage are required with --checkpoint")

    import environments

    wt = os.environ.get("PYTHONPATH", "").split(os.pathsep)[0]
    env_file = str(Path(environments.__file__).resolve())
    if wt and not env_file.startswith(str(Path(wt).resolve())):
        raise SystemExit(f"environments imported from {env_file}, not from PYTHONPATH worktree {wt}")

    from environments.shared.config import load_stage_config
    from environments.shared.constants import PUBLICATION_SEED_START
    from environments.shared.curriculum.stance_gate import StanceGateThresholds
    from environments.shared.plant_contract import current_plant_identity, validate_environment_plant
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import SPECIES_FACTORIES
    from environments.shared.stage_manifest import load_stage_manifest

    species = args.species
    stage: "int | str" = int(args.stage) if str(args.stage).isdigit() else str(args.stage)
    entry = load_stage_manifest(species).resolve(stage)
    stage_cfg = load_stage_config(species, stage)
    env_kwargs = dict(stage_cfg["env_kwargs"])
    curriculum = stage_cfg.get("curriculum_kwargs", {})
    gate_kind = curriculum.get("gate_kind")
    drift: dict[str, Any] = {}
    if args.stage_config_json:
        import inspect

        rec = json.loads(Path(args.stage_config_json).read_text()).get("reward_weights", {})
        env_cls_for_defaults = SPECIES_FACTORIES[species]().env_class
        defaults = {
            name: prm.default
            for name, prm in inspect.signature(env_cls_for_defaults.__init__).parameters.items()
            if prm.default is not inspect.Parameter.empty
        }
        effective = {**defaults, **env_kwargs}  # what the constructor actually receives/uses
        for k in sorted(set(rec) | set(env_kwargs)):
            a = effective.get(k, "<absent>")
            b = rec.get(k, "<absent>")
            a_cmp = list(a) if isinstance(a, tuple) else a
            if a_cmp != b:
                drift[k] = {"toml_main": a_cmp, "recorded_run": b}
        if drift:
            print(f"WARNING: env kwargs drifted between main's TOML and the run's stage_config.json: {drift}")
        if args.use_recorded_env_kwargs:
            env_kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in rec.items()}
    thresholds = StanceGateThresholds.from_curriculum(curriculum, require_criteria=False)
    gate_settle = int(thresholds.settle_steps)
    scheme = args.seed_scheme
    if scheme == "auto":
        scheme = "panel" if gate_kind in ("stance_quality/v1", "recovery_quality/v1") else "vecenv"
    seed_base = PUBLICATION_SEED_START if args.seed_base is None else args.seed_base
    n_episodes = args.episodes or max(20, int(curriculum.get("min_eval_episodes", 20)))
    horizon = int(env_kwargs.get("max_episode_steps", 1000))

    env_class = SPECIES_FACTORIES[species]().env_class
    plant = current_plant_identity(species)
    from environments.shared.plant_contract import PlantContractError
    from environments.shared.policy_loading import PolicyLoadError

    vec_path = None
    if args.zero_action:
        _probe_env = env_class(**env_kwargs)
        _zero = np.zeros(_probe_env.action_space.shape[0], dtype=np.float32)
        _probe_env.close()

        def predict(_obs: np.ndarray) -> np.ndarray:
            return _zero

    try:
        if args.zero_action:
            raise StopIteration
        model, normalizer, vec_path = load_sb3_checkpoint(
            args.checkpoint,
            args.vecnorm,
            lambda: env_class(**env_kwargs),
            guess_sidecar=args.vecnorm is None,
            allow_unnormalized=False,
            plant_identity=plant,
            allow_legacy_plant=args.allow_legacy_plant,
        )
    except StopIteration:
        pass
    except (PlantContractError, PolicyLoadError) as exc:
        print(f"CHECKPOINT REFUSED by the plant contract / loader (not forced): {type(exc).__name__}: {exc}")
        (stem.with_name(stem.name + "_REFUSED.txt")).write_text(str(exc) + "\n")
        return 3
    else:
        obs_rms = normalizer.obs_rms
        if isinstance(obs_rms, dict):
            raise SystemExit(f"{vec_path} holds per-key statistics for a Dict observation space; refusing")
        mean = np.asarray(obs_rms.mean, dtype=np.float64)
        var = np.asarray(obs_rms.var, dtype=np.float64)
        eps = float(normalizer.epsilon)
        clip = float(normalizer.clip_obs)

        def predict(obs: np.ndarray) -> np.ndarray:
            # VecNormalize.normalize_obs (eval mode) then deterministic predict
            normalized = np.clip((obs - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)
            return np.asarray(model.predict(normalized, deterministic=True)[0])

    if scheme == "vecenv":
        from stable_baselines3.common.utils import set_random_seed

        set_random_seed(seed_base)  # train_base.make_env: set_random_seed(seed + rank)
    env = env_class(**env_kwargs)
    validate_environment_plant(env, plant, artifact=f"{species} {entry.id} gait probe environment")
    rd = RobotDescription(env)
    dt = float(env.unwrapped.dt)
    settle = (
        args.settle_steps
        if args.settle_steps is not None
        else (gate_settle if gate_settle > 0 else int(round(1.0 / dt)))
    )
    if scheme == "vecenv":
        env.reset(seed=seed_base)  # make_env's construction-time reset
    print(
        f"{species} {entry.id} gate={gate_kind} scheme={scheme} seed_base={seed_base} episodes={n_episodes} "
        f"dt={dt} settle={settle} feet={rd.display} root={rd.root_name} L={rd.leg_length:.4f} m"
    )
    rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    t_start = time.time()
    for i in range(n_episodes):
        reset_seed = seed_base + i if scheme == "panel" else None
        tr = rollout_episode(env, predict, rd, reset_seed)
        if scheme == "vecenv":
            env.reset()  # DummyVecEnv auto-reset after done; eval_policy then resets again
        row = analyze_episode(tr, rd, dt, settle, gate_settle)
        row = {"episode": i, "reset_seed": reset_seed if reset_seed is not None else f"vecenv:{seed_base}#{i}", **row}
        rows.append(row)
        traces.append(tr)
        print(
            f"  ep {i:2d}: R={row['reward']:.2f} len={int(row['length'])} v={row['mean_forward_velocity_full']:.4f} "
            f"touch-class={row.get('touch_gait_class', '?')} | floor-class={row.get('floor_gait_class', '?')} "
            f"({time.time() - t_start:.0f}s)",
            flush=True,
        )
    env.close()
    summary = summarize(rows)
    expected = _read_expected(args.expected)
    panel = gate_style_panel(rows, horizon)
    sanity = sanity_check(rows, expected, scheme, seed_base, panel)
    rep = _pick_representative(rows, args.trace_episode)
    tr = traces[rep]
    title = f"{species} {entry.id} run {args.run_id} ep{rep} ({scheme} seed {seed_base})"
    plots = make_plots(tr, rd, dt, args.plot_start, args.plot_span, stem, title)
    npz_path = stem.with_name(stem.name + f"_ep{rep}_trace.npz")
    np.savez_compressed(
        npz_path,
        dt=dt,
        foot_labels=np.array(rd.display),
        joint_names=np.array(rd.hinge_names),
        contact_threshold_n=REPO_CONTACT_THRESHOLD_N,
        forces_aggregated=tr["forces"],
        floor_force_min=tr["floor_force_min"],
        floor_force_mean=tr["floor_force_mean"],
        floor_contact_substep_frac=tr["floor_contact_substep_frac"],
        nonfloor_force_mean=tr["nonfloor_force_mean"],
        foot_foot_contact_frac=tr["foot_foot_contact_frac"],
        forces_last_substep=tr["forces_inst"],
        foot_site_pos=tr["foot_pos"],
        foot_height=tr["foot_h"],
        foot_site_vel=tr["foot_vel"],
        root_pos=tr["root_pos"],
        quat=tr["quat"],
        root_linvel=tr["root_vel"],
        joint_qpos=tr["qj"],
        reward=tr["reward"],
        forward_vel=tr["forward_vel"],
        stance_flags_unsup_bilat_single=tr["stance_flags"],
        **{f"component_{k}": v for k, v in tr["components"].items()},
    )
    import hashlib

    def sha(p: "str | None") -> "str | None":
        return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest() if p else None

    doc = {
        "schema": PROBE_VERSION,
        "meta": {
            "species": species,
            "stage": entry.id,
            "stage_arg": str(args.stage),
            "run_id": args.run_id,
            "gate_kind": gate_kind,
            "checkpoint": "ZERO ACTION (do-nothing reference)" if args.zero_action else str(args.checkpoint),
            "checkpoint_sha256": None if args.zero_action else sha(args.checkpoint),
            "vecnorm": vec_path,
            "vecnorm_sha256": sha(vec_path),
            "plant_identity_policy_interface_revision": getattr(plant, "policy_interface_revision", None),
            "seed_scheme": scheme,
            "seed_base": seed_base,
            "episodes": n_episodes,
            "dt": dt,
            "settle_steps_gait_window": settle,
            "settle_steps_gate": gate_settle,
            "contact_threshold_n": REPO_CONTACT_THRESHOLD_N,
            "loaded_threshold_n": LOADED_FRACTION_OF_BW * rd.body_weight_n,
            "debounce_min_run_steps": MIN_RUN_STEPS,
            "touchdown_merge_steps": MERGE_STEPS,
            "env_kwargs": env_kwargs,
            "env_kwargs_drift_vs_recorded": drift,
            "robot": rd.describe(),
            "environments_file": env_file,
            "representative_episode": rep,
            "elapsed_s": time.time() - t_start,
        },
        "sanity": sanity,
        "gate_style_stance_panel": panel,
        "summary": summary,
        "episodes": rows,
        "files": {"trace_npz": str(npz_path), "plots": plots, "contact_sheet": video_info},
    }
    json_path = stem.with_name(stem.name + ".json")
    json_path.write_text(json.dumps(_json_safe(doc), indent=1))
    print(f"wrote {json_path}\n      {npz_path}\n      {plots}")
    for c in sanity["comparisons"]:
        print(
            f"  sanity {c['quantity']}: recorded={c['recorded']:.4f} replayed={c['replayed']:.4f} diff={c['diff']:+.4f}"
            + (f" z={c['z_vs_se']:+.2f}" if c["z_vs_se"] is not None else "")
        )
    for tag, pe in sanity.get("per_episode", {}).items():
        print(
            f"  per-episode {tag}: max |dR|={pe['max_abs_reward_diff']}, exact(<1e-3)={pe['n_exact_to_1e-3']}/{len(pe['rows'])}"
        )
    if panel:
        print(f"  gate-style stance panel: {panel}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
