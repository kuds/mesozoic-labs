"""Build the WALK-label test cases for test_pilot_score.py with the frozen gait probe's OWN analysis code.

Scratch, never merged.  Writes tools/fixtures/gait_cases.json with two kinds of case:

synthetic  Deterministic two-legged traces fed through ``analyze_episode`` of the frozen probe
           (docs/investigations/gait_2026_09/gait_probe.py, loaded from --tree, never modified), so every metric,
           the touchdown merge, the kinematic lead swaps, the hip-pitch joint lag and ``floor_gait_class`` are the
           probe's own numbers.  dt 0.01 s (the T. rex control step), 1,000 steps, gait window from step 100.
             staggered_hop_30ms  the verifier's case: an 8.3 Hz two-footed hop (cycle 12 steps), right foot landing
                                 3 steps (30 ms) after the left, each foot down 8 of 12 steps (flight 1 of 12), legs
                                 swinging with the same 30 ms offset, feet level on average (a mirror-symmetric
                                 stance's start), 0.3 m/s.
             staggered_hop_40ms  the same with a 4-step (40 ms) offset.
             staggered_hop_50ms  the same with a 5-step (50 ms) offset: touchdown gaps 50 and 70 ms, near anti-phase.
             synchronous_hop     the same with no offset (the September hop's pattern).
             alternating_walk    a 1 Hz walk (cycle 100 steps), each foot down 60 of 100, right half a cycle after
                                 the left (double support 0.2, no flight), hips in anti-phase, 0.3 m/s.
           The second set ("ct_" names) uses a continuous-time schedule (trace_ct, copied from the round-2 verifier's
           own generator, vfixpilot/my_gait_cases.py, so the hop frequencies need not be whole numbers of steps):
           left foot down at t = n/f, right foot td_offset later, each foot down duty x P, hips oscillating at f with
           the right hip lagging by hip_offset (default: the touchdown offset).  It holds the verifier's grid
           (staggered hops at 7.7, 8.3, 9.2 and 10 Hz with 20-50 ms offsets, hips-in-phase and fixed-lead variants, the
           8.3 Hz scissor hop, slow and limping walks, a 3 Hz run) and the limits of the final rule (a scissor hop and
           staggered hops at 4.5 Hz, an alternating shuffle at 5.5 Hz, a 4 Hz walk).
           The third set ("ev_" names, this round) uses trace_ev, the round-3 verifier's event-based generator, copied
           unchanged from vfixpilot2's my_gait_events.py (sha256 ecbd21ce...; there called episode()): touchdown event
           lists with per-cycle jitter, contact sampled on 5 substeps per control step, a skewed hip waveform, the
           right hip following its own foot or locked to the left foot's phase.  It holds the short-lift hops that
           read WALK under the round-2 rule (feet lifting for 1-2 control steps, so the probe's 2-step debounce fills
           a lift and the stride frequency reads far below the true 6-16 Hz cadence; the duty guard rejects them),
           the verifier's quoted 10 Hz case, a 16 Hz chatter at duty 0.80, and creeping walks either side of the duty
           guard.  "ct_shortlift_"/"ct_chatter_" rows are the same regime through trace_ct.
real       The audit's genuine biped walker, compsognathus 20260921_203149 locomotion, final checkpoint, re-run with
           the frozen probe at the audit commit 7ae0a19 (the audit README's recipe: --seed-scheme auto, 30 episodes,
           seed base 3042) from the checkpoint pair read from Drive; the per-episode rows are copied (the keys the
           scorer reads plus the probe's class and joint-lag numbers).  Its means reproduce the audit CSV's
           "final" row.

    python3 make_gait_cases.py --tree <pilot tree> --real-json <probe JSON of the real walker> --real-meta <json>
            --out fixtures/gait_cases.json
    python3 -B make_gait_cases.py --tree <pilot tree> --keep-real fixtures/gait_cases.json --out <new fixture>
            (round 2: recompute the synthetic rows, copy the real walker's rows from the existing fixture)
    python3 -B make_gait_cases.py --tree <pilot tree> --keep-real fixtures/gait_cases.json
            --real-json <the real walker's probe JSON> --out <new fixture>
            (final round: also re-extract the real walker's rows with the current KEYS from the probe JSON the
            provenance names, refused unless its sha256 equals the provenance's probe_json_sha256)
Needs numpy (the probe's only module-level dependency); imports nothing from the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

DT = 0.01
T = 1000
SETTLE = 100
KEYS = ("length", "speed_displacement_mps", "speed_forward_vel_mps", "floor_touchdowns_per_s_all_feet",
        "floor_alternation_index", "floor_simultaneous_fraction", "floor_lead_swaps_per_stride",
        "floor_flight_fraction", "floor_double_support_fraction", "floor_single_support_fraction",
        "floor_stride_frequency_hz", "floor_stride_length_over_L", "left_leads_fraction", "froude",
        "floor_gait_class", "touch_gait_class", "jointlag_frac_hip_pitch", "jointlag_corr_best_hip_pitch",
        "joint_lag_period_steps", "joint_lag_period_source", "floor_footfall_sequence_head", "floor_duty_L",
        "floor_duty_R", "floor_swing_mean_s_L", "floor_swing_mean_s_R")


def load_probe(tree: Path):
    sys.dont_write_bytecode = True  # never leave a __pycache__ in the tree (python -I ignores PYTHONDONTWRITEBYTECODE)
    path = tree / "docs" / "investigations" / "gait_2026_09" / "gait_probe.py"
    spec = importlib.util.spec_from_file_location("gait_probe_frozen", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod, path


def robot() -> SimpleNamespace:
    # T. rex-like description: feet R then L (the probe's order on the T. rex), hip_pitch pair, 85.7 kg, L 0.8766 m.
    return SimpleNamespace(n_feet=2, labels=["r", "l"], display=["R", "L"], biped=True, quadruped=False,
                           env_contact_threshold=0.1, body_weight_n=840.9132, gravity=9.81, leg_length=0.8766,
                           pairs=[{"name": "hip_pitch", "a": 0, "b": 1, "sign_b": 1.0, "kind": "L_vs_R"}])


def trace(period: int, down: int, offset: int, speed: float = 0.3, swing_amp: float = 0.12,
          hip_amp: float = 0.3) -> dict:
    """Left foot down for steps [0, down) of each cycle; right foot the same, ``offset`` steps later."""
    k = np.arange(T + 1)
    t_ctrl = np.arange(T)

    def is_down(steps, off):
        return ((steps - off) % period) < down

    # index 0 = right foot, 1 = left foot (labels ["r", "l"])
    down_ctrl = np.stack([is_down(t_ctrl, offset), is_down(t_ctrl, 0)], axis=1)
    down_state = np.stack([is_down(k, offset), is_down(k, 0)], axis=1)
    phase_l = 2 * math.pi * k / period
    phase_r = 2 * math.pi * (k - offset) / period
    root = np.stack([speed * DT * k, np.zeros_like(k, dtype=float), 0.9 + 0.01 * np.sin(2 * phase_l)], axis=1)
    x_rel = np.stack([swing_amp * np.sin(phase_r), swing_amp * np.sin(phase_l)], axis=1)  # level feet on average
    h = np.where(down_state, 0.0, 0.05)
    foot_pos = np.zeros((T + 1, 2, 3))
    foot_pos[:, :, 0] = root[:, None, 0] + x_rel
    foot_pos[:, 0, 1], foot_pos[:, 1, 1] = -0.2, 0.2
    foot_pos[:, :, 2] = h
    foot_vel = np.zeros((T, 2, 3))
    foot_vel[:, :, :2] = (foot_pos[1:, :, :2] - foot_pos[:-1, :, :2]) / DT
    qj = np.zeros((T + 1, 2))
    qj[:, 0], qj[:, 1] = hip_amp * np.sin(phase_l), hip_amp * np.sin(phase_r)  # a = l_hip_pitch, b = r_hip_pitch
    force = np.where(down_ctrl, 400.0, 0.0)
    quat = np.tile([1.0, 0.0, 0.0, 0.0], (T + 1, 1))
    return {"length": T, "total_reward": 0.0, "forward_vel": np.full(T, speed), "distance_traveled": speed * T * DT,
            "terminated": False, "termination_reason": "truncated", "stance_flags": np.full((T, 3), np.nan),
            "components": {}, "floor_contact_substep_frac": down_ctrl.astype(float), "forces": force,
            "floor_force_min": force, "floor_force_mean": force, "nonfloor_force_mean": np.zeros((T, 2)),
            "foot_foot_contact_frac": np.zeros(T), "foot_pos": foot_pos, "root_pos": root, "quat": quat, "qj": qj,
            "foot_h": h, "foot_vel": foot_vel}


def trace_ct(freq_hz: float, duty_l: float, duty_r: float, td_offset_s: float, hip_offset_s: float | None = None,
             speed: float = 0.25, swing: float = 0.15, hip_amp: float = 0.25, lead_bias: float = 0.0,
             noise: float = 0.0, seed: int = 0, hip_harm2: float = 0.0) -> dict:
    """Continuous-time schedule (the round-2 verifier's generator): left foot touches down at t = n/f, right at
    t = n/f + td_offset_s; hips oscillate at f, the right hip lagging the left by hip_offset_s (default: the touchdown
    offset); lead_bias moves the left foot forward (m); hip_harm2 adds a 2f pelvis term common to both hips."""
    rng = np.random.default_rng(seed)
    P = 1.0 / freq_hz
    hip_offset_s = td_offset_s if hip_offset_s is None else hip_offset_s
    t_state = np.arange(T + 1) * DT
    t_ctrl = np.arange(T) * DT

    def is_down(t, off, duty):
        return ((t - off) % P) < duty * P

    dc = np.stack([is_down(t_ctrl, td_offset_s, duty_r), is_down(t_ctrl, 0.0, duty_l)], axis=1)
    ds = np.stack([is_down(t_state, td_offset_s, duty_r), is_down(t_state, 0.0, duty_l)], axis=1)
    ph_l = 2 * math.pi * t_state / P
    ph_r = 2 * math.pi * (t_state - hip_offset_s) / P
    root = np.stack([speed * t_state, np.zeros_like(t_state), 0.9 + 0.01 * np.sin(2 * ph_l)], axis=1)
    x_rel = np.stack([swing * np.sin(ph_r), swing * np.sin(ph_l) + lead_bias], axis=1)
    h = np.where(ds, 0.0, 0.04)
    fp = np.zeros((T + 1, 2, 3))
    fp[:, :, 0] = root[:, None, 0] + x_rel
    fp[:, 0, 1], fp[:, 1, 1] = -0.2, 0.2
    fp[:, :, 2] = h
    fv = np.zeros((T, 2, 3))
    fv[:, :, :2] = (fp[1:, :, :2] - fp[:-1, :, :2]) / DT
    qj = np.zeros((T + 1, 2))
    qj[:, 0] = hip_amp * np.sin(ph_l) + hip_harm2 * np.sin(2 * ph_l)
    qj[:, 1] = hip_amp * np.sin(ph_r) + hip_harm2 * np.sin(2 * ph_l)
    if noise:
        qj += rng.normal(0, noise, qj.shape)
    force = np.where(dc, 420.0, 0.0)
    return {"length": T, "total_reward": 0.0, "forward_vel": np.full(T, speed), "distance_traveled": speed * T * DT,
            "terminated": False, "termination_reason": "truncated", "stance_flags": np.full((T, 3), np.nan),
            "components": {}, "floor_contact_substep_frac": dc.astype(float), "forces": force,
            "floor_force_min": force, "floor_force_mean": force, "nonfloor_force_mean": np.zeros((T, 2)),
            "foot_foot_contact_frac": np.zeros(T), "foot_pos": fp, "root_pos": root,
            "quat": np.tile([1.0, 0, 0, 0], (T + 1, 1)), "qj": qj, "foot_h": h, "foot_vel": fv}


# ---- trace_ev: the round-3 verifier's event-based generator (vfix2pilot/my_gait_events.py, sha256
# ecbd21ce48025fc9e977e0e47d26fc36e5f6d73ee8a39421f1a23f13d5e2ac28), wave / phase_from_events / episode copied
# unchanged; only episode() is renamed trace_ev and its module constants spelled out (DT, T, SUB = 0.01, 1000, 5).
SUB = 5


def wave(ph: np.ndarray, duty: float) -> np.ndarray:
    """Skewed hip waveform on cycle phase in [0,1): +1 at touchdown, linear to -1 at the end of stance (duty),
    cosine-eased back to +1 over the swing."""
    ph = np.mod(ph, 1.0)
    out = np.empty_like(ph)
    st = ph < duty
    out[st] = 1.0 - 2.0 * ph[st] / duty
    s = (ph[~st] - duty) / (1.0 - duty)
    out[~st] = -np.cos(math.pi * s)
    return out


def phase_from_events(t: np.ndarray, td: np.ndarray) -> np.ndarray:
    """Continuous cycle phase (cycles, unwrapped) from a touchdown time list."""
    n = np.searchsorted(td, t, side="right") - 1
    n = np.clip(n, 0, td.size - 2)
    return n + (t - td[n]) / (td[n + 1] - td[n])


def trace_ev(freq, td_offset_s, hip_phase="follow", duty=0.7, *, speed=0.3, period_jitter=0.03, offset_jitter_s=0.003,
             hip_amp=0.3, lead_bias=0.0, noise=0.0, retouch=None, seed=0, duty_r=None):
    rng = np.random.default_rng(seed)
    P = 1.0 / freq
    n_cyc = int(T * DT * freq) + 6
    periods = P * (1.0 + period_jitter * rng.standard_normal(n_cyc))
    periods = np.clip(periods, 0.5 * P, 1.5 * P)
    t0 = -P * rng.uniform(1.0, 2.0)
    td_l = t0 + np.concatenate([[0.0], np.cumsum(periods)])
    td_r = td_l + td_offset_s + offset_jitter_s * rng.standard_normal(td_l.size)
    duty_r = duty if duty_r is None else duty_r
    per = np.diff(td_l)
    iv = {1: [(a, a + duty * p) for a, p in zip(td_l[:-1], per)],
          0: [(a, a + duty_r * p) for a, p in zip(td_r[:-1], per)]}
    if retouch is not None:  # a short lift of `retouch` seconds in the middle of every stance
        for f in (0, 1):
            new = []
            for a, b in iv[f]:
                m = 0.5 * (a + b)
                new += [(a, m - retouch / 2), (m + retouch / 2, b)]
            iv[f] = new
    t_sub = (np.arange(T)[:, None] + (np.arange(SUB)[None, :] + 0.5) / SUB) * DT  # (T, SUB)
    sub_down = np.zeros((T, SUB, 2), dtype=bool)
    for f in (0, 1):
        for a, b in iv[f]:
            sub_down[:, :, f] |= (t_sub >= a) & (t_sub < b)
    frac = sub_down.mean(axis=1)
    allsub = sub_down.all(axis=1)
    t_state = np.arange(T + 1) * DT
    ph_l = phase_from_events(t_state, td_l)
    if hip_phase == "follow":
        ph_r = phase_from_events(t_state, td_r)
    else:
        ph_r = ph_l - float(hip_phase)
    hip_l, hip_r = hip_amp * wave(ph_l, duty), hip_amp * wave(ph_r, duty_r)
    qj = np.stack([hip_l, hip_r], axis=1)
    if noise:
        qj = qj + rng.normal(0.0, noise, qj.shape)
    root = np.stack([speed * t_state + 0.005 * np.sin(2 * math.pi * 2 * ph_l), np.zeros_like(t_state),
                     0.9 + 0.01 * np.cos(2 * math.pi * 2 * ph_l)], axis=1)
    x_rel = np.stack([0.45 * np.sin(qj[:, 1]), 0.45 * np.sin(qj[:, 0]) + lead_bias], axis=1)  # [R, L]
    sd = np.zeros((T + 1, 2), dtype=bool)
    sd[1:] = frac >= 0.5
    sd[0] = sd[1]
    h = np.where(sd, 0.0, 0.03)
    fp = np.zeros((T + 1, 2, 3))
    fp[:, :, 0] = root[:, None, 0] + x_rel
    fp[:, 0, 1], fp[:, 1, 1] = -0.18, 0.18
    fp[:, :, 2] = h
    fv = np.zeros((T, 2, 3))
    fv[:, :, :2] = (fp[1:, :, :2] - fp[:-1, :, :2]) / DT
    force_min = np.where(allsub, 410.0, 0.0)
    force_mean = 410.0 * frac
    return {"length": T, "total_reward": 0.0, "forward_vel": np.full(T, speed), "distance_traveled": speed * T * DT,
            "terminated": False, "termination_reason": "truncated", "stance_flags": np.full((T, 3), np.nan),
            "components": {}, "floor_contact_substep_frac": frac, "forces": force_min,
            "floor_force_min": force_min, "floor_force_mean": force_mean, "nonfloor_force_mean": np.zeros((T, 2)),
            "foot_foot_contact_frac": np.zeros(T), "foot_pos": fp, "root_pos": root,
            "quat": np.tile([1.0, 0.0, 0.0, 0.0], (T + 1, 1)), "qj": qj, "foot_h": h, "foot_vel": fv}


SYNTHETIC = {
    "staggered_hop_30ms": dict(period=12, down=8, offset=3),
    "staggered_hop_40ms": dict(period=12, down=8, offset=4),
    "staggered_hop_50ms": dict(period=12, down=8, offset=5),
    "synchronous_hop": dict(period=12, down=8, offset=0),
    "alternating_walk": dict(period=100, down=60, offset=50, swing_amp=0.3),
}
SYNTHETIC_CT: dict[str, dict] = {}
# the verifier's grid: staggered two-footed hops at the September hops' frequencies and 10 Hz, level feet, hips
# following the touchdown stagger
for _f in (7.7, 8.3, 9.2, 10.0):
    for _off in (20, 30, 40, 50):
        SYNTHETIC_CT[f"ct_stagger_hop_{_f}Hz_{_off}ms"] = dict(freq_hz=_f, duty_l=0.7, duty_r=0.7,
                                                              td_offset_s=_off / 1000)
SYNTHETIC_CT.update({
    "ct_stagger_hop_9.2Hz_40ms_hips_in_phase": dict(freq_hz=9.2, duty_l=0.7, duty_r=0.7, td_offset_s=0.04,
                                                    hip_offset_s=0.0),
    "ct_stagger_hop_8.3Hz_30ms_left_lead": dict(freq_hz=8.3, duty_l=0.7, duty_r=0.7, td_offset_s=0.03, lead_bias=0.3),
    # scissor hop: hips (and feet) in anti-phase, touchdowns 30 ms apart
    "ct_scissor_hop_8.3Hz_30ms": dict(freq_hz=8.3, duty_l=0.7, duty_r=0.7, td_offset_s=0.03, hip_offset_s=0.5 / 8.3),
    "ct_walk_1.0Hz_sym": dict(freq_hz=1.0, duty_l=0.6, duty_r=0.6, td_offset_s=0.5, speed=0.2),
    "ct_walk_1.6Hz_sym_noisy": dict(freq_hz=1.6, duty_l=0.62, duty_r=0.62, td_offset_s=0.5 / 1.6, speed=0.3,
                                    noise=0.03, seed=3),
    "ct_walk_0.7Hz_slow": dict(freq_hz=0.7, duty_l=0.7, duty_r=0.7, td_offset_s=0.5 / 0.7, speed=0.1),
    "ct_walk_1.2Hz_pelvis2f": dict(freq_hz=1.2, duty_l=0.6, duty_r=0.6, td_offset_s=0.5 / 1.2, speed=0.2,
                                   hip_harm2=0.12),
    "ct_limp_walk_1.2Hz_40pct": dict(freq_hz=1.2, duty_l=0.6, duty_r=0.6, td_offset_s=0.4 / 1.2, speed=0.2),
    "ct_limp_walk_1.2Hz_30pct": dict(freq_hz=1.2, duty_l=0.6, duty_r=0.6, td_offset_s=0.3 / 1.2, speed=0.2),
    "ct_alt_run_3Hz_flight": dict(freq_hz=3.0, duty_l=0.3, duty_r=0.3, td_offset_s=0.5 / 3.0, speed=1.0),
    # the final rule's limits (this round)
    "ct_walk_4.0Hz_sym": dict(freq_hz=4.0, duty_l=0.6, duty_r=0.6, td_offset_s=0.5 / 4.0, speed=0.3),
    "ct_alt_shuffle_5.5Hz": dict(freq_hz=5.5, duty_l=0.6, duty_r=0.6, td_offset_s=0.5 / 5.5, speed=0.3),
    "ct_scissor_hop_4.5Hz_30ms": dict(freq_hz=4.5, duty_l=0.7, duty_r=0.7, td_offset_s=0.03, hip_offset_s=0.5 / 4.5),
    "ct_stagger_hop_4.5Hz_40ms": dict(freq_hz=4.5, duty_l=0.7, duty_r=0.7, td_offset_s=0.04),
    "ct_stagger_hop_4.5Hz_80ms": dict(freq_hz=4.5, duty_l=0.7, duty_r=0.7, td_offset_s=0.08),
    # final round: short-lift hops (feet lifting for 1-2 control steps) through trace_ct. The round-3 verifier's
    # quoted case (6 Hz, duty 0.9, right foot 60 ms after the left, hips following: stride 4.02 Hz, lag 0.353, WALK
    # under the round-2 rule), the lowest-duty one of its 70 (11 Hz, duty 0.85: both feet 0.842) and a 15 Hz chatter
    # at duty 0.80 with anti-phase hips (both feet 0.800-0.802)
    "ct_shortlift_hop_6Hz_d0.90_60ms": dict(freq_hz=6.0, duty_l=0.9, duty_r=0.9, td_offset_s=0.06),
    "ct_shortlift_hop_11Hz_d0.85_-40ms": dict(freq_hz=11.0, duty_l=0.85, duty_r=0.85, td_offset_s=-0.04),
    "ct_chatter_15Hz_d0.80_40ms_scissor": dict(freq_hz=15.0, duty_l=0.8, duty_r=0.8, td_offset_s=0.04,
                                              hip_offset_s=0.5 / 15.0),
    # the residual's lowest duty found: a 22 Hz chatter at duty 0.74 (both feet 0.740) still reads WALK
    "ct_chatter_22Hz_d0.74_-40ms": dict(freq_hz=22.0, duty_l=0.74, duty_r=0.74, td_offset_s=-0.04),
    # creeping walks either side of the duty guard (feet half a cycle apart)
    "ct_walk_1.0Hz_d0.75": dict(freq_hz=1.0, duty_l=0.75, duty_r=0.75, td_offset_s=0.5, speed=0.15),
    "ct_walk_1.0Hz_d0.85": dict(freq_hz=1.0, duty_l=0.85, duty_r=0.85, td_offset_s=0.5, speed=0.15),
})
# final round: trace_ev cases (the round-3 verifier's generator; its grid-2 specs, seed included)
SYNTHETIC_EV: dict[str, dict] = {
    # short-lift hops that read WALK under the round-2 rule (contact bar, band and cap all pass); the duty guard
    # rejects every one
    "ev_shortlift_hop_6Hz_d0.90_-50ms_hip0.4": dict(freq=6.0, td_offset_s=-0.05, hip_phase=0.4, duty=0.9, seed=1),
    "ev_shortlift_hop_8Hz_d0.90_-30ms_hip0.6": dict(freq=8.0, td_offset_s=-0.03, hip_phase=0.6, duty=0.9, seed=1),
    "ev_shortlift_hop_10Hz_d0.85_-30ms_follow": dict(freq=10.0, td_offset_s=-0.03, hip_phase="follow", duty=0.85,
                                                     seed=1),
    "ev_shortlift_hop_12Hz_d0.85_-30ms_hip0.8": dict(freq=12.0, td_offset_s=-0.03, hip_phase=0.8, duty=0.85, seed=1),
    # this round's extension: a 16 Hz chatter at duty 0.80, the lowest-duty false WALK found (both feet 0.799-0.800)
    "ev_chatter_16Hz_d0.80_-30ms_hip0.6": dict(freq=16.0, td_offset_s=-0.03, hip_phase=0.6, duty=0.8, seed=1),
    # walks: the real walker's regime (4 Hz, duty 0.62) and creeping walks either side of the duty guard
    "ev_walk_4Hz_d0.62": dict(freq=4.0, td_offset_s=0.125, hip_phase="follow", duty=0.62, seed=1, speed=0.3,
                              noise=0.01),
    "ev_walk_1Hz_d0.70": dict(freq=1.0, td_offset_s=0.5, hip_phase="follow", duty=0.7, seed=1, speed=0.2,
                              noise=0.01),
    "ev_walk_1Hz_d0.85": dict(freq=1.0, td_offset_s=0.5, hip_phase="follow", duty=0.85, seed=1, speed=0.2,
                              noise=0.01),
}


def _clean(v):
    if isinstance(v, (np.floating, float)):
        v = float(v)
        return None if not math.isfinite(v) else v
    if isinstance(v, np.integer):
        return int(v)
    return v


def synthetic_rows(probe) -> dict:
    rd = robot()
    out = {}
    for name, spec in SYNTHETIC.items():
        row = probe.analyze_episode(trace(**spec), rd, DT, SETTLE, 0)
        out[name] = {"spec": spec, "row": {k: _clean(row.get(k)) for k in KEYS if k in row}}
    for name, spec in SYNTHETIC_CT.items():
        row = probe.analyze_episode(trace_ct(**spec), rd, DT, SETTLE, 0)
        out[name] = {"spec": dict(spec, generator="trace_ct"), "row": {k: _clean(row.get(k)) for k in KEYS if k in row}}
    for name, spec in SYNTHETIC_EV.items():
        row = probe.analyze_episode(trace_ev(**spec), rd, DT, SETTLE, 0)
        out[name] = {"spec": dict(spec, generator="trace_ev"), "row": {k: _clean(row.get(k)) for k in KEYS if k in row}}
    return out


def sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tree", required=True, help="a pilot tree holding the frozen probe")
    ap.add_argument("--keep-real", default=None, help="copy real_walker from this existing fixture instead")
    ap.add_argument("--real-json", default=None, help="the probe JSON of the real walker re-run")
    ap.add_argument("--real-meta", default=None, help="provenance of the real walker re-run (JSON)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    probe, probe_path = load_probe(Path(a.tree))
    doc = {"schema": "walking-pilot-gait-cases/v3", "probe": str(probe_path), "probe_sha256": sha(probe_path),
           "dt": DT, "steps": T, "settle": SETTLE, "synthetic": synthetic_rows(probe)}
    if a.keep_real and a.real_json:
        kept = json.loads(Path(a.keep_real).read_text())["real_walker"]
        want = kept["provenance"].get("probe_json_sha256")
        if sha(Path(a.real_json)) != want:
            raise SystemExit(f"{a.real_json}: sha256 {sha(Path(a.real_json))} is not the provenance's {want}")
        real = json.loads(Path(a.real_json).read_text())
        doc["real_walker"] = {"provenance": kept["provenance"],
                              "rows": [{k: _clean(r.get(k)) for k in KEYS if k in r} for r in real["episodes"]]}
    elif a.keep_real:
        doc["real_walker"] = json.loads(Path(a.keep_real).read_text())["real_walker"]
    elif a.real_json:
        real = json.loads(Path(a.real_json).read_text())
        rows = [{k: _clean(r.get(k)) for k in KEYS if k in r} for r in real["episodes"]]
        meta = json.loads(Path(a.real_meta).read_text()) if a.real_meta else {}
        doc["real_walker"] = {"provenance": dict(meta, probe_json=a.real_json, probe_json_sha256=sha(Path(a.real_json)),
                                                 probe_meta={k: real["meta"].get(k) for k in (
                                                     "species", "stage", "run_id", "checkpoint_sha256",
                                                     "vecnorm_sha256", "seed_scheme", "seed_base", "episodes", "dt",
                                                     "settle_steps_gait_window")}),
                              "rows": rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1))
    for name, c in doc["synthetic"].items():
        r = c["row"]
        print(f"{name:40s} f {r.get('floor_stride_frequency_hz')} alt {r.get('floor_alternation_index')} sim {r.get('floor_simultaneous_fraction')} "
              f"flight {r.get('floor_flight_fraction')} swaps {r.get('floor_lead_swaps_per_stride')} "
              f"hip lag {r.get('jointlag_frac_hip_pitch')} duty L/R {r.get('floor_duty_L')}/{r.get('floor_duty_R')} "
              f"class {r.get('floor_gait_class')!r}")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
