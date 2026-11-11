"""Stage 1 fail-fast gates G1 (plant), G2 (references), G3 (transitions) and G4 (tracker), ``STAGE1_PLAN.md``
section 3, steps 2-5.  G3 and G4 are documented at ``g3`` and ``g4``: G4 is judged at the tracker's tuning start with
the 20-start panel rule and attributes a failing condition's falls over its starts (decisions 15-20, ``g4_decided``,
``attribute``); G3 reports the stop references' stops under decision 18 (``stop_criterion``).

Every gate takes the plant options of ``plant.build`` (``plant_kw``: ``ankle_gains`` and ``ankle_armature``, the bus,
the envelope's braking rule) and records them.  The defaults are the plant and the G2 rule that ``STAGE1_PLAN.md``
decided during the build (decisions 11-13: joint-space ankle gains, the linkage's armature, the 20-start panel);
G2 records them as ``decided``, with what each run used, and keeps the other choices' results as evidence.

G1, the plant:

* the hashes match (``plant.verify`` on the model the plant loads; ``refs.source`` on the rebuilt one);
* with zero command (every servo at its home target, no feedforward) the body stands ``STAND_S`` on hard floor and
  on low pile from its settled stance: the stance settled (``plant.Plant.settle``), no fall (``Plant.fall_cause``)
  and a trunk tilt from home that stays below 12 deg (``carpet.UPRIGHT_TILT_DEG``, the Stand behaviour's limit);
* the tendon map equals ``sizing._motor_space`` (for the record's leg layout) to ``TENDON_TOL`` on random torques and
  speeds, through the compiled model's own transmission (motor torques from the sizing map, applied as raw motor
  torques, give back the joint torques; joint speeds give the sizing's motor speeds), and the ankle reaches, but
  never exceeds, ``|tau_p| + |tau_r| = 2 peak``;
* seeded runs repeat exactly: two fresh builds with the same seed (random start noise and random servo targets
  drawn from the plant's generator) give bit-identical traces, another seed does not, and reseeding one plant
  (``Plant.reset(seed=...)``) repeats its first trace;
* the plant (envelope included) runs at least ``MIN_STEPS_PER_S`` physics steps per second per core with zero
  command on hard floor, alone and with one process per core: each process times ``RATE_SAMPLES`` runs of
  ``RATE_STEPS`` steps (the processes per core start their timed runs together), every sample is recorded, and the
  median of the lone process's samples and the median of all the per-core samples are judged (one short sample is
  at the mercy of a passing load: a single 20k-step sample once read 18.5k steps/s alone, while the four per-core
  processes right after it averaged 26.5k).  The machine's cores are recorded (``machine_cores``); a per-core run
  with fewer processes than cores (``PYTHON_CPU_COUNT`` below the core count, on a shared machine) leaves the step
  rate, and G1's pass, provisional (None: ``PROVISIONAL_RATE_RULE``) until G1 runs with one process per core.

G2, the references:

* every reference matches its records to 6 digits (``refs.compare``, recorded in ``refs.json``);
* the feedforward replay of the nominal walk on the plant with the envelope walks at least ``MIN_REPLAY_STRIDES``
  strides.  The replay is the reference with feedforward and servo PD only (R0, no feedback): every 2 ms the servos
  get the reference joint angles and rates and a feedforward torque, from the reference's first frame (already
  walking) until a fall or ``REPLAY_STRIDES``.  The feedforward is the inverse-dynamics torque, plus the MJCF's
  passive joint terms (damping x qd_ref + frictionloss x tanh(qd_ref / 0.05)), plus the armature the plant has
  beyond the screened model the inverse dynamics ran on, ``(A_plant - A_screened) qdd_ref``
  (``Plant.armature_extra``: the second motor's armature on each ankle axis; zero with
  ``ankle_armature="screened"``).

The pass rule is ``rule``.  "panel" (``STAGE1_PLAN.md`` decision 13, the default) judges the median of
``PANEL_RUNS`` replays from starts whose root velocity is offset in x and y by at most ``PANEL_OFFSET_M_S`` (seed
``PANEL_SEED``; the same offsets for every plant) against ``MIN_REPLAY_STRIDES``.  "single" (the plan's original
wording) judges the one unperturbed run.  Both verdicts are always recorded, and the unperturbed run is reported.

Reported with G2, not gated:

* the open-loop replay has no feedback, so its fall time is chaotic (on the 60 cm bodies start offsets of 1 um/s
  already move it by several strides); the panel is run on every combination of the two plant choices
  (``PLANT_CHOICES``, the 2 x 2 plant matrix), and ``decided`` cites the plan's decisions 11-13 with that matrix
  as evidence.  The decisions rest on reasons that do not depend on the gate's outcome; nothing is tuned to pass;
* every other reference replayed the same way, with and without the envelope (``walking_turns`` lists the walking
  turns);
* for each replay: the fall cause, the envelope's clamps (driving or braking side), the smallest margin to each
  joint's plant range, the lowest toe or claw point (they do not collide, so they can pass through the floor), and,
  on the gated plant with the envelope, the leg-to-leg gap sampled every 20 ms (``legs.LegGap``);
* the joint-range margins of every reference against the plant's ranges, under the screens' reading of
  ``joint_range_usable`` (the plant's) and the literal one (``plant.usable_ranges``), listing the joints left with
  less than the 0.10 rad of usable range plan section 4 asks every success to keep;
* the nominal replay without the armature feedforward term.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from .. import carpet, statics
from . import drive, estimator, plant
from .legs import LegGap
from .refs import Reference

STAND_S = 20.0
STAND_FLOORS = ("hard", "low")
TENDON_TOL = 1e-9
TENDON_SAMPLES = 500
REPEAT_S = 2.0
MIN_STEPS_PER_S = 20_000
RATE_STEPS = 50_000  # physics steps per timed sample (about 2 s)
RATE_SAMPLES = 5  # timed samples per process
RATE_WARMUP = 1000
RATE_BARRIER_S = 600.0  # the longest the per-core processes wait for each other before timing
REPLAY_STRIDES = 60.0
MIN_REPLAY_STRIDES = 10.0
#: The design study's replay of the probe's bodies on the screened MJCF (plan section 3 step 3); Stage 1b's bodies had
#: none (None).
DESIGN_STUDY_STRIDES = {"60-N-T-10": 13.6, "60-N-H-10": 16.6, "75-M-H-10": 60.0}
FRICTION_SMOOTH = 0.05  # rad/s, the frictionloss compensation's tanh width
#: G2 pass rules; the first is ``STAGE1_PLAN.md`` decision 13's and the default.
RULES = ("panel", "single")
PANEL_RUNS = 20
PANEL_OFFSET_M_S = 1e-3
PANEL_SEED = 0
PANEL_CHUNK = 5
SAMPLE_EVERY = 10  # physics steps between trace samples (20 ms): toe heights and leg gaps
PRE_FALL_S = 0.5  # the last part of a falling run, left out of the *_before_fall values (plan section 4's window)
MIN_RANGE_RAD = 0.10  # plan section 4: usable range every joint must keep
#: The two plant choices (``plant.build``), every combination: G2's 2 x 2 plant matrix.
PLANT_CHOICES = {
    "motor_gains+linkage_armature": dict(ankle_gains="motor", ankle_armature="linkage"),
    "joint_gains+linkage_armature": dict(ankle_gains="joint", ankle_armature="linkage"),
    "motor_gains+screened_armature": dict(ankle_gains="motor", ankle_armature="screened"),
    "joint_gains+screened_armature": dict(ankle_gains="joint", ankle_armature="screened"),
}
RULE_TEXT = {
    "single": f"the unperturbed nominal replay with the envelope walks at least {MIN_REPLAY_STRIDES:g} strides "
              "(plan section 3 step 3 as written)",
    "panel": f"the median of {PANEL_RUNS} nominal replays with the envelope, from starts whose root velocity is offset "
             f"in x and y by at most {1e3 * PANEL_OFFSET_M_S:g} mm/s (seed {PANEL_SEED}), is at least "
             f"{MIN_REPLAY_STRIDES:g} strides (STAGE1_PLAN.md decision 13)",
}
#: Where the plan records the choices G2 depends on.
DECISIONS_SOURCE = "STAGE1_PLAN.md, section 9, 'Decisions taken during the build (step 3, 8 Nov)'"
#: The plan's decisions 11-13: the option each one takes and its reason (which does not depend on the gate's outcome).
DECISIONS = {
    11: dict(choice="ankle_gains", taken=plant.ANKLE_GAINS[0], title="Ankle servo gains: joint space",
             reason="carpet.hold_servos defines one hold servo per joint for the screened model, which drives each "
                    "ankle axis with one motor; applying that rule to each of the linkage's two motors doubles the "
                    "ankle's joint-space stiffness, and half the gain per motor reproduces the screened stiffness "
                    "exactly (the damping that results is in plant.json, gains.ankle_joint_space)"),
    12: dict(choice="ankle_armature", taken=plant.ANKLE_ARMATURE[0], title="Ankle armature: the linkage's",
             reason="both motors' rotors are geared to both ankle axes, so each axis carries twice the screened "
                    "single-motor armature: the physical two-motor ankle the plan specifies"),
    13: dict(choice="g2_rule", taken=RULES[0], title="G2 replay rule: a 20-start panel",
             reason="an open-loop replay has no feedback, so its fall time is chaotic (start offsets of 1 mm/s move "
                    "one body's run from 25.8 to 5 strides); G2 is judged on the panel's median and the unperturbed "
                    "run is reported"),
}


def _saturated(m, d) -> np.ndarray:
    f, fr = d.actuator_force, m.actuator_forcerange
    return (f >= fr[:, 1] - 1e-9) | (f <= fr[:, 0] + 1e-9)


# ------------------------------------------------------------------------------------------------
# G1
# ------------------------------------------------------------------------------------------------
def stand(body: str, floor: str, seconds: float = STAND_S, **plant_kw) -> dict:
    """Zero-command stance from the settled home pose."""
    t0 = time.perf_counter()
    p = plant.build(body, floor, **plant_kw)
    build_s = time.perf_counter() - t0
    if not p.settled:
        s = p.meta["settle"]
        return dict(floor=floor, stands=False, reason=f"no settled stance: the zero-command stance {s['status']} "
                    f"after {s['seconds']:.1f} s", settle=s, floor_info=p.meta["floor_info"], build_s=build_s)
    m, d = p.m, p.d
    p.reset()
    p.hold()
    root = p.trunk
    com0 = d.subtree_com[root].copy()
    n = int(round(seconds / m.opt.timestep))
    tilt_max, z_min, fell_at, cause = 0.0, float(d.qpos[2]), None, None
    sat = np.zeros(m.nu, bool)
    t1 = time.perf_counter()
    for k in range(n):
        p.step()
        sat |= _saturated(m, d)
        tilt_max = max(tilt_max, p.tilt_from_home())
        z_min = min(z_min, float(d.qpos[2]))
        cause = p.fall_cause()
        if cause is not None:
            fell_at = float(d.time)
            break
    wall = time.perf_counter() - t1
    tilt_max_deg = float(np.degrees(tilt_max))
    com = d.subtree_com[root]
    return dict(floor=floor, seconds=float(d.time), fell_at_s=fell_at, fall_cause=cause,
                stands=fell_at is None and tilt_max_deg < carpet.UPRIGHT_TILT_DEG, tilt_from_home_max_deg=tilt_max_deg,
                tilt_from_home_end_deg=float(np.degrees(p.tilt_from_home())), root_z_min_m=z_min,
                com_drift_mm=1e3 * float(np.linalg.norm(com[:2] - com0[:2])), com_drop_mm=1e3 * float(com0[2] - com[2]),
                foot_load_end_N=[float(v) for v in p.foot_loads()],
                saturated=[p.drives[i].actuator for i in np.flatnonzero(sat)], build_s=build_s, wall_s=wall,
                steps_per_s=(k + 1) / wall, settle=p.meta["settle"],
                floor_info=p.meta["floor_info"])


def tendon_map(body: str, samples: int = TENDON_SAMPLES, seed: int = 0, **plant_kw) -> dict:
    """The compiled transmission against ``sizing._motor_space`` (module docstring)."""
    p = plant.build(body, "hard", settle=False, **plant_kw)
    m, d = p.m, p.d
    layout = p.meta["leg_layout"]
    rng = np.random.default_rng(seed)
    peak = np.array([dr.peak for dr in p.drives])
    err_tau = err_w = err_tau_G = err_w_G = 0.0
    m.actuator_gainprm[:, 0] = 1.0  # raw motor torques: force = ctrl
    m.actuator_biasprm[:] = 0.0
    m.actuator_forcelimited[:] = 0
    for _ in range(samples):
        tau = rng.uniform(-1.0, 1.0, len(p.joints)) * peak
        omega = rng.uniform(-10.0, 10.0, len(p.joints))
        f_s, w_s = drive.sizing_motor_values(p.joints, tau, omega, layout)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = p.q_home
        d.qpos[p.qadr] += rng.uniform(-0.2, 0.2, len(p.joints))
        d.qvel[p.dofs] = omega
        d.ctrl[:] = f_s
        mujoco.mj_forward(m, d)
        err_tau = max(err_tau, float(np.max(np.abs(d.qfrc_actuator[p.dofs] - tau))))
        err_w = max(err_w, float(np.max(np.abs(d.actuator_velocity - w_s))))
        err_tau_G = max(err_tau_G, float(np.max(np.abs(drive.motor_torques(p.G, tau) - f_s))))
        err_w_G = max(err_w_G, float(np.max(np.abs(p.G @ omega - w_s))))
    # the linkage limit: motors at +-peak (forces clamped by the force range) give at most 2 peak on one axis
    m.actuator_forcelimited[:] = 1
    m.actuator_forcerange[:, 0], m.actuator_forcerange[:, 1] = -peak, peak
    ankle = {}
    for side in statics.SIDES:
        ia, ib = (p.joints.index(f"{side}_{j}") for j in ("ankle_pitch", "ankle_roll"))
        worst = 0.0
        for _ in range(samples):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = p.q_home
            d.ctrl[:] = rng.uniform(-10.0, 10.0, m.nu) * peak
            mujoco.mj_forward(m, d)
            worst = max(worst, abs(d.qfrc_actuator[p.dofs[ia]]) + abs(d.qfrc_actuator[p.dofs[ib]]))
        corners = []
        for ca, cb in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = p.q_home
            d.ctrl[ia], d.ctrl[ib] = 10 * ca * peak[ia], 10 * cb * peak[ib]
            mujoco.mj_forward(m, d)
            corners.append([float(d.qfrc_actuator[p.dofs[ia]]), float(d.qfrc_actuator[p.dofs[ib]])])
        ankle[side] = dict(peak_Nm=float(peak[ia]), max_abs_pitch_plus_abs_roll_Nm=float(worst),
                           corners_pitch_roll_Nm=corners,
                           within=bool(worst <= 2 * peak[ia] + TENDON_TOL),
                           reaches=bool(all(abs(abs(a) + abs(b) - 2 * peak[ia]) <= TENDON_TOL for a, b in corners)))
    ok = max(err_tau, err_w, err_tau_G, err_w_G) <= TENDON_TOL and all(v["within"] and v["reaches"]
                                                                        for v in ankle.values())
    return dict(samples=samples, tolerance=TENDON_TOL, leg_layout=layout, max_err_joint_torque_Nm=err_tau,
                max_err_motor_speed_rad_s=err_w, max_err_plant_matrix_torque_Nm=err_tau_G,
                max_err_plant_matrix_speed_rad_s=err_w_G, ankle_limit=ankle, matches=bool(ok))


def _trace_hash(p: plant.Plant, seed: int, seconds: float = REPEAT_S) -> str:
    """A random start and random servo targets (50 Hz) from the plant's generator, reseeded with ``seed``."""
    p.reset(joint_noise=0.01, root_velocity_noise=0.02, seed=seed)
    d = p.d
    h = hashlib.sha256()
    q0 = p.q_home[p.qadr]
    for k in range(int(round(seconds / p.m.opt.timestep))):
        if k % 10 == 0:
            p.command(q0 + p.rng.normal(0.0, 0.02, len(q0)))
        p.step()
        for arr in (d.qpos, d.qvel, d.sensordata, d.actuator_force):
            h.update(arr.tobytes())
    return h.hexdigest()


def repeatability(body: str, seed: int = 1, **plant_kw) -> dict:
    """Fresh builds with one seed repeat bit for bit, another seed differs, and a reseeded plant repeats."""
    first = plant.build(body, "hard", seed=seed, **plant_kw)
    a = _trace_hash(first, seed)
    b = _trace_hash(plant.build(body, "hard", seed=seed, **plant_kw), seed)
    c = _trace_hash(plant.build(body, "hard", seed=seed + 1, **plant_kw), seed + 1)
    e = _trace_hash(first, seed)
    return dict(seconds=REPEAT_S, seed=seed, trace_sha256=a, repeat_sha256=b, other_seed=seed + 1, other_sha256=c,
                reseeded_sha256=e, repeats=a == b, other_seed_differs=a != c, reseeding_repeats=a == e,
                ok=a == b and a != c and a == e)


_RATE_BARRIER = None  # in a step-rate pool's workers: the per-core processes start their timed runs together


def _rate_init(barrier) -> None:
    global _RATE_BARRIER
    _RATE_BARRIER = barrier


def _rate(body: str, steps: int = RATE_STEPS, samples: int = RATE_SAMPLES, plant_kw: dict | None = None) -> list:
    """Steps per second of ``samples`` consecutive timed runs of ``steps`` physics steps each (zero command, hard
    floor), after ``RATE_WARMUP`` untimed steps; in a step-rate pool the timed runs start once every process is
    ready."""
    p = plant.build(body, "hard", **(plant_kw or {}))
    p.reset()
    p.hold()
    p.step(RATE_WARMUP)
    if _RATE_BARRIER is not None:
        _RATE_BARRIER.wait(RATE_BARRIER_S)
    out = []
    for _ in range(samples):
        t0 = time.perf_counter()
        p.step(steps)
        out.append(steps / (time.perf_counter() - t0))
    return out


#: When the per-core run's process count is not the machine's core count (``machine_cores``), the step rate is not
#: judged: ``ok`` is None (provisional) and the samples' reading against the threshold is kept beside it.
PROVISIONAL_RATE_RULE = (
    "the rule times one process per core; a run with another number (a pool sized by PYTHON_CPU_COUNT below the core "
    "count, so that a build elsewhere on the machine keeps its cores) leaves the step rate unjudged (ok None, and G1's "
    "pass with it, unless another check fails): ok_on_these_processes is these samples against the threshold, and "
    "the verdict waits for G1 again with one process per core on an otherwise idle machine")


def machine_cores() -> dict:
    """The cores the step-rate rule's "one process per core" counts: the processors this process may run on
    (``os.sched_getaffinity``, which ``PYTHON_CPU_COUNT`` does not change), else ``os.cpu_count()``; with what
    ``os.cpu_count()`` returns here and ``PYTHON_CPU_COUNT``, which sets it and so every pool's default size."""
    try:
        n, source = len(os.sched_getaffinity(0)), "os.sched_getaffinity(0)"
    except AttributeError:  # (not on every platform)
        n, source = os.cpu_count() or 1, "os.cpu_count() (no sched_getaffinity here: PYTHON_CPU_COUNT may set it)"
    return dict(cores=int(n), cores_source=source, os_cpu_count=os.cpu_count(),
                python_cpu_count=os.environ.get("PYTHON_CPU_COUNT"))


def rate_summary(single: list, parallel: list[list], steps: int = RATE_STEPS, machine: dict | None = None) -> dict:
    """G1's step-rate verdict from every sample: the lone process's samples (``single``) and those of each process of
    the per-core run (``parallel``, one list per process).  The rate per core is the smaller of the two medians (the
    lone process's samples, and all the per-core samples together), judged against ``MIN_STEPS_PER_S``.

    With ``machine`` (``machine_cores``, recorded) the per-core run must have one process per core: with another
    number the verdict is provisional (``PROVISIONAL_RATE_RULE``: ``ok`` None)."""
    single = [float(v) for v in single]
    parallel = [[float(v) for v in s] for s in parallel]
    med_single = float(np.median(single))
    med_par = float(np.median(np.concatenate(parallel)))
    per_core = min(med_single, med_par)
    out = dict(rule=f"median of every sample ({len(single)} per process, {steps} steps each), alone and with one "
                    f"process per core, against {MIN_STEPS_PER_S} steps/s",
               steps_per_sample=int(steps), samples_per_process=len(single), statistic="median",
               single_process_samples=single, single_process=med_single,
               processes=len(parallel), per_process_samples=parallel,
               per_process_median=[float(np.median(s)) for s in parallel], per_process_all_median=med_par,
               per_process_mean=float(np.mean(np.concatenate(parallel))),
               lowest_sample=float(min(min(single), *(min(s) for s in parallel))),
               per_core=per_core, threshold=MIN_STEPS_PER_S, ok=bool(per_core >= MIN_STEPS_PER_S))
    if machine is not None:
        out["machine"] = dict(machine)
        if len(parallel) != int(machine["cores"]):
            env_n = machine.get("python_cpu_count")
            out.update(ok=None, ok_on_these_processes=out["ok"], provisional_rule=PROVISIONAL_RATE_RULE,
                       provisional=f"step rate timed on {len(parallel)} of {machine['cores']} cores"
                                   + (f" (PYTHON_CPU_COUNT={env_n})" if env_n is not None else ""))
    return out


def step_rate(body: str, steps: int = RATE_STEPS, samples: int = RATE_SAMPLES, procs: int | None = None,
              **plant_kw) -> dict:
    """Plant steps per second (zero command, hard floor, envelope on): one process alone, then one per core at once,
    ``samples`` timed runs of ``steps`` steps in each (``rate_summary``, judged against the machine's cores:
    ``machine_cores``).  The per-core run takes ``procs`` processes (default ``os.cpu_count()``, which
    ``PYTHON_CPU_COUNT`` sets)."""
    machine = machine_cores()
    procs = procs or os.cpu_count() or 1
    single = _rate(body, steps, samples, plant_kw)
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, initializer=_rate_init, initargs=(ctx.Barrier(procs),)) as pool:
        par = pool.starmap(_rate, [(body, steps, samples, plant_kw)] * procs, chunksize=1)
    return rate_summary(single, par, steps, machine=machine)


def combine(verdicts) -> bool | None:
    """Several verdicts as one: False if any is false, else None (provisional) if any is None, else True."""
    vs = list(verdicts)
    if any(v is not None and not v for v in vs):
        return False
    if any(v is None for v in vs):
        return None
    return True


def g1(body: str, out_dir: Path | None = None, **plant_kw) -> dict:
    """G1; with ``out_dir`` (``refs.prepare`` output) the rebuilt model's hash check is included as well."""
    t0 = time.perf_counter()
    xml, _ = plant.model_files(body)
    hashes = dict(plant_model=plant.verify(body, xml))
    path = Path(out_dir) / "refs.json" if out_dir is not None else None
    if path is not None and path.exists():
        hashes["reference_model"] = json.loads(path.read_text())["model"]
    out = dict(plant_options=plant.build(body, "hard", settle=False, **plant_kw).meta["options"], hashes=hashes)
    out["stand"] = {f: stand(body, f, **plant_kw) for f in STAND_FLOORS}
    out["tendon_map"] = tendon_map(body, **plant_kw)
    out["repeatability"] = repeatability(body, **plant_kw)
    out["step_rate"] = step_rate(body, **plant_kw)
    out["pass"] = combine([all(h["matches_record"] and h["matches_plan"] for h in hashes.values()),
                           all(s["stands"] for s in out["stand"].values()), out["tendon_map"]["matches"],
                           out["repeatability"]["ok"], out["step_rate"]["ok"]])
    if out["pass"] is None:  # every other check passes; the step rate waits for its per-core run
        out["provisional"] = out["step_rate"]["provisional"]
    out["seconds"] = time.perf_counter() - t0
    return out


# ------------------------------------------------------------------------------------------------
# G2
# ------------------------------------------------------------------------------------------------
def _yaw(R) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def replay(p: plant.Plant, ref: Reference, strides: float = REPLAY_STRIDES, start_velocity_offset=None,
           armature_ff: bool = True, legs: LegGap | None = None, sample_every: int = SAMPLE_EVERY) -> dict:
    """Feedforward replay (module docstring) of ``ref`` on ``p``; ``start_velocity_offset`` (m/s, world) is added
    to the root velocity of the start frame; ``armature_ff`` adds the plant's extra armature term; ``legs`` samples
    the leg-to-leg gap every ``sample_every`` steps.

    Joint-range margins are tracked every step (``range_margin_min_rad``); they, the toe heights and the leg gaps
    are also sampled every ``sample_every`` steps, so that a fall's last ``PRE_FALL_S`` (when the body is already
    going down) can be told apart: the ``*_before_fall`` values leave it out.  The worst margin and the limits
    reached leave out an end a joint rests on by design (``plant.REST_ON_STOP_RULE``), which ``rest_on_stop``
    reports (on a body with such a joint)."""
    if list(ref.joints) != p.joints:
        raise ValueError("reference and plant joints differ")
    m, d, env = p.m, p.d, p.envelope
    nj = len(p.joints)
    S = np.hstack(ref.joint_series(p.qadr, p.dofs) + (ref.qdd_series(p.dofs),))
    damp, fric = m.dof_damping[p.dofs].copy(), m.dof_frictionloss[p.dofs].copy()
    extra = p.armature_extra if armature_ff and np.any(p.armature_extra) else None
    qd0 = ref.qd[0].copy()
    if start_velocity_offset is not None:
        qd0[:3] += np.asarray(start_velocity_offset, float)
    p.set_state(ref.q[0], qd0)
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in p.joints]
    lo, hi = m.jnt_range[jid, 0].copy(), m.jnt_range[jid, 1].copy()
    rest = plant.rest_ends(p.joints)
    glo, ghi = lo, hi  # the ends the margins judge: every end but one a joint rests on by design
    if rest.any():
        glo, ghi = np.where(rest < 0, -np.inf, lo), np.where(rest > 0, np.inf, hi)
    x0 = d.qpos[:2].copy()
    R0 = d.xmat[p.trunk].reshape(3, 3).copy()
    n = int(round(strides * ref.period / m.opt.timestep))
    peak = env.peak
    w, wa, wmax = np.zeros(m.nu), np.zeros(m.nu), np.zeros(m.nu)
    sat_steps = np.zeros(m.nu)
    clamp_drive, clamp_brake = np.zeros(m.nu, int), np.zeros(m.nu, int)
    qmin, qmax = d.qpos[p.qadr].copy(), d.qpos[p.qadr].copy()
    past = clamped = 0
    fell_at = cause = None
    samples = []  # (t, toe height, toe geom, smallest range margin, its joint)
    frames = []
    yaw_prev, yaw_turned = _yaw(R0), 0.0
    t0 = time.perf_counter()
    k = 0
    for k in range(n):
        x = ref.interp(d.time, S)
        q, qd, tau = x[:nj], x[nj : 2 * nj], x[2 * nj : 3 * nj]
        ff = tau + damp * qd + fric * np.tanh(qd / FRICTION_SMOOTH)
        if extra is not None:
            ff += extra @ x[3 * nj :]
        p.command(q, qd, ff)
        np.dot(env.moment, d.qvel, out=w)  # the drive speeds the envelope sets this step's limits from
        p.step()
        f = d.actuator_force
        if env.active:  # some drive past its corner: its force range was narrowed for this step
            past += 1
            fr = m.actuator_forcerange
            at_hi, at_lo = f >= fr[:, 1] - 1e-9, f <= fr[:, 0] + 1e-9
            sat_steps += at_hi | at_lo
            env_cl = (at_hi & (fr[:, 1] < peak - 1e-12)) | (at_lo & (fr[:, 0] > -peak + 1e-12))
            if env_cl.any():
                clamped += 1
                brake = f * w < 0
                clamp_brake += env_cl & brake
                clamp_drive += env_cl & ~brake
        else:
            sat_steps += np.abs(f) >= peak - 1e-9
        np.maximum(wmax, np.abs(w, out=wa), out=wmax)
        qj = d.qpos[p.qadr]
        np.minimum(qmin, qj, out=qmin)
        np.maximum(qmax, qj, out=qmax)
        yaw = _yaw(d.xmat[p.trunk].reshape(3, 3))
        yaw_turned += (yaw - yaw_prev + np.pi) % (2 * np.pi) - np.pi  # unwrapped
        yaw_prev = yaw
        if (k + 1) % sample_every == 0:
            z, g = p.toe_lowest()
            mr = np.minimum(qj - glo, ghi - qj)
            j = int(np.argmin(mr))
            samples.append((float(d.time), z, g, float(mr[j]), j))
            if legs is not None:
                frames.append(d.qpos.copy())
        cause = p.fall_cause()
        if cause is not None:
            fell_at = float(d.time)
            break
    wall = time.perf_counter() - t0
    steps = k + 1
    dur = float(d.time)
    heading = _yaw(R0)
    dxy = d.qpos[:2] - x0
    fwd = np.array([np.cos(heading), np.sin(heading)])
    ratio_max = wmax / env.rated
    worst = int(np.argmax(ratio_max))
    m_lo, m_hi = qmin - lo, hi - qmax
    g_lo, g_hi = qmin - glo, ghi - qmax
    mr = np.minimum(g_lo, g_hi)
    wj = int(np.argmin(mr))
    cut = dur if fell_at is None else fell_at - PRE_FALL_S
    keep = [s for s in samples if s[0] <= cut]
    first_limit = next((s for s in samples if s[3] <= 0.0), None)
    toe = min(samples, key=lambda s: s[1], default=None)
    toe_pre = min(keep, key=lambda s: s[1], default=None)
    rng_pre = min(keep, key=lambda s: s[3], default=None)
    rows = None if legs is None else legs.per_pose(frames)
    out = dict(reference=ref.name, envelope=env.enabled, braking=env.braking, armature_feedforward=extra is not None,
               strides_max=strides, strides=dur / ref.period, fell=fell_at is not None, fall_time_s=fell_at,
               fall_cause=cause, seconds=dur, mean_forward_speed_m_s=float(dxy @ fwd / max(dur, 1e-9)),
               lateral_drift_m=float(dxy @ np.array([-fwd[1], fwd[0]])),
               yaw_turned_deg=float(np.degrees(yaw_turned)), mean_yaw_rate_rad_s=yaw_turned / max(dur, 1e-9),
               reference_yaw_rate_rad_s=float(ref.stride_motion[0]) / ref.period,
               steps_past_corner_frac=past / steps,
               speed_ratio_max={dr.actuator: float(v) for dr, v in zip(p.drives, ratio_max, strict=True)},
               worst_speed_ratio=[p.drives[worst].actuator, float(ratio_max[worst])],
               saturated_frac={p.drives[i].actuator: float(sat_steps[i] / steps) for i in np.flatnonzero(sat_steps)},
               envelope_clamped_steps_frac=clamped / steps,
               envelope_clamps={p.drives[i].actuator: dict(driving=int(clamp_drive[i]), braking=int(clamp_brake[i]))
                                for i in np.flatnonzero(clamp_drive + clamp_brake)},
               range_margin_min_rad={jn: [float(a), float(b)] for jn, a, b in zip(p.joints, m_lo, m_hi, strict=True)},
               range_margin_worst=[p.joints[wj], "low" if g_lo[wj] <= g_hi[wj] else "high", float(mr[wj])],
               joint_limit_reached=[jn for jn, v in zip(p.joints, mr, strict=True) if v <= 0.0],
               joint_limit_first_reached=None if first_limit is None else dict(t_s=first_limit[0],
                                                                               joint=p.joints[first_limit[4]]),
               range_margin_worst_before_fall=None if rng_pre is None else [p.joints[rng_pre[4]], rng_pre[3]],
               toe_lowest_above_floor_mm=None if toe is None else 1e3 * toe[1],
               toe_lowest_geom=None if toe is None else toe[2],
               toe_lowest_before_fall_mm=None if toe_pre is None else 1e3 * toe_pre[1],
               pre_fall_s=PRE_FALL_S, sample_every_steps=sample_every,
               leg_gap=None if rows is None else legs.summary(rows),
               leg_gap_before_fall=None if rows is None or fell_at is None else legs.summary(rows[:len(keep)]),
               wall_s=wall, steps_per_s=steps / wall)
    if rest.any():
        out["rest_on_stop"] = rest_report(p.joints, rest, m_lo, m_hi)
    return out


def rest_report(joints, rest: np.ndarray, m_lo: np.ndarray, m_hi: np.ndarray) -> dict:
    """joint -> the end it rests on by design and its margin there (``plant.REST_ON_STOP_RULE``; negative: pressed
    past its stop), which the judged margins leave out."""
    out = {}
    for i in np.flatnonzero(rest):
        v = float(m_lo[i] if rest[i] < 0 else m_hi[i])
        out[joints[i]] = dict(end="low" if rest[i] < 0 else "high", margin_rad=v, pressed_past_stop_rad=max(0.0, -v),
                              rule="the end it rests on: no range limit (plant.REST_ON_STOP_RULE)")
    return out


def _offsets(runs: int = PANEL_RUNS, offset: float = PANEL_OFFSET_M_S, seed: int = PANEL_SEED) -> list:
    rng = np.random.default_rng(seed)
    return [np.r_[rng.uniform(-offset, offset, 2), 0.0] for _ in range(runs)]


def _replay_job(job: dict) -> list[dict]:
    """Worker: one plant, one reference, the replays of ``job`` (``g2``); ``full`` False keeps only the outcome."""
    p = plant.build(job["body"], "hard", settle=False, **job["plant_kw"])
    ref = Reference.load(job["ref"])
    legs = LegGap(job["body"]) if job.get("legs") else None
    out = []
    for off in job["offsets"]:
        r = replay(p, ref, job["strides"], off, armature_ff=job.get("armature_ff", True), legs=legs)
        out.append(r if job.get("full", True) else {k: r[k] for k in ("strides", "fell", "fall_cause")})
    return out


def run_jobs(jobs: list[dict], procs: int | None = None) -> list[list[dict]]:
    """``_replay_job`` over ``jobs``, on one process per core."""
    procs = min(procs or os.cpu_count() or 1, len(jobs))
    if procs <= 1:
        return [_replay_job(j) for j in jobs]
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.map(_replay_job, jobs, chunksize=1)


def panel_stats(strides) -> dict:
    v = np.asarray(strides, float)
    return dict(runs=len(v), max_offset_m_s=PANEL_OFFSET_M_S, seed=PANEL_SEED, strides=[float(x) for x in v],
                median=float(np.median(v)), min=float(v.min()), max=float(v.max()),
                runs_at_least_min_strides=int(np.sum(v >= MIN_REPLAY_STRIDES)),
                median_at_least_min_strides=bool(np.median(v) >= MIN_REPLAY_STRIDES))


def reference_range_margins(p: plant.Plant, ref: Reference) -> dict:
    """The reference's joint excursions against the plant's ranges, under both readings of ``usable_ranges``."""
    q = ref.q[:, p.qadr]
    rest = plant.rest_ends(p.joints)
    out = {}
    for reading in ("mjcf", "keep"):
        rng = plant.usable_ranges(p.body, inter_leg=reading)
        lo, hi = np.array([rng[j][0] for j in p.joints]), np.array([rng[j][1] for j in p.joints])
        m_lo, m_hi = q.min(0) - lo, hi - q.max(0)
        g_lo, g_hi = m_lo, m_hi  # an end a joint rests on by design is no range limit (plant.REST_ON_STOP_RULE)
        if rest.any():
            g_lo, g_hi = np.where(rest < 0, np.inf, m_lo), np.where(rest > 0, np.inf, m_hi)
        mr = np.minimum(g_lo, g_hi)
        w = int(np.argmin(mr))
        r = out["screens_reading" if reading == "mjcf" else "literal_reading"] = dict(
            min_margin_rad=float(mr[w]), joint=p.joints[w], side="low" if g_lo[w] <= g_hi[w] else "high",
            below_min_range={jn: float(v) for jn, v in zip(p.joints, mr, strict=True) if v < MIN_RANGE_RAD})
        if rest.any():
            r["rest_on_stop"] = rest_report(p.joints, rest, m_lo, m_hi)
    return out


def decided(choices: dict, gated: str, rule: str) -> dict:
    """The three choices behind G2 as ``STAGE1_PLAN.md`` decided them during the build (``DECISIONS``), what this run
    used, and the evidence the plan keeps: the 2 x 2 plant matrix with its panels (``choices``, ``g2``'s
    ``reported_not_gated.plant_choices``) and, for the rule, the unperturbed run beside the panel."""
    used = dict(ankle_gains=choices[gated]["options"]["ankle_gains"],
                ankle_armature=choices[gated]["options"]["ankle_armature"], g2_rule=rule)
    matrix = {name: dict(ankle_gains=c["options"]["ankle_gains"], ankle_armature=c["options"]["ankle_armature"],
                         unperturbed_strides=c["nominal_strides"], unperturbed_fall_cause=c["nominal_fall_cause"],
                         panel_median_strides=c["panel"]["median"], panel_min_strides=c["panel"]["min"],
                         panel_max_strides=c["panel"]["max"],
                         panel_runs_at_least_min_strides=c["panel"]["runs_at_least_min_strides"],
                         panel_runs=c["panel"]["runs"], single_rule_pass=c["single_rule_pass"],
                         panel_rule_pass=c["panel_rule_pass"], used_here=name == gated)
              for name, c in choices.items()}
    rows = []
    for number, dec in DECISIONS.items():
        rows.append(dict(decision=number, source=f"{DECISIONS_SOURCE}, decision {number}", choice=dec["choice"],
                         title=dec["title"], taken=dec["taken"], used=used[dec["choice"]],
                         follows=used[dec["choice"]] == dec["taken"], reason=dec["reason"],
                         evidence=("rule_evidence" if dec["choice"] == "g2_rule" else "plant_matrix")))
    g = choices[gated]
    return dict(source=f"{DECISIONS_SOURCE}, decisions {min(DECISIONS)}-{max(DECISIONS)}",
                note="decided on reasons that do not depend on the gate's outcome; the results with the other choices "
                     "stay recorded as evidence (here, and every panel run in reported_not_gated.plant_choices)",
                decisions=rows, follows_all=all(r["follows"] for r in rows), plant_matrix=matrix,
                rule_evidence=dict(rules={r: RULE_TEXT[r] for r in RULES}, used=rule,
                                   unperturbed_strides=g["nominal_strides"], panel_median_strides=g["panel"]["median"],
                                   panel_min_strides=g["panel"]["min"], panel_max_strides=g["panel"]["max"],
                                   single_rule_pass=g["single_rule_pass"], panel_rule_pass=g["panel_rule_pass"]))


def g2(body: str, out_dir: Path, strides: float = REPLAY_STRIDES, rule: str = RULES[0], procs: int | None = None,
       **plant_kw) -> dict:
    """G2 from the references saved in ``out_dir`` (``refs.prepare``), on the plant of ``plant_kw``."""
    if rule not in RULES:
        raise ValueError(f"rule {rule!r}: choose from {', '.join(RULES)}")
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    doc = json.loads((out_dir / "refs.json").read_text())
    checks = {name: dict(matches=e["matches_records"],
                         records=[dict(record=c["record"], leaves=c["leaves"], max_rel_diff=c["max_rel_diff"],
                                       n_mismatched=c["n_mismatched"], n_missing=c["n_missing"],
                                       n_noise_floor_matches=c.get("n_noise_floor_matches"))
                                  for c in e["record_checks"]])
              for name, e in doc["references"].items()}
    nom = doc["references"]["nominal"]
    path = {name: str(out_dir / e["file"]) for name, e in doc["references"].items()}
    gated_plant = plant.build(body, "hard", settle=False, **plant_kw)
    options = gated_plant.meta["options"]
    gated = next(c for c, o in PLANT_CHOICES.items()
                 if o["ankle_gains"] == options["ankle_gains"] and o["ankle_armature"] == options["ankle_armature"])
    base = {k: v for k, v in plant_kw.items() if k not in ("ankle_gains", "ankle_armature", "envelope")}
    jobs, keys = [], []

    def add(key, ref, kw, offsets, **opt):
        jobs.append(dict(body=body, ref=path[ref], plant_kw=kw, offsets=offsets, strides=strides, **opt))
        keys.append(key)

    for name in doc["references"]:
        for flag in (True, False):
            add(("replay", name, flag), name, dict(plant_kw, envelope=flag), [None], legs=flag)
    add(("no_armature_ff",), "nominal", dict(plant_kw, envelope=True), [None], armature_ff=False)
    offsets = _offsets()
    for cname, copt in PLANT_CHOICES.items():
        kw = dict(base, **copt, envelope=True)
        add(("choice", cname), "nominal", kw, [None])
        for i in range(0, PANEL_RUNS, PANEL_CHUNK):
            add(("panel", cname, i), "nominal", kw, offsets[i : i + PANEL_CHUNK], full=False)
    results = dict(zip(keys, run_jobs(jobs, procs), strict=True))
    rep = {name: {("with_envelope" if flag else "without_envelope"): results[("replay", name, flag)][0]
                  for flag in (True, False)} for name in doc["references"]}
    choices = {}
    for cname, copt in PLANT_CHOICES.items():
        runs = [r for i in range(0, PANEL_RUNS, PANEL_CHUNK) for r in results[("panel", cname, i)]]
        single = results[("choice", cname)][0]
        pnl = panel_stats([r["strides"] for r in runs])
        pnl["fall_causes"] = sorted({r["fall_cause"] for r in runs if r["fall_cause"]})
        choices[cname] = dict(options=dict(base, **copt), gated=cname == gated, nominal_strides=single["strides"],
                              nominal_fall_cause=single["fall_cause"], panel=pnl,
                              single_rule_pass=bool(single["strides"] >= MIN_REPLAY_STRIDES),
                              panel_rule_pass=pnl["median_at_least_min_strides"])
    walked = rep["nominal"]["with_envelope"]["strides"]
    panel = choices[gated]["panel"]
    ranges = {name: reference_range_margins(gated_plant, Reference.load(path[name])) for name in doc["references"]}
    walking_turns = {name: dict(with_envelope=v["with_envelope"]["strides"],
                                without_envelope=v["without_envelope"]["strides"])
                     for name, v in rep.items() if name.startswith("walk_")}
    info = dict(
        plant_choices=choices,
        deterministic_across_processes=bool(choices[gated]["nominal_strides"] == walked),
        nominal_without_armature_feedforward=results[("no_armature_ff",)][0],
        reference_range_margins=ranges,
        min_range_rad=MIN_RANGE_RAD)
    out = dict(rule=rule, rule_text=RULE_TEXT[rule], plant_options=options,
               references_match=checks, all_match=all(c["matches"] for c in checks.values()),
               nominal_speed_m_s=round(nom["speed_m_s"], 6),
               nominal_min_cop_margin_single_m=round(nom["min_cop_margin_single_m"], 6),
               replay=rep, walking_turns=walking_turns, nominal_strides_with_envelope=walked,
               nominal_strides_without_envelope=rep["nominal"]["without_envelope"]["strides"],
               design_study_strides_screened_model=DESIGN_STUDY_STRIDES.get(body), min_strides=MIN_REPLAY_STRIDES,
               nominal_panel_with_envelope=panel, single_run_pass=bool(walked >= MIN_REPLAY_STRIDES),
               panel_pass=panel["median_at_least_min_strides"], reported_not_gated=info,
               decided=decided(choices, gated, rule))
    out["pass"] = bool(out["all_match"] and (out["single_run_pass"] if rule == "single" else out["panel_pass"]))
    out["seconds"] = time.perf_counter() - t0
    return out


# ------------------------------------------------------------------------------------------------
# G3: transitions
# ------------------------------------------------------------------------------------------------
def g3(body: str, out_dir: Path) -> dict:
    """G3 (plan section 3 step 4) from ``transitions.json`` (``transitions.prepare``): every transition's last output
    stride within ``transitions.G3_COM_MM`` of the periodic walk's CoM and ``G3_ZMP_MM`` of its multibody ZMP, every
    frame's IK solved.  The transitions that fail would fall back to Design 3's amplitude ramps (``fallback``)."""
    from . import transitions as tr

    doc = json.loads((Path(out_dir) / "transitions.json").read_text())
    rows = {}
    for name, e in doc["transitions"].items():
        rows[name] = dict(from_=e["from_"], to=e["to"], kind=e["kind"], pass_=e["g3"]["pass_"],
                          com_max_mm=e["g3"]["com_max_mm"], zmp_max_mm=e["g3"]["zmp_max_mm"], ik_ok=e["ik_ok"],
                          first_stride_com_max_mm=e["join_a"]["com_max_mm"],
                          first_stride_zmp_max_mm=e["join_a"]["zmp_max_mm"], join_first_frame=e["join_a"]["at_join"],
                          join_last_frame=e["join_b"]["at_join"], zmp_error_final_m=e["zmp_error_history_m"][-1],
                          min_cop_margin_single_m=e["min_cop_margin_single_m"], stop=e["stop"], file=e["file"],
                          sha256=e["sha256"])
    missing = [n for n in tr.TRANSITIONS if n not in rows]
    failed = [n for n, r in rows.items() if not r["pass_"]]
    worst_c = max(rows, key=lambda n: rows[n]["com_max_mm"])
    worst_z = max(rows, key=lambda n: rows[n]["zmp_max_mm"])
    return dict(rule=f"the last output stride of each transition within {tr.G3_COM_MM:g} mm of the periodic walk's CoM "
                     f"(3D) and {tr.G3_ZMP_MM:g} mm of its multibody ZMP (the home stance for a stop), every frame's IK "
                     "solved; else that transition falls back to Design 3's amplitude ramps (plan section 3 step 4)",
                method=doc.get("method"), transitions=rows, missing=missing, failed=failed, fallback=failed,
                worst_com=dict(transition=worst_c, mm=rows[worst_c]["com_max_mm"]),
                worst_zmp=dict(transition=worst_z, mm=rows[worst_z]["zmp_max_mm"]),
                stops=stop_criterion(rows), pass_=bool(not failed and not missing))


#: Plan section 4, Start and stop: "stopped within 1.5 strides (double support or a step allowed)", read as
#: ``STAGE1_PLAN.md`` decision 18 reads it (``transitions.STOP_RULE``).
STOP_WITHIN_STRIDES = 1.5


def stop_criterion(rows: dict) -> dict:
    """Reported with G3, not gated: each stop reference's own stop (``transitions.stop_time``) under decision 18: the
    forward speed below 5 % of the command within 1.5 strides of the stop command (``within``; an in-place turn has
    no forward command, None), and the time to rest, reported.  A perfect tracker of a reference that is not within
    would fail the behaviour."""
    from . import transitions as tr

    out = {}
    for name, r in rows.items():
        s = r.get("stop")
        if not s:
            continue
        fwd = s.get("forward_stopped_strides")
        out[name] = dict(forward_stopped_strides=fwd, within=None if fwd is None else bool(fwd <= STOP_WITHIN_STRIDES),
                         at_rest_strides=s.get("at_rest_strides"), at_rest_s=s.get("at_rest_s"),
                         command_speed_m_s=s.get("command_speed_m_s"), command_phase=s.get("command_phase"))
    return dict(rule=tr.STOP_RULE, source="STAGE1_PLAN.md, section 9, decision 18", limit_strides=STOP_WITHIN_STRIDES,
                speed_frac=tr.STOP_SPEED_FRAC, stops=out,
                every_within=all(v["within"] for v in out.values() if v["within"] is not None))


# ------------------------------------------------------------------------------------------------
# G4: the tracker
# ------------------------------------------------------------------------------------------------
G4_STRIDES = 60.0  # run length of the gated runs (strides), judged against the thresholds
G4_THRESHOLDS = {"R1_nominal_hard": 40.0, "R1_nominal_mu054": 20.0, "R2_nominal_hard": 20.0}
G4_MU = 0.54
G4_REPORTED = ("brisk", "fr05", "walk_0.35_left", "walk_0.35_right", "in_place_0.50_left", "in_place_0.50_right")
G4_REPORTED_STRIDES = 40.0
G4_NOISE_SEEDS = 0  # R2's sensor-noise seeds: G4_NOISE_SEEDS + run index (not the plan's tuning or panel seeds)
G4_STANCE_STRIDES = 20.0
#: ``STAGE1_PLAN.md`` decision 17: G4 is judged at the tuning start (decision 16) with decision 13's panel rule.
G4_GAINS = "start"
G4_RULES = ("panel", "single")
#: The rule that decides G4's ``pass`` (decision 17: the first of ``G4_RULES``); the other is reported.
G4_RULE = G4_RULES[0]
G4_RULE_TEXT = {
    "panel": "the median of 20 runs of each gated condition from the reference's first frame, the root velocity offset "
             f"in x and y by at most {1e3 * PANEL_OFFSET_M_S:g} mm/s (seed {PANEL_SEED}, G2's offsets; R2's sensor "
             f"noise seeded {G4_NOISE_SEEDS} + run), {G4_STRIDES:g} strides long, reaches its threshold (R1 on hard "
             f"floor {G4_THRESHOLDS['R1_nominal_hard']:g} strides, R1 at friction {G4_MU} "
             f"{G4_THRESHOLDS['R1_nominal_mu054']:g}, R2 on hard floor {G4_THRESHOLDS['R2_nominal_hard']:g}), the "
             "tracker at the tuning start (STAGE1_PLAN.md decisions 16, 17 and 13)",
    "single": "the one unperturbed run of each gated condition reaches its threshold (reported)",
}
#: Decision 20's attribution of a failing condition's falls (``attribute``).
ATTRIBUTION_ALPHA = 0.05
ATTRIBUTION_NULL_M_S = 1e-3
ATTRIBUTION_RULE = (
    "STAGE1_PLAN.md decision 20: a condition that fails is body-limited by a hardware limit only if lifting that "
    "limit from the start (control.relaxed), over the condition's 20 starts (the same root-velocity offsets and noise "
    "seeds), improves survival significantly: a one-sided exact sign test on the paired survival times (strides to "
    "the fall; a run that does not fall outlasts one that does; two runs that do not fall tie; ties are dropped), "
    f"p < {ATTRIBUTION_ALPHA:g}.  The limits tested are the relaxable hardware limits that are candidates of at least "
    f"one of the condition's falls (control.CANDIDATE_RULE); otherwise its falls are controller-limited.  The null, the "
    f"same starts with the root velocity changed by a further {1e3 * ATTRIBUTION_NULL_M_S:g} mm/s in x and nothing "
    "lifted, is reported as a chaos diagnostic and does not change the label.  "
    "Candidates the plant cannot lift (the CoP at a pad edge: the pads' size) are listed; lifting the candidates "
    "together is reported, not the label")
#: The decisions G4 follows (``STAGE1_PLAN.md`` section 9, "Decisions taken during the build (steps 4-5, 9 Nov)").
G4_DECISIONS_SOURCE = "STAGE1_PLAN.md, section 9, 'Decisions taken during the build (steps 4-5, 9 Nov)'"
G4_DECISIONS = {
    15: ("ankle_cap", "box+pad", "Ankle torque in the tracker: each axis at most one motor's peak, a loaded foot's CoP "
                                 "5 mm inside its pad"),
    16: ("gains", "start", "Tuning start: the prototype's terms, heading and phase-rate gains zero"),
    17: ("g4_rule", "panel", "G4 judged at that start with decision 13's panel rule"),
    18: ("stop_reading", "forward_5pct_of_command", "Stopped within 1.5 strides: forward speed below 5 % of the "
                                                    "command within 1.5 strides of the stop command"),
    19: ("estimator_stance_point", estimator.STANCE_POINTS[0], "Estimator stance point: the loaded pad's contact point"),
    20: ("attribution_and_inward_bound", "panel_attribution+pad_clearance",
         "Fall attribution over the condition's starts; the inward bound the swing pad 5 mm from the stance pad"),
}


def g4_decided(used: dict) -> dict:
    """Decisions 15-20 as the plan took them and what this G4 run used (``used``: choice -> value)."""
    rows = [dict(decision=n, source=f"{G4_DECISIONS_SOURCE}, decision {n}", choice=c, title=title, taken=taken,
                 used=used.get(c), follows=used.get(c) == taken) for n, (c, taken, title) in G4_DECISIONS.items()]
    return dict(source=f"{G4_DECISIONS_SOURCE}, decisions 15-20", decisions=rows,
                follows_all=all(r["follows"] for r in rows))


def _compact(r: dict) -> dict:
    """A panel run's outcome: strides, the fall and its candidates, how often the ankle cap and the inward bound bound,
    how far the realised ankle torques broke the cap, the estimator's errors (over the episode and before the
    fall)."""
    out = {k: r.get(k) for k in ("strides", "fell", "fall_cause", "fall_candidates", "dcm_error_rms_mm", "bound_frac",
                                 "relaxed")}
    c = r.get("ankle_cap_residual")
    if c:  # the realised ankle torques against the cap (control.CAP_RESIDUAL_RULE)
        out["ankle_cap_residual"] = {k: c[k] for k in ("box_frac", "box_max_excess_Nm", "pad_frac", "pad_max_excess_Nm",
                                                       "heavy_pad_frac", "heavy_pad_max_excess_Nm")}
    e = r.get("estimator")
    if e:
        out.update(estimator_com_vel_err_rms_m_s=e["com_vel_err_rms_m_s"],
                   estimator_com_rel_feet_err_rms_mm=e["com_rel_feet_err_rms_mm"])
        b = e.get("before_fall")
        if b and b.get("updates"):
            out.update(estimator_before_fall_com_vel_err_rms_m_s=b["com_vel_err_rms_m_s"],
                       estimator_before_fall_com_rel_feet_err_rms_mm=b["com_rel_feet_err_rms_mm"])
    return out


def _g4_job(job: dict) -> list[dict]:
    """Worker: one plant and track, the episodes of ``job`` (``g4``)."""
    from . import control

    p = plant.build(job["body"], "hard", settle=job.get("start") == "stance")
    if job.get("mu") is not None:
        control.set_friction(p, job["mu"])
    first = Reference.load(Path(job["run"]) / "refs" / "nominal.npz")
    omega = control.omega0(p, first.q_home)
    track = control.build_track(p, job["run"], job["plan"], omega)
    tk = control.Tracker(p, track, job["mode"], gains=job.get("gains", G4_GAINS), terms=job.get("terms", control.TERMS),
                         inward_bound=job.get("inward_bound", control.INWARD_BOUNDS[0]),
                         ankle_cap=job.get("ankle_cap", control.ANKLE_CAP))
    out = []
    period = Reference.load(Path(job["run"]) / "refs" / f"{job['period_ref']}.npz").period
    for off, seed in zip(job["offsets"], job["seeds"], strict=True):
        r = control.run(tk, job["seconds"], start=job.get("start", "frame"), velocity_offset=off, seed=seed,
                        period=period, relax=job.get("relax", ()))
        out.append(r if job.get("full", True) else _compact(r))
    return out


def _pool_map(jobs: list[dict], procs: int | None) -> list:
    procs = min(procs or os.cpu_count() or 1, len(jobs))
    if procs <= 1:
        return [_g4_job(j) for j in jobs]
    with mp.get_context("spawn").Pool(procs) as pool:
        return pool.map(_g4_job, jobs, chunksize=1)


def _panel(rows: list[dict], threshold: float) -> dict:
    v = np.array([r["strides"] for r in rows])
    cands = {}
    for r in rows:
        for c in r.get("fall_candidates") or ():
            cands[c] = cands.get(c, 0) + 1
    return dict(runs=len(v), strides=[float(x) for x in v], median=float(np.median(v)), min=float(v.min()),
                max=float(v.max()), runs_at_least_threshold=int(np.sum(v >= threshold)), threshold=threshold,
                median_at_least_threshold=bool(np.median(v) >= threshold), falls=int(sum(r["fell"] for r in rows)),
                survived=int(sum(not r["fell"] for r in rows)), fall_candidates=cands,
                fall_causes=sorted({r["fall_cause"] for r in rows if r["fall_cause"]}))


def sign_test(base: list[dict], other: list[dict]) -> dict:
    """Paired one-sided exact sign test that ``other`` survives longer than ``base`` (runs pairwise; each a dict with
    ``fell`` and ``strides``): a run that does not fall outlasts one that does, two that do not fall tie, two falls
    compare their strides; ties are dropped; p = P(Binomial(better + worse, 1/2) >= better)."""
    better = worse = tied = 0
    for a, b in zip(base, other, strict=True):
        if not a["fell"] and not b["fell"]:
            tied += 1
        elif a["fell"] != b["fell"]:
            better, worse = (better + 1, worse) if a["fell"] else (better, worse + 1)
        elif b["strides"] > a["strides"]:
            better += 1
        elif b["strides"] < a["strides"]:
            worse += 1
        else:
            tied += 1
    n = better + worse
    p = sum(math.comb(n, k) for k in range(better, n + 1)) / 2**n if n else 1.0
    return dict(better=better, worse=worse, tied=tied, p=float(p))


def attribute(baseline: list[dict], lifted: dict, null: list[dict] | None, candidates: list[str],
              together=None) -> dict:
    """Decision 20 (``ATTRIBUTION_RULE``) on a failing condition: ``baseline`` its 20 runs, ``lifted`` relaxation ->
    the same starts with that hardware limit lifted from the start, ``null`` the starts changed by 1 mm/s (None: not
    run; it is a diagnostic and does not change the label), ``candidates`` the hardware limits its falls had,
    ``together`` the starts with every relaxable candidate lifted."""
    from . import control

    def side(rows):
        return dict(strides=[float(r["strides"]) for r in rows], survived=int(sum(not r["fell"] for r in rows)),
                    median=float(np.median([r["strides"] for r in rows])))

    tests = {r: dict(side(rows), sign_test=sign_test(baseline, rows)) for r, rows in lifted.items()}
    for t in tests.values():
        t["significant"] = bool(t["sign_test"]["p"] < ATTRIBUTION_ALPHA)
    nt = None
    if null is not None:
        nt = dict(side(null), offset_m_s=ATTRIBUTION_NULL_M_S, sign_test=sign_test(baseline, null))
        nt["significant"] = bool(nt["sign_test"]["p"] < ATTRIBUTION_ALPHA)
    binding = [r for r, t in tests.items() if t["significant"]]
    label = "body-limited: " + ", ".join(binding) if binding else "controller-limited"  # decision 20; the null is reported
    tog = None
    if together is not None:
        tog = dict(side(together), lifted=sorted(lifted), sign_test=sign_test(baseline, together))
        tog["significant"] = bool(tog["sign_test"]["p"] < ATTRIBUTION_ALPHA)
    return dict(rule=ATTRIBUTION_RULE, alpha=ATTRIBUTION_ALPHA, baseline=side(baseline), candidates=list(candidates),
                relaxable={c: control.RELAXES[c] for c in candidates if control.RELAXES[c]},
                not_relaxable=[c for c in candidates if control.RELAXES[c] is None], lifted=tests, null=nt,
                together=tog, binding=binding, label=label)


def g4(body: str, out_dir: Path, procs: int | None = None, attribution: bool = True) -> dict:
    """G4 (plan section 3 step 5) on the saved references and transitions of ``out_dir``, the tracker at the tuning
    start (decisions 15-20): R1 walks the nominal walk at least 40 strides on hard floor and 20 at friction 0.54, R2
    at least 20 on hard floor, judged by the 20-start panel's median (``G4_RULE_TEXT``, decision 17); the unperturbed
    run of each, from the reference's first frame as the design-study prototype ran, is reported beside it.  Each
    failing condition is attributed by decision 20 (``attribute``) when ``attribution``.  Reported, not gated: R0's
    strides, R1 and R2 on the brisk walk, the Fr 0.05 walk and the target turns, a start-walk-stop from the settled
    stance with its stop read by decision 18, what the turn gap is (``control.turn_gap_info``) and what the inward
    bound allows (``control.inward_step_clearance``).  Every run records how far its realised ankle torques broke the
    cap (``control.CAP_RESIDUAL_RULE``); ``decided`` reads what the run used from the runs and the rule that decided
    ``pass`` (``G4_RULE``; the estimator's stance point from the R2 runs' records)."""
    from . import control, estimator
    from . import transitions as tr

    t0 = time.perf_counter()
    run_dir = Path(out_dir)
    refs_doc = json.loads((run_dir / "refs.json").read_text())
    period = {n: e["period_s"] for n, e in refs_doc["references"].items()}
    offsets = _offsets()
    jobs, keys = [], []

    def add(key, mode, ref, strides, offs, seeds, **kw):
        plan = kw.pop("plan", None) or [("ref", ref, strides + 1.0)]
        jobs.append(dict(body=body, run=str(run_dir), mode=mode, plan=plan, period_ref=kw.pop("period_ref", ref),
                         seconds=strides * period[kw.pop("period_name", ref)], offsets=offs, seeds=seeds, **kw))
        keys.append(key)

    def add_panel(prefix, mode, mu, offs, **kw):
        for i in range(0, PANEL_RUNS, PANEL_CHUNK):
            add(prefix + (i,), mode, "nominal", G4_STRIDES, offs[i : i + PANEL_CHUNK],
                [G4_NOISE_SEEDS + j for j in range(i, i + PANEL_CHUNK)], mu=mu, full=False, **kw)

    gated = {"R1_nominal_hard": ("R1", None), "R1_nominal_mu054": ("R1", G4_MU), "R2_nominal_hard": ("R2", None)}
    for key, (mode, mu) in gated.items():
        add(("gated", key), mode, "nominal", G4_STRIDES, [None], [G4_NOISE_SEEDS], mu=mu)
        add_panel(("panel", key), mode, mu, offsets)
    for name in ("nominal",) + G4_REPORTED:
        add(("R0", name), "R0", name, G4_REPORTED_STRIDES if name != "nominal" else G4_STRIDES, [None], [0])
    for mode in ("R1", "R2"):
        for name in G4_REPORTED:
            add(("reported", mode, name), mode, name, G4_REPORTED_STRIDES, [None], [G4_NOISE_SEEDS])
        stance_plan = [("hold", 1.0), ("transition", "start"), ("ref", "nominal", G4_STANCE_STRIDES),
                       ("transition", "stop"), ("hold", 2.0)]
        dur = 1.0 + G4_STANCE_STRIDES * period["nominal"] + 2.0 + 2 * 5 * period["nominal"]
        add(("stance", mode), mode, "nominal", dur / period["nominal"], [None], [G4_NOISE_SEEDS], start="stance",
            plan=stance_plan)
    R = dict(zip(keys, _pool_map(jobs, procs), strict=True))

    def rows_of(prefix, key):
        return [r for i in range(0, PANEL_RUNS, PANEL_CHUNK) for r in R[prefix + (key, i)]]

    gated_out = {}
    for key in gated:
        single = R[("gated", key)][0]
        thr = G4_THRESHOLDS[key]
        rows = rows_of(("panel",), key)
        gated_out[key] = dict(threshold_strides=thr, panel=dict(_panel(rows, thr), rows=rows),
                              panel_pass=bool(np.median([r["strides"] for r in rows]) >= thr), unperturbed=single,
                              unperturbed_pass=bool(single["strides"] >= thr))
    # decision 20: the failing conditions' attribution, over the same starts
    attr = {}
    if attribution:
        jobs, keys = [], []
        plans = {}
        for key, (mode, mu) in gated.items():
            g = gated_out[key]
            if g["panel_pass"]:
                continue
            cands = [c for c in control.HW_LIMITS if any(c in (r["fall_candidates"] or ()) for r in g["panel"]["rows"])]
            lift = [r for r in control.RELAXATIONS if any(control.RELAXES[c] == r for c in cands)]
            plans[key] = (cands, lift)
            for r in lift:
                add_panel(("lift", key, r), mode, mu, offsets, relax=(r,))
            add_panel(("null", key), mode, mu, [o + np.array([ATTRIBUTION_NULL_M_S, 0.0, 0.0]) for o in offsets])
            if len(lift) > 1:
                add_panel(("together", key), mode, mu, offsets, relax=tuple(lift))
        if jobs:
            R.update(zip(keys, _pool_map(jobs, procs), strict=True))
        for key, (cands, lift) in plans.items():
            attr[key] = attribute(gated_out[key]["panel"]["rows"], {r: rows_of(("lift", key), r) for r in lift},
                                  rows_of(("null",), key), cands,
                                  rows_of(("together",), key) if len(lift) > 1 else None)
    reported = {mode: {name: R[("reported", mode, name)][0] for name in G4_REPORTED} for mode in ("R1", "R2")}
    r0 = {name: R[("R0", name)][0] for name in ("nominal",) + G4_REPORTED}
    stance = {mode: R[("stance", mode)][0] for mode in ("R1", "R2")}
    tk_doc = dict(gains={n: dict(start=float(s_), earlier_default=float(d_), bounds=[float(lo), float(hi)], note=note)
                         for n, s_, d_, lo, hi, note in control.GAIN_SPEC},
                  gain_set=G4_GAINS, gains_source=control.GAINS_SOURCE[G4_GAINS],
                  terms=list(control.TERMS), feedback_hz=1.0 / (control.FEEDBACK_EVERY * 0.002),
                  feedback_hold=control.FEEDBACK_HOLD, cop_inset_m=control.COP_INSET,
                  ankle_cap=dict(parts=list(control.ANKLE_CAP), contact_load_frac=control.CONTACT_LOAD_FRAC,
                                 rule="decision 15: each ankle's total torque command (feedforward, the servo's PD "
                                      "predicted from the encoders, the feedback) at most one motor's peak on each "
                                      "axis, and on a loaded foot no more than puts its CoP 5 mm inside its pad; the "
                                      "nearest such torque is commanded (control.cap_ankle_torque)",
                                 residual=control.CAP_RESIDUAL_RULE),
                  inward_bound=dict(bound=control.INWARD_BOUNDS[0], pad_clearance_m=control.PAD_CLEARANCE,
                                    rule="decision 20: the swing pad, at its reference touchdown relative to the stance "
                                         "foot moved by the step offset, stays at least 5 mm from the stance pad "
                                         "(control.inward_room); the turn gap bounds nothing",
                                    clearance=control.inward_step_clearance(body, run_dir)),
                  turn_gap=dict(cases=control.turn_gap_info(body), turn_gap_m=control.turn_gap(body),
                                counted_as_hardware_limit=False, bounds_the_step=False,
                                note="the turning screen's turn gap is the pelvis plate's clearance to the same-side "
                                     "thigh in the turning pose, not a gap between the legs (decision 20)"),
                  phase_rate_limit=control.PHASE_RATE_LIMIT,
                  estimator=dict(gains=estimator.EstimatorGains().__dict__, noise=estimator.SensorNoise().describe(),
                                 stance_point="decision 19: the loaded pad's contact point (the mean of the foot's pad "
                                              "bottom corners within contact_tol_m of its lowest one)",
                                 contact_tol_m=estimator.CONTACT_TOL),
                  falls=dict(onset_rad=control.FALL_ONSET_RAD, window_s=control.LIMIT_WINDOW_S,
                             hardware_limits=list(control.HW_LIMITS), reported_limits=list(control.REPORTED_LIMITS),
                             relaxations=list(control.RELAXATIONS), relaxes=control.RELAXES,
                             relaxed_friction=control.CF_FRICTION, relaxed_range_rad=control.CF_RANGE_RAD,
                             slip_speed_m_s=control.SLIP_SPEED, slip_load_frac=control.SLIP_LOAD_FRAC,
                             candidates=control.CANDIDATE_RULE, attribution=ATTRIBUTION_RULE),
                  stop_reading=tr.STOP_RULE)
    out = dict(rule=G4_RULE_TEXT[G4_RULE], rule_name=G4_RULE, rules=G4_RULE_TEXT, judged_at=G4_GAINS,
               tracker=tk_doc, gated=gated_out, attribution=attr, R0=r0, reported=reported, stance_start_stop=stance,
               single_rule_pass=bool(all(g["unperturbed_pass"] for g in gated_out.values())),
               panel_rule_pass=bool(all(g["panel_pass"] for g in gated_out.values())))
    out["pass"] = out[f"{G4_RULE}_rule_pass"]
    # what this run used, read from the runs and the rule that decided ``pass`` (decisions 17 and 19 included)
    r2_runs = [gated_out["R2_nominal_hard"]["unperturbed"], stance["R2"], *reported["R2"].values()]
    points = sorted({(r.get("estimator") or {}).get("gains", {}).get("stance_point") for r in r2_runs} - {None})
    used = dict(ankle_cap="+".join(control.ANKLE_CAP), gains=G4_GAINS, g4_rule=out["rule_name"],
                stop_reading="forward_5pct_of_command" if tr.STOP_SPEED_FRAC == 0.05 else str(tr.STOP_SPEED_FRAC),
                estimator_stance_point=points[0] if len(points) == 1 else "+".join(points) or None,
                attribution_and_inward_bound="panel_attribution+pad_clearance" if control.PAD_CLEARANCE == 0.005
                else "other")
    out["decided"] = g4_decided(used)
    out["seconds"] = time.perf_counter() - t0
    return out
