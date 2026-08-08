"""Validate the support-duty metrics against kinematic ground truth.

``bilateral_support_duty`` / ``single_support_duty`` / ``unsupported_duty``
(``stance_diagnostics.derive_stance_info``) classify each step by thresholding
the foot touch sensors at 0.1 N.  Those readings are the evidence behind the
claim that the stage-1 policy spends ~21% of its steps off the ground, so the
metric needs to be shown accurate before the behaviour is argued from it -- a
sensor that under-reports contact looks exactly like a foot in flight.

``foot_sensor_report.py`` checks the sensors on a SETTLED plant.  That is one
operating point: quiet bilateral stance, steady load.  This script checks the
regime the duty numbers actually come from -- touchdown transients, rapid load
transfer, real flight phases -- by sweeping driver policies that produce
airborne fractions from ~3% to ~67%, and comparing three signals per step, two
of which never touch the sensor path:

  A. summed foot touch sensors                     (what the duty metric reads)
  B. ``mj_contactForce`` over foot-geom floor contacts       (force reference)
  C. ``mj_geomDistance`` from every foot geom to the floor   (KINEMATIC truth)

C is the decisive one: it is computed from geometry alone, so if the steps
labelled ``unsupported`` are the steps where the feet are genuinely clear of
the ground, the metric measures what it claims.

Two traps this script exists to avoid, both hit while writing it:

* **Compare against foot contacts only.** Summing every floor contact counts
  tail and torso strikes as sensor disagreement and produces nonsense.
* **Enumerate the foot geoms from actual contacts, not from their names.** A
  ``foot|toe|metatarsus|pad`` name filter misses ``r_plantar_geom`` and
  ``l_plantar_geom``, which carry most of the load. That is the same mistake
  class as the sensor-scope defects in FOOT_SENSOR_VERIFICATION.md.

Usage::

    python environments/shared/scripts/stance_duty_validation.py
    python environments/shared/scripts/stance_duty_validation.py --episodes 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

#: Threshold the duty classifier uses, from ``derive_stance_info``.
DUTY_THRESHOLD = 0.1

#: Bodies whose geoms make up the feet.  Resolved to geoms at runtime so a
#: plant revision that adds a digit is picked up without editing a name list.
FOOT_BODY_MARKERS = ("plantar", "metatarsus", "toe")


def _foot_geom_ids(model) -> list[int]:
    import mujoco

    ids = []
    for gid in range(model.ngeom):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        haystack = f"{body or ''} {name}"
        if any(marker in haystack for marker in FOOT_BODY_MARKERS):
            ids.append(gid)
    return ids


def _foot_geom_ids_by_side(model) -> "tuple[list[int], list[int]]":
    """The foot geoms split right/left, for per-foot kinematic truth.

    The duty metric takes each foot's MIN force over the control step's
    substeps INDEPENDENTLY -- foot R unloading at substep 2 and foot L at
    substep 4 classifies as unsupported even though the animal was never
    simultaneously airborne.  That is the stricter, fail-closed certification
    ("each foot continuously loaded"), and the kinematic truth must use the
    same per-foot semantics or violent regimes read as misclassification.
    """
    import mujoco

    right: list[int] = []
    left: list[int] = []
    for gid in _foot_geom_ids(model):
        body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[gid]) or ""
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        side = body or name
        (right if side.startswith("r_") else left).append(gid)
    if not right or not left:
        raise ValueError("could not split foot geoms into r_/l_ sides; fix the name heuristic before trusting truth")
    return right, left


def _probe(env, foot_gids, floor_gid):
    import mujoco

    model, data = env.model, env.data
    # ``_foot_contact_forces`` returns ONE VALUE PER FOOT, so its arity is
    # species-dependent: 2 on the bipeds, 4 on the quadrupeds.  This probe is
    # bipedal by construction -- it feeds ``derive_stance_info``, which is
    # defined on an r/l pair -- so check rather than let a quadruped raise an
    # opaque unpacking error several frames away from the cause.
    forces = env._foot_contact_forces()
    if len(forces) != 2:
        raise ValueError(
            f"{type(env).__name__} reports {len(forces)} feet; this validation is bipedal "
            "(derive_stance_info takes an r/l pair). Generalise both before pointing it at a quadruped."
        )
    right, left = forces

    foot_force = other_force = 0.0
    for i in range(data.ncon):
        contact = data.contact[i]
        if floor_gid not in (contact.geom1, contact.geom2):
            continue
        other = contact.geom2 if contact.geom1 == floor_gid else contact.geom1
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        fz = abs(float((np.asarray(contact.frame).reshape(3, 3).T @ force[:3])[2]))
        if other in foot_gids:
            foot_force += fz
        else:
            other_force += fz

    clearance = min(float(mujoco.mj_geomDistance(model, data, gid, floor_gid, 2.0, None)) for gid in foot_gids)
    return float(right) + float(left), foot_force, other_force, clearance, float(right), float(left)


def _collect(driver, episodes, max_steps, spawn_dz=0.0):
    import mujoco

    from environments.shared.stance_diagnostics import derive_stance_info
    from environments.trex.envs.trex_env import TRexEnv

    rows = []
    for episode in range(episodes):
        env = TRexEnv(reset_noise_scale=0.10)
        try:
            floor_gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
            foot_gids = _foot_geom_ids(env.model)
            r_gids, l_gids = _foot_geom_ids_by_side(env.model)
            env.reset(seed=1000 + episode)
            if spawn_dz:
                env.data.qpos[2] += spawn_dz
                mujoco.mj_forward(env.model, env.data)
            rng = np.random.default_rng(episode)

            # Per-substep, per-foot kinematic ground truth, recorded through
            # the REAL step path via the base env's probe hook -- physics runs
            # frame_skip times per control step, and a boundary-only probe
            # samples the same instant the (pre-fix) metric did, making
            # within-control-step unloading invisible to metric and truth
            # alike.  A shadow env stepped substep-wise could drift from the
            # primary; the hook cannot.
            substep_clearances: list[tuple[float, float]] = []

            def _side_clearance(gids):
                return min(
                    float(mujoco.mj_geomDistance(env.model, env.data, gid, floor_gid, 2.0, None)) for gid in gids
                )

            def _probe_substep():
                substep_clearances.append((_side_clearance(r_gids), _side_clearance(l_gids)))

            env._substep_probe_hook = _probe_substep

            for t in range(max_steps):
                substep_clearances.clear()
                _, _, terminated, truncated, info = env.step(driver(t, env, rng))
                if len(substep_clearances) != env.frame_skip:
                    # A certification that silently measures nothing is worse
                    # than none: if the probe hook stops firing per substep,
                    # every truth column degenerates to False and the gates
                    # pass vacuously.
                    raise RuntimeError(
                        f"probe hook fired {len(substep_clearances)} times for a "
                        f"frame_skip={env.frame_skip} step; the per-substep ground "
                        "truth is not being recorded"
                    )
                sensors, foot_f, other_f, clearance, right_b, left_b = _probe(env, foot_gids, floor_gid)

                # The duty classification the gate certifies from: the
                # substep-MIN aggregated forces carried by the info keys.
                stance = derive_stance_info(
                    {"r_foot_contact": info["r_foot_contact"], "l_foot_contact": info["l_foot_contact"]},
                    DUTY_THRESHOLD,
                )
                # The classification the pre-fix metric would have made,
                # from the control-boundary sample alone.
                boundary = derive_stance_info({"r_foot_contact": right_b, "l_foot_contact": left_b}, DUTY_THRESHOLD)
                # Kinematic truth with the metric's own per-foot semantics:
                # each foot's clearance judged independently across substeps.
                r_unloaded = any(rc > 1e-6 for rc, _ in substep_clearances)
                l_unloaded = any(lc > 1e-6 for _, lc in substep_clearances)
                truth_unsupported = r_unloaded and l_unloaded
                # Physically airborne: BOTH feet clear at the SAME substep --
                # the quantity the seed-43 bounce hid from boundary sampling.
                any_airborne = any(rc > 1e-6 and lc > 1e-6 for rc, lc in substep_clearances)
                rows.append(
                    (
                        sensors,
                        foot_f,
                        other_f,
                        clearance,
                        stance["unsupported_duty"],
                        float(truth_unsupported),
                        # Aliased step: some substep was kinematically
                        # airborne but the boundary sample read as supported
                        # -- the exact blindness the aggregation closes.
                        float(any_airborne and not boundary["unsupported_duty"]),
                    )
                )
                if terminated or truncated:
                    break
        finally:
            env._substep_probe_hook = None
            env.close()
    return np.array(rows)


def _summarize(label, rows):
    sensors, foot_f, other_f = rows[:, 0], rows[:, 1], rows[:, 2]
    unsupported = rows[:, 4].astype(bool)
    # Kinematic truth with the metric's per-foot semantics: each foot judged
    # independently across the substeps (see _foot_geom_ids_by_side).
    airborne = rows[:, 5].astype(bool)
    aliased = rows[:, 6].astype(bool)  # simultaneous-airborne substep the boundary sample missed
    n = len(rows)
    peak = max(float(np.max(foot_f)), 1.0)
    err = np.abs(sensors - foot_f)

    # The two error directions are NOT symmetric for a ceiling-gated metric:
    # false-supported understates airtime and can certify a hopping policy
    # (the failure the substep aggregation exists to close); false-airborne
    # merely overstates it -- the fail-closed direction, inflated by grazing
    # instants (geometric contact bearing < 0.1 N) that a 5-instant MIN
    # samples more often than the old boundary probe did.
    false_airborne = int(np.sum(unsupported & ~airborne))  # overstates airtime (conservative)
    false_supported = int(np.sum(~unsupported & airborne))  # understates airtime (DANGEROUS)
    false_airborne_pct = 100 * false_airborne / n
    false_supported_pct = 100 * false_supported / n

    print(f"\n=== {label} ===")
    print(f"  steps {n:6}   kinematically airborne {airborne.mean():7.2%}   duty-unsupported {unsupported.mean():7.2%}")
    print(
        f"  sensors vs foot mj_contactForce: mean err {err.mean():8.3f} N, max {err.max():10.1f} N of {peak:.0f} N peak"
    )
    if (other_f > 1e-6).any():
        print(
            f"  non-foot floor contact on {(other_f > 1e-6).mean():.1%} of steps"
            f" (max {other_f.max():.1f} N) -- excluded"
        )
    print(
        f"  within-control-step aliasing: {int(aliased.sum())} steps ({aliased.mean():7.2%}) were airborne at some"
        f" substep while the boundary sample read supported -- caught by the substep-MIN aggregation"
    )
    print(
        f"  misclassified: {false_airborne} false-airborne ({false_airborne_pct:.3f}%, conservative), "
        f"{false_supported} false-supported ({false_supported_pct:.3f}%, dangerous)"
    )
    return {
        "label": label,
        "steps": n,
        "airborne": float(airborne.mean()),
        "unsupported_duty": float(unsupported.mean()),
        "aliased_pct": 100 * float(aliased.mean()),
        "false_airborne_pct": false_airborne_pct,
        "false_supported_pct": false_supported_pct,
    }


def _zero(t, env, rng):
    return np.zeros(env.action_space.shape)


def _jitter(t, env, rng):
    """Small fast noise — the near-threshold regime the duty metric lives in."""
    return rng.uniform(-0.25, 0.25, size=env.action_space.shape)


def _hop(t, env, rng):
    return np.full(env.action_space.shape, 0.9 * np.sin(2 * np.pi * t / 40.0))


def _hop20(t, env, rng):
    """Control-clock-locked hop: period 5 control steps = 20 Hz at 100 Hz control.

    The regime the seed-43 stage-1 bounce actually occupies -- a cycle
    commensurate with the frame_skip=5 substep structure, which is exactly
    where a boundary-sampled duty metric aliases (one sample per cycle lands
    wherever the phase puts it).  The 2.5 Hz ``_hop`` above cannot express
    this; every earlier certification of the metric swept only regimes where
    metric and truth were sampled at the same instant.
    """
    return np.full(env.action_space.shape, 0.9 * np.sin(2 * np.pi * t / 5.0))


def _thrash(t, env, rng):
    return rng.uniform(-1.0, 1.0, size=env.action_space.shape)


REGIMES = (
    ("drop test (zero action, +0.25 m spawn)", _zero, dict(episodes=8, max_steps=300, spawn_dz=0.25)),
    ("settled stance (zero action)", _zero, dict(episodes=8, max_steps=400)),
    ("low-amplitude jitter", _jitter, dict(episodes=40, max_steps=1000)),
    ("forced hop (sinusoidal leg drive)", _hop, dict(episodes=40, max_steps=1000)),
    ("control-locked hop (20 Hz, period-5)", _hop20, dict(episodes=40, max_steps=1000)),
    ("random thrash", _thrash, dict(episodes=40, max_steps=1000)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=None, help="override per-regime episode count")
    parser.add_argument(
        "--max-disagreement",
        type=float,
        default=5.0,
        help="fail if any regime UNDERSTATES airtime (false-supported) on more than this %% of steps",
    )
    parser.add_argument(
        "--max-conservatism",
        type=float,
        default=15.0,
        help=(
            "fail if any regime OVERSTATES airtime (false-airborne) on more than this %% of steps. "
            "Looser than --max-disagreement on purpose: a ceiling-gated duty that overstates can only "
            "fail policies conservatively, and the substep MIN samples grazing instants (geometric "
            "contact bearing < 0.1 N) five times as often as the old boundary probe"
        ),
    )
    args = parser.parse_args()

    results = []
    for label, driver, kwargs in REGIMES:
        if args.episodes is not None:
            kwargs = {**kwargs, "episodes": args.episodes}
        results.append(_summarize(label, _collect(driver, **kwargs)))

    # The 20 Hz regime is the aliasing certifier: if it stopped producing
    # airborne substeps (drive too weak for a revised plant, or episodes dying
    # immediately), its 0% false-supported is vacuous, not reassuring.
    control_locked = next(r for r in results if "control-locked" in r["label"])
    if control_locked["airborne"] < 0.10:
        print(
            f"FAILED: the control-locked regime measured only {control_locked['airborne']:.1%} airborne "
            "steps -- the aliasing certification is vacuous; retune the drive for this plant."
        )
        return 1

    worst_dangerous = max(results, key=lambda r: r["false_supported_pct"])
    worst_conservative = max(results, key=lambda r: r["false_airborne_pct"])
    span = (min(r["airborne"] for r in results), max(r["airborne"] for r in results))
    print(
        f"\nAirborne fraction swept {span[0]:.1%} to {span[1]:.1%}; worst understatement "
        f"{worst_dangerous['false_supported_pct']:.3f}% ({worst_dangerous['label']}), worst overstatement "
        f"{worst_conservative['false_airborne_pct']:.3f}% ({worst_conservative['label']})."
    )
    if worst_dangerous["false_supported_pct"] > args.max_disagreement:
        print("FAILED: the duty metrics UNDERSTATE airtime -- a hopping policy could certify as supported.")
        return 1
    if worst_conservative["false_airborne_pct"] > args.max_conservatism:
        print("FAILED: the duty metrics overstate airtime beyond the accepted conservatism bound.")
        return 1
    print("The support-duty metrics track kinematic ground truth across the swept range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
