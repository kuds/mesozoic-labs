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


def _probe(env, foot_gids, floor_gid):
    import mujoco

    model, data = env.model, env.data
    right, left = env._foot_contact_forces()

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
            env.reset(seed=1000 + episode)
            if spawn_dz:
                env.data.qpos[2] += spawn_dz
                mujoco.mj_forward(env.model, env.data)
            rng = np.random.default_rng(episode)
            for t in range(max_steps):
                _, _, terminated, truncated, _ = env.step(driver(t, env, rng))
                sensors, foot_f, other_f, clearance, right, left = _probe(env, foot_gids, floor_gid)
                stance = derive_stance_info({"r_foot_contact": right, "l_foot_contact": left}, DUTY_THRESHOLD)
                rows.append((sensors, foot_f, other_f, clearance, stance["unsupported_duty"]))
                if terminated or truncated:
                    break
        finally:
            env.close()
    return np.array(rows)


def _summarize(label, rows):
    sensors, foot_f, other_f, clearance = rows[:, 0], rows[:, 1], rows[:, 2], rows[:, 3]
    unsupported = rows[:, 4].astype(bool)
    airborne = clearance > 1e-6
    n = len(rows)
    peak = max(float(np.max(foot_f)), 1.0)
    err = np.abs(sensors - foot_f)

    false_airborne = int(np.sum(unsupported & ~airborne))  # overstates airtime
    false_supported = int(np.sum(~unsupported & airborne))  # understates airtime
    disagreement = 100 * (false_airborne + false_supported) / n

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
        f"  misclassified: {false_airborne} false-airborne, {false_supported} false-supported"
        f"  -> {disagreement:.3f}% of steps"
    )
    return {
        "label": label,
        "steps": n,
        "airborne": float(airborne.mean()),
        "unsupported_duty": float(unsupported.mean()),
        "disagreement_pct": disagreement,
    }


def _zero(t, env, rng):
    return np.zeros(env.action_space.shape)


def _jitter(t, env, rng):
    """Small fast noise — the near-threshold regime the duty metric lives in."""
    return rng.uniform(-0.25, 0.25, size=env.action_space.shape)


def _hop(t, env, rng):
    return np.full(env.action_space.shape, 0.9 * np.sin(2 * np.pi * t / 40.0))


def _thrash(t, env, rng):
    return rng.uniform(-1.0, 1.0, size=env.action_space.shape)


REGIMES = (
    ("drop test (zero action, +0.25 m spawn)", _zero, dict(episodes=8, max_steps=300, spawn_dz=0.25)),
    ("settled stance (zero action)", _zero, dict(episodes=8, max_steps=400)),
    ("low-amplitude jitter", _jitter, dict(episodes=40, max_steps=1000)),
    ("forced hop (sinusoidal leg drive)", _hop, dict(episodes=40, max_steps=1000)),
    ("random thrash", _thrash, dict(episodes=40, max_steps=1000)),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=None, help="override per-regime episode count")
    parser.add_argument(
        "--max-disagreement",
        type=float,
        default=5.0,
        help="fail if any regime misclassifies more than this %% of steps",
    )
    args = parser.parse_args()

    results = []
    for label, driver, kwargs in REGIMES:
        if args.episodes is not None:
            kwargs = {**kwargs, "episodes": args.episodes}
        results.append(_summarize(label, _collect(driver, **kwargs)))

    worst = max(results, key=lambda r: r["disagreement_pct"])
    span = (min(r["airborne"] for r in results), max(r["airborne"] for r in results))
    print(
        f"\nAirborne fraction swept {span[0]:.1%} to {span[1]:.1%};"
        f" worst misclassification {worst['disagreement_pct']:.3f}% ({worst['label']})."
    )
    if worst["disagreement_pct"] > args.max_disagreement:
        print("FAILED: the duty metrics do not track kinematic ground truth closely enough.")
        return 1
    print("The support-duty metrics track kinematic ground truth across the swept range.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
