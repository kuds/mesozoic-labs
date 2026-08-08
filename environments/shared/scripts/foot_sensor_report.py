"""Cross-check foot touch sensors against ``mj_contactForce`` and body weight.

A MuJoCo touch sensor sums only contacts on geoms belonging to its site's OWN
body, so a sensor mounted on a parent segment silently misses everything the
child geoms carry.  That has already produced two under-reads in this
repository -- the T-Rex repair (``aa87445``) and the raptor repair
(``aa3395c``) -- and it is invisible from the reward trace, because a sensor
reading zero looks exactly like a foot that is genuinely off the ground.

Two independent checks, both of which must hold on a settled plant:

* **sensor / contact = 1** -- the touch sensors see every floor contact.
* **contact / weight = 1** -- total floor reaction equals body weight.  This is
  a physics invariant at rest (and for the time-average of any periodic
  motion), so a deviation is an accounting bug rather than a policy result.

The weight denominator is the mass of the ANIMAL's kinematic subtree, not
``mj_getTotalmass``.  Several species carry a prey or food body -- on T-Rex the
prey is 65.45 kg of a 151.17 kg model total -- so dividing by the model mass
reports a false 0.57 on a plant standing in perfect equilibrium.

Usage::

    python environments/shared/scripts/foot_sensor_report.py
    python environments/shared/scripts/foot_sensor_report.py trex brachiosaurus
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from pathlib import Path

import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

SPECIES_ENVS = {
    "trex": ("environments.trex.envs.trex_env", "TRexEnv"),
    "velociraptor": ("environments.velociraptor.envs.raptor_env", "RaptorEnv"),
    "brachiosaurus": ("environments.brachiosaurus.envs.brachio_env", "BrachioEnv"),
    "dibothrosuchus": ("environments.dibothrosuchus.envs.dibothrosuchus_env", "DibothrosuchusEnv"),
}

# Keys the TOML [env] block carries for the MJX path only.
JAX_ONLY_ENV_KEYS = frozenset({"foot_contact_weight", "foot_contact_gate"})

#: Sensor attribute names to try, in order, per species.
_FOOT_SENSOR_ATTRS = (
    ("_sensor_r_foot", "_sensor_l_foot"),
    ("_sensor_fr_foot", "_sensor_fl_foot", "_sensor_rr_foot", "_sensor_rl_foot"),
)


def _foot_sensor_sum(env) -> float | None:
    """Sum the species' foot touch sensors, including any digit sensors."""
    import mujoco  # noqa: F401  (imported for parity with the env module)

    if hasattr(env, "_foot_contact_forces"):
        forces = env._foot_contact_forces()
        # The base class now defines this method for every species, returning
        # () when no _foot_sensor_groups are declared -- an empty tuple means
        # "no sensors", not "0.0 N of contact", so fall through instead of
        # reporting a spurious sensor-vs-force disagreement.
        if forces:
            return float(np.sum(forces))
    for group in _FOOT_SENSOR_ATTRS:
        if all(hasattr(env, attr) for attr in group):
            return float(sum(env.data.sensordata[getattr(env, attr)] for attr in group))
    return None


def _floor_normal_force(env) -> tuple[float, int]:
    """Total vertical floor reaction over every floor contact."""
    import mujoco

    ground = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    total, count = 0.0, 0
    for i in range(env.data.ncon):
        contact = env.data.contact[i]
        if ground != -1 and ground not in (contact.geom1, contact.geom2):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, i, force)
        frame = np.asarray(contact.frame).reshape(3, 3)
        total += abs(float((frame.T @ force[:3])[2]))
        count += 1
    return total, count


def _animal_weight(env) -> float:
    """Weight of the animal's kinematic subtree, excluding prey/food bodies."""

    model = env.model
    root_body = model.body_rootid[1] if model.nbody > 1 else 0
    # The animal is the subtree containing the actuated joints; every other
    # rooted subtree (prey, food) is scenery.
    actuated_roots = {int(model.body_rootid[model.jnt_bodyid[model.actuator_trnid[a, 0]]]) for a in range(model.nu)}
    if not actuated_roots:
        actuated_roots = {int(root_body)}
    mass = sum(float(model.body_mass[i]) for i in range(model.nbody) if int(model.body_rootid[i]) in actuated_roots)
    return mass * 9.81


def report(species: str, settle_steps: int = 200) -> dict:
    module_name, class_name = SPECIES_ENVS[species]
    env_cls = getattr(importlib.import_module(module_name), class_name)

    config_path = Path(_repo_root) / "configs" / species / "stage1_balance.toml"
    env_kwargs = {k: v for k, v in tomllib.loads(config_path.read_text())["env"].items() if k not in JAX_ONLY_ENV_KEYS}
    env_kwargs["reset_noise_scale"] = 0.0

    env = env_cls(**env_kwargs)
    try:
        env.reset(seed=0)
        zero = np.zeros(env.action_space.shape)
        for _ in range(settle_steps):
            env.step(zero)

        sensors = _foot_sensor_sum(env)
        contact, n_contacts = _floor_normal_force(env)
        weight = _animal_weight(env)
        return {
            "species": species,
            "sensor_total_n": sensors,
            "contact_total_n": contact,
            "n_floor_contacts": n_contacts,
            "animal_weight_n": weight,
            "sensor_over_contact": (sensors / contact) if (sensors is not None and contact) else None,
            "contact_over_weight": (contact / weight) if weight else None,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "species",
        nargs="*",
        default=[],
        help=f"species to check (default: all of {', '.join(sorted(SPECIES_ENVS))})",
    )
    parser.add_argument("--settle-steps", type=int, default=200)
    args = parser.parse_args()
    species_list = args.species or sorted(SPECIES_ENVS)
    unknown = [s for s in species_list if s not in SPECIES_ENVS]
    if unknown:
        parser.error(f"unknown species {unknown}; choose from {sorted(SPECIES_ENVS)}")

    print(f"{'species':16} {'sensors':>10} {'contact':>10} {'weight':>10} {'sens/con':>9} {'con/wt':>8}")
    bad = []
    for species in species_list:
        r = report(species, args.settle_steps)
        s = "n/a" if r["sensor_total_n"] is None else f"{r['sensor_total_n']:10.1f}"
        soc = "  n/a" if r["sensor_over_contact"] is None else f"{r['sensor_over_contact']:9.3f}"
        cow = "  n/a" if r["contact_over_weight"] is None else f"{r['contact_over_weight']:8.3f}"
        print(f"{species:16} {s} {r['contact_total_n']:10.1f} {r['animal_weight_n']:10.1f} {soc} {cow}")
        if r["sensor_over_contact"] is not None and not 0.95 <= r["sensor_over_contact"] <= 1.05:
            bad.append(f"{species}: touch sensors see {r['sensor_over_contact']:.1%} of measured floor contact")
        if r["contact_over_weight"] is not None and not 0.95 <= r["contact_over_weight"] <= 1.05:
            bad.append(f"{species}: floor reaction is {r['contact_over_weight']:.1%} of body weight")

    if bad:
        print("\nFAILED invariants:")
        for line in bad:
            print(f"  {line}")
        return 1
    print("\nAll species: touch sensors agree with mj_contactForce, and floor reaction equals body weight.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
