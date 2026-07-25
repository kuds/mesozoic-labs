#!/usr/bin/env python3
"""Chassis-width and lateral-actuator study for the $10k Dibothrosuchus quadruped.

Answers the two questions that ``docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md``
section 4 explicitly assigns to simulation:

1. **Which internal shoulder/pelvic spacing?** The doc asks for a 180 / 210 /
   240 mm sweep at the same exterior length, selecting the narrowest closed-box
   chassis that still survives lateral-push recovery and 2-3 m/s work.
2. **Do 4x AK60-6 at the lateral hips have margin?** The doc permits the
   cheaper 8x AK80-9 + 4x AK60-6 set "only if simulation shows adequate margin
   for cornering, disturbance recovery, and self-righting".

**This is a study, not a species plant.** The geometry is provisional by the
doc's own account -- the transmission is "a planning allowance, not a
purchase-ready BOM", 5M belts are "only a starting family", and the drive is to
be frozen only after a one-leg fixture passes torque, impact, thermal,
tooth-jump, backlash, bandwidth and EMI tests. So the model is generated in
memory and never written into a species ``assets/`` directory: it stays outside
the plant contract, the generated catalogs, and the revision-gating machinery
until the hardware geometry is actually frozen.

Deliberately **not** modelled, because inventing the numbers would be false
precision rather than conservatism:

* belt torsional compliance and backlash -- both are fixture measurements;
* motor rotor inertia -- ``ARMATURE_*`` below are placeholders pending the
  dynamometer test section 4 mandates, and they only matter at the top of the
  excitation sweep;
* thermal derating -- the rated-torque comparisons here are instantaneous.

The excitation also drives hip pitch and knee **in phase**, where a real trot
leads swing with knee flexion. That contributes to the knee pinning at its cap
in every configuration, so the knee column discriminates nothing and is not
reported as a margin result. The ab/ad column, which is what the AK60-6
question turns on, is driven at 0.25x and does discriminate.

What the study *can* bound is the mechanical envelope: stability geometry,
rotational inertia, and joint-torque demand against published actuator
ratings. It says nothing about whether a trained policy runs at 3 m/s; section
4 already identifies controls as the research risk.

Usage::

    python environments/shared/scripts/hw_chassis_study.py
    python environments/shared/scripts/hw_chassis_study.py --spacings 180 210 240
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


# ---------------------------------------------------------------------------
# Catalog actuators, as quoted in RAPTOR_SCALING_AND_ALTERNATIVES.md section 4
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Actuator:
    """Published catalog figures for one integrated QDD module."""

    name: str
    peak_nm: float
    rated_nm: float
    mass_kg: float
    price_usd: float


# The doc uses the CubeMars page's conservative 490 g detailed-table value for
# the AK80-9 rather than its conflicting 480 g summary value.  The AK60-6 mass
# is back-derived from the doc's stated ~440 g saving across four units.
AK80_9 = Actuator("AK80-9 V3", peak_nm=22.0, rated_nm=9.0, mass_kg=0.490, price_usd=479.90)
AK60_6 = Actuator("AK60-6 V3", peak_nm=9.0, rated_nm=3.0, mass_kg=0.380, price_usd=229.90)

# Reflected rotor inertia at the joint (kg m^2).  PLACEHOLDERS -- section 4
# requires a dynamometer test of one AK80-9 and one complete remote-knee leg
# before the structure is frozen.  Reported sensitivity, not a specification.
ARMATURE_AK80 = 0.008
ARMATURE_AK60 = 0.004

# Non-actuator mass budget (kg).  Battery is the Gens Ace 12S 3.5 Ah pack
# priced in section 4; the shell uses section 4's 0.6 kg morphology cap.
FRAME_KG = 1.80
BATTERY_KG = 1.054
ELECTRONICS_KG = 0.70
SHELL_KG = 0.60
FEMUR_KG = 0.12
TIBIA_KG = 0.08
FOOT_KG = 0.05

# Fixed geometry from section 4 (structural body 45-50 cm, effective leg
# 25-30 cm).  Only the lateral spacing is swept.
BODY_LENGTH_M = 0.47
BODY_HEIGHT_M = 0.12
HIP_X_M = 0.185  # fore/aft hip axis offset from body centre
ABAD_OFFSET_M = 0.060  # lateral offset from ab/ad axis to the leg plane
EFFECTIVE_LEG_M = 0.28  # hip axis to foot origin, standing
SEGMENT_DX_M = 0.096  # fore/aft zig-zag of the authored stance
FOOT_PAD_M = 0.024  # foot origin to pad underside

LEGS = (("fr", 1, -1), ("fl", 1, 1), ("rr", -1, -1), ("rl", -1, 1))
JOINT_CLASSES = ("hip_abad", "hip_pitch", "knee")


@dataclass(frozen=True)
class ChassisConfig:
    """One point in the sweep."""

    hip_spacing_m: float
    lateral_actuator: Actuator

    @property
    def label(self) -> str:
        return f"{self.hip_spacing_m * 1000:.0f}mm/{self.lateral_actuator.name}"

    def actuator_for(self, joint_class: str) -> Actuator:
        return self.lateral_actuator if joint_class == "hip_abad" else AK80_9

    @property
    def actuator_cost_usd(self) -> float:
        return 8 * AK80_9.price_usd + 4 * self.lateral_actuator.price_usd

    @property
    def trunk_mass_kg(self) -> float:
        """Frame, battery, electronics, shell, and the four ab/ad stators."""
        return FRAME_KG + BATTERY_KG + ELECTRONICS_KG + SHELL_KG + 4 * self.lateral_actuator.mass_kg


# ---------------------------------------------------------------------------
# Parameterized MJCF
# ---------------------------------------------------------------------------


def build_mjcf(cfg: ChassisConfig) -> str:
    """Generate the study chassis at one lateral spacing.

    Mini-Cheetah-class topology per section 4: 12 DOF (hip ab/ad, hip pitch,
    knee), every motor body-proximal, the knee driven remotely so no actuator
    sits at the anatomical knee, and passive sprung ankles.  The stance is
    authored into the body offsets, so every hinge is 0 at the home keyframe
    and the neutral control vector is zero.
    """
    trunk_z = 0.02 + EFFECTIVE_LEG_M + FOOT_PAD_M
    half_width = cfg.hip_spacing_m / 2.0
    # Two motors ride each ab/ad link: the hip-pitch unit and the remote-knee
    # unit.  Section 4 forbids putting the knee motor at the knee.
    abad_link_mass = 2 * AK80_9.mass_kg

    legs_xml = []
    for name, fore, side in LEGS:
        hip_x = HIP_X_M * fore
        hip_y = half_width * side
        leg_y = ABAD_OFFSET_M * side
        legs_xml.append(f"""
      <body name="{name}_abad" pos="{hip_x:.4f} {hip_y:.4f} -0.02">
        <joint name="{name}_hip_abad" type="hinge" axis="1 0 0" range="-45 45"
               damping="0.6" armature="{_armature(cfg, "hip_abad"):.4f}"/>
        <geom name="{name}_abad_geom" type="cylinder" fromto="0 0 0  0 {leg_y:.4f} 0"
              size="0.049" mass="{abad_link_mass:.4f}" rgba="0.35 0.35 0.38 1"/>

        <body name="{name}_upper" pos="0 {leg_y:.4f} 0">
          <joint name="{name}_hip_pitch" type="hinge" axis="0 1 0" range="-70 70"
                 damping="0.6" armature="{ARMATURE_AK80:.4f}"/>
          <geom name="{name}_upper_geom" type="capsule"
                fromto="0 0 0  {SEGMENT_DX_M:.4f} 0 -0.14" size="0.018"
                mass="{FEMUR_KG:.4f}" rgba="0.55 0.55 0.58 1"/>

          <body name="{name}_lower" pos="{SEGMENT_DX_M:.4f} 0 -0.14">
            <joint name="{name}_knee" type="hinge" axis="0 1 0" range="-80 80"
                   damping="0.5" armature="{ARMATURE_AK80:.4f}"/>
            <geom name="{name}_lower_geom" type="capsule"
                  fromto="0 0 0  {-SEGMENT_DX_M:.4f} 0 -0.14" size="0.012"
                  mass="{TIBIA_KG:.4f}" rgba="0.55 0.55 0.58 1"/>

            <body name="{name}_foot" pos="{-SEGMENT_DX_M:.4f} 0 -0.14">
              <!-- Passive compliant ankle: sprung, unactuated.  Section 4 asks
                   for compliance here instead of four more motors. -->
              <joint name="{name}_ankle" type="hinge" axis="0 1 0" range="-30 30"
                     damping="0.15" stiffness="6.0" armature="0.0005"/>
              <geom name="{name}_foot_geom" type="sphere" pos="0 0 -0.012" size="0.012"
                    mass="{FOOT_KG:.4f}" rgba="0.2 0.2 0.22 1"/>
              <site name="{name}_contact" type="sphere" pos="0 0 -0.012" size="0.02" group="4"/>
            </body>
          </body>
        </body>
      </body>""")

    actuators_xml = []
    for name, _, _ in LEGS:
        for joint_class, kp, ctrl in (
            ("hip_abad", 40.0, 0.7854),
            ("hip_pitch", 60.0, 1.2217),
            ("knee", 60.0, 1.3963),
        ):
            peak = cfg.actuator_for(joint_class).peak_nm
            actuators_xml.append(
                f'    <position name="{name}_{joint_class}_act" joint="{name}_{joint_class}" '
                f'kp="{kp:.1f}" forcerange="{-peak:.2f} {peak:.2f}" '
                f'ctrlrange="{-ctrl:.4f} {ctrl:.4f}"/>'
            )

    sensors_xml = "\n".join(f'    <touch name="{n}_touch" site="{n}_contact"/>' for n, _, _ in LEGS)
    excludes_xml = "\n".join(
        f'    <exclude body1="trunk" body2="{n}_abad"/>\n    <exclude body1="{n}_abad" body2="{n}_upper"/>'
        for n, _, _ in LEGS
    )
    home_qpos = " ".join(["0"] * (len(LEGS) * 4))
    home_ctrl = " ".join(["0"] * (len(LEGS) * 3))

    return f"""<mujoco model="dibothrosuchus_hw_study">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast">
    <flag warmstart="enable"/>
  </option>

  <default>
    <geom contype="1" conaffinity="1" friction="1.0 0.005 0.0001"/>
  </default>

  <worldbody>
    <geom name="floor" type="plane" size="50 50 0.1" rgba="0.3 0.3 0.3 1"/>

    <body name="trunk" pos="0 0 {trunk_z:.4f}">
      <freejoint name="root"/>
      <geom name="trunk_geom" type="box"
            size="{BODY_LENGTH_M / 2:.4f} {half_width:.4f} {BODY_HEIGHT_M / 2:.4f}"
            mass="{cfg.trunk_mass_kg:.4f}" rgba="0.4 0.42 0.36 1"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
{"".join(legs_xml)}
    </body>
  </worldbody>

  <actuator>
{chr(10).join(actuators_xml)}
  </actuator>

  <sensor>
{sensors_xml}
  </sensor>

  <contact>
{excludes_xml}
  </contact>

  <keyframe>
    <key name="home" qpos="0 0 {trunk_z:.4f} 1 0 0 0 {home_qpos}" ctrl="{home_ctrl}"/>
  </keyframe>
</mujoco>
"""


def _armature(cfg: ChassisConfig, joint_class: str) -> float:
    return ARMATURE_AK60 if cfg.actuator_for(joint_class) is AK60_6 else ARMATURE_AK80


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def _settle(model: mujoco.MjModel, steps: int = 2000) -> mujoco.MjData:
    data = mujoco.MjData(model)
    key = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key)
    data.ctrl[:] = model.key_ctrl[key]
    for _ in range(steps):
        mujoco.mj_step(model, data)
    return data


def _com_and_inertia(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, float, np.ndarray]:
    """Whole-robot CoM, mass, and inertia tensor about that CoM."""
    total_mass = float(model.body_mass[1:].sum())
    com = np.zeros(3)
    for i in range(1, model.nbody):
        com += float(model.body_mass[i]) * data.xipos[i]
    com /= total_mass

    inertia = np.zeros((3, 3))
    for i in range(1, model.nbody):
        mass = float(model.body_mass[i])
        rot = data.ximat[i].reshape(3, 3)
        body_inertia = rot @ np.diag(model.body_inertia[i]) @ rot.T
        offset = data.xipos[i] - com
        inertia += body_inertia + mass * (np.dot(offset, offset) * np.eye(3) - np.outer(offset, offset))
    return com, total_mass, inertia


def _joint_class_forces(model: mujoco.MjModel, forces: np.ndarray) -> dict[str, float]:
    """Peak |actuator force| grouped by joint class."""
    peaks = dict.fromkeys(JOINT_CLASSES, 0.0)
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
        for joint_class in JOINT_CLASSES:
            if name.endswith(f"_{joint_class}_act"):
                peaks[joint_class] = max(peaks[joint_class], abs(float(forces[i])))
    return peaks


def static_report(cfg: ChassisConfig) -> dict:
    """Stance geometry, rotational inertia, and settled holding torques."""
    model = mujoco.MjModel.from_xml_string(build_mjcf(cfg))
    data = _settle(model)
    com, mass, inertia = _com_and_inertia(model, data)

    foot_ys = [
        float(data.geom_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{n}_foot_geom")][1])
        for n, _, _ in LEGS
    ]
    track_m = max(foot_ys) - min(foot_ys)
    com_height = float(com[2])
    # Lateral margin from the CoM ground projection to the roll-over edge, and
    # the roll angle at which the projection leaves the support polygon.
    lateral_margin = track_m / 2.0 - abs(float(com[1]))
    tip_angle_deg = float(np.degrees(np.arctan2(lateral_margin, com_height)))

    return {
        "mass_kg": mass,
        "com_height_m": com_height,
        "track_m": track_m,
        "lateral_margin_m": lateral_margin,
        "tip_angle_deg": tip_angle_deg,
        "roll_inertia": float(inertia[0, 0]),
        "pitch_inertia": float(inertia[1, 1]),
        "yaw_inertia": float(inertia[2, 2]),
        "static_torque": _joint_class_forces(model, data.actuator_force),
        "settled_trunk_z": float(data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")][2]),
    }


def push_recovery(cfg: ChassisConfig, impulse_ns: float, duration_s: float = 0.05) -> dict:
    """Lateral shove on the trunk; peak ab/ad torque and whether it stayed up.

    Section 4 gates the chassis on lateral-push recovery, and the lateral hips
    are exactly the joints the AK60-6 question is about.
    """
    model = mujoco.MjModel.from_xml_string(build_mjcf(cfg))
    data = _settle(model)
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
    start_z = float(data.xpos[trunk][2])

    force_n = impulse_ns / duration_s
    push_steps = int(duration_s / model.opt.timestep)
    peaks = dict.fromkeys(JOINT_CLASSES, 0.0)
    min_z = start_z

    for step in range(3000):
        data.xfrc_applied[trunk] = np.zeros(6)
        if step < push_steps:
            data.xfrc_applied[trunk][1] = force_n
        mujoco.mj_step(model, data)
        for joint_class, value in _joint_class_forces(model, data.actuator_force).items():
            peaks[joint_class] = max(peaks[joint_class], value)
        min_z = min(min_z, float(data.xpos[trunk][2]))

    data.xfrc_applied[trunk] = np.zeros(6)
    return {
        "impulse_ns": impulse_ns,
        "peak_torque": peaks,
        "min_trunk_z": min_z,
        "recovered": min_z > 0.6 * start_z,
    }


def impulse_sweep(cfg: ChassisConfig, impulses_ns: list[float]) -> dict:
    """Survival across a grid of lateral impulses.

    Deliberately a grid rather than a bisection: pass/fail is **not monotonic**
    in impulse near the threshold -- contact makes the outcome jumpy, and an
    earlier bisection converged to 11.8 N s for a chassis that survives 12.0.
    Reporting the last success, the first failure, and the count of mixed
    outcomes in between exposes that instead of averaging it away.
    """
    outcomes = []
    for impulse in impulses_ns:
        result = push_recovery(cfg, impulse)
        outcomes.append((impulse, result["recovered"], result["peak_torque"]["hip_abad"]))

    survived = [imp for imp, ok, _ in outcomes if ok]
    failed = [imp for imp, ok, _ in outcomes if not ok]
    last_success = max(survived) if survived else 0.0
    first_failure = min(failed) if failed else float("inf")
    # Anything that survived above the first failure is a non-monotonic outcome.
    mixed = sum(1 for imp in survived if imp > first_failure)
    peak_at_last = next((t for imp, ok, t in outcomes if ok and imp == last_success), 0.0)

    return {
        "last_success_ns": last_success,
        "first_failure_ns": first_failure,
        "mixed_count": mixed,
        "n_points": len(outcomes),
        "peak_abad_nm": peak_at_last,
    }


def trot_excitation(cfg: ChassisConfig, hz: float, amplitude: float = 0.25, steps: int = 4000) -> dict:
    """Scripted diagonal-pair excitation; peak torque and saturation fraction.

    This is the plant characterization the repo already applies to its animal
    models (``actuator_saturation_report.py``): open-loop excitation bounding
    actuator demand.  It is not a gait and not evidence of achieved speed.
    """
    model = mujoco.MjModel.from_xml_string(build_mjcf(cfg))
    data = _settle(model)

    phase_by_leg = {"fr": 0.0, "rl": 0.0, "fl": np.pi, "rr": np.pi}
    lo = model.actuator_ctrlrange[:, 0]
    hi = model.actuator_ctrlrange[:, 1]
    span = (hi - lo) / 2.0
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)]

    forcerange = model.actuator_forcerange[:, 1]
    clip_hits = np.zeros(model.nu)
    peaks = dict.fromkeys(JOINT_CLASSES, 0.0)

    for step in range(steps):
        t = step * model.opt.timestep
        for i, name in enumerate(names):
            leg = name.split("_")[0]
            wave = np.sin(2 * np.pi * hz * t + phase_by_leg[leg])
            # Hips and knees drive the stride; ab/ad only trims laterally.
            scale = 0.25 if name.endswith("_hip_abad_act") else 1.0
            data.ctrl[i] = amplitude * scale * span[i] * wave
        mujoco.mj_step(model, data)
        clip_hits += np.abs(data.actuator_force) >= 0.999 * np.maximum(forcerange, 1e-9)
        for joint_class, value in _joint_class_forces(model, data.actuator_force).items():
            peaks[joint_class] = max(peaks[joint_class], value)

    clip_by_class = dict.fromkeys(JOINT_CLASSES, 0.0)
    for i, name in enumerate(names):
        for joint_class in JOINT_CLASSES:
            if name.endswith(f"_{joint_class}_act"):
                clip_by_class[joint_class] = max(clip_by_class[joint_class], clip_hits[i] / steps)

    return {"hz": hz, "peak_torque": peaks, "clip_fraction": clip_by_class}


def froude_speed(hz: float, leg_m: float = EFFECTIVE_LEG_M) -> float:
    """Rough stride-frequency to speed mapping (stride ~ 2x leg length)."""
    return hz * 2.0 * leg_m


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _margin(peak_nm: float, actuator: Actuator) -> str:
    return f"{actuator.peak_nm / peak_nm:.2f}x" if peak_nm > 1e-6 else "n/a"


def run(spacings_mm: list[float], impulse_ns: float, frequencies: list[float]) -> None:
    configs = [ChassisConfig(mm / 1000.0, lateral) for mm in spacings_mm for lateral in (AK80_9, AK60_6)]

    print("\n=== Build validity check against RAPTOR_SCALING_AND_ALTERNATIVES.md section 4 ===")
    for lateral in (AK80_9, AK60_6):
        cfg = ChassisConfig(0.210, lateral)
        static = static_report(cfg)
        print(
            f"  lateral={lateral.name:11s} actuator cost ${cfg.actuator_cost_usd:8,.2f}"
            f"   all-up {static['mass_kg']:5.2f} kg (doc target 10-12 kg)"
        )

    print("\n=== 1. Static geometry and rotational inertia ===")
    print(f"  {'config':22s} {'mass':>7} {'CoM z':>7} {'track':>7} {'tip':>7} {'I_roll':>8} {'I_yaw':>8}")
    for cfg in configs:
        s = static_report(cfg)
        print(
            f"  {cfg.label:22s} {s['mass_kg']:6.2f}kg {s['com_height_m']:6.3f}m "
            f"{s['track_m']:6.3f}m {s['tip_angle_deg']:6.1f}d "
            f"{s['roll_inertia']:8.4f} {s['yaw_inertia']:8.4f}"
        )

    print("\n=== 2. Settled holding torque vs rated (continuous) torque ===")
    print(f"  {'config':22s} {'abad':>8} {'rated':>7} {'margin':>8} {'knee':>8} {'rated':>7} {'margin':>8}")
    for cfg in configs:
        s = static_report(cfg)
        abad = s["static_torque"]["hip_abad"]
        knee = s["static_torque"]["knee"]
        lat = cfg.lateral_actuator
        abad_margin = f"{lat.rated_nm / abad:.2f}x" if abad > 1e-6 else "n/a"
        knee_margin = f"{AK80_9.rated_nm / knee:.2f}x" if knee > 1e-6 else "n/a"
        print(
            f"  {cfg.label:22s} {abad:7.2f}N {lat.rated_nm:6.1f}N {abad_margin:>8} "
            f"{knee:7.2f}N {AK80_9.rated_nm:6.1f}N {knee_margin:>8}"
        )

    print(f"\n=== 3. Lateral push recovery at a fixed {impulse_ns:.1f} N s shove ===")
    print(f"  {'config':22s} {'abad peak':>10} {'peak margin':>12} {'min z':>8} {'recovered':>10}")
    for cfg in configs:
        r = push_recovery(cfg, impulse_ns)
        abad = r["peak_torque"]["hip_abad"]
        print(
            f"  {cfg.label:22s} {abad:9.2f}N {_margin(abad, cfg.lateral_actuator):>12} "
            f"{r['min_trunk_z']:7.3f}m {str(r['recovered']):>10}"
        )

    grid = [6.0 + i for i in range(15)]
    print(f"\n=== 4. Lateral impulse survival sweep ({grid[0]:.0f}-{grid[-1]:.0f} N s grid) ===")
    print(f"  {'config':22s} {'last ok':>9} {'first fail':>11} {'mixed':>7} {'dv at last':>11} {'abad margin':>12}")
    for cfg in configs:
        s = static_report(cfg)
        r = impulse_sweep(cfg, grid)
        delta_v = r["last_success_ns"] / s["mass_kg"]
        print(
            f"  {cfg.label:22s} {r['last_success_ns']:7.1f}Ns {r['first_failure_ns']:9.1f}Ns "
            f"{r['mixed_count']:3d}/{r['n_points']:<3d} {delta_v:10.2f}m/s "
            f"{_margin(r['peak_abad_nm'], cfg.lateral_actuator):>12}"
        )

    print("\n=== 5. Scripted diagonal-pair excitation (plausible trot excursions) ===")
    print(f"  {'config':22s} {'Hz':>5} {'~m/s':>6} {'abad pk':>8} {'margin':>8} {'knee pk':>8} {'clip abad':>10}")
    for cfg in configs:
        for hz in frequencies:
            e = trot_excitation(cfg, hz)
            abad = e["peak_torque"]["hip_abad"]
            print(
                f"  {cfg.label:22s} {hz:5.1f} {froude_speed(hz):6.2f} {abad:7.2f}N "
                f"{_margin(abad, cfg.lateral_actuator):>8} {e['peak_torque']['knee']:7.2f}N "
                f"{e['clip_fraction']['hip_abad']:9.1%}"
            )

    print(
        "\nBounds the mechanical envelope only. Belt compliance, backlash, rotor inertia,"
        "\nand thermal derating are unmodelled; the excitation is open-loop, not a gait."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spacings", type=float, nargs="*", default=[180.0, 210.0, 240.0], help="mm")
    parser.add_argument("--impulse", type=float, default=12.0, help="lateral push impulse (N s)")
    parser.add_argument("--frequencies", type=float, nargs="*", default=[3.6, 4.5, 5.4], help="Hz")
    args = parser.parse_args(argv)
    run(args.spacings, args.impulse, args.frequencies)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
