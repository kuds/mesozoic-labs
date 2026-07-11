"""Report per-actuator force saturation on the raptor plant.

Measures what fraction of simulation steps each position actuator spends
at its ``forcerange`` cap, under (a) static settling and (b) gait-like
alternating-leg sinusoids, and how much the bounded plant diverges from
an unbounded copy under identical commands. Written for the stage-2
locomotion investigation (docs/investigations/STAGE2_INVESTIGATION.md):
the original ~0.8x kp sizing was validated against *static* behavior
only, while dynamic gaits lived in the clipped regime; hip pitch and
ankle now carry 1.5x kp headroom (docs/investigations/
STAGE2_RECOMMENDATIONS.md R2), which this report measures at 0% gait
clipping. Re-run after any actuator re-sizing.

Usage::

    python environments/velociraptor/scripts/actuator_saturation_report.py
"""

import re
from pathlib import Path

import mujoco
import numpy as np

XML_PATH = Path(__file__).resolve().parents[1] / "assets" / "raptor.xml"
CLIP_EPS = 0.999
GAIT_HZ = 2.5
GAIT_AMPLITUDE = 0.8


def _gait_ctrl(model, t):
    lo, hi = model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
    mid, amp = (lo + hi) / 2, (hi - lo) / 2
    phase = 2 * np.pi * GAIT_HZ * t * model.opt.timestep
    ctrl = mid.copy()
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        if name.startswith("r_"):
            s = np.sin(phase)
        elif name.startswith("l_"):
            s = np.sin(phase + np.pi)
        else:
            s = 0.4 * np.sin(phase)
        ctrl[i] = mid[i] + GAIT_AMPLITUDE * amp[i] * s
    return ctrl


def _run(xml_str, gait, steps):
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    fr = model.actuator_forcerange[:, 1]
    clip_hits = np.zeros(model.nu)
    zs = []
    for t in range(steps):
        if gait:
            data.ctrl[:] = _gait_ctrl(model, t)
        mujoco.mj_step(model, data)
        clip_hits += np.abs(data.actuator_force) >= CLIP_EPS * np.maximum(fr, 1e-9)
        zs.append(data.qpos[2])
    return model, clip_hits / steps, np.array(zs)


def main(steps: int = 2500):
    xml = XML_PATH.read_text()
    xml_unbounded = re.sub(r' forcerange="[^"]*"', "", xml)

    model, clip_settle, _ = _run(xml, gait=False, steps=steps)
    _, clip_gait, z_bounded = _run(xml, gait=True, steps=steps)
    _, _, z_unbounded = _run(xml_unbounded, gait=True, steps=steps)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    is_position = model.actuator_biasprm[:, 1] < 0

    print(
        f"Raptor actuator saturation over {steps} steps ({GAIT_HZ} Hz, {GAIT_AMPLITUDE:.0%} amplitude gait excitation)"
    )
    print(f"{'actuator':26s} {'kp':>6s} {'forcerange':>10s} {'settle':>8s} {'gait':>8s}")
    order = np.argsort(-clip_gait)
    for i in order:
        if not is_position[i]:
            continue  # motors (claw) pin their gear force by construction
        kp = model.actuator_gainprm[i, 0]
        fr = model.actuator_forcerange[i, 1]
        print(f"{names[i]:26s} {kp:6.0f} {fr:10.0f} {clip_settle[i]:8.1%} {clip_gait[i]:8.1%}")

    rms = float(np.sqrt(np.mean((z_bounded - z_unbounded) ** 2)))
    print(f"\npelvis-z RMS divergence, bounded vs unbounded plant under identical commands: {rms:.4f} m")


if __name__ == "__main__":
    main()
