"""Report per-actuator force saturation for any species plant.

Measures what fraction of simulation steps each position actuator spends
at its ``forcerange`` cap, under (a) static settling and (b) scripted
gait excitation (alternating legs for bipeds, diagonal-pair trot for the
brachiosaurus), and how much the bounded plant diverges from an
unbounded copy under identical commands. Written for the stage-2
locomotion investigation (docs/investigations/STAGE2_RECOMMENDATIONS.md):
spike-cap ``forcerange`` sizing validated only against static settling
clipped a large fraction of gait torque on every species. Re-run this
after any actuator re-sizing, before spending GPU hours.

Usage::

    python environments/shared/scripts/actuator_saturation_report.py [species[:hz] ...]

    # e.g. all species at their default cadence:
    python environments/shared/scripts/actuator_saturation_report.py \\
        velociraptor trex brachiosaurus:1.5
"""

import re
import sys
from pathlib import Path

import mujoco
import numpy as np

# Add repo root to path
_repo_root = str(Path(__file__).resolve().parents[3])
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# The scripted excitation is shared with the per-species plant tests so the
# report and the CI pins always measure the same thing.
from environments.shared.tests.actuator_bounds_helpers import (
    CLIP_EPS,
    GAIT_AMPLITUDE,
    GAIT_HZ,
    gait_commands,
)

_ROOT = Path(__file__).resolve().parents[2]
XML_PATHS = {
    "velociraptor": _ROOT / "velociraptor" / "assets" / "raptor.xml",
    "trex": _ROOT / "trex" / "assets" / "trex.xml",
    "brachiosaurus": _ROOT / "brachiosaurus" / "assets" / "brachiosaurus.xml",
}


def _run(xml_str, gait, steps, hz, amplitude):
    model = mujoco.MjModel.from_xml_string(xml_str)
    data = mujoco.MjData(model)
    if model.nkey:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    fr = model.actuator_forcerange[:, 1]
    clip_hits = np.zeros(model.nu)
    zs = []
    for t in range(steps):
        if gait:
            data.ctrl[:] = gait_commands(model, t, hz=hz, amplitude=amplitude)
        mujoco.mj_step(model, data)
        clip_hits += np.abs(data.actuator_force) >= CLIP_EPS * np.maximum(fr, 1e-9)
        zs.append(data.qpos[2])
    return model, clip_hits / steps, np.array(zs)


def report(species, hz=GAIT_HZ, amplitude=GAIT_AMPLITUDE, steps=2500):
    xml = XML_PATHS[species].read_text()
    xml_unbounded = re.sub(r' forcerange="[^"]*"', "", xml)

    model, clip_settle, _ = _run(xml, gait=False, steps=steps, hz=hz, amplitude=amplitude)
    _, clip_gait, z_bounded = _run(xml, gait=True, steps=steps, hz=hz, amplitude=amplitude)
    _, _, z_unbounded = _run(xml_unbounded, gait=True, steps=steps, hz=hz, amplitude=amplitude)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    is_position = model.actuator_biasprm[:, 1] < 0

    print(f"\n{species} actuator saturation over {steps} steps ({hz} Hz, {amplitude:.0%} amplitude gait excitation)")
    print(f"{'actuator':26s} {'kp':>6s} {'forcerange':>10s} {'settle':>8s} {'gait':>8s}")
    for i in np.argsort(-clip_gait):
        if not is_position[i]:
            continue  # motors (claw) pin their gear force by construction
        kp = model.actuator_gainprm[i, 0]
        fr = model.actuator_forcerange[i, 1]
        print(f"{names[i]:26s} {kp:6.0f} {fr:10.0f} {clip_settle[i]:8.1%} {clip_gait[i]:8.1%}")

    rms = float(np.sqrt(np.mean((z_bounded - z_unbounded) ** 2)))
    print(f"root-z RMS divergence, bounded vs unbounded plant under identical commands: {rms:.4f} m")


def main(argv):
    targets = argv or ["velociraptor", "trex", "brachiosaurus"]
    for arg in targets:
        species, _, hz = arg.partition(":")
        if species not in XML_PATHS:
            raise SystemExit(f"unknown species {species!r}; choose from {sorted(XML_PATHS)}")
        report(species, hz=float(hz) if hz else GAIT_HZ)


if __name__ == "__main__":
    main(sys.argv[1:])
