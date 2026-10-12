"""Report per-actuator force saturation for any model's plant.

Measures what fraction of simulation steps each position actuator spends
at its ``forcerange`` cap, under (a) settling with the XML home-keyframe
controls and (b) scripted
gait excitation (alternating legs for bipeds, diagonal-pair trot for the
brachiosaurus), and how much the bounded plant diverges from an
unbounded copy under identical commands. Written for the stage-2
locomotion investigation (docs/investigations/STAGE2_RECOMMENDATIONS.md):
spike-cap ``forcerange`` sizing validated only against home-control settling
clipped a large fraction of gait torque on every species. Re-run this
after any actuator re-sizing, before spending GPU hours.

Usage (``--model-id`` repeats; with none, the four anatomical models of
:data:`DEFAULT_MODEL_IDS`; ``--hz`` applies to every listed model)::

    python environments/shared/scripts/actuator_saturation_report.py [--model-id SPECIES EMBODIMENT ...] [--hz F]

    # e.g. the brachiosaurus at a 1.5 Hz cadence:
    python environments/shared/scripts/actuator_saturation_report.py \\
        --model-id brachiosaurus_altithorax anatomical --hz 1.5
"""

import argparse
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
from environments.shared.model_id import ModelId
from environments.shared.paths import ENVIRONMENTS_ROOT
from environments.shared.species_names import add_model_id_argument, parsed_model_ids
from environments.shared.tests.actuator_bounds_helpers import (
    CLIP_EPS,
    GAIT_AMPLITUDE,
    GAIT_HZ,
    gait_commands,
)

#: The models measured when no ``--model-id`` is given: the panel this tool
#: has always measured, written out as models (decision D-D33).  Each plant
#: is the model's MJCF (``ModelId.asset_path``).
DEFAULT_MODEL_IDS = (
    ModelId("velociraptor_mongoliensis", "anatomical"),
    ModelId("tyrannosaurus_rex", "anatomical"),
    ModelId("brachiosaurus_altithorax", "anatomical"),
    ModelId("dibothrosuchus_elaphros", "anatomical"),
)


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


def report(model_id: ModelId, hz=GAIT_HZ, amplitude=GAIT_AMPLITUDE, steps=2500):
    xml = model_id.asset_path(ENVIRONMENTS_ROOT).read_text()
    xml_unbounded = re.sub(r' forcerange="[^"]*"', "", xml)

    model, clip_home_control, _ = _run(xml, gait=False, steps=steps, hz=hz, amplitude=amplitude)
    _, clip_gait, z_bounded = _run(xml, gait=True, steps=steps, hz=hz, amplitude=amplitude)
    _, _, z_unbounded = _run(xml_unbounded, gait=True, steps=steps, hz=hz, amplitude=amplitude)

    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
    is_position = model.actuator_biasprm[:, 1] < 0

    print(f"\n{model_id} actuator saturation over {steps} steps ({hz} Hz, {amplitude:.0%} amplitude gait excitation)")
    print(f"{'actuator':26s} {'kp':>6s} {'forcerange':>10s} {'home':>8s} {'gait':>8s}")
    for i in np.argsort(-clip_gait):
        if not is_position[i]:
            continue  # motors (claw) pin their gear force by construction
        kp = model.actuator_gainprm[i, 0]
        fr = model.actuator_forcerange[i, 1]
        print(f"{names[i]:26s} {kp:6.0f} {fr:10.0f} {clip_home_control[i]:8.1%} {clip_gait[i]:8.1%}")

    rms = float(np.sqrt(np.mean((z_bounded - z_unbounded) ** 2)))
    print(f"root-z RMS divergence, bounded vs unbounded plant under identical commands: {rms:.4f} m")


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_model_id_argument(parser, default=DEFAULT_MODEL_IDS, what="measure")
    parser.add_argument(
        "--hz", type=float, default=GAIT_HZ, help=f"gait excitation cadence for every model (default {GAIT_HZ})"
    )
    args = parser.parse_args(argv)
    for model_id in parsed_model_ids(parser, args):
        report(model_id, hz=args.hz)


if __name__ == "__main__":
    main(sys.argv[1:])
