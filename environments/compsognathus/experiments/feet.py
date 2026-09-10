"""Controlled foot-mechanism ablations; never replace the registered plant.

All variants retain geom masses and neutral contact geometry. ``fixed`` removes
the whole-foot motor/hinge; ``coupled`` then articulates all digits together;
``independent`` gives each digit a passive hinge. Coupled spring, damping and
armature sum three per-digit values, avoiding a hidden 3x compliance difference.
"""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from typing import Literal

import mujoco
import numpy as np

from environments.compsognathus import MODEL_PATHS

Mechanism = Literal["baseline", "fixed", "coupled", "independent"]
MECHANISMS: tuple[Mechanism, ...] = ("baseline", "fixed", "coupled", "independent")


def _numbers(values) -> str:
    return " ".join(format(float(x), ".15g") for x in values)


def _required(root: ET.Element, query: str) -> ET.Element:
    element = root.find(query)
    if element is None:
        raise ValueError("Unsupported source model: missing " + query)
    return element


def foot_xml(
    mechanism: Mechanism,
    *,
    stiffness: float = 0.15,
    damping: float = 0.002,
    armature: float = 0.000001,
) -> str:
    """Build an isolated MJCF string; stiffness/damping/armature are per digit.

    No production files, registrations, manifests or model identities are
    modified. Outputs have incompatible physics and potentially dimensions.
    """
    if mechanism not in MECHANISMS:
        raise ValueError("Unknown foot mechanism: " + str(mechanism))
    if not all(np.isfinite(x) and x > 0 for x in (stiffness, damping, armature)):
        raise ValueError("stiffness, damping and armature must be finite and positive")
    source = MODEL_PATHS["biological"].read_text()
    if mechanism == "baseline":
        return source
    root = ET.fromstring(source)
    root.set("model", "compsognathus_research_" + mechanism)
    original = mujoco.MjModel.from_xml_string(source)
    home = original.key("home").id
    actuator = _required(root, "actuator")
    sensors = _required(root, "sensor")
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for side in ("r", "l"):
        foot = _required(root, ".//body[@name='" + side + "_foot']")
        foot.remove(_required(foot, "joint[@name='" + side + "_toe']"))
        actuator.remove(_required(actuator, "position[@joint='" + side + "_toe']"))
        for sensor in list(sensors):
            if sensor.get("joint") == side + "_toe":
                sensors.remove(sensor)
        if mechanism == "fixed":
            continue
        groups = ((2, 3, 4),) if mechanism == "coupled" else ((2,), (3,), (4,))
        bodies = []
        for digits in groups:
            name = side + "_digits" + "".join(str(i) for i in digits)
            bodies.append(name)
            body = ET.SubElement(foot, "body", name=name)
            count = len(digits)
            ET.SubElement(
                body,
                "joint",
                name=name + "_pitch",
                axis="0 1 0",
                range="-0.3 0.4",
                stiffness=str(count * stiffness),
                damping=str(count * damping),
                armature=str(count * armature),
                springref="0",
            )
            for digit in digits:
                geom = _required(foot, "geom[@name='" + side + "_toe_d" + str(digit) + "_geom']")
                foot.remove(geom)
                body.append(geom)
            ET.SubElement(
                body,
                "site",
                name=name + "_touch_volume",
                type="box",
                pos=".027 0 -.002",
                size=".045 .03 .008",
                group="4",
            )
            ET.SubElement(sensors, "touch", name=name + "_touch", site=name + "_touch_volume")
        # Digits overlap at their common anatomical root. Suppress only these
        # intentional sibling overlaps, not inter-foot or body-ground contact.
        for first, second in itertools.combinations(bodies, 2):
            ET.SubElement(contact, "exclude", body1=first, body2=second)
    keys = root.findall("./keyframe/key")
    if len(keys) != 1 or keys[0].get("name") != "home":
        raise ValueError("Foot experiments require the source's single home keyframe")
    for attr in ("qpos", "qvel", "ctrl"):
        keys[0].attrib.pop(attr, None)
    compiled = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    qpos = compiled.qpos0.copy()
    qpos[:7] = original.key_qpos[home, :7]
    for j in range(1, compiled.njnt):
        old = mujoco.mj_name2id(original, mujoco.mjtObj.mjOBJ_JOINT, compiled.joint(j).name)
        if old >= 0:
            qpos[compiled.jnt_qposadr[j]] = original.key_qpos[home, original.jnt_qposadr[old]]
    ctrl = [original.key_ctrl[home, original.actuator(compiled.actuator(a).name).id] for a in range(compiled.nu)]
    keys[0].set("qpos", _numbers(qpos))
    keys[0].set("ctrl", _numbers(ctrl))
    return ET.tostring(root, encoding="unicode")


def obstacle_xml(xml: str, *, side: str, location: str, height: float) -> str:
    """Add one obstacle and a 4 mm + extra-height settling drop to each arm.

    The digit obstacles sit beyond the pad or beside it; ``pad`` is an adverse
    control where digit flexibility alone should not solve the support problem.
    """
    if side not in ("r", "l") or location not in ("middle", "outer", "pad"):
        raise ValueError("Invalid obstacle side or location")
    if not np.isfinite(height) or not 0 < height <= 0.01:
        raise ValueError("Obstacle height must be in (0, 0.01] meters")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    # Both source feet use the same fan geometry: the -Y digit is 49 mm long
    # and the +Y digit 45 mm. Place each outer obstacle under its own endpoint.
    lateral = -0.020 if side == "r" else 0.020
    outer_x = 0.049 if side == "r" else 0.045
    offset = {"middle": (0.058, 0), "outer": (outer_x, lateral), "pad": (0.012, 0)}[location]
    pos = d.xpos[m.body(side + "_foot").id].copy()
    pos[:2] += offset
    pos[2] = height / 2
    root = ET.fromstring(xml)
    half_x, half_y = (0.004, 0.002) if location == "outer" else (0.005, 0.004)
    ET.SubElement(
        _required(root, "worldbody"),
        "geom",
        name="test_obstacle",
        type="box",
        pos=_numbers(pos),
        size=_numbers((half_x, half_y, height / 2)),
        mass="0",
        material="ground_mat",
    )
    key = _required(root, "./keyframe/key[@name='home']")
    qpos = np.fromstring(key.get("qpos", ""), sep=" ")
    qpos[2] += height + 0.001
    key.set("qpos", _numbers(qpos))
    return ET.tostring(root, encoding="unicode")


def distal_clearance_xml(xml: str, clearance: float) -> str:
    """Shorten the distal metatarsal capsule envelope, not the leg linkage.

    Raise each capsule's distal endpoint in local Z by up to 3 mm. Retain its
    explicit mass and every joint, sole and digit location. Its inertia and
    COM change slightly, so this is a separate geometry experiment.
    """
    if not np.isfinite(clearance) or not 0 <= clearance <= 0.003:
        raise ValueError("Distal clearance must be finite and in [0, 0.003] meters")
    if clearance == 0:
        return xml
    root = ET.fromstring(xml)
    for side in ("r", "l"):
        geom = _required(root, ".//geom[@name='" + side + "_metatarsus_geom']")
        endpoints = np.fromstring(geom.get("fromto", ""), sep=" ")
        if len(endpoints) != 6:
            raise ValueError("Distal clearance requires fromto metatarsal capsules")
        endpoints[5] += clearance
        geom.set("fromto", _numbers(endpoints))
    return ET.tostring(root, encoding="unicode")


def foot_sensor_groups(model: mujoco.MjModel) -> tuple[tuple[int, ...], ...]:
    """Sensor addresses, grouped by foot, including every moving digit body."""
    return tuple(
        tuple(
            int(model.sensor(i).adr[0])
            for i in range(model.nsensor)
            if model.sensor(i).name == side + "_foot_touch"
            or (model.sensor(i).name.startswith(side + "_digits") and model.sensor(i).name.endswith("_touch"))
        )
        for side in ("r", "l")
    )


def foot_geom_groups(model: mujoco.MjModel) -> tuple[tuple[int, ...], ...]:
    return tuple(
        (model.geom(side + "_plantar_pad").id, *(model.geom(side + "_toe_d" + str(i) + "_geom").id for i in (2, 3, 4)))
        for side in ("r", "l")
    )


def spectral_metrics(samples: np.ndarray, dt: float) -> dict[str, float | None]:
    """Dominant and >10 Hz power of demeaned samples, with a Hann window.

    A static/too-short signal has no meaningful dominant frequency. Values are
    measurements, not an estimate of required feedback bandwidth.
    """
    x = np.asarray(samples, dtype=float)
    if len(x) < 8:
        return {"peak_hz": None, "power_above_10hz": None, "rms": None}
    x = x - x.mean(axis=0)
    rms = float(np.sqrt(np.mean(x**2)))
    if rms < 1e-10:
        return {"peak_hz": None, "power_above_10hz": 0.0, "rms": rms}
    if x.ndim == 1:
        x = x[:, None]
    spectrum = np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)
    power = np.sum(abs(spectrum) ** 2, axis=1)
    power[0] = 0
    frequencies = np.fft.rfftfreq(len(x), dt)
    return {
        "peak_hz": float(frequencies[np.argmax(power)]),
        "power_above_10hz": float(power[frequencies > 10].sum() / power.sum()),
        "rms": rms,
    }
