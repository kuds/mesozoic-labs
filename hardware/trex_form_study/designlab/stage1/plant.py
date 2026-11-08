"""Stage 1 plant: a chosen body's screened MJCF made ready for closed-loop simulation (``STAGE1_PLAN.md`` section 2).

The model is the round-two sizing model, ``<case>/models/<id>.xml``, used only after its SHA-256 matches the
record's ``xml_sha256`` (and the plan's prefix of it).  If the file is missing (it is git-ignored) it is rebuilt
from the record with ``pipeline.build_candidate``, the call of ``refine.carpet_lift_one``, into
``stage1/models/<id>/`` and checked the same way.  What the plant adds, by editing a copy of the MJCF text (every
original value is kept verbatim):

* sensors: an IMU site on the trunk at the ledger's ``imu`` row (``<case>/models/<id>.json``) with a gyro, an
  accelerometer and the trunk frame orientation (xbody framequat); joint encoders (position and velocity of the
  14 hinges); and the load on each foot from that foot's floor contacts only (a contact sensor, floor against the
  foot body, net force in the world frame), so a pad touching the other foot never counts as support;
* the two-motor ankle (``drive``): each ankle's pitch and roll servos become motors on the tendons q_p + q_r and
  q_p - q_r, each motor's armature on its tendon;
* joint ranges cut to the clearance screen's usable ranges (``review/screening/<id>_clearance.json``,
  ``summary.joint_range_usable``: the range up to the first new interference, else the MJCF limit).

After compiling:

* hold-servo gains by the rule of ``carpet.hold_servos``, applied to each drive's own coordinate:
  ``kp = sustained / carpet.SERVO_ERROR_RAD`` (0.01 rad) and ``kv = 2 zeta sqrt(kp I)``, I the drive's inertia
  with every other drive held, at the home pose (for a joint servo the diagonal of the mass matrix, as
  ``hold_servos``; for an ankle motor that of the motor coordinates).  In joint space an ankle's pitch and roll
  are therefore twice as stiff as one joint servo would be: two motors share each axis (``ankle_gains="motor"``,
  the default).  ``ankle_gains="joint"`` halves each ankle motor's kp instead, so pitch and roll get exactly the
  joint-space stiffness ``hold_servos`` gives one ankle joint (G1 and G2 are gated on the default; see
  ``gates``).  The MJCF's own gains (``mjcf.write_mjcf``: kp = max(5, 8 x rated torque), kv = 0.05 kp) are
  replaced (DECISION_LOG P23, decision 4);
* ctrl limits off: the servo target carries the feedforward, ``ctrl = G q* + (kv G qd* + f_ff) / kp`` in motor
  coordinates (``Plant.command``);
* the torque-speed envelope (``drive.Envelope``), applied before every physics step;
* the home keyframe's ctrl (the home pose in motor coordinates).

Floors: ``hard`` (as compiled), ``robot_floor_carpet`` (``carpet.floor_variants`` + ``carpet.apply_floor``) and
the probe piles ``low``, ``medium``, ``high`` (``carpet.floor_variants`` + ``carpet.pile_model``; ``pile_grid``
builds a coarser pile with ``carpet.pile_floor``).  carpet.py's caveat holds: the piles match static stiffness only.

Every build settles the home pose for ``SETTLE_S`` with zero command (every servo at its home target), starting
from the keyframe lowered by the predicted sinkage, as ``carpet.standing_check`` starts; on 75-M-H-10 the keyframe
pads sit 0.3 um above the floor, so the stance has no contact before this settle.  ``Plant.reset`` returns to the
settled pose at rest.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

import mujoco
import numpy as np

from .. import carpet, statics
from . import SHA256_PREFIX, STUDY, case_dir, drive

FLOORS = ("hard", "robot_floor_carpet") + carpet.PILES
SETTLE_S = 1.0
IMU_SITE = "imu"
#: Falls (plan section 4): trunk tilt from vertical above this, or the root below FALL_HEIGHT_FRAC x its home height.
FALL_TILT_RAD = 0.6
FALL_HEIGHT_FRAC = 0.7


class PlantError(RuntimeError):
    """The model on disk is not the one the record sized, or a body input is missing."""


# ------------------------------------------------------------------------------------------------
# Inputs: record, model, ledger, clearance screen
# ------------------------------------------------------------------------------------------------
def record(body: str) -> dict:
    return json.loads((case_dir(body) / "candidates" / f"{body}.json").read_text())


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_files(body: str, rebuild_dir: Path | None = None) -> tuple[Path, Path]:
    """(MJCF, ledger) of a body: the round-two files if present, else rebuilt from the record."""
    xml = case_dir(body) / "models" / f"{body}.xml"
    ledger = xml.with_suffix(".json")
    if xml.exists() and ledger.exists():
        return xml, ledger
    from .. import round2
    from ..pipeline import build_candidate

    out = Path(rebuild_dir) if rebuild_dir is not None else STUDY / "stage1" / "models" / body
    xml, ledger = out / f"{body}.xml", out / f"{body}.json"
    if not (xml.exists() and ledger.exists()):
        rec = record(body)
        build_candidate(round2.candidate(body, rec.get("overrides", ())), out, selection=dict(rec["selection"]),
                        B=rec["B"], write=True)
    return xml, ledger


def verify(body: str, xml: Path) -> dict:
    """Check the MJCF against its record's ``xml_sha256`` and the plan's prefix; raises ``PlantError``."""
    rec_sha = record(body)["xml_sha256"]
    got = sha256(xml)
    out = dict(file=str(Path(xml).resolve().relative_to(STUDY)) if Path(xml).resolve().is_relative_to(STUDY)
               else str(xml), file_sha256=got, record_sha256=rec_sha, plan_prefix=SHA256_PREFIX[body],
               matches_record=got == rec_sha, matches_plan=rec_sha.startswith(SHA256_PREFIX[body]))
    if not (out["matches_record"] and out["matches_plan"]):
        raise PlantError(f"{body}: {xml} has SHA-256 {got[:12]}..., the record {rec_sha[:12]}..., the plan "
                         f"{SHA256_PREFIX[body]}...: not the model the record sized")
    return out


def usable_ranges(body: str) -> dict:
    """joint -> (low, high): the clearance screen's range up to the first new interference, else the MJCF limit."""
    doc = json.loads((case_dir(body) / "review" / "screening" / f"{body}_clearance.json").read_text())
    out = {}
    for jn, r in doc["summary"]["joint_range_usable"].items():
        lo, hi = r["first_new_interference"]
        out[jn] = (float(r["limits"][0] if lo is None else lo), float(r["limits"][1] if hi is None else hi))
    return out


def imu_position(ledger: Path) -> list:
    rows = [r for r in json.loads(Path(ledger).read_text())["rows"] if r["name"] == "imu"]
    if len(rows) != 1 or rows[0]["body"] != "trunk":
        raise PlantError(f"{ledger}: expected one 'imu' row on the trunk")
    return [float(v) for v in rows[0]["com_body"]]


def provenance() -> dict:
    """Library versions and the repository commit."""
    vers = {"python": platform.python_version(), "numpy": np.__version__, "mujoco": mujoco.__version__}
    for pkg in ("scipy", "gymnasium", "stable-baselines3", "torch"):
        try:
            vers[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            vers[pkg] = None
    git = dict(commit=None, dirty=None)
    try:
        git["commit"] = subprocess.run(["git", "-C", str(STUDY), "rev-parse", "HEAD"], capture_output=True,
                                       text=True, check=True).stdout.strip()
        st = subprocess.run(["git", "-C", str(STUDY), "status", "--porcelain", "--", "."], capture_output=True,
                            text=True, check=True).stdout
        git["dirty"] = bool(st.strip())
    except (OSError, subprocess.CalledProcessError):
        pass
    return dict(versions=vers, git=git)


# ------------------------------------------------------------------------------------------------
# Model text
# ------------------------------------------------------------------------------------------------
def augment(text: str, joints: list[str], drives: list[drive.Drive], ranges: dict, imu_pos) -> str:
    """The screened MJCF text with the plant's sensors, ankle tendons, motors and usable joint ranges."""
    root = ET.fromstring(text)
    by_coord = {d.coordinate: d for d in drives}
    for j in root.iter("joint"):
        name = j.get("name")
        if name in ranges:
            j.set("range", f"{ranges[name][0]!r} {ranges[name][1]!r}")
        if name is not None and name.endswith(("_ankle_pitch", "_ankle_roll")):
            j.set("armature", "0")  # the motors' armature moves to their tendons
    trunk = root.find("./worldbody/body[@name='trunk']")
    if trunk is None:
        raise PlantError("no trunk body")
    ET.SubElement(trunk, "site", name=IMU_SITE, pos=" ".join(repr(v) for v in imu_pos), size="0.005")
    act_el = root.find("actuator")
    tendon = ET.Element("tendon")
    root.insert(list(root).index(act_el), tendon)
    for side in statics.SIDES:
        for motor, roll in drive.ANKLE_MOTORS:
            name = drive.tendon_name(side, motor)
            t = ET.SubElement(tendon, "fixed", name=name, armature=repr(by_coord[name].armature))
            ET.SubElement(t, "joint", joint=f"{side}_ankle_pitch", coef="1")
            ET.SubElement(t, "joint", joint=f"{side}_ankle_roll", coef=repr(roll))
    replaced = {f"{side}_{jn}": drive.tendon_name(side, motor) for side in statics.SIDES
                for motor, jn in drive.REPLACES.items()}
    for a in act_el:
        jn = a.get("joint")
        if jn in replaced:
            del a.attrib["joint"]
            a.attrib.pop("ctrlrange", None)
            a.set("tendon", replaced[jn])
            a.set("name", drive.actuator_name(replaced[jn]))
    names = [a.get("name") for a in act_el]
    if names != [d.actuator for d in drives]:
        raise PlantError(f"actuator order {names} is not the drive order")
    sens = root.find("sensor")
    if sens is None:
        sens = ET.SubElement(root, "sensor")
    ET.SubElement(sens, "gyro", name="imu_gyro", site=IMU_SITE)
    ET.SubElement(sens, "accelerometer", name="imu_accel", site=IMU_SITE)
    ET.SubElement(sens, "framequat", name="imu_quat", objtype="xbody", objname="trunk")
    for jn in joints:
        ET.SubElement(sens, "jointpos", name=f"enc_q_{jn}", joint=jn)
    for jn in joints:
        ET.SubElement(sens, "jointvel", name=f"enc_qd_{jn}", joint=jn)
    for side in statics.SIDES:
        ET.SubElement(sens, "contact", name=f"{side}_foot_load", geom1="floor", body2=f"{side}_foot",
                      data="found force", reduce="netforce", num="1")
    return ET.tostring(root, encoding="unicode")


def _hinges(m) -> list[str]:
    return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(m.njnt)
            if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]


def _moment(m, d) -> np.ndarray:
    dense = np.zeros((m.nu, m.nv))
    mujoco.mju_sparse2dense(dense, d.actuator_moment, d.moment_rownnz, d.moment_rowadr, d.moment_colind)
    return dense


# ------------------------------------------------------------------------------------------------
# Plant
# ------------------------------------------------------------------------------------------------
class Plant:
    """One body on one floor: ``m``, ``d``, the drives and their envelope, the settled home pose and metadata.

    ``joints`` are the 14 hinges in model order; joint-space arrays (targets, encoders) use that order, and the
    drives (actuators) follow it with each ankle's pitch and roll replaced by its motors a and b."""

    def __init__(self, body: str, floor: str, seed: int, m: mujoco.MjModel, joints: list[str],
                 drives: list[drive.Drive], envelope: drive.Envelope, meta: dict):
        self.body, self.floor, self.seed = body, floor, int(seed)
        self.m, self.d = m, mujoco.MjData(m)
        self.joints, self.drives, self.envelope, self.meta = joints, drives, envelope, meta
        envelope.bind(self.m, self.d)
        self.rng = np.random.default_rng(seed)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in joints]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dofs = np.array([m.jnt_dofadr[j] for j in jid])
        self.G = drive.motor_matrix(joints)
        self._GinvT = np.linalg.inv(self.G.T)
        self.kp = m.actuator_gainprm[:, 0].copy()
        self.kv = -m.actuator_biasprm[:, 2].copy()
        self.trunk = statics.bid(m, "trunk")
        self.feet = [statics.bid(m, f"{s}_foot") for s in statics.SIDES]
        self.q_home = m.key_qpos[0].copy()
        self.ctrl_home = m.key_ctrl[0].copy()
        self.z_home = float(self.q_home[2])
        self._R_home = self._trunk_R(self.q_home)
        self._sens = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i): (int(m.sensor_adr[i]), int(m.sensor_dim[i]))
                      for i in range(m.nsensor)}
        self.q_settled = self.q_home.copy()

    # --------------------------------------------------------------- state
    def _trunk_R(self, q) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[3:7])
        return R.reshape(3, 3)

    def reset(self, joint_noise: float = 0.0, root_velocity_noise: float = 0.0) -> None:
        """Back to the settled home pose at rest, servos at home; optional uniform noise drawn from ``rng``."""
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.q_settled
        if joint_noise:
            d.qpos[self.qadr] += self.rng.uniform(-joint_noise, joint_noise, len(self.qadr))
        if root_velocity_noise:
            d.qvel[:3] = self.rng.uniform(-root_velocity_noise, root_velocity_noise, 3)
        d.ctrl[:] = self.ctrl_home
        self.envelope.restore()
        mujoco.mj_forward(m, d)

    def set_state(self, qpos, qvel) -> None:
        """Start from ``qpos`` and ``qvel`` (e.g. a reference frame), servos at home."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        self.d.ctrl[:] = self.ctrl_home
        self.envelope.restore()
        mujoco.mj_forward(self.m, self.d)

    # --------------------------------------------------------------- actuation
    def command(self, q=None, qd=None, tau=None) -> None:
        """Joint-space servo targets: angle ``q`` (home if None), rate ``qd`` and feedforward torque ``tau``, set
        in motor coordinates as ``ctrl = G q + (kv G qd + G^-T tau) / kp``."""
        ctrl = self.ctrl_home.copy() if q is None else self.G @ np.asarray(q, float)
        if qd is not None:
            ctrl += self.kv * (self.G @ np.asarray(qd, float)) / self.kp
        if tau is not None:
            ctrl += (self._GinvT @ np.asarray(tau, float)) / self.kp
        self.d.ctrl[:] = ctrl

    def hold(self) -> None:
        """Zero command: every servo at its home target, no feedforward."""
        self.d.ctrl[:] = self.ctrl_home

    def step(self, n: int = 1) -> None:
        """``n`` physics steps, the envelope set from the drive speeds before each."""
        m, d, update = self.m, self.d, self.envelope.update
        for _ in range(n):
            update()
            mujoco.mj_step(m, d)

    # --------------------------------------------------------------- sensing
    def sensor(self, name: str) -> np.ndarray:
        adr, dim = self._sens[name]
        return self.d.sensordata[adr : adr + dim].copy()

    def sensors(self) -> dict:
        """The plant's sensors: IMU (gyro rad/s, accelerometer m/s^2, trunk quaternion), encoders (joint angles and
        rates in ``joints`` order) and foot loads (vertical force N, contact count, net force vector per foot).

        MuJoCo evaluates sensors inside ``mj_step`` before it integrates, so after a step they describe the state at
        that step's start, one physics step (2 ms) before ``d.qpos``: what a controller sampling before the step
        sees.  They are not recomputed here, since an extra ``mj_forward`` would change the solver's warm start."""
        sd = self.d.sensordata
        enc = [self._sens[f"enc_q_{jn}"][0] for jn in self.joints]
        encv = [self._sens[f"enc_qd_{jn}"][0] for jn in self.joints]
        feet = [sd[self._sens[f"{s}_foot_load"][0] : self._sens[f"{s}_foot_load"][0] + 4] for s in statics.SIDES]
        return dict(gyro=self.sensor("imu_gyro"), accel=self.sensor("imu_accel"), quat=self.sensor("imu_quat"),
                    q=sd[enc].copy(), qd=sd[encv].copy(), foot_load=np.array([f[3] for f in feet]),
                    foot_contacts=np.array([f[0] for f in feet]), foot_force=np.array([f[1:4] for f in feet]))

    def tilt(self) -> float:
        """Trunk tilt from vertical (rad): the angle between the trunk's z axis and the world's."""
        return float(np.arccos(np.clip(self.d.xmat[self.trunk][8], -1.0, 1.0)))

    def tilt_from_home(self) -> float:
        """Trunk tilt relative to the home orientation (rad), as ``carpet.standing_check`` measures it."""
        Rw = self.d.xmat[self.trunk].reshape(3, 3) @ self._R_home.T
        return float(np.arccos(np.clip(Rw[2, 2], -1.0, 1.0)))

    def fallen(self) -> bool:
        return self.tilt() > FALL_TILT_RAD or float(self.d.qpos[2]) < FALL_HEIGHT_FRAC * self.z_home

    def foot_loads(self) -> np.ndarray:
        return np.array([self.d.sensordata[self._sens[f"{s}_foot_load"][0] + 3] for s in statics.SIDES])

    # --------------------------------------------------------------- settle
    def _predicted_sinkage(self, floor: dict | None, info: dict | None) -> float:
        """Static sinkage of each foot under half the weight, as ``carpet.standing_check`` predicts it."""
        m = self.m
        W = float(m.body_subtreemass[self.trunk]) * carpet.G
        if info is not None:
            return float(carpet.pile_sinkage(info["cells"]["left"], floor["k_A_N_per_m3"], 0.5 * W))
        pads = carpet.pad_geoms(m)["left"]
        pp = carpet.pair_params(m, statics.gid(m, "floor"), pads[0])
        A_hat = float(m.body_invweight0[self.feet[0], 0])
        return float(carpet.static_depth(pp["solref"], pp["solimp"], A_hat, 0.5 * W / (4 * len(pads)), m.opt.timestep))

    def settle(self, sinkage: float, seconds: float = SETTLE_S) -> dict:
        """Settle the home pose with zero command and keep it as the start of every episode."""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        mujoco.mj_forward(m, d)
        ncon_key = [int(v) for v in self.sensors()["foot_contacts"]]
        d.qpos[2] -= sinkage
        mujoco.mj_forward(m, d)
        ncon0 = [int(v) for v in self.sensors()["foot_contacts"]]
        self.step(int(round(seconds / m.opt.timestep)))
        s = self.sensors()
        out = dict(seconds=seconds, foot_contacts_at_keyframe=ncon_key, start_lowered_by_m=sinkage,
                   foot_contacts_at_start=ncon0, foot_contacts_settled=[int(v) for v in s["foot_contacts"]],
                   foot_load_N=[float(v) for v in s["foot_load"]],
                   weight_N=float(m.body_subtreemass[self.trunk]) * carpet.G,
                   root_z_change_m=float(d.qpos[2] - self.z_home),
                   tilt_from_home_deg=float(np.degrees(self.tilt_from_home())),
                   max_abs_qvel=float(np.max(np.abs(d.qvel))),
                   joint_offset_max_rad=float(np.max(np.abs(d.qpos[self.qadr] - self.q_home[self.qadr]))))
        self.q_settled = d.qpos.copy()
        self.reset()
        return out


def _gains(m, d, joints: list[str], G: np.ndarray, drives: list[drive.Drive], ankle_gains: str = "motor") -> dict:
    """Hold-servo gains (module docstring) at the home pose, written into the model's position actuators."""
    d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, Mfull)
    dofs = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in joints]
    Ginv = np.linalg.inv(G)
    inertia = np.diag(Ginv.T @ Mfull[np.ix_(dofs, dofs)] @ Ginv).copy()
    kp = np.array([dr.sustained for dr in drives]) / carpet.SERVO_ERROR_RAD
    if ankle_gains == "joint":  # G^T diag(kp) G on each ankle = the one-joint hold stiffness sustained / 0.01 rad
        kp[[i for i, dr in enumerate(drives) if dr.group in drive.REPLACES]] *= 0.5
    elif ankle_gains != "motor":
        raise ValueError(f"ankle_gains {ankle_gains!r}: 'motor' or 'joint'")
    kv = 2.0 * carpet.SERVO_ZETA * np.sqrt(kp * inertia)
    m.actuator_gainprm[:, 0] = kp
    m.actuator_biasprm[:, 1] = -kp
    m.actuator_biasprm[:, 2] = -kv
    m.actuator_ctrllimited[:] = 0
    return dict(kp=kp, kv=kv, inertia=inertia)


def build(body: str, floor: str = "hard", seed: int = 0, envelope: bool = True, speed_scale: float = 1.0,
          torque_scale: float = 1.0, pile_grid: float | None = None, settle: bool = True,
          ankle_gains: str = "motor") -> Plant:
    """The Stage 1 plant of ``body`` on ``floor`` (``FLOORS``), settled at home unless ``settle`` is False."""
    if floor not in FLOORS:
        raise ValueError(f"floor {floor!r}: choose from {', '.join(FLOORS)}")
    xml, ledger = model_files(body)
    check = verify(body, xml)
    rec = record(body)
    ranges = usable_ranges(body)
    imu = imu_position(ledger)
    text = xml.read_text()
    m0 = mujoco.MjModel.from_xml_string(text)
    joints = _hinges(m0)
    drives = drive.drives(rec["selection"], joints)
    aug = augment(text, joints, drives, ranges, imu)
    m = mujoco.MjModel.from_xml_string(aug)
    q_home = m.key_qpos[0].copy()
    floor_info, pile = None, None
    if floor == "hard":
        floor_params = None
    elif floor == "robot_floor_carpet":
        floor_params = carpet.floor_variants(m, piles={})[floor]
        carpet.apply_floor(m, floor_params)
    else:
        if pile_grid is None:
            floor_params = carpet.floor_variants(m)[floor]
        else:
            pc = carpet.pile_classes()[floor]
            floor_params = carpet.pile_floor(m, pc["stiffness_pa_per_m"], pc["depth_m"], pc["height_m"],
                                             grid=pile_grid)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / f"{body}_stage1.xml"
            path.write_text(aug)
            m, pile = carpet.pile_model(path, q_home, floor_params)
        floor_info = dict(probes_per_foot=pile["probes_per_foot"], geometry=pile["geometry"])
    if _hinges(m) != joints:
        raise PlantError("hinge order changed")
    G = drive.motor_matrix(joints)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    moment = _moment(m, d)
    dofs = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in joints]
    embed = np.zeros((m.nu, m.nv))
    embed[:, dofs] = G
    if not np.array_equal(moment, embed):
        raise PlantError("the compiled actuator transmission differs from drive.motor_matrix")
    gains = _gains(m, d, joints, G, drives, ankle_gains)
    m.key_ctrl[0] = G @ q_home[[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in joints]]
    env = drive.Envelope(drives, moment, enabled=envelope, speed_scale=speed_scale, torque_scale=torque_scale)
    meta = dict(body=body, floor=floor, seed=int(seed), model=check, H_m=float(rec["H"]),
                mass_kg=float(m.body_subtreemass[statics.bid(m, "trunk")]), timestep_s=float(m.opt.timestep),
                integrator=mujoco.mjtIntegrator(m.opt.integrator).name, nq=int(m.nq), nv=int(m.nv), nu=int(m.nu),
                ngeom=int(m.ngeom), imu_site_trunk_m=imu, floor_params=_jsonable(floor_params), floor_info=floor_info,
                gains=dict(rule="carpet.hold_servos on each drive's own coordinate: kp = sustained / "
                                f"{carpet.SERVO_ERROR_RAD} rad, kv = 2 x {carpet.SERVO_ZETA} sqrt(kp I) at home, "
                                "I with every other drive held",
                           ankle_gains=ankle_gains,
                           ankle_joint_space_stiffness="2 kp (two motors per axis)" if ankle_gains == "motor" else
                           "kp of one hold servo (each motor at half kp)",
                           drives={dr.actuator: dict(kp=float(kp), kv=float(kv), inertia_kg_m2=float(i),
                                                     coordinate=dr.coordinate)
                                   for dr, kp, kv, i in zip(drives, gains["kp"], gains["kv"], gains["inertia"],
                                                            strict=True)},
                           mjcf_own={mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_ACTUATOR, a):
                                     dict(kp=float(m0.actuator_gainprm[a, 0]), kv=float(-m0.actuator_biasprm[a, 2]))
                                     for a in range(m0.nu)}),
                drives=[dr.as_dict() for dr in drives], envelope=env.describe(),
                ankle=dict(linkage="motor a on q_p + q_r, motor b on q_p - q_r (unit cranks): "
                                   "|tau_p| + |tau_r| <= 2 peak; joint-space stiffness 2 kp on pitch and roll",
                           armature="each motor's catalog armature on its tendon; the ankle joints' own set to 0"),
                joint_ranges=dict(source="review/screening/<id>_clearance.json summary.joint_range_usable",
                                  mjcf={jn: [float(v) for v in m0.jnt_range[mujoco.mj_name2id(
                                      m0, mujoco.mjtObj.mjOBJ_JOINT, jn)]] for jn in joints},
                                  plant={jn: list(ranges[jn]) for jn in joints}),
                ctrl=dict(limited=False, home=[float(v) for v in m.key_ctrl[0]]),
                sensors=[dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i),
                              type=mujoco.mjtSensor(m.sensor_type[i]).name, dim=int(m.sensor_dim[i]))
                         for i in range(m.nsensor)])
    p = Plant(body, floor, seed, m, joints, drives, env, meta)
    if settle:
        meta["settle"] = p.settle(p._predicted_sinkage(floor_params, pile))
    return p


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def describe(p: Plant) -> dict:
    """``plant.json``: the plant's gains, envelopes, ranges, sensors and settle, with hashes, versions and commit."""
    return dict(p.meta, provenance=provenance(), environment=dict(omp_num_threads=os.environ.get("OMP_NUM_THREADS")))
