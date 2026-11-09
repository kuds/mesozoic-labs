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
* the two-motor ankle (``drive``; the record's ``leg_layout`` must be a parallel-linkage layout): each ankle's pitch
  and roll servos become motors on the tendons q_p + q_r and q_p - q_r.  The ankle armature (``ANKLE_ARMATURE``) is
  the linkage's, ``"linkage"`` (``STAGE1_PLAN.md`` decision 12; the default): each motor's armature on its own tendon
  (the value the screened MJCF writes on the ankle joint whose servo that motor replaces, i.e. the motor's catalog
  armature as written there), which in joint space is twice the screened MJCF's armature on pitch and on roll (both
  motors' rotors are geared to both axes).  ``"screened"`` (kept for G2's 2 x 2 plant matrix) keeps the screened
  MJCF's armature on the ankle joints (one motor's armature per joint) and none on the tendons, so the mass matrix
  and the feet's inverse weights are exactly the screened model's.  With ``"linkage"`` the feet's
  ``body_invweight0`` drops (9 % on the 60 cm bodies, 5 % on 75 cm): contacts with a time-constant solref (hard
  floor, robot_floor_carpet) are that much stiffer than in the screened model and in ``carpet.standing_check``, and
  the pile probes, re-derived on the plant model, keep carpet.py's stiffness while their damping grows as
  1 / sqrt(invweight0) and their regularisation inertia as 1 / invweight0 (5 % and 10 % on the 60 cm bodies);
  ``meta["ankle"]["armature"]["feet"]`` records these ratios per foot;
* joint ranges cut to the clearance screen's usable ranges (``usable_ranges``).

After compiling:

* hold-servo gains by the rule of ``carpet.hold_servos``, applied to each drive's own coordinate:
  ``kp = sustained / carpet.SERVO_ERROR_RAD`` (0.01 rad) and ``kv = 2 zeta sqrt(kp I)``, I the drive's inertia
  with every other drive held, at the home pose (for a joint servo the diagonal of the mass matrix, as
  ``hold_servos``; for an ankle motor that of the motor coordinates).  The ankle gains (``ANKLE_GAINS``) are in
  joint space, ``"joint"`` (``STAGE1_PLAN.md`` decision 11; the default): each ankle motor gets half its hold-servo
  kp, so pitch and roll get exactly the joint-space stiffness ``hold_servos`` gives one ankle joint (two motors share
  each axis).  The damping follows from the motors' own inertia, kv_a + kv_b on pitch and on roll alike, where
  ``hold_servos`` damps each joint by its own inertia; ``meta["gains"]["ankle_joint_space"]`` records both.
  ``"motor"`` (kept for G2's 2 x 2 plant matrix) gives each ankle motor its own full hold servo: twice the
  joint-space stiffness.  The MJCF's own gains (``mjcf.write_mjcf``: kp = max(5, 8 x rated torque), kv = 0.05 kp)
  are replaced (DECISION_LOG P23, decision 4);
* ctrl limits off: the servo target carries the feedforward, ``ctrl = G q* + (kv G qd* + f_ff) / kp`` in motor
  coordinates (``Plant.command``);
* the torque-speed envelope (``drive.Envelope``: driving side derated, braking at peak unless ``braking="derated"``),
  applied before every physics step, on the record's 24 V bus unless ``bus`` is given (plan section 5);
* the home keyframe's ctrl (the home pose in motor coordinates).

Floors: ``hard`` (as compiled), ``robot_floor_carpet`` (``carpet.floor_variants`` + ``carpet.apply_floor``) and
the probe piles ``low``, ``medium``, ``high`` (``carpet.floor_variants`` + ``carpet.pile_model``; ``pile_grid``
builds a coarser pile with ``carpet.pile_floor``).  carpet.py's caveat holds: the piles match static stiffness only.
The pile probes are re-derived on the plant model, so they keep carpet.py's stiffness per probe whatever the ankle
armature (their damping and regularisation follow the plant's feet, as above).  The toes and claws (``toe_geoms``)
are picked on the compiled hard-floor model, before ``carpet.pile_model`` takes the pads off the floor: the pads
under the toes (``pad_toe*``) would otherwise pass carpet.py's toe rule on a pile.

Every build settles the home pose with zero command (every servo at its home target), starting from the keyframe
lowered by the predicted sinkage, as ``carpet.standing_check`` starts; on 75-M-H-10 the keyframe pads sit 0.3 um
above the floor, so the stance has no contact before this settle.  The settle runs at least ``SETTLE_S`` (the plan's
1 s), then on in 1 s steps until ``carpet._settled`` holds over the last second (tilt span < 0.05 deg, CoM movement
< 0.5 mm, not speeding up), the stance tips (tilt from home above 12 deg or the CoM down by 0.1 H, carpet.py's
limits) or ``SETTLE_MAX_S`` passes.  ``meta["settle"]`` records the outcome and the residual rates; a plant whose
stance did not settle (``Plant.settled`` False: the 60 cm bodies on medium and high pile) does not stand on that
floor, and ``Plant.reset`` refuses to start an episode from it unless asked to.  ``Plant.reset`` otherwise returns
to the settled pose at rest.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from importlib import metadata
from pathlib import Path

import mujoco
import numpy as np

from .. import carpet, catalog, statics
from . import SHA256_PREFIX, STUDY, case_dir, drive

FLOORS = ("hard", "robot_floor_carpet") + carpet.PILES
#: The plant choices; the first of each is the one ``STAGE1_PLAN.md`` decided during the build (decision 11: ankle
#: gains in joint space; decision 12: the linkage's armature) and the default.  The others stay for G2's 2 x 2 plant
#: matrix (``gates.PLANT_CHOICES``).
ANKLE_GAINS = ("joint", "motor")
ANKLE_ARMATURE = ("linkage", "screened")
SETTLE_S = 1.0
SETTLE_MAX_S = 10.0
IMU_SITE = "imu"
#: Falls (plan section 4): trunk tilt from vertical above this, the root below FALL_HEIGHT_FRAC x its home height, or
#: a ``cl_*`` shell proxy below the floor.
FALL_TILT_RAD = 0.6
FALL_HEIGHT_FRAC = 0.7


class PlantError(RuntimeError):
    """The model on disk is not the one the record sized, a body input is missing, or a stance is not real."""


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


def leg_layout(body: str) -> str:
    """The record's leg layout; ``PlantError`` unless its ankle is the two-motor parallel linkage."""
    layout = record(body)["candidate"]["leg_layout"]
    if layout not in drive.LINKAGE_LAYOUTS:
        raise PlantError(f"{body}: leg layout {layout!r}; the plant models the two-motor ankle linkage of the "
                         f"{' and '.join(drive.LINKAGE_LAYOUTS)} layouts only")
    return layout


def _inter_leg(by) -> bool:
    """A limit set by the two legs touching: the interfering pair names a left_ and a right_ geom."""
    return bool(by) and any("left_" in g for g in by) and any("right_" in g for g in by)


def usable_ranges(body: str, inter_leg: str = "mjcf") -> dict:
    """joint -> (low, high): the clearance screen's single-joint sweeps (``summary.joint_range_usable``) read as the
    clearance and turning screens read them (``clearance._range_margins``, ``turning._range_margins``): the first new
    interference with the trunk or the same leg, else the MJCF limit.

    A limit set by the two legs touching (the left and right toes meeting in a sweep of one hip roll from home, with
    the other leg held at home) is not a range limit while walking, since the other leg moves; with ``inter_leg``
    ``"mjcf"`` (the screens' reading, the plant's) it falls back to the MJCF limit, and leg-to-leg proximity is
    checked on the simulated traces instead (``legs``).  ``"keep"`` keeps it (the literal reading of
    ``joint_range_usable``, for comparison).  The screens' reading is checked against the screen's own
    ``summary.joint_range_margins[joint]["usable"]``."""
    if inter_leg not in ("mjcf", "keep"):
        raise ValueError(f"inter_leg {inter_leg!r}: 'mjcf' or 'keep'")
    doc = json.loads((case_dir(body) / "review" / "screening" / f"{body}_clearance.json").read_text())
    margins = doc["summary"].get("joint_range_margins") or {}
    out = {}
    for jn, r in doc["summary"]["joint_range_usable"].items():
        lim = []
        for i in (0, 1):
            fni, by = r["first_new_interference"][i], r["limited_by"][i]
            drop = fni is None or (inter_leg == "mjcf" and _inter_leg(by))
            lim.append(float(r["limits"][i] if drop else fni))
        if inter_leg == "mjcf" and jn in margins and [float(v) for v in margins[jn]["usable"]] != lim:
            raise PlantError(f"{body} {jn}: usable range {lim} differs from the clearance screen's own "
                             f"{margins[jn]['usable']}")
        out[jn] = (lim[0], lim[1])
    return out


def dropped_limits(body: str) -> dict:
    """joint -> [(direction, first new interference, interfering geoms)] of the inter-leg limits ``usable_ranges``
    drops."""
    doc = json.loads((case_dir(body) / "review" / "screening" / f"{body}_clearance.json").read_text())
    out = {}
    for jn, r in doc["summary"]["joint_range_usable"].items():
        for i, direction in enumerate(("low", "high")):
            fni, by = r["first_new_interference"][i], r["limited_by"][i]
            if fni is not None and _inter_leg(by):
                out.setdefault(jn, []).append(dict(direction=direction, first_new_interference=float(fni),
                                                   limited_by=list(by), plant=float(r["limits"][i])))
    return out


def imu_position(ledger: Path) -> list:
    rows = [r for r in json.loads(Path(ledger).read_text())["rows"] if r["name"] == "imu"]
    if len(rows) != 1 or rows[0]["body"] != "trunk":
        raise PlantError(f"{ledger}: expected one 'imu' row on the trunk")
    return [float(v) for v in rows[0]["com_body"]]


#: The Stage 1 package's sources (``sources_sha256``): the code a run record names.
SOURCES = Path(__file__).resolve().parent
SOURCES_RULE = ("sha256 over designlab/stage1/*.py in path order, each file as its path relative to the package, a "
                "NUL byte, its bytes and a NUL byte")


def sources_sha256(files: dict[str, bytes] | None = None) -> str:
    """The hash of the Stage 1 sources (``SOURCES_RULE``): ``files`` maps each relative path to its bytes (default:
    the package's ``*.py`` files as they are on disk)."""
    if files is None:
        files = {p.name: p.read_bytes() for p in SOURCES.glob("*.py")}
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b"\0" + files[name] + b"\0")
    return h.hexdigest()


def sources_sha256_at(commit: str) -> str | None:
    """``sources_sha256`` of the package as committed in ``commit`` (None if git cannot tell)."""
    rel = SOURCES.relative_to(STUDY).as_posix()
    try:
        names = subprocess.run(["git", "-C", str(STUDY), "ls-tree", "--name-only", commit, f"{rel}/"],
                               capture_output=True, text=True, check=True).stdout.split()
        files = {}
        for path in names:
            if path.endswith(".py") and "/" not in path[len(rel) + 1:]:
                files[path[len(rel) + 1:]] = subprocess.run(["git", "-C", str(STUDY), "show", f"{commit}:./{path}"],
                                                            capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return sources_sha256(files) if files else None


def provenance() -> dict:
    """Library versions, the repository commit with a dirty flag (any change under hardware/trex_form_study), and the
    hash of the Stage 1 sources that ran (``sources_sha256``), which names the code even in a dirty tree."""
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
    return dict(versions=vers, git=git, stage1_sources=dict(sha256=sources_sha256(), rule=SOURCES_RULE,
                                                            files=sorted(p.name for p in SOURCES.glob("*.py"))))


# ------------------------------------------------------------------------------------------------
# Model text
# ------------------------------------------------------------------------------------------------
def augment(text: str, joints: list[str], drives: list[drive.Drive], ranges: dict, imu_pos,
            ankle_armature: str = "linkage") -> str:
    """The screened MJCF text with the plant's sensors, ankle tendons, motors and usable joint ranges."""
    if ankle_armature not in ANKLE_ARMATURE:
        raise ValueError(f"ankle_armature {ankle_armature!r}: choose from {', '.join(ANKLE_ARMATURE)}")
    root = ET.fromstring(text)
    by_coord = {d.coordinate: d for d in drives}
    screened_arm = {}  # the armature the screened MJCF gives each ankle joint: the motor that replaces its servo's
    for j in root.iter("joint"):
        name = j.get("name")
        if name in ranges:
            j.set("range", f"{ranges[name][0]!r} {ranges[name][1]!r}")
        if name is not None and name.endswith(("_ankle_pitch", "_ankle_roll")):
            screened_arm[name] = j.get("armature")
            if ankle_armature == "linkage":
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
            written = screened_arm.get(f"{side}_{drive.REPLACES[motor]}")
            arm = "0" if ankle_armature == "screened" else written or repr(by_coord[name].armature)
            t = ET.SubElement(tendon, "fixed", name=name, armature=arm)
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


def _dofs(m, joints: list[str]) -> list[int]:
    return [int(m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)]) for jn in joints]


def joint_armature(m, joints: list[str]) -> np.ndarray:
    """Joint-space armature over ``joints`` (dof armature plus each tendon's armature through its Jacobian): the
    mass matrix at the keyframe minus the same with every armature set to zero."""
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    saved = m.dof_armature.copy(), m.tendon_armature.copy()
    dofs = _dofs(m, joints)
    out = []
    try:
        for zero in (False, True):
            if zero:
                m.dof_armature[:] = 0.0
                m.tendon_armature[:] = 0.0
            mujoco.mj_forward(m, d)
            M = np.zeros((m.nv, m.nv))
            mujoco.mj_fullM(m, d, M)
            out.append(M[np.ix_(dofs, dofs)])
    finally:
        m.dof_armature[:], m.tendon_armature[:] = saved
    return out[0] - out[1]


def _home_inertia(m, joints: list[str]) -> np.ndarray:
    d = mujoco.MjData(m)
    d.qpos[:] = m.key_qpos[0]
    mujoco.mj_forward(m, d)
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, M)
    return np.diag(M)[_dofs(m, joints)]


def armature_effects(m0, m, joints: list[str], choice: str) -> dict:
    """What the plant's ankle armature changes against the screened model ``m0`` (module docstring).

    Per foot, plant over screened model: the stiffness of a time-constant contact (hard floor, robot_floor_carpet),
    and the stiffness, damping and regularisation inertia of the pile probes.  ``carpet.pile_floor`` derives every
    probe (time constant ``1 / (zeta sqrt(K (1 - d0) A_hat))``) from the left foot's ``A_hat = invweight0``, and
    MuJoCo regularises each contact with its own foot's: force per depth ``K A_hat / A``, damping
    ``2 zeta sqrt(K A_hat / (1 - d0)) / A`` and regularisation inertia ``d0 / ((1 - d0) A)``, A the foot's own
    inverse weight."""
    A, A0 = joint_armature(m, joints), joint_armature(m0, joints)
    inertia, inertia0 = _home_inertia(m, joints), _home_inertia(m0, joints)
    ankles = [i for i, jn in enumerate(joints) if jn.endswith(("_ankle_pitch", "_ankle_roll"))]
    left = statics.bid(m, "left_foot")
    a_hat, a_hat0 = float(m.body_invweight0[left, 0]), float(m0.body_invweight0[left, 0])  # carpet.pile_floor's
    feet = {}
    for side in statics.SIDES:
        b = statics.bid(m, f"{side}_foot")
        w, w0 = float(m.body_invweight0[b, 0]), float(m0.body_invweight0[b, 0])
        feet[side] = dict(invweight0_translation_plant=w, invweight0_translation_screened=w0,
                          change_frac=w / w0 - 1.0, time_constant_contact_stiffness_ratio=w0 / w,
                          pile_probe_stiffness_ratio=(a_hat / w) / (a_hat0 / w0),
                          pile_probe_damping_ratio=math.sqrt(a_hat / a_hat0) * w0 / w,
                          pile_probe_regularisation_ratio=w0 / w)
    return dict(choice=choice,
                source="STAGE1_PLAN.md decision 12" if choice == ANKLE_ARMATURE[0] else "argument (not the decided "
                                                                                       "choice)",
                rule="each motor's armature (as the screened MJCF writes it) on its tendon (joint space: 2x the "
                     "screened per-joint armature on pitch and roll)" if choice == "linkage" else
                     "the screened MJCF's armature on each ankle joint, none on the tendons (the screened mass matrix)",
                joint_space_plant={joints[i]: float(A[i, i]) for i in ankles},
                joint_space_screened={joints[i]: float(A0[i, i]) for i in ankles},
                ankle_inertia_at_home_plant={joints[i]: float(inertia[i]) for i in ankles},
                ankle_inertia_at_home_screened={joints[i]: float(inertia0[i]) for i in ankles},
                ankle_inertia_change_frac={joints[i]: float(inertia[i] / inertia0[i] - 1.0) for i in ankles},
                feet=feet,
                contacts="hard floor and robot_floor_carpet use time-constant solrefs, whose force per depth scales "
                         "with 1 / invweight0: time_constant_contact_stiffness_ratio is the plant's foot contact "
                         "stiffness over the screened model's (and carpet.standing_check's); the piles' probes are "
                         "re-derived on the plant model (carpet.pile_floor, from the left foot's invweight0) and keep "
                         "carpet.py's stiffness (pile_probe_stiffness_ratio), while their damping scales with "
                         "1 / sqrt(invweight0) and their regularisation inertia d0 / ((1 - d0) invweight0) with "
                         "1 / invweight0: pile_probe_damping_ratio and pile_probe_regularisation_ratio are the plant's "
                         "over the screened model's (MuJoCo's efc_KBIP and efc_R give the same ratios)",
                feedforward="the references' inverse dynamics has the screened armature; replays add "
                            "(A_plant - A_screened) qdd_ref (gates.replay)")


def toe_geoms(m) -> list[str]:
    """Names of the feet's toe and claw geoms (``carpet._is_toe``'s rule, on a foot body, with extent), in model order.

    Read on the compiled hard-floor model: ``carpet.pile_model`` turns the pads' contype to 0, and the pads under the
    toes (``pad_toe*``) would then pass the rule; the pile probes are unnamed."""
    feet = {statics.bid(m, f"{s}_foot") for s in statics.SIDES}
    out = []
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if (m.geom_bodyid[g] in feet and m.geom_contype[g] != 1 and m.geom_rbound[g] > 0
                and any(t in name for t in carpet.TOE_TOKENS)):
            out.append(name)
    return out


class _SelectionDesign:
    """``build.Design.act`` from a record's ``selection`` (the drive of a sizing role), for ``carpet.hold_servos``."""

    def __init__(self, selection: dict):
        self.selection = selection

    def act(self, role: str) -> catalog.Actuator:
        sel = self.selection
        return catalog.actuator(sel["joints"][role] if role in sel["joints"] else sel[role])


def hold_servo_reference(m0, selection: dict) -> dict:
    """joint -> ``carpet.hold_servos`` gains (kp, kv, joint inertia) on a copy of the screened model ``m0``: one hold
    servo per joint, the screened stance's servos."""
    m = copy.copy(m0)
    return carpet.hold_servos(m, mujoco.MjData(m), m.key_qpos[0].copy(), _SelectionDesign(selection))


def ankle_joint_space(joints: list[str], G: np.ndarray, kp, kv, hold: dict) -> dict:
    """Each ankle's servo stiffness and damping in joint space (``G^T diag(kp) G`` and ``G^T diag(kv) G`` over its
    pitch and roll; ``pitch_roll`` is the coupling) against ``hold_servos``' one servo per joint (``hold``,
    ``hold_servo_reference``)."""
    kp, kv = np.asarray(kp, float), np.asarray(kv, float)
    out = {}
    for side in statics.SIDES:
        names = (f"{side}_ankle_pitch", f"{side}_ankle_roll")
        idx = [joints.index(jn) for jn in names]
        Gs = G[np.ix_(idx, idx)]
        K, D = Gs.T @ np.diag(kp[idx]) @ Gs, Gs.T @ np.diag(kv[idx]) @ Gs
        hs = dict(zip(("pitch", "roll"), (hold[jn] for jn in names), strict=True))
        out[side] = dict(
            stiffness_Nm_per_rad=dict(pitch=float(K[0, 0]), roll=float(K[1, 1]), pitch_roll=float(K[0, 1])),
            damping_Nms_per_rad=dict(pitch=float(D[0, 0]), roll=float(D[1, 1]), pitch_roll=float(D[0, 1])),
            hold_servos={ax: dict(kp=float(h["kp"]), kv=float(h["kv"]), joint_inertia_kg_m2=float(h["joint_inertia"]))
                         for ax, h in hs.items()},
            stiffness_over_hold_servos=dict(pitch=float(K[0, 0] / hs["pitch"]["kp"]),
                                            roll=float(K[1, 1] / hs["roll"]["kp"])),
            damping_over_hold_servos=dict(pitch=float(D[0, 0] / hs["pitch"]["kv"]),
                                          roll=float(D[1, 1] / hs["roll"]["kv"])))
    return out


# ------------------------------------------------------------------------------------------------
# Plant
# ------------------------------------------------------------------------------------------------
class Plant:
    """One body on one floor: ``m``, ``d``, the drives and their envelope, the settled home pose and metadata.

    ``joints`` are the 14 hinges in model order; joint-space arrays (targets, encoders) use that order, and the
    drives (actuators) follow it with each ankle's pitch and roll replaced by its motors a and b.  The model belongs
    to this plant alone (the envelope writes its force ranges); ``reset`` and ``set_state`` re-bind the envelope to
    ``m`` and ``d``.  ``armature_extra`` is the plant's joint-space armature minus the screened model's (14 x 14),
    the term the references' inverse dynamics lacks.  ``toes`` names the toe and claw geoms (``toe_geoms`` on the
    hard-floor model; ``build`` passes them, since on a pile ``m``'s pads no longer tell themselves apart by
    contype); None reads them from ``m``, right only for a model whose pads keep contype 1."""

    def __init__(self, body: str, floor: str, seed: int, m: mujoco.MjModel, joints: list[str],
                 drives: list[drive.Drive], envelope: drive.Envelope, meta: dict, layout: str = "tandem",
                 armature_extra: np.ndarray | None = None, toes: list[str] | None = None):
        self.body, self.floor, self.seed = body, floor, int(seed)
        self.m, self.d = m, mujoco.MjData(m)
        self.joints, self.drives, self.envelope, self.meta = joints, drives, envelope, meta
        envelope.bind(self.m, self.d)
        self.rng = np.random.default_rng(seed)
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn) for jn in joints]
        self.qadr = np.array([m.jnt_qposadr[j] for j in jid])
        self.dofs = np.array([m.jnt_dofadr[j] for j in jid])
        self.G = drive.motor_matrix(joints, layout)
        self._GinvT = np.linalg.inv(self.G.T)
        self.armature_extra = (np.zeros((len(joints), len(joints))) if armature_extra is None
                               else np.asarray(armature_extra, float))
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
        fl = statics.gid(m, "floor")
        if m.geom_type[fl] != mujoco.mjtGeom.mjGEOM_PLANE:
            raise PlantError("the floor is not a plane")
        self.z_floor = float(m.geom_pos[fl][2])
        gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(m.ngeom)]
        self._gname = gname
        self._cl = np.array([g for g in range(m.ngeom) if gname[g].startswith("cl_")], int)
        self._cl_r = m.geom_rbound[self._cl].copy()
        toes = toe_geoms(m) if toes is None else list(toes)
        self._toes = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, t) for t in toes], int)
        if np.any(self._toes < 0) or np.any(np.isin(m.geom_bodyid[self._toes], self.feet, invert=True)):
            raise PlantError(f"toe geoms {toes} are not all geoms of the feet of this model")
        self._toe_aabb = m.geom_aabb[self._toes].copy()
        self._toe_size = m.geom_size[self._toes].copy()
        self._toe_capsule = m.geom_type[self._toes] == mujoco.mjtGeom.mjGEOM_CAPSULE
        self.q_settled = self.q_home.copy()

    # --------------------------------------------------------------- state
    def _trunk_R(self, q) -> np.ndarray:
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q[3:7])
        return R.reshape(3, 3)

    @property
    def settled(self) -> bool | None:
        """True if the zero-command stance settled on this floor, False if it did not, None if no settle was run."""
        s = self.meta.get("settle")
        return None if s is None else bool(s["settled"])

    def reset(self, joint_noise: float = 0.0, root_velocity_noise: float = 0.0, seed: int | None = None,
              allow_unsettled: bool = False) -> None:
        """Back to the settled home pose at rest, servos at home; optional uniform noise drawn from ``rng``, which
        ``seed`` replaces with a fresh generator (one per episode, so an episode does not depend on the draws of the
        ones before it).  Refuses a stance that did not settle unless ``allow_unsettled``."""
        if self.settled is False and not allow_unsettled:
            s = self.meta["settle"]
            raise PlantError(f"{self.body} does not stand on {self.floor}: the zero-command stance {s['status']} "
                             f"after {s['seconds']:.1f} s; no settled stance to start from")
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.q_settled
        if joint_noise:
            d.qpos[self.qadr] += self.rng.uniform(-joint_noise, joint_noise, len(self.qadr))
        if root_velocity_noise:
            d.qvel[:3] = self.rng.uniform(-root_velocity_noise, root_velocity_noise, 3)
        d.ctrl[:] = self.ctrl_home
        self.envelope.bind(m, d)
        mujoco.mj_forward(m, d)

    def set_state(self, qpos, qvel) -> None:
        """Start from ``qpos`` and ``qvel`` (e.g. a reference frame), servos at home."""
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = qpos
        self.d.qvel[:] = qvel
        self.d.ctrl[:] = self.ctrl_home
        self.envelope.bind(self.m, self.d)
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
        return math.acos(max(-1.0, min(1.0, float(self.d.xmat[self.trunk][8]))))

    def tilt_from_home(self) -> float:
        """Trunk tilt relative to the home orientation (rad), as ``carpet.standing_check`` measures it."""
        Rw = self.d.xmat[self.trunk].reshape(3, 3) @ self._R_home.T
        return float(np.arccos(np.clip(Rw[2, 2], -1.0, 1.0)))

    def proxy_below_floor(self) -> str | None:
        """The first ``cl_*`` shell proxy whose lowest point is below the floor, else None (positions as of the last
        step's start, like the sensors).  A bounding-sphere test first; exact lowest points only near the floor."""
        d = self.d
        if not len(self._cl):
            return None
        z = d.geom_xpos[self._cl, 2] - self._cl_r
        if z.min() > self.z_floor:
            return None
        for g in self._cl[z <= self.z_floor]:
            if carpet.lowest_point(self.m, d, g)[2] < self.z_floor:
                return self._gname[g]
        return None

    def fall_cause(self) -> str | None:
        """Why the episode counts as a fall (plan section 4), or None: "tilt", "height" or "proxy <geom>"."""
        if self.tilt() > FALL_TILT_RAD:
            return "tilt"
        if float(self.d.qpos[2]) < FALL_HEIGHT_FRAC * self.z_home:
            return "height"
        g = self.proxy_below_floor()
        return None if g is None else f"proxy {g}"

    def fallen(self) -> bool:
        return self.fall_cause() is not None

    def toe_lowest(self) -> tuple[float, str | None]:
        """Height of the lowest toe or claw point (``toe_geoms``, the same on every floor) above the floor (m;
        negative: through it, since toes and claws do not collide) and its geom (positions as of the last step's
        start).  Exact: in closed form for capsules (every
        toe and claw of these bodies); any other geom by ``carpet.lowest_point``, in the order of the world-AABB lower
        bounds until no remaining bound is lower."""
        d, g = self.d, self._toes
        if not len(g):
            return math.inf, None
        cap = self._toe_capsule
        lb = d.geom_xpos[g, 2] - np.abs(d.geom_xmat[g, 8]) * self._toe_size[:, 1] - self._toe_size[:, 0]
        if not cap.all():
            r2 = d.geom_xmat[g, 6:9]
            lb = np.where(cap, lb, d.geom_xpos[g, 2] + np.einsum("ij,ij->i", r2, self._toe_aabb[:, :3])
                          - np.einsum("ij,ij->i", np.abs(r2), self._toe_aabb[:, 3:]))
        i = int(np.argmin(lb))
        best, at = float(lb[i]), int(g[i])
        if not cap.all():
            best = math.inf
            for i in np.argsort(lb):
                if lb[i] >= best:
                    break
                z = float(lb[i]) if cap[i] else float(carpet.lowest_point(self.m, d, g[i])[2])
                if z < best:
                    best, at = z, int(g[i])
        return best - self.z_floor, self._gname[at]

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

    def settle(self, sinkage: float, H: float, min_seconds: float = SETTLE_S, max_seconds: float = SETTLE_MAX_S) -> dict:
        """Settle the home pose with zero command (module docstring) and keep it as the start of every episode."""
        m, d = self.m, self.d
        mujoco.mj_resetDataKeyframe(m, d, 0)
        self.envelope.bind(m, d)
        mujoco.mj_forward(m, d)
        ncon_key = [int(v) for v in self.sensors()["foot_contacts"]]
        d.qpos[2] -= sinkage
        mujoco.mj_forward(m, d)
        ncon0 = [int(v) for v in self.sensors()["foot_contacts"]]
        dt = float(m.opt.timestep)
        every = max(1, int(round(carpet.LOG_DT / dt)))
        per_s = int(round(1.0 / dt))
        lag = per_s // every
        n_min, n_max = int(round(min_seconds / dt)), int(round(max(min_seconds, max_seconds) / dt))
        tilt, com = [float(np.degrees(self.tilt_from_home()))], [d.subtree_com[self.trunk].copy()]
        com0 = com[0].copy()
        status, k = "did not settle", 0
        while True:
            self.step(every)
            k += every
            tilt.append(float(np.degrees(self.tilt_from_home())))
            com.append(d.subtree_com[self.trunk].copy())
            if tilt[-1] > carpet.UPRIGHT_TILT_DEG or com0[2] - com[-1][2] > carpet.COLLAPSE_FRAC_H * H:
                status = "tipped"
                break
            if k >= n_min and k % per_s == 0 and len(tilt) > 2 * lag and carpet._settled(tilt, com, lag):
                status = "settled"
                break
            if k >= n_max:
                break
        s = self.sensors()
        span_tilt, span_com = carpet._spans(tilt[-1 - min(lag, len(tilt) - 1) :], com[-1 - min(lag, len(com) - 1) :])
        out = dict(status=status, settled=status == "settled", seconds=k * dt, min_seconds=min_seconds,
                   max_seconds=max_seconds,
                   rule="carpet._settled over the last second (tilt span < "
                        f"{carpet.SETTLED_TILT_DEG} deg, CoM movement < {carpet.SETTLED_COM_MM} mm, not speeding up), "
                        f"checked each second from 2 s; tipped: tilt from home > {carpet.UPRIGHT_TILT_DEG} deg or CoM "
                        f"down by {carpet.COLLAPSE_FRAC_H} H",
                   foot_contacts_at_keyframe=ncon_key, start_lowered_by_m=sinkage,
                   foot_contacts_at_start=ncon0, foot_contacts_settled=[int(v) for v in s["foot_contacts"]],
                   foot_load_N=[float(v) for v in s["foot_load"]],
                   weight_N=float(m.body_subtreemass[self.trunk]) * carpet.G,
                   root_z_change_m=float(d.qpos[2] - self.z_home), tilt_from_home_deg=tilt[-1],
                   tilt_change_last_s_deg=span_tilt, com_move_last_s_mm=span_com,
                   max_abs_qvel=float(np.max(np.abs(d.qvel))),
                   trunk_rate_rad_s=float(np.linalg.norm(d.qvel[3:6])),
                   max_abs_joint_rate_rad_s=float(np.max(np.abs(d.qvel[self.dofs]))),
                   joint_offset_max_rad=float(np.max(np.abs(d.qpos[self.qadr] - self.q_home[self.qadr]))))
        self.q_settled = d.qpos.copy()
        self.meta["settle"] = out
        self.reset(allow_unsettled=True)
        return out


def _gains(m, d, joints: list[str], G: np.ndarray, drives: list[drive.Drive], ankle_gains: str = ANKLE_GAINS[0]) -> dict:
    """Hold-servo gains (module docstring) at the home pose, written into the model's position actuators."""
    if ankle_gains not in ANKLE_GAINS:
        raise ValueError(f"ankle_gains {ankle_gains!r}: choose from {', '.join(ANKLE_GAINS)}")
    d.qpos[:] = m.key_qpos[0]
    d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    Mfull = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, d, Mfull)
    dofs = _dofs(m, joints)
    Ginv = np.linalg.inv(G)
    inertia = np.diag(Ginv.T @ Mfull[np.ix_(dofs, dofs)] @ Ginv).copy()
    kp = np.array([dr.sustained for dr in drives]) / carpet.SERVO_ERROR_RAD
    if ankle_gains == "joint":  # G^T diag(kp) G on each ankle = the one-joint hold stiffness sustained / 0.01 rad
        ankle = [i for i, dr in enumerate(drives) if dr.group in drive.REPLACES]
        if len({drives[i].sustained for i in ankle}) > 1:
            raise PlantError("joint-space ankle gains need the two ankle motors to share one sustained torque")
        kp[ankle] *= 0.5
    kv = 2.0 * carpet.SERVO_ZETA * np.sqrt(kp * inertia)
    m.actuator_gainprm[:, 0] = kp
    m.actuator_biasprm[:, 1] = -kp
    m.actuator_biasprm[:, 2] = -kv
    m.actuator_ctrllimited[:] = 0
    return dict(kp=kp, kv=kv, inertia=inertia)


def build(body: str, floor: str = "hard", seed: int = 0, envelope: bool = True, speed_scale: float = 1.0,
          torque_scale: float = 1.0, pile_grid: float | None = None, settle: bool = True,
          ankle_gains: str = ANKLE_GAINS[0], ankle_armature: str = ANKLE_ARMATURE[0], bus: float | None = None,
          braking: str = "peak") -> Plant:
    """The Stage 1 plant of ``body`` on ``floor`` (``FLOORS``), settled at home unless ``settle`` is False.

    ``ankle_gains`` and ``ankle_armature`` are the two plant choices ``STAGE1_PLAN.md`` decisions 11 and 12 settle
    (module docstring; the defaults are the decided ones, the others are kept for G2's 2 x 2 plant matrix); ``bus``
    (V) replaces the record's 24 V bus in the envelope and ``braking`` selects the envelope's braking rule
    (``drive``)."""
    if floor not in FLOORS:
        raise ValueError(f"floor {floor!r}: choose from {', '.join(FLOORS)}")
    if ankle_gains not in ANKLE_GAINS:
        raise ValueError(f"ankle_gains {ankle_gains!r}: choose from {', '.join(ANKLE_GAINS)}")
    xml, ledger = model_files(body)
    check = verify(body, xml)
    rec = record(body)
    layout = leg_layout(body)
    ranges = usable_ranges(body)
    imu = imu_position(ledger)
    text = xml.read_text()
    m0 = mujoco.MjModel.from_xml_string(text)
    joints = _hinges(m0)
    drives = drive.drives(rec["selection"], joints, bus=bus, layout=layout)
    aug = augment(text, joints, drives, ranges, imu, ankle_armature)
    m = mujoco.MjModel.from_xml_string(aug)
    q_home = m.key_qpos[0].copy()
    toes = toe_geoms(m)  # on the hard-floor model, before any pile takes the pads off the floor
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
    G = drive.motor_matrix(joints, layout)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    moment = _moment(m, d)
    dofs = _dofs(m, joints)
    embed = np.zeros((m.nu, m.nv))
    embed[:, dofs] = G
    if not np.array_equal(moment, embed):
        raise PlantError("the compiled actuator transmission differs from drive.motor_matrix")
    gains = _gains(m, d, joints, G, drives, ankle_gains)
    m.key_ctrl[0] = G @ q_home[[m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)] for jn in joints]]
    env = drive.Envelope(drives, moment, enabled=envelope, speed_scale=speed_scale, torque_scale=torque_scale,
                         braking=braking)
    arm = armature_effects(m0, m, joints, ankle_armature)
    extra = joint_armature(m, joints) - joint_armature(m0, joints)
    ankle_js = ankle_joint_space(joints, G, gains["kp"], gains["kv"], hold_servo_reference(m0, rec["selection"]))
    bus_v = float(rec["selection"]["bus"] if bus is None else bus)
    meta = dict(body=body, floor=floor, seed=int(seed), model=check, H_m=float(rec["H"]), leg_layout=layout,
                options=dict(ankle_gains=ankle_gains, ankle_armature=ankle_armature, bus_V=bus_v,
                             bus_source="record selection.bus" if bus is None else "argument", braking=braking,
                             envelope=bool(envelope), speed_scale=float(speed_scale), torque_scale=float(torque_scale),
                             pile_grid_m=pile_grid),
                mass_kg=float(m.body_subtreemass[statics.bid(m, "trunk")]), timestep_s=float(m.opt.timestep),
                integrator=mujoco.mjtIntegrator(m.opt.integrator).name, nq=int(m.nq), nv=int(m.nv), nu=int(m.nu),
                ngeom=int(m.ngeom), imu_site_trunk_m=imu, floor_params=_jsonable(floor_params), floor_info=floor_info,
                gains=dict(rule="carpet.hold_servos on each drive's own coordinate: kp = sustained / "
                                f"{carpet.SERVO_ERROR_RAD} rad, kv = 2 x {carpet.SERVO_ZETA} sqrt(kp I) at home, "
                                "I with every other drive held",
                           ankle_gains=ankle_gains,
                           ankle_gains_source="STAGE1_PLAN.md decision 11" if ankle_gains == ANKLE_GAINS[0] else
                           "argument (not the decided choice)",
                           ankle_joint_space_stiffness="2 kp (two motors per axis)" if ankle_gains == "motor" else
                           "kp of one hold servo (each motor at half kp)",
                           ankle_joint_space=dict(
                               rule="G^T diag(kp) G and G^T diag(kv) G over each ankle's pitch and roll (pitch_roll: "
                                    "the coupling), against carpet.hold_servos' one servo per joint on the screened "
                                    "model (kv = 2 zeta sqrt(kp M_jj), M with the screened armature); the plant's kv "
                                    "comes from each motor's own inertia, so pitch and roll get the same damping",
                               **ankle_js),
                           drives={dr.actuator: dict(kp=float(kp), kv=float(kv), inertia_kg_m2=float(i),
                                                     coordinate=dr.coordinate)
                                   for dr, kp, kv, i in zip(drives, gains["kp"], gains["kv"], gains["inertia"],
                                                            strict=True)},
                           mjcf_own={mujoco.mj_id2name(m0, mujoco.mjtObj.mjOBJ_ACTUATOR, a):
                                     dict(kp=float(m0.actuator_gainprm[a, 0]), kv=float(-m0.actuator_biasprm[a, 2]))
                                     for a in range(m0.nu)}),
                drives=[dr.as_dict() for dr in drives], envelope=env.describe(),
                ankle=dict(linkage="motor a on q_p + q_r, motor b on q_p - q_r (unit cranks): "
                                   "|tau_p| + |tau_r| <= 2 peak",
                           armature=arm),
                joint_ranges=dict(source="review/screening/<id>_clearance.json summary.joint_range_usable",
                                  reading="read as the clearance and turning screens read it (clearance._range_margins):"
                                          " the first new interference with the trunk or the same leg, else the MJCF "
                                          "limit; a limit set by the two legs touching in a single-joint sweep from "
                                          "home falls back to the MJCF limit (only the pads collide in the plant, so "
                                          "leg-to-leg proximity is checked on the traces: legs.LegGap)",
                                  mjcf={jn: [float(v) for v in m0.jnt_range[mujoco.mj_name2id(
                                      m0, mujoco.mjtObj.mjOBJ_JOINT, jn)]] for jn in joints},
                                  plant={jn: list(ranges[jn]) for jn in joints},
                                  dropped_inter_leg_limits=dropped_limits(body)),
                ctrl=dict(limited=False, home=[float(v) for v in m.key_ctrl[0]]),
                sensors=[dict(name=mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i),
                              type=mujoco.mjtSensor(m.sensor_type[i]).name, dim=int(m.sensor_dim[i]))
                         for i in range(m.nsensor)],
                falls=dict(tilt_rad=FALL_TILT_RAD, root_height_frac=FALL_HEIGHT_FRAC,
                           proxies="any cl_* shell proxy below the floor plane"))
    p = Plant(body, floor, seed, m, joints, drives, env, meta, layout, extra, toes)
    if settle:
        p.settle(p._predicted_sinkage(floor_params, pile), float(rec["H"]))
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
