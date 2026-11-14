"""Stage 1 references: the designlab reference walks and turns with their inverse-dynamics feedforward
(``STAGE1_PLAN.md`` section 2; gate G2 in section 3).

Each reference is one periodic stride regenerated exactly as the round-two sizing and screens made it, at round
two's walk accuracy (``gait.set_accuracy("r2")``), on the record's own model and home pose.  The body is rebuilt
from its record with ``pipeline.build_candidate`` (the call of ``refine.carpet_lift_one``) in a temporary
directory, the rebuilt MJCF is checked against the record's ``xml_sha256``, and its stance comes from the solver at
full precision (the MJCF keyframe rounds it to 6 significant digits).

* Straight walks (``gait.generate`` + ``gait.inverse_dynamics`` + ``gait.summarize``): ``nominal``
  (``sizing.NOMINAL``), ``brisk`` (``sizing.BRISK``), ``fr05`` (``envelope.FAST``, Froude 0.05) and ``carpet``,
  the nominal walk with the carpet swing (``lift_frac`` from ``CARPET_LIFT``: 0.19 / 0.195 / 0.18 H).
* Steady turns (``turning.generate`` + ``turning.inverse_dynamics`` + ``turning.summarize`` with the turning
  screen's settings, ``turning.screen_model``): every case of ``turning.CASES`` in both directions, ``<case>_left``
  (positive yaw rate) and ``<case>_right``.

The feedforward ``tau`` is the inverse-dynamics joint torque (contact-consistent; the MJCF's joint damping and
friction are not in it).  Each summary is compared leaf by leaf with its record: the record's ``gaits`` for the
straight walks (for the carpet swing, where the record sized that lift: ``lift19`` on 60-N-T-10), the turning
screen's ``runs`` for the turns, and the carpet-lift pass (``round2/refine/carpet_lift_*.json``) for the carpet
swing's toe clearance.  A leaf matches when it agrees to 6 significant digits (``REL_TOL``).  Leaves that are only
numerical noise of the IK and the inverse dynamics (the hip-yaw speed peak of a straight walk, about 2e-7 rad/s, and
its power, about 2e-8 W) have no physical sixth digit: a leaf whose record and regenerated values both lie within the
noise floor of its unit (``NOISE_FLOORS``: 1e-6 rad/s, 1e-6 W, 1e-6 N m, 1e-7 m) matches whatever its digits, and is
listed as such (``noise_floor_matches``); on this machine every leaf is bit-identical, so none needs the floor.

Files: ``refs.json`` (metadata and the record comparisons) and ``refs/<name>.npz`` (arrays; ``Reference.load``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from .. import carpet, envelope, gait, sizing, statics, turning
from . import BODIES, STUDY, candidate_id, case_dir, plant

ACCURACY = "r2"
#: The carpet swing per body (round-two carpet-lift pass): the lowest lift that meets the 0.06 H target above every
#: pile within the drives and joint ranges (ROUND2_REVIEW, DECISION_LOG P16).
#: Stage 1b's arm K bodies take their picks' lifts, the lowest that meet the target there too (the same toe clearance as
#: the picks'), but no lift is within their joint ranges: the faster knee's upper-thigh shell stops the knee at 0.855 rad
#: (the picks': 1.00-1.05), so at 0.19-0.195 H the walking clearance gates fail (60-N-T-10-K keeps 0.0065 rad of knee
#: range at 0.19 H, under the 0.10 rad gate; 60-N-H-10-K's hip pitch goes 0.08 rad past its usable range at 0.195 H).
#: Their carpet reference is outside this rule, kept only so that a K body has its pick's references.
#: Stage 2's expressive-head bodies (S2-2) take their picks' lifts; the same carpet-lift pass, run on their own cases
#: with the picks' lifts and walks (``stage2/x/run_x.sh build``), is their record, and the rule above reads the same
#: lift from it (``tests/test_stage2_x.py``).
CARPET_LIFT = {"60-N-T-10": 0.19, "60-N-H-10": 0.195, "75-M-H-10": 0.18, "60-N-T-10-K": 0.19, "60-N-H-10-K": 0.195,
               "60-N-T-10-X": 0.19, "75-M-H-10-X": 0.18}
CARPET_LIFT_RECORD = {"60-N-T-10": "round2/refine/carpet_lift_narrow_carpet_60_0.190_0.195_nominal_only.json",
                      "60-N-H-10": "round2/refine/carpet_lift_narrow_carpet_60_0.190_0.195_nominal_only.json",
                      "75-M-H-10": "round2/refine/carpet_lift_feet_75_0.175_0.180_0.190.json",
                      "60-N-T-10-K": "round2/refine/carpet_lift_narrow_knee_60_0.190_0.195_nominal_only.json",
                      "60-N-H-10-K": "round2/refine/carpet_lift_narrow_knee_60_0.190_0.195_nominal_only.json",
                      "60-N-T-10-X": "round2/refine/carpet_lift_head_x_60_0.190_0.195_nominal_only.json",
                      "75-M-H-10-X": "round2/refine/carpet_lift_head_x_75_0.175_0.180_0.190.json"}
DIRECTIONS = (("left", 1.0), ("right", -1.0))
#: Half a unit in the sixth significant digit.
REL_TOL = 5e-7
ABS_TOL = 1e-12
#: Noise floor per unit, by the leaf's last key (module docstring).
NOISE_FLOORS = (
    (re.compile(r"(speed|speed_peak|yaw_rate\w*|omega)$"), 1e-6),  # rad/s
    (re.compile(r"power_mean$"), 1e-6),  # W
    (re.compile(r"(_nm|rms|rms_stacked|peak|mz_abs_max)$"), 1e-6),  # N m
    (re.compile(r"_mm$"), 1e-4),  # mm
    (re.compile(r"(_m|margin\w*|origin|error_max|lean_inward|sway_amp|lift|stride|radius|r_eff|r_cop_min)$"), 1e-7),
)


def walk_specs(body: str) -> dict:
    return {"nominal": sizing.NOMINAL, "brisk": sizing.BRISK, "fr05": envelope.FAST,
            "carpet": replace(sizing.NOMINAL, lift_frac=CARPET_LIFT[body])}


def turn_cases() -> dict:
    return {f"{case}_{label}": dict(case=case, yaw_rate=sign * c["yaw_rate"], in_place=c["in_place"], role=c["role"])
            for case, c in turning.CASES.items() for label, sign in DIRECTIONS}


def reference_names(body: str) -> list[str]:
    return list(walk_specs(body)) + list(turn_cases())


# ------------------------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------------------------
def source(body: str) -> dict:
    """The body rebuilt from its record: model, full-precision home pose, design and H (module docstring)."""
    from .. import round2
    from ..pipeline import build_candidate

    rec, cid = plant.record(body), candidate_id(body)
    with tempfile.TemporaryDirectory() as tmp:
        res = build_candidate(round2.candidate(cid, rec.get("overrides", ())), Path(tmp),
                              selection=dict(rec["selection"]), B=rec["B"], write=False)
        check = plant.verify(body, Path(tmp) / f"{cid}.xml")
    check["file"] = "rebuilt by pipeline.build_candidate"
    design = res["design"]
    return dict(body=body, M=res["model"], q_home=res["q_home"], design=design, H=float(design.sk.H),
                layout=design.cand.leg_layout, record=rec, model=check)


def generate(src: dict, name: str) -> dict:
    """One reference of ``src``'s body: trajectory, inverse dynamics and summary."""
    gait.set_accuracy(ACCURACY)
    M, q0, H = src["M"], src["q_home"], src["H"]
    t0 = time.perf_counter()
    walks = walk_specs(src["body"])
    if name in walks:
        spec = walks[name]
        tr = gait.generate(M, q0, H, spec)
        idr = gait.inverse_dynamics(M, tr)
        s = gait.summarize(M, tr, idr)
        how = dict(generator="gait.generate", dynamics="gait.inverse_dynamics", spec=asdict(spec))
    else:
        c = turn_cases()[name]
        spec = sizing.NOMINAL
        tr = turning.generate(M, q0, H, spec, yaw_rate=c["yaw_rate"], in_place=c["in_place"])
        idr = turning.inverse_dynamics(M, tr, turning.MU, split="exact", ds_friction=True)
        s = turning.summarize(M, tr, idr)
        s["groups"] = turning.run_groups(M.names, idr, src["layout"])
        how = dict(generator="turning.generate", dynamics="turning.inverse_dynamics (exact split, mu 0.8)",
                   spec=asdict(spec), **c)
    return dict(name=name, traj=tr, idr=idr, summary=s, how=how, seconds=time.perf_counter() - t0)


def stride_motion(traj: dict) -> list:
    """(phi, a_x, a_y, b_x, b_y): the next stride's pose is ``turning.move_pose(q, phi, a, b)``."""
    if "transform" in traj:
        phi, a, b = traj["transform"]
        return [float(phi), *map(float, a), *map(float, b)]
    return [0.0, 0.0, 0.0, float(traj["info"]["stride"]), 0.0]


def arrays(src: dict, g: dict) -> dict:
    """The arrays of one reference for its npz file."""
    M, tr, idr = src["M"], g["traj"], g["idr"]
    n = len(tr["q"])
    feet = np.zeros((n, 2, 3))
    com = np.zeros((n, 3))
    for k in range(n):
        M.forward(tr["q"][k])
        feet[k] = [M.foot_pose(s)[0] for s in statics.SIDES]
        com[k] = M.com()
    out = dict(t=tr["t"], q=tr["q"], qd=idr["qd"], qdd=idr["qdd"], tau=idr["tau"], omega=idr["omega"],
               support=np.array([[s in ph for s in statics.SIDES] for ph in tr["phases"]]),
               com_ref=tr["com_ref"], zmp_ref=tr["zmp_ref"], zmp_mb=tr["zmp_mb"], cop_margin=idr["cop_margin"],
               normal=idr["normal"], friction=idr["friction"], feet=feet, com=com, q_home=src["q_home"],
               joints=np.array(M.names), dt=np.array(float(tr["dt"])), stride_motion=np.array(stride_motion(tr)))
    if "schedule" in tr:
        sc = tr["schedule"]
        out.update(path=tr["path"], heading=tr["heading"],
                   foot_target=np.stack([sc["pos"][s] for s in statics.SIDES], axis=1),
                   foot_yaw=np.stack([sc["yaw"][s] for s in statics.SIDES], axis=1),
                   swing=np.stack([sc["swing"][s] for s in statics.SIDES], axis=1))
    return out


# ------------------------------------------------------------------------------------------------
# Record comparison
# ------------------------------------------------------------------------------------------------
def _plain(x):
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def leaves(x, pre: str = "") -> dict:
    """Flatten nested dicts and lists to ``path -> value``."""
    out = {}
    if isinstance(x, dict):
        for k, v in x.items():
            out.update(leaves(v, f"{pre}.{k}" if pre else str(k)))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            out.update(leaves(v, f"{pre}[{i}]"))
    else:
        out[pre] = x
    return out


def agree(a, b) -> bool:
    """Numbers agree to 6 significant digits (or both NaN); other values are equal."""
    if isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return a == b
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return abs(a - b) <= max(REL_TOL * max(abs(a), abs(b)), ABS_TOL)


def _number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def noise_floor(key: str) -> float | None:
    """The noise floor of a leaf from its last key (``NOISE_FLOORS``), or None (dimensionless and flag leaves)."""
    last = re.sub(r"\[\d+\]", "", key).rsplit(".", 1)[-1]
    return next((fl for pat, fl in NOISE_FLOORS if pat.search(last)), None)


def within_noise(key: str, a, b) -> bool:
    """Both values lie within the leaf's noise floor."""
    fl = noise_floor(key)
    return fl is not None and _number(a) and _number(b) and max(abs(a), abs(b)) <= fl


def compare(ours, record) -> dict:
    """Every leaf of ``record`` against ``ours``: how many, how many differ or are missing, how many match only
    within their noise floor (module docstring), the largest relative difference."""
    a, b = leaves(_plain(ours)), leaves(record)
    missing = [k for k in b if k not in a]
    bad, noise = [], []
    for k in b:
        if k in a and not agree(a[k], b[k]):
            (noise if within_noise(k, a[k], b[k]) else bad).append(k)
    worst, at = 0.0, None
    for k, v in b.items():
        x = a.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) and isinstance(x, (int, float)) and x != v:
            if not (math.isnan(v) or math.isnan(x)):
                r = abs(x - v) / max(abs(x), abs(v))
                if r > worst:
                    worst, at = r, k
    return dict(leaves=len(b), mismatched=bad[:20], n_mismatched=len(bad), missing=missing[:20], n_missing=len(missing),
                noise_floor_matches=noise[:20], n_noise_floor_matches=len(noise),
                n_noise_level_leaves=sum(within_noise(k, v, v) for k, v in b.items()),
                max_rel_diff=worst, max_rel_diff_at=at, match=not bad and not missing)


def toe_values(M: statics.Model, traj: dict, H: float) -> dict:
    """The carpet-lift pass's toe-clearance values of one walk (``refine.carpet_lift_one``)."""
    tc = carpet.toe_catch(M, traj, H, carpet.pile_classes())
    return dict(above_pile_top_mm={p: v["mid_swing_min_above_pile_top_mm"] for p, v in tc["piles"].items()},
                meets_carpet_target={p: v["meets_carpet_target"] for p, v in tc["piles"].items()},
                hard_mid_swing_mm=tc["hard_floor"]["mid_swing_min_mm"])


def record_checks(src: dict, g: dict) -> list:
    """Comparisons of one reference with every record that holds it."""
    body, name, rec = src["body"], g["name"], src["record"]
    cid = candidate_id(body)
    out = []
    if name in ("nominal", "brisk", "fr05"):
        out.append(dict(record=f"{BODIES[body]}/candidates/{cid}.json gaits.{name}",
                        **compare(g["summary"], rec["gaits"][name])))
    elif name == "carpet":
        lift = CARPET_LIFT[body]
        key = f"lift{round(100 * lift)}"
        if abs(100 * lift - round(100 * lift)) < 1e-9 and key in rec["gaits"]:
            out.append(dict(record=f"{BODIES[body]}/candidates/{cid}.json gaits.{key}",
                            **compare(g["summary"], rec["gaits"][key])))
        doc = json.loads((STUDY / CARPET_LIFT_RECORD[body]).read_text())
        row = next(r for r in doc if r["id"] == cid)
        toe = toe_values(src["M"], g["traj"], src["H"])
        out.append(dict(record=f"{CARPET_LIFT_RECORD[body]} {cid} lifts.{lift:g}.toe.nominal",
                        **compare(toe, row["lifts"][f"{lift:g}"]["toe"]["nominal"])))
    else:
        doc = json.loads((case_dir(body) / "review" / "screening" / f"{cid}_turning.json").read_text())
        out.append(dict(record=f"{BODIES[body]}/review/screening/{cid}_turning.json runs.{name}",
                        **compare(g["summary"], doc["runs"][name])))
    return out


# ------------------------------------------------------------------------------------------------
# Files
# ------------------------------------------------------------------------------------------------
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(src: dict, g: dict, out_dir: Path) -> dict:
    """Save one reference's arrays and return its metadata entry."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{g['name']}.npz"
    np.savez(path, **arrays(src, g))
    info = _plain(g["traj"]["info"])
    s = g["summary"]
    checks = record_checks(src, g)
    return dict(name=g["name"], file=f"refs/{path.name}", sha256=_sha(path), frames=len(g["traj"]["q"]),
                dt_s=float(g["traj"]["dt"]), period_s=float(info["period"]), speed_m_s=float(info["speed"]),
                stride_m=float(info["stride"]), lift_m=float(info["lift"]), cadence_steps_s=float(info["cadence_hz"]),
                yaw_rate_rad_s=float(info.get("yaw_rate", 0.0)), stride_motion=stride_motion(g["traj"]),
                min_cop_margin_single_m=float(s["min_cop_margin_single"]),
                max_friction_ratio=float(s["max_friction_ratio"]), ik_ok=bool(s["ik_ok"]), **g["how"],
                info=info, seconds=g["seconds"], record_checks=checks,
                matches_records=all(c["match"] for c in checks))


def prepare(body: str, out_dir: Path, names: list[str] | None = None) -> dict:
    """Generate, check and save a body's references; writes ``<out_dir>/refs.json`` and ``<out_dir>/refs/``."""
    t0 = time.perf_counter()
    gait.set_accuracy(ACCURACY)
    src = source(body)
    entries = {}
    for name in names or reference_names(body):
        g = generate(src, name)
        entries[name] = write(src, g, Path(out_dir) / "refs")
        print(f"  {body} {name}: {g['seconds']:.1f} s, records match: {entries[name]['matches_records']}", flush=True)
    doc = dict(body=body, accuracy=ACCURACY, model=src["model"], H_m=src["H"],
               q_home=[float(v) for v in src["q_home"]], joints=list(src["M"].names),
               feedforward="inverse-dynamics joint torque, contact-consistent; MJCF joint damping and friction not "
                           "included",
               match_rule=f"every record leaf agrees to 6 significant digits (rel {REL_TOL}); a leaf whose two values "
                          "both lie within its unit's noise floor (" +
                          ", ".join(f"{pat.pattern} {fl:g}" for pat, fl in NOISE_FLOORS) + ") matches and is listed",
               references=entries, matches_records=all(e["matches_records"] for e in entries.values()),
               seconds=time.perf_counter() - t0, provenance=plant.provenance())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "refs.json").write_text(json.dumps(doc, indent=1))
    return doc


# ------------------------------------------------------------------------------------------------
# Loading
# ------------------------------------------------------------------------------------------------
@dataclass
class Reference:
    """One saved reference stride, periodic in its joint coordinates."""

    name: str
    arrays: dict

    @classmethod
    def load(cls, path) -> "Reference":
        with np.load(path, allow_pickle=False) as z:
            return cls(Path(path).stem, {k: z[k] for k in z.files})

    def __getattr__(self, key):
        try:
            return self.__dict__["arrays"][key]
        except KeyError:
            raise AttributeError(key) from None

    @property
    def n(self) -> int:
        return len(self.arrays["t"])

    @property
    def step(self) -> float:
        return float(self.arrays["dt"])

    @property
    def period(self) -> float:
        return self.n * self.step

    def joint_series(self, qadr, dofs) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Joint angles, rates and feedforward torques per frame for the hinges at ``qadr`` / ``dofs``."""
        return self.arrays["q"][:, qadr], self.arrays["qd"][:, dofs], self.arrays["tau"]

    def qdd_series(self, dofs) -> np.ndarray:
        """Joint accelerations per frame for the hinges at ``dofs``."""
        return self.arrays["qdd"][:, dofs]

    def at(self, t: float, series: tuple) -> tuple:
        """Each array of ``series`` (e.g. ``joint_series``, optionally with ``qdd_series``) linearly interpolated at
        time t, wrapped periodically."""
        s = (t / self.step) % self.n
        k0 = int(s) % self.n
        k1 = (k0 + 1) % self.n
        a = s - int(s)
        return tuple((1 - a) * x[k0] + a * x[k1] for x in series)

    def interp(self, t: float, stacked: np.ndarray) -> np.ndarray:
        """One row of ``stacked`` (frames x columns, e.g. the arrays of ``joint_series`` side by side) linearly
        interpolated at time t, wrapped periodically: ``at`` in one operation."""
        s = (t / self.step) % self.n
        k0 = int(s) % self.n
        a = s - int(s)
        return (1 - a) * stacked[k0] + a * stacked[(k0 + 1) % self.n]
