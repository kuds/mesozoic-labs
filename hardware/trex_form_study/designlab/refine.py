"""Refinement experiments on the round-two leaders (overnight session after the round-two review package).

Each experiment changes one lever on a leading body, re-sizes it with the round-two rules (generator revision 2,
Pi 5 compute, round-two walk accuracy, the packaging search) and screens it exactly like the round-two candidates
(``round2.screen_one``), so its row compares like for like with ``round2/`` and is scored with the round-two
scorecard.  No controller is involved: these are design levers checked with the same prescribed-motion screens.

Levers (``CASES``):

* ``headroom_60``: the 60 cm leaders sized with 20 % headroom on every planning limit and the Froude 0.05 walk as
  a sizing case (``sizing.HEADROOM`` 0.80, ``sizing.EXTRA_WALKS``), as the size study does at 75 and 80 cm.
* ``toes_long``, ``pads_large``, ``foot_both``: digit length x1.15 (override ``toe_scale``), contact-pad footprint
  x1.25 in each direction (``pad_scale``), or both.  A longer and broader foot moves the pads' front edge forward
  and spreads them (one-foot stopping, push recovery, the yaw moment a pad can carry in a turn), at the cost of
  distal mass and of toe-to-toe room in turns.
* ``narrow_3mm``: the narrow stance searched until the target turns keep at least 3 mm of closest approach
  (``round2.TURN_SEARCH_MIN_GAP_MM``), instead of merely no contact: stance width traded for build tolerance.
* ``no_calf``: the shank left uncovered (override ``calf_fairing`` 0; owner question on the calf fairings).  On the
  round-two bodies knee flexion stops where the upper-thigh shell meets the calf fairing (0.81 rad at 60 cm,
  0.76-0.81 rad at 75 cm), and a higher carpet swing needs more knee flexion.

``carpet_lift`` (separate pass, no re-sizing): the carpet swing target needs a higher swing than the reference
walks' 0.10-0.11 H.  For each listed body and each lift, the nominal and brisk walks are regenerated with that
lift and checked against the body's selected drives (sizing rules: continuous = max(nominal RMS, hold),
peak = 1.25 x max(walk peaks, hold), speed = 1.15 x walk peak speed), against the round-two clearance gates on
those walks, and against the carpet screen's toe catch (pile-top clearance and the 0.06 H / 15 mm target).

Run from hardware/trex_form_study:
    python -m designlab.refine --out round2/refine --workers 3 [--cases ...] [--carpet-lift]
Outputs: <out>/<case>/candidates, <out>/<case>/review/..., <out>/carpet_lift.json, <out>/REFINE.md, refine.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

LEADERS_60 = ("60-M-T-10", "60-M-H-10", "60-N-T-10", "60-N-H-10")
#: material and power levers (owner question overnight).  ``cf`` uses the provisional 2 mm carbon floor (the
#: optimistic sensitivity); ``cfqi`` the sourced 3 mm quasi-isotropic plate (hardware.STRUCT_MATERIALS, MATERIALS.md)
MAT_LEVERS = (("cf", (("structure_material", "cf_plate"),), "carbon-fibre frame plates (2 mm floor, optimistic)"),
              ("cfqi", (("structure_material", "cf_qi"),), "carbon-fibre frame plates (3 mm quasi-isotropic, sourced)"),
              ("thin", (("cover_wall_scale", 0.75),), "covering walls x0.75"),
              ("tether", (("power_mode", "tether"),), "tether-only power"),
              ("light", (("structure_material", "cf_plate"), ("cover_wall_scale", 0.75)),
               "carbon plates (2 mm) + covering walls x0.75"),
              ("lightqi", (("structure_material", "cf_qi"), ("cover_wall_scale", 0.75)),
               "carbon plates (3 mm, sourced) + covering walls x0.75"),
              # the research's recommended coverings: MJF PA11 (ductile) with graded 1.4 / 1.2 / 1.0 mm walls, about
              # x0.75 of the 1.6-1.8 mm walls on average
              ("pa11", (("cover_material", "PA11_MJF"), ("cover_wall_scale", 0.75)), "PA11 coverings, walls x0.75"),
              ("cfqi4", (("structure_material", "cf_qi_eqrig"),),
               "carbon-fibre frame plates at equal rigidity (3.99 mm quasi-isotropic)"),
              ("thinal", (("structure_material", "al6061_t254"),), "6061 frame plates at 2.54 mm (brackets 3 mm)"),
              ("best", (("structure_material", "cf_qi"), ("cover_material", "PA11_MJF"), ("cover_wall_scale", 0.75)),
               "carbon plates (3 mm, sourced) + PA11 coverings, walls x0.75"))
CASES = {
    "headroom_60": dict(ids=LEADERS_60, overrides=(), headroom=0.80, fr05=True, min_gap=0.0,
                        label="20 % headroom + Froude 0.05 sizing"),
    "toes_long": dict(ids=("60-M-T-10", "60-M-H-10", "60-N-H-10"), overrides=(("toe_scale", 1.15),), headroom=1.0,
                      fr05=False, min_gap=0.0, label="toes x1.15"),
    "pads_large": dict(ids=("60-M-T-10", "60-M-H-10", "60-N-H-10"), overrides=(("pad_scale", 1.25),), headroom=1.0,
                       fr05=False, min_gap=0.0, label="pads x1.25"),
    "foot_both": dict(ids=("60-M-T-10", "60-M-H-10", "60-N-H-10", "75-M-T-10"),
                      overrides=(("toe_scale", 1.15), ("pad_scale", 1.25)), headroom=1.0, fr05=False, min_gap=0.0,
                      label="toes x1.15 + pads x1.25"),
    "narrow_3mm": dict(ids=("60-N-T-10", "60-N-H-10"), overrides=(), headroom=1.0, fr05=False, min_gap=3.0,
                       label="narrow stance, target turns >= 3 mm"),
    "no_calf": dict(ids=("60-M-T-10", "60-M-H-10", "60-N-H-10", "75-M-T-10"), overrides=(("calf_fairing", 0),),
                    headroom=1.0, fr05=False, min_gap=0.0, label="no calf fairing (exposed ankle drives)"),
    # carpet-ready 60 cm: the ankle drives lowered to 0.60 of the shank (knee range 0.81 -> about 1.0 rad with the
    # calf fairing kept, found overnight) and sized for the carpet swing (nominal walk, swing lifted to 0.19 H) and
    # the Froude 0.05 walk
    "carpet_60": dict(ids=("60-M-T-10", "60-M-H-10"), overrides=(("ankle_frac", 0.60),), headroom=1.0, fr05=True,
                      walks=("fr05", "lift19"), min_gap=0.0,
                      label="60 cm: ankle drives at 0.60, sized for the 0.19 H carpet swing and Froude 0.05"),
    # material and power experiments on the two refined favorites (owner question overnight): carbon-fibre frame
    # plates (motor brackets stay aluminium), covering walls 25 % thinner, tether-only power, and the lightest
    # combination; each on top of carpet_60 (60 cm) or targeted_75 (75 cm)
    **{f"mat60_{k}": dict(ids=("60-M-T-10",), overrides=(("ankle_frac", 0.60),) + ov, headroom=1.0, fr05=True,
                          walks=("fr05", "lift19"), min_gap=0.0, label=f"60 cm carpet-ready + {lab}")
       for k, ov, lab in MAT_LEVERS},
    **{f"mat75_{k}": dict(ids=("75-M-T-10",), overrides=ov, headroom=1.0, fr05=True, min_gap=0.0,
                          headroom_by_group={"neck": 0.85, "hip_roll": 0.85}, label=f"75 cm targeted + {lab}")
       for k, ov, lab in MAT_LEVERS},
    # targeted upgrade at 75 cm (cross-check recommendation): the what-if's Froude 0.05 sizing case plus 15 %
    # headroom on the two drives every judge found thin, the neck servo and the hip roll
    "targeted_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(), headroom=1.0, fr05=True, min_gap=0.0,
                        headroom_by_group={"neck": 0.85, "hip_roll": 0.85},
                        label="75 cm: Froude 0.05 sizing + 15 % headroom on neck and hip roll"),
    # carpet-ready 75 cm: targeted_75 plus the carpet swing as a sizing walk.  The lifted-walk pass on targeted_75
    # puts the 0.06 H target (30.1-30.4 mm above the pile) at a lift of about 0.176 H, so the nominal walk is
    # lifted to 0.18 H; the knee clears its range there (0.135 rad at 0.19 H on Tapered), so no ankle change.
    # Result: the sizing selects exactly the targeted_75 drives (same SHA-256), so carpet_75 is not screened
    # separately; the lifted walks are checked with ``--carpet-lift --lift-cases targeted_75 --lifts 0.175 0.18``
    "carpet_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(), headroom=1.0, fr05=True, screen=False,
                      walks=("fr05", "lift18"), min_gap=0.0, headroom_by_group={"neck": 0.85, "hip_roll": 0.85},
                      label="75 cm: targeted drives, sized for the 0.18 H carpet swing and Froude 0.05"),
    # knee headroom: on the carpet-ready bodies the governing limit is the knee's speed (0.95 at both sizes), a
    # kinematic requirement that a lighter body does not change (same-drive pass); 15 % headroom on the knee
    # lets the sizing pick a faster knee drive instead of the lightest one that passes
    # the narrow stance with 3 mm turn clearance tops the version-2 scorecard but keeps the round-two ankle station;
    # combined here with the carpet-ready ankle drives (0.60 of the shank) and the carpet swing sizing
    "narrow_carpet_60": dict(ids=("60-N-T-10", "60-N-H-10"), overrides=(("ankle_frac", 0.60),), headroom=1.0, fr05=True,
                             walks=("fr05", "lift19"), min_gap=3.0,
                             label="60 cm narrow stance (turns >= 3 mm) + carpet-ready ankle drives"),
    # the foot lever on the refined bodies: longer toes (x1.15) and larger pads (x1.25) cut the friction need with the
    # stance foot's yaw moment by about 8 % and raise the one-foot stop from 1.24 to 1.45 m/s^2 on the round-two
    # bodies (foot_both), so they are combined here with the carpet-ready 60 cm and targeted 75 cm bodies
    "feet_60": dict(ids=("60-M-T-10", "60-M-H-10"),
                    overrides=(("ankle_frac", 0.60), ("toe_scale", 1.15), ("pad_scale", 1.25)), headroom=1.0, fr05=True,
                    walks=("fr05", "lift19"), min_gap=0.0,
                    label="60 cm carpet-ready + toes x1.15 and pads x1.25"),
    "feet_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(("toe_scale", 1.15), ("pad_scale", 1.25)), headroom=1.0,
                    fr05=True, min_gap=0.0, headroom_by_group={"neck": 0.85, "hip_roll": 0.85},
                    label="75 cm targeted + toes x1.15 and pads x1.25"),
    "knee_60": dict(ids=("60-M-T-10", "60-M-H-10"), overrides=(("ankle_frac", 0.60),), headroom=1.0, fr05=True,
                    walks=("fr05", "lift19"), min_gap=0.0, headroom_by_group={"knee": 0.85},
                    label="60 cm carpet-ready + 15 % knee headroom"),
    "knee_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(), headroom=1.0, fr05=True, walks=("fr05", "lift18"), min_gap=0.0,
                    headroom_by_group={"neck": 0.85, "hip_roll": 0.85, "knee": 0.85},
                    label="75 cm carpet-ready + 15 % knee headroom"),
    # Stage 1b arm K (STAGE1B_PLAN.md section 2): the narrow carpet-ready picks (narrow_carpet_60) with knee_60's 15 %
    # knee headroom, which on the base-width bodies selects a RobStride 06 knee (ROUND2_REVIEW.md section 7.3)
    "narrow_knee_60": dict(ids=("60-N-T-10", "60-N-H-10"), overrides=(("ankle_frac", 0.60),), headroom=1.0, fr05=True,
                           walks=("fr05", "lift19"), min_gap=3.0, headroom_by_group={"knee": 0.85},
                           label="60 cm narrow carpet-ready + 15 % knee headroom"),
}
CARPET_LIFT_IDS = ("60-N-H-10", "60-N-T-10", "60-M-T-10", "60-M-H-10", "75-M-T-10-bighead")
LIFTS = (0.13, 0.16, 0.19, 0.22)


def extra_walks(names) -> dict:
    """Extra sizing walks by name: ``fr05`` (the behavior envelope's Froude 0.05 walk) and ``lift<NN>`` (the
    nominal walk with its swing lifted to 0.NN H, the carpet swing case)."""
    from . import envelope, sizing

    out = {}
    for n in names:
        if n == "fr05":
            out[n] = envelope.FAST
        elif n.startswith("lift"):
            out[n] = replace(sizing.NOMINAL, lift_frac=int(n[4:]) / 100.0)
        else:
            raise ValueError(n)
    return out


def _init(case: str | None) -> None:
    from . import gait, round2, sizing

    gait.set_accuracy(round2.GAIT_ACCURACY)
    c = CASES.get(case) if case else None
    sizing.HEADROOM = c["headroom"] if c else 1.0
    sizing.HEADROOM_BY_GROUP = dict(c.get("headroom_by_group") or {}) if c else {}
    names = (c.get("walks") or (("fr05",) if c["fr05"] else ())) if c else ()
    sizing.EXTRA_WALKS = extra_walks(names)
    round2.TURN_SEARCH_MIN_GAP_MM = c["min_gap"] if c else 0.0


def _size(a):
    from . import round2

    case, cid, out = a
    _init(case)
    r = round2.size_one(cid, out, str(Path(out) / "models"), base_overrides=CASES[case]["overrides"])
    p = Path(out, "candidates", f"{cid}.json")
    d = json.loads(p.read_text())
    c = CASES[case]
    d.update(refine_case=case, headroom=c["headroom"], headroom_by_group=dict(c.get("headroom_by_group") or {}),
             fr05_sizing=c["fr05"], turn_min_gap_mm=c["min_gap"])  # records sized before 10 Nov lack the groups
    p.write_text(json.dumps(d, indent=1))
    return dict(case=case, **r)


def _screen(a):
    from . import round2

    case, cid, out = a
    _init(case)
    return dict(case=case, **round2._screen_star((cid, Path(out))))


# ------------------------------------------------------------------------------------------------
# Carpet swing height
# ------------------------------------------------------------------------------------------------
def carpet_lift_one(a) -> dict:
    """One body, every lift in ``LIFTS``: drives, clearance and toe catch of the lifted walks.

    ``a = (id, case dir, scratch dir[, lifts[, nominal_only]])``; with ``nominal_only`` only the nominal walk is
    lifted and the brisk walk stays at its reference lift (the carpet gait walks at the nominal speed)."""
    import numpy as np

    from . import carpet, clearance, round2, sizing
    from .pipeline import build_candidate

    cid, r2dir, scratch = a[:3]
    lifts = a[3] if len(a) > 3 else LIFTS
    nominal_only = bool(a[4]) if len(a) > 4 else False
    _init(None)
    t0 = time.time()
    rec = json.loads(Path(r2dir, "candidates", f"{cid}.json").read_text())
    res = build_candidate(round2.candidate(cid, rec.get("overrides", ())), Path(scratch), selection=dict(rec["selection"]),
                          B=rec["B"], write=False)
    M, q0, design = res["model"], res["q_home"], res["design"]
    H, layout, bus = design.sk.H, design.cand.leg_layout, rec["selection"]["bus"]
    xml = str(Path(scratch) / f"{cid}.xml")
    piles = carpet.pile_classes()
    unders = carpet.foot_underside(M, q0, "whole_foot")
    nominal0, brisk0 = sizing.NOMINAL, sizing.BRISK
    out = dict(id=cid, H_m=H, carpet_target_mm=1e3 * carpet.carpet_target(H), reference_lift=[nominal0.lift_frac, brisk0.lift_frac],
               lifted="nominal walk only" if nominal_only else "nominal and brisk walks", lifts={})
    try:
        for lift in lifts:
            sizing.NOMINAL = replace(nominal0, lift_frac=lift)
            if not nominal_only:
                sizing.BRISK = replace(brisk0, lift_frac=lift)
            req = sizing.requirements(M, q0, H, layout)
            ratios = {}
            for g, r in req["groups"].items():
                act = design.act(g)
                spd = act.speed_at(bus if act.box is None else None)
                ratios[g] = max(r["cont"] / act.sustained, r["peak_req"] / act.peak, r["speed_req"] / max(spd, 1e-6))
            worst = max(ratios.items(), key=lambda t: t[1])
            trajs = {k: v["traj"] for k, v in req["runs"].items()}
            cl = clearance.screen(xml, q0, trajs, H=H, gates="r2")["summary"]
            mg = cl.get("joint_range_margins") or {}
            tight = min(((j, v) for j, r_ in mg.items() for v in (r_.get("margin_min"), r_.get("margin_max")) if v is not None),
                        key=lambda t: t[1], default=(None, None))
            toe = {}
            for name, tr in trajs.items():
                idr = req["runs"][name]["idr"]
                F = float(np.max(idr["normal"]))
                sink = {p: carpet.pile_sinkage(unders["left"], pc["stiffness_pa_per_m"], F) for p, pc in piles.items()}
                tc = carpet.toe_catch(M, tr, H, piles, sink)
                toe[name] = dict(above_pile_top_mm={p: v["mid_swing_min_above_pile_top_mm"] for p, v in tc["piles"].items()},
                                 meets_carpet_target={p: v["meets_carpet_target"] for p, v in tc["piles"].items()},
                                 hard_mid_swing_mm=tc["hard_floor"]["mid_swing_min_mm"])
            out["lifts"][f"{lift:g}"] = dict(
                drives_within_limits=all(v <= 1.0 + 1e-9 for v in ratios.values()), worst_group=worst[0],
                worst_ratio=worst[1], ratios=ratios, ik_ok=all(req["gaits"][k]["ik_ok"] for k in req["gaits"]),
                clearance_pass=cl["pass"], walking_gates=all(cl["pass"].get(k) is True for k in round2.WALK_GATES),
                tightest_joint=tight, toe=toe,
                meets_target_all_piles=all(all(t["meets_carpet_target"].values()) for t in toe.values()),
                min_above_pile_top_mm=min(v for t in toe.values() for v in t["above_pile_top_mm"].values()),
                walk_power_hint_w=None)
    finally:
        sizing.NOMINAL, sizing.BRISK = nominal0, brisk0
    out["seconds"] = time.time() - t0
    return out


# ------------------------------------------------------------------------------------------------
# Same drives: what a lighter body buys when the drives are kept
# ------------------------------------------------------------------------------------------------
#: reference case whose drives and pack a material/power case keeps in the same-drive pass
FIXED_REF = {"mat60": "carpet_60", "mat75": "targeted_75"}


def fixed_one(a) -> dict:
    """One material or power case rebuilt with its reference case's drives and pack (no re-selection).

    Re-sizing a lighter body selects lighter drives, so its margins stay near the limits and the saving shows up
    as drive mass and cost.  Keeping the reference drives shows the other use of the saving: margin.  Same
    sizing walks as the case (``_init``), same ratio definitions as the sizing record (continuous / sustained,
    peak / peak, speed / speed at the bus).  Clearance is not re-run: the shapes are the reference body's, apart
    from the covering wall thickness."""
    from . import cost, power, round2, sizing
    from .pipeline import build_candidate

    case, cid, out, scratch = a
    _init(case)
    t0 = time.time()
    ref_case = FIXED_REF[case.split("_")[0]]
    ref = json.loads(Path(out, ref_case, "candidates", f"{cid}.json").read_text())
    rec_p = Path(out, case, "candidates", f"{cid}.json")
    if rec_p.exists():  # re-sized: the case's own overrides, packaging-search changes included
        ov = tuple(tuple(o) for o in json.loads(rec_p.read_text()).get("overrides", ()))
    else:  # not re-sized: the reference body's overrides with the case's levers on top
        merged = dict(tuple(o) for o in ref.get("overrides", ()))
        merged.update(dict(CASES[case]["overrides"]))
        ov = tuple(merged.items())
    sel = dict(ref["selection"])
    cand = round2.candidate(cid, ov)
    res = build_candidate(cand, Path(scratch), selection=sel, write=False)
    M, q0, design = res["model"], res["q_home"], res["design"]
    req = sizing.requirements(M, q0, design.sk.H, cand.leg_layout)
    ratios = {}
    for g, r in req["groups"].items():
        act = design.act(g)
        spd = act.speed_at(sel["bus"] if act.box is None else None)
        ratios[g] = dict(actuator=act.id, cont=r["cont"] / act.sustained, peak=r["peak_req"] / act.peak,
                         speed=r["speed_req"] / max(spd, 1e-6), hold=r["hold"])
    worst = max(((g, k, v[k]) for g, v in ratios.items() for k in ("cont", "peak", "speed")), key=lambda t: t[2])
    leg_worst = max(((g, k, v[k]) for g, v in ratios.items() if g not in ("neck", "tail")
                     for k in ("cont", "peak", "speed")), key=lambda t: t[2])
    nom = req["runs"]["nominal"]["idr"]
    ms = sizing._motor_space(M.names, nom["tau"], nom["omega"], cand.leg_layout)
    pw = power.walking_power(design, req, ms)
    tether = getattr(design, "power_mode", "battery") == "tether"
    rt = None if tether else power.runtime(power.catalog.part(sel["battery"]), pw["total_w"])
    ref_m = ref["margins"]
    ref_worst = max(((g, k, m[f"{k}_ratio"]) for g, m in ref_m.items() for k in ("cont", "peak", "speed")),
                    key=lambda t: t[2])
    ref_leg = max(((g, k, m[f"{k}_ratio"]) for g, m in ref_m.items() if g not in ("neck", "tail")
                   for k in ("cont", "peak", "speed")), key=lambda t: t[2])
    return dict(case=case, id=cid, reference_case=ref_case, label=CASES[case]["label"], mass_kg=M.mass(),
                reference_mass_kg=ref["metrics"]["mass"], com_ahead_of_hip_m=_com_ahead(M, q0, design),
                reference_com_ahead_of_hip_m=ref["metrics"].get("com_ahead_of_hip"),
                worst=worst, leg_worst=leg_worst, reference_worst=ref_worst, reference_leg_worst=ref_leg,
                ratios=ratios, holds={g: r["hold"] for g, r in req["groups"].items()},
                reference_holds={g: m["hold"] for g, m in ref_m.items()},
                walk_w=pw["total_w"], reference_walk_w=ref["power"]["total_w"],
                runtime_min=None if rt is None else rt["minutes"], reference_runtime_min=ref["runtime"]["minutes"],
                tether=tether, walk_feasible=req["walk_feasible"],
                cost=cost.estimate(design, res["rows"], sel), reference_cost=ref["cost"],
                ik_ok=all(req["gaits"][k]["ik_ok"] for k in req["gaits"]), seconds=time.time() - t0)


def _com_ahead(M, q0, design) -> float:
    from . import metrics

    return float(metrics.measure(M, design, q0)["com_ahead_of_hip"])


def fixed_markdown(rows: list) -> str:
    lines = ["", "## Same drives: the material and power cases with the reference body's drives and pack", "",
             "Each case rebuilt with the drives and pack of its reference (60 cm: `carpet_60`; 75 cm: `targeted_75`) "
             "instead of being re-sized.  Ratios use the same sizing walks; clearance is not re-run.", "",
             "| Body | Case | Mass (reference) | CoM ahead of hip | Governing leg drive (reference) | Governing drive, any "
             "(reference) | Hip-pitch hold | Walking power | Runtime | Cost nominal (reference) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lw, rlw, w, rw = r["leg_worst"], r["reference_leg_worst"], r["worst"], r["reference_worst"]
        hp, rhp = r["holds"].get("hip_pitch"), r["reference_holds"].get("hip_pitch")
        rt = "tethered" if r["tether"] else f"{r['runtime_min']:.1f} min ({r['reference_runtime_min']:.1f})"
        lines.append(
            f"| `{r['id']}` | {r['label']} | {r['mass_kg']:.2f} kg ({r['reference_mass_kg']:.2f}) | "
            f"{1e3 * r['com_ahead_of_hip_m']:.0f} mm ({1e3 * r['reference_com_ahead_of_hip_m']:.0f}) | "
            f"{lw[0]} {lw[1]} {lw[2]:.2f} ({rlw[0]} {rlw[1]} {rlw[2]:.2f}) | {w[0]} {w[1]} {w[2]:.2f} "
            f"({rw[0]} {rw[1]} {rw[2]:.2f}) | {hp:.2f} N·m ({rhp:.2f}) | {r['walk_w']:.1f} W ({r['reference_walk_w']:.1f}) | "
            f"{rt} | ${r['cost']['nominal'] / 1e3:.1f}k (${r['reference_cost']['nominal'] / 1e3:.1f}k) |")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------
# Plates: round two against the refined bodies
# ------------------------------------------------------------------------------------------------
PLATE_PAIRS = (("60-M-T-10", "carpet_60"), ("60-M-H-10", "carpet_60"), ("75-M-T-10", "targeted_75"))
#: The recommended verification set (scorecard version 2 picks, ROUND2_REVIEW.md section 6).
PICK_PAIRS = (("60-N-T-10", "narrow_carpet_60"), ("75-M-H-10", "feet_75"), ("60-N-H-10", "narrow_carpet_60"))


def plates_refined(out: Path, r2: Path = Path("round2"), pairs=PLATE_PAIRS, stem: str = "refined",
                   cut_title: str = "Installed hardware: 60 cm ankle drives lowered to 0.60 of the shank; 75 cm drive upgrade",
                   side_title: str = "Round two against the refined bodies") -> list:
    """Side exterior and side cutaway grids: each row one body, round two on the left, the refined body on the
    right, at one physical scale (``plates.matrix``)."""
    from . import plates

    grid = [[str(r2 / "models" / f"{cid}.xml"), str(out / case / "models" / f"{cid}.xml")] for cid, case in pairs]
    labels = [f"{cid} ({case})" for cid, case in pairs]
    P = out / "plates"
    P.mkdir(parents=True, exist_ok=True)
    made = [plates.matrix(grid, labels, ["Round two", "Refined overnight"], str(P / f"{stem}_side.png"), view="side",
                          mode="exterior", title=side_title),
            plates.matrix(grid, labels, ["Round two", "Refined overnight"], str(P / f"{stem}_cutaway.png"),
                          view="side", mode="cutaway", title=cut_title)]
    return made


# ------------------------------------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------------------------------------
def compare(out: Path, r2: Path = Path("round2")) -> dict:
    from .shortlist import detail
    from .sizestudy import row

    def load(base: Path, cid: str):
        p = base / "candidates" / f"{cid}.json"
        if not p.exists():
            return None, {}
        idx = base / "review" / "review_index.json"
        rv = json.loads(idx.read_text()).get(cid, {}) if idx.exists() else {}
        return json.loads(p.read_text()), dict(rv, _detail=detail(base, cid))

    rows = []
    ids = sorted({cid for c in CASES.values() for cid in c["ids"]})
    for cid in ids:
        d, rv = load(r2, cid)
        if d:
            rows.append(dict(case="round2", **row("round two", d, rv, None, rv.get("_detail"))))
    for case, c in CASES.items():
        if c.get("screen") is False:  # sized only (identical to another case's body)
            continue
        for cid in c["ids"]:
            d, rv = load(out / case, cid)
            ref, _ = load(r2, cid)
            if d:
                rows.append(dict(case=case, **row(c["label"], d, rv, ref, rv.get("_detail"))))
    lift = json.loads((out / "carpet_lift.json").read_text()) if (out / "carpet_lift.json").exists() else None
    fixed = json.loads((out / "fixed_drives.json").read_text()) if (out / "fixed_drives.json").exists() else None
    return dict(cases={k: v["label"] for k, v in CASES.items()}, rows=rows, carpet_lift=lift, fixed_drives=fixed)


def markdown(cmp: dict) -> str:
    from .sizestudy import markdown as md_rows

    text = md_rows(dict(rows=cmp["rows"])).replace("# Size study (round two)", "# Refinement experiments (round two)")
    text = text.replace("Generated by `python -m designlab.sizestudy`.", "Generated by `python -m designlab.refine`.")
    lift = cmp.get("carpet_lift")
    if lift:
        lines = ["", "## Carpet swing height (lifted reference walks, the selected drives)", "",
                 "| Body | Carpet target | Lift (H) | Drives within limits (governing) | Walking clearance gates | "
                 "Tightest joint margin | Lowest clearance above a pile top | Meets the target on every pile |",
                 "|---|---|---|---|---|---|---|---|"]
        for r in lift:
            for L, v in r["lifts"].items():
                tj = v["tightest_joint"]
                lines.append(f"| `{r['id']}` | {r['carpet_target_mm']:.1f} mm | {L} | "
                             f"{'yes' if v['drives_within_limits'] else '**no**'} ({v['worst_group']} {v['worst_ratio']:.2f}) | "
                             f"{'pass' if v['walking_gates'] else '**fail**'} | "
                             f"{'–' if tj[0] is None else f'{tj[0]} {tj[1]:.2f} rad'} | {v['min_above_pile_top_mm']:.1f} mm | "
                             f"{'yes' if v['meets_target_all_piles'] else 'no'} |")
        text += "\n".join(lines) + "\n"
    if cmp.get("fixed_drives"):
        text += fixed_markdown(cmp["fixed_drives"])
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="round2/refine")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--carpet-lift", action="store_true")
    ap.add_argument("--lift-cases", nargs="*", default=None, help="run the carpet swing pass on these cases' bodies")
    ap.add_argument("--lifts", nargs="*", type=float, default=None, help="swing lifts (fraction of H) for that pass")
    ap.add_argument("--nominal-only", action="store_true", help="carpet swing pass: lift the nominal walk only")
    ap.add_argument("--fixed", action="store_true", help="same-drive pass over the material and power cases")
    ap.add_argument("--fixed-cases", nargs="*", default=None, help="limit the same-drive pass to these cases (merged)")
    ap.add_argument("--plates", action="store_true", help="draw round two against the refined bodies")
    ap.add_argument("--plates-picks", action="store_true", help="draw round two against the recommended bodies")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip bodies already sized or screened")
    a = ap.parse_args()
    import designlab  # noqa: F401

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    if not a.report_only:
        cases = a.cases if a.cases is not None else list(CASES)
        tasks = [(c, cid, str(out / c)) for c in cases for cid in CASES[c]["ids"]]
        with ctx.Pool(a.workers) as pool:
            if tasks:
                # --resume: keep sized records and finished screens from an interrupted run
                size_tasks = [t for t in tasks if not (a.resume and (Path(t[2]) / "candidates" / f"{t[1]}.json").exists())]
                for r in pool.imap_unordered(_size, size_tasks):
                    print("sized", r, flush=True)
                screen_tasks = [t for t in tasks if not (a.resume and (Path(t[2]) / "review" / "screening"
                                                                       / f"{t[1]}_envelope.json").exists())]
                results = list(pool.imap_unordered(_screen, screen_tasks))
                if a.resume:  # keep the earlier screens' index entries
                    for c in cases:
                        idx = out / c / "review" / "review_index.json"
                        if idx.exists():
                            done = json.loads(idx.read_text())
                            results += [dict(case=c, **v) for k, v in done.items()
                                        if not any(r["case"] == c and r["id"] == k for r in results)]
                for c in cases:
                    idx = out / c / "review" / "review_index.json"
                    index = json.loads(idx.read_text()) if idx.exists() else {}
                    index.update({r["id"]: {k: v for k, v in r.items() if k != "case"} for r in results if r["case"] == c})
                    idx.parent.mkdir(parents=True, exist_ok=True)
                    idx.write_text(json.dumps(index, indent=1, default=str))
                for r in results:
                    print("screened", r["case"], r["id"], "sha ok" if r.get("matches_sized") else "SHA DIFFERS",
                          {k: v for k, v in r.items() if k.endswith("_error") or k == "error"}, flush=True)
            if a.carpet_lift:
                scratch = out / "carpet_lift_models"
                todo = [(cid, "round2", str(scratch)) for cid in CARPET_LIFT_IDS] if not a.lift_cases else [
                    (cid, str(out / c), str(scratch / c)) for c in a.lift_cases for cid in CASES[c]["ids"]]
                lifts = tuple(a.lifts) if a.lifts else LIFTS
                lift = list(pool.imap(carpet_lift_one, [t + (lifts, a.nominal_only) for t in todo]))
                for r, t in zip(lift, todo, strict=True):
                    r["source"] = t[1]
                name = "carpet_lift.json" if not a.lift_cases else f"carpet_lift_{'_'.join(a.lift_cases)}.json"
                if a.lifts:
                    name = name.replace(".json", "_" + "_".join(f"{v:.3f}" for v in lifts) + ".json")
                if a.nominal_only:
                    name = name.replace(".json", "_nominal_only.json")
                (out / name).write_text(json.dumps(lift, indent=1, default=str))
                print("carpet lift done", flush=True)
            if a.fixed:
                names = a.fixed_cases or [c for c in CASES if c.split("_")[0] in FIXED_REF]
                todo = [(c, cid, str(out), str(out / "fixed_models" / c)) for c in names
                        for cid in CASES[c]["ids"]
                        if (out / FIXED_REF[c.split("_")[0]] / "candidates" / f"{cid}.json").exists()]
                fixed = list(pool.imap(fixed_one, todo))
                fp = out / "fixed_drives.json"
                if a.fixed_cases and fp.exists():  # merge: keep the other cases' rows, in CASES order
                    new_keys = {(r["case"], r["id"]) for r in fixed}
                    keep = [r for r in json.loads(fp.read_text()) if (r["case"], r["id"]) not in new_keys]
                    order = {c: k for k, c in enumerate(CASES)}
                    fixed = sorted(keep + fixed, key=lambda r: (order.get(r["case"], 999), r["id"]))
                fp.write_text(json.dumps(fixed, indent=1, default=str))
                print("same-drive pass done", flush=True)
    if a.plates:
        for m in plates_refined(out):
            print("plate", m, flush=True)
    if a.plates_picks:
        for m in plates_refined(out, pairs=PICK_PAIRS, stem="picks", side_title="Round two against the recommended bodies",
                                cut_title="Installed hardware of the recommended bodies"):
            print("plate", m, flush=True)
    cmp = compare(out)
    (out / "refine.json").write_text(json.dumps(cmp, indent=1, default=str))
    (out / "REFINE.md").write_text(markdown(cmp))
    print(markdown(cmp))


if __name__ == "__main__":
    main()
