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
  round-two bodies knee flexion stops where the upper-thigh shell meets the calf fairing (0.80 rad at 60 cm,
  0.76 rad at 75 cm), and a higher carpet swing needs more knee flexion.

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
    # targeted upgrade at 75 cm (cross-check recommendation): the what-if's Froude 0.05 sizing case plus 15 %
    # headroom on the two drives every judge found thin, the neck servo and the hip roll
    "targeted_75": dict(ids=("75-M-T-10", "75-M-H-10"), overrides=(), headroom=1.0, fr05=True, min_gap=0.0,
                        headroom_by_group={"neck": 0.85, "hip_roll": 0.85},
                        label="75 cm: Froude 0.05 sizing + 15 % headroom on neck and hip roll"),
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
    d.update(refine_case=case, headroom=c["headroom"], fr05_sizing=c["fr05"], turn_min_gap_mm=c["min_gap"])
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
    """One body, every lift in ``LIFTS``: drives, clearance and toe catch of the lifted walks."""
    import numpy as np

    from . import carpet, clearance, round2, sizing
    from .pipeline import build_candidate

    cid, r2dir, scratch = a
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
               lifts={})
    try:
        for lift in LIFTS:
            sizing.NOMINAL = replace(nominal0, lift_frac=lift)
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
            out["lifts"][f"{lift:.2f}"] = dict(
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
# Comparison
# ------------------------------------------------------------------------------------------------
def compare(out: Path, r2: Path = Path("round2")) -> dict:
    from .sizestudy import row

    from .shortlist import detail

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
        for cid in c["ids"]:
            d, rv = load(out / case, cid)
            ref, _ = load(r2, cid)
            if d:
                rows.append(dict(case=case, **row(c["label"], d, rv, ref, rv.get("_detail"))))
    lift = json.loads((out / "carpet_lift.json").read_text()) if (out / "carpet_lift.json").exists() else None
    return dict(cases={k: v["label"] for k, v in CASES.items()}, rows=rows, carpet_lift=lift)


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
    return text


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="round2/refine")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--carpet-lift", action="store_true")
    ap.add_argument("--lift-cases", nargs="*", default=None, help="run the carpet swing pass on these cases' bodies")
    ap.add_argument("--report-only", action="store_true")
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
                for r in pool.imap_unordered(_size, tasks):
                    print("sized", r, flush=True)
                results = list(pool.imap_unordered(_screen, tasks))
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
                lift = list(pool.imap(carpet_lift_one, todo))
                for r, t in zip(lift, todo, strict=True):
                    r["source"] = t[1]
                name = "carpet_lift.json" if not a.lift_cases else f"carpet_lift_{'_'.join(a.lift_cases)}.json"
                (out / name).write_text(json.dumps(lift, indent=1, default=str))
                print("carpet lift done", flush=True)
    cmp = compare(out)
    (out / "refine.json").write_text(json.dumps(cmp, indent=1, default=str))
    (out / "REFINE.md").write_text(markdown(cmp))
    print(markdown(cmp))


if __name__ == "__main__":
    main()
