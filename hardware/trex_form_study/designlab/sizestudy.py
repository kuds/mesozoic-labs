"""Size study (owner question during round two): does a 60 cm body 5-10 % smaller gain drive margin, and can the
75 cm body take larger drives without growing to 80 cm?

Cases, all on round-two rules (generator revision 2, Pi 5 compute, round-two walk accuracy, the packaging
search), Tapered and Hybrid base bodies (balanced width, 10 deg chest-up):

* ``small_resized``: 54 and 57 cm crown, sized from scratch like round two (the lightest-cheapest parts that meet
  every requirement).
* ``small_60drives``: the same 54 and 57 cm bodies with the 60 cm base's drives, servos, bus and battery kept (no
  re-selection; the structure is re-sized for the new mass): the headroom that shrinking buys with unchanged
  hardware.
* ``large_headroom``: the 75 cm bases sized with 20 % headroom on every planning limit (``sizing.HEADROOM`` 0.80,
  the headroom the 60 cm bases have without it) and the Froude 0.05 walk as a sizing case.
* ``80_resized``: 80 cm bases sized like round two.
* ``80_headroom``: 80 cm bases with the 75 cm headroom rule.

Study heights take every per-class rule from their class (``params.SIZE_CLASS``: 54 and 57 cm from 60 cm, 80 cm
from 75 cm), so only the size changes.  Every body is screened exactly like the round-two candidates
(``round2.screen_one``) and scored with the round-two scorecard (``shortlist``), so rows compare like for like
with ``round2/``.

Run from hardware/trex_form_study:
    python -m designlab.sizestudy --out round2/size_study --workers 3
Writes <out>/<case>/candidates/<id>.json, <out>/<case>/review/{mjcf,screening}, <out>/<case>/review/review_index.json,
and <out>/SIZE_STUDY.md + size_study.json (the comparison, with the round-two 60 and 75 cm bases and the 75 cm
Froude 0.05 what-if).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path

SMALL = ("54-M-T-10", "54-M-H-10", "57-M-T-10", "57-M-H-10")
CASES = {
    "small_resized": dict(ids=SMALL, mode="resize", headroom=1.0, fr05=False,
                          label="5-10 % smaller, drives re-selected"),
    "small_60drives": dict(ids=SMALL, mode="fixed", headroom=1.0, fr05=False,
                           label="5-10 % smaller, 60 cm drives kept"),
    "large_headroom": dict(ids=("75-M-T-10", "75-M-H-10"), mode="resize", headroom=0.80, fr05=True,
                           label="75 cm, 20 % headroom + Froude 0.05 sizing"),
    "80_resized": dict(ids=("80-M-T-10", "80-M-H-10"), mode="resize", headroom=1.0, fr05=False,
                       label="80 cm, round-two sizing"),
    "80_headroom": dict(ids=("80-M-T-10", "80-M-H-10"), mode="resize", headroom=0.80, fr05=True,
                        label="80 cm, 20 % headroom + Froude 0.05 sizing"),
}
SEARCH_MAX_STEPS = 10


def _init(case: str) -> None:
    from . import envelope, gait, round2, sizing

    c = CASES[case]
    gait.set_accuracy(round2.GAIT_ACCURACY)
    sizing.HEADROOM = c["headroom"]
    sizing.EXTRA_WALKS = {"fr05": envelope.FAST} if c["fr05"] else {}


def source_of(cid: str) -> str:
    """The round-two 60 cm base whose drives a small body keeps."""
    return f"60-M-{cid.split('-')[2]}-10"


def size_fixed(cid: str, out: str, xml_dir: str, r2dir: str = "round2") -> dict:
    """Build ``cid`` with the round-two 60 cm base's selection (drives, servos, bus, battery) and evaluate it as
    ``pipeline.size_candidate``'s final evaluation does, without re-selecting parts and without the final-load
    repair.  The structure mass guess is iterated to the built mass.  Walking-gate failures are searched like
    round two's packaging search (legs out, ankle drives down)."""
    from . import cost, metrics, mjcf, pipeline, power, round2, sizing
    from .pipeline import build_candidate
    from .stage_a import _mass_by_category, _mass_by_volume_class

    t0 = time.time()
    src_id = source_of(cid)
    src = json.loads(Path(r2dir, "candidates", f"{src_id}.json").read_text())
    sel = dict(src["selection"])
    search, extra = [], {}
    try:
        for _ in range(SEARCH_MAX_STEPS):
            cand = round2.candidate(cid, tuple(extra.items()))
            B = pipeline.solve_B(cand)
            for _ in range(6):  # structure sized for the built mass
                res = build_candidate(cand, Path(xml_dir), selection=sel, B=B, write=False)
                m = res["summary"]["mass"]
                if abs(m - sel.get("mass_guess", 0.0)) < 0.002 * m:
                    break
                sel["mass_guess"] = m
            ok, gates, cause = round2._walk_gates(cand, dict(sel), B, Path(xml_dir))
            search.append(dict(overrides=dict(extra), walk_gates_pass=ok, gates=gates, cause=cause))
            if ok or cause is None:
                break
            if cause == "hip_spacing" and extra.get("hip_spacing_add_m", 0.0) < round2.NARROW_MAX_ADD_M - 1e-9:
                extra["hip_spacing_add_m"] = round(extra.get("hip_spacing_add_m", 0.0) + round2.NARROW_STEP_M, 4)
            elif cause == "knee":
                fa = extra.get("ankle_frac", res["design"]._ankle_frac())
                if fa >= round2.ANKLE_MAX - 1e-9:
                    break
                extra["ankle_frac"] = round(min(round2.ANKLE_MAX, fa + round2.ANKLE_STEP), 3)
            else:
                break
        res = build_candidate(cand, Path(xml_dir), selection=sel, B=B, write=False)
        M, q_home, design = res["model"], res["q_home"], res["design"]
        req = sizing.requirements(M, q_home, design.sk.H, cand.leg_layout)
        nom = req["runs"]["nominal"]["idr"]
        ms = sizing._motor_space(M.names, nom["tau"], nom["omega"], cand.leg_layout)
        pw = power.walking_power(design, req, ms)
        rt = power.runtime(power.catalog.part(sel["battery"]), pw["total_w"])
        margins = {}
        for g, r in req["groups"].items():
            a = design.act(g)
            spd = a.speed_at(sel["bus"] if a.box is None else None)
            margins[g] = dict(actuator=a.id, sustained=a.sustained, peak=a.peak, speed=spd,
                              cont_req=r["cont"], peak_req=r["peak_req"], speed_req=r["speed_req"],
                              cont_ratio=r["cont"] / a.sustained, peak_ratio=r["peak_req"] / a.peak,
                              speed_ratio=r["speed_req"] / max(spd, 1e-6), rms=r["rms"], hold=r["hold"],
                              double_support=r["double_support"])
        cst = cost.estimate(design, res["rows"], sel)
        evaluation = dict(requirements={g: r for g, r in req["groups"].items()}, holds=req["holds"], gaits=req["gaits"],
                          walk_feasible=req["walk_feasible"], margins=margins, power=pw, runtime=rt, sizing_history=[],
                          cost=cst)
        mjcf.write_ledger(design, res["rows"], Path(xml_dir) / f"{cid}.json",
                          extra=dict(summary=res["summary"], evaluation=evaluation))
        meas = metrics.measure(M, design, q_home)
        doc = dict(id=cid, rev=round2.REV, mode="fixed", drives_from=src_id,
                   candidate=dict(height=cand.height, width=cand.width, family=cand.family, posture=cand.posture,
                                  proportion=cand.proportion, leg_layout=cand.leg_layout, battery_pos=cand.battery_pos),
                   B=B, H=design.sk.H, xml_sha256=res["summary"]["xml_sha256"], selection=dict(sel), metrics=meas,
                   margins=margins, holds=req["holds"], power=pw, runtime=rt, cost=cst, gaits=req["gaits"],
                   sizing_history=[], walk_feasible=req["walk_feasible"], geometry_checks=design.geometry_checks,
                   notes=design.notes, overrides=[list(o) for o in cand.overrides if o[0] != "rev"],
                   packaging_search=search, mass_by_category=_mass_by_category(res["rows"]),
                   mass_by_volume_class=_mass_by_volume_class(res["rows"]), gait_accuracy=round2.GAIT_ACCURACY,
                   seconds=time.time() - t0, ok=True)
    except Exception as e:  # keep failed cases
        doc = dict(id=cid, rev=round2.REV, mode="fixed", ok=False, error=repr(e), traceback=traceback.format_exc(),
                   seconds=time.time() - t0)
    Path(out, "candidates").mkdir(parents=True, exist_ok=True)
    Path(out, "candidates", f"{cid}.json").write_text(json.dumps(doc, indent=1, default=round2._json))
    return dict(id=cid, ok=doc["ok"], seconds=doc["seconds"])


def _size(a):
    from . import round2

    case, cid, out = a
    _init(case)
    xml_dir = str(Path(out) / "models")
    if CASES[case]["mode"] == "fixed":
        r = size_fixed(cid, out, xml_dir)
    else:
        r = round2.size_one(cid, out, xml_dir)
        p = Path(out, "candidates", f"{cid}.json")
        d = json.loads(p.read_text())
        d.update(mode="resize", headroom=CASES[case]["headroom"], fr05_sizing=CASES[case]["fr05"])
        p.write_text(json.dumps(d, indent=1))
    return dict(case=case, **r)


def _screen(a):
    from . import round2

    case, cid, out = a
    _init(case)
    r = round2._screen_star((cid, Path(out)))
    return dict(case=case, **r)


# ------------------------------------------------------------------------------------------------
# Comparison
# ------------------------------------------------------------------------------------------------
def _worst(doc):
    best = ("", 0.0, "")
    for g, m in doc["margins"].items():
        for kind in ("cont", "peak", "speed"):
            if m[f"{kind}_ratio"] > best[1]:
                best = (g, m[f"{kind}_ratio"], kind)
    return best


def row(label: str, doc: dict, rv: dict, ref: dict | None, det: dict | None = None) -> dict:
    from . import shortlist

    if not doc.get("ok"):
        return dict(label=label, id=doc.get("id"), ok=False, error=doc.get("error"))
    env = rv.get("envelope") or {}
    tu = rv.get("turning") or {}
    ca = rv.get("carpet") or {}
    cl = rv.get("clearance") or {}
    mg = cl.get("joint_range_margins") or {}
    tight = min((v for r in mg.values() for v in (r.get("margin_min"), r.get("margin_max")) if v is not None), default=None)
    g, r, kind = _worst(doc)
    sc = shortlist.scores(doc, rv, det)
    gates = shortlist.gates(doc, rv, det)
    cf = (doc.get("geometry_checks") or {}).get("camera_fit") or {}
    changed = {}
    if ref:
        changed = {k: [ref["margins"][k]["actuator"], doc["margins"][k]["actuator"]] for k in doc["margins"]
                   if ref["margins"][k]["actuator"] != doc["margins"][k]["actuator"]}
    return dict(
        label=label, id=doc["id"], ok=True, crown_m=doc["metrics"]["crown"], H_m=doc["H"], mass_kg=doc["metrics"]["mass"],
        bus=doc["selection"]["bus"], battery=doc["selection"]["battery"], walk_w=doc["power"]["total_w"],
        runtime_min=doc["runtime"]["minutes"], cost_nominal=doc["cost"]["nominal"], cost_high=doc["cost"]["high"],
        drives={k: m["actuator"] for k, m in doc["margins"].items()}, drives_changed_vs_ref=changed,
        governing=[g, kind, r], ratios={k: max(m["cont_ratio"], m["peak_ratio"], m["speed_ratio"]) for k, m in doc["margins"].items()},
        fast_walk=dict(ok=env.get("fast_walk_drives_ok"), worst=env.get("fast_walk_worst"), speed_ms=env.get("fast_walk_speed_ms")),
        turning=dict(targets_pass=tu.get("targets_pass"), worst=tu.get("worst_ratio_targets"), worst_where=tu.get("worst_targets"),
                     twist_speed_in_place=(tu.get("hip_yaw_speed_required") or {}).get("in_place_0.50"),
                     twist_speed_limit=tu.get("hip_yaw_speed_limit"), mu_needed=tu.get("mu_needed_max"),
                     min_gap_mm=tu.get("min_gap_mm")),
        carpet=dict(passes=ca.get("pass"), toe_margin_mm=ca.get("toe_clearance_margin_mm"), friction_margin=ca.get("friction_margin"),
                    medium_upright=((ca.get("standing") or {}).get("medium") or {}).get("upright")),
        stop_ms2=env.get("stop_max_one_foot_decel_ms2"), push_ms=env.get("push_max_dv_ms"),
        session_worst=env.get("session_worst"), clearance_all=(cl.get("pass") or {}).get("all"),
        tightest_joint_margin_rad=tight, overrides=doc.get("overrides"),
        packaging_builds=len(doc.get("packaging_search") or []),
        camera=dict(fits=cf.get("fits"), station=cf.get("station"), drop_mm=None if cf.get("drop_m") is None else 1e3 * cf["drop_m"]),
        person_in_view=((rv.get("sensors") or {}).get("torso_follow") or {}).get("home", {}).get("ok"),
        gates=gates, scores=sc["scores"], total=shortlist.total(sc["scores"]),
        stressed=shortlist.stress(doc, rv, det)["worst_stressed_ratio"])


def compare(out: Path, r2: Path = Path("round2")) -> dict:
    from .shortlist import detail

    def load(base: Path, cid: str):
        p = base / "candidates" / f"{cid}.json"
        if not p.exists():
            return None, {}
        idx = base / "review" / "review_index.json"
        rv = json.loads(idx.read_text()).get(cid, {}) if idx.exists() else {}
        rv = dict(rv, _detail=detail(base, cid))
        return json.loads(p.read_text()), rv

    ref = {cid: load(r2, cid)[0] for cid in ("60-M-T-10", "60-M-H-10", "75-M-T-10", "75-M-H-10")}
    rows = []
    for cid in ("60-M-T-10", "60-M-H-10", "75-M-T-10", "75-M-H-10"):
        d, rv = load(r2, cid)
        if d:
            rows.append(dict(case="round2", **row("round two", d, rv, None, rv.get("_detail"))))
    for cid in ("75-M-T-10", "75-M-H-10"):
        d, rv = load(r2 / "whatif_fr05", cid)
        if d:
            rows.append(dict(case="whatif_fr05", **row("75 cm, Froude 0.05 sizing (what-if)", d, rv, ref[cid], rv.get("_detail"))))
    for case, c in CASES.items():
        for cid in c["ids"]:
            d, rv = load(out / case, cid)
            if not d:
                continue
            fam = cid.split("-")[2]
            refd = ref.get(f"60-M-{fam}-10") if cid.startswith(("54", "57")) else ref.get(f"75-M-{fam}-10")
            rows.append(dict(case=case, **row(c["label"], d, rv, refd, rv.get("_detail"))))
    return dict(cases={k: v["label"] for k, v in CASES.items()}, rows=rows)


def _f(v, nd=2, unit=""):
    return "–" if v is None else f"{v:.{nd}f}{unit}"


def markdown(cmp: dict) -> str:
    from .report import short_part
    from .report2 import GROUP_LABEL

    head = ["Body", "Case", "Crown / hip axis", "Mass", "Governing sizing ratio", "Froude 0.05 walk", "Target turns",
            "Carpet toe margin", "Tightest joint margin", "Packaging changes", "Runtime", "Cost nominal", "Score",
            "Stressed worst", "All gates"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in cmp["rows"]:
        if not r.get("ok"):
            lines.append(f"| `{r['id']}` | {r['label']} | failed: {r.get('error')} |" + " |" * (len(head) - 3))
            continue
        g, kind, ratio = r["governing"]
        fw = r["fast_walk"]
        fws = "–" if not fw.get("worst") else f"{'within' if fw['ok'] else '**over**'}: {GROUP_LABEL.get(fw['worst'][0], fw['worst'][0])} {fw['worst'][1]:.2f}"
        tu = r["turning"]
        tus = "–" if tu.get("worst") is None else f"{'pass' if tu['targets_pass'] else '**fail**'} ({tu['worst']:.2f})"
        ov = ", ".join(f"{k} {v}" for k, v in (r.get("overrides") or [])) or "none"
        lines.append("| " + " | ".join([
            f"`{r['id']}`", r["label"], f"{100 * r['crown_m']:.1f} / {100 * r['H_m']:.1f} cm", f"{r['mass_kg']:.2f} kg",
            f"{GROUP_LABEL.get(g, g)} {kind} {ratio:.2f}", fws, tus, _f(r["carpet"]["toe_margin_mm"], 1, " mm"),
            _f(r["tightest_joint_margin_rad"], 2, " rad"), ov, f"{r['runtime_min']:.1f} min", f"${r['cost_nominal'] / 1000:.1f}k",
            f"{r['total']:.2f}", _f(r["stressed"], 2), "yes" if r["gates"]["all"] else "**no**"]) + " |")
    dl = ["| Body | Case | Drives changed against the round-two base of its class |", "|---|---|---|"]
    for r in cmp["rows"]:
        if r.get("ok") and r["case"] != "round2":
            ch = "; ".join(f"{GROUP_LABEL.get(k, k)}: {short_part(a)} → {short_part(b)}" for k, (a, b) in r["drives_changed_vs_ref"].items())
            dl.append(f"| `{r['id']}` | {r['label']} | {ch or 'none'} |")
    return "\n".join(["# Size study (round two)", "",
                      "Generated by `python -m designlab.sizestudy`. Method: module docstring. Every body is screened like "
                      "the round-two candidates and scored with the round-two scorecard. Model estimates, none measured.", "",
                      "## Comparison", ""] + lines + ["", "## Drive changes", ""] + dl) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="round2/size_study")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    import designlab  # noqa: F401

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cases = a.cases or list(CASES)
    if not a.report_only:
        tasks = [(c, cid, str(out / c)) for c in cases for cid in CASES[c]["ids"]]
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers) as pool:
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
    cmp = compare(out)
    (out / "size_study.json").write_text(json.dumps(cmp, indent=1, default=str))
    (out / "SIZE_STUDY.md").write_text(markdown(cmp))
    print(markdown(cmp))


if __name__ == "__main__":
    main()
