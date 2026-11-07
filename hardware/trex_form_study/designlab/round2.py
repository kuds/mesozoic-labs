"""Round two: the owner's round-one choices (decision log P10-P14), sized from scratch and screened.

Candidates: 60 and 75 cm; Tapered and Hybrid bodies; 10 deg chest-up; for each, the balanced-width base
and four single-change variants (narrow stance, larger skull, compact proportions, and the level 0 deg
posture the owner asked to keep researching, P15): 20 in all.  The level variant is the same body as its
base (sizes are matched at 10 deg), so its crown is lower.  Every
candidate uses generator revision 2 (``Candidate.overrides`` ``("rev", 2)``; the changes are F8-F10 in
DECISION_LOG.md) and the Raspberry Pi 5 + AI HAT+ compute, and is sized by the same loop as Stage A
(loads -> drives and battery -> rebuild until stable).

Run from hardware/trex_form_study:
  python -m designlab.round2 --out round2 --workers 3        # size, screen and plate everything
  python -m designlab.round2 --out round2 --plates-only      # redraw the plates
Outputs (under --out):
  candidates/<id>.json          sized record (same fields as a Stage A record, plus ``rev``)
  review/mjcf/<id>.json         ledger of the rebuilt model (the XML regenerates; SHA-256 checked)
  review/screening/<id>_*.json  clearance (round-two gates), sensors, turning and carpet screens
  review/plates/*.png           review plates
  review/review_index.json      one entry per candidate: SHA match and screen summaries
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import replace
from pathlib import Path

from .params import Candidate

HEIGHTS = (60, 75)
FAMILIES = ("T", "H")
#: (width, proportion, posture) per variant; each variant changes one thing against the 10 deg,
#: balanced-width base
VARIANTS = (("M", "base", 10), ("N", "base", 10), ("M", "bighead", 10), ("M", "compact", 10), ("M", "base", 0))
VARIANT_LABEL = {("M", "base", 10): "base", ("N", "base", 10): "narrow stance", ("M", "bighead", 10): "larger skull",
                 ("M", "compact", 10): "compact", ("M", "base", 0): "level (0°)"}
COMPUTE = "rpi5_8gb_ai_hat_plus_13tops"
REV = 2


def candidate(cid: str) -> Candidate:
    from .params import parse

    return replace(parse(cid), overrides=(("rev", REV),))


def candidate_ids() -> list[str]:
    return [Candidate(h, w, f, po, proportion=p).id for h in HEIGHTS for f in FAMILIES for w, p, po in VARIANTS]


def variant_key(cid: str) -> tuple:
    from .params import parse

    c = parse(cid)
    return (c.width, c.proportion, c.posture)


def base_ids() -> list[str]:
    return [f"{h}-M-{f}-10" for h in HEIGHTS for f in FAMILIES]


def start_selection(height: int) -> dict:
    from . import hardware as hw

    sel = hw.default_selection(height)
    sel["compute"] = COMPUTE
    return sel


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _json(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return str(x)


# ------------------------------------------------------------------------------------------------
# Sizing (as Stage A)
# ------------------------------------------------------------------------------------------------
def size_one(cid: str, out: str, xml_dir: str) -> dict:
    import designlab  # noqa: F401

    from . import metrics, pipeline
    from .stage_a import _mass_by_category, _mass_by_volume_class

    t0 = time.time()
    cand = candidate(cid)
    try:
        res = pipeline.size_candidate(cand, Path(xml_dir), selection=start_selection(cand.height))
        meas = metrics.measure(res["model"], res["design"], res["q_home"])
        ev = res["evaluation"]
        d = res["design"]
        doc = dict(id=cid, rev=REV, candidate=dict(height=cand.height, width=cand.width, family=cand.family,
                                                   posture=cand.posture, proportion=cand.proportion,
                                                   leg_layout=cand.leg_layout, battery_pos=cand.battery_pos),
                   B=d.B, H=d.sk.H, xml_sha256=res["summary"]["xml_sha256"], selection=dict(res["selection"]),
                   metrics=meas, margins=ev["margins"], holds=ev["holds"], power=ev["power"], runtime=ev["runtime"],
                   cost=ev["cost"], gaits=ev["gaits"], sizing_history=ev["sizing_history"],
                   walk_feasible=ev["walk_feasible"], geometry_checks=d.geometry_checks, notes=d.notes,
                   mass_by_category=_mass_by_category(res["rows"]),
                   mass_by_volume_class=_mass_by_volume_class(res["rows"]), seconds=time.time() - t0, ok=True)
    except Exception as e:  # keep failed cases
        doc = dict(id=cid, rev=REV, ok=False, error=repr(e), traceback=traceback.format_exc(),
                   seconds=time.time() - t0)
    Path(out, "candidates").mkdir(parents=True, exist_ok=True)
    Path(out, "candidates", f"{cid}.json").write_text(json.dumps(doc, indent=1, default=_json))
    return dict(id=cid, ok=doc["ok"], seconds=doc["seconds"])


def _size_star(a):
    return size_one(*a)


# ------------------------------------------------------------------------------------------------
# Rebuild and screen
# ------------------------------------------------------------------------------------------------
def screen_one(cid: str, out: Path) -> dict:
    """Rebuild one sized candidate (SHA-checked), write its ledger and run every screen."""
    import designlab  # noqa: F401

    from . import gait, metrics, mjcf, sizing
    from .pipeline import build_candidate

    rec = json.loads((out / "candidates" / f"{cid}.json").read_text())
    sel = dict(rec["selection"])
    cand = candidate(cid)
    rv = out / "review"
    models = rv / "mjcf"
    res = build_candidate(cand, models, selection=sel, B=rec["B"], write=False)
    xml = models / f"{cid}.xml"
    design, M, q_home = res["design"], res["model"], res["q_home"]
    meas = metrics.measure(M, design, q_home)
    mjcf.write_ledger(design, res["rows"], models / f"{cid}.json",
                      extra=dict(summary=res["summary"], metrics=meas, sized_record_sha256=rec.get("xml_sha256"),
                                 rev=REV))
    H = design.sk.H
    trajs = {name: gait.generate(M, q_home, H, spec) for name, spec in (("nominal", sizing.NOMINAL),
                                                                          ("brisk", sizing.BRISK))}
    scr = rv / "screening"
    scr.mkdir(parents=True, exist_ok=True)
    result = dict(id=cid, xml_sha256=_sha(xml), sized_sha256=rec.get("xml_sha256"))
    result["matches_sized"] = result["xml_sha256"] == result["sized_sha256"]
    for name, fn in (("clearance", _clearance), ("sensors", _sensors), ("turning", _turning), ("carpet", _carpet)):
        try:
            summ = fn(cid, xml, q_home, trajs, design, rec, scr)
            if summ is not None:
                result[name] = summ
        except Exception as e:
            result[f"{name}_error"] = repr(e)
            result[f"{name}_trace"] = traceback.format_exc()
    return result


def _clearance(cid, xml, q_home, trajs, design, rec, scr):
    from . import clearance

    cl = clearance.screen(str(xml), q_home, trajs, out_png=str(scr / f"{cid}_clearance.png"), H=design.sk.H, gates="r2")
    (scr / f"{cid}_clearance.json").write_text(json.dumps(cl, indent=1, default=_json))
    return cl.get("summary", {})


def _sensors(cid, xml, q_home, trajs, design, rec, scr):
    from . import sensors

    se = sensors.screen(str(xml), q_home, trajs["nominal"], None, png_path=str(scr / f"{cid}_sensors.png"),
                        gait_stride=2, title=cid)
    (scr / f"{cid}_sensors.json").write_text(json.dumps(se, indent=1, default=_json))
    return se.get("summary", {})


def _turning(cid, xml, q_home, trajs, design, rec, scr):
    try:
        from . import turning
    except ImportError:
        return None
    return turning.screen_record(cid, rec, str(xml), q_home, design, scr)


def _carpet(cid, xml, q_home, trajs, design, rec, scr):
    try:
        from . import carpet
    except ImportError:
        return None
    return carpet.screen_record(cid, rec, str(xml), q_home, design, trajs, scr)


def _screen_star(a):
    try:
        return screen_one(*a)
    except Exception as e:
        return dict(id=a[0], error=repr(e), trace=traceback.format_exc())


# ------------------------------------------------------------------------------------------------
# Plates
# ------------------------------------------------------------------------------------------------
def make_plates(out: Path, r1_models: Path | None = None) -> list[str]:
    import designlab  # noqa: F401

    from . import plates
    from .round1 import compact_png

    rv = out / "review"
    m = rv / "mjcf"
    P = rv / "plates"
    P.mkdir(parents=True, exist_ok=True)
    have = lambda cid: (m / f"{cid}.xml").exists()  # noqa: E731
    x = lambda cid: str(m / f"{cid}.xml") if have(cid) else None  # noqa: E731
    made = []
    for cid in candidate_ids():
        if have(cid):
            made.append(plates.candidate_plate(x(cid), str(P / f"{cid}_four_views.png"), cid, ""))
    for cid in base_ids():
        if have(cid):
            made.append(plates.cutaway_plate(x(cid), str(P / f"{cid}_cutaway.png"), f"{cid} installed hardware"))
    lab = lambda ids: [(x(i), i) for i in ids if have(i)]  # noqa: E731
    made.append(plates.lineup(lab(base_ids()), str(P / "bases_common_scale.png"), mode="exterior", normalize=False,
                              title="Round two: 60 and 75 cm, Tapered and Hybrid, at one physical scale"))
    made.append(plates.lineup(lab(base_ids()), str(P / "bases_silhouettes.png"), mode="silhouette", normalize=False,
                              title="Silhouettes at one physical scale"))
    fam = {"T": "Tapered", "H": "Hybrid"}
    for h in HEIGHTS:
        rows = [[x(Candidate(h, w, f, po, proportion=p).id) for w, p, po in VARIANTS] for f in FAMILIES]
        cols = [VARIANT_LABEL[v] for v in VARIANTS]
        if not any(c for r in rows for c in r):
            continue
        made.append(plates.matrix(rows, [fam[f] for f in FAMILIES], cols, str(P / f"variants_{h}_side.png"), view="side",
                                  mode="exterior", title=f"{h} cm: base and single-change variants, side view"))
        made.append(plates.matrix(rows, [fam[f] for f in FAMILIES], cols, str(P / f"variants_{h}_front.png"),
                                  view="front", mode="exterior", title=f"{h} cm: base and variants, front view"))
    made.append(plates.matrix([[x(f"{h}-M-{f}-10") for f in FAMILIES] for h in HEIGHTS], ["60 cm", "75 cm"],
                              [fam[f] for f in FAMILIES], str(P / "families_r2.png"), view="side", mode="exterior",
                              title="Tapered and Hybrid at 60 and 75 cm (10° chest-up, balanced width)"))
    made.append(plates.matrix([[x(f"{h}-M-{f}-10") for f in FAMILIES] for h in HEIGHTS], ["60 cm", "75 cm"],
                              [fam[f] for f in FAMILIES], str(P / "families_r2_cutaway.png"), view="side", mode="cutaway",
                              title="Installed hardware: Raspberry Pi 5 compute, head camera, tail-base battery"))
    if r1_models is not None:
        pairs = [(str(r1_models / f"{c}.xml"), x(c)) for c in ("60-M-T-10", "60-M-H-10") if have(c)
                 and (r1_models / f"{c}.xml").exists()]
        if pairs:
            made.append(plates.matrix([list(p) for p in pairs], [p[1].split("/")[-1][:-4] for p in pairs],
                                      ["Round one", "Round two"], str(P / "r1_vs_r2_60_cutaway.png"), view="side",
                                      mode="cutaway", title="60 cm: round one against round two, installed hardware"))
    made.append(plates.silhouette_sheet(lab(candidate_ids()), str(P / "silhouettes_r2.png")))
    made = [str(p) for p in made if p]
    for png in made + [str(q) for q in (rv / "screening").glob("*.png")]:
        compact_png(png)
    return made


def rebuild_round1(ids: list[str], out: Path) -> Path:
    """Round-one models of the same ids (generator revision 1, Jetson) for side-by-side plates."""
    from .pipeline import build_candidate
    from .round1 import _final_mass_guess

    stage_a = Path("round1/stage_a/candidates")
    out.mkdir(parents=True, exist_ok=True)
    from .params import parse

    for cid in ids:
        rec = json.loads((stage_a / f"{cid}.json").read_text())
        sel = dict(rec["selection"])
        sel["mass_guess"] = _final_mass_guess(rec)
        build_candidate(parse(cid), out, selection=sel, B=rec["B"], write=False)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="round2")
    ap.add_argument("--xml-dir", default=None, help="sizing models (default <out>/models, not committed)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--skip-sizing", action="store_true")
    ap.add_argument("--plates-only", action="store_true")
    ap.add_argument("--r1-models", default=None, help="directory for rebuilt round-one comparison models")
    a = ap.parse_args()
    import designlab  # noqa: F401

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = a.only or candidate_ids()
    ctx = mp.get_context("spawn")
    if not a.plates_only:
        if not a.skip_sizing:
            xml_dir = a.xml_dir or str(out / "models")
            with ctx.Pool(a.workers) as pool:
                for r in pool.imap_unordered(_size_star, [(i, str(out), xml_dir) for i in ids]):
                    print("sized", r, flush=True)
        with ctx.Pool(a.workers) as pool:
            results = list(pool.imap_unordered(_screen_star, [(i, out) for i in ids]))
        idx_path = out / "review" / "review_index.json"
        index = json.loads(idx_path.read_text()) if idx_path.exists() else {}
        index.update({r["id"]: r for r in results})
        idx_path.parent.mkdir(parents=True, exist_ok=True)
        idx_path.write_text(json.dumps(index, indent=1, default=_json))
        for r in results:
            print("screened", r["id"], "sha ok" if r.get("matches_sized") else "SHA DIFFERS", {k: v for k, v in r.items()
                  if k.endswith("_error") or k == "error"}, flush=True)
    r1 = rebuild_round1(["60-M-T-10", "60-M-H-10"], Path(a.r1_models)) if a.r1_models else None
    made = make_plates(out, r1)
    (out / "review" / "plates" / "plates_index.json").write_text(json.dumps(made, indent=1))
    print(f"{len(made)} plates")


if __name__ == "__main__":
    main()
