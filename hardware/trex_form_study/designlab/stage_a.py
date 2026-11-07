"""Stage A: broad geometry, packaging, load, power and cost sweep (no rendering).

Every candidate is built, sized (actuators and battery re-selected from computed loads until
stable), measured and costed.  Outputs (under --out):
  candidates/<id>.json   compact result: candidate, selection, metrics, margins, power, runtime,
                         cost, holds, sizing history, XML SHA-256 (the XML itself regenerates with
                         ``python -m designlab.cli build <id>``)
  summary.csv / summary.json   one row per candidate
Run from hardware/trex_form_study:  python -m designlab.stage_a --out round1/stage_a --workers 3
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
import traceback
from dataclasses import replace
from pathlib import Path

from .params import FAMILIES, HEIGHT_CLASSES, POSTURES, WIDTHS, Candidate, parse


def candidate_list() -> list[Candidate]:
    cands = [Candidate(h, w, f, p) for h in HEIGHT_CLASSES for w in WIDTHS for f in FAMILIES for p in POSTURES]
    extra = []
    for h in HEIGHT_CLASSES:
        base = Candidate(h, "M", "S", 10)
        extra.append(replace(base, battery_pos="belly"))
        for prop in ("anat", "compact", "bighead"):
            extra.append(replace(base, proportion=prop))
        extra.append(replace(base, leg_layout="side"))
        extra.append(replace(base, leg_layout="serial"))
    return cands + extra


def run_one(cid: str, out: str, xml_dir: str) -> dict:
    import designlab  # noqa: F401  (GL backend before mujoco)

    from . import metrics, pipeline

    t0 = time.time()
    cand = parse(cid)
    try:
        res = pipeline.size_candidate(cand, Path(xml_dir))
        meas = metrics.measure(res["model"], res["design"], res["q_home"])
        ev = res["evaluation"]
        doc = dict(id=cid, candidate=dict(height=cand.height, width=cand.width, family=cand.family, posture=cand.posture,
                                          proportion=cand.proportion, leg_layout=cand.leg_layout,
                                          battery_pos=cand.battery_pos),
                   B=res["design"].B, H=res["design"].sk.H, xml_sha256=res["summary"]["xml_sha256"],
                   selection=dict(res["selection"]),
                   metrics=meas, margins=ev["margins"], holds=ev["holds"], power=ev["power"], runtime=ev["runtime"],
                   cost=ev["cost"], gaits=ev["gaits"], sizing_history=ev["sizing_history"],
                   walk_feasible=ev["walk_feasible"],
                   mass_by_category=_mass_by_category(res["rows"]), seconds=time.time() - t0, ok=True)
    except Exception as e:  # keep failed cases
        doc = dict(id=cid, ok=False, error=repr(e), traceback=traceback.format_exc(), seconds=time.time() - t0)
    Path(out, "candidates").mkdir(parents=True, exist_ok=True)
    Path(out, "candidates", f"{cid}.json").write_text(json.dumps(doc, indent=1, default=_json))
    return dict(id=cid, ok=doc["ok"], seconds=doc["seconds"])


def _run_star(args):
    return run_one(*args)


def remeasure_one(cid: str, out: str, xml_dir: str) -> dict:
    """Rebuild a sized candidate from its record (same selection and B; the XML SHA-256 must match)
    and recompute its metrics with the current ``metrics.measure``.  Loads, power and cost are not
    touched: they do not depend on the measurement code."""
    import hashlib

    import designlab  # noqa: F401

    from . import metrics, pipeline

    path = Path(out, "candidates", f"{cid}.json")
    doc = json.loads(path.read_text())
    if not doc.get("ok"):
        return dict(id=cid, skipped=True)
    res = pipeline.build_candidate(parse(cid), Path(xml_dir), selection=dict(doc["selection"]), B=doc["B"],
                                   write=False)
    sha = hashlib.sha256((Path(xml_dir) / f"{cid}.xml").read_bytes()).hexdigest()
    if sha != doc["xml_sha256"]:
        return dict(id=cid, error="rebuilt model differs from the sweep record")
    doc["metrics"] = metrics.measure(res["model"], res["design"], res["q_home"])
    doc["metrics_revision"] = "exact primitive extremes (round-one review)"
    path.write_text(json.dumps(doc, indent=1, default=_json))
    return dict(id=cid, ok=True)


def _remeasure_star(args):
    try:
        return remeasure_one(*args)
    except Exception as e:  # keep going; report
        return dict(id=args[0], error=repr(e))


def _mass_by_category(rows):
    out = {}
    for r in rows:
        out[r["category"]] = out.get(r["category"], 0.0) + r["mass_kg"]
    return out


def _json(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def flatten(doc: dict) -> dict:
    if not doc.get("ok"):
        return dict(id=doc["id"], ok=False, error=doc.get("error", ""))
    m = doc["metrics"]
    r = m["ratios"]
    row = dict(id=doc["id"], ok=True, **doc["candidate"], B=doc["B"], H=doc["H"], crown=m["crown"], length=m["length"],
               mass=m["mass"], torso_width=m["torso_width"], hardware_width=m["hardware_width"],
               thigh_width=m["thigh_width"], hip_pitch_spacing=m["hip_pitch_spacing"],
               roll_axis_offset=m["roll_axis_offset"], foot_spacing=m["foot_spacing"], support_width=m["support_width"],
               support_area=m["support_area"], max_width=m["max_width"], skull_length=m["skull_length"],
               tail_length=m["tail_length"], com_ahead_over_H=r["com_ahead_of_hip_over_H"],
               crown_over_torso_width=r["crown_over_torso_width"], crown_over_hardware_width=r["crown_over_hardware_width"],
               hip_over_foot_spacing=r["hip_height_over_foot_spacing"], torso_len_over_depth=r["torso_length_over_depth"],
               skull_over_torso_len=r["skull_length_over_torso_length"], length_over_crown=r["length_over_crown"],
               tail_mass_frac=r["tail_mass_fraction"], head_neck_mass_frac=r["head_neck_mass_fraction"],
               bus=doc["selection"].get("bus"), battery=doc["selection"].get("battery"),
               walk_w=doc["power"]["total_w"], runtime_min=doc["runtime"]["minutes"],
               cost_low=doc["cost"]["low"], cost_nom=doc["cost"]["nominal"], cost_high=doc["cost"]["high"])
    worst = max(doc["margins"].items(), key=lambda kv: max(kv[1]["cont_ratio"], kv[1]["peak_ratio"], kv[1]["speed_ratio"]))
    row["worst_group"] = worst[0]
    row["worst_ratio"] = max(worst[1]["cont_ratio"], worst[1]["peak_ratio"], worst[1]["speed_ratio"])
    for g in ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "neck", "tail"):
        mm = doc["margins"][g]
        row[f"{g}_part"] = mm["actuator"]
        row[f"{g}_hold"] = mm["hold"]
        row[f"{g}_rms"] = mm["rms"]
        row[f"{g}_cont_ratio"] = mm["cont_ratio"]
    row["ss_hold_feasible"] = all(h["feasible"] for h in doc["holds"].values())
    row["walk_feasible"] = doc.get("walk_feasible", None)
    row["walk_min_cop_margin_mm"] = 1000 * doc["gaits"]["nominal"]["min_cop_margin_single"]
    row["sizing_notes"] = "; ".join(sum((h.get("notes", []) for h in doc["sizing_history"]), []))
    return row


def aggregate(out: Path):
    docs = [json.loads(p.read_text()) for p in sorted((out / "candidates").glob("*.json"))]
    rows = [flatten(d) for d in docs]
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "summary.json").write_text(json.dumps(rows, indent=1, default=_json))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="round1/stage_a")
    ap.add_argument("--xml-dir", default=None, help="where generated MJCFs go (default: <out>/models, not committed)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--remeasure", action="store_true",
                    help="recompute metrics of finished candidates from rebuilt models (SHA-checked)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.remeasure:
        ids = sorted(p.stem for p in (out / "candidates").glob("*.json"))
        xml_dir = a.xml_dir or str(out / "models")
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers) as pool:
            for r in pool.imap_unordered(_remeasure_star, [(i, str(out), xml_dir) for i in ids], chunksize=4):
                if not r.get("ok"):
                    print(r, flush=True)
        a.aggregate_only = True
    if not a.aggregate_only:
        ids = a.only or [c.id for c in candidate_list()]
        done = {p.stem for p in (out / "candidates").glob("*.json")} if (out / "candidates").exists() else set()
        todo = [i for i in ids if i not in done]
        xml_dir = a.xml_dir or str(out / "models")
        print(f"{len(ids)} candidates, {len(todo)} to run, {a.workers} workers", flush=True)
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers) as pool:
            for r in pool.imap_unordered(_run_star, [(i, str(out), xml_dir) for i in todo], chunksize=1):
                print(r, flush=True)
    rows = aggregate(out)
    print(f"aggregated {len(rows)} rows")


if __name__ == "__main__":
    main()
