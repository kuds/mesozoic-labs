"""Round one review set: rebuild the selected candidates exactly as sized in Stage A, screen them
(clearance, sensors) and produce every review plate.

Run from hardware/trex_form_study after stage_a:   python -m designlab.round1 --out round1/review
Each review model is rebuilt from its Stage A selection, so its XML SHA-256 equals the one recorded
in round1/stage_a/candidates/<id>.json (checked and recorded in review_index.json).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import traceback
from pathlib import Path

SIZES = ["60-M-S-10", "75-M-S-10", "100-M-S-10", "120-M-S-10"]
FAMILY_GRID = [f"100-M-{f}-{p:02d}" for f in "TSH" for p in (0, 10, 20)]
FAMILIES_60 = ["60-M-T-10", "60-M-S-10", "60-M-H-10"]
POSTURES_75 = ["75-M-S-00", "75-M-S-10", "75-M-S-20"]
WIDTHS_100 = ["100-N-S-10", "100-M-S-10", "100-W-S-10"]
PROPORTIONS_100 = ["100-M-S-10", "100-M-S-10-anat", "100-M-S-10-compact", "100-M-S-10-bighead"]
LAYOUTS_60 = ["60-M-S-10", "60-M-S-10-side", "60-M-S-10-serial", "60-M-S-10-belly"]
REVIEW_SET = list(dict.fromkeys(SIZES + FAMILY_GRID + FAMILIES_60 + POSTURES_75 + WIDTHS_100 + PROPORTIONS_100
                                + LAYOUTS_60))


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rebuild(cid: str, stage_a: Path, out: Path) -> dict:
    """Rebuild one candidate with its Stage A selection; write MJCF, ledger, clearance and sensors."""
    import designlab  # noqa: F401

    from . import gait, metrics, mjcf, sizing
    from .params import parse
    from .pipeline import build_candidate

    rec = json.loads((stage_a / "candidates" / f"{cid}.json").read_text())
    sel = dict(rec["selection"])
    sel["mass_guess"] = _final_mass_guess(rec)
    cand = parse(cid)
    models = out / "mjcf"
    res = build_candidate(cand, models, selection=sel, B=rec["B"], write=False)
    xml = models / f"{cid}.xml"
    design, M, q_home = res["design"], res["model"], res["q_home"]
    meas = metrics.measure(M, design, q_home)
    mjcf.write_ledger(design, res["rows"], models / f"{cid}.json",
                      extra=dict(summary=res["summary"], metrics=meas, stage_a=rec.get("xml_sha256")))
    trajs = {}
    for name, spec in (("nominal", sizing.NOMINAL), ("brisk", sizing.BRISK)):
        trajs[name] = gait.generate(M, q_home, design.sk.H, spec)
    out_screen = out / "screening"
    out_screen.mkdir(parents=True, exist_ok=True)
    result = dict(id=cid, xml=str(xml), xml_sha256=_sha(xml), stage_a_sha256=rec.get("xml_sha256"))
    result["matches_stage_a"] = result["xml_sha256"] == result["stage_a_sha256"]
    try:
        from . import clearance

        cl = clearance.screen(str(xml), q_home, trajs, out_png=str(out_screen / f"{cid}_clearance.png"))
        (out_screen / f"{cid}_clearance.json").write_text(json.dumps(cl, indent=1, default=_json))
        result["clearance"] = cl.get("summary", {})
    except Exception as e:
        result["clearance_error"] = repr(e)
        result["clearance_trace"] = traceback.format_exc()
    try:
        from . import sensors

        se = sensors.screen(str(xml), q_home, trajs["nominal"], None, png_path=str(out_screen / f"{cid}_sensors.png"),
                            gait_stride=2, title=cid)
        (out_screen / f"{cid}_sensors.json").write_text(json.dumps(se, indent=1, default=_json))
        result["sensors"] = se.get("summary", {})
    except Exception as e:
        result["sensors_error"] = repr(e)
        result["sensors_trace"] = traceback.format_exc()
    return result


def _final_mass_guess(rec: dict) -> float:
    """The structure-sizing mass guess used by the final Stage A build (see pipeline.size_candidate):
    the mass of the last sizing iteration, or of the one before it when the spiral guard stopped."""
    if "mass_guess" in rec["selection"]:
        return rec["selection"]["mass_guess"]
    its = [h for h in rec["sizing_history"] if isinstance(h.get("iteration"), int)]
    last = its[-1]
    if any("sizing stopped" in n for n in last.get("notes", [])) and len(its) > 1:
        return its[-2]["mass"]
    return last["mass"]


def _json(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def _rebuild_star(a):
    try:
        return rebuild(*a)
    except Exception as e:
        return dict(id=a[0], error=repr(e), trace=traceback.format_exc())


def compact_png(path) -> None:
    """Re-encode a rendered PNG as a 256-colour palette image (Floyd-Steinberg), about 60 % smaller with no
    visible change at review sizes.  Applied to every committed plate and screening image."""
    from PIL import Image

    im = Image.open(path).convert("RGB")
    im.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.FLOYDSTEINBERG).save(path, optimize=True)


def make_plates(out: Path, index: dict):
    import designlab  # noqa: F401

    from . import plates

    m = out / "mjcf"
    P = out / "plates"
    P.mkdir(parents=True, exist_ok=True)
    x = lambda cid: str(m / f"{cid}.xml")  # noqa: E731
    made = []
    for cid in REVIEW_SET:
        if (m / f"{cid}.xml").exists():
            made.append(plates.candidate_plate(x(cid), str(P / f"{cid}_four_views.png"), cid, ""))
    for cid in ("60-M-S-10", "100-M-S-10", "120-M-S-10", "100-M-H-10", "100-M-T-10", "60-M-S-10-serial"):
        made.append(plates.cutaway_plate(x(cid), str(P / f"{cid}_cutaway.png"), f"{cid} installed hardware"))
    lab = lambda ids: [(x(i), i) for i in ids]  # noqa: E731
    made.append(plates.lineup(lab(SIZES), str(P / "sizes_common_scale.png"), mode="exterior", normalize=False,
                              title="Four height classes at one physical scale (S family, balanced width, 10°)"))
    made.append(plates.lineup(lab(SIZES), str(P / "sizes_height_normalized.png"), mode="exterior", normalize=True,
                              title="Same four candidates scaled to equal crown height"))
    made.append(plates.lineup(lab(SIZES), str(P / "sizes_silhouettes.png"), mode="silhouette", normalize=False,
                              title="Silhouettes at one physical scale"))
    grid = [[x(f"100-M-{f}-{p:02d}") for p in (0, 10, 20)] for f in "TSH"]
    made.append(plates.matrix(grid, ["Tapered", "Substantial", "Hybrid"], ["Level", "10° chest-up", "20° chest-up"],
                              str(P / "family_posture_matrix_100.png"), view="side", mode="exterior",
                              title="Body family × posture, 100 cm class, balanced width"))
    made.append(plates.matrix([[x(c) for c in FAMILIES_60]], ["60 cm"], ["Tapered", "Substantial", "Hybrid"],
                              str(P / "families_60.png"), view="side", mode="exterior", title="Body families, 60 cm class"))
    made.append(plates.matrix([[x(c) for c in POSTURES_75]], ["75 cm, S"], ["Level", "10°", "20°"],
                              str(P / "postures_75.png"), view="side", mode="exterior", title="Posture, 75 cm class"))
    made.append(plates.matrix([[x(c) for c in WIDTHS_100]], ["100 cm, S, 10°"], ["Narrow", "Balanced", "Broad"],
                              str(P / "widths_100_front.png"), view="front", mode="exterior",
                              title="Width variants, front view"))
    made.append(plates.matrix([[x(c) for c in WIDTHS_100]], ["100 cm, S, 10°"], ["Narrow", "Balanced", "Broad"],
                              str(P / "widths_100_top.png"), view="top", mode="exterior", title="Width variants, top view"))
    made.append(plates.lineup(lab(PROPORTIONS_100), str(P / "proportions_100.png"), mode="exterior", normalize=False,
                              title="Proportion variants, 100 cm class (default, anatomical tail, compact, large skull)"))
    made.append(plates.silhouette_sheet(lab(FAMILY_GRID), str(P / "silhouettes_family_posture.png")))
    made.append(plates.matrix([[x(c) for c in LAYOUTS_60]], ["60 cm, S, 10°"],
                              ["Tandem roll (default)", "Side-by-side roll", "Serial (PR #662-like)", "Belly battery"],
                              str(P / "layouts_60_cutaway.png"), view="side", mode="cutaway",
                              title="Leg and battery layouts, cutaway"))
    made = [str(p) for p in made if p]
    for png in made + [str(q) for q in (out / "screening").glob("*.png")]:
        compact_png(png)
    return made


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="round1/review")
    ap.add_argument("--stage-a", default="round1/stage_a")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--plates-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    idx_path = out / "review_index.json"
    if not a.plates_only:
        ctx = mp.get_context("spawn")
        with ctx.Pool(a.workers) as pool:
            results = list(pool.imap_unordered(_rebuild_star, [(c, Path(a.stage_a), out) for c in REVIEW_SET]))
        index = {r["id"]: r for r in results}
        idx_path.write_text(json.dumps(index, indent=1, default=_json))
    index = json.loads(idx_path.read_text())
    made = make_plates(out, index)
    (out / "plates" / "plates_index.json").write_text(json.dumps(made, indent=1))
    print(f"{len(index)} review candidates; {len(made)} plates")


if __name__ == "__main__":
    main()
