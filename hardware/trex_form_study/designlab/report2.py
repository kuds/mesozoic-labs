"""Round-two tables, generated from the round-two records and screens (and round one for comparison).

Run from hardware/trex_form_study after ``designlab.round2``:
    python -m designlab.report2 --round2 round2 --round1 round1 --out round2
writes round2/TABLES.md and round2/review_data.json.  Nothing here computes physics: it only selects,
groups and formats numbers recorded by the sizing runs and the screens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .params import Candidate
from .report import _cm, _f, _usd, flags, short_part, table, volume_class_mass, worst
from .round2 import FAMILIES, HEIGHTS, VARIANT_LABEL, VARIANTS, base_ids, candidate_ids, variant_key

FAMILY_NAMES = {"T": "Tapered", "H": "Hybrid"}
GROUPS = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b", "neck", "tail")
GROUP_LABEL = {"hip_roll": "hip roll", "hip_pitch": "hip pitch", "hip_yaw": "thigh twist", "knee": "knee",
               "ankle_a": "ankle A", "ankle_b": "ankle B", "neck": "neck", "tail": "tail"}
BATTERY = {"lipo_6s_3300": "6S 3.3 Ah", "lipo_6s_5000": "6S 5 Ah", "lipo_2x6s_3300_series": "2 × 6S 3.3 Ah"}


def load(r2: Path, r1: Path) -> tuple[dict, dict, dict, dict]:
    D = {p.stem: json.loads(p.read_text()) for p in sorted((r2 / "candidates").glob("*.json"))}
    R = json.loads((r2 / "review" / "review_index.json").read_text()) if (r2 / "review" / "review_index.json").exists() else {}
    D1 = {i: json.loads((r1 / "stage_a" / "candidates" / f"{i}.json").read_text()) for i in base_ids()
          if (r1 / "stage_a" / "candidates" / f"{i}.json").exists()}
    R1 = json.loads((r1 / "review" / "review_index.json").read_text()) if (r1 / "review" / "review_index.json").exists() else {}
    return D, R, D1, R1


def variant_of(cid: str) -> str:
    return VARIANT_LABEL.get(variant_key(cid), cid)


def _bat(doc):
    b = doc["selection"]["battery"]
    return BATTERY.get(b, b)


def t_candidates(D: dict) -> str:
    rows = []
    for cid in candidate_ids():
        d = D.get(cid)
        if not d or not d.get("ok"):
            rows.append([cid, variant_of(cid), "failed"] + ["–"] * 7)
            continue
        m = d["metrics"]
        g, r, kind = worst(d)
        rows.append([f"`{cid}`", variant_of(cid), f"{_cm(m['crown'])} / {_cm(m['hip_height'])} / {_cm(m['length'])} cm",
                     f"{_cm(m['max_width'])} cm", _f(m["mass"], 2, " kg"), f"{d['selection']['bus']:.0f} V, {_bat(d)}",
                     _f(d["power"]["total_w"], 0, " W"), _f(d["runtime"]["minutes"], 1, " min"),
                     f"{_usd(d['cost']['nominal'])} / {_usd(d['cost']['high'])}", f"{GROUP_LABEL.get(g, g)} {kind} ({r:.2f})"])
    return table(["Candidate", "Variant", "Crown / hip axis / length", "Max width", "Mass", "Bus, battery",
                  "Walking power", "Runtime", "Cost nominal / high", "Governing ratio"], rows)


def t_drives(D: dict) -> str:
    rows = []
    for cid in candidate_ids():
        d = D.get(cid)
        if not d or not d.get("ok"):
            continue
        cells = [f"`{cid}`"]
        for g in GROUPS:
            mm = d["margins"][g]
            r = max(mm["cont_ratio"], mm["peak_ratio"], mm["speed_ratio"])
            cells.append(f"{short_part(mm['actuator'])} ({r:.2f})")
        rows.append(cells)
    return table(["Candidate"] + [GROUP_LABEL[g] for g in GROUPS], rows)


def t_loads(D: dict) -> str:
    rows = []
    for cid in candidate_ids():
        d = D.get(cid)
        if not d or not d.get("ok"):
            continue
        feas = all(h["feasible"] for h in d["holds"].values())
        mm = d["margins"]
        hold = (lambda g: _f(mm[g]["hold"], 2)) if feas else (lambda g: "no balanced pose")
        rows.append([f"`{cid}`", hold("hip_pitch"), hold("hip_roll"), hold("hip_yaw"), hold("knee"),
                     _f(mm["hip_pitch"]["rms"], 2), _f(d["metrics"]["ratios"]["com_ahead_of_hip_over_H"], 3, " H"),
                     _f(1000 * d["gaits"]["nominal"]["min_cop_margin_single"], 1, " mm")])
    return table(["Candidate", "Hip-pitch hold, N·m", "Hip-roll hold", "Thigh-twist hold", "Knee hold",
                  "Hip-pitch walk RMS", "CoM ahead of hip", "Walk CoP margin (single support)"], rows)


def t_r1_vs_r2(D: dict, D1: dict, R: dict, R1: dict) -> str:
    rows = []
    for cid in base_ids():
        d, o = D.get(cid), D1.get(cid)
        if not d or not o:
            continue
        gc = d.get("geometry_checks", {})
        cf = gc.get("camera_fit", {})
        knee2 = ((R.get(cid, {}).get("clearance") or {}).get("joint_range_usable") or {}).get("left_knee", {})
        knee1 = ((R1.get(cid, {}).get("clearance") or {}).get("joint_range_usable") or {}).get("left_knee", {})
        k2 = (knee2.get("first_new_interference") or [None, None])[1]
        k1 = (knee1.get("first_new_interference") or [None, None])[1]
        for label, a, b in (("Mass", _f(o["metrics"]["mass"], 2, " kg"), _f(d["metrics"]["mass"], 2, " kg")),
                            ("Compute", "Jetson Orin Nano kit (31 W budget)", "Pi 5 + AI HAT+ (9.5 W budget)"),
                            ("Walking power", _f(o["power"]["total_w"], 0, " W"), _f(d["power"]["total_w"], 0, " W")),
                            ("Runtime", _f(o["runtime"]["minutes"], 1, " min"), _f(d["runtime"]["minutes"], 1, " min")),
                            ("Battery", _bat(o), _bat(d)),
                            ("Cost nominal / high", f"{_usd(o['cost']['nominal'])} / {_usd(o['cost']['high'])}",
                             f"{_usd(d['cost']['nominal'])} / {_usd(d['cost']['high'])}"),
                            ("Hip spacing (leg plane)", _cm(o["metrics"]["hip_pitch_spacing"]) + " cm",
                             _cm(d["metrics"]["hip_pitch_spacing"]) + " cm"),
                            ("Camera station / depth below roof", "0.24 / 4 mm (box cut by the cover)",
                             f"{cf.get('station', 0):.3f} / {1000 * cf.get('drop_m', 0):.0f} mm" if cf.get("fits") else "no fit"),
                            ("Knee flexion usable", _f(k1, 2, " rad") if k1 else ("not screened" if cid not in R1 else "full"),
                             _f(k2, 2, " rad") if k2 else "full")):
            rows.append([f"`{cid}`", label, a, b])
    return table(["Candidate", "Quantity", "Round one", "Round two"], rows)


def t_variants(D: dict) -> str:
    rows = []
    for h in HEIGHTS:
        for f in FAMILIES:
            base = D.get(f"{h}-M-{f}-10")
            if not base or not base.get("ok"):
                continue
            for w, p, po in VARIANTS[1:]:
                cid = Candidate(h, w, f, po, proportion=p).id
                d = D.get(cid)
                if not d or not d.get("ok"):
                    continue
                bm, dm = base["metrics"], d["metrics"]
                rows.append([f"`{cid}`", VARIANT_LABEL[(w, p, po)],
                             f"{d['metrics']['mass'] - bm['mass']:+.2f} kg",
                             f"{100 * (d['margins']['hip_pitch']['hold'] / base['margins']['hip_pitch']['hold'] - 1):+.0f} %",
                             f"{100 * (d['margins']['hip_roll']['hold'] / base['margins']['hip_roll']['hold'] - 1):+.0f} %",
                             f"{d['runtime']['minutes'] - base['runtime']['minutes']:+.1f} min",
                             f"{(d['cost']['nominal'] - base['cost']['nominal']) / 1000:+.1f}k",
                             f"{dm['foot_spacing'] / dm['hip_height']:.2f} H", _cm(dm["skull_length"]) + " cm",
                             _f(dm["ratios"]["com_ahead_of_hip_over_H"], 3, " H")])
    return table(["Candidate", "Variant", "Mass vs base", "Hip-pitch hold vs base", "Hip-roll hold vs base",
                  "Runtime vs base", "Nominal cost vs base", "Foot spacing", "Skull length", "CoM ahead of hip"], rows)


def t_clearance(R: dict) -> str:
    rows = []
    for cid in candidate_ids():
        cl = (R.get(cid) or {}).get("clearance") or {}
        if not cl:
            rows.append([f"`{cid}`"] + ["–"] * 9)
            continue
        ps = cl.get("pass", {})
        sw = (cl.get("swing_clearance") or {}).get("nominal", {})
        mg = cl.get("joint_range_margins", {})
        tight = min(((n, v) for n, r in mg.items() for v in (r.get("margin_min"), r.get("margin_max")) if v is not None),
                    key=lambda t: t[1], default=(None, None))
        rows.append([f"`{cid}`"] + [_f(ps.get(k)) for k in ("interference_free_home", "interference_free_nominal_gait",
                                                            "interference_free_brisk_gait", "component_overlap_free",
                                                            "swing_clearance", "tail_clearance", "battery_path")]
                    + [f"{tight[0].replace('left_', '').replace('_', ' ')} {tight[1]:.2f} rad" if tight[0] else "–",
                       _f(sw.get("mid_swing_min_mm"), 0, " mm")])
    return table(["Candidate", "Home", "Nominal walk", "Brisk walk", "No part overlaps", "Swing ≥ 0.04 H", "Tail ≥ 0.25 H",
                  "Battery path", "Tightest joint-range margin", "Mid-swing toe clearance"], rows)


def t_sensors(R: dict, D: dict) -> str:
    rows = []
    for cid in candidate_ids():
        se = (R.get(cid) or {}).get("sensors") or {}
        if not se:
            continue
        tf = se.get("torso_follow", {}).get("home", {})
        tof = (se.get("tof", {}).get("tof_l", {}) or {}).get("floor_ahead_of_toes_m") or [None, None]
        cf = (D.get(cid) or {}).get("geometry_checks", {}).get("camera_fit", {})
        rows.append([f"`{cid}`", f"{cf.get('station', 0):.3f} / {1000 * cf.get('drop_m', 0):.0f} mm" if cf else "–",
                     f"{100 * se['camera_blocked']['home']:.1f} %", "yes" if tf.get("ok") else "**no**",
                     _f(se.get("centre_column_nearest_visible_floor_ahead_of_toes_m"), 2, " m"),
                     f"{tof[0]:.2f}–{tof[1]:.2f} m" if tof[0] is not None else "–",
                     f"{100 * se['near_strip_coverage']['foot_path']['combined']:.0f} %"])
    return table(["Candidate", "Camera station / depth below roof", "View blocked by own body",
                  "Person's torso at 1.5–3 m (home)", "Nearest floor seen, ahead of toes", "ToF floor band", "Foot path covered"],
                 rows)


def t_mass_classes(D: dict) -> str:
    rows = []
    for cid in base_ids():
        d = D.get(cid)
        if not d or not d.get("ok"):
            continue
        v = volume_class_mass(d)
        rows.append([f"`{cid}`", _f(v["A"], 2, " kg"), _f(v["B"], 2, " kg"), _f(v["C"], 2, " kg"), _f(v["total"], 2, " kg")])
    return table(["Candidate", "Hardware (A)", "Arrangement (B)", "Coverings (C)", "Total"], rows)


def t_cost_lines(D: dict) -> str:
    keys = None
    rows = []
    for cid in base_ids():
        d = D.get(cid)
        if not d or not d.get("ok"):
            continue
        lines = d["cost"]["lines"]
        keys = keys or list(lines)
        rows.append([f"`{cid}`"] + [_usd(lines[k][1]) for k in keys] + [_usd(d["cost"]["nominal"])])
    return table(["Candidate"] + (keys or []) + ["Total nominal"], rows) if rows else ""


def t_flags(D: dict) -> str:
    rows = [[f"`{cid}`", "; ".join(flags(D[cid])) or "none"] for cid in candidate_ids() if cid in D]
    return table(["Candidate", "Flags"], rows)


def t_turning(R: dict) -> str:
    rows = []
    for cid in candidate_ids():
        tu = (R.get(cid) or {}).get("turning")
        if not tu:
            continue
        rows.append([f"`{cid}`", json.dumps(tu)[:400]])
    return table(["Candidate", "Turning summary"], rows) if rows else "_Not run._"


def t_carpet(R: dict) -> str:
    rows = []
    for cid in candidate_ids():
        ca = (R.get(cid) or {}).get("carpet")
        if not ca:
            continue
        rows.append([f"`{cid}`", json.dumps(ca)[:400]])
    return table(["Candidate", "Carpet summary"], rows) if rows else "_Not run._"


def build(r2: Path, r1: Path) -> tuple[str, dict]:
    D, R, D1, R1 = load(r2, r1)
    ok = [c for c in candidate_ids() if D.get(c, {}).get("ok")]
    passed = [c for c in candidate_ids() if ((R.get(c) or {}).get("clearance") or {}).get("pass", {}).get("all")]
    md = ["# Round-two tables", "",
          "Generated by `python -m designlab.report2` from `round2/candidates/*.json` and the screens in "
          "`round2/review/`. Every number is a model estimate; none is measured.", "",
          f"Round two: {len(D)} candidates, {len(ok)} built and sized; {len(passed)} of {len(R)} screened pass every "
          "round-two clearance gate.", "",
          "## 1. Candidates (Raspberry Pi 5 + AI HAT+ compute; 10° chest-up except the level variants)", "",
          t_candidates(D), "",
          "## 2. Selected drives (part and requirement / planning-limit ratio of the governing limit)", "", t_drives(D), "",
          "## 3. Joint loads and balance", "", t_loads(D), "",
          "## 4. Round one against round two, base bodies", "", t_r1_vs_r2(D, D1, R, R1), "",
          "## 5. Variants against their base", "", t_variants(D), "",
          "## 6. Clearance (round-two gates)", "", t_clearance(R), "",
          "## 7. Sensor coverage", "", t_sensors(R, D), "",
          "## 8. Turning loads", "", t_turning(R), "",
          "## 9. Carpet", "", t_carpet(R), "",
          "## 10. Mass by volume class (ledger rows), base bodies", "", t_mass_classes(D), "",
          "## 11. Cost lines, nominal, base bodies", "", t_cost_lines(D), "",
          "## 12. Flags", "", t_flags(D), ""]
    data = dict(candidates={c: D[c] for c in D}, review=R, round1_bases=D1)
    return "\n".join(md), data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round2", default="round2")
    ap.add_argument("--round1", default="round1")
    ap.add_argument("--out", default="round2")
    a = ap.parse_args()
    md, data = build(Path(a.round2), Path(a.round1))
    out = Path(a.out)
    (out / "TABLES.md").write_text(md)
    (out / "review_data.json").write_text(json.dumps(data, indent=1, default=str))
    print(f"wrote {out / 'TABLES.md'} and {out / 'review_data.json'}")


if __name__ == "__main__":
    main()
