"""Round-one tables, generated from the Stage A records and the review-set screens.

Run from hardware/trex_form_study after stage_a and round1:
    python -m designlab.report --stage-a round1/stage_a --review round1/review --out round1
writes round1/TABLES.md (every comparison table quoted in ROUND1_REVIEW.md and SCREENING.md) and
round1/review_data.json (the same numbers for the review page).  Nothing here computes physics:
it only selects, groups and formats numbers already recorded by the sweep and the screens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEIGHTS = (60, 75, 100, 120)
WIDTH_NAMES = {"N": "Narrow", "M": "Balanced", "W": "Broad"}
FAMILY_NAMES = {"T": "Tapered", "S": "Substantial", "H": "Hybrid"}
VARIANTS = (("", "default"), ("-anat", "anatomical tail"), ("-compact", "compact"), ("-bighead", "large skull"),
            ("-side", "side-by-side hip roll"), ("-serial", "serial leg (PR #662-like)"), ("-belly", "belly battery"))
GROUP_LABEL = {"hip_roll": "hip roll", "hip_pitch": "hip pitch", "hip_yaw": "thigh twist", "knee": "knee",
               "ankle_a": "ankle (motor A)", "ankle_b": "ankle (motor B)", "neck": "neck", "tail": "tail"}
CLASS_A = ("leg_actuator", "battery", "power", "compute", "control", "sensors", "expression_actuator")
CLASS_C = ("cover", "cosmetic")


def load_stage_a(stage_a: Path) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted((stage_a / "candidates").glob("*.json"))}


def load_review(review: Path) -> dict:
    p = review / "review_index.json"
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- formatting


def _f(x, nd=1, unit=""):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "yes" if x else "**no**"
    return f"{x:.{nd}f}{unit}"


def _cm(x):
    return _f(100 * x, 1) if x is not None else "–"


def _usd(x):
    return f"${x / 1000:.1f}k" if x is not None else "–"


def table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def short_part(key: str | None) -> str:
    if not key:
        return "–"
    k = key.replace("myactuator_rmd_", "RMD-").replace("cubemars_", "").replace("robstride_", "RobStride ")
    k = k.replace("feetech_", "Feetech ").replace("dynamixel_", "Dynamixel ").replace("_e_l", "-L").replace("_e", "")
    k = k.replace("_v3", "").replace("_", "-")
    return k.upper() if k.startswith(("ak", "AK")) else k


def worst(doc: dict) -> tuple[str, float, str]:
    """Group with the highest requirement/capability ratio, its ratio and which limit it is."""
    best = ("", 0.0, "")
    for g, m in doc["margins"].items():
        for kind in ("cont", "peak", "speed"):
            r = m[f"{kind}_ratio"]
            if r > best[1]:
                best = (g, r, kind)
    return best


def hold_txt(doc: dict, group: str, nd: int = 1) -> str:
    """Single-support hold for a joint group, or a plain statement when no balanced pose exists."""
    if not all(h["feasible"] for h in doc["holds"].values()):
        return "no balanced pose"
    return _f(doc["margins"][group]["hold"], nd, " N·m")


def volume_class_mass(doc: dict) -> dict:
    mb = doc["mass_by_category"]
    a = sum(v for k, v in mb.items() if k in CLASS_A)
    c = sum(v for k, v in mb.items() if k in CLASS_C)
    total = sum(mb.values())
    return dict(A=a, B=total - a - c, C=c, total=total)


def flags(doc: dict) -> list[str]:
    """Plain-language problems with a candidate (empty = none found by Stage A)."""
    out = []
    if not doc.get("ok"):
        return [f"build/sizing failed: {doc.get('error', '')}"]
    if not all(h["feasible"] for h in doc["holds"].values()):
        out.append("no balanced single-support pose within joint limits")
    if doc.get("walk_feasible") is False:
        out.append("reference walk leaves the support pad (CoP margin < -10 mm)")
    g, r, kind = worst(doc)
    if r > 1.0 + 1e-9:
        out.append(f"{GROUP_LABEL.get(g, g)} {kind} requirement exceeds the largest suitable catalog part ({r:.2f}x)")
    if doc["runtime"]["minutes"] < 20:
        out.append(f"runtime {doc['runtime']['minutes']:.0f} min < 20 min target")
    if doc["cost"]["high"] > 20000:
        out.append(f"high-side cost {_usd(doc['cost']['high'])} > $20k")
    notes = sum((h.get("notes", []) for h in doc.get("sizing_history", [])), [])
    if any("sizing stopped" in n for n in notes):
        out.append("sizing mass-growth guard triggered")
    return out


# ---------------------------------------------------------------- tables


def t_sizes(D: dict) -> str:
    head = ["Class", "Crown", "Hip H", "Length", "Max width", "Mass", "Bus", "Battery", "Walk power", "Runtime",
            "Cost low / nom / high"]
    rows = []
    for h in HEIGHTS:
        d = D.get(f"{h}-M-S-10")
        if not d or not d.get("ok"):
            continue
        m = d["metrics"]
        c = d["cost"]
        rows.append([f"{h} cm", _cm(m["crown"]) + " cm", _cm(m["hip_height"]) + " cm", _cm(m["length"]) + " cm",
                     _cm(m["max_width"]) + " cm", _f(m["mass"], 1, " kg"), f"{d['selection']['bus']:.0f} V",
                     d["selection"]["battery"].replace("_", " "), _f(d["power"]["total_w"], 0, " W"),
                     _f(d["runtime"]["minutes"], 0, " min"),
                     f"{_usd(c['low'])} / {_usd(c['nominal'])} / {_usd(c['high'])}"])
    return table(head, rows)


def t_actuators(D: dict) -> str:
    """Selected part per joint group and the governing requirement ratio (requirement / planning limit)."""
    groups = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "neck", "tail")
    head = ["Class"] + [GROUP_LABEL[g] for g in groups]
    rows = []
    for h in HEIGHTS:
        d = D.get(f"{h}-M-S-10")
        if not d or not d.get("ok"):
            continue
        cells = [f"{h} cm"]
        for g in groups:
            mm = d["margins"][g]
            r = max(mm["cont_ratio"], mm["peak_ratio"], mm["speed_ratio"])
            cells.append(f"{short_part(mm['actuator'])} ({r:.2f})")
        rows.append(cells)
    return table(head, rows)


def t_loads(D: dict) -> str:
    groups = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "neck", "tail")
    head = ["Class"] + [f"{GROUP_LABEL[g]} hold / RMS / peak" for g in groups]
    rows = []
    for h in HEIGHTS:
        d = D.get(f"{h}-M-S-10")
        if not d or not d.get("ok"):
            continue
        cells = [f"{h} cm"]
        for g in groups:
            mm = d["margins"][g]
            cells.append(f"{mm['hold']:.1f} / {mm['rms']:.1f} / {mm['peak_req'] / 1.25:.1f}")
        rows.append(cells)
    return table(head, rows)


def t_widths(D: dict) -> str:
    head = ["Candidate", "Torso width", "Hip-pitch spacing", "Foot spacing", "Thigh width", "Hardware width",
            "Max width", "Crown / torso width", "Crown / hardware width", "Crown / max width", "Hip H / foot spacing",
            "Hip-roll hold", "Walk CoP margin", "Mass"]
    rows = []
    for h in HEIGHTS:
        for w in "NMW":
            d = D.get(f"{h}-{w}-S-10")
            if not d or not d.get("ok"):
                continue
            m, r = d["metrics"], d["metrics"]["ratios"]
            rows.append([f"{h}-{w}-S-10 ({WIDTH_NAMES[w]})", _cm(m["torso_width"]) + " cm",
                         _cm(m["hip_pitch_spacing"]) + " cm", _cm(m["foot_spacing"]) + " cm",
                         _cm(m["thigh_width"]) + " cm", _cm(m["hardware_width"]) + " cm", _cm(m["max_width"]) + " cm",
                         _f(r["crown_over_torso_width"], 2), _f(r["crown_over_hardware_width"], 2),
                         _f(m["crown"] / m["max_width"], 2),
                         _f(r["hip_height_over_foot_spacing"], 2), hold_txt(d, "hip_roll"),
                         _f(1000 * d["gaits"]["nominal"]["min_cop_margin_single"], 0, " mm"),
                         _f(m["mass"], 1, " kg")])
    return table(head, rows)


def t_families(D: dict) -> str:
    head = ["Candidate", "Torso width × depth", "Torso length / depth", "Cover mass (C)", "Hardware (A)",
            "Arrangement (B)", "Total", "CoM ahead of hip", "Hip-pitch hold", "Covers cost (nom)"]
    rows = []
    for h in HEIGHTS:
        for f in "TSH":
            d = D.get(f"{h}-M-{f}-10")
            if not d or not d.get("ok"):
                continue
            m = d["metrics"]
            v = volume_class_mass(d)
            rows.append([f"{h}-M-{f}-10 ({FAMILY_NAMES[f]})",
                         f"{_cm(m['torso_width'])} × {_cm(m['torso_depth'])} cm",
                         _f(m["ratios"]["torso_length_over_depth"], 2), _f(v["C"], 2, " kg"), _f(v["A"], 2, " kg"),
                         _f(v["B"], 2, " kg"), _f(v["total"], 2, " kg"),
                         _f(m["ratios"]["com_ahead_of_hip_over_H"], 3, " H"),
                         _f(d["margins"]["hip_pitch"]["hold"], 1, " N·m"), _usd(d["cost"]["lines"]["covers"][1])])
    return table(head, rows)


def t_postures(D: dict) -> str:
    head = ["Candidate", "Crown", "Hip H", "CoM ahead of hip", "CoM height", "Hip-pitch hold / RMS", "Knee hold",
            "Neck hold", "Walk power", "Runtime"]
    rows = []
    for h in HEIGHTS:
        for p in (0, 10, 20):
            d = D.get(f"{h}-M-S-{p:02d}")
            if not d or not d.get("ok"):
                continue
            m = d["metrics"]
            hp = d["margins"]["hip_pitch"]
            rows.append([f"{h}-M-S-{p:02d}", _cm(m["crown"]) + " cm", _cm(m["hip_height"]) + " cm",
                         _f(m["ratios"]["com_ahead_of_hip_over_H"], 3, " H"), _cm(m["com_height"]) + " cm",
                         f"{hp['hold']:.1f} / {hp['rms']:.1f} N·m", _f(d["margins"]["knee"]["hold"], 1, " N·m"),
                         _f(d["margins"]["neck"]["hold"], 2, " N·m"), _f(d["power"]["total_w"], 0, " W"),
                         _f(d["runtime"]["minutes"], 0, " min")])
    return table(head, rows)


def t_variants(D: dict) -> str:
    head = ["Candidate", "Variant", "Length", "Tail length", "Mass", "CoM ahead of hip", "Hip-roll hold",
            "Hip-pitch hold", "Walk power", "Runtime", "Cost nom / high"]
    rows = []
    for h in HEIGHTS:
        for suf, label in VARIANTS:
            d = D.get(f"{h}-M-S-10{suf}")
            if not d or not d.get("ok"):
                continue
            m = d["metrics"]
            rows.append([f"{h}-M-S-10{suf}", label, _cm(m["length"]) + " cm", _cm(m["tail_length"]) + " cm",
                         _f(m["mass"], 1, " kg"), _f(m["ratios"]["com_ahead_of_hip_over_H"], 3, " H"),
                         hold_txt(d, "hip_roll"), hold_txt(d, "hip_pitch"), _f(d["power"]["total_w"], 0, " W"),
                         _f(d["runtime"]["minutes"], 0, " min"),
                         f"{_usd(d['cost']['nominal'])} / {_usd(d['cost']['high'])}"])
    return table(head, rows)


def t_mass_fractions(D: dict) -> str:
    head = ["Class", "Head + neck", "Tail", "One leg", "Trunk and its contents", "CoM ahead of hip"]
    rows = []
    for h in HEIGHTS:
        d = D.get(f"{h}-M-S-10")
        if not d or not d.get("ok"):
            continue
        r = d["metrics"]["ratios"]
        leg1 = r["legs_mass_fraction"] / 2
        rows.append([f"{h} cm", _f(100 * r["head_neck_mass_fraction"], 1, " %"), _f(100 * r["tail_mass_fraction"], 1, " %"),
                     _f(100 * leg1, 1, " %"),
                     _f(100 * (1 - r["head_neck_mass_fraction"] - r["tail_mass_fraction"] - r["legs_mass_fraction"]), 1,
                        " %"), _f(r["com_ahead_of_hip_over_H"], 3, " H")])
    return table(head, rows)


def t_cost_lines(D: dict) -> str:
    lines = None
    cols = []
    for h in HEIGHTS:
        d = D.get(f"{h}-M-S-10")
        if d and d.get("ok"):
            cols.append((h, d["cost"]))
            lines = lines or list(d["cost"]["lines"])
    if not cols:
        return ""
    for _, c in cols:
        for k in c["lines"]:
            if k not in lines:
                lines.append(k)
    head = ["Line (nominal; high)"] + [f"{h} cm" for h, _ in cols]
    rows = []
    for k in lines:
        rows.append([k.replace("_", " ")] + [
            (f"{_usd(c['lines'][k][1])}; {_usd(c['lines'][k][2])}" if k in c["lines"] else "–") for _, c in cols])
    rows.append(["**total**"] + [f"**{_usd(c['nominal'])}; {_usd(c['high'])}**" for _, c in cols])
    return table(head, rows)


def t_matrix(D: dict) -> str:
    head = ["Candidate", "Crown", "Mass", "Bus", "Walk power", "Runtime", "Cost nom / high", "Worst group (ratio)",
            "Flags"]
    rows = []
    for cid, d in sorted(D.items(), key=lambda kv: (int(kv[0].split("-")[0]), kv[0])):
        if not d.get("ok"):
            rows.append([cid, "–", "–", "–", "–", "–", "–", "–", "; ".join(flags(d))])
            continue
        m = d["metrics"]
        g, r, kind = worst(d)
        rows.append([cid, _cm(m["crown"]) + " cm", _f(m["mass"], 1, " kg"), f"{d['selection']['bus']:.0f} V",
                     _f(d["power"]["total_w"], 0, " W"), _f(d["runtime"]["minutes"], 0, " min"),
                     f"{_usd(d['cost']['nominal'])} / {_usd(d['cost']['high'])}",
                     f"{GROUP_LABEL.get(g, g)} {kind} ({r:.2f})", "; ".join(flags(d)) or "–"])
    return table(head, rows)


def matrix_stats(D: dict) -> dict:
    ok = [d for d in D.values() if d.get("ok")]
    out = dict(n=len(D), ok=len(ok), failed=[k for k, d in D.items() if not d.get("ok")])
    out["ss_infeasible"] = sorted(k for k, d in D.items() if d.get("ok") and not all(
        h["feasible"] for h in d["holds"].values()))
    out["walk_infeasible"] = sorted(k for k, d in D.items() if d.get("ok") and d.get("walk_feasible") is False)
    out["over_catalog"] = sorted(k for k, d in D.items() if d.get("ok") and worst(d)[1] > 1 + 1e-9)
    out["runtime_short"] = sorted(k for k, d in D.items() if d.get("ok") and d["runtime"]["minutes"] < 20)
    out["high_over_20k"] = sorted(k for k, d in D.items() if d.get("ok") and d["cost"]["high"] > 20000)
    out["nominal_over_20k"] = sorted(k for k, d in D.items() if d.get("ok") and d["cost"]["nominal"] > 20000)
    out["guard"] = sorted(k for k, d in D.items() if d.get("ok") and any(
        "sizing stopped" in n for h in d["sizing_history"] for n in h.get("notes", [])))
    by_h = {}
    for h in HEIGHTS:
        ds = [d for d in ok if d["candidate"]["height"] == h]
        if not ds:
            continue
        rng = lambda f: (min(f(d) for d in ds), max(f(d) for d in ds))  # noqa: E731
        by_h[h] = dict(n=len(ds), mass=rng(lambda d: d["metrics"]["mass"]), walk_w=rng(lambda d: d["power"]["total_w"]),
                       runtime=rng(lambda d: d["runtime"]["minutes"]), cost_nom=rng(lambda d: d["cost"]["nominal"]),
                       cost_high=rng(lambda d: d["cost"]["high"]),
                       hip_roll_hold=rng(lambda d: d["margins"]["hip_roll"]["hold"]),
                       hip_pitch_hold=rng(lambda d: d["margins"]["hip_pitch"]["hold"]),
                       com_ahead=rng(lambda d: d["metrics"]["ratios"]["com_ahead_of_hip_over_H"]),
                       foot_spacing_over_H=rng(lambda d: d["metrics"]["foot_spacing"] / d["metrics"]["hip_height"]),
                       head_neck_frac=rng(lambda d: d["metrics"]["ratios"]["head_neck_mass_fraction"]),
                       tail_frac=rng(lambda d: d["metrics"]["ratios"]["tail_mass_fraction"]),
                       leg_frac=rng(lambda d: d["metrics"]["ratios"]["legs_mass_fraction"] / 2),
                       buses=sorted({d["selection"]["bus"] for d in ds}),
                       batteries=sorted({d["selection"]["battery"] for d in ds}))
    out["by_height"] = by_h
    return out


# ---------------------------------------------------------------- screening tables


def t_screening(R: dict) -> str:
    """One row per review candidate: the five clearance gates and the main sensor numbers (see SCREENING.md)."""
    head = ["Candidate", "No contact at home", "No contact, nominal walk", "Swing ≥ 0.04 H", "Tail ≥ 0.25 H",
            "Battery path", "Camera view blocked", "Torso visible ahead", "Nearest camera floor ahead of toes",
            "Foot path covered (ToF + camera)", "Lidar blocked ≤ 1 m", "Same model as Stage A"]
    rows = []
    pf = lambda v: "–" if v is None else ("pass" if v else "**fail**")  # noqa: E731
    for cid in sorted(R, key=lambda k: (int(k.split("-")[0]), k)):
        r = R[cid]
        ps = (r.get("clearance") or {}).get("pass", {})
        se = r.get("sensors") or {}
        cam = se.get("camera_blocked", {}).get("home")
        torso = se.get("torso_follow", {}).get("home", {}).get("ok")
        floor = se.get("centre_column_nearest_visible_floor_ahead_of_toes_m")
        foot = (se.get("near_strip_coverage", {}).get("foot_path", {}) or {}).get("combined")
        lid = (se.get("lidar_blocked_1m") or {}).get("as_built")
        rows.append([cid, pf(ps.get("interference_free_home")), pf(ps.get("interference_free_nominal_gait")),
                     pf(ps.get("swing_clearance")), pf(ps.get("tail_clearance")), pf(ps.get("battery_path")),
                     "–" if cam is None else f"{100 * cam:.1f} %", pf(torso),
                     "–" if floor is None else f"{floor:.2f} m", "–" if foot is None else f"{100 * foot:.0f} %",
                     "none fitted" if lid is None else f"{100 * lid:.0f} %",
                     _f(r.get("matches_stage_a")) if r.get("matches_stage_a") is not None else "–"])
    return table(head, rows)


def build(stage_a: Path, review: Path) -> tuple[str, dict]:
    D = load_stage_a(stage_a)
    R = load_review(review)
    st = matrix_stats(D)
    parts = [
        "# Round-one tables (generated)",
        "",
        "Generated by `python -m designlab.report` from `round1/stage_a/candidates/*.json` and "
        "`round1/review/review_index.json`. Do not edit by hand. Definitions: `DEFINITIONS.md`.",
        "",
        f"Stage A: {st['n']} candidates, {st['ok']} built and sized.",
        "",
        "## 1. Height classes (balanced width, substantial family, 10° posture)",
        "",
        t_sizes(D),
        "",
        "## 2. Selected actuators (requirement / planning-limit ratio of the governing limit)",
        "",
        t_actuators(D),
        "",
        "## 3. Joint loads, N·m: single-support hold / nominal-walk RMS / brisk-walk peak",
        "",
        t_loads(D),
        "",
        "## 4. Mass distribution",
        "",
        t_mass_fractions(D),
        "",
        "## 5. Widths (substantial family, 10°)",
        "",
        t_widths(D),
        "",
        "## 6. Body families (balanced width, 10°)",
        "",
        t_families(D),
        "",
        "## 7. Postures (balanced width, substantial family)",
        "",
        t_postures(D),
        "",
        "## 8. Proportion, layout and battery variants (balanced width, substantial family, 10°)",
        "",
        t_variants(D),
        "",
        "## 9. Cost lines (balanced width, substantial family, 10°)",
        "",
        t_cost_lines(D),
        "",
    ]
    if R:
        parts += ["## 10. Review-set screening", "", t_screening(R), ""]
    parts += ["## 11. Every Stage A candidate", "", t_matrix(D), ""]
    data = dict(stats=st, candidates={k: _page_record(d) for k, d in D.items()},
                review={k: {kk: v for kk, v in r.items() if kk not in ("clearance_trace", "sensors_trace")}
                        for k, r in R.items()})
    return "\n".join(parts), data


def _page_record(d: dict) -> dict:
    if not d.get("ok"):
        return dict(ok=False, error=d.get("error"))
    m = d["metrics"]
    g, r, kind = worst(d)
    return dict(ok=True, candidate=d["candidate"], crown=m["crown"], hip=m["hip_height"], length=m["length"],
                max_width=m["max_width"], torso_width=m["torso_width"], torso_depth=m["torso_depth"],
                foot_spacing=m["foot_spacing"], mass=m["mass"], walk_w=d["power"]["total_w"],
                runtime=d["runtime"]["minutes"], cost=[d["cost"]["low"], d["cost"]["nominal"], d["cost"]["high"]],
                bus=d["selection"]["bus"], battery=d["selection"]["battery"], worst=[g, r, kind],
                parts={k: v["actuator"] for k, v in d["margins"].items()},
                holds={k: v["hold"] for k, v in d["margins"].items()}, flags=flags(d),
                com_ahead=m["ratios"]["com_ahead_of_hip_over_H"], classes=volume_class_mass(d),
                metrics=m, margins=d["margins"], power=dict(total_w=d["power"]["total_w"],
                                                             drives_w=d["power"]["drives_w"],
                                                             aux_w=d["power"]["aux"]["input_w"],
                                                             peak_drives_w=d["power"]["peak_drives_w"]),
                runtime_range=d["runtime"]["minutes_range"], cost_lines=d["cost"]["lines"],
                mass_by_category=d["mass_by_category"], ss_feasible=all(h["feasible"] for h in d["holds"].values()),
                walk_feasible=d.get("walk_feasible"),
                walk_cop_margin_mm=1000 * d["gaits"]["nominal"]["min_cop_margin_single"],
                selection=d["selection"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage-a", default="round1/stage_a")
    ap.add_argument("--review", default="round1/review")
    ap.add_argument("--out", default="round1")
    a = ap.parse_args()
    md, data = build(Path(a.stage_a), Path(a.review))
    out = Path(a.out)
    (out / "TABLES.md").write_text(md)
    (out / "review_data.json").write_text(json.dumps(data, indent=1))
    print(f"wrote {out / 'TABLES.md'} and {out / 'review_data.json'}")


if __name__ == "__main__":
    main()
