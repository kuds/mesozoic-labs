"""Round-two scorecard and shortlist: which bodies are most likely to achieve the target behaviors.

Version 2 (overnight, after the independent cross-check, round2/CROSSCHECK.md).  Version 1 counted the 75 cm neck
servo in four categories, reported a stress test it never used, scored packaging without the turn clearances, kept
a sub-score that was constant on every body and did not require the animal-like walk; it also could not score the
what-if and refined bodies.  Every input is still a recorded screen result; nothing is recomputed.

1. Gates.  A candidate is out if it fails any of: every round-two clearance gate; every drive within its planning
   limits in the sizing walks; the target turns (0.35 rad/s walking, 0.5 rad/s in place: drives, IK, no contact);
   the Froude 0.05 walk within the drives (new in version 2: the owner's first priority is natural walking, and a
   body whose drives cannot produce the animal's walking pace is not carried into verification as sized); a
   person's torso visible at 1.5-3 m with the head level; at least 20 min runtime; the nominal reference walk's
   CoP within 10 mm of the pad; the carpet gate when it has run.
2. Margins.  Each behavior gets a 0-1 score from how far its governing quantity is from its limit,
   ``min(1, max(0, headroom / reference))``.  Each drive counts once: the leg drives (hip roll, hip pitch, thigh
   twist, knee, ankles) set natural walking, turning and endurance; the neck and tail servos set head and tail.
   Natural walking uses the spec walks' requirement from the turning screen's like-for-like straight walks, the
   same measure for every body whether or not it was sized with the Froude 0.05 walk, and the Froude 0.05 walk.
3. Weights from the owner's priorities (``WEIGHTS``), unchanged.
4. Stress test, used in the robustness score: torque ratios x 1.10 / 0.85 (robot 10 % heavier, drives 15 % weaker)
   and speed ratios / 0.85 (weaker drives; mass does not change the speed a walk needs), over the sizing walks,
   the target turns and the Froude 0.05 walk, leg drives only.
5. Shortlist.  The best passing body at each height, then the other family's counterpart of the overall best body
   (same height and variant), so verification compares size on one family and family at one size.  Looks are the
   owner's call and are not scored.

Run from hardware/trex_form_study:  python -m designlab.shortlist --round2 round2 [--extra <dir> ...]
``--extra`` adds bodies from what-if, size-study or refinement directories (same layout as round2/), labelled by
directory, so they are ranked with the round-two candidates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .round2 import VARIANT_LABEL, candidate_ids, variant_key

VERSION = 2
WEIGHTS = dict(natural_walking=0.20, turning=0.20, carpet=0.10, robustness=0.15, following=0.12, endurance=0.10,
               head_and_tail=0.05, packaging=0.05, cost=0.03)
STRESS = 1.10 / 0.85  # torque ratios: 10 % heavier, drives 15 % weaker
STRESS_SPEED = 1.0 / 0.85  # speed ratios: drives 15 % weaker (mass does not change the speed a walk needs)
LEG = ("hip_roll", "hip_pitch", "hip_yaw", "knee", "ankle_a", "ankle_b")
HEAD_TAIL = ("neck", "tail")
GROUP_LABEL = {"hip_roll": "hip roll", "hip_pitch": "hip pitch", "hip_yaw": "thigh twist", "knee": "knee",
               "ankle_a": "ankle A", "ankle_b": "ankle B", "neck": "neck", "tail": "tail"}


def _clip(x, ref):
    return float(max(0.0, min(1.0, x / ref))) if ref > 0 else 0.0


def _ratio(m: dict) -> float:
    return max(m["cont_ratio"], m["peak_ratio"], m["speed_ratio"])


def _worst_margin(doc, groups=None) -> tuple[str, float]:
    best = ("", 0.0)
    for g, m in doc["margins"].items():
        if groups is not None and g not in groups:
            continue
        r = _ratio(m)
        if r > best[1]:
            best = (g, r)
    return best


def _turn_worst(tu: dict | None) -> float | None:
    if not tu:
        return None
    v = tu.get("worst_ratio_targets")
    return float(v) if v is not None else None


def detail(base: Path, cid: str) -> dict:
    """The full envelope and turning screens of one body (``<base>/review/screening``), if present."""
    out = {}
    for k in ("envelope", "turning"):
        p = Path(base) / "review" / "screening" / f"{cid}_{k}.json"
        if p.exists():
            out[k] = json.loads(p.read_text())
    return out


def _turn_leg(det: dict) -> dict | None:
    """Leg-drive ratios in the target turns, per metric (cont, peak_req, speed_req): worst over groups and cases."""
    tu = det.get("turning")
    if not tu:
        return None
    tg = [c for c, v in tu["cases"].items() if v["role"] == "target"]
    out = {}
    for c in tg:
        for g in LEG:
            v = tu["verdict"][c].get(g)
            if not v:
                continue
            for k, r in v["ratio_to_limit"].items():
                if r is not None and r > out.get(k, (0.0, None, None))[0]:
                    out[k] = (float(r), g, c)
    return out


def _spec_leg(det: dict) -> dict | None:
    """Leg-drive ratios in the spec's walks (nominal and brisk), per group and metric, from the turning screen's
    like-for-like straight requirement (``verdict[case][group]["straight_required"]`` over its planning limits).
    The same measure for every body, whether or not it was sized with an extra walk (Froude 0.05)."""
    tu = det.get("turning")
    if not tu:
        return None
    case = next(c for c, v in tu["cases"].items() if v["role"] == "target")
    out = {}
    for g in LEG:
        v = tu["verdict"][case].get(g)
        if v:
            out[g] = {k: v["straight_required"][k] / v["limits"][k] for k in ("cont", "peak_req", "speed_req")
                      if v["limits"].get(k)}
    return out


def _fast_leg(det: dict) -> dict | None:
    """Leg-drive ratios in the Froude 0.05 walk (envelope), or None when inconclusive or absent."""
    ev = det.get("envelope")
    if not ev or ev["fast_walk"].get("drives_ok") is None:
        return None
    gs = ev["fast_walk"]["groups"]
    return {g: gs[g] for g in LEG if g in gs}


def gates(doc: dict, rv: dict, det: dict | None = None) -> dict:
    cl = (rv.get("clearance") or {}).get("pass", {})
    se = rv.get("sensors") or {}
    tu = rv.get("turning") or {}
    ca = rv.get("carpet") or {}
    env = rv.get("envelope") or {}
    g, r = _worst_margin(doc)
    out = dict(
        clearance=bool(cl.get("all")),
        drives_sizing_walks=r <= 1.0 + 1e-9,
        turning_targets=(tu.get("targets_pass") if tu else None),
        animal_walk=(env.get("fast_walk_drives_ok") if env else None),
        person_in_view=bool((se.get("torso_follow") or {}).get("home", {}).get("ok")),
        runtime_20min=doc["runtime"]["minutes"] >= 20.0,
        reference_walk_cop=bool(doc.get("walk_feasible")),
        carpet=(ca.get("pass") if ca else None),
    )
    out["all"] = all(v is not False for v in out.values())
    return out


def stress(doc: dict, rv: dict, det: dict | None = None) -> dict:
    """Leg drives with the robot 10 % heavier and the drives 15 % weaker: torque ratios x 1.10/0.85, speed ratios
    / 0.85, over the sizing walks, the target turns and the Froude 0.05 walk.  Worst value and where it is."""
    det = det or {}
    vals = []
    sp = _spec_leg(det)
    for g in LEG:
        if sp and g in sp:
            r = sp[g]
            vals += [(max(r.get("cont", 0.0), r.get("peak_req", 0.0)) * STRESS, g, "spec walks torque"),
                     (r.get("speed_req", 0.0) * STRESS_SPEED, g, "spec walks speed")]
        else:
            m = doc["margins"][g]
            vals += [(max(m["cont_ratio"], m["peak_ratio"]) * STRESS, g, "sizing torque"),
                     (m["speed_ratio"] * STRESS_SPEED, g, "sizing speed")]
    tl = _turn_leg(det)
    if tl:
        for k, (r, g, c) in tl.items():
            vals.append((r * (STRESS_SPEED if k.startswith("speed") else STRESS), g, f"turn {c} {k}"))
    fl = _fast_leg(det)
    if fl:
        for g, v in fl.items():
            vals += [(max(v["cont_ratio"], v["peak_ratio"]) * STRESS, g, "Froude 0.05 torque"),
                     (v["speed_ratio"] * STRESS_SPEED, g, "Froude 0.05 speed")]
    worst = max(vals)
    return dict(worst_stressed_ratio=worst[0], where=[worst[1], worst[2]],
                spec_walks_hold=max(v for v, _, w in vals if w.startswith(("sizing", "spec"))) <= 1.0)


def scores(doc: dict, rv: dict, det: dict | None = None) -> dict:
    """0-1 per behavior, with the quantities behind each."""
    det = det or {}
    env = rv.get("envelope") or {}
    se = rv.get("sensors") or {}
    tu = rv.get("turning") or {}
    ca = rv.get("carpet") or {}
    cl = rv.get("clearance") or {}
    H = doc["H"]
    s, why = {}, {}
    # natural walking: leg-drive headroom in the spec walks (nominal and brisk) and in the Froude 0.05 walk
    sp = _spec_leg(det)
    wr = max(max(r.values()) for r in sp.values()) if sp else _worst_margin(doc, LEG)[1]
    fl = _fast_leg(det)
    fw = max(_ratio_e(v) for v in fl.values()) if fl else None
    parts = [_clip(1.0 - wr, 0.30)] + ([_clip(1.0 - fw, 0.30)] if fw is not None else [])
    s["natural_walking"] = sum(parts) / len(parts)
    why["natural_walking"] = dict(spec_walks_worst_leg_ratio=wr, fast_walk_worst_leg_ratio=fw)
    # turning: leg-drive headroom in the target turns, thigh-twist range used, friction needed (yaw moment included)
    if tu:
        tl = _turn_leg(det)
        tr = max(v[0] for v in tl.values()) if tl else _turn_worst(tu)
        rng = tu.get("hip_yaw_range_used_max")
        mu = tu.get("mu_needed_max")
        parts = []
        if tr is not None:
            parts.append(_clip(1.0 - tr, 0.30))
        if rng is not None:
            parts.append(_clip(1.0 - rng, 0.40))
        if mu is not None:
            parts.append(_clip(0.8 - mu, 0.30))
        s["turning"] = sum(parts) / len(parts) if parts else None
        why["turning"] = dict(worst_leg_ratio=tr, thigh_twist_range_used=rng, mu_needed=mu)
    else:
        s["turning"], why["turning"] = None, "turning screen not run"
    # carpet
    if ca:
        cm = ca.get("toe_clearance_margin_mm")
        fm = ca.get("friction_margin")
        parts = []
        if cm is not None:
            parts.append(_clip(cm, 15.0))
        if fm is not None:
            parts.append(_clip(fm, 0.30))
        s["carpet"] = sum(parts) / len(parts) if parts else None
        why["carpet"] = dict(toe_clearance_margin_mm=cm, friction_margin=fm)
    else:
        s["carpet"], why["carpet"] = None, "carpet screen not run"
    # robustness: one-step push (Froude-normalised), one-foot stopping and the stress test
    pf = env.get("push_max_dv_ms")
    st = env.get("stop_max_one_foot_decel_ms2")
    sw = stress(doc, rv, det)["worst_stressed_ratio"]
    parts = [_clip(1.25 - sw, 0.40)]
    if pf is not None:
        parts.append(_clip(pf / (9.81 * H) ** 0.5, 0.35))
    if st is not None:
        parts.append(_clip(st, 1.5))
    s["robustness"] = sum(parts) / len(parts)
    why["robustness"] = dict(push_dv_ms=pf, push_froude=None if pf is None else pf / (9.81 * H) ** 0.5,
                             one_foot_stop_ms2=st, stressed_worst_leg_ratio=sw)
    # following: torso in view with the head level and at the neck limits, self-occlusion, floor seen
    tf = se.get("torso_follow") or {}
    ok = [bool((tf.get(k) or {}).get("ok")) for k in ("home", "neck_min", "neck_max")]
    blk = (se.get("camera_blocked") or {}).get("home")
    floor = se.get("centre_column_nearest_visible_floor_ahead_of_toes_m")
    foot = ((se.get("near_strip_coverage") or {}).get("foot_path") or {}).get("combined")
    parts = [sum(ok) / 3.0]
    if blk is not None:
        parts.append(_clip(0.10 - blk, 0.08))
    if floor is not None:
        parts.append(_clip(3.5 - floor, 2.0))
    if foot is not None:
        parts.append(_clip(foot, 0.9))
    s["following"] = sum(parts) / len(parts)
    why["following"] = dict(torso_home_down_up=ok, view_blocked=blk, nearest_floor_m=floor, foot_path_covered=foot)
    # endurance: runtime and the hottest leg drive over a session
    rt = doc["runtime"]["minutes"]
    shg = ((det.get("envelope") or {}).get("session_heat") or {}).get("groups") or {}
    sh = max((v["ratio"] for g, v in shg.items() if g in LEG), default=None)
    parts = [_clip(rt - 20.0, 25.0)]
    if sh is not None:
        parts.append(_clip(1.0 - sh, 0.30))
    s["endurance"] = sum(parts) / len(parts)
    why["endurance"] = dict(runtime_min=rt, session_hottest_leg_ratio=sh)
    # head and tail: neck and tail servos in the sizing walks and over a session
    nr = _ratio(doc["margins"]["neck"])
    tr2 = _ratio(doc["margins"]["tail"])
    nh = (shg.get("neck") or {}).get("ratio")
    parts = [_clip(1.0 - nr, 0.30), _clip(1.0 - tr2, 0.30)] + ([_clip(1.0 - nh, 0.30)] if nh is not None else [])
    s["head_and_tail"] = sum(parts) / len(parts)
    why["head_and_tail"] = dict(neck_ratio=nr, tail_ratio=tr2, neck_session_ratio=nh)
    # packaging: tightest joint-range margin, closest approach in the target turns, tight pairs in the walks
    mg = cl.get("joint_range_margins") or {}
    tight = min((v for r in mg.values() for v in (r.get("margin_min"), r.get("margin_max")) if v is not None), default=None)
    gaps = [v for c, v in (tu.get("min_gap_mm") or {}).items() if c in ("walk_0.35", "in_place_0.50") and v is not None]
    gap = min(gaps) if gaps else None
    counts = cl.get("counts") or {}
    tp = sum((counts.get(k) or {}).get("tight_pairs", 0) for k in ("gait_nominal", "gait_brisk"))
    parts = []
    if tight is not None:
        parts.append(_clip(tight - 0.10, 0.20))
    if gap is not None:
        parts.append(_clip(gap, 4.0))
    parts.append(_clip(6.0 - tp, 6.0))
    s["packaging"] = sum(parts) / len(parts)
    why["packaging"] = dict(tightest_joint_margin_rad=tight, closest_turn_gap_mm=gap, tight_pairs_in_walks=tp)
    # cost: nominal cost against the $10-20k prototype budget (P4)
    cn = doc["cost"]["nominal"]
    s["cost"] = _clip(25000.0 - cn, 10000.0)
    why["cost"] = dict(nominal_usd=cn, high_usd=doc["cost"]["high"])
    return dict(scores=s, why=why)


def _ratio_e(v: dict) -> float:
    return max(v["cont_ratio"], v["peak_ratio"], v["speed_ratio"])


def total(sc: dict) -> float:
    num = sum(WEIGHTS[k] * v for k, v in sc.items() if v is not None)
    den = sum(WEIGHTS[k] for k, v in sc.items() if v is not None)
    return num / den if den else 0.0


def _rows_from(base: Path, label: str, ids=None) -> list[dict]:
    cdir = Path(base) / "candidates"
    idx = Path(base) / "review" / "review_index.json"
    if not cdir.exists() or not idx.exists():
        return []
    R = json.loads(idx.read_text())
    rows = []
    for p in sorted(cdir.glob("*.json")):
        cid = p.stem
        if ids is not None and cid not in ids:
            continue
        d = json.loads(p.read_text())
        if not d.get("ok") or cid not in R:
            continue
        rv = R[cid]
        det = detail(base, cid)
        gt = gates(d, rv, det)
        sc = scores(d, rv, det)
        key = cid if label == "round2" else f"{cid} [{label}]"
        rows.append(dict(key=key, id=cid, source=label, variant=VARIANT_LABEL.get(variant_key(cid), cid),
                         height=d["candidate"]["height"], family=d["candidate"]["family"], overrides=d.get("overrides"),
                         gates=gt, scores=sc["scores"], why=sc["why"], total=total(sc["scores"]),
                         stress=stress(d, rv, det)))
    return rows


def build(r2: Path, extra: list[Path] | None = None) -> dict:
    rows = _rows_from(r2, "round2", set(candidate_ids()))
    for e in extra or []:
        rows += _rows_from(e, Path(e).name if Path(e).parent.name == "round2" else f"{Path(e).parent.name}/{Path(e).name}")
    passing = sorted([r for r in rows if r["gates"]["all"]], key=lambda r: -r["total"])
    picks = []
    for h in (60, 75):
        best = next((r for r in passing if r["height"] == h), None)
        if best:
            picks.append(dict(key=best["key"], id=best["id"], why=f"best-scoring passing body at {h} cm"))
    if picks:
        top = next(r for r in passing if r["key"] == picks[0]["key"]) if passing else None
        best_all = passing[0] if passing else None
        anchor = best_all or top
        other = next((r for r in passing if r["key"] not in {p["key"] for p in picks} and r["height"] == anchor["height"]
                      and r["source"] == anchor["source"] and r["family"] != anchor["family"]
                      and variant_key(r["id"]) == variant_key(anchor["id"])), None)
        why = f"the other family's counterpart of {anchor['key']} (same height and variant)"
        if other is None:
            other = next((r for r in passing if r["key"] not in {p["key"] for p in picks} and r["family"] != anchor["family"]), None)
            why = f"the best remaining passing body of the other family (no counterpart of {anchor['key']} was run)"
        if other:
            picks.append(dict(key=other["key"], id=other["id"], why=why))
    return dict(version=VERSION, weights=WEIGHTS, stress_factor=STRESS, stress_speed_factor=STRESS_SPEED, rows=rows,
                ranking=[r["key"] for r in passing],
                failing=[dict(key=r["key"], gates={k: v for k, v in r["gates"].items() if v is False}) for r in rows
                         if not r["gates"]["all"]], picks=picks)


def markdown(sl: dict) -> str:
    cats = list(WEIGHTS)
    head = ["Rank", "Body", "Variant", "Total"] + [c.replace("_", " ") for c in cats] + ["Stressed worst (leg drives)"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    by = {r["key"]: r for r in sl["rows"]}
    for i, key in enumerate(sl["ranking"], 1):
        r = by[key]
        cells = [str(i), f"`{key}`", r["variant"], f"{r['total']:.2f}"]
        cells += ["–" if r["scores"].get(c) is None else f"{r['scores'][c]:.2f}" for c in cats]
        st = r["stress"]
        cells.append(f"{st['worst_stressed_ratio']:.2f} ({GROUP_LABEL.get(st['where'][0], st['where'][0])}, {st['where'][1]})")
        lines.append("| " + " | ".join(cells) + " |")
    out = ["# Round-two scorecard and shortlist (version 2)", "",
           "Generated by `python -m designlab.shortlist`. Method: module docstring (version 2 replaces version 1 after "
           "the independent cross-check, `round2/CROSSCHECK.md`). Weights: "
           + ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in WEIGHTS.items())
           + ". Stress: torque ratios x %.2f, speed ratios x %.2f, leg drives only." % (STRESS, STRESS_SPEED), "",
           "## Picks", ""]
    out += [f"{k + 1}. `{p['key']}`: {p['why']}." for k, p in enumerate(sl["picks"])]
    out += ["", "## Passing bodies, ranked", "", "\n".join(lines), "", "## Failing a gate", ""]
    out += [f"* `{f['key']}`: " + ", ".join(k.replace("_", " ") for k in f["gates"]) for f in sl["failing"]] or ["None."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round2", default="round2")
    ap.add_argument("--extra", nargs="*", default=[], help="what-if / refinement directories to rank alongside")
    ap.add_argument("--out", default=None, help="output stem (default <round2>/SHORTLIST.md and shortlist.json)")
    a = ap.parse_args()
    r2 = Path(a.round2)
    sl = build(r2, [Path(e) for e in a.extra])
    stem = Path(a.out) if a.out else r2
    (stem / "shortlist.json").write_text(json.dumps(sl, indent=1, default=str))
    (stem / "SHORTLIST.md").write_text(markdown(sl))
    print(markdown(sl))


if __name__ == "__main__":
    main()
