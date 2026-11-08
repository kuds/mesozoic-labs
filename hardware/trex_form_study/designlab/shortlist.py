"""Round-two scorecard and shortlist: which bodies are most likely to achieve the target behaviors.

Method (stated in ROUND2_REVIEW.md; every input is a recorded screen result, nothing is recomputed):

1. Gates.  A candidate is out if it fails any of: every round-two clearance gate; every drive within its
   planning limits for the sizing walks; no drive over its limits in the target turns (0.35 rad/s walking,
   0.5 rad/s in place); a person's torso visible at 1.5-3 m with the head level; at least 20 min runtime
   (75 % usable energy); the nominal reference walk's CoP within 10 mm of the pad; the carpet screen's
   gates when it has run.
2. Margins, not pass/fail.  Each behavior gets a 0-1 score from how far the governing quantity is from its
   limit, ``min(1, max(0, headroom / reference))`` with the reference headroom stated per metric.
3. Weights from the owner's priorities (P1, P2, P4, P10-P15): natural walking and turning first, then carpet
   and robustness, then following, then runtime and cost (``WEIGHTS``).  The scorecard shows every
   category so the weighting can be argued with.
4. Stress test.  Every drive ratio is re-evaluated with the drives 15 % weaker and the robot 10 % heavier
   (ratios x 1.10 / 0.85); a candidate whose lead depends on optimistic numbers drops.  Reported as the
   worst stressed ratio and the stressed total.
5. Shortlist.  The best-scoring passing candidate at each height, then the best remaining candidate of the
   other body family, so the next verification step compares real alternatives rather than near-twins.
   Looks are the owner's call and are not scored.

Run from hardware/trex_form_study:  python -m designlab.shortlist --round2 round2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .round2 import VARIANT_LABEL, candidate_ids, variant_key

WEIGHTS = dict(natural_walking=0.20, turning=0.20, carpet=0.10, robustness=0.15, following=0.12, endurance=0.10,
               head_and_tail=0.05, packaging=0.05, cost=0.03)
STRESS = 1.10 / 0.85  # 10 % heavier, drives 15 % weaker
GROUP_LABEL = {"hip_roll": "hip roll", "hip_pitch": "hip pitch", "hip_yaw": "thigh twist", "knee": "knee",
               "ankle_a": "ankle A", "ankle_b": "ankle B", "neck": "neck", "tail": "tail"}


def _clip(x, ref):
    return float(max(0.0, min(1.0, x / ref))) if ref > 0 else 0.0


def _worst_margin(doc) -> tuple[str, float]:
    best = ("", 0.0)
    for g, m in doc["margins"].items():
        r = max(m["cont_ratio"], m["peak_ratio"], m["speed_ratio"])
        if r > best[1]:
            best = (g, r)
    return best


def _turn_worst(tu: dict | None) -> float | None:
    if not tu:
        return None
    v = tu.get("worst_ratio_targets")
    return float(v) if v is not None else None


def gates(doc: dict, rv: dict) -> dict:
    cl = (rv.get("clearance") or {}).get("pass", {})
    se = rv.get("sensors") or {}
    tu = rv.get("turning") or {}
    ca = rv.get("carpet") or {}
    g, r = _worst_margin(doc)
    out = dict(
        clearance=bool(cl.get("all")),
        drives_sizing_walks=r <= 1.0 + 1e-9,
        turning_targets=(tu.get("targets_pass") if tu else None),
        person_in_view=bool((se.get("torso_follow") or {}).get("home", {}).get("ok")),
        runtime_20min=doc["runtime"]["minutes"] >= 20.0,
        reference_walk_cop=bool(doc.get("walk_feasible")),
        carpet=(ca.get("pass") if ca else None),
    )
    out["all"] = all(v is not False for v in out.values())
    return out


def scores(doc: dict, rv: dict) -> dict:
    """0-1 per behavior, with the quantities behind each."""
    env = rv.get("envelope") or {}
    se = rv.get("sensors") or {}
    tu = rv.get("turning") or {}
    ca = rv.get("carpet") or {}
    cl = rv.get("clearance") or {}
    H = doc["H"]
    s, why = {}, {}
    # natural walking: drive headroom at Froude 0.05 and in the brisk sizing walk, nominal CoP margin
    # an animal-speed walk whose prescribed CoP leaves the pad is inconclusive and is not scored
    fw = (env.get("fast_walk_worst") or [None, None])[1] if env.get("fast_walk_reference_cop_on_pad") else None
    _, wr = _worst_margin(doc)
    cop = 1000 * doc["gaits"]["nominal"]["min_cop_margin_single"]
    parts = [_clip(1.0 - wr, 0.30), _clip(cop + 10.0, 30.0)]
    if fw is not None:
        parts.append(_clip(1.0 - fw, 0.30))
    s["natural_walking"] = sum(parts) / len(parts)
    why["natural_walking"] = dict(fast_walk_worst_ratio=fw, sizing_worst_ratio=wr, nominal_cop_margin_mm=cop)
    # turning: drive headroom in the target turns, thigh-twist range used, friction needed
    if tu:
        tr = _turn_worst(tu)
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
        why["turning"] = dict(worst_ratio=tr, thigh_twist_range_used=rng, mu_needed=mu)
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
    # robustness: one-step push recovery (Froude-normalised) and one-foot stopping
    pf = env.get("push_max_dv_ms")
    st = env.get("stop_max_one_foot_decel_ms2")
    parts = []
    if pf is not None:
        parts.append(_clip(pf / (9.81 * H) ** 0.5, 0.35))
    if st is not None:
        parts.append(_clip(st, 1.5))
    s["robustness"] = sum(parts) / len(parts) if parts else None
    why["robustness"] = dict(push_dv_ms=pf, push_froude=None if pf is None else pf / (9.81 * H) ** 0.5,
                             one_foot_stop_ms2=st)
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
    # endurance: runtime and the hottest drive over a session
    rt = doc["runtime"]["minutes"]
    sh = (env.get("session_worst") or [None, None])[1]
    parts = [_clip(rt - 20.0, 25.0)]
    if sh is not None:
        parts.append(_clip(1.0 - sh, 0.30))
    s["endurance"] = sum(parts) / len(parts)
    why["endurance"] = dict(runtime_min=rt, session_hottest_ratio=sh)
    # head and tail drives
    nr = max(doc["margins"]["neck"][k] for k in ("cont_ratio", "peak_ratio", "speed_ratio"))
    tr2 = max(doc["margins"]["tail"][k] for k in ("cont_ratio", "peak_ratio", "speed_ratio"))
    s["head_and_tail"] = (_clip(1.0 - nr, 0.30) + _clip(1.0 - tr2, 0.30)) / 2
    why["head_and_tail"] = dict(neck_ratio=nr, tail_ratio=tr2)
    # packaging: tightest joint-range margin and closest near-contact while walking
    mg = cl.get("joint_range_margins") or {}
    tight = min((v for r in mg.values() for v in (r.get("margin_min"), r.get("margin_max")) if v is not None), default=None)
    s["packaging"] = _clip((tight or 0.0) - 0.10, 0.20) if tight is not None else None
    why["packaging"] = dict(tightest_joint_margin_rad=tight)
    # cost: nominal cost against the $10-20k prototype budget (P4)
    cn = doc["cost"]["nominal"]
    s["cost"] = _clip(25000.0 - cn, 10000.0)
    why["cost"] = dict(nominal_usd=cn, high_usd=doc["cost"]["high"])
    return dict(scores=s, why=why)


def total(sc: dict) -> float:
    num = sum(WEIGHTS[k] * v for k, v in sc.items() if v is not None)
    den = sum(WEIGHTS[k] for k, v in sc.items() if v is not None)
    return num / den if den else 0.0


def stress(doc: dict, rv: dict) -> dict:
    """Worst drive ratio with drives 15 % weaker and the robot 10 % heavier."""
    _, wr = _worst_margin(doc)
    env = rv.get("envelope") or {}
    fw = (env.get("fast_walk_worst") or [None, None])[1] if env.get("fast_walk_reference_cop_on_pad") else None
    tu = _turn_worst(rv.get("turning") or {})
    vals = [v for v in (wr, fw, tu) if v is not None]
    worst = max(vals) * STRESS if vals else None
    return dict(worst_stressed_ratio=worst, sizing_walks_hold=wr * STRESS <= 1.0,
                fast_walk_holds=None if fw is None else fw * STRESS <= 1.0)


def build(r2: Path) -> dict:
    D = {p.stem: json.loads(p.read_text()) for p in sorted((r2 / "candidates").glob("*.json"))}
    R = json.loads((r2 / "review" / "review_index.json").read_text())
    rows = []
    for cid in candidate_ids():
        d, rv = D.get(cid), R.get(cid, {})
        if not d or not d.get("ok"):
            continue
        gt = gates(d, rv)
        sc = scores(d, rv)
        rows.append(dict(id=cid, variant=VARIANT_LABEL.get(variant_key(cid), cid), height=d["candidate"]["height"],
                         family=d["candidate"]["family"], gates=gt, scores=sc["scores"], why=sc["why"],
                         total=total(sc["scores"]), stress=stress(d, rv)))
    passing = sorted([r for r in rows if r["gates"]["all"]], key=lambda r: -r["total"])
    picks = []
    for h in (60, 75):
        best = next((r for r in passing if r["height"] == h), None)
        if best:
            picks.append(dict(id=best["id"], why=f"best-scoring passing body at {h} cm"))
    fams = {r["family"] for r in passing if r["id"] in {p["id"] for p in picks}}
    other = next((r for r in passing if r["id"] not in {p["id"] for p in picks} and r["family"] not in fams), None)
    if other is None:
        other = next((r for r in passing if r["id"] not in {p["id"] for p in picks}), None)
    if other:
        picks.append(dict(id=other["id"], why="best remaining passing body" + (" of the other family" if other["family"]
                                                                             not in fams else "")))
    return dict(weights=WEIGHTS, stress_factor=STRESS, rows=rows, ranking=[r["id"] for r in passing],
                failing=[dict(id=r["id"], gates={k: v for k, v in r["gates"].items() if v is False}) for r in rows
                         if not r["gates"]["all"]], picks=picks)


def markdown(sl: dict) -> str:
    cats = list(WEIGHTS)
    head = ["Rank", "Candidate", "Variant", "Total"] + [c.replace("_", " ") for c in cats] + ["Worst drive ratio, stressed"]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    by = {r["id"]: r for r in sl["rows"]}
    for i, cid in enumerate(sl["ranking"], 1):
        r = by[cid]
        cells = [str(i), f"`{cid}`", r["variant"], f"{r['total']:.2f}"]
        cells += ["–" if r["scores"].get(c) is None else f"{r['scores'][c]:.2f}" for c in cats]
        ws = r["stress"]["worst_stressed_ratio"]
        cells.append("–" if ws is None else f"{ws:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    out = ["# Round-two scorecard and shortlist", "", "Generated by `python -m designlab.shortlist`. Method: module "
           "docstring and ROUND2_REVIEW.md. Weights: " + ", ".join(f"{k.replace('_', ' ')} {v:.2f}" for k, v in WEIGHTS.items())
           + ". Stress: drive ratios x %.2f (drives 15 %% weaker, robot 10 %% heavier)." % STRESS, "",
           "## Picks", ""]
    out += [f"{k + 1}. `{p['id']}`: {p['why']}." for k, p in enumerate(sl["picks"])]
    out += ["", "## Passing candidates, ranked", "", "\n".join(lines), "", "## Failing a gate", ""]
    out += [f"* `{f['id']}`: " + ", ".join(k.replace("_", " ") for k in f["gates"]) for f in sl["failing"]] or ["None."]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--round2", default="round2")
    a = ap.parse_args()
    r2 = Path(a.round2)
    sl = build(r2)
    (r2 / "shortlist.json").write_text(json.dumps(sl, indent=1, default=str))
    (r2 / "SHORTLIST.md").write_text(markdown(sl))
    print(markdown(sl))


if __name__ == "__main__":
    main()
