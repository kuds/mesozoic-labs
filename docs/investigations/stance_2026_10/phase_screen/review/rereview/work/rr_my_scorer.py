"""Re-reviewer's quick independent scorer check (P2) on two synthetic PS-shaped roots.

The roots are built with the scorer tests' own helpers (tools/tests/test_score_bs.py: Screen), then edited so that
each continuation carries its own slope at S.  My check reads the roots' files directly (markers.json, check.json,
lost.json, the run's policy_probe.npz) and computes, from the criteria text alone (THE PRIMARY OUTCOME, THE TEST,
THE INFORMATIVENESS CONDITION, THE READING, THE SECONDARIES, REPORT ONLY slope at E against S), the line, the filled
and as-counted p, the control rise, each secondary's filled p / verdict / guard note, and the slope E - S read with its
counts.  It compares them with score_bs.score(root) (which writes nothing).  Writes only under
<PS>/review/rereview/synth/.
"""
import importlib.util
import itertools
import json
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
PS = Path("/home/user/work/phase_screen")
OUT = PS / "review" / "rereview" / "synth"
spec = importlib.util.spec_from_file_location("test_score_bs_rr", str(PS / "tools/tests/test_score_bs.py"))
T = importlib.util.module_from_spec(spec)
spec.loader.exec_module(T)
SB = T.SB

# ---- constants from the criteria text (not from the scorer); asserted against the scorer below
S_STEP, E_STEP = 2_097_152, 3_538_944
PRIMARY_STEPS = tuple(3_473_408 + 8192 * i for i in range(8))
B_S = Fraction(135930855, 1073741824)
CTRL, PHASE = "control", "critic_phase"
SEEDS = (2901, 2911, 2921, 2931)
GATE = Fraction(1, 5)
assert tuple(SB.PRIMARY_STEPS) == PRIMARY_STEPS and Fraction(SB.BASELINE_S) == B_S and SB.SEEDS == SEEDS
assert (SB.CONTROL, SB.INTERVENTION) == (CTRL, PHASE)
INF_NEG, INF_POS = "-inf", "+inf"


def name(arm, seed):
    return f"{arm}_{seed}"


def classes(root):
    lost = {}
    lp = root / "lost.json"
    if lp.exists():
        lost = json.loads(lp.read_text())["lost"]
    out = {}
    for seed in SEEDS:
        for arm in (CTRL, PHASE):
            n = name(arm, seed)
            if n in lost:
                out[n] = "LOST"
                continue
            ck = json.loads((root / "runs" / n / "check.json").read_text())
            assert ck["passed"] is True
            out[n] = "COUNTED" if (ck["stop_reason"] == "end" and ck["T"] == E_STEP) else "STOPPED"
    return out


def markers(root, n):
    p = root / "reads" / n / "markers.json"
    return json.loads(p.read_text()) if p.exists() else None


def primary(m):
    steps, vals = m["probe"]["steps"], m["probe"]["rollout_mu14_mean"]
    got = []
    for st in PRIMARY_STEPS:
        idx = [i for i, s in enumerate(steps) if s == st]
        if len(idx) != 1 or not np.isfinite(vals[idx[0]]):
            return None
        got.append(Fraction(vals[idx[0]]))
    return sum(got, Fraction(0)) / 8


def loop_row(m, step):
    rows = [r for r in m.get("loop", []) if r.get("steps") == step]
    return rows[0] if len(rows) == 1 else None


def key(v):
    """THE FILL as a comparable pair (infinite part, finite part)."""
    if v == INF_NEG:
        return (-1, Fraction(0))
    if v == INF_POS:
        return (1, Fraction(0))
    return (0, Fraction(v))


def perm_count(hyp, oth):
    """#relabelings (choose len(hyp) of all as the hypothesis arm) whose hypothesis-arm sum >= observed, and total."""
    vals = [key(v) for v in hyp] + [key(v) for v in oth]
    k = len(hyp)
    obs = (sum(v[0] for v in vals[:k]), sum((v[1] for v in vals[:k]), Fraction(0)))
    count = total = 0
    for idx in itertools.combinations(range(len(vals)), k):
        s = (sum(vals[i][0] for i in idx), sum((vals[i][1] for i in idx), Fraction(0)))
        total += 1
        count += s >= obs
    return count, total


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if n == 0:
        return None
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def my_score(root):
    cl = classes(root)
    prim = {}
    for n, c in cl.items():
        m = markers(root, n) if c == "COUNTED" else None
        prim[n] = primary(m) if m is not None else None
    ctrl_read = [prim[name(CTRL, s)] for s in SEEDS if prim[name(CTRL, s)] is not None]
    med = median(ctrl_read)
    rise = None if med is None else med - B_S
    informative = rise is not None and rise >= GATE
    hyp = [prim[name(CTRL, s)] if prim[name(CTRL, s)] is not None else INF_NEG for s in SEEDS]
    oth = [prim[name(PHASE, s)] if prim[name(PHASE, s)] is not None else INF_POS for s in SEEDS]
    cf, tf = perm_count(hyp, oth)
    ac = perm_count([v for v in hyp if v != INF_NEG], [v for v in oth if v != INF_POS])
    line = 1 if not informative else (2 if Fraction(cf, tf) <= Fraction(1, 20) else 3)
    # secondaries
    sec = {}
    for keyname, hyp_arm, getter in (
            ("SEC-CMD", CTRL, lambda m, n: m["ledger"]["direct"]["per_evaluation"].get(str(E_STEP), {}).get("cmd_l")),
            ("SEC-MARGIN", PHASE, lambda m, n: (loop_row(m, E_STEP) or {}).get("OWN", {}).get("P", {}).get("b_star")),
            ("SEC-MECH", CTRL, lambda m, n: (loop_row(m, E_STEP) or {}).get("critic_slope_obs6", {}).get(
                "ONPOLICY", {}).get("effective", {}).get("dV_dq3_per_deg")),
            ("SEC-TAIL2", PHASE, lambda m, n: tail2(root, n))):
        vals = {}
        for n, c in cl.items():
            v = None
            if c == "COUNTED" and prim[n] is not None:
                m = markers(root, n)
                v = getter(m, n)
                if v is not None and not np.isfinite(float(v)):
                    v = None
            vals[n] = v
        oth_arm = CTRL if hyp_arm == PHASE else PHASE
        h = [vals[name(hyp_arm, s)] if vals[name(hyp_arm, s)] is not None else INF_NEG for s in SEEDS]
        o = [vals[name(oth_arm, s)] if vals[name(oth_arm, s)] is not None else INF_POS for s in SEEDS]
        c_, t_ = perm_count(h, o)
        any_missing = any(v is None for v in vals.values())
        sec[keyname] = {"count": c_, "total": t_, "supported": Fraction(c_, t_) <= Fraction(1, 20),
                        "guard_missing_note": keyname == "SEC-TAIL2" and any_missing}
    # slope at E against S (report only)
    slope = {}
    cnt = {CTRL: [0, 0, 0], PHASE: [0, 0, 0]}
    for n, c in cl.items():
        m = markers(root, n)
        r = None
        if m is not None:
            rs, re_ = loop_row(m, S_STEP), loop_row(m, E_STEP)
            if rs is not None and re_ is not None:
                s_ = rs["critic_slope_obs6"]["ONPOLICY"]["effective"]["dV_dq3_per_deg"]
                e_ = re_["critic_slope_obs6"]["ONPOLICY"]["effective"]["dV_dq3_per_deg"]
                r = {"S": s_, "E": e_, "E_minus_S": e_ - s_, "E_le_S": e_ <= s_, "E_le_0": e_ <= 0.0}
                arm = CTRL if n.startswith(CTRL) else PHASE
                cnt[arm][0] += r["E_le_S"]
                cnt[arm][1] += r["E_le_0"]
                cnt[arm][2] += 1
        slope[n] = r
    return {"line": line, "p_filled": (cf, tf), "p_as_counted": ac, "control_rise": rise, "sec": sec,
            "slope": slope, "slope_counts": {a: {"E_le_S": v[0], "E_le_0": v[1], "of": v[2]} for a, v in cnt.items()}}


def tail2(root, n):
    ck = json.loads((root / "runs" / n / "check.json").read_text())
    path = Path(ck["sources"]["policy_probe.npz"][0])
    with np.load(path) as z:
        steps = list(z["update_timesteps"])
        mu = z["rollout_mu_mean"]
    vals = []
    for st in PRIMARY_STEPS:
        v = mu[steps.index(st), 13]
        if not np.isfinite(v):
            return None
        vals.append(Fraction(float(v)))
    return sum(vals, Fraction(0)) / 8


def set_slope_at(root, n, step, value):
    p = root / "reads" / n / "markers.json"
    m = json.loads(p.read_text())
    row = [r for r in m["loop"] if r["steps"] == step][0]
    row["critic_slope_obs6"]["ONPOLICY"]["effective"]["dV_dq3_per_deg"] = value
    if m["arm"] == CTRL:                                   # control: effective = raw (the stock critic)
        row["critic_slope_obs6"]["ONPOLICY"]["raw"]["dV_dq3_per_deg"] = value
    p.write_text(json.dumps(m, indent=1))


def compare(root, label):
    res, lines, code = SB.score(root)
    mine = my_score(root)
    diffs = []

    def eq(what, a, b):
        if a != b:
            diffs.append(f"{what}: scorer {a!r} != mine {b!r}")

    eq("line", res["line"], mine["line"])
    eq("p_filled", (res["p_filled"]["count"], res["p_filled"]["total"]), mine["p_filled"])
    eq("p_as_counted", (res["p_as_counted"]["count"], res["p_as_counted"]["total"]), mine["p_as_counted"])
    want_add = []
    if mine["line"] == 1 and Fraction(*mine["p_filled"]) <= Fraction(1, 20):
        want_add = [SB.ADDITION_NOT_INFORMATIVE]
    if mine["line"] == 3 and Fraction(*mine["p_as_counted"]) <= Fraction(1, 20):
        want_add = [SB.ADDITION_AS_COUNTED]
    eq("additions", res["additions"], want_add)
    eq("control_rise", Fraction(res["control_rise"]) if res["control_rise"] is not None else None,
       Fraction(float(mine["control_rise"])) if mine["control_rise"] is not None else None)
    for s in res["secondaries"]:
        mk = mine["sec"][s["key"]]
        eq(f'{s["key"]} p', (s["p_filled"]["count"], s["p_filled"]["total"]), (mk["count"], mk["total"]))
        eq(f'{s["key"]} supported', bool(s["supported"]), mk["supported"])
        eq(f'{s["key"]} guard MISSING note', SB.SEC_GUARD_MISSING in s["notes"], mk["guard_missing_note"])
    for n, r in mine["slope"].items():
        got = res["report_only"].get(n, {}).get("slope_end_vs_start") if n in res["report_only"] else None
        eq(f"slope {n}", got, r)
    eq("slope counts", res["slope_end_vs_start"], mine["slope_counts"])
    text = "\n".join(lines)
    print(f"{label}: scorer exit {code} line {res['line']} p_filled {res['p_filled']['count']}/"
          f"{res['p_filled']['total']} p_as_counted {res['p_as_counted']['count']}/{res['p_as_counted']['total']} "
          f"rise {res['control_rise']!r}; secondaries "
          + "; ".join(f"{s['key']} {s['p_filled']['count']}/70 {s['verdict']} notes {s['notes']}"
                      for s in res["secondaries"]))
    print(f"   slope counts {res['slope_end_vs_start']}; per run "
          + "; ".join(f"{n} {None if r is None else round(r['E_minus_S'], 5)}" for n, r in mine["slope"].items()))
    print(f"   LINE_TEXT[{res['line']}] in text: {SB.LINE_TEXT[res['line']] in text}")
    print(f"   DIFFERENCES: {len(diffs)}")
    for d in diffs:
        print("     " + d)
    return len(diffs)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    f32 = T.f32
    # ROOT A: control climbs; one phase continuation LOST (fill; guard MISSING note); as-counted p small
    ra = OUT / "rootA"
    assert not ra.exists()
    sa = T.Screen(ra)
    ctrl = [f32(float(B_S) + x) for x in (0.25, 0.45, 0.21, 0.60)]
    ph = [f32(float(B_S) + x) for x in (0.05, 0.03, 0.10, 0.02)]
    per = {(CTRL, 2901): dict(mech=0.05, tail2=0.03125, cmd=0.75, b=0.5),
           (CTRL, 2911): dict(mech=0.03, tail2=0.015625, cmd=0.875, b=0.25),
           (CTRL, 2921): dict(mech=0.021, tail2=0.0234375, cmd=0.5, b=0.75),
           (CTRL, 2931): dict(mech=0.07, tail2=0.0, cmd=0.9, b=0.125),
           (PHASE, 2901): dict(mech=0.0, tail2=0.0625, cmd=0.25, b=1.25),
           (PHASE, 2911): dict(mech=-0.004, tail2=0.046875, cmd=0.125, b=1.5),
           (PHASE, 2921): dict(mech=0.022, tail2=0.0546875, cmd=0.375, b=1.0)}
    for i, seed in enumerate(SEEDS):
        sa.run(CTRL, seed, ctrl[i], **per[(CTRL, seed)])
        if seed != 2931:
            sa.run(PHASE, seed, ph[i], **per[(PHASE, seed)])
    sa.lost(name(PHASE, 2931))
    for n, v in ((name(CTRL, 2901), 0.02142920531332493), (name(PHASE, 2901), 0.02142920531332493),
                 (name(PHASE, 2911), 0.02142920531332493), (name(PHASE, 2921), 0.022),
                 (name(CTRL, 2911), 0.02142920531332493), (name(CTRL, 2921), 0.0214), (name(CTRL, 2931), 0.03)):
        set_slope_at(ra, n, S_STEP, v)
    # ROOT B: all 8 present; a cross-arm tie on the primary; slope ties at S; line 2 or 3 by the numbers
    rb = OUT / "rootB"
    assert not rb.exists()
    sb_ = T.Screen(rb)
    ctrl = [f32(float(B_S) + x) for x in (0.40, 0.22, 0.31, 0.50)]
    ph = [f32(float(B_S) + x) for x in (0.22, 0.10, 0.35, 0.05)]       # phase 2901 ties control 2911
    perb = {}
    mechs = {CTRL: (0.04, 0.02142920531332493, 0.06, 0.03), PHASE: (0.02142920531332493, -0.01, 0.0, 0.05)}
    t2 = {CTRL: (0.0078125, 0.015625, 0.0234375, 0.03125), PHASE: (0.0078125, 0.0390625, 0.0, 0.0625)}
    for arm in (CTRL, PHASE):
        for i, seed in enumerate(SEEDS):
            perb[(arm, seed)] = dict(mech=mechs[arm][i], tail2=t2[arm][i], cmd=0.5 + 0.1 * i if arm == CTRL else 0.6,
                                     b=1.0 + 0.125 * i)
    for i, seed in enumerate(SEEDS):
        sb_.run(CTRL, seed, ctrl[i], **perb[(CTRL, seed)])
        sb_.run(PHASE, seed, ph[i], **perb[(PHASE, seed)])
    for arm in (CTRL, PHASE):
        for seed in SEEDS:
            set_slope_at(rb, name(arm, seed), S_STEP, 0.02142920531332493)
    total = compare(ra, "rootA") + compare(rb, "rootB")
    print(f"TOTAL DIFFERENCES: {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
