"""PS V3 (files only): the reader's outputs on the V2 runs and on the parent.

  1. the reference read (<PS>/validate/v3/parent_s42/markers.json): the parent's G, c0, own14, REF G and the critic's
     on-policy tail slope at 2,097,152 and 3,145,728 equal ew_r10.jsonl and eb_realized.json (read_bs.REPRO) bit for
     bit (max |diff| 0.0, relative 0.0), the verifier's 3-decimal values within 5e-4;
  2. control at S (each V2 run's start copy): the same values as the parent's at 2,097,152, exactly;
  3. critic_phase at S: the actor's readings (OWN G, c0, own14, G_H, G_L, b*, REF G, REF c0) equal control's exactly;
     the critic's tail slope with the phase fed (effective), with the slots zero (raw) and with every state settled
     equal the parent's exactly on every state set (ONPOLICY, OWN, REF; the start transform zeroes the phase columns);
     the record of what was fed (REF k = 200, OWN k 500-1000, ONPOLICY each state's own counter with its pre-settle
     count) and the actor's invariance to the fed phase (0.0);
  4. at the V2 end pair (2,359,296, no update; the statistics moved): critic_phase's readings equal control's exactly
     (same seed: the same env side, the same statistics outside the slots, zero phase columns), both arms reported;
  5. the probe markers (M-PROBE) of the two arms equal (bank actions and values: the phase fed, zero columns), and
     critic_phase's markers name the probe's phase record.

    PY -I -B v3_check.py <PS>/validate        -> exit 0 when every expectation holds
"""
import importlib.util
import json
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"
_spec = importlib.util.spec_from_file_location("read_bs", str(TOOLS / "read_bs.py"))
RB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RB)
C = RB.C
V = Path(sys.argv[1])
ok = True
out = []


def expect(cond, text):
    global ok
    ok &= bool(cond)
    out.append(f"  [{'ok' if cond else 'FAIL'}] {text}")


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


ACTOR = [("OWN", "P", "G"), ("OWN", "P", "c0"), ("OWN", "own14_median"), ("OWN", "H", "G"), ("OWN", "L", "G"),
         ("OWN", "P", "b_star"), ("OWN", "b_star_L"), ("REF", "P", "G"), ("REF", "P", "c0")]
SETS = ("ONPOLICY", "OWN", "REF")


def slope(row, s, form):
    return g(row, "critic_slope_obs6", s, form, "dV_dq3_per_deg")


ref = json.loads((V / "v3" / "parent_s42" / "markers.json").read_text())
out.append("1. reference read (parent, F2 pairs):")
for st, r in sorted(ref["reproduction"].items()):
    expect(r["max_abs_diff_ew"] == 0.0 and r["rel_diff_eb"] == 0.0 and r.get("max_abs_diff_printed_3dp", 1) <= 5e-4,
           f"{st}: ew G/c0/own14/REF G max |diff| {r['max_abs_diff_ew']}, eb slope rel {r['rel_diff_eb']}, 3-dp "
           f"verifier values {r.get('max_abs_diff_printed_3dp'):.2g}")
parent_S = next(row for row in ref["loop"] if row["steps"] == C.S)
want = RB.REPRO[C.S]
for seed in (9401, 9411):
    mc = json.loads((V / "v2" / f"control_{seed}" / "read" / "markers.json").read_text())
    mp = json.loads((V / "v2" / f"critic_phase_{seed}" / "read" / "markers.json").read_text())
    lc, lp = {r["steps"]: r for r in mc["loop"]}, {r["steps"]: r for r in mp["loop"]}
    out.append(f"seed {seed}: control pairs {sorted(lc)}, critic_phase pairs {sorted(lp)}")
    c0, p0 = lc[C.S], lp[C.S]
    vals = {"G": g(c0, "OWN", "P", "G"), "c0": g(c0, "OWN", "P", "c0"), "own14": g(c0, "OWN", "own14_median"),
            "REF_G": g(c0, "REF", "P", "G"), "slope": slope(c0, "ONPOLICY", "raw")}
    expect(vals == {"G": want["ew_own"]["G"], "c0": want["ew_own"]["c0"], "own14": want["ew_own"]["own14"],
                    "REF_G": want["ew_ref_G"], "slope": want["eb_dV_dq3_per_deg"]},
           f"2. control at S reproduces the parent's 2,097,152 values exactly: {vals}")
    expect(all(g(c0, *k) == g(parent_S, *k) for k in ACTOR) and all(
        slope(c0, s, f) == slope(parent_S, s, f) for s in SETS for f in ("raw", "effective")),
        "2. control at S = the reference read's 2,097,152 row on every actor reading and slope")
    expect(all(g(p0, *k) == g(c0, *k) for k in ACTOR),
           "3. critic_phase at S: the actor's readings (G, c0, own14, G_H, G_L, b*_P, b*_L, REF G, REF c0) = control's")
    sl = {s: {f: slope(p0, s, f) for f in ("effective", "raw", "settled")} for s in SETS}
    expect(all(sl[s][f] == slope(c0, s, "raw") for s in SETS for f in ("effective", "raw", "settled")),
           f"3. critic_phase at S: the tail slope with the phase fed / slots zero / all settled = the parent's on "
           f"every set: {sl}")
    ph = p0.get("phase") or {}
    fed = ph.get("fed") or {}
    expect(ph.get("start_transform") is True and ph.get("actor_phase_invariance_max_abs") == 0.0
           and ph.get("value_phase_cols_max_abs_used") == 0.0 and ph.get("value_phase_cols_max_abs_read", 0) > 0
           and g(fed, "REF", "k_min") == g(fed, "REF", "k_max") == C.SETTLE_STEPS
           and g(fed, "OWN", "k_min") == 500 and g(fed, "OWN", "k_max") == 1000
           and g(fed, "ONPOLICY", "k_min") == 0 and (g(fed, "ONPOLICY", "n_presettle") or 0) > 0,
           f"3. critic_phase at S: the start transform applied (columns read {ph.get('value_phase_cols_max_abs_read')}"
           f" -> used {ph.get('value_phase_cols_max_abs_used')}), the actor invariant to the fed phase "
           f"({ph.get('actor_phase_invariance_max_abs')}), fed {fed}")
    end = max(lc)
    ce, pe = lc[end], lp[end]
    expect(end == C.VALIDATE_UNTIL and max(lp) == end, f"4. both arms read at their end pair {end}")
    expect(all(g(pe, *k) == g(ce, *k) for k in ACTOR) and all(
        slope(pe, s, f) == slope(ce, s, "raw") for s in SETS for f in ("effective", "raw", "settled")),
        f"4. at {end}: critic_phase's readings = control's exactly (G {g(pe, 'OWN', 'P', 'G')}, c0 "
        f"{g(pe, 'OWN', 'P', 'c0')}, on-policy slope eff {slope(pe, 'ONPOLICY', 'effective')} / control "
        f"{slope(ce, 'ONPOLICY', 'raw')})")
    phe = pe.get("phase") or {}
    expect(phe.get("start_transform") is False and phe.get("value_phase_cols_max_abs_read") == 0.0,
           "4. critic_phase's end pair read without the start transform (its saved columns already zero)")
    pm_c, pm_p = mc["probe"], mp["probe"]
    keys = ("steps", "tail3_statue_settled", "trailing8_mean", "trailing8_first_above")
    expect(all(pm_c.get(k) == pm_p.get(k) for k in keys) and all(
        json.dumps(pm_c.get(k)) == json.dumps(pm_p.get(k)) for k in pm_c if "value" in k),
        "5. M-PROBE identical between the arms (bank tail_3 commands, trailing means, the critic's bank values)")
    expect((mp.get("probe_phase") or {}).get("installed") is True and (mc.get("probe_phase") or {}).get(
        "installed") is False, "5. the markers name the probe's phase record (critic_phase fed, control stock)")
    out.append(f"   critic_phase {seed} at S: OWN G {g(p0, 'OWN', 'P', 'G')}, c0 {g(p0, 'OWN', 'P', 'c0')}, own14 "
               f"{g(p0, 'OWN', 'own14_median')}, REF G {g(p0, 'REF', 'P', 'G')}; on-policy slope (phase fed) "
               f"{slope(p0, 'ONPOLICY', 'effective')} per deg; OWN {slope(p0, 'OWN', 'effective')}; REF "
               f"{slope(p0, 'REF', 'effective')}")
print("\n".join(out))
print("V3 CHECK: " + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
