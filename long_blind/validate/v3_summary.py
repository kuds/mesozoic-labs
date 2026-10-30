"""LB V3 summary (files only): the reader's full reads of the four V2 runs and the parent reference read.

    PY -I -B v3_summary.py <LB dir>

  - the reference read's reproduction at 2,097,152 and 3,145,728 (max_abs_diff_ew, rel_diff_eb, the 3-decimal ones):
    the parent's G, c0, own14, REF G and critic slope against early_warning / early_bias / the verifier /
    warning_margin;
  - each V2 read: markers written, the M-LOOP at the start copy equal to the parent's 2,097,152 row (G, c0, own14,
    REF G, on-policy raw critic slope; the critic_blind read's EFFECTIVE slope 0 by construction), the M5 reading (no
    flag, C6 counting failures 0), the probe markers' row count, the loop schedule;
  - the loop cache's cpu_seconds per pair (the reader's cost).
Exit 0 when the reproduction holds (ew exact, eb relative 0.0, printed values within 5e-4) and every read has its
markers.
"""
import json
import sys
from pathlib import Path

LB = Path(sys.argv[1])
ok = True


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


ref = json.loads((LB / "validate" / "v3" / "parent_s42" / "markers.json").read_text())
m5r = ref["ledger"]["m5"]
print(f"reference read: version {ref['version']}; pairs {ref.get('reference_pairs')}; probe rows "
      f"{ref['probe']['n_rows']}; M5 C6 passed {m5r['c6_passed']} counting failures {m5r['n_c6_counting_failures']}")
for steps, r in ref["reproduction"].items():
    print(f"  REPRO {steps}: max_abs_diff_ew {r['max_abs_diff_ew']}; rel_diff_eb {r.get('rel_diff_eb')}; "
          f"max_abs_diff_printed_3dp {r.get('max_abs_diff_printed_3dp')}")
    for k in ("ew_own_G", "ew_own_c0", "ew_own14", "ew_ref_G", "eb_dV_dq3_per_deg"):
        print(f"      {k}: reader {r[k][0]!r} target {r[k][1]!r}")
    good = (r["max_abs_diff_ew"] == 0.0 and r.get("rel_diff_eb") == 0.0
            and (r.get("max_abs_diff_printed_3dp") or 0) <= 5e-4)
    ok &= good
    print(f"      reproduction {'exact' if good else 'NOT EXACT'}")
parent_S = next(row for row in ref["loop"] if row["steps"] == 2_097_152)
for arm in ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm"):
    p = LB / "validate" / "v2" / f"{arm}_9301" / "read" / "markers.json"
    if not p.is_file():
        print(f"{arm}: NO MARKERS")
        ok = False
        continue
    m = json.loads(p.read_text())
    s0 = next(row for row in m["loop"] if row["steps"] == 2_097_152)
    keys = (("OWN", "P", "G"), ("OWN", "P", "c0"), ("OWN", "own14_median"), ("REF", "P", "G"),
            ("critic_slope_obs6", "ONPOLICY", "raw", "dV_dq3_per_deg"))
    same = all(g(s0, *k) == g(parent_S, *k) for k in keys)
    eff = g(s0, "critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg")
    m5 = m["ledger"]["m5"]
    print(f"{arm}: T {m['T']:,}; loop schedule {m.get('loop_schedule')}; pairs read {[r['steps'] for r in m['loop']]}; "
          f"S row = parent's 2,097,152 in G, c0, own14, REF G, raw slope: {same}; effective on-policy slope {eff}; "
          f"probe rows {m['probe']['n_rows']}; M5 flags any {m5['any_flagged']} tail {m5['tail_flagged']}, C6 counting "
          f"failures {m5['n_c6_counting_failures']}; resumed_from {m.get('resumed_from')}")
    ok &= same and m["probe"]["n_rows"] == 31
    if arm == "critic_blind_tail3":
        ok &= eff == 0.0
    if arm == "frozen_obs_norm":
        end = m["loop"][-1]
        print(f"    frozen: end pair t3 normaliser mean {end['t3_norm_mean_deg']} sd {end['t3_norm_sd_deg']} = start's "
              f"{s0['t3_norm_mean_deg']} {s0['t3_norm_sd_deg']}")
cache = []
for f in sorted((LB / "reads" / "_cache").glob("*.json")):
    row = json.loads(f.read_text())
    cache.append((row.get("num_timesteps"), row.get("critic_form"), row.get("cpu_seconds"), f.name[:40]))
print(f"loop cache: {len(cache)} entries; cpu_seconds per pair {[c[2] for c in cache]}")
for c in cache:
    print("   ", c)
print("V3:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
