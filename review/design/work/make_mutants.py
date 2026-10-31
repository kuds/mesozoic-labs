"""Mutants of score_bs.py (text-to-code mapping), written under <D>/work/mutants/ (the real file is only read)."""
from pathlib import Path
SRC = Path("/home/user/work/phase_screen/tools/score_bs.py").read_text()
OUT = Path("/home/user/work/phase_screen/review/design/work/mutants")
MUT = {
    "m01_median_is_mean": ("return statistics.median(values) if values else None",
                           "return statistics.mean(values) if values else None"),
    "m02_primary_rows_shift": ("PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K, 0, -1))",
                               "PRIMARY_STEPS = tuple(C.E - C.ROLLOUT * k for k in range(PRIMARY_K + 1, 1, -1))"),
    "m03_tail2_col12": ("ACT_TAIL2 = 13 ", "ACT_TAIL2 = 12 "),
    "m04_mech_raw": ('v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg")',
                     'v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "raw", "dV_dq3_per_deg")'),
    "m05_mech_settled": ('v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg")',
                         'v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "settled", "dV_dq3_per_deg")'),
    "m06_mech_own_states": ('v = _get(rows[0], "critic_slope_obs6", "ONPOLICY", "effective", "dV_dq3_per_deg")',
                            'v = _get(rows[0], "critic_slope_obs6", "OWN", "effective", "dV_dq3_per_deg")'),
    "m07_tail2_efficacy": ('{"key": "SEC-TAIL2", "family": GUARD,', '{"key": "SEC-TAIL2", "family": EFFICACY,'),
    "m08_tail2_direction": ('{"key": "SEC-TAIL2", "family": GUARD, "higher": INTERVENTION,',
                            '{"key": "SEC-TAIL2", "family": GUARD, "higher": CONTROL,'),
    "m09_sec_fill_swapped": ('vals[c["arm"]].append(NEG_INF if c["arm"] == higher else POS_INF)',
                             'vals[c["arm"]].append(POS_INF if c["arm"] == higher else NEG_INF)'),
    "m10_bonferroni_4": ("filled[0] <= ALPHA / N_EFFICACY", "filled[0] <= ALPHA / 4"),
    "m11_alpha_4_70": ("ALPHA = Fraction(1, 20)", "ALPHA = Fraction(4, 70)"),
    "m12_ties_for": ("count = sum(1 for s in sums.values() if s >= obs)",
                     "count = sum(1 for s in sums.values() if s > obs) + 1"),
    "m13_gate_on_mean": ("rise = med[CONTROL] - base if med[CONTROL] is not None else None",
                         "rise = (sum(counted[CONTROL]) / len(counted[CONTROL]) - base) if counted[CONTROL] else None"),
    "m14_sec_ignores_unreadable": ('    if primary_cause is not None:\n        return None, f"UNREADABLE ({primary_cause})"\n',
                                   ''),
    "m15_baseline_rounded": ("BASELINE_S = 0.1265954738482833", "BASELINE_S = 0.12660"),
    "m16_margin_last_saved": ('''        rows = loop_rows_at(m, C.E)
        if len(rows) != 1:
            return None, f"{len(rows)} M-LOOP rows at E''',
                              '''        rows = loop_rows_at(m, C.SAVED_STEPS[-1])
        if len(rows) != 1:
            return None, f"{len(rows)} M-LOOP rows at E'''),
    "m17_cmd_max": ('v, what = e.get("cmd_l"), "CMD_L at E"', 'v, what = e.get("cmd_l_max"), "CMD_L at E"'),
    "m18_primary_fill_swapped": ("INF_CONTROL, INF_PHASE = NEG_INF, POS_INF", "INF_CONTROL, INF_PHASE = POS_INF, NEG_INF"),
    "m19_gate_strict": ("informative = rise is not None and rise >= RISE_MIN",
                        "informative = rise is not None and rise > RISE_MIN"),
    "m20_line1_note_missing": ("    if line == 1:\n        notes.append(SEC_UNDER_LINE1)\n", ""),
    "m21_mech_direction": ('{"key": "SEC-MECH", "family": EFFICACY, "higher": CONTROL,',
                           '{"key": "SEC-MECH", "family": EFFICACY, "higher": INTERVENTION,'),
    "m22_as_counted_addition_any_line": ("if line == 3 and as_counted is not None and as_counted[0] <= ALPHA:",
                                         "if line != 1 and as_counted is not None and as_counted[0] <= ALPHA:"),
}
for name, (a, b) in MUT.items():
    n = SRC.count(a)
    if n != 1:
        print(f"{name}: pattern found {n} times: NOT WRITTEN")
        continue
    (OUT / f"{name}.py").write_text(SRC.replace(a, b))
    print(f"{name}: written")
