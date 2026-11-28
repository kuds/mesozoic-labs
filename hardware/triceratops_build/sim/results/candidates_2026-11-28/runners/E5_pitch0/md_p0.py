"""E5_PITCH0_SUMMARY.md from the pitch-0 branch's result files (e5p0_*) and E5's (for the pitched comparison). Every
number is read from those files; only sums, maxima and counts of the files' own figures are computed here."""
import json
from pathlib import Path

OUT = Path("/home/user/mesozoic-labs/hardware/triceratops_build/sim/results/candidates_2026-11-28/E5")
P0 = OUT / "pitch0"
STAGE = Path("/tmp/claude-0/-home-user-mesozoic-labs/ccd22965-ed7a-50aa-9e57-30e96c9784a3/scratchpad/candidates/"
             "E5_pitch0/staging")
SIZES = [s for s in ("0.60", "0.50") if (P0 / f"e5p0_heldout_{s}.json").exists()]
SETS = ("301-310", "401-410", "501-510", "601-610")


def vals(v):
    return (f"period {v['period_s']:.3f} s, duty {v['duty']:.3f}, body pitch {v['body_pitch_deg']:+.2f} deg, "
            f"lift {1000 * v['lift_m']:.1f} mm, reach {1000 * v['max_reach_m']:.1f} mm")


def cell(su):
    hk = max(float(x) for x in su["worst_hind_knee"].values())
    return (f"{'pass' if su['SUCCEEDS'] else 'FAIL'}: falls {sum(su['falls_J'].values())}, J3 "
            f"{su['J3_speed_mps']:.4f} m/s, slow-gait max {max(su['slow_gait'].values())}, knee {hk:.3f}")


def failed(su):
    return [k for k, v in su["checks"].items() if not v]


L = ["# E5 pitch-0 branch: summary (SIMULATION)", "",
     "28 Nov 2026. **Changed after a result:** this branch was planned after E5's pitched held-out results and "
     "review finding E5-R1. Plan: `E5_PITCH0_PLAN.md` (sha256 recorded before any episode; unchanged, see "
     "`e5p0_harness_checks.json`). It runs prereg section 3 item 9's alternative: plan.json's stage-1 points with "
     "body pitch set to 0, as a contingency for the lead's open ruling. It does not make that ruling. Nothing "
     "measured; nothing adopted.", ""]
hc = json.loads((P0 / "e5p0_harness_checks.json").read_text())
rr = json.loads((P0 / "e5p0_run_record.json").read_text())

L += ["## Outcome (plan item 7: SUCCEEDS only if all four held-out sets pass)", "",
      "| Size | Chosen pitch-0 setting (seeds 101-110 only) | " + " | ".join(SETS) + " | Lead's sets 301-410 | "
      "Rerun sets 501-610 | Pitch-0 retune |", "|---|---|" + "---|" * len(SETS) + "---|---|---|"]
held = {}
for sk in SIZES:
    h = json.loads((P0 / f"e5p0_heldout_{sk}.json").read_text())
    held[sk] = h
    hd = h["heldout"]
    cells = [cell(hd["by_seed_set"][s]["rule"]) for s in SETS]
    L.append(f"| {'60 cm' if sk == '0.60' else '50 cm'} | {hd['id']}: {vals(hd['values'])} | " + " | ".join(cells)
             + f" | {'pass' if hd['reading_lead_sets_301_310_401_410'] else 'FAIL'} | "
             f"{'pass' if hd['reading_rerun_sets_501_510_601_610'] else 'FAIL'} | "
             f"**{'SUCCEEDS' if hd['SUCCEEDS_all_four_sets'] else 'FAILS'}** |")
L += ["", "Per set: falls in J1-J4; J3 mean speed 2-20 s (rule >= 0.225 m/s); the largest slow-gait count in any "
      "condition (rule <= 1 of 10); the worst hind knee RMS over its 6.75 N m continuous rating (rule <= 1.0). "
      "One controller (the trot), as the lead's rule is written.", ""]
for sk in SIZES:
    hd = held[sk]["heldout"]
    for s in SETS:
        f_ = failed(hd["by_seed_set"][s]["rule"])
        if f_:
            L.append(f"- {sk} m, seeds {s}: failed checks {f_}; slow-gait per condition "
                     f"{hd['by_seed_set'][s]['rule']['slow_gait']}; worst hind knee per condition "
                     f"{ {c: round(float(v), 3) for c, v in hd['by_seed_set'][s]['rule']['worst_hind_knee'].items()} }.")
L.append("")

L += ["## Tuning (seeds 101-110 only)", ""]
for sk in SIZES:
    ch = json.loads((P0 / f"e5p0_choice_{sk}.json").read_text())
    tu = json.loads((P0 / f"e5p0_tuning_{sk}.json").read_text())
    b = ch["budget"]
    L.append(f"**{sk} m.** {b['distinct_settings']} distinct settings, {b['setting_runs_counted_over_stages']} "
             f"setting runs (budget {b['max']}; plan item 5 caps this branch at 47). Stage 3 "
             f"{'ran around ' + str(ch['stage3_centre']) if ch['stage3_ids'] else 'not run: a stage-2 setting met the rule'}."
             f" Chosen: {ch['chosen_id']} ({ch['chosen_stage']}), "
             f"{('source point ' + ch['chosen_id'].replace('-p0-', '-')) if '-s1-' in ch['chosen_id'] else 'a stage-3 neighbour'}"
             f"; meets the rule on 101-110: "
             f"{'yes' if ch['chosen_meets_rule_on_tuning_seeds'] else 'no'}.")
    L.append("")
    L.append("| Rank | Setting | Values (pitch 0) | Falls J1+J3 | Slow-gait (of 20) | Worst hind knee | J3 speed, m/s |")
    L.append("|---|---|---|---|---|---|---|")
    rows = {r["id"]: r for r in tu["rows"] if r["stage"] == "s1"}
    for i, r in enumerate(ch["stage1_ranking"]):
        if i < 6 or r["id"].endswith("-s1-00"):
            sc = r["score"]
            L.append(f"| {i + 1} | {r['id']} | {vals(rows[r['id']])} | {sc[0]} | {sc[1]} | {float(sc[2]):.3f} | "
                     f"{-float(sc[3]):.4f} |")
    L.append("")
    L.append("| Stage | Setting | Falls J1-J4 | J3 speed, m/s | Slow-gait total (max) | Worst hind knee | Meets rule |")
    L.append("|---|---|---|---|---|---|---|")
    for r in tu["rows"]:
        if r["stage"] == "s1":
            continue
        su = r["success"]
        hk = max(float(x) for x in su["worst_hind_knee"].values())
        L.append(f"| {r['stage']} | {r['id']} | {sum(su['falls_J'].values())} | {su['J3_speed_mps']:.4f} | "
                 f"{sum(su['slow_gait'].values())} ({max(su['slow_gait'].values())}) | {hk:.3f} | "
                 f"{'yes' if su['SUCCEEDS'] else 'no'} |")
    L.append("")

L += ["## The frozen controller on the rerun seeds (supporting, not judged)", ""]
for sk in SIZES:
    fz = held[sk]["heldout_frozen"]
    for s in ("501-510", "601-610"):
        su = fz["by_seed_set"][s]["rule"]
        L.append(f"- {sk} m, seeds {s}: {'would pass' if su['SUCCEEDS'] else 'would fail'} ({cell(su)}); "
                 f"slow-gait per condition {su['slow_gait']}.")
L.append("")

L += ["## Reported, not judged", ""]
for sk in SIZES:
    hd = held[sk]["heldout"]
    tr, pk, p99, pr = [], [], [], []
    for s in SETS:
        ds = hd["by_seed_set"][s]["drive_speeds"]
        pk.append(ds["x4_peak_rpm"])
        p99.append(ds["x4_p99_rpm"])
        for c, v in hd["by_seed_set"][s]["reported"].items():
            pr.append(v["pitch_10_20_mean_deg"])
            for e in v["per_seed"]:
                if ((e["speed_2_8"] is not None and e["speed_2_8"] < 0.225)
                        or (e["hind_knee_2_8"] is not None and e["hind_knee_2_8"] >= 0.9)):
                    tr.append(f"{c} seed {e['seed']} (2-8 s: {e['speed_2_8']:.4f} m/s, knee {e['hind_knee_2_8']:.3f}; "
                              f"10-20 s {e['speed_10_20']:.4f} m/s)")
    L.append(f"- {sk} m: X4 drive speed peak {max(pk):.1f} rpm, 99th percentile {max(p99):.1f} rpm over the four "
             f"sets (83 rpm cap; reported). Condition-mean 10-20 s pitch {min(pr):+.2f} to {max(pr):+.2f} deg "
             f"(+ nose down; the setpoint is 0). Episodes slow or hot over 2-8 s (speed under 0.225 m/s or hind knee "
             f">= 0.9; review E5-R2, context only): {len(tr)} of 400" + (": " + "; ".join(tr) if tr else "") + ".")
for sk in SIZES:
    hd = held[sk]["heldout"]
    sl = []
    for s in SETS:
        for c, v in hd["by_seed_set"][s]["reported"].items():
            for e in v["per_seed"]:
                if e["slow"]:
                    sl.append(f"seeds {s} {c} seed {e['seed']} (10-20 s {e['speed_10_20']:.4f} m/s, worst hind knee "
                              f"{e['worst_hind_knee']:.3f}, slip ratio {e['slip_max']:.3f}, pitch {e['pitch_10_20']:+.2f} deg)")
    L.append(f"- {sk} m: \"slow\" episodes (10-20 s speed under 0.225 m/s; the slow-gait rule also needs the hind knee "
             f"at or over 1.0, so these are not slow-gait episodes): {len(sl)} of 400" + (": " + "; ".join(sl) if sl else "") + ".")
L.append("")

L += ["## Harness checks", ""]
for sk, v in hc["check1_setting00_reproduces_E5_stage1"].items():
    L.append(f"- {sk} m: setting 00 (the frozen controller) reproduces E5's stored stage-1 episodes: "
             f"{v['differing_fields_total']} differing fields over {v['episodes']} episodes.")
for sk, v in hc["check2_stage2_vs_stage1"].items():
    L.append(f"- {sk} m: stage 2 reran its {v['shared_episodes']} shared stage-1 episodes: "
             f"{len(v['mismatching_fields'])} mismatching fields.")
for sk, v in hc["check3_all_pitch_0_and_seeds_allowed"].items():
    L.append(f"- {sk} m: {v['episode_files']} episode files, every one at body pitch 0 with an allowed seed: "
             f"{'yes' if v['pass'] else 'NO ' + str(v['bad'][:5])}.")
c4 = hc["check4_files_unchanged"]
L.append("- Choice files unchanged since written: " + ", ".join(
    f"{s} {'yes' if v['pass'] else 'NO'} ({v['now'][:16]})" for s, v in c4["choice"].items())
         + f"; plan unchanged: {'yes' if c4['plan']['pass'] else 'NO'} ({c4['plan']['now'][:16]}).")
L.append("- Choice written before any held-out episode file existed: " + ", ".join(
    f"{s} {'yes' if v['pass'] else 'NO'}" for s, v in hc["check5_choice_before_heldout"].items()) + ".")
walls = [(st["size"], p["stage"], p["episodes"], p["wall_s"]) for st in rr["stages"] for p in st["pools"]]
L.append("- Wall time on 2 workers (s; episodes): " + ", ".join(f"{a} {b} {d} ({c})" for a, b, c, d in walls) + ".")
L.append("")
L += ["## Against E5's pitched choice (supporting)", ""]
for sk in SIZES:
    e5 = json.loads((OUT / f"e5_heldout_{sk}.json").read_text())["heldout"]
    hd = held[sk]["heldout"]
    a = [max(float(x) for x in e5["by_seed_set"][s]["rule"]["worst_hind_knee"].values()) for s in ("301-310", "401-410")]
    b = [max(float(x) for x in hd["by_seed_set"][s]["rule"]["worst_hind_knee"].values()) for s in ("301-310", "401-410")]
    L.append(f"- {sk} m, seeds 301-310 / 401-410: worst hind knee {a[0]:.3f} / {a[1]:.3f} with {e5['id']} "
             f"(pitch {e5['values']['body_pitch_deg']:+.2f} deg) against {b[0]:.3f} / {b[1]:.3f} with {hd['id']} "
             f"(pitch 0); J3 speed {e5['by_seed_set']['301-310']['rule']['J3_speed_mps']:.4f} / "
             f"{e5['by_seed_set']['401-410']['rule']['J3_speed_mps']:.4f} against "
             f"{hd['by_seed_set']['301-310']['rule']['J3_speed_mps']:.4f} / "
             f"{hd['by_seed_set']['401-410']['rule']['J3_speed_mps']:.4f} m/s.")
L.append("")
(STAGE / "E5_PITCH0_SUMMARY.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
