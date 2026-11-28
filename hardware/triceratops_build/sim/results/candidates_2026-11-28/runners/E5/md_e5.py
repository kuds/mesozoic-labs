"""E5_SUMMARY.md from the committed E5 result files (e5_choice_*, e5_tuning_*, e5_heldout_*). Every number in the
Markdown is read from those files; nothing is recomputed here except sums and maxima of the files' own figures."""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("/home/user/mesozoic-labs/hardware/triceratops_build/sim/results/candidates_2026-11-28/E5")
SIZES = ("0.60", "0.50")
CONDS = ("J1", "J2", "J3", "J4", "push_back_0.3", "push_back_0.4", "push_back_0.5", "push_right_0.3",
         "push_right_0.4", "push_right_0.5")
PARAMS = ("period_s", "duty", "body_pitch_deg", "lift_m", "max_reach_m")


def f(x, n=3):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if isinstance(x, bool):
        return "yes" if x else "no"
    return f"{x:.{n}f}"


def vals(v):
    return (f"period {v['period_s']:.3f} s, duty {v['duty']:.3f}, body pitch {v['body_pitch_deg']:+.2f} deg, "
            f"lift {1000 * v['lift_m']:.1f} mm, reach {1000 * v['max_reach_m']:.1f} mm")


def rule_lines(su):
    ch = su["checks"]
    fj = su["falls_J"]
    return [f"no falls in J1-J4: falls {fj} -> {'pass' if ch['no_falls_J1_J4'] else 'FAIL'}",
            f"J3 mean speed 2-20 s >= 0.225 m/s: {su['J3_speed_mps']:.4f} m/s -> "
            f"{'pass' if ch['J3_speed'] else 'FAIL'}",
            f"slow-gait episodes <= 1 of 10 per condition: {su['slow_gait']} -> "
            f"{'pass' if ch['slow_gait_at_most_1_per_condition'] else 'FAIL'}",
            "worst hind knee <= 1.0 of continuous in every episode: max per condition "
            + str({c: round(float(v), 3) if not isinstance(v, str) else v for c, v in su["worst_hind_knee"].items()})
            + f" -> {'pass' if ch['hind_knee_at_most_1'] else 'FAIL'}"]


def main():
    L = []
    L.append("# E5 retuned trot: summary (SIMULATION)")
    L.append("")
    L.append("28 Nov 2026. Pre-registration: `design/PREREG_CANDIDATES_2026_11_28.md` section 4 E5 (the lead's rule, "
             "quoted there, binds). Runner: the session scratch `candidates/E5/code/run_e5.py` (sha256 in every result "
             "file's `provenance`), calling only `sim/candidate_checks.py`'s frozen E5 code. MuJoCo 3.10.0, the "
             "build's env at 1 ms. Nothing measured; nothing adopted: any trot setting is the owner's decision.")
    L.append("")
    verdict = {}
    for sk in SIZES:
        hp = OUT / f"e5_heldout_{sk}.json"
        if hp.exists():
            h = json.loads(hp.read_text())["heldout"]
            verdict[sk] = h
    L.append("## Outcome of the pre-registered rule")
    L.append("")
    L.append("| Size | Chosen setting (tuning seeds 101-110 only) | Seeds 301-310 | Seeds 401-410 | Retune |")
    L.append("|---|---|---|---|---|")
    for sk, h in verdict.items():
        cells = []
        for sname in ("301-310", "401-410"):
            su = h["by_seed_set"][sname]["rule"]
            hk = max(float(x) for x in su["worst_hind_knee"].values())
            cells.append(f"{'pass' if su['SUCCEEDS'] else 'FAIL'}: falls J1-J4 {sum(su['falls_J'].values())}, J3 "
                         f"{su['J3_speed_mps']:.4f} m/s, slow-gait max {max(su['slow_gait'].values())} of 10, worst "
                         f"hind knee {hk:.3f}")
        L.append(f"| {'60 cm' if sk == '0.60' else '50 cm'} | {h['id']}: {vals(h['values'])} | {cells[0]} | "
                 f"{cells[1]} | **{'SUCCEEDS' if h['SUCCEEDS_both_sets'] else 'DOES NOT SUCCEED'}** |")
    L.append("")
    L.append("Rule (lead, 28 Nov 2026): on both held-out sets, no falls in J1-J4, J3 mean speed (2-20 s) >= 0.225 m/s, "
             "slow-gait episodes (`fore128_slow_gait.py`'s slow-and-hot rule, a fall counting as one) <= 1 of 10 in "
             "each of the 10 conditions, and the larger hind knee's RMS <= 1.0 of 6.75 N m (the worse of 2-20 s and "
             "10-20 s) in every episode. One controller (the trot), as the rule is written. Hind knee: RMS over "
             "continuous; speeds m/s; pitch deg, + nose down.")
    L.append("")
    hc = json.loads((OUT / "e5_harness_checks.json").read_text())
    rr = json.loads((OUT / "e5_run_record.json").read_text())
    L.append("## Harness checks (code, not hypotheses)")
    L.append("")
    c1 = hc["check1_E1_reproduced"]
    L.append(f"- Before stage 1, a smoke test on tuning seed 101; then, from the stored files: E5's frozen 60 cm "
             f"setting reproduces E1's {c1['episodes']} frozen-trot episodes (J1, J3, push back 0.5, push right 0.5; "
             f"seeds 101-104) with {c1['differing_fields_total']} differing record fields.")
    for sk, c in hc["check2_stage2_vs_stage1"].items():
        L.append(f"- {sk} m: stage 2 reran the {c['shared_episodes']} episodes it shares with stage 1: "
                 f"{len(c['mismatching_fields'])} mismatching fields (the episodes are deterministic).")
    c3 = hc["check3_push_applied_post_hoc"]
    L.append("- **Post hoc** (decided after the 60 cm held-out result was seen, because the chosen setting's push "
             "episodes looked almost alike; it changes no result): four held-out push episodes rerun in a separate "
             "folder equal the stored ones, and the push was applied: "
             + "; ".join(f"{r['size']} m {r['condition']} seed {r['seed']}: {r['applied_steps']} steps, "
                         f"{r['impulse_ns']} N s" for r in c3["rows"]) + ".")
    L.append(f"- Choice files unchanged since written: " + ", ".join(
        f"{s} {rr['choice_sha256_at_write'][s][0][:16]} at {rr['choice_sha256_at_write'][s][1]}, now "
        f"{rr['choice_sha256_now'][s][:16]}" for s in ("0.60", "0.50")) + ".")
    walls = {(st["size"], p["stage"]): (p["episodes"], p["wall_s"]) for st in rr["stages"] for p in st["pools"]}
    L.append("- Wall time on 2 workers (s; episodes): " + ", ".join(
        f"{sz} {sg} {w} ({n})" for (sz, sg), (n, w) in walls.items()) + ".")
    L.append("")
    for sk in SIZES:
        cp, tp, hp = OUT / f"e5_choice_{sk}.json", OUT / f"e5_tuning_{sk}.json", OUT / f"e5_heldout_{sk}.json"
        if not cp.exists():
            continue
        ch, tu = json.loads(cp.read_text()), json.loads(tp.read_text())
        name = "60 cm, the adopted model" if sk == "0.60" else "50 cm, build_xml(0.50, 'D0', 'adopted', layout='L1')"
        L.append(f"## {name}")
        L.append("")
        b = ch["budget"]
        L.append(f"Tuning (seeds 101-110 only): {b['distinct_settings']} distinct settings, "
                 f"{b['setting_runs_counted_over_stages']} setting runs over the stages (budget {b['max']}). "
                 f"Stage-2 determinism check: {len(ch['determinism_check_stage2_vs_stage1']['mismatching_fields'])} "
                 f"mismatching fields over {ch['determinism_check_stage2_vs_stage1']['shared_episodes']} episodes "
                 "shared with stage 1.")
        L.append("")
        L.append("Stage 1 (J1, J3, push back 0.5, push right 0.5; seeds 101-105; 20 episodes each), the six best and "
                 "the frozen setting (score: falls in J1+J3, slow-gait episodes, worst hind knee, J3 speed m/s):")
        L.append("")
        L.append("| Rank | Setting | Values | Falls J1+J3 | Slow-gait (of 20) | Worst hind knee | J3 speed, m/s |")
        L.append("|---|---|---|---|---|---|---|")
        rows = {r["id"]: r for r in tu["rows"] if r["stage"] == "s1"}
        for i, r in enumerate(ch["stage1_ranking"]):
            if i < 6 or r["id"].endswith("-s1-00"):
                sc = r["score"]
                L.append(f"| {i + 1} | {r['id']} | {vals(rows[r['id']])} | {sc[0]} | {sc[1]} | {f(sc[2])} | "
                         f"{f(-sc[3], 4)} |")
        L.append("")
        L.append("Stages 2 and 3 (all 10 conditions, seeds 101-110, 100 episodes each), the rule on the tuning seeds:")
        L.append("")
        L.append("| Stage | Setting | Values | Falls J1-J4 | J3 speed, m/s | Slow-gait total (max per condition) | "
                 "Worst hind knee | Failed checks | Meets rule |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in tu["rows"]:
            if r["stage"] == "s1":
                continue
            su = r["success"]
            hk = max(float(x) for x in su["worst_hind_knee"].values())
            L.append(f"| {r['stage']} | {r['id']} | {vals(r)} | {sum(su['falls_J'].values())} | "
                     f"{su['J3_speed_mps']:.4f} | {sum(su['slow_gait'].values())} ({max(su['slow_gait'].values())}) | "
                     f"{hk:.3f} | {r['failed_checks']} | {'yes' if su['SUCCEEDS'] else 'no'} |")
        L.append("")
        L.append(f"**Chosen** (`e5_choose`, written to `e5_choice_{sk}.json` before any held-out seed ran): "
                 f"{ch['chosen_id']} ({ch['chosen_stage']}), {vals(ch['chosen_values'])}; meets the rule on the "
                 f"tuning seeds: {'yes' if ch['chosen_meets_rule_on_tuning_seeds'] else 'no'}.")
        L.append("")
        if hp.exists():
            h = json.loads(hp.read_text())
            hj = h["heldout"]
            L.append(f"### Held-out, judged once: {'SUCCEEDS' if hj['SUCCEEDS_both_sets'] else 'DOES NOT SUCCEED'}")
            L.append("")
            for sname, per in hj["by_seed_set"].items():
                su = per["rule"]
                L.append(f"Seeds {sname}: **{'pass' if su['SUCCEEDS'] else 'FAIL'}**")
                for ln in rule_lines(su):
                    L.append(f"- {ln}")
                L.append("")
            L.append("Per condition (chosen setting; both held-out sets, 10 episodes each):")
            L.append("")
            L.append("| Condition | Set | Falls | Slow-gait | Slow | Mean speed 2-20 s, m/s | Worst hind knee | "
                     "Pitch 10-20 s, deg (+ nose down) | X4 peak / p99 rpm | X6 knee peak / p99 rpm | Slip ratio max |")
            L.append("|---|---|---|---|---|---|---|---|---|---|---|")
            for c in CONDS:
                for sname, per in hj["by_seed_set"].items():
                    t = per["reported"][c]
                    L.append(f"| {c} | {sname} | {t['falls']} | {t['slow_gait']} | {t['slow']} | "
                             f"{f(t['mean_speed_2_20_mps'], 4)} | {f(t['worst_hind_knee_max'])} | "
                             f"{f(t['pitch_10_20_mean_deg'], 2)} | {f(t['x4_peak_rpm_max'], 1)} / "
                             f"{f(t['x4_p99_rpm_max'], 1)} | {f(t['x6_peak_rpm_max'], 1)} / {f(t['x6_p99_rpm_max'], 1)}"
                             f" | {f(t['slip_ratio_max'])} |")
            L.append("")
            for sname, per in hj["by_seed_set"].items():
                d = per["drive_speeds"]
                L.append(f"Drive speeds, seeds {sname} (reported, not judged): X4 peak {d['x4_peak_rpm']:.1f} rpm, "
                         f"p99 {d['x4_p99_rpm']:.1f} rpm (83 rpm cap; peak line 91.3); X6 fore knees peak "
                         f"{d['x6_fore_knee_peak_rpm']:.1f}, p99 {d['x6_fore_knee_p99_rpm']:.1f} rpm (83 rpm; 108 "
                         "rpm beside).")
            L.append("")
            if "heldout_frozen" in h:
                hf = h["heldout_frozen"]
                L.append("### The frozen controller on the same held-out seeds (supporting, not judged)")
                L.append("")
                for sname, per in hf["by_seed_set"].items():
                    su = per["rule"]
                    L.append(f"Seeds {sname}: the rule would {'pass' if su['SUCCEEDS'] else 'fail'}")
                    for ln in rule_lines(su):
                        L.append(f"- {ln}")
                    L.append("")
    tail = Path(sys.argv[1]).read_text() if len(sys.argv) > 1 else ""
    Path(OUT / "E5_SUMMARY.md").write_text("\n".join(L) + "\n" + tail)
    print("\n".join(L))


if __name__ == "__main__":
    sys.exit(main())
