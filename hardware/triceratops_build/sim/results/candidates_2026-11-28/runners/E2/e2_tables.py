"""Print Markdown tables from E2's result files (sim/results/candidates_2026-11-28/E2/e2a_summary.json and
e2b_summary.json), for the short summary. Reads only; computes nothing new."""
import json
import sys
from pathlib import Path

RES = Path("/home/user/mesozoic-labs/hardware/triceratops_build/sim/results/candidates_2026-11-28/E2")


def e2a():
    s = json.loads((RES / "e2a_summary.json").read_text())
    print("| Transmission (eta) | Placement | r | C1 worst, C (where) | C2 worst (where) | speed p99 / peak worst, rpm"
          " | C1 | C2 | speed | PASSES | policies passing all three (both blocks) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in s["verdicts"].items():
        if not v["judged"]:
            continue
        ok = [p.split("-")[1] for p, x in v["per_policy_all_pass"].items() if x]
        print(f"| {v['transmission']} ({v['eta']:g}) | {v['placement']} | {v['r']:g} | {v['C1_worst_c']:.1f} "
              f"({v['C1_worst_at']}) | {v['C2_worst']:.3f} ({v['C2_worst_at']}) | {v['speed_p99_worst_rpm']:.1f} / "
              f"{v['speed_peak_worst_rpm']:.1f} | {'pass' if v['C1_all_pass'] else 'FAIL'} | "
              f"{'pass' if v['C2_all_pass'] else 'FAIL'} | {'pass' if v['speed_all_pass'] else 'FAIL'} | "
              f"**{'PASSES' if v['PASSES'] else 'fails'}** | {', '.join(ok) or 'none'} |")
    print()
    print("Supporting (rod at 0.90):")
    for k, v in s["verdicts"].items():
        if v["judged"]:
            continue
        print(f"  {k}: C1 {v['C1_worst_c']:.1f} C2 {v['C2_worst']:.3f} speed {v['speed_all_pass']} "
              f"heat per policy {v['per_policy_heat_pass']}")
    print()
    for k, v in s["per_policy_block"].items():
        t = v["today"]
        print(f"  {k}: method {v['method']} repro {v['reproduction_rel_max']} today C1 {t['C1_c']:.2f} "
              f"({t['C1_joint']}) C2 {t['C2']:.4f} ({t['C2_joint']}) minutes {v['minutes']:.2f}")


def e2b():
    s = json.loads((RES / "e2b_summary.json").read_text())
    print("unstable at start:", s["unstable_runs_at_start"])
    for v, e in s["variants"].items():
        print(f"\n### {v} ({e['role']}, {e['physics_dt']:g} s, vs {e['baselines']}): COMPATIBLE = "
              f"{e['verdict']['COMPATIBLE']}, per policy {e['verdict']['per_policy']}")
        print("| Policy | Baseline | Condition | new falls | unstable | d speed m/s | worst drive ratio (variant / "
              "baseline) | d ratio | pass |")
        print("|---|---|---|---|---|---|---|---|---|")
        for pol, bb in e["comparisons"].items():
            for b, cc in bb.items():
                for cond, c in cc.items():
                    sv, sb = c.get("speed_variant"), c.get("speed_baseline")
                    rv, rb = c.get("worst_ratio_variant"), c.get("worst_ratio_baseline")
                    print(f"| {pol} | {b} | {cond} | {c['new_falls']} | {c['unstable']} | "
                          f"{'' if sv is None else f'{sv - sb:+.4f}'} | "
                          f"{'' if rv is None else f'{rv:.3f} ({c.get(chr(119)+'orst_joint_variant')}) / {rb:.3f}'} | "
                          f"{'' if rv is None else f'{rv - rb:+.3f}'} | {c['pass']} |")
        for extra in ("supporting_0.25ms_run", "supporting_rod_eta0.9"):
            if extra in e:
                print(f"{extra}: {e[extra]['verdict']}")
    print("\nrun stats (falls, cap share, deflection, drive rpm):")
    for rid, pp in s["run_stats"].items():
        for pol, cc in pp.items():
            for cond, st in cc.items():
                extra = ""
                if "at_cap_share_max" in st:
                    extra += f" cap max {st['at_cap_share_max']:.4f}"
                if "deflection_max_mrad" in st:
                    extra += f" defl max {max(st['deflection_max_mrad'].values()):.2f} mrad"
                p99 = max(st["drive_rpm_p99"].values())
                pk = max(st["drive_rpm_peak"].values())
                print(f"  {rid} {pol} {cond}: falls {st['falls']} unst {st['unstable']} v {st['speed_mean_all']:.4f} "
                      f"rpm p99 {p99:.1f} peak {pk:.1f}{extra} delay {st['delay_steps']} wall {st['wall_s_mean']:.1f}")


if __name__ == "__main__":
    {"a": e2a, "b": e2b}[sys.argv[1]]()
