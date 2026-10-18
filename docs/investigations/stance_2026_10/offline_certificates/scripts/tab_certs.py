"""offline_certificates.txt: every offline six-heading certificate run in this directory (report only)."""
import json, sys, collections
from pathlib import Path
O = Path(sys.argv[1])
dirs = sorted([d for d in O.iterdir() if d.is_dir() and (d / "run_summary.json").exists() and not d.name.startswith("smoke")])
L = []
L.append("OFFLINE SIX-HEADING CERTIFICATES (stance_quality_heading/v1), main 2e734e29, report only, no training.")
L.append("Function: environments/shared/reporting/stance_report.py:2089 build_stance_heading_report (the call the notebook's")
L.append("end-of-stage certificate makes, environments/shared/reporting/stage_artifacts.py:357-369), stage config from")
L.append("load_stage_config(tyrannosaurus_rex/anatomical, 1) = configs/tyrannosaurus_rex/anatomical/stance.toml.")
L.append("opt off = stance.toml [env] as committed; opt armL = plus lateral_lean_weight 0.25, lateral_lean_tolerance_m 0.015,")
L.append("leg_home_pose_reference 'settled' (environments/tyrannosaurus_rex/base_env.py:273-275). Runner: scripts/run_cert.py.")
L.append("")
for d in dirs:
    s = json.loads((d / "run_summary.json").read_text())
    rep = json.loads((d / "stance_heading_report.json").read_text())
    unclean = collections.defaultdict(set); reasons = collections.Counter()
    for h in rep["headings"]:
        for r in h["episode_evidence"]:
            if not r["clean"]:
                unclean[r["seed"]].add(h["yaw_deg"])
                for x in r["reasons"]:
                    reasons[x.split(":")[0].split(" ")[0]] += 1
    L.append(f"== {d.name}")
    L.append(f"   model {Path(s['model']).name} sha256 {s['model_sha256']}")
    L.append(f"   vecnorm {Path(s['vecnorm']).name} sha256 {s['vecnorm_sha256']}")
    L.append(f"   options {s['options']}; seeds {rep['seeds'][0]}-{rep['seeds'][-1]}; task_sha256 {s['task_sha256']}; gate_sha256 {s['gate_sha256']}; {s['seconds']:.0f} s")
    L.append(f"   VERDICT {'PASS' if s['passed'] else 'FAIL'}: {s['n_clean_seeds']}/{s['n_seeds']} seeds clean at all six headings, LCB {s['clean_seed_lcb']:.4f} (bar 0.80)")
    for h in rep["headings"]:
        hr = h["result"]
        L.append(f"     heading {h['yaw_deg']:+6.0f}: clean {hr['n_clean']}/40 LCB {hr['clean_lcb']:.4f} full-horizon {hr['n_full_horizon']}/40 "
                 f"hop_or_fall {hr['hop_or_fall_episodes']} reward {hr['mean_reward']:.1f} statue {hr['statue_mean_reward']:.1f} "
                 f"ratio {hr['mean_reward']/hr['statue_mean_reward']:.3f} heading {'PASS' if hr['passed'] else 'FAIL'}")
    if unclean:
        L.append("   unclean seeds (headings): " + "; ".join(f"{k} ({len(v)}/6)" for k, v in sorted(unclean.items())))
        L.append("   episode-reason counts over 240: " + ", ".join(f"{k} {v}" for k, v in reasons.most_common()))
    if s["failures"]:
        L.append("   failures: " + " | ".join(s["failures"][:4]) + (" ..." if len(s["failures"]) > 4 else ""))
    L.append("")
(O / "offline_certificates.txt").write_text("\n".join(L) + "\n")
print("\n".join(L))
