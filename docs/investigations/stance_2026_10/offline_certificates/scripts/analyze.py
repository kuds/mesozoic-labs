"""Tabulate the offline six-heading certificates and cross-check them (report-only)."""
import json, sys, collections
from pathlib import Path
S = Path(sys.argv[1]); O = S / "foot/next_signals/howclose"
out = {}
lines = []
for seed in (42, 52, 60):
    judge = json.loads((S / f"foot/symmetry/screen/runs/sym_lean_settled_s{seed}/judge_6291456.json").read_text())
    jrows = {r["seed"]: r for r in judge["panel"]["episodes_detail"]}
    per_opt = {}
    for opt in ("off", "armL"):
        p = O / f"cert_r9armL_s{seed}_6291456_opt{opt}/stance_heading_report.json"
        if not p.exists():
            continue
        rep = json.loads(p.read_text())
        res = rep["result"]
        heads = []
        bad_by_seed = collections.defaultdict(list)
        reason_counts = collections.Counter()
        for h in rep["headings"]:
            m = h["metrics"]; hr = h["result"] or {}
            st = (h["statue"] or {}).get("reference", {})
            n_hof = hr.get("hop_or_fall_episodes")
            heads.append({"yaw": h["yaw_deg"], "n_clean": m["n_clean"], "lcb": m["clean_lcb"], "full_horizon": m["full_horizon_fraction"],
                          "reward_mean": m["reward_mean"], "statue_reward_mean": st.get("mean_reward"), "hop_or_fall": n_hof,
                          "heading_passed": hr.get("passed"), "heading_failures": hr.get("failures")})
            for r in h["episode_evidence"]:
                if not r["clean"]:
                    bad_by_seed[r["seed"]].append((h["yaw_deg"], r["reasons"]))
                    for reason in r["reasons"]:
                        reason_counts[reason.split(" ")[0]] += 1
        zero = [h for h in rep["headings"] if h["yaw_deg"] == 0][0]
        diffs = []
        for r in zero["episode_evidence"]:
            j = jrows[r["seed"]]
            if j["clean"] != r["clean"] or abs(j["reward"] - r["reward"]) > 1e-6 or abs(j["max_sole_cop_fore_aft"] - r["max_sole_cop_fore_aft"]) > 1e-9:
                diffs.append((r["seed"], j["clean"], r["clean"], j["reward"], r["reward"]))
        per_opt[opt] = {"passed": rep["passed"], "n_clean_seeds": res["n_clean_seeds"], "n_seeds": res["n_seeds"],
                        "clean_seed_lcb": res["clean_seed_lcb"], "failures": rep["failures"], "task_sha256": rep["task_sha256"],
                        "gate_sha256": rep["gate_sha256"], "headings": heads,
                        "unclean_seeds": {str(k): v for k, v in sorted(bad_by_seed.items())},
                        "reason_counts_over_240": dict(reason_counts),
                        "zero_deg_vs_scratch_judge_diffs": diffs if opt == "off" else "n/a (scratch judge ran options off)"}
    # trajectory identity between opt off and armL: compare floor metrics per heading/seed
    if "off" in per_opt and "armL" in per_opt:
        a = json.loads((O / f"cert_r9armL_s{seed}_6291456_optoff/stance_heading_report.json").read_text())
        b = json.loads((O / f"cert_r9armL_s{seed}_6291456_optarmL/stance_heading_report.json").read_text())
        maxd = 0.0; label_diff = 0
        for ha, hb in zip(a["headings"], b["headings"]):
            for ra, rb in zip(ha["episode_evidence"], hb["episode_evidence"]):
                label_diff += ra["clean"] != rb["clean"]
                for k in ("max_sole_cop_fore_aft", "max_sole_tilt_deg", "max_sole_corner_lift_m", "min_sole_contacts", "length"):
                    if ra.get(k) is not None and rb.get(k) is not None:
                        maxd = max(maxd, abs(float(ra[k]) - float(rb[k])))
        out_cmp = {"max_abs_floor_metric_diff": maxd, "clean_label_differences": label_diff}
    else:
        out_cmp = None
    out[f"s{seed}"] = {"by_options": per_opt, "off_vs_armL_trajectory_check": out_cmp}
(O / "cert_summary.json").write_text(json.dumps(out, indent=1, default=str))
for s, v in out.items():
    for opt, r in v["by_options"].items():
        print(f"{s} opt={opt}: {'PASS' if r['passed'] else 'FAIL'} {r['n_clean_seeds']}/{r['n_seeds']} LCB {r['clean_seed_lcb']:.4f}; per heading clean "
              + ", ".join(f"{h['yaw']:+g}:{h['n_clean']}" for h in r["headings"]) + f"; reward 0deg {[h['reward_mean'] for h in r['headings'] if h['yaw']==0][0]:.1f} statue {[h['statue_reward_mean'] for h in r['headings'] if h['yaw']==0][0]}")
        print("   unclean seeds:", {k: [(y, [x.split(' ')[0] for x in rs]) for y, rs in vv] for k, vv in r["unclean_seeds"].items()})
        print("   failures:", r["failures"][:3], "| reasons", r["reason_counts_over_240"], "| 0deg diffs vs scratch:", r["zero_deg_vs_scratch_judge_diffs"])
    print("  off vs armL:", v["off_vs_armL_trajectory_check"])
