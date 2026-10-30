"""BS2 V2 summary (files only; nothing simulated): for each no-update run of one seed, the proof of no optimizer step
(parameter and Adam digests and n_updates equal at start and end, train() reached and returned), the arm's change on
EVERY rollout read from arm_evidence.jsonl row by row (independently of bs_hooks.evidence_problems), CHECK's result
(check.json) and the probe's rows (seeded rows at or before S, window rows, ent_coef / learning_rate against the
parent's F2 probe at the same steps, read here again).

    PY -I -B v2_summary.py V2_DIR SEED
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "bs_common", str(Path(__file__).resolve().parents[1] / "tools" / "bs_common.py"))
C = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(C)

root, seed = Path(sys.argv[1]), int(sys.argv[2])
ARMS = ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm")
with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
    P = {k: np.asarray(z[k]) for k in ("update_timesteps", "ent_coef", "learning_rate")}
pw = {int(u): i for i, u in enumerate(P["update_timesteps"].tolist())}
ok_all = True
for arm in ARMS:
    run = root / f"{arm}_{seed}"
    m = json.loads((run / "run_meta.json").read_text())
    c = json.loads((run / "check.json").read_text())
    ev = [json.loads(x) for x in (run / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
    d0, d1 = m["digest_start"], m["digest_end"]
    T = m["T"]
    n_exp = C.expected_update_rows(C.S, T)
    p = []

    def want(cond, msg, p=p):
        if not cond:
            p.append(msg)
    want(m["status"] == "done" and m["exit"] == 0 and m["stop_reason"] == "end", "status")
    want(T == C.VALIDATE_UNTIL and m["S"] == C.S and m["E"] == C.E, "S/E/T")
    want(d0["params"] == d1["params"] and d0["adam"] == d1["adam"], "params / Adam digest changed")
    want(d0["n_updates"] == d1["n_updates"], "n_updates changed")
    want(d0["num_timesteps"] == C.S and d1["num_timesteps"] == T, "num_timesteps")
    want(m["train_calls"] == n_exp, f"train_calls {m['train_calls']} != {n_exp}")
    want(m["no_update_check"]["pass"] is True, "no_update_check")
    want(m["code_seal_after"] == "unchanged" and m["modules_outside_code"] == [], "seal / modules")
    want(m["arm_state_end"] == "in force", "arm state end")
    want(m["ent_coef_after_load"] == 0.005, f"ent_coef after load {m['ent_coef_after_load']}")
    want(c["passed"] is True and c["failures"] == [], f"CHECK {c['failures']}")
    want([e["num_timesteps"] for e in ev] == [C.S + C.ROLLOUT * k for k in range(1, n_exp + 1)], "evidence steps")
    # the arm's signature on every rollout (row by row)
    blind = arm == "critic_blind_tail3"
    lam1 = arm == "gae_lambda_1"
    frozen = arm == "frozen_obs_norm"
    per = {
        "critic sees obs 6 (dv_predict_values > 0)": sum(e["dv_predict_values"] > 0 for e in ev),
        "critic blind (all 4 value paths exactly 0)": sum(all(e[k] == 0.0 for k in (
            "dv_predict_values", "dv_forward", "dv_evaluate_actions", "dv_mirrored")) for e in ev),
        "actor sees obs 6 (dmu_actor > 0)": sum(e["dmu_actor"] > 0 for e in ev),
        "advantages = GAE(0.95) (<= 1e-4 x scale)": sum(e["adv_err_vs_gae095"] <= 1e-4 * e["adv_scale"] for e in ev),
        "advantages = MC - V (<= 1e-4 x scale)": sum(e["adv_err_vs_mc"] <= 1e-4 * e["adv_scale"] for e in ev),
        "obs_rms equal to the start's": sum(bool(e["obs_rms_equal_start"]) for e in ev),
        "buffer values = run's critic": sum(e["buffer_values_vs_critic"] <= 1e-5 * e["value_scale"] for e in ev),
        "buffer values != raw value MLP": sum(e["buffer_values_vs_raw_value_mlp"] > 1e-5 * e["value_scale"]
                                              for e in ev),
    }
    n = len(ev)
    want(per["critic sees obs 6 (dv_predict_values > 0)"] == (0 if blind else n), "critic dependence")
    want(per["critic blind (all 4 value paths exactly 0)"] == (n if blind else 0), "blind paths")
    want(per["actor sees obs 6 (dmu_actor > 0)"] == n, "actor")
    want(per["advantages = GAE(0.95) (<= 1e-4 x scale)"] == (0 if lam1 else n), "GAE 0.95")
    want(per["advantages = MC - V (<= 1e-4 x scale)"] == (n if lam1 else 0), "MC")
    want(per["obs_rms equal to the start's"] == (n if frozen else 0), "obs_rms")
    want(per["buffer values = run's critic"] == n, "buffer values")
    want(per["buffer values != raw value MLP"] == (n if blind else 0), "raw value MLP")
    # the probe: seeded rows and the window rows' schedule against the parent's
    with np.load(run / "01_stance" / "policy_probe.npz", allow_pickle=False) as z:
        ut = np.asarray(z["update_timesteps"]).astype(np.int64)
        ent = np.asarray(z["ent_coef"], dtype=np.float32)
        lr = np.asarray(z["learning_rate"], dtype=np.float32)
    w = [i for i, u in enumerate(ut.tolist()) if C.S < u < T]
    pe = np.asarray(P["ent_coef"], np.float32)[[pw[int(ut[i])] for i in w]]
    pl = np.asarray(P["learning_rate"], np.float32)[[pw[int(ut[i])] for i in w]]
    seeded = int((ut <= C.S).sum())
    want(seeded == C.S // C.ROLLOUT, f"seeded rows {seeded}")
    want(len(w) == n_exp and bool((ent[w] == pe).all()) and bool((lr[w] == pl).all()), "probe schedule")
    pd = c["items"]["probe_detail"]
    print(f"{arm:20s} exit {m['exit']} stop {m['stop_reason']} T {T:,} train_calls {m['train_calls']} "
          f"n_updates {d0['n_updates']}->{d1['n_updates']} params {d0['params'][7:19]}=={d1['params'][7:19]} "
          f"adam {d0['adam'][7:19]}=={d1['adam'][7:19]} ent_coef_after_load {m['ent_coef_after_load']} "
          f"gae_lambda {m['gae_lambda_after_load']} wall {m['wall_seconds']} s cpu {m['cpu_seconds']} s")
    print(f"    CHECK passed {c['passed']}; probe_detail {pd}")
    print(f"    probe: seeded rows <= S {seeded}; window rows {len(w)}; ent_coef {float(ent[w][0]):.9f}.."
          f"{float(ent[w][-1]):.9f} "
          f"= parent's {int((ent[w] == pe).sum())}/{len(w)}; lr {float(lr[w][0]):.6e}..{float(lr[w][-1]):.6e} "
          f"= parent's {int((lr[w] == pl).sum())}/{len(w)}")
    print(f"    per rollout ({n}): " + "; ".join(f"{k} {v}/{n}" for k, v in per.items()))
    s = m["arm_evidence"]
    print(f"    summary: max |adv-GAE095| {s['max_adv_err_vs_gae095']:.2e} min {s['min_adv_err_vs_gae095']:.3f}; "
          f"max |adv-MC| {s['max_adv_err_vs_mc']:.2e} min {s['min_adv_err_vs_mc']:.3f}; MC vs lambda-1 recursion "
          f"{s['max_mc_vs_gae_lambda1_recursion']:.1e}; dV pred min {s['min_dv_predict_values']:.4f} max "
          f"{s['max_dv_predict_values']:.4f}; dV fwd/eval/mirror max {s['max_dv_forward']:.4f}/"
          f"{s['max_dv_evaluate_actions']:.4f}/{s['max_dv_mirrored']:.4f}; min dmu {s['min_dmu_actor']:.3f}; buffer vs "
          f"critic max {s['max_buffer_values_vs_critic']:.1e}; buffer vs raw MLP min "
          f"{s['min_buffer_values_vs_raw_value_mlp']:.2e}; obs_rms distinct {s['obs_rms_distinct_digests']}; "
          f"frozen calls {s['frozen_update_calls_first_last']}; ret_rms count {s['ret_rms_count_first_last']}")
    print(f"    problems {p}")
    ok_all &= not p
print("V2 SUMMARY:", "PASS" if ok_all else "FAIL")
sys.exit(0 if ok_all else 1)
