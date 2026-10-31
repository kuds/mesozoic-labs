"""PS V2 (files only; nothing simulated): for each seed, the control and critic_phase no-update runs.

  per run: no optimizer step (parameter and Adam digests and n_updates equal at start and end; train() reached as many
      times as there are update rows); run_meta's status / stop / seal / arm state; CHECK passed (check.json); the
      evidence row by row, read here independently of bs_hooks.evidence_problems:
        control       the critic and the actor see observation 6 on every rollout; the actor responds to the command
                      slots (it is not masked) on every rollout; no phase evidence; GAE(0.95) recomputation;
        critic_phase  the actor's mean (all five paths) moves by exactly 0.0 when the slots are perturbed, on every
                      rollout; the critic's four value paths move by exactly 0.0 too (its phase columns are zero and no
                      update happens in a no-update run: the zero-init case on every rollout) and its phase columns'
                      max |w| is 0.0; the buffer's slots equal the phase of the env counters exactly (max error 0.0);
                      the counters consistent at both ends; the slots' statistics frozen at (0, 1); the mirror map maps
                      the slots to themselves; every training env carries the phase; the actor's phase columns and
                      their Adam moments unchanged; the critic and the actor still see observation 6; GAE(0.95);
  pairing (same seed, train() a no-op): the env side WITHOUT the phase slots identical on every rollout and the full env
      side different on every rollout (the phase is in the observations); rewards, values, advantages identical (the
      critic's phase columns are zero); the observation statistics without the slots identical; the evaluation ledger
      identical (time columns dropped); the probe's bank actions and bank values identical on every row (values: the
      phase fed, zero columns), its statistics identical outside the slots, its slots (0, 1) for critic_phase.

    PY -I -B v2_check.py V2_DIR SEED [SEED ...]     -> exit 0 when every expectation holds
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(TOOLS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


C, H = _load("bs_common"), _load("bs_hooks")
LO, HI = C.PHASE_SLOTS[0], C.PHASE_SLOTS[-1] + 1
TIME_COLS = ("seconds", "wall", "utc", "time_")


def evidence(run):
    return [json.loads(x) for x in (run / C.EVIDENCE).read_text().splitlines() if x.strip()]


def ledger_rows(run):
    with open(run / "01_stance" / "eval_episodes.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    keep = [k for k in rows[0] if not any(t in k for t in TIME_COLS)]
    return keep, [[r[k] for k in keep] for r in rows]


def probe(run):
    with np.load(run / "01_stance" / "policy_probe.npz", allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in ("update_timesteps", "bank_action_now", "bank_value_now", "bank_raw_obs",
                                               "obs_rms_mean", "obs_rms_var", "obs_rms_count", "ret_rms_var",
                                               "bank_value_frozen", "bank_action_frozen")}


def main():
    root = Path(sys.argv[1])
    ok = True
    lines = []

    def expect(cond, text):
        nonlocal ok
        ok &= bool(cond)
        lines.append(f"  [{'ok' if cond else 'FAIL'}] {text}")

    for seed in [int(x) for x in sys.argv[2:]]:
        runs = {a: root / f"{a}_{seed}" for a in C.ARMS}
        ev, metas = {}, {}
        for arm, run in runs.items():
            m = json.loads((run / C.RUN_META).read_text())
            c = json.loads((run / C.CHECK_JSON).read_text())
            e = evidence(run)
            ev[arm], metas[arm] = e, m
            T = m["T"]
            n = C.expected_update_rows(C.S, T)
            d0, d1 = m["digest_start"], m["digest_end"]
            lines.append(f"{arm}_{seed}: T {T}, {len(e)} rollout ends, status {m['status']} exit {m['exit']}, "
                         f"stop {m['stop_reason']}, CPU {m['cpu_seconds']} s, wall {m['wall_seconds']} s")
            expect(m["status"] == "done" and m["exit"] == 0 and m["stop_reason"] == "end" and T == C.VALIDATE_UNTIL,
                   "done at VALIDATE_UNTIL with stop reason end")
            expect(d0["params"] == d1["params"] and d0["adam"] == d1["adam"] and d0["n_updates"] == d1["n_updates"]
                   and m["train_calls"] == n and m["no_update_check"]["pass"] is True,
                   f"no optimizer step: params / Adam / n_updates ({d0['n_updates']}) unchanged; train() reached "
                   f"{m['train_calls']} = {n} times")
            expect(m["code_seal_after"] == "unchanged" and m["modules_outside_code"] == [] and
                   m["arm_state_end"] == "in force" and m["ent_coef_after_load"] == 0.005,
                   "seal unchanged, modules from the copy, the arm in force at the end, ent_coef 0.005 after load")
            expect(c["passed"] is True and c["failures"] == [], f"CHECK passed ({c['failures']})")
            expect([x["num_timesteps"] for x in e] == [C.S + C.ROLLOUT * k for k in range(1, n + 1)],
                   "one evidence line per rollout end")
            tol = H.ADV_REL_TOL * max(x["adv_scale"] for x in e)
            expect(all(x["adv_err_vs_gae095"] <= tol and x["buffer_gae_lambda"] == 0.95 for x in e),
                   f"advantages = GAE(0.95) recomputation on every rollout (max err "
                   f"{max(x['adv_err_vs_gae095'] for x in e):.3g} <= {tol:.3g})")
            expect(all(x["dv_predict_values"] > 0 for x in e) and all(x["dmu_actor"] > 0 for x in e),
                   f"the critic and the actor see observation 6 on every rollout (min dV "
                   f"{min(x['dv_predict_values'] for x in e):.4g}, min dmu {min(x['dmu_actor'] for x in e):.4g})")
            act = [max(x[k] for k in H.PHASE_ACTOR_KEYS) for x in e]
            crit = [max(x[k] for k in H.PHASE_CRITIC_KEYS) for x in e]
            if arm == "control":
                expect(all(a > 0 for a in act) and all(x["phase_buffer_max_err"] is None for x in e),
                       f"control: the actor responds to the slots on every rollout (min {min(act):.4g}); no phase "
                       f"evidence; its critic responds too (min {min(crit):.4g})")
                expect(m["probe_phase"]["installed"] is False and m["probe_phase"]["class"] == "PolicyProbeCallback",
                       "control: the stock policy probe")
            else:
                expect(all(a == 0.0 for a in act), f"critic_phase: the actor moved by exactly 0.0 on all "
                                                   f"{len(e)} rollouts and all 5 paths (max {max(act)})")
                expect(all(c_ == 0.0 for c_ in crit) and all(x["value_phase_cols_max_abs"] == 0.0 for x in e)
                       and all(x["n_updates_since_load"] == 0 and x["adam_steps_since_load"] == 0 for x in e),
                       f"critic_phase: no update, so the zero-initialised critic columns stay 0 and the critic's "
                       f"4 paths move by exactly 0.0 on all {len(e)} rollouts (max {max(crit)})")
                expect(all(x["phase_buffer_max_err"] == 0.0 and x["phase_rows"] == 8192 for x in e),
                       "critic_phase: every buffer row's slots = the phase of the env counter, exactly (8,192 rows "
                       "per rollout)")
                expect(all(x["phase_k_start_consistent"] and x["phase_k_end_consistent"] for x in e),
                       "critic_phase: the counters reconstructed from the rollout's start reach the envs' counters")
                expect(all(x["phase_stats_frozen"] and x["mirror_phase_self_map"] and x["envs_swapped"]
                           and x["policy_phase_cols_unchanged"] and x["policy_phase_adam_max_abs"] == 0.0 for x in e),
                       "critic_phase: slots' statistics (0, 1), mirror self-map +1, 4 envs with the phase, the actor's "
                       "phase columns and Adam moments unchanged, on every rollout")
                pre = sum(x["phase_rows_presettle"] for x in e)
                lines.append(f"    phase: k from {min(x['phase_k_min'] for x in e)} to "
                             f"{max(x['phase_k_max'] for x in e)}; {pre} of {8192 * len(e)} buffer rows before the "
                             f"settle (k < 200); slots' statistics update calls "
                             f"{e[0]['phase_rms_calls']} -> {e[-1]['phase_rms_calls']}")
                pp = m["probe_phase"]
                expect(pp["installed"] is True and pp["class"] == "PhaseFedPolicyProbeCallback"
                       and pp["phase_fed"] == [[1.0, 1.0, 1.0]] * len(pp["k"]) and min(pp["k"]) >= C.SETTLE_STEPS,
                       f"critic_phase: the probe feeds the settled phase to all {len(pp['k'])} bank rows (k "
                       f"{sorted(set(pp['k']))})")
        c_ev, p_ev = ev["control"], ev["critic_phase"]
        same = {k: [x[k] == y[k] for x, y in zip(c_ev, p_ev, strict=True)]
                for k in ("digest_env_side_no_phase", "digest_env_side", "digest_rewards", "digest_values",
                          "digest_advantages", "obs_rms_digest_no_phase", "obs_rms_digest")}
        lines.append(f"seed {seed} pairing, rollouts equal: { {k: f'{sum(v)}/{len(v)}' for k, v in same.items()} }")
        expect(all(same["digest_env_side_no_phase"]) and not any(same["digest_env_side"]),
               "the env side without the slots identical on every rollout; with them different on every rollout")
        expect(all(same["digest_rewards"]) and all(same["digest_values"]) and all(same["digest_advantages"]),
               "rewards, values and advantages identical on every rollout (the zero-initialised critic)")
        expect(all(same["obs_rms_digest_no_phase"]) and not any(same["obs_rms_digest"]),
               "the observation statistics identical outside the slots, different in them, on every rollout")
        lk, lc = ledger_rows(runs["control"])
        k2, l2 = ledger_rows(runs["critic_phase"])
        expect(k2 == lk and l2 == lc, f"the evaluation ledger identical ({len(lc)} rows, {len(lk)} columns; time "
                                      f"columns dropped)")
        pc, pp_ = probe(runs["control"]), probe(runs["critic_phase"])
        mask = np.ones(pc["obs_rms_mean"].shape[1], dtype=bool)
        mask[LO:HI] = False
        win = pc["update_timesteps"] > C.S
        expect(np.array_equal(pc["update_timesteps"], pp_["update_timesteps"])
               and np.array_equal(pc["bank_action_now"], pp_["bank_action_now"])
               and np.array_equal(pc["bank_value_now"], pp_["bank_value_now"])
               and np.array_equal(pc["bank_action_frozen"], pp_["bank_action_frozen"])
               and np.array_equal(pc["bank_value_frozen"], pp_["bank_value_frozen"])
               and np.array_equal(pc["bank_raw_obs"], pp_["bank_raw_obs"]),
               f"the probe: the same {len(pc['update_timesteps'])} rows ({int((~win).sum())} seeded at or before S), "
               "bank actions and values (now and frozen) identical, the recorded bank identical")
        expect(np.array_equal(pc["obs_rms_mean"][:, mask], pp_["obs_rms_mean"][:, mask])
               and np.array_equal(pc["obs_rms_var"][:, mask], pp_["obs_rms_var"][:, mask])
               and np.array_equal(pc["obs_rms_count"], pp_["obs_rms_count"])
               and np.array_equal(pc["ret_rms_var"], pp_["ret_rms_var"])
               and np.all(pp_["obs_rms_mean"][win][:, LO:HI] == 0.0)
               and np.all(pp_["obs_rms_var"][win][:, LO:HI] == 1.0),
               "the probe's statistics identical outside the slots; critic_phase's slots (0, 1) on every window row")
    print("\n".join(lines))
    print("V2 CHECK: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
