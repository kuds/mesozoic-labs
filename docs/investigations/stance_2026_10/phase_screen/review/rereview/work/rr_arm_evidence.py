"""Re-reviewer: the arm's evidence rules once more, on the FIXER's re-validation V2 records (read only).
(1) the actor never sees the phase; (2) the critic uses the phase (after updates); (3) the tail angle (obs 6) is still
visible to the critic.  Uses the pinned bs_hooks.summarize / evidence_problems and read_bs.phase_meta_problems on the
real records, then on in-memory mutants of those records (nothing written; no optimizer step)."""
import copy
import importlib.util
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
PS = Path("/home/user/work/phase_screen")
sys.path.insert(0, str(PS / "tools"))


def load(name):
    spec = importlib.util.spec_from_file_location(name, str(PS / "tools" / f"{name}.py"))
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


C = load("bs_common")
H = load("bs_hooks")
RB = load("read_bs")
V2 = PS / "validate" / "fixer" / "v2"
fails = 0
for run in ("control_9401", "control_9411", "critic_phase_9401", "critic_phase_9411"):
    arm = run.rsplit("_", 1)[0]
    rows = [json.loads(x) for x in (V2 / run / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
    meta = json.loads((V2 / run / "run_meta.json").read_text())
    check = json.loads((V2 / run / "check.json").read_text())
    s = H.summarize(rows)
    ev = H.evidence_problems(arm, s, n_expected=len(rows))
    pm = RB.phase_meta_problems(meta, arm, "validation")
    dmu_phase = [r.get(k) for r in rows for k in ("dmu_phase_mode", "dmu_phase_forward_actions", "dmu_phase_log_prob",
                                                    "dmu_phase_entropy", "dmu_phase_mirrored") if k in r]
    print(f"{run}: rollouts {len(rows)}; CHECK passed {check.get('passed')}; evidence_problems {ev}; "
          f"phase_meta_problems {pm}")
    print(f"   obs6 -> critic: min dv_predict_values {s.get('min_dv_predict_values')}, max {s.get('max_dv_predict_values')}"
          f"; obs6 -> actor: min dmu_actor {s.get('min_dmu_actor')}")
    print(f"   phase -> actor: max over rows/paths {max(dmu_phase) if dmu_phase else None} ({len(dmu_phase)} values); "
          f"summary max_dmu_phase {s.get('max_dmu_phase')} min_dmu_phase {s.get('min_dmu_phase')}")
    if arm == "critic_phase":
        print(f"   phase -> critic (no update: must be 0): zero_init_max_dv_phase {s.get('zero_init_max_dv_phase')}; "
              f"phase_n_zero_init {s.get('phase_n_zero_init')} phase_n_updated {s.get('phase_n_updated')}; "
              f"max_policy_phase_adam_max_abs {s.get('max_policy_phase_adam_max_abs')}; "
              f"phase_state_end {meta.get('phase_state_end')}; arm_state_end {meta.get('arm_state_end')!r}; "
              f"tensorboard_log {meta.get('tensorboard_log')}")
        # mutants of the real rows (in memory): each must be caught
        muts = {}
        r1 = copy.deepcopy(rows)
        r1[5]["dmu_phase_mode"] = 1e-9
        muts["actor moves with the phase by 1e-9"] = r1
        r2 = copy.deepcopy(rows)
        for r in r2:
            r["dv_predict_values"] = 0.0
        muts["critic blind to obs 6 (dv_predict_values 0)"] = r2
        r3 = copy.deepcopy(rows)
        r3[7]["policy_phase_adam_max_abs"] = 1e-15
        muts["actor phase Adam moment 1e-15"] = r3
        r4 = copy.deepcopy(rows)
        r4[9]["policy_phase_cols_unchanged"] = False
        muts["actor phase columns changed"] = r4
        # a continuation-like record: rollouts 2.. 'updated' but the critic's phase response stays 0
        r5 = copy.deepcopy(rows)
        for i, r in enumerate(r5):
            r["n_updates_since_load"] = 0 if i == 0 else i
            r["adam_steps_since_load"] = 0.0 if i == 0 else float(10 * i)
        muts["critic never uses the phase after updates (responses 0)"] = r5
        for label, rr in muts.items():
            try:
                sm = H.summarize(rr)
                pr = H.evidence_problems(arm, sm, n_expected=len(rr))
            except Exception as exc:          # a refusal by exception also counts as caught
                pr = [f"exception {exc!r}"]
            caught = bool(pr)
            fails += not caught
            print(f"   MUTANT {label}: {'caught' if caught else 'NOT CAUGHT'}: {pr[:2]}")
        # the end-of-run check (fixer P-i3) on the real record and mutants
        R = load("run_bs")
        pse = meta.get("phase_state_end") or {}
        print(f"   end_phase_problems(real) {R.end_phase_problems(pse)}; "
              f"(cols changed) {R.end_phase_problems({**pse, 'policy_phase_cols_unchanged': False})}; "
              f"(adam 1e-15) {R.end_phase_problems({**pse, 'policy_phase_adam_max_abs': 1e-15})}")
    if ev or pm or check.get("passed") is not True:
        fails += 1
print(f"FAILS {fails}")
sys.exit(1 if fails else 0)
