"""Process review of PS: does CHECK reject a critic_phase continuation whose actor ever saw the phase, or whose critic
never used it?  Pure: reads PS's V2 evidence (validate/v2/critic_phase_9401, a no-update run that passed CHECK) and
run_meta read only, makes a synthetic CONTINUATION from it in memory (rollout 1 zero-init; rollouts 2..31 after
updates, with the critic's phase columns and responses > 0), checks it passes PS's own bs_hooks.evidence_problems and
read_bs.phase_meta_problems, then applies one mutation at a time.  Nothing is written but stdout.
Usage: PY -I -B check_mutants.py <PS tools dir> <V2 run dir>"""
import sys

sys.dont_write_bytecode = True

import copy  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS, RUN = Path(sys.argv[1]), Path(sys.argv[2])


def load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = load("bs_common")
H = load("bs_hooks")
R = load("read_bs")
lines0 = [json.loads(x) for x in (RUN / "arm_evidence.jsonl").read_text().splitlines() if x.strip()]
meta0 = json.loads((RUN / "run_meta.json").read_text())
n = len(lines0)
assert meta0["arm"] == "critic_phase" and n == 31, (meta0["arm"], n)


def as_continuation(lines):
    out = copy.deepcopy(lines)
    for i, e in enumerate(out):
        if i == 0:
            continue                                   # rollout 1: before the first train()
        e["n_updates_since_load"] = 10 * i
        e["adam_steps_since_load"] = 640.0 * i
        e["value_phase_cols_max_abs"] = 1e-4 * i
        for k in H.PHASE_CRITIC_KEYS:
            e[k] = 1e-3 * i
    return out


def verdict(lines, kind="continuation"):
    s = H.summarize(lines)
    ev = H.evidence_problems("critic_phase", s, n_expected=n)
    meta = copy.deepcopy(meta0)
    meta["arm_evidence"] = json.loads(json.dumps(s))
    meta["mode"] = kind
    mp = R.phase_meta_problems(meta, "critic_phase", kind)
    return ev, mp


base = as_continuation(lines0)
ev, mp = verdict(base)
print(f"BASE synthetic continuation (rollout 1 zero-init, 2..{n} updated): evidence_problems {ev}; "
      f"phase_meta_problems {mp}")
assert not ev and not mp


def mut(title, fn, kind="continuation"):
    lines = copy.deepcopy(base)
    fn(lines)
    ev, mp = verdict(lines, kind)
    caught = bool(ev or mp)
    print(f"{'CAUGHT ' if caught else 'MISSED '} {title}\n         evidence: {ev}\n         run_meta: {mp}")
    return caught


res = []
# the actor saw the phase
res.append(mut("A1 actor's mean moved by 1e-9 with the phase slots on ONE rollout (17)",
               lambda L: L[16].__setitem__("dmu_phase_mode", 1e-9)))
res.append(mut("A2 actor's mirrored mean moved by 1e-12 on the LAST rollout",
               lambda L: L[-1].__setitem__("dmu_phase_mirrored", 1e-12)))
res.append(mut("A3 actor's log-prob through evaluate_actions moved on rollout 2",
               lambda L: L[1].__setitem__("dmu_phase_log_prob", 3e-7)))
res.append(mut("A4 actor's first-layer phase columns changed after an update (rollout 9)",
               lambda L: L[8].__setitem__("policy_phase_cols_unchanged", False)))
res.append(mut("A5 actor's phase columns got Adam moments (a gradient reached them) on rollout 30",
               lambda L: L[29].__setitem__("policy_phase_adam_max_abs", 1e-15)))
# the critic never used the phase
def never(L):
    for e in L:
        e["n_updates_since_load"], e["adam_steps_since_load"] = 0, 0.0
        e["value_phase_cols_max_abs"] = 0.0
        for k in H.PHASE_CRITIC_KEYS:
            e[k] = 0.0
res.append(mut("B1 critic never learned: every rollout zero-init (as if train() never ran on it)", never))
def cols_zero(L):
    for e in L[1:]:
        e["value_phase_cols_max_abs"] = 0.0
        for k in H.PHASE_CRITIC_KEYS:
            e[k] = 0.0
res.append(mut("B2 updates ran but the critic's phase columns stayed 0 and it never responded (input cut)", cols_zero))
res.append(mut("B3 after updates the critic did not respond on ONE rollout's predict_values path (rollout 50%)",
               lambda L: L[15].__setitem__("dv_phase_predict_values", 0.0)))
res.append(mut("B4 after updates the mirrored critic did not respond on one rollout",
               lambda L: L[20].__setitem__("dv_phase_mirrored", 0.0)))
res.append(mut("B5 critic responded on rollout 1 (not zero-initialised at the load)",
               lambda L: L[0].__setitem__("dv_phase_forward", 1e-6)))
res.append(mut("B6 a zero-init rollout after an updated one (order)",
               lambda L: L[5].__setitem__("n_updates_since_load", 0)))
# other phase integrity
res.append(mut("C1 the buffer's phase slots off by 1 ulp on one rollout",
               lambda L: L[3].__setitem__("phase_buffer_max_err", 5.96e-8)))
res.append(mut("C2 the slot statistics unfrozen on one rollout",
               lambda L: L[7].__setitem__("phase_stats_frozen", False)))
res.append(mut("C3 one training env not carrying the phase class on one rollout",
               lambda L: L[11].__setitem__("envs_swapped", False)))
res.append(mut("C4 critic blind to observation 6 on one rollout (C4)",
               lambda L: L[12].__setitem__("dv_predict_values", 0.0)))
res.append(mut("C5 one rollout's evidence line missing (30 of 31)", lambda L: L.pop(10)))
print(f"\n{sum(res)} of {len(res)} mutations caught")
