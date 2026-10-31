"""PS build: run_bs.py, V1's as-loaded bank outputs taken before the hook instead of a deep copy of the policy."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''def phase_bank_check(policy: Any, train_env: Any, policy_load: Any, rms_load: Any) -> dict:
    """V1 for critic_phase: the hooked policy on the parent probe's bank, the phase of each counter in PHASE_V1_K
    (and each row's own, k = kept step + 1) fed into slots 60-62 and normalised with the hooked statistics, against
    the policy AS LOADED (a deep copy taken before the hook) on the bank as recorded (slots zero) under the loaded
    statistics: the actions and the values must be equal BITWISE at every counter (the actor's mask; the critic's
    zero columns)."""
    import numpy as np

    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        raw = np.asarray(z["bank_raw_obs"], dtype=np.float64)
        steps = np.asarray(z["bank_steps"], dtype=np.int64)
    eps, clip = float(train_env.epsilon), float(train_env.clip_obs)
    act0, val0 = _bank_outputs(policy_load, rms_load, eps, clip, raw)
    out: dict[str, Any] = {''', '''def bank_outputs_as_loaded(policy: Any, train_env: Any) -> tuple[Any, Any]:
    """V1: the policy AS LOADED (before the hook) on the parent probe's bank as recorded (slots 60-62 zero) under the
    loaded statistics: (actions, values)."""
    import numpy as np

    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        raw = np.asarray(z["bank_raw_obs"], dtype=np.float64)
    return _bank_outputs(policy, train_env.obs_rms, float(train_env.epsilon), float(train_env.clip_obs), raw)


def phase_bank_check(policy: Any, train_env: Any, act0: Any, val0: Any) -> dict:
    """V1 for critic_phase: the hooked policy on the parent probe's bank, the phase of each counter in PHASE_V1_K
    (and each row's own, k = kept step + 1) fed into slots 60-62 and normalised with the hooked statistics, against
    (act0, val0), the policy AS LOADED on the bank as recorded under the loaded statistics (bank_outputs_as_loaded,
    taken before the hook): the actions and the values must be equal BITWISE at every counter (the actor's mask; the
    critic's zero columns)."""
    import numpy as np

    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        raw = np.asarray(z["bank_raw_obs"], dtype=np.float64)
        steps = np.asarray(z["bank_steps"], dtype=np.int64)
    eps, clip = float(train_env.epsilon), float(train_env.clip_obs)
    out: dict[str, Any] = {''')
rep('''        if mode == "save_loaded_only":             # V1: the policy and statistics as loaded, before the hook
            holder["policy_load"] = copy.deepcopy(model.policy)
            holder["rms_load"] = copy.deepcopy(train_env.obs_rms)''', '''        if mode == "save_loaded_only":             # V1: the policy's bank outputs as loaded, before the hook
            holder["bank_load"] = bank_outputs_as_loaded(model.policy, train_env)''')
rep('''                holder["v1_phase_bank"] = phase_bank_check(model.policy, train_env, holder["policy_load"],
                                                           holder["rms_load"])''', '''                holder["v1_phase_bank"] = phase_bank_check(model.policy, train_env, *holder["bank_load"])''')
p.write_text(s)
print("ok")
