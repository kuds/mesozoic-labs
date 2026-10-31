"""PS build: the edits of run_bs.py (each old string asserted unique), run once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''"""BS2: one continuation of r10 stance6 s42 from its 2,097,152 pair under one arm, or a run making no optimizer step.''',
    '''"""PS: one continuation of r10 stance6 s42 from its 2,097,152 pair under one arm, or a run making no optimizer step.''')
rep('''ARM: control | critic_blind_tail3 | gae_lambda_1 | frozen_obs_norm (bs_common.ARMS; what each changes:
bs_common.ARM_CHANGE, applied by bs_hooks.install_arm / arm_config here).  S = 2,097,152, E = S + 1,441,792 =
3,538,944 (11 evaluations of 131,072; 176 rollouts, 175 updates).  BS2 = BS's run_bs.py (sha256 0a5a9851...) with the
window, the start pair and the validation seeds taken from BS2's bs_common.py, the code copy BS's own (read only), and
GIT_CEILING_DIRECTORIES also naming the code copy's parent folder (so the code's `git rev-parse` stops there, as in BS).''',
    '''ARM: control | critic_phase (bs_common.ARMS; what each changes: bs_common.ARM_CHANGE, applied by
bs_hooks.install_arm here).  S = 2,097,152, E = S + 1,441,792 = 3,538,944 (11 evaluations of 131,072; 176 rollouts,
175 updates).  PS = BS2's run_bs.py (sha256 acd5b499...) with: the root, arms and validation seeds of PS's
bs_common.py; critic_phase's load-time transform (the critic's phase columns zeroed, the statistics' phase slots
reseeded; every other tensor and the optimizer state verified unchanged) recorded as digest_load (the pair as loaded)
and digest_start (the arm's starting state; equal for control); the policy probe hook at training start
(bs_hooks.probe_phase: critic_phase feeds the bank's phase, control only checks the probe is stock); the envs' step
counters at each rollout's start for critic_phase's evidence; V1's bank check and pair comparison for critic_phase
(the actor and, at the load, the critic bitwise the start pair's at every phase value).  BS2 = BS's run_bs.py (sha256
0a5a9851...) with the window, the start pair and the validation seeds of BS2's bs_common.py, the code copy BS's own
(read only), and GIT_CEILING_DIRECTORIES also naming the code copy's parent folder (so the code's `git rev-parse`
stops there, as in BS).''')
rep('''VERSION = "run_bs 1.1 (2026-10-30; bias screen 2 from s42 2,097,152; derived from run_bs 1.0 and run_ecs 1.0)"''',
    '''VERSION = "run_bs 2.0 (2026-10-31; phase screen from s42 2,097,152; derived from run_bs 1.1 of bias screen 2)"''')
# bank_check: the phase variant
rep('''def bank_check(policy: Any, train_env: Any) -> dict:
    """V1: the loaded (arm-applied) policy on the parent probe's bank under the loaded statistics, against the probe's
    row at S (bank_action_now, bank_value_now): the probe's own formula (VecNormalize's, in numpy, float32 tensor)."""
    import numpy as np
    import torch

    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        raw = np.asarray(z["bank_raw_obs"], dtype=np.float64)''', '''def _bank_outputs(policy: Any, rms: Any, epsilon: float, clip: float, raw: Any) -> tuple[Any, Any]:
    """(deterministic actions, values) of a policy on raw observations under (rms, epsilon, clip): the probe's own
    formula (VecNormalize's, in numpy, then a float32 tensor); the policy's action_dist put back as found."""
    import numpy as np
    import torch

    normed = np.clip((raw - rms.mean) / np.sqrt(rms.var + float(epsilon)), -float(clip), float(clip))
    stored = dict(vars(policy.action_dist))
    try:
        with torch.no_grad():
            t = torch.as_tensor(normed, dtype=torch.float32)
            return (policy.get_distribution(t).mode().double().numpy(),
                    policy.predict_values(t).flatten().double().numpy())
    finally:
        vars(policy.action_dist).clear()
        vars(policy.action_dist).update(stored)


PHASE_V1_K = (0, 1, 50, 100, 150, 199, 200, 201, 600, 1000)     # the step counters V1 feeds (settled from 200)


def phase_bank_check(policy: Any, train_env: Any, policy_load: Any, rms_load: Any) -> dict:
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
    out: dict[str, Any] = {"k_fed": list(PHASE_V1_K), "k_own": (steps + 1).tolist(), "per_k": {}}
    worst_a = worst_v = 0.0
    for label, ks in [(str(k), np.full(len(raw), k)) for k in PHASE_V1_K] + [("own", steps + 1)]:
        fed = raw.copy()
        fed[:, list(C.PHASE_SLOTS)] = H.phase_matrix(ks)
        act, val = _bank_outputs(policy, train_env.obs_rms, eps, clip, fed)
        da, dv = float(np.max(np.abs(act - act0))), float(np.max(np.abs(val - val0)))
        out["per_k"][label] = {"max_abs_diff_actions": da, "max_abs_diff_values": dv}
        worst_a, worst_v = max(worst_a, da), max(worst_v, dv)
    out.update(max_abs_diff_actions=worst_a, max_abs_diff_values=worst_v,
               bitwise_actor=worst_a == 0.0, bitwise_critic=worst_v == 0.0)
    return out


def bank_check(policy: Any, train_env: Any) -> dict:
    """V1: the loaded (arm-applied) policy on the parent probe's bank under the loaded statistics, against the probe's
    row at S (bank_action_now, bank_value_now): the probe's own formula (VecNormalize's, in numpy, float32 tensor).
    critic_phase: the bank's own phase (every row settled) fed, as the probe feeds it."""
    import numpy as np
    import torch

    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        raw = np.asarray(z["bank_raw_obs"], dtype=np.float64)
        if getattr(type(policy.mlp_extractor), "ps_phase_slots", None):
            raw = raw.copy()
            raw[:, list(C.PHASE_SLOTS)] = H.phase_matrix(np.asarray(z["bank_steps"], dtype=np.int64) + 1)''')
rep('''        holder["digest_start"] = state_digest(model, train_env)
        if problems:
            raise Refused("load check: " + " | ".join(problems))
        try:
            holder["arm_applied"] = H.install_arm(arm, model, train_env)
        except H.HookError as exc:
            raise Refused(f"arm hook: {exc}") from exc
        state = H.arm_state_problems(arm, model, train_env)
        digest_hooked = state_digest(model, train_env)
        if state or {k: digest_hooked[k] for k in ("params", "adam", "vecnormalize")} != {
                k: holder["digest_start"][k] for k in ("params", "adam", "vecnormalize")}:
            raise Refused(f"arm hook: state {state}; digests after the hook {digest_hooked} != at the load")
        holder["start_obs_digest"] = H.rms_digest(train_env.obs_rms)
        if mode == "save_loaded_only":
            holder["v1_bank"] = bank_check(model.policy, train_env)''', '''        holder["digest_load"] = state_digest(model, train_env)
        if problems:
            raise Refused("load check: " + " | ".join(problems))
        if mode == "save_loaded_only":             # V1: the policy and statistics as loaded, before the hook
            holder["policy_load"] = copy.deepcopy(model.policy)
            holder["rms_load"] = copy.deepcopy(train_env.obs_rms)
        try:
            holder["arm_applied"] = H.install_arm(arm, model, train_env)
        except H.HookError as exc:
            raise Refused(f"arm hook: {exc}") from exc
        state = H.arm_state_problems(arm, model, train_env)
        digest_hooked = state_digest(model, train_env)
        # control: nothing may change.  critic_phase: the optimizer state may not change; the parameters and the
        # statistics change exactly as install_arm verified (transform_verified: the critic's phase columns zeroed,
        # the slots reseeded; every other tensor equal).
        same = ("params", "adam", "vecnormalize") if arm != "critic_phase" else ("adam",)
        if state or {k: digest_hooked[k] for k in same} != {k: holder["digest_load"][k] for k in same} or (
                arm == "critic_phase" and holder["arm_applied"].get("transform_verified") is not True):
            raise Refused(f"arm hook: state {state}; digests after the hook {digest_hooked} != at the load")
        holder["digest_start"] = digest_hooked
        holder["start_obs_digest"] = H.rms_digest(train_env.obs_rms)
        if arm == "critic_phase":
            holder["phase_ref"] = H.phase_reference(model)
        if mode == "save_loaded_only":
            holder["v1_bank"] = bank_check(model.policy, train_env)
            if arm == "critic_phase":
                holder["v1_phase_bank"] = phase_bank_check(model.policy, train_env, holder["policy_load"],
                                                           holder["rms_load"])''')
rep('''    """StopAtEnd (run_ecs's): _on_step returns False when num_timesteps == target; at each rollout's end (after GAE,
    before that rollout's update) it calls evidence(locals) then heartbeat(n), neither of which changes training."""

    class StopAtEnd(base_callback):  # type: ignore[misc, valid-type]
        def __init__(self, target: int, heartbeat: Any = None, evidence: Any = None) -> None:''', '''    """StopAtEnd (run_ecs's): _on_step returns False when num_timesteps == target; at each rollout's end (after GAE,
    before that rollout's update) it calls evidence(locals) then heartbeat(n), neither of which changes training.
    PS: at training start (after every earlier callback's, so the probe's bank exists) training_start(siblings); at
    each rollout's start rollout_start() (critic_phase: the envs' step counters, read only)."""

    class StopAtEnd(base_callback):  # type: ignore[misc, valid-type]
        def __init__(self, target: int, heartbeat: Any = None, evidence: Any = None, training_start: Any = None,
                     rollout_start: Any = None) -> None:''')
rep('''            self.heartbeat = heartbeat
            self.evidence = evidence

        def _on_training_start(self) -> None:
            self.siblings = list(flatten_callbacks(self.locals.get("callback")))
''', '''            self.heartbeat = heartbeat
            self.evidence = evidence
            self.training_start = training_start
            self.rollout_start = rollout_start

        def _on_training_start(self) -> None:
            self.siblings = list(flatten_callbacks(self.locals.get("callback")))
            if self.training_start is not None:
                self.training_start(self.siblings)

        def _on_rollout_start(self) -> None:
            if self.rollout_start is not None:
                self.rollout_start()
''')
rep('''        "arm_change": C.ARM_CHANGE[arm], "arm_ppo": dict(C.ARM_PPO_EDIT[arm]) or "unchanged", "ppo_applied": None,
        "arm_applied": None, "arm_evidence": None,
    }''', '''        "arm_change": C.ARM_CHANGE[arm], "arm_ppo": dict(C.ARM_PPO_EDIT[arm]) or "unchanged", "ppo_applied": None,
        "arm_applied": None, "arm_evidence": None, "probe_phase": None,
        "phase_definition": ({"slots": list(C.PHASE_SLOTS), "features": list(C.PHASE_FEATURES),
                              "settle_steps": C.SETTLE_STEPS, "gamma": C.PHASE_GAMMA,
                              "k": "the env's _step_count when the observation is made (0 at reset)"}
                             if arm == "critic_phase" else None),
    }''')
rep('''        def evidence(locals_: dict) -> None:
            ev = H.rollout_evidence(arm, holder["model"], holder["train_env"], locals_,
                                    start_obs_digest=holder["start_obs_digest"])''', '''        def evidence(locals_: dict) -> None:
            ctx = {"k0": holder.get("k0"), "ref": holder.get("phase_ref")} if arm == "critic_phase" else None
            ev = H.rollout_evidence(arm, holder["model"], holder["train_env"], locals_,
                                    start_obs_digest=holder["start_obs_digest"], phase_ctx=ctx)''')
rep('''        def stage_callback(*, eval_env: Any, eval_callback: Any) -> Any:
            wp = wrapper_problems(holder, mode=mode, update_classes=(MirrorKLPPO, sb3["PPO"]))
            if wp:
                raise Refused(" | ".join(wp))
            holder["stop"] = stop_cls(target, heartbeat=heartbeat, evidence=evidence)
            return holder["stop"]''', '''        def training_start(siblings: list) -> None:
            holder["probe_phase"] = H.probe_phase(arm, siblings)
            meta["probe_phase"] = holder["probe_phase"]

        def rollout_start() -> None:
            holder["k0"] = H.env_step_counts(holder["train_env"])

        def stage_callback(*, eval_env: Any, eval_callback: Any) -> Any:
            wp = wrapper_problems(holder, mode=mode, update_classes=(MirrorKLPPO, sb3["PPO"]))
            if wp:
                raise Refused(" | ".join(wp))
            holder["stop"] = stop_cls(target, heartbeat=heartbeat, evidence=evidence, training_start=training_start,
                                      rollout_start=rollout_start if arm == "critic_phase" else None)
            return holder["stop"]''')
rep('''            "torch_threads": holder.get("torch_threads"), "torch_interop_threads": holder.get("torch_interop_threads"),
            "digest_start": holder.get("digest_start"), "digest_end": digest_end,''', '''            "torch_threads": holder.get("torch_threads"), "torch_interop_threads": holder.get("torch_interop_threads"),
            "digest_load": holder.get("digest_load"), "digest_start": holder.get("digest_start"), "digest_end": digest_end,
            "probe_phase": holder.get("probe_phase"),''')
# V1 branch
rep('''        loaded = holder.get("digest_start") or {}
        v1 = {"start_zip": start_zip, "saved_zip": saved, "loaded": {k: loaded.get(k) for k in ("params", "adam")},
              "start_pkl_vecnormalize": vn_start, "saved_pkl_vecnormalize": vn_saved,
              "loaded_vecnormalize": loaded.get("vecnormalize"), "saved": holder.get("saved"),
              "saved_pkl_rms_classes": saved_rms_cls, "num_timesteps_loaded": loaded.get("num_timesteps"),
              "saved_sha256": {p.name: sha256_file(p) for p in (saved_zip, saved_pkl)},
              "start_zip_sha256": st["zip_sha256"], "bank": holder.get("v1_bank")}
        v1["params_equal"] = start_zip["params"] == saved["params"] == loaded.get("params")
        v1["adam_equal"] = start_zip["adam"] == saved["adam"] == loaded.get("adam")
        v1["vecnormalize_equal"] = vn_start == vn_saved == loaded.get("vecnormalize")
        v1["num_timesteps_S"] = loaded.get("num_timesteps") == S
        v1["saved_pkl_stock_classes"] = saved_rms_cls == {"obs_rms": "RunningMeanStd", "ret_rms": "RunningMeanStd"}
        bank = holder.get("v1_bank") or {}
        v1["bank_tail3_equal"] = bank.get("max_abs_diff_tail3_statue_settled", 1.0) <= 1e-6
        ok = all(v1[k] for k in ("params_equal", "adam_equal", "vecnormalize_equal", "num_timesteps_S",
                                 "saved_pkl_stock_classes", "bank_tail3_equal"))''', '''        loaded = holder.get("digest_load") or {}
        hooked = holder.get("digest_start") or {}
        v1 = {"start_zip": start_zip, "saved_zip": saved, "loaded": {k: loaded.get(k) for k in ("params", "adam")},
              "hooked": {k: hooked.get(k) for k in ("params", "adam")},
              "start_pkl_vecnormalize": vn_start, "saved_pkl_vecnormalize": vn_saved,
              "loaded_vecnormalize": loaded.get("vecnormalize"), "hooked_vecnormalize": hooked.get("vecnormalize"),
              "saved": holder.get("saved"),
              "saved_pkl_rms_classes": saved_rms_cls, "num_timesteps_loaded": loaded.get("num_timesteps"),
              "saved_sha256": {p.name: sha256_file(p) for p in (saved_zip, saved_pkl)},
              "start_zip_sha256": st["zip_sha256"], "bank": holder.get("v1_bank")}
        # the start pair as loaded = the start zip / pkl; the saved pair = the arm's starting state (hooked); for
        # control the hook changes nothing (BS2's three equalities), for critic_phase exactly the phase transform
        v1["load_equal_start"] = (start_zip["params"] == loaded.get("params") and start_zip["adam"] == loaded.get("adam")
                                  and vn_start == loaded.get("vecnormalize"))
        v1["saved_equal_hooked"] = (saved["params"] == hooked.get("params") and saved["adam"] == hooked.get("adam")
                                    and vn_saved == hooked.get("vecnormalize"))
        if arm == "critic_phase":
            v1["phase_pair"] = phase_pair_compare(stage_dir / "models" / C.periodic_names(S)[0], saved_zip,
                                                  stage_dir / "models" / C.periodic_names(S)[1], saved_pkl)
            v1["phase_bank"] = holder.get("v1_phase_bank")
            pb = v1["phase_bank"] or {}
            v1["params_equal"] = v1["load_equal_start"] and v1["saved_equal_hooked"] and v1["phase_pair"]["ok"]
            v1["adam_equal"] = start_zip["adam"] == saved["adam"] == loaded.get("adam") == hooked.get("adam")
            v1["vecnormalize_equal"] = v1["load_equal_start"] and v1["saved_equal_hooked"]
            v1["phase_bitwise"] = pb.get("bitwise_actor") is True and pb.get("bitwise_critic") is True
        else:
            v1["params_equal"] = start_zip["params"] == saved["params"] == loaded.get("params") == hooked.get("params")
            v1["adam_equal"] = start_zip["adam"] == saved["adam"] == loaded.get("adam") == hooked.get("adam")
            v1["vecnormalize_equal"] = vn_start == vn_saved == loaded.get("vecnormalize") == hooked.get("vecnormalize")
            v1["phase_bitwise"] = True
        v1["num_timesteps_S"] = loaded.get("num_timesteps") == S
        v1["saved_pkl_stock_classes"] = saved_rms_cls == {"obs_rms": "RunningMeanStd", "ret_rms": "RunningMeanStd"}
        bank = holder.get("v1_bank") or {}
        v1["bank_tail3_equal"] = bank.get("max_abs_diff_tail3_statue_settled", 1.0) <= 1e-6
        ok = all(v1[k] for k in ("params_equal", "adam_equal", "vecnormalize_equal", "num_timesteps_S",
                                 "saved_pkl_stock_classes", "bank_tail3_equal", "phase_bitwise"))''')
rep('''                     "digest_start": holder.get("digest_start"), "utc_end": utc_now(),''',
    '''                     "digest_load": holder.get("digest_load"),
                     "digest_start": holder.get("digest_start"), "utc_end": utc_now(),''')
rep('''def pkl_stats(pkl_path: str | os.PathLike[str]) -> Any:''', '''def phase_pair_compare(start_zip: Any, saved_zip: Any, start_pkl: Any, saved_pkl: Any) -> dict:
    """V1 for critic_phase, tensor by tensor (torch.load weights_only; pickle for the pkls): the saved pair equals the
    start pair except value_net.0.weight[:, 60:63], zero in the saved one (nonzero in the start one), and the
    observation statistics' slots 60-62, mean 0 / var 1 in the saved pkl (mean 0 / var ~2.4e-11 in the start one);
    every Adam moment and the step equal; ret_rms and the counts equal."""
    import io
    import zipfile

    import numpy as np
    import torch

    def load(path: Any) -> tuple[dict, dict]:
        with zipfile.ZipFile(path) as z:
            return (torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True),
                    torch.load(io.BytesIO(z.read("policy.optimizer.pth")), map_location="cpu", weights_only=True))

    (p0, o0), (p1, o1) = load(start_zip), load(saved_zip)
    lo, hi = C.PHASE_SLOTS[0], C.PHASE_SLOTS[-1] + 1
    name = "mlp_extractor.value_net.0.weight"
    out: dict[str, Any] = {"keys_equal": list(p0) == list(p1)}
    out["other_tensors_equal"] = out["keys_equal"] and all(p0[k].equal(p1[k]) for k in p0 if k != name)
    keep = torch.ones(p0[name].shape[1], dtype=torch.bool)
    keep[lo:hi] = False
    out["value_net0_other_cols_equal"] = bool(p0[name][:, keep].equal(p1[name][:, keep]))
    out["value_net0_phase_cols_saved_max_abs"] = float(p1[name][:, lo:hi].abs().max())
    out["value_net0_phase_cols_start_max_abs"] = float(p0[name][:, lo:hi].abs().max())
    out["policy_net0_phase_cols_equal"] = bool(p0["mlp_extractor.policy_net.0.weight"][:, lo:hi].equal(
        p1["mlp_extractor.policy_net.0.weight"][:, lo:hi]))
    out["adam_equal"] = _hash_arrays(adam_arrays(o0)) == _hash_arrays(adam_arrays(o1)) and \\
        o0["param_groups"] == o1["param_groups"]
    s0, s1 = pkl_stats(start_pkl), pkl_stats(saved_pkl)
    keep_np = np.ones(len(s0.obs_rms.mean), dtype=bool)
    keep_np[lo:hi] = False
    out["obs_rms_other_dims_equal"] = bool(np.array_equal(s0.obs_rms.mean[keep_np], s1.obs_rms.mean[keep_np])
                                           and np.array_equal(s0.obs_rms.var[keep_np], s1.obs_rms.var[keep_np])
                                           and s0.obs_rms.count == s1.obs_rms.count)
    out["obs_rms_slots_start"] = {"mean": s0.obs_rms.mean[lo:hi].tolist(), "var": s0.obs_rms.var[lo:hi].tolist()}
    out["obs_rms_slots_saved"] = {"mean": s1.obs_rms.mean[lo:hi].tolist(), "var": s1.obs_rms.var[lo:hi].tolist()}
    out["ret_rms_equal"] = bool(np.array_equal(s0.ret_rms.mean, s1.ret_rms.mean) and np.array_equal(
        s0.ret_rms.var, s1.ret_rms.var) and s0.ret_rms.count == s1.ret_rms.count)
    out["ok"] = bool(out["other_tensors_equal"] and out["value_net0_other_cols_equal"]
                     and out["value_net0_phase_cols_saved_max_abs"] == 0.0
                     and out["value_net0_phase_cols_start_max_abs"] > 0.0 and out["policy_net0_phase_cols_equal"]
                     and out["adam_equal"] and out["obs_rms_other_dims_equal"] and out["ret_rms_equal"]
                     and out["obs_rms_slots_saved"] == {"mean": [0.0] * 3, "var": [1.0] * 3})
    return out


def pkl_stats(pkl_path: str | os.PathLike[str]) -> Any:''')
rep('''    label = f"BS {arm} {seed}"''', '''    label = f"PS {arm} {seed}"''')
p.write_text(s)
print("ok")
