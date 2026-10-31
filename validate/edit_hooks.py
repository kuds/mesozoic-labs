"""PS build: the edits of bs_hooks.py (each old string asserted unique), run once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''"""Bias screen (BS): the ARM HOOK and its evidence.
''', '''"""Bias screen (BS), phase screen (PS): the ARM HOOK and its evidence.
''')
rep('''                       ret_rms keeps updating and the rewards are normalised as the run normalises them.
Importing this module needs neither torch nor the code copy (they are imported inside the functions).
"""''', '''                       ret_rms keeps updating and the rewards are normalised as the run normalises them.
  critic_phase (PS)    the critic alone also sees the episode phase (ramp, settled, discount of the env's own step
                       counter k: bs_common THE PHASE) in the observation's three command slots 60-62, which are
                       always zero in stance.  Five pieces, each a class swap on an instance or an in-place edit
                       verified here: (1) every TRAINING env instance's _get_obs writes the phase into slots 60-62
                       (the eval env and the probe's bank env are not touched); (2) the policy's mlp_extractor gets a
                       forward_actor that sets the normalised slots 60-62 to 0.0 before the policy MLP (the value the
                       actor always saw: the slots' statistics had mean 0), so every actor path (get_distribution,
                       forward, evaluate_actions, MirrorKLPPO's mirrored KL) is the start pair's actor and the actor's
                       first-layer phase columns get exactly zero gradient; (3) value_net.0.weight[:, 60:63] := 0 (their
                       Adam moments are checked to be exactly zero already: the slots never carried a value), so the
                       critic at the load is the start pair's on every input, then learns the phase (main's
                       widen_checkpoint.py precedent: zero columns for a new input, moments zero); (4) the training
                       VecNormalize's slots 60-62 are reseeded to mean 0 / var 1 (main's command_frame
                       .reseed_command_slice rule: a slot that was constant zero has var ~2.4e-11 and would normalise
                       a phase of 1 to 1e4, clipped at 10) and kept there after every update (the other 60 dims and the
                       count update as in control; it pickles and deep-copies as the stock RunningMeanStd); (5)
                       MirrorKLPPO's observation mirror map gets sign +1 on slots 60-62 (a phase is its own mirror
                       image; the command map flips v_y and yaw_rate).  install_probe_phase makes the policy probe feed
                       each bank state's own phase (kept step + 1; all ten are settled) to the critic.
Importing this module needs neither torch nor the code copy (they are imported inside the functions).
"""''')
rep('''ARMS = ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm")
GAE_LAMBDA = {"control": 0.95, "critic_blind_tail3": 0.95, "gae_lambda_1": 1.0, "frozen_obs_norm": 0.95}''',
    '''ARMS = ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm", "critic_phase")   # PS uses control and
GAE_LAMBDA = {"control": 0.95, "critic_blind_tail3": 0.95, "gae_lambda_1": 1.0,              # critic_phase only
              "frozen_obs_norm": 0.95, "critic_phase": 0.95}                                # (bs_common.ARMS)''')
rep('''VALUE_REL_TOL = 1e-5                            # buffer values against the critic on the same rows (batching noise)
''', '''VALUE_REL_TOL = 1e-5                            # buffer values against the critic on the same rows (batching noise)
# critic_phase (bs_common.OBS_DIM, PHASE_SLOTS, PHASE_FEATURES, SETTLE_STEPS, PHASE_GAMMA; repeated: no import)
OBS_DIM = 63
PHASE_SLOTS = (60, 61, 62)
PHASE_LO, PHASE_HI = 60, 63
PHASE_FEATURES = ("ramp", "settled", "discount")
SETTLE_STEPS = 200
PHASE_GAMMA = 0.98
PHASE_RESEED = (0.0, 1.0)                       # the frozen slots' (mean, var): main's reseed_command_slice rule
PHASE_PERTURB = (("zero", "set", 0.0), ("settled", "set", 1.0), ("plus1", "add", 1.0), ("set3", "set", 3.0),
                 ("flip", "flip", None))        # normalised units, on slots 60-62 together
PHASE_ACTOR_KEYS = ("dmu_phase_mode", "dmu_phase_forward_actions", "dmu_phase_log_prob", "dmu_phase_entropy",
                    "dmu_phase_mirrored")
PHASE_CRITIC_KEYS = ("dv_phase_predict_values", "dv_phase_forward", "dv_phase_evaluate_actions", "dv_phase_mirrored")
PHASE_STRUCT_KEYS = ("phase_rows", "phase_buffer_max_err", "phase_k_start_consistent", "phase_k_end_consistent",
                     "phase_k_min", "phase_k_max", "phase_rows_presettle", "phase_stats_frozen", "phase_rms_calls",
                     "mirror_phase_self_map", "mirror_phase_tensors_ok", "envs_swapped", "value_phase_cols_max_abs",
                     "policy_phase_cols_unchanged", "policy_phase_adam_max_abs", "n_updates_since_load",
                     "adam_steps_since_load")
''')
rep('''def frozen_counter(rms: Any) -> dict | None:
    c = getattr(type(rms), "bs_counter", None)
    return dict(c) if isinstance(c, dict) else None
''', '''def frozen_counter(rms: Any) -> dict | None:
    c = getattr(type(rms), "bs_counter", None)
    return dict(c) if isinstance(c, dict) else None


# ------------------------------------------------------------------------------------------------ critic_phase
def phase_tuple(k: int) -> tuple[float, float, float]:
    """(ramp, settled, discount) at episode step counter k: bs_common.phase_features, the same float arithmetic."""
    if isinstance(k, bool) or int(k) != k or int(k) < 0:
        raise ValueError(f"step counter {k!r} is not a non-negative integer")
    k = int(k)
    return (min(k, SETTLE_STEPS) / SETTLE_STEPS, 1.0 if k >= SETTLE_STEPS else 0.0,
            PHASE_GAMMA ** max(0, SETTLE_STEPS - k))


def phase_matrix(ks: Any) -> Any:
    """(n, 3) float64: phase_tuple of each counter in ks (element for element the same floats)."""
    import numpy as np

    flat = np.asarray(ks).reshape(-1)
    return np.array([phase_tuple(int(k)) for k in flat], dtype=np.float64).reshape(len(flat), 3)


def phase_env_class(base: type) -> type:
    """A subclass of a training env's class whose _get_obs writes the phase of the env's own step counter into the
    command slots 60-62 (the counter as the reward gates read it: 0 at reset, incremented before _get_obs)."""
    key = ("phase_env", base)
    if key not in _CLASS_CACHE:

        def _get_obs(self: Any) -> Any:
            import numpy as np

            obs = base._get_obs(self)
            if np.shape(obs) != (OBS_DIM,):
                raise HookError(f"critic_phase: an observation of shape {np.shape(obs)}, not ({OBS_DIM},)")
            command = getattr(self, "_command", None)
            if command is not None and np.any(np.asarray(command) != 0.0):
                raise HookError("critic_phase: a live command in the observation's command slots")
            obs[PHASE_LO:PHASE_HI] = phase_tuple(int(getattr(self, "_step_count", 0)))
            return obs

        _CLASS_CACHE[key] = type(f"PhaseObs{base.__name__}", (base,),
                                 {"_get_obs": _get_obs, "ps_phase_env": True, "bs_base": base,
                                  "__module__": base.__module__})
    return _CLASS_CACHE[key]


def critic_phase_class(base: type) -> type:
    """A subclass of the policy's MlpExtractor class whose forward_actor sees the normalised slots 60-62 as 0.0."""
    key = ("critic_phase", base)
    if key not in _CLASS_CACHE:

        def forward_actor(self: Any, features: Any) -> Any:
            masked = features.clone()            # never write into the caller's tensor (the critic reads it too)
            masked[..., PHASE_LO:PHASE_HI] = 0.0
            return base.forward_actor(self, masked)

        _CLASS_CACHE[key] = type(f"CriticPhase{base.__name__}", (base,),
                                 {"forward_actor": forward_actor, "ps_phase_slots": PHASE_SLOTS, "bs_base": base,
                                  "__module__": base.__module__})
    return _CLASS_CACHE[key]


def phase_rms_class(base: type) -> type:
    """A subclass of the observation RunningMeanStd class whose update() is the base's followed by putting slots
    60-62 back to PHASE_RESEED (mean 0, var 1); counted; pickles and deep-copies as the stock RunningMeanStd."""
    key = ("phase_rms", base)
    if key not in _CLASS_CACHE:
        from stable_baselines3.common.running_mean_std import RunningMeanStd

        counter = {"calls": 0, "rows": 0}

        def update(self: Any, arr: Any) -> None:
            import numpy as np

            base.update(self, arr)
            counter["calls"] += 1
            counter["rows"] += int(np.shape(arr)[0]) if np.ndim(arr) else 0
            reseed_phase_slots(self)

        def __reduce__(self: Any) -> Any:  # noqa: N807 - the pickle protocol's name
            state = {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}
            return (RunningMeanStd, (1e-4, tuple(self.mean.shape)), state)

        _CLASS_CACHE[key] = type(f"PhaseSlots{base.__name__}", (base,),
                                 {"update": update, "__reduce__": __reduce__, "ps_counter": counter,
                                  "bs_base": base, "__module__": base.__module__})
    return _CLASS_CACHE[key]


def reseed_phase_slots(rms: Any) -> None:
    """mean[60:63] = 0, var[60:63] = 1 on new arrays (never written into an array another object may hold)."""
    import numpy as np

    mean = np.array(rms.mean, dtype=np.float64, copy=True)
    var = np.array(rms.var, dtype=np.float64, copy=True)
    mean[PHASE_LO:PHASE_HI], var[PHASE_LO:PHASE_HI] = PHASE_RESEED
    rms.mean, rms.var = mean, var


def phase_slots_frozen(rms: Any) -> bool:
    import numpy as np

    return bool(np.all(np.asarray(rms.mean)[PHASE_LO:PHASE_HI] == PHASE_RESEED[0])
                and np.all(np.asarray(rms.var)[PHASE_LO:PHASE_HI] == PHASE_RESEED[1]))


def phase_counter(rms: Any) -> dict | None:
    c = getattr(type(rms), "ps_counter", None)
    return dict(c) if isinstance(c, dict) else None


def phase_mirror_map(mirror_map: Any) -> Any:
    """The run's mirror map with slots 60-62 mapped to themselves with sign +1 (a phase is its own mirror image)."""
    import dataclasses
    from types import SimpleNamespace

    import numpy as np

    obs = mirror_map.obs
    perm = np.array(obs.perm, dtype=np.int64, copy=True)
    sign = np.array(obs.sign, dtype=np.float64, copy=True)
    if perm.shape != (OBS_DIM,) or any(int(perm[i]) != i for i in PHASE_SLOTS):
        raise HookError(f"the observation mirror map moves a command slot (perm {perm[PHASE_LO:PHASE_HI].tolist()})")
    sign[PHASE_LO:PHASE_HI] = 1.0
    if dataclasses.is_dataclass(obs) and dataclasses.is_dataclass(mirror_map):
        names = list(obs.names) if obs.names else [""] * OBS_DIM
        rules = list(obs.rules) if obs.rules else [""] * OBS_DIM
        for i, f in zip(PHASE_SLOTS, PHASE_FEATURES, strict=True):
            names[i] = f"critic_phase.{f}"
            rules[i] = "critic_phase (PS): a function of the episode step counter, its own mirror image: sign +1"
        new_obs = type(obs)(perm=perm, sign=sign, names=tuple(names), rules=tuple(rules))
        return dataclasses.replace(mirror_map, obs=new_obs)
    return SimpleNamespace(**{**vars(mirror_map), "obs": SimpleNamespace(perm=perm, sign=sign, dim=OBS_DIM)})


def mirror_phase_self_map(omap: Any) -> bool:
    import numpy as np

    if omap is None:
        return False
    perm, sign = np.asarray(omap.perm), np.asarray(omap.sign)
    return bool(all(int(perm[i]) == i for i in PHASE_SLOTS) and np.all(sign[PHASE_LO:PHASE_HI] == 1.0))


def train_envs(train_env: Any) -> list:
    """The training VecNormalize's env instances (unwrapped): DummyVecEnv.envs (HookError if there are none)."""
    envs = getattr(getattr(train_env, "venv", None), "envs", None)
    if not isinstance(envs, list) or not envs:
        raise HookError("the training VecNormalize does not wrap a DummyVecEnv with env instances")
    return [getattr(e, "unwrapped", e) for e in envs]


def env_step_counts(train_env: Any) -> list[int]:
    """Each training env's episode step counter now (read only)."""
    return [int(e._step_count) for e in train_envs(train_env)]


def _adam_state(policy: Any, param: Any) -> dict:
    st = getattr(policy, "optimizer", None)
    st = st.state.get(param) if st is not None else None
    return st if isinstance(st, dict) else {}


def _adam_phase_max_abs(policy: Any, param: Any) -> float:
    st = _adam_state(policy, param)
    vals = [float(st[k][:, PHASE_LO:PHASE_HI].abs().max()) for k in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq")
            if k in st and hasattr(st[k], "abs")]
    return max(vals) if vals else 0.0


def _adam_steps(policy: Any) -> float:
    opt = getattr(policy, "optimizer", None)
    steps = [float(v["step"]) for v in (opt.state.values() if opt is not None else []) if "step" in v]
    return max(steps) if steps else 0.0


def phase_reference(model: Any) -> dict:
    """What critic_phase's evidence compares with (taken right after install_arm): the actor's phase columns, the
    update and Adam step counts at the load (not JSON: kept by the caller in memory)."""
    w = model.policy.mlp_extractor.policy_net[0].weight
    return {"policy_phase_cols": w[:, PHASE_LO:PHASE_HI].detach().clone(),
            "n_updates": int(getattr(model, "_n_updates", 0)), "adam_steps": _adam_steps(model.policy)}


def phase_probe_class(base: type) -> type:
    """A subclass of the policy probe callback whose per-update row feeds each bank state's own phase (k = kept step
    + 1: the observation kept after step index s carries _step_count s + 1) into the bank's command slots for the
    duration of the row (the bank the file records, and compares with a prior session's, is left as built)."""
    key = ("phase_probe", base)
    if key not in _CLASS_CACHE:

        def _row(self: Any) -> Any:
            import numpy as np

            bank = self.bank
            fed = np.array(bank.raw_obs, dtype=np.float64, copy=True)
            fed[:, PHASE_LO:PHASE_HI] = phase_matrix(np.asarray(bank.steps, dtype=np.int64) + 1)
            self.bank = bank._replace(raw_obs=fed)
            try:
                return base._row(self)
            finally:
                self.bank = bank

        _CLASS_CACHE[key] = type(f"PhaseFed{base.__name__}", (base,),
                                 {"_row": _row, "ps_phase_fed": True, "bs_base": base, "__module__": base.__module__})
    return _CLASS_CACHE[key]


def probe_phase(arm: str, callbacks: list) -> dict:
    """At training start (after the probe built its bank): critic_phase swaps the one PolicyProbeCallback to
    phase_probe_class; every other arm only checks the probe is the stock one.  Returns the record (which phase the
    probe feeds, per bank row).  HookError when the probe is not as expected."""
    import numpy as np

    probes = [c for c in callbacks if any(k.__name__ == "PolicyProbeCallback" for k in type(c).__mro__)]
    if len(probes) != 1:
        raise HookError(f"{len(probes)} policy probe callbacks, expected 1")
    probe = probes[0]
    bank = getattr(probe, "bank", None)
    if bank is None:
        raise HookError("the policy probe built no state bank")
    if getattr(type(probe), "ps_phase_fed", False):
        raise HookError("the policy probe already feeds a phase")
    k = (np.asarray(bank.steps, dtype=np.int64) + 1).tolist()
    rec = {"arm": arm, "class_before": type(probe).__name__, "labels": list(bank.labels),
           "kept_steps": np.asarray(bank.steps).tolist(), "k": k, "installed": False, "phase_fed": None}
    if arm == "critic_phase":
        probe.__class__ = phase_probe_class(type(probe))
        rec.update(installed=True, phase_fed=phase_matrix(k).tolist(),
                   settled=[bool(x >= SETTLE_STEPS) for x in k],
                   rule="each bank row's critic input carries phase(kept step + 1) in slots 60-62; the actor's are "
                        "masked; the file's bank_raw_obs stays as built")
    rec["class"] = type(probe).__name__
    return rec
''')
# install_arm: the critic_phase branch, before the final rec.update
rep('''        rec["vecnormalize_training"] = bool(train_env.training)
    rec.update(mlp_extractor_class=type(policy.mlp_extractor).__name__, obs_rms_class=type(train_env.obs_rms).__name__)
    return rec
''', '''        rec["vecnormalize_training"] = bool(train_env.training)
    if arm == "critic_phase":
        rec.update(_install_critic_phase(model, train_env))
    rec.update(mlp_extractor_class=type(policy.mlp_extractor).__name__, obs_rms_class=type(train_env.obs_rms).__name__)
    return rec


def _install_critic_phase(model: Any, train_env: Any) -> dict:
    """critic_phase's five pieces (module docstring), each checked before and after; HookError on any deviation
    (nothing has trained: the caller refuses before learn())."""
    import numpy as np
    import torch
    from stable_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor

    policy = model.policy
    ext = policy.mlp_extractor
    extractors = [policy.features_extractor] if policy.share_features_extractor else [
        policy.pi_features_extractor, policy.vf_features_extractor]
    if any(type(x) is not FlattenExtractor for x in extractors):
        raise HookError("the features extractor is not SB3's FlattenExtractor: features 60-62 might not be slots 60-62")
    obs_dim = int(model.observation_space.shape[0])
    w_pi, w_vf = ext.policy_net[0].weight, ext.value_net[0].weight
    if obs_dim != OBS_DIM or w_pi.shape[1] != OBS_DIM or w_vf.shape[1] != OBS_DIM:
        raise HookError(f"observation width {obs_dim}, first layers {w_pi.shape[1]} / {w_vf.shape[1]}, not {OBS_DIM}")
    state = getattr(model, "_mirror", None)
    omap = _mirror_obs_map(model)
    if state is None or omap is None:
        raise HookError("critic_phase needs the run's MirrorKLPPO mirror state (the mirrored critic must see the phase)")
    adam_vf, adam_pi = _adam_phase_max_abs(policy, w_vf), _adam_phase_max_abs(policy, w_pi)
    if adam_vf != 0.0 or adam_pi != 0.0:
        raise HookError(f"the first layers' Adam moments on slots 60-62 are not zero (critic {adam_vf}, actor "
                        f"{adam_pi}): the slots carried a value during training")
    rms = train_env.obs_rms
    if not (getattr(train_env, "training", False) and getattr(train_env, "norm_obs", False)):
        raise HookError("the training VecNormalize does not train / normalise observations")
    if isinstance(rms, dict) or any(getattr(type(rms), a, None) is not None for a in ("bs_counter", "ps_counter")):
        raise HookError(f"obs_rms {type(rms).__name__} cannot take the phase slots")
    if not np.all(np.asarray(rms.mean)[PHASE_LO:PHASE_HI] == 0.0):
        raise HookError(f"the slots' running mean is {np.asarray(rms.mean)[PHASE_LO:PHASE_HI].tolist()}, not 0: "
                        "they carried a value")
    envs = train_envs(train_env)
    for e in envs:
        if getattr(type(e), "ps_phase_env", False):
            raise HookError("a training env already carries the phase")
        if getattr(e, "direction_controller", None) is not None or np.any(np.asarray(getattr(e, "_command", 0)) != 0):
            raise HookError("a training env has a live command: its command slots are not free")
    if type(ext) is not MlpExtractor:
        raise HookError(f"the policy's mlp_extractor is {type(ext).__name__}, not MlpExtractor")
    rec: dict[str, Any] = {
        "phase_slots": list(PHASE_SLOTS), "phase_features": list(PHASE_FEATURES), "settle_steps": SETTLE_STEPS,
        "phase_gamma": PHASE_GAMMA, "obs_rms_slots_before": {"mean": np.asarray(rms.mean)[PHASE_LO:PHASE_HI].tolist(),
                                                             "var": np.asarray(rms.var)[PHASE_LO:PHASE_HI].tolist()},
        "value_phase_cols_max_abs_before": float(w_vf[:, PHASE_LO:PHASE_HI].abs().max()),
        "policy_phase_cols_max_abs": float(w_pi[:, PHASE_LO:PHASE_HI].abs().max()),
        "adam_phase_moments_max_abs_before": {"critic": adam_vf, "actor": adam_pi},
        "mirror_obs_sign_slots_before": np.asarray(omap.sign)[PHASE_LO:PHASE_HI].tolist(),
        "n_train_envs": len(envs), "train_env_class_before": type(envs[0]).__name__}
    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    opt_before = _hash([(f"{i}.{n}", np.asarray(v[n].detach().cpu() if hasattr(v[n], "detach") else v[n]))
                        for i, v in sorted(policy.optimizer.state_dict().get("state", {}).items()) for n in sorted(v)])
    # (3) the critic's phase columns := 0 (the widen precedent's zero columns)
    with torch.no_grad():
        w_vf[:, PHASE_LO:PHASE_HI] = 0.0
    # (4) the slots' statistics: reseeded and frozen (the count and the other dims update as in control)
    reseed_phase_slots(rms)
    rms.__class__ = phase_rms_class(type(rms))
    # (2) the actor's mask
    ext.__class__ = critic_phase_class(MlpExtractor)
    # (1) the training envs' observations carry the phase
    for e in envs:
        e.__class__ = phase_env_class(type(e))
    # (5) the mirror map: slots 60-62 map to themselves with sign +1; MirrorKLPPO rebuilds its tensors from it
    state.mirror_map = phase_mirror_map(state.mirror_map)
    if getattr(state, "tensors", None):
        state.tensors = {}
    # the transform, verified: every tensor equal except value_net.0.weight[:, 60:63], now exactly zero
    after = policy.state_dict()
    name = "mlp_extractor.value_net.0.weight"
    if list(before) != list(after):
        raise HookError("the critic_phase hook changed the policy's state_dict keys")
    for k in before:
        a, b = before[k], after[k]
        if k == name:
            keep = torch.ones(a.shape[1], dtype=torch.bool)
            keep[PHASE_LO:PHASE_HI] = False
            if not (a[:, keep].equal(b[:, keep]) and torch.all(b[:, PHASE_LO:PHASE_HI] == 0.0)):
                raise HookError("the critic's first layer changed outside slots 60-62 or is not zero on them")
        elif not a.equal(b):
            raise HookError(f"the critic_phase hook changed {k}")
    opt_after = _hash([(f"{i}.{n}", np.asarray(v[n].detach().cpu() if hasattr(v[n], "detach") else v[n]))
                       for i, v in sorted(policy.optimizer.state_dict().get("state", {}).items()) for n in sorted(v)])
    if opt_after != opt_before:
        raise HookError("the critic_phase hook changed the optimizer state")
    rec.update(transform_verified=True, obs_rms_slots_after={"mean": list(PHASE_RESEED[:1]) * 3,
                                                             "var": list(PHASE_RESEED[1:]) * 3},
               train_env_class=type(envs[0]).__name__, mirror_obs_sign_slots_after=[1.0, 1.0, 1.0],
               mirror_map_sha256_source=getattr(state.mirror_map, "sha256", None))
    return rec
''')
# arm_state_problems
rep('''    p = []
    ext = type(model.policy.mlp_extractor)
    blind = ext is critic_blind_class(MlpExtractor)
    if (arm == "critic_blind_tail3") != blind or (not blind and ext is not MlpExtractor):
        p.append(f"mlp_extractor class {ext.__name__} is not the {arm} one")
    frozen = getattr(type(train_env.obs_rms), "bs_counter", None) is not None
    if (arm == "frozen_obs_norm") != frozen:
        p.append(f"obs_rms class {type(train_env.obs_rms).__name__} is not the {arm} one")''', '''    p = []
    ext = type(model.policy.mlp_extractor)
    want = {"critic_blind_tail3": critic_blind_class(MlpExtractor),
            "critic_phase": critic_phase_class(MlpExtractor)}.get(arm, MlpExtractor)
    if ext is not want:
        p.append(f"mlp_extractor class {ext.__name__} is not the {arm} one")
    frozen = getattr(type(train_env.obs_rms), "bs_counter", None) is not None
    if (arm == "frozen_obs_norm") != frozen:
        p.append(f"obs_rms class {type(train_env.obs_rms).__name__} is not the {arm} one")
    phased = getattr(type(train_env.obs_rms), "ps_counter", None) is not None
    if (arm == "critic_phase") != phased:
        p.append(f"obs_rms class {type(train_env.obs_rms).__name__} is not the {arm} one (phase slots)")
    if arm == "critic_phase" and not phase_slots_frozen(train_env.obs_rms):
        p.append("the phase slots' statistics are not mean 0 / var 1")
    if arm == "critic_phase" and not mirror_phase_self_map(_mirror_obs_map(model)):
        p.append("the mirror map does not map slots 60-62 to themselves with sign +1")
    try:
        swapped = [bool(getattr(type(e), "ps_phase_env", False)) for e in train_envs(train_env)]
    except HookError:
        swapped = None
    if arm == "critic_phase" and swapped is None:
        p.append("no training env instances to check for the phase")
    if swapped is not None and any(swapped) != (arm == "critic_phase") or (swapped and arm == "critic_phase"
                                                                          and not all(swapped)):
        p.append(f"training envs carrying the phase {swapped} do not fit arm {arm}")''')
# evidence: phase dependence, buffer check
rep('''def rollout_evidence(arm: str, model: Any, train_env: Any, locals_: dict, *, start_obs_digest: str | None,
                     rows: int = EVIDENCE_ROWS) -> dict:''', '''def _phase_perturbed(obs: Any, kind: str, value: float | None) -> Any:
    out = obs.clone()
    if kind == "add":
        out[:, PHASE_LO:PHASE_HI] = out[:, PHASE_LO:PHASE_HI] + value
    elif kind == "flip":
        out[:, PHASE_LO:PHASE_HI] = 1.0 - out[:, PHASE_LO:PHASE_HI]
    else:
        out[:, PHASE_LO:PHASE_HI] = value
    return out


def phase_dependence(policy: Any, obs: Any, actions: Any, mirror_obs_map: Any = None) -> dict:
    """Max |change| of the actor (the mean, the deterministic actions, the log-probability of the given actions, the
    entropy, the mirrored observations' mean) and of the critic (predict_values, forward, evaluate_actions, the
    mirrored observations' predict_values) when the normalised slots 60-62 are perturbed PHASE_PERTURB's five ways.
    no_grad, deterministic; the policy's action_dist is put back as found."""
    stored = dict(vars(policy.action_dist))
    try:
        return _phase_dependence(policy, obs, actions, mirror_obs_map)
    finally:
        vars(policy.action_dist).clear()
        vars(policy.action_dist).update(stored)


def _phase_dependence(policy: Any, obs: Any, actions: Any, omap: Any) -> dict:
    import torch

    def mx(a: Any, b: Any) -> float:
        return float((a.flatten() - b.flatten()).abs().max())

    out = {k: 0.0 for k in (*PHASE_ACTOR_KEYS, *PHASE_CRITIC_KEYS)}
    with torch.no_grad():
        mu0 = policy.get_distribution(obs).mode()
        a0, vf0, lpf0 = policy.forward(obs, deterministic=True)
        ve0, lp0, ent0 = policy.evaluate_actions(obs, actions)
        pv0 = policy.predict_values(obs)
        om = None
        if omap is not None:
            om = obs[:, torch.as_tensor(omap.perm)] * torch.as_tensor(omap.sign, dtype=obs.dtype)
            mm0, mv0 = policy.get_distribution(om).mode(), policy.predict_values(om)
        for _, kind, val in PHASE_PERTURB:
            x = _phase_perturbed(obs, kind, val)
            a, vf, lpf = policy.forward(x, deterministic=True)
            ve, lp, ent = policy.evaluate_actions(x, actions)
            got = {"dmu_phase_mode": mx(policy.get_distribution(x).mode(), mu0),
                   "dmu_phase_forward_actions": max(mx(a, a0), mx(lpf, lpf0)), "dmu_phase_log_prob": mx(lp, lp0),
                   "dmu_phase_entropy": mx(ent, ent0) if ent is not None else 0.0,
                   "dv_phase_predict_values": mx(policy.predict_values(x), pv0), "dv_phase_forward": mx(vf, vf0),
                   "dv_phase_evaluate_actions": mx(ve, ve0)}
            if om is not None:
                xm = _phase_perturbed(om, kind, val)
                got.update(dmu_phase_mirrored=mx(policy.get_distribution(xm).mode(), mm0),
                           dv_phase_mirrored=mx(policy.predict_values(xm), mv0))
            for k, v in got.items():
                out[k] = max(out[k], v)
    if om is None:
        out.update(dmu_phase_mirrored=None, dv_phase_mirrored=None)
    return out


def phase_buffer_check(buf: Any, train_env: Any, k0: list, dones: Any, k_now: list) -> dict:
    """critic_phase: every buffer row's normalised slots 60-62 against VecNormalize's own formula applied to the phase
    of the row's step counter, reconstructed from the envs' counters at the rollout's start (k0) and the buffer's
    episode starts; the reconstruction must also reach the envs' counters now (k_now, after the last step)."""
    import numpy as np

    starts = np.asarray(buf.episode_starts) > 0
    n, n_env = starts.shape
    k = np.zeros((n, n_env), dtype=np.int64)
    k0 = np.asarray(k0, dtype=np.int64)
    k[0] = k0
    for t in range(1, n):
        k[t] = np.where(starts[t], 0, k[t - 1] + 1)
    end = np.where(np.asarray(dones, dtype=bool), 0, k[n - 1] + 1)
    rms = train_env.obs_rms
    raw = phase_matrix(k.reshape(-1)).astype(np.float32).reshape(n, n_env, 3)
    mean, var = np.asarray(rms.mean)[PHASE_LO:PHASE_HI], np.asarray(rms.var)[PHASE_LO:PHASE_HI]
    expect = np.clip((raw - mean) / np.sqrt(var + float(train_env.epsilon)), -float(train_env.clip_obs),
                     float(train_env.clip_obs)).astype(np.float32)
    got = np.asarray(buf.observations)[..., PHASE_LO:PHASE_HI]
    return {"phase_rows": int(n * n_env),
            "phase_buffer_max_err": float(np.max(np.abs(got.astype(np.float64) - expect.astype(np.float64)))),
            "phase_k_start_consistent": bool(np.array_equal(starts[0], k0 == 0)),
            "phase_k_end_consistent": bool(np.array_equal(end, np.asarray(k_now, dtype=np.int64))),
            "phase_k_min": int(k.min()), "phase_k_max": int(k.max()),
            "phase_rows_presettle": int((k < SETTLE_STEPS).sum())}


def phase_state(model: Any, train_env: Any, ref: dict) -> dict:
    """critic_phase's per-rollout state: the weights' phase columns, the statistics' slots, the mirror map, the envs."""
    import torch

    policy = model.policy
    w_pi = policy.mlp_extractor.policy_net[0].weight
    w_vf = policy.mlp_extractor.value_net[0].weight
    st = getattr(model, "_mirror", None)
    tensors = getattr(st, "tensors", None) or {}
    tens_ok = None
    if tensors:
        tens_ok = bool(torch.all(tensors["obs_sign"][PHASE_LO:PHASE_HI] == 1.0)
                       and torch.equal(tensors["obs_perm"][PHASE_LO:PHASE_HI], torch.arange(PHASE_LO, PHASE_HI)))
    try:
        envs_ok = all(getattr(type(e), "ps_phase_env", False) for e in train_envs(train_env))
    except HookError:
        envs_ok = False
    pc = phase_counter(train_env.obs_rms)
    return {"value_phase_cols_max_abs": float(w_vf[:, PHASE_LO:PHASE_HI].detach().abs().max()),
            "policy_phase_cols_unchanged": bool(torch.equal(w_pi[:, PHASE_LO:PHASE_HI].detach(),
                                                            ref["policy_phase_cols"])),
            "policy_phase_adam_max_abs": _adam_phase_max_abs(policy, w_pi),
            "n_updates_since_load": int(getattr(model, "_n_updates", 0)) - int(ref["n_updates"]),
            "adam_steps_since_load": _adam_steps(policy) - float(ref["adam_steps"]),
            "phase_stats_frozen": phase_slots_frozen(train_env.obs_rms),
            "phase_rms_calls": pc["calls"] if pc else None,
            "mirror_phase_self_map": mirror_phase_self_map(_mirror_obs_map(model)),
            "mirror_phase_tensors_ok": tens_ok, "envs_swapped": envs_ok}


def rollout_evidence(arm: str, model: Any, train_env: Any, locals_: dict, *, start_obs_digest: str | None,
                     rows: int = EVIDENCE_ROWS, phase_ctx: dict | None = None) -> dict:''')
rep('''        "digest_advantages": _hash([("advantages", buf.advantages), ("returns", buf.returns)]),
    }''', '''        "digest_advantages": _hash([("advantages", buf.advantages), ("returns", buf.returns)]),
        # PS: the env side without the phase slots (critic_phase's observations carry the phase there; with train()
        # a no-op and the same seed this digest is control's), and the statistics without them
        "digest_env_side_no_phase": _hash([("obs", np.asarray(buf.observations)[..., :PHASE_LO]),
                                           ("actions", buf.actions), ("episode_starts", buf.episode_starts)]),
    }''')
rep('''    ev.update(value_dependence(policy, obs, acts, _mirror_obs_map(model)))''',
    '''    ev.update(value_dependence(policy, obs, acts, _mirror_obs_map(model)))
    ev.update(phase_dependence(policy, obs, acts, _mirror_obs_map(model)))    # PS: every arm (control: the actor
    if arm == "critic_phase":                                                # responds to the slots; critic_phase:
        if not phase_ctx or "k0" not in phase_ctx or "ref" not in phase_ctx:  # exactly not)
            raise HookError("critic_phase evidence needs the rollout's start counters and the load reference")
        ev.update(phase_buffer_check(buf, train_env, phase_ctx["k0"], dones, env_step_counts(train_env)))
        ev.update(phase_state(model, train_env, phase_ctx["ref"]))
    else:
        ev.update({k: None for k in PHASE_STRUCT_KEYS})''')
rep('''    ev.update(obs_rms_digest=rms_digest(rms), obs_rms_equal_start=(rms_digest(rms) == start_obs_digest),''',
    '''    ev.update(obs_rms_digest_no_phase=_hash([("mean", np.asarray(rms.mean)[:PHASE_LO]),
                                             ("var", np.asarray(rms.var)[:PHASE_LO]), ("count", np.float64(rms.count))]))
    ev.update(obs_rms_digest=rms_digest(rms), obs_rms_equal_start=(rms_digest(rms) == start_obs_digest),''')
rep('''    out["steps_first_last"] = [evidence[0]["num_timesteps"], evidence[-1]["num_timesteps"]]
    return out
''', '''    out["steps_first_last"] = [evidence[0]["num_timesteps"], evidence[-1]["num_timesteps"]]
    out.update(_phase_summary(evidence))
    return out


def _phase_summary(evidence: list[dict]) -> dict:
    """PS: the phase evidence's summary (pure).  Actor: the max over its five paths; critic: per path.  critic_phase
    rows split by n_updates_since_load: 0 (the zero-initialised columns: no response allowed) and >= 1 (after at
    least one optimizer step: a response required)."""
    out: dict[str, Any] = {}
    acts = [max(e[k] for k in PHASE_ACTOR_KEYS if isinstance(e.get(k), (int, float))) for e in evidence
            if any(isinstance(e.get(k), (int, float)) for k in PHASE_ACTOR_KEYS)]
    out["max_dmu_phase"] = max(acts) if acts else None
    out["min_dmu_phase"] = min(acts) if acts else None
    for k in PHASE_CRITIC_KEYS:
        vals = [e[k] for e in evidence if isinstance(e.get(k), (int, float))]
        out[f"max_{k}"] = max(vals) if vals else None
        out[f"min_{k}"] = min(vals) if vals else None
    pe = [e for e in evidence if e.get("phase_buffer_max_err") is not None]
    out["phase_rows_recorded"] = len(pe)
    if not pe:
        return out
    out["max_phase_buffer_max_err"] = max(e["phase_buffer_max_err"] for e in pe)
    out["phase_all_k_consistent"] = all(e["phase_k_start_consistent"] and e["phase_k_end_consistent"] for e in pe)
    for key in ("phase_stats_frozen", "mirror_phase_self_map", "policy_phase_cols_unchanged", "envs_swapped"):
        out[f"phase_all_{key}"] = all(e.get(key) is True for e in pe)
    out["phase_all_mirror_tensors_ok"] = all(e.get("mirror_phase_tensors_ok") in (None, True) for e in pe)
    out["max_policy_phase_adam_max_abs"] = max(e["policy_phase_adam_max_abs"] for e in pe)
    flags = [e["n_updates_since_load"] >= 1 for e in pe]
    out["phase_rows_ordered"] = flags == sorted(flags)
    zero, upd = [e for e in pe if e["n_updates_since_load"] == 0], [e for e in pe if e["n_updates_since_load"] >= 1]
    out["phase_n_zero_init"], out["phase_n_updated"] = len(zero), len(upd)
    out["phase_n_negative_updates"] = sum(1 for e in pe if e["n_updates_since_load"] < 0)
    out["zero_init_max_value_phase_cols"] = max((e["value_phase_cols_max_abs"] for e in zero), default=None)
    out["zero_init_max_dv_phase"] = max((e[k] for e in zero for k in PHASE_CRITIC_KEYS
                                         if isinstance(e.get(k), (int, float))), default=None)
    out["updated_min_value_phase_cols"] = min((e["value_phase_cols_max_abs"] for e in upd), default=None)
    out["updated_min_dv_phase"] = {k: min((e[k] for e in upd if isinstance(e.get(k), (int, float))), default=None)
                                   for k in PHASE_CRITIC_KEYS}
    out["updated_min_adam_steps_since_load"] = min((e["adam_steps_since_load"] for e in upd), default=None)
    out["phase_rows_presettle_total"] = sum(e["phase_rows_presettle"] for e in pe)
    out["phase_k_max"] = max(e["phase_k_max"] for e in pe)
    return out
''')
rep('''    if arm == "frozen_obs_norm":
        if not s.get("obs_rms_equal_start_all"):''', '''    p += _phase_problems(arm, s, n)
    if arm == "frozen_obs_norm":
        if not s.get("obs_rms_equal_start_all"):''')
rep('''        if n > 1 and not s.get("obs_rms_distinct_digests", 0) >= n:
            p.append(f"evidence: {arm}: the observation statistics did not move every rollout")
    return p
''', '''        if n > 1 and not s.get("obs_rms_distinct_digests", 0) >= n:
            p.append(f"evidence: {arm}: the observation statistics did not move every rollout")
    return p


def _phase_problems(arm: str, s: dict, n: int) -> list[str]:
    """PS: critic_phase's phase evidence; every other arm: no phase evidence and an actor that responds to the
    command slots (it is not masked)."""
    p: list[str] = []
    if arm != "critic_phase":
        if s.get("phase_rows_recorded") != 0:
            p.append(f"evidence: {arm}: phase evidence recorded on {s.get('phase_rows_recorded')} rollouts")
        if not (isinstance(s.get("min_dmu_phase"), (int, float)) and s["min_dmu_phase"] > 0.0):
            p.append(f"evidence: {arm}: the actor did not respond to slots 60-62 on some rollout (masked?)")
        return p
    if s.get("phase_rows_recorded") != n:
        p.append(f"evidence: critic_phase: phase evidence on {s.get('phase_rows_recorded')} of {n} rollouts")
        return p
    if s.get("max_dmu_phase") != 0.0:
        p.append(f"evidence: critic_phase: the actor moved by {s.get('max_dmu_phase')} with the phase slots")
    if s.get("max_phase_buffer_max_err") != 0.0:
        p.append(f"evidence: critic_phase: buffer slots differ from the env counters' phase by "
                 f"{s.get('max_phase_buffer_max_err')}")
    for key in ("phase_all_k_consistent", "phase_all_phase_stats_frozen", "phase_all_mirror_phase_self_map",
                "phase_all_policy_phase_cols_unchanged", "phase_all_envs_swapped", "phase_all_mirror_tensors_ok",
                "phase_rows_ordered"):
        if s.get(key) is not True:
            p.append(f"evidence: critic_phase: {key} is {s.get(key)!r}")
    if s.get("max_policy_phase_adam_max_abs") != 0.0:
        p.append(f"evidence: critic_phase: the actor's phase columns have Adam moments "
                 f"{s.get('max_policy_phase_adam_max_abs')}")
    if s.get("phase_n_negative_updates") != 0 or not (s.get("phase_n_zero_init") or 0) >= 1:
        p.append(f"evidence: critic_phase: {s.get('phase_n_zero_init')} rollouts before the first update (>= 1: the "
                 f"first rollout precedes every train())")
    if s.get("phase_n_zero_init"):
        if s.get("zero_init_max_value_phase_cols") != 0.0 or s.get("zero_init_max_dv_phase") != 0.0:
            p.append(f"evidence: critic_phase: before any update the critic's phase columns "
                     f"({s.get('zero_init_max_value_phase_cols')}) or its response "
                     f"({s.get('zero_init_max_dv_phase')}) are not exactly zero")
    if s.get("phase_n_updated"):
        mins = s.get("updated_min_dv_phase") or {}
        if not ((s.get("updated_min_value_phase_cols") or 0.0) > 0.0
                and all(isinstance(mins.get(k), (int, float)) and mins[k] > 0.0 for k in PHASE_CRITIC_KEYS)):
            p.append(f"evidence: critic_phase: after updates the critic does not respond to the phase on every "
                     f"rollout and path (columns {s.get('updated_min_value_phase_cols')}, responses {mins})")
    return p
''')
p.write_text(s)
print("ok")
