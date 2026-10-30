"""Bias screen (BS): the ARM HOOK and its evidence.

    install_arm(arm, model, train_env)   applies an arm to a loaded model and its training VecNormalize, in place,
                                         after the run's own load (train_base._create_or_load_model, which also made
                                         the model a MirrorKLPPO) and before learn(); returns what it applied.
    arm_state_problems(arm, model, train_env)   the arm (and no other) in force: [] or the reasons.
    rollout_evidence(arm, model, train_env, locals_)   at a rollout's end (after GAE, before train()): what the arm
                                         did on that rollout (read only: no_grad forward passes, numpy; no sampling,
                                         no RNG draw, nothing written to the model, the buffer or the env).
    evidence_problems(arm, summary)      CHECK's reading of the evidence summary (pure).

Arms (bs_common.ARM_CHANGE says each in words; none is chosen here):
  control              nothing.
  critic_blind_tail3   the policy's mlp_extractor becomes a subclass (class swap on the instance; no parameter, no
                       state_dict key, no shape changes) whose forward_critic replaces the NORMALISED observation
                       index 6 by 0.0 before the value MLP.  SB3 2.9.0's ActorCriticPolicy.forward / evaluate_actions
                       (through MlpExtractor.forward) and predict_values all call mlp_extractor.forward_critic, so every
                       value
                       the run computes (the rollout's values, the bootstrap of the last and of time-limited states,
                       the PPO value loss and MirrorKLPPO's mirrored value loss) is the blinded critic's; the actor's
                       path (forward_actor on the unchanged features) is untouched.  The observation mirror map keeps
                       index 6 in place with sign +1 (checked), so the mirrored critic is blind to the same input.
  gae_lambda_1         nothing here: the stage config's [ppo] gae_lambda is 1.0 before the load (run_bs.arm_config);
                       this module only checks the loaded model and its rollout buffer carry it.
  frozen_obs_norm      the training VecNormalize's obs_rms (a SymmetricRunningMeanStd after enable_mirror_kl) becomes a
                       subclass whose update() is a counted no-op; it still pickles (and deep-copies) as the stock
                       RunningMeanStd with the frozen mean / var / count.  VecNormalize.training stays True, so
                       ret_rms keeps updating and the rewards are normalised as the run normalises them.
Importing this module needs neither torch nor the code copy (they are imported inside the functions).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any

OBS_TAIL3 = 6                                   # bs_common.OBS_TAIL3
ARMS = ("control", "critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm")
GAE_LAMBDA = {"control": 0.95, "critic_blind_tail3": 0.95, "gae_lambda_1": 1.0, "frozen_obs_norm": 0.95}
EVIDENCE_ROWS = 256                             # buffer rows the value-dependence check uses (the first 256)
PERTURB = (("plus1", "add", 1.0), ("minus_self", "neg", None), ("set3", "set", 3.0))
ADV_REL_TOL = 1e-4                              # |buffer advantage - recomputation| <= tol x max(1, max |recomputed|)
VALUE_REL_TOL = 1e-5                            # buffer values against the critic on the same rows (batching noise)


class HookError(Exception):
    """An arm could not be applied to the loaded objects as specified (the run refuses: no optimizer step)."""


# ------------------------------------------------------------------------------------------------ class factories
_CLASS_CACHE: dict[tuple[str, type], type] = {}


def critic_blind_class(base: type) -> type:
    """A subclass of the policy's MlpExtractor class whose forward_critic sees features[..., 6] = 0.0."""
    key = ("critic_blind", base)
    if key not in _CLASS_CACHE:

        def forward_critic(self: Any, features: Any) -> Any:
            blinded = features.clone()           # never write into the caller's tensor (the actor reads it too)
            blinded[..., OBS_TAIL3] = 0.0
            return base.forward_critic(self, blinded)

        _CLASS_CACHE[key] = type(f"CriticBlindTail3{base.__name__}", (base,),
                                 {"forward_critic": forward_critic, "bs_blind_index": OBS_TAIL3,
                                  "bs_base": base, "__module__": base.__module__})
    return _CLASS_CACHE[key]


def frozen_rms_class(base: type) -> type:
    """A subclass of the observation RunningMeanStd class whose update() is a counted no-op and which pickles as the
    stock RunningMeanStd (the base's own __reduce__ form: SymmetricRunningMeanStd.__reduce__)."""
    key = ("frozen", base)
    if key not in _CLASS_CACHE:
        from stable_baselines3.common.running_mean_std import RunningMeanStd

        counter = {"calls": 0, "rows": 0}

        def update(self: Any, arr: Any) -> None:
            counter["calls"] += 1
            counter["rows"] += int(getattr(arr, "shape", (0,))[0]) if getattr(arr, "shape", None) else 0

        def __reduce__(self: Any) -> Any:  # noqa: N807 - the pickle protocol's name
            state = {"mean": self.mean.copy(), "var": self.var.copy(), "count": float(self.count)}
            return (RunningMeanStd, (1e-4, tuple(self.mean.shape)), state)

        _CLASS_CACHE[key] = type(f"FrozenObs{base.__name__}", (base,),
                                 {"update": update, "__reduce__": __reduce__, "bs_counter": counter,
                                  "bs_base": base, "__module__": base.__module__})
    return _CLASS_CACHE[key]


def frozen_counter(rms: Any) -> dict | None:
    c = getattr(type(rms), "bs_counter", None)
    return dict(c) if isinstance(c, dict) else None


# ------------------------------------------------------------------------------------------------ digests
def _hash(arrays: list) -> str:
    import numpy as np

    d = hashlib.sha256()
    for name, a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        d.update(name.encode() + str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return "sha256:" + d.hexdigest()


def rms_digest(rms: Any) -> str:
    import numpy as np

    return _hash([("mean", rms.mean), ("var", rms.var), ("count", np.float64(rms.count))])


# ------------------------------------------------------------------------------------------------ install
def _mirror_obs_map(model: Any) -> Any:
    state = getattr(model, "_mirror", None)
    return getattr(getattr(state, "mirror_map", None), "obs", None)


def install_arm(arm: str, model: Any, train_env: Any) -> dict:
    """Apply *arm* in place; return the record of what was applied.  HookError (nothing half-applied that could
    train: the caller refuses before learn()) when the loaded objects are not what the hook is written for."""
    from stable_baselines3.common.torch_layers import FlattenExtractor, MlpExtractor

    if arm not in ARMS:
        raise HookError(f"unknown arm {arm!r}")
    policy = model.policy
    rec: dict[str, Any] = {"arm": arm, "policy_class": type(policy).__name__,
                           "mlp_extractor_class_before": type(policy.mlp_extractor).__name__,
                           "obs_rms_class_before": type(train_env.obs_rms).__name__,
                           "gae_lambda": float(model.gae_lambda),
                           "rollout_buffer_gae_lambda": float(model.rollout_buffer.gae_lambda)}
    want_lam = GAE_LAMBDA[arm]
    if float(model.gae_lambda) != want_lam or float(model.rollout_buffer.gae_lambda) != want_lam:
        raise HookError(f"arm {arm}: gae_lambda {model.gae_lambda} / buffer {model.rollout_buffer.gae_lambda} != "
                        f"{want_lam}")
    if type(policy.mlp_extractor) is not MlpExtractor:
        raise HookError(f"the policy's mlp_extractor is {type(policy.mlp_extractor).__name__}, not MlpExtractor")
    if arm == "critic_blind_tail3":
        extractors = [policy.features_extractor] if policy.share_features_extractor else [
            policy.pi_features_extractor, policy.vf_features_extractor]
        if any(type(x) is not FlattenExtractor for x in extractors):
            raise HookError("the features extractor is not SB3's FlattenExtractor: feature 6 might not be "
                            "observation 6")
        obs_dim = int(model.observation_space.shape[0])
        if policy.mlp_extractor.value_net[0].in_features != obs_dim or obs_dim <= OBS_TAIL3:
            raise HookError(f"the value MLP's input width {policy.mlp_extractor.value_net[0].in_features} is not the "
                            f"observation width {obs_dim}")
        omap = _mirror_obs_map(model)
        if omap is not None and (int(omap.perm[OBS_TAIL3]) != OBS_TAIL3 or float(omap.sign[OBS_TAIL3]) != 1.0):
            raise HookError(f"the observation mirror map moves index 6 (perm {omap.perm[OBS_TAIL3]}, sign "
                            f"{omap.sign[OBS_TAIL3]}): the mirrored critic would not be blind to the same input")
        before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        policy.mlp_extractor.__class__ = critic_blind_class(MlpExtractor)
        after = policy.state_dict()
        if list(before) != list(after) or any(not before[k].equal(after[k]) for k in before):
            raise HookError("the class swap changed the policy's state_dict")
        rec.update(blind_index=OBS_TAIL3, mirror_obs_perm_6=None if omap is None else int(omap.perm[OBS_TAIL3]),
                   mirror_obs_sign_6=None if omap is None else float(omap.sign[OBS_TAIL3]))
    if arm == "frozen_obs_norm":
        rms = train_env.obs_rms
        if not (getattr(train_env, "training", False) and getattr(train_env, "norm_obs", False)
                and getattr(train_env, "norm_reward", False)):
            raise HookError("the training VecNormalize does not train / normalise observations and rewards")
        if isinstance(rms, dict) or getattr(type(rms), "bs_counter", None) is not None:
            raise HookError(f"obs_rms {type(rms).__name__} cannot be frozen here")
        rec["obs_rms_digest_at_freeze"] = rms_digest(rms)
        rms.__class__ = frozen_rms_class(type(rms))
        rec["vecnormalize_training"] = bool(train_env.training)
    rec.update(mlp_extractor_class=type(policy.mlp_extractor).__name__, obs_rms_class=type(train_env.obs_rms).__name__)
    return rec


def arm_state_problems(arm: str, model: Any, train_env: Any) -> list[str]:
    """The arm's hook in force and no other arm's: [] or the reasons (cheap; class identities and gae_lambda)."""
    from stable_baselines3.common.torch_layers import MlpExtractor

    p = []
    ext = type(model.policy.mlp_extractor)
    blind = ext is critic_blind_class(MlpExtractor)
    if (arm == "critic_blind_tail3") != blind or (not blind and ext is not MlpExtractor):
        p.append(f"mlp_extractor class {ext.__name__} is not the {arm} one")
    frozen = getattr(type(train_env.obs_rms), "bs_counter", None) is not None
    if (arm == "frozen_obs_norm") != frozen:
        p.append(f"obs_rms class {type(train_env.obs_rms).__name__} is not the {arm} one")
    for what, lam in (("model", model.gae_lambda), ("rollout buffer", model.rollout_buffer.gae_lambda)):
        if float(lam) != GAE_LAMBDA[arm]:
            p.append(f"{what} gae_lambda {lam} != {GAE_LAMBDA[arm]}")
    if not getattr(train_env, "training", False):
        p.append("the training VecNormalize's training flag is off")
    return p


# ------------------------------------------------------------------------------------------------ evidence
def gae(rewards: Any, values: Any, episode_starts: Any, last_values: Any, dones: Any, gamma: float,
        lam: float) -> Any:
    """GAE(lam) advantages in float64, written independently of SB3's buffer (the same recursion as the paper:
    delta_t = r_t + gamma V_{t+1} (1 - start_{t+1}) - V_t; A_t = delta_t + gamma lam (1 - start_{t+1}) A_{t+1};
    the last step bootstraps with last_values unless dones).  With lam = 1 it equals the discounted return to the
    episode end or the rollout end (bootstrapped there with V) minus V: the bootstrapped Monte Carlo advantage."""
    import numpy as np

    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    st = np.asarray(episode_starts, dtype=np.float64)
    n = r.shape[0]
    adv = np.zeros_like(r)
    last = np.zeros(r.shape[1:], dtype=np.float64)
    for t in reversed(range(n)):
        if t == n - 1:
            nonterm = 1.0 - np.asarray(dones, dtype=np.float64)
            nxt = np.asarray(last_values, dtype=np.float64).reshape(r.shape[1:])
        else:
            nonterm = 1.0 - st[t + 1]
            nxt = v[t + 1]
        delta = r[t] + gamma * nxt * nonterm - v[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv


def mc_return(rewards: Any, episode_starts: Any, last_values: Any, dones: Any, gamma: float) -> Any:
    """The discounted return to the episode end, or to the rollout end bootstrapped with V there (float64), by the
    direct sum (not the GAE recursion): G_t = r_t + gamma (1 - start_{t+1}) G_{t+1}."""
    import numpy as np

    r = np.asarray(rewards, dtype=np.float64)
    st = np.asarray(episode_starts, dtype=np.float64)
    n = r.shape[0]
    G = np.zeros_like(r)
    nxt = (1.0 - np.asarray(dones, dtype=np.float64)) * np.asarray(last_values, dtype=np.float64).reshape(r.shape[1:])
    for t in reversed(range(n)):
        G[t] = r[t] + gamma * nxt
        nxt = (1.0 - st[t]) * G[t]
    return G


def _perturbed(obs: Any, kind: str, value: float | None) -> Any:
    out = obs.clone()
    if kind == "add":
        out[:, OBS_TAIL3] = out[:, OBS_TAIL3] + value
    elif kind == "neg":
        out[:, OBS_TAIL3] = -out[:, OBS_TAIL3]
    else:
        out[:, OBS_TAIL3] = value
    return out


def raw_values(policy: Any, obs: Any) -> Any:
    """The value MLP without any hook (MlpExtractor.forward_critic itself) on observations (no_grad by the caller)."""
    from stable_baselines3.common.torch_layers import MlpExtractor

    features = policy.extract_features(obs, policy.vf_features_extractor) if not policy.share_features_extractor \
        else policy.extract_features(obs)
    return policy.value_net(MlpExtractor.forward_critic(policy.mlp_extractor, features))


def value_dependence(policy: Any, obs: Any, actions: Any, mirror_obs_map: Any = None) -> dict:
    """Max |change| of the critic's output through each value path (predict_values, forward, evaluate_actions, the
    mirrored observations' predict_values) and of the actor's mean when observation 6 is perturbed three ways
    (+1, sign flip, set to 3: normalised units), on the given normalised observations.  no_grad, deterministic."""
    # get_distribution() / forward() / evaluate_actions() store the distribution they build on policy.action_dist;
    # the one the rollout left there is put back (the policy probe's own rule), so the policy is left as found.
    stored = dict(vars(policy.action_dist))
    try:
        return _value_dependence(policy, obs, actions, mirror_obs_map)
    finally:
        vars(policy.action_dist).clear()
        vars(policy.action_dist).update(stored)


def _value_dependence(policy: Any, obs: Any, actions: Any, mirror_obs_map: Any) -> dict:
    import torch

    out: dict[str, float] = {}
    with torch.no_grad():
        v0 = policy.predict_values(obs).flatten()
        _, f0, _ = policy.forward(obs, deterministic=True)
        e0, _, _ = policy.evaluate_actions(obs, actions)
        mu0 = policy.get_distribution(obs).mode()
        om = None
        if mirror_obs_map is not None:
            perm = torch.as_tensor(mirror_obs_map.perm)
            sign = torch.as_tensor(mirror_obs_map.sign, dtype=obs.dtype)
            om = obs[:, perm] * sign
            m0 = policy.predict_values(om).flatten()
        dv = df = de = dm = da = 0.0
        for _, kind, val in PERTURB:
            p = _perturbed(obs, kind, val)
            dv = max(dv, float((policy.predict_values(p).flatten() - v0).abs().max()))
            df = max(df, float((policy.forward(p, deterministic=True)[1].flatten() - f0.flatten()).abs().max()))
            de = max(de, float((policy.evaluate_actions(p, actions)[0].flatten() - e0.flatten()).abs().max()))
            da = max(da, float((policy.get_distribution(p).mode() - mu0).abs().max()))
            if om is not None:
                dm = max(dm, float((policy.predict_values(_perturbed(om, kind, val)).flatten() - m0).abs().max()))
        out.update(dv_predict_values=dv, dv_forward=df, dv_evaluate_actions=de, dmu_actor=da,
                   dv_mirrored=dm if om is not None else None)
    return out


def rollout_evidence(arm: str, model: Any, train_env: Any, locals_: dict, *, start_obs_digest: str | None,
                     rows: int = EVIDENCE_ROWS) -> dict:
    """What the arm did on the rollout that just ended (see the module docstring).  Read only."""
    import numpy as np
    import torch

    buf = model.rollout_buffer
    policy = model.policy
    last_values = locals_["values"]
    last_values = last_values.detach().cpu().numpy().flatten() if hasattr(last_values, "detach") else last_values
    dones = np.asarray(locals_["dones"])
    gamma = float(model.gamma)
    adv = np.asarray(buf.advantages, dtype=np.float64)
    ref095 = gae(buf.rewards, buf.values, buf.episode_starts, last_values, dones, gamma, 0.95)
    ref_mc = mc_return(buf.rewards, buf.episode_starts, last_values, dones, gamma) - np.asarray(buf.values, np.float64)
    ref_lam1 = gae(buf.rewards, buf.values, buf.episode_starts, last_values, dones, gamma, 1.0)
    scale = max(1.0, float(np.max(np.abs(ref_mc))), float(np.max(np.abs(ref095))))
    ev: dict[str, Any] = {
        "num_timesteps": int(model.num_timesteps), "buffer_gae_lambda": float(buf.gae_lambda),
        "model_gae_lambda": float(model.gae_lambda), "adv_scale": scale,
        "adv_err_vs_gae095": float(np.max(np.abs(adv - ref095))),
        "adv_err_vs_mc": float(np.max(np.abs(adv - ref_mc))),
        "mc_vs_gae_lambda1_recursion": float(np.max(np.abs(ref_mc - ref_lam1))),
        "gae095_vs_mc": float(np.max(np.abs(ref095 - ref_mc))),
        "returns_minus_adv_minus_values": float(np.max(np.abs(np.asarray(buf.returns, np.float64) - adv
                                                               - np.asarray(buf.values, np.float64)))),
        "episode_starts_in_rollout": int(np.asarray(buf.episode_starts).sum()),
        # digests of what the env produced and what the critic / GAE produced on this rollout: with train() a no-op
        # and the same seed, an arm that leaves the actor and the statistics alone reproduces the control's env side
        # (observations, actions, episode starts) bit for bit (V2's pairing); the buffer's rewards carry SB3's
        # time-limit bootstrap gamma V(terminal observation), so they follow the critic at truncated steps
        "digest_env_side": _hash([("obs", buf.observations), ("actions", buf.actions),
                                  ("episode_starts", buf.episode_starts)]),
        "digest_rewards": _hash([("rewards", buf.rewards)]),
        "digest_values": _hash([("values", buf.values), ("last_values", np.asarray(last_values))]),
        "digest_advantages": _hash([("advantages", buf.advantages), ("returns", buf.returns)]),
    }
    n_obs = buf.observations.shape[-1]
    obs = torch.as_tensor(np.asarray(buf.observations).reshape(-1, n_obs)[:rows], dtype=torch.float32)
    acts = torch.as_tensor(np.asarray(buf.actions).reshape(-1, buf.actions.shape[-1])[:rows], dtype=torch.float32)
    vals = np.asarray(buf.values, dtype=np.float64).reshape(-1)[:rows]
    ev.update(value_dependence(policy, obs, acts, _mirror_obs_map(model)))
    with torch.no_grad():                        # value paths only: action_dist is not touched
        v_eff = policy.predict_values(obs).flatten().double().numpy()
        v_raw = raw_values(policy, obs).flatten().double().numpy()
    vscale = max(1.0, float(np.max(np.abs(vals))))
    ev.update(buffer_values_vs_critic=float(np.max(np.abs(vals - v_eff))),
              buffer_values_vs_raw_value_mlp=float(np.max(np.abs(vals - v_raw))), value_scale=vscale)
    rms = train_env.obs_rms
    ev.update(obs_rms_digest=rms_digest(rms), obs_rms_equal_start=(rms_digest(rms) == start_obs_digest),
              obs_rms_count=float(rms.count), ret_rms_count=float(train_env.ret_rms.count),
              ret_rms_var=float(np.asarray(train_env.ret_rms.var).reshape(-1)[0]),
              frozen_update=frozen_counter(rms), state_problems=arm_state_problems(arm, model, train_env))
    return ev


SUMMARY_MAX = ("adv_err_vs_gae095", "adv_err_vs_mc", "mc_vs_gae_lambda1_recursion", "returns_minus_adv_minus_values",
               "dv_predict_values", "dv_forward", "dv_evaluate_actions", "dv_mirrored", "buffer_values_vs_critic")
SUMMARY_MIN = ("adv_err_vs_gae095", "adv_err_vs_mc", "gae095_vs_mc", "dmu_actor", "dv_predict_values",
               "buffer_values_vs_raw_value_mlp")


def summarize(evidence: list[dict]) -> dict:
    """The run_meta summary of the per-rollout evidence (pure)."""
    out: dict[str, Any] = {"n": len(evidence)}
    if not evidence:
        return out
    for k in SUMMARY_MAX:
        vals = [e[k] for e in evidence if isinstance(e.get(k), (int, float))]
        out[f"max_{k}"] = max(vals) if vals else None
    for k in SUMMARY_MIN:
        vals = [e[k] for e in evidence if isinstance(e.get(k), (int, float))]
        out[f"min_{k}"] = min(vals) if vals else None
    out["max_adv_scale"] = max(e["adv_scale"] for e in evidence)
    out["max_value_scale"] = max(e["value_scale"] for e in evidence)
    out["buffer_gae_lambda"] = sorted({e["buffer_gae_lambda"] for e in evidence})
    out["model_gae_lambda"] = sorted({e["model_gae_lambda"] for e in evidence})
    out["obs_rms_equal_start_all"] = all(e["obs_rms_equal_start"] for e in evidence)
    out["obs_rms_equal_start_any"] = any(e["obs_rms_equal_start"] for e in evidence)
    out["obs_rms_distinct_digests"] = len({e["obs_rms_digest"] for e in evidence})
    out["ret_rms_count_first_last"] = [evidence[0]["ret_rms_count"], evidence[-1]["ret_rms_count"]]
    out["obs_rms_count_first_last"] = [evidence[0]["obs_rms_count"], evidence[-1]["obs_rms_count"]]
    fz = [e["frozen_update"] for e in evidence if e.get("frozen_update")]
    out["frozen_update_calls_first_last"] = [fz[0]["calls"], fz[-1]["calls"]] if fz else None
    out["state_problems"] = sorted({p for e in evidence for p in e.get("state_problems") or []})
    out["steps_first_last"] = [evidence[0]["num_timesteps"], evidence[-1]["num_timesteps"]]
    return out


def evidence_problems(arm: str, s: dict, *, n_expected: int | None = None) -> list[str]:
    """CHECK's reading of a summary: [] when the evidence shows the arm, and no other, in force on every rollout."""
    p: list[str] = []
    if arm not in ARMS:
        return [f"evidence: unknown arm {arm!r}"]
    n = s.get("n")
    if not isinstance(n, int) or n < 1 or (n_expected is not None and n != n_expected):
        return [f"evidence: {n!r} rollouts recorded, expected {n_expected}"]
    if s.get("state_problems"):
        p.append(f"evidence: hook state problems {s['state_problems']}")
    tol = ADV_REL_TOL * float(s.get("max_adv_scale") or math.inf)
    vtol = VALUE_REL_TOL * float(s.get("max_value_scale") or math.inf)
    lam = GAE_LAMBDA[arm]
    if s.get("buffer_gae_lambda") != [lam] or s.get("model_gae_lambda") != [lam]:
        p.append(f"evidence: gae_lambda buffer {s.get('buffer_gae_lambda')} model {s.get('model_gae_lambda')} != "
                 f"[{lam}]")
    own, other = ("adv_err_vs_mc", "adv_err_vs_gae095") if lam == 1.0 else ("adv_err_vs_gae095", "adv_err_vs_mc")
    if not (s.get(f"max_{own}") is not None and s[f"max_{own}"] <= tol):
        p.append(f"evidence: advantages differ from the lambda {lam} recomputation by {s.get(f'max_{own}')} > {tol}")
    if not (s.get(f"min_{other}") is not None and s[f"min_{other}"] > 10 * tol):
        p.append(f"evidence: advantages are not distinct from the other lambda's (min {s.get(f'min_{other}')})")
    if not (s.get("max_mc_vs_gae_lambda1_recursion") is not None and s["max_mc_vs_gae_lambda1_recursion"] <= tol):
        p.append("evidence: the direct Monte Carlo sum and the lambda-1 recursion disagree (the recomputation)")
    if not (s.get("max_returns_minus_adv_minus_values") is not None and s["max_returns_minus_adv_minus_values"] <= tol):
        p.append("evidence: buffer returns != advantages + values")
    if not (s.get("min_dmu_actor") is not None and s["min_dmu_actor"] > 0.0):
        p.append(f"evidence: the actor's mean did not depend on observation 6 on some rollout "
                 f"({s.get('min_dmu_actor')})")
    if not (s.get("max_buffer_values_vs_critic") is not None and s["max_buffer_values_vs_critic"] <= vtol):
        p.append(f"evidence: the rollout's values are not the run's critic's ({s.get('max_buffer_values_vs_critic')})")
    paths = ("dv_predict_values", "dv_forward", "dv_evaluate_actions", "dv_mirrored")
    if arm == "critic_blind_tail3":
        for k in paths:
            if s.get(f"max_{k}") != 0.0:
                p.append(f"evidence: critic_blind: the value through {k[3:]} moved by {s.get(f'max_{k}')} with obs 6")
        if not (s.get("min_buffer_values_vs_raw_value_mlp") is not None
                and s["min_buffer_values_vs_raw_value_mlp"] > vtol):
            p.append("evidence: critic_blind: the rollout's values equal the unblinded value MLP's on some rollout")
    else:
        if not (s.get("min_dv_predict_values") is not None and s["min_dv_predict_values"] > 0.0):
            p.append(f"evidence: {arm}: the critic did not depend on observation 6 on some rollout")
    if arm == "frozen_obs_norm":
        if not s.get("obs_rms_equal_start_all"):
            p.append("evidence: frozen_obs_norm: the observation statistics moved from the start pair's")
        fl = s.get("frozen_update_calls_first_last")
        if not (isinstance(fl, list) and fl[0] >= 1 and fl[1] > fl[0] or (n == 1 and fl and fl[0] >= 1)):
            p.append(f"evidence: frozen_obs_norm: the no-op update was not reached as the rollouts ran ({fl})")
        rc = s.get("ret_rms_count_first_last") or [0, 0]
        if not rc[1] > rc[0] and n > 1:
            p.append("evidence: frozen_obs_norm: the return statistics stopped updating")
    else:
        if s.get("obs_rms_equal_start_any"):
            p.append(f"evidence: {arm}: the observation statistics equal the start pair's after a rollout (frozen?)")
        if n > 1 and not s.get("obs_rms_distinct_digests", 0) >= n:
            p.append(f"evidence: {arm}: the observation statistics did not move every rollout")
    return p
