"""Switch-on checks with my own code against WT.  usage: on_checks.py OUTDIR RUN_DIR CAPS_NPZ
RUN_DIR: an equiv_capture run with the switch on (its run/models/stage1_final.zip), CAPS_NPZ its caps.npz."""
import json
import math
import pickle
import sys
from types import SimpleNamespace

WT = "/home/user/work/pc_branch"
sys.path.insert(0, WT)
OUT, RUN, CAPS = sys.argv[1], sys.argv[2], sys.argv[3]

import numpy as np  # noqa: E402
import torch as th  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.policies import ActorCriticPolicy  # noqa: E402
from stable_baselines3.common.vec_env import VecNormalize  # noqa: E402

import environments.shared.train_base as tb  # noqa: E402

assert tb.__file__.startswith(WT)
from environments.shared import critic_phase_ppo as CPP  # noqa: E402
from environments.shared.config import load_all_stages  # noqa: E402
from environments.shared.k3_panel import policy_mean_action  # noqa: E402
from environments.shared.mirror_kl_ppo import enable_mirror_kl  # noqa: E402
from environments.shared.plant_contract import current_plant_identity  # noqa: E402
from environments.shared.policy_loading import load_sb3_model  # noqa: E402
from environments.shared.species_names import require_model_id  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402
from environments.shared.stage_manifest import load_stage_manifest  # noqa: E402
from environments.shared.symmetry import MirrorMap, pop_symmetry_config  # noqa: E402

RESULTS = []


def check(name, cond, detail=""):
    detail = str(detail)
    RESULTS.append({"check": name, "pass": bool(cond), "detail": detail})
    print(("PASS " if cond else "FAIL ") + name + (f"  [{detail}]" if detail != "" else ""), flush=True)


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


S, G = 200, 0.98


def phi(k):
    # my own rule, written from the task text: ramp min(k,200)/200, settled k>=200, discount 0.98^max(0,200-k)
    return np.array([min(k, S) / S, 1.0 if k >= S else 0.0, G ** max(0, S - k)], dtype=np.float32)


model_id = require_model_id("tyrannosaurus_rex", "anatomical")
cfg = get_species_config(model_id)
stages = load_all_stages(model_id)
ref = load_stage_manifest(model_id).resolve("stance").reference
plant = current_plant_identity(model_id)

# ------------------------------------------------------------------ A. the real env through create_vec_env
part = sys.argv[4] if len(sys.argv) > 4 else "all"
if part in ("all", "A"):
    E = 2
    on = tb.create_vec_env(cfg, stages, ref, E, seed=777, algorithm="ppo", gamma=0.98, plant_identity=plant,
                           critic_phase_gamma=0.98)
    off = tb.create_vec_env(cfg, stages, ref, E, seed=777, algorithm="ppo", gamma=0.98, plant_identity=plant)
    check("A0 training env class", type(on).__name__ == "CriticPhaseVecNormalize" and type(off) is VecNormalize)
    check("A0 observation_space stays 63", on.observation_space.shape == (63,) == off.observation_space.shape)
    o1, o0 = on.reset(), off.reset()
    check("A1 reset obs 66 wide, first 63 = stock VecNormalize bitwise", o1.shape == (E, 66) and same(o1[:, :63], o0))
    check("A1 reset phase = phi(0)", all(same(o1[e, 63:], phi(0)) for e in range(E)), o1[0, 63:])
    zero = np.zeros((E, 15), dtype=np.float32)
    bad_obs = bad_phase = bad_term = 0
    first_paid = {}
    truncs = []
    settled_first_k = None
    keys_before = {}
    for t in range(1, 1121):
        n1, r1, d1, i1 = on.step(zero)
        n0, r0, d0, i0 = off.step(zero)
        if not (same(n1[:, :63], n0) and same(r1, r0) and same(d1, d0)):
            bad_obs += 1
        for e in range(E):
            k = int(i1[e]["step"])
            if d1[e]:
                term = i1[e]["terminal_observation"]
                if not (same(term[63:], phi(k)) and same(term[:63], i0[e]["terminal_observation"])):
                    bad_term += 1
                if not same(n1[e, 63:], phi(0)):
                    bad_phase += 1
                truncs.append((e, k, bool(i1[e].get("TimeLimit.truncated", False)), term[63:].tolist(), n1[e, 63:].tolist()))
            else:
                if not same(n1[e, 63:], phi(k)):
                    bad_phase += 1
                if settled_first_k is None and n1[e, 64] == 1.0:
                    settled_first_k = k
            for key, val in i1[e].items():
                if key.startswith("reward_") and isinstance(val, (int, float, np.floating)) and val != 0 and key not in first_paid:
                    first_paid[key] = k
    check("A2 every step: obs[:63], rewards, dones bitwise = stock VecNormalize (1120 steps x 2 envs)", bad_obs == 0, bad_obs)
    check("A2 every non-final step: phase = phi(info['step'])", bad_phase == 0, bad_phase)
    check("A3 settled first 1 at k = 200", settled_first_k == 200, settled_first_k)
    gated = {k: v for k, v in first_paid.items() if v > 1}
    print("first nonzero step of reward terms paid late:", json.dumps(gated))
    print("reward terms paid from step 1:", sorted(k for k, v in first_paid.items() if v == 1))
    GATED = ("reward_stance_width", "reward_lateral_lean", "reward_stance_stagger", "reward_stance_cop_band")
    seen = {k: first_paid.get(k) for k in GATED}
    check("A3 the settle-gated terms (base_env.py 1315/1339/1421 gates) first paid at info step 201, none before",
          all(v in (None, 201) for v in seen.values()) and sum(v == 201 for v in seen.values()) >= 3, seen)
    check("A4 truncation at k=1000 with TimeLimit.truncated, terminal phase phi(1000)=(1,1,1), next obs phi(0)",
          len(truncs) >= 1 and all(k == 1000 and tr and term == [1.0, 1.0, 1.0] and nxt == phi(0).tolist()
                                   for _, k, tr, term, nxt in truncs), truncs)
    check("A4 terminal observations' phase and obs[:63] (stock-normalized) right", bad_term == 0, bad_term)
    check("A5 obs_rms stays 63 wide; ret_rms scalar", on.obs_rms.mean.shape == (63,) and np.shape(on.ret_rms.mean) == ())
    check("A5 obs_rms of the phase env = stock env's bitwise", same(on.obs_rms.mean, off.obs_rms.mean) and same(on.obs_rms.var, off.obs_rms.var))
    on.save(f"{OUT}/vn_on.pkl")
    off.save(f"{OUT}/vn_off.pkl")
    b1, b0 = open(f"{OUT}/vn_on.pkl", "rb").read(), open(f"{OUT}/vn_off.pkl", "rb").read()
    check("A6 phase env's sidecar is byte-identical to the stock env's after the same 1120 steps", b1 == b0, (len(b1), len(b0)))
    loaded = pickle.loads(b1)
    check("A6 sidecar unpickles as exactly VecNormalize, no critic attrs", type(loaded) is VecNormalize
          and not any(a in vars(loaded) for a in ("critic_phase", "critic_observation_space")))
    # cannot blow up: feed extreme raw obs and a constant 'settled' history; the phase columns never pass normalize
    raw = np.full((E, 63), 1e9, dtype=np.float32)
    norm = on.normalize_obs(raw)
    check("A7 normalize_obs clips the observation only (63 wide) at clip_obs", norm.shape == (E, 63) and np.abs(norm).max() <= on.clip_obs)
    on.close(); off.close()

# ------------------------------------------------------------------ B. the trained model
if part in ("all", "B"):
    caps = np.load(CAPS)
    zip_path = f"{RUN}/run/models/stage1_final.zip"
    vn_path = f"{RUN}/run/models/stage1_final_vecnorm.pkl"
    m = load_sb3_model(zip_path, algorithm=PPO)  # deploy loader, no env
    pol = m.policy
    pol.set_training_mode(False)
    check("B0 deploy load: policy class and spaces", type(pol).__name__ == "CriticPhaseActorCriticPolicy"
          and m.observation_space.shape == (63,) and pol.mlp_extractor.policy_net[0].weight.shape == (512, 63)
          and pol.mlp_extractor.value_net[0].weight.shape == (512, 66), type(pol).__name__)
    rec = getattr(m, "mesozoic_critic_phase", None)
    check("B0 checkpoint record", isinstance(rec, dict) and rec.get("settle_steps") == 200 and rec.get("gamma") == 0.98, rec)
    w = pol.mlp_extractor.value_net[0].weight.detach().numpy()
    check("B1 critic learned the phase: phase columns nonzero after 2 updates", np.abs(w[:, 63:]).max() > 0, float(np.abs(w[:, 63:]).max()))
    # final params equal the run's own final capture
    fin_ok = all(same(v.detach().numpy(), caps[f"final/param/{k}"]) for k, v in pol.state_dict().items())
    check("B1 loaded params = the trained model's final params bitwise", fin_ok)
    obs = np.concatenate([caps["roll0/observations"].reshape(-1, 66), caps["roll1/observations"].reshape(-1, 66)])
    acts = np.concatenate([caps["roll0/actions"].reshape(-1, 15), caps["roll1/actions"].reshape(-1, 15)])
    rng = np.random.default_rng(5)
    base = th.as_tensor(obs, dtype=th.float32)
    A = th.as_tensor(acts, dtype=th.float32)
    perturb = {
        "zeros": np.zeros((len(obs), 3)), "ones": np.ones((len(obs), 3)), "uniform": rng.uniform(0, 1, (len(obs), 3)),
        "+1e6": np.full((len(obs), 3), 1e6), "-1e6": np.full((len(obs), 3), -1e6),
        "nan": np.full((len(obs), 3), np.nan), "inf": np.full((len(obs), 3), np.inf),
        "shuffled": obs[rng.permutation(len(obs)), 63:],
    }
    with th.no_grad():
        d0 = pol.get_distribution(base)
        mean0, std0 = d0.distribution.mean.numpy().copy(), d0.distribution.stddev.numpy().copy()
        v0, lp0, en0 = (x.numpy().copy() for x in pol.evaluate_actions(base, A))
        fa0, fv0, flp0 = (x.numpy().copy() for x in pol.forward(base, deterministic=True))
        p63 = pol.predict(obs[:, :63], deterministic=True)[0]
        mean63 = pol.get_distribution(th.as_tensor(obs[:, :63], dtype=th.float32)).distribution.mean.numpy()
    check("B2 actor on 63-wide obs = actor on 66-wide obs (bitwise)", same(mean63, mean0))
    for name, ph in perturb.items():
        x = base.clone()
        x[:, 63:] = th.as_tensor(ph, dtype=th.float32)
        with th.no_grad():
            d = pol.get_distribution(x)
            v, lp, en = (t.numpy() for t in pol.evaluate_actions(x, A))
            fa, fv, flp = (t.numpy() for t in pol.forward(x, deterministic=True))
            pa = pol.predict(x.numpy(), deterministic=True)[0]
        ok = (same(d.distribution.mean.numpy(), mean0) and same(d.distribution.stddev.numpy(), std0) and same(lp, lp0)
              and same(en, en0) and same(fa, fa0) and same(flp, flp0) and same(pa, p63))
        vchg = not same(v, v0) if name not in ("shuffled",) else not same(v, v0)
        check(f"B2 phase={name}: get_distribution/evaluate_actions log_prob+entropy/forward/predict actor outputs bitwise unchanged", ok)
        check(f"B2 phase={name}: critic value changes", vchg)
    # gradient: nothing of the phase reaches the actor
    x = base.clone().requires_grad_(True)
    v, lp, en = pol.evaluate_actions(x, A)
    (lp.sum() + en.sum() if en is not None else lp.sum()).backward()
    g = x.grad.numpy()
    check("B3 d(log_prob+entropy)/d(phase) exactly 0; d/d(obs) nonzero", not g[:, 63:].any() and g[:, :63].any())
    x = base.clone().requires_grad_(True)
    v, lp, en = pol.evaluate_actions(x, A)
    v.sum().backward()
    check("B3 d(value)/d(phase) nonzero (critic reads it)", x.grad.numpy()[:, 63:].any())
    # mirrored KL term with the real MirrorKLPPO hooks
    venv = tb.create_vec_env(cfg, stages, ref, 1, seed=3, algorithm="ppo", gamma=0.98, plant_identity=plant, critic_phase_gamma=0.98)
    mt = load_sb3_model(zip_path, algorithm=PPO, env=venv)
    symcfg = pop_symmetry_config(dict(stages[ref]["ppo_kwargs"]))
    mm = MirrorMap.load(model_id, plant)
    mt = enable_mirror_kl(mt, symcfg, mm)
    upd = mt._mirror_begin_update()
    tens = mt._mirror._mirror_state().tensors if hasattr(mt._mirror, "_mirror_state") else mt._mirror.tensors
    perm, sign = tens["obs_perm"].numpy(), tens["obs_sign"].numpy()
    check("B4 mirror map: phase columns 63,64,65 map to themselves with sign +1; obs part = the T. rex map",
          perm.shape == (66,) and list(perm[63:]) == [63, 64, 65] and list(sign[63:]) == [1, 1, 1]
          and same(perm[:63].astype(np.int64), np.asarray(mm.obs.perm).astype(np.int64)), perm[60:])
    mirrored = obs[:, perm] * sign
    check("B4 mirrored phase = phase", same(mirrored[:, 63:].astype(np.float32), obs[:, 63:]))
    mt.policy.set_training_mode(False)
    returns = th.as_tensor(rng.normal(size=len(obs)), dtype=th.float32)
    kls, vls = {}, {}
    for name, ph in [("asis", obs[:, 63:]), *perturb.items()]:
        x = base.clone()
        x[:, 63:] = th.as_tensor(ph, dtype=th.float32)
        x.requires_grad_(True)
        u = mt._mirror_begin_update()
        rd = SimpleNamespace(observations=x, returns=returns)
        vl = mt._mirror_value_loss(u, rd, th.zeros(()))
        loss = mt._mirror_loss(u, rd, th.zeros(()))
        loss.backward()
        kls[name] = (u["kl"][0], u["asymmetry"][0], x.grad.numpy()[:, 63:].any())
        vls[name] = u["value"][0] if u["value"] else None
    check("B5 mirror KL loss and asymmetry identical whatever the phase; d(KL)/d(phase) = 0",
          all(kls[n][0] == kls["asis"][0] and kls[n][1] == kls["asis"][1] and not kls[n][2] for n in kls), kls["asis"])
    check("B5 mirrored critic loss reads the phase (changes with it)", vls["zeros"] != vls["asis"] and vls["asis"] is not None, (vls["asis"], vls["zeros"]))
    venv.close()
    # stock clone of the actor: a stock ActorCriticPolicy with the same actor (and the critic's 63 columns)
    kw = dict(pol._get_constructor_parameters())
    kw.pop("critic_phase_settle_steps"); kw.pop("critic_phase_gamma")
    stock = ActorCriticPolicy(**kw)
    sd = {k: v.clone() for k, v in pol.state_dict().items()}
    sd["mlp_extractor.value_net.0.weight"] = sd["mlp_extractor.value_net.0.weight"][:, :63].clone()
    stock.load_state_dict(sd)
    stock.set_training_mode(False)
    # randomise the phase critic columns of a twin: evaluation must not move
    twin = type(pol)(**pol._get_constructor_parameters())
    sd2 = {k: v.clone() for k, v in pol.state_dict().items()}
    sd2["mlp_extractor.value_net.0.weight"][:, 63:] = th.randn(512, 3) * 100
    twin.load_state_dict(sd2)
    twin.set_training_mode(False)
    # C. evaluation episode on the stock eval env (sidecar statistics, training off)
    ev = tb.create_vec_env(cfg, stages, ref, 1, seed=31, algorithm="ppo", gamma=0.98, plant_identity=plant)
    side = pickle.load(open(vn_path, "rb"))
    check("C0 run sidecar is exactly VecNormalize with 63-wide obs_rms", type(side) is VecNormalize and side.obs_rms.mean.shape == (63,))
    ev.obs_rms, ev.ret_rms, ev.training, ev.norm_reward = side.obs_rms, side.ret_rms, False, False
    o = ev.reset()
    n_bad = n_steps = 0
    ep_ret = 0.0
    for t in range(1000):
        a1 = m.predict(o, deterministic=True)[0]
        a2 = stock.predict(o, deterministic=True)[0]
        a3 = twin.predict(o, deterministic=True)[0]
        a4 = policy_mean_action(pol, o[0].astype(np.float32))
        a5 = policy_mean_action(twin, o[0].astype(np.float32))
        n_bad += int(not (same(a1, a2) and same(a1, a3) and same(a4, a5)))
        o, r, d, info = ev.step(a1)
        ep_ret += float(r[0]); n_steps += 1
        if d[0]:
            break
    check("C1 deterministic evaluation episode: phase model = stock clone = phase-randomised twin; K3 forward_actor twin equal",
          n_bad == 0, f"steps {n_steps}, mismatches {n_bad}, raw return {ep_ret:.2f}")
    ev.close()
    # K3 twin as k3_panel builds it
    k3twin = type(pol)(**pol._get_constructor_parameters())
    k3twin.load_state_dict({k: v.detach().to("cpu", copy=True) for k, v in pol.state_dict().items()})
    check("C2 K3 panel twin (type(policy)(**_get_constructor_parameters())) loads and matches",
          same(policy_mean_action(k3twin, obs[0, :63].astype(np.float32)), policy_mean_action(pol, obs[0, :63].astype(np.float32))))
    # predict_values without the phase is refused, not guessed
    try:
        pol.predict_values(th.as_tensor(obs[:, :63], dtype=th.float32))
        check("C3 predict_values on a 63-wide obs refused", False)
    except Exception as exc:  # noqa: BLE001
        check("C3 predict_values on a 63-wide obs refused", "phase" in str(exc), type(exc).__name__)

    # D. probe: recompute the run's probe rows with my own code from the captured parameters
    pr = np.load(f"{RUN}/run/policy_probe.npz")
    for row in range(len(pr["update_timesteps"])):
        net = type(pol)(**pol._get_constructor_parameters())
        net.load_state_dict({k: th.as_tensor(caps[f"pre{row}/param/{k}"]) for k in pol.state_dict()})
        net.set_training_mode(False)
        mean = pr["obs_rms_mean"][row].astype(np.float64); var = pr["obs_rms_var"][row].astype(np.float64)
        raw = pr["bank_raw_obs"]
        nobs = np.clip((raw - mean) / np.sqrt(var + 1e-8), -10, 10).astype(np.float32)
        ks = pr["bank_steps"] + 1
        ph = np.stack([phi(int(k)) for k in ks])
        with th.no_grad():
            val = net.predict_values(th.as_tensor(np.concatenate([nobs, ph], 1))).numpy().ravel()
            val_wrong = net.predict_values(th.as_tensor(np.concatenate([nobs, np.stack([phi(int(k) - 1) for k in ks])], 1))).numpy().ravel()
            act = net.get_distribution(th.as_tensor(nobs)).mode().numpy()
        check(f"D1 probe row {row}: bank values = critic(obs, phi(kept step + 1)) (max abs diff)",
              np.allclose(val, pr["bank_value_now"][row], atol=1e-5, rtol=0), float(np.abs(val - pr["bank_value_now"][row]).max()))
        check(f"D1 probe row {row}: bank actions = actor(obs) (max abs diff)",
              np.allclose(act, pr["bank_action_now"][row], atol=1e-5, rtol=0), float(np.abs(act - pr["bank_action_now"][row]).max()))
        print(f"   row {row}: off-by-one alternative (phi(kept step)) max diff from recorded: {float(np.abs(val_wrong - pr['bank_value_now'][row]).max())}")
    print("bank steps", pr["bank_steps"].tolist())

json.dump(RESULTS, open(f"{OUT}/results_{part}.json", "w"), indent=1)
print("TOTAL", len(RESULTS), "FAILED", sum(not r["pass"] for r in RESULTS))
