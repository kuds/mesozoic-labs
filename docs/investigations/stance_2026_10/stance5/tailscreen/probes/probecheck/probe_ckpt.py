"""Report-only probe of one fetched checkpoint (loaded only via the judge's load_checkpoint), against its run's policy_probe.npz.
argv: RUN STEPS ZIP PKL NPZ OUTJSON.  E0 actuator order on the live env; E1 bank rebuild + probe-row reproduction;
E2 tail_3 joint-angle substitution on the bank rows; E3 deterministic rollouts (raw mean recorded, clip(mean) stepped)."""
import os, sys, json, time
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"; sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, HERE)
import numpy as np
import judge_stance5 as J
run, steps, zp, pkl, npz, out = sys.argv[1], int(sys.argv[2]), *sys.argv[3:7]
CODE = HERE + "/code"
J.isolate(CODE)
import torch
torch.set_num_threads(1)
ctx = J.Ctx("LTSB")
J.modules_under(CODE)
model, normalizer = J.load_checkpoint(ctx, zp, pkl)
policy = model.policy
T3, Q3, NQ = 14, 6, 21
res = {"run": run, "steps": steps, "num_timesteps": int(model.num_timesteps)}

def mode(raw):
    o = normalizer.normalize_obs(np.asarray(raw, dtype=np.float32).reshape(-1, raw.shape[-1]))
    with torch.no_grad():
        return policy.get_distribution(torch.as_tensor(o, dtype=torch.float32)).mode().numpy().astype(np.float64)

# E0: actuator order in the live env
import mujoco
env = ctx.env_class(**ctx.env_kwargs)
m = env.unwrapped.model
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
j3 = m.actuator_trnid[T3, 0]
res["E0"] = {"actuators": names, "act14": names[T3], "act14_joint": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j3),
             "act14_obs_q_index": int(m.jnt_qposadr[j3] - 7), "ctrlrange14": m.actuator_ctrlrange[T3].tolist(),
             "home_ctrl14": float(env.unwrapped._home_ctrl[T3]), "action_space": [env.action_space.shape, float(env.action_space.low.min()), float(env.action_space.high.max())]}
# _scale_action is elementwise per index: a +1 on index 14 alone moves ctrl[14] only
a = np.zeros(m.nu); a[T3] = 1.0
c0 = env.unwrapped._scale_action(np.zeros(m.nu)); c1 = env.unwrapped._scale_action(a); c2 = env.unwrapped._scale_action(a * 1.3)
res["E0"]["scale_action_+1_changes"] = np.flatnonzero(np.abs(c1 - c0) > 0).tolist()
res["E0"]["ctrl14_at_+1"] = float(c1[T3]); res["E0"]["ctrl14_at_+1.3_equal_+1"] = bool(c2[T3] == c1[T3])
env.close()

# E1: rebuild the bank with the project's own builder; reproduce the probe row at STEPS
from environments.shared import policy_probe as PP
z = np.load(npz, allow_pickle=False)
bank = PP.build_state_bank(lambda h: ctx.env_class(**PP.bank_env_kwargs(ctx.env_kwargs, h)), ctx.mid, stage_horizon=ctx.horizon)
res["E1"] = {"bank_rebuilt_equal_npz": bool(np.array_equal(bank.raw_obs, z["bank_raw_obs"])),
             "bank_rebuilt_maxabs_diff": float(np.max(np.abs(bank.raw_obs - z["bank_raw_obs"]))),
             "labels": list(bank.labels), "seeds": bank.seeds.tolist(), "steps": bank.steps.tolist()}
t = z["update_timesteps"]; k = int(np.flatnonzero(t == steps)[0])
raw = z["bank_raw_obs"]
mu = mode(raw)
row = z["bank_action_now"][k].astype(np.float64)
res["E1"].update({"npz_row_t3": row[:, T3].round(4).tolist(), "ckpt_mode_t3": mu[:, T3].round(4).tolist(),
                  "maxabs_diff_all": float(np.max(np.abs(mu - row))),
                  "obs_rms_mean_maxabs_diff": float(np.max(np.abs(z["obs_rms_mean"][k] - normalizer.obs_rms.mean))),
                  "std_t3_npz": float(z["std_per_actuator"][k, T3]), "std_t3_ckpt": float(torch.exp(policy.log_std)[T3])})
for kk in (k - 1, k + 1):
    if 0 <= kk < len(t):
        res["E1"][f"maxabs_diff_vs_row_{int(t[kk])}"] = float(np.max(np.abs(mu - z["bank_action_now"][kk])))

# E2: tail_3 joint angle substituted on every bank row (other 62 obs entries kept)
grid = np.round(-0.0924 + 0.0863 * np.linspace(0, 1, 11), 5)
sweep = []
for q in grid:
    r2 = raw.copy(); r2[:, Q3] = q
    sweep.append(mode(r2)[:, T3])
sweep = np.array(sweep)  # (grid, bank)
slope = np.polyfit(grid, sweep, 1)[0]  # d mu / d q3 per bank row
fp = []
for j in range(raw.shape[0]):
    a_ = mode(raw[j:j+1])[0, T3]
    for _ in range(300):  # static plant map q3 = -0.0924 + 0.0863 clip(a)
        r2 = raw[j:j+1].copy(); r2[0, Q3] = -0.0924 + 0.0863 * np.clip(a_, -1, 1)
        a_ = 0.8 * a_ + 0.2 * mode(r2)[0, T3]
    fp.append(a_)
res["E2"] = {"q3_grid": grid.tolist(), "mu_t3_by_q3_row_major": sweep.round(4).tolist(),
             "dmu_dq3_per_row": slope.round(3).tolist(), "loop_gain_per_row(0.0863*slope)": (0.0863 * slope).round(3).tolist(),
             "static_fixed_point_per_row": np.round(fp, 4).tolist()}

# E3: deterministic rollouts, raw mean recorded, clip(mean) stepped (the judge's deterministic play, outside its recorder)
env = ctx.env_class(**ctx.env_kwargs)
eps = []
t0 = time.time()
for seed in range(3042, 3042 + int(os.environ.get("NEP", "6"))):
    obs, _ = env.reset(seed=seed)
    mus, qs = [], []
    for s in range(ctx.horizon):
        mu_s = mode(np.asarray(obs)[None])[0]
        mus.append(mu_s[T3]); qs.append(float(obs[Q3]))
        obs, r, term, trunc, _ = env.step(np.clip(mu_s, -1, 1).astype(np.float32))
        if term or trunc:
            break
    mus = np.array(mus); qs = np.array(qs); L = len(mus)
    w = slice(200, L)
    first99 = int(np.flatnonzero(mus >= 0.99)[0]) if np.any(mus >= 0.99) else None
    eps.append({"seed": seed, "length": L, "mu_at_step199": float(mus[199]) if L > 199 else None, "q3_at_step199": float(qs[199]) if L > 199 else None,
                "window_mu_mean": float(mus[w].mean()), "window_frac_cmd_ge_0.99": float(np.mean(np.clip(mus[w], -1, 1) >= 0.99)),
                "window_frac_ge_0.9": float(np.mean(mus[w] >= 0.9)), "window_mu_max": float(mus[w].max()), "window_q3_mean": float(qs[w].mean()),
                "first_step_mu_ge_0.99": first99,
                "mu_trace_every50": mus[::50].round(3).tolist(), "q3_trace_every50": qs[::50].round(4).tolist()})
res["E3"] = {"episodes": eps, "seconds": time.time() - t0,
             "mean_window_frac99": float(np.mean([e["window_frac_cmd_ge_0.99"] for e in eps])),
             "mean_window_mu": float(np.mean([e["window_mu_mean"] for e in eps]))}
env.close()
json.dump(res, open(out, "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "E3"}, indent=None)[:6000])
print("E3", [(e["seed"], e["length"], round(e["mu_at_step199"] or 0, 3), round(e["window_mu_mean"], 3), round(e["window_frac_cmd_ge_0.99"], 3), e["first_step_mu_ge_0.99"]) for e in eps], res["E3"]["mean_window_frac99"])
