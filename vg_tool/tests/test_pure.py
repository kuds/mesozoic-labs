"""vg_sigma's pure parts: buckets, returns-to-go (bootstrap at truncation, none at termination), the reward
normalisation (the critic's units), the bucket statistics, the arm summary, the sigma extraction and its floor, the
phase-feeding self-check."""
import io
import math
import zipfile

import numpy as np
import pytest
import vg_sigma as T

EDGES = [(0, 0), (49, 0), (50, 1), (149, 1), (150, 2), (199, 2), (200, 3), (249, 3), (250, 4), (999, 4), (1000, 4)]


# ------------------------------------------------------------------------------------------------ buckets
@pytest.mark.parametrize("k,b", EDGES)
def test_bucket_edges(k, b):
    assert T.bucket_of(k) == b
    assert int(T.buckets_of(np.array([k], dtype=np.int64))[0]) == b
    assert T.BUCKET_NAMES[b] == ("[0,50)", "[50,150)", "[150,200)", "[200,250)", "[250,1000]")[b]


def test_buckets_vectorised_equals_scalar_for_every_k():
    ks = np.arange(0, 1001, dtype=np.int64)
    assert [int(x) for x in T.buckets_of(ks)] == [T.bucket_of(int(k)) for k in ks]


@pytest.mark.parametrize("bad", [-1, 1001, 2.5, True])
def test_bucket_refuses_out_of_range(bad):
    with pytest.raises(ValueError):
        T.bucket_of(bad)


def test_buckets_of_refuses_out_of_range():
    for arr in (np.array([0, 1001]), np.array([-1, 5]), np.array([1.5])):
        with pytest.raises(ValueError):
            T.buckets_of(arr)


# ------------------------------------------------------------------------------------------------ returns-to-go
def test_G_termination_no_bootstrap_exact():
    # r = 1, 2, 3, gamma 0.5: G2 = 3; G1 = 2 + 0.5 * 3 = 3.5; G0 = 1 + 0.5 * 3.5 = 2.75 (exact binary values)
    G = T.returns_to_go(np.array([1, 2, 3], dtype=np.float32), 0.5, None)
    assert G.dtype == np.float64
    assert G.tolist() == [2.75, 3.5, 3.0]


def test_G_truncation_bootstrap_exact():
    # the same with V(terminal) = 4: G2 = 3 + 0.5 * 4 = 5; G1 = 2 + 2.5 = 4.5; G0 = 1 + 2.25 = 3.25
    G = T.returns_to_go(np.array([1, 2, 3], dtype=np.float32), 0.5, 4.0)
    assert G.tolist() == [3.25, 4.5, 5.0]
    # bootstrap 0.0 is a bootstrap of value 0 (not None): equal to the termination case here
    assert T.returns_to_go([1, 2, 3], 0.5, 0.0).tolist() == [2.75, 3.5, 3.0]


def test_G_bootstrap_equals_sb3_fold_into_last_reward():
    """SB3 collect_rollouts: rewards[last] += gamma * V(terminal_observation) at a time-limit truncation."""
    rng = np.random.default_rng(1)
    r = rng.normal(size=37).astype(np.float32).astype(np.float64)
    gamma, v = 0.98, 4.73
    folded = r.copy()
    folded[-1] += gamma * v
    np.testing.assert_allclose(T.returns_to_go(r, gamma, v), T.returns_to_go(folded, gamma, None), rtol=0,
                               atol=1e-12)


def test_G_direct_sum_and_bs_hooks_mc_return():
    """Against the direct sum and against the phase screen's own independent mc_return (bs_hooks)."""
    rng = np.random.default_rng(2)
    n, gamma, v = 60, 0.98, 3.3
    r = rng.normal(size=n)
    direct_t = np.array([sum(gamma ** (j - t) * r[j] for j in range(t, n)) for t in range(n)])
    direct_b = direct_t + np.array([gamma ** (n - t) * v for t in range(n)])
    np.testing.assert_allclose(T.returns_to_go(r, gamma, None), direct_t, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T.returns_to_go(r, gamma, v), direct_b, rtol=1e-12, atol=1e-12)
    starts = np.zeros((n, 1))
    mc_trunc = T.H.mc_return(r[:, None], starts, np.array([v]), np.array([0.0]), gamma)[:, 0]
    mc_term = T.H.mc_return(r[:, None], starts, np.array([v]), np.array([1.0]), gamma)[:, 0]
    np.testing.assert_allclose(T.returns_to_go(r, gamma, v), mc_trunc, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(T.returns_to_go(r, gamma, None), mc_term, rtol=1e-12, atol=1e-12)


def test_episode_rows_bootstrap_only_truncated():
    eps = [{"k": np.arange(3), "r_norm": np.array([1, 2, 3], np.float32), "trunc": True},
           {"k": np.arange(2), "r_norm": np.array([1, 1], np.float32), "trunc": False},
           {"k": np.arange(1), "r_norm": np.array([2], np.float32), "trunc": True}]
    V = np.arange(6, dtype=np.float64)
    VT = np.array([4.0, 10.0])
    rows = T.episode_rows(eps, V, VT, 0.5)
    assert rows[0]["G"].tolist() == [3.25, 4.5, 5.0]
    assert rows[1]["G"].tolist() == [1.5, 1.0]                       # terminated: no bootstrap
    assert rows[2]["G"].tolist() == [2.0 + 0.5 * 10.0]
    assert [r["V"].tolist() for r in rows] == [[0, 1, 2], [3, 4], [5]]
    with pytest.raises(T.Refused):
        T.episode_rows(eps, V, VT[:1], 0.5)                           # a truncated episode without its V(terminal)
    with pytest.raises(T.Refused):
        T.episode_rows(eps, np.arange(7.0), VT, 0.5)                  # states left over


# ------------------------------------------------------------------------------------------------ the critic's units
def _vecnorm(var, clip_reward=50.0):
    import gymnasium as gym
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    class _E(gym.Env):
        observation_space = gym.spaces.Box(-1, 1, (63,), np.float32)
        action_space = gym.spaces.Box(-1, 1, (15,), np.float32)

        def reset(self, seed=None, options=None):
            return np.zeros(63, np.float32), {}

        def step(self, a):
            return np.zeros(63, np.float32), 0.0, False, False, {}

    vn = VecNormalize(DummyVecEnv([_E]), norm_obs=True, norm_reward=True, clip_reward=clip_reward, gamma=0.98)
    vn.training = False
    vn.ret_rms.var = np.float64(var)
    return vn


def test_unit_normalisation_hand_values():
    var, eps = 2057.4123244768098, 1e-8           # the start pair's ret_rms.var [measured, val/explore.out]
    vn = _vecnorm(var)
    raw = np.array([1.0, -3.5, 4.75, 4000.0, -4000.0], dtype=np.float32)
    got = T.normalised_rewards(vn, raw)
    sig = math.sqrt(var + eps)
    want = np.clip(raw.astype(np.float64) / sig, -50.0, 50.0).astype(np.float32)
    assert got.dtype == np.float32
    assert np.array_equal(got, want)
    assert got[3] == np.float32(50.0) and got[4] == np.float32(-50.0)     # 4000 / 45.36 = 88.2: clipped
    assert abs(float(got[0]) - 1.0 / 45.35870726207714) < 1e-8
    assert float(vn.ret_rms.var) == var                                    # frozen: never updated


def test_G_in_critic_units_is_raw_G_over_sigma_r():
    var = 2057.4123244768098
    vn = _vecnorm(var)
    rng = np.random.default_rng(3)
    raw = (4.0 + rng.normal(size=200)).astype(np.float32)
    g_norm = T.returns_to_go(T.normalised_rewards(vn, raw), 0.98, None)
    g_raw = T.returns_to_go(raw, 0.98, None)
    np.testing.assert_allclose(g_norm, g_raw / math.sqrt(var + 1e-8), rtol=1e-6)


def test_unit_normalisation_real_start_pkl():
    from stable_baselines3.common.vec_env import VecNormalize

    vn0 = _vecnorm(1.0)
    vn = VecNormalize.load(T.C.f2_file(T.C.START["pkl_rel"]), vn0.venv)
    vn.training, vn.norm_reward = False, True
    assert float(np.sqrt(vn.ret_rms.var + vn.epsilon)) == 45.35870726207714
    assert vn.clip_reward == 50.0 and vn.gamma == 0.98
    raw = np.array([4.2, -100.0], dtype=np.float32)
    got = T.normalised_rewards(vn, raw)
    assert np.array_equal(got, np.clip(raw / np.sqrt(vn.ret_rms.var + vn.epsilon), -50, 50).astype(np.float32))


# ------------------------------------------------------------------------------------------------ statistics
def test_bucket_stats_hand_built():
    # episode A: 1000 states, V - G = +1 for k < 50, -2 for 200 <= k < 250, 0 elsewhere
    kA = np.arange(1000)
    eA = np.where(kA < 50, 1.0, np.where((kA >= 200) & (kA < 250), -2.0, 0.0))
    # episode B: a fall at k = 120 (states 0..119): V - G = +3 for k < 50, +1 for 50 <= k < 120
    kB = np.arange(120)
    eB = np.where(kB < 50, 3.0, 1.0)
    rows = [{"k": kA, "V": eA + 5.0, "G": np.full(1000, 5.0)}, {"k": kB, "V": eB, "G": np.zeros(120)}]
    st = T.bucket_stats(rows)
    b0, b1, b2, b3, b4 = st
    assert (b0["n_episodes"], b0["n_states"]) == (2, 100)
    assert b0["mean_err"] == 2.0 and b0["pooled_err"] == 2.0
    assert b0["se_err"] == pytest.approx(np.std([1.0, 3.0], ddof=1) / math.sqrt(2))
    assert (b1["n_episodes"], b1["n_states"]) == (2, 170)
    assert b1["mean_err"] == pytest.approx(0.5)                      # episode means 0 and 1, equally weighted
    assert b1["pooled_err"] == pytest.approx(70.0 / 170.0)          # state-weighted
    assert (b2["n_episodes"], b2["n_states"]) == (1, 50) and b2["se_err"] is None
    assert b3["mean_err"] == -2.0 and b3["mean_abs_err"] == 2.0 and b3["mean_V"] == 3.0 and b3["mean_G"] == 5.0
    assert b4["n_states"] == 750 and b4["mean_err"] == 0.0
    empty = T.bucket_stats([{"k": np.arange(10), "V": np.zeros(10), "G": np.zeros(10)}])
    assert empty[1]["n_episodes"] == 0 and empty[1]["mean_err"] is None


def _run(name, arm, seed, errs, abs_errs=None):
    abs_errs = abs_errs or [abs(e) for e in errs]
    return {"name": name, "arm": arm, "seed": seed,
            "buckets": [{"mean_err": e, "mean_abs_err": a} for e, a in zip(errs, abs_errs, strict=True)]}


def test_arm_summary_medians_difference_paired_and_expectations():
    runs = [_run("control_1", "control", 1, [0.0, 0.10, 0.08, -0.06, -0.02]),
            _run("control_2", "control", 2, [0.0, 0.12, 0.06, -0.04, -0.02]),
            _run("control_3", "control", 3, [0.0, 0.14, 0.10, -0.08, -0.02]),
            _run("critic_phase_1", "critic_phase", 1, [0.0, 0.01, 0.01, -0.01, 0.0]),
            _run("critic_phase_2", "critic_phase", 2, [0.0, 0.03, 0.03, -0.02, 0.0])]
    s = T.arm_summary(runs)
    assert s[1]["control"]["median_err"] == pytest.approx(0.12)
    assert s[1]["critic_phase"]["median_err"] == pytest.approx(0.02)        # midpoint of 2
    assert s[1]["control_minus_phase"]["median_err"] == pytest.approx(0.10)
    assert [p["seed"] for p in s[2]["paired"]] == [1, 2]
    assert s[2]["paired"][0]["control_minus_phase_err"] == pytest.approx(0.07)
    ex = T.expectations(s)
    assert ex == {"E1 [150,200)": "consistent", "E1 [200,250)": "consistent", "E2 control sign pattern": "consistent"}
    runs[3]["buckets"][3]["mean_err"] = -0.2
    runs[4]["buckets"][3]["mean_err"] = -0.2
    assert T.expectations(T.arm_summary(runs))["E1 [200,250)"] == "not consistent"
    assert T.expectations(T.arm_summary(runs[:3]))["E1 [150,200)"] == "not assessable"
    assert T.median([3, 1, 2]) == 2 and T.median([]) is None and T.median([1, None, 3]) == 2.0


# ------------------------------------------------------------------------------------------------ sigma
def test_sigma_record_and_floor():
    ls = np.full(15, np.log(0.2), dtype=np.float32)
    ls[13] = np.float32(np.log(0.005))             # tail_2 below the floor
    s = T.sigma_record(ls)
    assert s["tail_3"]["action"] == 14 and s["tail_3"]["actuator"] == "tail_3_pitch_act"
    assert s["tail_2"]["action"] == 13 and s["tail_2"]["actuator"] == "tail_2_pitch_act"
    assert s["tail_3"]["sigma"] == float(np.exp(np.float64(ls[14])))
    assert s["tail_3"]["sigma_floored"] == s["tail_3"]["sigma"] and not s["tail_3"]["floor_binds"]
    assert s["tail_2"]["sigma"] == pytest.approx(0.005, rel=1e-6)
    assert s["tail_2"]["sigma_floored"] == 0.01 and s["tail_2"]["floor_binds"]
    assert s["floor"] == 0.01 == T.SIGMA_FLOOR
    with pytest.raises(T.Refused):
        T.sigma_record(np.zeros(14, np.float32))


def test_read_log_std_from_a_zip(tmp_path):
    import torch

    ls = torch.tensor(np.linspace(-3, -1, 15), dtype=torch.float32)
    buf = io.BytesIO()
    torch.save({"log_std": ls, "mlp_extractor.policy_net.0.weight": torch.zeros(2, 2)}, buf)
    z = tmp_path / "m.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("policy.pth", buf.getvalue())
        zf.writestr("data", "{}")
    got = T.read_log_std(z)
    assert got.dtype == np.float32 and np.array_equal(got, ls.numpy())
    with zipfile.ZipFile(tmp_path / "n.zip", "w") as zf:
        buf2 = io.BytesIO()
        torch.save({"x": torch.zeros(1)}, buf2)
        zf.writestr("policy.pth", buf2.getvalue())
    with pytest.raises(T.Refused):
        T.read_log_std(tmp_path / "n.zip")


def test_read_log_std_real_start_pair():
    """The start pair's tail sigma; the same values as the loaded model's log_std [measured, val/explore.out]."""
    s = T.sigma_record(T.read_log_std(T.C.f2_file(T.C.START["zip_rel"])))
    assert s["tail_2"]["sigma"] == 0.1918830645898447
    assert s["tail_3"]["sigma"] == 0.18164397017949013
    assert not s["tail_2"]["floor_binds"] and not s["tail_3"]["floor_binds"]


# ------------------------------------------------------------------------------------------------ phase self-check
def test_phase_feed_problems():
    ok = {"value_phase_cols_max_abs_used": 0.4, "max_abs_eff_minus_raw": 0.2, "max_abs_eff_minus_raw_term": 0.1}
    assert T.phase_feed_problems("p", ok, 3) == []
    assert len(T.phase_feed_problems("p", dict(ok, max_abs_eff_minus_raw=0.0), 3)) == 1          # states not fed
    assert len(T.phase_feed_problems("p", dict(ok, max_abs_eff_minus_raw_term=0.0), 3)) == 1     # terminal not fed
    assert T.phase_feed_problems("p", dict(ok, max_abs_eff_minus_raw_term=None), 0) == []        # no truncation
    zero = {"value_phase_cols_max_abs_used": 0.0, "max_abs_eff_minus_raw": 0.0, "max_abs_eff_minus_raw_term": 0.0}
    assert T.phase_feed_problems("p", zero, 3) == []                     # columns zero (S, PS validate): nothing due
    assert len(T.phase_feed_problems("p", {}, 3)) == 1                   # no record: a problem, never silent
