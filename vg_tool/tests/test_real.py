"""vg_sigma on the real code copy and the real start pair (a few seconds each; read only): the phase feeding equals
read_bs's and the training env's; _form_values' phase form with nonzero critic phase columns equals the critic on the
training env's observations (states phi(k), terminal phi(1000)); the start pair's two critic forms agree bitwise; the
episode loop equals SB3's own collect_rollouts on the same seeds; the actuator indices."""
import os

import numpy as np
import pytest
import vg_sigma as T

TMP = os.environ.get("TMPDIR") or str(T.HERE / "tmp")
T.setup_sim(os.path.join(TMP, "vg_tests_sim"))
START_ZIP, START_PKL = T.C.f2_file(T.C.START["zip_rel"]), T.C.f2_file(T.C.START["pkl_rel"])


def _plain_and_phase_states(n_steps=260, seed=6281):
    """Lockstep: a plain stance env and the same env class with bs_hooks.phase_env_class (the TRAINING envs' class
    swap), the same seed and actions; returns the plain observations, the phase observations and k."""
    fac = T.env_factory()
    plain, phase = fac(), fac()
    phase.__class__ = T.H.phase_env_class(type(phase))
    o1, _ = plain.reset(seed=seed)
    o2, _ = phase.reset(seed=seed)
    rng = np.random.default_rng(seed)
    X1, X2, K = [o1.copy()], [o2.copy()], [int(plain._step_count)]
    for _ in range(n_steps):
        a = rng.normal(0, 0.05, size=15).astype(np.float32)
        o1, *_ = plain.step(a)
        o2, *_ = phase.step(a)
        X1.append(o1.copy())
        X2.append(o2.copy())
        K.append(int(plain._step_count))
    plain.close()
    phase.close()
    return np.asarray(X1, np.float32), np.asarray(X2, np.float32), np.asarray(K, np.int64)


@pytest.fixture(scope="module")
def states():
    return _plain_and_phase_states()


def test_env_phase_slots_are_float32_phase(states):
    X1, X2, K = states
    assert K.tolist() == list(range(len(K)))
    assert np.array_equal(X1[:, :60], X2[:, :60]) and not X1[:, 60:].any()
    assert np.array_equal(X2[:, 60:63], T.H.phase_matrix(K).astype(np.float32))
    assert np.array_equal(np.asarray(T.R.with_phase(X1, K), np.float32), X2)      # read_bs's feeding = the env's


def test_phase_feeding_equals_read_bs_and_training_path(states):
    """The start pair read with critic_phase's start transform (read_bs.phase_prepare), then its critic's phase columns
    set to nonzero values in memory (as after learning): vg_sigma.critic_values(X, K) equals (1) read_bs.critic_slope's
    V_mean on the same states, (2) the critic on VecNormalize-normalised training-env observations (the phase written
    by the env), state by state, bitwise; and the phase input then matters."""
    import torch

    X1, X2, K = states
    net = T.R.Net(START_ZIP, START_PKL, T.env_factory())
    rec = T.R.phase_prepare(net, start=True)
    assert rec["value_phase_cols_max_abs_used"] == 0.0 and T.H.phase_slots_frozen(net.norm.obs_rms)
    with torch.no_grad():
        g = torch.Generator().manual_seed(5)
        net.policy.mlp_extractor.value_net[0].weight[:, 60:63] = 0.3 * torch.randn(
            net.policy.mlp_extractor.value_net[0].weight.shape[0], 3, generator=g)
    v = T.critic_values(net.policy, net.norm, X1, K)
    vm = T.R.critic_slope(net.policy, net.norm, X1, K)["V_mean"]
    assert float(torch.as_tensor(v.astype(np.float32)).mean()) == pytest.approx(vm, abs=1e-6)
    with torch.no_grad():
        v_train = net.policy.predict_values(torch.as_tensor(net.norm.normalize_obs(X2)))[:, 0].numpy()
    assert np.array_equal(v.astype(np.float32), v_train)
    v_raw = T.critic_values(net.policy, net.norm, X1)
    assert np.max(np.abs(v - v_raw)) > 1e-3


def _terminal_pairs(seed=6282, n_steps=60, at=(20, 40, 60)):
    """Time-limit terminal observations as the tool receives them (the plain env's observation, slots zero) and as the
    TRAINING env writes them (bs_hooks.phase_env_class with the counter at 1000: phi(1000) in slots 60-62): the same
    two envs in lockstep; at the steps in `at` the counter is set to 1000 for one _get_obs and put back."""
    fac = T.env_factory()
    plain, phase = fac(), fac()
    phase.__class__ = T.H.phase_env_class(type(phase))
    plain.reset(seed=seed)
    phase.reset(seed=seed)
    rng = np.random.default_rng(seed)
    T1, T2 = [], []
    for t in range(1, n_steps + 1):
        a = rng.normal(0, 0.05, size=15).astype(np.float32)
        plain.step(a)
        phase.step(a)
        if t in at:
            k = int(plain._step_count)
            plain._step_count = phase._step_count = 1000
            T1.append(plain._get_obs().copy())
            T2.append(phase._get_obs().copy())
            plain._step_count = phase._step_count = k
    plain.close()
    phase.close()
    return np.asarray(T1, np.float32), np.asarray(T2, np.float32)


def test_phase_form_values_with_nonzero_phase_columns(states):
    """_form_values' phase form (the path measure_pair takes for critic_phase at E) on a pair whose critic SEES the
    phase: the start pair with the arm's frozen slots (mean 0 / var 1, as after S) and nonzero
    value_net.0.weight[:, 60:63].  V on the states and V on the time-limit terminal observations equal the critic on the
    training env's observations (phi(k), and phi(1000) at the terminal) bitwise, and differ from the slots-zero values;
    the runtime self-check (phase_feed_problems) passes on it and flags a record whose phase was not fed."""
    import torch

    X1, X2, K = states
    TX1, TX2 = _terminal_pairs()
    assert np.array_equal(TX1[:, :60], TX2[:, :60]) and not TX1[:, 60:].any()
    assert np.array_equal(TX2[:, 60:63], np.ones((len(TX2), 3), np.float32))        # phi(1000) = (1, 1, 1)
    TK = np.full(len(TX1), 1000, np.int64)
    net = T.R.Net(START_ZIP, START_PKL, T.env_factory())
    T.H.reseed_phase_slots(net.norm.obs_rms)                 # the arm's frozen slots (a critic_phase pair after S)
    net.refresh_stats()
    with torch.no_grad():
        g = torch.Generator().manual_seed(5)
        w = net.policy.mlp_extractor.value_net[0].weight
        w[:, 60:63] = 0.3 * torch.randn(w.shape[0], 3, generator=g)
    fv = T._form_values(net, "phase", X1, K, TX1, TK)        # start=False: the slots must be frozen, columns kept
    rec = fv["record"]
    assert not rec["start_transform"] and rec["value_phase_cols_max_abs_used"] > 0.1

    def critic(X):
        with torch.no_grad():
            return net.policy.predict_values(torch.as_tensor(net.norm.normalize_obs(X)))[:, 0].numpy()

    assert np.array_equal(fv["V"].astype(np.float32), critic(X2))
    assert np.array_equal(fv["V_term"].astype(np.float32), critic(TX2))
    assert np.array_equal(fv["V_raw"].astype(np.float32), critic(X1))
    assert np.array_equal(fv["V_raw_term"].astype(np.float32), critic(TX1))
    assert np.max(np.abs(fv["V"] - fv["V_raw"])) > 1e-3 and np.min(np.abs(fv["V_term"] - fv["V_raw_term"])) > 1e-3
    assert rec["max_abs_eff_minus_raw"] > 1e-3 and rec["max_abs_eff_minus_raw_term"] > 1e-3
    assert T.phase_feed_problems("x", rec, len(TX1)) == []
    assert len(T.phase_feed_problems("x", dict(rec, max_abs_eff_minus_raw=0.0), len(TX1))) == 1
    assert len(T.phase_feed_problems("x", dict(rec, max_abs_eff_minus_raw_term=0.0), len(TX1))) == 1


def test_start_pair_forms_agree_bitwise(states):
    X1, _X2, K = states
    net = T.R.Net(START_ZIP, START_PKL, T.env_factory())
    term = X1[-3:]
    vp = T._form_values(net, "plain", X1, K, term, np.array([1000] * 3))
    vq = T._form_values(net, "phase", X1, K, term, np.array([1000] * 3), start=True)
    assert np.array_equal(vp["V"], vq["V"]) and np.array_equal(vp["V_term"], vq["V_term"])
    assert vq["record"]["start_transform"] and vq["record"]["max_abs_eff_minus_raw"] == 0.0


def test_phase_form_refuses_unfrozen_slots_after_S(states):
    X1, _X2, K = states
    net = T.R.Net(START_ZIP, START_PKL, T.env_factory())
    with pytest.raises(T.Refused):
        T._form_values(net, "phase", X1, K, X1[:0], np.zeros(0, np.int64), start=False)


def test_collect_equals_sb3_collect_rollouts():
    """The episode loop against SB3 2.9.0's own collect_rollouts (read_bs.onpolicy_states' callback set-up) on two
    fresh builds with the same seed: sampled actions, values and normalised rewards bitwise over 40 vector steps."""
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Logger

    n = 40
    b1 = T.build_collection(START_ZIP, START_PKL, T.VAL_SEED)
    done, cur, steps = T.collect(b1["vn"], b1["model"], 10 ** 6, max_vec_steps=n)
    b1["vn"].close()
    assert steps == n and all(not d for d in done)
    A1 = np.stack([np.stack(cur[i]["a"]) for i in range(4)], 1)
    V1 = np.array([cur[i]["V_loop"] for i in range(4)]).T
    R1 = np.array([cur[i]["r_norm"] for i in range(4)]).T
    K1 = np.array([cur[i]["k"] for i in range(4)]).T
    assert K1.tolist() == [[t] * 4 for t in range(n)]
    b2 = T.build_collection(START_ZIP, START_PKL, T.VAL_SEED)
    model = b2["model"]
    model.set_logger(Logger(folder=None, output_formats=[]))
    model._last_obs = None

    class Rec(BaseCallback):
        def __init__(self):
            super().__init__(0)
            self.a, self.v, self.r = [], [], []

        def _on_step(self):
            self.a.append(np.asarray(self.locals["actions"]).copy())
            self.v.append(self.locals["values"].detach().numpy()[:, 0].copy())
            self.r.append(np.asarray(self.locals["rewards"]).copy())
            return len(self.a) < n

    rec = Rec()
    _t, cb = model._setup_learn(model.n_steps * model.n_envs, rec, False, "vg_equiv", False)
    model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=model.n_steps)
    b2["vn"].close()
    assert np.array_equal(A1, np.stack(rec.a)[:n])
    assert np.array_equal(V1.astype(np.float32), np.stack(rec.v)[:n])
    assert np.array_equal(R1, np.stack(rec.r)[:n])


def test_actuator_indices_and_floor():
    b = T.build_collection(START_ZIP, START_PKL, T.VAL_SEED)
    u = b["vn"].venv.envs[0].unwrapped
    assert u.model.nu == 15
    assert u.model.actuator(13).name == "tail_2_pitch_act" and u.model.actuator(14).name == "tail_3_pitch_act"
    assert b["mirror_action_perm"][13:15] == [13, 14]
    assert b["sym"].sigma_floor == T.SIGMA_FLOOR
    assert b["model"].mesozoic_symmetry["sigma_floor"] == T.SIGMA_FLOOR
    assert float(b["model"].gamma) == T.GAMMA == float(b["vn"].gamma)
    ls = b["model"].policy.log_std.detach().numpy()
    assert np.array_equal(ls, T.read_log_std(START_ZIP))
    b["vn"].close()
