"""PS build: the edits of tests/test_bs_hooks.py (each old string asserted unique) and the critic_phase tests
appended, run once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''RolloutBuffer, rollout_evidence and evidence_problems for every arm (and mutants that must fail).
"""''', '''RolloutBuffer, rollout_evidence and evidence_problems for every arm (and mutants that must fail).
PS: critic_phase's pieces (the phase features against bs_common, the env's _get_obs on the real T. rex env in lockstep
with a plain one incl. the terminal observation, the actor's mask and its zero gradient, the critic's zero columns,
the frozen statistics' slots incl. pickling and MirrorKLPPO's install, the mirror map on the real map file, the probe
feeding the bank's phase, install_arm's refusals), its per-rollout evidence and its mutants, and the real start pair's
actor and critic bitwise unchanged at every phase value after the transform.  BS2's arms stay in bs_hooks (unreachable
in PS) and keep their tests: the shared machinery is the same.
"""''')
rep('''def _evidence(arm, n=3, mutate=None):
    lam = C.ARM_GAE_LAMBDA[arm]''', '''def _evidence(arm, n=3, mutate=None):
    if arm == "critic_phase":
        return _phase_evidence(n, mutate)
    lam = H.GAE_LAMBDA[arm]''')
rep('''@pytest.mark.parametrize("arm", C.ARMS)
def test_evidence_passes_for_each_arm(arm):
    ev = _evidence(arm)
    s = H.summarize(ev)
    assert H.evidence_problems(arm, s, n_expected=3) == []
    for other in C.ARMS:''', '''@pytest.mark.parametrize("arm", H.ARMS)
def test_evidence_passes_for_each_arm(arm):
    ev = _evidence(arm)
    s = H.summarize(ev)
    assert H.evidence_problems(arm, s, n_expected=3) == []
    for other in H.ARMS:''')
s += '''

# ------------------------------------------------------------------------------------------------ critic_phase (PS)
SL = slice(H.PHASE_LO, H.PHASE_HI)


class FakeEnv:
    """A training env stand-in: a step counter, a zero command, an observation with zero command slots."""

    def __init__(self, k=0):
        self._step_count = k
        self._command = np.zeros(3, dtype=np.float32)
        self.direction_controller = None

    def _get_obs(self):
        return np.ones(OBS, dtype=np.float32)


def _slot_free_rms(symmetric=True):
    """Statistics whose slots 60-62 carried only zeros (mean exactly 0), like the start pkl's."""
    data = np.random.default_rng(5).normal(size=(50, OBS))
    data[:, SL] = 0.0
    if symmetric:
        from environments.shared.mirror_kl_ppo import SymmetricRunningMeanStd

        r = SymmetricRunningMeanStd(SimpleNamespace(perm=np.arange(OBS), sign=np.ones(OBS), dim=OBS), shape=(OBS,))
    else:
        r = RunningMeanStd(shape=(OBS,))
    r.update(data)
    return r


def _phase_setup(seed=3, n_env=4):
    p = small_policy(seed)
    with torch.no_grad():                               # nonzero first-layer slot columns, like the start pair's
        p.mlp_extractor.value_net[0].weight[:, SL] = 0.3
        p.mlp_extractor.policy_net[0].weight[:, SL] = -0.2
    env = fake_env(_slot_free_rms())
    env.epsilon, env.clip_obs = 1e-8, 10.0
    env.venv = SimpleNamespace(envs=[FakeEnv() for _ in range(n_env)])
    m = fake_model(p, lam=0.95)
    m._n_updates = 100
    return p, env, m


def _phase_buffer(p, env, k0, seed):
    """A 64-step buffer whose slots carry the phase of counters that start at k0 and reset at episode starts, as
    VecNormalize writes them (slots mean 0 / var 1); the envs' counters set to where the rollout leaves them."""
    rng = np.random.default_rng(seed)
    buf = RolloutBuffer(64, OBS_SPACE, ACT_SPACE, device="cpu", gamma=0.98, gae_lambda=0.95, n_envs=4)
    k = np.asarray(k0, dtype=np.int64).copy()
    starts = (k == 0).astype(np.float32)
    for t in range(64):
        if t:
            k = np.where(starts > 0, 0, k + 1)
        obs = rng.normal(size=(4, OBS)).astype(np.float32)
        raw = H.phase_matrix(k).astype(np.float32)
        obs[:, SL] = np.clip((raw - 0.0) / np.sqrt(1.0 + env.epsilon), -10, 10).astype(np.float32)
        with torch.no_grad():
            v = p.predict_values(torch.as_tensor(obs)).flatten()
        buf.add(obs, rng.normal(size=(4, ACT)).astype(np.float32), rng.normal(size=4).astype(np.float32), starts, v,
                torch.zeros(4))
        starts = (rng.random(4) < 0.03).astype(np.float32)
    dones = starts > 0                                 # the last step's dones are the next start flags
    for e, ke in zip(env.venv.envs, np.where(dones, 0, k + 1), strict=True):
        e._step_count = int(ke)
    with torch.no_grad():
        last = p.predict_values(torch.as_tensor(rng.normal(size=(4, OBS)).astype(np.float32))).flatten()
    buf.compute_returns_and_advantage(last_values=last, dones=dones)
    return buf, last.numpy(), dones


def _phase_evidence(n=3, mutate=None):
    p, env, m = _phase_setup()
    H.install_arm("critic_phase", m, env)
    ref = H.phase_reference(m)
    start = H.rms_digest(env.obs_rms)
    out = []
    k0 = [0, 0, 150, 990]
    for e, k in zip(env.venv.envs, k0, strict=True):
        e._step_count = k
    for j in range(n):
        if j >= 1 and mutate != "no_learning":            # an "update": the critic's phase columns move
            with torch.no_grad():
                p.mlp_extractor.value_net[0].weight[:, SL] += 0.01 * (j + 1)
            m._n_updates += 7
        if mutate == "nonzero_init" and j == 0:
            with torch.no_grad():
                p.mlp_extractor.value_net[0].weight[0, H.PHASE_LO] = 1e-3
        if mutate == "unmask" and j == 1:
            p.mlp_extractor.__class__ = MlpExtractor
        k_start = H.env_step_counts(env)
        buf, last, dones = _phase_buffer(p, env, k_start, seed=40 + j)
        if mutate == "slot_error" and j == 2:
            buf.observations[5, 1, H.PHASE_LO + 2] += 1e-3
        if mutate == "counter" and j == 1:
            env.venv.envs[0]._step_count += 1
        if mutate == "unfreeze" and j == 1:
            env.obs_rms.var = env.obs_rms.var.copy()
            env.obs_rms.var[H.PHASE_LO] = 2.0
        if mutate == "mirror" and j == 2:
            m._mirror.mirror_map.obs.sign[H.PHASE_LO + 1] = -1.0
        if mutate == "actor_cols" and j == 2:
            with torch.no_grad():
                p.mlp_extractor.policy_net[0].weight[0, H.PHASE_LO] += 1e-3
        m.rollout_buffer = buf
        m.num_timesteps = C.S + C.ROLLOUT * (j + 1)
        env.obs_rms.update(np.random.default_rng(20 + j).normal(size=(9, OBS)))
        env.ret_rms.update(np.random.default_rng(30 + j).normal(size=9))
        out.append(H.rollout_evidence("critic_phase", m, env, {"values": torch.as_tensor(last), "dones": dones},
                                      start_obs_digest=start, phase_ctx={"k0": k_start, "ref": ref}))
    return out


def test_phase_features_and_env_obs_on_the_real_env():
    """The phase in the observation is the env's own counter's (0 at reset, k after k steps; incremented before
    _get_obs, as the reward gates read it), on a real T. rex env class-swapped, in lockstep with a plain twin: every
    other value equal; the terminal observation at the 1,000-step limit carries k = 1000 (settled); VecNormalize's
    formula with the frozen slots gives back the float32 phase exactly at every k."""
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    a, b = env_class(mid)(**ctor), env_class(mid)(**ctor)
    a.__class__ = H.phase_env_class(type(a))
    oa, _ = a.reset(seed=11)
    ob, _ = b.reset(seed=11)
    assert np.array_equal(oa[:60], ob[:60]) and np.all(ob[SL] == 0.0) and tuple(oa[SL]) == tuple(
        np.float32(x) for x in H.phase_tuple(0))
    zero = np.zeros(a.action_space.shape, dtype=np.float32)
    seen = {}
    for _ in range(1000):
        oa, ra, ta, tra, _ = a.step(zero)
        ob, rb, tb, trb, _ = b.step(zero)
        k = a._step_count
        assert np.array_equal(oa[:60], ob[:60]) and ra == rb and (ta, tra) == (tb, trb)
        assert oa[SL].tolist() == [float(np.float32(x)) for x in H.phase_tuple(k)]
        seen[k] = oa[SL].tolist()
        if ta or tra:
            break
    assert k == 1000 and tra and seen[1000] == [1.0, 1.0, 1.0]       # the time-limit's terminal observation
    assert seen[199] == [float(np.float32(0.995)), 0.0, float(np.float32(0.98 ** 1))] and seen[200] == [1.0, 1.0, 1.0]
    a.close()
    b.close()
    raw = H.phase_matrix(np.arange(0, 1001)).astype(np.float32)
    normed = np.clip((raw - 0.0) / np.sqrt(1.0 + 1e-8), -10.0, 10.0).astype(np.float32)
    assert np.array_equal(normed, raw)


def test_phase_env_refuses_a_live_command():
    e = FakeEnv()
    e.__class__ = H.phase_env_class(FakeEnv)
    assert e._get_obs()[SL].tolist() == list(H.phase_tuple(0))[:2] + [float(np.float32(H.phase_tuple(0)[2]))]
    e._command = np.array([0.1, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(H.HookError):
        e._get_obs()


def test_critic_phase_install_transform_mask_and_zero_init():
    p, env, m = _phase_setup()
    x = torch.randn(64, OBS)
    with torch.no_grad():
        x0 = x.clone()
        x0[:, SL] = 0.0
        mu_load, v_load = p.get_distribution(x0).mode().clone(), p.predict_values(x0).clone()
    sd = {k: v.clone() for k, v in p.state_dict().items()}
    rms_before = (env.obs_rms.mean.copy(), env.obs_rms.var.copy(), float(env.obs_rms.count))
    rec = H.install_arm("critic_phase", m, env)
    assert rec["transform_verified"] is True and rec["mlp_extractor_class"] == "CriticPhaseMlpExtractor"
    assert rec["obs_rms_class"] == "PhaseSlotsSymmetricRunningMeanStd" and rec["n_train_envs"] == 4
    assert rec["value_phase_cols_max_abs_before"] == pytest.approx(0.3)
    after = p.state_dict()
    for k in sd:
        if k == "mlp_extractor.value_net.0.weight":
            assert torch.all(after[k][:, SL] == 0) and torch.equal(after[k][:, :60], sd[k][:, :60])
        else:
            assert torch.equal(after[k], sd[k]), k
    assert np.all(env.obs_rms.mean[SL] == 0.0) and np.all(env.obs_rms.var[SL] == 1.0)
    assert np.array_equal(env.obs_rms.mean[:60], rms_before[0][:60]) and env.obs_rms.count == rms_before[2]
    assert np.all(rms_before[1][SL] != 1.0)            # the reseed changed them
    assert all(type(e).__name__ == "PhaseObsFakeEnv" for e in env.venv.envs)
    assert H.mirror_phase_self_map(m._mirror.mirror_map.obs) and H.arm_state_problems("critic_phase", m, env) == []
    assert H.arm_state_problems("control", m, env) != []
    acts = torch.randn(64, ACT)
    with torch.no_grad():
        for fill in (0.0, 1.0, 0.5, -3.0, 7.0):
            y = x.clone()
            y[:, SL] = fill
            # the actor: bitwise the loaded actor (slots zero) whatever the slots hold; the critic at the load too
            assert torch.equal(p.get_distribution(y).mode(), mu_load)
            assert torch.equal(p.forward(y, deterministic=True)[0], mu_load.clamp(-1e9, 1e9))
            assert torch.equal(p.evaluate_actions(y, acts)[1], p.evaluate_actions(x0, acts)[1])
            assert torch.equal(p.predict_values(y), v_load)
    # the caller's tensor is never written
    xc = x.clone()
    p.get_distribution(x)
    assert torch.equal(x, xc)


def test_critic_phase_gradients_actor_zero_critic_nonzero():
    """One backward pass of a value + policy loss: the actor's first-layer phase columns get exactly zero gradient,
    the critic's (zero-initialised) get a nonzero one; autograd on the input: d mu / d slots = 0 exactly."""
    p, env, m = _phase_setup(4)
    H.install_arm("critic_phase", m, env)
    x = torch.randn(32, OBS)
    x[:, SL] = torch.rand(32, 3)
    acts = torch.randn(32, ACT)
    values, logp, _ = p.evaluate_actions(x, acts)
    loss = ((values.flatten() - 1.0) ** 2).mean() - logp.mean()
    p.optimizer.zero_grad()
    loss.backward()
    g_pi = p.mlp_extractor.policy_net[0].weight.grad
    g_vf = p.mlp_extractor.value_net[0].weight.grad
    assert torch.all(g_pi[:, SL] == 0) and torch.any(g_pi[:, :60] != 0)
    assert torch.any(g_vf[:, SL] != 0)
    z = x.clone().requires_grad_(True)
    gz = torch.autograd.grad(p.get_distribution(z).mode().sum(), z)[0]
    assert torch.all(gz[:, SL] == 0) and torch.any(gz[:, :60] != 0)
    p.optimizer.step()
    assert torch.equal(p.mlp_extractor.policy_net[0].weight[:, SL], torch.full_like(g_pi[:, SL], -0.2))
    assert float(p.mlp_extractor.value_net[0].weight[:, SL].abs().max()) > 0.0
    assert H._adam_phase_max_abs(p, p.mlp_extractor.policy_net[0].weight) == 0.0
    assert H._adam_phase_max_abs(p, p.mlp_extractor.value_net[0].weight) > 0.0


def test_critic_phase_install_refusals():
    p, env, m = _phase_setup()
    env.obs_rms.mean = env.obs_rms.mean.copy()
    env.obs_rms.mean[H.PHASE_LO] = 1e-3                  # the slots carried a value
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    env.venv.envs[2]._command = np.array([0.0, 0.2, 0.0], dtype=np.float32)
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    env.venv = SimpleNamespace(envs=[])
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    perm = np.arange(OBS)
    perm[60], perm[59] = 59, 60
    m._mirror.mirror_map.obs.perm = perm
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    del m._mirror
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    x = torch.randn(8, OBS)
    loss = p.predict_values(x).sum() + p.get_distribution(x).mode().sum()
    p.optimizer.zero_grad()
    loss.backward()
    p.optimizer.step()                                  # nonzero Adam moments on the slot columns (x had values)
    with pytest.raises(H.HookError):
        H.install_arm("critic_phase", m, env)
    p, env, m = _phase_setup()
    H.install_arm("critic_phase", m, env)
    with pytest.raises(H.HookError):                     # twice
        H.install_arm("critic_phase", m, env)


def test_phase_rms_keeps_slots_other_dims_as_control_and_pickles_stock():
    from environments.shared.mirror_kl_ppo import install_symmetric_obs_rms

    a, b = _slot_free_rms(), _slot_free_rms()            # b: control's statistics, the same updates
    H.reseed_phase_slots(a)
    a.__class__ = H.phase_rms_class(type(a))
    c0 = H.phase_counter(a)["calls"]
    for j in range(4):
        data = np.random.default_rng(60 + j).normal(size=(13, OBS))
        b.update(np.where(np.arange(OBS) >= 60, 0.0, data))
        data[:, SL] = H.phase_matrix(np.arange(13) * 37)            # the phase in the slots
        a.update(data)
    assert H.phase_counter(a)["calls"] == c0 + 4 and H.phase_slots_frozen(a)
    assert np.array_equal(a.mean[:60], b.mean[:60]) and np.array_equal(a.var[:60], b.var[:60]) and a.count == b.count
    for clone in (pickle.loads(pickle.dumps(a)), copy.deepcopy(a)):
        assert type(clone) is RunningMeanStd and H.phase_slots_frozen(clone) and np.array_equal(clone.mean, a.mean)
    vn = SimpleNamespace(obs_rms=a, norm_obs=True)
    install_symmetric_obs_rms(vn, SimpleNamespace(obs=SimpleNamespace(perm=np.arange(OBS), sign=np.ones(OBS),
                                                                      dim=OBS)))
    assert vn.obs_rms is a                              # MirrorKLPPO.learn()'s _mirror_install keeps it


def test_phase_mirror_map_on_the_real_map():
    from environments.shared.model_id import ModelId
    from environments.shared.plant_contract import current_plant_identity
    from environments.shared.symmetry import MirrorMap, SignedPermutation

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    mm = MirrorMap.load(mid, current_plant_identity(mid), verify=False)
    assert mm.obs.sign[61] == -1.0 and mm.obs.sign[62] == -1.0 and mm.obs.sign[60] == 1.0
    new = H.phase_mirror_map(mm)
    assert isinstance(new.obs, SignedPermutation) and new.action is mm.action and new.sha256 == mm.sha256
    assert np.array_equal(new.obs.perm, mm.obs.perm) and np.all(new.obs.sign[SL] == 1.0)
    assert np.array_equal(new.obs.sign[:60], mm.obs.sign[:60]) and new.obs.names[60] == "critic_phase.ramp"
    assert mm.obs.sign[61] == -1.0                      # the run's map object is not written
    x = np.random.default_rng(1).normal(size=(5, OBS)).astype(np.float32)
    x[:, SL] = [0.3, 1.0, 0.7]
    assert np.array_equal(new.obs.apply(x)[:, SL], x[:, SL])


class PolicyProbeCallback:                              # a stand-in with the real class's name and bank shape
    def __init__(self, bank):
        self.bank = bank
        self.seen = []

    def _row(self):
        self.seen.append(np.array(self.bank.raw_obs, copy=True))
        return {"row": len(self.seen)}


def _bank():
    from typing import NamedTuple

    class Bank(NamedTuple):
        labels: tuple
        steps: np.ndarray
        raw_obs: np.ndarray

    return Bank(labels=("statue_settled", "statue", "ankle_minus_0.05_at400"), steps=np.array([199, 599, 400]),
                raw_obs=np.zeros((3, OBS)))


def test_probe_phase_feeds_the_bank_phase_and_restores_the_bank():
    probe = PolicyProbeCallback(_bank())
    rec = H.probe_phase("critic_phase", [SimpleNamespace(), probe])
    assert rec["installed"] and rec["k"] == [200, 600, 401] and rec["phase_fed"] == [[1.0, 1.0, 1.0]] * 3
    assert rec["class"] == "PhaseFedPolicyProbeCallback" and all(rec["settled"])
    bank = probe.bank
    assert probe._row() == {"row": 1} and probe.bank is bank and np.all(bank.raw_obs == 0.0)
    assert np.all(probe.seen[0][:, SL] == 1.0) and np.all(probe.seen[0][:, :60] == 0.0)
    with pytest.raises(H.HookError):
        H.probe_phase("critic_phase", [probe])          # already feeding
    plain = PolicyProbeCallback(_bank())
    rec = H.probe_phase("control", [plain])
    assert rec["installed"] is False and type(plain) is PolicyProbeCallback and rec["phase_fed"] is None
    with pytest.raises(H.HookError):
        H.probe_phase("control", [plain, PolicyProbeCallback(_bank())])
    with pytest.raises(H.HookError):
        H.probe_phase("control", [])
    with pytest.raises(H.HookError):
        H.probe_phase("critic_phase", [PolicyProbeCallback(None)])


def test_phase_evidence_zero_init_then_response():
    ev = _phase_evidence(3)
    s = H.summarize(ev)
    assert H.evidence_problems("critic_phase", s, n_expected=3) == [], H.evidence_problems("critic_phase", s)
    assert s["phase_n_zero_init"] == 1 and s["phase_n_updated"] == 2 and s["max_dmu_phase"] == 0.0
    assert s["zero_init_max_dv_phase"] == 0.0 and s["zero_init_max_value_phase_cols"] == 0.0
    assert all(v > 0 for v in s["updated_min_dv_phase"].values()) and s["max_phase_buffer_max_err"] == 0.0
    assert ev[0]["n_updates_since_load"] == 0 and ev[2]["n_updates_since_load"] == 14
    assert ev[0]["phase_k_start_consistent"] and ev[0]["phase_rows_presettle"] > 0
    assert ev[0]["dv_predict_values"] > 0                # the critic still sees observation 6
    assert H.evidence_problems("control", s, n_expected=3) != []


@pytest.mark.parametrize("mutate", ["unmask", "nonzero_init", "no_learning", "slot_error", "counter", "unfreeze",
                                    "mirror", "actor_cols"])
def test_phase_evidence_mutants_fail(mutate):
    s = H.summarize(_phase_evidence(3, mutate=mutate))
    assert H.evidence_problems("critic_phase", s, n_expected=3) != [], mutate


def test_phase_evidence_needs_its_context():
    p, env, m = _phase_setup()
    H.install_arm("critic_phase", m, env)
    buf, last, dones = _phase_buffer(p, env, [0, 0, 0, 0], seed=1)
    m.rollout_buffer = buf
    with pytest.raises(H.HookError):
        H.rollout_evidence("critic_phase", m, env, {"values": torch.as_tensor(last), "dones": dones},
                           start_obs_digest=None, phase_ctx=None)


def test_start_pair_actor_and_critic_bitwise_after_the_phase_transform():
    """The real start pair (read only): after the critic's phase columns are zeroed and the slots reseeded, the bank's
    actions and values with ANY phase fed equal the loaded pair's on the bank as recorded, bitwise."""
    from environments.shared.config import load_stage_config
    from environments.shared.model_id import ModelId
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_registry import env_class

    mid = ModelId("tyrannosaurus_rex", "anatomical")
    ctor = dict(load_stage_config(mid, "stance")["env_kwargs"])
    model, norm, _ = load_sb3_checkpoint(C.f2_file(C.START["zip_rel"]), C.f2_file(C.START["pkl_rel"]),
                                         lambda: env_class(mid)(**ctor))
    with np.load(C.f2_record("policy_probe.npz"), allow_pickle=False) as z:
        bank = np.asarray(z["bank_raw_obs"], dtype=np.float64)
    p = model.policy
    rms = norm.obs_rms

    def outs(raw, r):
        t = torch.as_tensor(np.clip((raw - r.mean) / np.sqrt(r.var + norm.epsilon), -10, 10), dtype=torch.float32)
        with torch.no_grad():
            return p.get_distribution(t).mode().clone(), p.predict_values(t).clone()

    mu0, v0 = outs(bank, rms)
    assert float(p.mlp_extractor.value_net[0].weight[:, SL].abs().max()) > 0.1
    with torch.no_grad():
        p.mlp_extractor.value_net[0].weight[:, SL] = 0.0
    r2 = copy.deepcopy(rms)
    H.reseed_phase_slots(r2)
    p.mlp_extractor.__class__ = H.critic_phase_class(MlpExtractor)
    for k in (0, 50, 199, 200, 600, 1000):
        fed = bank.copy()
        fed[:, SL] = H.phase_matrix([k] * len(bank))
        mu1, v1 = outs(fed, r2)
        assert torch.equal(mu0, mu1) and torch.equal(v0, v1), k
'''
p.write_text(s)
print("ok")
