"""PS build: the edits of read_bs.py (each old string asserted unique), run once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''"""Bias screen 2 (BS2): the reader.  Report only: it never trains; the loop markers simulate (deterministic rollouts and
stochastic on-policy rollouts with frozen statistics, no update, no optimizer).  BS's read_bs.py (sha256 04036e53...)''',
    '''"""Phase screen (PS): the reader.  Report only: it never trains; the loop markers simulate (deterministic rollouts and
stochastic on-policy rollouts with frozen statistics, no update, no optimizer).  PS = BS2's read_bs.py (sha256
06f4173d...) with PS's root, arms and validation seeds (bs_common.py) and the critic_phase reading: at every pair of a
critic_phase run the actor's mask is installed and the critic is fed the phase of each state's own step counter (REF:
the bank's kept step + 1 = 200, settled; OWN: the snapshot's counter, 500-1000, settled; ONPOLICY: each state's counter
recorded in the collection), with the slots-zero reading ("raw") and the all-settled reading beside it; at the start
copy (S) the arm's load transform is applied first (the critic's phase columns zeroed, the statistics' slots reseeded:
bs_hooks), every later pair must carry the frozen slots (mean 0 / var 1).  CHECK also reads critic_phase's run_meta
(the hook's verified transform, the probe's phase record, the split of the phase evidence before / after the first
update).  BS2 = BS's read_bs.py (sha256 04036e53...)''')
rep('''               and the raw value MLP are reported.''', '''               and the raw value MLP are reported; for a critic_phase run (PS) the EFFECTIVE critic is fed each
               state's phase, "raw" is the critic with the phase slots zero and "settled" feeds every state the
               settled phase (1, 1, 1); the record "phase" says what was fed.''')
rep('''VERSION = "read_bs 1.1 (2026-10-30; bias screen 2 from s42 2,097,152; derived from read_bs 1.0)"''',
    '''VERSION = "read_bs 2.0 (2026-10-31; phase screen from s42 2,097,152; derived from read_bs 1.1 of bias screen 2)"''')
rep('''def cache_key(zip_sha256, pkl_sha256, *, blind, onpolicy, own, reader=READER_SHA256, deps=DEPS_SHA256):
    """The loop cache's file stem: the reader's own sha256 and its modules' digest first (an entry never outlives the
    reader that wrote it: BS's process lens P3), then the pair and the read's form (pure)."""
    return (f"r{reader[:16]}_d{deps[:8]}_{zip_sha256[:16]}_{pkl_sha256[:16]}_{'blind' if blind else 'plain'}_"
            f"{'on' if onpolicy else 'off'}{'_own' if own else ''}")''', '''def cache_key(zip_sha256, pkl_sha256, *, blind, onpolicy, own, phase=False, start=False, reader=READER_SHA256,
              deps=DEPS_SHA256):
    """The loop cache's file stem: the reader's own sha256 and its modules' digest first (an entry never outlives the
    reader that wrote it: BS's process lens P3), then the pair and the read's form (pure).  PS: the form 'phase' (a
    critic_phase run's reading), with '_S' when the pair is read as the start copy (the arm's load transform applied)."""
    if blind and phase:
        raise ValueError("a read is blind or phase, not both")
    form = "blind" if blind else ("phase" + ("_S" if start else "") if phase else "plain")
    return (f"r{reader[:16]}_d{deps[:8]}_{zip_sha256[:16]}_{pkl_sha256[:16]}_{form}_"
            f"{'on' if onpolicy else 'off'}{'_own' if own else ''}")''')
# check_meta: critic_phase classes and records
rep('''    want_cls = {"critic_blind_tail3": ("CriticBlindTail3MlpExtractor", "SymmetricRunningMeanStd"),
                "frozen_obs_norm": ("MlpExtractor", "FrozenObsSymmetricRunningMeanStd")}.get(
        arm, ("MlpExtractor", "SymmetricRunningMeanStd"))
    if aa.get("arm") != arm or (aa.get("mlp_extractor_class"), aa.get("obs_rms_class")) != want_cls:
        fails.append(f"arm_applied {aa.get('arm')!r} {aa.get('mlp_extractor_class')!r} {aa.get('obs_rms_class')!r} "
                     f"!= {arm} {want_cls}")''', '''    want_cls = {"critic_blind_tail3": ("CriticBlindTail3MlpExtractor", "SymmetricRunningMeanStd"),
                "frozen_obs_norm": ("MlpExtractor", "FrozenObsSymmetricRunningMeanStd"),
                "critic_phase": ("CriticPhaseMlpExtractor", "PhaseSlotsSymmetricRunningMeanStd")}.get(
        arm, ("MlpExtractor", "SymmetricRunningMeanStd"))
    if aa.get("arm") != arm or (aa.get("mlp_extractor_class"), aa.get("obs_rms_class")) != want_cls:
        fails.append(f"arm_applied {aa.get('arm')!r} {aa.get('mlp_extractor_class')!r} {aa.get('obs_rms_class')!r} "
                     f"!= {arm} {want_cls}")
    fails += phase_meta_problems(meta, arm, kind)''')
rep('''def check_run(run_dir, root=None):''', '''def phase_meta_problems(meta, arm, kind):
    """PS: critic_phase's run_meta records (the verified load transform, the phase definition, the probe feeding the
    bank's phase, the evidence's split: one rollout before the first update in a continuation, all of them in a
    no-update run); any other arm: none of them (pure)."""
    fails = []
    aa, pp, pd = meta.get("arm_applied") or {}, meta.get("probe_phase") or {}, meta.get("phase_definition")
    ev = meta.get("arm_evidence") or {}
    want_def = {"slots": list(C.PHASE_SLOTS), "features": list(C.PHASE_FEATURES), "settle_steps": C.SETTLE_STEPS,
                "gamma": C.PHASE_GAMMA}
    d0, dl = meta.get("digest_start") or {}, meta.get("digest_load") or {}
    if arm != "critic_phase":
        if pd is not None or pp.get("installed") is not False or aa.get("transform_verified") is not None:
            fails.append(f"{arm}: phase records present (definition {pd is not None}, probe {pp.get('installed')!r}, "
                         f"transform {aa.get('transform_verified')!r})")
        if not dl or {k: dl.get(k) for k in ("params", "adam", "vecnormalize")} != {
                k: d0.get(k) for k in ("params", "adam", "vecnormalize")}:
            fails.append(f"{arm}: the hook changed the loaded state (digest_load != digest_start)")
        return fails
    if not isinstance(pd, dict) or {k: pd.get(k) for k in want_def} != want_def:
        fails.append(f"critic_phase: phase_definition {pd!r} != {want_def}")
    if aa.get("transform_verified") is not True or aa.get("phase_slots") != list(C.PHASE_SLOTS):
        fails.append("critic_phase: the load transform was not verified")
    if not dl or dl.get("adam") != d0.get("adam") or dl.get("params") == d0.get("params"):
        fails.append("critic_phase: digest_load / digest_start do not show the transform (Adam equal, params changed)")
    if pp.get("installed") is not True or pp.get("class") != "PhaseFedPolicyProbeCallback" or not pp.get("k") or \\
            any(k < C.SETTLE_STEPS for k in pp.get("k")) or pp.get("phase_fed") != [[1.0, 1.0, 1.0]] * len(pp["k"]):
        fails.append(f"critic_phase: the policy probe does not feed the settled bank's phase ({pp})")
    n = ev.get("n")
    z, u = ev.get("phase_n_zero_init"), ev.get("phase_n_updated")
    if kind == "continuation" and (z != 1 or u != (n - 1 if isinstance(n, int) else None)):
        fails.append(f"critic_phase: {z} rollouts before the first update and {u} after (1 and {n} - 1 expected)")
    if kind != "continuation" and (z != n or u != 0):
        fails.append(f"critic_phase: a no-update run with {z} zero-init and {u} updated rollouts (all zero-init)")
    return fails


def check_run(run_dir, root=None):''')
# own rollouts: record the counter at each snapshot
rep('''    eps, X = [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)''', '''    eps, X, K = [], [], []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)''')
rep('''            if k in SNAP_STEPS:
                X.append(np.array(obs, dtype=np.float64, copy=True))''', '''            if k in SNAP_STEPS:
                X.append(np.array(obs, dtype=np.float64, copy=True))
                K.append(int(env.unwrapped._step_count))          # PS: the snapshot's own counter (= k)''')
rep('''                    "frac99_win": float(np.mean(np.abs(np.asarray(w)) >= 0.99)) if len(own14) > 500 else None})
    return eps, np.array(X, dtype=np.float64)''', '''                    "frac99_win": float(np.mean(np.abs(np.asarray(w)) >= 0.99)) if len(own14) > 500 else None})
    return eps, np.array(X, dtype=np.float64), np.array(K, dtype=np.int64)''')
# on-policy states: record the counters
rep('''        def _on_rollout_start(self):
            self.prev = vn.get_original_obs().copy()

        def _on_step(self):
            self.raw.append(self.prev.copy())
            self.prev = vn.get_original_obs().copy()
            return True''', '''        def _on_rollout_start(self):
            self.prev = vn.get_original_obs().copy()
            self.prev_k = [int(e.unwrapped._step_count) for e in vn.venv.envs]   # PS: read only

        def _on_step(self):
            self.raw.append(self.prev.copy())
            self.k.append(list(self.prev_k))
            self.prev = vn.get_original_obs().copy()
            self.prev_k = [int(e.unwrapped._step_count) for e in vn.venv.envs]
            return True''')
rep('''            super().__init__(0)
            self.raw = []
''', '''            super().__init__(0)
            self.raw = []
            self.k = []
''')
rep('''    raw = np.concatenate(rec.raw, 0)
    vn.close()
    return raw''', '''    raw = np.concatenate(rec.raw, 0)
    ks = np.asarray(rec.k, dtype=np.int64).reshape(-1)
    vn.close()
    return raw, ks''')
rep('''def critic_slope(policy, norm, X):
    """(c) eb_realized.sens(): mean (SD) over in-clip states of dV/dz6 x 1/sqrt(var6 + eps) x pi/180, per degree."""
    import numpy as np
    import torch

    x = np.asarray(X, dtype=np.float32)''', '''def with_phase(X, K):
    """PS: X with the phase of each state's step counter K in slots 60-62 (float64 copy)."""
    import numpy as np

    Y = np.array(X, dtype=np.float64, copy=True)
    Y[:, list(C.PHASE_SLOTS)] = H.phase_matrix(K)
    return Y


def critic_slope(policy, norm, X, K=None):
    """(c) eb_realized.sens(): mean (SD) over in-clip states of dV/dz6 x 1/sqrt(var6 + eps) x pi/180, per degree.
    PS: with K (the states' step counters) the critic is fed their phase in slots 60-62."""
    import numpy as np
    import torch

    x = np.asarray(X if K is None else with_phase(X, K), dtype=np.float32)''')
rep('''def loop_markers_one(zip_path, pkl_path, *, blind, onpolicy, own, env, factory, bank_ss):
    """M-LOOP at one pair.  blind: the critic_blind_tail3 hook is installed for the EFFECTIVE critic (both reported)."""
    import numpy as np
    from stable_baselines3.common.torch_layers import MlpExtractor

    t0 = time.process_time()
    net = Net(zip_path, pkl_path, factory)''', '''def phase_prepare(net, *, start):
    """PS, a critic_phase pair: at the start copy the arm's load transform (bs_hooks: the critic's phase columns := 0,
    the statistics' slots := mean 0 / var 1); at any later pair the slots must already be mean 0 / var 1 (written by
    the arm); then the actor's mask.  Returns the record."""
    import torch
    from stable_baselines3.common.torch_layers import MlpExtractor

    w_vf = net.policy.mlp_extractor.value_net[0].weight
    lo, hi = C.PHASE_SLOTS[0], C.PHASE_SLOTS[-1] + 1
    rec = {"start_transform": bool(start), "value_phase_cols_max_abs_read": float(w_vf[:, lo:hi].abs().max())}
    if type(net.policy.mlp_extractor) is not MlpExtractor:
        raise InputError(f"the loaded mlp_extractor is {type(net.policy.mlp_extractor).__name__}")
    if start:
        with torch.no_grad():
            w_vf[:, lo:hi] = 0.0
        H.reseed_phase_slots(net.norm.obs_rms)
        net.refresh_stats()
    elif not H.phase_slots_frozen(net.norm.obs_rms):
        raise InputError("a critic_phase pair after S whose statistics' slots 60-62 are not mean 0 / var 1")
    net.policy.mlp_extractor.__class__ = H.critic_phase_class(MlpExtractor)
    rec["value_phase_cols_max_abs_used"] = float(w_vf[:, lo:hi].abs().max())
    return rec


def loop_markers_one(zip_path, pkl_path, *, blind, onpolicy, own, env, factory, bank_ss, phase=False, start=False,
                     bank_k=None):
    """M-LOOP at one pair.  blind: the critic_blind_tail3 hook is installed for the EFFECTIVE critic (both reported).
    phase (PS): critic_phase's reading (phase_prepare; the critic fed each state's phase; "raw" with the slots zero;
    "settled" with every state's phase (1, 1, 1)); start: the pair is the start copy (S)."""
    import numpy as np
    from stable_baselines3.common.torch_layers import MlpExtractor

    t0 = time.process_time()
    net = Net(zip_path, pkl_path, factory)
    phase_rec = phase_prepare(net, start=start) if phase else None''')
rep('''           "t3_norm_sd_deg": math.degrees(net.std[C.OBS_TAIL3]), "critic_form": "blind+raw" if blind else "plain"}
    out["REF"] = tail_sweeps(net, bank_ss)
    sets = {"REF": bank_ss}
    if own:
        eps, X = own_rollouts(net, env)
        sets["OWN"] = X''', '''           "t3_norm_sd_deg": math.degrees(net.std[C.OBS_TAIL3]),
           "critic_form": "blind+raw" if blind else ("phase+raw+settled" if phase else "plain")}
    out["REF"] = tail_sweeps(net, bank_ss)
    sets = {"REF": bank_ss}
    ks = {"REF": np.asarray(bank_k if bank_k is not None else [C.SETTLE_STEPS] * len(bank_ss), dtype=np.int64)}
    if own:
        eps, X, K_own = own_rollouts(net, env)
        sets["OWN"] = X
        ks["OWN"] = K_own''')
rep('''    if onpolicy:
        sets["ONPOLICY"] = np.concatenate([onpolicy_states(zip_path, pkl_path, s) for s in ONPOLICY_SEEDS], 0)
    crit = {}
    for name, X in sets.items():
        crit[name] = {"raw": critic_slope(net.policy, net.norm, X)}
    if blind:''', '''    if onpolicy:
        got = [onpolicy_states(zip_path, pkl_path, s) for s in ONPOLICY_SEEDS]
        sets["ONPOLICY"] = np.concatenate([g[0] for g in got], 0)
        ks["ONPOLICY"] = np.concatenate([g[1] for g in got], 0)
    crit = {}
    for name, X in sets.items():
        crit[name] = {"raw": critic_slope(net.policy, net.norm, X)}
    if phase:
        for name, X in sets.items():
            crit[name]["effective"] = critic_slope(net.policy, net.norm, X, K=ks[name])
            crit[name]["settled"] = critic_slope(net.policy, net.norm, X, K=np.full(len(X), C.SETTLE_STEPS))
        mu_plain, mu_fed = net.mu(bank_ss), net.mu(with_phase(bank_ss, ks["REF"]))
        phase_rec.update(
            actor_phase_invariance_max_abs=float(np.max(np.abs(mu_fed - mu_plain))),
            fed={n: {"k_min": int(k.min()), "k_max": int(k.max()), "n": int(len(k)),
                     "n_presettle": int((k < C.SETTLE_STEPS).sum())} for n, k in ks.items()},
            rule={"effective": "each state's own phase: REF kept step + 1, OWN the snapshot's counter, ONPOLICY the "
                               "counter recorded in the collection", "raw": "slots 60-62 zero (no phase input)",
                  "settled": "every state at k = 200: (1, 1, 1)"})
        out["phase"] = phase_rec
    elif blind:''')
rep('''    out["critic_slope_obs6"] = crit
    out["cpu_seconds"] = round(time.process_time() - t0, 1)
    return out''', '''    out["critic_slope_obs6"] = crit
    out["cpu_seconds"] = round(time.process_time() - t0, 1)
    return out''')
rep('''def loop_markers(pairs, *, blind, onpolicy, own, tag):
    """M-LOOP over pairs, with the cache under <BS>/reads/_cache/."""''', '''def loop_markers(pairs, *, blind, onpolicy, own, tag, phase=False):
    """M-LOOP over pairs, with the cache under <BS>/reads/_cache/.  phase (PS): critic_phase's reading; the pair at
    steps S is the start copy (the arm's load transform applied)."""''')
rep('''        bank_ss = np.asarray(z["bank_raw_obs"], dtype=np.float64)[list(C.BANK_STATUE_SETTLED)]''',
    '''        bank_ss = np.asarray(z["bank_raw_obs"], dtype=np.float64)[list(C.BANK_STATUE_SETTLED)]
        bank_k = np.asarray(z["bank_steps"], dtype=np.int64)[list(C.BANK_STATUE_SETTLED)] + 1     # PS: k = 200''')
rep('''        key = cache_key(C.sha256_file(z), C.sha256_file(p), blind=blind, onpolicy=onpolicy, own=own)''',
    '''        key = cache_key(C.sha256_file(z), C.sha256_file(p), blind=blind, onpolicy=onpolicy, own=own, phase=phase,
                        start=phase and steps == C.S)''')
rep('''            row = loop_markers_one(z, p, blind=blind, onpolicy=onpolicy, own=own, env=env, factory=factory,
                                   bank_ss=bank_ss)''', '''            row = loop_markers_one(z, p, blind=blind, onpolicy=onpolicy, own=own, env=env, factory=factory,
                                   bank_ss=bank_ss, phase=phase, start=phase and steps == C.S, bank_k=bank_k)''')
rep('''    markers["loop"] = loop_markers(saved_pairs(run, T), blind=(arm == "critic_blind_tail3"), onpolicy=onpolicy,
                                   own=own, tag=run.name)''', '''    markers["probe_phase"] = meta.get("probe_phase")
    markers["loop"] = loop_markers(saved_pairs(run, T), blind=(arm == "critic_blind_tail3"), onpolicy=onpolicy,
                                   own=own, tag=run.name, phase=(arm == "critic_phase"))''')
# Net.refresh_stats
rep('''        self.std = np.sqrt(self.var + self.eps)
        self.clip = float(self.norm.clip_obs)
''', '''        self.std = np.sqrt(self.var + self.eps)
        self.clip = float(self.norm.clip_obs)

    def refresh_stats(self):
        """PS: re-read the normaliser's statistics (after critic_phase's start transform reseeded slots 60-62)."""
        import numpy as np

        rms = self.norm.obs_rms
        self.mean = np.asarray(rms.mean, dtype=np.float64)
        self.var = np.asarray(rms.var, dtype=np.float64)
        self.std = np.sqrt(self.var + self.eps)
''')
p.write_text(s)
print("ok")
