"""PS build: the edits of tests/test_read_bs.py (each old string asserted unique), run once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''VS = C.VALIDATION_SEEDS[0]                             # BS2: 9201 (BS's tests used 9101)''',
    '''VS = C.VALIDATION_SEEDS[0]                             # PS: 9401 (BS2's tests used 9201, BS's 9101)''')
rep('''def _good_meta(arm, kind="validation", T=S + 2 * R8):
    lam = C.ARM_GAE_LAMBDA[arm]
    cls = {"critic_blind_tail3": ("CriticBlindTail3MlpExtractor", "SymmetricRunningMeanStd"),
           "frozen_obs_norm": ("MlpExtractor", "FrozenObsSymmetricRunningMeanStd")}.get(
        arm, ("MlpExtractor", "SymmetricRunningMeanStd"))
    d0 = {"params": "p0", "adam": "a0", "n_updates": 10}
    d1 = dict(d0) if kind == "validation" else {"params": "p1", "adam": "a1", "n_updates": 30}''',
    '''def _good_meta(arm, kind="validation", T=S + 2 * R8):
    lam = C.ARM_GAE_LAMBDA[arm]
    cls = {"critic_blind_tail3": ("CriticBlindTail3MlpExtractor", "SymmetricRunningMeanStd"),
           "frozen_obs_norm": ("MlpExtractor", "FrozenObsSymmetricRunningMeanStd"),
           "critic_phase": ("CriticPhaseMlpExtractor", "PhaseSlotsSymmetricRunningMeanStd")}.get(
        arm, ("MlpExtractor", "SymmetricRunningMeanStd"))
    d0 = {"params": "p0", "adam": "a0", "vecnormalize": "v0", "n_updates": 10}
    d1 = dict(d0) if kind == "validation" else {"params": "p1", "adam": "a1", "n_updates": 30}
    phase = arm == "critic_phase"
    dl = {**d0, "params": "pL", "vecnormalize": "vL"} if phase else dict(d0)
    n = C.expected_update_rows(C.S, T)
    aa = {"arm": arm, "mlp_extractor_class": cls[0], "obs_rms_class": cls[1]}
    if phase:
        aa.update(transform_verified=True, phase_slots=list(C.PHASE_SLOTS))
    k = [200, 600, 600, 600, 401, 200, 600, 600, 600, 401]
    probe = ({"installed": True, "class": "PhaseFedPolicyProbeCallback", "k": k, "phase_fed": [[1.0, 1.0, 1.0]] * 10}
             if phase else {"installed": False, "class": "PolicyProbeCallback", "k": k, "phase_fed": None})
    ev = {"n": n, "phase_n_zero_init": (n if kind == "validation" else 1) if phase else None,
          "phase_n_updated": (0 if kind == "validation" else n - 1) if phase else None}
    pdef = ({"slots": list(C.PHASE_SLOTS), "features": list(C.PHASE_FEATURES), "settle_steps": C.SETTLE_STEPS,
             "gamma": C.PHASE_GAMMA, "k": "x"} if phase else None)''')
rep('''            "arm_applied": {"arm": arm, "mlp_extractor_class": cls[0], "obs_rms_class": cls[1]},''',
    '''            "arm_applied": aa, "probe_phase": probe, "phase_definition": pdef, "arm_evidence": ev,
            "digest_load": dl,''')
rep('''    m = _good_meta(arm)
    m.update(seed=9101, name=C.run_name(arm, 9101))      # BS's validation seed is not BS2's
    assert any("validation seed" in f for f in RB.check_meta(m, "validation", C.run_name(arm, 9101))[0])''',
    '''    for seed in (9101, 9201):                            # BS's and BS2's validation seeds are not PS's
        m = _good_meta(arm)
        m.update(seed=seed, name=C.run_name(arm, seed))
        assert any("validation seed" in f for f in RB.check_meta(m, "validation", C.run_name(arm, seed))[0])


@pytest.mark.parametrize("arm", C.ARMS)
def test_check_meta_phase_records(arm):
    """PS: critic_phase's run_meta must show the verified transform, the phase definition, the probe feeding the
    settled bank's phase and the evidence split (no-update: every rollout before an update; continuation: exactly
    the first); control must show none of them and an unchanged loaded state."""
    name = C.run_name(arm, VS)
    assert RB.check_meta(_good_meta(arm), "validation", name)[0] == []
    mutants = []
    if arm == "critic_phase":
        mutants += [("arm_applied", "transform_verified", False), ("arm_applied", "phase_slots", [57, 58, 59]),
                    ("probe_phase", "installed", False), ("probe_phase", "class", "PolicyProbeCallback"),
                    ("probe_phase", "k", [199] + [600] * 9), ("phase_definition", "settle_steps", 201),
                    ("phase_definition", "gamma", 0.99), ("arm_evidence", "phase_n_zero_init", 0),
                    ("arm_evidence", "phase_n_updated", 1), ("digest_load", "adam", "aX"),
                    ("digest_load", "params", "p0")]
    else:
        mutants += [("probe_phase", "installed", True), ("arm_applied", "transform_verified", True),
                    ("digest_load", "params", "pX"), ("digest_load", "vecnormalize", "vX")]
    for top, key, val in mutants:
        m = _good_meta(arm)
        m[top] = dict(m[top] or {}, **{key: val})
        assert RB.check_meta(m, "validation", name)[0] != [], (top, key, val)
    m = _good_meta(arm)
    m["phase_definition"] = None if arm == "critic_phase" else {"slots": [60, 61, 62]}
    assert RB.check_meta(m, "validation", name)[0] != []


def test_phase_meta_continuation_split():
    m = _good_meta("critic_phase", kind="continuation", T=C.E)
    assert m["arm_evidence"] == {"n": 175, "phase_n_zero_init": 1, "phase_n_updated": 174}
    assert RB.phase_meta_problems(m, "critic_phase", "continuation") == []
    assert RB.phase_meta_problems(m, "critic_phase", "validation") != []
    m["arm_evidence"] = {"n": 175, "phase_n_zero_init": 175, "phase_n_updated": 0}
    assert RB.phase_meta_problems(m, "critic_phase", "continuation") != []''')
rep('''def _ev_lines(arm, n):
    base = {"buffer_gae_lambda": C.ARM_GAE_LAMBDA[arm], "model_gae_lambda": C.ARM_GAE_LAMBDA[arm], "adv_scale": 1.0,''',
    '''def _phase_keys(arm, k):
    """PS: the phase evidence of rollout k (k = 0 before any update)."""
    phase = arm == "critic_phase"
    out = {key: (0.0 if phase else 0.2) for key in H.PHASE_ACTOR_KEYS}
    out.update({key: (0.0 if phase and k == 0 else 0.004) for key in H.PHASE_CRITIC_KEYS})
    if not phase:
        out.update({key: None for key in H.PHASE_STRUCT_KEYS})
        return out
    out.update(phase_rows=8192, phase_buffer_max_err=0.0, phase_k_start_consistent=True, phase_k_end_consistent=True,
               phase_k_min=0, phase_k_max=999, phase_rows_presettle=1600, phase_stats_frozen=True,
               phase_rms_calls=2049 * (k + 1), mirror_phase_self_map=True, mirror_phase_tensors_ok=None if k == 0
               else True, envs_swapped=True, value_phase_cols_max_abs=0.0 if k == 0 else 0.002 * k,
               policy_phase_cols_unchanged=True, policy_phase_adam_max_abs=0.0, n_updates_since_load=7 * k,
               adam_steps_since_load=float(400 * k))
    return out


def _ev_lines(arm, n):
    base = {"buffer_gae_lambda": C.ARM_GAE_LAMBDA[arm], "model_gae_lambda": C.ARM_GAE_LAMBDA[arm], "adv_scale": 1.0,''')
rep('''                 frozen_update={"calls": 2 * (k + 1), "rows": 8 * (k + 1)} if frozen else None)
        out.append(e)
    return out''', '''                 frozen_update={"calls": 2 * (k + 1), "rows": 8 * (k + 1)} if frozen else None)
        e.update(_phase_keys(arm, k))
        out.append(e)
    return out''')
rep('''def test_decided_and_outcome_blind(tmp_path):
    q = [["control", 2101], ["gae_lambda_1", 2101]]
    (tmp_path / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "gae_lambda_1",
                                                      "queue": q}))
    ok, und = RB.decided(tmp_path)
    assert not ok and und == ["control_2101", "gae_lambda_1_2101"]
    for name in ("control_2101",):
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / C.CHECK_JSON).write_text(json.dumps({"passed": True, "mode": "continuation"}))
    assert RB.decided(tmp_path) == (False, ["gae_lambda_1_2101"])
    (tmp_path / "lost.json").write_text(json.dumps({"schema": C.LOST_SCHEMA, "lost": {"gae_lambda_1_2101": {}}}))''',
    '''def test_decided_and_outcome_blind(tmp_path):
    q = [["control", 2101], ["critic_phase", 2101]]
    (tmp_path / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "critic_phase",
                                                      "queue": q}))
    ok, und = RB.decided(tmp_path)
    assert not ok and und == ["control_2101", "critic_phase_2101"]
    for name in ("control_2101",):
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / C.CHECK_JSON).write_text(json.dumps({"passed": True, "mode": "continuation"}))
    assert RB.decided(tmp_path) == (False, ["critic_phase_2101"])
    (tmp_path / "lost.json").write_text(json.dumps({"schema": C.LOST_SCHEMA, "lost": {"critic_phase_2101": {}}}))''')
rep('''    assert RB.cache_key(z, p, blind=False, onpolicy=True, own=False) != k''', '''    assert RB.cache_key(z, p, blind=False, onpolicy=True, own=False) != k
    # PS: critic_phase's reading has its own form, and the start copy (the load transform applied) another
    kp = RB.cache_key(z, p, blind=False, onpolicy=True, own=True, phase=True)
    ks = RB.cache_key(z, p, blind=False, onpolicy=True, own=True, phase=True, start=True)
    assert kp.endswith("_phase_on_own") and ks.endswith("_phase_S_on_own") and len({k, kp, ks}) == 3
    assert RB.cache_key(z, p, blind=False, onpolicy=True, own=True, start=True) == k     # start: phase reads only
    with pytest.raises(ValueError):
        RB.cache_key(z, p, blind=True, onpolicy=True, own=True, phase=True)''')
p.write_text(s)
print("ok")
