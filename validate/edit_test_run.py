"""PS build: the edits of tests/test_run_bs.py (each old string asserted unique), run once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new, count=1):
    global s
    assert s.count(old) == count, (s.count(old), old[:90])
    s = s.replace(old, new)


rep('''V0, V4 = C.VALIDATION_SEEDS[0], C.VALIDATION_SEEDS[-1]     # BS2: 9201, 9241 (BS's tests used 9101 / 9141)''',
    '''V0, V4 = C.VALIDATION_SEEDS[0], C.VALIDATION_SEEDS[-1]     # PS: 9401, 9441 (BS2's tests used 9201 / 9241)''')
rep('''    assert C.WAVE_SLOTS == 4 and C.WAVE_NICE == 10 and C.MAX_CONTINUATIONS == 9
    assert C.DISK_FLOOR_BYTES == 1_610_612_736 and C.LAUNCH_DISK_BYTES == 1_610_612_736 + 9 * 50 * 1024**2
    assert C.BS.endswith("/r10round/bias_screen2") and C.CODE == C.BS1 + "/code" and C.BS1 != C.BS''',
    '''    assert C.WAVE_SLOTS == 4 and C.WAVE_NICE == 10 and C.MAX_CONTINUATIONS == 8
    assert C.DISK_FLOOR_BYTES == 1_610_612_736 and C.LAUNCH_DISK_BYTES == 1_610_612_736 + 8 * 50 * 1024**2
    assert C.BS == "/home/user/work/phase_screen" and C.CODE == C.BS1 + "/code" and C.BS1 != C.BS
    assert C.BS1.endswith("/r10round/bias_screen") and C.BS2_DIR.endswith("/r10round/bias_screen2")''')
rep('''def test_validation_seeds_are_bs2s():
    assert C.VALIDATION_SEEDS == (9201, 9211, 9221, 9231, 9241)
    assert all(s not in C.VALIDATION_SEEDS for s in (2101, 2111, 2121, 2131))''',
    '''def test_validation_seeds_are_pss():
    assert C.VALIDATION_SEEDS == (9401, 9411, 9421, 9431, 9441)
    assert all(s not in C.VALIDATION_SEEDS for s in (2101, 2111, 2121, 2131, 9201, 9241, 9101, 9301))


def test_phase_definition():
    """PS: the phase features of a step counter k (bs_common THE PHASE), the off-by-one at the gate, the slots."""
    assert C.PHASE_SLOTS == (60, 61, 62) and C.OBS_DIM == 63 and C.SETTLE_STEPS == 200 and C.PHASE_GAMMA == 0.98
    assert C.PHASE_FEATURES == ("ramp", "settled", "discount")
    assert C.phase_features(0) == (0.0, 0.0, 0.98 ** 200)
    assert C.phase_features(100) == (0.5, 0.0, 0.98 ** 100)
    assert C.phase_features(199) == (0.995, 0.0, 0.98)          # its own reward: _step_count 200, not > 200: unpaid
    assert C.phase_features(200) == (1.0, 1.0, 1.0)             # its own reward: _step_count 201 > 200: the first paid
    assert all(C.phase_features(k) == (1.0, 1.0, 1.0) for k in (201, 599, 600, 999, 1000, 5000))
    ramps = [C.phase_features(k)[0] for k in range(0, 201)]
    discs = [C.phase_features(k)[2] for k in range(0, 201)]
    assert ramps == sorted(ramps) and discs == sorted(discs) and len(set(ramps)) == 201
    for bad in (-1, 1.5, True, "3"):
        with pytest.raises(ValueError):
            C.phase_features(bad)
    assert all(R.H.phase_tuple(k) == C.phase_features(k) for k in range(0, 1201))
    assert (R.H.PHASE_SLOTS, R.H.SETTLE_STEPS, R.H.PHASE_GAMMA, R.H.OBS_DIM, R.H.PHASE_FEATURES) == (
        C.PHASE_SLOTS, C.SETTLE_STEPS, C.PHASE_GAMMA, C.OBS_DIM, C.PHASE_FEATURES)''')
rep('''    assert C.parse_run_name("gae_lambda_1_2301")["arm"] == "gae_lambda_1"
    assert C.parse_run_name("s42_A_1101") is None and C.parse_run_name("critic_2101") is None''',
    '''    assert C.parse_run_name("critic_phase_2301")["arm"] == "critic_phase"
    assert C.parse_run_name("s42_A_1101") is None and C.parse_run_name("critic_2101") is None
    assert C.parse_run_name("critic_blind_tail3_2101") is None and C.parse_run_name("gae_lambda_1_2101") is None''')
rep('''def _screen(inter="critic_blind_tail3", queue=None):''', '''def _screen(inter="critic_phase", queue=None):''')
rep('''    assert C.queue_of(_screen())[1] == ("critic_blind_tail3", 2101)
    nine = [["control", 2000 + i] for i in range(5)] + [["frozen_obs_norm", 3000 + i] for i in range(4)]
    assert C.screen_problems(_screen("frozen_obs_norm", nine)) == []
    bad = {
        "ten": _screen("frozen_obs_norm", nine + [["control", 2999]]),
        "unknown intervention": _screen("ent_coef_0"),
        "two interventions": _screen(queue=[["control", 1], ["critic_blind_tail3", 2], ["gae_lambda_1", 3]]),
        "control only": _screen(queue=[["control", 1], ["control", 2]]),
        "validation seed": _screen(queue=[["control", V0], ["critic_blind_tail3", 2]]),
        "repeat": _screen(queue=[["control", 5], ["control", 5], ["critic_blind_tail3", 2]]),
        "bool seed": _screen(queue=[["control", True], ["critic_blind_tail3", 2]]),''',
    '''    assert C.queue_of(_screen())[1] == ("critic_phase", 2101)
    eight = [["control", 2000 + i] for i in range(4)] + [["critic_phase", 3000 + i] for i in range(4)]
    assert C.screen_problems(_screen("critic_phase", eight)) == []
    bad = {
        "nine": _screen("critic_phase", eight + [["control", 2999]]),
        "unknown intervention": _screen("ent_coef_0"),
        "a BS2 arm": _screen("critic_blind_tail3"),
        "two interventions": _screen(queue=[["control", 1], ["critic_phase", 2], ["gae_lambda_1", 3]]),
        "control only": _screen(queue=[["control", 1], ["control", 2]]),
        "validation seed": _screen(queue=[["control", V0], ["critic_phase", 2]]),
        "BS2 validation seed is fine but PS's is not": _screen(queue=[["control", 9201], ["critic_phase", V4]]),
        "repeat": _screen(queue=[["control", 5], ["control", 5], ["critic_phase", 2]]),
        "bool seed": _screen(queue=[["control", True], ["critic_phase", 2]]),''')
rep('''    assert set(C.ARM_PPO_EDIT) == set(C.ARMS) == set(C.ARM_GAE_LAMBDA) == set(C.ARM_CHANGE)
    assert C.ARMS[0] == "control" and C.INTERVENTIONS == C.ARMS[1:]
    assert C.ARM_PPO_EDIT["gae_lambda_1"] == {"gae_lambda": 1.0}
    assert all(not C.ARM_PPO_EDIT[a] for a in C.ARMS if a != "gae_lambda_1")
    assert R.H.GAE_LAMBDA == C.ARM_GAE_LAMBDA and R.H.ARMS == C.ARMS and R.H.OBS_TAIL3 == C.OBS_TAIL3''',
    '''    assert set(C.ARM_PPO_EDIT) == set(C.ARMS) == set(C.ARM_GAE_LAMBDA) == set(C.ARM_CHANGE)
    assert C.ARMS == ("control", "critic_phase") and C.INTERVENTIONS == C.ARMS[1:]
    assert all(not C.ARM_PPO_EDIT[a] for a in C.ARMS) and set(C.ARM_GAE_LAMBDA.values()) == {0.95}
    # bs_hooks keeps BS2's three arms (unreachable in PS: preflight, SCREEN and CHECK use bs_common.ARMS)
    assert set(C.ARMS) <= set(R.H.ARMS) and all(R.H.GAE_LAMBDA[a] == C.ARM_GAE_LAMBDA[a] for a in C.ARMS)
    assert R.H.OBS_TAIL3 == C.OBS_TAIL3''')
rep('''    hp = "sha256:" + "1" * 64 if arm == "gae_lambda_1" else C.HYPERPARAMETERS_SHA256_RUN''',
    '''    hp = C.HYPERPARAMETERS_SHA256_RUN                  # PS: no arm edits [ppo]''')
rep('''    w["hyperparameters"]["gae_lambda"] = 1.0 if arm != "gae_lambda_1" else 0.95''',
    '''    w["hyperparameters"]["gae_lambda"] = 1.0''')
rep('''    bad_hp = C.HYPERPARAMETERS_SHA256_RUN if arm == "gae_lambda_1" else "sha256:" + "2" * 64''',
    '''    bad_hp = "sha256:" + "2" * 64''')
rep('''    wrong = 0.95 if arm == "gae_lambda_1" else 1.0''', '''    wrong = 1.0''')
rep('''    cb.model.num_timesteps += 4
    with pytest.raises(RuntimeError):
        cb._on_step()
''', '''    cb.model.num_timesteps += 4
    with pytest.raises(RuntimeError):
        cb._on_step()


def test_stop_at_end_training_and_rollout_start_hooks():
    """PS: training_start gets the flattened siblings (after the earlier callbacks' starts); rollout_start runs at each
    rollout's start; neither is called when not given (control's rollout_start)."""
    calls = []
    cls = R.make_stop_at_end_class(_BaseCB)
    inner = SimpleNamespace(callbacks=[SimpleNamespace(name="probe"), SimpleNamespace(name="eval")])
    cb = cls(C.E, training_start=lambda sib: calls.append(("ts", [x.name for x in sib])),
             rollout_start=lambda: calls.append(("rs",)))
    cb.locals = {"callback": SimpleNamespace(callbacks=[inner, SimpleNamespace(name="stop")])}
    cb._on_training_start()
    cb._on_rollout_start()
    assert calls == [("ts", ["probe", "eval", "stop"]), ("rs",)]
    plain = cls(C.E)
    plain.locals = {"callback": None}
    plain._on_training_start()
    plain._on_rollout_start()
    assert plain.siblings == []
''')
rep('''    assert any("validation seed" in x for x in p)          # BS's validation seeds are not BS2's
    assert any("arm" in x for x in R.preflight(_ns(arm="B", save_loaded_only=True)))''',
    '''    assert any("validation seed" in x for x in p)          # BS's validation seeds are not PS's
    p = R.preflight(_ns(no_update=True, seed=9201, until=C.S + C.ROLLOUT, out=str(Path(C.VALIDATE) / "nm")))
    assert any("validation seed" in x for x in p)          # nor BS2's
    assert any("arm" in x for x in R.preflight(_ns(arm="B", save_loaded_only=True)))
    for bs2_arm in ("critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm"):   # BS2's arms are not PS's
        assert any("arm" in x for x in R.preflight(_ns(arm=bs2_arm, save_loaded_only=True, seed=V4)))
    assert R.preflight(_ns(arm="critic_phase", save_loaded_only=True, seed=V4,
                           out=str(Path(C.VALIDATE) / "never_made"))) == []''')
rep('''    assert R.screen_entry_problems("critic_blind_tail3", 2111) == []
    assert R.screen_entry_problems("critic_blind_tail3", 2121) != []
    assert R.screen_entry_problems("gae_lambda_1", 2101) != []
    (tmp_path / "SCREEN.json").write_text(json.dumps(_screen(queue=[["control", 2101], ["gae_lambda_1", 2101]])))''',
    '''    assert R.screen_entry_problems("critic_phase", 2111) == []
    assert R.screen_entry_problems("critic_phase", 2121) != []
    assert R.screen_entry_problems("critic_blind_tail3", 2101) != []
    (tmp_path / "SCREEN.json").write_text(json.dumps(_screen(queue=[["control", 2101], ["critic_phase", 2131]])))''')
p.write_text(s)
print("ok")
