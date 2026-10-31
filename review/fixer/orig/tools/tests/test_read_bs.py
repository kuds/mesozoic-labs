"""Unit tests of read_bs.py on synthetic inputs (no simulation): the probe check, the evidence check, run_meta's check,
M5's window reading, the ledger's direct reduction, the probe markers, decided() and the outcome-blind refusal.

    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <BS>/tools/tests/test_read_bs.py
"""
from __future__ import annotations

import copy
import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("read_bs", str(TOOLS / "read_bs.py"))
RB = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(RB)
C, H = RB.C, RB.H
S, R8 = C.S, C.ROLLOUT
VS = C.VALIDATION_SEEDS[0]                             # PS: 9401 (BS2's tests used 9201, BS's 9101)
N_PARENT = C.E // R8 + 8                               # a synthetic parent probe that runs past E (BS's used 399 rows)


def _rms(mean, var, count):
    return SimpleNamespace(mean=np.asarray(mean, dtype=np.float64), var=np.asarray(var, dtype=np.float64),
                           count=float(count))


def _probe_cols(T, frozen=False, parent=False, start=None):
    n_seed = S // R8                                   # rows at or before S, as a resumed probe keeps them
    steps = [R8 * k for k in range(1, n_seed + 1)] + [S + R8 * k for k in range(1, (T - S) // R8)]
    if parent:
        steps = [R8 * k for k in range(1, N_PARENT)]
    n = len(steps)
    mean = np.tile(np.asarray(start.mean, np.float32), (n, 1))
    var = np.tile(np.asarray(start.var, np.float32), (n, 1))
    cnt = np.full(n, start.count)
    if not frozen:
        mean[-5:] += 0.01
        cnt[-5:] += 100
    ent = np.linspace(0.005, 0.004, n).astype(np.float32)
    return {"update_timesteps": np.asarray(steps, np.int64), "ent_coef": ent,
            "learning_rate": np.full(n, 3e-5, np.float32), "obs_rms_mean": mean, "obs_rms_var": var,
            "obs_rms_count": cnt}


def _parent_cols():
    steps = np.asarray([R8 * k for k in range(1, N_PARENT)], np.int64)
    return {"update_timesteps": steps, "ent_coef": np.linspace(0.005, 0.004, len(steps)).astype(np.float32),
            "learning_rate": np.full(len(steps), 3e-5, np.float32)}


def test_probe_check_arms():
    start = _rms(np.zeros(63), np.ones(63), 10.0)
    T = S + 4 * R8
    par = _parent_cols()
    pidx = {int(u): i for i, u in enumerate(par["update_timesteps"])}
    for arm in C.ARMS:
        cols = _probe_cols(T, frozen=(arm == "frozen_obs_norm"), start=start)
        # the run's ent_coef / lr at the window rows = the parent's at the same steps
        for col in ("ent_coef", "learning_rate"):
            for i, u in enumerate(cols["update_timesteps"]):
                cols[col][i] = par[col][pidx[int(u)]]
        fails, det = RB.probe_check(cols, par, arm, S, T, SimpleNamespace(obs_rms=start))
        assert fails == [], (arm, fails)
        assert det["rows_in_window"] == 3
        other = copy.deepcopy(cols)
        if arm == "frozen_obs_norm":
            other["obs_rms_mean"][-1] += 1.0
        else:
            other["obs_rms_mean"][-5:] = np.asarray(start.mean, np.float32)
            other["obs_rms_count"][-5:] = start.count
        assert RB.probe_check(other, par, arm, S, T, SimpleNamespace(obs_rms=start))[0] != []
        late = copy.deepcopy(cols)
        late["update_timesteps"][-1] = T
        assert RB.probe_check(late, par, arm, S, T, SimpleNamespace(obs_rms=start))[0] != []
        ent = copy.deepcopy(cols)
        ent["ent_coef"][-1] = 0.0
        assert RB.probe_check(ent, par, arm, S, T, SimpleNamespace(obs_rms=start))[0] != []


def _good_meta(arm, kind="validation", T=S + 2 * R8):
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
             "gamma": C.PHASE_GAMMA, "k": "x"} if phase else None)
    return {"schema": C.RUN_SCHEMA, "name": C.run_name(arm, VS), "arm": arm, "seed": VS, "S": C.S, "E": C.E,
            "mode": "no_update" if kind == "validation" else "continuation", "status": "done", "exit": 0,
            "code": {"tree_digest": C.TREE_DIGEST, "commit": C.COMMIT}, "code_seal_after": "unchanged",
            "modules_outside_code": [],
            "start_pair": {"zip_sha256": C.START["zip_sha256"], "pkl_sha256": C.START["pkl_sha256"]},
            "parent_records": {k: C.START["records"][k] for k in C.PARENT_RECORDS},
            "arm_ppo": dict(C.ARM_PPO_EDIT[arm]) or "unchanged", "arm_change": C.ARM_CHANGE[arm],
            "ppo_applied": {"gae_lambda": lam}, "gae_lambda_after_load": lam, "ent_coef_after_load": 0.005,
            "arm_applied": aa, "probe_phase": probe, "phase_definition": pdef, "arm_evidence": ev,
            "digest_load": dl,
            "arm_state_end": "in force", "obs_rms_end_equal_start": arm == "frozen_obs_norm",
            "recipe_v5": {"problems": []}, "torch_threads": 1, "T": T, "target": T, "stop_reason": "end",
            "stop_at_end": {"flag": True}, "collapse": {"found": 1, "patience": 10, "tripped": False},
            "digest_start": d0, "digest_end": d1, "train_calls": C.expected_update_rows(C.S, T)}


@pytest.mark.parametrize("arm", C.ARMS)
def test_check_meta_validation(arm):
    fails, ident = RB.check_meta(_good_meta(arm), "validation", C.run_name(arm, VS))
    assert fails == [] and ident["T"] == S + 2 * R8
    for key, val in (("arm_state_end", ["x"]), ("gae_lambda_after_load", 0.5), ("ent_coef_after_load", 0.0),
                     ("status", "failed"), ("stop_reason", "other"), ("train_calls", 0),
                     ("obs_rms_end_equal_start", arm != "frozen_obs_norm"), ("arm_change", "x"), ("seed", 2101)):
        m = _good_meta(arm)
        m[key] = val
        assert RB.check_meta(m, "validation", C.run_name(arm, VS))[0] != [], key
    m = _good_meta(arm)
    m["arm_applied"]["mlp_extractor_class"] = "Other"
    assert RB.check_meta(m, "validation", C.run_name(arm, VS))[0] != []
    m = _good_meta(arm)
    m["digest_end"] = {"params": "changed", "adam": "a0"}
    assert RB.check_meta(m, "validation", C.run_name(arm, VS))[0] != []
    m = _good_meta(arm)
    m.update(S=1_048_576, E=2_490_368)                    # BS's window is not BS2's
    assert any("S, E" in f for f in RB.check_meta(m, "validation", C.run_name(arm, VS))[0])
    for seed in (9101, 9201):                            # BS's and BS2's validation seeds are not PS's
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
    assert RB.phase_meta_problems(m, "critic_phase", "continuation") != []


def test_check_meta_continuation_needs_screen_and_updates(tmp_path, monkeypatch):
    # an empty root (no SCREEN.json), independent of the real <BS>'s state (criteria writer, 2026-10-30:
    # <BS>/SCREEN.json now exists, so the build-time assumption "no SCREEN.json exists in this build" no longer holds)
    monkeypatch.setattr(C, "BS", str(tmp_path))
    m = _good_meta("control", kind="continuation", T=C.E)
    m["seed"] = 2101
    m["name"] = "control_2101"
    fails, _ = RB.check_meta(m, "continuation", "control_2101")
    assert any("SCREEN" in f for f in fails)            # no SCREEN.json under this root
    m["digest_end"] = dict(m["digest_start"])
    assert any("no update" in f for f in RB.check_meta(m, "continuation", "control_2101")[0])
    with pytest.raises(RB.InputError):
        RB.check_meta(m, "continuation", "control_2101_crashed1")


def _phase_keys(arm, k):
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
    base = {"buffer_gae_lambda": C.ARM_GAE_LAMBDA[arm], "model_gae_lambda": C.ARM_GAE_LAMBDA[arm], "adv_scale": 1.0,
            "value_scale": 5.0, "mc_vs_gae_lambda1_recursion": 1e-14, "returns_minus_adv_minus_values": 1e-7,
            "gae095_vs_mc": 0.2, "dmu_actor": 0.3, "buffer_values_vs_critic": 1e-6, "state_problems": [],
            "frozen_update": None}
    lam1 = arm == "gae_lambda_1"
    out = []
    for k in range(n):
        e = dict(base, num_timesteps=S + R8 * (k + 1), adv_err_vs_gae095=0.2 if lam1 else 3e-6,
                 adv_err_vs_mc=3e-6 if lam1 else 0.2, ret_rms_count=100.0 + k, obs_rms_count=200.0 + k)
        blind = arm == "critic_blind_tail3"
        e.update(dv_predict_values=0.0 if blind else 0.05, dv_forward=0.0 if blind else 0.05,
                 dv_evaluate_actions=0.0 if blind else 0.05, dv_mirrored=0.0 if blind else 0.05,
                 buffer_values_vs_raw_value_mlp=0.1 if blind else 1e-6)
        frozen = arm == "frozen_obs_norm"
        e.update(obs_rms_digest="d0" if frozen else f"d{k + 1}", obs_rms_equal_start=frozen,
                 frozen_update={"calls": 2 * (k + 1), "rows": 8 * (k + 1)} if frozen else None)
        e.update(_phase_keys(arm, k))
        out.append(e)
    return out


@pytest.mark.parametrize("arm", C.ARMS)
def test_evidence_check(arm, tmp_path):
    lines = _ev_lines(arm, 3)
    (tmp_path / C.EVIDENCE).write_text("".join(json.dumps(x) + "\n" for x in lines))
    meta = {"arm_evidence": json.loads(json.dumps(H.summarize(lines)))}
    fails, det = RB.evidence_check(tmp_path, meta, arm, 3)
    assert fails == [] and det["n"] == 3
    assert RB.evidence_check(tmp_path, meta, arm, 4)[0] != []
    assert RB.evidence_check(tmp_path, {"arm_evidence": {}}, arm, 3)[0] != []
    for other in C.ARMS:
        if other != arm:
            assert RB.evidence_check(tmp_path, meta, other, 3)[0] != [], (arm, other)


def test_m5_window():
    T = S + 3 * C.EVAL_FREQ
    g = C.EVAL_FREQ
    m5 = {"per_evaluation": {str(S): {"n": 10, "any": 6}, str(S + g): {"n": 10, "any": 5, "tail": 5},
                             str(S + 2 * g): {"n": 10, "any": 0}, str(S + 3 * g): {"n": 10, "any": 7}},
          "any_flagged": [S, S + g, S + 3 * g], "tail_flagged": [S + g], "any_alarms": [S + g], "tail_alarms": [],
          "c6": {"E": T, "passed": True, "failures": ["evaluation 5 episode 3: TAIL input missing (x)"]}}
    w = RB.m5_window(m5, S, T)
    assert w["any_flagged"] == [S + g, S + 3 * g] and w["tail_flagged"] == [S + g]
    assert w["any_alarms"] == [] and w["alarms_straddling_S"] == [S + g]
    assert w["n_c6_counting_failures"] == 0 and sorted(w["per_evaluation"]) == [S + g, S + 2 * g, S + 3 * g]
    with pytest.raises(RB.InputError):
        RB.m5_window({**m5, "c6": {"E": T + g}}, S, T)


def test_ledger_direct(tmp_path):
    T = S + 2 * C.EVAL_FREQ
    rows = []
    for t, cmds, sat in ((S, [0.9] * 10, [("tail_3_pitch_act", 0.5)] * 10),
                         (S + C.EVAL_FREQ, [0.2] * 10, [("r_knee_act", 0.2)] + [("x", 0.0)] * 9),
                         (T, [0.65] * 9 + [""],
                          [("tail_3_pitch_act", 0.3), ("tail_1_yaw_act", 0.11)] + [("x", 0.0)] * 8)):
        for e in range(10):
            rows.append({"evaluation": 0, "timesteps": t, "episode": e, "tail_3_pitch_act_command_mean": cmds[e],
                         "saturated_actuator": sat[e][0], "saturated_actuator_fraction": sat[e][1]})
    path = tmp_path / "eval_episodes.csv"
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    d = RB.ledger_direct(path, S, T)
    assert sorted(d["per_evaluation"]) == [S + C.EVAL_FREQ, T]                  # the row at S is the parent's
    e1, e2 = d["per_evaluation"][S + C.EVAL_FREQ], d["per_evaluation"][T]
    assert e1["n_over"] == 1 and e1["n_tail_over"] == 0 and abs(e1["cmd_l"] - 0.2) < 1e-12
    assert e2["n_over"] == 2 and e2["n_tail_over"] == 2 and e2["n_t3_over"] == 1 and e2["n_cmd"] == 9
    assert d["firsts"]["FOB_tail"] == T and d["firsts"]["cmd_l_ge_0.60"] == T and d["firsts"]["cmd_l_ge_0.90"] is None
    assert d["firsts"]["first_over_any"] == S + C.EVAL_FREQ and d["evaluations_absent"] == []


def test_probe_markers(tmp_path):
    T = S + 12 * R8
    steps = np.asarray([S - R8, S] + [S + R8 * k for k in range(1, 12)], np.int64)
    n = len(steps)
    act = np.zeros((n, 10, 15), np.float32)
    act[:, 0, 14] = np.linspace(0, 0.33, n)
    act[:, 5, 14] = np.linspace(0, 0.11, n)
    labels = np.array(["statue_settled", "statue", "a", "b", "c", "statue_settled", "statue", "a", "b", "c"])
    np.savez(tmp_path / "p.npz", update_timesteps=steps, bank_action_now=act, bank_value_now=np.ones((n, 10)),
             obs_rms_mean=np.zeros((n, 63)), obs_rms_var=np.ones((n, 63)), std_per_actuator=np.ones((n, 15)),
             rollout_mu_mean=np.zeros((n, 15)), bank_labels=labels)
    m = RB.probe_markers(tmp_path / "p.npz", S, T, "x")
    assert m["n_rows"] == 11 and m["steps"][0] == S + R8 and m["steps"][-1] == S + 11 * R8
    ss = (act[2:, 0, 14] + act[2:, 5, 14]) / 2
    assert np.allclose(m["tail3_statue_settled_mean"], ss)
    assert m["trailing8_mean"][6] is None and abs(m["trailing8_mean"][7] - ss[:8].mean()) < 1e-7
    labels[5] = "statue"
    np.savez(tmp_path / "q.npz", update_timesteps=steps, bank_action_now=act, bank_value_now=np.ones((n, 10)),
             obs_rms_mean=np.zeros((n, 63)), obs_rms_var=np.ones((n, 63)), std_per_actuator=np.ones((n, 15)),
             rollout_mu_mean=np.zeros((n, 15)), bank_labels=labels)
    with pytest.raises(RB.InputError):
        RB.probe_markers(tmp_path / "q.npz", S, T, "x")


def test_decided_and_outcome_blind(tmp_path):
    q = [["control", 2101], ["critic_phase", 2101]]
    (tmp_path / "SCREEN.json").write_text(json.dumps({"schema": C.SCREEN_SCHEMA, "intervention": "critic_phase",
                                                      "queue": q}))
    ok, und = RB.decided(tmp_path)
    assert not ok and und == ["control_2101", "critic_phase_2101"]
    for name in ("control_2101",):
        (tmp_path / "runs" / name).mkdir(parents=True)
        (tmp_path / "runs" / name / C.CHECK_JSON).write_text(json.dumps({"passed": True, "mode": "continuation"}))
    assert RB.decided(tmp_path) == (False, ["critic_phase_2101"])
    (tmp_path / "lost.json").write_text(json.dumps({"schema": C.LOST_SCHEMA, "lost": {"critic_phase_2101": {}}}))
    assert RB.decided(tmp_path) == (True, [])
    assert RB.decided(tmp_path / "nowhere")[0] is False


def test_require_under():
    RB.require_under(Path(C.BS) / "reads" / "x", C.BS)
    with pytest.raises(RB.InputError):
        RB.require_under("/tmp/x", C.BS)


# ------------------------------------------------------------------------------------------------ BS2 additions
def test_loop_cache_key_carries_the_reader_sha256():
    """P3 (BS's process lens): the cache key starts with this reader's sha256 (and its modules' digest), so an entry
    written by another reader is never found; an entry whose recorded digests differ is refused, never used."""
    z, p = "a" * 64, "b" * 64
    k = RB.cache_key(z, p, blind=False, onpolicy=True, own=True)
    assert k.startswith("r" + RB.READER_SHA256[:16] + "_d" + RB.DEPS_SHA256[:8] + "_")
    assert RB.READER_SHA256 == C.sha256_file(TOOLS / "read_bs.py")
    assert k == f"r{RB.READER_SHA256[:16]}_d{RB.DEPS_SHA256[:8]}_{'a' * 16}_{'b' * 16}_plain_on_own"
    assert RB.cache_key(z, p, blind=False, onpolicy=True, own=True, reader="c" * 64) != k
    assert RB.cache_key(z, p, blind=False, onpolicy=True, own=True, deps="d" * 64) != k
    assert RB.cache_key(z, p, blind=True, onpolicy=True, own=True) != k
    assert RB.cache_key(z, p, blind=False, onpolicy=False, own=True) != k
    assert RB.cache_key(z, p, blind=False, onpolicy=True, own=False) != k
    # PS: critic_phase's reading has its own form, and the start copy (the load transform applied) another
    kp = RB.cache_key(z, p, blind=False, onpolicy=True, own=True, phase=True)
    ks = RB.cache_key(z, p, blind=False, onpolicy=True, own=True, phase=True, start=True)
    assert kp.endswith("_phase_on_own") and ks.endswith("_phase_S_on_own") and len({k, kp, ks}) == 3
    assert RB.cache_key(z, p, blind=False, onpolicy=True, own=True, start=True) == k     # start: phase reads only
    with pytest.raises(ValueError):
        RB.cache_key(z, p, blind=True, onpolicy=True, own=True, phase=True)
    good = {"reader_sha256": RB.READER_SHA256, "deps_sha256": RB.DEPS_SHA256}
    assert RB.cache_entry_problem(good) is None
    assert RB.cache_entry_problem({**good, "reader_sha256": "0" * 64}) is not None
    assert RB.cache_entry_problem({**good, "deps_sha256": "0" * 64}) is not None
    assert RB.cache_entry_problem({"OWN": {}}) is not None                 # a BS-era entry (no reader digests)


def test_reference_pairs_and_reproduction_targets():
    assert RB.REFERENCE_PAIRS == (2_097_152, 3_145_728)
    assert all(C.S <= s <= C.E for s in RB.REFERENCE_PAIRS) and set(RB.REPRO) == set(RB.REFERENCE_PAIRS)
    for s in RB.REFERENCE_PAIRS:
        w = RB.REPRO[s]
        assert set(w["ew_own"]) == {"G", "c0", "own14"}
        # the verifier's b*_P = 1 - G - c0 (printed to 3 decimals) agrees with the early-warning values
        assert abs((1.0 - w["ew_own"]["G"] - w["ew_own"]["c0"]) - w["vew_b_star_P"]) <= 5e-4
        assert w["wm_b_star_P"] == w["vew_b_star_P"]


# ------------------------------------------------------------------------------------------------ PS additions
def test_with_phase_feeds_each_state_its_own_counter():
    X = np.random.default_rng(0).normal(size=(5, 63))
    K = np.array([0, 100, 199, 200, 1000])
    Y = RB.with_phase(X, K)
    assert np.array_equal(Y[:, :60], X[:, :60]) and not np.shares_memory(X, Y)
    assert [tuple(r) for r in Y[:, 60:63]] == [C.phase_features(int(k)) for k in K]


def _fake_net(value_cols=0.3, slot_var=2.4e-11):
    import torch
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.running_mean_std import RunningMeanStd

    import gymnasium as gym

    torch.manual_seed(0)
    pol = ActorCriticPolicy(gym.spaces.Box(-10, 10, (63,), np.float32), gym.spaces.Box(-1, 1, (15,), np.float32),
                            lambda _: 1e-3, net_arch=[8, 8])
    with torch.no_grad():
        pol.mlp_extractor.value_net[0].weight[:, 60:63] = value_cols
    rms = RunningMeanStd(shape=(63,))
    rms.var = np.ones(63)
    rms.var[60:63] = slot_var
    rms.mean = np.zeros(63)
    net = SimpleNamespace(policy=pol, norm=SimpleNamespace(obs_rms=rms), refreshed=[])
    net.refresh_stats = lambda: net.refreshed.append(True)
    return net


def test_phase_prepare_start_and_later_pairs():
    """PS: at the start copy the arm's load transform (the critic's phase columns zeroed, the slots reseeded, the
    normaliser re-read) and the actor's mask; a later pair must already carry the frozen slots, else refused."""
    net = _fake_net()
    rec = RB.phase_prepare(net, start=True)
    assert rec["start_transform"] and rec["value_phase_cols_max_abs_read"] == pytest.approx(0.3)
    assert rec["value_phase_cols_max_abs_used"] == 0.0 and net.refreshed == [True]
    assert H.phase_slots_frozen(net.norm.obs_rms)
    assert type(net.policy.mlp_extractor).__name__ == "CriticPhaseMlpExtractor"
    net = _fake_net(value_cols=0.05, slot_var=1.0)
    rec = RB.phase_prepare(net, start=False)
    assert rec["value_phase_cols_max_abs_used"] == pytest.approx(0.05) and net.refreshed == []
    with pytest.raises(RB.InputError):
        RB.phase_prepare(_fake_net(), start=False)        # a later pair without the frozen slots
