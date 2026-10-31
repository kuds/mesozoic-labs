"""Unit tests of run_bs.py's pure parts and of bs_common.py: no learn(), no simulation.

    cd /tmp && OMP_NUM_THREADS=1 nice -n 10 PY -I -B -m pytest -q -p no:cacheprovider <BS>/tools/tests/test_run_bs.py
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.dont_write_bytecode = True
TOOLS = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, str(TOOLS / f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


R = _load("run_bs")
C = R.C
V0, V4 = C.VALIDATION_SEEDS[0], C.VALIDATION_SEEDS[-1]     # PS: 9401, 9441 (BS2's tests used 9201 / 9241)


# ------------------------------------------------------------------------------------------------ bs_common
def test_window_and_grid():
    assert (C.S, C.E, C.LENGTH) == (2_097_152, 3_538_944, 1_441_792)
    assert C.E - C.S <= C.MAX_LENGTH == 1_500_000
    assert C.N_EVALS == 11 and C.LENGTH % C.EVAL_FREQ == 0 and C.LENGTH % C.ROLLOUT == 0
    assert C.S % C.EVAL_FREQ == 0 and C.E % C.EVAL_FREQ == 0 and C.E // C.EVAL_FREQ == 27   # E on the 131,072 grid
    assert C.S % C.SAVE_FREQ == 0 and C.S % C.ROLLOUT == 0
    assert C.LENGTH // C.ROLLOUT == 176 and C.expected_update_rows(C.S, C.E) == 175
    assert C.SAVED_STEPS == (2_359_296, 2_621_440, 2_883_584, 3_145_728, 3_407_872)
    assert C.E % C.SAVE_FREQ != 0                       # no periodic pair at E: the final pair is the end pair
    assert C.VALIDATE_UNTIL == C.S + 262_144 == 2_359_296
    assert C.WAVE_SLOTS == 4 and C.WAVE_NICE == 10 and C.MAX_CONTINUATIONS == 8
    assert C.DISK_FLOOR_BYTES == 1_610_612_736 and C.LAUNCH_DISK_BYTES == 1_610_612_736 + 8 * 50 * 1024**2
    assert C.BS == "/home/user/work/phase_screen" and C.CODE == C.BS1 + "/code" and C.BS1 != C.BS
    assert C.BS1.endswith("/r10round/bias_screen") and C.BS2_DIR.endswith("/r10round/bias_screen2")


def test_start_pair_pins():
    st = C.START
    assert st["S"] == C.S and st["probe_row_at_S"] == C.S // C.ROLLOUT - 1 == 255
    assert st["zip_rel"] == "01_stance/models/" + C.periodic_names(C.S)[0]
    assert st["pkl_rel"] == "01_stance/models/" + C.periodic_names(C.S)[1]
    assert st["zip_sha256"] == "83b6c88edb933005adcb2089bdac67654f4e77809ca5506f4e47364995ef1b61"
    assert st["pkl_sha256"] == "991d434f385187e87301755ddf9a12fe2b3adb65cf186e7e32b175b3574bed90"
    assert set(st["records"]) == {*C.PARENT_RECORDS, "stage_config.json"}


def test_validation_seeds_are_pss():
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
        C.PHASE_SLOTS, C.SETTLE_STEPS, C.PHASE_GAMMA, C.OBS_DIM, C.PHASE_FEATURES)


def test_run_names():
    for arm in C.ARMS:
        n = C.run_name(arm, 2101)
        assert C.parse_run_name(n) == {"arm": arm, "seed": 2101, "suffix": None, "k": None}
        assert C.parse_run_name(n + "_crashed2") == {"arm": arm, "seed": 2101, "suffix": "crashed", "k": 2}
    assert C.parse_run_name("critic_phase_2301")["arm"] == "critic_phase"
    assert C.parse_run_name("s42_A_1101") is None and C.parse_run_name("critic_2101") is None
    assert C.parse_run_name("critic_blind_tail3_2101") is None and C.parse_run_name("gae_lambda_1_2101") is None


def _screen(inter="critic_phase", queue=None):
    q = queue if queue is not None else [["control", 2101], [inter, 2101], ["control", 2111], [inter, 2111]]
    return {"schema": C.SCREEN_SCHEMA, "intervention": inter, "queue": q}


def test_screen_valid_and_invalid():
    for inter in C.INTERVENTIONS:
        assert C.screen_problems(_screen(inter)) == []
    assert C.queue_of(_screen())[1] == ("critic_phase", 2101)
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
        "bool seed": _screen(queue=[["control", True], ["critic_phase", 2]]),
        "schema": {**_screen(), "schema": "x"},
        "empty": _screen(queue=[]),
    }
    for why, d in bad.items():
        assert C.screen_problems(d) != [], why
    assert C.screen_problems([]) != []


def test_arm_tables_complete():
    assert set(C.ARM_PPO_EDIT) == set(C.ARMS) == set(C.ARM_GAE_LAMBDA) == set(C.ARM_CHANGE)
    assert C.ARMS == ("control", "critic_phase") and C.INTERVENTIONS == C.ARMS[1:]
    assert all(not C.ARM_PPO_EDIT[a] for a in C.ARMS) and set(C.ARM_GAE_LAMBDA.values()) == {0.95}
    # bs_hooks keeps BS2's three arms (unreachable in PS: preflight, SCREEN and CHECK use bs_common.ARMS)
    assert set(C.ARMS) <= set(R.H.ARMS) and all(R.H.GAE_LAMBDA[a] == C.ARM_GAE_LAMBDA[a] for a in C.ARMS)
    assert R.H.OBS_TAIL3 == C.OBS_TAIL3


# ------------------------------------------------------------------------------------------------ arm_config, V5
def test_arm_config():
    cfg = {"ppo_kwargs": {"gae_lambda": 0.95, "ent_coef": 0.005}, "x": [1]}
    for arm in C.ARMS:
        out = R.arm_config(cfg, arm)
        assert out["ppo_kwargs"]["gae_lambda"] == C.ARM_GAE_LAMBDA[arm]
        assert out["ppo_kwargs"]["ent_coef"] == 0.005 and out["x"] == [1] and out["x"] is not cfg["x"]
    assert cfg["ppo_kwargs"]["gae_lambda"] == 0.95
    with pytest.raises(ValueError):
        R.arm_config(cfg, "B")
    with pytest.raises(ValueError):
        R.arm_config({"ppo_kwargs": {}}, "gae_lambda_1")


def _parent_sc():
    return json.loads(Path(C.f2_record("stage_config.json")).read_text())


def _written(parent, arm, seed=V0, hp_digest=None):
    w = copy.deepcopy(parent)
    w["hyperparameters"].update(C.ARM_PPO_EDIT[arm])
    w["run"].update(seed=seed, timesteps=C.TOTAL_TIMESTEPS - C.S, label=f"BS {arm} {seed}", duration_seconds=0.0,
                    hyperparameters_sha256=hp_digest or C.HYPERPARAMETERS_SHA256_RUN, load_path="/x/z.zip",
                    load_mode="resume_same_stage", parent_checkpoint_sha256="sha256:" + C.START["zip_sha256"],
                    parent_task_sha256=parent["task_fingerprint"]["task_sha256"])
    w["gpu"] = None
    return w


@pytest.mark.parametrize("arm", C.ARMS)
def test_recipe_problems(arm):
    parent = _parent_sc()
    hp = C.HYPERPARAMETERS_SHA256_RUN                  # PS: no arm edits [ppo]
    kw = dict(arm=arm, seed=V0, label=f"BS {arm} {V0}", load_path="/x/z.zip", zip_sha256=C.START["zip_sha256"],
              hp_sha256_expected=hp)
    assert R.recipe_problems(_written(parent, arm, hp_digest=hp), parent, **kw) == []
    w = _written(parent, arm, hp_digest=hp)
    w["hyperparameters"]["clip_range"] = 0.3
    assert R.recipe_problems(w, parent, **kw) != []
    w = _written(parent, arm, hp_digest=hp)
    w["hyperparameters"]["gae_lambda"] = 1.0
    assert R.recipe_problems(w, parent, **kw) != []
    w = _written(parent, arm, hp_digest=hp)
    w["reward_weights"] = {}
    assert R.recipe_problems(w, parent, **kw) != []
    bad_hp = "sha256:" + "2" * 64
    assert R.recipe_problems(_written(parent, arm, hp_digest=bad_hp), parent, **{**kw, "hp_sha256_expected": bad_hp})


# ------------------------------------------------------------------------------------------------ load checks
class _Mirror:
    pass


def _fake_loaded(arm, **over):
    env = SimpleNamespace(training=True, norm_reward=True)
    lam = C.ARM_GAE_LAMBDA[arm]
    m = _Mirror()
    m.__dict__.update(num_timesteps=C.S, _total_timesteps=C.TOTAL_TIMESTEPS, device="cpu", ent_coef=0.005,
                      gae_lambda=lam, rollout_buffer=SimpleNamespace(gae_lambda=lam))
    m.__dict__.update(over)
    m.get_vec_normalize_env = lambda: env
    return m, env


@pytest.mark.parametrize("arm", C.ARMS)
def test_loaded_problems(arm):
    m, env = _fake_loaded(arm)
    assert R.loaded_problems(m, mirror_cls=_Mirror, arm=arm, train_env=env, threads=1) == []
    assert R.loaded_problems(m, mirror_cls=_Mirror, arm=arm, train_env=env, threads=2)
    wrong = 1.0
    for over in ({"gae_lambda": wrong}, {"rollout_buffer": SimpleNamespace(gae_lambda=wrong)}, {"num_timesteps": 0},
                 {"ent_coef": 0.0}, {"device": "cuda"}):
        m, env = _fake_loaded(arm, **over)
        assert R.loaded_problems(m, mirror_cls=_Mirror, arm=arm, train_env=env, threads=1), over
    m, env = _fake_loaded(arm)
    env.training = False
    assert R.loaded_problems(m, mirror_cls=_Mirror, arm=arm, train_env=env, threads=1)


def test_post_run_status_and_stop_reason():
    assert R.post_run_status([], [], [], []) == ("done", 0)
    assert R.post_run_status([], [], [], ["hook gone"]) == ("failed", C.EXIT_POST_CHECK)
    assert R.post_run_status(["x"], [], []) == ("failed", C.EXIT_POST_CHECK)
    assert R.stop_reason(flag=True, T=C.E, target=C.E, collapse_tripped=False) == "end"
    assert R.stop_reason(flag=False, T=C.S + 8192 * 20, target=C.E, collapse_tripped=True) == "collapse"
    assert R.stop_reason(flag=False, T=C.E, target=C.E, collapse_tripped=False) == "other"


class _BaseCB:
    def __init__(self, verbose=0):
        self.locals = {}
        self.model = SimpleNamespace(num_timesteps=0)


def test_stop_at_end_order_and_stop():
    calls = []
    cls = R.make_stop_at_end_class(_BaseCB)
    cb = cls(C.S + 2 * C.ROLLOUT, heartbeat=lambda n: calls.append(("hb", n)),
             evidence=lambda loc: calls.append(("ev", loc.get("k"))))
    cb.locals = {"k": 7}
    cb.model.num_timesteps = C.S + C.ROLLOUT
    cb._on_rollout_end()
    assert calls == [("ev", 7), ("hb", C.S + C.ROLLOUT)]
    assert cb._on_step() is True
    cb.model.num_timesteps = C.S + 2 * C.ROLLOUT
    assert cb._on_step() is False and cb.flag and cb.step == C.S + 2 * C.ROLLOUT
    cb.model.num_timesteps += 4
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


def test_no_update_class_swap():
    class M:
        def train(self):
            raise AssertionError("an optimizer step")

    m, holder = M(), {}
    R.install_no_update(m, holder)
    m.train()
    m.train()
    assert holder["train_calls"] == 2 and type(m).__mro__[1] is M
    holder = {"digest_start": 1, "model_class": type(m), "arm_applied": {}, "no_update_class": type(m)}
    assert R.wrapper_problems(holder, mode="no_update", update_classes=(M,)) == []
    assert R.wrapper_problems({"digest_start": 1, "model_class": M, "arm_applied": {}}, mode="no_update",
                              update_classes=(M,))
    assert R.wrapper_problems({"digest_start": 1, "model_class": M}, mode="continuation")


# ------------------------------------------------------------------------------------------------ preflight
def _ns(**kw):
    base = dict(arm="control", seed=V0, save_loaded_only=False, no_update=False, until=None, out=None,
                preflight_only=True)
    base.update(kw)
    return SimpleNamespace(**base)


def test_preflight_validation_modes():
    assert R.preflight(_ns(save_loaded_only=True, seed=V4, out=str(Path(C.VALIDATE) / "never_made"))) == []
    p = R.preflight(_ns(no_update=True, seed=V4, until=C.S + 100, out=str(Path(C.VALIDATE) / "never_made")))
    assert any("--until" in x for x in p)
    p = R.preflight(_ns(no_update=True, seed=2101, until=C.S + C.ROLLOUT, out=str(Path(C.VALIDATE) / "nm")))
    assert any("validation seed" in x for x in p)
    p = R.preflight(_ns(no_update=True, seed=V4, until=C.S + C.ROLLOUT, out="/tmp/elsewhere"))
    assert any("not under" in x for x in p)
    assert any("--no-update needs --until" in x for x in R.preflight(_ns(no_update=True, seed=V4)))
    p = R.preflight(_ns(no_update=True, seed=9101, until=C.S + C.ROLLOUT, out=str(Path(C.VALIDATE) / "nm")))
    assert any("validation seed" in x for x in p)          # BS's validation seeds are not PS's
    p = R.preflight(_ns(no_update=True, seed=9201, until=C.S + C.ROLLOUT, out=str(Path(C.VALIDATE) / "nm")))
    assert any("validation seed" in x for x in p)          # nor BS2's
    assert any("arm" in x for x in R.preflight(_ns(arm="B", save_loaded_only=True)))
    for bs2_arm in ("critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm"):   # BS2's arms are not PS's
        assert any("arm" in x for x in R.preflight(_ns(arm=bs2_arm, save_loaded_only=True, seed=V4)))
    assert R.preflight(_ns(arm="critic_phase", save_loaded_only=True, seed=V4,
                           out=str(Path(C.VALIDATE) / "never_made"))) == []


def test_preflight_continuation_refused_without_approval(tmp_path, monkeypatch):
    # an empty root (no APPROVED, no SCREEN.json), independent of the real <BS>'s state (criteria writer, 2026-10-30)
    monkeypatch.setattr(R, "bs_root", lambda: tmp_path)
    p = R.preflight(_ns(seed=2101))
    assert any(x.startswith("APPROVED") for x in p) and any(x.startswith("SCREEN") for x in p)


def test_approval_and_screen_against_a_temp_root(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "bs_root", lambda: tmp_path)
    (tmp_path / "criteria.txt").write_text("criteria\n")
    (tmp_path / "SCREEN.json").write_text(json.dumps(_screen()))

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()
    approved = {"criteria_sha256": sha(tmp_path / "criteria.txt"), "screen_sha256": sha(tmp_path / "SCREEN.json"),
                "code_tree_digest": C.TREE_DIGEST, "tool_sha256": {n: sha(TOOLS / n) for n in C.TOOL_FILES}}
    (tmp_path / "APPROVED").write_text(json.dumps(approved))
    assert R.approval_problems() == []
    assert R.screen_entry_problems("critic_phase", 2111) == []
    assert R.screen_entry_problems("critic_phase", 2121) != []
    assert R.screen_entry_problems("critic_blind_tail3", 2101) != []
    (tmp_path / "SCREEN.json").write_text(json.dumps(_screen(queue=[["control", 2101], ["critic_phase", 2131]])))
    assert any("screen_sha256" in x for x in R.approval_problems())
    approved["tool_sha256"]["run_bs.py"] = "0" * 64
    (tmp_path / "APPROVED").write_text(json.dumps(approved))
    assert any("run_bs.py" in x for x in R.approval_problems())


def test_inputs_and_seal_pass():
    assert R.input_problems() == []
    assert R.seal_problems() == []


def test_environment_keeps_git_inside_the_copy_s_parent(monkeypatch, tmp_path):
    # setup_environment only sets variables (the folder it makes is under the given root)
    monkeypatch.setattr(R, "bs_root", lambda: tmp_path)
    saved = dict(R.os.environ)
    try:
        env = R.setup_environment("x")
    finally:
        R.os.environ.clear()
        R.os.environ.update(saved)
    ceil = env["GIT_CEILING_DIRECTORIES"].split(R.os.pathsep)
    assert ceil == [str(tmp_path), R.os.path.dirname(R.os.path.realpath(C.CODE))]
    assert env["TMPDIR"] == str(tmp_path / "tmp" / "x")
