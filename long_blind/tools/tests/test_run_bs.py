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
V0, V4 = C.VALIDATION_SEEDS[0], C.VALIDATION_SEEDS[-1]     # LB: 9301, 9341 (BS 9101 / 9141, BS2 9201 / 9241)


# ------------------------------------------------------------------------------------------------ bs_common
def test_window_and_grid():
    assert (C.S, C.E, C.LENGTH) == (2_097_152, 12_001_280, 9_904_128)
    assert C.E == 1_465 * C.ROLLOUT and C.E - C.S == C.MAX_LENGTH == C.LENGTH     # the parent's own end
    assert C.N_EVALS == 75 and C.LENGTH % C.ROLLOUT == 0 and C.S % C.EVAL_FREQ == 0
    assert C.E % C.EVAL_FREQ == 73_728 and C.S + C.N_EVALS * C.EVAL_FREQ == 11_927_552   # E off the evaluation grid
    assert C.S % C.SAVE_FREQ == 0 and C.S % C.ROLLOUT == 0
    assert C.LENGTH // C.ROLLOUT == 1_209 and C.expected_update_rows(C.S, C.E) == 1_208
    assert len(C.SAVED_STEPS) == 37 and C.SAVED_STEPS[0] == 2_359_296 and C.SAVED_STEPS[-1] == 11_796_480
    assert all(b - a == C.SAVE_FREQ for a, b in zip(C.SAVED_STEPS, C.SAVED_STEPS[1:], strict=False))
    assert C.E % C.SAVE_FREQ != 0                       # no periodic pair at E: the final pair is the end pair
    assert C.VALIDATE_UNTIL == C.S + 262_144 == 2_359_296
    assert C.WAVE_SLOTS == 4 and C.WAVE_NICE == 10 and C.MAX_CONTINUATIONS == 4
    assert C.RUN_BYTES_EST == 200 * 1024**2
    assert C.DISK_FLOOR_BYTES == 1_610_612_736 and C.LAUNCH_DISK_BYTES == 1_610_612_736 + 4 * 200 * 1024**2
    assert C.BS.endswith("/r10round/long_blind") and C.CODE == C.BS1 + "/code" and C.BS1 != C.BS
    assert C.HOP_WATCH_AFTER == 7_000_000 and C.HOP_WATCH_FIRST_EVAL == 54 * C.EVAL_FREQ == 7_077_888
    assert C.RESUME_GRID == C.SAVE_FREQ and C.LOOP_READ_EVERY % C.SAVE_FREQ == 0 and C.EVAL_EPISODES == 10
    assert "bs_resume.py" in C.TOOL_FILES and len(C.TOOL_FILES) == 6 and C.SCORER_FILE == "score_bs.py"


def test_start_pair_pins():
    st = C.START
    assert st["S"] == C.S and st["probe_row_at_S"] == C.S // C.ROLLOUT - 1 == 255
    assert st["zip_rel"] == "01_stance/models/" + C.periodic_names(C.S)[0]
    assert st["pkl_rel"] == "01_stance/models/" + C.periodic_names(C.S)[1]
    assert st["zip_sha256"] == "83b6c88edb933005adcb2089bdac67654f4e77809ca5506f4e47364995ef1b61"
    assert st["pkl_sha256"] == "991d434f385187e87301755ddf9a12fe2b3adb65cf186e7e32b175b3574bed90"
    assert set(st["records"]) == {*C.PARENT_RECORDS, "stage_config.json"}


def test_validation_seeds_are_lbs():
    assert C.VALIDATION_SEEDS == (9301, 9311, 9321, 9331, 9341)
    assert all(s not in C.VALIDATION_SEEDS for s in (2101, 2111, 2121, 2131, 9201, 9101))


def test_run_names():
    for arm in C.ARMS:
        n = C.run_name(arm, 2101)
        assert C.parse_run_name(n) == {"arm": arm, "seed": 2101, "suffix": None, "k": None}
        assert C.parse_run_name(n + "_crashed2") == {"arm": arm, "seed": 2101, "suffix": "crashed", "k": 2}
    assert C.parse_run_name("gae_lambda_1_2301")["arm"] == "gae_lambda_1"
    assert C.parse_run_name("s42_A_1101") is None and C.parse_run_name("critic_2101") is None


def _screen(inter="critic_blind_tail3", queue=None):
    q = queue if queue is not None else [["control", 2101], [inter, 2101], ["control", 2111], [inter, 2111]]
    return {"schema": C.SCREEN_SCHEMA, "intervention": inter, "queue": q}


def test_screen_valid_and_invalid():
    for inter in C.INTERVENTIONS:
        assert C.screen_problems(_screen(inter)) == []
    assert C.queue_of(_screen())[1] == ("critic_blind_tail3", 2101)
    four = [["control", 2000]] + [["critic_blind_tail3", 3000 + i] for i in range(3)]       # AUTH2: 4 runs
    assert C.screen_problems(_screen("critic_blind_tail3", four)) == []
    bad = {
        "five": _screen("critic_blind_tail3", four + [["control", 2999]]),
        "unknown intervention": _screen("ent_coef_0"),
        "two interventions": _screen(queue=[["control", 1], ["critic_blind_tail3", 2], ["gae_lambda_1", 3]]),
        "control only": _screen(queue=[["control", 1], ["control", 2]]),
        "validation seed": _screen(queue=[["control", V0], ["critic_blind_tail3", 2]]),
        "repeat": _screen(queue=[["control", 5], ["control", 5], ["critic_blind_tail3", 2]]),
        "bool seed": _screen(queue=[["control", True], ["critic_blind_tail3", 2]]),
        "schema": {**_screen(), "schema": "x"},
        "empty": _screen(queue=[]),
    }
    for why, d in bad.items():
        assert C.screen_problems(d) != [], why
    assert C.screen_problems([]) != []


def test_arm_tables_complete():
    assert set(C.ARM_PPO_EDIT) == set(C.ARMS) == set(C.ARM_GAE_LAMBDA) == set(C.ARM_CHANGE)
    assert C.ARMS[0] == "control" and C.INTERVENTIONS == C.ARMS[1:]
    assert C.ARM_PPO_EDIT["gae_lambda_1"] == {"gae_lambda": 1.0}
    assert all(not C.ARM_PPO_EDIT[a] for a in C.ARMS if a != "gae_lambda_1")
    assert R.H.GAE_LAMBDA == C.ARM_GAE_LAMBDA and R.H.ARMS == C.ARMS and R.H.OBS_TAIL3 == C.OBS_TAIL3


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
    hp = "sha256:" + "1" * 64 if arm == "gae_lambda_1" else C.HYPERPARAMETERS_SHA256_RUN
    kw = dict(arm=arm, seed=V0, label=f"BS {arm} {V0}", load_path="/x/z.zip", zip_sha256=C.START["zip_sha256"],
              hp_sha256_expected=hp)
    assert R.recipe_problems(_written(parent, arm, hp_digest=hp), parent, **kw) == []
    w = _written(parent, arm, hp_digest=hp)
    w["hyperparameters"]["clip_range"] = 0.3
    assert R.recipe_problems(w, parent, **kw) != []
    w = _written(parent, arm, hp_digest=hp)
    w["hyperparameters"]["gae_lambda"] = 1.0 if arm != "gae_lambda_1" else 0.95
    assert R.recipe_problems(w, parent, **kw) != []
    w = _written(parent, arm, hp_digest=hp)
    w["reward_weights"] = {}
    assert R.recipe_problems(w, parent, **kw) != []
    bad_hp = C.HYPERPARAMETERS_SHA256_RUN if arm == "gae_lambda_1" else "sha256:" + "2" * 64
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
    wrong = 0.95 if arm == "gae_lambda_1" else 1.0
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
    t = C.HOP_WATCH_FIRST_EVAL
    assert R.stop_reason(flag=False, T=t, target=C.E, collapse_tripped=False, hop_tripped=True) == "hop_watch"
    assert R.stop_reason(flag=False, T=t, target=C.E, collapse_tripped=True, hop_tripped=True) == "collapse"
    assert R.stop_reason(flag=True, T=C.E, target=C.E, collapse_tripped=False, hop_tripped=True) == "end"


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
    assert any("validation seed" in x for x in p)          # BS's validation seeds are not LB's
    assert any("arm" in x for x in R.preflight(_ns(arm="B", save_loaded_only=True)))


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
    assert R.screen_entry_problems("critic_blind_tail3", 2111) == []
    assert R.screen_entry_problems("critic_blind_tail3", 2121) != []
    assert R.screen_entry_problems("gae_lambda_1", 2101) != []
    (tmp_path / "SCREEN.json").write_text(json.dumps(_screen(queue=[["control", 2101], ["gae_lambda_1", 2101]])))
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


# ------------------------------------------------------------------------------------------------ LB additions
def test_recipe_and_load_checks_at_a_resume_point():
    """A resume at P loads P's pair: run.timesteps = 12,000,000 - P and num_timesteps = P (S's would be refused)."""
    parent, P = _parent_sc(), C.SAVED_STEPS[3]
    w = _written(parent, "critic_blind_tail3")
    w["run"]["timesteps"] = C.TOTAL_TIMESTEPS - P
    w["run"]["parent_checkpoint_sha256"] = "sha256:" + "a" * 64
    kw = dict(arm="critic_blind_tail3", seed=V0, label=f"BS critic_blind_tail3 {V0}", load_path="/x/z.zip",
              zip_sha256="a" * 64, hp_sha256_expected=C.HYPERPARAMETERS_SHA256_RUN)
    assert R.recipe_problems(w, parent, **kw, load_steps=P) == []
    assert any("timesteps" in x for x in R.recipe_problems(w, parent, **kw))           # the default load step is S
    m, env = _fake_loaded("critic_blind_tail3", num_timesteps=P)
    assert R.loaded_problems(m, mirror_cls=_Mirror, arm="critic_blind_tail3", train_env=env, threads=1,
                             expected_steps=P) == []
    assert R.loaded_problems(m, mirror_cls=_Mirror, arm="critic_blind_tail3", train_env=env, threads=1)


def test_hop_state():
    class HopRegimeWatchCallback:
        _tripped, stop, after_timesteps, max_unsupported_duty = True, True, 7_000_000, 0.05

    class Other:
        pass
    st = R.hop_state([Other(), HopRegimeWatchCallback()])
    assert st == {"found": 1, "tripped": True, "stop": True, "after_timesteps": 7_000_000,
                  "max_unsupported_duty": 0.05}
    assert R.hop_state([Other()])["found"] == 0 and R.hop_state([Other()])["tripped"] is False


def _crashed(tmp_path, arm="critic_blind_tail3", seed=V0, k=1, **meta):
    d = tmp_path / f"{arm}_{seed}_crashed{k}"
    d.mkdir()
    m = {"arm": arm, "seed": seed, "mode": "no_update", "S": C.S, "E": C.E, "target": C.S + 4 * C.SAVE_FREQ,
         "status": "running"}
    m.update(meta)
    (d / C.RUN_META).write_text(json.dumps(m))
    return d


def test_resume_from_problems(tmp_path):
    ns = _ns(arm="critic_blind_tail3", no_update=True, until=C.S + 4 * C.SAVE_FREQ, out=str(tmp_path))
    d = _crashed(tmp_path)
    assert R.resume_from_problems(SimpleNamespace(**{**vars(ns), "resume_from": str(d)})) == []
    assert R.resume_from_problems(SimpleNamespace(**{**vars(ns), "resume_from": None})) == []
    cases = (({"status": "done"}, "nothing to resume"), ({"seed": 1}, "seed"), ({"E": 3_538_944}, "E"),
             ({"target": C.E}, "target"), ({"mode": "continuation"}, "mode"))
    for i, (over, why) in enumerate(cases):
        bad = _crashed(tmp_path, k=2 + i, **over)
        p = R.resume_from_problems(SimpleNamespace(**{**vars(ns), "resume_from": str(bad)}))
        assert any(why in x for x in p), (over, p)
    other = _crashed(tmp_path, arm="control")
    assert R.resume_from_problems(SimpleNamespace(**{**vars(ns), "resume_from": str(other)}))
    plain = tmp_path / "critic_blind_tail3_9301"
    plain.mkdir()
    assert any("crashed" in x for x in R.resume_from_problems(SimpleNamespace(**{**vars(ns),
                                                                                  "resume_from": str(plain)})))
    elsewhere = tmp_path / "sub"
    elsewhere.mkdir()
    d2 = _crashed(elsewhere)
    assert any("run root" in x for x in R.resume_from_problems(SimpleNamespace(**{**vars(ns),
                                                                                  "resume_from": str(d2)})))
    assert R.resume_from_problems(SimpleNamespace(**{**vars(ns), "save_loaded_only": True, "no_update": False,
                                                     "resume_from": str(d)}))


def test_attempt_summary_carries_no_outcome(tmp_path):
    s = R.attempt_summary(tmp_path / "control_9301_crashed1",
                          {"attempt": 1, "status": "crashed", "exit": 1, "heartbeat": {"rollout_end": 2_400_256},
                           "last_step": 2_400_260, "utc_start": "a", "utc_end": "b", "error": "x" * 999,
                           "arm_evidence": {"n": 5}, "stop_reason": None})
    assert s["last_step"] == 2_400_260 and s["folder"] == "control_9301_crashed1" and len(s["error"]) == 300
    assert set(s) == {"attempt", "folder", "status", "exit", "resumed_from", "load_step", "last_step", "utc_start",
                      "utc_end", "error"}
