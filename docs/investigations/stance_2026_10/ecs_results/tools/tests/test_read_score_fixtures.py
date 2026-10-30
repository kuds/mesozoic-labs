"""Shared builders for the lane R tests (read_ecs.py, score_ecs.py).  No test functions here.

Every file a test writes is under <ECS>/tmp/lane_r_tests/ (HARD RULES: write only under <ECS>).  Nothing trains or
simulates: the runs are synthetic records (run_meta.json, policy_probe.npz built from the parent's F2 probe, a
synthetic eval_episodes.csv); the only subprocess is the pinned M5, a ledger reader.
"""
import csv
import importlib.util
import json
import shutil
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent.parent


def _load(name, path=None):
    if name in sys.modules and path is None:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path or TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


E = _load("ecs_common")
R = _load("read_ecs")
S = _load("score_ecs")
GRID = E.EVAL_FREQ
WORK = Path(E.TMP) / "lane_r_tests"


def work_dir(name):
    """A fresh folder under <ECS>/tmp/lane_r_tests/."""
    d = WORK / name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    return d


def load_mutant(name, source_path, old, new):
    """A copy of a tool with one text replacement, loaded as its own module (the real ones stay in sys.modules)."""
    src = Path(source_path).read_text(encoding="utf-8")
    assert src.count(old) == 1, f"mutation anchor not unique in {source_path}: {old!r}"
    d = WORK / "mutants"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.py"
    p.write_text(src.replace(old, new), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------------------------------------ ledgers
LEDGER_COLUMNS = ("evaluation", "timesteps", "episode", "env", "return", "length", "full_horizon", "cause",
                  "screen_v2_clean", "screen_v2_reasons", "saturated_actuator", "saturated_actuator_fraction",
                  "saturated_actuator_positive_share", "saturated_actuator_ties", "actuators_saturated")


def clean_row(step, ep):
    return dict(timesteps=step, episode=ep, env=0, **{"return": 100.0}, length=1000, full_horizon=1,
                cause="truncated", screen_v2_clean=1, screen_v2_reasons="", saturated_actuator="",
                saturated_actuator_fraction=0.0, saturated_actuator_positive_share="", saturated_actuator_ties=0,
                actuators_saturated=0)


def over_row(step, ep, act="tail_3_pitch_act", pos=1.0, frac=0.2):
    """A row over the bar on one actuator (ANY over; TAIL when the actuator is a tail actuator)."""
    r = clean_row(step, ep)
    r.update(screen_v2_clean=0, screen_v2_reasons=f"max_actuator_saturation_fraction:{frac}", saturated_actuator=act,
             saturated_actuator_fraction=frac, saturated_actuator_positive_share=pos, actuators_saturated=1)
    return r


def any_missing_row(step, ep):
    """Screen unmeasured and fraction blank: ANY input missing (C6 counts it)."""
    r = clean_row(step, ep)
    r.update(screen_v2_clean="", saturated_actuator_fraction="")
    return r


def tail_missing_row(step, ep):
    """A measured screen listing the key, fraction over the bar, actuator blank: ANY over, TAIL input missing only."""
    r = clean_row(step, ep)
    r.update(screen_v2_clean=0, screen_v2_reasons="max_actuator_saturation_fraction:0.3", saturated_actuator="",
             saturated_actuator_fraction=0.3, actuators_saturated=1)
    return r


def evaluation(step, n_over=0, act="tail_3_pitch_act", kind=None, n_rows=10):
    """Rows of one evaluation: n_over rows over the bar on act, the rest clean; kind 'any_missing' or 'tail_missing'
    makes episode 9 that kind."""
    rows = []
    for ep in range(n_rows):
        if kind == "any_missing" and ep == 9:
            rows.append(any_missing_row(step, ep))
        elif kind == "tail_missing" and ep == 9:
            rows.append(tail_missing_row(step, ep))
        elif ep < n_over:
            rows.append(over_row(step, ep, act=act))
        else:
            rows.append(clean_row(step, ep))
    return rows


def write_ledger(path, steps, special=None, extra_rows=()):
    """eval_episodes.csv with an evaluation at each step; special: {step: kwargs of evaluation() or None (absent)}."""
    special = special or {}
    rows = []
    for i, s in enumerate(steps):
        if s in special and special[s] is None:
            continue
        for r in evaluation(s, **special.get(s, {})):
            r["evaluation"] = i
            rows.append(r)
    rows.extend(extra_rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in LEDGER_COLUMNS})
    return path


def grid_to(T):
    return [k * GRID for k in range(1, T // GRID + 1)]


# ------------------------------------------------------------------------------------------------ probes and runs
def parent_arrays(start, keys):
    path = E.f2_record(start, "policy_probe.npz")
    return R.load_probe(path, keys)


def write_probe(path, start, S, T, arm="A", mutate=None):
    """A continuation probe: the parent's rows at or before S (seeded) and in (S, T), from the parent's F2 file; arm B's
    ent_coef 0.0 after S; mutate(arrays, window_index) may edit the arrays before they are saved."""
    import numpy as np

    keys = ("update_timesteps", "ent_coef", "learning_rate", *E.M6_COLUMNS)
    keys = tuple(dict.fromkeys(keys))
    a = parent_arrays(start, keys)
    u = a["update_timesteps"].astype(np.int64)
    keep = [i for i, x in enumerate(u.tolist()) if x < T]
    out = {k: np.array(v[keep]) for k, v in a.items()}
    win = [j for j, x in enumerate(out["update_timesteps"].tolist()) if S < x < T]
    if arm == "B":
        out["ent_coef"][win] = np.float32(0.0)
    if mutate is not None:
        out = mutate(out, win) or out
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **out)
    return path


def run_meta(start, arm, seed, T=None, stop="end", mode="continuation", target=None, **over):
    st = E.STARTS[start]
    S_, E_ = E.window(start)
    target = E_ if target is None else target
    T = target if T is None else T
    m = dict(schema=E.RUN_SCHEMA, name=E.run_name(start, arm, seed), mode=mode, status="done", exit=0, start=start,
             arm=arm, seed=seed, S=S_, E=E_, target=target, T=T, stop_reason=stop,
             start_pair=dict(zip="z", zip_sha256=st["zip_sha256"], pkl="p", pkl_sha256=st["pkl_sha256"]),
             parent_records={k: st["records"][k] for k in E.PARENT_RECORDS},
             code=dict(dir=E.CODE, commit=E.COMMIT, tree_digest=E.TREE_DIGEST, files=E.TREE_FILES), stack=dict(E.STACK),
             criteria_sha256=None, arm_ppo=dict(E.ARM_PPO_EDIT[arm]) or "unchanged",
             ent_coef_after_load=E.ARM_ENT_COEF_AFTER_LOAD[arm], recipe_v5={"problems": []},
             torch_threads=1, torch_interop_threads=4,
             digest_start=dict(params="p0", adam="a0", vecnormalize="v0", all="x0", n_updates=3458),
             digest_end=dict(params="p1", adam="a1", vecnormalize="v1", all="x1", n_updates=5368),
             collapse=dict(found=1, consecutive_drops=10 if stop == "collapse" else 0, patience=10,
                           tripped=stop == "collapse"),
             stop_at_end=dict(target=target, flag=stop == "end", step=T if stop == "end" else None), wall_seconds=1.0)
    if mode == "no_update":
        m["digest_end"] = dict(m["digest_start"], vecnormalize="v1", all="x1")  # n_updates unchanged too
        m["train_calls"] = E.expected_update_rows(S_, T)
    m.update(over)
    return m


def make_run(run_dir, start, arm, seed, T=None, stop="end", mode="continuation", target=None, special=None,
             probe_mutate=None, meta_over=None, trace_root=None):
    """A synthetic finished run folder: run_meta.json, 01_stance/policy_probe.npz, eval_episodes.csv, the end zip;
    with trace_root, <trace_root>/m1_summary.json for that end pair."""
    run_dir = Path(run_dir)
    meta = run_meta(start, arm, seed, T=T, stop=stop, mode=mode, target=target)
    meta.update(meta_over or {})
    T = meta["T"]
    S_ = meta["S"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / E.RUN_META).write_text(json.dumps(meta, indent=1), encoding="utf-8")
    write_probe(run_dir / R.PROBE_REL, start, S_, T, arm=arm, mutate=probe_mutate)
    write_ledger(run_dir / R.LEDGER_REL, grid_to(T), special=special)
    z = R.end_zip(run_dir, stop, T)
    z.parent.mkdir(parents=True, exist_ok=True)
    z.write_bytes(f"zip {run_dir.name} {T}".encode())
    if trace_root is not None:
        write_trace(Path(trace_root) / R.TRACE_NAME, z, T)
    return run_dir


def write_trace(path, zip_path, T, **over):
    d = dict(schema="r10m-m1/1", physics_revision=10, tool_sha256=E.PIN_SHA256[E.M1],
             checkpoint=dict(sha256=E.sha256_file(zip_path), num_timesteps=T), n_episodes=20,
             seeds=list(E.M1_SEEDS), full_horizon_episodes=20, cmd_a={"14": 0.7},
             cmd_all_actuators={str(i): 0.01 * i for i in range(15)}, tail_flags=[[3042, 14]], fall_flags=[],
             any_actuator_over_bar_episodes=[3042], max_abs_command_any_step=1.0)
    d.update(over)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(d), encoding="utf-8")
    return path


def passing_check(root, start, arm, seed, T=None, stop="end", criteria_sha256=None):
    """A minimal passing check.json for a continuation (as read_ecs --check writes it), for decided() and the scorer."""
    S_, E_ = E.window(start)
    name = E.run_name(start, arm, seed)
    d = dict(schema=R.CHECK_SCHEMA, version=R.VERSION, tool_sha256=R.tool_sha256(), name=name, mode="continuation",
             passed=True, failures=[], start=start, arm=arm, seed=seed, S=S_, E=E_, T=E_ if T is None else T, end=E_,
             stop_reason=stop, criteria_sha256=criteria_sha256, sources={})
    p = Path(root) / "runs" / name / R.CHECK_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def synthetic_read(root, start, arm, seed, positive, T=None, stop="end", c6_ok=True, first=None):
    """A read.json as read_ecs's full read writes it (R1 values only), bound to the check.json beside it."""
    S_, E_ = E.window(start)
    T = E_ if T is None else T
    name = E.run_name(start, arm, seed)
    csha = E.sha256_file(Path(root) / "runs" / name / R.CHECK_NAME)
    first = (S_ + 4 * GRID if positive else None) if first is None else first
    r1 = dict(S=S_, T=T, positive=positive, first_any_flag=first, n_any_flagged=3 if positive else 0,
              any_flagged=[first] if positive else [], any_alarms=[], any_alarms_straddling_S=[], tail_flagged=[],
              tail_alarms=[], tail_alarms_straddling_S=[], c6_passed_m5=c6_ok, c6_ok=c6_ok,
              c6_counting_failures=[] if c6_ok else [dict(failure=f"evaluation {T} absent", category="grid", step=T,
                                                          where="window")],
              c6_tail_missing=[], m5_sha256=E.PIN_SHA256[E.M5], m5_common_sha256=E.PIN_SHA256[E.M5_COMMON])
    d = dict(schema=R.READ_SCHEMA, tool_sha256=R.tool_sha256(), name=name, start=start, arm=arm, seed=seed, S=S_, T=T,
             stop_reason=stop, check_sha256=csha, r1=r1, r2={}, r3=dict(status="missing"), returns={})
    p = R.read_dir(root, name) / R.READ_NAME
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d), encoding="utf-8")
    return p


def outcomes_tree(root, table, special=None, criteria_sha256=None):
    """A complete synthetic screen: table {start: (a, b)} POSITIVE counts of 6 per arm (the first a / b seeds POSITIVE);
    special {name: ("lost", cause) | ("censored", T) | ("stopped_pos", T) | ("c6",)}."""
    special = special or {}
    lost = {}
    for start, arm, seed in E.QUEUE:
        name = E.run_name(start, arm, seed)
        j = E.SEEDS[(start, arm)].index(seed)
        a, b = table[start]
        positive = j < (a if arm == "A" else b)
        sp = special.get(name)
        if sp and sp[0] == "lost":
            lost[name] = {"cause": "crashes", "detail": sp[1], "utc": "2026-10-28T00:00:00Z", "by": "run_waves_ecs.sh"}
            continue
        if sp and sp[0] in ("censored", "stopped_pos"):
            passing_check(root, start, arm, seed, T=sp[1], stop="collapse", criteria_sha256=criteria_sha256)
            synthetic_read(root, start, arm, seed, positive=sp[0] == "stopped_pos", T=sp[1], stop="collapse")
            continue
        passing_check(root, start, arm, seed, criteria_sha256=criteria_sha256)
        synthetic_read(root, start, arm, seed, positive=positive, c6_ok=not (sp and sp[0] == "c6"))
    if lost:
        (Path(root) / "lost.json").write_text(json.dumps({"schema": E.LOST_SCHEMA, "lost": lost}), encoding="utf-8")
    return root


def sealed_code(root):
    """A small code tree under <root>/code sealed by M1's own r10m_common (its tree digest and file count), as M1's
    build-copy seals the real copy.  Returns (digest, files); a test sets ecs_common TREE_DIGEST/TREE_FILES to them."""
    common = _load("lane_r_r10m_common", E.M1_COMMON)
    code = Path(root) / "code"
    (code / "environments").mkdir(parents=True, exist_ok=True)
    (code / "environments" / "__init__.py").write_text("", encoding="utf-8")
    (code / "environments" / "x.py").write_text("x = 1\n", encoding="utf-8")
    digest, n = common.tree_digest(code)
    (code / E.SEAL_NAME).write_text(json.dumps({"commit": E.COMMIT, "tree_digest": digest, "files": n}),
                                    encoding="utf-8")
    return digest, n
