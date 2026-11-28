"""E5 pitch-0 branch (sim/results/candidates_2026-11-28/E5/pitch0/E5_PITCH0_PLAN.md), runner. SIMULATION only.
Generated from E5's run_e5.py (sha256 ad49597e...) by make_runner.py's literal replacements; everything else is E5's
runner unchanged, including its readings R1-R6. Changed after a result: written after E5's pitched held-out results
and review finding E5-R1 (prereg section 3 item 9's pitch-0 alternative, run as a contingency; the lead's ruling is
open).
Differences from run_e5.py:
  - stage 1 is plan.json's 33 points with body_pitch_deg set to 0.0 (ids H<size>-p0-s1-NN); every episode refuses a
    nonzero pitch;
  - stage 3 (only if no stage-2 setting meets the rule) drops e5_neighbours' two body-pitch neighbours (ids
    H<size>-p0-s3-NN);
  - held-out: the chosen setting on 301-310, 401-410, 501-510 and 601-610; the frozen controller (supporting) on
    501-510 and 601-610 only; SUCCEEDS only if all four sets pass;
  - files: scratch candidates/E5_pitch0/, results sim/results/candidates_2026-11-28/E5/pitch0/ (e5p0_*), the plan's
    sha256 checked before tune and heldout and recorded in every result file; tuning rows carry seeds and episode
    wall time; per-seed tables carry the 2-8 s speed and hind knee (context, review E5-R2).
Usage: python run_e5_pitch0.py {tune,heldout,summarise} --size {0.60,0.50} [--workers 2]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

BUILD = Path("/home/user/mesozoic-labs/hardware/triceratops_build")
sys.path.insert(0, str(BUILD / "sim"))
import numpy as np  # noqa: E402

import candidate_checks as C  # noqa: E402

RUNNER = Path(__file__).resolve()
SCR = C.CAND_SCRATCH / "E5_pitch0"
EPI = SCR / "episodes"
MODELS = C.CAND_SCRATCH / "E5" / "models"     # E5's 50 cm model, reused (sha256 checked before every use)
OUT = C.RESULTS / "E5" / "pitch0"
PLAN = OUT / "E5_PITCH0_PLAN.md"
HELD = {**C.E5_HELDOUT, "501-510": tuple(range(501, 511)), "601-610": tuple(range(601, 611))}
FROZEN_SETS = ("501-510", "601-610")
P0 = "p0"
SIZES = {"0.60": 0.60, "0.50": 0.50}
J_CONDS = ("J1", "J2", "J3", "J4")
CONDS = J_CONDS + tuple(C.E5_PUSHES)          # the 10 conditions
TOP_N = 6
X6_ALLOW_RPM = 108.0                          # rl/analysis/gait_pipe1.py RPM_ALLOW (0.70 x 176 x 42 / 48), reported
FORE_KNEES = ("left_fore_flexion", "right_fore_flexion")


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def model_path(size_key: str) -> Path:
    if size_key == "0.60":
        return C.MODEL
    path = MODELS / "trike_scaled_H0.50_D0_adopted_L1.xml"
    info_p = MODELS / "trike_scaled_H0.50_D0_adopted_L1.info.json"
    if not path.exists():
        import build_scaled_trike as BS
        text, info = BS.build_xml(0.50, "D0", "adopted", layout="L1")
        MODELS.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        info_p.write_text(json.dumps(info, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    info = json.loads(info_p.read_text())
    got = hashlib.sha256(path.read_text().encode()).hexdigest()
    if got != info["xml_sha256"]:
        raise RuntimeError(f"50 cm model sha256 {got} differs from its build record {info['xml_sha256']}")
    return path


# --------------------------------------------------------------------------------------------- the worker
_STASH: dict = {}


class LazyLogger:
    """A StepLogger built on the episode's own env at its first reset (the trot's episode functions make their env
    inside), as E1's runner does."""

    def __init__(self):
        self.lg = None

    def on_reset(self, env):
        if self.lg is None:
            self.lg = C.StepLogger(env)
        self.lg.on_reset(env)

    def __call__(self, env):
        self.lg(env)

    def result(self):
        return self.lg.result()


def _init():
    """Pass-through wrappers (R5): store what the build's own functions return; change nothing."""
    import gait_metrics
    F = C._f128()
    if not getattr(gait_metrics.measures, "_e5_wrapped", False):
        orig_m = gait_metrics.measures

        def measures(a, command, *args, **kw):
            out = orig_m(a, command, *args, **kw)
            _STASH["ms"] = out
            return out
        measures._e5_wrapped = True
        gait_metrics.measures = measures
    if not getattr(F.record_with_speed, "_e5_wrapped", False):
        orig_r = F.record_with_speed

        def record_with_speed(arr, p, seed, pusher):
            out = orig_r(arr, p, seed, pusher)
            _STASH["rec"] = out
            return out
        record_with_speed._e5_wrapped = True
        F.record_with_speed = record_with_speed


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x


def _clean(x):
    """For the committed result files: plain JSON (a non-finite float written as the string "inf" / "nan")."""
    x = _jsonable(x)
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


ALLOWED_SEEDS = {"s1": set(C.E5_SCREEN_SEEDS), "s2": set(C.E5_TUNING_SEEDS), "s3": set(C.E5_TUNING_SEEDS),
                 "heldout": {s for k in HELD for s in HELD[k]},
                 "heldout_frozen": {s for k in FROZEN_SETS for s in HELD[k]}}


def _check_plan():
    rec = SCR / "plan_sha256.txt"
    if not PLAN.exists() or not rec.exists() or rec.read_text().split()[0] != sha(PLAN):
        raise RuntimeError("the pitch-0 plan is missing or changed since its sha256 was recorded")


def _extra(size_key):
    ex = {"pitch0_plan": PLAN}
    if size_key == "0.50":
        ex["model_50cm"] = model_path(size_key)
    return ex


def ep_file(size_key, stage, sid, cond, seed) -> Path:
    return EPI / size_key / stage / f"{sid}__{cond}__{seed}.json"


def run_job(job):
    size_key, stage, setting, cond, seed = job
    sid = setting["id"]
    if seed not in ALLOWED_SEEDS[stage]:
        raise RuntimeError(f"seed {seed} not allowed in stage {stage}")
    fn = ep_file(size_key, stage, sid, cond, seed)
    if fn.exists():
        return (size_key, stage, sid, cond, seed), "kept"
    _init()
    t0 = time.time()
    size = SIZES[size_key]
    mp_ = model_path(size_key)
    P = C.walk_params(setting, size)
    pitch = float(setting["body_pitch_deg"])
    if pitch != 0.0:
        raise RuntimeError(f"pitch-0 branch: setting {sid} has body pitch {pitch}")
    lg = LazyLogger()
    _STASH.clear()
    if cond in J_CONDS:
        rec = C.trot_judged_episode(mp_, cond, seed, P, pitch, extra_hooks=(lg,))
        ms = _STASH.get("ms") or {}
        slip = {"source": "gait_metrics.measures slip_ratio", "by_foot": ms.get("slip_ratio")}
    else:
        axis, dv = C.E5_PUSHES[cond]
        rec = C.trot_push_episode(mp_, axis, dv, seed, P, pitch, extra_hooks=(lg,))
        sr = _STASH.get("rec") or {}
        slip = {"source": "sweep.episode_record slip (gait_pipe1 stance_slip slip_ratio)",
                "by_foot": dict(zip(C.CORNERS, sr["slip"])) if sr.get("slip") is not None else None}
    res = lg.result()
    p99 = C.weighted_percentile_from_hist([res["rpm_hist"]], [1.0], res["rpm_bin"], 99.0)
    acts = list(res["actuators"])
    fell = bool(rec["fell"])
    s1020 = rec.get("speed_10_20_mps")
    out = {"size": size_key, "stage": stage, "setting": setting, "condition": cond, "seed": seed,
           "walk_params": {k: getattr(P, k) for k in ("period_s", "duty", "lift_m", "max_reach_m")},
           "body_pitch_deg": pitch, "record": _jsonable(rec), "fell": fell, "termination": rec["termination"],
           "slow_gait": bool(C.is_slow_gait(rec)), "worst_hind_knee": _jsonable(C.worst_hind_knee(rec)),
           "slow": (not fell) and s1020 is not None and s1020 < 0.225,
           "actuators": acts, "rpm_peak": _jsonable(res["rpm_peak"]), "rpm_p99": _jsonable(p99),
           "window_samples": int(res["window_samples"]), "slip": _jsonable(slip),
           "model": mp_.name, "wall_s": round(time.time() - t0, 2)}
    fn.parent.mkdir(parents=True, exist_ok=True)
    tmp = fn.with_suffix(".tmp")
    tmp.write_text(json.dumps(out))
    tmp.replace(fn)
    return (size_key, stage, sid, cond, seed), f"{'FELL ' + str(rec['termination']) if fell else 'ok'} " \
                                                 f"slowgait={out['slow_gait']} hk={out['worst_hind_knee']} " \
                                                 f"{time.time() - t0:.1f}s"


def load_ep(size_key, stage, sid, cond, seed) -> dict:
    return json.loads(ep_file(size_key, stage, sid, cond, seed).read_text())


def as_rule_ep(e: dict) -> dict:
    """The episode as candidate_checks' rule functions read it (the episode function's own record)."""
    return dict(e["record"])


def by_condition(size_key, stage, sid, conds, seeds) -> dict:
    return {c: [as_rule_ep(load_ep(size_key, stage, sid, c, s)) for s in seeds] for c in conds}


# --------------------------------------------------------------------------------------------- pool
def pool_map(items, workers, on_done=None, log=None):
    import multiprocessing as mp
    t0 = time.time()
    with mp.get_context("spawn").Pool(workers, initializer=_init) as pool:
        for i, (k, msg) in enumerate(pool.imap_unordered(run_job, items, chunksize=1)):
            line = f"[{i + 1}/{len(items)} {time.time() - t0:.0f}s] {k}: {msg}"
            print(line, flush=True)
            if on_done is not None:
                on_done(k)


def plan_stage1(size_key) -> list:
    plan = json.loads((C.RESULTS / "plan.json").read_text())
    frozen = plan["E5"]["sizes"][size_key]["stage1"]
    mine = C.e5_settings(SIZES[size_key])
    if json.dumps(frozen, sort_keys=True) != json.dumps(mine, sort_keys=True):
        raise RuntimeError("stage-1 settings differ from plan.json's")
    out = []
    for i, st in enumerate(mine):          # the same points, body pitch set to 0 (prereg item 9)
        assert st["id"] == f"H{SIZES[size_key]:.2f}-s1-{i:02d}"
        out.append({**st, "body_pitch_deg": 0.0, "id": f"H{SIZES[size_key]:.2f}-{P0}-s1-{i:02d}",
                    "source_point": st["id"], "source_body_pitch_deg": st["body_pitch_deg"]})
    return out


def values(s):
    return {k: s[k] for k in C.E5_PARAMS}


# --------------------------------------------------------------------------------------------- tuning log
def tuning_log_path(size_key) -> Path:
    return SCR / f"tuning_log_{size_key}.jsonl"


def append_log(size_key, row):
    with tuning_log_path(size_key).open("a") as f:
        f.write(json.dumps(_clean(row)) + "\n")


def tune(size_key, workers):
    _check_plan()
    size = SIZES[size_key]
    model_path(size_key)
    (SCR / f"runner_sha256_at_tune_{size_key}.txt").write_text(
        f"{sha(RUNNER)}  run_e5.py\n{sha(BUILD / 'sim' / 'candidate_checks.py')}  sim/candidate_checks.py\n"
        f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    logged = set()
    if tuning_log_path(size_key).exists():
        for ln in tuning_log_path(size_key).read_text().splitlines():
            r = json.loads(ln)
            logged.add((r["stage"], r["id"]))

    # ---- stage 1
    s1 = plan_stage1(size_key)
    seeds1 = C.E5_SCREEN_SEEDS
    jobs = [(size_key, "s1", st, c, sd) for st in s1 for c in C.E5_SCREEN_CONDITIONS for sd in seeds1]
    need = {st["id"]: len(C.E5_SCREEN_CONDITIONS) * len(seeds1) for st in s1}
    done = {k: 0 for k in need}
    by_id = {st["id"]: st for st in s1}

    def s1_done(k):
        sid = k[2]
        done[sid] += 1
        if done[sid] == need[sid] and ("s1", sid) not in logged:
            bc = by_condition(size_key, "s1", sid, C.E5_SCREEN_CONDITIONS, seeds1)
            sc = C.e5_screen_score(bc)
            append_log(size_key, {"stage": "s1", "id": sid, **values(by_id[sid]), "score": list(sc),
                                  "episodes": need[sid], "finished_utc": time.strftime("%H:%M:%S", time.gmtime())})
            logged.add(("s1", sid))
    pool_map(jobs, workers, s1_done)
    for st in s1:  # any setting whose episodes were all kept from an earlier invocation
        if ("s1", st["id"]) not in logged:
            done[st["id"]] = need[st["id"]] - 1
            s1_done((None, None, st["id"]))
    scores = {st["id"]: C.e5_screen_score(by_condition(size_key, "s1", st["id"], C.E5_SCREEN_CONDITIONS, seeds1))
              for st in s1}
    order = sorted(range(len(s1)), key=lambda i: (scores[s1[i]["id"]], i))
    top = [s1[i] for i in order[:TOP_N]]
    print("stage 1 ranking:", [(s1[i]["id"], scores[s1[i]["id"]]) for i in order], flush=True)

    # ---- stage 2
    def run_full(stage, settings):
        jobs = [(size_key, stage, st, c, sd) for st in settings for c in CONDS for sd in C.E5_TUNING_SEEDS]
        need = {st["id"]: len(CONDS) * len(C.E5_TUNING_SEEDS) for st in settings}
        done = {k: 0 for k in need}
        ids = {st["id"]: st for st in settings}

        def fin(k):
            sid = k[2]
            done[sid] += 1
            if done[sid] == need[sid] and (stage, sid) not in logged:
                bc = by_condition(size_key, stage, sid, CONDS, C.E5_TUNING_SEEDS)
                su = C.e5_success(bc)
                append_log(size_key, {"stage": stage, "id": sid, **values(ids[sid]), "success": su,
                                      "failed_checks": sum(not v for v in su["checks"].values()),
                                      "episodes": need[sid],
                                      "finished_utc": time.strftime("%H:%M:%S", time.gmtime())})
                logged.add((stage, sid))
        pool_map(jobs, workers, fin)
        for st in settings:
            if (stage, st["id"]) not in logged:
                done[st["id"]] = need[st["id"]] - 1
                fin((None, None, st["id"]))
        return {st["id"]: C.e5_success(by_condition(size_key, stage, st["id"], CONDS, C.E5_TUNING_SEEDS))
                for st in settings}

    res2 = run_full("s2", top)
    # R2: the 20 shared episodes must equal stage 1's
    mism = []
    for st in top:
        for c in C.E5_SCREEN_CONDITIONS:
            for sd in seeds1:
                a = load_ep(size_key, "s1", st["id"], c, sd)
                b = load_ep(size_key, "s2", st["id"], c, sd)
                for k in ("record", "rpm_peak", "rpm_p99", "slip", "fell", "slow_gait", "worst_hind_knee"):
                    if json.dumps(a[k], sort_keys=True) != json.dumps(b[k], sort_keys=True):
                        mism.append((st["id"], c, sd, k))
    print(f"stage 2 determinism check: {len(mism)} mismatching fields over {len(top) * 20} shared episodes", flush=True)
    results = dict(res2)
    stage_of = {k: "s2" for k in res2}
    settings_of = {st["id"]: st for st in top}
    s3 = []
    if not any(r["SUCCEEDS"] for r in res2.values()):
        best_id = C.e5_choose(res2)                        # R1
        s3 = [st for st in C.e5_neighbours(settings_of[best_id], size, 0) if st["body_pitch_deg"] == 0.0]
        for i, st in enumerate(s3):
            st["id"] = f"H{size:.2f}-{P0}-s3-{i:02d}"
        print(f"stage 3 around {best_id}: {[values(s) for s in s3]}", flush=True)
        res3 = run_full("s3", s3)
        results.update(res3)
        stage_of.update({k: "s3" for k in res3})
        settings_of.update({st["id"]: st for st in s3})
    else:
        best_id = None
    n_settings = len(s1) + len(s3)
    n_runs = len(s1) + len(top) + len(s3)
    assert n_runs <= C.E5_MAX_SETTINGS, n_runs
    chosen = C.e5_choose(results)
    choice = {"experiment": "E5 pitch-0 branch (changed after a result)", "size": size_key, "evidence": "SIMULATION", "date": C.DATE,
              "written_before_any_heldout_seed": True,
              "heldout_episode_files_present_when_written": sorted(
                  str(p.relative_to(EPI)) for st in ("heldout", "heldout_frozen")
                  for p in (EPI / size_key / st).glob("*.json")) if (EPI / size_key).exists() else [],
              "seeds_used": sorted({*C.E5_SCREEN_SEEDS, *C.E5_TUNING_SEEDS}),
              "rule_of_choice": "candidate_checks.e5_choose over every setting run on seeds 101-110 (stages 2 and 3): "
                                "of those meeting e5_success, the lowest worst hind knee, then fewer slow-gait episodes, "
                                "then the higher J3 speed; if none meets it, the fewest failed checks first",
              "chosen_id": chosen, "chosen_stage": stage_of[chosen], "chosen_values": values(settings_of[chosen]),
              "chosen_meets_rule_on_tuning_seeds": results[chosen]["SUCCEEDS"],
              "any_setting_meets_rule_on_tuning_seeds": any(r["SUCCEEDS"] for r in results.values()),
              "stage1_ranking": [{"id": s1[i]["id"], "score": list(scores[s1[i]["id"]])} for i in order],
              "stage2_ids": [st["id"] for st in top], "stage3_centre": best_id,
              "stage3_ids": [st["id"] for st in s3],
              "tuning_results": _clean({k: {"stage": stage_of[k], "values": values(settings_of[k]), **v}
                                           for k, v in results.items()}),
              "budget": {"distinct_settings": n_settings, "setting_runs_counted_over_stages": n_runs,
                         "max": C.E5_MAX_SETTINGS},
              "determinism_check_stage2_vs_stage1": {"shared_episodes": len(top) * 20, "mismatching_fields": mism},
              "provenance": C.provenance(RUNNER, extra=_extra(size_key))}
    OUT.mkdir(parents=True, exist_ok=True)
    cp = OUT / f"e5p0_choice_{size_key}.json"
    if cp.exists():
        old = json.loads(cp.read_text())
        if old["chosen_id"] != chosen:
            raise RuntimeError(f"an earlier choice {old['chosen_id']} exists and differs from {chosen}")
        print(f"choice file exists (chosen {chosen}); kept", flush=True)
    else:
        cp.write_text(json.dumps(_clean(choice), indent=1) + "\n")
        (SCR / f"e5p0_choice_{size_key}.sha256").write_text(
            f"{sha(cp)}  {cp.name}\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
        print(f"wrote {cp}: chosen {chosen} ({stage_of[chosen]}) {values(settings_of[chosen])}, "
              f"meets rule on 101-110: {results[chosen]['SUCCEEDS']}", flush=True)
    return choice


def heldout(size_key, workers):
    _check_plan()
    cp = OUT / f"e5p0_choice_{size_key}.json"
    shp = SCR / f"e5p0_choice_{size_key}.sha256"
    if not cp.exists() or not shp.exists():
        raise RuntimeError("no choice file: run tune first")
    if shp.read_text().split()[0] != sha(cp):
        raise RuntimeError("the choice file changed after it was written")
    choice = json.loads(cp.read_text())
    st = {**choice["chosen_values"], "id": choice["chosen_id"]}
    frozen = {**{k: v for k, v in C.e5_default(SIZES[size_key]).items()}, "id": f"H{size_key}-frozen"}   # R4
    model_path(size_key)
    (SCR / f"runner_sha256_at_heldout_{size_key}.txt").write_text(
        f"{sha(RUNNER)}  run_e5.py\n{sha(BUILD / 'sim' / 'candidate_checks.py')}  sim/candidate_checks.py\n"
        f"{sha(cp)}  {cp.name}\n{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
    seeds = [s for k in HELD for s in HELD[k]]
    jobs = [(size_key, "heldout", st, c, sd) for c in CONDS for sd in seeds]
    pool_map(jobs, workers)
    fseeds = [s for k in FROZEN_SETS for s in HELD[k]]
    jobs = [(size_key, "heldout_frozen", frozen, c, sd) for c in CONDS for sd in fseeds]
    pool_map(jobs, workers)


# --------------------------------------------------------------------------------------------- summaries
def ep_reported(e: dict) -> dict:
    acts = e["actuators"]
    pk, p99 = np.array(e["rpm_peak"], dtype=float), np.array(e["rpm_p99"], dtype=float)
    x4 = [i for i, a in enumerate(acts) if a not in FORE_KNEES]
    x6 = [i for i, a in enumerate(acts) if a in FORE_KNEES]
    r = e["record"]
    slip = (e["slip"] or {}).get("by_foot") or {}
    sv = [v for v in slip.values() if v is not None]
    return {"fell": e["fell"], "slow_gait": e["slow_gait"], "slow": e["slow"],
            "worst_hind_knee": float(e["worst_hind_knee"]),
            "speed_2_20": r.get("forward_speed_mps"), "speed_10_20": r.get("speed_10_20_mps"),
            "pitch_10_20": r.get("pitch_10_20_deg"),
            "x4_peak": float(pk[x4].max()), "x4_p99": float(p99[x4].max()),
            "x6_peak": float(pk[x6].max()), "x6_p99": float(p99[x6].max()),
            "x4_peak_drive": acts[x4[int(np.argmax(pk[x4]))]], "x4_p99_drive": acts[x4[int(np.argmax(p99[x4]))]],
            "slip_max": max(sv) if sv else None, "wall_s": e["wall_s"],
            "speed_2_8": r.get("speed_2_8_mps"), "hind_knee_2_8": r.get("hind_knee_rms_over_rating_2_8")}


def cond_table(size_key, stage, sid, seeds) -> dict:
    out = {}
    for c in CONDS:
        eps = [ep_reported(load_ep(size_key, stage, sid, c, s)) for s in seeds]
        ok = [e for e in eps if not e["fell"]]

        def mx(k, rows=eps):
            v = [e[k] for e in rows if e[k] is not None]
            return max(v) if v else None

        def mn(k, rows=eps):
            v = [e[k] for e in rows if e[k] is not None]
            return float(np.mean(v)) if v else None
        out[c] = {"episodes": len(eps), "falls": sum(e["fell"] for e in eps),
                  "slow_gait": sum(e["slow_gait"] for e in eps), "slow": sum(e["slow"] for e in eps),
                  "mean_speed_2_20_mps": mn("speed_2_20"), "min_speed_10_20_mps": (
                      min(e["speed_10_20"] for e in ok if e["speed_10_20"] is not None) if ok else None),
                  "worst_hind_knee_max": mx("worst_hind_knee"),
                  "pitch_10_20_mean_deg": mn("pitch_10_20", ok),
                  "pitch_10_20_range_deg": ([min(e["pitch_10_20"] for e in ok if e["pitch_10_20"] is not None),
                                             max(e["pitch_10_20"] for e in ok if e["pitch_10_20"] is not None)]
                                            if ok else None),
                  "x4_peak_rpm_max": mx("x4_peak"), "x4_p99_rpm_max": mx("x4_p99"),
                  "x6_peak_rpm_max": mx("x6_peak"), "x6_p99_rpm_max": mx("x6_p99"),
                  "slip_ratio_max": mx("slip_max"),
                  "per_seed": [{"seed": s, **{k: e[k] for k in ("fell", "slow_gait", "slow", "worst_hind_knee",
                                                                "speed_2_20", "speed_10_20", "pitch_10_20",
                                                                "x4_peak", "x4_p99", "x6_peak", "x6_p99",
                                                                "slip_max", "speed_2_8", "hind_knee_2_8")}}
                               for s, e in zip(seeds, eps)]}
    return out


def drive_speed_flags(tab: dict) -> dict:
    x4pk = max(v["x4_peak_rpm_max"] for v in tab.values())
    x4p99 = max(v["x4_p99_rpm_max"] for v in tab.values())
    x6pk = max(v["x6_peak_rpm_max"] for v in tab.values())
    x6p99 = max(v["x6_p99_rpm_max"] for v in tab.values())
    return {"x4_peak_rpm": x4pk, "x4_p99_rpm": x4p99, "x6_fore_knee_peak_rpm": x6pk, "x6_fore_knee_p99_rpm": x6p99,
            "x4_p99_le_83": x4p99 <= C.CAP_RPM, "x4_peak_le_91.3": x4pk <= C.PEAK_FACTOR * C.CAP_RPM,
            "x6_p99_le_83": x6p99 <= C.CAP_RPM, "x6_p99_le_108": x6p99 <= X6_ALLOW_RPM,
            "x6_peak_le_118.8": x6pk <= C.PEAK_FACTOR * X6_ALLOW_RPM,
            "note": "reported, not judged (E5 rule); X4 against 83 rpm (DECISION_LOG.md E1), the X6 fore knees "
                    "against 83 rpm and, beside it, 108 rpm (gait_pipe1 RPM_ALLOW); peak against 1.1 x each"}


def summarise(size_key):
    choice = json.loads((OUT / f"e5p0_choice_{size_key}.json").read_text())
    rows = [json.loads(ln) for ln in tuning_log_path(size_key).read_text().splitlines()]
    for r in rows:                      # review E5-R4: each row's seeds and episode wall time
        sd = list(C.E5_SCREEN_SEEDS) if r["stage"] == "s1" else list(C.E5_TUNING_SEEDS)
        cs = C.E5_SCREEN_CONDITIONS if r["stage"] == "s1" else CONDS
        r["seeds"] = sd
        r["conditions"] = list(cs)
        r["episode_wall_s_sum"] = round(sum(load_ep(size_key, r["stage"], r["id"], c, s)["wall_s"]
                                            for c in cs for s in sd), 1)
    s1_rank = {r["id"]: i + 1 for i, r in enumerate(choice["stage1_ranking"])}
    fields = ["order_finished", "id", "stage", *C.E5_PARAMS, "episodes", "seeds", "conditions", "stage1_rank",
              "screen_falls_J1_J3", "screen_slow_gait", "screen_worst_hind_knee", "screen_J3_speed_mps",
              "rule_no_falls_J1_J4", "rule_J3_speed", "rule_slow_gait_le_1", "rule_hind_knee_le_1", "failed_checks",
              "meets_rule_on_101_110", "falls_J1_J4", "J3_speed_mps", "slow_gait_total", "slow_gait_max_condition",
              "worst_hind_knee_max", "chosen"]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"e5p0_tuning_{size_key}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            d = {"order_finished": i + 1, "id": r["id"], "stage": r["stage"], **{k: r[k] for k in C.E5_PARAMS},
                 "episodes": r["episodes"], "stage1_rank": s1_rank.get(r["id"]) if r["stage"] == "s1" else "",
                 "chosen": r["id"] == choice["chosen_id"] and r["stage"] == choice["chosen_stage"]}
            if r["stage"] == "s1":
                sc = r["score"]
                d.update(seeds="101-105", conditions="J1 J3 push_back_0.5 push_right_0.5", screen_falls_J1_J3=sc[0],
                         screen_slow_gait=sc[1], screen_worst_hind_knee=sc[2], screen_J3_speed_mps=-sc[3])
            else:
                su = r["success"]
                ch = su["checks"]
                d.update(seeds="101-110", conditions="J1-J4 and the six pushes",
                         rule_no_falls_J1_J4=ch["no_falls_J1_J4"], rule_J3_speed=ch["J3_speed"],
                         rule_slow_gait_le_1=ch["slow_gait_at_most_1_per_condition"],
                         rule_hind_knee_le_1=ch["hind_knee_at_most_1"], failed_checks=r["failed_checks"],
                         meets_rule_on_101_110=su["SUCCEEDS"], falls_J1_J4=sum(su["falls_J"].values()),
                         J3_speed_mps=round(su["J3_speed_mps"], 5), slow_gait_total=sum(su["slow_gait"].values()),
                         slow_gait_max_condition=max(su["slow_gait"].values()),
                         worst_hind_knee_max=round(max(float(x) for x in su["worst_hind_knee"].values()), 4))
            w.writerow(d)
    tun = {"experiment": "E5 pitch-0 branch (changed after a result)", "size": size_key, "evidence": "SIMULATION", "date": C.DATE,
           "prereg": "design/PREREG_CANDIDATES_2026_11_28.md section 4 E5",
           "model": (C.MODEL.name if size_key == "0.60" else
                     "build_scaled_trike.build_xml(0.50, 'D0', 'adopted', layout='L1') (scratch copy)"),
           "rows": rows, "choice_file": f"e5p0_choice_{size_key}.json",
           "stage2_determinism_check": choice["determinism_check_stage2_vs_stage1"], "budget": choice["budget"],
           "provenance": C.provenance(RUNNER, extra=_extra(size_key))}
    (OUT / f"e5p0_tuning_{size_key}.json").write_text(json.dumps(_clean(tun), indent=1) + "\n")

    held = {}
    for stage, sid in (("heldout", choice["chosen_id"]), ("heldout_frozen", f"H{size_key}-frozen")):
        if not (EPI / size_key / stage).exists():
            continue
        per = {}
        sets = HELD if stage == "heldout" else {k: HELD[k] for k in FROZEN_SETS}
        for name, seeds in sets.items():
            if not all(ep_file(size_key, stage, sid, c, s).exists() for c in CONDS for s in seeds):
                per = None
                break
            su = C.e5_success(by_condition(size_key, stage, sid, CONDS, seeds))
            tab = cond_table(size_key, stage, sid, seeds)
            per[name] = {"rule": su, "reported": tab, "drive_speeds": drive_speed_flags(tab)}
        if per is None:
            continue
        held[stage] = {"id": sid, "values": (choice["chosen_values"] if stage == "heldout"
                                             else values(C.e5_default(SIZES[size_key]))),
                       "judged": stage == "heldout", "by_seed_set": per,
                       "SUCCEEDS_all_four_sets" if stage == "heldout" else "would_meet_rule_both_sets_supporting":
                           all(v["rule"]["SUCCEEDS"] for v in per.values())}
        if stage == "heldout":
            held[stage]["reading_lead_sets_301_310_401_410"] = all(per[k]["rule"]["SUCCEEDS"]
                                                                   for k in ("301-310", "401-410"))
            held[stage]["reading_rerun_sets_501_510_601_610"] = all(per[k]["rule"]["SUCCEEDS"]
                                                                    for k in FROZEN_SETS)
    if held:
        wall = []
        for stage in held:
            for p in (EPI / size_key / stage).glob("*.json"):
                wall.append(json.loads(p.read_text())["wall_s"])
        out = {"experiment": "E5 pitch-0 branch (changed after a result)", "size": size_key, "evidence": "SIMULATION", "date": C.DATE,
               "prereg": "design/PREREG_CANDIDATES_2026_11_28.md section 4 E5",
               "rule": "e5_success per held-out seed set (no falls in J1-J4; J3 mean speed 2-20 s >= 0.225 m/s; "
                       "slow-gait episodes <= 1 of 10 in every condition, a fall counting as one; worst hind knee "
                       "<= 1.0 of 6.75 N m in every episode of every condition); the pitch-0 branch SUCCEEDS only if all four "
                       "sets (301-310, 401-410, 501-510, 601-610) pass (E5_PITCH0_PLAN.md item 7)",
               "command_mps": C.E5_COMMAND, "conditions": list(CONDS), "heldout_seeds": HELD, "heldout_frozen_seeds": list(FROZEN_SETS),
               "choice": {k: choice[k] for k in ("chosen_id", "chosen_stage", "chosen_values",
                                                 "chosen_meets_rule_on_tuning_seeds")},
               "units": {"speed": "m/s", "hind_knee": "RMS / 6.75 N m continuous", "rpm": "rev/min at the joint "
                         "(1:1 drives)", "pitch": "deg, + nose down"},
               **held, "episode_wall_s_total": round(sum(wall), 1),
               "provenance": C.provenance(RUNNER, extra=_extra(size_key))}
        (OUT / f"e5p0_heldout_{size_key}.json").write_text(json.dumps(_clean(out), indent=1) + "\n")
    print(f"summarised {size_key}: {sorted(p.name for p in OUT.glob(f'*{size_key}*'))}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("tune", "heldout", "summarise"))
    ap.add_argument("--size", choices=tuple(SIZES), required=True)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)
    assert a.workers <= 2
    EPI.mkdir(parents=True, exist_ok=True)
    if a.stage == "tune":
        tune(a.size, a.workers)
    elif a.stage == "heldout":
        heldout(a.size, a.workers)
    else:
        summarise(a.size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
