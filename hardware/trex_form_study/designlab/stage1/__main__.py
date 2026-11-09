"""Command line for Stage 1 (``STAGE1_PLAN.md`` section 7).

Run from hardware/trex_form_study:

  python -m designlab.stage1 prepare --body 60-N-T-10 [--body ...] --out stage1/runs/<run> [--no-gates]
                                    [--parts plant,refs,transitions,stops] [--transitions start,stop,...]
  python -m designlab.stage1 gates   --body 60-N-T-10 --out stage1/runs/<run> [--only g1|g2|g3|g4]
  python -m designlab.stage1 tune    --body 60-N-T-10 --seed 42 [--evals 720] [--resume] [--workers 1]
                                    [--out stage1/runs/2026-11-08]
  python -m designlab.stage1 g5      --body 60-N-T-10 [--workers 1] [--out stage1/runs/2026-11-08]
  python -m designlab.stage1 train   --body 60-N-T-10 --seed 42 [--mode R3|R4] [--steps 5000000] [--resume]
                                    [--n-envs 4] [--vec dummy|subproc] [--allow-untuned] [--tune-json PATH]
                                    [--out stage1/runs/2026-11-08]
  python -m designlab.stage1 judge   [--mode R3|R4] [--final] [--out stage1/runs/2026-11-08]
  python -m designlab.stage1 noise   [--body 75-M-H-10] [--tune-json PATH] [--out stage1/runs/2026-11-08]
  python -m designlab.stage1 eval    --body 60-N-T-10 [--body ...] [--rows R0,R1,R2/42,R2/43] [--behaviors stand,...]
                                    [--conditions walk/nominal,...] [--workers 1] [--attribute] [--dry-run]
                                    [--episodes 20] [--row-module MODULE] [--no-traces] [--inputs DIR]
                                    --out stage1/runs/2026-11-08
  python -m designlab.stage1 sweep   --body 60-N-T-10 [--body ...] [--rows R2,R3/42] [--margins grip,push,...]
                                    [--workers 1] [--dry-run] [--no-measure] [--row-module MODULE] [--inputs DIR]
                                    --out stage1/runs/2026-11-08
  python -m designlab.stage1 report  --out stage1/runs/2026-11-08 [--dest DIR] [--body ...]

``prepare`` writes, per body, ``<out>/<body>/plant.json`` (gains, envelopes, ranges, sensors, floors, hashes,
library versions, commit), ``refs.json`` with ``refs/<name>.npz`` (``refs``), ``transitions.json`` with
``transitions/<name>.npz`` (``transitions``: starts, stops, speed changes and turn entries and exits) and, unless
``--no-gates``, ``gates.json`` (``gates``: G1-G4 of plan section 3; G4 runs the tracker R0-R2 of ``control``).
``--parts`` limits ``prepare`` to some of its products (the others must exist for the later ones); ``stops`` re-reads
the saved stop transitions' stops under ``STAGE1_PLAN.md`` decision 18 (``transitions.reread_stops``) without
regenerating them.  ``gates`` reruns the gates (or one) on the saved products.  Every gate record carries the
provenance of the code that made it (``plant.provenance``: commit, dirty flag, the hash of the designlab/stage1
sources).  G4 is judged at the tracker's tuning start with the panel rule (decisions 16 and 17); a G4 record made at
the build's earlier defaults is kept in ``G4.evidence.earlier_defaults``.
``--ankle-gains``, ``--ankle-armature`` and ``--g2-rule`` set the choices G2 depends on; the defaults are the ones
``STAGE1_PLAN.md`` decided during the build (decisions 11-13: joint-space ankle gains, the linkage's armature, the
20-start panel), and the others stay available for comparison.  They are recorded in plant.json and gates.json
(``G2.decided`` says whether a run follows the decisions).  The exit status is 1 if a gate fails (plan section 3:
stop).

``tune`` is plan section 3 step 6 (``tune``): the CMA-ES fit of one body and seed, checkpointed every generation to
``<out>/<body>/tune/<seed>/cma.json`` with every evaluation in ``evals.jsonl``; ``--resume`` continues a checkpointed
run exactly (``--evals`` may raise its budget; refused if the tuner's own modules, ``tune.TUNER_SOURCES_RULE``, or its
inputs, ``tune.INPUTS_RULE``, changed).  ``g5`` selects each finished seed's gains on the validation episodes and judges
gate G5 into ``<out>/<body>/tune.json`` (exit status 1 if no seed passes, or if a seed is refused because its fit ran
on other tuner sources or inputs).  One process per (body, seed) by default; ``--workers`` evaluates a generation (or
the validation episodes) in that many processes (a worker that dies or hangs is replaced, ``tune.Pool``).

``train`` is plan section 3 step 7 (``train``): one PPO run of R3 (the residual on R2, ``env.ResidualEnv``) or R4
(policy only, ``env.PolicyOnlyEnv``) into ``<out>/<body>/ppo/<seed>/`` (``policy_only`` for R4), checkpointed, with
``--resume`` and the G6 checks; it needs Stable-Baselines3, Gymnasium and torch (the PPO venv).  R3 trains on
tune.json's body-level G5 selection (``env.GAINS_RULE``) and is refused without a G5 pass unless ``--allow-untuned``.
``judge`` writes the checkpoint every run of a mode is judged at (``train.JUDGED_RULE``; ``--final`` once the stage's
cap stopped training) into their run.json; ``noise`` measures R3's survival under the initial action noise
(``train.action_noise_survival``) into ``<out>/ppo_action_noise.json``, the record ``policy.LOG_STD_INIT`` comes from.
``eval`` is plan section 3 step 8's behaviour panels (``panels``): the rows R0 and R1 (hard floor) and R2/42 and
R2/43 (every condition) on every condition of plan section 4, 20 episodes each (seeds 3042-3061), one record per row and
condition in ``<out>/<body>/eval/<row>/<behavior>/<condition>.json``, each episode's 50 Hz trace under
``<out>/<body>/eval/traces/`` (``panels.TRACE_RULE``; not with ``--no-traces``) and the grid with the verdicts and gate
G7 in ``<out>/<body>/eval/summary.json``; plan section 7's ``--seed`` is a row here (``--rows R2/42``);
``--attribute`` adds decision 20's fall attribution to the R2 rows' conditions with falls; a condition whose record is
up to date is skipped and an episode that failed to run is run again (``panels.RESUME_RULE``); ``--row-module``
imports a module that adds rows (``panels.register_rows``: the R3 and R4 rows); ``--dry-run`` lists every episode and
estimates the core-hours from a cost per simulated second measured on each floor (``panels.COST_RULE``).
``sweep`` is plan section 5's margin ledger (``ledger``): each margin bisected to failure on the rows R2 (the body-level
G5 selection's CMA-ES seed, ``--rows R2``) and R3/42 (the policy rows' row, ``policy_rows``, which the ledger registers
when no ``--row-module`` has: run it in the PPO venv), on seeds 3042-3044, one record per row and margin in
``<out>/<body>/sweeps/<row>/<margin>.json``, resumed from the evaluations the records hold (``ledger.RESUME_RULE``);
``--dry-run`` lists every search with the episodes it can still need and their cost (``ledger.COST_RULE``: a policy
row's own).  ``report`` writes the machine-made tables of plan section 8 (``report``) from whatever records
``--out`` holds into ``<dest>/STAGE1_TABLES.md`` and ``<dest>/stage1_tables.json`` (``--dest`` default: ``--out``).
``all`` is not implemented yet.
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")  # one core per process: the step-rate gate is per core

import argparse  # noqa: E402
import importlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import designlab  # noqa: E402,F401  (selects the offscreen GL backend before mujoco is imported)

from . import BODIES  # noqa: E402

LATER = {"all": "every step of plan section 3"}
#: PPO env steps per second at 50 Hz on 2 ms physics are at most a tenth of the physics rate (plan section 3, G6a).
ENV_SUBSTEPS = 10
G6A_ENV_STEPS_PER_S = 600
COARSE_PILE_GRID_M = 0.008  # the training pile of plan section 2


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=float))


def _floor_rate(body: str, floor: str, plant_kw: dict, pile_grid: float | None = None, steps: int = 2000) -> dict:
    from . import plant

    t0 = time.perf_counter()
    p = plant.build(body, floor, pile_grid=pile_grid, **plant_kw)
    build_s = time.perf_counter() - t0
    p.reset(allow_unsettled=True)
    p.hold()
    t1 = time.perf_counter()
    p.step(steps)
    rate = steps / (time.perf_counter() - t1)
    s = p.meta["settle"]
    return dict(params=p.meta["floor_params"], info=p.meta["floor_info"], settle=s, stands=bool(s["settled"]),
                stand_status="stands" if s["settled"] else f"does not stand: the zero-command stance {s['status']} "
                f"after {s['seconds']:.1f} s", build_s=build_s, steps_per_s_zero_command=rate,
                env_steps_per_s_bound_50hz=rate / ENV_SUBSTEPS,
                below_g6a_env_rate=bool(rate / ENV_SUBSTEPS < G6A_ENV_STEPS_PER_S))


def plant_doc(body: str, plant_kw: dict | None = None) -> dict:
    """``plant.json``: the hard-floor plant in full, and every floor's parameters, settle, stand status and step rate
    (also on the coarse training pile), with the floors whose physics alone caps PPO below G6a's 600 env steps/s."""
    from .. import carpet
    from . import plant

    plant_kw = plant_kw or {}
    doc = plant.describe(plant.build(body, "hard", **plant_kw))
    doc["floors"] = {floor: _floor_rate(body, floor, plant_kw) for floor in plant.FLOORS}
    doc["floors_coarse_pile"] = {floor: _floor_rate(body, floor, plant_kw, COARSE_PILE_GRID_M) for floor in carpet.PILES}
    doc["throughput"] = dict(
        rule=f"physics steps/s with zero command / {ENV_SUBSTEPS} (50 Hz env steps on 2 ms physics) is an upper bound "
             f"on the PPO env rate; G6a asks for {G6A_ENV_STEPS_PER_S} per job",
        below_g6a={f"{floor} ({grid})": round(v["env_steps_per_s_bound_50hz"])
                   for grid, floors in (("4 mm probes", doc["floors"]), ("8 mm probes", doc["floors_coarse_pile"]))
                   for floor, v in floors.items() if v["below_g6a_env_rate"]})
    return doc


GATES = ("g1", "g2", "g3", "g4")


def earlier_g4_evidence(prev: dict | None, gates_path: Path) -> dict | None:
    """The G4 record of the build's earlier defaults, kept as evidence (``STAGE1_PLAN.md`` decision 17): carried over
    from a record that already keeps it, or taken from a record made before decisions 15-20 (no ``judged_at``) with
    a note of the code it was committed with (that record names none)."""
    from . import plant

    if not prev:
        return None
    if prev.get("judged_at"):
        return (prev.get("evidence") or {}).get("earlier_defaults")
    commit = None
    try:
        import subprocess

        commit = subprocess.run(["git", "-C", str(plant.STUDY), "log", "-1", "--format=%H", "--",
                                 str(Path(gates_path).resolve())], capture_output=True, text=True,
                                check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        pass
    return dict(note="G4 as the build first ran it, before decisions 15-20: the tracker at its earlier (untuned) "
                     "defaults (control.EARLIER_DEFAULTS: heading 0.5 / 0.1, phase rate 2.0 / 1.0), no ankle cap, "
                     "the inward step clamped at the turn gap less 1 mm, the estimator's stance odometry at the foot's "
                     "centre of mass, every fall labelled by the episode-level counterfactual, judged on the "
                     "unperturbed runs; kept as evidence (decision 17)",
                record_names_code=False, committed_in=commit,
                committed_sources_sha256=plant.sources_sha256_at(commit) if commit else None,
                sources_rule=plant.SOURCES_RULE, record=prev)


def _provenanced(fn, *args, **kw) -> dict:
    """``fn(*args, **kw)`` with the provenance of the code that ran it (``plant.provenance``, taken before the run),
    and whether the Stage 1 sources were still the same when it finished (``unchanged_during_run``)."""
    from . import plant

    prov = plant.provenance()
    out = fn(*args, **kw)
    prov["stage1_sources"]["unchanged_during_run"] = plant.sources_sha256() == prov["stage1_sources"]["sha256"]
    return dict(out, provenance=prov)


def run_gates(body: str, out: Path, only: str | None = None, plant_kw: dict | None = None,
              rule: str | None = None) -> dict:
    """G1-G4 (or one of them) into ``<out>/gates.json``; ``rule`` is G2's (default: ``gates.RULES[0]``, the plan's
    decision 13).  G3 and G4 run on the decided plant (``plant.build``'s defaults).  Each gate record gets the
    provenance of the code that ran it (``_provenanced``).  ``pass`` needs every gate recorded so far to pass (and G1
    and G2 on the same plant)."""
    from . import gates

    plant_kw = plant_kw or {}
    rule = rule or gates.RULES[0]
    path = out / "gates.json"
    doc = json.loads(path.read_text()) if path.exists() else dict(body=body)
    if only in (None, "g1"):
        doc["G1"] = _provenanced(gates.g1, body, out, **plant_kw)
    if only in (None, "g2"):
        doc["G2"] = _provenanced(gates.g2, body, out, rule=rule, **plant_kw)
    if only in (None, "g3"):
        doc["G3"] = _provenanced(gates.g3, body, out)
    if only in (None, "g4"):
        evidence = earlier_g4_evidence(doc.get("G4"), path)
        doc["G4"] = dict(_provenanced(gates.g4, body, out), evidence=dict(earlier_defaults=evidence))
    same = (doc.get("G1", {}).get("plant_options") is not None
            and doc.get("G1", {}).get("plant_options") == doc.get("G2", {}).get("plant_options"))
    doc["same_plant_options"] = same
    passes = [doc[g]["pass" if g in ("G1", "G2", "G4") else "pass_"] for g in ("G1", "G2", "G3", "G4") if g in doc]
    plant_gates = "G1" in doc or "G2" in doc  # G1 and G2 must then both be there, on the same plant
    doc["pass"] = bool(passes and all(passes) and (same or not plant_gates))
    _write(path, doc)
    return doc


def summary(doc: dict) -> str:
    lines = []
    g1, g2 = doc.get("G1"), doc.get("G2")
    if g1:
        st = ", ".join(f"{f} {'stands' if s['stands'] else 'FAILS'}"
                       + (f" (tilt max {s['tilt_from_home_max_deg']:.2f} deg)" if "tilt_from_home_max_deg" in s else "")
                       for f, s in g1["stand"].items())
        sr = g1["step_rate"]
        hashes = all(h["matches_record"] and h["matches_plan"] for h in g1["hashes"].values())
        tm = g1["tendon_map"]
        per = sr.get("per_process_all_median", sr.get("per_process_mean"))
        lines.append(f"G1 {'pass' if g1['pass'] else 'FAIL'}: hashes {hashes}; {st}; tendon map err "
                     f"{max(tm['max_err_joint_torque_Nm'], tm['max_err_motor_speed_rad_s']):.1e}; "
                     f"repeat {g1['repeatability']['ok']}; step rate (median of {sr.get('samples_per_process', 1)} x "
                     f"{sr.get('steps_per_sample', sr.get('steps'))} steps) {sr['single_process']:.0f} steps/s alone, "
                     f"{per:.0f} per process x {sr['processes']} (lowest sample "
                     f"{sr.get('lowest_sample', min(sr.get('per_process', [per]))):.0f})")
    if g2:
        o = g2["plant_options"]
        pnl = g2["nominal_panel_with_envelope"]
        lines.append(f"G2 {'pass' if g2['pass'] else 'FAIL'} (rule {g2['rule']}; ankle gains {o['ankle_gains']}, "
                     f"armature {o['ankle_armature']}): references match {g2['all_match']} (nominal "
                     f"{g2['nominal_speed_m_s']} m/s, CoP margin {g2['nominal_min_cop_margin_single_m']} m); nominal "
                     f"panel median {pnl['median']:.1f} strides ({pnl['runs_at_least_min_strides']}/{pnl['runs']} at "
                     f"least {g2['min_strides']:g}, range {pnl['min']:.1f}-{pnl['max']:.1f}); unperturbed run "
                     f"{g2['nominal_strides_with_envelope']:.1f} strides with the envelope, "
                     f"{g2['nominal_strides_without_envelope']:.1f} without (design study, screened MJCF: "
                     f"{g2['design_study_strides_screened_model']})")
        others = ", ".join(f"{name} {v['with_envelope']['strides']:.1f}" for name, v in g2["replay"].items()
                           if name != "nominal")
        lines.append(f"   other replays with the envelope (strides): {others}")
        info = g2["reported_not_gated"]
        ch = "; ".join(f"{name}: {c['nominal_strides']:.1f}, panel median {c['panel']['median']:.1f} "
                       f"({c['panel']['runs_at_least_min_strides']}/{c['panel']['runs']})"
                       for name, c in info["plant_choices"].items())
        lines.append(f"   plant choices (nominal strides unperturbed, panel): {ch}")
        dec = g2.get("decided")
        if dec:
            lines.append("   decided (STAGE1_PLAN.md): " + "; ".join(
                f"({d['decision']}) {d['choice']} = {d['taken']}"
                + ("" if d["follows"] else f", NOT FOLLOWED: this run used {d['used']}") for d in dec["decisions"]))
    g3 = doc.get("G3")
    if g3:
        lines.append(f"G3 {'pass' if g3['pass_'] else 'FAIL'}: {len(g3['transitions'])} transitions; worst CoM "
                     f"{g3['worst_com']['mm']:.3f} mm ({g3['worst_com']['transition']}), worst ZMP "
                     f"{g3['worst_zmp']['mm']:.3f} mm ({g3['worst_zmp']['transition']}); failed: "
                     f"{', '.join(g3['failed']) or 'none'} (fall back to amplitude ramps: {', '.join(g3['fallback']) or 'none'})")
        st = g3.get("stops", {}).get("stops", {})
        if st:
            lines.append("   stops (decision 18, strides from the command): " + "; ".join(
                f"{n} forward {'-' if s.get('forward_stopped_strides') is None else format(s['forward_stopped_strides'], '.3f')}"
                f" (within {s.get('within')}), rest {s.get('at_rest_strides', float('nan')):.3f}" for n, s in st.items()))
    g4 = doc.get("G4")
    if g4 and g4.get("judged_at"):
        parts = []
        for key, g in g4["gated"].items():
            u, pnl = g["unperturbed"], g["panel"]
            parts.append(f"{key} panel median {pnl['median']:.1f} (>= {g['threshold_strides']:g}: {g['panel_pass']}; "
                         f"{pnl['runs_at_least_threshold']}/{pnl['runs']} reach it, {pnl['survived']} survive); "
                         f"unperturbed {u['strides']:.1f}")
        lines.append(f"G4 {'pass' if g4['pass'] else 'FAIL'} (panel rule, at the tuning start): " + "; ".join(parts))
        for key, a in (g4.get("attribution") or {}).items():
            tests = ", ".join(f"{r} p {t['sign_test']['p']:.3g} ({t['survived']} survive)" for r, t in a["lifted"].items())
            lines.append(f"   {key}: {a['label']}; candidates {', '.join(a['candidates']) or 'none'}; lifted: "
                         f"{tests or 'nothing relaxable'}; null p {a['null']['sign_test']['p']:.3g}")
        u2 = g4["gated"]["R2_nominal_hard"]["unperturbed"].get("ankle_cap_residual")
        if u2:
            lines.append(f"   R2 nominal (unperturbed): realised ankle torques beyond the cap: box {u2['box_frac']:.3f} of "
                         f"updates (max {u2['box_max_excess_Nm']:.2g} N m), pad {u2['pad_frac']:.3f} (max "
                         f"{u2['pad_max_excess_Nm']:.2g} N m; feet over 20 % of the weight {u2['heavy_pad_frac']:.3f}, "
                         f"max {u2['heavy_pad_max_excess_Nm']:.2g} N m)")
        lines.append("   R0: " + ", ".join(f"{n} {r['strides']:.1f}" for n, r in g4["R0"].items()))
        for mode, rows in g4["reported"].items():
            lines.append(f"   {mode}: " + ", ".join(f"{n} {r['strides']:.1f}" for n, r in rows.items()))
        for mode, r in g4["stance_start_stop"].items():
            stop = (r.get("stops") or [{}])[0]
            lines.append(f"   {mode} start-walk-stop: {r['strides']:.1f} strides, fell {r['fell']}; stop "
                         + (f"forward {stop.get('forward_stopped_strides')} strides (within {stop.get('within')}), rest "
                            f"{stop.get('at_rest_strides')}" if stop.get("reached") else "not reached"))
    return "\n".join(lines)


def _tuning(a) -> int:
    """``tune`` and ``g5`` (plan section 3 step 6)."""
    from . import tune

    out = Path(a.out)
    if a.cmd == "tune":
        doc = tune.tune(a.body, a.seed, evals=a.evals, resume=a.resume, out=out, workers=a.workers,
                        inputs=a.inputs, log=lambda msg: print(msg, flush=True))
        h = doc["history"][-1] if doc["history"] else None
        print(f"{a.body} seed {a.seed}: {doc['evals']} evaluations ({doc['status']}), "
              + (f"last generation J min {h['J_min']:.4f}, sigma {doc['state']['sigma']:.4f}" if h else "none run"),
              flush=True)
        return 0
    seeds = tuple(int(x) for x in a.seeds.split(","))
    doc = tune.g5(a.body, out=out, seeds=seeds, workers=a.workers, episodes=a.episodes,
                  require_done=not a.partial, inputs=a.inputs, log=lambda msg: print(msg, flush=True))
    print(tune.g5_summary(doc), flush=True)
    if doc["seeds_refused"]:  # a fit on other tuner sources or inputs: no verdict, and not a silent one
        return 1
    return 0 if doc["pass_"] or doc["pass_"] is None else 1


def _training(a) -> int:
    """``train`` (plan section 3 step 7): one PPO run of R3 or R4 (``train``); ``judge``; ``noise``."""
    from . import train

    if a.cmd == "judge":
        rec = train.judge(Path(a.out), a.mode, a.budget, a.final)
        print(f"{a.mode}: judged at {rec['steps']} steps ({'final' if rec['final'] else 'provisional'}); "
              + ", ".join(f"{k} {v['last_checkpoint']}" for k, v in rec["runs"].items()), flush=True)
        return 0
    if a.cmd == "noise":
        doc = train.action_noise_survival(a.body, run_dir=str(Path(a.out) / a.body), tune_json=a.tune_json,
                                          log=lambda msg: print(msg, flush=True))
        _write(Path(a.out) / "ppo_action_noise.json", doc)
        print(f"{a.body}: survival {doc['survived']} of {doc['of']}; sigma {doc['sigma']}, log_std_init "
              f"{doc['log_std_init']}", flush=True)
        return 0
    config = dict(tune_json=a.tune_json) if a.tune_json else None
    steps = train.BUDGET_ASKED if a.steps is None else a.steps
    fix = None
    if a.fix_task_weights:
        from . import env as env_mod

        w = tuple(float(x) for x in a.fix_task_weights.split(","))
        if len(w) != len(env_mod.TASKS) or min(w) < 0 or not sum(w) > 0:
            raise SystemExit(f"--fix-task-weights: {len(env_mod.TASKS)} non-negative weights over {list(env_mod.TASKS)}")
        fix = dict(task_weights=w)
    doc = train.train(a.body, a.seed, mode=a.mode, steps=steps, out=Path(a.out), resume=a.resume, n_envs=a.n_envs,
                      vec=a.vec, checkpoint_every=a.checkpoint_every, g6=not a.no_g6, log_std_init=a.log_std_init,
                      allow_untuned=a.allow_untuned, allow_source_change=a.allow_source_change, config=config,
                      fix=fix, fix_reason=a.fix_reason, log=lambda msg: print(msg, flush=True))
    g6 = doc.get("g6", {})
    print(f"{a.body} {a.mode} seed {a.seed}: {doc.get('num_timesteps')} steps ({doc['status']}); G6: "
          + ("; ".join(f"{k} {'pass' if v.get('pass_', v.get('identical')) else 'FAIL'}" for k, v in g6.items())
             or "not run"), flush=True)
    return 0


def _evaluation(a) -> int:
    """``eval`` (plan section 3 step 8): the behaviour panels (``panels``), or their dry run."""
    from . import panels

    for mod in a.row_module or ():
        importlib.import_module(mod)  # it registers its rows (panels.register_rows)
    rows = tuple(x for x in a.rows.split(",") if x)
    behaviors = tuple(a.behaviors.split(",")) if a.behaviors else None
    names = tuple(a.conditions.split(",")) if a.conditions else None
    for body in a.body:
        if a.dry_run:
            panels.dry_run(body, Path(a.out), rows, behaviors, names, a.episodes, a.inputs, measure=not a.no_measure,
                           workers=a.workers, log=lambda msg: print(msg, flush=True))
            continue
        doc = panels.evaluate(body, Path(a.out), rows, behaviors, names, a.workers, a.attribute, a.episodes, a.inputs,
                              traces=not a.no_traces, log=lambda msg: print(msg, flush=True))
        print(panels.summary_text(doc), flush=True)
    return 0


def _sweeping(a) -> int:
    """``sweep`` (plan section 5's margin ledger, ``ledger``), or its dry run."""
    from . import ledger

    for mod in a.row_module or ():
        importlib.import_module(mod)  # it registers its rows (panels.register_rows)
    rows = tuple(x for x in a.rows.split(",") if x)
    names = tuple(a.margins.split(",")) if a.margins else None
    for body in a.body:
        try:
            if a.dry_run:
                ledger.dry_run(body, Path(a.out), rows, names, a.inputs, a.cut_seconds, a.halvings,
                               measure=not a.no_measure, workers=a.workers, log=lambda msg: print(msg, flush=True))
                continue
            ledger.sweep(body, Path(a.out), rows, names, a.workers, a.inputs, a.cut_seconds, a.halvings,
                         log=lambda msg: print(msg, flush=True))
        except ValueError as exc:
            if "register_rows" in str(exc):
                raise SystemExit(f"{exc}\n(a row of another controller comes from its own module: pass --row-module, "
                                 "or --rows R2)") from None
            from .policy_rows import PolicyRowError  # (imported already if it raised one)

            if isinstance(exc, PolicyRowError):  # the policy rows' R3 or R4 could not be built
                raise SystemExit(f"{exc}\n(the R3 and R4 rows are the policy rows' (designlab.stage1.policy_rows): "
                                 "run them in the PPO venv with the training run's checkpoints, STAGE1_POLICY_RUNS, "
                                 "or sweep --rows R2)") from None
            raise
    return 0


def _reporting(a) -> int:
    """``report`` (plan section 8's tables, ``report``)."""
    from . import report

    doc = report.write(Path(a.out), Path(a.dest) if a.dest else None, tuple(a.body) if a.body else None)
    print(f"{doc['paths']['markdown']} and {doc['paths']['json']} written ({', '.join(doc['bodies'])})", flush=True)
    return 0


def main(argv=None) -> int:
    from . import gates, plant, tune

    ap = argparse.ArgumentParser(prog="python -m designlab.stage1", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    tp = sub.add_parser("train", help="plan section 3 step 7: PPO of R3 (the residual on R2) or R4 (policy only)")
    tp.add_argument("--body", choices=sorted(BODIES), required=True)
    tp.add_argument("--seed", type=int, required=True, help="PPO seed (plan section 6: 42, 43)")
    tp.add_argument("--mode", choices=("R3", "R4"), default="R3")
    tp.add_argument("--steps", type=int, default=None,
                    help="env steps, rounded up to whole rollouts (default: train.BUDGET_ASKED, train.BUDGET_RULE)")
    tp.add_argument("--resume", action="store_true", help="continue from the newest checkpoint")
    tp.add_argument("--n-envs", type=int, default=None, help="envs per job (default: design 2's 4)")
    tp.add_argument("--vec", choices=("dummy", "subproc"), default="dummy",
                    help="the envs in this process (default: one env process per job, the plan's per-job rate) or one "
                         "process each")
    tp.add_argument("--checkpoint-every", type=int, default=250_000)
    tp.add_argument("--no-g6", action="store_true", help="skip the G6 checks")
    tp.add_argument("--log-std-init", type=float, default=None,
                    help="initial action noise (default: policy.LOG_STD_INIT, measured: policy.LOG_STD_RULE)")
    tp.add_argument("--allow-untuned", action="store_true",
                    help="train R3 although tune.json is missing or no seed passes G5 (env.GAINS_RULE; flagged)")
    tp.add_argument("--allow-source-change", action="store_true",
                    help="resume although the Stage 1 sources differ from the run's first session (flagged)")
    tp.add_argument("--fix-task-weights", default=None,
                    help="G6b's one fix (train.FIX_RULE): comma-separated task weights over env.TASKS, from this resume on")
    tp.add_argument("--fix-reason", default=None, help="why the fix (recorded with it)")
    tp.add_argument("--tune-json", default=None, help="the G5 record R3's gains come from (default: <out>/<body>/"
                                                      "tune.json)")
    tp.add_argument("--out", default=str(tune.RUN), help="run directory (default: stage1/runs/2026-11-08)")
    jp = sub.add_parser("judge", help="the judged checkpoint (train.JUDGED_RULE) into every run.json of a mode")
    jp.add_argument("--mode", choices=("R3", "R4"), default="R3")
    jp.add_argument("--budget", type=int, default=5_000_000)
    jp.add_argument("--final", action="store_true", help="the stage's cap stopped training: the judgement is final")
    jp.add_argument("--out", default=str(tune.RUN), help="run directory (default: stage1/runs/2026-11-08)")
    npr = sub.add_parser("noise", help="R3's survival under the initial action noise (policy.LOG_STD_RULE)")
    npr.add_argument("--body", choices=sorted(BODIES), default="75-M-H-10")
    npr.add_argument("--tune-json", default=None, help="the G5 record (default: <out>/<body>/tune.json)")
    npr.add_argument("--out", default=str(tune.RUN), help="run directory (default: stage1/runs/2026-11-08)")
    ep = sub.add_parser("eval", help="plan section 3 step 8: the behaviour panels of R0, R1 and R2, and G7")
    ep.add_argument("--body", action="append", choices=sorted(BODIES), required=True)
    ep.add_argument("--rows", default="R0,R1,R2/42,R2/43", help="comma-separated rows (default: R0,R1,R2/42,R2/43)")
    ep.add_argument("--behaviors", default=None, help="comma-separated behaviours (default: all of stand, walk, turn, "
                                                      "start_stop, push, carpet)")
    ep.add_argument("--conditions", default=None, help="comma-separated conditions, e.g. walk/nominal (default: all)")
    ep.add_argument("--workers", type=int, default=1, help="episode processes (default 1)")
    ep.add_argument("--attribute", action="store_true", help="decision 20's fall attribution of the R2 rows' falls")
    ep.add_argument("--dry-run", action="store_true", help="list the episodes and estimate the core-hours; run none")
    ep.add_argument("--no-measure", action="store_true", help="dry run without measuring the cost per simulated s")
    ep.add_argument("--episodes", type=int, default=20,
                    help="seeds per condition from 3042 (default 20; fewer is a smoke check, not judged)")
    ep.add_argument("--row-module", action="append", default=None,
                    help="a module to import first that adds rows (panels.register_rows), e.g. the R3 and R4 rows'")
    ep.add_argument("--no-traces", action="store_true", help="write no 50 Hz episode traces (panels.TRACE_RULE)")
    ep.add_argument("--inputs", default=None, help="run directory holding <body>/refs and transitions (default: --out)")
    ep.add_argument("--out", required=True, help="run directory, e.g. stage1/runs/2026-11-08")
    sp = sub.add_parser("sweep", help="plan section 5: the margin ledger (each margin bisected to failure)")
    sp.add_argument("--body", action="append", choices=sorted(BODIES), required=True)
    sp.add_argument("--rows", default="R2,R3/42",
                    help="comma-separated rows (default: R2,R3/42; R2 is the body-level G5 selection's CMA-ES seed, "
                         "R3/42 the policy rows' row: the PPO venv)")
    sp.add_argument("--margins", default=None, help="comma-separated margins (default: all of grip, push, stop, "
                                                    "drive_speed, drive_torque, kp, latency, mass, pile_damping, "
                                                    "voltage; model_doubts names kp, latency, mass and pile_damping)")
    sp.add_argument("--workers", type=int, default=1, help="episode processes (default 1)")
    sp.add_argument("--dry-run", action="store_true", help="list the searches and estimate the core-hours; run none")
    sp.add_argument("--no-measure", action="store_true", help="dry run without measuring the cost per simulated s")
    sp.add_argument("--row-module", action="append", default=None,
                    help="a module to import first that adds rows (panels.register_rows); the policy rows' R3 and R4 "
                         "need none (ledger.register_policy_rows)")
    sp.add_argument("--cut-seconds", type=float, default=None,
                    help="cut every condition to this many simulated seconds (a smoke check; its keys differ)")
    sp.add_argument("--halvings", type=int, default=7, help="halvings per search (default 7, the plan's; fewer is a "
                                                            "smoke check)")
    sp.add_argument("--inputs", default=None, help="run directory holding <body>/refs and transitions (default: --out)")
    sp.add_argument("--out", required=True, help="run directory, e.g. stage1/runs/2026-11-08")
    rp = sub.add_parser("report", help="plan section 8: the machine-made tables from whatever records exist")
    rp.add_argument("--out", required=True, help="the run directory to read, e.g. stage1/runs/2026-11-08")
    rp.add_argument("--dest", default=None, help="where STAGE1_TABLES.md and stage1_tables.json go (default: --out)")
    rp.add_argument("--body", action="append", choices=sorted(BODIES), default=None,
                    help="a body to report (default: every body)")
    for name in ("tune", "g5"):
        p = sub.add_parser(name)
        p.add_argument("--body", choices=sorted(BODIES), required=True)
        p.add_argument("--out", default=str(tune.RUN), help="run directory (default: stage1/runs/2026-11-08)")
        p.add_argument("--inputs", default=None, help="run directory holding <body>/refs and transitions (default: "
                                                      "--out)")
        p.add_argument("--workers", type=int, default=1, help="evaluation processes (default 1)")
        if name == "tune":
            p.add_argument("--seed", type=int, required=True, help=f"CMA-ES seed (plan section 6: {tune.SEEDS})")
            p.add_argument("--evals", type=int, default=tune.BUDGET,
                           help=f"evaluation budget, whole generations of {tune.POPSIZE} (default {tune.BUDGET})")
            p.add_argument("--resume", action="store_true", help="continue the checkpointed run")
        else:
            p.add_argument("--seeds", default=",".join(str(s) for s in tune.SEEDS))
            p.add_argument("--episodes", type=int, default=tune.G5_RUNS,
                           help=f"validation episodes (default {tune.G5_RUNS}; fewer is a smoke check, no verdict)")
            p.add_argument("--partial", action="store_true", help="also select from unfinished runs (smoke check)")
    for name in ("prepare", "gates", *LATER):
        p = sub.add_parser(name)
        p.add_argument("--body", action="append", choices=sorted(BODIES), required=name != "all")
        p.add_argument("--out", required=True, help="run directory, e.g. stage1/runs/<run>")
        p.add_argument("--seed", type=int, default=None, help="tuning or training seed (later steps)")
        if name in ("prepare", "gates"):
            p.add_argument("--ankle-gains", choices=plant.ANKLE_GAINS, default=plant.ANKLE_GAINS[0],
                           help="ankle hold-servo gain: joint space (default, STAGE1_PLAN.md decision 11) or per "
                                "motor (for comparison)")
            p.add_argument("--ankle-armature", choices=plant.ANKLE_ARMATURE, default=plant.ANKLE_ARMATURE[0],
                           help="ankle armature: the linkage's, on the motor tendons (default, decision 12) or as "
                                "screened (for comparison)")
            p.add_argument("--g2-rule", choices=gates.RULES, default=gates.RULES[0],
                           help="G2 replay criterion: the median of the 20-start panel (default, decision 13) or the "
                                "one unperturbed run (the plan's original wording)")
        if name == "prepare":
            p.add_argument("--no-gates", action="store_true", help="skip the gates")
            p.add_argument("--refs", default=None, help="comma-separated reference names (default: all)")
            p.add_argument("--parts", default="plant,refs,transitions",
                           help="comma-separated products to make: plant, refs, transitions (default), and stops "
                                "(re-read the saved stops, decision 18)")
            p.add_argument("--transitions", default=None, help="comma-separated transition names (default: all)")
        if name == "gates":
            p.add_argument("--only", choices=GATES, default=None)
        if name == "all":
            p.add_argument("--cap-hours", type=float, default=None)
    a = ap.parse_args(argv)
    if a.cmd in ("tune", "g5"):
        return _tuning(a)
    if a.cmd in ("train", "judge", "noise"):
        return _training(a)
    if a.cmd == "eval":
        return _evaluation(a)
    if a.cmd == "sweep":
        return _sweeping(a)
    if a.cmd == "report":
        return _reporting(a)
    if a.cmd in LATER:
        raise NotImplementedError(f"'{a.cmd}' is {LATER[a.cmd]}, not implemented yet; this package implements plan "
                                  "section 3 steps 2-7 (prepare: plant, references and transitions; gates G1-G4; "
                                  "tune and g5; train), step 8's behaviour panels (eval) and margin ledger (sweep), "
                                  "and step 9's tables (report)")
    plant_kw = dict(ankle_gains=a.ankle_gains, ankle_armature=a.ankle_armature)
    status = 0
    for body in a.body:
        out = Path(a.out) / body
        t0 = time.perf_counter()
        if a.cmd == "prepare":
            from . import refs, transitions

            parts = set(a.parts.split(","))
            if parts - {"plant", "refs", "transitions", "stops"}:
                raise SystemExit(f"--parts: unknown {sorted(parts - {'plant', 'refs', 'transitions', 'stops'})}")
            if "plant" in parts:
                _write(out / "plant.json", plant_doc(body, plant_kw))
                print(f"{body}: plant.json written", flush=True)
            if "refs" in parts:
                names = a.refs.split(",") if a.refs else None
                doc = refs.prepare(body, out, names)
                print(f"{body}: {len(doc['references'])} references, records match: {doc['matches_records']}", flush=True)
            if "transitions" in parts:
                names = a.transitions.split(",") if a.transitions else None
                doc = transitions.prepare(body, out, names)
                print(f"{body}: {len(doc['transitions'])} transitions", flush=True)
            if "stops" in parts:
                doc = transitions.reread_stops(out)
                print(f"{body}: stops re-read (decision 18): " + ", ".join(
                    f"{n} {e['stop']['forward_stopped_strides']}" for n, e in doc["transitions"].items()
                    if e["kind"] == "stop"), flush=True)
            if a.no_gates:
                continue
        doc = run_gates(body, out, getattr(a, "only", None), plant_kw, a.g2_rule)
        print(f"{body} ({time.perf_counter() - t0:.0f} s)\n{summary(doc)}", flush=True)
        status |= 0 if doc["pass"] else 1
    return status


if __name__ == "__main__":
    sys.exit(main())
