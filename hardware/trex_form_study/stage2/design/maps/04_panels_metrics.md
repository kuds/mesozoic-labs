# Map 04: panels, metrics, rows and record keys

Study directory: `/home/user/mesozoic-labs/hardware/trex_form_study` (all paths below are relative to it).
Read on 2026-11-14; no code file was modified in the tree while I read (four `.md` files were being edited by another
session); the last commit touching `panels.py` is `0e41e91`. Nothing was run except short inspections (no episodes,
no pytest). "Not found" means I looked and it is not there.

Files read in full: `designlab/stage1/panels.py` (2260 lines), `designlab/stage1/metrics.py` (451),
`designlab/stage1/__main__.py` (747), `tests/test_stage1_panels.py`, `tests/test_stage1_bodies.py`.
Read in part: `speedfit.py` (1-570, 1075-1324), `tune.py` (150-670, 868-1070), `control.py` (748-800, 950-1135,
1171-1200, 1386-1431, 1950-2000, 2215-2494), `transitions.py` (112-152, 755-835), `ledger.py` (326-352, 656-975),
`env.py` (60-108, 160-344), `plant.py` (255-300), `gates.py` (796-935), `STAGE1_PLAN.md` sections 4-7 and decisions
14, 18, 20, 30, 31, 33.

---

## 0. Three facts to know before anything else

1. **Every committed panel record already reads "made by other code".** The panels' source hash today is
   `6a316fc82bf5...`; the Stage 1 records (`stage1/runs/2026-11-08/*/eval/**`) hold `7516b9271f1c...`, the speed
   round's (`stage1/runs/2026-11-10-speed/*/eval/**`) hold `733f2b8093f4...`. (`designlab/stage2/look.py:107-110`
   says the same of its branch: it edited `designlab/gait.py`, which the panel hash covers.)
2. **`eval` overwrites a record whose key differs.** `panels.evaluate` reuses a record only if `old["key"] == key`
   (`panels.py:1862-1870`); otherwise it runs the 20 episodes and writes over the file (`panels.py:1871`, `1888`).
   It also rewrites `<out>/<body>/eval/summary.json` and `g7.json` on every call, even when everything was skipped
   (`panels.py:1921`, `1981`, `1988`). So an `eval --out stage1/runs/2026-11-08` (or `2026-11-10-speed`) today would
   rerun and replace cited records. New panels need a new `--out`. `--dry-run` writes nothing.
3. **No traces are on this machine.** Records name each episode's trace by path and sha256, but
   `find stage1/runs -path '*traces*' -name '*.npz'` finds 0 files (git-ignored: `stage1/.gitignore` lists `*.npz`
   and `traces/`). The reference arrays were still being restored while I read: at my last look
   `stage1/runs/2026-11-08/` had `refs/` (10 files) for all three probe bodies but `transitions/` (18 files) only for
   60-N-T-10. New readings cannot be computed from earlier episodes here.

---

## 1. CLI entry points (`designlab/stage1/__main__.py`)

Run from the study directory: `python -m designlab.stage1 <cmd> ...`. Dispatch is in `main` (`__main__.py:694-705`).

### eval (parser `__main__.py:562-581`, handler `_evaluation` `:424-442`)

| Argument | Default | Meaning |
|---|---|---|
| `--body` (repeatable, required) | | must be in `stage1.BODIES` (`choices=sorted(BODIES)`) |
| `--rows` | `R0,R1,R2/42,R2/43` | comma-separated row names |
| `--behaviors` | all | subset of `panels.BEHAVIORS` |
| `--conditions` | all | keys (`walk/nominal`) or names (`nominal`) |
| `--workers` | 1 | episode processes |
| `--attribute` | off | decision 20's fall attribution for attributed rows |
| `--dry-run`, `--no-measure` | off | list episodes and cost; `--no-measure` skips timing |
| `--episodes` | 20 | seeds from 3042; fewer is a smoke check, not judged |
| `--row-module` (repeatable) | none | module imported first; it registers rows |
| `--no-traces` | off | write no 50 Hz traces |
| `--allow-rebuilt-models` | off | run without `round2/refine/<case>/models` (keys then differ) |
| `--inputs` | `--out` | run directory holding `<body>/refs`, `<body>/transitions` |
| `--out` (required) | | run directory records are written under |

Handler order: `_check_models` (refuses a checkout without the round-two model files unless dry run or
`--allow-rebuilt-models`, `:411-421`); `importlib.import_module(mod)` for each `--row-module` (`:429-430`); then per
body `panels.dry_run(...)` or `panels.evaluate(body, Path(out), rows, behaviors, names, workers, attribute, episodes,
inputs, traces=not no_traces, log=...)` and `print(panels.summary_text(doc))`.

### sweep (parser `:582-604`, handler `_sweeping` `:445-473`)

`--body` (repeatable, required), `--rows` (default `R2,R3`), `--margins` (default all of grip, push, stop,
drive_speed, drive_torque, kp, latency, mass, pile_damping, voltage), `--workers`, `--dry-run`, `--no-measure`,
`--row-module`, `--cut-seconds`, `--halvings` (default 7), `--inputs`, `--allow-rebuilt-models`, `--out` (required).
Calls `ledger.sweep(body, out, rows, names, workers, inputs, cut_seconds, halvings, log=)` or `ledger.dry_run(...)`.
Records: `<out>/<body>/sweeps/<row>/<margin>.json`.

### report (parser `:605-609`, handler `_reporting` `:476-482`)

`--out` (required), `--dest` (default `--out`), `--body` (repeatable). Calls `report.write(out, dest, bodies)` and
writes `<dest>/STAGE1_TABLES.md` and `<dest>/stage1_tables.json`. It flags a record whose `key.sources` differs from
the hash now as "other panel sources" (`report.py:265-266`).

### speed round (parsers `:610-649`, handler `_speed` `:485-519`)

`speedfit --body B [--seed 42] [--evals 360] [--workers 1] [--start-from BODY] [--resume] [--stage1 DIR] [--inputs DIR] --out DIR`;
`speedg5 --body B [--seeds 42] [--workers 1] [--at-generation G] [--partial] [--episodes 20] [--inputs DIR] --out DIR`;
`speedcost --body B [--start-from BODY] [--stage1 DIR] [--inputs DIR] [--record PATH] --out DIR`;
`speedreport --out DIR [--stage1 DIR] [--body B] [--dest DIR]`.
The speed round's validation rows ran as (`stage1/runs/2026-11-10-speed/run_speed_round.sh:94-95`):

```
python -m designlab.stage1 eval --body "$b" --row-module designlab.stage1.speedfit --rows "$(rows "$b")" \
    --behaviors walk,turn --attribute --workers 1 $(inputs "$b") --out "$OUT"
```

---

## 2. Rows

### 2.1 The row interface (`panels.Row`, `panels.py:689-742`)

A row must provide:

- `name: str` (`"R0"`, `"R2/42"`, `"R2s/42@10ms"`); `hard_only: bool`.
- `runs(cond) -> bool` (default: `not hard_only or cond.floor == "hard"`).
- `gated(cond) -> bool` (default `gated_for(self.name, cond)`).
- `describe() -> dict`: JSON description; it is the record's `controller` and enters the key as `key.row`.
- `sources() -> dict`: `{path: sha256}` of modules outside the panels' sources that the row runs; enters the key as
  `key.row_sources`.
- `spec() -> dict`: picklable; its `"cls"` is the dotted class path (the docstring says "class", the code key is
  `cls`, `panels.py:831`, `857`). `row_from_spec(s)` (`:855-858`) does
  `getattr(importlib.import_module(mod), cls).from_spec(s)` in the worker.
- `episode(ctx, cond, seed, relax=()) -> (built, out, rec)`: one episode. `out` is `control.run`'s outcome passed
  through `tune.warned_fall`; `rec` a filled `metrics.Recorder`.
- `run(ctx, cond, seed, relax=(), trace_root=None) -> dict` (`:735-742`), the episode record:

```python
b, out, rec = self.episode(ctx, cond, int(seed), tuple(relax))
e = judge(b, out, rec, int(seed), ctx.lg if b.legs else None)
if trace_root is not None:
    e["trace"] = write_trace(trace_root, trace_path(self.name, cond, int(seed)), rec)
e.update(relaxed=list(relax), wall_s=time.perf_counter() - t0)
```

`run` is the only place the live `Recorder` and the judgement meet. `judge`, `write_trace` and `trace_path` are
module globals looked up at call time.

### 2.2 `TrackerRow` (`panels.py:804-852`)

```python
TrackerRow(name, mode, gains=None, gains_info=None, terms=control.TERMS, ankle_cap=control.ANKLE_CAP,
           hard_only=False, policy=None, rate=RATE, controller="")
```

`describe()` keys (`:816-820`): `row, controller, tracker_mode, terms, ankle_cap, inward_bound, rate, gains` (dict of
the 14 `control.GAIN_NAMES` or None), `gains_info, hard_only, policy`.
`episode` (`:841-852`):

```python
b = ctx.built(cond); p = b.p
tk = control.Tracker(p, b.track, self.mode, gains=self.gains, terms=self.terms, ankle_cap=self.ankle_cap, rate=self.rate)
rec = b.recorder()
hook = rec if self.policy is None else _Acting(rec, self.policy, b, seed)
s = episode_seeds(cond, seed)
out = control.run(tk, b.seconds, start="stance", seed=s["start"], noise_seed=s["noise"], recorder=hook,
                  joint_noise=tune.START_JOINT_NOISE_RAD, root_velocity_noise=tune.START_ROOT_VELOCITY_NOISE_M_S,
                  push=b.push(seed), relax=tuple(relax))
return b, tune.warned_fall(out, tune.sim_warnings(p.d)), rec
```

`Policy` (`:751-767`): `every = env.ENV_SUBSTEPS` (10), `begin(tk, built, seed)`, `act(tk, k)` after physics step k
for `(k + 1) % every == 0`; it sets `tk.residual` (joint offsets, rad, plant order, or None) and `tk.rate_factor`.
`ZeroResidual` (`:770-778`) sets `None` and `1.0`: the row is then its tracker row bit for bit.

### 2.3 The built-in rows (`tracker_rows`, `panels.py:874-897`; `ROWS_RULE` `:165-174`)

`ROWS = ("R0", "R1", "R2/42", "R2/43")` (`:80`).

| Row | Construction | Gains |
|---|---|---|
| `R0` | `TrackerRow("R0", "R1", None, {...}, terms=(), ankle_cap=(), hard_only=True)` | none. Mode is `"R1"` with no feedback term, because `control.Tracker` refuses mode `"R0"` at a rate other than 1 (`control.py:1180-1181`) |
| `R1` | `TrackerRow("R1", "R1", g, info, hard_only=True)` | `env.tuned_gains(run_dir, body)`: the body-level G5 selection |
| `R2/<s>` | `TrackerRow(name, "R2", g, info)` | `r2_gains(run_dir, s)` |
| other | `ROW_FACTORIES[controller_of(name)](body, Path(run_dir), name)` | the factory's |

**How R2/42 and R2/43 gains are loaded** (`r2_gains`, `panels.py:677-686`): reads `<run_dir>/tune.json`, takes
`doc["seeds"][str(seed)]`, raises `ValueError` if it has no `"selected"`, and builds the vector
`[float(v["selected_gains"][n]) for n in control.GAIN_NAMES]`. `gains_info` = `rule, file` (absolute path string),
`sha256` (of tune.json), `cma_seed, candidate, g5_pass, g5_label`.

- `run_dir` is `<inputs or out>/<body>` (`panels.py:1824`, `1829`). With `--inputs` pointing at a new reference
  directory, `tune.json` must be in that directory too, or the row must come from a factory that reads it elsewhere.
- `tune.json` exists only for the three probe bodies, under `stage1/runs/2026-11-08/`:
  60-N-T-10 `seed42_gen20` / `seed43_gen40`; 60-N-H-10 `seed42_gen60` / `seed43_gen60`; 75-M-H-10 `seed42_gen60` /
  `seed43_gen40`; all `pass_` true. For `-K` and `-X` bodies: not found (the speed round reads
  `speedfit/start.json` and `speedg5.json` instead).
- `name.startswith("R2/")` is tested before `ROW_FACTORIES` and parsed with `int(name.split("/", 1)[1])`
  (`panels.py:887-888`), so `R2/42@4ms` raises. `register_rows` refuses the controllers `R0`, `R1`, `R2`
  (`:869-870`). A delayed variant of R2/42 therefore needs another controller prefix.

### 2.4 Rows from another module (`--row-module`)

- `register_rows(controller, factory)` (`panels.py:865-871`) stores `factory(body, run_dir, name) -> Row` in
  `ROW_FACTORIES` (`:862`). The controller is the name before the first `/` (`controller_of`, `:494-496`).
- `eval --row-module M` imports `M` in the main process before `--rows` is resolved (`__main__.py:429-431`). The
  module registers at import.
- Worker processes are **spawned** when `--workers > 1` (`tune.py:966`, `mp.get_context("spawn")`); with
  `--workers 1` the evaluator is built in-process (`tune.py:961-965`). A spawned worker imports the row's module
  only inside `row_from_spec` (`panels.py:1777`), which runs **after** `PanelWorker.__init__` has read the inputs
  with `panel_files()` (`panels.py:1771`).
- Already there for new rows: `VERDICT_ROWS` names R3 (`:82`), `ATTRIBUTED = ("R2", "R3")` (`:86`).
- Existing row modules: `designlab.stage1.speedfit` (section 9), `designlab.stage1.policy_rows` (R3, R4).

---

## 3. Behaviours and conditions

### 3.1 `Condition` (`panels.py:418-450`), a frozen dataclass

```python
behavior: str; name: str; gated: bool; judge: str
floor: str = "hard"; friction: float | None = None; gait: str = "nominal"
case: str | None = None; direction: str | None = None
seconds: float | None = None      # a stand's length; for any other condition, a cut (smoke checks)
push_at: float | None = None; push_spread: float | None = None
recovery_s: float = RECOVERY_S; hold_s: float = PUSH_HOLD_S
```

- `key` = `f"{behavior}/{name}"`; `kind` = `{"carpet": "walk"}.get(judge, judge)` (tune's episode kind and the seed
  slot); `describe()` = `dataclasses.asdict(self)` plus `key`.
- It is the dict key of `Context._built`, part of each pickled job `((row_name, cond, relax), seed)`, and enters the
  record key whole (`key.condition`).
- `judge` is one of: `stand, walk, walking_turn, in_place_turn, start_stop, push_stand, push_walk, carpet`.

### 3.2 The 28 conditions (`conditions()`, `panels.py:453-474`; `CONDITIONS_RULE` `:148-159`)

`BEHAVIORS = ("stand", "walk", "turn", "start_stop", "push", "carpet")` (`:89`).

| Key | judge | floor | friction | gait / case | gated |
|---|---|---|---|---|---|
| `stand/{hard,robot_floor_carpet,low}` | stand | same as name | None | | yes |
| `walk/nominal`, `walk/brisk`, `walk/fr05` | walk | hard | None | gait = name | yes |
| `walk/nominal_mu0.54` | walk | hard | 0.54 | nominal | yes |
| `walk/brisk_mu0.54`, `walk/fr05_mu0.54` | walk | hard | 0.54 | brisk, fr05 | reported |
| `turn/walking_0.35_{left,right}` | walking_turn | hard | None | case `walk_0.35` | yes |
| `turn/in_place_0.50_{left,right}` | in_place_turn | hard | None | case `in_place_0.50` | yes |
| `turn/walking_0.50_{left,right}` | walking_turn | hard | None | case `walk_0.50` | reported |
| `start_stop/twice` | start_stop | hard | None | | yes |
| `push/{standing,walking}_{forward,backward,left,right}` | push_stand / push_walk | hard | None | | yes |
| `carpet/{robot_floor_carpet,low}` | carpet | same as name | 0.54 | | yes (reported for R0-R2) |
| `carpet/{medium,high}` | carpet | same as name | 0.54 | | reported |

`select(conds, behaviors, names)` (`:477-491`) raises `ValueError` for a behaviour outside `BEHAVIORS` or an unknown
key. `gated_for(row, cond)` (`:499-502`): `cond.gated and not (cond.behavior == "carpet" and controller_of(row) in
("R0", "R1", "R2"))` (decision 23).

### 3.3 Track plans (`template`, `panels.py:519-547`)

Plan items are `("ref", name, strides)`, `("transition", name)`, `("hold", seconds)`, as `control.build_track` takes
them (`control.py:1099-1132`). `None` lengths are filled by `Planner`.

| judge | Template |
|---|---|
| stand, push_stand | `[("hold", None), ("transition", "start")]` |
| walk, gait nominal | `tune.episode_plan("walk", None)` = start, `("ref", "nominal", 7)`, stop |
| walk, gait brisk | start, `("ref", "nominal", 1)`, `nominal_to_brisk`, `("ref", "brisk", None)` |
| walk, gait fr05 | start, nominal x1, `nominal_to_brisk`, brisk x1, `brisk_to_fr05`, `("ref", "fr05", None)` |
| walk, any other gait | `raise ValueError(f"walk gait {cond.gait!r}")` (`:532`) |
| walking_turn | start, `enter_{case}_{dir}`, `("ref", "{case}_{dir}", tune.TURN_STRIDES)` (5), `exit_{case}_{dir}`, stop |
| in_place_turn | `tune.episode_plan("in_place_turn", dir)` = `start_in_place_0.50_{dir}`, ref x3, `stop_in_place_0.50_{dir}` |
| start_stop | (start, nominal x1, stop, `("hold", 2.0)`) x 2 |
| push_walk | start, `("ref", "nominal", None)` |
| carpet | `env.TASKS["carpet_walk"][0]` = start, `("ref", "carpet", 7)`, stop |

`files_of(cond)` (`:550-558`) = `refs/nominal.npz` (always) plus `refs/<name>.npz` and `transitions/<name>.npz` of
the template. `panel_files(conds=None)` (`:561-563`) is the union over **all** of `conditions()` when called without
arguments. Today that is 26 files:

```
refs: brisk carpet fr05 in_place_0.50_left in_place_0.50_right nominal walk_0.35_left walk_0.35_right
      walk_0.50_left walk_0.50_right
transitions: brisk_to_fr05 enter_walk_0.35_{left,right} enter_walk_0.50_{left,right} exit_walk_0.35_{left,right}
      exit_walk_0.50_{left,right} nominal_to_brisk start start_in_place_0.50_{left,right} stop
      stop_in_place_0.50_{left,right}
```

### 3.4 `Planner` (`panels.py:566-660`)

`Planner(run_dir, refs_doc, tr_doc)`; `load(item)` loads `Reference` / `Transition` from `run_dir/refs|transitions`;
`segment_times(plan) -> [(t0, duration)]` on the reference clock, following `control.build_track`;
`phase_time(plan, i, phase)`; `period(name)`; `resolve(cond) -> dict` with keys
`plan, seconds, steady_ref_s, starts_ref_s, push` (and `cut_from_s` when `cond.seconds` cuts a non-stand).

- stand / push_stand: hold of `seconds * CLOCK_MAX + 1.0` s (`:632`); `CLOCK_MAX = 1.05 * 1.05 * 1.25 = 1.378125`
  (`:122`).
- walk with gait other than nominal (`:634-642`): K strides of the target gait, the fewest with
  `need = (steady + 0.5 P) / (1 - 0.05) + 10.0 * RATE`; `steady` is phase 1.0 of the last transition.
- push_walk (`:643-653`): uses `self.period("nominal")`.
- start_stop (`:655-658`): `duty = tr_doc["start"]["gaits"]["b"]["duty"]`; `starts_ref_s` are the phase times of
  `stop_command_phase(duty)` in every item equal to `("transition", "start")`.
- `seconds = t_end / RATE` (`:660`): the simulated length is the reference length over 1.05.

### 3.5 `Context` and `Built` (`panels.py:903-1020`)

`Context(body, run_dir, docs)`, one per worker. `docs` is `tune.read_inputs`' second value
(`{"refs.json": {...}, "transitions.json": {...}}`).

- `plant(floor, friction)` (`:979-988`): `plant.build(body, floor)`, cached per `(floor, friction)`; raises
  `PlantError` if the stance does not settle; **`if friction is not None: control.set_friction(p, friction)`**.
- `built(cond)` (`:990-1020`): `omega = control.omega0(p, Reference.load(run_dir/"refs"/"nominal.npz").q_home)`.
  For `cond.judge in TUNE_JUDGED` (`walk, walking_turn, in_place_turn, carpet`): if the plan is tune's own
  (`plan_is_tunes`, `:663-667`) `tune.build_spec(p, run_dir, kind, direction, omega, refs_doc, tr_doc, RATE)`, else
  `tune.build_spec(..., RATE, plan=plan, case=cond.case, gait=cond.gait)`; track and required margins from the spec.
  Otherwise `control.build_track(p, run_dir, plan, omega)` and `required = 0.10 rad` on every joint.
- `Built` fields: `cond, p, plan, track, spec, seconds, resolved, lo, hi, required, H, mass, speed_cmd, stride_s,
  push_bound_m_s, home_yaw`. `speed_cmd = refs_doc["nominal"]["speed_m_s"] * RATE`,
  `stride_s = refs_doc["nominal"]["period_s"] / RATE` (`:1014-1017`), `H = p.meta["H_m"]`,
  `mass = m.body_subtreemass[p.trunk]`, `push_bound_m_s = env.PUSH_BOUND_M_S[body]`, `home_yaw = control._yaw(p._R_home)`.
- `Built.legs` (`:926-927`): True for `walking_turn`, `in_place_turn`. `Built.recorder()` (`:929-931`):
  `metrics.Recorder(legs_every=tune.LEGS_EVERY if legs else 0, foot_tables=env.foot_tables if floor in carpet.PILES else None)`.
- `Built.push(seed)` -> `(t0, 0.1, fx, fy)` or None.

---

## 4. The episode rule

`EPISODE_RULE` (`panels.py:175-186`), in the code's words: "seeds 3042-3061 for every condition and row (pushes
paired by seed); episode seed s of a condition starts from the plant's settled stance with tune's start noise ...
the nominal plant (no randomisation, no latency), the floor and friction of the condition, commanded 5 % above
(tune.COMMAND_RATE, control.COMMAND_RATE_RULE)".

- **Seeds.** `SEEDS = tuple(range(3042, 3062))` (`:75`), `RUNS = 20`. `--episodes n` takes `SEEDS[:n]` (`:1825`).
- **Derived seeds.** `episode_seeds(cond, seed)` (`:505-508`): `slot = SLOTS.get(cond.kind, 0)` with
  `SLOTS = {"walk": 0, "carpet": 0, "walking_turn": 1, "in_place_turn": 2}` (`:124`); then
  `tune.derived_seed(seed, slot, purpose)` for `PURPOSES = {"start": 0, "noise": 1, "push": 2}` (`:125`).
  `derived_seed` = `np.random.SeedSequence([seed, episode, purpose]).generate_state(1)[0]` (`tune.py:263-265`).
  `speedfit.LATENCY_PURPOSE = 3` (`speedfit.py:108`) already uses purpose 3. Example: seed 3042, walk ->
  `start 460813939, noise 3401712831, push 2237509858`.
- **Start noise.** `joint_noise = 0.01 rad`, `root_velocity_noise = 0.02 m/s` (`tune.py:113-114`), from the settled
  stance (`start="stance"`).
- **"Commanded 5 % above".** `RATE = tune.COMMAND_RATE = 1.0 + COMMAND_MARGIN = 1.05` (`panels.py:79`,
  `tune.py:111-112`). It is the tracker's clock rate, not a different reference: `control.COMMAND_RATE_RULE`
  (`control.py:1956-1960`), "a command of r times the reference's speed plays the track r times faster than real
  time ... cadence, speed and yaw rate times r, stride and footprints unchanged". Success compares against the
  reference's own speed (`spec.speed_target`), the command is `speed_target * 1.05`. Consequence for any new reading
  stated in seconds: reference time runs 1.05 x simulated time (plus the phase-rate term's +-5 %);
  `Recorder.sim_time(t_ref)` (`metrics.py:192-197`) converts.
- **Servo delay.** The panels' own rows have none. Delay exists only in `speedfit.DelayedRow`, which wraps
  `TrackerRow.episode` in `servo_delay(b.p, self.latency_steps)` (`speedfit.py:1110-1113`). One step is one physics
  step of 2 ms; row names carry even milliseconds (`ms_steps`, `speedfit.py:1116-1121`, 0-20 ms). The model
  (`speedfit.py:324-361`): a `deque(maxlen=k + 1)` of `p.d.ctrl` copies; `p.command` and `p.reset` are replaced on
  the instance for the block and restored after; `k = 0` changes nothing. `ledger._latency` (`ledger.py:708-735`) is
  the same queue; `tests/test_stage1_speedfit.py::test_the_rows_delay_reaches_the_servos` pins the two as equal bit
  for bit.
- **Friction.** Per condition only: `Condition.friction` (None = the floor as compiled, mu 0.9 on hard floor; 0.54 =
  `MU_LOW`, `:92`). `control.set_friction` (`control.py:1988-1992`) sets `geom_friction[:, 0]` of every colliding
  geom on that condition's cached plant. A row has no friction field. The ledger's friction variant shows the
  save / set / restore pattern for one episode (`ledger.py:806-813`).
- **Fall.** `plant.Plant.fall_cause`; a MuJoCo warning during the episode counts as a fall at its start
  (`tune.warned_fall`).
- **Failed episode.** `failed_episode(cond, seed, reason, relax)` (`:1367-1373`): `fell True`, `error` set,
  `measures {}`; counted apart, run again on the next eval of the same key.

---

## 5. Judgement

### 5.1 Success terms per judge (`terms_of`, `panels.py:1031-1047`)

| judge | Terms |
|---|---|
| stand | `survived, tilt, drive_sustained, drive_speed, joint_range` |
| walk, gait nominal | `tune.SUCCESS["walk"]`: `survived, speed, duty, no_flight, cadence, stride, grf, roll_rms, pitch_rms, roll_peak, pitch_peak, toe_clearance, slip_median, slip_worst, drive_sustained, drive_speed, joint_range` |
| walk, other gait | the same without `cadence, stride, grf` (`NOMINAL_ONLY_TERMS`, `:133`) |
| walking_turn | `survived, yaw_rate, duty, no_flight, grf, roll/pitch rms and peak, toe_clearance, slip_median, slip_worst, drive_sustained, drive_speed, joint_range, no_leg_interference` |
| in_place_turn | the walking turn's plus `drift` |
| carpet | `tune.SUCCESS["walk"]` plus `toe_target, no_toe_catch` |
| start_stop | `survived`, `start{1,2}_speed_up`, `start{1,2}_acceleration`, `stop{1,2}_stopped`, `stop{1,2}_acceleration`, limits |
| push_stand, push_walk | `survived, recovered`, limits |

Unknown judge: `ValueError` (`:1047`). Limits in `tune.py:152-167`: roll/pitch 4 deg RMS and 8 deg peak, duty
0.55-0.65, cadence 1.6-2.2 steps/s, stride 0.5-0.65 H, GRF 1.6 BW, toe 0.04 H, slip 5 mm median and 15 mm worst,
drift 0.1 m, range 0.10 rad, start within 2.0 strides, stop within 1.5 strides, acceleration 1.5 m/s^2.

### 5.2 `judge(b, out, rec, seed, lg=None)` (`panels.py:1099-1153`)

- `_base` (`:1064-1081`) gives every episode: `seed, fell, fall_time_s, fall_cause, fall_onset_s, fall_candidates,
  limits_before_onset, survival_s, planned_s, strides, worst_sustained_ratio, worst_speed_ratio,
  tilt_from_home_max_deg, joint_margin_min_rad, joint_shortfall_rad, joint_worst, estimator, ankle_cap_residual,
  captures, phase_rate_range` (and `mujoco_warnings` when any).
- TUNE_JUDGED: `res = tune.measure(b.spec, rec, out, lg, window=window)`. For a walk whose gait is not nominal,
  `window = (rec.t[-1] - 10.0, rec.t[-1])` when it did not fall (`:1109-1111`), and `start`, `stop_reading`, `stop`
  are set to None (`:1116-1117`). `measures` keeps `MEASURE_KEYS` (`:1095`) and `window` cut to `WINDOW_KEYS`
  (`:1090-1094`). Walk and carpet add `measures["look"]` (sway, head, tail); carpet adds `measures["carpet"]`.
- stand: `tilt = tilt_from_home_max_deg < 12` plus `_limits_ok` (`:1084-1087`).
- start_stop: `measures["start_stop"] = start_stop(b, out, rec)`.
- push: `measures["push"]`, `measures["recovery"] = recovery(...)`.
- Unknown judge: `ValueError` (`:1145`).
- A fall fails every term; its `failed_terms` is `["survived"]` alone (`:1147-1152`).

### 5.3 The verdict rule

- **Per record** (`counts`, `panels.py:1390-1400`; `PASS_RULE` `:284-285`):

```python
judged = n == RUNS and [e["seed"] for e in episodes] == list(seeds) == list(SEEDS) and errors == 0
pass_  = bool(surv == RUNS and succ >= MIN_SUCCEED) if judged else None      # 20/20 up and >= 17 succeed
```

  Returned keys: `episodes, survived, succeeded, falls, errors, survival_lcb, success_lcb, judged, pass_`.
  `lcb` (`:1379-1387`) is the one-sided 95 % Clopper-Pearson bound (20/20 -> 0.861, 17/20 -> 0.656).
- **Per controller** (`verdict`, `:1442-1465`; `VERDICT_RULE` `:323-331`), over the two rows of
  `VERDICT_ROWS = {"R2": ("R2/42", "R2/43"), "R3": ("R3/42", "R3/43")}` (`:82`): `shown` (both pass), `provisional`
  (one), `body-limited`, `not attributed`, `controller-limited`, `incomplete`, `not run`. No other controller has a
  verdict; `speedreport.READING_RULE` (`speedreport.py:35-44`) reads the speed rows per record instead.
- **Labels.** `success_label` (`:1403-1409`): body-limited by the `HARDWARE_TERMS` (`:130-131`) that failed in more
  than half of the surviving-but-failed episodes. `failure` (`:1412-1439`): keys `falls, fall_causes,
  fall_candidates, success, success_failures, falls_label, success_label, kind, label, untestable`.

### 5.4 Fall attribution (decision 20): the entry point

`eval --attribute` -> `evaluate(attribute=True)`; after a condition's episodes (`panels.py:1895-1913`):

1. `_attribution_needed(rec, attribute)` (`:1679-1689`): status done, `controller_of(row) in ATTRIBUTED`, falls > 0,
   no errors, no attribution yet (or one with `lifted_errors`).
2. `attribution_plan(rec)` (`:1692-1698`): candidates in `control.HW_LIMITS` order; relaxations from
   `control.RELAXES`; nothing lifted below `ATTRIBUTION_MIN_FALLS = 5` (`:135`).
3. Jobs `((row.name, cond, (r,)), s)` for each relaxation over the same seeds; they leave no trace (`:1785`).
4. `attribution_record(rec, cands, lifted, wall_s)` (`:1701-1731`) -> `gates.attribute(base, rows, None, cands)`
   (`gates.py:919`; `sign_test` `gates.py:898-915`, alpha 0.05 `gates.py:798`).

`control.run(..., relax=)` lifts a limit for the whole episode (`control.relaxed`, `control.py:2056`).
`speedfit` appends its controllers to `panels.ATTRIBUTED` at import (`speedfit.py:1216`).

---

## 6. `metrics.py`

### 6.1 `Recorder` (`metrics.py:103-206`)

`Recorder(legs_every=0, foot_tables=None)`. `control._episode` calls `begin(tk, n)`, `step(k)` after every physics
step, `update(k, loads, slip_v, Rr, yaw)` at every feedback update (`k % control.FEEDBACK_EVERY == 0`, 5 steps, 100
Hz), `end(steps, out)` (`control.py:2306-2307`, `2363-2364`, `2406-2407`, `2491-2492`). It only reads.

| Attribute | Shape | Units and meaning |
|---|---|---|
| `dt` | scalar | physics step, 0.002 s |
| `weight` | scalar | N: `body_subtreemass[trunk] * 9.81` |
| `loads` | `[steps, 2]` | N per physics step (500 Hz): each foot's vertical floor load, the contact sensor `<side>_foot_load` (`data="found force"`, index +3; `plant.py:376-377`, `metrics.py:128`, `138`) |
| `force_sq` | `[steps, nu]` | squared motor-space actuator force per physics step |
| `sustained`, `actuators` | `[nu]` | each drive's sustained rating and actuator name |
| `U` | `[rows, 31]` | per update (100 Hz), columns `COLS` |
| `Q` | `[rows, len(p.joints)]` | joint angles (rad), plant joint order |
| `poses` | list of qpos or None | every `legs_every` updates, for `legs.LegGap` |
| `rows`, `steps`, `out` | | counts and `control.run`'s outcome |

`COLS` (`metrics.py:70-72`), index = `C[name]`:

```
0 t          simulated s: the START of the physics step just taken (d.time - dt)
1 t_ref      reference-clock s of that state: tk.tau_ref - dt*rate*(1+delta)
2-3 com_x com_y      whole-robot CoM, world m        4-5 v_x v_y        CoM velocity, m/s
6-9 foot0_x foot0_y foot1_x foot1_y   foot body origins, world m (0 left, 1 right)
10-11 low0 low1      lowest point of each foot (every foot geom) above the floor plane p.z_floor, m
12-13 roll pitch     trunk about its home attitude, rad    14 yaw   trunk world yaw, rad, wrapped
15-16 slip0 slip1    loaded feet's slip speed, m/s
17-18 acc_x acc_y    IMU point acceleration, world horizontal, m/s^2
19-20 e_x e_y        true DCM error against the reference, anchor-foot frame, m
21 ref_swing         the REFERENCE's swing foot at t_ref: 0 left, 1 right, -1 none (control.py:769-774)
22 ref_u             the REFERENCE's swing progress, 0 at lift-off (control.py:777-798)
23-26 sole0_x sole0_y sole1_x sole1_y   sole centres, world m
27-28 load0 load1    N, the loads at this update        29-30 root_x root_y   trunk origin, world m
```

Helpers: `col(name)`, `t`, `sim_time(t_ref)` (None outside the trace), `contact()`, `yaw_unwrapped()`.

### 6.2 The 2 % load contact and touchdown rule

`contact()` (`metrics.py:199-203`):

```python
raw = self.loads > control.CONTACT_LOAD_FRAC * self.weight        # 0.02 (control.py:170), per physics step
k = max(1, int(round(DEBOUNCE_S / self.dt)))                      # 0.02 s -> 10 steps
return np.column_stack([debounce(raw[:, i], k) for i in (0, 1)])
```

`debounce` (`:220-229`) merges every **inner** run shorter than `k` into its neighbours, shortest first; edge runs
are kept. A touchdown is the first step `a > 0` of a contact run (`:262-268`); a stance is a run that also ends
before the trace does.

### 6.3 `window(rec, ta, tb, heading, H, yaw_target_sign=0.0)` (`metrics.py:235-329`)

`sel` = updates with `ta <= t <= tb`; `k0, k1` = the physics steps of the same span. `complete False` if fewer than
2 updates or steps.

| Reading | Lines | How it is measured |
|---|---|---|
| `duty` `[left, right]` | 247 | each foot's share of the window's physics steps in debounced contact |
| `flight_s`, `longest_flight_s` | 248-251 | steps with neither foot in debounced contact, x dt |
| `unloaded_steps` | 252 | the same before the debounce |
| `grf_peak_bw` | 253 | max total vertical load / weight |
| `drive_sustained_worst` `[actuator, ratio]` | 254-256 | each drive's RMS over the window's steps / its sustained rating; the worst |
| `roll_rms_deg`, `pitch_rms_deg`, peaks | 257-259 | over the window's updates |
| `touchdowns` `[left, right]` | 280 | touchdowns with `k0 <= k < k1` |
| `cadence_steps_s` | 274-281 | 2 / mean interval between one foot's successive touchdowns in the window |
| `stride_H` | 274-282 | mean distance between one foot's successive touchdown positions / H |
| `swings`, `toe_clearance_m` | 283-291 | see below |
| `stances`, `slip_median_m`, `slip_worst_m` | 292-304 | per stance wholly inside the window: the sole centre's largest horizontal displacement while the foot carries more than `control.SLIP_LOAD_FRAC` (0.2) of the weight |
| `slip_contact_path_*` | 299, 305-306 | slip speed integrated over the stance |
| `speed_m_s` | 308-311 | CoM displacement from `ta` to `tb` along `heading`, / (`tb - ta`) |
| `lateral_m` | 312 | the same displacement across `heading` (net, start to end; not a largest deviation) |
| `heading_change_rad` | 313-315 | unwrapped yaw at `tb` minus at `ta` |
| `yaw_rate_rad_s`, `turned_180` | 316-325 | see below |
| `dcm_error_rms_m` | 326-327 | RMS of `(e_x, e_y)` over the window |

**Toe clearance is read on the reference's clock** (`metrics.py:283-291`):

```python
# toe clearance: each reference swing's middle half in the window (the screens' reading, on the gait's clock)
lows, mins = U[:, [C["low0"], C["low1"]]], []
sw, u = U[:, C["ref_swing"]], U[:, C["ref_u"]]
mid = sel & (u >= clearance.MID_SWING[0]) & (u <= clearance.MID_SWING[1])     # (0.25, 0.75)
for i in (0, 1):
    s, e, v = runs(mid & (sw == i))
    mins += [float(np.min(lows[a:b, i])) for a, b, x in zip(s, e, v, strict=True) if x]
out["swings"] = len(mins)
out["toe_clearance_m"] = float(min(mins)) if mins else None
```

The swing foot and the progress are the reference's at each update's reference time, not the simulated foot's own
lift-off and touchdown. A foot that leaves late or lands early relative to the reference (servo delay, a clock
shifted by the phase term) has rows inside this band while it is on or near the floor. The reading is the worst
swing's minimum. On a pile the recorder's foot tables leave out the probe spheres (`env.foot_tables`).

**Speed over the judged 10 s.** The window: for the nominal walk the 10 s before the stop command
(`tune.py:558-560`, `tb = stop["command_s"]`), for a walk that ends in its gait the episode's last 10 s
(`panels.py:1109-1111`). `heading = rec.col("yaw")[0]`, the trunk's yaw at the first update (`tune.py:555`).
Success: `w["speed_m_s"] >= spec.speed_target` (`tune.py:636`), the reference's own speed
(`refs_doc[gait]["speed_m_s"]`, `tune.py:490`, `497`).

**Turn-rate success.** With `yaw_target_sign` set (`metrics.py:316-325`): `yaw_rate = pi / (time to turn through
180 deg from the window's start)`, `turned_180 True`; if the window ends first, `yaw_rate = turned[-1] / (tb - ta)`
and `turned_180 False`. The window is the steady turn: enter phase 1.0 to exit phase 0.0 for a walking turn
(`tune.py:504-505`), start phase 1.0 to the stop command in place (`tune.py:508`). Success:
`yr >= spec.yaw_target` (`tune.py:645`), `yaw_target = abs(refs_doc[case_dir]["yaw_rate_rad_s"])`.

### 6.4 Other measures

- `start_reading(rec, t_cmd, speed_cmd, stride_s, heading)` (`:332-348`): strides from the command until the CoM
  speed along `heading`, averaged over the last commanded stride, first reaches `START_SPEED_FRAC` (0.9) of the
  command. Returns `command_s, command_speed_m_s, stride_s, speed_frac, reached, strides`.
- `forward_acceleration(rec)` (`:361-365`), `acceleration_peak(rec, ta, tb, a_lp)` (`:368-371`): IMU acceleration
  along the trunk heading, zero-phase Butterworth low-pass (1 Hz, order 2).
- `joint_margins(q, lo, hi, rest=None)` (`:374-383`): each joint's smallest distance to its range; an end a joint
  rests on by design (`plant.rest_ends`, the `-X` jaw) is left out.
- `leg_gap(rec, lg)` (`:386-392`).
- `standing_noise_floor(body, run_dir, ...)` (`:422-451`): behind R2's rest speed.

### 6.5 The stop rule (decision 18, 1.5 strides)

- **The command.** `control.Track.add_transition` adds a stop event only when
  `TRANSITIONS.get(tr.name, (None, None))[1] == "stand" and "phase" in tr.arrays` (`control.py:994`). `tr.name` is
  the file stem (`transitions.py:972-974`). The event (`control.py:1000-1001`):
  `dict(kind="stop", name, t_ref, command_phase, speed_m_s, period_s)`, with
  `command_phase = stop_command_phase(duty) = 0.5 - duty/2 - PATH_HALF_WIDTH` (`transitions.py:783-785`), `duty` the
  mean support of the first `N_FRAMES` frames, `speed_m_s` from `tr.path`.
- **Sampling.** `control._stop_sample` (`control.py:2221-2234`): at the first update where the reference clock
  passes `t_ref`, the forward direction is fixed to the trunk's heading; after it each update appends
  `(time, v_forward, |v_horizontal|)` to **the latest stop reached**.
- **Reading.** `transitions.stop_reading(t, v_forward, command_speed, t_command, period, rest=None)`
  (`transitions.py:796-810`): `forward_stopped_strides` = (first time from which `|v| < 0.05 * command` holds to the
  end, minus the command) / stride; `within = fwd <= 1.5`; `at_rest_strides`, `at_rest_s` from the `rest` flags.
  `STOP_SPEED_FRAC = 0.05`, `STOP_WITHIN_STRIDES = 1.5` (`transitions.py:762-763`). At rate 1.05 the command is the
  reference's speed x 1.05 and a stride its period / 1.05 (`control._stop_result`, `control.py:2246-2260`). Rest
  speed: 1 mm/s for R0 and R1, 3.5 mm/s for R2 (`control.rest_speed`, `control.py:2237-2243`).
- **In the panels.** `start_stop` (`panels.py:1181-1210`) keeps only readings with `s["name"] == "stop"` (`:1200`)
  and ends each stop's part at the next segment named `"start"` (`:1202`). `stop_part_reading(rec, s, end_ref)`
  (`:1216-1240`) re-reads decision 18 over the stop's own part from the recorder's updates and keeps
  `control.run`'s own reading as `control_reading`.
- Nothing in the stock code counts touchdowns after a stop command. `transitions.stop_time_arrays` reports
  `single_step=True` and `last_touchdown_phase` for the reference (`transitions.py:825`).

### 6.6 Push recovery (decision 30)

`recovery(rec, t_end, standing, speed_cmd, stride_s, limit_s=3.0, until_s=None)` (`panels.py:1273-1316`).

- Standing: `hypot(v_x, v_y) < 0.05 * speed_cmd` at every update from some time at most 3 s after the pulse's end to
  the end.
- Walking: `stride_velocity` (`:1259-1270`) gives the CoM's mean velocity over the stride that starts at each
  update; on command means 0.9-1.1 x `speed_cmd` along `rec.col("yaw")[0]` and at most 0.1 x across.
- `recovered = full and after_s is not None and after_s <= limit_s`; `full` needs the trace to reach the planned end
  within two feedback updates. Returns `recovered, after_s, never_left_command, limit_s, trace_until_s,
  planned_until_s, on_command_frac_after_pulse, com_displacement_m` and the reading's thresholds.
- `speed_cmd` and `stride_s` are the nominal walk's (`Built`, `:1014-1017`).

---

## 7. Records

### 7.1 File naming

- Record: `<out>/<body>/eval/<row>/<behavior>/<condition>.json` (`record_path`, `panels.py:1471-1473`). A row
  `R2/42` is two directories; `R2s/42@10ms` is `R2s/42@10ms/`.
- Trace: `<out>/<body>/eval/traces/<row>/<behavior>/<condition>/<seed>.npz` (`trace_path`, `:1737-1739`).
- Summary: `<out>/<body>/eval/summary.json`, `<out>/<body>/eval/g7.json`.
- Example listing, `stage1/runs/2026-11-08/75-M-H-10/eval/`: `R0/`, `R1/` (22 records each: hard floor only),
  `R2/42/`, `R2/43/` (28 each: `carpet/{high,low,medium,robot_floor_carpet}.json`, `push/*.json` x 8,
  `stand/{hard,low,robot_floor_carpet}.json`, `start_stop/twice.json`, `turn/*.json` x 6, `walk/*.json` x 6),
  `R3/42`, `R3/43`, `R4/42`, `g7.json`, `summary.json`.

### 7.2 A condition record (`condition_record`, `panels.py:1653-1667`)

Top-level keys, as in `stage1/runs/2026-11-08/75-M-H-10/eval/R2/42/walk/nominal.json`:

```
schema 2 | body | row "R2/42" | behavior "walk" | condition "nominal" | key_name "walk/nominal"
gated | plan_gated | status "done"
controller   = row.describe()
definition   = cond.describe()
plan         [["transition","start"],["ref","nominal",7],["transition","stop"]]
seconds      18.3859...          (planned simulated s)
seeds        [3042..3061]
counts       {episodes, survived, succeeded, falls, errors, survival_lcb, success_lcb, judged, pass_}
failure      null or the failure dict        attribution  null or the attribution dict
terms        list                             term_classes {term: "hardware" | "controller" | null}
episodes     [20 episode records]
rules        {name: text}: COMMON_RULES plus the judge's (rules_of)
key          section 8.1
provenance   plant.provenance(): versions, git {commit, dirty, dirty_paths}, stage1_sources {sha256, rule, files}
inputs       {dir, rule, files {file: sha256}, records {"refs.json": sha, "transitions.json": sha}}
wall_s | episode_wall_s | simulated_s | started_at | ended_at
```

A "not run" record (`not_run_record`, `:1670-1676`) has `status "not run"`, `reason`, `settle`, empty `episodes`.

### 7.3 One episode record (seed 3042 of the record above)

```
seed 3042 | fell false | fall_time_s null | fall_cause null | fall_onset_s null | fall_candidates null
limits_before_onset null | survival_s 18.386 | planned_s 18.3859 | strides 17.00
worst_sustained_ratio ["tail_yaw_servo", 0.748] | worst_speed_ratio ["left_knee_servo", 0.531]
tilt_from_home_max_deg 0.506 | joint_margin_min_rad 0.123 | joint_shortfall_rad -0.023 | joint_worst "left_hip_roll"
estimator {com_vel_err_rms_m_s, com_rel_feet_err_rms_mm, tilt_err_rms_deg}
ankle_cap_residual {box_frac, box_max_excess_Nm, pad_frac, pad_max_excess_Nm, heavy_pad_frac, heavy_pad_max_excess_Nm}
captures 0 | phase_rate_range [-0.0233, 0.0208]
success {term: bool x 17} | succeeded true | failed_terms []
measures:
  survived_frac | dcm_error_rms_m | drift_m | legs | window_sustained_ratio
  start {command_s, command_speed_m_s, stride_s, speed_frac, reached, strides, acceleration_peak_m_s2}
  stop_reading {forward_stopped_strides, within, at_rest_strides, rest_speed_m_s, acceleration_peak_m_s2}
  stop null
  window {complete, t_start_s 5.152, t_end_s 15.152, speed_m_s 0.2496, lateral_m -0.106, heading_change_rad,
          yaw_rate_rad_s, turned_180, duty [0.605, 0.594], flight_s, cadence_steps_s, stride_H, grf_peak_bw,
          roll_rms_deg, pitch_rms_deg, roll_peak_deg, pitch_peak_deg, toe_clearance_m 0.0238, slip_median_m,
          slip_worst_m, slip_contact_path_median_m, slip_contact_path_worst_m, drive_sustained_worst, swings 19,
          stances 18, unloaded_steps}
  look {sway {strides, mean_H, max_H, per_stride_H, spec_H},
        head {joint, amplitude_rad, reference_amplitude_rad, error_rms_rad, error_max_rad, band_rad, in_band}, tail {...}}
trace {path "R2/42/walk/nominal/3042.npz", sha256, rows 920, every_s 0.02}
relaxed [] | wall_s 8.36
```

Other judges add: turns `measures.legs {poses, interference_free, interference_poses, tight_poses, min_gap_all_mm,
min_gap_all_pair, min_gap_moving_mm, min_gap_moving_pair}` and a window with `yaw_rate_rad_s`, `turned_180`;
start_stop `measures.start_stop {starts [...], stops [...], ok {...}}` (each stop: `name, command_ref_s, command_s,
command_speed_m_s, rest_speed_m_s, reached, part_from_s, part_to_s, samples, part_ends, command_update_offset_s,
stride_s, control_reading, forward_stopped_strides, within, at_rest_strides, at_rest_s, acceleration_peak_m_s2`);
pushes `measures.push {onset_s, end_s, direction, force_N, dv_m_s, impulse_N_s, bound_m_s, fell_before_pulse}` and
`measures.recovery`; carpet `measures.carpet {target_m, toe_clearance_m, swings, catching_swings}`.

**Not in an episode record:** touchdown counts (`window()` computes `touchdowns`, `WINDOW_KEYS` drops it, with
`longest_flight_s`, `seconds`, `updates`, `steps`, `dcm_error_rms_m`); any per-swing clearance; the CoM path;
`control.run`'s `yaw_turned_deg`, `lateral_drift_m`, `mean_forward_speed_m_s`, `segment_reached`, `stops` (raw).
`tests/test_stage1_panels.py:493-497` pins the top-level key set (`EPISODE_KEYS`) for the stock rows.

### 7.4 The trace (`write_trace`, `panels.py:1742-1754`; `TRACE_RULE` `:368-374`)

`np.savez_compressed` with:

| Array | dtype, shape | Content |
|---|---|---|
| `t` | float64 `[n]` | simulated s |
| `t_ref` | float64 `[n]` | reference-clock s |
| `rows` | float32 `[n, 31]` | `rec.U[:rec.rows:2]`: every second update, 50 Hz |
| `cols` | str `[31]` | `metrics.COLS` |
| `q` | float32 `[n, len(p.joints)]` | joint angles, plant order |

- **Written by default: yes.** `evaluate(..., traces=True)` (`panels.py:1817`, `1831`); `--no-traces` turns it off
  (`__main__.py:576`, `440`). Not written for the attribution's lifted runs (`panels.py:1785`) or by the ledger
  (`ledger.py:941`, `traces=None`). Git-ignored.
- **Enough for the new readings?**
  - Trunk heading: yes, `yaw` (wrapped; `np.unwrap`).
  - CoM drift: yes for the path, `com_x`, `com_y`. The commanded line itself is not stored; `Built.home_yaw` and
    `rec.col("yaw")[0]` are the two headings the stock code uses.
  - Own-swing toe clearance: `low0`, `low1` and `load0`, `load1` are there, but only at 50 Hz, and the 500 Hz
    `rec.loads` that `contact()` debounces is not in the trace. Lift-off and touchdown from the trace are 20 ms
    samples (the debounce width itself), and the weight for the 2 % threshold is not in the file.
  - Touchdown counts: the same limit.
  - The live `Recorder` inside `Row.run` has all of it at full rate (`loads` 500 Hz, `U` 100 Hz).

### 7.5 `summary.json` (`summarize`, `panels.py:1950-1989`)

Keys: `schema, body, rules, verdict_rows, verdict_controller, verdict_controller_note, grid, gated_verdicts,
gated_verdicts_by_controller, episodes, attribution_episodes, G7, sessions, written_at`.
`grid[behavior][condition] = {gated, plan_gated, floor, friction, rows {row: cell}, verdict, verdicts}`. The grid
iterates `conditions()` only (`:1965`); rows are found from the directory tree (`:1959-1961`). Each `evaluate` call
appends its `session` (`started_at, rows, behaviors, conditions, seeds, workers, attribute, traces, episodes_run,
attribution_episodes_run, skipped, rerun_failed, not_run, incidents, ended_at, wall_s, provenance, panel_sources,
key_context`).

---

## 8. Keys and hashes

### 8.1 `record_key` (`panels.py:1639-1650`)

```python
dict(schema=SCHEMA, body=body, row=_portable(row.describe()), row_sources=_portable(row.sources()),
     condition=cond.describe(), plan=[list(x) for x in resolved["plan"]], seconds=resolved["seconds"],
     seeds=list(seeds), push_bound_m_s=float(env.PUSH_BOUND_M_S[body]) if push else None,
     inputs=dict(files={f: inputs["files"][f] for f in files_of(cond)}, records=inputs["records"]),
     context=context, sources=sources)
```

- `inputs.files`: sha256 of exactly the condition's own reference and transition files. `inputs.records`: sha256 of
  the whole `refs.json` and `transitions.json`, so any change to either JSON changes every key of that directory.
- `_portable` (`:1627-1636`) makes paths under the study directory relative. **Observed:** the Stage 1 R1 and R2
  records (9 Nov, commit `3565509`) hold an absolute path in `key.row.gains_info.file`; later records (R3, the speed
  round) hold relative ones. `tests/test_stage1_bodies.py:143-158` recomputes an R0 key for that reason ("R0 has no
  tuned gains, so its key names no file path").
- `record_key` calls `files_of(cond)` -> `template(cond)`, so it raises for a condition `template` does not know.

### 8.2 `key_context(body)` (`panels.py:1609-1618`)

```python
files = dict(record=cd / "candidates" / f"{cid}.json", mjcf=xml, ledger=ledger,
             clearance=cd / "review" / "screening" / f"{cid}_clearance.json")
return dict(plant_inputs={k: plant.sha256(v) for k, v in files.items()}, versions=library_versions(), env=env_values(body))
```

- `plant_inputs`: sha256 of the body's round-two record, MJCF, model-ledger JSON and clearance screen
  (`plant.model_files(body)`, `case_dir`, `candidate_id`).
- `versions`: python, numpy, mujoco, scipy (`:1586-1597`). The records hold 3.13.18 / 2.5.4 / 3.10.0 / 1.18.1.
- `env` (`env_values`, `:1600-1606`): `push_bound_m_s, residual_rate, env_substeps, carpet_walk` (the plan),
  `carpet_friction`, `foot_tables_sha256` (sha256 of `inspect.getsource(env.foot_tables)`).

### 8.3 `panel_sources()` (`panels.py:1572-1583`; `PANEL_SOURCES_RULE` `:347-359`)

- **designlab/stage1 modules** (`key_modules`, `:1559-1569`): from `__init__.py` and `panels.py`, following
  `tune._imports` (every import in a module's source, module level or inside functions, read with `ast.walk`,
  `tune.py:389-414`), never descending into `KEY_EXCLUDED = ("env.py", "gates.py")` (`:1504`). Today:
  `__init__.py, control.py, drive.py, estimator.py, legs.py, metrics.py, panels.py, plant.py, refs.py,
  transitions.py, tune.py`.
- **designlab modules outside stage1** (`designlab_modules`, `:1545-1556`): only top-level `designlab/*.py`
  (`designlab_files`, `:1510-1512`), reached through **module-level** imports (top-level statements and those inside
  top-level `if` / `try`, `_module_imports`, `:1515-1542`), followed transitively. Today: `__init__.py, build.py,
  carpet.py, catalog.py, clearance.py, envelope.py, gait.py, geometry.py, hardware.py, mjcf.py, params.py,
  shapes.py, sizing.py, skeleton.py, statics.py, template.py, turning.py`.
- Hash: each list through `plant.sources_sha256` (name, NUL, bytes, NUL, in name order; `plant.py:267-275`), then
  `sha256(f"designlab/stage1\0{stage1_sha}\0designlab\0{dl_sha}\0")`.
- Returned: `sha256, rule, modules, stage1_sha256, designlab_modules, designlab_sha256, excluded`.
- **Not in it:** anything under `designlab/stage2/` (a sub-package: not globbed, and a stage1 import of it would be
  filtered out by `& set(files)`), `speedfit.py`, `ledger.py`, `policy_rows.py`, `report.py`, `speedreport.py`,
  `train.py`, `policy.py`, `replay.py`, `__main__.py`; `env.py` (held by value, 8.2) and `gates.py` (the
  attribution record holds `gates_sha256`, `:1728`).
- So an edit to `designlab/gait.py`, `designlab/turning.py`, `designlab/clearance.py` or `designlab/carpet.py`
  changes every panel key, as an edit to `metrics.py`, `tune.py`, `control.py`, `transitions.py` or `refs.py` does.
  The second group is also in the tuner's sources (`tune.tuner_modules()`: the same stage1 list without
  `panels.py`) and in the speed fit's (`speedfit.speed_sources`, rooted at `speedfit.py`, which includes
  `panels.py`).

### 8.4 Other hashes in a record or session

- `provenance.stage1_sources.sha256`: `plant.sources_sha256()` over **every** `designlab/stage1/*.py`
  (`plant.py:262-275`, `294-316`). Adding a file to `designlab/stage1/` changes it.
- `fingerprint(sources, context)` (`panels.py:1621-1624`): sha256 of the panel sources and key context; every
  worker recomputes it and refuses to start on a mismatch (`PanelWorker.__init__`, `:1765-1769`).
- `row.sources()`: `{}` for the stock tracker rows; `{"designlab/stage1/speedfit.py": sha}` for the speed rows; a
  policy row names `env.py`, `policy_rows.py`, `policy.py`. `module_sha256(obj)` (`:745-748`) hashes the one file
  that defines `obj`: for a subclass defined elsewhere that is the subclass's file only.
- Trace sha256 per episode; `inputs` sha256 (`tune.read_inputs`, `tune.py:320-347`).

### 8.5 Resume (`RESUME_RULE`, `panels.py:335-346`; `evaluate` `:1856-1871`)

Reused if the old record has the same key, `status "done"` and the same seed list; episodes with `error` are run
again alone and merged. Anything else is run whole and overwritten.

---

## 9. `speedfit.py` as a row module (R2s, R2@ms)

Importing it runs `_hook_panels()` (`speedfit.py:1210-1221`):

```python
panels.register_rows("R2s", r2s_row)
for ms in R2_ROW_DELAYS_MS:                       # (0, 10)
    panels.register_rows(f"R2@{ms}ms", r2_delayed_row)
panels.ATTRIBUTED = tuple(dict.fromkeys(tuple(panels.ATTRIBUTED) + SPEED_CONTROLLERS))
if not getattr(panels.g7, "speed_round", False):
    panels.g7 = _g7_guarded(panels.g7)            # FileNotFoundError for a body outside panels.G7_PLAN
```

| Row name | Factory | Gains | Delay |
|---|---|---|---|
| `R2s/<s>` | `r2s_row` (`:1172-1181`), regex `^R2s/(\d+)(?:@(\d+)ms)?$` | `r2s_gains`: `<speed run>/<body>/speedg5.json` `seeds[s].selected_gains`, refused unless `judged` | 0 |
| `R2s/<s>@<ms>ms` | same | same | `ms // 2` steps; even, 0-20 ms |
| `R2@10ms`, `R2@0ms` | `r2_delayed_row` (`:1184-1193`) | `start_gains`: `<speed run>/<body>/speedfit/start.json` (Stage 1's **body-level** G5 selection; a `-K` body's pick's) | 10 or 0 ms only |

- The speed run directory is `$STAGE1B_SPEED_RUN` if set, else `stage1/runs/2026-11-10-speed`, whatever `--inputs`
  is (`speed_run_of`, `:1124-1128`).
- `DelayedRow(name, gains, gains_info, latency_steps=0, controller="")` (`:1081-1113`) subclasses
  `panels.TrackerRow` in mode R2. `runs(cond)` is `cond.key in ROW_CONDITIONS` (`:1092-1093`); `ROW_CONDITIONS`
  (`:116-117`) = `walk/nominal, walk/brisk, walk/fr05, turn/walking_0.35_{left,right},
  turn/in_place_0.50_{left,right}`. `describe()` adds `latency_steps, latency_ms, latency_rule, rows_rule,
  conditions`; `sources()` is `panels.module_sha256(type(self))`; `spec()` adds `latency_steps`.
- There is no row for R2/42 or R2/43 (each seed's own selection) with a delay, and none at 2, 4, 6 or 8 ms for the
  `R2@` family.
- Fixed delay draws in the fit: `latency_steps(episode_seed, j)` (`:282-285`) =
  `default_rng(tune.derived_seed(episode_seed, j, 3)).integers(0, 6)`; `Evaluator.episode(..., latency=k)`
  (`:480-501`) forces one.
- Speed walks in the fit are the panels' conditions: `panel_condition(kind)` looks the key up in
  `panels.conditions()` (`:293-295`), and `Evaluator` builds them with `panels.Planner.resolve` and
  `tune.build_spec(plan=, gait=)` (`:438-445`).

---

## 10. `ledger.py` (skim): the other precedent

- Its own runner on the panels' classes: `LedgerContext(panels.Context)` (`ledger.py:738-900`),
  `LedgerWorker(panels.PanelWorker)` with its own top-level factory `_worker` (`:931-958`),
  `tune.Pool(..., mode="ledger", factory=...)` (`:1543`), its own records under `sweeps/`, its own source hash
  (the sha256 of `ledger.py`, `:970-973`) beside the sha256 of `panels.record_key(...)` per condition (`:1400`).
- `Variant(kind, value)` kinds (`PARAMETERS`, `:87-92`): `friction, push_dv, kp, latency_steps, mass, pile_damping,
  speed_scale, torque_scale, bus_v`. `LedgerContext.variant(cond, v)` (`:865-894`) applies one to the condition's
  already built plant for one episode and restores it exactly (`_apply`, `:801-863`).
- Seeds `panels.SEEDS[:3]`; body tables `MASSES` and `STAGE0` hold the three probe bodies only (`:79`, `:96`).
- Decision 33's reading of "works", the binding limit and the stop row "not run" are `WORKS_RULE`, `LABEL_RULE`,
  `LIFT_RULE`, `STOP_RULE` (`:152-257`).

---

## 11. Where the plan's decisions live in this subsystem

| Decision | Text (STAGE1_PLAN.md) | Code |
|---|---|---|
| 14 (`:210-214`) | turns: no joint limit reached and at least the reference's own smallest margin less 0.02 rad | `tune.build_spec` `tune.py:517-520`: `np.maximum(np.minimum(0.10, ref_margins - 0.02), 1e-3)`; walks 0.10 rad. The finding it records: the 0.5 rad/s turns leave 0.046-0.095 rad of hip-yaw range |
| 18 (`:236-238`) | stopped = forward speed below 5 % of the command within 1.5 strides of the stop command | section 6.5 |
| 20 (`:241-247`) | body-limited only if lifting the limit over the condition's 20 starts improves survival, p < 0.05 | section 5.4 |
| 30 (`:356-370`) | back to the command within 3 s, read ahead from the pulse's end, held 2 s | section 6.6 |
| 31 (`:371-409`) | how the panels read section 4 | the rule strings of `panels.py:148-385` |
| 33 (`:451-492`) | how the ledger reads section 5; "eval and sweep refuse a checkout without those files" | section 10; `__main__._check_models` |

---

## 12. Tests that pin behaviour

`tests/test_stage1_panels.py` (BODY 60-N-T-10; the episode tests skip without `stage1/runs/2026-11-08` arrays):

| Test | Asserts |
|---|---|
| `test_conditions_and_the_gated_split` | exactly 28 conditions with unique keys; the six reported ones; R0-type rows run 22, R2-type 28; nominal-only terms; decision 23 gating; `select` raises on `"swim"` |
| `test_seeds_slots_and_record_paths` | `episode_seeds` equals `tune.derived_seed` in slots 0 / 1 / 2; `SEEDS == range(3042, 3062)`; `record_path` layout |
| `test_planned_lengths_are_the_built_tracks` | `Planner.resolve` seconds equal `control.build_track(...).t_end / RATE`; the last 10 s of brisk and fr05 stay in the steady gait; `plan_is_tunes` true for nominal, 0.35 turn, in-place turn |
| `test_stands_hold_at_the_fastest_clock` | `CLOCK_MAX == 1.05 * 1.05 * (1 + env.RESIDUAL_RATE)` |
| `test_floors_without_a_settled_stance_are_not_run` | medium and high pile stand only on 75-M-H-10 |
| `test_clopper_pearson_lower_bounds` | 0.861, 0.656, 0.72526 |
| `test_counts_and_the_pass_rule` | 20 up and 17 succeed passes, 16 fails; fewer seeds unjudged; a failed episode is an error, not a fall |
| `test_success_failures_are_labelled_by_their_terms`, `test_the_failure_label_follows_most_failures`, `test_falls_are_labelled_by_the_attribution`, `test_verdict_logic` | the labels, `ATTRIBUTION_MIN_FALLS == 5`, `VERDICT_ROWS` exactly as in 5.3 |
| `test_stand_success` | tilt 12 deg, the limit terms, a fall's `failed_terms == ["survived"]` (synthetic, `NJ = 14`) |
| `test_push_recovery_rule` | decision 30 on synthetic traces |
| `test_carpet_toe_terms`, `test_start_and_stop_terms`, `test_each_stop_is_read_over_its_own_part`, `test_sway_on_a_synthetic_walk` | the readings of 5.2 and 6.5 |
| `test_rows_are_the_g5_selections` | R0 `terms == ()`, mode R1; R2/s gains equal tune.json's; `row_from_spec(r.spec()).describe() == r.describe()` |
| `test_r0_row_is_r0_on_the_commanded_clock` | at rate 1, `Tracker(p, track, "R1", terms=(), ankle_cap=())` equals mode R0 bit for bit (qpos) |
| `test_one_tiny_episode_per_row_type` | `set(e) == EPISODE_KEYS`; a `ZeroResidual` row equals R2/42 on every key but `wall_s`; its `sources()` names `env.py` and `panels.py` |
| `test_new_behaviours_on_real_episodes` | the push force, the carpet recorder's foot tables, friction 0.54 on the floor geom |
| `test_a_recorded_g5_episode_reproduces_through_the_runner` | 60-N-H-10's G5 episode 2010 equals the panels' walk on seed 2010, value for value |
| `test_records_and_resume` | key contents; trace path `R2/42/stand/hard/3042.npz`, `rows.shape == (n, len(COLS))`, `q.shape[1] == 14`, 0.02 s spacing; skip on the same key; rerun of a failed episode; a changed source hash reruns the record |
| `test_eval_command_line` | `eval --dry-run` writes nothing; `eval` writes a record and prints "G7 pass" |
| `test_g7_gate_and_the_panel_sources` | G7 passes; `panels.py, tune.py, control.py, metrics.py, plant.py, transitions.py` in the modules, `env.py, gates.py, train.py, policy.py` not; `carpet, statics, clearance, sizing, gait` in the designlab list; an `env.py` edit leaves the hash, a `control.py` or `carpet.py` edit changes it |
| `test_attribution_runs_on_the_falls_of_r2_rows`, `test_a_worker_refuses_other_panel_sources`, `test_rows_from_another_module`, `test_a_policy_acts_every_env_step` | attribution, the worker fingerprint, `register_rows`, `Policy.every == 10` |

`tests/test_stage1_bodies.py`: `test_the_probe_bodies_plant_inputs_are_their_records_keys` (today's
`key_context(body)` equals the committed R0 record's `plant_inputs` and `env`);
`test_a_probe_body_panel_record_key_recomputes` (the committed R0 `walk/nominal` key recomputes with the record's
own `sources`; with today's hash only `sources` differs); `test_a_probe_body_ledger_key_recomputes`.

`tests/test_stage1.py`: `test_recorder_changes_nothing_and_the_commanded_rate_runs_the_clock_faster` (`:1696`),
`test_walk_measures_on_a_short_walk` (`:1735`; `touchdowns == [3, 3]`, toe clearance 0.04-0.06 H, load columns equal
`rec.loads[::5]`), `test_drive_rms_is_judged_over_the_window` (`:1760`), `test_contact_runs_and_debounce` (`:1810`),
`test_stop_reading_is_decision_18` (`:1468`), `test_stop_reading_on_the_commanded_clock` (`:1835`).

`tests/test_stage1_speedfit.py`: `test_the_rows_delay_reaches_the_servos` (`:431`),
`test_the_rows_are_registered_and_read_their_gains` (`:498`; bad names `R2s/42@3ms`, `R2s/42@22ms`, `R2@6ms`
raise), `test_a_row_runs_through_the_panels` (`:532`), `test_the_speed_rows_are_attributed` (`:654`),
`test_stage1_tune_objective_and_records_are_unchanged` (`:693`; `speedfit.py` is in neither the tuner's nor the
panels' sources).

`tests/test_stage2_x.py::test_an_x_reference_and_zero_action_walk_can_meet_joint_range` (`:563`): `panels._base` and
`_limits_ok` on a 16-joint body with the jaw on its stop.

---

## 13. Hard-coded assumptions the build list would break

**New behaviours and new references (A, B, D, E)**

| Where | Assumption |
|---|---|
| `panels.py:453-474` | the condition list is a function body. Called by `evaluate` `:1826`, `dry_run` `:2096`, `summarize` `:1965`, `panel_files` `:563`, `PanelWorker` `:1771`, `measure_costs` `:2057`, and by `speedfit.panel_condition`, `ledger`, `speedreport` |
| `panels.py:89`, `:481-483` | `BEHAVIORS` is closed; `select` raises on another name |
| `panels.py:524-532` | a walk's gait is `nominal`, `brisk` or `fr05`; the transition names `start`, `nominal_to_brisk`, `brisk_to_fr05` |
| `panels.py:533-536`, `tune.py:116` | a walking turn is start, enter, **5 strides**, exit, stop |
| `tune.py:115`, `:278`, `:495-496` | the nominal walk is 7 strides; `build_spec` raises "10 s before the stop command reaches into the start's ramp" if 7 strides are too short for the window |
| `panels.py:552`, `:996`, `:1014-1017`, `:644` | `refs/nominal.npz` and `refs_doc["nominal"]` set omega0, `speed_cmd`, `stride_s` and the walking push's period for every condition |
| `panels.py:656-658`, `:1200`, `:1202`, `:113`, `:1026-1028` | start-and-stop reads transitions named exactly `start` and `stop`, twice |
| `tune.py:484` | the first track segment must be a transition whose `transitions.json` entry has `gaits.b.duty` |
| `tune.py:486-488` | exactly one stop event, or none for a custom walk plan |
| `tune.py:500-505` | a turn needs `refs_doc[f"{case}_{direction}"]["yaw_rate_rad_s"]` and segments named `enter_...`, `exit_...` |
| `control.py:994`, `transitions.py:120-131` | a stop command exists only for a file whose stem is in `transitions.TRANSITIONS` with target `"stand"` and which holds a `phase` array. A two-step stop saved under a new name yields no stop event and no reading |
| `transitions.py:783-785`, `:825` | the stop command phase formula and `single_step=True` describe the one-step stop |
| `metrics.py:316-325`, `panels.py:220-226` | yaw rate is read through 180 deg; a 90 deg turn falls to the window's mean rate with `turned_180 False` |
| `panels.py:124`, `:447` | seed slots by kind; a new kind gets slot 0 |
| `panels.py:887-888`, `:869-870` | `R2/...` names are parsed as integers; `R2` cannot be registered |
| `speedfit.py:1092-1093`, `:121`, `:1187` | `DelayedRow` runs seven stock conditions; `R2@` rows exist at 0 and 10 ms only |
| `panels.py:979-988` | friction belongs to the condition, not the row |

**New readings (C)**

| Where | Assumption |
|---|---|
| `metrics.py:283-291` | toe clearance on the reference's swing clock |
| `metrics.py:308-312`, `tune.py:555` | speed and lateral displacement are net, along and across the trunk's first yaw; no largest deviation from a line |
| `panels.py:1090-1096` | `WINDOW_KEYS` and `MEASURE_KEYS` fix what a record keeps; `touchdowns` is dropped |
| `panels.py:1031-1047`, `:1099-1145`, `:404-412`, `:926-927` | `terms_of`, `judge`, `JUDGE_RULES`, `Built.legs` dispatch on the eight judge strings and raise on another |
| `panels.py:1661-1662` | a record's `terms`, `term_classes` and `rules` come from the stock functions, whatever the row wrote into its episodes |

**Bodies (F and the -X, -K bodies)**

| Where | Assumption |
|---|---|
| `panels.py:387`, `:2246` | `G7_PLAN` has the three probe bodies; `g7` raises `KeyError` for another unless `speedfit` is imported. `summarize` catches only `FileNotFoundError` (`:1979-1983`); a `KeyError` (`doc["info"]["sway_amp"]`, `:2242`) or `ValueError` (non-straight reference, `:2166-2167`) leaves `evaluate` after the records are written |
| `panels.py:1018`, `:1603`, `env.py:257-272` | `env.PUSH_BOUND_M_S[body]` reads the body's envelope screen for an unlisted body and raises `KeyError` without it |
| `panels.py:140`, `:1354` | `LOOK_JOINTS` names `neck_pitch` and `tail_yaw` (`joints.index`) |
| `panels.py:1821` | `body` must be in `stage1.BODIES` |
| `env.py:286-290` | `ACTION_DIM = 15`, `WIDTH_JOINTS`: the envs refuse another joint count (another map's subject) |
| `test_stage1_panels.py:310`, `:642` | the tests' `NJ = 14` and `q.shape[1] == 14` |

No 14-joint assumption found in `panels.py` or `metrics.py` themselves: `Recorder.Q` and `Built.lo/hi/required` are
sized by `len(p.joints)`.

**Running**

| Where | Assumption |
|---|---|
| `panels.py:1827`, `:1771` | `read_inputs(run_dir, panel_files())` checks all 26 stock files, whatever `--conditions` selects; a missing or unlisted file raises `tune.InputError` |
| `panels.py:1829`, `:2059` | row gains (`tune.json`) are read from the `--inputs` directory; the dry run's cost measure needs `R2/42` and the stock nominal walk there |
| `panels.py:1862-1871`, `:1921` | a different key overwrites; the summary is always rewritten |
| `tune.py:966`, `panels.py:1771-1777` | spawned workers read their inputs before any row module is imported |

---

## 14. Extension points and hazards (short)

**What already works without touching `designlab/stage1`:**

1. **A new row** in a module under `designlab/stage2/`, registered with `panels.register_rows` and loaded with
   `--row-module`. The class path in `spec()["cls"]` makes spawned workers import it. A subclass of
   `speedfit.DelayedRow` gets the servo delay; it should override `runs` (else only the seven stock conditions) and
   return both files from `sources()` (its own and `speedfit.py`, since `module_sha256(type(self))` hashes only the
   subclass's file). A row for each CMA-ES seed's own selection can call `panels.r2_gains(dir, seed)` on Stage 1's
   run directory rather than on `--inputs`.
2. **New readings** inside an overridden `Row.run`: call `self.episode(...)` for `(b, out, rec)`, then
   `panels.judge(...)`, then read `rec` (500 Hz `loads` / `contact()`, 100 Hz `U`, `Q`) and add to
   `e["measures"]`. Rule texts placed in `describe()` enter both the record and the key (`DelayedRow` does this with
   `latency_rule`). `counts` reads only `fell`, `succeeded`, `seed`, `error`.
3. **A separate output directory** with `--out`, references from `--inputs`.
4. **Stop events are data**: `Track.events` is a list (`control.py:961`) that `control._episode` reads at the
   start of an episode (`control.py:2325`).

**What does not work without either rebinding `panels` attributes or a runner of one's own:** new behaviour names,
a walk on a gait other than `nominal / brisk / fr05`, a turn of other than 5 strides, any new judge string. Both
precedents exist in the tree:

- *Rebinding at import* (`speedfit._hook_panels` rebinds `panels.ATTRIBUTED` and `panels.g7`). `conditions`,
  `template`, `BEHAVIORS`, `judge`, `_worker` are all looked up as module globals at call time, so a stage2 module
  can rebind them. Hazards: it changes `panels` for everything else in the process (`speedfit.panel_condition`,
  `ledger`, `speedreport`, `summarize` all call `panels.conditions()`); with `--workers > 1` the spawned worker
  checks inputs with the stock `conditions()` before the module is imported, so the 26 stock files must be present
  and the new files are not verified in the workers (`--workers 1` avoids this; the speed round ran with 1).
- *An own runner* on the panels' classes (`ledger.py`): a `Planner` subclass overriding `_resolve`, a `Context`
  subclass that swaps `self.planner`, a `PanelWorker` subclass with a top-level factory in the stage2 module (so
  spawn imports it first), `tune.Pool(factory=...)`, own record directory and own source hash beside the panel
  key. Stock `Context.built` then accepts any plan through `tune.build_spec(plan=, case=, gait=)`. Hazards:
  `panels.record_key` and `files_of` call `template`, and `condition_record` calls `terms_of` and `rules_of`, so
  conditions must keep a stock `judge` string and the key needs its own file list.

**If edits to `designlab/stage1` were chosen instead**, these are the places: `panels.py:89` (`BEHAVIORS`),
`:124` (`SLOTS`), `:404-407` (`JUDGE_RULES`), `:445-447` (`Condition.kind`), `:453-474` (`conditions`), `:519-547`
(`template`), `:926-927` (`Built.legs`), `:1031-1047` (`terms_of`), `:1090-1096` (kept keys), `:1099-1145`
(`judge`); `metrics.py:235-329` (`window`); `tune.py:469-529` (`build_spec`); `control.py:994` with
`transitions.py:120-131` for a new stop name. Every one of them is inside the panels', the tuner's and the speed
fit's source hashes (8.3) and inside `plant.sources_sha256()` (8.4).

**Other hazards**

- Wall-clock statements ("at 2.0 s", "the judged 10 s") need a clock: the reference runs 1.05 x simulated time.
  A reference built as 90 deg in 2.0 s of reference time finishes at about 1.905 s simulated.
- The stock walk success `speed_m_s >= target` over 10 s is equivalent to at least `10 x target` metres along the
  first heading (4.0 m at 0.40 m/s), measured net, not as path length.
- Reusing stock file names (`nominal`, `start`, `stop`) for new arrays in a new `--inputs` directory runs with no
  code at all, but the records then say `walk/nominal` for another walk, G7 compares it with Stage 1's plan figures,
  and the 7-stride plan may not hold a 10 s window for a shorter period.
- A new controller prefix is gated on carpet (only `R0, R1, R2` are "tracker controllers", `panels.py:88`) and is
  not attributed until it is in `panels.ATTRIBUTED`.
- Purpose index 3 of `tune.derived_seed` is taken by the speed fit's delay draw.
