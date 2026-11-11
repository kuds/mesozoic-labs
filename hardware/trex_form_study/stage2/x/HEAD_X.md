# Stage 2 S2-2: the expressive head (`-X`), how its bodies were built and checked, and what is open

`STAGE2_PLAN.md` S2-2 and decision P29: a neck yaw and a hinged jaw on the two picks, modelled in simulation beside
today's head. Nothing is adopted (P24), and P2's deferral of a jaw for the build stands. These notes follow arm K's
(`stage1/runs/2026-11-10-speed/ARM_K.md`). The commands are in `run_x.sh`. The git-ignored inputs are listed with
their sha256 in `x_inputs.json`, and `x_inputs.py` checks, copies and regenerates them. "a / b" figures are
`60-N-T-10-X` / `75-M-H-10-X`.

## 1. The bodies

Each `-X` body is its pick's sized record (`round2/refine/{narrow_carpet_60,feet_75}`) with two joints and their
drives added. It is built once and not re-sized (refine cases `head_x_60`, `head_x_75`; `refine.HEAD_X_RULE`).

| | `60-N-T-10-X` | `75-M-H-10-X` |
|---|---|---|
| Model `xml_sha256` (`SHA256_PREFIX`) | `9718b3b5…` | `66f531bb…` |
| Neck yaw | at the neck base, ±0.6 rad, STS3215 | the same |
| Jaw | hinge 0.10 m ahead of the neck-pitch axis, 0–0.6 rad, STS3215 | the same, at 0.12 m |
| Added mass | 0.150 kg (2 × 0.055 kg servo, 2 × 0.020 kg horn and bracket) | the same |
| Locked (both new joints at 0) | today's body plus the added parts (mass, CoM, inertia; `tests/test_stage2_x.py`) | the same |
| Round-two screens | all pass | all pass |
| Stage 1 gates | G1 provisional, G2 and G3 pass; G4 does not apply | the same |

Every model the Stage 1 registry used before (`60-N-T-10`, `60-N-H-10`, `75-M-H-10` and the two K bodies)
regenerates with its recorded `xml_sha256`. The tests check this.

## 2. What a checkout needs

A merge does not bring the git-ignored inputs. `x_inputs.json` lists all 60 of them. This review left every one
unchanged (`run_x.sh check` passes):

- the cases' model files: `round2/refine/head_x_60/models/60-N-T-10.{xml,json}` and
  `round2/refine/head_x_75/models/75-M-H-10.{xml,json}`;
- `stage2/x/<body>/refs/*.npz` (10 per body) and `stage2/x/<body>/transitions/*.npz` (18 per body).

After the merge, run these in the main checkout from `hardware/trex_form_study`:

```
bash stage2/x/run_x.sh copy-from ../../.claude/worktrees/stage2-x
bash stage2/x/run_x.sh check
```

Without the worktree, `run_x.sh run` regenerates the same files and checks them against the list.

For the review's tests, this worktree also holds git-ignored copies of the main checkout's Stage 1 run arrays and
three PPO checkpoint directories (`stage1/runs/2026-11-08/*/{refs,transitions}`, `.../ppo/42/checkpoints`,
`.../policy_only/42/checkpoints`, and arm K's arrays). They are the main checkout's own files and need no copying
back.

## 3. The gates: G1 is provisional

G1's step-rate rule times one process per core. This branch ran beside another build and was held to 2 processes,
so G1 timed the step rate on 2 of the machine's 4 cores. The first record (commit e72ccc5, titled "all pass") read
that timing as a pass. The rule was not met, so that was wrong.

`gates.step_rate` now records the machine's cores (`gates.machine_cores`: `os.sched_getaffinity`, which
`PYTHON_CPU_COUNT` does not change). With fewer processes than cores the step rate is not judged:

- `ok` is None, with `ok_on_these_processes` beside it (`gates.PROVISIONAL_RATE_RULE`);
- G1's `pass` and the record's `pass` are None, and the record lists G1 under `provisional`;
- the gates summary prints "G1 provisional (step rate timed on 2 of 4 cores (PYTHON_CPU_COUNT=2))".

The records now read: **G2 and G3 pass; G1 is provisional on its step rate.** Every other G1 check passes.

| | `60-N-T-10-X` | `75-M-H-10-X` |
|---|---|---|
| G1 stands (tilt max, hard / low pile) | 0.54 / 0.53 deg | 0.51 / 0.53 deg |
| G1 step rate on 2 of 4 cores: alone, per process (lowest sample) | 21,095, 23,069 (19,763) | 23,396, 22,258 (19,762) |
| G2: nominal panel median (runs ≥ 10 strides), unperturbed with / without envelope | 45.9 (20/20), 50.9 / 50.9 | 60.0 (20/20), 60.0 / 60.0 |
| G3: transitions, worst CoM, worst ZMP | 18/18, 0.280 mm, 0.895 mm | 18/18, 0.312 mm, 1.075 mm |

**To finish G1:** on an otherwise idle machine, run `bash stage2/x/run_x.sh g1`. That command uses one process per
core, as arm K's does (`PYTHON_CPU_COUNT` unset), and replaces both G1 records. On this machine, arm K's G1 failed
its step rate with 4 processes on the K bodies and on the pick alike (Stage 1b decision 11). So the `-X` bodies may
fail it too.

On a shared machine, `G1_PROCS=N` times on N processes, and the record marks it provisional. The old `g1` default,
`$(nproc)`, would have timed on 1 process: the script exports `OMP_NUM_THREADS=1`, and `nproc` reads it.

## 4. The joint ranges: the seams the clearance screen cannot see

The plan asks for the neck yaw to be "cut to the clearance screen's range". The screen cannot cut either new range.
It pairs no body with its parent, so its sweeps never see the neck against the chest (neck yaw) or the jaw against
the skull and neck (jaw). The plant therefore keeps the MJCF limits: ±0.6 rad and 0–0.6 rad.

`clearance.seam_sweep` (`SEAM_RULE`) now sweeps those parent/child seams. It sweeps each joint alone, from home to
each limit, over the screen's own geoms and distance oracle. It reports two things: new contacts, and home overlaps
that grow by more than 2 mm. The case records hold its results in `head_x.joint_ranges`, with today's neck pitch and
tail yaw for comparison. `plant.json`'s `joint_ranges.not_screened_at_seam` points to them.

| Joint (moving on parent) | New contact | Deepest growth of a home overlap, a / b |
|---|---|---|
| neck yaw, ±0.6 (head on trunk) | none | 17.8 / 23.6 mm (`cl_neck` × chest proxy); 11.3 / 11.3 mm against the yaw servo |
| jaw, 0 to 0.6 (jaw on head) | none | 21.6 / 28.7 mm (`cl_jaw` × `cl_neck`); 6.9 / 6.5 mm against the jaw servo |
| today's neck pitch, ±0.35 | `cl_neck` × `hw_spine_beam` from −0.30 rad (both sizes) | 17.7 / 27.2 mm |
| today's tail yaw, ±0.45 | `cl_tail` × `hw_battery_cassette` from ±0.35 rad (60 cm) | 13.6 / 16.2 mm |

**Decision: annotate the ranges, do not cut them** (`refine.HEAD_X_RANGES_DECISION`):

- Neither new joint makes a new contact.
- What grows is the overlap between coarse proxies that already overlap at home. The growth is of the same order as
  today's neck pitch over its range, which rounds one and two accepted.
- The neck proxy grows against two servos: the neck-yaw servo, and today's pitch servo, which this model keeps on the
  trunk (on a real yaw stage it would turn with the neck). That is packaging for the yaw stage's own design to settle.

**The jaw cover overhangs its hinge.** The hinge is not at the jaw's rear, as build.py's comment said (now
corrected). Today's jaw cover starts at the skull's rear, 17.7 / 18.4 mm behind the hinge. Opening the jaw to 0.6 rad
lifts that part 18.3 / 21.6 mm into the skull and neck (`head_x.joint_ranges.cover_behind_hinge`).

A build would trim the cover at the hinge. The model keeps the cover whole, so the locked body stays today's plus the
added parts. Renders of an open jaw will show the overhang. Trimming it is a model change, so every `-X` record would
change. If Track A's look review wants it trimmed, do it before any `-X` training.

**The blind spot predates this branch.** The last two rows show it on today's bodies. Those bodies are unchanged and
were not examined further here.

## 5. The jaw's closed stop and Stage 1's judges

The jaw rests closed on its lower stop by design (`clearance.REST_ON_STOP`). Stage 1's judges required 0.10 rad of
range on every hinge in a walk. That had three effects:

- every judged `-X` walk would fail "every limit" on the jaw;
- the controller's joint-range hardware flag fired at every update, which polluted decision 20's attribution;
- G2's replays listed the jaw as reaching its limit.

Pressed by the walk, the jaw goes up to 1.4 / 2.4 mrad past its stop.

**Option B was taken** (`plant.REST_ON_STOP_RULE`): an end a joint rests on is not a range limit. The changes:

- the margins drop that end (`metrics.joint_margins(..., rest)`), as do tune's spec and measure and the panels' `_base`;
- the joint-range hardware flag ignores it (`control.limit_bounds`);
- the joint-range counterfactual does not widen it;
- G2's replay margins and limits reached leave it out, and report it apart as `rest_on_stop`.

The open end keeps the 0.10 rad rule. No model changed. Every earlier body has no such joint, so its margins, bounds
and masks are unchanged bit for bit. Stage 1's suites pass with their run arrays present, including the check that
replays a recorded G5 episode value for value.

Option A was not taken. A 0.10 rad walk margin at a closed stop means one of two things: a jaw that can close 0.1 rad
through the skull, or a mouth that hangs 5.7° open at rest. The reviewer's example range, (−0.05, 0.6), would still
fall short of the rule by 0.05 rad.

Tests (`tests/test_stage2_x.py`):

- the nominal walk's track (`tune.build_spec`) keeps 0.10 rad on every joint of each `-X` body;
- a 1.5 s R0 walk, the panels' zero-action row, meets the panels' `joint_range` limit, where the old reading failed
  it by 0.10 rad on the jaw alone.

## 6. The cost of the head: two stances

The build re-solves the stance (`statics.solve_stance`), so the `-X` home is not the pick's:

- hip pitch −0.016 / −0.0088 rad;
- knee −0.0012 / −0.0034 rad;
- ankle pitch +0.017 / +0.012 rad;
- the trunk's pose is unchanged.

The case records now hold:

- `head_x.home_pose`: the changes listed above;
- the CoM change at the pick's home, beside the change at each body's own home;
- `against_base_at_pick_pose`: the `-X` body posed at the pick's home with the new joints at 0, through
  `sizing.requirements`, against the pick's drives;
- `loads_between_stances`: how far the two comparisons differ.

What the two stances do and do not change:

- **CoM ahead of the hip:** +6.39 / +5.24 mm at each body's own home, but +5.19 / +4.43 mm at the pick's home. The
  extra 1.2 / 0.8 mm comes from the re-solved stance, not from the added parts.
- **Loads:** the two comparisons agree. The ratios to the drives' limits differ by at most 0.0008 / 0.0004, and the
  holds and RMS torques by at most 3e-5 / 1e-5 (relative). The sizing's holds and walks re-centre the CoM over the
  support from whichever home they start. So the load changes in `against_base` come from the added parts alone. The
  knee hold falls (3.05 → 2.73 / 2.79 → 2.36 N·m) and the knee's speed ratio rises (0.93 → 0.97 / 0.93 → 0.96) at
  the pick's home too.
- **The plant's leg ranges:** the clearance screen reads them by sweeping from home, so they move with the home, by up
  to 0.0001 / 0.0017 rad.

The head's costs at either stance:

| | 60 cm | 75 cm |
|---|---|---|
| Hip-pitch hold, N·m (share of sustained) | 3.27 → 3.92 (0.55 → 0.65) | 8.69 → 9.48 (0.76 → 0.83) |
| Neck, continuous ratio | 0.76 → 0.85 | 0.66 → 0.71 |
| Knee speed ratio (60 cm: now the governing limit) | 0.93 → 0.97 | 0.93 → 0.96 |

## 7. Open S2-2 items

1. **G1 with one process per core** (section 3).
2. **The env's 14-joint widths.** `env.ACTOR_OBS`, `CRITIC_OBS` and `ACTION_DIM` are built for 14 hinges. The `-X`
   bodies have 16. They are not generalised yet, because Track B's R5 env sets its own inputs (the last three
   actions, the head command on `-X`). Until then the PPO envs refuse a plant with another hinge count
   (`env.WIDTH_JOINTS`) rather than build spaces of the wrong width. Generalise them before any `-X` training or
   policy row, and test them on an `-X` plant.
3. **Done in this review: the leg/head split.** It is now based on role, not name:
   - `plant.leg_drives` (`sizing.SINGLE_GROUPS`) gives the controller's leg-drive mask, which sets the saturation
     flags, the attribution and the `drive_torque` counterfactual;
   - `plant.leg_joints` (`mjcf.HEAD_TAIL_ROLES`) gives the residual scales in env.py and policy_rows.py.

   The neck yaw and the jaw are now head drives and joints, like the neck pitch and the tail yaw. Before, the
   controller counted both new drives as legs, and the env's name prefix counted the jaw joint as a leg.
4. **The jaw cover's overhang** (section 4): trim it before `-X` training if the look needs it.
5. **Not modelled:** the yaw stage's own bearing (mass and cost), and today's pitch servo riding on the yaw stage.
6. **Today's seams** (section 4, last two rows): not examined further here.

`tests/test_stage1_policy_rows.py::test_the_ledger_sweeps_the_policy_rows_r3` fails before and after this branch.
The committed `run.json` files judge the 60 cm R3 with R2 (decision 38), so the ledger leaves R3 out, and the test
still expects it.
