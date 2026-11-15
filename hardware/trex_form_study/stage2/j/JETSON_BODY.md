# Stage 2, Jetson track task 6: the Jetson body (`75-M-H-10-J`), how it was built and checked, and what is open

`JETSON_PLEM_REVIEW.md` section 5 task 6 and the owner's decisions of 14 Nov 2026 (`DECISION_LOG.md` P30): the Jetson
Orin Nano Super dev kit is studied as a refine case, a simulated body on which every round-two screen is run. Nothing is
adopted (P24). P14's Raspberry Pi 5 + AI HAT+ stays the adopted compute, the walking loop stays on the microcontroller,
and no power mode is chosen. The compute choice is the owner's, after these screens and the delay bench test.

These notes follow the expressive head's (`stage2/x/HEAD_X.md`). The commands are in `run_j.sh`. The git-ignored inputs
are listed with their sha256 in `j_inputs.json`, and `j_inputs.py` checks, copies and regenerates them. Everything here
is simulation; nothing is built.

Every number carries its kind of evidence, named as `JETSON_PLEM_REVIEW.md` names them:

- (a) a model result of this study: a static or prescribed-walk reading of the simulated body;
- (b) a simulated run of this study: a forward simulation of the body under its joint servos. There are three: the
  carpet screen's standing, G1's stands and G2's feedforward replays of the references. No balance controller or
  policy was run;
- (c) a sourced fact, with its catalog id (`designlab/data/parts.json`) or document;
- (d) an estimate or assumption.

G1's step rates are timings of this machine, not model results.

The tables in section 4 are written by `j_tables.py` from the committed records, so no figure in them is typed by hand
(`tests/test_stage2_j.py` checks the block). The prose quotes them rounded.

## 1. The body

`75-M-H-10-J` is the 75 cm pick's sized record (`round2/refine/feet_75/candidates/75-M-H-10.json`) with its compute
part changed and one part added. It is built once and not re-sized: the drives, the pack, the bus, the structure and
the covers are the pick's (refine case `jetson_75`; `refine.JETSON_RULE`). This is how the expressive head's bodies were
made.

**Compute.** The selection's compute part is the catalog's dev kit, `nvidia_orin_nano_super_devkit` (c): 0.174 kg,
103 × 90.5 × 34.77 mm, $399. `designlab/build.py` places it in the chest tray exactly where it places the Pi set, at
0.60 B along the trunk. The tray's envelope grows with the board, as it does for any compute part. Two cautions from the
catalog's own notes: the 0.174 kg is an NVIDIA forum figure for the original kit, not a datasheet value, and NVIDIA's
two pages give different dimensions (the larger set is used).

**Control layer: the owner's baseline, Jetson + microcontroller.** The Teensy control stack (0.18 kg, 4 W; d, PR #662's
allowances) and the IMU stay exactly as the pick has them.

**The regulator.** The dev kit's carrier takes 9–20 V at its DC jack (c, the compute survey,
`designlab/data/research/compute_2026-11-07.json`). A 6S pack is at 25.2 V when full, so the carrier needs a step-down
regulator (decision-log E7). The catalog has two 12 V parts from Pololu, both safe on 6S (E7 rules the smaller one out
only on a 12S bus):

| Catalog id (c) | Rated output | Mass | Price | 15 W mode | 25 W mode |
|---|---|---|---|---|---|
| `pololu_d36v50f12` | 4.5 A, 54 W | 7 g | $39.95 | covers it (48 W) | does not (58 W) |
| `pololu_d42v110f12` | 9 A, 108 W | 15 g | $69.95 | covers it | covers it |

The body places **`pololu_d42v110f12`**, because no power mode is chosen and it is the lighter of the parts that cover
both (`refine.JETSON_REGULATOR_RULE`; `jetson.regulator` in the record). "Covers" is read against the worst case at
the jack:

- the mode's rail budget of section 7: 18 W or 28 W (c + d);
- plus the most the carrier can pass to its USB ports: two port pairs at 3 A and 5 V, 30 W (c, the dev kit's entry in
  `designlab/data/research/parts_2026-11-07.json`). The head camera is fed from those ports.

At the rail budget alone, both parts would do: the smaller one runs at 33 % and 52 % of its rating. So the choice rests
on the 30 W for the USB ports, and that worst case is this study's construction (d), not a sourced figure for the jack:
the catalog does not give the rating of the kit's own 19 V supply (open item 6). What the choice costs is in the note
under table 1: with the smaller part the body would be 8.0 g lighter, 8.5 g with the reserve.

The rating itself is the catalog's one point, 12 V times the vendor's current at one stated input voltage. The vendor
gives a range of currents over input voltage and no figure at 20–25 V (d).

The regulator is a 31.8 × 43.2 × 9 mm box (c) lying flat on the power box, 5 mm above it on standoffs (d), in the
abdomen. The 6S bus is there, and the 12 V lead runs forward to the carrier. Its mass in the model is 25 g: the
catalog's 15 g (c) plus 10 g for standoffs, the lead and its plug (d). Its ledger row says so in its basis
(`build.compute_regulator_basis`), so the ledger's sum by basis does not book the 10 g as sourced (table 2's note).

**It may be counted twice.** The power box's 0.30 kg is PR #662's power bay (d). That robot carried a Jetson, and its
bay held three regulators: a `D36V50F12` for the computer, a `D42V110F12` for the servos and a 5 V one for logic
(`r0_posture_study_2026-11-07/study/hardware/component_basis.md`). The bay's own breakdown is not in this repository;
`stage2/jetson75/BUDGET_75.md` reaches the same point for its price. So the box may already hold a 12 V regulator
for the computer, the smaller part, which by the rule above covers the 15 W mode only. The body is the heavier
reading: at most 26.6 g too heavy, the regulator's row with its reserve (d). The dev kit's rail here is its own,
separate from the servos'.

**What leaves with the Raspberry Pi set.** The pick's record was read before deciding (`jetson.pi_set` in the record):

| The pick's record holds | Leaves? |
|---|---|
| `compute`: catalog part `rpi5_8gb_ai_hat_plus_13tops`, one part of 0.100 kg (the Pi 5, its cooler and the AI HAT+) | yes: the dev kit takes its place |
| `compute_tray`: 0.12 kg for NVMe, fan duct, mounts and leads (d) | no: the build gives it to every compute part |
| Power rows `pi5_ai_hat_plus` 8.0 W and `usb_ssd` 1.5 W (A14) | yes: the dev kit's rows replace them |
| A 5 V supply for the Pi as its own item | not in the record. The catalog part's note names a 5 V, 5 A buck; no ledger row holds one, so there is none to remove |
| A USB SSD as its own item | not in the record. Its 1.5 W is a power row (above); its mass is in no row |
| `power_distribution`: the 0.30 kg power box (d) | no |
| `control_stack`, `imu`, the 4 W MCU row | no: the Teensy stays |

**Not placed: the Jetson-direct option's parts.** A USB-CAN adapter and a supervisor belong to the option in which the
Jetson drives the legs itself. That is not the baseline (P30), so neither is in the model, its mass or its screens.
Table 7 gives what they would add, as estimates.

**Heat is not modelled.** No thermal model of the chest, the electronics or the air path exists in this study. The
drives' heating is read only through their sustained-torque ratios (the envelope screen's `session_heat`; table 4 gives
the session's worst drive). See open item 2.

## 2. How it was built

| File | Change | Why earlier models are unchanged |
|---|---|---|
| `designlab/build.py` | Override `compute_regulator` places the selection's `compute_regulator` part on the power box, after the covers; `compute_regulator_basis` writes its ledger row's basis. `Design.profile_margin` reads an item against the as-built trunk profile | The override is off by default; with it off the new code is not reached. `profile_margin` only reads |
| `designlab/power.py` | `COMPUTE_MODE_BUDGET` (the dev kit's two mode rows), `compute_loads`, and an optional `compute_mode` argument on `aux_power` and `walking_power` | `COMPUTE_BUDGET` is untouched, and a call without a mode books its row as before |
| `designlab/refine.py` | Case `jetson_75` and its functions (`jetson_one` and others). `head_x_at_pick_pose` takes two optional arguments | The new case is one more entry. The two arguments default to what the function did (both `-X` records regenerate unchanged) |
| `designlab/stage1/__init__.py`, `refs.py` | `75-M-H-10-J` registered: `BODIES`, `CANDIDATE_IDS`, `SHA256_PREFIX`, `JETSON_BODIES`, the carpet lift and its record | Entries added; none changed |
| `tests/test_stage2_x.py`, `tests/test_stage1_bodies.py` | Both pin the registry's exact set; each now names the new body | – |
| `round2/refine/refine.json`, `REFINE.md` | Regenerated by the refine CLI, as for every case: the case's label and its two rows added. The label says that the runtime shown (28.7 min) is at round one's 31 W allowance, not at either power mode | No other row changed |
| `stage2/jetson75/budget_75.json` | Regenerated (see below) | Two input hashes changed |

Outside the brief's list of files, each as the `-X` branch did for the same purpose: `tests/test_stage1_bodies.py` (it
pins the registry's exact set and fails otherwise), `round2/refine/refine.json` and `round2/refine/REFINE.md`.

- **Every earlier model keeps its recorded `xml_sha256`.** `tests/test_stage2_j.py` rebuilds the seven bodies the
  registry held before (the three picks, the two `-K` and the two `-X`) and round two's own records of the two picks'
  candidates, byte for byte. With the override at 0 and the regulator's key present, the build is still the pick's.
- **With the compute swap undone, the model is the pick's.** The Jetson record differs from the pick's in three
  entries: the compute part, the regulator's part and its override. Put back, the build gives the pick's sha256.
- **The model is built with one BLAS thread**, as the expressive head's were (`run_j.sh` exports it). The XML does not
  depend on that; the model ledger JSON's last digits do.
- **Hashes that move**, as they did for the `-X` branch. `stage1/__init__.py` and `stage1/refs.py` are inside all four
  of Stage 1's source hashes, and `build.py` inside the panels'. This branch against its merge base (`feb84ad`), first
  12 digits:

  | Hash | What reads it | Before → after |
  |---|---|---|
  | `panels.panel_sources` | Every panel record's key. A panel worker refuses to start on other sources than its session's | `6a316fc82bf5` → `4583a1876f54` |
  | `tune.tuner_sources` | The fit's sessions. `tune.g5` refuses to judge a fit whose sessions ran on other sources | `1d93654822ff` → `36268fb6c66d` |
  | `speedfit.speed_sources` | The speed fit's sessions, with the same refusal | `93a968124ee8` → `d712aeb49367` |
  | `plant.sources_sha256` (`plant.provenance`) | Every run record's provenance; recorded, not gated | `cc2a5c418b94` → `9ee925ef3e8d` |

  So a fit or a panel session started before this branch is merged cannot be resumed or judged after it: **merge between
  sessions, not during one.** No committed record is rewritten. On the base branch's tip (`89111c3`) no committed
  record holds the old panel, tuner or speed-fit value. Five records of the `-X` package hold the old
  `plant.sources_sha256` (the two bodies' `plant.json` and `gates.json`, and `x_costs.json`): it is their provenance,
  and no gate or test compares a committed record's value with today's code. The merged tree's values are its own:
  they also depend on what else the merge brings.
- **`stage2/jetson75/budget_75.json`** hashes `build.py` and `refine.py`, so it was regenerated with this branch's
  `stage2/jetson75/budget_75.py`: against the merge base, those two hashes in its `inputs` changed and nothing else.
  **At the merge this file needs care.** The base branch has since rewritten `budget_75.py`, its record and its test
  (P31), so this branch's copy is the older script's output, and it is the one file both sides changed. Take the
  three-way merge result for the file (neither side's copy), then on the merged tree run
  `python stage2/jetson75/budget_75.py` and `python -m pytest tests/test_stage2_budget75.py`, and commit the record if
  it changed. A trial merge with the base tip `89111c3` (`git merge-tree`, nothing checked out) has no conflict. Its
  `budget_75.json` is the base's file with this branch's two hashes; in a scratch copy of that tree the base's script
  regenerates it byte for byte and the test passes (17 passed).

## 3. What a checkout needs

A merge does not bring the git-ignored inputs. `j_inputs.json` lists all 30:

- the case's model files, `round2/refine/jetson_75/models/75-M-H-10.{xml,json}`;
- `stage2/j/75-M-H-10-J/refs/*.npz` (10) and `stage2/j/75-M-H-10-J/transitions/*.npz` (18).

After the merge, from `hardware/trex_form_study`, with this worktree still there:

```
bash stage2/j/run_j.sh copy-from /home/user/mesozoic-wt/jetson-body
bash stage2/j/run_j.sh check
```

Without the worktree, `bash stage2/j/run_j.sh run` regenerates the same files and checks them against the list
(section 8 gives the times). This was tried: with the two model files, one reference and one transition moved away,
`j_inputs.py models` and `arrays` rebuilt all four to their listed sha256.

**For the restore tool** (`tools/restore_inputs.py`, which this package does not edit): a group `j` beside `x`.

- Helper: `stage2/j/j_inputs.py`, list: `stage2/j/j_inputs.json` (keys `rule`, `bodies`, `cases`, `files`, as
  `x_inputs.json`).
- The helper has the `-X` helper's interface: the functions `model_paths(body)`, `recorded(body)`, `listed()` and
  `check()`, and the commands `check`, `copy-from DIR`, `models`, `arrays BODY refs|transitions`, `records` and
  `manifest`. So `restore_helper("j", procs, scratch)` drives it as it drives `x`.
- BLAS threads: one, the tool's default for helpers.
- It needs no other group's files: `models` reads only the pick's committed record, and `arrays` reads the case's own
  models.
- `tools/README.md` and `tests/test_restore_inputs.py` list the groups, so they need the new one too.

## 4. The tables

<!-- j_tables.py: begin -->
**Table 1. The two bodies**

|  | `75-M-H-10` (the pick, Pi 5 set) | `75-M-H-10-J` | Kind |
|---|---|---|---|
| Model `xml_sha256` | `0517e4d3…` | `fc3cf6b4…` | a |
| Refine case and record | `feet_75` (sized) | `jetson_75` (built once from that record, not re-sized) | a |
| Compute part | `rpi5_8gb_ai_hat_plus_13tops`: 0.100 kg, 85 × 56 × 30 mm, $290 | `nvidia_orin_nano_super_devkit`: 0.174 kg, 103 × 90.5 × 34.77 mm, $399 | c (catalog) |
| Compute tray's envelope, in the chest | 102 × 76 × 60 mm | 120 × 110.5 × 64.77 mm | a (the board plus 17 × 20 × 30 mm, `designlab/build.py`) |
| Compute tray's allowance | 0.120 kg | 0.120 kg | d (PR #662's allowance, any compute part) |
| 6S-to-12 V regulator | none placed | `pololu_d42v110f12`: 15 g, 31.8 × 43.2 × 9 mm, $69.95; on the power box, 5 mm above it | c (catalog); its place: a |
| Regulator's standoffs, 12 V lead and plug | none | 10 g | d |
| Control stack (Teensy 4.1, CAN × 2, IMU) | 0.180 kg | 0.180 kg, kept | d (PR #662's allowance) |
| Drives, pack, bus, structure, covers | 8 drive groups, `lipo_6s_3300`, 24 V | the same (8 of 8 drive groups the pick's, `lipo_6s_3300`, 24 V) | a |
| Hip height (the body unit B) | 0.5019 m | 0.5019 m | a |

What the regulator's choice costs (d). `pololu_d36v50f12` covers both modes on the rail budgets alone: yes (33 % and 52
% of its rating; 89 % and 107 % with the USB ports' 30 W). With it the body would be 8.0 g lighter, 8.5 g with the
reserve. If the power box already holds the computer's regulator, the body is 26.6 g lighter than modelled: the placed
row's 25.0 g with its reserve.

**Table 2. The ledger rows that differ**

| Ledger row |  | Pi body | Jetson body | Change | Centre in the trunk frame, mm (x, y, z) | Kind |
|---|---|---|---|---|---|---|
| `compute` | changed | 100.0 g | 174.0 g | +74.0 g | 313.8, 0.0, -12.8 | c (catalog masses) |
| `compute_tray` | changed | 120.0 g | 120.0 g | +0.0 g | 313.8, 0.0, -12.8 | a (its box grows with the board; the mass is the allowance) |
| `compute_regulator` | added | 0.0 g | 25.0 g | +25.0 g | 181.2, 0.0, -42.2 | c (15 g) + d (10 g) |
| `integration_reserve` | changed | 891.3 g | 897.6 g | +6.3 g | 96.6, 0.0, 10.7 | d (6 % of installed mass) |
| **Sum of the rows** |  |  |  | **+105.3 g** |  | sum |
| **The two models' masses** |  | 14.855 kg | 14.960 kg | **+105.3 g** |  | a |

Centre of mass at the pick's home pose, x forward, y left, z up: (+1.636, 0.000, +0.484) mm in the model, (+1.636,
0.000, +0.484) mm from the rows above (a). The ledgers' own masses differ by +105.319 g; the models' masses are written
to six digits.

The ledger's own sum by basis (`mass_by_basis`): sourced 6.092 kg on the Pi body and 6.092 kg on the Jetson body. The
regulator's 25 g row is not booked as sourced: its basis reads "partially_sourced; catalog part 15 g (sourced) + 10 g of
standoffs, 12 V lead and plug (estimate)".

**Table 3. Against the Pi body**

| Reading | Pi body | Jetson body | Change | Kind |
|---|---|---|---|---|
| Mass | 14.855 kg | 14.960 kg | +105.3 g | a |
| Centre of mass ahead of the hip axis, each body at its own home | 67.07 mm | 69.01 mm | +1.94 mm | a |
| The same, both at the pick's home (the changed parts alone) | 67.07 mm | 68.71 mm | +1.64 mm | a |
| Centre of mass height | 435.90 mm | 436.40 mm | +0.50 mm | a |
| Home pose: largest joint change (the stance is re-solved) | – | left ankle pitch +0.0045 rad | trunk pose change 0 | a |
| Crown height; overall width | 750.80 mm; 313.44 mm | 750.80 mm; 313.44 mm | 0.000 mm; 0.000 mm | a |
| Hip-pitch hold (share of the drive's sustained torque) | 8.687 N·m (0.762) | 9.012 N·m (0.791) | +0.324 N·m | a |
| The same at the pick's home | 8.687 N·m | 9.012 N·m (0.791) | +0.324 N·m | a |
| Knee hold | 2.793 N·m | 2.649 N·m | -0.144 N·m | a |
| Hip-pitch ratios to the drive's planning limits: continuous / peak / speed | 0.762 / 0.493 / 0.834 | 0.791 / 0.502 / 0.830 |  | a |
| Governing ratio of any drive (the sizing walks, Froude 0.05 included) | knee speed 0.935 | knee speed 0.945 |  | a |
| Drive groups over a planning limit; groups one selection step would change | none (sized) | 0; 0 |  | a |
| Dev kit's tray to the power box | 29.7 mm | 20.7 mm |  | a |
| Compute board to the power box | 38.2 mm | 29.2 mm |  | a |
| Tray to the hip-roll motors, front / rear | 181.9 mm / 264.9 mm | 172.6 mm / 255.6 mm |  | a |
| Power box to the hip-roll motors, front / rear | 65.3 mm / 144.3 mm | 65.3 mm / 144.3 mm |  | a |
| Tray to the spine beam | 20.7 mm | 17.0 mm |  | a |
| Tray to the neck servo | 68.9 mm | 59.6 mm |  | a |
| Tray to a ToF sensor | 40.1 mm | 37.7 mm |  | a |
| Regulator to the power box (its standoffs) | – | 5.0 mm |  | a |
| Regulator to the tray; to the compute board | – | 59.8 mm; 68.3 mm |  | a |
| Regulator to the hip-roll motors, front / rear | – / – | 92.8 mm / 175.7 mm |  | a |
| Regulator to the spine beam; to a pelvis plate; to the control stack | – | 78.2 mm; 55.4 mm; 150.1 mm |  | a |
| Compute board inside the trunk profile: dorsal / ventral / half-width | 126.1 / 78.1 / 58.9 mm | 123.2 / 74.6 / 39.9 mm |  | a |
| Compute tray inside the trunk profile (the larger envelope) | 110.6 / 62.1 / 47.2 mm | 107.6 / 58.6 / 28.0 mm |  | a |
| Regulator inside the trunk profile | – | 153.9 / 64.0 / 37.3 mm |  | a |
| Component overlaps on one body at home (the screen's gate) | 0 | 0 |  | a |
| Battery removal: clear directions; least clearance on the way out | -z; 2.55 mm | -z; 2.55 mm |  | a |
| ToF floor window ahead of the toes (each sensor; 64 of 64 zones on the floor) | 0.2754 to 1.1100 m | 0.2735 to 1.1081 m |  | a |
| ToF zones on the robot itself | 0 | 0 |  | a |
| Foot path covered by the two ToF sensors | 0.6484 | 0.6523 |  | a |
| Camera view blocked by the robot at home; nearest floor it sees ahead of the toes | 0.0425; 2.3950 m | 0.0425; 2.3931 m |  | a |
| Cost model's nominal total (the compute line at the catalog price; no regulator line) | $18,521 | $18,638 | +117 $ | d (cost model, A8) |

The trunk-profile margins are each item's corners against the as-built profile with the build's 8 mm + 2 mm allowance
(positive: inside). The build's own profile rule (`Design._required_profile`) reads class A items only, and the tray is
class B, so the build does not check the tray against the profile: the tray's row above is that check. Its smallest
margin on the Jetson body is 28.0 mm (47.2 mm on the pick).

**Table 4. The round-two screens, gate by gate**

The screened model is the sized one: pass (`fc3cf6b4…`). Every row is a prescribed-walk or static reading (a) but the
carpet screen's standing, which is a simulated run (b).

| Screen | Gate or reading | Pi body | Jetson body | Jetson body's verdict | Kind |
|---|---|---|---|---|---|
| clearance (walking gate) | No interference at home | 0 | 0 | pass | a |
| clearance (walking gate) | No interference over the nominal walk | 0 | 0 | pass | a |
| clearance (walking gate) | No interference over the brisk walk | 0 | 0 | pass | a |
| clearance (walking gate) | No component overlap on one body | 0 | 0 | pass | a |
| clearance (walking gate) | Joint-range margin at least 0.10 rad | left hip roll 0.1689 rad | left hip roll 0.1691 rad | pass | a |
| clearance | Swing clearance at least 0.04 H | 25.09 / 26.05 mm | 25.09 / 26.05 mm | pass | a |
| clearance | Tail at least 0.25 H above the floor | 0.8151 H | 0.8151 H | pass | a |
| clearance | Battery has a clear way out | -z | -z | pass | a |
| clearance | Every gate (`pass.all`) | pass | pass | pass | a |
| sensors | Readings, no gate of its own: a person's torso in view at home, straight ahead and at ±30° | yes | yes | readings (more in table 3) | a |
| turning | Target turns (0.35 rad/s walking, 0.5 rad/s in place): within limits, no interference, IK | worst ratio 0.767; thigh twist 0.773 of its range; grip needed 0.409 | worst ratio 0.791; thigh twist 0.776 of its range; grip needed 0.405 | pass | a |
| turning | Closest approach in the target turns, walking / in place | 4.00 / 3.98 mm | 4.00 / 3.95 mm | pass | a |
| turning | Stretch turn (0.5 rad/s walking) | pass | pass | pass | a |
| carpet | Friction below the carpet band's lower edge and every pile top cleared in mid-swing | toe margin 11.00 mm; friction margin 0.300 | toe margin 11.00 mm; friction margin 0.300 | pass | a |
| carpet | Stands, servo-held, on the hard floor and four carpets (a forward simulation) | 5 of 5 | 5 of 5 | pass | b |
| carpet-lift pass | Carpet swing at 0.18 H: target met on every pile, drives within limits, walking gates | drives within limits (knee 0.969); 31.07 mm above a pile top | drives within limits (knee 0.974); 31.07 mm above a pile top | pass | a |
| carpet-lift pass | One step lower, 0.175 H | target met on every pile: no; drives within limits: yes (knee 0.9533) | target met on every pile: no; drives within limits: yes (knee 0.9589) | as the pick's: 0.18 H is the rule's lift | a |
| carpet-lift pass | One step higher, 0.19 H (not used) | target met on every pile: yes; drives within limits: yes (knee 0.9997); walking gates: no | target met on every pile: yes; drives within limits: no (knee 1.0051); walking gates: no | not the rule's lift on either body | a |
| envelope | Froude 0.05 walk: drives within limits, reference CoP on the pad, IK | knee 0.935 | knee 0.945 | pass | a |
| envelope | Readings, not gated: one-foot stop; one-step push bound; session's worst drive | 1.414 m/s²; 0.5885 m/s; neck 0.662 | 1.412 m/s²; 0.5845 m/s; neck 0.662 | readings | a |

**Table 5. Power rows and runtime**

| Compute row | At the compute rail | Drives (nominal walk) | Walking power at the pack | Runtime | Against the Pi body | Kind |
|---|---|---|---|---|---|---|
| Pi body, `rpi5_8gb_ai_hat_plus_13tops`: 8.0 W + 1.5 W SSD (A14) | 9.5 W | 69.42 W | 89.77 W | 36.7 min | – | d (survey's typical draw, unmeasured) |
| Jetson body, 15 W mode: 15 W cap + 3 W carrier, storage, cooling | 18.0 W | 70.28 W | 100.28 W | 32.9 min | -3.9 min | c (the caps) + d (the 3 W); draft A19 |
| Jetson body, 25 W mode: 25 W cap + 3 W | 28.0 W | 70.28 W | 111.64 W | 29.5 min | -7.2 min | c (the caps) + d (the 3 W); draft A19 |
| Jetson body, the record's own `power`: 25 W + 6 W (A13, round one's allowance) | 31.0 W | 70.28 W | 115.05 W | 28.7 min | -8.1 min | d (A13) |
| For comparison only: the survey's typical figure for this kit, 12 W + 3 W. Not a measurement in the 15 W mode: the 12 W was measured under MAXN SUPER clocks and a heavier load (arXiv 2609.13551), and the survey has none in the 15 W mode | 15.0 W | 70.28 W | 96.87 W | 34.0 min | -2.7 min | d (survey; a typical figure, as the Pi row is) |

Every row adds the microcontroller's 4 W and the sensors' 4.4 W, passes the sum through the 0.88 converter (A6) and uses
75 % of the `lipo_6s_3300` pack's 73.26 Wh (A6): `designlab/power.py`, as every round-two record. The drives' power is
the model's (a); the runtime inherits the estimates (d). Both modes meet the 20 minute target: yes.

**Table 6. Stage 1 gates**

| Gate | Verdict in `gates.json` | Readings | Kind |
|---|---|---|---|
| G1 plant | provisional (not judged) | stands 20 s on hard floor and low pile (tilt max 0.49 / 0.50 deg); hashes, motor map and repeatability pass; step rate 20,783 steps/s alone, 19,837 per process on 2 of 4 cores (lowest sample 19,257); the rule asks 20,000 with one process per core, and on these 2 processes the samples are **below it** | b (the stands, repeatability); the step rates are timings of this machine, not model results |
| G2 references | pass | every reference matches its records (True); nominal panel median 60.0 strides (20/20 runs of at least 10); unperturbed 60.0 strides with the envelope, 60.0 without | a (the references against their records); b (the panel and the replays: forward simulations) |
| G3 transitions | pass | 18/18 pass; worst CoM 0.312 mm (brisk_to_fr05), worst ZMP 1.093 mm (fr05_to_brisk) | a (prescribed transitions against the periodic walks) |
| The record's `pass` | provisional (not judged) | G1: step rate timed on 2 of 4 cores (PYTHON_CPU_COUNT=2) | – |

The CLI's own summary of that record (`python -m designlab.stage1 gates`), first words of each line:

- `G1 provisional (step rate timed on 2 of 4 cores (PYTHON_CPU_COUNT=2)): hashes True; hard stands (tilt max 0.49 deg), low stands (tilt max 0.50 deg); t…`
- `G2 pass (rule panel; ankle gains joint, armature linkage): references match True (nominal 0.24306 m/s, CoP margin 0.045303 m); nominal panel median 60…`
- `G3 pass: 18 transitions; worst CoM 0.312 mm (brisk_to_fr05), worst ZMP 1.093 mm (fr05_to_brisk); failed: none (fall back to amplitude ramps: none)`

**Table 7. Not placed: the Jetson-direct option's parts**

| Part of the Jetson-direct option | Catalog | Would add | With the model's 6 % reserve | Kind |
|---|---|---|---|---|
| USB-CAN adapter, `pcan_usb_fd`, one per CAN bus | 68 g each (c, catalog); $368 each (c) | 68 g for one bus, 136 g for the study's two | 72.3 g, 144.7 g | d (estimate; not placed) |
| Supervisor | no catalog part | no figure | no figure | – |
<!-- j_tables.py: end -->

## 5. Results against the Pi body

Table 3 has every figure. In short, all (a) unless marked:

- **Mass: +105 g.** That is 74 g for the board (c), 25 g for the regulator and its lead (c + d), and 6 g of the model's
  6 % integration reserve on both (d). PR #668 reported +78.7 g, which is the board with its reserve and no regulator.
- **Centre of mass: 1.6 mm forward and 0.5 mm up** with both bodies at the pick's home pose. The build re-solves the
  stance, which moves the feet under the new centre of mass: each body at its own home, the centre of mass is 1.9 mm
  further ahead of the hip than the pick's.
- **Hip-pitch hold: 8.69 → 9.01 N·m**, from 0.76 to 0.79 of the drive's sustained torque. It is the load that moves
  most. The same drives stay within every planning limit, and one selection step on the new loads would change none.
- **The governing limit is still the knee's speed** on the Froude 0.05 walk: 0.935 → 0.945 of the planning limit.
- **The exterior is the pick's**: crown, widths and covers unchanged. The trunk profile is not widened for the dev kit.
- **At the chest:**
  - the dev kit's tray clears the power box by 20.7 mm, 9.0 mm less than the Pi's tray, because the tray is 18 mm
    longer. This is PR #668's figure, now from a recorded body;
  - on this body the power box does not move. The build puts its centre at 0.22 B plus half its length, or 10 mm behind
    the tray if that is further back; at 75 cm the first holds for both boards;
  - the hip-roll motors are 173 mm and 256 mm behind the tray. PR #668 reported a conflict at 60 cm (its record, not
    rerun here); nothing of the kind arises on this body;
  - the tray clears the spine beam by 17.0 mm (20.7 mm before);
  - the tray, the larger envelope, is inside the as-built trunk profile by 28.0 mm at its tightest, the half-width
    (47.2 mm on the pick). The build does not check this itself, because its profile rule reads class A items and the
    tray is class B: the record's reading is the check (table 3's note);
  - the regulator clears the power box by its 5 mm standoffs and everything else by 55 mm or more, and sits 37 mm or
    more inside the trunk profile.
- **The battery's removal path is the pick's**: downward only, with 2.55 mm to spare.
- **The sensors' views are the pick's.** Both ToF sensors keep all 64 zones on the floor and none on the robot. The
  floor window moves 2 mm, with the feet. The camera's blocked share and its nearest visible floor are unchanged to
  2 mm.

## 6. The screens

Every round-two screen was run on the body by the stock path (`python -m designlab.refine --cases jetson_75`, which
calls `round2.screen_one`), then the carpet-lift pass with the pick's lifts. Table 4 gives each gate beside the pick's.

- **Clearance: every gate passes**, the five walking gates among them.
- **Sensors:** the screen has no gate of its own; its readings are in table 3.
- **Turning: the target turns and the stretch turn pass.** The worst drive ratio in the target turns rises from 0.767
  (thigh twist, continuous) to 0.791 (hip pitch, continuous): the hip-pitch hold now governs there.
- **Carpet: passes**, with the pick's toe and friction margins to the second decimal.
- **Carpet swing: the pick's lift, 0.18 H, by the pick's rule.** The knee's speed ratio there is 0.974 (0.969 on the
  pick), so the carpet walk has 2.6 % of knee speed left where the pick has 3.1 %. At 0.19 H the Jetson body's knee
  goes over its limit (1.005, against 0.9997); that lift fails the walking gates on both bodies and is not used.
- **Envelope: the Froude 0.05 walk stays within the drives' limits.** The one-step push bound falls from 0.589 to
  0.584 m/s, and the one-foot stop from 1.414 to 1.412 m/s².

## 7. Power rows and runtime

`designlab/power.py` gains two rows for the dev kit, one per power mode, beside the existing ones. Nothing existing
changes: the dev kit's 31 W row (A13) and the Pi set's 9.5 W row (A14) are as they were, and the pick's recorded power
and runtime recompute (`tests/test_stage2_j.py`).

| Row, at the compute rail | Figure | Source | Kind |
|---|---|---|---|
| 15 W mode, module | 15 W | The mode's cap: catalog `nvidia_orin_nano_super_devkit` (`power_typ_w`), NVIDIA's power modes | c |
| 25 W mode, module | 25 W | The mode's cap: the same part (`power_max_w`) | c |
| Carrier, NVMe and fan, either mode | 3 W | The top of the compute survey's 2–3 W, which it calls unmeasured (`compute_2026-11-07.json`, architecture `A_orin_nano_super_devkit`) | d |
| 15 W mode, total | 18.0 W | The survey: "Max 18.0 W = 15 W mode cap (vendor) + 3 W" | c + d |
| 25 W mode, total | 28.0 W | The survey: "25 W mode (the Super configuration's default): 25 + 3 = 28.0 W at the board" | c + d |
| Microcontroller, sensors, converter, usable energy | 4 W, 4.4 W, 0.88, 75 % | A6 and the round-two records, unchanged | d |

- **Each mode is booked at its cap.** Nobody has measured the kit at the converter input in either mode (`COMPUTE.md`
  bench test 1).
- **Runtime on the pick's 3,300 mAh pack** (table 5): **32.9 min in the 15 W mode and 29.5 min in the 25 W mode**,
  against 36.7 min for the Pi body (d; the drives' power in them is the model's, a). Both stay above the 20 minute
  target, and neither would change the pack.
- **The comparison is not like for like.** The Pi row is the survey's typical draw; these are caps. At the survey's
  typical figure for this kit, 15.0 W, the model gives 34.0 min (d). That figure is not a measurement in the 15 W mode:
  its 12 W at the module was measured with locked MAXN SUPER clocks under a heavier detector load (arXiv 2609.13551),
  and the survey says "No light-load measurement exists in the 15 W mode".
- **The record's own `power` and `runtime` are at round one's 31 W** (28.7 min), because every record books its compute
  part at `COMPUTE_BUDGET`'s row. Read the mode rows in `jetson.power_modes` instead.
- **One converter efficiency, 0.88, for every rail** (A6). The vendor gives over 90 % for the regulator placed (c), so
  0.88 is the cautious one. PR #668 used 0.93 for this rail.
- **The heavier body costs the drives 0.9 W**: 69.4 → 70.3 W on the nominal walk (a).

## 8. Stage 1 inputs and gates: G1 is provisional

The body is registered like the `-X` bodies, and its references and transitions were prepared with the stock command
(`python -m designlab.stage1 prepare`, through `run_j.sh run`) into `stage2/j/75-M-H-10-J/`. Table 6 has the gates.

- **G2 and G3 pass**, by the CLI's own summary: every reference matches its records and the nominal panel walks 60
  strides in 20 of 20 starts; all 18 transitions meet G3's limits.
- **G1 is provisional, not a pass.** Its step-rate rule times one process per core. This branch ran beside two other
  builds and was held to 2 processes, so the rate was timed on 2 of the machine's 4 cores and is not judged
  (`gates.PROVISIONAL_RATE_RULE`; `HEAD_X.md` section 3). Every other G1 check passes.
- **On those 2 processes the rate was under the rule's 20,000 steps/s**: 20,783 alone and 19,837 per process, with the
  machine shared (the record's `ok_on_these_processes` is false). The pick's own G1 record reads 25,419 per process
  with one process per core (8 Nov), and `75-M-H-10-X`'s 22,258 on two processes. The Jetson model has the pick's
  joints and contacts and one more box that collides with nothing, so this reads as the machine's load, but that is
  not shown.
- **To finish G1:** on an otherwise idle machine, run `PY=<the venv's python> bash stage2/j/run_j.sh g1`. It may fail
  on the step rate alone, as arm K's did on this machine (Stage 1b decision 11).
- G4, the tracker's gate, was not run.
- **One test failed in a run of other test files, and not through this package.** `logs/j_test_others.log` ends
  "1 failed, 39 passed": `tests/test_carpet.py::test_round_one_60cm_toe_catch_friction_standing_and_shortlist_hook`.
  The cause is the order of the files in that run. `tests/test_stage2_x_costs.py` (line 92) sets the walks' accuracy to
  round two's with `gait.set_accuracy` and does not put it back, so a carpet test run after it in the same process
  reads the wrong accuracy. Reproduced: those two together give "1 failed, 9 passed"; the carpet test alone passes, and
  it passes after `tests/test_stage2_j.py`, which puts the accuracy back. The usual alphabetical order hides it. The
  fix belongs in `tests/test_stage2_x_costs.py`, the `-X` package's test, which this package does not edit.
- **Provenance.** The run's records carry commit `c446ecc`, which holds all the code that ran. Their `dirty` flag is
  set only by files that were untracked then (the run's own outputs, this document, its tables script and the tests);
  no tracked file was changed.
- **After the review** the regulator's ledger basis and three texts of the record changed, so the case was run again
  with the stock command on the final code (`python -m designlab.refine --cases jetson_75`, one BLAS thread):
  - the model is the same, `xml_sha256` `fc3cf6b4…`;
  - the candidate record differs in its run time and in three entries of its `jetson` block: the heat note, the rule
    of the survey's typical row, and the regulator's new `consequence`;
  - the screens gave the same results. The index, the envelope record and the four plots came out byte for byte; the
    other four records differed in their timing fields alone, and the committed ones were kept;
  - the screen's ledger (`review/mjcf/75-M-H-10.json`) differs in the regulator row's basis and the sum by basis;
  - the model ledger JSON's sha256 in `j_inputs.json` moved with the basis (`7404d897…` → `2447c288…`), and
    `j_inputs.py models` rebuilds both model files to the list from nothing;
  - Stage 1's records and arrays are untouched: they depend on the XML and on `designlab/stage1`, and neither changed.

Times on this machine, at low priority beside two other builds: the case's build and screens about 7 min; the
carpet-lift pass 2 min; the plant and references 3 min; the transitions 14 min on 2 processes; G1 to G3 3 min.

## 9. Open items

1. **G1 with one process per core** (section 8).
2. **Heat.** The chest, the electronics and the air path have no thermal model (section 1). The dev kit's module
   dissipates 15–25 W (18–28 W at the rail, c + d) where the Pi set dissipates 8–9.5 W (d), in a closed PA11 chest,
   and the regulator adds its own loss in the abdomen. The survey sized the tray's airflow for 31 W and notes that
   the stock fan needs a ducted inlet and outlet (d). This is a hardware question for a thermal test, not for this
   model.
3. **The mode budgets are caps, not measurements** (draft A19, section 10). `COMPUTE.md` bench test 1 replaces them.
4. **No closed-loop comparison with the Pi body.** The review's task 6 is done "when the body passes its gates, and a
   paired-seed table against the Pi body reports each difference". The screens and G2–G3 are done; the paired-seed
   panels are not, and no controller was run on this body beyond the gates' own replays.
5. **The power box's contents.** Whether its 0.30 kg already holds a 12 V regulator for the computer is not
   established (section 1), so the regulator may be counted twice.
6. **The regulator** (section 1). Its rating at 6S: the catalog has one point, at another input voltage. And the
   worst case that chose it: check the rating of the kit's own 19 V supply. If it is below 54 W, the kit is not built
   to draw the 58 W worst case at its jack, and the smaller part (`pololu_d36v50f12`, 54 W) covers both modes; the body
   would then be 8.5 g lighter (table 1's note).
7. **Masses that are allowances or forum figures:** the dev kit's 0.174 kg (c, a forum figure), the tray's 0.12 kg and
   the regulator's 10 g of lead and standoffs (d). An NVMe drive has no row of its own. One row of the pick has the
   same mix that the regulator's row now declares: the camera's 101 g is its catalog mass plus an estimated 40 g
   mount, booked under the part's basis, `sourced` (`designlab/build.py`). That row is the pick's and is not changed
   here.
8. **Not modelled:** the wiring, the fan duct's shape and air path, and the Jetson-direct option's adapter and
   supervisor (table 7).
9. **Not built:** a body with both the dev kit and the expressive head (`-XJ`), and a 60 cm Jetson body. PR #668
   reported (its record, not rerun here) that at 60 cm the power box meets a hip-roll motor; that packaging is neither
   checked nor solved here.
10. **Before any training on this body:** `env.PUSH_BOUND_M_S` lists the three picks only, and the PPO env falls back to
    0.5 m/s for any other body. This body's own bound is 0.584 m/s (a). Stage 1's margin ledger and Stage 2's look
    ladder do not list it either.
11. **The cost model has no regulator line.** The record's cost moves by the compute line alone;
    `stage2/jetson75/BUDGET_75.md` prices the regulator both ways.

## 10. Draft decision-log row (A19)

A draft for `DECISION_LOG.md`'s assumptions table; it is not in the log. The runtimes are table 5's.

| # | Assumption | What would replace it |
|---|---|---|
| A19 (draft) | Jetson body (`75-M-H-10-J`, refine case `jetson_75`): the Jetson Orin Nano Super dev kit is booked at its power mode's cap plus 3 W for the carrier, the NVMe and the fan, at the compute rail before the 0.88 converter: **18.0 W in the 15 W mode and 28.0 W in the 25 W mode** (`power.COMPUTE_MODE_BUDGET`). The caps are NVIDIA's (catalog `nvidia_orin_nano_super_devkit`); the 3 W is the top of the compute survey's unmeasured 2–3 W. The microcontroller's 4 W and the sensors are as in A14. No mode is chosen (P30). With these, the model gives 32.9 min (15 W mode) and 29.5 min (25 W mode) of walking on the 3,300 mAh pack, against the Pi body's 36.7 min. The Pi figure rests on a typical draw (A14) and these on caps; at the survey's typical figure for this kit (15.0 W, whose 12 W at the module was measured under MAXN SUPER clocks: the survey has no measurement in the 15 W mode) the model gives 34.0 min. A13's 31 W row and A14's 9.5 W row are unchanged, and the Jetson record's own runtime (28.7 min) is still at A13's 31 W. The chest and the electronics have no thermal model. | `COMPUTE.md` bench test 1: the dev kit measured at the converter input and with `tegrastats`, running the real pipeline in the 15 W and the 25 W mode; and bench test 9 for the real converter efficiency |
