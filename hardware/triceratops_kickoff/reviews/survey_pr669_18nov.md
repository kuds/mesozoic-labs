# PR #669 review: "Propose a 75 cm Triceratops quadruped and 60 cm companion"

Read on 18 Nov 2026. The head is 84a13e1e, with 2 commits (b9797e38, 84a13e1e) on base 0748ac12, and that base is still the tip of origin/main. The PR adds 8 files, all new, all under `hardware/triceratops_design_study/` (+6,517 / −0). I read the PR body, both READMEs, the OpenSCAD header, robot.xml (header, bodies, actuators, sensors, masses), check_models.py, and the full 9-page PDF text.

## Verdict

As a concept study it is sound. The geometry is consistent across the PDF, the OpenSCAD file and the MuJoCo model, and the load arithmetic holds. It can be merged as a study. It is not a build spec: it prices only the drives, sizes the belt only as a placeholder, gives no continuous or thermal analysis, and verifies no walking.

At 20 kg the 1:1 belt on the X6 runs short of margin. Fixing that cleanly needs one of these:
- a 48 V bus with a belt reduction;
- a larger flexion drive;
- holding the robot to 15 kg.

## 1. What it proposes (all from the PR files)

**Size (75 cm)**
- 750 mm to the frill crown; 1,500 mm long; 430 mm wide.
- Front-to-rear stance 650 mm; track 350 mm.
- Shoulder and hip joints 400 mm above the floor; ankle pin at 45 mm.
- Leg links 235 mm + 230 mm.
- Pads 110 × 80 mm; overhang 365 mm in front, 485 mm behind.
- The 60 cm companion is 600 / 1,200 / 348 mm, with stance 520 / 280 mm, joints at 320 mm, links 188 / 184 mm and an 88 × 64 mm sole. It is a "provisional packaging study" only.

**Mass budget (15.00 kg)**

| Item | kg |
| --- | ---: |
| Drives | 6.16 |
| Frame, transmissions, feet and bearings | 2.70 |
| Moving head | 1.00 |
| Skins, tail and tail servo | 0.95 |
| Compute and sensors | 0.75 |
| Battery | 0.70 |
| Power, wiring and cooling | 0.90 |
| Reserve | 1.84 |

- I parsed robot.xml: the geom masses sum to 15.000 kg.
- The static front share comes out at about 55.2 %, so the claimed 55 % holds.

**Actuators per leg (3 axes)**
- **Roll and upper-leg pitch:** MyActuator RMD-X4-P36-36-E-L, 8 in total.
  - Repo catalog `designlab/data/actuators.json`, accessed 7 Nov 2026: 36:1, 10.5 N·m rated, 30 N·m peak (36 N·m stall), 24 V nominal, 20–55 V window, 111 rpm no-load, 0.36 kg, $269.
- **Elbow or knee flexion:** RMD-X6-P20-60-E, 4 in total, driving a guarded 1:1 belt.
  - The motor sits 90 mm down the upper link; pulley radius 20 mm; 12 mm width allowed; "tooth profile TBD".
  - Same catalog: 19.612:1, 20 N·m rated, 60 N·m peak, **48 V nominal**, 176 rpm no-load, 0.82 kg, $428.
  - The catalog flags that the US store page lists 0.78 kg / 50 N·m, which matches the X6-50-L. Which variant ships for $428 is unconfirmed.
- **Torque limits in the model:** 21 N·m for each X4. For each X6, 37.8 N·m at the joint, which is 42 N·m at the motor after 90 % belt efficiency.
  - My inference: both limits are 70 % of the catalog peaks.
- **Neck:** ROBOTIS XM540-W270-T, spring-assisted pitch, ±10°, model limit 3.0 N·m. Head gravity is at most 0.98 N·m (1.0 kg with its centre of mass within 100 mm of the pivot). Repo catalog: 12 V, $482.89.
- **Tail:** Feetech STS3215 yaw, model limit 1.8 N·m. That is above the repo's planning peak of 1.23 N·m (catalog: 12 V, $22–27.99).
- **Fixed:** jaw and neck yaw.
- **Drive cost:** $3,864 (8 × 269 + 4 × 428), from the same catalog.

**Power**
- 6S 5 Ah pack, 111 Wh; assumes 80 % usable and 150–210 W average draw, giving 25–36 min. I re-derived this range.
- Fuse, disconnect, independent stop circuit, regulated logic rails, voltage and current logging.
- No tether or switchover is mentioned.

**Compute and bus**
- Pi 5 + AI HAT+ for vision and behaviour.
- Teensy 4.1 with two CAN buses, 6 drives each, running the joint and contact control. It falls back to a damped crouch on stale commands.
- The Jetson Orin Nano Super is named as a "studied alternative".

**Sensors**
- Trunk IMU, drive telemetry, 4 foot-load sensors, an OAK-D Lite behind a chest aperture, and 2 downward ToF sensors.
- The models of the IMU, ToF sensors and foot-load sensors are not named.

**Frame and shells**
- Two 6061-T6 rails and cross-members tie the shoulder and pelvic bulkheads.
- Joint shafts are supported separately; limb modules are removable; a dorsal hatch gives access to the compute tray and the aft battery.
- Thin ribbed MJF PA11 panels, soft replaceable horn tips (TPU in the model), urethane soles.

**Costs**
- Only the drives are priced.
- The T. rex turnkey estimate (about $20k: $20,057 with the Pi, $20,174 with the Jetson) is cited as a comparator.
- Machining, feet, shell tooling and integration "need separate quotes".

**Gait**
- Four-beat lateral-sequence walk, 75 % of each cycle on the ground, 1.5 Hz.
- Stride 200 mm, stance sweep 150 mm, swing 167 ms, lift 32 mm.
- Speed target 0.30 m/s, with 0.40 m/s as a stretch.
- Turning circle 1.92 m for a walking turn and 1.62 m in place; the tail sets both. I did not check these.

**Test plan**
1. Full free-base simulation with losses, delay, battery sag and at least 10 mm shell clearance.
2. The T. rex speed test applied at 0.30 m/s:
   - on grippy and slippery hard floors, at 0–10 ms added delay;
   - two independent controllers, each 20/20 upright and at least 17/20 meeting every limit;
   - a 90° turn, a two-step stop, and pushes from four directions at 4 ms.
3. One instrumented foreleg on the bench:
   - the most heavily loaded trajectory, at full and at low pack voltage;
   - with 12 drives' bus traffic emulated at 500 Hz, measuring the p99 command-to-torque delay.
4. A walk of the bare chassis at full mass under an overhead catch, then repeated with the skin fitted.

**MuJoCo models**
- robot.xml: 14 position actuators, passive ankles (3 N·m/rad, ±22°), a 1 ms implicitfast timestep.
- anatomical.xml: a 7.30 m, 5,936 kg adult with 26 torque axes.

## 2. Leg-load calculation: assumptions and my recompute

The script is at `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/scripts/legload.py`.

**What the PDF assumes**
- A rigid, planar, static leg.
- The vertical contact force passes beneath the ankle pin. This is consistent with a passive ankle, which can only carry its small spring moment.
- One forefoot briefly carries the whole front-axle load. This is exactly right for static three-foot support: the lone forefoot carries W × 358/650 whatever the sideways shift.
- The largest elbow lever arm during stance is 179 mm.
- A 1.5× allowance for loads, and 90 % belt efficiency.

**At 15 kg, with the PDF's geometry: reproduced**
- Front axle: 0.55 × 15 × 9.81 = 80.9 N.
- Elbow lever arm against foot position:
  - 150.2 mm in the neutral pose;
  - **179.1 mm with the foot 75 mm ahead at touchdown**;
  - 105.4 mm at lift-off.
- Elbow torque: 80.9 × 0.179 = **14.5 N·m** at the joint; 21.7 N·m with the 1.5× allowance; **24.1–24.2 N·m at the motor**.
- Rear: 66.2 N on the same 179 mm arm gives **19.8 N·m** at the motor. Matches.
- I also reproduced the swing speeds:
  - 69.0 rpm upper pitch and 48.1 rpm flexion at 75 cm (62.4 / 42.5 rpm at 60 cm);
  - about 92° peak flexion.

**What the PDF leaves out**
- **Its figure is conservative for the steady gait.** At 75 % ground time, a forefoot stands alone only between 33 % and 67 % of its own stance, with the foot within ±25 mm of the shoulder. The lever arm there is at most 161.5 mm, giving 13.1 N·m at the joint and 21.8 N·m at the motor.
- **Static balance needs a sideways shift.** With the centre of mass on the centreline 358 mm ahead of the hind feet, the diagonal support line passes 17.8 mm off-centre. The body must shift at least 18 mm, plus margin, before each forefoot lifts. The PDF asks for a shift but gives no number.
- **Loads the calculation does not include:**
  - Fore-aft ground force: at 0.15 × the vertical force, it adds about 2 N·m about the elbow, which sits 174 mm above the ankle.
  - The centre of pressure moving to the pad edge once the ankle reaches its ±22° stop: lever arm up to about 234 mm, about 31.6 N·m at the motor with the same factors.
  - A lower crouch (the fallback pose), body pitch, and turning loads.
- **Continuous loads (none in the PDF):**
  - Four-foot standing: 40.5 N per forefoot, 6.1 N·m at the joint, 6.75 N·m at the motor.
  - A held "paw" pose with one forefoot lifted: 12.1 N·m at the joint, **13.5 N·m at the motor, sustained**.
  - Repo thermal reference: X6-60 rated 20 N·m at 48 V and 153 rpm, with no thermal condition stated. Its sibling, the X6-50-L, reaches thermal equilibrium only at 9–13 N·m.
- **6S costs speed, not low-speed torque** (my estimate, scaling the catalog no-load speeds linearly with voltage):
  - Repo 6S range is 21.0–25.2 V.
  - X6-60 at 6S: about 81 rpm no-load at 22.2 V, about 77 rpm at 21 V. The 48.1 rpm swing uses 59–62 % of that.
  - X4 at 6S: about 103 / 97 rpm. The 69 rpm pitch swing uses 67–71 %.
  - At 0.40 m/s with the same cadence and a 267 mm stride: pitch 91 rpm (88–94 %) and flexion 58 rpm. **The 0.40 m/s stretch is not credible on 6S.**
  - The longer stride also raises the elbow lever arm to 185 mm.

**At 20 kg, same geometry and 55 % front load**
- Front axle 107.9 N, rear 88.3 N.
- Elbow torque by the PDF's method: 19.3 N·m at the joint, 29.0 with the 1.5× allowance, **32.2 N·m at the front motor**; rear 26.3 N·m.
- Margin against the model's 42 N·m limit: 1.30× at 20 kg, against 1.74× at 15 kg.
- If the $428 unit is the 50 N·m X6-50-L, the same 70 % rule gives 35 N·m, so 32.2 N·m is **92 % used**.
- Continuous: 4-foot standing about 9.0 N·m at the motor, already in the X6-50-L equilibrium band; held paw pose about 18.0 N·m.
- Belt pull at the 20 mm pulley, from torque ÷ radius:

| Case | Torque | Belt pull |
| --- | ---: | ---: |
| 15 kg design point | 24.2 N·m | 1.21 kN |
| 20 kg design point | 32.2 N·m | 1.61 kN |
| Model torque limit | 42 N·m | 2.1 kN |

- Run time, my estimate assuming power scales with mass: 200–280 W, so 19–27 min on the 6S 5 Ah pack.
- The X4s stay comfortable at either mass: pitch carries at most about 2–8 N·m statically, plus the fore-aft force terms.

## 3. Verified versus target

**Checked by the PR**
- The standing envelope (OpenSCAD 2021.01 renders) and the planar inverse kinematics.
- The arithmetic, which I reproduced.
- MuJoCo 3.10.0: both XMLs load, and the robot stands on four loaded feet for 3 s at zero command.
- About 55 % of the weight on the forefeet in the simulated stance.

**Targets or estimates**
- The 0.30 / 0.40 / 0.218 m/s speeds.
- The 15 kg mass, the 150–210 W power figure, the torque limits, the servo gains, belt transfer, pad friction and ankle compliance.
- The PR says so itself: free-base walking, clearances through a stride, torque–speed and heat at 6S, belt play, and bus delay are all open.

## 4. Strengths

- **Consistent and honest.** The PDF, OpenSCAD and robot.xml dimensions and masses agree, and the PR labels targets as targets throughout.
- **It finds the critical axis.** Fore flexion is the limiting joint, and the PR says the 48 V rating of the X6 must be re-measured at 6S.
- **The test plan is staged well.** A single foreleg is tested on the bench before four limbs are ordered, and the T. rex purchase-gate method is reused.
- **Easy to service.** Dorsal hatch, removable limb modules, and a spine load path that stays continuous with the shell off.
- **Strong hardware overlap with the T. rex.** The 75 cm T. rex (75-M-H-10, per the PDF's p.8 table) uses 10 × X4-P36-36-E-L + 2 × X6-P20-60-E + XM540-W270-T + STS3215. The Triceratops uses 8 × X4 + 4 × X6 and the same neck and tail parts.
  - It shares the Teensy dual-CAN architecture and the 6061 / PA11 / urethane materials.
  - One X6 + belt + X4 bench serves both robots; the X6 is the T. rex hip pitch.
  - Drive sets at repo prices: $3,864 for the Triceratops, $3,546 for the T. rex.

## 5. Gaps and risks

1. **The 1:1 belt to the X6.**
   - The 12 mm, 40 mm-pulley space allowance looks badly undersized for a 1.2–2.1 kN belt pull. One unverified retailer listing gives 285 N working tension for a 12 mm HTD 5M belt.
   - Pretension adds hub load on top. The X6 output bearing is rated 2 kN radial (repo note), so outboard shaft support is mandatory.
   - The belt is modelled as rigid, so stretch and lost motion are not in the model.
   - A belt reduction would cut torque, but on 6S, 1.5:1 needs 72 rpm, which is at the no-load limit.
2. **6S versus 48 V.**
   - The X6-60 is a 48 V part.
   - The deck says 24 V. Main's own $10k quadruped plan uses 12S.
   - 6S starves swing speed, rules out a belt reduction, and a sagging pack nears the drives' 20 V minimum.
   - 12S (50.4 V full) sits inside the 55 V maximum of both drives, but needs a regen clamp and precharge.
   - The 12 V neck and tail servos need a regulator either way.
3. **Pi 5 versus Jetson.**
   - The PR's baseline is the Pi.
   - The PDF's p.6 plan-view label says "Orin / MCU", which contradicts its own text.
   - WIM Robotics' PLEM is pitched on a Jetson, and the T. rex modelled a +$117 difference. A Jetson keeps one compute platform for both robots.
   - Compute is not the bottleneck; the bus is.
4. **CAN bus load** (repo `stage2/jetson75/LATENCY_AND_BUS.md`).
   - 2 classic buses × 6 drives at 500 Hz: 67–81 % load, and a worst-case exchange of 2.2 ms against a 2 ms cycle.
   - At 1 kHz: 133–162 %, does not fit. 3 buses at 1 kHz: 89–108 %, does not fit.
   - 3 buses at 500 Hz: 44–54 %; the Teensy has a third CAN port.
   - By the same arithmetic (mine), 4 buses × 3 drives at 1 kHz come to 67–81 %.
   - No CAN FD mode was found for these drives, so **the deck's "dual CAN FD" and PLEM's 1 kHz torque loop do not fit the PR's two classic buses.**
   - The repo catalog lists EtherCAT on both drive models; that is unconfirmed for the units that actually ship.
5. **The 55 % front load.**
   - About 1.8 points of it come from the 1.84 kg reserve block placed at x = +118 mm. Without that block the split is about 53.4 % (my parse).
   - Moving the battery barely trims it: about 0.7 points per 100 mm of travel at 15 kg.
   - A heavier, more faithful frill pushes the front load up further. Designing to 50 % would cut the front X6 peak from 24.2 to 22.0 N·m.
6. **No continuous or thermal sizing.** A held paw pose or long standing sessions at public demos are the real duty, especially at 20 kg.
7. **Battery and pack bay.**
   - The repo's 6S 5000 mAh pack (Gens Ace G-Tech) is 0.755 kg and 165 × 45 × 47 mm.
   - The PR budgets 0.70 kg and a 160 × 90 × 65 mm bay, which is 5 mm short in length.
8. **No bill of materials beyond the drives.** No foot-sensor, IMU or ToF parts; no stop-circuit hardware; no tether or switchover, although the deck has both; no spares; no charger or bench supply.
   - The STS3215 and the OAK-D Lite are the low-tier choices against the "not a cheap knockoff" brief. The repo catalog also holds an OAK-D Pro W at $529.
9. **Stale T. rex appendix.** PDF pp.7–9 predate P47–P51: they say the branch "reached the Colab smoke-test handoff", but P50 skipped the smoke and P47 paused training. They also list 12 open PRs as of 18 Nov.
10. **Look of the gait.** A 200 mm stride at 1.5 Hz looks quick and busy for a 1.5 m animal. Longer, slower steps raise the elbow lever arm.
11. **Scope.** The 60 cm companion is a packaging concept that splits attention.

## 6. Merge readiness

- **Mergeable.** GitHub reports `mergeable_state: clean`.
- **No conflicts.**
  - Local `git merge-tree` against origin/main (0748ac12) is clean.
  - Against the pushed T. rex branch it is also clean. That branch's remote head is now **c5274905 (P51)**, not b1d5e98e.
  - There is no path overlap with `hardware/trex_form_study/`, so either merge order works.
- **No tests ran.**
  - There are 0 check runs.
  - The path filter in `python-ci.yml` excludes `hardware/**`.
  - check_models.py is not in pytest's testpaths and needs mujoco==3.10.0. I did not run it.
- **File sizes** are all under the 500 KB pre-commit limit; the largest is preview.png at 304 KB.
- **No reviews or comments** on the PR.
- **Links.** The PR's links pinned to commit 42e6f507 resolve; STAGE2_75_REPORT.md and BENCH_SHEET.md exist there.
- **Fit to merge as a concept study.** Suggested follow-ups:
  - fix the "Orin / MCU" label;
  - note the belt sizing and the 20 kg limits;
  - correct the battery mass and bay length;
  - date-stamp the T. rex appendix as superseded by P47–P51.

## 7. Facts the parts list and the new-session prompt must carry over

- **Base design:** PR #669 (head 84a13e1e, base 0748ac12), 75 cm crown, 1.50 m long, 0.43 m wide.
  - Stance 650 mm, track 350 mm, joints at 400 mm, ankle at 45 mm, links 235 / 230 mm, pads 110 × 80 mm.
  - 15.00 kg budget including a 1.84 kg reserve; 55 % of the weight on the front axle.
- **Drives:** 8 × RMD-X4-P36-36-E-L (roll and pitch) and 4 × RMD-X6-P20-60-E (flexion through a 1:1 belt).
  - Repo catalog, 7 Nov 2026: $269 / $428, 0.36 / 0.82 kg, 30 / 60 N·m peak, 24 V / 48 V nominal, 20–55 V window.
  - The X6 store listing conflicts (50 N·m / 0.78 kg); confirm which variant ships.
- **Neck and tail:** XM540-W270-T for the neck ($482.89, 12 V), STS3215 12 V for the tail; jaw and neck yaw fixed.
- **Load figures at 15 kg** (my check, see section 2):
  - Front elbow at the motor: peak 24.2 N·m (21.8 in the steady gait); rear 19.8 N·m.
  - Continuous: 6.75 N·m standing, 13.5 N·m for a held paw.
  - Belt pull 1.21 kN.
- **Load figures at 20 kg:** 32.2 N·m front, 26.3 N·m rear, 9.0 / 18.0 N·m continuous, 1.61 kN belt pull. The margin against the 42 N·m limit falls to 1.30×.
- **Swing speeds:** 69.0 rpm pitch and 48.1 rpm flexion at 0.30 m/s; 91 / 58 rpm at 0.40 m/s. On 6S the X4 has an estimated 97–103 rpm no-load, so 0.40 m/s is not credible.
- **Static balance:** at least 18 mm sideways shift, plus margin, before a forefoot lifts.
- **Power:** 6S 5 Ah, 111 Wh. The repo's pack is 0.755 kg and 165 mm long, against the PR's 0.70 kg and 160 mm bay. 150–210 W gives 25–36 min (assumed).
- **Bus:** 2 classic CAN buses × 6 drives at 500 Hz is 67–81 % load.
  - 1 kHz needs at least 4 buses or EtherCAT.
  - No CAN FD mode was found for these drives.
  - The neck and tail servos need their own serial buses.
- **Compute:** the PR's baseline is Pi 5 + AI HAT+ ($290) with a Teensy 4.1. The alternative is a Jetson Orin Nano Super ($399, plus a 6S-to-12 V regulator at $39.95–69.95), which matches PLEM and the T. rex.
- **Sensors:** OAK-D Lite ($269, 0.061 kg), 2 downward ToF, 4 foot-load sensors (part not chosen), trunk IMU (part not chosen).
- **Open choices:** 6S vs 12S; belt profile and width or a reduction; Jetson vs Pi; a third or fourth CAN bus; 15 vs 20 kg; the stop-circuit, tether and switchover parts; spares.
- **Gates already written:** single-foreleg bench before buying four limbs; T. rex-style panels at 0.30 m/s, 0–10 ms delay, two controllers, 20/20 upright and at least 17/20; turn, stop and push tests at 4 ms.
- **Shared with the T. rex:** the same two drive models and the same neck and tail servos (75-M-H-10 = 10 × X4 + 2 × X6), Teensy dual-CAN, 6061 / PA11 / urethane, and the X6 + belt bench.
- **T. rex timeline:** the T. rex is paused (P47) and kept as the flagship for end 2027 / early 2028 (P51). Data branch `claude/mesozoic-trex-stage2-bundle-s2-b1` must not move (P50). The T. rex branch head is c5274905.
- **Not verified anywhere:** free-base walking, belt rating, 6S torque–speed and heat, bus delay, the turning circle, any cost beyond the drives.

## Not done

- I did not run check_models.py: MuJoCo is not installed here, and the rules say no simulation runs.
- I did not render the OpenSCAD file, recompute the turning circle, read anatomical.xml in full, or look at the PDF drawings (I used the text only).
- I did not open a belt maker's table: `www.tyma.eu` is outside the session's allowed network, and the session settings would need to allow it.
- I did not check live vendor prices; all prices are the repo catalog's, accessed 7 Nov 2026.
- No comments, reviews, pushes or file writes in the checkout.

## Files

- Extracted PR files: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/pr669/src/`
- PDF text: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/pr669/package.txt`
- My scripts:
  - `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/scripts/legload.py`
  - `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/scripts/swing.py` (this copy is now edited to the 0.40 m/s case)
  - `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/trike/scripts/massparse.py`

## Sources

- Web search, 18 Nov 2026, result summary only (page not opened, figure unverified): retailer listing for a 12 mm HTD 5M belt giving 285 N working tension — [desertcart HTD5M 12 mm](https://www.desertcart.com.eg/products/682679346-htd5m-timing-belt-pitch-5mm-width-12mm-open-ended-synchronous).
- All other figures come from the repo:
  - PR #669 files;
  - `hardware/trex_form_study/designlab/data/actuators.json` and `parts.json`;
  - `stage2/jetson75/LATENCY_AND_BUS.md` and `BUDGET_75.md`;
  - `DECISION_LOG.md` P47–P51;
  - `docs/hardware/RAPTOR_SCALING_AND_ALTERNATIVES.md` on main.