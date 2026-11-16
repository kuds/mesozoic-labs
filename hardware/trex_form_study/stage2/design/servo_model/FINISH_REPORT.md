<!-- Working paper, 16 Nov 2026: the report of the agent that applied the second check and the lead's first list of rulings to the design note. Nothing here is a rule. -->

## Result

The note is finished: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_fix/SERVO_MODEL_DESIGN.md`. It is the only note changed; `git status` in the main checkout is clean (head `cd68f2c`).

- No episode, fit or training was run and no web page was opened.
- Run through `tools/lowprio.sh` with the venv python, one at a time: my arithmetic (`calc_final.py`), the second checker's `probe_ref_speed.py` (stored references only), and three text-only scripts (splice, row move, table check).
- My scripts and a copy of the note before this revision (sha256 `e84f9389…`) are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_final/`.

**One instruction could not be met in full.** `LEAD_RULINGS.md` names 18 points. Section 8 has 22 older choices plus 5 from the second check.
- Ruled, in whole or part: 1, 3 (the X6's low end only), 10, 11, 15, 16, 18 (the route only), 19, 20, 21, 22, and the new 23 to 27.
- The owner's: 9, and the sending in 1.
- Named by no ruling: 2, 4, 5, 6, 8, 12, 13, 14, 17. They read "Not ruled" with the old recommendation; I did not decide them.
- Choice 7 (kv under a drawn inertia) reads "covered by the ruling on the tail's kv", from its words "one rule for every drive". The lead may say otherwise.

## Findings of the second check

11 findings: 11 applied, none not taken in whole. Finding 11's nine points are all in. Three details were not taken as worded:

- **Finding 3, "a sentence the owner approved".** `BUY_GATE_DRAFT.md:6-10` marks that sentence neither approved nor proposed. The note says "a sentence of the gate" that stands in the draft the owner answered on and in P40's row.
- **Finding 6, option 2 at 640 runs.** The lead ruled that the stricter option keeps the drawn panel, so it is 720 runs. Also "20 runs with randomly drawn motors" became "a panel of runs with 20 randomly drawn sets of motor values", because the panel is 80 of the 560 runs.
- **Finding 5, "a bench could find four of them later".** As the table stood this was not true: N6 sent the leg drives' encoder side to the maker's document only. I added one clause to N6 (N1's load reversal, read on both encoders) and then used the check's sentence.

## Where my number differs from the second check's

- **Tail kv under the new inertia:** 3.59 N·m·s/rad with the table's nominal 0.028 kg·m². The check's 3.60 uses the fit's unrounded 0.0284. The note gives both. The ratio 0.62 / kv is 0.17 either way.
- **Option 2:** 720 runs and 49 % at a 99.9 % rate. The check's replacement text had 640; its own last line gave 720 and 49 %.
- **Table count:** 39 = 1 + 4 + 6 + 28. The check's 38 = 1 + 4 + 6 + 27 was right for the first revision; I added the row `ts_zero` for the lead's ruling on the X6's low end.
- **Not a check number, but changed:** the tail's inertia against today's is 126 times (0.0284 against 0.000226), where the note had 120.

Everything else I recomputed agrees with the check: 0.351 and 0.243 m/s; 57 %, 53 %, 92 %; 0.861; 87, 57, 98, 94 %; median 0.0024; 7.61 against 7.24 rad/s; 214 W.

## Left for the lead (not decided by me)

1. **What a failed second-block record of the drawn panel does to the verdict.** The ruling says only that it is run.
   - (a) The bar's own form (H4): printed beside the verdict as "not replicated", no verdict of its own.
   - (b) Judged: 640 runs must then stay up (53 %), and eight records must each have 17 of 20 within every limit.
   - I recommend (a); the note and the owner's text are worded as (a) and say it is open.
2. **A rule that keeps the X6's drawn corner below its drawn zero-torque speed.** Without one the slope has no bound. I recommend: draw the corner first, then the zero between the larger of 8.01 and corner + 1.20 rad/s and 9.215.
3. **The names of the served rows** (choice 18): the ruling settles the route only.

Also added while carrying rulings through: work package S8 (the nominal-model estimator for policy rows) and section 9's text for the draft's lines 34-36 and 109, to be written only if the owner approves.

## Section "For the owner, in plain words" (verbatim)

## 11. For the owner, in plain words

Simulation and design only. Nothing here was measured on a motor of this build.

**What today's simulation leaves out.** Its motors are ideal. They push exactly as told, with one small friction value that nobody sourced, no looseness in the gears, full strength up to their rated speed, a battery that never dips, and joint sensors with no zero error.

**What the proposed model adds, one part per line.** In training, each run draws these at random within a table of ranges, so the walking controller cannot rely on one lucky value.
- Friction: each motor type gets its own, and it grows when the joint carries load.
- Gear looseness: a leg motor does nothing until the slack is taken up, and its sensor misreads by the slack. Leg motors only; the neck and tail wait until we know where their sensor sits.
- Strength at speed: a motor pushes less the faster it turns, and less again on a low battery.
- Battery dip: the voltage drops while the legs draw current.
- Response: the push builds up over a millisecond or so instead of at once, and the neck and tail get new targets 100 times a second, not 500.
- Sensors: each joint sensor can be off by up to about a degree, and its reading can be a few milliseconds old.
- Rotor weight: the spinning part inside each motor resists speeding up; on the main leg motor it may be up to 33 times heavier than today's guess.

**Where the numbers come from.** The table has 39 lines. One is a maker's own figure (the voltage printed on the battery). Four start from a maker's figure for something related (how hard a leg motor is to turn by hand; the upper limit on its gear looseness). Six are borrowed from other people's tests of similar servos, not these. The other 28 are the study's estimates or guesses, and the table says so line by line. The real values would come from a bench, and six measurements it would need are not on the bench sheet yet (gear looseness, strength at speed, battery dip, how fast the push builds, rotor weight, sensor zero); this note proposes adding them. The maker can give five facts for free, before any part is bought; a bench could find four of them later. The lead has ruled that the maker is asked before gate 1 is judged. Sending the questions is yours: the five are in appendix A, ready to send.

**What a pass at gate 1 would tell you, and what it would not.** It would tell you that two separately trained walking controllers keep the simulated robot walking at 0.30 m/s in two kinds of test. First, with every motor value at the table's expected value, on both hard floors, at each delay from 0 to 10 ms in 2 ms steps (condition 1, which you approved). Second, on the grippy floor at 4 and 10 ms (condition 2, proposed here): with 20 sets of motor values drawn at random from the table, and with friction, gear looseness and trunk weight each put at its extreme in turn. At an extreme the robot only has to stay up; the other limits are recorded there, not judged. The 20 random sets keep the rotor weight and the wiring resistance at their expected values. A pass does not show that the controller stays up everywhere between these points, or with two extremes at once.

Three more things about these tests.
- Under the model each delay carries a small extra lag in the motors' response, about 0.5 to 1.2 ms on the leg motors. The tests are therefore a little harsher than the plain delay numbers you approved. The lead has ruled that the lag stays in and that you are told.
- A random-panel result that passes is run once more with 20 other random sets, and you are shown whether it repeats.
- This note does not cover gate 1's pushes, turn and stop (condition 3). The tail servo's load (condition 4) would be read with this model; it is still a proposal and is not asked here.

A pass would not tell you that the real motors are inside the table's ranges. It says nothing about heat, flexing parts, worn gears or a motor switching off on low voltage. The test is strict in one way and weak in another: condition 2 alone needs 560 runs that all stay up, so a pair of controllers that truly falls once in 1,000 runs would still fail it about four times in ten; yet 20 clean runs of one controller at one setting and one delay show, with 95 % confidence, only that it stays up more than 86 % of the time there.

**The question for you (gate 1, condition 2): which motor settings may stop the purchase of the bench parts?** Each count of runs below is the number of settings × 2 delays × 2 controllers × 20 runs.
1. *Recommended.* Six extremes and the random panel can stop it: friction low and high, gear looseness none and 1°, trunk weight 10 % down and up, and a panel of runs with 20 randomly drawn sets of motor values. 560 runs must all stay up. At these extremes the robot must stay up; the other limits are recorded there, not judged. The high-friction extreme is above the study's own estimate, and 1° is four times the maker's limit. Two harsher settings are run and shown to you but cannot stop a purchase yet: the heavy rotor, which hangs on a catalogue number the maker must explain, and the weak battery, which uses a wiring resistance nobody has measured.
2. The same, plus the heavy rotor and the weak battery: 720 runs must all stay up. Stricter; the purchase could then stop on a misread catalogue number or on a wiring resistance nobody has measured, and the weak-battery setting as defined falls below the motors' own minimum voltage at the walk's peak draw.
3. Only the expected values and the random panel can stop it: 80 runs in this condition must all stay up. Lightest: a controller that falls at an extreme would still pass. Every extreme is still run and shown to you.

Under every option:
- **At an extreme the robot must only stay up. The other limits, such as how hard and how long each motor works, are recorded there and not judged.** On the random panel the bar's whole rule applies: all 20 runs stay up and at least 17 meet every limit. This is an exception to the gate's sentence that its conditions use the bar's own pass rule. Options 1 and 2 judge extremes, so choosing one of them approves the exception; the buy gate's text is changed only if you approve.
- For scale, by the study's arithmetic with the runs taken as independent: a pair of controllers that truly stays up in 999 runs of 1,000 would pass option 1 about 57 % of the time, option 2 about 49 % and option 3 about 92 %.
- Each random-panel record that passes is run once more on 20 other seeds: 80 more runs if all four pass. The result is shown beside the verdict. Whether a failed repeat may also stop the purchase is not settled; it is open for the lead (section 8, choice 27).
- In this condition the slippery floor, the motors' gain limits, today's strength-at-speed line, the other hip motor that may ship and two combined extremes are run and shown without a verdict.

## Appendix "Five questions for the makers" (verbatim)

## Appendix A. Five questions for the makers

Ruled by the lead on 16 Nov 2026: the five questions are asked before gate 1 is judged, and sending them is the owner's. Each text below stands alone and can be sent to a maker's support address as it is. None names a project, a person or an organisation. All five concern the two leg drives, which have one maker. Nothing is asked of the makers of the neck and tail servos: their open point, the side of the position sensor, is the bench's (N6).

The figures quoted in the questions are the maker's catalogue values as the study's research record transcribes them [M]. I did not re-open the maker's pages; the record read the store listing of question 4 on 7 Nov 2026 (RES:146). The line under each text says where its figures stand and what the answer changes in this note. Those lines are not part of the message.

**Question 1. The rotor inertia: before or after the gearbox.**

> Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: is the listed inertia before or after the gearbox?
>
> Hello. Your data for the RMD-X4-P36-36-E-L (X4-36-L, V4.1) list an inertia of 0.000082 kg·m². Your data for the RMD-X6-P20-60-E (X6-60, V4) list 0.66 kg·cm². For each part: is this the inertia of the rotor alone, before the gearbox, or the inertia seen at the output shaft, after the gearbox (ratio 36 and 19.612)? Please also give the inertia seen at the output shaft of each part, in kg·m². Thank you.

Figures: RES:85 and RES:70 (X4); RES:154 and RES:138 (X6; the record gives "0.66 kg·cm²" without the word inertia). The answer sets the `armature` rows of the two leg drives and decides whether `J+` stays a setting (section 3; section 8, choice 1).

**Question 2. The gain ranges of the motion mode, and their units.**

> Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: range and units of KP and KD in the Motion Mode Control Command
>
> Hello. I plan to drive the RMD-X4-P36-36-E-L and the RMD-X6-P20-60-E with the Motion Mode Control Command over CAN (0x400 + ID). For each part: (1) Does it accept this command? (2) What are the smallest and the largest KP and KD it accepts? (3) In which units do KP and KD act? I need them as N·m/rad and N·m·s/rad at the output shaft. If they are defined on the motor side or as a current, please give the conversion. The manual of another model of the series, the RMD-X4-P36-24, gives KP 0 to 500 and KD 0 to 5 with no units: do those ranges apply to these two parts? Thank you.

Figures: LAT:441-442 [M via search extract]. The answer sets `gain_cap` and decides whether `G5` becomes the nominal (section 3; section 8, choice 2).

**Question 3. The RMD-X6-P20-60-E at 24 V.**

> Subject: RMD-X6-P20-60-E: torque against speed at 24 V, and the back-EMF constant
>
> Hello. The RMD-X6-P20-60-E (X6-60, V4) is listed with a rated torque of 20 N·m at 153 rpm and a no-load speed of 176 rpm, both at 48 V, and with a supply range of 20 to 55 V. I plan to run it from a 24 V supply. (1) At 24 V, what is the no-load speed in rpm, and up to which speed in rpm can it give its peak torque of 60 N·m? A torque-speed curve at 24 V, or a few pairs of N·m and rpm, would answer this. (2) The listed back-EMF constant is 16 Vdc/krpm. At 176 rpm at the output and a gear ratio of 19.612 that gives about 55 V, which is more than the 48 V supply. Is the constant per 1,000 rpm of the motor or of the output shaft, and how is its voltage defined? Please give it in V per 1,000 rpm of the motor, with the definition. Thank you.

Figures: RES:131-138, RES:154; 55 V is my arithmetic [E]. The answer sets the X6's `ts_corner` and `ts_zero` and ends the draw of the zero-torque speed (section 3; section 8, choice 3).

**Question 4. Which hip-pitch part ships.**

> Subject: Store listing "RMD-X6-P20-60": is it the X6-60 (V4) or the X6-50-L (V4.1)?
>
> Hello. The store listing "RMD-X6-P20-60" gives a mass of 0.78 kg, a peak torque of 50 N·m and a backlash of 10 arcmin. The manual and the data sheet of the RMD-X6-P20-60-E (X6-60, V4) give 0.82 kg, 60 N·m and at most 15 arcmin. The listing's figures look like those of the RMD-X6-P20-50-E-L (X6-50-L, V4.1). Which part is shipped today when this listing is ordered? Please answer with the full model name and version, and with its mass in kg, its peak torque in N·m and its backlash in arcmin. Thank you.

Figures: RES:146, RES:154. The answer decides whether `X6L` becomes the model of the hip pitch: a planning peak of 31.5 N·m in place of 42 N·m, and another friction (sections 1.2, 4 and 7).

**Question 5. Which encoder closes the position loop.**

> Subject: RMD-X4-P36-36-E-L and RMD-X6-P20-60-E: which encoder closes the position loop, and which one is reported?
>
> Hello. Both parts are listed with two encoders (17 and 18 bit on the RMD-X4-P36-36-E-L, 17 and 17 bit on the RMD-X6-P20-60-E) and with a backlash of up to 15 arcmin. (1) In position control and in the Motion Mode Control Command, does the position loop close on the encoder on the motor side of the gearbox, or on the one on the output side? (2) Which encoder's angle does the drive report in its replies: motor side or output side? In which unit, and with how many counts per turn of the output shaft? (3) Can the user choose between the two? Thank you.

Figures: RES:75-76, RES:143-144. The answer decides whether the dead zone of 2b is the right model of the leg drives' gear play, and whether the reported angle carries the play (2b, 2e; N6).

## The file

- Path: `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/servo_fix/SERVO_MODEL_DESIGN.md`
- Lines: 834
- sha256: `d1952204a8a2f2640c45084dc5bedfed52251f65e019d89db4e4cddc0e05a471`
