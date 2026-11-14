# Lessons learned, 7–14 Nov 2026

Written on 14 Nov 2026, at the end of the study's first long working session (rounds one and two, Stage 1, Stage 1b
and Stage 2's first phase). It is for whoever continues the work, the next session first, and for the owner. It covers
what the work found about the robot, how the work was done, what went wrong and how it was caught, and how to work in
the cloud container. It summarises and points to sources. The records and reports it cites are the evidence, and where
they disagree with this file, they win.

Paths are relative to `hardware/trex_form_study/`. "a / b" figures are 60 cm / 75 cm unless stated. S1 is the Stage 1
report (`stage1/runs/2026-11-08/STAGE1_PROBE.md`) and S1b the Stage 1b report
(`stage1/runs/2026-11-10-speed/STAGE1B_SPEED.md`). P, F, E and A rows are in `DECISION_LOG.md`. The next session's
starting prompt is `NEXT_SESSION_PROMPT.md`.

## 1. Where things stand

| Step | What it did | Main document | State on 14 Nov |
|---|---|---|---|
| Round one (7 Nov) | 4 sizes, 3 widths, 3 body families, 3 postures, sized from sourced parts and screened | `ROUND1_REVIEW.md` | done; the owner answered P10–P14 |
| Round two (7–8 Nov) | 20 candidates at 60 and 75 cm, an overnight refinement, three picks | `ROUND2_REVIEW.md`, `round2/SHORTLIST.md` | done; the picks were adopted (P16–P23) |
| Stage 1 (8–10 Nov) | closed-loop probe of the picks, controllers R0–R4 (§2.1) | `STAGE1_PLAN.md`, S1 | done; S1 corrected in place on 10 Nov |
| Stage 1b (10 Nov) | the tracker refitted for speed (R2s) on the 60 cm picks and on a faster-knee variant; servo delay | `STAGE1B_PLAN.md`, S1b | done |
| Stage 2 plan (10–11 Nov) | character plus "somewhat agile": the agility bar, R5, Colab training | `STAGE2_PLAN.md` | owner decisions 3–6 taken (P29) |
| Stage 2, Track A, phase 1 (11 Nov) | look features in the gait generator, the V0–V6 ladder, its Stage 0 screen, the tail-phase check, clips, the owner's look review page | `stage2/look/SCREEN.md`, `stage2/look/RENDERS.md` | awaiting the owner's L1 picks |
| Stage 2, S2-2 (11 Nov) | the expressive head (`-X`: neck yaw and jaw) built and costed | `stage2/x/HEAD_X.md`, `stage2/x/X_COSTS.md` | done; its G1 is provisional (§5) |
| PR #668 review (14 Nov) | the 75 cm build with a Jetson Orin Nano Super (PR kuds/mesozoic-labs#668, unmerged), reviewed, not re-run | `JETSON_PLEM_REVIEW.md` | its track comes first in the next session (owner's request) |

The bodies are `60-N-T-10` (the 60 cm Tapered), `60-N-H-10` (the 60 cm Hybrid, dropped for Stage 2 by P29) and
`75-M-H-10` (the 75 cm Hybrid with the targeted drives and bigger feet), plus the variants `-K` (a faster knee, Stage
1b) and `-X` (the expressive head). Nothing from Stage 1 onward has been adopted (P24). The draft decision-log rows
wait for the owner (§7). All of this is on the branch `claude/mesozoic-trex-design-iteration-azm2fs`, whose pull request
is kuds/mesozoic-labs#667 (https://github.com/kuds/mesozoic-labs/pull/667).

## 2. What the robot can and cannot do (in simulation)

### 2.1 The controllers

R0 is the reference walk's feedforward with PD servos. R1 adds a balance tracker on the true centre of mass, and R2 the
same tracker on the estimated one, with 13 gains fitted by CMA-ES; each body has two fits. R3 adds a PPO residual on R2,
and R4 is a policy alone. R2s is Stage 1b's speed refit of R2. Stage 2 plans R5, an imitation policy under speed and
turn commands whose zero action is R0L, and R6, a policy alone. "Shown" means that all 20 episodes stay up and at least
17 meet every limit, under both fits or seeds; with one, the verdict is provisional.

### 2.2 What is shown

- **At no servo delay (hard floor at μ 0.9 unless stated), R2 shows all three picks on the same 14 of 20 gated
  conditions.** These are standing on hard floor, robot_floor_carpet and low pile; the nominal walk (0.217–0.218 /
  0.243 m/s) at μ 0.9 and 0.54; the 0.35 rad/s walking turn and the 0.5 rad/s turn in place; starting and stopping; and standing shoves at
  half the Stage 0 bound (S1 §2; draft F26).
- **The 75 cm body walks and turns even open loop** (R0) and is the most tolerant body on most margins (S1 Answers).
- **The fastest walk shown is the nominal one at both sizes.** It is the walk the owner found too slow (S1b Answers).

### 2.3 Where it stops

- **Speed.**
  - At 60 cm the brisk walk (0.315 m/s) is not shown. The Tapered speed refit came closest: 19 of 20 up and 17
    meeting every limit. That is one fall short of passing on its single fit, and a pass there would still be
    provisional.
  - The Tapered's animal-like walk (Froude 0.05, 0.446 m/s) stays up in 16 of 20 after the refit (1 before), about 3 %
    short of its target speed. The Hybrid's refit kept 1 of 20 up.
  - At 75 cm the brisk walk (0.35 m/s) stays up in every Stage 1 episode. Under R2, with both fits, it misses only the
    tail servo's planning limit, 1.04–1.07 × A1. The judged R3's run 43 also misses speed and duty (S1, S1b).
  - A faster knee did not change the 60 cm answer: the refit, not the knee, raised survival (S1b Q2).
- **Servo delay is the largest unknown.**
  - At 10 ms of command delay every 60 cm walk and walking turn falls in all 20 episodes, refitted or not, on every
    body, with one exception, a Hybrid nominal walk (S1b Answers, §4).
  - The 75 cm nominal walk stays up to 8 ms and falls at 10 ms (S1's correction).
  - At 60 cm, faster walks tolerate less: about 4 ms for brisk and about 2 ms for Froude 0.05. These are hints from
    fit and selection episodes (S1b §4). No 75 cm speed walk has been run with delay.
  - The 75 cm walk falls at 10 ms yet stays up at 20 ms, and nobody knows why. 10 ms is also one 100 Hz feedback period
    and the bar's judged delay, so read any 10 ms verdict beside its neighbouring delays.
  - The model's delay comes on top of a loop that already lags (S1b N8):
    - the whole target, feedforward included, arrives k physics steps of 2 ms late;
    - the sensors are one step old;
    - the tracker's feedback is computed every 10 ms and ramped in over 10 ms.
    A bench figure is an end-to-end delay to set against that.
  - Nobody has measured the real servos' delay (draft A17).
- **Falls concentrate in transitions.**
  - At 10 ms, 116 of 120 speed walks fell before the brisk gait began, and 169 of 240 walking-turn falls came in the
    ramp into the turn.
  - At 0 ms, 30 of 79 60 cm animal-pace falls came before the Froude reference started (S1b §4, N5; S1 N5).
  - A rung that starts from standing is decided in its start and change. Read every rung's falls by segment before its
    verdict, and weight R5's starts, changes and stops to match.
- **Walking shoves.** At half the Stage 0 bound, 0–8 of 20 walks stay up per direction. Walking shoves are recovered
  only up to 0.07–0.15 / 0.10–0.17 m/s. The tracker's fit never saw a shove (S1 Answers).
- **Thin carpet.** Every walking start falls, on every body, in the unmeasured carpet model (S1).
- **Learning so far.**
  - R3 added no new pass and used about 1 % of its authority at σ 0.02 (S1 N3). R4 learned only to stand still.
  - Budgets were small: R3 trained to at most 2.5M steps.
  - Character policies of the kind Disney publishes train for about 22 GPU-days each (`stage2/research/research_look.md`
    §2), so "not learned" at this scale says nothing about the body.
- **Fit-to-fit spread.** Two fits of one body differed by 15–17 of 20 at the brisk walk (S1b Answers). One fit is a
  narrow test, which is why Stage 2 requires two seeds.

### 2.4 What limits it

- **Not the leg drives, as far as the model shows.**
  - Removing their torque limits never helped significantly where enough walks fell to test it, and only falls and
    single stumbles pass a drive's rated speed (S1b Answers; N4 for the lifts, §5 for rated speed).
  - That headroom is read at a full battery and under an ideal, unmeasured envelope.
    - The 60 cm plant runs at 25.2 V. At the pack's nominal 22.2 V the knee's peak at the animal pace rises from about
      0.87 to 0.94 of rated speed, and it reaches 1.00 at 21 V (computed, S1b §5).
    - The envelope gives full planning peak torque up to rated speed; A2's dyno curves would show whether it is so.
    - The K knee's rating is 2.9 times RobStride's own 100 rpm figure (S1b N6).
  - In the reference walks the binding limit is the knee's speed: 0.93–0.95 of it at Froude 0.05.
  - Lighter materials buy torque headroom and runtime, not speed headroom (F22).
- **The tail servo (STS3215) is near its sustained planning limit.**
  - Closed loop, it sits at 0.97–1.03 of A1 at the 60 cm Froude 0.05 walk and 1.04–1.07 at the 75 cm brisk walk.
  - Stage 1's closed loop, where the plant carries the joint damping and friction, ran the 75 cm animal-pace tail at
    1.17–1.31 of A1 (S1 §2).
  - The Stage 0 screen of the look walks with today's tail phase (V0–V4) adds those terms and reads up to 1.071 of A1
    and 1.178 of the planning peak. With look feature F9's phase it reads 0.39–0.43 of A1 (`stage2/look/SCREEN.md`).
- **Today's tail swings in the wrong phase.**
  - It adds 8–12 % to the stance foot's peak yaw moment.
  - Look feature F9's phase cuts the friction need by 14–18 %.
  - Counting the plant's default damping and friction, it cuts the tail servo's sustained load by 40–61 %, roughly
    half; without them the cut is 11–40 %.
  - It works because the body's sway then drives the tail and the servo brakes it (SCREEN.md, "In brief").
- **The 60 cm neck is close to its heat budget.**
  - The STS3250 neck runs at 0.83–0.84 of A1 in the closed-loop nominal walk (S1 §3).
  - With the look's hip bob (look features F1/F2) carried through today's head nod, and the plant's damping, its peak
    torque reaches up to 1.17 of the planning peak at the animal pace (V1B100). It is 0.99 with F10's stabilised head,
    and the sustained ratio stays at or below 0.89 (SCREEN.md).
  - The `-X` jaw servo in the head would take the closed-loop nominal-walk load to about 0.93 of A1. That is an
    estimate scaled from Stage 0; the static hold goes from 0.75 to 0.84 (X_COSTS.md).
- **Joint damping and friction dominate the tail's load**, and on `-X` the neck yaw's and the jaw's.
  - On the 60 cm neck pitch they add about 7–12 % to what is mostly a gravity load.
  - The values (damping 0.2 N·m·s/rad, friction loss 0.15 N·m) are `designlab/mjcf.py`'s defaults on every joint, legs
    included. They are unsourced, and no decision-log row records them: A16 is the carpet screen, and a draft row is
    proposed in §7.
  - A bench measurement could move these loads more than any part change.
- **Grip.**
  - Stance-foot spin, the pad's yaw moment, is the friction limit that binds (F21).
  - At 60 cm the straight walk at 0.40 m/s or faster needs more than μ 0.54 on every look rung, and more than 0.9 on
    several (SCREEN.md).
  - The treads' estimated grip is 0.42–0.8 on vinyl and laminate (F22).
- **Size.**
  - At the same gait, speed grows with the square root of size: v = √(Fr·g·H).
  - The 75 cm body is 11–12 % faster than the 60 cm one, and 120 cm would be about 40 % faster: bigger buys speed
    slowly (S1b Answers).
  - Froude speeds at 60 cm (H 0.405 m): nominal 0.218, brisk 0.315 and Fr 0.05 0.446 m/s. At 75 cm (H 0.502 m):
    0.243, 0.351 and 0.496 m/s.

### 2.5 The expressive head (`-X`)

- It adds 0.150 kg and moves the centre of mass 5.2 / 4.4 mm forward.
- The one-foot hip-pitch hold rises from 0.55 to 0.65 / 0.76 to 0.83 of its limit.
- With the new joints held still, the nominal and Froude 0.05 walks stay within every drive limit.
- At the animal pace the knee's speed, the tightest limit, goes from 0.93 to 0.97 / 0.93 to 0.96, with the 15 % margin.
  At the nominal walk it goes from 0.56 to 0.58 / 0.56 to 0.57.
- A 0.6 rad gaze shift fits in 0.5 s at 60 cm and in 0.6 s at 75 cm.
- A 0.6 rad jaw opening in 0.4 s runs the servo at its rated speed; in 0.5 s it keeps both margins. A quick snap is out
  of reach.
- The cost to watch is the 60 cm neck's heat budget (§2.4; `stage2/x/X_COSTS.md`).

### 2.6 The look

- **The Stage 1 reference walk looks like a ZMP humanoid's, because it was built to size drives.**
  - The hips stay at a constant height and the trunk stays level and rigid while it slides sideways.
  - The feet lift and land flat.
  - Slow walks stretch the swing into slow motion.
  - The head nod and the tail swing run on the stride clock (`stage2/research/research_look.md`, Bottom line 1).
  - This is the look the owner asked to avoid. On 10 Nov, in conversation: "I don't want some sort of like Asimo
    robot where it takes one foot at a time and it's very mechanical looking". `STAGE2_PLAN.md` §1 records it as
    "lifelike, not mechanical".
- **Stage 2 adds the look as options** in `designlab/gait.py`, with today's walk unchanged by default.
  - They are the look features F1–F14 of `research_look.md` §3.1.
  - These F numbers are not the decision log's F rows; this file says "look feature F3" where it could be confused.
  - F8 imposes no trunk pitch.
  - F11 and F12, for turns, starts and stops, are deferred (`look.DEFERRED`).
  - The variant ladder V0–V6 (with extras) runs at three speeds per body.
- **The renders are kinematic.** Whether a feature reads as alive is the owner's call at L1, with viewers' ratings.
  What each feature costs is in the Stage 0 screen (`stage2/look/SCREEN.md`).

### 2.7 The owner's questions of 10 Nov and where their answers stand

These were answered in conversation after the Stage 1b report; P28 and `STAGE2_PLAN.md` record what followed.

- **Why does the 75 cm body do slightly better?**
  - At the same speed its legs move more slowly for their size: a lower Froude number.
  - It also carries the targeted drives and bigger feet (F19, F23, F24).
- **What can be shown robustly today?** The nominal walk, the target turns, starting and stopping, and standing shoves,
  all at no servo delay (§2.2). Delay may undo them (§2.3).
- **Would a plain torso or a 100 cm body help?**
  - No plain-torso body has been simulated, so the model cannot answer this directly. P6 rejected a rectangular
    chassis.
  - As far as the model shows, the leg drives do not limit speed (S1b Answers).
  - The 60 cm failures read controller-limited (S1 §2), 10 ms of delay fells every 60 cm walk (§2.3), and grip binds
    (§2.4).
  - Parts of the T. rex layout, the head and tail servos, run near or over their planning limits (§2.4).
  - Size buys speed only as √H.
  - Larger sizes wait until much later (P28).
- **How fast could a robot this size walk?**
  - The Stage 0 screen prices 0.40 m/s and the Froude 0.05 walks (0.446 / 0.496 m/s) within every leg drive limit,
    except the knee speed of the 75 cm V6 and V6S at the animal pace (1.012, 1.009; `stage2/look/SCREEN.md`). The same
    screen puts the 60 cm friction need above μ 0.9 on several rungs at those speeds (§2.4).
  - Whether a controller can walk them is Stage 2's question, and so far it is unanswered.
- **"Controller" means the software** (the fitted tracker or a learned policy running on the robot's computer), not
  the motor drivers.
- **The decision.** Character (the powered head and tail) with some agility, not speed (P28). The bar is a walk at 0.4
  m/s, a 90° turn in about 2 s, a nudge while walking and a stop within a step or two, judged at 10 ms of delay (P29).

### 2.8 What none of this shows

- **Simulation only.** The servos are ideal PD behind a torque-speed envelope and a delay. There is no backlash, no
  heat beyond A1 and no fall damage.
- **Unmeasured inputs.** Joint friction and damping, the carpet, grip and delay are unmeasured.
- **Untestable flag.** The tracker's pad-edge flag could not be tested (S1 N1).
- **No self-collision.**
  - Only the foot pads collide: every other geom has `contype` 0, so the legs can pass through each other and through
    the body.
  - Leg interference is a 50 Hz geometric check (`legs.LegGap`), judged on turns only, and the env's reward and
    termination never see it.
  - A learned policy could exploit this. Put it in R5's termination or reward, or judge it on every bar condition.
- **No hardware.** No hardware has been built or measured.
- **Reference walks are not controllers.** Kinematic clips and Stage 0 screens show prescribed motions with perfect
  tracking.

## 3. How the work was done: method lessons

1. **Pre-register the pass rules, stop rules, seeds and budgets before any run, and label any later change as such.**
   An example is decision 39: a tolerance set after seeing a 17.997 s against 18.000 s miss is marked in every place it
   is used. Plans carry numbered decisions (`STAGE1_PLAN.md` has 39) so that a report can cite the rule it applied.
2. **Run two fits or seeds.** One is a narrow test (§2.3), and the plan's verdicts are provisional with one.
3. **Keep prospective design, simulation evidence and measured results apart, in words as well as files.** No result
   so far is measured. Kinematic clips carry "kinematic reference, not a simulation" on every frame.
4. **Have every build, report and owner page reviewed by an independent agent.** Give the reviewer the code, the records
   and the rule, never the lead's conclusion, then fix and re-check. The reviews caught real errors:
   - S1 said the 75 cm walk survives at least 20 ms of delay. Its own records showed falls at 10 ms (corrected 10 Nov;
     lesson 5).
   - The `-X` gates were first recorded "all pass" while G1's step rate had been timed on 2 of 4 cores. G1 is now
     provisional, and the cores are recorded (`stage2/x/HEAD_X.md` §3).
   - The look review page's blind clips leaked their labels. A separate viewers' page was made, and the mechanical
     "tells" list was corrected.
   - The Stage 1b report mixed measured 75 cm speeds with targets. It now gives targets.
5. **Do not trust a search's reading; check every point it evaluated.**
   - The stay-up search evaluated 0 and 20 ms only. It assumed that a walk falling at some delay falls at every
     larger one, and so read "works across the interval".
   - The falls at 10 ms were in the same records, from another search.
   - No delay search ran 2 ms or 12–18 ms, and the grip searches skipped μ 0.10–0.20 (S1, correction note).
6. **Change the generator only behind options that default to today's behaviour, and test that bit for bit.**
   - G2: with default GaitSpec fields, Stage 1's references regenerate byte for byte (`tests/test_stage2_look.py`).
   - Every existing model keeps its recorded `xml_sha256` (`tests/test_stage2_x.py`).
7. **Hash the inputs into the records, and make reports from the records.**
   - Every Stage 1 record key holds the sha256 of the model files the plant reads (`panels.key_context`).
   - The tables are machine-made (`python -m designlab.stage1 report`, `python -m designlab.stage2.screen md`,
     `python -m designlab.stage2.motion md`), and reports cite record paths.
   - The cost: the git-ignored inputs must be restored exactly (§4.1).
   - Records hash their code too. The panels, tuner and speed fit hash their modules, and `train.py` hashes every
     `designlab/stage1/*.py`. Any edit there, such as registering a body, generalising `env.py` or adding R5, makes
     earlier records read as made by other code.
   - So Stage 1's and 1b's records are cited, never rerun or extended with newer code (`STAGE1B_PLAN.md` decisions 3,
     5 and 9). The code stays frozen from a merge until the fits and runs built on it have finished.
8. **Read a metric's definition before reading its verdict.**
   - The speed term passes only when the judged 10 s mean reaches the target; the command is 5 % above it.
   - Toe clearance read on the reference's clock misreads a foot that lifts 10 ms late (S1b N5).
   - Sideways drift went unjudged: a median 0.62 m in 10 s with the heading gain near zero (S1b N3).
   - Stage 2's bar fixes both of the last two (`STAGE2_PLAN.md` §5).
9. **Fall attribution is a hint (decision 20).**
   - It needs at least 5 falls.
   - It lifts relaxable limits and applies a sign test.
   - Its labels name a likely cause, not a proof. The pad-edge candidate cannot be tested.
10. **Timing gates need an idle machine with one process per core.** G1's step rate failed on arm K's bodies and on the
    pick alike because the machine was slower that day (`STAGE1B_PLAN.md` decision 11). On a shared machine a timing is
    provisional, never a pass.
11. **Price before rendering, and render only what the owner must judge.** The Stage 0 screen (drives, centre of
    pressure, sway, accelerations, clearance, passive neck and tail loads, grip) ran on every rung before any clip was
    made (`stage2/look/SCREEN.md`).
12. **Measure before budgeting.**
    - The Python tracker ran about 200 env steps/s against the 750 planned, and the fit took 6.5 h instead of 2–3. That
      forced decision 21's 24 h cap.
    - Do not oversubscribe the machine. On 9 Nov, 9 PPO jobs and 6 panel and ledger processes ran on 4 cores at load
      15. Memory fell by about 70 MB a minute, because the panels cache every model they build for the life of the
      process.
13. **Build agent briefs from the data files, and report counts only once every check has returned.**
    - A compute brief carried a stale Jetson price ($249 against the catalog's $399).
    - An audit's error count was reported as 56, then corrected to 62.
14. **Small traps worth remembering.**
    - Record keys formatted with `.2f` turned 0.175 into "0.17"; use `%g`.
    - Inverse dynamics with the prescribed centre of pressure off the pad gives artifact loads; such cases are marked
      inconclusive (one "1.19" reading was one).
    - Use one OSMesa renderer per process: several in one process corrupted renders.
15. **A 3-episode monitor gain is not a verdict.**
    - Both 75 cm R3 runs passed G6c on reward per step, one of them while falling in all 3 low-pile episodes.
    - Neither gain became a limit met over the 20-episode panels (S1 N3, §4).
    - Stage 2's monitor decides only the stop rules; judge on the panels.
16. **PPO's KL early stop can silently shrink updates.**
    - PPO's KL per update grows as 1/σ², and `target_kl` stops an update early without a message. At σ 0.02 and a
      1e-4 learning rate, each update ran only 1–9 of its 160 minibatch steps (decision 24).
    - Record the minibatch steps run per update in every long run.
    - Repeat decision 24's probe whenever σ or the learning rate changes.
17. **Fix the delay draws in a fit.**
    - In Stage 1b's speed fit, which R2L repeats, every member of a generation ran the same single episode per kind.
      The delays were drawn from the seed alone, so every body drew the same ones.
    - Generation scores mostly tracked those draws, and selecting over pooled delays hid the Hybrid's 0 ms brisk loss
      (S1b N1, N2).
    - Give R2L's fit and selection fixed or repeated delay draws per kind, and read its verdicts per delay.
18. **Communicate with the owner plainly.**
    - Answer the question first, in plain words, then give the evidence.
    - Never mix measured values with targets.
    - Correct a report in place, with a dated note at the top that lists every changed place.
    - Ask decisions as questions with a recommended option.
    - Propose decision-log rows as drafts; the owner adds them (P24).
    - Publish owner pages as private claude.ai artifacts, checked at desktop and phone width.
    - For blind ratings, make sure no file name, caption, alt text or order reveals a variant.
    - Lead with plain numbers and units ("0.22 m/s (0.5 mph)") rather than a pass/fail summary. The owner asked for the
      walking answer to be rephrased once.
    - Ask through the question tool rather than a lettered list in text: at most 4 questions per call, labelled "Q1 of
      8", recommended option first. The owner asked three times to be prompted. Read a free-text answer before acting
      on it: one answer was a request for more time.
    - Send a push notification when long work finishes; the owner asked to be pinged four times.
    - "Help me think" means discussion, not new runs ("Don't do any big experiments", 10 Nov).

## 4. Working in the cloud container

### 4.1 A new session starts from a fresh clone

Nothing outside git reaches a new session. Each piece comes back in its own way:

- **The venv:** `tools/setup_venv.sh`.
- **The git-ignored model files and arrays:** `tools/restore_inputs.py`, which regenerates them from the committed
  records and checks each sha256.
- **Stage 1's PPO checkpoints and VecNormalize statistics:** no restore.
  - They are under `stage1/runs/2026-11-08/<body>/{ppo,policy_only}/*/checkpoints/`, about 150 MB.
  - Only retraining makes them again, and `run.json` records their steps, not their hashes.
  - Stage 2 does not need them.
- **The evaluation traces:** no restore. They are about 1.2 GB under `<body>/eval/traces/` and back the committed
  evaluation rows.
- **The clips' media:** they are in the published look pages and regenerate with `designlab.stage2.clips`.
- **The scratchpad's scripts:** the useful ones are now in `tools/`.
- **Files the reports cite that lived only in the scratchpad:** now committed at their cited paths:
  - `stage1/runs/2026-11-08/gap_options.json`;
  - the memory logs in `stage1/runs/2026-11-08/machine_logs/`;
  - `round2/refine/nominal_lift.py`;
  - `stage2/research/loads.py`;
  - the speed round's logs in `stage1/runs/2026-11-10-speed/logs/` (force-added: the root `.gitignore` ignores every
    `logs/`).
- **Two Python environments made the records.**
  - `tools/requirements-study.txt` covers the plant, fits, panels, ledger, look and `-X` records.
  - Stage 1's PPO training and its R3/R4 evaluation rows ran in a second venv: torch 2.9.0+cpu, stable-baselines3
    2.9.0 and gymnasium 1.3.0 (`tools/requirements-stage1-ppo.txt`; `run.json` `sessions[].versions`).

Without the model files the plant rebuilds models elsewhere, with another ledger. Every record key then differs, and
`eval` and `sweep` refuse the checkout (`STAGE1_PLAN.md` decision 33, "Where it runs"). Restore first, then run the
tests.

### 4.2 Restarts, keep-alive and resuming

- **The container restarts without warning, and often.**
  - It started at least nine times in seven days (UTC): 7 Nov about 22:25; 8 Nov about 14:35; 9 Nov 20:37 and about
    21:15; 10 Nov 01:29 and about 03:58; 11 Nov 02:23 and 06:38; 14 Nov about 15:40.
  - Six restarts followed a turn that ended with only detached (`setsid`/`nohup`) jobs alive, some within minutes.
  - Three came while tracked work was running (a workflow, a heartbeat).
  - The keep-alive below makes idle reclaim less likely; it does not prevent restarts.
  - No restart's cause is recorded. `STAGE1_PLAN.md` decision 36 guessed memory, but S1 N8's logs do not support it.
  - Each time, the filesystem (checkout and scratchpad) survived and every process died.
- **Every long job is an idempotent chain that resumes from its own checkpoints.**
  - A watchdog starts any chain that is neither done nor running; `tools/watchdog_example.sh` is the speed round's,
    generalised (its chains: `stage1/runs/2026-11-10-speed/run_speed_round.sh`).
  - `tools/keepalive.sh` runs it every 10 min.
  - After any pause, check `uptime` first.
- **Keep the container alive.**
  - Run `sh tools/keepalive.sh <log>` with the Bash tool's `run_in_background` and `timeout` 7200000, the maximum.
    The default timeout, 1800000, stops it after 30 min.
  - Put the log outside the checkout or under a `logs/` directory (ignored).
  - The session tracks the command and is re-invoked when it ends after about 115 min; re-arm it then.
  - As a backstop, hourly scheduled tasks wake the session after a restart (create_trigger, send_later). Delete them
    when the work is done; none were active on 14 Nov.
- **Workflows resume with `resumeFromRunId`.**
  - Completed agents return their cached results, and an agent that was running starts again from the beginning.
  - Edit its prompt to add a "RESUMPTION NOTE" saying what it left behind (as in the `-X` build, which a restart cut
    off).
  - A workflow runs at most min(16, CPUs − 2) agents at once: 2 on this 4-core machine.

### 4.3 Worktrees

- **Give each builder agent its own git worktree, and name it by absolute path in the prompt.**
- **A worktree has none of the git-ignored inputs.**
  - Link or copy them in and check their sha256 (`stage2/x/x_inputs.py copy-from`,
    `stage1/runs/2026-11-10-speed/arm_k_inputs.py`).
  - Exclude the link directories through `.git/info/exclude`; the round-two refine model links showed as untracked
    files.
  - After a merge, copy the generated ignored files back the same way.
- **Do not `cd` into a worktree in the lead's own shell.** It moved the primary working directory once. Use `git -C`
  or a subshell.
- **Remove merged worktrees.** Each holds 0.2–0.5 GB; thirteen held about 3.5 GB on 14 Nov.

### 4.4 CPU, memory and disk

- **The machine.** 4 cores and 15 GB.
- **Training speed.** One R3 training job (4 envs in one process, PPO) runs about 200 env steps/s. MuJoCo is only
  14–20 % of an env step; the Python tracker and estimator are most of the rest
  (`stage2/research/research_pipeline.md` §3).
- **Priority and threads.**
  - Run side jobs through `tools/lowprio.sh`; the kernel's autogroup makes `nice` alone ineffective across shells.
  - Use `tools/memguard.sh` when training and evaluation share the machine.
  - Set `OMP_NUM_THREADS=1` in worker processes.
- **Process counts.**
  - `nproc` reads `OMP_NUM_THREADS`: `run_x.sh g1`'s old default timed 1 process instead of 4.
  - `PYTHON_CPU_COUNT` does not change `os.sched_getaffinity`, which G1 now records.
- **Disk is a fixed allowance per session.**
  - On 14 Nov about 4 GB were free, beside a 6 GB venv (torch from PyPI brings its CUDA libraries) and 1.2 GB of
    traces.
  - `TORCH=cpu bash tools/setup_venv.sh` installs the CPU-only wheel of the same torch instead: 1.4 GB in all, built
    in 50 s, and it passed `tests/test_stage1_ppo.py` (36 of 36).
  - Deleting files frees space at once.

### 4.5 MuJoCo, OpenGL and torch

- **MuJoCo picks its GL backend when it is first imported.**
  - `designlab/__init__.py` sets `MUJOCO_GL` only if it is unset: EGL when `DISPLAY` is unset and `/dev/dri` exists,
    otherwise OSMesa. It sets `PYOPENGL_PLATFORM` to match.
  - This container has no `/dev/dri`.
- **With libOSMesa loaded, torch segfaults while building a stable-baselines3 policy, whichever is imported first.**
  This was seen on 11 Nov after a restart (exit 139).
- **Shared test processes and training run with `MUJOCO_GL=disable`.** `tests/conftest.py` sets it.
- **Rendering runs in a child process with `MUJOCO_GL=osmesa`.** `tests/test_stage2_clips.py` re-runs its two rendering
  tests that way.
- **The R5 and R6 entry points must assign `MUJOCO_GL=disable` themselves, not `setdefault` it.**
  - Stage 1's `train.py` relies on the default (`STAGE2_PLAN.md` S2-5).
  - On Colab, the scaffold's setup cell sets `MUJOCO_GL=egl` with NVIDIA's ICD for the whole kernel.
  - Clips must set `MUJOCO_GL=osmesa` in their own subprocess. Rendering on a CPU runtime stays untested until the
    smoke.
- **The first suite crash was GLFW asking for a display.** A test module had imported mujoco before designlab set the
  backend.

### 4.6 Tests, git and the tools

- **The full suite is `MUJOCO_GL=disable PYTHONPATH=. $PY -m pytest -p no:cacheprovider tests` from this
  directory.**
  - On 11 Nov it gave 575 passed, 107 skipped and 0 failed in the first session's checkout. That checkout held the PPO
    checkpoints and eval traces, which a fresh clone lacks even after the restore.
  - Compare a fresh clone with the restored counts in `tools/README.md` instead.
  - Write the log under a `logs/` directory (`mkdir -p logs`; ignored at any depth). A file such as `LOG` in the
    checkout is untracked, and the stop hook would then ask to commit it.
  - Do not add `-q`. The repository's `pyproject.toml` already sets `addopts = "--tb=short -q"`, and a second `-q`
    hides pytest's "N passed, M skipped" line.
  - Write pytest's own exit code into the log (`; echo PYTEST_EXIT=$? >> LOG`). A pass was once reported from a
    wrapper's exit code; it was corrected the same day.
- **The Bash tool's safety check can refuse a long heredoc that contains `rm`.** Write the script to a file with the
  Write tool and run the file. Use Edit or Python, not sed, for multi-line edits.
- **A stop hook requires every turn to end with all changes committed and pushed.**
  - Push with `git push -u origin <branch>`, retrying up to 4 times on network errors (2, 4, 8, 16 s).
  - Never force-push or rewrite history.
  - End each commit message with the attribution lines exactly as the session's system reminder gives them. Name no
    model anywhere else in a message or in code.
- **GitHub goes through the GitHub MCP tools.** `gh` is only for search.
  - Comment only when the owner asks, with the attribution footer.
  - Create no pull request unless asked; kuds/mesozoic-labs#667 already carries the branch.
- **Owner pages are private claude.ai artifacts.**
  - Build them with `tools/report_page.py`, or `designlab/stage2/look_page.py` for the look pages.
  - Check them with `node tools/shot_page.js` at 1280 and 390 px.
  - Republishing to the same URL needs the same file path in the same session, or the artifact's `url` (read it
    first).
- **The Google Drive tools are connected but untested.** The first session made no Drive call at all, and Stage 2
  moves binaries both ways through the owner's Drive (`STAGE2_PLAN.md` §6, [pipeline] §4).
  - **Up.** `create_file` takes a local file by reference: set `base64Content` to
    `sha256:<hex>:<absolute path>`, and the app uploads the file once the owner approves the call. The size limit is
    unknown.
  - **Down.** `download_file_content` returns the file inline as base64. A judged checkpoint is about 2 MB, and whether
    a result that size reaches the disk is untested.
  - **Test both early,** before the notebook is built: upload a small archive, then the real inputs bundle; download a
    checkpoint-sized `.zip` and check its sha256.
  - **If either fails,** name a fallback the owner performs and put it in the steps sheet. For inputs: the notebook
    rebuilds them with `tools/restore_inputs.py` and checks every sha256 (unproven on Colab's numpy and BLAS; the smoke
    tests it).
  - The inputs for one body and its `-X` come to about 40–65 MB, and for all seven registry bodies about 185 MB.
  - **Liveness and order on Drive.** A run's liveness is `STATUS.json`'s modified time, not `run.json`'s status: Stage
    1's R4 and both 75 cm R3 `run.json` files still read "running" (S1 N8). Order checkpoints by the step count in
    their names, never by modified time, which on a Drive mount is upload order.
  - **Check checkpoint pairs before a resume.** `train.checkpoints()` checks only that both files of a pair exist. Add
    the scaffold's checkpoint-pair integrity check before any resume from Drive.

### 4.7 Harness mechanics that cost time

- **Foreground `sleep` is refused.** To wait for something, use a Monitor until-loop (it expires after at most 30
  min) or a command run with `run_in_background`, which re-invokes the session when it ends.
- **Background commands.**
  - A `&` inside a `run_in_background` command returns at once, and the job is no longer tracked.
  - Shell variables do not persist between Bash calls; use absolute paths in every call (an unset `$SP` once meant
    a run never launched).
- **Use the venv's Python.** The system `python3` has no MuJoCo.
- **Do not edit a shell script while it runs.** `sh` reads a script as it goes. On 14 Nov an edit to
  `tools/keepalive.sh` during its run produced a syntax error when its loop ended. Run long loops from a copy.
- **Scheduled tasks.**
  - A `send_later` reminder cannot be turned into a recurring task (`update_trigger` failed with
    `missing_worker_target`); create a new task and delete the old one.
  - Their firings arrive as queued notifications: read them with ReadNotifications.
  - An hourly schedule written for minute 0 runs at its creation minute instead.
- **Workflows.**
  - Prompts that a script builds reach the agents indented. That broke a parser of their output once.
  - Resuming with changed `args` re-runs every agent whose prompt changed.
  - The cap of 2 agents at once applies per workflow: several workflows run side by side.
- **Usage limits are real.**
  - The account's weekly usage limit ran out on 13 Nov, after 39 workflows with 271 agents in all (one audit used 76).
  - Size a workflow's fan-out to the question, and keep a reserve for the owner's own use.
- **The repository's root rules drop or rewrite files silently.**
  - The root `.gitignore` ignores `logs/`, `models/`, `*.zip`, `*.pkl`, `*.npz` and `tensorboard/` at any depth. On 7
    Nov it dropped the round-one ledgers, kept in a `models/` folder, until it was renamed `mjcf/`.
  - `.gitattributes`' `* text=auto` rewrites line endings: the posture archive needed `* -text` for its SHA256SUMS to
    verify.
  - Check every new output with `git check-ignore -v` before relying on it, above all checkpoints, Colab logs and any
    sha256-manifested bundle.

## 5. Traps and open issues

### 5.1 First

1. **Restore before anything else in a fresh clone** (§4.1).
2. **Merge PR kuds/mesozoic-labs#667 with a merge commit, never squash or rebase.**
   - 616 committed records cite 42 commits by hash, and all 42 exist only on its branch.
   - A squash or rebase merge would leave main without them.
   - Its auto-generated description is wrong in places ("quadruped", "hip heights" for crown heights). Correct it
     before merging.

### 5.2 The bodies and the references

3. **G1 of the `-X` bodies is provisional.**
   - To finish it, run `PY=.venv/bin/python bash stage2/x/run_x.sh g1` on an otherwise idle machine. The script
     defaults to `python`, which has no MuJoCo.
   - Arm K's G1 failed its step rate on the K bodies and on the pick alike, so `-X` may fail it too (HEAD_X.md §3).
4. **The `-X` joint ranges are not cut.**
   - The clearance screen pairs no body with its parent, so it cannot see the neck against the chest or the jaw
     against the skull. The plant therefore keeps the MJCF limits: neck yaw ±0.6 rad, jaw 0–0.6 rad.
   - A seam sweep found no new contact, only growth of overlaps between proxies that already overlap at home:
     17.8 / 23.6 mm at the neck and 21.6 / 28.7 mm at the jaw. The ranges are therefore annotated, not cut
     (`refine.HEAD_X_RANGES_DECISION`).
   - The jaw cover overhangs its hinge (HEAD_X.md §4). Trimming it changes every `-X` record, so settle it at L1 and
     before any `-X` training.
   - The yaw stage's bearing, and today's pitch servo riding on the yaw stage, are not modelled.
5. **The Stage 1 env assumes 14 joints.** Its action and observation widths (`designlab/stage1/env.py:138-142,
   284-286`) need generalising before R5 can drive `-X`'s neck yaw and jaw.
6. **Look-aware turns and transitions do not exist yet.**
   - `turning.generate` refuses a look spec, and `transitions.gait_of` takes only `refs.walk_specs`' walks
     (`look.DEFERRED`).
   - The bar's 90° turn from 0.40 m/s, the two-step stop and every look-aware start, change and turn therefore need
     new paths beside the old ones. G2 keeps the old ones bit-identical.
   - The quick turn may need more than the plant's ±0.30 rad thigh twist, a range never assessed for packaging
     (`STAGE2_PLAN.md` §8; S1 §8, G5). Screen the turn reference's twist need first. If it does not fit, take it to the
     owner as Track D's lever 6.
7. **R5's zero action, R0L, falls in most 60 cm walks.**
   - Open loop at 0 ms, `60-N-T-10` keeps 3 of 20 nominal walks up and 2 of 20 of each walking turn, and no brisk or
     animal-pace walk (S1 §4).
   - So the 60 cm R5 must learn balance from a base that falls, and Stage 1's σ rule cannot be applied there.
   - S2-4's 2M steps on `75-M-H-10`, where R0 walks, say nothing about the 60 cm body. Its first evidence will be G8b
     at 5M steps on Colab, unless a short 60 cm run is made here first.
8. **Duty and look feature F3.** A fixed swing time puts the nominal walk's duty at about 0.69, over the spec's 0.65.
   Widening the band to 0.72 below 0.30 m/s is owner decision 2.
9. **The 75 cm V6 is over the knee's speed limit at the animal pace** (1.012; V6S 1.009). By the plan's rule it keeps
   its features and loses that speed rung unless the owner decides otherwise.
10. **`stage1/runs/2026-11-08/STAGE1_TABLES.md` still reads "works to 10 steps (20 ms)" for the 75 cm delay.**
    - That is the search's own machine-made reading; S1's correction note has the right one.
    - Leave the file as it is. Regenerate it only from commit 07a73f4 or earlier (`STAGE1B_PLAN.md` decision 6),
      because today's code reads Stage 1's records as made by other code.
11. **`designlab/stage1/env.py` imports `environments.shared.direction_commands` from the main repository.** It adds the
    repository root to `sys.path` if needed. A Colab bundle therefore needs the repository checkout, not the study
    alone.

### 5.3 Training, Colab and judging checkpoints

12. **The delay model works in whole 2 ms physics steps** (`env.py`'s latency queue).
    - The plan's 5 and 15 ms, and the 25 ms top of its randomisation, cannot be run as stated.
    - Before any run, fix the delays (for example 4 or 6 ms, 14 or 16 ms, 0–24 ms) or build a sub-step delay model.
    - Record the choice as a decision taken before results.
13. **The panels' policy rows refuse a checkpoint trained under other library versions** (`policy_rows.check_versions`:
    Python, numpy, MuJoCo, scipy, torch, stable-baselines3, gymnasium).
    - They read torch's version from `importlib.metadata` ("2.15.0"), while `train.py` records `torch.__version__`
      ("2.15.0+cu…" for PyPI's CUDA build). A run trained with Colab's torch is refused even in its own venv.
    - Before the first Colab run, decide how Colab checkpoints are judged here: either in a local venv that matches
      Colab exactly, Python's patch version and torch build included, or under a flagged cross-version rule recorded in
      every policy-row record.
14. **`train.py` resumes a run only when everything matches.**
    - It needs the same libraries (Python's patch version included), the same `designlab/stage1/*.py` sources
      (otherwise `--allow-source-change`, flagged), the same settings (`n_envs` included) and the same env configs.
    - The env configs hold `run_dir`. So keep one `REPO_REF` and one run path for a run's whole life, and pin numpy,
      scipy and torch in the notebook.
    - Colab's image changes without notice, so build the plan's flagged resume across library versions before the
      first launch (`docs/KNOWN_ISSUES.md`).
15. **SB3 archives segfault across Python minor versions.**
    - Every archive pickles `policy.linear_schedule`, a closure, and `train.py` and `policy_rows.py` load it with a
      plain `PPO.load`.
    - Loading an archive saved under another Python minor version segfaults, with no traceback
      (`docs/KNOWN_ISSUES.md`).
    - Before the first Colab run, replace the closure with a picklable schedule class, or load through
      `environments/shared/policy_loading.load_sb3_model`. Have the smoke load a Colab-saved archive here.
16. **G9's toe-clearance check reads Stage 1b's 10 ms turn traces, and a fresh clone has none.** Before G9, regenerate
    them, or amend G9 (labelled, before any run) to use R0L's own 10 ms turn traces. To regenerate, re-run the R2
    10 ms turn conditions into a scratch run directory with `--inputs stage1/runs/2026-11-10-speed`, as
    `run_speed_round.sh` runs them.
17. **S2-4 has no reading point.**
    - A 2M-step run never reaches the plan's 2.5M-step monitor.
    - Before the run, fix S2-4's rule in the plan and label it. For example: §6's monitor on the final checkpoint
      against R0L on the same 18 episodes, with §6's gain rule.
18. **Moving binaries through Drive is untested** (§4.6).
19. **`tools/memguard.sh` protects only `designlab.stage1 train` by default.** Set `PROTECT` for Stage 2 trainers, or
    the guard kills them first.

### 5.4 Pages, logs and scope

20. **The viewers' blind sheet stores nothing.** Its "Copy my scores" button is the only way back: ratings arrive only
    if the owner pastes them.
21. **`MUJOCO_LOG.TXT` fills with "Nan ... QVEL" warnings.** A deliberate test writes them (`tests/test_stage1.py`, the
    BADQVEL check), and they are harmless.
22. **Out of scope.** Eleven registry and contract tests fail on PR #662's clean checkout, and merging PR #664
    conflicts in 4 files (`REPRODUCTION.md`). They belong to the main repository's integration, not to this study.

## 6. Next steps toward long training runs

This is `STAGE2_PLAN.md` §7 in order, with what a new session does first.

1. **Environment.**
   - Run `tools/setup_venv.sh`, then `tools/restore_inputs.py restore` and `check`, then the full test suite.
   - Compare the counts with the restored counts in `tools/README.md`, not with §4.6's first-session figures.
2. **The owner's look picks (L1).**
   - The owner picks the features, decides whether a feature may cost a speed rung, and settles the duty band
     (decisions 1 and 2; `STAGE2_PLAN.md` §7).
   - Then freeze the approved reference by hash (A.5).
   - The look review page and the viewers' blind sheet are in §8.
3. **Before any run, settle the open rules (§5.3):**
   - the whole-step delays;
   - S2-4's reading point;
   - G9's 10 ms turn traces;
   - how Colab checkpoints are judged here;
   - the resume across library versions;
   - a picklable schedule.
4. **R5** (B.1).
   - Build: generalise the env for `-X`; add commands, the references at nominal and 0.30–0.50 m/s, the look-aware turns
     and transitions (§5.2), the reward terms and the randomisation.
   - Gates: G8a (R5's zero action reproduces the panels' R0 exactly, at 0 and 10 ms) and G9 (the new measures read as
     designed on the references).
5. **R2L**, the tracker refitted as Stage 1b's speed fit on the bar's tasks, here (B.2), with fixed delay draws (§3).
6. **Validation on this CPU** (B.3).
   - Measure rates and memory per floor and process count.
   - Check kill and resume.
   - Run the σ and learning-rate probes.
   - Start a training run with `MUJOCO_GL=disable`.
7. **S2-4.** Train 2M steps here on `75-M-H-10` with training's delay draw. Colab starts only if R5's 10 ms monitor
   shows a gain over R0L under the rule fixed in step 3. Otherwise tell the owner first.
8. **S2-5, the notebook.**
   - Write `stage2/stage2_training.ipynb`, on the scaffold `notebooks/sb3_training.ipynb`, with its steps sheet and a
     sha256-manifested Drive bundle.
   - Then run it here end to end on a tiny budget, with a kill and resume, and dry-run the bundle from a fresh clone.
   - The Drive transfers are tested first (§4.6).
   - Then comes the owner's day-one smoke session, about an hour.
9. **S2-8.** Fix the budget once after the smoke: 40M env steps per run, or fewer if the smoke's rate cannot reach that
   in 20 h. Then run Colab's three phases (High-RAM CPU, up to 3 runtimes, 24 h sessions, a 220 runtime-hour cap,
   P29).
10. **Panels, levers and report.**
    - Run the panels here on the judged checkpoints from Drive (Track C).
    - Run the hardware levers (Track D; `STAGE2_PLAN.md` §4 D), each labelled "untrained transfer" until fine-tuned.
    - Then write the report.

**The owner's 14 Nov request: the 75 cm build with a Jetson Orin Nano.** PR kuds/mesozoic-labs#668 studies it.
`JETSON_PLEM_REVIEW.md` reviews it and sets out the track the next session takes (`NEXT_SESSION_PROMPT.md`).

**The hardware questions that matter most are the bench tests:**
- each drive type's command-to-motion delay and stiffness (draft A17);
- the STS3215 tail servo at the speed walks' loads (A1);
- the treads' grip on the owner's floors (P21);
- the joints' friction and damping (the MJCF defaults, with no decision-log row yet).

The delay matters most. At 10 ms every 60 cm walk falls under the fitted tracker (S1b §4). A measured delay above S1b
N8's figures rules out that controller family, not every controller, and one below them still leaves grip and the
controller to check.

**Before any hardware lever builds on the 60 cm picks' ankle drive,** confirm its supply. MyActuator lists the
RMD-X4-P12-10-E (V4) as discontinued, against E6, though it is still sold. The other route is to re-check the ankle
against its successor, the X4-10-L (S1 §6 item 11).

## 7. Waiting for the owner

- **Stage 2's open decisions** (`STAGE2_PLAN.md` §7):
  - 1: the look features, and whether one may cost a speed rung.
  - 2: the duty band.
  - 7: raters for L1 (3–5 new viewers) and L2 (at least 12).
  - 8: when the bench tests run.
- **Colab confirmation.** P29 records the choice; the owner still has to confirm that the plan provides 24 h sessions,
  background execution, High-RAM CPU runtimes and 3 runtimes at once (`STAGE2_PLAN.md` §6).
- **Draft decision-log rows, not yet in `DECISION_LOG.md`:**
  - S1 §7: F26 (with the 10 Nov correction from S1b N11), the A15 update and the A16 update.
  - S1b §8 and N11: F27, the A1 update (bench-test the tail servo first) and A17 (new: the servo delay and stiffness
    assumption).
  - Proposed here: a new A row for the joints' damping (0.2 N·m·s/rad) and friction loss (0.15 N·m), the MJCF
    defaults on every joint, to be replaced by bench measurement.
- **P25 stays open.** On 8 Nov the owner asked that, once Stage 1's results were in, the study explore what covering
  the remaining gap at 60 cm would take. S1 §8's gap map began it.
- **Owner constraints that bind the work:**
  - The demonstration floors are hardwood, vinyl and museum floors, perhaps thin compact carpet; long pile is outside
    the use (P26). Training and verdicts stay on hard floor and robot_floor_carpet.
  - The jaw is modelled in simulation only; P2's deferral stands for the build (P29).
  - P4's prototype budget is $10–20k. The high estimates already exceed it: $20.5k for `60-N-T-10` and $25.7k for
    `75-M-H-10`, against nominals of $14.8k and $18.5k (their round-two records). Cost every hardware lever against it.
- **PR kuds/mesozoic-labs#668's decisions** are listed in `JETSON_PLEM_REVIEW.md`.

## 8. Published pages

The owner's review pages are private claude.ai artifacts; only the owner can open them. Their sources were built in the
first session's scratchpad and are not in the repository. Each regenerates from the committed records and reports:
`designlab/review_page.py` and `review2_page.py` for rounds one and two, `tools/report_page.py` for the Stage 1 and 1b
reports, and `designlab/stage2/look_page.py` with the clips for the look pages.

| Page | Link |
|---|---|
| Round one review | https://claude.ai/artifact/ApMGC29DvSyDmKPycr9vSs |
| Round two review | https://claude.ai/artifact/A67P2YGERSSfa7CAUic6qT |
| Stage 1 report (corrected 10 Nov) | https://claude.ai/artifact/XMkc9GjZzRcrqF3ANiRqWZ |
| Stage 1b speed round | https://claude.ai/artifact/VbBcrpxuyxjF2YhoJqG4rn |
| Stage 2 look review (gate L1) | https://claude.ai/artifact/MG1yh5ztqJwVdFPs6GmkkY |
| Viewers' blind sheet (L1 ratings) | https://claude.ai/artifact/QVktxN3pHsov2LTVs6d69c |
