# Stage 2, Track B: rules fixed before the runs

**Status (written and settled 15 Nov 2026).** These rules were fixed on 15 Nov 2026. The study's lead gave the rulings
they rest on at 02:45 UTC, before the build of Track B's code began (02:50 UTC) and before any look set was built. When
this file was committed, no Track B episode, fit or training run had run on a Stage 2 reference set: no look set had
been frozen, no closed-loop episode had run on a look set, no fit of the refitted tracker had been started, and the only
training runs made were the builds' tests of a few updates on stub envs. What the builds ran between the rulings and
this commit, none of it on a Stage 2 set and none a record, is listed under the next heading. Simulation only: the text
changes no body, and nothing is adopted without the owner (P24). It stands under the owner's decisions of 14 Nov 2026
(`DECISION_LOG.md` P30 and P31; P31 sets money budgets aside and leaves the compute budgets below as they are). Track B
is the learned controller of `STAGE2_PLAN.md` section 4 B, with the compute plan and stop rules of its section 6 and the
amendments S2-9 to S2-18 of its section 9. This text holds Track B's rules: the learned controller R5, the policy-only
baseline R6, the refitted tracker R2L, the gates G8a and G9, the monitor and the stop rules, validation, the notebook,
the Drive bundle and the day-one smoke; its section L holds the rules by which the look is carried through moves and the
look sets are built. Sections A, B, C, H and R of `STAGE2_75_PLAN.md` apply to every body and set as S2-16 extends them;
this text adds to them and names every place where it departs. It is self-contained: every formula, scale and number a
rule needs is in it, and the designs that explain them (session notes of 14 and 15 Nov) are not in the repository. It
was written from the architect's second draft, made after three independent reviews of the design, with the rulings that
the study's lead gave on 15 Nov 2026 on every point that draft had left open: each ruling stands in the rule it settles,
marked "[lead, 15 Nov, before any Track B run]", and the section "Settled by the lead" lists them. The text was then
reviewed independently. What the review found wrong in wording, labels, arithmetic or cross-references is corrected
here, and the last paragraph of "Settled by the lead" says where. The review also left 25 points on which it asked for a
decision that nobody had made. The commit that adds this file (15 Nov 2026, 03:34 UTC) holds them in a closing section,
"Open for the lead", each with the reviewers' proposed wording. The lead ruled on every one of them on 15 Nov 2026,
after that commit and before any closed-loop episode on a look set, any fit and any training run of Track B: they too
are rules fixed before results. Each settled wording stands in the rule it settles, marked
"[settled 15 Nov, before any Track B run]", and the last section, "Settled by the lead on 15 Nov 2026, before any Track
B run (second list)", lists the 25 by point. Some of them change or refine an earlier ruling of the lead and say so
where they stand: the learning rate is one per body (O3; the ruling of 02:45 UTC had one rate for every body and head);
inside a move the feet are flat while the gait is changing (L10 and L11; the choice of 02:45 UTC read "inside every
move"); and the fallback of join rule J1 has one cut, taken per body (L12 and L13; the choice of 02:45 UTC had two cuts
in order, for a V6 move). The settled wording of section L (points 4, 5 and 6) was fixed after the look work package's
first kinematic numbers existed, which the next heading describes, and is labelled so. On 16 Nov 2026 the lead settled
ten conflicts between rules that the writing-in of the second list had left (the third list), again before any Track B
run, fit or training run and before any look set was frozen; the 75 cm round's panels and their add-on had run by then.
Each of those wordings stands in its rule, marked "[settled 16 Nov 2026, before any Track B run]", and the amendment
that closes the last section lists the ten. Paths are relative to `hardware/trex_form_study/`. A rule changed after a
result is labelled so. Check: the commit that adds this file, the commit that adds the settled wording and the last of
the commits that add the wording settled on 16 Nov 2026 are ancestors of the commit named in every Track B record
(`sessions[].commit`, or `commit` in a gate's, a probe's or a fit's record) and in every look set's `freezes[].commit`
that a Track B record uses.

## What had been run when these rules were fixed

- Everything listed in `STAGE2_75_PLAN.md` under the same heading.
- No Track B episode, fit or training run on a Stage 2 reference set, and no fit of R2L at all. No look set had been
  frozen, and no closed-loop episode had run on one.
- Not in that text's list, and outside Track B: the record of gate L1's outcome (`stage2/look/L1_FREEZE.json`: the
  hashes of the look ladder's committed V6 rungs; nothing was simulated for it; committed at 00:41 UTC on 15 Nov, after
  that text); the Jetson body `75-M-H-10-J`, made in a parallel worktree before that text's commit (its build and
  screens at 00:14 UTC on 15 Nov, its Stage 1 inputs and gates at 00:35 UTC) and merged at 02:02 UTC, with its screens,
  its Stage 1 gates G2 and G3, and gate G1 provisional (`stage2/j/JETSON_BODY.md`; no rule below rests on them); the
  study's test suite. The builds of the 75 cm track, of Track B and of the look work package were under way in their own
  worktrees: rule X1 below and rule R4 of `STAGE2_75_PLAN.md` say what they may run, and every check episode they run on
  a Stage 2 set is listed in the two `dev_episodes.json` files those rules name.
- By the build of Track B's code, between the lead's rulings (02:45 UTC) and this commit, none of it on a Stage 2 set
  and none a record: test episodes of the R5 env and of gate G8a's comparison on the view `S1` (Stage 1's stock
  references; first committed in the builders' worktree at 02:59 UTC); test runs of PPO of a few updates on stub envs
  (the loader's and the resume's tests).
- The look work package's kinematic scratch, begun at 02:54 UTC on 15 Nov, after the lead's rulings on section L (02:45
  UTC) and before this file's commit: on `75-M-H-10`, the V6 `nominal` and `v0.4` walks written as windows in Stage 1's
  format from the look ladder's committed specs, and V6 moves built with the first version of the look's laws
  (`moves_look.py`, branch `wt/wp-l`), among them the two-touchdown stop at 0 correction passes and a change at 12, with
  their join numbers printed. At 03:18 UTC that work package started its first build of part of set `V6` on `75-M-H-10`
  (the walks `nominal` and `v0.4`, the start, the change, the stop and the quick turn to the left); at 03:33 UTC,
  minutes before this commit, that build was still running: it had written those two walks, the start, the change and
  the stop, and a `set.json` without a freeze (`freezes` empty), in its own worktree and uncommitted. No look set was
  frozen. All of it is prescribed motion: no record and no closed-loop episode. The rules of section L as committed were
  the design's of 14 Nov and the lead's choices of 02:45 UTC: none was written or changed on those numbers, which the
  author of this text did not read. The points of section L that the review left open (points 4, 5 and 6 of the second
  list) were settled by the lead after those numbers existed: each settled wording says so (L10 to L13 and "Windows"),
  and a rule of section L changed after this file's commit is labelled as changed after them. This list is as found on
  disk at the commit.
- [settled 15 Nov, before any Track B run] The lead confirmed this list as written (point 25 of the second list), with
  this completion: the Track B builds' own test episodes ran since 02:59 UTC on the view `S1` and on stub environments
  only; the look work package's kinematic builds of V6 windows and moves began at 02:54 UTC, are prescribed-motion
  builds and no closed-loop episode, and their join numbers are what L12's trigger reads. The corrections that the
  author applied after the review are confirmed. The builders of the remaining packages are given this file's name and
  its citation convention by the lead.
- When the second list was settled, later on 15 Nov 2026: the lead's rulings are headed "Fixed before any closed-loop
  episode on a look set, any fit and any training run of Track B". When their wording was written into this text (03:43
  UTC) no look set had been frozen: the one look set on disk, `V6` on `75-M-H-10` in the look work package's worktree,
  had a `set.json` with `freezes` empty.
- When the third list was settled, on 16 Nov 2026 (`stage2/design/TRACKB_RULINGS3_16NOV.md`): no Track B run, fit or
  training run had been made, and no look set was frozen (when the wording was written in, the look work package's
  builds of `V6` and `V6noF5`, in its own worktree on branch `wt/wp-l`, had `set.json` files with `freezes` empty, and
  the working branch held no look set and no `stage2/tb/` directory). The 75 cm round's panels (tasks 3 and 4 of
  `STAGE2_75_PLAN.md`) and their add-on had run. On 16 Nov the lead lifted that text's rule R4 and X1's first sentence
  for the frozen 75 cm sets, and the training env's fix round compared the env with the bar's runner on `V0` under `R0`
  rows, a check that leaves no record (`STAGE2_75_PLAN.md`, amendment 13). Where a ruling of the third list leans on the
  75 cm round it says so: Q5's caps use the rates of that round's committed `COST.md`, and X1's first sentence names the
  lifting. For those caps the bar runner's dry run was made on 16 Nov on the frozen sets `V0` and `F25F9cl`; a dry run
  plans episodes and runs none. It was also made once on the look work package's unfrozen build of `V6noF5`, as a probe
  that is no record (Q5). For L13 the arrays of that work package's unfrozen turn in place were read (prescribed
  motion).
- Scratch experiments of 15 Nov on Stage 1's own run directory (`stage1/runs/2026-11-08`: stock references, no Stage 2
  set), by the architect and by the three reviewers of the design, on a stub env (Stage 1's env with the open-loop base
  in place of the tracker; Stage 1's observation and reward), one to six episodes per setting. They are scratch: none is
  a record and none may be cited as a result.

| Scratch experiment | Numbers (not records) | Bears on |
|---|---|---|
| The panels' open-loop row against a tracker in R2 mode with no feedback term and no ankle cap, stepped as an env steps it, on two separately built plants; `75-M-H-10`, 6 s, one seed | state digests equal at 3,000 of 3,000 physics steps at 0 ms and at 10 ms | Z1 |
| The stub env on `75-M-H-10`, one process | 411-460 env steps/s on hard floor, 352-422 on robot_floor_carpet | O1, Y11 |
| The base under action noise, 10 Hz filter, k = 5 | `75-M-H-10`: sigma 0 and 0.01 up 6 of 6; 0.02 up 1 of 6; 0.05 and 0.1 none (3.8 s, 3.3 s). `60-N-T-10`: sigma 0 up 1 of 6; 0.1 none (2.6 s). At equal filtered action RMS a 20 Hz filter survives 0.98 x as long as 10 Hz; at equal sigma 0.85 x | O2, O3, O4, M6 |
| PPO on the stub, sigma 0.1, 5 epochs, batch 256 | with a 16,384-step rollout the learning rate 1e-4 ran 320 then 206 of 320 minibatch steps, 3e-5 all; at lower sigma far fewer run | O1, O3 |
| The zero action on `60-N-T-10`, 5 episodes per task | walk up in 2 of 5 at 0 ms and at 10 ms; brisk walk and walking turn 0 of 5 at both; turn in place and stand 5 of 5 | M6 |
| A clock held 25 % slow against the zero action | costs 1-7 % of Stage 1's reward per step; on `60-N-T-10` it survives longer | W5, M3 |
| The zero action's drift on `75-M-H-10`, 5 episodes | 0.08-0.22 m from the line in 10 s (limit 0.25 m), heading within about 1 degree | E3, W5 |
| A start from a frame inside the nominal stride with the delay queue filled from the reference; `75-M-H-10` | up 8 s from 6 frames at 0 and at 10 ms on hard floor; stance-foot load 0.8-1.4 body weights at reset; falls after 2.4-3.6 s on robot_floor_carpet | T3 |
| The leg-gap reading on `60-N-T-10` | the smallest gap is a fixed bracket pair at 4.0 mm in every pose | W4 |
| A body standing at home while the reference walks, under the first draft's reward | 0.39 of 1.25 per step | E2 |
| An archive saved with module-level schedule objects; a dummy env, 3 PPO updates | no bytecode member; fingerprint reproduced with difference 0.0; 70 of 120 minibatch steps ran | K9, O3 |
| A resume on a stub env with SB3 2.9.2 | loading with the env and letting SB3 reset: every tensor differs from the uninterrupted run; with `force_reset=False` and the boundary at the first rollout start: bit-identical. An env after 7 randomised episodes equals a fresh one after a seeded reset when its trackers were built on the nominal model | K1, K3 |

No constant below was set on a Stage 2 closed-loop result. Constants marked "(scratch)" lean on the table. The join
limits that section L cites (1.0 mm, 2.0 mm, 1e-3 rad) are those of `STAGE2_75_PLAN.md` A6 and A7, set on that text's
prototypes; section L's two figures for the V6 windows (8.9 ms, 3.5 mm) were read from the look ladder's committed
arrays when the design was written and are no record.

## In brief, for the owner

- **What is trained.** One learned controller per body, head and training seed, called R5. Every 20 ms it nudges the
  servo targets of the reference walk (by at most 0.35 rad on the legs and 0.15 rad on the head and tail) and the speed
  of the walk's clock (by at most 25 %). Its "zero action", doing nothing, is the reference played open loop with no
  balance feedback: by construction the open-loop row that the agility bar carries as its baseline (R0L); gate G8a
  checks that it is so, step for step.
- **Where each thing is made.** The long training runs are yours, in the notebook on Google Colab ("Colab"), saved to
  your Google Drive: R5 on both sizes, both heads and two seeds, and one comparison per body, R6, a policy that sets the
  servo targets alone with no reference underneath. R6 is expected to learn only to stand or shuffle, as Stage 1's
  policy of that kind did; that will not be news. Everything else is made or checked here first: the look's reference
  motions, a balance controller refitted with servo delay in its mix (R2L, a baseline beside R5), the two gates below,
  and one short training run.
- **Two gates before any training.** G8a: R5's "do nothing" must reproduce the bar's open-loop row exactly, step for
  step, at 0 and at 10 ms of servo delay; if it does not, nothing trains on that body. G9: the new readings (toe
  clearance over each foot's own swing, and the four look measures: body bob, head pitch, tail phase, toe-off) must read
  as designed on the reference motions themselves, before any closed-loop episode runs on them.
- **One go or no-go before Colab.** 2,000,000 training steps here on the 75 cm body, then 18 fixed episodes at 10 ms of
  delay, beside the open-loop row on the same 18. Colab starts only on a gain: 2 more of the 18 succeeding with the
  look, or 1 s longer on its feet on average (time bought by slowing the walk's clock is not counted). A second route,
  ruled in by the lead before the run: where the open-loop row already stays up to within a second of the episodes' full
  length, so that the survival route is closed, a gain is also 2 more of the 18 succeeding with the look bands when the
  neck and tail servos' sustained load is set aside. That second route belongs to this one run: in every later run's
  checks the same count is printed and decides nothing. The run is made once, and its policy is never judged on the
  panels. No gain comes to you first, with a table of what failed for both rows, whatever the cause.
- **Early training episodes are short, and that is expected.** The plan's starting exploration noise is five to ten
  times what the open-loop walk tolerated in scratch tests: it knocks the base over after about 3 seconds, on both
  sizes. Half of the training episodes therefore start in the middle of a task, so that the policy still meets every
  part of it.
- **The 60 cm body is the hard case.** Its open-loop walk falls within seconds. A longer run here (5,000,000 steps)
  comes before any 60 cm Colab job. If its slow walk does not stay up in 2 of 3 episodes by then, you are told before
  any 60 cm Colab hours are spent, with the remedy that a mapping fixed in advance reads from the run's record.
- **What the monitor stops.** During a Colab run the same 18 episodes run every 2,500,000 steps. They decide only
  whether a run goes on: at 5,000,000 steps the slow walk must stay up in 2 of 3 episodes (if it does not, one remedy
  fixed in advance is applied inside the running job, and a second failure at the 10,000,000-step check stops the run);
  from 20,000,000 steps, three checks in a row without a gain stop it, a gain being counted against the best of the
  open-loop row and of every earlier check of the run. The second route of the go or no-go does not count here, so a 75
  cm run that is ahead only in that printed count can be stopped from 25,000,000 steps. A run ends by its budget, by one
  of these two stop rules or by the cap; a run that halts or disconnects is resumed, and none is stopped or left
  unresumed for any other reason once its checks have been seen. A check is never a verdict.
- **How a Colab checkpoint is judged.** Here, never on Colab: on the study's 20-episode panels, in the study's own
  software versions. Every saved policy carries a fingerprint (256 fixed inputs with the outputs it gave when it was
  saved). It is judged only if it reproduces them here to within 0.00001; if it does not, it is not judged and you are
  told. A checkpoint trained under other software versions is judged under a rule that flags it. Every Colab run that
  passed the first gate is judged at its last kept checkpoint, whatever ended it, and every table says what ended it.
- **The day-one smoke (your first session, about an hour).** It checks that the pinned software installs, that memory
  suffices, how fast the runtime trains (the first smoke session that records this rate on the runtime type you use
  fixes the length of every run; a repeated smoke does not change it), that a killed job resumes from Drive, and that a
  saved policy loads here. No training job is queued until those pass.
- **Look choices made now, for you to know.** Toe-off with a rolling foot lives in the steady walks: inside a start,
  speed change, stop or turn the feet stay flat while the gait is changing, and the move's last stances, which end after
  the change is over, roll as the walk does, so that the move joins the walk (the first wording, "flat inside every
  move", was refined by the lead before any run). The turn in place is built as a steady gait of its own: one stride,
  repeated, without stride-to-stride variation and with flat feet. The refitted balance controller gets the look without
  toe-off, which it cannot follow. A speed at which a leg drive is over its limit is dropped, and the look is kept. The
  expressive-head references hold the neck yaw and the jaw at their home angles. A look walk is a loop of 9 to 14
  slightly different strides: if a start, speed change, stop or turn cannot leave that loop in mid-course within its
  gates, every look set of that body, for both heads, is rebuilt so that moves leave and enter the loop only at its end,
  and a command then comes after about 10 s of walking. There is no second fallback: a move that still fails is recorded
  "not available", but for two: if the start or the change to the 0.40 m/s walk still fails, that body's look set is not
  frozen and the point goes to the lead [settled 16 Nov 2026, before any Track B run: the two exceptions].
- **What may come back to you.** No gain at the go or no-go. A failing 60 cm local run. A body whose turn in place
  cannot be built within its gate: it then trains without it, and the report says the plan's list of references is not
  met there. A quick turn that cannot be built within its gates on a body: its turn items then read "not run". R5
  failing the slow-walk check (G8b) on both seeds of a body: that body is then reported with the refitted balance
  controller, and you are told before more Colab hours. Uploads of the input files that fail (then a fallback that you
  perform or approve). A Colab rate too low for 15,000,000 steps in 20 hours. At each launch you check four printed
  lines; a fourth line ending `BELOW FLOOR` means the runtime is too slow: tell the lead.
- **Compute.** Here, in core-hours, all estimates but R2L's budget: about 6.5 for the gates, the probes and the go or
  no-go run (inside the plan's 10 for validation); about 5.2 for the 60 cm run, on top of the plan's estimate of about
  80; a budget of 12 for R2L in place of the plan's 8 (a third fit, on the candidate walk of the 75 cm track). R2L's
  panel records, and the re-runs of the 75 cm track's two balance-controller rows beside them, have caps of their own
  beside that budget (Q5). Each cap is set from a planning run on the frozen reference set before that row's first
  record; none has a number yet [settled 16 Nov 2026, before any Track B run: these two sentences]. On Colab: 40,000,000
  steps per run, or fewer if the smoke's rate cannot reach that in 20 hours (about 556 steps per second), never under
  15,000,000 without you. The plan's cap of 220 runtime-hours or 8 days stands: when it is reached, running jobs stop at
  their next checkpoint, every run is judged at the last checkpoint its comparison reached, and jobs not yet started are
  cut in the plan's order (fine-tunes, then R6, then the expressive head's second seed); the 8 days count from the start
  of the first training session. Nothing is adopted.
- **Settled after the review.** The independent review of these rules left 25 points for the study's lead. The lead
  settled all of them on 15 Nov 2026, before any Track B run (the last section lists them), and this brief was changed
  where an answer touched it. Four answers have no bullet above. Each size trains at its own learning rate: a short
  probe finds, for each head of that size, the largest rate at which every training step runs, and the size's runs use
  the smaller of the two (the first runs made here use the heads probed by then). The two sizes may differ, comparisons
  are made within a size, and the report says so; the lead's first ruling had one rate for all and was changed before
  any run. A policy that stands still cannot pass the stop test: for these controllers a stop counts only if the robot
  walked at no less than 0.9 of the commanded speed over the stride before the stop command. The refitted balance
  controller on the 75 cm track's candidate walk is reported with its counts and the candidate's label and without a
  verdict word, and it is not run again on fresh seeds. The training reward has one penalty more than the plan lists,
  for the two legs coming within 3 mm of each other, and a training episode ends when they overlap at two readings in a
  row.
- **Settled on 16 Nov 2026.** Writing those answers in left ten places where two rules disagreed. The lead settled them
  on 16 Nov 2026, again before any Track B run (the amendment that closes the last section lists them), and this brief
  was changed where an answer touched it: the two exceptions to "not available" under the look choices, and the caps
  under "Compute".

## Words, names and tags

As `STAGE2_75_PLAN.md`, whose list of words stands, with these additions.

- **Documents.** "The plan" is `STAGE2_PLAN.md`; "plan B.1" to "plan B.4" are the items of its section 4 B, "S2-n" its
  rule n, and S2-4 its go or no-go run. "Decision n" is `STAGE1_PLAN.md`'s. S1 and S1b are the Stage 1 and 1b reports.
  "Lesson n" is `LESSONS_LEARNED.md` section 3's. [look] is `stage2/research/research_look.md`. C8 and "rule R4" are
  rules of `STAGE2_75_PLAN.md`, and a rule of that text is cited with its file wherever it could be mistaken for one of
  this text's. "The design" is the session notes of 14 and 15 Nov behind these rules; they are not in the repository,
  and each rule states its content in full. Tracks C and D are the plan's agility tests and its hardware levers.
- **Controllers (rows).** R5: the learned controller, the plan's judged policy (plan B.1). R6: the policy-only baseline.
  R0: the panels' open-loop row; R0L: that row in Stage 2, which is R5's zero action. R2: Stage 1's balance tracker;
  R2L: R2 refitted with servo delay in its mix. R5G: R5 on an expressive head with the bar's head script (E5). A "policy
  row" is R5, R6 or R5G. `@10ms` after a row's name is its servo delay.
- **Gates.** G1 to G7 are Stage 1's; G5 is its selection of the fitted gains. G8a (the zero action is R0), G8b (the slow
  walk at 5,000,000 steps) and G9 (the readings read as designed) are the plan's. "Gate L1" is the owner's pick of the
  look (S2-15).
- **Bodies and sets.** `75-M-H-10` and `60-N-T-10` are the two picks with the current head; `-X` marks the expressive
  head (neck yaw and jaw). nj is a body's number of hinges (14; 16 on `-X`); H its hip height. A *look set* is a
  reference set built with the look's laws: `V6` and `V6noF5` (section L). `S1` is the read-only view of Stage 1's own
  run directory, used for development only. "Frozen" is rule A13 of `STAGE2_75_PLAN.md`.
- **Training words.** An "env" is one simulated robot with its task, as the trainer steps it; an "env step" is 20 ms, 10
  physics steps of 2 ms. k is a servo delay in physics steps. "Rate" is the commanded clock, 1.05; c is the reference
  clock. PPO (proximal policy optimisation) is the training algorithm, from the library stable-baselines3 (SB3); a
  "rollout" is the steps gathered between two updates, a "minibatch step" one gradient step of an update; the "KL limit"
  (`target_kl`, a bound on the Kullback-Leibler divergence) ends an update early when the policy has moved too far.
  Sigma is the exploration noise, as a share of each action's range. A "check" is a monitor run (M1). CMA-ES (covariance
  matrix adaptation evolution strategy) is the search that fits the tracker's gains. mu is the floor's friction
  coefficient; RMS root mean square; CoM centre of mass; ZMP zero-moment point; DCM the divergent component of motion;
  IMU the inertial measurement unit (gyro and accelerometer).
- **The bar.** "The bar", "the panels" and "readings 1, 2 and 3" are as in `STAGE2_75_PLAN.md`: reading 1 is "shown with
  the look", reading 2 "shown" (every limit), reading 3 "shown but for A1" (every limit but a neck or tail servo's
  sustained planning limit; A1 is the decision log's row). The "look bands" are reading 1's four limits (U3).
- **The look.** "Look feature Fn" is [look] 3.1's: F1 and F2 the body's bob, F3 a swing of fixed duration, F4 the shape
  of the foot's lift, F5 toe-off with rolling contact, F6 trunk roll, F9 the counter-phased tail, F10 the stabilised
  head, F11 the head-first turn, F13 a longer stride at the animal pace, F14 stride-to-stride variation. These are not
  the decision log's rows of the same names. The "look ladder" is the plan's ladder of look variants, V0 (today's walk)
  to V6 (every feature), built and screened at three speeds before gate L1 (`stage2/look/`); a "rung" is one speed of a
  ladder or of a set.
- **Colab words.** Colab is Google's hosted notebook service; a "runtime" is one of its machines, a "session" one start
  of the notebook on one; vCPUs are its processor cores, RAM its memory; Drive is Google Drive. REPO_REF, RUN_ID, RESUME
  and the other names in capitals are the notebook's parameters. The "launcher" is the notebook's process that runs the
  queue of jobs and copies files to Drive. SIGKILL is the signal that ends a process at once. sha256 is a file's hash.
  OSMesa is the software renderer of the clips; MP4 their format.
- **This text's rules** are E1-E6, T1-T3, W1-W5, J1-J5, O1-O4, M1-M6 and M3b, U1-U6, Z1-Z2, Q1-Q6, K1-K9, Y1-Y11, N1-N5,
  X1, and in section L the rules L1-L8 and L10-L15 and the join rules J1-J3. Two names are used twice, because both
  sources' names are kept: section J's rules J1 to J5 are randomisation rules, and the design's join rules, which the
  lead's rulings cite by their names J1, J2 and J3, stand in section L and are always written "join rule J1" to "join
  rule J3"; "rule L1" is this text's, "gate L1" the owner's look pick. Outside a rule's head, S1 is Stage 1 (its report;
  in a tag, its rule) and the code-styled `S1` the view of its run directory, G1 to G9 are gates, H is the hip height,
  and `-X` is a head. Code and records that cite a rule of this text name the file with it ("STAGE2_TRACKB_PLAN rule
  K3").
- **Tags.** **[plan]**, **[S1]**, **[NEW]**, **[differs: plan]** and **[differs: S1]** as in `STAGE2_75_PLAN.md`;
  **[S2-n]** an amendment of the plan's section 9; **[differs: C8]** a departure from `STAGE2_75_PLAN.md` rule C8;
  **(scratch)** a constant leaning on the table above. **[lead, 15 Nov, before any Track B run]** stands on a rule that
  the study's lead settled on 15 Nov 2026, on a point the draft of these rules had left open; where the rule departs
  from the plan or from another rule the tag goes on "; differs:" and names what it departs from, and the rule's text
  says what was ruled. **[look choice]** marks a choice about the look that the lead made on 15 Nov 2026, before any
  look set was built, and that the owner is told about (the brief above). **[design]** marks a rule taken from the
  design of 14 Nov that the lead let stand. The rulings are session notes, not in the repository.
  **[settled 15 Nov, before any Track B run]** stands directly before wording that the study's lead settled on 15 Nov
  2026, after this file's commit, on one of the 25 points that the independent review of this text had left for the lead
  ("point n of the second list"; the last section lists them); the wording so marked runs to the end of its paragraph or
  list item, or to that paragraph's "Check:". Where the settled wording is a phrase inside an older sentence or a table
  cell, the mark follows it and names the phrase; in a rule's tags the mark names what was settled in that rule. Where
  such a ruling changes or refines an earlier ruling of the lead, the mark or the sentence after it says so; in section
  L the mark also says that the wording was settled after the look work package's first kinematic numbers existed. Those
  rulings too are session notes. **[settled 16 Nov 2026, before any Track B run]** stands in the same way before wording
  that the lead settled on 16 Nov 2026, on one of ten conflicts between rules that the writing-in of the second list had
  left ("ruling n of the third list": the amendment that closes the last section lists them, and the rulings are
  `stage2/design/TRACKB_RULINGS3_16NOV.md`, rulings 1 to 10). Where it follows the wording it names it, and a sentence
  that it replaces is quoted once. In section L it also says that the look work package had built look sets by then
  (prescribed motion; none frozen). **[differs: C7]**, **[differs: A8, A11]**, **[differs: B6 and T1]** and **[differs:
  rule R4's wording]** are departures from those rules of `STAGE2_75_PLAN.md` (T1 is this text's). Every rule ends with
  "Check:", the record field or file a reviewer compares with it.

## E. The controllers

**E1. R5.** [plan S2-3] `control.Tracker` in R2 mode with `terms=()`, `ankle_cap=()`, at the commanded clock 1.05, on a
frozen set's references; every 20 ms the policy's clipped, low-passed action a sets joint offsets `a[:nj] x scale` (0.35
rad on leg joints, 0.15 rad on every other joint) added to the reference's servo angles, and the clock factor `1 + 0.25
a[nj]`; rates and feedforward stay the reference's. The offsets are not clipped to the joint ranges (W5's `range` term
prices them). A zero action with the head neutral sends R0L's targets exactly. Check: `run.json`
`settings.env_configs[].action_leg_rad`, `action_other_rad`, `action_rate`; gate Z1.

**E2. R6.** [plan B.2; lead, 15 Nov, before any Track B run: the expected outcome may be stated in advance, as drafted]
Joint targets `clip(home + a x span, range)` with Stage 1's span rule, no feedforward, its own estimator; R5's inputs,
commands, reward, starts, randomisation, settings and budget; one seed (42) per body with the current head. At a
mid-task start its action history is set to the actions that give the reference's targets. **Expected outcome, stated in
advance:** Stage 1's policy-only controller learned on every body only to stand still, and under the first draft's
reward a body standing at home earned about a third of the positive weight (scratch); this text's `progress` term, which
that draft did not have, takes up to 0.10 more per step from it. R6 standing or shuffling is not news, and its G8b
reading (M4) can pass by standing. [settled 15 Nov, before any Track B run] On the stop item a row that stands or creeps
succeeds in no reading: U4's term `on_command_at_stop`. Check: `run.json` `mode`, `settings`.

**E3. Inputs.** [plan; NEW widths; lead, 15 Nov, before any Track B run; differs: plan B.1's input list, by one input,
the heading error] The actor's blocks, in this order, with their first and last-plus-one index for 14 and for 16 joints:

| Block | 14 joints | 16 joints | Content |
|---|---|---|---|
| encoder angles, rates | 0:14, 14:28 | 0:16, 16:32 | the estimator's, plant joint order |
| gravity direction, gyro | 28:31, 31:34 | 32:35, 35:38 | |
| foot loads | 34:36 | 38:40 | over the nominal weight |
| gait phase | 36:38 | 40:42 | sine and cosine |
| reference support | 38:40 | 42:44 | |
| command | 40:42 | 44:46 | E4 |
| heading error | 42:43 | 46:47 | the estimator's yaw about home less the commanded heading (E4), over 0.5 rad, clipped to +-2 **[differs: plan B.1's input list]** |
| the last three actions | 43:88 | 47:98 | clipped, unfiltered, newest first |
| head command (`-X`) | - | 98:102 | E5 |

Actor 88 and 102 inputs. The critic also sees: root velocity (3), height (1), roll and pitch (2), DCM error (2), joint
errors (nj), heading error and friction (2), floor (2), push (3), plant factors (5: mass, stiffness, delay, joint
damping, joint friction), foot errors (6), the reference's contact 24 ms earlier (2): 130 and 146 in all. Actions: 15
and 17. The heading error was accepted by the lead as drafted: it is the estimator's own yaw against the commanded
heading, in the ruling's words "a quantity the robot has"; it is labelled wherever the inputs are listed as one input
beyond plan B.1's list. Those words mean a quantity of the same kind as a robot's own, and no more: in simulation it is
the gyro's yaw integrated from the episode's start, with no magnetometer and an assumed residual bias within 2e-4 rad/s
times the noise scale, which is under 0.6 degree of drift in a 30 s episode; how a real gyro's heading holds over a
session is not modelled and not shown. The input was proposed on the scratch drift of the zero action (scratch). Check:
`run.json` `layout`; `checkpoint_<n>.json` `layout`.

**E4. The command and the commanded heading.** [NEW; differs: S1, whose input was the reference's own speed] The input
is `(v / 0.5 m/s, w / 1.0 rad/s)` times 1.05, piecewise constant on the reference clock: the gait's speed and yaw rate
in a periodic gait; the next gait's from the first frame of a start or change; zero from the first frame of a stop; in a
finite turn of angle A at rate w0 the walk's speed and `w0` for `A / |w0|` seconds of reference time from the turn's
first frame, then zero. The commanded heading is the integral of w over reference time. Nothing before a segment's first
frame depends on that segment. Check: `tests/test_stage2_r5lib.py`; the env description's `command_rule`.

**E5. The head command on `-X`, and the bar's head script.** [plan; NEW; lead, 15 Nov, before any Track B run: head
script H1's numbers accepted as drafted] `[yaw now, jaw now, yaw goal, jaw goal] / 0.6 rad`; the commanded yaw and jaw
angles are added to the reference's neck-yaw and jaw targets before the policy's offsets. Head neutral is all zero. The
bar's "cost of using the head" (plan section 5) is run under the row `R5G/<seed>` with head script H1: for episode seed
s, from the generator `SeedSequence([s, 77])`, a train of gaze shifts for the whole episode (yaw target uniform on +-0.5
rad, minimum-jerk over 0.8-1.2 s, held 2-6 s) and one jaw event at a time uniform in the episode's middle half (open to
0.3-0.6 rad over 0.4-0.6 s, held 1.0-1.5 s, closed over the same time); both always present. Head-first turns (F11) stay
deferred by the lead's ruling of 14 Nov on the references' design and are not run [differs: plan section 5, labelled].
Check: the env description's `head_rule`; the rows' `controller.head`; Z1 on the `-X` bodies with the head neutral.

**E6. Where the code lives.** [lead, 15 Nov, before any Track B run; differs: plan B.4, which has the notebook run
`designlab/stage1/train.py`, extended] Confirmed by the lead: the build of Track B's code (the code behind sections E to
X of this text) edits no existing file. No file under `designlab/stage1/` and no module of the bar (`bar.py`,
`readings.py`, `setio.py`, `s2io.py`) is edited. The trainer is the new module `designlab/stage2/train2.py`, not an
extension of `train.py` (a labelled change to plan B.4's wording). There is no hook in `bar.py`: the look reading is the
new module `lookread.py` with the row wrapper of `lookbar.py`. The env, the fit and the rows are new modules in
`designlab/stage2/` too. S2-17 as settled allows B.4's extension; this rule does not use it. Check: over the Track B
build's own commits (the branches `wt/wpc-*`, merges of other branches set aside), `git log --no-merges
--diff-filter=MDR --name-only` lists no file that those commits did not add themselves; the look work package's files
are section L's, and the one existing file it may edit is `designlab/stage2/sets.py`; `git diff --stat <state A>.. --
designlab/stage1 designlab/stage2/bar.py designlab/stage2/readings.py designlab/stage2/setio.py
designlab/stage2/s2io.py` is empty; `stage2/regress/regress.json` is current.

## T. Tasks and starts

**T1. The task table.** [plan B.1; NEW weights; lead, 15 Nov, before any Track B run: the weights accepted as drafted;
settled 15 Nov, before any Track B run: the stride counts under L12's cut 3] Each task is a plan over the run's set; `L`
is `start`, one window of `nominal`, `nominal_to_v0.4`. Weights are relative.

| Task | Plan | Weight |
|---|---|---|
| `stand` | a 12 s hold of the home stance (10 s played) | 1 |
| `walk_nominal` | `start`, `nominal` to the end | 2 |
| `walk_v0.4` | L, `v0.4` to the end | 3 |
| `walk_v0.3`, `walk_v0.35`, `walk_v0.45`, `walk_v0.5` | `start`, one window of `nominal`, the change to the rung, the rung to the end | 0.5 each |
| `turn_q90_left`, `turn_q90_right` | L, 2 (right: 2.5) strides of `v0.4`, `turn90_pi4_<dir>`, `v0.4` to the end | 1.5 each |
| `turn_<0.35, 0.5, 1>_<left, right>` | the same with `turn90_<rate>_<dir>` | 0.25 each |
| `stop` | L, 2.5 strides of `v0.4`, `stop2_v0.4`, a hold | 2 |
| `start_stop` | the `stop` task, a 2 s hold, `start`, 2 windows of `nominal` | 1 |
| `inplace_left`, `inplace_right` | `start_inplace_0.5_<dir>`, 3 strides of `inplace_0.5_<dir>`, `stop_inplace_0.5_<dir>` | 0.5 each |

Episodes last at most 30 s. "To the end" means whole windows of the gait's file until the plan holds 30 s at the
commanded clock, and at least one [NEW precision]. The weights were set by the architect on 15 Nov 2026, before any run
and on no result (a judgement, not a measurement), and the lead accepted them. Nudges are no task: pushes are drawn on
every task (J3). [settled 15 Nov, before any Track B run; differs: this table, under L12's cut 3] On a body whose look
sets are rebuilt under L12's cut 3, the "2 (2.5) strides of `v0.4`" of this table read "one window of `v0.4` (and half a
stride for a right-foot command)", and the env description and every record say so (L12). Check: `settings.tasks` (the
resolved table: names, plans with their counts of windows, weights) equals this table less the absent tasks, with that
reading where `set.json` `rules` names cut 3.

**T2. Absent references; what is not trained.** [NEW; plan B.1 read; lead, 15 Nov, before any Track B run;
settled 15 Nov, before any Track B run: a reference that fails part (a) of G9-ref; differs: plan B.1's list of
references, as read, and plan section 5's "No loss"] A task whose reference the set does not hold (`available: false`,
or not built) is dropped, the remaining weights renormalised, and the env description lists it under `absent` with the
set's reason. A freeze of the set made after a run's first session changes no task table of that run (K4).
[settled 15 Nov, before any Track B run] A reference that fails part (a) of G9-ref is treated by Track B as not
available, and its tasks are absent in the same way (Z2).

*Walking turns* [lead, 15 Nov, before any Track B run; differs: plan B.1's "walking turns at 0.35-1.0 rad/s", as read]:
accepted by the lead as drafted. "Walking turns at 0.35-1.0 rad/s" are the finite 90 degree turns; no periodic
walking-turn gait is built or trained. Also not trained, labelled: Stage 1's `carpet_walk` task (two of the eight envs
run every task on robot_floor_carpet instead, J5); stops from walks other than 0.40 m/s (the bar and the ladders hold
none).

*A set without the turn in place* [lead, 15 Nov, before any Track B run]: it may be trained on; the plan's B.1 list is
then not met for that body and the report says so (L13).

*No loss for R5 and R6* [lead, 15 Nov, before any Track B run; differs: plan section 5 "No loss"]: accepted by the lead
as drafted. Stage 1's twelve gated hard-floor conditions are defined on Stage 1's stock references, and a policy is
judged on the references it imitates, so they cannot be run on the view `S1`. They are run as look-set counterparts:
stand; `walk/nominal`, `walk/v0.35`, `walk/v0.5`; the finite 0.35 rad/s turn left and right in place of the periodic
walking turn; the turn in place left and right; start-stop; the standing pushes; at 0 ms, Stage 1's committed counts
printed beside them. Each departure is labelled in the records and in the report. The walks and the finite turns are
read by the bar's own items (`STAGE2_75_PLAN.md` C4, C5). The four kinds that the bar does not define on a set (stand,
turn in place, start-stop, standing push) have no episode or reading rule in this text: Track C fixes them in its own
rules before any of them runs, and until then they read "not run".

Check: the env description's `absent`; `set.json` `not_available`; the label of each no-loss counterpart in its record
and in the report.

**T3. Starts.** [plan] Half of the episodes of every task but `stand` start mid-task: the reference time is uniform over
the task's non-hold segments between 0.2 s after the first non-hold frame and 2.0 s x 1.05 before the plan's end,
snapped to a file frame (never inside a hold); the plant is set to that frame (rates x 1.05) with Stage 1's start noise
(0.01 rad on the joints, 0.02 m/s on the root's velocity), the estimator is reset from that state, and the delay queue
holds the k commands the zero action would have given before it. The others start from the settled stance with Stage 1's
start noise. (scratch) Check: `settings.env_configs[].mid_start == 0.5`; `tests/test_stage2_r5env.py` (the mid-start
tests).

## W. Reward

**W1. Form.** [plan] `r = sum of the positive terms x their weights - sum of the penalties x their weights`, each term
in [0, 1], read on the true state at the end of every env step (c the reference clock then, c0 at its start). A zero
action on a kinematic playback scores every positive term 1 and every penalty 0. Check: the env description's
`reward_rule`; `tests/test_stage2_r5core.py`.

**W2. Terms, weights, scales and their source.** [S1 for Stage 1's terms; NEW for the others; lead, 15 Nov, before any
Track B run, for the last three rows; settled 15 Nov, before any Track B run, for the `interference` row; differs: plan
B.1's reward list]

| Term | Weight | Formula | Source |
|---|---|---|---|
| joint imitation (+) | 0.55 | `exp(-mean_j ((q_j - q_ref_j(c)) / 0.1 rad)^2)` over all hinges | Stage 1 |
| speed and yaw-rate tracking (+) | 0.35 | Stage 1's Gaussian tracking reward of the CoM's speed along the previous heading and the trunk's yaw rate over the step, against the reference's own at c0 times 1.05 | Stage 1 |
| foot imitation (+) | 0.15 | `exp(-mean_f (d_f / (0.10 H))^2)`, d_f the distance between r_f and r_ref_f(c), r a foot's position less the root's in the root's heading frame | new (plan) |
| contact imitation (+) | 0.10 | the share, over both feet and the step's 10 physics steps, of steps in which the foot's contact (load above 2 % of the weight) equals the reference's support at the clock or 24 ms x 1.05 earlier | new (plan) |
| head attitude (+) | 0.10 | `exp(-((roll - roll_ref)^2 + (pitch - pitch_ref)^2) / (3 deg)^2)` of the head in the world | new (plan) |
| duty, slip, impact, tilt | 0.02 each | Stage 1's (W3 for the duty band) | Stage 1 |
| flight | 0.05 | Stage 1's | Stage 1 |
| clearance | 0.02 | W3 | Stage 1, read as the bar reads |
| action rate | 0.03 | `min(1, mean_i (a_i - a_prev_i)^2 / 0.25)` on the clipped actions | new (plan) |
| touchdown speed | 0.02 | at a foot's touchdown: `min(1, max(0, v_down - 0.15 m/s) / 0.30 m/s)` | new (the plan's "impact") |
| servo | 0.05 | per neck, tail, neck-yaw and jaw drive: a bias-corrected exponential mean of the squared force, time constant 1.0 s; `min(1, max_d max(0, RMS_d / sustained_d - 1) / 0.5)` | new (plan) |
| interference | 0.10 | W4 | new, beyond the plan's list (W4); accepted by the lead [settled 15 Nov, before any Track B run] |
| progress | 0.10 | W5 | new, beyond the plan's list; accepted by the lead |
| heading | 0.05 | W5 | new, beyond the plan's list; accepted by the lead |
| range | 0.05 | W5 | new, beyond the plan's list; accepted by the lead |

The new weights were set on 15 Nov on the recipe of `stage2/research/research_look.md` 3.3, before any run, so that no
new term outweighs Stage 1's joint imitation. They are the same for every body, head, seed and mode and are not changed
by M4's fix. Check: `settings.env_configs[].reward` and the env description's `reward_scales` in every `run.json` equal
this table.

**W3. Two of Stage 1's terms read differently.** [differs: S1] `clearance` is the larger of (a) for a foot in the air
between 25 % and 75 % of the planned swing time since its **own** lift-off and lower than 0.04 H: the shortfall over
0.04 H; (b) 1 if in any physics step of the env step a foot is loaded while the reference has that foot between 25 % and
75 % of its swing both at the clock and 24 ms x 1.05 earlier (a dragged foot or a stub step). `duty`'s band is 0.55-0.72
while the commanded target speed is below 0.30 m/s (owner decision 2), else 0.55-0.65. Check: the env description's
`reward_rule`.

**W4. Leg-to-leg interference.** [NEW; `LESSONS_LEARNED.md` 2.8; settled 15 Nov, before any Track B run: the penalty and
the termination accepted; differs: plan B.1's reward list, by one penalty and one termination] Read at the end of every
env step with `legs.LegGap` (every left-leg against right-leg pair of the screened geometry, the reading of the bar's
turn item). Penalty: `clip((3 mm - g) / 3 mm, 0, 1)` with g the smallest distance over the **moving** pairs (pairs that
stay at one distance in every pose, such as the 60 cm bodies' hip brackets at 4.0 mm, carry no penalty and are listed in
the env description). Termination: the smallest distance over **all** pairs below zero at two consecutive env-step ends.
A one-pose contact is therefore penalised and not terminated, while the bar's turn item fails on one pose: the training
signal is weaker than the bar here, by choice. A leg against the trunk or the tail is not read. The draft gave this
term's source as the plan; plan B.1's reward list holds no such term, so the lead's ruling of 02:45 UTC on the terms
beyond that list (W5) covered three terms and not this one. (scratch) [settled 15 Nov, before any Track B run] The lead
accepted the penalty, with the weight W2 gives it, and accepted the termination; `LESSONS_LEARNED.md` section 2.8 is the
reason. It is a fourth term beyond plan B.1's list and is tagged so. Check: `settings.env_configs[].interference ==
"terminate"`; the env description's `fixed_pairs`; `info["interference"]`.

**W5. Three terms beyond plan B.1's list.** [lead, 15 Nov, before any Track B run; differs: plan B.1's reward list, by
three terms] (scratch) The lead accepted each of the three separately, with the weight W2 gives it, each labelled as
beyond plan B.1's list. `progress`: only inside a periodic walking segment, from one stride after its first frame, with
a positive command: `min(1, max(0, 1 - v_bar / (1.05 v_cmd)) / 0.10)`, v_bar the bias-corrected exponential mean (time
constant: the stride period over 1.05) of the CoM's speed along the commanded heading (the bar's walk item fails 4.8 %
short of the commanded distance; Stage 1's tracking term charges 0.2 % of the reward for 5 %). `heading`: `clip((|yaw -
yaw_home - commanded heading(c)| - 3 deg) / 17 deg, 0, 1)` on the true yaw, zero in a hold. `range`: `min(1, max_j
max(0, required_j - margin_j) / 0.02 rad)`, margin the joint's distance to its nearer limit; required, per joint: in a
task without a finite turn `min(0.10 rad, the reference's own smallest margin)` (0.10 rad is Stage 1's walk rule,
`tune.WALK_RANGE_RAD`); in a task with one, decision 14's `max(min(0.10 rad, the reference's own smallest margin - 0.02
rad), 0.001 rad)`, as the bar computes it. The reference itself therefore scores zero. Check:
`settings.env_configs[].reward`.

## J. Randomisation

**J1. Delay.** [S2-9] k uniform on the integers 0 to 12 per episode. Check: `env_configs[].latency_steps == [0, 12]`.

**J2. The plant.** [plan] Per episode: servo stiffness x 0.6-1.2; sliding friction 0.4-1.2; trunk mass and inertia x
0.9-1.1 (`env.MASS_RULE`); joint damping x 0.5-1.5 and joint friction-loss x 0.5-1.5, one factor each for all hinges
[NEW precision]; IMU noise scale 0.5-1.5 [S1]. The controller's model and feedforward stay nominal: every tracker is
built on the nominal model before any episode. Check: `env_configs[]`;
`tests/test_stage2_r5env.py::test_factors_are_applied_and_restored`, `::test_trackers_are_built_nominal`.

**J3. Pushes.** [plan; S1 for the pulse] With probability 0.5 in an episode longer than 4 s: one 0.1 s pulse at the
trunk's CoM at a time uniform in `[2 s, end - 2 s]`, direction uniform, `dv = u x` the body's Stage 0 forward bound (its
own envelope screen; no default value), u uniform on 0-0.5 if the reference walks at that time and on 0-1.0 if it
stands. Check: `env_configs[].push_walking`, `push_standing`; the env description's `push_bound_m_s`.

**J4. Gaze and jaw events on `-X`.** [plan; NEW numbers, from `research_look.md` 5.1] With probability 0.5 a train of
gaze shifts for the whole episode: yaw target uniform on +-0.5 rad, minimum-jerk over 0.8-1.2 s, held 2-6 s. With
probability 0.25 one jaw event: open to 0.3-0.6 rad over 0.4-0.6 s, held 1.0-1.5 s, closed over the same time. Check:
`env_configs[].gaze_prob`, `jaw_prob`; the env description's `head_rule`.

**J5. Floors and seeds.** [plan; S1] Env i of 8 runs on hard floor for `i % 4` in 0-2 and on robot_floor_carpet for `i %
4 == 3`; no pile. The task, start, factors, push and head events come from the env's own generator; the start's and the
sensors' noise from the episode seed (`env.EPISODE_SEED_RULE`); env i resets at boundary b of PPO seed s with seed
`100,000 s + 100 b + i` (K1). Check: `sessions[].floors`, `boundary_seeds`.

## O. PPO and the probes

**O1. Settings.** [S1 design 2; NEW for 8 envs; lead, 15 Nov, before any Track B run: `n_steps` 1024 with 8 envs
accepted] 8 envs in every run and session; `n_steps` 1024 (a rollout is 8,192 steps, Stage 1's size; 160 minibatch steps
per update; the first draft of these rules had 2048, changed on the reviewers' advice and their scratch runs of PPO on
the stub (scratch)); batch 256; 5 epochs; gamma 0.99; lambda 0.95; clip 0.2; `target_kl` 0.03; networks (256, 256) with
the privileged critic; observation and reward normalisation clipped at 10 and 50; one torch thread; CPU. The number of
worker processes is the runtime's and changes no result. Every update's minibatch steps run, of the 160 possible, are
recorded (lesson 16). Check: `settings.ppo`; `updates.jsonl`; `tests/test_stage2_vecpool.py`.

**O2. Exploration.** [plan; lead, 15 Nov, before any Track B run: the plan's sigma is kept;
settled 15 Nov, before any Track B run: the probe's floor, push and head events] Initial sigma 0.1 of each action's
range (`log_std_init = ln 0.1`), as the plan has it; the lead kept it, and the owner is told that early training
episodes are short (the brief). Known in advance (scratch): the open-loop base tolerates about 0.01-0.02, so early
training episodes on both sizes end in a fall after about 3 s; mid-task starts (T3) are what shows the policy the whole
task. The sigma probe is reported and decides nothing: on each current-head body whose V6 set is frozen, the
stance-start `walk_nominal` and `walk_v0.4` tasks at k = 5 on the nominal plant, on hard floor, with no push and no head
event [settled 15 Nov, before any Track B run: "on hard floor, with no push and no head event"], 10 episodes each
(episode seeds 7000-7009), actions `clip(sigma z)` with the same z for every sigma, sigma in 0, 0.01, 0.025, 0.05, 0.1,
0.2: survivals and mean survival time per sigma. Check: `settings.log_std_init`; `stage2/tb/validation/sigma_probe.json`
(its episodes on hard floor, with no push and no head event).

**O3. The learning rate, and updates cut by the KL limit.** [plan; decision 24's probe; NEW second sigma; lead, 15 Nov,
before any Track B run: the probe must pass at sigma 0.1 and at 0.03, and one rate is used for every body and head;
settled 15 Nov, before any Track B run: the second of those two rulings is changed, to one rate per body] For each body
and head whose V6 set is frozen, for each of 1e-4, 3e-5, 1e-5 from the largest down: a fresh R5 model, seed 42,
training's randomisation, the rate held constant, 6 updates, once at sigma 0.1 and once with `log_std` set to ln 0.03
(the second sigma was added on the scratch observation that fewer minibatch steps run at a lower sigma; scratch). A rate
*passes on a body* when all 160 minibatch steps ran in each of the 6 updates at both sigmas; the probe of a body stops
at the first rate that passes.

[settled 15 Nov, before any Track B run; this changes the lead's ruling 10 of 02:45 UTC, "one rate for every body and
head", before any run] The group is the body. The probe is run once per body and head, with O1's settings, 8 envs on
J5's floors and O4's filter (O4 is measured first), and is not run again. Every run of a body (S2-4, M6, the Colab runs,
the parents of fine-tunes) uses the smallest passing rate over the heads of that body probed when the run starts. A
body's `-X` head is probed before any Colab job of that body is queued; if its rate is lower, later runs of the body use
the lower one, and the record of every earlier run says which heads its rate rested on. The two bodies may differ, and
the report says so. Why the earlier ruling was changed: comparisons are made within a body, and the 75 cm runs then need
not wait for the 60 cm sets.

As committed, this rule read here: "The value used is the smallest of the bodies' passing rates (one value for every
run, accepted by the lead)" and "A body or head whose set is frozen after S2-4 is probed before its first run; if that
lowers the value, the Colab runs use the lower one, S2-4 is not repeated, and the record says so"; the paragraph above
replaces both. The schedule is linear from the run's rate to 0.3 x it over the run's budget. If none passes on a body,
nothing trains and it goes to the lead. During a run: if the median share of minibatch steps run over a check interval
is under one half, STATUS.json and `run.json` carry `flags.kl_cut` and the lead is told; it stops nothing. Check:
`stage2/tb/validation/lr_probe.json` (per body and head: the rates tried at both sigmas, the rate that passed, the
filter and the floors; one probe each); `settings.lr`; in each run's `run.json`, the heads of its body whose passing
rates its rate rested on; the report's line on the two bodies' rates; `run.json` `flags`.

**O4. The action filter.** [plan: "re-chosen in validation"; NEW rule; lead, 15 Nov, before any Track B run: the 0.9
accepted; settled 15 Nov, before any Track B run: which body the rule rests on] Candidates: a first-order low-pass at 10
Hz (Stage 1's) and at 20 Hz.

[settled 15 Nov, before any Track B run] Measured once, before S2-4, on `75-M-H-10` alone (S2-4's body): O2's 20
episodes under each cut-off, with the same z. On `60-N-T-10` the same measurement is made when its V6 set is frozen,
reported, and decides nothing.

In those episodes the noise is scaled so that the **filtered** action has the same RMS under both cut-offs (sigma 0.1 at
10 Hz; sigma 0.1 x sqrt((a10 / (2 - a10)) / (a20 / (2 - a20))) = 0.081 at 20 Hz, a = 1 - exp(-2 pi f x 0.02 s)). 20 Hz
is used if its mean survival time, pooled over `75-M-H-10`'s 20 episodes, is at least 0.9 x that at 10 Hz; otherwise 10
Hz. One value for every run of the round; it is not measured again. What this shows: whether the faster filter costs the
base stability at equal action amplitude. What it cannot show: which filter learns better. As committed, the measurement
was made "on the current-head bodies whose V6 set is frozen (at least `75-M-H-10`)", which left open which bodies the
pooled mean held. (scratch) Check: `stage2/tb/validation/filter.json` (the body the rule rests on, `75-M-H-10`, with its
20 episodes per cut-off; the 60 cm body's measurement labelled as reported); `env_configs[].filter_hz`.

## M. The monitor and the stop rules

**M1. Episodes and outcomes.** [plan section 6; S2-10; NEW seeds and outcomes; lead, 15 Nov, before any Track B run;
settled 15 Nov, before any Track B run: an *error* episode, R0L's baseline, a check without a record;
settled 16 Nov 2026, before any Track B run: how the attempts at a check are counted, what a record of a check is, a
check counted without one, and attempts never closed; differs: C8, for policy rows] 18 episodes: the bar's conditions
`walk/nominal`, `walk/v0.4`, `turn/q90_left`, `stop/two_step`, `nudge/forward`, `nudge/left` of the run's set at mu 0.9,
three episodes each, **every episode on its own seed**: condition i (0-5 in that order), repetition r (0-2): seed `6001
+ 3 i + r` (6001-6018; none is a panel, fit or validation seed), at 10 ms, as the bar builds and judges them. Each
episode has one outcome: *read*; *fell*; *unread* (an episode of a policy row, `R5`, `R6` or `R5G`, that did not fall
and that the bar could not read: not succeeding in any reading, recorded with the reference time reached and its reason,
`clock_behind` when its mean clock factor is below 1, so that its command or judged window was not reached, else
`unreadable`) **[differs: C8]**; *error* (an infrastructure fault, or an unreadable episode of a row without a policy,
for which C8 stands). The lead accepted this as drafted: an unread episode of a policy row is "not succeeding" and is
never run again; C8 stands for rows without a policy.

[settled 15 Nov, before any Track B run; in place of the committed sentence "The monitor never runs an episode twice."]
The monitor never runs a policy row's episode twice: an *error* there counts as not succeeding, as zero credited
survival and as not up (M4), and the record names it. R0L's 18 episodes are a baseline: each must end *read* or *fell*;
if one ends *error* the fault is repaired and all 18 are run again before the run's first rollout (C8), the failed
attempt kept beside them, and no check is read against a baseline that holds an error. A check whose monitor leaves no
record is run again on the same checkpoint files, once; if it again leaves none, that check counts as without a gain and
as a failed G8b reading, and the lead is told.

[settled 16 Nov 2026, before any Track B run; this makes precise the "once" of the paragraph above, beside K5's resume]
An attempt at a check is logged in `run.json` `monitor[]` before the monitor starts, with the checkpoint's file hashes,
and closed with the monitor's exit code when it ends. The "once" counts closed attempts that left no record: after two
of them the check counts as without a gain and as a failed G8b reading. An attempt that was never closed (the session
died while the check ran) is an interruption: K5's resume runs that check again, and it does not count toward the two.

[settled 16 Nov 2026, before any Track B run; ruling 4 of the fourth list, on three cases that the paragraph above left
open] A record of a check is a `monitor_<n>.json` that names the checkpoint's hashes and holds an outcome for all 18
episodes. A check counted without a record counts each of its 18 policy episodes as an *error*. After two attempts at
one check that were never closed, the run halts (`HALT.json`) and goes to the lead.

What follows from those three sentences (the author's note, not the lead's wording). By the paragraph settled on 15 Nov
an *error* of a policy row's episode is not succeeding, zero credited survival and not up. A check counted without a
record therefore has N = 0 and S = 0 (M3), and at a G8b reading every condition of M4's table has fewer than 2 of 3 up.
A file that names other hashes, or holds fewer than 18 outcomes, is no record. The halt is one of M5's five endings, and
what follows it is the lead's to say.

A condition without a reference is "not run" and counts as not succeeding and as zero survival for both rows. Each
record holds the failures per term, the mean clock factor and the number of distinct start seeds. Check:
`monitor2.CONDITIONS`, `SEEDS` (committed before any run); each monitor record's `episodes[].outcome` (an *error*
named); `monitor/r0l.json` holds no *error* episode, and a failed attempt at the baseline is kept beside it; a check
that was run a second time names the same checkpoint file hashes; `run.json` `monitor[]`: every attempt at a check,
logged before its monitor started with the checkpoint's file hashes and closed with the monitor's exit code, and for a
check counted without a record its two closed attempts and its 18 episodes entered as *error*; each `monitor_<n>.json`
that counts as a record names its checkpoint's hashes and holds 18 outcomes; `HALT.json` `reason` for a run halted after
two attempts at one check that were never closed [settled 16 Nov 2026, before any Track B run: the 18 *error* episodes,
the record and the halt; ruling 4 of the fourth list].

**M2. Rows.** [NEW] The checkpoint's row `R5/<seed>@10ms` (or `R6/...`), loaded from its saved files through the loader
of K9, and once per run `R0L@10ms` on the same 18 episodes. A check runs at the first rollout start at or past each
multiple of 2,500,000 steps and on a run's final checkpoint; S2-4 checks every 500,000 steps (its earlier checks are
reported, S2-10). Check: `monitor/monitor_<n>.json` (it names its checkpoint's file hashes), `monitor/r0l.json`.

**M3. A gain; S2-10's reading.** [plan section 6; S2-10 as settled; NEW precision; lead, 15 Nov, before any Track B run;
settled 15 Nov, before any Track B run: N* and S*, the baseline's record, S2-4 run once; differs: plan section 6's "mean
survival" and its "the best earlier check"] N is the number of the 18 episodes succeeding with the look (U4; its three
stop episodes with U4's term `on_command_at_stop`). S is the mean over the 18 of the **credited survival**: the survival
time (the fall time, or the episode's length) times `min(1, reference time covered / (1.05 x survival time))`, which is
the survival time itself for a row that does not slow its clock, R0L exactly (scratch: a slowed clock can otherwise buy
survival). S reads staying up, not walking: a policy that keeps its clock and stays up without walking is credited in
full; N and the failures per term show which it was. The lead accepted the credited survival as drafted, labelled as
differing from section 6's "mean survival". A check shows a gain when `N >= N* + 2` or `S >= S* + 1.0 s`. As committed,
N* and S* were "each the best value over the run's earlier checks, taken separately [NEW precision], and for the first
check R0L's", which did not say whether R0L's values stay in that best at the later checks; the next paragraph settles
that they do.

[settled 15 Nov, before any Track B run] N* is the largest N, and S* the largest S, over R0L's 18 episodes and every
earlier check of the run, each taken separately [NEW precision]; for the first check they are R0L's.

S2-4 (2,000,000 steps asked, whole rollouts: 2,007,040; `75-M-H-10`, current head, seed 42, training's delay draw) is
read on its final checkpoint against R0L by this rule; its earlier checks are reported and decide nothing.

[settled 15 Nov, before any Track B run; in place of the committed sentence "R0L's 18 episodes are run and committed
before S2-4 starts."] R0L's 18 episodes are run by the monitor's baseline command and committed as
`stage2/tb/runs/s2-4/R0L_BASELINE.json` before S2-4 starts; S2-4's own `monitor/r0l.json` must equal it episode for
episode (outcome, survival and state digest), else the run stops before its first rollout. S2-4 is run once: its
checkpoint is never judged on the panels, and a second go or no-go run, whatever its settings, is a change after a
result and is labelled so.

A gain lets Colab start; no gain goes to the owner first, with the per-term failure table of both rows, whatever term it
rests on [lead, 15 Nov, before any Track B run]. The record also prints, beside N: the counts in readings 2 and 3 and
the count succeeding but for A1 with the look bands kept [NEW, reported]; and beside S: the plain mean survival time of
both rows, which decides nothing (S2-10's "the two mean survivals"). Check: `stage2/tb/runs/s2-4/S2_10.json` (both rows'
18 episodes, N, S, the plain mean survival of both rows, the reported counts, `failures_by_term`, `mean_clock_factor`);
[settled 15 Nov, before any Track B run] the commit that adds `R0L_BASELINE.json` is an ancestor of S2-4's
`sessions[0].commit`; `S2_10.json` `baseline_equal`.

**M3b. When the survival route is closed.** [lead, 15 Nov, before any Track B run;
settled 15 Nov, before any Track B run: the route's scope; differs: S2-10, by one more route to a gain] Ruled in by the
lead, as drafted, before S2-4 starts. If R0L's S is within 1.0 s of the mean planned length of the 18 episodes, a gain
is also: at least 2 more of the 18 succeeding in reading 3 with the look bands kept (every limit but a neck or tail
servo's sustained planning limit, and the four bands) than R0L.

[settled 15 Nov, before any Track B run; this fixes the scope of the lead's ruling 1 of 02:45 UTC] This route belongs to
S2-4's reading alone. In every other run's checks (M4, M5, M6) a gain is M3's two routes and no other; the count this
route reads is printed beside N at every check and decides nothing there.

Why: on `75-M-H-10` R0L is expected to stay up (so S cannot rise by 1 s) and the tail servo's sustained limit failed in
20 of 20 Stage 1 episodes at 0.35 m/s and above, so N may be 0 for a reason no policy changes much. "Within 1.0 s" is
read by that reason [NEW precision, the author's, after the review]: the route is closed when `L - S(R0L) < 1.0 s`,
where L is the mean over the 18 episodes of the planned length in simulated seconds and a condition that is "not run"
counts 0 in L as it does in S (M1); L is then the largest S a row can reach, and at a difference of exactly 1.0 s the
survival route of M3 is still open. Check: `S2_10.json` `survival_route_closed`, `rule`, and L and R0L's S beside them;
in every other run's `run.json` `monitor[]`, the count `N_but_A1_look` printed at each check and in no `gain`.

**M4. G8b and its one fix.** [plan section 6; NEW table; settled 16 Nov 2026, before any Track B run: a check counted
without a record, and what the decision is a function of] At the check of the multiple 5,000,000 (M2: the first rollout
start at or past it, 5,005,312 steps): `walk/nominal` stays up in at least 2 of its 3 episodes. "Up" means that the
episode did not fall: its outcome is *read* or *unread* (M1; by M1 as settled an *error* of the policy row's episode
counts as not up, and a check with two closed attempts that left no record counts as a failed reading, each of its 18
episodes being an *error*, so that every condition of the table below then has fewer than 2 of 3 up
[settled 16 Nov 2026, before any Track B run: the clause from "each of its 18 episodes"; ruling 4 of the fourth list]).
A policy that stands still or slows its clock therefore passes this reading (the plan's wording is "stays up"); S (M3)
does not credit a slowed clock. Otherwise, once, inside the running job: each training task that this table gives to at
least one monitor condition with fewer than 2 of 3 up has its weight doubled, once, however many such conditions name
it: `walk/nominal`: `walk_nominal`; `walk/v0.4`, `nudge/forward`, `nudge/left`: `walk_v0.4`; `turn/q90_left`:
`turn_q90_left` and `turn_q90_right`; `stop/two_step`: `stop` and `start_stop`. The reward's weights do not change. At
the check of the multiple 10,000,000 (10,002,432 steps) the same reading; a second failure stops the run. Checks are
named by their multiple, never by the step count at which the fix was applied [corrected after the review: this sentence
read "at the first check at or past the fix + 5,000,000 steps", which is 10,005,312 steps, so that the 10,000,000 check
came before it and the stop fell 2,500,000 steps later than the plan's "5M later"]. The decision is a function of the
monitor records and of M1's log of attempts alone, which a resume keeps (K5), so a killed and resumed run takes the same
one [settled 16 Nov 2026, before any Track B run: "and of M1's log of attempts" and "which a resume keeps"; the sentence
read "a function of the monitor records alone", which did not hold for a check counted without a record; it follows
rulings 3 and 4 of the fourth list]. Check: `run.json` `gates.g8b`, `fix`, `gates.g8b_after_fix`, `stopped`.

**M5. Stopping without a gain; judging; the cap.** [plan section 6; NEW precision;
settled 15 Nov, before any Track B run: how a run ends, what is judged, how the cap is counted;
settled 16 Nov 2026, before any Track B run: which runs are judged] Counting from the check at 20,000,000 steps: the run
stops at the third consecutive check without a gain (25,000,000 at the earliest). No best-model selection; a body and
seed's two heads are compared at the step both reached, each run's upper end at its own last checkpoint; the 220
runtime-hour and 8-day cap and the order of cuts are the plan's; runtime-hours are tallied per session from its start to
its last STATUS.json.

[settled 15 Nov, before any Track B run] A run ends in one of five ways and no other: its budget; M4's second failure;
M5's stop; the cap; a halt (`HALT.json` with its reason). A halted or disconnected run is resumed until one of the first
four ends it; a run that cannot be resumed is reported `unfinished: <reason>`. No run is stopped, and none is left
without a resume, on account of a monitor record otherwise. Every run of the Colab queue that passed G8a is judged on
the panels at its last kept checkpoint, whatever ended it [settled 16 Nov 2026, before any Track B run: "of the Colab
queue"; in place of the committed sentence "Every run that passed G8a is judged on the panels at its last kept
checkpoint, whatever ended it" (settled 15 Nov)], and every table names that ending. The 8 days count from the first
training session's `started_at`; when `hours.json` reaches 220 runtime-hours or the 8 days end, running jobs are stopped
at their next checkpoint and jobs not started are cut in the plan's order.

[settled 16 Nov 2026, before any Track B run] S2-4 (M3) and the local 60 cm run (M6) are not runs of the Colab queue.
Their checkpoints are never judged on the panels, as M3 and M6 say. The Colab queue is the jobs of a launch's
`QUEUE.json`; not the smoke's jobs, N5's tiny jobs or K3's two runs [this last sentence: ruling 11 of the fourth list].

Check: `run.json` `monitor[]`, `stopped`, `status`; `stage2/tb/hours.json` (the first training session's `started_at`);
`HALT.json` `reason`; the report's cells `unfinished: <reason>`; in every table, each run's ending.

**M6. A local 60 cm run before 60 cm Colab hours.** [NEW; `LESSONS_LEARNED.md` 5.2 item 7; lead, 15 Nov, before any
Track B run: accepted; settled 15 Nov, before any Track B run: the remedy's thresholds; differs: plan section 6's
estimate for this CPU, which does not hold its cost] R5 on `60-N-T-10`, current head, seed 41 (no judged seed),
5,000,000 steps asked (5,005,312), here, with the rules above. It is read at its 5M check by M4's first sentence.
Reported at each of its checks, to separate the causes of a failure: the same 18 episodes at 0 ms for the policy and for
R0L; sigma; the share of minibatch steps run; S against R0L. It starts and stops no Colab run by itself; if the reading
fails, the owner is told before any 60 cm job is queued, with the remedy the record points to: the nominal walk up at 0
ms but not at 10 ms: a delay curriculum; up at neither with sigma near 0.1 or most minibatch steps cut: exploration (a
lower starting sigma or rate); up at neither with a small sigma: a feedback base, or launching under M4.

[settled 15 Nov, before any Track B run] Here "up" is at least 2 of the 3 `walk/nominal` episodes of the 5M check not
falling, at that delay; "sigma near 0.1" is the last update's `std_mean` before that check at or above 0.05; "most
minibatch steps cut" is `flags.kl_cut` set for the last check interval (O3); "a small sigma" is every other case. The
mapping names what the owner is told; it starts and changes nothing by itself.

Its checkpoint is never judged on the panels. As the lead ruled: it is to be run before any 60 cm Colab job, not before
the day-one smoke, and its cost (about 5.2 core-hours, an estimate) comes on top of the plan's estimate. (scratch)
Check: `stage2/tb/runs/local-60/jobs/R5-60-N-T-10-s41/run.json` (`flags.kl_cut`; its 5M check's records at 10 ms and at
0 ms); that job's `updates.jsonl` `std_mean`.

## U. The look reading (reading 1)

**U1. The four measures.** [plan section 5; `research_look.md` 4.2; NEW definitions] From 50 Hz poses of an episode:
*bob*, the mean over the robot's own steps (touchdown to next touchdown of either foot, wholly inside the window) of the
range of the hip-midpoint's height; *head*, the RMS of the head's world pitch about its mean; *tail phase*, the phase of
the least-squares sinusoid of the tail-yaw angle on the robot's own phase (zero at each of its left lift-offs);
*toe-off*, the mean over its lift-offs of the lifting foot's pitch at that instant. Contact is the 2 % load rule with 20
ms debounce. Check: each companion episode's `look.own`.

**U2. Window.** [NEW] A walk item: its judged last 10 s. A turn item: the 3 s after `t_c + at_s`. Nudge and stop items:
not read. Check: `look.window`.

**U3. Bands.** [plan section 5; settled 15 Nov, before any Track B run: the threshold stays, and the records print its
side] Against the reference's own values (U5): bob within +-30 %; head pitch at most 1.5 degrees RMS (not relative);
tail phase within +-0.1 cycle; toe-off at least 70 %. The tail term is not read when the reference's tail amplitude in
the window is under 0.02 rad, the toe-off term when the reference's toe-off is under 0.02 rad [NEW]; both then pass,
labelled. On `V6noF5`, whose walks have no toe-off (L11), the toe-off term is therefore never read, and every table that
sets R2L's reading 1 beside R5's says so. Known when this was fixed (gate L1's record, kinematic; its flags give the
toe-off pitch reached by lift-off): the look ladder's V6 walks reach 0.056 rad on `75-M-H-10` and 0.031 rad on
`60-N-T-10` at 0.40 m/s, and 0.020 rad on `60-N-T-10` at 0.4455 m/s. At 0.40 m/s the 70 % band therefore asks for 0.039
rad and 0.022 rad, and on the 60 cm body's rungs above 0.40 m/s the reference's toe-off lies at the 0.02 rad threshold,
so whether the term is read there may turn on rounding. [settled 15 Nov, before any Track B run] The threshold stays
0.02 rad, and every record prints the reference's value and the side of 0.02 rad it lies on. Check: `look.terms`,
`look.not_read`, `look.reference` (the reference's toe-off and the side of 0.02 rad it lies on).

**U4. The readings on a look set.** [plan; settled 15 Nov, before any Track B run: a turn frozen with `a1_over`, reading
1's words, one more term of the stop item; settled 16 Nov 2026, before any Track B run: where the companion's reading is
the verdict, where that term decides, and where the bar's reading is printed beside the companion's; differs: C7, by
that term, for Track B rows] Reading 1 ("shown with the look"), the headline for walks and turns: the bar's `succeeded`
and all four terms. For nudge and stop items reading 1 is reading 2, labelled [NEW precision: the plan names walks and
turns]. Readings 2 and 3 as `STAGE2_75_PLAN.md` C2. Printed beside: reading 3 with the bands kept; it is never a verdict
but in the one case settled below, a turn frozen with `a1_over` (as committed it was "never a verdict"). Reading 1
counts on the sets V6 and V6noF5; on any other set the numbers are recorded and its count is null. It is held in a
companion record beside each bar record (U6); the bar's own record keeps `counts_look: null`. A turn frozen with
`a1_over` (`STAGE2_75_PLAN.md` A11 and C2, which S2-16 extends to the look sets): reading 2 gives it no verdict and only
reading 3 can pass, so reading 1, which holds reading 2, gives it no pass either, and its cell reads "not read:
`a1_over`".

[settled 15 Nov, before any Track B run] On a turn frozen with `a1_over` reading 1 reads "not read: a1_over"; its
headline on a look set is then reading 3 with the four bands kept, printed "shown but for A1, with the look" (or
"provisional but for A1, with the look"), a verdict for that case only. Reading 1's two words are "shown with the look"
and "provisional with the look".

[settled 15 Nov, before any Track B run] A stop item of a Track B row has one more term, held in the companion record:
`on_command_at_stop`, the CoM's mean forward speed over the stride before `t_c` at least 0.9 x the commanded speed (C6's
lower band). An episode that fails it succeeds in no reading [NEW; differs: C7, which has no such term; a row that walks
to the command, as R0L and R2L do, meets it].

[settled 16 Nov 2026, before any Track B run; this fixes where the term of the paragraph above decides] On the look sets
`V6` and `V6noF5`, reading 1 lives in the companion record and is the verdict of a Track B row. The bar's own record is
printed beside it as "the bar's reading (C7)". Wherever the two differ on a stop item, the table prints both and names
the term `on_command_at_stop`. On any other set, `F25F9cl` included, reading 1 does not count, as this rule's first
paragraph says. The row `R2L-F25F9cl/42` is therefore judged there by the bar's own readings, exactly as `R2/42` and
`R2/43` beside it. The term `on_command_at_stop` is recorded in its companion and decides nothing on that set (Q6).

[settled 16 Nov 2026, before any Track B run; ruling 8 of the fourth list, which makes the paragraph above precise] The
bar's own record is printed beside the companion's reading on stop items only: for walks and turns the bar's own record
holds no reading 1 (`counts_look: null`). On any other set the companion's counts equal the bar's.

Why this was settled (the author's account, not the lead's wording): `bar.py` is not edited (E6), so a stop item's bar
record holds no such term and can differ from its companion; and on `F25F9cl` the row `R2L-F25F9cl/42` would have
carried the term while `R2/42` and `R2/43` beside it carried none. The sentence "An episode that fails it succeeds in no
reading" of the paragraph settled on 15 Nov therefore holds for the companion's readings on `V6` and `V6noF5`.

Check: the companion's `counts_look`, `counts`, `counts_but_A1`, `reading_1`; in the companion, each stop episode's
`on_command_at_stop` (the speed and the term); the report's cells of a turn frozen with `a1_over` (reading 1 "not read:
`a1_over`"; its headline from the episodes' `succeeded_but_A1_look`, with its word).
[settled 16 Nov 2026, before any Track B run] In the report's tables of a look set, each stop item of a Track B row
shows the companion's count and, beside it, the bar's own count headed "the bar's reading (C7)"; where the two differ,
both, with the term's name; on every other item the companion's count stands alone. On any other set, the row's counts
are taken from the bar's own record, and the companion's counts there equal the bar's [the fourth list: the stop items
and the equality; this Check read "beside each count of a Track B row from its companion, the bar's own count"].

**U5. The reference's values; too few steps.** [NEW; lead, 15 Nov, before any Track B run; differs: C8, in this rule's
second sentence] The same functions on a kinematic playback of the episode's own track over the window's reference times
at the nominal commanded clock. When that playback has at least 4 steps and 4 lift-offs in the window and the robot has
fewer, the bob, tail and toe-off terms **fail** (labelled `too_few_steps`); it is not an error and the episode is not
run again (accepted by the lead as drafted). When the playback itself has fewer, it is an error of the definition, which
gate Z2 (b) checks for every item; such an episode is an *error* (M1). Check: `look.reference`, `look.too_few_steps`.

**U6. The companion record.** [NEW; lead, 15 Nov, before any Track B run; differs: C8, in how an unread episode is
counted] For every bar record of a Track B row (`R0L`, `R2L-<set>/<seed>`, `R5/<seed>`, `R6/<seed>`, `R5G/<seed>`): one
record that names the bar record by path, sha256 and key hash and holds, per episode, the outcome (M1's four), the look
numbers, both sides' values, the terms, the clock factor, and the three readings' counts by the bar's own arithmetic,
with an `unread` episode counted as survived and not succeeded in each **[differs: C8]** (accepted by the lead as
drafted); for a policy row also both version sets, `cross_version`, the largest fingerprint difference and the tolerance
(K9). The wrapper that makes it changes nothing an episode executes. Check: `tests/test_stage2_lookbar.py` (`R0L` equals
`R0` digest for digest and record for record); `<run>/<set>/<body>/<block>/look/<row>/<condition>.json`.

## Z. Gates

**Z1. G8a.** [plan B.3; NEW alignment and pass rule] For a set and a body or head: the R5 env with the zero action (head
neutral, nominal plant, the condition's friction, stance start with the row's start seed and Stage 1's start noise, the
condition's push, the delay through the bar's own wrapper, interference termination off) against the bar's row `R0@0ms`
and `R0@10ms`, on the eight gated mu 0.9 conditions the set has references for, `walk/v0.4_mu0.54` and `walk/nominal`,
panel seeds 3042 and 3043 (40 episodes). The row runs `round(seconds / 2 ms)` physics steps; the env runs `ceil` of a
tenth of that many env steps, so it covers them all. Compared: sha256 of `qpos`, `qvel`, `ctrl`, `sensordata` after
every physics step the row ran. Same end: if the row fell at physics step j, the env ends in the env step holding j; if
the row did not fall, the env does not end in any env step lying wholly inside the row's steps (the env's last, partial
step is not read). **Pass: every episode identical.** Any other outcome: no run of that body and head, and it goes to
the lead. No module in a bar key may change for R5 (S2-17, E6), and this rule fixes the conditions, the seeds and what
is compared: the gate is therefore run again only after a change to Track B's own code, the failed record is kept beside
the new one, and any other change is a change after a result and is labelled so. The trainer repeats one condition
(`walk/v0.4`, seed 3042, both delays) before every new run, here and on Colab; that is the notebook's line 3. Check:
`stage2/tb/gates/<set>/<body>/g8a.json` (`identical == of`; `compared_steps == reference_steps` in every episode);
`run.json` `gates.g8a`.

**Z2. G9.** [plan B.3; S2-11 as settled; NEW order and limits; settled 15 Nov, before any Track B run: the toe-off limit
of (b), G9-ref on `V6noF5`, what follows a failure] Two records per body and head.

*G9-ref*, kinematic only; it must pass before the first closed-loop episode on that set and body: (a) every frozen
reference played kinematically at the commanded clock passes its own item's readings, the quick turn also reaching 80
degrees at the slowest clock (the bar's reference check, cited); (b) on each rung the look ladder holds for the body,
the look measures of the walk's playback agree with the ladder's own numbers (bob within 5 %, head pitch within 0.1
degree, tail phase within 0.01 cycle, toe-off within 0.02 rad and within 10 % of the ladder's number
[settled 15 Nov, before any Track B run: "and within 10 % of the ladder's number"]) and the set's look-format hash for
that rung equals `stage2/look/L1_FREEZE.json`'s; on a rung the ladder does not hold the four numbers are finite with at
least 4 steps in the window; on an `-X` body each walk's playback agrees with the plain body's playback of the same rung
within the same four limits; and for every walk and turn item the set has, the playback of its window (U2) holds at
least 4 steps and 4 lift-offs (what U5 relies on); (c) four degraded playbacks each fail exactly their term (bob x 0.5;
a 3 degree neck sinusoid; the tail 0.2 cycle late; flat feet), the unchanged playback passes all four with ratio 1, and
a playback cut to 3 steps reads `too_few_steps`; (f) each reference no bar item plays (the turn in place with its start
and stop; the changes to rungs other than 0.40 m/s) has its screen record cited, and every task's track is continuous at
its joins (joint targets within 1e-6 rad) and plays 2 s with finite targets. As committed, (b)'s toe-off limit was 0.02
rad alone, which is a third to two thirds of the ladder's own toe-off at 0.40 m/s (U3).

*G9-ref on `V6noF5`.* As committed, parts (b) and (c) could not pass on `V6noF5`, whose walks have no toe-off (L11): its
look-format hash and its toe-off cannot equal the ladder's V6 rungs', and a flat-feet playback fails no term there,
since the toe-off term is not read (U3). The draft and the design ran G9-ref on set V6 only and did not meet the case.

[settled 15 Nov, before any Track B run] On `V6noF5` part (b) makes no hash and no toe-off comparison with the ladder
and records the toe-off as "not read"; bob, head and tail are compared with the same body's V6 set's own playback; part
(c)'s flat-feet playback applies to V6 only. R2L's set must pass G9-ref in this form before its first episode (X1).

The ruling names no limits for that comparison. The author's reading, which is not the lead's wording: it is part (b)'s
comparison with another comparand, so each `V6noF5` walk's playback is set against the same body's V6 playback of the
same rung within (b)'s limits for bob, head pitch and tail phase; parts (a) and (f), and the rest of (b) and (c), read
on `V6noF5` as on V6.

*G9-run*, after R0L's records of that set and body: (d) every episode of the body's `R0L@10ms` records holds every
reading of its item and, where its window was reached, a finite look reading, and every 10 ms bar record holds both
toe-clearance readings; (e) for the 60 cm body and head the re-run Stage 1b 10 ms turn episodes of `60-N-T-10` are
trace-identical to their committed records (the bar's re-read record, cited).

*What follows a failure.* [settled 15 Nov, before any Track B run; in place of the committed sentence "If (b) fails,
U1's definitions go back to the lead before any look-set episode."] If (a) fails for a reference, Track B treats that
reference as not available (its tasks absent, T2; its conditions "not run", H3) and it goes to the lead; a failing
`start`, `nominal` or `v0.4` blocks the body and head. If (b), (c) or (f) fails, no closed-loop episode runs on that set
and body and it goes to the lead. If (d) or (e) fails, no training run of that body and head starts. A definition, limit
or band changed on any such account is labelled "changed after G9's result" (G9-ref's result is kinematic).

Check: `stage2/tb/gates/<set>/<body>/g9_ref.json`, `g9_run.json`; `g9_ref.json`'s time before the first look-set episode
in `stage2/tb/dev_episodes.json` and in any record; in the two records, each part's outcome and what followed a failure
(a reference treated as not available, a blocked body and head); in `g9_ref.json` of a `V6noF5` set, the form it was
read in (no ladder hash; the toe-off under `not_read`; the V6 playbacks it was compared with, by path and sha256; no
flat-feet playback); the label "changed after G9's result" wherever a definition, limit or band was changed.

## Q. R2L

**Q1. What is fitted.** [plan B.2; S2-18; owner's allowance of the candidate (P30); lead, 15 Nov, before any Track B
run: the candidate's fit is in the budget (Q5); settled 15 Nov, before any Track B run: when a fit may run] Stage 1's
tracker (R2, 13 searched gains, capture margin held) on the bar's tasks, seed 42, here: on set `V6noF5` for `75-M-H-10`
and `60-N-T-10` (current heads), and, labelled a candidate, on set `F25F9cl` for `75-M-H-10`, in Q5's order.
`speedfit.py` is not edited. The candidate's fit on `F25F9cl` waits for task 3's first unit of the 75 cm round (X1's
first sentence). [settled 15 Nov, before any Track B run] By X1 as settled, a fit may run on `V6noF5` once G9-ref has
passed there in the form Z2 gives for that set, and the order of R2L and S2-4 is free (a labelled change to plan section
7's order; as committed, the fits started after S2-4's start). Check: `stage2/tb/r2l/<set>/<body>/r2lfit/42/cma.json`
`settings`; for a fit on `V6noF5`, that set and body's `g9_ref.json` dated before the fit's `sessions[0].started_at`.

**Q2. Mix and rotation.** [S2-18] Every evaluation runs five bar episodes at mu 0.9: `walk/v0.4`, a quick turn, the
two-touchdown stop, a nudge at 0.25 x the bound, `walk/nominal` (slots j = 0-4). In generation g, with block `B = g //
6`: the episode seed is `tune.training_order(42)[g]`; slot j runs at `k = (g + j) mod 6`; the turn is to the left for
even B and to the right for odd B; the nudge's direction is forward, left, backward, right for `B mod 4` = 0, 1, 2, 3.
So within a block every slot meets all six delays in one direction. Left and right turns meet every delay, 10 ms
included, in any two consecutive blocks (12 generations), the four nudge directions in any four (24 generations). A fit
that Q5 ends earlier does not give every (kind, direction) every delay: its record and the report name the directions
and delays it did not meet [corrected after the review: this sentence claimed every delay for every direction whatever
the fit's length]. Start and noise seeds are derived per slot (Stage 1b's rule: five different starts per evaluation;
unlike the bar's one seed slot, on purpose). Every member of a generation runs the same five episodes. The table for 30
generations is written before generation 0. Check: `rotation.json` (its `written_at` before `sessions[0].started_at`;
per (kind, direction) the delays met); `evals.jsonl`'s delays and directions equal it.

**Q3. Objective and search.** [S1b; NEW terms for the bar's items] `tune.score` (falls first, then tracking and limit
terms). Tracking terms: `tune.measure`'s over the item's judged window. Limit terms: `tune.measure`'s window limits
without the three "at nominal" ones, with toe clearance on the own-swing reading, plus the item's own, each graded on
[0, 1]: walk: the shortfall of the distance along the heading over 4.0 m, the line deviation over 0.25 m; turn: the
heading error at the judged time and the largest in the 3 s after, each over 20 degrees; stop: touchdowns beyond two
over 2, the stop time over 1.5 strides; nudge: the recovery time over 3 s. CMA-ES, population 12, sigma0 0.1, from the
body's Stage 1 G5 selection; at most 30 generations. Check: `cma.json` `objective`, `start`.

**Q4. Selection.** [S2-18; one partial departure; settled 15 Nov, before any Track B run: the order of the reported
cells] Candidates: the start and the mean after each third of the generations judged. **Decisive, run first and always
whole:** each candidate runs the five kinds on validation seeds 2000-2019 at k = 5 (turn to the left for even seeds, to
the right for odd; nudge direction forward, left, backward, right for seed mod 4 = 0-3), ranked by `tune.selection_key`
on those 100 episodes per candidate alone (fewer falls, then the lower mean S, mean P and mean J, then the earlier
candidate). Reported and not used, as far as Q5 reaches: at each of k = 0, 2, 4, 1, 3 in that order, the selected
candidate and then the start [settled 15 Nov, before any Track B run: "at each of k = 0, 2, 4, 1, 3 in that order, the
selected candidate and then the start"], on validation seeds 2000-2004 with the same directions (25 episodes per cell);
a cell not reached reads "not run: budget" **[differs: S2-18's "the others reported", partly]**. Check: `r2lg5.json`
`selection`, `by_delay` (the cells in the order run).

**Q5. Budget.** [lead, 15 Nov, before any Track B run; settled 15 Nov, before any Track B run: how a fit's share is
spent; settled 16 Nov 2026, before any Track B run: when a reported cell starts and what a fit's share then holds, and
the caps of R2L's panel records and of the re-runs beside them, with where each cap's number comes from and how a cap is
spent; differs: plan section 6's 8 core-hours for R2L; NEW split] As the lead ruled: R2L's budget is 12 core-hours,
counted as the sum of evaluation CPU seconds in `ledger.json`, for three fits in this order: `V6noF5` on `75-M-H-10`,
then `V6noF5` on `60-N-T-10` (8 core-hours for the two), then the candidate `F25F9cl` on `75-M-H-10` (4 core-hours).

[settled 15 Nov, before any Track B run] The three fits have separate shares of 4.0 core-hours and run in that order; an
overrun comes off no other fit's share. Within a share: block 0 (six generations) always runs; after each whole block
the next one starts only if the fit's summed seconds plus the seconds of its costliest block so far are at most 2.4
core-hours, or 30 generations are done; a started block finishes, and any excess over 2.4 is recorded and comes off the
0.4 of the reported cells. The decisive cell then finishes whatever it costs (1.2 reserved; its overrun is recorded and
is the only one that may pass the share without limit [settled 16 Nov 2026, before any Track B run: "without limit"]). A
fit is "cut by the pre-registered order" only if it has not started when the round's report is written; nothing else
cuts one.

[settled 16 Nov 2026, before any Track B run; in place of the committed sentence "The reported cells run in Q4's order
while the share lasts; a started cell finishes." (settled 15 Nov), which stood in the paragraph above] The reported
cells run in Q4's order. A reported cell starts only when the fit's share still holds its estimate: the decisive cell's
mean seconds per episode, times 25, times 1.25. What the share still holds is 4.0 core-hours less every second the fit
has spent: its blocks, its decisive cell and its earlier reported cells [settled 16 Nov 2026, before any Track B run:
this sentence, which says what "the share" is here; ruling 9 of the fourth list]. Otherwise it reads "not run: budget".
A started cell finishes. If it passes the share all the same, the excess is recorded (`overrun_reported`) and no further
cell starts.

The paragraph settled on 15 Nov stands in place of these committed sentences, the first of which was the lead's ruling
of 02:45 UTC (ruling 11 of the first list) and is made precise by it: "A fit that does not fit the budget is cut and
reported so ("cut by the pre-registered order"). Per set and body 4.0 core-hours: the fit at most 2.4 (it ends at the
last whole block of six generations inside that, or at 30); 1.2 reserved for the decisive cell, which finishes even if
it needs more (the overrun is recorded; it is the only one allowed); 0.4 for the reported cells."

[settled 16 Nov 2026, before any Track B run; NEW: as settled on 15 Nov, R2L's panel records and the re-runs beside them
had no budget] Beside the fits' shares, the panel records of each R2L row, and the re-runs of `R2/42` and `R2/43` beside
the candidate's row (Q6), each get a cap: 1.5 times the bar runner's dry-run total at the committed rates
(`stage2/runs/2026-11-15-75cm/COST.md`), per set and body, counted in the bar's own `budget.json`. The dry run takes the
16 gated conditions and `walk/nominal` at 10 ms, and the same at 0, 4 and 6 ms where the plan asks for them. When a cap
is reached the 6, 4 and 0 ms records are cut first, in that order; the 10 ms records are never cut.

[settled 16 Nov 2026, before any Track B run; rulings 1 and 2 of the fourth list, which say where each cap's number
comes from and how a cap is spent] Each cap is 1.5 times the total that the dry run prints on the frozen set itself, for
the 60 cm body as for the 75 cm body, at the same `COST.md`. For a `V6noF5` row the dry run is
`--set V6noF5 --body <body> --rows R2/42@10ms --conditions @gated,walk/nominal` (the three other delays added where S2-9
asks). It is made and committed with its printed total after the set's freeze and before the row's first panel record.
No cap's number is fixed now: the numbers this rule holds are planning figures and cap nothing. Each cap is spent by the
bar's unit rule (`STAGE2_75_PLAN.md` H6): a unit starts only if what the cap has spent plus the unit's estimate fits,
and a started unit finishes. The 10 ms records are one unit and start first; then one unit per delay, in the order 0, 4,
6 ms. The first unit that does not fit is cut with every later one. If the 10 ms unit passes the cap, the excess is
recorded and no further unit starts. Fall attribution of these records is not in the dry-run total: it runs last and is
cut first. Each cap has its own run directory and its own `budget.json`, under one task key. A cap is one and a half
times a total that holds its 10 ms unit's estimate, so that unit always fits and starts: "never cut" holds.

How these two rulings are read where they name no case (the author's reading, not the lead's wording). On `V6noF5`,
"where S2-9 asks" is a body on which R5 failed G8b on both seeds (Q6), which is known only after the row's 10 ms
records. The first dry run therefore takes the 17 conditions at 10 ms. If S2-9 then asks, the dry run is made again with
all four delays and committed before the first record at another delay; 1.5 times its total is the row's cap from then
on, and what the 10 ms unit spent counts in it. On `F25F9cl` the dry runs are the two pairs of commands listed below,
made again and committed before `R2L-F25F9cl/42`'s first panel record and before the first re-run record. They take the
22 records that Q6 names for the candidate's row (the 16 gated conditions at 10 ms, which are its 10 ms unit, then
`walk/v0.4` and `walk/v0.4_mu0.54` at 0, at 4 and at 6 ms; Q6 names no `walk/nominal` record there, and none is
counted), and the same 22 for each of `R2/42` and `R2/43` where they are run again. The dry run of an R2L row names the
row `R2/42`: the runner gives a row without a measured rate the largest measured one, which is R2's (`bar.rate_of`). The
rulings name no path for a cap's run directory or for its committed dry run, and no task key.

Planning figures. They cap nothing; each is an estimate and no record: a dry run plans each episode, multiplies its
planned seconds by a rate and runs none. The rate is `COST.md`'s for an R2 row, 0.1574 wall seconds per simulated
second, measured by this study on 15 Nov 2026 on Stage 1's own condition `walk/brisk` of `75-M-H-10`. The last column is
the printed total times 1.5 (arithmetic), not rounded. The dry runs were made on 16 Nov 2026 from
`hardware/trex_form_study` at commit `82ba1ad` of the working branch, when the third list was written in.

| For | Records | Dry run made on | Episodes | Simulated hours | Printed total, core-hours | 1.5 x, core-hours |
|---|---|---|---|---|---|---|
| `R2L-V6noF5/42` on `75-M-H-10`, 10 ms | 17 | the look work package's unfrozen build of `V6noF5` (branch `wt/wp-l` at `e2696b2`, read through `--set-root`), row `R2/42` | 340 | 3.602 | 0.567 | 0.8505 |
| `R2L-F25F9cl/42` on `75-M-H-10` | 22 | the frozen set `F25F9cl`, row `R2/42` | 440 | 2.999 | 0.472 | 0.708 |
| `R2/42` and `R2/43` beside it, where run again | 44 | the frozen set `F25F9cl`, the two rows themselves | 880 | 5.998 | 0.944 | 1.416 |

Each was made as

```
python -m designlab.stage2.bar dry-run --body 75-M-H-10 --cost stage2/runs/2026-11-15-75cm/COST.md \
    --stage1-run stage1/runs/2026-11-08 --out <an empty scratch directory> <the arguments below>
```

with these arguments; the totals are the runner's own printed lines:

- `--set V6noF5 --rows R2/42@10ms --conditions @gated,walk/nominal`, on the unfrozen build: 340 episodes, 3.602
  simulated hours, 0.567 core-hours;
- `--set F25F9cl --rows R2/42@10ms --conditions @gated`: 320 episodes, 2.186 simulated hours, 0.344 core-hours;
  `--set F25F9cl --rows R2/42@0ms,R2/42@4ms,R2/42@6ms --conditions @walk`: 120 episodes, 0.813 simulated hours,
  0.128 core-hours; the table's row is their sum;
- `--set F25F9cl --rows R2/42@10ms,R2/43@10ms --conditions @gated`: 640 episodes, 4.372 simulated hours, 0.688
  core-hours; `--set F25F9cl --rows R2/42@0ms,R2/43@0ms,R2/42@4ms,R2/43@4ms,R2/42@6ms,R2/43@6ms --conditions
  @walk`: 240 episodes, 1.626 simulated hours, 0.256 core-hours; the table's row is their sum.

No dry run was made on a `V6noF5` set of `60-N-T-10`, so the table holds no figure for that body.

What went with ruling 1 of the fourth list (the author's account). As written in under ruling 6 of the third list, this
rule held caps for the `V6noF5` rows of both bodies that were taken from dry runs on a stand-in, the frozen set `V0` of
the 75 cm body under row `R2/42`: 0.444 core-hours at 10 ms (1.5 x 0.296) and 1.7775 with 0, 4 and 6 ms (1.5 x 1.185),
with the same numbers standing in for the 60 cm body. They are gone and cap nothing. The table's first row shows why: on
the unfrozen `V6noF5` build the same 17 conditions are planned at 0.567 core-hours, 1.9 times the stand-in's 0.296 and
above its cap of 0.444, so the bar's ledger would have refused the 10 ms unit itself (H6: a unit whose estimate does not
fit is cut). Its episodes are planned at 24.87 to 52.89 s each, against 14.06 to 21.08 s on `V0`; the likely reason,
which was not checked, is that a look walk's file is a window of 9 to 14 strides and a plan counts whole windows
("Windows"). Rulings 1 and 2 were given after these planning figures existed; no episode had run for them.

Check: `ledger.json` (per fit: the seconds of each block, of the decisive cell and of each reported cell, in the order
run); `r2lg5.json` `judged_at_generation`, `overrun`; the report's cell for a fit that was cut.
[settled 16 Nov 2026, before any Track B run] `r2lg5.json` `overrun_reported`, and for each reported cell its estimate,
what the share still held when it was decided and whether it started; for each cap, its committed dry run (the command
and the printed total, on the frozen set, in a commit that is an ancestor of the `sessions[].commit` of the row's first
panel record) and its own run directory's `budget.json` (one task key; the cap, which is 1.5 times that total; the units
in the order started, which is 10, 0, 4, 6 ms and then attribution; the cut units, which are the tail of that order,
with no 10 ms unit among them; the excess of a 10 ms unit that passed its cap) [the fourth list: this Check read "the
bar's `budget.json` (the cap, the units in the order started, and the cut units: 6 ms, then 4 ms, then 0 ms, and no 10
ms record among them)"].

**Q6. Rows and reading.** [plan; settled 15 Nov, before any Track B run: the row on the candidate set;
settled 16 Nov 2026, before any Track B run: how that row is judged, and a pointer to Q5's caps] Rows are named by their
fit: `R2L-V6noF5/42@<ms>ms`, `R2L-F25F9cl/42@<ms>ms`, resolved through the committed index `stage2/tb/r2l/index.json`
(set, body, seed, the gains file's path and sha256); a row refuses a condition of another set. Read at 10 ms as a
baseline beside R0L and R5 (and at 0, 4 and 6 ms where the plan asks); one seed, so every verdict of R2L is provisional
(in reading 1, U4's word "provisional with the look"). If R5 fails G8b on both seeds of a body, that body is reported
with R2L at 0, 4, 6 and 10 ms (S2-9) and the owner is told before more Colab hours. The row on the candidate set
`F25F9cl` falls under `STAGE2_75_PLAN.md` A3's label, and by its H4 a candidate's verdict is never printed without its
replication status; H4 and H6 name no R2L row, and no R5 runs on that set to stand beside.

[settled 15 Nov, before any Track B run] `R2L-F25F9cl/42`'s bar records are reported with A3's label and no verdict
word; the row is not replicated. It runs the 75 cm plan's `walk/v0.4` and `walk/v0.4_mu0.54` at 10, 0, 4 and 6 ms and
the other fourteen gated conditions at 10 ms, on seeds 3042-3061, beside `R2/42` and `R2/43`, which are run again from
the same commit where a key module differs (S2-17).

[settled 16 Nov 2026, before any Track B run] On the candidate set `F25F9cl`, where reading 1 does not count (U4), the
row `R2L-F25F9cl/42` is judged by the bar's own readings, exactly as `R2/42` and `R2/43` beside it; the term
`on_command_at_stop` is recorded in its companion and decides nothing on that set. The panel records of an R2L row, and
those re-runs of `R2/42` and `R2/43`, have the caps of Q5.

Check: the bar records' `controller.gains_info` (path and sha256 of `r2lg5.json`); `index.json`; the report's cells of
`R2L-F25F9cl/42` (the counts, which are the bar's own record's, and A3's label; no verdict word and no replication
word); no `seeds5042` record under that row; its 8 + 14 records' conditions and delays; where `R2/42` and `R2/43` were
run again, both directories and the `sessions[].commit` they share with the R2L row's records.
[settled 16 Nov 2026, before any Track B run] In that row's companion records, each stop episode's `on_command_at_stop`,
which enters no count of the report; Q5's caps, each in its own run directory's `budget.json`.

## K. Checkpoints, resume, versions

**K1. Boundaries.** [NEW; differs: S1, where each session drew new seeds] At step 0 and at the first rollout start at or
past each multiple of 250,000 steps a checkpoint is saved; then, in every run whether interrupted or not, everything
random is reseeded with `100,000 x seed + 100 x b` (b = that multiple / 250,000), every env is reset, and the running
returns of the reward normalisation are zeroed. A resume loads the newest intact checkpoint without letting the library
reset the envs, continues the step count and the schedule, and does the same boundary before its first rollout. Check:
`sessions[].boundary_seeds`; `tests/test_stage2_train2.py`.

**K2. Kept checkpoints.** [plan B.4; NEW precision for Drive] Locally: each check's (M2) and the newest two. To Drive
the launcher copies the checkpoints at multiples of 1,000,000 steps and each check's, each as a whole (three files, then
the record), and keeps each check's and the newest two of those; an older one is deleted on Drive only after a newer one
has stayed copied for one further copy cycle. A copy read back through the mount is "copied", not verified. For each
check's checkpoint one tar of its three binaries is also written. Check: `run.json` `kept`; STATUS.json
`last_checkpoint_copied`; the files on Drive.

**K3. Kill and resume.** [plan B.3; NEW pass rule and settings] On this machine, with one set of library versions, on
the real env: `75-M-H-10`, set V6, R5, seed 42, 8 envs, `n_steps` 1024, a checkpoint every 32,768 steps, 131,072 steps
in all; one run uninterrupted with 2 workers; one run with 2 workers killed with SIGKILL (the whole process group) when
its progress reads 81,920 steps and resumed with 3 workers. Pass: every policy tensor, the optimizer's state, the
normalisation statistics (their count included) and the fingerprint actions of the two final checkpoints are **equal**
(tolerance zero). If they are not, the build stops and goes to the lead; no tolerance is set after a difference has been
seen without a label. Across machines or versions nothing is asserted. Check: `stage2/tb/validation/kill_resume.json`.

**K4. What a resume refuses.** [S2-13; S1] Changed settings or env configs: always. Changed inputs (the sha256 or the
entry hash of a used file; the plant's inputs): always; a later freeze of the set is accepted when every used file's and
entry's hash and the resolved task table are unchanged, and both freeze indices are recorded. Changed sources (a fixed
list of files, hashed per session, children included): unless `--allow-source-change`. Changed library versions (Python,
numpy, MuJoCo, scipy, torch without its build label, stable-baselines3, gymnasium, cloudpickle): unless
`--allow-version-change`. A checkpoint that fails its integrity check is skipped with its reason recorded; a resume
needs one whose run id (`run_uuid`) and settings hash are the run's and whose fingerprint is reproduced within 1e-5. A
flagged session records both sets and the flag, and every later record of the run carries it. In the notebook the two
flags are the parameters `ALLOW_SOURCE_CHANGE` and `ALLOW_VERSION_CHANGE`, set only on the lead's word. Check:
`sessions[]` (`version_change`, `source_change`, `skipped_checkpoints`, `fingerprint_max_abs_diff`, `freeze_index`).

**K5. The state a resume starts from.** [NEW; settled 16 Nov 2026, before any Track B run: a check's attempts, beside
M1] Every checkpoint's record holds the run state at its step: the task weights in force, the fix, the monitor history,
the no-gain count, the checks done, any stop. A resume rebuilds the run's live record from it, moves later entries to
`discarded_after_resume`, but keeps the attempts (M1) logged for a check on the checkpoint it resumes from
[settled 16 Nov 2026, before any Track B run: "but keeps the attempts (M1) logged for a check on the checkpoint it
resumes from", so that M1's count of closed attempts survives a resume; ruling 3 of the fourth list], cuts
`updates.jsonl` back to the resumed step, runs any check that is due at that step and has no record naming that
checkpoint, and then applies M3-M5 to the records exactly as the uninterrupted run does. A job with a run record and no
checkpoint restarts from step 0 in place and is the same run (`restarted_from_zero`).
[settled 16 Nov 2026, before any Track B run] An attempt at a check that was never closed (M1: the session died while
the check ran) is an interruption: the resume runs that check again, and the attempt does not count toward M1's two.
[from here to "Check:", the author's reading of M1 and of rulings 3 and 4 of the fourth list, not the lead's wording] A
record is what M1 says it is. In two cases a resume does not run a due check that has no record. A check that M1 has
counted, after two closed attempts that left none, stays counted and is not run again. And after two attempts at one
check that were never closed, the check is not run a third time: the run halts and goes to the lead (M1). Check:
`checkpoint_<n>.json` `run_state`; `sessions[].discarded_after_resume`; `tests/test_stage2_train2.py` (a kill during a
check; after a fix); after a resume, `run.json` `monitor[]` still holds every attempt logged for a check on the
checkpoint resumed from [settled 16 Nov 2026, before any Track B run: this last item].

**K6. One commit, one path, one queue.** [S2-13; NEW precision] A run keeps one RUN_ID, one bundle, one queue, one
budget and one commit for its whole life: REPO_REF is the 40-hex SHA of the launch commit (the notebook refuses any
other form for a launch), a resume takes it from `QUEUE.json`, and another commit is accepted only with
`ALLOW_SOURCE_CHANGE`. The notebook refuses a RUN_ID that exists without RESUME, and RESUME with another bundle, queue
or budget. Check: `QUEUE.json`; `SESSION_<k>.json`.

**K7. One writer.** [NEW; lead, 15 Nov, before any Track B run: accepted as drafted, the 10 minutes included] The
launcher writes `LIVE.json` (its session's id, host, time) every 60 s and re-reads it before every copy or delete on
Drive; if the id is not its own it halts its job ("superseded"). A session with RESUME is refused while `LIVE.json` is
younger than 10 minutes (an assumption about how late Drive shows one runtime's files to another; not measured), unless
`TAKEOVER` is set on the lead's word. Check: `LIVE.json`; `SESSION_<k>.json` `takeover`; `HALT.json` `reason`.

**K8. Fine-tunes.** [plan S2-6; settled 15 Nov, before any Track B run: the learning rate] A new job directory from a
parent's checkpoint and statistics, the parent's `log_std`, 5,000,000 steps, its own boundaries and checks,
[settled 15 Nov, before any Track B run; in place of the committed clause on the rate] a constant learning rate r. Let p
be the parent's schedule value at the parent's step. If all 160 minibatch steps of each of 6 updates run at p from the
parent's checkpoint, r = p. Otherwise r is the largest of 1e-4, 3e-5, 1e-5, 3e-6 that is below p and at which they all
run. If none does, the fine-tune does not run and it goes to the lead. The probe is run once and its record is kept
(`run.json` `parent.lr_probe`).

As committed, the clause read "a constant learning rate: the parent's schedule value at the parent's step, lowered to
the largest of 1e-4, 3e-5, 1e-5 not above it at which all minibatch steps of 6 updates run from the parent's checkpoint
[NEW: the parent's sigma is below 0.1]", which left the rate undefined when the parent's schedule value is under 1e-5 (a
parent started at 3e-5 ends at 9e-6; one started at 1e-5 is below it from its first step) and when no listed rate runs
every step. The probe starts from the parent's checkpoint because the parent's sigma is below 0.1 [NEW]. Check:
`run.json` `parent`, `parent.lr_probe`, `settings.lr`.

**K9. The fingerprint and the loader.** [S2-12; S2-14] Every checkpoint is saved with 256 observations from the
generator seeded `[20261115, observation width, actor width]`, the deterministic float32 actions its policy and
statistics give them, and the statistics as plain arrays. Every load, anywhere, goes through one loader that checks the
files' hashes, passes the schedule objects explicitly and requires the actions reproduced within 1e-5; a load for
judging also passes the spaces and reads the statistics from the plain arrays, so that only tensors and the policy class
cross library versions. A checkpoint that fails is not used and, if it was to be judged, the owner is told. Schedules
are instances of module-level classes; an archive holds no bytecode. A judged row is refused if the code that builds its
observations and applies its actions differs from the run's recorded hashes, unless a flag records `source_change`.
Check: `checkpoint_<n>.json`; the rows' `controller` (both version sets, `cross_version`); the companion records'
`policy` (`fingerprint_max_abs_diff`); `tests/test_stage2_s2load.py`.

## Y. The day-one smoke and the budget

On the chosen runtime type, one session, the notebook's `SMOKE` mode, on `75-M-H-10` with the current head. The steps
sheet gives the expected minutes of each item.

| Rule | Item | Recorded | Pass |
|---|---|---|---|
| Y1 | setup | the versions against the pins (N2), torch's build label, Python's version, ffmpeg | the pins installed; a Python minor other than 3.13 is reported (K9 covers it) |
| Y2 | the machine | vCPUs, RAM, CPU model | reported |
| Y3 | memory | resident memory of the trainer and of each worker with 8 envs after one rollout | the total under 80 % of RAM |
| Y4 | the rate at 8 envs | env steps per wall second of an R5 job from its first step to the end of its third update (three rollouts and three updates) [settled 15 Nov, before any Track B run: the span; as committed, "over its first three rollouts, updates included"] | Y11 gives at least 15,000,000 steps |
| Y5 | a test clip | an MP4 of 3 s from a monitor trace, rendered in a subprocess with OSMesa | reported (a clip is no stop condition) |
| Y6 | kill and resume on Drive | a tiny job killed after its first checkpoint is on Drive; its local directory deleted; resumed in a new process | the resume finds an intact checkpoint on Drive, reproduces its fingerprint, refuses nothing and reaches its budget |
| Y7 | setup time | seconds from the setup cell to the first training step | reported |
| Y8 | the inputs rebuilt once (last; capped at 40 minutes) | one body's models and as many of its V6 arrays as the cap allows, each sha256 against the manifest | reported; "not finished" is an outcome; if every file compared matches and all were compared, the rebuild is an allowed fallback (N4) |
| Y9 | an archive that loads here | the smoke's checkpoint tar, downloaded by the lead | the loader reproduces its fingerprint within 1e-5 in the study's venv |

**Y10.** [NEW; lead, 15 Nov, before any Track B run: accepted as drafted; settled 15 Nov, before any Track B run: a
pointer to Y11; settled 16 Nov 2026, before any Track B run: when the smoke passes] `SMOKE.json` is written after Y7 and
again after Y8. The smoke passes when Y1, Y3, Y6 and Y9 pass and Y4 passes or has been answered by the owner's budget
(`QUEUE.json` `budget_by: owner` with `budget_set_on`, written when the queue is made and before its first job starts)
[settled 16 Nov 2026, before any Track B run: this sentence, in place of the committed sentence "The smoke passes when
Y1, Y3, Y4, Y6 and Y9 pass."; it changes ruling 12 of the first list in this rule's list of items; its parenthesis is
ruling 10 of the fourth list's, in place of "(`QUEUE.json` `budget_by: owner`, with the date)", which named no field for
the date]. Until it passes no training job is queued. [the author's note of 16 Nov 2026, not the lead's wording: where
the owner's answer is what lets the smoke pass, the queue file that carries it is made first, and none of its jobs
starts before the answer is in it.] [settled 15 Nov, before any Track B run] Y4 fails under 15,000,000 steps, and no job
is then queued until the owner has set the budget (Y11). Check: `runs/<run_id>/smoke/SMOKE.json`;
`stage2/tb/validation/colab_archive.json`; where Y4 failed, `QUEUE.json` `budget_by` (`owner`) and `budget_set_on`, a
time before the start of the queue's first job.

**Y11. The budget (S2-8).** [plan section 6, as stated, with its "about 560 env steps/s" made exact;
settled 15 Nov, before any Track B run: which smoke fixes the budget, and a rate under the floor;
settled 16 Nov 2026, before any Track B run: the floor written as an inequality]
[settled 15 Nov, before any Track B run] Fixed once for every Colab R5 and R6 run, from Y4's rate r in the first smoke
session that records Y4 on the runtime type the launches use; every `SMOKE.json` is kept, and a later smoke changes the
budget only by an amendment labelled with its reason. The budget is 40,000,000 env steps if `r x 72,000 >= 40,000,000`
(r at least 555.56 env steps/s), else `r x 72,000` rounded down to a multiple of 5,000,000. Under 15,000,000 (`r x
72,000 < 15,000,000`) Y4 fails and no job is queued until the owner has set the budget, recorded in `QUEUE.json`
`budget_by` and `budget_set_on` (Y10) [settled 16 Nov 2026, before any Track B run: "(`r x 72,000 < 15,000,000`)", the
floor as the inequality alone, in place of a parenthesis that gave a rounded rate; and, by ruling 10 of the fourth list,
"and `budget_set_on` (Y10)"].

That paragraph stands in place of this rule's first sentence as committed: "Fixed once after the smoke for every run:
40,000,000 env steps, or, if Y4's rate r cannot reach that in 20 h (r under about 556 env steps/s; exactly, `r x 72,000
< 40,000,000`, r under 555.56), `r x 72,000` rounded down to a multiple of 5,000,000; never under 15,000,000 (...)
without the owner." The parenthesis left out of this quotation gave a rounded rate; by the ruling of 16 Nov 2026 the
floor is the inequality alone. Asked steps are run in whole rollouts. Runtime-hours per run (steps over rate, plus
setup) are tabled per runtime type. Check: `QUEUE.json` `budget_steps`, `budget_by`, `budget_set_on`; every `SMOKE.json`
(`rate`, `budget`), the first that records Y4 on the launches' runtime type being the one the budget rests on.

## N. The notebook and the bundle

**N1. The four lines.** [plan section 6; NEW text; NEW: line 4's floor] Exactly:

```
S2 LINE 1/4 commit: <40 hex> (REPO_REF is a commit: yes; checkout clean: yes)
S2 LINE 2/4 inputs: bundle <bundle_id>, manifest <first 12 hex of the content hash>, <n> of <n> files match their sha256: OK
S2 LINE 3/4 G8a <body> set <set>: zero action = R0 at 0 ms and 10 ms, <n> of <n> episodes identical (<steps> physics steps): PASS
S2 LINE 4/4 first rate <job>: <r> env steps/s (8 envs, <W> workers, <c> vCPUs); <B>M steps in about <h> h
```

Line 2 ending `FAILED (...)` or line 3 ending `FAIL (...)` stops the session before any training. Line 4 ends ` BELOW
FLOOR <f>` when r is under the launch's floor, the rate at which the budget takes 22 h (`budget / 79,200`; 505 env
steps/s for 40,000,000); the owner then tells the lead. Check: the acceptance run (N5) prints them; `STATUS.json`
`lines`.

**N2. Pins.** [plan B.4; NEW values] The notebook installs the study venv's versions: mujoco 3.10.0, numpy 2.5.4, scipy
1.18.1, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0 (the CPU build, forced when another build is installed),
cloudpickle 3.1.2, pillow 13.0.0, PyOpenGL 3.1.10; ffmpeg and OSMesa from the system. There is no unpinned fallback: a
pin that does not install stops the session. Training processes run with `MUJOCO_GL=disable`, assigned by the entry
point itself. Check: `SESSION_<k>.json` `versions`, `pins_match`, `torch_build`;
`tests/test_stage2_nb.py::test_pins_equal_the_study_venv`.

**N3. The bundle.** [plan B.4] The model files and the frozen sets' arrays of the bodies and heads of a launch, in plain
tar parts of at most 20 MB, with a committed manifest of every file's and part's sha256. The manifest's content block
alone is hashed and quoted; upload receipts are kept in a separate file. The notebook verifies every part and file
before anything is built; any mismatch stops the session. Check: `stage2/tb/bundle/<bundle_id>/MANIFEST.json`
(`content_sha256`), `UPLOADS.json`; line 2.

**N4. Fallbacks if an upload fails.** [NEW] In this order: parts of at most 5 MB; the model parts alone with the arrays
rebuilt on Colab, allowed only after Y8 found every hash equal, the rebuilt arrays kept on Drive and reused; with the
owner's consent, the parts on a data branch that is never merged. Nothing is rebuilt or fetched from anywhere else.
Check: `SESSION_<k>.json` `bundle.source`.

**N5. Before the owner's first session.** [plan B.3; the plan's dry run and tiny notebook run, made one] The acceptance
run: a fresh clone of the launch commit from the remote, holding nothing git ignores, plus the bundle in a directory
standing in for Drive; the notebook's own cells: every hash verified, every env built, Z1's short form passed for each
body and head of the bundle, a tiny R5 job and a tiny R6 job; the launcher's process group killed with SIGKILL while a
copy to the stand-in Drive is in progress; the local run directory and the clone deleted; a second clone; RESUME to
DONE; N1's lines printed; a second launcher on the live run refused. Check:
`stage2/tb/bundle/<bundle_id>/ACCEPTANCE.json` (the remote's SHA, the clone's HEAD, the content hash, the lines, the
kill point, the sessions).

## X. Episodes that are no records

**X1. Development and check episodes.** [`STAGE2_75_PLAN.md` rule R4 as settled; NEW for the look sets;
settled 15 Nov, before any Track B run: the reading of rule R4, two checks on the view `S1`, episodes after S2-4's
start, R2L's fits; settled 16 Nov 2026, before any Track B run: the first sentence lifted for the frozen 75 cm sets;
differs: rule R4's wording; differs: plan section 7's order] Until task 3's first unit of the 75 cm round has run, no
R2-mode, policy or training episode runs on V0, F25F9 or F25F9cl (lifted for the frozen 75 cm sets on 16 Nov 2026, after
the panels) [settled 16 Nov 2026, before any Track B run: "(lifted for the frozen 75 cm sets on 16 Nov 2026, after the
panels)"; the lead's amendment 13 of `STAGE2_75_PLAN.md` records it there]; the build's tests and acceptances use Stage
1's own run directory (the read-only view `S1`) or a synthetic set in a temporary directory. On a look set, for each
body and head: no closed-loop episode before G9-ref passed; between G9-ref and S2-4's start only these: Z1's episodes,
R0L's baseline and panel records, Z2's G9-run, the probes of O2-O4 (zero action, noise, freshly initialised policies),
the rate measurement, K3's two runs, the acceptance run's tiny jobs, and R2L's fits (section Q; added by the lead,
below). None but Z1's, R0L's and Z2's, and the fits', is a record; each is listed in `stage2/tb/dev_episodes.json` with
its command, set, body, rows or mode, and count.

[settled 15 Nov, before any Track B run; differs: rule R4's wording] `STAGE2_75_PLAN.md` rule R4's "a Stage 2 set" means
the 75 cm round's three sets (V0, F25F9, F25F9cl); rule R4 is the 75 cm round's own by S2-16.

What that ruling settles, in the author's words: rule R4 as settled reads "the only closed-loop episodes run on a Stage
2 set are the build's checks: single episodes on set V0 under `R0` rows" until task 3's first unit; read literally that
covers the look sets too, while this rule, as the design does, limits them by G9-ref and S2-4's start alone, and that is
the reading the lead ruled. As committed, this rule said that the literal reading stood until the point was settled, so
that no closed-loop episode ran on a look set before task 3's first unit; that clause is gone. `STAGE2_75_PLAN.md`
itself is not changed by this text.

[settled 15 Nov, before any Track B run] The training start with `MUJOCO_GL` set against it and R6's start (plan B.3)
run on the view `S1`. From S2-4's start on, every closed-loop episode on a look set belongs to a run, check, gate, probe
or panel that a rule of this text names; any other is listed in `stage2/tb/dev_episodes.json` before it runs, with its
reason, is no record, and no rule changes on its account except by an amendment labelled as made after a result.

[settled 15 Nov, before any Track B run; differs: plan section 7's order "R2L; validation and S2-4", labelled] R2L's
fits (section Q) are added to the list of what may run on a look set once G9-ref has passed there, and the order of R2L
and S2-4 is free. R2L's set, `V6noF5`, must pass G9-ref in the form Z2 gives for it before its first episode.

As committed, the list held neither the two checks nor a fit, so that the fits started after S2-4's start.

Check: `stage2/tb/dev_episodes.json` (each entry's reason, and its time before the episode ran; after S2-4's start, no
closed-loop episode on a look set outside a named run, check, gate, probe or panel that is not listed);
`stage2/runs/2026-11-15-75cm/dev_episodes.json`; `stage2/tb/validation/gl_start.json` and `r6_start.json` (both on the
view `S1`); for a fit on `V6noF5`, that set and body's `g9_ref.json` dated before the fit's `sessions[0].started_at`
(Q1).

## L. The look inside moves and the look sets

The look sets are `V6`, the look the owner picked at gate L1 (S2-15), and `V6noF5` (L11, L14). By S2-16, rules A4 to A13
of `STAGE2_75_PLAN.md` apply to them as to every set: the one builder, Stage 1's construction for starts and changes,
causal stops and turns, the gates of every move, the fallbacks, the two-touchdown stop, the quick turn with its stride
rule and its appendage rule, the Stage 0 screen and what blocks a freeze, and freezing. This section adds three things:
how each look feature is carried through a move (rules L1 to L8 and L10); how a move joins a walk whose file holds a
window of several strides (join rules J1 to J3); and the choices the lead made on 15 Nov 2026, before any look set was
built (L11 to L15), each a look choice that the owner is told about. Four of these rules (L10 to L13) and the paragraph
"Windows" also hold wording that the lead settled later on 15 Nov 2026, on points 4, 5 and 6 of the second list, after
the look work package's first kinematic numbers existed ("What had been run": prescribed motion, no closed-loop
episode); each is marked and labelled so, and L10 to L13 say what they change in the choices of 02:45 UTC. Rules L1 to
L10 and the join rules are those of the design of 14 Nov (its section 3.9), which the lead let stand on 15 Nov; each is
stated here in full. That design's rule L9 (toe-off with rolling contact inside a move) is not built (L11), so this text
has no rule L9.

Words of this section. A *move* is a start, a speed change, a stop or a turn: a finite file between two periodic gaits
or the stand; gait a is the one it leaves, gait b the one it enters. p is the gait phase in strides (left lift-offs at
whole numbers, right lift-offs at half numbers); T(p) the stride period at phase p; `amp` the move's amplitude ramp (0
in a stand, 1 in a walk); "the ramp's weight" is gait b's share in [0, 1] along a move's ramp (0: gait a's law; 1: gait
b's); s is the minimum-jerk step from 0 to 1. A *stance* of a foot runs from its touchdown to its lift-off. A look
feature's fields are the fields of `designlab/gait.py`'s `GaitSpec` named with it. The look's laws live in a new module,
`designlab/stage2/moves_look.py`, which the builder `designlab/stage2/moves.py` loads by name; `moves.py` is not edited.
The two look sets are added as entries of `designlab/stage2/sets.py`, a module that is in no record key. "On the arrays"
means: on the move's array file, in `q`, `t`, `swing`, `feet` and the keys the look's laws append (`phase`, `swing_u`,
`foot_pitch`, `rolling`, `trunk_quat`).

Check, for each of L1 to L10: the entry's `move.laws` is `"look"` and its `move.rules` names every rule that built it;
`tests/test_stage2_moves_look.py` holds one check on the arrays per rule. Each rule's own Check says what a reviewer
reads on the arrays.

**L1. Duty (F3; field `swing_time`).** [design; NEW] A gait's duty is its own spec's, `describe(H)["duty"]` (with F3 it
follows from the fixed swing time). Inside a ramp the duty mixes with the ramp's weight like every other parameter, and
each swing takes the duty sampled at its own lift-off. F13's longer stride (`stride_rel`) reaches a move through the
same `describe` and needs no rule. Check: the entry's `gaits.a.duty`, `gaits.b.duty`; on the arrays (`swing`, `phase`),
each swing lasts `1 - duty` of a stride, the duty mixed at its lift-off.

**L2. Tail (F9; fields `tail_phase`, `tail_amp`).** [design; NEW] `tail = amp x tail_amp(p) x sin(2 pi p +
tail_phase(p))`, the amplitude and the phase each mixed with the ramp's weight; a stand has amplitude 0. In a turn's
quicker steps the tail's amplitude and the neck's carry the factor `(T(p) / T_walk)^2` of `STAGE2_75_PLAN.md` A10.
Check: on the arrays, the tail-yaw column of `q`: its fitted amplitude and phase at the two ends of a change equal the
two gaits'; a turn entry's `turn.appendage_rule`, with `S2A-appendage` in its `move.rules`.

**L3. Head (F10; fields `head_mode`, `head_gain`, `head_settle`, `settle_lag`).** [design; NEW] With `head_mode`
"stabilise": `neck = -head_gain x the trunk's pitch about home` (zero in these plans, which impose no trunk pitch) plus
the settle dip after each touchdown of the plan (depth `head_settle`, deepest `settle_lag` after the touchdown), scaled
by `amp` and, between a nodding and a stabilising gait, by the ramp's weight. Check: on the arrays, the neck-pitch
column of `q`: a dip follows each touchdown of the plan by `settle_lag`.

**L4. Foot lift (F4; field `lift_peak`).** [design; NEW] A swing's lift is `gait.lift_profile(u, lift_peak)`, u the
fraction of the swing, with `lift_peak` sampled at the swing's lift-off. Check: on the arrays (`feet`, `swing_u`), each
swing's highest point falls at `lift_peak` of the swing.

**L5. Bob (F1 and F2; fields `bob`, `bob_skew`, `bob_drop`).** [design; NEW] The pelvis height is `z = root0_z - amp x
(crouch + bob(p) x gait.bob_profile(tau))`: `root0_z` is the home height, tau the fraction of the interval between the
plan's consecutive touchdowns, and the trough is placed by `gait.bob_trough`. The bob's amplitude mixes with the ramp's
weight. A move runs 12 correction passes (`look.LOOK_ZMP_ITERS`), the number its gait's spec holds. Check: the entry's
`zmp_iters` (12); on the arrays, the root's height in `q`: its trough follows each touchdown as in the periodic walk at
both ends.

**L6. Trunk roll (F6; field `roll_amp`).** [design; NEW] The trunk's orientation is yaw(heading) x home x roll(`amp x
roll_amp(p) x` the walk's roll law in phase). In a turn the hip-roll joints' range margin is screened (A12). Check: on
the arrays, `trunk_quat`: the roll is toward the stance leg at its mid-stance; the `range` block of the turn in
`screen.json`.

**L7. Step times (F14; fields `strides`, `phase_jitter`, `asym`).** [design; NEW] The step-time deviations of a window
enter a move's period step by step, through the law `period_factor`, at the rate of the walk's own warp (`gait._Warp`).
Check: with L8.

**L8. Where a move carries the variation (F14; also `gaze_var`, `tail_var`).** [design; NEW] The variation's envelope is
1 on the frames a move shares with a periodic window (its warm-up stride and its join strides) and fades with s to 0
across the next stride inward. Between the fades a move has no F14 variation. Check: on the arrays, between the fades
the phase runs at the plan's own rate, `1 / T(p)`, and the head and the tail carry no slow variation; on the shared
strides the move equals the resampled window (join rules J1 and J2).

**L10. Flat feet, stance by stance.** [design; NEW; with L11, the rule of every move;
settled 15 Nov, before any Track B run, after the look work package's first kinematic numbers existed: which stances are
flat, and the instant of a start or a change; this refines the lead's look choice "flat feet inside every move";
settled 16 Nov 2026, before any Track B run: what holds before the instant of a start or a change]
[settled 15 Nov, before any Track B run; labelled as changed after the look work package's first kinematic numbers
existed] For a start or a change, the instant is the start of the move's ramp (the entry's `move.command_phase`, Stage
1's convention). Every stance that ends between that instant and the end of the move's last ramp is flat; a stance that
ends after the last ramp has ended follows the walk's own law, so that the move's last frame equals the walk's first,
foot pitch included [a clause on the frames before the instant followed here; the next paragraph replaces it]. The same
per-stance rule holds for a causal move with its command instant in place of the ramp's start. This refines the lead's
choice "flat feet inside every move" (L11): flat while the gait is changing.

[settled 16 Nov 2026, before any Track B run; labelled as changed after the look work package had built look sets
(prescribed motion; none frozen); in place of the committed clause "the frames before the instant equal the look walk's,
foot pitch included" (settled 15 Nov), which was wrong as written: a start and a change are built with foreknowledge
(`STAGE2_75_PLAN.md` A5), so the body's motion before the instant is the move's own] Before the instant, the foot pitch
of a change follows the law of the walk it leaves, toe-off and rolling entry included. The rest of the frame is the
move's own, built with foreknowledge (A5), and A7's gates read it. A start has no frame before its instant but the
stand.

How this rule read as committed, and what stands of it; this paragraph is the author's account, not the lead's wording.
It read: "In a commanded move every stance that ends after the command is flat: the toe-off amplitude sampled at its
lift-off is 0, and there is no rolling entry at its start. The swing that lifts at the command carries the walk's
toe-off pitch to zero by its touchdown. Toe-off returns with the first stance that begins after the move's last ramp. A
stop's two remaining stances are flat. The frames before and at the command equal the look walk's, foot pitch included."
The settled wording changes two things. A stance that begins before the end of the move's last ramp and ends after it
was flat, and now follows the walk's own law. And the rule, which was worded for a commanded move, now names the instant
of a start and of a change, which are not causal (`STAGE2_75_PLAN.md` A5). These stand as committed: what a flat stance
is (the toe-off amplitude sampled at its lift-off is 0, and there is no rolling entry at its start); in a causal move,
whose command is given at a lift-off (A6), the swing that lifts at the command carries the walk's toe-off pitch to zero
by its touchdown; a stop's two remaining stances are flat. The rule is stated per stance so that it is causal: a toe-off
faded "over the stride before any ramp" would begin before the command, and the look's commanded moves would then know
of the command in advance (A6). The committed sentence on the frames before and at the command stands for a causal move
alone, whose reference is the walk, frame for frame, up to the command (A6). The wording settled on 15 Nov had carried
it over to a start and a change; the paragraph settled on 16 Nov 2026 says what holds there instead.

Check: on the arrays, `foot_pitch` is 0 and `rolling` is false through every stance that ends between the instant and
the end of the move's last ramp, and a stance that ends after that ramp carries the walk's own `foot_pitch` and
`rolling`; `join_command` (the joints within 1e-3 rad at the command frame); the entry's `move.command_phase` (for a
start or a change, the instant); in `tests/test_stage2_moves_look.py`, the test that in a causal move the frames before
and at the command equal the look walk's, the test that before the instant of a change `foot_pitch` and `rolling` follow
the law of the walk it leaves [settled 16 Nov 2026, before any Track B run: these two tests, in place of one test "that
the frames before the instant (in a causal move, before and at the command) equal the look walk's"], and the test that
the move's last frame equals the walk's first, `foot_pitch` included.

**Windows.** [design; settled 15 Nov, before any Track B run, after the look work package's first kinematic numbers
existed: a nudge's onset] With F14 a look walk's file holds a window of K whole strides that is periodic as a whole (9
to 14 strides in the look ladder's V6 rungs; `stage2/look/L1_FREEZE.json` `strides_in_window`): Stage 1's keys with 160
K frames and the stride motion of the whole window, then `window_strides` (K) and the look keys. Its entry holds
`period_s` (the stride period), `window_s` and `strides` (K). A plan counts windows of such a file and takes a periodic
item's duration from `window_s`, never from strides times the period. F14 samples the window uniformly in time and warps
the phase, so inside a window no lift-off falls on a frame or on a multiple of the stride period (in V6 by up to 8.9 ms,
or 3.5 mm of the CoM's travel); only the window's boundary is exact. On such a set the stride counts of
`STAGE2_75_PLAN.md` B6 read so [NEW precision; differs: B6's wording; from the design, in which a plan item counts
windows and the bar's lead-in is "`nominal` x 1"]: the lead-in's "one stride of `nominal`" is one window of it (T1's L),
and the 2 strides of `v0.4` before a turn (2.5 before the right turn and before the stop) are join rule J1's leave
point, j = 2 strides into the window (j + 0.5 for a right-foot command); under L12's cut 3 they read one window of
`v0.4`, and half a stride for a right-foot command (L12). [settled 15 Nov, before any Track B run; labelled as changed
after the look work package's first kinematic numbers existed] A nudge's onset "2 + u strides into the walk" (B6) is
read on a K-stride window as (2 + u) x the walk's stride period `period_s` after the walk segment's first frame. Check:
each walk entry's `strides`, `period_s`, `window_s`; the file's `window_strides`; the records' `plan`; a nudge episode's
`command_ref_s` against the walk segment's first frame and `period_s`.

**Join rule J1. Leaving a window.** [design; NEW] A move leaves a K-stride walk at the warp's own time of its command
phase: `t_leave = _Warp.time(j)`, or `_Warp.time(j + 0.5)` for a right-foot command. The track plays `m + t_leave / W`
windows before the move (j is a whole number of strides into the window, m a whole number of windows, W the window's
duration). The move's warm-up stride, its initial state and its `join_command` frames are the window **resampled on the
move's phase grid** (`q`, `com_ref` and `zmp_mb` read through `_Warp.phase`; the window is twice continuously
differentiable in time). Check: the entry's `a_phase`, `a_time_s`; `join_command` within A6's limits against the
resampled window.

**Join rule J2. Entering a window.** [design; NEW] A move enters a K-stride walk at the window's boundary: its last
output stride is compared (`join_b`: CoM within 1.0 mm, multibody ZMP within 2.0 mm, A7) with the window's last stride
resampled in the same way, and the walk then starts at its frame 0. Check: the entry's `join_b`.

**Join rule J3. Shared strides.** [design; NEW] By L8, a move's own frames carry the window's variation on those shared
strides and nowhere else. Check: as L8.

**L11. F5 lives in the periodic walks only; inside every move the feet are flat.** [look choice; lead, 15 Nov, before
any look set was built; settled 15 Nov, before any Track B run, after the look work package's first kinematic numbers
existed: the choice is refined to "flat while the gait is changing", and the instant of a start or a change is named;
settled 16 Nov 2026, before any Track B run: what holds before that instant, with L10; differs: S2-15's "V6 (every look
feature)", inside moves] Toe-off with rolling contact (F5; fields `toe_off`, `toe_off_start`, `toe_off_margin`,
`toe_off_rate`, `toe_off_accel`, `toe_reach`) is in the periodic walks of V6 and nowhere else. Inside every move (start,
change, stop, turn) the feet are flat by rule L10, which is stated per stance so that commanded moves stay causal. These
are the first two cuts the design named (F5 inside turns and stops; F5 inside starts and changes too), taken from the
start: the design's rule L9 (F5 inside a move, as rolling contact per stance about the yawed foot's own lateral axis) is
not built. `V6noF5` is V6 with F5 off in the walks as well.

[settled 15 Nov, before any Track B run; labelled as changed after the look work package's first kinematic numbers
existed; this refines the lead's look choice of 02:45 UTC, which the paragraph above states] The feet are flat while the
gait is changing. Rule L10 as settled names the stances: every stance that ends between the move's instant and the end
of its last ramp is flat, and a stance that ends after the last ramp has ended follows the walk's own law (in V6, F5's
toe-off), so that the move's last frame equals the walk's first, foot pitch included. For a start or a change, which are
not causal (`STAGE2_75_PLAN.md` A5), the instant is the start of the move's ramp (the entry's `move.command_phase`,
Stage 1's convention), and before it the foot pitch of a change follows the law of the walk it leaves, while the rest of
the frame is the move's own (L10) [settled 16 Nov 2026, before any Track B run: this clause, in place of "and the frames
before it equal the look walk's, foot pitch included"]. "Nowhere else" and "inside every move" in the paragraph above
are read with this refinement.

Check: each V6 walk entry's `definition.spec` (F5's fields as the look ladder's V6 rung has them); each V6noF5 walk
entry's `definition.spec` (`toe_off` and `toe_reach` 0); in every move entry of both sets, L10 named in `move.rules`, L9
in none, and on its arrays `foot_pitch` 0 and `rolling` false in the stances L10 names; for a start or a change, the
instant read from the entry's `move.command_phase`.

**L12. Join rule J1 is built as designed; its fallback order is fixed now.** [look choice; lead, 15 Nov, before any look
set was built; settled 15 Nov, before any Track B run, after the look work package's first kinematic numbers existed:
the trigger, the scope and the order of the fallback; settled 16 Nov 2026, before any Track B run: a `start` or a
`nominal_to_v0.4` that still fails; differs: A8, A11, for a join gate failed on a K-stride window; cut 3 differs: B6 and
T1] [settled 15 Nov, before any Track B run; labelled as changed after the look work package's first kinematic numbers
existed, whose join numbers are what this trigger reads ("What had been run")] The fallback is taken only when a move of
a look set, built by the builder after an independent review of its join code (the review's record named by path and
sha256 in `set.json` `rules`), has `join_command` or `join_b` outside A7's limits. A change or a start that fails is
first retried by A8's steps under join rule J1; cut 3 is taken for a causal move (the stop, a turn), or when A8 is
exhausted. A cut is taken for a body, not for a move: every look set of that body and of its `-X` head (`V6`, `V6noF5`)
is rebuilt under it, so that a body's two heads and its R2L set never differ in it; two bodies may differ, and the
report says so. A `start` or a `nominal_to_v0.4` that still fails under the cuts open to it stops that body's set, which
is then not frozen, and goes to the lead: A8 stands on a look set. Any other move that still fails is "not available"
with its numbers: a ladder change (A8 step 3), the stop (its conditions are then "not run", H3), a turn (A11) and the
turn in place (L13) [settled 16 Nov 2026, before any Track B run; labelled as changed after the look work package had
built look sets (prescribed motion; none frozen): the two sentences before this mark, in place of the first clause of
the committed sentence "A move that still fails under the cuts open to it is "not available" with its numbers (A8 step
3; A11 for a turn); no limit is changed for it without the label "changed after a result"" (settled 15 Nov); the
sentence after this mark is that sentence's closing clause, set on its own by the author, and it holds as committed, for
every move that still fails]. No limit is changed for a move that still fails without the label "changed after a
result". Under cut 3 the "2 (2.5) strides of `v0.4`" of T1 and of B6 read "one window of `v0.4` (and half a stride for a
right-foot command)", and the env description and every record say so.

The design's cut 3 [differs: B6 and T1]: a move leaves and enters a K-stride window only at its boundary (a right-foot
command at the exact time of phase 0.5 after the boundary); episodes then walk whole windows (about 10 s) before a
command. A fallback taken is recorded per set with the failing gate's numbers.

[settled 15 Nov, before any Track B run; labelled as changed after the look work package's first kinematic numbers
existed; with L13] The design's cut 4 (no F14 inside periodic turns) is in force from the start for the turn in place,
the only periodic turn of a look set (L13); it is no further fallback.

[settled 15 Nov, before any Track B run; labelled as changed after the look work package's first kinematic numbers
existed] A set rebuilt under a cut before any closed-loop episode on that set is not "a change after a result" in A8's
and A11's sense (a join gate is a kinematic build reading); after one, it is.

What these three settled paragraphs change in the lead's look choice of 02:45 UTC (ruling 15 of the first list); this
paragraph is the author's account. As committed, the rule began: "The fallback is used only if a V6 move fails a join
gate (`join_command` or `join_b`, at A7's limits) after the builder's implementation has been reviewed. It is then taken
in this order and no other. First, the design's cut 3", and went on: "Then, the design's cut 4: no F14 inside periodic
turns (one-stride turn references; F14 stays in the straight walks)." Now: any move of a look set triggers it, not a V6
move alone; the review has a record; A8's retries come first for a start or a change; a cut is taken for a body with
both heads and its R2L set, where it was recorded per set; and cut 4 is no second fallback, being in force from the
start for the turn in place.

Check: `set.json` `rules` (the join code's review by path and sha256; the fallback in force, or none, with the failing
move's `join_command` or `join_b` numbers; the same in every look set of the body and of its `-X` head); the move
entries' `a_phase`, `a_time_s`, `join_command`, `join_b`, and `available: false` with its numbers for a move that still
fails, other than `start` and `nominal_to_v0.4`: a set in which one of those two still fails holds no freeze (A8)
[settled 16 Nov 2026, before any Track B run: this exception]; under cut 3, the reading of T1's and B6's stride counts
in the env description and in the records' `plan`; a set rebuilt under a cut after a closed-loop episode on it carries
the label "changed after a result", and one rebuilt before any does not.

**L13. The turn in place.** [look choice; lead, 15 Nov, before any look set was built;
settled 15 Nov, before any Track B run, after the look work package's first kinematic numbers existed: the plan's
length, the stride that is cut, the strides the gate compares, no F14; settled 16 Nov 2026, before any Track B run: flat
feet as this gait's own law] The 0.5 rad/s turn in place, left and right, is a periodic look gait (`inplace_0.5_<dir>`)
cut from the steady part of a long plan of the same builder and laws (the start from the stand, then 4 hold strides more
than Stage 1's start writes [settled 15 Nov, before any Track B run: this parenthesis, in place of "(enter, hold N
strides, exit)"]), with its start from the stand and its stop to the stand (`start_inplace_0.5_<dir>`,
`stop_inplace_0.5_<dir>`; every name is made by `setio`). It is gated for periodicity: consecutive hold strides, mapped
onto each other by the stride motion, agree within 1.0 mm of the CoM, 2.0 mm of the ZMP and 1e-3 rad of the joints. If
it fails that gate on a body it is recorded "not available" there; the set may still be trained on (T2), and the report
then says that plan B.1's list is not met for that body. It is the only periodic turning gait of a look set: no periodic
walking-turn gait is built (T2).

[settled 15 Nov, before any Track B run; labelled as changed after the look work package's first kinematic numbers
existed; the numbers 4 and "second" are confirmed by the lead] The reference is the second of those hold strides: one
stride, with no F14 variation (L8: between the fades a move carries none) and flat feet (the next paragraph)
[settled 16 Nov 2026, before any Track B run: "(the next paragraph)", in place of "(L10)"]; the gate compares it with
the hold stride before it and the hold stride after it. No other plan length and no other stride is tried: a gait that
fails is "not available". The design's cut 4 is in force from the start for the turn in place and is no further fallback
(L12).

[settled 16 Nov 2026, before any Track B run; labelled as changed after the look work package had built look sets
(prescribed motion; none frozen)] The turn in place has flat feet as its own law: no toe-off and no rolling entry in any
stride, the start and the stop included. Rule L10 is about starts and changes of a walk and does not govern this gait.

Why the citation went, and what was checked when this was written in (the author's account, not the lead's wording): by
L10 as settled, a stance that ends after a move's last ramp follows the gait's own law, so L10 did not say that the hold
strides are flat. The ruling's sentence on L10 names a walk's starts and changes; L10's per-stance rule also holds for a
walk's causal moves, its stop and its turns, from their command. The look work package's code (branch `wt/wp-l` at
`e2696b2`; `designlab/stage2/sets_look.py`, the builder's `inplace`) builds this gait from the set's nominal look gait
with F5's fields at their defaults (`toe_off` 0, `toe_reach` 0), for the long plan and for the stop. In its build of
`V6` on `75-M-H-10`, which is not frozen, `foot_pitch` is 0 and `rolling` false in every frame of the two gaits, of
their starts and of their stops (six files, read on 16 Nov 2026). Its rule texts still cite L10 for the flat feet.

Known when this was settled (the review's note, as the commit that adds this file holds it): the look work package's
first version (branch `wt/wp-l`, before that commit) already wrote the turn in place as one stride with no F14
variation, with 4 hold strides by default and a free choice of the stride cut.

Check: the two entries' `yaw_rate_rad_s` (0.5 rad/s, signed), `speed_m_s` (0) and `strides` (1); or `available: false`
with its `reason` and `numbers` (the periodicity readings) and its name in `set.json` `not_available`; the periodicity
test in `tests/test_stage2_moves_look.py`. [settled 16 Nov 2026, before any Track B run] On the arrays of the two gaits,
of their starts and of their stops, `foot_pitch` is 0 and `rolling` false in every frame.

**L14. The sets, and the order in which they are built.** [look choice; lead, 15 Nov, before any look set was built] In
this order: (1) `V6` on `75-M-H-10`; (2) `V6noF5` on `75-M-H-10`; (3) `V6` and `V6noF5` on `60-N-T-10`; (4) `V6` on
`75-M-H-10-X` and on `60-N-T-10-X`, built on the `-X` models with the neck yaw and the jaw at their home angles in the
references. `V6noF5` is R2L's set (Q1): the tracker's flat-pad ankle cap and estimator cannot follow a toe-off (plan
B.2). Each set holds L15's walks, the start, the changes, the two-touchdown stop and the 90 degree turns as rules A5, A9
and A10 of `STAGE2_75_PLAN.md` define them, and L13's turn in place with its start and stop. Check: the six files
`stage2/sets/<set>/<body>/set.json` (`set`, `body`, and `freezes[].at` and `freezes[].commit`, in which the order
shows); on the `-X` sets' arrays, the neck-yaw and jaw columns of `q` constant at the home angles (`refs.json`
`q_home`).

**L15. Speeds, and which rungs are dropped.** [look choice; lead, 15 Nov, before any look set was built] The walks are
those of `STAGE2_75_PLAN.md` A2: `nominal` (Froude 0.012 on the body's own hip height) and `v0.3`, `v0.35`, `v0.4`,
`v0.45`, `v0.5`. A rung whose walk is over a leg drive's Stage 0 limit is dropped from that set and body with its
numbers (A12): the look keeps its features and loses that speed (S2-15). Known from gate L1's record (kinematic): on
`75-M-H-10` the V6 walk at the animal pace (Froude 0.05, 0.4961 m/s) is over the knee's Stage 0 speed limit (1.012) and
is not in the look ladder. That walk is the look ladder's rung, not one of A2's; `v0.5` is faster and is therefore
expected to be dropped on that body, which the Stage 0 screen decides when the set is built. A dropped rung's tasks are
absent (T2) and its conditions are "not run" (`STAGE2_75_PLAN.md` H3). Check: `set.json` `speeds` and `not_available`
(each dropped rung with its `reason` and `numbers`); the rung's block in `screen.json` (`verdict.blocks_freeze`);
`stage2/look/L1_FREEZE.json` `not_in_ladder`.

## Not run in this round, by name

Periodic walking-turn gaits; head-first turns (F11); toe-off with rolling contact inside a move while the gait is
changing (the design's rule L9; L10 and L11 as settled); any R5 or R6 run on a set other than V6; the four no-loss
condition kinds the bar does not define on a set (stand, turn in place, start-stop, standing push: T2 names their
counterparts; Track C fixes their episodes and readings in its own rules before any of them runs, and builds and runs
them); a carpet verdict. Each appears in the report as "not run".

[settled 15 Nov, before any Track B run] Outside this text: Track D's levers; a fine-tune runs under plan 4 D and
section 6, by rule K8; third seeds only from freed hours.

As committed, the list above ended "the hardware levers and their fine-tunes; third seeds", which read as dropping from
the round what plan section 6 holds (up to 8 fine-tunes as Colab's third phase, and freed hours for fine-tunes, then for
third seeds). With the ruling the list drops none of it: these three are outside this text and are not declared "not
run".

## Where this text changes the plan's wording

Each item names the sentence of `STAGE2_PLAN.md` that it replaces or makes precise. "[lead]" marks an item the lead
ruled on at 02:45 UTC (the section "Settled by the lead"), "[settled]" an item or a clause that rests on a ruling of the
second list (the last section), "[settled 16 Nov]" one that rests on a ruling of the third list (the amendment that
closes that section). Items 7, 25, 27, 28, 29 and 30 also name rules of `STAGE2_75_PLAN.md`.

1. E6 [lead]: in plan B.4, "It runs `designlab/stage1/train.py`, extended with R5 and R6, a picklable schedule, ..." is
   replaced: the notebook runs the new module `designlab/stage2/train2.py`, which has those abilities, and `train.py` is
   not edited. S2-17's "B.4's extensions of `train.py`, which is in no such key, stand" is not used. Reading 1 is held
   in companion records beside the bar's (U4, U6), not in the bar's own record: in S2-16's "the three readings (shown
   with the look, shown, shown but for A1) carried by every episode", reading 1 is carried by the companion's episode,
   and the bar's own record keeps `counts_look: null`; in S2-12's "every policy-row record then holds both version sets
   (trained under, judged under), the largest fingerprint difference and `cross_version: true`", the largest fingerprint
   difference stands in the companion record (U6, K9).
2. E3, W5 [lead]: plan B.1's "Inputs: Stage 1's deployable ones (`env.py:138`), the last three actions and, on `-X`, the
   head command" gains one input, the heading error. B.1's "Reward: Stage 1's (`REWARD_RULE`) plus foot and contact
   imitation, head attitude in the world, action rate, impact, and neck and tail torque over their limit" gains three
   terms: `progress`, `heading`, `range`.
3. W4 [settled]: the same sentence of B.1 gains a fourth term, a leg-to-leg interference penalty, and an episode ends on
   interference at two consecutive env-step ends (plan section 5 judges interference in the turn item only). [NEW;
   accepted by the lead: point 3 of the second list]
4. W3: within the same sentence, two of Stage 1's reward terms are read differently (`clearance`; the band of `duty`).
5. E4, W2: plan B.1's "commanded by speed and yaw rate" is made precise: the policy's command input is piecewise
   constant on the reference clock, while the tracking reward keeps Stage 1's target, the reference's own speed and yaw
   rate.
6. T2 [lead]: in plan B.1's "References (the approved look): walks at nominal and 0.30–0.50 m/s, walking turns at
   0.35–1.0 rad/s, the 0.5 rad/s turn in place, their starts, two-step stops and changes", the walking turns are the
   finite 90 degree turns and no periodic walking-turn gait is built or trained; a set without the turn in place may be
   trained on. In B.1's "Tasks: Stage 1's plus section 5's", Stage 1's `carpet_walk` task and stops from walks other
   than 0.40 m/s are not trained. Plan section 5's "No loss: Stage 1's 12 gated hard-floor conditions (S1 §2) at 0 ms"
   is run for R5 and R6 on look-set counterparts, with the finite 0.35 rad/s turn in place of the periodic walking turn.
7. M3 [lead]: in plan section 6's "A gain: at least 2 more of its 18 episodes succeeding with the look than at the best
   earlier check, or mean survival 1 s longer", and in S2-10's "or a mean survival at least 1 s longer", the survival is
   the credited survival. M1, U5, U6 [lead]: for policy rows, rule C8 of `STAGE2_75_PLAN.md`, which S2-16 extends to
   every set ("is run again; it is never a failed term"), is replaced: an unread or too-short episode is "not
   succeeding" and is never run again. M1 [settled]: in the monitor an *error* of a policy row's episode counts as not
   succeeding and is not run again either, while R0L's baseline is run again whole (C8).
8. M3b [lead]: S2-10's "A gain is section 6's: at least 2 more of the 18 episodes succeeding with the look (reading 1)
   than R0L, or a mean survival at least 1 s longer" gains one more route when the survival route is closed; the route
   belongs to S2-4's reading alone [settled].
9. M1, M2: in plan section 6's "3 episodes each at 10 ms every 2.5M steps", each of the 18 episodes has its own seed,
   and S2-4 is checked every 500,000 steps.
10. U4: in plan section 5's "(1) shown with the look, the headline for walks and turns", reading 1 is reading 2 for
    nudge and stop items; a fourth count (reading 3 with the bands kept) is printed beside the three readings and is a
    verdict in one case only [settled]: it is the headline of a turn frozen with `a1_over`, which reading 1 does not
    read, printed "shown but for A1, with the look" or "provisional but for A1, with the look". Reading 1's words are
    "shown with the look" and "provisional with the look" [settled].
11. M5: plan section 6's "From 20M, three checks without a gain stop the run" is read as three consecutive checks
    counted from the one at 20M. M5 [settled]: plan section 6's "Cap: 220 runtime-hours or 8 days; then every run is
    judged at the last checkpoint its comparison reached" is made precise: a run ends in one of five ways; every run of
    the Colab queue that passed G8a is judged at its last kept checkpoint, whatever ended it (S2-4 and the local 60 cm
    run are not runs of the queue and are never judged on the panels) [settled 16 Nov]; the 8 days count from the first
    training session's start; at the cap running jobs stop at their next checkpoint.
12. M6 [lead]: a local 60 cm run and a tell-the-owner rule that the plan does not have; its about 5.2 core-hours come on
    top of plan section 6's "about 80 core-hours, estimated". The thresholds of its remedy are fixed [settled].
13. K1: plan B.3's "kill and resume" is made a rule in which every checkpoint is a boundary that reseeds and resets the
    envs, in an uninterrupted run too. K2: in plan B.4's "Checkpoints: each 2.5M check's and the newest two, written
    locally, copied to Drive in pairs", Drive receives the checkpoints at multiples of 1,000,000 steps and each check's,
    each as three files and a record.
14. O1 [lead]: with plan B.4's "8 envs in every run and session", a rollout is 8,192 steps (`n_steps` 1024). O3 [lead;
    settled]: in plan B.1's "the learning rate, the largest of 1e-4, 3e-5 and 1e-5 that runs every minibatch step in
    decision 24's probe", the probe must pass at two sigmas, and one value per body, the smallest passing rate over that
    body's heads, serves every run of the body (the lead's first ruling had one value for every body and head). O4
    [lead; settled]: plan B.1's "The 10 Hz action filter (`env.py:64`) is re-chosen in validation" gets its rule, which
    rests on `75-M-H-10` alone. K8 [settled]: S2-6's "a 5M-step fine-tune from the judged checkpoint" gets its learning
    rate, probed from the parent's checkpoint, with 3e-6 as one more value. N2: plan B.4's "also pinning numpy, scipy
    and torch by version" gets its values, and more packages are pinned.
15. Q1, Q5 [lead]: plan section 6's "R2L 8 (each fit twice Stage 1b's 1.6, S1b N10)" becomes 12 core-hours for three
    fits, and plan B.2's "current heads, one seed per body" gains a third, labelled fit on the 75 cm candidate set. Q5
    [settled]: the three fits have separate shares of 4.0 core-hours, and a fit is cut only if it has not started when
    the report is written. Q4: in S2-18's "with the 10 ms cell decisive and the others reported", a reported cell may
    read "not run: budget"; Q5 [settled 16 Nov]: it starts only when the fit's share still holds its estimate. Q5
    [settled 16 Nov]: R2L's panel records and the re-runs of `R2/42` and `R2/43` beside the candidate's row, which plan
    section 6 counts in "panels and attribution 45 (...; 0.8 per other row)", get caps of 1.5 times the bar runner's
    dry-run total, per set and body; the dry run is made on the frozen set itself and committed before the row's first
    panel record, and a cap is spent unit by unit in the order 10, 0, 4, 6 ms, attribution last.
16. E5: in plan section 5's "`-X` runs the bar head neutral (the hardware's cost) and with gaze events and head-first
    turns (the cost of using it)", head-first turns are not run.
17. N5: plan B.3's "a fresh-clone dry run of the bundle" and "the notebook on a tiny budget" are one acceptance run.
18. Z2: S2-11's parts are ordered anew in two records per body and head: Z2's (a) is S2-11's (c), Z2's (e) is its (a),
    and Z2's (d) holds its (b), asserted for every body and head. Z2 [settled]: on `V6noF5`, part (b) is read without
    the ladder's hash and toe-off and part (c) without the flat-feet playback; (b)'s toe-off limit gains "within 10 % of
    the ladder's number"; each part gets what follows its failure, which S2-11 does not say.
19. L11 [look choice]: S2-15's "The frozen look reference is V6 (every look feature) on both bodies" holds in the
    periodic walks; inside starts, changes, stops and turns the look is carried with flat feet while the gait is
    changing [settled: the choice "inside every move" refined], and a stance of a move that ends after its last ramp
    follows the walk's own law. Plan B.2's "without F5's toe-off" for R2L is the set `V6noF5`.
20. Added, replacing no sentence: among the added rules, the two the lead ruled on, K7 (one writer per run) [lead] and
    Y10 (which smoke items must pass) [lead], and X1 (which episodes may run before S2-4, and after its start
    [settled]). The other added rules carry the tag [NEW] where they stand (among them J4, K5, N4, U1 and U2).
21. Y11: S2-8's "(about 560 env steps/s)" is 40,000,000 / 72,000 = 555.56 env steps/s. Y11, Y4 [settled]: S2-8's "fixed
    once after the smoke for every run" is the first smoke session that records Y4 on the runtime type the launches use,
    for every Colab R5 and R6 run; its "never under 15M without the owner" is: Y4 fails, and no job is queued until the
    owner has set the budget; the rate is read to the end of the third update. Y10 [settled 16 Nov]: the smoke also
    passes when Y4 has failed and the owner has set the budget; the floor is written as `r x 72,000 < 15,000,000` alone.
    N1: the floor of line 4 (the rate at which the budget takes 22 h) is new.
22. M3: in plan section 6's "than at the best earlier check", the first check of every run is read against R0L (S2-10
    says so for S2-4 alone), and R0L's values stay in that best at every later check [settled]. S2-10 [settled]: R0L's
    18 episodes are committed as a baseline record before S2-4 starts, and S2-4 is run once. S2-10's "the two mean
    survivals" are both printed: the credited survival, which decides, and the plain mean survival beside it.
23. K6: S2-13's "One REPO_REF and one run path are kept for a run's whole life" holds unless a session carries
    `ALLOW_SOURCE_CHANGE`, set only on the lead's word and recorded (S2-13's own sentence on a changed source hash).
24. U3: in plan section 5's reading 1, the tail-phase and toe-off terms are not read, and pass, where the reference's
    own amplitude is under 0.02 rad; on `V6noF5` reading 1 therefore has no toe-off band, and every table that sets
    R2L's reading 1 beside R5's says so.
25. Section L, "Windows": on a look set, rule B6's "one stride of `nominal`" is one window of it, and its 2 or 2.5
    strides before a command are join rule J1's leave point [differs: `STAGE2_75_PLAN.md` B6's wording]; B6's nudge
    onset "at 2 + u strides into the 0.40 m/s walk" is (2 + u) x the walk's `period_s` after the walk segment's first
    frame [settled].
26. M4: plan section 6's "the failing monitor tasks' training weights ×2" is made a fixed table from monitor conditions
    to training tasks, each task doubled once; its "a second failure 5M later" is the check of the multiple 10,000,000.
27. X1 [settled]: `STAGE2_75_PLAN.md` rule R4's "the only closed-loop episodes run on a Stage 2 set are the build's
    checks" is read with "a Stage 2 set" as the 75 cm round's three sets (V0, F25F9, F25F9cl), rule R4 being that
    round's own by S2-16; a look set's episodes are limited by X1 [differs: rule R4's wording]. In plan section 7's
    order "R2L; validation and S2-4", the order of R2L and S2-4 is free: R2L's fits may run on a look set once G9-ref
    has passed there [differs: plan section 7's order]. Plan B.3's training start with `MUJOCO_GL` set against it and
    R6's start run on the view `S1`.
28. Q6 [settled]: on the candidate set `F25F9cl`, `STAGE2_75_PLAN.md` H4's replication of a candidate's verdict names R2
    rows only and its H6 holds no R2L unit: `R2L-F25F9cl/42`'s records are reported with A3's label and no verdict word
    and are not replicated; where a key module differs, `R2/42` and `R2/43` are run again beside it from the same commit
    (S2-17). Q6 [settled 16 Nov]: that row is judged by the bar's own readings, as `R2/42` and `R2/43` beside it.
29. U4 [settled]: for a Track B row, `STAGE2_75_PLAN.md` C7, which S2-16 extends to every set, and plan section 5's
    "Stop within a step or two" gain one term, `on_command_at_stop`: the CoM's mean forward speed over the stride before
    the command is at least 0.9 x the commanded speed [differs: C7]. U4 [settled 16 Nov]: the term decides on the look
    sets `V6` and `V6noF5`, where the companion's reading is the verdict of a Track B row and the bar's own record is
    printed beside it, on stop items only, as "the bar's reading (C7)"; on any other set it is recorded and decides
    nothing, and the companion's counts equal the bar's.
30. L12 [settled]: for a join gate failed on a K-stride window, `STAGE2_75_PLAN.md` A8's "retried in this order and no
    other" and "A start and the stop have no fallback", and A11's "A turn has no other fallback", gain the look sets'
    cut 3, taken for a body [differs: A8, A11]; under cut 3, B6's "2 strides of `v0.4` (2.5 for the right turn)" and
    "2.5 strides of `v0.4`", and T1's, read one window of `v0.4` and half a stride for a right-foot command [differs: B6
    and T1]. L12 [settled 16 Nov]: A8's "A failed `start` or `nominal_to_v0.4` stops the set and goes to the lead"
    stands on a look set.
31. "Not run in this round" [settled]: the list drops nothing of plan section 6. Its "at most 8 fine-tunes of 5M" and
    its "Freed hours go to fine-tunes, then third seeds for the head comparison" stand outside this text: a fine-tune
    runs under plan 4 D and section 6, by rule K8, and third seeds only from freed hours.

## Settled by the lead on 15 Nov 2026, before any Track B run

The draft of these rules closed with a list of points open for the study's lead, and marked the rules that waited for a
ruling. The lead ruled on every one of them at 02:45 UTC on 15 Nov 2026, before the build of Track B's code began,
before any look set was built, and so before any Track B episode, fit or training run on a Stage 2 set: they are rules
fixed before results. Each line below says what was decided and, after its last colon, in which rule the wording now
stands; there it is marked "[lead, 15 Nov, before any Track B run]", or "[look choice]" in section L. The rulings
themselves are session notes, not in the repository. A later answer is added as an amendment with its date; one given
after a result is labelled so.

1. The further route to a gain when the survival route is closed is ruled in, as drafted, labelled as differing from
   S2-10; a "no gain" goes to the owner with the per-term failure table of both rows, whatever term it rests on: M3b,
   M3.
2. The heading error is accepted as an actor input, and the reward terms `progress`, `heading` and `range` are each
   accepted with the drafted weight, each labelled as beyond plan B.1's lists: E3, W2, W5.
3. "Walking turns at 0.35-1.0 rad/s" are the finite 90 degree turns, and no periodic walking-turn gait is built or
   trained; "no loss" for R5 and R6 is run on the look-set counterparts, at 0 ms, with Stage 1's committed counts beside
   them and each departure labelled: T2.
4. Survival is credited as progress; an unread or too-short episode of a policy row is "not succeeding" and is never run
   again, and C8 stands for rows without a policy: M3; M1, U5, U6.
5. The plan's initial sigma 0.1 is kept; the owner is told that early training episodes are short; the learning-rate
   probe must pass at sigma 0.1 and at 0.03: O2, O3, and the brief.
6. `n_steps` 1024 with 8 envs is accepted; there is no extension of `designlab/stage1/train.py` and no hook in `bar.py`,
   and the build of Track B's code edits no existing file: O1, E6.
7. R6's expected outcome may be stated in advance; the numbers of head script H1 are accepted: E2, E5.
8. The task weights are accepted; a set without the turn in place may be trained on, and the report then says that plan
   B.1's list is not met for that body: T1, T2, L13.
9. The local 60 cm run is accepted (seed 41, 5,000,000 steps, read by G8b's rule), to be run before any 60 cm Colab job,
   not before the day-one smoke; its cost comes on top of the plan's estimate: M6.
10. One learning rate serves every body and head; the action filter's 0.9 is accepted: O3, O4.
11. R2L's budget is 12 core-hours: two fits on `V6noF5` and one on the candidate `F25F9cl`, in a fixed order; a fit that
    does not fit the budget is cut and reported so: Q5, Q1.
12. Which smoke items must pass, and the 10 minutes of the one-writer rule, are accepted as drafted: Y10, K7.

The look inside moves, each a look choice that the owner is told about:

13. The design's rules for the look in moves stand, and rules A4 to A13 of `STAGE2_75_PLAN.md` apply to the look sets
    (S2-16): section L's opening, rules L1 to L8 and L10, join rules J1 to J3.
14. F5 lives in the periodic walks only; inside every move the feet are flat by rule L10; `V6noF5` is V6 with F5 off in
    the walks as well: L11.
15. Join rule J1 is built as designed; its fallback, used only after a V6 move fails a join gate and the implementation
    has been reviewed, is the design's cut 3 and then cut 4, recorded per set with the failing gate's numbers: L12.
16. The turn in place is a periodic look gait cut from a long plan, gated for periodicity, and "not available" on a body
    where it fails: L13.
17. The sets and the order in which they are built: L14.
18. The speeds are those of rule A2; a rung over a leg drive's Stage 0 limit is dropped with its numbers: L15.

What else differs from the draft, by the author of this text and not the lead's wording, as the text stood before its
independent review. Outside rules E to X: the file's name and place (the draft and the design name it
`stage2/tb/PREREG_TRACKB.md`), and with it the way code and records cite a rule ("STAGE2_TRACKB_PLAN rule K3"), which
the draft did not have; the opening, the brief, "Words, names and tags" (rewritten, with the tags "[look choice]" and
"[design]" added), this section and section L are new; in "What had been run" the bullets on what had not been run and
on what was made outside Track B, and the paragraph after the table, are new, and the scratch experiments' "4-6 episodes
per setting" reads "one to six"; in "Not run in this round" the design's rule L9 is added and the clause on the no-loss
kinds is reworded; "Where this text changes the plan's wording" names the plan's sentences and has five more items (3,
9, 18, 19 and 20: W4; M1 and M2; Z2; L11; the rules added without replacing a sentence). Inside rules E to X: the
clauses that said what would follow if the lead struck a rule are gone (E3, W5, M3b, Q5); E5 names the lead's ruling "of
14 Nov on the references' design"; Q1 names P30; "as ruled" reads "as settled" in E6, M3 and Z2; in W2 the source of the
`interference` term reads "new (W4; not in plan B.1's list)" where the draft read "new (plan)", plan B.1's reward list
holding no such term, and the foot term's distance is written in words, to keep the table readable; T2 and Z2 are set in
several paragraphs, each with one Check at its end; T2's Check also names the label of the no-loss counterparts, and
Q5's the report's cell for a fit that was cut.

Changed after the independent review, by the author and not by the lead, on findings that needed no new decision; the
lead confirmed them later on 15 Nov 2026 (point 25 of the second list). The Status and "What had been run": the claim
"before any Track B episode, fit or training run" is scoped to Stage 2 sets and given its times; the Jetson body's dates
and its provisional G1; the bullets on the builds' tests and on the look work package's scratch. The brief: G8a's "by
construction"; the 10,000,000-step check; the cap's wording in place of a promise; the look choices of L12 and L14; four
more things that may come back to the owner; the line on the open points (since replaced by the bullet on their
settlement). E2, E3, O1, O3 and K7: scratch and assumption labels, and what "a quantity the robot has" means in
simulation. E6: a Check for "edits no existing file". T1: "to the end", and where the weights come from. T2: the four
no-loss kinds that have no reading rule yet. W2 and W4: the label "differs: plan B.1's reward list". M3: what S credits;
the plain mean survival printed beside it. M3b: "within 1.0 s" read by the rule's own reason. M4: checks named by their
multiple, which corrects an error of 2,500,000 steps; "up"; each task doubled once. U3: V6noF5, and what gate L1's
record already showed of the ladder's toe-off. U4: reading 1 of a turn frozen with `a1_over`. U5 and Z2 (b): the 4 steps
and 4 lift-offs checked for every item. Z1: what follows a failure. Q2: the claim on directions and delays, which held
only from 24 generations; Q4: the order of the selection key. Y11 and N1: the exact rates, and the label of line 4's
floor. "Windows": how B6's stride counts read on a K-stride set. L15: which rung gate L1's record names. "Not run in
this round": the four no-loss kinds. The change list: items 1 and 20, and the new items 21 to 28. Every pointer to an
open point was new as well; those pointers went when the points were settled (the second list).

## Settled by the lead on 15 Nov 2026, before any Track B run (second list)

The independent review of this text, made before its commit, left 25 points for the study's lead. Each needed a decision
that neither the draft, the lead's rulings of 02:45 UTC nor the designs held, or, for point 25, a confirmation that only
the lead can give. The commit that adds this file (15 Nov 2026, 03:34 UTC) holds them in a section "Open for the lead",
each with what was unsettled and with the reviewers' proposed wording. The lead ruled on every one of them on 15 Nov
2026, after that commit and before any closed-loop episode on a look set, any fit and any training run of Track B: they
are rules fixed before results. Each line below says what was decided and, after its last colon, in which rule the
wording now stands; there it is marked "[settled 15 Nov, before any Track B run]". Where the lead accepted a proposed
text word for word, the rule holds that text; where the lead gave another wording or chose one of two proposals, the
rule holds that. The rulings themselves are session notes, not in the repository. Points 4, 5 and 6 are about section L
and were settled after the look work package's first kinematic numbers existed ("What had been run"): each settled
wording says so. Six rulings change, refine or make precise an earlier ruling or look choice of the lead (the first
list): points 4, 5, 6, 8, 11 and 17; each says so in its rule and in its line below. A later answer is added as an
amendment with its date; one given after a result is labelled so.

1. G9-ref on `V6noF5`: part (b) makes no hash and no toe-off comparison with the ladder and records the toe-off as "not
   read"; bob, head and tail are compared with the same body's V6 set's own playback; part (c)'s flat-feet playback
   applies to V6 only; R2L's set must pass G9-ref in this form before its first episode: Z2, X1, Q1.
2. Rule R4's "a Stage 2 set" (`STAGE2_75_PLAN.md`) means the 75 cm round's three sets, rule R4 being that round's own by
   S2-16; a look set's episodes are limited by X1, which is tagged "differs: rule R4's wording" and listed in the change
   list (item 27): X1.
3. The leg-to-leg interference penalty is accepted with its drafted weight, and the termination is accepted: a fourth
   term beyond plan B.1's reward list, tagged so, for the reason of `LESSONS_LEARNED.md` section 2.8: W4, W2.
4. The fallback of join rule J1: the proposed text is accepted word for word (the trigger and the review's record; A8's
   retries first for a start or a change; cut 3 for a causal move or when A8 is exhausted; a cut taken for a body, with
   both heads and its R2L set; "not available" for a move that still fails; T1's and B6's stride counts under cut 3),
   tagged as differing from A8 and A11, and cut 3 from B6 and T1. A set rebuilt under a cut before any closed-loop
   episode on it is no change after a result, and after one it is. A nudge's onset on a K-stride window is (2 + u) x
   `period_s` after the walk segment's first frame. This changes look choice 15 of the first list in its trigger and its
   scope: L12, T1, "Windows".
5. The turn in place is cut from a plan with 4 hold strides more than Stage 1's start writes; the reference is the
   second of them, one stride without F14 variation and with flat feet, compared with the hold strides before and after
   it; no other length and no other stride is tried. Cut 4 is in force from the start for the turn in place and is no
   further fallback, which changes look choice 15 of the first list ("cut 3 and then cut 4"): L13, L12.
6. The instant of a start or a change is the start of the move's ramp. Every stance that ends between the instant and
   the end of the move's last ramp is flat; a stance that ends after the last ramp follows the walk's own law; the
   frames before the instant equal the look walk's; the same holds for a causal move from its command. This refines look
   choice 14 of the first list, "flat feet inside every move": flat while the gait is changing: L10, L11.
7. `R2L-F25F9cl/42`'s bar records are reported with A3's label and no verdict word, and the row is not replicated; it
   runs `walk/v0.4` and `walk/v0.4_mu0.54` at 10, 0, 4 and 6 ms and the other fourteen gated conditions at 10 ms, on
   seeds 3042-3061, beside `R2/42` and `R2/43`, which are run again from the same commit where a key module differs: Q6.
8. How a fit's share is spent (three separate shares of 4.0 core-hours; block by block; the decisive cell always whole;
   a fit "cut" only if it has not started when the report is written), which makes precise ruling 11 of the first list;
   the reported cells run delay by delay, the selected candidate and then the start. Both proposed texts word for word:
   Q5, Q4.
9. In the monitor an *error* of a policy row's episode counts as not succeeding, as zero credited survival and as not
   up, and is not run again; R0L's baseline is repaired and run again whole; a check without a record is run again once.
   The first proposal, word for word: M1, with a pointer in M4.
10. N* and S* are the largest N and the largest S over R0L's 18 episodes and every earlier check of the run, each taken
    separately; for the first check they are R0L's. The first of the two proposals: M3.
11. The second route to a gain belongs to S2-4's reading alone; in every other run's checks its count is printed and
    decides nothing. The first of the two proposals; it fixes the scope of ruling 1 of the first list: M3b.
12. R0L's baseline is committed as `R0L_BASELINE.json` before S2-4 starts, and S2-4's own baseline must equal it; S2-4
    is run once, and its checkpoint is never judged on the panels. The proposed text and Check, word for word: M3.
13. A run ends in one of five ways; a halted or disconnected run is resumed; every run that passed G8a is judged at its
    last kept checkpoint, whatever ended it; the 8 days count from the first training session; at the cap running jobs
    stop at their next checkpoint. The proposed text, word for word: M5.
14. The thresholds of the 60 cm run's remedy ("up", "sigma near 0.1", "most minibatch steps cut", "a small sigma"). The
    proposed text, word for word: M6.
15. A stop item of a Track B row has one more term, `on_command_at_stop`; an episode that fails it succeeds in no
    reading (NEW; differs: C7). The first of the two proposals: U4, with pointers in E2 and M3.
16. A turn frozen with `a1_over` reads "not read: a1_over" in reading 1; its headline on a look set is reading 3 with
    the four bands kept, "shown but for A1, with the look" or "provisional but for A1, with the look", a verdict for
    that case only. Reading 1's two words are "shown with the look" and "provisional with the look": U4.
17. The learning rate is one per body. The probe runs once per body and head; every run of a body uses the smallest
    passing rate over that body's heads probed when the run starts; a body's `-X` head is probed before any Colab job of
    that body is queued; the two bodies may differ, and the report says so. This changes ruling 10 of the first list,
    "one learning rate serves every body and head", before any run: O3.
18. The filter rule rests on `75-M-H-10` alone, the 60 cm body's measurement being reported; the sigma probe's episodes
    run on hard floor with no push and no head event. Both proposed texts word for word: O4, O2.
19. What follows a failure of each part of G9. The proposed text, word for word: Z2, with a pointer in T2.
20. Part (b)'s toe-off limit is 0.02 rad and 10 % of the ladder's number; U3's threshold stays 0.02 rad, and every
    record prints the reference's value and the side of the threshold it lies on: Z2, U3.
21. The fine-tune's learning rate: the parent's schedule value if every minibatch step runs at it, else the largest of
    1e-4, 3e-5, 1e-5, 3e-6 below it at which they all run; if none does, no fine-tune. The proposed text, word for word:
    K8.
22. The budget is fixed from the first smoke session that records Y4's rate on the runtime type the launches use; under
    15,000,000 steps Y4 fails and no job is queued until the owner has set the budget; Y4's rate is read to the end of
    the third update. The proposed texts, word for word: Y11, Y4 (in the table), with a pointer in Y10.
23. The training start with `MUJOCO_GL` set against it and R6's start run on the view `S1`; from S2-4's start every
    closed-loop episode on a look set belongs to a named run, check, gate, probe or panel, or is listed before it runs
    (the second of the two proposals). R2L's fits are added to what may run on a look set once G9-ref has passed there,
    and the order of R2L and S2-4 is free, a labelled change to plan section 7's order: X1, Q1.
24. Track D's levers, the fine-tunes (under plan 4 D and section 6, by rule K8) and third seeds (only from freed hours)
    are outside this text and are not declared "not run". The first of the two wordings: "Not run in this round".
25. The list of what had been run is confirmed as written, with its completion; the corrections that the author applied
    after the review are confirmed; the builders of the remaining packages are given this file's name and its citation
    convention by the lead: "What had been run", and the last paragraph of the first list's section.

What else changed in this text with the second list, by the author and not the lead's wording. This section took the
place of the list of open points. The Status, "What had been run", the brief, the Tags entry, the change list (items 3,
7, 8, 10, 11, 12, 14, 15, 18, 19, 20, 21, 22, 25, 27 and 28, and the new items 29 to 31) and the "Check:" lines follow
the settled wording, and every clause that pointed to an open point, or said what held until one was settled, is gone.
Rules were set in several paragraphs so that each marked wording ends with its paragraph (O3, O4, M1, M3, M3b, M5, M6,
U4, Z2, Q5, Q6, K8, Y11, X1, L10 to L13, "Not run in this round"). Inside the marked wording, a lead-in that names its
subject and a cross-reference to a rule of this text are the author's, as is the second word of U4's pair, which the
ruling gives as "provisional ..." and which is written out here. Outside the marks, and labelled where they stand, these
are the author's: the paragraphs that quote a rule's committed wording and say what the settled wording changes in it
(O3, O4, M3, Z2, Q5, K8, Y11, X1, L10, L12, the list "Not run"); in Z2, the reading of the comparison on `V6noF5`, for
which the ruling names no limits; in L10, the account of what stands of the committed rule; in L13, the review's note on
the look work package's first version; the pointers in E2, T1, T2, M3, M4, Q1, Q6 and Y10 to wording settled in another
rule; in "Not run in this round", the words "while the gait is changing"; in the brief, the sentence on a 75 cm run
stopped while ahead in the printed count, which follows from M3b and M5. The time of day of the second list's rulings is
not given in this text.

**Amendment of 16 Nov 2026: the third list.** Writing the second list in left ten places where two rules disagreed, or
where a rule disagreed with `STAGE2_75_PLAN.md`
(`stage2/design/checkpoint/reviews/trackb_rules__integrate_trackb-rulings.md`). The lead ruled on all ten on 16 Nov 2026
(`stage2/design/TRACKB_RULINGS3_16NOV.md`, rulings 1 to 10), before any Track B run, fit or training run and before any
look set was frozen; the 75 cm round's panels and their add-on had run ("What had been run"). Each line below says what
was decided and, after its last colon, in which rule the wording now stands; there it is marked
"[settled 16 Nov 2026, before any Track B run]". Rulings 2, 3 and 4 are about section L and were settled after the look
work package had built look sets (prescribed motion; none frozen): each settled wording says so. Rulings 11 to 13 of
that file are about the policy files and the reward's wording and are not written in here.

1. Which runs are judged: every run of the Colab queue that passed G8a, at its last kept checkpoint, whatever ended it.
   S2-4 and the local 60 cm run are not runs of the queue, and their checkpoints are never judged on the panels. This
   makes precise point 13 of the second list: M5.
2. The turn in place has flat feet as its own law, in every stride, the start and the stop included; L13 no longer cites
   L10, which does not govern this gait. The look work package's code was read for it: L13.
3. Before the instant of a change the foot pitch follows the law of the walk it leaves; the rest of the frame is the
   move's own, built with foreknowledge (A5), and A7's gates read it; a start has no frame before its instant but the
   stand. This corrects point 6 of the second list, whose "the frames before the instant equal the look walk's" was
   wrong as written: L10, L11.
4. A8 stands on a look set: a `start` or a `nominal_to_v0.4` that still fails stops that body's set, which is then not
   frozen, and goes to the lead. "Not available" is for the other moves: a ladder change, the stop, a turn and the turn
   in place. This corrects point 4 of the second list in its clause on a move that still fails: L12, and the brief.
5. On `V6` and `V6noF5` reading 1 lives in the companion record and is the verdict of a Track B row; the bar's own
   record is printed beside it as "the bar's reading (C7)", and where the two differ on a stop item the table prints
   both and names `on_command_at_stop`. On any other set reading 1 does not count: `R2L-F25F9cl/42` is judged by the
   bar's own readings, as `R2/42` and `R2/43` beside it, and the term is recorded and decides nothing. This fixes the
   scope of point 15 of the second list: U4, Q6.
6. The panel records of each R2L row, and the re-runs of `R2/42` and `R2/43` beside the candidate's row, each get a cap:
   1.5 times the bar runner's dry-run total at the committed rates, per set and body, counted in the bar's own
   `budget.json`; the 6, 4 and 0 ms records are cut first, in that order, and the 10 ms records never. The dry runs'
   commands and numbers stand in the rule; the `V6noF5` numbers are estimates from stand-in rows: Q5, with a pointer in
   Q6.
7. An attempt at a check is logged before the monitor starts and closed with its exit code. M1's "once" counts closed
   attempts that left no record: after two the check counts as without a gain and as a failed G8b reading. An attempt
   never closed is an interruption, which K5's resume runs again and which does not count. This makes precise point 9 of
   the second list: M1, K5, with a pointer in M4.
8. The smoke passes when Y1, Y3, Y6 and Y9 pass and Y4 passes or has been answered by the owner's budget; Y11's floor is
   the inequality `r x 72,000 < 15,000,000` alone, with no rounded rate beside it. This changes ruling 12 of the first
   list in Y10's list of items: Y10, Y11.
9. A reported cell starts only when the fit's share still holds its estimate (the decisive cell's mean seconds per
   episode, times 25, times 1.25), and otherwise reads "not run: budget"; a started cell finishes, an excess is recorded
   (`overrun_reported`) and no further cell starts; the decisive cell's overrun is "the only one that may pass the share
   without limit". This makes precise point 8 of the second list: Q5.
10. X1's first sentence gains "(lifted for the frozen 75 cm sets on 16 Nov 2026, after the panels)"; the look sets' part
    of X1 is unchanged. The lead's own amendment 13 of `STAGE2_75_PLAN.md` records the lifting there: X1.

What the integrator checked for the third list, and what else changed in this text with it; this paragraph is the
author's and not the lead's wording. For ruling 2, the look work package's code and its unfrozen build were read (L13
says what was seen). For ruling 6, the dry runs were made (Q5 gives the commands and the printed totals), and the
re-runs Q6 names are `R2/42` and `R2/43` on the candidate set from the R2L row's own commit, as the ruling assumes.
These follow the settled wording and are the author's: the sentence added to the Status, the bullet in "What had been
run", the Tags entry, the brief's two changed sentences and its bullet, the change list's clauses marked "[settled 16
Nov]", the additions to the "Check:" lines, the pointers in M4 and Q6, and L11's clause, which repeats L10's; in K5, the
sentence that a check counted by M1 is not run again by a resume, a reading of M1's count that is labelled where it
stands; in Q5, the reading of "where the plan asks", the choice of the frozen set `F25F9cl` itself for the candidate's
two dry runs, the labels of the stand-in numbers and the note on the unfrozen build of `V6noF5`; in U4, L10 and L13, the
accounts of why a wording was settled; in L12, the closing clause of the committed sentence set as a sentence of its
own; in Y11, the parenthesis left out of the quotation of the committed sentence.

Changed after the independent review of this writing-in (16 Nov 2026), by the author and not by the lead, on findings
that needed no new decision. The Status's Check names the last of the commits that add the wording of 16 Nov (it named
one commit, and there are several). U4: on any other set the sentence names the row `R2L-F25F9cl/42`, as the ruling does
(it read "a Track B row"). Q5: the 60 cm body's caps no longer stand "until its own rates exist", a condition the ruling
does not hold. K5: the author's sentence is labelled as such where it stands, with what the ruling does not say. L12:
the mark quotes the whole committed sentence, and its closing clause again holds for every move that still fails (it
read "such a move"); the Check's exception names the set in which the move fails, as the ruling's "that body's set" does
(it read "no set of a body"). L13: the sentence on L10 is the ruling's own (it read "a walk's moves (its starts and
changes, and its stop and turns from their command)"), and the note on L10's causal moves stands in the author's
account. Seen in that review and left as written: this text calls the lead's rulings of 15 Nov session notes that are
not in the repository (the Tags entry, and the two sections that list them); the working branch has held both lists as
working papers since commit `a0314fc` (`stage2/design/TRACKB_RULINGS_15NOV.md`,
`stage2/design/TRACKB_RULINGS2_15NOV.md`). The points of that review that need a decision are with the lead and are not
written in.
