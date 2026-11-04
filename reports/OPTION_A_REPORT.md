# Option A: the owed CPU checks on the three PHASE finals (report only)

2026-11-03 (UTC). CPU only. Nothing was trained, Drive was used read-only, and nothing was pushed.

**Labels.** **[measured: source]** means read or computed from the named file. **[inferred]** means reasoning, with its
basis given.

**Paths.** `W` = `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work`.
- Lane results and the verifiers' corrections: `W/measure/LANES_AND_VERIFY.txt`.
- This report's own adversarial check: `W/optA/check/`, where every correction it listed has been applied.
- Run log: `W/run_log.txt`.

**What does not change.** The pinned reading is unchanged: line 4, "CONSISTENT WITH THE PHASE CRITIC REMOVING THE
HABIT, NOT CONFIRMATORY". THE ONE TEST stays NOT SEPARATED (p = 20/20). Everything below is report only, with one
exception: M4 feeds the pinned parent rule, as the round's criteria require (criteria.txt lines 269-272).

## Short answer

- **K1-K4 hold in all three PHASE runs** [measured: `W/measure/m4/kscore/`]. These are run-level criteria judged over
  the six J checkpoints (4.2M to final); K1's pass test, K3 and K4 are scored at 6.3M and the final. The same-seed
  STANCE6 runs held K1-K4 in 1 of 3: s42 failed K1, and s80 failed K1 and K3 [measured as recorded: origin/main
  `stance6_results/kscore/`]. Reading those STANCE6 results as the OFF runs' results is [inferred; see "OFF =
  STANCE6" below].
- **The pinned parent rule names a tie: s42 and s80** [measured: `W/measure/m4/parent/parent_M4.json`].
  - All three finals had 40/40 clean panel episodes. The tie-break counts noise falls, which were 1 / 0 / 0 for s60 /
    s42 / s80.
  - s60's one fall is a single tail_contact at step 226 on seed 3061. That seed ends early under joint noise now and
    then in every run, so the tie rests on a thin basis.
  - The rule reads only the final. It does not break ties.
- **The tail stayed near zero at all 12 checkpoints measured** (the 11 on the 1,048,576-step grid plus the final) in
  all three PHASE runs [measured: `W/measure/m2m1/` M1 at the 6 non-J labels; F2's M1 at the 6 J labels].
  - The 20-episode mean command of each tail motor was at most 0.068 in size.
  - Single episodes reach 0.151, and single steps 0.168 [measured: `W/optA/check/tail_check.txt`, `tail_step.txt`].
  - There was no saturation and no flag.
  - This covers 1.0-3.1M steps, inside the 1.2-4.5M window where earlier habits began [measured as recorded:
    ANSWER.md citing SCORE 322/326/330].
  - Each run also saved 34 other checkpoints (every 262,144 steps), which were not measured.
- **The head sits at about 0.90 m** [measured: `W/measure/m2m1/head_figures.json`].
  - Panel means at the finals are 0.902 / 0.904 / 0.902 m: about 5.5 cm below the statue's 0.959 m, and just above
    the reward's 0.90 m full-credit bar.
  - Single episodes dip below 0.90 m in 10 of the 18 panels (lowest 0.888 m) [measured: `W/optA/check/head.txt`].
  - 719 of 720 M2 episodes are clean. Head posture is report-only under D-D46.
- **The critic readouts are consistent with the hidden-clock explanation and do not confirm it** [inferred; the
  numbers and exceptions are below].
- **PHASE s60 falls more than s42 and s80.** Its 7 evaluation falls after step 200 each carry a neck command of 0.5 or
  more [measured]. Calling that a "neck-involved fall mode" is a classification [inferred]. From 4.2M steps, 69 of its
  training episodes ended in a fall, against 2 for s42 and 2 for s80 [measured]. This does not block anything,
  because the rule did not name s60.

**What this means for the next decision.** Both tied runs meet the bar recommended earlier (K1-K4 pass). Choosing
between them is a judgement on small, mixed differences (table below). Neither can be a walking parent yet:
- the branch refuses a warm start from a PHASE checkpoint, and main cannot load one (needs the warm-start rule);
- walking from either needs your declared exception to GQ-3 (a) and GQ-17;
- neither is certified on main (0 of 2).

## s42 against s80 (the tied pair)

Every row is [measured]. The "favours" column is [inferred].

| Readout | s42 | s80 | Favours |
|---|---|---|---|
| Foot centre-of-pressure (K2c median, 4.2M → final) | flat, 52.6 → 50.6 mm, range 2.1 mm, final offset −2.2 mm | steady fall, 50.2 → 46.2 mm, range 3.9 mm, final offset −6.5 mm | s42 |
| tail_2 command at the checkpoints (M1 mean) | −0.025 at 5.2M → −0.037 at final | +0.010 at 7.3M → +0.019, +0.025, +0.035, +0.061 → **+0.066** at final: a steady rise | s42 |
| tail_1_pitch / tail_3 at final (M1 mean) | −0.017 / +0.015 (tail_3 rose from −0.004 at 4.2M to +0.025 at 11.5M) | +0.042 / +0.028 | s42, slightly |
| Head pitch (M2, 6.3M → final) | 12.9° → 14.7° | rises steadily from 14.0° to 17.8° | s42 |
| Neck pitch (M2) | the most nose-down of the three, 16.0-16.7° (final 16.2°; statue 12.3°) | 13.9-15.6° | s80 |
| Critic error just after step 200 (V − G in [150,200) / [200,250)) | −0.023 (9.1 SE) / −0.015 (7.8 SE) | −0.010 (5.4 SE) / −0.008 (4.4 SE) | s80 |
| Loop margin (smallest b* over 12 pairs) | +1.53 (G −0.43 at 1.0M) | +1.97 | s80 |
| Action noise at the final (hip_roll / neck std) | 0.0520 / 0.0833, the lowest of all six runs; fell below 0.08 the earliest (5.1M) | 0.0521 / 0.0882; below 0.08 at 5.8M | s80, slightly |
| Observation drift (hip_roll) | 0.243, the largest | 0.200 | s80 |
| Clean training evaluations (F2) | 899 of 910 | 903 of 910 | s80, slightly |

[inferred] For a walking parent, the actor's behaviour carries over and the critic is retrained or replaced. On that
basis the actor rows weigh most: K2c, tail_2, tail_1 and head pitch favour s42, and neck pitch favours s80.
- s80's tail_2 rise is small (+0.066), but it is steady, and it is the very kind of slow tail drift this study is
  about. In walking, nothing prices a parked tail.
- s42's main drawback is having the lowest action noise. The noise-reset option for the pilot would address that.
- The critic rows favour s80 but matter less: the warm-start rule drops or replaces the critic's phase inputs.

So s42 is a slight preference, not a clear win.

## What was done

1. **Re-fetch** [measured: `W/fetch/fetch_log.csv`; helper `W/tools/fetchcopy.py`]. Six runs were downloaded again,
   read-only. Each run's set was its 11 checkpoint pairs on the 1,048,576-step grid plus the final pair (OFF s42 also
   has its 3,670,016 pair) and the run files the measurements read: 199 files in all. Each was stored only if its size
   and sha256 equalled the PC round's own F2 manifest.
2. **Tools ported, then validated against stance6's records.**
   - M4 judge, kscore, the parent rule and M2/M1: only paths, the commit binding and the approval record changed.
   - The critic tools: the phase-critic input path was added (`W/port/critic/NOTES.md` section 2).
   - The D-item reader: two new reducers were added, th_read.py and collect.py. Neither was validated against a
     record.

   On the OFF checkpoints, each tool reproduced STANCE6's recorded results:
   - the M4 judge on OFF s80 final, bitwise in every panel, noise and check value, on both code copies;
   - kscore on all 18 of STANCE6's judge records;
   - M2 on OFF s80 final;
   - loop gain and critic slopes, within the 5-dp rounding of the early-warning records;
   - the D-items against STANCE6's READOUTS, and D-9/D-10 against its d9_d10 records;
   - the V − G tool, on the phase screen's recorded stance6 checkpoint.

   An adversarial reviewer checked each tool, with no blocking finding [measured: `W/port/review_*`].
   Validation gaps:
   - 2 of the V − G tool's 55 tests fail here, because root mode refuses this machine's checkpoint; only pair mode was
     used.
   - The phase-critic path of the critic tools has no external record to compare against. It was checked with
     synthetic critics and for reproducibility.
3. **Measured** [measured: `W/measure/<lane>/`]:
   - M4 at the 6 J labels × 3 PHASE runs;
   - M2 at J × 3 PHASE runs, plus a statue panel;
   - M1 at the 6 non-J grid labels × 3 PHASE runs, after first reproducing F2's PHASE s60 final M1 exactly (all 20
     per-episode CSV hashes equal);
   - critic readouts at 12 pairs × 6 runs, and V − G at the 6 finals;
   - D-items for all six runs, and D-10 at the PHASE finals;
   - the s60 falls.

   An independent verifier per lane then re-ran 3-4 of that lane's simulations. All were **bitwise equal**. The
   verifiers also recomputed the lanes' headline numbers with their own code. This report's own derived statements
   (ratios, comparisons, the s42/s80 table) were checked separately by `W/optA/check/`.
4. **A container restart** stopped the first measurement attempt at about 00:38 UTC. Complete, valid outputs were
   kept. The verifiers confirmed that nothing was re-measured to replace a record:
   - M4 shows 19 judge launches: 12 kept, 1 killed by the restart, and 6 after the resume;
   - the critic resume skipped the 8 rows it already had.

## Results

### M4: K1-K4 and the parent [measured: `W/measure/m4/SUMMARY.md`, kscore JSONs; verified by `W/measure/verify_m4/`]

| Run | Clean panel episodes /40 at 4.2M, 5.2M, 6.3M, 8.4M, 10.5M, final | Joint-noise falls /20 (6.3M, final) | K2c range | K1 | K2 | K3 | K4 |
|---|---|---|---|---|---|---|---|
| PHASE s60 | 39, 40, 40, 40, 40, 40 | 0, 1 | 2.85 mm | PASS | PASS | PASS | PASS |
| PHASE s42 | 40 at all six | 0, 0 | 2.11 mm | PASS | PASS | PASS | PASS |
| PHASE s80 | 40 at all six | 0, 0 | 3.92 mm | PASS | PASS | PASS | PASS |
| STANCE6 s60 (as recorded) | 40, 39, 40, 34, 38, 40 | 1, 1 | 7.12 mm | PASS | PASS | PASS | PASS |
| STANCE6 s42 (as recorded) | 3, 30, 39, 4, 7, 38 | 1, 0 | 19.53 mm | **FAIL** | PASS | PASS | PASS |
| STANCE6 s80 (as recorded) | 39, 37, 5, 40, 40, 40 | 5, 2 | 12.83 mm | **FAIL** | PASS | **FAIL** | PASS |

- **Own-sigma falls and roll rails.** Own-sigma noise falls were 0 of 20 at all 18 PHASE checkpoints, and there were
  no roll-rail episodes.
- **The one unclean episode.** Of 720 PHASE panel episodes, one was unclean: s60 at 4.2M, seed 3072, a backward fall
  at step 221. That panel still passed, with LCB 0.887.
- **Joint-noise falls on seed 3061.** At 6 of the 18 PHASE checkpoints, one episode in 20 ended early, always on seed
  3061: s60 at 4.2M, 5.2M, 10.5M and final; s42 at 8.4M; s80 at 8.4M.
  - Two of the six ended in tail_contact rather than a fall: s60 final at step 226, and s80 8.4M at step 292.
  - Seed 3061 also ended early under joint noise in 10 of STANCE6's 18 J cells [measured: `W/port/m4/ref` judge
    records].
- **The parent rule** was run unchanged (via `W/port/m4/pc/parent_m4.py`), with the pinned classes from
  `score_pc.json`.
  - With no M4 records it reproduces the recorded "not named: an input is absent".
  - With them, it names [s42, s80], tie true.
- **OFF = STANCE6.** The last round measured equal ledgers, training rows, and final policy and optimizer weights
  (run log line 970). On this machine, M4 at two OFF cells (s80 final and 6.3M) equals STANCE6's records. For the
  other 16 cells, OFF = STANCE6 is [inferred]. The OFF and STANCE6 zip files differ in bytes while every judged value
  is equal [measured: `W/port/m4` V2].

### M2 head posture, and M1 tail at the non-J labels [measured: `W/measure/m2m1/SUMMARY.md`; verified]

- **PHASE head tip at J (panel means): 0.902-0.941 m.**
  - That is 0.002-0.041 m above 0.90 m, and 0.018-0.057 m below the statue's 0.9586 m.
  - Single episodes reach 0.888 m.
  - Neck pitch is 13.5-16.7° (statue 12.27°), and head pitch 12.9-17.8° (statue 12.64°): a little more nose-down than
    the statue.
- **OFF (= STANCE6, as recorded).**
  - The head tip is above the statue in 15 of 18 panels, up to 1.35 m.
  - Neck pitch is below the statue's in 15 of 18 panels, and negative in 5.
  - The tail_3 command reaches 0.98.
- **Trend.** From 4.2M to the final, the head drops in all three PHASE runs: s60 0.935 → 0.902, s42 0.913 → 0.904,
  s80 0.939 → 0.902. Only s80's drop is monotonic.
- [inferred: code] Above 0.90 m the clearance reward is flat. A weak pull remains through the neck_posture term
  (weight 0.20), which is centred on the statue's less nose-down neck.

### Critic readouts [measured: `W/measure/critic/SUMMARY.md`; verified. Interpretation inferred]

| V − G arm median by episode step | [0,50) | [50,150) | [150,200) | [200,250) | [250,1000] |
|---|---|---|---|---|---|
| OFF (stock critic) | +0.019 | **+0.071** | +0.013 | **−0.094** | −0.014 |
| PHASE (phase fed) | −0.003 | −0.001 | −0.014 | −0.009 | −0.003 |

- **OFF at 12M** [measured]. In all three seeds the OFF critic overvalues steps 50-150 (3.5-5.5 SE) and undervalues
  steps 200-250 (7-12 SE). That this is the "clock-shaped" error is [inferred: matches the why-probe and the phase
  screen].
- **PHASE against OFF, mean |V − G|** [measured]. Across the step-200 boundary, PHASE is 4-8 times smaller on the arm
  medians, and 3.0-9.8 times smaller per seed and bucket.
- **The PHASE critic is not error-free** [measured]. It sits 0.008-0.023 below the true return in [150,200) and
  [200,250) in all three runs, at 4-9 SE. The pre-registered expectation E1 for [150,200) reads "not consistent" as
  worded.
- **Loop gain** [measured]. Loop gain G (how far a higher tail angle raises the actor's own tail command) is negative
  at all 36 sampled PHASE pairs. OFF's is positive at 20 of 36. A negative loop gain is also what any run that never
  formed the habit would show [inferred].
- **The critic's tail_3 slope** [measured]. In s42 and s80 it is smaller under PHASE than under OFF at 10 and 11 of 12
  pairs. **In s60 it is not smaller** (median 0.0145 vs 0.0125).
- **Reliance on the phase** [measured]. With the phase inputs zeroed, the PHASE critic's value drops by 0.26-0.48
  after step 150.
- [inferred] Consistent with the hidden-clock explanation being cut at its first link. It is not confirmed: 3 seeds,
  PHASE and OFF diverge from step 0, and there is no sham-input arm.

### PHASE s60's falls [measured: `W/measure/dread/falls/`; corrections from `W/measure/verify_dread/`]

- **Evaluation falls.** 9 of 910 evaluation episodes ended "fallen".
  - Two ended before step 200 (settle failures). The other 7 ended 11-178 steps after step 200.
  - PHASE s42 and s80 had none; each OFF run had 1.
- **Neck commands in the 7 post-200 falls.**
  - Each carries a window-mean neck command of 0.5 or more.
  - 3 have neck pitch about −0.64 and end with root pitch +34° to +38°.
  - 4 have neck yaw ±0.57 to ±0.89 and end with root roll 12-28°.
  - Grouping them as "backward" and "lateral" falls is [inferred].
  - Tail commands are small, and no actuator is over the bar.
- **The comparison base is not like for like.** None of the 901 full-horizon episodes has a neck command that large.
  But an 11-178-step window is compared with an 800-step one, which dilutes the full-horizon mean 5-70 times.
- **Training falls.** From 4,194,304 steps, training episodes that ended "fallen" were 69 of 7,860 for s60, 2 of
  7,808 for s42, 2 of 7,809 for s80, and 8 of 7,816 for OFF s80.
- **In-run K3 panel.** In s60's panel, seed 3061 falls under joint noise at 7 of 12 panels. OFF s80's panel shows the
  same seed falling at 7 panels with no evaluation fall, which weakens any link to s60's evaluation falls.
- **Candidate explanations** [inferred]:
  - a narrow balance margin, with the neck as a failing recovery;
  - the neck excursion itself causing the fall;
  - spawn severity (fits only some falls);
  - chance.

  Per-step traces at a fall checkpoint, a neck hold at 0, and per-step neck traces of full-horizon episodes would
  separate them. None of these needs training.

### Other D-items [measured: `W/measure/dread/out/collect.txt`; verified]

- **D-1.** No PHASE evaluation row has a saturation fraction over 0.10.
  - PHASE s60 has two pre-settle rows (3,407,872 episode 9 and 3,538,944 episode 4) with a nan fraction and the key
    listed. Under the pinned ANY rule these count as ANY-over.
  - s42 and s80 are clean under either reading.
- **D-10.** Own-noise replays at the PHASE finals ran 20 of 20 to full horizon, with no flags. Their tail_3 command is
  0.009-0.027 in size; OFF's is 0.53-0.85.
- **D-15, action noise.** The PHASE finals' hip_roll std is 0.052-0.054, against OFF's 0.062-0.064. It fell below 0.08
  earlier (5.1-5.9M vs 7.8-8.1M steps). The entropy schedule is identical in all six runs. This bears on the
  noise-reset choice for a walking pilot [inferred].
- **D-8.** The notebook's robust_best pair passes 40/40 at 9,306,112 / 8,650,752 / 6,815,744 steps (s60 / s42 / s80)
  [measured: each run's training_history.json checkpoint_steps].

## Owed or suggested items not done

These are listed so nothing reads as covered when it is not.
- **M3 six-heading gate at earlier PHASE checkpoints** (NEXT_STEPS suggested 6.3M and 10.5M). Not run: it was not in
  the list you approved. Cost is about 33 CPU-minutes per checkpoint.
- **The critic's neck_pitch slope** (NEXT_STEPS asked to add it). Not measured.
- **The in-run K3 panels of s42 and s80** (k3_panel.csv/json). Not fetched, so the 1M-cadence noise-fall series is
  missing for exactly the two tied runs. M4's K3 at the 6 J labels covers them at lower cadence.
- **The other 34 periodic checkpoints per run** (every 262,144 steps). Not measured.

## How the watch list changes

1. **No mid-training or K1-K4 evidence: closed for K1-K4** (they hold in all three runs). The tail is near zero at all
   12 measured checkpoints. The in-run K3 series for s42 and s80 has not been read.
2. **s60 falls:** confirmed, with neck excursions in every post-200 fall [measured]. The fall-mode reading is
   [inferred]. s60 was not named.
3. **Low action noise:** confirmed lower than OFF's. s42 is the lowest. It remains a pilot design choice.
4. **Head at about 0.90 m:** confirmed at all J labels, with single episodes down to 0.888 m. Report-only under D-D46.
5. **s80's tail_2 creep:** confirmed at the checkpoints (+0.010 → +0.066 from 7.3M to the final). It is the largest
   creeping tail readout. s42's tail_2 and tail_3 also drift, by less.

## Limits

- **Coverage:** three seeds, and screen measurements, not certification. Nothing here is on main.
- **Quality beyond standing:** the statue passes the gate. M4 adds durability and noise tests, but no test here
  measures quality beyond standing.
- **OFF comparisons:** the OFF numbers are STANCE6's records, read through the measured identity of the finals.
- **Fall analysis:** the s60 falls analysis reads ledger window means only.

## Process notes

- **Workflows and cost.** Five workflows: fetch, port, a measurement attempt and its resume, and this report's check.
  Subagent tokens were about 1.96M + 2.81M + 1.86M + 0.29M, plus the part of the first measurement attempt lost to the
  restart. Wall clock was about 4.5 hours (22:04-02:35 UTC).
- **Rule slips** (all recorded in the run log):
  - Device-path references:
    - Two agents ran greps whose patterns were built to match the null or zero device.
    - One verifier's process listing wrote the harness's own command line, which contains a device path, into a file
      in its folder; it deleted the file within a minute.
    - Another verifier's process listing showed such a line on screen only.
    - One verifier searched for the device folder's directory prefix.
  - Polling and waits:
    - One agent briefly ran a sleep-polling Monitor.
    - One agent used six foreground sleeps of 45-100 s.
    - Several agents used timed foreground "tail --pid" waits.
    - One verifier read status files about 50 times while its jobs ran.
  - Command logs: many CMDLOG lines are paraphrases or were logged after the fact.
  - File handling:
    - One builder stored 2 extra files with its own checked helper, inside its own folder.
    - The harness refused one subagent's report-file write; the content came back in its structured result.
- **Integrity checks.** Every fetched file is sha256-equal to the PC F2 manifest. No tool file changed during
  measurement: all 224 match the pre-measurement snapshot.
