# Is the T. rex stance solved?

## Short answer
On behaviour, mostly yes. On the record, not yet.
- **(a) Tail habit: fixed in 3 of 3 runs.** No tail was held at its limit or held high, and no habit moved to another motor, across all 91 evaluations [measured: SCORE 321-330].
- **(b) A stance that stands: yes, cleanly.** But the do-nothing statue passes the same tests, so quality beyond standing still is not shown [measured: RUNLOG 620; MAIN stance.toml 680].
- **(c) Criteria across seeds: every test that ran passed 3 of 3, at the final checkpoint only.** The four extra stance tests (M4: K1-K4), which include a mid-training check, never ran, so no parent is named [measured: SCORE 357, 361].
- **(d) Certified on main: no, 0 of 2 seeds.** These runs used a branch recipe main doesn't have [measured: MAIN stages.toml 34; SCORE 74-75].

## What is true in your read
- **All 3 seeds pass the six-heading gate 40/40.** Every stance-quality bar has 0/40 failures: chatter, hops, width, foot pressure, tilt, slide, yaw and saturation [measured: M3 reports l.37-59].
- **The result held to 12M steps,** past the 1.2-4.5M window where earlier habits began [measured: SCORE 322, 326, 330].
- **Less tilt and drift than the statue,** with a little more foot slide; no motor is near its limit [measured: skeptic gate_margins.out, m1_actuators.out].
- **Hard seeds:** all three had a tail habit before [measured: RUNLOG 613; SUMMARY E1-E12].

## What is not true yet
- **Quality is not proven.** The statue also passes 40/40, and the runs beat its reward by only +8.9 to +17.2 on 4470.4 [measured: M3 reports l.15-20, 61].
- **Six headings are one panel** for this policy: 40 episodes of evidence per run, not 240 [measured: M3 report l.11].
- **Only the final was judged.** Last round, s80 was 40/40 at its final but failed K1 at 6.29M. M4 checks for this; it was left out of the final measurements [measured: REVIEW6 400-402; RUNLOG 967; CRIT 269-272].
- **Three seeds give no rate.** A new seed passing is likely but not established [inferred: CRIT 492].
- **The pre-registered test failed on procedure.** It asks whether the phase critic caused the fix. The controls were opened early, so it reads NOT SEPARATED (p = 20/20), although the numbers separate completely. It is not a stance bar [measured: SCORE 362; DEV 393].
- **Not a usable walking parent today.** The branch refuses a warm start from these checkpoints, and main can't load them [measured: BR train_base.py 606-610, 671-676; code text, not run].

## What I'd watch (ranked)
1. **No mid-training or K1-K4 evidence.** This blocks naming a parent. CPU work, about a day.
2. **s60 falls:** 9 of 910 evaluation episodes ended early, against 0-1 in the other runs. 7 of the 9 ended at steps 211-378, just after the reward terms switch on at step 200 [measured: W/short_eps.out]. It is within gate tolerance: s60 is the weakest candidate, not blocked. s42 looks best [inferred].
3. **Low exploration noise:** action spread 0.057-0.060, below the 0.095 of pilot 1's parent, and pilot 1 stood still. But pilot 1's own noise rose to about 0.11 and it still stood, so a noise reset is a guess [measured: BRP 1319]. Not a blocker; a pilot design choice.
4. **The head sits at 0.90 m,** about 5 cm below the statue's, exactly where the head reward stops paying [measured: W/head_neck.out; BR stance.toml 315-318]. You ruled head posture report-only (D-D46) [measured: RUNLOG 607]. Not a blocker; check it on video.
5. **s80's tail_2 is creeping up,** +0.066 at the final (tail_1 too), still small [measured: f2/measure M1]. Not a blocker, but it argues against s80 as parent.

## Should we move on?
**Yes, toward walking. More GPU time on stance tuning wouldn't fix anything:** the gaps are evidence and plumbing, not a known defect [inferred: NEXT_STEPS §1]. But two open plan rules stand in the way: a two-seed stance retrain "before any trex walk", and "no walker session on today's rewards" [measured: GQP 185, 700, 1090]. So a walk now needs a declared exception, as pilot 1 had (D-D36).

**Shortest path to a first walk:** about 4 days and 15-30 L4-hours [inferred: NEXT_STEPS cost table].
1. **Run the owed M4 tests on the 3 finals,** plus head posture and s60's falls. CPU only: 4-8 CPU-hours, about 1 day. The round's criteria already owe this, and it names the parent your D-D45 waits for.
2. **Add a warm-start rule:** drop the critic's 3 phase inputs, or use a fresh critic; a noise reset is optional. About 1 day on a pilot branch, or 1-2 days on main, off by default. **Needs your approval** (training code).
3. **Fix the pass criteria before training:** a frozen gait probe plus video. Main's walk gate passed both September hoppers [measured: GQP 86, 89].
4. **Train 3 walking runs on Colab L4s:** about 30 L4-hours at 8M steps, or 15 at 4.2M, about 1.5 days. **Needs your approval** (GPU time and the exception).
5. **Run the probe, then your video review:** about 3 CPU-hours.

**What would change this:**
- **M4 fails on all three finals:** stay on stance.
- **You want a certified line first:** certify on main. That takes 30-60 L4-hours and about a week, plus a recipe choice: main's stance has 1 of the 4 step-timed terms [measured: MAIN stance.toml 135-188; RB stance.toml 418-433].
- **You want the phase critic on by default:** more seeds and a no-timing control input, 45-150 L4-hours.

## Decisions for you
1. **Run M4 and the owed CPU checks now?** Recommended: yes.
2. **What bar must the parent meet?** Recommended: K1-K4 pass at the named final. Alternatively, accept the rule's pick, which names a parent whatever K1-K4 read.
3. **Walk from an unmerged branch parent as a declared exception?** Recommended: yes, while landing the warm-start code on main, off by default. Alternatively, certify on main first.
4. **Include the noise reset in the pilot?** Recommended: yes, labelled untested.
