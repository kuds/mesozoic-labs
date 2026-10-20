# Night continuations: what they show, read by spec 7.2 and 7.3 (report only)

Written 2026-10-19, about 08:00 UTC. This is a report only:
- nothing on Drive or GitHub changed, and nothing was pushed;
- nothing ran in /home/user/mesozoic-labs;
- no simulation ran for this file. The one new computation is a CSV count of the Colab training falls (section 4.3).

"Measured" means read from the files listed at the end, or computed by the script named there. "Inferred" is marked
wherever it applies. Paths: S3 = scratchpad/foot/stance3,
C = S3/night/cont, spec = S3/night/design/spec.md.

## 0. Bottom line

| hypothesis | verdict | the deciding numbers |
|---|---|---|
| **H_A** (Run 3 + stagger penalty closes the split by teaching the restoring twist) | **INCONCLUSIVE** | 5 of the 6 rows with bars land on the supports side. The mechanism row (hip slope +0.52) is on the refute side, at the bar. The no-change Colab control at E is on the supports side of the same 5 rows, with **no** term. |
| **H_B** (Run 3 + ankle anchor; bringing the ankles back is enough for K2a) | **REFUTES** | Ankle +0.13 deg and common CoP -0.039 both pass their bars, but K2a went 6 -> **10**. That is the pre-registered refute line exactly. The control went 6 -> 0. |
| **H_C** (the stagger penalty also closes Run 1's stagger) | **REFUTES** | Window stagger sd 29.3 -> **22.6** mm (refute bar: 20 or more). The control: 24.2 mm. |

Safety flags:
- **A:** none.
- **B:** none. Clean fell 34 -> 30, but only for reasons it already had at the start.
- **C:** five flags. Four of them, plus an all-feet-support failure, also appear in the Run 1 control at E, on the same two
  fall seeds. So they are not the term's (inferred). Only C's two yaw-change failures, both inside its falls, are C's own.

**Spec 7.3 reading: 1, its "C refutes" branch.** The full argument is in section 5.
- Its stated condition is met: A is on the supports side, and B's stagger is 18.1 mm, at least 15. C refutes, so the
  branch says "the closure is not general: propose a from-scratch screen with the term rather than a commit".
- Its headline claim, "the stagger term is the active ingredient for the split", is **not** established. The matched
  no-change control at E did everything A did, without the term.
- That is reading 4 in substance: A and the control improved alike. Reading 4's letter fails only because B got worse.
- Spec section 6's own follow-up still stands: a local no-change continuation before anything goes to Colab.

**In plain words, for the next Colab screen:**
- **Drop the ankle anchor.** It did exactly what it was built to do, and the stance got worse.
- **Do not adopt the stagger penalty as a fix.** Over the same steps, the unchanged recipe fixed the same failure by
  itself. At most, test the term as one arm, trained from scratch, beside an unchanged L+T arm on the same seeds.
- **Keep the guards.** Add a saturation alarm to the ledger read.
- **Leave mirrored spawn noise and prey-ahead out of the training task.** Section 7 gives the reasons.

## 1. What this rests on (validity)

**The numbers used here, and their checks:**
- **R1 judge files for A, B and C at E** (C/judged/<run>/judge_9961472.json):
  - They are relaxed-binding, judged in the exp_night clone at 80328abf. The build's equivalence check covered only
    2 seeds.
  - Their R2 rollouts in the **screen** clones reproduce them exactly on all 40 seeds: lengths, terminations, per-foot
    foot_cop_fore_aft and the pad metric (|diff| 0), and the K2a and K2b flags (C/r2/r2_night.txt, top).
  - So the K numbers are the screen judge's.
- **Control judge files at E** (S3/judged/<RUN>/judge_9961472.json):
  - These are fully bound, judged in the screen clones.
  - Their R2 rollouts reproduce them on all 40 seeds, and the model sha256 matches (C/control/r2_control.txt, top).
- **Start values.** The start rollouts (R3 and R1 at 8,388,608) also reproduce their judge files, and they equal the
  spec's numbers.
- **Every rollout number** cited below comes from one of these reproduced rollouts.

**Endpoint and lineage:**
- **The endpoint.** E = 9,961,472 is the spec's pre-registered fallback (section 4.3). The 06:41 restart killed the arms
  at 10.38-10.40M. The Colab control was fetched and judged **at E itself**, so there is no step offset (spec section 6,
  item 4, is met).
- **The restarts.** All three sessions.json files record two starts:
  - 04:30 from the Colab 8,388,608 pair;
  - 05:31:10 from the run's own 8,912,896 checkpoint, with seeds 52 / 52 / 60.
- **What each E checkpoint contains.** 0.52M steps of session 1, then 1.05M steps of session 2. Session 1's abandoned
  stretch (8.92-9.23M) is not in its lineage and is not in R4.
- **The A/B pairing holds.** Both resumed together with the same seed. One sign of matched evaluation streams
  (inferred): both have their only eval fall at the same evaluation, 9,306,112.

## 2. Spec 7.2, row by row

In the tables, "control @E" is the Colab no-change continuation of the same start checkpoint, judged at E. Its row
verdict is a reading of the arm's bars, not a test of a hypothesis.

### 2.1 Run A, `night_R3_stagger`: H_A

| readout | supports | refutes | start (8.39M) | **A @E** | row | control @E | control's side |
|---|---|---|---|---|---|---|---|
| window stagger sd (mm) | <= 10 | >= 15 | 20.9 | **7.5** | supports | 9.7 | supports |
| six-seed mean \|s\| (mm) | <= 20 | >= 35 | 44.0 | **13.3** | supports | 10.0 | supports |
| stagger sd i200 -> i999 (mm) | i999 < i200 | i999 > i200 | 20.1 -> 24.2 | **10.7 -> 5.6** | supports | 19.3 -> 6.1 | supports |
| r-l hip servo-error slope | <= 0 | >= +0.5 | +0.62 | **+0.52** | **refute side** (by 0.02) | +0.21 | between the bars |
| CoP split sd | <= 0.60 | >= 0.68 | 0.685 | **0.471** | supports | 0.490 | supports |
| K2a / K2b / roll | <= 3 / <= 3 / <= 1 | K2a >= 6 with stagger sd <= 10 | 6 / 3 / 3 | **0 / 0 / 0** | supports | 0 / 0 / 0 | supports |
| ankle offset; common CoP (expected unchanged) | no bar | no bar | +2.00 deg; -0.253 | **+1.02; -0.133** | expectation **failed**: both moved about halfway to 6.29M (+1.08; -0.113) | +1.17; -0.179 | moved too |

**Safety (spec 7.2): none.**

| flag | A |
|---|---|
| clean | 40 (start 34; verdict PASS, LCB 0.928) |
| hop-or-fall / panel falls | 0 / 0 |
| noise falls, joint / own | 0/20 / 0/20 |
| touchdown, support, settle-touchdown and yaw-change failures | 0 |
| saturation over the 0.10 bar | 0 episodes; worst share 0.000 |
| roll rail | 3 -> 0 |
| x_c vs statue | -13.7 mm |
| reward ratio | 0.983 |
| hop-watch or collapse stop | none |

**R4 value mismatch:** no flag. The minimum EV after 8.91M is 0.939, against 0.967 at the first update.

**Verdict on H_A: INCONCLUSIVE.**
- **The outcome H_A predicts happened.** Five rows land on the supports side and no safety flag fired.
- **The mechanism it names did not.**
  - H_A says the policy "learns the restoring twist it lacks". The row that measures this, the hip slope, sits on the
    refute side at +0.52.
  - The control's slope is lower, at +0.21.
  - Inferred: the hip-slope row may not be a necessary route at all. Run 3 at 6.29M had a small stagger (7.2 mm) with
    a slope of +0.74 (diag_split notes).
- **The harness-matched ankle check failed.** A moved the ankle level, which the spec said it would leave alone.
- **The same outcome happened without the term.** At the same step, from the same checkpoint, the control is on the
  supports side of all five of those rows.
- **Spec section 6 says what this means.** One control against one arm reads only effects larger than the window's
  natural swing. None of A's five supports rows beats the control by more than 2.2 mm or 0.019 in split sd. On the
  six-seed row the control is lower, 10.0 against 13.3.

### 2.2 Run B, `night_R3_anchor`: H_B

| readout | supports | refutes | start | **B @E** | row | control @E |
|---|---|---|---|---|---|---|
| ankle offset, mean (deg) | <= +1.1 | >= +1.8 | +2.00 | **+0.13** (r +0.16, l +0.10) | supports | +1.17 (between) |
| common CoP | >= -0.15 | <= -0.22 | -0.253 | **-0.039** | supports | -0.179 (between) |
| x_c (expected 7-10 mm forward; no bar) | | | -21.2 mm vs statue | **-5.4** (+15.8 forward) | about twice the inferred move | -17.0 (+4.2) |
| K2a | <= 3 | ankle <= +1.1 **and** common >= -0.15 **and** K2a >= 6 | 6 | **10** | **REFUTES**: all three parts hold | 0 |
| window stagger sd (mm) | >= 15 (consistent) | <= 10 (A's attribution fails) | 20.9 | **18.1** | consistent | 9.7 |

Also measured:

| quantity | start | B @E | control @E |
|---|---|---|---|
| K2b | 3 | 13 | 0 |
| clean (LCB) | 34 | 30 (0.613, FAIL) | 40 |
| CoP split sd | 0.685 | 1.029 | 0.490 |
| r-l hip slope | +0.62 | +1.09 | +0.21 |
| corr(yaw change, bearing) | +0.880 | +0.778 | +0.705 |

**Safety: none by the letter.**
- Clean fell 34 -> 30, but the unclean reasons at E are pad (10) and corner lift (1), and both were already present at
  the start. So this is not a "new reason".
- Falls 0, noise 0/0, saturation 0 (maximum window |a| 0.95, on tail_2_pitch), roll 3 -> 1, ratio 0.985.
- The anchor's named side door does not show: knee / hip-pitch offsets went -0.44 / -0.64 -> +0.28 / -0.21 deg.
- R4: no flag. The minimum EV is 0.926, against 0.966 at the first update.

**Verdict on H_B: REFUTES, by its own pre-registered line.**
- **What it claimed.** H_B is a sufficiency claim: remove the common shift, and the recombination arithmetic takes K2a
  to 3 or fewer.
- **What happened.** B removed the common shift more completely than predicted: ankle +0.13 against a +1.1 target. K2a
  rose to 10 and K2b to 13.
- **Against the control,** which went 6 -> 0 from the same checkpoint, the anchor is associated with 10 more K2a
  episodes (n = 1).
- **A possible cause (inferred).** The split grew: sd 1.029, and the hip slope became more anti-restoring at +1.09. One
  explanation: with the feet staggered, flat soles force the two ankle angles apart (diag_split item 3). A term that
  pulls both ankles to one settled angle then loads the forward foot's heel harder.

### 2.3 Run C, `night_R1_stagger`: H_C

| readout | supports | refutes | start (8.39M) | **C @E** | row | control @E |
|---|---|---|---|---|---|---|
| window stagger sd (mm) | <= 15 | >= 20 | 29.3 (exact sites, re-measured) | **22.6** | **REFUTES** | 24.2 (refute side) |
| mean \|ankle L-R\| (deg) | (falls with the stagger, inferred; no bar) | | 2.44 | **1.52** | fell | 1.87 |
| K2a, K2b | <= 3 each | | 0, 0 | **2, 2** (3046, 3063: the two panel falls) | within the bar, but both are falls (see safety) | 2, 2 (the same seeds, also falls) |
| K3 noise falls, joint / own | exploratory, no prediction | | 7/20, 0/20 | **10/20, 1/20** | | 14/20, 2/20 |

**Safety: five flags.** Spec 7.2 says any flag overrides "supports". C already refutes, so the flags change no verdict.
The control column shows whether each flag is the term's:

| flag | C @E | Run 1 control @E (no term) |
|---|---|---|
| clean below start for a NEW reason | YES: 36 < 38; falls, support, yaw, displacement, load share | YES: 35 < 38; falls, support, displacement, load share, sole contacts |
| hop-or-fall >= 2 | YES (2) | YES (2) |
| any panel fall | YES: 3046 at 342, 3063 at 337 | YES: 3046 at 338, 3063 at 290 |
| touchdown, support, settle-touchdown or yaw-change failure | YES: support 1, yaw change 2 (47.4 and 52.0 deg), all inside the two falls | support 1 (3063, in its fall); yaw change 0 |
| reward ratio < 0.95 | YES (0.946) | YES (0.940) |
| saturation, roll up by 2 or more, x_c window, hop-watch or collapse stop | no | no |

- **Shared with the control.** Four of C's five flags, and its support failure, appear in the no-change control at the
  same step, on the same seeds. Inferred: they belong to Run 1's lineage around 9.96M. The control was back to PASS
  (39/40, 0 falls) at 10,485,760.
- **C's own:**
  - the two yaw-change failures. Both are in fall episodes; the full-horizon episodes' maximum yaw change is 8.55 deg.
    Inferred: they are a consequence of falling, not the turning side door;
  - the training-fall pattern in section 4.3.
- **R4:** no flag. The minimum EV is 0.948, against 0.935 at the first update.

**Verdict on H_C: REFUTES.** The stagger fell 29.3 -> 22.6 mm. That is inside Run 1's natural band and 1.6 mm from the
no-change control's 24.2. On arm L, seed 60, the term did not close the stagger beyond what the unchanged recipe did.

## 3. What the matched Colab control at E shows

| | Run 3 (stance3_armLT_s52), no change | Run 1 (stance3_armL_s60), no change |
|---|---|---|
| verdict, clean | 8.39M FAIL 34 -> **E PASS 40** -> 10.49M PASS 40 -> 12M final 39 (K2b 4: K2 fails again at the final) | 8.39M PASS 38 -> **E FAIL 35 (2 falls)** -> 10.49M PASS 39 -> 12M final 38 (1 fall; K2 range and K3 fail) |
| K2a / K2b | 6/3 -> **0/0** | 0/0 -> **2/2** (both falls) |
| window stagger sd | 20.9 -> **9.7** mm | 29.3 -> **24.2** mm |
| ankle offset; common CoP | +2.00; -0.253 -> **+1.17; -0.179** | +3.12; -0.349 -> **+0.72; -0.062** |
| r-l hip slope | +0.62 -> **+0.21** | +0.06 -> **+0.11** |
| ledger stagger sd per evaluation, 8.39M -> E | **10.2 to 51.9 mm**; blocks 24.0 / 35.8 / 15.9 | 17.9 to 31.1 mm; blocks 23.5 / 25.2 / 21.9 |

1. **Run 3's 8.39M K2a failure was transient under the unchanged recipe.**
   - By E it had cleared every symptom the night design targeted: stagger, split, its growth in the window, the K2a
     heel split, and about 60% of the ankle opening.
   - It relapsed by 12M: final K2b 4, and a K2c range of 30.6 mm against the 30 mm bar.
2. **Run 1's control was itself in a transient failure at E,** with the same two falls as C. It was clean again 0.52M
   steps later.
3. **Single checkpoints swing a lot within 0.5-2M steps** under an unchanged recipe:
   - Run 3: K2a 0 at 6.29M -> 6 at 8.39M -> 0 at 9.96M -> K2b 4 at 12M;
   - Run 1: PASS -> FAIL -> PASS across 8.39, 9.96 and 10.49M;
   - Run 2: a saturation episode, with 28 of 40 panel episodes over the bar at E and 0 at 10.49M. Its ledger shows 8-10
     of 10 eval episodes saturated from 9.31M to 10.09M.
   - The night arms are judged at **one** checkpoint each. Effects smaller than these swings cannot be read.
4. **The spec's attribution line for A fails, on the control's numbers.**
   - The spec wrote "<= 10 mm: the stagger closed without its term, so A's attribution fails" as B's line.
   - The no-change control at E meets that condition exactly: 9.7 mm with no term.
   - So the A-vs-B stagger gap (7.5 against 18.1) cannot be credited to A's term. With the control at 9.7, B's 18.1 more
     plausibly reflects the anchor **holding the stagger open** (inferred): B's split sd and hip slope both moved the
     wrong way relative to the control.

## 4. Specific to the term, or also in the control? (reading 4)

### 4.1 Pre-registered rows

| arm | readout | arm @E | control @E | specific to the term? |
|---|---|---|---|---|
| A | stagger sd; six-seed \|s\|; closure; split sd; K2a/K2b/roll | 7.5; 13.3; 10.7 -> 5.6; 0.471; 0/0/0 | 9.7; 10.0; 19.3 -> 6.1; 0.490; 0/0/0 | **No.** Same side of every bar, and within 2.2 mm / 0.02. The control is lower on the six-seed row. |
| A | hip slope | +0.52 | +0.21 | No. The control is nearer the supports bar. |
| A | ankle; common | +1.02; -0.133 | +1.17; -0.179 | No. Both moved, by about the same amount. |
| B | ankle; common; x_c | +0.13; -0.039; -5.4 mm | +1.17; -0.179; -17.0 mm | **Yes**, in the targeted direction, and well beyond the control. |
| B | K2a; K2b; clean; split sd | 10; 13; 30; 1.029 | 0; 0; 40; 0.490 | **Yes, as harm** (n = 1). |
| C | stagger sd | 22.6 | 24.2 | No (1.6 mm). |
| C | \|ankle L-R\| | 1.52 | 1.87 | Not shown: 0.35 deg, with no bar and n = 1. |
| C | safety flags | 5 | 4 of the same, plus support | No, except C's 2 yaw failures inside its falls. |

### 4.2 Readouts where A differs from both the control and B

These are post hoc observations, not pre-registered, with n = 1:

| readout | A @E | control @E | B @E | for reference |
|---|---|---|---|---|
| corr(yaw change i200 -> i999, prey bearing) | **+0.154** | +0.705 | +0.778 | start +0.880; 6.29M +0.04 |
| mean \|yaw creep\| in the window (deg) | 0.92 | 2.00 | 1.32 | start 1.56 |
| stagger sd at step 200 (mm) | **10.7** | 19.3 | 17.7 | start 20.1; 6.29M 11.7 |
| ledger stagger sd per evaluation after 8.91M (mm) | 5.6-10.9 | 10.2-51.9 | 8.9-20.5 | |

- **What A seems to have learned (inferred).** A stopped turning toward the prey during the window, and its stagger was
  already smaller when the window opened. The term is paid only after step 200, so the second point would be the policy
  pricing the stagger in advance.
- **Why these are weak evidence:**
  - None of these rows has a bar.
  - Both of A's values lie inside what the same lineage showed at 6.29M with no term.
  - The ledger comparison with Colab mixes harnesses and eval seeds. The A-vs-B ledger comparison does not, but B is
    not neutral on the stagger (section 3, item 4).
  - They are the best case for the term. They do not make it the cause of A's recovery.

### 4.3 Training falls, Run 1: C against the Colab control

This is new here, from a CSV count with the night R4 code's own definition: a fall is cause != "truncated".

| | the same six 262,144-step bins, 8.39M -> E | falls (fraction) | last 4 bins (9.04M on) | causes |
|---|---|---|---|---|
| C | 1,588 episodes | 29 (0.0183) | 28 of 1,064 | tail_contact 12, fallen 7, head_contact 6, nosedive 4 |
| Colab Run 1, no change | 1,587 episodes | 22 (0.0139) | 16 of 1,058 | fallen 13, nosedive 6, tail_contact 3 |

- **The counts are close.** The one-sided conditional binomial p is 0.20 over all six bins.
- **The last four bins differ more:** p 0.048. But that window is chosen after the fact, and the harnesses differ.
- **The causes differ in kind.** Tail or head contact is 18 of C's 29 falls, against 3 of the control's 22. Inferred: a
  posture change under the term on arm L. C's knee / hip-pitch offsets are -2.51 / +2.33 deg, against the control's
  -1.03 / +1.62.
- **This is not a pre-registered readout,** and it sets no flag.
- **For comparison, the L+T runs over the same bins:** A 1 fall, B 4, Colab Run 3 2.

**Reading 4's answer:**
- **A:** no improvement on any pre-registered row is specific to the term. The no-change control improved alike.
- **B:** its changes are specific to the term: the targeted ones in the right direction, and the K2 outcome in the
  wrong one.
- **C:** nothing on its pre-registered rows is specific to the term.

## 5. Spec 7.3: which reading applies

| reading | condition | met? |
|---|---|---|
| 1 | A supports, and B leaves the stagger at 15 mm or more | **Yes, by the letter.** B 18.1 mm. A is on the supports side of 5 of 6 barred rows; the sixth is at the bar (+0.52 against +0.5); no safety flag. C refutes, so the branch is: "**the closure is not general: propose a from-scratch screen with the term rather than a commit**". |
| 2 | B supports (ankle and common back, K2a <= 3), and the stagger stays | No: K2a 10. |
| 3 | A and B both reach K2a <= 3 | No: B 10. |
| 4 | A, B and the Colab control all improve alike | **Letter: no**, B did not improve. **Substance: yes for A**: A and the control improved alike on every A row (section 4). |
| 5 | No arm moves its own mechanism readout | No. A moved the stagger, B the ankle and common, and C the stagger part way (29.3 -> 22.6). |
| 6 | A mechanism met through a side door (window touchdowns, yaw past 25 deg, roll up) | No. A and B have no side-door failure (roll 3 -> 0 and 3 -> 1). C did not meet its mechanism, and its yaw failures are inside falls that the control shares. |

**The reading: reading 1, through its C-refutes branch, with reading 4's no-attribution caveat on A.**
- **The letter.** Reading 1 is the only reading whose condition holds. Its branch for "C refutes" is the weaker of its
  two: a from-scratch screen with the term, never a commit.
- **What it does not establish.** Its premise, "the stagger term is the active ingredient for the split", rests on B
  being a neutral comparator for the stagger. The control at E shows the stagger closes without any term (9.7 mm), so B
  is not neutral (section 3, item 4). Reading 1 therefore licenses at most its weaker branch, and the attribution is
  open.
- **The precondition.** Spec section 6 fixes one step for any arm that "wins": a local no-change continuation of the
  same checkpoint, with the same launcher and no override, "before anything is proposed for Colab". Tonight's evidence
  makes it the one experiment that could still credit the term. If a local no-change run also reaches about 10 mm and
  K2a 0 by E, A's outcome has no residue the term can claim.
- **Its cost (inferred):** one process to 9,961,472 is about 65-80 minutes, at the measured 334-410 steps/s.

## 6. n = 1 and the restarts

**n = 1:**
- one run per hypothesis;
- A and B share one start checkpoint and seed (52);
- C is the only other seed, and it is on the other arm;
- one Colab control per checkpoint;
- a K2a count is binomial on 40 seeds.

**What n = 1 can and cannot say:**
- It can read effects larger than the natural swing of a single checkpoint. Section 3 shows that swing is large:
  - K2a 0 <-> 6;
  - PASS <-> FAIL within 0.5M steps;
  - ledger stagger 10 to 52 mm inside this one window.
- **B's ankle and common moves exceed that swing** (+0.13 against +1.17). So does its K2a gap against the control
  (10 against 0), but the same lineage went 0 -> 6 unchanged, so that gap is strong rather than decisive.
- **B's refutation does not depend on n.** H_B claims that the common fix is enough. One clear counterexample, with the
  mechanism fully met (ankle +0.13) and K2a still 10, is evidence against that claim. A single success would have been
  weak evidence for it.
- **A's verdict is limited by the control, not by n.** Even with more seeds, a term-free recovery at the same step says
  the outcome needs no term on this checkpoint.

**The restarts:**
- **First restart (05:08).** It added a second resume boundary at 8,912,896 to all three arms. Each resume re-seeds the
  env streams, collects a fresh rollout and rebuilds the eval history. The resume path dropped the 2 evaluations
  recorded after 8.91M.
  - The weights come from the checkpoint. Inferred: the optimizer state does too; the Colab zips carry
    policy.optimizer.pth.
  - **A against B:** both resumed together, same seed, same checkpoint step. So this is not an A-vs-B difference.
  - **The arms against Colab:** it is one more harness difference. The Colab run has no resume in this window.
- **Second restart (06:41).** It only moved E to the pre-registered fallback. The arms got the minimum budget,
  1,572,864 steps. The 10,485,760 point, where all three Colab controls passed, is not available for the arms.
- **Inferred: neither restart changes a verdict.** H_B and H_C are refuted on their own rows, and H_A's verdict turns on
  the control, which no restart touched.

## 7. Recommendation for the next Colab screen

This is written for the project maintainer, whose goal is a T. rex stance that holds: across seeds, across
checkpoints, and under noise. Every item says what the evidence does and does not support.

### Terms

1. **Do not use the ankle anchor** (w 0.4, tol 0.10 rad).
   - It did exactly what it was designed to do: ankles back to settled (+0.13 deg) and the backward CoP shift gone
     (-0.039).
   - Heel-split failures still rose from 6 to 10 of 40 (and K2b from 3 to 13), while the same checkpoint with no
     change went to 0.
   - So the "fix the common ankle opening" idea is refuted, not just under-dosed.
   - Inferred, not tested: this also weakens the v6b passive ankle spring as a K2 fix, since it acts on the same common
     ankle opening (diag_split implication 6).
2. **Do not adopt the stagger penalty as the fix, and do not commit it as a default.**
   - Over the same steps, the unchanged recipe closed Run 3's stagger and cleared its heel split by itself.
   - On Run 1 the term did no better than no change (22.6 against 24.2 mm).
   - The evidence supports **at most** one thing, which is reading 1's own branch: a from-scratch test arm, L+T plus
     stagger (w 0.30, tol 45 mm), beside an **unchanged L+T arm on the same seeds**, with at least 2 seeds per arm.
     - **In its favour:** on L+T it showed no harm. No safety flag, 40/40 PASS, noise 0/0, reward ratio 0.983, no
       saturation, and 1 training fall. It also showed the post-hoc hints in section 4.2, chiefly that it stopped
       turning toward the prey.
     - **First:** run the spec's local no-change continuation (section 5). If it also recovers, there is no case for
       spending a Colab arm on the term.
3. **Use L+T as the base, not arm L.**
   - Arm L's Run 1 fails joint-noise robustness at every judged point: 7/20 at 8.39M, 14/20 at E (control), 10/20 with
     the term, and 11/20 at 12M. It also swings 50-65 mm (diag_drift).
   - The stagger term did not change that on arm L, and C's training falls shifted toward tail and head contact.
   - L+T kept noise falls at 0-2/20 in every L+T judgement used here.

### Guards

- **Keep every side-door check as a pre-registered flag:**
  - touchdown rate, all-feet support, settle touchdowns and episode yaw change;
  - the roll rail, saturation, reward ratio and the x_c window;
  - the hop watch and the collapse stop.
  - Tonight none fired on L+T with either term. No reading-6 guard is called for.
- **Add a saturation alarm to the eval-ledger read.**
  - Run 2 spent about 1M steps (9.31-10.09M) with 8-10 of 10 eval episodes saturated. The judge at E failed it 28/40
    on saturation alone.
  - The ledger showed it every 131,072 steps, and nothing watched for it.
  - The signal already exists: eval_episodes.csv's screen_v2_reasons column lists max_actuator_saturation_fraction.
    Only the alarm is new.
- **For a stagger arm, also watch:**
  - episode yaw change and the training-fall causes (tail and head contact);
  - the stagger at step 200 against the window, so that paying the stagger in advance stays visible.
- **Judge more than one checkpoint, as K1-K4 already does, and put an unchanged arm in the same screen.**
  - Tonight's main lesson is that a single checkpoint can swing PASS <-> FAIL in 0.5M steps.
  - A screen without its own no-change arm cannot credit any term.

### Task fixes: mirrored spawn noise and prey straight ahead

**They do not belong in the next round's training task. The evidence:**
- **They were never trained.** They were only counterfactual rollouts of one fixed policy (Run 3 at 8.39M): K2a 6 -> 3
  with mirrored noise, 6 -> 4 with the prey ahead, and 6 -> 1 with both (diag_split, measured). Even with both, that
  policy still built a 15 mm stagger sd.
- **The unchanged task recovered without them.** Run 3's control reached K2a 0 and a 9.7 mm stagger at E. It even kept
  turning toward the prey (corr +0.705) while closing the stagger to 6.1 mm by step 999. Inferred: the prey offset is
  not what has to be removed.
- **They change the test.** They change the reset distribution and the prey range, and so the task digest. The judge
  would then roll an easier task, the K numbers would no longer compare with this screen, and the policy would never
  practise the asymmetric spawns and off-axis prey that a robust stance has to handle (inferred).

**What the evidence does support:** keep them as **diagnostic panels** in the readouts, rolled beside the standard
panel. That separates "the policy cannot hold a stance" from "the spawn or prey asymmetry pushed it over", without
changing what is trained or judged.

## Files

**Inputs (read):**
- scratchpad/foot/stance3/night/design/spec.md (sections 2, 6, 7.1-7.3, 8)
- scratchpad/foot/stance3/night/cont/readouts_night.md
- scratchpad/foot/stance3/night/cont/readouts_control.md
- scratchpad/foot/stance3/night/cont/r2/r2_night.txt
- scratchpad/foot/stance3/night/cont/r2/r2_readouts.json
- scratchpad/foot/stance3/night/cont/control/r2_control.txt
- scratchpad/foot/stance3/night/cont/control/r2_control.json
- scratchpad/foot/stance3/night/cont/r1_summary.txt
- scratchpad/foot/stance3/night/cont/control/r1_control.txt
- scratchpad/foot/stance3/night/cont/r3/ledger_night.txt
- scratchpad/foot/stance3/night/cont/r4/train_side.txt
- scratchpad/foot/stance3/night/runs/relaunch.log
- scratchpad/foot/stance3/night/runs/{night_R3_stagger,night_R3_anchor,night_R1_stagger}/sessions.json
- scratchpad/foot/stance3/night/diag_split/notes.md
- scratchpad/foot/stance3/night/diag_drift/notes.md
- scratchpad/foot/stance3/live/run_log.txt
- scratchpad/foot/stance3/scores/score_final.txt

**New here:**
- scratchpad/foot/stance3/night/cont/interp/colab_train_falls.py
  -> scratchpad/foot/stance3/night/cont/interp/colab_train_falls.txt
  (reads S3/runs/<RUN>/01_stance/train_episodes.csv; it counts like C/r4/train_side.py; no simulation)
- This file.
