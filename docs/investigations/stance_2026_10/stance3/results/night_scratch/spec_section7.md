## 7. Pre-registered readouts and expectations (report only; never in a verdict)

### 7.1 Readouts, at E for each run (and the control at E)

- **R1, full judge.** S3/judge/judge_stance3.py, full mode (40-episode panel, statue and both noise tests), run in the
  SCREEN clone of the arm (L+T: screen_s3_LT @ 1af2d4d7; arm L: screen_s3_L @ 7e6b557d).
  - Flags: `--task-binding report --allow-commit-mismatch`, with seed 52 or 60. The checkpoint records the candidate
    task and the candidate commit.
  - Dynamics and observations are identical (G3), and the term-off env is the screen's own task, so the K metrics are
    the screen's.
  - These files are relaxed-binding. score_stance3.py refuses them by design, so they never enter K1-K4.
  - Output: S3/night/cont/judged/<RUN>/judge_<E>.json.
- **R2, split readout.** S3/night/diag_split/roll_split.py on the same checkpoint, in the same screen clone.
  - It must first reproduce R1's per-seed lengths and terminations and each foot's foot_cop_fore_aft exactly; if not,
    its numbers are not used.
  - It gives:
    - window stagger sd, |stagger| of the six former K2a seeds (3050, 3051, 3052, 3071, 3073, 3081), and stagger at
      index 200 and 999;
    - CoP split sd, common CoP, and the heel-foot minimum CoP;
    - ankle offset from settled (mean) and |ankle L-R|;
    - the r-l hip servo-error slope against the r-l hip angle;
    - corr(yaw change, prey bearing);
    - foot slide and episode yaw change.
  - Also run on: Run 1's start checkpoint (an exact-site baseline for C; its 29.3 mm is FK), and the Colab control
    checkpoints at E.
- **R3, ledgers (no simulation).** D/scripts/ledger_stagger.py extended to S3/night/cont/runs/<RUN>/01_stance and to
  the refreshed Colab ledgers: per-evaluation stagger, ankle offset, CoP split, COM, and clean / pad over-bar /
  K2b-outside counts, from 8.39M to E.
- **R4, training side (TensorBoard).** train/explained_variance, value_loss, approx_kl, clip_fraction, std,
  mirror_kl_loss and learning_rate. Also hop_watch.json if written, and the late-fall fraction per 262,144-step bin
  (train_episodes.csv).
- **R5, saturation.** Per episode, the worst actuator's share of window steps at |a| >= 0.99, from R1. At the start
  checkpoints, no episode fails the 0.10 bar (Run 3 and Run 1 at 8.39M).
- **R6, gate-reason tally from R1.** Start values:
  - Run 3: pad 6, sole tilt 2, corner lift 3; no touchdown, support, yaw-change or saturation failure.
  - Run 1: sole tilt 2, corner lift 1.

### 7.2 Expectations, with the lines that support or refute

Start values are the judged 8,388,608 checkpoint.

**Run A, R3 + stagger** (start: clean 34/40, K2a 6, K2b 3, roll 3, x_c -21.2 mm, falls 0, noise 0/0; stagger sd 20.9,
six-seed |s| 44.0 mm; split sd 0.685; common -0.253; ankle +2.00 deg; |L-R| 0.99 deg; heel-foot min -0.974; hip slope
+0.62):

| readout | expectation | supports H_A | refutes H_A |
|---|---|---|---|
| window stagger sd | falls toward the 6.29M level | <= 10 mm | >= 15 mm |
| six-seed mean \|s\| | | <= 20 mm | >= 35 mm |
| stagger i999 vs i200 | closes in the window (start: grows 20.1 -> 24.2) | i999 < i200 | i999 > i200 |
| r-l hip servo slope | turns restoring | <= 0 | >= +0.5 |
| CoP split sd | toward 0.585 (6.29M) | <= 0.60 | >= 0.68 |
| K2a / K2b / roll | 2 by the arithmetic (6.29M split + 8.39M common) | K2a <= 3, K2b <= 3, roll <= 1 | K2a >= 6 with stagger sd <= 10 (closing the split was not enough: the common shift dominates) |
| ankle offset, common CoP | unchanged: not this term's target | | |

**Run B, R3 + anchor** (same start):

| readout | expectation | supports H_B | refutes H_B |
|---|---|---|---|
| ankle offset (mean) | toward +1.1 deg | <= +1.1 deg | >= +1.8 deg |
| common CoP | toward -0.11 | >= -0.15 | <= -0.22 |
| x_c | forward by 7-10 mm (inferred: -7.4 to -10.6 mm per ankle degree) | | |
| K2a | 0 by the arithmetic (6.29M common + 8.39M split) | <= 3 | ankle <= +1.1 and common >= -0.15 but K2a >= 6 (the arithmetic does not hold) |
| window stagger sd | stays: the anchor is blind to it (AUC 0.495) | >= 15 mm (consistent) | <= 10 mm: the stagger closed without its term, so A's attribution fails; read A and B as harness or natural |

**Run C, R1 + stagger** (start: clean 38/40, K2a 0, K2b 0, roll 2, x_c -28.0 mm, falls 0, noise 7/0; stagger sd 29.3
mm FK, to be re-measured exactly; ankle +3.12 deg; |L-R| 2.44 deg):

| readout | expectation | supports H_C | refutes H_C |
|---|---|---|---|
| window stagger sd | halves | <= 15 mm (below R1's natural band of 16-32 mm) | >= 20 mm (inside it) |
| mean \|ankle L-R\| | falls with the stagger (inferred) | | |
| K2a, K2b | stay clean | <= 3 each | |
| K3 noise falls (7/20 at start) | no prediction: exploratory | | |

**Safety, every run.** Any one of these is a side-door or harm flag, and overrides "supports":
- clean below its start value for a NEW reason;
- hop-or-fall >= 2;
- any panel fall (start 0);
- joint-noise falls >= 3/20 or own-sigma >= 2/20 (A and B; start 0/0);
- any episode failing max_touchdown_rate, min_all_feet_support, max_settle_touchdowns or max_episode_yaw_change_deg
  (start 0 in all). This is the stepping or turning side door;
- saturation >= 2 episodes over the 0.10 bar (start 0);
- the roll rail up by >= 2 (start 3 / 2);
- x_c outside [-60, +30] mm of the statue;
- reward ratio to statue below 0.95 (start 0.979 / 0.989);
- a hop-watch stop or a collapse stop.

**Value mismatch (R4).** Inferred: the critic refits within about 0.26M steps.
- Flag if explained_variance is still more than 0.2 below the run's own first-update value after 524,288 steps.
- A and B start identical, so their difference is the term's effect on the value fit.

### 7.3 Reading the three together (fixed now)

1. **A supports, B leaves the stagger at >= 15 mm:** the stagger term is the active ingredient for the split. If C
   also supports, propose L+T + stagger for the next 12M Colab screen on 2 seeds. If C refutes, the closure is not
   general: propose a from-scratch screen with the term rather than a commit.
2. **B supports (ankle and common back, K2a <= 3) and the stagger stays:** the common shift is enough for this
   checkpoint, but Run 2's 4.2M split (common +0.03) stays uncovered. Propose both terms for the next round.
3. **A and B both reach K2a <= 3, each through its own mechanism:** the two-part model holds. Propose both terms.
4. **A, B and the Colab control all improve alike** (K2a <= 3 and the stagger closing in all): natural or harness
   recovery, with no attribution. Run a local no-change continuation first.
5. **No arm moves its own mechanism readout:**
   - About 2M steps at lr 1.6e-5 -> 1.25e-5 with ent_coef 0 is too short or too weak to unlearn the habit.
   - Exploration noise is low: the own-sigma mean is 0.08 at the start.
   - This is not evidence against the terms. Next is a from-scratch screen.
6. **A mechanism met through a side door** (window touchdowns, yaw past 25 deg, roll up): the term needs a guard before
   any further use.
