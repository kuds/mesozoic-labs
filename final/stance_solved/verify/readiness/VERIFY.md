# VERIFY: readiness note (short)

**Verdict: mostly confirmed.**
- One overclaim needs fixing. "'Stance solved' holds only for the screen recipe" is too strong. The screen recipe passed screen measurements only: no M4 (K1-K4) run, and the reading is NOT CONFIRMATORY, with the test NOT SEPARATED.
- Six smaller corrections follow.
- Six omissions are listed after them, chiefly the action-noise counter-evidence.

**Confirmed** [measured unless marked]
- stages.toml l.34; SCORE l.361 (no parent named because M4 is absent); GQP l.185, l.199, l.700, l.1090, l.86 and l.89.
- MAIN has one gated stance term. RB/BR have four, at RB l.418/425/433 and BR `base_env` l.1278/1315/1339/1421.
- The recovery, locomotion, stages and follow_direction configs are identical on MAIN and BR.
- Plant files are identical: physics 10, interface 15.
- MAIN has no critic_phase code. SB3 2.9.0 save_util l.166 catches only `(RuntimeError, TypeError, AttributeError)`.
- BR refuses the load both ways. The code is at train_base l.606-610 and l.671-676, and critic_phase_ppo l.299-304 and l.310-311. The specs are recovery (140, 0.98) and locomotion (200, 0.995), against stance's (200, 0.98).
- MAIN's walking has no step-gated term. Its T. rex env reads `_step_count` at l.850/1149/1157/1194/1251.
- Pilot history: BRP l.1319/1328 and RUNLOG l.606-607.
- PHASE action std: 0.0597, 0.0569 and 0.0585, independently re-read.
- New measurement: the PHASE checkpoint data records `CriticPhaseActorCriticPolicy` and `mesozoic_critic_phase` with settle_steps 200 and gamma 0.98 (`zip_classes.out`).

**Corrections**
1. The bottom line's "stance solved for the screen recipe" is an overclaim.
2. The BR recovery four-term claim is tagged [measured] in the bottom line and [inferred] in §5.
3. Citations:
   - The ramp is at `locomotion.toml` l.161-162, not `base_env`.
   - Stance's settle step of 200 is at `stance.toml` l.188 (MAIN), not l.396.
   - The D-D37 parking is at BRP l.1320 and RUNLOG l.173.
   - `warm_start_log_std` and `locomotion_gait` both appear in docs (30 and 13 lines), just not in code.
4. Only pilot 1 ran; pilot 2 never launched.
5. The PR-G5..G7 sizes and prerequisites are in GQP l.489-492; they are not "not found".
6. Step 2's M4 is already owed under CRIT l.270-272, so a new approval is not clearly needed.

**Missing**
- Pilot 1's std rose to 0.108–0.111 and it still stood still.
- Locomotion has its own entropy: warm-up 0.02, then 0.005 → 0.001.
- Pilot 2's rebuilt parent had std 0.064, a closer comparator.
- Option E needs reward code on main for stagger and band, or a recipe the phase critic was never trained on.
- Stock-critic finals load with no new code but carry a habit (OFF s80's neck).
- Step 2 is not strictly required (D-D36's parent had no gate verdict).
- Pushes recur every 2.0 ± 0.5 s; there is not a single push.
