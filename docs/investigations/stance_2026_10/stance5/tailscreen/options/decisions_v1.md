# Tail_3 bound screen: decisions before any build or training (v1)

Criteria text: `criteria_tailscreen_v1.txt` (same folder). It applies the recommended option of each decision below. Nothing has been built or trained.

**Predicted outcome at the recommended settings** [inferred]: most likely NOT ADOPTED. s42 would read RAIL or PARKED, and s80 PARKED.
- The bound at 0.90 caps training rows but does not push the deterministic command below 0.60.
- P1's training-row tail_3 mean of 0.566 still gave a CMD of 0.837 with 12/20 flagged [measured, TCSV].
- The loop gain is about 0.66 at both starts [measured hold cells, PROBES noisevalue].

The screen still settles whether a 0.90 bound keeps the rail away across replicates.

## D1. How PARKED treats the clean holding states
- (a) Literal ruling: PARKED means no rail and CMD >= 0.60 at both states, with no upper end. Stance3 Run 2's final (+0.841), Run 2 at 10.49M (+0.866) and the s80 start (+0.629) all read PARKED.
- (b) Same line, but at s80 PARKED is only reported. This departs from the ruling at s80.
- (c) A line at 0.85. This spares s80 and Run 2's final, but calls P1's parked 0.70 and 0.79 CURED.
- **Recommend (a).** No CMD line separates the clean holding states from P1's parked checkpoints [measured]. SAT already read Run 2's final as "not a cure". So at s80 the bound must lower the command (start SE 0.048).

## D2. Bound level b
- (a) 0.90: where the saturation ramp starts, below the gate's 0.99.
- (b) 1.00: P1's level, which never brought the deterministic mean under 1.0 at 7 of 9 checkpoints.
- (c) 0.60: a posture bound. Its start pull is about 2x the G_pg RMS, which makes it close to a reward change.
- **Recommend (a).** The maintainer scoped a rail guard, and posture belongs to the B/C decision. Option (b) repeats P1, and (c) changes what is being screened.

## D3. Coef
- (a) P1's rule: RMS(G_pg) / (2 x 0.11), pooled over both states. Expected 2.7. The realised start pull is 0.064, about half the outward pull of 0.122.
- (b) A separate coef per state (about 3% apart).
- (c) A dose that balances the measured outward pull at the start: about 5.2 (0.122 / 0.02362 on the proxy rows), recalibrated in coef.json.
- **Recommend (a).** It keeps P1's per-row stiffness (5.4 vs 5.47) and the brief's calibration rule. A NOT ADOPTED at 2.7 does not reject (c), and either dose leaves the deterministic posture untargeted.

## D4. Horizon
- (a) The draft's plan: 0.5M steps, stop if either state is RAIL, otherwise extend both to 1M. Failure gets two looks and a pass one. P1 would have read final RAIL at 0.5M before it cleared at +0.79M. This needs resume code (PHASE 2 folder, new seed, env reset, test T6).
- (b) The brief's plan: extend only states that have not separated; a 0.5M PASS is final.
- (c) One continuous 1M run per replicate (4.19M -> 5.24M), read at 5,111,808 and at the final checkpoint only.
- **Recommend (c).** It reads pass and fail the same way and needs no resume path or seed replay. It also catches P1's late relapse (0 of 20 flagged at +0.92M, 14 of 20 at +1.05M), for about 4 h instead of 2-5 h.

## D5. State reading order
- (a) Revised order: RAIL, then PARKED, then OTHER, then NOT SEPARATED (s42 only), then PASS. At s80, a CONTROL that takes the rail only adds an "onset tested" note.
- (b) The brief's rule: s42 passes with >= 2 BOUND CURED and <= 1 CONTROL CURED, even if one BOUND replicate is on the rail.
- **Recommend (a).** With this order, more onset in the controls never worsens the verdict, which fixes the draft's non-monotone onset reading. A parked bound always routes to the posture decision (ruling 3).

## D6. How R1 and R2 combine for PARKED / CURED
- (a) The mean of CMD at R1 and R2 (40 episodes) against 0.60.
- (b) PARKED if either checkpoint is >= 0.60 (the draft).
- (c) PARKED only if both are >= 0.60 (the brief).
- **Recommend (a).** It treats PARKED and CURED symmetrically, with an SE of about 0.03. Options (b) and (c) each push near-line replicates one way: a true CMD of 0.55 reads CURED about 0.72 of the time under (b).

## D7. What counts as RAIL
- (a) Any flagged episode on any actuator, with short episodes counting.
- (b) Only tail_3 tail-up flags (more than 10% of window steps at +0.99) count as RAIL. Flags on other actuators and short episodes are compared between arms: if BOUND has more, the state reads OTHER (not adopted, sent to harm review).
- (c) >= 2 flags.
- **Recommend (b).** It labels the habit correctly and still blocks a bound that shifts saturation or causes falls. The stray rate is 1 flag in 1,560 non-habit episodes [measured, K3DIAG], which is about 0.07 per state under either (a) or (b). Option (b) also replaces the draft's fall guard.

## D8. Durable copy if the container is lost
The previous container's checkpoints and venv were lost [measured, RUNLOG 15:14-16:08].
- (a) An orphan handoff-style branch, never merged, as at 13:59 UTC. It carries the data (run_meta, R1/R2 pairs, panels, coef.json, criteria) and the bound commit as a patch file (each file <= 5 MB). This puts the scratch patch on origin as a file, which needs your explicit OK under "never pushed".
- (b) A Drive folder you name. This needs Drive write access.
- (c) No copy: container loss restarts the whole screen.
- **Recommend (a).** P1's module was lost in exactly this way. Without the patch, a rerun cannot match the pinned commit.

## D9. Seeds and the pipeline anchor
- (a) Seeds 101, 202 and 303, shared across arms and states, plus V3. V3 is a seed-42 CONTROL run of 131,072 steps compared with P1's CONTROL row. It is report only and never blocks, since P1's stack is gone.
- (b) Include each run's own seed. At s42, seed 42 is P1 CONTROL's draw, which is known RAIL.
- **Recommend (a).** Fresh seeds avoid draws whose outcome is known in advance, and V3 costs about 10 minutes.

## Compute
- Measured on this container (4 CPUs, 25 GB free) [FEAS throughput]: 21.8 s per rollout alone, and 22.9-23.9 s with three at once. A 20-episode panel takes about 1 min.
- Training [inferred]: 1M steps is about 50 min per replicate. The 12 replicates run in 4 waves of 3, about 3.4 h.
- Panels: 108, run on the fourth core alongside training.
- Total: about 4 h wall, range 3-7 h.
- Disk: about 5.2 GB peak. Training pauses if free space drops below 5 GB.

## Build list (nothing built)
1. **Scratch clone** at cf80ecb1: push URL disabled, never pushed.
2. **Bound commit** in `mirror_kl_ppo.py` and `tests/test_mirror_kl_ppo.py` only. The bound state is excluded from save, and a JSON record `mesozoic_tail3_bound` is written.
3. **Tests** T1-T5 and T7:
   - T1: coef-0 bit-identity.
   - T2: same-seed reproducibility.
   - T3: scope.
   - T4: mirror.
   - T5: no stage key.
   - T7: a BOUND zip loads with stock PPO in the measuring copy.
4. **`run_tailscreen.py`**: `--state`, `--rep-seed`, `--arm`, `--steps 1048576`. Stack versions go into run_meta.
5. **`calib_tailscreen.py`**: writes coef.json.
6. **`measure_tailscreen.py`**: runs the trace tool from cf80ecb1 plus the pinned overlays. It writes TAIL-UP and OTHER flags to panels.csv, then gzips the per-step CSVs.
7. **`score_tailscreen.py`**: synthetic-row tests, including monotonicity.
8. **`copy_durable.sh`**: for D8.
9. **Before training**: the tests pass; V1 reproduces both start panels exactly; V2 (coef.json); V3 report; pins and sha256 are recorded in RUNLOG.
