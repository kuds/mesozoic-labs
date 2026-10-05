# T. rex locomotion gait-r1: CPU re-scoring of the hops (2026-10-04)

**Status**: dated investigation note, frozen once merged; corrections are appended, never edited in. It is the
evidence for the T. rex locomotion task revision `gait-r1` (`configs/trex/locomotion.toml`, a commit of its own
merged just before its training session; plan §5.2 item 5).
The method is step 2 of [../GAIT_QUALITY_PLAN_2026_09.md](../GAIT_QUALITY_PLAN_2026_09.md) §5.5. Runs were made
on 2026-10-04 on CPU, at the gait reward kit's commit `52a6962` (branch `trex-gait-reward`, parent `a07eafb`).

The scripts are not in the repository. They are a recomposition specific to `TRexEnv`'s reward sum, and they
read checkpoints copied from Drive. §2 gives the method in enough detail to redo it.

## 1. Question and answer

The kit adds four contact terms to the T. rex locomotion reward (`environments/shared/gait_rewards.py`): gait
phase, flight, foot slip and floor-force support. Which weights, together with the existing forward-speed
terms, pay a genuine 1.0 m/s walk more than the hops the certified walkers learned? The plan's criterion has
two parts:

- (a) no exploit earns more than under the legacy reward;
- (b) a walk at the 1.0 m/s gate speed clearly out-earns every hop and the statue. This note takes "clearly"
  to mean at least 10 % with a gait-phase quality Q of 0.8, and still positive at Q 0.6 with 3 % airborne
  steps.

The answer:

| Setting | Value | Was |
|---|---|---|
| `support_source` | `"floor"` | `"touch"` |
| `gait_phase_weight` | 0.5 | 0; the plan's starting value is 0.3 |
| `flight_penalty_weight` | 1.0 | 0; the plan's starting value is 0.5 |
| `foot_slip_penalty_weight` | 0.2 | 0 |
| `forward_vel_max` | 1.25 | 2.5 |
| `forward_vel_weight` | 1.0 | 2.0 |

On the development seeds this passes with margins of +729, +599 and +529 per 1000 steps (Q 0.8, 0.6 and 0.4).
The plan's starting set fails (b): under it the 2.02 m/s hop still out-earns a 1.0 m/s walk by 278. No trained
T. rex walker exists, so the walk side is a model anchored on scripted puppet walks, not a measurement (§3).

## 2. Method

**Sources.** Each source is rolled deterministically with its recorded environment kwargs, 1000 steps.

- **Hop 0914:** the handoff of run `20260914_123816`, locomotion, seed 42. The certified hop at 1.05 m/s.
- **Hop 0925:** the handoff of run `20260925_033501`, locomotion, seed 44. The certified hop at 1.57 m/s.
- **Hop 0930:** the locomotion handoff of run `20260930_024929`, a hop at 2.02 m/s.
- **The statue:** zero action.
- **E4:** an open-loop synchronous leg pattern, the best two-footed T. rex hop an open-loop search found. It
  wiggles in place.
- **Eleven puppet gaits:** walks, a run, hops, a limp, one leg and a shuffle. A puppet drives `TRexEnv`
  through its own action interface, with a harness carrying half the body weight (30 % in the "load70" rows).
  Only its contact terms are meaningful; the harness distorts the alive, height and posture terms.

**Step 1: the replay is unchanged by the knobs.** Every gait-r1 knob except `terminate_on_leg_contact` is
reward-only, and `forward_vel_max` enters only the forward and backward terms. On three seeds per hop, six
variants were rolled: legacy, the plan's starting set (alive conditioning off, then on), the starting set with
the cap at 1.5, and the chosen set with the cap at 1.25 and at 1.5.

- Observations, actions, qpos, qvel, lengths and terminations are bit-identical to the legacy rollout in every
  variant.
- So each episode is rolled once with every kit term at weight 1. The return under any weight set is
  recomposed per control step from the logged terms: legacy reward, minus the kit at unit weight, alive
  regated on floor support, forward re-capped, plus the weighted kit terms.
- The recomposition matches the real environments to within 1.8e-15 per step.

**Seeds.** A first pass replayed the hops on seeds 3042 to 3051, part of the certification block that
[../GAIT_CERTIFICATION.md](../GAIT_CERTIFICATION.md) reserves from reward design. Every measurement was then
repeated on the development block: seeds 9000 to 9009, 9000 to 9002 for step 1 and E4, and 9100 to 9104 for
the 2000-step survival check. The tables below are the development-block repeat.

- It reproduces every pass and fail of the first pass, and moves no margin by more than 13.
- The first pass gave +741, +611 and +541 for the chosen set.
- The certification block was not consulted again.

**The walk model.** No genuine walk exists to replay, so a walk's return per 1000 steps is modelled:

- Forward pay comes from the environment's formula, with a ±15 % speed ripple within each stride.
- Alive is 0.5 × ((1 − f) + f × 0.042). Here f is `support_conditioned_alive_fraction`, and 0.042 is the
  gate-speed puppets' share of steps with both feet loaded.
- Gait phase is the weight times Q = P · A · S. Clean puppet walks measure P · A = 0.97, and the 1.04 m/s
  puppet walks Q 0.80 to 0.88.
- Slip is 0.0084 per step.
- Posture, heading and the regularizers are 0.553 per step, the lowest of the three hops; the statue earns
  0.592.

The "puppet" rows instead use the 1.04 m/s puppet walks' measured kit terms, speed trace and support.

## 3. Results (development block, per 1000-step episode)

**Per term, legacy against the chosen set** (mean of 10 episodes; E4 has 3; no episode fell):

| Source | Set | Forward | Alive | Gait phase | Flight | Slip | Posture | Heading | Total | Change |
|---|---|---|---|---|---|---|---|---|---|---|
| hop 0914 (1.07 m/s here) | legacy | 858.6 | 500 | 0 | 0 | 0 | 297.0 | 295.6 | 1945.8 | — |
| | gait-r1 | 836.0 | 500 | 0.0 | −392.7 | −8.9 | 297.0 | 295.6 | 1521.6 | −424.2 |
| hop 0925 (1.56 m/s) | legacy | 1246.4 | 500 | 0 | 0 | 0 | 272.0 | 297.5 | 2308.5 | — |
| | gait-r1 | 913.9 | 500 | 0.0 | −503.2 | −10.5 | 272.0 | 297.5 | 1462.2 | −846.2 |
| hop 0930 (2.02 m/s) | legacy | 1585.3 | 500 | 0 | 0 | 0 | 265.4 | 295.9 | 2638.1 | — |
| | gait-r1 | 921.0 | 500 | 0.1 | −518.5 | −9.5 | 265.4 | 295.9 | 1445.9 | −1192.2 |
| statue | legacy / gait-r1 | −1.0 | 500 | 0 | 0 | 0 / −0.2 | 297.9 | 297.3 | 1093.7 / 1093.6 | −0.2 |
| E4 | legacy / gait-r1 | −1.2 | 500 | 0 | 0 / −1.0 | 0 / −1.0 | 249.4 | 297.3 | 1041.1 / 1039.1 | −2.0 |

The regularizers (−0.4 to −8.6) are in the totals but not shown. Leg contact and foot-on-foot contact were 0 on
every step of every source.

The hops' contact profile, per control step:

| Hop | Airborne | Slip / √(gL) | Touchdowns / s | Gait-phase pay at weight 1 | Both feet loaded |
|---|---|---|---|---|---|
| hop 0914 | 0.39 | 0.045 | 14.6 | 0.0001 | 0.51 |
| hop 0925 | 0.50 | 0.053 | 18.7 | 0.0000 | 0.38 |
| hop 0930 | 0.52 | 0.047 | 15.8 | 0.0001 | 0.30 |
| statue | 0.00 | 0.001 | 0.1 | 0 | 0.997 |

**Puppets, kit terms per second** (puppet seeds of their own, never panel seeds). The kit nets are gait phase
minus flight minus slip.

| Puppet | Speed (m/s) | Q per touchdown | Airborne steps / s | Kit net / s, starting set | Kit net / s, gait-r1 |
|---|---|---|---|---|---|
| walk, 0.26 m step | 0.48 | 0.43 | 0 | +13.1 | +21.8 |
| walk, 0.40 m step | 0.85 | 0.69 | 0 | +21.2 | +35.5 |
| walk at 1.0 m/s, 0.44 m step | 1.04 | 0.80 | 0 | +23.2 | +38.9 |
| walk at 1.0 m/s, 0.51 m step | 1.04 | 0.88 | 5.7 | +23.5 | +38.4 |
| run | 0.82 | 0.74 | 22.1 | +11.3 | +15.3 |
| hop | 0.57 | 0 | 49.7 | −24.9 | −49.8 |
| staggered hop | 0.57 | 0.001 | 38.9 | −19.5 | −38.9 |
| limp | 0.48 | 0.08 | 10.1 | −2.9 | −6.4 |
| one leg | 0.48 | 0 | 39.3 | −19.7 | −39.3 |
| shuffle | 0.48 | 0.05 | 26.3 | −9.2 | −19.6 |

**The criterion over candidate sets** (exploits measured; walk modelled; margin = walk minus best exploit):

| Set | hop 0914 | hop 0925 | hop 0930 | Statue | Largest gain vs legacy | Margin Q 0.8 / 0.6 / 0.4 | (a) | (b) |
|---|---|---|---|---|---|---|---|---|
| legacy | 1946 | 2308 | 2638 | 1094 | 0 | −785 | pass | fail |
| plan start (gait phase 0.3, flight 0.5, cap 2.5) | 1741 | 2046 | 2369 | 1094 | 0 | −278 / −353 / −398 | pass | fail |
| plan start + alive fraction 1.0 | 1495 | 1735 | 2021 | 1092 | −1 | −409 / −484 / −529 | pass | fail |
| plan start + cap 1.5, weight 2.0 | 2311 | 2570 | 2605 | 1093 | **+366** | +20 / −55 / −100 | fail | fail |
| plan start + cap 1.5, slope kept (weight 1.2) | 1740 | 1862 | 1876 | 1094 | 0 | +215 / +140 / +95 | pass | thin |
| gait phase 0.5, flight 1.0, cap 2.5 | 1544 | 1795 | 2110 | 1094 | 0 | +141 / +11 / −59 | pass | fail |
| gait phase 0.5, flight 1.0, cap 1.5, weight 1.2 | 1543 | 1610 | 1617 | 1094 | 0 | +634 / +504 / +434 | pass | pass |
| **gait-r1 (cap 1.25, weight 1.0)** | **1522** | **1462** | **1446** | **1094** | **0** | **+729 / +599 / +529** | pass | pass |
| gait-r1 with gait phase 0.3 | 1522 | 1462 | 1446 | 1094 | 0 | +569 / +479 / +449 | pass | pass |
| gait-r1 with flight 0.5 | 1718 | 1714 | 1705 | 1094 | 0 | +533 / +418 / +333 | pass | pass |
| gait-r1 + alive fraction 1.0 | 1276 | 1151 | 1098 | 1092 | −1 | +496 / +366 / +296 | pass | pass |

Under gait-r1 a 1.0 m/s walk is modelled at 2251 (Q 0.8), 2121 (Q 0.6, 3 % airborne) and 2051 (Q 0.4). The
1.04 m/s puppet walks, with their measured terms, come out at 2261 to 2312.

**The forward cap** (forward-term return of each hop; a 1.0 m/s walk earns 800 under any cap at or above 1.2
that keeps the 0.8 per m/s slope):

| Hop | Cap 2.5, weight 2.0 (legacy) | Cap 1.5, weight 2.0 | Cap 1.5, weight 1.2 | Cap 1.25, weight 1.0 |
|---|---|---|---|---|
| hop 0914 | 859 | 1429 | 858 | 836 |
| hop 0925 | 1246 | 1770 | 1062 | 914 |
| hop 0930 | 1585 | 1821 | 1092 | 921 |

**Modelled alternatives re-optimised against the new reward** (same per-step model as the walk; not measured):

| Set | Walk 1.0 m/s | Walk at the cap | Hop at the cap, 30 % airborne | Run at the cap (Q 0.75, 20 % airborne) | Grounded synchronous bounce, 10 % airborne | Slow perfect walk, 0.6 m/s |
|---|---|---|---|---|---|---|
| cap 1.5, weight 1.2 | 2251 | 2619 | 1885 | **2369** | 2085 | 2006 |
| **gait-r1** | 2251 | 2428 | 1695 | 2178 | 1895 | 2006 |
| gait-r1 with flight 0.5 | 2251 | 2428 | 1845 | **2278** | 1945 | 2006 |

**At the start of the forward ramp** (`forward_vel_weight` 0.2 from the stance checkpoint, an absolute value):

| Set | Statue | Hops | Walk 1.0 m/s | Walk minus statue |
|---|---|---|---|---|
| legacy | 1095 | 1173–1211 | 1133 | +38: the hops already beat both |
| plan start + alive fraction 1.0 | 1093 | 595–722 | 892 | **−201** |
| **gait-r1** | 1094 | 709–853 | 1611 | **+517** |

**The 2000-step horizon** (gait-r1 doubles `max_episode_steps`). Every hop completed 2000 steps on all five of
seeds 9100 to 9104, at 1.03–1.09, 1.57–1.59 and 2.21–2.30 m/s. Its per-episode values under the new horizon
are therefore about twice the 1000-step values above, and every margin scales with them. The zero-action
statue scores 2186.81 ± 9.70 over 40 episodes at 2000 steps (`zero_action_baseline.py trex:2 --episodes 40`,
seeds 3042 to 3081, the stage's statue reference). At 1000 steps it scores 1091.34 ± 5.48 under the gait-r1
weights and 1091.51 ± 5.46 under the legacy ones.

## 4. Why these values

1. **The cap is lowered with its slope kept.** Lowering only `forward_vel_max` raises the pay per m/s below
   the cap, so every hop gains (up to +366) and the walk's margin falls to about 0. Scaling
   `forward_vel_weight` with it keeps 0.8 per m/s. A walk at or under the cap is then paid exactly what it
   is paid today, the statue is unchanged, and a non-walk's forward advantage over a 1.0 m/s walk is capped
   at 0.8 × (cap − 1.0) per step.
2. **1.25 rather than 1.5.** At 1.25 every modelled alternative, a run at the cap included, loses to a
   1.0 m/s walk. At 1.5 a jog at the cap beats it (2369 against 2251), though not a 1.2 m/s walk. The old
   cap also sat 25 % above its 2.0 m/s gate.
3. **Flight 1.0.** An airborne step then nets −0.5 against the alive bonus of 0.5. It roughly doubles what
   the hops lose, and it is what makes a run at the cap lose. The 1.0 m/s puppet walks pay 0 to 57 for it.
4. **Gait phase 0.5.** Once the cap and flight sink the hops, this term ranks the walk above the other
   grounded gaits at 1.0 m/s: walk 2251, step-to about 2050, limp or shuffle about 1900, no qualifying step
   1851. It moves the stance checkpoint off the statue at the ramp start (+517). It stays small beside the
   forward slope, so a slow perfect walk still loses by 245; at weight 1.0 that gap would shrink to 170. A
   weight of 0.3 also passes (+569).
5. **The alive fraction stays 0.** The plan's starting set has 1.0, which is rejected. A gate-speed walk has
   both feet loaded on about 4 % of its steps, so f = 1.0 costs it about 480 per episode, against 245 to 350
   for the hops and 1 for the statue. At the ramp start it makes standing pay more than walking (−201). The
   flight penalty already removes what alive paid the hops in the air.
6. **Slip 0.2,** as the plan has it. It costs a walk about 2 and a hop about 10, so it decides nothing.

## 5. What the new reward still pays a non-walk

- **A run at the cap.** It ranks below any walk at or above 1.0 m/s but above every hop. `biped_walk` refuses
  airborne time above 0.10.
- **A skimming foot.** The gait-phase term has no clearance requirement:
  - a foot carrying under 4.2 N counts as swinging, so a skimming foot earns swing and step-through credit;
  - it pays no slip, which counts only feet carrying at least 42 N;
  - the scripted shuffle earns 16 % of a gate-speed walk's gait-phase rate.

  The gate's clearance and swing-ground rails catch skimming. A follow-up could make swing credit depend on
  clearance.
- **Near-walks earn part of the pay:**
  - a step-to gets half the alternation credit;
  - a limp gets about 0.17 of it;
  - a quick, short-stepping walk reaches Q about 0.4.

  All of them still beat every hop at the same speed, and the gate refuses them.
- **The walk side is an estimate.** If a real walker loses more posture or heading reward than modelled, the
  buffer is the Q 0.4 margin (+529).
- **Gait-phase pay is sparse.** It comes to about 20 per touchdown at weight 0.5 and a 1 s stride, against
  about 2.3 per step for everything else, which adds variance to PPO returns.
- **The pilot ends early in the ramp.** A 1.5M-step pilot ends inside the 2M-step forward ramp, at 80 % of
  the final forward weight. It judges the gait before speed is fully paid.

## 6. Limits

- Every replay is a CPU replay of checkpoints trained on GPU: fresh evidence on this runtime, not the
  recorded trajectories ([GAIT_AUDIT_2026_09.md](GAIT_AUDIT_2026_09.md) §2).
- The modelled walk and the re-optimised alternatives are arithmetic on measured per-step rates, not
  trained policies. Only a pilot shows what PPO finds under the new reward.

## 7. Correction (2026-10-05): two non-walks the re-scoring did not model

A review of the kit found two contact patterns that the gait-phase term paid and §3 did not model. Both broke
criterion (a). They are fixed in `environments/shared/gait_rewards.py` before any training under `gait-r1`, and
the revision's TOML values are unchanged. The figures below come from scripted contact sequences at the T. rex
scale, scored with this note's per-step constants.

- **Marching in place in a split stance.** `S` measured only how far the landing foot lands past the other
  foot's footprint. A T. rex stepping in place with one foot always ahead therefore earned gait phase at zero
  speed: 0.250 per control step with the feet 0.56 m apart, half a 1.0 m/s walk's 0.497. That is 1340 per
  1000 steps, against the statue's 1094. `S` is now capped at half the foot's own stride, measured touchdown
  to touchdown along the same heading (the checker's in-place stride test). A walk is paid as before. A split
  or level march earns 0, and so do feet sliding back under a body that stays put (1090 per 1000 steps).
- **A one-legged hop whose other foot taps with an impact spike.** A stance counted as a step once a single
  substep reached half the foot's share of body weight. In the 2 ms traces of the certified hop runs, 33 of
  the 34 contacts shorter than 30 ms reach that load, for 9 ms at the median. Such a tap made the hopping
  foot's steps alternate: 0.442 per control step, 89 % of a walk's rate, or 2075 per 1000 steps against about
  1949 under legacy. A step now needs its stance to hold the load for 30 ms in all. The spiked-tap hop earns no
  gait phase, like the same hop without the tap (1633).

Nothing else in this note moves:

- **The hop rows of §3.** Rescored from the certified runs' traces, the recorded hops' gait-phase rate falls
  from at most 0.0009 to at most 0.0004 per control step.
- **The modelled walk.** Its steps advance by half their stride, and its stances hold the load for hundreds
  of milliseconds.
- **§4 item 4's step-to.** A step-to's leading foot is now credited for half its own stride. At 1.0 m/s and
  the walk's 1.12 s cadence, that stride is 1.12 m, twice `l`, so the step-to keeps half the alternation
  credit (about 2050). A step-to whose lead steps are only `l` long now earns a quarter of a walk's gait-phase
  rate instead of half.

A step now registers when its stance completes the 30 ms, about 30 ms after touchdown instead of a few
milliseconds. In the digest harness's T. rex locomotion capture, the roll's two steps (each foot's first, which
pays nothing) register two control steps later, and the three-step truncation probe ends before its step
completes. No reward value in that capture changes.
