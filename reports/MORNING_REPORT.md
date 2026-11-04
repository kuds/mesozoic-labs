# Overnight report, 2026-11-04 (6 a.m. CST)

CPU only. Nothing was trained, pushed, or changed in the repository or the physics.
- **Checks.** Every headline result was re-run by an independent verifier, and the verifiers' corrections are applied.
  This report itself was then checked against the records. Some secondary figures were not re-run; the verifiers list
  them.
- **Labels.** [measured] means re-run and checked; [inferred] means reasoning; "test-only" means scored offline on a
  copy, never applied.
- **Paths.** `work` = `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work`.
  - `work/run_log.txt`: the run log.
  - `work/feas/FEAS_RESULTS.txt`: the feasibility study, with its verifiers in `feas/verify_existence` and
    `feas/verify_analyses`.
  - `work/overnight/<stage>/RESULT.md` and `work/overnight/verify_<stage>/`: the overnight stages and their verifiers.
    The hop stage is verified in `verify_screen`.

## Bottom line

1. **The r10 T. rex can walk.**
   - **The controller.** A hand-built controller walks it with alternating steps, using only the actions a policy could
     output.
   - **Speeds held:**
     - 1.0 m/s on 40/40 fresh seeds;
     - 1.2 m/s with no flight on 39/40.
   - **Actuators.** No actuator stops a 1.2 m/s walk. The ankle is closest to its limit. [measured; "no actuator
     stops it" inferred]
2. **Today's walking reward prefers a two-footed hop to a walk at the same speed (0.7-1.2 m/s).**
   - **Speed.** A hand-built hop holds up to 2.0 m/s, and the reward pays it.
   - **Cost.** Flight and hard landings cost nothing.
   - **Discounted return.** Measured in discounted return early in training, which is PPO's own objective, walking
     already scores below standing.
   - **History.** This is a likely reason September's runs hopped. [measured; "likely reason" inferred]
3. **Pilot 3 as frozen would start badly.** Its -1.5 noise reset knocks the s42 parent over in about 93% of episodes.
   At -2.0 the parent stands in 359 of 360. [measured]
4. **Reward fixes exist, tested offline on hand-built behaviours.** The most robust needs a small amount of new
   reward code. A config-only setting also passes, but it is weaker in a way that matters (see below). [measured]
5. **The 1.0 m/s gate is a fair speed but a poor test.** It averages over the whole episode, including the start from
   standing, and checks no gait. It passes fast hops and no strict hand-built walk. [measured]
6. **For the slow walk, the policy's inputs are not the barrier, and no plant tweak is needed.** [inferred from
   measured tests; see Findings for scope]

## Findings

### Walking ability

- **Speeds held** [measured]:
  - 0.2 m/s: 10/10 seeds and 20/20 held-out seeds.
  - 1.0 m/s: 40/40 fresh seeds; stride about 0.55 leg lengths at about 2 Hz.
  - 1.2 m/s: 39/40 strict; stride 0.64 L; Froude 0.17.
  - 1.35 and 1.5 m/s: survive 15/15, but with brief airborne steps. At 1.5 m/s about half of these are short flights
    before touchdown, of unknown cause.
- **Actuators** [measured]:
  - At 1.2 m/s, leg force stays at or below 0.52 of its force limit.
  - Measured against what the soft position servos can actually deliver, the ankle peaks at 0.91-0.99, and its raw
    action is clipped on 9-11% of steps.
  - The plant's top walking speed is not measured.
- **Static balance.** Neither body can hold a one-leg pose without active balance. That includes Compsognathus, the
  control, which walks (`feas/support`, `feas/verify_analyses`). So walking needs active balance, which the
  controller provides. [measured; "needs" inferred]
- **Caveat.** The controller keeps memory (a step timer).

### The reward (`overnight/reward`, `hop`, `screen`, `configonly`)

- **Early in training (forward weight 0.2)** [measured]:
  - A slow walk beats standing by only 0.5% per episode.
  - The crouched 0.7 m/s walk loses to standing.
  - In discounted return (gamma 0.995), the 0.7 and 1.2 m/s walks score 212 and 214 against standing's 217.
  - Walks win clearly by the end of the ramp.
- **Gait** [measured]:
  - The reward never reads which foot is down and charges nothing for flight.
  - A hand-built two-footed hop holds 0.7, 1.0 and 1.4 m/s on 40/40 seeds each, and 2.0 m/s on 37/40. It lands at
    3.5-4.2 body weights (p95), with one landing at 14.6.
  - At 0.7, 1.0 and 1.2 m/s the hop out-earns the walk at the same speed command: by 60, 86 and 111 per episode on
    average (weight 2.0), with the walk ahead on 0 of 20 seeds. The hop runs 0.01-0.03 m/s faster, which is worth
    7-25 of that lead. At 0.2-0.5 m/s the walk leads.
- **Test-only fixes.** The test is whether every stable walk beats standing and its matched hop on every seed, at
  both ends of the ramp. Six walks were run on 20 seeds and four on 20 more held-out seeds. Two fixes need new code:
  - **R6** combines:
    - foot-support gating of the alive, posture and height terms, as Compsognathus's walking reward does;
    - an airborne penalty;
    - `forward_vel_max` 1.5, which stops pay above 1.5 m/s and steepens it below;
    - a small reward for alternating touchdowns, a simplified form of the gait plan's GQ-15 option (a).

    Worst seed: +87 over standing (weight 0.2), +778 over the matched hop (weight 2.0), and +42 over the 2.0 m/s hop
    for the 0.5-1.2 m/s walks. The 0.2-0.3 m/s walks stay below the 2.0 m/s hop. In discounted terms the walks score
    227-234 against standing's 217.
  - **R15** is support gating plus the touchdown reward. It passes everything except the 2.0 m/s hop, which still
    out-earns the 1.0 m/s walk (1780 against 1725).
  - [measured]
- **A config-only setting, I6** (test-only):
  - **What it is.** Two existing flight penalties (`airborne_substep_weight` 1.0, and
    `foot_load_balance_airborne_penalty` 0.8 at 33.6 N) plus `idle_penalty_weight` 0.1.
  - **Margins.** Worst seed +42 over standing and +760 over the matched hop. +109 over the 2.0 m/s hop for the 0.5-1.2
    m/s walks.
  - **But** [measured, with weaknesses inferred]:
    - It was chosen after seeing part of the data.
    - In discounted terms its margin over standing is about 4.
    - It wins over standing by charging for stillness, not by paying for stepping, so a learned fidget that keeps
      moving would escape the charge.
    - It lowers the statue's return by 77, so the stage's collapse reference has to be re-measured.
- **The only pre-declared config-only setting that passes (X4v12)** needs `forward_vel_max` 1.2. That conflicts with
  the 1.0 m/s episode-mean gate. [measured]
- **Existing settings that don't work** [measured]:
  - `support_conditioned_alive_fraction` pays the alive bonus only while both feet are loaded, so it ranks standing
    first.
  - `gait_symmetry_weight` does not reward alternation: it pays the statue 997.6 of 1000.
- **Limit.** All these screens rank 21 fixed hand-built or checkpointed behaviours. They don't show what PPO will
  learn, and an untested gait could rank differently. [inferred]

### Pilot 3's start (`overnight/logstd`)

- **How often the parent stands, by noise reset** (stochastic actions; independent panels pooled) [measured]:

  | log_std | std | stands |
  |---|---|---|
  | -2.0 | 0.135 | 359/360 |
  | -1.875 | 0.153 | 309/320 |
  | -1.75 | 0.174 | 253/320 |
  | -1.5 (frozen) | 0.22 | 16/240 |

- **At -2.0.** The probe classes 118/120 episodes as tapping in place: median lift 2-3 mm, largest 18 mm. The parent
  slides backward about 0.23 m per 10 s.
- **At -1.875.** Most episodes are classed as shuffling.
- **Reading.** None of these settings shows a step. -2.0 is the safe start; -1.875 is at the edge. [measured;
  reading inferred]

### The gate (`overnight/benchmark`, `screen`)

- **As a speed.** 1.0 m/s is Froude 0.12 for this body: a fair walk.
  - A Froude-scaled full-size T. rex walk would be about 0.65-0.7 m/s here [literature, recalled, unverified: van
    Bijlert et al. 2021].
- **As a test.** It averages speed over the whole episode.
  - These hand-built walks, with their 3.5-4 s start-ups, would need about 1.3 m/s once moving to pass; a quicker
    start would need 1.03-1.18 m/s.
  - It passed no strict walk, but it passed the 1.4-2.0 m/s hops and a mostly-falling fast walk.
  - [measured; quick-start figure inferred]
- **A better test** (test-only). A last-5-s speed plus the gait probe's walk class, counted on full-length episodes
  only, rejects every hop and passes the falling walk on only 3/20 episodes.
  - At 1.0 m/s it passes the 1.2 m/s walk on 20/20 episodes, but the 0.98 m/s walk on only 2-3/20.
  - At 0.7 m/s it passes both.
  - [measured]

### Inputs and plant (`overnight/observability`, `variants`)

- **Inputs** [measured]:
  - **On its own limit cycle.** For the slow walker (0.2 m/s), the 63 inputs already determine about 99% of its
    actions. The one missing piece is its own step timer (memory), not the previous action or a phase.
  - **Under training-like action noise** (sigma 0.1-0.3), the inputs miss 25-41% of the action. The previous applied
    action recovers about 2.4 points of the raw action, and the filter state recovers up to about 5 points of the
    intended command.
  - **Not analysed:** the gate-speed gait.
- **Plant: stiffer foot contact (solref 0.01, test copy).**
  - It stops the foot-roll creep.
  - Stance stays inside every gate bar (parent 40/40), but moves toward two of them.
  - It needs a geom-priority setting to take effect.
  - It isn't needed for walking, since the real plant already walks at 1.2 m/s. [measured; "not needed" inferred]

## What it means for pilot 3 [inferred]

- **The noise reset must change** to -2.0.
- **On today's reward,** the likely learned outcomes are a hop or standing still: hops pay, and early-training
  discounted returns favour standing.
- **A reward fix makes the pilot test the right question:** can a habit-free stance trunk learn to walk when walking
  is what pays?
  - R6 is the most robust option. It needs new reward code (support gating, touchdown event) with tests, off by
    default, on the pilot branch.
  - I6 is config only, but weaker.
- **The gate cannot judge the pilot.** It is not what judges pilot 3: the gait probe is. But as written, a fixed-reward
  pilot could fail it while walking or pass it while hopping. Changing it is a separate decision.
- **Amendments.** Any change means amending the frozen text and branch commit 2 (plus new reward code for R6), then
  re-pinning and a short review, before your approval and the launch.
- **Stance is unaffected.** The walking reward is separate from the stance reward, and the parent is unchanged.

## Process notes

- **Restarts.** Two container restarts hit overnight runs, at about 01:52 and 06:21 UTC. Completed results were kept.
  Partial runs cut by a restart were re-run deterministically, and no completed record was replaced.
- **Rule slips by agents** (all logged in the run log):
  - Two verifiers redirected a command's error output to the forbidden discard device, at about 01:59 and 05:58 UTC.
    One of them redacted an earlier line of its own log.
  - Several agents built search patterns from the device-folder prefix.
  - Several process listings showed the harness's own command line on screen.
  - Some interpretive sentences were labelled [measured] (corrected).
  - Command logs are often paraphrased.
  - I removed 52 unused venv activation scripts, which contain a discard-device token, from all scratch venvs.
- **Cost.** Overnight subagent tokens were at least 4M, plus work lost in the restarts.
