# T-Rex Simulation Review — July 2026

**Scope:** `environments/trex/**`, `configs/trex/**`, and the parts of
`environments/shared/**` the T-Rex actually executes (dependency verified by reading
the call path, not inferred from the directory).
**Out of scope:** `website/**`, hardware/BOM docs, and the other three species except
where a T-Rex change reaches them through `environments/shared/**`.
**Tree reviewed:** `081a2020` (`main` at review time).
**Phase assumption:** simulation only. `configs/species_manifest.toml` records
`hardware_prototype = planned`, `sim_to_real_validation = not_started`; nothing below
evaluates the plant against hardware fidelity. Welded arms, the fused distal tail and
the idealised position actuators are deliberate simplifications and are not findings.

---

## TL;DR

Three findings, one fixed.

1. **Stage 1 pays for existence, not for balance.** A do-nothing policy that survives
   the horizon scores **2834.95 ± 6.25** under the shipped stage-1 config on the current
   plant. The best trained checkpoint on the *same commit and same plant* scores
   **2404.73 ± 350.64**. Conditional on standing, six million PPO steps produce something
   **12% worse per step** than `np.zeros(21)`. The stage clears its 1840 gate on one thing
   only: the statue falls out of 43% of episodes and the policy mostly does not. That is a
   real skill, but it is the only one the reward can distinguish, and the reward's own
   argmax is "emit zeros and get lucky".
   Proposed, not fixed — reward weights are out of bounds for this pass.
2. **The SB3 and MJX paths terminate T-Rex stages 2 and 3 at different pitch angles**
   (0.62 vs 0.50), while `environments/trex/mjx_config.py:45` states in a comment that
   both use 0.62. **Fixed** in this branch, with a parity test that fails before and
   passes after.
3. **No one-way door in the action space, the control rate, or the actuator interface.**
   21 position targets at 100 Hz around a named home keyframe is something a physical
   robot could accept as-is. The observation vector is a different story: **7 of its 61
   elements are privileged**, and the layout is the single most expensive thing in the
   repo to change later.
4. **Ablating the trained checkpoint says those privileged slots are load-bearing — but the
   worst of them is free to fix.** Removing all seven takes stage-1 return *negative* and
   survival to 0%. Removing the three world-frame root-velocity slots costs **−71.5%**.
   Rotating those same three into the **body frame** costs **+4.1%** — nothing — even though
   that rotation moves the policy's input 69–92% as far as deleting it does. Body-frame root
   velocity is what a real state estimator produces, so the one genuinely unmeasurable group
   in the observation can be fixed at zero policy cost. That is NS-6, and it is now the
   cheapest item on the list rather than a blocked one.

---

## 1. Review window

`git log --since="2026-07-20" --date=short --pretty="%h %ad %s" -- environments/trex configs/trex environments/shared`

```
081a202 2026-07-27 Merge pull request #468 from kuds/claude/action-bound-report
cbf7c65 2026-07-27 Repurpose action_bound_report as a saturation probe, not a reward audit
4a96313 2026-07-27 Fix mypy assignment error in action_bound_report.py
ed70392 2026-07-27 Add action_bound_report.py: size the unclipped-action reward gap
530c6e6 2026-07-27 Widen the MJX/CPU foot-contact tolerance to a float32 solver residual
50cc7fa 2026-07-27 Unblock CI: fix two lint failures inherited from main
5a93c42 2026-07-27 Stand the T-Rex on a flexed theropod limb, not a columnar one
b1fedbf 2026-07-27 Add joint-excursion tooling and the raptor comparison to the scaling plan
2709725 2026-07-27 Escalate T-Rex stage-1 smoothness_weight from 0.7 to 2.0
8be56d2 2026-07-26 Drop the env annotation from the height-target helpers; keep only the float cast
f1597e1 2026-07-26 Annotate the height-target test helpers so mypy can infer their return types
d852669 2026-07-26 Guard the new height target against path drift, and document the gate's couplings
121822f 2026-07-26 Fix the T-Rex stage-1 jitter lever, stage-2 entropy anchor, and stale height target
a5225b3 2026-07-25 Use a float32 summation tolerance for the MJX foot-contact assertions
f7a8bba 2026-07-25 Assert the MJX foot observation against pad + digits, not the pad alone
84ade97 2026-07-25 Unify the T-Rex alive envelope across the SB3 and MJX paths
aa87445 2026-07-25 Make the T-Rex foot sensors see the digits and realign the MJX pitch reference
bb2ba15 2026-07-25 Fix the Docusaurus build: .gitignore was swallowing the new model page
70a42f7 2026-07-25 Fix the stage-1 reset calibration and the snout-gate test
8af680c 2026-07-25 Add the $10k chassis-width and lateral-actuator study
833c606 2026-07-25 Add Dibothrosuchus elaphros: an erect-limbed crocodylomorph species
d6f44c1 2026-07-24 Calibrate stage-1 reset noise against a zero-action baseline
98caef4 2026-07-24 Revise the T-Rex plant: load-bearing digits, welded arms, fused distal tail
2c28bc3 2026-07-24 Add run-diagnostics comparison script
60fd853 2026-07-24 Fix T-Rex stage 1 leg jitter: anchor entropy decay to 3M steps
4094486 2026-07-23 Fix T-Rex home equilibrium and contact sensing
2358c21 2026-07-23 Format SB3 bundle regression test
966ab77 2026-07-23 Fix duplicate SB3 bundle finalization
3d21f49 2026-07-22 Make callback signature test backend-independent
7d13d5d 2026-07-22 Format collapse callback
2334bc7 2026-07-22 Harden eval-collapse early stopping
187f8b0 2026-07-22 Harden collapse peak against early-window spikes; bump version to 0.3.4.dev0
cbb14a9 2026-07-22 Smooth the collapse backstop symmetrically (trailing mean) to stop false aborts
209736f 2026-07-22 Raise velociraptor stage-3 budget to 12M timesteps
bde7d3e 2026-07-21 Update catalog test for velociraptor visual revision 3
d06d2b6 2026-07-21 Hide the enlarged foot touch sites from rendering (site group 4)
db0421e 2026-07-21 Harden eval-collapse backstop; align metrics.json with promoted checkpoint
aa3395c 2026-07-21 Fix dead raptor foot touch sensors; raise knee forcerange to 1.5x kp
db153b2 2026-07-20 Document Velociraptor Stage 1 lessons and bump version
352af12 2026-07-20 Center Velociraptor posture reward on natural lean
c8af06d 2026-07-20 Center Velociraptor actions on home pose
```

40 commits. **The window was neither widened nor narrowed.** Six of these
(`c8af06d`, `352af12`, `aa3395c`, `d06d2b6`, `db0421e`…`3d21f49`, `209736f`) are
velociraptor- or infrastructure-driven, but they land in `base_env.py`,
`reward_functions.py`, `curriculum.py` and `train_base.py`, all of which the T-Rex
executes, so they are in scope.

Merged PRs in the same window (`list_pull_requests`, state=closed, merged ≥ 2026-07-19),
T-Rex-relevant ones in bold:

| PR | merged | title |
|---|---|---|
| **#468** | 07-27 | Add action_bound_report.py: a per-actuator saturation probe |
| **#467** | 07-27 | Record the action-saturation finding, and the pre/post-clip metric hazard |
| **#466** | 07-27 | Record the measured joint envelope as the pre-fix baseline |
| **#465** | 07-27 | Widen the MJX/CPU foot-contact tolerance to a float32 solver residual |
| **#464** | 07-27 | Stand the T-Rex on a flexed theropod limb, not a columnar one |
| **#463 / #462** | 07-27 | Escalate T-Rex stage-1 smoothness_weight from 0.7 to 2.0 |
| **#461** | 07-26 | Fix the T-Rex jitter lever, stage-2 entropy anchor, and stale height target |
| **#460** | 07-25 | Fix the T-Rex foot sensors, pitch reference, and SB3/MJX envelope divergence |
| #459 / #458 | 07-25 | Add Dibothrosuchus elaphros; document the $10k quadruped |
| **#457** | 07-24 | Revise the T-Rex plant and calibrate stage-1 against a zero-action baseline |
| #456 | 07-24 | Document a sub-$5k quadruped starter |
| **#455** | 07-24 | Fix T-Rex stage 1 leg jitter by anchoring entropy decay schedule |
| **#454** | 07-23 | Fix T-Rex home equilibrium and contact sensing |
| #453 / #452 / #451 / #450 | 07-21…07-23 | SB3 bundle finalization; eval-collapse hardening; raptor plant defects |
| #449 / #448 | 07-20 | Velociraptor posture / action centring (shared code) |
| #447 / #446 | 07-19 | Hardware planning docs; result-bundle integrity |

No discrepancy between the commit list and the PR list: every T-Rex commit in the window
arrived through one of the PRs above.

### Runs covered

`MyDrive/mesozoic-labs/logs/trex/` contains two `<algo>` folders — **`ppo`** and **`jax`** —
plus a set of legacy flat run folders (`ppo_20260301_160855`, `sac_20260326_153101`, …)
that predate the `<algo>/<timestamp>/` layout.

* `jax/` — newest run `20260403_145327`. **Nothing in the window.**
* legacy flat folders — newest `sac_20260326_153101`. **Nothing in the window.**
* `ppo/` — 6 runs in the window; all 6 are covered below.

So although the layout admits more than one algorithm, **every T-Rex run in this review
window is SB3 PPO.** Everything measured below therefore describes the SB3 path; where
the MJX path differs, that is called out as a divergence, not as a property of these runs.

---

## 2. Run comparison and cohorts

The plant changed twice and the stage-1 reward changed three times inside the window, so
the six runs are **not** mutually comparable. Provenance comes from each run's own
`plant_identity.json` / `stage1/stage_config.json`, not from timestamps.

| run | algo | stages | plant (phys/policy) | obs | stage-1 config | gate faced | S1 final | S1 best (@step) | S1 len | S2 final / fwd vel | S3 final / len | wall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `20260723_204941` | ppo | 1,2,3 | **2 / 2** | 83 | smooth 0.1, noise 0.05 | 100.0 | 2079.00 ± 751 | 2419.77 ± 316 (5.75M) | 844.0 | 2920.61 / **3.23** | 1301.77 / 196.3 | 13h49m |
| `20260724_140441` | ppo | 1,2,3 | **2 / 2** | 83 | smooth 0.1, noise 0.05 | 100.0 | 1256.87 ± 490 | 1755.42 ± 745 (4.55M) | 605.1 | 2814.21 / **2.87** | 1262.98 / 213.5 | 13h03m |
| `20260725_194916` | ppo | 1,2,3 | **4 / 6** | 61 | smooth 0.1, noise 0.10 | 100.0 | 2651.80 ± 256 | 2704.44 ± 31 (5.90M) | 984.5 | 2995.71 / **2.96** | 1241.89 / 175.0 | 13h04m |
| `20260726_191730` | ppo | 1,2,3 | **4 / 6** | 61 | smooth 0.7, noise 0.10 | 1900.0 | 2503.81 ± 87 | 2564.98 ± 234 (4.95M) | 1000.0 | 2964.00 / **6.36** | 1236.06 / 131.7 | 12h53m |
| `20260727_130726` | ppo | 1,2,3 | **4 / 6** | 61 | smooth 2.0, noise 0.10 | 1900.0 | 2654.66 ± 12 | 2652.63 ± 17 (6.00M) | 1000.0 | 2896.00 / **4.36** | 1266.11 / 208.7 | 13h41m |
| `20260728_122755` | ppo | **1 only** | **5 / 7** | 61 | smooth 2.0, noise 0.10 | 1840.0 | 2344.74 ± 366 | 2423.83 ± 187 (5.95M) | 946.9 | *in flight* | — | 3h38m so far |

Sources: `training_summary.txt` and `stage1/stage_config.json` in each run folder;
`plant_identity.json` for the revision pairs.

### Cohorts

**Cohort A — pre-revision plant (`physics_revision 2`, `nq 40 / nv 37`, obs 83).**
`20260723_204941`, `20260724_140441`. Git `c2accbee` (recorded in
`20260723_204941/provenance.json`). This is a **structurally different robot**: 33 hinge
joints against today's 21, before the arms were welded and the distal tail fused. Its
observation vector is 83 wide. Nothing in cohort A is comparable to anything after it,
and its checkpoints cannot even be loaded against the current plant.

**Cohort B — columnar plant (`physics_revision 4 / policy 6`, obs 61).**
`20260725_194916`, `20260726_191730`, `20260727_130726`. These three form the intended
single-variable `smoothness_weight` ladder 0.1 → 0.7 → 2.0. They *are* comparable to each
other. Note the stage-1 gate also moved (100 → 1900) between the first and the second, but
the gate does not enter the reward, so the returns remain comparable.

**Cohort C — theropod plant (`physics_revision 5 / policy 7`, obs 61).**
`20260728_122755` only, and stage 1 only. Its `stage_config.json` records
`git_commit 081a20208c76931c85ff39eae8a3c0d0f619860f`, which is exactly the tree reviewed
here — so every measurement in this document applies to it without translation. **This run
has no stage 2 or stage 3.** At review time its `stage2/` folder exists and its
`training_summary.txt` lists Stage 1 only: the run is still executing, not abandoned.

### Explicitly not comparable

* **A ↔ B, A ↔ C:** different plant topology and a different observation width. Any
  reading of "stage 1 improved from 2079 to 2651" across `20260724` → `20260725` is
  meaningless.
* **B ↔ C:** the theropod stance moved `healthy_z_range` (0.75, 1.6) → (0.70, 1.55),
  `natural_pitch` 0.05 → 0.027 and every leg `ctrlrange`. `plant_versions.toml:72`
  invalidates all cohort-B checkpoints. Stage-1 returns across B and C differ by a plant
  change *and* a floor change, and cannot be differenced.
* **Within B:** comparable. This is the only clean comparison in the window.
* **Stage 2 forward velocity (3.23 / 2.87 / 2.96 / 6.36 / 4.36 m/s)** spans two cohorts
  and is not a monotone function of anything in the table. See Open Question OQ-2.

No run in this window is "provenance unknown"; every one carries a `plant_identity.json`
and a `stage_config.json` with a git commit.

---

## 3. Findings

Ordered per the review objective: sim-correctness / learning-signal defects first,
one-way doors second.

---

### F1 — Stage 1's reward pays for existence, and a trained policy scores below a statue that stands. **Confirmed.** *(Proposed — reward weights are excluded from this pass.)*

**Claim.** Of the reward available in T-Rex stage 1, 94.9% is `alive_bonus` +
`reward_height`, both of which a constant action collects in full. Every term that could
distinguish a *good* stance from a *passive* one is a small negative correction. The
consequence is that the reward's argmax over the reachable policy space is
"emit zeros and do not fall", and the trained policies land strictly below that.

**Evidence — reward decomposition, zero action, current plant and shipped stage-1 config.**

```
$ python scratch/reward_decomp.py --stage 1 --episodes 40 --seed 3042
T-Rex stage 1: zero action   (40 eps, seed 3042, reset_noise=0.1)
  total return    1743.73 +/- 1275.54   (mean-std 468.19)
  ep length       638.1 of 1000   full-horizon 57%
  terminations    {'truncated': 23, 'nosedive': 13, 'fallen': 4}
  pelvis height   0.9282 +/- 0.0243
  RMS action change/joint  0.0000

  component                    sum/episode    per step  share of |R|
  reward_drift                       -0.27     -0.0004          0.0%
  reward_alive                     1116.76      1.7500         60.5%
  reward_tail                        -1.26     -0.0020          0.1%
  reward_posture                     -5.29     -0.0083          0.3%
  reward_nosedive                   -18.08     -0.0283          1.0%
  reward_height                     635.46      0.9958         34.4%
  reward_heading                     63.44      0.0994          3.4%
  reward_spin                        -0.47     -0.0007          0.0%
  reward_speed                       -4.08     -0.0064          0.2%
  fall_penalty (terminal)           -42.50
  CHECK sum                        1743.73  vs measured 1743.73
```

The decomposition closes exactly against the env's own `reward_total`, so the shares are
not an approximation. `reward_height` runs at **0.9958 per step against
`height_weight = 1.0`** — that is, the height term is still essentially a constant at the
operating point after the July fix, because `target_z = 0.9260`
(`environments/trex/envs/trex_env.py:456`, `121822f`) equals the settled stance
(measured 0.926215) and `height_frac` clips at 1.0 for everything at or above it.

**Evidence — the statue's ceiling.** Removing the reset lottery isolates what "standing
without falling" is worth:

```
$ python - <<'PY'   # 5 eps at reset_noise 0.0, 40 eps at the shipped 0.10
reset_noise=0.0: n=5  return 2845.45 +/- 0.31, len 1000.0, full-horizon 100%
reset_noise=0.1: n=40 return 1743.73 +/- 1275.54, len 638.1, full-horizon 57%
    full-horizon episodes only: return 2834.95 +/- 6.25  (n=23)
    terminated episodes only:   return  267.38 +/- 193.71 (n=17)
PY
```

**Evidence — the trained policy, same commit, same plant.** `20260728_122755/stage1/stage_summary.txt`:

```
Best Model Evaluation (30 episodes)
 Reward: 2404.73 +/- 350.64
 Ep length: 965.9 +/- 132.2 steps
```

| | return | steps | return / step |
|---|---|---|---|
| zero action, standing (full-horizon episodes) | **2834.95** | 1000 | **2.835** |
| zero action, all episodes | 1743.73 | 638.1 | 2.732 |
| trained best model, `20260728_122755` | 2404.73 | 965.9 | **2.490** |

The trained policy beats the *unconditional* statue — which is what the 1840 gate compares
against — by converting falls into survivals. Per surviving step it earns **12% less** than
doing nothing, because it pays `smoothness_weight = 2.0`, `energy_penalty_weight = 0.075`,
`spin`, `speed` and `posture` that a constant action does not.

**Falsification attempt.** Four ways this could have been wrong, all checked:

1. *"The eval reward is VecNormalize-scaled, so the numbers aren't comparable."* — No.
   `environments/shared/train_base.py:317,320,909,919,928` force `eval_env.norm_reward = False`
   on every evaluation path. Eval returns are in raw env units.
2. *"The statue was measured on a different config."* — No.
   `20260728_122755/stage1/stage_config.json` records
   `git_commit 081a20208c76931c85ff39eae8a3c0d0f619860f`, identical to the reviewed tree,
   and its 32 recorded `reward_weights` match `configs/trex/stage1_balance.toml` at HEAD
   field for field (`alive_bonus 1.75`, `smoothness_weight 2.0`, `height_weight 1.0`,
   `nosedive_termination_threshold 0.493`, `healthy_z_range [0.7, 1.55]`,
   `reset_noise_scale 0.1`).
3. *"The seeds favour the statue."* — Both use the run's own `publication_evaluation`
   seed family, 3042 (`provenance.json → evaluation_protocols`).
4. *"Conditioning on full-horizon episodes cherry-picks easy resets."* — It does, and that
   is the comparison being made: *given that both stand, which earns more?* The per-step
   column answers it without conditioning, and agrees (2.490 vs 2.732 against the
   unconditional statue).

The finding survived. What would still kill it: a trained checkpoint whose per-step
stage-1 return exceeds 2.835. None of the six runs has one.

**Corroboration on cohort A.** The same measurement on the cohort-A tree (worktree at
`c2accbee`, that cohort's own config, `reset_noise_scale = 0.05`):

```
trex stage 1: zero-action baseline (40 episodes, reset_noise=0.05)
  reward             2650.30 +/- 680.92
  reward mean-std    1969.39
  episode length     935.5 of 1000
  full-horizon share 92%
```

Both cohort-A runs — `20260723_204941` (best 2419.77, len 844.0) and `20260724_140441`
(best 1755.42, len 605.1) — are **below that floor on all three of reward,
mean-minus-std and full-horizon share**, and both advanced to stage 2 anyway because the
gate was `min_avg_reward = 100.0` at the time. Roughly 19 hours of stage-2/3 GPU time
across the two runs was spent on top of a stage-1 policy worse than `np.zeros(21)`. This
reproduces the claim already recorded in
`environments/shared/scripts/zero_action_baseline.py:5-9`; it is independent
confirmation of an existing diagnosis, not a new one.

**Why this is a finding and not a tuning preference.** The objective for this review is
whether the stage gate reflects the intended behaviour. `stage1_balance.toml:141-200`
argues at length that raising the gate to 1840 makes it bind, and it does — against a
*falling* statue. It does not bind against a *standing* one, and the reward has no term
that separates a policy holding its stance actively from a plant held up by its own
position servos and passive leg springs (`trex.xml:17`, `stiffness="40.0"`).

**What I did not do.** I did not change any weight. Retuning would make every run in
section 2 incomparable, and the choice of which term to add is a design decision. See
Next Steps NS-1 for the specific proposal and its measurement.

---

### F2 — SB3 and MJX terminate T-Rex stages 2 and 3 at different pitch angles. **Confirmed. FIXED in this branch.**

**Claim.** `nosedive_termination_threshold` is set in `configs/trex/stage1_balance.toml:19`
(0.493) but not in the stage-2 or stage-3 TOMLs. The SB3 env then falls back to its class
default of **0.62** (`environments/trex/envs/trex_env.py:100`); the MJX path falls back to
a shared generic default of **0.50** (`environments/shared/mjx_env.py:728`, and the same
literal at `jax_setup.py:562,660` and `jax_reward_termination.py:251`). The two backends
run different termination envelopes for the two stages where a head-forward posture is
explicitly wanted.

**Evidence.** Reproducing the merge in `mjx_env.py:383-427` without needing JAX, then
reading the value `mjx_env.py:728` would resolve:

```
$ python - <<'PY'
...reproduces MJXDinoEnv's registry-then-TOML merge, then weights.get(..., 0.5)
PY
stage     SB3 threshold  MJX threshold   verdict
1                 0.493          0.493   MATCH
2                  0.62            0.5   *** DIVERGES ***
3                  0.62            0.5   *** DIVERGES ***
```

Corroborated from the run side: `20260727_130726/stage2/stage_config.json` records the
SB3-resolved `"nosedive_termination_threshold": 0.62`.

In physical terms, with `natural_forward_z = -sin(0.027)`, SB3 terminates at
`forward_z < -0.647` (40.3° nose-down) and MJX at `forward_z < -0.527` (31.8°) — MJX
kills the episode 8.5° earlier.

**Why it is a defect and not a deliberate backend difference.**
`environments/trex/mjx_config.py:44-46` states the opposite as fact: *"Stages 2 and 3 leave
`nosedive_termination_threshold` at the 0.62 default"*. It then derives 0.734 rad from it —
a figure computed with the pre-theropod `natural_pitch = 0.05`; at the shipped 0.027 the
correct figure is 0.7035 rad. And the neighbouring parameters in exactly this category are
pinned by a dedicated parity class,
`environments/shared/tests/test_species_integration.py:454` ("The Gymnasium env and the MJX
registry must agree on the alive envelope"), which covers `healthy_z_range` and
`max_tilt_angle` and was added by `84ade97` for precisely this failure mode.
`nosedive_termination_threshold` was missed.

**Falsification attempt.** *"Maybe the registry deliberately omits it so the shared 0.5
applies."* — If so, `mjx_config.py:44-46` would not assert 0.62, and the parity class would
not exist. *"Maybe registering it would shadow stage 1's 0.493."* — Checked: the merge at
`mjx_env.py:414-427` applies registry weights **first** and then overlays canonicalised TOML
keys, so stage 1's 0.493 still wins. The hazard flagged for `approach_weight`
(`mjx_config.py:88-90`) applies to *legacy* key names bypassing canonicalisation;
`nosedive_termination_threshold` is the canonical name and appears in the reward-weight key
list at `mjx_env.py:74`. *"Would this move a plant fingerprint and need a revision bump?"* —
No. `plant_contract.py:895-913` builds the policy-interface payload from sensor layout, body
IDs, action mapping and observation callables, with an explicit comment that
*"Reward/termination code stays out of the interface fingerprint"*; `source_closure_sha256`
hashes the MJCF bytes (`plant_contract.py:1415`), not Python.
`python -m environments.shared.plant_contract --check` reports "Plant manifest is current"
on this tree, and the fix commit re-runs it to confirm it still does.

**Fix applied.** One line registering `nosedive_termination_threshold: 0.62` for T-Rex, plus
a correction to the stale comment. Regression test added to the existing parity class;
verified to fail before the fix.

**Cross-species note (objective 3).** The shared MJX default is 0.5 for every species, while
the SB3 defaults differ per species — 0.62 for T-Rex (`trex_env.py:100`), 0.55 for
Dibothrosuchus (`dibothrosuchus_env.py:109`). The same class of gap therefore plausibly
exists for the other three, but auditing them is out of scope for this review. NS-4 proposes
the general fix.

---

### F3 — SB3 pays the alive bonus flat; MJX scales it by height and gates it on foot contact. Same TOML, 3.8× different dominant term. **Confirmed.** *(Proposed.)*

**Claim.** In stage 1, `alive_bonus = 1.75` is the single largest reward term (60.5% of
the zero-action return, F1). The two backends compute it completely differently:

* SB3 — `BaseDinoEnv._reward_alive` (`base_env.py:182-184`) → `reward_alive(self.alive_bonus)`
  → `reward_functions.py:101-103` → returns the constant. No height scaling, no contact gate.
* MJX — `mjx_env.py:601-613`: `r_alive = raw_alive * height_frac * has_foot_contact`, where
  `height_frac = (z - 0.70) / (1.55 - 0.70)` = **0.2659** at the settled stance 0.926215.

Same config, same stage: **1.75/step on SB3, 0.465/step on MJX.** The two paths are not
optimising the same objective, and `healthy_z_range` is a live reward parameter on one path
and a pure termination bound on the other.

**Evidence.** Code as cited. The asymmetry is acknowledged in scattered comments —
`trex_env.py:119-121` computes the 0.266 figure, and `stage1_balance.toml:13-14` marks
`foot_contact_gate` and `foot_contact_weight` "JAX-only" — but no test pins it, and
`environments/shared/scripts/zero_action_baseline.py:60` and the two other diagnostics
*drop* those keys entirely (`JAX_ONLY_ENV_KEYS`) when building the env.

**Why it matters now.** The stage-1 gate `min_avg_reward = 1840` was calibrated with the
SB3 numbers (`stage1_balance.toml:141-161`, reproduced in F1). If a T-Rex run is ever
launched on the MJX path against the same TOML, the gate is calibrated against the wrong
scale by a factor of ~3.8 on the dominant term, and a perfectly good policy fails stage 1 —
which, per `stage1_balance.toml:163-168`, halts the entire curriculum.

**Falsification attempt.** *"Maybe `foot_contact_weight = 0.8` compensates on the MJX side."*
It does not close the gap: MJX adds at most 0.8/step (`mjx_env.py:645-648`) against the
1.285/step it removes from the alive term, and it is paid on a different condition. *"Maybe
the divergence is intended and documented as such."* Partly — the two `foot_contact_*` keys
are labelled JAX-only, but the **height scaling of the alive bonus is not labelled anywhere
as a backend difference**, and `84ade97`'s stated purpose was to *unify* the alive envelope.

**Not fixed** because either direction is a design decision that changes the reward scale
and would invalidate the run comparisons in section 2. NS-3 proposes it.

---

### F4 — MJX recomputes the "forward" reference every step; SB3 freezes it at episode start. **Confirmed.** *(Proposed. Shared code — affects all four species.)*

**Claim.** `TRexEnv` caches `self._initial_prey_dir_2d` in `_spawn_target`
(`trex_env.py:612`) and uses it for both the forward-velocity and heading rewards
(`trex_env.py:349`). The comment at `trex_env.py:183-187` states the reason explicitly:
*"the 'forward' reference direction stays fixed for the whole episode, preventing the
reward from flipping sign when the T-Rex passes the prey."*

`mjx_env.py:576-577` recomputes it every step:

```python
target_rel_2d = target_pos[:2] - pelvis_xpos[:2]
forward_ref = target_rel_2d / (jnp.linalg.norm(target_rel_2d) + 1e-8)
```

so the MJX path has exactly the behaviour the SB3 comment says the design avoids. It feeds
both `reward_forward_velocity` (`mjx_env.py:582`) and `reward_heading_alignment`
(`mjx_env.py:698`).

**Why it bites in stage 2 specifically.** `stage2_locomotion.toml:31` spawns prey at
8–12 m; at the 2.9–6.4 m/s the cohort-B runs reach over a 10 s horizon the animal covers
29–64 m, so it passes the prey mid-episode in most episodes. Past that point the MJX
forward-velocity reward (weight 2.0) inverts and the policy is paid to turn around.

**Falsification attempt.** *"Tracking the target continuously is a legitimate design for a
chase task, not a bug."* — For stage 3 (bite) that is arguable, and I am not calling it
wrong there. For stage 2, whose stated task is "Learn forward walking/running" with the
prey used only as a direction reference and `bite_bonus = 0.0`, sign inversion mid-episode
is not a defensible objective. *"Maybe MJX terminates before the crossing."* — It does not;
nothing in `mjx_env.py:721-740` triggers on passing the target.

**Not fixed** because it sits in `environments/shared/mjx_env.py`, changes the objective for
all four species, and choosing per-stage semantics is a design call. It is also currently
dormant: no T-Rex run in the window used the MJX path.

---

### F5 — 7 of 61 observation elements are privileged, and the layout is a one-way door. **Confirmed.**

`plant_versions.toml:36` says the T-Rex observation is 61 wide. **Verified against the
code**, not trusted: `TRexEnv()._get_obs().shape == (61,)`, built by
`trex_env.py:299-341` and mirrored by `obs_functions.build_bipedal_obs`
(`obs_functions.py:77-134`).

| slots | contents | source | class |
|---|---|---|---|
| 0–20 | 21 hinge joint angles, `qpos[7:]` | joint encoder | **measurable** |
| 21–41 | 21 hinge joint rates, `qvel[6:]` | encoder difference / tachometer | **measurable** |
| 42–45 | pelvis quaternion, `sensordata[6:10]` | `framequat` on the `imu` site — absolute world attitude | **derivable** ¹ |
| 46–48 | pelvis gyro, `sensordata[0:3]` | gyroscope, body frame | **measurable** |
| **49–51** | **pelvis linear velocity, `qvel[0:3]`, world frame** | **free-joint state** | **privileged** |
| 52–54 | pelvis accelerometer, `sensordata[3:6]` | accelerometer, body frame | **measurable** |
| 55–56 | per-foot contact force, `sensordata[10]+[24,25,26]` / `[11]+[27,28,29]` | 4 touch sensors summed per foot | **measurable** |
| **57–59** | **prey unit direction, world frame** | `mocap_pos[0] - xpos[pelvis]`, normalised | **privileged** ² |
| **60** | **prey distance (m)** | `‖mocap_pos[0] - xpos[pelvis]‖` | **privileged** ² |

**Counts: 50 measurable, 4 derivable, 7 privileged.**
The table is the full enumeration: it was generated by walking `model.jnt_qposadr` /
`jnt_dofadr` in joint order and the sensor addresses in `sensor_adr` order, with the row
count asserted equal to `TRexEnv()._get_obs().shape[0]`, so it cannot silently drift from
the code. Slots 0–41 are collapsed here only because all 42 carry the same classification;
per-joint names are `neck_pitch, neck_yaw, head_pitch, tail_1_pitch, tail_1_yaw,
tail_2_pitch, tail_3_pitch, r_hip_pitch, r_hip_roll, r_knee, r_ankle, r_toe_d2..d4`, then
the same seven left-leg joints.

¹ A `framequat` sensor reports absolute world orientation including yaw. Roll and pitch come
free from gravity on any IMU; yaw needs a magnetometer or drifts. Classified derivable with
that caveat rather than privileged.

² The *quantities* — bearing and range to a target — are ordinary robot sensing (camera,
lidar). What is privileged is the **frame**: they are expressed in world coordinates, which
presupposes global localisation of both robot and target. A body-frame version of the same
four numbers would be measurable. This is a layout change, not a sensing change.

**Slots 49–51 are the strict case.** No sensor produces world-frame linear velocity of a body
origin. An on-board estimator gives *body-frame* velocity with drift, and the vertical
component worst of all. The policy is a feedforward MLP with no recurrent state, so it cannot
reconstruct these by integration from slots 46–48 and 52–54 either — the observation is
supplying information the network provably could not derive from the rest of the vector.

**Falsification attempt.** *"Slots 49–51 might be the pelvis centre-of-mass velocity rather
than the body-origin velocity, in which case the reward is measuring the wrong point too."*
Measured and **disproved**: with a randomised `qvel`, `qvel[0:3]` matches `d(xpos_pelvis)/dt`
to 0.0066 and `d(xipos_pelvis)/dt` to 0.0585 — it is the body-frame-origin velocity in world
coordinates, which is what `_compute_forward_velocity` intends. That candidate bug is dead;
the privilege classification is unaffected.

**Why this is a one-way door.** `plant_versions.toml:9-11` and `:40-43` are explicit: changing
the observation layout, width or ordering bumps `policy_interface_revision` for **every**
species, not just T-Rex, and invalidates every checkpoint.

* **Cost now:** one `policy_interface_revision` bump ×4 species, a regenerated
  `plant_manifest.generated.json`, and the loss of T-Rex checkpoints. T-Rex checkpoints
  **were already invalidated on 2026-07-27** by the theropod stance (`plant_versions.toml:72`),
  and the only cohort-C run is a stage-1 that will need re-running against any reward change
  anyway (F1). This is the cheapest this change will ever be.
* **Cost later:** the same bump, plus discarding every stage-2 and stage-3 curriculum result
  built on the current layout, plus re-running the ~13 h/run pipeline for each.

**Measured dependence (was OQ-1, now resolved).** The checkpoint from `20260728_122755`
(the run trained on this exact commit) was downloaded and re-evaluated with each slice
ablated. 30 episodes, seeds 3042–3071, deterministic, VecNormalize loaded:

```
$ python environments/shared/scripts/observation_ablation_report.py trex 1 \
      --model .../20260728_122755/stage1/models/robust_best_model.zip

  condition                              return             delta    ep len   full-hz
  baseline                     2489.65 +/- 107.67                --    1000.0     100%
  root_linvel -> mean           709.97 +/- 998.34   -1779.68 (-71.5%)     808.3      37%
  root_linvel -> zero (raw)     810.50 +/- 873.05   -1679.15 (-67.4%)     799.6      33%
  root_linvel -> body frame    2591.81 +/- 50.24     +102.16 ( +4.1%)    1000.0     100%
  target_dir -> mean           2142.54 +/- 445.09    -347.11 (-13.9%)     886.8      70%
  target_dir -> zero (raw)       55.76 +/- 76.93    -2433.89 (-97.8%)      66.3       0%
  target_dir -> body frame      172.57 +/- 119.07   -2317.08 (-93.1%)     110.5       0%
  target_dist -> mean          2290.41 +/- 499.77    -199.24 ( -8.0%)     939.2      90%
  target_dist -> zero (raw)     167.56 +/- 457.63   -2322.09 (-93.3%)     128.2       7%
  all privileged -> mean       -150.96 +/- 873.70   -2640.61 (-106.1%)     676.4       0%
```

Three results, in order of how much they change the decision:

1. **The door is load-bearing.** Removing all seven privileged slots takes the return
   *negative* and full-horizon survival to 0%, with 25 of 30 episodes ending in
   `head_contact`. This policy does not stand up without simulator state.
2. **The strictly-unmeasurable group needs its information, not its frame.** Mean-substituting
   `root_linvel` costs **−71.5%** and drops survival 100% → 37%. Rotating the same three
   numbers into the **body frame** costs nothing: **+4.1%**, still 100% full-horizon, and with
   a *tighter* spread than baseline (±50.24 against ±107.67). Body-frame root velocity is what
   an on-board state estimator produces, so slots 49–51 can move from **privileged** to
   **derivable** with no retraining at all. That is a materially cheaper door than F5 assumed.
3. **The prey slots are cheap to delete and expensive to re-frame.** Deleting the information
   costs only −13.9% (direction) and −8.0% (distance) — stage 1 sets `forward_vel_weight = 0`
   and both bite weights to 0, so the prey enters only through `heading_weight = 0.1`. But
   rotating them into the body frame costs −93.1%. See the falsification note for why those
   two numbers are not in conflict.

**Falsification attempt.** The +4.1% is the load-bearing number, and the obvious objection is
that the rotation was a no-op — the T-Rex starts at the keyframe's identity quaternion and SB3
adds no yaw noise at reset, so if the animal never tilts, world and body frames coincide and
the policy was never actually challenged. **Measured, and disproved.** Perturbation injected
by each intervention along a baseline rollout, in normalised (post-VecNormalize) units, which
is what the policy sees:

```
  intervention                mean |d|   p95 |d|   max |d|
  root_linvel:mean               0.594     1.150     3.371
  root_linvel:zero               0.606     1.145     3.326
  root_linvel:body               0.409     1.061     1.502     <- not a no-op
  target_dir:mean                1.346     3.612     5.990
  target_dir:zero                9.602    10.705    10.770
  target_dir:body                8.091    10.697    11.495
  target_dist:mean               0.752     1.757     2.012
  target_dist:zero               7.737     9.084     9.133
```

The body-frame rotation of `root_linvel` moves the input **69% as far as full removal at the
mean and 92% as far at p95**. The policy absorbs a perturbation of that size when it preserves
the information and collapses when it does not — which is the definition of depending on the
quantity rather than the frame. The finding survived.

The same table explains the apparent contradiction in result 3: the prey direction is a unit
vector whose components barely vary across an episode, so its running variance is tiny and a
small absolute rotation becomes an **8.1-sigma** input excursion against the direction slice's
own scale (clipped at ±10). `target_dir -> body frame` collapsing is a statement about
normalisation scale, not about the information being precious — which is exactly why the
mean-substitution row, at −13.9%, is the one to quote.

**Caveat on the baseline.** 2489.65 ± 107.67 does not exactly reproduce the run's recorded
`Best Model Evaluation` of 2404.73 ± 350.64 at 965.9 steps. The harness seeds per episode
(`seed + index`, the convention `zero_action_baseline.py:92` already uses) while SB3's
`evaluate_policy` seeds the vector env once and runs 30 episodes from it, so the two draw
different reset sequences. The mean sits 3.5% high and the spread much tighter. Every
condition above shares one seed sequence, so the deltas are unaffected; only the absolute
baseline is protocol-dependent.

**This also sharpens F1.** At **100% full-horizon survival** the trained policy scores
2489.65, against the standing statue's 2834.95 — the policy gives up 345 points, −12.2%,
matching the per-step figure in F1 exactly, now measured at identical survival rather than
inferred from it.

---

### F6 — A T-Rex-only observation fix moves every species' policy fingerprint. **Confirmed.** *(Proposed. Objective-3 risk.)*

**Claim.** `plant_contract.py:910` hashes the *source tokens* of
`mjx_env.build_mjx_observation` into the policy-interface payload:

```python
"jax_observation_callers": {
    "training_reset_and_step": _callable_semantics(mjx_env_module.build_mjx_observation),
    "cpu_evaluation": _callable_semantics(jax_setup_module.make_obs_fn),
},
```

and `_callable_semantics` (`plant_contract.py:311-323`) digests `inspect.getsource(...)`.
That function is shared by all four species, so **any** edit to it moves
`policy_interface_sha256` for all four and forces a `policy_interface_revision` bump on each.

**Evidence that this already happened.** `plant_versions.toml:40-43`, written for the T-Rex
foot-sensor repair (`aa87445`): *"build_mjx_observation is fingerprinted directly and had to
learn to sum, so every species' policy interface revision increments; only the T-Rex's
observation values actually change."* Velociraptor went to `policy_interface_revision 6` and
brachiosaurus/dibothrosuchus to 3 for a change that did not alter a single number in their
observations.

**Why it matters for the near-term goal.** The goal is all four species working in
simulation. Under this coupling, every T-Rex-driven observation change spends a compatibility
counter for the other three, and any of their in-flight checkpoints are marked incompatible
for no behavioural reason. F5's proposed layout change would do it again.

**Falsification attempt.** *"Maybe the fingerprint is per-species because the observation
probe values differ."* — The probe (`plant_contract.py:823-829`) is per-species, but the
`_callable_semantics` token digest is not: identical source, identical digest contribution,
and it changes for everyone whenever the shared function changes. Both feed the same
`_semantic_digest`, so a source-only edit with unchanged probe values still moves the digest.
That is exactly what `plant_versions.toml:40-43` documents happening.

**Not fixed** — narrowing the fingerprint to the per-species-reachable branch of
`build_mjx_observation` is a contract change, not a local bug fix, and CI enforces the
contract. NS-5 proposes it.

---

### F7 — The height-target guard tests exercise a band the shipped config no longer uses. **Confirmed.** *(Proposed. Test hygiene, no production effect.)*

`TestHeightTargetTracksStance` (`environments/trex/tests/test_trex_env.py:298-420`) was added
by `d852669` to stop `target_z` drifting away from the plant's settled stance. All three of
its cases construct the env with `healthy_z_range=(0.75, 1.6)` — the **columnar** band. The
shipped band since `5a93c42` is `(0.70, 1.55)` (`trex_env.py:127`), and neither the stage-1,
stage-2 nor stage-3 TOML overrides it.

The tests still pass, and the property they assert (monotonic decay below the stance, SB3/MJX
agreement) is band-independent, so **nothing in production is wrong**. But the guard runs a
`height_frac` denominator of 0.176 where production uses 0.226, so it no longer measures the
shipped reward.

A second, smaller inaccuracy in the same class: its docstring says *"the target sits a hair
above the settled stance, which leaves a sliver of headroom instead of clipping"*. Measured,
the settled stance is **0.926215** and `target_z` is **0.9260**, so the term clips at exactly
1.0 at the operating point — the headroom is on the other side. This is also why
`reward_height` measures 0.9958/step in F1 rather than something with a gradient.

**Falsification attempt.** *"Fixing the fixture would change a result."* — It would not: with
the production floor of 0.70 the saturation case still yields exactly 1.0 and the monotonic
case still decreases. That is precisely why this is reported as hygiene and **not** fixed
under Step 7: a test-only edit that passes both before and after proves nothing, which fails
the "provable by a test" bar.

---

### F8 — `_compute_initial_direction_2d` assumes the robot spawns at the world origin. **Confirmed. Latent, not live.** *(Proposed.)*

`base_env.py:732-746` normalises `target_pos[:2]` — the direction from the **world origin** to
the prey — and `trex_env.py:612` uses it as the episode-fixed forward reference. It is correct
only while the robot's XY spawn is exactly (0, 0).

Today it is: `BaseDinoEnv.reset` (`base_env.py:850-864`) perturbs `qpos[7:]`, `qvel[:]` and
`qpos[2]`, and never `qpos[0:2]`. So there is **no live error** in any run in section 2.

It is worth recording because the MJX path *does* jitter XY —
`init_qpos_noise = 0.01` in all three stage configs — and computes its own reference from
`pelvis_xpos` instead (F4). If SB3 ever gains XY reset noise for robustness, this silently
mis-aims the stage-2 forward-velocity reward by up to `atan(noise / distance)` with no test
failing. A one-line change to difference against `self.data.qpos[0:2]` removes the trap.

**Falsification attempt.** *"Maybe some config already sets XY noise on the SB3 path."* —
Grepped: `init_qpos_noise` and `init_yaw_noise` appear only under `[jax]` in all three T-Rex
stage TOMLs, and `zero_action_baseline.py:60` and its siblings never pass them to the
Gymnasium env. No live path.

**Not fixed** because it is not currently wrong, which fails the "genuinely a bug" bar.

---

### No one-way door in the action space, the control rate, or the actuator interface

This is a positive result and worth recording as a baseline.

* **Control rate.** `frame_skip = 5` × `timestep = 0.002` = **100 Hz** policy rate
  (`trex.xml:4`, `trex_env.py:69`, `mjx_config.py:38`). Well inside what a physical joint
  controller accepts; no policy would need re-training for rate reasons.
* **Action space.** 21 normalised position targets, one per actuator, in `Box(-1, 1)`
  (`base_env.py:112-117`). Verified against the compiled model: 21 actuators, each a
  `<position>` driving exactly one hinge, no coupled or virtual transmissions.
* **Action mapping.** `home-keyframe-residual/v1` (`trex_env.py:62,282-297`): action zero
  commands the named `home` keyframe control vector, and ±1 reach the actuator endpoints on
  each side independently. Verified that the mapping does real work — `neck_pitch_act`
  (`ctrlrange [-0.5236, 0.6981]`, home 0) and `head_pitch_act` (`[-0.4363, 0.6109]`, home 0)
  are asymmetric about home, so the two-sided scaling is not a no-op. This is a *more*
  hardware-friendly convention than midpoint mapping, not less: it makes "do nothing" mean
  "hold the nominal pose".
* **Actuator idealisation.** `kp` 60–1500, `forcerange` up to ±2250 N·m on an 85.7 kg plant —
  far beyond any real servo. Per the review's own framing this is a deliberate
  simplification, not a finding, and it is not a one-way door: derating actuators later
  changes a number in the MJCF and costs a physics-revision bump, not a rebuild of anything
  downstream.

**The only one-way door found is F5 (observation layout), with F6 as its multiplier.**

---

### Candidates investigated and dismissed

Recorded so the next review does not re-open them.

| candidate | verdict |
|---|---|
| `qvel[0:3]` is the pelvis **CoM** velocity, so `forward_vel` measures the wrong point | **Disproved.** Measured against finite differences: matches `d(xpos)/dt` to 0.0066, `d(xipos)/dt` to 0.0585. |
| The raw, unclipped action reaches `_get_reward_info` (`base_env.py:783`) while `_scale_action` clips (`base_env.py:765`), so energy/smoothness could be charged on magnitude the plant never sees | **Real in code, not live.** SB3 clips in `on_policy_algorithm.py:214-218` and `policies.py:379`; the JAX trainer at `jax_trainer.py:365`. Already recorded by PR #467 and in `action_bound_report.py:217-220`. A direct caller that skips the clip would hit it. |
| Stage 2's 6.36 m/s (`20260726_191730`) is a reward exploit | **No evidence.** `20260727_130726/stage2/stage_config.json` confirms `forward_vel_max = 2.5`, so the forward term saturates at 2.5 m/s and speed above it earns exactly nothing. Nothing rewards it and nothing penalises it. See OQ-2. |
| Prey and body geoms could collide and corrupt the bite proxy | **No.** `prey_geom` and `head_bite` are both `contype=2 conaffinity=2` (`trex.xml:48-49,168-169`); every other body geom is 1/1 and the floor is 1/1, so the prey interacts with the bite box and nothing else. |
| Sensor index constants have drifted from the MJCF | **No.** Compiled model: `pelvis_gyro@0`, `accel@3`, `framequat@6`, `r/l_foot_touch@10,11`, `tail_tip_angvel@21`, per-digit touch `@24–29`. All six constants in `mjx_config.py:24-34` and `trex_env.py:264-273` match. |
| `body_ids={"pelvis": 2}` in `mjx_config.py:65` is stale | **No.** Compiled body order is `world=0, prey=1, pelvis=2`. Also enforced at `plant_contract.py:734-742`. |
| Stage 3 passes its 50% success gate on a degenerate strategy | **No evidence of one.** `20260727_130726/stage3/stage_summary.txt` reports 100% success at 389.0 ± 182.7 steps with prey at 2–6 m and ~1 m/s — consistent with actually walking to the prey and touching it. Success is real geom contact on the SB3 path (`trex_env.py:581-591`), not a proximity proxy. |

---

## 4. Open questions

**OQ-1 — RESOLVED.** *Does the trained policy actually use the privileged observation slots?*
Yes, heavily — and the frame turns out not to matter for the group that needed it most. Full
result, falsification attempt and caveats are in F5; the tool is
`environments/shared/scripts/observation_ablation_report.py`. Short version: all seven
privileged slots removed → return goes negative and survival to 0%; the three world-frame
root-velocity slots removed → −71.5%; the same three rotated into the body frame → +4.1%,
i.e. free.

**OQ-2 — Is the stage-2 gait at 4.4–6.4 m/s a gait, or a ballistic artefact?**
`20260726_191730` reaches 6.36 m/s and `20260727_130726` 4.36 m/s, both at full horizon,
both against a reward that saturates at 2.5 m/s and penalises neither. Froude number at
6.36 m/s with a 0.877 m hip is 4.7 — high, but the actuators are strong enough for it (see
"Actuator idealisation"). Nothing in the numbers distinguishes a real running gait from a
bounding/ballistic one.
*What would resolve it:* `stage2/trex_ppo_stage2_best.mp4` plus a duty-factor and
flight-phase count from `diagnostics.npz` foot-contact traces. Both are in Drive; same
download constraint as OQ-1.

**OQ-3 — Does cohort C reproduce cohort B's stage-1 result?**
`20260728_122755` is the only run on the theropod plant and it is still executing. Its
stage 1 (best 2423.83) is 229 below cohort B's best (2652.63 on the columnar plant), but the
two are not comparable (§2) and its zero-action floor is also lower (1743.73 vs 1800.56
recorded in `stage1_balance.toml:149-150`).
*What would resolve it:* the run finishing, plus one more seed.

**OQ-4 — Was the `smoothness_weight` ladder actually monotone in cost?**
`stage1_balance.toml:50-55` predicts higher smoothness costs reward and stability. Within
cohort B the stage-1 finals are 2651.80 (w=0.1) → 2503.81 (w=0.7) → **2654.66** (w=2.0) —
the strongest weight scored *highest*, and at 1000.0 ± 0.0 episode length with a ±12 spread.
That contradicts the config's own sizing argument, which expected ~2340 before adaptation.
*What would resolve it:* `compare_run_diagnostics.py` across the three cohort-B runs on
`diagnostics/action_delta` and `algo_std`, to separate "the penalty bound and the policy
adapted" from "the penalty never bound". Requires the `diagnostics.npz` files.

**OQ-5 — Do the other three species carry F2's gap?**
The MJX default of 0.5 is shared and every SB3 default differs (T-Rex 0.62, Dibothrosuchus
0.55). Not audited — out of scope for this review by instruction.
*What would resolve it:* running the F2 probe for each species, or the general fix in NS-4.

---

## 5. Next steps, ranked

**NS-1 — Give stage 1 a term the statue cannot collect. (Addresses F1.)**
*Change:* add a stance-quality term that a constant action scores near zero on — the
strongest candidate is a centre-of-pressure / support-polygon margin term, since the plant
already carries per-digit touch sensors that make CoP computable; a cheaper stand-in is
re-enabling `reset_noise_scale` sweeps so the reward has to buy recoveries rather than luck.
*Touches:* `configs/trex/stage1_balance.toml` `[env]`, plus a new helper in
`environments/shared/reward_functions.py`.
*Expected:* the zero-action full-horizon score drops well below the trained score; the
per-step comparison in F1 inverts.
*Measurement that decides it:* re-run `zero_action_baseline.py trex` and the F1 per-step
table. Success is the trained per-step return exceeding the standing statue's, on the same
seed family. Until that inverts, stage 1 is not measuring balance.
*Note:* this makes future runs incomparable with §2. Do it once, deliberately, and re-baseline.

**NS-2 — Re-derive `min_avg_reward` against the *standing* floor, not the falling one.**
*Change:* `stage1_balance.toml:141`. The current 1840 is +5.5% over the all-episode
zero-action mean (1743.73). The standing-statue score is 2834.95.
*Expected:* a gate that a statue cannot clear even when it survives.
*Measurement:* the gate must sit above the full-horizon zero-action return under the NS-1
reward, and the cohort-C run must still clear it. Do NS-1 first — raising the gate against
today's reward would just fail every run.

**NS-3 — Decide the alive-bonus semantics once, and pin it with a test. (F3.)**
*Change:* either scale SB3's `_reward_alive` by `height_frac` to match `mjx_env.py:602-613`,
or drop the scaling from MJX. Then extend the parity class at
`test_species_integration.py:454` to assert equal per-step alive reward at the settled stance.
*Expected:* one number instead of two; the stage-1 gate calibration becomes backend-independent.
*Measurement:* the new parity test, plus re-running `zero_action_baseline.py` to re-derive
the gate at whatever scale is chosen.

**NS-4 — Make per-species termination parameters registry-required rather than defaulted. (F2, OQ-5.)**
*Change:* drop the `0.5` literal at `mjx_env.py:728` (and `jax_setup.py:562,660`,
`jax_reward_termination.py:251`) in favour of a required registry field, so a species that
forgets to declare it fails loudly at registration instead of silently getting a generic value.
*Expected:* the same class of gap in the other three species surfaces immediately.
*Measurement:* the test suite; any species missing the field raises at import.

**NS-5 — Narrow the shared observation fingerprint. (F6.)**
*Change:* `plant_contract.py:908-912` — fingerprint the per-species-reachable behaviour of
`build_mjx_observation` (the existing per-species observation probe already does this) rather
than the whole function's source tokens.
*Expected:* a T-Rex-only observation change stops bumping `policy_interface_revision` for
brachiosaurus, velociraptor and dibothrosuchus.
*Measurement:* regenerate `plant_manifest.generated.json` after a no-op edit to
`build_mjx_observation` and confirm only T-Rex's digest moves.

**NS-6 — Rotate slots 49–51 into the body frame. Cheapest item on this list. (F5, OQ-1.)**
*Change:* `environments/trex/envs/trex_env.py:313` and `environments/shared/obs_functions.py:113`
— rotate `qvel[:3]` by the transpose of the pelvis rotation before concatenating.
*Expected:* the last strictly-unmeasurable slots in the T-Rex observation become body-frame
root velocity, which is what an on-board state estimator produces. **The existing checkpoint
already tolerates this: +4.1% with 100% full-horizon survival, measured (F5).** So this is a
provenance improvement that costs no policy performance, and it is a `policy_interface_revision`
bump the repo is otherwise going to spend anyway.
*Measurement:* re-run `observation_ablation_report.py` on the retrained policy; the
`root_linvel -> body frame` row should collapse to ~0% delta (it becomes the identity), and
`root_linvel -> mean` should still cost heavily — confirming the information is intact.
*Cost note:* T-Rex checkpoints are already invalid as of `5a93c42`, and NS-1 will invalidate
the reward anyway. Doing this in the same re-baseline costs one bump instead of two.
*Caveat:* per F6 this bumps `policy_interface_revision` for the other three species too. Doing
NS-5 first makes it free for them.

**NS-6b — Leave the four prey slots alone for now.** Same finding, opposite conclusion:
deleting them costs only 8–14%, but re-framing them into body coordinates costs −93% because
their normalised variance is tiny (F5). There is no cheap version of this change, and stage 1
barely uses them. Revisit when stage 2/3 provenance matters — those stages weight the prey far
more heavily, so the 8–14% figure does not carry over and would need re-measuring per stage.

**NS-7 — Refresh `TestHeightTargetTracksStance` to the shipped band, and fix its docstring. (F7.)**
*Change:* `environments/trex/tests/test_trex_env.py:316,332,355,384` — drop the hardcoded
`healthy_z_range=(0.75, 1.6)`.
*Expected:* no behaviour change; the guard starts measuring the shipped reward.
*Measurement:* the suite stays green (verified — the assertions are band-independent).

**NS-8 — Make `_compute_initial_direction_2d` spawn-relative. (F8.)**
*Change:* `base_env.py:732-746` — difference against the root XY instead of assuming the origin.
*Expected:* no change today; removes a silent failure mode if SB3 ever gains XY reset noise.
*Measurement:* a test that sets a non-zero root XY and asserts the reference points at the prey.

**NS-9 — Wire `zero_action_baseline.py` in as an automatic pre-stage check.**
Already suggested by `stage1_balance.toml:159-161`. Cohort A shows the cost of not having it:
two runs, ~19 h of stage-2/3 GPU time, on top of a stage-1 policy worse than `np.zeros(21)`,
because the gate was 100.0 and nothing compared against the floor.

---

## 6. Fixed vs proposed

**Fixed in this branch (one commit each, after this report was committed):**

| finding | change | test |
|---|---|---|
| **F2** | Register `nosedive_termination_threshold = 0.62` for T-Rex in `environments/trex/mjx_config.py`, and correct the stale comment that asserted it was already in force | New case in `test_species_integration.py`'s SB3/MJX parity class; verified to fail before the fix |
| **F5 / OQ-1** | Added `environments/shared/scripts/observation_ablation_report.py` — the diagnostic that resolved OQ-1. Not a fix; it is the tool the finding needed, in the same family as `zero_action_baseline.py` and `action_bound_report.py`, so the numbers in F5 are reproducible rather than one-off | No behaviour change; lint and both suites green |

**Deliberately not fixed, and why:**

| finding | why not |
|---|---|
| **F1** | Reward-weight tuning, excluded by instruction, and it makes every run in §2 incomparable. Needs the NS-1/NS-2 pair done together and re-baselined. |
| **F3** | Either direction changes the reward scale on one backend; a design decision, not a bug fix. |
| **F4** | Shared code affecting all four species, and the per-stage semantics (chase vs straight-line) are a design call. |
| **F5** | One-way door — your call by instruction. Also blocked on OQ-1. |
| **F6** | Contract change enforced by CI, not a local fix. |
| **F7** | Test-only. The corrected test passes both before and after, so it fails the "provable by a test" bar. |
| **F8** | Latent, not live. Fails the "genuinely a bug" bar. |
| all MJCF work | `environments/trex/assets/trex.xml` is off-limits: it moves the physics/visual fingerprints, needs a revision bump plus a regenerated manifest, and invalidates checkpoints. Nothing in this review required it. |

---

## Appendix — reproducing the measurements

Environment: `pip install -e ".[train]"`.

On a first attempt in a clean container this failed while building `cloudml-hypertune`, and
an earlier draft of this document recorded that as the package being unbuildable. **That was
wrong, and the correction matters because the obvious response — dropping the dependency —
would have been the wrong fix.** `cloudml-hypertune` is on PyPI (`0.1.0.dev6`, sdist only,
no wheel) and is a real dependency: `environments/shared/train_base.py:838` imports it to
report `best_mean_reward` to Vertex AI hyperparameter tuning, which is the mechanism
`website/docs/training/sweeps.md:405-412` documents for picking the best sweep trial. The
import sits behind `try/except ImportError` with a "not installed" log line
(`train_base.py:851`), so removing it would not fail loudly — sweeps would just silently
stop reporting their objective.

The actual failure is environmental:

```
File "/usr/lib/python3/dist-packages/setuptools/command/install_lib.py", line 17,
    in finalize_options
  self.set_undefined_options('install', ('install_layout', 'install_layout'))
AttributeError: install_layout. Did you mean: 'install_platlib'?
```

That is Debian's *patched* setuptools (68.1.2, in `/usr/lib/python3/dist-packages`), whose
`install_lib` references a Debian-only `install_layout` option that plain `distutils` does
not define — it breaks legacy `setup.py`-based sdists generally, not this package. Once a
pip-installed setuptools shadowed the system one in `/usr/local/lib`,
`pip install -e ".[train]"` succeeded and `import hypertune` works. **Workaround:
`pip install -U setuptools` before installing the extra.** Nothing in the repo needs changing.

Two toolchain notes found while running the pre-commit gate on this tree (`081a2020`),
neither caused by this review and neither breaking CI today:

* **CI and `.pre-commit-config.yaml` do not run the same linters.**
  `.github/workflows/python-ci.yml:50` installs **unpinned** `ruff mypy`, while the hook
  config pins `ruff-pre-commit v0.4.4` and `mirrors-mypy v1.15.0`. On this tree the CI
  commands are clean (`ruff check`: all passed; `ruff format --check`: 172 files already
  formatted; `mypy`: no issues in 172 files), but `pre-commit run --all-files` reports
  `ruff-format` wanting to reformat 12 test files and 3 mypy errors
  (`curriculum.py:741`, `species_catalog.py:133`, `jax_trainer.py:1232`). A contributor who
  trusts the local hook will "fix" files CI is happy with, and vice versa. Pinning CI to the
  hook revisions — or unpinning the hooks — removes the divergence.
* `end-of-file-fixer` rewrites `website/static/img/logo.svg` and `logo-dark.svg` on every
  run. Out of scope here (`website/**`), but it means `pre-commit run --all-files` never
  exits clean on a fresh checkout.

```bash
# Zero-action floor, current plant + shipped stage-1 config  (§F1, reproduces the
# 1743.73 +/- 1275.54 recorded at configs/trex/stage1_balance.toml:142)
python environments/shared/scripts/zero_action_baseline.py trex --episodes 40 --seed 3042

# Untrained-Gaussian action saturation  (reproduces action_bound_report.py:44-49)
python environments/shared/scripts/action_bound_report.py trex 1 --std 1.0 \
    --episodes 10 --seed 3042
#   |a| > 1     32.3% of components
#   |a| >= 0.99 32.7%

# Observation-dependence ablation on a trained checkpoint  (§F5, resolves OQ-1)
python environments/shared/scripts/observation_ablation_report.py trex 1 \
    --model <run>/stage1/models/robust_best_model.zip --episodes 30 --seed 3042
#   root_linvel -> mean        -71.5%
#   root_linvel -> body frame   +4.1%
#   all privileged -> mean    -106.1%

# T-Rex suite
python -m pytest environments/trex/tests -q          # 77 passed
```

The checkpoint used for F5 is
`MyDrive/mesozoic-labs/logs/trex/ppo/20260728_122755/stage1/models/robust_best_model.zip`
(4,023,328 bytes, `sha256:7e4255aa0e496fd1713de79069eafc5827ab8892bd7b132f0db873f5aff98b73`)
with its `robust_best_model_vecnorm.pkl` sidecar (4,758 bytes). The sidecar's embedded
`_mesozoic_plant_identity` reads `physics_revision 5 / policy_interface_revision 7`, which is
the plant at this commit — so the ablation ran against the plant it was trained on, with no
compatibility shim.

The three ad-hoc scripts used above — the per-component reward decomposition, the
observation-provenance enumeration, and the SB3/MJX threshold probe — are reproduced inline
in the findings that cite them. The cohort-A floor in F1 was measured in a detached worktree
at `c2accbee` with the current `zero_action_baseline.py` copied in (the script did not exist
at that commit; it was added by `d6f44c1`).
