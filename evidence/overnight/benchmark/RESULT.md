# BENCHMARK: is 1.0 m/s a valid walking benchmark for the r10 T. rex?

Stage folder: `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/overnight/benchmark/`. All paths below are relative to it.
Worked 02:12-02:33 UTC on 2026-11-04 (ANALYSE stage, write-up due 08:30 UTC).

What I did and did not do:
- Analysis only. Nothing was trained or fitted. The repository, the main 4d5f4e7 export, its configs and the physics were not changed.
- Every hypothetical reward setting (forward_vel_max 1.0 and 1.2) was recomputed offline from the recorded per-step `info["forward_vel"]`. No env, config or model was edited, in memory or on disk.
- The forward weight 0.2 (ramp start) was also recomputed offline. It is exact because the dynamics do not depend on the weight.
- Simulations ran one process at a time through `scripts/sim.sh`: nice -n 10, OMP_NUM_THREADS=1, TMPDIR in this folder, the optB venv python with -I -B.
- Every env was the export's own `build_env(ModelId(species, "anatomical"), "locomotion")`, driven only by `env.reset(seed)` and `env.step(action)`. Every walker action was asserted to lie in [-1, 1].
- No extra packages were needed (numpy 2.5.4, mujoco 3.10.0 are in the optB venv), so I created no venv.

Units: m, s, m/s. Fr = v²/(gL), v̂ = v/√(gL), g = 9.81. "Episode" means one 10 s, 1000-step locomotion episode; "return" means the undiscounted sum of rewards over it.

## Short answer

1. **As a speed, 1.0 m/s is a walking speed for this body.** It is a moderate one, not a slow one. [measured L; inferred reading]
   - With the probe's leg length L = 0.8766 m it is Fr 0.116 (v̂ 0.34).
   - That is below the speed matching human preferred walking (Fr ≈ 0.25, 1.47 m/s here) and well below the walk-run transition (Fr ≈ 0.5, 2.07 m/s here).
   - It is above the Froude-converted preferred walking speed published for full-size T. rex: about 0.64-0.69 m/s on this model [literature, recalled, unverified: van Bijlert, van Soest & Schulp 2021].
   - The plant can walk at that speed. The SPEED stage's alternating walk V18 held a steady 0.98 m/s on 10 of 10 new seeds, with probe class "walk", floor alternation ≥ 0.919 and no flight. [measured: out/bm_summary.json]
2. **As the gate is written, it asks a walk for more than 1.0 m/s.** The gate averages `info["forward_vel"]`, the pelvis velocity along the prey direction fixed at reset, over the whole episode, including the start from standing.
   - The hand-built walks average only 0.75-0.91 of their steady speed over an episode. With their start-ups the gate therefore needs a steady speed of about 1.30-1.33 m/s (Fr ≈ 0.20). [measured: out/fv_spread.json]
   - No strict hand-built walk passed. V35, steady 1.17 m/s, averaged 0.88 over the episode. V27, steady 1.32 m/s with rare landing bounces, averaged 0.992 and passed on 5 of 10 seeds. [measured]
   - A walk that reached its steady speed with a 0.5-3 s linear ramp would need 1.03-1.18 m/s steady. [inferred: arithmetic]
3. **The gate checks no gait, and Froude cannot stand in for one.** The September runs that passed it hopped at 1.05 and 1.57 m/s, which is Fr 0.13-0.29: well inside the walking band by Froude. [doc claim speeds; measured L]
   - The reward pays a hop and a walk the same at the same speed. No term with a non-zero weight reads feet, contact or flight (REWARD stage; this stage's term sums agree).
4. **The reward does not aim at 1.0 m/s.**
   - **Ramp end** (forward weight 2.0, cap 2.5 m/s): the return rises linearly with speed across every walk measured. In steady walking that is +771 to +881 per 10 s per m/s, against 800 from the formula alone. It keeps rising to the cap at 2.5 m/s, Fr 0.73, past the walk-run transition. [measured slopes; inferred extrapolation]
   - **Ramp start** (forward weight 0.2): speed is worth only 80 per episode per m/s. The 3-5 cm crouch that the faster hand-built walks use costs about as much as their speed earns.
     - The crouched 0.4 and 0.55 m/s walks lose to the statue (−41, −16.5 per episode; 0 of 10 seeds).
     - The upright 0.43 m/s walk wins (+18).
     - The non-falling walks at 0.75 m/s or more (episode mean) win by +13 to +35.
     - Discounted from reset, every crouched walk is below the statue at the ramp start.
     [measured]
5. **Hypothetical caps** (offline; everything else unchanged).
   - **Below the cap, they pay 2.1x (cap 1.2) or 2.5x (cap 1.0) more per m/s.** At the ramp start every non-falling walk except the crouched 0.4 m/s one then beats the statue on 10 of 10 seeds, by +21 to +115. The crouched one gets −6 with cap 1.2 and +7 with cap 1.0.
   - **Above the cap, they stop paying, but only gradually.** The pelvis speed swings ±25-35% within a stride (p5/p95 0.73/1.30 m/s at a steady 0.98). With a 1.0 cap, steady walking still earns +648 per 10 s per m/s between 0.98 and 1.17 m/s, and +323 between 1.17 and 1.32. With a 1.2 cap the figures are +1006 and +706. [measured]
   - Neither cap changes that a hop and a walk at equal speed earn the same.
   - A 1.0 cap saturates the per-step pay at the speed the gate wants as an episode mean. A policy must cruise above 1.0 to pass, and there the 1.0 cap pays little more. [inferred]

**Bottom line** [inferred]: 1.0 m/s is a sensible walking speed for this body, but the gate as written is not a walking benchmark.
- It measures an episode mean from a standing start, so a walk effectively needs about 1.05-1.33 m/s steady. The low end assumes an ideal start; the high end is what these hand-built walks need. The top of that range sits at the edge of the strict hand-built walk envelope (SPEED: strict to 1.2 m/s, with rare bounces to 1.35-1.5).
- It accepts any gait.
- The reward behind it keeps paying speed linearly to 2.5 m/s, past the walk-run transition.

If the intent is "walks at a firm pace", the evidence points to three changes:
- state the speed over a steady window (for example t ≥ 2 s);
- add a gait check from the frozen probe (floor alternation ≥ 0.9, flight fraction near 0, double support > 0). The non-falling hand-built walks score ≥ 0.919 / ≤ 0.0022 / 0.07-0.18 on these [measured]; the September hops scored alternation 0.010 and 35-43% airborne [doc claim];
- if the reward should stop pulling past a brisk walk, cap the forward pay near 1.2-1.5 m/s, ideally on a stride-smoothed speed.

These are options for the maintainer, not decisions. They are evidence-based but untested in training.

## 1. Froude table

L was measured three ways on the built locomotion envs:
- the frozen gait probe's own `RobotDescription.leg_length`: mean world z of the hip_pitch joint anchors at the 'home' keyframe;
- the same anchors on the zero-action statue after settling (mean over t = 2 s to the end, seeds 3042-3051, 10/10 full episodes for every species);
- the statue's CoM height.

The probe's L and the settled hip height agree to 0.5 mm or better. [measured: out/measure_L.json, out/measure_L_stdout.txt]

| | mass (kg) | L probe (m) | L settled hips (m) | CoM settled (m) | √(gL) (m/s) | gate (m/s): Fr / v̂ | forward_vel_max (m/s): Fr / v̂ | v at Fr 0.25 | v at Fr 0.5 |
|---|---|---|---|---|---|---|---|---|---|
| T. rex (r10) | 85.72 | 0.8766 | 0.8771 | 0.8671 | 2.932 | 1.0: **0.116 / 0.341** | 2.5: 0.727 / 0.853 | 1.466 | 2.074 |
| compsognathus | 1.000 | 0.2444 | 0.2445 | 0.2283 | 1.548 | 0.08: 0.0027 / 0.052 | 0.25: 0.026 / 0.161 | 0.774 | 1.095 |
| velociraptor | 13.50 | 0.5055 | 0.5055 | 0.4631 | 2.227 | 2.0: 0.807 / 0.898 | 2.5: 1.260 / 1.123 | 1.113 | 1.575 |

[measured L, mass; config gate and cap read from each species' anatomical locomotion.toml and confirmed on the built env (`forward_vel_max`, `forward_vel_weight`)]

- Fr 0.25 is human preferred walking and Fr 0.5 the walk-run transition [literature, recalled, unverified: Alexander 1989, Physiol. Rev.; Alexander & Jayes 1983, J. Zool.].
- The three gates span Fr 0.003 (compsognathus: a shuffle-level speed), 0.12 (T. rex: a moderate walk) and 0.81 (velociraptor: above the walk-run transition, a run). [inferred from the table]
- On the T. rex body the same Froude numbers are:
  - the compsognathus gate: 0.15 m/s;
  - the compsognathus cap: 0.47 m/s;
  - the velociraptor gate: 2.63 m/s.
- Against the T. rex config comment (locomotion.toml:139-148):
  - The comment says 1.0 m/s is "Froude 0.14-0.16". With the probe's hip-height L it is 0.116, and 0.118 with the settled CoM height. The comment's figure implies L = 0.64-0.72 m.
  - The comment says the velociraptor runs its 2.0 m/s gate "at Froude ~1.07". With this export's velociraptor L = 0.5055 m it is Fr 0.81 (0.88 with CoM height), and the dynamically similar T. rex speed is 2.63 m/s, not 2.7-3.0.
  - The comment's direction holds: 2.0 m/s was a raptor copy at Fr 0.465 here. Its numbers are somewhat off. [measured vs config comment]

### T. rex speeds on one axis (L 0.8766 m)

| v (m/s) | Fr | v̂ | what | label |
|---|---|---|---|---|
| 0.187 / 0.211 | 0.004 / 0.005 | 0.06 / 0.07 | FINAL3 walk, episode mean / steady 5-10 s | measured |
| 0.280 / 0.308 | 0.009 / 0.011 | 0.10 / 0.11 | f2v0.25 walk | measured |
| 0.430 / 0.507 | 0.022 / 0.030 | 0.15 / 0.17 | V05 walk, 0.5 command, upright | measured |
| 0.401 / 0.473 | 0.019 / 0.026 | 0.14 / 0.16 | V23 walk, 0.5 command, crouched | measured |
| 0.554 / 0.682 | 0.036 / 0.054 | 0.19 / 0.23 | V24 walk, 0.7 command | measured |
| 0.64-0.69 | 0.048-0.056 | 0.22-0.24 | full-size preferred walk 1.28 m/s, Froude-converted | literature, recalled, unverified: van Bijlert, van Soest & Schulp 2021, R. Soc. Open Sci. |
| 0.751 / 0.979 | 0.066 / 0.112 | 0.26 / 0.33 | V18 walk, 1.0 command | measured |
| 0.882 / 1.171 | 0.091 / 0.159 | 0.30 / 0.40 | V35 walk, 1.2 command | measured |
| 0.992 / 1.322 | 0.114 / 0.203 | 0.34 / 0.45 | V27 walk, 1.35 command (landing bounces) | measured |
| **1.0** | **0.116** | **0.341** | **gate min_avg_forward_vel (episode mean)**; also the HYPOTHETICAL cap 1.0 | config |
| 1.05 | 0.128 | 0.358 | September hop 20260914_123816 (gate pass) | doc claim, GAIT_QUALITY_PLAN_2026_09.md:86 |
| 1.2 | 0.168 | 0.409 | HYPOTHETICAL cap 1.2; SPEED's strict-walk ceiling | hypothetical; SPEED RESULT.md |
| 1.466 | 0.25 | 0.50 | human preferred walk by Froude | literature, recalled, unverified: Alexander 1989 |
| 1.57 | 0.287 | 0.535 | September hop 20260925_033501 (gate pass) | doc claim, GAIT_QUALITY_PLAN_2026_09.md:89 |
| 2.0 | 0.465 | 0.682 | the old gate (raptor copy) | config comment |
| 2.074 | 0.5 | 0.707 | walk-run transition by Froude | literature, recalled, unverified: Alexander & Jayes 1983 |
| **2.5** | **0.727** | **0.853** | **forward_vel_max: the reward pays speed linearly up to here** | config |
| 2.65-2.86 | 0.82-0.95 | 0.90-0.98 | full-size maximum speed about 5.3 m/s, converted | literature, recalled, unverified: Sellers, Pond, Brassey, Manning & Bates 2017, PeerJ |
| 2.50-5.95 | 0.73-4.1 | 0.85-2.0 | full-size maximum speed about 5-11 m/s (fast running at about 20 m/s ruled out), converted | literature, recalled, unverified: Hutchinson & Garcia 2002, Nature; Hutchinson 2004, J. Morphol. |
| 4.00-4.32 | 1.9-2.2 | 1.36-1.47 | full-size maximum about 8 m/s, converted | literature, recalled, unverified: Sellers & Manning 2007, Proc. R. Soc. B |

[measured: out/froude_stdout.txt, out/froude.json; walk speeds are `info["forward_vel"]` means, out/bm_summary.json]

Literature conversion: v_model = v_full · √(0.8766 / L_full).
- I assumed an adult hip height L_full of 3.0-3.5 m [literature, recalled, unverified; no single paper checked].
- Inverse: this model's 1.0 m/s corresponds to 1.85-2.0 m/s at full size, about 1.5 times the recalled preferred-walking estimate. That is a brisk walk for the animal, still far under its recalled top-speed estimates. [inferred]
- Every literature figure here is from memory and unchecked.
- Froude similarity matches only gravity and length. It says nothing about whether this plant's servos, damping and 85.7 kg mass make a given Fr as easy or as hard as for the animal. A geometrically similar ~7 t animal scaled to L = 0.88 m would weigh ~110-175 kg [inferred from a recalled adult mass of 6-9 t; unverified].

The forward-velocity cap compared with the literature [inferred]: forward_vel_max 2.5 m/s (Fr 0.73) is past the walk-run transition. It is near the low end of the converted published maximum-speed estimates for the animal. So the reward keeps paying speed from a brisk walk up to roughly the animal's estimated top speed.

## 2. Reward versus walking speed

### Setup

Seeds and behaviours:
- Reset seeds 3042-3051, 10 episodes per behaviour, on the export's T. rex locomotion env as built (forward_vel_weight 2.0, forward_vel_max 2.5).
- Each episode ran through the frozen gait probe's own `rollout_episode`. It records the env's per-step reward, every `info` `reward_*` term and `info["forward_vel"]`. `analyze_episode` then gave the gait class.
- The walkers were:
  - the EXISTENCE `wlib.Walker` (FINAL3, f2v0.25; parameters from `feas/existence/out/<tag>_tyrannosaurus_rex_s0.json`);
  - the SPEED stage's `wlib2.Walker2` at its stable settings, parameters from `overnight/speed/out/<tag>.json`: V01 (f2v0.25 without k_yp), V05 (0.5 m/s, upright), V23 (0.5, crouched), V24 (0.7), V18 (1.0), V35 (1.2), V27 (1.35), V29 (1.5).

Offline recompute: NF = Σ (reward − reward_forward), which includes the fall penalty. For any weight w and cap vmax, R(w, vmax) = NF + w · Σ clip(fv/vmax, −1, 1).

Checks [measured: out/analyze_bm_stdout.txt]:
- R(2.0, 2.5) matches the env's summed reward on every episode to within 4.6e-13.
- The recomputed reward_forward matches the env's on every step exactly (0.0).
- The statue, FINAL3 and f2v0.25 reproduce the REWARD stage's measured returns at w 0.2 and 2.0 to within 2.3e-12 on all 10 shared seeds. REWARD set the weight in the env, so the offline w-recompute is validated.
- The SPEED walker V35 reproduced SPEED's final root position bitwise on seeds 0 and 1 [measured: out/chk_V35_s01.json].

### Gait and speed on these seeds

| walk | full | probe class (t ≥ 1 s) | alternation min | flight max | episode-mean v (gate measure) | steady v 5-10 s | gate (mean ≥ 1.0) |
|---|---|---|---|---|---|---|---|
| statue | 10/10 | stand 10 | - | 0 | −0.002 | 0.000 | fail |
| FINAL3 | 10/10 | walk 10 | 1.0 | 0 | 0.187 | 0.211 | fail |
| f2v0.25 | 10/10 | walk 10 | 1.0 | 0 | 0.280 | 0.308 | fail |
| V01 | 10/10 | walk 10 | 1.0 | 0 | 0.272 | 0.303 | fail |
| V05 | 10/10 | walk 10 | 1.0 | 0 | 0.430 | 0.507 | fail |
| V23 | 10/10 | walk 10 | 1.0 | 0 | 0.401 | 0.473 | fail |
| V24 | 10/10 | walk 10 | 1.0 | 0 | 0.554 | 0.682 | fail |
| V18 | 10/10 | walk 10 | 0.919 | 0 | 0.751 [0.686, 0.771] | 0.979 [0.912, 1.001] | fail (0/10) |
| V35 | 10/10 | walk 10 | 0.973 | 0 | 0.882 [0.788, 0.910] | 1.171 [1.049, 1.212] | fail (0/10) |
| V27 | 10/10 | walk 10 | 0.951 | 0.0022 | 0.992 [0.932, 1.023] | 1.322 [1.267, 1.354] | fail on the mean (5/10 seeds ≥ 1.0) |
| V29 | **8/10** (2 falls at 7.12, 8.55 s) | walk 10 | 0.86 | 0.041 | 0.990 [0.658, 1.140] | 1.283 [0.701, 1.522] | fail (4/10) |

[measured: out/analyze_bm_stdout.txt, out/bm_<tag>_stdout.txt]

What the gate measure loses [measured: out/fv_spread.json]:
- **Start-up.** The episode mean is 0.75-0.77 of the steady speed for V18, V35 and V27, which start with a 0.5 s settle and a 3.5-4 s ramp from 0.2 m/s. It is 0.85-0.91 for the slower walks: FINAL3, f2v0.25 and V01 have no speed ramp, and V05 and V23 have a 2 s ramp.
- **Heading.** `forward_vel` along the reset prey direction is 0.97-0.99 of the probe's displacement speed for the strict walks; the walkers have no yaw control.
- At 1.5 m/s it collapses. On seed 3050, V29 covered ground at 1.51 m/s but only 0.70 m/s along the prey direction.

### Return versus speed

Δ = paired difference from the statue on the same seed, per episode, mean (seeds won of 10). Speeds are episode means. CURRENT = forward_vel_max 2.5 at the ramp start (w 0.2) and ramp end (w 2.0). HYP = hypothetical cap, offline, everything else unchanged. [measured: out/analyze_bm_stdout.txt, out/bm_summary.json]

| walk | v_ep | CURRENT w 0.2 | CURRENT w 2.0 | HYP cap 1.2, w 0.2 | HYP cap 1.2, w 2.0 | HYP cap 1.0, w 0.2 | HYP cap 1.0, w 2.0 |
|---|---|---|---|---|---|---|---|
| statue (return) | −0.002 | 1093.4 | 1091.7 | 1093.2 | 1089.6 | 1093.1 | 1088.8 |
| FINAL3 | 0.187 | +4.9 (9) | +141.3 (10) | +21.3 (10) | +305.5 (10) | +27.7 (10) | +368.6 (10) |
| f2v0.25 | 0.280 | +4.1 (9) | +207.5 (10) | +28.6 (10) | +452.2 (10) | +38.0 (10) | +546.4 (10) |
| V01 | 0.272 | +1.8 (8) | +199.5 (10) | +25.6 (10) | +437.3 (10) | +34.8 (10) | +528.8 (10) |
| V23 (crouched) | 0.401 | **−41.0 (0)** | +249.5 (10) | −6.0 (2) | +599.2 (10) | +7.4 (8) | +733.7 (10) |
| V05 (upright) | 0.430 | +18.2 (10) | +329.4 (10) | +55.7 (10) | +704.0 (10) | +70.1 (10) | +848.1 (10) |
| V24 | 0.554 | **−16.5 (0)** | +384.3 (10) | +31.7 (10) | +866.7 (10) | +50.3 (10) | +1052.2 (10) |
| V18 | 0.751 | +13.4 (10) | +555.7 (10) | +77.5 (10) | +1197.0 (10) | +95.7 (10) | +1378.2 (10) |
| V35 | 0.882 | +21.0 (10) | +658.0 (10) | +89.5 (10) | +1342.0 (10) | +103.7 (10) | +1484.4 (10) |
| V27 | 0.992 | +35.5 (10) | +751.5 (10) | +104.2 (10) | +1438.7 (10) | +114.9 (10) | +1546.2 (10) |
| V29 (2 falls) | 0.990 | −66.5 (7) | +618.7 (10) | −3.0 (7) | +1253.6 (10) | +6.8 (8) | +1351.7 (10) |

**Along the current ramp**, Δ vs the statue at w 0.2 / 0.47 (300k steps) / 1.1 (1M) / 2.0:
- FINAL3: +4.9 / +25.4 / +73.1 / +141.3
- V18: +13.4 / +94.8 / +284.6 / +555.7
- V27: +35.5 / +142.9 / +393.5 / +751.5

**Discounted from reset** (γ 0.995), Δ vs the statue at the current w 0.2:
- FINAL3 +0.66, f2v0.25 +0.69, V05 +2.35
- **V23 −6.2, V24 −4.7, V18 −3.2, V35 −2.6, V27 −1.3** (every crouched walk is below the statue)
- At w 2.0: +19 to +67.
- With either hypothetical cap at w 0.2: +0.6 to +8.2, except V23 (−2.0 / −0.3).

[measured: out/analyze_bm_stdout.txt]

**Where the non-forward deficit comes from**, per episode vs the statue [measured]:
- FINAL3 −10.2: height −4.5, lateral −3.6.
- V05 −16.4: height −7.3.
- V23 −73.3: height −64.2.
- V24 −61.0: height −50.4.
- V18 −46.8: height −33.3, nosedive −5.4, tail −3.3.
- V35 −49.7.
- V27 −44.1.

The height term is one-sided below 0.926 m and costs 1.33 per episode per mm [REWARD stage]. It therefore prices the SPEED controller's crouch (CoM target 0.93 L), which is a controller choice, not a property of walking fast. [inferred]

### Speed at which a walk first beats the statue

- **Ramp end** (w 2.0, any cap): every walk beats it, from FINAL3 at 0.19 m/s on 10 of 10 seeds. For the non-falling walks' measured non-forward deficits, the speed at which the forward pay alone breaks even is 0.013-0.09 m/s (current cap) and 0.005-0.044 m/s (caps 1.0 / 1.2). [measured deficits; inferred break-even]
- **Ramp start, current** (w 0.2):
  - Upright walks win at almost any speed: FINAL3 at 0.19 m/s wins by +4.9 (9 of 10). Their break-even episode-mean speed is 0.13-0.23 m/s.
  - Crouched walks need an episode mean of 0.55-0.92 m/s. V23 (0.40) and V24 (0.55) lose on every seed. V18 (0.75) and faster win by +13 to +35.
- **Ramp start, caps 1.2 / 1.0**: break-even falls to 0.05-0.11 m/s (upright) and 0.22-0.44 m/s (crouched).

[measured: crossing section of out/analyze_bm_stdout.txt]

### How steeply the reward pays speed past a brisk walk

These are steady-window rates (t = 5-10 s, full episodes), relative to the statue's own rate, scaled to 10 s. The slope is between consecutive walks in steady speed, per 10 s per m/s. [measured: out/analyze_bm_stdout.txt, steady section]

| from → to (steady m/s) | CURRENT w 2.0 | CURRENT w 0.2 | cap 1.2, w 2.0 | cap 1.0, w 2.0 | non-forward part |
|---|---|---|---|---|---|
| V24 → V18 (0.68 → 0.98) | +881 | +161 | +1678 | +1615 | +81 |
| V18 → V35 (0.98 → 1.17) | +771 | +51 | +1006 | +648 | −29 |
| V35 → V27 (1.17 → 1.32) | +845 | +125 | +706 | +323 | +45 |
| formula, while unsaturated | 800 | 80 | 1667 | 2000 | - |

- **Current cap.** The pay is linear, at the formula's 0.8 per step per m/s at w 2.0, from a slow walk through 1.32 m/s.
  - At the steady rate, V27 earns +1021 per 10 s over the statue, against the statue's whole 1092. [measured]
  - Nothing in the measured non-forward terms pushes back: the deficit stays −37 to −63 per 10 s from 0.68 to 1.32 m/s.
  - Extrapolating the formula to the cap: steady walking at 2.5 m/s would earn about +950 per 10 s more than at 1.32 m/s, if it did not fall and its other terms held. [inferred]
- **Ramp start** (w 0.2). Speed earns 80 per 10 s per m/s. Between 0.98 and 1.17 m/s the measured slope is only +51, because the non-forward terms move by ±30-80 between controller settings. At the ramp start the reward is close to flat in speed. [measured]
- **Hypothetical caps.** Saturation is gradual because the per-step pelvis speed swings widely within a stride. Steady p5 / median / p95 [measured: out/fv_spread.json]:
  - V18: 0.73 / 0.96 / 1.30
  - V35: 0.87 / 1.14 / 1.57
  - V27: 0.97 / 1.30 / 1.78

  Share of steady steps above the cap [measured]:
  - V18 (steady 0.98): 43% above 1.0.
  - V27 (steady 1.32): 91% above 1.0, 61% above 1.2.

  With a 1.0 cap, V27's steady rate is +1955 per 10 s over the statue. Its forward part is +1992, which is 99.6% of the 2000 maximum. The last +323 per m/s (V35 → V27) comes from lifting the slowest steps of the stride above 1.0. [measured; inferred reading]

### The September hops on the same axis [inferred]

Upper bounds assume the hop never falls and pays statue-like non-forward terms, as in the REWARD stage. The hop speeds are doc claims from physics r8/r9 runs, not re-measured.

| | w 2.0, cap 2.5 | w 2.0, cap 1.2 | w 2.0, cap 1.0 |
|---|---|---|---|
| hop 1.05 m/s: forward pay ≤ | 840 | 1750 | 2000 |
| hop 1.57 m/s: forward pay ≤ | 1256 | 2000 | 2000 |
| fastest strict-ish walk V27 (v_ep 0.99): forward pay | 794 [measured] | 1479 [measured] | 1586 [measured] |

- At the same episode-mean speed, a hop and a walk earn the same forward pay under every setting. The caps remove the pay for speed above the cap. They do not make the reward prefer alternation.
- The difference between hop and walk lies in the gait, which no term and no gate condition reads.

## 3. What this implies [inferred]

1. **Is 1.0 m/s sensible?** Yes, as a target speed for a walk on this body.
   - Fr 0.116 sits inside the walking band. It is about 1.5 times the Froude-converted preferred walking speed recalled for full-size T. rex, and about half the walk-run transition speed.
   - The plant can walk at it: the hand-built V18 holds 0.98 m/s steady on 10 of 10 new seeds, alternating, with no flight.
   - It is a sensible middle between the compsognathus gate (Fr 0.003) and the velociraptor's running gate (Fr 0.81).
   - "Firm walk" (the config comment's phrase) is a fair description.
2. **The gate as implemented does not test that.**
   - (a) The episode mean from a standing start, along the reset prey direction, makes the effective requirement 1.03-1.18 m/s steady with a fast start, and about 1.3 m/s (Fr ≈ 0.20) with the hand-built walks' start-ups. That is at the top of the strict-walk envelope found so far. No strict hand-built walk passes it. The best, V27 at the 1.35 command, misses by 0.008 on the mean.
   - (b) It has no gait condition. The two T. rex runs that passed it hopped at walking Froude numbers (0.13-0.29). A speed threshold, Froude-scaled or not, cannot separate a hop from a walk.
   - (c) The 750-step length condition does not help: hops and walks both survive.
3. **The reward and the gate point in different places.**
   - At the ramp end the reward pays about 0.8 per step per m/s, linearly, up to 2.5 m/s (Fr 0.73). That is past the walk-run transition and near the converted top-speed estimates.
   - The gate then certifies the first speed over 1.0 m/s, whatever the gait.
   - Under these settings the best-paid behaviour is the fastest gait that does not fall. In September that was a hop, not a walk at 1.0 m/s.
   - At the ramp start the reward is nearly flat in speed. There a crouched faster walk is worth less than standing, so speed only starts to matter once the weight has ramped up.
4. **Lowering the cap to 1.0 or 1.2 m/s** (offline only; what training would do under it is not measured):
   - It would make slow and moderate walking clearly better than standing from the ramp start.
   - It would sharply cut, though not end, the pay for speed beyond the cap.
   - A 1.0 cap equal to the episode-mean gate is self-defeating: the gate needs cruising above 1.0, which the 1.0 cap barely pays. A cap of about 1.2-1.5 m/s on a stride-smoothed speed would match a 1.0 m/s gate better.
   - It does nothing about gait.

## Limits

- **Hand-built controllers only.** One controller family per speed, as tuned by the EXISTENCE and SPEED stages; these are not learned policies.
  - The crouch in V23-V29 dominates their non-forward deficit, so it is a property of this controller.
  - A learned walk could be more upright (like V05) or less.
  - The start-up profile, which sets how far the gate measure falls below the steady speed, is also a controller choice.
- **Seeds.** Seeds 3042-3051 only (10 episodes per behaviour), at the locomotion config's reset noise. SPEED tested on seeds 0-14.
  - V29 (1.5 m/s) fell on 2 of 10 of these seeds and drifted badly in heading on a third. Treat it as outside the stable set.
- **The offline recompute is exact for these trajectories,** but it is a recompute of a fixed behaviour. A policy trained under a hypothetical cap would move to different behaviours; that was not measured.
- **Hops not simulated.** The hop numbers are bounds built from doc-claim speeds on earlier physics revisions.
- **Literature figures** are recalled, unverified, and depend on an assumed 3.0-3.5 m adult hip height. Froude similarity ignores strength and mass scaling.
- **The "walk" label** is the frozen probe's per-control-step floor classification. SPEED reports isolated airborne physics substeps even in walks the probe counts flight-free.

## Slips

1. **A write was aimed at the export.** The first run of `scripts/analyze_bm.py` through `scripts/sim.sh` received the export path as its output argument, because sim.sh passes the export as argv[1]. It tried to write its JSON to the export's directory path. `open()` failed with IsADirectoryError, so nothing was written.
   - Afterwards, `find` on the export for anything newer than this stage's first file returned nothing.
   - Both `analyze_bm.py` and `froude.py` now refuse any output path outside `out/`.
   - Recorded in CMDLOG at 02:27Z.
2. **System python3 used twice.** It ran without -I -B and without nice:
   - once in a heredoc to string-replace one label in my own `scripts/froude.py` (dropped "walking gaits only" from the Sellers 2017 entry, which I could not vouch for);
   - once in a pointless `python3 -c 'print(1)'` inside a read-only L.sh line.

   No simulation ran outside `scripts/sim.sh`.
3. **Not logged literally** (each is listed in a CMDLOG NOTE after the fact):
   - the first shell call (date, ls) and the creation of the logger itself;
   - file reads done with the Read tool;
   - two read-only JSON-print python processes inside L.sh, run without nice, OMP_NUM_THREADS or TMPDIR.

   One NOTE line abbreviates a harness path with "..."; the next NOTE line gives it in full.
4. **Five overlapping wait/monitor loops** watched `out/batch1_log.txt`; four were redundant. They were read-only and are noted in CMDLOG.
5. **CMDLOG ordering.** The background batch's START/END lines interleave with foreground NOTE lines; every line carries its own UTC time.
6. **Harness files outside this folder.** The harness wrote its own files there:
   - background-task output files under `/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/tasks/`;
   - a persisted listing at `/root/.claude/projects/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/tool-results/bkrlp51w9.txt`.

## Files

- `CMDLOG.txt`: literal commands with UTC times, plus NOTE lines.
- `scripts/`:
  - `L.sh`: logger for shell commands.
  - `sim.sh`: logger plus nice, OMP and TMPDIR, for every python run.
  - `batch1.sh`: the sequential batch.
  - `measure_L.py`: L per species.
  - `rollout_bm.py`: episodes through the frozen probe, with per-step reward terms.
  - `analyze_bm.py`: offline returns, hypothetical caps, crossings, slopes, the steady window.
  - `fv_spread.py`: within-stride spread, start-up and heading losses.
  - `froude.py`: the Froude table and the literature conversions.
- `out/`:
  - `measure_L.json` and `measure_L_stdout.txt`;
  - `bm_<tag>.json`, `bm_<tag>_steps.npz` and `bm_<tag>_stdout.txt` for statue, final3, f2v025, V01, V05, V23, V24, V18, V35, V27, V29;
  - `chk_V35_s01*` (the bitwise check against SPEED);
  - `analyze_bm_stdout.txt` and `bm_summary.json`;
  - `fv_spread_stdout.txt` and `fv_spread.json`;
  - `froude_stdout.txt` and `froude.json`;
  - `batch1_log.txt`.
