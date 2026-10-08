# Stance-hack audit and the floor-truth stance gate (2026-10-06)

**Status**: dated investigation note, frozen once merged; corrections are appended, never edited in. It records the
2026-10 stance-hack audit's verified findings, the design of the gate kind `stance_quality/v2` that answers them
(decision D-D23, which the maintainer took on 2026-10-06; [../BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md)
§6.2), and the kind's validation on 40-episode panels. The audit's replays ran on 2026-10-05 and 2026-10-06 on CPU
against `main` = `61a3424`; the validation panels in §5 ran on 2026-10-06 on the branch that registers the kind,
whose base is the same commit and which changes no MJCF, keyframe, species env or stage TOML, so the plants and
stages are the ones the audit replayed. It extends the 2026-09-28 audit, [GAIT_AUDIT_2026_09.md](GAIT_AUDIT_2026_09.md);
the plan it amends is [../GAIT_QUALITY_PLAN_2026_09.md](../GAIT_QUALITY_PLAN_2026_09.md) (its §11).

## 1. Question and answer

Do the current stance nodes on Drive stand the way the zero-action statue stands, and can a gate certify a stance
without admitting what they do and without refusing the statue? Twelve checkpoints were replayed, with each
species' statue: the nine certified stance nodes of the five species that have one, the failed trex seed-45 node and
two final checkpoints (§2.1).

- **No stance node stands as its statue does.**
  - **Trex.** All three gate-passing stances (seed 42 `20260914_123816`, seed 44 `20260920_010912` and
    `20260930_024929`) hop at reset, replant their feet 5–7 cm wider inside the unscored 200-step settle window and
    stand with one leg splayed as a prop, its pad rolled 3–4.4° onto the front-inner corner. That includes the
    seed-42 stance the 2026-09-28 audit called the one clean stance: its hop is in the settle window and its rolled
    pad in the foot geometry, and that audit's probe read neither. The failed seed-45 node `20261001_225601`
    micro-hops on both feet at 11.6 Hz.
  - **Velociraptor** `20260922_125248` holds a saturated hip-extension crouch, chatters its feet at about 20 Hz
    (airborne on 15% of steps), keeps its right digit IV off the floor and slides 0.6 m; its final checkpoint parks
    the tail on its command limit.
  - **Compsognathus** `20260921_203149` and `20261001_225856` march in place at about 3 Hz on the tip of one toe,
    the pad off the floor.
  - **Compsognathus_robot** `20260924_031815` stands with its right sole stacked on the left, which carries 81% of
    the floor load on its inner edge.
  - **Dibothrosuchus** `20260923_020654` and `20261004_025237` certified their untrained 50k-step network on the
    statue, three-legged (right forefoot hovering) in 3 and 9 of 40 episodes and rocking onto a diagonal in most of
    the rest; the 6M final checkpoint is a 50 Hz hopping skid.
  - **Brachiosaurus** has no current stance node (the July nodes are refused by the plant contract).
- **Why the gates admit them.** `stance_quality/v1` reads touch sensors, takes the minimum over substeps, averages
  unsupported duty over the panel and does not score the settle window; it counts only steps with neither foot down,
  so a prop, a march, a stack, a crouch and a hop between steps all pass, as the statue does by design. The other
  three stances gate on a reward rail the statue clears.
- **`stance_quality/v2` separates them.** It classifies each panel episode on floor truth (the floor's normal force
  on each leg, per substep) and certifies the lower confidence bound on clean episodes. Under per-species candidate
  bars (§5.1) every statue is clean on 40/40 episodes and every audited checkpoint's panel fails: ten with 0/40
  clean, where each unclean episode fails at least two independent criteria (trex seed 42 and `20260920_010912`),
  three (`20260930_024929`) or more; the two untrained dibothrosuchus checkpoints with 27/40 and 21/40 clean (bounds
  0.534 and 0.385 against the 0.80 bar), where an unclean episode can fail a single criterion, the forefoot it leaves
  unloaded (§5).

## 2. Method

### 2.1 The audit

Three lanes per species, each with its own probe: a behavior lane (what each policy does), a mechanics lane (the plant
and its zero-action statue), and a verification lane that re-measured every claim of the other two with a probe of
its own, written from scratch (the floor's normal force per geom, averaged over every substep through
`_substep_probe_hook`; pad corners from `geom_xmat` and the box half-sizes; the centre of pressure in the sole frame),
and marked each claim confirmed, partly confirmed or refuted, with corrections. **This note uses the verification
lane's versions only.** Panels are 40 deterministic episodes on seeds 3042–3081 (the certification panel, D-B17),
except where a publication-protocol replay is named.

Every replayed checkpoint and VecNormalize sidecar matched the sha256 recorded on Drive (`gate_verdict.json`,
`artifact_manifest.json` or an `ancestors/stance/ancestor.json`). Four sidecars that came back as inline base64
(trex `20260920_010912`, and three dibothrosuchus ones) carried duplicated bytes, were repaired and then matched.
The plant contract accepted every current checkpoint; the July velociraptor and brachiosaurus nodes are refused by it
(policy-interface revision) and were not replayed on the current plant.

| Species | Node | Verdict | Checkpoint / sidecar sha256 |
|---|---|---|---|
| trex | `20260914_123816` stance, seed 42 (reused by the two `20261005_*` runs) | `stance_quality/v1` PASS | `b556bdf5…` / `52f53cd2…` |
| trex | `20260920_010912` stance, seed 44 (the r11 widen) | `stance_quality/v1` PASS | `7f4284ad…` / `963ac712…` |
| trex | `20260930_024929` stance, seed 44 | `stance_quality/v1` PASS | `60156fb1…` / `e4a4a267…` |
| trex | `20261001_225601` stance, seed 45 | `stance_quality/v1` FAIL | `3cc57af7…` / `623844e1…` |
| velociraptor | `20260922_125248` stance (reused by `20260929_112244`) | `reward_and_length/v1` PASS | `732b5a71…` / `c9c6d5d5…` |
| velociraptor | the same stage's `stage1_final` | — | `1bc387f5…` / `36afa6fa…` |
| compsognathus | `20260921_203149` stance | `stance_quality/v1` PASS | `1d46747f…` / `ec278f8f…` |
| compsognathus | `20261001_225856` stance | `stance_quality/v1` PASS | `7b7243ec…` / `cc2c2c2f…` |
| compsognathus_robot | `20260924_031815` stance | `stance_quality/v1` PASS | `b4093e69…` / `cd3071d1…` |
| dibothrosuchus | `20260923_020654` stance (reused by `20260928_012318` and `20261004_025204`) | `reward_and_length/v1` PASS | `1d3527cc…` / `b3078b27…` |
| dibothrosuchus | `20261004_025237` stance | `reward_and_length/v1` PASS | `7460346a…` / `9affbb0c…` |
| dibothrosuchus | the same stage's `stage1_final` | — | `76b4d914…` / `beb7d38c…` |

### 2.2 The validation panels

- **The repository's own path.** Each panel was rolled by `reporting.stance_report.build_stance_gate_report` on the
  species' current stance stage (`load_stage_config(species, 1)`: `stance.toml` for trex, compsognathus and
  compsognathus_robot, `stage1_balance.toml` for the other three), with its `[curriculum]` gate replaced by the
  candidate v2 block of §5.1, which `gate_schema.validate_gate_config` accepts as it would a TOML block. The
  checkpoint pair was loaded with the plant contract enforced (no `allow_legacy_plant`); the floor-truth recorder ran
  on the rollout env; seeds 3042–3081; deterministic actions. Each report was written by `write_stance_gate_report`
  (`stance_gate_report.{json,txt}`, schema `mesozoic.stance-gate-report/v3`, and `stance_panel_selected.csv`), and
  its verdict was re-derived from the CSV alone, through `read_stance_v2_panel` and `evaluate_stance_v2_gate`, as
  publication re-derives it: identical on all 18 panels, and every CSV row on seed 3042 + i.
- **The statue-relative keys** (§5.4) were re-scored on the same rows through the same `evaluate_stance_v2_gate`,
  against the reference reduced from the species' statue panel. That is the reference the report builds when such a
  key is declared, because the statue is deterministic: one panel (dibothrosuchus `20260923_020654`) was also rolled
  with the keys declared, so that the report rolled its own statue panel, and its statue reference and verdict were
  bit-identical to the re-score's.
- **Cost.** 81–154 s per 40-episode statue panel and 101–215 s per checkpoint panel on one CPU core
  (`OMP_NUM_THREADS=1`, three panels in parallel on a four-core machine), the recorder and the reduction included.

### 2.3 Limits

- **One realization, on CPU.** Episodes diverge between the training runtime and a CPU replay (the gait plan's §8);
  a count near a bar can move. The margins in §5.3 say which keys are thin.
- **Candidate bars on today's plants.** The bars of §5.1 were set from the floor-truth prototype's panels and the
  audit; none is committed to a TOML. The T. rex and velociraptor plant revisions the audit motivates will move
  every statue number here, so each adopting TOML re-measures its statue panel (≥ 37/40 clean) on its own plant.
- **The sole-corner metrics assume the plane z = 0**, where every stance stage stands; on a heightfield the corner
  heights would need the ground height subtracted, and box-on-heightfield contact counts differ from box-on-plane.
- **The registry is measurement data** (§4.2). A changed registry entry or constant is a new `MEASUREMENT_VERSION`,
  and a report measured under another is refused, so the panels here cannot certify under a later definition.
- **The judge's refusals** (another checkpoint, another measurement, a tampered statue reference) were exercised by
  the PR's tests, not by these panels; only the publication re-derivation from the CSV was run on them.

## 3. Verified findings per species

Each species: the statue (zero action, which is the home control on every species), then the nodes. BW is body
weight, `body_subtreemass` of the root times g.

### 3.1 Trex

**Statue.** 40/40 full horizon at 3495.2 ± 13.9, 420.4 N on each foot (touch equal to floor, ratio 1.000; pad
315.1 N, digits 105.3 N), no non-foot or foot-on-foot contact, reset peak 1.24 BW (1.18–1.34), width 0.282 → 0.279 m.
The feet are flat on average (median maximum roll 0.55°), but reset noise alone leaves a foot rolled more than 2°,
its outer digit lifted, in 4/40 episodes: edge-standing is a plant property that any hip-roll offset triggers, because
the hip-roll servos (kp 150, forcerange 120, ±25°) are weaker than the gravity sway stiffness and there is no ankle
roll. Passive capacity is about 6% BW in every direction; `tail_1` rests on its ventral stop under every command.

**Nodes.**

- **The prop (all three passing nodes, 39–40/40 episodes).** One hind leg is abducted and its pad tips 3–4.4°
  onto its inner edge and 3.5–4° heel-up: on `20260930_024929` the right pad rolls −4.37° (episodes −6.23 to
  −1.10), its back-outer corner 18.0 mm up in episode 0, the outer toe loaded on 3.8% of steps, the centre of
  pressure at the pad's front edge, while the foot still carries 50.2% BW. `20260914_123816` the same on the right
  (back-outer corner 12.5 mm up); `20260920_010912` on the left in 30/40 episodes and the right in 9/40. The pelvis
  rolls toward the flat-footed leg, not the prop; in the seed-42 and seed-44 widen nodes both hips are abducted.
- **The jump-and-spread at reset (missed by both first lanes).** Right after reset every passing node pushes off at
  2.0–2.4 BW (up to 3.4 BW) at steps 5–9, lifts both feet on at least one step in steps 5–200 in 18/40
  (`20260914`), 32/40 (`20260920`) and 35/40 (`20260930`) episodes, and replants 5–7 cm wider within 40 steps
  (0.283 → 0.329–0.353 m); the statue widens 0 cm. With the feet planted, hip roll cannot make that width or the
  4° splay (full abduction on both hips adds 2 cm; a one-sided command barely moves the joint or topples the animal),
  so the hop inside the unscored settle window is how the prop is set.
- **The rest.** The head is raised 0.43–0.47 m above the statue's (neck about 20° up, beyond the `neck_posture`
  target, so other terms or balance pay for it). Two-footed catch-up hops after the settle window in 6/40
  (`20260920`, drift up to 0.57 m) and 7/40 (`20260930`) episodes, 1/40 on seed 42. Seed 45 (`20261001_225601`,
  FAIL): an actively pumped 11.6 Hz two-footed micro-hop, both feet off the floor on 15.9% of steps by floor truth
  (touch, the minimum over substeps, says 33%). The prop gives `20260930_024929` a one-sided push-recovery envelope
  (8/8 toward the propped side, 3/8 the other way; the statue 0/8).

### 3.2 Velociraptor

**Statue.** 40/40 full horizon at 1745.8 ± 5.0, 50/50 load, per foot digit III 27.6%, metatarsus 13.1% and digit IV
9.2% BW; the single touch sensor reads 0.553 of the floor force (it covers digit III only). The keyframe toes sit
44.6 mm inside the floor, so every reset pops at 1.76 ± 0.20 BW, which at stage noise yaws the statue up to 14.6° and
moves the pelvis 38 mm in the first second. The leg springs are anchored at qpos 0 and hold the pose (0/10 stand
without them).

**Node `20260922_125248`.**

- **A saturated hip-extension crouch.** Hip pitch is commanded at +1 (the extension stop) on about 95% of steps and
  the knee near −1: the thighs sweep back about 36° below horizontal and the knees sit 131–143 mm **behind** the
  hips, a knee-behind-hip pose. Both first lanes described it as a forward-thigh sit; in this model positive hip pitch
  swings the thigh back. The mean action held constant falls (0/10), so the pose needs feedback.
- **Chatter that is not load-bearing.** Feet lift 11.6 and 13.0 times per second with a 20 Hz force peak, airborne
  on 15.2% of steps by floor truth (24.9% by touch). Low-passing the same policy keeps it standing 39/40 with
  0.37 lift-offs per second: the chatter is unpenalised excess control, not balance.
- **Foot use and slide.** The right digit IV carries 0.0% BW (commanded to −30° on 66% of steps), the right foot
  rides its metatarsal head, the left stands 85% on digit III; load 58/42; stance 0.357 m against the statue's
  0.160 m; tangential to normal force 0.478 (statue 0.027); drift 0.6 m and yaw 16° after the first second.
- **Final checkpoint.** `tail_1_pitch` parked at its −15° command limit on 95% of steps as a counterweight, both knees
  saturated, 37% of all action components saturated.

### 3.3 Compsognathus

**Statue.** 2998.7 ± 1.2, 50/50, flat (pad 44%, digits 56% of each foot's load), drift 1.3 mm. It is noise-fragile:
under PPO-scale action noise it falls in a median of 6.5–76 steps, mostly on the 1.07 mm metatarsus clearance, whose
floor contact terminates.

**Nodes `20260921_203149` and `20261001_225856`.** Both march in place on one foot at a time: by floor truth one foot
carries over 90% of the load on 88% of steps, at 3.05–3.12 touchdowns per foot per second, with about one 20 ms step
of double support per handoff and no true flight. Whenever a foot is loaded, 99–99.7% of its force goes through the tip
of digit III, the foot pitched 7.6–10.5° toe-down and the pad 5–7 mm off the floor. The single tip runs on friction
(24–37% tangential force; 9% of loaded substeps at the friction cone on `20260921`) and, with condim 3, resists no yaw,
which fits the 8–25° heading wander. v1's 1.3% "unsupported" steps are those handoffs, not flight. The march scores
0.934 of the statue.

### 3.4 Compsognathus_robot

**Statue.** 2997.8 ± 1.4, 50/50 on four loaded corners per sole, soles flat to 0.013°, 130 mm apart, no leg-leg force.
It too falls under any PPO-scale action noise.

**Node `20260924_031815`.** From control step 3 in 40/40 episodes the right sole's bottom face rests on the left sole's
top inner edge, clamped at 1.51 BW of sole-on-sole force by stalled servos (right knee at its torque limit on 99.7% of
steps); the left foot carries 81% of the floor load, both feet are down on 62% of steps, and the left sole stands on
its inner edge (2 or fewer loaded corners on 77% of substeps). Each touch sensor counts all of the clamp, so touch
totals 3.04 BW against 1.00 on the floor. The stack is a closed-loop state: the policy's mean action held constant is
a balanced two-footed wedge. The right sole sinks up to 6.1 mm into the left one, which rigid printed soles could
not do.

### 3.5 Dibothrosuchus

**Statue.** 2598.3 ± 0.9, four flat feet at 14.2 / 14.2 / 35.8 / 35.8% BW, all four down on every post-settle step.

**Nodes.** The certified checkpoints of both runs are the 50k-step evaluation of an essentially untrained network
(`obs_rms.count` 50004, policy std 1.002), on top of the home-keyframe residual. In `20261004_025237` the right
forefoot is effectively unloaded (under 0.1% BW, hovering 0–1.5 mm) in 9/40 panel episodes and 4/30 publication
episodes; in `20260923_020654`, 3/40. Whether an episode ends up on three legs or four is a closed-loop latch of the
network, not reset noise on a fixed offset (each mode's mean action, held open loop, reproduces that mode on 40/40
seeds). In most "four-footed" episodes a forefoot carries under half its statue share (under 7.1% BW): in 22/40
panel episodes of `20261004_025237` and 26/40 of `20260923_020654`. A three-legged stance costs 0.02–0.13% of return.
The 6M final checkpoint of `20261004_025237` is a chattering, splayed hopping skid: four feet down on 2% of steps, none
on 17%, 36.5% of actuator steps saturated, 2.45 m of travel, pads pitched −10° to −43°.

### 3.6 Brachiosaurus

**Statue.** 1739.1 ± 1.2, loads 18 / 19 / 31 / 32% BW, touch equal to floor, no non-foot load. The fore pads stand
25–27° toe-up on their rear rim, the keyframe's authored tilt plus servo sag; the home pose is not an equilibrium
(hips sag about 5° and the body moves back about 11 cm in the first second), and `neck_1` and `tail_1` end on their
stops. No current stance node exists: the July nodes are refused by the plant contract.

## 4. The design of `stance_quality/v2` (D-D23)

### 4.1 The statistic

An episode is **clean** when it reaches the horizon and every declared criterion holds on a finite metric; a metric
that could not be measured (an empty window, a sole metric on a foot with no sole, saturation with no recorded
action) fails its criterion. The panel **passes** when it holds at least `min_eval_episodes` episodes, all measured
with the declared `settle_steps`, the exact one-sided Clopper-Pearson bound on the clean count
(`recovery_gate.binomial_lcb`) clears `min_clean_stance_lcb`, and the declared panel rails hold. At n = 40 the bound is
0.928 at 40 clean, 0.887 at 39, 0.851 at 38, 0.817 at 37 and 0.786 at 36, so a bar of 0.80 admits 37/40 (the gait
plan's GQ-7 (a), taken for stance).

The settle window is scored, which v1 never did: the `max_settle_*` criteria read `[g, settle_steps)` after a spawn
grace g of 0.1 s (10 steps at dt 0.01, 5 at 0.02), and the other criteria read `[settle_steps, T)`. The grace exists
because the reset pop is a plant property: five statues reach 2.06–3.05 BW at spawn (trex 1.34) and some are briefly
airborne, every pop over by 0.04 s; after the grace every statue is at or below 1.52 BW with no airborne substep
(§5.3). A
`settle_steps` at or inside the grace is refused before anything is rolled.

### 4.2 Floor truth: `environments/shared/gait/`

- **Down.** A leg is down on a control step when the floor's normal force on its geoms exceeds 0.1 N on at least half
  of the step's substeps (the 2026-09-28 audit's floor-truth channel), decoded per substep exactly as
  `mj_contactForce` reports it (elliptic and pyramidal cones), through `_substep_probe_hook`, chaining any hook already
  there. Debounce is 20 ms, defined in seconds. A leg is the limb subtree above a foot's touch-sensor sites, from the
  highest body that holds no other foot's site.
- **Constants** (`gait/constants.py`, `MEASUREMENT_VERSION = "floor-truth/v1"`): `CONTACT_THRESHOLD_N` 0.1,
  `DOWN_SUBSTEP_FRACTION` 0.5, `DEBOUNCE_S` 0.020, `SPAWN_GRACE_S` 0.10, `LOAD_WINDOW_S` 1.0, `SATURATION_ABS` 0.99,
  `SOLE_ROLLED_DEG` 2.0, `SUPPORT_GEOM_DOWN_FRACTION` 0.5. Each report records the measurement manifest (the
  constants, the registry's per-foot geom names, body weight, frame skip and control step) and its sha256.
- **The support-geom registry**, explicit for all six species because the generic rule (the colliding geoms on the
  touch-site bodies) is wrong for four: trex drops `*_metatarsus_geom`, the capsule above the pad, which carries no
  statue load; velociraptor adds digit IV and the metatarsus to digit III and has no sole (its claw never collides
  with the floor), and pins its reference site at `{s}_foot` so its planned sensor change cannot move it; the robot
  drops its roll cheeks, since a cheek on the floor is a rolled foot; brachiosaurus drops the metapodial capsule. On
  every statue the post-settle floor load is entirely on registered support geoms (`nonfoot_load_fraction` 0;
  off-support 0, except brachiosaurus's metapodial graze, at most 3.2e-4).
- **Inert and exact.** With and without the recorder the trajectory is bit-identical; its forces equal a per-contact
  `mj_contactForce` loop exactly; its touch column equals `_aggregated_foot_contact_forces()` bitwise; under SB3's
  auto-resetting `DummyVecEnv` it records the same trace. It runs on evaluation envs only: the contacts are copied per
  substep and reduced once per episode: +0.19 to +0.41 ms per control step on a quiet machine, measured on the
  library's prototype.

### 4.3 Threshold keys

All keys are numeric; `GATE_SCHEMA_VERSION` stays 1, and registering moves no digest (undeclared keys are not
projected into the gate view).

| Key | Kind | Metric (per episode) | Why (finding) |
|---|---|---|---|
| `min_eval_episodes`, `min_clean_stance_lcb`, `settle_steps` | required | panel size, bound, settle window | the statistic (§4.1) |
| `min_all_feet_support` | required | fraction of window steps with every leg down | march, stack, three legs, chatter |
| `max_touchdown_rate` | required | debounced touchdowns per foot per second | march, chatter, hop bouts |
| `max_window_displacement_m` | required | root travel over the window | slide, hop drift |
| `min_foot_load_share` | required | the lowest foot's share of floor load | one-foot standing, stack, three legs |
| `max_actuator_saturation_fraction` | required | the most saturated actuator's share of window steps at \|a\| ≥ 0.99 | velociraptor crouch, seed-45 bounce, skid |
| `max_settle_airborne_substeps` | required | substeps with every leg unloaded, after the grace | the reset hop |
| `max_settle_peak_floor_force_bw` | required | the settle window's peak total floor force, in BW | the reset push-off and the stomp |
| `max_foot_contact_fraction`, `max_phantom_support_fraction` | optional | foot-on-foot substeps; touch above 0.1 N while the floor says up | the robot's stack |
| `max_nonfoot_load_fraction` | optional | floor load outside the legs | kneeling, tail props |
| `min_foot_load_share_windowed` | optional | the lowest share over 1 s blocks | temporary one-leg standing |
| `max_settle_stance_width_change_m` | optional | foot-pair distance change from spawn to the settle window's end | the replant |
| `min_support_geom_duty`, `min_support_geom_coverage` | optional | the least-loaded support geom's duty; the least-covered foot's mean fraction of its support geoms loaded | velociraptor's lifted digit IV, the toe-tip march |
| `max_sole_tilt_deg`, `max_sole_tilt_excess_deg` | optional | the sole's tilt; tilt beyond the keyframe's | the rolled pad; brachiosaurus's authored tilt |
| `max_sole_corner_lift_m`, `min_sole_contacts` | optional | highest minus lowest box corner; loaded sole contact points | the corner prop, edge standing |
| `min_full_horizon_fraction`, `min_avg_reward` | optional rails | panel full-horizon fraction; absolute reward | v1's rails, kept as floors |
| `min_avg_reward_statue_ratio`, `min_foot_load_share_statue_ratio` | optional, statue-relative | panel reward ÷ statue's; per foot, share ÷ statue's share | the near-statue that under-loads a foot (§3.5) |
| `required_consecutive` | allowed | in-training hysteresis | the screen, not built |

The statue-relative ratios divide by a zero-action panel rolled in the same report, with the same env kwargs, seeds
and recorder, reduced to the mean over its full-horizon episodes (so one statue fall cannot make a policy episode
unmeasured); they recalibrate themselves when the reward or plant changes. A declared ratio with no statue reference
fails closed. Reported, never gated: yaw change (the statues reach 12.6° on trex and 15.7° on velociraptor, the
latter at the reset pop), touch-against-floor agreement, the per-foot tuples, the spawn peak and settle touchdowns.
The width and foot-shift bars proposed from eight seeds fail the trex statue at 40 (its maximum width change 0.054 m,
foot shift 0.060 m), so the settle width change is optional and left out of the candidate blocks, and the foot shift
is reported only.

### 4.4 Plumbing

- **The report** (`build_stance_gate_report` on a v2 stage) rolls the panel with the recorder, rolls a statue panel
  when a statue-relative key is declared (the policy's own panel when the policy is the statue), and writes a
  `mesozoic.stance-gate-report/v3` report: `scored_gate_kind`, the thresholds as the gate view projects them
  (`settle_steps` included), the measurement manifest and its sha256, the handoff digests, the `task_sha256` of the
  `[env]` block and plant it rolled, the statue block and every episode's metrics and reasons;
  `stance_panel_selected.csv` carries the same rows, stamped with the handoff, task, measurement (version, manifest
  digest and definition digest) and statue columns. A probe on a v2 stage keeps v1's report, with no recorder and no
  statue panel.
- **The judge** (`evaluate_stage_gate`) refuses a report scored by other code, under other thresholds or another
  measurement, for another checkpoint or sidecar, on another task than the stage recorded (or for a stage that records
  no task or horizon), on other seeds, another horizon or another control step, with the plant contract waived, a probe,
  the statue's own report, a statue block that is not a separately rolled zero-action panel or a statue reference the
  rows contradict, and re-derives the verdict from the rows; a disagreement, or a report it cannot re-derive, is
  refused.
- **Publication** re-derives the verdict from `stance_panel_selected.csv`, bound to the certification-panel seeds, the
  certified checkpoint and sidecar, the stage's recorded task and this checkout's `MEASUREMENT_VERSION` and measurement
  definition (constants and registry entry); without that arm a v2 PASS would have published on the reward rail.
  **Backfill** refuses, never writes a FAIL, for a report the judge does not admit or cannot re-derive. The
  Drive-summary reader (`evaluate_recorded_gate`) returns None unless clean counts are recorded. `gate_verdict.json`
  persists the clean count, panel size, clean fraction, bound and statue reward; the species catalog gains the kind's
  headline.
- **In training** the manager refuses the kind with a log naming where its certificate comes from; the command-line
  curriculum instead judges a v2 node after training, from the stance report on its handoff pair (`judged_by`
  `train_base.train_curriculum/stance_gate_report`). `zero_action_baseline.py` prints the stance gate's verdict on the
  statue for both stance kinds.
- **Not built.** The in-training screen (the recorder on the evaluation env and a screen in `should_advance`); no
  stage TOML declares the kind; the website's gate adapter has no v2 arm, which its two-sided test requires as soon
  as a stage adopts it.

## 5. Validation

### 5.1 The candidate blocks

Every species declares `min_eval_episodes = 40`, `min_clean_stance_lcb = 0.80`, `min_all_feet_support = 0.98`,
`max_touchdown_rate = 0.25`, `max_window_displacement_m = 0.10`, `max_actuator_saturation_fraction = 0.10`,
`max_settle_airborne_substeps = 0`, `max_foot_contact_fraction = 0.02` and `max_phantom_support_fraction = 0.05`, the
floor-truth prototype's candidate criteria, unchanged. Per species:

| Species | `settle_steps` | `min_foot_load_share` | `max_settle_peak_floor_force_bw` | Flatness and coverage |
|---|---|---|---|---|
| trex | 200 | 0.30 | 1.5 | `max_sole_corner_lift_m = 0.006`, `min_sole_contacts = 1.5` |
| velociraptor | 100 | 0.30 | 1.75 | `min_support_geom_duty = 0.90` (no box sole) |
| compsognathus | 200 | 0.30 | 1.5 | `max_sole_corner_lift_m = 0.003`, `min_support_geom_duty = 0.90` |
| compsognathus_robot | 200 | 0.30 | 1.5 | `max_sole_corner_lift_m = 0.003` |
| dibothrosuchus | 100 | 0.05 | 1.5 | `max_sole_corner_lift_m = 0.005` |
| brachiosaurus | 100 | 0.05 | 1.5 | `max_sole_tilt_excess_deg = 10.0` (authored tilt) |

`settle_steps` is the stage's own value where it declares one (trex, compsognathus, compsognathus_robot) and the
audit's 1 s window on the three reward-gated stages. The velociraptor peak bar is 1.75 because its statue still
reaches 1.52 BW after the grace. Trex's support-geom duty and coverage are left out because its statue lifts its outer
toe on some seeds (duty 0 at p0). No bar was changed after the panels ran.

### 5.2 Results

| Panel | Clean | LCB | Verdict | Fewest failed criteria per unclean episode | Mean reward |
|---|---|---|---|---|---|
| trex statue | 40/40 | 0.928 | PASS | — | 3495.2 |
| trex `20260914_123816` (seed 42) | 0/40 | 0.000 | FAIL | 2 | 3459.8 |
| trex `20260920_010912` (seed 44) | 0/40 | 0.000 | FAIL | 2 | 3418.0 |
| trex `20260930_024929` (seed 44) | 0/40 | 0.000 | FAIL | 3 | 3439.5 |
| trex `20261001_225601` (seed 45) | 0/40 | 0.000 | FAIL | 6 | 2781.5 |
| velociraptor statue | 40/40 | 0.928 | PASS | — | 1745.8 |
| velociraptor `20260922_125248` | 0/40 | 0.000 | FAIL | 7 | 1755.6 |
| velociraptor `20260922_125248` final | 0/40 | 0.000 | FAIL | 4 | 1771.4 |
| compsognathus statue | 40/40 | 0.928 | PASS | — | 2998.7 |
| compsognathus `20260921_203149` | 0/40 | 0.000 | FAIL | 5 | 2799.6 |
| compsognathus `20261001_225856` | 0/40 | 0.000 | FAIL | 5 | 2795.9 |
| compsognathus_robot statue | 40/40 | 0.928 | PASS | — | 2997.8 |
| compsognathus_robot `20260924_031815` | 0/40 | 0.000 | FAIL | 8 | 2979.4 |
| dibothrosuchus statue | 40/40 | 0.928 | PASS | — | 2598.3 |
| dibothrosuchus `20260923_020654` | 27/40 | 0.534 | FAIL | 1 | 2597.2 |
| dibothrosuchus `20261004_025237` | 21/40 | 0.385 | FAIL | 1 | 2597.5 |
| dibothrosuchus `20261004_025237` final | 0/40 | 0.000 | FAIL | 7 | 2264.8 |
| brachiosaurus statue | 40/40 | 0.928 | PASS | — | 1739.1 |

The statue panels match the floor-truth prototype's 40/40 on every species, and the trex and velociraptor hack panels
its 0/40. The policies score 0.80–1.01 of their statues' reward (the velociraptor ones above it, the passing trex
stances within 2.2%), so a reward rail at 0.60 of the statue, v1's and the reward-gated stages', refuses none of
them.

### 5.3 Per-criterion failures and margins

Episodes failing each criterion (of 40; an unclean episode usually fails several):

| Panel | Criteria failed (episodes) |
|---|---|
| trex seed 42 | settle peak 40, corner lift 40, sole contacts 39, settle airborne 30 |
| trex `20260920_010912` | settle airborne 37, settle peak 37, corner lift 33, sole contacts 16, saturation 8, support, touchdowns and displacement 5 each |
| trex `20260930_024929` | settle peak 40, corner lift 40, settle airborne 39, sole contacts 39, displacement 3, touchdowns 2, support and saturation 1 each |
| trex seed 45 | support, touchdowns, saturation, settle airborne, settle peak and corner lift 40 each, displacement 37, sole contacts 2 |
| velociraptor | support, touchdowns, displacement, saturation, settle airborne, settle peak and support-geom duty 40 each; horizon, load share, foot-on-foot and phantom 1 each |
| velociraptor final | saturation, settle airborne, settle peak and support-geom duty 40 each, support 29, touchdowns 27, displacement 5, horizon and foot-on-foot 1 each |
| compsognathus `20260921_203149` | support, touchdowns, settle peak, support-geom duty and corner lift 40 each, settle airborne 29 |
| compsognathus `20261001_225856` | support, touchdowns, settle peak, support-geom duty and corner lift 40 each, settle airborne 36, displacement 3 |
| compsognathus_robot | support, touchdowns, load share, settle airborne, settle peak, foot-on-foot, phantom and corner lift 40 each |
| dibothrosuchus `20260923_020654` | load share 13, support 3 |
| dibothrosuchus `20261004_025237` | load share 14, support 11, corner lift 5 |
| dibothrosuchus final | support, touchdowns, displacement, saturation, settle airborne, settle peak and corner lift 40 each |

Statue extremes against the bars, and the nearest hack episodes (min or max over the panel):

| Key (bar) | Statue | Nearest hack episodes |
|---|---|---|
| trex settle peak (≤ 1.5 BW) | ≤ 1.27 | 1.41 (`20260920`), 1.55 (seed 42), 1.84 (`20260930`) |
| trex corner lift (≤ 6 mm) | ≤ 5.5 mm | 5.1 mm (`20260920`), 6.6 mm (`20260930`), 8.4 mm (seed 42) |
| trex sole contacts (≥ 1.5) | ≥ 2.0 | ≤ 2.53 (`20260920`), ≤ 1.99 (`20260930`), ≤ 1.54 (seed 42) |
| trex sole tilt (not declared) | ≤ 3.42° | ≥ 2.11° (`20260920`), ≥ 3.03° (seed 42) |
| settle airborne substeps (0) | 0 on every statue | 0 in some episodes of every trex passing node and both compsognathus nodes |
| velociraptor settle peak (≤ 1.75 BW) | ≤ 1.52 | ≥ 2.78 |
| other statues' settle peak (≤ 1.5 BW) | ≤ 1.16 / 1.10 / 1.09 / 1.22 (compsognathus / robot / dibothrosuchus / brachiosaurus) | ≥ 1.82 (compsognathus), 2.60 (robot) |
| compsognathus corner lift (≤ 3 mm) | ≤ 0.05 mm | ≥ 7.4 mm |
| dibothrosuchus load share (≥ 0.05) | ≥ 0.139 | 0 (three-legged episodes) |
| brachiosaurus tilt excess (≤ 10°) | 6.6–8.4° | — |

No single key separates every trex episode: the seed-44 widen node has episodes that clear each of the settle peak,
corner lift and sole-contact bars, and the seed-42 node clears every GAIT plan §4.3 C criterion (all-feet support
0.996–1, touchdowns ≤ 0.125, displacement ≤ 0.069 m, load share ≥ 0.453) and fails only on the settle window and the
foot geometry. The conjunction separates them: no unclean trex episode fails fewer than two keys. The trex margins
are thin (corner lift 5.5 mm against 6 mm, settle peak 1.27 against 1.5), so a T. rex plant revision should be
accepted on a statue that improves them, and the bars re-measured on that statue's panel.

### 5.4 The statue-relative keys

Re-scored with `min_avg_reward_statue_ratio = 0.60` and `min_foot_load_share_statue_ratio = 0.80` on bipeds, 0.50 on
quadrupeds (the verification lane's proposal for dibothrosuchus, half the statue's share), every statue stays 40/40
and every hack panel still fails; the only clean counts the share ratio changes are the two near-statue
dibothrosuchus panels', from 27/40 and 21/40 to 14/40 each (26/40 episodes under half the statue's forefoot share on
`20260923_020654`). The reward ratio fails no panel: the hack panels score 0.80–1.01 of their statues. The lowest
statue-episode share ratio is 0.92 on trex and 0.795 on brachiosaurus (the right hind foot, 0.247 of the load against
its 0.311 mean on seed 3076), so a quadruped bar must sit below 0.795. The statue reference the report rolled itself
(dibothrosuchus) equalled the re-score's bit for bit, and the extra statue panel cost 81 s.

## 6. Evidence

The audit's lane outputs (per-node JSON, traces, renders) and the validation panels' reports and CSVs are not in the
repository and are not uploaded anywhere. The validation regenerates from the checkpoint pairs on Drive (the sha256s
in §2.1) at a checkout of this note's commit: for each pair, `build_stance_gate_report(species, 1,
stage_config=..., model_path=..., vecnorm_path=...)` with `stage_config = load_stage_config(species, 1)` and its
`curriculum_kwargs` replaced by the stage's non-threshold keys plus the §5.1 block (`zero_action=True` for the
statue), then `write_stance_gate_report`; about 2 minutes per panel on one CPU core. The numbers are deterministic on
one machine; across machines the counts may move by the realization effect of §2.3.

## 7. Addendum 2026-10-06: the T. rex adoption on physics r8 (appended)

*Appended 2026-10-06; §1–§6 above are unchanged.* The maintainer chose on 2026-10-06 to land the T. rex plant
revision this note's §5.3 asks for (physics r7 → r8: the hip-roll servos kp 150 → 600, forcerange ±120 → ±480;
`configs/plant_versions.toml` note 13) with a revised stance task, and to adopt `stance_quality/v2` on the T. rex
stance with it, as decision D-D24 ([../BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2). The bars are
re-measured on the r8 statue, as §2.3 requires, and committed in `configs/trex/stance.toml`, each with its statue and
hack values in its comment.

**Method.** The r8 statue was rolled through `build_stance_gate_report` on the r8 stance stage (its new `[env]`
included), with the block below swapped in for its v1 gate and accepted by `validate_gate_config`; the committed
block's gate view is identical to it. The four audited checkpoints load only on the r7 plant (the plant contract
compares `physics_sha256`), so they and the r7 statue were rolled the same way on a checkout of the commit that
registers the kind, with the same block swapped in, the plant contract enforced and the sha256 of every checkpoint
and sidecar matching §2.1's. Floor truth does not read the reward, so the r7 `[env]` they ran under moves only the
reward rails, which none of these panels is near. Every verdict was re-derived from its CSV and matched. Then the
committed stage was put through the post-stage pipeline itself, with the r8 statue as a scripted checkpoint (the
loader stubbed to return the zero command, the handoff pair two byte files, as the kind's own tests do):
`stage_artifacts._write_stance_gate_report` rolled the panel and, for the statue-relative rail, a separate statue
panel; `_apply_stage_gate` judged it PASS and wrote `gate_verdict.json` (gate `sha256:3571f209…`, the stage's
`task_sha256` `sha256:6da0b7dd…`); publication's re-derivation from `stance_panel_selected.csv` and the backfill tool
both admitted it; and all four declared probes ran on the v2 report (one episode per row, 300-step horizon) and
wrote their files, leaving the panel CSV untouched.

**The block.** `min_eval_episodes = 40`, `min_clean_stance_lcb = 0.80`, `settle_steps = 200`; the required
`min_all_feet_support = 0.98`, `max_touchdown_rate = 0.25`, `max_window_displacement_m = 0.10`,
`min_foot_load_share = 0.40`, `max_actuator_saturation_fraction = 0.10`, `max_settle_airborne_substeps = 0`,
`max_settle_peak_floor_force_bw = 1.5`; the pad bars `max_sole_tilt_deg = 2.0`, `max_sole_corner_lift_m = 0.004`,
`min_sole_contacts = 1.5`; the guards `min_foot_load_share_windowed = 0.35`, `max_foot_contact_fraction = 0.02`,
`max_phantom_support_fraction = 0.05`, `max_nonfoot_load_fraction = 0.02`; and the rails `min_full_horizon_fraction
= 0.95`, `min_avg_reward = 2260.0` (0.60 × the r8 statue's 3766.1) and `min_avg_reward_statue_ratio = 0.60`. Against
§5.1's candidate, the corner-lift bar tightens 6 → 4 mm and the load share rises 0.30 → 0.40 because the r8 statue
allows it; the tilt bar, the windowed load-share guard and the non-foot load guard are added (foot-on-foot and phantom
support were already in §5.1), and so are the three rails.

| Panel (seeds 3042–3081) | Plant | Clean | LCB | Verdict | Fewest bars failed per unclean episode | Mean reward |
|---|---|---|---|---|---|---|
| statue | r8 | 40/40 | 0.928 | PASS | — | 3766.1 |
| statue, seeds 3082–3161 | r8 | 80/80 | 0.963 | PASS | — | 3766.6 |
| statue, seeds 3162–3201 | r8 | 39/40 | 0.887 | PASS | 4, and the horizon | 3694.0 |
| statue | r7 | 36/40 | 0.786 | FAIL | 2 | 3495.2 |
| `20260914_123816` (seed 42) | r7 | 0/40 | 0.000 | FAIL | 3 | 3459.8 |
| `20260920_010912` (seed 44) | r7 | 0/40 | 0.000 | FAIL | 3 | 3418.0 |
| `20260930_024929` (seed 44) | r7 | 0/40 | 0.000 | FAIL | 4 | 3439.5 |
| `20261001_225601` (seed 45) | r7 | 0/40 | 0.000 | FAIL | 7 | 2781.5 |

The r7 statue's four unclean episodes (seeds 3047, 3059, 3074, 3077) each fail tilt (2.47–3.42°) and corner lift
(4.07–5.54 mm). On r7 a looser flatness bar still separates the statue from the props, as §5.2 found at 6 mm (alone,
6 mm leaves the r7 statue clean on 40/40 and `20260920` on 7/40, bound 0.085; 5 mm, 37/40 and 0/40), but only by sub-millimetre and
sub-degree margins (corner lift 5.54 mm against the nearest hack episode's 5.06, tilt 3.42° against 2.11°). The plant
revision is what lets the bars tighten to 4 mm and 2°.

The third r8 statue block (seeds 3162–3201, rolled the same way once the bars were committed) was not part of the
calibration: its one unclean episode is seed 3174, which nosedives at step 265 on reset noise alone (the r7 statue
nosedives on the same seed, at step 228); its worst settle peak is 1.331 BW (seed 3169), tilt 1.353°, corner lift
2.6 mm, sole contacts 2.17 and windowed share 0.469. So the statue figures in the margins table are the extremes of
the panels named, not bounds.

Episodes failing each bar (of 40):

| Panel | Bars failed (episodes) |
|---|---|
| seed 42 | settle peak 40, tilt 40, corner lift 40, sole contacts 39, settle airborne 30 |
| `20260920_010912` | tilt 40, corner lift 40, settle airborne 37, settle peak 37, sole contacts 16, saturation 8, support, touchdowns and displacement 5 each |
| `20260930_024929` | settle peak 40, tilt 40, corner lift 40, settle airborne 39, sole contacts 39, displacement 3, touchdowns 2, support and saturation 1 each |
| seed 45 | support, touchdowns, saturation, settle airborne, settle peak, tilt and corner lift 40 each, displacement 37, sole contacts 2 |

Margins: the r8 statue's least favourable episode in sample (out of sample) against each bar, and the nearest-passing
hack episode:

| Bar | r8 statue | Nearest hack episodes |
|---|---|---|
| tilt ≤ 2° | ≤ 1.154° (1.534°) | ≥ 2.107° (`20260920`), 3.03° (seed 42), 3.98° (`20260930`), 4.28° (seed 45) |
| corner lift ≤ 4 mm | ≤ 2.27 mm (2.80 mm) | ≥ 5.06 mm (`20260920`), 6.65 mm (`20260930`), 8.44 mm (seed 42), 11.1 mm (seed 45) |
| sole contacts ≥ 1.5 | ≥ 2.35 (2.00) | ≤ 2.53 (`20260920`), 1.99 (`20260930`), 1.55 (seed 45), 1.54 (seed 42) |
| settle peak ≤ 1.5 BW | ≤ 1.272 (1.276) | ≥ 1.41 (`20260920`), 1.55 (seed 42), 1.84 (`20260930`), 2.48 (seed 45) |
| settle airborne 0 | 0 (0) | 0 in some episodes of each passing node; ≥ 175 (seed 45) |
| support ≥ 0.98 | 1.000 (1.000) | 1.000 on the passing nodes; ≤ 0.724 (seed 45) |
| touchdowns ≤ 0.25/s | 0 (0) | 0 on the passing nodes; ≥ 9.5 (seed 45) |
| displacement ≤ 0.10 m | ≤ 0.023 m (0.023 m) | ≤ 0.069 m on seed 42 (never fails); hop drift up to 0.49, 0.62 and 0.82 m on `20260920`, `20260930` and seed 45 |
| load share ≥ 0.40 | ≥ 0.498 (0.498) | ≥ 0.436 on every hack episode (a guard, not a separator) |
| saturation ≤ 0.10 | 0 (0) | 0 on the passing nodes; ≥ 0.81 (seed 45) |
| windowed share ≥ 0.35 | ≥ 0.477 (0.474) | ≥ 0.372 on every hack episode (a guard) |
| foot-on-foot, phantom, non-foot | 0 (0) | 0 on every hack episode (guards) |

The two families back each other up. The pad bars alone fail every hack episode (tilt alone does too, its nearest
hack episode 0.11° over the bar); the settle bars alone fail all but one, a `20260920_010912` episode (seed 3045: no
airborne substep and a 1.496 BW peak, 4 mBW under the bar, on a pad tilted 5.2° with its corners 12.2 mm apart);
without both, the window bars pass seed 42 on 40/40 and `20260930_024929` on 37/40 (bound 0.817, a PASS). So the pad
bars are load-bearing on trex and the margins to the nearest hack episodes are thin on both families;
`environments/shared/tests/test_stance_gate_config.py` pins that both stay declared, with the seed-3045 episode as
its case.

**Push capacity.** Note 13 and the CHANGELOG quote the statue's quasi-static capacity as 3.3% of body weight
laterally on r7, 5.1% on r8, and 3.2% fore-aft on both: a horizontal force on the pelvis through `xfrc_applied`
after a 200-step settle, ramped over 0.5 s, held 2.5 s and released for 2 s, reset noise 0, bisected on the largest
force the zero-action statue survives. §3.1's "about 6% BW in every direction" was measured under a different
protocol, so the two sets of figures are not comparable with each other; each compares only within its own protocol.

**Limits.** The hacks are r7 policies on the r7 plant: the r8 policies a retrain produces may find what none of them
found, and a v2 verdict on one is only as good as these bars. The realization effect of §2.3 applies. The statue is
the only r8 policy measured.

## 8. Addendum 2026-10-06: the velociraptor adoption on physics r3 (appended)

*Appended 2026-10-06; §1–§7 above are unchanged.* The maintainer chose on 2026-10-06 to land the velociraptor plant
revision §3.2's findings ask for (physics r2 → r3, policy interface r10 → r11, visual r3 → r4;
`configs/plant_versions.toml` note 14: the leg springs anchored at the standing pose, a flat-footed keyframe whose home
ctrl carries the gravity preload, and the metatarsus and digit-IV touch sensors summed per foot), to declare the
velociraptor SB3-only with it, to revise its stance task, and to adopt `stance_quality/v2` on the velociraptor stance,
as decision D-D25 ([../BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2). The bars are re-measured on the
r3 statue, as §2.3 requires, and committed in `configs/velociraptor/stage1_balance.toml`, each with its statue and
hack values in its comment.

**Method.** As §7's. The r3 statue was rolled through `build_stance_gate_report` on the committed r3 stance stage (its
new `[env]` and the committed `[curriculum]`, whose gate view equals the block below). The audited stance's two
checkpoints load only on the r2 plant, so they and the r2 statue were rolled on a checkout of the commit before the
revision, with the block swapped in for the r2 stage's `reward_and_length/v1` gate, the plant contract enforced and the
sha256 of every checkpoint and sidecar matching §2.1's. Floor truth does not read the reward, so the r2 `[env]` they
ran under moves only the reward rails, which the hacks clear (1.006 and 1.015 of their own statue). Every verdict was
re-derived from its CSV and matched. Then the committed stage was put through the post-stage pipeline with the r3
statue as a scripted checkpoint: `stage_artifacts._write_stance_gate_report` rolled the panel and a separate statue
panel for the statue-relative bars, `_apply_stage_gate` judged it PASS and wrote `gate_verdict.json` (gate
`sha256:f0ef0a6c…`, the stage's `task_sha256` `sha256:d19abe5c…`), and publication's re-derivation from
`stance_panel_selected.csv` and the backfill tool both admitted it. The velociraptor stance declares no probes.

**The block.** `min_eval_episodes = 40`, `min_clean_stance_lcb = 0.80`, `settle_steps = 100` (the r3 statue has both
feet down by step 5 and its floor force within 10% of its weight by step 15, and it classifies identically at any value
from 25 to 200); the required `min_all_feet_support = 0.98`, `max_touchdown_rate = 0.25`, `max_window_displacement_m
= 0.10`, `min_foot_load_share = 0.40`, `max_actuator_saturation_fraction = 0.10`, `max_settle_airborne_substeps = 0`,
`max_settle_peak_floor_force_bw = 2.0`; the foot bars of a foot with no box sole, `min_support_geom_duty = 0.50` and
`min_support_geom_coverage = 0.80` (digit III, digit IV and the metatarsus: the gait library's registry); the settle
splay `max_settle_stance_width_change_m = 0.05`; the statue-relative `min_foot_load_share_statue_ratio = 0.80`; the
guards `min_foot_load_share_windowed = 0.35`, `max_foot_contact_fraction = 0.02`, `max_phantom_support_fraction =
0.05`, `max_nonfoot_load_fraction = 0.01`; and the rails `min_full_horizon_fraction = 0.95`, `min_avg_reward = 1710.0`
(0.60 × the r3 statue's 2842.76, recorded as `collapse_peak_floor_reference = 2842.8` with
`statue_constants_physics_revision = 3`) and `min_avg_reward_statue_ratio = 0.60`. Against §5.1's candidate, the load
share rises 0.30 → 0.40 because the r3 statue allows it (its reset pop is gone: 1.01 BW noise-free against 2.26 on r2),
the coverage, splay, statue-ratio and windowed guards and the rails are added, and the settle peak (1.75 → 2.0 BW) and
the foot duty (0.90 → 0.50) sit in the middle of the statue–hack gap, for the reason below.

**Why the foot bars and the settle peak sit mid-gap.** The bars were first set just under the statue's worst: duty
0.90, coverage 0.95, settle peak 1.5 BW. Those refused a near-statue. The statue with N(0, σ) noise added to its zero
command on every step, rolled through the same report path (seeds 3042–3081), keeps 98.5% of its reward at σ = 0.03,
but its digit IV, keyed 0.17 mm into the floor at 0.087 BW, unloads on part of the window:

| σ | Mean reward | Clean, edge bars (0.90 / 0.95 / 1.5) | Clean, committed bars (0.50 / 0.80 / 2.0) | Worst duty, coverage, settle peak |
|---|---|---|---|---|
| 0.02 | 2831.6 | 40/40 | 40/40 | 0.966, 0.989, 1.36 BW |
| 0.03 | 2798.7 | 13/40 and 16/40 (two noise draws) | 40/40 and 40/40 | 0.842, 0.947, 1.54 BW |
| 0.04 | 2750.4 | 0/40 | 40/40 | 0.680, 0.893, 1.72 BW |
| 0.05 | 2697.1 | 0/40 | 40/40 | 0.553, 0.844, 1.89 BW |
| 0.06 | 2643.5 | — | 31/40 (bound 0.64, FAIL) | 0.448, 0.799, 2.07 BW |

At σ = 0.03 the duty bar alone refused 23 of the 27 unclean episodes. Constant offsets do not do this: digit IV
held ±0.02 or ±0.05 off its home command, or digit III or the ankle +0.02, stay clean on 20/20, so it is jitter, not a
posture, that unloads the digit. The reward cannot see it: the bilateral support term reads each foot's summed touch, which digit III
and the metatarsus keep saturated, the leg-pose tolerance is 0.20 rad, and the jerk term charges about 0.3 per episode
at σ = 0.03. A gate bar that the reward does not shape would have refused a trained near-statue policy for a loss its
training never priced. At the committed bars the jittered statue is clean up to σ = 0.05; coverage 0.80 sits above
2/3, a foot standing on two of its three support geoms throughout, so a digit that never loads still fails it whatever
the duty reads; and every audited hack episode stays far outside (duty ≤ 0.023, coverage ≤ 0.667, settle peak
≥ 2.78 BW). A reward term that sees per-geom support, from the r3 sensors at sensordata 10/27/28 and 11/29/30, would let
the bars move back toward the statue; it is not part of this revision.

| Panel (seeds 3042–3081) | Plant | Clean | LCB | Verdict | Fewest bars failed per unclean episode | Mean reward |
|---|---|---|---|---|---|---|
| statue | r3 | 40/40 | 0.928 | PASS | — | 2842.8 |
| statue, seeds 3082–3161 | r3 | 80/80 | 0.963 | PASS | — | 2842.2 |
| statue, seeds 3162–3201 | r3 | 40/40 | 0.928 | PASS | — | 2843.0 |
| statue, seeds 5042–5241 (fresh) | r3 | 200/200 | — | PASS (5 × 40) | — | 2841.4 |
| statue | r2 | 40/40 | 0.928 | PASS | — | 1745.8 |
| `20260922_125248` robust_best | r2 | 0/40 | 0.000 | FAIL | 9 | 1755.6 |
| `20260922_125248` final | r2 | 0/40 | 0.000 | FAIL | 6 | 1771.4 |

Mean rewards are each plant's own stance `[env]`, so the r2 and r3 rows are not comparable. The third r3 statue block
was rolled after the bars were first set, and the fresh blocks after that; their seed 5202–5241 block was rolled
again through the committed stage and is clean on 40/40. Every row's verdict under the committed block is re-derived
from its recorded per-episode metrics, the CSV where the panel wrote one (floor truth reads no threshold, so a panel
need not be re-rolled when only a bar moves). Under the edge
bars the r2 statue scored 39/40, its seed 3046 failing on a 1.516 BW settle peak from the reset pop the r3 keyframe
removed. Each hack reaches the horizon on 39/40 (robust_best
falls at step 69 on seed 3066, final at step 318 on seed 3077).

Episodes failing each bar (of 40):

| Panel | Bars failed (episodes) |
|---|---|
| robust_best | support, touchdowns, displacement, saturation, settle airborne, settle peak, splay, support-geom duty and coverage 40 each, load share and statue-ratio share 11 each, windowed share 2, foot-on-foot, phantom and non-foot 1 each (the fall, whose window is empty, so unmeasured) |
| final | saturation, settle airborne, settle peak, splay, support-geom duty and coverage 40 each, support 29, touchdowns 27, displacement 5, foot-on-foot 1 (the fall) |

Margins: the r3 statue's least favourable episode in sample (out of sample; third block; the 200 fresh seeds)
against each bar, and the nearest-passing hack episode:

| Bar | r3 statue | Nearest hack episodes |
|---|---|---|
| saturation ≤ 0.10 | 0 (0; 0; 0) | ≥ 0.899 (final), 1.000 (robust_best) |
| settle airborne 0 | 0 (0; 0; 0) | ≥ 35 substeps (final), ≥ 51 (robust_best) |
| settle peak ≤ 2.0 BW | ≤ 1.285 (1.290; 1.166; 1.400) | ≥ 2.78 (final), ≥ 2.98 (robust_best) |
| support-geom duty ≥ 0.50 | 1.000 (0.979; 0.996; 0.956) | ≤ 0.023 (final), ≤ 0.001 (robust_best) |
| support-geom coverage ≥ 0.80 | 1.000 (0.993; 0.999; 0.985) | ≤ 0.667 (final), ≤ 0.342 (robust_best) |
| splay ≤ 0.05 m | ≤ 0.016 m (0.020; 0.021; 0.022) | ≥ 0.116 m (robust_best), ≥ 0.134 m (final) |
| support ≥ 0.98 | 1.000 (1.000; 1.000; 1.000) | ≤ 0.97 (robust_best); ≥ 0.98 on 11 final episodes |
| touchdowns ≤ 0.25/s | 0 (0; 0; 0) | ≥ 0.56 (robust_best); ≤ 0.25 on 13 final episodes, 0 on 11 |
| displacement ≤ 0.10 m | ≤ 0.018 m (0.019; 0.014; 0.031) | ≥ 0.110 m (robust_best); ≥ 0.011 m on final, 5 over |
| load share ≥ 0.40 | ≥ 0.494 (0.494; 0.494; 0.488) | down to 0.368 (robust_best, 11 under); ≥ 0.426 (final) |
| statue-ratio share ≥ 0.80 | ≥ 0.987 (0.988; 0.988; 0.977) | down to 0.735 (robust_best, 11 under); ≥ 0.851 (final) |
| windowed share ≥ 0.35 | ≥ 0.457 (0.460; 0.457; 0.416) | down to 0.344 (robust_best, 2 under); ≥ 0.374 (final) |
| foot-on-foot, phantom, non-foot | 0 (0; 0; 0) | 0 on every full-horizon hack episode; final's fall reaches 0.026 foot-on-foot and 0.023 phantom (guards) |

The fresh seeds' tails are wider than the first 160 seeds' on every measured bar, which is why their column is the
statue's side of each margin. The settle peak's tail was checked further, with the gait library's recorder on the
settle window alone: over 1200 fresh seeds (5042–5241 and 10000–10999) it reaches 1.400 BW at most (seed 5046), p99
1.255, and never 1.5. The bar still reads part of the reset transient: on 36% of those seeds the window's peak is its
first step, step 10, just after the spawn grace (seed 5046: 1.145 BW over steps 0–9, 1.400 at step 10), and with a
15-step grace the worst would be 1.089. Support-geom duty was checked the same way over 600 more seeds and never falls
below 0.95.

Report-only, never gated: the r3 statue yaws up to 10.6° (11.1°; 12.0°; 16.1°) over an episode, and its spawn peak
(the first 0.1 s, before the grace ends) reaches 1.76 BW (1.94; 1.86; 2.12); the summed touch agrees with the floor on
every r3 statue step (1.000), against 0.70–1.00 on the hacks' r2 plant, whose touch saw digit III only.

The families do not back each other up on these two checkpoints the way §7's did on trex: each of them alone refuses
every hack episode. Saturation alone, the two settle bars alone, the two foot bars alone and the splay bar alone each
fail all 80; the window bars alone (support, touchdowns, displacement and the load shares) admit 10 of the final
checkpoint's 40 (bound 0.142, still a FAIL). The quietest hack episode, final seed 3043, clears every window bar (both
feet down throughout, no touchdown, 27 mm of travel, load split 0.547 / 0.453) and fails exactly the four families:
saturation 1.000, 42 settle airborne substeps and a 4.46 BW landing, a 0.20 m splay, and digit IV never loaded (duty 0,
coverage 0.667). `environments/shared/tests/test_raptor_stance_gate_config.py` pins that the four families stay
declared, with that episode as its case, and that the jittered statue's least favourable episodes at σ = 0.05 are clean
under the committed bars and were refused by the edge bars.

**Limits.** The hacks are r2 policies on the r2 plant; the r3 policies a retrain produces may find what neither found,
and a v2 verdict on one is only as good as these bars. The realization effect of §2.3 applies. The statue, and the
statue with command jitter, are the only r3 policies measured; the stance reward revision priced the audited stance from its r2 replays
(`configs/velociraptor/stage1_balance.toml`'s `[env]` comments), not from an r3 policy.

## 9. Addendum 2026-10-07: the T. rex physics-r8 training runs and the stance follow-up (appended)

*Appended 2026-10-07; §1–§8 above are unchanged.* The maintainer trained the first two T. rex stances on the
physics-r8 task of §7 (decision D-D24) on 2026-10-06, seeds 42 and 44, and asked on 2026-10-07, after a review of both
runs, for the follow-up fixes to the T. rex stance: decision D-D27 ([../BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md)
§6.2). This section records the review's verified findings and what the follow-up changes. The plant does not change
(physics r8, policy interface r13, visual r4). Neither checkpoint is certified, and neither can be: both fail
`stance_quality/v2` as recorded and as re-derived. The review ran on 2026-10-07 on CPU against `main` = `46e0b7c`,
whose T. rex digests equal the ones the runs recorded (the velociraptor revision between the two commits moved no
T. rex digest); the follow-up's numbers were measured on the branch that carries it, whose base is the same commit.

**The runs.** Both trained `configs/trex/stance.toml` at commit `7b8b5d1` (clean), fresh (no trunk, no ancestors), on
4 environments on an L4, for 11,001,856 steps. Their recorded `stage_config.json` files differ only in the seed and
the duration: hyperparameters `sha256:8c8b9ab3…`, task `sha256:6da0b7dd…`, gate `sha256:3571f209…`, measurement
floor-truth/v1 `sha256:0a70082d…`, and the plant identity is the r8 entry of `configs/plant_manifest.generated.json`.
Neither sets `log_std_init`, so both started at SB3's std 1.0. The handoff pair of each, selected by evaluation
reward, was downloaded and its sha256 recomputed against `gate_verdict.json`, `provenance.json` and
`artifact_manifest.json`; both load on `main` under the plant contract with no override.

| Run | Seed | Wall clock | Handoff `robust_best_model.zip`, sha256 | `robust_best_model_vecnorm.pkl`, sha256 | Recorded v2 verdict |
|---|---|---|---|---|---|
| `20261006_185343` | 42 | 14h55m | `2053f4875d0ca02c612a4e41a1d8c4198181216a1479d451b1deb93c17c40bd2` (the 10.8M checkpoint) | `b491a28b9ff8d18531cf045a8b2e3cec48ee103a3a06627c656220f65f6b0d00` | FAIL: 0/40 clean, bound 0.000; reward 3012.1 ± 7.2 |
| `20261006_185704` | 44 | 15h46m | `d320d274f9ed298a5b3aa430dbe48f91be7a920c00b0abb2b9dac9c43020e5ab` (the 10.35M best evaluation, 3778.59 ± 9.82) | `17ee31e998a0ed29bd9351498f047258d422d20852fbcb5116593a2bdecb090d` | FAIL: 13/40 clean, bound 0.204; reward 3772.7 ± 17.7 |

**The verdicts, recorded and re-derived.** Each panel was re-rolled on `main` with the committed r8 block, on the
certification seeds and on a fresh block, with the zero-action statue on the same seeds; and, for the follow-up, on
the D-D27 task with the D-D27 block below (no reward term moves a trajectory, so the floor-truth rows are the r8
task's). Cells: clean episodes, bound, mean reward.

| Panel | Seeds | Seed 42 | Seed 44 | Statue |
|---|---|---|---|---|
| recorded, at `7b8b5d1` | 3042–3081 | 0/40, 0.000 | 13/40, 0.204 | 40/40, 0.928 (3766.1) |
| re-derived, r8 block | 3042–3081 | 0/40, 0.000 (3012.1) | 13/40, 0.204, the same clean set (3771.8 ± 22.1) | 40/40, 0.928 (3766.1) |
| re-derived, r8 block | 7042–7081 | 0/40, 0.000 (3011.3) | 8/40, 0.104 (3672.4 ± 469.9) | 40/40, 0.928 (3769.7) |
| D-D27 task and block | 3042–3081 | 0/40, 0.000; 40 hop-or-fall episodes; 1059.9, under both reward rails | 13/40, 0.204; 1 hop-or-fall (3739.2 ± 43.3) | 40/40, 0.928; 0 (3756.8 ± 17.3) |
| D-D27 task and block | 7042–7081 | not rolled | 8/40, 0.104; 4 hop-or-fall, so the rail refuses it too (3607.9 ± 526.2) | 40/40, 0.928; 0 (3761.0 ± 16.5) |

Recorded and re-derived panels are not bit-identical (floating-point divergence across machines: seed 42's episode
rewards differ by at most 3.6, and seed 44's seed 3080 diverges into a window hop on `main`, with 31 of 40 rows within
1e-3), but the verdicts and the clean sets match, and the judge (`reporting.gates`, under the recorded and the current
block) returns `passed = false` for both runs. So v2 behaved as designed: two true negatives and a true positive (the
statue), and no false pass.

Seed 42 fails five bars on every episode: all-feet support 0.625 (bar 0.98), 12.5 touchdowns/s (0.25), raw-command
saturation about 0.94 (0.10), about 250 settle airborne substeps (0) and a settle peak of about 2.8 BW (1.5); window
displacement fails on 19–24 and sole contacts on 20–30 of 40 across the panels. Seed 44 is refused on the
certification seeds by the settle bars alone: settle peaks over 1.5 BW on 27/40 (32/40 on 7042–7081; median 1.586 /
1.622 BW against the statue's 1.235 / 1.237) and settle airborne substeps on 6/40 (4/40). Without the settle criteria
it would be clean on 39/40 there (bound 0.887, a PASS), on 36/40 of the fresh seeds (0.786, a FAIL) and on 75/80
pooled.

**How each stands.** *Seed 42* is a 12.5 Hz synchronous two-foot micro-hop around a near-statue mean pose: both feet
leave the floor together on a rigid 8-step cycle driven by raw ±1 commands (knees at 50 Hz, hip pitch and ankles at
25 Hz, hip rolls and neck at 12.5 Hz). On floor-truth traces both legs carry 0 N for the whole step on 22–25% of
window steps, some substep is airborne on 37.5% of them, each foot touches down 100 times per episode, and the sole
clears the floor by only 5–8 mm. The settle is a 43-touchdown hop sequence that lands at about 2.8 BW. It also steers
to an absolute heading of about −45°: episode yaw change median 45.3–45.9° against the statue's 3.4–3.9°. Its pads
stay flat (tilt 0.96°, corner lift 2.6 mm) and its load splits 0.49. Its 0.80 of the statue's reward is the support
terms paying nothing in flight: a per-term replay against the statue on the same seeds puts 629 of its 767 shortfall
in foot load balance (−224), bilateral support (−221) and the alive bonus (−184).

*Seed 44*, after the settle, stands like the statue on about 90–95% of episodes (all-feet support 1.0, no
touchdown, 4.6 mm of drift, no saturation), and on its window-clean episodes it scores 8–9 per episode above the
statue. Its failures: (1) in the settle it fires raw ±1 leg commands in the first ~60 steps (7–125 actuator-steps at
|a| ≥ 0.95 per episode, the statue none; the 10 Hz filter lets only 0–16 of them reach the bound) and stomps or hops
its feet back to the keyframe width; (2) on 9 of 207 resets (4.3%, 95% CI 2.0–8.1%) that burst does not die out and
the legs lock into a 10–12.5 Hz two-foot hop for the whole episode (on 7042–7081 up to 3.75 touchdowns/s, 1.31 m of
drift and saturation 0.31), and on 1 of 207 (seed 7044) it rolls out slowly onto one leg and falls at step 370 with no
airborne substep; (3) its left foot stands on the front edge of its pad: pitched 0.69–0.75° toe-down against the
statue's 0.14°, on 2.02–2.17 loaded contact points against 3.62–3.68, with the window-mean centre of pressure at
0.85–1.00 of the pad's fore half-length on all 80 panel episodes (the statue at most 0.362 over 120) and the digits
carrying about 1% each (the statue 6–8%); the load splits 0.517 / 0.483, the head sits 6.5 cm above the statue's and
the tail is commanded into its stop. Every item of (3) is inside every v2 bar.

**The r7 hacks are gone** at their r7 size in both seeds (§7's audited stances): no hip-roll splay (asymmetry −0.5° on
seed 42, −0.68° / −0.50° on seed 44; hip-roll std 0.14–0.38 against r7's 0.64–0.79), no rolled-pad prop (worst-pad
tilt median 0.77–0.96° and corner lift 2.1–2.6 mm, against r7's about 4.4° and 12–18 mm), no widened stance (0.279 /
0.283 m against the keyframe's 0.280; r7 replanted at 0.340–0.344 m) and no 0.45 m head lift. What remains of r7's
failure modes is the reset jump (r7's 2–2.4 BW push-off is seed 44's settle stomp, which now replants toward the
keyframe width instead of wider) and the two-foot hop (r7's failed seed-45 node `20261001_225601` hopped at 11.6 Hz
and duty 0.333; seed 42 is the same mode at 12.5 Hz and duty 0.375).

**Heading.** Both policies depend on the world heading the stance stage always spawns at (yaw 0, the prey within about
±11° of +x). The observation carries heading only in world-frame quantities, the pelvis quaternion, the pelvis linear
velocity and the prey vector, and the reset never varies it. With the spawn turned about vertical after `reset()` (the
follow-up's report-only probe below, on the D-D27 task; seeds 3042–3049, and 3042–3045 for seed 42):

| Spawn turned by | Seed 44, the animal alone | Seed 44, the animal and the prey (only the observation differs) | Seed 42, either way | Statue, either way |
|---|---|---|---|---|
| 0° | 8/8 at the horizon | — | 4/4 | 8/8 |
| −45° / +45° | 5/8 / 8/8 | 0/8 / 0/8 | not rolled | 8/8 / 8/8 |
| −90° / +90° | 0/8 / 0/8 | 0/8 / 0/8 | 0/4 / 0/4 | 8/8 / 8/8 |

The review's own probe (the animal alone, six seeds) found seed 44 at the horizon on 4/4, 10/12, 11/12, 6/8, 4/12
and 0/8 episodes at |offset| 0 / 15 / 30 / 45 / 60 / 90°. The statue stands at every offset, so the physics is
heading-neutral; with the prey turned too its reward equals the unturned one (3756.8), and turned alone it loses about
100 at ±90° to the heading term, which still pays alignment with the unturned prey. Seed 44 holds its spawn heading
while it stands; its VecNormalize statistics put the quaternion's w at a z-score of −0.49 at 30°, −1.53 at 45°, −2.96
at 60° and −6.88 at 90°, which is where it starts to fall (the causal link is inferred from that match). Seed 42 falls
at ±90° and, spawned at +45°, turns −84° back toward its preferred heading.

**Why 0.80 against 1.00: basin escape against lock-in.** One config, two basins. Both runs started in a hop regime:
their stochastic rollouts were unsupported on 55% / 57% of steps at 0.1M (std 1.0), and the unsupported duty of
their deterministic evaluations, zero to 200k, rose from 250–400k on, to 0.128 / 0.134 at 1M and 0.418 / 0.321 at 4M
(seed 42 / seed 44). Seed 44 escaped between 5.0M and 6.0M (0.139, 0.038 and 0.0001 at 5M, 5.5M and 6M; evaluation reward
2978 → 3651) while the entropy coefficient was still above zero. Seed 42 stayed (0.365, 0.362, 0.372), locked at
0.375 by 6.5M, just before the entropy anneal reached 0 at 7M, and held exactly 0.375 at every evaluation to 11M.
Policy std fell alike in both (1.00 at 1M, 0.750 / 0.724 at 4M, 0.606 / 0.606 at 6M). Why one seed escaped and the
other did not is not established: n = 2, and no start-std, entropy-schedule or learning-rate ablation was run.

**Root causes in the reward and the recipe** (the review's verified findings; what is inferred says so):

1. *The keyframe-referenced width term paid for re-seating the spawn.* Both seeds change their stance width in the
   settle by 3.6–4.4 times the statue's (median 2.80 / 2.68 cm against 0.64 on 3042–3081, 2.33 / 2.29 against 0.65 on
   7042–7081), and per seed the two independently trained policies' changes correlate at Spearman 0.88 / 0.91, so
   both correct the same spawn error; both end at the keyframe width (0.279 / 0.283 m), narrower than where the
   statue settles (0.288–0.293). The term paid seed 44 +12.5 per episode over the statue for it, and nothing priced
   how it was done. That this term, rather than nosedive or head clearance, drives the re-seat is inferred.
2. *Nothing priced a settle impact or an airborne substep.* The support terms read each step's substep MIN, and v1
   never scored the settle, so seed 44's stomp (settle reward about +2.5 to +5 against the statue, for about 2.9 of
   action penalties) and seed 42's 2.8 BW landings cost nothing as impacts.
3. *The action penalties priced the filtered command.* T. rex alone sets `action_filter_cutoff_hz = 10`, and
   `BaseDinoEnv.step` hands the reward the filtered command, but a first-order 10 Hz pole passes 24–54% of 12.5–50 Hz
   content: seed 42's raw command holds at least four actuators at |a| ≥ 0.999 on 86–94% of steps while the filtered
   one passes 0.9 on 1.8–3.6%, so smoothness charged it 8.8 times and jerk 11.3–11.5 times too little. Actuator
   forces stayed at or below 0.75 of forcerange: this is command saturation, which v2's raw-command bar sees.
4. *Foot flatness and stance width paid on unloaded feet.* On seed 42's airborne steps a lifted foot earned about 80%
   of the flatness reward and 98% of the width reward (flatness 0.109–0.114 per step against 0.139–0.141 grounded).
5. *The recipe's own prediction failed.* `stance.toml`'s `ent_coef_decay_timesteps` comment predicted algo_std "well
   under 0.2" and unsupported duty below 0.28 by about 4M, and named the learning rate as the next suspect if either
   tracked the old curve; both runs did (algo_std 0.750 / 0.724, rollout duty 0.409 / 0.358, evaluation 0.418 /
   0.321 at 4M). Exploration started at std 1.0, where half of all stochastic steps are unsupported (below).

**What the follow-up changes (D-D27).** Seven `TRexEnv` kwargs, each inert at its default (the legacy arithmetic bit
for bit), set in `configs/trex/stance.toml` and inherited by recovery; a base-env record they read; the recipe's start
std; an opt-in trainer guard; four gate keys; a report-only probe; and the statue constants re-derived. Each key's
measured provenance is in its TOML comment.

- *Width against the animal's own settled width:* `stance_width_reference = "settled"`, `stance_width_settle_steps =
  200` (the gate's `settle_steps`; recovery captures at 140, before its first push at 200 ± 50). The statue earns 77.4 ±
  2.8 of the 80 the window can pay (77.1 on 7042–7081, 76.5 on 5042–5081); a quiet constant hip-roll re-seat no longer
  gains width (+6.7 → −2.0 per episode, seeds 3042–3061), though it still nets about +3 in total, mostly through
  flatness, and seed 44's width advantage over the statue falls +12.5 → +2.5 (+16.8 → +3.4 on 5042–5081). A spawn
  reference was measured and rejected: the statue's own settle moves its width by up to 5.2 cm, so it scored 88.7 ± 14.5
  (25.0 at worst). What the settled reference gives up is pricing a stance widened during the settle: a constant ±0.15
  hip-roll splay went from −54.1 to −28.5 per episode against the statue, still below it, and stays priced by flatness,
  the sole bars and the settle width bar below.
- *Load-gated foot terms:* `foot_terms_min_support_force = 168.0` N (0.20 of the 840.9 N body weight): a foot earns
  its flatness share, and the pair the width term, only while its substep-MIN touch carries that load. The statue's
  lightest foot after the spawn grace carries 0.244–0.316 BW over five seed blocks, so the gate costs it 0.2 per
  episode (its spawn steps); seed 42's flatness falls by 45.0.
- *Floor impact and airborne substeps over the whole episode, settle included:* `floor_impact_weight = 2.0` per BW
  of a step's peak summed foot force above `floor_impact_threshold_bw = 1.4`, and `airborne_substep_weight = 1.0` per
  fully airborne step, both read from every physics substep's foot touch, which `BaseDinoEnv.step` now keeps beside
  its MIN (`_substep_foot_force_block`). Touch equals floor truth to 1e-12 N on every recorded step of these panels
  (only the feet touch the floor). The statue's per-step peak after the spawn grace is at most 1.331 BW over 200
  episodes, and it is never airborne after its spawn steps.
- *The raw command priced:* `action_penalty_source = "raw"`: smoothness, jerk and saturation price the policy's own
  clipped command; energy stays on the applied one. Seed 42's raw bang-bang now costs 8.8 / 11.4 / 222 times more
  (49.4 → 434.2, 35.0 → 397.3 and 0.9 → 201.6 per episode); the statue commands zero and pays nothing either way.

Re-scored exactly on the recorded panels (a trajectory does not depend on the reward). The committed implementation,
re-rolling the panels, reproduces the prototype re-scoring to the decimal on every row but 5042–5081, which an
independent re-implementation measured:

| Policy | Seeds | r8 reward | D-D27 reward | Against the statue, r8 → D-D27 | Seeds above the statue |
|---|---|---|---|---|---|
| statue | 3042–3081 | 3766.1 ± 25.6 | 3756.8 ± 17.3 | — | — |
| statue | 7042–7081 | 3769.7 | 3761.0 ± 16.5 | — | — |
| seed 44 | 3042–3081 | 3771.8 | 3739.2 | +5.7 → −17.5 | 23/40 → 11/40 |
| seed 44 | 5042–5081 | — | — | +17.3 → −5.6 | 23/40 → 14/40 |
| seed 44 | 7042–7081 | 3672.4 | 3607.9 | −97.3 → −153.1 | 19/40 → 9/40 |
| seed 42 | 3042–3081 | 3012.1 | 1059.9 | −754.0 → −2696.9 (0.80 → 0.28 of the statue) | 0/40 → 0/40 |

The statue's −9.3 is the settled width paying 800 steps instead of 1000 (−9.1) and the load gate on its spawn steps
(−0.2). The shift on seed 44 against the statue is about −23 per episode on 3042–3081 and 5042–5081 (−21 on its 39
standing episodes of 3042–3081) and −56 on 7042–7081 (−18 on its 36 standing episodes), so whether it ends below the
statue depends on its old margin. Its stomp settles (a settle peak over 1.5 BW or any settle airborne substep) go from
+6.7 to −14.2 per episode against the statue (54 standing episodes on 3042–3081 and 7042–7081; +18.2 → −7.2 on 30 of
5042–5081) and its quiet settles from +13.0 to −1.9 (21 episodes; +14.3 → −0.8), while its hop episodes lose about twice
what they lost before (7065 −574 → −1369, 7071 −495 → −1076, 7074 −173 → −369). The reward now prefers the quiet settle,
but the v2 settle bars, not the reward, are what refuse the stomp.

- *The start std:* `[ppo.policy_kwargs] log_std_init = -1.5` (std 0.22). With the new terms a fresh policy (actions
  `clip(N(0, std²))` about the statue's mean) pays per step 2.88 / 1.00 / 0.41 at std 1.0 / 0.37 / 0.22 (2.83 / 0.92 /
  0.42 on other seeds), against the recipe's rule of about half the 1.0 alive bonus: only −1.5 meets it, and at std
  1.0 the per-step reward is negative. Short PPO runs on the task (this `[ppo]`, 4 environments): the stochastic
  policy's two-foot flight is the exploration noise itself, unsupported duty 0.52–0.54 at std 1.0 under the D-D27
  reward and the r8 one alike (the r8 runs: 0.55–0.57 at 0.1M), 0.21–0.23 at 0.37 and 0.044–0.046 at 0.22 from the
  first rollout; at 0.22 it falls under the D-D27 reward (0.028 at 123k and 0.003 at 451k on the committed task, seed
  42) and does not under the r8 reward (0.038–0.048 to 123k, seed 42), so the start sets the early flight and the
  reward its descent. Single-foot chatter barely moves with the start (6.9–7.6 touchdowns per foot per second at 0.22,
  7.6–10.0 at 1.0, to 123k) and falls with training (4.3 by 369k, seed 44; 3.4 by 451k, seed 42). The deterministic
  policy is weaker evidence: in the first 123k, 2/12 evaluation episodes at the horizon at std 1.0, 9/12 at 0.37 and
  7/12 at 0.22. The r8 runs' deterministic hop began between 0.25M and 0.4M; two runs at 0.22 span that window (seed
  44, 369k on the prototype env: 2/2 standing at 246k and 328k; seed 42, 451k on the committed env: 12/12 at the
  horizon from 8k to 410k with no flight). Their final checkpoints on seeds 3042–3061: seed 44 stood 20/20 and was
  clean on 18/20 under the r8 block (pad near misses on 3047 and 3059); seed 42 was clean on 17/20 under the D-D27
  block (a slow tipping fall at step 515 on 3047; pad tilt near misses of 2.95° and 2.73° on 3045 and 3059); neither
  stomps (settle peaks at most 1.238 / 1.296 BW, no settle flight; seed 42 at most one settle touchdown). That is
  early-training evidence, not a certification: 20 episodes and under 0.5M of 11M steps.
- *A trainer guard:* the opt-in hop watch (`curriculum/hop_watch.py`; `[curriculum] hop_watch_max_unsupported_duty =
  0.05`, `hop_watch_after_timesteps = 7000000`, `hop_watch_stop = true`) reads the evaluation unsupported duty already
  recorded in `gate_progress.npz` and, at the first evaluation from 7M on above the bar, warns, writes `hop_watch.json`
  and ends training; the post-stage v2 report then judges the handoff pair as usual. On the r8 series it would have
  stopped seed 42 at 7M with 4M of its 11M steps (about 5 h on the L4) unspent and left seed 44 alone (under the bar
  from 5.35M; 0.013 at 7M and at most 0.018 after, 2.8 times under it). Its timing was read off the two std-1.0 runs and
  is untested at −1.5. Its keys configure early stopping and enter no digest.
- *The gate (the trex v2 block):*

| Key | Statue (largest, per seed block) | Seed 44 | Seed 42 | Kind |
|---|---|---|---|---|
| `max_settle_stance_width_change_m = 0.08` | 0.0523 m (3042–3081); 0.0201, 0.0311, 0.0508, 0.0396 on four other blocks | fails 1/40 on each panel (up to 0.0875) | fails 1/40 (0.0938) | an existing optional criterion |
| `max_settle_touchdowns = 2` | at most 1 on every episode of 3042–3081 and 7042–7081 (a foot that spawns in the air lands once) | fails 4/40 on each panel (3–13) | fails 40/40 (median 43) | a new optional criterion on an existing metric |
| `max_episode_yaw_change_deg = 25` | 12.4° (3042–3081); 8.0°, 10.4°, 12.0°, 11.5° on four other blocks | at most 21.4°, passes | fails 40/40 (median 45.3°) | a new optional criterion, report-only before |
| `max_hop_or_fall_episodes = 1` | 0 / 0 (1 on 3162–3201: seed 3174's nosedive) | 1 on 3042–3081 (passes), 4 on 7042–7081 (refused) | 40 | a new optional panel rail |

  The rail counts the episodes that end early or fail all-feet support, touchdown rate, window displacement or command
  saturation, the signature of a hop or a fall; pad, load-share, yaw and settle misses stay with the bound, as D-D23
  sized it. At n = 40 a panel with the statue's measured window-failure rate (1 in 120) passes it with probability 0.956
  (0.939 at 2 in 200); a policy with seed 44's hop-or-fall rate (9 whole-episode hops and a roll-over in 207 resets,
  4.8%) passes it about 0.42 of the time, against 0.87 under the bound alone. So it halves the admission of a
  few-percent hop or fall mode, it does not remove it, and it would have passed seed 44 on the certification seeds; a
  bar of 0 would refuse the statue on 28% of panels. The settle touchdown bar closes the one settle re-seat the reward
  leaves unpriced, a quiet single-foot re-plant (single support, landing under 1.4 BW). The yaw bar measures turning,
  not heading dependence. The new keys are declared by the T. rex stance only, so no other species' `gate_sha256` or
  report moves; `min_sole_contacts` stays 1.5 (the statue reaches 2.086 out of sample, 2 of 40 episodes below 2.25), and
  `min_avg_reward` moves 2260 → 2250 with the statue.
- *Heading, report only:* `[curriculum] stance_probe_spawn_yaw_deg = [-90.0, -45.0, 45.0, 90.0]` re-rolls the policy
  and the statue after the gate report with the spawn turned by each offset, the animal alone and with the prey and
  the heading term's reference direction, beside an unturned control, and writes `stance_heading_probe.{txt,json}`
  (standalone: `stance_gate_report.py --spawn-yaw-offsets`). It is not a gate: spawn-yaw randomisation and a
  heading-invariant observation would each be a policy-interface revision, and the maintainer has not decided whether
  stance certification should require heading robustness.
- *Statue constants:* the stance statue scores 3756.8 ± 17.3 under the D-D27 task (`zero_action_baseline.py`, seed
  3042, 40 episodes), so `min_avg_reward` is 2250 and `collapse_peak_floor_reference` 3756.8; the pushed recovery
  statue, with the capture at 140, 1329.1 ± 507.3 (the same harness reproduces 1337.5 ± 506.8 on the r8 task), so
  recovery's reference is 1329.1. Locomotion (1091.5) and behavior (602.0) set none of the new kwargs and keep theirs.
  `statue_constants_physics_revision` stays 8.
- *Digests:* 36 lines of the digest golden move, all T. rex: the four stages' `task_sha256` and each stage's two
  `stage_config_view_sha256` lines (PPO, SAC), the stance `gate_sha256`, the stance and recovery
  `hyperparameters_sha256.PPO`, the 11 behavior recipes' `task_sha256`, and the reward `summary` of stance and recovery
  and `shape_sha256` and `rounded_values_sha256` of all four T. rex reward stages. No plant, interface or other species'
  line moves.

**What stays open.**

1. *Heading robustness* (the maintainer's decision): whether stance certification should require it, through spawn-yaw
   randomisation with the prey placed relative to the spawn or a heading-invariant observation (a yaw-free quaternion, a
   yaw-frame linear velocity, a body-frame prey vector), both policy-interface revisions; or whether it belongs to
   recovery and locomotion. The probe now measures it on every T. rex stance report.
2. *A few-percent hop or fall mode against a 40-episode bound:* the rail halves its admission; what caught seed 44's
   mode was the re-judgement on 7042–7081, which `docs/NEXT_STEPS.md` now asks for before a T. rex stance is handed to
   recovery or locomotion.
3. *The front-edge foot:* a per-foot centre-of-pressure fore-aft bar separates it cleanly (the statue at most 0.362
   of the half-length over 120 episodes against seed 44's left foot at 0.85–1.00 on 80/80; a bar near 0.7), but it
   needs a new `StanceEpisodeMetrics` field, and `from_row` is strict, so every recorded v2 CSV would stop re-deriving
   until migrated. Raising `min_sole_contacts` is not the fix.
4. *Basin selection:* whether the −1.5 start changes the 5–7M escape the r8 seeds split on, and the learning rate and
   entropy anneal the falsified prediction points at, are untested; train at least two seeds (plan three) and let the
   hop watch stop a seed still hopping at 7M.
5. *The D-D27 reward has trained no policy past 0.45M steps.* The keyframe-referenced `leg_home_pose` term is now the
   largest spawn-correction payoff (seed 44 earns +7.9 per episode over the statue from it, the 369k D-D27 policy
   +14.8, with nosedive +7.0 and head clearance +5.1): a quiet correction is legitimate and the new terms price a
   violent one, but if the next runs re-seat through the legs, a settled reference for that term is the analogous
   fix. That run's and the 451k run's pad near misses (2.2–2.95° against the 2.0° bar) are the first thing to check
   on the next T. rex panel.
6. *Recovery* inherits the impact and airborne terms; what a stepping recovery policy's landings cost is unmeasured
   (the pushed statue never steps, never passes 1.35 BW and is never airborne).
7. *Unrolled checkpoints:* only `robust_best` was judged in each run; `best_model`, `stage1_final` and the periodic
   10.6–11.0M checkpoints were not rolled.
8. *The collapse backstop's warm-up* (1.0M) was derived from a std-1.0 series; at −1.5 the dip after the first
   evaluation was shallower in the short runs (7/12 deterministic episodes at the horizon in the first 123k against 2/12
   at std 1.0; the first evaluation itself reads the policy's initial mean, the statue, at every start), so its premise
   is unmeasured on a full run.
9. *Head, neck and tail exploration* stays unconverged on seed 44 (std 0.70–0.99), where the reward is nearly flat;
   narrowing `neck_posture_tolerance` was proposed by the review and not measured.

**Limits.** Every D-D27 reward figure above is an exact re-scoring of r8 checkpoints trained under the r8 reward,
plus short PPO runs (at most 0.45M steps, one or two seeds per setting); the reward has trained no full policy, and
the next T. rex stance it trains may find what neither r8 seed found. The review's inferred causes (the basin
mechanism, the width term as the driver of the re-seat, the link between the out-of-distribution quaternion and the
heading falls) were not ablated. The hop and fall rates rest on about 207 seed-44 episodes. As §6 says of the audit,
the review's and the follow-up's scripts, traces and panel outputs are not in the repository; the panels regenerate
from the checkpoint pairs on Drive (the sha256s above) with `stance_gate_report.py` at this note's commit.

## 10. Addendum 2026-10-07: the compsognathus adoption on physics r2 (appended)

*Appended 2026-10-07; §1–§9 above are unchanged.* §3.3 found both certified compsognathus stances, `20260921_203149`
and `20261001_225856`, marching in place on the tip of digit III. A standing review of the species followed on
2026-10-07: four lanes (anatomy, feet, mechanics, training), each re-measured by an independent verifier, then a
design and a critique that re-measured the design on disjoint seeds and noise streams. The maintainer took its answers
the same day as decision D-D26 ([../BEHAVIOR_RECIPES_PLAN.md](../BEHAVIOR_RECIPES_PLAN.md) §6.2): the anatomical plant
revision (physics r1 → r2, policy interface r2 → r3, visual r1 → r2; `configs/plant_versions.toml` note 15), a
soft-cubic leg interface, a revised stance task and `stance_quality/v2` on the compsognathus stance, in one pull
request, and the same day the toe armature the implementation needed. The bars are re-measured on the landed r2
statue, as §2.3 requires, and committed in `configs/compsognathus/stance.toml`, each with its statue and hack values
in its comment; two of them are new optional criteria of the kind, the window hop pair. The robot is untouched.

**Root causes (verified).** The march is the product of four things, and the review found no one of them sufficient:

- *Action authority.* Every leg servo saturates at forcerange / kp = 0.05 rad of position error, while the linear
  home-keyframe residual spreads ±1 over the 0.35–1.0 rad half-ranges, so 0.007–0.015 of action is already the whole
  knee or ankle holding torque and 86–93% of the action range is saturated torque. The zero-action r1 statue survived
  white action noise only to σ 0.01 (σ 0.02: 4–5 of 20 full episodes; σ 0.03: 0/20); at the recipe's initial σ 0.135
  (`log_std_init` −2.0) it fell in a median of 7–12 steps, so PPO never sampled the quiet stance, and a fresh,
  untrained PPO policy sampled at `log_std_init` −4.6 held it on 20/20. The ankle, not the knee, is the most
  noise-sensitive group, and 100 Hz control does not help open-loop noise (at equal per-step σ the statue falls as
  often).
- *The plant's mass and clearance.* The tail's 178 g put a 6.2 g/cm³ tip on the end of a 0.58 g/cm³ base, which held
  the centre of mass at 0.25 of the foot from the heel: a 19.6 mm backward margin, and every real noise fall went
  backward onto the tail. The metatarsus capsule sat 1.07 mm above the floor on the settled statue; every metatarsus
  touch carried 0 N (48 of 48 first touches), yet the contact terminates, so upright, flat-footed episodes ended as
  falls; that trigger accounts for about 60% of the time-to-termination under high noise. The right foot was built as
  a left foot (digit II lateral), mechanically neutral.
- *An all-or-nothing foot.* The foot is a rigid plate: 0.05° of toe-down pitch moves all of its load onto the
  digit-III tip. A servo-held flat foot stays flat over a toe-action band of −0.20 to +0.04, so the tiptoe is not a
  passive default: the marches hold it actively, the MTP servo at its 0.6 N·m force cap on 64–65% of loaded steps
  (83–91% in single support), and the full-load moment on that tip, 0.594 N·m, equals the cap. The digit capsules
  start at the MTP joint, under the pad, so the statue's "digit" load (56% in §3.3) is mostly MTP-point load; the
  distal digits carry about 14%.
- *What the reward and the gate paid.* The legacy reward paid the deterministic marches 0.928–0.936 of the statue, and
  v1 admitted them (§3.3).

**The maintainer's decisions** (D-D26), each as the review recommended: the MTP servo keeps its 0.6 N·m cap, the cut
deferred until training runs (the maintainer first chose 0.4 N·m and revised it to the recommendation the same day);
the soft-cubic leg mapping, b = 0.1a + 0.9a³, as a new plant-contract mode `home-keyframe-residual-softcubic/v1`; a
new `CompsognathusBiologicalEnv` subclass, so `CompsognathusEnv` and the robot keep their code and digests; recovery
inherits the new stance terms through `extends` and is recalibrated; tail mass only, with the proportions fidelity
pass deferred (an anatomically long tail at the same mass gives back about 43% of the centre-of-mass gain); land, then
train, at least two seeds from `main`; one pull request. The same day the implementation found the MTP servo ringing
(below), and the maintainer took its fix into the same revision: the toe joint's armature at 2e-4 kg·m², a
physics-only generator parameter. With the cap kept, the single-support tiptoe stays physically possible on r2: the
`20260921_203149` march, replayed on the landed r2 plant through its own linear interface (out of distribution),
still survives 20 of 20 episodes on its digit-III tip (seeds 3042–3061; `20261001_225856` survives 9 of 20). The
reward (the marches re-score at 0.61 of the statue) and the gate (support-geom duty and coverage, sole tilt, corner
lift, touchdowns) are what refuse it. A walker's measured toe-torque budget is the evidence to collect before the cap
is revisited.

**The toe servo's heel chatter (found in the implementation).** On the review's other r2 changes alone, the
zero-action statue chattered after the settle on 15 of 40 seeds (3042–3081): 9 with one foot at 20.0 Hz, 6 with both
at 20.8 Hz, up to 3.6% of body weight rms and 7.9% at peak (r1 had it on 6 of 40, at 18.5 Hz and 0.8% rms). The pad
unloads on about 4% of substeps while the rest of the foot stays loaded, so no touchdown, support or support-geom bar
sees it, but `validate_models` failed 2 of its 10 seeded holds on it (`supports_weight`, a 5.4% ground-force error
against the 3% bar) and with them `test_balance.py`'s seeds 49 and 51. It is numerical: the 20 g foot plate gives the
toe DOF 1.6e-5 kg·m², so with the model's default joint armature (5e-5, the generator's default class) its kp-12 servo
rang at 68 Hz, ω·dt 0.85 at the 2 ms step, the stiffest servo DOF in the model. Halving the timestep removes it on 40
of 40 seeds, and so does toe armature 2e-4 (37 Hz, ω·dt 0.47); ankle damping × 3, softer foot contacts, 1000 solver
iterations and the implicit integrator do not, and of the r2 changes the uniform tail is what spreads it (the tail
alone fails 3 of 10 holds, the inset or the mirror alone none). With the armature (note 15 e) no statue of 40
chatters, every hold passes (the worst settled ground-force error of the ten is 0.31%), the regenerated
`preflight_training_v1.json` records `passed: true`, and no spawn of seeds 3042–3081 leaves the floor (one substep of
80 spawns out of sample); statics, keyframe and preload are unchanged, and the armature moves the physics digest
alone. Every number below is the landed plant's, armature included.

**The landed package, measured.** Zero-action statue on the stance task (`frame_skip` 10), r1 → r2: settled metatarsus
clearance 1.07 → 4.81 mm; centre of mass 19.6 → 29.3 mm ahead of the pad's rear edge (0.25 → 0.38 of the 78 mm foot);
pad / distal-digit share of each foot's load 0.44 / 0.14 → 0.41 / 0.25 (seeds 5100–5109, at step 200); the largest
0.1 s pelvis push held on 5 of 5 seeds forward 0.225 → 0.275, backward 0.125 → 0.20 and lateral 0.45 → 0.45 body
weights; the largest initial tilt held on 3 of 3 seeds nose-down 2 → 5°, nose-up 4 → 7°, roll 15 → 15° (every
fore-aft push or pitch failure ends on tail_4). Zero-mean white action noise, each revision through its own interface
(paired streams, seeds 3042 + i, noise stream `default_rng(123456 + 17 i)` unless noted):

| σ | r1, linear residual | r2, soft-cubic legs |
|---|---|---|
| 0.02 | 5/20 full episodes (8 on the metatarsus, 7 on tail_4) | 20/20 |
| 0.10 | — | 40/40 |
| 0.135 (`log_std_init` −2.0) | 0/20 (median 12 steps; 16 on the metatarsus) | 39/40; 37/40 on seeds 3082–3121 (the stream continued) and on 7042–7081; 38/40 on 9042–9081 with an independent stream (`default_rng(700001 + 31 i)`); 151/160 pooled, all nine falls on tail_4, at steps 174–938 |
| 0.20 | — | 0/40 (median 180 steps; 34 on the tail, 6 on tilt); 0/40 on seeds 11042–11081 with an independent stream (`default_rng(880001 + 7919 i)`, median 129 steps) |
| 1.0 | — | 0/20 (median 16 steps) |

On the same streams the plant without the armature kept 30/40 at σ 0.135, and the review's design 36–37/40 on a plant
that also capped the toe at 0.4 N·m. The cliff between the recipe's σ and 0.20 is a constraint on `log_std_init` and
on any entropy schedule; `environments/compsognathus/tests/test_noise_tolerance.py` pins σ 0.10 on the first 5
episodes and σ 0.135 on at least 10 of the first 12 (the review's 0.8; the measured run keeps 11).

**The reward, re-scored.** The stance `[env]`'s ten terms were weighted by re-scoring recorded rollouts (a trajectory
does not depend on the reward; the re-score equals the env's own total to 5e-13 per episode), each figure in the
`[env]` comments. The statue stays the optimum: 4570.4 ± 7.2 per episode on seeds 3042–3081 (4565.8 on 7042–7081),
1571.3 of the 1580 the terms can pay, short mainly on coverage, because the settle seats the digit tips last (the last
by step 308; coverage is 8.1 of the 8.7 per episode it leaves on average, 25.9 at most in one episode), and no
constant toe offset wins it back. The certified marches, on the r1 plant through their own interface, re-score at
0.612 and 0.610 of that plant's statue (0.936 and 0.934 under the legacy reward). On a 20-episode re-score (seeds
3042–3061) rolled feet keep 0.852–0.892, the swept-back crouch 0.887 (0.804 on seeds 11042–11061), the post-settle hop 0.649–0.911 and the settle
stomps 0.957–0.989; the noisy statue keeps 0.824 at σ 0.10 and 0.710 at σ 0.135 (1.9% and 4.2% below what the legacy
reward alone keeps), and the same noise around every non-statue posture measured falls on 19–20 of 20 episodes (median
23–440 steps over two noise streams) and keeps at most about 0.25. At σ 0.135 the penalties cost 0.169 per step.

**Method.** Each panel was rolled through `build_stance_gate_report` on the committed stance stage as training builds
it (`load_stage_config` into the registered `CompsognathusBiologicalEnv`, `frame_skip` 10, the real reset), the policy
standing in for the report's zero-action predictor (zero action, N(0, σ) command jitter, scripted offsets, stomps,
hops and one-leg shifts, the two PPO runs and the audited checkpoints), and every verdict below is re-derived from the
recorded per-episode metrics under the committed block (floor truth reads no threshold, so a panel need not be
re-rolled when a bar moves); the statue-relative keys read the same-seed statue panel. The certified marches load only
on the r1 plant and its linear interface, so they and the r1 statue were rolled through the same report path with the
r1 XML swapped in and `CompsognathusEnv`'s linear `_scale_action`, the plant check bypassed for that scratch plant and
recorded, the checkpoints the audit's (§2.1, sha256 verified). The two PPO runs trained 200k steps each on the
committed `[env]` and recipe (seeds 11 and 12, `log_std_init` −2.0) and were rolled with their deterministic means and
stochastically. Then, as in §7 and §8, the committed stage was put through the post-stage pipeline with the zero
command as a scripted checkpoint: `stage_artifacts._write_stance_gate_report` rolled the panel and a separate statue
panel for the statue-relative bars, `_apply_stage_gate` judged it PASS on 40/40 (bound 0.928) and wrote
`gate_verdict.json` (gate `sha256:f38e3b26…`, the stage's `task_sha256` `sha256:6c667a99…`), and publication's
re-derivation from `stance_panel_selected.csv` and the backfill tool both admitted it. The compsognathus stance
declares no probes.

**The block.** `min_eval_episodes = 40`, `min_clean_stance_lcb = 0.80`, `settle_steps = 200` (the r2 statue's last
one-foot unload, a foot under 5% of body weight on a substep, is at step 4 at most, inside the 0.1 s spawn grace, and
its total floor force stays within 0.01% of its weight after step 200; 200 equals v1's prefix and the `[env]`'s
`stance_width_settle_steps`); the required `min_all_feet_support = 0.98`, `max_touchdown_rate = 0.25`,
`max_window_displacement_m = 0.05`, `min_foot_load_share = 0.40`, `max_actuator_saturation_fraction = 0.10`,
`max_settle_airborne_substeps = 15`, `max_settle_peak_floor_force_bw = 3.0`; the pad bars `max_sole_tilt_deg = 2.0`,
`max_sole_corner_lift_m = 0.003`, `min_sole_contacts = 1.5`; the foot bars `min_support_geom_duty = 0.50` and
`min_support_geom_coverage = 0.80` (the plantar pad and digits II–IV: the gait library's registry); the settle width
`max_settle_stance_width_change_m = 0.011`; the window hop pair `max_window_airborne_substeps = 40` and
`max_window_peak_floor_force_bw = 2.0` (below); the statue-relative `min_foot_load_share_statue_ratio = 0.80`; the
guards `min_foot_load_share_windowed = 0.35`, `max_foot_contact_fraction = 0.02`, `max_phantom_support_fraction =
0.05`, `max_nonfoot_load_fraction = 0.01`; and the rails `min_full_horizon_fraction = 0.95`, `min_avg_reward = 2740.0`
(0.60 × the r2 statue's 4570.4 under the new `[env]` is 2742.2, rounded to the nearest 10; recorded as
`collapse_peak_floor_reference = 4570.4` with `statue_constants_physics_revision = 2`) and
`min_avg_reward_statue_ratio = 0.60`. The certified marches re-score at 0.610–0.612 of their own statue, about at the
0.60 rail, which does not refuse them by itself; the bars do. The collapse detector stays unarmed, as before.

**Why the bars sit where they do.** Four choices differ from the trex's and the velociraptor's:

- *The settle bars are looser* (15 airborne substeps and 3.0 BW, against 0 and 1.5–2.0 on the trex and the
  velociraptor): noise bounces this light plant. The statue with σ 0.03 command jitter reaches 5 airborne substeps and
  1.77 BW in the settle, and with σ 0.05 10 and 2.12, so the trex's bars would refuse a flat, jittery stance; the
  settle stomps reach at least 13 substeps and 3.96 BW. The review's first candidate was looser still (20 and 4.0),
  and the trex r8 runs showed why that is not enough on its own: their seed-44 stance stomped its feet back to the
  keyframe width in the unscored settle, and 0 of its 80 panel episodes exceed 4 BW. So the reward now prices impact
  above 2.5 BW and airborne substeps from the first step (`floor_impact_*`, `airborne_substep_weight` in `[env]`), and
  the settle width bar is tight.
- *The settle width bar* is 0.011 m, about three times the statue's worst change (3.6 mm; 3.2 out of sample): both
  trex r8 seeds re-seated their feet in the settle by 3.6–4.4 times their statue's change, and the review's 0.02 m is
  21% of this 0.094 m stance. The reward's width term is referenced to the animal's own width at the end of the
  settle, not the keyframe's, so a re-seat earns nothing.
- *The foot bars sit mid-gap*, as on the velociraptor, but with a reward term that sees them: support-geom coverage is
  priced on the gate's substep rule (`support_geom_coverage_weight`: a geom is loaded on a substep when its floor
  contacts' normal forces sum above 0.1 N, and the term pays each geom loaded on at least half of the step's
  substeps), and the reward adds a rule of its own that the gate does not apply: a digit counts only through its
  contacts beyond the capsule's centre, away from the MTP joint, because the capsule starts at that joint, under the
  pad, and its joint end carries load whether or not the tip reaches the floor. The gate counts a digit by any
  contact, so its bar is the looser of the two, and on the statue it reads duty 1.000 from step 200, while the
  reward's rule seats the last tip by step 308. The review's training lane had found a learned near-statue under an
  earlier candidate reward that left digits II and III unloaded (coverage 0.5) and failed the gate 0/20, the
  reward-gate mismatch §8 avoided on the velociraptor; the critique then found rolled feet paid 0.90–0.93 of the
  statue. The committed reward pays rolled feet 0.852–0.892 and the gate refuses them outright.
- *The window hop pair.* A leg is down on a step when it is loaded on half of the step's substeps, so the window's
  support, flight and touchdown bars see a hop only when its flight lasts half a control step or more. This plant's
  two-foot hop at 10–12.5 Hz flies for 1–3 of its ten 2 ms substeps a step, so a hop started after the settle reads
  both legs down on every step, no touchdown and the statue's settle: the review measured the block without the pair
  certifying such a hop on 40/40 (before the armature), and on the landed plant 61 of the 120 episodes at a = 0.07 and
  10 Hz fail nothing else. The kind gains two optional criteria for it, applied only where declared and needing no
  statue panel (`stance_gate_v2.py`; undeclared on the trex and the velociraptor, whose digests do not move):
  `max_window_airborne_substeps`, the window's both-feet-unloaded substeps, and `max_window_peak_floor_force_bw`, its
  peak summed floor force, the settle bars' own detectors read over steps 200–999. A report row or panel CSV recorded
  before the pair reads both as unmeasured, which fails only a gate that declares them. Both bars are set on
  square-wave hops. The airborne bar, 40, sits about at the geometric middle of the jittered statue's worst at σ 0.03
  (23) and the softest-landing square-wave hops' least (72, at a = 0.05); the peak bar, 2.0 BW, between that jitter's
  worst and the least of the 410 square-wave post-settle hops at a = 0.07–0.15 (2.22), but not mid-gap: on the
  committed noise stream the jitter's worst is 1.75, and on seeds 11042–11081 with an independent stream
  (`default_rng(880001 + 7919 i)`) 1.88 (2 of 40 above 1.75), so the bar sits 0.12 BW above the worst σ 0.03 episode
  measured, and a trained mean a little noisier than that starts failing on noise. At σ 0.05 the jittered statue lands
  at 1.78–2.27 BW (1.82–2.52 on the independent stream), inside the a = 0.07, 10 Hz hop's own range (2.22–2.42), and
  the peak bar refuses 64 of its 120 episodes (23 of 40 on that stream): no pair of window bars both admits that
  jitter and refuses the hop (hop seed 9076 lands at 2.23 BW with 9 airborne substeps, inside the jitter's 2.27 and
  29), so the bar refuses the hop with a margin rather than admit the noise. The reward prices the two alike (0.935
  and 0.906–0.917 of the statue). The pair narrows the gap the step-level bars leave on this plant but does not close
  it: a softer hop, and a foot that lifts for less than half a step, pass every bar (the hack table's last six rows,
  and Limits).

| Panel (seeds) | Plant | Clean | LCB | Verdict | Fewest bars failed per unclean episode | Mean reward |
|---|---|---|---|---|---|---|
| statue (3042–3081) | r2 | 40/40 | 0.928 | PASS | — | 4570.4 |
| statue (7042–7081) | r2 | 40/40 | 0.928 | PASS | — | 4565.8 |
| statue (9042–9081) | r2 | 40/40 | 0.928 | PASS | — | 4570.4 |
| statue, N(0, 0.005) jitter (3042–3061) | r2 | 20/20 | 0.861 | — | — | 4569.3 |
| statue, N(0, 0.01) jitter (3042–3061) | r2 | 20/20 | 0.861 | — | — | 4557.2 |
| statue, N(0, 0.02) jitter (3042–3081) | r2 | 40/40 | 0.928 | PASS | — | 4507.2 |
| statue, N(0, 0.03) jitter (3042–3081; 7042–7081; 9042–9081) | r2 | 40/40 each | 0.928 | PASS | — | 4441.5; 4442.9; 4440.5 |
| statue, N(0, 0.05) jitter (3042–3081; 7042–7081; 9042–9081) | r2 | 18/40; 17/40; 21/40 | 0.315; 0.292; 0.385 | FAIL | 1 (window peak) | 4273.7; 4270.0; 4272.2 |
| statue spawned yawed +45°, −45° (3042–3051) | r2 | 10/10 each | 0.741 | — | — | 4571.4; 4571.5 |
| statue spawned yawed +90°, −90° (3042–3061) | r2 | 20/20 each | 0.861 | — | — | 4569.9; 4570.0 |
| PPO seed 11, 200k steps, deterministic (3042–3081) | r2 | 40/40 | 0.928 | PASS | — | 4569.5 |
| PPO seed 12, 200k steps, deterministic (3042–3081) | r2 | 40/40 | 0.928 | PASS | — | 4569.2 |
| PPO seed 11 spawned yawed +90°, seed 12 −90°, deterministic (3042–3061) | r2 | 20/20 each | 0.861 | — | — | 4551.0; 4566.0 |
| PPO seeds 11 / 12, stochastic (3042–3061) | r2 | 0/20 each | 0.000 | FAIL | 8 / 7 | 3457.4 / 3501.4 |
| statue (3042–3061) | r1 | 20/20 | 0.861 | — | — | 4558.9 |
| `20260921_203149` robust_best, deterministic (3042–3061) | r1 | 0/20 | 0.000 | FAIL | 9 | 2788.8 |
| `20261001_225856` robust_best, deterministic (3042–3061) | r1 | 0/20 | 0.000 | FAIL | 8 | 2780.5 |

Mean rewards are the committed stance `[env]`'s; the r1 rows are the r1 plant's statue and marches under the same
terms, so 2788.8 and 2780.5 are 0.612 and 0.610 of their own statue. A 20-episode panel cannot pass (its bound at
20/20 is 0.861, but `min_eval_episodes` is 40), so those rows give the clean count only. The two PPO runs show only
that the deterministic mean does not drift to a march or a hop in the first 200k steps (window airborne 0, window
peak 1.000 BW); the trex r8 runs entered a hop regime by 2–4M, seed 42 freezing into it by 6.5M while seed 44 left it
at 5–6M. Their stochastic rollouts (σ 0.130) reach the horizon on 20/20 each but bounce (touchdowns 1.2–2.4 per foot
per second, both feet down 0.90–0.95 of the window, window peaks 3.0–4.9 BW): the exploration noise bounces this
plant, and the gate reads the mean.

Episodes failing each bar (of 20):

| Panel | Bars failed (episodes) |
|---|---|
| `20260921_203149` | support, touchdowns, support-geom duty and coverage, sole tilt, corner lift and contacts, window peak 20 each, settle width change 20, displacement 5 |
| `20261001_225856` | support, touchdowns, support-geom duty and coverage, sole tilt, corner lift and contacts, window peak 20 each, displacement 7, settle width change 1 |
| PPO seed 11, stochastic | support, touchdowns, displacement, support-geom duty and coverage, sole contacts, window airborne and window peak 20 each, settle airborne and settle peak 17 each |
| PPO seed 12, stochastic | support, touchdowns, displacement, support-geom coverage, sole contacts, window airborne and window peak 20 each, support-geom duty 19, settle peak 13, settle airborne 11 |

Scripted hacks on the r2 plant (10-episode gate panels from seed 3042 unless noted; the post-settle hops also on seeds
7042 + i and 9042 + i; the last six rows are 40-episode panels through the report path, seeds 3042–3081, with the
report's own statue panel), with their reward over the same-seed statue's on that panel. The hops are square waves
unless noted. The scripts command raw actions; under the soft-cubic map a leg command of a moves its servo targets b =
0.1a + 0.9a³ of the span (a = 0.15: about 1.8%). The offsets are servo-target offsets in radians.

| Hack | Clean | Bars failed (episodes) | Reward / statue |
|---|---|---|---|
| both hip-roll offsets +0.05 rad (both pads rolled the same way) | 0/10 | sole tilt, support-geom duty and coverage 10 each | 0.877 |
| hip roll rolled outward (right −0.05, left +0.05) | 0/10 | sole tilt, support-geom duty and coverage, settle width change 10 each | 0.892 |
| both hip-roll offsets +0.10 rad | 0/10 | sole tilt, support-geom duty and coverage, corner lift, sole contacts 10 each | 0.852 |
| hip roll rolled outward ±0.10 rad | 0/10 | the same and settle width change, 10 each | 0.857 |
| 60-step settle stomp (knees ±a, ankles and hip pitch ∓a, then zero), a = 0.15 at 5 Hz | 0/10 | settle peak 10, settle airborne 8 | 0.989 |
| the same, a = 0.2 at 5 Hz | 0/10 | settle airborne and settle peak 10 each | 0.986 |
| the same, a = 0.3 at 8.3 Hz | 0/10 | settle airborne and settle peak 10 each | 0.956 |
| two-foot hop from step 200 (the same pattern, held on), a = 0.05 at 10 Hz | 0/10 | window airborne 10, displacement 1 | 0.912 |
| the same, a = 0.05 at 12.5 Hz | 0/10 | window airborne, window peak and displacement 10 each | 0.841 |
| the same, a = 0.07 at 10 Hz (3042–3081, 7042–7081, 9042–9081) | 0/120 | window peak 120, window airborne 103, displacement 59 | 0.906–0.917 |
| the same, a = 0.07 at 12.5 Hz (3042–3051, 7042–7081, 9042–9081) | 0/90 | window peak and displacement 90 each, window airborne 71 | 0.866–0.868 |
| the same, a = 0.10 at 10 Hz (3042–3051, 9042–9081) | 0/50 | window peak and displacement 50 each, window airborne 44 | 0.876–0.882 |
| the same, a = 0.10 at 12.5 Hz; a = 0.12 at 10 and 12.5 Hz | 0/10 each | window airborne, window peak and displacement 10 each | 0.830; 0.775, 0.707 |
| the same, a = 0.15 at 10 Hz (3042–3081, 9042–9081) | 0/80 | window airborne, window peak and displacement 80 each | 0.657 |
| the same, a = 0.15 at 12.5 Hz (3042–3081) | 0/40 | window airborne, window peak and displacement 40 each, support-geom coverage 12 | 0.647 |
| the same, a = 0.10 at 10 Hz, from step 600 | 0/10 | window airborne and window peak 10 each | 0.930 |
| whole-episode 12.5 Hz hop, a = 0.05 | 0/10 | displacement, window airborne and window peak 10 each, settle airborne 7 | 0.791 |
| the same, a = 0.07 | 0/10 | displacement and window peak 10 each, window airborne 5, settle airborne 4 | 0.836 |
| the same, a = 0.08 | 0/10 | displacement and window peak 10 each, window airborne 9, settle airborne 8 | 0.838 |
| the same, a = 0.10 | 0/10 | displacement, settle peak, window airborne and window peak 10 each, settle airborne 8 | 0.831 |
| swept-back crouch (hip pitch +0.3, ankles −0.2 rad) | 8/10 | displacement, settle airborne and window airborne 2 each | 0.847 |
| one-leg stance (hip roll −0.03 rad, then the left knee, ankle and hip pitch 0.2 rad): the right foot at 0.4 N | 0/10 | displacement, both foot-share bars and the statue ratio, support-geom duty and coverage, sole tilt, corner lift and contacts 10 each | 0.626 |
| weight shift (hip roll +0.06 rad, then the left knee +0.3 and hip pitch −0.3 rad): 39/61 on both feet | 0/10 | foot share and the statue ratio, support-geom duty and coverage, sole tilt, corner lift and contacts 10 each, windowed share 5 | 0.768 |
| two-foot bounce from step 200, the post-settle hop's pattern driven by a sine, a = 0.09 at 5 Hz | **40/40** | — (both feet off on 0–32 window substeps, on 39 of 40 episodes; landing 1.70–1.80 BW) | 0.989 |
| the same, a = 0.12 at 2.5 Hz | **40/40** | — (0–27 substeps; 1.33–1.54 BW) | 0.990 |
| one 10 Hz square cycle of that pattern every 0.4 s from step 200, a = 0.04 | **39/40** | window airborne 1 (0–41 substeps; 1.65–1.66 BW) | 0.974 |
| the right leg alone (knee +a, ankle and hip pitch −a) from step 200, a = 0.07 at 6.25 Hz | **40/40** | — (the right foot lifts for one substep 130–190 times an episode; never both feet) | 0.972 |
| the right toe alone, ±0.25 at 10 Hz from step 200 | **40/40** | — (the right foot lifts 146–175 times an episode) | 0.984 |
| the legs in antiphase from step 200 (a sine, a = 0.10 at 6.25 Hz) | **40/40** | — (a foot unloads at most twice an episode; the lighter foot carries under 20% of the floor load on 44–48% of window steps, down to 14%) | 0.905 |

The square-wave post-settle hops are 410 episodes at a = 0.07–0.15: the window peak bar refuses all 410, the window
airborne bar 368 and the displacement guard 349, and on 61 of the 120 at a = 0.07 and 10 Hz, as on the hop from step
600, nothing but the pair refuses them. Every one has both legs down on every window step, no touchdown and a statue's
settle; the 42 that fly 40 substeps or fewer all land at 2.23 BW or more. The whole-episode hop at a = 0.07 is a
low-amplitude two-foot hop: up to 200 airborne window substeps per episode, landing at 2.24–2.59 BW. The swept-back
crouch is a level, bilateral stance on every support geom that no floor-truth bar refuses (its two unclean episodes
walk off), which the `[env]`'s leg pose term prices instead (0.887 on the 20-episode re-score, 0.804 and 13/20 clean
on seeds 11042–11061; the posture wanders, so its ratio is seed-dependent). Two other open-loop one-leg shifts fell on
tilt (steps 292–294 and 465).

The block certifies the table's last six rows. Driven by a sine, the post-settle hop's pattern lands at 1.33–1.80 BW,
under the peak bar, and flies 0–32 window substeps, no more than the admitted jitter; the 5 Hz bounce lifts each foot
16–54 times an episode and moves the root through 1.0–1.1 mm, about the vertical travel of the refused square-wave
hops at a = 0.05–0.07 (0.8–1.1 mm). One 10 Hz cycle every 0.4 s flies 0–41. A leg is down when it is loaded on half a
step's substeps and the pair counts only substeps with both feet off the floor, so the one-leg pump and the toe tap,
whose foot lifts for one substep at a time, read both legs down on every step and no airborne substep, and the
antiphase shuttle unloads a foot at most twice an episode; its lighter foot's share falls under 20% on 44–48% of
window steps, but both foot-share bars average over the window or over 1 s blocks. The reward keeps 0.905–0.990 of the
statue on all six: the legacy substep-minimum support gate prices a substep with both feet unloaded (the 5 Hz bounce's
raw alive is 968–999 of 1000) but not a one-foot unload, which leaves the other foot's minimum above it (the one-leg
pump keeps raw alive at 999–1000), so only the bilateral and coverage terms price a one-foot chatter.

Margins: the least favourable episode in each group, toward the bar (the r2 statue on seeds 3042–3081, in brackets
seeds 7042–7081 and 9042–9081; the jittered statue on all three blocks, and in braces, where given, on seeds
11042–11081 with an independent noise stream; the PPO deterministic means; the 410 square-wave post-settle hops; the
two marches):

| Bar | r2 statue | Jitter σ 0.03 | Jitter σ 0.05 | PPO means | Post-settle hops | March episodes (r1) |
|---|---|---|---|---|---|---|
| support ≥ 0.98 | 1.000 (1.000) | 1.000 | 1.000 | 1.000 | 1.000 | ≤ 0.111 |
| touchdowns ≤ 0.25/s | 0 (0) | 0 | 0 | 0 | 0 | ≥ 2.97 |
| displacement ≤ 0.05 m | ≤ 0.03 mm (0.03) | ≤ 21 mm | ≤ 39 mm | ≤ 0.08 mm | 0.038–0.155 m; 349 over | 0.011–0.216 m; 12 of 40 over |
| load share ≥ 0.40 | ≥ 0.499 (0.499) | ≥ 0.493 | ≥ 0.493 | ≥ 0.471 | ≥ 0.460 | ≥ 0.448 |
| windowed share ≥ 0.35 | ≥ 0.496 (0.496) | ≥ 0.467 | ≥ 0.465 | ≥ 0.468 | ≥ 0.453 | ≥ 0.381 |
| saturation ≤ 0.10 | 0 (0) | 0 | 0 | 0 | 0 | ≤ 0.004 |
| settle airborne ≤ 15 | 0 (0) | ≤ 5 | ≤ 10 | 0 | 0 | ≤ 9 |
| settle peak ≤ 3.0 BW | ≤ 1.07 (1.08) | ≤ 1.77 | ≤ 2.12 | ≤ 1.10 | ≤ 1.08 | ≤ 2.27 |
| settle width change ≤ 0.011 m | ≤ 3.6 mm (3.2) | ≤ 4.2 mm | ≤ 4.0 mm | ≤ 3.7 mm | ≤ 3.6 mm | 0.1–33 mm; 21 of 40 over |
| support-geom duty ≥ 0.50 | 1.000 (1.000) | ≥ 0.939 | ≥ 0.810 | 1.000 | ≥ 0.745 | 0 |
| support-geom coverage ≥ 0.80 | 1.000 (1.000) | ≥ 0.972 | ≥ 0.904 | 1.000 | ≥ 0.751; 12 under | ≤ 0.135 |
| sole tilt ≤ 2.0° | ≤ 0.021° (0.020) | ≤ 0.12° | ≤ 0.20° | ≤ 0.050° | ≤ 0.27° | ≥ 7.0° |
| corner lift ≤ 3 mm | ≤ 0.016 mm (0.016) | ≤ 0.11 mm | ≤ 0.18 mm | ≤ 0.04 mm | ≤ 0.26 mm | ≥ 7.4 mm |
| sole contacts ≥ 1.5 | ≥ 3.83 (3.84) | ≥ 2.72 | ≥ 2.32 | ≥ 3.62 | ≥ 2.09 | ≤ 0.002 |
| window airborne ≤ 40 | 0 (0) | ≤ 23 {≤ 15} | ≤ 29 {≤ 24} | 0 | 3–535; 368 over | 3–20 |
| window peak ≤ 2.0 BW | ≤ 1.0002 (1.0002) | ≤ 1.75 {≤ 1.88} | 1.78–2.27; 64 over {1.82–2.52; 23 of 40 over} | ≤ 1.000 | 2.22–4.41; all over | 2.08–2.48; all over |
| foot-on-foot, phantom, non-foot | 0 (0) | 0 | 0 | 0 | 0 | 0 |

The window bars alone (support and touchdowns) and the foot bars alone (duty, coverage, tilt, corner lift, contacts)
each refuse every march episode; the settle bars do not, because the marches' settle is quiet (at most 9 airborne
substeps and 2.27 BW). No step-level bar refuses a post-settle hop by itself; the window peak bar refuses every
square-wave one. The armature took the statue's corner lift from 0.15 mm to 0.016 mm.
`environments/shared/tests/test_compsognathus_stance_gate_config.py` pins that the support, sole and foot families
each refuse a recorded march episode alone, that a recorded post-settle hop is refused by the window pair and by each
of its keys alone, that the window peak refuses both the softest-landing square-wave hop and the σ 0.05 jitter that
lands like it, that the statue's and the jittered statue's least favourable episodes are clean, that the settle width
bar stays about three times the statue's worst, and that the sine bounce's and the one-leg pump's least favourable
episodes are clean (the blind spot above, pinned so that closing it updates these records);
`environments/shared/tests/test_stance_gate_v2_report.py` rolls a real-physics post-settle hop through the report and
the judge, which refuse it, beside a zero-action statue they certify. Report-only, never gated: the r2 statue yaws up
to 2.6° over an episode (2.7° out of sample), and its spawn peak (the first 0.1 s, before the grace ends) reaches 1.97
BW (1.93 out of sample, 2.01 on seeds 11042–11081).

**Limits.** The marches are r1 policies on the r1 plant, and the hacks are open-loop scripts; the r2 policies a
retrain produces may find what neither found, and a v2 verdict on one is only as good as these bars. The realization
effect of §2.3 applies. The only r2 policies measured are the statue, the jittered and yawed statue and two 200k-step
PPO runs. The window pair is set on square-wave hops, and it narrows the gap the step-level bars leave on this plant
without closing it: a two-foot hop that lands under 2.0 BW and flies no more window substeps than the jitter passes
it, and the same pattern driven by a sine at 2.5–5 Hz does, at 0.989–0.990 of the statue's reward. On this plant a
foot that lifts for less than half a control step, and a weight shuttle faster than 1 Hz, are invisible to every v2
bar: a one-leg pump, a toe tap and an antiphase shuttle certify on 40/40 (the hack table's last rows). No count bar
separates these from noise: the σ 0.03 jitter the block admits by design flies up to 23 two-foot window substeps and
lifts each foot off 53–99 times over the window, against 130–190 for the measured one-foot flutters, 16–54 for the 5
Hz bounce and 109–156 for σ 0.05 jitter; a periodicity criterion, for example a spectral peak of the window's floor
force, is the measured follow-up (KNOWN_ISSUES), and until it lands each trained stance's per-foot lift-offs and root
vertical oscillation are checked by hand (the compsognathus recipe review's amendment). The peak bar already refuses
the statue under σ 0.05 command noise on about half its episodes, so a learned stance as jittery as that fails on
noise it does not choose. The single-support tiptoe stays physically possible on r2, and
`max_actuator_saturation_fraction` reads commands, so it is blind to the MTP servo's force saturation that holds a
tiptoe; the support-geom and sole bars refuse it instead. The anatomical observation reads the world heading (the
pelvis quaternion, linear velocity and target direction in the world frame) and every spawn faces +x, so a trained
policy could learn a heading-dependent stance; the r2 statue is clean at every spawn yaw tried (±45° and ±90°), and
the two short PPO means with seed 11 at +90° and seed 12 at −90° (20/20 each), but a trained stance is untested, and
spawn yaw is not randomised in training. The noise cliff (σ 0.10: 40/40, σ 0.135: 151/160, σ 0.20: 0/40) applies to a
policy as much as to the statue: a run whose exploration σ grows trains on a plant that falls.
