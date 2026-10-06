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
