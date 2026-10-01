# Gait-quality plan (2026-09)

**Status**: proposed 2026-09-28; nothing decided. Landed with PR-G0, the docs-only records PR. Written at `c8b66a6`
(PR-B's head), whose tree #566's merge `2b9219d` equals, so line numbers hold at `2b9219d`; re-read before editing.
The evidence is the dated note [investigations/GAIT_AUDIT_2026_09.md](investigations/GAIT_AUDIT_2026_09.md):
2026-09-28 replays of every certified node on Drive, each reproducing its recorded numbers. Nothing here enters
D-D21's 0.3.9 gate; every code PR builds on 0.3.9.

## How to use this document

§1 says what the replays found, what the plan does and where the cleanup it waits for stands; §2 lists the decisions,
which the maintainer takes (each takes the next free D-D id, D-D23 onward since D-D22 went to the cleanup plan's decision 16 on 2026-09-29, appended to
[BEHAVIOR_RECIPES_PLAN.md](BEHAVIOR_RECIPES_PLAN.md) §6.2 and the consolidation plan's table); §3 defines the
measurement, §4 the gates and their calibration, §5 the reward changes, §6 the PR order, §7 the records, §8 the risks,
§9 what the plan leaves alone, §10 a prompt for continuing the work in a fresh session. Companions:
[CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md), [CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md),
[NEXT_STEPS.md](NEXT_STEPS.md), [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [PLANT_CONTRACT.md](PLANT_CONTRACT.md).

## 1. Summary

### 1.1 What the evidence shows

"Floor truth": the real floor normal force on a leg exceeds 0.1 N on at least half the substeps of a control step.
"Touch": the env's own reading (summed touch sensors, minimum over substeps, above 0.1 N).

| Node (run, stage) | Gate | Verdict | What it really does (floor truth) | Intent met |
|---|---|---|---|---|
| trex `20260914_123816` stance (seed 42) | `stance_quality/v1` | PASS | Quiet wide stance; both feet down 0.9999, drift 5.9 cm | Yes |
| trex `20260914_123816` locomotion | `reward_and_length/v1`, v ≥ 1.0 | PASS, 1.05 m/s | Two-footed hop at 7.7 Hz, 35% airborne, simultaneous touchdowns 0.97, alternation 0.010, left foot 14.7 cm ahead | No |
| trex `20260920_010912` stance (seed 44) | `stance_quality/v1` | PASS | Quiet in 34/40 episodes; hops to rebalance in 6/40 (drift up to 0.57 m), hidden by the panel-mean duty 0.0069 | Partly |
| trex `20260920_010912` recovery | `recovery_quality/v1` | PASS, 28/40 | Keeps posture, but answers 40 of 41 forward pushes with two-footed hops and hops between pushes | Balance only |
| trex `20260925_033501` locomotion (seed 44) | `reward_and_length/v1`, v ≥ 1.0 | PASS, 1.57 m/s | Same hop at 9.2 Hz, 43% airborne, simultaneous 0.99, 70 cm stance | No |
| velociraptor `20260922_125248` stance | `reward_and_length/v1` | PASS | Feet chatter at about 10 Hz, 15% airborne, slides 0.65 m; the statue passes too | No |
| velociraptor `20260922_125248` locomotion | `reward_and_length/v1`, v ≥ 2.0 | PASS, 3.17 m/s | Genuine alternating run (alternation 0.93, simultaneous 0.018); a 9.8 Hz scurry, stride 0.70 L | Yes, a run |
| compsognathus `20260921_203149` stance | `stance_quality/v1` | PASS | Marches in place at 3.05 Hz; both feet down 10% | No |
| compsognathus `20260921_203149` locomotion | `reward_and_length/v1`, v ≥ 0.08 | PASS, 0.36 m/s | Genuine walk: alternation 0.997, flight 0.004, 4.07 Hz, stride 0.37 L | Yes |
| compsognathus_robot `20260924_031815` stance | `stance_quality/v1` | PASS | On the left foot (81% of load) with the right sole resting on it; reported two-foot support 0.9988 is 0.62 | Still, not two-footed |
| compsognathus_robot `20260924_031815` locomotion | `reward_and_length/v1`, v ≥ 0.04 | PASS, 0.23 m/s | Micro-hop at 9.2 Hz from that stacked stance; 48% airborne while touch reports 97% two-foot support; about 838 of about 2627 reward paid for phantom support | No |
| dibothrosuchus `20260923_020654` stance | `reward_and_length/v1` | PASS | The statue; right forefoot under 2% of load in 6/40 panel episodes (lifted for a whole episode in 1/30) | Statue only |
| dibothrosuchus `20260923_020654` locomotion | `reward_and_length/v1`, v ≥ 0.9 | FAIL | Statue; `gait_symmetry` pays 1997.3 of 2249.9 | — |
| dibothrosuchus `20260928_012318` locomotion | `reward_and_length/v1`, v ≥ 0.9 | training | 4.3M robust-best: three-legged skid at 2.20 m/s displacement speed (gate speed 1.95), 34% airborne, feet slipping 1.2–2.0 m/s; clears the gate as written | Would not be |
| brachiosaurus `20260717_162659` locomotion (July, r1) | July rail v ≥ 0.75 | no verdict | Neck-pumping asymmetric bound | No |

Three facts explain it (audit note §3): no locomotion gate reads a foot contact (the six test mean root velocity, with
reward floors 3–22× below the statue); the stance gates admit stances that chatter, march, stand on one foot or hop
(the zero-action statue passes all six, by design); the robot's touch sensors count sole-on-sole contact as floor
support.

### 1.2 What the plan does

1. **Measures gait on floor truth** through the existing per-substep hook (`base_env.py:171`, called at :1180-1181),
   on evaluation envs only, so no digest moves (§3).
2. **Judges each episode** and certifies with the existing exact binomial bound (`binomial_lcb`,
   `curriculum/recovery_gate.py:63`): `locomotion_gait/v1`, later `stance_quality/v2` and `recovery_quality/v2` (§4).
3. **Retires the hoppers without touching them.** When the locomotion TOMLs adopt the new kind, reuse rule 7
   (`ancestors.py:40-60`) refuses every verdict judged under the old gate; the two genuine walkers are re-paneled into
   fresh runs (§6, PR-G4/PR-G5).
4. **Changes what the rewards pay** through per-species task revisions (`gait-r1`) built from new constructor kwargs
   whose legacy defaults reproduce today's arithmetic, merged just before each retrain (§5).
5. **Orders the work** so no genuine walker is stranded: locomotion gate first; stance and recovery gates only inside
   revisions that retrain those chains anyway.
6. **The check waits for 0.3.9**; only PR-G0, the docs-only records PR, may land earlier (GQ-1).

### 1.3 Where the cleanup stands

The gait code waits for the 0.3.9 cut (§1.2 item 6), which follows the last PR of the cleanup's gate. On 2026-09-28
(`main` = `2b9219d`) the cleanup is not finished. Landed: the cleanup plan (#560), CU-1 (#561), CU-3 (#562), the 0.3.8
release cut (#563) and the backend retirement, PR-A (#564), PR-A2 (#565) and PR-B (#566), which completes D-D17. Ten PRs
of D-D21's gate remain, in [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §3 order: CU-2, CU-4, CU-5, CU-7, CU-8,
CU-9, CU-11, CU-12, CU-14 and CU-16; then the 0.3.9 cut. Deferred, not dropped: CU-6 (now waits only for CU-4), CU-10
(lowest priority; it also fixes the curriculum horizon defect), CU-13 (before PR-11), CU-15 (optional) and CU-17 (last,
before PR-15). Consolidation has landed PR-1 (#542), the automatic trunk selection (#543), PR-2 (#544), PR-3 to PR-6
(#546–#549), PR-12's notebook-only slice (#552), PR-14a/b/c (#553–#555) and PR-7 (#556); PR-8, PR-9, PR-10, PR-11, the
rest of PR-12, PR-13 and PR-15 remain and build on 0.3.9
([CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md)). PR-G1 needs CU-9; the later G PRs interleave with
PR-8..PR-10 and precede PR-11's trex and robot sessions and PR-13 (§6). *Updated 2026-10-01: CU-2 landed as #568, CU-4
as #569, CU-14b (the first part of CU-14, the cleanup plan's decision 10 (c), settled without a D-D id) as #570 and the
digest-snapshot check of D-D22 (ROW-16) as #571, all on 2026-09-29, and CU-7a (the first part of CU-7) as #572 and
CU-11 (the reward, info and termination golden) as #573, both on 2026-09-30, which completes wave 1, and CU-9
(coverage of the certification code, which PR-G1 needs) as #574 the same day, and CU-5 (the SB3 notebook's text and
dead parameters) as #575 the same day, and CU-7b (the second part of CU-7, the retired-backend wording) as #576 the
same day, which completes CU-7, and CU-16a (the docs text of CU-16) as #577 the same day, which completes wave 2;
CU-16b (the orphan assets of CU-16) as #578 the same day, which completes CU-16; CU-8a (the first part of CU-8, one
task-fingerprint derivation) as #579 on 2026-10-01; CU-12 (the species env dedup, which PR-G6's GQ-16 refers to) as
#580 the same day, which completes CU-12; CU-14a (the workflow structure of CU-14, with the cleanup plan's decision
10 (a), settled on 2026-09-30 without a D-D id) is carried out,
and the rest of the gate follows the order the maintainer accepted in the cleanup plan's §3.1 item 4;
CU-6 also waits for CU-8b and the notebook PR for that plan's decisions 4 and 6.*

## 2. Decisions needed

| # | Question | Options | Recommendation (why) | Blocks |
|---|---|---|---|---|
| GQ-1 | Does 0.3.9 wait? | (a) Everything, records included, after the cut. (b) PR-G0 (records, docs only) lands when PR-B merges, outside D-D21's gate list (neither in CU-16 nor delaying the cut); all code after the cut. (c) The gate inside 0.3.9. | **(b).** The living docs must be true now (`README.md:485` calls a hop a certified walker, and KNOWN_ISSUES holds verified unfixed defects); the check itself waits for the cut, as the maintainer asked. Gate revisions move `gate_sha256` by design and cannot enter a no-digest base. | Nothing on D-D21 |
| GQ-2 | The in-progress dibothrosuchus re-run `20260928_012318` | (a) Let it finish and be judged; label it "gait audit: three-legged skid; not a behavior parent". (b) Stop it now (saves about 5 h, leaves an unjudged node: cleanup decision 6's hazard). | **(a).** By 4.8M its in-training evaluation fell to 855 (the replay: 19/20 tilt); unless a later evaluation beats 3686, the handoff stays the 4.3M skid; the verdict records that v1 admits a scramble, and PR-G5 retires it. Also confirm from `ancestors/stance/ancestor.json` whether it reused the statue-level stance `1d3527cc…` (the replay's reading) though `NEXT_STEPS.md:257` planned `RETRAIN_FROM = "stance"`. | — |
| GQ-3 | New sessions meanwhile | (a) Hold session 5 until brachiosaurus `gait-r1`; no trex or robot hunt/follow session; no robot recovery session; no dibothrosuchus hunt session; no walker session on today's rewards (amends G3). (b) Session 5 stance only. (c) As planned. | **(a).** The statue passes brachiosaurus stance (1739.1 against 1040) and earns 2200 of 2242.7 on the walk; about 14.5 h would buy nothing reusable. Children inherit gaits (the robot's stacked stance became its hop). | Session 5 |
| GQ-4 | The hopping certificates (trex `20260914_123816`, `20260925_033501` locomotion; robot `20260924_031815` stance and locomotion) | (a) Untouched; labelled in NEXT_STEPS; rule 7 retires the three locomotion nodes at PR-G5; the robot stance is retired at the robot stance `gait-r1` (PR-G7/PR-G8). (b) An exclusion list read by `select_trunk` (`ancestors.py:1051`; amends D-A25). (c) Sidecars in the runs. (d) Rewritten verdicts. | **(a).** (c) and (d) are refused: complete bundles are immutable (`result_bundle/reentry.py:11-22`). Until then the exposure is a trex or robot hunt/follow session, a robot `BEHAVIOR="stand"` session, and a dibothrosuchus hunt on the re-run if GQ-2 (a) certifies it. | — |
| GQ-5 | Gate, reward, or both | (a) Gate only. (b) Reward only. (c) Both: gate report-only then enforced; reward kit inert; each species' revision just before its retrain. | **(c).** A gate alone turns each retrain into an 8 h FAIL; rewards alone certify nothing. | PR-G5..PR-G9 |
| GQ-6 | Contact source | (a) Floor truth: gates via the hook (no digest); rewards via a `support_source` kwarg (legacy `"touch"`) inside each task revision. (b) Touch sensors. | **(a).** Touch shows 0.50–0.57 phantom support on the robot and reads handoffs as flight (velociraptor 49% against 33%; dibothrosuchus 0.39 body weights against 1.00). Touch stays in the observation. | PR-G1 |
| GQ-7 | Certification statistic | (a) Per-episode qualification, binomial LCB over a 40-episode panel (seeds 3042–3081, D-B17's role), bar 0.80 (37/40). (b) Bar 0.70 (34/40). (c) Panel means, as v1. (d) A 60-episode panel. | **(a).** Means hid trex seed 44's hop bouts and admit lunge-then-fall. Against `stance_gate.py:53-61`: at 37/40 a policy qualifying on 99% of episodes fails 0.1% of panels, 96.7% (velociraptor's fall rate) 4%, 95% 14%, 90% 58%. | PR-G3 |
| GQ-8 | What `locomotion_gait/v1` enforces first | (a) The separators that differ by at least 4× (alternation, simultaneous, lead swaps, flight, foot-on-foot and phantom by ≥ 10×; hop-flight 4.7×) and hold on both the certified and the final checkpoints in the CPU replays (§4.3; stability across realizations is PR-G5's five-realization acceptance), per-episode length and displacement speed, and a flight cap on bipeds whose gate speed is below Froude 0.5; the rest report-only, provisional like D-B2. (b) The full literature table. (c) (a) plus `min_stride_over_L` 0.3 (report-only or enforced): genuine walkers 0.336–0.742 L including the final checkpoints, hops 0.114–0.218 L. | **(a).** Literature bands fail the genuine walkers (§4.4). Tightening later is a re-judge, not a retrain. | PR-G5 |
| GQ-9 | Quadrupeds | (a) Enforce a minimal set now: flight ≤ 0.05, minimum limb duty ≥ 0.20, fore and hind pair alternation ≥ 0.75; Hildebrand band report-only. (b) Report-only until a quadruped walker is accepted on video. (c) Full set with the band. | **(a).** The false pass is live (the skid); no quadruped walker is certified, so nothing is stranded; walk, trot and pace all meet these floors, which PR-G1's synthetic fixtures must prove. | PR-G5 |
| GQ-10 | Order of gate revisions | (a) Locomotion on all six first; stance v2 per species only inside a revision that retrains that chain (the robot with its `gait-r1`); recovery separately. (b) All at once. | **(a).** Stance v2 fails the compsognathus and velociraptor stances, and `select_trunk` stops at a run's first refused node, stranding both genuine walkers; (b) also costs at least 15 h more. | PR-G8 |
| GQ-11 | Re-certifying the genuine walkers (compsognathus `20260921_203149`, velociraptor `20260922_125248`) | (a) Re-panel: copy the certified handoff pair byte-identically into a fresh run with its `ancestors/` records; JUDGE rolls the panel. (b) A re-judge record that reuse follows (amends D-A23). (c) Retrain both walks (3h49m + 6h30m). (d) Re-judge in place. | **(a).** No reuse rule changes; it has the widen tool's shape (a node judged in a new run). (d) is refused: both bundles are complete. | PR-G4 |
| GQ-12 | Stance v2 bar against trex's two-seed bar | 0.80: seed 44 fails (35/40, LCB 0.755), trex stance falls to 1 of 2 seeds. 0.70: it passes. | **0.80**, applied when trex stance adopts v2, with a third seed scheduled then. | PR-G8 |
| GQ-13 | `recovery_quality/v2` (support clause) | (a) Register; adopt at the trex recovery's next revision, re-freezing the nulls; compsognathus_robot recovery adopts v2 inside the robot `gait-r1` (its `[env]` follows the stance through CU-13's `extends`; its `[curriculum]` names v2 directly, since `extends` never inherits `[curriculum]`), compsognathus at its next recovery session. (b) Adopt now: the certified seed-44 recovery fails (5/40). (c) Never. | **(a).** | PR-G9 |
| GQ-14 | Robot feet | (a) Keep the sole-to-sole collision (contype/conaffinity 5/26 and 9/22, `compsognathus_robot.xml:181,254`); add floor-truth support and a foot-collision penalty. (b) `<exclude>` the pair: physics revision r1 → r2, every robot digest moves, and legs pass through each other in sim only. (c) Narrower soles or wider hips (physics and hardware). | **(a)**; (c) if the stance pilot shows stacking persists. | Robot `gait-r1` |
| GQ-15 | Alternation incentive | (a) A clock-free event reward at floor-truth touchdowns (§5.3). (b) A phase clock in the observation: a policy-interface revision widening cannot carry (only the +3 command columns, `PLANT_CONTRACT.md:93-97`), so fresh stances everywhere. (c) Penalties only. | **(a)**; (b) if pilots stall. | PR-G7 |
| GQ-16 | Where the reward kit lands | (a) After PR-9, which edits the same constructors and deletes the behavior identity. (b) Before PR-9 under CU-12's acceptance. | **(a)**, unless the robot retrain must precede PR-9. | PR-G6 |
| GQ-17 | Retrain order, robot budget and cap | Robot, trex seed-42 walk, dibothrosuchus, brachiosaurus. Robot stance 11M over two sessions (best evaluation came at 10.15M) or 8M. Robot cap 0.15 or 0.25 m/s. | **As listed; 11M; the cap by pilot.** The robot is the hardware target; trex follow recipes need a trex walker. | PR-G7 |
| GQ-18 | PR-13 scope | (a) `terrain_command/v1` gains a per-episode gait clause (appended to D-D6). (b) A later kind. | **(a).** The certificate tracks pelvis velocity, so it would certify a hop or slide. | PR-13 |

## 3. The measurement

### 3.1 The gait library (`environments/shared/gait/`)

It only reads `data` and edits no species env file (those are byte-hashed, `CLEANUP_PLAN_2026_09.md:330`).

- **`morphology.py`**: feet from `_foot_sensor_groups` (2 or 4); floor geoms from `_static_floor_geoms`
  (`base_env.py:1349`), so heightfields count; animal geoms from the free-joint subtree without prey or food; body
  weight from the kinematic subtree, never `mj_getTotalmass` (`STAGE1_SPLIT_PLAN.md:1028-1033`); a per-species registry
  of foot geoms (touch-site bodies plus declared digits and pads).
- **`recorder.py`**: `SubstepContactRecorder` sets `_substep_probe_hook`, chaining any previous hook, and calls
  `mj_contactForce` only on contacts that touch the animal. Per control step it stores per-limb fraction of substeps
  above 0.1 N and mean floor force, limb-to-limb contact, non-foot floor force, the env's
  `_aggregated_foot_contact_forces()` (`base_env.py:915`) and kinematics. Evaluation envs only: a Python contact loop
  costs about 50 µs per substep (`base_env.py:1174-1176`).
- **`events.py`**: a limb is down when its floor force exceeds 0.1 N on at least half the substeps, the audit's floor-truth channel, which
  agreed with video wherever a frame could resolve the contact (audit §2.5; hops, taps and 20 ms flights come from the
  contact data alone). Debounce and merge windows are 20 ms, defined in seconds (2 steps is 20 ms at 0.01 s
  but 40 ms at 0.02 s); left and right touchdowns within 20 ms are one simultaneous event `B`, per pair on quadrupeds.
- **`metrics.py`**: `episode_gait_metrics(trace, morph, settle)`, a frozen dataclass; an undefined value is NaN,
  which gates treat as unmeasured (`finite_gate_metric`, `gate_schema.py:377`).
- **`report.py`**: rolls the handoff pair with the plant contract enforced; writes `gait_report.json`
  (`mesozoic.gait-report/v1`: definitions, thresholds scored under, handoff sha256s, verdict) and a hash-bound
  per-episode `gait_panel.csv` with CU-3's atomic writers.
- **`curriculum/gait_gate.py`** (pure numpy, like `stance_gate.py`): thresholds, `classify_episode`,
  `evaluate_gait_gate`. **`harnesses/gait_probe.py`**: the audit probe on the library, with the copies'
  `--video-frame-dt` fix (replays at 0.01 s steps play at half speed).

The down-state uses whole-limb floor force, as the calibration did. The foot/non-foot split depends on the registry
(the velociraptor right metatarsus carries 18% of standing load) and stays report-only until a test shows each of the
six statues putting 99% of its floor load on foot geoms.

### 3.2 Why floor truth

| Node | Touch | Floor truth | Cause |
|---|---|---|---|
| robot locomotion | two feet 0.97, flight 0.002 | flight 0.48 | right sole on the left; 1.12 body weights of sole-on-sole force per sensor |
| robot stance | two-foot 0.9988 | 0.62 | same |
| velociraptor run / stance | flight 0.49 / 0.25 | 0.33 / 0.15 | minimum over substeps drops brief contacts |
| compsognathus stance | unsupported 0.013 | 0.000 | handoffs inside one 20 ms step |
| dibothrosuchus skid | 0.39 body weights | 1.00 | 10–30 ms contacts lost to the minimum |
| trex, all nodes | agrees; misses 5–8% at contact edges on locomotion, under 2% on stance and recovery | — | — |

### 3.3 Metric definitions

Window: after 1 s for locomotion, after `settle_steps` for stance. Pairs: (L, R); fore (LF, RF) and hind (LH, RH).

| Metric | Definition | Status |
|---|---|---|
| Flight fraction | steps with no limb down ÷ window steps | enforced (walk profile) |
| All-limb support s_F; low support s₀+s₁ | fraction of steps with every / at most one limb down | s_F enforced (stance) |
| Hop-flight fraction | flight steps in intervals not opened by exactly one limb lifting and closed by exactly one other limb landing (±20 ms), ÷ window steps | enforced (bipeds) |
| Alternation; simultaneous | on the merged touchdown sequence: foot-switching pairs ÷ pairs; `B` events ÷ events (per pair on quadrupeds) | enforced |
| Lead swaps per stride | sign changes of the debounced fore-aft offset x_L − x_R along the heading ÷ strides | enforced (bipeds) |
| Limb duty | fraction of steps down; minimum over limbs | enforced (quadrupeds) |
| Load share | a limb's mean floor force ÷ total | enforced (stance v2) |
| Phantom support; foot-on-foot | touch > 0.1 N while floor says up; substeps with limb-to-limb contact | enforced |
| Touchdown rate; window displacement | touchdowns ÷ (limbs × seconds); root displacement | enforced (stance v2) |
| Episode speed | window displacement along the mean heading ÷ window time | enforced (replaces the mean of per-episode means) |
| Left/right phase φ (circular μ, R, PCI); Hildebrand limb phase; same-foot; duty asymmetry; stride against 2.3·Fr^0.3·L; clearance; slip; non-foot contact; Froude; pelvis bob; joint lag | the probe's definitions | report |

## 4. The gates

### 4.1 Kinds and schema rules

- `locomotion_gait/v1` (new); `stance_quality/v2` (v1 stays registered for historical verdicts);
  `recovery_quality/v2`, which joins `FROZEN_NULL_GATE_KINDS` (`gate_schema.py:139`) and re-freezes its nulls.
- Each enters `GATE_KINDS` (:54) and `_REQUIRED_THRESHOLD_KEYS` (:148) in one commit.
- Keys are numeric (a string fails `same_threshold`, :343) and new: `min_avg_forward_vel` keeps its mean meaning in
  v1, so `GATE_SCHEMA_VERSION` (:49) stays 1.
- Undeclared keys are not projected into the gate digest (`gate_config_view`, :305-328): registering moves nothing.

| Kind | Panel keys (required \*) | Criterion keys |
|---|---|---|
| `locomotion_gait/v1` | `min_eval_episodes`\*, `min_gait_episode_lcb`\*, `min_episode_forward_vel`\*, `min_episode_length`\*, `min_strides_per_episode`\*, `required_consecutive`, `min_avg_reward` (collapse rail only) | `min_alternation_index`\*, `max_simultaneous_fraction`\*, `max_hop_flight_fraction`, `min_lead_swaps_per_stride`, `max_foot_contact_fraction`, `max_phantom_support_fraction`, `max_flight_fraction`, `min_limb_duty`, `min_pair_alternation_index` |
| `stance_quality/v2` | `min_eval_episodes`\*, `min_clean_stance_lcb`\*, `settle_steps`\*, `min_full_horizon_fraction`, `min_avg_reward` (rail) | `min_all_feet_support`\*, `max_touchdown_rate`\*, `max_window_displacement`\*, `min_foot_load_share`\*, `max_foot_contact_fraction`, `max_phantom_support_fraction` |
| `recovery_quality/v2` | v1's | v1's plus `max_push_hop_touchdowns`, `min_dwell_all_feet_support`, `max_unpushed_touchdown_rate` (touchdowns per foot per s outside push windows, bar 0.25 as stance v2) |

### 4.2 Per-episode against panel criteria

An episode **qualifies** when it reaches `min_episode_length`, its displacement speed reaches
`min_episode_forward_vel`, it has `min_strides_per_episode` strides, and every declared criterion holds with every
metric finite. The gate passes when `binomial_lcb(k, n) ≥ min_gait_episode_lcb` with `n ≥ min_eval_episodes`. This
closes the audit's lunge-then-fall example (three episodes at 0.7 m/s and one at 2.0 m/s falling at step 250 pass v1
on means). Stance v2 counts clean episodes the same way. In training the manager screens on the raw qualifying
fraction with `required_consecutive` as hysteresis, as stance v1 does (`stance_gate.py:43-51`); the certificate comes
only from the post-training panel on the handoff pair.

### 4.3 Thresholds and calibration

Per-episode [min–max] on the 2026-09-28 replays (30 episodes, evaluation seed scheme, unless noted); † a
representative trace only; — not measured. PR-G2 rolls the 40-episode certification panel before any value freezes.

**A. Biped locomotion** (bar 0.80: 28/30 or 37/40)

| Node | Length ≥ rail | Speed ≥ bar (m/s) | Hop-flight ≤ 0.15 | Alternation ≥ 0.75 | Simultaneous ≤ 0.15 | Lead swaps ≥ 1.0 | Foot-on-foot ≤ 0.02; phantom ≤ 0.05 | Flight ≤ 0.05 | Qualifying, LCB |
|---|---|---|---|---|---|---|---|---|---|
| compsognathus walk | 780–1000 (900) | 0.348–0.378 (0.08) | 0–0.001 | 0.987–1.0 | 0–0.007 | 1.38–1.79 | 0; 0 | 0.001–0.007 | **29/30, 0.851** |
| compsognathus, final | 1000 | 0.333–0.353 | — | 0.987–1.0 | 0–0.007 | 1.55–1.88 | 0; 0 | 0–0.006 | **30/30, 0.905** |
| velociraptor run | 59–1000 (750) | 3.34–3.55 (2.0) | 0.002–0.069 | 0.857–0.983 | 0–0.043 | 1.86–2.16 | 0; 0 | not declared | **29/30, 0.851** |
| velociraptor, final | 1000 | 3.40–3.63 | 0.001–0.040 | 0.947–1.0 | 0–0.029 | 1.90–2.01 | 0; 0 | not declared | **30/30, 0.905** |
| trex s42 hop | 1000 (750) | 1.06–1.21 (1.0) | 0.323–0.374 | 0–0.055 | 0.865–1.0 | 0–0.015 | 0; 0 | 0.324–0.374 | 0/30 |
| trex s42, final | 1000 | 1.08–1.21 | — | 0–0.058 | 0.914–1.0 | 0–0.030 | 0; 0 | 0.327–0.374 | 0/30 |
| trex s44 hop | 1000 | 1.65–1.75 | 0.40–0.41† | 0–0.013 | 0.963–1.0 | 0–0.05 | 0; 0 | 0.397–0.446 | 0/30 |
| robot hop | 1000 (900) | 0.215–0.224 (0.04) | 0.46–0.49† | 0–0.049 | 0.811–0.894 | 0–0.011 | **0.953–0.964; 0.553–0.592** | 0.461–0.497 | 0/30 |
| robot, final | 1000 | 0.217–0.224 | — | 0–0.049 | 0.853–0.920 | 0 | 0.949–0.966; 0.526–0.579 | 0.466–0.497 | 0/30 |

Every hop clears its velocity bar and fails at least four independent criteria. The tightest genuine margins are
velociraptor's alternation (0.857 against 0.75) and compsognathus's lead swaps (1.38 against 1.0). The
compsognathus fall happens on CPU only (0 falls on 100 fresh seeds). The flight cap is declared where the gate speed
is below Froude 0.5 (trex 0.12, compsognathus 0.003, robot 0.0008); velociraptor (Froude 0.82) may fly, but hop-flight
confines flight to alternating footfalls, and grounded running (bouncing mechanics, duty near 0.5, no aerial phase)
passes everywhere.

**B. Quadruped locomotion** (no passing example exists)

| Node | Length ≥ 750 | Speed ≥ bar | Flight ≤ 0.05 | Min limb duty ≥ 0.20 | Fore alternation ≥ 0.75 | Hind alternation ≥ 0.75 | Qualifying |
|---|---|---|---|---|---|---|---|
| dibothrosuchus `20260923_020654` (statue) | 1000 | 0.0000–0.0003 (0.9) | 0 | 0–1 | no strides | no strides | 0/30 |
| same run, final checkpoint | 20–62 | 0.61–0.89 | 0.16–0.61 | 0–0.18 | 0–1 | 0–1 | 0/30 |
| re-run 4.3M (20 episodes) | 286–1000 | 1.48–2.57 | 0.22–0.39 | 0.02–0.12 | 0.07–0.32 | 0.24–0.53 | 0/20 |
| re-run 4.8M (20 episodes) | 80–1000 | 1.20–2.59 | 0.15–0.46 | 0–0.08 | 0–0.27 | 0–0.48 | 0/20 |
| brachiosaurus July (0.75) | 1000 | 1.20–1.77 | 0.013–0.041 | 0.25–0.42 | 0.18–0.87 | 0.19–0.53 | 0/30 |

Only pair alternation fails the July bound (flight 0.013–0.041 and limb duty 0.25–0.42 pass): hind alternation fails
30 of 30 episodes, fore alternation 29 of 30, so the quadruped set cannot drop pair alternation. Report-only
Hildebrand band: brachiosaurus [0.10, 0.45] (lateral-sequence walk or amble, Froude 0.06); dibothrosuchus [0.10, 0.60]
(walk to trot, Froude 0.28).

**C. Stance** (`stance_quality/v2`; bar 0.80 at n = 40 needs 37/40)

| Node | All-feet support ≥ 0.98 | Touchdowns per foot per s ≤ 0.25 | Displacement ≤ 0.10 m | Min load share ≥ 0.30 biped / 0.05 quadruped | Foot-on-foot; phantom | Clean, LCB |
|---|---|---|---|---|---|---|
| trex s42 | 0.996–1.0 | 0–0.125 | 0.001–0.069 | 0.453–0.494 | 0; 0 | **40/40, 0.928** |
| trex s44 | 0.951–1.0 | 0–1.19 | 0.000–0.494 | 0.431–0.500 | 0; 0 | 35/40, 0.755 |
| compsognathus (march) | 0.084–0.109 | 2.94–3.16 | 0.006–0.089 | 0.432–0.448 | 0; 0 | 0/40 |
| robot (stacked) | 0.552–0.756 | 1.03–2.78 | 0.005–0.053 | 0.135–0.283 | 0.998–1.0; 0.231–0.448 | 0/40 |
| velociraptor (chatter; 30) | 0.447–0.913 | 1.61–13.2 | 0.12–1.07 | 0.319–0.457 | 0; 0 | 0/30 |
| dibothrosuchus (40-episode panel) | 0–1.0 | 0 | 0.000–0.003 | 0–0.105 | 0; 0 | 27/40, 0.534 |
| statues: compsognathus, robot, dibothrosuchus | 1.0 | 0 | ≤ 0.0012 | 0.139–0.500 | 0; 0 | 40/40, 0.928 |
| statue: velociraptor (30) | 1.0 | 0 | ≤ 0.021 | 0.492–0.500 | 0; 0 | 29/30 (1 fall), 0.851 |

Windows: the `stance_quality` stances skip the stage's 200-step settle; the velociraptor and dibothrosuchus stances
(`reward_and_length`) skip 1 s (100 steps). The statue passes by design ("match, not beat", `stance_gate.py:13-20`).
v2 covers four feet, where `derive_stance_info` reads only r/l, the forefeet on quadrupeds (`stance_diagnostics.py:75`).

**D. Recovery, trex seed 44** (bar 0.30): v1 34/40 (LCB 0.725) on the CPU replay, 28–34 over five chaos
realizations. v2 with no `B` touchdown per push window: 5/40 (0.051); at most one: 9/40 (0.123); at most two: 15/40
(0.247). 60 of 155 pushes involve two-footed touchdowns. 37% of the touchdowns fall outside push windows, which only
`max_unpushed_touchdown_rate` reads; its seed-44 value is not yet computed.

### 4.4 Report-only metrics, and why

| Metric (literature band) | Genuine walkers | Hops |
|---|---|---|
| Stride within 0.67–1.5× of 2.3·Fr^0.3·L | compsognathus 0.35–0.38 L (0.38×); velociraptor 0.65–0.74 L (0.23×) | 0.11–0.22 L |
| Walking duty 0.55–0.65 | compsognathus right foot 0.49–0.52 | trex seed 42 0.56–0.66, seed 44 0.48–0.53 |
| Robinson duty symmetry ≤ 10% | compsognathus 15–21%; velociraptor 13–30% | trex 0–9% |
| PCI ≲ 5% | 25–36% (10–12 steps per stride) | — |
| \|μφ − 0.5\| ≤ 0.15 | 0.06†, 0.04†, 0.07† | 0.44–0.49† |
| Slip while down | velociraptor 0.9–1.1 m/s (toe roll); compsognathus 0.05 | 0.07–0.35 |
| Clearance ≥ 0.05·h | compsognathus 0.04 L | trex 17–25 mm; robot 6–7 mm |

Symmetry and slip bands pass the hops and fail the walkers; a mirror-symmetry loss cannot catch a hop (a synchronous
hop is mirror-symmetric). Phase separates well but was measured on eight traces; it becomes enforceable once PR-G2's
panels confirm it.

### 4.5 Plumbing

- **Manager**: a branch beside `manager.py:290-297`; the `else` stays fail-closed (:298-311). The supplementary
  evaluation (`advancement.py:225-277`, called at :329) and the standalone evaluation (:350-432) run their eval env
  through the recorder.
- **Judge**: `evaluate_stage_gate` (`reporting/gates.py:688`) gains an arm before the closed fall-through (:776-784)
  that reads the verdict off `gait_report.json`, as `_stance_stage_gate` (:213) does, refusing a report scored under
  other thresholds or another handoff pair; `evaluate_recorded_gate` (:60) returns None without the gait counts.
- **Artifacts**: `_write_gait_report` beside `_write_stance_gate_report` (`stage_artifacts.py:174`, called at :1466),
  through `_apply_stage_gate` (:1021); `_PERSISTED_STAGE_RESULT_KEYS` (`result_bundle/gate_verdict.py:66`) gains the
  count, panel size and LCB; `gait_report_episodes` joins `_DIAGNOSTIC_KEYS` (`gate_schema.py:225`).
- **Backfill** refuses gait kinds without a matching, hash-bound report, as it refuses stance re-derivation
  (`backfill_gate_verdict.py:13-20`); no Drive artifact holds per-step contacts.
- **CLI**: the in-training verdict (`train_base.py:2683-2697`) comes from EvalCallback seeds and may score another
  checkpoint; for gait kinds the CLI builds the gait report on the handoff pair and writes the verdict from it.
- **Invariant 10**: fail-closed and "is consulted" tests per kind (`BEHAVIOR_RECIPES_PLAN.md:1237-1240`).

## 5. Incentive fixes

### 5.1 What pays for each exploit

| Node | What pays |
|---|---|
| robot locomotion | alive 0.5, posture 0.4, height 0.4 are multiplied by `support` = touch sum > 4% of body weight (0.62 N; `compsognathus_env.py:189,239`): one foot, or a foot on the other, is enough. On 99.6% of steps by touch, 34.3% by floor. Paid while airborne: forward 564, alive 240, posture 189, height 188. Speed at the 0.15 m/s cap on 90% of steps |
| robot stance | the same flag at weights 1.0: one-leg standing is paid in full |
| trex locomotion | alive 0.5 with no contact condition (`support_conditioned_alive_fraction` defaults to 0, `trex_env.py:149`): paid airborne, alive 164 / 205 and forward 261 / 475 (seeds 42 / 44); nothing rewards alternation |
| velociraptor stance | alive 1.75, no contact term: 276 per episode paid in flight |
| dibothrosuchus, brachiosaurus locomotion | `gait_symmetry` pays the standing policy 1997.3 of 2249.9 and the brachiosaurus statue 2200 of 2242.7 (history never decays; reset counts as touchdowns); dibothrosuchus pays speed flat above 2.0 m/s |

### 5.2 How a reward change becomes a named revision

1. **The task digest does not cover reward code**: `task_sha256` hashes the effective `[env]` kwargs
   (`task_fingerprint.py:117-172`), and only CU-11's golden checks reward arithmetic. So every change is a new
   constructor kwarg whose legacy default reproduces today's arithmetic bit for bit; setting it in a TOML is the
   revision.
2. **Carve-out**: a new kwarg enters every stage's effective config through the signature (:133-136).
   `_effective_env_kwargs` pops each kit key absent from `[env]` and equal to its legacy value, like the compsognathus
   push keys (:141-158) and the command keys (:159-170). `config.save_stage_config` (`config.py:756-788`), which
   records every constructor default in `reward_weights`, gets the same carve-out, so the recorded `stage_config.json`
   and the harness's `stage_config_view_sha256` lines are unchanged at defaults.
3. **Hashed code untouched**: `_get_obs`, `_scale_action`, `reset`, and `_cache_ids` on the compsognathus pair
   (`plant_contract/policy_layer.py:360-361,397,399-404`). Kit state resets in `_reset_gait_state`
   (`base_env.py:700`), which every species' `_spawn_target` already calls; the substep loop (:1178-1185) is not
   hashed. No MJCF or frozen-core edit; the `foot_contact_*` knobs stay.
4. **Byte-hashed files**: the kit's calls in the species env modules move each touched species' `behavior` identity
   until PR-9 deletes it (GQ-16). At defaults CU-11's golden and the digest-snapshot harness are otherwise unchanged.
5. **Naming**: each species×stage TOML that sets kit knobs is revision `gait-r1`, recorded in a decision row, the
   CHANGELOG and a TOML comment. Re-measure the statue (`zero_action_baseline.py`), re-derive the absolute
   `collapse_peak_floor`s, and name a gate revision alongside if a rail moves. Old verdicts then fail rule 3, and
   anything warm-started from them fails rule 4, so each TOML merges just before its session; pilots use `--override`.

### 5.3 The kit (`environments/shared/gait_rewards.py`, inert at defaults)

| Knob (legacy value) | Effect |
|---|---|
| `support_source` (`"touch"`) | `"floor"`: support flags and bilateral terms read per-substep floor force on foot geoms, computed only when a knob needs it |
| `gait_phase_weight` (0) | at each debounced floor-truth touchdown of foot i: w · (T_i ÷ (feet · dt)) · P · A · S. P: biped exp(−(d(φ, 0.5) ÷ 0.15)²), 0 when both feet land within 0.1·T\* or the other foot has not landed since; quadruped: match against `lateral_walk` or `trot`. A = min(swing, T_sw) ÷ T_sw, T_sw = 1.3·√(L/g). S = clip(Δx ÷ ℓ, 0, 1), how far the landing foot lands ahead (bipeds). Statue, hop and pronk earn 0; stagger and march earn 0 through S |
| `flight_penalty_weight` (0), `flight_min_feet` | −w on steps with fewer feet down (1 biped walk, 2 quadruped walk; not the velociraptor run) |
| `foot_slip_penalty_weight` (0) | −w · Σ\|v_xy\| ÷ √(gL) over feet down |
| `foot_collision_penalty_weight` (0) | −w on steps with foot-to-foot contact |
| `leg_contact_penalty_weight` (0), `terminate_on_leg_contact` (False) | penalise or terminate on named shin, thigh and tibia geoms touching the floor |
| compsognathus `bilateral_support_weight`, `foot_contact_saturation_force`, `support_conditioned_alive_fraction` | trex's knobs under their names (`trex_env.py:143-149`), fed floor forces |

`gait_symmetry` stays byte-identical (a fix in place would change two tasks without moving their digests); revised
TOMLs set its weight to 0 (cleanup §2 row 15, :106), and its KNOWN_ISSUES entry (:596-615) stays while any TOML uses it.

### 5.4 Per-species revisions

| Stage | `gait-r1` starting values | Moves | Retrain |
|---|---|---|---|
| robot stance | floor support; bilateral 0.6, saturation 4.7 N (0.3 body weight), conditioned alive 0.5; collision 0.5; slip 0.2; stance v2 | task and gate digests; its recovery via CU-13's `extends` | 11M (19h31m measured), two sessions |
| robot locomotion | floor support; gait phase 0.4 (T_sw 0.19 s, ℓ 0.06 m); flight 0.5; collision 0.5; slip 0.2 | task digest; follow nodes via `extends` | 3M, about 4h38m |
| trex locomotion | gait phase 0.3 (T_sw 0.39 s, ℓ 0.56 m); flight 0.5; slip 0.2; `TRUNK_FROM = "20260914_123816"` (`auto` picks seed 44 on the tie) | task digest | 8M, 8h46m |
| dibothrosuchus stance, locomotion | stance: `terminate_on_leg_contact` (shins 0.047 m up), stance v2. Locomotion: gait_symmetry 0; gait phase 1.0 (`lateral_walk` or `trot`); flight 0.25 (min 2); slip 0.5; leg termination; cap 2.0 → 1.2 m/s (Froude 0.5, above the 0.9 bar) | task and gate digests | about 4.2 h + 8.1 h |
| brachiosaurus stance, locomotion | leg termination (shins 0.067 m), stance v2; gait_symmetry 0; gait phase 1.0 (`lateral_walk`); flight 0.26 (min 2); slip 0.5 | task and gate digests | about 4.5 h + 10 h (session 5) |
| optional: compsognathus, velociraptor stances; trex recovery | floor support, bilateral 0.6; velociraptor adds flight 1.0 and a metatarsus penalty | their digests; strands their walkers (GQ-10) | about 14 h; 4h46m; 3h30m |

About 60 h of sessions if every first attempt succeeds; budget 120 h. The robot's touch observation still sees
sole-on-sole force; it waits for the robot's next policy-interface revision (the one carrying `SIM_TO_REAL_PLAN.md` §6
item 6), whose checklist must add a touch reading that excludes foot-on-foot force.

### 5.5 Validation ladder

1. **Unit tests** on scripted sequences (statue, hop, staggered hop, walk, scoot, march, pronk, bound, trot, lateral
   walk) pin every term; statue, hop and pronk earn 0 from `gait_phase`.
2. **CPU re-scoring in minutes**: reward-only knobs do not change dynamics, so deterministic replays of the certified
   checkpoints follow the same trajectories. Pass: each exploit's net change ≤ 0 with P ≤ 0.1; both genuine gaits
   reach P ≥ 0.9. Robot: the hop loses 838 (phantom), 241 (flight) and 480 (collision), leaving about 1070, against
   an estimated 2670 for a walk at the cap.
3. **Pilots** from the command line into a scratch directory (`curriculum --trunk-from <run> --target <stage>
   --override <stage>.env.<knob>=…`), one seed, judged by the gait probe on 20 episodes against the certified run's
   periodic checkpoint at the same step count:

| Pilot | Steps | Proceed if | Stop if |
|---|---|---|---|
| robot stance | 2M | feet touching ≤ 5%, two-foot support ≥ 0.9, load share 0.4–0.6 | stacking persists (GQ-14 (c)) |
| robot locomotion, caps 0.15 and 0.25 | 1M each | simultaneous ≤ 0.3, alternation ≥ 0.6, flight ≤ 0.15, < 6 Hz | speed < 0.02 m/s |
| trex locomotion | 1.5M | alternation ≥ 0.6, ≥ 1 lead swap per stride, flight ≤ 0.2 | speed < 0.5 m/s |
| quadruped locomotion | 2M | every limb duty ≥ 0.3, two feet down ≥ 80%, P ≥ 0.5 | statue |

The robot's retrained walk also gets a hardware-readiness report, not a gate: airborne ≤ 0.02 (today 0.48),
alternation ≥ 0.9 (0.02), foot-to-foot contact 0 (96%), stride ≥ 0.5 L at ≤ 2.5 Hz (0.12 L at 9.2 Hz), clearance
≥ 1.5 cm (7–9 mm), torques inside the servo table (`SIM_TO_REAL_PLAN.md` §6 item 4, which must not be fed by the hop).

## 6. The PR sequence

Left of D-D21's gate (`CLEANUP_PLAN_2026_09.md:111`), with PR-B landed as #566: CU-2, CU-4, CU-5, CU-7, CU-8, CU-9,
CU-11, CU-12, CU-14, CU-16 (§1.3). The G PRs interleave with consolidation PR-8..PR-10 and precede PR-11's trex and
robot sessions and PR-13 (amends D-D13).

| PR | Scope | Size | Depends on | Digests | Acceptance | When |
|---|---|---|---|---|---|---|
| PR-G0 Records | §7: this plan, the audit note and its evidence files, KNOWN_ISSUES, NEXT_STEPS, README, CHANGELOG | docs, plus the evidence files under `investigations/gait_2026_09/` | PR-B (merged 2026-09-28 as #566) | none (**digest-free; may precede the cut**) | links resolve; each entry states how it was verified | Now: the maintainer approved opening it on 2026-09-28, outside D-D21's gate (GQ-1 itself stays open) |
| PR-G1 Library | §3.1 | about +1,100, tests +600 | CU-9 | none (new files) | reproduces the 2026-09-28 summaries on CPU for every audited node (the July brachiosaurus through a worktree at `e179198`, as the audit did); synthetic fixtures (walk, lateral-sequence walk, grounded and aerial run, hop, skip, scoot, stacked feet, march, chatter, trot, pace, pronk, bound, three-legged); the six-statue registry test; harness diff empty | First after the cut |
| PR-G2 Report-only | JUDGE writes `gait_report.json` and `gait_panel.csv` on every stage, declared in the bundle layout; no gate reads them; a four-foot `derive_stance_info`; `zero_action_baseline.py` prints the stance-gate verdict for `stance_quality` stages | about +400, tests +600 | PR-G1 | none | JUDGE time measured; the 40-episode panel rolled on every audited handoff and §4.3 re-frozen from it | With PR-8..PR-10 |
| PR-G3 Kinds | §4.1, §4.5 | about +500, tests +700 | PR-G2 | none (no TOML declares them) | fail-closed and "is consulted" tests; backfill refuses without a report | Before PR-13 |
| PR-G4 Re-panel | `scripts/repanel_checkpoint.py` (GQ-11) | about +350, tests +600 | PR-G3 | none | refuses a failed or non-handoff source and an occupied target; copies hash equal; the verdict records the source handoff | Before PR-G5 |
| PR-G5 Locomotion gate | six locomotion TOMLs declare `locomotion_gait/v1`; re-panel the two genuine walkers | TOML + tests | PR-G3, PR-G4 | **gate revision**: only those stages' `gate_sha256` and `stage_config_view_sha256` lines | fails both trex walkers, the robot, the skid and every statue; passes compsognathus and velociraptor (certified and final checkpoints) identically over five chaos realizations | Before PR-11 and any retrain |
| PR-G6 Reward kit | §5.2–5.3, inert | about +600, tests +500 | CU-11, CU-12, PR-9 (GQ-16) | none at defaults (`behavior` lines only if before PR-9) | CU-11 golden unchanged; harness diff empty; `plant_contract --check`; the §5.5 re-scoring table | After PR-9 |
| PR-G7 `gait-r1` | one TOML PR per species×stage (§5.4), merged just before its session | TOML | PR-G5, PR-G6, pilots; CU-13 for the robot stance (its recovery follows it through `extends`) | **task revision** (and gate revision if rails move) | statue re-measured; harness diff names only revised stages; retrained node passes PR-G5 | Robot, trex, dibothrosuchus, brachiosaurus |
| PR-G8 Stance v2 | per species inside its `gait-r1` (GQ-10, GQ-12) | TOML | PR-G3 | **gate revision** | the retrained stance passes; trex stance keeps v1 until a trex stance retrain (GQ-12's third seed), since adopting v2 without one makes rule 7 refuse the seed-42 stance that §5.4's `TRUNK_FROM = "20260914_123816"` reuses | With each revision |
| PR-G9 Recovery v2 | register; re-freeze nulls; adopt per species (GQ-13) | about +300 | PR-G3 | **gate revision** at adoption | §4.3 D reproduces | Later |
| PR-G10 PR-13 clause | `terrain_command/v1` reads `classify_episode` (GQ-18) | in PR-13 | PR-G3 | PR-13's | PR-13's | With PR-13 |

Only PR-G5, PR-G7, PR-G8 and PR-G9 move task or gate digests, each as a named revision; PR-G6 moves the touched
species' `behavior` identity lines only under GQ-16 (b), stated in its PR as `CLEANUP_PLAN_2026_09.md:330` requires.
Only PR-G0, a docs-only PR outside the gate, may land before the cut.

## 7. Records

**Investigation note.** `docs/investigations/GAIT_AUDIT_2026_09.md`, frozen once merged, with its evidence files in
[investigations/gait_2026_09/](investigations/gait_2026_09/README.md): the hand-run probe `gait_probe.py`,
`gait_audit_2026_09.csv` (one row per audited node and checkpoint), `SHA256SUMS` (the 153 per-node JSON, trace, plot
and contact-sheet files, 64.7 MB) and a README on regenerating them. The audit's 559 MB working tree is not in the
repository and is not uploaded anywhere; this plan first proposed Drive (`mesozoic-labs/investigations/gait_2026_09/`),
which the maintainer has not decided.

**KNOWN_ISSUES** (verified, unfixed items only):

| Entry (section, severity) | Verified | Leaves with |
|---|---|---|
| No locomotion gate reads a foot contact: three of five certified walkers hop, and the in-training dibothrosuchus skid would pass (Training / RL, HIGH) | replayed | PR-G5 and the retrains |
| The robot's feet stack and its touch sensors count sole-on-sole force (1.12 body weights per sensor): about 838 of about 2627 reward is phantom; the stance's 0.9988 two-foot support is 0.62 (Training / RL, HIGH, hardware target) | replayed | robot `gait-r1` |
| Stance gates admit non-stances: certified stances chatter and slide, march, stand on one foot, stack feet, or hop in 6/40 episodes that the panel mean hides; the statue passing is by design and stays under v2 (HIGH) | replayed | PR-G8 per species; open for compsognathus and velociraptor until their optional revisions |
| Locomotion criteria are means of per-episode means, so a lunge-then-fall passes (MEDIUM) | executed 2026-09-28 at `2b9219d` on trex locomotion through the repository's evaluation and gate code: three 1,000-step walks at 0.7 m/s and one 2.0 m/s lunge that falls at step 250 give mean length 812.5 and mean speed 1.025 m/s (recorded 1.02), and `evaluate_stage_gate` passes them (distance over time is 0.80 m/s); at the 30-episode panel 23 walks and 7 lunges pass both the post-training judge and the in-training `CurriculumManager`, 24 and 6 fail on speed | PR-G5 |
| The recovery safe set has no support clause (`min_foot_force_n = 0`); trex seed 44 answers forward pushes with two-footed hops (MEDIUM) | replayed | PR-G9 per species |
| Hunt success is counted on a step that ends in a height, tilt, nosedive, head-tip or skull-height fall (trex, velociraptor, brachiosaurus, dibothrosuchus); training and panel definitions disagree (MEDIUM) | executed 2026-09-28 at `2b9219d` on each hunt stage's env built from its TOML, body rolled past `max_tilt_angle`, prey or food on the success geometry: all four end `excessive_tilt` with the success flag at 1.0, counted by the panel (trex's `task_success/v1` passes 30/30) and as 0 in training; upright, the same placement counts in both. Floor contact is checked after success, so a tail strike on the contact step counts in both; compsognathus reads success after its fall checks and counts the falling step in neither | each hunt's task revision |
| The brachiosaurus statue reaches the food in 11/40 episodes (MEDIUM) | executed | its hunt task revision |
| Knee, shin and proximal-tail floor contact never ends an episode on four species (MEDIUM) | read; clearances measured | `terminate_on_leg_contact` in `gait-r1` |
| The stance diagnostic reads only the forefeet on quadrupeds (LOW) | read from the code | a four-foot `derive_stance_info` (PR-G2's scope) |
| `zero_action_baseline.py`'s FAILS line for `stance_quality` stages judges only the reward rail (LOW) | read from the code | PR-G2 prints the stance-gate verdict |

The lunge-then-fall and hunt-success rows were to be executed before PR-G0. Both checks ran on
2026-09-28 at `2b9219d` (scripted episodes through `eval_policy`, the stage judge and `CurriculumManager`; a scripted
falling step on each hunt env), both confirmed their row, and both rows enter KNOWN_ISSUES with that evidence.

Corrections: the gait-symmetry entry (:596-615) gains the measurement that a synchronous landing scores 1.000 on the
biped version; the foot-sensor section (:1000) gains the minimum-over-substeps finding (velociraptor 49% against 33%
flight; dibothrosuchus 0.39 body weights); the behavior-framework entry (:828-872) notes that the certificate has no
contact criteria; `ROADMAP.md:132` claimed a phase-difference metric the code does not compute; PR-G0 corrects it.

**NEXT_STEPS, README, CHANGELOG, decisions.** NEXT_STEPS §2 gives each audited node's row "gait audit 2026-09-28:
`<plain label>`" (for example "two-footed hop, not a walk"), keeping its certification facts untouched; §3 records
session 4's re-run and its reuse discrepancy and says that this plan recommends holding session 5 (GQ-3, open), not
that it is held; rows for the re-panels, pilots and retrains come with the decisions that schedule them; §7 adds the
risk of a gate without reward changes. `README.md:485` notes that the walker was certified on forward speed, reward
and episode length with no foot-contact check, and that the audit found it a two-footed hop. CHANGELOG `[Unreleased]`:
PR-G0 under Added; each revision under Changed, naming the digest lines it moves. Decision rows D-D23 onward (D-D22 went to the cleanup plan's decision 16 on 2026-09-29), append-only, in both plans.

## 8. Risks

- **Thin passing side**: two genuine gaits and one genuine stance calibrate the gate; hop-flight and phase come from
  four panels and eight traces; the quadruped floors have met no quadruped walk. Mitigated by PR-G2's full panels,
  synthetic fixtures and GQ-8's separators (at least 4× apart).
- **Hardware chaos**: episodes diverge between the recording GPU and a CPU replay (trex seed 44, seed 3042: duty
  0.0675 recorded, 0.009 replayed, 0.0625 after a 1e-7 nudge; recovery successes 28–34 over five realizations; one
  compsognathus fall on CPU only). Verdicts held and the hop bouts fell in the same six seeds, but counts moved: recovery
  28–34, and seed 3042 clears stance v2 on the CPU replay (duty 0.009) but probably not at the recorded 0.0675, so
  §4.3 C's 35/40 depends on the realization. PR-G5 requires identical
  verdicts over five realizations.
- **Registry errors** fail a good node or blind the non-foot check, which stays report-only until tested.
- **Sampling**: the robot's stride is 5 control steps; windows are in seconds; 2 s contact sheets alias hops, so video
  acceptance uses consecutive-frame sheets.
- **Rewards may not suffice**: the trex stage-1 bounce was not reward-preferred (KNOWN_ISSUES :33-44); the event
  reward may be too sparse for PPO warm-started at `log_std_init = -2.0`; removing phantom pay may make standing still
  easier (the dibothrosuchus precedent). Pilot stop rules bound the loss; the phase clock is the costly fallback.
- **Cost**: about 60 h of sessions, 120 h budgeted; the robot stance alone nears the 24 h Colab cap.
- **Record churn**: every gate revision refuses old verdicts; PR-G4 is a new certification tool.

## 9. What this plan does not do

- Rewrite, delete or re-judge in place any verdict or complete bundle; the hoppers stay on Drive as the record of what
  v1 certified.
- Move a task or gate digest outside PR-G5, PR-G7, PR-G8 and PR-G9, or a `behavior` identity line outside PR-G6
  under GQ-16 (b); edit the frozen MJX core; delete a `foot_contact_*` knob; change
  `gait_symmetry`'s code.
- Repair the velociraptor foot sensor (its planned plant revision, KNOWN_ISSUES :1000-1028) or the robot's touch
  observation; floor truth makes neither a prerequisite.
- Fix hunt success or the brachiosaurus food spawn, or wire the behavior certificate (PR-13 deletes it).
- Gate the hunts' approach gait (`task_success/v1` and the reward-and-length hunts keep no gait clause; a hunt child
  may regress to a hop).
- Reject an alternating shuffle (short alternating steps) unless GQ-8 (c) enforces `min_stride_over_L`.
- Gate on literature bands, joint phase, symmetry indices or sim-to-real targets; add motion priors or a backend; add a
  phase clock unless the pilots stall and GQ-15 (b) is decided, as its own policy-interface revision (fresh stances
  everywhere).
- Decide: §2 proposes; the maintainer decides.

## 10. Continuing this work

Paste the block below into a fresh working session to pick the work up.

```text
Continue the gait-quality work in the mesozoic-labs repository.

1. Read first: docs/GAIT_QUALITY_PLAN_2026_09.md (the plan), docs/investigations/GAIT_AUDIT_2026_09.md (its
   evidence), docs/NEXT_STEPS.md, docs/CLEANUP_PLAN_2026_09.md, docs/CONSOLIDATION_PLAN_2026_09.md and
   docs/KNOWN_ISSUES.md. Line numbers drift with every merge; re-read each file:line before relying on it. If
   PR-G0, the docs-only records PR that adds the plan and the audit note, has not merged, read them on its open pull
   request (the maintainer can name it) and ask the maintainer to review and merge it (GQ-1 is open; do not merge or
   rebuild it yourself).
2. Check the maintainer's answers to GQ-1..GQ-18 (plan §2; a taken decision appears as a D-D row in
   docs/BEHAVIOR_RECIPES_PLAN.md §6.2). Ask for any still open before building on it, especially GQ-2 (the
   in-training dibothrosuchus re-run), GQ-3 (sessions meanwhile) and GQ-5 (gate, reward or both).
3. Confirm whether the 0.3.9 release has been cut (the git tags, CHANGELOG, D-D21 in CLEANUP_PLAN §2). The
   maintainer wants the gait check after the cleanup. If the cut has not happened, the next work is the remaining
   cleanup PRs in CLEANUP_PLAN §3 order, not gait code.
4. Once 0.3.9 is cut, start PR-G1 (the gait library, plan §3.1; it depends on CU-9), then PR-G2 (report-only gait
   reports). Use docs/investigations/gait_2026_09/gait_probe.py and gait_audit_2026_09.csv as the calibration
   reference: PR-G1 must reproduce the audit's per-node summaries on CPU. Hop-flight and the stance-v2 scores in plan
   §4.3 are in neither the probe nor the CSV; compute them from §3.1's definitions. The per-node JSON, traces and plots are
   not in the repository; gait_2026_09/SHA256SUMS pins them, and gait_2026_09/README.md says how to regenerate them
   from the certified checkpoints on Drive.

House rules (binding):
- No digest moves except in a named revision; verify with environments/shared/harnesses/digest_snapshot.py on the
  base and the head.
- Gate verdicts and complete result bundles are immutable: never rewrite, delete or re-judge one in place.
- Decision rows are append-only, and decision ids never renumber.
- Existing knobs keep their names; the frozen MJX core stays untouched; the foot_contact_* knobs stay.
- KNOWN_ISSUES holds verified, unfixed items only.
- No AI model or vendor names in any repository file.
- Work on the branch the session names, one PR at a time, restarting it from origin/main after each merge.
- Before any push: the checks in docs/NEXT_STEPS.md §8 "Branch and validation rules" (ruff, mypy with SB3 and torch
  installed, the test suites, the notebook checks), and an adversarial review of the diff.
```
