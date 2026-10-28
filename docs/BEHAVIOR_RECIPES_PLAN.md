# Behavior Recipes Plan — from a stage ladder to a deliverable DAG

**Date**: 2026-09-05. **Baselined on**: `main` @ 640d2ce (the tree after the
six gap-review phases, PRs #514–#519, merged). **Companions**:
`docs/STAGE1_SPLIT_PLAN.md` (rev 5, the stance/recovery split and the
semantic stage manifest), `docs/STAGE1B_IMPLEMENTATION_PLAN.md` (the recovery
stage as a warm-started, post-gated mini-stage — the pattern this plan
generalises), and `docs/reviews/RL_PIPELINE_GAP_REVIEW_2026_08.md` (whose
behavior-gate and seed-multiplicity decisions, CF2/CF3/SS2 and SS1, this plan
absorbs). Statements about current code cite `file:line` in the baseline
tree; they will drift and should be read as anchors, not contracts.
Statements tagged `[probe]` were measured by throwaway scripts during the
2026-09-05 assessment and are re-measured by the tests §8 pins before the
corresponding phase lands.

*Note 2026-09-30 (cleanup CU-16a, #577, decision D-D17):* after this plan was
written, D-D17 retired the sweeps (cleanup PR-A, 2026-09-27) and the JAX/MJX
runtime (cleanup PR-B, 2026-09-28). Where §3, §4 and §7 describe a sweep, the
JAX runner, saver or preflight, the JAX notebook or its pin test, or an MJX
environment as present or still to be changed, they record the design of
2026-09-05: each decision row carries its D-D17 amendment (§6), Stable-Baselines3
is the only training backend, and the frozen MJX interface core only feeds the
plant contract (D-D17's row).

*Note 2026-10-16 (decision D-D35):* the recovery node's recipe label is
`recover`, not `stand`. Where §3 and §4 show `recipe = "stand"` on a recovery
entry, or read "stand" as stance → recovery, they record the design of
2026-09-05: `stand` is now the stance node alone on every model, `recover`
the recovery node of the T-Rex and both Compsognathus models, and D-A11 and
D-A24 carry the amendment (§6).

*Note 2026-10-18 (decision D-D38):* the stance node's recipe label is
`stance`, its own stage id, not `stand`. Where §1–§5, §7 and §8 and the
D-D35 note above show `stand` as the stance node's label (`recipe = "stand"`
on a stance entry, `"stand"` in a `BEHAVIOR` example or a label list, "a
label never equals a stage id"), they record the design before D-D38: the
loader now accepts a label equal to a stage id only when it resolves to that
stage, `"stand"` is refused naming `"stance"`, and D-A11, D-A24 and D-D35
carry the amendment (§6).

---

## TL;DR

The maintainer's goal: every species learns a named set of behaviors —
**stand/balance**, **walk**, **hunt/eat**, **follow direction** (move the way
a controller stick says) — each as its own policy, each built by a short
curriculum of mini-stages. The question was whether to move away from the
linear stage curriculum to "recipes" that target behaviors.

The answer is yes, and the tree is closer than the phrasing suggests. Three
of the four behaviors already exist as separately trained, separately gated,
separately warm-started checkpoints with their own TOMLs: stance and recovery
(standing), locomotion (walking), behavior (hunting). What the tree cannot do
is (a) declare which node warm-starts from which — parentage is inferred as
"the previous numbered stage" in four places, (b) publish more than one policy
per run — a failed bite gate suppresses the walking and standing policies
entirely, and (c) name a new behavior — the stage vocabulary is a closed set
of four ids. Follow-direction is the one behavior the environments cannot
express: there is no command input, only a static prey target.

The adopted design is a **DAG on one shared certified trunk**, not four
independent ladders:

- **Manifest v2** adds three optional keys per stage — `warm_start_from`,
  `deliverable` and a `recipe` label — and opens the id vocabulary. Recipes
  are *derived* from the edges (a deliverable plus its ancestor chain), never
  declared in a second table. Legacy numbers, config files, stage ids and
  task fingerprints do not move, so the certified stance evidence and any
  run's frozen recovery resolution stay valid.
- **Publication becomes per deliverable**: a bundle is complete for a
  deliverable iff that node and its ancestors passed; a run publishes every
  certified deliverable it contains; a failed leaf no longer hides the trunk.
- **Every deliverable is gated on a measured capability against a null with
  a confidence bound**, never on episode-mean reward. The hunting gate's
  unattainable 2.0 m/s term goes (speed is the walking deliverable's claim,
  carried by lineage), bite success moves to a Clopper-Pearson lower bound,
  and the hunting stage gets a measured collapse floor. Seed replication
  becomes a provenance field; single-seed deliverables are labelled
  provisional.
- **Follow-direction** is a two-node leaf chain warm-started from the
  certified walker. It needs one policy-interface revision that reserves three body-relative
  command dims (forward speed, lateral speed, yaw rate) at the end of every
  species' observation. That revision invalidates every existing checkpoint
  at three independent layers, so it is done **once, early, for all species
  at the same time**, and the certified stance checkpoint is *widened*
  (zero columns, re-paneled) rather than retrained. The follower starts as
  exactly the walker (zeroed command columns) and learns command dependence.
- **The SB3 notebook** gains a `BEHAVIOR` knob and one chain loop: it
  resolves the deliverable's ancestor chain from the manifest, reuses
  certified ancestors (from this run or an earlier one), trains only the
  missing nodes, and publishes whatever certified.

Sequencing (§5): the manifest edges and per-deliverable publication first
(no env change, no checkpoint invalidation), then the hunting-gate and
seed-provenance fixes, then the one interface bump, then the follow leaf
(pilots landed 2026-09-15 as a separate pipeline; consolidation pending,
§6.2).

---

## 1. Motivation

### 1.1 The proposal

Each species should learn four behaviors, each its own policy for now:
standing/balancing, walking, eating/hunting, and follow-direction ("move in a
certain direction, like a video-game controller would do"). Training should be
organised as *recipes* that target a behavior, with mini stages inside a
recipe for curriculum learning, instead of one linear ladder whose last rung
is the only product.

### 1.2 Why the ladder is the wrong shape for that goal

The ladder makes every stage a stepping stone and only the last a product.
Under the maintainer's goal every behavior is a product. Three consequences of
the ladder shape are load-bearing today:

1. **One published model per run.** Provenance's `selected_model_path` must
   equal the *terminal advancing* stage's checkpoint
   (`environments/shared/result_schema.py:557-566`), the bundle is
   `complete` only when every advancing stage passed
   (`environments/shared/reporting/bundles.py:158-161`), and `summary.json`
   is not written otherwise (`bundles.py:164`). A T-Rex run whose bite gate
   fails publishes neither its walking policy nor its stance policy.
2. **Parentage is inferred from position.** `train_curriculum` carries one
   `load_path` forward through a single loop
   (`environments/shared/train_base.py:1656-1660`), the stage-entry warm-up
   and reward ramp trigger on manifest position > 1
   (`train_base.py:857`), the notebook infers the load mode the same way
   (`notebooks/sb3_training.ipynb` cell 14, lines 239 and 245), and the JAX
   runner carries `params` and `obs_stats` forward through one integer loop
   (`environments/shared/jax_curriculum.py:503, 548-570`). A second leaf off the
   same parent cannot be expressed.
3. **Closed vocabulary.** `KNOWN_STAGE_IDS` is four ids
   (`environments/shared/stage_manifest.py:38`) and an unknown id is fatal in
   the reader and in `stage_label` (`:206`, `:253-254`). Integer references
   1/2/3 are pinned to stance/locomotion/behavior forever (`:43`,
   `:226-237`). "Follow-direction" has no identity anywhere.

### 1.3 Why not four independent ladders

Separate *policies* per behavior is right for now: SB3 has no multi-task
machinery, each behavior wants its own reward tuning and its own gate, and
the 2026-08-23 lineage rule already forbids an uncertified policy from feeding
another (`docs/STAGE1B_IMPLEMENTATION_PLAN.md` §4: feeding an ungated policy
forward "would launder uncertified robustness into the curriculum").
Separate *ladders* would be wrong: the stance stage alone costs 11M steps
(11h 30m per 10M-step seed run; 13h 20m for the one recorded 11M leg) and is
seed-sensitive (2 of 3 seeds certify,
`docs/KNOWN_ISSUES.md:59-97`), so every behavior must branch from one
certified trunk rather than re-roll it.

---

## 2. Vocabulary

| Term | Meaning |
|---|---|
| **Node** (mini-stage) | One stage TOML with `[stage]`, `[env]`, `[curriculum]`, `[ppo]` and `[sac]` tables and a `gate_kind` (`load_stage_config` has refused a `[jax]` table since D-D17, cleanup PR-B); one training run with one checkpoint and one gate verdict. Today's "stage". |
| **Edge** (`warm_start_from`) | The declared parent a node initialises from, under `task_load_mode = initialize_next_stage`, with lineage recorded. |
| **Trunk** | The shared certified chain every behavior builds on: stance → (recovery) → locomotion. |
| **Leaf** | A node with no children: today `behavior` (hunt); new `follow_direction_speed`. |
| **Deliverable** | A node whose certified checkpoint is a published policy (`deliverable = true`). |
| **Recipe** | A deliverable plus its ancestor chain, *derived* from the edges; the optional per-entry `recipe` label groups nodes under a behavior name. "walk" = stance → locomotion. "hunt" = stance → locomotion → behavior. "follow" = stance → locomotion → follow_direction → follow_direction_speed. "stand" = stance → recovery. A label resolves to its deepest deliverable in manifest order. |
| **Certified** | A property of one checkpoint: the node's own gate passed *and* every ancestor's gate passed, all evidence hash-bound to the checkpoints it describes. **Replication** (how many distinct-seed runs certified the same task) is a separate count, §4.5. |

The internal id of the hunting node stays `behavior`: the manifest pins
legacy number 3 to that id and refuses a rewrite
(`environments/shared/stage_manifest.py:43, 226-231`), and every run names
its directory `04_behavior`. (For a numbered stage the task fingerprint
records the legacy integer, `environments/shared/task_fingerprint.py:173`, so
the id itself is not what the fingerprint binds.) Its label in the manifest,
catalog and website is "hunt". Species-specific task
names (bite, strike, food reach, snap) come from each stage TOML's `[stage]
name`.

---

## 3. Where the tree stands (verified 2026-09-05)

### 3.1 The manifest is already a recipe system in embryo

`configs/trex/stages.toml` (schema `mesozoic.stage-manifest/v1`) orders four
stages by semantic id — stance/1, recovery/–, locomotion/2, behavior/3 — and
the three other species get a synthesized manifest from their
`stage{1,2,3}_*.toml` filenames (`stage_manifest.py:153-172`). Each entry may
carry exactly `id`, `config`, `legacy_number` (`:202-204`, pinned by
`test_stage_manifest.py:123`). "Advancing" is *defined* as "has a legacy
number" (`:136-150`, pinned by `:326`), and every consumer derives "terminal"
as "the last advancing entry" (`result_schema.py:557`, `bundles.py:314`,
`environments/shared/reporting/summaries.py:166-171`). The split plan sketched
a per-stage `terminal:` field that never shipped
(`docs/STAGE1_SPLIT_PLAN.md:856-866`).

Recovery is the maintainer's "mini stage" already built: its `[env]` mirrors
stance's plus exactly the perturbation block (pinned by
`test_stage_manifest.py:146`), it warm-starts from stance under
`initialize_next_stage` with lineage recorded
(`environments/shared/config.py:334-368`), its gate is judged once,
post-stage, against a frozen record (`environments/shared/reporting/gates.py`
dispatch on `recovery_quality/v1`), and it is non-advancing in the ladder.
The notebook already runs a de-facto fork from stance (stance → recovery and
stance → locomotion both load `path_1`); it works only because recovery is
excluded from the linear machinery.

Everything below the orchestration layer is parent-agnostic today: any
checkpoint can be warm-started into any stage with
`train --stage <ref> --load <ckpt> --load-mode initialize_next_stage`
(`environments/shared/cli.py:219-243`); `select_handoff_checkpoint` reads one
stage directory (`environments/shared/curriculum/checkpoints.py:95`); the
task fingerprint records whichever parent was used
(`task_fingerprint.py:76`, `config.py:334`); gate kinds are declared per
TOML. A DAG can be driven by hand today. The code that must change is the
orchestration and publication layers, not training or gating.

### 3.2 What the environments can express

Every species' observation is `[joint_pos, joint_vel, root_quat(4),
root_gyro(3), root_linvel(3), root_accel(3), foot_contact(2|4),
target_direction(3), target_distance(1), command(3)]` — 64 / 70 / 86 / 80 /
56 / 46 dims for trex / velociraptor / brachiosaurus / dibothrosuchus /
compsognathus / compsognathus_robot since the Phase C interface revision
(`configs/plant_manifest.generated.json`; segments at
`environments/shared/plant_contract/policy_layer.py` `observation_segments`).
The trailing `command(3)` segment is the body-relative `(v_x_cmd, v_y_cmd,
yaw_rate_cmd)` frame of §4.6, pre-scaled to `[-1, 1]` and constant zero
under `command_mode = "none"` — the only mode implemented in Phase C, so
the *only* live goal signal is still a world-frame unit vector from the
pelvis to a mocap target placed once at reset
(`environments/trex/envs/trex_env.py`, `_get_obs`). At the time this
section was written (before Phase C) there was no command conditioning of
any kind: no target heading, no desired speed, no moving goal; the slot now
exists, and Phase D fills it.

"Walking" is `dot(qvel[0:2], initial_target_direction)` with the direction
frozen at reset from the world origin to the target spawn
(`trex_env.py:965-968`, `environments/shared/base_env.py:983-997`);
`heading_weight` rewards facing that same fixed vector. The MJX backend
recomputes the reference live each step (`environments/shared/mjx_env.py:1034-1036`),
a documented divergence (`docs/KNOWN_ISSUES.md`). In the trex, velociraptor
and brachiosaurus locomotion stages the target is 8–12 m ahead within ±2 m
lateral (dibothrosuchus: 6–10 m within ±1 m), and the SB3 reset never
randomises yaw, so the animal always starts facing +X with the target within
about ±14° (±9.5° for dibothrosuchus).

Eating/hunting is a one-shot episodic event with a static target: trex bite
= head geom contacts prey; velociraptor strike = a sickle-claw geom contacts
prey; brachiosaurus food reach = head-tip site within a threshold of an
elevated food body; dibothrosuchus snap = snout geom contacts prey. All four
pay a 1000-point bonus, set `info["success"]`, and terminate the episode
(`trex_env.py:936-947` and the species equivalents).

Adding command dims breaks warm-starting at three independent layers: the
plant contract hashes `_get_obs`, the shared observation builders and
`build_mjx_observation` into every species' `policy_interface_sha256`
(`policy_layer.py:352-366`) and refuses a mismatched checkpoint; SB3's
`PPO.load(env=...)` raises on an observation-space mismatch; and
`VecNormalize.set_venv` asserts equal shapes before `obs_rms` is copied.
`build_bipedal_obs` and `build_mjx_observation` are in every species' hash
(and `build_quadruped_obs` in both quadrupeds'), so an edit to those shared
builders — which the command segment requires — is an all-species interface
revision; an edit confined to one species' `_get_obs` bumps that species
alone.

### 3.3 Gates and publication

Gate kinds are a versioned, fail-closed registry
(`environments/shared/curriculum/gate_schema.py`): `stance_quality/v1`
(full-horizon ≥ 0.95, unsupported-duty one-sided 95% UCB ≤ 0.02, 40-episode
panel on seeds 3042–3081), `recovery_quality/v1` (episode-level success — full horizon *and* every
scheduled push recovered, where "recovered" is re-entering the calibrated
safe set and dwelling there — on an exact Clopper-Pearson LCB, plus a paired
per-seed policy-minus-null bound against a frozen null panel; frozen
2026-08-28 at LCB 0.30 / paired 0.20),
`reward_and_length/v1` (episode-mean reward, length, forward velocity, raw
success mean), `none/v1`. Publication re-derives every verdict from
per-episode evidence files bound to checkpoint and normalization hashes
(`environments/shared/result_bundle/evidence.py`).

The trex hunting stage is gated by `reward_and_length/v1` on a 30-episode
panel: reward ≥ 100, episode-mean forward velocity ≥ 2.0 m/s, bite success
≥ 0.5, three consecutive passes (`configs/trex/behavior.toml:88-107`). The
gap review's findings against it stand: bite-terminated episodes structurally
cannot average 2.0 m/s from a standing start (CF2); the raw success mean at
n=30 blocks a genuinely 50% policy 43% of the time (SS2); the absolute
collapse floor of 100 sits far below the stage's measured do-nothing reward
of 557 ± 142 and certifies nothing (CF3).

The catalog and website hard-code a "stage 3" headline
(`environments/shared/species_catalog.py:533, 549`), one success metric per
backend per species, and integer-keyed stage videos
(`configs/species_manifest.toml:79-96`). The website already tolerates a
semantic-id row (schema v3 shipped with recovery), and the Python catalog
renderer already describes the stance and recovery gates.

Provenance records a scalar `training_seed` plus role-labelled evaluation
seeds; nothing records replication (gap review SS1). The stance/recovery
certification panel block 3042–3081 is not bound to any provenance role.

**Phase B (2026-09-13; #533, #534 and #535).** The paragraphs
above describe the 2026-09-05 baseline. Since Phase B the registry also
holds `task_success/v1` (`curriculum/task_success_gate.py`: the exact
one-sided 95% Clopper-Pearson lower bound on the selected checkpoint's
per-episode `task_success` over `evaluation_selected.csv`, hash-bound to
the handoff pair, at a declared panel size; `min_avg_reward` a collapse
rail only), and the trex hunting stage is gated on it — the velocity term
and the raw success mean are retired from `configs/trex/behavior.toml`
(D1; §4.4 as implemented). `gate_verdict.json` records the gate it was
judged under (`gate` / `gate_sha256`, D-A22) and reuse rule 7 checks it.
Provenance records replication per deliverable (§4.5 as implemented), and
the 3042–3081 panel block is bound as the `certification_panel` seed role,
per evidence file (D-B17).

### 3.4 The SB3 notebook

`notebooks/sb3_training.ipynb` is a hand-threaded ladder: cells 18, 21, 23
and 26 run stance, recovery (behind `RUN_RECOVERY_STAGE`), locomotion and
behavior, each passing `path_N` / `vecnorm_N` to the next by hand. A numbered
stage that fails its gate disconnects the runtime and raises. The final
evaluation cell hard-codes stage 3; the replay cell iterates
`completed_stages`. Cell 29 resumes an interrupted stage inside the same run.
There is no way to reuse a certified checkpoint from an earlier run, so every
"Run all" pays the stance stage again.

### 3.5 Lessons from the record that bind this design

The T-Rex curriculum was reshaped four times in August 2026, each time for a
measured reason (`docs/STAGE1_SPLIT_PLAN.md`,
`docs/STAGE1B_IMPLEMENTATION_PLAN.md`, `docs/KNOWN_ISSUES.md`, the
investigations they cite). The recipe design keeps the machinery each lesson
produced:

- **Gate on a capability against a null, never on return.** The zero-action
  statue is the reward optimum of the undisturbed stance task; every
  species' stage-1 reward gate was cleared by its statue. Hence
  `stance_quality/v1` and the frozen-null recovery gate.
- **Semantic ids, never renumbering.** Renumbering silently changes what
  "stage 2" means in every historical artifact. Recipes are layered on top
  of the existing ids and legacy numbers.
- **Declare the load mode on every launch path.** Two stance runs died at
  the recovery boundary on a fingerprint bug (a cwd-relative model path,
  PR #508); the gap review found (OP1) that cross-stage sweep chaining would
  crash at worker startup on the missing load mode; the 20260821 recovery
  pilot trained its first 500k
  steps on a forward-velocity ramp its own fingerprint denied. Entry shaping
  must key on "has a parent", not on position.
- **Derive each node's schedule from its own budget.** Recovery mirrored
  stance's 7M entropy horizon against a 5M budget and peaked at ~1.6M of 5M;
  the 5M checkpoint was indistinguishable from the 3M one on every panel, and
  the budget was cut to 3M. On the velociraptor, 8M-step stage-2 runs did no
  better than 2M. Warm-started leaves should be short (3–4M).
- **Do not carry forgetting-mitigation folklore.** The stance→recovery
  transfer needed no ramp; the velociraptor stage-2 configs tuned to "match
  stage 1 to prevent forgetting" (high alive and posture weights) collapsed
  catastrophically, while low alive/posture weights were what let the agent
  leave the stand-still optimum (`docs/investigations/TRAINING_REVIEW.md`).
  Forgetting is measured per recipe against the warm-start parent, not
  assumed.
- **Relative collapse floors with a physics-revision pin and an arming
  delay.** Two absolute floors never armed (they sat above the runs' peaks);
  the statue-relative floor then armed on an initialisation spike and killed
  run 20260803_012355 at 14.5% of budget, which is why stance and recovery
  carry `collapse_peak_warmup_timesteps` beside the reference/fraction pair
  (`configs/trex/stance.toml:449-462`).
- **One seed is not a result.** Stance replicates 2 of 3.
- **Fail closed wherever absence can read as a pass**, and pin every gate with
  a test that answers "what code would have to be deleted for it to stop
  being consulted".

---

## 4. Adopted design

### 4.1 Manifest v2

Schema `mesozoic.stage-manifest/v2`. Per `[[stages]]` entry the allowed keys
become `id`, `config`, `legacy_number`, and three new optional keys:

- `warm_start_from = "<id of an EARLIER entry>"` — this node initialises from
  that node's handoff checkpoint (`select_handoff_checkpoint`:
  `robust_best_model`, then `best_model`) under `initialize_next_stage`.
  Absent means root, trained from scratch. "Earlier entry" is enforced by the
  loader, so list order is a valid topological order by construction and
  every existing in-order walker stays correct without a toposort.
- `deliverable = true` — this node's certified checkpoint is a published
  policy. Recipes are derived: a deliverable plus its `warm_start_from`
  chain.
- `recipe = "<label>"` — the behavior name a node belongs to (`stand`,
  `walk`, `hunt`, `follow`), used by the notebook's `BEHAVIOR` knob and the
  catalog. A label resolves to its deepest deliverable in manifest order.
  Optional; a node without one is addressed by its id.

Ids become an open vocabulary matching `^[a-z][a-z0-9_]*$`; the four existing
ids stay reserved and `LEGACY_STAGE_IDS` with its no-rewrite / no-reorder
rules (`stage_manifest.py:43, 226-237`) is unchanged. Reading a v1 manifest,
or a synthesized one, derives `warm_start_from` = the previous advancing entry
and `deliverable` = the last advancing entry only — bit-identical to today's
behaviour. Phase A therefore commits a small v2 `stages.toml` for every
species (§4.8), since stand and walk become deliverables only when declared.

The T-Rex manifest under v2:

```toml
schema = "mesozoic.stage-manifest/v2"

[[stages]]                       # quiet stance
id = "stance"
config = "stance.toml"
legacy_number = 1
recipe = "stand"
deliverable = true               # stance_quality/v1; certified 2/3 seeds

[[stages]]                       # balance under pushes
id = "recovery"
config = "recovery.toml"
warm_start_from = "stance"
recipe = "stand"
deliverable = true               # recovery_quality/v1, judged against a per-run frozen resolution

[[stages]]
id = "locomotion"
config = "locomotion.toml"
legacy_number = 2
warm_start_from = "stance"       # the 2026-08-23 lineage rule, now declared;
recipe = "walk"                  # flips to "recovery" once a recovery run passes
deliverable = true

[[stages]]                       # id stays "behavior"; the label is the recipe
id = "behavior"
config = "behavior.toml"
legacy_number = 3
warm_start_from = "locomotion"
recipe = "hunt"
deliverable = true

[[stages]]                       # heading commands at cruise — lands with Phase D
id = "follow_direction"
config = "follow_direction.toml"
warm_start_from = "locomotion"
recipe = "follow"
deliverable = true               # a usable turn-to-heading policy in its own right

[[stages]]                       # + commanded speed and mid-episode switches
id = "follow_direction_speed"
config = "follow_direction_speed.toml"
warm_start_from = "follow_direction"
recipe = "follow"
deliverable = true               # "follow" resolves here (deepest deliverable)
```

Stage directories keep their `{position:02d}_{id}` names, file labels keep
`stage{N}` for legacy ints and the bare id otherwise, and the task
fingerprint's `stage` field is unchanged, so every existing artifact key,
CSV row, `summary.json` key, the certified stance evidence and any run's
frozen recovery resolution remain valid.

### 4.2 Orchestration: parents by edge, ancestors reused

Three derivations are retargeted from position to the edge:

1. `_stage_entry_shaping_callbacks` triggers on *has a parent* (an edge and
   an `initialize_next_stage` load) instead of `stage_position > 1`
   (`train_base.py:857`); the notebook's inference at cell 14 lines 239 and
   245 follows.
2. `train_curriculum` walks the manifest in order and resolves each node's
   parent from its edge — the parent's stage directory, its handoff
   checkpoint, its VecNormalize sidecar — instead of carrying one
   `load_path` (`train_base.py:1656-1660`). A node whose parent has no
   certified checkpoint in this run is skipped with a warning naming the
   missing ancestor, never trained from scratch silently. The JAX runner's
   chain (`jax_curriculum.py:503, 548-570`) gets the same treatment when it is
   next touched; SB3 remains the evidence backend (§4.9).
3. `train()` refuses an `initialize_next_stage` load whose recorded
   `parent_stage` is not the manifest's declared parent for that node, with
   the existing warn-only valve for unfingerprinted parents kept.

**Reuse of certified ancestors across runs.** A node may be satisfied by an
existing certified checkpoint instead of trained. The rule, applied by the
notebook loop (§4.7) and by a new `--trunk-from <run_dir>` on
`train_curriculum`: the candidate's stage directory must carry a recorded
gate verdict that passed — a new per-node `gate_verdict.json` written by
`generate_stage_artifacts` beside the handoff checkpoint and hash-bound to
it (today the verdict lives only in run-level `collected_results.csv` /
`summary.json`) — its plant identity must validate against the
current plant (the existing `validate_model_plant`), and its recorded
`task_sha256` must equal the fingerprint derived from the current stage
config (the same check `resume_same_stage` applies). On reuse the child run
copies the ancestor's `stage_config.json`, gate record and checkpoint hashes
into `ancestors/<stage_id>/` (small files, never the checkpoint), and the
lineage gains `parent_run_id`, so the bundle audit can verify a cross-run
parent it cannot verify today (`environments/shared/result_bundle/audit.py`
checks parents only inside one bundle).

Two rules adopted while implementing Phase A sharpen this. *The chain is
checked by digest, root-first*: a non-root candidate is reused only when
its recorded `parent_checkpoint_sha256` equals the digest of the checkpoint
resolved for its declared parent in the child run — two runs that both
certified stance produced two different checkpoints, and a walk descends
from exactly one of them — so a child is reusable only once its parent is,
and once a node is trained in the child run none of its descendants are
looked up. *The target is never reused*: the node a run exists to certify
is always trained; an earlier run's certified target is that run's
deliverable, published from there. The notebook loop (§4.7) codes against
the same two rules.

**Curriculum manager.** `CurriculumManager` stays integer-keyed through
Phase A (`environments/shared/curriculum/manager.py:109-149`): semantic-id
nodes are judged post-stage, exactly as recovery is today, and the leaf's
in-training advancement callback is not needed because a leaf has no child.
Re-keying the manager by manifest entry is deferred until a semantic node
needs in-training advancement.

### 4.3 Publication per deliverable

`RESULT_SCHEMA_VERSION` 3 → 4, with v3 still readable. Changes:

- `provenance.deliverables`: a map from deliverable stage key to
  `{model_path, model_hash, normalization_hash, gate_kind, certified,
  replication}`; `selected_model_path` / `model_hash` point at the *primary*
  deliverable — the deliverable the run targeted, or else the deepest
  certified deliverable present — so v3 readers still see one model. The
  v3 rules "a checkpoint for every advancing stage" and "the primary is the
  manifest's last advancing entry" (`result_schema.py:487-492, 557-566`,
  `bundles.py:314`) are replaced by "a checkpoint for every present
  deliverable": under the old rules a walk-only run could not write a bundle
  and a failed-hunt run would headline the uncertified hunt checkpoint.
- Bundle status: `complete` iff *every* deliverable present in the run is
  certified; `partial` iff at least one is; `failed` iff none. `summary.json`
  is written whenever at least one deliverable is certified. Completeness
  per deliverable = its node and every `warm_start_from` ancestor present
  (in the bundle or as an `ancestors/` record) and passed
  (`bundles.py:129-164`, `result_schema.py:479-497, 554-566`).
- Results stay at `results/<species>/<algo>/summary.json`; the two-level
  path (`result_schema.py:222`) is unchanged, and the summary carries the
  deliverables map.
- Catalog: stage rows gain `deliverable` and `warm_start_from`; result rows
  gain a `deliverables` list with a per-behavior headline metric chosen by
  gate kind (stance: duty UCB and full-horizon fraction; recovery: recovery
  LCB; walk: certified velocity; hunt: success LCB; follow: tracking-success
  LCB). `stage3_success_rate` stays for the historical ladder rows.
  `species_manifest.toml` gains per-deliverable success metrics beside the
  existing per-backend ones, and stage videos are keyed by stage id with
  integer aliases. The TypeScript `AdvancementGate` type and `formatGate`
  learn the non-reward kinds the Python renderer already describes.

### 4.4 Gates per deliverable

| Deliverable | Gate | Change from today |
|---|---|---|
| stand — stance | `stance_quality/v1` | none |
| stand — recovery | `recovery_quality/v1` (frozen) | none |
| walk — locomotion | `reward_and_length/v1`, `min_avg_forward_vel = 1.0`, `min_avg_episode_length = 750` | none now; a cruise-window velocity kind is a later refinement (CF2's suggested metric), not a prerequisite |
| hunt — behavior | `task_success/v1` (new) | **delete `min_avg_forward_vel = 2.0`** (`behavior.toml:97`) — this plan's decision, chosen over the gap review's suggested windowed/peak velocity metric or no-prey probe episodes, which move to the walk gate as a later refinement: speed is the walking deliverable's claim, carried by lineage; gate success on `recovery_gate.binomial_lcb` at a declared n; replace the absolute `collapse_peak_floor = 100.0` with the `collapse_peak_floor_reference` / `fraction` pair plus `statue_constants_physics_revision`, measured with `zero_action_baseline.py trex:3`, together with `collapse_peak_warmup_timesteps` (the pair without the arming delay is what killed run 20260803_012355); set the collapse patience keys explicitly, no tighter than locomotion's 20/10/0.5. **Landed 2026-09-13 as the note below records** |
| follow — follow_direction | `command_tracking/v1` (new, §4.6) | new |

`task_success/v1` keys: `min_success_lcb`, `min_eval_episodes`,
`min_avg_reward` (collapse rail only), `min_avg_episode_length` (optional).
Sizing: a one-sided 95% Clopper-Pearson bound ≥ 0.5 needs 20/30 or 26/40
successes; at n=40 a true 70% policy passes 81% of the time, a true 80%
policy 99%, a true 50% policy 4%. The one committed hunting result (29/30,
LCB 0.85) passes comfortably. Thresholds are frozen attainable-not-
aspirational from the first pilot, as recovery's were. Each new kind is
registered in `GATE_KINDS` and `_REQUIRED_THRESHOLD_KEYS`, dispatched in
`reporting/gates.py`, re-derived from per-episode evidence at publication,
listed in the sweep's offline-evaluable set, rendered by the catalog and the
TSX, and pinned by a fail-closed dispatch test.

**As implemented (2026-09-13; #534, WS-B1
+ WS-B2).** `configs/trex/behavior.toml` declares `gate_kind =
"task_success/v1"` with `min_eval_episodes = 30` (D-B1: the panel size
every existing panel uses; the notebook's selected panel and the
provenance `evaluation_episodes` are coupled to it and pinned) and
`min_success_lcb = 0.5` (D-B2) — **provisional**: frozen before any
Phase-B pilot on the strength of the committed 2026-03 row alone, to be
re-frozen attainable-not-aspirational from the first pilot's
`evaluation_selected.csv` (§10). Sizing at the bar: 20/30 → 0.5006 clears
and 19/30 → 0.4669 does not; 26/40 → 0.5081 and 25/40 → 0.4828; the
committed 29/30 → 0.8514 (a recomputation from a rounded mean, not a
re-judgement — that row holds no per-episode evidence and was judged at
`min_success_rate` 0.25 with no velocity term). The collapse floor is the
measured pair: the hunting statue is **602.13 ± 175.35 over 40 episodes**
(seed 3042, 40/40 full horizon, `zero_action_baseline.py trex:3 --episodes
40 --seed 3042`, physics revision 7), so `collapse_peak_floor_reference =
602.0`, `collapse_peak_floor_fraction = 0.45` (floor 270.9) and
`min_avg_reward = 361` = round(0.6 × 602.13) as the rail (D-B4), pinned by
`statue_constants_physics_revision = 7`; `collapse_peak_warmup_timesteps =
1_000_000` is a judgment (D-B5, §10), and the patience keys are explicit
at 20 / 10 / 0.5. `required_consecutive = 3` stays as in-training
hysteresis only (D-B3). The certifying verdict is one post-stage panel:
`generate_stage_artifacts` judges `evaluation_selected.csv` hash-bound to
the handoff pair (rolling a panel on the publication seed when none is
bound), publication re-derives the pass from the same rows bound to the
certified checkpoint, and the sweep judges a row from the trial's recorded
`success_count` / `n_success_episodes` (D-B12). The CLI curriculum's hunt
verdict is the in-training manager's bound over the last EvalCallback
panel (recorded as `success_count` / `n_success_samples` in the verdict's
`stage_result`); it writes no evidence CSV and is not backfillable, so a
CLI-certified hunt is a training-time signal, not a certification. The
JAX path refuses the kind and, per D-B13, refuses a final stage declaring
it before any training — on the declared kind alone. Every other species'
hunt stays on `reward_and_length/v1` (D-B14).

### 4.5 Seed replication as provenance

A run trains one seed, so replication is a property of a *set* of runs,
not of a checkpoint. Provenance keeps its scalar `training_seed` and gains,
per deliverable, a `replication` record: the count of published runs at the
same `task_sha256` and plant identity whose gate passed, with their run ids
and seeds, aggregated by the catalog from the published bundles (a run may
also list known `replicates` explicitly). The 3042–3081 panel block is
declared as a `certification_panel` seed role so the auditor binds it the
way it binds the publication seed. Stage configs gain
`certification_seeds = N` (default 1 = today's behaviour); a deliverable
whose replication is below its config's `N` is labelled **provisional** in
the summary and the catalog, and the catalog's per-species headline names
the count. Enforcement bites only at publication; training and checkpoint
advancement are unchanged, since advancing a checkpoint that itself passed
is valid at n=1. Initial setting: 2 for trex stance, 1 elsewhere. Four
deliverables per species at one seed each is four draws of the seed lottery,
which is why this lands before any deliverable is called more than
provisional (Phase B).

**As implemented (2026-09-13; #535,
WS-B4).** The count is writer-recorded, not catalog-aggregated (D-B10):
`environments/shared/replication.py` discovers a deliverable's replicates
among the run's `LOG_BASE/<species>/<algo>/` siblings when the notebook's
publication cell runs, and `save_result_bundle(replicates=...)` records
them in each deliverable's `replication` record — this run first, then its
replicates, with distinct run ids and seeds. A replicate is a sibling of
the SAME recipe (D-B16, tighter than "same task and plant, gate passed"
above): equal `task_sha256`, plant identity, `gate_sha256` (the verdict's
own record of the gate it was judged under) and `hyperparameters_sha256`,
a passed reusable verdict, and a different training seed — two seeds of
different gates or algorithm blocks are not replication of one
deliverable. A sibling whose run block predates D-A21 has its
hyperparameters digest DERIVED from its recorded blocks
(`config.recorded_hyperparameters_sha256`) and is never skipped for that
alone; a sibling whose verdict predates D-A22 (no `gate_sha256`) is skipped
until re-backfilled, consistent with rule 7 and D-B6. Discovery never
raises on a malformed neighbour (every skip is logged with its reason);
the record is what fails closed (schema and audit). `certification_seeds`
is a `[curriculum]` publication key (D-B9: positive int, default 1, in no
digest; trex stance declares 2 on the seed 42 PASS / 43 FAIL / 44 PASS
record), written into the record beside `provisional = count <
certification_seeds` (D-B11); the catalog validates the record and
re-derives only `provisional` from the CURRENT declaration, never
aggregating across bundles, and renders `N run(s) of M seed(s)` plus
`provisional` identically to the website. A replicate that finishes later
is counted by re-running the counting run's publication cell with the
sibling present: a `partial` bundle is rebuilt, and a verified `complete`
bundle regenerates its derived artifacts when the replication record is
the only change (the writer masks `replication` and `provisional` — not
`certification_seeds` — when comparing the existing and prospective summaries) and is refused as immutable on any other
difference. The `certification_panel` role is a default provenance role
(`PUBLICATION_SEED_START`, 3042, for every SB3 bundle) bound per evidence
(D-B17): stance panel row *i* must carry `panel_seed == role + i`, a
recovery `gate_resolution.json` must register `panel_seed_start == role`,
a declared role must equal the registered block, and a recorded stance
PASS in a bundle without the role is refused at publication. The JAX saver
passes no replicates, so its records count the run alone.

### 4.6 Follow-direction

**Command frame (decision D2).** Three body-relative dims appended at the
end of the observation: `v_x_cmd` (forward speed), `v_y_cmd` (lateral speed),
`yaw_rate_cmd`, each pre-scaled to [-1, 1] by the stage's declared ranges.
Body-relative matches the controller analogy (stick = velocity, triggers =
turn), the sim-to-real plan's objection to anchoring on absolute yaw
(`docs/hardware/SIM_TO_REAL_PLAN.md:119-123, 265`), and the convention in
legged-robot velocity-command work. Walking is the fixed-command special case
of following *as a task*: a follower holding `v_x_cmd` at cruise with the
others zero is asked to walk. It is not the special case *at warm-start*: the
trunk walker is trained with the command channel all-zero, so its zero is a
placeholder it cannot read, not a "stop" command, and the freshly
warm-started follower walks at the walker's natural speed for every command
until the leaf learns the dependence, stop included. "Cruise" is defined as
the parent walker's mean forward velocity measured on its certification
panel (locomotion declares no target speed, only the 1.0 m/s gate floor and
the 2.5 m/s reward cap), recorded in the follow node's config as the fixed
point of `command_speed_range`.

**The one interface revision (Phase C).** Appending the segment at the end
keeps every existing slice offset; dims become 64 / 70 / 86 / 80. Edits:
each species' `_get_obs`, `obs_functions.build_bipedal_obs` /
`build_quadruped_obs` (a `command` argument defaulting to zeros),
`mjx_env.build_mjx_observation` (zeros), `policy_layer.observation_segments`
(a `command` segment of width 3), `configs/plant_versions.toml`
`policy_interface_revision` for all four species (trex 12 → 13,
velociraptor 9 → 10, brachiosaurus 7 → 8, dibothrosuchus 6 → 7), and the
regenerated plant manifest. **The queued height-channel removal
(`base_env.py:1410-1411`) is deliberately *not* batched with this bump**:
removing that reset draw shifts every subsequent seeded draw, which would
turn the stance re-panel below from a near-reproduction into a fresh roll of
a 2-of-3 seed-sensitive certificate. It stays queued for a revision that
already plans a trunk re-certification.

**Widening instead of retraining.** A tool `widen_checkpoint` maps an SB3
checkpoint plus VecNormalize sidecar from revision r to r+1: it pads the
first `Linear` of `policy_net` and `value_net` with zero columns (net_arch
`[512, 256]`, separate first layers, so the padded network computes the same
function on zero-padded observations), pads `obs_rms` mean/var, updates the
saved observation space, pads the Adam moments of those two weight tensors
in the saved optimizer state (or strips the optimizer state so it
re-initialises; the archive carries `policy.optimizer`, which the repo's
warm-start path restores through `alg_cls.load`), and re-stamps the plant
identity **with the parent checkpoint hash recorded as lineage**, never
silently. Pinned by two tests: the padded columns are exactly zero and
actions on zero-padded observations are allclose to the parent's over a
seeded rollout, and the widened checkpoint completes at least one PPO update
under `initialize_next_stage`. The widened stance checkpoint is then
re-paneled (40 episodes, seeds 3042–3081): a real roll whose pass/fail is
recorded; if it fails, the widened checkpoint is not certified and the trunk
is retrained under the new interface with seed replicates. There is no
committed recovery resolution to re-freeze — the resolution is per-run
pre-registration written into that run's stage directory (§4.7) — but
because `task_sha256` carries `policy_interface_sha256`
(`task_fingerprint.py:164`), the next recovery run freezes fresh null panels
under the new task hash; Phase C runs one such freeze from the widened
checkpoint to record how the statue and brace nulls re-roll.

**As implemented (Phase C, 2026-09-13/14; decisions D-C1–D-C17, §6.1).**
The scope is all six plants, not the four above: compsognathus (53 → 56,
policy interface r1 → 2) and compsognathus_robot (43 → 46, r1 → 2) took the
segment beside trex 61 → 64 (r12 → 13), velociraptor 67 → 70 (r9 → 10),
brachiosaurus 83 → 86 (r7 → 8) and dibothrosuchus 77 → 80 (r6 → 7); physics
and visual revisions, the `observation_schema` strings and the Box bounds are
unchanged, and the height-channel removal stayed queued (D3). The
plant-contract probes inject the non-zero `COMMAND_PROBE_VECTOR =
(0.25, -0.5, 0.75)` on both backends so the SB3/MJX parity of invariant 9 is
asserted on the new slot rather than on zeros (the SB3-only compsognathus
pair reports parity `None`). The six `command_*` kwargs landed on
`BaseDinoEnv` / `MJXEnvConfig` with inert defaults, and while the effective
`command_mode` is `"none"` they are carved out of the task-fingerprint env
section for every species (the compsognathus perturbation carve-out's
pattern, amendment A1), so a stage's `task_sha256` moved only through the
plant's `policy_interface_sha256` and the payload carries a `command` section
only when a `command_manifest` is passed (Phase D). `widen_checkpoint`
(`environments/shared/scripts/widen_checkpoint.py`) implements the tool above
with these differences from the wording: the Adam moments are padded, never
stripped (D-C10); SAC is handled too (the critics' observation-action inputs
get the columns INSERTED after the observation block); `num_timesteps` is
kept (D-C12); the identity gate is the parent's recorded identity one
interface-only revision behind the current plant (never hand-edited; a
parent without one needs `--allow-legacy-plant`); the run block records the
eight `config.WIDEN_LINEAGE_KEYS` (`widened_from_path`,
`widened_from_checkpoint_sha256`, `widened_from_normalization_sha256`,
`widened_from_task_sha256`, `widened_from_policy_interface_sha256`,
`widened_from_policy_interface_revision`, `widened_from_run_id`,
`widened_by`) and never the `LOAD_LINEAGE_KEYS` — a widened node is a root
(D-C8) — with the same parent hashes in the archive's
`mesozoic_widen_lineage` attribute; and both artifacts are re-stamped with
the current plant identity AND the stage's CURRENT task fingerprint
(`derive_stage_task_fingerprint` under `stable-baselines3` with the stage's
current `[env]` and the current identity; the parent's `mesozoic_task_lineage`
is kept), which is what lets `validate_declared_parent` and the recovery
harness's `resume_same_stage` check accept the widened pair. The output is
the parent's own handoff name plus byte-identical `<stage_label>_final.*`
copies (D-C9) so the notebook's JUDGE branch fires; no verdict, provenance,
resolution, evaluation or periodic checkpoint is written. The pins are the
two above plus the probe: exact-zero padded columns, actions equal to the
parent's over a seeded 200-step rollout (seed 3042, `allclose(atol=1e-6)`,
the measured delta recorded in `widen_report.json`) both with the command
slice zero and with the probe vector in it, and one PPO update under
`initialize_next_stage` (`test_widen_checkpoint.py`); the tool runs the
first two as self-verification before it returns. The reseed rule is
implemented as `load_vecnorm_stats(reseed_command_slice=True)` on both the
train and eval destinations, with `train_base._load_vecnorm_into_envs`
deriving the flag from the stage's `command_mode` at both call sites and
never on a `resume_same_stage` load (whose parent is the same task and
already trained under the live slice); the widen tool applies the same
values to the appended slice at creation (`pad_running_stats`), and in Phase
C every stage is `"none"`, so the flag is always false. The SB3 notebook
gained a `WIDEN_FROM` knob (a run id or absolute run directory) and one widen
cell after the RESOLVE cell that widens the parent's certified ROOT handoff
into this run's root stage directory in a NEW run id — never by pointing
`RUN_ID` at the old run (D-C13) — and refuses `SEED != ` the parent's
recorded `run.seed` (D-C14); the chain loop then refuses the verdict-less
directory and judges it. The re-panel of the two certified stance parents,
the recovery-freeze re-roll and the C½ walker were the maintainer sessions
owed at that point (WS-C4;
`docs/investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md` is their
template, and KNOWN_ISSUES lists what the bump strands). As of 2026-09-19
the C½ walker exists (`20260914_123816`, a fresh r13 stance, §5), the
seed-42 widen is superseded by it, the recovery freeze is covered by the
recovery node the seed-44 session trains, and only that seed-44 widen +
re-panel of `20260815_205206` was still owed (`docs/NEXT_STEPS.md`
session 1; it ran 2026-09-20 as `20260920_010912` and passed). The two
certified parents themselves are r11 archives (trex
r11 → r12 landed 2026-08-16 after they trained, a fingerprint-only bump
with `observation_dim` 61 on both sides), two revisions behind r13: the tool's
gate admits a bounded revision gap (D-C17 — `max_revision_gap`, default 1,
with every other identity field still checked so only fingerprint-only
bumps can be crossed, and the gap recorded in the report), the notebook
threads its `WIDEN_MAX_REVISION_GAP` knob into the call, and the seed-44
session sets it to 2.

**Amended by D-D14 (2026-09-24, consolidation PR-14a, decision D-D15).** Once
both widen sessions were decided, the notebook's widen cell and its
`WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs were removed and widening became
a command-line step: `widen_checkpoint --to-stage-dir
<LOG_BASE>/<species>/<algo>/<new run id>/<stage_dirname(species, root)>` (with
`--max-revision-gap N`, D-C17) writes the widened root into a new run, which
the notebook re-enters with `RUN_ID` set to that id and judges in the chain
loop's JUDGE branch. D-C14 holds through the storage cell, which refuses a
`SEED` other than the widened root's recorded `run.seed` before
`initialize_result_bundle` writes anything; D-C13 through the resolve cell,
which refuses a resolved trunk until the widened root holds a verdict
(`environments.shared.result_bundle.reentry`).

**Amended 2026-10-03 (the cleanup plan's §2 row 6, ROW-4/6; not a D-D row).**
The resolve cell no longer refuses a trunk over an unjudged widened root: the
chain loop judges the widened root before it consults the trunk
(decision 6 (b), which the maintainer took on 2026-10-02); a widened pair cut
short is refused as an interrupted node, and a root widened into a run that
already holds it as an `ancestors/` record is refused when it is an ancestor
of `BEHAVIOR`'s node, naming a new run id.
`refuse_trunk_over_unjudged_widened_root` stays exported, and the notebook no
longer calls it.
ROW-4/6 landed as #591 on 2026-10-03.

**Normalization of the command slice.** `[probe]` A constant-zero dim
accumulates running variance ≈ 1e-11 over 8M samples; a live command of 1.0
would normalise to ≈ 1e4 and clip at `clip_obs = 10`, and un-reseeded
statistics need on the order of the trunk's own sample count to adapt. So on
every load into a node whose `command_mode != "none"`,
`load_vecnorm_stats` (`checkpoints.py:418`) reseeds the command slice to
mean 0 / variance 1 at the carried count; commands are emitted pre-scaled, so
normalised values stay O(1) from step 0. Pinned by a test.

**Exact transfer.** `[probe]` With the command columns of both first layers
zeroed, walker and follower produce a 0.0 action delta on every observation,
live commands included; random columns also give a 0.0 delta while the
command dims are zero (so an allclose test on zero-padded observations
cannot tell the two apart) and a measurable perturbation once commands are
live. Hence the two pins above: the columns must be *exactly* zero, and the
follow leaf's warm-start asserts a zero action delta on non-zero commands
before its first update. *Consolidation PR-10 (2026-10-05): `train_base`'s
warm start does both for every live child whose parent never saw a
command (`policy_loading.neutralize_command_columns`, proven by
`assert_command_blind` on the seeded probe against the parent's own
outputs on zero commands) and records the zeroed parameters in the task
lineage.*

**Env changes (Phase D).** `[env]` keys `command_mode` (`none` |
`heading` | `heading_and_speed`), `command_speed_range`,
`command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`
and `_jitter`. Constructor kwargs join the task fingerprint automatically
(`task_fingerprint.py:109`); the switch schedule gets its own fingerprint
block like `perturbation`. The per-episode command is drawn in `reset()`
*after* every existing draw and *only when* `command_mode != "none"`, so the
trunk's seeded reset sequence is unchanged; mid-episode switches are applied
in `step()` from a seeded schedule using the push-schedule template
(`base_env.py:241-255`), which keeps policy and null panels paired.
`info` gains `command_v_x`, `command_v_y`, `command_yaw_rate`,
`tracking_error_v`, `tracking_error_yaw`. Reward: a Gaussian tracking term
on body-frame velocity error and one on yaw-rate error (the idiom of
`reward_target_centered_height`), with `forward_vel_weight = 0` so the
hard-wired ramp is skipped and the ramp callback's attribute comes from the
TOML. MJX: `EnvState.command`, the same tracking terms, and **`command_mode
!= "none"` raises on MJX until implemented** — unknown `[env]` keys only warn
there today (`mjx_env.py:136-153`), which would otherwise train a different
task silently.

*Consolidation PR-9 (2026-10-05, decision D-D2): the keys are `command_mode`
and one `command_config`, a `DirectionCommandConfig` (cruise speed, speed
range, stop and straight probabilities, turn increment, switch interval and
jitter, the scales and the adapter gains), so the follow node's cruise is
its `cruise_speed` rather than a fixed point of `command_speed_range`; the
five numeric kwargs are `command_frame.RETIRED_COMMAND_ENV_KEYS`, refused
by the constructors and, by name, by the task fingerprint. A
live mode needs a config, `"none"` refuses one, and `"heading"` holds
`cruise_speed` (it refuses `speed_range` and stops). `BaseDinoEnv` builds
the controller under a live mode, seeds it in the reset hook after every
existing draw and advances it at the end of `step()`, after the reward;
the task fingerprint records the config in its `env` section and the
controller's manifest as its `command` section, passed exactly when the
mode is live. The tracking reward and its `info` keys stay in the recipe
env until PR-11.*

**Gate `command_tracking/v1`.** Per command *event* (episode start and each
switch): success iff the velocity error is within
`tracking_velocity_tolerance` and the yaw-rate error within
`tracking_yaw_tolerance` inside `tracking_settle_steps` and held for
`tracking_dwell_steps` — the recovery gate's re-enter-and-dwell event
(`environments/shared/curriculum/recovery_gate.py:125-165`) applied to
commands. Required keys: `min_tracking_success_lcb` (Clopper-Pearson over
events), the tolerances, settle and dwell, `min_eval_episodes`, and a
worst-of-eight-heading-bins floor so a front-hemisphere-only policy cannot
pass on the mean. The panel **includes the walk command** (forward at cruise)
and is paired against the **command-blind walker** as a frozen null via
`paired_difference_lcb`, so forgetting of walking is measured, not assumed.
Evidence CSVs gain the command and error columns. Thresholds are frozen from
the first pilot.

**Two short mini-stages.** `follow_direction` trains heading commands at a
fixed cruise speed (3M, warm-started from the certified walker); a second
node `follow_direction_speed` adds commanded speed and mid-episode switches
(3M, warm-started from the first). Each derives its entropy and LR schedule
from its own budget (the recovery lesson).

**Zero-interface pilot (optional, before Phase C).** A
`heading_follow_pilot.toml` that mirrors locomotion's `[env]` and spawns the
existing target on the full circle far away (`prey_distance_range` and
`prey_lateral_range` spanning ±40 m; `_spawn_target_2d` has no positivity
check, `base_env.py:973-974`) turns the walker's fixed initial-direction
reward into direction-following with no observation change. It gives no
speed channel, its heading distribution is square-uniform rather than
angle-uniform, and the target distance leaves the normaliser's learned range,
so it is a measurement, not the deliverable. It runs through the notebook's
manual single-node cell under `none/v1` (verdict recorded, not enforced, as
the recovery cell does today; `none/v1` refuses to pass by design and the
chain loop would stop on it), and what it yields is qualitative —
heading-following videos plus the existing velocity and success columns —
not a tracking-gate baseline, since the command and error evidence columns
only exist after Phase D.

### 4.7 The SB3 notebook

Configuration gains:

```python
BEHAVIOR = "follow"             # a recipe label ("stand" | "walk" | "hunt" | "follow") or a deliverable's stage id
TRUNK_FROM = "auto"             # "auto" (D-A25): the sibling run covering the most of the chain; a run id pins one; "" trains every node here
```

Cells 18, 21, 23 and 26 collapse into one chain loop (decision D5), and
`RUN_RECOVERY_STAGE` goes away — recovery runs when the chosen behavior's
chain includes it. The loop resolves `BEHAVIOR` through the manifest (a
label to its deepest deliverable, an id to itself), walks the deliverable's
ancestor chain in order, and for each node:

1. **Reuse** if a certified checkpoint for the node exists in `RUN_DIR` or
   in `TRUNK_FROM` under the §4.2 rule; record it under `ancestors/` and
   continue. Under `TRUNK_FROM = "auto"` (the default, decision D-A25) the
   resolve cell first selects the trunk run: the run beside this one under
   `LOG_BASE/<species>/<algo>/` whose certified ancestors cover the most of
   the chain root-first, the greatest run directory name on a tie (the
   newest timestamp id), with every refusal printed.
2. **Train** otherwise, warm-started from the parent's handoff checkpoint
   and sidecar (`task_load_mode = "initialize_next_stage"` when it has a
   parent, `resume_same_stage` never inferred from position), generate the
   node's artifacts, judge its gate through the shared
   `reporting.gates.evaluate_stage_gate`, and, for a frozen-null kind,
   freeze the resolution before training and roll the panel after, as the
   recovery cell does today.
3. **Judge a trained-but-unjudged node** — one whose budget was spent by
   cell 29's resume but whose `gate_verdict.json` is absent: generate its
   artifacts, judge the gate, record the verdict, continue. Today the
   verdict is produced only by the per-stage artifact cells the loop
   absorbs, so without this branch a resumed node could never be reused.
   **Amended 2026-10-03 (the cleanup plan's §2 row 6, ROW-4/6; not a D-D
   row):** item 1 does not consult `TRUNK_FROM` for an ancestor (not the
   target, not covered by `RETRAIN_FROM`) whose loop directory in `RUN_DIR`
   holds a checkpoint (any `models/*.zip`) and no `gate_verdict.json`, unless
   the run holds that ancestor as an `ancestors/` record, in which case item 1
   still consults `TRUNK_FROM` for it as before (and a root widened beside
   such a record is refused, naming a new run id). For such an ancestor the
   loop first checks, under a trunk, that it was trained on the parent
   resolved here (§4.2 rule 4), and refuses it otherwise (train it in a fresh
   `RUN_ID`). This item then judges an intact final pair; any other checkpoint
   is refused as an interrupted node, naming `RESUME_STAGE`.
   ROW-4/6 landed as #591 on 2026-10-03.
4. **On a gate failure**, stop the chain, write the bundle (which now
   publishes every certified deliverable above the failure), and disconnect
   as today. A leaf failure no longer suppresses the trunk's deliverables.

The loop uses `_stage_entry_shaping_callbacks` unfiltered, so SAC gets the
same stage-entry warm-up the CLI gives it (this closes the open question from
gap-review DU1). One manual single-node cell stays as a debugging escape
hatch, and cell 29 (resume an interrupted node in the same run) is unchanged.
The evaluation and replay cells become deliverable-aware: evaluate
`BEHAVIOR`'s policy, replay every node in its chain. For the follow
deliverable the video cell scripts a command sequence (forward, turn, stop)
and renders it, which is the controller-stick demonstration. The chain loop's structure
is pinned at the AST level like `test_jax_notebook_pins.py` pins the JAX
notebook.

### 4.8 Other species

Velociraptor, brachiosaurus and dibothrosuchus get stand (stance only), walk
and hunt when Phase A commits a small v2 `stages.toml` for each (three
entries, the implicit edges made explicit, `deliverable = true` on all
three); a synthesized manifest alone would yield only hunt as a deliverable
(§4.1). Their stage TOMLs, ids and legacy numbers do not change. Their follow-direction
leaves wait on plant preflight — the velociraptor's single toe site reads
about 55% of true load (`docs/STAGE1B_IMPLEMENTATION_PLAN.md` §5,
`configs/plant_versions.toml` note 8), the brachiosaurus statue has stood
40/40 since the physics-r4 repair (note 7) but its shins are uninstrumented,
so a kneeling pose reads identically to airborne (note 8), and the
dibothrosuchus has had no stance-quality or perturbation preflight at all —
and their stance gates remain reward-cleared by their statues, so "stand" for them is labelled by gate kind
in the catalog rather than claimed as certified stance quality.

### 4.9 What does not change

The four trex stage ids, config files, `[env]`/`[ppo]` blocks, legacy
numbers, manifest positions and directory names; every existing
`task_sha256` outside the Phase C bump; `stance_quality/v1`,
`recovery_quality/v1`, the resolver and `freeze_recovery_gate.py`;
`load_stage_config` / `load_all_stages` / `build_env`; the load modes and
lineage record; `select_handoff_checkpoint`; the CLI `train` path; the
published v2/v3 bundles (rendered as historical ladder results); SB3 as the
evidence backend (the JAX path reads v2 manifests and fails closed on a
command-mode config until Phase D reaches it); the sweep tooling (trunk-only
and integer-keyed until a recipe sweep is needed).

---

## 5. Phases and PRs

| Phase | PR | Content | Exit criteria | Effort |
|---|---|---|---|---|
| **Doc** | this | This plan; docs index and changelog entries | Reviewed; decisions §6 confirmed or vetoed | — |
| **A** | manifest edges + publication + notebook | §4.1 manifest v2 (loader, validators, v1/synthesized compatibility); §4.2 retargets, `gate_verdict.json`, `--trunk-from`; §4.3 schema v4, per-deliverable status, catalog and TSX; §4.7 chain loop | v1 manifests read bit-identically (pinned); v2 manifests committed for all four species; a run with a failed leaf publishes its certified trunk (pinned); notebook AST pins; full suite green | ~1.5 weeks |
| **B** | **landed 2026-09-13** — #533 (gate-configuration digest, reuse rule 7, backfill `--gate`; WS-B3), #534 (the `task_success/v1` hunting gate and `behavior.toml`; WS-B1 + WS-B2), #535 (seed replication; WS-B4); this docs pass is WS-B5 | §4.4 `task_success/v1`, `behavior.toml` edits, measured statue floor; §4.5 provenance fields, `certification_panel` role, provisional labels; decisions D-B1–D-B17 (§6.1) | Met: fail-closed dispatch test for the new kind (`test_gate_dispatch_fail_closed.py`); `load_all_stages("trex")` accepts `behavior.toml` under `task_success/v1` and the gate-schema tests pass; the catalog and the site render `N run(s) of M seed(s); provisional` identically (`test_species_catalog.py::test_website_replication_formatter_mirrors_python`); trex stance read `1 run of 2 seeds; provisional` on the r13 run `20260914_123816` (one certified seed against the two-seed bar) until the seed-44 parent was widened and re-paneled as `20260920_010912` on 2026-09-20 (§7, `docs/NEXT_STEPS.md` session 1), whose bundle records `2 runs of 2 seeds` (2026-09-21; the seed-42 run's own run-level records still read 1); the pre-Phase-C stance bundles are never republished and none is committed (§10, KNOWN_ISSUES) | 2–3 days |
| **C** | **implemented 2026-09-13/14** in three PRs — PR-C1 (the one interface revision on all six species, `command_frame.py`, the command-slice reseed plumbing, the non-zero probes, the restamped compsognathus calibrations; WS-C0 + WS-C1), PR-C2 (`widen_checkpoint`; WS-C2), PR-C3 (the notebook `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs and widen cell, the AST pins, this docs pass; WS-C3); of the maintainer Colab sessions (WS-C4: widen + re-panel both stance parents, one recovery freeze re-rolled, the C½ walker) only the seed-44 widen + re-panel of `20260815_205206` was still owed as of 2026-09-19, with the recovery node trained in the same run (`docs/NEXT_STEPS.md` session 1; template note `investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md`), and it ran 2026-09-20 as `20260920_010912` and passed (recovery 2026-09-21); the seed-42 widen and the C½ walker are superseded by the fresh r13 run `20260914_123816` (the C½ row) | §4.6 command dims across SB3 and MJX, plant revisions, `widen_checkpoint`, command-slice reseed, MJX fail-closed; decisions D-C1–D-C17 (§6.1) | Met: zero-column and action-equality pins (on zero and probe commands) and one PPO update from the widened checkpoint (`test_widen_checkpoint.py`); the regenerated manifests pass `plant_contract --check` with SB3/MJX parity on the non-zero probe; the seeded reset stream is pinned by the golden fixture. Met 2026-09-20/21: the seed-44 widen + re-panel of `20260815_205206` (an r11 archive, two revisions behind r13, so `WIDEN_MAX_REVISION_GAP = 2`, D-C17) with the recovery node trained in the same session, as `20260920_010912`, outcome recorded in the investigation note (§7, §8); the seed-42 parent no longer needs widening (`docs/NEXT_STEPS.md` session 1; KNOWN_ISSUES lists what the bump strands) | ~1 week |
| **C½** | **landed 2026-09-15** as run `20260914_123816` (a maintainer Colab session on the merged Phase C tree, commit 35dd44c; no PR): the certified r13 walker exists | A from-scratch r13 stance (seed 42, 11M steps, 13h13m, final eval 3460.6 ± 19.2, `stance_quality/v1` PASS) followed by locomotion (8M, 8h46m, 1.07 m/s mean forward velocity, 1000-step episodes, mean reward 1940.8, `reward_and_length/v1` PASS judged 2026-09-15): the walker came from a fresh r13 stance, not from the widened r11 stance this row planned (no widen session had run by then) | Met: both nodes carry a passed `gate_verdict.json` whose `task_sha256` (stance 82528a2e…, locomotion 31383192…) and `gate_sha256` equal the digests derived from the current stage TOMLs, verified at 22c1fc8, so `TRUNK_FROM = "auto"` selects this run and reuses BOTH nodes for a hunt session; a stand or walk session never reuses its target (D-A18), so it considers stance alone, and since 2026-09-20 the newer seed-44 run `20260920_010912` wins that tie. Cosmetic gap: the run-level `summary.json`, `provenance.json` and `artifact_manifest.json` were written after the stance and never refreshed after the locomotion verdict (reuse reads per-node files, so nothing breaks; they cannot be refreshed in place: the bundle is `complete` under a stance target and refuses the `03_locomotion/` files that appeared after publication; a complete bundle is immutable, [RESULT_BUNDLES.md](RESULT_BUNDLES.md), and since consolidation PR-14a the notebook refuses such a session before training, CHANGELOG "Fixed"). The second stance seed (§7) is `20260920_010912` (seed 44, certified 2026-09-20) | 13 h + 9 h of Colab (measured) |
| **D** | **landed 2026-09-15 as a separate pipeline** (#540 T. rex pilots, #541 all species + certified library); consolidation into the reserved hook pending (`docs/CONSOLIDATION_PLAN_2026_09.md`, PR-9..PR-13; the sequence was released 2026-09-20 in the notebook-first order of D-D13) | Designed: §4.6 env sampler, tracking reward, `command_tracking/v1`, the two mini-stage TOMLs, notebook command video; T-Rex pilot. Shipped: a direction controller, tracking reward, heightfield terrain generator and replay recorder (kept), built beside the canonical machinery: two behavior env classes that refuse the reserved `command_mode` and write `self._command` directly instead of filling `_draw_episode_command` (D-C5), a second trainer (`environments/shared/train_behaviors.py`) with its own recipe dialect, 66 behavior TOMLs (11 templates × 6 species) plus 8 trex pilot TOMLs, a certificate outside `GATE_KINDS`, a checkpoint identity keyed on source-file hashes, a certified library and a notebook mode switch; §6.2 (D-D1..D-D10, G1..G4) folds them back into manifest nodes under `locomotion` | Not met in the canonical machinery: the pilots write no `gate_verdict.json`, are not reusable through `find_certified_ancestor`, and any bundle they write is evaluation-only (D-D9; none is on Drive, [CLEANUP_PLAN_2026_09.md](CLEANUP_PLAN_2026_09.md) §5.2). The original criteria (pilot from the C½ walker with frozen thresholds recorded; MJX fail-closed raise test; gate consulted-test; live-command SB3/MJX parity a Phase E criterion) transfer to PR-9..PR-13 with G4's first thresholds | ~1.5–2 weeks + one pilot run (as designed); the consolidation PRs are sized individually in the consolidation plan |
| **E** | follow-on | Other species' follow leaves after preflight; cruise-window walk gate; manager re-keying; sweep re-keying | as needed | — |

Two PRs landed 2026-09-16 outside the phase table: #542 flipped the
notebook's `PUBLISH_CERTIFIED` default to `False` (the consolidation plan's
PR-1: library publication required a training-origin stamp that a widened
root handoff lacks, so a `WIDEN_FROM` session would have disconnected the
Colab runtime right after the stance passed its gate), and #543 landed
automatic trunk selection, D-A25 (`ancestors.select_trunk`, the notebook
default `TRUNK_FROM = "auto"`, CLI `curriculum --trunk-from auto`;
canonical chains no longer consult the certified library, and a widen
session selects no trunk).

Each PR follows the established process: adversarial pre-PR review with
lensed finders, three refuters per finding, a completeness audit; ruff, mypy
and the full suite in chunks; the notebook round-trips through `json.dump`
with indent 1.

Session order after Phase C follows the maintainer's priority (2026-09-13,
decision D-C15): **stance → recovery → walking before hunting** — widen and
re-panel the certified stance (both seeds), re-freeze one recovery
resolution from the widened handoff, then the C½ walker; the hunting set-up
comes after a walker is certified under the new interface. As of
2026-09-19 the walker exists (the C½ row) and the seed-42 widen is
superseded by it; what remained of that order was the seed-44 widen +
re-panel with the recovery node trained in the same session
(`docs/NEXT_STEPS.md` session 1; it ran 2026-09-20/21 as
`20260920_010912` and passed), and hunting stays after it.

---

## 6. Decisions and assumptions

Decisions taken 2026-09-05 (the maintainer approved the program with these
as recommendations; each may be vetoed at review of this document):

| # | Decision | Veto changes |
|---|---|---|
| D1 | The gap review's held decisions are settled here: CF2 by dropping the 2.0 m/s hunting term outright (this plan's choice; the review itself suggested a windowed/peak metric or probe episodes), SS2 (success LCB), CF3 (measured collapse floor with its arming delay), SS1 (replication as provenance, default 1, provisional labelling) | Phase B scope |
| D2 | Body-relative command frame (`v_x`, `v_y`, `yaw_rate`), three dims, pre-scaled | Phase C interface; cannot change after the bump without another revision |
| D3 | The interface bump is done early, once, for all species, **without** the height-channel removal; the certified stance checkpoint is widened and re-paneled, and one recovery freeze is re-rolled from it | Phase C ordering and which checkpoints survive |
| D4 | "Stand" has two certifiable nodes: stance publishes as quiet stance, recovery as balance under pushes once a run passes | Manifest `deliverable` flags; catalog rows |
| D5 | The notebook's four stage cells are replaced by the chain loop, with one manual single-node escape hatch | Notebook shape |

Assumptions (state an objection and the plan changes):

- A1 The hunting node keeps id `behavior` (recipe label "hunt"); the new
  nodes are `follow_direction` and `follow_direction_speed` (recipe
  "follow").
- A2 Locomotion's parent stays `stance` until a recovery checkpoint passes
  the frozen gate; the flip is a manifest edit.
- A3 A speed command dim is reserved even though the first follow node
  trains at fixed cruise speed.
- A4 The follow leaf is T-Rex only until other species pass plant preflight.
- A5 Results stay at `results/<species>/<algo>/` with deliverables inside. *Superseded 2026-10-11 by D-D33:* `results/<species>/<embodiment>/<algo>/`.
- A6 Sweeps stay trunk-only and integer-keyed for now. Superseded by D-D17: the sweeps are retired (cleanup PR-A); adding one back is a new decision (the cleanup plan's §4.10).
- A7 SB3 is the evidence backend; MJX fails closed on command-mode configs. Narrowed by D-D17: SB3 is the evidence backend; nothing trains on MJX, and the frozen core only feeds the probe.
- A8 `certification_seeds` defaults to 1; n=1 deliverables are labelled
  provisional; trex stance declares 2.
- A9 The notebook loop applies stage-entry warm-up to SAC as the CLI does.
- A10 Ancestor reuse copies records, never checkpoints, into the child run.

### 6.1 Decisions taken during Phases A, B and C (the D-A, D-B and D-C series)

Taken while implementing Phase A (2026-09-06 to 2026-09-12, D-A1–D-A24;
D-A25 landed 2026-09-16 in #543), Phase B (2026-09-13, D-B1–D-B17) and
Phase C (2026-09-13/14, D-C1–D-C17);
the pull requests, CHANGELOG and code comments cite them by number. Each
D-B and D-C row states the decision as taken and as implemented; where the
implementation deviated from the design's wording, the row says what the
code does (the D-C rows fold in the amendments adopted at the Phase C
critique, numbered A1–A15 in that record — distinct from the §6 assumptions
A1–A10 above — which override the original wording where they conflict).

| # | Decision |
|---|---|
| D-A1 | `save_result_bundle(target_deliverable=...)` (default the manifest's last deliverable; the notebook passes `BEHAVIOR`'s node). `complete` = target present and certified and every present deliverable certified; `partial` = at least one certified deliverable; `failed` = none. A stance-only run targeting hunt is partial, never complete, so the bundle stays mutable. |
| D-A2 | `validate_result_summary` / `validate_result_bundle` gain `require_publishable` (at least one certified deliverable); `require_complete` keeps meaning `complete`. The audit, the bundle writer and the catalog use `require_publishable`; the notebook completion cell keeps `require_complete`, its failure branch `require_publishable`. |
| D-A3 | A root node under `initialize_next_stage` accepts a parent whose recorded stage is the node itself, refuses any other fingerprinted parent, and warns on an unfingerprinted one. |
| D-A4 | Two v2-only manifest validators: a numbered reserved id must declare its legacy number; a v2 manifest with no deliverable is fatal. |
| D-A5 | `gate_verdict.json` is written by `generate_stage_artifacts` (post-stage, evidence-backed) and by `train_curriculum`'s in-training manager verdict, with `judged_by` recorded; reuse accepts any well-formed passed verdict. `train()` and the JAX saver do not write it in Phase A. Amended by D-D17: the JAX saver (`reporting.save_jax_stage_artifacts`) is retired (cleanup PR-B); `generate_stage_artifacts` and `train_curriculum`'s in-training verdict remain the writers. |
| D-A6 | `scripts/backfill_gate_verdict.py` re-derives a verdict for a pre-Phase-A stage directory from its evidence through `evaluate_stage_gate`; it refuses when evidence is missing. |
| D-A7 | Reused nodes write no `curriculum_results.csv` row; `train_curriculum` stops (does not skip past) a node whose declared ancestor has no certified checkpoint; non-advancing nodes are still skipped by the CLI curriculum in Phase A. |
| D-A8 | Catalog `schema_version` 3 → 4; `species_manifest.toml` `schema_version` 1 → 2. |
| D-A9 | Stance and recovery deliverable headlines render the metric name with a null value in Phase A; exporting per-stage gate metrics into the summary is Phase B (deferred to a later phase by D-B15; the hunt's `selected_model_success_lcb` is the exception). |
| D-A10 | The README's generated SPECIES table gains Recipe and Warm-start-from columns; the generated RESULTS block and the website's published run summaries stay byte-identical (golden regression). **Amended 2026-10-04 (cleanup CU-17; the maintainer's choice of 2026-10-04):** the full SPECIES table, with these Recipe and Warm-start-from columns, and the RESULTS block move from the root README to the generated `docs/SPECIES_CATALOG.md`, which `species_catalog` writes whole and `--check` verifies; their links there are relative to `docs/`. The golden regression pins the RESULTS rendering byte for byte, and the committed RESULTS block on that page byte for byte apart from its four source-summary links, which read `../results/`. The root README's SPECIES block is one generated row per species that keeps every species anchor and links the species' catalog entry, its README and its model page; the website's published run summaries are unchanged. **Amended 2026-10-10 (PR #607's proposal; approved by the maintainer on 2026-10-11; not a D-D row):** the RESULTS block and the website's published run summaries show a stage recorded as `balance` under the stance display name, so the stage-1 rows of the four historical ladder summaries read `1 — Stance` (they read `1 — Balance` in the RESULTS block and `1 — balance` on the website). The golden regression still pins the rendering, and the committed block on that page apart from its four source-summary links, byte for byte against the unchanged 2026-09-06 fixture with those four `1 — Balance` stage cells relabelled `1 — Stance`. Every published number and verdict is unchanged (each of the four still reads `passed retired gate (reward gate)`), as are the run summaries in the website's data file, whose recorded stage name stays `balance`. **Amended 2026-10-11 (decision D-D33):** the four committed historical summaries move to `results/<species>/anatomical/<algo>/` with only their identity rewritten (`species` the binomial, `embodiment = "anatomical"`; they keep schema 2, now documented as amended in place), so every published number, verdict and stage cell is unchanged. The RESULTS block on `docs/SPECIES_CATALOG.md` changes in its four source-summary links and its four headings, which name the model (`Tyrannosaurus Rex (Anatomical)`). The golden fixture is unchanged, and the test compares the rendering with those links and headings normalised from the catalog. The root README's SPECIES block has one anchor per model (`<species>-<embodiment>`); the alias anchors go. The website's published run summaries render byte-identical on their model's page `/docs/models/<species>/<embodiment>`. |
| D-A11 | The notebook's `BEHAVIOR` defaults to `"hunt"`; the recovery gate is enforced by the chain under `BEHAVIOR="stand"`; the manual single-node cell never swallows a `ResultBundleError` silently. **Amended 2026-10-16 (decision D-D35):** the recovery node's label is `recover`, so the recovery gate is enforced by the chain under `BEHAVIOR="recover"`, and `BEHAVIOR="stand"` trains the stance node alone on every model. **Amended 2026-10-18 (decision D-D38):** the stance node's label is `stance`, its own id, so `BEHAVIOR="stance"` trains the stance node alone on every model, and `BEHAVIOR="stand"` is refused naming `"stance"`. |
| D-A12 | Species-free readers (`detect_stage_from_path`, the sweep collector) stay reserved-id only; species-aware readers accept any declared id. Amended by D-D17: the sweep collector is retired (cleanup PR-A); `detect_stage_from_path` is the remaining species-free reader. |
| D-A13 | The website `RawStage` / index-page defect fix ships inside the catalog workstream. |
| D-A14 | CSV deliverable columns, the training summary's deliverables block and `species_registry.describe_stages` are out of Phase A. |
| D-A15 | `duration_seconds` is recorded into `stage_config.json`'s run block on every `train_stage` exit so a resumed-then-judged node reports its real duration. Amended by consolidation PR-14c (D-D11): `train_base.train` records it at the stage's final save, for the CLI `train` subcommand and Vertex sweep trials too, summed over the sessions that reached a final save in that stage directory; a notebook session stopped before that save records nothing, as before, so an interrupted-then-resumed node reports the resumed session's time. Amended by D-D17: the Vertex sweep trials are retired (cleanup PR-A); `train_base.train` records the duration for the CLI `train` subcommand and the notebook's `train_stage`. **Amended 2026-10-03 (D-D11's amendment, cleanup CU-10b):** the CLI `curriculum` subcommand records it too, for each node it trains: from the point the node is trained rather than reused to its final save, recorded right after that save, an interrupted node's included; a curriculum node never resumes, so it records the one session that trained it. `curriculum_results.csv`'s `training_duration_seconds` stays the time `learn()` took. |
| D-A16 | `target_deliverable` / `primary_deliverable` are finalization fields written by `save_result_bundle`, not identity fields. |
| D-A17 | Reuse is chain-aware by digest (§4.2 rule 4): a non-root candidate's recorded `parent_checkpoint_sha256` must equal the digest resolved for its declared parent; a root candidate must not have entered from a parent. Reuse is root-first and a child of a node trained in the same run is never looked up. **Amended 2026-10-03 (the cleanup plan's §2 row 6, ROW-4/6; not a D-D row):** in a session with a trunk, an ancestor that is neither the target nor covered by `RETRAIN_FROM`, whose loop directory in `RUN_DIR` holds a checkpoint (any `models/*.zip`) and no `gate_verdict.json`, and that the run does not hold as an `ancestors/` record, is never looked up in `TRUNK_FROM`: the chain loop judges an unjudged `RUN_DIR` node before it consults the trunk (decision 6 (b), which the maintainer took on 2026-10-02), and the `ancestors/` record exemption and the rule-4 check below are ROW-4/6's own forms. Rule 4 is applied to such an ancestor first (a non-root must have been trained on the checkpoint resolved here for its parent, and a root must not have entered from a parent); if it fails, the loop refuses, naming a fresh `RUN_ID`. Otherwise the node is judged here (an intact final pair) or refused as an interrupted node toward the RESUME cell. A node the run already holds as an `ancestors/` record (a run that took the trunk's copy before ROW-4/6) is still looked up in `TRUNK_FROM` as before (the trunk's copy is taken while the trunk certifies it), and a root widened into the run beside such a record is refused, with or without a trunk, naming a new run id. The target (D-A18), and every node in a session without a trunk, are judged as before, with no rule-4 check (KNOWN_ISSUES.md, LOW: the chain loop's JUDGE branch applies no chain check). ROW-4/6 landed as #591 on 2026-10-03. |
| D-A18 | The target node is never reused across runs; an earlier run's certified target is that run's deliverable, published from there. |
| D-A19 | A retrain-from knob (`curriculum --retrain-from <stage_id>`, the notebook's `RETRAIN_FROM`) reuses certified ancestors strictly above the named node and trains it and every descendant; it generalises D-A18. Landed with the notebook loop (#530). |
| D-A20 | Training refuses to write into a stage directory that already holds a verdict or stage config unless the load is an explicit same-stage resume, so a fresh run directory per variant is enforced. Landed with the notebook loop (#530). |
| D-A21 | Every trained node records a `hyperparameters_sha256` (its algorithm block and stage-entry shaping keys) and an optional label, propagated to `provenance.deliverables` and the W&B run; at reuse time the loop compares the current digest against the ancestor's copied config and warns, naming the differing keys, when an edit is being ignored. The task fingerprint and the reuse rule are unchanged. Landed with the notebook loop (#530). |
| D-A22 | `gate_verdict.json` records the gate configuration it was judged under (kind, schema version, every threshold, and a `gate_sha256` over them) and reuse gains rule 7: a candidate judged under a different gate configuration is refused, naming the differing thresholds. Landed in #533 (the payload is D-B7's threshold projection, not literally every key). |
| D-A23 | Trunks compose: reuse rule 1 prefers the candidate's stage directory and otherwise follows its `ancestors/<stage_id>/ancestor.json` to the `source_run_dir` it names (as recorded, else the sibling of the same name beside the candidate; else refused naming both paths), applying every rule at the source and requiring the source's handoff pair to hash to the record's `handoff` digests. At most 8 records are followed, a cycle is refused, and refusals on the followed path are prefixed with the hop taken. The result describes the source (`via` lists the followed runs; logged, never persisted), so a child's `parent_run_id` names the run that certified the node. Following is opt-in (`follow_records=True`): `train_curriculum --trunk-from` passes it (the trunk is another run by construction); the notebook loop does not, because it tries its own `RUN_DIR` first and reads `same_run` off the candidate — a followed record there would present the trunk's node as this run's own on a re-run — so the loop passes `follow_records=candidate is not RUN_DIR`: `TRUNK_FROM` composes, this run's own directory never follows (pinned). `ancestor.json`'s schema is unchanged; `source_run_dir` / `source_stage_dir` are written as absolute paths so a record made from a relative `--trunk-from` follows from any working directory. |
| D-A24 | `curriculum --target BEHAVIOR` (a recipe label, a deliverable's stage id or a legacy number; `train_curriculum(target=)`) walks the target's chain (`chain_for`) and stops at the target, so a walk-only certified run exists on the command line as through the notebook's `BEHAVIOR`. A label or id resolves as the notebook's `BEHAVIOR` does; a legacy number as `--stage` does. Every node of an explicit target's chain must be advancing and the chain a prefix of the advancing ladder — a chain through a non-advancing node (`stand` on T-Rex / Compsognathus) or one that skips a ladder node (an edge rewired past it, which the loader accepts) is refused before any directory is written, naming the notebook, since the integer-keyed manager would judge the node after the gap against the skipped stage's thresholds — and `--retrain-from` must name a chain node. The default (the last advancing stage) walks the whole advancing ladder unchanged. The manager stays integer-keyed over the full ladder and is never advanced past the target. **Amended 2026-10-16 (decision D-D35):** the chain through the non-advancing recovery node is `recover` on the T-Rex and both Compsognathus models, refused as before and naming `BEHAVIOR = 'recover'`; `--target stand` is the stance node alone on every model, a stance-only run the command line accepts on those three models too. **Amended 2026-10-18 (decision D-D38):** that stance-only run is `--target stance`; `--target stand` is refused before any directory is written, naming `stance`. |
| D-A25 | Automatic trunk selection (landed 2026-09-16 in #543). `TRUNK_FROM = "auto"` (the notebook default) and `curriculum --trunk-from auto` call `ancestors.select_trunk`: every run beside the new one under `LOG_BASE/<species>/<algo>/`, newest first by directory name, is tried against the target's ancestors root-first (above `RETRAIN_FROM`; the target is never reused, D-A18) under the §4.2 rule with `follow_records=True`, each child chained onto the ancestor found for its parent; the run covering the most consecutive nodes wins, the newest on a tie — one coherent trunk, never one node from one run and its child from another (rule 4 refuses that anyway). The selection copies and writes nothing; the loop then reuses from the selected `TRUNK_DIR` exactly as from a hand-picked run, and the chain loop re-derives `TRUNK_DIR` from that selection (a storage-cell rerun resets nothing it relies on; a widen session uses no trunk). A run whose `provenance.json` or root `stage_config.json` names another species, algorithm or backend is refused before the rules run (the identity check a pinned trunk makes; the rules never read the algorithm), and no unreadable neighbour aborts the scan. It prints the choice, the replication each covered node rests on (informative: a provisional ancestor is reused and labelled, never refused — §4.5 makes replication provenance, not a gate), the refused runs with the rule that refused them (the first 20; the selection holds every run scanned), and any run whose root passed under an older policy interface that the widen gate can bridge (a `WIDEN_FROM` candidate, D-C13/D-C17). A `WIDEN_FROM` session selects nothing: its widened root is judged in the new run and no earlier checkpoint descends from it. The certified library's recommendation (PRs #540/#541) no longer feeds canonical chains: it read a separate store none of the certified runs had been published into, refused a single-seed stance as provisional under trex's two-seed publication bar, and its publish step disconnected the runtime on a widened root. **Amended by D-D14 (2026-09-24, PR-14a):** the notebook's `WIDEN_FROM` and `select_trunk`'s `widen_from` parameter are gone; an older-interface root is reported as a candidate for the command-line `widen_checkpoint` (its `--max-revision-gap N` named), and a session that judges a root widened on the command line runs without a trunk — the resolve cell refuses a resolved trunk until that root holds a verdict (D-C13). **Clarified 2026-09-25 (#557, the notebook follow-up to PR-7):** "newest" is the greatest run directory name (`select_trunk` ranks by coverage, then name), which is the newest run only among names of one timestamp form (the storage cell's `YYYYMMDD_HHMMSS` ids, or the command line's default `curriculum_<YYYYMMDD_HHMMSS>`); any other name, such as a custom `RUN_ID` like `my_run` or a caller-named `--output-dir`, sorts by its text (`my_run` after every timestamp id). The code is unchanged; the notebook, the command-line help and the docstrings now say so. **Amended 2026-10-03 (the cleanup plan's §2 row 6, ROW-4/6; not a D-D row):** on the D-C13 clause, the resolve cell no longer refuses a trunk over an unjudged widened root: the chain loop judges the widened root before it consults the trunk (decision 6 (b), which the maintainer took on 2026-10-02); a widened pair cut short is refused as an interrupted node, and a root widened into a run that already holds it as an `ancestors/` record is refused when it is an ancestor of `BEHAVIOR`'s node, naming a new run id. `refuse_trunk_over_unjudged_widened_root` stays exported, and the notebook no longer calls it. ROW-4/6 landed as #591 on 2026-10-03. |
| D-B1 | The declared hunting panel is n = 30 (`min_eval_episodes = 30`), matching every existing panel; the coupled knobs (the notebook's selected panel, the provenance `evaluation_episodes`, `train_base`'s velocity episodes, the Ray Tune worker) are untouched and `test_trex_behavior_gates_on_task_success` pins the coupling. n was not sized up, so the bar separates ≥ 0.67 from 0.5, not 0.45 from 0.60. Amended by D-D17: the Ray Tune worker is retired (cleanup PR-A), so it is no longer among the coupled knobs. |
| D-B2 | `min_success_lcb = 0.5` (20/30 needed, LCB 0.5006; 19/30 → 0.4669 fails). PROVISIONAL: frozen before any Phase-B pilot on the committed 2026-03 row alone (29/30 → 0.8514, a recomputation from a rounded mean with no per-episode evidence), to be re-frozen from the first pilot's `evaluation_selected.csv`; the TOML comment, §4.4, KNOWN_ISSUES and the catalog say so. A verdict minted under the provisional bar is re-judged, never retrained, when the bar moves (D-B7/D-B8). |
| D-B3 | `required_consecutive` is an ALLOWED, not required, key of `task_success/v1`: scheduler hysteresis for the in-training manager only, in the digest like every advancing kind's copy of it. The certifying verdict is one post-stage panel judged once; re-running a deterministic panel is not replication. |
| D-B4 | The collapse rail is `min_avg_reward = round(0.6 × statue reference) = 361` from the measured hunting statue (602.13 ± 175.35, n = 40, seed 3042, 40/40 full horizon, physics revision 7 — the stance rail's statue ratio), pinned by `statue_constants_physics_revision = 7`; the test asserts rail < reference. Re-derived WITH `collapse_peak_floor_reference` whenever a reward weight or the plant moves; a moved rail is a re-judge of every certified trunk, never a retrain. |
| D-B5 | `collapse_peak_warmup_timesteps = 1_000_000`: a judgment (the stance/recovery value, bounded below by the 600k stage-entry window and above by the ≤ 0.5 × budget test pin), not a replay — no behavior-stage evaluation series under the current config exists. Re-measured from the first full hunting run's `evaluations.npz` (KNOWN_ISSUES follow-up; §10). |
| D-B6 | A verdict without `gate_sha256` (judged before D-A22) is REFUSED by reuse rule 7 until re-judged — through the notebook's JUDGE branch (`generate_stage_artifacts`) or `backfill_gate_verdict.py --force [--gate current]`; recovery verdicts need a notebook re-roll; never a retrain. The refusal names both paths, and a refused trunk is trained in the new run. Every pre-D-A22 verdict on the log tree (the certified stance run `20260810_145546` and the seed-44 replicate `20260815_205206` included) is on the KNOWN_ISSUES inventory. |
| D-B7 | The gate digest payload is thresholds only: `gate_config_view` projects the kind, the schema version and `GATE_KINDS[kind] ∩ declared` (every advancing kind's set carries `min_eval_episodes` / `required_consecutive`; `none/v1` is empty; `_ALL_THRESHOLD_KEYS` for a null or unregistered kind), numerics float-normalised (`100` and `100.0` are one gate); schedule, collapse, diagnostic, retention and publication keys and any `[curriculum.jax]` table are excluded; a recovery verdict hashes the declared block. `min_avg_reward` STAYS in the digest, so a moved rail (a statue re-measure) is a re-judge of every certified trunk under the new gate. |
| D-B8 | Judge-time WARNING adopted; rule 7b NOT adopted. `generate_stage_artifacts` logs a warning naming the `gate_config_differences` when the block it judges under is not the one the directory's `stage_config.json` recorded, and never refuses. The stricter 7b variant — refusing reuse when the candidate's own recorded block does not digest to the verdict's `gate_sha256` — was rejected because under D-A22 re-judging a directory under an edited gate IS the intended path (edit a threshold, re-judge, never retrain): 7b would refuse every legitimately re-judged directory and make retraining the only escape. Only the gate the verdict was judged under counts. Replacement checks: a verdict must agree with itself — when both `gate` and `gate_sha256` are present, `gate_sha256 == gate_config_sha256(gate)`, enforced in `read_gate_verdict` and hence by rule 7, `load_ancestor_records` and the audit — and the warning names the differing thresholds. `backfill_gate_verdict.py --gate recorded\|current` judges under the recorded block (the default) or the checkout's, digests whichever it used, and refuses a stance report scored under other thresholds under either. |
| D-B9 | `certification_seeds` is a `[curriculum]` key in `gate_schema`'s new `_PUBLICATION_KEYS` class (positive int, default 1; `declared_certification_seeds`), never in any digest — `gate_sha256`, `hyperparameters_sha256` or `task_sha256` — so raising it relabels published bundles and invalidates no verdict; trex stance declares 2, every other stage the default. |
| D-B10 | Replicates are writer-recorded: `replication.discover_replicates_for_run` finds them among the run's `LOG_BASE/<species>/<algo>/` siblings when the publication cell runs, `save_result_bundle(replicates=...)` records them, and the catalog validates the record and re-derives only `provisional` from the current config — never aggregating across bundles. The `certification_panel` role and its collision question are settled by D-B17. |
| D-B11 | `provisional` iff `count < certification_seeds`; count and N are always rendered — `N run(s) of M seed(s)`, plus `provisional`, and `headlineFor` appends ` (provisional, N of M seeds)` — pinned identically in `species_catalog._format_replication` and the site's `formatReplication`. |
| D-B12 | `task_success/v1` is offline-evaluable in the sweep from the trial's recorded `success_count` / `n_success_episodes` (new `metrics.json` keys and CSV columns beside `success_lcb_threshold`), railing `min_avg_reward` on the same panel's `selected_mean_reward`. One shared trainer panel (`train_base.run_success_panel`) writes the trial's `evaluation_selected.csv` hash-bound to the handoff pair and records the count only beside it, so a trial never carries a FAILED task_success verdict without evidence and the row verdict and the on-disk verdict agree by construction. Amended by D-D17: the sweeps are retired (cleanup PR-A), so no trial row is evaluated offline; `train_base.run_success_panel` and the `success_count` / `n_success_episodes` keys and columns stay. |
| D-B13 | The JAX preflight (`jax_curriculum`) refuses a FINAL stage declaring a registered, non-none kind no JAX path can judge — `task_success/v1`; a final stage declaring `recovery_quality/v1` (the single-stage pilot's designed shape; `_FINAL_STAGE_PILOT_KINDS`, with no condition on chain length) keeps its allowance — before any training, on the declared `gate_kind` alone: the final stage's block is still not schema-validated there, so single-stage pilots train as before. Amended by D-D17: `jax_curriculum` and its preflight are retired (cleanup PR-B); the final-stage rule stays as one of D-D17's entry conditions for adding a backend back: a re-added backend refuses, before any training, a FINAL stage declaring a registered, non-none kind it has no path to judge. |
| D-B14 | Every other species' hunt, compsognathus and compsognathus_robot included, keeps `reward_and_length/v1`; every reader keeps both paths. The committed-config test loops cover every species with a stage manifest, and the compsognathus pair's collapse-floor omission passes those loops by the species-level decision their stance TOMLs record ("Deliberately omit a collapse floor"), not by a per-stage statement (§10). |
| D-B15 | Exporting stance and recovery gate statistics into summary stage rows (D-A9's "Phase B" pointer) is DEFERRED to a later phase (Phase E); every pointer — `species.ts`, the catalog docstring, `species_manifest.toml`, recipes.md — defers it to a later phase citing D-B15. The hunt is the exception: `task_success/v1` exports `selected_model_success_lcb` with the count and panel size. |
| D-B16 | A replicate is a sibling run of the SAME recipe: equal `task_sha256`, plant identity, `gate_sha256` (the verdict's) and `hyperparameters_sha256`, a passed reusable verdict and a distinct training seed — tightening §4.5's "same task and plant, gate passed" (two seeds of different recipes are not replication of one deliverable). A sibling with no recorded `hyperparameters_sha256` (pre-D-A21) gets it DERIVED from its recorded blocks (`config.recorded_hyperparameters_sha256`), never skipped for that alone; one whose verdict lacks `gate_sha256` (pre-D-A22) IS skipped until re-backfilled (D-B6). Implemented reading of a complete bundle: a re-run of the publication cell that changes nothing but the replication fields (`replication`, `provisional`) of a VERIFIED-COMPLETE bundle regenerates its derived artifacts — the writer masks those two fields when comparing the existing and prospective summaries (`_without_replication` in `reporting/bundles.py`) — and any other difference, a changed `certification_seeds` included, is still refused as immutable; a raised bar reaches published bundles through the catalog's re-derivation of `provisional` from the current declaration (D-B9/D-B10), not through republication; a late replicate is therefore counted by re-running the counting run's publication cell with the sibling present. The audit reads an absent curriculum block as the default bar 1. |
| D-B17 | `certification_panel` is a default seed role: `initialize_result_bundle`'s default seed roles (`result_bundle/provenance.py`) carry `certification_panel = PUBLICATION_SEED_START` (3042) for every SB3 bundle, and a declared role must equal that block (refused otherwise). It is bound PER EVIDENCE, not through `evaluation_protocols` (the name deliberately lacks "evaluation"; the protocol family is untouched): stance panel row *i* must carry `panel_seed == role + i`, and a recovery `gate_resolution.json`'s `decision_procedure.panel_seed_start` must equal the role. A recorded stance PASS in a bundle whose provenance lacks the role is refused at publication (fail closed); pre-Phase-B stance bundles need republishing (none are committed; KNOWN_ISSUES). `seed_role_collisions` is unchanged — it checks training and selection roles only, so the role may equal the publication seed. |
| D-C1 | All six plants take the 3-dim command segment now, appended LAST: velociraptor r9 → 10 (67 → 70), trex r12 → 13 (61 → 64), brachiosaurus r7 → 8 (83 → 86), dibothrosuchus r6 → 7 (77 → 80), compsognathus r1 → 2 (53 → 56), compsognathus_robot r1 → 2 (43 → 46); physics and visual revisions, the `observation_schema` strings and the Box bounds unchanged (`plant_versions.toml` note 12). |
| D-C2 | The two compsognathus recovery calibrations are RESTAMPED (plant identity + task hash + a `restamp_history` entry) by `scripts/restamp_recovery_calibration.py`, not re-measured: fixed-command nulls never read the observation and the reset draw stream is pinned by the golden fixture (`tests/fixtures/phase_c_reset_golden.json`). The `profile_sha256` moves, so every pre-Phase-C compsognathus recovery freeze must be re-frozen (RECOVERY_CALIBRATION.md, CHANGELOG, note 12; amendment A10). |
| D-C3 | The six command kwargs (`command_mode`, `command_speed_range`, `command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`, `command_switch_jitter`) land on `BaseDinoEnv` / `MJXEnvConfig` now with inert defaults. As amended (A1): while the effective `command_mode == "none"` the six keys are carved out of the task-fingerprint env section for EVERY species (the compsognathus perturbation carve-out's pattern), so off-configs keep their pre-Phase-C env encoding, a stage's `task_sha256` moves only through the plant's `policy_interface_sha256`, and `test_compsognathus_task_compatibility` stays green. Amended by D-D17: the `MJXEnvConfig` dataclass is retired with the JAX/MJX runtime (cleanup PR-B), so the six keys live on `BaseDinoEnv` only; the carve-out and every committed `task_sha256` are unchanged. Amended by D-D2 (consolidation PR-9, 2026-10-05): the five numeric kwargs are gone (`command_frame.RETIRED_COMMAND_ENV_KEYS` names them; the constructors and the task fingerprint refuse each); `command_mode` and one `command_config: DirectionCommandConfig \| None` remain on `BaseDinoEnv` and the five species constructors, the carve-out pops both while the effective mode is `"none"` (no committed `task_sha256` moved) and records the config field by field under a live mode, and the payload's `command` section is the controller's manifest. |
| D-C4 | A non-zero plant-contract probe: `COMMAND_PROBE_VECTOR = (0.25, -0.5, 0.75)` is injected on both backends (`env._command` in the SB3 probe, `command=` in the MJX probe) so SB3/MJX parity is non-vacuous on the new slot; SB3-only species keep `backend_observation_equal = None`. Amended by D-D17: nothing trains on MJX (cleanup PR-B), but the MJX probe still runs on the frozen MJX interface core (`build_mjx_observation` and the `mjx_config` registrations) with the same injected vector, so the parity check is unchanged. |
| D-C5 | The per-episode command comes from a `BaseDinoEnv._draw_episode_command()` hook called once in `reset()` after the push block and before `_get_obs()`; it draws nothing under `"none"`, and Phase D replaces the hook body, never `reset()`. (As found 2026-09-17: the #540/#541 pilots bypass this hook — their env classes refuse `command_mode != "none"` on `BaseDinoEnv` and assign `self._command` directly after `super().reset()`; decision D-D1 folds them back through it, PR-9.) Since consolidation PR-9 (2026-10-05) the hook body seeds `BaseDinoEnv`'s direction controller under a live mode (one draw on the reset stream after every existing draw, through `_command_rng()`, which the recipe env overrides with its own command stream), `step()` advances it last through `_update_command()`, and the recipe env writes `self._command` only through those two. |
| D-C6 | No Box bounds change (stays `(-inf, inf)` float32) and no `observation_schema` rename. |
| D-C7 | SB3 also refuses `command_mode != "none"` in Phase C (`validate_command_mode` raises the "reserved for … Phase D" text in `BaseDinoEnv.__init__`); MJX refuses every live mode in `canonicalize_env_kwargs` and `MJXDinoEnv.__init__` until an MJX command path lands (A7); an unknown mode is refused on every backend naming the valid set. Amended by D-D17: `canonicalize_env_kwargs` and `MJXDinoEnv` are retired with the JAX/MJX runtime (cleanup PR-B), so no MJX environment exists to refuse a mode; `command_frame.validate_command_mode` keeps its `jax-mjx` refusal until PR-9 rewrites the file (the cleanup plan's §4.9). Since consolidation PR-9 (2026-10-05) the SB3 branch and `SB3_COMMAND_REFUSAL` are gone: SB3 builds the controller for both live modes, and the `jax-mjx` refusal stays (invariant 9). |
| D-C8 | Widened checkpoints record the eight `config.WIDEN_LINEAGE_KEYS` in the run block (never the `LOAD_LINEAGE_KEYS` — a widened node is a root; `ancestors._check_chain` refuses a root that entered from a parent and the audit binds `parent_run_id` to an `ancestors/` record) and re-stamp the current plant identity and the stage's CURRENT task fingerprint on BOTH artifacts, with the parent hashes also in the archive's `mesozoic_widen_lineage` attribute. |
| D-C9 | Widen output = the parent's own handoff name (`robust_best_model` or `best_model`, exactly one) plus byte-identical `<stage_label>_final.*` copies, so the notebook JUDGE branch fires on the widened directory; never a verdict, provenance, resolution, evaluation, metrics or periodic checkpoint. |
| D-C10 | Adam `exp_avg` / `exp_avg_sq` (and `max_exp_avg_sq`) moments are padded identically to their weights, matched by parameter order with shape assertions; a missing optimizer member, a stale moment, or a non-Adam moment still shaped like the pre-pad weight is refused rather than carried (unpadded moments load but crash the first update). |
| D-C11 | SUPERSEDED by A1/A9: the task payload carries a `command` section only when a command manifest is passed (`command_manifest: Mapping \| None = None` on `compute_task_fingerprint` / `derive_stage_task_fingerprint`; both backends expose `command_manifest() -> None`); `"command"` is still added to the differing-sections tuple and the v1 valve list; the schema string stays `mesozoic.task-fingerprint/v2`. |
| D-C12 | `num_timesteps` is preserved on widen (recorded in `widen_report.json`). |
| D-C13 | Notebook knob `WIDEN_FROM` (configuration cell; a run id under `LOG_BASE/<species>/<algo>/` or an absolute run directory) plus ONE widen cell after the RESOLVE cell, which widens the parent's certified ROOT handoff into `RUN_DIR`'s root stage directory (refusing an occupied target, this run's own directory, a parent without `provenance.json`, and a parent of another species / algorithm / backend) and trains nothing; the chain loop then refuses the verdict-less directory and JUDGES its `<stage_label>_final.*` pair. Never via `RUN_ID` pointing at the old run — for a pre-Phase-C run the storage cell's `initialize_result_bundle` refuses the directory first (the recorded `plant_identity` is not this checkout's), and for a same-plant run the chain loop raises "mint a fresh RUN_ID" on a refused verdict. The cell imports every name it uses (it runs before the infra cell; A14a), refuses a parent stage without a `stage_config.json` or a recorded run seed before the tool runs, and its `SEED` refusal names the remedy (correct `SEED`, restart the runtime so a fresh `RUN_ID` is minted — the storage cell will not re-mint the directory under another seed). **Amended by D-D14 (2026-09-24, PR-14a):** the widen cell and `WIDEN_FROM` left the notebook. A root is widened on the command line into a NEW run id (`widen_checkpoint --to-stage-dir <LOG_BASE>/<species>/<algo>/<new run id>/<stage_dirname(species, root)>`), which the notebook re-enters with `RUN_ID` set to that id — still never the parent's — and whose chain loop judges the root; the resolve cell refuses a resolved trunk until the widened root holds a verdict. The restart-the-runtime remedy of the `SEED` refusal no longer applies: the storage cell refuses a wrong `SEED` before anything is written (D-C14). **Amended 2026-10-03 (the cleanup plan's §2 row 6, ROW-4/6; not a D-D row):** the resolve cell no longer refuses a trunk over an unjudged widened root: the chain loop judges the widened root before it consults the trunk (decision 6 (b), which the maintainer took on 2026-10-02); a widened pair cut short is refused as an interrupted node, and a root widened into a run that already holds it as an `ancestors/` record is refused when it is an ancestor of `BEHAVIOR`'s node, naming a new run id. `refuse_trunk_over_unjudged_widened_root` stays exported, and the notebook no longer calls it. ROW-4/6 landed as #591 on 2026-10-03. |
| D-C14 | Both certified stance parents (`20260810_145546` seed 42, `20260815_205206` seed 44) are widened and re-paneled in two sessions so trex stance reads `2 runs of 2 seeds` instead of `1 run of 2 seeds; provisional` (replicates need distinct `run.seed` values, which the copied run blocks provide, and the same `task_sha256`, which only widened siblings share); each session sets `SEED` to its parent's seed — the widen cell refuses `SEED != ` the parent's recorded `run.seed`, because the minted provenance publishes `training_seed = SEED` (A14b). **As found at implementation (2026-09-14):** both parents are policy-interface r11 archives — trex r11 → r12 (commit `8795e28`, 2026-08-16, fingerprint-only: `observation_dim` 61 on both sides) landed after they trained, and an archive's identity is stamped at training time — and `widen_checkpoint`'s default bound is one revision, so it refuses them at r13 (`policy_interface_revision: parent=11, current=13 (gap 2 exceeds max_revision_gap=1; pass --max-revision-gap 2 / max_revision_gap=2 …)`; `--allow-legacy-plant` covers only an archive with no identity). Remedy, D-C17: the gate takes a bounded gap — both sessions set `WIDEN_MAX_REVISION_GAP = 2` (the CLI's `--max-revision-gap 2`), under which every other gate field is still checked and the report records `revision_gap` 2; the r12 stance PASSes of 2026-08-16..18 widen under the default bound but are not the certified parents (KNOWN_ISSUES, the Phase C entry). **Status 2026-09-19:** seed 42 is certified at r13 by the fresh run `20260914_123816` (stance and locomotion PASS, selected by `TRUNK_FROM = "auto"`), which supersedes the Session 1 widen of `20260810_145546`; only the seed-44 widen of `20260815_205206` (`WIDEN_MAX_REVISION_GAP = 2`, `SEED = 44`) is still owed for the two-seed bar. Seed 43 (`20260915_160239`, a fresh r13 stance) failed the duty rail (mean 0.0323, UCB 0.0350 against 0.02). **Status 2026-09-21:** the seed-44 widen ran 2026-09-20 as `20260920_010912`; the re-panel reproduced the r11 certificate to every printed digit, its bundle records trex stance at replication 2 (with `20260914_123816`), and the two-seed bar is met. **Amended by D-D14 (2026-09-24, PR-14a):** the widen cell that enforced `SEED` = the parent's `run.seed` is gone; the storage cell enforces it from disk before `initialize_result_bundle` (`result_bundle.refuse_widened_seed_mismatch`, over every stage directory of the run whose run block records `widened_from_run_id`; a `provenance.json` minted under another `training_seed` before the widen is refused too, with the remedy of widening again into a new run id), so a wrong `SEED` refuses with nothing written; the sessions' `WIDEN_MAX_REVISION_GAP = 2` is the command line's `--max-revision-gap 2` from now on. |
| D-C15 | Session order follows the maintainer's priority: stance (widen + re-panel) → recovery freeze re-roll → walk (C½); hunting later. |
| D-C16 | `validate_mjx_environment_plant` gains an observation-width check against the plant identity (`_deterministic_probe_data` imported lazily inside it, A8). Amended by D-D17: `validate_mjx_environment_plant` is retired with the MJX trainers that called it (cleanup PR-B); the plant-contract probe still checks the MJX observation's shape against the environment's on the frozen MJX interface core. |
| D-C17 | The widen gate takes a bounded revision gap (adopted 2026-09-14, from the PR-C3 review): `widen_checkpoint(..., max_revision_gap: int = 1)`, the CLI's `--max-revision-gap N` and the notebook's `WIDEN_MAX_REVISION_GAP` (threaded into the widen cell's call) admit a parent `1 <= current - parent.policy_interface_revision <= N` revisions behind; the default stays 1 (fail closed: the Phase C bump alone). Every other field is checked whatever the bound — same species, `physics_sha256`, `nq` / `nv` / `nu`, `action_dim`, `observation_dim + COMMAND_WIDTH == current` — so only fingerprint-only intermediate bumps can be crossed; a parent further behind than N is refused with both revisions, the measured gap, the bound and the flag named. `widen_report.json` records `revision_gap` / `max_revision_gap`, the archive's `mesozoic_widen_lineage` carries `revision_gap`, and the run block's `widened_from_policy_interface_revision` names the parent's revision. Opting in asserts, from `plant_versions.toml`'s numbered notes, that the intermediate bumps changed nothing the widening cannot bridge — for trex r11 → r13, note 11 recorded the perturbation engine and note 12 appended the command segment. The two r11 stance parents widen under `WIDEN_MAX_REVISION_GAP = 2` (WS-C4 Sessions 1 and the seed-44 repeat); a hand re-stamp is never the path. **Amended by D-D14 (2026-09-24, PR-14a):** the notebook's `WIDEN_MAX_REVISION_GAP` knob left with the widen cell; the bound is `--max-revision-gap N` on the command line and `max_revision_gap=N` in the API, and `DEFAULT_MAX_REVISION_GAP` stays 1 (pinned in test_widen_checkpoint.py and, for the shared test matrix, in test_sb3_notebook_pins.py). |

### 6.2 Decisions taken at the 2026-09-17 consolidation review (the D-D and G series)

Taken by the maintainer on 2026-09-17 at the review of the direction/terrain
pilots (#540/#541) against this design; the review's working labels D1..D12
map one-to-one onto D-D1..D-D12 (D-D13 was added on 2026-09-20 when the hold
lifted), and nothing in §6 or §6.1 is renumbered.
The sequence of consolidation PRs each row unblocks — fifteen in all, with
sizes, per-PR file lists and acceptance tests — is
`docs/CONSOLIDATION_PLAN_2026_09.md` (§3 the sequence, §6 and §7 the same
decisions with the questions they answered). Which PRs of that sequence have
landed is in its status table (the one-landing-record rule, `docs/README.md`
Conventions); PR-2 is the 2026-09-19 documentation pass that records these
rows, **PR-3..PR-15 were released on 2026-09-20 in the notebook-first order of
D-D13**, and PR-8 follows the deferred cleanup, which the maintainer put first
on 2026-10-02. Training runs in parallel
(G3): the walker sessions in `docs/NEXT_STEPS.md` use the current notebook.
D-D11 and D-D12 were recommended on 2026-09-17 and confirmed by the
maintainer on 2026-09-20, when the hold lifted (D-D13) and the widen path's
future was fixed (D-D14); D-D15 (the PR-14 split) was taken on 2026-09-24, and D-D16
(the notebook-safety PR, outside the numbered sequence) on 2026-09-25.
D-D17 (the backend retirement), D-D18 (mypy with SB3 in CI), D-D19 (the
CHANGELOG release cut) and D-D20 (atomic final and handoff checkpoint pairs)
were taken on 2026-09-26 from the decisions `docs/CLEANUP_PLAN_2026_09.md` §2
lists (its rows 1, 8, 3 and 7), and D-D21 (0.3.9 as the clean base release) on
2026-09-27 (its row 20). D-D22 (the digest snapshot as a CI check) was taken
on 2026-09-29 (its row 16) and carried out by its own PR, ROW-16, which landed as
#571 the same day. D-D23 (the floor-truth stance gate `stance_quality/v2`) was
taken on 2026-10-06 from `docs/GAIT_QUALITY_PLAN_2026_09.md` §2, its GQ-6 (a)
and GQ-7 (a), for stance only, and D-D24 (the T. rex physics revision r8 and
the T. rex stance's adoption of that gate) the same day, overtaking the same
plan's GQ-12 and its §5.4 trex `TRUNK_FROM`. D-D25 (the velociraptor plant
revision r3, its SB3-only exit and the velociraptor stance's adoption of the
gate) was taken the same day too, overtaking KNOWN_ISSUES' deferral of the
velociraptor plant review until the T-Rex clears stages 1–3 and the same
plan's §9, which left the velociraptor foot-sensor repair out. D-D26 (the
compsognathus plant revision r2, its soft-cubic leg interface and the
compsognathus stance's adoption of the gate) was taken on 2026-10-07,
overtaking the same plan's GQ-11 (a) for the compsognathus walker, which the
plant revision strands. D-D27 (the T. rex stance follow-up after the first
two physics-r8 stance runs: its reward terms, its recipe's start, its gate
block and a report-only heading probe, on the unchanged plant) was taken on
2026-10-07, amending D-D24's stance task and gate block, and D-D28 (the pad
centre-of-pressure bar, after the review of the seed-44 run found its left
foot on its pad's front edge) the same day, adding a criterion to D-D23's
kind and to D-D27's T. rex block. D-D29 (the T. rex physics revision r9, an
elliptic friction cone) was taken on 2026-10-11 with the maintainer's heading
decisions of that day (`docs/HEADING_INVARIANCE_DESIGN_2026_10.md`), as the
separate physics revision that design keeps out of its observation-only
migration. D-D30 (the elliptic friction cone on the velociraptor, the
brachiosaurus and the dibothrosuchus too) was taken the same day, so every
species has one friction model. D-D31 (heading-free observations for the
T. rex and the anatomical compsognathus, the policy-interface half of that
design) was approved by the maintainer the same day, carrying out D-D27's
2026-10-07 amendment, and D-D32 (the six-heading stance certificate, the
design's gate, for the T. rex and anatomical compsognathus stances) was
taken the same day too.
D-D33 (binomial species IDs and the embodiment axis) was taken the same day.
D-D35 (the recipe label `recover` for the recovery node, so `stand` is the
stance node on every model) was taken on 2026-10-16 (D-D34, the T. rex
physics r10 foot, was taken the same day in a draft still open then).
D-D36 (one short T. rex walking pilot, a deliberate exception to
`docs/GAIT_QUALITY_PLAN_2026_09.md`'s GQ-3 (a) and GQ-17) was taken on
2026-10-17, after the T. rex physics-r10 stance screen was judged.
D-D37 (three full T. rex stance screen runs in place of walking pilot 2,
superseding for them the maintainer's rule of 2026-10-16 against further
stance-only reward experiments) was taken on 2026-10-18, and D-D38 (one
name for the standing behavior: the stance node's recipe label is `stance`,
and `stand` is refused with a pointer) the same day.
D-D39 (the next T. rex stance screen: three runs of D-D37's L+T with a
foot-stagger penalty, from scratch, against two D-D37 runs as the control)
was taken on 2026-10-19.
D-D40 (a from-scratch T. rex stance screen of a per-foot centre-of-pressure
band on D-D39's arm, after D-D39's screen read its line 5) was taken on
2026-10-20, the day D-D39 (its outcome, and its runtime confirmed) and
D-D36 (walking pilot 2's parent) were amended.
D-D41 (a replication of D-D40's band recipe on two more seeds, after D-D40's
screen read its line 3) was taken on 2026-10-22, the day D-D40 was amended
(its build, runtime and outcome); D-D36's amendment, dated 2026-10-21 (walking
pilot 2 rebuilt), was recorded that day, with its parent kept by D-D41.
D-D42 (the next T. rex stance round targets the tail saturation habit, its
design and criteria to be fixed and approved before any training, with a
training-history report and an actuator saturation report, both report only,
after D-D41's replication read its line 2) was taken on 2026-10-24, the day
D-D41 was amended (its outcome); the round's design is pending.
D-D43 (a scratch CPU continuation screen of a training-loss bound on the T. rex
tail actuator's mean, upper side only, in three arms, the first step of D-D42
(1)'s round) was taken on 2026-10-24, the day D-D42 was amended a second time
(three more report-only tools) and D-D43 itself once (its gate re-pin); its
outcome, neither bound adopted, and the maintainer's decision after it are its
amendment of 2026-10-25.
D-D44 (the T. rex physics revision r10, the tail pitch springs preloaded so
the passive tail holds level, with `tail_home_pose` re-centred on the new
settled tail) was taken on 2026-10-25, after a posture study, a CPU probe and
CPU checks, none of which trained; its "physics r10" is the plant counter's
next integer, not D-D34's foot of the parked draft #623.
D-D45 (walking pilot 2 retired, a new pilot to be designed once r10 has a
stance parent) and D-D46 (report-only telemetry for the first r10 stance
round) were taken on 2026-10-25, once D-D44's revision was on `main`, with
D-D47 (no MuJoCo sensor before that round) and D-D48 (the in-run K3 noise
panel) the same day; D-D48 was amended on 2026-10-26 (the panel on in that
round). D-D49 (the first T. rex stance round on physics r10: arm LTSB on
seeds 60, 42 and 80, from scratch, 12M steps on Colab L4s) was taken on
2026-10-25 and amended that day (its statistic's wording) and on 2026-10-26
(its run branch); D-D50 (that round's all-or-nothing run handling) on
2026-10-26, amended that day and on 2026-10-27; and D-D51 (its criteria
approved and its three runs launched) on 2026-10-27, amended the same day (a
report-only look at the running files), its results pending.
D-D52 (the next step after that round: a CPU continuation screen, from the
round's own checkpoints, of the entropy coefficient as recorded against 0, its
criteria to be fixed and approved before any training, and a records pull
request for the round's results) was taken on 2026-10-28, the day D-D51 was
amended (its outcome: line 1, a tail rail formed on r10 in training); its
design is pending.

| # | Decision |
|---|---|
| D-D1 | Direction-following and terrain traversal are ordinary manifest nodes under `locomotion` (this plan's Phase D shape: stage TOMLs, registered gate kinds, reuse through `find_certified_ancestor`), not a standalone pipeline. Consequence accepted: a follow/terrain node needs a certified locomotion ancestor at the current policy interface (r13 for trex, each other species at its own revision) — trex has one (`20260914_123816`); the other five species get theirs from the walker sessions (G3); the manual single-node cell is the escape hatch until then. Unblocks PR-9 to PR-15. |
| D-D2 | One `command_config: DirectionCommandConfig \| None` kwarg replaces the five numeric reserved kwargs (`command_speed_range`, `command_lateral_range`, `command_yaw_rate_max`, `command_switch_interval`, `command_switch_jitter` — none has a reader anywhere, and the pilots' dataclass is already validated and JSON-able); the task-fingerprint carve-out is extended so no canonical `task_sha256` moves while `command_mode` is `"none"`. Amends D-C3. Unblocks PR-9. |
| D-D3 | Command-slice normalisation follows invariant 8 (§8: reseed to mean 0 / variance 1, statistics keep updating); the pilots' `BehaviorVecNormalize` passthrough subclass (a pickled class in every sidecar) is deleted; #540/#541-trained policies saw different inputs and are not continuations. Unblocks PR-8 and PR-10. |
| D-D4 | Automatic parent selection is kept and is `ancestors.select_trunk` (D-A25, landed as #543 before PR-4/PR-5 delete the library trio, so the capability never lapses); the certified library is deleted outright and no recommendation pointer survives. Unblocks PR-4 and PR-5. |
| D-D5 | Four base stage TOMLs per species (`follow_direction`, `follow_direction_difficult_terrain`, `difficult_terrain`, and the existing `locomotion` as the `extends` parent) with a ~20-line `extends` key in `load_stage_config`; the terrain templates become a `terrain_families` list inside `difficult_terrain`; single-template runs are documented `[env]` overrides, not files. Refined by G1: three new files per species, `follow_direction_speed` folded into `follow_direction`. Unblocks PR-11 and PR-12. **Amended 2026-10-02 (the maintainer, as the cleanup plan's decision 9), carried out by its CU-13:** CU-13 pulls this decision's `extends` forward for recovery ← stance in trex, compsognathus and compsognathus_robot, in the form the maintainer chose on 2026-10-02: a per-table list, `extends = { stage = "stance", tables = [...] }`, one level deep, with no chains, never naming `curriculum`, and trex recovery not inheriting `[sac]`. As CU-13 carries it out, `extends = { stage = "<parent id>", tables = [...] }` is a top-level key naming a stage of the same species (resolved through the stage manifest of the file's own directory) and the tables it inherits, each the parent's with the file's own keys overriding in place or appended (a nested table the file declares replaces the parent's whole); trex recovery lists `stage`, `env` and `ppo` and stays PPO-only, and the compsognathus pair lists `sac` too. All 21 resolved stage configs are byte-identical, key order included, so no digest moves. CU-13's step 2 (compsognathus_robot ← compsognathus) waits for PR-12 and for an amendment of this form: a parent id resolves only through the stage manifest of the file's own directory, and the robot's recovery already extends the robot's stance, so a robot stance that extended compsognathus's would put that recovery in a chain. This decision's own nodes wait for PR-11, where each names `locomotion` in this form. CU-13 landed as #588 on 2026-10-02. |
| D-D6 | Honest gate name first: the pilots run `none/v1` (recorded, not enforced); then the certificate's per-episode statistic is registered as `terrain_command/v1` with one shared threshold block (first values from G4). This plan's `command_tracking/v1` (§4.6: per-event settle/dwell, the worst-of-heading-bins floor, the paired null against the command-blind walker) is a later second kind — new work outside the sequence; no implementation of it exists anywhere. Unblocks PR-13. |
| D-D7 | Notebook depth: only the `train_stage` wrapper over `train_base.train` now (PR-14; the §4.7 AST pins on the chain loop stay); whether to move the chain loop / widen / resume cells into a package module is decided after PR-14 has settled. Fixes PR-14's scope. **Amended 2026-10-03 (the cleanup plan's §2 row 9), carried out by its CU-6:** a slice of the resume cell and the archive-load preflight move into the package ahead of the deferred full move. The RESUME cell's periodic-pair walk is `curriculum.newest_intact_periodic_pair`, which reads `policy_loading._PERIODIC_CHECKPOINT_RE`; the archive-load preflight cell is one call of `policy_loading.sb3_archive_load_preflight`; and `train_stage` takes a keyword-only `evaluate` (default `True`; the RESUME cell passes `False`, since the chain loop's JUDGE branch evaluates, from disk, every node the RESUME cell trains, cleanup ROW-4/6). The chain loop, the rest of the RESUME cell (its refusals, warnings and report), `node_budget` (in the infrastructure cell, read by the chain loop, the RESUME cell and the manual cell), and the storage and resolve cells stay in the notebook, and the §4.7 AST pins on the chain loop stay, as this decision says; the full package move stays deferred (the cleanup plan's §7). |
| D-D8 | No interim behaviors notebook; the `COMMAND_TERRAIN_BEHAVIOR` mode switch (19 code cells at 22c1fc8) is tolerated until PR-12 deletes it — #542 already removed the dangerous default. Unblocks PR-12. |
| D-D9 | The #540/#541 behavior bundles on Drive are evaluation-only; none is carried forward as a training parent (their checkpoint identity hashes source files, so any edit to `behavior_env.py` strands exact resume anyway). Unblocks PR-6, PR-7 and PR-9, and is PR-12's acceptance basis. |
| D-D10 | Terrain stays an opt-in env subclass; no r14 interface bump to move the model swap into `reset()` (the reset source is fingerprinted; the queued height-channel removal stays queued, D3). Unblocks PR-7 and PR-9. |
| D-D11 | Confirmed 2026-09-20: CLI runs record stage duration and seed model construction like the notebook does (PR-14 aligns `train()` with the notebook's `alg_kwargs["seed"]` line and its duration recording). PR-14. Amended by consolidation PR-14c: implemented in `train()`, so for the `train` subcommand and Vertex sweep trials (`sweep/trial.py`), with a seed the stage's algorithm block names (`--override ppo.seed=N`) kept; `train_curriculum` (the `curriculum` subcommand) and the Ray Tune worker are not aligned (neither seeds construction nor records a duration), outside PR-14c's scope (D-D7). Amended by D-D17: the Vertex sweep trials and the Ray Tune worker are retired (cleanup PR-A); `train()` serves the `train` subcommand and the notebook's `train_stage`, and `train_curriculum` stays unaligned. **Amended 2026-10-03:** CLI curriculum runs now seed model construction and record stage duration as the notebook and `train` do (cleanup CU-10b, as the maintainer chose on 2026-10-02; the cleanup plan's §2 row 20). `train()` and `train_curriculum` train a stage through one shared body, `train_base._train_stage_body`, which seeds construction with `alg_kwargs.setdefault("seed", seed)` for every stage it trains: each node the `curriculum` subcommand trains builds its model, or re-seeds its warm start, under the run's seed (a seed the stage's algorithm block names, `--override ppo.seed=N`, is kept, as in `train()`), so without a block seed every checkpoint a node saves records the run's seed; and each trained node records `run.duration_seconds` in its `stage_config.json` (D-A15). Without a block seed the curriculum's initial weights, and everything trained from them, change; a run whose block names a seed was seeded before and only gains the duration. No certified result comes from the CLI curriculum, whose verdicts are training-time signals, so none changes. CU-10b landed as #590 on 2026-10-03. |
| D-D12 | Confirmed 2026-09-20: the dead `lateral_speed_scale` field (it always divides a zero) is dropped when the TOMLs are rewritten. PR-11 / PR-12. |
| D-D14 | Taken 2026-09-20: the widen path stays for the two pending parents (NEXT_STEPS.md sessions 1 and 2: the trex seed-44 r11 stance and the compsognathus seed-42 r1 stance) and is then demoted to CLI-only — the notebook refactor (PR-14, or a PR right after it once both sessions are decided) deletes the widen cell and the `WIDEN_FROM` / `WIDEN_MAX_REVISION_GAP` knobs, while `widen_checkpoint` stays a command-line tool for the next interface bump. Amends the "knobs kept" list of the consolidation plan §4: those two knobs leave once sessions 1 and 2 are decided. The 2026-09-19 kernel deaths were the cross-interpreter archive fault (KNOWN_ISSUES), not the widening; the alternative (deleting the path now) would cost about a day of Colab retraining for the two parents. Both sessions decided PASS by 2026-09-21 (`20260920_010912`, `20260921_203149`: each re-panel reproduced its parent's report to every printed digit); the knobs leave with PR-14. **Scheduled by D-D15:** PR-14a is the PR that deletes them. |
| D-D13 | Taken 2026-09-20 (the maintainer released the consolidation hold): the sequence lands notebook-first — PR-3, PR-4, PR-5, PR-6, then a notebook-only slice of PR-12 (the `COMMAND_TERRAIN_BEHAVIOR` mode switch, the ten `BEHAVIOR_*` knobs, the direction/terrain cells and their guard sites, and `behavior_notebook.py` with its tests and pins; `train_behaviors.py` stays a CLI-only path) pulled ahead of PR-11, then PR-14, then PR-7 .. PR-11, the rest of PR-12, PR-13 and PR-15. Amends D-D8: the mode switch is tolerated only until that slice, and between the slice and PR-11 the direction/terrain pilots have no notebook path (they are evaluation-only under D-D9). |
| D-D15 | Taken 2026-09-24: consolidation PR-14 lands as three PRs, in the order PR-14a, PR-14b, PR-14c. PR-14a (the storage path): the widen cell, `WIDEN_FROM`, `WIDEN_MAX_REVISION_GAP` and `select_trunk(widen_from=)` go (D-D14); the pre-split item (b)'s resolve-before-storage reorder is dropped — a root widened on the command line keeps D-C14 through an on-disk seed check before `initialize_result_bundle` and D-C13 through a resolve-cell refusal of a trunk until it holds a verdict; `RUN_ID` becomes a configuration-cell knob resolved into the `_ACTIVE_RUN_ID` memo that later cells read (an explicit id is memoised once the provenance accepts it); a session that would judge or train a node into a run whose bundle is complete is refused before anything is trained or written (by the resolve cell, and by the chain loop for what the resolve cell cannot predict), and the zero-action cell stops rewriting such a run's copy. PR-14b: the one disconnect path, IPython video display and the baseline cells. PR-14c: `train_stage` over `train_base.train` (D-D7, D-D11). Amends D-D13's order. **Amended 2026-10-03 (the cleanup plan's §2 row 6, ROW-4/6; not a D-D row):** a root widened on the command line still keeps D-C14 through the storage cell's on-disk seed check, and keeps D-C13 through the chain loop instead of a resolve-cell refusal: the resolve cell no longer calls `refuse_trunk_over_unjudged_widened_root` (still exported, and uncalled), and the loop judges a node `RUN_DIR` holds trained but unjudged, the widened root among them, before it consults any trunk (a widened pair cut short is refused as an interrupted node), and refuses a root widened into a run that already holds it as an `ancestors/` record when the root is an ancestor of `BEHAVIOR`'s node, naming a new run id. ROW-4/6 landed as #591 on 2026-10-03. |
| D-D16 | Taken 2026-09-25, when the maintainer put the notebook-safety PR first among the cleanup before the direction/terrain path: in the SB3 notebook (a) the RESUME cell trains nothing for a node that holds `gate_verdict.json` or its final pair (`<stage_label>_final.zip` and its `_vecnorm.pkl`): the chain loop reuses, refuses or judges it. Those files decide that a node finished, not the newest periodic step (an early stop or an `N_ENVS` that does not divide the checkpoint cadence leaves it short of the budget); a spent budget without the final pair is refused; training a node further after an early stop is a new attempt in a fresh `RUN_ID`. (b) `QUICK_TEST` runs live under `LOG_BASE/<species>/<algo>_quick_test/`, outside the tree `select_trunk` and `discover_replicates_for_run` scan, with no digest or provenance change. (c) The RESUME cell moves ahead of the chain loop, so a resume is `RUN_ID` + `RESUME_STAGE` + Run all, and the sections regroup (3 open the run and resolve the chain, 4 preflights, 5 resume, 6 chain, 7 manual node, 8 report, 9 disconnect). Not taken: a library guard refusing `resume_same_stage` into a judged directory in `config.refuse_occupied_stage_dir` (it would amend D-A20 and also cover the CLI) and a reuse-rule guard on short run budgets (it would reclassify existing Drive runs). **Amended 2026-09-25 (the #558 follow-up, after two reviews of #558 once it merged):** only an intact final pair (checked like a periodic pair by `curriculum.checkpoint_pair_problem`: the zip's `testzip` and SB3 members, a sidecar that unpickles) marks a finished node, in the RESUME cell and in the chain loop, whose JUDGE branch sends a broken one to the interrupted-node refusal; the RESUME cell resumes over it from the newest intact periodic pair when that pair is within one checkpoint cadence of the budget, and refuses it further short (an early stop whose final save was cut short: a new attempt); the run memo counts only in the tree `SPECIES`, `ALGORITHM` and `QUICK_TEST` select, so `RUN_ID = ""` mints a fresh run after one of them changes; a resume of a node `RETRAIN_FROM` covers is refused (the chain loop would train it instead of judging it); such a node, trained here although the trunk run certifies it, is resumed with `BEHAVIOR` set to it (the loop looks for its target only in this run), and the deeper chain continues in a fresh `RUN_ID` trunked from this run. Clause (b)'s replicate scan is per run (`discover_replicates_for_run` scans a run's own siblings), so quick tests may count one another as replicates but never a real run. **Amended 2026-10-03 (the cleanup plan's §2 rows 4 and 6, ROW-4/6; not a D-D row):** the `BEHAVIOR` route is retired. A resume of a node `RETRAIN_FROM` covers is still refused, and the refusal now says to set `RETRAIN_FROM = ""` and re-run sections 2-3; such a node, trained here although the trunk run certifies it, is then resumed in place like any other, with `TRUNK_FROM` set to the value the run's `trunk_run.json` records (decision 4 (a), which the maintainer took on 2026-10-02 as a run-level record the resolve cell writes and the RESUME cell and the recipe read), and the chain loop judges it before it consults the trunk (decision 6 (b)). The RESUME cell refuses, before it trains, a node the run holds as an `ancestors/` record (that record stays the node in the run, so the resume could never be judged into its bundle) and, once the run holds such a record, a resume under another trunk than the file names or with a file it cannot read; a run without the file keeps the recipe's manual route. ROW-4/6 landed as #591 on 2026-10-03. |
| D-D17 | Taken 2026-09-26 (the cleanup plan's decision 1, widened by the maintainer to retire the single-job Vertex AI route and GCS upload as well): Stable-Baselines3 (PPO and SAC) is the only training, evaluation and evidence backend until every species' stage chain and behavior is certified. Retired in the two PRs of `docs/CLEANUP_PLAN_2026_09.md` §4: PR-A removes the Ray Tune sweep, the Vertex AI hyperparameter-tuning sweep (`environments/shared/scripts/sweep/`, `configs/*/sweep_*.json`, `configs/quality_scoring.toml`, `cloudml-hypertune` and its report in `train_base`), the mjlab scaffold, the single-job Vertex AI route (`scripts/setup_vertex_ai.sh`, the `Dockerfile`, `.dockerignore`, `website/docs/training/vertex-ai.md`) and GCS artifact upload (`curriculum --gcs-bucket` / `--gcs-project`, the helpers they reach and the `[gcp]` extra); PR-B removes the JAX/MJX trainer, evaluator, stage writer and notebook, the `[jax]` / `[jax-cpu]` extras, the `[jax]` stage tables and the `test-jax-cpu` job. Where each half stays recoverable (the archive points, the cleanup plan's decision 2) is not part of this decision and is still open; it is settled before PR-A merges, and its outcome, with any tag SHAs, is appended to this row. No plant, task, gate, hyperparameter, stage-config, recipe, behavior-identity or recovery digest moves: trex, velociraptor, brachiosaurus and dibothrosuchus keep their dual-backend declaration and a frozen MJX interface core whose tokens their policy-interface digests hash; it is never edited, and each of those species drops it only by declaring itself SB3-only inside its next deliberate `policy_interface_revision` bump. Supersedes A6, narrows A7 to "SB3 is the evidence backend", and amends D-A5, D-A12, D-A15, D-B1, D-B12, D-B13, D-C3, D-C4, D-C7, D-C16, D-D11 and G1; the PR that carries out each change appends the amendment to its row. Adding any backend back is a new decision; its entry conditions are D-B13, invariant 9, the drift list PR-B records in the CHANGELOG, and a CI job that runs one real training trial. **Amended 2026-09-26:** PR-A is split into two PRs under the same D-D17, each with its own review: PR-A deletes the sweeps and mjlab and stays pure deletion; PR-A2 removes the single-job Vertex AI route and GCS upload, with an end-to-end test of the command-line curriculum path, so the edits to live SB3 code (`cli.py`, `config.py`, `train_curriculum`, `reporting/csv_output.py`) get their own review (the cleanup plan lands PR-A2 after PR-A). **Archive points settled 2026-09-27** (the cleanup plan's decision 2, option (c)): no archive tags; the merge commits keep the retired code reachable in git history, and the last release before the retirement is tagged `0.3.8` (a lightweight tag on `afad625`, published as a GitHub pre-release on 2026-09-27). As carried out by PR-A: Ray Tune, the Vertex AI tuning sweeps (with `cloudml-hypertune` and its report in `train_base`) and the mjlab scaffold are removed, 42 files; `_report_hpt_metrics` keeps its name and still writes `metrics.json`; D-A12, D-A15, D-B1, D-B12 and D-D11 are amended and A6 is superseded; no digest moves. PR-A landed as #564 on 2026-09-27. As carried out by PR-A2 (2026-09-27): the single-job Vertex AI route (`scripts/setup_vertex_ai.sh`, the `Dockerfile`, `.dockerignore` and `website/docs/training/vertex-ai.md`, 4 files and 957 lines) and GCS artifact upload (`config.upload_curriculum_artifacts` and `_upload_to_gcs`, the `gs://` branch of `reporting.write_results_csv`, which now refuses such a path, and the `[gcp]` extra) are removed. The `curriculum --gcs-bucket` / `--gcs-project` flags and `train_curriculum`'s `gcs_bucket` / `gcs_project` parameters go under this decision, so a command-line caller gets argparse's usage error (exit 2, "unrecognized arguments") and a Python caller of `train_curriculum` a `TypeError`. An end-to-end test of the command-line curriculum path (`test_curriculum_cli_end_to_end.py`) lands first and passes before and after the removal. `tb_sync.py` and the `/gcs/` mount detection stay; no digest moves. PR-A2 landed as #565 on 2026-09-28. As carried out by PR-B (2026-09-28): the JAX/MJX trainer, evaluator, stage writer (`save_jax_stage_artifacts`) and notebook (`jax_training.ipynb`), `validate_mjx_environment_plant`, the `[jax]` / `[jax-cpu]` extras, the `[jax]` and `[jax.policy_kwargs]` tables of the 12 stage TOMLs and the `test-jax-cpu` job are removed (30 whole files and 15,665 lines), and `load_stage_config` now refuses a `[jax]` table. The 549-line frozen MJX interface core stays, excluded from ruff, and `test_plant_contract_frozen_mjx.py` pins its seven anchor token digests, trex's two low-pass digests and each registration's key set; the `foot_contact_*` keys and parameters, the readers of recorded JAX runs and the `jax-mjx` catalog rows stay. A7 is narrowed and D-A5, D-B13, D-C3, D-C4, D-C7, D-C16 and G1 are amended; the drift list is in PR-B's CHANGELOG entry; no digest moves. PR-B landed as #566 on 2026-09-28. |
| D-D18 | Taken 2026-09-26 (the cleanup plan's decision 8, option (a), through its CU-1): CI type-checks the tree with Stable-Baselines3 and torch installed. The SB3 job pins SB3 as the SB3 notebook pins it, beside its CPU `torch==2.13.0` wheel, and runs `mypy environments/ --ignore-missing-imports`, which must report no errors; ruff and mypy are each pinned to one version for CI and pre-commit. The lint job keeps its SB3-free mypy run as the first, fast signal. As carried out by CU-1: `stable-baselines3[extra]==2.9.0`; ruff 0.16.9 is pinned in the lint job, `.pre-commit-config.yaml` and `pyproject.toml`'s `dev` extra, and mypy 2.3.1 in those three places and in the SB3 job's mypy step; the pre-commit ruff hooks cover `environments/` only, as CI's ruff does; `environments/shared/tests/test_ci_tool_pins.py` keeps these pins, and the SB3 pin, in agreement. Unblocks PR-10 (torch-facing code). Landed as #561 on 2026-09-26. |
| D-D19 | Taken 2026-09-26 (the cleanup plan's decision 3): the CHANGELOG release is cut before PR-A, so the retirement's Removed and Migration entries head a short section. The three undated `[Unreleased]` headings (v0.3.8, v0.3.2, v0.3.0) are dated, an empty `[Unreleased]` opens above them, `pyproject.toml`'s version moves from `0.3.8.dev0` to `0.3.9.dev0`, and the maintainer tags `v0.3.8` on the commit that dates its section; v0.4.0 stays ROADMAP's "Into the Wild" milestone. The version is provenance (`stage_config.json`'s `library_version`, the HPT metrics payload, `provenance.json`'s `dependency_versions`), so a resume after the bump records environment drift, as a new repository commit already does. As carried out by the release PR, with the details the maintainer chose on 2026-09-26: its first commit, the one the maintainer tags `v0.3.8`, sets `version = "0.3.8"` (so an install from the tag reports 0.3.8) and dates `[0.3.8] - 2026-09-27`, `[0.3.2] - 2026-07-21` and `[0.3.0] - 2026-07-09`, each older section by the UTC date of the commit that opened the heading above it and moved the version on (`db153b2`, `3f176ce`), with the titles kept after the date; its second commit opens a bare `## [Unreleased]` above them and sets `0.3.9.dev0`. The digest-snapshot harness output is byte-identical on the base `89d814a`, the tagged commit and the head (848 lines, 0 errors). Landed as #563 on 2026-09-27 (merged as `04d107a`); the maintainer tagged `afad625` as `0.3.8`, a lightweight tag without the `v` prefix named above, and published it as a GitHub pre-release on 2026-09-27. |
| D-D20 | Taken 2026-09-26 (the cleanup plan's decision 7, option (a); carried out by its CU-3): a Colab reclaim may not leave the final pair or a handoff pair truncated, nor a new file beside the previous pair's other half that the pair's checks accept. On a Drive/GCS mount `train()` saves the final pair and the best and robust-best handoff pairs to local scratch and publishes them into `models/`, as it already did the periodic pairs; because these names are reused, the destination file published last is removed before anything is published, and before the final pair is saved an empty placeholder takes the final zip's place and the final sidecar is removed. So a reclaim part-way leaves at worst a handoff sidecar without its zip (which `select_handoff_checkpoint` skips) or a final zip that `checkpoint_pair_problem` rejects: the placeholder or the new zip without its sidecar or, in the instant before the previous final sidecar is removed, the empty placeholder beside it (it reports each, as it reported a final pair a save straight to the mount cut short, so D-D16's RESUME and chain-loop rules, the early-stop refusal included, apply unchanged). Off a mount the pairs are written in place, as before. File names and bytes are unchanged, so no digest moves. As carried out by CU-3: `curriculum.publish_staged_pair` does the publishing, `RobustBestModelCallback` takes a `staging_dir` and `SaveVecNormalizeCallback` a `publish_dir`, and `_save_final_and_sync_tb` places the placeholder and stages the final pair. Unblocks PR-13, whose evidence writer follows it by writing through `file_io`'s atomic helpers. Landed as #562 on 2026-09-26. |
| D-D21 | Taken 2026-09-27 (the cleanup plan's decision 20, after the release cut of D-D19): 0.3.9 is the clean, refactored base release that the consolidation work builds on; 0.3.8 stays the pre-cleanup release (the maintainer's `0.3.8` tag on `afad625`) and is not re-opened. 0.3.9 is cut once its gate has landed: PR-A, PR-A2, PR-B, CU-2, CU-4, CU-5, CU-7, CU-8, CU-9, CU-11, CU-12, CU-14 and CU-16 (thirteen PRs); CU-4 and CU-14 still take the cleanup plan's open decisions 9 and 10 when they open. Deferred, not dropped, each at the point the cleanup plan records for it: CU-6 (after session 6's resume and CU-4; amends D-D7), CU-13 (before PR-11; amends D-D5), CU-10 (lowest priority), CU-15 (optional; after PR-A), CU-17 (last, before PR-15), and consolidation PR-8..PR-15, which build on 0.3.9 (PR-8 needs CU-11, PR-9 needs CU-8 and CU-12, all in the gate). Session 6's resume finished on 2026-09-28, so CU-6 waits only for CU-4. **Amended 2026-09-29:** the maintainer accepted splitting four gate PRs into parts: CU-7 into CU-7a (dead code and import cost) and CU-7b (retired-backend wording, with CU-16's stage-TOML comments folded in), CU-7's sha256 regex and validators moving to CU-8c; CU-8 into CU-8a (one task-fingerprint derivation), CU-8b (`policy_loading` owns the SB3 import and the sidecar resolver) and CU-8c (one reader, one root, one regex, one set of validators); CU-14 into CU-14b (the plant-identity cache, the cleanup plan's decision 10 (c)), first, and CU-14a (the workflow YAML); and CU-16 into CU-16a (docs text) and CU-16b (orphan assets). The gate's names cover all their parts, so the gate itself is unchanged and 0.3.9 is cut once every part has landed. *Corrected 2026-09-29: CU-6 also follows CU-5, CU-8b and the notebook PR for the cleanup plan's decisions 4 and 6 (the cleanup plan's §3.2, CU-6 row), not CU-4 alone.* As carried out by the 0.3.9 cut PR (2026-10-02), once the gate had landed with CU-8c (#583, 2026-10-01), with the details the maintainer chose on 2026-10-01 (the title kept without its "(v0.3.9)" suffix, the date rule and `0.4.0.dev0` as the next development version): its first commit, the one the maintainer tags, sets `version = "0.3.9"` and dates the section `## [0.3.9] - 2026-10-02 — Backend Retirement & Cleanup`, by that commit's UTC author date (re-dated if it is rebuilt on a later UTC day); its second opens a bare `## [Unreleased]` above it, as the 0.3.8 cut did, and sets `0.4.0.dev0`. The digest-snapshot harness reports the golden current (932 lines, 0 errors) on the base, the tagged commit and the head. From `## [0.3.9]` to the end, `CHANGELOG.md` has sha256 `88c579c35fb4ee3a8a29ce1807e0e98210f5861a4e0fd3e12e8800a43e398dca` at the tagged commit and the head; from `## [0.3.8]` it is unchanged (`c750a3fd…773b`). The cut PR merges with a merge commit, so the tagged commit keeps its SHA on `main`, and never through "Update branch": if `main` moves first, both commits are rebuilt on it. Landed as #584 on 2026-10-02 (merged as `fb9b2a7`); the maintainer tagged `20ab100` as `0.3.9`, a lightweight tag, and published it as a GitHub pre-release on 2026-10-02. **Amended 2026-10-02 (the maintainer):** the deferred PRs land before consolidation PR-8 and the gait plan's code PRs. With the maintainer's answers of that day they go one at a time, in the order CU-10a, CU-13, CU-15 (reduced scope), CU-10b, ROW-4/6 (the notebook PR for the cleanup plan's decisions 4 and 6, which the maintainer took the same day), CU-6 and CU-17, then PR-8. CU-10 is split into CU-10a (the curriculum horizon fix chosen on 2026-09-27) and CU-10b (the stage body on a private helper shared with `train()`, with `eval_env_seed`, then D-D11's alignment for the curriculum as its own last commit), and CU-10's optional split of `train_base.py` into modules is not taken (the cleanup plan's §3.1 item 5). *2026-10-03 (the cleanup plan's §3.1 item 5): of the deferred PRs, CU-10a, CU-13, CU-15, CU-10b and ROW-4/6 have landed (#587 to #591), and CU-6 is carried out; CU-17 is the last of them, before PR-8.* |
| D-D22 | Taken 2026-09-29 (the cleanup plan's decision 16, option (a), with the full run on pull requests): the output of the digest-snapshot harness (`environments/shared/harnesses/digest_snapshot.py`, every plant, policy, stage, recovery and behavior digest, one per line) is committed as a golden, and CI checks it: the full harness run on pull requests, not `--skip-behaviors`, in the plant-contract job, so a change that moves only behavior identities is caught on its own pull request. A PR that deliberately moves a digest updates the golden in its own diff, where the move is reviewed; every other PR leaves it unchanged. The check is built by its own PR, not by CU-4, before CU-7b, CU-8a, CU-12 and CU-13, which wait for it. As carried out by the ROW-16 PR: the full run at `a919985` (848 lines, 0 errors) is committed as `configs/digest_snapshot.generated.txt`, and the plant-contract job's step "Verify the digest snapshot golden" runs `python -m environments.shared.harnesses.digest_snapshot --block-optional-backends --check` on every event; a mismatch names every moved line and prints the regeneration command, the same module with `--write`. Unblocks CU-7b, CU-8a, CU-12 and CU-13. Landed as #571 on 2026-09-29. |
| D-D23 | Taken 2026-10-06 (the maintainer chose it on that day, in the session that commissioned the stance work; the gait plan's GQ-6 (a) and GQ-7 (a), taken for stance only; GQ-10 and GQ-12 are not answered here): a stance is certified on floor truth, episode by episode. The gate kind `stance_quality/v2` is registered beside `stance_quality/v1`, which stays for historical verdicts, under `GATE_SCHEMA_VERSION` 1. Each episode of the 40-episode certification panel (seeds 3042 + i, D-B17) is clean when it reaches the horizon and every declared criterion holds on a finite metric (an unmeasured metric fails it). The metrics come from the measurement library `environments/shared/gait/` (`MEASUREMENT_VERSION = "floor-truth/v1"`): a leg is down when the floor's normal force on its geoms exceeds 0.1 N on at least half the substeps of a control step, decoded exactly as `mj_contactForce` reports it, through `_substep_probe_hook`, on evaluation envs only, with an explicit support-geom registry for all six species. The panel passes when it holds `min_eval_episodes`, the exact one-sided Clopper-Pearson bound on the clean count clears `min_clean_stance_lcb` (0.80 admits 37/40 and refuses 36/40) and the declared panel rails hold. Beyond the plan's §4.1 list, the kind takes the audit's extensions: the required set adds `max_actuator_saturation_fraction`, `max_settle_airborne_substeps` and `max_settle_peak_floor_force_bw` (the settle window is scored after a 0.1 s spawn grace, because the reset pop is a plant property) to `min_eval_episodes`, `min_clean_stance_lcb`, `settle_steps`, `min_all_feet_support`, `max_touchdown_rate`, `max_window_displacement_m` and `min_foot_load_share`; the optional criteria, applied only when declared, are foot-on-foot contact, phantom touch support, non-foot floor load, the windowed load share, the settle stance-width change, support-geom duty and coverage, sole tilt and tilt in excess of the keyframe's, sole corner lift and loaded sole contacts, with the absolute reward rail, a full-horizon floor and two statue-relative ratios (the panel's mean reward, and each foot's load share, against a zero-action panel rolled through the same recorder on the same seeds in the same report). A changed measurement constant or registry entry is a new `MEASUREMENT_VERSION`, and a report measured under another is refused. The certificate is the post-stage stance report on the handoff pair: the judge refuses a report scored by other code, under other thresholds or another measurement, or for another checkpoint or task than the stage recorded, and re-derives its verdict; publication re-derives it from `stance_panel_selected.csv` bound to the certified pair and the stage's task; the in-training manager refuses the kind with a log (the in-training screen is not built), and the command-line curriculum judges a v2 node after training. Registering moves no digest, and no stage adopts the kind with this decision: each adoption is its own decision and a gate revision, which moves that stage's `gate_sha256` and `stage_config_view_sha256` lines and makes reuse rule 7 refuse its v1 verdicts. The stance subset of the plan's PR-G1..PR-G3 comes ahead of the locomotion library and gate (amends the plan's §6 order for stance). Validated on 2026-10-06: every zero-action statue is clean on 40/40 episodes on the current plants, and every audited stance checkpoint's panel fails (`docs/investigations/STANCE_HACK_AUDIT_2026_10.md` §5). Unblocks a stance stage's adoption of the kind. |
| D-D24 | Taken 2026-10-06 (the maintainer chose it on that day, in the session that commissioned the stance work; it overtakes the gait plan's GQ-12 and PR-G8 row, which kept the trex stance on `stance_quality/v1` until a trex stance retrain, and its §5.4 trex locomotion `TRUNK_FROM = "20260914_123816"`): the T. rex plant revision and the T. rex stance's adoption of `stance_quality/v2` land now, together, retiring every T. rex certificate. Physics r7 → r8: the hip-roll servos `r/l_hip_roll_act` go kp 150 → 600 and forcerange ±120 → ±480 N·m (`configs/plant_versions.toml` note 13; the policy interface stays r13 and the visual layer r4), because the soft servos (2 × (kp + 40) = 380 against the settled statue's m·g·h of 704.4 N·m/rad) left the zero-action statue on a rolled pad in 4 of its 40 panel episodes, so on r7 the flatness bars separated the statue from the audited props only by sub-millimetre and sub-degree margins (the statue's corner lift reached 5.54 mm against the nearest hack episode's 5.06, its tilt 3.42° against 2.11°); the r8 statue's worst, 2.27 mm and 1.15°, lets the bars tighten to 4 mm and 2°, under which the r7 statue is clean on only 36/40 (bound 0.786). The stance task gains three `TRexEnv` kwargs, inert at their defaults and set in `configs/trex/stance.toml` (recovery inherits them): the neck term centred on the statue's settled pose, a pad-flatness term (0.15 at 3°) and a stance-width term (0.1 at 0.05 m), sized so that no constant deviation in a hack direction beats the statue and each audited hack loses more than it gained (a policy that corrects its spawn still gains on the statue, in the intended direction). The stance declares `stance_quality/v2` with bars set on the r8 statue's 40-episode panel: the bound ≥ 0.80 at `settle_steps` 200, every required bar, the pad bars `max_sole_tilt_deg` 2.0, `max_sole_corner_lift_m` 0.004 and `min_sole_contacts` 1.5, four guards (windowed load share, foot-on-foot, phantom support, non-foot load) and the rails `min_full_horizon_fraction` 0.95, `min_avg_reward` 2260 and `min_avg_reward_statue_ratio` 0.60. The revision leaves `certification_seeds` at 2; GQ-12's third seed is not part of this decision and stays the maintainer's to schedule. Validated on 2026-10-06 (`docs/investigations/STANCE_HACK_AUDIT_2026_10.md` §7): the r8 statue is clean on 40/40 and on 80/80 more seeds, and `20260914_123816`, `20260920_010912`, `20260930_024929` and `20261001_225601`, rolled on the r7 plant they load on, on 0/40 each. Consequences accepted: the plant contract refuses every T. rex checkpoint and none can be widened; the seed-42 and seed-44 stances, the seed-44 recovery, both walkers and the hunt retrain from a fresh stance; the recovery gate resolution and `recovery_evaluation.py`'s `CALIBRATED_*` constants are re-measured on the next certified stance; every T. rex task digest and the stance gate digest move (39 golden lines). Unblocks the T. rex stance retrain under v2, and with it every T. rex chain. |
| D-D25 | Taken 2026-10-06 (the maintainer chose it on that day, in the session that commissioned the stance work; it overtakes KNOWN_ISSUES' "execution is deferred until the T-Rex clears stages 1–3" for the velociraptor plant review, and the gait plan's §9, which kept the velociraptor foot-sensor repair out of that plan): the velociraptor plant revision, its SB3-only exit and the velociraptor stance's adoption of `stance_quality/v2` land now, together, retiring every velociraptor certificate. Physics r2 → r3, policy interface r10 → r11, visual r3 → r4 (`configs/plant_versions.toml` note 14): every leg joint's `springref` at the standing pose (unset, the springs pulled toward qpos 0 and the servos spent 10.4 / 15.1 / 34.5 N·m cancelling them); a keyframe whose ankle and toes flatten the foot (digit III sat 44.6 mm inside the floor, and every reset settled that deep) while hip 38° and knee −50° keep the centre of mass over it (an 18° hip flattens the toes too but puts the support 7.6 cm ahead of the centre of mass, and the statue falls backward in 0.66 s); a home ctrl carrying the measured gravity preload, without which springs anchored at the stance drop the statue onto its body within 1.4 s; and metatarsus and digit-IV touch sensors appended and summed per foot (digit III alone read 55% of the floor load; the sum reads 1.000). The review's "re-size the leg actuators at the same time" (`docs/reviews/VELOCIRAPTOR_PLANT_REVIEW.md` §5 R2, KNOWN_ISSUES) is refuted by measurement: with the preload the nominal servos, kp, forcerange and ctrlrange unchanged, hold the pose at 0.2–5.8% of forcerange, on 40/40 episodes even with every leg spring deleted. The summed foot cannot be mirrored by the frozen MJX registration, which D-D17 never edits, so the velociraptor declares itself SB3-only inside this policy-interface revision, as KNOWN_ISSUES and the cleanup plan's §4.1 prescribe, and `environments/velociraptor/mjx_config.py` is deleted; §4.1's batched checklist (one base `_scale_action` and `_get_obs`, the midpoint mapping, the reset height channel) is not part of this decision. The stance task gains five reward terms through eight `RaptorEnv` kwargs with the `TRexEnv` names, inert at their defaults and set in `configs/velociraptor/stage1_balance.toml` (support-conditioned alive 0.2, bilateral support 0.5, leg home pose 0.5, action saturation 0.5, action jerk 1.0), sized so that the statue stays the reward optimum and the support, leg-pose and saturation terms each alone cost the audited stance more than it gained over the statue. The stance declares `stance_quality/v2` with bars set on the r3 statue's 40-episode panel: the bound ≥ 0.80 at `settle_steps` 100, every required bar, the support-geom duty and coverage bars of a foot with no box sole (0.50 and 0.80), the settle width change (0.05 m), the load share against the statue's (0.80), four guards and the rails `min_full_horizon_fraction` 0.95, `min_avg_reward` 1710 and `min_avg_reward_statue_ratio` 0.60; the foot bars and the settle peak (2.0 BW) sit mid-gap between the statue and the audited hacks, because bars at the statue's edge refused the statue with small command jitter, whose lightly loaded digit IV unloads on part of the window, a loss no reward term sees. Validated on 2026-10-06 (`docs/investigations/STANCE_HACK_AUDIT_2026_10.md` §8): the r3 statue is clean on 40/40, on 80/80 more seeds, on 40/40 of a third block and on 200/200 fresh seeds, the statue with N(0, σ) command jitter on 40/40 at every σ from 0.02 to 0.05, and the audited stance `20260922_125248` (robust_best and final), rolled on the r2 plant it loads on, on 0/40 each. Consequences accepted: the plant contract refuses every velociraptor checkpoint and none can be widened; the certified stance and walker of `20260922_125248`, and `20260929_112244`, which reuses that stance, are stranded, so GQ-11's velociraptor re-panel and PR-G5's calibration on that walker can only run on the r2 plant, in a checkout of an earlier commit; every velociraptor task digest and the stance gate digest move, and the frozen MJX core keeps three species. Unblocks a velociraptor stance retrain under v2 on a plant that stands on its own servos, and with it every velociraptor chain. |
| D-D26 | Taken 2026-10-07 (the maintainer chose it on that day, answering the open questions of the compsognathus standing review, and took its toe armature the same day when the implementation found the MTP servo ringing; it overtakes the gait plan's GQ-11 (a) for the compsognathus walker `20260921_203149`, which the plant revision strands, and takes that plan's §5.4 optional compsognathus stance row with the species' own kwargs rather than §5.3's kit): the anatomical compsognathus plant revision, a new policy interface for it and the compsognathus stance's adoption of `stance_quality/v2` land now, in one PR, retiring every anatomical compsognathus certificate; the robot keeps every digest. Physics r1 → r2, policy interface r2 → r3, visual r1 → r2 (`configs/plant_versions.toml` note 15): the 178 g tail spread at one density over its tapered meshes (tail_1..tail_4 0.080 / 0.055 / 0.028 / 0.015 → 0.1173 / 0.0461 / 0.0125 / 0.0021 kg; the r1 tip, at 6.2 g/cm³, was denser than bone), which moves the settled statue's centre of mass from 19.6 to 29.3 mm ahead of the pad's rear edge (0.25 → 0.38 of the foot); the metatarsus collision capsule ending 4 mm short of the MTP joint (settled clearance 1.07 → 4.81 mm: on r1 a 1 mm sink of the sole touched it with 0 N on it, 48 of 48 first touches, and the terminating contact ended upright, flat-footed episodes as falls); digit II medial on both feet (r1 built the right foot as a left foot); the MTP joint's armature 5e-5 → 2e-4 kg·m² (`toe_armature_kg_m2`), because at the default the toe servo rang at 68 Hz (ω·dt 0.85 at the 2 ms step) and on the other changes sustained a numerical 19–21 Hz heel chatter on 15 of 40 zero-action statues that failed 2 of `validate_models`' 10 seeded holds (at 2e-4: no chatter, every hold passes, `preflight_training_v1.json` `passed: true`); and the species' env `CompsognathusBiologicalEnv`, whose plant-contract mode `home-keyframe-residual-softcubic/v1` shapes the ten leg residuals to b = 0.1a + 0.9a³ before the unchanged home-keyframe span (action zero is still the home ctrl and ±1 the ctrlrange ends), because every leg servo saturates at 0.05 rad of position error and the linear residual put the whole knee or ankle holding torque inside 0.007–0.015 of action, so under PPO's initial exploration noise (σ 0.135, `log_std_init` −2.0) the r1 statue fell on 20 of 20 episodes (median 12 steps) and PPO never sampled the quiet stance; on r2 it reaches the horizon on 40/40 at σ 0.10, 151/160 at σ 0.135 (four seed blocks; 30/40 without the armature) and 0/40 at σ 0.20, a cliff that constrains `log_std_init` and any entropy schedule. The proportions stay but for the tail's mass (the fidelity pass is deferred), and the MTP servo keeps its 0.6 N·m cap, now the generator parameter `toe_forcerange_nm` (the maintainer first chose 0.4 N·m and revised it the same day): the cap holds one body weight on the digit-III tip (0.594 N·m), so the single-support tiptoe of both certified r1 stances stays physically possible on r2 (the `20260921_203149` march, replayed on the r2 plant through its own linear interface, still survives 20 of 20 episodes on that tip), and the reward and the gate are what refuse it; the cut waits on a measured walker toe-torque budget after training. The stance task gains ten reward terms through twenty `CompsognathusBiologicalEnv` kwargs, inert at their defaults and set in `configs/compsognathus/stance.toml` (recovery inherits them through `extends`, and its calibration is re-measured on r2: multiple 1.3, r1 1.32): bilateral support and a support-conditioned share of the alive bonus on each foot's substep-mean load (the legacy alive, posture and height terms keep their support gate on the substep minimum, which prices a two-foot hop; a one-foot unload leaves the other foot's minimum above it, so only the bilateral and coverage terms price a one-foot chatter), sole flatness and stance width paid only on a loaded foot, the width around the animal's own settled width (the keyframe reference paid both trex r8 seeds to re-seat their feet in the unscored settle), the home leg pose, support-geom coverage on the gate's substep rule (a digit counting, by this term's own rule, only through its contacts beyond the capsule's centre, away from the MTP joint), floor impact and airborne substeps from the first step, and action saturation and jerk on the raw command (this species has no action filter). The statue stays the reward optimum (4570.4 per episode, 1571.3 of the 1580 the terms can pay, short only where the settle seats the digit tips last); the certified marches score 0.61 of their own statue (0.94 under the old reward) and, on a 20-episode re-score, rolled feet 0.85–0.89, legs swept back 0.89, a hop started after the settle 0.65–0.91 and settle stomps 50–198 below it. The shared step loop, which keeps every substep's foot force beside the minimum since D-D27, gains a read-only per-substep hook, and every species stays bit-identical. The stance declares `stance_quality/v2` with bars set on the r2 statue's 40-episode panel: the bound ≥ 0.80 at `settle_steps` 200, every required bar, the pad bars (tilt 2.0°, corner lift 3 mm, 1.5 contacts), the support-geom duty and coverage bars (0.50 and 0.80; the pad and digits II–IV), the settle width change (0.011 m, about three times the statue's worst, the trex r8 runs' lesson), settle bars looser than the trex's and the velociraptor's (15 airborne substeps, 3.0 BW), because noise bounces this light plant and the reward prices both from the first step, the load-share ratio against the statue's (0.80), four guards and the rails `min_full_horizon_fraction` 0.95, `min_avg_reward` 2740 (0.60 × 4570.4, rounded to the nearest 10) and `min_avg_reward_statue_ratio` 0.60. It also declares the window hop pair, two optional `stance_quality/v2` criteria added for this plant and applied only where declared (`max_window_airborne_substeps` 40, `max_window_peak_floor_force_bw` 2.0 BW): this plant's hop flights last 1–3 of a step's ten substeps, so a two-foot hop started after the settle reads both legs down on every step and a statue on every step-level bar, and the pair refuses it (0/410 square-wave hops over three seed blocks), though not a sine-driven bounce that lands under 2.0 BW, a foot lifted for less than half a step or a fast weight shuttle, which certify (KNOWN_ISSUES; until a periodicity criterion is measured, a trained stance's lift-offs are checked by hand). Validated on 2026-10-07 (`docs/investigations/STANCE_HACK_AUDIT_2026_10.md` §11): the r2 statue is clean on 40/40 on each of three seed blocks, with N(0, σ) command jitter on 20/20 at σ 0.005 and 0.01, on 40/40 at 0.02 and on 120/120 at 0.03 (at 0.05 the window peak bar refuses 64 of 120: that noise lands the plant as hard as the softest square-wave hop), and spawned yawed ±45° and ±90°; two short PPO runs (200k steps at `log_std_init` −2.0) are clean on 40/40 with their deterministic means, which shows only that the mean does not drift to a march or a hop early; the certified r1 marches `20260921_203149` and `20261001_225856`, rolled on the r1 plant they load on, are clean on 0/20 each; and the committed stance passes the post-stage pipeline (report, judge, publication and backfill) with the zero command as a scripted checkpoint. Consequences accepted: the plant contract refuses every anatomical compsognathus checkpoint and none can be widened; the certified stance and walker of `20260921_203149` and the stance of `20261001_225856` are stranded, so GQ-11's re-panel of the compsognathus walker and PR-G5's calibration on it can only run on the r1 plant, in a checkout of an earlier commit; every compsognathus task digest, the stance gate digest and the recovery calibration move (48 golden lines); no r2 policy has trained, and the maintainer trains at least two seeds from `main` after landing. Unblocks a compsognathus stance retrain under v2 on a plant PPO can explore around, and with it every compsognathus chain. |
| D-D27 | Taken 2026-10-07 (the maintainer asked that day for the follow-up fixes to the T. rex stance after the review of its first two physics-r8 stance runs, `20261006_185343` seed 42 and `20261006_185704` seed 44, both refused by `stance_quality/v2`; it amends D-D24's stance task and gate block, not its plant): the T. rex stance task, recipe and gate are revised on the unchanged plant (physics r8, policy interface r13, visual r4), and no T. rex stance is certified. Both runs trained D-D24's config byte for byte and fail v2 as recorded and as re-derived: seed 42 is clean on 0/40, a 12.5 Hz two-foot micro-hop on raw ±1 commands at 0.80 of the statue's reward; seed 44 on 13/40 (bound 0.204) and on 8/40 of the fresh seeds 7042–7081: it stands like the statue after the settle, but stomps or hops its feet back to the keyframe width in the settle, hops for the whole episode on about 4% of resets and stands only near its training heading. The review traced the failures to the task and the recipe: a keyframe-referenced width term that paid for re-seating the spawn, settle impacts and airborne substeps priced by nothing, action penalties priced after the 10 Hz command filter, foot terms paid on unloaded feet, and a start at std 1.0 that put half of all early stochastic steps in the air, the recipe's own prediction for which failed in both runs. The stance task gains seven `TRexEnv` kwargs, inert at their defaults and set in `configs/trex/stance.toml` (recovery inherits them and captures the settled width at step 140, before its first push): the width term centred on the animal's own width at the end of the settle (`stance_width_reference = "settled"`, `stance_width_settle_steps = 200`), the flatness and width terms paid only on a foot carrying 0.20 BW (`foot_terms_min_support_force = 168.0`), a floor-impact term (2.0 per BW above 1.4 BW) and an airborne-substep term (1.0) over the whole episode, settle included, read from the per-substep foot touch `BaseDinoEnv.step` now keeps, and smoothness, jerk and saturation priced on the policy's own command (`action_penalty_source = "raw"`). The stance recipe starts at `log_std_init = -1.5` (std 0.22), the only one of the three measured starts at which the new terms cost a fresh policy under half the alive bonus, and declares an opt-in hop watch that stops a run whose evaluation unsupported duty is still above 0.05 after 7M steps (`curriculum/hop_watch.py`; its keys enter no digest). The v2 block declares `max_settle_stance_width_change_m` 0.08, two new optional per-episode criteria, `max_settle_touchdowns` 2 and `max_episode_yaw_change_deg` 25, and a new optional panel rail, `max_hop_or_fall_episodes` 1 (the episodes that end early or fail support, touchdowns, drift or saturation), each undeclared, and so unprojected, for every other stage; `min_sole_contacts` stays 1.5, and `min_avg_reward` moves 2260 → 2250 with the statue (3756.8 under the new task; recovery's pushed statue 1329.1). Spawn-yaw randomisation and a heading-invariant observation, each a policy-interface revision, are **not** taken: whether stance certification should require heading robustness stays open, and a report-only spawn-yaw probe (`stance_probe_spawn_yaw_deg`, written to `stance_heading_probe.{txt,json}`) measures it. Validated on 2026-10-07 (`docs/investigations/STANCE_HACK_AUDIT_2026_10.md` §9): the statue is clean on 40/40, with no hop-or-fall episode, on 3042–3081 and on 7042–7081; re-scored exactly on the recorded panels, seed 42 falls from 0.80 to 0.28 of the statue and seed 44's margin over the statue by about 23 per episode on 3042–3081 and 5042–5081 and by 56 on 7042–7081, where its three hop episodes lose about twice what they lost before (18 on its 36 standing episodes), so the v2 settle bars, not the reward, remain what refuse its stomp; short PPO runs at the new start hold the stochastic policy's unsupported duty under 0.05 from the first update (0.52–0.54 at std 1.0). Consequences accepted: every T. rex `task_sha256`, the stance `gate_sha256` and the stance and recovery `hyperparameters_sha256.PPO` move (36 golden lines, all T. rex), so a T. rex run in flight on D-D24's task cannot resume across it; no certificate is stranded, because no T. rex stance has certified on r8. Unblocks the next T. rex stance retrain: at least two seeds, each judged on 3042–3081 and re-judged on 7042–7081. **Amended 2026-10-07 (the maintainer's heading decision of that day; it takes its own id in the PR that carries it out):** the question D-D27 left open is answered: heading-free observations for the T. rex and the compsognathus (body orientation without world yaw, linear velocity in the yaw frame, the prey direction relative to the body, and every other world-heading channel), in one cross-species PR after the compsognathus revision lands and before either species trains, where the spawn-yaw probe becomes a gate check; each species' policy-interface revision moves. The velociraptor follows only after the maintainer's two current velociraptor runs are reviewed (`docs/KNOWN_ISSUES.md`). |
| D-D28 | Taken 2026-10-07 (the maintainer, after the review of the T. rex physics-r8 seed-44 stance `20261006_185704` found its left foot standing on the front edge of its pad, which no other `stance_quality/v2` bar sees: on the episodes every other bar passes, 13/40 on 3042–3081 and 8/40 on 7042–7081, that foot is still on its edge; in a gate-only PR, re-applicable to any checkpoint by re-rolling its panel, which lands after the compsognathus revision D-D26 and is rebased on it): `stance_quality/v2` gains an optional per-episode criterion, `max_sole_cop_fore_aft`, and the T. rex stance block declares it at 0.80. `min_sole_contacts` stays 1.5: that foot stands on 2.0–2.2 loaded points, and the statue reaches 2.086 out of sample. The metric (`StanceEpisodeMetrics.max_sole_cop_fore_aft`) is the worse box pad's window mean of its per-step \|fore-aft centre of pressure\| divided by the half-length (0 centred, 1 on the heel or toe edge), read off the force-weighted sole CoP the recorder already computes: per step, because the seed-42 stance rocks its pads heel to toe: their signed window means (−0.36 to −0.30 and 0.14 to 0.28 on 3042–3081) sit no further from the centre than the statue's (up to 0.36), while the per-step reading is 0.96–0.99; a window step with no loaded sole contact (none above `CONTACT_THRESHOLD_N`, as `min_sole_contacts` counts them) counts 1, so lifting the pad, or grazing the floor with it, cannot hide an edge stance; NaN on a foot without a box sole (the velociraptor's soleless foot; the brachiosaurus's ellipsoids, whose single contact sits where the authored pitch puts it). A declared bar must lie strictly inside (0, 1) (`CRITERION_BAR_RANGES`, checked at config load and by the reader), and the key is not a hop-or-fall key. The field is the first of `STANCE_METRIC_LATER_FIELDS`, the metric fields added after the panel CSV contract shipped: a report or panel CSV recorded before it reads it as unmeasured, so every earlier report keeps its verdict under a block that does not declare the key, and under a block that does such a panel is refused, never failed as unmeasured: publication, re-deriving from the CSV, refuses it by name (re-roll the panel); the judge and backfill refuse a report of that age because the thresholds it records lack the key, and by name when they match. The "reader/version migration" the decision names is carried out as a reader-only migration, the implementation's reading of it, for the maintainer to confirm: `MEASUREMENT_VERSION` stays `floor-truth/v1` and no measurement digest or report schema moves, since a new metric changes no adopted key's meaning, so a panel CSV written before the change and one written after carry the same stamps and differ only by the column; the recorder, the plant and every task are unchanged. Calibrated on 2026-10-07 and re-rolled on 2026-10-08 under the loaded-contact reading, which moved no number (`docs/investigations/STANCE_HACK_AUDIT_2026_10.md` §10): the statue reads at most 0.368 on five seed blocks and 0.480 on 5042–5081 (seed 5048, a fore-aft sway clean on every other bar), over 238 full-horizon episodes, and up to 0.745 under N(0, 0.05) command jitter on every step; the seed-44 left foot 0.850–1.000, over the bar on 140/140 episodes on four seed blocks; seed 42 0.96–0.99. The bar is 0.80 rather than 0.75: the D-D27 369k-step study checkpoint stands on a flat pad loaded forward to 0.88 (on the four 3042–3081 episodes 0.75 refuses, 2.9–3.9 loaded points and its CoP past 0.9 of the half-length on at most 0.5% of window steps), and 0.75 would fail its 3042–3081 panel (38/40 clean → 34/40, and 36/40 on 5042–5081) where 0.80 leaves 37/40. Consequences accepted: three digest-golden lines move, all T. rex stance (`gate_sha256`, `stage_config_view_sha256.PPO` and `.SAC`); a partial edge stance still passes, the bar being a window mean: an edge stance on up to (0.80 − s)/(1 − s) of the window beside a centred s (71% at the statue's 0.30), so a panel whose pads drift onto their front edge after about the first third of the window certifies, and a pad rolled about 1° at seed 44's toe-down pitch, on three loaded corners, reads about 0.76; no other species declares the key: the velociraptor has no box sole and the brachiosaurus's soles are ellipsoids, and the box-soled compsognathus, robot and dibothrosuchus are not on v2, so each gets a bar when it adopts v2, on its own statue. The compsognathus adopts v2 in D-D26, whose base has no such key, so its bar is measured on the D-D26 statue when D-D28 is rebased on it (the r1 statue's heel-side 0.45–0.47 does not carry over: the D-D26 statue reads about 0.14–0.18). **Amended 2026-10-08:** D-D28 landed before D-D26, and D-D26 measured the compsognathus bar on its r2 statue after it: 0.70, against the statue's 0.143–0.155 (heel side, 200 episodes on five seed blocks), and declared it in `configs/compsognathus/stance.toml` (`docs/investigations/STANCE_HACK_AUDIT_2026_10.md` §11). |
| D-D29 | Taken 2026-10-11 (the maintainer, with the heading decisions of that day, in `docs/HEADING_INVARIANCE_DESIGN_2026_10.md`'s "Scope of the symmetry": a physics revision of its own, landed before the first heading-free T. rex stance and kept out of that observation-only migration): T. rex physics r8 → r9, `trex.xml`'s `<option>` gains `cone="elliptic"` and nothing else (`configs/plant_versions.toml` note 16; the policy interface stays r13 and the visual layer r4; `impratio` stays 1, 10 was measured and rejected). The box pads keep their contact tangent frames on the world axes, and MuJoCo's default pyramidal cone is weaker between them, so a ±45° whole-scene turn was a small physical perturbation (the statue's rewards off by up to 2.6e-2 on seeds 3042–3049); under the elliptic cone every heading agrees to 3e-11. Re-measured on the r9 statue: the four stage files' statue references carry, every move inside one standard error, and every stance bar is kept, although the statue is clean on 39/40 of the certification seeds (seed 3081 nosedives), falls on 4 of 280 panel seeds against r8's 2 and tilts a pad 2.061° on 1 of 240 out-of-sample episodes. Consequences accepted: the plant contract refuses every physics-r8 T. rex checkpoint and none can be widened (no trunk is lost; the one r8 v2 PASS, `20261009_155723`, is research-only); every T. rex task digest moves (38 golden lines); no T. rex trains between this revision and the heading-free policy revision, which lands on r9. Unblocks the first heading-free T. rex stance. |
| D-D30 | Taken 2026-10-11 (the maintainer: one friction model for every species, for consistency across the zoo even where the benefit is small, after the T. rex's D-D29 the same day): velociraptor physics r3 → r4, brachiosaurus r4 → r5 and dibothrosuchus r1 → r2, each MJCF's `<option>` gaining `cone="elliptic"` and nothing else that moves a physics, policy-interface or visual digest (the other MJCF edits are comments; `configs/plant_versions.toml` note 17; `impratio` stays 1), so all six plants share the cone the compsognathus plants were built with. The brachiosaurus's ellipsoid feet and the dibothrosuchus's box pads keep their contact tangent frames on the world axes, so under MuJoCo's default pyramidal cone a ±45° whole-scene turn moved their statues' rewards by up to 0.84 and 3.7e-2 (seeds 3042–3049); the velociraptor's capsule feet were already invariant standing (1.8e-12), not once it fell onto its ellipsoid head. Under the elliptic cone every heading agrees to 1.4e-12 on all three. Re-measured statues: the velociraptor's `stance_quality/v2` statue is clean on 40/40 of the certification seeds and on 238/240 out of sample (r3 239/240; seed 5060 fails the 0.35 windowed load-share bar at 0.342), every bar kept with the r4 extremes in its comment and the statue reference carried (2843.00 against 2842.76); the brachiosaurus and dibothrosuchus statues move by −0.7 and +0.2 and their rails round to the same values. Consequences accepted: the plant contract refuses every velociraptor physics-r3 checkpoint (the 6M stances and the #608 reward pilots, none certified) and every dibothrosuchus physics-r1 checkpoint (the statue-level stance PASS `1d3527cc…` of `20260923_020654` included), and none can be widened; nothing on brachiosaurus r4 was trained; 79 golden lines of the three species move. The brachiosaurus and dibothrosuchus `jax-mjx` declarations stand: nothing compiles them under MJX (D-D17), and MJX supports the elliptic cone. Unblocks the velociraptor exploration follow-up and the brachiosaurus and dibothrosuchus sessions, on the new plants. |
| D-D31 | Taken 2026-10-11 (the maintainer approved the migration of `docs/HEADING_INVARIANCE_DESIGN_2026_10.md` that day for the T. rex and the anatomical compsognathus only, the robot and the velociraptor unchanged; it carries out D-D27's 2026-10-07 amendment, whose heading certificate, the design's other half, is D-D32): T. rex policy interface r13 → r14 and anatomical compsognathus r3 → r4, schema `bipedal-target-heading-free/v1` (`configs/plant_versions.toml` note 18; physics and visual layers unchanged). The policy no longer reads its world heading: projected gravity `Rᵀ(0, 0, −1)` replaces the 4-wide world quaternion, the root's linear velocity is expressed in the gravity-aligned heading frame (forward, left, up) and the prey direction in pelvis coordinates, all from the free root's quaternion `qpos[3:7]` normalized in float64 (`environments/shared/heading_observation.py`), with the design's covariant fallback when pelvis X is vertical and a state error for an invalid quaternion; joint state, the site-frame gyro and accelerometer, the per-foot contacts, the prey distance (still from `xpos[pelvis]`) and the trailing command are unchanged. Widths 64 → 63 and 56 → 55; the compsognathus robot keeps `CompsognathusEnv`'s observation and every digest. The plant contract records the schema's segments with their frames and thresholds, checks that their widths sum to the observation width, hashes the frame helpers with `_get_obs`, and refuses the schema under `jax-mjx`: the T. rex is SB3-only from r14 and its `mjx_config.py` registration is deleted (its low-pass filter pins stay), and its `_cache_ids` tokens now enter its policy digest (the implementation's reading of the approved SB3-only declaration, for the maintainer to confirm). A whole-scene turn leaves the float32 observation unchanged (0 on 100 lifted, tilted, moving states × 7 turns per species), and six-heading paired rollouts repeat to solver tolerance. Statistics or a checkpoint of another observation width are now refused by plant on every load path (training, evaluation, `load_sb3_checkpoint`), not by SB3's shape errors. Consequences accepted: the plant contract refuses every T. rex r13 and anatomical compsognathus r3 checkpoint and none can be widened (the width moves by −1), so the pre-migration T. rex stances (`20261008_163256`, `20261008_163410`, `20261009_155723` and its recovery) and every r3 compsognathus run retrain; the compsognathus recovery calibration is restamped, not re-measured; every T. rex and compsognathus task digest moves (66 golden lines replaced or removed, 60 new). Unblocks the first heading-free T. rex and compsognathus stances, which certify under D-D32's gate. |
| D-D32 | Taken 2026-10-11 (the maintainer approved the migration in `docs/HEADING_INVARIANCE_DESIGN_2026_10.md` for the T. rex and anatomical compsognathus only, with the choices below; this is the design's gate, D-D31 its heading-free observations): the two stances certify under a new gate kind, `stance_quality_heading/v1`, whose report schema is `mesozoic.stance-heading-report/v1`. Protocol `whole_scene_yaw/v1`: after reset the animal, every mocap body and the episode's reference direction turn about vertical through the root, on a stage with no live command, no push and no heightfield (refused before anything is rolled); the grid −90, −45, 0, 45, 90 and 180° is fixed in code (a list-valued threshold would defeat `same_threshold`; another grid is another protocol). At each heading the 40 certification seeds are rolled with the full `stance_quality/v2` decision against the zero-action statue rolled at that heading (its bound, its full-horizon, reward and statue rails, and the T. rex's hop-or-fall rail), and the certificate passes only when every heading passes and the exact one-sided 95% bound on seeds clean at every heading clears `min_clean_stance_lcb` at `min_eval_episodes` seeds (37/40 pass, 36/40 fail); the kind reuses v2's keys, so those two keys each state the per-heading and the grouped panel, and a stance moves to it by `gate_kind` alone. The pooled 240 episodes are reported and labelled as not the statistic. The stage writes `stance_heading_report.{txt,json}` and `stance_heading_panel_selected.csv` and no yaw-zero `stance_gate_report.*`; the judge, publication and backfill re-derive it from its rows and refuse a v2 report, the report-only probe and an absent certificate, never falling back to yaw zero; reuse gains rule 8, which re-derives the certificate at the source instead of trusting the passed verdict, and ancestor records copy no certificate; the in-training manager refuses the kind as it refuses v2. The report-only probe is dropped from the two stances and refused at config load beside the kind; the velociraptor and compsognathus robot stances keep it, warn-only. The notebook's zero-action preflight scores the 0° cell only (the plant's heading invariance is the tests'); the 7042–7081 fresh-audit grid stays a recorded manual command-line step; the statue grid is not cached; the compsognathus declares no hop-or-fall rail, unchanged. Consequences accepted: a checkpoint's certificate rolls 12 panels (about 13 minutes on the T. rex, 21 on the compsognathus, on one CPU) where v2 rolled 2; the two stances' `gate_sha256` and stage-view digests move (6 golden lines), so no v2 verdict of theirs is reused, and none exists on their current plants. Unblocks the first heading-free T. rex and compsognathus stances. |
| D-D33 | Taken 2026-10-11 (the maintainer approved decisions A–G that day): species are named by their binomial (`velociraptor_mongoliensis`, `tyrannosaurus_rex`, `brachiosaurus_altithorax`, `dibothrosuchus_elaphros`, `compsognathus_longipes`, matching the manifest's display names), and every model is the pair (species, embodiment), embodiment `anatomical` or `robot` (one word, `anatomical`, for the non-robot model; a robot's hardware iterations are physics revisions). The six models are the five anatomical ones and `compsognathus_longipes`/`robot`, formerly `compsognathus_robot`. Layout: `configs/<species>/<embodiment>/` (stage files named by their stage with no number, `stages.toml`, `behaviors/`, per-embodiment data) and `environments/<species>/` (species-level material, including one base class per species) with `environments/<species>/<embodiment>/` (env module, assets and body data). Records carry the two fields, never a joined string; schema-versioned records bump; plant versions are `[plants.<species>.<embodiment>]`; explicit embodiment tables replace `variant_of`; classes `<Species><Embodiment>Env`, Gymnasium IDs `MesozoicLabs/<Species>-<Embodiment>-v0`, registered from the manifest. Every alias and abbreviation is removed as an identifier, and old runs and checkpoints stop resolving (accepted). The stray `configs/` package markers go and `docs/SPECIES_NAMING.md` is rewritten. Shared robot infrastructure (servo and IMU models, system identification, the ROS 2 bridge) will be one shared package, not per species, and is not built now. Consequences: policy-interface revision +1 on all six plants (`configs/plant_versions.toml` note 19); D-A10 amended a third time; A5 superseded. |
| D-D35 | Taken 2026-10-16 (the maintainer's decision that day: someone choosing `stand` in the notebook could not tell that it trained stance then push recovery on some models and stance alone on others, and the label `stand` and the stage id `stance` read as the same thing): the recovery node gets its own recipe label, `recover` (a label never equals a stage id, and `recover` is not one). `stand` then resolves to the stance node on every model, so `BEHAVIOR = "stand"` and `"stance"` give the same chain, and `recover` resolves to recovery, chain stance → recovery, on the three models that have one (`tyrannosaurus_rex`, `compsognathus_longipes` anatomical and robot); the labels read as four verbs: stand, recover, walk, hunt. Before it the recovery node was labelled `stand` too, and a label resolves to its deepest deliverable, so `stand` meant stance → recovery on those three and stance alone on the other three. Trade-off accepted: on those three models `stand` no longer publishes the push-robust standing policy; that deliverable is now `recover`. The species manifest names every model's deliverables by label (`stand`, `recover`, `walk`, `hunt`), the command line's `--target stand` is a stance-only run on every model and `--target recover` takes the refusal of a chain through the non-advancing recovery node, and the notebook's `BEHAVIOR` dropdown is `stand`, `recover`, `walk`, `hunt`. Recipe labels enter no training, task, gate, hyperparameter or plant digest, so no digest moves, no checkpoint is invalidated and nothing retrains. D-A11 and D-A24 amended. **Amended 2026-10-18 (decision D-D38):** the stance node's label is now `stance`, its own id, and `stand` is refused naming it, so "a label never equals a stage id" reads "a label equals a stage id only when it resolves to that stage", the four labels are `stance`, `recover`, `walk` and `hunt`, and the species manifest's stance deliverables and the notebook's dropdown say `stance`. |
| D-D36 | Taken 2026-10-17 (the maintainer's decision that day, after the T. rex physics-r10 stance screen was judged; a deliberate exception to two open decisions of `docs/GAIT_QUALITY_PLAN_2026_09.md`, GQ-3 and GQ-17, answering neither; that plan's dated §17): one short T. rex walking pilot runs now, before that plan's preconditions for a T. rex walk are met. It is an exception to GQ-3's recommended option (a), which includes "no walker session on today's rewards" (the pilot trains today's walk reward), and to GQ-17's retrain order, which puts the robot first; its parent is an uncertified stance checkpoint, so it also runs ahead of the certified T. rex stance retrain that the plan's §12 puts before any trex walk. GQ-3 and GQ-17 stay open, the trex `gait-r1` walk (the plan's §5.4 and §14) still waits for a certified stance, and the order of the retrains stays the maintainer's. The screen it follows trained arm L (the T. rex stance with #624's three switches on: mirror-symmetric PPO, the lateral lean term and the settled leg target, all off by default on `main`) on the r10 foot of draft #623 (D-D34), seeds 42, 52 and 60, and failed all four of its fixed criteria: K1, `stance_quality/v2` at 6M passed on seed 42 alone (40/40 clean; seeds 52 and 60 0/40; bar: 2 of 3 seeds); K2, seeds 52 and 60 stood on their toes, with 37 to 40 of their 40 panel episodes full-horizon and over the whole-foot fore-aft centre-of-pressure rail at each of 4M, 5M and 6M, and seed 42 had 5 at 4M (bar: at most 3, every seed, each checkpoint); K3, 16 of 60 pooled joint-space noise falls at 6M (bar 7); K4, 12 of 120 pooled roll-rail failures at 6M (bar 6 pooled and 3 per seed; seeds 52 and 60 had 4 and 8). Arm L on physics r9 had failed only K2 and K3 on the same tests, its K2 on the pad's centre-of-pressure rail (the symmetry screen's verdict O9). D-D34 (approved on 2026-10-16 on the condition that this screen show the r10 foot at least as good as r9's) has no row on `main`: the screen did not clear that condition, so #623 is parked as a draft, `main` keeps physics r9 and the r10 foot is not adopted; revisiting the foot would be a new physics decision and a new screen. The screens' evidence (the symmetry screen's verdict, the r10 screen's fixed criteria, scores and independent recomputation, and the toe-lean diagnosis) is in `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §5–§7 and its evidence folder `docs/investigations/stance_2026_10/`. (1) What runs: the production T. rex locomotion stage (stock PPO, the production `[env]` walk reward, `[ppo]` hyperparameters and `reward_and_length/v1` gate, whose verdict the manual cell records and never enforces), warm-started under `initialize_next_stage` through the SB3 notebook's manual single-node cell, on training seeds 42, 52 and 60, 4,194,304 steps each against production's 8,000,000 (the learning rate follows production's first 4,194,304 steps). (2) The parent: r9 arm L seed 60 at 6,291,456 steps, a periodic checkpoint of the symmetry screen's run on `main`'s physics-r9 plant, sha256 `70f3a4447794a01036c27bca5e0377a17db96d0e3d7bfb2d179e9ce94e6f353e`: 40/40 clean `stance_quality/v2` panel episodes at the panel's single heading, 6 of 20 joint-space and 0 own-sigma noise falls, and no gate verdict (the T. rex stance's gate, D-D32's six-heading `stance_quality_heading/v1`, has not judged it). It is the seed behind 6 of r9 arm L's 8 pooled K3 joint-space falls, and the same run's 5M checkpoint had 39 of its 40 panel episodes over the pad's centre-of-pressure rail (the 6M parent had none); its recorded stance task (`a2af3c5c…`) is the screen harness's, which no committed stance config reproduces. None of arm L's options is on in the pilot: the lean term and the settled leg target are stance reward terms (the locomotion stage leaves `lateral_lean_weight` and `leg_home_pose_weight` at their default 0), and the pilot trains stock PPO, not `MirrorKLPPO`. The parent departs from the r10 screen's pre-registered parent rule, which on a failed screen takes the best of the six 6M arm L checkpoints (most clean panel episodes, then fewer noise falls) and so named r10 arm L seed 42 at 6M (40/40 clean, 0 noise falls); that checkpoint is on the unmerged r10 foot, which the project is not adopting. (3) How it is judged: by the frozen 2026-09 gait-audit probe (`docs/investigations/gait_2026_09/gait_probe.py`, run unchanged through an adapter) at 2,097,152, 3,145,728 and 4,194,304 steps, on 40 deterministic floor-truth episodes per seed (resets 3042–3081), against hop-detecting criteria fixed before any pilot training step. An episode is labelled FALL, HOP or STILL first, and WALK only when its alternation index is at least 0.6, its lead swaps at least 1.0 per stride, its flight fraction at most 0.2 (the gait plan's §5.5 bar for a T. rex locomotion pilot), its hip-pitch joint lag within [0.35, 0.65] of the stride, its measured stride frequency at most 5 Hz and each foot's floor duty at most 0.80. A seed WALKS at 4,194,304 steps with at least 30 WALK and at most 4 FALL of its 40 episodes, and the pilot's outcome is the one at least 2 of the 3 seeds share (MIXED when every seed has an outcome and no two share one, INCOMPLETE while a missing seed could still decide it). A seed's WALKS counts only when the maintainer's review of its videos (the notebook's final replay and its WALK-labelled panel episodes) agrees: the final replay, where the seed has one, shows a walk and at most (WALK count − 30) of those episodes are not walks on video; a disagreement makes the seed MIXED, and while a review that could change the outcome is pending the outcome is PENDING_VIDEO_REVIEW, never WALKS. A collapse-stopped seed is scored from its newest periodic checkpoint, and a run resumed before the forward-weight ramp ended is off-schedule and never scored. Known limits: some short-lift chatter at about 16 Hz or faster, and some hops the probe measures at 5 Hz or below, can still read WALK; the video review is the check. Scope: the pilot runs from the branch `claude/pilot-walk-r9` (`eaeef50f`; never merged, no PR), which, unlike `main`, carries the parent (its provenance in `pilot_checkpoints/r9_arm_l_s60_6291456/SOURCE.txt`), code that lets a stage config set its evaluation and checkpoint cadence (here an evaluation of 10 episodes every 131,072 steps, against production's 30 every 50,000, so the collapse backstop, which counts evaluations, cannot stop a run before 3,801,088 steps; production's earliest is 1,450,000) and the shortened locomotion stage. The criteria in full, the probe adapter (`gait_probe_pilot.py`) and the scoring and judging scripts (`pilot_score.py`, `pilot_judge.py`) are not on the branch; byte-identical copies are in `docs/investigations/stance_2026_10/pilot/` (the criteria's sha256 `1cdd495c…`), and §9 of the stance record states the criteria; the video renderer `pilot_video.py`, which renders the panel episodes the video review watches, stays in the session's scratch. Nothing it trains is certified, published or reused, and no result bundle is written. `main` and its digests are unchanged by this decision; on the branch the locomotion stage's task and gate digests equal `main`'s, and only its PPO hyperparameter digest and its two stage-config-view digests differ (the shortened budget, the learning-rate end and the cadence keys). The outcome is not yet known; a later PR records it. **Amended 2026-10-18 (pilot 1's outcome, and the maintainer's decision of 2026-10-17 to run a second pilot):** the pilot, now called pilot 1, was judged on 2026-10-17 by its fixed criteria and read STILL on all three seeds, final: at 4,194,304 steps each seed was STILL on 40 of its 40 panel episodes, with no WALK, HOP or FALL episode (median speed 0.001–0.002 m/s, no touchdowns; on the trajectory, which decides nothing, seed 42 had one FALL at 2,097,152 steps), so no video review was needed; by the criteria the policy is not moving by 4.2M and the pilot was too short to say anything about the gait. The outcome is the criteria's, not a gate's, and certifies nothing. Report only, and no criterion: the forward-velocity term earned 0.5–3.2 per episode at the 2M, 3M and 4.2M evaluations (0.1–7.5 over every evaluation from 2M to 4.2M steps), against the September T. rex walkers' 39–54 at 2.1M and 148–234 at 4.2M, and the policy's action std (the mean over the 15 actuators) was 0.108–0.111 at 4,194,304 steps against the parent's 0.095. Its nine judgements are rows of the offline ledger `results/offline_judgements.csv`; the evidence is in `docs/investigations/stance_2026_10/pilot1_judging/` and the stance record's §15. After the judging, the maintainer chose on 2026-10-17 to run a second pilot ("Second pilot, wider exploration"): pilot 1 again, with the same parent, stage, budget, seeds, probe and criteria, and one change: at the warm start only, every actuator's `log_std` is set to −1.5, an action std of 0.2231 (the T. rex stance stage's own starting noise, D-D27; the maintainer was told 0.22), after the parent's parameters load and before the first update. A pilot-only key on the branch, `[curriculum] warm_start_log_std = -1.5` in the locomotion stage, sets it through pilot-only code there; on the branch it moves that stage's hyperparameter and stage-config-view digests (PPO hyperparameters `ec9925f9…`, pilot 1's `af78e500…`), and its task and gate digests stay `main`'s. The criteria add three rules that follow from the change: the judging checks that each run's `stage_config.json` records the reset, a run that does not is not scored and its seed counts as missing, and a resume never applies the reset. It runs from the branch `claude/pilot-walk-r9-b` (`ba464855`: pilot 1's `eaeef50f` plus three pilot-only commits; never merged, no PR) under the run ids `pilot2_walk_s42`, `pilot2_walk_s52` and `pilot2_walk_s60`; its criteria (written 2026-10-17 and amended 2026-10-18, both before any of its training steps) and its Colab steps are in `docs/investigations/stance_2026_10/pilot2/`. It is still a deliberate exception to GQ-3 (a) and GQ-17, both still open; nothing it trains is certified, published or reused, no result bundle is written, and `main` and its digests are unchanged. It is not yet trained and has no result; a later PR records its outcome. **Amended 2026-10-20 (walking pilot 2's parent; the maintainer's decision of that day, 7a, option i):** walking pilot 2 will be rebuilt for D-D37 Run 2's final (`stance3_armLT_s60`'s `stage1_final`, sha256 `cc21604a…`; 40 of 40 clean, no noise falls) as its parent, the parent that D-D39's comparison keeps after its screen failed (D-D39's amendment of 2026-10-20). Pilot 2's criteria still name r9 arm L seed 60 at 6,291,456 steps (`70f3a444…`), so they are amended before any pilot-2 training step, the new parent is committed on pilot 2's branch and its start-up checks are re-run. The rebuild comes right after the package of D-D40's screen is ready, and the maintainer starts the pilot when a fourth Colab slot is free; until then pilot 2 stays parked and untrained. It is still a deliberate exception to GQ-3 (a) and GQ-17, both still open; nothing it trains is certified, published or reused, and `main` and its digests are unchanged. The amended criteria and the rebuilt branch are not yet written; a later PR records them and the pilot's outcome. The record is `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §20.4. **Amended 2026-10-21 (walking pilot 2 rebuilt for its new parent, as the amendment of 2026-10-20 decided; recorded on 2026-10-22):** walking pilot 2 was rebuilt on 2026-10-21, from 17:00 to 19:25 UTC by the session's log, after D-D40's package was done (about 07:15 UTC). The new parent, D-D37 Run 2's final (`cc21604a…`, its normaliser `894e86f4…`; an action std of 0.064 on average, 0.009 at the knees and 0.125 at most), is committed on `claude/pilot-walk-r9-b` in three pilot-only commits on `ba464855`: `6a13a303` (the parent and its `SOURCE.txt`), `8fc76eec` (a closing note in pilot 1's parent's `SOURCE.txt`) and `8c5478a6` (two of the new `SOURCE.txt`'s sentences corrected after review); only `pilot_checkpoints/` changes. The criteria were amended on 2026-10-21, before any pilot-2 training step (sha256 `bdcdae4b…`; the pre-amendment text is `a240ff5d…`): the parent, the branch head, the parent's numbers, the readings reworded for the confound (pilot 2 now differs from pilot 1 in the parent and the noise reset, so no outcome is attributed to either alone), a hold on the branch and the pre-launch checks; no outcome definition, threshold, guard, label, seed, budget or tool changed. The start-up checks were re-run: on `8c5478a6` the notebook smoke, the judge accepting the run and refusing pilot 1's parent, and 29 warm-start tests; on `8fc76eec` the full suites (5,937 passed). The push, a fast-forward from `ba464855` to `8c5478a6`, was made at 19:49 UTC on 2026-10-21 with the maintainer's OK, after a first attempt was refused by the session's permission classifier and not worked around; nothing more is pushed to the branch until all three seeds are judged. Pilot 2 is not yet launched. It is still a deliberate exception to GQ-3 (a) and GQ-17, both still open; nothing it trains is certified, published or reused, and `main` and its digests are unchanged. On 2026-10-22 the maintainer kept this parent for now: no parent change now, and D-D40's s52 final, which ties it under stance3's parent rule, is reconsidered after the replication (D-D41 (3)). The amended files are in `docs/investigations/stance_2026_10/pilot2/`, and the record is the stance record's §21.6 and §22. Pilot 2 was retired on 2026-10-25 by D-D45, never launched; its branch is kept, never merged. |
| D-D37 | Taken 2026-10-18 (the maintainer's decision that day, after walking pilot 1 read STILL: the maintainer asked for three stance runs with "different reward settings, slightly different mechanics, or a refinement", and the offline diagnosis of 2026-10-18, which trained nothing, designed them): instead of walking pilot 2, three full T. rex stance screen runs train on `main`'s physics r9, started on 2026-10-18 on Colab L4 runtimes (the maintainer kept L4 for all three; the Colab steps as written named a CPU runtime, the only one the mirror-symmetric trainer had trained 12M on, and asked for one type for all three). Run 1, `stance3_armL_s60`, is arm L as #624 ships it (the lateral lean term, weight 0.25 and tolerance 0.015 m, and the settled leg target on the stance task; `MirrorKLPPO` with kappa 0.3, sigma floor 0.01 and the mirrored critic), seed 60, from the branch `claude/screen-trex-r9-arml-12m` at `7e6b557db999900bbc95df63ca651b1f2f4e2df1`. Runs 2 and 3, `stance3_armLT_s60` and `stance3_armLT_s52`, are arm L plus `leg_home_pose_weight` 1.0 (0.5 on `main`) and `leg_home_pose_tolerance` 0.05 rad (0.10), seeds 60 and 52, from the branch `claude/screen-trex-r9-armlt` at `1af2d4d767aebc89fa2cf73fc59a11004048f797`. Both branches are built on #630's code with the cadence keys of `412f19ae` cherry-picked; neither is ever merged, and neither has a pull request. Every run trains 12,000,000 steps with the committed PPO schedules, gate and bars, evaluates 10 episodes every 131,072 steps and keeps a checkpoint every 524,288 steps, all of them; its collapse reference is production's 3756.8 plus its statue's measured reward gain (3981.4 under arm L, 4460.5 under arm L with the two leg keys). The criteria were fixed before any training step (sha256 `1ce910dcd6e2ff7b41a4f592b9f34b9a6c9cb398c523e88f0ebd19af754f18fa`), judged on the 40-episode panel at heading 0 at 4,194,304, 5,242,880, 6,291,456, 8,388,608 and 10,485,760 steps and the final: K1, the gate held (`stance_quality/v2` PASS at 6,291,456 and at the final, and at most one judged checkpoint below 37 of 40 clean and none below 33); K2, fore-aft held at every judged checkpoint (K2a, the pad rail over 0.80 on at most 3 of 40; the screen-only K2b, each foot's signed whole-foot centre of pressure outside [−5, +100] mm of its MTP III on at most 3 of 40; the screen-only K2c, no drift, the panel median within −60 to +30 mm of the statue's and its range over the six checkpoints at most 30 mm); K3, robustness (at most 2 of 20 joint-space noise falls and 1 of 20 own-noise falls at 6,291,456 and at the final); K4, roll (at most 3 of 40 tilt or corner-lift failures at the same two); with a decision table fixed with them (arm L with the two leg keys holding on both seeds is proposed for committing and certifying on two seeds; each other outcome names the next round; the best final checkpoint of the three is offered as walking pilot 2's parent). Fails stay fails, and no run is stopped or changed on its readouts. Screen runs certify nothing: their configurations are not committed, so the official judge refuses their task fingerprint, and the notebook's end-of-stage certificate on them is report only. Whether screen runs could ever count toward `certification_seeds` if their configuration were later committed byte-identically is not ruled; it is an open question for the maintainer. Run 1's `stage_config.json` on Drive confirms commit `7e6b557d`, arm L's keys, the task `sha256:cd963518…`, the PPO hyperparameters `sha256:054ab483…` and an L4 GPU, training from 16:10 UTC, and its run folder's `zero_action_baseline.json` a zero-action baseline standing at 3980.37 (39 of 40); Runs 2 and 3's `stage_config.json` files confirm commit `1af2d4d7`, seeds 60 and 52, the two leg keys, collapse reference 4460.5 and an L4, from 16:20 and 16:23 UTC, with the statue at 4459.40 (39 of 40). It supersedes, for these runs, the maintainer's rule of 2026-10-16, "no more stance-only reward experiments unless walking points back to stance": pilot 1 read STILL, which by its criteria does not point back at the stance. Walking pilot 2 (D-D36's amendment of 2026-10-18) is parked: its branch `claude/pilot-walk-r9-b` and its criteria are kept, and it is not trained. The basis is the offline diagnosis of 2026-10-18 (rolls of existing checkpoints, the zero-action statue and held poses; no training): the committed stance reward pays for a forward lean of about 23 mm, arm L's settled leg target removes that pay but prices a 14 mm lean at only about 0.3% of the reward, and the lean lives in the ankles, about 10.7 mm per degree. Its findings, the runs and their evidence (the criteria, the Colab steps, the recommendation and frozen copies of the judge and scorer) are in `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §16 and `docs/investigations/stance_2026_10/stance3/`. `main` and its digests are unchanged. The outcome is not yet known; a later PR records it. **Amended 2026-10-19 (the three runs' outcome; no decision taken):** the three runs trained to 12,001,280 steps and were judged on 2026-10-18 and 2026-10-19 by their fixed criteria, final: all three fail, so none holds and none holds fore-aft. Run 1 fails K2 on K2c (the range of the panel median over the six judged checkpoints is 33.9 mm against 30) and K3 (7 and 11 of 20 joint-space noise falls at 6,291,456 and the final, against 2); Run 2 fails K1 and K2 early (clean 0, 0 and 36 of 40 at 4,194,304, 5,242,880 and 6,291,456, with one tail actuator pinned at full command; K2a 10 and 6, K2b 14, 10 and 6); Run 3 fails K2 (K2a 6 at 8,388,608, K2b 4 at the final, a K2c range of 30.6 mm). The decision table's line is 3b, which reads that a joint-space price does not hold the stance and names the next round's candidates, a fore-aft term that needs code or the v6b ankle spring; neither configuration is proposed for committing. The best final checkpoint by the fixed rule is Run 2's (40 of 40 clean, no noise falls), which the criteria offer as walking pilot 2's parent. The verdicts are the criteria's and certify nothing, and the stage's six-heading certificate rolled offline on the three finals (all three PASS) is report only. Not decided, and listed in the record as proposals for the maintainer: what the next round is, whether walking pilot 2 runs and from which parent (its own criteria still name r9 arm L seed 60 at 6M), and any criteria or telemetry change. The report-only readouts (arm L's ankle-carried whole-body drift, L+T's left-right foot split left from the spawn stagger, Run 2's tail-actuator saturation) and an overnight scratch experiment on three continuations, never merged and report only (the ankle anchor refuted at its pre-registered dose, a foot-stagger penalty inconclusive against the matched no-change control, the same penalty on Run 1 refuted), are in `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §17 and `docs/investigations/stance_2026_10/stance3/results/`. `main` and its digests are unchanged. |
| D-D38 | Taken 2026-10-18 (the maintainer's decision that day, "make the standing behavior's name consistent": the notebook's `BEHAVIOR` dropdown said `stand` while the stage id, its config `stance.toml`, its run folder `01_stance` and its gates `stance_quality…` said `stance`): the single name is `stance`. The stance entry of every model's `stages.toml` says `recipe = "stance"`, so the stance node's label is its own stage id on every model; stage ids, configs, run folders, gates, legacy numbers and edges are unchanged. The loader, which has refused any label equal to a stage id since manifest v2 (2026-09-06; restated by D-D35), now accepts one only when it resolves to that stage: the deepest deliverable carrying the label must be the node with that id, so `resolve_behavior`'s label-before-id precedence still never names two nodes. A label equal to another stage's id, equal to its own while a deeper deliverable shares it, or carried only by non-deliverables is refused as before. `stand` is retired with a pointer, not aliased, by D-D33's rule that retired identifiers do not resolve: `resolve_behavior("stand")` raises "<model> has no behavior 'stand': 'stand' was renamed to 'stance' (decision D-D38); use 'stance'", so every place that takes a label refuses it (the notebook's `BEHAVIOR`, `curriculum --target` and `train_curriculum(target=)`, and the species manifest's deliverables), and a manifest declaring `recipe = "stand"` is refused at load with the same pointer. Before → after, on every model: `stand` resolved to the stance node and is refused; `stance` named the stance node as a stage id and names it as a label too, with the same chain (stance alone); `recover`, `walk` and `hunt` are unchanged; recipe labels `stand, [recover,] walk, hunt` → `stance, [recover,] walk, hunt`. The notebook's dropdown is `stance`, `recover`, `walk`, `hunt`, and the species manifest names the stance deliverables `stance` (the species catalog regenerated). Recipe labels enter no training, task, gate, hyperparameter or plant digest, and no artifact records one (a bundle names its target by the stage key, "1" for stance), so no digest moves, no checkpoint is invalidated and nothing retrains; a run started with `BEHAVIOR = "stand"` re-enters with `"stance"`, the same chain, target and trunk selection. D-A11, D-A24 and D-D35 amended. |
| D-D39 | Taken 2026-10-19 (the maintainer's decisions that day, after the three D-D37 runs failed their fixed criteria on line 3b and an overnight scratch experiment left a foot-stagger penalty unattributed at one run per arm): the next T. rex stance screen trains three new runs of one arm, LTS ("L+T + stagger"), and reuses two D-D37 runs as its control. LTS is D-D37's L+T recipe (the branch `claude/screen-trex-r9-armlt` at `1af2d4d767aebc89fa2cf73fc59a11004048f797`) plus a stance stagger penalty: after the settle (step > 200), while the command is zero, each step pays −0.30 × (1 − exp(−(s / 0.045)²)), where s is the right foot site's lead over the left along the gravity-aligned pelvis heading (`stance_stagger_weight` 0.30, `stance_stagger_tolerance_m` 0.045). It runs from the screen branch `claude/screen-trex-r9-armlt-stagger`: `1af2d4d7` plus `ff7114c0ca7f1bf977211c43012b4f5f86825ae9` (the term as an opt-in `[env]` option, off by default, which leaves the L+T task digest unchanged) and `c1e146a82658112a9a1c0838250020b214ef303a` (the two keys on in `configs/tyrannosaurus_rex/anatomical/stance.toml`, the task `sha256:3fa46708…`, and the collapse reference 4459.3, production's 3756.8 plus the statue's measured gain under the arm); it is never merged and has no pull request. Runs `stance4_LTS_s52`, `stance4_LTS_s60` and `stance4_LTS_s42` (seeds 52, 60 and 42) train from scratch on physics r9 for 12,000,000 steps (final 12,001,280) with the SB3 notebook, `n_envs` 4 and the committed PPO schedules, evaluating 10 episodes every 131,072 steps and keeping a checkpoint every 524,288 steps, all of them. The runtime: the maintainer's decision named Colab CPU runtimes, as D-D37's Colab steps had. Measured, D-D37's runs trained on Colab L4 GPU runtimes: the `gpu` block of each run's `01_stance/stage_config.json`, as fetched for the judging at 4,194,304 steps, reads L4. That file records only the session that wrote it, so that each control run stayed on L4 throughout is inferred. Before any of the three runs started, the criteria were re-fixed for L4, the control's runtime type, for the maintainer to confirm when starting the runs; the Colab steps ask for L4 and for each run's runtime type at its start, and a later PR records the maintainer's confirmation with the runs. If no L4 is available, the maintainer may pick another runtime; each run's runtime is read from its `stage_config.json` and reported beside the verdicts, and the runtime never changes a verdict. The control is not retrained: it is D-D37's Run 2 (`stance3_armLT_s60`) and Run 3 (`stance3_armLT_s52`), as judged and scored under D-D37's criteria (Run 2 fails K1 and K2 and passes K3 and K4; Run 3 fails K2 and passes K1, K3 and K4). The criteria were fixed before any training step (sha256 `978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38`; they replace a first file, sha256 `8bda933e…`, that named a Colab CPU runtime, as the decision did, and differs from them only in the runtime): K1–K4 exactly as D-D37's, with the same judge settings, the same six judged checkpoints (4,194,304, 5,242,880, 6,291,456, 8,388,608, 10,485,760 and the final), the same HOLDS (K1–K4 all hold) and FORE-AFT HOLDS (K1 and K2), and K2c's 30 mm range bar unchanged. The credit rule: the term is adopted, that is proposed to the maintainer for committing, only if (a) at least 2 of the 3 stagger runs hold and (b) for each of K1–K4 the fraction of stagger runs passing it is at least the control's; with the control's results (K1 1/2, K2 0/2, K3 2/2, K4 2/2) and (a), (b) means that no stagger run fails K3 or K4. The decision table, with H the number of stagger runs that hold and K2p the number passing K2: (1) H ≥ 2 and (b): adopt, proposing the option (off by default) and the two keys as the T. rex stance recipe, with the best holding final as the stance parent; (2) H ≥ 2 without (b): not adopted, replicate on 2 more seeds before any commit; (3) H = 1: not adopted, not separated from seed luck, replicate on 2 more seeds before any commit; (4) H = 0 and K2p ≥ 2: not adopted, the next round targets the other failures (for K3, the leg exploration-noise floor that line 2 of D-D37's decision table names); (5) H = 0 and K2p ≤ 1: not adopted, the next round tests a per-foot centre-of-pressure band term (new contact-reward code) or the v6b ankle spring. In every case the best stagger final by D-D37's rule (most clean panel episodes, ties to fewer joint-space plus own-sigma noise falls) is compared with D-D37 Run 2's final (40 of 40 clean, no noise falls; in the maintainer's words, "the current walking pilot 2 parent"), and the better is offered as walking pilot 2's parent; pilot 2 stays parked until the maintainer says otherwise. The ankle anchor and arm L are dropped from further screens. Report only, never in a verdict: every 1,048,576-step checkpoint outside the six, judged like them, for the stagger runs and for the controls where fetched; diagnostic panels at each judged checkpoint for both arms (mirrored leg noise, the prey straight ahead, both); D-D37's pre-registered readouts; a saturation alarm on the eval ledger (two evaluations in a row with at least 5 of their 10 episodes listing `max_actuator_saturation_fraction` among their screen reasons); and stagger watches (the episode yaw change, training-fall causes with tail and head contact, the stagger at step 200 against the window). Fails stay fails, and no run is stopped or changed on its readouts. Not retraining the control departs from the overnight interpretation's recommendation of an unchanged L+T arm in the same screen on the same seeds; the criteria name it as a caveat, with the control having trained a day earlier and seed 42 having no control run. Screen runs certify nothing. The criteria are in `docs/investigations/stance_2026_10/stance4/criteria.txt` and the decision's record in `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §18. `main` and its digests are unchanged. The runs had not started when this was recorded; a later PR records the outcome. **Amended 2026-10-20 (the three runs' outcome, and the runtime confirmed):** the three runs trained to 12,001,280 steps and were judged on 2026-10-20 by their fixed criteria, final, the verdict of record written by the scorer once the runs had ended: all three fail. `stance4_LTS_s52` and `stance4_LTS_s60` fail K1 (five of their six judged checkpoints below 33 of 40 clean, 21 to 30, and `stance_quality/v2` FAIL at 6,291,456) and K2 (K2a 10 to 19 of 40 at each of those five; K2c passes); both are 39 of 40 clean at the final. `stance4_LTS_s42` fails K1–K4: 0 to 17 of 40 clean, with the actuator-saturation rail failing in most of its panel episodes up to the final (which actuator is not recorded), K2a 4 and 5 at 6,291,456 and the final, and 8 of 20 joint-space noise falls and 4 of 40 roll-rail failures at the final. H = 0 and K2p = 0, so the credit rule fails ((a): no stagger run holds; (b): K1 0 of 3 against the control's 1 of 2, K3 and K4 2 of 3 against 2 of 2), and the decision table's line is 5, final: the stagger penalty is not adopted, and the next round tests a per-foot centre-of-pressure band term or the v6b ankle spring. The best stagger final by D-D37's rule, `stance4_LTS_s52`'s (39 of 40 clean, no noise falls), is one episode short of D-D37 Run 2's final (40 of 40, no noise falls), which stays the walking pilot 2 parent offered. The runtime: the `gpu` block of each run's `stage_config.json` reads an NVIDIA L4 (measured), and on 2026-10-20 the maintainer confirmed the Colab L4 GPU runtime for all three, which closes the confirmation this row left pending, and kept it for the next screen; that each run stayed on the L4 throughout is inferred (the `progress.json` files of seeds 60 and 42 record one session each; seed 52's was not fetched). The verdicts are the criteria's and certify nothing. Report only: the 1,048,576-step and selection checkpoints, the fresh, tail and common-noise panels on the finals, the notebook's own six-heading gate on each run's early `robust_best` (seeds 52 and 42 PASS, seed 60 FAIL), the diagnostic panels at every judged checkpoint of both arms, the readouts (by 4,194,304 the penalty had lined the feet up, but on seeds 52 and 60 the left-right split stayed, as a per-foot loading split driven as before by the spawn's leg asymmetry and the prey's bearing) and two overnight scratch experiments, report only and entering no verdict, whose new runs' judge files are relaxed-binding (a replicate of the stagger penalty in continuation, in which the window stagger sd separated completely and K2a did not; a continuation test of a per-foot centre-of-pressure band prototype, lower on K2a and the split for both parents, with new safety flags on one), in `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §19 and `docs/investigations/stance_2026_10/stance4/`. The next round is D-D40, and walking pilot 2's parent D-D36's amendment of 2026-10-20. `main` and its digests are unchanged. |
| D-D40 | Taken 2026-10-20 (the maintainer's decisions that day, in the session's chat, logged in short and numbered 1a to 6 in the session's log, after D-D39's stance4 screen read its line 5): the next T. rex stance screen trains a per-foot centre-of-pressure band from scratch, line 5's "per-foot CoP band term (new contact-reward code)", rather than the v6b ankle spring, line 5's other candidate (1a). The band is a new reward term, an opt-in `[env]` option of the T. rex env, off by default: each step after the settle (step > 200), while the command is zero, it reads each foot's whole-foot centre of pressure as the judge's K2b statistic does and charges a foot whose centre of pressure lies outside [−0.6, +0.4] span half-lengths, or whose pad centre of pressure lies outside [−0.65, +0.65], up to half the weight (tolerance 0.357 span half-lengths); a foot that is unloaded or stands off its pad pays the full per-foot price, and a lightly loaded foot pays the load-share guard's price on the ramp of 3a below. It is the prototype of the design note `docs/investigations/stance_2026_10/stance4/copband/design.md` (a scratch commit, never pushed) with three fixes and its weight, locked as recommended (3a–3d): the note's 35% load-share guard, a step, becomes a ramp (the full price at a load share of 0.25 or less, no guard price at 0.35 or more, linear between); a speed pass, one fused per-step reduction pinned byte-identical to the recorder's readings (the prototype measured +0.62 ms per env step over band-off; no numeric target was decided); the contact reduction moves out of `environments/shared/gait/` into a neutral shared module, so that D-D23's gait layering rule holds with no exception, and the note's recorder bit-identity test, which pins the band's readings, moves with it; and the band weight 0.30. Three 12M runs train from scratch on seeds 52, 60 and 42 on Colab L4 GPU runtimes (the runtime D-D39's amendment of 2026-10-20 confirms), judged by criteria fixed before any training step, with K1–K4 unchanged; the seeds and the 12M are in the option text the maintainer locked in the chat, not in the log's 1a line, and physics r9 and `n_envs` 4 follow from 2A (the stance4 recipe plus the band only). The arm is L+T + the stagger penalty (0.30, 0.045 m) + the band, D-D39's arm plus the band only, so the three stance4 runs are its control, one difference apart (2A). The design note had recommended L+T plus the band alone, with D-D37's L+T runs as the control (a band-plus-stagger arm was its alternative if the stance4 screen read line 4 rather than 5); the recommendation changed, and the maintainer was told so with the decision, because the overnight band test ran on stagger parents and the stagger term prices the route the note calls its main risk (each foot's centre of pressure held in band by asymmetric ankle torque while the feet stay staggered), which the band's guards cannot see. The v6b ankle spring (ankle stiffness 40 → 250 N·m/rad, damping 45 → 90, the spring reference at the statue's settled 78.6°; a plant revision) is held in reserve, with a trigger line in the band screen's fixed decision table: if the band fixes the split but runs fail K2 on drift (a common fore-aft shift), the next round tests the spring (4a). Actuator saturation gets no training change: during the band screen each run's evaluation ledger is read at every check-in and the maintainer is told at once when the saturation alarm fires (run, step, actuator), and an offline, report-only diagnosis of the saturation of `stance4_LTS_s42` (which actuator, its onset, its trigger) runs in parallel, started after the band build's timing measurement; fails stay fails, and no run is stopped or restarted on an alarm (5a). Mirrored-leg spawns and the prey straight ahead stay diagnostic panels only, adopted into neither training nor the criteria, run at every scored checkpoint of the band screen, report only (6). The band's code, criteria, judge and Colab steps are being built separately and are not committed anywhere public; this row records the decision, and a later PR records them and the screen's outcome. Screen runs certify nothing. The record is `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §20. `main` and its digests are unchanged. **Amended 2026-10-22 (its build, its runtime and its outcome):** the band was built as decided, on the scratch branch `scratch/cop-band-v2` on the prototype `f94452ad`: 3c (`3a1d5de3`), 3b (`4819eb6c`) and 3a (`a15494ed`) on 2026-10-20, and the review's fixes (`99e205bb`) on 2026-10-21. Measured, band off is byte-identical to `c1e146a8` (10 of 10), and band on equals the recorder's readings on 16,000 foot-steps with no mismatch; the band-on overhead is +0.245 to +0.378 ms per env step (+18–30%, against the prototype's +0.62 ms), so the build's own goal of 0.15 ms, which no decision set (3b), was not met. 3d is the config commit `cf80ecb1` (weight 0.30, collapse reference 4457.4, the band-on stance task `8c6998d2…`). The criteria (sha256 `3b28963f…`, fixed before any training step and re-fixed after their review, before any), the judge (`d9286724…`), the scorer (`d5515041…`) and the Colab steps are in `docs/investigations/stance_2026_10/stance5/` (the judge's and scorer's copies carry the folder's prefix change, so their sha256 differs), and the screen branch `claude/screen-trex-r9-armlts-copband` was pushed at `cf80ecb1` on 2026-10-21 (by about 07:15 UTC), never merged and with no PR; its history holds the prototype `f94452ad`. This supersedes the row's "being built separately and are not committed anywhere public" and the design note's "never pushed". The three runs, `stance5_LTSB_s52`, `stance5_LTSB_s60` and `stance5_LTSB_s42`, trained in one session each from 13:36:43–13:38:28 UTC on 2026-10-21 to 03:14:04–04:03:00 UTC on 2026-10-22, at 230.9 to 245.2 steps per second (each run's `progress.json`); measured, each recorded a Colab L4 GPU runtime (the `gpu` block of its `stage_config.json`), so that each stayed on it throughout is inferred. They were judged on 2026-10-22 by the fixed criteria, final: s52 holds (K1–K4); s60 fails K3 only (3 of 20 joint-space noise falls at the final); s42 fails K1 and K3 (15 to 38 of 40 clean, mostly on the actuator-saturation rail, and 13 of 20 joint-space noise falls at the final). Fore-aft holds on s52 and s60. Runs passing, band against control: K1 2 of 3 against 0 of 3, K2 3 of 3 against 0 of 3, K3 1 of 3 against 2 of 3, K4 3 of 3 against 2 of 3; H = 1 and K2p = 3, so (a) and (b) fail, and the decision table's line is 3, final: "H == 1: not adopted; not separated from seed luck; replicate on 2 more seeds before any commit." The band is not adopted. The parent comparison (report only) ties s52's final with D-D37 Run 2's final (40 of 40, no noise falls). Report only, under 5a: s42's saturation alarm fired at 3,670,016 (its first flagged evaluation 3,538,944), and rollouts of its 4,194,304 checkpoint named `tail_3_pitch_act` at +1 (19:48 UTC on 2026-10-21), unchanged at 5,242,880 and 6,291,456, the actuator and sign of D-D39's s42 and D-D37 Run 2; s52 and s60 had no flag; no run was stopped or changed. The offline diagnosis of D-D39's s42 saturation that 5a asked for is done, report only. Screens certify nothing. The maintainer's decisions that followed are D-D41. The record is the stance record's §21. |
| D-D41 | Taken 2026-10-22 (the maintainer's decisions that day, in the session's chat, logged in short in the session's log in its entries timed 15:08, 15:11 and 16:32 UTC, after D-D40's stance5 screen read its line 3): (1) the unchanged band recipe is replicated: D-D40's arm LTSB from the screen branch `claude/screen-trex-r9-armlts-copband` at `cf80ecb1`. (2) Two band runs only, seeds 70 and 80, judged on K1–K4 against the existing stance4 control, which is not retrained, with new criteria fixed before training (the log names "noise level for the new seeds, how the 5 band runs are read"); 12M steps, physics r9, `n_envs` 4 and a Colab L4 runtime are in no decision's words and follow from "unchanged" (inferred). The reading rule (15:11): ADOPT only if both new seeds hold (H = 3 of 5); the K3 pass fraction against the control (3 of 5 against 2 of 3 at best) is reported, not a veto. The log gives the reason: under stance5's rule (b) adoption would be impossible, since the band arm already has two K3 failures. The new seeds' joint-space noise sigma (the K3 test reads a per-seed `control_sigma`, which the frozen parameters hold only for seeds 42, 52 and 60) is the element-wise mean of the three: at 15:11 the session's default, stated to the maintainer with the reading rule, as the log has it ("my default, stated to the maintainer"); in the entry timed 16:32 the maintainer confirmed it, one vector for both new seeds. (3) No parent change now: walking pilot 2 stays on D-D37 Run 2's final, and D-D40's s52 final is reconsidered after the replication. (4) A records PR now, documentation only: this change. In the entry timed 16:32 the maintainer also adopted run-handling defaults for the replication: a run cut off before its first periodic checkpoint and restarted under `<RUN_ID>_b` is that run; a judge refusal on the run's own records counts as an invalid J file; a resumed run is judged as it stands, with superseded files moved aside, never deleted; a cut-short final resumes from the newest intact periodic checkpoint in the same run id, and the final slot is only a `stage1_final` pair at 12,001,280; an early stop makes the final J file invalid; "ended" and "exists" are defined as proposed; on a refusal to reproduce stance5 nothing is read until the cause is found; and with no L4 on a resume, the maintainer is asked first. The replication's package was built and validated by that entry (a judge that reproduces stance5's s52 judge file at 4,194,304 field for field, a scorer that reproduces stance5's three verdicts, 37 tests passing, criteria fixed at sha256 `10464eab…`) and is being re-fixed for those defaults before its Colab steps go out; nothing has trained, and a later PR records the package and the outcome. Separately, at 15:52 UTC the maintainer agreed to two report-only workflows: a diagnosis of the band runs' joint-space noise falls (K3), and a scratch experiment on the tail habit (P1 of the stance4 diagnosis, a penalty on the policy mean beyond ±1, in two 1M-step CPU continuations from stance5 s42's 4,194,304 checkpoint against a control, on a scratch clone never pushed); neither has a result yet, and neither enters a verdict. Screen runs certify nothing. The record is `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §22. `main` and its digests are unchanged. **Amended 2026-10-24 (its outcome):** the package was made final at 17:46 UTC on 2026-10-22 and re-pinned at 18:00 UTC for the run log's time correction, before any training step: the criteria at sha256 `a38536de…` (with a RUN HANDLING section for the defaults above), the judge `judge_stance5r.py` (`16d39f7d…`), the scorer `score_stance5r.py` (`55c37ba7…`) and the sigma sidecar (`89778b3f…`; mean 0.160713); these and the Colab steps are in `docs/investigations/stance_2026_10/stance5/replication/` (the judge's, scorer's and sidecar's copies carry the folder's prefix change, so their sha256 differs). The two runs, `stance5_LTSB_s70` and `stance5_LTSB_s80`, started at 19:33 UTC on 2026-10-22 (the maintainer's report; `session_started_at` 19:33:38 and 19:34:03 UTC) and ended at 12,001,280 steps in one session each (`progress.json` written at 09:15:41 and 09:20:53 UTC on 2026-10-23), at 243.3 and 241.9 steps per second; measured, each recorded a Colab L4 GPU runtime, so that each stayed on it throughout is inferred. They were judged on 2026-10-24 by the fixed criteria, final (the score of record in the run log's entry timed 13:04 UTC), with the three D-D40 runs counted as recorded and reproduced: s70 holds (40, 40, 40, 39, 40 and 40 of 40 clean, a K2c range of 13.7 mm, 0 and 1 of 20 joint-space noise falls); s80 fails K1 only (40, 4, 14, 26, 39 and 38 of 40 clean, the three checkpoints below 33 of 40 on the actuator-saturation rail, 36, 25 and 14 episodes). H = 2 of 5, and the decision table's line is 2, final: "Exactly one of s70 and s80 HOLDS (H = 2 of 5): not adopted." The band is not adopted. Report only: the K3 pass fraction is 3 of 5 against the control's 2 of 3, not a veto; per test, band against control, K1 3 of 5 against 0 of 3, K2 5 of 5 against 0 of 3, K3 3 of 5 against 2 of 3, K4 5 of 5 against 2 of 3; the parent comparison still ties D-D40's s52 final with D-D37 Run 2's final (40 of 40, no noise falls), and walking pilot 2's parent stays D-D37 Run 2's final under (3). Under D-D40 5a, s80's saturation alarm fired at 4,456,448 (its first flagged evaluation 4,325,376), and rollouts of its 4,718,592 checkpoint named `tail_3_pitch_act` at +1, the actuator and sign of D-D40's s42, D-D39's s42 and D-D37 Run 2, at 74% of window steps (92% at 5,242,880 and 63% at 6,291,456); s70 had no flag; no run was stopped or changed. Not judged, to save usage after the account hit its spend limit: the 1,048,576-step checkpoints 7,340,032, 9,437,184 and 11,534,336 and the selection pairs. The two report-only workflows of 15:52 UTC reported on 2026-10-22: the K3 diagnosis (D-D40's s42 noise falls are not the tail habit; hip-pitch noise, mostly its left-right differential part, is the main cause; the mechanism is open) and the scratch tail experiment (a bound on the policy mean lowered tail saturation against its paired control but relapsed, and its effect against training alone is not established; the scratch clone was never pushed). On the maintainer's OK of 22:56 UTC on 2026-10-22, report-only K3 readouts were fixed by an addendum before any J checkpoint existed and read on both runs at 6,291,456 and the final (full-noise falls 0, 1, 1 and 2 of 20). None enters a verdict. Screen runs certify nothing. The maintainer's decisions that followed are D-D42. The record is the stance record's §23. |
| D-D42 | Taken 2026-10-24 (the maintainer's decisions that day, in the session's chat, logged in short in the session's log in its entry timed 15:20 UTC, after D-D41's replication read its line 2; the design of (1) and (3) is pending): (1) the next T. rex stance round targets the tail saturation habit, the first of the three options of D-D40's line 4 that D-D41's line 2 offers; the round is designed with fixed criteria and brought back to the maintainer for approval before any training. (2) The training-history report goes to a pull request at scope (a), the standalone report as built (report-only pages drawn from the run ledgers, built as the first piece of the stage report's design of 2026-10-18, carried to the new session as patches). (3) An actuator saturation report, both during training and for finished models, from the ledger: report-only telemetry naming the saturating actuator and its sign on each evaluation-episode row, drawn over training by the training-history report and read for finished models from their evaluation rows. (4) A records PR now, documentation only: this change. No training follows from it and no round is designed: at 15:27 UTC an options study for (1) started, with no training and no Drive fetch, its brief to come to the maintainer before any design is fixed; (2) and (3) are to come in their own pull requests, not in this change (the run log is recorded to 15:27 UTC). Walking pilot 2 is not launched, and its parent stays D-D37 Run 2's final. The record is `docs/investigations/TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md` §23.7. `main` and its digests are unchanged. **Amended 2026-10-24 ((2) widened to scope (b)):** the scope (a) report is #636, and a further decision of the maintainer's that day widens (2) to scope (b), as the stage report's design of 2026-10-18 specifies: `generate_stage_artifacts` also draws the training-history report at the end of every stage, into `figures/training_history/` of the stage directory, best-effort in its own guard after the verdict is recorded, so the SB3 training notebook has it without a manual step. Report only: no gate, reward, observation, action, task, checkpoint selection, plant or digest change. Its pull request carries this note; `docs/TRAINING_ARTIFACT_PRESENTATION.md` ("At stage end: `figures/training_history/`") documents it. **Amended 2026-10-24 (three more report-only tools; (1)'s first step):** at 15:56 UTC, after the session answered the maintainer's questions on the report, the replays and the policy probe, the maintainer decided four more report-only items: (a) the band runs' `policy_probe.npz` read from Drive in one batched, read-only fetch (read in the session's scratch: the probe's bank states are not an early detector of the tail habit); (b) a per-step trace tool for deterministic episodes at a checkpoint (per actuator: the raw mean, the command, the servo target, the joint angle and the force), with the command, raw mean and joint position in the replay's per-frame stance CSV, in #638; (c) the training rollouts' clip shares per actuator in the policy probe, in #640 beside (3), the saturating actuator named on each evaluation-episode row; (d) the stage-end report, the widening of (2) to scope (b) amended above, in #639, after #636. None changes a gate, reward, observation, action, task, checkpoint selection, plant or digest. For (1), the options study reported at 15:58 UTC and its corrected brief at 16:10 UTC (a training-loss bound on `tail_3_pitch_act`'s mean recommended for a CPU screen only, after two skeptics rejected a five-seed from-scratch round of a 0.90 bound); the round's first step is D-D43. The record is the stance record's §24.1–§24.3. |
| D-D43 | Taken 2026-10-24 (the maintainer's decisions that day, in the session's chat, logged in short in the session's log in its entries timed 16:00 and 17:46 UTC, on the options study's corrected brief, as the first step of D-D42 (1)'s round; the criteria approved at 19:06 UTC): (1) probes, then a CPU continuation screen: one batched, read-only Drive fetch (the five band runs' `policy_probe.npz` and the 4,194,304 checkpoint pairs of `stance5_LTSB_s42` and `stance5_LTSB_s80`), CPU probes without training, then a scratch continuation screen, bound against no bound, three replicates per arm from both states, whose criteria come to the maintainer before it runs, since it changes the training loss. (2) The bound's scope: `tail_3_pitch_act` (actuator index 14), upper side only. (3) "Rail cleared but tail parked" (a command of about 0.6 to 0.9) counts as NOT ADOPTED and goes to the posture decision; at 17:46 UTC the rule stays literal: a window-mean command of 0.60 or more with the rail clear is PARKED, and clean holding states such as D-D37 Run 2's final read PARKED as postures, with no verdict or parent change. (4) Three arms: CONTROL (coef 0), BOUND90 (b = 0.90, a rail guard) and BOUND60 (b = 0.60, a posture bound), each a term coef × mean over rows of relu(μ₁₄ − b)² in the PPO loss, three replicates on each of the two states (18 runs, about 6 h of CPU). (5) A durable copy on an orphan, never-merged handoff-style branch, `claude/handoff-tailscreen`, carrying the screen's data and the bound's patch as a file (the maintainer's explicit OK for the scratch patch on origin as a file). (6) One continuous run of 1,048,576 steps per replicate (4,194,304 to 5,242,880), read at 5,111,808 and the end. The technical defaults of the session's decisions list were taken with it: the coef by P1's calibration rule per bound, the reading order, the mean of the two read checkpoints, RAIL counted on `tail_3` tail-up flags with OTHER flags as a harm review, and seeds 101, 202 and 303. The maintainer approved the criteria at 19:06 UTC (sha256 `efd8e297…`), coef 2.4 for both bounds as calibrated (`coef.json` `f6d90f99…`, pooled policy-gradient RMS 0.5267; no optimizer step), the scratch commit `4c0c8da7` (two commits on `cf80ecb1`, changing `mirror_kl_ppo.py` and its test file only, never pushed) and the durable branch. Each bound is read against CONTROL only; per bound, the first of RAIL, PARKED, OTHER, NOT SEPARATED (s42) or PASS that applies at either state gives NOT ADOPTED (with its reason) or PASSED; the screen's outcome is A (both passed), B (one) or C (neither). It is scratch and runs on the session's CPUs; it certifies nothing, and nothing it runs is on `main`. Walking pilot 2 is not launched, and its parent stays D-D37 Run 2's final. **Amended 2026-10-24 (the gate re-pin):** with the approval recorded at 19:07 UTC, the pre-training gate failed on one test of the scratch clone's file, `test_no_stage_config_enables_symmetry`, `main`'s rule that no shipped stage sets `[ppo] symmetry*`, which fails identically on an untouched `cf80ecb1` copy, whose band arm sets it by design; at 19:09 UTC the maintainer decided to deselect that one test in the gate and re-pin `gate.sh`, `screen_common.py`, the criteria and the approval record, with no other change and no new calibration. Re-pinned at 19:10 UTC (criteria `9930ee72…`, its APPENDIX G recording the change); the gate passed at 19:11 UTC (200 passed, 2 skipped, 1 deselected). Nothing had trained. **Amended 2026-10-25 (its outcome):** the 18 replicates ran from 19:11 to 23:46 UTC on 2026-10-24, all exiting 0 with no crashed attempt, and the screen was scored at 23:53 UTC, final: outcome C, neither bound adopted. BOUND90 NOT ADOPTED (RAIL: every replicate had tail-up saturation episodes at R1 and R2, s42's from a railed start and s80's from a clean one, R1/R2 mean commands 0.886 to 0.971); BOUND60 NOT ADOPTED (PARKED: no tail-up flag, but R1/R2 mean commands of 0.600 to 0.632, at or above the 0.60 line, s42's three clearing it by as little as 0.00017, while the verdict also holds on s80 alone; routed to the posture decision); CONTROL read RAIL, CURED and RAIL from s42 and RAIL three times from s80 (the onset test informative); OTHER 0 against 0 everywhere. An independent verification at 00:13 UTC on 2026-10-25 re-scored it from the criteria alone, checked the provenance and re-measured four panels, with no blocking finding; report only, the bound acted on the training rows (the share of rows above b lower under the bound in every seed-paired comparison). A NOT ADOPTED at coef 2.4 does not reject a stronger dose. The maintainer decided at 00:15 UTC on 2026-10-25: a records pull request for the round first, documentation only, then the posture options with evidence, before any decision; it chooses no round and changes no criterion, so it takes no decision id of its own. The criteria, the approval, the patch, the gate records, the tools, the score and the verification are in `docs/investigations/stance_2026_10/stance5/tailscreen/` (its scripts carrying the folder's changes, so five of the eight tools do not have their pinned sha256); the record is the stance record's §24. |
| D-D44 | Taken 2026-10-25 (the maintainer's decisions that day, in the session's chat, logged in short in the session's log in its entries timed 01:33, 02:18 and 04:55 UTC, after the D-D43 screen and the posture options study that followed it; physics changes are the maintainer's to approve): T. rex physics r9 → r10, the probe's candidate SR3, and nothing else. At 01:33 UTC, on the posture study (report only; its two skeptics found that no posture reachable on r9 holds a level tail, the r9 postures differing by 1.5–3° inside a band 22–28° below horizontal with `tail_1` on its ventral stop, and that passive support, not servo reach, is the route that holds a level tail off the rail), the maintainer decided to probe an r10 tail first, on CPU with no training and scratch MJCF copies only, and then to decide whether r10 comes before more r9 work. The probe proposed SR3: `springref` 52.5, 25.9 and 10.4° on `tail_1_pitch`, `tail_2_pitch` and `tail_3_pitch`, so that the compiled model differs from r9's only in `qpos_spring` and the zero-action statue rests its tail at the keyframe zeros on its springs, its servos near 0 N·m, where r9's statue settled `tail_1` on its ventral stop (`tail_2` just above its own) and needed commands of +3.50, +2.16 and +1.15 to hold a level tail; the centre of mass moves 10.7 mm back and open-loop falls under command noise shift backward (16/40 against 21/40 reach the horizon at SD 0.10, within chance). At 02:18 UTC the maintainer asked for more CPU checks first, with no training: eight r9 checkpoints, the walking-pilot parent included, zero-shot in closed loop on SR3 (all upright; gate-clean episodes 293 → 229 of 320, two checkpoints pinning a tail actuator at +1 and one losing its rail; the parent's head tip 1.24 → 0.61 m while it still passes the stance gate); the statue through the real gate code on an SR3 tree (278/280 against 275/280, its one fall, seed 9047, backward; the six-heading certificate 40/40 against 39/40); and the heading-invariance test's failure diagnosed as float32 rounding that already fails 7 of 20 r9 seeds, not a heading or SR3 effect. Their skeptic found no blocking error and corrected the reading: the pooled McNemar p-values overreach (checkpoint-level sign test p 0.29), zero-shot gives no sign that SR3 cures the rail, the backward-fall risk is untested at useful power, and the zero-shot rewards used r9's tail target. At 04:55 UTC the maintainer decided: (1) build r10 as SR3, `physics_revision` 10, policy interface r15 kept, with no training, every r9-pinned constant and golden re-measured and adversarially checked, and no pull request until the maintainer asks; the first r10 training round gets its own criteria for approval; (2) a reward change for r10: `tail_home_pose`'s target (`_TAIL_SETTLED_QPOS`) re-centred on SR3's measured settled pose, its tolerance kept at 0.05 rad; (3) the heading-invariance test's fragility goes to its own test-only pull request now (the full-horizon traces test only: a closed-loop reward bound of 1e-6, the `METRIC_ATOL` precedent, and a float32-aware observation bound; the 250-step test keeps its bounds; no gate, physics or interface change). Built on the scratch branch `claude/physics-r10-sr3` off `main` at `3ffc18e9` (`configs/plant_versions.toml` note 20): the tail target (−0.2107, 0, −0.2029, −0.0926) → (0.00002, 0, 0.00006, 0.00018) rad; the settled neck and leg targets, `natural_pitch` 0.027 → 0.017 and the stance `nosedive_termination_threshold` 0.493 → 0.503 (the absolute cut kept), the settled m·g·h 704.6 → 729.1 N·m/rad (the test's literal 704.4 was r8's figure), re-measured; the statue references by the precedent's one-standard-error rule: stance 3756.8 → 3766.6 with `min_avg_reward` 2250 → 2260, recovery's pushed statue 1329.1 → 1429.1, locomotion 1091.5 → 1092.6, behavior 602.0 and its 361 rail carried, `statue_constants_physics_revision` 10 in all four files; every stance bar kept, with the r10 statue's extremes in its comment (40/40 at every heading; out of sample, seed 9047's backward fall and seed 3121's 2.087° pad tilt over the 2.0° bar). Consequences accepted: the plant contract refuses every physics-r9 T. rex checkpoint and none can be widened (no certified trunk is lost; the stance screens' runs and walking pilot 2's parent stop loading on `main`, and walking pilot 2 can run only on its r9 branch); every T. rex task digest moves, and the stance gate digest with the rail (39 golden lines, no policy line); retraining is estimated at 85–170 GPU-h (inferred). Not part of the decisions: the heading test's T. rex case fails on r10 until (3), the test-only pull request #642, lands, and since that test runs only at CI's full depth, landing (3) first or with r10 avoids a red full-depth run (inferred, a recommendation). Nothing trains, and the first r10 round is not designed. This "physics r10" is the plant counter's next integer, not the foot of the parked draft #623 that D-D34 named so. Amended 2026-10-25 (the maintainer's decisions on the built branch, in the session's chat, logged in short in the session's log in its entry timed 12:06 UTC): (1) `healthy_z_range`'s 0.70 floor, `floor_impact_threshold_bw` 1.4, the 25° yaw bar and the 2.0° sole-tilt bar stay as they are, their comments updated only (on r10 the floor, not the tail, is the backstop in a level squat: `tail_3`, `tail_4` and `tail_5` sit 0.13, 0.11 and 0.075 m below a level pelvis, against r9's 0.46, 0.55 and 0.60 m; the statue's spawn pop is charged 2.49, 3.67 and 10.02 over the 40 episodes from each of seeds 3042, 7042 and 3162, nothing after step 10; 25° is 1.69× the r10 statue's largest yaw change, 14.8°; seed 3121's pad tilts 2.087° against the 2.0° bar, as r9's 2.061° did); (2) the MJCF's sensor-block comment that still named the deleted `mjx_config.py`, which `configs/plant_versions.toml` note 18 deferred to the next T. rex physics revision, is fixed in this revision (it names `TyrannosaurusRexEnv._cache_ids`; note 20 says so); (3) the three commit messages that later commits corrected are reworded before the first push, every tree unchanged; (4) the pull request is to be opened now, with the heading-test fix's branch merged in (a no-op once that fix is on `main`, inferred) and the `full-ci` label, and driven to green. The evidence is in `docs/investigations/stance_2026_10/stance5/tail_sr3/`; the record is the stance record's §25. |
| D-D45 | Taken 2026-10-25 (the maintainer's decision (3) at 14:13 UTC, after T. rex physics r10 reached `main`, logged in short in the session's log): walking pilot 2 (D-D36; branch `claude/pilot-walk-r9-b`, physics r9, never launched) is retired, and a new pilot is designed once r10 has a stance parent; the branch is kept and never merged. Its parent, `stance3_armLT_s60`'s final, is a physics-r9 checkpoint that `main` refuses since D-D44, so the pilot could have run only on its r9 branch, its result not carrying to r10. The record is the stance record's §26. |
| D-D46 | Taken 2026-10-25 (the maintainer's decisions at 14:13 UTC, (2) and (4), 15:44 UTC, (1) to (4), 16:15 UTC, (2) and (4), and 17:42 UTC), amended 2026-10-26 (01:12 UTC), logged in short in the session's log: report-only telemetry before the first r10 round, appended columns and pages only, training unchanged bit for bit. First (#644, opened on the decision of 17:42 UTC): each evaluation episode's head-tip height and neck and head pitch ("report-only telemetry ... no gate change", the log's words: the stance gate does not see a dropped head) and per-foot hip roll, drawn over training, and the stance5 replication judge's K3 read out over training from its judge files; the Velociraptor and the Brachiosaurus keep the documented gap (they run no evaluation recorder), and a compact per-run K3 extract of the judge files goes to the next records change. Then (#645, opened as built on 2026-10-26): per-actuator commands, tail and body posture, energy and smoothness, drift and slip, four `MjData` extras (joint stops; CoM and capture-point margins with the centroidal angular momentum; tail-base and neck-base wrenches; the head's angular rate) and two views of existing data (the exploration std beside K3, the observation normalization's drift), as pages 13 to 19. No reward, observation, gate, checkpoint selection or digest moves. The record is the stance record's §26. |
| D-D47 | Taken 2026-10-25 (the maintainer's decision (1) at 16:15 UTC, on the read-only sensor assessment asked for at 15:44 UTC, "Maybe include new sensors in the MuJoCo model if that would help"; logged in short in the session's log): no new MuJoCo sensor before the first r10 round; D-D46's extras are computed from `MjData`. A sensor is a plant change (physics r10 → r11 and policy interface r15 → r16, 31 digest lines) and adds nothing in simulation that `MjData` does not hold. |
| D-D48 | Taken 2026-10-25 (the maintainer's decisions (6) at 15:44 UTC, "K3 noise robustness tracked by an in-run sparse panel" (the log's words), and (3) at 16:15 UTC), amended 2026-10-26 (01:12 and 12:54 UTC), logged in short in the session's log: the in-run K3 noise panel. The stance5 replication judge's full K3 (spawns 3042–3061, own-sigma and joint-space noise) is rolled every 1,048,576 steps and once after the final save, synchronously, in an env and generators of its own, into `k3_panel.csv` and `k3_panel.json` only; off by default and enabled at launch; the offline judge's K3 stays the K3 of record. The joint-space sigma is set per seed from the replication's sidecar file. Amended: on in the first r10 round, with its launch validation reworded to the measured form. Shipped in #645. |
| D-D49 | Taken 2026-10-25 (the maintainer's decisions (1) at 14:13 UTC, the budget at 14:30 and 14:45 UTC, D2, D15 and D17 at 16:01 UTC, D1 at 16:03 UTC and D3 to D18 at 19:01 UTC), amended 2026-10-25 (22:18 UTC, D10's wording) and 2026-10-26 (12:54 UTC, the run branch), logged in short in the session's log: the first T. rex training round on physics r10, its criteria fixed and approved before any training. Three runs from scratch, 12M steps each on three Colab L4s ("I really should have been specifying the full 12 million steps that we've done in the past"), seeds 60, 42 and 80 (D2), on arm LTSB of stance5 (`cf80ecb1`) ported to r10 (D1, after the reasoning between the options); the runs start only once both telemetry changes of D-D46 are on `main` (D15); a periodic checkpoint every 262,144 steps, all kept (D3); one pre-registered statistic, the runs with an alarm on any actuator against LTSB's r9 2 of 5 (D10); a parked tail counts against success (D6 to D8); the gate is the six-heading certificate on the final (D9); K1–K4 report only (D12); a durable copy on an orphan branch, never merged (D17); the maintainer checks the compute balance (D18). The run branch `claude/screen-trex-r10-armltsb` carries the port onto `main` with five CoP-band tests re-calibrated on r10 and the collapse reference 4470.4 (D16); on 2026-10-26 the maintainer accepted its 25 configuration-caused test failures as the known set, its one hand-resolved merge hunk and its amended docstring-only commit. The record is the stance record's §26. |
| D-D50 | Taken 2026-10-26 (the maintainer's decisions at 13:06 and 15:47 UTC), amended 2026-10-26 (21:21 UTC) and 2026-10-27 (01:32 UTC), logged in short in the session's log: the first r10 round's runs are read all or nothing (LEAN-R with PRESENCE). At 13:06 UTC the maintainer asked why the criteria carried so much edge-case handling, paused their next revision and asked for a lean variant (at 13:08 UTC: "I'm wondering if we're trying to solve for a problem that really doesn't happen very often. It's very rare where I see a Google Colab session reclaimed"), and at 15:47 UTC chose, after a rule model applied both rule sets to the same worlds, the lean rules: only a run completed in one session with complete records can read no rail, not parked or positive; no run is resumed; a seed gets at most one rule-decided relaunch, for a run that dies before its first checkpoint or that the launch check stops for a fault its files show; any other incomplete run is LOST and never counted, but a tail alarm its ledger shows still counts in the headline (PRESENCE: its ledger, provenance and stage config each read from two agreeing copies, its session and configuration facts in order, no trace or end rail read); a read-only dry run of the completeness check on a finished r9 run folder before launch. The full rules of the criteria drafts of 2026-10-25 and 2026-10-26 are retired, but for what the lean rules keep of the maintainer's decisions on them: no notebook change, and a run whose final save is cut short is LOST (2026-10-25 22:18 UTC, (2)); a run stopped by the trainer's own rule reads RAIL or UNREAD (22:18 UTC, (1)); no resumes (2026-10-26 01:12 UTC); a relaunch's timing and when a launch-check stop is replaceable (12:54 UTC, (2)); and the in-run K3 panel on (12:54 UTC, (4)). Amended with ten rule wordings and one for a mean not formed. The record is the stance record's §26. |
| D-D51 | Taken 2026-10-27 (the maintainer's approval at 05:55 UTC, "Approve", and the launches reported at 14:57 and 15:55 UTC, "I have kicked off all 3 runs"; logged in short in the session's log): the round's criteria are approved before any training step (sha256 `99a44469…`, with the two texts they cite, `0ab5edeb…` and `353f4536…`, and the run branch's head `8737cd3d`), the branch is pushed at that head (never merged), and the compute (about 46.5 L4-hours, about 62 with one relaunch) and fetch budgets are confirmed. The runs `stance6_r10LTSB_s60`, `_s42` and `_s80` were launched at about 14:57 UTC (s42's and s80's time an estimate set no later than their likely start; an L4 runtime and the Colab steps' settings as their launch lines record them, which the launch check does not see) and each passed its configuration-only launch check. Amended 2026-10-27 (18:41 UTC): at the maintainer's request, "could you check the contents? We can do full review once the training is complete", a report-only snapshot of the running runs' files, kept outside the round's folder, a change from the criteria's decision of no interim fetch; the approved rules are unchanged, the round is still read from its final fetch only, and nothing the snapshot shows stops, changes or decides anything. The results are pending. The pre-registration is `docs/investigations/stance_2026_10/stance6/`; the record is the stance record's §26. **Amended 2026-10-28 (its outcome):** at the maintainer's requests and decisions of 23:59 UTC on 2026-10-27 and 01:09, 11:42, 13:31 and 15:05 UTC on 2026-10-28, logged in short in the session's log: at 23:59 UTC, "Can you check out the Google Drive logs so far? You don't need to a full review", a second report-only snapshot of the running files under the first one's terms (outside the round's folder; nothing it showed stopped, changed or decided anything); at 01:09 UTC, "Once training is done, please start the full review and next steps", with each run's end noted from Drive while the maintainer slept, the note saying that it rests on Drive (the run's `artifact_manifest.json` present, its `progress.json` reading ended, and 30 minutes with no new write in its folder), the final fetch at least 6 h after the last such note, then the measurements, the scorer, the full review and next-step proposals; at 11:42 UTC, "You can go ahead and start the analysis", preparation that reads no run data, the final fetch at 12:25 UTC as pre-registered; at 13:31 UTC, after the fetch brief's hand transcription of an inline download was cut off on `01_stance/evaluations.npz` (one transfer error, one download never recorded, one fetch session ended; none of them a file the reading uses), each inline response copied byte for byte from the session's own record of the download by an exact-copy script and checked by id, title, size and sha256 as large files are, the criteria's fetch rules unchanged (a departure from the fetch brief's wording, named in the review); and at 15:05 UTC, "Yes, push the durable copy branch": the first commit of the round's durable copy (D17) pushed to the orphan branch `claude/handoff-stance6`, never merged. The three runs ended in one session each at 12,001,280 steps, their ends noted from Drive at 06:25 UTC; the final fetch ran from 12:25 to 13:55 UTC; all three runs are COMPLETE, and the scorer read, final: line 1, "TAIL RAIL FORMED ON r10 IN TRAINING (ledger)", R 1, U 0, P 1, G+ 1; statistic A 2 of 3 against LTSB's r9 2 of 5, one-sided Fisher p 0.93, nothing significant. `stance6_r10LTSB_s60` is PARKED (`tail_3_pitch_act`'s MEAN_a +0.642 against the 0.60 line), gate PASS (40/40 at each of the six headings), NEGATIVE; `_s42` is RAIL (`tail_3_pitch_act` at +1, 22 ledger tail alarms from 3,932,160 steps), gate FAIL (38/40 at each heading), NEGATIVE; `_s80` is CLEAR, its 6 alarms on `neck_pitch_act`, not a tail actuator, gate PASS (40/40), POSITIVE. An independent check from the fetched files agreed on every class, A, p and the line. The line is an existence result, not a rate: r10 as shipped, with arm LTSB, does not prevent the tail rail (inferred in the review, from the criteria's and the evidence text's wording of line 1). It goes to the maintainer with the diagnostic readouts and names no parent; the review and next-step proposals that followed are report only. Screen runs certify nothing. The maintainer's decisions after the review are D-D52. The record of the outcome is the stance record's §27. |
| D-D52 | Taken 2026-10-28 (the maintainer's decisions at 19:30 UTC, after the first r10 round's review and its next-step proposals, logged in short in the session's log; the design of (1) is pending): (1) the next step is the proposals' first option, a CPU continuation screen from the round's own checkpoints contrasting the entropy coefficient (`ent_coef`) as the runs recorded it with 0; its pre-registration is drafted and brought back to the maintainer for approval, and nothing trains until the maintainer approves it. As proposed (the proposer's numbers, for the pre-registration to fix): continuations of `stance6_r10LTSB_s42` and `_s80` from their 3,145,728-step checkpoints, already fetched and before either run's first alarm, in two arms, the recorded schedule (0.005 falling linearly to 0 at 7,000,000 steps) against `ent_coef` 0 from the start, four training seeds per arm and start, read by the round's ledger rule, on the session's CPUs; the change it makes is a training-loss change for scratch diagnostic continuations only, approved, if at all, with its criteria. (2) A records pull request for the round's results, built locally and brought back before any push: this change. No physics, reward, task, gate or policy-interface change follows, and nothing trains on `main`. Report only, the review behind (1) (inferred there): on r10 the habit is not only a tail habit (one run railed `tail_3` at +1, one parked it, one formed a saturation habit on `neck_pitch_act`), and the entropy bonus is no direct driver (its gradient reaches the action noise only, and bound crossings and re-formed rails came with it at 0), its role in the timing of the early onsets untested, since every run had the same schedule. The record is the stance record's §27. |
| G1 | Chain shape `walk → follow_direction` (the full command set on flat ground) `→ follow_direction_difficult_terrain` (the target deliverable: every species follows a direction on difficult terrain); a commands-free `difficult_terrain` node stays as an optional diagnostic sibling; the recipe label `follow` resolves to the deepest deliverable; `follow_direction_speed` is folded into `follow_direction` (this supersedes the `follow_direction_speed` node in §4.1's manifest sketch and assumption A1); three new stage files per species; the final node is SB3-only (MJX fails closed on live commands and has no terrain). Shapes PR-11's node set. Amended by D-D17: the JAX/MJX runtime is retired (cleanup PR-B), so every node trains on SB3 only; the frozen MJX interface core only feeds the plant contract's digests and probe. |
| G2 | Command set = heading, speed (half to full cruise), stops and restarts, switching every few seconds — the pilots' combined recipe: `follow_direction` carries the full set from the start (`command_config` with `speed_range = [0.5, 1.0]` of the cruise speed) and `follow_direction_difficult_terrain` inherits it. PR-11. |
| G3 | Walker sessions start now on the current notebook, trex first (its r13 chain `20260914_123816` is selected automatically under `TRUNK_FROM = "auto"`), the other five species one at a time, in parallel with the consolidation PRs; the sessions are `docs/NEXT_STEPS.md` §3, and every consolidation PR keeps the chain loop, `TRUNK_FROM`, `WIDEN_FROM`, `RETRAIN_FROM` and the resume cell working so those sessions run on whatever `main` is at the time. Changes nothing in the PR order. **Amended by D-D14 and D-D15 (2026-09-24):** `WIDEN_FROM` leaves the notebook with PR-14a, once both widen sessions were decided; widening stays available on the command line. |
| G4 | The first `terrain_command/v1` gate adopts the pilots' certificate thresholds (20 episodes per terrain family, 20 s minimum horizon, survival LCB 0.80, success LCB 0.60, tracking and settle fractions 0.60; today `configs/behavior_certification.toml`), carried once into PR-13's `[curriculum]` block — never 66 copies — and tightened after the first certified species; a tightening is a gate-digest change (rule 7), so a node certified under the first values is re-judged, not silently reused. PR-13. |

---

## 7. Risks

- **Phase A leaves two vocabularies** ("advancing" for the legacy trio,
  "deliverable" for publication) until the manager and sweeps are re-keyed.
  Mitigation: semantic leaves are judged post-stage like recovery; the
  overlap is documented in `stage_manifest.py` and pinned.
- **The interface bump invalidates every checkpoint for warm-start.**
  Mitigation: D3 — once, early, with the widening tool and recorded
  lineage; the re-panel is budgeted as a real roll with a retrain fallback.
- **The certified walker exists, and the stance two-seed bar is met
  (2026-09-21).** The 20260821 locomotion leg was interrupted at 5.49M of
  8M with 0/109 gate passes at 0.55 m/s
  (`docs/reviews/TREX_REVIEW_2026_08_MERGES_AND_NEXT_STEPS.md`); the
  September run `20260914_123816` (a fresh r13 stance, seed 42, then
  locomotion 8M at 1.07 m/s over 1000-step episodes, `reward_and_length/v1`
  PASS) closed Phase C½ and is what `TRUNK_FROM = "auto"` selects, so
  the follow leaf can branch. The bar: trex stance declares
  `certification_seeds = 2` and seed 42 certifies at r13 — seed 43
  (`20260915_160239`) passed reward and full horizon but failed the
  unsupported-duty rail (mean 0.0323, UCB 0.0350 against 0.02), and seed
  44 certified on 2026-09-20 when the r11 parent `20260815_205206` was
  widened and re-paneled as `20260920_010912` (`WIDEN_MAX_REVISION_GAP = 2`,
  `SEED = 44`), whose bundle records trex stance as `2 runs of 2 seeds`.
  A failed panel is a measured deficit — re-rolling it does not help,
  another seed does — so the fallback was a fresh r13 stance at
  `SEED = 45` (about 13 h), not needed.
- **LCB gates change the meaning of thresholds** (20/30, not 15/30, for
  0.5). Mitigation: freeze thresholds attainable-not-aspirational from the
  first pilot, as recovery did.
- **Seed lottery multiplies** with four deliverables per species.
  Mitigation: §4.5 lands before any deliverable is called certified; the
  catalog says "provisional" out loud.
- **The publication layer is the largest change**; if it lags, leaves
  accumulate as unpublishable pilots. Mitigation: Phase A ships it before any
  new leaf trains.
- **MJX only warns on unknown `[env]` keys** (`mjx_env.py:136-153`).
  Mitigation: `command_mode != "none"` raises on MJX until implemented.
  Retired by D-D17 (cleanup PR-B, 2026-09-28): the MJX environment, its
  `[env]` handling and its trainer are deleted, so no MJX run reads an
  `[env]` key or a command mode, and `load_stage_config` now refuses a
  `[jax]` stage table. Only the frozen MJX interface core stays, which the
  plant contract hashes and its probe runs; nothing trains on it. A backend
  added back meets this risk again (D-D17's entry conditions).
- **About fifteen tests pin the derivations being replaced** (advancing trio,
  carried load path, position-keyed shaping, terminal selection, notebook
  cells). Each is re-pinned to the new invariant, never deleted.
- **Naming overclaims**: "hunt" certifies reach-and-bite of a static target
  spawned 2–6 m ahead; pursuit of moving prey (ROADMAP Stage 4) is a future
  hunting mini-stage and is not implied by the label.

---

## 8. Invariants to pin

1. A v1 or synthesized manifest under the v2 reader yields the same entries,
   the same advancing trio, derived edges equal to "previous advancing
   entry", and one deliverable equal to the last advancing entry; the
   committed per-species v2 files are the only place stand and walk become
   deliverables.
2. `warm_start_from` must name an earlier entry; self and forward references
   are fatal; legacy rewrite and reorder rules unchanged.
3. Entry shaping fires iff the node has an edge and the load is
   `initialize_next_stage`; never on a same-stage resume; never on a root.
4. `train()` refuses an `initialize_next_stage` load whose recorded parent
   stage differs from the declared edge.
5. A run whose leaf gate fails writes a bundle whose certified trunk
   deliverables are published, whose status is `partial`, and whose
   `selected_model_path` is a certified deliverable, never the failed leaf;
   a walk-only run writes a valid bundle.
6. Ancestor reuse refuses a candidate whose gate did not pass, whose plant
   identity mismatches, or whose recorded task hash differs from the current
   stage config, or whose verdict's recorded gate-configuration digest
   differs from the current stage config's (rule 7, D-A22), or which lacks
   one (D-B6); a directory re-judged under the current gate is reusable
   although it trained under another (D-B8; `test_ancestors.py`).
7. `widen_checkpoint`: the padded columns are exactly zero, actions on
   zero-padded observations are allclose to the parent's over a seeded
   rollout, one PPO update completes from the widened checkpoint, and
   lineage records the parent hash.
8. Normalised command values are O(1) from the first step after a
   command-mode load.
9. `command_mode != "none"` raises on MJX until the reward path exists; the
   plant contract's SB3/MJX observation parity covers the widened (zero)
   command segment from Phase C on.

   Under D-D17 (cleanup PR-B, 2026-09-28) no MJX environment or trainer
   exists, so the first half holds vacuously and is an entry condition for
   adding a backend back; `command_frame.validate_command_mode` keeps the
   MJX refusal until PR-9. The second half still holds: for trex,
   velociraptor, brachiosaurus and dibothrosuchus the plant-contract probe
   runs `build_mjx_observation` on the frozen MJX interface core, and
   `validate_environment_plant` requires SB3/MJX observation equality, with
   the non-zero probe command of D-C4. *Consolidation PR-9 (2026-10-05)
   rewrote `command_frame.py` and kept the MJX refusal: only the SB3 branch
   went.*
10. For every new gate kind: the fail-closed dispatch test and the
    "what code would have to be deleted for it to stop being consulted"
    test (`test_gate_dispatch_fail_closed.py`; for `task_success/v1`,
    `TestTaskSuccessGateIsConsulted`).
11. The follow panel contains the walk command and is paired against the
    command-blind walker; the evidence CSV carries the command columns.
12. The notebook chain loop: AST pins for the `BEHAVIOR` knob, the reuse
    rule, the no-position-inference of load mode, and the escape hatch.

---

## 9. Relationship to the gap review and the roadmap

- **CF2 / SS2 / CF3** (behavior-stage gate): resolved by §4.4. CF2 is
  resolved by *removing* the velocity term from the hunting gate rather than
  by the review's suggested windowed/peak metric — the recipe framing makes
  the term's misplacement structural rather than a tuning question — and
  the windowed metric becomes the walk gate's later refinement. Landed
  2026-09-13 in #534 (WS-B1 + WS-B2);
  the review carries dated Status lines under each finding.
- **SS1** (seed multiplicity): resolved by §4.5. Landed 2026-09-13 in the
  seed-replication PR #535 (WS-B4); the review carries its
  Status line. Trex stance published at 1 of 2 seeds — provisional — until
  the seed-44 parent `20260815_205206` was widened and re-paneled under r13
  (the r11 verdicts cannot be re-backfilled into reuse; `docs/KNOWN_ISSUES.md`,
  the Phase C entry) as `20260920_010912` on 2026-09-20; its bundle records
  trex stance at 2 runs of 2 seeds (2026-09-21).
- **DU1's open question** (SAC warm-up in the notebook): closed by §4.7.
- **ROADMAP "Turning and steering"** (`docs/ROADMAP.md:253`) is the
  follow-direction leaf; **"Hierarchical RL architecture"** (`:320`) and
  **"Stage 4: Prey Pursuit"** (`:259`) are future leaves and mini-stages on
  the same DAG, not part of this plan.
- **`docs/STAGE1_SPLIT_PLAN.md` §4's `terminal:` field** ships as
  `deliverable`.

---

## 10. Open questions to measure, not decide

- Does the widened stance checkpoint reproduce its 40-episode panel under
  the new interface? (Expected yes up to float rounding — the widen tool
  already pins a 0.0 max |padded column| and an action delta ≤ 1e-6 over a
  seeded rollout; the panel itself is verified by the WS-C4 re-panel of both
  stance parents, owed: `investigations/TREX_STANCE_WIDENED_INTERFACE_2026_09.md`.)
  Answered 2026-09-20/21: yes, to every printed digit, for both parents that
  were widened — trex seed 44 (`20260920_010912`, r11 → r13: 3408.3 ± 88.5,
  full-horizon 1.0000, duty 0.0069 / UCB 0.0117, the r11 certificate's
  numbers) and compsognathus seed 42 (`20260921_203149`, r1 → r2: 2801.6 ±
  51.2, duty 0.0131 / UCB 0.0141); the seed-42 trex re-panel was not needed,
  since seed 42 certified from scratch at r13 in `20260914_123816`.
- Which direction/terrain thresholds survive contact with a certified
  species? G4 adopts the pilots' certificate thresholds (20 episodes per
  terrain family, 20 s minimum horizon, survival LCB 0.80, success LCB
  0.60, tracking and settle fractions 0.60) for the first
  `terrain_command/v1` gate and tightens after the first certified species;
  a tightening moves the gate digest (rule 7), so nodes certified under the
  first values are re-judged, never silently reused.
- What tracking tolerances and settle/dwell windows does the walker-derived
  follower reach at 3M steps? (Frozen from the Phase D pilot.)
- Does the command-blind walker's paired null show forgetting of straight
  walking in the follower? (The Phase D panel answers it.)
- Is a cruise-window velocity gate for walking worth its new evidence
  columns, or does the 1.0 m/s episode-mean gate with the 750-step length
  floor suffice? (Decide after the first certified walker under Phase C.)
- Where does the hunting bar settle? `min_success_lcb = 0.5` is
  provisional (D-B2): re-freeze it attainable-not-aspirational from the
  first Phase-B pilot's `evaluation_selected.csv` (20/30 clears at 0.5006,
  19/30 does not), then update the TOML comment and §4.4 and re-judge —
  never retrain — any verdict minted under the provisional bar.
- Is 1.0M the right arming delay for the hunting collapse detector?
  `collapse_peak_warmup_timesteps` is a judgment (D-B5): replay the first
  full hunting run's `evaluations.npz` through
  `EvalCollapseEarlyStopCallback` and re-derive it, and re-measure the
  statue (`zero_action_baseline.py trex:3 --episodes 40 --seed 3042`)
  whenever a behavior reward weight or the plant moves, re-deriving the
  361 rail and the 602.0 reference together.
- Should the compsognathus pair's locomotion and behavior TOMLs state
  their collapse-floor omission per stage? Today the committed-config
  loops exempt them on the species-level decision their stance TOMLs
  record (D-B14); a maintainer decision to add the relative pair or the
  unarmed statement to those stages would let the exemption narrow to
  per-stage statements.

**Phase B items owed, deferred by the maintainer on 2026-09-13** (recorded
here with the Phase C docs pass; the priority is stance → recovery → walking
before the hunting set-up, D-C15). To revisit later, in the maintainer's own
words: re-freeze the provisional `min_success_lcb` bar from the first hunt
pilot (D-B2, the bullet above); the replay warm-up delay re-measure
(`collapse_peak_warmup_timesteps`, D-B5, the bullet above); the
compsognathus collapse-floor statement (D-B14, the bullet above); and the
maintainer log-tree actions — re-backfill verdicts with `--force`, republish
stance bundles with the panel role. The log-tree actions changed shape under
Phase C: for every pre-Phase-C run the re-backfill and the republish are
dead (KNOWN_ISSUES), and their replacement is widening both stance parents
into new runs and re-paneling them there (WS-C4 Session 1 and the seed-44
repeat — as of 2026-09-19 only the seed-44 repeat was still needed, seed 42
having certified from scratch at r13 in `20260914_123816`; it ran 2026-09-20
as `20260920_010912` and passed), whose bundles
carry the `certification_panel` role from the start;
the pre-Phase-C directories stay on the log tree as history. Both certified
parents are r11 archives, two revisions behind r13, so those sessions set
`WIDEN_MAX_REVISION_GAP = 2` (D-C17: the gate's default bound of 1 refuses
them; under 2 it still checks every other identity field, and the crossed
r11 → r12 bump was fingerprint-only — KNOWN_ISSUES, the Phase C entry).
