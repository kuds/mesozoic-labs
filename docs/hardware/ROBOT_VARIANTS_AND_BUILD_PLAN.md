# Same-species robot variants and a mini-to-50 cm build plan

**Status: proposed architecture and planning estimate, 2026-10-24.** This document
does not implement a registry migration or register a new training environment.
It follows the [size trade study](TREX_SIZE_TRADE_STUDY.md) and reviews the code
at commit `fdbc6345012d4284e1f80e5e9fd4a5fbb59caae2`.

**Recommendation:** keep one Tyrannosaurus Rex species, retain `anatomical` and
`robot` as the embodiment kinds, and add an explicit **variant** for each
concurrently maintained robot design. Keep revision counters within each
variant. Build a deliberately limited mini to learn the hardware/control workflow,
then make 50 cm the main character platform. The 80 cm design remains a study.

The mini is a learning stage, not a requirement to finish every possible feature
before starting 50 cm. Its motor qualification, telemetry and reproducible test
workflow are valuable even if it does not become the final public demonstrator.
If its small motors or packaging require a disproportionate redesign, reassess
instead of spending the larger robot's budget on a miniature detour.

## What the current repository supports

The current [naming contract](../SPECIES_NAMING.md) and
[ModelId](../../environments/shared/model_id.py) define a model as a pair:
`(species, embodiment)`. The [schema-3 manifest](../../configs/species_manifest.toml)
has one model path/entrypoint per embodiment. T-Rex has only its anatomical
model registered; the mini is a standalone example. Compsognathus has a registered
robot. Concurrent sizes are not already supported by the runtime.

A read-only probe of the exact current `ModelId` implementation confirmed that
adding `variant = "mini"` or `variant = "medium"` to a record has no effect:
`from_record()` reads only the two existing fields. The resulting identities,
config paths and Gymnasium IDs are equal, and serialization drops `variant`.
Using `robot_mini` as the embodiment is rejected by the closed vocabulary.
**An extra metadata tag alone therefore cannot implement this plan.** Existing
plant physics/interface hashes already protect against incompatible checkpoints;
the probe does not show that arbitrary cross-robot policies can currently load.

There is a second integration issue: the
[T-Rex base environment](../../environments/tyrannosaurus_rex/base_env.py) assumes
21 anatomical hinges, passive toe joints, anatomical foot sensors, a fixed-head
bite proxy and its own observation/action mapping. The mini has 14 or 15 motors;
the larger concepts propose 18. The robot cannot become a training environment
simply by replacing `MODEL_PATH` in an anatomical subclass.

## Identity: species, embodiment, variant, then revisions

| Layer | Example | Meaning |
| --- | --- | --- |
| Species | `tyrannosaurus_rex` | Biological inspiration and species-level presentation |
| Embodiment | `robot` | Engineered hardware rather than the anatomical research model |
| Variant | `mini`, `medium`, `large` | Separately maintained design with its own dynamics and interface |
| Physics revision | `3` within `mini` | Changed mass, geometry, contact or actuator dynamics |
| Policy-interface revision | `2` within `mini` | Changed observation/action meaning, ordering or dimensions |
| Visual revision | `4` within `mini` | Render-only change leaving dynamics/interface unchanged |
| Physical unit | A recorded unit serial | The individual assembly and its calibration/firmware records |

Nominal size is metadata: mini currently 25.8 cm, medium targets 50 cm, large
targets 80 cm. Measured height belongs in validation reports. Avoid encoding
`50cm`, a motor brand or `rev2` into the primary size-family name by default:
a small height correction should not rename every run and result. Use full
display labels such as **Tyrannosaurus Rex (Robot · Medium · 50 cm target)**.

A variant is a complete physical/control configuration, not a collection of
unrecorded options. The fixed-tail and motorized-tail minis have different
actuator counts; if both become registered models, use `mini` and `mini_tail_yaw`,
grouped under `size_class = "mini"`. A different wagging policy on unchanged
mechanics is a behavior/configuration fingerprint, not another hardware variant.
Cosmetic colors alone also do not create variants.

The two medium actuator options remain alternatives in a study until a choice
is made. If both are maintained as executable comparisons, give them distinct
variant IDs. If one replaces the other in the same development line, record the
physics/interface revision change and preserve old evidence at its recorded
commit. Do not generate every possible component combination.

## Proposed layout

Use one derived path rule for named variants: append `variants/<variant>` to
the existing species/embodiment directory. Legacy models retain their paths.
These are proposed paths; empty packages are not needed now.

| Material | Proposed location |
| --- | --- |
| Robot-specific T-Rex environment base | `environments/tyrannosaurus_rex/robot/base_env.py` |
| Mini environment adapter | `environments/tyrannosaurus_rex/robot/variants/mini/env.py` |
| Mini model and physical specification | `environments/tyrannosaurus_rex/robot/variants/mini/assets/` |
| Medium / large packages | Corresponding `robot/variants/medium/` and `robot/variants/large/`, once executable models exist |
| Stages, recipes, limits and recovery calibration | `configs/tyrannosaurus_rex/robot/variants/<variant>/` |
| CAD, drawings, wiring, dated costed BOM and assembly sources | `hardware/tyrannosaurus_rex/<variant>/`, when those source files exist |
| Build instructions and hardware evidence index | `docs/hardware/robots/tyrannosaurus_rex/<variant>/` |
| New runs | `<LOG_BASE>/tyrannosaurus_rex/robot/variants/<variant>/<algo>/<run>/<stage>/` |
| Curated results | `results/tyrannosaurus_rex/robot/variants/<variant>/<algo>/` |
| Website | One T-Rex category, Anatomical page, Robot overview and variant pages |

Keep one authoritative structured physical specification per variant, packaged
beside its model. It names joints/order, actuators/limits, mass allocations,
sensors and calibration requirements. Check CAD exports and MJCF against it;
do not independently maintain several contradictory mass/limit tables. Prices
live in a dated cost ledger and should not change a physics digest. Rendered
eye color is visual; a printed shell or camera mount that changes mass is physical.

Installable distributions must include the environment adapters and required
model/specification assets. CAD and large videos are not runtime dependencies.
Keep small authoritative specs/reports in Git; use hashed release-asset manifests
for large fabrication exports/evidence. The plant source closure must cover any
physical specification that model generation consumes.

Initially leave `examples/mini_tyrannosaurus_rex/` intact. When promoting the mini,
move its authoritative model and reusable checks into the variant package; leave
a small example using that package. Avoid a second editable XML copy. Preserve
existing diagnostic reports as historical evidence at their paths/commits.

## Runtime and manifest migration

Extend the existing typed `ModelId` with an optional `variant` field. A missing
variant is valid only for a manifest entry declared as a legacy singleton; it
must never select whichever robot was added most recently. All named-variant
execution paths carry the complete identity rather than parsing filenames.

Proposed selector, **not supported by the current CLI**:

```sh
python -m environments.shared.train \
  --species tyrannosaurus_rex --embodiment robot --variant mini \
  train --stage stance
```

Omitting `--variant` for a variant family should list choices and fail. Existing
anatomical and Compsognathus singleton commands keep working. The notebook gets
a variant selector when applicable. Multi-model tools need an explicit selector;
do not overload a two-token argument with an ambiguous optional third token.

Advance the species manifest schema with mutually exclusive shapes: an existing
singleton embodiment entry, or an embodiment containing named `variants`. Do
not expose a parent runtime entry and named children as ambiguous equivalents.
Distinguish study-only entries from executable entries. Only executable entries
with real assets, adapters and complete runtime metadata register and appear
in training selectors. Study-only medium/large entries can appear on a roadmap
without fake environments, results or empty stage graphs.

Derive a separate Gymnasium ID for executable variants, for example
`MesozoicLabs/TyrannosaurusRex-Robot-Mini-v0`, plus package, asset and display names.
Keep singleton IDs, paths and labels unchanged. The Gymnasium `v0` suffix is not
a hardware revision. Add per-variant plant-version and generated-manifest entries;
each variant has its own counters, with the legacy singleton representation retained.

The change must cover the whole identity chain:

- Registry/constructors, equality, ordering, serialization, paths, identity-keyed
  caches, stage/config lookup and installed assets.
- Plant identity/digests, checkpoints, VecNormalize sidecars, recovery calibration,
  task fingerprints, curriculum ancestry and resume checks.
- Run/provenance records, exported summaries, replay/video names, W&B labels and
  Drive indexing.
- Generated catalogs, notebook selectors, website readers and tests that currently
  enumerate pairs or glob two directory levels.

Extend the existing [plant identity](../../environments/shared/plant_contract/identity.py)
and [result provenance](../../environments/shared/result_bundle/provenance.py).
Bump relevant serialized schemas and check nested plant/run/recipe identities
against each other. Reject variant-bearing records claiming an old identity
schema instead of ignoring the field. Preserve known legacy singleton records
through an explicit versioned reader; never assign an old unqualified robot
record to a new named variant by guesswork.

Keep existing singleton digest payloads unchanged where possible. Review and
version any unavoidable semantic changes under the plant-contract process; do
not refresh snapshots just to make tests green. Preserve the frozen MJX core.
Read historical artifacts at their recorded commit when exact migration is not
possible. The custom identity sort order must handle legacy and named entries
without comparing `None` directly with a string.

Reject cross-variant checkpoints and normalizers even when dimensions match.
Intentional transfer experiments record source and target and produce new
target-specific artifacts; they are not normal resumes and inherit no certificate.

## Share behavior tools; specialize mechanics

Share training runners, result bundles, gate mathematics, diagnostics,
controller interfaces and reusable actuator/sensor models. Share character
intent such as a curious head tilt or gentle wag, with per-variant limits and
timing. Miniature weights, PID gains, thresholds and normalization are not
assumed transferable to 50 cm.

Start with a T-Rex robot base alongside the anatomical base, using the shared
`BaseDinoEnv` and reusable pure helpers where appropriate. Variant adapters
declare joint/sensor/body layouts and control contracts. Avoid changing anatomical
observations/rewards to accommodate the robot during this integration. Extract
a generic robot-biped base only when actual reuse warrants it. Drivers, IMU
processing and system identification belong in shared infrastructure, not copies
inside each species.

This requires explicitly amending the contribution rules and
[identity-contract tests](../../environments/shared/tests/test_model_identity_contract.py)
that insist on direct inheritance from the existing species base and the current
package/class/path formulas. Verify unchanged anatomical behavior; do not replace
these contracts with arbitrary exceptions.

Keep semantic behavior labels `stance`, `recover`, `walk`, with their actual
stage IDs `stance`, `recovery`, `locomotion`. Use explicit per-variant configs and
freeze resolved configurations in runs. Initially prefer clear small files over
cross-variant inheritance/override chains. Calibrate physical thresholds for each
robot. A museum character need not advertise the anatomical `hunt` task; adapt
capability enumeration rather than copying unsupported recipes to satisfy a count.

## Tests and readiness

| Check | Purpose |
| --- | --- |
| Identity and artifact isolation | Different variant paths/IDs; reject cross-loaded checkpoints, normalization, calibration and ancestry even with equal dimensions |
| Legacy compatibility | Preserve promised singleton commands, artifact readers, installed assets and reviewed digest behavior |
| Variant mechanics | Verify joint ordering, mass/inertia, clearance, support geometry and actuator envelopes against the specification |
| Control regression | Unsupported stance, perturbation and walking/turning trials with head/arms/tail poses; detect deliberate mass/torque regressions |
| Evidence integrity | Record variant, revisions, config, protocol/seed, support condition and physical unit when applicable |
| Catalog readiness | Study-only designs stay out of training; mini results never mark medium/large successful |

Run inexpensive checks on relevant PRs and broader seeded gait tests when a
trained policy and meaningful acceptance protocol exist. Shared robot-code edits
test every executable robot variant; variant edits test that model plus identity
and catalog contracts. Update CI path filters for hardware directories and example
migration. Heating, endurance and fault behavior still require physical tests.

Track simulation readiness and hardware evidence separately. A supported display
animation is not walking, and simulation is not hardware evidence. The current
mini has balance/diagnostic checks but fails all recorded 30-second walking cases.
Medium and large have analytical studies only.

## Implementation sequence

1. **Document:** adopt the identity/layout decisions and preserve current experiments.
   This document is the proposal, not an implemented migration.
2. **Identity and artifacts:** update registry, schemas, routing, selectors and
   compatibility checks. Prove legacy behavior and variant isolation first.
3. **Mini integration:** add the robot adapter, promote one complete mini
   configuration, port mechanical checks and declare stages/gates. Wire it into
   notebook/logging/catalog. A tail topology change cannot hide behind a boolean.
4. **Medium integration:** after actuator/layout selection, add its physical
   specification, model and adapter. Reuse infrastructure, then establish new
   calibration and hardware evidence.

Use focused implementation PRs. Avoid combining a broad identity migration,
50 cm mechanical design and new gait claim. Reserve **32–56 focused hours** for
initial identity/core integration, included in the mini total below. Unexpected
artifact compatibility work can enlarge that estimate.

## Effort and calendar assumptions

Assume **15–20 focused hours/week dedicated to this robot effort**, a provisional
**November 2026** start, an existing development computer, and fabrication access
or a printing service. Buying a printer and paid engineering are outside the
cost study. Estimates include hardware iteration and controller work; they are
not supplier lead-time quotes or guarantees of walking success.

| Mini phase | Focused hours | Exit evidence |
| --- | ---: | --- |
| Variant identity and core integration | 32–56 | Artifact isolation, legacy checks and basic robot adapter |
| Layout and actuator bench | 28–48 | Real component fit, measured mass and usable thermal/load envelope |
| Assembly and calibration | 40–64 | Joint zeros/directions, timing, IMU/contact telemetry and supported fault tests |
| Gait and sim-to-real iteration | 64–112 | Measured model, unsupported stance and repeatable short stepping trials |
| Gestures and repeatable demonstration | 24–40 | Arms/tail within limits, repeatable routine and documented reset |
| **Total** | **188–320** | Useful bounded learning prototype; walking readiness requires passing tests |

This is approximately 9–21 working weeks. Plan **12–24 calendar weeks, or about
3–6 months**, including procurement/reprints and overlap during waits. It does
not guarantee a polished untethered walker by week 24. A separately selected
motorized-tail configuration needs its own tests before becoming the demo target.

| 50 cm phase | Focused hours | Exit evidence |
| --- | ---: | --- |
| Actuator comparison and full packaging | 45–80 | Motor choice, thermal data and reconciled CAD/mass/power budget |
| One six-axis leg and fixture | 45–90 | Loaded structure, contact, cable travel and actuator response |
| Two legs, pelvis and electrical integration | 55–105 | Calibrated hardware, supported stepping and dependable telemetry |
| Locomotion and endurance | 110–240 | Repeated unsupported walking/turning, thermal/runtime measurements |
| Character parts and demonstration | 45–85 | Repeatable complete-character routine and service/transport process |
| **Total** | **300–600** | Independent qualification; mini policies are not assumed reusable |

This is approximately 15–40 working weeks. Plan **20–44 calendar weeks, or about
5–10 months**, including procurement/fabrication. This refines the earlier 4–9
month effort estimate by making the calendar allowance explicit. Reusing the mini
workflow helps but does not remove new locomotion and thermal validation.

At ten hours/week, mini effort becomes about 19–32 working weeks and medium
30–60, before waiting. Courtside Dynamics, anatomical training, writing and other
projects reduce the hours available here. Do not budget the same weekly hours twice.

## Example calendar and overlap

| Period from a November 2026 start | Primary focus | Possible overlap |
| --- | --- | --- |
| November–December 2026 | Mini identity/interface, motor bench and layout | Medium packaging requirements and supplier/package checks |
| January–February 2027 | Mini assembly, telemetry, stance and stepping | Limited medium motor bench during genuine waits |
| February–April 2027 | Target window for a useful mini and move-on review | Finalize medium actuator choice and layout |
| Spring–summer 2027 | Medium one-leg, pelvis and full-frame work | Retain mini for regression and character experiments |
| Summer–late 2027 | Medium gait, thermal/endurance work and gestures | Prepare repeatable videos and an educational routine once supported by evidence |
| Early 2028 fallback | Additional control/hardware iterations at the upper effort range | Reassess scope instead of promising an event date |

Combined effort is **488–920 focused hours**: about 24–61 working weeks at the
stated allocation, before non-overlapped delays. Treat **late 2027 as a working
target for a useful 50 cm demonstrator**, with early 2028 a plausible slip.
Faster completion is possible; it is not the public-event assumption. The month
bands are a scenario, not phase deadlines.

Overlap only tasks that fit otherwise idle waiting/training time or the same
weekly budget. If 20% of a week goes to medium work, mini gets the remaining
80%. Avoid two full mechanical builds at once. Keep 80 cm documented until the
50 cm workflow and a concrete demonstration use case justify another project.

## Decision gates and staged spending

**Before assembling a complete mini:** demonstrate selected leg-actuator duty
and packaging. Its USD 2,100–3,100 allowance includes provisional upgrades, not
a qualified BOM. As a planning cap within that total, aim for an initial
**USD 300–600** bench/sample-part stage; verify an actual cart and fixture list
before purchase. A simulated torque cap is not motor qualification.

**Mini review after about 8–12 active weeks:** examine measured actuator margin,
assembly feasibility and remaining controls work. Preserve useful telemetry and
identification results. If tiny mechanics require another major motor redesign,
compare the next expense with the medium bench instead of continually extending
mini scope. This is a review point, not a required walking deadline.

**Before buying a full medium frame/motor set:** finish the **USD 700–1,200**
loaded-motor/packaging stage, select the economical or stronger motor budget,
and establish a credible CAD/mass/thermal case. The mini need not be cosmetically
finished before that bench work. If it still cannot walk, carry the unresolved
control risk explicitly; scaling is not its remedy.

**Before public walking claims:** predeclare a physical protocol. An initial
target to refine during integration is ten 60-second unsupported walking/turning
trials without falls, followed by repeated three-to-five-minute routines with
temperature/voltage monitoring. A slack catch fixture may arrest a fall but must
not support weight; intervention fails that trial. This small panel is a milestone,
not a reliability guarantee or unattended-public-operation qualification. Label
supported demonstrations accurately.

The project allowances total about **USD 7,700–11,100** for mini plus economical
medium, or **USD 11,300–14,800** with stronger medium motors, before the exclusions
in the size study. Those sums conservatively include basic bench equipment in
each project. Deduct only identified reusable equipment after inventory; mini
motors should not be assumed reusable in the 50 cm robot. Staged spending is
part of these totals, not an extra fee.

## Review basis

The review covered naming/contribution contracts, manifests, registry/registration,
model and plant identity, config/stage routing, result provenance, behavior
checkpoint handling and identity-contract tests. The pair-only identity probe
ran against the retrieved module without changing production code. Phase-hour
totals were reconciled with weekly capacity and prior cost studies. Runtime
migration tests and larger robot hardware/gait tests remain future work.

## 2026-10-25 amendment: one-metre T-Rex as the large concept

The selected next character is now a long-bodied, dinosaur-like **one-metre T-Rex**,
rather than a mandatory 80 cm stepping stone or a biologically exact raptor.
The earlier size references above record the prior proposal.

The [one-metre example](../../examples/tyrannosaurus_rex_100cm/README.md) has its
own generator, MJCF, component ledger, parts budget, sensor placement and basic
stance checks. Revision 2 follows the user's WowWee reference with a longer
skull, horizontal body, 860 mm tail and raised ankles: 1.000 m tall, 2.104 m
long and 33.210 kg of allocated mass. It shares the T-Rex species with the mini
and medium concepts, while using its own surfaces and independently sized
hardware. The mini and medium retain silhouette C. It remains a
checkout-only example; `large` is a proposed future variant identity, not a
registered runtime selector. This does not implement the variant migration or
the generic walking-feasibility solver.

The revised large budget is about $11,500 hardware / $17,300 first prototype
before tax, freight and engineering labor. Stronger pitch/knee drives and
explicit cooling allocations support the basic standing result, but thermal
duty, supplier-CAD fit and single-support/stepping are unqualified. A snout
camera clears the longer jaw; the eyes stay cosmetic. An instrumented large
leg and the power/control bench should precede the complete large chassis.

The development sequence remains mini → medium → large. The medium can answer
questions about calibration, wiring, state estimation and serviceability before
the metre build commits to its heavier actuators and structure. Advancing size
does not transfer walking qualification.
