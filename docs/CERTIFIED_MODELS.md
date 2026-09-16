# Shared certified models

The SB3 notebook can select a compatible certified parent automatically, save
the complete selected bundle inside the new training run, and compare newly
trained candidates with the current recommendation. Select the species and
behavior as usual; model paths are optional in automatic mode.

## Notebook controls

| Setting | Default | Meaning |
| --- | --- | --- |
| `SOURCE_SELECTION` | `"auto"` | Select compatible recommended ancestors or behavior sources automatically. Use `"manual"` to require explicit behavior paths and disable automatic canonical ancestor selection. |
| `CERTIFIED_LIBRARY_ROOT` | `""` | Use the `certified` directory beside `logs`; set another path to share a different library. |
| `PUBLISH_CERTIFIED` | `True` | Preserve candidates and their evidence in the shared version history and consider eligible candidates for recommendation. |
| `CERTIFIED_COMPARISON_EPISODES` | `50` | Number of paired comparison episodes **per model**. Both models run the same cases. |

With Google Drive, the default library is
`/content/drive/MyDrive/mesozoic-labs/certified`. Local runs use
`<repository>/certified`. Training output remains in the existing `logs` tree.
Building the notebook run plan creates neither a run nor a library directory.
Source resolution happens after the runner knows the exact task identity.

### Canonical behaviors: stand, walk, hunt and stage IDs

The chain resolves root first. For each ancestor it tries:

1. A valid checkpoint already trained in the current run.
2. The explicitly selected `TRUNK_FROM`, when provided.
3. A compatible shared recommendation, when automatic selection is enabled and
   no explicit trunk was provided.

The selected target still trains in the new run. `RETRAIN_FROM` continues to
force training of that node and its descendants; `WIDEN_FROM` retains its
existing validation and re-judging requirements. An ancestor recommendation
must match the exact parent model and normalization pair selected earlier in the chain. A
recommended child of a different parent is not silently substituted.

If no compatible recommendation exists, the canonical chain trains that
ancestor here. Invalid or tampered evidence is refused. Manual trunk selection
does not silently fall back to an unrelated library source.

### Direction and terrain behaviors

Leave **both** `BEHAVIOR_CHECKPOINT` and `BEHAVIOR_VECNORMALIZE` blank to use
automatic selection:

- `BEHAVIOR_LOAD_MODE="prepare"` selects a compatible certified locomotion
  parent for the chosen species.
- `BEHAVIOR_LOAD_MODE="resume"` selects the exact behavior's recommended
  bundle and continues its remaining training budget.
- `BEHAVIOR_EVAL_ONLY=True` selects the exact behavior for scoring.

Provide both paths to override automatic selection. Resume and adaptation
still require a matched `bundle.json`; selecting only one source path is an
error. `BEHAVIOR_LOAD_MODE="adapt"` requires an explicit source pair because
changing from another behavior is an intentional task change.

A missing behavior recommendation or locomotion parent stops the behavior
runner with a reason. Train and certify the required source, or explicitly
select a compatible matched pair. Automatic preparation does not launch a
hidden locomotion training run.

## Complete copies in each training run

Automatic selection copies the **complete selected version** beneath the new
run's `certified_inputs/`, including its model, normalization, saved evidence,
configuration, manifests, and available videos and maps. Canonical manual
trunk selection also makes a complete local copy of the selected stage and
its required ancestry evidence. Training handoffs use those copied paths.
An explicitly pinned shared-library version is copied in the same way.

The copy retains the source identity and selection record. Evidence hashes
are checked again after copying. The original run and shared version remain
available, and moving a new training run does not require its videos to be
loaded from the original source directory. Newly trained models and ordinary
run artifacts continue to be saved in their usual locations.

## Candidates, certification and independent training seeds

A version records the model and its evidence together. Publication preserves
passing and failed attempts instead of overwriting an older model. Existing
versions are immutable; re-publishing identical evidence is safe.

Passing one run's gate is distinct from having sufficient replication for a
recommendation. A passing candidate can remain **provisional** while its
configuration lacks the required independent training seeds. Repeated
evaluations of one checkpoint do not create independent training runs.
Duplicate seeds and duplicate model evidence cannot manufacture replication.
Failed attempts remain visible in the candidate history.

Canonical checkpoints record the original stage seed, exact parent pair, and
the optimizer update count at stage entry. A same-stage resume preserves that
origin; changing its requested seed cannot create another independent
replica. A newly trained stage must complete optimizer updates before it can
be published. Legacy checkpoints without a verifiable origin remain available
for explicit manual training and reuse; automatic publication refuses to
invent their training history.

Exact continuation retains the original training seed and parent model and
normalization pair. A new sampling seed during continuation cannot count as
another independent training run. A model must also have completed optimizer
steps in its recorded stage; evaluation-only copies cannot create replicas.

Compatibility groups separate species, algorithm, task and environment
identity, gate settings, and required ancestry. Evidence from a different
group cannot qualify a candidate. Resolving a recommendation verifies its
files and the evidence supporting its replication again.

`QUICK_TEST` does not lower a gate, reduce the required number of independent
training seeds, or turn a short replay panel into certification. Canonical
quick tests skip recommendation benchmarking. Behavior quick tests skip the
separate certification panel and shared publication, and save that reason in
their run record. Use `PUBLISH_CERTIFIED=False`
when only local diagnostic artifacts are wanted.

## Comparing with the current recommendation

The default comparison runs **50 episodes for the candidate and 50 for the
incumbent**, using the same fixed seeds and task cases. For the general
difficult-terrain behaviors, five enabled terrain families therefore receive
ten cases each per model. A different episode count is configurable; the
recorded protocol identifies the exact panel and terrain coverage.
The minimum is two episodes for canonical comparison and two per enabled
terrain family for behavior comparison (ten for all five families).

This setting is separate from `BEHAVIOR_EVAL_EPISODES`, which controls the
diagnostic replay panel, and from the distinct training seeds needed for
replication. Increasing comparison episodes does not replace independent
training runs. When comparison settings change, the incumbent must also be
evaluated on that panel. Incompatible older measurements cannot count as a win.
Verified incumbent scores are reused when its exact saved pair and the whole
comparison protocol still match; changing the episode count causes a fresh
incumbent evaluation.

Recommendation changes require eligible evidence and a clear paired
improvement under the comparison policy, including its safety checks. Ties,
uncertain differences, incompatible panels and missing comparison evidence
keep the incumbent. The first recommendation also needs valid benchmark
evidence. A newer version is not automatically a better version.

For direction and terrain skills, priority is survival, tracking, settling,
stopping and turning when enabled, then course progress. Each terrain family
has its own constraint. Every metric must rule out a material regression;
the first priority with a clear improvement decides. The comparison uses a
paired bootstrap interval, so an uncertain higher-priority result retains the
incumbent even when a lower-priority average improves.

The separate behavior certificate uses the initial, versioned policy in
`configs/behavior_certification.toml`: 20 episodes per terrain family, a
minimum 20-second horizon, confidence bounds for survival and task success,
and explicit stop/turn coverage. Nonflat terrain success requires travel
beyond the flat starting area and its transition. These are initial acceptance
rules, not evidence that any current trained model has passed them. The small
species must demonstrate that terrain exposure too.

The library records the comparison and decision reason. The notebook shows
the behavior certificate, candidate status, replication count and
recommendation reason separately from ordinary diagnostic results. Canonical
publication reports are saved in `certified_publications.json` in the run;
behavior publication is recorded in `run.json` under `certification`.

### Comparison reports inside each training run

Every passing candidate that receives a comparison benchmark also saves a
self-contained report in its training folder:

- Direction and terrain behaviors: `comparison/`.
- Canonical behaviors: `comparison/<stage>/`, so stages keep separate reports.

Open `summary.md` for the side-by-side scores, paired uncertainty intervals,
and recommendation outcome. `head_to_head.json` retains both scored panels,
model hashes, exact episode settings, per-episode metric samples and the
publication decision. `candidate.json`, `incumbent.json` and `decision.json`
provide those parts separately. The incumbent file is present only when a
previous recommendation existed.

Reused incumbent scores are copied into the report too. These files remain
readable after moving the training run or removing access to the shared
library. A first passing candidate is explicitly marked as having no incumbent;
it is not reported as winning a head-to-head comparison. A passing candidate
that still needs independent training seeds retains its report with provisional
status. Failed gates and quick tests do not produce a comparison report.

## Practical workflow

1. Keep automatic source selection enabled and choose a species and behavior.
2. Train with a fresh training seed when adding a genuine replication.
3. Review the gate result, per-terrain results, saved replay maps and the
   library's decision reason.
4. Keep provisional candidates in history while gathering the missing
   independent runs. A recommendation changes only after the requirements
   and comparison support it.
5. Start a new run to train another variant; occupied run outputs and existing
   version evidence are never overwritten.

Publish sequentially when separate Colab sessions mount the same Drive
library. File locks coordinate writers on one mounted filesystem; they do not
provide a distributed lock between separate Drive mounts.
