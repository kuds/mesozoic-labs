# Shared certified models

**Status (2026-09-20): scheduled for removal.** Canonical chains stopped
reading the library on 2026-09-16 (#543, automatic trunk selection, decision
D-A25). Consolidation PR-4 deleted the canonical publish wrapper
(`certified_canonical.py`), the notebook's `CERTIFIED_LIBRARY_ROOT` /
`PUBLISH_CERTIFIED` / `CERTIFIED_COMPARISON_EPISODES` knobs, its stamp, copy
and publish blocks and the `certified_inputs/` copies of canonical ancestors;
PR-5 ([CONSOLIDATION_PLAN_2026_09.md](CONSOLIDATION_PLAN_2026_09.md)) deletes
`certified_library.py`, the command-line consumers and this document. Until
then `SOURCE_SELECTION` applies to the direction and terrain behaviors only,
and nothing canonical depends on a certified directory on Drive.

The library is reached from the command line only: `train_behaviors
--auto-source --resume` selects the exact behavior's recommendation and
`--publish-certified` records candidates. The SB3 notebook's direction/terrain
path selects (`SOURCE_SELECTION`) but never publishes or compares; a fresh
pilot needs an explicit `BEHAVIOR_CHECKPOINT` / `BEHAVIOR_VECNORMALIZE` pair.

## Notebook controls

| Setting | Default | Meaning |
| --- | --- | --- |
| `SOURCE_SELECTION` | `"auto"` | Direction and terrain behaviors only: select a compatible recommended source automatically; `"manual"` requires explicit behavior paths. Canonical chains select their trunk through `TRUNK_FROM` (below), never through the library. |

With Google Drive, the default library is
`/content/drive/MyDrive/mesozoic-labs/certified`. Local runs use
`<repository>/certified`. Training output remains in the existing `logs` tree.
Building the notebook run plan creates neither a run nor a library directory.
Source resolution happens after the runner knows the exact task identity.

### Canonical behaviors: stand, walk, hunt and stage IDs

The chain resolves root first. For each ancestor it tries:

1. A valid checkpoint already trained in the current run.
2. The trunk run: the run `TRUNK_FROM` names, or, under the default
   `TRUNK_FROM = "auto"`, the run beside this one under
   `logs/<species>/<algorithm>/` whose certified ancestors cover the most of
   the chain root-first, newest on a tie (decision D-A25 in
   `docs/BEHAVIOR_RECIPES_PLAN.md`). The resolve cell prints the selection,
   the replication each reused node rests on and the runs it refused (the
   first 20; the selection object holds every run scanned). A run whose
   `provenance.json` names another species, algorithm or backend is refused
   before the reuse rules run.

The shared library's recommendation is not consulted for canonical ancestors;
`SOURCE_SELECTION` applies to the direction and terrain behaviors below.
The selected target still trains in the new run. `RETRAIN_FROM` continues to
force training of that node and its descendants; `WIDEN_FROM` retains its
existing validation and re-judging requirements. A reused ancestor must
descend from the exact parent checkpoint resolved earlier in the chain; a
certified child of a different parent is not silently substituted.

If no run covers an ancestor, the canonical chain trains it here. Invalid or
tampered evidence is refused. A pinned trunk never falls back to another run.

### Direction and terrain behaviors

Leave **both** `BEHAVIOR_CHECKPOINT` and `BEHAVIOR_VECNORMALIZE` blank to use
automatic selection of an existing behavior:

- `BEHAVIOR_LOAD_MODE="resume"` selects the exact behavior's recommended
  bundle and continues its remaining training budget.
- `BEHAVIOR_EVAL_ONLY=True` selects the exact behavior for scoring.
- `BEHAVIOR_LOAD_MODE="prepare"` (a fresh pilot) needs an explicit locomotion
  checkpoint pair: the canonical wrapper that selected one left with
  consolidation PR-4, and the runner refuses blank paths in this mode with a
  message naming the pair.

Provide both paths to override automatic selection. Resume and adaptation
still require a matched `bundle.json`; selecting only one source path is an
error. `BEHAVIOR_LOAD_MODE="adapt"` requires an explicit source pair because
changing from another behavior is an intentional task change.

A missing behavior recommendation stops the behavior runner with a reason.
Train and certify the required source, or explicitly select a compatible
matched pair. Nothing launches a hidden locomotion training run.

## Complete copies in each training run

Automatic selection on the command line (`--auto-source --resume`) copies the
**complete selected version** beneath the new behavior run's
`certified_inputs/`, including its model, normalization, saved evidence,
configuration, manifests, and available videos and maps; an explicitly pinned
shared-library version is copied in the same way. Canonical chains copy
nothing since consolidation PR-4: a reused trunk ancestor is recorded under
`ancestors/` and loaded from the run that certified it (plan A10), and runs
made between #543 and PR-4 that hold `certified_inputs/` copies stay valid
because their `ancestors/` records point at the copy.

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

Exact continuation retains the original training seed and parent model and
normalization pair. A new sampling seed during continuation cannot count as
another independent training run. A model must also have completed optimizer
steps in its recorded stage; evaluation-only copies cannot create replicas.

Compatibility groups separate species, algorithm, task and environment
identity, gate settings, and required ancestry. Evidence from a different
group cannot qualify a candidate. Resolving a recommendation verifies its
files and the evidence supporting its replication again.

`QUICK_TEST` does not lower a gate, reduce the required number of independent
training seeds, or turn a short replay panel into certification. Behavior
quick tests skip the separate certification panel and shared publication, and
save that reason in their run record. The notebook never publishes into the
shared library (its publication knob left with consolidation PR-4);
`train_behaviors --publish-certified` is the command-line option.

## Comparing with the current recommendation

The default comparison runs **50 episodes for the candidate and 50 for the
incumbent**, using the same fixed seeds and task cases. For the general
difficult-terrain behaviors, five enabled terrain families therefore receive
ten cases each per model. A different episode count is configurable; the
recorded protocol identifies the exact panel and terrain coverage.
The minimum is two episodes per enabled terrain family (ten for all five
families).

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
recommendation reason separately from ordinary diagnostic results. Behavior
publication is recorded in `run.json` under `certification`.

### Comparison reports inside each training run

Every passing candidate that receives a comparison benchmark also saves a
self-contained report in its training folder:

- Direction and terrain behaviors: `comparison/`.

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
