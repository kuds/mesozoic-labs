# The review of #655: its reproduction and the cross-check with this record (2026-10-31)

Evidence for decision D-D61
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) and for
§10 of [TREX_TAIL_SATURATION_HABIT_2026_10.md](../../TREX_TAIL_SATURATION_HABIT_2026_10.md),
where its short names are given (RP and XR). #655 is a report-only pull
request on the T. rex stance pop, the displaced pose and saturation (its
note `docs/investigations/TREX_STANCE_POP_POSE_SATURATION_2026_10.md` and an
evidence folder, at its head `cfe5947f` on `main` `358651fe`; open, not part
of this record). At the maintainer's request of 12:53 UTC on 2026-10-31 the
session reviewed it in two lenses with no training, and at the maintainer's
word ("Post a review comment (Recommended)", 13:26 UTC) posted one comment on
it, nothing else:
<https://github.com/kuds/mesozoic-labs/pull/655#issuecomment-6543958379>.
The run log's lines 809 to 811 (lines 8 to 10 of
[`../phase_critic_results/run_log.txt`](../phase_critic_results/README.md))
give the review's findings in short.

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README). Each copy's source is the file at the same path
below `work/pr655_review/` (the session's work folder, `/home/user/work`, the
run log's W). Every copy is byte-identical to its source, absolute paths
included; none held a session tool id.

- **`repro/`** (RP): the reproduction lens. It re-ran #655's evidence
  scripts from #655's committed copies (on that branch, outside this
  repository) and compared each output with the committed one:
  `SHA256SUMS_reruns.txt` lists the sha256 of the reruns and of the
  committed copies they equal (`cmp`: identical), with the t4b output's
  sha256 and size (`t4b_rerun_sha.json`; the 23 MB output itself was
  deleted after the comparison); the `*.rerun.log`, `*.rerun.txt` and `*.time`
  files are the reruns' printed outputs and timings. `rb_probe.json` is the
  reviewer's own probe (`work_rb/scratchpad/rb_probe.py`, not part of #655):
  #655's statue window rewards and lean cases re-run open loop on the run
  branch's code copy (`8737cd3d`, whose stance configuration the trained r10
  policies used), with the run branch's default leg reference (`settled`)
  and with `keyframe`, 4 cases at 5 seeds each.
- **`xref/`** (XR): the cross-reference lens. `XREF.txt` sets #655 against
  this record, the second critic-blind screen and the why-probe: where they
  agree, where they differ (the step-200 aliasing 3.1 to 3.3 times
  `main`'s on trained run-branch policies, and about 3.5 times on the run
  branch's statue by `repro/rb_probe.json`; #655's open question 1
  measured; its AUC of 0.71
  being the noise-free value), and what #655 adds for a screen with a
  critic-only phase input; `NOTES.txt` its method; `scripts/` its two
  read-only reductions of the why-probe's existing rollouts and `out/` their
  outputs (`oq1_buckets.*`: V − G by #655's buckets; `hidden_terms_by_bucket.*`:
  the reward terms by bucket), with `sigma_by_checkpoint.txt` (exploration σ
  per actuator from the rollouts' metadata) and
  `stance_toml_main_vs_runbranch.diff` (the `key = value` lines of `main`'s
  `stance.toml` against the run branch's); `SHA256SUMS.txt` its sums.

Not kept: `repro/work/scratchpad/`, #655's own scripts, README and committed
outputs as the reproduction ran them (they are #655's files, on its branch;
`SHA256SUMS_reruns.txt` gives the outputs' sha256), `repro/work_rb/`'s copies
of three of those scripts that `rb_probe.py` imports (identical to them) and
its empty marker, and `repro/tmp/`, the text of a GitHub issue the review
read. The scripts read the session's paths they name and are kept as
records of what was run, not to be run from here; none reads a name
`test_policy_loading.py` guards.

Kept as their sources have them, so to leave out of a manual pre-commit run:
`repro/rb_probe.json` (288,450 bytes), `repro/t4b_rerun_sha.json` and
`xref/out/hidden_terms_by_bucket.json` and `oq1_buckets.json` have no final
newline. No copy holds a model identifier, an email address or a commit
trailer.
