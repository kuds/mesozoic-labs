# T. rex stance evidence (2026-10-12 to 2026-10-17)

Evidence for [TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md),
frozen with it on 2026-10-17. These are copies of the files that decided or
document each verdict, taken from the session's scratch tree before it is lost.
The scripts are frozen copies as they ran: research tools, imported by nothing,
outside the library, its tests and CI (CI's ruff, mypy and pytest cover
`environments/` only). Checkpoints, rollouts, traces and per-episode judge files
are not here; where they are, and their sha256, is in the note's §14 and in
`r10_screen/fetch_record.json`.

Seven folders were added on 2026-10-18 with the evidence of the follow-ups of
2026-10-17, which the note predates: walking pilot 1's judging
(`pilot1_judging/`), pilot 2's pre-registration (`pilot2/`), the offline
six-heading certificates (`offline_certificates/`), the offline battery
(`battery/`), the foot and ankle audit (`audit/`), the literature report
(`literature/`) and the recommendation on what to log and sense next
(`next_signals/`). Every one of
them is offline and report only: the certificates were rolled on uncertified
scratch-recipe checkpoints that the judge would refuse on task, the pilot's
verdict is its fixed criteria's and no gate's, and nothing here certifies a
run, writes a result bundle or promotes anything. The judgements among them
are rows of the repository's ledger,
[`results/offline_judgements.csv`](../../../results/README.md#offline-judgements-offline_judgementscsv)
(28 rows, each pointing at its file here).

One more folder was added later on 2026-10-18: `stance3/`, the
pre-registration of the three full T. rex stance screen runs of decision
D-D37 (the note's §16): their fixed criteria, Colab steps, the recommendation
that designed them from the offline diagnosis of 2026-10-18, and frozen copies
of their judge and scorer. It holds no judgement: the runs were launched on
2026-10-18 and are not yet judged, and as screens they certify nothing.

*Added 2026-10-19:* `stance3/results/` holds the three runs' judgement by
their fixed criteria (all three fail; the decision table's line is 3b), the
report-only readouts and the overnight scratch experiment that followed, and
an index of its own, [stance3/results/README.md](stance3/results/README.md)
(the note's §17). As screens they certify nothing.

*Added 2026-10-19, later:* `stance3/night/` holds the overnight
experiment's harness-matched no-change comparison (the note's §17.8), and
`stance4/` the criteria of the next screen, decision D-D39 (the note's
§18), fixed before training. `stance4/` holds no judgement, and as a screen
it certifies nothing.

*Added 2026-10-20:* `stance4/results/` holds the stance4 screen's judgement
by its fixed criteria (all three runs fail; the decision table's line is 5,
final; the note's §19), with report-only readouts and an index of its own,
[stance4/results/README.md](stance4/results/README.md). `stance4/night/`
holds the replicate study of the stagger penalty and `stance4/night2/` the
CoP-band continuation test, both overnight scratch experiments of
2026-10-20 and report only, and `stance4/copband/` the design note of the
per-foot CoP band prototype, the basis of decision D-D40 (the note's §20).
As a screen and as scratch experiments they certify nothing.

*Added 2026-10-22:* `stance5/` holds the pre-registration of the stance5 screen
of decision D-D40 (the note's §20): its criteria, fixed before training, their
notes, the Colab steps, and frozen copies of the judge and scorer.
`stance5/results/` holds the screen's judgement by those criteria (Run 1, seed
52, holds; Runs 2 and 3 do not; the decision table's line is 3, final; the
note's §21), with report-only readouts and an index of its own,
[stance5/results/README.md](stance5/results/README.md). As a screen it certifies
nothing.

*Added 2026-10-22, later:* `stance4/satdiag/` holds the offline, report-only
diagnosis of stance4 s42's actuator saturation that decision D-D40 5a asked for
(the note's §21.5): the report and the five tables it cites. It changes no
verdict.

*Added 2026-10-22, later:* `pilot2/` gains walking pilot 2's criteria and Colab
steps as amended on 2026-10-21, beside the copies of 2026-10-18, which are left
as they were, and the summary of the rebuild for its new parent (decision D-D36
as amended on 2026-10-20; the note's §21.6). Pilot 2 is not yet trained, and
nothing here is a result.

*Added 2026-10-24:* `stance5/replication/` holds the pre-registration of the
stance5 replication of decision D-D41 (the note's §22): its criteria, fixed
before training, the Colab steps, the joint-noise sigma sidecar for the new
seeds, and frozen copies of the judge and scorer.
`stance5/replication/results/` holds the replication's judgement by those
criteria (Run 4, seed 70, holds; Run 5, seed 80, does not; the decision table's
line is 2, final; the note's §23), with report-only readouts and an index of its
own, [stance5/replication/results/README.md](stance5/replication/results/README.md).
As a screen it certifies nothing.

*Added 2026-10-24, later:* `stance5/k3diag/` and `stance5/tailexp/` hold the
reports of two report-only workflows the maintainer agreed to on 2026-10-22
(the note's §22.2 and §23): a diagnosis of the stance5 band runs' joint-space
noise falls (K3), and a scratch experiment on the tail habit (P1, a bound on the
policy mean). Each puts its skeptic checks first. They change no verdict.

*Added 2026-10-24, later:* `stance5/replication/` gains the report-only K3
mechanism readouts that the maintainer agreed to on 2026-10-22, after the K3
diagnosis and before any J checkpoint of the replication existed: their
addendum (`readouts_addendum_k3.txt`), the tool, its runner and validation, and
its outputs (`k3readouts/`; the note's §23). They are post hoc, enter no
verdict, decision line or tie-break, and are not a veto.

*Added 2026-10-25:* `stance5/tailscreen/options/` and `stance5/tailscreen/probes/`
hold the first steps of the tail-habit round of decision D-D42 (the note's
§24): the options study's draft brief and its corrected version (v2, with the
two skeptics' findings and their verdicts), the screen design's decisions list
(the technical defaults the maintainer took with decision D-D43), and the
no-training CPU probes of 2026-10-24 with the read of the band runs'
`policy_probe.npz`. All of it is report only: nothing in it trained a policy,
and it enters no verdict.

*Added 2026-10-25, later:* `stance5/tailscreen/` holds the tail_3 bound screen
of decision D-D43 (the note's §24): a scratch CPU continuation screen of a
training-loss bound on `tail_3_pitch_act`'s mean, upper side, in three arms and
18 replicates, its criteria fixed and approved before any training step, its
approval and gate records, the bound's scratch commits as a patch file, the
tools, the score (outcome C: neither bound adopted, final) and the independent
verification, with an index of its own,
[stance5/tailscreen/README.md](stance5/tailscreen/README.md). As a screen it
certifies nothing, and nothing it ran is on `main`.

*Added 2026-10-25, later still:* `stance5/tail_sr3/` holds the evidence of
decision D-D44, T. rex physics r10 (the note's §25): the posture options study
that followed the tail_3 screen, the r10 tail probe and its candidate SR3 (the
three tail pitch springs preloaded, nothing else), the CPU checks of SR3 (eight
r9 checkpoints zero-shot in closed loop, the statue through the gate's code on
an SR3 tree, a diagnosis of the heading-invariance test, a skeptic), the
re-measurement of every constant pinned to r9 on the branch that builds r10,
and the run log's lines 578 to 603, with an index of its own,
[stance5/tail_sr3/README.md](stance5/tail_sr3/README.md). Nothing in it trained
a policy. Its "physics r10" is the plant counter's next integer on `main`, not
the foot of `r10_screen/` (the parked draft #623, D-D34).

## What changed in the copies

[SOURCES.txt](SOURCES.txt) lists every file with the scratch file it came from,
that file's sha256, and what the copy changed. Three kinds of change, nothing
else:

- **prefix:** the absolute path of the session's scratch directory is replaced
  by `scratchpad` (and the session directory above it by `session`). Paths in
  these files such as `scratchpad/foot/...` or `foot/...` name that tree; only
  the files listed here survive. Scripts with a `SCRATCH` constant
  (`judge_r10.py`, `score_r10.py`) now resolve it relative to the working
  directory.
- **whitespace:** trailing whitespace removed and a final newline added, as the
  repository's pre-commit hooks require; JSON content is unchanged.
- **trailer:** `symmetry/frozen_params.json` held the scratch clone's two
  commit-trailer lines (`repo.commit_trailer`); they are replaced by a
  placeholder. Its sha256 is therefore not the `58b2c8fb…` that READY-v1
  recorded before the symmetry screen trained (the original's sha256 is in
  SOURCES.txt). `judge_r10.py` and `score_r10.py` assert that original value,
  and `verify_final.py` and `recount_final.py` the value READY-v1 passes them,
  so against this copy they stop with a sha256 mismatch; every threshold they
  read is unchanged (`decide_final.py` and `test_decide_final.py` run on it as
  it is).

*Added 2026-10-18:* the copies of 2026-10-18 add two more kinds of
change, nothing else:

- **links:** `literature/trex_foot_and_standing_balance.md` linked four times to
  files of the session's checkout (`/home/user/mesozoic-labs/...`); the links
  now lead to the same files relative to the repository root.
- **gzip:** the ten largest JSON files copied (pilot 1's nine probe outputs, about
  0.7 MB each, and the battery's `push_summary.json`, 0.65 MB) are kept as
  `<name>.gz` (Python's `gzip.compress`, level 9, mtime 0) after the other
  changes; `gunzip` gives the changed copy back.

No copy of 2026-10-18 needed a redaction: none held a model identifier, an
email address or a commit trailer. The folders added on 2026-10-18 rename the
scratch's `logs/` directories `run_logs/` or `judge_logs/`, because the
repository's `.gitignore` ignores every directory named `logs`. SOURCES.txt
ends with the per-episode certificate files that were not copied (about 1.8 MB
and 0.76 MB each), with their sha256 and size.

*Added 2026-10-18 (`stance3/`):* `stance3/criteria.txt` and
`stance3/colab_steps.txt` are byte-identical to the files the maintainer
received (the criteria's sha256 `1ce910dc…` is the one recorded before
training, and the file has no final newline, which is kept so that its hash
stays that one; the repository's pre-commit `end-of-file-fixer` would add one,
so leave it out of a manual pre-commit run). `stance3/recommendation.md` and
the judge and scorer carry the **prefix** change only; as in `judge_r10.py`,
their `SCRATCH` constant now resolves `scratchpad` relative to the working
directory. Nothing else differs, and none of the six held a model
identifier, an email address or a commit trailer.

*Added 2026-10-19 (`stance3/results/`):* the copies of 2026-10-19 add four
more kinds of change, nothing else: **excerpt** (one section of the overnight
design, verbatim), **trim** (the six-heading certificates' `run_summary.json`
cut to their verdict fields), **extract** (one key of the night analysis
workflow's output, written on its own) and **redact** (the session's
workflow run ids and the report document's id removed from the run log);
SOURCES.txt defines them, and
[stance3/results/README.md](stance3/results/README.md) says which file each
applies to. A source outside the scratch root is written below `session`. No
copy of 2026-10-19 held a model identifier, an email address or a commit
trailer.

*Added 2026-10-19, later (`stance3/night/`):* its two copies carry the
**prefix** change, and `nochange_compare.md` the **whitespace** change too
(one line of trailing spaces); nothing else. `stance3/night/interpretation.md`
has the same source as `stance3/results/night_scratch/interpretation.md` and
is byte-identical to it.

*Added 2026-10-19, later (`stance4/`):* `stance4/criteria.txt` is
byte-identical to the file fixed before training (sha256 `978a93da…`; it
ends with a newline). It holds no model identifier, email address or commit
trailer.

*Added 2026-10-20 (`stance4/results/`, `stance4/night/`, `stance4/night2/`,
`stance4/copband/`):* the copies carry the **prefix**, **whitespace** (here a
final newline added to three JSON files and a blank last line removed from
`stance4/results/split_form.md`), **trim** (the notebook's gate verdicts cut to
their verdict fields) and **redact** changes, and one more kind, nothing else:
**checkout**, the session's checkout directory (an absolute path below the home
directory) replaced by `checkout`, once each in `stance4/night/spec.md`,
`stance4/night2/spec.md` and `stance4/copband/design.md`. The redaction is in
`stance4/results/run_log.txt` only: 6 workflow run ids (3 distinct), one
scheduled-task id and the report document's artifact link, each now
`[id removed]`, and the assistant's name in a GitHub check's name, now
`[name removed]`. SOURCES.txt defines them and
[stance4/results/README.md](stance4/results/README.md) says which file each
applies to. No copy holds a model identifier, an email address or a commit
trailer. `stance4/copband/design.md` is kept as written: what later decisions
supersede in it is in its entry under Contents, not in the file.

*Added 2026-10-22 (`stance5/`, `stance5/results/`):* the copies carry the
**prefix**, **whitespace** (here a final newline added to nine JSON files),
**trim** (the notebook's gate verdicts cut as stance4's, the certificates'
summaries as stance3's), **redact** and **checkout** changes, and one more kind,
nothing else: **lines**, a line range of the source, the rest left out:
`stance5/results/run_log.txt` is lines 208 to 395 of the session's run log,
whose first 207 lines `stance4/results/run_log.txt` copies. The redaction is in
that log only: 30 workflow run ids (18 distinct), 2 scheduled-task ids and the
report document's artifact link, each now `[id removed]`; the assistant's name
in a GitHub check's name, now `[name removed]`; and, an item not redacted
before, 9 shortened Drive folder ids (seven or eight characters and `...`), each
now `[id removed]`. The **checkout** change is in that log and in
`stance5/judge/judge_stance5.py`. `stance5/criteria.txt` is byte-identical to
the file fixed before training (sha256 `3b28963f…`; it ends with a newline), and
so are `stance5/criteria_notes.md`, `stance5/colab_steps.txt` and
`stance5/judge/test_score_stance5.py`. The judge and scorer carry the **prefix**
change (the judge the **checkout** change too): as in `stance3/judge/`, their
`SCRATCH` constant now resolves `scratchpad` relative to the working directory,
and their sha256 is not the one the judge files and the Colab steps record.
SOURCES.txt defines the changes and
[stance5/results/README.md](stance5/results/README.md) says which file each
applies to. No copy holds a model identifier, an email address or a commit
trailer.

*Added 2026-10-22, later (`stance4/satdiag/`):* `report.md` carries the
**prefix** change (one line) and `rollout/tables/onset_within_episode.txt` the
**whitespace** change (a blank last line removed); the other four are
byte-identical. Nothing else differs, and no copy holds a model identifier, an
email address or a commit trailer.

*Added 2026-10-22, later (`pilot2/`):* `pilot2/criteria_2026-10-21.txt` and
`pilot2/rebuild_1021_summary.md` carry the **redact** change, one workflow run
id each, now `[id removed]`, so the criteria copy's sha256 is not its source's
`bdcdae4b…` (SOURCES.txt gives it); `pilot2/colab_steps_2026-10-21.txt` is
byte-identical to its source (`60845fdc…`). `pilot2/criteria.txt` and
`pilot2/colab_steps.txt` stay the texts of 2026-10-18 (`a240ff5d…` and
`bc94bda3…`); the amended criteria cite the first as their pre-amendment copy.
No copy holds a model identifier, an email address or a commit trailer.

*Added 2026-10-24 (`stance5/replication/`, `stance5/replication/results/`):* the
copies carry the **prefix**, **whitespace**, **gzip**, **trim**, **redact**,
**checkout** and **lines** changes, nothing else and no new kind:
`stance5/replication/results/run_log.txt` is lines 396 to 504 of the session's
run log, whose lines 1 to 207 and 208 to 395 are the stance4 and stance5 copies.
The redaction is in that log only: 49 workflow run ids (28 distinct) and 6
shortened Drive folder ids, as before, and, items not redacted before, 15
background-job ids of the session's queue workers (13 distinct) and 1 task id
of the session's task list, each now `[id removed]`.
`stance5/replication/results/scores/score_final.json.gz` is gzipped, as the
JSON is above the repository's 500 KB limit for an added file.
`stance5/replication/criteria.txt` is byte-identical to the file fixed before
training (sha256 `a38536de…`; it ends with a newline), and so are
`stance5/replication/colab_steps.txt` (`d646fc3c…`, the steps sent at 18:00 UTC
on 2026-10-22) and `stance5/replication/judge/test_score_stance5r.py`. The
judge, the scorer and the sidecar `sigma_s70_s80.json` carry the **prefix**
change (the judge the **checkout** change too), so their sha256 is not the
`16d39f7d…`, `55c37ba7…` and `89778b3f…` the criteria, the Colab steps and the
judge files record; SOURCES.txt gives those as the sources' sha256.
[stance5/replication/results/README.md](stance5/replication/results/README.md)
says which file each change applies to. No copy holds a model identifier, an
email address or a commit trailer.

*Added 2026-10-24, later (`stance5/k3diag/`, `stance5/tailexp/`):* each
`report.md` carries the **redact** change (one workflow run id, now `workflow
[id removed]`) and the **prefix** change, and `stance5/tailexp/report.md` the
**checkout** change too (once). Nothing else differs, and no copy holds a model
identifier, an email address or a commit trailer.

*Added 2026-10-24, later (`stance5/replication/readouts_addendum_k3.txt`,
`stance5/replication/k3readouts/`):* the addendum (sha256 `7a79e9f6…`) and
`k3readouts/VALIDATION.md` are byte-identical to their sources; the tool, its
runner, `validate_k3r.py`, `README.md` and the nine outputs carry the
**prefix** change only, so the tool's and the runner's sha256 are not the
`d0aed5a1…` and `5e3c66cd…` the outputs record. No copy holds a model
identifier, an email address or a commit trailer.

*Added 2026-10-25 (`stance5/tailscreen/options/`, `stance5/tailscreen/probes/`):*
their sources are in a second scratch tree, that of the session that took the
work over at 15:14 UTC on 2026-10-24, which SOURCES.txt writes below
`scratchpad2`. One more kind of change, nothing else: **prefix2**, that tree's
absolute path replaced by `scratchpad2`, as **prefix** replaces the first
tree's by `scratchpad`. The two briefs carry **prefix2**, **prefix** and
**checkout**; `options/decisions_v1.md` is byte-identical; seven probe files
carry **prefix2** (paths in logs and printouts) and
`probes/noisevalue/analyze.py` **prefix2** and **prefix**; **whitespace** removes the trailing spaces of 80 lines of
`probes/unclip/out/analyze_h040.out` and adds a final newline to four JSON
copies. The rest are byte-identical. No copy holds a model identifier, an email
address or a commit trailer.

*Added 2026-10-25, later (`stance5/tailscreen/`):* no new kind of change.
`results/run_log.txt` carries **lines** (lines 505 to 577 of the session's run
log) and **redact** (22 workflow run ids, 12 distinct, and one background-job
id); the patch file carries **trailer** (each commit's two trailer lines),
**redact** (the commits' `From:` lines, an author name and an email address) and
**whitespace** (the single space of its blank context lines; `git apply` of the
copy still gives the scratch commit's two files exactly);
`tools/copy_durable.sh` carries **trailer** too (the trailer it writes into its
commits). The criteria (`9930ee72…`), `APPROVED`, `calib/coef.json`,
`gate/gate_pass.json` and the score files are byte-identical, so their sha256
are the ones the approval and the run log pin (`coef.json` and `gate_pass.json`
have no final newline, which is kept; leave them out of a manual pre-commit
run). Five of the eight tools carry **prefix2**, **prefix** or **checkout**, so
their sha256 is not the one `screen_common.py` pins (SOURCES.txt gives the
pinned one as the source's).
[stance5/tailscreen/README.md](stance5/tailscreen/README.md) says which file
each change applies to. `criteria.txt` keeps the two absolute session-scratch
paths that define its `SP` and `S2` (its lines 18 and 19), the only copy here
that does, because its sha256 is the approved pin. Eight small per-episode files
are kept, an exception to this folder's rule on per-episode files: the
verification's four re-measured `final_flags.csv` (`verify/remeasure/out/`, 20
episodes each) and four probe outputs (`probes/probecheck/probe_s42.json` and
`probe_s80.json`, 6 episodes each, `probes/skeptic/noise_shift_s80.json`, 90,
and `probes/skeptic/unclip_alt_s42.json`, 40; at most 14 KB each), because they
are the evidence of the re-measurement's field-for-field result and of the
probes' counts; the panels' own `flags.csv` and per-step CSVs are not here.
After these changes no copy holds a model identifier, an email address or a
commit trailer.

*Added 2026-10-25, later still (`stance5/tail_sr3/`):* no new kind of change.
Its sources are in the second scratch tree but for the run log.
`results/run_log.txt` carries **lines** (lines 578 to 603 of the session's run
log) and **redact** (15 workflow run ids, 7 distinct, and one scheduled-task
id); `posture/brief_corrected_summary.md` carries **redact** (one workflow run
id); 26 copies carry **prefix2** where they name a scratch path, and
`posture/brief_draft.md` **prefix** and **checkout** too; **whitespace**
removes trailing spaces and adds final newlines, and in
`cpucheck/gate/sr3.patch` removes the single space of the blank context lines
(`patch -p1` of the copy on a `git archive` of `9f2355a3` still gives the CPU
checks' SR3 tree exactly); the eight Python scripts of `remeasure/scripts/` are
kept as **gzip** copies, since `test_policy_loading.py` parses every Python
file under `docs/`. The rest are byte-identical. No copy holds a model
identifier, an email address or a commit trailer.

*Added 2026-10-27 (`stance6_setup/`):* no new kind of change.
`results/run_log.txt` carries **lines** (lines 600 to 680 of the session's run
log; its lines 600 to 603 are also the last four of
`stance5/tail_sr3/results/run_log.txt`), **redact** (47 workflow run ids, 23
distinct, 3 scheduled-task ids, 1 background-job id, 1 id of the session's
task list and 2 Drive folder ids) and **checkout** (once);
`results/stance6_launch_lines.txt` carries **lines** (the run log's six
`STANCE6` lines, its lines 666, 668 and 671 to 674, the same bytes as those
lines of `results/run_log.txt`) and **redact** (three workflow run ids, two
distinct). No copy holds a model identifier, an email address or a commit
trailer.

*Added 2026-10-27, later (`stance5/k3extract/`):* no new kind of change. Its
39 judge copies carry **trim** (cut to the keys `stance5/k3extract/README.md`
names, each panel's `episodes_detail` rows to their seed, length and horizon
flag) and **whitespace** (a final newline); `k3_extract.csv` and
`make_k3extract.py` were written for this record and are byte-identical to
their scratch files. No copy holds a model identifier, an email address or a
commit trailer.

*Added 2026-10-27, later still (`stance6/`):* no new kind of change. Every
copy is byte-identical to its source, its absolute session-scratch paths
included (no **prefix**, **prefix2** or **checkout**), so that each sha256 is
the one the approval and the criteria pin; the 13 Python tools are **gzip**
copies (`test_policy_loading.py` parses every Python file under `docs/`).
Every file but its README existed before the first training step; the run
log's launch and launch-check lines, written after the launch, are in
`stance6_setup/`. No copy holds a model identifier, an email address or a
commit trailer.

*Added 2026-10-28 (`stance6_results/`):* no new kind of change.
`run_log.txt` carries **lines** (lines 681 to 720 of the session's run log;
it continues `stance6_setup/results/run_log.txt` with no line in both) and
**redact** (14 workflow run ids, 10 distinct, and 2 scheduled-task ids). No
copy holds a model identifier, an email address or a commit trailer.

*Added 2026-10-28, later (`stance6_results/score/` to `fetch/`):* one more
kind of change, nothing else: **lineends**, CRLF line ends made LF, as the
repository's `.gitattributes` (`text=auto`) stores a text file, in
`fetch/manifest.csv` only; SOURCES.txt defines it. Every copy is
byte-identical to its source, absolute session-scratch paths included (no
**prefix**, **prefix2** or **checkout**, as in `stance6/`;
`review/NEXT_STEPS.txt` keeps the one checkout path it names, as
`stance6/fetch_agent_brief.txt` does), so that its sha256 is the one the run
log, the review or the round's durable copy records, but for **redact**,
**lineends** and **gzip**: the scorer's and the completeness check's texts and
the scorer's JSON lose the workflow run ids and Drive folder ids of the four
run-log lines they print (which then read as those lines of
`stance6_setup/results/run_log.txt`), the fixer's record one workflow run id,
and the fetch manifest and folder record their 216 distinct Drive ids; the
scorer's JSON and the readouts' JSON are **gzip** copies (3.5 and 0.6 MB,
above the repository's 500 KB limit for an added file), and
`fetch/manifest.csv`'s CRLF line ends are LF (**lineends**). The five D-9 and
D-10 summaries keep their missing final newline and `score/score.txt` its 4
lines ending in spaces; leave them out of a manual pre-commit run. No copy
holds a model identifier, an email address or a commit trailer.

*Added 2026-10-28, later still (`stance6_results/panels/`):* no new kind of
change. All 122 copies are byte-identical to their sources, absolute
session-scratch paths included: none held a session tool id or a Drive id,
and none is over 500 KB. The 18 judge files keep their missing final
newline; leave them out of a manual pre-commit run. No copy holds a model
identifier, an email address or a commit trailer.

*Added 2026-10-29 (`ecs_results/`):* no new kind of change.
`ecs_results/run_log.txt` carries **lines** (lines 721 to 759 of the
session's run log; it continues `stance6_results/run_log.txt` with no line
in both) and **redact** (22 workflow run ids, 11 distinct, and 1
background-job id). No copy holds a model identifier, an email address or a
commit trailer.

[SHA256SUMS](SHA256SUMS) pins every file here but this README and itself
(`sha256sum -c SHA256SUMS` in this directory). *Added 2026-10-19:* nor
`stance3/results/README.md`, which, like this README, has no source.
*Added 2026-10-20:* nor `stance4/results/README.md`, likewise.
*Added 2026-10-22:* nor `stance5/results/README.md`, likewise.
*Added 2026-10-24:* nor `stance5/replication/results/README.md`, likewise.
*Added 2026-10-25:* nor `stance5/tailscreen/README.md`, likewise.
*Added 2026-10-25, later still:* nor `stance5/tail_sr3/README.md`, likewise.
*Added 2026-10-27:* nor `stance6_setup/README.md`, likewise.
*Added 2026-10-27, later:* nor `stance5/k3extract/README.md`, likewise.
*Added 2026-10-27, later still:* nor `stance6/README.md`, likewise.
*Added 2026-10-28:* nor `stance6_results/README.md`, likewise.
*Added 2026-10-29:* nor `ecs_results/README.md`, likewise.

## Contents

**`run_a/`: run A and the 2026-10-12 foot and CoP investigation (§3).**

- `report.md`: the corrected report (mechanism, reward, per-species anatomy,
  probes, recommendation, open questions, 11 corrections). It cites further
  scratch files under `foot/`; the ones that back the note are copied beside
  it.
- `stance_gate_report_20261011_165957.txt`: run A's `stance_quality/v2` report
  from Drive (1/40 clean).
- `ablate_table.txt` (held-pose ablation of run A's mean action),
  `thresholds.txt` (ramped single-ankle thresholds per model),
  `an_moment.txt` (the verifier's ankle-moment balance).
- `reward_pairs.txt` (per-term paired returns), `observability.txt` (CoP fits
  from observation channels).
- `noise_saturation.txt`, `noise_s48.txt`, `noise_s56.txt`: the statue and the
  trained policies under the policies' own noise.
- `probe0_analysis.txt`, `probe1_results.txt`, `probe2_results.txt`: the three
  1M probes on `main` `8565d299`. Read probe 2 with the report's Corrections 8
  (its episode-level p-values are not evidence).
- `anatomy_summary.txt`, `statue_per_species.txt`: statue panels per species.

**`noise_rank/`: the offline noise-ranking check (§4.1).** `summary.txt` is the
corrected verdict (it supersedes a first version, not copied);
`pass_corrected.txt` the corrected pass table; `verify_summary.txt` the
independent re-derivation that withdrew the "does not reward falling early"
claim.

**`cap_screens/`: the screens before symmetry and the roll diagnosis (§4).**

- `screen_1m_analysis.txt`, `screen_3m_noise_analysis.txt`,
  `screen_3m_cop_authority_analysis.txt`, `screen_6m_authority_analysis.txt`,
  `screen_hip_roll_analysis.txt`: each screen's analysis with its
  predeclared criteria. In `screen_3m_noise_analysis.txt` the matched-sigma
  noise comparison ("about 8x ... 14x") and the proposed 12M entry rule built
  on it are withdrawn; `control_s{42,52,60}_uniform_sigma016.json` are the
  control under the same uniform σ 0.16 (9 + 11 + 17 = 37 of 60 falls).
- `roll_diag_analysis.txt`: the lateral pad-roll diagnosis.
  `roll_verify_reward_terms.json` is the independent check's per-term returns;
  it corrects the analysis's "the reward does not price the lean" and its
  option (b) range (net spread ±32).

**`symmetry/`: the symmetry screen, arms L and C, verdict O9 (§5).**

- `frozen_params.json`: every frozen parameter, threshold, seed and checkpoint
  (redacted as above).
- `interim.json`, `verdict.json`: the 3M interim and the 6M verdict;
  `J1_recount.log`, `J2_recount.log`: the independent recount (120 and 537
  quantities, no disagreement); `rerun_ledger.json`: the two infrastructure
  re-runs.
- `verdict_scripts/`: `verify_final.py`, `recount_final.py`,
  `decide_final.py`, `arbiter_final.py` and `test_decide_final.py`,
  byte-identical to the versions whose sha256 `frozen_params.json` records.
- `mirror_ratio.json` (why the KL-form mirror loss, not augmentation),
  `lean_calibration.json` (the lateral lean term's weight and width),
  `ema_credit.json` (PPO's credit for an EMA reward term),
  `groundwork_verify_summary.json` (independent check of the mirror map,
  equivariance, π_sym and lean calibration, with its discrepancies list).

**`r10_screen/`: the r10 foot and arm L screen (§6).**

- `criteria.txt`: K1-K4, written before training, with the dated tie-break
  clarification. `colab_steps.txt`: how the three Colab runs were set up.
- `score.txt`, `score.json`: the scored verdict beside r9 arm L, the best
  stance by the fixed rule and the report-only block. Two reading notes: the
  best-stance listing prints joint-space falls only, though the rule sums
  joint-space and own-sigma falls (the pick is the same); and `score.json`'s
  r9 `foot_cop_fh` fields count unmeasured episodes as over the rail and are
  unused (r9's K2 uses its pad counts).
- `judge_r10.py` (judges one checkpoint, binding plant, task, seed and
  commit), `score_r10.py` (computes K1-K4), `test_score_r10.py` (33 boundary
  cases), `validate_r9.json` (the judge reproducing an r9 judge file before
  use).
- `verify_recompute.py`, `verify_recompute.json`: the independent
  recomputation of every cell from the per-episode rows.
- `report_only_summary.json`: the notebook's end-of-stage certificates and the
  6M-12M in-training statistics. `fetch_record.json`: Drive file ids, sizes,
  md5 and sha256 of the 36 files fetched.

**`diagnosis/`: why seeds 52 and 60 stood on their toes (§7).**
`findings_v3.txt` is the current version. `findings_v1_superseded.txt` is kept
because its figures reached the public #623 comment: its noise hypothesis (H1)
is withdrawn, its edge "1.06" and its slopes -192 / -196 are superseded (v3,
section 4 and its header).

**`pilot/`: the walking pilot's pre-registration (§9; launched on 2026-10-17,
not yet judged).**
*Added 2026-10-18: it was judged on 2026-10-17; see `pilot1_judging/` below.*

- `criteria.txt`: the fixed criteria, byte-identical (sha256 `1cdd495c…`).
  Its first line says "revised three times on 2026-10-17"; its Revisions
  list has four (the fourth, wording only), and the copy is left as frozen.
- `pilot_score.py` (applies the rule), `pilot_judge.py` (runs the probe and
  binds the run, including the parent's sha256), `pilot_parent.py` (the
  screen's parent rule), `gait_probe_pilot.py` (the adapter that runs the
  frozen [gait probe](../gait_2026_09/gait_probe.py) on today's API),
  `make_gait_cases.py` and `fixtures/gait_cases.json` (the probe-computed hop
  and walk cases the rule is tested against), `test_pilot_score.py`,
  `test_pilot_parent.py`.
- `duty_cut_summary.json`: the cost table behind the 0.80 duty guard.
  `pilot_parent_on_screen_score.log.txt`: `pilot_parent.py` on the screen's
  score, naming r10 arm L seed 42 6M, the rule's choice that the maintainer
  departed from.
- `pilot_video.py` and the Colab steps stay with the session; the branch
  `claude/pilot-walk-r9` (`eaeef50f`) holds the parent checkpoint and its
  `SOURCE.txt`.

**`telemetry/`.** `posture_drift_replay.json`: the `PostureDrift` default
replayed on the r10 runs' `gate_progress.npz`, on the 20:20 snapshot and on the
full series (two in a row against three).

### Added on 2026-10-18

**`pilot1_judging/`: walking pilot 1's judging (D-D36; judged 2026-10-17,
final).** Outcome **STILL on 3 of 3 seeds** at 4,194,304 steps: each seed
STILL on 40 of 40 panel episodes, no WALK episode, median speed 0.001-0.002 m/s,
no touchdowns, foot duty 1.00/1.00; at 2,097,152 seed 42 read STILL 39 and
FALL 1. 2,097,152 and 3,145,728 are the trajectory and decide nothing, and no
video review was needed (only WALKS needs one). The rule is the fixed criteria
in `pilot/criteria.txt`; the verdict is the criteria's, not a gate's.

- `run_log.txt`: the runs (Drive folder ids, launch times, what each
  `stage_config.json` was checked for), the 2M pipeline check and the final
  judging, with report-only figures: the in-training evaluation reward "flat at
  ~1,097-1,100 throughout" and the forward-reward term "0.5-3.2 at 2-4.2M (max
  7.5)", both of which hold at the 2M, 3M and 4.2M evaluations only (the
  note's §15.1 gives every evaluation's range, from the runs'
  `gate_progress.npz`, which stay on Drive), the policy's action std
  0.108-0.111 at 4.2M against the parent's 0.095, and the notebook's own
  locomotion gate FAIL on speed (recorded, not a pilot criterion). The log is
  kept as written.
- `OUT/score.json`: `pilot_score.py --seeds 42,52,60` on the nine panels (the
  original's sha256 `86d7350e…`, the value `run_log.txt` records; the copy adds
  a final newline).
- `OUT/s<seed>/<steps>/binding.json`: what `pilot_judge.py` bound (tree
  `eaeef50f`, the checkpoint's and VecNormalize sidecar's sha256, the parent
  `sha256:70f3a444…`, the budget, the plant). `probe.json.gz`: the frozen gait
  probe's 40-episode output for that checkpoint (gzipped; `pilot_score.py` reads
  it unzipped). The traces and plots stay with the session; the checkpoints are
  on Drive (folder ids in `run_log.txt`).
- `judge_logs/`: each judge run's log and the scorer's printout.
- `tools/`: `loadcheck.py` (loads every VecNormalize pickle under an allow-list
  of classes and checks each run's `stage_config.json`, `stage_summary.txt` and
  `gate_progress.npz`), `globals.py` (lists the globals a pickle names) and
  `stdcheck.py` (the policy's action std from a checkpoint's `log_std`, the
  source of the run log's std figures).

**`pilot2/`: walking pilot 2's pre-registration (decided 2026-10-17; not yet
trained, no result).** The maintainer chose "Second pilot, wider exploration":
pilot 1 again with one change, every actuator's `log_std` set to -1.5 (std
0.2231, the stance stage's own starting noise) at the warm start.
`criteria.txt` (written 2026-10-17 and amended 2026-10-18, both before any
pilot-2 training step) states the change, its checks and everything kept from
pilot 1; `colab_steps.txt` is how the runs are launched (run IDs
`pilot2_walk_s42/52/60`). The pilot trains from the branch
`claude/pilot-walk-r9-b` (`ba464855`: pilot 1's `eaeef50f` plus three commits),
pushed with no pull request and never merged.

**`offline_certificates/`: the stage's own six-heading certificate rolled
offline on the r9 arm L symmetry-screen checkpoints (2026-10-17, report only).**
`scripts/run_cert.py` calls `build_stance_heading_report`, the function the
notebook's end-of-stage certificate calls, on `main` `2e734e29` (physics r9)
with `stance.toml` as committed (`--options off`) or with arm L's three env
options added (`--options armL`). Eleven certificates:

| Checkpoint (run `sym_lean_settled_s<seed>`) | Panel | Options | Verdict | Seeds clean at all six headings | LCB (bar 0.80) |
|---|---|---|---|---|---|
| s42 6,291,456 | 3042-3081 | off; armL | FAIL; FAIL | 29/40; 29/40 | 0.5861 |
| s52 6,291,456 | 3042-3081 | off; armL | PASS; PASS | 38/40; 38/40 | 0.8508 |
| s60 6,291,456 (the pilots' parent) | 3042-3081 | off; armL | PASS; PASS | 40/40; 40/40 | 0.9278 |
| s42 robust_best (655,360) | 3042-3081 | off | PASS | 40/40 | 0.9278 |
| s52 robust_best (917,504) | 3042-3081 | off | PASS | 39/40 | 0.8868 |
| s60 robust_best (524,288) | 3042-3081 | off | PASS | 40/40 | 0.9278 |
| s52 6,291,456, audit | 7042-7081 | off | PASS | 38/40 | 0.8508 |
| s60 6,291,456, audit | 7042-7081 | off | PASS | 39/40 | 0.8868 |

Source: `offline_certificates.txt` and each folder's `run_summary.json`. **None
is gate evidence.** The runs were trained in the session's scratch harness
(MirrorKLPPO with scratch `experiment_*` keys, not on Drive) and recorded task
`a2af3c5c…`; these certificates roll `bde44225…` (off) or `cd963518…` (armL), so
the judge would refuse them on task, and an audit panel on 7042-7081 is refused
as gate evidence by design (`evidence_table.txt` §E). Both audits sit on the
`max_hop_or_fall_episodes` 1 rail: both checkpoints fall on seed 7047, where the
statue stands.

- `offline_certificates.txt`: every certificate per heading, its unclean seeds
  and the reasons. `evidence_table.txt`: the bar with its `stance.toml` lines,
  every earlier result and these. `cert_margins.txt`: each bar's worst and
  median value over the 240 policy and 240 statue episodes (s60 6M's worst pad
  CoP is 0.780 against the 0.80 bar). `cert_summary.json`: the cross-checks (the
  0° cells of the three 6M options-off certificates reproduce the scratch
  judge's v2 panels; options off and armL give identical floor metrics, so the
  options change only the reward). `rederive_from_csv.json`: each verdict
  re-derived from its panel CSV by the library's own gate (all 11 agree).
- `<certificate>/run_summary.json` (checkpoint and sidecar sha256, verdict,
  counts, LCB, per-heading means) and `stance_heading_report.txt` (the
  certificate as printed). The per-episode `stance_heading_report.json` and
  `stance_heading_panel_selected.csv` stay with the session (sha256 at the end
  of SOURCES.txt).
- `scripts/`: `run_cert.py` and the shell lines that ran it (`run_all.sh`,
  `run_audit.sh`, `run_robust.sh`), `rederive.py`, `analyze.py` (wrote
  `cert_summary.json`), `margins.py` and `tab_certs.py`. `run_logs/`: each
  run's log.

**`battery/`: the offline battery on the same checkpoints (2026-10-17, report
only).** `battery_report.md` is the report; its citation prefixes `B/`, `H/`,
`N/` and `V/` (printed as `scratchpad/foot/...` paths) are `battery/`,
`offline_certificates/`, `next_signals/` and `battery/verify_skeptic/` here.
Its verdict: **s60 6M is a real but thin pass; s52 6M is on the rail.**

- `fresh_seeds/`: eight more certificates (options off) on fresh panels
  5042-5081 and 9042-9081 for s52 and s60 at 6M and at robust_best, by the same
  `run_cert.py` (`scripts/one_cert.sh`). All eight PASS: s60 6M 40/40 and
  39/40, s52 6M 37/40 (the minimum passing count, LCB 0.8174) and 39/40, both
  robust_best checkpoints 40/40 on both. `fresh_seed_summary.txt` and `.json`
  pool them with the 3042 and 7042 panels (exact Clopper-Pearson intervals; a
  pooled rate is not a gate statistic); `offline_certificates.txt`,
  `cert_margins.txt`, `rederive_from_csv.json` (all eight agree) and each
  certificate's `run_summary.json` and `stance_heading_report.txt` as above.
- `shared_fall/`: seed 7047 replayed at six headings for s60 6M, s52 6M and the
  statue (`shared_fall_summary.json`, `shared_fall_report.txt`, the figure
  `shared_fall_7047.png`): the reset leaves the animal tipped forward, the
  statue survives by folding (its hips sag and its heels come down), and both
  policies hold the hips at or above home and topple forward.
- `push/`: `main`'s impulse probe as shipped and a 16-seed copy with a finer
  grid and fore-aft rows (`push_summary.json.gz`, `push_report.txt`,
  `repo_probe/*/stance_gate_probe_impulse.txt`). The weak push direction
  follows where each policy parks its pad CoP, and no policy beats the statue
  in every direction.
- `own_sigma/`: heading-0 v2 panels with actions sampled from each policy's own
  Gaussian beside deterministic re-runs (`own_sigma_summary.json`,
  `own_sigma_report.txt`): no policy falls, the 6M verdicts hold, and every
  robust_best checkpoint is clean only when run deterministically (0 of 40
  clean under its own noise).
- `report/`: `panel_pass_prob.py` and its output (an illustration under an
  i.i.d. assumption, not a gate statistic) and `disk_usage.txt`.
  `verify_skeptic/`: the independent verifier's scripts and outputs; the
  corrections they forced are applied in the report (its last table).

**`audit/`: the T. rex foot, ankle and posture audit (2026-10-17, main
`2e734e29`, physics r9; measurement only).** `foot_ankle_audit.md` is the
report; its evidence paths are relative to `audit/`. It finds nothing in the
anatomy that pushes the quiet stance forward (in the statue and both 6M
policies the COM projects 2.4-6.8 cm ahead of MTP III, inside the zone where
tyrannosaur tracks are deepest); the COM sits 4-18 cm too high relative to the
hip, the digit spread is 3-5× too narrow, the ankle extends 11° past the only
published T. rex range, and the gate's unsigned pad statistic would fail a
track-like stance over the MTP joints. Its seven candidate physics changes each
need the maintainer's approval.

- `geometry/` (`geometry_audit.json`, `geometry_summary.txt`, the scripts),
  `verify/` (the verifier's static, settle and posture measurements and its
  scripts), `posture/` (`posture_table.txt`, `posture_timecourse.json`, the
  tools), `literature/` (`out/`: the published values restated in the model's
  frame, the digitised Brochu 2003 figures and `literature_crosscheck.md`;
  `scripts/` with `published_inputs.json`, the hand-entered values with their
  locators). The downloaded papers the scripts parse (`literature/dl/`), the
  posture rolls and the per-episode posture files stay with the session.

**`literature/`: the literature report of 2026-10-17 and its research notes.**
`trex_foot_and_standing_balance.md` ("Longer toes cannot cure a forward lean")
and `research_notes/`, the four notes it draws on and cites (T. rex foot
anatomy and posture, bird standing balance, standing-balance mechanics and foot
control, RL for standing). Read its model figures with the audit's
"Corrections to the literature report" (`audit/foot_ankle_audit.md`), which
supersede them where they differ.

**`next_signals/`: what to log and sense next (2026-10-17, report only; nothing
trained).** `recommendation.md`: in behaviour a certified stance is close, in
certification it is not (0 of the 2 certifying seeds), and sixteen numbered
items, from report-only telemetry to policy-interface changes, each with its
approval. Its `NS/` citations are the files beside it here (`certify/`,
`diagnose/`, `questions/`, `sensing/`, `sensing_robotics/`), except
`NS/howclose/`, which is `offline_certificates/`. Its code citations are to
`main` `2e734e29`.

### Added on 2026-10-18 (decision D-D37)

**`stance3/`: three full T. rex stance screen runs, pre-registered
(decided 2026-10-18; launched that day on Colab L4 runtimes, not yet judged).**
Run 1 `stance3_armL_s60` (arm L as #624 ships it, seed 60; branch
`claude/screen-trex-r9-arml-12m` at `7e6b557d`), Runs 2 and 3
`stance3_armLT_s60` and `stance3_armLT_s52` (arm L plus
`leg_home_pose_weight` 1.0 and `leg_home_pose_tolerance` 0.05 rad; branch
`claude/screen-trex-r9-armlt` at `1af2d4d7`), each 12M steps on physics r9;
neither branch is ever merged or has a pull request.

- `criteria.txt`: K1-K4 and the decision table, written before any training
  step (sha256 `1ce910dc…`). Its title's "2026-10-19" is the date first
  planned; the runs started on 2026-10-18.
- `colab_steps.txt`: how the runs are launched and resumed, as sent. It names
  a CPU runtime as the maintainer's choice of 2026-10-18; the runs went on
  L4, which the maintainer kept for all three (the note's §16.2).
- `recommendation.md`: the offline diagnosis of 2026-10-18 ("What the offline
  work found"), the three runs, the criteria, an alternative layout, the
  concerns, the decisions it asked for and the build specification. Its `T/`
  paths (`scratchpad/foot/tonight/`) and `SP/` paths name the session's
  scratch, where the diagnosis' tables and scripts stay.
- `judge/judge_stance3.py` (judges one checkpoint against its branch clone,
  binding plant, task, symmetry record and commit), `judge/score_stance3.py`
  (scores K1-K4, the run verdicts and the decision-table line, and the
  report-only readouts) and `judge/test_score_stance3.py` (80 cases at every
  bar).

### Added on 2026-10-19 (decision D-D37's outcome)

**`stance3/results/`: the three stance3 screen runs judged (2026-10-18 to
2026-10-19, final), with report-only readouts and the overnight scratch
experiment.** Outcome: **all three runs fail** their fixed criteria, so none
holds and none holds fore-aft: Run 1 (arm L, seed 60) fails K2 on the K2c
range (33.9 mm against 30) and K3 (joint-space noise falls 7 and 11 of 20);
Run 2 (L+T, seed 60) fails K1 and K2 early (from 4,194,304 to 6,291,456);
Run 3 (L+T, seed 52) fails K2 (K2a 6 at 8,388,608, K2b 4 at the final, K2c
range 30.6 mm). The decision table's line is **3b**. The best final
checkpoint by the fixed rule is Run 2's (40/40 clean, no noise falls). Its
[README](stance3/results/README.md) indexes the folder:

- `run_log.txt` (the judging log, ids removed), `scores/` (the frozen
  scorer's printouts: `score_final2.txt` and `.json` are the verdict of
  record), `checkpoints.csv` (one row per judge file, 37, with the
  `checkpoints_csv.py` that recomputes it with the scorer's own counts) and
  `certificates/` (the stage's six-heading certificate rolled offline on the
  three finals, report only: all three PASS, none gate evidence).
- `readouts/`: report-only readouts of the Colab checkpoints (Run 2's tail
  actuator saturation and left-right foot split at 4,194,304, Run 1's
  ankle-carried drift and falls, Run 3's split at 8,388,608, and the
  whole-body drift in the eval ledgers).
- `night_scratch/`: the overnight scratch experiment of 2026-10-19 (three
  continuations with a candidate term each, from a scratch clone never
  merged; report only): its pre-registered readouts, the readouts of the arms
  and of the matched no-change controls, the interpretation as written, and
  the verifier's verdicts, which correct it.

### Added on 2026-10-19, later (the night continuations' no-change comparison)

**`stance3/night/`: the harness-matched no-change comparison of the overnight
scratch experiment (2026-10-19, report only).** The spec's local no-change
continuation of Run 3 (`night_R3_nochange`, from the Colab 8,388,608
checkpoint with no added term) was judged and rolled at 9,437,184 and
9,961,472 like the arms. It did not recover: at 9,961,472 it was clean on 25
of 40, with K2a 15 and a window stagger sd of 25.9 mm, against the stagger
arm's 40, 0 and 7.5 mm and the Colab no-change run's 40, 0 and 9.7 mm. The
ankle anchor (H_B) and the stagger term on Run 1 (H_C) stay refuted; the
stagger term on Run 3 (H_A) stays inconclusive, since the two no-change runs
differ by as much as the stagger arm and the local one do, with one run per
arm. None of it enters K1–K4 or certifies anything.

- `nochange_compare.md`: the comparison as written (R1 and R2 at both
  checkpoints, a paired seed bootstrap and exact McNemar tests, the eval
  ledgers and the reading).
- `interpretation.md`: the night interpretation it cites by name,
  byte-identical to `stance3/results/night_scratch/interpretation.md`; read
  it with that folder's `verify_result.json`.

Its judge files, rollouts and scripts stay in the session's scratch
(`scratchpad/foot/stance3/night/cont/`). Its paths: `S3` is
`scratchpad/foot/stance3` and `C` is `S3/night/cont`.

### Added on 2026-10-19, later (decision D-D39)

**`stance4/`: the next T. rex stance screen, pre-registered (decided
2026-10-19; not yet trained when recorded).** Three new runs of arm LTS,
the stance3 L+T recipe plus a stance stagger penalty (weight 0.30,
tolerance 0.045 m), from scratch on physics r9: `stance4_LTS_s52`,
`stance4_LTS_s60` and `stance4_LTS_s42`, from the branch
`claude/screen-trex-r9-armlt-stagger` (`1af2d4d7` plus `ff7114c0` and
`c1e146a8`), which is never merged and has no pull request. The control is
stance3 Runs 2 and 3 as already judged, not retrained. The maintainer's
decision named Colab CPU runtimes, as stance3's Colab steps had. Measured,
stance3's runs trained on Colab L4 GPU runtimes (in the `stage_config.json`
files fetched at 4,194,304 steps; that the control stayed on L4 throughout
is inferred). Before any of the three runs started, the criteria were
re-fixed for L4, the control's runtime type, for the maintainer to confirm
when starting the runs.

- `criteria.txt`: what is trained, how it is judged (K1–K4 exactly as
  stance3's), the control and its caveats, the credit rule, the decision
  table, when the decision is read and the pre-registered readouts, written
  before any training step (sha256 `978a93da…`). Its `SP/` paths name the
  session's scratch (`scratchpad/`), where the judge, the scorer, the build
  record and the statue measurement stay. It replaces a first file (sha256
  `8bda933e…`) that named a Colab CPU runtime, as the decision did. The
  control's L4 is measured in the `gpu` block of its `stage_config.json`
  files, as fetched at 4,194,304 steps; each records the session that wrote
  it, so that the control stayed on L4 throughout is inferred. The first
  file was re-fixed before any of the three runs started and differs only
  in the runtime (measured: line 7 changed and two lines added); it stays
  in the session's scratch (`scratchpad/foot/stance4/superseded/`).

### Added on 2026-10-20 (decision D-D39's outcome)

**`stance4/results/`: the stance4 screen judged (2026-10-20, final), with
report-only readouts.** Outcome: **all three runs fail** their fixed
criteria (H = 0, K2p = 0): Run 1 `stance4_LTS_s52` and Run 2
`stance4_LTS_s60` fail K1 and K2, and Run 3 `stance4_LTS_s42` fails K1–K4.
The decision table's line is **5**, final: the stagger penalty is not
adopted. The best stagger final by stance3's rule is Run 1's (39/40 clean, no
noise falls), one episode short of stance3 Run 2's final (40/40), which stays
the walking pilot 2 parent the criteria offer. Measured, all three runs
recorded a Colab L4 GPU runtime (the `gpu` block of each run's
`stage_config.json`), and the maintainer confirmed it. Its
[README](stance4/results/README.md) indexes the folder:

- `run_log.txt` (the stance4 log, ids removed, ending with the maintainer's
  decisions of 2026-10-20), `scores/` (`score_final.txt` and `.json`, the
  verdict of record), `checkpoints.csv` (one row per judge file, 51, with the
  `checkpoints_csv.py` that recomputes it with the frozen scorer's own
  counts), `extras.txt` and `extras.py` (the fresh, tail and common-noise
  panels on each final, report only) and `notebook_gates/` (the notebook's
  own six-heading gate on each run's `robust_best`, trimmed; report only).
- `panels_summary.csv` and `panels_summary.py`: the diagnostic panels
  (mirrored-leg spawns, the prey straight ahead, both) at every J checkpoint
  of the three runs and the two controls, report only.
- `split_form.md`: where the left-right split went, at 4,194,304 (the
  stance4 feet nearly in line there, the per-foot CoP split not; report
  only).

**`stance4/night/`: the replicate study of the stagger penalty (2026-10-20,
scratch only, report only).** Pre-registered continuations from stance3 Run
3's Colab 8,388,608 checkpoint to 9,961,472, on two new training seeds (152
and 252) with the stagger penalty and without, read beside the three runs
of the night before: stagger n = 3 against no change n = 4. Its reading is
line (2), partial, met through a flagged run (the seed-152 stagger run had 2
panel falls and a reward ratio of 0.945; the seed-252 no-change run was
flagged too). The window stagger sd separates completely (7.5, 7.7 and 4.2
mm against 25.9, 9.7, 14.8 and 29.3; one-sided p 1/35 = 0.029, optimistic
because three of the seven runs suggested the idea, and 0.086 given their
known order); K2a does not (0, 3 and 0 against 15, 0, 3 and 8; p 0.200).
None of it enters a stance4 verdict.

- `spec.md`: the pre-registration, frozen at 01:48:58 UTC (its header says
  01:50, written a minute before; `review_handling.md`), before the queue
  launched at 01:49:08 UTC and so before any training step (sha256
  `24af508a…` as frozen; this copy carries the prefix and checkout changes).
- `review_handling.md`: the review's findings and how each was handled
  before launch.
- `readout.md` and `readout.json`: the readouts R1 (the judge) and R2 (the
  split rollout) at 9,961,472, the tests and the reading, with the R2
  reproduction log.

**`stance4/night2/`: the CoP-band continuation test (2026-10-20, scratch
only, report only).** Pre-registered, paired continuations from stance4
s52's and s60's Colab 4,194,304 checkpoints to 5,767,168, with the band
prototype on (CB) and off (NC), one run per arm per parent, from the scratch
commit `f94452ad` (never pushed). Its reading is line (2), partial: CB is
lower than NC on both readouts for both parents (K2a 0 against 18 for s52
and 3 against 22 for s60; CoP split sd 0.29 against 1.20 and 0.11 against
1.27), but the s60 band run raised new safety flags (3 panel falls,
hop-or-fall 3, a reward ratio of 0.928). In that run the guards left 242 of
59,836 window foot-steps uncounted, all in those 3 episodes (a foot unloaded
or under the 35% load share; in 2 of them a foot also on its digits alone);
in the s52 band run, none. The guards cannot see the route the design calls
its main risk (each foot's CoP held in band by asymmetric ankle torque while
the feet stay staggered); in the s52 pair the band run's window stagger sd
was the higher (9.4 against 6.7 mm). A continuation is not a from-scratch
screen, and none of it enters a stance4 verdict.

- `spec.md`: the pre-registration, frozen at 06:29 UTC before any training
  step (sha256 `b1606f74…` as frozen; this copy carries the prefix and
  checkout changes).
- `review_handling.md`: the review's eight findings and how they were
  handled; no frozen file changed.
- `readout2.md` and `readout2.json`: R1 and R2 at 5,767,168, the guard
  readouts, a paired panel-seed bootstrap, the Colab no-change references
  and the diagnostic panels of the s52 pair (the s60 pair's were skipped at
  the deadline).

**`stance4/copband/design.md`: the per-foot CoP band prototype's design note
(2026-10-20, 00:35–01:30 UTC; a proposal, nothing trained), the basis of
decision D-D40. Decisions 3a–3c of 2026-10-20 supersede its sections on the
step guard, the gait import exception and the speed**; the file is kept as
written:

- 3a, the step guard: guard 1's 35% load-share step (§4.1, and the reward
  cliff of §7 item 4) becomes a ramp, the full price at a load share of 0.25
  or less, no guard price at 0.35 or more, linear between;
- 3b, the speed: the measured +0.62 ms per env step (one thread, 1.40 to
  2.02 ms; §6.3, §7 item 6 and §8 item 3) is to come down, with one fused
  per-step reduction byte-identical to the recorder's readings (no numeric
  target was decided);
- 3c, the gait import exception: the contact reduction moves out of
  `environments/shared/gait/` into a neutral shared module, so the gait
  layering rule (D-D23) holds with no exception (§1's file table, §6.1's
  layering-test exception, §7 item 5 and §8 item 2); the recorder
  bit-identity test, which pins the band's readings, moves with it (§8 items
  2 and 3).

Decision 1a takes the note's term (§2 and §4) as an opt-in `[env]` option,
off by default, with the fixes above, and the runs its §8 item 7 names
(three seeds from scratch, 12M steps, Colab L4); 3d takes its proposed
weight, 0.30 (§0 and §3.4). No recorded decision adopts or supersedes the
note's other proposals, such as §8's reward plumbing (item 4) and the
criteria's contents (item 6), except its recommended arm: §8 item 5
recommended L+T plus the band alone for a line-5 outcome; the arm decided
(2A) is L+T plus the stagger penalty plus the band, with the three stance4
runs as its single-difference control, because the overnight band test ran
on stagger parents and the stagger term prices the route the note calls its
main risk (§5's last row). The prototype is the scratch commit `f94452ad` on
`c1e146a8`, never pushed: an opt-in `[env]` option of the T. rex env, off by
default, that charges a foot whose whole-foot CoP lies outside [−0.6, +0.4]
span half-lengths. The band's code, criteria, judge and Colab steps are not
part of this change: they are being built separately and are not committed
anywhere public.

### Added on 2026-10-22 (decision D-D40's outcome)

**`stance5/`: the stance5 screen of the per-foot CoP band, pre-registered
(decided 2026-10-20 as D-D40; criteria fixed on 2026-10-21 before any training
step; launched that day on Colab L4 runtimes).** Three runs of arm LTSB,
stance4's arm LTS plus the per-foot centre-of-pressure band
(`stance_cop_band_weight` 0.30), from scratch on physics r9, 12M steps each:
`stance5_LTSB_s52`, `stance5_LTSB_s60` and `stance5_LTSB_s42`, from the branch
`claude/screen-trex-r9-armlts-copband` at `cf80ecb1` (stance4's `c1e146a8` plus
the five band commits `f94452ad`, `3a1d5de3`, `4819eb6c`, `a15494ed` and
`99e205bb` and the config commit `cf80ecb1`), pushed on 2026-10-21 and never
merged, with no pull request. The control is the three stance4 runs as already
judged, not retrained.

- `criteria.txt`: what is trained (the term, and what it costs and moves,
  measured on the statue), how it is judged (K1–K4 exactly as stance4's and
  stance3's), the control and its caveats, the credit rule, the drift-or-split
  rule, the decision table, when the decision is read and the pre-registered
  readouts, written before any training step (sha256 `3b28963f…`). It was
  re-fixed after its review on 2026-10-21, before any training step; the first
  text (`c2809201…`) stays in the session's scratch. Its `SP/` and `T/` paths
  name the session's scratch.
- `criteria_notes.md`: where each of the maintainer's decisions of 2026-10-20
  lives in the criteria, and the review's outcome.
- `colab_steps.txt`: how the runs are launched and resumed, as sent; it holds
  the criteria's full text, byte for byte.
- `judge/judge_stance5.py` (judges one checkpoint against its branch clone,
  binding plant, task, symmetry record and commit, and records the band
  readouts), `judge/score_stance5.py` (scores K1–K4 with `score_stance3.py`'s
  own code, loaded unchanged, then the credit rule, the drift-or-split class and
  the decision line, and the report-only readouts) and
  `judge/test_score_stance5.py` (101 test cases).

*Added 2026-10-22:* the band's criteria, judge and Colab steps are now here, and
its code is on the screen branch above, pushed at `cf80ecb1`, whose history
holds the prototype `f94452ad`. This supersedes "being built separately and are
not committed anywhere public" and "never pushed" in the entries of 2026-10-20
(`stance4/night2/` and `stance4/copband/design.md`), which are left as written.

**`stance5/results/`: the stance5 screen judged (2026-10-22, final), with
report-only readouts.** Outcome: Run 1 `stance5_LTSB_s52` **holds** (K1–K4); Run
2 `stance5_LTSB_s60` fails K3 only (3 of 20 joint-space noise falls at the
final, against a bar of 2); Run 3 `stance5_LTSB_s42` fails K1 and K3 (mostly the
actuator-saturation rail, and 13 of 20 joint-space noise falls at the final).
Fore-aft holds on s52 and s60. H = 1 and K2p = 3; the band arm passes K3 on 1 of
3 runs against the control's 2 of 3, so the credit rule's (b) fails too, and the
decision table's line is **3**, final: not adopted, "replicate on 2 more seeds
before any commit". Under the parent rule s52's final ties stance3 Run 2's final
(report only). Measured, all three runs recorded a Colab L4 GPU runtime (the
`gpu` block of each run's `stage_config.json`). Its
[README](stance5/results/README.md) indexes the folder:

- `run_log.txt` (the stance5 log, ids removed, from #633's opening to the
  maintainer's decisions of 2026-10-22 on the replication), `scores/`
  (`score_final.txt` and `.json`, the verdict of record), `checkpoints.csv` (one
  row per judge file, 36, with the `checkpoints_csv.py` that recomputes it with
  the frozen scorer's own counts), `extras.txt` and `extras.py` (the fresh, tail
  and common-noise panels on each final, report only, measured for this record),
  `notebook_gates/` (the notebook's own six-heading gate on each run's
  `robust_best`, trimmed: all three PASS; report only) and `certificates/` (the
  stage's six-heading certificate rolled offline on the s52 and s60 finals,
  trimmed: both PASS; report only, not gate evidence).
- `panels_summary.csv` and `panels_summary.py`: the diagnostic panels
  (mirrored-leg spawns, the prey straight ahead, both) at every J checkpoint of
  the three runs, with the band readouts and the ankle asymmetry, report only.
- `saturation/`: the saturation watch's files for s42, report only: the ledger's
  alarm at 3,670,016, and the actuator named by rollouts of the 4,194,304
  checkpoint, `tail_3_pitch_act` at +1 (tail raised), unchanged at 5,242,880 and
  6,291,456.

### Added on 2026-10-22, later (decision D-D40's 5a)

**`stance4/satdiag/`: the offline diagnosis of stance4 s42's actuator saturation
(2026-10-20 to 2026-10-21, report only).** Deterministic rollouts of stance4
s42's checkpoints, with stance4 s52's and s60's and stance3 Run 2's beside them,
through the judge's own loading path; the five runs' eval ledgers; a
counterfactual; and eight verifiers. Measured: the saturating actuator is
`tail_3_pitch_act`, commanded at +1 (tail tip up), the actuator and sign of
stance3 Run 2; the command is pinned, not the joint (0.216 rad below its stop)
or the motor (0.26 of its force limit), so the ledger's force columns cannot see
it; the onset lies between the 1,048,576 checkpoint (clean) and 2,097,152 (17 of
20 rolled episodes flagged), the ledger flags from 1,835,008 and the alarm comes
at 2,228,224; holding tail_3 at 0 gains +85.9 (at 2,097,152) and +73.1 (at the
final) per 800-step window, deterministically; why training keeps the habit is
not measured. It lists candidate fixes (T1, T2, P1, R1, R2, Y1, N0), none acted
on. None of it enters a verdict.

- `report.md`: the diagnosis as written on 2026-10-21, each line labelled
  measured or inferred, with its verification. Its `O` is
  `scratchpad/foot/stance4/satdiag`, and its Sources section names files that
  stay there (the rollouts, the counterfactual and the verifiers' folders and
  scripts; about 14 MB in all).
- `rollout/tables/summary_tables.md`, `checkpoint_table.txt`,
  `actuator_matrix.md` and `onset_within_episode.txt`: the rollout tables the
  report cites (per checkpoint, the flagged episodes and each actuator's share
  of window steps at 0.9 and 0.99; the window-mean command per actuator; the
  ramp within an episode).
- `ledger/timeline.csv`: the eval-ledger timeline, one row per evaluation of
  s42, s52 and s60 and stance3 Runs 2 and 3 (saturated, clean and full-horizon
  episodes, returns, the centre of mass and the joints, and the training-episode
  columns).

### Added on 2026-10-22, later (decision D-D36's rebuilt pilot 2)

**`pilot2/` (three files added): walking pilot 2 rebuilt for its new parent
(2026-10-21; not yet trained, no result).** Decision D-D36 as amended on
2026-10-20 (7a, option i): pilot 2 warm-starts from stance3 Run 2's final
(`stance3_armLT_s60` at 12,001,280 steps; `stage1_final.zip` `cc21604a…`, its
normaliser `894e86f4…`) instead of r9 arm L seed 60 at 6M, which stays pilot 1's
parent. The new parent is committed on `claude/pilot-walk-r9-b` in three
pilot-only commits on `ba464855` (`6a13a303`, `8fc76eec` and `8c5478a6`; only
`pilot_checkpoints/` changes), pushed on 2026-10-21 at 19:49 UTC with the
maintainer's OK, a fast-forward from `ba464855` to `8c5478a6`; nothing more is
pushed to the branch until all three seeds are judged.

- `criteria_2026-10-21.txt`: the criteria as amended on 2026-10-21, before any
  pilot-2 training step (pilot 2 was never launched): the parent, the branch
  head, the parent's numbers, the readings reworded (pilot 2 now differs from
  pilot 1 in the parent and the noise reset, so no outcome is attributed to
  either alone), the hold on the branch, and the pre-launch checks re-run on
  `8fc76eec` and on `8c5478a6`. No outcome definition, threshold, guard, label,
  seed, budget or tool changes. It cites the 2026-10-18 text, `criteria.txt`
  here (sha256 `a240ff5d…`), as its pre-amendment copy.
- `colab_steps_2026-10-21.txt`: the launch steps for the rebuilt pilot, with the
  push recorded.
- `rebuild_1021_summary.md`: the rebuild's summary, written at 19:16 UTC (its
  file time), before the push (so it lists the push as open): the commits, the
  parent's action std
  (mean 0.064, knees 0.009, max 0.125; pilot 1's parent 0.095, 0.018 and 0.164),
  the criteria amendment, and the pre-launch checks (the notebook smoke; the
  judge accepting the run and refusing pilot 1's parent; 29 warm-start tests;
  the full suites on `8fc76eec`, 5,937 passed).

### Added on 2026-10-24 (decision D-D41's outcome)

**`stance5/replication/`: the stance5 replication of the per-foot CoP band,
pre-registered (decided 2026-10-22 as D-D41; criteria fixed that day before any
training step; launched that day on Colab L4 runtimes).** Two runs of arm LTSB,
unchanged, from scratch on physics r9, 12M steps each: `stance5_LTSB_s70` (Run
4) and `stance5_LTSB_s80` (Run 5), from the branch
`claude/screen-trex-r9-armlts-copband` at `cf80ecb1`, the commit the three
stance5 runs trained, never merged and with no pull request. The three stance5
runs count as recorded and are not re-judged; the stance4 control enters no
decision.

- `criteria.txt`: why it is run, what is trained (as stance5's), the stance5
  runs and the control (not retrained), how it is judged (the judge, the new
  seeds' joint-noise sigma and the scorer), the run handling of 2026-10-22, K1–K4
  as stance5's, the decision table of three lines (adopt only if both new runs
  hold), when the decision is read, the pre-registered readouts, what is
  measured and inferred, and the known limits, written before any training step
  (sha256 `a38536de…`). It was made final at 17:46 UTC on 2026-10-22 (sha256
  `cbf98ba7…`) and re-fixed at 18:00 UTC for the run log's time correction,
  before any training step; the earlier texts are not copied. Its `SP/` paths
  name the session's scratch.
- `colab_steps.txt`: how the runs are launched and resumed, as sent at 18:00
  UTC; it holds the criteria's full text.
- `sigma_s70_s80.json`: the joint-space noise sigma for seeds 70 and 80, one
  vector, the element-wise mean of the frozen `control_sigma` of seeds 42, 52
  and 60 (mean 0.160713), with its computation.
- `judge/judge_stance5r.py` (stance5's judge with `--seed` also taking 70 and
  80, whose sigma it reads from the sidecar, the sigma file recorded, and the
  schema `stance5r-judge/v1`; `judge/judge_diff.txt` is the whole diff),
  `judge/score_stance5r.py` (loads `score_stance5.py` and through it
  `score_stance3.py` unchanged, checks the stance5 runs' recorded verdicts and
  the control's, and reads the replication's decision line) and
  `judge/test_score_stance5r.py`.

**`stance5/replication/results/`: the replication judged (2026-10-24, final),
with report-only readouts.** Outcome: Run 4 `stance5_LTSB_s70` **holds**
(K1–K4); Run 5 `stance5_LTSB_s80` fails K1 only (40, 4, 14, 26, 39 and 38 of 40
clean, the three below 33 on the actuator-saturation rail). H = 2 of 5 (s52 and
s70), so the decision table's line is **2**, final: "Exactly one of s70 and s80
HOLDS (H = 2 of 5): not adopted." The maintainer chooses the next round from
stance5 line 4's menu (the tail saturation habit, the dose, the leg
exploration-noise floor). Under the parent rule s52's final still ties stance3
Run 2's final (report only). Measured, both runs recorded a Colab L4 GPU
runtime (the `gpu` block of each run's `stage_config.json`). Its
[README](stance5/replication/results/README.md) indexes the folder:

- `run_log.txt` (the replication's log, ids removed, from the package made
  final at 17:46 UTC on 2026-10-22 to the entry timed 15:27 UTC on 2026-10-24),
  `scores/` (`score_final.txt` and `score_final.json.gz`, the verdict of
  record), `notebook_gates/` (the notebook's own six-heading gate on each run's
  `robust_best`, trimmed: both PASS; report only) and `certificates/` (the
  stage's six-heading certificate rolled offline on the s70 and s80 finals,
  trimmed: 40/40 and 38/40, both PASS; report only, not gate evidence).
- `saturation/`: the saturation watch's files for s80, report only: the alarm
  at 4,456,448 (first flagged evaluation 4,325,376), and the actuator named by
  rollouts of the 4,718,592, 5,242,880 and 6,291,456 checkpoints,
  `tail_3_pitch_act` at +1 (tail raised).

### Added on 2026-10-24, later (the two report-only workflows of 2026-10-22)

**`stance5/k3diag/`: the K3 diagnosis of the stance5 band runs (2026-10-22,
report only).** A file pass over the judge files and ledgers and a rollout pass
that re-runs the judge's joint-space noise test with ablations (it reproduces
all 12 judged conditions exactly), then four skeptic checks, each by an
independent re-roll or re-reduction; where a check differs from the synthesis,
the check governs, and three of the four corrected it. Measured and checked:
stance5 s42's joint-noise falls are not the tail habit (holding `tail_3` at 0
leaves 0, 9 and 12 falls against 0, 9 and 13 at 6,291,456, 10,485,760 and the
final; tail-only noise gives 0 falls at every checkpoint); hip-pitch noise is
the main cause (alone 8, 3, 9 and 15 of 20 at 7.3M, 8.4M, 10.5M and the final;
the full noise without hip pitch 0, 0, 0 and 2), mostly its left-right
differential part at the judge's real amplitude (5, 2, 7 and 15) against the
common part (0, 0, 0 and 1); control s42's final needs knee and hip-pitch noise
together; the falls are slow topples of about 1 s after a heading-frame stagger
of at least 81 mm, which comes mostly from the pelvis yawing over planted feet,
associated, not a threshold. The policy's direct response to a left-right
hip-pitch difference halves between 6,291,456 and 7,340,032 (−1.04 to −0.51 per
rad), but restoring it from outside leaves the falls: a correlate, and the
mechanism is open. The band-against-control readings (43 against 11 paired
falls, the ±0.3 kick probe) were not independently checked.

- `report.md`: the report as written on 2026-10-22, the skeptic checks first,
  then the synthesis as written before them. Its Files section names files that
  are not copied (`scratchpad/foot/stance5/k3diag/`: the file pass, the rollout
  scripts, tables and npz files, and the verifiers' folders); all but the npz
  files are in the session's scratch and on the never-merged handoff branch
  `claude/handoff-stance5r`, which took text files only.

**`stance5/tailexp/`: the scratch tail experiment, P1 (2026-10-22, report
only).** A quadratic penalty on the policy's Gaussian mean beyond ±1 (P1 of the
stance4 s42 diagnosis, `stance4/satdiag/`), λ 41, in a 1M-step CPU
continuation from stance5 s42's 4,194,304 checkpoint, against a paired no-change
control from the same checkpoint; one seed, one pair of runs, then three skeptic
checks (one corrected the headline). Measured and checked: P1's tail_3 share at
|cmd| ≥ 0.99 was lower than the control's at 8 of 8 periodic checkpoints (means
0.26 against 0.86) and at the final, but not monotone, and it relapsed to 0.470
and 0.488 at 5,242,880 and the final; P1 did not keep the deterministic mean
under +1 (beyond +1 on 21–46% of window steps at 7 of 9 checkpoints), only the
training batches' (2.5% of rows against 27.5%); the original no-penalty GPU run
read 0.171 at 5,242,880, below P1, so the effect against training alone is not
established; no cost was measured, but the window ends before any K3 fall slot.
The code was a scratch clone, `scratch_tailfix` (branch
`scratch/tail-mean-bound` at `1d2d22b4`, push URL disabled), never pushed and
not copied; the report describes it. A real round would need a config key,
criteria fixed before training and the maintainer's approval, since it changes
the PPO training loss.

- `report.md`: the report as written on 2026-10-22, the skeptic checks first,
  then the findings and caveats as returned. Its Files section names files that
  are not copied (`scratchpad/foot/stance5/tailexp/`: the runs, the
  measurements, the telemetry and the verifiers' folders); the scripts,
  `results.csv`, `lambda.json` and `train_telemetry.csv` are in the session's
  scratch and on the never-merged handoff branch `claude/handoff-stance5r`, and
  the runs, the measurement files and the verifiers' folders were not carried.

### Added on 2026-10-24, later (the K3 readouts addendum and its results)

**`stance5/replication/readouts_addendum_k3.txt` and
`stance5/replication/k3readouts/`: report-only K3 mechanism readouts on the
replication (2026-10-22 to 2026-10-24).** The addendum was written at 22:55 UTC
on 2026-10-22 on the maintainer's OK ("Yes please add those reports"), when only
the replication's 1,048,576-step checkpoint had been judged and no J checkpoint
existed; it changes neither the criteria (`a38536de…`) nor the judge, scorer or
sidecar, and says it is post hoc. It fixes, for s70 and s80 at 6,291,456 and the
final, on the judge's own joint-space noise loop: the falls of 20 under the full
noise (which must reproduce the judge file), hip-pitch noise only, the full
noise without hip pitch, and the differential and common parts of the actual
hip-pitch noise; the ±0.2 and ±0.3 left-right hip-pitch kick (0.2 s); the
policy's direct response to a left-right hip-pitch difference; its own action
noise per actuator; and the heading-frame stagger sd. n = 20 per condition, so
differences of 2 or 3 falls are within binomial noise.

- `readouts_addendum_k3.txt`: the addendum, pinned by the tool and unedited.
- `k3readouts/k3readout5r.py` and `k3readouts/k3readout_one5r.sh`: the tool
  (one checkpoint per call; it refuses a judge, addendum, judge file, checkpoint,
  sigma or clone that is not the pinned one) and its runner; `README.md`: their
  record, definitions and the lines they copy from the K3 diagnosis' scripts.
- `k3readouts/validate_k3r.py` and `k3readouts/VALIDATION.md`: the validation
  against stance5's skeptic-checked values (s42 at 6,291,456, 7,340,032 and the
  final, s52's and s60's finals): 223 checked rows, 0 mismatches, written at
  02:16 UTC on 2026-10-23; and the two reviews' 13 minor findings (none
  blocking), what was done about each, and their tests.
- `k3readouts/out/<run id>/k3r_<slot>.json`: the four replication readouts
  (s70 and s80 at 6,291,456 and the final; `addendum_slot` `readout`) and the
  five stance5 references rolled by the same tool (`reference`); all nine are
  valid (the full noise reproduces the judge file per episode, rewards
  included).

Measured (the outputs): falls of 20 under the full noise 0 and 1 (s70 at
6,291,456 and the final) and 1 and 2 (s80); hip-pitch noise only 0, 0, 1 and 1;
without hip pitch 0 throughout; the differential part 0, 0, 0 and 1; the common
part 0 throughout; the ±0.3 kick 18, 20, 20 and 20 of 20, the ±0.2 kick 0, 0, 0
and 1; the direct response −1.023, −0.428, −0.484 and −0.012 per rad; the own
hip-pitch sigma 0.0220, 0.0146, 0.0212 and 0.0149. The references: stance5 s42's
final fell 13 of 20 (hip pitch only 15, the differential part 15), and every
stance5 band final fell 20 of 20 to the ±0.3 kick. None of it enters a verdict.

### Added on 2026-10-25 (decision D-D42's options study and probes)

**`stance5/tailscreen/options/`: the options study for the tail habit
(2026-10-24, report only).** D-D42 (1) asked for the next round to be designed
with fixed criteria and brought to the maintainer before any training; at 15:27
UTC an options study started (mechanism and intervention research, a synthesis
brief and two skeptics, with CPU probes on a copy of the r9 MJCF and no
checkpoint). Nothing in it trained or was fetched from Drive.

- `brief_draft.md`: the draft brief (sha256 of the source `363fb6bd…`), which
  recommended a five-seed from-scratch round of a 0.90 mean bound. Both skeptics
  rejected that design as blocking (the 0.90 rationale unsupported, the alarm
  count near a manipulation check, five seeds unable to separate from chance).
- `brief_v2.md`: the corrected brief written at 16:10 UTC (`903d7218…`, 302
  lines), the version the maintainer decided on: the mechanism re-ranked (the
  deterministic fixed point through the joint-angle loop off the training
  distribution first; the clipped Gaussian's dead zone at most a contributor
  beyond a mean of about 1.2), the options (A, a training-loss bound on tail_3's
  mean, recommended for a CPU screen only; B, tail_3 reach, held; C, a broad
  tail-pose reward term; D, a clipped-action policy gradient, not a round on its
  own), and a replicated CPU continuation screen as the first step. Its Appendix
  A gives each of the skeptics' 17 findings with its check and verdict (all hold
  in substance), and Appendix B what did not hold as stated. Its MECH, LAND,
  SKF, SKD and V2 folders are in the second scratch tree, not copied.
- `decisions_v1.md`: the screen design's decisions list (D1–D9) of 17:37 UTC
  (`3a78622e…`), from which the maintainer took the technical defaults with
  D-D43 at 17:46 UTC.

**`stance5/tailscreen/probes/`: the read of the band runs' `policy_probe.npz`
and the no-training CPU probes (2026-10-24, report only).** The maintainer's
decisions of 15:56 and 16:00 UTC asked for one batched, read-only Drive fetch
(the five band runs' `policy_probe.npz` and the 4,194,304 checkpoint pairs of
stance5 s42 and stance5r s80, whose sha256 equal those in their judge files)
and CPU probes without training before any screen. Measured, on those checkpoints
only:

- `policy_probe_read/`: the coordinator's read of `tail_3_pitch_act`'s raw
  deterministic mean at the probe's ten bank states over training, per run
  (`read_probe.out`).
- `probecheck/`: its independent check. The probe is not an early detector:
  s42's bank values above 1.0 are all on ankle-minus lean rows and do not lead
  the ledger (first saturation episode at 3,145,728, first flagged evaluation
  3,538,944); s80's bank never passed 0.82 while its ledger flagged 37
  evaluations (at least 5 saturation episodes each) from 4,325,376 to
  10,223,616; s52's bank passed 1.0 with no flag.
  `probe_ckpt.py` rebuilds the bank from each checkpoint and matches the npz
  rows.
- `noisevalue/`: what tail-up is worth under the policy's own noise (40 paired
  stochastic episodes, all 15 actuators, the arm's reward, seeds 3042–3081, and
  20 deterministic ones): holding tail_3 at 0 gains reward in 40 of 40 pairs at
  both checkpoints, +54.6 per 200–999 window on s42 (95% CI +51.5 to +57.5) and
  +36.2 on s80 (+31.8 to +40.5), +56.5 and +39.3 deterministically; holding it
  at 0.9 gains +7.6 on s42 (mostly the `action_saturation` term) and loses 13.4
  on s80 (mostly `tail_home_pose`); no falls in any cell.
- `unclip/`: tail_3 unclipped on a scratch MJCF copy (ctrlrange ±0.40 for
  tail_3 only; the wrapper reproduces the clipped baseline exactly): the
  deterministic mean settles at a stable fixed point of a positive loop
  (in-episode dmu/da about 0.5–0.6), spread 0.45–1.07 over s42's episodes, and
  above 1.073 the mean is below the hold in 20 of 20 episodes; s80 never passes
  1 (window mean 0.629, maximum 0.902). Neither "a straight tail" nor "as high
  as possible".
- `skeptic/`: the skeptic's re-runs (bootstrap intervals, a ±0.15 shift of
  s80's tail_3 mean under noise, the npz spot checks and an alternative unclip
  run); no blocking finding and five minor corrections, applied in the run
  log's entry timed 16:52 UTC.

Not supported, at these two checkpoints: a reward incentive for tail-up
(untested: s42 at its onset, 3.1M to 3.5M, and the discounted, normalised
objective). Inferred: an in-episode deterministic readout is a better early
signal than the probe's bank.

### Added on 2026-10-25, later (decision D-D43 and its outcome)

**`stance5/tailscreen/`: the tail_3 bound screen (decided 2026-10-24 as D-D43;
criteria approved at 19:06 UTC and re-pinned at 19:10 UTC that day, before any
training step; run from 19:11 to 23:46 UTC on this container's CPUs; scored at
23:53 UTC, final; verified at 00:13 UTC on 2026-10-25).** Three arms, CONTROL
(coef 0), BOUND90 (b = 0.90) and BOUND60 (b = 0.60), coef 2.4 for both bounds,
each continuing stance5 s42's and stance5r s80's 4,194,304 checkpoints for one
1,048,576-step run on seeds 101, 202 and 303 (18 replicates), from two scratch
commits on `cf80ecb1` (`4c0c8da7`, never pushed). Outcome C: BOUND90 NOT ADOPTED
(RAIL: 6 of 6 replicates had tail-up flags at R1 and R2, s42's from a railed
start and s80's from a clean one, R1/R2 mean commands 0.886 to 0.971) and
BOUND60 NOT ADOPTED (PARKED: 6 of 6 cleared the rail with 0 tail-up flags, but
had R1/R2 mean commands of 0.600 to 0.632, at or above the 0.60 line; s42's
three clear it by as little as 0.00017, and the verdict also holds on s80
alone); CONTROL read RAIL, CURED and RAIL at s42 and RAIL three times at s80
(the onset note INFORMATIVE); OTHER 0 everywhere. Its
[README](stance5/tailscreen/README.md) indexes the folder: the criteria
byte-identical, the approval, request and pins, the calibration, the patch, the
gate records (the first gate's failure on one test of `main`'s rule and the
passing gate after the re-pin), the tools, the results and the verification.

### Added on 2026-10-25, later still (decision D-D44)

**`stance5/tail_sr3/`: T. rex physics r10, the preloaded tail springs (the
posture study 01:02–01:31 UTC, the r10 tail probe 01:34–02:07 UTC and the CPU
checks 02:36–04:27 UTC on 2026-10-25, all without training; decided at 04:55
UTC; built and re-measured on the scratch branch `claude/physics-r10-sr3`,
05:34–06:21 UTC).** The candidate SR3 gives `tail_1_pitch`, `tail_2_pitch` and
`tail_3_pitch` a spring rest angle of 52.5, 25.9 and 10.4° and changes nothing
else, so the zero-action statue rests its tail level on its springs instead of
on `tail_1`'s ventral stop. Its files: the posture study's draft and corrected
summary (`posture/`); the probe's brief, settle sweep, statue panels' summaries,
repository-cost outputs and its skeptic's checks (`probe/`); the CPU checks'
reports (`cpucheck/coord/`), the zero-shot closed loop of eight r9 checkpoints
(`cpucheck/zeroshot/`: 293 → 229 of 320 deterministic episodes clean on SR3,
no falls, two checkpoints pinning a tail actuator at +1 and one losing its
rail), the gate worker's SR3 tree as a patch with its statue panels (278/280
against r9's 275/280; the six-heading certificate 40/40) and its test
classification (`cpucheck/gate/`), the heading-invariance diagnosis
(`cpucheck/heading/`: the test's own check fails 7 of 20 r9 seeds too) and the
skeptic's statistics (`cpucheck/skeptic/`); the re-measurement on the branch
(`remeasure/`: the tail target (0.00002, 0, 0.00006, 0.00018) rad,
`natural_pitch` 0.017, m·g·h 729.1, the stance, recovery and locomotion statue
references 3766.6, 1429.1 and 1092.6, the stance rail 2260, and the
six-heading certificate re-rolled under the final gate block); and the run
log's lines 578 to 603 (`results/run_log.txt`). Its
[README](stance5/tail_sr3/README.md) indexes the folder and gives the skeptics'
corrections that apply to its files.

### Added on 2026-10-27 (decisions D-D45 to D-D51)

**`stance6_setup/`: the run log of the first T. rex stance round on physics
r10, from its set-up to its launch (2026-10-25 08:21 UTC to 2026-10-27 18:41
UTC; lines 600 to 680 of the session's run log).** It holds the landings of
the heading-test fix, physics r10 and the two report-only telemetry changes
with their CI runs; the maintainer's decisions D-D45 to D-D51 (walking pilot
2 retired; the telemetry, no MuJoCo sensor and the in-run K3 panel; the
round's design, its all-or-nothing run handling, its approval and launch)
and the version decision; the criteria drafts and their checks; the run
branch; and the `STANCE6` launch and launch-check lines of the three runs,
`stance6_r10LTSB_s60`, `_s42` and `_s80` (configuration only), also kept on
their own as `results/stance6_launch_lines.txt` (written after the launch).
The runs were training when it was copied, and it holds no result of the
round. Its [README](stance6_setup/README.md) names the decision entries.

### Added on 2026-10-27, later (the compact K3 extract of D-D46)

**`stance5/k3extract/`: the five band runs' judge files cut to the keys the
training-history report's judge-K3 readout reads (the maintainer's decision
of 2026-10-25 17:42 UTC).** One directory per run (`stance5_LTSB_s52`,
`_s60`, `_s42`, `_s70`, `_s80`; 39 files: every `judge_<steps>.json` and
`judge_final.json`), which `plot_training_history.py --judged` reads as it
reads the judge's own, an index `k3_extract.csv` with each source's sha256,
and the script that wrote them. Its readout equals the judge files' for
each run, and its K3 checkpoints give the replication score of record's K3
values (measured); a test in the suite checks it. Its
[README](stance5/k3extract/README.md) names the keys kept.

### Added on 2026-10-27, later still (decision D-D51: the first r10 round's pre-registration)

**`stance6/`: the pre-registration of the first T. rex stance round on
physics r10, approved at 05:55 UTC on 2026-10-27 before any training step;
results pending.** The approved criteria (`99a44469…`, 829 lines) and the
two texts they cite (criteria v12, `0ab5edeb…`, and their evidence,
`353f4536…`), byte copies whose sha256 are `APPROVED`'s; the approval record
`APPROVED` and the request it answered; the launch steps and the fetch
brief; and each tool lane's `SHA256SUMS.txt` with its pinned tools (the
Python files gzipped). Every file but its README existed before the first
training step. The runs were training when it was copied; it holds none of
their files, and the run log's launch and launch-check lines, written after
the launch, are in `stance6_setup/`. Its [README](stance6/README.md)
indexes it.

### Added on 2026-10-28 (decision D-D51's outcome: the first r10 round's results)

**`stance6_results/`: the results of the first T. rex stance round on
physics r10, from its last day of training to the maintainer's decisions
after its review (2026-10-27 19:13 UTC to 2026-10-28 19:32 UTC; lines 681
to 720 of the session's run log).** The run log holds the landing of #647;
the second report-only snapshot of the running files; the three runs'
ends, noted from Drive at the maintainer's decision; the final fetch, its
transfer method decided by the maintainer; the reading and its independent
check; the durable copy; the review and next-step proposals; and the
maintainer's decisions after the review (D-D51's amendment and D-D52). Its
[README](stance6_results/README.md) names the decision entries.

### Added on 2026-10-28, later (decision D-D51's outcome: the reading, readouts and review)

**`stance6_results/score/` to `fetch/`: the round's reading by its approved
criteria, final, and what was read after it (25 files, about 1.8 MB).** The
scorer's output (`score/`: line 1, "TAIL RAIL FORMED ON r10 IN TRAINING
(ledger)", statistic A 2 of 3 against LTSB's r9 2 of 5), the completeness
check per run (`complete/`: all three COMPLETE), the independent check of the
reading (`check1/NOTES.txt`: agrees), the pre-registered diagnostic readouts
and the entropy check (`readouts/`), the review, the next-step proposals, the
fixer's record and the headline sent to the maintainer (`review/`), K1–K4 per
run (`kscore/`), the D-9 and D-10 summaries (`d9_d10/`) and the fetch
manifest and run-folder record (`fetch/`). All but the reading are report
only. Its [README](stance6_results/README.md) indexes it.

### Added on 2026-10-28, later still (decision D-D51's outcome: the panels' summaries)

**`stance6_results/panels/`: the measurement panels' summaries and the
judge's files of the round's three runs (122 files, about 13 MB), byte
copies.** For each traced checkpoint the per-step traces' reduction (M1),
and at the judged checkpoints the 40-episode panel's summary (M2), the
six-heading certificate's summary and report (M3) and the judge's file (M4);
the per-step traces and the per-episode panel files are not here. Report
only but for the M1 figures and the final certificates the classes read.
Its [README](stance6_results/README.md) indexes it.

### Added on 2026-10-29 (decision D-D52's outcome and decisions D-D53 to D-D55: the run log)

**`ecs_results/`: the entropy continuation screen and the work on the T. rex
tail saturation habit that followed it (2026-10-28 21:05 UTC to 2026-10-29
23:21 UTC; lines 721 to 759 of the session's run log).** The run log holds
the maintainer's decisions on the screen's design, its approval and launch,
the check-ins of its 24 CPU continuations, its reading (line 3, no
difference detected) and independent check, the landing of #652, the
screen's durable copy, the centre-of-mass revisit, the tail clamp test, the
normaliser probe, the posture tests, the record of everything tried on the
tail habit, and the maintainer's decisions on each (D-D52's amendment and
D-D53 to D-D55). Its [README](ecs_results/README.md) names the decision
entries.

## How to re-check each verdict

Commands as run on 2026-10-17 from a copy of this directory (Python 3.13;
`PYTHONDONTWRITEBYTECODE=1`). None needs the repository except where stated.
The three r10 rows needed the session's scratch tree and are not reproducible
from the repository alone: the original `frozen_params.json` (sha256
`58b2c8fb…`, which `judge_r10.py` and `score_r10.py` assert) and the r9 judge
files are not kept, so `verify_recompute.json` and `score.json` are the
retained records.

*Added 2026-10-18:* the rows below the first "The files themselves" row were
added that day; those marked 2026-10-18 were run that day on the committed
copies.

| Check | Command | Result |
|---|---|---|
| The symmetry screen's outcome from its stored per-arm records | in `symmetry/verdict_scripts/`: `python3 -c "import json, decide_final as D; P = json.load(open('../frozen_params.json')); V = json.load(open('../verdict.json')); print(D.outcome(V['arms']['L'], V['arms']['C'], P))"` | `O9`, no PR arm, no 12M arms |
| The symmetry decision table | `python3 test_decide_final.py ../frozen_params.json` | `"ok": true`, every combination mapped |
| The r10 verdict from the per-episode rows (session scratch only) | `python3 r10_screen/verify_recompute.py <judge dir> out.json` on the nine judge JSONs (not copied; `judge_r10.py` regenerates them from the Drive checkpoints in `fetch_record.json` on a clone of `claude/screen-trex-r10-arm-l` at `026eb833`, but only with the original `frozen_params.json`) | identical to `verify_recompute.json`: K1-K4 all FAIL |
| The r10 score (session scratch only) | `python3 r10_screen/score_r10.py <judge dir> --json score.json`, with the r9 judge files and the original `frozen_params.json` under `scratchpad/foot/...` | identical to `score.json` apart from the judge directory's path |
| The r10 scorer at every bar (session scratch only) | `python3 r10_screen/test_score_r10.py` (reads the original `frozen_params.json` at `scratchpad/foot/symmetry/design/`) | 33/33 |
| The pilot rule | in `pilot/`: `python3 test_pilot_score.py` | 203/203 (probe recomputation skipped) |
| The pilot rule, with every fixture row recomputed by the frozen probe | `python3 test_pilot_score.py <tree>` with `<tree>` a checkout of `claude/pilot-walk-r9` at `eaeef50f` and numpy, MuJoCo and the repository's dependencies installed | 254/254 |
| The parent rule and the judge's bindings | `python3 test_pilot_parent.py` | 40/40 |
| The files themselves | `sha256sum -c SHA256SUMS` | all 70 files OK |
| Pilot 1's outcome from its stored panels (2026-10-18) | copy `pilot1_judging/OUT` to `<work>`, `gunzip <work>/s*/*/probe.json.gz`, then `python3 pilot/pilot_score.py <work> --seeds 42,52,60 --json score.json` | `PILOT OUTCOME: STILL (3 of 3 launched seeds)`; `score.json` has the original's sha256 `86d7350e…` (`OUT/score.json` is it plus a final newline), and the printout is `judge_logs/score.log.txt` apart from the directory it names |
| Pilot 1's panels themselves (Drive checkpoints) | `pilot/pilot_judge.py` on a clone of `claude/pilot-walk-r9` at `eaeef50f`, with the checkpoints from the Drive folders in `pilot1_judging/run_log.txt` (sha256 in each `binding.json`) | not re-run on 2026-10-18 |
| The ledger's 28 rows against their evidence, and each certificate's LCB against its counts (2026-10-18; needs the repository) | from the repository root, the block below | `28 rows agree with their evidence` |
| A certificate's verdict from its per-episode rows (session scratch only) | `python3 offline_certificates/scripts/rederive.py <dir>` on `main` `2e734e29`, `<dir>` holding the certificate folders with their `stance_heading_panel_selected.csv` (not copied; sha256 at the end of SOURCES.txt) | as `rederive_from_csv.json` here and in `battery/fresh_seeds/`: all 11 and all 8 agree with their reports |
| The certificates themselves (session scratch only) | `offline_certificates/scripts/run_cert.py` on the scratch-harness checkpoints (not on Drive; sha256 in each `run_summary.json`) on `main` `2e734e29` | not re-run on 2026-10-18 |
| The battery's panel-pass illustration (2026-10-18) | `python3 battery/report/panel_pass_prob.py battery/fresh_seeds/fresh_seed_summary.json` (writes `panel_pass_prob.json` in the working directory) | the source's bytes (sha256 `09d0be6b…`; the copy adds a final newline) |
| The push report from its summary (2026-10-18) | `gunzip` a copy of `battery/push/push_summary.json.gz` into `<dir>`, then `python3 battery/push/scripts/make_report.py <dir>` | `push_report.txt`, apart from the summary path it names and one trailing blank line |
| The shared-fall report from its summary (2026-10-18) | in a directory holding a copy of `battery/shared_fall/shared_fall_summary.json` at `scratchpad/foot/battery/shared_fall/`: `python3 <this directory>/battery/shared_fall/scripts/make_report.py` | identical to `shared_fall_report.txt` |
| The audit's geometry summary (2026-10-18) | `python3 audit/geometry/scripts/trex_geom_summary.py audit/geometry/geometry_audit.json out.txt` | the source's bytes (sha256 `1632ed48…`; the copy drops one trailing space) |
| The rest of `battery/` and `audit/` (session scratch only) | their scripts read per-episode panels, traces, MuJoCo rolls on `main` `2e734e29` and downloaded papers, none of which is kept here | not re-run on 2026-10-18 |
| The files themselves (2026-10-18) | `sha256sum -c SHA256SUMS` | all 285 files OK |
| The stance3 scorer at every bar (2026-10-18; session scratch only) | from a directory holding the session's scratch as `scratchpad/` (the scorer asserts the original `frozen_params.json`, sha256 `58b2c8fb…`, and the criteria's `1ce910dc…` there): `python3 -I <this directory>/stance3/judge/test_score_stance3.py` | 80/80 cases as expected |
| The stance3 criteria (2026-10-18) | `sha256sum stance3/criteria.txt` | `1ce910dcd6e2ff7b41a4f592b9f34b9a6c9cb398c523e88f0ebd19af754f18fa` |
| The files themselves (2026-10-18, with `stance3/`) | `sha256sum -c SHA256SUMS` | all 291 files OK |
| The stance3 verdicts, decision line and best final checkpoint (2026-10-19; session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I <this directory>/stance3/judge/score_stance3.py scratchpad/foot/stance3/judged --runs-root scratchpad/foot/stance3/runs --json score.json` | identical to `stance3/results/scores/score_final2.txt` and `.json` apart from the paths they name |
| The stance3 scorer at every bar (2026-10-19; session scratch only) | as on 2026-10-18 | 80/80 cases as expected |
| `stance3/results/checkpoints.csv` (2026-10-19; session scratch only) | `python3 -I stance3/results/checkpoints_csv.py <judged dir> out.csv --check <score_final2.json>` | 28 cells agree with the scorer; `out.csv` identical |
| The files themselves (2026-10-19, with `stance3/results/`) | `sha256sum -c SHA256SUMS` | all 319 files OK |
| The files themselves (2026-10-19, later, with `stance3/night/`) | `sha256sum -c SHA256SUMS` | all 321 files OK |
| The stance4 criteria (2026-10-19) | `sha256sum stance4/criteria.txt` | `978a93da8da469c1acde7b9f7261d9236e4799eb2111b3af0f0cbeaa153f0f38` |
| The files themselves (2026-10-19, later, with `stance4/`) | `sha256sum -c SHA256SUMS` | all 322 files OK |
| The stance4 verdicts, decision line and parent inputs (2026-10-20; session scratch only) | in the session's scratch: `python3 -I -B scratchpad/foot/stance4/judge/score_stance4.py --runs-ended --json score.json` | identical to `stance4/results/scores/score_final.txt` and `.json` apart from the JSON's path and s52's two selection cells, judged after the record |
| `stance4/results/checkpoints.csv` (2026-10-20; session scratch only) | `python3 -I -B stance4/results/checkpoints_csv.py <judged dir> out.csv --check <score_final.json>` | 40 cells agree with the scorer; `out.csv` identical |
| `stance4/results/panels_summary.csv` (2026-10-20; session scratch only) | `python3 -I -B stance4/results/panels_summary.py <panels dir> out.csv --judged <stance4 judged dir> --control-judged <stance3 judged dir>` | `out.csv` identical; the model sha256 and the standard K2a and K2b equal the judge files' |
| `stance4/results/extras.txt` (2026-10-20; session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I -B <this directory>/stance4/results/extras.py` | identical to `stance4/results/extras.txt` |
| The notebook gates' checkpoints (2026-10-20, `stance4/results/notebook_gates/`) | compare each `gate_verdict.json`'s `checkpoint_sha256` with the `robust_best` row of `stance4/results/checkpoints.csv` | equal for all three |
| The files themselves (2026-10-20, with `stance4/results/`, `night/`, `night2/` and `copband/`) | `sha256sum -c SHA256SUMS` | all 344 files OK |
| The stance5 verdicts, decision line and parent comparison (2026-10-22; session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I -B scratchpad/foot/stance5/judge/score_stance5.py --runs-ended --json score.json` | identical to the scratch's `scores/score_final.txt` and `.json` (the copies in `stance5/results/scores/` differ from those only by the prefix change) apart from the JSON's path and s60's two selection cells, judged after the record; the committed copy of the scorer differs as `stance5/results/README.md` says |
| The stance5 scorer at every bar (2026-10-22; session scratch only) | in `scratchpad/foot/stance5/judge/`: `python3 -m pytest -p no:cacheprovider test_score_stance5.py` | 101 passed |
| `stance5/results/checkpoints.csv` (2026-10-22; session scratch only) | `python3 -I -B stance5/results/checkpoints_csv.py <judged dir> out.csv --check <score_final.json>` | 25 cells agree with the scorer; `out.csv` identical |
| `stance5/results/panels_summary.csv` (2026-10-22; session scratch only) | `python3 -I -B stance5/results/panels_summary.py <panels dir> out.csv --judged <stance5 judged dir>` | `out.csv` identical; the model sha256 and the standard K2a and K2b equal the judge files' |
| `stance5/results/extras.txt` (2026-10-22; session scratch only) | from a directory holding the session's scratch as `scratchpad/`: `python3 -I -B <this directory>/stance5/results/extras.py` | identical to `stance5/results/extras.txt` |
| The notebook gates' and certificates' checkpoints (2026-10-22, `stance5/results/`) | compare each `notebook_gates/<run id>/gate_verdict.json`'s `checkpoint_sha256` with the `robust_best` row of `stance5/results/checkpoints.csv`, and each `certificates/<run id>/run_summary.json`'s `model_sha256` with the final row | equal for all three gates and both certificates |
| The stance5 criteria (2026-10-22) | `sha256sum stance5/criteria.txt` | `3b28963f644ffb765e666a8cf6ada5d31ab8a2769dd1a77ff53cbc1e0cf3cf16` |
| The files themselves (2026-10-22, with `stance5/`) | `sha256sum -c SHA256SUMS` | all 372 files OK |
| The files themselves (2026-10-22, later, with `stance4/satdiag/`) | `sha256sum -c SHA256SUMS` | all 378 files OK |
| The amended pilot-2 criteria against their source (2026-10-22; session scratch only) | `diff scratchpad/foot/pilot2/criteria.txt pilot2/criteria_2026-10-21.txt` | line 149 only (the redaction) |
| The files themselves (2026-10-22, later, with the `pilot2/` files) | `sha256sum -c SHA256SUMS` | all 381 files OK |
| The replication's verdicts, decision line and parent comparison (2026-10-24; session scratch only, restored from the handoff branch) | in the session's scratch: `python3 -I scratchpad/foot/stance5/replication/judge/score_stance5r.py --runs-ended --json score.json` | the same verdicts, cells, H and decision line (2) as `stance5/replication/results/scores/`; the differences the handoff's missing files cause are in `stance5/replication/results/README.md` |
| The replication scorer's tests (2026-10-24; session scratch only, restored) | in `scratchpad/foot/stance5/replication/judge/`: `python3 -m pytest -p no:cacheprovider test_score_stance5r.py` | 42 passed, 2 skipped, 1 failed, as `stance5/replication/results/README.md` explains |
| The replication certificates' checkpoints (2026-10-24; session scratch only) | compare each `stance5/replication/results/certificates/<run id>/run_summary.json`'s `model_sha256` with the scratch's `judged/<run id>/judge_final.json` | equal for both |
| The replication criteria (2026-10-24) | `sha256sum stance5/replication/criteria.txt` | `a38536de152e735947fd5322960f9e75b6520b59a5526a414afc0d0c4a0f3ef2` |
| The files themselves (2026-10-24, with `stance5/replication/`) | `sha256sum -c SHA256SUMS` | all 401 files OK |
| The files themselves (2026-10-24, later, with `stance5/k3diag/` and `stance5/tailexp/`) | `sha256sum -c SHA256SUMS` | all 403 files OK |
| The K3 readouts' tool and runner (2026-10-24; session scratch only) | `sha256sum scratchpad/foot/stance5/replication/k3readouts/k3readout5r.py scratchpad/foot/stance5/replication/k3readouts/k3readout_one5r.sh`, and each output's `identity` | `d0aed5a1…` and `5e3c66cd…`, the values all nine copied outputs record |
| The files themselves (2026-10-24, later, with the K3 readouts) | `sha256sum -c SHA256SUMS` | all 418 files OK |
| The briefs (2026-10-25) | `sha256sum` of the sources `scratchpad2/tailround/brief_draft.md` and `brief_v2.md` (session scratch only) | `363fb6bd…` and `903d7218…`, the values SOURCES.txt gives |
| The probes' start checkpoints (2026-10-25; session scratch only) | `probes/probecheck/fetched_sha.txt` against the `judge_4194304.json` of stance5 s42 and stance5r s80 | equal (the zips `d3c90db8…` and `649db94c…`, the pkls `3903cbcf…` and `98fe3043…`) |
| The files themselves (2026-10-25, with `stance5/tailscreen/options/` and `probes/`) | `sha256sum -c SHA256SUMS` | all 471 files OK |
| The tail_3 bound screen's criteria, approval, calibration, gate and score (2026-10-25) | `sha256sum stance5/tailscreen/criteria.txt stance5/tailscreen/APPROVED stance5/tailscreen/calib/coef.json stance5/tailscreen/gate/gate_pass.json stance5/tailscreen/results/score.txt stance5/tailscreen/results/score.json` | `9930ee72…`, `42627d1f…`, `f6d90f99…`, `d7118520…`, `66c4a92d…` and `471ad8e4…`, the values in the run log's entries timed 19:10, 19:11 and 23:53 UTC |
| The screen's outcome from the criteria alone (2026-10-25; session scratch only) | as `stance5/tailscreen/README.md` gives it | `DIFFS []` |
| The files themselves (2026-10-25, later, with `stance5/tailscreen/`) | `sha256sum -c SHA256SUMS` | all 528 files OK |
| The CPU checks' SR3 tree (2026-10-25, later still; needs the repository's history and the session scratch) | `git archive 9f2355a3 \| tar -x -C <dir>`, then `patch -p1 -d <dir> < stance5/tail_sr3/cpucheck/gate/sr3.patch`, and `diff -rq` against `scratchpad2/tailround/r10cpu/gate/tree_sr3` | applies; no difference |
| The r10 constants (2026-10-25, later still) | read `stance5/tail_sr3/remeasure/remeasure_summary.json` against the branch's `base_env.py` and four stage TOMLs | each value as committed |
| The r10 statue's six-heading certificate under the final stance gate block (2026-10-25, later still; the branch's review) | `stance_gate_report.py --species tyrannosaurus_rex --embodiment anatomical --zero-action` on the branch | gate `sha256:d26c34e6…`; 40/40 seeds clean at every heading, as `stance5/tail_sr3/remeasure/panels/stance_heading_report_final_gate.txt` gives |
| The files themselves (2026-10-25, later still, with `stance5/tail_sr3/`) | `sha256sum -c SHA256SUMS` | all 634 files OK |
| The run log's copy against its source (2026-10-27; session scratch only) | `head -680 scratchpad/foot/stance4/live/run_log.txt \| sha256sum`, and its lines 600 to 680 against `stance6_setup/results/run_log.txt` with the ids SOURCES.txt names put back | `f37b68ba…`, the value SOURCES.txt gives; equal |
| The launch lines against the run log's copy (2026-10-27) | `grep ' STANCE6 ' stance6_setup/results/run_log.txt \| cmp - stance6_setup/results/stance6_launch_lines.txt` | no difference: the same six lines |
| The files themselves (2026-10-27, with `stance6_setup/`) | `sha256sum -c SHA256SUMS` | all 636 files OK |
| The K3 extract against the judge files (2026-10-27; session scratch only) | from the repository root, `read_judge_k3` (`environments/shared/reporting/training_history.py`) on each judge directory `k3_extract.csv`'s `source` names and on `stance5/k3extract/<run id>`, with the run's seed | equal for all five runs, the files listed as not read aside |
| The K3 extract against its index and the replication's score (2026-10-27) | `python -m pytest environments/shared/tests/test_training_history.py -k committed_k3_extract` | passes |
| The files themselves (2026-10-27, later, with `stance5/k3extract/`) | `sha256sum -c SHA256SUMS` | all 677 files OK |
| The first r10 round's approved texts (2026-10-27, later still) | `sha256sum stance6/criteria.txt stance6/criteria_v12_cited.txt stance6/evidence.txt`, against `stance6/APPROVED` | `99a44469…`, `0ab5edeb…` and `353f4536…`: its `criteria_sha256`, `this12_sha256` and `evid_sha256` |
| The pinned tools against their lanes' lists (2026-10-27, later still) | in a copy of `stance6/tools/laneA/` and of `laneB/` (with the two `.txt` steps beside it): `gunzip` the `.gz` files, then `sha256sum -c --ignore-missing SHA256SUMS.txt` | 7 and 10 files OK |
| The files themselves (2026-10-27, later still, with `stance6/`) | `sha256sum -c SHA256SUMS` | all 701 files OK |
| The second run-log copy against its source (2026-10-28; session scratch only) | `head -720 scratchpad/foot/stance4/live/run_log.txt \| sha256sum`, and its lines 681 to 720 against `stance6_results/run_log.txt` with the ids SOURCES.txt names put back | `de6c3d73…`, the value SOURCES.txt gives; equal |
| The two run-log copies against each other (2026-10-28) | `tail -1 stance6_setup/results/run_log.txt` and `head -1 stance6_results/run_log.txt` | the entries timed 18:41 and 19:13 UTC on 2026-10-27, the source's lines 680 and 681: no gap and no line in both |
| The files themselves (2026-10-28, with `stance6_results/run_log.txt`) | `sha256sum -c SHA256SUMS` | all 702 files OK |
| The round's reading and the review's texts (2026-10-28, later) | `sed -n '/^READING/,/^STATUS/p' stance6_results/score/score.txt`; `sha256sum stance6_results/review/REVIEW.txt stance6_results/review/NEXT_STEPS.txt` | line 1, "TAIL RAIL FORMED ON r10 IN TRAINING (ledger)", R 1 (COMPLETE 1, PRESENCE 0), U 0, P 1, G+ 1, A 2 of 3 against 2 of 5 (one-sided Fisher p 0.9286), `STATUS: FINAL`; `5c76f556…` and `9fd7fe64…`, the values the run log's entry timed 19:27 UTC gives |
| The redacted run-log lines against the run-log copy (2026-10-28, later) | each run-log line that `score/score.txt`, `complete/*.txt` and the gunzipped `score/score.json.gz` print ("RUNLOG line <n> names a RUN_ID …") against line <n> of `stance6_setup/results/run_log.txt` | equal, or its leading part where the tool shortens it: 16 lines in the four texts and 4 in the JSON |
| The files themselves (2026-10-28, later, with `stance6_results/score/` to `fetch/`) | `sha256sum -c SHA256SUMS` | all 727 files OK |
| The M1 summaries against the reading (2026-10-28, later still) | for each run and tail actuator, the mean of `cmd_a` in `panels/<run>/10485760/m1/m1_summary.json` and `panels/<run>/final/m1/m1_summary.json` (actuators 11 to 14) against the `MEAN` that `score/score.txt` prints on the line after the run's `CLASS` line | equal (12 values; the largest difference 2.8e-17): `tail_3` +0.6420, +0.9148 and +0.5052 for s60, s42 and s80 |
| The panels against the durable copy (2026-10-28, later still; needs the durable branch) | `git show e4733d8:SHA256SUMS` of `claude/handoff-stance6` (its `round/panels/` lines) against `sha256sum` of each file in `stance6_results/panels/` | equal for all 122 |
| The files themselves (2026-10-28, later still, with `stance6_results/panels/`) | `sha256sum -c SHA256SUMS` | all 849 files OK |
| The third run-log copy against its source (2026-10-29; session scratch only) | `head -759 scratchpad/foot/stance4/live/run_log.txt \| sha256sum`, and its lines 721 to 759 against `ecs_results/run_log.txt` with the ids SOURCES.txt names put back | `f2a95d7b…`, the value SOURCES.txt gives; equal |
| The run-log copies against each other (2026-10-29) | `tail -1 stance6_results/run_log.txt` and `head -1 ecs_results/run_log.txt` | the entries timed 19:32 UTC and 21:05 UTC on 2026-10-28, the source's lines 720 and 721: no gap and no line in both |
| The files themselves (2026-10-29, with `ecs_results/run_log.txt`) | `sha256sum -c SHA256SUMS` | all 850 files OK |

`test_pilot_score.py` prints sample "Walking pilot score" blocks while it runs;
they score synthetic panels, not the pilot.

The ledger check (from the repository root, with its dependencies installed):

```python
import json
from environments.shared.curriculum.recovery_gate import binomial_lcb
from environments.shared.reporting import load_offline_judgements

rows = load_offline_judgements()
for r in rows:
    ev = json.load(open(r["evidence"]))
    if r["gate_kind"] == "stance_quality_heading/v1":
        assert abs(binomial_lcb(ev["n_clean_seeds"], ev["n_seeds"]) - ev["clean_seed_lcb"]) < 1e-12
        got = (ev["passed"], ev["n_clean_seeds"], ev["n_seeds"], round(ev["clean_seed_lcb"], 4),
               "sha256:" + ev["model_sha256"])
        want = (r["verdict"] == "PASS", r["passing_count"], r["panel_size"], r["lower_bound"],
                r["checkpoint_sha256"])
    else:  # walking pilot 1: the outcome in score.json, the checkpoint in binding.json
        seed, step = r["run_path"][-2:], str(r["checkpoint_step"])
        cell = ev["seeds"][seed][step]
        binding = json.load(open(r["evidence"].replace("score.json", f"s{seed}/{step}/binding.json")))
        got = (cell["outcome"], cell["counts"]["WALK"], sum(cell["counts"].values()),
               binding["checkpoint_sha256"])
        want = (r["verdict"], r["passing_count"], r["panel_size"], r["checkpoint_sha256"])
    assert got == want, (r["evidence"], got, want)
print(len(rows), "rows agree with their evidence")
```

It checks what the rows say against the files here; it does not re-run a
judgement.

## Lint

Under the repository's ruff configuration (ruff 0.16.9, the version CI and
pre-commit pin; 0.16.8 gives the same), `ruff check` passes on
14 of the 16 scripts here. It reports 11 findings in
`r10_screen/verify_recompute.py` (a verifier's quick script: a multi-module
import, an unused `os`, statements joined by `;` and `:`) and 2 in
`symmetry/verdict_scripts/recount_final.py` (an unused `math`, a variable named
`I`). `ruff format --check` would reformat all 16. They are left as they ran,
so that each stays the file that decided its verdict (and the verdict scripts
keep the sha256 the screen recorded).

The 43 scripts added on 2026-10-18 (ruff 0.16.8, the same configuration) are
left as they ran too, apart from the prefix and the two lines of trailing
whitespace SOURCES.txt records. `ruff check` passes on 12 of them and reports
193 findings in the other 31, all of them quick-script style: statements
joined by `;` (67) or `:` (19), unsorted imports (31), several modules on one
import line (23), variables named `l` or `O` (16), imports below
code (14), unused imports (14), f-strings without placeholders (4), stray
semicolons (2), and one each of a bare `except`, a lambda assigned to a name
and an unused variable. Most are in the verifiers' scripts
(`battery/verify_skeptic/`, 91). `ruff format --check` would reformat all 43.

The three `stance3/judge/` scripts added later on 2026-10-18 pass `ruff check`
under the same configuration (ruff 0.16.9); `ruff format --check` would
reformat all three. They are left as they ran.

Of the two scripts added on 2026-10-19 in `stance3/results/`,
`checkpoints_csv.py`, written for the record, passes `ruff check` and
`ruff format --check` under the same configuration (ruff 0.16.8);
`certificates/run_cert_clone.py` is left as it ran, with 8 findings (several
modules on one import line, unsorted imports, imports below code), and
`ruff format --check` would reformat it.

Of the three scripts added on 2026-10-20 in `stance4/results/`,
`checkpoints_csv.py` and `panels_summary.py`, written for the record, pass
`ruff check` and `ruff format --check` under the same configuration (ruff
0.16.8); `extras.py` is left as it ran, with 3 findings (several modules on
one import line, unsorted imports, two statements joined by `;`), and
`ruff format --check` would reformat it.

*Added 2026-10-22:* of the six scripts added in `stance5/`, the three in
`stance5/results/`, written for the record (`checkpoints_csv.py` and
`panels_summary.py` adapted from stance4's), pass `ruff check` and `ruff format
--check` under the same configuration (ruff 0.16.9). The three in
`stance5/judge/` are left as they ran: `ruff check` reports 4 findings in
`score_stance5.py` (an unused `math` and three variables named `l`) and none in
the other two, and `ruff format --check` would reformat all three. All six parse
on Python 3.11, 3.12 and 3.13.

*Added 2026-10-24:* the three scripts added in `stance5/replication/judge/` are
left as they ran: `ruff check` (ruff 0.16.9, the same configuration) reports 1
finding in `test_score_stance5r.py` (an f-string without placeholders) and none
in the other two, and `ruff format --check` would reformat all three. All three
parse on Python 3.11, 3.12 and 3.13.

*Added 2026-10-24, later:* the two scripts added in
`stance5/replication/k3readouts/` are left as they ran: `ruff check` reports no
finding in either, `ruff format --check` would reformat both, and both parse on
Python 3.11, 3.12 and 3.13.

*Added 2026-10-25:* the 21 scripts added in `stance5/tailscreen/probes/` are
left as they ran: `ruff check` (ruff 0.16.9, the same configuration) passes on
`noisevalue/roll_value.py` and `unclip/analyze.py` and reports 125 findings in
the other 19, quick-script style (most in `probecheck/probe_ckpt.py`, 22, and
`skeptic/unclip_alt.py`, 19); `ruff format --check` would reformat all 21, and
all 21 parse on Python 3.11, 3.12 and 3.13.

*Added 2026-10-25, later:* the 16 Python scripts added in `stance5/tailscreen/`
(one, `tools/run_tailscreen.py`, kept gzipped) are left as they ran: `ruff check` (ruff 0.16.9, the same configuration) passes
on the four tools other than `run_tailscreen.py` and on the four tools' tests,
and reports 107 findings in the other eight (15 in `tools/run_tailscreen.py`,
the rest in the verifiers' quick scripts, most in `verify/rescore/rescore.py`,
32); `ruff format --check` would reformat all 16, and all 16 parse on Python
3.11, 3.12 and 3.13. The three shell scripts in `tools/` are left as they ran.

*Added 2026-10-25, later still:* the eight Python scripts added in
`stance5/tail_sr3/remeasure/scripts/` (kept gzipped) are left as they ran:
`ruff check` (ruff 0.16.9, the same configuration) passes on five of them and
reports 27 findings in the other three (`analyze_panels_r10.py` 8,
`cop_probe.py` 10, `statue_v2_blocks.py` 9: statements joined by `;`, imports
after the `sys.path` line that points them at the tree they measure, unsorted
or joined imports); `ruff format --check` would reformat six; all eight parse
on Python 3.11, 3.12 and 3.13. The two shell scripts there are left as they
ran.
