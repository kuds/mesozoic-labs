# The first T. rex stance round on physics r10: its pre-registration (approved 2026-10-27, results pending)

The pre-registration of the round that decisions D-D49 to D-D51
([BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2) set up:
arm LTSB of the stance5 screens ported to T. rex physics r10, three runs from
scratch (`stance6_r10LTSB_s60`, `_s42` and `_s80`), 12M steps each on Colab
L4s, from the never-merged branch `claude/screen-trex-r10-armltsb` at
`8737cd3d`. **Everything in this folder but this README existed before the
first training step.** Each file is a byte copy (each Python tool gzipped;
gunzip restores it byte for byte) of a session-scratch file last changed
between 06:31 UTC on 2026-10-26 and 05:54 UTC on 2026-10-27, when the approval
record `APPROVED` was written; the maintainer approved the criteria at 05:55
UTC, and the runs were launched at about 14:57 UTC that day, their sessions
starting from 15:01 to 15:03 UTC. Every sha256 here but the request's and
`APPROVED`'s own was fixed at the approval (`APPROVED` names the request's
path): the criteria's and the two cited texts' by `APPROVED`, the Colab
steps', the fetch brief's and the two lanes' lists' by the criteria, and each
tool's by its lane's list. `APPROVED` itself has been read-only since 05:54:35
UTC (its `approved_at_utc` reads 05:54:34Z), and the run log's approval entry
names it by its sha256 prefix `07246480`. This README, an index, was written
for this record after the launch. The run log's six `STANCE6` lines (the three
launches and their configuration-only launch checks), written after the
launch, from 14:57 to 15:54 UTC, are kept with the run log, in
[`../stance6_setup/results/stance6_launch_lines.txt`](../stance6_setup/results/stance6_launch_lines.txt),
not here. The runs were training when this folder was copied (2026-10-27):
**results pending**. Nothing here is a result of the round, and the
launch-check fetches' copies of the runs' own files are not copied. The record
is §26 of
[TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md](../../TREX_STANCE_SYMMETRY_AND_FOOT_2026_10.md),
and the run log of the round's set-up is
[`../stance6_setup/`](../stance6_setup/README.md).

Every file here is a byte copy of its source (the tools gzipped), absolute
session-scratch paths included, so that each sha256 is the one the
approval and the criteria pin. [SOURCES.txt](../SOURCES.txt) gives each
file's source and sha256; [SHA256SUMS](../SHA256SUMS) pins every file but
this README.

- **`criteria.txt`**: the approved criteria (lean v6 with its pins filled,
  829 lines), sha256
  `99a44469a387d01c55b98d76595d3633029681866ed14afcb185521eca0164db`.
- **`criteria_v12_cited.txt`** and **`evidence.txt`**: the two texts it
  cites by line (criteria v12, `0ab5edeba04b319a4233a4700ebf9abf7d7d7127c30afc7565a7fa3bc34369ac`)
  and for its tables (its evidence, `353f4536ef3c0e06a1937714c44a22c7d52819be2e613cbc1ad23b47821f4e20`).
  The three sha256 equal `APPROVED`'s `criteria_sha256`, `this12_sha256` and
  `evid_sha256`.
- **`APPROVED`**: the approval record (JSON, sha256 `07246480…`): the three
  sha256, the branch head `8737cd3d` and its parent `cf9402bc`, the
  validations (V0–V5, V7 on the parent and the head, V8 (i) and (ii), V9,
  V10, all PASS), the scoring interpreter and the budgets the maintainer
  confirmed.
- **`APPROVAL_REQUEST.txt`**: the request the maintainer approved (written
  02:55–03:50 UTC on 2026-10-27): what was asked, the V-items' results with
  the files behind them, and the open items, none blocking.
- **`colab_steps_r10round.txt`** and **`fetch_agent_brief.txt`**: the launch
  steps (sha256 `f2eb9a24…`) and the brief for the read-only Drive fetches
  (`24ac49d6…`), as the criteria pin them.
- **`tools/laneA/`** and **`tools/laneB/`**: each lane's `SHA256SUMS.txt`
  (54 and 133 files; sha256 `55a21707…` and `932f2050…`, the values the
  criteria pin) and the pinned tools: lane A's judge (`judge_r10ltsb.py`,
  made by `make_judge_r10ltsb.py`), its K-criteria scorer
  (`kscore_r10ltsb.py`), the measuring runner (`measure_r10round.py`,
  `r10m_common.py`, `zs_harness_copy.py`) and `runs/run_validations.sh`;
  lane B's ledger, completeness, K3-panel, scoring and fetch tools
  (`ledger_r10round.py`, `complete_r10round.py`, `k3panel_r10round.py`,
  `score_r10round.py`, `fetch_r10round.py`, `v9_bundle.py`,
  `r10round_common.py`) and its pins file `pins_r10round.json`. Each Python
  file is kept as `<name>.py.gz` (gunzip restores it byte for byte, with the
  sha256 its lane's list gives), as the folder keeps scripts that the
  repository's tests would otherwise parse. Their tests, outputs and reports
  are listed in the two `SHA256SUMS.txt` files and stay in the session's
  scratch.

To check a lane: in a copy of `tools/laneA/` (or `laneB/`), `gunzip` the
`.gz` files, then `sha256sum -c --ignore-missing SHA256SUMS.txt`; the files
here all pass. `sha256sum criteria.txt criteria_v12_cited.txt evidence.txt`
gives `APPROVED`'s three values.
