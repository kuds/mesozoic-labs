# The tail-habit record: its ledger, build notes and checks (2026-10-29)

The ledger and the build record of
[TREX_TAIL_SATURATION_HABIT_2026_10.md](../../TREX_TAIL_SATURATION_HABIT_2026_10.md),
one record of everything tried on the T. rex tail saturation habit from
2026-07-09 to 2026-10-29, asked for by the maintainer at 18:18 UTC on
2026-10-29 and put in the repository at the maintainer's answer of 23:21 UTC
(decision D-D55 (1) and (3),
[BEHAVIOR_RECIPES_PLAN.md](../../../BEHAVIOR_RECIPES_PLAN.md) §6.2; the run
log's lines 753 and 759, [`../ecs_results/run_log.txt`](../ecs_results/run_log.txt)
lines 33 and 39). Records only: the record decides nothing and certifies
nothing.

[SOURCES.txt](../SOURCES.txt) lists each file's source, its sha256 and the
changes the copy made; [SHA256SUMS](../SHA256SUMS) pins them (every file of
the folder but this README, which has no source). The sources are in the
session's scratch, `scratchpad2/tail_record/`.

## The files

- **`ledger.csv`**: one row per merged entry, L001 to L187 (187 rows and a
  header; rows L168 to L187 were appended on 2026-10-30, as the last section
  says), the record's bracketed ids. Columns: id, date, kind, title,
  verdict, label, what, result, sources, merged_from (the sweep entries it
  merges), where_in_record, sources_disagree. Its source is the draft's
  `draft/ledger.csv` as the update's fact check left it (sha256 in
  SOURCES.txt); the copy carries **redact** (its 7 session workflow run ids,
  each after the word "workflow") and **lineends** (its 168 CRLF line ends,
  as Python's `csv` module writes them, made LF, as the repository's
  `.gitattributes` (`text=auto`) stores a text file, so that its pinned
  sha256 holds on a fresh checkout); no field holds a line break. Its
  `sources` column cites files as the draft did: repository paths at `main`
  `378ae2f8`, the record's short names (ECS, COMREV, CLAMP and the others,
  which the record's table maps to this change's folders) and scratch paths,
  which the record's §8.2 maps to the copies, or marks as not kept.
- **`build_ledger.py`**: the merge map that built rows L001 to L153 from the
  three source sweeps (byte copy of the draft's `build/build_ledger.py`). The
  sweeps themselves (`sweep_repo/`, `sweep_session/` and
  `sweep_today/entries.json`, 91, 85 and 43 entries) are session scratch, not
  kept; each row's `merged_from` names the entries it merges.
- **`fix/FIXES.txt`** and **`fix/numcheck.py`**: the fixer's record of the
  first draft's two checks (a completeness critic, 24 findings, and a
  number-by-number fact check, 31; 54 fixed and 1 not valid, each with its
  evidence) and its number check, each fixer-touched number or quote read at
  its cited lines (byte copies). The fixer's edit scripts and pre-fix copies
  are not kept.
- **`update/CHANGES.txt`**: the update of about 23:01 UTC that added the
  normaliser probe and the posture tests (rows L157 to L167), every edit and
  its source (**redact**: 2 workflow run ids). Its edit script and
  pre-update copies are not kept.
- **`update_check/FIXES.txt`** and **`update_check/check_cites.py`**: the
  update's fact check, each finding FIXED or NOT VALID with its evidence
  (**redact**: 2 workflow run ids), and the script that compares the ledger
  ids each section cites with `where_in_record` (byte copy). Its pre-check
  copies, diff and row extract are not kept.

The scripts read the session's scratch paths they name and are kept as
records of what was run, not to be run from here.

## How the record was built

1. 18:18 UTC: three source sweeps, of the repository's records at `main`
   `378ae2f8`, of the session's records and of the day's work, 219 entries,
   merged with two writer entries into rows L001 to L153 by
   `build_ledger.py`; the draft written from them.
2. A completeness critic and a number-by-number fact check read the draft; a
   fixer verified each finding against its source, applied the valid ones
   (rows L154 to L156 added) and re-checked links, ledger ids, tables and
   numbers (`fix/FIXES.txt`). Drafted by 20:11 UTC (the run log's line 755),
   with the normaliser probe's result left as a placeholder.
3. 22:39 to 23:20 UTC: the update added the verified normaliser probe and
   posture tests (§3.6, §4.8, §4.11 and rows L157 to L167) and brought §1.1,
   §2, §4.7, §5.2, §6, §7 and §8 up to date (`update/CHANGES.txt`); its fact
   checker checked the added text against the five notes, the run log and
   the cited files: 31 findings, 19 fixed, 12 not valid
   (`update_check/FIXES.txt`; the run log's lines 757 and 758).
4. The draft, so checked (sha256
   `1f1e7bdd2c7959a5df78a328c3360b3dad1b6c60725c6554b2a187cdf5d58fc7`, 1,891
   lines), became the repository note with these edits, each replacing text
   that occurs exactly once, and nothing else: the status block (where it is,
   what it does and does not change, how it was checked, with links to the
   files here); in "How to read this record", the "Links and paths" bullet
   (repository links, line numbers at `378ae2f8`, what a remaining scratch
   path means) and "The run log" bullet (every line now in a repository
   copy); the short names ECS, COMREV, CLAMP, VCLAMP, POL, VPOL, PROBE,
   VPROBE, PTIME, PPOL and VPOST, now links to this change's folders, and
   DVREV, marked as not kept; §3.6's opening paragraph; five session workflow
   ids removed from rows L129, L131, L151, L153 and L160; the scratch paths of
   row L152 (now PROBE's) and of rows L056, L058, L059 and L061 and §6.2 item
   11 (marked as not kept, with where their content is found where it is);
   in §8, the heading and evidence list of §8.1, §8.2's run-log paragraph,
   range table and scratch-file rows, and §8.5. No finding, number, label,
   ledger id or verdict changed.

## The update of 2026-10-30: rows L168 to L187

The records change of decision D-D58 (3) added §9 to the record (the work of
2026-10-29 23:47 UTC to 2026-10-30 16:32 UTC: the early-bias probe, the
early-warning curve and its follow-ups, and the two critic-blind screens)
and appended rows L168 to L187 to `ledger.csv`: one row per entry of §9.1
(L168 to L186) and one for §9.8's lessons (L187). §9 and these rows were
written in the repository from the copies that change keeps and the run
log's lines 760 to 801 (`../critic_blind_results/run_log.txt`); there was no
session draft, sweep or merge map for them, so `build_ledger.py` rebuilds
none of them, and their `merged_from` names the update's own entries, "N01
(update of 2026-10-30; …)" to "N20 (update of 2026-10-30)". Their `sources`
cite the copies by their paths from `docs/investigations/`
(`stance_2026_10/overnight_probes/…` and the like) and the run log by the
copy's line with the source's "RUNLOG lN". The rows were appended as
Python's `csv` module writes them (every field quoted, LF line ends), by a
script of the change's build notes (session scratch, not kept) that first
checks that the last row is L167, so the file's first 168 lines are
unchanged (sha256
`ff3acd39336355b45c0e85a02a9ba0c9060bf24c35eb5e3d84a649fa9d4a9012`, before
and after); SOURCES.txt marks the copy's row with the kind **appended**. §9
cites no row before L168, and no earlier row changed. The ids each section
of §9 cites equal the rows' `where_in_record` (`update_check/check_cites.py`
on the note and the ledger), but for the status block, §9's opening and
§9.9, which name the rows' range, as the status block and §8.5 do for the
earlier rows. §9 and these rows have not been through the record's
completeness critic or its fact checks.
