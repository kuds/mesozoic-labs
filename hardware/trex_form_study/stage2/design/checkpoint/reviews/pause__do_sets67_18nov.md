Both items are done and committed, and nothing was pushed or tagged. On origin the archive branch is still at 9c2d2a52 and the working branch at 23bfbfcc (checked with `ls-remote` after the commits). The main checkout is still at 23bfbfcc, and `git status --porcelain --untracked-files=all` there printed 0 lines both before and after.

## Commits
- **(A) `f71cba4bc3c14ae97ad29caedc5423f2a4e93e6a`** on `claude/mesozoic-trex-stage2-archive-a1`, in worktree `/home/user/mesozoic-wt/archive-a1`. Its parent is 9c2d2a52. It was not pushed, and the worktree is clean, ignored files included.
- **(B) `93805771344507063ce713e9236bce7c8bb5a714`** on the new branch `wt/resume-fix2`, in the new worktree `/home/user/mesozoic-wt/resume-fix2`. Its parent is 23bfbfcc, and the worktree is clean.
- Both commit messages end with exactly the two required lines and name no model anywhere else.

## The 67 files
- I listed them with `restore_inputs.listed('sets')` in the main checkout at 23bfbfcc. `agreement('sets')` returned `[]`.
- `git diff 61ae6fa0 23bfbfcc` shows no change in `stage2/sets` or in `tools/restore_inputs.py`, so the lists are the same as at 61ae6fa0.
- They are exactly the 67 that 9c2d2a52's MANIFEST lists as "rebuilt by restore": same paths and same sha256. 52 + 17 + 67 = the 136 listed.
- I read each file from the main checkout (read only) and compared its sha256 with the tool's list before packing: 0 mismatches.

| set on 75-M-H-10 | files | bytes |
|---|---|---:|
| F25F9 | 15 (`look` 3, `refs` 3, `transitions` 9) | 9,726,013 |
| F25F9cl | 26 (`look` 6, `refs` 6, `transitions` 14) | 16,020,730 |
| V0 | 26 (`look` 6, `refs` 6, `transitions` 14) | 16,344,570 |
| **all** | **67** | **42,091,313** |

The largest of the 67 is 1,384,064 bytes. The archive now holds 119 files, 161,645,299 bytes in all.

## Parts
They follow the same tar rules as the existing parts and the same 41,000,000-byte split, applied in path order over the 67 files:

| part | bytes | files | sets | sha256 |
|---|---:|---:|---|---|
| frozen-04.tar | 40,263,680 | 65 | F25F9, F25F9cl, V0 | b0d5b9af3eddf276200aad1e72ce44c54edf7b75d81dfad71b60f0dd46767276 |
| frozen-05.tar | 1,884,160 | 2 | V0 | d30cc7fcb071392dcaa9ea4cd870febef74dfbd785869b037aac00a769c3a1ee |

- `frozen-05.tar` exists only because I kept the documented 41,000,000-byte cap. A single part of about 42.15 MB would also have been under 50 MB.
- Two builds of the new parts gave identical sha256.
- `frozen-01.tar` to `frozen-03.tar` are unchanged: same sha256, and git shows no change to them.
- The five parts total 161,761,280 bytes.

## MANIFEST.json and README.md
**MANIFEST.json:**
- `files` now has 119 entries, each with path, set, body, kind, bytes, sha256 and part.
- `parts` has 5 entries.
- `totals`: 119 files, 161,645,299 bytes, 5 parts, 161,761,280 bytes of parts.
- `group_sets`: 136 listed, 119 in the archive, 17 in s2-b1, 0 rebuilt, with a per-set breakdown.
- New keys: `updated_at`, `history` (the two commits) and `source_of_the_67` (main checkout at 23bfbfcc, with its `check` output). `source` is tagged as the source of the 52.
- `tar.split` describes the two runs of parts.
- `not_in_this_archive` now covers only the six other groups: 313 files, 208,677,412 bytes in the main checkout. `restore_order` is updated.
- `made_at` and `schema` are unchanged.

**README.md:**
- Contents: 119 files, about 162 MB, with a table of all six sets and the five parts.
- Restore: `copy-from --group sets` now restores all 136 sets files with s2-b1, prints "copied 136", exits 0, and `restore` rebuilds no set.
- Its "Checked" section describes the check of 9c2d2a52 (restated from the old README, not re-run by me) and my checks made before this commit. A README cannot report a check of its own commit, so the proof of the commit itself is only in this report.

## Proof on f71cba4b (after the commit)
1. **Extraction.** I extracted the commit with `git archive`, then the parts with tarfile's "data" filter into a new scratch directory.
   - Each part's size and sha256 equal the manifest's, and each is under 50,000,000 bytes.
   - Every member is a plain ustar regular file: mode 0644, mtime, uid and gid 0, empty names, no pax headers.
   - Each part's members are exactly the manifest's files for that part, in path order: 13, 20, 19, 65 and 2.
   - The extraction gave 119 files and nothing else, 161,645,299 bytes.
   - sha256 mismatches: 0 against the manifest, 0 against the tool's list. Size mismatches: 0.
   - The 17 listed files not in the archive are exactly V6/75-M-H-10 `refs` (6) and `transitions` (11).
2. **Fresh clone.** I cloned origin and detached at 23bfbfcc. I fetched the archive from the local repo (FETCH_HEAD f71cba4b) and s2-b1 from origin (FETCH_HEAD ecf95a81), and symlinked `.venv`.
   - `sha256sum` printed the five part hashes above, plus `11d7bd3e…` and `6739c8b7…` for s2-b1's two V6 parts. Those equal the values in s2-b1's MANIFEST.
   - Extraction gave 136 files.
   - Before copy-from: `sets 136 0 136 0 agree`, EXIT=1.
   - `copy-from X --group sets` printed `== sets: copied 136, already here 0`, then `sets            136      136       0     0  agree`, then "all here with their listed sha256", EXIT=0.
   - The full `check` then printed `sets 136 136 0 0 agree`, with the other six groups all missing and all agreeing, EXIT=1.
   - `check --group sets` printed 136 of 136, EXIT=0.
   - `git status` showed only `?? hardware/trex_form_study/.venv`.

I ran the same extraction check and fresh-clone run once before committing, using the worktree's files. The results were identical.

## diff --stat
- **(A) 9c2d2a52..f71cba4b:** 4 files changed, 798 insertions(+), 355 deletions(-).
  - MANIFEST.json | 1008
  - README.md | 145
  - frozen-04.tar Bin 0 -> 40263680 bytes
  - frozen-05.tar Bin 0 -> 1884160 bytes
- **(B) 23bfbfcc..93805771:** `hardware/trex_form_study/RESUME.md` | 58 (33 insertions, 25 deletions).

## RESUME.md changes (B)
- **What is kept:** the archive row now reads 119 files, about 162 MB.
- **Not kept:** 313 files, about 208.7 MB, rebuilt in about 2.5 h.
- **Step 0:** pins `f71cba4b…`. Its parent 9c2d2a52 holds only the 52. A new line says that if the branch reads 9c2d2a52, `restore --group sets` (never tried) rebuilds the 67.
- **Step 3:**
  - 119 + 17 = 136 sets files, none rebuilt; 313 other files rebuilt.
  - The `restore` comment now reads 313.
  - The order paragraph now says 136.
  - What to expect: five part hashes; `copy-from` copies 136 and exits 0; a full `check` exits 1 until `restore` has run. This is described as run in a fresh clone at 23bfbfcc with the archive at f71cba4b.
  - "With no archive branch" now says 119.
- All 65 relative links resolve.

## Not done, and points for you
- I did not run `restore`, a full `check` to exit 0, or any tests. I pushed nothing.
- `f71cba4b` must be pushed before RESUME.md's step 0 is true on origin.
- These still say "about 120 MB" or the 52-file state, and I left them unchanged:
  - DECISION_LOG P49 (the owner's record);
  - the reviews and the worktrees note.
- Scratch data copies are removed. Scripts and logs (no data) remain in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/sets67/`; the proof log is `post_proof.log` there. 9.5 GB of disk is free.
