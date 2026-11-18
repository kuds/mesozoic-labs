Verdict: merge; archive tip: f71cba4bc3c14ae97ad29caedc5423f2a4e93e6a; resume tip: 93805771344507063ce713e9236bce7c8bb5a714

Verdict: merge. I checked every claim that matters against my own tool output, and each one holds. The only flaw is a small wording error in the report's own notes (point 1 under Minor findings).

## (A) The archive commit
- **Parent and branch.** f71cba4b has one parent, 9c2d2a52, so nothing was rewritten. It sits on `claude/mesozoic-trex-stage2-archive-a1` in /home/user/mesozoic-wt/archive-a1. That worktree is clean, ignored files included.
- **Origin is unchanged.** `ls-remote` shows the archive branch at 9c2d2a52, the working branch at 23bfbfcc and s2-b1 at ecf95a81. No tags point at either new commit.
- **What changed.** `git diff --name-status 9c2d2a52 f71cba4b` lists four files: MANIFEST.json and README.md modified, frozen-04.tar and frozen-05.tar added. The stat is 4 files, 798 insertions and 355 deletions, with 40263680 and 1884160 bytes for the two new tars, as reported.
- **Old parts.** frozen-01.tar to frozen-03.tar have the same sha256 at both commits: 9baed6c2…, 3ea1ab3b… and d2d3ca06….
- **The 67 files are exactly the right ones.** I ran `tools/restore_inputs.py` in the main checkout at 23bfbfcc:
  - `listed('sets')` gives 136 files and `agreement('sets')` returns [].
  - The manifest's 67 new entries are exactly the tool's V0/F25F9/F25F9cl paths. Each has the tool's sha256, and each matches the main checkout file's actual sha256 and size.
  - The 52 old entries are byte-identical to 9c2d2a52's manifest.
  - 9c2d2a52's `sets_files_rebuilt_by_restore` list is the same 67 files with the same sha256.
- **By set:**

| set | files | look / refs / transitions | bytes |
|---|---:|---|---:|
| F25F9 | 15 | 3 / 3 / 9 | 9,726,013 |
| F25F9cl | 26 | 6 / 6 / 14 | 16,020,730 |
| V0 | 26 | 6 / 6 / 14 | 16,344,570 |
| **all** | **67** | | **42,091,313** |

  The largest of the 67 is 1,384,064 bytes. The manifest's totals are 119 files and 161,645,299 bytes, and recomputing them gives the same.
- **Parts.** I extracted the commit with `git archive`, then the parts with tarfile's "data" filter.
  - All 5 parts have the manifest's size and sha256, and each is under 50,000,000 bytes.
  - Members are 13, 20, 19, 65 and 2. In every part they are exactly the manifest's files for that part, in path order.
  - Every member is a ustar regular file: mode 0644, mtime 0, uid and gid 0, empty names, no pax headers. Each part ends in zero blocks.
  - The manifest's files, content_bytes and sets fields are right for every part.
  - The extraction gave 119 files and 161,645,299 bytes. sha256 mismatches: 0 against the manifest, 0 against the tool's list.
  - The 17 listed files not in the archive are V6/75-M-H-10's refs (6) and transitions (11). They equal the manifest's `in_bundle_s2_b1` list, with matching sha256.
- **Split rule.** Restarting a part when it would pass 41,000,000 bytes, applied to each of the two runs of parts, reproduces 13/20/19 and 65/2 (counted either by content bytes or by tar bytes).
- **Independent rebuild.** I rebuilt frozen-04 and frozen-05 in memory from the main checkout's files under bundle.py's tar rules. The result is byte-identical: b0d5b9af3eddf276200aad1e72ce44c54edf7b75d81dfad71b60f0dd46767276 and d30cc7fcb071392dcaa9ea4cd870febef74dfbd785869b037aac00a769c3a1ee.
- **README.** Its counts, sizes, the part table, the restore commands and what to expect all match what I saw. It has no links to resolve.

## Restore proof in my own fresh clone
I cloned origin and detached at 23bfbfcc. I fetched the archive from the local repo (FETCH_HEAD f71cba4b) and s2-b1 from origin (FETCH_HEAD ecf95a81), symlinked the main checkout's .venv, and followed the README's commands.
- `sha256sum` printed the five part hashes above.
- s2-b1's two parts printed 11d7bd3e…2387 and 6739c8b7…ba39. Both appear in s2-b1's MANIFEST.json.
- Extraction gave 136 files: F25F9 15, F25F9cl 26, V0 26, V6/75-M-H-10 23, V6/75-M-H-10-X 23, V6noF5 23.
- Before copy-from, `check --group sets` printed `sets 136 0 136 0 agree`, EXIT=1.
- `copy-from X --group sets` printed `== sets: copied 136, already here 0`, then `sets 136 136 0 0 agree` and "all here with their listed sha256", EXIT=0.
- The full `check` then printed `sets 136 136 0 0 agree`, with the six other groups (6, 84, 60, 60, 30 and 73 files) all missing and all agreeing, EXIT=1. `check --group sets` printed 136 of 136, EXIT=0.
- `git status` showed only `?? hardware/trex_form_study/.venv`.

I did not run `restore`.

## (B) The RESUME.md commit
- 93805771 is on `wt/resume-fix2` in /home/user/mesozoic-wt/resume-fix2. Its parent is 23bfbfcc, the worktree is clean, and it is not on origin.
- The diff touches only hardware/trex_form_study/RESUME.md: 58 lines, 33 insertions and 25 deletions. Every hunk is about the archive:
  - the kept row: 119 files, about 162 MB;
  - not kept: 313 files, about 208.7 MB, about 2.5 h;
  - step 0 pins f71cba4bc3c14ae97ad29caedc5423f2a4e93e6a and adds a fallback for when the branch reads 9c2d2a52;
  - step 3: 119 + 17 = 136, none rebuilt; 313 other files; the restore comment; the order paragraph;
  - what to expect: five hashes, copy-from 136 with exit 0, a full check exiting 1 before restore;
  - "With no archive branch" now says 119.
- I confirmed these numbers against the tool: the six other groups are 313 files and 208,677,412 bytes in the main checkout, and 449 = 136 + 313.
- No other line of RESUME.md states the archive's contents or size. The group j row still says correctly that j is not in the archive.
- All 65 relative links resolve.

## Commit messages
Both messages end with exactly the two required lines, and the last byte is the newline after the session URL. Neither names a model anywhere else.

## Main checkout
It is still at 23bfbfcc, and `git status --porcelain --untracked-files=all` prints 0 lines, before and after my runs.

## Minor findings (none blocks a merge)
1. The report says the worktrees note still says "about 120 MB" or describes the 52-file state. It does not: stage2/design/checkpoint/worktrees-2026-11-18.md only says that the archive branch came later. DECISION_LOG P49 ("about 120 MB") and the reviews are historical records and do state the old state.
2. The manifest's file entries use the key `bytes`, not `size`. That is the schema of 9c2d2a52's manifest, and the README calls it "size". This is consistent, not a defect.
3. The order matters on origin. f71cba4b must be pushed before wt/resume-fix2 lands on the working branch, or RESUME.md's step 0 will name a commit origin lacks. Step 0's 9c2d2a52 fallback covers that case.

## Not done by me
I did not run `restore`, a full `check` to exit 0, or any tests. I pushed and committed nothing.

Scratch data copies are removed. My scripts and their output remain in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/sets67/check/:
- verify.py
- rebuild.py
- lists.py
- links.py
- mylists.json
- copyfrom.out
