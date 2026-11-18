## Report: slim archive a1 and the RESUME.md fixes

Both items are done and committed. Nothing was pushed or tagged, and no file in the main checkout was changed: `git status --porcelain --untracked-files=all` there prints 0 lines.

### (A) The slim archive

**Branches**
- The first build was renamed locally: `wt/archive-a1-full` = be474190.
- In `/home/user/mesozoic-wt/archive-a1`, I ran `git switch --orphan claude/mesozoic-trex-stage2-archive-a1`. The index and tree came out empty.
- **Commit: `9c2d2a52bd7d7c8c4661c49efaf76b23a44ff359`.** It is a root commit (no parents), local only (`ls-remote origin` shows 0 lines), and the worktree is clean.
- The tree holds `hardware/trex_form_study/stage2/archive/a1/{MANIFEST.json, README.md, parts/frozen-01..03.tar}`.
- The message ends with exactly the two attribution lines.

**Deriving the 52**
- From `restore_inputs.listed("sets")` in the main checkout at 61ae6fa0: 136 files.
- The three sets on 75 cm (V6, V6 on the -X head, V6noF5) have 69 files. These are exactly the 69 the main checkout lacks (`tally`: 67 present, 69 missing, 0 wrong).
- s2-b1's committed MANIFEST at `origin/...bundle-s2-b1` (ecf95a81) lists 17 sets files. All 17 are among the 69, with equal sha256. **69 = 52 + 17.**
- The 52 were read from wp-p42 (`wt/wp-p42` at f9dc89cc). It is an ancestor of 61ae6fa0, and `git diff` shows no change in `stage2/sets` or `restore_inputs.py`.
- All 52 sha256 equal the tool's list. `check --group sets` in wp-p42 printed 136 of 136 matching, records agree, exit 0.

**The 52 files, by set (119,553,986 bytes in total)**

| set | files | bytes |
|---|---|---:|
| V6 on 75-M-H-10 | `look` 6 (nominal, v0.3 to v0.5) | 17,227,350 |
| V6 on 75-M-H-10-X | `look` 6, `refs` 6, `transitions` 11 (nominal_to_v0.3/0.35/0.4/0.45, start, stop1_v0.4, stop2_v0.4, turn90_0.35 and 0.5, left and right) | 52,293,386 |
| V6noF5 on 75-M-H-10 | the same 23 | 50,033,250 |

**Parts** (ustar, the tar rules of bundle.py, split in path order at a 41,000,000-byte cap)

| part | bytes | files | sha256 |
|---|---:|---:|---|
| `frozen-01.tar` | 39,956,480 | 13 | `9baed6c20965c1ceaaa4c3d399d3667116085a6291d4817a714c4147f8c8f476` |
| `frozen-02.tar` | 40,499,200 | 20 | `3ea1ab3be35aaa0642d751961c057835e1325cae912b2e4e0f7dbdd205830059` |
| `frozen-03.tar` | 39,157,760 | 19 | `d2d3ca06c57c8d62bb6f1e39493e08932dd7c9132d374497b321b06cebcdb78f` |

Two builds of the parts gave identical sha256.

**MANIFEST.json** holds:
- every file's path, set, body, kind, size, sha256 and part;
- the parts' sizes and sha256;
- the source commit 61ae6fa0 and the wp-p42 worktree and commit;
- `check` as printed in wp-p42 and in the main checkout;
- the 17 files in s2-b1, with their sha256 and parts;
- the other 380 files (the six other groups and the 67 V0/F25F9/F25F9cl set arrays, 250,768,725 bytes, recomputed at 61ae6fa0), rebuilt by `restore`;
- the restore order.

**README.md** covers what the branch is and why, the fresh-clone restore (fetch both branches, extract, `copy-from <dir> --group sets`, then `restore`, then `check`), and what to expect.

**Proof**
1. **Extraction.** I extracted `git archive` of the commit with the tarfile "data" filter.
   - Each part's size and sha256 equal the manifest's, and each is under 50,000,000 bytes.
   - Each part's members are exactly that part's manifest files: 13, 20 and 19.
   - The extraction gave 52 files and no others, 119,553,986 bytes.
   - sha256 mismatches: 0 against the manifest, 0 against the tool's list. Size mismatches: 0.
2. **Fresh clone.** I cloned from origin, detached at 61ae6fa0, fetched the archive from the local repo and s2-b1 from origin (ecf95a81), and symlinked `.venv`.
   - `sha256sum` printed the three part hashes above, plus s2-b1's `11d7bd3e...` and `6739c8b7...`. Extraction gave 69 files.
   - Before copy-from: `sets 136 0 136 0 agree`.
   - `copy-from X --group sets`: "copied 69, already here 0", listing all 23 of each of the three V6 sets. It reported the 67 V0, F25F9 and F25F9cl files as "not made" and exited 1.
   - **Then `check` printed `sets 136 files, 69 matching, 67 missing, 0 wrong, agree`.** The 67 missing are V0, F25F9 and F25F9cl, which are not restored in that clone. The other six groups were all missing and all agreeing. Exit 1.
   - `git status` in the clone showed only `?? .venv` (the symlink).
   - The clone and the extraction dirs are removed. The full log is `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/slim/proof.log`.
3. **Commit order.** The proof ran on a first commit, 84a01217. I then amended it only to add the "Checked on 18 Nov" section to README.md. `git diff --stat 84a01217 9c2d2a52` shows only README.md (+28), and the parts' and MANIFEST's blob ids are identical.

### (B) RESUME.md and the bundle note

- New worktree `/home/user/mesozoic-wt/resume-fix`, branch `wt/resume-fix` from 61ae6fa0.
- **Commit: `aee38d0077191c6b1557b54144b67af3971bd4d8`.**
- `diff --stat 61ae6fa0..HEAD`:
  - `hardware/trex_form_study/RESUME.md` | 99 +++++++++++++---------
  - `.../design/checkpoint/worktrees-2026-11-18.md` | 6 +-
  - 2 files changed, 63 insertions(+), 42 deletions(-)

**RESUME.md changes**
- **P51 at the top:** a new "Paused, not dropped (P51)" paragraph (flagship end 2027 / early 2028; Triceratops the Q3 2027 museum demo), and "for the museum outreach" added to section 1's P51 paragraph.
- **V6noF5:** now "V6 without F5's toe-off (V6noF5), with the plain head."
- **"What is kept" table:** the archive row now reads 52 files, about 120 MB, with the other 17 in the data branch.
- **"Not kept":** now says the other 380 files (about 251 MB) are rebuilt in about 2.5 h.
- **Removed:** the "One gap" paragraph and the "archive's gap" row in section 3.
- **Step 0:** the archive branch should read 9c2d2a52; be474190 was an earlier build that was not pushed.
- **Step 3, rewritten:**
  - the three sources: 52 + 17 + 380 = 449;
  - the commands in order: both branches, `copy-from --group sets`, `restore`, `check`;
  - what to expect, from the proof;
  - the 2.5 h caveat and the 2-CPU note.
- **The group-j row** no longer says its files are in the archive.

**worktrees-2026-11-18.md:** "at 03:55 UTC 18 Nov", plus a sentence that the archive branch came later and goes to origin by its own push (P49).

All 65 relative links in RESUME.md resolve. The worktrees note has none.

### Not done, and points for the lead
- **RESUME.md has no sentence saying the four branches were the only local ones with commits not on origin.** Only the worktrees note says it, so for item (2) I changed only that note.
- **RESUME.md pins the archive commit 9c2d2a52.** If the archive commit is amended before the push, step 0 must change too. The part sha256 values in step 3 would stay valid.
- **The 67 V0/F25F9/F25F9cl set arrays (42.1 MB) are also frozen and not in s2-b1.** The first build held them; this archive does not. They, group `j` and the picks' models now depend on `restore`:
  - `restore --group sets` has never been tried;
  - `j` and `sets` were never timed in a fresh worktree;
  - the picks' models need at least 2 CPUs.

  The README and RESUME.md say this. Adding the 67 would bring the archive to about 162 MB.
- **I did not run `restore` or a full `check` exit 0, any tests, or a push.**
- The worktrees note's last paragraph ("They reach origin only after...") is stale since 61ae6fa0. I left it as it was.
- The scripts and small logs (no data copies) are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/slim/`. The disk shows 11 GB free.
