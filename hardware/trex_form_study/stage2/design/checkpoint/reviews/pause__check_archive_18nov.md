Verdict: merge; tip: be47419033b8705e8943072bbea1ced2ed98b32a

## Review of archive a1 (P49): accept. The archive matches the job, with one gap the lead should decide on.

I checked everything below myself, from tool output I ran. I changed nothing in the main checkout: `git status --porcelain` printed 0 lines there at the start and the end. I did not commit, push or tag anything.

### The branch
- `claude/mesozoic-trex-stage2-archive-a1` is at be47419033b8705e8943072bbea1ced2ed98b32a. It is a root commit (no parents), so it is a true orphan.
- Its tree holds only 8 files under `hardware/trex_form_study/stage2/archive/a1/`: MANIFEST.json, README.md and parts/inputs-01.tar to inputs-06.tar.
- The worktree `/home/user/mesozoic-wt/archive-a1` is clean, and its MANIFEST matches the committed one byte for byte.
- The branch is local only: `git ls-remote origin` has no such branch.
- The commit message ends with exactly the two attribution lines. The only place it names a model is the Co-Authored-By line.

### The list matches restore_inputs.py
I imported the tool and called `listed(g)` for all 7 groups, in the main checkout at c5274905. That commit's only change from b1d5e98e is one line in DECISION_LOG.md (`git diff --stat`), so the lists are the same as at b1d5e98e.

| group | listed | present with the listed sha256 | bytes |
|---|---:|---:|---:|
| picks-models | 6 | 6 | 2,954,007 |
| picks-arrays | 84 | 84 | 54,654,576 |
| k | 60 | 60 | 38,419,843 |
| x | 60 | 60 | 41,310,856 |
| j | 30 | 30 | 19,190,577 |
| look | 73 | 73 | 52,147,553 |
| sets | 136 | 67 | 42,091,313 |
| **total** | **449** | **380** | **250,768,725** |

- 69 listed files are missing and none has a wrong sha256.
- The manifest holds exactly these 380 paths, and every entry's group, size and sha256 is right (0 differences either way). The largest file is 3,543,737 bytes.
- `restore_inputs.py check` in the main checkout exited 1. Its output was identical to the manifest's `check_at_source`, apart from the manifest's extra `EXIT=1` line.
- One small error in the item report: its "Verbatim" check block shows the `sets missing:` line with 1 path and then "... 63 more". The tool actually printed 6 paths and then "... 63 more". This is only in the report, not in the archive.

### Extraction proof (my own, from the committed blobs)
- I ran `git archive be474190` into a fresh scratch dir, and then opened each part with Python's tarfile using the "data" filter.
- Each part's size and sha256 match the manifest. Every part is under 50,000,000 bytes, the largest being 47,872,000.
- Every member is a regular file with a relative path. Each part's members are exactly the manifest's files for that part: 79, 70, 76, 63, 78 and 14.
- The extraction gave 380 files with no extras and no duplicates. Every sha256 matched both the manifest and the main checkout's file (0 mismatches each way).

### The README's restore steps, run as written in scratch
I cloned fresh from origin (22 s) and checked out the README's branch, which is at c5274905 on origin. Two steps had to be swapped, both marked in the script:
- **Fetch:** `git fetch` came from the local repository, not origin, because the archive branch is not pushed.
- **Venv:** `.venv` was a symlink to the main checkout's venv instead of `setup_venv.sh`. I did not run `setup_venv.sh`.

Everything else ran as written:
- `sha256sum parts/*.tar` printed the manifest's six hashes, and extraction gave 380 files.
- `copy-from` copied 6/84/60/60/30/73/67 files, reported 69 "not copied ... not made", and exited 1.
- `check` exited 1, and its output was identical (diffed) to the main checkout's.
- The clone's `git status` was clean.

This is what the README's "What to expect" section says. **The README route needs the branch pushed** (P49 allows the push; I did not push).

### The 69 missing set arrays: the claims hold
- The 69 missing files are V6 on 75-M-H-10, V6 on 75-M-H-10-X and V6noF5 on 75-M-H-10, 23 each. The manifest's `not_archived` list is exactly this set.
- 17 of them are in bundle s2-b1's committed parts with the listed sha256 (V6 on 75-M-H-10, refs and transitions).
- On this machine, all 69 are in `/home/user/mesozoic-wt/wp-p42` and 65 are in `wp-l`, with the listed sha256.
- `git log --all` finds none of them committed on any branch.
- **So 52 frozen arrays (119,553,986 bytes) exist only in worktrees on this machine** and are lost when it is deleted, unless `restore --group sets` rebuilds them. That rebuild was not tried, by the worker or by me.
- This follows the task's rule to archive from the main checkout only, and the gap is disclosed, so I do not count it against the work. But the lead should put it to the owner:
  - Adding the 52 from wp-p42 as a seventh part would bring the total to about 370 MB.
  - Adding all 69 would bring it to about 403 MB, just over the 400 MB stop line.

### The left-out list: right
I categorised every git-ignored file under the study:

| what | files | bytes |
|---|---:|---:|
| stage2 eval traces | 12,620 | 2,080,952,168 |
| .venv | 35,959 | 1,350,226,829 |
| logs | 190 | 19,572,346 |
| caches | 127 | 16,477,939 |
| zero-byte done markers | 108 | 0 |
| MUJOCO_LOG.TXT | 1 | 357 |

- Nothing else is there.
- The per-run trace sizes match the report: 86.1, 976.9, 895.3, 5.9 and 116.8 MB.
- `not-restored` printed 0 files for each item. It listed 12,728 other files (the traces plus the markers), and the look clips' media are kept outside the checkout.

### Not done by me
- I did not run `setup_venv.sh`, `restore --group sets`, or a restore combining this archive with the bundle.
- I did not run a test suite.

My scratch copies (the clone and the extractions, about 1.3 GB) are removed, and the disk shows 11 GB free. My scripts and their outputs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/pause/check_archive/`.
