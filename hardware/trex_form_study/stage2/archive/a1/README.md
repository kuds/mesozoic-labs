# Archive a1: the 119 frozen T. rex set arrays that no other branch holds

This branch, `claude/mesozoic-trex-stage2-archive-a1`, is an orphan data branch that is never merged. It holds only
this directory: 119 git-ignored arrays of the T. rex study's frozen Stage 2 reference sets, packed as five tar parts.
With the 17 in bundle s2-b1, they are all 136 arrays of the frozen sets. It is not code and not records. The study's
code and records are on the working branch `claude/mesozoic-trex-stage2-refine-4jffre`.

## Why

- The owner paused T. rex development on 18 Nov 2026 and moved the project to a Triceratops build (P47 in
  `hardware/trex_form_study/DECISION_LOG.md`). The T. rex stays on the roadmap (P51).
- Git ignores the study's arrays (`*.npz`), so a fresh clone has none. `tools/restore_inputs.py` lists each one with
  its sha256 and rebuilds them from the committed records.
- The owner chose to keep the frozen T. rex arrays that bundle s2-b1 does not hold on a never-merged data branch, so
  that a restart needs no rebuild of them (P49). They are about 162 MB.
- Until this archive, these arrays were only in checkouts on the cloud machine, which is deleted after a while.
- This branch has two commits:
  - `9c2d2a52` holds the 52 arrays of the three sets frozen on 17 Nov (about 120 MB, `frozen-01.tar` to
    `frozen-03.tar`). It left the 67 arrays of the three sets frozen on 15 Nov to `restore`, and
    `restore --group sets` has never been tried.
  - The commit after it adds those 67 (about 42 MB, `frozen-04.tar` and `frozen-05.tar`). The first three parts are
    unchanged.
- An earlier build of this branch (`be474190`) held instead the 380 files that `restore` rebuilt then (the six other
  groups and these 67), and not the 52. It was not pushed.

## What it holds

119 files, 161,645,299 bytes. They are the arrays that `restore_inputs.py` lists in group `sets` and that bundle s2-b1
does not hold. The lists are those of commit `61ae6fa0e611cb44ca4f5cef88557911ee4a9ad6` of the working branch, and they
are the same at `23bfbfccda93633232d64d34d335a838d1e6799e`: `git diff` between the two shows no change in
`stage2/sets` or in `tools/restore_inputs.py`.

| set | body | frozen | listed in `sets` | in this archive | in bundle s2-b1 | bytes in this archive | parts |
|---|---|---|---:|---|---|---:|---|
| F25F9 | 75-M-H-10 | 15 Nov | 15 | 15 (`look` 3, `refs` 3, `transitions` 9) | 0 | 9,726,013 | 04 |
| F25F9cl | 75-M-H-10 | 15 Nov | 26 | 26 (`look` 6, `refs` 6, `transitions` 14) | 0 | 16,020,730 | 04 |
| V0 | 75-M-H-10 | 15 Nov | 26 | 26 (`look` 6, `refs` 6, `transitions` 14) | 0 | 16,344,570 | 04, 05 |
| V6 | 75-M-H-10 | 17 Nov | 23 | 6 (`look`) | 17 (`refs` 6, `transitions` 11) | 17,227,350 | 02 |
| V6 | 75-M-H-10-X | 17 Nov | 23 | 23 (`look` 6, `refs` 6, `transitions` 11) | 0 | 52,293,386 | 01, 02 |
| V6noF5 | 75-M-H-10 | 17 Nov | 23 | 23 (`look` 6, `refs` 6, `transitions` 11) | 0 | 50,033,250 | 02, 03 |
| **all** | | | **136** | **119** | **17** | **161,645,299** | |

- V6 is the lifelike walk the owner chose. 75-M-H-10-X is the 75 cm body with the expressive head.
- V6noF5 is V6 without F5's toe-off (`hardware/trex_form_study/stage2/design/WP_L.md`).
- V0, F25F9 and F25F9cl are the three earlier walks, frozen on 15 Nov (the `freezes` of their `set.json`).
- The 52 V6 and V6noF5 arrays were read from the worktree of `wt/wp-p42` (`f9dc89cc`) on the cloud machine. That
  commit is an ancestor of `61ae6fa0`, and the set records and `tools/restore_inputs.py` are the same at both. There
  `restore_inputs.py check --group sets` printed 136 files, 136 matching, records agree, and exited 0.
- The 67 V0, F25F9 and F25F9cl arrays were read, read only, from the main checkout of the working branch at
  `23bfbfcc` on the cloud machine. There `check` printed `sets 136 files, 67 matching, 69 missing, 0 wrong, agree`:
  these 67 were there, and the 69 V6 and V6noF5 arrays were not.
- Each file's sha256 was compared with the tool's list before packing. The largest file is 4,008,709 bytes.

`parts/frozen-NN.tar` are uncompressed ustar archives, each under 50 MB. Their member paths are relative to
`hardware/trex_form_study`, in path order within each part. Each member has mtime 0, uid and gid 0, empty user and
group names and mode 0644 (the tar rules of `designlab/stage2/bundle.py`). The parts come in two runs: `frozen-01.tar`
to `frozen-03.tar` hold the 52 V6 and V6noF5 arrays, and `frozen-04.tar` and `frozen-05.tar` the 67 V0, F25F9 and
F25F9cl arrays. Within each run, in path order, a new part starts when the next file would take the part past
41,000,000 bytes.

| part | bytes | files | sets | sha256 |
|---|---:|---:|---|---|
| `frozen-01.tar` | 39,956,480 | 13 | V6 on 75-M-H-10-X | `9baed6c20965c1ceaaa4c3d399d3667116085a6291d4817a714c4147f8c8f476` |
| `frozen-02.tar` | 40,499,200 | 20 | V6 on 75-M-H-10-X, V6 on 75-M-H-10, V6noF5 on 75-M-H-10 | `3ea1ab3be35aaa0642d751961c057835e1325cae912b2e4e0f7dbdd205830059` |
| `frozen-03.tar` | 39,157,760 | 19 | V6noF5 on 75-M-H-10 | `d2d3ca06c57c8d62bb6f1e39493e08932dd7c9132d374497b321b06cebcdb78f` |
| `frozen-04.tar` | 40,263,680 | 65 | F25F9, F25F9cl and V0 on 75-M-H-10 | `b0d5b9af3eddf276200aad1e72ce44c54edf7b75d81dfad71b60f0dd46767276` |
| `frozen-05.tar` | 1,884,160 | 2 | V0 on 75-M-H-10 | `d30cc7fcb071392dcaa9ea4cd870febef74dfbd785869b037aac00a769c3a1ee` |
| **all** | **161,761,280** | **119** | | |

`MANIFEST.json` holds:

- every file, with its study path, set, body, kind, size, sha256 and part;
- the parts, with their sizes and sha256;
- the sources: commit `61ae6fa0` and the wp-p42 worktree for the 52, the main checkout at `23bfbfcc` for the 67, with
  `check` as it printed there;
- the branch's two commits (`history`);
- the 17 V6 arrays in bundle s2-b1, with their sha256 and parts;
- the six other groups, which `restore` rebuilds, and the restore order.

## Not in it, and where it is

The tool's seven groups list 449 files. This archive holds 119 of them. The rest:

- **17 in bundle s2-b1.** V6 on 75-M-H-10, its `refs` (6) and `transitions` (11). They are on the pushed data branch
  `claude/mesozoic-trex-stage2-bundle-s2-b1` (`ecf95a81`), in `hardware/trex_form_study/stage2/tb/bundle/s2-b1/parts/`
  `V6-75-M-H-10-1.tar` and `V6-75-M-H-10-2.tar`. Their members are at study paths, each with the listed sha256.
- **313 files (208,677,412 bytes) rebuilt by `restore_inputs.py restore`** from the committed records: the groups
  `picks-models` (6), `picks-arrays` (84), `k` (60), `x` (60), `j` (30) and `look` (73). `check` at `23bfbfcc` reports
  every group's list in agreement with the committed records.
- **No file of group `sets`.** After `copy-from`, all 136 are in place, so `restore` rebuilds no set.
- **The time.** About 2.5 h: on 14 Nov, `restore` of five groups (`picks-models`, `picks-arrays`, `k`, `x`, `look`)
  took 151 min on 2 processes in a fresh worktree (`tools/README.md`). The rebuild of `j` was not timed in a fresh
  worktree, so the whole may take longer.
- **At least 2 CPUs.** On one CPU the picks' ledger JSON cannot be reproduced (`SIZE_BLAS_THREADS` in
  `restore_inputs.py`). Then bring those files in with `copy-from`; bundle s2-b1's `models-75-M-H-10.tar` holds
  75-M-H-10's two.
- The eval traces are not inputs and are not kept (P49).

## How to restore, in a fresh clone

Run these from the clone's root. The lists that `restore_inputs.py` checks are those of the commit you check out:
`61ae6fa0` or `23bfbfcc` (the same lists), or any later commit whose lists are the same.

```bash
git clone https://github.com/kuds/mesozoic-labs && cd mesozoic-labs
git checkout claude/mesozoic-trex-stage2-refine-4jffre        # or the commit you restart from
ROOT=$(pwd)
A=$(mktemp -d) B=$(mktemp -d) X=$(mktemp -d)
git fetch origin claude/mesozoic-trex-stage2-archive-a1
git archive FETCH_HEAD hardware/trex_form_study/stage2/archive/a1 | tar x -C "$A"
git fetch origin claude/mesozoic-trex-stage2-bundle-s2-b1
git archive FETCH_HEAD hardware/trex_form_study/stage2/tb/bundle/s2-b1/parts | tar x -C "$B"
(cd "$A/hardware/trex_form_study/stage2/archive/a1" && sha256sum parts/*.tar)   # compare with MANIFEST.json "parts"
for p in "$A"/hardware/trex_form_study/stage2/archive/a1/parts/frozen-*.tar \
         "$B"/hardware/trex_form_study/stage2/tb/bundle/s2-b1/parts/V6-75-M-H-10-*.tar; do tar xf "$p" -C "$X"; done
cd "$ROOT/hardware/trex_form_study"
TORCH=cpu bash tools/setup_venv.sh                             # if there is no .venv yet (tools/README.md)
PYTHONPATH=. .venv/bin/python tools/restore_inputs.py copy-from "$X" --group sets
rm -rf "$A" "$B" "$X"
PYTHONPATH=. .venv/bin/python tools/restore_inputs.py restore   # the six other groups: about 2.5 h, see above
PYTHONPATH=. .venv/bin/python tools/restore_inputs.py check
```

- **The order matters.** `copy-from` comes first, so that `restore` finds all 136 `sets` arrays in place. `restore`
  rebuilds only missing or wrong files, so it then leaves them alone and rebuilds no set.
- **Run `restore` detached** so that no time limit stops it, with a keep-alive (`tools/README.md`, "A fresh
  container, in order", steps 3 and 4). If it is stopped, run it again: files in place are kept.
- **`copy-from`** puts each listed file in place only when its sha256 is the listed one. It leaves a right file alone
  and writes no tracked file.

What to expect:

- `sha256sum` prints the five sha256 of the parts table above.
- Extracting the five parts and s2-b1's two V6 parts gives 136 files.
- `copy-from ... --group sets` prints "copied 136, already here 0". Its closing `check` of `sets` prints
  `sets 136 files, 136 matching, 0 missing, 0 wrong, agree` and "all here with their listed sha256", and it exits 0.
- A full `check` before `restore` prints `sets` at 136 of 136 matching and the six other groups missing, and exits 1.
- After `restore`, `check` should print every group matching and agreeing, and exit 0. `restore` was not run with this
  archive.
- `check` must exit 0 before any run that reads the inputs.

## Checked on 18 Nov 2026

1. **The first commit, `9c2d2a52` (the 52).** Its three parts were extracted with Python's tarfile ("data" filter);
   every file's size and sha256 equalled the manifest's and the tool's list (0 mismatches). In a fresh clone at
   `61ae6fa0`, `copy-from --group sets` with s2-b1's two V6 parts copied 69 files, and `check` then printed
   `sets 136 files, 69 matching, 67 missing, 0 wrong, agree`. The 67 missing were those of V0, F25F9 and F25F9cl,
   which the second commit adds.
2. **The second commit's files, before the commit (the 119).** The five parts as written in this directory, with the
   sha256 of the parts table, were checked:
   - Two builds of `frozen-04.tar` and `frozen-05.tar` gave identical sha256. `frozen-01.tar` to `frozen-03.tar` kept
     the sha256 they have at `9c2d2a52`.
   - The parts were extracted with tarfile's "data" filter into a new scratch directory. Each part's size and sha256
     equalled the manifest's, and each is under 50,000,000 bytes. Every member is a plain ustar regular file with the
     rules above, in path order. Each part's members are exactly the manifest's files for that part: 13, 20, 19, 65
     and 2.
   - The extraction gave 119 files and no other, 161,645,299 bytes. Every file's size and sha256 equalled the
     manifest's, and every sha256 equalled the one `restore_inputs.py` lists in group `sets` (0 mismatches).
3. **A fresh clone, before the commit.** `https://github.com/kuds/mesozoic-labs` was cloned into scratch and checked
   out at `23bfbfcc`. Two things differed from the commands above:
   - this directory was copied from the branch's worktree, because it was not yet committed;
   - `.venv` was a symlink to the main checkout's venv, instead of `setup_venv.sh`.

   The data branch was fetched from origin at `ecf95a81`. Then:
   - `sha256sum` printed the five sha256 of the parts table. s2-b1's two V6 parts printed the sha256 of its
     MANIFEST.json (`11d7bd3e...` and `6739c8b7...`).
   - Extracting the seven parts gave 136 files.
   - Before `copy-from`, `check --group sets` printed `sets 136 files, 0 matching, 136 missing, 0 wrong, agree`.
   - `copy-from <dir> --group sets` printed "copied 136, already here 0", then
     `sets 136 files, 136 matching, 0 missing, 0 wrong, agree` and "all here with their listed sha256". It exited 0.
   - A full `check` then printed `sets 136 files, 136 matching, 0 missing, 0 wrong, agree`. The other six groups were
     all missing, all agreeing, as in any fresh clone. It exited 1.
   - `git status` showed only the `.venv` symlink as untracked.
   - `restore` was not run. The scratch clone was removed.
