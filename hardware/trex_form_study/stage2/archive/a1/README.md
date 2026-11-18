# Archive a1: the 52 frozen T. rex set arrays that no other branch holds

This branch, `claude/mesozoic-trex-stage2-archive-a1`, is an orphan data branch that is never merged. It holds only
this directory: 52 git-ignored arrays of the T. rex study's frozen Stage 2 reference sets, packed as three tar parts.
It is not code and not records. The study's code and records are on the working branch
`claude/mesozoic-trex-stage2-refine-4jffre`.

## Why

- The owner paused T. rex development on 18 Nov 2026 and moved the project to a Triceratops build (P47 in
  `hardware/trex_form_study/DECISION_LOG.md`). The T. rex stays on the roadmap (P51).
- Git ignores the study's arrays (`*.npz`), so a fresh clone has none. `tools/restore_inputs.py` lists each one with
  its sha256 and rebuilds them from the committed records.
- The owner chose to keep the frozen T. rex arrays that bundle s2-b1 does not hold (about 120 MB) on a never-merged
  data branch, so that a restart needs no rebuild of them (P49).
- Until this archive, these 52 were only in worktrees of the cloud machine, which is deleted after a while.
- An earlier build of this branch (`be474190`) held instead the 380 files that `restore` rebuilds, and not these 52.
  It was not pushed.

## What it holds

52 files, 119,553,986 bytes. They are the arrays that `restore_inputs.py` lists in group `sets` for the three newest
frozen sets (frozen on 17 Nov) and that bundle s2-b1 does not hold. The lists are those of commit
`61ae6fa0e611cb44ca4f5cef88557911ee4a9ad6` of the working branch.

| set | body | listed in `sets` | in this archive | in bundle s2-b1 | bytes in this archive |
|---|---|---:|---|---|---:|
| V6 | 75-M-H-10 | 23 | 6 (`look`) | 17 (`refs` 6, `transitions` 11) | 17,227,350 |
| V6 | 75-M-H-10-X | 23 | 23 (`look` 6, `refs` 6, `transitions` 11) | 0 | 52,293,386 |
| V6noF5 | 75-M-H-10 | 23 | 23 (`look` 6, `refs` 6, `transitions` 11) | 0 | 50,033,250 |
| **all** | | **69** | **52** | **17** | **119,553,986** |

- V6 is the lifelike walk the owner chose. 75-M-H-10-X is the 75 cm body with the expressive head.
- V6noF5 is V6 without F5's toe-off (`hardware/trex_form_study/stage2/design/WP_L.md`).
- The files were read from the worktree of `wt/wp-p42` (`f9dc89cc`) on the cloud machine. That commit is an ancestor
  of `61ae6fa0`, and the set records and `tools/restore_inputs.py` are the same at both.
- There `restore_inputs.py check --group sets` printed 136 files, 136 matching, records agree, and exited 0.
- Each file's sha256 was compared with the tool's list before packing. The largest file is 4,008,709 bytes.

`parts/frozen-NN.tar` are uncompressed ustar archives, each under 50 MB. Their member paths are relative to
`hardware/trex_form_study`, in path order. Each member has mtime 0, uid and gid 0, empty user and group names and mode
0644 (the tar rules of `designlab/stage2/bundle.py`). A new part starts when the next file would take the part past
41,000,000 bytes.

| part | bytes | files | sets | sha256 |
|---|---:|---:|---|---|
| `frozen-01.tar` | 39,956,480 | 13 | V6 on 75-M-H-10-X | `9baed6c20965c1ceaaa4c3d399d3667116085a6291d4817a714c4147f8c8f476` |
| `frozen-02.tar` | 40,499,200 | 20 | V6 on 75-M-H-10-X, V6 on 75-M-H-10, V6noF5 on 75-M-H-10 | `3ea1ab3be35aaa0642d751961c057835e1325cae912b2e4e0f7dbdd205830059` |
| `frozen-03.tar` | 39,157,760 | 19 | V6noF5 on 75-M-H-10 | `d2d3ca06c57c8d62bb6f1e39493e08932dd7c9132d374497b321b06cebcdb78f` |

`MANIFEST.json` holds:

- every file, with its study path, set, body, kind, size, sha256 and part;
- the parts, with their sizes and sha256;
- the source commit `61ae6fa0` and the worktree the files were read from, with `check` as it printed there and in the
  main checkout;
- the 17 V6 arrays in bundle s2-b1, with their sha256 and parts;
- the 67 other `sets` arrays and the other groups, which `restore` rebuilds, and the restore order.

## Not in it, and where it is

The tool's seven groups list 449 files. This archive holds 52 of them. The rest:

- **17 in bundle s2-b1.** V6 on 75-M-H-10, its `refs` (6) and `transitions` (11). They are on the pushed data branch
  `claude/mesozoic-trex-stage2-bundle-s2-b1` (`ecf95a81`), in `hardware/trex_form_study/stage2/tb/bundle/s2-b1/parts/`
  `V6-75-M-H-10-1.tar` and `V6-75-M-H-10-2.tar`. Their members are at study paths, each with the listed sha256.
- **380 files (250,768,725 bytes) rebuilt by `restore_inputs.py restore`** from the committed records:
  - the groups `picks-models` (6), `picks-arrays` (84), `k` (60), `x` (60), `j` (30) and `look` (73);
  - in `sets`, the 67 arrays of the three earlier frozen sets, V0, F25F9 and F25F9cl on 75-M-H-10.

  `check` at `61ae6fa0` reports every group's list in agreement with the committed records.
- **The time.** About 2.5 h: on 14 Nov, `restore` of five groups (`picks-models`, `picks-arrays`, `k`, `x`, `look`)
  took 151 min on 2 processes in a fresh worktree (`tools/README.md`). The rebuild of `j` and of the three earlier sets
  was not timed in a fresh worktree, and `restore --group sets` has not been tried.
- **At least 2 CPUs.** On one CPU the picks' ledger JSON cannot be reproduced (`SIZE_BLAS_THREADS` in
  `restore_inputs.py`). Then bring those files in with `copy-from`; bundle s2-b1's `models-75-M-H-10.tar` holds
  75-M-H-10's two.
- The eval traces are not inputs and are not kept (P49).

## How to restore, in a fresh clone

Run these from the clone's root. The lists that `restore_inputs.py` checks are those of the commit you check out:
`61ae6fa0`, or any later commit whose lists are the same.

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
PYTHONPATH=. .venv/bin/python tools/restore_inputs.py restore   # the other groups: about 2.5 h, see above
PYTHONPATH=. .venv/bin/python tools/restore_inputs.py check
```

- **The order matters.** `copy-from` comes first, so that `restore` finds the 69 V6 arrays in place. `restore`
  rebuilds only missing or wrong files, so it then leaves them alone.
- **Run `restore` detached** so that no time limit stops it, with a keep-alive (`tools/README.md`, "A fresh
  container, in order", steps 3 and 4). If it is stopped, run it again: files in place are kept.
- **`copy-from`** puts each listed file in place only when its sha256 is the listed one. It leaves a right file alone
  and writes no tracked file.

What to expect:

- `sha256sum` prints the three sha256 of the parts table above.
- `copy-from ... --group sets` copies the 69 V6 arrays. Its closing `check` of `sets` reports the 67 arrays of V0,
  F25F9 and F25F9cl as missing, and it exits 1, until `restore` has rebuilt them.
- After `restore`, `check` should print every group matching and agreeing, and exit 0. `restore` was not run with this
  archive.
- `check` must exit 0 before any run that reads the inputs.

## Checked on 18 Nov 2026

1. **The parts.** The committed parts were extracted with Python's tarfile ("data" filter) into a new scratch
   directory.
   - Each part's size and sha256 equalled the manifest's, and each is under 50,000,000 bytes.
   - Every member is a regular file with a relative path. Each part's members are exactly the manifest's files for
     that part: 13, 20 and 19.
   - The extraction gave 52 files and no other, 119,553,986 bytes. Every file's size and sha256 equalled the
     manifest's, and every sha256 equalled the one `restore_inputs.py` lists in group `sets` (0 mismatches).
2. **A fresh clone.** `https://github.com/kuds/mesozoic-labs` was cloned into scratch and checked out at `61ae6fa0`.
   Two things differed from the commands above:
   - this branch was fetched from the cloud machine's repository, because it was not yet pushed;
   - `.venv` was a symlink to the main checkout's venv, instead of `setup_venv.sh`.

   The data branch was fetched from origin at `ecf95a81`. Then:
   - `sha256sum` printed the three sha256 of the parts table. s2-b1's two V6 parts printed the sha256 of its
     MANIFEST.json (`11d7bd3e...` and `6739c8b7...`).
   - Extracting the five parts gave 69 files.
   - Before `copy-from`, `check --group sets` printed `sets 136 files, 0 matching, 136 missing, 0 wrong, agree`.
   - `copy-from <dir> --group sets` printed "copied 69, already here 0": the 23 arrays of each of V6 on
     75-M-H-10-X, V6 on 75-M-H-10 and V6noF5 on 75-M-H-10. It listed the 67 arrays of V0, F25F9 and F25F9cl as
     "not copied ... not made" and exited 1.
   - `check` then printed `sets 136 files, 69 matching, 67 missing, 0 wrong, agree`. The 67 missing are those of V0,
     F25F9 and F25F9cl, which `restore` rebuilds. The other six groups were all missing, all agreeing, as in any
     fresh clone. It exited 1.
   - `git status` showed only the `.venv` symlink as untracked.
   - `restore` was not run. The scratch clone was removed.
