# T. rex stance pop, displaced pose and saturation: evidence (2026-10-30)

Evidence for
[TREX_STANCE_POP_POSE_SATURATION_2026_10.md](../../TREX_STANCE_POP_POSE_SATURATION_2026_10.md),
the root-cause diagnosis of #489 and #491 on T. rex physics r10 (policy
interface r15) at `main` `358651f`, written at the maintainer's request on
2026-10-30.

It is report only:
- Every rollout here is open loop: scripted commands on the real
  `TyrannosaurusRexAnatomicalEnv`, with no network.
- Nothing trained or loaded a policy.
- Nothing here enters a verdict, a gate or a decision.

The scripts are research tools. Nothing imports them, and they sit outside
the library, its tests and CI. [SOURCES.txt](../SOURCES.txt) gives, for each
file, its source in the session folder, that source's sha256 and what the
copy changed. [SHA256SUMS](../SHA256SUMS) pins every file here except this
README.

## Files

| File | Note's tracks | What it is |
|---|---|---|
| `common.py` | — | Env factory (`build_env(ModelId("tyrannosaurus_rex", "anatomical"), 1)`), joint and actuator tables, `custom_reset`, and `rollout` with per-term reward capture |
| `t1_pop.py`, `t1/t1_results.json` | 1, 6 | The reset pop: settling timeline, kinetic energy and foot force, reset penetration, joint settling against `qpos0`, and per-window rewards (noise-free and 20 native-noise seeds) |
| `t2_pins.py`, `t2/pins.json.gz`, `t2/patterns.json.gz` | 2, 4, 5 | Each actuator held at ±1 open loop, then the #491 and satdiag patterns and random static offsets, with their release |
| `an_pins.py` | — | Summariser of the two `t2/` files (prints to stdout) |
| `t2b.py`, `t2b.json.gz` | 2, 4, 5 | Leg-offset tolerance, neck and tail counterweights for a leg lean, and early lean unwinds |
| `t4b.py` | 4 | The counterbalanced lean + neck pair: releasing both against releasing one half, abruptly against slowly (its output is not kept) |
| `t4c_grid.py`, `t4c_grid.json` | 4 | Static return surface over lean scale × neck command |
| `t4d_ref.py`, `t4d_ref.txt` | 3, 4 | The statue against small leans under `leg_home_pose_reference` = keyframe and = settled (`t4d_ref.txt` is its printout) |
| `t5_price.py`, `t5_price.json` | 3, 4, 5 | Per-actuator price of rails, flips and sinusoids under the committed weights (analytic) |
| `t6_vecnorm.py`, `t6_vecnorm.json` | 6 | Observation statistics behind VecNormalize: pop against quiet variance shares, std ratios, clip fraction and pop separability |
| `t7_noise.py`, `t7_noise.json` | 4, 5 | The statue under per-step Gaussian action noise by channel group and σ, with a 100-seed statue baseline |
| `t8_headtail.py`, `t8_headtail.json`, `t8_headtail.txt` | 2 | Head–tail balance: gravity moments about the hip axis by body group, CoP and contact span, and how far the neck, head and tail joints move the centre of mass (`t8_headtail.txt` is its printout) |

Seeds and arguments are in the note's Appendix.

## Not kept

Each of these files is over the repository's 500 KB limit for an added file,
even gzipped. The script named rewrites it.

| File | Bytes | sha256 | Written by |
|---|---|---|---|
| `t1/t1_traces.npz` | 3,404,738 | `4a280787361b3b47e1d724f73f412bff5786ba2189365c43d19b3eb25105c41e` | `t1_pop.py` |
| `t4b.json` | 23,057,516 | `072863e5c6b753b77838651440d4fb99cbca7b285fd98aa1be0cc1cffde91ee9` | `t4b.py` |

## How to rerun

1. Create a Python 3.12 venv with the repository's core dependencies only:
   `python3 -m venv /tmp/mesovenv && /tmp/mesovenv/bin/pip install mujoco==3.10.0 gymnasium numpy`.
   It needs no torch or SB3. The original runs used gymnasium 1.4.0 and
   numpy 2.5.4.
2. Make an empty working folder `<work>`. Copy this folder to
   `<work>/scratchpad` and link the checkout as `<work>/checkout`. The
   scripts' `D` is `scratchpad` and `common.py`'s `REPO` is `checkout`, both
   relative to the working directory.
3. From `<work>`, run
   `PYTHONDONTWRITEBYTECODE=1 /tmp/mesovenv/bin/python scratchpad/<script> [arg]`.
   The outputs are written into `<work>/scratchpad/`. The scripts write their
   JSON without a final newline; the copies here add one.

This was done on 2026-10-30 at `358651f`, and every check matched byte for
byte (the JSON after adding the final newline):
- `t4d_ref.py` and `t8_headtail.py` reproduced their printouts.
- `t5_price.py` and `t8_headtail.py` reproduced their JSON.

## The note and its draft

The note is the session's report draft (`session/files/REPORT.md`, 1,043
lines, sha256 `131a71280c233d9dad7d28e827d7dcede8cd8ae423196e46a7ad15eba06ad4b3`)
with these edits only:

1. Under the title, a **Date** and **Status** block that names the brief,
   what comes with the note and its related records.
2. In **Scope**, "I made no repository changes" became "The investigation
   made no repository changes: this note and its evidence folder are its only
   additions".
3. In **Method**, "Scripts and raw outputs are in this folder" now points to
   this folder. The [M] label "measured by me in this session" became
   "measured for this note on r10", with this folder's path.
4. In the Appendix:
   - The two bullets on the hard-coded paths and how to run were replaced by
     this folder's path substitutions and the rerun steps above.
   - An **Outputs kept** paragraph follows the scripts table.

Nothing else differs.
