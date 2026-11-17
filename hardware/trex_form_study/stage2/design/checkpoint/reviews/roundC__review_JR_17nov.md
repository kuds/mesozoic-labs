JOIN-CODE REVIEW (L12; ruling 19). Reviewer: independent, read-only. Prescribed-motion scratch probes only. No file was changed anywhere, nothing was committed, no record was written. The verdict on cut 3 is the lead's.

VERDICT PER GATE

1. join_command, CoM part (model CoM within 1.0 mm over the warm-up and at the command frame): RIGHT. V6 turns read 0.033-0.066 mm; at the command 0.004-0.051 mm.

2. join_command, joints at the command frame (within 1e-3 rad): RIGHT. Reads 0.7e-4 to 2.3e-4 rad.

3. join_command, warm-up multibody ZMP (4.0 mm, frames up to two before the command, against the resampled window): frames, limit and comparison are RIGHT. The warm-up it reads is NOT CAUSAL, so the gate is WRONG as a reading of A6/A7.
 - Frames: `dz[:-2]` gates warm-up frames 2..k_cmd-2 (157 frames; frames 0-1 have no qdd). Frame k-2 is the last one whose 5-point stencil ends at the command frame. That is what A6 says.
 - Comparison: a null join takes the window's own q, resamples it on the move's warm-up frames, puts it through the move's own difference formula and compares it with the resampled zmp_mb. It reads 0.115 mm max on the left turns and 0.078 mm on the right turns and stop. The comparison is sound.
 - Builder's observation: true that frame k-2 reads q at the command frame. But it carries no post-command information. The plan's warm-up, solved on the window's own LIPM path, reads 0.130 mm identically at f 1.0, 0.8 and 0.6. On a causal grid the warm-up is identical across f, and for the right side identical to the stop's.
 - Dropping k-2 (`dz[:-3]`) is not the fix. The f dependence stays: left 5.543 / 3.317 / 3.180, right 5.519 / 3.661 / 3.111 at f 1.0 / 0.8 / 0.6.
 - The real channel is the preview grid. `moves.com_path_causal` defaults the grid to `min(diff(pl.t))` over the whole plan (moves.py:835). `_generate_causal` (moves.py:1090) and `_kinematic` pass no `dt`. That sets the grid, the gains, the preview length and k_switch of the controller run that produces the warm-up.
 - On these turns the minimum lies after the command. At f 0.8 / 0.6 it is at phase 0.66, in the turn's quicker steps (4.295 / 3.221 ms). At f 1.0 it is at phase 6.76, in window b's F14 fade-in near the move's end (5.140 ms). The warm-up's own minimum is 5.224 ms.
 - So the warm-up, and the gate's reading, depend on what happens after the command, which A6 rules out ("the preview controller first sees the command at that frame"). This channel is the whole of the 5.54 / 4.40 / 3.54 (left) and 5.70 / 3.66 / 3.69 (right) spread.

4. at-command ZMP (recorded, not gated): RIGHT as a record. It straddles the command: 0.44 to 7.67 mm.

5. join_b (J2: last output stride against the window's last stride resampled, offset K - last, carried by the plan's pose at last-1; CoM 1.0 mm, ZMP 2.0 mm): RIGHT on reading the code. The V6 turns read CoM 0.20-0.22 mm and ZMP 1.16-1.74 mm. Not probed numerically.

6. join_a (moves of Stage 1's construction: first output stride against the window's first stride, View offset -first, boundary leave per S2L-J2): RIGHT on reading the code. Not probed.

7. Causal ZMP windows (consistency error over phase < c+1, ZMP error from phase >= c+1, both 4.0 mm): RIGHT on reading `causal_errors`, matching A7's changed paragraph. Not probed.

8. Warm-up frames (`slice(0, k_cmd+1)`, 161 frames): RIGHT.
 - My single-pass warm-up equals the recorded final-pass reading to every printed digit on all 9 causal moves. The warm-up therefore depends only on the plan and the grid, not on the correction passes.

9. join_events (moves_look; a reading, not a gate): the numbers are right (gate frames and limit, after 108ae27a). Its stated cause ("different samplings ... differ by millimetres whatever the join") is WRONG. The sampling floor is 0.08-0.13 mm. The mm-level excess near events is the controller's re-run near ZMP-reference transitions: the maxima sit at the right lift-off (phase -0.51) and at k-2.

SMALLEST CORRECT READING (gate 3) AND WHAT IT READS ON THE V6 TURNS
The warm-up is solved on a causal grid: the smallest frame interval up to the command frame, `min(diff(pl.t[:k_cmd+1]))`, which is 5.2244 ms on every V6 move. Everything else stays as coded.

| Move | f | Code (recorded) | Causal grid | Result |
|---|---|---|---|---|
| Left turns | 1.0 | 5.543 | 4.621 | fails |
| Left turns | 0.8 | 4.397 | 4.621 | fails |
| Left turns | 0.6 | 3.536 | 4.621 | fails |
| Right turns | 1.0 | 5.695 | 3.868 | passes |
| Right turns | 0.8 | 3.661 | 3.868 | passes |
| Right turns | 0.6 | 3.691 | 3.868 | passes |
| stop2_v0.4 | - | 3.868 | 3.868 | passes (grid was already causal) |

For the lead:
- These causal-grid numbers are for the warm-up only. Making the grid causal changes the controller run, so the post-command output and its gates must be rebuilt and gated again. I did not do that.
- The change sits in moves.py, which is shared with the round's frozen sets. Changing the default would move their bits; D18's test keeps moves.py in its list (ruling 27).
- The round's records show the same signature, all within 4.0 mm:
  - V0: 1.349 at f 1.0 = stop; 1.289 at 0.8; 2.114 at 0.65
  - F25F9: 2.083 at f 1.0 = stop; 3.485 at 0.7
  - F25F9cl: 1.249 at f 1.0 = stop; 2.321 at 0.95; 1.307 at 0.65
- A look-only fix still needs a hook in moves.py, because `_generate_causal` calls `com_path_causal` directly; `_walk_reader` does not cover it.
- Alternatives:
  - (a) Keep the gate as coded. It reads post-command information.
  - (b) Causal grid, as above.
  - (c) The builder's `dz[:-3]`. Not causal, not recommended.
  - (d) Gate the plan only: the warm-up solved on the window's own path, J1's "warm-up stride ... is the window resampled" taken literally. It reads 0.130 mm at every f, but it no longer reads the controller history that feeds the move's state at the command. It is a different gate, and these numbers already exist.
- Bearing on L12's trigger: even on a causal grid the excess is the controller re-running the walk. The CoM differs from the window's by at most 0.066 mm, and the second differences turn that into 3.9-4.6 mm. A7's P34 paragraph already says this for the round's sets. It is not caused by J1's leave point inside the window.
  - At cut 3's leave point (the window boundary, a_phase 12.0 / 12.5) the same gate reads 4.038 mm left (all f) and 4.361 mm right on a causal grid.
  - On the code's grid it reads 4.956 (f 1.0 left), 5.448 (f 1.0 right), 3.659 (f 0.8 left) and 3.204 (f 0.6 left).
  - The floor there is 0.33 / 0.60 mm.
  - These are scratch readings, not a decision.

EVIDENCE (commands run from the worktree study directory with the prescribed environment, via tools/lowprio.sh; output quoted)
- Record-key hashes printed `45719d03...528d c0027992...c1c1`, the registered pair, at the start and at the end.
- probe_warmup.py, 9 moves. For every move A = recorded exactly, e.g. "turn90_0.35_left ... rec 5.543442246091401, A 5.543442246091401, argmax -82, B 5.5434, C 4.6214, D 0.1152".
- probe_noise.py: tightening IK (2000 iterations, tol 1e-10) and the CoM loop (60 iterations, COM_TOL 1e-10) gives "tight 5.5556" against "code 5.5434". It is not solver-convergence noise.
- probe_split.py (plan solved on the window's com_ref): "window_com_path max_mm 0.12989" at f 1.0, 0.8 and 0.6.
- probe_grid.py: the minimum interval's location in each plan (above).
- probe_cut3.py: boundary readings (above).
- Scripts and outputs are in /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wave7/jr/ (probe_warmup.py 789407ed..., probe_noise.py f59b69bf..., probe_split.py a2437a7d..., probe_cut3.py b3673296..., probe_grid.py 7d6dd4b1...; probe1/2.json, split1.json, noise1.json, cut3.json, per-frame .npz). They are outside any repository, so git check-ignore does not apply. `git status --short` was empty in the worktree and in the main checkout at the end.

FILES REVIEWED (sha256)
- designlab/stage2/moves.py: ccee9b2cdf5acfef7b1f1bf592d617b71f035617016b6ecffd3e727aa5b0e77e. Same on wt/wp-l 628d60b7 and 8555f72f, in the main checkout, and on the working branch.
- designlab/stage2/moves_look.py (wt/wp-l): a303295b0b2d28dddb92a037cc71f06df3d53af1c05cc1217a0462cd6c5c22e8. Unchanged by 036ec9d6, 9889291c and 8555f72f, which landed during the review.
- Read for the gate's formulas: designlab/stage1/transitions.py 72dfe5ee93ec41dbb4147cd006f0fd0b9b25dc5e69ecc47209f5ba572fbf7332.
- Inputs: stage2/sets/V6/75-M-H-10/refs/v0.4.npz c4f44b48d0fb9daa4b42957addbda3e82b96e43678482d1e0dec8ac597912a62 (matches refs.json); transitions.json 5ff79b23...01f7; refs.json 0870a576...a811.
- The turns' arrays are not kept (their moves are "not available"), so each warm-up was rebuilt from the window with the code as committed.

NOT CHECKED
- No pytest run.
- join_b and join_a not probed numerically (code read only).
- Post-command gates under a causal grid not rebuilt.
- V6noF5, the 60 cm bodies and the -X heads not probed; only their records were read.
- plan(), the stride rule and the acceleration readings not reviewed.
- No L12 review record written into set.json; that is the lead's.
