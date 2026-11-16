# fix:G (roundA): the agent's final report

WP G review fixes: all three findings were right and are fixed, with the record remade. Final commit: `d31999d Stage 2 round A (WP G): the regression record at the exact-stride control.py (full tier, 248 distinct episodes)`.

**Findings**
1. **A kept case whose eval failed never ran again (regress.py `_kept`)**: confirmed and fixed. A kept result whose `returncode` is not 0 (or is missing) now runs again; `RESUME_RULE` says so. Tested in `test_run_case_keeps_its_result_and_the_evidence` and `test_run_writes_the_record_and_keeps_finished_cases` (codes 3, -9, None, absent).
2. **Stride of a K-stride window (ruling G-R1)**: confirmed and fixed. `(K*n*step)/K != n*step` on 4 of the 20 references for some K in 3..7. `add_periodic` now uses `stride = (ref.n // K) * ref.step` and raises `ValueError` when `K < 1` or `ref.n % K`.
   - Default path: old-against-new `add_periodic` on all 20 references of both bodies, against Stage 1's control.py (70cd798) and the first edit (e6ca00a), is bit-identical in every segment field and info array (20 of 20 each). Four short R0/R1 episodes are identical.
   - `test_window_strides_key` gains K = 3 on 75-M-H-10 brisk (a reference where the old expression is one ulp off), the refused keys, and key = 1.
3. **256 counts 8 episodes twice**: confirmed and fixed. The record carries `distinct` (248) beside `of` (256), with a `COUNT_RULE`. The closing line prints both, `check` requires it, and the tests assert 248.

Each new assertion was seen to fail with its fix removed (in-memory source change; three runs, pytest exit 1 each).

**Acceptance (at f4c5eee for the runs, d31999d for the rest)**
- `regress run --tier fast --scratch logs/regress-fix-fast`: exit 0. Last line: "fast tier: 8 of 8 episode runs match their committed trace and episode record (8 episode runs, 8 distinct; 27 s of evals)".
- `regress run --tier full --scratch logs/regress-fix-full --record stage2/regress/regress.json`: exit 0, 8 min 54 s wall. Last line: "full tier: 256 of 256 episode runs match their committed trace and episode record (256 episode runs, 248 distinct; 531 s of evals)". Every trace hash equals the previous record's.
- `regress check`: exit 0.
- `STAGE2_SLOW=1 pytest -p no:cacheprovider tests/test_stage2_regress.py tests/test_stage1.py tests/test_stage1_panels.py tests/test_stage1_speedfit.py`: exit 0, "169 passed, 15 skipped in 205.16s" (the skips are STAGE1_SLOW).
- `git status --porcelain`: empty.

**Public names changed**
- New: `regress.distinct(entries)`, `regress.COUNT_RULE`, record keys `distinct` and `rules.count`.
- The run's closing line now reads "episode runs ... N distinct".
- `Track.add_periodic` raises `ValueError` for a `window_strides` under 1 or not dividing the frames.

**For the lead**
- Remake the record at code state A with the same `run --tier full --record` command; `test_regress_record_is_current` fails after the Jetson merge until then, by design.
- No pre-registered text quotes 256: STAGE2_75_PLAN.md R4 and STAGE2_PLAN.md S2-17 name the record without a count.
- 256 does appear in working papers: `stage2/design/checkpoint/reviews/roundA__build_G.md` (lines 3, 14, 15, 22), `checkpoint/generated.json:22`, and the subjects of commits afc0a3d and e6ca00a (not rewritten).
- ROUND_A_DESIGN.md:597 (2.4, D18) still gives `ref.period / K`; I recommend updating it to `(ref.n // K) * ref.step` plus the refusal.
- Not changed: `check` does not compare the record's library versions with today's (the reviewer's note). I recommend adding that comparison if you want it guarded.

Logs are in `/home/user/mesozoic-wt/wp-g/hardware/trex_form_study/logs/` (`wpg-fix-ab.out`, `regress-fix-fast.out`, `regress-fix-full.out`, `wpg-fix-acceptance.out`, `wpg-fix-mut1..3.out`); scratch scripts are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/wpg_fix/`.
