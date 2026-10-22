# stance5 criteria: where each maintainer decision lives

Notes on `SP/foot/stance5/criteria.txt` (SP = the session scratchpad). Line numbers refer to the text as re-fixed after
the review of 2026-10-21 (the section "After the review" below says what changed and why).

| decision (2026-10-20, D-D40) | what it says | criteria lines |
|---|---|---|
| 1a | From-scratch screen of the per-foot CoP band: three 12M runs, seeds 52, 60 and 42, physics r9, n_envs 4, Colab L4. Criteria fixed before any training step. K1-K4 unchanged. | 3 (fixed before the first step, not edited after); 5-13 (r9, n_envs 4, L4, from scratch, 12M, cadences); 49-52 (the three runs, names and RUN_LABELs); 66-83 (K1-K4 word for word as stance4 and stance3) |
| 2A | Arm = L+T + stagger (0.30 / 0.045) + band. The three stance4 runs, as judged and scored, are the control (one difference). They are not retrained. | 15-19 (the recipe: c1e146a8 + five band commits + config commit); 85-105 (the control, pinned by score_final.json sha256 7c98d8ae..., passes K1 0/3, K2 0/3, K3 2/3, K4 2/3; why it is valid; caveats); 34 (2A's reason, the twist route) |
| 3a | Ramp guard: full price at load share <= 0.25, none at >= 0.35, linear between. | 17 (commit a15494ed); 27 (g_k); 30 (full-price cases); 33 (the lighten door is BOUNDED, not closed, with the threshold formula and both load-share rails); 175 and 179 (guard-firing and lighten-door readouts) |
| 3b | Speed pass (one fused per-step reduction, byte-identical to the recorder). No speed target decided. | 17 (commit 4819eb6c); 23 (16,000 foot-steps byte-identical); 104 (the overhead as a caveat); 185 (throughput readout, "no speed target was decided") |
| 3c | Neutral module: no exception to the gait layering rule. | 17 (commit 3a1d5de3, environments/shared/floor_contact) |
| 3d | Band weight 0.30. | 18 (stance_cop_band_weight 0.30); 29-30 (the term and its range); 141 (w 0.50 named only as line 6's alternative) |
| 4a | v6b ankle spring held in reserve; a TRIGGER LINE: the band fixes the split but runs fail K2 on drift -> the spring next. | 115-130 (DRIFT OR SPLIT, the objective class, D and S); 140 (line 5, the trigger line: D >= 2 and S == 0, with the spring's numbers); 142 (D and S are read only at H == 0 and K2p <= 1) |
| 5a | Saturation: no training change; eval-ledger alarm read at every check-in, reported at once with run, step and actuator; fails stay fails; no stop or restart. | 147 (no run stopped or changed on an alarm); 173 (the saturation watch: alarm rule, every check-in, run / step / actuator, the 20-episode per-actuator rollout of a checkpoint at or after the first flagged evaluation, the "not yet named" fallback, references); 137 (line 4 names the tail habit, with satdiag's evidence) |
| 6 | Mirrored-leg spawns and prey-straight-ahead stay diagnostic panels only, run at every scored checkpoint, report only. | 172 (standard, mirror_legs, prey_ahead, mirror_prey_ahead, each with its statue, at every J checkpoint, report only) |
| 7a (option i; D-D36's dated amendment) | Walking pilot 2's parent is stance3 Run 2's final; pilot 2 is rebuilt separately after this package. | 144 (line 1's stance parent rule, as stance4's); 145 (the parent comparison is report only and does not choose pilot 2's parent) |
| 7c (D-D39's dated amendment) | The L4 is confirmed and is this screen's runtime. | 7, 13 (the non-L4 fallback marked as stance4's rule, carried over), 98 |

## Choices the review should look at, and the review's outcome

1. **The drift-or-split rule (lines 115-130).** As first written it classed every K2a- or K2b-failing episode in all
   six J files by the sign of (r - m0)(l - m0). The review of 2026-10-21 upheld two defects, and the rule was re-fixed
   before any training step:
   - *Counted where K2 failed.* The first text counted episodes at checkpoints where K2a and K2b passed (3 or fewer of
     40), so a run failing K2 through K2c alone could be SPLIT on one stray episode, and its parenthetical ("this
     includes a run that fails K2 through K2c alone") was false in general. Now only the J checkpoints where K2a or
     K2b fails are counted (run_criteria's own "over" lists), and a run whose K2a and K2b pass at all six J is DRIFT
     outright (K2c, the drift test, failed alone).
   - *A near-reference margin.* The sign test put a one-foot excursion (one foot far out, the other at the reference)
     on the COMMON/SPLIT boundary, classed by the sign of the foot that stayed home: the control's only two COMMON
     episodes (stance4 s42 final seed 3080, nearer foot 2.5 mm; s60 8.4M seed 3052, 5.0 mm) and a SPLIT one (s60 8.4M
     seed 3081, 0.3 mm) were of this kind. Now an episode with a foot within 15 mm (15/70 span half-lengths) of m0 is
     NEAR-REFERENCE and not classed. 15 mm lies above the statue's own scatter (every statue foot within 13.5 mm of
     m0, the nearer foot within 12.7 mm) and inside the empty gap of the known K2-failing episodes (11.9 to 20.3 mm;
     10 mm would sit 0.4 mm from stance3 LT s52's 8.4M seed 3071). Near-reference episodes are left unclassed rather
     than called SPLIT, so the margin treats both classes alike and a one-foot excursion cannot outvote K2c.
   - Measured (`criteria_calib/drift_split_calib.txt`, and the scorer's --validate-on-stance4 and tests): every known
     run keeps its class. stance4 s52 0/101/0, s60 0/67/2, s42 0/27/1 (n_c/n_s/n_u): SPLIT; stance3 Run 2 0/30/0 and
     Run 3 0/6/4: SPLIT; arm L s60: DRIFT through K2c alone.
   - Kept from the first text: per episode, the sign test against the statue's m0 rather than "both feet outside the
     K2b band" (which leaves K2a-only episodes unclassed and calls a common shift that has pushed only one foot over
     the edge a split); per run, K2c decides a tie.
   - Not adopted from the review (a minority alternative): calling one-foot excursions SPLIT, and a minimum count
     before counts alone can give DRIFT. No known run is affected either way.
2. **The trigger line (line 140).** The first text read line 5 at D >= 2 and stated "the band fixes the split" as fact,
   untested: two runs at n_c 30 / n_s 29 and a third at 0 / 100 read line 5. Line 5 now also requires S == 0 (no
   K2-failing band run classed SPLIT), which is D-D40 4a's first clause as the criteria can test it, and its text says
   what is measured and what is not (how far the split is fixed is not tested beyond the class; the scorer prints n_c
   and n_s beside the control's). Line 6 takes D <= 1 or S >= 1, and with D >= 2 names the spring among the candidates
   the maintainer may choose. A limit that stays (inferred): three runs at, say, n_c 31 / n_s 30 each still read line 5.
   - *The H == 0 gating* (line 142): the trigger line is read only at H == 0 and K2p <= 1, as stance4's line 5 was.
     At H == 1 line 3 (replicate) decides and the spring stays in reserve while the band is not ruled out; D, S and
     every class are still printed, report only. The review asked that this be listed here for the maintainer; it was
     kept, not changed.
3. **Line 2 is unreachable** with the control's stated passes, because (a) implies (b). It is kept so the table is
   exhaustive under (b)'s general form, and it is marked PROVISIONAL if it is ever reached.
4. **Line 4's targets** (tail saturation habit, a too-strong dose, the leg noise floor) are named as candidates for
   the maintainer. None of them is decided. Line 6 names the stronger band (w 0.50, design.md section 3.4) and "another
   candidate the maintainer names"; the first text's "observing foot pressure" had no source in the decisions and was
   removed.
5. **The band-on task_sha256 and the collapse reference were measured on the uncommitted config change.** Line 47
   requires the config commit's tree to reproduce them before the first step; they were measured again on the
   amended config commit (`build/task_shas_f_amended_commit.json`).
6. **Other changes after the review** (no verdict changes): the decision labels (7c is D-D39's dated amendment, 7a(i)
   D-D36's); the non-L4 fallback marked as stance4's rule; the twist route's wording (the band alone does not close
   it); both load-share rails on the judge side of the lighten door (lines 33 and 179); the saturation watch names the
   actuator only from a checkpoint at or after the first flagged evaluation, with an "actuator not yet named" report
   until one is placed (line 173); the twist-route watch's control reference, the 18 control J checkpoints rolled
   before training (`panels/control/control_ankle_asymmetry.json`, line 178); and a placed but unjudged J checkpoint
   makes a --runs-ended decision PROVISIONAL (line 158).

## What the next steps must build to match the text

- judge_stance5.py: arm LTSB (clone SP/fix/scratch_copband, HEAD = the config commit, collapse 4457.4) and the
  schema stance5-judge/v1.
- score_stance5.py: the stance3 code unchanged; the control from stance4's score_final.json (sha256 7c98d8ae...);
  the cross-check through score_stance4.py's stagger instance; the DRIFT OR SPLIT class with D and S printed for every
  band run; the specific and general forms of (b); the PROVISIONAL reasons of lines 153-159; the six-line table,
  printed verbatim.
  `criteria_calib/drift_split_calib.py` is a calibration prototype of the class (report only), not the scorer.
- The panels tool for arm LTSB, which must also record the band info keys (lines 174-180) and the ankle-asymmetry
  columns (line 178), and the control's ankle reference (`panels/control/`).
- A port of satdiag's roll.py to arm LTSB for the actuator naming (line 173), and a watch that never names it from a
  checkpoint before the first flagged evaluation.

## Files

- `SP/foot/stance5/criteria.txt`: the criteria (read-only; sha256 in `criteria.sha256`). The text as first fixed
  (sha256 c2809201...) is kept as `build/fix2/criteria_c2809201.txt`, and `build/fix2/edit_criteria.py` holds every
  edit the review made.
- `SP/foot/stance5/criteria_calib/drift_split_calib.py` and `drift_split_calib.txt`: the measured class of the
  stance4 and stance3 runs under the re-fixed rule, beside the rule as first written (report only; the first
  version's script and output are in `build/fix2/drift_split_calib_before_review.*`).
