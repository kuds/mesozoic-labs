# Walking pilot 2 rebuild (decision 7a i), 2026-10-21

Written by the coordinator from workflow [id removed]'s findings (subagents could not write this file).

- Branch claude/pilot-walk-r9-b in fix/pilot_walk_r9_b2: ba464855 + 6a13a303 (new parent pilot_checkpoints/stance3_armLT_s60_final/
  with SOURCE.txt) + 8fc76eec (closing note in pilot 1's parent's SOURCE.txt) + 8c5478a6 (two SOURCE.txt sentences corrected).
  Head 8c5478a6ca96309be820959efe00465893eb6010, clean. Diff vs ba464855: only pilot_checkpoints/ (4 files). Committed parent:
  stage1_final.zip cc21604a..., stage1_final_vecnorm.pkl 894e86f4... (byte-identical to foot/stance3 placed copies).
- Parent: stance3_armLT_s60 final, 12,001,280 steps, plant r9/r15 identical to the branch's, obs 63 / act 15, same policy
  architecture as pilot 1's parent. Std mean 0.064 / min 0.009 (knees) / max 0.125 (pilot 1's parent 0.095 / 0.018 / 0.164).
  Stance3 final: v2 PASS 40/40, 0/20 joint-space and 0/20 own-sigma noise falls; Run 2 failed K1/K2 ("does not hold"); not certified.
- Criteria amended 2026-10-21 (pre-amendment copy criteria_2026-10-18.txt, 0444; pre-review copy criteria_before_review_fixes_1021.txt).
  No outcome definition, threshold, guard, label, seed, budget or tool changed. The confound is stated: pilot 2 now differs from
  pilot 1 in parent and reset; no outcome is attributable to either alone. The reset is a larger relative jump (x3.46 mean, x25 knees).
- Pre-launch checks on 8c5478a6: notebook smoke rc 0 (Warm start line with 0.064/0.009/0.125; stage_config git_commit 8c5478a6,
  parent cc21604a, warm_start_log_std -1.5 twice; hyperparameters ec9925f9 unchanged); pilot_judge accepted (40/40 stand), refused with
  pilot 1's parent sha and refused the 8fc76eec run; warm-start tests 29 passed; digest current; judging-tool sha256s match.
  On 8fc76eec (carry over, text-only change since): full suites 5,937 passed / 1 skipped / 1 deselected (needs clone-root cwd and
  tensorboard on PYTHONPATH on this machine), ruff, mypy, species_catalog, plant_contract, notebook_cells.
- Colab steps updated (pre-review copy colab_steps_before_review_fixes_1021.txt; 10-18 copy colab_steps_2026-10-18.txt).
- Open: the push (fast-forward ba464855 -> 8c5478a6) and the PENDING push line in colab_steps.txt. Hold: no further push to the
  branch after 8c5478a6 until all three seeds are judged. Inference only: an SB3 GPU UserWarning may print between "====" and the
  "Warm start" line on L4 (step 8 says to ignore it).
