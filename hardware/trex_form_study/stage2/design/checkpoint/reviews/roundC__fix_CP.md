# fix:CP (roundC): the agent's final report

WP CP, after review. Final commit 36d0721 on wt/wpc-cp (worktree clean; only s2load.py and tests/test_stage2_s2load.py changed since 8aa5ce3).

FINDINGS
1. (major, K9) Right. I took option (a), because it makes the code match the committed rule K9 and D14's "the .pkl is read only by a training resume"; it departs from the brief's "pair_problem first" for inference loads only. An inference load now checks the record, all three files' sizes and sha256, then the zip (`_archive_problem`, held to the repository's sentence over 8 archives by a test) and unpickles nothing. Training loads and `newest_intact` keep `pair_problem`. LOADER_RULE and docstrings updated. Test: raising .pkl, resealed: inference 0.0, import ok, training refused, `newest_intact` skips it. Lead to confirm.
2. (major) Right, fixed. Stored actions equal SB3's own `predict` on `VecNormalize.normalize_obs` and differ from raw-observation actions; changed mean, var, clip, epsilon and one output weight are each refused (re-save control 0.0).
3. Right, fixed. Any read failure of the .npz is a CheckpointError naming the checkpoint (cut 50 %, 98 %, empty, noise, .npy tested).
4. Right, fixed. The script entry assigns `disable`; a child started under OSMesa is tested. Optional loader guard not built: it would rest on mujoco internals and change library behaviour (lead's call).
5. Right. Both tables pinned by sha256 (reviewer's value confirmed; 88/130: 0a9aa56c…7527).
6. Right, all three pinned.
7. Right, one case per branch. This exposed a defect: `stats_of` on `norm_obs=False` raised AttributeError (SB3 creates no `obs_rms`); now CheckpointError.
8. Not built: a record-schema addition (design 3.6) is the lead's. Recommend accepting `actions_at_bound` (one line, one assertion).
9. Right; the rules are CB's file. The test is split: the second half is reported as skipped, naming the 8 uncovered paths, until `stage2/tb/.gitignore` exists, then asserts by itself. CB's planned lines cover all 8.

29 scratch mutations (the reviewer's survivors plus the new code) are all caught.

ACCEPTANCE (36d0721; log /home/user/mesozoic-wt/wpc-cp/hardware/trex_form_study/logs/wpc_cp_accept_fix_final.log)
- pytest, three files: exit 0, "61 passed, 1 skipped in 18.60s"
- scratch helper: exit 0, "MUJOCO_GL=disable, mujoco imported, no GL library"
- `s2load check <job> 64`: exit 0, "intact; fingerprint 0.0"
- `pack` twice: exit 0, identical bytes; `import --write`: exit 0, ok true, 0.0
- `git status --porcelain`: empty

PUBLIC NAMES: none changed. LOADER_RULE's text changed; fingerprint-file refusals now start "checkpoint <n> of <job>:".

OPEN FOR THE LEAD
- Confirm finding 1 (a); D14's "runs checkpoint_pair_problem" and the brief's line then apply to training loads only. Option (b) is a one-line revert plus a K9 text change.
- `actions_at_bound` (finding 8).
- If CB's .gitignore never arrives, the one skip stays.
- `newest_intact` does not open the .npz, so a resume on a checkpoint with an unreadable fingerprint gets CheckpointError from `load_checkpoint`; the trainer (CT) must handle it.
- Reviewer's note stands: the fingerprint cannot see critic weights, `log_std`, or weight changes moving actions under 1e-5 (the rule's tolerance).
