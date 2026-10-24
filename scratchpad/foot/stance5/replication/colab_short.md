# stance5 replication: two more T. rex stance runs (seeds 70 and 80), the unchanged band recipe
Start whenever you are ready: the branch is the stance5 one, already on GitHub at cf80ecb1. No control runs.

**RUN_ID is the folder name, NOT the label.** Run 4: RUN_ID `"stance5_LTSB_s70"`, RUN_LABEL `"stance screen LTSB s70"`. Run 5: RUN_ID `"stance5_LTSB_s80"`, RUN_LABEL `"stance screen LTSB s80"`. On stance5 the label went into RUN_ID once and the run had to be restarted. After Run all, the "Run directory:" line must end in `/ppo/stance5_LTSB_s70` (or `_s80`); if it shows the label, stop at once and tell me. Only a restart before the first checkpoint (below) ends in `_s70_b` / `_s80_b` (a second such restart: `_c`); it is the same run.

1. Open https://colab.research.google.com/github/kuds/mesozoic-labs/blob/claude/screen-trex-r9-armlts-copband/notebooks/sb3_training.ipynb, then Save a copy in Drive per run ("stance5r LTSB s70" / "s80"). Runtime > Change runtime type: L4 GPU for both, as stance5. If no L4 is offered, tell me before starting.
2. Section 1: `REPO_REF = "claude/screen-trex-r9-armlts-copband"` (both).
3. Section 2 settings:
   - SPECIES "tyrannosaurus_rex", EMBODIMENT "anatomical", ALGORITHM "ppo", N_ENVS 4.
   - QUICK_TEST False, USE_GOOGLE_DRIVE True, AUTO_DISCONNECT True.
   - BEHAVIOR "stance" (type it), TRUNK_FROM "", RETRAIN_FROM "".
   - RESUME_STAGE stays None.
   - Run 4: SEED 70, RUN_ID "stance5_LTSB_s70", RUN_LABEL "stance screen LTSB s70".
   - Run 5: SEED 80, RUN_ID "stance5_LTSB_s80", RUN_LABEL "stance screen LTSB s80".
   Use a fresh copy from the link for each run; do not reuse a stance5 copy.
   Before Run all: SEED, RUN_ID and RUN_LABEL all end in 70 (Run 5: 80); a fresh copy starts at SEED 42 and the seed is never printed.
4. Run all and allow Drive. Check:
   - "Repository ref: claude/screen-trex-r9-armlts-copband; commit: cf80ecb14d1630ce3b4dcdea28a05835d143470b". This is the decisive check.
   - "Run directory: .../ppo/stance5_LTSB_s70 (new run: ...)" (or _s80), with the RUN_ID exactly, and "Behavior 'stance' -> deliverable 'stance'; chain: stance".
   - The baseline "standing" row reads 4456.3 +/- 0.3 (reward about 4367.8; both measured locally, and on Colab the stance5 runs recorded the same reward, 4367.8). Other values mean a wrong config: 4458.2 +/- 0.3 is band off (the stance4 config), 4459.4 is band and stagger off, about 3980 is arm L keys only, about 3756 is the screen config off. Anything else: stop.
   - "Timesteps: 12,000,000", and an "Eval num_timesteps=..." line every 131,072 steps, the first within about 15-20 min.
   - Anything else (another commit, the label in the Run directory, "re-entering run" on a new run, another standing value, any other check line): stop and tell me.
5. Training takes about 14 h (measured on stance5: 13.7-14.5 h on the L4), then about 30 min of end-of-stage evaluation. The run disconnects itself.
   - On a PASS, if the runtime stays connected after "stance: gate PASSED", run section 9 by hand.
   - PASS and FAIL are both fine. Ignore "never beaten the zero-action baseline".
   - Do not stop or change a run. Tell me if you see "collapse" or an early stop.
   - I watch the saturation alarm at every check-in and tell you if it fires; nothing is stopped for it.

If a run disconnects, check its folder (MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo/<RUN_ID>/) in this order:
- 01_stance/gate_verdict.json exists: it finished. Do nothing, tell me.
- stage1_final.zip without stage1_final_vecnorm.pkl in 01_stance/models/: stop, tell me. It is finished from the newest intact periodic checkpoint in the same RUN_ID folder, never a new RUN_ID: once I have checked the ledgers, I ask you to rename the final file(s) there to `.cut` (deleting nothing) and resume.
- No stage1_*_steps.zip in 01_stance/models/ yet: restart as "<RUN_ID>_b" (a _b run cut off the same way: "<RUN_ID>_c") with RESUME_STAGE None and the same SEED and settings. Leave the first folder alone. It is the same run.
- Otherwise resume: the same copy, a fresh L4 runtime, the same settings (the SAME RUN_ID), `RESUME_STAGE = "stance"`, then Run all. If no L4 is offered, tell me before resuming on another runtime.
  - On a resume, "re-entering run" and "Timesteps: <remaining budget>" are expected.
  - If section 5 prints "Nothing to resume", that is fine too: section 6 re-runs the end-of-stage evaluation.
- An error in section 5 or 6: stop, tell me.

Tell me:
- when both have started: each run's commit line, its "Run directory:" line, SEED as set in the config cell, the baseline row and the runtime type (I then check each run's provenance.json within about 10 minutes);
- when each passes 6.3M (about 7.1-7.6 h, computed from the stance5 runs' rates);
- each finish, stop, resume or _b/_c restart (on a resume, also the new session's runtime type; on a restart, the new folder name).

The decision is read when both have ended: ADOPT only if both hold.

Criteria sha256 a38536de152e735947fd5322960f9e75b6520b59a5526a414afc0d0c4a0f3ef2. The full steps and criteria text are in colab_steps.txt.
