<!-- Step sheet for the foreleg comparison's eight runs (decisions D22, D23, D24; rules rl/analysis/PREREG_FORELEG_D22.md).
Built from run 3's sheets (runs/trike60-run3-s1_STEPS.md); the notebook and train.py are the same; only MODEL, RUN_ID and SEED
change between runs. The code the notebook runs is commit f84a6919dda31c39f1a56967b1f57b759c31da5c; this file only fills in the sheet. -->

# Triceratops 60 cm: the foreleg comparison on Colab. The steps.

*** FINAL SHEET for the eight runs of the foreleg comparison (D22), 23 Nov 2026. ***
Every value is filled in: the commit (f84a6919dda31c39f1a56967b1f57b759c31da5c), and for each run its name, model and seed (the table
in 1). The steps are run 3's: you ran them twice already. Lines marked "(seen in run 3)" appeared on your screen then;
the rest are printed by the program itself and checked against the code.

Everything here is a simulation on Google's computers. Nothing touches the robot.

Three rules for the whole sheet (as before):
  - If a line on screen differs from the line written here, stop and tell me. Send me the line (a screenshot is fine).
  - Never change REPO_REF, RUN_ID, MODEL, TOTAL_STEPS, N_ENVS, SEED or ENV_PRESET of a run because an error message
    seems to suggest it. Never delete anything on your Drive. I tell you when something changes.
  - A run keeps its settings; a run is never extended.


## 1. WHAT RUNS

    The purpose:  train the 60 cm model with each of PR #697's repackaged forelegs, 128 mm and 160 mm upper links,
                  exactly as run 3 was trained (rough ground, pushes, slippery floors, the drift cost), on four
                  seeds each, to compare foot loads, falls and drive heat (D22, D23, D24). The rules are fixed in
                  rl/analysis/PREREG_FORELEG_D22.md; nothing is judged until I have read all eight runs.
    The code:     one fixed commit of the repository,
                    f84a6919dda31c39f1a56967b1f57b759c31da5c
                  This is REPO_REF below, the same for all eight runs.
    The machine:  Runtime type "CPU" with "High-RAM" on. No GPU.
    The length:   TOTAL_STEPS = 10000000 per run. Run 3's two runs took 3.05 and 5.05 hours (909 and 550 steps per
                  second: the speed depends on the machine Colab gives), plus a few minutes for installs and the
                  final test. Two runs can go at the same time, each in its own notebook copy, as you did for run 3.
    Drive:        each run in  My Drive / mesozoic-labs / triceratops60 / <RUN_ID> ; about 90 MB each.

    The eight runs (any order; I suggest the rounds below, two at a time):

      round  RUN_ID                MODEL                               SEED
      1      trike60-fore128-s0    trike60_C_closed_capsule_fore128    0
      1      trike60-fore160-s0    trike60_C_closed_capsule_fore160    0
      2      trike60-fore128-s1    trike60_C_closed_capsule_fore128    1
      2      trike60-fore160-s1    trike60_C_closed_capsule_fore160    1
      3      trike60-fore128-s2    trike60_C_closed_capsule_fore128    2
      3      trike60-fore160-s2    trike60_C_closed_capsule_fore160    2
      4      trike60-fore128-s3    trike60_C_closed_capsule_fore128    3
      4      trike60-fore160-s3    trike60_C_closed_capsule_fore160    3

    The same values for every run: TOTAL_STEPS 10000000, N_ENVS 8, ENV_PRESET run3_terrain, EVAL_EPISODES 5,
    USE_GOOGLE_DRIVE ticked, AUTO_DISCONNECT ticked.


## 2. EACH RUN (do this once per row of the table)

  2.1  Open this address:
         https://colab.research.google.com/github/kuds/mesozoic-labs/blob/f84a6919dda31c39f1a56967b1f57b759c31da5c/hardware/triceratops_build/rl/trike60_training.ipynb
       Expect a notebook titled "Mesozoic Labs: Triceratops 60 cm, walking-policy training (simulation only)".

  2.2  Menu File > Save a copy in Drive. Name the copy "trike60 <RUN_ID>". Work in that copy, and use the same copy
       if this run ever needs a restart.

  2.3  Menu Runtime > Change runtime type. Choose CPU. Switch High-RAM on. Save.

  2.4  In the first grey box ("1. PARAMETERS") set exactly, with this row's RUN_ID, MODEL and SEED:
         REPO_REF           f84a6919dda31c39f1a56967b1f57b759c31da5c
         RUN_ID             <RUN_ID>
         MODEL              <MODEL>          (the menu now offers both fore128 and fore160)
         TOTAL_STEPS        10000000
         N_ENVS             8
         SEED               <SEED>
         ENV_PRESET         run3_terrain
         EVAL_EPISODES      5
         USE_GOOGLE_DRIVE   ticked
         AUTO_DISCONNECT    ticked

  2.5  Menu Runtime > Run all. Allow Google Drive when box 4 asks.

  2.6  Check these lines as they appear:
       (a) Under box 2:
             Repository commit: f84a6919dda31c39f1a56967b1f57b759c31da5c (matches REPO_REF: yes; checkout clean: yes)      (seen in run 3)
       (b) Under box 3:
             Packages: mujoco 3.10.0, numpy 2.5.5, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu, cloudpickle 3.1.2
             All pins hold: yes                                                                  (seen in run 3)
       (c) Under box 4:
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> (new run)
       (d) Under box 5:
             Running: train.py --out /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> --total-steps 10000000 --n-envs 8 --seed <SEED> --model <MODEL> --resume --exit-with-parent <pid> --env-config /content/mesozoic-labs/hardware/triceratops_build/rl/configs/run3_terrain.json
             Machine: <V> vCPUs; 8 envs; Python 3.<x>.<y>; torch 2.15.0+cpu                    (seen in run 3: 8 vCPUs)
             Starting a new run in /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> (0 of 10000000 steps).
           Check that --seed and --model show this row's values.
       (e) Every 100,000 steps a progress line, and every 250,000 steps a "Saved checkpoint pair" line and two
           "eval at ..." lines, as in run 3. Nothing to judge in them.

  2.7  THE END. Box 5 ends with
             Session completed: 10002432 steps (target 10000000); <R> env steps/s over <T> s; last pair trike60_10002432_steps.zip
       Box 6 then tests the final policy (about 2 minutes) and writes eval/eval_10002432.json; box 7 prints
             Run <RUN_ID> is finished; its files are in /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID>.
       and the runtime lets go by itself.

  2.8  If Colab stops the run by itself, or the machine goes away: in the same copy, Edit > Clear all outputs, then
       Runtime > Run all with every value unchanged, and allow Drive. It resumes from the newest save that reached
       Drive ("Resuming ... from trike60_<n>_steps.zip"; "WARNING: skipping checkpoint ..." lines before it are fine).
       Tell me it happened. (This is run 3's stop rule; the resume path on Colab has not been tested yet, so I want to know.)

  2.9  WHAT TO SEND ME: just tell me when a round has finished. I read each run's folder on your Drive myself
       (run_record.json, the final checkpoint pair, eval/). Only if something differed from this sheet, or the run
       stopped, send me the lines.


## 3. MESSAGES THAT MEAN: STOP AND TELL ME

    As in run 3's sheet (runs/trike60-run3-s1_STEPS.md, section 5): any "REFUSED: ...", "CRASHED: ...", "RuntimeError:
    ...", a "NO" in the commit line, "Pins do not hold", or anything else in red except pip's dependency warnings.
    Do not change any value, do not delete anything. Send me the line.


## 4. WHERE THE RESULTS ARE

    My Drive / mesozoic-labs / triceratops60 / <RUN_ID> / : run_record.json, checkpoints/, eval_during_training.jsonl,
    monitor/, eval/eval_10002432.json (as for run 3).
