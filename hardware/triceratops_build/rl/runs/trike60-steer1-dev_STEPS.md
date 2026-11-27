<!-- Step sheet for the two steering dev runs (decisions D20, D25, D26, D28-D31; plan rl/PLAN_STEERING_RUN.md).
Built from the foreleg comparison's sheet (runs/trike60-d22_STEPS.md); the notebook and train.py are the same; MODEL,
ENV_PRESET and RUN_ID differ. The code the notebook runs is commit <<REPO_REF>>; this file only fills in the sheet. -->

# Triceratops 60 cm: the steering dev runs on Colab. The steps.

*** DRAFT, 27 Nov 2026: do not start. The code is being re-checked independently; this line changes to FINAL, with the
commit filled in, once that check is clean. ***

Everything here is a simulation on Google's computers. Nothing touches the robot.

Three rules for the whole sheet (as before):
  - If a line on screen differs from the line written here, stop and tell me. Send me the line (a screenshot is fine).
  - Never change REPO_REF, RUN_ID, MODEL, TOTAL_STEPS, N_ENVS, SEED or ENV_PRESET of a run because an error message
    seems to suggest it. Never delete anything on your Drive. I tell you when something changes.
  - A run keeps its settings; a run is never extended.


## 1. WHAT RUNS

    The purpose:  the first training of the steering task (D20): the robot follows speed and heading requests that
                  change every 3-5 s, on flat ground (D28) with pushes and slippery floors, on the 128 mm foreleg
                  (D25), with the new cost on the hottest drive's heat (D26, D29). Two runs that differ only in
                  that cost's weight, 1 and 2. They are NOT judged: they show whether steering learns and which
                  weight to use. The rules for the judged runs come after them.
    The code:     one fixed commit of the repository,
                    <<REPO_REF>>
                  This is REPO_REF below, the same for both runs.
    The machine:  Runtime type "CPU" with "High-RAM" on. No GPU.
    The length:   TOTAL_STEPS = 10000000, so the learning rate and every ramp are those of a judged run (D31). I read
                  the save at 5,000,000 steps. At the 128 mm foreleg runs' speeds (542-617 steps per second) that is
                  2.3-2.6 hours in; the whole run takes 4.5-5.1 hours. Once I tell you the 5 M save is on your Drive,
                  you may stop the runs (2.8) or let them finish: either is fine. Both runs can go at the same
                  time, each in its own notebook copy and its own runtime.
    Drive:        each run in  My Drive / mesozoic-labs / triceratops60 / <RUN_ID> ; about 90 MB each.

    The two runs:

      RUN_ID                          MODEL                               SEED   ENV_PRESET
      trike60-steer1-dev-heat1        trike60_C_closed_capsule_fore128    0      steer1_dev_heat1
      trike60-steer1-dev-heat2        trike60_C_closed_capsule_fore128    0      steer1_dev_heat2

    The same values for both: TOTAL_STEPS 10000000, N_ENVS 8, SEED 0, EVAL_EPISODES 5, USE_GOOGLE_DRIVE ticked,
    AUTO_DISCONNECT ticked.


## 2. EACH RUN (do this once per row of the table)

  2.1  Open this address:
         https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<<REPO_REF>>/hardware/triceratops_build/rl/trike60_training.ipynb
       Expect a notebook titled "Mesozoic Labs: Triceratops 60 cm, walking-policy training (simulation only)".
       Its first text names COLAB_STEPS.md and a 2 M-step pipeline test: that is from the first run; follow this
       sheet instead.

  2.2  Menu File > Save a copy in Drive. Name the copy "trike60 <RUN_ID>". Work in that copy, and use the same copy
       if this run ever needs a restart.

  2.3  Menu Runtime > Change runtime type. Choose CPU. Switch High-RAM on. Save.

  2.4  In the first grey box ("1. PARAMETERS") set exactly, with this row's RUN_ID and ENV_PRESET:
         REPO_REF           <<REPO_REF>>
         RUN_ID             <RUN_ID>
         MODEL              trike60_C_closed_capsule_fore128
         TOTAL_STEPS        10000000
         N_ENVS             8
         SEED               0
         ENV_PRESET         <ENV_PRESET>     (the menu offers steer1_dev_heat1 and steer1_dev_heat2)
         EVAL_EPISODES      5
         USE_GOOGLE_DRIVE   ticked
         AUTO_DISCONNECT    ticked
       The box refuses a steer1 preset with any other MODEL ("The steer1 presets train the 128 mm build model ...
       Nothing was run."): then set MODEL as above and run again.

  2.5  Menu Runtime > Run all. Allow Google Drive when box 4 asks.

  2.6  Check these lines as they appear:
       (a) Under box 2:
             Repository commit: <<REPO_REF>> (matches REPO_REF: yes; checkout clean: yes)            (seen in run 3)
       (b) Under box 3:
             Packages: mujoco 3.10.0, numpy 2.5.5, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu, cloudpickle 3.1.2
             All pins hold: yes                                                                  (seen in run 3)
       (c) Under box 4:
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> (new run)
       (d) Under box 5:
             Running: train.py --out /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> --total-steps 10000000 --n-envs 8 --seed 0 --model trike60_C_closed_capsule_fore128 --resume --exit-with-parent <pid> --env-config /content/mesozoic-labs/hardware/triceratops_build/rl/configs/<ENV_PRESET>.json
             Machine: <V> vCPUs; 8 envs; Python 3.<x>.<y>; torch 2.15.0+cpu                    (seen in run 3: 8 vCPUs)
             Starting a new run in /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> (0 of 10000000 steps).
           Check that --model and --env-config show this row's values.
       (e) Every 100,000 steps a progress line, and every 250,000 steps a "Saved checkpoint pair" line and two
           "eval at ..." lines, as in the earlier runs. Nothing to judge in them.

  2.7  THE END, if you let it finish. Box 5 ends with
             Session completed: 10002432 steps (target 10000000); <R> env steps/s over <T> s; last pair trike60_10002432_steps.zip
       Box 6 then tests the final policy (a few minutes) and writes eval/eval_10002432.json; box 7 prints
             Run <RUN_ID> is finished; its files are in /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID>.
       and the runtime lets go by itself. After a restart (2.9) the number is another one from 10000000 to
       10008184 (a resumed session counts on from its save), and the eval file is named after it: that is expected.

  2.8  STOPPING AFTER 5 M (only once I have told you the 5 M save is on your Drive; letting the run finish is also
       fine). Click the stop button at the left of box 5 (the square in a circle while it runs) ONCE, and wait: do
       not click again until the line that begins "Session interrupted" and the two Drive lines have appeared.
       Expect, in this order (from the notebook's own code):
             Stop requested: the step is asked to stop (train.py first saves a checkpoint pair). Wait for its last line; further clicks are ignored.
             Interrupted: saving a checkpoint pair before exiting.
             Session interrupted: <n> steps (target 10000000); <R> env steps/s over <T> s; last pair trike60_<n>_steps.zip
             Flushing Google Drive writes...
             Drive flushed and unmounted; the next run of cell 5 mounts it again.
       then a red "KeyboardInterrupt" under the box. That red text is expected: it is how a stopped box ends.
       Then menu Runtime > Disconnect and delete runtime (a stopped run does not let go by itself). Do not run the
       boxes again for a stopped run.
       If nothing new appears for 5 minutes after you click stop: Runtime > Disconnect and delete runtime, and tell me.

  2.9  If Colab stops the run by itself, or the machine goes away, before 5 M: in the same copy, Edit > Clear all
       outputs, then Runtime > Run all with every value unchanged, and allow Drive. It resumes from the newest save
       that reached Drive ("Resuming ... from trike60_<n>_steps.zip"; "WARNING: skipping checkpoint ..." lines before
       it are fine). Tell me it happened. After 5 M, tell me before restarting: it may not be needed.

  2.10 WHAT TO SEND ME: just tell me when both runs have started, and again if one stopped by itself. I read each
       run's folder on your Drive myself (run_record.json, the checkpoint pairs, eval_during_training.jsonl,
       train_health.jsonl). Only if something differed from this sheet, send me the lines.


## 3. MESSAGES THAT MEAN: STOP AND TELL ME

    As in run 3's sheet (runs/trike60-run3-s1_STEPS.md, section 5): any "REFUSED: ...", "CRASHED: ...", "RuntimeError:
    ...", a "NO" in the commit line, "Pins do not hold", or anything else in red except pip's dependency warnings and
    the KeyboardInterrupt after a stop (2.8). Do not change any value, do not delete anything. Send me the line.


## 4. WHERE THE RESULTS ARE

    My Drive / mesozoic-labs / triceratops60 / <RUN_ID> / : run_record.json, checkpoints/ (trike60_5000000_steps.zip
    and its vecnormalize pair is the one I read), eval_during_training.jsonl, train_health.jsonl, monitor/, and
    eval/eval_<n>.json if the run finished (n is 10002432, or another number from 10000000 to 10008184 after a
    restart).


## 5. WHAT I DO WITH THEM (not for you to do)

    - The 5 M save: trike60_5000000_steps.zip and its vecnormalize pair, copied with the run's run_record.json into a
      folder of their own, and every tool pointed there (the tools take the newest pair in a folder, and a stopped or
      finished run has newer ones); each output's checkpoint sha256 is checked against that pair.
    - From the 5 M save of each run: the gait report, the sweep (with its x1.1 mass, payload and 20 ms delay points),
      the foot loads, steer_eval.py's courses and the demo course with turns, all on the flat floor with per-body
      randomisation only (README, "The steering run's training additions").
    - Compare the two weights: the hottest drive (target case and stress points), walking and turning, falls under
      pushes, and whether either run drifted towards standing still. Both runs use seed 0, so the weight is the
      only setting that differs; a difference in the hottest drive under about 0.1 of its rating is within the
      seed-to-seed spread seen in D22 (0.83-0.95 over four seeds) and is not read as a weight effect.
    - THE JUDGED WEIGHT, by this rule, fixed before the runs (27 Nov 2026). The measures, all from each run's 5 M save
      unless said otherwise, read from gait_report.json (its Markdown summary is not needed and may be missing for a
      run that stands still):
        speed  the speed at the 0.25 m/s command: cases.target_0.25.summary.speed.forward_speed_mps (the randomised
               target case, 5 episodes);
        falls  cases.target_0.25.summary.falls + cases.cmd_0.15.summary.falls (the two randomised cases, 10
               episodes);
        heat   H, the largest of cases.target_0.25.summary.drives.per_joint[*].rms_over_continuous.mean: the worst
               joint's RMS torque over its continuous rating (F2's figure, judge_d22's H_all);
        noise  the action noise at the rollout nearest 3,000,000 steps in the run's train_health.jsonl: the 4-decimal
               action_std that analysis/d22_action_std.py writes to its JSON (it keeps a resumed run's latest
               branch).
      A weight is out if any of:
        - speed under 0.20 m/s;
        - 2 or more falls of the 10;
        - noise over its limit: over 0.1240 for weight 2, over 0.1580 for weight 1 (the 4-decimal value). Each
          limit is where that weight's worst
          margin over standing, at the two nominal speeds on the four D22 policies at full scale
          (analysis/results/steer1/heat_weight/heat_weight_margins_d22noise_3M_5M.md and heat_weight_margins.md),
          crosses the README's 0.3, by straight-line interpolation between the measured noise rows (weight 2: +0.37 at
          0.12 and +0.29 at 0.125, at 0.25 m/s, crossing at 0.1242; weight 1: +0.38 at 0.15 and +0.13 at 0.175, at
          0.15 m/s, crossing at 0.1577), rounded to three decimals. Weight 2 is on that line at the D22 runs' own 3 M
          noise (0.1230, 0.1268, 0.1280, 0.1282): it stays in only if its run's noise is at the bottom of that range
          or lower, which r_heat_peak itself may cause, since it also prices noise.
      Then:
        - none left: I come back to you before any judged run;
        - one left: that one if its H is under 0.85; otherwise I come back to you (as below, the planned heat rule
          would likely fail with it);
        - both left: if both H are 0.85 or more, I come back to you (the planned heat rule would likely fail with
          either); otherwise weight 2 if its H is at least 0.1 below weight 1's; otherwise weight 1 if its H is under
          0.85; otherwise weight 2 (its H is under 0.85 and weight 1's is not).
      The check uses the D22 policies' margins with each dev run's measured noise; it does not re-measure margins on
      the dev runs' own policies.
    - Then the pre-registration of the judged runs (rules, seeds, stop rules), independently reviewed before any
      judged run.
