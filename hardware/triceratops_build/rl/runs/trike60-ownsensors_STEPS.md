<!-- Step sheet for the run on the robot's own sensors (decision D46; reading fixed in rl/analysis/PREREG_OWNSENSORS.md),
and a short second part: the delay and contact check on the policies already trained (D45).
Built from the foreleg comparison's sheet (runs/trike60-d22_STEPS.md) and the steering dev runs' FINAL sheet
(runs/trike60-steer1-dev_STEPS.md). The notebook and train.py are the same; ENV_PRESET and RUN_ID differ from the
four 128 mm runs. The code the notebook runs is commit <COMMIT>; this file only fills in the sheet.

PLACEHOLDERS FOR THE LEAD, all to be filled before this sheet is marked FINAL:
  - <COMMIT>: the 40 hex characters of the commit that holds rl/trike_env.py's obs_foot_loads,
    rl/configs/run3_terrain_ownsensors.json and the notebook's new preset (sections 1, 2.1, 2.4, 2.6 a). It
    should also hold demo_course.py's room setting (--room-c, the D48 batch), which the pre-registration's 35 C
    measure needs.
  - Part 2, section 7: see the box there.
Remove this paragraph and the DRAFT line below when the sheet is FINAL. -->

# Triceratops 60 cm: one run on the robot's own sensors on Colab. The steps.

*** DRAFT SHEET, NOT FOR USE YET: the commit below is a placeholder. I tell you when it is final. ***

Everything here is a simulation on Google's computers. Nothing touches the robot.

Three rules for the whole sheet (as before):
  - If a line on screen differs from the line written here, stop and tell me. Send me the line (a screenshot is fine).
  - Never change REPO_REF, RUN_ID, MODEL, TOTAL_STEPS, N_ENVS, SEED or ENV_PRESET of a run because an error message
    seems to suggest it. Never delete anything on your Drive. I tell you when something changes.
  - A run keeps its settings; a run is never extended.


# PART 1. THE RUN ON THE ROBOT'S OWN SENSORS

## 1. WHAT RUNS

    The purpose:  train the 128 mm model (D25) exactly as its four runs in the foreleg comparison were trained
                  (rough ground, pushes, slippery floors, the drift cost), but reading only what the robot's own
                  sensors will give. The policy no longer reads its foot loads, its ankle angles or the simulator's
                  true body speed (D46): 57 inputs instead of 68. One run, seed 0. How it is read is fixed before
                  it starts (rl/analysis/PREREG_OWNSENSORS.md). Nothing is judged, and one seed is not a result on
                  its own: I read it, then put the next step to you.
    The code:     one fixed commit of the repository,
                    <COMMIT>
                  This is REPO_REF below.
    The machine:  Runtime type "CPU" with "High-RAM" on. No GPU.
    The length:   TOTAL_STEPS = 10000000. The four 128 mm runs took 4.51-5.13 hours (542-617 steps per second; the
                  speed depends on the machine Colab gives), plus a few minutes for installs and the final test.
                  Expect the same: fewer inputs make no difference to the simulation.
    Drive:        My Drive / mesozoic-labs / triceratops60 / trike60-ownsensors-s0 ; about 90 MB.

    The run:

      RUN_ID                     MODEL                               SEED   ENV_PRESET
      trike60-ownsensors-s0      trike60_C_closed_capsule_fore128    0      run3_terrain_ownsensors

    And: TOTAL_STEPS 10000000, N_ENVS 8, EVAL_EPISODES 5, USE_GOOGLE_DRIVE ticked, AUTO_DISCONNECT ticked.


## 2. THE STEPS

  2.1  Open this address:
         https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<COMMIT>/hardware/triceratops_build/rl/trike60_training.ipynb
       Expect a notebook titled "Mesozoic Labs: Triceratops 60 cm, walking-policy training (simulation only)".
       Its first text names COLAB_STEPS.md and a 2 M-step pipeline test: that is from the first run; follow this
       sheet instead.

  2.2  Menu File > Save a copy in Drive. Name the copy "trike60 trike60-ownsensors-s0". Work in that copy, and use
       the same copy if this run ever needs a restart.

  2.3  Menu Runtime > Change runtime type. Choose CPU. Switch High-RAM on. Save.

  2.4  In the first grey box ("1. PARAMETERS") set exactly:
         REPO_REF           <COMMIT>
         RUN_ID             trike60-ownsensors-s0
         MODEL              trike60_C_closed_capsule_fore128
         TOTAL_STEPS        10000000
         N_ENVS             8
         SEED               0
         ENV_PRESET         run3_terrain_ownsensors     (new in the menu, below the steer1 presets)
         EVAL_EPISODES      5
         USE_GOOGLE_DRIVE   ticked
         AUTO_DISCONNECT    ticked
       The box refuses this preset with any other MODEL ("The run3_terrain_ownsensors preset trains the 128 mm
       build model ... Nothing was run."): then set MODEL as above and run again.

  2.5  Menu Runtime > Run all. Allow Google Drive when box 4 asks.

  2.6  Check these lines as they appear:
       (a) Under box 2:
             Repository commit: <COMMIT> (matches REPO_REF: yes; checkout clean: yes)            (seen in run 3)
       (b) Under box 3:
             Packages: mujoco 3.10.0, numpy 2.5.5, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu, cloudpickle 3.1.2
             All pins hold: yes                                                                  (seen in run 3)
       (c) Under box 4:
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-ownsensors-s0 (new run)
       (d) Under box 5:
             Running: train.py --out /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-ownsensors-s0 --total-steps 10000000 --n-envs 8 --seed 0 --model trike60_C_closed_capsule_fore128 --resume --exit-with-parent <pid> --env-config /content/mesozoic-labs/hardware/triceratops_build/rl/configs/run3_terrain_ownsensors.json
             Machine: <V> vCPUs; 8 envs; Python 3.<x>.<y>; torch 2.15.0+cpu                    (seen in run 3: 8 vCPUs)
             Starting a new run in /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-ownsensors-s0 (0 of 10000000 steps).
           Check that --model and --env-config show the values above.
       (e) Every 100,000 steps a progress line, and every 250,000 steps a "Saved checkpoint pair" line and two
           "eval at ..." lines, as in the earlier runs. Nothing to judge in them.

  2.7  THE END. Box 5 ends with
             Session completed: 10002432 steps (target 10000000); <R> env steps/s over <T> s; last pair trike60_10002432_steps.zip
       Box 6 then tests the final policy (about 2 minutes) and writes eval/eval_10002432.json; box 7 prints
             Run trike60-ownsensors-s0 is finished; its files are in /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-ownsensors-s0.
       and the runtime lets go by itself. After a restart (2.8) the number is another one from 10000000 to
       10008184 (a resumed session counts on from its save), and the eval file is named after it: that is expected.

  2.8  If Colab stops the run by itself, or the machine goes away: in the same copy, Edit > Clear all outputs, then
       Runtime > Run all with every value unchanged, and allow Drive. It resumes from the newest save that reached
       Drive ("Resuming ... from trike60_<n>_steps.zip"; "WARNING: skipping checkpoint ..." lines before it are fine).
       Tell me it happened. (A resume on Colab has still not been tested, so I want to know.)

  2.9  WHAT TO SEND ME: just tell me when it has started and when it has finished. I read the run's folder on your
       Drive myself (run_record.json, the final checkpoint pair, eval/). Only if something differed from this sheet,
       or the run stopped, send me the lines.


## 3. MESSAGES THAT MEAN: STOP AND TELL ME

    As in run 3's sheet (runs/trike60-run3-s1_STEPS.md, section 5): any "REFUSED: ...", "CRASHED: ...", "RuntimeError:
    ...", a "NO" in the commit line, "Pins do not hold", or anything else in red except pip's dependency warnings.
    Do not change any value, do not delete anything. Send me the line.


## 4. WHERE THE RESULTS ARE

    My Drive / mesozoic-labs / triceratops60 / trike60-ownsensors-s0 / : run_record.json, checkpoints/,
    eval_during_training.jsonl, train_health.jsonl, monitor/, eval/eval_10002432.json (or eval_<n>.json after a
    restart).


## 5. WHAT I DO WITH IT (not for you to do)

    - The final pair and run_record.json copied into a folder of their own; each tool's output is checked against
      that pair's sha256 where the tool records one (gait report, sweep, foot loads), and against its file name
      otherwise (demo_course.py).
    - First the record: one session (or its resumes) at <COMMIT> with a clean tree, seed 0, the fore128 model, and
      the env config of configs/run3_terrain_ownsensors.json over the defaults (57 inputs). Anything else is told to
      you before anything is read.
    - The measures of rl/analysis/PREREG_OWNSENSORS.md, with the tools at <COMMIT>: the gait report, the
      robustness sweep, the demo course in a 35 C room (D48) and at 25 C, and the foot loads. Each is read against
      the four 128 mm runs' stored results, line by line.
    - The reading, fixed before the run: no sign of a cost, a sign of a cost, or inconclusive. Each comes with the
      next step to put to you: a second seed in every case; if there is a sign of a cost, also a run with only the
      foot loads given back, which is the one that bears on buying foot load cells (D46). Nothing is ordered.


# PART 2. THE DELAY AND CONTACT CHECK ON THE POLICIES ALREADY TRAINED

## 6. WHAT IT IS

    The purpose:  two open questions, on trained policies rather than the scripted walk (D45). Each policy is
                  tested at servo (leg-command) delays of 16, 17, 18, 19 and 20 ms. In the stored sweeps the 128 mm
                  policies' worst drive is at 0.87-1.02 of its rating at 15 ms and 1.07-1.20 at 20 ms at 0.25 m/s,
                  and at 0.73-0.88 and 1.10-1.29 at 0.15 m/s (rl/analysis/REPORT_FORELEG_D22_APPENDIX.md:186-187
                  and :200-201; simulation). Nothing was measured in between. It is also tested with the stiffer
                  contact setting (the simulator's impratio 10) that the scripted walk and the foot study were
                  also judged under.
    The policies: the final policies of seven runs you have already trained:
                    trike60-fore128-s0, -s1, -s2, -s3   (trike60_10002432_steps.zip in each run's checkpoints/)
                    trike60-run3-s0, trike60-run3-s1     (trike60_10002432_steps.zip)
                    trike60-run2b-ankle                  (trike60_8003584_steps.zip)
                  Nothing is trained, and nothing on your Drive changes.
    What runs:    the robustness sweep (rl/robustness/sweep.py) on each policy, at 0.25 and 0.15 m/s: 10 ms (the
                  reference), 16, 17, 18, 19 and 20 ms, and impratio 10 at 10 ms. That is 14 points of 8 episodes,
                  112 per policy and 784 in all. The 10 and 20 ms points are already in the stored sweeps, so they
                  also check whether the new run repeats them exactly.
    The length:   an estimate for points the sweep cannot run yet (section 7's box): about 20 minutes in my
                  session's container, with 2 workers.
                  - The delay points, from the stored sweeps' own timing (each episode's wall_s in
                    rl/robustness/results/<run>/sweep_<run>.json.gz, 2 workers): 2.0-3.3 s per 20 s episode at 15 and
                    20 ms, 2.7, 2.7, 3.3 and 3.2 s on the four 128 mm policies, 2.2 and 2.0 s on run 3's and 2.6 s
                    on run2b's. 784 episodes at those times is about 35 minutes of one worker.
                  - The impratio 10 points, timed here on 27 Nov 2026 with sweep.py's own episode code and impratio
                    set on the model by hand (a scratch script, not the sweep; one process; 8 episodes per case at
                    10 ms on fore128-s0 and -s3): 2.86-3.29 s per episode against 2.45-2.75 s at the model's own
                    setting, 15-27 % longer. The default-setting episodes equalled the stored sweep's, speed and
                    falls, episode for episode. At 27 % longer the two impratio points add about 4 %: about 36
                    minutes of one worker.
                  On Colab it has not been timed.


## 7. WHAT YOU DO

    Nothing yet. I tell you if this part needs you.

  [PLACEHOLDER FOR THE LEAD, to settle and remove before FINAL]
    - Either add the two sweep options below (with tests and an independent check) and run part 2 here, or remove
      part 2 from this sheet before it is FINAL. As written, with the sweep, part 2 cannot be run by anyone. A
      third, partial route with no code change is evaluate.py (the last bullet but one); it is not the sweep's
      measurement.
    - sweep.py at aa150f89 cannot run these points. Its delays are fixed at 0, 2, 5, 10, 15, 20, 30, 40 and 60 ms
      (_DELAYS, rl/robustness/sweep.py:127). It has no contact (impratio) setting: make_point refuses unknown keys,
      and impratio is not one. No notebook runs sweep.py on Colab: the training notebook runs train.py and
      evaluate.py only.
    - The seven final pairs are already in this session's scratchpad, downloaded for the D22 and run 3 analyses.
      Each zip's and VecNormalize pickle's sha256 equals the one recorded in that policy's stored sweep
      (rl/robustness/results/<run>/sweep_<run>.json.gz; checked 27 Nov 2026). So with two sweep options (a delay
      list for both commands and an impratio setting, applied to every env the sweep builds and recorded in its
      JSON), part 2 runs here with no owner step. Then replace section 7 with "Nothing for you to do: I have
      copies of these policies, checked against their recorded fingerprints."
    - For the records: D45 says the delay sweep on trained policies "needs the checkpoints on the owner's Drive, so
      it becomes a Colab step for the owner" (DECISION_LOG.md, D45). That premise does not hold, since the seven
      pairs are local and match their fingerprints. The change belongs in the records (a note to D45, or a finding),
      which this package may not edit.
    - The partial route: rl/evaluate.py already takes any list of delays (--delays, rl/evaluate.py:63; one block
      per delay, :222-224). So "evaluate.py --checkpoint <pair>.zip --delays 15 16 17 18 19 20 --command 0.25
      --episodes 8", and again with --command 0.15, runs today on the local pairs (one 3 s episode at 16 ms on
      fore128-s0 ran on 28 Nov 2026; no figure from it is used). Its figures cannot be read against the stored
      sweep's 15 and 20 ms points, only against its own 15 and 20 ms blocks: it runs with domain randomisation and
      observation noise both on, or with --no-randomization both off (:140), where the sweep runs the nominal
      model with noise on (sweep.py:3-4, :260); its episodes are its own (seeds --seed + i in a randomised model);
      and its summary heat is each joint's worst episode (rl/gait_metrics.py:187-193), where the sweep reports the
      largest joint mean over episodes (sweep.py:500; the ratio and the ratings are the same, and evaluate.py's
      per-episode output gives the sweep's form). It has no impratio setting either, so the impratio 10 points cannot
      run this way. Run 2b's pair also needs its run_record.json one folder above the zip (or --model).
    - Only if it must run on Colab: it needs those options plus a reviewed notebook (or a box) that checks out
      <COMMIT>, installs the pins, copies each run's final pair and run_record.json from Drive into a folder of
      its own, and runs the sweep. Then write section 7's steps for that notebook, in this sheet's format.
