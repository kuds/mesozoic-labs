<!-- Filled-in copy of rl/COLAB_STEPS.md for run trike60-run3-s0 (decisions D18; rules rl/analysis/PREREG_RUN3.md):
every placeholder replaced; MODEL the roll-ankle model; SEED 0; ENV_PRESET run3_terrain. Written 19 Nov 2026.
The code the notebook runs is commit f0a03c6e; this file only fills in the sheet. -->

# Triceratops 60 cm: a training run on Colab, with the stop and resume tests. The steps.

*** FINAL SHEET for run trike60-run3-s0 (one of two: seeds 0 and 1), 19 Nov 2026. Start from this one. ***
Every value is filled in: above all the commit (f0a03c6e109e6e0723b0650b0c05df83e516ef1b), the
run's name (trike60-run3-s0) and its length (10000000), and the numbers that follow from them. Run2b ran this notebook on
Colab straight through; its first attempt tested the stop button, but a resume on Colab has never been tested, so this run does the stop and resume tests (3.7, 3.8). Every line quoted below is printed by the program itself (I checked the
wording against the code and a local run); lines marked "(expected, not yet seen)" come from Colab and I have not
seen them yet.

Everything here is a simulation on Google's computers. Nothing touches the robot.

You need: a browser (a laptop is easier), your Google account, and this sheet. You do not need to understand the
output. You compare a few lines with the ones written here, and send me the ones I ask for.

Three rules for the whole sheet:
  - If a line on screen differs from the line written here, stop and tell me. Send me the line (a screenshot is
    fine).
  - Never change REPO_REF, RUN_ID, MODEL, TOTAL_STEPS, N_ENVS, SEED or ENV_PRESET of a run because an error message
    seems to suggest it. Never delete anything on your Drive. I tell you when something changes.
  - A run keeps its settings. To train longer or differently later, I give you a NEW RUN_ID; a run is never
    extended (the program refuses a changed TOTAL_STEPS, and the learning rate would jump if forced).


## 1. WHAT RUNS

    The purpose:  train the roll-ankle 60 cm model for 10 million steps on rough ground (small steps, ramps, sideways pushes and slippery floors, phased in between 1 and 4 million steps) with a cost on sideways drift, seed 0 of two, to see whether it keeps its flat walking and copes with what run2b could not. On the way it tests the stop button (3.7) and the resume (3.8).
                  The rules it is judged by are fixed in rl/analysis/PREREG_RUN3.md; nothing is judged until I
                  have read the results of both seeds.
    The code:     one fixed commit of the repository,
                    f0a03c6e109e6e0723b0650b0c05df83e516ef1b
                  This is REPO_REF below. The notebook fetches only that commit (no history).
    The notebook: hardware/triceratops_build/rl/trike60_training.ipynb at that commit.
    The machine:  Runtime type "CPU" with "High-RAM" on (more processors). No GPU: it adds nothing here.
    The length:   TOTAL_STEPS = 10000000 simulation steps. Run2b ran at about 692 steps per second
                  overall on this runtime (8 processors, the checks during training included), and the terrain costs
                  about 3 % more, so this run trains for about 4.1 hours, plus a few minutes for the installs and the final evaluation. (If Colab lets you
                  open a second runtime, the two seeds can run at the same time, each in its own notebook copy.)
                  The progress lines print the speed (3.6 (f)).
    Drive:        everything goes into  My Drive / mesozoic-labs / triceratops60 / trike60-run3-s0 ; about 90 MB
                  (a save is about 2 MB, one every 250,000 steps, plus one at every stop).


## 2. BEFORE YOU START

    Nothing to upload or install. Keep this sheet open next to the notebook.


## 3. THE RUN, WITH THE STOP AND RESUME TESTS

  3.1  Open this address:
         https://colab.research.google.com/github/kuds/mesozoic-labs/blob/f0a03c6e109e6e0723b0650b0c05df83e516ef1b/hardware/triceratops_build/rl/trike60_training.ipynb
       Expect: a notebook titled "Mesozoic Labs: Triceratops 60 cm, walking-policy training (simulation only)".
       Opening it runs nothing.

  3.2  Menu File > Save a copy in Drive. Name the copy "trike60 trike60-run3-s0". Work in the copy from now on, and use
       the same copy for every restart of this run (3.8, 3.9).

  3.3  Menu Runtime > Change runtime type. Choose CPU. Switch High-RAM on. Save.

  3.4  In the first grey box ("1. PARAMETERS") set exactly:
         REPO_REF           f0a03c6e109e6e0723b0650b0c05df83e516ef1b
         RUN_ID             trike60-run3-s0
         MODEL              trike60_C_closed_ankleroll
         TOTAL_STEPS        10000000
         N_ENVS             8
         SEED               0
         ENV_PRESET         run3_terrain
         EVAL_EPISODES      5
         USE_GOOGLE_DRIVE   ticked
         AUTO_DISCONNECT    ticked

  3.5  Menu Runtime > Run all.
       Box 3 installs the software; that takes a few minutes and asks nothing.          (expected, not yet seen)
       Then box 4 opens a window asking whether this notebook may access your Google Drive files. Choose "Connect
       to Google Drive", pick your account, allow. The run waits for your answer.       (expected, not yet seen)

  3.6  Check these lines as they appear, in this order:

       (a) Under box 1:  Parameters of this session:   and your values, one per line.

       (b) Under box 2 (one line):
             Repository commit: f0a03c6e109e6e0723b0650b0c05df83e516ef1b (matches REPO_REF: yes; checkout clean: yes)
           Another commit, or a "NO": stop and tell me.

       (c) Under box 3:
             Python 3.<x>.<y> (CPython 3.12-3.14 needed): yes
             Packages: mujoco 3.10.0, numpy 2.5.5, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu, cloudpickle 3.1.2
             All pins hold: yes
           Pip may print warnings in red about other packages ("dependency conflicts"); they do not matter here.
                                                                                        (expected, not yet seen)
           If box 3 stops with "the pinned packages need CPython 3.12, 3.13 or 3.14", "did not install" or "Pins do
           not hold": stop and send me everything box 3 printed.

       (d) Under box 4:
             Mounted at /content/drive                                                  (expected, not yet seen)
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-run3-s0 (new run)

       (e) Under box 5 (each is one line on screen):
             Running: train.py --out /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-run3-s0 --total-steps 10000000 --n-envs 8 --seed 0 --model trike60_C_closed_ankleroll --resume --exit-with-parent <pid> --env-config /content/mesozoic-labs/hardware/triceratops_build/rl/configs/run3_terrain.json
             Machine: <V> vCPUs; 8 envs; Python 3.<x>.<y>; torch 2.15.0+cpu
             Starting a new run in /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-run3-s0 (0 of 10000000 steps).
           <pid> is a number (the notebook's own process: train.py stops by itself if that process goes away).
           <V> is the number of processors Colab gave (8 expected with High-RAM).       (expected, not yet seen)
           If the Machine line ends "WARNING: more envs than vCPUs, each env runs slower": go on, and tell me <V>.

       (f) Every 100,000 steps a progress line (one line on screen), the first one:
             steps 100000 of 10000000 (1.0%), <R> env steps/s, penalty scale 0.14, late scale 0.00, terrain scale 0.00; training rollouts (stochastic actions), last <k> episodes: return ..., length ... steps, speed ... m/s at command ... m/s
           <R> is the speed. "Stochastic actions" means these numbers come from the training's random exploration;
           they say nothing about how the policy walks. The penalty scale (the weight of the energy and slip costs)
           rises from 0.10 to 1.00 over the first 2,000,000 steps; the late scale (drive heat, torque limits, short
           steps) stays at 0.00 until 1,000,000 steps and then rises to 1.00 at 3,000,000; the terrain scale
           (steps, ramps, pushes, slippery floors) stays at 0.00 until 1,000,000 steps and rises to 1.00 at
           4,000,000. The checks of the policy in (g) are always on a flat floor.

       (g) Every 250,000 steps three lines: a save, and a short check of the policy in two lines:
             Saved checkpoint pair at 250000 steps: trike60_250000_steps.zip and trike60_vecnormalize_250000_steps.pkl
             eval at 250000 steps (deterministic, delay 10 ms): command -> speed m/s: 0.00 -> ..., 0.15 -> ..., 0.25 -> ...; falls ... of 3
             eval at 250000 steps, command 0.25 m/s: RMS torque/rating roll ... pitch ... elbow ... knee ... (worst ...); at >=99% nominal limit .../.../.../... %; slip ratio LF/RF/LH/RH .../.../.../...; touchdowns/s .../.../.../...; power ... W; drift ... m, heading ... deg
           The third line is for me (drive heat, torque limits, foot slip, steps); nothing to judge.
           (At 500,000 steps the progress line stands between the save and the check.) A "Saved" line means the file is written to
           the Drive folder as the runtime sees it; Drive uploads it to Google in the background, so it can reach
           your Drive a little later.                                                   (expected, not yet seen)

  3.7  THE STOP TEST. Wait until you have seen the "Saved checkpoint pair" line for 500000 steps (two saves; about
       13 minutes at 670 steps per second). Then click the stop button at the left of box 5 (the square in a
       circle while it runs) ONCE, and wait: do not click again until the line that begins "Session interrupted"
       and the two Drive lines have appeared (a few seconds). A second click changes nothing; it only prints
             (the step is still saving: this click is ignored; please wait)
       Expect, in this order:
             Stop requested: the step is asked to stop (train.py first saves a checkpoint pair). Wait for its last line; further clicks are ignored.
             Interrupted: saving a checkpoint pair before exiting.
             Session interrupted: <n> steps (target 10000000); <R> env steps/s over <T> s; last pair trike60_<n>_steps.zip
             Flushing Google Drive writes...
             Drive flushed and unmounted; the next run of cell 5 mounts it again.
       then a red "KeyboardInterrupt" under the box. That red text is expected: it is how a stopped box ends.
                                                                                        (expected, not yet seen)
       Write down <n>.
       If nothing new appears for 5 minutes after you click stop: Runtime > Disconnect and delete runtime, and tell
       me. (While train.py saves, the notebook ignores further clicks; a save takes seconds, so 5 minutes of silence
       means something is stuck. What was saved before is on your Drive; 3.9 resumes from it when I ask you to.)

  3.8  THE RESUME TEST. Click the play button of box 5 (only box 5). If a window asks for Drive again, allow it as
       in 3.5.                                                                          (expected, not yet seen)
       Expect:
             Mounted at /content/drive                                                  (expected, not yet seen)
             Running: train.py ... --resume --exit-with-parent <pid> --env-config /content/mesozoic-labs/hardware/triceratops_build/rl/configs/run3_terrain.json
             Machine: <V> vCPUs; 8 envs; ...
             Resuming /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-run3-s0 from trike60_<n>_steps.zip (<n> of 10000000 steps).
       <n> must be the number you wrote down in 3.7: the step count carries on. Then the progress lines go on from
       there. If the Resuming line names another number, or says "Starting a new run": stop and tell me.

  3.9  (Optional) THE RECONNECT TEST. After the next "Saved checkpoint pair" line, click Runtime > Disconnect and
       delete runtime. This throws the machine away without a stop. Then Edit > Clear all outputs, then Runtime >
       Run all with every value unchanged, and allow Drive. Box 3 installs again (a fresh machine). Expect:
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-run3-s0 (resuming: <k> checkpoint(s) found)
             Resuming /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-run3-s0 from trike60_<m>_steps.zip (<m> of 10000000 steps).
       <m> is the newest save that had fully reached Drive: the last "Saved" number, or an earlier one if its upload
       had not finished when the machine went away. Lines before it like
             WARNING: skipping checkpoint trike60_<j>_steps.zip: ...
       mean exactly that and are fine. How much a disconnect loses is what this step measures; it is not known yet.
                                                                                        (expected, not yet seen)

  3.10 THE END. Leave it running. Box 5 ends with one line:
             Session completed: <e> steps (target 10000000); <R> env steps/s over <T> s; last pair trike60_<e>_steps.zip
       <e> is a little above 10000000: training works in blocks of 8 x 1024 = 8192 steps. Without a stop it
       is 10002432; after a stop (3.7) it is another number from
       10000000 to 10008184, because the blocks then count from the stop.
       If you started box 5 alone (3.8), now click the play button of box 6, and when it ends, of box 7. After a
       Run all (3.9) they run by themselves.

       Box 6 tests the final policy at fixed commands, about 2 minutes. It prints lines like
             delay 10 ms: 5 episodes, forward speed ... m/s (command ...), |error| ... m/s, falls ..., length ... s
             delay  0 ms: 5 episodes, ...
             sweep, delay 10 ms, command 0.00 m/s: 5 episodes, ...
             (and the same for 0.10, 0.20, 0.25, 0.27)
             TARGET 0.25 m/s at 10 ms: 5 episodes, ...
             TARGET 0.25 m/s at 10 ms, gait: RMS torque/rating roll ... pitch ... elbow ... knee ... (worst ...); at >=99% nominal limit ...; slip ratio ...; touchdowns/s ...; power ... W; drift ... m, heading ... deg
             Wrote /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-run3-s0/eval/eval_<e>.json
       Nothing to judge here: I read the file.

       Box 7 prints
             Run trike60-run3-s0 is finished; its files are in /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-run3-s0.
             Flushing Google Drive writes...
             Disconnecting runtime in 5 seconds...
       and the runtime lets go by itself.                                               (expected, not yet seen)

  3.11 WHAT TO SEND ME (copy the lines from the screen; a screenshot is fine):
         - box 3's three lines (Python, Packages, All pins hold);
         - the "Machine:" line and the first two progress lines ("steps 100000 ..." and "steps 200000 ...");
         - every "Saved checkpoint pair" line;
         - the stop lines of 3.7 and the "Resuming" line of 3.8 (and of 3.9 if you did it, with any WARNING lines);
         - the "Session completed" line;
         - the file eval/eval_<e>.json from the run folder (download it from Drive and attach it, or tell me it is
           there and I read it).
       And the time it took from Run all to the end.


## 4. LATER RUNS

    Only when I send you their values: a new copy of the notebook (3.2) with a new RUN_ID and the TOTAL_STEPS, N_ENVS
    and MODEL I give you; then 3.3 to 3.6 and 3.10 as above. The stop and reconnect tests (3.7 to 3.9) only when I
    ask; to resume after a stop: 3.8; after a disconnect: 3.9.


## 5. MESSAGES THAT MEAN: STOP AND TELL ME

    Do not change any value, do not delete anything. Send me the line.

      REFUSED: these settings differ from the run's ...   a value in box 1 differs from the run's first session
      REFUSED: the policy's shape ...
      REFUSED: ... already holds checkpoints ...
      Run <RUN_ID> was started from commit ...            the run belongs to another REPO_REF (box 4)
      ... (matches REPO_REF: NO ...) or (checkout clean: NO)
      ... the pinned packages need CPython 3.12, 3.13 or 3.14 ...
      Pins do not hold: ...   or   ... did not install ...
      Google Drive is not mounted; nothing was started ...
      RuntimeError: train.py stopped with exit code ...     (the reason is in the lines above it; send both)
      RuntimeError: evaluate.py stopped with exit code ...
      CRASHED: ...
      WARNING: the parameters are not finite ... saved as checkpoints/nonfinite/...
                                                           training diverged; that save is set aside and never
                                                           resumed from (a resume uses the newest finite save)
      REFUSED: the checkpoint files could not be read ...  Drive did not answer, also after three re-reads 20 s
                                                           apart; nothing was moved or changed. Wait a few
                                                           minutes, run box 5 once more; if it repeats, tell me
      REFUSED: no checkpoint is listed in ..., but ... Starting at step 0 could overwrite them ...
                                                           Drive shows the run folder without the saves an earlier
                                                           session made; nothing was changed. Wait a few minutes,
                                                           run box 5 once more; if it repeats, tell me
      REFUSED: the checkpoint folder ... could not be listed ...
                                                           the same, when Drive does not even list the folder
      REFUSED: there is no finite checkpoint to resume from ...
                                                           training diverged and every save since its last restart
                                                           is non-finite; nothing was changed; tell me (a new run
                                                           with other settings is needed)
      REFUSED: the parent process ... is gone ...          should never appear; tell me
      Anything else in red, except pip's dependency warnings in box 3 and the KeyboardInterrupt after a stop (3.7).

    Tell me, but leave the run going:
      WARNING: no intact checkpoint pair; moved <k> broken file(s) to checkpoints/broken_<time>/ and starting this
               run again from step 0.                       every save on Drive was damaged (not merely unreadable);
                                                           they are kept in that folder, and the run started over

    These are fine and need nothing from you:
      WARNING: skipping checkpoint ...                     a save that had not fully reached Drive (3.9)
      WARNING: reading the checkpoints failed ...; trying again in 20 s
                                                           a passing Drive hiccup; it reads again by itself
      (the step is still saving: this click is ignored; please wait)
                                                           a second click on the stop button (3.7)
      Note: removed the unfinished temporary file ...      a save that was being written when the machine went away

    After you have sent the line, you may let the machine go (Runtime > Disconnect and delete runtime). What the
    run saved is on your Drive and stays there.


## 6. WHERE THE RESULTS ARE

    My Drive / mesozoic-labs / triceratops60 / <RUN_ID> /
      run_record.json              the settings, software versions, commit, and one entry per session
      checkpoints/                 trike60_<n>_steps.zip and trike60_vecnormalize_<n>_steps.pkl, in pairs
      eval_during_training.jsonl   the short checks of 3.6 (g), one line each, with every drive, slip and step measure
      monitor/session<k>/          one table per simulation with every finished training episode
      eval/eval_<e>.json           the final test (box 6)


## NOT YET SEEN (for me, not for you)

    Every "(expected, not yet seen)" mark, how much a disconnect loses (3.9), and whether a run survives a closed
    browser tab (not tested by this sheet: keep the tab open).
