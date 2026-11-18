# Triceratops 60 cm: the Colab pipeline test (and later runs). The steps.

*** DRAFT of 18 Nov 2026. DO NOT START FROM THIS DRAFT. ***
Every value written <<LIKE THIS>> is filled in when I send you the final sheet. Nothing on this sheet has been run
on Colab yet: the notebook's cells were run on my machine only, where the Colab parts (fetching the code, installing
the packages, Google Drive, releasing the runtime) are skipped. Every line quoted below is printed by the program
itself (I checked the wording against the code and a local run), but lines marked "(expected, not yet seen)" come
from Colab, and I check those at this first run.

Everything here is a simulation on Google's computers. Nothing touches the robot.

You need: a browser (a laptop is easier), your Google account, and this sheet. You do not need to understand the
output. You compare a few lines with the ones written here, and send me the ones I ask for.

Three rules for the whole sheet:
  - If a line on screen differs from the line written here, stop and tell me. Send me the line (a screenshot is
    fine).
  - Never change REPO_REF, RUN_ID, MODEL, TOTAL_STEPS, N_ENVS or SEED of a run because an error message seems to
    suggest it. Never delete anything on your Drive. I tell you when something changes.
  - A run keeps its settings. To train longer or differently later, I give you a NEW RUN_ID; the pipeline test's
    run is never extended (the program refuses a changed TOTAL_STEPS, and the learning rate would jump if forced).


## 1. WHAT RUNS

    The purpose:  a PIPELINE TEST of about one hour (decision D16). It proves that the software installs, that
                  saves reach your Drive, that the stop button saves and that a run resumes where it stopped. It
                  does NOT train a walking policy; nothing is judged on how the robot moves.
    The code:     one fixed commit of the repository,
                    <<C1: THE 40 CHARACTERS OF THE COMMIT>>
                  This is REPO_REF below. The notebook fetches only that commit (no history).
    The notebook: hardware/triceratops_build/rl/trike60_training.ipynb at that commit.
    The machine:  Runtime type "CPU" with "High-RAM" on (more processors). No GPU: it adds nothing here.
    The length:   TOTAL_STEPS = 2000000 simulation steps. On my 4-processor machine training runs at about 800
                  steps per second, so 2 million steps take about 42 minutes; with the installs, the checks during
                  training and the final evaluation, about 50 minutes. Colab's speed is not known yet: the progress
                  lines print it (3.6 (f)).
    Drive:        everything goes into  My Drive / mesozoic-labs / triceratops60 / <RUN_ID> ; about 25 MB for the
                  pipeline test (a save is about 2 MB, one every 250,000 steps, plus one at every stop).


## 2. BEFORE YOU START

    Nothing to upload or install. Keep this sheet open next to the notebook.


## 3. THE PIPELINE TEST

  3.1  Open this address:
         https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<<C1>>/hardware/triceratops_build/rl/trike60_training.ipynb
       Expect: a notebook titled "Mesozoic Labs: Triceratops 60 cm, walking-policy training (simulation only)".
       Opening it runs nothing.

  3.2  Menu File > Save a copy in Drive. Name the copy "trike60 pipeline test". Work in the copy from now on.

  3.3  Menu Runtime > Change runtime type. Choose CPU. Switch High-RAM on. Save.

  3.4  In the first grey box ("1. PARAMETERS") set exactly:
         REPO_REF           <<C1>>
         RUN_ID             trike60-pipe1
         MODEL              trike60_C_closed
         TOTAL_STEPS        2000000
         N_ENVS             8
         SEED               0
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
             Repository commit: <<C1>> (matches REPO_REF: yes; checkout clean: yes)
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
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-pipe1 (new run)

       (e) Under box 5 (each is one line on screen):
             Running: train.py --out /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-pipe1 --total-steps 2000000 --n-envs 8 --seed 0 --model trike60_C_closed --resume --exit-with-parent
             Machine: <V> vCPUs; 8 envs; Python 3.<x>.<y>; torch 2.15.0+cpu
             Starting a new run in /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-pipe1 (0 of 2000000 steps).
           <V> is the number of processors Colab gave (8 expected with High-RAM).       (expected, not yet seen)
           If the Machine line ends "WARNING: more envs than vCPUs, each env runs slower": go on, and tell me <V>.

       (f) Every 100,000 steps a progress line (one line on screen), the first one:
             steps 100000 of 2000000 (5.0%), <R> env steps/s, penalty scale 0.14; training rollouts (stochastic actions), last <k> episodes: return ..., length ... steps, speed ... m/s at command ... m/s
           <R> is the speed. "Stochastic actions" means these numbers come from the training's random exploration;
           they say nothing about how the policy walks. The penalty scale rises from 0.10 to 1.00 over the run.

       (g) Every 250,000 steps two lines: a save, and a short check of the policy:
             Saved checkpoint pair at 250000 steps: trike60_250000_steps.zip and trike60_vecnormalize_250000_steps.pkl
             eval at 250000 steps (deterministic, delay 10 ms): command -> speed m/s: 0.00 -> ..., 0.15 -> ..., 0.25 -> ...; falls ... of 3
           (At 500,000 steps the progress line stands between them.) A "Saved" line means the file is written to
           the Drive folder as the runtime sees it; Drive uploads it to Google in the background, so it can reach
           your Drive a little later.                                                   (expected, not yet seen)

  3.7  THE STOP TEST. Wait until you have seen the "Saved checkpoint pair" line for 500000 steps (two saves; about
       11 minutes at 800 steps per second). Then click the stop button at the left of box 5 (the square in a
       circle while it runs). Expect, in this order:
             Stop requested: the step is asked to stop (train.py first saves a checkpoint pair)...
             Interrupted: saving a checkpoint pair before exiting.
             Session interrupted: <n> steps (target 2000000); <R> env steps/s over <T> s; last pair trike60_<n>_steps.zip
             Flushing Google Drive writes...
             Drive flushed and unmounted; the next run of cell 5 mounts it again.
       then a red "KeyboardInterrupt" under the box. That red text is expected: it is how a stopped box ends.
                                                                                        (expected, not yet seen)
       Write down <n>.

  3.8  THE RESUME TEST. Click the play button of box 5 (only box 5). If a window asks for Drive again, allow it as
       in 3.5.                                                                          (expected, not yet seen)
       Expect:
             Mounted at /content/drive                                                  (expected, not yet seen)
             Running: train.py ... --resume --exit-with-parent
             Machine: <V> vCPUs; 8 envs; ...
             Resuming /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-pipe1 from trike60_<n>_steps.zip (<n> of 2000000 steps).
       <n> must be the number you wrote down in 3.7: the step count carries on. Then the progress lines go on from
       there. If the Resuming line names another number, or says "Starting a new run": stop and tell me.

  3.9  (Optional) THE RECONNECT TEST. After the next "Saved checkpoint pair" line, click Runtime > Disconnect and
       delete runtime. This throws the machine away without a stop. Then Edit > Clear all outputs, then Runtime >
       Run all with every value unchanged, and allow Drive. Box 3 installs again (a fresh machine). Expect:
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-pipe1 (resuming: <k> checkpoint(s) found)
             Resuming /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-pipe1 from trike60_<m>_steps.zip (<m> of 2000000 steps).
       <m> is the newest save that had fully reached Drive: the last "Saved" number, or an earlier one if its upload
       had not finished when the machine went away. Lines before it like
             WARNING: skipping checkpoint trike60_<j>_steps.zip: ...
       mean exactly that and are fine. How much a disconnect loses is what this step measures; it is not known yet.
                                                                                        (expected, not yet seen)

  3.10 THE END. Leave it running. Box 5 ends with one line:
             Session completed: <e> steps (target 2000000); <R> env steps/s over <T> s; last pair trike60_<e>_steps.zip
       <e> is a little above 2000000: training works in blocks of 8 x 1024 steps (2007040 if the run was never
       stopped; another number between 2000000 and 2008184 after a stop).
       If you started box 5 alone (3.8), now click the play button of box 6, and when it ends, of box 7. After a
       Run all (3.9) they run by themselves.

       Box 6 tests the final policy at fixed commands, about 2 minutes. It prints lines like
             delay 10 ms: 5 episodes, forward speed ... m/s (command ...), |error| ... m/s, falls ..., length ... s
             delay  0 ms: 5 episodes, ...
             sweep, delay 10 ms, command 0.00 m/s: 5 episodes, ...
             (and the same for 0.10, 0.20, 0.25, 0.27)
             TARGET 0.25 m/s at 10 ms: 5 episodes, ...
             Wrote /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-pipe1/eval/eval_<e>.json
       Nothing to judge: after a 2-million-step pipeline test the policy is not expected to walk.

       Box 7 prints
             Run trike60-pipe1 is finished; its files are in /content/drive/MyDrive/mesozoic-labs/triceratops60/trike60-pipe1.
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


## 4. A LATER TRAINING RUN

    Only when I send you its values: a new copy of the notebook, a new RUN_ID, and the TOTAL_STEPS, N_ENVS and MODEL
    I give you; then 3.5, 3.6 and 3.10 as above. To resume it after a stop: 3.8. After a disconnect: 3.9.


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
      Anything else in red, except pip's dependency warnings in box 3 and the KeyboardInterrupt after a stop (3.7).

    These are fine and need nothing from you:
      WARNING: skipping checkpoint ...                     a save that had not fully reached Drive (3.9)
      WARNING: no intact checkpoint pair; moved <k> broken file(s) to checkpoints/broken_<time>/ and starting this
               run again from step 0.                       every save was cut short; they are kept, not deleted
      Note: removed the unfinished temporary file ...      a save that was being written when the machine went away

    After you have sent the line, you may let the machine go (Runtime > Disconnect and delete runtime). What the
    run saved is on your Drive and stays there.


## 6. WHERE THE RESULTS ARE

    My Drive / mesozoic-labs / triceratops60 / <RUN_ID> /
      run_record.json              the settings, software versions, commit, and one entry per session
      checkpoints/                 trike60_<n>_steps.zip and trike60_vecnormalize_<n>_steps.pkl, in pairs
      eval_during_training.jsonl   the short checks of 3.6 (g), one line each
      monitor/session<k>/          one table per simulation with every finished training episode
      eval/eval_<e>.json           the final test (box 6)


## STILL OPEN IN THIS DRAFT (for me, not for you)

    <<C1>>, every "(expected, not yet seen)" mark, the speed on Colab, how much a disconnect loses (3.9), and
    whether a run survives a closed browser tab (not tested by this sheet: keep the tab open).
