# Triceratops 60 cm: a training run on Colab, with the stop and resume tests. The steps.

*** DRAFT of 19 Nov 2026 for the next run. DO NOT START FROM THIS DRAFT. ***
Every value written <<LIKE THIS>> is filled in when I send you the final sheet: above all the commit (<<C1>>), the
run's name (<<RUN_ID>>) and its length (<<TOTAL_STEPS>>), and the numbers that follow from them. The pipeline test
(trike60-pipe1, 19 Nov 2026) ran this notebook on Colab straight through; its stop test and resume test (3.7, 3.8)
were not done, so this run does them. Every line quoted below is printed by the program itself (I checked the wording
against the code and a local run); lines marked "(expected, not yet seen)" come from Colab and I have not seen them
yet.

Everything here is a simulation on Google's computers. Nothing touches the robot.

You need: a browser (a laptop is easier), your Google account, and this sheet. You do not need to understand the
output. You compare a few lines with the ones written here, and send me the ones I ask for.

Three rules for the whole sheet:
  - If a line on screen differs from the line written here, stop and tell me. Send me the line (a screenshot is
    fine).
  - Never change REPO_REF, RUN_ID, MODEL, TOTAL_STEPS, N_ENVS or SEED of a run because an error message seems to
    suggest it. Never delete anything on your Drive. I tell you when something changes.
  - A run keeps its settings. To train longer or differently later, I give you a NEW RUN_ID; a run is never
    extended (the program refuses a changed TOTAL_STEPS, and the learning rate would jump if forced).


## 1. WHAT RUNS

    The purpose:  <<PURPOSE: ONE LINE>>. On the way it tests the stop button (3.7) and the resume (3.8), which the
                  pipeline test did not do. Nothing is judged on how the robot moves until I have read the results.
    The code:     one fixed commit of the repository,
                    <<C1: THE 40 CHARACTERS OF THE COMMIT>>
                  This is REPO_REF below. The notebook fetches only that commit (no history).
    The notebook: hardware/triceratops_build/rl/trike60_training.ipynb at that commit.
    The machine:  Runtime type "CPU" with "High-RAM" on (more processors). No GPU: it adds nothing here.
    The length:   TOTAL_STEPS = <<TOTAL_STEPS>> simulation steps. The pipeline test ran at about 850 steps per
                  second overall on this runtime (8 processors, the checks during training included), so this run
                  trains for about <<HOURS>>, plus a few minutes for the installs and the final evaluation. (The
                  reward has more to compute since then, so the speed may be somewhat lower.)
                  The progress lines print the speed (3.6 (f)).
    Drive:        everything goes into  My Drive / mesozoic-labs / triceratops60 / <<RUN_ID>> ; about <<DRIVE_MB>> MB
                  (a save is about 2 MB, one every 250,000 steps, plus one at every stop).


## 2. BEFORE YOU START

    Nothing to upload or install. Keep this sheet open next to the notebook.


## 3. THE RUN, WITH THE STOP AND RESUME TESTS

  3.1  Open this address:
         https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<<C1>>/hardware/triceratops_build/rl/trike60_training.ipynb
       Expect: a notebook titled "Mesozoic Labs: Triceratops 60 cm, walking-policy training (simulation only)".
       Opening it runs nothing.

  3.2  Menu File > Save a copy in Drive. Name the copy "trike60 <<RUN_ID>>". Work in the copy from now on, and use
       the same copy for every restart of this run (3.8, 3.9).

  3.3  Menu Runtime > Change runtime type. Choose CPU. Switch High-RAM on. Save.

  3.4  In the first grey box ("1. PARAMETERS") set exactly:
         REPO_REF           <<C1>>
         RUN_ID             <<RUN_ID>>
         MODEL              trike60_C_closed
         TOTAL_STEPS        <<TOTAL_STEPS>>
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
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/<<RUN_ID>> (new run)

       (e) Under box 5 (each is one line on screen):
             Running: train.py --out /content/drive/MyDrive/mesozoic-labs/triceratops60/<<RUN_ID>> --total-steps <<TOTAL_STEPS>> --n-envs 8 --seed 0 --model trike60_C_closed --resume --exit-with-parent <pid>
             Machine: <V> vCPUs; 8 envs; Python 3.<x>.<y>; torch 2.15.0+cpu
             Starting a new run in /content/drive/MyDrive/mesozoic-labs/triceratops60/<<RUN_ID>> (0 of <<TOTAL_STEPS>> steps).
           <pid> is a number (the notebook's own process: train.py stops by itself if that process goes away).
           <V> is the number of processors Colab gave (8 expected with High-RAM).       (expected, not yet seen)
           If the Machine line ends "WARNING: more envs than vCPUs, each env runs slower": go on, and tell me <V>.

       (f) Every 100,000 steps a progress line (one line on screen), the first one:
             steps 100000 of <<TOTAL_STEPS>> (<<P: 100000 AS A PERCENTAGE OF TOTAL_STEPS, ONE DECIMAL>>%), <R> env steps/s, penalty scale 0.44; training rollouts (stochastic actions), last <k> episodes: return ..., length ... steps, speed ... m/s at command ... m/s
           <R> is the speed. "Stochastic actions" means these numbers come from the training's random exploration;
           they say nothing about how the policy walks. The penalty scale (the weight of the energy, slip, heat and
           torque-limit costs) rises from 0.30 to 1.00 over the first 500,000 steps and then stays at 1.00.

       (g) Every 250,000 steps three lines: a save, and a short check of the policy in two lines:
             Saved checkpoint pair at 250000 steps: trike60_250000_steps.zip and trike60_vecnormalize_250000_steps.pkl
             eval at 250000 steps (deterministic, delay 10 ms): command -> speed m/s: 0.00 -> ..., 0.15 -> ..., 0.25 -> ...; falls ... of 3
             eval at 250000 steps, command 0.25 m/s: RMS torque/rating roll ... pitch ... elbow ... knee ... (worst ...); at >=99% nominal limit .../.../.../... %; slip ratio LF/RF/LH/RH .../.../.../...; touchdowns/s .../.../.../...; power ... W; drift ... m, heading ... deg
           The third line is for me (drive heat, torque limits, foot slip, steps); nothing to judge.
           (At 500,000 steps the progress line stands between the save and the check.) A "Saved" line means the file is written to
           the Drive folder as the runtime sees it; Drive uploads it to Google in the background, so it can reach
           your Drive a little later.                                                   (expected, not yet seen)

  3.7  THE STOP TEST. Wait until you have seen the "Saved checkpoint pair" line for 500000 steps (two saves; about
       10 minutes at 850 steps per second). Then click the stop button at the left of box 5 (the square in a
       circle while it runs) ONCE, and wait: do not click again until the line that begins "Session interrupted"
       and the two Drive lines have appeared (a few seconds). A second click changes nothing; it only prints
             (the step is still saving: this click is ignored; please wait)
       Expect, in this order:
             Stop requested: the step is asked to stop (train.py first saves a checkpoint pair). Wait for its last line; further clicks are ignored.
             Interrupted: saving a checkpoint pair before exiting.
             Session interrupted: <n> steps (target <<TOTAL_STEPS>>); <R> env steps/s over <T> s; last pair trike60_<n>_steps.zip
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
             Running: train.py ... --resume --exit-with-parent <pid>
             Machine: <V> vCPUs; 8 envs; ...
             Resuming /content/drive/MyDrive/mesozoic-labs/triceratops60/<<RUN_ID>> from trike60_<n>_steps.zip (<n> of <<TOTAL_STEPS>> steps).
       <n> must be the number you wrote down in 3.7: the step count carries on. Then the progress lines go on from
       there. If the Resuming line names another number, or says "Starting a new run": stop and tell me.

  3.9  (Optional) THE RECONNECT TEST. After the next "Saved checkpoint pair" line, click Runtime > Disconnect and
       delete runtime. This throws the machine away without a stop. Then Edit > Clear all outputs, then Runtime >
       Run all with every value unchanged, and allow Drive. Box 3 installs again (a fresh machine). Expect:
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/<<RUN_ID>> (resuming: <k> checkpoint(s) found)
             Resuming /content/drive/MyDrive/mesozoic-labs/triceratops60/<<RUN_ID>> from trike60_<m>_steps.zip (<m> of <<TOTAL_STEPS>> steps).
       <m> is the newest save that had fully reached Drive: the last "Saved" number, or an earlier one if its upload
       had not finished when the machine went away. Lines before it like
             WARNING: skipping checkpoint trike60_<j>_steps.zip: ...
       mean exactly that and are fine. How much a disconnect loses is what this step measures; it is not known yet.
                                                                                        (expected, not yet seen)

  3.10 THE END. Leave it running. Box 5 ends with one line:
             Session completed: <e> steps (target <<TOTAL_STEPS>>); <R> env steps/s over <T> s; last pair trike60_<e>_steps.zip
       <e> is a little above <<TOTAL_STEPS>>: training works in blocks of 8 x 1024 = 8192 steps. Without a stop it
       is <<E0: TOTAL_STEPS ROUNDED UP TO A MULTIPLE OF 8192>>; after a stop (3.7) it is another number from
       <<TOTAL_STEPS>> to <<TOTAL_STEPS + 8184>>, because the blocks then count from the stop.
       If you started box 5 alone (3.8), now click the play button of box 6, and when it ends, of box 7. After a
       Run all (3.9) they run by themselves.

       Box 6 tests the final policy at fixed commands, about 2 minutes. It prints lines like
             delay 10 ms: 5 episodes, forward speed ... m/s (command ...), |error| ... m/s, falls ..., length ... s
             delay  0 ms: 5 episodes, ...
             sweep, delay 10 ms, command 0.00 m/s: 5 episodes, ...
             (and the same for 0.10, 0.20, 0.25, 0.27)
             TARGET 0.25 m/s at 10 ms: 5 episodes, ...
             TARGET 0.25 m/s at 10 ms, gait: RMS torque/rating roll ... pitch ... elbow ... knee ... (worst ...); at >=99% nominal limit ...; slip ratio ...; touchdowns/s ...; power ... W; drift ... m, heading ... deg
             Wrote /content/drive/MyDrive/mesozoic-labs/triceratops60/<<RUN_ID>>/eval/eval_<e>.json
       Nothing to judge here: I read the file.

       Box 7 prints
             Run <<RUN_ID>> is finished; its files are in /content/drive/MyDrive/mesozoic-labs/triceratops60/<<RUN_ID>>.
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


## STILL OPEN IN THIS DRAFT (for me, not for you)

    <<C1>> <<RUN_ID>> <<TOTAL_STEPS>> <<PURPOSE>> <<HOURS>> <<DRIVE_MB>> <<P>> <<E0>> <<TOTAL_STEPS + 8184>>, every
    "(expected, not yet seen)" mark (the pipeline test's own screen lines are not in my hands yet), how much a
    disconnect loses (3.9), and whether a run survives a closed browser tab (not tested by this sheet: keep the tab
    open).
