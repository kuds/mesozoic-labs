# Triceratops 60 cm: training the walking policy on Colab. The steps.

*** DRAFT of 18 Nov 2026. DO NOT START FROM THIS DRAFT. ***
Every value written <<LIKE THIS>> is not known yet. I fill them in and send you the final sheet. Nothing on this
sheet has been run on Colab yet: the notebook's cells were run on my machine only, where the Colab parts (the
download of the code, the package installs, Google Drive, the release of the runtime) are skipped. Lines marked
"(expected, not yet seen)" are what Colab should print; I check them at the first run (section 3).

Everything here is a simulation on Google's computers. Nothing touches the robot.

You need: a browser (a laptop is easier), your Google account, and this sheet. You do not need to understand the
output. You only compare a few lines with the ones written here.

Three rules for the whole sheet:
  - If a line on screen differs from the line written here, stop and tell me. Send me the line (a screenshot is
    fine).
  - Never change REPO_REF or RUN_ID because an error message seems to suggest it. Never delete anything on your
    Drive. I tell you when something changes.
  - There is nothing to send me except the lines I ask for. I read the run's files from your Drive.


## 1. WHAT RUNS

    The code:     one fixed version of the repository, commit
                    <<C1: THE 40 CHARACTERS OF THE COMMIT>>
                  This is REPO_REF in every step below.
    The notebook: hardware/triceratops_build/rl/trike60_training.ipynb at that commit.
    The machine:  Runtime type "CPU". Switch "High-RAM" on if it is offered (more processors, faster). No GPU: this
                  training runs on the processor, and a GPU would cost more and add nothing.
    The length:   TOTAL_STEPS = <<TOTAL_STEPS>> simulation steps, about <<HOURS>> hours on that machine. I fix the
                  number from the speed the first run prints (section 3, line (f)).


## 2. BEFORE THE FIRST RUN

    Nothing to upload and nothing to install: the notebook fetches the code and the software itself, and writes
    everything into one folder on your Drive:
        My Drive / mesozoic-labs / triceratops60 / <RUN_ID>
    It needs about <<DRIVE_MB>> MB there per run (one save is about 2 MB and it saves every 250,000 steps, so
    about 8 MB per million steps: 160 MB for 20 million).


## 3. START A RUN

  3.1  Open this address:
         https://colab.research.google.com/github/kuds/mesozoic-labs/blob/<<C1>>/hardware/triceratops_build/rl/trike60_training.ipynb
       Expect: a notebook titled "Mesozoic Labs: Triceratops 60 cm, walking-policy training (simulation only)".
       Opening it runs nothing.

  3.2  Menu File > Save a copy in Drive. Name the copy "trike60 <RUN_ID>" (for example "trike60 r1"). Work in the
       copy from now on, and use the same copy for every restart of this run (section 5).

  3.3  Menu Runtime > Change runtime type. Choose CPU. Switch High-RAM on if it is there. Save.

  3.4  In the first grey box ("1. PARAMETERS") set exactly:
         REPO_REF           <<C1>>
         RUN_ID             <<RUN_ID, for example trike60-r1>>
         TOTAL_STEPS        <<TOTAL_STEPS>>
         N_ENVS             <<N_ENVS: 8 on a High-RAM machine>>
         SEED               0
         EVAL_EPISODES      10
         USE_GOOGLE_DRIVE   ticked
         AUTO_DISCONNECT    ticked

  3.5  Menu Runtime > Run all.
       Box 3 installs the software first; that takes a few minutes and asks nothing.   (expected, not yet seen)
       Then box 4 opens a window asking whether this notebook may access your Google Drive files. Choose "Connect
       to Google Drive", pick your account, allow. The run waits for your answer.      (expected, not yet seen)
       If you refuse by mistake: nothing was written. Run all again and allow it.

  3.6  Check these lines as they appear, in this order:

       (a) Under box 1:  Parameters of this session:   and your values, one per line.

       (b) Under box 2 (one line on screen):
             Repository commit: <<C1>> (matches REPO_REF: yes; checkout clean: yes)
           Another commit, or a "NO" anywhere in it: stop and tell me.

       (c) Under box 3, two lines:
             Packages: mujoco 3.10.0, numpy 2.5.5, gymnasium 1.4.0, stable-baselines3 2.9.2, torch 2.15.0+cpu, cloudpickle 3.1.2
             All pins hold: yes
           Pip may print warnings in red about other packages ("dependency conflicts"); they do not matter here.
                                                                                       (expected, not yet seen)
           If box 3 stops with "did not install" or "Pins do not hold": stop and send me everything box 3 printed.

       (d) Under box 4:
             Mounted at /content/drive                                                 (expected, not yet seen)
             Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> (new run)
           If it says "(resuming: ...)" on a first start: this RUN_ID was used before. Stop and tell me.

       (e) Under box 5:
             Running: train.py --out /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> --total-steps <<TOTAL_STEPS>> --n-envs <<N_ENVS>> --seed 0
             Starting a new run in /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> (0 of <<TOTAL_STEPS>> steps).

       (f) Within a few minutes, and then about every 100,000 steps, one progress line:
             steps 106496 of <<TOTAL_STEPS>> (0.5%), <<RATE>> env steps/s; last 96 episodes: return ..., length ... steps, final speed ... m/s
           Send me the first two of these lines: the number before "env steps/s" sets the run's length.
           About every 250,000 steps two more lines say a save was written:
             Saving model checkpoint to /content/drive/MyDrive/.../checkpoints/trike60_<n>_steps.zip
             Saving model VecNormalize to /content/drive/MyDrive/.../checkpoints/trike60_vecnormalize_<n>_steps.pkl

  3.7  Leave it running. Keep the tab open: whether the run goes on with the tab closed is not known yet.
                                                                                       (expected, not yet seen)


## 4. THE END OF A RUN

  4.1  Box 5 ends with one line:
         Session completed: <<TOTAL_STEPS>> of <<TOTAL_STEPS>> steps; ... env steps/s over ... s; saved trike60_<n>_steps.zip
       (<n> can be slightly more than TOTAL_STEPS: training works in blocks.)

  4.2  Box 6 tests the policy, about <<EVAL_MIN>> minutes, and prints two lines, for example:
         delay 10 ms: 10 episodes, forward speed 0.xxx m/s (command 0.xxx), |error| 0.xxx m/s, falls k, length xx.x s
         delay  0 ms: 10 episodes, ...
         Wrote /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID>/eval/eval_<n>.json
       Nothing to judge here: I read the file.

  4.3  Box 7 prints
         Run <RUN_ID> is finished; its files are in ...
         Flushing Google Drive writes...
         Disconnecting runtime in 5 seconds...
       and the runtime lets go by itself.                                              (expected, not yet seen)
       Tell me the RUN_ID and that it finished. Do not start it again.


## 5. IF THE RUNTIME DISCONNECTED (RESUME)

  A disconnect loses at most the training since the last save (up to 250,000 steps, a few minutes).

  5.1  Open the same copy of the notebook (My Drive / Colab Notebooks / "trike60 <RUN_ID>").
  5.2  Runtime > Disconnect and delete runtime (if it is offered). Then Edit > Clear all outputs, so an old line is
       not read as a new one. The run's files on your Drive stay as they are.         (expected, not yet seen)
  5.3  Check the first grey box: every value exactly as before. Change nothing.
  5.4  Runtime > Run all. Allow Drive as in 3.5.
  5.5  Expect lines (a) to (c) as before, then:
         Run folder: /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> (resuming: <k> checkpoint(s) found)
         Running: train.py ... --seed 0 --resume
         Resuming /content/drive/MyDrive/mesozoic-labs/triceratops60/<RUN_ID> from trike60_<n>_steps.zip (<n> of <<TOTAL_STEPS>> steps).
       A line before it like
         WARNING: skipping checkpoint trike60_<m>_steps.zip: ...
       is fine: the newest save was cut off by the disconnect, and the one before it is used.
       Then the progress lines go on from <n>, as in 3.6 (f).
       If instead box 5 says
         Run already complete: newest checkpoint is at ... Nothing to do.
       the run had finished: boxes 6 and 7 run as in section 4.
  5.6  Tell me the RUN_ID, the <n> it resumed from, and the time.


## 6. TO STOP A RUN EARLY

  6.1  Click the stop button at the left of box 5 (the square in a circle while it runs).
  6.2  Expect:
         Stop requested: the step is asked to stop (train.py first saves a checkpoint pair)...
         Interrupted: saving a checkpoint pair before exiting.
         Session interrupted: <n> of <<TOTAL_STEPS>> steps; ...; saved trike60_<n>_steps.zip
       and then a red "KeyboardInterrupt" under the box. That red text is expected: it stops "Run all".
                                                                                       (expected, not yet seen)
  6.3  Then Runtime > Disconnect and delete runtime. What was saved is on your Drive; a later Run all resumes it
       (section 5) if I ask you to.


## 7. WHERE THE RESULTS ARE

    My Drive / mesozoic-labs / triceratops60 / <RUN_ID> /
      run_record.json          the settings, software versions, commit and one entry per session (with its speed)
      checkpoints/             trike60_<n>_steps.zip and trike60_vecnormalize_<n>_steps.pkl, in pairs
      monitor/session<k>/      one table per simulation with every finished episode
      eval/eval_<n>.json       the test of the final policy (box 6)

    There is nothing to send me: I read these from your Drive.


## 8. MESSAGES THAT MEAN: STOP AND TELL ME

    For every one of these: do not change REPO_REF or RUN_ID, do not delete anything. Send me the line.

      REFUSED: ... already holds checkpoints ...    box 5 was asked to start a new run where one exists
      REFUSED: --resume found checkpoints but none is intact ...
      REFUSED: the env config differs ...           the run was started with other settings
      Run <RUN_ID> was started from commit ...      the run belongs to another REPO_REF (box 4)
      ... (matches REPO_REF: NO ...) or (checkout clean: NO)
      Pins do not hold: ...   or   ... did not install ...
      RuntimeError: train.py stopped with exit code ...    (the reason is in the lines above it; send both)
      RuntimeError: evaluate.py stopped with exit code ...
      Anything else in red, except pip's dependency warnings in box 3 and the KeyboardInterrupt after a stop (6.2).

    After you have sent the line, you may let the machine go (Runtime > Disconnect and delete runtime). What the
    run saved is on your Drive and stays there.


## STILL OPEN IN THIS DRAFT (for me, not for you)

    <<C1>> <<RUN_ID>> <<TOTAL_STEPS>> <<N_ENVS>> <<HOURS>> <<RATE>> <<DRIVE_MB>> <<EVAL_MIN>>, every
    "(expected, not yet seen)" mark, and whether a run survives a closed tab (3.7).
