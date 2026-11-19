# Training runs (simulation)

Results of Colab runs, read from the owner's Google Drive. Simulation only. None of these runs is judged against
pre-registered rules; they are reported, not verdicts. The controller order stays as decided (D10: model-based walk
first, learned later).

## trike60-pipe1: the pipeline test (D16), 19 Nov 2026

- **Run:** commit 6aa3a679 (clean), model `trike60_C_closed` (revision 3, PR #669's pitch-only ankle), 8 envs, seed 0,
  2,000,000 steps; Colab CPU High-RAM, 8 vCPUs; Python 3.13.18, mujoco 3.10.0, numpy 2.5.5, torch 2.15.0+cpu,
  SB3 2.9.2, gymnasium 1.4.0 (`run_record.json`).
- **Pipeline:** one session, 03:36:18 to 04:15:37 UTC, status `completed` at 2,007,040 steps (about 850 env steps/s
  overall, evaluations included); nine checkpoint pairs (250k ... 2,000,000 and the final 2,007,040), each 2.1 MB +
  6 KB; nothing quarantined. **Not exercised:** the step sheet's stop test (3.7) and resume test (3.8) were not run, so
  resume on Colab is still unproven.
- **Deterministic evaluation during training** (one 20 s episode per command, domain randomisation on, delay 10 ms;
  `eval_during_training.jsonl`), forward speed in m/s:

  | Steps | at 0 | at 0.15 | at 0.25 | Falls |
  |---|---|---|---|---|
  | 250k | 0.002 | 0.021 | 0.046 | 0 |
  | 500k | 0.001 | 0.108 | 0.167 | 0 |
  | 750k | 0.002 | 0.132 | 0.210 | 0 |
  | 1.0M | 0.002 | 0.136 | 0.226 | 0 |
  | 1.25M | -0.002 | 0.142 | 0.237 | 0 |
  | 1.5M | -0.002 | 0.144 | 0.240 | 0 |
  | 1.75M | -0.001 | 0.144 | 0.235 | 0 |
  | 2.0M | 0.001 | 0.145 | 0.237 | 0 |

- **Final sweep** (`eval/eval_2007040.json`; 5 episodes per command, seeds 100000-100004, friction draws 0.61-0.98,
  randomisation on, delay 10 ms, deterministic): mean forward speed 0.003 / 0.099 / 0.191 / **0.239** / 0.257 m/s at
  commands 0 / 0.10 / 0.20 / **0.25** / 0.27 m/s; no falls in 25 episodes, every episode the full 20 s. With commands
  drawn as in training: 0.150 m/s against 0.157 commanded at 10 ms, 0.161 at 0 ms.

**What this does not show:** what the gait looks like or how good it is. Not measured yet: foot slip, which feet are
down when, joint torques and heat against the drives' ratings (the hind knees are at their continuous limit by
estimate, F12), energy, behaviour with the stiffer contact setting, or the roll-ankle model. The policy also observes
the body's true velocity from the simulator (`obs_privileged_lin_vel`), which the robot would have to estimate. An
earlier local probe of the same setup reached similar speeds with heavy sole slip.

**Gait analysis (19 Nov 2026, corrected after the independent review; `analysis/REPORT_GAIT_PIPE1.md`, plan committed
first):** a quick diagonal, trot-like gait at about 2.4 Hz (about 10 cm strides); no falls in 25 episodes including
stiffer contact and the unseen roll-ankle foot; sideways drift up to 0.23 m in 18 s. Heat (RMS torque / continuous
rating, mean of episodes): front pitch joints 1.02 ×, left hind knee 1.10 × (worst episode 1.15 ×), the rest under
1.0; an asymmetric gait. It uses the full torque available (2-10 % of steps at each episode's randomised limit; at the
nominal limit essentially only the left hind knee, 0.9 %). Slip ratio 0.23-0.29; 42 W of joint mechanical power. A working
pipeline and a first gait, not a usable one; reward changes proposed in the report.
