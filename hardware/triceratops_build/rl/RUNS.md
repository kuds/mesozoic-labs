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

## trike60-run2b-ankle: the roll-ankle run with the new reward (D17), 19 Nov 2026

- **Run:** commit e92fd9ef (clean), model `trike60_C_closed_ankleroll` (revision 3), 8 envs, seed 0, 8,000,000 steps;
  Colab CPU High-RAM, 8 vCPUs; the same package versions as pipe1. Reward: heat (w 0.3) and torque-limit (w 2.0)
  penalties and the negative part of the air-time term on the late schedule (0 until 1 M steps, full at 3 M); energy
  and slip ramped 0.1 to 1.0 over 2 M; phase clock off.
- **Pipeline:** one session, 16:26:40 to 19:39:31 UTC, status `completed` at 8,003,584 steps (692 env steps/s); 33
  checkpoint pairs; nothing quarantined; the last pair finite. A first attempt, `trike60-run2-ankle`, was started with
  the default 2,000,000 steps and stopped by the owner at 16:24:59: it ended `interrupted` at 81,952 steps with an
  intact, finite pair saved and recorded, the first stop on real Colab. Resume on Colab is still untested.
- **Deterministic evaluation during training** (one 20 s episode per command, randomisation on, delay 10 ms; 32
  evaluations, no falls in any); at the 0.25 m/s command, worst RMS torque / continuous rating per group:

  | Steps | Speed at 0 / 0.15 / 0.25 m/s | Pitch | Front elbow | Hind knee | Worst slip ratio | Joint mech. power |
  |---|---|---|---|---|---|---|
  | 0.5 M | 0.000 / 0.117 / 0.208 | 1.00 | 1.00 | 1.00 | 0.45 | 31 W |
  | 1 M | -0.004 / 0.140 / 0.232 | 1.08 | 1.01 | 0.98 | 0.39 | 45 W |
  | 2 M | -0.001 / 0.143 / 0.238 | 0.82 | 0.86 | 0.96 | 0.20 | 35 W |
  | 3 M | 0.001 / 0.142 / 0.242 | 0.77 | 0.77 | 0.90 | 0.14 | 34 W |
  | 5 M | 0.000 / 0.149 / 0.241 | 0.69 | 0.68 | 0.81 | 0.10 | 27 W |
  | 8 M | 0.000 / 0.148 / 0.243 | 0.66 | 0.61 | 0.70 | 0.08 | 22 W |

- **Final sweep** (`eval/eval_8003584.json`; 5 episodes per command, seeds 100000-100004, randomisation on, delay
  10 ms, deterministic): 0.001 / 0.098 / 0.195 / **0.244** / 0.263 m/s at 0 / 0.10 / 0.20 / **0.25** / 0.27 m/s; no
  falls. At 0.25 m/s: worst RMS / rating left hind knee 0.76, right fore pitch 0.73, left fore elbow 0.64 (all under
  1.0); left hind knee at its nominal limit 0.4 % of steps in the worst episode; slip ratio 0.05-0.09; 22 W of joint
  mechanical power; sideways drift up to 0.26 m in 18 s.

Against pipe1 (PR #669's foot, the first reward): about the same speed, but the drives run cooler (worst joint 0.76 ×
against 1.10 ×), the feet slip about a quarter as much, and the joints do half the mechanical work. Simulation, one
policy each: reported, not a verdict. The full analysis follows `analysis/PLAN_RUN2B.md`.

**Analysis (19 Nov 2026, revised after the independent review; `analysis/REPORT_RUN2B.md`, plan committed first):** a
clean trot at 1.22 Hz with 20 cm strides; every joint under its continuous rating in every gait-report case (worst
0.72 × as a mean, left hind knee; 0.86 × in one episode on the unseen foot); time at the nominal torque limit about 1 %
on several drives, which one depending on the condition (pipe1 3.7-22.5 %); leg electrical estimate 40-50 W against
pipe1's 106-123 W. At 0.25 m/s it breaks at the same robustness points as pipe1 but slows more on a 3 cm step (-54 %)
and an 8° ramp (-34 %), 7 of 8 episodes fall from a 1.25 m/s push to the left, and on steps, ramps and slippery floors
it drifts sideways much more than pipe1 (about 1.1 m in 18 s on 2-3 cm steps, 0.7-0.9 m on 5-8° ramps, 0.3-0.8 m at friction
0.3-0.15). Reward, foot, training length and schedules all differ from pipe1, so no single change gets the credit.
Proposals for the next run in the report.
