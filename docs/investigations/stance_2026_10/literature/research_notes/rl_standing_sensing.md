# RL for legged standing and balance: observation design, symmetry, robustness, reward/checkpoint pitfalls, musculoskeletal and animal RL

Tags: [HW] = hardware result; [SIM] = simulation only; [PREPRINT] = arXiv only / not peer reviewed as far as found. Most papers below were read in full text (PDF downloaded from arXiv or Europe PMC); quotes and numbers are from that text unless marked otherwise.

## 1. Foot contact / force / CoP sensing in observations

### Takeaway
State-of-the-art hardware locomotion policies mostly do NOT put foot force or CoP in the deployed actor's observation (ANYmal, Cassie, Digit work is proprioception plus history). Contact states and forces are instead given to a privileged teacher, to an asymmetric critic, or to the reward. Where foot contact flags are cheap and reliable they are used directly: RMA on Unitree A1 feeds binary foot contacts. Ablations that isolate contact inputs are few and show small or mixed effects. A learned contact-probability estimate barely changed final reward but stabilised and sped up learning (Ji 2022). The only clean with/without test of contact versus force inputs found is a weak simulation preprint with a linear policy.

### Cited Findings
**Privileged teacher-student (quadruped)**
- Lee et al. 2020 (ANYmal-B/C, Science Robotics 5(47) eabc5986) [HW]: the deployed controller "does not use any exteroceptive input (e.g., no haptic sensors...)" and "does not contain any handcrafted features such as foot contact states". The teacher's privileged vector x_t "mainly consists of information related to foot-ground interactions such as terrain profile, foot contact states and forces, friction coefficients, and external disturbance forces". The student is a temporal convolutional network (TCN) over proprioceptive history and learns by imitating the teacher — [arXiv 2010.11251](https://arxiv.org/abs/2010.11251); [DOI](https://doi.org/10.1126/scirobotics.abc5986)
- Lee et al. 2020 ablations (5 seeds, 95% CIs) [SIM]:
  - A TCN-20 student trained directly with RL, without the privileged teacher, "fails to learn to balance and locomote" and cannot handle a slope or step.
  - Under a 50 N lateral push for 5 s, the lateral deviation of the TCN-100 controller (long history) was 35.5% lower than that of TCN-1.
  - A decoder can reconstruct foot contact states and terrain from the student's proprioceptive history embedding.
  - Source: [arXiv 2010.11251](https://arxiv.org/abs/2010.11251)
- Lee et al. 2020 on hardware sensors: conventional systems "employ contact sensors at the feet, which can become unreliable in field conditions" — [arXiv 2010.11251](https://arxiv.org/abs/2010.11251)
- Kumar et al. 2021 RMA (Unitree A1, RSS 2021) [HW]:
  - The base-policy state is 30-D: 12 joint positions, 12 joint velocities, torso roll and pitch, and "binary foot contact indicators (4 values)" "from the foot sensors".
  - The privileged environment vector e_t (17-D) is mass and its position (3), motor strength (12), friction (1) and local terrain height (1), encoded to an 8-D extrinsics vector.
  - The reward includes a "Ground Impact" penalty ||f_t − f_{t−1}||² on foot ground reaction forces (GRFs).
  - Training terminates if pitch exceeds 0.2 rad or roll exceeds 0.4 rad.
  - Source: [arXiv 2107.04034](https://arxiv.org/abs/2107.04034)
- Ji et al. 2022 (MIT Mini Cheetah, IEEE RA-L) [SIM ablation, HW deployment]: an estimator network trained concurrently with the policy predicts base linear velocity, foot height and contact probability, which are fed to the policy. Final total reward in the ablation (Table III):

  | Variant | Total reward |
  |---|---|
  | Concurrent (full) | 4.7212 |
  | w/o contact estimator | 4.7125 |
  | w/o linear-velocity estimator | 4.5555 |
  | w/o estimator | 4.4577 |
  | Implicit estimator | 4.439 |

  The authors: "Foot contact probability estimation makes the least impact on the final performance, but it stabilizes and accelerates learning processes". Linear velocity matters most. They also found a sequentially trained estimator degrades because "the policy trained with a state estimator tends to avoid states where the state estimator becomes unreliable" — [arXiv 2202.05481](https://arxiv.org/abs/2202.05481)

**Bipeds: Cassie and Digit**
- Siekmann et al. 2021, "Blind bipedal stair traversal" (Cassie, RSS 2021) [HW]:
  - The policy state is pelvis orientation (quaternion), pelvis angular velocity, joint positions and velocities, operator commands and two clock inputs. There is no foot contact or force input.
  - Foot forces and foot velocities appear only in the periodic reward, which carries 0.140 weight terms of the form 1 − E[I_force(φ)]·exp(−0.01‖F‖).
  - A mirror loss is used.
  - LSTM policies far outperform feedforward ones on stairs, suggesting memory matters.
  - Hardware: 80% success ascending and 100% descending, 10 trials each.
  - Source: [arXiv 2105.08328](https://arxiv.org/abs/2105.08328)
- Siekmann et al. 2021, periodic reward composition (ICRA 2021): reward terms use "the magnitudes of foot forces and velocities" by gait phase. During swing they penalise foot forces; during stance they penalise foot velocities — [arXiv 2011.01387](https://arxiv.org/abs/2011.01387)
- Li et al. 2024 (Cassie, IJRR) [HW]:
  - Observable states o ∈ R^26: motor positions and velocities, base orientation from the IMU, and base linear velocity from an extended Kalman filter (EKF). No foot force.
  - The policy also gets a short I/O history (4 steps) and a long I/O history (66 steps, about 2 s).
  - Privileged quantities (global height, foot height, terrain height) appear in the reward "but are excluded from the actor's observation".
  - Source: [arXiv 2401.16889](https://arxiv.org/abs/2401.16889)
- Cassie has no dedicated foot force sensors: "The robot is equipped with a spring on each leg, which serves as a contact detection sensor". Lin et al. (CoRL 2021) trained a proprioceptive contact estimator and note that "some robots do not have dedicated contact sensors, and the addition of such sensors is non-trivial without redesigning the hardware" — [arXiv 2106.15713](https://arxiv.org/abs/2106.15713)
- Radosavovic et al. 2024 (Digit humanoid, Science Robotics) [HW]:
  - The deployed controller is a causal transformer over "the history of proprioceptive observations and actions". The authors hypothesise the history lets it infer "gait and contact states".
  - A teacher "state policy" with privileged information supervises the student through a KL term, combined with RL.
  - Ablations (sim, 95% CIs): the transformer beats MLP, TCN and LSTM; longer context helps; joint imitation+RL beats either alone.
  - The four toe motors are not policy-controlled; they are held at default positions with fixed PD gains.
  - The exact observation list is in a supplement I did not see.
  - Source: [arXiv 2303.03381](https://arxiv.org/abs/2303.03381)

**Humanoid frameworks**
- Humanoid-Gym [PREPRINT] [HW, zero-shot]:
  - Actor observation: clock, commands, joint position and velocity, base angular velocity, Euler angles, last actions.
  - Critic only: friction, body mass, base linear velocity, push force and torque, tracking difference, periodic stance mask and "Feet Contact detection" (asymmetric actor-critic).
  - Source: [arXiv 2404.05695](https://arxiv.org/abs/2404.05695)
- DBHL, Xie et al. 2025 (Unitree H1-2) [PREPRINT] [HW + SIM ablation]:
  - Actor: 4-step proprioceptive history only.
  - Critic: privileged information including base height, linear velocity and "feet contact indicator".
  - The ZMP reward is computed from privileged information while the policy runs on proprioception.
  - Source: [arXiv 2502.17219](https://arxiv.org/abs/2502.17219)
- Rudin et al. 2021 (ANYmal, legged_gym, CoRL 2021) [HW]:
  - Observation: base linear and angular velocity, gravity vector, joint positions and velocities, previous actions, plus 108 terrain height samples. No foot contact.
  - Contacts enter the reward (collision penalties and a step-length term).
  - Source: [arXiv 2109.11978](https://arxiv.org/abs/2109.11978)
- Tan et al. 2018 (Minitaur, RSS 2018) [HW]: "a compact observation space helps to transfer". They excluded drifting IMU yaw and noisy motor velocities: "If the observation space is high dimensional, the learned policy can easily overfit the simulated environment" — [arXiv 1804.10332](https://arxiv.org/abs/1804.10332)

**Direct with/without-contact ablations**
- Mhaske & Mandhare 2025 [PREPRINT] [SIM, PyBullet; linear policy trained with Augmented Random Search (ARS) on a small quadruped]:
  - Observation variants: IMU, IMU+contacts and IMU+force.
  - Episodes reaching more than 90 m alive: IMU 348, IMU+Force 410, IMU+Contacts 490.
  - On a slope test (50 episodes), IMU+Contact had a 100% success rate.
  - IMU+Force policies were "slower ... but less likely to fall over".
  - Weak evidence: linear policy, small study, simulation only.
  - Source: [arXiv 2503.01102](https://arxiv.org/abs/2503.01102)
- Ferigo et al. 2021 iCub push-recovery [SIM]: the observation explicitly includes foot-contact booleans and "the vertical forces applied to the local CoP of the feet" (62-D observation). It has no with/without ablation — [arXiv 2104.14534](https://arxiv.org/abs/2104.14534)
- Yang, Komura & Li 2017 (Valkyrie-like model, Humanoids 2017) [SIM]: state features include "ground contact information", pelvis height, joint angles and velocities, torso and pelvis orientation and angular velocity, and link CoM displacements — [arXiv 1809.02074](https://arxiv.org/abs/1809.02074)

**What real robots expose**
- Unitree Go1 datasheet: four foot force sensors, EDU version only, stated resolution 5 g — [Go1 datasheet (vendor-hosted)](https://www.generationrobots.com/media/unitree/Go1%20Datasheet_EN%20v3.0.pdf)
- Unitree Go2 brochure lists a foot force sensor, with no numeric spec — [Go2 introduction PDF](https://w-technika.cz/out/media/Go2%20Introductionv1.2-English%202023.7.19.pdf)
- A1 has foot sensors used as binary contacts — [RMA](https://arxiv.org/abs/2107.04034)
- ANYmal policies above use no foot sensors — [Lee 2020](https://arxiv.org/abs/2010.11251); [Rudin 2021](https://arxiv.org/abs/2109.11978)

### Inferences
- The project's observation (joint angles and velocities plus torso orientation, without foot force or CoP) matches the "blind proprioceptive" style of Cassie, Digit and ANYmal work. Those works rely on (a) history or memory (LSTM, TCN, transformer, I/O history) and (b) privileged contact information in a teacher, critic or reward. A stock memoryless SB3 PPO MLP has neither, so it cannot infer where the weight sits on the foot except through joint-angle proxies. This is consistent with the lean costing nothing until a fall.
- Since mesozoic-labs is simulation-first, the low-risk literature pattern is to give foot contact, normal forces and CoP to the critic (asymmetric actor-critic, as in Humanoid-Gym and DBHL) and/or the reward. This improves value estimates and pays for weight placement without creating a sensor dependency for a later robot. Putting CoP or force in the actor is defensible in simulation; robots with foot force sensors exist (Go1 EDU, Go2, A1), but hardware evidence says such sensors are noisy or unreliable in the field.
- The Ji 2022 result suggests contact-probability inputs mainly help learning stability, not final performance. That is relevant to a "one seed in three" problem, but it is a quadruped-locomotion result, not standing.

### Gaps
- I found no peer-reviewed, hardware-validated ablation of foot force or CoP as a policy input for standing or balance specifically. The only clean with/without contact study is a sim-only linear-policy preprint.
- The Radosavovic 2024 actor observation list is in the supplementary material, which I did not access.
- Official foot-sensor specs for Unitree H1/G1, Digit and Atlas were not found.

## 2. Symmetry (mirror loss, augmentation, limits)

### Takeaway
Mirror-symmetry methods reliably improve left-right symmetry and motion quality, but not reliably learning speed. The loss variant competes with the RL gradient, and data augmentation often trains better. Symmetric policies have a known "neutral pose" failure: they cannot break symmetry from a symmetric state, and a standing pose is one. Mirror symmetry constrains only left-right structure. It says nothing about fore-aft (sagittal) posture.

### Cited Findings
- Yu, Turk & Liu 2018 (ACM TOG 37(4):144, SIGGRAPH) [SIM]:
  - Mirror symmetry loss L_sym = Σ‖π_θ(s_i) − Ψ_a(π_θ(Ψ_o(s_i)))‖² on the mean action, with weight w = 4, over 20,000 samples per iteration.
  - It is placed in the policy loss, not the reward, because "adding L_sym in the reward function would break the assumption in the Policy Gradient Theorem".
  - Symmetry is measured on actions, not states, so an asymmetric task can still be solved asymmetrically (walking with a 10 kg load in the right hand).
  - Without the mirror loss, "the resulting policy usually exhibits asymmetric gaits and the training process is notably slower".
  - Without curriculum, policies "learn to either fall slowly or stand still".
  - Source: [arXiv 1801.08093](https://arxiv.org/abs/1801.08093)
- Abdolhosseini, Ling, Xie, Peng & van de Panne 2019 (MIG 2019) [SIM], reading Yu et al.'s plots: the symmetric loss "is shown to be beneficial only in the context of a given curriculum learning algorithm; in its absence, there was no significant improvement over a vanilla-PPO baseline, and in one case (the humanoid) using the symmetric loss proved to be detrimental" — [PDF](https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/2019-MIG-symmetry.pdf); DOI 10.1145/3359566.3360070
- Abdolhosseini et al. 2019 compare four methods: DUP (mirrored duplicate tuples), LOSS (auxiliary loss), PHASE (phase-based mirroring) and NET (symmetric network). Findings:
  - "There is not a single best mirroring method"; LOSS and PHASE are "the most consistent".
  - PHASE is best for imitation-guided tasks (Cassie, DeepMimic).
  - "Symmetry enforcement methods have no consistent and predictable impact" on learning speed; use them "for producing higher quality symmetric motions ... but not necessarily for faster learning".
  - LOSS is "not very sensitive to the choice of w".
  - Source: [PDF](https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/2019-MIG-symmetry.pdf)
- Abdolhosseini et al. 2019, the neutral-pose limitation: "a symmetric policy is incapable of escaping from a neutral pose since the action that it takes would also be symmetric", and "states near the neutral states can also become problematic". It is worst for NET; DUP and LOSS "can suffer from the same issue". Workaround: random noise on each joint of the initial pose. Characters sometimes converged on a hopping gait — [PDF](https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/2019-MIG-symmetry.pdf)
- Mittal, Rudin, Klemm, Allshire & Hutter 2024 (ICRA 2024) [SIM + HW on ANYmal-D]:
  - "PPO with symmetry augmentation obtains the highest return and fastest convergence while having a low symmetry loss".
  - The symmetry loss "comes at the cost of performance and slower convergence"; increasing w "hinders learning as the gradients ... compete".
  - The loss "will push the policy to adopt equivalent actions for all equivalent states without considering the respective rewards", whereas augmentation pushes "towards the best-performing actions".
  - Using both together "does not necessarily improve the performance or convergence".
  - Augmentation assumes a near-symmetric initial policy: large initial weights give lower performance and higher asymmetry, and a small loss coefficient restores symmetry.
  - "Once an asymmetry in a behavior arises, it tends to get magnified with further training".
  - Strict equivariant networks are problematic because "a symmetric policy cannot lift the right front foot to take the first step" from a symmetric stance.
  - Source: [arXiv 2403.04359](https://arxiv.org/abs/2403.04359)
- Symmetry is used in practice on hardware bipeds:
  - Siekmann 2021 uses a mirror loss "to ensure that the control policy does not learn asymmetric gaits" — [arXiv 2105.08328](https://arxiv.org/abs/2105.08328)
  - Xie et al. 2019 (Cassie) "transform the input and output every half walking cycle to their symmetric form" (phase mirroring) — [arXiv 1903.09537](https://arxiv.org/abs/1903.09537)
- Learning to Run 2017 solutions (musculoskeletal human, DDPG) [SIM]: mirrored states and actions from replay doubled the data and "improves stability of learned policy". Without it, "muscles for only one leg are active and other leg just follows the first leg" — [arXiv 1804.00361](https://arxiv.org/abs/1804.00361)
- DEP-RL (ICLR 2023) [SIM]: exploration type changes whether a symmetric gait emerges. In human-run, 5 of 5 DEP-MPO seeds reached an alternating gait with symmetric leg extensions, against 1 of 5 for MPO — [arXiv 2206.00484](https://arxiv.org/abs/2206.00484)

### Inferences
- The left-right mirror operator maps a forward lean onto itself: a sagittal-plane posture is a fixed point of left-right reflection. A mirror KL or loss term therefore gives no gradient against a symmetric fore-aft lean. This explains why mirror training fixed lopsidedness but not the toe-lean, and it agrees with Mittal's point that the loss pushes equivalent actions "without considering the respective rewards".
- Quiet bipedal standing is close to the "neutral pose" both Abdolhosseini and Mittal flag. A strongly symmetric standing policy may struggle to produce asymmetric corrective actions, such as a single step. Randomised initial joint noise is the documented mitigation.
- If the project keeps a mirror term, the literature favours mirrored data augmentation (DUP-style) over a large-weight loss, or a small-weight loss.

### Gaps
- No paper found studies symmetry methods for quiet standing or balance specifically; all evaluations are locomotion or manipulation.
- No source found that tests whether symmetry methods change sagittal-plane posture.

## 3. Robustness: action/joint noise, domain randomisation, push recovery, exploration noise and posture

### Takeaway
The policy's own exploration noise is a robustness mechanism. Fixed or entropy-driven noise during training produces policies that tolerate disturbances, while deterministic deployment of a policy whose std has shrunk can be brittle. Action-robust formulations (Tessler 2019) and MaxEnt RL (Eysenbach & Levine 2022) formalise this. Domain randomisation buys robustness but can make policies overly conservative. Standing-only policies, even push-trained ones, fall once pushed past the support region. Mixing in stepping or walking skills gives recovery.

### Cited Findings
- Xie et al. 2019 (Cassie; arXiv preprint, iterative design paper) [HW]:
  - "Instead of making the covariance ... a learnable parameter, we use a fixed covariance ... standard deviation of exp(−2) ≈ 0.1 radians. A benefit of the fixed covariance is that because of the noise constantly injected into the system during training, the resulting policy will adapt itself to handle unmodeled disturbances during testing."
  - Robustness was evaluated "by injecting noise of varying magnitude to the policy actions, increasing the mass of the pelvis by 20%, and applying pushes of 50N".
  - Larger networks (256, 256) learned faster and were more robust than (64, 64) or (32, 32).
  - Source: [arXiv 1903.09537](https://arxiv.org/abs/1903.09537)
- Tessler, Efroni & Mannor 2019 (ICML 2019) [SIM, MuJoCo]:
  - Two action-robust criteria: PR-MDP (with probability α an adversarial action replaces the agent's) and NR-MDP (an adversary adds a perturbation to the action).
  - These "are related to common forms of uncertainty in robotics domains, such as the occurrence of abrupt forces".
  - The method "not only does ... produce robust policies, but it also improves the performance in the absence of perturbations", acting as implicit regularisation.
  - Source: [arXiv 1901.09184](https://arxiv.org/abs/1901.09184)
- Eysenbach & Levine 2022 (ICLR 2022) [theory + SIM]: "maximum entropy (MaxEnt) RL maximizes a lower bound on a robust RL objective, and thus can be used to learn policies that are robust to some disturbances in the dynamics and the reward function" — [arXiv 2103.06257](https://arxiv.org/abs/2103.06257)
- Andrychowicz et al. 2021, "What matters in on-policy RL" (ICLR 2021) [SIM]:
  - "The initial policy appears to have a surprisingly high impact".
  - Performance is "very sensitive to the initial action standard deviation with 0.5 performing best on all environments except Hopper".
  - Recommendation: last policy layer 100× smaller, softplus std with a negative offset.
  - Per-state versus global std showed "no clear benefits"; the minimum std "seems to matter little, if it is not set too large".
  - Their evaluation used "the stochastic policy".
  - Source: [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
- Stable-Baselines3 defaults (code):
  - PPO's ActorCriticPolicy has `log_std_init: float = 0.0`, an initial std of 1.0 in action units — [SB3 policies.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/policies.py)
  - PPO supports gSDE via `use_sde` and `sde_sample_freq` — [SB3 ppo.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py)
  - `evaluate_policy` and `EvalCallback` default to `deterministic=True` — [SB3 evaluation.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/evaluation.py); [SB3 callbacks.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py)
- Montenegro et al. 2024 (ICML 2024) [theory]: "In common practice, stochastic (hyper)policies are learned only to deploy their deterministic version". They show "how to tune the exploration level used for learning to optimize the trade-off between the sample complexity and the performance of the deployed deterministic policy" — [arXiv 2405.02235](https://arxiv.org/abs/2405.02235)
- Raffin, Kober & Stulp, gSDE (CoRL 2021) [SIM + HW on a tendon-driven arm, a quadruped and an RC car]: step-wise independent Gaussian noise "leads to shaky behavior" and "jerky motion patterns can damage the motors". gSDE gives smooth exploration whose sampling interval trades smoothness against performance — [arXiv 2005.05719](https://arxiv.org/abs/2005.05719)
- CAPS, Mysore et al. 2021 (ICRA 2021) [HW quadrotor]: temporal and spatial smoothness regularisation of the policy removed high-frequency control components, with "an almost 80% reduction in power consumption" — [arXiv 2012.06644](https://arxiv.org/abs/2012.06644)
- Peng & van de Panne 2017 (arXiv 1611.01055; ICLR 2017 submission, later published at SCA 2017 — venue not verified here) [SIM; 2D biped, dog and a 19-link "raptor"]: under random trunk perturbation forces, "PD policies are among the most robust for all the motions". Torque policies were least robust for the biped walk, and muscle-tendon unit (MTU) policies least robust for the raptor run — [arXiv 1611.01055](https://arxiv.org/abs/1611.01055)
- Xie et al. 2021, "Dynamics randomization revisited" (Laikago, ICRA 2021) [SIM + HW]:
  - "Unnecessary randomization of parameters can produce conservative policies with limited actual gains in robustness". Trotting top speed was 0.9 m/s with dynamics randomisation and 1.1 m/s without, "both in simulation and on the physical robot".
  - Randomising only the parameter that matters works: training with latency up to 20 ms raised the tolerated latency from 17 ms to 32 ms.
  - Source: [arXiv 2011.02404](https://arxiv.org/abs/2011.02404)
- Tan et al. 2018 [HW]: "dynamics randomization trades optimality for robustness", so ranges were chosen "to prevent learning overly conservative running gaits". Random perturbations push the robot "out of balance so that it needs to learn how to recover" — [arXiv 1804.10332](https://arxiv.org/abs/1804.10332)
- Rudin et al. 2021 [HW]:
  - Robots are randomly pushed every 10 s, with the base accelerated by up to ±1 m/s, "to teach them a more stable stance".
  - Observation noise is based on measured robot data; friction is drawn from U[0.5, 1.25].
  - The critic is bootstrapped on timeouts, which "is ignored by popular implementations such as Stable-Baselines". Current SB3 on-policy rollouts do bootstrap when the env reports `TimeLimit.truncated` ([on_policy_algorithm.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/on_policy_algorithm.py)).
  - Source: [arXiv 2109.11978](https://arxiv.org/abs/2109.11978)
- Li et al. 2024 (Cassie, real-world standing case study) [HW]:
  - "When policies are exclusively trained for the standing skill, regardless of whether they were trained with external perturbations, the robot keeps losing its balance if it is forced to lean beyond its support region."
  - A versatile policy trained on walking plus standing "transits to a walking gait by breaking contact, executes several steps ... and then smoothly reverts to a standing pose". This happened even though perturbations were not simulated when training that standing skill.
  - Interpretation: task randomisation widens the training distribution more than dynamics randomisation.
  - Source: [arXiv 2401.16889](https://arxiv.org/abs/2401.16889)
- Yang, Komura & Li 2017 [SIM]: a hierarchical deep RL framework "learns a diversity of policies for humanoid balance control" and produced "human-like balancing behaviors such as active push-off of ankles" (foot tilting), unlike ZMP-based controllers. The reward was designed from capture-point and inverted-pendulum ideas — [arXiv 1809.02074](https://arxiv.org/abs/1809.02074)
- Ferigo et al. 2021 (iCub) [SIM]: whole-body push-recovery strategies emerged, and the policy "can withstand repeated applications of strong external pushes". The authors list as a limitation that "relying only on state space exploration ... requires a carefully designed reward function" — [arXiv 2104.14534](https://arxiv.org/abs/2104.14534)
- Learning to Run 2017 solutions [SIM]: "explicit penalization of falling over ... created an ultra-cautious agent, who always concluded the whole episode but at the cost of a big drop of the average speed" — [arXiv 1804.00361](https://arxiv.org/abs/1804.00361)
- HoST (Unitree G1 standing-up) [PREPRINT] [HW]: deployment needed "smoothness regularization and implicit motion speed bound to alleviate oscillatory and violent motions" — [arXiv 2502.08378](https://arxiv.org/abs/2502.08378)

### Inferences
- The observed failure of a joint-space action-noise test fits the Xie 2019 and Tessler picture. If PPO's learnable log-std shrinks late in training, the policy learns to stand in a way that needs precise actions. That includes balancing with the CoP at the front edge of the foot, where small action errors tip it over. Deterministic evaluation hides this, because SB3 evaluates with `deterministic=True` by default. Remedies the literature actually used:
  - a fixed or floor-clamped action std, around 0.1 rad in joint-target units (Xie 2019);
  - training with explicit action perturbations of the test magnitude (NR-MDP/PR-MDP style);
  - gSDE for temporally smooth exploration;
  - random pushes;
  - mixing in stepping tasks so falls past the support edge are recoverable.
- The SB3 default initial std of 1.0 is well above the ~0.5 Andrychowicz found best. The std schedule over training plausibly shapes posture: high early noise favours conservative, wide-margin stances, and collapsing noise later allows drift to marginal postures. This is an inference; no paper found shows the toe-lean mechanism directly.
- Dynamics randomisation is not free (Xie 2021). Randomise the parameters that plausibly differ (foot friction, foot geometry, mass distribution) rather than everything.

### Gaps
- No paper found that directly measures how the training action-noise level or log-std schedule moves a standing biped's CoP fore-aft or toward the support-polygon edge.
- I did not verify "Pink noise is all you need" (Eberhard et al., ICLR 2023) or other noise-colour studies; they are not included.

## 4. Reward pitfalls in standing; CoP/CoM/ZMP reward terms; checkpoint selection and seed variance

### Takeaway
Benchmark "stand" rewards (DeepMind Control Suite) pay for head height, uprightness, small control and low CoM velocity. They pay nothing for where the weight sits within the support polygon, so a lean is free until it causes a fall — exactly the project's gap. Balance-focused RL work adds explicit terms:
- CoM horizontal position at the foot centre (Yang 2017);
- CoM projection inside a shrunken support polygon plus each foot's CoP at the sole centre (Ferigo 2021);
- ZMP pulled toward the support-polygon centre (DBHL 2025).

Ferigo et al. saw foot-tipping (feet not flat) emerge and had to add a foot-orientation term. That is the closest literature analogue to the T. rex toe-lean. On evaluation:
- seed variance is large (Henderson);
- "max over training" protocols inflate results (Agarwal);
- picking the best of many noisy estimates has maximisation bias (Patterson);
- SB3's EvalCallback best-model logic does exactly this with deterministic rollouts.

### Cited Findings
**Benchmark standing rewards**
- DeepMind Control Suite humanoid `get_reward` [SIM]:
  - `standing = tolerance(head_height, bounds=(_STAND_HEIGHT, inf), margin=_STAND_HEIGHT/4)`
  - `upright = tolerance(torso_upright, bounds=(0.9, inf), ...)`
  - `small_control = (4 + small_control)/5`
  - for stand, `dont_move = tolerance(horizontal CoM velocity, margin=2)`
  - reward = `small_control * stand_reward * dont_move`
  - There is no term for CoM or CoP relative to the feet — [dm_control humanoid.py](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/humanoid.py)
- DeepMind Control Suite walker "stand" reward is "a combination of terms encouraging an upright torso and some minimal torso height". The humanoid tasks admit "many movement styles ... This facilitates exploration of local optima" — [arXiv 1801.00690](https://arxiv.org/abs/1801.00690)

**Balance reward terms that pay for weight placement**
- Yang, Komura & Li 2017 [SIM]: balance reward r = Σ w·exp(−α·error²) over:
  - torso and pelvis orientation;
  - "keeping the horizontal position of the COM close to the center of the foot" (r_xCOM);
  - CoM height (weight 5);
  - CoM horizontal and vertical velocity.

  The design rests on the capture point being inside the support polygon, with "maximum reachability of the capture point ... at the edge of the foot" — [arXiv 1809.02074](https://arxiv.org/abs/1809.02074)
- Ferigo et al. 2021 (iCub) [SIM], reward terms:
  - Boolean reward if the CoM ground projection is inside the support polygon "shrink[ed] ... by a 2.5 cm margin".
  - Target horizontal CoM velocity pointing to the polygon centre.
  - Feet contact forces near half body weight each.
  - "Feet CoPs: ... promote their positions to be located at the center of the corresponding sole".
  - A foot-orientation term added because "in early experiments, the policy was converging towards feet tipping behaviors, i.e. the feet were not in full contact with the ground".

  Source: [arXiv 2104.14534](https://arxiv.org/abs/2104.14534)
- DBHL, Xie et al. 2025 [PREPRINT]: a ZMP-based reward that "encourages the ZMP coordinates to be close to the center of the humanoid's support polygon", extended to a zero-moment line on non-planar ground. A MuJoCo ablation shows "significantly better success rate and [max distance] in most settings" with the ZMP reward than without, with one anomaly at slope 0.2. Hardware is Unitree H1-2 on narrow terrain and under pushes — [arXiv 2502.17219](https://arxiv.org/abs/2502.17219)
- Li et al. 2024 commonly use privileged quantities (global height, foot height, terrain height) inside reward terms while excluding them from the actor — [arXiv 2401.16889](https://arxiv.org/abs/2401.16889)

**Emergent posture and termination exploits**
- OstrichRL [SIM]: with a simple run-forward reward, "the same gait repeatedly emerged: the toes were unrealistically bent backwards (plantarflexed) and the ankles remained straight". Model adjustments were needed — [arXiv 2112.06061](https://arxiv.org/abs/2112.06061)
- Learning to Run [SIM]:
  - agents walked "never bending one of the legs" until a knee-bend reward was added, which also sped training;
  - another agent learned to "deliberately step on a large obstacle ... as a stepping stone".

  Source: [arXiv 1804.00361](https://arxiv.org/abs/1804.00361)
- Peng & van de Panne 2017 [SIM]: an "unnatural jittery" dog torque policy scored as robust because it "prevents the trunk from contacting the ground ... thereby escaping our system's fall detection", an exploit of the termination criterion — [arXiv 1611.01055](https://arxiv.org/abs/1611.01055)
- DeepMimic (ACM TOG 2018) [SIM]: reference-state initialisation and early termination are "critical"; early termination fires when the torso or head contacts the ground. The paper cites prior RL artifacts including "unrealistic posture" — [arXiv 1804.02717](https://arxiv.org/abs/1804.02717)
- RMA terminates episodes at pitch above 0.2 rad or roll above 0.4 rad. That kind of orientation threshold makes large leans terminal but leaves sub-threshold leans free — [arXiv 2107.04034](https://arxiv.org/abs/2107.04034)

**Seed variance and evaluation**
- Henderson et al. 2018 (AAAI 2018) [SIM]:
  - Ten TRPO HalfCheetah runs with identical hyperparameters, split into two groups of 5 seeds, gave "statistically different distributions just from varying random seeds" (t = −9.0916, p = 0.0016).
  - "It is not uncommon for the top-N trials to be selected from among several trials ... this can be potentially misleading."
  - Source: [arXiv 1709.06560](https://arxiv.org/abs/1709.06560)
- Agarwal et al. 2021 (NeurIPS 2021):
  - Protocols "reporting the maximum evaluation score achieved during training ... are generally incomparable with end-performance reported results". On Atari 100k they shift results "of a magnitude greater than the actual difference in score".
  - Recommended: interquartile mean (IQM), stratified-bootstrap confidence intervals and performance profiles.
  - CIs for the median are wide "even with 50 runs".
  - Source: [arXiv 2108.13264](https://arxiv.org/abs/2108.13264)
- Chan et al. 2020 (ICLR 2020): reliability metrics separate variability within training runs (dispersion, short- and long-term risk, drawdown), across runs, and across rollouts of a fixed policy after learning. Risk is measured with CVaR — [arXiv 1912.05663](https://arxiv.org/abs/1912.05663)
- Patterson et al. 2024 (JMLR 25), maximisation bias: "E[max_h Ḡ_h] ≥ max_h E[Ḡ_h]". Selecting the maximum over many noisy estimates overstates performance, and "the probability that we overestimate at least one configuration increases" with the number compared. A two-stage method (select, then re-evaluate on fresh runs) is recommended — [arXiv 2304.01315](https://arxiv.org/abs/2304.01315)
- SB3 `EvalCallback` keeps `best_mean_reward` and saves the best model to `best_model_save_path`, with `deterministic=True` by default — [SB3 callbacks.py](https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/callbacks.py)
- Andrychowicz et al. score configurations as the median of 3 seeds of area-under-curve, evaluated over 100 stochastic-policy episodes, "to reduce the impact of training noise" — [arXiv 2006.05990](https://arxiv.org/abs/2006.05990)
- DEP-RL reports the "fastest policy checkpoint from all runs" explicitly as a best-checkpoint statistic, separate from mean curves — [arXiv 2206.00484](https://arxiv.org/abs/2206.00484)
- Abdolhosseini et al. 2019: the baseline "achieves relatively high rewards in Stepper, [but] it was unable to make forward progress in any of the five runs". Reward is not the same as task quality — [PDF](https://www.cs.ubc.ca/~van/papers/2019-MIG-symmetry/2019-MIG-symmetry.pdf)

### Inferences
- Three mechanisms plausibly push reward-based best-checkpoint selection toward early checkpoints:
  - maximisation bias over many noisy evaluations;
  - a reward that does not penalise the toe-lean, so the gradual drift into it does not lower reward until falls begin;
  - deterministic evaluation differing from the stochastic training distribution.

  The literature remedy is to select on task-quality metrics: CoP margin from the support edge, survival under action noise and pushes, and CVaR over many rollouts. Re-evaluate the chosen checkpoint on fresh seeds, and report IQM with bootstrap CIs across at least 5–10 seeds.
- "One seed in three clean" is within normal deep-RL seed variance (Henderson; DEP-RL's 1 of 5 for MPO). Claims about a fix need enough seeds for confidence intervals rather than a single clean run.
- A CoP-centring term (Ferigo), a CoM-over-foot-centre term (Yang) or a ZMP-to-centre term (DBHL), possibly with a shrunken-polygon margin, is the literature's direct answer to "no reward term pays for where the weight sits fore-aft".

### Gaps
- I found no published RL standing study that reports the specific phenomenon of drift toward the front edge of the foot after N million steps; the closest is Ferigo's feet-tipping observation.
- I found no ablation isolating a CoP-centre reward term in standing; the DBHL ablation is locomotion on narrow terrain.

## 5. Musculoskeletal and animal RL; dinosaur and extinct-animal simulations

### Takeaway
Musculoskeletal RL benchmarks (OstrichRL, Learning to Run, MyoSuite/MyoLeg) mostly use proprioceptive and kinematic observations with simple rewards. They report unnatural emergent postures unless rewards are enriched with mocap tracking, effort/pain terms, knee-bend or similar shaping, or symmetry. Animal-imitation RL (DeepMimic, Peng 2020) controls posture through reference tracking, not balance sensing.

Extinct-dinosaur locomotion work has used optimisation, not RL:
- evolutionary or genetic optimisation in GaitSym for T. rex (Sellers 2017);
- 3D muscle-driven optimal control for Coelophysis (Bishop 2021).

Both use simple contact-sphere feet. No peer-reviewed RL theropod gait or standing study was found.

### Cited Findings
**OstrichRL**
- OstrichRL, La Barbera, Pardo, Tassa, Daley et al. [arXiv 2112.06061; venue not confirmed] [SIM, MuJoCo]:
  - The model is built from CT scans and dissection, with 120 muscles (68 legs, 52 neck).
  - MuJoCo was "roughly 600 times faster than OpenSim" over 97 forward simulations.
  - Run-forward observation: heights of head, pelvis and feet; joint positions and velocities; muscle forces, activations, lengths and velocities; CoM forward velocity. There is no contact force.
  - Termination: head below 0.9 m, pelvis below 0.6 m, or torso pitch beyond ±0.8 rad.
  - The simple reward produced plantarflexed toes with straight ankles.
  - Mocap tracking clips with "standing and abrupt changes in speed" took longer to learn.
  - Muscle excitation patterns were compared against emu electromyography (EMG).
  - The Cassie model "remains quite unstable on the motion capture tracking task".
  - Source: [arXiv 2112.06061](https://arxiv.org/abs/2112.06061)

**Learning to Run and later musculoskeletal work**
- Learning to Run 2017 (NIPS competition; Kidziński et al.) [SIM, OpenSim]:
  - The 18-muscle model receives a 41-D state: rotations and angular velocities of pelvis, hip, knee and ankle; positions and velocities of pelvis, CoM, head, torso, toes and talus; obstacle information.
  - Ground contact uses Hunt-Crossley spheres at heel and toes. GRFs are not in the observation.
  - Source: [arXiv 1804.00198](https://arxiv.org/abs/1804.00198)
- Learning to Run solution tricks (frameskip, reward shaping, symmetry, parameter noise, policy blending) are documented in [arXiv 1804.00361](https://arxiv.org/abs/1804.00361); see sections 2–4 above.
- MyoSuite (Caggiano et al. 2022) [PREPRINT at time of arXiv v1] is a MuJoCo musculoskeletal suite. The 2022 paper's tasks centre on the arm and hand (finger, elbow, hand object hold) — [arXiv 2205.13600](https://arxiv.org/abs/2205.13600)
- Schumacher et al. 2023 [PREPRINT] [SIM; Hyfydy and MuJoCo MyoLeg]:
  - Reward = velocity − effort − "pain", where pain covers joint-limit violation and vertical GRF above 1.2× body weight.
  - This produced gaits whose kinematics and GRFs are closer to human data.
  - Agents "only have access to proprioceptive feedback", yet were robust on a dynamic drawbridge terrain despite flat-ground training.
  - MyoLeg showed more seed variance and "unnatural lateral torso oscillations".
  - Source: [arXiv 2309.02976](https://arxiv.org/abs/2309.02976)
- DEP-RL (ICLR 2023) [SIM]: differential extrinsic plasticity (DEP) exploration improves learning in over-actuated and musculoskeletal systems, including ostrich-run and human-run. It gave the "largest ever measured top speed" checkpoint in ostrich-run and symmetric alternation in 5 of 5 seeds against 1 of 5 for MPO in human-run — [arXiv 2206.00484](https://arxiv.org/abs/2206.00484)

**Animal-motion imitation**
- DeepMimic, Peng et al. 2018 [SIM]: example-guided RL for humans and animals. Reference-state initialisation and early termination are critical for dynamic skills — [arXiv 1804.02717](https://arxiv.org/abs/1804.02717)
- Peng et al. 2020, "Learning agile robotic locomotion skills by imitating animals" (Laikago, RSS 2020) [HW]:
  - State: IMU root orientation and local joint rotations over the 3 previous timesteps, plus 3 previous actions.
  - Goal: target poses at 4 future timesteps (about 1 s).
  - No foot-contact input; root position deliberately excluded "to avoid the need to estimate the root position during real-world deployment".
  - PD targets are low-pass filtered.
  - Transfer uses domain randomisation plus latent-space adaptation.
  - Source: [arXiv 2004.00784](https://arxiv.org/abs/2004.00784)
- Peng & van de Panne 2017 [SIM]: trained a 19-link planar "raptor" (dinosaur-like biped) run. PD-target actions gave the best robustness overall; MTU actions were least robust for the raptor — [arXiv 1611.01055](https://arxiv.org/abs/1611.01055)

**Dinosaur and extinct-animal simulations**
- Sellers, Pond, Brassey, Manning & Bates 2017, T. rex (PeerJ 5:e3420) [SIM, optimisation, not RL]:
  - Multibody dynamic analysis (MBDA) in GaitSym combined with skeletal stress analysis.
  - "Machine learning algorithms" (a genetic algorithm searching for the fastest gait) generate the muscle activation patterns.
  - The model is restricted to the parasagittal plane. "Generating a stable gait took approximately 5,000 core hours"; over 200 optimisation runs were made.
  - Contact uses "contact spheres attached to the digits", acting "like stiff, damped springs". They "do not attempt to model the complex, non-linear interactions" between foot and ground.
  - Maximum speed peaks at 7.7 m/s with stress limits above 200 MPa. Realistic stress limits constrain T. rex to walking gaits.

  Source: [DOI 10.7717/peerj.3420](https://doi.org/10.7717/peerj.3420); [PMC5518979](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5518979/)
- Sellers 2017 cites the earlier "Sellers & Manning, 2007" evolutionary-robotics speed estimates — [PMC5518979](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5518979/)
- Bishop, Falisse, De Groote & Hutchinson 2021 (Science Advances) [SIM, optimal control, not RL]:
  - "First fully predictive, three-dimensional, muscle-driven simulations of locomotion in an extinct terrestrial vertebrate", Coelophysis.
  - Tail lateroflexion acts "as a passive, physics-based mechanism for regulating angular momentum and improving locomotor economy".
  - The optimal control problems converged "within 7 min using a single processing core".
  - The restricted metatarsophalangeal (MTP) angles may come from "the use of a single contact sphere to model foot-ground interactions".

  Source: [DOI 10.1126/sciadv.abi7348](https://doi.org/10.1126/sciadv.abi7348); [PMC8457660](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8457660/)

### Inferences
- In both dinosaur simulation lines (Sellers, Bishop), foot-ground contact is crude (spheres) and the authors flag it as a limitation affecting toe or MTP behaviour. A simulated T. rex whose toe and CoP behaviour depends on foot contact geometry is in line with those warnings.
- OstrichRL's emergent toe plantarflexion under a simple reward is the closest documented RL analogue to a theropod-like biped drifting onto its toes. Its authors fixed it with model changes and richer (mocap) rewards, not observation changes.
- None of the musculoskeletal or animal RL works found used foot force or CoP in the policy observation. Posture realism came from reward design (tracking, effort, pain, knee terms) and symmetry.

### Gaps
- No peer-reviewed RL study of theropod or T. rex standing or gait was found.
  - A 2026 JpGU conference abstract surfaced in search describes PPO (Unity ML-Agents) on a Parasaurolophus (hadrosaur) model. The page returned HTTP 403, so details are unverified — [JpGU 2026 O12-P55](https://pub.confit.atlas.jp/en/event/jpgu2026/presentation/O12-P55)
  - A 2014 Usami paper on T. rex running via evolutionary computation surfaced but was not read — [SciTePress PDF](https://www.scitepress.org/Papers/2014/50317/50317.pdf)
- The MyoChallenge (NeurIPS 2022–2024) leg and locomotion tracks were not researched here; the 2022 MyoSuite paper covers mostly upper-limb tasks.
- The OstrichRL publication venue was not confirmed; treat it as arXiv.
