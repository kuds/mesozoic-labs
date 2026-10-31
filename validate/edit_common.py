"""PS build: the edits of bs_common.py (each old string asserted unique), run once."""
import sys
from pathlib import Path

p = Path(sys.argv[1])
s = p.read_text()


def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old[:80])
    s = s.replace(old, new)


rep('''"""Bias screen 2 (BS2): pins and constants.  No logic beyond lookups and checks.

Importable without torch, numpy or the code copy.  BS2 is the bias screen (BS, S2/r10round/bias_screen, finished
and verified; its bs_common.py sha256 3b3727e2...) re-run from s42's 2,097,152 checkpoint: only the root, the start
pair, the window, the validation seeds and the code-copy path (BS's own copy, reused read only) differ from BS's file.
Every value cites its source:''', '''"""Phase screen (PS): pins and constants.  No logic beyond lookups and checks.

Importable without torch, numpy or the code copy.  PS is BS2 (S2/r10round/bias_screen2, finished and verified; its
bs_common.py sha256 6c4c36ea...) re-run from the same s42 2,097,152 pair over the same window with ONE new arm,
critic_phase (the critic also sees the episode phase; the actor never does): only the root, the arms, the phase
definition, the validation seeds and the continuation cap differ from BS2's file.  (The BS2 arms critic_blind_tail3,
gae_lambda_1 and frozen_obs_norm are not arms of this screen: bs_hooks.py keeps their hook code, unreachable here.)
Every value cites its source:''')
rep('''  AUTH   = the maintainer's overnight authorisation, S2/r10round/FOLLOW_UPS.txt, last entry (2026-10-30 00:5x UTC).
''', '''  AUTH   = the maintainer's overnight authorisation, S2/r10round/FOLLOW_UPS.txt, last entry (2026-10-30 00:5x UTC).
  TASK3  = PS's build task of 2026-10-31 (the maintainer's choice of 2026-10-30 18:27 UTC, "Test the phase fix first
           (Recommended)", resumed 2026-10-31: BS2's design from 2,097,152 to 3,538,944 with the arm critic_phase,
           4 critic_phase + 4 new controls, validation seeds 9401-9441, the root /home/user/work/phase_screen).
  CODE_  = the code copy (BS/code, commit 8737cd3d): environments/shared/base_env.py step() l1335 (_step_count += 1
           BEFORE _get_obs l1349 and the reward l1352) and reset() l1642 / l1668 (_step_count = 0 before _get_obs);
           environments/tyrannosaurus_rex/base_env.py _get_obs l974-1023 (63 values, the last three the command
           v_x, v_y, yaw_rate, zero in stance) and the four gated stance terms l1278 (width, paid only for
           _step_count > stance_width_settle_steps), l1315 (lateral lean), l1339 (stagger), l1421 (CoP band);
           configs/tyrannosaurus_rex/anatomical/stance.toml stance_width_settle_steps = 200.
''')
rep('''pinned by sha256 in <BS2>/APPROVED (screen_sha256) with the criteria and these tools.  (In this file and the other BS2
tools "<BS>" means the screen's own root, BS2.)''', '''pinned by sha256 in <PS>/APPROVED (screen_sha256) with the criteria and these tools.  (In this file and the other PS
tools "<BS>" means the screen's own root, PS.)''')
rep('''BS = S2 + "/r10round/bias_screen2"              # every file the screen writes is under it (TASK2)
BS1 = S2 + "/r10round/bias_screen"              # the finished first screen: read only (its code copy is reused)''',
    '''BS = "/home/user/work/phase_screen"            # PS: every file the screen writes is under it (TASK3)
BS1 = S2 + "/r10round/bias_screen"              # the finished first screen: read only (its code copy is reused)
BS2_DIR = S2 + "/r10round/bias_screen2"         # the finished second screen: read only (the harness PS derives from)''')
rep('''MAX_CONTINUATIONS = 9                           # AUTH (b): <= 9 continuations''',
    '''MAX_CONTINUATIONS = 8                           # TASK3: 4 critic_phase + 4 controls (AUTH (b) allowed <= 9)''')
rep('''S = 2_097_152                                   # TASK2: s42's 2,097,152 checkpoint (the early-bias probe's driver)
LENGTH = 1_441_792                              # TASK2 = BS's: 11 evaluations of 131,072; 176 rollouts, 175 updates''',
    '''S = 2_097_152                                   # TASK2 = TASK3: s42's 2,097,152 checkpoint (BS2's S)
LENGTH = 1_441_792                              # TASK2 = TASK3 = BS's: 11 evaluations of 131,072; 176 rollouts''')
rep('''# ------------------------------------------------------------------------------------------------------------- arms
CONTROL = "control"
INTERVENTIONS = ("critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm")
ARMS = (CONTROL, *INTERVENTIONS)
# In-memory [ppo] edits of the stage config before the load (only gae_lambda_1 has one; the other two are hooks on
# the loaded objects, bs_hooks.py, and leave the stage config and its hyperparameters digest unchanged).
ARM_PPO_EDIT = {"control": {}, "critic_blind_tail3": {}, "gae_lambda_1": {"gae_lambda": 1.0}, "frozen_obs_norm": {}}
ARM_GAE_LAMBDA = {"control": 0.95, "critic_blind_tail3": 0.95, "gae_lambda_1": 1.0, "frozen_obs_norm": 0.95}''',
    '''# ------------------------------------------------------------------------------------------------------------- phase
# THE PHASE (critic_phase only; TASK3 2(a)).  k = the env's own per-episode step counter _step_count AT THE MOMENT THE
# OBSERVATION IS MADE (0 at reset; step() increments it before _get_obs, CODE_), so an observation with counter k is
# the state from which the (k+1)-th action is taken, and that action's reward is computed with _step_count = k + 1.
# The four gated stance terms pay iff _step_count > SETTLE_STEPS at the reward (CODE_ l1278, l1315, l1339, l1421),
# i.e. iff the observation's k >= SETTLE_STEPS: k = 200 is the first observation whose own reward is gated.  The
# critic gets three features, in the observation's three command slots (always zero in stance; CODE_):
#   ramp     = min(k, 200) / 200                        a linear clock over the reset transient and the pre-settle
#                                                       window (the pop, steps 0-50; the quiet settle, 50-200)
#   settled  = 1.0 if k >= 200 else 0.0                exactly the reward gate of the observation's own reward
#   discount = GAMMA ** max(0, 200 - k)                 the discount of the first gated reward seen from k: the
#                                                       gated stream enters V(k) with this weight (the why-probe's
#                                                       primary phase control R(k), phase_probe PREREG.txt l56)
# All three equal 1.0 from k = 200 on (the value has no other phase after the settle: the 1,000-step time limit is
# bootstrapped by SB3).  The actor never sees them (bs_hooks: its input slots are set to 0.0, the value they always
# had); they are not normalised by running statistics (their VecNormalize slots are reseeded to mean 0 / var 1 and
# frozen there, main's command_frame.reseed_command_slice rule, so the critic receives f / sqrt(1 + 1e-8) = f in
# float32).
PHASE_SLOTS = (60, 61, 62)                      # the command slots v_x, v_y, yaw_rate (CODE_ _get_obs l1019)
PHASE_FEATURES = ("ramp", "settled", "discount")
SETTLE_STEPS = 200                              # stance.toml stance_width_settle_steps (FMAN stage_config: 200)
PHASE_GAMMA = 0.98                              # the stage's [ppo] gamma (stage_config.json hyperparameters)
OBS_DIM = 63


def phase_features(k: int) -> tuple[float, float, float]:
    """(ramp, settled, discount) of an observation made at episode step counter k >= 0 (pure; see THE PHASE)."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"step counter {k!r} is not a non-negative integer")
    return (min(k, SETTLE_STEPS) / SETTLE_STEPS, 1.0 if k >= SETTLE_STEPS else 0.0,
            PHASE_GAMMA ** max(0, SETTLE_STEPS - k))


# ------------------------------------------------------------------------------------------------------------- arms
CONTROL = "control"
INTERVENTIONS = ("critic_phase",)
ARMS = (CONTROL, *INTERVENTIONS)
# In-memory [ppo] edits of the stage config before the load: none in PS (critic_phase is a hook on the loaded objects,
# bs_hooks.py, and leaves the stage config and its hyperparameters digest unchanged).
ARM_PPO_EDIT = {"control": {}, "critic_phase": {}}
ARM_GAE_LAMBDA = {"control": 0.95, "critic_phase": 0.95}''')
old_change_start = s.index('''ARM_CHANGE = {''')
old_change_end = s.index('''}

# ------------------------------------------------------------------------------------------------------------- start''')
s = s[:old_change_start] + '''ARM_CHANGE = {
    "control": "nothing: the run's own same-stage resume, every setting as recorded",
    "critic_phase": ("the critic alone also sees the episode phase: each TRAINING env's observation carries "
                     "(ramp, settled, discount) of its own step counter in the three command slots 60-62 (always "
                     "zero in stance; a class swap on each env instance's _get_obs); the policy's mlp_extractor gets a "
                     "forward_actor that sets the NORMALISED slots 60-62 to 0.0 (the value the actor always saw) "
                     "before the policy MLP, so the actor never sees the phase and its first-layer phase columns get "
                     "no gradient; the critic's first-layer columns 60-62 (value_net.0.weight[:, 60:63]) are set to "
                     "zero at the load (their Adam moments are already exactly zero), so the critic at the load is "
                     "the start pair's on every input and learns the phase from there; the training VecNormalize's "
                     "slots 60-62 are reseeded to mean 0 / var 1 and kept there after every update (a class swap on "
                     "the SymmetricRunningMeanStd instance, pickling as the stock RunningMeanStd); MirrorKLPPO's "
                     "observation mirror map maps slots 60-62 to themselves with sign +1; the policy probe feeds "
                     "each bank state's phase to the critic (all ten bank states are settled, k >= 200); the "
                     "evaluation env, the probe's bank env and every other setting are the run's own")''' + s[old_change_end:]
rep('''# Validation runs (V1, V2, timing) use only these seeds and write only under <BS>/validate/; SCREEN.json may not use
# them, so a validation folder and a continuation can never share a name.  BS2's own (TASK2; BS used 9101-9141).
VALIDATION_SEEDS = (9201, 9211, 9221, 9231, 9241)''', '''# Validation runs (V1, V2, timing) use only these seeds and write only under <BS>/validate/; SCREEN.json may not use
# them, so a validation folder and a continuation can never share a name.  PS's own (TASK3; BS2 used 9201-9241, BS
# 9101-9141).
VALIDATION_SEEDS = (9401, 9411, 9421, 9431, 9441)''')
p.write_text(s)
print("ok")
