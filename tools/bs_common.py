"""Bias screen (BS): pins and constants.  No logic beyond lookups and checks.

Importable without torch, numpy or the code copy.  Derived from the entropy continuation screen's ecs_common.py
(S2/r10round/ecs/tools/ecs_common.py, sha256 05864c64...); every value cites its source:
  ECS    = ecs_common.py (the pins it shares with this screen: the code copy, M1/M5, the stack, the s42 records);
  FMAN   = S1/foot/stance6/fetch/manifest.csv (F2 COPY rows, copy 1) [measured 2026-10-30 03:53 UTC, check_inputs.py];
  TASK   = the build task of 2026-10-30 (S = 1,048,576; E = S + 1,441,792; at most 9 continuations of at most 1.5M
           steps; 4 at a time at nice 10; disk floor 1.5 GB; arms control / critic_blind_tail3 / gae_lambda_1 /
           frozen_obs_norm, none chosen);
  AUTH   = the maintainer's overnight authorisation, S2/r10round/FOLLOW_UPS.txt, last entry (2026-10-30 00:5x UTC).
The intervention and the queue are NOT here: they are <BS>/SCREEN.json, written when the intervention is chosen and the
criteria are reviewed, and pinned by sha256 in <BS>/APPROVED (screen_sha256) with the criteria and these tools.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re

# ------------------------------------------------------------------------------------------------------------- paths
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
BS = S2 + "/r10round/bias_screen"               # every file the screen writes is under it (TASK)
CODE = BS + "/code"                             # cp -a of ECS/code; tree digest = TREE_DIGEST (never edited)
TOOLS = BS + "/tools"
RUNS = BS + "/runs"                             # <BS>/runs/<arm>_<seed>/
VALIDATE = BS + "/validate"                     # the no-optimizer-step modes write only here
READS = BS + "/reads"                           # read_bs.py's full reads of continuations: <BS>/reads/<name>/
LOGS = BS + "/logs"
TMP = BS + "/tmp"                               # TMPDIR and MPLCONFIGDIR per run
LOST_JSON = BS + "/lost.json"
APPROVED = BS + "/APPROVED"                     # never written by the build
CRITERIA = BS + "/criteria.txt"                 # never written by the build
SCREEN = BS + "/SCREEN.json"                    # never written by the build (the chosen intervention and the queue)
PY = S2 + "/venv/bin/python"
FETCH = S1 + "/foot/stance6/fetch"
FMAN = FETCH + "/manifest.csv"
F2_ROOT = FETCH + "/F2"
M1_COMMON = S2 + "/r10round/lean/build/laneA/tools/r10m_common.py"
M5 = S2 + "/r10round/lean/build/laneB/tools/ledger_r10round.py"
M5_COMMON = S2 + "/r10round/lean/build/laneB/tools/r10round_common.py"
SEAL_NAME = ".r10m_seal.json"

# ------------------------------------------------------------------------------------------------------------- pins
COMMIT = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"        # ECS / the seal
TREE_DIGEST = "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1"   # ECS / the seal (TASK item 1)
TREE_FILES = 1406
PIN_SHA256 = {                                  # ECS PIN_SHA256 (M1_COMMON for the seal check, M5 for the ledger)
    M1_COMMON: "97431323a84a45026671d681dbd0dc5fe3914aa4068ab20a49e428620b26f5bf",
    M5: "0d89bcb858e19f780c3ce0ac7a5886fc01b33d78164d9ee37d2e50dc1d4b0511",
    M5_COMMON: "ea2c5ea6984b8dc00c1c066a4441a59526c515ceb111da17eaf1688a009226c5",
}
STACK = {"python": "3.13.17", "torch": "2.13.0+cpu", "numpy": "2.5.4", "gymnasium": "1.4.0",
         "stable_baselines3": "2.9.0", "mujoco": "3.10.0"}           # ECS STACK
DISK_FLOOR_BYTES = 1_610_612_736                # 1.5 GiB (TASK "1.5 GB"; the binary reading, the stricter one)
RUN_BYTES_EST = 50 * 1024**2                    # one continuation's folder (an ECS one is 36 MB with 8 zips) [measured]
MAX_CONTINUATIONS = 9                           # AUTH (b): <= 9 continuations
LAUNCH_DISK_BYTES = DISK_FLOOR_BYTES + MAX_CONTINUATIONS * RUN_BYTES_EST   # a fresh queue starts only above this
TOOL_FILES = ("run_bs.py", "read_bs.py", "run_waves_bs.sh", "bs_common.py", "bs_hooks.py")   # pinned in APPROVED

# ------------------------------------------------------------------------------------------------------------- run grid
TOTAL_TIMESTEPS = 12_000_000                    # the run's _total_timesteps; the body's total = 12M - S (ECS)
N_ENVS = 4
N_STEPS = 2048
ROLLOUT = N_ENVS * N_STEPS                      # 8,192 steps per rollout / update
EVAL_FREQ = 131_072                             # stage eval_freq (ECS)
SAVE_FREQ = 262_144                             # stage checkpoint_freq (ECS)
S = 1_048_576                                   # TASK: s42's 1,048,576 checkpoint (before the tail bias appears)
LENGTH = 1_441_792                              # TASK: 11 evaluations of 131,072; 176 rollouts, 175 updates
E = S + LENGTH                                  # 2,490,368 (on the evaluation grid; NOT on the 262,144 save grid)
MAX_LENGTH = 1_500_000                          # AUTH (b): <= 1.5M steps each
N_EVALS = LENGTH // EVAL_FREQ                   # 11
# The saved pairs of a continuation: the periodic ones at S + 262,144 k inside (S, E), and the final pair at E
# (stage1_final.zip / stage1_final_vecnorm.pkl: E is not on the save grid, so no periodic pair is written there).
SAVED_STEPS = tuple(range(S + SAVE_FREQ, E, SAVE_FREQ))       # 1,310,720 ... 2,359,296 (5)
FINAL_ZIP, FINAL_PKL = "stage1_final.zip", "stage1_final_vecnorm.pkl"
VALIDATE_UNTIL = S + 2 * EVAL_FREQ              # V2: S + 262,144 (TASK item 5)
EVAL_SEED_OFFSET = 1000                         # eval env seed = seed + 1000 (ECS)
STAGE_DIR = "01_stance"
STAGE_KEY = "stance"
ALGORITHM = "ppo"
COLLAPSE_PATIENCE = 10                          # stance.toml (ECS)
PROBE_CHECK_COLUMNS = ("update_timesteps", "ent_coef", "learning_rate")
OBS_TAIL3 = 6                                   # observation index of the tail_3 pitch angle (rad) (ew_common I_T3)
ACT_TAIL3 = 14                                  # action index of tail_3_pitch_act (ew_common A_T3)
BANK_STATUE_SETTLED = (0, 5)                    # the probe bank's two statue_settled rows (seeds 3042, 3043)

# ------------------------------------------------------------------------------------------------------------- arms
CONTROL = "control"
INTERVENTIONS = ("critic_blind_tail3", "gae_lambda_1", "frozen_obs_norm")
ARMS = (CONTROL, *INTERVENTIONS)
# In-memory [ppo] edits of the stage config before the load (only gae_lambda_1 has one; the other two are hooks on
# the loaded objects, bs_hooks.py, and leave the stage config and its hyperparameters digest unchanged).
ARM_PPO_EDIT = {"control": {}, "critic_blind_tail3": {}, "gae_lambda_1": {"gae_lambda": 1.0}, "frozen_obs_norm": {}}
ARM_GAE_LAMBDA = {"control": 0.95, "critic_blind_tail3": 0.95, "gae_lambda_1": 1.0, "frozen_obs_norm": 0.95}
ENT_COEF_AFTER_LOAD = 0.005                     # the config's ent_coef, every arm (ECS arm A)
HYPERPARAMETERS_SHA256_RUN = "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c"  # ECS (copy1)
# What each arm changes, recorded verbatim in run_meta ("arm_change").
ARM_CHANGE = {
    "control": "nothing: the run's own same-stage resume, every setting as recorded",
    "critic_blind_tail3": ("the value path only: the policy's mlp_extractor gets a forward_critic that replaces the "
                           "NORMALISED observation index 6 (tail_3 pitch angle) by 0.0 before the value MLP; "
                           "ActorCriticPolicy.forward, evaluate_actions and predict_values (and MirrorKLPPO's "
                           "mirrored-critic loss, which calls predict_values) all reach the value MLP through it; the "
                           "actor's inputs, every parameter, its shape and the optimizer are unchanged (a class swap "
                           "on the mlp_extractor instance, so the state_dict and the saved zip are stock)"),
    "gae_lambda_1": ("[ppo] gae_lambda 0.95 -> 1.0 in memory before the load (SB3's load applies it and builds the "
                     "rollout buffer with it): advantages are the bootstrapped Monte Carlo returns minus V; the "
                     "value target becomes the same Monte Carlo return; hyperparameters_sha256 changes"),
    "frozen_obs_norm": ("the training VecNormalize's observation statistics are frozen at the start pair's: its "
                        "obs_rms.update becomes a counted no-op (a class swap on the SymmetricRunningMeanStd "
                        "instance, pickling as the stock RunningMeanStd); VecNormalize.training stays True, so the "
                        "return statistics (ret_rms) keep updating and rewards are normalised exactly as the run "
                        "does; the evaluation env receives the frozen statistics by the run's own sync"),
}

# ------------------------------------------------------------------------------------------------------------- start
PARENT_RECORDS = ("eval_episodes.csv", "evaluations.npz", "gate_progress.npz", "train_episodes.csv",
                  "policy_probe.npz")             # copied under their real names (ECS)
START = {                                       # FMAN F2 COPY rows, copy 1, sha256 in full [measured, check_inputs.py]
    "run_id": "stance6_r10LTSB_s42", "own_seed": 42, "S": S,
    "zip_rel": "01_stance/models/stage1_1048576_steps.zip",
    "zip_sha256": "0f765bd692371f212555f6b203c4f2ce14aaa05df62f58202017c227424d2b80",
    "pkl_rel": "01_stance/models/stage1_vecnormalize_1048576_steps.pkl",
    "pkl_sha256": "52e52a7b9820af808265acb2d4f6d1573e6edebb550393db8ee47d9676229bcb",
    "records": {
        "eval_episodes.csv": "d1bcb78964691cf194e6276f06740d1f24aa69ad36831a046ad6307c1c4db11e",
        "evaluations.npz": "3e11136120622b90cc63203c46332004a9989188a0b8fe91620dbabb9f2563e9",
        "gate_progress.npz": "4097099b6880fa234a38e1b43819a21a8531f39ca0aa1daefaaeeb35a06c5a0e",
        "train_episodes.csv": "14b2f9e49b67e9c90cc79362f740e1768de5a2d159035176d5299b9c476d851d",
        "policy_probe.npz": "4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11",
        "stage_config.json": "5d54e6deac8ddda5c37914da97d4d59b6838a8daef52cb7bec237c49d63d24ec",
    },
    "probe_row_at_S": 127,                      # update_timesteps == 1,048,576 in the parent's probe [measured]
}

# ------------------------------------------------------------------------------------------------------------- seeds
# Validation runs (V1, V2, timing) use only these seeds and write only under <BS>/validate/; SCREEN.json may not use
# them, so a validation folder and a continuation can never share a name.
VALIDATION_SEEDS = (9101, 9111, 9121, 9131, 9141)
WAVE_SLOTS = 4                                  # TASK item 3
WAVE_NICE = 10
MAX_CRASHES = 3                                 # ECS: a third crash of its own writes it to lost.json

# ------------------------------------------------------------------------------------------------------------- codes
EXIT_OK, EXIT_FAIL, EXIT_REFUSED, EXIT_PROVISIONAL, EXIT_CHECK_FAIL = 0, 1, 2, 3, 4
EXIT_POST_CHECK = 5                             # finished, a post-run check failed (ECS)
TORCH_THREADS = 1
STOP_REASONS = ("end", "collapse", "other")

# ------------------------------------------------------------------------------------------------------------- records
RUN_SCHEMA = "bs-run/1"
RUN_META = "run_meta.json"
EVIDENCE = "arm_evidence.jsonl"                 # one line per rollout end (bs_hooks.rollout_evidence)
CHECK_JSON = "check.json"
RUN_MODES = ("continuation", "save_loaded_only", "no_update")
APPROVED_KEYS = ("criteria_sha256", "screen_sha256", "code_tree_digest", "tool_sha256")
SCREEN_SCHEMA = "bs-screen/1"
LOST_SCHEMA = "bs-lost/1"
_ARM_RE = "|".join(re.escape(a) for a in ARMS)
RUN_NAME_RE = re.compile(rf"^({_ARM_RE})_(\d+)(?:_(crashed|void|refused)(\d+))?$")


# ------------------------------------------------------------------------------------------------------------- lookups
def run_name(arm: str, seed: int) -> str:
    """<arm>_<seed>, for example control_2101."""
    return f"{arm}_{int(seed)}"


def parse_run_name(name: str) -> dict | None:
    """{"arm", "seed", "suffix", "k"} for a run folder name, or None."""
    m = RUN_NAME_RE.match(os.path.basename(str(name).rstrip("/")))
    if not m:
        return None
    return {"arm": m.group(1), "seed": int(m.group(2)), "suffix": m.group(3),
            "k": int(m.group(4)) if m.group(4) else None}


def f2_copies() -> str:
    return f"{F2_ROOT}/{START['run_id']}/copies"


def f2_file(rel: str) -> str:
    """The F2 copy of a run-relative path: <copies>/<rel>.copy1 (read only)."""
    return f"{f2_copies()}/{rel}.copy1"


def f2_record(name: str) -> str:
    return f2_file(f"{STAGE_DIR}/{name}")


def periodic_names(steps: int) -> tuple[str, str]:
    """The resume path's periodic checkpoint names (ECS)."""
    return f"stage1_{int(steps)}_steps.zip", f"stage1_vecnormalize_{int(steps)}_steps.pkl"


def expected_update_rows(S_: int, T: int) -> int:
    """Probe rows with update_timesteps in (S, T): (T - S)/8,192 - 1 (175 at E)."""
    if (T - S_) % ROLLOUT or T <= S_:
        raise ValueError(f"T - S = {T - S_} is not a positive multiple of {ROLLOUT}")
    return (T - S_) // ROLLOUT - 1


def derived_seeds(seed: int) -> tuple[int, ...]:
    """The seeds a continuation uses: model and training env 0 (seed), envs 1-3 (seed + rank), eval (seed + 1000)."""
    return (*(int(seed) + r for r in range(N_ENVS)), int(seed) + EVAL_SEED_OFFSET)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stack_versions() -> dict:
    out = {"python": platform.python_version()}
    for key in ("torch", "numpy", "gymnasium", "stable_baselines3", "mujoco"):
        try:
            out[key] = importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:
            out[key] = None
    return out


def check_stack() -> list[str]:
    got = stack_versions()
    return [f"{k}: running {got.get(k)} != pinned {v}" for k, v in STACK.items() if got.get(k) != v]


def check_pins() -> list[str]:
    out = []
    for path, want in PIN_SHA256.items():
        try:
            got = sha256_file(path)
        except OSError as exc:
            out.append(f"{path}: {exc}")
            continue
        if got != want:
            out.append(f"{path}: sha256 {got} != pinned {want}")
    return out


# ------------------------------------------------------------------------------------------------------------- SCREEN
# <BS>/SCREEN.json: {"schema": "bs-screen/1", "intervention": one of INTERVENTIONS,
#                    "queue": [[arm, seed], ...]}   (launch order; arms only "control" and the intervention)
def screen_problems(d) -> list[str]:
    """Every reason a SCREEN.json object is not a valid screen; empty means valid (pure)."""
    if not isinstance(d, dict):
        return ["SCREEN: not a JSON object"]
    p = []
    if d.get("schema") != SCREEN_SCHEMA:
        p.append(f"SCREEN: schema {d.get('schema')!r} != {SCREEN_SCHEMA!r}")
    inter = d.get("intervention")
    if inter not in INTERVENTIONS:
        p.append(f"SCREEN: intervention {inter!r} not one of {INTERVENTIONS}")
    q = d.get("queue")
    if not isinstance(q, list) or not q:
        return p + ["SCREEN: queue is not a non-empty list"]
    if len(q) > MAX_CONTINUATIONS:
        p.append(f"SCREEN: {len(q)} continuations > {MAX_CONTINUATIONS} (AUTH)")
    seen = set()
    arms = set()
    for i, item in enumerate(q):
        if (not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str)
                or isinstance(item[1], bool) or not isinstance(item[1], int)):
            p.append(f"SCREEN: queue[{i}] {item!r} is not [arm, integer seed]")
            continue
        arm, seed = item
        if arm not in (CONTROL, inter):
            p.append(f"SCREEN: queue[{i}] arm {arm!r} is neither control nor the intervention {inter!r}")
        if not 0 < seed < 2**31 - EVAL_SEED_OFFSET - N_ENVS:
            p.append(f"SCREEN: queue[{i}] seed {seed} out of range")
        if seed in VALIDATION_SEEDS:
            p.append(f"SCREEN: queue[{i}] seed {seed} is a validation seed")
        if (arm, seed) in seen:
            p.append(f"SCREEN: queue[{i}] {arm} {seed} repeated")
        seen.add((arm, seed))
        arms.add(arm)
    if inter in INTERVENTIONS and arms != {CONTROL, inter}:
        p.append(f"SCREEN: the queue's arms {sorted(arms)} are not control and {inter}")
    return p


def load_screen(path: str = SCREEN) -> tuple[dict | None, list[str]]:
    """(SCREEN object, problems).  Missing or unreadable is a problem."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except FileNotFoundError:
        return None, [f"SCREEN: {path} missing (the intervention is not chosen yet)"]
    except (OSError, ValueError) as exc:
        return None, [f"SCREEN: {path} does not parse ({exc.__class__.__name__})"]
    return d, screen_problems(d)


def queue_of(d: dict) -> tuple[tuple[str, int], ...]:
    return tuple((str(a), int(s)) for a, s in d["queue"])
