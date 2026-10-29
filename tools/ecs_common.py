"""Entropy continuation screen (ECS): pins and constants (build list v2, item 2).

No logic beyond lookups and checks.  Importable without torch, numpy or the code copy.  Every value cites its source:
criteria_ecs_v2.txt (CRIT), decisions_ecs_v2.txt (DEC), build_list_ecs_v2.txt (BL), survey_harness.txt (HS), the
round's fetch manifest (FMAN), COPY/.r10m_seal.json (SEAL).  The names below are imported by read_ecs.py and
score_ecs.py (lane R) and by run_ecs.py / run_waves_ecs.sh (lane H): they are not renamed.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
import re

# ------------------------------------------------------------------------------------------------------------- paths
S1 = "/tmp/claude-0/-home-user-mesozoic-labs/0fdf74a4-2989-56c9-a773-92101004d7de/scratchpad"
S2 = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad"
ECS = S2 + "/r10round/ecs"                       # every file the screen writes is under it (CRIT SHORT NAMES)
CODE = ECS + "/code"                            # the code copy (BL 1, T2)
TOOLS = ECS + "/tools"
RUNS = ECS + "/runs"                            # <ECS>/runs/<start>_<arm>_<seed>/ (CRIT WHERE AND HOW)
VALIDATE = ECS + "/validate"                    # V1-V5 outputs, the no-optimizer-step modes (BL 3, 10)
TRACES = ECS + "/traces"                        # R3 end traces, <ECS>/traces/<name>/ (BL 8)
SCORE = ECS + "/score"
LOGS = ECS + "/logs"
TMP = ECS + "/tmp"                              # TMPDIR and MPLCONFIGDIR (BL 3)
LOST_JSON = ECS + "/lost.json"
APPROVED = ECS + "/APPROVED"                    # never written by the build (BL 12)
CRITERIA = ECS + "/criteria.txt"                # never written by the build (BL 13)
PY = S2 + "/venv/bin/python"
RUNLOG = S1 + "/foot/stance4/live/run_log.txt"
ROUND = S1 + "/foot/stance6"
ROUND_APPROVED = ROUND + "/APPROVED"            # M1 build-copy --approved (BL 1)
CRIT6 = ROUND + "/criteria.txt"
FMAN = ROUND + "/fetch/manifest.csv"
F2_ROOT = ROUND + "/fetch/F2"                   # F2/<RUN_ID>/copies/ (read only)
PANELS = ROUND + "/panels"                      # PAN = PANELS/<RUN_ID>/<label>/m1/m1_summary.json
COPY = S2 + "/r10round/f2/copy_8737cd3d"        # the sealed measuring copy (read only)
CLONE = S2 + "/r10round/runbranch/clone"        # the run branch at 8737cd3d (read only)
M1 = S2 + "/r10round/lean/build/laneA/tools/measure_r10round.py"
M1_COMMON = S2 + "/r10round/lean/build/laneA/tools/r10m_common.py"
M5 = S2 + "/r10round/lean/build/laneB/tools/ledger_r10round.py"
M5_COMMON = S2 + "/r10round/lean/build/laneB/tools/r10round_common.py"
SEAL_NAME = ".r10m_seal.json"

# ------------------------------------------------------------------------------------------------------------- pins
COMMIT = "8737cd3d86431de1bdd0fcdb49a0d956a45e084d"        # SEAL commit; GITHUB_SHA in the run env (BL 3)
TREE_DIGEST = "331fe8fc3f6bdf73acc0d3cc324dd753d9d9d695d7ce914e4e011d5c6437ebf1"   # SEAL tree_digest (BL 1)
TREE_FILES = 1406
PIN_SHA256 = {                                  # the lane SHA256SUMS files (BL 2) [measured, sha256sum 21:12 UTC]
    M1: "b293a511d8beb5b6d6f24cec8744e601382748ce9b2f89e6bb653b6c97f5840c",
    M1_COMMON: "97431323a84a45026671d681dbd0dc5fe3914aa4068ab20a49e428620b26f5bf",
    M5: "0d89bcb858e19f780c3ce0ac7a5886fc01b33d78164d9ee37d2e50dc1d4b0511",
    M5_COMMON: "ea2c5ea6984b8dc00c1c066a4441a59526c515ceb111da17eaf1688a009226c5",
}
STACK = {"python": "3.13.17", "torch": "2.13.0+cpu", "numpy": "2.5.4", "gymnasium": "1.4.0",
         "stable_baselines3": "2.9.0", "mujoco": "3.10.0"}           # BL 2
DISK_FLOOR_BYTES = 3 * 1024**3                  # before each continuation (T4; M1's r10m_common.py l107)
LAUNCH_DISK_BYTES = 6 * 1024**3                 # before the launch (T4)
# The ECS tool files, pinned in <ECS>/APPROVED (tool_sha256) after the review; ecs_common.py is pinned there too.
TOOL_FILES = ("run_ecs.py", "read_ecs.py", "score_ecs.py", "run_waves_ecs.sh", "ecs_common.py")

# ------------------------------------------------------------------------------------------------------------- run grid
TOTAL_TIMESTEPS = 12_000_000                    # the run's _total_timesteps; the body's total = 12M - S (BL 3)
N_ENVS = 4
N_STEPS = 2048
ROLLOUT = N_ENVS * N_STEPS                      # 8,192 steps per rollout / update
EVAL_FREQ = 131_072                             # stage config eval_freq (BL 3)
SAVE_FREQ = 262_144                             # stage config checkpoint_freq (BL 3)
LENGTH = 1_572_864                              # D3 (a): 12 evaluations, 192 rollouts, 191 updates
VALIDATE_LENGTH = 262_144                       # V2: S + 262,144 (BL 10)
EVAL_SEED_OFFSET = 1000                         # eval env seed = seed + 1000 (TB l1215-1226)
STAGE_DIR = "01_stance"
STAGE_KEY = "stance"                            # the T. rex stance stage (stage 1)
STAGE_INDEX = 1
STAGE_TOML_REL = "configs/tyrannosaurus_rex/anatomical/stance.toml"
ALGORITHM = "ppo"
COLLAPSE_PATIENCE = 10                          # stance.toml l1041 (D8)
M1_SEEDS = tuple(range(3042, 3062))             # R3 end-trace seeds
PROBE_CHECK_COLUMNS = ("update_timesteps", "ent_coef", "learning_rate")       # R4 (BL 6)
M6_COLUMNS = ("rollout_mu_beyond_high_share", "rollout_mu_beyond_low_share", "rollout_action_clip_high_share",
              "rollout_action_clip_low_share", "rollout_mu_mean", "rollout_mu_q50", "rollout_mu_q95",
              "std_per_actuator")                                              # R2 (CRIT THE READINGS)

# ------------------------------------------------------------------------------------------------------------- arms
ARMS = ("A", "B")                               # D1 (a)
ARM_PPO_EDIT = {"A": {}, "B": {"ent_coef": 0.0, "ent_coef_end": 0.0}}         # in memory, before the load
ARM_ENT_COEF_AFTER_LOAD = {"A": 0.005, "B": 0.0}                              # the config value (HS 3.1)
ARM_EXPECTED_DIFF_KEYS = {"A": frozenset(), "B": frozenset({"ent_coef", "ent_coef_end"})}   # V5
HYPERPARAMETERS_SHA256_A = "sha256:054ab48303167eda1bd66c8894a64ef516bedd041a9e12f57dbd20e75c026b9c"   # copy1 run

# ------------------------------------------------------------------------------------------------------------- starts
# D3 (a).  Start pairs and parent records: FMAN F2 COPY rows, sha256 in full [measured, FMAN 21:11 UTC].
PARENT_RECORDS = ("eval_episodes.csv", "evaluations.npz", "gate_progress.npz", "train_episodes.csv",
                  "policy_probe.npz")             # copied under their real names (CRIT Inputs, T3)
STARTS = {
    "s42": {
        "run_id": "stance6_r10LTSB_s42", "own_seed": 42, "S": 3_145_728, "E": 4_718_592,
        "zip_rel": "01_stance/models/stage1_3145728_steps.zip",
        "zip_sha256": "3c054bd738304965bbbcfde588982635ed047436311a267b91f8855e02edad18",
        "pkl_rel": "01_stance/models/stage1_vecnormalize_3145728_steps.pkl",
        "pkl_sha256": "997f31421d805f6e6cd366b9a9bd05451ed7ab679fd615e7567d259a310f22e0",
        "records": {
            "eval_episodes.csv": "d1bcb78964691cf194e6276f06740d1f24aa69ad36831a046ad6307c1c4db11e",
            "evaluations.npz": "3e11136120622b90cc63203c46332004a9989188a0b8fe91620dbabb9f2563e9",
            "gate_progress.npz": "4097099b6880fa234a38e1b43819a21a8531f39ca0aa1daefaaeeb35a06c5a0e",
            "train_episodes.csv": "14b2f9e49b67e9c90cc79362f740e1768de5a2d159035176d5299b9c476d851d",
            "policy_probe.npz": "4e0e7220668faf124d394bffb760ba246d54bed8f48cb79925cb36234257fc11",
            "stage_config.json": "5d54e6deac8ddda5c37914da97d4d59b6838a8daef52cb7bec237c49d63d24ec",
        },
        "first_any_flag": 3_801_088, "flagged_to_E": 8,        # V4 (CRIT; RF m5)
        "pan_label": "3145728",
    },
    "s80": {
        "run_id": "stance6_r10LTSB_s80", "own_seed": 80, "S": 4_194_304, "E": 5_767_168,
        "zip_rel": "01_stance/models/stage1_4194304_steps.zip",
        "zip_sha256": "51197dbd7f90e481a99214ba7bd63ebbe92804367c2e5edab988c0b4c4840096",
        "pkl_rel": "01_stance/models/stage1_vecnormalize_4194304_steps.pkl",
        "pkl_sha256": "f405244b94662bdffae9bbe1b047fda9b34bb62012bec9491ed09131ef6155cb",
        "records": {
            "eval_episodes.csv": "b677ce762e739513c18eaa35ffb7b149c0532753746259d5037356580e934d13",
            "evaluations.npz": "89e18b1006deb304b368284d6106299fa219cce9e7f1d73c3ef37f5120c39be8",
            "gate_progress.npz": "02dc43115adff9096f175cc6387f01e81cc816aca54c7070b1e8a9b6400cb65c",
            "train_episodes.csv": "fd65c2058c4b6970c5fca7a32a6461131f41a768ec792b5d625ea8a5049af8c6",
            "policy_probe.npz": "0e61919dc9a0a2a508095cec5e14a673c2d909b61420d9b6c95c74d7562b3957",
            "stage_config.json": "ec1c49a551badd146416a12d874a9ecf5db5574886e21c6df414a07d533e72c5",
        },
        "first_any_flag": 4_849_664, "flagged_to_E": 4,
        "pan_label": "4194304",
    },
}
# V4's NEGATIVE control: s60's ledger truncated at 8,388,608 (FMAN F2 row).
V4_NEGATIVE = {"run_id": "stance6_r10LTSB_s60", "truncate_at": 8_388_608,
               "eval_episodes_sha256": "1821fd2e9f65d58cb8a0dc83d096230fb04af3448f850657055de001ef8e3466"}

# ------------------------------------------------------------------------------------------------------------- seeds
SEEDS_PER_ARM = 6                               # D4 (a): k = 6
SEEDS = {                                       # T1: distinct per start and arm, in steps of 10
    ("s42", "A"): (1101, 1111, 1121, 1131, 1141, 1151),
    ("s42", "B"): (1201, 1211, 1221, 1231, 1241, 1251),
    ("s80", "A"): (1301, 1311, 1321, 1331, 1341, 1351),
    ("s80", "B"): (1401, 1411, 1421, 1431, 1441, 1451),
}
# BL 8: for j = 1..6: s42 A j, s42 B j, s80 A j, s80 B j.
QUEUE = tuple((start, arm, SEEDS[(start, arm)][j]) for j in range(SEEDS_PER_ARM)
              for start, arm in (("s42", "A"), ("s42", "B"), ("s80", "A"), ("s80", "B")))
N_CONTINUATIONS = len(QUEUE)                    # 24
WAVE_SLOTS = 3                                  # D7 (a)
WAVE_NICE = 10
MAX_CRASHES = 3                                 # a third crash of its own writes it to lost.json

# ------------------------------------------------------------------------------------------------------------- test
ALPHA_PER_SIDE = 0.025                          # D6 (a): two-sided, 0.025 per side
STOP_REASONS = ("end", "collapse", "other")     # run_meta stop_reason (BL 3)
STATUSES = ("COUNTED", "STOPPED", "LOST", "VOID", "CRASH")
EXIT_OK, EXIT_FAIL, EXIT_REFUSED, EXIT_PROVISIONAL, EXIT_CHECK_FAIL = 0, 1, 2, 3, 4
# run_ecs.py: the run finished but a post-run check failed (the code seal or the modules' origin after the run; in
# the no-optimizer-step modes the digests): status "failed".  Not a crash: the runner checks it and CHECK fails (VOID).
EXIT_POST_CHECK = 5
TORCH_THREADS = 1                               # torch intra-op threads (BL 3; HS 2.1: D-D43 recorded torch_threads)

# ------------------------------------------------------------------------------------------------------------- records
# run_meta.json (run_ecs.py; written with "x" at start, rewritten at the end).  The keys read_ecs.py --check reads:
#   schema RUN_SCHEMA; name; mode ("continuation" | "save_loaded_only" | "no_update"); status ("starting" |
#   "running" | "done" | "refused" | "crashed" | "failed" (EXIT_POST_CHECK)); exit (int; null until the end);
#   start ("s42" | "s80"); arm; seed;
#   S; E; target (E, or --until); T (num_timesteps at the end); stop_reason (STOP_REASONS; null until the end);
#   start_pair {zip, zip_sha256, pkl, pkl_sha256}; parent_records {name: sha256}; code {dir, commit, tree_digest,
#   files}; stack; criteria_sha256 (null before approval); arm_ppo (ARM_PPO_EDIT[arm], or "unchanged" for arm A);
#   ppo_applied {ent_coef, ent_coef_end, ent_coef_decay_timesteps} (the [ppo] values after the edit);
#   ent_coef_after_load;
#   digest_start / digest_end {params, adam, vecnormalize, all, n_updates, num_timesteps}; collapse {found,
#   consecutive_drops, patience, tripped}; stop_at_end {target, flag, step, calls, last_step}; train_calls (no_update
#   only); torch_threads, torch_interop_threads (after the load); wall_seconds.  During the run: heartbeat
#   {rollout_end, utc}, rewritten (atomically) at each rollout's end, before its update; on a crash: last_step (the
#   last num_timesteps StopAtEnd saw), error, traceback.  The runner reads only status, exit, stop_reason, heartbeat
#   and last_step (the early-crash rule, criteria RUN HANDLING CRASH).
RUN_SCHEMA = "ecs-run/1"
RUN_META = "run_meta.json"
CHECK_JSON = "check.json"                       # read_ecs.py --check writes <run>/check.json (BL 6)
RUN_MODES = ("continuation", "save_loaded_only", "no_update")
# <ECS>/APPROVED (JSON; written after approval, never by the build): the keys preflight reads.  criteria_sha256 and
# code_tree_digest are bare 64-hex sha256 values; tool_sha256 is {file name (as in TOOL_FILES): bare 64-hex sha256}.
# run_ecs.py compares them as strings and score_ecs.py reports any other form as PROVISIONAL (review RR-10).
APPROVED_KEYS = ("criteria_sha256", "runlog_line", "code_tree_digest", "tool_sha256")
# <ECS>/lost.json: {"schema": LOST_SCHEMA, "lost": {<name>: {"cause": "crashes" | "C6" | "R1", "detail": str,
#   "utc": str, "by": "run_waves_ecs.sh" | "read_ecs.py"}}}.  A name is the continuation's own name (no _crashed
#   suffix).  Cause "R1": the full read could not run M5 on the continuation's ledger (criteria RUN HANDLING LOST).
LOST_SCHEMA = "ecs-lost/1"

RUN_NAME_RE = re.compile(r"^(s42|s80)_(A|B)_(\d+)(?:_(crashed|void)(\d+))?$")


# ------------------------------------------------------------------------------------------------------------- lookups
def run_name(start: str, arm: str, seed: int) -> str:
    """<start>_<arm>_<seed>, for example s42_A_1101 (CRIT Outputs)."""
    return f"{start}_{arm}_{int(seed)}"


def parse_run_name(name: str) -> dict | None:
    """{"start", "arm", "seed", "suffix", "k"} for a run folder name, or None."""
    m = RUN_NAME_RE.match(os.path.basename(str(name).rstrip("/")))
    if not m:
        return None
    return {"start": m.group(1), "arm": m.group(2), "seed": int(m.group(3)), "suffix": m.group(4),
            "k": int(m.group(5)) if m.group(5) else None}


def window(start: str) -> tuple[int, int]:
    """(S, E) for a start."""
    st = STARTS[start]
    return st["S"], st["E"]


def f2_copies(start: str) -> str:
    """F2/<RUN_ID>/copies (read only)."""
    return f"{F2_ROOT}/{STARTS[start]['run_id']}/copies"


def f2_file(start: str, rel: str) -> str:
    """The F2 copy of a run-relative path: <copies>/<rel>.copy1."""
    return f"{f2_copies(start)}/{rel}.copy1"


def f2_record(start: str, name: str) -> str:
    """The F2 copy of a parent record under 01_stance/ (including stage_config.json)."""
    return f2_file(start, f"{STAGE_DIR}/{name}")


def pan_summary(start: str) -> str:
    """PAN at the start's label (V1)."""
    st = STARTS[start]
    return f"{PANELS}/{st['run_id']}/{st['pan_label']}/m1/m1_summary.json"


def periodic_names(steps: int) -> tuple[str, str]:
    """The resume path's periodic checkpoint names (HS 2.3)."""
    return f"stage1_{int(steps)}_steps.zip", f"stage1_vecnormalize_{int(steps)}_steps.pkl"


def expected_update_rows(S: int, T: int) -> int:
    """Probe rows with update_timesteps in (S, T): (T - S)/8,192 - 1 (191 at E)."""
    if (T - S) % ROLLOUT or T <= S:
        raise ValueError(f"T - S = {T - S} is not a positive multiple of {ROLLOUT}")
    return (T - S) // ROLLOUT - 1


def seed_ok(start: str, arm: str, seed: int) -> bool:
    return (start, arm) in SEEDS and int(seed) in SEEDS[(start, arm)]


def arm_of(start: str, seed: int) -> str | None:
    for (st, arm), seeds in SEEDS.items():
        if st == start and int(seed) in seeds:
            return arm
    return None


def derived_seeds(seed: int) -> tuple[int, ...]:
    """The distinct seeds a continuation uses (T1; CHK 5's 120 = 24 x 5): the model and training env 0 (seed),
    training envs 1-3 (seed + rank) and the eval env (seed + 1000)."""
    return (*(int(seed) + r for r in range(N_ENVS)), int(seed) + EVAL_SEED_OFFSET)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stack_versions() -> dict:
    """The running interpreter's versions from package metadata (nothing is imported)."""
    out = {"python": platform.python_version()}
    for key in ("torch", "numpy", "gymnasium", "stable_baselines3", "mujoco"):
        try:
            out[key] = importlib.metadata.version(key)
        except importlib.metadata.PackageNotFoundError:
            out[key] = None
    return out


def check_stack() -> list[str]:
    """Differences between the running versions and STACK; empty means equal."""
    got = stack_versions()
    return [f"{k}: running {got.get(k)} != pinned {v}" for k, v in STACK.items() if got.get(k) != v]


def check_pins() -> list[str]:
    """Differences between M1/M5 (and their commons) and their pins; empty means equal."""
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
