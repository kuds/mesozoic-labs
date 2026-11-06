"""Run walking pilot 4's Colab steps locally on a clone of the pilot branch: the notebook smoke (not a judging tool).

Scratch, never merged; frozen by sha256 before the smoke. Walking pilot 3's nb_pilot_runner3.py (7f533128...) adapted
to pilot 4 (criteria_pilot4 v3, "TOOL PLAN", the smoke runner, and "Pre-launch checks", the notebook smoke): the
Colab steps' cells, with the five MANUAL lines, the guard line typed as the chain-loop cell's first line, "Run all"
over the code cells, which the guard stops at the chain loop, then the manual cell alone, then section 9's cell; one
namespace (one kernel); stdout and stderr to logs; record.json. What pilot 4 changes against pilot 3's runner:

  The pilot-4 cells    Cells are found by their banners on the pilot-4 branch's notebook (setup 3, config 6, resume
                       18, chain 20, manual 22, disconnect 32 at 0e6a57b); the indices found are recorded.
  The local parent pair  The Colab steps read each seed's own pilot-3 8M pair from Drive by path (v3 Q4). params
                       "path_map" lists [colab_prefix, local_prefix] pairs applied to the MANUAL lines (the clone root
                       for /content/mesozoic-labs is always added). In the launch phase the runner refuses, before any
                       cell, unless MANUAL_NODE is "locomotion", MANUAL_LOAD_MODE "initialize_next_stage",
                       MANUAL_TIMESTEPS None, MANUAL_LOAD_PATH a string literal naming <dir>/stage2_8000000_steps,
                       and the zip and its sidecar are on disk with the sha256s of v3's parent table for SEED.
  MANUAL_VECNORM_PATH explicit  It must be a string literal naming <dir>/stage2_vecnormalize_8000000_steps.pkl (the
                       default MANUAL_LOAD_PATH + "_vecnorm.pkl" names no periodic sidecar), else refused.
  --cadence-override 32768  In memory only, the locomotion stage's [curriculum] eval_freq and checkpoint_freq are set
                       to 32,768 in STAGE_CONFIGS right after the cell that builds it; the runner refuses unless they
                       read 65,536 / 250,000 (with eval_episodes 10, max_checkpoints 0) before, and unless the stage's
                       hyperparameters_sha256 is the same before and after and equals params
                       "expect_hyperparameters_sha256" (default pilot 4's PPO digest fbc94bef...). Any other value
                       than 32,768 is refused (v3: "the cadence override 65,536 / 250,000 -> 32,768 in memory").
  The expected entry lines  After the cells, runner.stdout.log of this invocation is read for the lines the setup
                       checks measured (build/check and postfix/code runs, nb/runner.stdout.log): required, in the
                       launch phase, "Running locally; using this checkout." (locally the setup cell prints this; the
                       Colab "Repository ref: ...; commit: ..." line is Colab-only), "Quick test mode: <QUICK_TEST>",
                       "Behavior recipe: walk", "In-run K3 panel: off", "Critic phase: off", "Run directory:
                       .../<RUN_ID> (new run: RUN_ID names no existing run)", "Node locomotion (stage 2): Locomotion
                       (ppo)", "Timesteps: 2,000,000" (50,000 under QUICK_TEST, the notebook's quick budget, measured
                       here with learn() stopped), "VecNormalize loaded from: <the explicit MANUAL_VECNORM_PATH>";
                       forbidden "Warm start: policy log_std set" and "Warm start: dropped the critic's"; and, once
                       learn() has run, the first "Eval num_timesteps=<N>," with N = the override (else 65,536). Any
                       other "Warm start:" line is listed (not a failure). A required line missing, or a forbidden one
                       present, makes the exit status 1; the record lists each.
  --stop-at-learn      A pre-launch check that trains nothing (launch phase only; refused with --phase resume and
                       with --stop-after-pair): before the cells, SB3's learn() (BaseAlgorithm,
                       OnPolicyAlgorithm, OffPolicyAlgorithm, PPO, SAC, A2C and the repository's MirrorKLPPO) is
                       replaced by a raise of LearnBlocked (a BaseException) and PPO/SAC train(), collect_rollouts(),
                       the torch optimizers' step() and backward() by guards that count and raise. The manual cell then
                       ends at learn() (the expected end; section 9 is not run) and the entry lines before learn() are
                       checked. The record names every stop, every guard and every hit (a hit makes the status 1).
  environments         The clone root is put first on sys.path; the runner refuses (exit 2) when `environments`
                       would resolve elsewhere, records the resolved file, and fails when the cells imported it from
                       anywhere but the clone. params "expect_head" (optional): the clone's HEAD must equal it, and
                       its tracked files must be unmodified (git status --porcelain --untracked-files=no), else
                       refused.
  QUICK_TEST only      (fix stage) Unless --stop-at-learn or --plan-only is given, params config QUICK_TEST must be
                       true (the JSON literal), else the runner refuses (exit 2) before reading the notebook: the only
                       training v3 permits before launch is the QUICK_TEST CPU smoke ("Pre-launch checks"), so no
                       launch or resume of the full 2,000,000-step stage can run here.
Kept from pilot 3: --phase resume (the RESUME cell: RESUME_STAGE = <params resume_stage>, then Run all, which the
guard stops after the RESUME cell trained; then section 9's cell; the manual cell is not run), --stop-after-pair N
(os._exit(75) once the periodic pair at N is on disk, no final save), --plan-only (cells_edited.json, nothing runs),
the Colab-host refusal, banners not indices, every config name on exactly one line.

    cd CLONE && OMP_NUM_THREADS=1 nice -n 10 python -I TOOLS/nb_pilot_runner4.py --params PARAMS.json --out LOGDIR
        [--phase launch|resume] [--cadence-override 32768] [--stop-after-pair 32768] [--stop-at-learn] [--plan-only]

PARAMS (JSON): {"repo_ref": "claude/pilot-walk-r10-p4", "config": {"SPECIES": "tyrannosaurus_rex", ..., "SEED": 42,
"QUICK_TEST": true, "RUN_ID": "pilot4_walk_s42"}, "manual": [the five lines as the Colab steps type them],
"path_map": [["/content/drive/MyDrive/mesozoic-labs/logs/tyrannosaurus_rex/anatomical/ppo", "LOCAL"]], "guard_line":
"raise RuntimeError(...)", "resume_stage": "locomotion", "expect_head": "0e6a57bb...",
"expect_hyperparameters_sha256": "sha256:fbc94bef..."}. Exit status: 0 when every cell ran as the phase expects (the
guard's RuntimeError at the chain loop is expected; under --stop-at-learn the manual cell's LearnBlocked too) and every
entry line is as expected, 1 otherwise, 2 for a refusal before any cell.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

TOOL = "nb_pilot_runner4"
STOP_EXIT = 75
CONTENT_ROOT = "/content/mesozoic-labs"
MARKERS = {  # cell role -> a line its source holds (the notebook's banners)
    "setup": "# Install dependencies (Colab auto-detected",
    "config": "# ===== SPECIES SELECTION =====",
    "resume": "# ===== RESUME AN INTERRUPTED STAGE",
    "chain": "# ===== BEHAVIOR CHAIN LOOP =====",
    "manual": "# ===== MANUAL SINGLE NODE",
    "disconnect": "disconnect_runtime(",
}
MANUAL_NAMES = ("MANUAL_NODE", "MANUAL_LOAD_PATH", "MANUAL_VECNORM_PATH", "MANUAL_LOAD_MODE", "MANUAL_TIMESTEPS")
STAGE_FOR_OVERRIDE = "locomotion"
CADENCE_OVERRIDE = 32_768  # v3: "the cadence override 65,536 / 250,000 -> 32,768 in memory"
CADENCE_BEFORE = {"eval_freq": 65_536, "checkpoint_freq": 250_000, "eval_episodes": 10, "max_checkpoints": 0}
FIRST_EVAL_DEFAULT = 65_536  # v3 "Run handling": a first "Eval num_timesteps=65536" line at the launch
PILOT4_PPO_SHA256 = "sha256:fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62"
PARENT_STEM = "stage2_8000000_steps"
PARENT_SIDECAR = "stage2_vecnormalize_8000000_steps.pkl"
#: v3 "Parent and entry": each seed's own pilot-3 8M pair (zip, sidecar).
PARENTS = {
    42: ("sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
         "sha256:b9e6dee894e549ca4e844f7ef3c162e0f52609eacede7b1790e5d649e77d9cce"),
    52: ("sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
         "sha256:cf29635feb8c4040ea9a9bb08255b97de60673365dce107445c115aa2927f37a"),
    60: ("sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2",
         "sha256:56e58a536d08198fdafbde3310e05116ff65e7cd471f536016858f87f9c32dc9"),
}
#: The Timesteps line the manual cell prints: the TOML budget, or the notebook's QUICK_TEST budget (measured with
#: learn() stopped on the pilot-4 clone, out/readouts; pilot 3's smoke printed the same 50,000).
TIMESTEPS_LINE = {False: "Timesteps: 2,000,000", True: "Timesteps: 50,000"}
FORBIDDEN_LINES = ("Warm start: policy log_std set", "Warm start: dropped the critic's")
RECORD: dict = {}
RECORD_PATH: Path | None = None


class RunnerRefusal(Exception):
    pass


class LearnBlocked(BaseException):
    """learn() was called under --stop-at-learn: the set-up path is complete; nothing rolled out or updated."""


class TrainingBlocked(BaseException):
    """A training primitive was called under --stop-at-learn (must never happen)."""


def cell_source(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else str(src)


def sha256_file(path: "str | Path") -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find_cells(nb: dict) -> dict:
    """{role: index} for every role in MARKERS: the one code cell whose source holds the marker (the disconnect cell:
    the last code cell that starts with it)."""
    found = {}
    code = [(i, cell_source(c)) for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"]
    for role, marker in MARKERS.items():
        if role == "disconnect":
            hits = [i for i, s in code if s.lstrip().startswith(marker)]
            hits = hits[-1:]
        else:
            hits = [i for i, s in code if marker in s]
        if len(hits) != 1:
            raise RunnerRefusal(f"the {role} cell (marker {marker!r}) is found {len(hits)} times, not once")
        found[role] = hits[0]
    return found


def replace_assignment(src: str, name: str, new_line: str) -> str:
    """Replace the one line that assigns *name* at the start of a line (``NAME = ...``) by *new_line*."""
    pattern = re.compile(rf"^{re.escape(name)}\s*=.*$", re.M)
    hits = pattern.findall(src)
    if len(hits) != 1:
        raise RunnerRefusal(f"{name} is assigned on {len(hits)} lines of its cell, not one")
    return pattern.sub(lambda _m: new_line, src, count=1)


def path_maps(params: dict, clone: Path) -> list[tuple[str, str]]:
    """The [colab_prefix, local_prefix] pairs applied to the MANUAL lines: params' own first, then the clone root."""
    maps = []
    for pair in params.get("path_map") or []:
        if not (isinstance(pair, (list, tuple)) and len(pair) == 2 and all(isinstance(x, str) and x for x in pair)):
            raise RunnerRefusal(f"path_map entry {pair!r} is not [colab_prefix, local_prefix]")
        maps.append((pair[0], pair[1]))
    maps.append((CONTENT_ROOT, str(clone)))
    return maps


def map_line(line: str, maps: list[tuple[str, str]]) -> str:
    for old, new in maps:
        line = line.replace(old, new)
    return line


def edit_cells(nb: dict, params: dict, clone: Path, phase: str) -> tuple[dict, dict, list]:
    """The Colab steps' edits, on a copy of the notebook's cells: (sources by index, roles, edits applied)."""
    roles = find_cells(nb)
    sources = {i: cell_source(c) for i, c in enumerate(nb["cells"]) if c.get("cell_type") == "code"}
    edits = []
    i = roles["setup"]
    if params.get("repo_ref"):
        sources[i] = replace_assignment(sources[i], "REPO_REF", f"REPO_REF = {params['repo_ref']!r}  # {TOOL}")
        edits.append(("setup", "REPO_REF", params["repo_ref"]))
    i = roles["config"]
    for name, value in params["config"].items():
        sources[i] = replace_assignment(sources[i], name, f"{name} = {value!r}  # {TOOL} (Colab steps)")
        edits.append(("config", name, value))
    i = roles["manual"]
    maps = path_maps(params, clone)
    manual = {}
    for line in params["manual"]:
        name = line.split("=", 1)[0].strip()
        if name not in MANUAL_NAMES:
            raise RunnerRefusal(f"manual line {line!r} sets {name!r}, not one of {MANUAL_NAMES}")
        manual[name] = map_line(line, maps)
    if sorted(manual) != sorted(MANUAL_NAMES):
        raise RunnerRefusal(f"params.manual sets {sorted(manual)}, not the five {sorted(MANUAL_NAMES)}")
    for name in MANUAL_NAMES:
        sources[i] = replace_assignment(sources[i], name, manual[name])
        edits.append(("manual", name, manual[name]))
    i = roles["chain"]
    guard = params["guard_line"]
    if "\n" in guard or not guard.startswith("raise RuntimeError("):
        raise RunnerRefusal("the guard line must be one line that raises RuntimeError")
    sources[i] = guard + "\n" + sources[i]
    edits.append(("chain", "guard line", guard))
    if phase == "resume":
        i = roles["resume"]
        sources[i] = replace_assignment(sources[i], "RESUME_STAGE",
                                        f"RESUME_STAGE = {params['resume_stage']!r}  # {TOOL} (resume)")
        edits.append(("resume", "RESUME_STAGE", params["resume_stage"]))
    for idx, src in sources.items():
        compile(src, f"<cell {idx}>", "exec")
    return sources, roles, edits


def manual_values(edits: list) -> dict:
    """The edited MANUAL lines' right-hand sides as Python literals ({name: value}); a non-literal is kept as
    ('<expression>', source) so the checks can name it."""
    out = {}
    for role, name, line in edits:
        if role != "manual":
            continue
        try:
            node = ast.parse(line).body[0]
            value = node.value if isinstance(node, ast.Assign) else None
        except SyntaxError:
            value = None
        if value is None:
            out[name] = ("<expression>", line)
            continue
        try:
            out[name] = ast.literal_eval(value)
        except ValueError:
            out[name] = ("<expression>", ast.unparse(value))
    return out


def parent_pair_problems(values: dict, seed: "int | None") -> tuple[list[str], dict]:
    """Launch-phase checks of the MANUAL lines and the local parent pair (v3 "Run handling", Q4, Q5)."""
    problems, info = [], {}
    if values.get("MANUAL_NODE") != "locomotion":
        problems.append(f"MANUAL_NODE is {values.get('MANUAL_NODE')!r}, not 'locomotion'")
    if values.get("MANUAL_LOAD_MODE") != "initialize_next_stage":
        problems.append(f"MANUAL_LOAD_MODE is {values.get('MANUAL_LOAD_MODE')!r}, not 'initialize_next_stage'")
    if values.get("MANUAL_TIMESTEPS") is not None:
        problems.append(f"MANUAL_TIMESTEPS is {values.get('MANUAL_TIMESTEPS')!r}, not None (the TOML budget)")
    load, vec = values.get("MANUAL_LOAD_PATH"), values.get("MANUAL_VECNORM_PATH")
    if not isinstance(load, str):
        problems.append(f"MANUAL_LOAD_PATH is not a string literal ({load!r})")
    if not isinstance(vec, str):
        problems.append(f"MANUAL_VECNORM_PATH is not an explicit string literal ({vec!r}); the default "
                        "MANUAL_LOAD_PATH + '_vecnorm.pkl' names no periodic sidecar")
    if seed not in PARENTS:
        problems.append(f"SEED {seed!r} is not one of the scored seeds {sorted(PARENTS)}")
    if problems:
        return problems, info
    zip_path, pkl_path = Path(load + ".zip"), Path(vec)
    info = {"zip": str(zip_path), "vecnorm": str(pkl_path)}
    if Path(load).name != PARENT_STEM:
        problems.append(f"MANUAL_LOAD_PATH {load!r} does not name {PARENT_STEM} (the 8M periodic pair)")
    if pkl_path != zip_path.parent / PARENT_SIDECAR:
        problems.append(f"MANUAL_VECNORM_PATH {vec!r} is not {zip_path.parent / PARENT_SIDECAR} (the pair's sidecar)")
    for label, path, want in (("zip", zip_path, PARENTS[seed][0]), ("vecnorm", pkl_path, PARENTS[seed][1])):
        if not path.is_file():
            problems.append(f"the parent {label} {path} is not on disk")
            continue
        got = sha256_file(path)
        info[f"{label}_sha256"] = got
        if got != want:
            problems.append(f"the parent {label} {path} is not s{seed}'s 8M pair ({got}, want {want})")
    return problems, info


def guard_message(guard_line: str) -> str:
    """The message the guard line raises (the string literal inside RuntimeError(...))."""
    node = ast.parse(guard_line).body[0]
    assert isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call)
    return str(ast.literal_eval(node.exc.args[0]))


def apply_cadence_override(ns: dict, value: int, digest=None, expect_digest: str = PILOT4_PPO_SHA256) -> dict:
    """Set the locomotion stage's curriculum eval_freq and checkpoint_freq to *value* in ns["STAGE_CONFIGS"], after
    checking they read CADENCE_BEFORE and that the PPO digest is *expect_digest* before and after."""
    if int(value) != CADENCE_OVERRIDE:
        raise RunnerRefusal(f"--cadence-override {value}: pilot 4's smoke override is {CADENCE_OVERRIDE:,} only")
    ref = ns["MANIFEST"].resolve(STAGE_FOR_OVERRIDE).reference
    cfg = ns["STAGE_CONFIGS"][ref]
    cur = cfg.setdefault("curriculum_kwargs", {})
    if digest is None:
        from environments.shared.config import hyperparameters_sha256 as digest
    before_digest = digest(cfg, "ppo")
    before = {k: cur.get(k) for k in CADENCE_BEFORE}
    rec = {"stage_reference": ref, "before": before, "hyperparameters_sha256_before": before_digest}
    if before != CADENCE_BEFORE:
        raise RunnerRefusal(f"the locomotion cadence reads {before}, not the committed {CADENCE_BEFORE}: {rec}")
    cur["eval_freq"] = int(value)
    cur["checkpoint_freq"] = int(value)
    after_digest = digest(cfg, "ppo")
    rec.update(after={k: cur.get(k) for k in CADENCE_BEFORE}, hyperparameters_sha256_after=after_digest)
    if before_digest != after_digest:
        raise RunnerRefusal(f"the cadence override moved hyperparameters_sha256: {rec}")
    if before_digest != expect_digest:
        raise RunnerRefusal(f"the locomotion PPO digest is {before_digest}, not {expect_digest}: {rec}")
    return rec


def expected_entry_lines(params: dict, phase: str, vecnorm: "str | None") -> dict:
    """{"required": [(label, regex)], "forbidden": [prefix]} for this phase (the setup checks' measured lines)."""
    cfg = params.get("config", {})
    quick = bool(cfg.get("QUICK_TEST", False))
    run_id = str(cfg.get("RUN_ID", ""))
    req = [
        ("setup (local)", r"^Running locally; using this checkout\."),
        ("quick test", rf"^Quick test mode: {quick}$"),
        ("behavior", r"^Behavior recipe: walk$"),
        ("K3 panel", r"^In-run K3 panel: off$"),
        ("critic phase", r"^Critic phase: off$"),
        ("run directory", rf"^Run directory: .*/{re.escape(run_id)} \(new run: RUN_ID names no existing run\)$"
         if phase == "launch" else rf"^Run directory: .*/{re.escape(run_id)} "),
    ]
    if phase == "launch":
        req += [("node", r"^Node locomotion \(stage 2\): Locomotion \(ppo\)$"),
                ("timesteps", rf"^{re.escape(TIMESTEPS_LINE[quick])}$")]
        if vecnorm:
            req.append(("vecnormalize", rf"^VecNormalize loaded from: {re.escape(vecnorm)}$"))
    else:
        req.append(("resuming", r"^Resuming 'locomotion' \(stage 2\) from "))
    return {"required": req, "forbidden": list(FORBIDDEN_LINES)}


def check_entry_lines(text: str, expected: dict, *, learn_ran: bool, first_eval: int) -> dict:
    """Which required lines are found, which forbidden ones appear, other 'Warm start:' lines, and (once learn()
    ran) whether the first 'Eval num_timesteps=' line is at *first_eval*."""
    lines = text.splitlines()
    found = {label: next((ln for ln in lines if re.search(rx, ln)), None) for label, rx in expected["required"]}
    forbidden = [ln for ln in lines if any(ln.startswith(p) for p in expected["forbidden"])]
    other_warm = [ln for ln in lines if ln.startswith("Warm start:") and ln not in forbidden]
    evals = [ln for ln in lines if ln.startswith("Eval num_timesteps=")]
    rec = {"required": found, "missing": sorted(k for k, v in found.items() if v is None),
           "forbidden_present": forbidden, "other_warm_start_lines": other_warm}
    if learn_ran:
        rec["first_eval_line"] = evals[0] if evals else None
        rec["first_eval_ok"] = bool(evals and evals[0].startswith(f"Eval num_timesteps={first_eval},"))
    else:
        rec["first_eval_line"] = "not reached (learn() did not run)"
        rec["first_eval_ok"] = None
    rec["ok"] = not rec["missing"] and not forbidden and rec["first_eval_ok"] in (None, True)
    return rec


def write_record() -> None:
    if RECORD_PATH is not None:
        RECORD_PATH.write_text(json.dumps(RECORD, indent=1, default=str))


def install_stop_after_pair(callback_cls, steps: int, prefix: str = "stage2") -> None:
    """Patch *callback_cls*._on_step (SB3's CheckpointCallback) to end the process once the pair at *steps* exists."""
    original = callback_cls._on_step

    def _on_step(self):
        result = original(self)
        zip_path = Path(self.save_path) / f"{prefix}_{steps}_steps.zip"
        pkl_path = Path(self.save_path) / f"{prefix}_vecnormalize_{steps}_steps.pkl"
        if getattr(self, "name_prefix", None) == prefix and zip_path.is_file() and pkl_path.is_file():
            RECORD["stopped_after_pair"] = {"steps": steps, "zip": str(zip_path), "vecnormalize": str(pkl_path),
                                            "num_timesteps": int(getattr(self, "num_timesteps", -1)),
                                            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            RECORD["exit"] = STOP_EXIT
            print(f"{TOOL}: simulated runtime loss after the {steps:,} pair ({zip_path.name}); exiting "
                  f"{STOP_EXIT} with no final save", flush=True)
            write_record()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(STOP_EXIT)
        return result

    callback_cls._on_step = _on_step


GUARD_HITS: list = []


def install_learn_stops() -> dict:
    """--stop-at-learn: learn() raises LearnBlocked; the training primitives raise TrainingBlocked (each hit kept)."""
    import stable_baselines3 as sb3
    import torch
    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
    from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

    classes = [BaseAlgorithm, OnPolicyAlgorithm, OffPolicyAlgorithm, sb3.PPO, sb3.SAC, sb3.A2C]
    try:
        from environments.shared.mirror_kl_ppo import MirrorKLPPO

        classes.append(MirrorKLPPO)
    except ImportError:
        MirrorKLPPO = None  # noqa: N806 - a tree without it (the synthetic tests' clones)

    def make_stop(cls):
        def learn(self, *args, **kwargs):
            RECORD.setdefault("learn_called", []).append(
                {"class": f"{type(self).__module__}.{type(self).__qualname__}",
                 "total_timesteps": kwargs.get("total_timesteps", args[0] if args else None),
                 "reset_num_timesteps": kwargs.get("reset_num_timesteps")})
            raise LearnBlocked(f"learn() stopped by {TOOL} --stop-at-learn ({cls.__qualname__}.learn, nothing trained)")
        return learn

    def make_guard(owner, name):
        def guard(*args, **kwargs):
            GUARD_HITS.append(f"{owner}.{name}")
            raise TrainingBlocked(f"{owner}.{name} is blocked by {TOOL} --stop-at-learn")
        return guard

    stops, guards = [], []
    for cls in classes:
        if "learn" in cls.__dict__:
            setattr(cls, "learn", make_stop(cls))
            stops.append(f"{cls.__module__}.{cls.__qualname__}.learn")
    pairs = [(sb3.PPO, "train"), (sb3.SAC, "train"), (OnPolicyAlgorithm, "collect_rollouts"),
             (OffPolicyAlgorithm, "collect_rollouts"), (torch.optim.Adam, "step"), (torch.optim.AdamW, "step"),
             (torch.optim.SGD, "step"), (torch.optim.Optimizer, "step"), (torch.Tensor, "backward")]
    if MirrorKLPPO is not None:
        pairs.append((MirrorKLPPO, "train"))
    for cls, name in pairs:
        if name in cls.__dict__:
            setattr(cls, name, make_guard(cls.__qualname__, name))
            guards.append(f"{cls.__module__}.{cls.__qualname__}.{name}")
    torch.autograd.backward = make_guard("torch.autograd", "backward")
    guards.append("torch.autograd.backward")
    return {"learn_stops": stops, "guards": guards}


class Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


def install_ipython_stand_in() -> bool:
    """A minimal IPython.display (Video, display) when IPython is not installed; True when installed here."""
    try:
        import IPython.display  # noqa: F401
        return False
    except ImportError:
        pass
    ipy = types.ModuleType("IPython")
    disp = types.ModuleType("IPython.display")

    class Video:
        def __init__(self, data=None, **kw):
            self.data, self.kw = data, kw

        def __repr__(self):
            return f"Video({self.data!r})"

    def display(*objs, **kw):
        for o in objs:
            print(f"[{TOOL} display] {o!r}", flush=True)

    disp.Video, disp.display = Video, display
    ipy.display = disp
    sys.modules["IPython"], sys.modules["IPython.display"] = ipy, disp
    return True


def run_cell(idx: int, src: str, ns: dict) -> dict:
    t0 = time.time()
    print(f"\n##### {TOOL}: cell {idx} start {time.strftime('%H:%M:%S', time.gmtime())} UTC", flush=True)
    try:
        exec(compile(src, f"<cell {idx}>", "exec"), ns)
        res = {"cell": idx, "ok": True, "seconds": round(time.time() - t0, 1)}
    except BaseException as exc:  # noqa: BLE001 - a cell's exception is the notebook's result, recorded
        traceback.print_exc()
        res = {"cell": idx, "ok": False, "seconds": round(time.time() - t0, 1), "exception": type(exc).__name__,
               "message": str(exc)[:2000]}
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            res["fatal"] = True
    print(f"##### {TOOL}: cell {idx} {'ok' if res['ok'] else 'RAISED ' + res['exception']} "
          f"({res['seconds']} s)", flush=True)
    return res


def environments_origin(clone: Path) -> "str | None":
    spec = importlib.util.find_spec("environments")
    return None if spec is None else str(Path(spec.origin).resolve()) if spec.origin else None


def git(clone: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=clone, text=True,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()


def main(argv=None) -> int:  # noqa: C901 - the procedure of the Colab steps, in order
    global RECORD_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--params", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clone", default=".", help="the clone's root (default: the working directory)")
    ap.add_argument("--notebook", default="notebooks/sb3_training.ipynb")
    ap.add_argument("--phase", choices=("launch", "resume"), default="launch")
    ap.add_argument("--cadence-override", type=int, default=None)
    ap.add_argument("--stop-after-pair", type=int, default=None)
    ap.add_argument("--stop-at-learn", action="store_true", help="stop at learn(): a set-up check, nothing trains")
    ap.add_argument("--plan-only", action="store_true")
    a = ap.parse_args(argv)
    clone = Path(a.clone).resolve()
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    RECORD_PATH = out / "record.json"
    params = json.loads(Path(a.params).read_text())
    nb_path = clone / a.notebook
    RECORD.clear()
    RECORD.update({"tool": TOOL, "tool_sha256": sha256_file(__file__),
                   "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "argv": sys.argv[1:] if argv is None else argv, "clone": str(clone), "notebook": str(nb_path),
                   "phase": a.phase, "params": params, "python": platform.python_version(), "cells": [],
                   "stop_at_learn": bool(a.stop_at_learn)})
    try:
        if os.path.exists("/content") or "COLAB_GPU" in os.environ or "COLAB_RELEASE_TAG" in os.environ:
            raise RunnerRefusal("this host looks like Colab (/content or a COLAB_ variable): the setup cell would take "
                                "its Colab branch")
        if a.cadence_override is not None and a.cadence_override != CADENCE_OVERRIDE:
            raise RunnerRefusal(f"--cadence-override {a.cadence_override}: pilot 4's smoke override is "
                                f"{CADENCE_OVERRIDE:,} only")
        if a.stop_after_pair is not None and a.cadence_override is not None and a.stop_after_pair % a.cadence_override:
            raise RunnerRefusal(f"--stop-after-pair {a.stop_after_pair} is not a multiple of the cadence "
                                f"{a.cadence_override}: no pair would be written there")
        if a.stop_at_learn and a.stop_after_pair is not None:
            raise RunnerRefusal("--stop-at-learn stops before any pair is written; --stop-after-pair cannot apply")
        if a.stop_at_learn and a.phase != "launch":
            raise RunnerRefusal("--stop-at-learn checks the launch's entry (the manual cell); the resume phase needs "
                                "the smoke's own pair and trains")
        quick_test = (params.get("config") or {}).get("QUICK_TEST")
        if quick_test is not True and not a.stop_at_learn and not a.plan_only:
            raise RunnerRefusal(f"config QUICK_TEST is {quick_test!r}, not true, and neither --stop-at-learn nor "
                                "--plan-only is given: this runner trains only the pre-registered QUICK_TEST CPU smoke "
                                "(v3 'Pre-launch checks')")
        nb = json.loads(nb_path.read_text())
        RECORD["notebook_sha256"] = sha256_file(nb_path)
        try:
            RECORD["clone_head"] = git(clone, "rev-parse", "HEAD")
            RECORD["clone_status_before"] = git(clone, "status", "--porcelain").splitlines()
            tracked = git(clone, "status", "--porcelain", "--untracked-files=no")
        except (OSError, subprocess.CalledProcessError) as exc:
            RECORD["clone_head"], tracked = f"unreadable: {exc}", None
        if params.get("expect_head"):
            if RECORD["clone_head"] != params["expect_head"]:
                raise RunnerRefusal(f"the clone is at {RECORD['clone_head']}, not the expected head "
                                    f"{params['expect_head']}")
            if tracked:
                raise RunnerRefusal("the clone has local edits to tracked files (git status --porcelain "
                                    "--untracked-files=no is not empty)")
        sources, roles, edits = edit_cells(nb, params, clone, a.phase)
        RECORD.update(roles=roles, edits=edits)
        values = manual_values(edits)
        RECORD["manual_values"] = {k: (v if not isinstance(v, tuple) else f"<expression> {v[1]}")
                                   for k, v in values.items()}
        vecnorm = None
        if a.phase == "launch":
            problems, info = parent_pair_problems(values, params.get("config", {}).get("SEED"))
            RECORD["parent_pair"] = info
            if problems:
                raise RunnerRefusal("; ".join(problems))
            vecnorm = info["vecnorm"]
        RECORD["expected_entry_lines"] = expected_entry_lines(params, a.phase, vecnorm)
        (out / "cells_edited.json").write_text(json.dumps({str(k): v for k, v in sorted(sources.items())}, indent=1))
        if str(clone) not in sys.path:
            sys.path.insert(0, str(clone))
        origin = environments_origin(clone)
        RECORD["environments_resolves_to"] = origin
        if origin is None or not origin.startswith(str(clone) + os.sep):
            raise RunnerRefusal(f"environments would be imported from {origin}, not from the clone {clone}")
    except RunnerRefusal as exc:
        RECORD.update(refused=str(exc), exit=2)
        write_record()
        print(f"{TOOL}: refusing: {exc}", flush=True)
        return 2
    if a.plan_only:
        RECORD["exit"] = 0
        write_record()
        print(f"{TOOL}: plan only; cells {roles}; {len(edits)} edits; wrote {out / 'cells_edited.json'}")
        return 0

    stdout_path = out / "runner.stdout.log"
    offset = stdout_path.stat().st_size if stdout_path.exists() else 0
    stdout_log = open(stdout_path, "a", encoding="utf-8")
    stderr_log = open(out / "runner.stderr.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, stdout_log), Tee(sys.__stderr__, stderr_log)
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(clone)] + [p for p in os.environ.get("PYTHONPATH", "").split(
        os.pathsep) if p and p != str(clone)])
    os.chdir(clone)
    RECORD["ipython_stand_in"] = install_ipython_stand_in()
    if a.stop_after_pair is not None:
        from stable_baselines3.common.callbacks import CheckpointCallback

        install_stop_after_pair(CheckpointCallback, a.stop_after_pair)
    if a.stop_at_learn:
        RECORD["learn_stops"] = install_learn_stops()
    RECORD["stop_after_pair_requested"] = a.stop_after_pair
    RECORD["cadence_override_requested"] = a.cadence_override

    def _no_ipython():
        raise RuntimeError(f"{TOOL}: get_ipython() is a Colab-only call here")

    ns: dict = {"__name__": "__main__", "get_ipython": _no_ipython}
    expected_guard = guard_message(params["guard_line"])
    run_all = sorted(i for i in sources if i <= roles["chain"])
    status = 0
    stopped_at = None
    for i in run_all:
        res = run_cell(i, sources[i], ns)
        RECORD["cells"].append(res)
        write_record()
        if res["ok"] and a.cadence_override is not None and "override" not in RECORD and "STAGE_CONFIGS" in ns \
                and "MANIFEST" in ns:
            try:
                RECORD["override"] = apply_cadence_override(
                    ns, a.cadence_override,
                    expect_digest=params.get("expect_hyperparameters_sha256", PILOT4_PPO_SHA256))
                print(f"{TOOL}: cadence override applied in memory: {RECORD['override']}", flush=True)
            except RunnerRefusal as exc:
                RECORD["override_refused"] = str(exc)
                print(f"{TOOL}: {exc}", flush=True)
                status = 1
                break
        if not res["ok"]:
            stopped_at = i
            break
    guard_ok = (stopped_at == roles["chain"] and RECORD["cells"][-1].get("exception") == "RuntimeError"
                and RECORD["cells"][-1].get("message") == expected_guard)
    RECORD["run_all"] = {"stopped_at": stopped_at, "stopped_by_the_guard": guard_ok}
    if not guard_ok:
        status = 1
        print(f"{TOOL}: Run all did not stop at the guard (stopped at {stopped_at})", flush=True)
    if a.cadence_override is not None and "override" not in RECORD:
        status = 1
        RECORD.setdefault("override_refused", "STAGE_CONFIGS and MANIFEST never appeared in the namespace")
    manual_done = False
    if status == 0 and a.phase == "launch":
        res = run_cell(roles["manual"], sources[roles["manual"]], ns)
        RECORD["cells"].append(res)
        write_record()
        if a.stop_at_learn:
            RECORD["manual_cell_stopped_at_learn"] = bool(not res["ok"] and res.get("exception") == "LearnBlocked")
            if not RECORD["manual_cell_stopped_at_learn"]:
                status = 1
        elif not res["ok"]:
            status = 1
        else:
            manual_done = True
    if status == 0 and not a.stop_at_learn:
        res = run_cell(roles["disconnect"], sources[roles["disconnect"]], ns)
        RECORD["cells"].append(res)
        if not res["ok"]:
            status = 1
    if a.stop_at_learn:
        RECORD["guard_hits"] = list(GUARD_HITS)
        if GUARD_HITS:
            status = 1
    mod = sys.modules.get("environments")
    imported = str(Path(mod.__file__).resolve()) if mod is not None and getattr(mod, "__file__", None) else None
    RECORD["environments_imported_from"] = imported
    if imported is not None and not imported.startswith(str(clone) + os.sep):
        status = 1
        print(f"{TOOL}: environments was imported from {imported}, not from the clone {clone}", flush=True)
    try:
        cur = ns["STAGE_CONFIGS"][ns["MANIFEST"].resolve(STAGE_FOR_OVERRIDE).reference]["curriculum_kwargs"]
        RECORD["locomotion_curriculum_at_end"] = {k: cur.get(k) for k in ("eval_freq", "checkpoint_freq",
                                                                          "eval_episodes", "max_checkpoints",
                                                                          "warm_start_log_std", "warmup_timesteps",
                                                                          "ramp_start_value", "timesteps",
                                                                          "initialize_from_own_checkpoint")}
    except Exception as exc:  # noqa: BLE001 - report only
        RECORD["locomotion_curriculum_at_end"] = f"unreadable: {type(exc).__name__}: {exc}"
    RECORD["run_dir"] = str(ns.get("RUN_DIR"))
    sys.stdout.flush()
    stdout_log.flush()
    text = stdout_path.read_text(encoding="utf-8")[offset:]
    first_eval = a.cadence_override if a.cadence_override is not None else FIRST_EVAL_DEFAULT
    entry = check_entry_lines(text, RECORD["expected_entry_lines"], learn_ran=manual_done, first_eval=first_eval)
    if a.phase == "launch" and stopped_at == roles["chain"] and RECORD["run_all"]["stopped_by_the_guard"]:
        RECORD["entry_lines"] = entry
        if not entry["ok"]:
            status = 1
            print(f"{TOOL}: entry lines not as expected: missing {entry['missing']}, forbidden present "
                  f"{entry['forbidden_present']}, first eval {entry['first_eval_line']}", flush=True)
    elif a.phase == "resume":
        RECORD["entry_lines"] = entry
        if not entry["ok"]:
            status = 1
    RECORD["ended"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    RECORD["exit"] = status
    write_record()
    print(f"{TOOL}: done, exit {status}; record {RECORD_PATH}", flush=True)
    return status


if __name__ == "__main__":
    sys.exit(main())
