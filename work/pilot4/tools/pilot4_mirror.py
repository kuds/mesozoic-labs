"""Walking pilot 4, report only: mirror equivariance of a walker (criteria_pilot4 v3, "Report only", 4th item).

Scratch, never merged; pinned by sha256 before launch. No outcome, label, bar or run decision reads anything this
tool writes. The method is the veer diagnosis's (SP/work/veer/): its policy lane's pi against pi_m
(veer/policy/scripts/lib_roll.py, run_conditions.py, aggregate3.py) and its screen lane's mirror-KL loss
(veer/screen/scripts/lk_check.py), on the repository's committed 63/15 mirror maps (symmetry.MirrorMap.load).

  pi      the walker's deterministic policy: clip((o - mean) / sqrt(var + eps), -clip, clip) as float32, then
          model.predict(deterministic=True) (the frozen gait probe's predict).
  pi_m    the mirrored policy pi_m(o) = M_a(pi(M_o(o))), M_o / M_a the committed observation / action maps.
  A. Panel episodes 0..9 (reset seeds 3042-3051, MIRROR_EPISODES) rolled under pi and, in a fresh env, under pi_m,
     each through the frozen probe's own rollout_episode and analyze_episode (window from 100 steps, as judged).
     Per episode: yaw_change_deg of each, the flip (pi_m turns the other way), and, where both run the full horizon,
     the handedness (yaw(pi) - yaw(pi_m)) / 2 (the policy's own left/right bias, which flips with the mirror) and the
     scene part (yaw(pi) + yaw(pi_m)) / 2 (aggregate3.py B1). Pilot 3's handedness medians at 8M: -26.1 / -47.2 /
     -32.5 deg (v3 rounds -26 / -47 / -33). Also per actuator, the mean of pi(o) - pi_m(o) along pi's own
     trajectories, steps >= 100, mean over the episodes (aggregate.py T5).
  B. The mirror-KL loss L_K = mean_rows sum_i (mu(o_m) - M_a mu(o))_i^2 / (2 sigma_i^2), sigma = exp(log_std)
     clamped at 0.01, computed by the tree's own mirror_kl_ppo.mirror_kl_loss under torch.no_grad (no gradient, no
     update), on the observations pi itself met on panel episodes 0..19 (LK_EPISODES, steps 0..T-1), with the
     checkpoint's own normalisation statistics (o_m from the mirrored raw observation) and with the symmetrised ones
     (mean_sym = (m + M_o m) / 2, var_sym = (v + v[perm]) / 2); kappa x L_K at kappa 0.3, and the asymmetry (row mean of
     max |dev| / 2, action units). Pilot 3's 8M walkers: 78.508 / 41.671 / 54.005 own, 70.724 / 41.119 / 54.832
     symmetrised (v3's "42-79").
  C. --mirstate: pi_m rolled from the MIRRORED reset state (symmetry.mirror_env_state right after the reset): on a
     mirror-symmetric plant it is the mirror image of pi's episode, so yaw(pi) + yaw(pi_m mirstate) ~ 0. In an env with
     a live command mode mirror_env_state refuses (the command state is not mirrored): reported "not run".

    cd TREE && ISO_WT=TREE python -I TOOLS/iso.py TOOLS/pilot4_mirror.py --cell OUT/s42/2000000 --json M/s42.json
    cd TREE && ISO_WT=TREE python -I TOOLS/iso.py TOOLS/pilot4_mirror.py --parent 42 [--runs RUNS] --json M/p42.json
        [--episodes 10] [--lk-episodes 20] [--mirstate] [--validation]

--cell is a judged cell (pilot4_judge.py's OUT/s<seed>/<steps>, schema walking-pilot4-judge/v1; with --validation
also pilot 3's walking-pilot3-judge/v1 cells, for checks only): binding.json names the checkpoint pair (sha256s
re-checked) and the head the tree must be at; the pi episodes are compared with probe.json's rows (every re-rolled
metric must equal the judged one, pilot3_video.py's rule; a difference is recorded and the exit status is 1).
--parent SEED rolls that seed's pilot-3 8M pair from RUNS/pilot3_walk_s<seed>/03_locomotion/models/ (default the
verified read-only run copies), sha256 checked against v3's parent table; in a tree whose locomotion env has a live
command mode it is prepared in memory as train()'s edge prepares it (command_frame.reseed_command_slice and
policy_loading.assert_command_blind with neutralize_command_columns; every step checks prepared(o) ==
original(o with obs[60:63] = 0), max delta recorded), in a command-free tree (pilot 3's) it is rolled as it is.

Refused, nothing written (exit 2): an existing --json (never overwritten); a --json inside the tree, inside the
cell's run copy or inside the run directory (provenance.json) of any input (the gait report plan's rule: a result
bundle rejects files it did not list); a tree with local edits (git status --porcelain --untracked-files=no) or at
another head than the cell's; `environments` imported from outside the tree; a cell whose pair is missing or no longer
its sha256, whose probe.json is not the panel or was rolled under other env kwargs or gate settle, that is
off-schedule, validation-only (unless --validation) or of pilot 3's schema (unless --validation); a parent whose
files are not v3's sha256s. Deterministic, CPU, no training of any kind.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SCHEMA = "walking-pilot4-mirror/v1"
SPECIES, EMBODIMENT, STAGE = "tyrannosaurus_rex", "anatomical", "locomotion"
HEADING_FREE_SCHEMA = "bipedal-target-heading-free/v1"
PILOT4_CELL_SCHEMA = "walking-pilot4-judge/v1"
PILOT3_CELL_SCHEMA = "walking-pilot3-judge/v1"
PANEL_SEED_BASE = 3042
MIRROR_EPISODES = 10  # v3: "on 10 panel resets" (the policy lane's reset seeds 3042-3051)
LK_EPISODES = 20  # the screen lane's lk_check.py: the walker's own 20 panel episodes (3042-3061)
WINDOW_FROM = 100  # the judge's --settle-steps; the T5 window (steps >= 100)
LK_SIGMA_FLOOR = 0.01  # lk_check.py / [ppo] symmetry sigma floor
LK_KAPPA = 0.3  # lk_check.py: kappa x L_K as a fine-tune with symmetry = "mirror_kl" would weigh it
COMMAND_SLICE = slice(60, 63)
DEFAULT_RUNS = ("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot3/"
                "runs/final")
#: v3 "Parent and entry": each seed's pilot-3 8M pair (zip, sidecar), sha256 as the verified run copies hold them.
PARENTS = {
    42: ("sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
         "sha256:b9e6dee894e549ca4e844f7ef3c162e0f52609eacede7b1790e5d649e77d9cce"),
    52: ("sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
         "sha256:cf29635feb8c4040ea9a9bb08255b97de60673365dce107445c115aa2927f37a"),
    60: ("sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2",
         "sha256:56e58a536d08198fdafbde3310e05116ff65e7cd471f536016858f87f9c32dc9"),
}
#: Pilot 3's 8M references, printed beside (report only), each with its source.
PILOT3_REFERENCE = {
    "source_handedness": "veer/policy/out/AGGREGATE3.txt B1, 8M, median of (yaw_pi - yaw_pim)/2 over full episodes",
    "handedness_median_deg": {42: -26.1, 52: -47.2, 60: -32.5},
    "source_pim_yaw": "veer/policy/out/AGGREGATE.txt T6, 8M, pi_m yaw min/median/max (deg), 10 resets",
    "pim_yaw_min_med_max_deg": {42: (19, 24, 43), 52: (36, 48, 52), 60: (23, 42, 57)},
    "source_lk": "veer/screen/out/lk_check.txt (own statistics / symmetrised statistics)",
    "lk_own": {42: 78.508, 52: 41.671, 60: 54.005},
    "lk_symmetrised": {42: 70.724, 52: 41.119, 60: 54.832},
    "lk_stance_parent_on_its_standing": 0.049,
}
REPORT_ONLY = ("report only (criteria_pilot4 v3 'Report only'): no outcome, label, bar, gate or run decision reads "
               "this; it certifies nothing")


class Refusal(Exception):
    """A reason to write nothing (exit 2)."""


# ---------------------------------------------------------------- pure helpers (unit-tested without the tree)


def sha256_file(path: "str | Path") -> str:
    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signed_perm_apply(x, perm, sign) -> np.ndarray:
    """SignedPermutation.apply's rule on the last axis: x[..., perm] * sign."""
    x = np.asarray(x)
    sign = np.asarray(sign, dtype=float)
    if np.issubdtype(x.dtype, np.floating):
        sign = sign.astype(x.dtype)
    return np.asarray(x[..., np.asarray(perm)] * sign)


def symmetrised_stats(mean, var, perm, sign) -> tuple[np.ndarray, np.ndarray]:
    """install_symmetric_obs_rms's statistics (lk_check.py): mean_sym = (m + M_o m)/2, var_sym = (v + v[perm])/2."""
    mean = np.asarray(mean, dtype=np.float64)
    var = np.asarray(var, dtype=np.float64)
    return 0.5 * (mean + signed_perm_apply(mean, perm, sign)), 0.5 * (var + var[np.asarray(perm)])


def normalize(x, mean, var, eps: float, clip: float) -> np.ndarray:
    """The probe's normalisation: clip((x - mean)/sqrt(var + eps), -clip, clip) as float32."""
    return np.clip((np.asarray(x, dtype=np.float64) - mean) / np.sqrt(var + eps), -clip, clip).astype(np.float32)


def lk_numpy(mu_m, mu, log_std, act_perm, act_sign, sigma_floor: float = LK_SIGMA_FLOOR) -> tuple[float, float]:
    """mirror_kl_ppo.mirror_kl_loss in numpy (a cross-check of the tree's function): (L_K, asymmetry)."""
    mu_m = np.asarray(mu_m, dtype=np.float64)
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.maximum(np.exp(np.asarray(log_std, dtype=np.float64)), sigma_floor)
    target = mu[:, np.asarray(act_perm)] * np.asarray(act_sign, dtype=np.float64)
    dev = mu_m - target
    loss = float(((dev ** 2) / (2 * sigma ** 2)).sum(-1).mean())
    asym = float(np.abs(dev).max(-1).mean() / 2)
    return loss, asym


def min_med_max(values) -> "dict | None":
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if not v.size:
        return None
    return {"n": int(v.size), "min": float(v.min()), "median": float(np.median(v)), "max": float(v.max())}


def pair_episodes(pi_rows: list[dict], pim_rows: list[dict], horizon: int) -> list[dict]:
    """Per reset seed (same index): yaw of pi and pi_m, flip, and where both run the full horizon the handedness
    (yaw_pi - yaw_pim)/2 and the scene part (yaw_pi + yaw_pim)/2 (aggregate3.py B1)."""
    out = []
    for a, b in zip(pi_rows, pim_rows):
        if a["reset_seed"] != b["reset_seed"]:
            raise ValueError(f"episodes out of step: reset {a['reset_seed']} against {b['reset_seed']}")
        ya, yb = a.get("yaw_change_deg"), b.get("yaw_change_deg")
        both_full = bool(a["length"] == horizon and b["length"] == horizon and not a["terminated"]
                         and not b["terminated"])
        rec = {"reset_seed": a["reset_seed"], "yaw_pi_deg": ya, "yaw_pim_deg": yb, "length_pi": a["length"],
               "length_pim": b["length"], "both_full": both_full,
               "flipped": (bool(np.sign(ya) == -np.sign(yb)) and ya != 0) if ya is not None and yb is not None
               else None}
        rec["handedness_deg"] = (ya - yb) / 2 if both_full else None
        rec["scene_part_deg"] = (ya + yb) / 2 if both_full else None
        out.append(rec)
    return out


def summarize_pairs(pairs: list[dict]) -> dict:
    turned = [p for p in pairs if p["yaw_pi_deg"] is not None and p["yaw_pim_deg"] is not None]
    return {
        "episodes": len(pairs),
        "comparable": len(turned),
        "both_full": sum(p["both_full"] for p in pairs),
        "flipped": sum(1 for p in turned if p["flipped"]),
        "pi_right": sum(1 for p in turned if p["yaw_pi_deg"] < 0),
        "pi_left": sum(1 for p in turned if p["yaw_pi_deg"] > 0),
        "pim_right": sum(1 for p in turned if p["yaw_pim_deg"] < 0),
        "pim_left": sum(1 for p in turned if p["yaw_pim_deg"] > 0),
        "yaw_pi_deg": min_med_max([p["yaw_pi_deg"] for p in pairs]),
        "yaw_pim_deg": min_med_max([p["yaw_pim_deg"] for p in pairs]),
        "handedness_deg": min_med_max([p["handedness_deg"] for p in pairs]),
        "scene_part_deg": min_med_max([p["scene_part_deg"] for p in pairs]),
    }


def run_directory(path: "str | Path") -> "Path | None":
    """episode_trace.run_directory's rule: the nearest ancestor (or itself) holding provenance.json."""
    resolved = Path(path).resolve()
    for candidate in (resolved, *resolved.parents):
        if (candidate / "provenance.json").is_file():
            return candidate
    return None


def output_problems(json_path: "str | Path", *, tree: "Path | None", inputs: list, run_copy: "str | None") -> list[str]:
    """Why --json may not be written: it exists; or it lies inside the tree, the cell's run copy, or the run
    directory of an input (the gait report plan's refusal, episode_trace.refuse_run_directory_output)."""
    problems = []
    target = Path(json_path).resolve()
    if target.exists():
        problems.append(f"--json {json_path} exists (never overwritten; move it or name another file)")
    if tree is not None and target.is_relative_to(Path(tree).resolve()):
        problems.append(f"--json {target} is inside the tree {tree} (the tree stays clean)")
    if run_copy and target.is_relative_to(Path(run_copy).resolve()):
        problems.append(f"--json {target} is inside the cell's run copy {run_copy}")
    for source in inputs:
        if source is None:
            continue
        run = run_directory(Path(source).parent)
        if run is not None and target.is_relative_to(run):
            problems.append(f"--json {target} is inside the run directory {run} (a result bundle rejects files it did "
                            "not list)")
    return problems


def cell_problems(binding: dict, *, head: str, dirty: str, validation: bool) -> list[str]:
    """Every reason a cell is refused before anything is rolled (pilot3_habit.py's and pilot3_video.py's rules)."""
    problems = []
    schema = binding.get("schema")
    if schema not in (PILOT4_CELL_SCHEMA, PILOT3_CELL_SCHEMA):
        problems.append(f"binding.json schema {schema!r} is not a judge cell's")
    elif schema == PILOT3_CELL_SCHEMA and not validation:
        problems.append("the cell is pilot 3's (walking-pilot3-judge/v1); --validation reads it for a check only")
    if binding.get("head") != head:
        problems.append(f"the cell was judged at {binding.get('head')}, this tree is {head}")
    if dirty:
        problems.append("this tree has local edits (git status --porcelain --untracked-files=no is not empty)")
    if binding.get("off_schedule"):
        problems.append(f"the cell is off-schedule ({binding.get('off_schedule_reason')}): never read")
    if binding.get("validation_only") and not validation:
        problems.append("the cell is validation-only; --validation reads it for a check only")
    return problems


def parent_paths(runs: "str | Path", seed: int) -> tuple[Path, Path]:
    base = Path(runs) / f"pilot3_walk_s{seed}" / "03_locomotion" / "models"
    return base / "stage2_8000000_steps.zip", base / "stage2_vecnormalize_8000000_steps.pkl"


def parent_problems(seed: int, zip_path: Path, pkl_path: Path) -> list[str]:
    if seed not in PARENTS:
        return [f"--parent {seed} is not one of the scored seeds {sorted(PARENTS)}"]
    problems = []
    for path, want in zip((zip_path, pkl_path), PARENTS[seed]):
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif sha256_file(path) != want:
            problems.append(f"{path} is not s{seed}'s 8M parent ({want})")
    return problems


def jsafe(value):
    if isinstance(value, dict):
        return {str(k): jsafe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsafe(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsafe(value.tolist())
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


# ---------------------------------------------------------------- the tree side


def install_probe_shims(embodiment: str = EMBODIMENT) -> None:
    """gait_probe_pilot.py's four shims (bare species string -> the model id), so the frozen probe's functions run
    (pilot3_video.py's install_probe_shims)."""
    from environments.shared import config, plant_contract, species_registry, stage_manifest
    from environments.shared.model_id import ModelId
    from environments.shared.species_names import require_model_id

    def mid(species):
        return species if isinstance(species, ModelId) else require_model_id(str(species), embodiment)

    _load_stage_config = config.load_stage_config
    _load_stage_manifest = stage_manifest.load_stage_manifest
    _current_plant_identity = plant_contract.current_plant_identity

    class _Factories(dict):
        def __missing__(self, species):
            return lambda: species_registry.get_species_config(mid(species))

    species_registry.SPECIES_FACTORIES = _Factories()  # type: ignore[attr-defined]
    config.load_stage_config = lambda species, stage, *a, **k: _load_stage_config(mid(species), stage, *a, **k)
    stage_manifest.load_stage_manifest = lambda species, *a, **k: _load_stage_manifest(mid(species), *a, **k)
    plant_contract.current_plant_identity = lambda species, *a, **k: _current_plant_identity(mid(species), *a, **k)


def load_probe(tree: Path):
    path = tree / "docs" / "investigations" / "gait_2026_09" / "gait_probe.py"
    spec = importlib.util.spec_from_file_location("gait_probe_frozen", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, path


def git(tree: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=tree, text=True,
                                   env=dict(os.environ, GIT_OPTIONAL_LOCKS="0")).strip()


class Recorder:
    """A predict for the probe's rollout_episode that records every observation it is given and its action, and
    (optionally) the mirrored policy's action on the same observation (not applied)."""

    def __init__(self, act, mirror=None, check=None):
        self.act, self.mirror, self.check = act, mirror, check
        self.obs, self.actions, self.mirror_actions = [], [], []
        self.max_check_delta = 0.0

    def __call__(self, obs):
        o = np.asarray(obs, dtype=np.float64).copy()
        a = np.asarray(self.act(o), dtype=np.float64)
        self.obs.append(o)
        self.actions.append(a.copy())
        if self.mirror is not None:
            self.mirror_actions.append(np.asarray(self.mirror(o), dtype=np.float64))
        if self.check is not None:
            self.max_check_delta = max(self.max_check_delta, float(np.max(np.abs(a - self.check(o)))))
        return a


def main(argv=None) -> int:  # noqa: C901 - one linear procedure, sectioned
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cell", help="a judged cell OUT/s<seed>/<steps> (binding.json, probe.json)")
    src.add_argument("--parent", type=int, help="a seed: its pilot-3 8M pair (edge-prepared in a live-command tree)")
    ap.add_argument("--runs", default=DEFAULT_RUNS, help="the folder holding pilot3_walk_s<seed>/ (--parent)")
    ap.add_argument("--json", required=True)
    ap.add_argument("--episodes", type=int, default=MIRROR_EPISODES)
    ap.add_argument("--lk-episodes", type=int, default=LK_EPISODES)
    ap.add_argument("--mirstate", action="store_true", help="also roll pi_m from the mirrored reset state")
    ap.add_argument("--validation", action="store_true", help="read pilot 3's or a validation-only cell (checks only)")
    a = ap.parse_args(argv)

    tree = Path.cwd().resolve()
    tool_sha = sha256_file(__file__)  # the bytes that run, read before anything else

    def refuse(problems: list[str]) -> int:
        print("pilot4_mirror: refusing (nothing written): " + "; ".join(problems), flush=True)
        return 2

    early = [] if 1 <= a.episodes <= 40 and 1 <= a.lk_episodes <= 40 else [
        "--episodes and --lk-episodes must be 1..40 (the panel)"]
    if Path(a.json).exists():
        early.append(f"--json {a.json} exists (never overwritten; move it or name another file)")
    if early:  # before anything is imported from the tree
        return refuse(early)
    import environments

    env_file = str(Path(environments.__file__).resolve())
    if not env_file.startswith(str(tree) + os.sep):
        return refuse([f"environments imported from {env_file}, not from the tree {tree} (run through iso.py)"])
    try:
        head, dirty = git(tree, "rev-parse", "HEAD"), git(tree, "status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError) as exc:
        return refuse([f"the tree {tree} is not a readable git checkout: {exc}"])

    binding, rows, meta, run_copy = None, None, None, None
    if a.cell:
        cell = Path(a.cell).resolve()
        try:
            binding = json.loads((cell / "binding.json").read_text())
            doc = json.loads((cell / "probe.json").read_text())
        except (OSError, ValueError) as exc:
            return refuse([f"the cell {cell} cannot be read: {exc}"])
        meta, rows = doc["meta"], doc["episodes"]
        problems = cell_problems(binding, head=head, dirty=dirty, validation=a.validation)
        ckpt, vec = Path(binding["checkpoint"]), Path(binding["vecnorm"])
        for p, want in ((ckpt, binding.get("checkpoint_sha256")), (vec, binding.get("vecnorm_sha256"))):
            if not p.is_file() or sha256_file(p) != want:
                problems.append(f"{p} is missing or no longer the judged file ({want})")
        if meta.get("seed_scheme") != "panel" or int(meta.get("seed_base", -1)) != PANEL_SEED_BASE:
            problems.append(f"probe.json's seeds are {meta.get('seed_scheme')!r} from {meta.get('seed_base')}, not the "
                            f"panel from {PANEL_SEED_BASE}")
        if len(rows) < max(a.episodes, a.lk_episodes):
            problems.append(f"probe.json holds {len(rows)} episodes, fewer than {max(a.episodes, a.lk_episodes)}")
        seed, steps = int(binding["seed"]), int(binding["steps"])
        run_copy = binding.get("run")
    else:
        seed, steps = int(a.parent), 8_000_000
        ckpt, vec = parent_paths(a.runs, seed)
        problems = parent_problems(seed, ckpt, vec)
        if dirty:
            problems.append("this tree has local edits (git status --porcelain --untracked-files=no is not empty)")
    problems += output_problems(a.json, tree=tree, inputs=[ckpt, vec], run_copy=run_copy)
    if problems:
        return refuse(problems)

    install_probe_shims()
    probe, probe_path = load_probe(tree)
    import gymnasium as gym
    import torch as th

    from environments.shared.config import load_stage_config
    from environments.shared.mirror_kl_ppo import mirror_kl_loss
    from environments.shared.plant_contract import current_plant_identity, validate_environment_plant
    from environments.shared.policy_loading import load_sb3_checkpoint
    from environments.shared.species_names import require_model_id
    from environments.shared.species_registry import get_species_config
    from environments.shared.symmetry import MirrorMap, build_env_mirror, mirror_env_state

    th.set_num_threads(1)
    mid = require_model_id(SPECIES, EMBODIMENT)
    stage_cfg = load_stage_config(mid, STAGE)
    env_kwargs = dict(stage_cfg["env_kwargs"])
    env_class = get_species_config(mid).env_class
    plant = current_plant_identity(mid)
    if meta is not None:
        from environments.shared.curriculum.stance_gate import StanceGateThresholds

        gate_settle = int(StanceGateThresholds.from_curriculum(stage_cfg.get("curriculum_kwargs", {}),
                                                               require_criteria=False).settle_steps)
        same = json.loads(json.dumps(probe._json_safe(env_kwargs))) == meta.get("env_kwargs")
        if not same or gate_settle != meta.get("settle_steps_gate"):
            return refuse(["this tree's locomotion env kwargs or gate settle differ from probe.json's"])
        settle = int(meta["settle_steps_gait_window"])
    else:
        gate_settle, settle = 0, WINDOW_FROM
    live = str(env_kwargs.get("command_mode", "none")) != "none"
    horizon = int(env_kwargs.get("max_episode_steps", 1000))

    def factory():
        return env_class(**env_kwargs)

    model, normalizer, _ = load_sb3_checkpoint(str(ckpt), str(vec), factory, guess_sidecar=False,
                                               allow_unnormalized=False, plant_identity=plant,
                                               allow_legacy_plant=False)
    prepared = {"edge_prepared": False}
    check = None
    if a.parent is not None and live:
        from environments.shared.command_frame import reseed_command_slice
        from environments.shared.policy_loading import assert_command_blind, neutralize_command_columns

        orig, orig_norm, _ = load_sb3_checkpoint(str(ckpt), str(vec), factory, guess_sidecar=False,
                                                 allow_unnormalized=False, plant_identity=plant,
                                                 allow_legacy_plant=False)
        reseed_command_slice(normalizer.obs_rms)
        assert_command_blind(model, normalizer, prepare=lambda: neutralize_command_columns(model, observation_dim=63))
        om, ov = np.asarray(orig_norm.obs_rms.mean, float), np.asarray(orig_norm.obs_rms.var, float)
        oe, oc = float(orig_norm.epsilon), float(orig_norm.clip_obs)

        def check(o):
            z = np.array(o, dtype=np.float64, copy=True)
            z[COMMAND_SLICE] = 0.0
            return np.asarray(orig.predict(normalize(z, om, ov, oe, oc), deterministic=True)[0], dtype=np.float64)

        prepared = {"edge_prepared": True, "how": "command_frame.reseed_command_slice(obs_rms); policy_loading."
                    "assert_command_blind(prepare=neutralize_command_columns(observation_dim=63)), as recov.py"}
    mean = np.asarray(normalizer.obs_rms.mean, dtype=np.float64)
    var = np.asarray(normalizer.obs_rms.var, dtype=np.float64)
    eps, clip = float(normalizer.epsilon), float(normalizer.clip_obs)
    mm = MirrorMap.load(mid, plant)

    def pi(o):
        return np.asarray(model.predict(normalize(o, mean, var, eps, clip), deterministic=True)[0], dtype=np.float64)

    def pi_m(o):
        return mm.action.apply(pi(mm.obs.apply(np.asarray(o, dtype=np.float64))))

    env = factory()
    validate_environment_plant(env, plant, artifact="pilot4_mirror env")
    rd = probe.RobotDescription(env)
    dt = float(env.unwrapped.dt)
    try:
        em = build_env_mirror(env.unwrapped, HEADING_FREE_SCHEMA)
        map_equal = bool(np.array_equal(em.obs.perm, mm.obs.perm) and np.array_equal(em.obs.sign, mm.obs.sign)
                         and np.array_equal(em.action.perm, mm.action.perm)
                         and np.array_equal(em.action.sign, mm.action.sign))
        em_note = None
    except Exception as exc:  # noqa: BLE001 - recorded: the readout says why
        em, map_equal, em_note = None, None, f"{type(exc).__name__}: {exc}"

    def row_of(tr) -> dict:
        r = probe.analyze_episode(tr, rd, dt, settle, gate_settle)
        return json.loads(json.dumps(probe._json_safe(r)))

    n_pi = max(a.episodes, a.lk_episodes)
    pi_rows, pi_recs, matches = [], [], []
    for i in range(n_pi):
        rec = Recorder(pi, mirror=pi_m if i < a.episodes else None, check=check)
        tr = probe.rollout_episode(env, rec, rd, PANEL_SEED_BASE + i)
        row = row_of(tr)
        if rows is not None:
            diff = sorted(k for k in (set(row) | set(rows[i])) - {"episode", "reset_seed"}
                          if row.get(k, "<absent>") != rows[i].get(k, "<absent>"))
            matches.append({"episode": i, "matches_probe": not diff, "differing_keys": diff[:20]})
        pi_rows.append({"reset_seed": PANEL_SEED_BASE + i, "length": int(tr["length"]),
                        "terminated": bool(tr["terminated"]), "yaw_change_deg": row.get("yaw_change_deg")})
        pi_recs.append(rec)
        print(f"  pi   ep {i:2d} reset {PANEL_SEED_BASE + i}: length {tr['length']} yaw {row.get('yaw_change_deg')}"
              + ("" if rows is None else ("  matches probe.json" if not matches[-1]["differing_keys"]
                                          else f"  DIFFERS {matches[-1]['differing_keys'][:4]}")), flush=True)
    env.close()
    env_m = factory()
    pim_rows = []
    for i in range(a.episodes):
        tr = probe.rollout_episode(env_m, pi_m, rd, PANEL_SEED_BASE + i)
        row = row_of(tr)
        pim_rows.append({"reset_seed": PANEL_SEED_BASE + i, "length": int(tr["length"]),
                         "terminated": bool(tr["terminated"]), "yaw_change_deg": row.get("yaw_change_deg")})
        print(f"  pi_m ep {i:2d} reset {PANEL_SEED_BASE + i}: length {tr['length']} yaw {row.get('yaw_change_deg')}",
              flush=True)
    env_m.close()
    pairs = pair_episodes(pi_rows[: a.episodes], pim_rows, horizon)
    summary = summarize_pairs(pairs)
    summary["pi_falls"] = sum(1 for r in pi_rows[: a.episodes] if r["length"] < horizon or r["terminated"])
    summary["pim_falls"] = sum(1 for r in pim_rows if r["length"] < horizon or r["terminated"])

    # T5: per actuator, mean over episodes of the episode mean of pi(o) - pi_m(o), steps >= WINDOW_FROM
    act_names = [n.replace("action.", "") for n in mm.action.names] if mm.action.names else \
        [f"a{i}" for i in range(mm.action.dim)]
    per_ep = []
    for rec in pi_recs[: a.episodes]:
        act, act_m = np.asarray(rec.actions), np.asarray(rec.mirror_actions)
        if act.shape[0] > WINDOW_FROM:
            per_ep.append((act[WINDOW_FROM:] - act_m[WINDOW_FROM:]).mean(axis=0))
    t5 = {n: float(np.mean([e[j] for e in per_ep])) for j, n in enumerate(act_names)} if per_ep else {}

    # mirstate
    mirstate = {"run": False}
    if a.mirstate:
        if em is None:
            mirstate["why_not"] = f"build_env_mirror refused: {em_note}"
        else:
            class MirroredReset(gym.Wrapper):
                def reset(self, **kw):
                    _obs, info = self.env.reset(**kw)
                    mirror_env_state(self.env.unwrapped, self.env.unwrapped, em, forward=True)
                    return self.env.unwrapped._get_obs(), info

            env_s = MirroredReset(factory())
            ms_rows = []
            try:
                for i in range(a.episodes):
                    tr = probe.rollout_episode(env_s, pi_m, rd, PANEL_SEED_BASE + i)
                    row = row_of(tr)
                    ms_rows.append({"reset_seed": PANEL_SEED_BASE + i, "length": int(tr["length"]),
                                    "yaw_change_deg": row.get("yaw_change_deg")})
                sums = [abs(p["yaw_change_deg"] + m["yaw_change_deg"]) for p, m in zip(pi_rows, ms_rows)
                        if p["yaw_change_deg"] is not None and m["yaw_change_deg"] is not None]
                mirstate = {"run": True, "episodes": ms_rows, "max_abs_yaw_pi_plus_pim_mirstate_deg":
                            float(max(sums)) if sums else None}
            except Exception as exc:  # noqa: BLE001 - a refusal of the mirror (live command state) is reported
                mirstate = {"run": False, "why_not": f"{type(exc).__name__}: {exc}"}
            finally:
                env_s.close()

    # L_K on pi's own observations (steps 0..T-1 of the first lk_episodes episodes)
    X = np.concatenate([np.asarray(r.obs) for r in pi_recs[: a.lk_episodes]])
    Xm = mm.obs.apply(X)
    pol = model.policy
    log_std = pol.log_std.detach()
    perm_a = th.as_tensor(np.asarray(mm.action.perm))
    sign_a = th.as_tensor(np.asarray(mm.action.sign, dtype=np.float32))
    mean_sym, var_sym = symmetrised_stats(mean, var, mm.obs.perm, mm.obs.sign)
    lk = {"states": int(X.shape[0]), "episodes": a.lk_episodes, "sigma_floor": LK_SIGMA_FLOOR, "kappa": LK_KAPPA,
          "log_std": {"min": float(log_std.min()), "mean": float(log_std.mean()), "max": float(log_std.max())}}
    for label, (m_, v_) in (("own_statistics", (mean, var)), ("symmetrised_statistics", (mean_sym, var_sym))):
        o = th.as_tensor(normalize(X, m_, v_, eps, clip))
        om = th.as_tensor(normalize(Xm, m_, v_, eps, clip))
        with th.no_grad():
            mu = pol.get_distribution(o).mode()
            mu_m = pol.get_distribution(om).mode()
            loss, asym = mirror_kl_loss(mu_m, mu, log_std, perm_a, sign_a, LK_SIGMA_FLOOR)
        n_loss, n_asym = lk_numpy(mu_m.numpy(), mu.numpy(), log_std.numpy(), mm.action.perm, mm.action.sign)
        lk[label] = {"L_K": float(loss), "kappa_L_K": LK_KAPPA * float(loss), "asymmetry": float(asym),
                     "numpy_cross_check_L_K": n_loss, "numpy_cross_check_asymmetry": n_asym}

    all_match = all(m["matches_probe"] for m in matches) if matches else None
    record = {
        "schema": SCHEMA, "report_only": REPORT_ONLY, "tool_sha256": tool_sha,
        "tree": str(tree), "head": head, "environments_file": env_file, "probe": str(probe_path),
        "probe_sha256": sha256_file(probe_path), "mode": "cell" if a.cell else "parent",
        "cell": str(Path(a.cell).resolve()) if a.cell else None, "binding_schema": (binding or {}).get("schema"),
        "validation": bool(a.validation), "seed": seed, "steps": steps,
        "checkpoint": str(ckpt), "checkpoint_sha256": sha256_file(ckpt), "vecnorm": str(vec),
        "vecnorm_sha256": sha256_file(vec), "env_command_mode": str(env_kwargs.get("command_mode", "none")),
        "env_prey_lateral_range": jsafe(env_kwargs.get("prey_lateral_range")), "horizon": horizon,
        "settle_steps_gait_window": settle, "settle_steps_gate": gate_settle, **prepared,
        "edge_check_max_action_delta": (max(r.max_check_delta for r in pi_recs) if check is not None else None),
        "mirror_map": {"path": mm.path, "sha256": mm.sha256, "obs_dim": mm.obs.dim, "action_dim": mm.action.dim,
                       "equals_build_env_mirror": map_equal, "build_env_mirror_note": em_note},
        "panel_seed_base": PANEL_SEED_BASE, "mirror_episodes": a.episodes, "lk_episodes": a.lk_episodes,
        "pi_episodes_match_probe": matches, "all_pi_episodes_match_probe": all_match,
        "pi_episodes": pi_rows, "pim_episodes": pim_rows, "pairs": pairs, "summary": summary,
        "actuator_mean_pi_minus_pim_steps_from_100": t5, "mirstate": mirstate, "mirror_kl": lk,
        "pilot3_reference": PILOT3_REFERENCE,
    }
    out = Path(a.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsafe(record), indent=1))
    for line in describe(record):
        print(line, flush=True)
    print(f"pilot4_mirror: wrote {out}", flush=True)
    return 0 if all_match in (None, True) else 1


def describe(r: dict) -> list[str]:
    s, seed = r["summary"], r["seed"]
    ref = r["pilot3_reference"]

    def mmm(d):
        return "n/a" if not d else f"{d['min']:+.1f} / {d['median']:+.1f} / {d['max']:+.1f} (n {d['n']})"

    lines = [f"MIRROR (report only) s{seed} at {r['steps']:,} [{r['mode']}{', edge-prepared' if r['edge_prepared'] else ''};"
             f" command_mode {r['env_command_mode']}]",
             f"  pi   yaw min/median/max {mmm(s['yaw_pi_deg'])}; right {s['pi_right']}, left {s['pi_left']}, falls "
             f"{s['pi_falls']}",
             f"  pi_m yaw min/median/max {mmm(s['yaw_pim_deg'])}; right {s['pim_right']}, left {s['pim_left']}, falls "
             f"{s['pim_falls']}; the turn flips in {s['flipped']} of the {s['comparable']} episodes with both yaws "
             f"(of {s['episodes']})",
             f"  handedness (yaw_pi - yaw_pim)/2 {mmm(s['handedness_deg'])}; pilot 3 at 8M: "
             f"{ref['handedness_median_deg'].get(seed, 'n/a')} (median)",
             f"  scene part (yaw_pi + yaw_pim)/2 {mmm(s['scene_part_deg'])}"]
    lk = r["mirror_kl"]
    lines.append(f"  mirror-KL L_K on own states ({lk['states']} steps of {lk['episodes']} episodes): own statistics "
                 f"{lk['own_statistics']['L_K']:.3f}, symmetrised {lk['symmetrised_statistics']['L_K']:.3f}; pilot 3 "
                 f"at 8M {ref['lk_own'].get(seed, 'n/a')} / {ref['lk_symmetrised'].get(seed, 'n/a')}")
    ms = r["mirstate"]
    lines.append("  pi_m from the mirrored reset: " + (
        f"max |yaw_pi + yaw_pim| {ms['max_abs_yaw_pi_plus_pim_mirstate_deg']:.4f} deg" if ms.get("run")
        else f"not run ({ms.get('why_not', 'not requested')})"))
    if r["all_pi_episodes_match_probe"] is not None:
        lines.append(f"  pi episodes equal probe.json's: {r['all_pi_episodes_match_probe']}")
    if r["edge_check_max_action_delta"] is not None:
        lines.append(f"  edge-prepared walker vs the original on command-zeroed observations: max |dA| "
                     f"{r['edge_check_max_action_delta']:.3g}")
    lines.append(f"  {REPORT_ONLY}")
    return lines


if __name__ == "__main__":
    sys.exit(main())
