"""End-to-end check of the judging tools on one real checkpoint (walking pilot 3 scratch, never merged).

No pilot-3 run exists yet, so a synthetic run copy stands in: its stage_config.json records what a pilot-3 run
records (built from the tree's own digests and plant identity, the parent's critic-phase record, -2.0, n_envs 4, seed
42, budget 8,000,000, the tree's HEAD), and its stage2_8000000_steps pair is the PARENT's pair (stage1_final.zip and
its sidecar, copied byte for byte). The checks are of the tools' paths, not of any pilot-3 result:
  1. pilot3_judge.py --check-only, then the full judge: the probe runs through iso.py and gait_probe_pilot.py, the
     cell OUT/s42/8000000 is written (probe.json, binding.json); a second judgement of the cell is refused.
  2. pilot3_habit.py seed on that cell (the trace rolls the cell's checkpoint) with the PC round's PHASE s42 stance
     ledger standing in for a locomotion ledger (read through 8,000,000), copied into the run copy's 03_locomotion/
     as eval_episodes.csv (FIX2: the tool reads only the judged run's own ledger).
  3. pilot3_score.py on OUT with --habit: the cell is scored, the existence and Q4 lines print.
  4. pilot3_video.py on episodes 0 and 1 (the panel prefix rolled in order) under Xvfb: both re-rolled episodes
     equal probe.json's, the boxes equal the probe's floor channel, two mp4s of 1,000 frames each per episode; on a
     copy of the cell whose probe.json has one metric of episode 1 nudged, the tool reports the difference, writes no
     video for episode 1 and exits 1; --standalone on episode 1 (rolled alone) is recorded as information.
By-product, reported as such and not the pre-launch check itself: the probe's reading of the parent on the
locomotion panel (the criteria expect 40/40 STILL).

    python3 -I -B tests/e2e_parent_cell.py TOOLS_DIR    (writes TOOLS_DIR/e2e/; one simulation at a time)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

TOOLS = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
TREE = TOOLS / "tree"
PY = TOOLS / "venv" / "bin" / "python"
E2E = TOOLS / "e2e"
PARENT_DIR = TREE / "pilot_checkpoints" / "stance7_r10LTSBpc_s42_final"
LEDGER = Path("/tmp/claude-0/-home-user-mesozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/fetch/"
              "stance7_r10LTSBpc_s42/01_stance/eval_episodes.csv")
PARENT = "sha256:1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6"
results: dict = {"checks": []}


def env(extra=None):
    e = dict(os.environ, ISO_WT=str(TREE), MUJOCO_GL="disable", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
             OPENBLAS_NUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1", GIT_OPTIONAL_LOCKS="0",
             MESA_SHADER_CACHE_DISABLE="true", MPLCONFIGDIR=str(TOOLS / "tmp" / "mpl"),
             XDG_CACHE_HOME=str(TOOLS / "tmp" / "xdg_cache"))
    e.pop("PYTHONPATH", None)
    e.update(extra or {})
    return e


def run(name, cmd, *, log, extra_env=None, ok_codes=(0,)):
    print(f"[e2e] {name}: {' '.join(map(str, cmd))}", flush=True)
    with open(E2E / log, "w") as fh:
        rc = subprocess.call(["nice", "-n", "10", *map(str, cmd)], cwd=TREE, env=env(extra_env), stdout=fh,
                             stderr=subprocess.STDOUT)
    text = (E2E / log).read_text()
    check(f"{name}: exit {rc} in {ok_codes}", rc in ok_codes, f"see {log}")
    return rc, text


def check(name, ok, detail=""):
    results["checks"].append({"name": name, "ok": bool(ok), "detail": detail})
    print(f"[e2e] {'ok  ' if ok else 'FAIL'} {name}" + (f" ({detail})" if detail else ""), flush=True)


def main() -> int:
    if E2E.exists():
        shutil.rmtree(E2E)
    E2E.mkdir()
    r = subprocess.run([str(PY), "-I", str(TOOLS / "iso.py"), str(TOOLS / "tests" / "_tree_facts.py")], cwd=TREE,
                       env=env(), capture_output=True, text=True, check=True)
    facts = json.loads(r.stdout.strip().splitlines()[-1])
    stage = E2E / "runs" / "pilot3_walk_s42" / "03_locomotion"
    models = stage / "models"
    models.mkdir(parents=True)
    shutil.copyfile(PARENT_DIR / "stage1_final.zip", models / "stage2_8000000_steps.zip")
    shutil.copyfile(PARENT_DIR / "stage1_final_vecnorm.pkl", models / "stage2_vecnormalize_8000000_steps.pkl")
    sc = {"species": "tyrannosaurus_rex", "embodiment": "anatomical", "stage": 2, "name": "Locomotion",
          "algorithm": "PPO", "git_commit": facts["head"], "hyperparameters": facts["hyperparameters"],
          "curriculum": facts["curriculum"], "plant_identity": facts["plant"],
          "run": {"seed": 42, "n_envs": 4, "timesteps": 8_000_000, "critic_phase_dropped": facts["parent_record"],
                  "warm_start_log_std": -2.0, "hyperparameters_sha256": facts["hp"],
                  "load_path": str(PARENT_DIR / "stage1_final.zip"), "load_mode": "initialize_next_stage",
                  "parent_checkpoint_sha256": PARENT}}
    (stage / "stage_config.json").write_text(json.dumps(sc, indent=1))
    (stage / "plant_identity.json").write_text(json.dumps(facts["plant"], indent=1))
    shutil.copyfile(LEDGER, stage / "eval_episodes.csv")  # FIX2: the habit tool reads only the judged run's own ledger
    run_dir, out = stage.parent, E2E / "OUT"
    judge = [PY, "-I", TOOLS / "iso.py", TOOLS / "pilot3_judge.py", run_dir, "--seed", "42", "--steps", "8000000",
             "--parent-sha256", PARENT, "--out", out, "--expect-head", facts["head"]]

    # 1. the judge
    rc, text = run("judge --check-only", [*judge, "--check-only"], log="judge_check_only.log.txt")
    check("judge --check-only wrote nothing", not out.exists())
    rc, text = run("judge (probe on the cell)", judge, log="judge.log.txt")
    cell = out / "s42" / "8000000"
    check("judge wrote probe.json and binding.json", (cell / "probe.json").is_file() and (cell / "binding.json").is_file())
    binding = json.loads((cell / "binding.json").read_text())
    probe = json.loads((cell / "probe.json").read_text())
    check("binding schema, final_fallback false, warm-start records bound",
          binding["schema"] == "walking-pilot3-judge/v1" and binding["final_fallback"] is False
          and binding["warm_start"]["run_warm_start_log_std"] == -2.0
          and binding["parent_critic_phase_record"] == facts["parent_record"] and binding["n_envs"] == 4)
    check("probe panel: 40 episodes, panel 3042, window 100",
          len(probe["episodes"]) == 40 and probe["meta"]["seed_base"] == 3042
          and probe["meta"]["settle_steps_gait_window"] == 100, f"L {probe['meta']['robot'].get('leg_length_m')}")
    rc2, _ = run("a second judgement of the cell is refused", judge, log="judge_again.log.txt", ok_codes=(2,))
    results["parent_probe_by_product"] = {
        "floor_classes": dict(Counter(e.get("floor_gait_class") for e in probe["episodes"])),
        "lengths": dict(Counter(int(e["length"]) for e in probe["episodes"])),
        "speed_displacement_mps_max": max(e.get("speed_displacement_mps") or 0 for e in probe["episodes"]),
        "leg_length_m": probe["meta"]["robot"].get("leg_length_m"),
        "plant_policy_interface_revision": probe["meta"].get("plant_identity_policy_interface_revision"),
        "elapsed_s": probe["meta"].get("elapsed_s")}

    # 2. the habit tool on the cell
    habit_json = E2E / "habit_s42.json"
    rc, text = run("habit seed", [PY, "-I", TOOLS / "iso.py", TOOLS / "pilot3_habit.py", "seed", "--seed", "42",
                                  "--ledger", stage / "eval_episodes.csv", "--cell", cell, "--json", habit_json],
                   log="habit_seed.log.txt")
    habit = json.loads(habit_json.read_text())
    check("habit seed reads NO HABIT on the parent's checkpoint with a quiet ledger",
          habit["reading"] == "NO HABIT" and habit["ledger_reading"]["window"]["evaluations"] == 61
          and habit["trace"]["full_horizon"] == 20, habit["reading"])
    check("habit record names the cell's checkpoint", habit["deciding_checkpoint"]["checkpoint_sha256"]
          == binding["checkpoint_sha256"] == probe["meta"]["checkpoint_sha256"])

    # 3. the scorer
    score_json = E2E / "score.json"
    rc, text = run("score", [PY, "-I", "-B", TOOLS / "pilot3_score.py", out, "--seeds", "42", "--habit", habit_json,
                             "--json", score_json], log="score.log.txt")
    score = json.loads(score_json.read_text())
    results["score_lines"] = [ln for ln in text.splitlines() if ln.startswith(("PILOT", "EXISTENCE", "Q4", "  s42"))]
    check("score: the cell is scored and Q4 is read", score["per_seed_primary"]["42"] is not None
          and score["q4"]["reading"] is not None, f"{score['per_seed_primary']['42']}; {score['q4'].get('reading')}")

    # 4. the video tool
    xvfb = ["xvfb-run", "-a", "-s", "-screen 0 1024x768x24"]
    vid = E2E / "videos"
    rc, text = run("video episodes 0,1 (panel prefix)", [*xvfb, PY, "-I", TOOLS / "iso.py", TOOLS / "pilot3_video.py",
                                                         cell, "--episodes", "0,1", "--out", vid],
                   log="video.log.txt", extra_env={"MUJOCO_GL": "glfw"})
    rec = json.loads((vid / "videos.json").read_text())
    ok = rec["all_rolled_match"] and all(e["matches_probe"] and e["boxes_equal_probe_floor_channel"]
                                         for e in rec["episodes"].values())
    check("video: re-rolled episodes equal probe.json's and the boxes equal the probe's floor channel", ok)
    frames = {}
    for ep, ent in rec["episodes"].items():
        for kind, f in ent["files"].items():
            p = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                "stream=nb_read_frames,r_frame_rate", "-of", "json", f["path"]],
                               capture_output=True, text=True, check=True)
            s = json.loads(p.stdout)["streams"][0]
            frames[f"{ep}/{kind}"] = (int(s["nb_read_frames"]), s["r_frame_rate"])
    check("video: 4 mp4s of 1,000 frames, 100 and 25 fps", sorted(frames.values()) ==
          sorted([(1000, "100/1"), (1000, "25/1")] * 2), str(frames))
    tampered = E2E / "tampered" / "s42" / "8000000"
    shutil.copytree(cell, tampered)
    doc = json.loads((tampered / "probe.json").read_text())
    doc["episodes"][1]["floor_duty_L"] = (doc["episodes"][1].get("floor_duty_L") or 0.0) + 1e-9
    (tampered / "probe.json").write_text(json.dumps(doc, indent=1))
    vid2 = E2E / "videos_tampered"
    rc, text = run("video on a nudged probe.json refuses episode 1", [*xvfb, PY, "-I", TOOLS / "iso.py",
                                                                      TOOLS / "pilot3_video.py", tampered, "--episodes",
                                                                      "1", "--out", vid2],
                   log="video_tampered.log.txt", extra_env={"MUJOCO_GL": "glfw"}, ok_codes=(1,))
    rec2 = json.loads((vid2 / "videos.json").read_text())
    check("video: the nudged episode gets no video and names the key",
          rec2["episodes"]["1"]["differing_keys"] == ["floor_duty_L"] and rec2["episodes"]["1"]["files"] == {}
          and not list(vid2.glob("*.mp4")))
    vid3 = E2E / "videos_standalone"
    rc, text = run("video --standalone episode 1 (information)", [*xvfb, PY, "-I", TOOLS / "iso.py",
                                                                  TOOLS / "pilot3_video.py", cell, "--episodes", "1",
                                                                  "--standalone", "--out", vid3],
                   log="video_standalone.log.txt", extra_env={"MUJOCO_GL": "glfw"}, ok_codes=(0, 1))
    results["standalone_episode_1_matches"] = json.loads((vid3 / "videos.json").read_text())["all_rolled_match"]
    st = subprocess.run(["git", "status", "--porcelain", "--ignored"], cwd=TREE, capture_output=True, text=True,
                        env=env()).stdout
    check("the tree is still clean (nothing written into it)", st.strip() == "", st[:300])
    bad = [c for c in results["checks"] if not c["ok"]]
    results["failed"] = len(bad)
    (E2E / "e2e_summary.json").write_text(json.dumps(results, indent=1, default=str))
    print(f"[e2e] {len(results['checks']) - len(bad)}/{len(results['checks'])} checks passed; by-product "
          f"{results['parent_probe_by_product']}; standalone ep 1 matches: {results['standalone_episode_1_matches']}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
