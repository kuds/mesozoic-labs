"""Tests of pilot4_judge.py and iso.py (pytest; walking pilot 4 scratch, never merged).

Adapted from walking pilot 3's tests/test_pilot3_judge.py (sha256 cc26edcd...), plus every refusal the pilot-4 TOOL
PLAN lists. Pure tests call the judge's functions on stage_config.json dicts built from SC_TEMPLATE: a verbatim copy of
the stage_config.json the pilot-4 setup check wrote through train()'s real path at 0bcb0cf for s42 (SP/work/pilot4/
build/check/judge/fake_runs/pilot4_walk_s42/03_locomotion/stage_config.json, a cp -p of the setup check's own stage
directory); test_the_template_is_the_setup_check_s_record compares it, and its s52 and s60 variants, with those files
when they are present. Integration tests run the judge through iso.py on a clean clone of the pilot branch
(PILOT4_TREE, default ../tree) against run copies built from the template with git_commit set to the tree's HEAD:
synthetic pairs (tiny zips holding SB3's member names and a JSON data member with the lineage record, a pickled dict
as the sidecar: they pass checkpoint_pair_problem but no probe could roll them) and, when the verified pilot-3 run
copies are present (PILOT3_RUNS), the real 8M pairs renamed as stand-in pairs. No test reaches the probe and nothing
is trained (the probe run on a stand-in cell is the lane's end-to-end script).
Run as: python -I -B -m pytest -p no:cacheprovider tests/test_pilot4_judge.py (no bytecode, no cache beside the tools).
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
import pilot4_judge as pj  # noqa: E402

TREE = Path(os.environ.get("PILOT4_TREE", str(TOOLS / "tree"))).resolve()
WORK = TOOLS.parents[1]  # SP/work when the tools sit in SP/work/pilot4/tools
RECORDED = Path(os.environ.get("PILOT4_SETUP_RECORDS",
                               str(WORK / "pilot4" / "build" / "check" / "judge" / "fake_runs")))
PILOT3_RUNS = Path(os.environ.get("PILOT3_RUNS", str(WORK / "pilot3" / "runs" / "final")))
PILOT3_JUDGE = Path(os.environ.get("PILOT3_JUDGE", str(WORK / "pilot3" / "tools" / "pilot3_judge.py")))

PARENTS = {42: "sha256:41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
           52: "sha256:2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
           60: "sha256:2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2"}
STANCE_PARENT = "sha256:1c9d87c3d0f817b048de429bb80cdc89fcd2c83d5fbafd7f444af0b5474b9ff6"  # walking pilot 3's parent
PILOT1_PARENT = "sha256:70f3a4447794a01036c27bca5e0377a17db96d0e3d7bfb2d179e9ce94e6f353e"
PILOT3_TASK = "sha256:782d95b607d0086d0cefe475c9caf8ac420c1f2b74d23b786808aaee98232a6a"
STANCE_TASK = "sha256:1e5964b61a4f0e3a76e9af2b6f76f485cf6e3d2409d48f32f3a7799626a6a456"
HP_P4 = "sha256:fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62"
TASK_P4 = "sha256:914a1caee2c086a09ef66e9e6d4ad234dd085b69c4da687cc29af82b110fcd7a"
PILOT3_HEAD = "31d02e42d148511e50f9cba9a598d9a9d36c23af"
#: The lineage the setup check measured on the edge-prepared model (SUMMARY_table.txt, setup_summary.json).
LINEAGE = {"mode": "initialize_next_stage", "parent_task_sha256": PILOT3_TASK, "child_task_sha256": TASK_P4,
           "parent_species": "tyrannosaurus_rex", "parent_embodiment": "anatomical", "parent_stage": 2,
           "zeroed_command_parameters": ["mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"]}
#: What pilot 3's 8M zips record (read from their JSON data member; dropped_critic_phase abbreviated).
PILOT3_8M_LINEAGE = {"mode": "initialize_next_stage", "parent_task_sha256": STANCE_TASK,
                     "child_task_sha256": PILOT3_TASK, "parent_species": "tyrannosaurus_rex",
                     "parent_embodiment": "anatomical", "parent_stage": 1,
                     "dropped_critic_phase": {"schema": "mesozoic.critic-phase-drop/v1"}}
#: The [env.command_config] table the text fixes ("What is trained"), written once, every number a float.
COMMAND_TABLE = {"cruise_speed": 1.5, "stop_probability": 0.0, "turn_increment_max": 0.0, "straight_probability": 1.0,
                 "speed_scale": 1.5, "lateral_speed_scale": 1.0, "yaw_rate_scale": 0.6, "yaw_gain": 1.5,
                 "yaw_rate_max": 0.3, "turn_slowdown": True, "minimum_turn_speed_fraction": 0.25}
P3_RECORD = {"schema": "mesozoic.critic-phase/v1", "features": ["ramp", "settled", "discount"], "settle_steps": 200,
             "gamma": 0.98, "width": 3}

SC_TEMPLATE_JSON = (
    '{"species":"tyrannosaurus_rex","embodiment":"anatomical","stage":2,"name":"locomotion","description"'
    ':"Learn forward walking/running","algorithm":"PPO","library_version":"0.4.0.dev2","git_commit":"0bcb'
    '0cf941ba6d9d44cc4020e8dff59f773085a9","reward_weights":{"frame_skip":5,"max_episode_steps":1000,"for'
    'ward_vel_weight":2.0,"alive_bonus":0.5,"energy_penalty_weight":0.002,"fall_penalty":-150.0,"tail_sta'
    'bility_weight":0.05,"bite_bonus":0.0,"bite_approach_weight":0.0,"bite_head_proximity_weight":0.0,"po'
    'sture_weight":0.2,"nosedive_weight":0.5,"natural_pitch":0.017,"height_weight":0.3,"height_target_tol'
    'erance":0.0,"gait_symmetry_weight":0.0,"smoothness_weight":0.05,"heading_weight":0.3,"lateral_penalt'
    'y_weight":0.1,"backward_vel_penalty_weight":0.0,"drift_penalty_weight":0.0,"spin_penalty_weight":0.0'
    ',"speed_penalty_weight":0.0,"speed_penalty_threshold":0.1,"idle_penalty_weight":0.0,"idle_velocity_t'
    'hreshold":0.05,"forward_vel_max":1.5,"foot_contact_weight":0.0,"foot_contact_gate":0.0,"bilateral_su'
    'pport_weight":0.0,"foot_contact_saturation_force":100.0,"foot_load_balance_weight":0.0,"foot_load_ba'
    'lance_min_support_force":0.0,"foot_load_balance_airborne_penalty":0.0,"action_jerk_weight":0.0,"supp'
    'ort_conditioned_alive_fraction":0.0,"leg_home_pose_weight":0.0,"leg_home_pose_tolerance":0.35,"head_'
    'clearance_weight":0.0,"head_clearance_target":0.6,"head_clearance_tolerance":0.48,"neck_posture_weig'
    'ht":0.0,"neck_posture_tolerance":0.35,"neck_posture_reference":"keyframe","tail_home_pose_weight":0.'
    '0,"tail_home_pose_tolerance":0.1,"action_saturation_weight":0.0,"action_saturation_threshold":0.9,"l'
    'eg_home_pose_broad_fraction":0.0,"leg_home_pose_broad_scale":6.0,"foot_flatness_weight":0.0,"foot_fl'
    'atness_tolerance_deg":3.0,"stance_width_weight":0.0,"stance_width_tolerance_m":0.05,"stance_width_re'
    'ference":"keyframe","stance_width_settle_steps":200,"foot_terms_min_support_force":0.0,"floor_impact'
    '_weight":0.0,"floor_impact_threshold_bw":1.4,"airborne_substep_weight":1.0,"action_penalty_source":"'
    'filtered","walk_support_gate_bw":0.04,"walk_touchdown_weight":0.1,"walk_heading_price_weight":0.2,"w'
    'alk_heading_price_scale_rad":0.5,"nosedive_termination_threshold":0.62,"prey_distance_range":[8.0,12'
    '.0],"prey_lateral_range":[0.0,0.0],"healthy_z_range":[0.7,1.55],"reset_noise_scale":0.05,"perturbati'
    'on_capture_velocity_multiple":0.0,"perturbation_interval":2.0,"perturbation_jitter":0.5,"perturbatio'
    'n_duration":0.2,"perturbation_direction":"uniform_horizontal","command_mode":"heading","command_conf'
    'ig":{"cruise_speed":1.5,"stop_probability":0.0,"turn_increment_max":0.0,"straight_probability":1.0,"'
    'speed_scale":1.5,"lateral_speed_scale":1.0,"yaw_rate_scale":0.6,"yaw_gain":1.5,"yaw_rate_max":0.3,"t'
    'urn_slowdown":true,"minimum_turn_speed_fraction":0.25}},"hyperparameters":{"learning_rate":5e-05,"le'
    'arning_rate_end":1e-05,"n_steps":4096,"batch_size":256,"n_epochs":10,"gamma":0.995,"gae_lambda":0.95'
    ',"clip_range":0.2,"ent_coef":0.005,"ent_coef_end":0.001,"ent_coef_decay_timesteps":4000000,"target_k'
    'l":0.05,"policy_kwargs":{"net_arch":[512,256]}},"curriculum":{"gate_schema_version":1,"gate_kind":"r'
    'eward_and_length/v1","timesteps":2000000,"eval_freq":65536,"eval_episodes":10,"checkpoint_freq":2500'
    '00,"max_checkpoints":0,"initialize_from_own_checkpoint":true,"min_avg_reward":100.0,"collapse_peak_f'
    'loor_reference":1064.0,"collapse_peak_floor_fraction":0.45,"statue_constants_physics_revision":10,"m'
    'in_avg_episode_length":750,"min_avg_forward_vel":1.0,"required_consecutive":3,"collapse_min_evals":2'
    '0,"collapse_patience":10,"collapse_drop_fraction":0.5,"warmup_timesteps":0,"warmup_clip_range":0.02,'
    '"warmup_ent_coef":0.02,"ramp_timesteps":2000000,"ramp_start_value":2.0},"run":{"seed":42,"n_envs":4,'
    '"timesteps":2000000,"hyperparameters_sha256":"sha256:fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e'
    '7b7ba4477418eaa62","label":"walking pilot 4 setup check s42","load_path":"/tmp/claude-0/-home-user-m'
    'esozoic-labs/aff94cce-91ba-5515-b81b-a9f018e73dc9/scratchpad/work/pilot4/build/check/pairs/s42/model'
    's/stage2_8000000_steps.zip","load_mode":"initialize_next_stage","parent_checkpoint_sha256":"sha256:4'
    '1fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4","parent_task_sha256":"sha256:782d95'
    'b607d0086d0cefe475c9caf8ac420c1f2b74d23b786808aaee98232a6a"},"plant_identity":{"schema":"mesozoic.pl'
    'ant-identity/v2","species":"tyrannosaurus_rex","embodiment":"anatomical","model_path":"environments/'
    'tyrannosaurus_rex/anatomical/assets/tyrannosaurus_rex_anatomical.xml","physics_revision":10,"policy_'
    'interface_revision":15,"visual_revision":4,"source_closure_sha256":"sha256:ccda5ddff3602676409ce79f2'
    '8373a56e8ac069f37ab4b83808ef02aae8442e0","policy_interface_sha256":"sha256:ffbf6631e4de0469cfcaa6771'
    'e80f47973b127342bc362f74190451e8555cbd8","physics_sha256":"sha256:6a47a0eb78164b9738752e2cf928aa0a6f'
    'd3ef98fc953eddc9bcb2eca366da4d","visual_sha256":"sha256:7042dcd597fbd0ce25489164a7f132d51e1c2adbc050'
    '729c9866d7bde42e35a4","nq":28,"nv":27,"nu":15,"observation_dim":63,"action_dim":15},"task_fingerprin'
    't":{"schema":"mesozoic.task-fingerprint/v3","species":"tyrannosaurus_rex","embodiment":"anatomical",'
    '"stage":2,"backend":"stable-baselines3","plant":{"physics_sha256":"sha256:6a47a0eb78164b9738752e2cf9'
    '28aa0a6fd3ef98fc953eddc9bcb2eca366da4d","policy_interface_sha256":"sha256:ffbf6631e4de0469cfcaa6771e'
    '80f47973b127342bc362f74190451e8555cbd8"},"env":{"frame_skip":5,"max_episode_steps":1000,"forward_vel'
    '_weight":2.0,"alive_bonus":0.5,"energy_penalty_weight":0.002,"fall_penalty":-150.0,"tail_stability_w'
    'eight":0.05,"bite_bonus":0.0,"bite_approach_weight":0.0,"bite_head_proximity_weight":0.0,"posture_we'
    'ight":0.2,"nosedive_weight":0.5,"natural_pitch":0.017,"height_weight":0.3,"height_target_tolerance":'
    '0.0,"gait_symmetry_weight":0.0,"smoothness_weight":0.05,"heading_weight":0.3,"lateral_penalty_weight'
    '":0.1,"backward_vel_penalty_weight":0.0,"drift_penalty_weight":0.0,"spin_penalty_weight":0.0,"speed_'
    'penalty_weight":0.0,"speed_penalty_threshold":0.1,"idle_penalty_weight":0.0,"idle_velocity_threshold'
    '":0.05,"forward_vel_max":1.5,"foot_contact_weight":0.0,"foot_contact_gate":0.0,"bilateral_support_we'
    'ight":0.0,"foot_contact_saturation_force":100.0,"foot_load_balance_weight":0.0,"foot_load_balance_mi'
    'n_support_force":0.0,"foot_load_balance_airborne_penalty":0.0,"action_jerk_weight":0.0,"support_cond'
    'itioned_alive_fraction":0.0,"leg_home_pose_weight":0.0,"leg_home_pose_tolerance":0.35,"head_clearanc'
    'e_weight":0.0,"head_clearance_target":0.6,"head_clearance_tolerance":0.48,"neck_posture_weight":0.0,'
    '"neck_posture_tolerance":0.35,"neck_posture_reference":"keyframe","tail_home_pose_weight":0.0,"tail_'
    'home_pose_tolerance":0.1,"action_saturation_weight":0.0,"action_saturation_threshold":0.9,"leg_home_'
    'pose_broad_fraction":0.0,"leg_home_pose_broad_scale":6.0,"foot_flatness_weight":0.0,"foot_flatness_t'
    'olerance_deg":3.0,"stance_width_weight":0.0,"stance_width_tolerance_m":0.05,"stance_width_reference"'
    ':"keyframe","stance_width_settle_steps":200,"foot_terms_min_support_force":0.0,"floor_impact_weight"'
    ':0.0,"floor_impact_threshold_bw":1.4,"airborne_substep_weight":1.0,"action_penalty_source":"filtered'
    '","walk_support_gate_bw":0.04,"walk_touchdown_weight":0.1,"walk_heading_price_weight":0.2,"walk_head'
    'ing_price_scale_rad":0.5,"nosedive_termination_threshold":0.62,"prey_distance_range":[8.0,12.0],"pre'
    'y_lateral_range":[0.0,0.0],"healthy_z_range":[0.7,1.55],"reset_noise_scale":0.05,"perturbation_captu'
    're_velocity_multiple":0.0,"perturbation_interval":2.0,"perturbation_jitter":0.5,"perturbation_durati'
    'on":0.2,"perturbation_direction":"uniform_horizontal","command_mode":"heading","command_config":{"cr'
    'uise_speed":1.5,"stop_probability":0.0,"turn_increment_max":0.0,"straight_probability":1.0,"speed_sc'
    'ale":1.5,"lateral_speed_scale":1.0,"yaw_rate_scale":0.6,"yaw_gain":1.5,"yaw_rate_max":0.3,"turn_slow'
    'down":true,"minimum_turn_speed_fraction":0.25}},"perturbation":null,"command":{"schema":"mesozoic.di'
    'rection-commands/v1","config":{"cruise_speed":1.5,"speed_range":null,"stop_probability":0.0,"turn_in'
    'crement_max":0.0,"straight_probability":1.0,"switch_interval_s":4.0,"switch_jitter_s":0.0,"speed_sca'
    'le":1.5,"lateral_speed_scale":1.0,"yaw_rate_scale":0.6,"yaw_gain":1.5,"yaw_rate_max":0.3,"turn_slowd'
    'own":true,"minimum_turn_speed_fraction":0.25},"request_frame":"world_heading_radians_and_speed_metre'
    's_per_second","policy_frame":"gravity_aligned_body_heading_vx_vy_yaw_rate","policy_components":["v_x'
    '_cmd","v_y_cmd","yaw_rate_cmd"],"policy_units":["m/s","m/s","rad/s"],"time_unit":"seconds","heading_'
    'wrap":"[-pi, pi); exact half-turn clockwise","schedule":"PCG64; increments relative to previous requ'
    'ested world heading","adapter":"bounded_proportional_yaw_with_optional_cosine_speed_reduction/v1","s'
    'top":"zero forward/lateral/yaw commands; heading masked","external_target":"persists until reset"},"'
    'task_sha256":"sha256:914a1caee2c086a09ef66e9e6d4ad234dd085b69c4da687cc29af82b110fcd7a"}}'
)
SC_TEMPLATE = json.loads(SC_TEMPLATE_JSON)


def template_sc(seed: int = 42) -> dict:
    """The setup check's record for --seed (its s52 and s60 records differ from s42's only in these four keys)."""
    sc = copy.deepcopy(SC_TEMPLATE)
    if seed != 42:
        run = sc["run"]
        run["seed"] = seed
        run["label"] = f"walking pilot 4 setup check s{seed}"
        run["load_path"] = run["load_path"].replace("/pairs/s42/", f"/pairs/s{seed}/")
        run["parent_checkpoint_sha256"] = PARENTS[seed]
    return sc


def good_sc(seed: int = 42, **run_over) -> dict:
    sc = template_sc(seed)
    sc["run"].update(run_over)
    return sc


def problems_of(sc: dict, *, seed=42, want=None, budget=2_000_000, report=False, lineage=LINEAGE, note=None,
                table=COMMAND_TABLE, ramp=2_000_000, warmup=0, ramp_start=2.0, forward=2.0, hp=HP_P4, task=TASK_P4):
    want = pj.norm_sha(PARENTS[seed] if want is None else want)
    return pj.binding_problems(sc, seed=seed, want_parent=want, expect_budget=budget, ramp=ramp, warmup=warmup,
                               ramp_start=ramp_start, forward_weight=forward, hp_here=hp, task_here=task,
                               committed_table=table, lineage=lineage, lineage_note=note, report_off_schedule=report)


# ---------------------------------------------------------------- the template and the constants

def test_the_template_is_the_setup_check_s_record():
    for seed in (42, 52, 60):
        path = RECORDED / f"pilot4_walk_s{seed}" / "03_locomotion" / "stage_config.json"
        if not path.is_file():
            pytest.skip(f"no setup-check record at {path}")
        assert template_sc(seed) == json.loads(path.read_text()), seed


def test_the_template_records_the_text_s_values():
    sc = SC_TEMPLATE
    assert sc["git_commit"] == "0bcb0cf941ba6d9d44cc4020e8dff59f773085a9"
    assert sc["run"]["hyperparameters_sha256"] == HP_P4 and sc["task_fingerprint"]["task_sha256"] == TASK_P4
    assert sc["run"]["parent_task_sha256"] == PILOT3_TASK and sc["run"]["parent_checkpoint_sha256"] == PARENTS[42]
    assert pj.same_json(sc["reward_weights"]["command_config"], COMMAND_TABLE)
    assert "critic_phase_dropped" not in sc["run"] and "warm_start_log_std" not in sc["run"]
    assert "warm_start_log_std" not in sc["curriculum"]


@pytest.mark.parametrize("budget, end", [(2_000_000, 2_015_232), (8_000_000, 8_011_776), (50_000, 65_536),
                                         (65_536, 65_536), (32_768, 32_768), (16_384, 16_384), (16_385, 32_768)])
def test_uninterrupted_end_is_whole_rollouts_at_or_past_the_budget(budget, end):
    assert pj.uninterrupted_end(budget, 4 * 4096) == end


def test_constants_are_the_criteria_s():
    assert pj.PILOT_BUDGET == 2_000_000 and pj.PILOT_UNINTERRUPTED_END == 2_015_232
    assert pj.uninterrupted_end(pj.PILOT_BUDGET, pj.N_ENVS * 4096) == pj.PILOT_UNINTERRUPTED_END == 123 * 16_384
    assert {s: "sha256:" + v for s, v in pj.PARENT_SHA256.items()} == PARENTS
    assert "sha256:" + pj.PARENT_TASK_SHA256 == PILOT3_TASK
    assert pj.N_ENVS == 4 and pj.WARMUP_TIMESTEPS == 0 and pj.RAMP_START_VALUE == 2.0
    assert pj.COMMAND_MODE == "heading" and pj.PREY_LATERAL_RANGE == (0.0, 0.0)
    assert pj.PRICE_KEYS == {"walk_heading_price_weight": 0.2, "walk_heading_price_scale_rad": 0.5}
    assert pj.R6_KEYS == {"walk_support_gate_bw": 0.04, "airborne_substep_weight": 1.0, "forward_vel_max": 1.5,
                          "walk_touchdown_weight": 0.1}
    assert pj.CADENCE == {"eval_freq": 65536, "eval_episodes": 10, "checkpoint_freq": 250000, "max_checkpoints": 0}
    assert pj.LINEAGE_ATTRIBUTE == "mesozoic_task_lineage" and pj.LINEAGE_PARENT_STAGE == 2
    assert pj.PROBE_ARGS == ["--species", "tyrannosaurus_rex", "--stage", "locomotion", "--episodes", "40",
                             "--seed-scheme", "panel", "--seed-base", "3042", "--settle-steps", "100"]
    assert pj.BRANCH_PREFIX.endswith("/pilot-walk-") and pj.BRANCH_PREFIX.count("/") == 1  # pilot 3's: next test
    assert not hasattr(pj, "PILOT_PARENT_SHA256") and not hasattr(pj, "PARENT_IN_TREE")
    assert not hasattr(pj, "warm_start_problems") and not hasattr(pj, "parent_phase_record")


@pytest.mark.parametrize("name", ["PROBE_ARGS", "BRANCH_PREFIX", "N_ENVS", "FINAL_ZIP", "FINAL_VEC"])
def test_the_kept_constants_are_pilot_3_s_unchanged(name):
    if not PILOT3_JUDGE.is_file():
        pytest.skip(f"no pilot-3 judge at {PILOT3_JUDGE}")
    tree = ast.parse(PILOT3_JUDGE.read_text())  # read as text: nothing of pilot 3's folder is imported or written
    found = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
    if not found:  # FINAL_ZIP, FINAL_VEC = ... is one tuple assignment
        found = [dict(zip((t.id for t in n.targets[0].elts), ast.literal_eval(n.value)))[name] for n in tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                 and name in [t.id for t in n.targets[0].elts]]
    assert found == [getattr(pj, name)]


def test_the_kept_functions_are_pilot_3_s_byte_for_byte():
    """The functions the plan keeps are pilot 3's source text but for their docstrings (resume_check and
    final_fallback_problems change as the plan says; binding_problems is rewritten around the new checks)."""
    if not PILOT3_JUDGE.is_file():
        pytest.skip(f"no pilot-3 judge at {PILOT3_JUDGE}")
    mine, theirs = (TOOLS / "pilot4_judge.py").read_text(), PILOT3_JUDGE.read_text()

    def source(text: str, name: str) -> str:
        node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == name)
        body = ast.get_source_segment(text, node)
        doc = ast.get_docstring(node, clean=False)
        return body.replace(doc, "") if doc else body
    for name in ("sha", "norm_sha", "refuse", "git", "read_zip_data", "cell_path", "run_end", "uninterrupted_end",
                 "other_run_cells", "commit_problem"):
        assert source(mine, name) == source(theirs, name), name


def test_the_parent_table_is_the_verified_run_copies():
    for seed, want in PARENTS.items():
        path = PILOT3_RUNS / f"pilot3_walk_s{seed}" / "03_locomotion" / "models" / "stage2_8000000_steps.zip"
        if not path.is_file():
            pytest.skip(f"no run copy at {path}")
        assert "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest() == want


# ---------------------------------------------------------------- pure functions: the binding

@pytest.mark.parametrize("seed", [42, 52, 60])
def test_a_run_trained_as_fixed_has_no_problem(seed):
    problems, info = problems_of(good_sc(seed), seed=seed)
    assert problems == [] and info == {"budget": 2_000_000, "resume_step": None, "off_schedule": None,
                                       "resumed": False, "cadence_checked": True}


def _cmd(sc):
    return sc["reward_weights"]["command_config"]


@pytest.mark.parametrize("edit, needle", [
    # entry_problems (the plan's list)
    (lambda sc: sc["run"].update(critic_phase_dropped=P3_RECORD), "run.critic_phase_dropped is recorded ({"),
    (lambda sc: sc["run"].update(critic_phase_dropped=None), "run.critic_phase_dropped is recorded (None)"),
    (lambda sc: sc["run"].update(warm_start_log_std=-2.0), "run.warm_start_log_std is recorded (-2.0)"),
    (lambda sc: sc["curriculum"].update(warm_start_log_std=-2.0), "curriculum.warm_start_log_std is recorded (-2.0)"),
    (lambda sc: sc["curriculum"].update(warm_start_log_std=-1.5), "curriculum.warm_start_log_std is recorded (-1.5)"),
    # fix stage (review: a null record must be refused too, as critic_phase_dropped's is)
    (lambda sc: sc["run"].update(warm_start_log_std=None), "run.warm_start_log_std is recorded (None)"),
    (lambda sc: sc["curriculum"].update(warm_start_log_std=None), "curriculum.warm_start_log_std is recorded (None)"),
    (lambda sc: sc["curriculum"].update(warmup_timesteps=300_000), "curriculum.warmup_timesteps is 300000, not 0"),
    (lambda sc: sc["curriculum"].pop("warmup_timesteps"), "curriculum.warmup_timesteps is absent (not 0)"),
    (lambda sc: sc["curriculum"].update(warmup_timesteps=False), "curriculum.warmup_timesteps is False, not 0"),
    (lambda sc: sc["curriculum"].update(warmup_timesteps="0"), "curriculum.warmup_timesteps is '0', not 0"),
    (lambda sc: sc["curriculum"].update(ramp_start_value=0.2), "curriculum.ramp_start_value is 0.2, not 2.0"),
    (lambda sc: sc["curriculum"].pop("ramp_start_value"), "curriculum.ramp_start_value is absent (not 2.0)"),
    (lambda sc: sc["curriculum"].update(ramp_start_value="2.0"), "curriculum.ramp_start_value is '2.0', not 2.0"),
    (lambda sc: sc["curriculum"].update(initialize_from_own_checkpoint=False),
     "curriculum.initialize_from_own_checkpoint is False, not true"),
    (lambda sc: sc["curriculum"].pop("initialize_from_own_checkpoint"),
     "curriculum.initialize_from_own_checkpoint is None, not true"),
    (lambda sc: sc["curriculum"].update(initialize_from_own_checkpoint="true"),
     "curriculum.initialize_from_own_checkpoint is 'true', not true"),
    (lambda sc: sc["curriculum"].update(initialize_from_own_checkpoint=1),
     "curriculum.initialize_from_own_checkpoint is 1, not true"),
    (lambda sc: sc["run"].update(n_envs=2), "run.n_envs is 2, not 4"),
    (lambda sc: sc["run"].pop("n_envs"), "run.n_envs is None, not 4"),
    (lambda sc: sc["run"].update(n_envs=4.0000001), "run.n_envs is 4.0000001"),
    (lambda sc: sc["run"].update(n_envs=True), "run.n_envs is True, not 4"),
    (lambda sc: sc["hyperparameters"].update(critic_phase=False), "hold a critic_phase key (False)"),
    (lambda sc: sc["hyperparameters"].update(critic_phase=True), "hold a critic_phase key (True)"),
    (lambda sc: sc["run"].pop("parent_task_sha256"), "run.parent_task_sha256 is None, not pilot 3's command-free task"),
    (lambda sc: sc["run"].update(parent_task_sha256=STANCE_TASK), "run.parent_task_sha256 is 'sha256:1e5964b6"),
    (lambda sc: sc["run"].update(parent_task_sha256=TASK_P4), "run.parent_task_sha256 is 'sha256:914a1cae"),
    # pilot 3's, kept
    (lambda sc: sc["run"].update(seed=52), "seed 52 (not --seed 42)"),
    (lambda sc: sc["run"].update(load_mode="resume_same_stage"), "load_mode 'resume_same_stage'"),
    (lambda sc: sc["run"].update(timesteps=8_000_000), "budget 8000000 (not 2000000)"),
    (lambda sc: sc["run"].update(timesteps=2_015_232), "budget 2015232 (not 2000000)"),
    (lambda sc: sc["run"].update(hyperparameters_sha256="sha256:" + "cd" * 32), "!= this tree's"),
    (lambda sc: sc.update(stage=1), "stage 1"),
    (lambda sc: sc["run"].update(parent_checkpoint_sha256=PARENTS[52]), "is not the chosen parent"),
    # the task digest
    (lambda sc: sc["task_fingerprint"].update(task_sha256=PILOT3_TASK), f"task_sha256 {PILOT3_TASK} != this tree's"),
    (lambda sc: sc.pop("task_fingerprint"), "stage_config.json records no task_fingerprint.task_sha256"),
    # the env binding
    (lambda sc: sc["reward_weights"].update(command_mode="none"),
     "reward_weights.command_mode is 'none', not 'heading'"),
    (lambda sc: sc["reward_weights"].pop("command_mode"), "reward_weights.command_mode is None, not 'heading'"),
    (lambda sc: _cmd(sc).update(yaw_gain=2.0), "key for key: yaw_gain = 2.0 (float), committed 1.5 (float)"),
    (lambda sc: _cmd(sc).update(switch_interval_s=4.0), "key for key: switch_interval_s = 4.0 added"),
    (lambda sc: _cmd(sc).pop("turn_slowdown"), "key for key: turn_slowdown missing"),
    (lambda sc: _cmd(sc).update(turn_slowdown=1), "turn_slowdown = 1 (int), committed True (bool)"),
    (lambda sc: _cmd(sc).update(stop_probability=0), "stop_probability = 0 (int), committed 0.0 (float)"),
    (lambda sc: _cmd(sc).update(speed_scale="1.5"), "speed_scale = '1.5' (str), committed 1.5 (float)"),
    (lambda sc: sc["reward_weights"].update(command_config=None), "the recorded command_config is not a table: None"),
    (lambda sc: sc["reward_weights"].update(prey_lateral_range=[-2.0, 2.0]),
     "reward_weights.prey_lateral_range is [-2.0, 2.0], not [0.0, 0.0]"),
    (lambda sc: sc["reward_weights"].update(prey_lateral_range=[0.0]), "prey_lateral_range is [0.0], not [0.0, 0.0]"),
    (lambda sc: sc["reward_weights"].pop("prey_lateral_range"), "prey_lateral_range is None, not [0.0, 0.0]"),
    (lambda sc: sc["reward_weights"].update(prey_lateral_range=[False, False]), "prey_lateral_range is [False, False]"),
    (lambda sc: sc["reward_weights"].update(walk_heading_price_weight=0.0),
     "reward_weights.walk_heading_price_weight is 0.0, not 0.2"),
    (lambda sc: sc["reward_weights"].update(walk_heading_price_weight=True),
     "reward_weights.walk_heading_price_weight is True, not 0.2"),
    (lambda sc: sc["reward_weights"].update(walk_heading_price_scale_rad=1.0),
     "reward_weights.walk_heading_price_scale_rad is 1.0, not 0.5"),
    (lambda sc: sc["reward_weights"].update(forward_vel_max=2.5), "reward_weights.forward_vel_max is 2.5, not 1.5"),
    (lambda sc: sc["reward_weights"].pop("walk_support_gate_bw"),
     "reward_weights.walk_support_gate_bw is absent (not 0.04)"),
    (lambda sc: sc["reward_weights"].update(airborne_substep_weight=0.0),
     "reward_weights.airborne_substep_weight is 0.0, not 1.0"),
    (lambda sc: sc["reward_weights"].update(walk_touchdown_weight="0.1"),
     "reward_weights.walk_touchdown_weight is '0.1', not 0.1"),
    (lambda sc: sc.pop("reward_weights"), "stage_config.json records no env kwargs table (reward_weights: None)"),
    # fix stage (review: near values; no tolerance, no widened prey, only 4 envs)
    (lambda sc: sc["reward_weights"].update(walk_heading_price_weight=0.2000001),
     "walk_heading_price_weight is 0.2000001, not 0.2"),
    (lambda sc: sc["reward_weights"].update(walk_heading_price_scale_rad=0.4999),
     "walk_heading_price_scale_rad is 0.4999, not 0.5"),
    (lambda sc: sc["reward_weights"].update(forward_vel_max=1.5001), "forward_vel_max is 1.5001, not 1.5"),
    (lambda sc: sc["reward_weights"].update(walk_support_gate_bw=0.0400001), "walk_support_gate_bw is 0.0400001"),
    (lambda sc: sc["reward_weights"].update(prey_lateral_range=[0.0, 0.5]),
     "prey_lateral_range is [0.0, 0.5], not [0.0, 0.0]"),
    (lambda sc: sc["reward_weights"].update(prey_lateral_range=[-1e-9, 0.0]), "prey_lateral_range is [-1e-09, 0.0]"),
    (lambda sc: sc["reward_weights"].update(prey_lateral_range=[0.0, 0.0, 0.0]),
     "prey_lateral_range is [0.0, 0.0, 0.0]"),
    (lambda sc: sc["run"].update(n_envs=8), "run.n_envs is 8, not 4"),
    # the cadence (a Run handling record)
    (lambda sc: sc["curriculum"].update(eval_freq=131072), "curriculum.eval_freq is 131072, not 65536"),
    (lambda sc: sc["curriculum"].update(eval_freq=65536.0), "curriculum.eval_freq is 65536.0, not 65536"),
    (lambda sc: sc["curriculum"].update(checkpoint_freq=500000), "curriculum.checkpoint_freq is 500000, not 250000"),
    (lambda sc: sc["curriculum"].update(max_checkpoints=16), "curriculum.max_checkpoints is 16, not 0"),
    (lambda sc: sc["curriculum"].pop("eval_episodes"), "curriculum.eval_episodes is None, not 10"),
])
def test_each_binding_record_is_refused_when_absent_or_different(edit, needle):
    sc = good_sc()
    edit(sc)
    problems, _ = problems_of(sc)
    assert len(problems) == 1 and needle in problems[0], problems


@pytest.mark.parametrize("edit", [
    lambda sc: sc["curriculum"].update(ramp_start_value=2),
    lambda sc: sc["curriculum"].update(warmup_timesteps=0.0),
    lambda sc: sc["reward_weights"].update(prey_lateral_range=[0, 0]),
    lambda sc: sc["reward_weights"].update(prey_lateral_range=(0.0, 0.0)),
    lambda sc: sc["reward_weights"].update(walk_touchdown_weight=0.1, forward_vel_max=1.5),
    lambda sc: sc["run"].update(parent_task_sha256=PILOT3_TASK.upper().replace("SHA256:", "sha256:")),
    lambda sc: sc["run"].update(parent_checkpoint_sha256=PARENTS[42][len("sha256:"):]),
])
def test_numbers_are_compared_as_numbers_and_digests_as_digests(edit):
    """Pinned reading: the entry, prey, price and R6 values are numbers (a bool or a string is refused, 2 equals 2.0);
    only the command table, which the task digest hashes as written, is compared type for type."""
    sc = good_sc()
    edit(sc)
    assert problems_of(sc)[0] == []


def test_the_price_off_is_refused_by_both_keys():
    """With the price off, save_stage_config records neither opt-in key (an inert opt-in records none)."""
    sc = good_sc()
    sc["reward_weights"].pop("walk_heading_price_weight")
    sc["reward_weights"].pop("walk_heading_price_scale_rad")
    problems, _ = problems_of(sc)
    assert problems == ["reward_weights.walk_heading_price_weight is absent (not 0.2)",
                        "reward_weights.walk_heading_price_scale_rad is absent (not 0.5)"]


def test_pilot_3_s_stance_parent_is_refused_both_as_the_argument_and_against_the_run():
    problems, _ = problems_of(good_sc(), want=STANCE_PARENT)
    assert len(problems) == 2, problems
    assert "is not s42's own 8M parent" in problems[0] and "is not the chosen parent" in problems[1]
    problems, _ = problems_of(good_sc(parent_checkpoint_sha256=STANCE_PARENT), want=STANCE_PARENT)
    assert len(problems) == 1 and "is not s42's own 8M parent" in problems[0]
    problems, _ = problems_of(good_sc(parent_checkpoint_sha256=PILOT1_PARENT), want=PILOT1_PARENT)
    assert len(problems) == 1 and "is not s42's own 8M parent" in problems[0]


@pytest.mark.parametrize("seed, other", [(42, 52), (42, 60), (52, 42), (52, 60), (60, 42), (60, 52)])
def test_another_seed_s_8m_parent_is_refused(seed, other):
    problems, _ = problems_of(good_sc(seed, parent_checkpoint_sha256=PARENTS[other]), seed=seed, want=PARENTS[other])
    assert len(problems) == 1 and f"is not s{seed}'s own 8M parent" in problems[0]
    assert f"(it is s{other}'s)" in problems[0]
    problems, _ = problems_of(good_sc(seed), seed=seed, want=PARENTS[other])
    assert len(problems) == 2 and "is not the chosen parent" in problems[1]


def test_a_seed_outside_the_table_is_refused():
    sc = good_sc()
    sc["run"]["seed"] = 7
    problems, _ = problems_of(sc, seed=7, want=PARENTS[42])
    assert problems == ["--seed 7 is not a pilot-4 seed (42, 52, 60): it has no parent in the table"]


def test_the_parent_problem_names_each_case():
    assert pj.parent_problems(42, pj.norm_sha(PARENTS[42]), PARENTS[42]) == []
    assert pj.parent_problems(42, pj.norm_sha(PARENTS[42]), None) == [
        f"parent_checkpoint_sha256 None is not the chosen parent {PARENTS[42]}"]


def test_pilot_3_s_own_record_is_not_trained_as_fixed():
    path = PILOT3_RUNS / "pilot3_walk_s42" / "03_locomotion" / "stage_config.json"
    if not path.is_file():
        pytest.skip(f"no run copy at {path}")
    problems, _ = problems_of(json.loads(path.read_text()), lineage=PILOT3_8M_LINEAGE)
    text = "\n".join(problems)
    for needle in ("is not the chosen parent", "budget 8000000",
                   "run.critic_phase_dropped is recorded", "run.warm_start_log_std is recorded (-2.0)",
                   "curriculum.warm_start_log_std is recorded (-2.0)", "curriculum.warmup_timesteps is 300000",
                   "curriculum.ramp_start_value is 0.2", "initialize_from_own_checkpoint is None",
                   "run.parent_task_sha256 is 'sha256:1e5964b6", "parent_stage is 1, not 2",
                   "records no zeroed_command_parameters", "command_mode is 'none'",
                   "prey_lateral_range is [-2.0, 2.0]", "walk_heading_price_weight is absent",
                   f"task_sha256 {PILOT3_TASK} != this tree's", "hyperparameters_sha256 sha256:85881fc4",
                   "curriculum.eval_freq is 131072", "curriculum.checkpoint_freq is 500000"):
        assert needle in text, (needle, problems)


def test_every_entry_problem_is_listed_at_once():
    sc = good_sc(n_envs=1, critic_phase_dropped=P3_RECORD, warm_start_log_std=-2.0)
    sc["run"].pop("parent_task_sha256")
    sc["curriculum"].update(warm_start_log_std=-2.0, warmup_timesteps=300_000, ramp_start_value=0.2,
                            initialize_from_own_checkpoint=False)
    sc["hyperparameters"]["critic_phase"] = False
    assert len(pj.entry_problems(sc, PILOT3_8M_LINEAGE)) == 11
    assert len(pj.entry_problems(sc, None, "the judged zip has no readable data member (KeyError: 'data')")) == 10


# the lineage record of the judged zip

@pytest.mark.parametrize("lineage, needles", [
    (None, ["records no mesozoic_task_lineage"]),
    ("initialize_next_stage", ["is not a record: 'initialize_next_stage'"]),
    (PILOT3_8M_LINEAGE, ["parent_stage is 1, not 2", "records no zeroed_command_parameters"]),
    (dict(LINEAGE, mode="resume_same_stage"), ["lineage mode is 'resume_same_stage', not 'initialize_next_stage'"]),
    ({k: v for k, v in LINEAGE.items() if k != "mode"}, ["lineage mode is None"]),
    (dict(LINEAGE, parent_stage="2"), ["parent_stage is '2', not 2"]),
    (dict(LINEAGE, parent_stage=True), ["parent_stage is True, not 2"]),
    (dict(LINEAGE, parent_stage=1), ["parent_stage is 1, not 2"]),
    ({k: v for k, v in LINEAGE.items() if k != "parent_stage"}, ["parent_stage is None, not 2"]),
    ({k: v for k, v in LINEAGE.items() if k != "zeroed_command_parameters"}, ["records no zeroed_command_parameters"]),
    (dict(LINEAGE, zeroed_command_parameters=[]), ["zeroed_command_parameters is [], not a non-empty list"]),
    (dict(LINEAGE, zeroed_command_parameters="mlp_extractor.policy_net.0.weight"), ["not a non-empty list"]),
    (dict(LINEAGE, zeroed_command_parameters=[""]), ["not a non-empty list"]),
    (dict(LINEAGE, zeroed_command_parameters=[3]), ["not a non-empty list"]),
])
def test_the_lineage_record_is_bound(lineage, needles):
    got = pj.lineage_problems(lineage)
    assert len(got) == len(needles) and all(n in g for n, g in zip(needles, got)), got
    problems, _ = problems_of(good_sc(), lineage=lineage)
    assert len(problems) == len(needles)


def test_the_lineage_binds_only_its_three_fields():
    """Pinned reading of the plan ("initialize_next_stage, parent stage 2, zeroed_command_parameters"): the lineage's
    parent and child task digests and extra keys are recorded in binding.json, not checked."""
    lineage = dict(LINEAGE, child_task_sha256=PILOT3_TASK, parent_task_sha256=None, extra={"x": 1})
    assert pj.lineage_problems(lineage) == []
    assert pj.lineage_problems(dict(LINEAGE, zeroed_command_parameters=["a"])) == []


def test_an_unreadable_lineage_refuses():
    note = "the judged zip has no readable data member (KeyError: \"There is no item named 'data' in the archive\")"
    problems, _ = problems_of(good_sc(), lineage=None, note=note)
    assert problems == [note]


def test_zip_lineage_reads_the_data_member_only(tmp_path):
    z = write_ckpt(tmp_path / "a.zip", 2_000_000, lineage=LINEAGE)
    assert pj.zip_lineage(z) == (LINEAGE, None)
    assert pj.zip_lineage(write_ckpt(tmp_path / "b.zip", 2_000_000, lineage=None)) == (None, None)
    got, note = pj.zip_lineage(write_ckpt(tmp_path / "c.zip", 1, members=("policy.pth",)))
    assert got is None and note.startswith("the judged zip has no readable data member (KeyError")
    (tmp_path / "d.zip").write_bytes(b"not a zip")
    assert pj.zip_lineage(tmp_path / "d.zip")[1].startswith("the judged zip has no readable data member (BadZipFile")


# the cadence

def test_the_cadence_is_checked_at_the_pilot_budget_only():
    sc = good_sc(timesteps=50_000)
    sc["curriculum"].update(eval_freq=32_768, checkpoint_freq=32_768)
    problems, info = problems_of(sc, budget=50_000)
    assert problems == [] and info["cadence_checked"] is False
    problems, info = problems_of(good_sc(), budget=2_000_000)
    assert info["cadence_checked"] is True
    sc = good_sc()
    sc["curriculum"].update(eval_freq=32_768, checkpoint_freq=32_768)
    assert problems_of(sc)[0] == ["curriculum.eval_freq is 32768, not 65536",
                                  "curriculum.checkpoint_freq is 32768, not 250000"]


# fix stage: the text's own digests, table and head (review: a record and a tree that moved together were accepted)

V3_TEXT = WORK / "pilot4" / "postfix" / "fix" / "criteria_pilot4_draft_v3.txt"
MOVED_TABLE = dict(COMMAND_TABLE, switch_interval_s=6.0, switch_jitter_s=1.0)
MOVED_HP = "sha256:" + "ab" * 32
MOVED_TASK = "sha256:" + "cd" * 32


def test_the_text_anchors_are_v3_s_values():
    assert pj.HYPERPARAMETERS_SHA256 == HP_P4 and pj.TASK_SHA256 == TASK_P4
    assert pj.same_json(pj.COMMAND_TABLE, COMMAND_TABLE) and pj.PILOT_HEAD == "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8"
    assert all(isinstance(v, float) for k, v in pj.COMMAND_TABLE.items() if k != "turn_slowdown")
    if not V3_TEXT.is_file():
        pytest.skip(f"no v3 text at {V3_TEXT}")
    text = V3_TEXT.read_text()
    for value in (pj.HYPERPARAMETERS_SHA256, pj.TASK_SHA256, pj.PILOT_HEAD):  # written out in full in v3
        assert value in text, value
    lines = text.splitlines()  # v3 "What is trained" writes the table as "key value" in its left column (47 wide)
    first = next(i for i, ln in enumerate(lines) if "[env.command_config], written once:" in ln)
    last = next(i for i, ln in enumerate(lines) if i > first and "prey_lateral_range = [0.0, 0.0]" in ln)
    left = " ".join(" ".join(ln[:47] for ln in lines[first + 1:last]).split())
    for k, v in pj.COMMAND_TABLE.items():
        assert f"{k} {str(v).lower()}," in left + ",", (k, v, left)
    assert len(left.split(",")) - 1 == len(pj.COMMAND_TABLE) - 1  # eleven "key value" entries, no other


def test_a_record_and_a_tree_that_moved_together_are_refused_at_the_pilot_budget():
    sc = good_sc(hyperparameters_sha256=MOVED_HP)
    sc["task_fingerprint"]["task_sha256"] = MOVED_TASK
    sc["reward_weights"]["command_config"] = copy.deepcopy(MOVED_TABLE)
    problems, info = problems_of(sc, hp=MOVED_HP, task=MOVED_TASK, table=MOVED_TABLE)
    assert info["cadence_checked"] is True
    assert problems == [f"this tree's recipe digest {MOVED_HP} is not v3's {HP_P4}",
                        f"this tree's task digest {MOVED_TASK} is not v3's {TASK_P4}",
                        "this tree's committed [env.command_config] is not v3's table: switch_interval_s = 6.0 added; "
                        "switch_jitter_s = 1.0 added"], problems
    for one in ({"hp": MOVED_HP}, {"task": MOVED_TASK}, {"table": MOVED_TABLE}):
        sc1 = good_sc(hyperparameters_sha256=one.get("hp", HP_P4))
        sc1["task_fingerprint"]["task_sha256"] = one.get("task", TASK_P4)
        sc1["reward_weights"]["command_config"] = copy.deepcopy(one.get("table", COMMAND_TABLE))
        got, _ = problems_of(sc1, **one)
        assert len(got) == 1 and "is not v3's" in got[0], (one, got)


def test_the_text_anchors_are_not_checked_at_a_validation_budget():
    sc = good_sc(timesteps=50_000, hyperparameters_sha256=MOVED_HP)
    sc["task_fingerprint"]["task_sha256"] = MOVED_TASK
    sc["reward_weights"]["command_config"] = copy.deepcopy(MOVED_TABLE)
    problems, info = problems_of(sc, budget=50_000, hp=MOVED_HP, task=MOVED_TASK, table=MOVED_TABLE)
    assert problems == [] and info["cadence_checked"] is False


def test_anchor_problems_alone():
    assert pj.anchor_problems(HP_P4, TASK_P4, copy.deepcopy(COMMAND_TABLE)) == []
    got = pj.anchor_problems(HP_P4, TASK_P4, dict(COMMAND_TABLE, yaw_gain=2.0))
    assert got == ["this tree's committed [env.command_config] is not v3's table: yaw_gain = 2.0 (float), committed "
                   "1.5 (float)"]
    assert len(pj.anchor_problems(HP_P4, TASK_P4, dict(COMMAND_TABLE, cruise_speed=2))) == 1  # an int is not 1.5
    # equal under Python's == but not the same JSON value (the task digest hashes the table as written)
    for edit in ({"stop_probability": 0}, {"turn_slowdown": 1}, {"straight_probability": 1}):
        got = pj.anchor_problems(HP_P4, TASK_P4, dict(COMMAND_TABLE, **edit))
        assert len(got) == 1 and "is not v3's table" in got[0], (edit, got)
    assert len(pj.anchor_problems(HP_P4.upper(), TASK_P4, COMMAND_TABLE)) == 1  # the digest as written


# fix stage: the env kwargs drift (review: a run whose other env weights differ was judged and scored)

def test_env_kwargs_drift_is_the_probe_s_comparison():
    rec = {"a": 1.0, "t": [0.0, 0.0], "dflt_same": 5, "dflt_diff": 6, "n": 2.0, "cfg": {"x": 1.0}}
    env_kwargs = {"a": 1.0, "t": (0.0, 0.0), "only_env": 2, "n": 2, "cfg": {"x": 1.0}}
    defaults = {"dflt_same": 5, "dflt_diff": 5, "a": 9.0, "unused": 7}
    assert pj.env_kwargs_drift(rec, env_kwargs, defaults) == {
        "dflt_diff": {"toml_main": 5, "recorded_run": 6},
        "only_env": {"toml_main": 2, "recorded_run": "<absent>"}}
    assert pj.env_kwargs_drift(dict(rec, extra=1.0), env_kwargs, defaults)["extra"] == {
        "toml_main": "<absent>", "recorded_run": 1.0}
    assert set(pj.env_kwargs_drift(None, env_kwargs, defaults)) == set(env_kwargs)
    assert pj.env_kwargs_drift(dict(rec, a=3.0), env_kwargs, defaults)["a"] == {"toml_main": 1.0, "recorded_run": 3.0}


def test_a_drift_record_refuses_and_an_empty_one_does_not():
    assert problems_of(good_sc())[0] == []
    drift = {"forward_vel_weight": {"toml_main": 2.0, "recorded_run": 3.0}}
    problems, _ = pj.binding_problems(
        good_sc(), seed=42, want_parent=pj.norm_sha(PARENTS[42]), expect_budget=2_000_000, ramp=2_000_000, warmup=0,
        ramp_start=2.0, forward_weight=2.0, hp_here=HP_P4, task_here=TASK_P4, committed_table=COMMAND_TABLE,
        lineage=LINEAGE, report_off_schedule=False, kwargs_drift=drift)
    assert len(problems) == 1 and "differ from the env this tree builds" in problems[0]
    assert '{"forward_vel_weight": {"recorded_run": 3.0, "toml_main": 2.0}}' in problems[0]
    for empty in ({}, None):
        assert pj.drift_problem(empty) is None
        got, _ = pj.binding_problems(
            good_sc(), seed=42, want_parent=pj.norm_sha(PARENTS[42]), expect_budget=2_000_000, ramp=2_000_000,
            warmup=0, ramp_start=2.0, forward_weight=2.0, hp_here=HP_P4, task_here=TASK_P4,
            committed_table=COMMAND_TABLE, lineage=LINEAGE, report_off_schedule=False, kwargs_drift=empty)
        assert got == []


def test_cadence_problems_lists_every_key():
    assert pj.cadence_problems({}) == ["curriculum.eval_freq is None, not 65536",
                                       "curriculum.eval_episodes is None, not 10",
                                       "curriculum.checkpoint_freq is None, not 250000",
                                       "curriculum.max_checkpoints is None, not 0"]
    assert pj.cadence_problems({"curriculum": {"eval_freq": 65536, "eval_episodes": 10, "checkpoint_freq": 250000,
                                               "max_checkpoints": False}}) == [
        "curriculum.max_checkpoints is False, not 0"]


# the JSON comparison and the command table

@pytest.mark.parametrize("a, b, same", [
    (1.5, 1.5, True), (4, 4.0, False), (True, 1, False), (True, True, True), (0.0, 0, False), ("1.5", 1.5, False),
    (None, None, True), (None, 0.0, False), ([0.0, 0.0], (0.0, 0.0), True), ([0, 0], [0.0, 0.0], False),
    ({"a": 1.0}, {"a": 1.0}, True), ({"a": 1.0}, {"a": 1.0, "b": 2.0}, False), ({"a": [1.0]}, {"a": [1]}, False),
])
def test_same_json_compares_values_and_json_types(a, b, same):
    assert pj.same_json(a, b) is same and pj.same_json(b, a) is same


def test_table_differences_names_each_key():
    assert pj.table_differences(dict(COMMAND_TABLE), COMMAND_TABLE) == []
    rec = dict(COMMAND_TABLE, yaw_gain=2.0, extra=1.0)
    rec.pop("cruise_speed")
    assert pj.table_differences(rec, COMMAND_TABLE) == ["cruise_speed missing", "extra = 1.0 added",
                                                        "yaw_gain = 2.0 (float), committed 1.5 (float)"]
    assert pj.table_differences(COMMAND_TABLE, None) == [("the tree's committed command_config is not a table "
                                                          "(NoneType)")]
    assert pj.table_differences([1.5], COMMAND_TABLE) == ["the recorded command_config is not a table: [1.5]"]


# the tree's branch and the RUN_ID

BP = pj.BRANCH_PREFIX  # the branches below are written from pilot 3's kept prefix


@pytest.mark.parametrize("current, refs, want", [
    (BP + "r10-p4", [], (BP + "r10-p4", "checked out")),
    (BP + "r10-p3", ["refs/heads/" + BP + "r10-p3"], (BP + "r10-p3", "checked out")),
    ("", ["refs/heads/" + BP + "r10-p4"], (BP + "r10-p4", "detached HEAD at the tip of the local branch")),
    ("", ["refs/heads/zz", "refs/heads/" + BP + "b", "refs/heads/" + BP + "a"],
     (BP + "a", "detached HEAD at the tip of the local branch")),
])
def test_the_tree_is_on_a_walking_pilot_head(current, refs, want):
    assert pj.pilot_branch_from(current, refs) == want


@pytest.mark.parametrize("current, refs, needle", [
    ("main", ["refs/heads/" + BP + "r10-p4"], "tree branch 'main' is not a walking-pilot branch"),
    ("pilot-walk-r10-p4", [], "tree branch 'pilot-walk-r10-p4' is not a walking-pilot branch"),
    ("", [], "is the tip of no local walking-pilot branch"),
    ("", ["refs/remotes/origin/" + BP + "r10-p4"], "is the tip of no local walking-pilot branch"),
    ("", ["refs/heads/main", "refs/tags/" + BP + "x"], "is the tip of no local walking-pilot branch"),
])
def test_a_tree_off_a_walking_pilot_head_is_refused(current, refs, needle):
    branch, why = pj.pilot_branch_from(current, refs)
    assert branch is None and needle in why


@pytest.mark.parametrize("name, seed", [
    ("pilot4_walk_s42", 42), ("pilot4_walk_s42_b", 42), ("pilot4_walk_s52", 52), ("pilot4_walk_s60_b", 60)])
def test_the_run_id_pattern_accepts_the_seed_s_run_and_its_restart(name, seed):
    assert pj.run_id_problem(name, seed) is None


@pytest.mark.parametrize("name, seed, needle", [
    ("pilot4_walk_s52", 42, "is not seed 42's RUN_ID (pilot4_walk_s42[_b])"),
    ("pilot4_walk_s042", 42, "is not seed 42's RUN_ID"), ("pilot4_walk_s4", 42, "is not seed 42's RUN_ID"),
    ("pilot4_walk_s52_b", 42, "is not seed 42's RUN_ID"),
    ("pilot4_walk_s42_c", 42, "is not a pilot-4 RUN_ID"), ("pilot4_walk_s42_b_b", 42, "is not a pilot-4 RUN_ID"),
    ("pilot3_walk_s42", 42, "is not a pilot-4 RUN_ID"), ("pilot4_walk_s42b", 42, "is not a pilot-4 RUN_ID"),
    ("x_pilot4_walk_s42", 42, "is not a pilot-4 RUN_ID"), ("pilot4_walk_s42_B", 42, "is not a pilot-4 RUN_ID"),
])
def test_the_run_id_pattern_refuses_every_other_name(name, seed, needle):
    why = pj.run_id_problem(name, seed)
    assert why is not None and needle in why, why


# the resume branch

def resumed(step: int, segment: int, seed: int = 42, **over) -> dict:
    return good_sc(seed, timesteps=segment, resume_load_path=f"/d/03_locomotion/models/stage2_{step}_steps.zip",
                   resume_checkpoint_sha256="sha256:" + "ef" * 32, **over)


@pytest.mark.parametrize("step", [250_000, 500_000, 1_000_000, 1_250_000, 1_750_000])
def test_a_resume_from_any_pair_is_on_schedule_and_reads_the_budget_as_the_segment_plus_the_step(step):
    problems, info = problems_of(resumed(step, 2_000_000 - step))
    assert problems == [] and info["budget"] == 2_000_000 and info["resume_step"] == step and info["resumed"]
    assert info["off_schedule"] is None


def test_a_smoke_resume_at_32_768_is_on_schedule():
    sc = resumed(32_768, 65_536 - 32_768)
    sc["curriculum"].update(eval_freq=32_768, checkpoint_freq=32_768)
    problems, info = problems_of(sc, budget=65_536)
    assert problems == [] and info["budget"] == 65_536 and info["resume_step"] == 32_768


def test_a_resume_whose_sum_is_not_the_budget_is_refused():
    problems, info = problems_of(resumed(1_000_000, 1_500_000))
    assert info["budget"] == 2_500_000 and problems == [
        "budget 2500000 (not 2000000); resumed from /d/03_locomotion/models/stage2_1000000_steps.zip"]


def test_a_resume_from_a_non_periodic_checkpoint_is_refused():
    sc = good_sc(timesteps=1_000_000, resume_load_path="/d/models/stage2_final.zip")
    problems, info = problems_of(sc)
    assert info["budget"] is None and info["resume_step"] is None
    assert any("not a periodic stage2_<steps>_steps checkpoint" in p for p in problems)
    assert any(p.startswith("budget None (not 2000000)") for p in problems)


@pytest.mark.parametrize("warmup, ramp_start, step, off", [
    (300_000, 0.2, 1_500_000, True), (300_000, 0.2, 2_000_000, False), (0, 0.2, 1_999_999, True),
    (300_000, 2.0, 250_000, True), (300_000, 2.0, 299_999, True), (0, 2.0, 32_768, False), (0, 2, 250_000, False),
])
def test_pilot_3_s_off_schedule_rule_holds_only_when_the_shaping_acts(warmup, ramp_start, step, off):
    problems, info = problems_of(resumed(step, 2_000_000 - step), warmup=warmup, ramp_start=ramp_start)
    assert bool(info["off_schedule"]) is off
    if off:
        assert len(problems) == 1 and problems[0].startswith("OFF-SCHEDULE: resumed at step")
        problems, info = problems_of(resumed(step, 2_000_000 - step), warmup=warmup, ramp_start=ramp_start,
                                     report=True)
        assert problems == [] and info["off_schedule"].startswith("OFF-SCHEDULE")
    else:
        assert problems == []


@pytest.mark.parametrize("warmup, ramp_start, forward, inert", [
    (0, 2.0, 2.0, True), (0, 2, 2.0, True), (0.0, 2.0, 2, True), (300_000, 2.0, 2.0, False), (0, 0.2, 2.0, False),
    (0, None, 2.0, False), (0, 2.0, None, False), (False, 2.0, 2.0, False), (0, True, 1.0, False),
])
def test_shaping_inert(warmup, ramp_start, forward, inert):
    assert pj.shaping_inert(warmup, ramp_start, forward) is inert


@pytest.mark.parametrize("edit, needle", [
    (lambda sc: sc["run"].pop("parent_task_sha256"), "run.parent_task_sha256 is None"),
    (lambda sc: sc["run"].pop("n_envs"), "run.n_envs is None"),
    (lambda sc: sc["run"].update(warm_start_log_std=-2.0), "run.warm_start_log_std is recorded"),
    (lambda sc: sc["run"].update(critic_phase_dropped=P3_RECORD), "run.critic_phase_dropped is recorded"),
    (lambda sc: sc["run"].update(load_mode="resume_same_stage"), "load_mode 'resume_same_stage'"),
    (lambda sc: sc["run"].pop("parent_checkpoint_sha256"), "is not the chosen parent"),
])
def test_a_resumed_copy_is_held_to_the_same_records(edit, needle):
    sc = resumed(1_250_000, 750_000)
    edit(sc)
    problems, _ = problems_of(sc)
    assert len(problems) == 1 and needle in problems[0], problems


def test_a_resumed_copy_keeps_the_zip_s_lineage():
    """A same-stage resume attaches no new lineage, so the resumed zips keep the edge's record (by code reading);
    a resumed copy whose zip lost it is refused."""
    sc = resumed(1_250_000, 750_000)
    assert problems_of(sc, lineage=LINEAGE)[0] == []
    assert problems_of(sc, lineage=None)[0] == [("the judged zip records no mesozoic_task_lineage (no edge lineage: "
                                                 "not a pilot-4 fine-tune)")]


def test_resume_check_and_cell_path():
    assert pj.resume_check({}, 2_000_000, 0, 2.0, 2.0) == (None, None)
    assert pj.resume_check({"resume_load_path": "stage2_250000_steps"}, 2_000_000, 0, 2.0, 2.0) == (250_000, None)
    step, why = pj.resume_check({"resume_load_path": "stage2_250000_steps"}, 2_000_000, 300_000)
    assert step == 250_000 and why.startswith("OFF-SCHEDULE")  # pilot 3's call, without the inert shaping
    run = Path("/x/pilot4_walk_s42")
    assert pj.cell_path(Path("/o"), 42, 2_000_000, None, run) == Path("/o/s42/2000000")
    assert pj.cell_path(Path("/o"), 42, 2_000_000, "x", run) == Path("/o/off_schedule/pilot4_walk_s42/s42/2000000")


# the final-model fallback

def write_ckpt(path: Path, num_timesteps: int | None = None, *, members=("data", "policy.pth"),
               lineage: dict | None = LINEAGE) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"num_timesteps": num_timesteps}
    if lineage is not None:
        data[pj.LINEAGE_ATTRIBUTE] = lineage
    with zipfile.ZipFile(path, "w") as z:
        for name in members:
            z.writestr(name, json.dumps(data) if name == "data" else "synthetic")
    return path


def write_vec(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps({"synthetic": True}))
    return path


def fallback(models: Path, *, steps=2_000_000, budget=2_000_000, resumed=False, intact=None, rollout=16_384):
    def pair_problem(z, v):  # the tree's function needs the tree; this one reports what the test says
        return intact
    return pj.final_fallback_problems(models, steps=steps, expect_budget=budget, rollout=rollout, resumed=resumed,
                                      pair_problem=pair_problem)


def test_fallback_accepts_a_final_pair_at_the_uninterrupted_end_when_the_deciding_pair_is_missing(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 2_015_232)
    problems, rec = fallback(tmp_path)
    assert problems == [] and rec["uninterrupted_end"] == 2_015_232 and rec["final_num_timesteps"] == 2_015_232
    assert rec["deciding_pair_problem"].startswith("missing")


def test_fallback_accepts_when_the_deciding_pair_is_broken(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 2_015_232)
    write_ckpt(tmp_path / "stage2_2000000_steps.zip", 2_000_000)
    problems, rec = fallback(tmp_path, intact="missing matched VecNormalize sidecar")
    assert problems == [] and rec["deciding_pair_problem"] == "missing matched VecNormalize sidecar"


@pytest.mark.parametrize("end", [2_000_000, 2_031_616, 1_998_848, 8_011_776, 2_015_233])
def test_fallback_refuses_any_other_num_timesteps(tmp_path, end):
    write_ckpt(tmp_path / "stage2_final.zip", end)
    problems, _ = fallback(tmp_path)
    assert len(problems) == 1 and "not the uninterrupted end 2,015,232" in problems[0]


def test_fallback_refuses_when_the_deciding_pair_is_intact(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 2_015_232)
    write_ckpt(tmp_path / "stage2_2000000_steps.zip", 2_000_000)
    problems, _ = fallback(tmp_path, intact=None)
    assert len(problems) == 1 and "is intact: judge it" in problems[0]


def test_fallback_refuses_another_step_a_resumed_run_and_an_unreadable_final(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 2_015_232)
    assert "stands in only for the deciding checkpoint" in fallback(tmp_path, steps=1_500_000)[0][0]
    assert "a resumed run's final pair" in fallback(tmp_path, resumed=True)[0][0]
    write_ckpt(tmp_path / "bad" / "stage2_final.zip", None)
    problems, _ = fallback(tmp_path / "bad")
    assert len(problems) == 1 and "has no readable num_timesteps" in problems[0]


def test_fallback_refuses_a_tree_whose_rollout_misses_the_text_s_end(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 2_007_040)
    problems, rec = fallback(tmp_path, rollout=8_192)
    assert rec["uninterrupted_end"] == 2_007_040 and problems == [
        ("the tree's rollout 8,192 ends the 2,000,000-step budget at 2,007,040, not the text's uninterrupted end "
         "2,015,232")]


def test_fallback_for_the_smoke_budget_is_65_536(tmp_path):
    write_ckpt(tmp_path / "stage2_final.zip", 65_536)
    problems, rec = fallback(tmp_path, steps=50_000, budget=50_000)
    assert problems == [] and rec["uninterrupted_end"] == 65_536


def test_run_end_reads_the_final_num_timesteps(tmp_path):
    for s in (250_000, 1_000_000, 2_000_000):
        write_ckpt(tmp_path / "models" / f"stage2_{s}_steps.zip", s)
    write_ckpt(tmp_path / "models" / "stage2_final.zip", 2_015_232)
    end = pj.run_end(tmp_path / "models", 2_000_000, tmp_path)
    assert end["periodic_steps"] == [250_000, 1_000_000, 2_000_000]
    assert end["newest_periodic_steps"] == 2_000_000 and end["final_num_timesteps"] == 2_015_232
    assert end["stopped_early"] is False


def test_run_end_names_the_newest_intact_pair(tmp_path):
    """A collapse-stopped seed (earliest stop 1,900,544) is scored at its newest INTACT pair."""
    models = tmp_path / "models"
    for s in (1_250_000, 1_500_000, 1_750_000):
        write_ckpt(models / f"stage2_{s}_steps.zip", s)
    write_vec(models / "stage2_vecnormalize_1250000_steps.pkl")
    write_vec(models / "stage2_vecnormalize_1500000_steps.pkl")  # 1,750,000 has no sidecar
    write_ckpt(models / "stage2_final.zip", 1_900_544)

    def pair_problem(z, v):
        return "does not unpickle" if "1500000" in Path(v).name else None
    end = pj.run_end(models, 2_000_000, tmp_path, pair_problem)
    assert end["newest_periodic_steps"] == 1_750_000 and end["newest_intact_periodic_steps"] == 1_250_000
    assert end["intact_periodic_steps"] == [1_250_000]
    assert end["broken_periodic"] == {"1500000": "does not unpickle",
                                      "1750000": "missing sidecar stage2_vecnormalize_1750000_steps.pkl"}
    assert end["stopped_early"] is True
    unchecked = pj.run_end(models, 2_000_000, tmp_path)
    assert unchecked["intact_periodic_steps"] is None and unchecked["newest_intact_periodic_steps"] is None
    none = pj.run_end(tmp_path / "empty", 2_000_000, tmp_path, pair_problem)
    assert none["intact_periodic_steps"] == [] and none["newest_intact_periodic_steps"] is None


def test_run_end_falls_back_to_the_stage_summary(tmp_path):
    (tmp_path / "models").mkdir()
    (tmp_path / "stage_summary.txt").write_text("Stage 2\nTimesteps: 2,015,232\n")
    end = pj.run_end(tmp_path / "models", 2_000_000, tmp_path)
    assert end["final_num_timesteps"] == 2_015_232 and end["final_source"] == "stage_summary.txt Timesteps"


@pytest.mark.parametrize("recorded, ok", [
    ("0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8", True), ("0E6A57BB4D6BA30AEF36DD9AA30FF4D175786CC8", True),
    ("0e6a57bb", True), ("0e6a57b", True), ("0e6a57", False), ("0e6a57bbx", False), (None, False), ("", False),
    ("0e6a57bb4d6ba30a" + "0" * 24, False), (PILOT3_HEAD, False), ("0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc80", False),
    ("7700f288a9d761dd0c5af5df811b0e302306f5b8", False), ("0bcb0cf941ba6d9d44cc4020e8dff59f773085a9", False)])
def test_commit_problem_needs_7_to_40_hex_digits_that_begin_head(recorded, ok):
    got = pj.commit_problem(recorded, "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8")
    assert (got is None) == ok, got
    if recorded in (None, ""):
        assert got == "stage_config.json records no git_commit"


def test_other_run_cells_lists_cells_of_another_run_copy(tmp_path):
    out = tmp_path / "OUT"
    for steps, run in ((500_000, "/a/pilot4_walk_s42"), (1_000_000, "/b/pilot4_walk_s42_b"),
                       (1_500_000, "/c/pilot4_walk_s42")):
        (out / "s42" / str(steps)).mkdir(parents=True)
        (out / "s42" / str(steps) / "binding.json").write_text(json.dumps({"run": run}))
    (out / "s42" / "1750000.partial").mkdir()
    (out / "s42" / "1750000.partial" / "binding.json").write_text(json.dumps({"run": "/z/other"}))
    (out / "s42" / "2000000").mkdir()
    (out / "s42" / "2000000" / "binding.json").write_text("not json")
    got = pj.other_run_cells(out, 42, "pilot4_walk_s42_b")
    assert got == ["1500000 (pilot4_walk_s42)", "2000000 (unreadable binding.json: JSONDecodeError)",
                   "500000 (pilot4_walk_s42)"]
    assert pj.other_run_cells(out, 52, "pilot4_walk_s52") == []


def test_isolated_path_puts_the_tree_first_and_drops_the_tools_folder_and_the_empty_entry():
    import iso
    got = iso.isolated_path("/t", ["/x", "/t"], ["", "/tools", "/lib", "/x"], "/tools")
    assert got == ["/t", "/x", "/lib"]


# ---------------------------------------------------------------- integration: the judge on a clean clone

needs_tree = pytest.mark.skipif(not (TREE / "environments" / "__init__.py").is_file(), reason=f"no tree at {TREE}")

FACTS_SCRIPT = '''
import json
import subprocess
from pathlib import Path

import environments
from environments.shared.config import hyperparameters_sha256, load_stage_config
from environments.shared.plant_contract import current_plant_identity
from environments.shared.species_names import require_model_id
from environments.shared.task_fingerprint import stage_task_fingerprint

mid = require_model_id("tyrannosaurus_rex", "anatomical")
config = load_stage_config(mid, 2)
print(json.dumps({
    "environments_file": environments.__file__,
    "hp": hyperparameters_sha256(config, "ppo"),
    "task": stage_task_fingerprint(mid, 2, stage_config=config)["task_sha256"],
    "plant": current_plant_identity(mid).to_dict(),
    "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path.cwd(), text=True).strip(),
    "n_steps": config["ppo_kwargs"]["n_steps"],
    "curriculum": config["curriculum_kwargs"],
    "env": config["env_kwargs"],
}, default=str))
'''


def iso_run(script: Path, *args: str, cwd: Path = TREE, extra_env: dict | None = None,
            tree: Path = TREE) -> subprocess.CompletedProcess:
    env = dict(os.environ, ISO_WT=str(tree), MUJOCO_GL="disable", PYTHONDONTWRITEBYTECODE="1")
    env.pop("PYTHONPATH", None)
    env.update(extra_env or {})
    return subprocess.run([sys.executable, "-I", str(TOOLS / "iso.py"), str(script), *args], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600, check=False)


@pytest.fixture(scope="module")
def facts(tmp_path_factory):
    if not (TREE / "environments" / "__init__.py").is_file():
        pytest.skip(f"no tree at {TREE}")
    script = tmp_path_factory.mktemp("facts") / "tree_facts.py"
    script.write_text(FACTS_SCRIPT)
    r = iso_run(script)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def make_run(root: Path, facts: dict, *, seed: int = 42, name: str | None = None, run_over: dict | None = None,
             periodic=(500_000, 2_000_000), final: int | None = 2_015_232, edit=None,
             lineage: dict | None = LINEAGE) -> Path:
    run = root / (name or f"pilot4_walk_s{seed}")
    stage = run / "03_locomotion"
    models = stage / "models"
    models.mkdir(parents=True, exist_ok=True)
    for s in periodic:
        write_ckpt(models / f"stage2_{s}_steps.zip", s, lineage=lineage)
        write_vec(models / f"stage2_vecnormalize_{s}_steps.pkl")
    if final is not None:
        write_ckpt(models / "stage2_final.zip", final, lineage=lineage)
        write_vec(models / "stage2_final_vecnorm.pkl")
    sc = template_sc(seed)
    sc["git_commit"] = facts["head"]
    for k, v in (run_over or {}).items():
        if v is None:
            sc["run"].pop(k, None)
        else:
            sc["run"][k] = v
    if edit:
        edit(sc)
    (stage / "stage_config.json").write_text(json.dumps(sc, indent=1))
    (stage / "plant_identity.json").write_text(json.dumps(facts["plant"], indent=1))
    return run


def judge(run: Path, out: Path, *args: str, seed=42, steps=2_000_000, parent=None,
          tree: Path = TREE) -> subprocess.CompletedProcess:
    parent = PARENTS[seed] if parent is None else parent
    return iso_run(TOOLS / "pilot4_judge.py", str(run), "--seed", str(seed), "--steps", str(steps),
                   "--parent-sha256", parent, "--out", str(out), *args, cwd=tree, tree=tree)


def assert_refused(r: subprocess.CompletedProcess, out: Path, needle: str) -> None:
    assert r.returncode == 2, (r.stdout, r.stderr)
    assert "refusing (nothing written)" in r.stdout and needle in r.stdout, r.stdout
    assert not out.exists() or not any(out.rglob("*")), list(out.rglob("*"))


@needs_tree
def test_the_tree_is_a_clean_clone_on_a_walking_pilot_head_and_derives_the_text_s_digests(facts):
    current = pj.git(TREE, "branch", "--show-current")
    refs = pj.git(TREE, "for-each-ref", "--points-at", "HEAD", "--format=%(refname)", "refs/heads/").splitlines()
    assert pj.pilot_branch_from(current, refs)[0] is not None
    assert pj.git(TREE, "status", "--porcelain", "--untracked-files=no") == ""
    assert facts["environments_file"].startswith(str(TREE) + "/")
    assert facts["hp"] == HP_P4 and facts["task"] == TASK_P4
    assert pj.same_json(facts["env"]["command_config"], COMMAND_TABLE)  # the committed table is the text's
    assert facts["env"]["command_mode"] == "heading" and facts["env"]["prey_lateral_range"] == [0.0, 0.0]
    assert facts["env"]["forward_vel_weight"] == 2.0 and facts["n_steps"] == 4096
    cur = facts["curriculum"]
    assert cur["warmup_timesteps"] == 0 and cur["ramp_start_value"] == 2.0 and cur["ramp_timesteps"] == 2_000_000
    assert cur["initialize_from_own_checkpoint"] is True and "warm_start_log_std" not in cur
    assert {k: cur[k] for k in pj.CADENCE} == pj.CADENCE and cur["timesteps"] == 2_000_000
    assert pj.shaping_inert(cur["warmup_timesteps"], cur["ramp_start_value"], facts["env"]["forward_vel_weight"])
    for k in ("physics_sha256", "policy_interface_sha256"):
        assert facts["plant"][k] == SC_TEMPLATE["plant_identity"][k]


@needs_tree
@pytest.mark.parametrize("seed", [42, 52, 60])
def test_check_only_accepts_a_run_trained_as_fixed_and_writes_nothing(tmp_path, facts, seed):
    out = tmp_path / "OUT"
    r = judge(make_run(tmp_path, facts, seed=seed), out, "--check-only", "--expect-head", facts["head"], seed=seed)
    assert r.returncode == 0, (r.stdout, r.stderr)
    assert "all checks passed (--check-only" in r.stdout and str(out / f"s{seed}" / "2000000") in r.stdout
    assert not out.exists()


def _set(path: str, value):
    def edit(sc):
        block, key = path.split(".")
        if value is None:
            sc[block].pop(key)
        else:
            sc[block][key] = value
    return edit


def _pop_price(sc):
    sc["reward_weights"].pop("walk_heading_price_weight")
    sc["reward_weights"].pop("walk_heading_price_scale_rad")


@needs_tree
@pytest.mark.parametrize("label, make, judge_kwargs, needle", [
    # the plan's list
    ("pilot 3's stance parent", {}, {"parent": STANCE_PARENT}, "is not s42's own 8M parent"),
    ("another seed's 8M parent", {"run_over": {"parent_checkpoint_sha256": PARENTS[52]}}, {"parent": PARENTS[52]},
     "(it is s52's)"),
    ("a -2.0 record", {"run_over": {"warm_start_log_std": -2.0}}, {}, "run.warm_start_log_std is recorded (-2.0)"),
    ("a -2.0 curriculum record", {"edit": _set("curriculum.warm_start_log_std", -2.0)}, {},
     "curriculum.warm_start_log_std is recorded (-2.0)"),
    ("a 0.2 ramp", {"edit": _set("curriculum.ramp_start_value", 0.2)}, {}, "curriculum.ramp_start_value is 0.2"),
    ("prey [-2, 2]", {"edit": _set("reward_weights.prey_lateral_range", [-2.0, 2.0])}, {},
     "prey_lateral_range is [-2.0, 2.0], not [0.0, 0.0]"),
    ("the price off", {"edit": _pop_price}, {}, "reward_weights.walk_heading_price_weight is absent (not 0.2)"),
    ("a rewritten command table", {"edit": lambda sc: _cmd(sc).update(switch_interval_s=4.0)}, {},
     "switch_interval_s = 4.0 added"),
    ("a command table with an int", {"edit": lambda sc: _cmd(sc).update(yaw_rate_max=0.3, cruise_speed=2)}, {},
     "cruise_speed = 2 (int), committed 1.5 (float)"),
    # pilot 3's, adapted
    ("a critic-phase drop record", {"run_over": {"critic_phase_dropped": P3_RECORD}}, {},
     "run.critic_phase_dropped is recorded"),
    ("n_envs 2", {"run_over": {"n_envs": 2}}, {}, "run.n_envs is 2, not 4"),
    ("another head", {"edit": lambda sc: sc.update(git_commit=PILOT3_HEAD)}, {}, "!= tree HEAD"),
    ("the setup check's commit", {"edit": lambda sc: sc.update(git_commit=SC_TEMPLATE["git_commit"])}, {},
     "!= tree HEAD"),
    ("a critic_phase key", {"edit": lambda sc: sc["hyperparameters"].update(critic_phase=False)}, {},
     "hold a critic_phase key"),
    ("seed 52 judged as 42", {"run_over": {"seed": 52}}, {}, "seed 52 (not --seed 42)"),
    ("no git_commit", {"edit": lambda sc: sc.pop("git_commit")}, {}, "stage_config.json records no git_commit"),
    ("a git_commit that only extends HEAD's prefix",
     {"edit": lambda sc: sc.update(git_commit=sc["git_commit"][:12] + "0" * 28)}, {}, "!= tree HEAD"),
    ("pilot 3's budget", {"run_over": {"timesteps": 8_000_000}}, {}, "budget 8000000 (not 2000000)"),
    # pilot 4's other records
    ("a warm-up", {"edit": _set("curriculum.warmup_timesteps", 300_000)}, {}, "curriculum.warmup_timesteps is 300000"),
    ("no own-checkpoint key", {"edit": _set("curriculum.initialize_from_own_checkpoint", None)}, {},
     "curriculum.initialize_from_own_checkpoint is None, not true"),
    ("the parent's task not pilot 3's", {"run_over": {"parent_task_sha256": STANCE_TASK}}, {},
     "run.parent_task_sha256 is 'sha256:1e5964b6"),
    ("another task digest", {"edit": _set("task_fingerprint.task_sha256", PILOT3_TASK)}, {},
     f"task_sha256 {PILOT3_TASK} != this tree's {TASK_P4}"),
    ("pilot 3's recipe digest",
     {"run_over": {"hyperparameters_sha256": "sha256:85881fc441355fc58f353a089d665c857b377aa1e7a9b6a5a5f89ef64271af4c"}},
     {}, f"!= this tree's {HP_P4}"),
    ("pilot 3's cadence", {"edit": _set("curriculum.eval_freq", 131072)}, {}, "curriculum.eval_freq is 131072"),
    ("command mode none", {"edit": _set("reward_weights.command_mode", "none")}, {},
     "reward_weights.command_mode is 'none'"),
    ("R6's touchdown off", {"edit": _set("reward_weights.walk_touchdown_weight", 0.0)}, {},
     "reward_weights.walk_touchdown_weight is 0.0, not 0.1"),
    ("a pilot-3 RUN_ID", {"name": "pilot3_walk_s42"}, {}, "is not a pilot-4 RUN_ID"),
    ("another seed's RUN_ID", {"name": "pilot4_walk_s52"}, {}, "is not seed 42's RUN_ID"),
    ("a second restart's RUN_ID", {"name": "pilot4_walk_s42_c"}, {}, "is not a pilot-4 RUN_ID"),
    ("no lineage in the zip", {"lineage": None}, {}, "records no mesozoic_task_lineage"),
    ("pilot 3's lineage in the zip", {"lineage": PILOT3_8M_LINEAGE}, {}, "lineage parent_stage is 1, not 2"),
    ("a lineage without zeroed columns",
     {"lineage": {k: v for k, v in LINEAGE.items() if k != "zeroed_command_parameters"}}, {},
     "records no zeroed_command_parameters"),
    # fix stage: an env weight the binding does not name, as the probe's drift record sees it
    ("another forward weight", {"edit": _set("reward_weights.forward_vel_weight", 3.0)}, {},
     'differ from the env this tree builds (constructor defaults and the committed [env]), so the probe would judge '
     'in another env than training\'s: {"forward_vel_weight": {"recorded_run": 3.0, "toml_main": 2.0}}'),
    ("an env kwarg the tree does not have", {"edit": _set("reward_weights.not_an_env_kwarg", 1.0)}, {},
     '{"not_an_env_kwarg": {"recorded_run": 1.0, "toml_main": "<absent>"}}'),
])
def test_refusals_on_the_tree_write_nothing(tmp_path, facts, label, make, judge_kwargs, needle):
    out = tmp_path / "OUT"
    r = judge(make_run(tmp_path, facts, **make), out, **judge_kwargs)
    assert_refused(r, out, needle)


@needs_tree
def test_refusals_for_the_tree_state(tmp_path, facts):
    out = tmp_path / "OUT"
    run = make_run(tmp_path, facts)
    assert_refused(judge(run, out, "--expect-head", "31d02e4"), out, "is not 31d02e4")
    r = judge(run, out, "--check-only", "--expect-head", facts["head"][:12])
    assert r.returncode == 0, r.stdout
    r = iso_run(TOOLS / "pilot4_judge.py", str(run), "--seed", "42", "--steps", "2000000", "--parent-sha256", "abc",
                "--out", str(out))
    assert_refused(r, out, "--parent-sha256 'abc' is not a sha256")


@needs_tree
def test_a_restart_run_id_is_accepted(tmp_path, facts):
    out = tmp_path / "OUT"
    r = judge(make_run(tmp_path, facts, name="pilot4_walk_s42_b"), out, "--check-only")
    assert r.returncode == 0, r.stdout


@needs_tree
def test_a_missing_or_broken_pair_is_refused(tmp_path, facts):
    out = tmp_path / "OUT"
    run = make_run(tmp_path, facts)
    (run / "03_locomotion/models/stage2_vecnormalize_2000000_steps.pkl").unlink()
    assert_refused(judge(run, out), out, "checkpoint missing or incomplete run copy")
    write_vec(run / "03_locomotion/models/stage2_vecnormalize_2000000_steps.pkl")
    write_ckpt(run / "03_locomotion/models/stage2_2000000_steps.zip", 2_000_000, members=("data",))
    assert_refused(judge(run, out), out, "not an intact pair: bad/truncated checkpoint zip")
    write_ckpt(run / "03_locomotion/models/stage2_2000000_steps.zip", 2_000_000)
    (run / "03_locomotion/models/stage2_vecnormalize_2000000_steps.pkl").write_bytes(b"not a pickle")
    assert_refused(judge(run, out), out, "does not unpickle")


@needs_tree
def test_the_resume_branch_on_the_tree(tmp_path, facts):
    out = tmp_path / "OUT"
    on = make_run(tmp_path / "a", facts, run_over={
        "timesteps": 1_750_000, "resume_load_path": "/d/03_locomotion/models/stage2_250000_steps.zip",
        "resume_checkpoint_sha256": "sha256:" + "ef" * 32})
    r = judge(on, out, "--check-only")
    assert r.returncode == 0 and "OFF-SCHEDULE" not in r.stdout, r.stdout
    early = make_run(tmp_path / "b", facts, run_over={
        "timesteps": 2_000_000 - 32_768, "resume_load_path": "/d/stage2_32768_steps.zip"})
    r = judge(early, out, "--check-only")
    assert r.returncode == 0 and "OFF-SCHEDULE" not in r.stdout, r.stdout
    final = make_run(tmp_path / "c", facts, run_over={
        "timesteps": 1_000_000, "resume_load_path": "/d/stage2_final.zip"})
    assert_refused(judge(final, out), out, "which is not a periodic stage2_<steps>_steps checkpoint")
    bad_sum = make_run(tmp_path / "d", facts, run_over={
        "timesteps": 1_000_000, "resume_load_path": "/d/stage2_500000_steps.zip"})
    assert_refused(judge(bad_sum, out), out, "budget 1500000 (not 2000000)")
    lost = make_run(tmp_path / "e", facts, lineage=None, run_over={
        "timesteps": 750_000, "resume_load_path": "/d/stage2_1250000_steps.zip"})
    assert_refused(judge(lost, out), out, "records no mesozoic_task_lineage")


@needs_tree
def test_final_fallback_on_the_tree(tmp_path, facts):
    out = tmp_path / "OUT"
    run = make_run(tmp_path / "a", facts, periodic=(500_000, 1_750_000))
    r = judge(run, out, "--final-fallback", "--check-only")
    assert r.returncode == 0 and "[final-model fallback]" in r.stdout, r.stdout
    assert_refused(judge(run, out, "--check-only"), out, "checkpoint missing or incomplete run copy")
    other = make_run(tmp_path / "b", facts, periodic=(500_000,), final=2_031_616)
    assert_refused(judge(other, out, "--final-fallback"), out, "not the uninterrupted end 2,015,232")
    intact = make_run(tmp_path / "c", facts)
    assert_refused(judge(intact, out, "--final-fallback"), out, "is intact: judge it")
    no_lineage = make_run(tmp_path / "e", facts, periodic=(500_000,), lineage=None)
    assert_refused(judge(no_lineage, out, "--final-fallback"), out, "records no mesozoic_task_lineage")

    def smoke_cadence(sc):
        sc["curriculum"].update(eval_freq=32_768, checkpoint_freq=32_768)
    smoke = make_run(tmp_path / "d", facts, periodic=(32_768, 65_536), final=65_536,
                     run_over={"timesteps": 50_000}, edit=smoke_cadence)
    r = judge(smoke, out, "--final-fallback", "--check-only", "--expect-budget", "50000", steps=50_000)
    assert r.returncode == 0, r.stdout
    r = judge(smoke, out, "--check-only", "--expect-budget", "50000", steps=65_536)
    assert r.returncode == 0, r.stdout
    smoke_at_budget = make_run(tmp_path / "f", facts, edit=smoke_cadence)
    assert_refused(judge(smoke_at_budget, out, "--check-only"), out, "curriculum.eval_freq is 32768, not 65536")


@needs_tree
def test_a_restart_is_not_judged_into_a_seed_folder_holding_the_retired_run_s_cells(tmp_path, facts):
    """The retired run's cells must be moved to OUT/retired/<run>/ first; then the _b run is judged."""
    out = tmp_path / "OUT"
    old = out / "s42" / "500000"
    old.mkdir(parents=True)
    (old / "binding.json").write_text(json.dumps({"run": str(tmp_path / "x" / "pilot4_walk_s42")}))
    b_run = make_run(tmp_path / "b", facts, name="pilot4_walk_s42_b")
    r = judge(b_run, out, "--check-only")
    assert r.returncode == 2 and "holds cells judged from another run than pilot4_walk_s42_b" in r.stdout, r.stdout
    assert "500000 (pilot4_walk_s42)" in r.stdout and "OUT/retired/" in r.stdout
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*")) == ["s42", "s42/500000",
                                                                             "s42/500000/binding.json"]
    retired = out / "retired" / "pilot4_walk_s42" / "s42"
    retired.mkdir(parents=True)
    old.rename(retired / "500000")
    r = judge(b_run, out, "--check-only")
    assert r.returncode == 0 and str(out / "s42" / "2000000") in r.stdout, r.stdout
    same = out / "s42" / "1000000"
    same.mkdir(parents=True)
    (same / "binding.json").write_text(json.dumps({"run": str(b_run)}))
    assert judge(b_run, out, "--check-only").returncode == 0
    done = out / "s42" / "2000000"
    done.mkdir(parents=True)
    (done / "binding.json").write_text(json.dumps({"run": str(b_run)}))
    r = judge(b_run, out, "--check-only")
    assert r.returncode == 2 and "exists (one judgement per checkpoint, never overwritten)" in r.stdout, r.stdout


def _real_pair(seed: int) -> tuple[Path, Path]:
    models = PILOT3_RUNS / f"pilot3_walk_s{seed}" / "03_locomotion" / "models"
    z, v = models / "stage2_8000000_steps.zip", models / "stage2_vecnormalize_8000000_steps.pkl"
    if not (z.is_file() and v.is_file()):
        pytest.skip(f"no verified run copy at {models}")
    return z, v


def _stand_in(run: Path, seed: int, *, lineage: dict | None) -> None:
    """The seed's real 8M pair renamed as the 2,000,000 pair: byte for byte (lineage None), or with ONLY the zip's
    JSON data member's lineage replaced (every other member byte for byte)."""
    z, v = _real_pair(seed)
    models = run / "03_locomotion" / "models"
    target = models / "stage2_2000000_steps.zip"
    (models / "stage2_vecnormalize_2000000_steps.pkl").write_bytes(v.read_bytes())
    if lineage is None:
        target.write_bytes(z.read_bytes())
        return
    with zipfile.ZipFile(z) as zin, zipfile.ZipFile(target, "w") as zout:
        for info in zin.infolist():
            body = zin.read(info.filename)
            if info.filename == "data":
                data = json.loads(body)
                data[pj.LINEAGE_ATTRIBUTE] = lineage
                body = json.dumps(data).encode()
            zout.writestr(info, body)


@needs_tree
@pytest.mark.parametrize("seed", [42, 52, 60])
def test_the_real_8m_pair_as_a_stand_in(tmp_path, facts, seed):
    """Pilot 3's own 8M zip records pilot 3's edge (parent stage 1, no zeroed columns): the judge refuses it as a
    pilot-4 checkpoint; with only its lineage replaced by the edge's record it passes every check (the real archive
    passes checkpoint_pair_problem)."""
    _real_pair(seed)
    out = tmp_path / "OUT"
    raw = make_run(tmp_path / "raw", facts, seed=seed, periodic=(), final=None)
    _stand_in(raw, seed, lineage=None)
    r = judge(raw, out, "--check-only", seed=seed)
    assert_refused(r, out, "the judged zip's lineage parent_stage is 1, not 2; the judged zip's lineage records no "
                           "zeroed_command_parameters")
    doctored = make_run(tmp_path / "doc", facts, seed=seed, periodic=(), final=None)
    _stand_in(doctored, seed, lineage=LINEAGE)
    r = judge(doctored, out, "--check-only", seed=seed)
    assert r.returncode == 0 and "all checks passed" in r.stdout, (r.stdout, r.stderr)


@needs_tree
def test_the_branch_rule_on_a_clone(tmp_path, facts):
    """A clean clone checked out on the pilot branch, or detached at its tip, is judged; on another branch, or
    detached elsewhere, it is refused before anything is imported."""
    clone = tmp_path / "clone"
    genv = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*args, cwd=clone):
        subprocess.run(["git", *args], cwd=cwd, env=genv, check=True, capture_output=True, text=True)
    git("clone", "-q", "--shared", "--no-checkout", str(TREE), str(clone), cwd=tmp_path)
    branch = pj.pilot_branch_from(pj.git(TREE, "branch", "--show-current"), pj.git(
        TREE, "for-each-ref", "--points-at", "HEAD", "--format=%(refname)", "refs/heads/").splitlines())[0]
    git("checkout", "-q", "-B", branch, facts["head"])
    out = tmp_path / "OUT"
    run = make_run(tmp_path / "runs", facts)
    r = judge(run, out, "--check-only", "--expect-head", facts["head"], tree=clone)
    assert r.returncode == 0 and "all checks passed" in r.stdout, (r.stdout, r.stderr)
    git("checkout", "-q", "--detach", branch)
    r = judge(run, out, "--check-only", tree=clone)
    assert r.returncode == 0, (r.stdout, r.stderr)
    git("checkout", "-q", "-b", "other-branch")
    assert_refused(judge(run, out, "--check-only", tree=clone), out, "tree branch 'other-branch' is not a "
                                                                     "walking-pilot branch")
    git("checkout", "-q", "--detach", facts["head"] + "~1")
    assert_refused(judge(run, out, "--check-only", tree=clone), out,
                   "the tree's HEAD is detached and is the tip of no local walking-pilot branch")


@needs_tree
def test_the_default_expect_head_is_the_text_s_head(tmp_path, facts):
    """Fix stage: without --expect-head the tree must be at PILOT_HEAD (0e6a57b): a clean clone on the pilot branch
    at HEAD~1 is refused for its head (the branch rule passes there); the tree under test is at PILOT_HEAD."""
    assert facts["head"] == pj.PILOT_HEAD
    clone = tmp_path / "clone"
    genv = dict(os.environ, GIT_OPTIONAL_LOCKS="0")

    def git(*args, cwd=clone):
        return subprocess.run(["git", *args], cwd=cwd, env=genv, check=True, capture_output=True, text=True)
    git("clone", "-q", "--shared", "--no-checkout", str(TREE), str(clone), cwd=tmp_path)
    branch = pj.BRANCH_PREFIX + "r10-p4"
    git("checkout", "-q", "-B", branch, facts["head"] + "~1")
    parent = git("rev-parse", "HEAD").stdout.strip()
    out = tmp_path / "OUT"
    run = make_run(tmp_path / "runs", facts)
    assert_refused(judge(run, out, "--check-only", tree=clone), out, f"tree HEAD {parent} is not {pj.PILOT_HEAD}")
    r = judge(run, out, "--check-only")  # the shared tree, no --expect-head: the default passes there
    assert r.returncode == 0 and "all checks passed" in r.stdout, (r.stdout, r.stderr)


@needs_tree
def test_iso_refuses_without_a_tree_and_rewrites_pythonpath(tmp_path):
    probe = tmp_path / "show.py"
    probe.write_text("import os, sys, environments\nprint(os.environ['PYTHONPATH'].split(os.pathsep)[0])\n"
                     "print(sys.path[0])\nprint(environments.__file__)\nprint(sys.dont_write_bytecode)\n")
    r = iso_run(probe, extra_env={"PYTHONPATH": str(tmp_path)})
    lines = r.stdout.split()
    assert r.returncode == 0 and lines[0] == str(TREE) and lines[1] == str(TREE)
    assert lines[2].startswith(str(TREE) + "/") and lines[3] == "True"
    r = iso_run(probe, extra_env={"ISO_WT": str(tmp_path)})
    assert r.returncode == 2 and "is not a tree" in r.stderr
    exits = tmp_path / "exits.py"
    exits.write_text("import sys\nsys.exit(7)\n")
    assert iso_run(exits).returncode == 7
