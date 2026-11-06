"""v3's first-hour list (Run handling, L517-525) and the parent table (L103-113) checked on a stage_config.json.

Usage: python -I first_hour_check.py STAGE_CONFIG.json SEED EXPECT_EVAL EXPECT_CKPT EXPECT_TIMESTEPS
Values are typed from v3's text here (not read from any tool). Prints each check and its result.
"""
import json
import sys

sc = json.load(open(sys.argv[1]))
seed, ev, ck, ts = (int(x) for x in sys.argv[2:6])
PARENT = {42: "41fc0b0cd7167ab25a2567779b89f431b829f6a2bae4c72fa3c89192732e87c4",
          52: "2a87a6874f252de9462960dbfe5bfd6dc0986241679800111d68ef8c68f34c00",
          60: "2c066770802541e47fb453b8dbd04b030bc8053d4224af27a52eb0281e4645a2"}
TABLE = {"cruise_speed": 1.5, "stop_probability": 0.0, "turn_increment_max": 0.0, "straight_probability": 1.0,
         "speed_scale": 1.5, "lateral_speed_scale": 1.0, "yaw_rate_scale": 0.6, "yaw_gain": 1.5, "yaw_rate_max": 0.3,
         "turn_slowdown": True, "minimum_turn_speed_fraction": 0.25}
run, cur, rw = sc.get("run", {}), sc.get("curriculum", {}), sc.get("reward_weights", {})


def strip(x):
    return x.split(":", 1)[1] if isinstance(x, str) and x.startswith("sha256:") else x


checks = [
    ("run.load_mode initialize_next_stage", run.get("load_mode") == "initialize_next_stage"),
    ("run.parent_checkpoint_sha256 = the seed's 8M zip", strip(run.get("parent_checkpoint_sha256")) == PARENT[seed]),
    ("run.parent_task_sha256 782d95b6...", str(strip(run.get("parent_task_sha256"))).startswith("782d95b6")),
    ("no run.warm_start_log_std", "warm_start_log_std" not in run),
    ("no curriculum.warm_start_log_std", "warm_start_log_std" not in cur),
    ("no run.critic_phase_dropped", "critic_phase_dropped" not in run),
    ("no hyperparameters.critic_phase", "critic_phase" not in sc.get("hyperparameters", {})),
    ("run.n_envs 4", run.get("n_envs") == 4),
    (f"curriculum.eval_freq {ev}", cur.get("eval_freq") == ev),
    ("curriculum.eval_episodes 10", cur.get("eval_episodes") == 10),
    (f"curriculum.checkpoint_freq {ck}", cur.get("checkpoint_freq") == ck),
    ("curriculum.max_checkpoints 0", cur.get("max_checkpoints") == 0),
    ("curriculum.warmup_timesteps 0", cur.get("warmup_timesteps") == 0),
    ("curriculum.ramp_start_value 2.0", cur.get("ramp_start_value") == 2.0),
    ("curriculum.initialize_from_own_checkpoint true", cur.get("initialize_from_own_checkpoint") is True),
    ("git_commit = 0e6a57bb...", sc.get("git_commit") == "0e6a57bb4d6ba30aef36dd9aa30ff4d175786cc8"),
    ("run.hyperparameters_sha256 fbc94bef...", strip(run.get("hyperparameters_sha256")) ==
     "fbc94bef69a91d5a65f1df99adb9c35e7897a259affaa3e7b7ba4477418eaa62"),
    ("task_fingerprint.task_sha256 914a1cae...", strip(sc.get("task_fingerprint", {}).get("task_sha256")) ==
     "914a1caee2c086a09ef66e9e6d4ad234dd085b69c4da687cc29af82b110fcd7a"),
    ("command_mode heading", rw.get("command_mode") == "heading"),
    ("command_config = the committed table, key for key, floats", rw.get("command_config") == TABLE and all(
        type(rw["command_config"][k]) is type(v) for k, v in TABLE.items())),
    ("prey_lateral_range [0, 0]", rw.get("prey_lateral_range") in ([0.0, 0.0], (0.0, 0.0))),
    ("walk_heading_price_weight 0.2", rw.get("walk_heading_price_weight") == 0.2),
    ("walk_heading_price_scale_rad 0.5", rw.get("walk_heading_price_scale_rad") == 0.5),
    ("R6 walk_support_gate_bw 0.04", rw.get("walk_support_gate_bw") == 0.04),
    ("R6 airborne_substep_weight 1.0", rw.get("airborne_substep_weight") == 1.0),
    ("R6 forward_vel_max 1.5", rw.get("forward_vel_max") == 1.5),
    ("R6 walk_touchdown_weight 0.1", rw.get("walk_touchdown_weight") == 0.1),
    ("forward_vel_weight 2.0", rw.get("forward_vel_weight") == 2.0),
    (f"run.seed {seed}", run.get("seed") == seed),
    (f"run.timesteps {ts}", run.get("timesteps") == ts),
    ("stage 2", sc.get("stage") == 2),
    ("species tyrannosaurus_rex / anatomical", sc.get("species") == "tyrannosaurus_rex" and sc.get("embodiment") == "anatomical"),
]
bad = [c for c, ok in checks if not ok]
for c, ok in checks:
    print(("ok   " if ok else "FAIL ") + c)
print(f"{len(checks) - len(bad)} of {len(checks)} ok; failing: {bad}")
