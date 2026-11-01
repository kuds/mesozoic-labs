"""A5 (switch ON): rollbuf.py of S2/r10round/runbranch/build/tools (read only) executed with exactly three changes so
the training construction path is train_base.train's with [ppo] critic_phase = true (train_base.py lines 1385-1476):
the stage config's ppo_kwargs carry critic_phase = true; the training env is created with critic_phase_gamma = the
stage's gamma (line 1401); critic_phase is popped from the PPO kwargs and the env's spec passed to
_create_or_load_model (lines 1446-1476).  No train(), no optimizer step.  usage as rollbuf.py."""
import sys
SRC = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/runbranch/build/tools/rollbuf.py"
src = open(SRC).read()
EDITS = [
    ("configs = load_all_stages(mid); cfg = configs[1]\n",
     "configs = load_all_stages(mid); cfg = configs[1]\ncfg[\"ppo_kwargs\"][\"critic_phase\"] = True\n"),
    ("train_env = tb.create_vec_env(species, configs, 1, n_envs, SEED, False, algorithm=\"ppo\", gamma=gamma, plant_identity=ident)\n",
     "train_env = tb.create_vec_env(species, configs, 1, n_envs, SEED, False, algorithm=\"ppo\", gamma=gamma, plant_identity=ident, critic_phase_gamma=gamma)\n"),
    ("alg_kwargs.setdefault(\"seed\", SEED)\n",
     "alg_kwargs.setdefault(\"seed\", SEED)\nfrom environments.shared.critic_phase import pop_critic_phase\nassert pop_critic_phase(alg_kwargs) is True\n"),
    ("task_load_mode=\"resume_same_stage\", symmetry=symmetry)\n",
     "task_load_mode=\"resume_same_stage\", symmetry=symmetry, critic_phase=train_env.critic_phase)\n"),
]
for old, new in EDITS:
    assert src.count(old) == 1, old
    src = src.replace(old, new)
sys.argv = [SRC] + sys.argv[1:]
exec(compile(src, SRC + " (critic_phase ON, rb_on.py)", "exec"), {"__name__": "__main__", "__file__": SRC})
