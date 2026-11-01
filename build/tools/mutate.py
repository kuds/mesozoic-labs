"""Mutation checks of the critic-phase tests, in ST/build/mut_tree (a git archive of the branch HEAD).

usage: python -I -B mutate.py   -- applies each mutation alone, runs the two critic-phase test files, restores the file;
a mutation must make at least one test fail ("KILLED"); prints one line per mutation."""
import subprocess
import sys
from pathlib import Path

TREE = Path("/home/user/work/pc_study/build/mut_tree")
PY = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/venv/bin/python"
P = "environments/shared/critic_phase_ppo.py"
MUTATIONS = [
    ("M1 actor reads the phase columns too", P,
     "latent: th.Tensor = self.policy_net(features[..., : self.actor_input_dim].contiguous())",
     "latent: th.Tensor = self.policy_net(features[..., : self.actor_input_dim].contiguous() + 0.0 * features[..., -1:].sum())"),
    ("M2 a finished env's next observation keeps the old counter", P,
     "            else:\n                counters[index] = int(step)",
     "            counters[index] = int(step)"),
    ("M3 the terminal observation gets the reset's phase (k = 0)", P,
     "np.asarray(terminal)[None], np.array([step], dtype=np.int64)",
     "np.asarray(terminal)[None], np.array([0], dtype=np.int64)"),
    ("M4 the phase columns flip sign under the mirror", "environments/shared/mirror_kl_ppo.py",
     "obs_sign = np.concatenate([obs_sign, np.ones(extra)])",
     "obs_sign = np.concatenate([obs_sign, -np.ones(extra)])"),
    ("M5 the probe gives the kept step, not kept step + 1", "environments/shared/policy_probe.py",
     "critic_phase.matrix(np.asarray(bank.steps, dtype=np.int64) + 1)",
     "critic_phase.matrix(np.asarray(bank.steps, dtype=np.int64))"),
    ("M6 the critic's phase columns start non-zero", P,
     "wide.weight[:, self.actor_input_dim :] = 0.0",
     "wide.weight[:, self.actor_input_dim :] = 0.01"),
    ("M7 no bootstrap observation with the phase (stock obs_to_tensor)", P,
     "            if array.ndim in (1, 2) and array.shape[-1] == width:",
     "            if False:"),
    ("M8 the mismatch check lets every load through", P,
     "    loaded = model_critic_phase(model)\n",
     "    return None\n    loaded = model_critic_phase(model)\n"),
    ("M9 the sidecar pickles the subclass", P,
     "            pickle.dump(stock, handle)",
     "            pickle.dump(self, handle)"),
    ("M10 the phase is normalised with running statistics", P,
     "        phase = self.critic_phase.matrix(counters).astype(observations.dtype)",
     "        phase = self.critic_phase.matrix(counters).astype(observations.dtype)\n        phase = (phase - phase.mean(0)) / np.sqrt(phase.var(0) + 1e-8)"),
    ("M11 settled flag off by one (k > S)", "environments/shared/critic_phase.py",
     "1.0 if k >= settle_steps else 0.0",
     "1.0 if k > settle_steps else 0.0"),
    ("M12 the switch-off path builds the phase buffer anyway", "environments/shared/train_base.py",
     "critic_phase_gamma=(0.99 if alg_gamma is None else alg_gamma) if critic_phase_on else None,",
     "critic_phase_gamma=(0.99 if alg_gamma is None else alg_gamma),"),
]
TESTS = ["environments/shared/tests/test_critic_phase.py", "environments/shared/tests/test_critic_phase_ppo.py"]
ONLY = sys.argv[1:]
for name, rel, old, new in MUTATIONS:
    if ONLY and not any(name.startswith(prefix + " ") for prefix in ONLY):
        continue
    path = TREE / rel
    text = path.read_text()
    if text.count(old) != 1:
        print(f"{name}: PATTERN NOT FOUND ({text.count(old)})")
        continue
    path.write_text(text.replace(old, new, 1))
    try:
        result = subprocess.run(
            [PY, "-I", "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", "-x", *TESTS,
             "--basetemp=/home/user/work/pc_study/build/scratch/mut_tmp"],
            cwd=TREE, capture_output=True, text=True, timeout=1200,
        )
        tail = [line for line in result.stdout.splitlines() if line.startswith("FAILED") or "passed" in line or "failed" in line]
        print(f"{name}: {'KILLED' if result.returncode != 0 else 'SURVIVED'} | {' | '.join(tail[-2:])}", flush=True)
    finally:
        path.write_text(text)
