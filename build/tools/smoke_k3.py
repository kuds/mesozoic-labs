"""K3 panel with the critic phase ON: train_base.train (the notebook's call) on the real stance stage, 1 tiny update
(n_steps 64, one env), k3_panel = K3PanelConfig() (own-sigma test only); the panel builds its CPU twin of the policy
(type(policy)(**policy._get_constructor_parameters())) and rolls the final panel after the final save.
usage: python -I -B smoke_k3.py TREE OUTDIR"""
import copy
import json
import sys
from pathlib import Path

tree, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
sys.path.insert(0, str(tree))
from environments.shared import train_base  # noqa: E402
from environments.shared.config import load_all_stages  # noqa: E402
from environments.shared.k3_panel import K3PanelConfig  # noqa: E402
from environments.shared.model_id import ModelId  # noqa: E402
from environments.shared.species_registry import get_species_config  # noqa: E402

mid = ModelId("tyrannosaurus_rex", "anatomical")
configs = copy.deepcopy(load_all_stages(mid))
configs[1]["ppo_kwargs"].update(n_steps=64, batch_size=64, n_epochs=1, critic_phase=True)
model = train_base.train(get_species_config(mid), configs, 1, 64, n_envs=1, seed=42, eval_freq=10**9, save_freq=10**9,
                         verbose=0, output_dir=str(out / "stage"), use_tensorboard=False, report_metrics=False,
                         save_on_interrupt=False, k3_panel=K3PanelConfig())
print("model", type(model).__name__, type(model.policy).__name__, model.num_timesteps, model._n_updates)
panel = json.loads((out / "stage" / "k3_panel.json").read_text())
print("k3_panel.json keys", sorted(panel)[:12])
print(json.dumps(panel, default=str)[:1500])
