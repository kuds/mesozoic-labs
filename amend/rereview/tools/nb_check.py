"""Re-reviewer: read WT HEAD's notebook (git show, read only) and print the cells COLAB_STEPS names, with the
CRITIC_PHASE paths (on / off)."""
import json
import re
import subprocess

WT = "/home/user/work/pc_branch"
raw = subprocess.run(["git", "--no-optional-locks", "-C", WT, "show", "HEAD:notebooks/sb3_training.ipynb"],
                     check=True, capture_output=True).stdout
nb = json.loads(raw)
code = [c for c in nb["cells"] if c["cell_type"] == "code"]
print("code cells:", len(code), "all cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] != "code":
        continue
    if re.search(r"CRITIC_PHASE|critic_phase|Critic phase|RUN_ID\s*=|RUN_LABEL\s*=|Run directory|new run|RESUME_STAGE\s*=|"
                 r"Repository ref|In-run K3 panel|Behavior recipe|Timesteps", src):
        n = sum(1 for x in nb["cells"][:i + 1] if x["cell_type"] == "code")
        print(f"===== cell index {i} (code cell #{n})")
        for ln in src.splitlines():
            if re.search(r"CRITIC_PHASE|critic_phase|Critic phase|^RUN_ID|^RUN_LABEL|Run directory|new run|"
                         r"^RESUME_STAGE|Repository ref|In-run K3 panel|Behavior recipe|Timesteps|^SEED|^# ?Cell|^#@title", ln):
                print("   ", ln[:160])
