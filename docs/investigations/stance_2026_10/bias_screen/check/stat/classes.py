import json, hashlib
BS = "/tmp/claude-0/-home-user-mesozoic-labs/184ee67c-2802-5235-8e52-6dc7daa72cd3/scratchpad/r10round/bias_screen"
appr_sha = hashlib.sha256(open(BS + "/APPROVED", "rb").read()).hexdigest()
for seed in (2101, 2111, 2121, 2131):
    for arm in ("control", "critic_blind_tail3"):
        n = f"{arm}_{seed}"
        rm = json.load(open(f"{BS}/runs/{n}/run_meta.json"))
        ck = json.load(open(f"{BS}/runs/{n}/check.json"))
        keys = {k: rm.get(k) for k in ("status", "exit", "stop_reason", "T", "num_timesteps_end", "arm", "seed")}
        shas = {k: v for k, v in rm.items() if "sha" in k.lower() and isinstance(v, str)}
        apr = [k for k, v in shas.items() if v == appr_sha]
        print(n, keys, "check pass:", ck.get("passed"), "failures:", ck.get("failures"), "APPROVED sha keys:", apr)
