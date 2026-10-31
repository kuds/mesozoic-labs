import hashlib, json, os, sys
PS = "/home/user/work/phase_screen"
def rd(p):
    fd = os.open(p, os.O_RDONLY | os.O_NOATIME)
    try:
        chunks = []
        while True:
            b = os.read(fd, 1 << 20)
            if not b: break
            chunks.append(b)
        return b"".join(chunks)
    finally:
        os.close(fd)
ap_bytes = open(f"{PS}/APPROVED", "rb").read()
ap = json.loads(ap_bytes); aps = hashlib.sha256(ap_bytes).hexdigest()
NAMES = ["control_2901", "critic_phase_2901", "control_2911", "critic_phase_2911",
         "control_2921", "critic_phase_2921", "control_2931", "critic_phase_2931"]
S, E = 2097152, 3538944
def pins_ok(m):
    probs = []
    if m.get("approved_sha256") != aps: probs.append("approved_sha256")
    if m.get("criteria_sha256") != ap["criteria_sha256"]: probs.append("criteria_sha256")
    if m.get("screen_sha256") != ap["screen_sha256"]: probs.append("screen_sha256")
    ts = m.get("tool_sha256") or {}
    for k, v in ts.items():
        if k in ap["tool_sha256"] and ap["tool_sha256"][k] != v: probs.append(f"tool {k}")
    missing = [k for k in ap["tool_sha256"] if k not in ts]
    code = m.get("code") or {}
    if code.get("tree_digest") != ap["code_tree_digest"]: probs.append("code tree_digest")
    return probs, sorted(ts), missing
bad = 0
print("== COUNTED")
for n in NAMES:
    m = json.loads(rd(f"{PS}/runs/{n}/run_meta.json"))
    probs, tools, missing = pins_ok(m)
    arm, seed = n.rsplit("_", 1)
    argv_ok = m.get("argv", [None])[1:] == ["--arm", arm, "--seed", seed]
    ok = (m.get("status") == "done" and m.get("exit") == 0 and m.get("stop_reason") == "end" and m.get("T") == E
          and m.get("S") == S and m.get("E") == E and m.get("arm") == arm and m.get("seed") == int(seed)
          and m.get("mode") == "continuation" and not probs and argv_ok)
    bad += not ok
    print(f"{n}: status={m.get('status')} exit={m.get('exit')} stop={m.get('stop_reason')} T={m.get('T')} S={m.get('S')} E={m.get('E')} "
          f"mode={m.get('mode')} argv_ok={argv_ok} start={m.get('utc_start')} end={m.get('utc_end')} pins_problems={probs} "
          f"tools_recorded={tools} missing_vs_APPROVED={missing}")
    print(f"    code.dir={m.get('code', {}).get('dir')} commit={m.get('code', {}).get('commit')} tree={str(m.get('code', {}).get('tree_digest'))[:12]} "
          f"modules_root={m.get('modules_root')} modules_outside_code={m.get('modules_outside_code')} code_seal_after={m.get('code_seal_after')}")
    sp = m.get("start_pair") or {}
    print(f"    start_pair zip_sha={str(sp.get('zip_sha256'))[:12]} pkl_sha={str(sp.get('pkl_sha256'))[:12]} run_id_parent={m.get('run_id_parent')} "
          f"tensorboard_log={m.get('tensorboard_log')} environment={json.dumps(m.get('environment'))[:300]}")
    cj = f"{PS}/runs/{n}/check.json"
    if os.path.exists(cj):
        c = json.loads(rd(cj))
        print(f"    check.json: check={c.get('check')} failures={c.get('failures')}")
        bad += c.get("check") != "pass"
    else:
        print("    check.json: absent"); bad += 1
print("== CRASHED (pins and allowed fields only)")
for n in NAMES[4:]:
    for k in (1, 2):
        d = f"{PS}/runs/{n}_crashed{k}"
        m = json.loads(rd(f"{d}/run_meta.json"))
        probs, tools, missing = pins_ok(m)
        hb = m.get("heartbeat") if isinstance(m.get("heartbeat"), dict) else {}
        re_ = hb.get("rollout_end")
        early = (re_ is None or re_ < S + 131072)
        print(f"{n}_crashed{k}: status={m.get('status')} exit={m.get('exit')} last_step={m.get('last_step')} heartbeat.rollout_end={re_} "
              f"(EARLY by the step rule: {early}) utc_start={m.get('utc_start')} argv={m.get('argv', [None])[1:]} pins_problems={probs} missing={missing} "
              f"code_tree={str((m.get('code') or {}).get('tree_digest'))[:12]}")
        bad += bool(probs)
print("anomalies", bad)
