"""Independent cross-check of one switch-off run's configuration files (adapted from the s60 cross-check's xcheck.py
and recipe.py; every digest recipe re-read from the repository at the run's commit).

Usage: python -I -B xcheck80.py DL_DIR PINS_PC REF_DIR SIGMA_FILE SNAPSHOT_TXT
DL_DIR holds provenance.json, stage_config.json, task_fingerprint.json, trunk_run.json (downloaded copies);
REF_DIR holds the measured reference stage_config.json and task_fingerprint.json (seed 42, CPU).
Prints key paths, configuration values and digests only.
"""
import hashlib
import json
import math
import struct
import sys

COMMIT = "1dfe2327f773007fa7fd68460dc0c53740936ff9"
RUN_ID = "stance7_r10LTSBoff_s80"
SEED = 80
# environments/shared/harnesses/digest_snapshot.py UNPINNED_STAGE_CONFIG_KEYS (line 145 at the commit) plus the
# two blocks the round's view leaves out (pinned tools r10round_common.VIEW_UNPINNED)
VIEW_UNPINNED = ("species", "embodiment", "stage", "library_version", "git_commit", "run", "gpu",
                 "plant_identity", "task_fingerprint")
LOAD_KEYS = ("load_path", "load_mode", "parent_checkpoint_sha256", "resume_load_path", "resume_checkpoint_sha256")


def canon(v):
    """environments/shared/result_bundle/hashing.py canonical_json_sha256 at the commit."""
    return "sha256:" + hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def task_digest(tf):
    """environments/shared/task_fingerprint.py at the commit: sha256 of json.dumps(payload, sort_keys,
    compact separators, allow_nan False), the payload being the fingerprint without task_sha256."""
    payload = {k: v for k, v in tf.items() if k != "task_sha256"}
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def hp_digest(sc):
    """environments/shared/config.py recorded_hyperparameters_sha256's derived branch at the commit:
    {"algorithm": ALGORITHM, "hyperparameters": block, "shaping": curriculum keys starting warmup_/ramp_}."""
    cur = sc.get("curriculum", {}) or {}
    view = {"algorithm": sc["algorithm"].upper(), "hyperparameters": sc.get("hyperparameters", {}),
            "shaping": {str(k): v for k, v in cur.items() if str(k).startswith(("warmup_", "ramp_"))}}
    return canon(view), sorted(view["shaping"])


def view_digest(sc):
    return canon({k: v for k, v in sc.items() if k not in VIEW_UNPINNED})


def load(p):
    raw = open(p, "rb").read()
    return raw, json.loads(raw.decode("utf-8"))


def flat(o, prefix=""):
    out = {}
    if isinstance(o, dict):
        if not o and prefix:
            out[prefix] = {}
        for k, v in o.items():
            out.update(flat(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = o
    return out


def search(o, needles, prefix=""):
    hits = []
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if any(n in str(k).lower() for n in needles):
                hits.append(("key", p))
            hits += search(v, needles, p)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            hits += search(v, needles, f"{prefix}[{i}]")
    elif isinstance(o, str):
        if any(n in o.lower() for n in needles):
            hits.append(("value", prefix))
    return hits


def keys_any_depth(o, names, prefix=""):
    found = []
    if isinstance(o, dict):
        for k, v in o.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            if k in names:
                found.append(p)
            found += keys_any_depth(v, names, p)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            found += keys_any_depth(v, names, f"{prefix}[{i}]")
    return found


def diff(a, b, la, lb):
    fa, fb = flat(a), flat(b)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        if k not in fa:
            rows.append((k, "<absent>", repr(fb[k])))
        elif k not in fb:
            rows.append((k, repr(fa[k]), "<absent>"))
        elif fa[k] != fb[k] or type(fa[k]) is not type(fb[k]):
            rows.append((k, repr(fa[k]), repr(fb[k])))
    print(f"--- leaf diff {la} vs {lb}: {len(rows)} differing leaf paths")
    for r in rows:
        print("   ", r[0], "|", r[1], "|", r[2])
    return rows


def dig(o, path):
    for part in path.split("."):
        if not isinstance(o, dict) or part not in o:
            return "<absent>"
        o = o[part]
    return o


def show(label, value):
    print(f"{label}: {json.dumps(value)}")


def main(dl, pins_p, ref_dir, sigma_p, snap_p):
    prov_raw, prov = load(f"{dl}/provenance.json")
    sc_raw, sc = load(f"{dl}/stage_config.json")
    tf_raw, tf = load(f"{dl}/task_fingerprint.json")
    tr_raw, tr = load(f"{dl}/trunk_run.json")
    _, pins = load(pins_p)
    ref_raw, ref = load(f"{ref_dir}/stage_config.json")
    reftf_raw, reftf = load(f"{ref_dir}/task_fingerprint.json")
    for name, raw in (("provenance", prov_raw), ("stage_config", sc_raw), ("task_fingerprint", tf_raw),
                      ("trunk_run", tr_raw), ("REF stage_config", ref_raw), ("REF task_fingerprint", reftf_raw)):
        print(f"sha256 {name}: {hashlib.sha256(raw).hexdigest()} ({len(raw)} bytes)")

    print("\n=== (a) commit / dirty")
    show("provenance.repository_commit", prov.get("repository_commit"))
    show("provenance.repository_dirty", prov.get("repository_dirty"))
    show("provenance.repository_patch_sha256", prov.get("repository_patch_sha256"))
    show("provenance.repository_url", prov.get("repository_url"))
    show("stage_config.git_commit", sc.get("git_commit"))
    print("commit == expected (prov, sc):", prov.get("repository_commit") == COMMIT, sc.get("git_commit") == COMMIT)
    print("dirty is JSON false:", prov.get("repository_dirty") is False)

    print("\n=== (b) seed")
    for p in ("training_seed", "seed_roles.training", "seed_roles.checkpoint_selection_evaluation",
              "seed_roles.publication_evaluation", "seed_roles.certification_panel", "evaluation_seeds"):
        show("provenance." + p, dig(prov, p))
    show("stage_config.run.seed", dig(sc, "run.seed"))
    print("seed checks: training_seed", prov.get("training_seed") == SEED, "| seed_roles.training",
          dig(prov, "seed_roles.training") == SEED, "| run.seed", dig(sc, "run.seed") == SEED,
          "| selection = seed+1000", dig(prov, "seed_roles.checkpoint_selection_evaluation") == SEED + 1000,
          "| publication = seed+3000", dig(prov, "seed_roles.publication_evaluation") == SEED + 3000)

    print("\n=== (c) run id")
    show("provenance.run_id", prov.get("run_id"))
    show("stage_config.run.label", dig(sc, "run.label"))
    print("run_id == expected:", prov.get("run_id") == RUN_ID)

    print("\n=== (d) stage / species / embodiment / algorithm / n_envs / timesteps")
    for p in ("name", "stage", "species", "embodiment", "algorithm", "run.n_envs", "run.timesteps",
              "curriculum.timesteps", "curriculum.gate_kind", "task_fingerprint.stage", "task_fingerprint.backend"):
        show("stage_config." + p, dig(sc, p))
    for p in ("species", "embodiment", "algorithm", "backend", "parallel_envs", "hardware"):
        show("provenance." + p, dig(prov, p))
    print("checks: name stance", sc.get("name") == "stance", "| species", sc.get("species") == "tyrannosaurus_rex"
          and prov.get("species") == "tyrannosaurus_rex", "| embodiment", sc.get("embodiment") == "anatomical"
          and prov.get("embodiment") == "anatomical", "| algorithm PPO", sc.get("algorithm") == "PPO"
          and prov.get("algorithm") == "PPO", "| n_envs 4", dig(sc, "run.n_envs") == 4 and prov.get("parallel_envs") == 4,
          "| timesteps 12,000,000", dig(sc, "run.timesteps") == 12_000_000
          and dig(sc, "curriculum.timesteps") == 12_000_000)

    print("\n=== (e) the switch")
    hp = sc.get("hyperparameters", {})
    print("hyperparameters keys:", sorted(hp))
    print("'critic_phase' in hyperparameters:", "critic_phase" in hp)
    print("dig hyperparameters.critic_phase:", dig(sc, "hyperparameters.critic_phase"))
    for name, raw in (("stage_config", sc_raw), ("provenance", prov_raw), ("task_fingerprint", tf_raw),
                      ("trunk_run", tr_raw)):
        low = raw.lower()
        print(f"raw {name}: contains 'critic_phase' {b'critic_phase' in raw}, 'critic' {b'critic' in low}, "
              f"'phase' {b'phase' in low}")
    print("key/value search stage_config phase/critic:", search(sc, ["phase", "critic"]))
    print("key/value search provenance phase/critic:", search(prov, ["phase", "critic"]))
    print("pins off_arm.critic_phase_switch:", json.dumps(pins["off_arm"].get("critic_phase_switch")))

    print("\n=== (f) trunk")
    show("trunk_run.json", tr)
    print("trunk_from is the empty string:", tr.get("trunk_from") == "" and isinstance(tr.get("trunk_from"), str))
    print("load keys at any depth in stage_config:", keys_any_depth(sc, LOAD_KEYS))
    print("load keys at any depth in provenance:", keys_any_depth(prov, LOAD_KEYS))
    print("search sc trunk/retrain/resume/load/warm/parent/ancestor/widen:",
          search(sc, ["trunk", "retrain", "resume", "load", "warm", "parent", "ancestor", "widen"]))
    print("search prov trunk/retrain/resume/load/warm/parent/ancestor/widen:",
          search(prov, ["trunk", "retrain", "resume", "load", "warm", "parent", "ancestor", "widen"]))
    show("provenance.sessions count", len(prov.get("sessions", [])))
    show("provenance.model_hash", prov.get("model_hash"))
    show("provenance.config_hash", prov.get("config_hash"))

    print("\n=== (g) K3 sigma readiness")
    sraw = open(sigma_p, "rb").read()
    sdoc = json.loads(sraw.decode("utf-8"))
    fsha = hashlib.sha256(sraw).hexdigest()
    print("sigma file sha256:", fsha, "| pins sigma_file_sha256:", pins.get("sigma_file_sha256"),
          "| equal:", fsha == pins.get("sigma_file_sha256"))
    node = sdoc
    for part in "control_sigma.80".split("."):
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict) and "vector" in node:
        node = node["vector"]
    if isinstance(node, list):
        vec = [float(x) for x in node]
        ok = len(vec) > 0 and all(math.isfinite(x) and x > 0 for x in vec)
        vsha = hashlib.sha256(struct.pack("<%dd" % len(vec), *vec)).hexdigest()
        print("control_sigma.80: length", len(vec), "| finite and positive:", ok)
        print("vector sha256 (<f8 bytes):", vsha, "| pins sigma_vector_sha256['80']:",
              pins.get("sigma_vector_sha256", {}).get("80"), "| equal:",
              vsha == pins.get("sigma_vector_sha256", {}).get("80"))
    else:
        print("control_sigma.80: NO VECTOR at that path")
    print("k3/panel/sigma in stage_config:", search(sc, ["k3", "panel", "sigma"]))
    print("k3/panel/sigma in provenance:", search(prov, ["k3", "panel", "sigma"]))

    print("\n=== (h) digests")
    rec_hp = dig(sc, "run.hyperparameters_sha256")
    my_hp, shaping = hp_digest(sc)
    my_view = view_digest(sc)
    rec_task = dig(sc, "task_fingerprint.task_sha256")
    file_task = tf.get("task_sha256")
    my_task_sc = task_digest(sc["task_fingerprint"])
    my_task_file = task_digest(tf)
    off = pins["off_arm"]
    print("recorded run.hyperparameters_sha256:   ", rec_hp)
    print("recomputed hyperparameters digest:     ", my_hp, "(shaping keys:", shaping, ")")
    print("pins off_arm.hyperparameters_sha256:   ", off["hyperparameters_sha256"])
    print("pins switch-ON hyperparameters_sha256: ", pins.get("hyperparameters_sha256"))
    print("  recorded == recomputed:", rec_hp == my_hp, "| == off pin:", rec_hp == off["hyperparameters_sha256"],
          "| == ON pin:", rec_hp == pins.get("hyperparameters_sha256"))
    print("recomputed stage_config_view_sha256:  ", my_view)
    print("pins off_arm.stage_config_view_sha256: ", off["stage_config_view_sha256"])
    print("pins switch-ON stage_config_view_sha256:", pins.get("stage_config_view_sha256"))
    print("  == off pin:", my_view == off["stage_config_view_sha256"], "| == ON pin:",
          my_view == pins.get("stage_config_view_sha256"))
    print("stage_config.task_fingerprint.task_sha256:", rec_task)
    print("task_fingerprint.json task_sha256:        ", file_task)
    print("recomputed task digest (sc block / file): ", my_task_sc, my_task_file)
    print("pins off_arm.task_sha256:                 ", off["task_sha256"], "| pins task_sha256 (ON):",
          pins.get("task_sha256"))
    print("  recorded == recomputed:", rec_task == my_task_sc and file_task == my_task_file,
          "| == off pin:", rec_task == off["task_sha256"] and file_task == off["task_sha256"])
    print("task_fingerprint.json == stage_config.task_fingerprint (parsed):", tf == sc["task_fingerprint"])
    print("task_fingerprint.json bytes == REF task_fingerprint.json bytes:", tf_raw == reftf_raw)
    print("plant_identity == pins.plant_identity:", sc.get("plant_identity") == pins.get("plant_identity"),
          "| provenance.plant_identity == sc.plant_identity:", prov.get("plant_identity") == sc.get("plant_identity"))
    print("reward_weights (minus command keys) == task_fingerprint.env:",
          {k: v for k, v in sc["reward_weights"].items() if k not in ("command_config", "command_mode")} == tf["env"])
    # the repository's own digest snapshot at the commit
    snap = {}
    for line in open(snap_p, encoding="utf-8"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) == 6 and parts[:4] == ["stage", "tyrannosaurus_rex", "anatomical", "stance"]:
            snap[parts[4]] = parts[5]
    print("repo digest snapshot (T. rex anatomical stance) at the commit:", json.dumps(snap))
    print("  snapshot hyperparameters_sha256.PPO == recorded:", snap.get("hyperparameters_sha256.PPO") == rec_hp,
          "| snapshot stage_config_view_sha256.PPO == recomputed view:",
          snap.get("stage_config_view_sha256.PPO") == my_view, "| snapshot task_sha256 == recorded:",
          snap.get("task_sha256") == rec_task)

    print("\n=== reference (measured off, seed 42, CPU) digests")
    print("REF run.hyperparameters_sha256:", dig(ref, "run.hyperparameters_sha256"), "| recomputed:", hp_digest(ref)[0])
    print("REF view:", view_digest(ref), "| REF task:", dig(ref, "task_fingerprint.task_sha256"))
    print("REF 'critic_phase' in hyperparameters:", "critic_phase" in ref.get("hyperparameters", {}))
    print("\n=== leaf diffs")
    diff(ref, sc, "REF(off, seed 42)", "RUN(off s80)")
    diff(reftf, tf, "REF task_fingerprint", "RUN task_fingerprint")
    print("top-level keys equal (REF vs RUN):", sorted(ref) == sorted(sc), sorted(set(ref) ^ set(sc)))


if __name__ == "__main__":
    main(*sys.argv[1:6])
