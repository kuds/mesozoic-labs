"""Independent cross-check of the switch-off s60 run's provenance.json and 01_stance/stage_config.json.

Usage: python -I -B xcheck.py PROVENANCE STAGE_CONFIG PINS_PC REF_STAGE_CONFIG
Prints key paths and configuration values only.
"""
import hashlib
import json
import sys


def canon(v):
    return "sha256:" + hashlib.sha256(
        json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


VIEW_UNPINNED = ("species", "embodiment", "stage", "library_version", "git_commit", "run", "gpu",
                 "plant_identity", "task_fingerprint")


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


def diff(a, b, label_a, label_b):
    fa, fb = flat(a), flat(b)
    rows = []
    for k in sorted(set(fa) | set(fb)):
        if k not in fa:
            rows.append((k, "<absent>", repr(fb[k])))
        elif k not in fb:
            rows.append((k, repr(fa[k]), "<absent>"))
        elif fa[k] != fb[k] or type(fa[k]) is not type(fb[k]):
            rows.append((k, repr(fa[k]), repr(fb[k])))
    print(f"--- diff {label_a} vs {label_b}: {len(rows)} differing leaf paths")
    for r in rows:
        print("   ", r[0], "|", r[1], "|", r[2])
    return rows


def main(prov_p, sc_p, pins_p, ref_p):
    raw_sc = open(sc_p, "rb").read()
    prov = json.loads(open(prov_p, "rb").read().decode("utf-8"))
    sc = json.loads(raw_sc.decode("utf-8"))
    pins = json.loads(open(pins_p, "rb").read().decode("utf-8"))
    ref = json.loads(open(ref_p, "rb").read().decode("utf-8"))
    print("sha256 stage_config", hashlib.sha256(raw_sc).hexdigest())
    print("sha256 provenance  ", hashlib.sha256(open(prov_p, "rb").read()).hexdigest())

    print("=== top-level keys stage_config:", sorted(sc))
    print("=== top-level keys provenance:", sorted(prov))
    print("=== hyperparameters keys:", sorted(sc.get("hyperparameters", {})))
    print("'critic_phase' in hyperparameters:", "critic_phase" in sc.get("hyperparameters", {}))
    print("raw bytes contain 'critic_phase':", b"critic_phase" in raw_sc)
    print("raw bytes contain 'phase':", b"phase" in raw_sc.lower())
    print("raw bytes contain 'critic':", b"critic" in raw_sc.lower())
    print("search sc for phase/critic:", search(sc, ["phase", "critic"]))
    print("search prov for phase/critic:", search(prov, ["phase", "critic"]))
    print("search sc for trunk/retrain/resume/init/pretrain/load/warm:",
          search(sc, ["trunk", "retrain", "resume", "init_from", "pretrain", "load", "warm", "parent", "source"]))
    print("search prov for trunk/retrain/resume/init/pretrain/load/warm:",
          search(prov, ["trunk", "retrain", "resume", "init_from", "pretrain", "load", "warm", "parent", "source"]))
    print("search sc for k3/panel/sigma/vector:", search(sc, ["k3", "panel", "sigma", "vector"]))
    print("search prov for k3/panel/sigma/vector:", search(prov, ["k3", "panel", "sigma", "vector"]))
    print("search sc for ref/branch/dir/path/run_id:", search(sc, ["ref", "branch", "dir", "path", "run_id", "folder"]))
    print("search prov for ref/branch/dir/path/run_id:", search(prov, ["ref", "branch", "dir", "path", "run_id", "folder"]))

    hp = sc.get("hyperparameters")
    print("=== digests")
    print("recomputed canonical(hyperparameters):", canon(hp))
    print("recorded run.hyperparameters_sha256:  ", sc.get("run", {}).get("hyperparameters_sha256"))
    print("pins off_arm.hyperparameters_sha256:   ", pins["off_arm"]["hyperparameters_sha256"])
    print("pins hyperparameters_sha256_round_switch_off:", pins.get("hyperparameters_sha256_round_switch_off"))
    view = canon({k: v for k, v in sc.items() if k not in VIEW_UNPINNED})
    print("recomputed stage_config_view_sha256:  ", view)
    print("pins off_arm.stage_config_view_sha256: ", pins["off_arm"]["stage_config_view_sha256"])
    print("recorded task_fingerprint.task_sha256: ", sc.get("task_fingerprint", {}).get("task_sha256"))
    print("pins off_arm.task_sha256:              ", pins["off_arm"]["task_sha256"])
    tf = dict(sc.get("task_fingerprint", {}))
    rec_task = tf.pop("task_sha256", None)
    print("recomputed canonical(task_fingerprint minus task_sha256):", canon(tf), "(informational; recipe may differ)")
    print("plant_identity == pins.plant_identity:", sc.get("plant_identity") == pins.get("plant_identity"))
    print("prov.plant_identity == sc.plant_identity:", prov.get("plant_identity") == sc.get("plant_identity"))
    print("pins switch-ON hyperparameters_sha256 (must NOT match):", pins.get("hyperparameters_sha256"),
          "match?", pins.get("hyperparameters_sha256") == sc.get("run", {}).get("hyperparameters_sha256"))
    print("pins switch-ON stage_config_view_sha256 (must NOT match):", pins.get("stage_config_view_sha256"),
          "match?", pins.get("stage_config_view_sha256") == view)

    # reference digests
    print("=== reference (dry pins_sc/off) digests")
    print("ref canonical(hyperparameters):", canon(ref.get("hyperparameters")))
    print("ref run.hyperparameters_sha256:", ref.get("run", {}).get("hyperparameters_sha256"))
    print("ref view:", canon({k: v for k, v in ref.items() if k not in VIEW_UNPINNED}))
    print("ref task_sha256:", ref.get("task_fingerprint", {}).get("task_sha256"))

    print("=== diffs")
    diff(ref, sc, "REF(dry off, seed 42)", "RUN(off s60)")
    # pinned subset compare
    print("--- off_arm pin vs run")
    for key, val in pins["off_arm"].items():
        print("   off_arm.", key, "=", json.dumps(val))
    print("=== provenance vs stage_config cross-fields")
    print("prov.repository_commit == sc.git_commit:", prov.get("repository_commit") == sc.get("git_commit"))
    print("prov.training_seed, seed_roles.training, sc.run.seed:", prov.get("training_seed"),
          prov.get("seed_roles", {}).get("training"), sc.get("run", {}).get("seed"))
    print("prov.parallel_envs, sc.run.n_envs:", prov.get("parallel_envs"), sc.get("run", {}).get("n_envs"))
    print("sc.run.timesteps, sc.curriculum.timesteps:", sc.get("run", {}).get("timesteps"),
          sc.get("curriculum", {}).get("timesteps"))
    print("prov.algorithm/backend:", prov.get("algorithm"), prov.get("backend"), "| sc.algorithm:", sc.get("algorithm"))
    print("prov species/embodiment:", prov.get("species"), prov.get("embodiment"),
          "| sc species/embodiment/name/stage:", sc.get("species"), sc.get("embodiment"), sc.get("name"), sc.get("stage"))
    print("task_fingerprint species/embodiment/stage/backend:", tf.get("species"), tf.get("embodiment"),
          tf.get("stage"), tf.get("backend"))
    print("reward_weights == task_fingerprint.env (minus command keys):",
          {k: v for k, v in sc["reward_weights"].items() if k not in ("command_config", "command_mode")} == tf.get("env"))


if __name__ == "__main__":
    main(*sys.argv[1:5])
