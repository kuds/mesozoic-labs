"""Read the render lane's records and recompute the facts the text quotes.

Reads only; writes one JSON file given as argv[2]. No repository code is imported.
"""
import hashlib
import json
import os
import sys

R = sys.argv[1]
OUT = sys.argv[2]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


facts = {}
v1 = os.path.join(R, "videos/run1/s42")
v2 = os.path.join(R, "videos/run2/s42")
mp1 = sorted(n for n in os.listdir(v1) if n.endswith(".mp4"))
mp2 = sorted(n for n in os.listdir(v2) if n.endswith(".mp4"))
facts["n_mp4_run1"] = len(mp1)
facts["n_mp4_run2"] = len(mp2)
facts["names_equal"] = mp1 == mp2
diff = [n for n in mp1 if sha(os.path.join(v1, n)) != sha(os.path.join(v2, n))]
facts["mp4_sha_differ"] = diff
sizes_rt = [os.path.getsize(os.path.join(v1, n)) for n in mp1 if n.endswith("_realtime.mp4")]
sizes_q = [os.path.getsize(os.path.join(v1, n)) for n in mp1 if n.endswith("_quarter.mp4")]
facts["n_realtime"] = len(sizes_rt)
facts["n_quarter"] = len(sizes_q)
facts["realtime_bytes_min_max"] = [min(sizes_rt), max(sizes_rt)]
facts["quarter_bytes_min_max"] = [min(sizes_q), max(sizes_q)]
facts["total_mp4_bytes_run1"] = sum(sizes_rt) + sum(sizes_q)
facts["ep00_realtime_sha"] = sha(os.path.join(v1, "s42_65536_ep00_reset3042_realtime.mp4"))
facts["videos_json_sha_run1"] = sha(os.path.join(v1, "videos.json"))
facts["videos_json_sha_run2"] = sha(os.path.join(v2, "videos.json"))

vj = json.load(open(os.path.join(v1, "videos.json")))
facts["videos_json_top_keys"] = sorted(vj.keys())
for k in ("schema", "head", "tool_sha256", "probe_sha256", "checkpoint_sha256", "vecnorm_sha256",
          "standalone", "all_rolled_match", "camera", "fps", "dt"):
    if k in vj:
        facts["vj_" + k] = vj[k]

cmp = json.load(open(os.path.join(R, "logs/compare_videos.json")))
facts["compare_videos"] = {k: cmp[k] for k in cmp if not isinstance(cmp[k], (list, dict)) or k in (
    "files_differing_sha256",)}

for run in ("run1", "run2"):
    d = os.path.join(R, "replay", run)
    fin = os.path.join(d, "tyrannosaurus_rex_anatomical_ppo_stage2_final.mp4")
    sel = os.path.join(d, "tyrannosaurus_rex_anatomical_ppo_stage2_selected.mp4")
    facts["replay_" + run] = {
        "final_sha": sha(fin), "final_bytes": os.path.getsize(fin),
        "selected_sha": sha(sel), "selected_bytes": os.path.getsize(sel),
        "record_sha": sha(os.path.join(d, "replay_record.json")),
    }
rr = json.load(open(os.path.join(R, "replay/run1/replay_record.json")))
rr2 = json.load(open(os.path.join(R, "replay/run2/replay_record.json")))
facts["replay_record_keys"] = sorted(rr.keys())
facts["replay_records_equal"] = rr == rr2
for k in ("replay_seed_calls", "warnings", "stage", "species", "variant", "algorithm", "seed"):
    if k in rr:
        facts["rr_" + k] = rr[k]

with open(OUT, "w") as f:
    json.dump(facts, f, indent=1, sort_keys=True)
print(json.dumps(facts, indent=1, sort_keys=True))
