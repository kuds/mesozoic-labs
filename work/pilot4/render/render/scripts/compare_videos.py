"""Compare two pilot4_video.py output folders byte for byte, and list every output file (render lane; scratch).

    python -I -B compare_videos.py RUN1_DIR RUN2_DIR OUT_JSON

For each file in either folder: size, sha256, and for an mp4 the ffprobe stream fields with a decoded frame count
(-count_frames). Then: the same file names in both, equal sha256 per name, and the two videos.json records equal once
the out-folder paths are mapped (the only field that names the folder). Imports no repository code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                          "stream=codec_name,profile,width,height,pix_fmt,r_frame_rate,nb_frames,nb_read_frames,"
                          "duration", "-of", "json", str(path)], capture_output=True, check=True, text=True).stdout
    return json.loads(out)["streams"][0]


def listing(d: Path) -> dict:
    res = {}
    for p in sorted(d.iterdir()):
        ent = {"bytes": p.stat().st_size, "sha256": sha(p)}
        if p.suffix == ".mp4":
            ent["ffprobe"] = probe(p)
        res[p.name] = ent
    return res


def main() -> int:
    a, b, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    la, lb = listing(a), listing(b)
    names_equal = sorted(la) == sorted(lb)
    mp4s = [n for n in la if n.endswith(".mp4")]
    differing = [n for n in sorted(set(la) | set(lb)) if n != "videos.json"
                 and (n not in la or n not in lb or la[n]["sha256"] != lb[n]["sha256"])]
    ja = json.loads((a / "videos.json").read_text())
    jb_text = (b / "videos.json").read_text().replace(str(b), str(a))
    jb = json.loads(jb_text)
    json_equal_after_path_map = ja == jb
    json_keys_differing = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
    frames = {n: int(la[n]["ffprobe"]["nb_read_frames"]) for n in mp4s}
    rec = {"run1": str(a), "run2": str(b), "files_run1": len(la), "files_run2": len(lb), "names_equal": names_equal,
           "mp4_count": len(mp4s), "mp4_bytes_total_run1": sum(la[n]["bytes"] for n in mp4s),
           "files_differing_sha256": differing, "videos_json_equal_after_path_map": json_equal_after_path_map,
           "videos_json_keys_differing": json_keys_differing,
           "decoded_frames_by_kind": {k: sorted(set(v for n, v in frames.items() if n.endswith(f"_{k}.mp4")))
                                      for k in ("realtime", "quarter")},
           "ffprobe_kinds": {k: sorted({json.dumps({f: la[n]["ffprobe"].get(f) for f in (
               "codec_name", "profile", "width", "height", "pix_fmt", "r_frame_rate")}) for n in mp4s
               if n.endswith(f"_{k}.mp4")}) for k in ("realtime", "quarter")},
           "listing_run1": la, "listing_run2": lb}
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if not k.startswith("listing")}, indent=1))
    return 0 if names_equal and not differing and json_equal_after_path_map else 1


if __name__ == "__main__":
    sys.exit(main())
