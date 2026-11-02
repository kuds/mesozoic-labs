"""Save the newest complete listing sequence (the last search_files call for parentId = FOLDER without a page token,
then each later call for it with a page token) for one folder, each page copied exactly as the session's record holds
it, to new files <outdir>/<stem>[_v<n>]_p<k>.json.  Prints the page files in order (space separated) on the last line.

    python -I -B save_listing.py --transcript T --folder-id ID --outdir DIR --stem STEM
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_record as SR  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--folder-id", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stem", required=True)
    a = ap.parse_args(argv)
    uses, results = SR.records(a.transcript)
    q = f"parentId = '{a.folder_id}'"
    seq = [(i, tid, inp) for tid, (i, name, inp) in uses.items()
           if name.endswith("search_files") and isinstance(inp, dict) and inp.get("query") == q]
    seq.sort()
    starts = [k for k, (_i, _t, inp) in enumerate(seq) if not inp.get("pageToken")]
    if not starts:
        print("REFUSED: no first-page call for this folder", file=sys.stderr)
        return 2
    chain = seq[starts[-1]:]
    # each later page must use the previous page's token
    pages = []
    prev_token = None
    for k, (_i, tid, inp) in enumerate(chain):
        r = results.get(tid)
        if r is None or r[3] or not isinstance(r[1], str):
            print(f"REFUSED: {tid} has no plain result", file=sys.stderr)
            return 2
        if k > 0 and inp.get("pageToken") != prev_token:
            print(f"REFUSED: page {k + 1} ({tid}) does not use the previous page's token", file=sys.stderr)
            return 2
        d = json.loads(r[1])
        prev_token = d.get("nextPageToken") if isinstance(d, dict) else None
        pages.append((tid, r[1]))
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    stem, v = a.stem, 1
    while any(out.glob(f"{stem}_p*.json")):
        v += 1
        stem = f"{a.stem}_v{v}"
    files = []
    for k, (tid, text) in enumerate(pages, 1):
        p = out / f"{stem}_p{k}.json"
        with open(p, "x", encoding="utf-8", newline="") as fh:
            fh.write(text)
        files.append(str(p))
        print(f"page {k}: {tid} -> {p} ({len(text)} chars)", file=sys.stderr)
    print(" ".join(files))
    return 0


if __name__ == "__main__":
    sys.exit(main())
