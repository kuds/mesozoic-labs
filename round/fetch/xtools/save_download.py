"""Save one download_file_content response for FILE_ID, exactly as the session's record holds it, to a new file
<outdir>/<stem>.t<k>.json: the earliest such response not yet used (used tool_use ids are kept in --used; each
download call is one transfer).  Prints the saved file's path on the last line; never prints the content.

    python -I -B save_download.py --transcript T --file-id ID --outdir DIR --stem STEM --used LEDGER
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
    ap.add_argument("--file-id", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--stem", required=True)
    ap.add_argument("--used", required=True)
    a = ap.parse_args(argv)
    uses, results = SR.records(a.transcript)
    led = Path(a.used)
    used = set(led.read_text().split()) if led.is_file() else set()
    calls = sorted((i, tid) for tid, (i, name, inp) in uses.items()
                   if name.endswith("download_file_content") and isinstance(inp, dict)
                   and inp.get("fileId") == a.file_id and tid not in used and tid in results)
    if not calls:
        print(f"REFUSED: no unused download_file_content response for {a.file_id}", file=sys.stderr)
        return 2
    _i, tid = calls[0]
    _j, text, ts, err = results[tid]
    if err or not isinstance(text, str):
        with open(led, "a", encoding="utf-8") as fh:
            fh.write(tid + "\n")
        print(f"REFUSED: {tid}'s result is an error or not one text block (marked used)", file=sys.stderr)
        return 3
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    k = 1
    while (out / f"{a.stem}.t{k}.json").exists():
        k += 1
    p = out / f"{a.stem}.t{k}.json"
    with open(p, "x", encoding="utf-8", newline="") as fh:
        fh.write(text)
    with open(led, "a", encoding="utf-8") as fh:
        fh.write(tid + "\n")
    print(f"{tid} (result ts {ts}) -> {p} ({len(text)} chars)", file=sys.stderr)
    print(str(p))
    return 0


if __name__ == "__main__":
    sys.exit(main())
