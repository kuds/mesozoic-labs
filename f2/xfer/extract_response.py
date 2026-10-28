"""Write the latest download_file_content response for one Drive file id, exactly as the connector returned it,
from a session transcript (JSONL) to a saved-response file for fetch_r10round.py transfer --saved.

No byte of the content is typed or edited: the response text is copied from the transcript's tool_result record.
Usage: python -I extract_response.py --transcripts DIR_OR_FILE [...] --file-id ID --after-utc ISO --out FILE
Picks, among tool_result records whose text parses as a JSON object with "content" and "id" == ID, the latest one
by the record's timestamp at or after --after-utc; refuses (exit 2) when none, or when --out exists.
A transcripts DIR is searched recursively for agent-*.jsonl (every session of every workflow under it).
"""
import argparse
import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def utc(text):
    """An ISO time ("2026-10-28T12:53:06.594Z" or "...06Z" or "+00:00") as an aware datetime; None if unreadable."""
    try:
        t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def texts(block):
    c = block.get("content")
    if isinstance(c, str):
        yield c
    elif isinstance(c, list):
        for x in c:
            if isinstance(x, dict) and x.get("type") == "text" and isinstance(x.get("text"), str):
                yield x["text"]


def candidates(path, file_id):
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if file_id not in line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict) or b.get("type") != "tool_result":
                    continue
                for t in texts(b):
                    try:
                        obj = json.loads(t)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and obj.get("id") == file_id and isinstance(obj.get("content"), str):
                        yield rec.get("timestamp") or "", t, obj


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", nargs="+", required=True)
    ap.add_argument("--file-id", required=True)
    ap.add_argument("--after-utc", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        print(f"REFUSED: {out} exists (each transfer gets a new file)", file=sys.stderr)
        return 2
    files = []
    for p in map(Path, a.transcripts):
        files += sorted(p.rglob("agent-*.jsonl")) if p.is_dir() else [p]
    after = utc(a.after_utc)
    if after is None:
        print(f"REFUSED: --after-utc {a.after_utc!r} is not an ISO time", file=sys.stderr)
        return 2
    found = [c for f in files for c in candidates(f, a.file_id) if utc(c[0]) is not None and utc(c[0]) >= after]
    if not found:
        print(f"REFUSED: no response for {a.file_id} at or after {a.after_utc}", file=sys.stderr)
        return 2
    ts, text, obj = max(found, key=lambda c: utc(c[0]))
    out.write_text(text, encoding="utf-8")
    n = len(base64.b64decode(obj["content"], validate=True))
    print(json.dumps({"out": str(out), "response_timestamp": ts, "id": obj["id"], "title": obj.get("title"),
                      "decoded_bytes": n, "responses_found": len(found)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
