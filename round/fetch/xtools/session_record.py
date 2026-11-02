"""Copy one Drive connector tool result, byte for byte as the session's own record (JSONL transcript) holds it, to a
new file (refuses when the file exists).  Nothing is typed or edited: the tool_result text is written as recorded.

    python -I -B session_record.py list    --transcript T            # Drive calls: line, tool_use_id, tool, input, result length
    python -I -B session_record.py extract --transcript T --id TOOLU --out FILE [--expect-tool NAME]

Prints only ids, tool names, inputs and lengths: never the result text itself.
"""
import argparse
import json
import sys
from pathlib import Path


def records(path):
    uses, results = {}, {}
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            msg = rec.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for b in content:
                if not isinstance(b, dict):
                    continue
                if b.get("type") == "tool_use" and "Google_Drive" in (b.get("name") or ""):
                    uses[b.get("id")] = (i, b.get("name"), b.get("input"))
                if b.get("type") == "tool_result":
                    c = b.get("content")
                    if isinstance(c, list):
                        txt = [x.get("text") for x in c if isinstance(x, dict) and x.get("type") == "text"]
                        c = txt[0] if len(txt) == 1 else None
                    results[b.get("tool_use_id")] = (i, c, rec.get("timestamp"), bool(b.get("is_error")))
    return uses, results


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "extract"))
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--id")
    ap.add_argument("--out")
    ap.add_argument("--expect-tool")
    a = ap.parse_args(argv)
    uses, results = records(a.transcript)
    if a.cmd == "list":
        for tid, (i, name, inp) in uses.items():
            r = results.get(tid)
            print(i, tid, name, json.dumps(inp), "result_len", None if r is None or r[1] is None else len(r[1]),
                  "ts", r and r[2], "error", r and r[3])
        return 0
    if a.id not in uses or a.id not in results:
        print(f"REFUSED: no Drive tool_use/tool_result pair for {a.id}", file=sys.stderr)
        return 2
    name = uses[a.id][1]
    if a.expect_tool and not name.endswith(a.expect_tool):
        print(f"REFUSED: {a.id} is a {name} call, not {a.expect_tool}", file=sys.stderr)
        return 2
    _i, text, ts, err = results[a.id]
    if err or not isinstance(text, str):
        print(f"REFUSED: {a.id}'s result is an error or not one text block", file=sys.stderr)
        return 2
    out = Path(a.out)
    if out.exists():
        print(f"REFUSED: {out} exists (each page or transfer gets a new file)", file=sys.stderr)
        return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "x", encoding="utf-8", newline="") as fh:
        fh.write(text)
    print(f"wrote {out}: {len(text)} chars from {name} {json.dumps(uses[a.id][2])} (result ts {ts})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
