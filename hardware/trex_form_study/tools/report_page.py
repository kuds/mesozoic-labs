"""Turn an owner report in Markdown into the self-contained page published as a private claude.ai artifact.

The body goes through pandoc (GitHub-flavoured Markdown to HTML5); the theme, light and dark, is the round-two review
page's (``designlab/page_kit.py``). Every table gets its own horizontal scroller so that the page never scrolls sideways
on a phone. The Stage 1 and Stage 1b report pages were made this way. Run from anywhere:

    python tools/report_page.py REPORT.md OUT.html --kicker "Mesozoic Labs robotic T. rex · Stage 2 · simulation" \
        --lede "One sentence on what the page answers." [--title "Tab title"] [--footer "Built from ..."]

Then publish OUT.html with the Artifact tool (see the artifact-design skill's page contract: the publisher wraps the
file in its own doctype and head) and check it at desktop and phone width with ``node tools/shot_page.js``. Needs
pandoc on PATH (``/usr/bin/pandoc`` in the cloud image).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY))
from designlab import page_kit  # noqa: E402

EXTRA_CSS = """
<style>
.report{display:grid;gap:12px;padding-block:8px 0}
.report>*{min-width:0;max-width:76ch}
.report>.tablewrap{max-width:min(100%,1120px)}
.report h2{margin-top:30px;padding-top:22px;border-top:1px solid var(--rule)}
.report h3{margin-top:14px}
.report ul,.report ol{margin:0;padding-left:1.25em;display:grid;gap:6px}
.report blockquote{margin:0;padding:10px 14px;background:var(--wash);border-radius:10px;color:var(--ink-2)}
.report .tablewrap{max-width:min(100%,1120px)}
.report table{font-size:13.5px}
.report hr{border:0;border-top:1px solid var(--rule);margin:8px 0}
</style>
"""


def build(src: Path, kicker: str, lede: str, title: str | None, footer: str | None) -> str:
    if shutil.which("pandoc") is None:
        raise SystemExit("pandoc is not on PATH")
    md = src.read_text()
    m = re.match(r"\s*#\s+(.+?)\n", md)  # the first H1 becomes the page header
    heading = m.group(1).strip() if m else src.stem
    body_md = md[m.end():] if m else md
    frag = subprocess.run(["pandoc", "--from", "gfm", "--to", "html5", "--no-highlight"], input=body_md, text=True,
                          capture_output=True, check=True).stdout
    frag = frag.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")
    header = (f'<header class="top"><div class="kicker">{page_kit.esc(kicker)}</div>'
              f"<h1>{page_kit.esc(heading)}</h1><p class=\"lede\">{page_kit.esc(lede)}</p></header>")
    foot = f"<footer><p>{page_kit.esc(footer or f'Built from {src.name}. Simulation only.')}</p></footer>"
    return page_kit.page(title or heading, EXTRA_CSS + header + '<article class="report">' + frag + "</article>" + foot)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--kicker", required=True, help="the small line above the title")
    ap.add_argument("--lede", required=True, help="one sentence under the title")
    ap.add_argument("--title", default=None, help="the browser tab's title, two to four words (default: the H1)")
    ap.add_argument("--footer", default=None)
    a = ap.parse_args(argv)
    html = build(a.src, a.kicker, a.lede, a.title, a.footer)
    a.out.write_text(html)
    print(a.out, len(html), "bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
