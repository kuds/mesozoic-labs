"""Grep the body text of a JATS XML file for keywords (context window).
Usage: python -I lit_xml_grep.py <xml> <window> <kw1> [kw2 ...]"""
import re, sys
import xml.etree.ElementTree as ET
p, win = sys.argv[1], int(sys.argv[2])
r = ET.parse(p).getroot()
txt = " ".join("".join(x.itertext()) for x in r.iter() if x.tag in ("p", "title", "caption", "td", "th", "label"))
txt = re.sub(r"\s+", " ", txt)
for kw in sys.argv[3:]:
    hits = list(re.finditer(kw, txt))
    print(f"### [{kw}] {len(hits)} hits")
    for m in hits[:6]:
        print("  ..." + txt[max(0, m.start() - win):m.start() + win] + "...")
