import json, sys, re, os, glob
d = sys.argv[1]
outcome = re.compile(r"phase_screen/(runs|reads|score|lost\.json|tmp|logs)\b|\$PS/(runs|reads|score|lost\.json|tmp|logs)\b|attempt\d*\.log|\.check\.log")
bare = re.compile(r"(phase_screen|\$PS|\$\{PS\})(?![/\w])")
for f in sorted(glob.glob(os.path.join(d, "agent-*.jsonl"))):
    agent = os.path.basename(f)[6:14]
    for raw in open(f, "rb"):
        try: x = json.loads(raw)
        except Exception: continue
        m = x.get("message") or {}
        cont = m.get("content") if isinstance(m, dict) else None
        if not isinstance(cont, list): continue
        for c in cont:
            if not isinstance(c, dict) or c.get("type") != "tool_use": continue
            inp = c.get("input") or {}
            s = inp.get("command") if "command" in inp else json.dumps(inp)
            if c.get("name") in ("Write", "Edit"):
                s = "PATH=" + str(inp.get("file_path"))   # content of written files is not an access
            tags = []
            if outcome.search(s): tags.append("OUTCOME-PATH-TOKEN")
            if bare.search(s): tags.append("PS-ROOT-BARE")
            if "find " in s and "/home/user/work" in s: tags.append("FIND-WORK")
            if "tb_buffer" in s: tags.append("TB_BUFFER")
            if tags:
                print(f"### {agent} {x.get('timestamp')} {c.get('name')} {tags}")
                print("   " + s[:1500].replace("\n", "\n   "))
