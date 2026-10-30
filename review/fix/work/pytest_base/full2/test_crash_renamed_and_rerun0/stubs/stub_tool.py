
import json, os, sys
root = os.environ["BS_ROOT"]
kind, args = sys.argv[1], sys.argv[2:]
with open(f"{root}/calls.log", "a") as fh:
    fh.write(kind + " " + " ".join(args) + "\n")
if kind == "check":
    meta = json.load(open(os.path.join(args[0], "run_meta.json")))
    if meta.get("stub_check") == "fail" or meta.get("status") != "done" or meta.get("exit") != 0:
        sys.exit(4)
    json.dump({"pass": True}, open(os.path.join(args[0], "check.json"), "w"))
sys.exit(0)
