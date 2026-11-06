"""Print the key tree (to a depth) of each JSON file named on the command line. Read only."""
import json
import sys


def walk(obj, prefix, depth, out):
    if depth < 0:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            kind = type(v).__name__
            short = ""
            if not isinstance(v, (dict, list)):
                short = " = " + repr(v)[:100]
            elif isinstance(v, list):
                short = f" [len {len(v)}]" + (" = " + repr(v)[:100] if len(repr(v)) < 100 else "")
            out.append(f"{prefix}{k} ({kind}){short}")
            walk(v, prefix + "  ", depth - 1, out)


def main():
    depth = int(sys.argv[1])
    for path in sys.argv[2:]:
        with open(path) as f:
            data = json.load(f)
        out = [f"== {path.split('/prelaunch/')[-1]}"]
        walk(data, "  ", depth, out)
        print("\n".join(out))


if __name__ == "__main__":
    main()
