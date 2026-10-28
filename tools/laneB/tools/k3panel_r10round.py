"""T. rex r10 round: the in-run K3 panel reader (criteria lean v5 M7; THIS12 lines 234-246).  Report only.

    PY -I -B k3panel_r10round.py --json k3_panel.json [--csv k3_panel.csv] --seed 60|42|80 [--pins P]
                                 [--copy MEASURING_COPY --models DIR] [--out OUT.json]

Reads the run's k3_panel.json (schema mesozoic.k3-panel/v1) and k3_panel.csv as written.  Per panel: the own and joint
falls of 20 and their seeds as the JSON records them (never recomputed), the panel's kind and failure, the sigma
vector's sha256 against the seed's pin, the panel's policy state-dict and observation-statistics sha256, matched
against the checkpoint pair at the same step when that pair is held (--models: a folder with the pair's zip and pkl;
the zip's policy.pth read with torch.load(weights_only=True), the pkl through the measuring copy's own stub unpickler
reporting.training_history.read_vecnormalize_stats, which runs no pickled code).  Panels with different sigma hashes
are printed as separate series, never joined.  A missing or unreadable k3_panel.json changes no class, statistic or
reading: it is printed (D-14).  Exit codes: 0 printed (also when the file is missing or unreadable); 2 bad arguments.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import csv  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import zipfile  # noqa: E402
from pathlib import Path  # noqa: E402

TOOLS = Path(__file__).resolve().parent


def _load(name):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


C = _load("r10round_common")


def pair_hashes(models_dir: Path, step, copy_dir):
    """(state_dict sha256, obs mean sha256, obs var sha256, problem) of the pair at step (k3_panel.py's hashes)."""
    import numpy as np
    zp, pp = C.pair_paths(step)
    z = Path(models_dir) / Path(zp).name
    p = Path(models_dir) / Path(pp).name
    if not z.is_file() or not p.is_file():
        return None, None, None, "pair not held"
    try:
        import torch
        with zipfile.ZipFile(z) as zz:
            state = torch.load(io.BytesIO(zz.read("policy.pth")), map_location="cpu", weights_only=True)
        d = hashlib.sha256()
        for key in sorted(state):
            d.update(str(key).encode("utf-8"))
            d.update(np.ascontiguousarray(state[key].detach().cpu().numpy()).tobytes())
        sd = d.hexdigest()
    except Exception as exc:  # noqa: BLE001 - report only
        return None, None, None, f"policy.pth unreadable: {exc!r}"
    try:
        sys.path.insert(0, str(copy_dir))
        from environments.shared.reporting.training_history import read_vecnormalize_stats
        st = read_vecnormalize_stats(p)

        def f64(v):
            return hashlib.sha256(np.ascontiguousarray(np.asarray(v, dtype="<f8")).tobytes()).hexdigest()
        return sd, f64(st["mean"]), f64(st["var"]), None
    except Exception as exc:  # noqa: BLE001 - report only
        return sd, None, None, f"pkl statistics unreadable: {exc!r}"


def read_panel(json_path, csv_path, seed, pins, models=None, copy_dir=None) -> dict:
    out = dict(file=str(json_path), seed=seed)
    try:
        raw = Path(json_path).read_bytes()
    except OSError as exc:
        out["absent"] = f"k3_panel.json missing or unreadable ({exc}); changes no class, statistic or reading (D-14)"
        return out
    out["sha256"] = C.sha256_bytes(raw)
    d = C.load_json_bytes(raw)
    if not isinstance(d, dict) or d.get("schema") != "mesozoic.k3-panel/v1":
        out["absent"] = "k3_panel.json is not a mesozoic.k3-panel/v1 document; changes nothing (D-14)"
        return out
    js = ((d.get("spec") or {}).get("joint_sigma") or {})
    want = (C.pin(pins, "sigma_vector_sha256") or {}).get(str(seed))
    out["joint_sigma"] = dict(key=js.get("key"), path_in_file=js.get("path_in_file"), sha256=js.get("sha256"),
                              file_sha256=js.get("file_sha256"), problem=js.get("problem"),
                              equals_pin=None if want is None else js.get("sha256") == want,
                              pin=want, file_pin=C.pin(pins, "sigma_file_sha256"))
    out["session_problem"] = d.get("session_problem")
    rows = []
    if csv_path and Path(csv_path).is_file():
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    out["csv_rows"] = len(rows)
    series = {}
    panels = []
    for p in d.get("panels") or []:
        tests = p.get("tests") or {}
        own, joint = tests.get("own") or {}, tests.get("joint") or {}
        x = dict(panel=p.get("panel"), timesteps=p.get("timesteps"), kind=p.get("kind"), failure=p.get("failure"),
                 own=f"{p.get('own_falls')} of {p.get('own_episodes')}", own_fall_seeds=own.get("fall_seeds"),
                 joint=f"{p.get('joint_falls')} of {p.get('joint_episodes')}", joint_fall_seeds=joint.get("fall_seeds"),
                 joint_sigma_sha256=p.get("joint_sigma_sha256"), policy_state_sha256=p.get("policy_state_sha256"),
                 obs_mean_sha256=p.get("obs_mean_sha256"), obs_var_sha256=p.get("obs_var_sha256"),
                 csv_rows=sum(1 for r in rows if str(r.get("panel")) == str(p.get("panel"))))
        if models and copy_dir and isinstance(p.get("timesteps"), int):
            step = "final" if p.get("kind") == "final" else p["timesteps"]
            sd, om, ov, prob = pair_hashes(Path(models), step, copy_dir)
            x["pair_match"] = dict(state=None if sd is None else sd == p.get("policy_state_sha256"),
                                   obs_mean=None if om is None else om == p.get("obs_mean_sha256"),
                                   obs_var=None if ov is None else ov == p.get("obs_var_sha256"), problem=prob)
        panels.append(x)
        series.setdefault(p.get("joint_sigma_sha256"), []).append(p.get("timesteps"))
    out["panels"] = panels
    out["series_by_sigma_sha256"] = series
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--seed", type=int, required=True, choices=C.SEEDS)
    ap.add_argument("--pins")
    ap.add_argument("--copy")
    ap.add_argument("--models")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    print(C.VERSION)
    pins = C.load_pins(a.pins)
    r = read_panel(a.json, a.csv, a.seed, pins, a.models, a.copy)
    if r.get("absent"):
        print(f"K3 PANEL (report only): {r['absent']}")
    else:
        print(f"K3 PANEL (report only) {a.json} sha256 {r['sha256'][:12]}...: joint sigma {json.dumps(r['joint_sigma'])}")
        for p in r["panels"]:
            print(f"  panel {p['panel']} at {p['timesteps']} ({p['kind']}): own {p['own']} {p['own_fall_seeds']}; joint "
                  f"{p['joint']} {p['joint_fall_seeds']}; sigma {str(p['joint_sigma_sha256'])[:12]}; failure {p['failure']}"
                  + (f"; pair {p['pair_match']}" if 'pair_match' in p else ""))
        print(f"  series by sigma sha256 (never joined): { {str(k)[:12]: v for k, v in r['series_by_sigma_sha256'].items()} }")
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1, default=str) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
