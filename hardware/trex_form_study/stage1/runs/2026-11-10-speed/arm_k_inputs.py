"""Stage 1b arm K's git-ignored inputs (``ARM_K.md``): their list, its check, a checked copy from another checkout,
and their regeneration from the committed records.

The inputs are git-ignored, so a merge does not bring them.  ``arm_k_inputs.json`` lists each with its sha256:

* the refine case's round-two models, ``round2/refine/narrow_knee_60/models/<id>.{xml,json}`` (the plant reads them
  and every record key hashes them, ``panels.key_context``; the XML's sha256 is the candidate record's
  ``xml_sha256``, the model ledger JSON's is the sizing's, recorded only here);
* each K body's reference and transition arrays, ``<body>/refs/*.npz`` and ``<body>/transitions/*.npz`` (their
  sha256 are in ``refs.json`` and ``transitions.json``).

Without the models the plant rebuilds them into ``stage1/models/<body>/`` with another model ledger JSON, so every K
record key would differ and ``eval`` refuses the checkout (``designlab/stage1/__main__.py``).  ``models`` regenerates
them exactly: one ``pipeline.size_candidate`` call per body with the record's final overrides, under the case's
sizing settings (``refine._init``), written to the case's relative path from a scratch working directory.  It is the
last sizing call of ``round2.size_one``; the full ``refine --cases narrow_knee_60`` would also rewrite the committed
records (each holds its run time).  ``arrays`` regenerates missing or wrong arrays with ``refs.prepare`` /
``transitions.prepare`` in a scratch directory and copies in only those that match the record; the committed records
are never rewritten.

Run from hardware/trex_form_study (``run_arm_k.sh`` calls it), with PYTHONPATH=. :

    python stage1/runs/2026-11-10-speed/arm_k_inputs.py check
    python stage1/runs/2026-11-10-speed/arm_k_inputs.py copy-from DIR      # DIR: another checkout (root or study)
    python stage1/runs/2026-11-10-speed/arm_k_inputs.py models
    python stage1/runs/2026-11-10-speed/arm_k_inputs.py arrays BODY refs|transitions
    python stage1/runs/2026-11-10-speed/arm_k_inputs.py manifest           # rewrite arm_k_inputs.json from here
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

RUN = Path(__file__).resolve().parent
STUDY = RUN.parents[2]  # hardware/trex_form_study
MANIFEST = RUN / "arm_k_inputs.json"
K_BODIES = ("60-N-T-10-K", "60-N-H-10-K")
CASE = "narrow_knee_60"
RULE = ("every git-ignored file arm K reads, as a path relative to hardware/trex_form_study, with its sha256: the "
        "refine case's round-two model files (the XML's sha256 is the candidate record's xml_sha256; the model "
        "ledger JSON's is the sizing's and is recorded only here) and each K body's reference and transition arrays "
        "(the sha256 that refs.json and transitions.json record)")


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _registry():
    if str(STUDY) not in sys.path:
        sys.path.insert(0, str(STUDY))
    import designlab  # noqa: F401  (selects the GL backend before mujoco is imported)
    from designlab.stage1 import BODIES, candidate_id

    return BODIES, candidate_id


def model_paths(body: str) -> tuple[str, str]:
    """The case's model files of ``body`` (XML, ledger JSON), relative to hardware/trex_form_study."""
    bodies, candidate_id = _registry()
    d = Path(bodies[body]) / "models"
    cid = candidate_id(body)
    return (d / f"{cid}.xml").as_posix(), (d / f"{cid}.json").as_posix()


def _entries(body: str, kind: str) -> dict:
    doc = json.loads((RUN / body / f"{kind}.json").read_text())
    return doc["references"] if kind == "refs" else doc["transitions"]


def recorded(body: str) -> dict:
    """path -> the sha256 the committed records hold (None for the model ledger JSON, which none holds)."""
    _registry()
    from designlab.stage1 import plant

    xml, ledger = model_paths(body)
    out = {xml: plant.record(body)["xml_sha256"], ledger: None}
    run = RUN.relative_to(STUDY).as_posix()
    for kind in ("refs", "transitions"):
        out.update({f"{run}/{body}/{e['file']}": e["sha256"] for e in _entries(body, kind).values()})
    return out


def manifest() -> int:
    """Write ``arm_k_inputs.json`` from the files here, refusing a file that is missing or differs from its record."""
    files, bad = {}, []
    for body in K_BODIES:
        for path, want in recorded(body).items():
            got = _sha(STUDY / path)
            if got is None or (want is not None and got != want):
                bad.append(f"{path}: {'missing' if got is None else f'sha256 {got[:8]}, record {want[:8]}'}")
            files[path] = got
    if bad:
        print("not written:\n  " + "\n  ".join(bad))
        return 1
    MANIFEST.write_text(json.dumps(dict(rule=RULE, bodies=list(K_BODIES), case=CASE, files=dict(sorted(files.items()))),
                                   indent=1) + "\n")
    print(f"{MANIFEST.relative_to(STUDY)}: {len(files)} files")
    return 0


def listed() -> dict:
    return json.loads(MANIFEST.read_text())["files"]


def check(quiet: bool = False) -> int:
    """Every listed file here with its sha256, and the list agreeing with the records; exit status 1 if not."""
    files, problems = listed(), []
    rec = {}
    for body in K_BODIES:
        rec.update(recorded(body))
    if set(rec) != set(files):
        problems += [f"{p}: in the records, not in {MANIFEST.name}" for p in sorted(set(rec) - set(files))]
        problems += [f"{p}: in {MANIFEST.name}, not in the records" for p in sorted(set(files) - set(rec))]
    problems += [f"{p}: {MANIFEST.name} has {files[p][:8]}, the record {want[:8]}" for p, want in sorted(rec.items())
                 if p in files and want is not None and files[p] != want]
    missing = [p for p in files if not (STUDY / p).exists()]
    wrong = [p for p in files if p not in missing and _sha(STUDY / p) != files[p]]
    problems += [f"{p}: missing" for p in missing] + [f"{p}: sha256 differs" for p in wrong]
    if problems:
        print(f"arm K inputs: {len(problems)} problems (of {len(files)} files)\n  " + "\n  ".join(problems))
        return 1
    if not quiet:
        print(f"arm K inputs: all {len(files)} files here with their recorded sha256")
    return 0


def _put(src: Path, dest: Path, want: str) -> None:
    """Copy ``src`` to ``dest`` through a temporary name and check it (an interrupted copy leaves no partial file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".part-{dest.name}")  # same suffix: git-ignored like the file
    shutil.copy2(src, tmp)
    if _sha(tmp) != want:
        tmp.unlink()
        raise SystemExit(f"{dest}: the copy's sha256 differs from {want[:8]}")
    os.replace(tmp, dest)


def copy_from(src_dir: str) -> int:
    """Copy every listed file that is missing or wrong here from another checkout, each checked by its sha256."""
    src = Path(src_dir).resolve()
    if (src / "hardware" / "trex_form_study").is_dir():
        src = src / "hardware" / "trex_form_study"
    if src == STUDY:
        raise SystemExit(f"copy-from {src_dir}: that is this checkout")
    done, kept, bad = [], 0, []
    for path, want in listed().items():
        if _sha(STUDY / path) == want:
            kept += 1
            continue
        got = _sha(src / path)
        if got != want:
            bad.append(f"{path}: {'missing' if got is None else f'sha256 {got[:8]}, listed {want[:8]}'} in {src}")
            continue
        replaced = (STUDY / path).exists()
        _put(src / path, STUDY / path, want)
        done.append(f"{path}{' (replaced a file with another sha256)' if replaced else ''}")
    print(f"copied {len(done)}, already here {kept}" + "".join(f"\n  {d}" for d in done))
    if bad:
        print(f"not copied ({len(bad)}):\n  " + "\n  ".join(bad))
        return 1
    return check()


def _size(job: tuple[str, str]) -> None:
    """One body's model files, sized as ``round2.size_one``'s last call sizes them, into ``work``/<case path>."""
    body, work = job
    os.chdir(work)  # the ledger JSON records the XML's path as given: the case's relative one
    _registry()
    from designlab import gait, pipeline, refine, round2
    from designlab.stage1 import plant

    refine._init(CASE)
    gait.set_accuracy(round2.GAIT_ACCURACY)
    pipeline.BUS_CONSISTENT_OSCILLATION = True
    xml, _ = model_paths(body)
    rec = plant.record(body)
    cand = round2.candidate(rec["id"], tuple(tuple(o) for o in rec["overrides"]))
    pipeline.size_candidate(cand, Path(xml).parent, selection=round2.start_selection(cand.height))


def models() -> int:
    """Regenerate the case's model files of each K body that lacks one or holds a wrong one, both bodies at once; a
    body's files are put in place only when both match the list."""
    import multiprocessing as mp

    files = listed()
    todo = [b for b in K_BODIES if any(_sha(STUDY / p) != files[p] for p in model_paths(b))]
    if not todo:
        print("models: all here with their listed sha256")
        return 0
    with tempfile.TemporaryDirectory(prefix="arm_k_models_") as tmp:
        jobs = [(b, str(Path(tmp) / b)) for b in todo]
        for _, w in jobs:
            Path(w).mkdir()
        print(f"models: sizing {', '.join(todo)} (one process each, about 3 min)", flush=True)
        with mp.get_context("spawn").Pool(len(jobs)) as pool:
            pool.map(_size, jobs, chunksize=1)
        status = 0
        for b, w in jobs:
            paths = model_paths(b)
            got = {p: _sha(Path(w) / p) for p in paths}
            if any(got[p] != files[p] for p in paths):
                status = 1
                print(f"models: {b} regenerated with other sha256 (" + ", ".join(
                    f"{Path(p).name} {(got[p] or 'missing')[:8]} vs {files[p][:8]}" for p in paths) + "); not put in place")
                continue
            put = [p for p in paths if _sha(STUDY / p) != files[p]]
            replaced = [Path(p).name for p in put if (STUDY / p).exists()]
            for p in put:
                _put(Path(w) / p, STUDY / p, files[p])
            print(f"models: {b} regenerated, matching {MANIFEST.name}" + (
                f" (replaced {', '.join(replaced)}, which had another sha256)" if replaced else ""))
    return status


def arrays(body: str, kind: str) -> int:
    """Regenerate ``body``'s missing or wrong arrays of ``kind`` (refs, transitions) in a scratch directory; put each
    in place only if it matches its record (the record stays as committed)."""
    if body not in K_BODIES or kind not in ("refs", "transitions"):
        raise SystemExit(f"arrays {body} {kind}: a body of {', '.join(K_BODIES)} and refs or transitions")
    entries = _entries(body, kind)
    bad = {n: e for n, e in entries.items() if _sha(RUN / body / e["file"]) != e["sha256"]}
    if not bad:
        print(f"{body} {kind}: all {len(entries)} arrays here with their recorded sha256")
        return 0
    if kind == "transitions" and arrays(body, "refs"):  # the transitions are made from the saved references
        return 1
    _registry()
    from designlab.stage1 import refs, transitions

    print(f"{body} {kind}: regenerating {len(bad)} of {len(entries)} ({', '.join(sorted(bad))})", flush=True)
    with tempfile.TemporaryDirectory(prefix=f"arm_k_{kind}_") as tmp:
        out = Path(tmp) / body
        out.mkdir()
        if kind == "refs":
            refs.prepare(body, out, sorted(bad))
        else:
            (out / "refs").symlink_to((RUN / body / "refs").resolve())
            transitions.prepare(body, out, sorted(bad))
        got = {n: _sha(out / e["file"]) for n, e in bad.items()}
        differ = {n: s for n, s in got.items() if s != bad[n]["sha256"]}
        if differ:
            print(f"{body} {kind}: regenerated with other sha256, nothing put in place: " + ", ".join(
                f"{n} {(s or 'missing')[:8]} vs {bad[n]['sha256'][:8]}" for n, s in sorted(differ.items())))
            return 1
        replaced = sorted(n for n, e in bad.items() if (RUN / body / e["file"]).exists())
        for e in bad.values():
            _put(out / e["file"], RUN / body / e["file"], e["sha256"])
    print(f"{body} {kind}: {len(bad)} arrays regenerated, matching their records" + (
        f" (replaced {', '.join(replaced)}, which had another sha256)" if replaced else ""))
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else ""
    if cmd == "check" and len(argv) == 1:
        return check()
    if cmd == "copy-from" and len(argv) == 2:
        return copy_from(argv[1])
    if cmd == "models" and len(argv) == 1:
        return models()
    if cmd == "arrays" and len(argv) == 3:
        return arrays(argv[1], argv[2])
    if cmd == "manifest" and len(argv) == 1:
        return manifest()
    print(__doc__.split("\n\n")[-1])
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
