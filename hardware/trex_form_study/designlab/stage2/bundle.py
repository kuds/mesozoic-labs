"""Track B's Drive bundle: the git-ignored inputs a Colab clone lacks, packed into a few deterministic ``.tar`` parts
with a committed JSON manifest, and their verification and unpacking (Track B rules N3 and N4; ``STAGE2_PLAN.md``
section 4 B.4).  Simulation inputs only.  This module opens no network connection and uploads nothing: the parts are
files on local disk, and whoever carries them elsewhere writes the receipts in another file (``UPLOADS.json``).

**What is packed**, read from committed records only (``build``; ``RULE``):

* per body the two round-two model files ``plant.model_files(body)`` names, ``<case>/models/<id>.xml`` and
  ``<id>.json`` (``model_paths``).  The XML's sha256 is the candidate record's ``xml_sha256`` (checked against the
  plan's prefix, ``designlab.stage1.SHA256_PREFIX``); the model ledger JSON's is the one a committed inputs list holds
  (``INPUT_LISTS``: the lists ``tools/restore_inputs.py`` restores from).  A file that is missing is refused: nothing
  is rebuilt here (``plant.model_files`` would rebuild it into ``stage1/models/<body>/`` with another ledger JSON);
* per set and body the files of the set's latest freeze (``set.json`` ``freezes[-1].manifest``, through
  ``setio.frozen``): a walk's ``refs/<name>.npz``, a move's ``transitions/<name>.npz``, each with the freeze's
  sha256.  Never a look-format twin (``look/<name>.npz``), never a file the freeze does not list.  ``sets=()`` packs
  the models only.

Every file must be on disk with its recorded sha256, and every record read must be committed and unchanged in the
tree it is read from (where that tree is a git checkout); ``build`` lists every difference and writes nothing.

**Parts** (``<out>/parts/``): ``models-<body>.tar`` and ``<set>-<body>-<i>.tar`` (i from 1; a set's files in sorted
order, a new part when the next file would take the part past ``part_mb``; a "MB" is 1,000,000 bytes, ``MB``).
**Tar rules**, so that the same files give the same bytes in any process and under any umask: USTAR
(``tarfile.USTAR_FORMAT``), members in sorted order, each named by its path relative to the study, mtime 0, uid and
gid 0, empty user and group names, mode 0644, no compression, the file's bytes streamed.

**MANIFEST.json** (``<out>/MANIFEST.json``, schema ``s2-bundle-2``; the one file of a bundle that is committed)::

    {schema,
     content: {bundle_id, bodies [...], models {body: {case, xml {path, sha256}, record {path, sha256}}},
               sets {set: {body: {freeze_index, manifest_sha256, files: n}}},
               files {<path relative to the study>: {sha256, bytes, part}},
               parts {<name>.tar: {sha256, bytes, files: n}}},
     content_sha256,
     provenance: {made_at, commit, dirty, dirty_paths, rule, tool_sha256, versions, ...}}

``content_sha256`` is ``s2io.sha256_json(content)`` (canonical JSON: sorted keys, no spaces), so it does not depend on
how the file is laid out and nothing outside ``content`` enters it.  Nothing is ever added to ``content`` after the
build.  ``build`` leaves an existing manifest with the same content as it is and refuses one with another content: a
bundle id names one content.  (``models.<body>.record`` is the model ledger JSON beside the XML, not the candidate
record: that one is committed.)

**verify** re-hashes every file of the manifest under a root (and, given a parts directory, every part) and never
raises on a difference: it returns the counts and the problems, each starting with its path.  **unpack** refuses,
before it writes anything, a part whose size or sha256 differs, a member that is not one of that part's files in
``content.files``, a member that is no regular file (a link, a directory), a path that leaves the root (absolute,
with ``..``, or through a symbolic link on the way) and a member whose bytes are not the manifest's; then it writes
each file through a temporary name (``<name>.tmp`` beside it, ``s2io``'s convention).  A file already there with the
right sha256 is left untouched, so a second ``unpack`` writes nothing.  A record, a manifest, a part or a place that
cannot be read or is not what it must be is a refusal (``BundleError``; for ``verify`` a reported problem) with its
path, never another exception.

A relative path given to any function or option is taken from the study directory (``hardware/trex_form_study``),
never from the working directory (``setio``'s rule).

This module is in no record key.  It imports ``setio`` (and with it Stage 1's controller and MuJoCo) only inside
``build`` and only when a set is packed: ``verify`` and ``unpack`` need the standard library, numpy and ``s2io``.

    python -m designlab.stage2.bundle build --bundle-id ID --bodies B1,B2 --sets V6 --out stage2/tb/bundle/ID [--part-mb 20]
    python -m designlab.stage2.bundle verify --manifest PATH --root DIR [--parts DIR]
    python -m designlab.stage2.bundle unpack --manifest PATH --parts DIR --root DIR [--only PART,...]
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":                       # before any numpy import (round A's module skeleton)
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, "1")

import argparse
import hashlib
import json
import platform
import re
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

import numpy as np

from designlab import stage1
from designlab.stage2 import s2io

SCHEMA = "s2-bundle-2"
MANIFEST = "MANIFEST.json"
PARTS = "parts"
#: Bytes in one "MB" of ``part_mb``: the decimal reading, the smaller of the two (a part within 20 x 1,000,000 bytes
#: is within 20 MiB too).  The limit is on the tar file's size, headers and padding included (``tar_bytes``).
MB = 1_000_000
#: The committed lists of git-ignored inputs with their sha256 (``{"files": {path: sha256}}``), the ones
#: ``tools/restore_inputs.py`` restores from (its groups picks-models, k, x and j).  A model ledger JSON's sha256 is
#: recorded in these and, for the bodies after the Stage 1 picks, nowhere else (each list's own rule text).
INPUT_LISTS = ("stage1/runs/2026-11-08/stage1_inputs.json", "stage1/runs/2026-11-10-speed/arm_k_inputs.json",
               "stage2/x/x_inputs.json", "stage2/j/j_inputs.json")
#: Where the reference sets are, relative to the study (``setio.ROOT``; a test pins the two together).
SETS_DIR = "stage2/sets"
RULE = (
    "the git-ignored inputs a clone lacks, from committed records only.  Per body: the two round-two model files "
    "plant.model_files names, <case>/models/<id>.xml with the candidate record's xml_sha256 (which must carry the "
    "plan's prefix, designlab.stage1.SHA256_PREFIX) and <id>.json with the sha256 a committed inputs list holds "
    "(bundle.INPUT_LISTS); where no committed record holds a file's sha256 the manifest holds the file's own and "
    "provenance.hash_sources lists none for it.  Per set and body: the files of the set's latest freeze manifest "
    "(set.json freezes[-1].manifest: refs/<name>.npz for a walk, transitions/<name>.npz for a move, with the "
    "freeze's sha256, which the entry in refs.json or transitions.json must hold too); never a look-format twin, "
    "never a file the freeze does not list.  A file that is missing or has another sha256 is refused; nothing is "
    "rebuilt.  Parts: models-<body>.tar; <set>-<body>-<i>.tar, i from 1, the set's files in sorted order, a new part "
    "when the next file would take the tar file past part_mb x 1,000,000 bytes.  Tar rules: USTAR, members sorted "
    "by name, each named by its path relative to the study, mtime 0, uid and gid 0, empty user and group names, mode "
    "0644, no compression.  content_sha256 is s2io.sha256_json(content): the content block alone; provenance is "
    "outside it and nothing is added to content after the build")

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK = 1 << 20


class BundleError(s2io.S2Error):
    """A bundle cannot be built or unpacked: a record, a file, a part or a member is not what it must be."""


# ------------------------------------------------------------------------------------------------
# Paths, hashes and tar sizes
# ------------------------------------------------------------------------------------------------
def _anchored(path) -> Path:
    """A relative path is relative to the study directory, never to the working directory."""
    p = Path(path)
    return p if p.is_absolute() else s2io.STUDY / p


def _hash(path) -> tuple[str, int]:
    """(sha256, bytes) of a file, read once."""
    h, n = hashlib.sha256(), 0
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(_BLOCK), b""):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def _unlink(path) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def path_problem(rel) -> str | None:
    """Why ``rel`` is no path a bundle may hold, as the end of a sentence ("is absolute"), or None: a bundle's path is
    a non-empty string of ``/``-separated names relative to the study, in its plain form, with no ``..``."""
    if not isinstance(rel, str) or not rel:
        return "is not a path"
    if "\x00" in rel or "\\" in rel:
        return "holds a NUL or a backslash"
    if rel.startswith("/"):
        return "is absolute"
    names = rel.split("/")
    if ".." in names:
        return "holds '..'"
    if "" in names or "." in names:
        return "is not in its plain form"
    return None


def tar_bytes(sizes) -> int:
    """The size of a part that holds files of these sizes, as ``tarfile`` writes it under the tar rules: one 512-byte
    header per member, its bytes padded to 512, two empty blocks at the end, the whole padded to 10,240."""
    blk, rec = tarfile.BLOCKSIZE, tarfile.RECORDSIZE
    n = sum(blk + -(-int(s) // blk) * blk for s in sizes) + 2 * blk
    return -(-n // rec) * rec


def model_paths(body: str) -> tuple[str, str]:
    """(MJCF, model ledger JSON) of a body as paths relative to the study: the files ``plant.model_files(body)``
    returns when both are there.  Nothing is looked up on disk and nothing is built."""
    if body not in stage1.BODIES:
        raise BundleError(f"{body!r} is not a Stage 1 body (choose from {', '.join(stage1.BODIES)})")
    d = PurePosixPath(stage1.BODIES[body]) / "models"
    cid = stage1.candidate_id(body)
    return (d / f"{cid}.xml").as_posix(), (d / f"{cid}.json").as_posix()


def _read_record(root: Path, rel: str, records: dict):
    """A committed JSON record under ``root``; its sha256 goes to ``records[rel]``."""
    try:
        raw = (root / rel).read_bytes()
    except OSError as exc:
        raise BundleError(f"{rel}: the record cannot be read ({exc.strerror or exc})") from None
    try:
        doc = json.loads(raw)
    except (ValueError, RecursionError) as exc:  # RecursionError: nested deeper than the parser goes
        raise BundleError(f"{rel}: the record is not JSON ({exc})") from None
    records[rel] = hashlib.sha256(raw).hexdigest()
    return doc


# ------------------------------------------------------------------------------------------------
# What a build packs
# ------------------------------------------------------------------------------------------------
def _input_lists(root: Path) -> dict:
    """{list: ({path: sha256}, the list's own sha256)} of every ``INPUT_LISTS`` file that is under ``root``."""
    out = {}
    for rel in INPUT_LISTS:
        if (root / rel).is_file():
            sha: dict = {}
            doc = _read_record(root, rel, sha)
            files = doc.get("files") if isinstance(doc, dict) else None
            if not isinstance(files, dict):
                raise BundleError(f"{rel}: no 'files' table")
            out[rel] = (files, sha[rel])
    return out


def _model(body: str, root: Path, lists: dict, records: dict, sources: dict, bad: list) -> tuple[dict, list]:
    """A body's ``content.models`` entry and its two files as ``[(path, sha256, bytes)]``; differences go to
    ``bad``, the records read to ``records`` and, per file, the records that hold its sha256 to ``sources``."""
    xml, ledger = model_paths(body)
    case_rel = stage1.BODIES[body]
    cand = f"{case_rel}/candidates/{stage1.candidate_id(body)}.json"
    doc = _read_record(root, cand, records)
    want = doc.get("xml_sha256") if isinstance(doc, dict) else None
    held: dict = {xml: {}, ledger: {}}
    if not (isinstance(want, str) and _HEX64.match(want)):
        bad.append(f"{cand}: no xml_sha256")
    else:
        prefix = stage1.SHA256_PREFIX.get(body)
        if not prefix or not want.startswith(prefix):
            bad.append(f"{cand}: xml_sha256 {want[:12]}... lacks the plan's prefix {prefix} for {body} "
                       "(designlab.stage1.SHA256_PREFIX)")
        held[xml][f"{cand}: xml_sha256"] = want
    for rel, (files, sha) in lists.items():
        for path in (xml, ledger):
            if path in files:
                records[rel] = sha
                if isinstance(files[path], str) and _HEX64.match(files[path]):
                    held[path][f"{rel}: files"] = files[path]
                else:
                    bad.append(f"{rel}: its entry for {path} is no sha256")
    out, packed = dict(case=PurePosixPath(case_rel).name), []
    for key, path in (("xml", xml), ("record", ledger)):
        shas = sorted(set(held[path].values()), key=str)
        sources[path] = sorted(held[path])
        if len(shas) > 1:
            bad.append(f"{path}: the committed records disagree ("
                       + "; ".join(f"{k} {str(v)[:12]}..." for k, v in sorted(held[path].items())) + ")")
            continue
        if not (root / path).is_file():
            bad.append(f"{path}: missing (a model file of {body}; nothing is rebuilt here: bring it in with "
                       "python tools/restore_inputs.py copy-from DIR or restore)")
            continue
        got, n = _hash(root / path)
        if shas and got != shas[0]:
            bad.append(f"{path}: sha256 {got[:12]}..., {sources[path][0]} holds {str(shas[0])[:12]}...")
            continue
        out[key] = dict(path=path, sha256=got)
        packed.append((path, got, n))
    return out, packed


def _set(set_name: str, body: str, root: Path, records: dict, sources: dict, bad: list) -> tuple[dict, list]:
    """A set and body's ``content.sets`` entry and the files of its latest freeze as ``[(path, sha256, bytes)]``."""
    from designlab.stage1 import tune
    from designlab.stage2 import setio

    rel_dir = f"{SETS_DIR}/{set_name}/{body}"
    names = ("set.json", "refs.json", "transitions.json")
    try:
        docs = setio.load(root / rel_dir)
    except tune.InputError as exc:
        bad.append(f"{rel_dir}: {str(exc).split(': ', 1)[-1]} (set {set_name} of {body})")
        return {}, []
    except (OSError, ValueError, RecursionError) as exc:  # one of the set's records cannot be read as JSON: which
        found = []
        for name in (*names, "screen.json"):
            try:
                if os.path.lexists(root / rel_dir / name):
                    _read_record(root, f"{rel_dir}/{name}", {})
            except BundleError as why:
                found.append(str(why))
        bad.extend(found or [f"{rel_dir}: the set's records cannot be read ({exc})"])
        return {}, []
    shapes = [f"{rel_dir}/{name}: the record is not a JSON object" for name in names
              if not isinstance(docs[name], dict)]
    if shapes:
        bad.extend(shapes)
        return {}, []
    for name in names:
        records[f"{rel_dir}/{name}"] = s2io.sha256_file(root / rel_dir / name)
    try:
        fz = setio.frozen(docs["set.json"])
    except (KeyError, IndexError, TypeError, AttributeError) as exc:
        bad.append(f"{rel_dir}/set.json: its latest freeze is not {{index, manifest, manifest_sha256}} "
                   f"({type(exc).__name__}: {exc})")
        return {}, []
    if fz is None:
        bad.append(f"{rel_dir}/set.json: the set is not frozen (no freeze): a bundle packs a freeze's files only")
        return {}, []
    entries = fz["entries"]
    if not isinstance(entries, dict):
        bad.append(f"{rel_dir}/set.json: freeze {fz['index']}'s manifest is not a table of entries")
        return {}, []
    if s2io.sha256_json(entries) != fz["manifest_sha256"]:
        bad.append(f"{rel_dir}/set.json: freeze {fz['index']}'s manifest_sha256 is not its manifest's")
        return {}, []
    packed = []
    for name in sorted(entries):
        row = entries[name] if isinstance(entries[name], dict) else {}
        kind = row.get("kind")
        if kind in setio.WALK_KINDS:
            record, key, sub = "refs.json", "references", "refs"
        elif kind in setio.MOVE_KINDS:
            record, key, sub = "transitions.json", "transitions", "transitions"
        else:
            bad.append(f"{rel_dir}/set.json: freeze {fz['index']} lists {name!r} with the kind {kind!r}")
            continue
        path, want = f"{rel_dir}/{sub}/{name}.npz", row.get("sha256")
        why = path_problem(path) or (None if Path(name).name == name else "is no file name")
        if why:
            bad.append(f"{rel_dir}/set.json: the frozen name {name!r} {why}")
            continue
        table = docs[record].get(key)
        e = table.get(name) if isinstance(table, dict) else None
        if not isinstance(e, dict):
            bad.append(f"{path}: in freeze {fz['index']}, and {record} has no entry {name!r}")
            continue
        if e.get("file") != f"{sub}/{name}.npz" or e.get("available") is False or e.get("sha256") != want:
            bad.append(f"{path}: freeze {fz['index']} holds sha256 {str(want)[:12]}..., {record} holds the file "
                       f"{e.get('file')!r} with {str(e.get('sha256'))[:12]}... (available: {e.get('available')})")
            continue
        sources[path] = [f"{rel_dir}/set.json: freezes[{fz['index']}].manifest", f"{rel_dir}/{record}: {key}"]
        if not (root / path).is_file():
            bad.append(f"{path}: missing (in freeze {fz['index']} of {set_name} {body}; nothing is rebuilt here: "
                       "python tools/restore_inputs.py restore --group sets)")
            continue
        got, n = _hash(root / path)
        if got != want:
            bad.append(f"{path}: sha256 {got[:12]}..., freeze {fz['index']} holds {str(want)[:12]}...")
            continue
        packed.append((path, got, n))
    return dict(freeze_index=fz["index"], manifest_sha256=fz["manifest_sha256"], files=len(entries)), packed


def _split(stem: str, packed: list, limit: int, bad: list) -> list:
    """``[(part name, [(path, sha256, bytes)])]`` for a set's files: sorted, a new part when the next file would take
    the tar past ``limit`` bytes; a file that alone is over the limit goes to ``bad``."""
    groups, cur = [], []
    for item in sorted(packed):
        if tar_bytes([item[2]]) > limit:
            bad.append(f"{item[0]}: {item[2]} bytes, a part of its own would be {tar_bytes([item[2]])} bytes, over "
                       f"the limit of {limit}")
            continue
        if cur and tar_bytes([x[2] for x in cur] + [item[2]]) > limit:
            groups.append(cur)
            cur = []
        cur.append(item)
    if cur:
        groups.append(cur)
    return [(f"{stem}-{i}.tar", g) for i, g in enumerate(groups, start=1)]


def _uncommitted(root: Path, rels) -> list[str] | None:
    """Which of the records ``rels`` (relative to ``root``) are not committed as they are on disk, each with its
    reason; None where ``root`` is in no git work tree (or git is missing): nothing can be told there."""
    def git(*args):
        return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)

    rels = sorted(rels)
    try:
        if git("rev-parse", "--is-inside-work-tree").stdout.strip() != "true":
            return None
        tracked = git("ls-files", "-z", "--", *rels)
        changed = git("diff", "--name-only", "-z", "--relative", "HEAD", "--", *rels)
    except OSError:
        return None
    if tracked.returncode != 0:
        return None
    have = set(tracked.stdout.split("\0"))
    out = [f"{r}: not tracked by git" for r in rels if r not in have]
    if changed.returncode != 0:
        return out + ["no commit to compare with (git diff HEAD failed)"]
    diff = set(changed.stdout.split("\0"))
    return out + [f"{r}: differs from the commit" for r in rels if r in diff and r in have]


# ------------------------------------------------------------------------------------------------
# Parts
# ------------------------------------------------------------------------------------------------
def _write_part(path: Path, root: Path, members: list) -> None:
    """One part under the tar rules: ``members`` (``[(path relative to the study, sha256, bytes)]``) in sorted order,
    the bytes streamed from ``root``.  Written in place: the caller gives a temporary name."""
    try:
        with open(path, "wb") as fh:
            with tarfile.open(fileobj=fh, mode="w", format=tarfile.USTAR_FORMAT) as tar:
                for rel, _, size in sorted(members):
                    info = tarfile.TarInfo(rel)
                    info.size, info.mtime, info.mode, info.type = size, 0, 0o644, tarfile.REGTYPE
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    with open(root / rel, "rb") as src:
                        tar.addfile(info, src)
            fh.flush()
            os.fsync(fh.fileno())
    except (ValueError, tarfile.TarError) as exc:  # a name USTAR cannot hold
        raise BundleError(f"{path.name}: cannot be written as USTAR ({exc})") from None
    except OSError as exc:  # a file that shrank or went away since it was hashed; a full disk
        raise BundleError(f"{path.name}: not written ({exc}): a file changed while it was packed, or the part "
                          "could not be stored") from None


def scan_part(path, expect: dict) -> None:
    """Read a part to its end and compare it with ``expect`` (``{member name: {sha256, bytes, ...}}``: the files the
    manifest gives this part).  ``BundleError``, naming the part and the member, for a member that is no regular
    file, whose name is no bundle path (it would leave the root), that is not in ``expect``, that appears twice or
    whose size or sha256 differs, and for a file of ``expect`` that is not in the part.  Nothing is written."""
    part, seen = Path(path).name, set()
    try:
        with tarfile.open(path, mode="r:") as tar:
            for info in tar:
                m = info.name
                if info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                    kind = ("a symbolic link" if info.issym() else "a hard link" if info.islnk()
                            else "a directory" if info.isdir() else "no regular file")
                    raise BundleError(f"{part}: member {m!r} is {kind}: a bundle holds regular files only")
                why = path_problem(m)
                if why:
                    raise BundleError(f"{part}: member {m!r} {why}: it would leave the root")
                if m not in expect:
                    raise BundleError(f"{part}: member {m!r} is not one of this part's files in the manifest "
                                      "(content.files)")
                if m in seen:
                    raise BundleError(f"{part}: member {m!r} appears twice")
                seen.add(m)
                h, src = hashlib.sha256(), tar.extractfile(info)
                for block in iter(lambda: src.read(_BLOCK), b""):
                    h.update(block)
                if info.size != expect[m]["bytes"] or h.hexdigest() != expect[m]["sha256"]:
                    raise BundleError(f"{part}: member {m!r} has {info.size} bytes and sha256 {h.hexdigest()[:12]}"
                                      f"..., the manifest holds {expect[m]['bytes']} and "
                                      f"{str(expect[m]['sha256'])[:12]}...")
    except (tarfile.TarError, EOFError, OSError) as exc:
        raise BundleError(f"{part}: not a readable plain tar ({exc})") from None
    missing = sorted(set(expect) - seen)
    if missing:
        raise BundleError(f"{part}: {len(missing)} of its {len(expect)} files are not in it (first: {missing[0]})")


# ------------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------------
def _names(what: str, values) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise BundleError(f"{what}: give a list of names, not {values!r}")
    out = [str(v) for v in values]
    for v in out:
        if not _NAME.match(v):
            raise BundleError(f"{what}: {v!r} is not a name (letters, digits, '.', '_' and '-')")
    if len(set(out)) != len(out):
        raise BundleError(f"{what}: a name is given twice ({', '.join(out)})")
    return sorted(out)


def _build(bundle_id, bodies, sets, out, part_mb, root) -> tuple[dict, dict]:
    """``build``'s work: (the manifest as it stands on disk, ``{out, manifest, parts_dir, written, notes}``)."""
    if not (isinstance(bundle_id, str) and _NAME.match(bundle_id)):
        raise BundleError(f"bundle id {bundle_id!r} is not a name (letters, digits, '.', '_' and '-')")
    bodies, sets = _names("bodies", bodies), _names("sets", sets)
    if not bodies:
        raise BundleError("no body given")
    for body in bodies:
        model_paths(body)  # refuses an unregistered body
    if isinstance(part_mb, bool) or not isinstance(part_mb, (int, float)) or not 0 < part_mb < float("inf"):
        raise BundleError(f"part_mb must be a positive number, not {part_mb!r}")
    limit = int(part_mb * MB)
    root = _anchored(s2io.STUDY if root is None else root)
    out_dir = _anchored(Path("stage2") / "tb" / "bundle" / bundle_id if out is None else out)

    # what to pack: read-only; every difference is collected
    bad: list[str] = []
    records: dict = {}
    sources: dict = {}
    lists = _input_lists(root)
    models, set_docs, groups = {}, {}, []
    for body in bodies:
        models[body], packed = _model(body, root, lists, records, sources, bad)
        name = f"models-{body}.tar"
        size = tar_bytes([x[2] for x in packed])
        if size > limit:
            bad.append(f"{name}: {size} bytes, over the limit of {limit} (part_mb {part_mb:g}); a body's two model "
                       "files are one part")
        groups.append((name, packed))
    for set_name in sets:
        for body in bodies:
            entry, packed = _set(set_name, body, root, records, sources, bad)
            set_docs.setdefault(set_name, {})[body] = entry
            groups += _split(f"{set_name}-{body}", packed, limit, bad)
    files: dict = {}
    part_names = set()
    for name, packed in groups:
        if name in part_names:
            bad.append(f"{name}: two parts would carry this name")
        part_names.add(name)
        for rel, sha, n in packed:
            if rel in files:
                bad.append(f"{rel}: listed twice (in {files[rel]['part']} and {name})")
            files[rel] = dict(sha256=sha, bytes=n, part=name)
    committed = None
    if not bad:
        loose = _uncommitted(root, records)
        committed = None if loose is None else not loose
        bad += [f"{x} (a bundle is built from committed records only: commit it first)" for x in loose or ()]
    if bad:
        raise BundleError(f"bundle {bundle_id} is not built ({_count(len(bad), 'difference')}): " + "; ".join(bad))

    parts_dir, target = out_dir / PARTS, out_dir / MANIFEST
    parts_dir.mkdir(parents=True, exist_ok=True)
    tmps = {name: parts_dir / f"{name}.tmp" for name, _ in groups}
    notes = [f"{rel}: no committed record holds its sha256; the manifest holds the file's own"
             for rel in sorted(files) if not sources.get(rel)]
    if committed is None:
        notes.append(f"{root} is no git checkout (or git could not tell): the records read were not compared with a "
                     "commit (provenance.records_committed is null)")
    with s2io.Lock(out_dir / ".lock"):
        try:
            parts = {}
            for name, packed in groups:
                _write_part(tmps[name], root, packed)
                scan_part(tmps[name], {rel: files[rel] for rel, _, _ in packed})  # a file that changed meanwhile
                sha, n = _hash(tmps[name])
                if n > limit:
                    raise BundleError(f"{name}: {n} bytes as written, over the limit of {limit}")
                parts[name] = dict(sha256=sha, bytes=n, files=len(packed))
            content = dict(bundle_id=bundle_id, bodies=bodies, models=models, sets=set_docs,
                           files=dict(sorted(files.items())), parts=dict(sorted(parts.items())))
            digest = s2io.sha256_json(content)
            old = None
            if target.exists():
                try:
                    old = s2io.read_json(target)
                    old_digest = content_sha256(old)
                    old_id = old["content"].get("bundle_id")
                except (OSError, ValueError, KeyError, TypeError, AttributeError, RecursionError) as exc:
                    raise BundleError(f"{target} is there and cannot be read as a manifest ({type(exc).__name__}: "
                                      f"{exc}): nothing is replaced; remove it or give another --out") from None
                if old_digest != digest:
                    raise BundleError(f"{target} holds bundle {old_id!r} with the content "
                                      f"hash {old_digest[:12]}..., this build gives {digest[:12]}...: a bundle id "
                                      "names one content; nothing is replaced (take another bundle id, or remove "
                                      "the old bundle's directory if it was never quoted)")
            stale = sorted(p.name for p in parts_dir.glob("*.tar") if p.name not in parts)
            if stale:
                raise BundleError(f"{parts_dir} holds parts that are not this bundle's ({', '.join(stale)}): "
                                  "nothing is replaced; remove them first")
            for name in parts:
                os.replace(tmps[name], parts_dir / name)
            if old is None:
                prov = dict(made_at=s2io.now_utc(), **s2io.git_state(own_outputs=[str(out_dir)]), rule=RULE,
                            tool_sha256=s2io.sha256_file(__file__),
                            versions=dict(python=platform.python_version(), numpy=np.__version__),
                            root=s2io.portable(str(root)) if root != s2io.STUDY else ".", part_mb=part_mb,
                            part_limit_bytes=limit, records=dict(sorted(records.items())),
                            records_committed=committed, hash_sources=dict(sorted(sources.items())))
                old = dict(schema=SCHEMA, content=content, content_sha256=digest, provenance=prov)
                s2io.write_json(target, old)
                written = True
            else:
                written = False
        finally:
            for tmp in tmps.values():
                _unlink(tmp)
    return s2io.read_json(target), dict(out=str(out_dir), manifest=str(target), parts_dir=str(parts_dir),
                                        written=written, notes=notes)


def build(bundle_id, bodies, sets=("V6",), out=None, part_mb=20, root=None) -> dict:
    """Pack the bodies' model files and the sets' frozen arrays (``RULE``) into ``<out>/parts/*.tar`` and write
    ``<out>/MANIFEST.json``; the manifest as it stands on disk is returned.

    ``bodies``: Stage 1 body names; ``sets``: reference-set names, each packed for every body (``()``: the models
    only); ``out``: default ``stage2/tb/bundle/<bundle_id>``; ``part_mb``: a part's largest size, in units of
    1,000,000 bytes; ``root``: the tree the records and files are read from (default: the study directory).

    ``BundleError`` lists every difference and nothing is written: a missing record, a set that is not frozen, a
    file that is missing or has another sha256 than its record's, records that disagree, a record that is not
    committed, a file too large for a part.  Each part is written under a temporary name, read back member by member
    against the manifest and moved into place; the manifest is written last.  An existing manifest with the same
    content is left as it is (its provenance is the first build's); one with another content, or a part in
    ``<out>/parts`` that is not this bundle's, is refused with nothing replaced."""
    return _build(bundle_id, bodies, sets, out, part_mb, root)[0]


# ------------------------------------------------------------------------------------------------
# The manifest
# ------------------------------------------------------------------------------------------------
def content_sha256(manifest: dict) -> str:
    """``s2io.sha256_json`` of the manifest's content block alone: what line 2, the steps sheet and
    ``ACCEPTANCE.json`` quote.  It is recomputed, never read from the ``content_sha256`` key."""
    return s2io.sha256_json(manifest["content"])


def load(manifest) -> dict:
    """The manifest document: ``manifest`` itself when it is one, else the JSON file it names.  ``BundleError`` when
    the file cannot be read or is not a JSON object."""
    if isinstance(manifest, dict):
        return manifest
    if not isinstance(manifest, (str, os.PathLike)):
        raise BundleError(f"a manifest is a document or the path of one, not {type(manifest).__name__}")
    path = _anchored(manifest)
    try:
        doc = json.loads(path.read_bytes())
    except OSError as exc:
        raise BundleError(f"{path}: the manifest cannot be read ({exc.strerror or exc})") from None
    except (ValueError, RecursionError) as exc:  # RecursionError: nested deeper than the parser goes
        raise BundleError(f"{path}: the manifest is not JSON ({exc})") from None
    if not isinstance(doc, dict):
        raise BundleError(f"{path}: the manifest is not a JSON object")
    return doc


def _is_count(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def manifest_problems(doc) -> list[str]:
    """Every way ``doc`` is not a manifest whose files and parts agree (empty: it is one): the schema, the content
    block and its hash, each file's path, sha256, size and part, each part's name, sha256, size and file count."""
    if not isinstance(doc, dict):
        return ["MANIFEST: not a JSON object"]
    out = []
    if doc.get("schema") != SCHEMA:
        out.append(f"MANIFEST: schema {doc.get('schema')!r}, not {SCHEMA!r}")
    content = doc.get("content")
    if not isinstance(content, dict):
        return out + ["MANIFEST: no content block"]
    files, parts = content.get("files"), content.get("parts")
    if not isinstance(files, dict) or not isinstance(parts, dict):
        return out + ["MANIFEST: content holds no files or no parts table"]
    if not isinstance(content.get("bundle_id"), str):
        out.append("MANIFEST: content holds no bundle_id")
    try:
        digest = s2io.sha256_json(content)
    except (TypeError, ValueError, RecursionError) as exc:  # no JSON value in it, or nested too deep
        return out + [f"MANIFEST: the content block cannot be hashed ({exc})"]
    if doc.get("content_sha256") != digest:
        out.append(f"MANIFEST: content_sha256 is {str(doc.get('content_sha256'))[:12]}..., the content block's hash "
                   f"is {digest[:12]}...")
    count = dict.fromkeys(parts, 0)
    for rel in sorted(files, key=str):
        e, why = files[rel], path_problem(rel)
        if why:
            out.append(f"{rel!r}: the path {why}")
        elif not (isinstance(e, dict) and isinstance(e.get("sha256"), str) and _HEX64.match(e["sha256"])
                  and _is_count(e.get("bytes")) and isinstance(e.get("part"), str)):
            out.append(f"{rel}: its entry is not {{sha256, bytes, part}}")
        elif e["part"] not in parts:
            out.append(f"{rel}: its part {e['part']!r} is not in content.parts")
        else:
            count[e["part"]] += 1
    for name in sorted(parts, key=str):
        e = parts[name]
        if not (isinstance(name, str) and _NAME.match(name) and name.endswith(".tar")):
            out.append(f"{name!r}: not a part's file name")
        elif not (isinstance(e, dict) and isinstance(e.get("sha256"), str) and _HEX64.match(e["sha256"])
                  and _is_count(e.get("bytes")) and _is_count(e.get("files"))):
            out.append(f"{name}: its entry is not {{sha256, bytes, files}}")
        elif e["files"] != count[name]:
            out.append(f"{name}: lists {e['files']} files, content.files gives it {count[name]}")
    return out


def _file_ok(e) -> bool:
    return isinstance(e, dict) and isinstance(e.get("sha256"), str) and _is_count(e.get("bytes"))


# ------------------------------------------------------------------------------------------------
# verify
# ------------------------------------------------------------------------------------------------
def verify(manifest, root, parts_dir=None) -> dict:
    """Re-hash every file of the manifest under ``root`` and, with ``parts_dir``, every part there.  Never raises on
    a difference (a manifest that cannot be read is one, and so is a file or a part that cannot be looked at): the
    caller prints the result.

    ``manifest``: the document or the path of ``MANIFEST.json``.  Returns ``{ok, bundle_id, content_sha256,
    files_ok, files, parts_ok, parts, first_problem, problems}``: ``files_ok`` of ``files`` files are under ``root``
    with the manifest's size and sha256 (two counts); ``parts_ok`` of ``parts`` parts likewise in ``parts_dir``
    (``parts_ok`` is None when no ``parts_dir`` is given); ``problems`` holds every difference, each starting with
    the path (relative to the study, or the part's file name) or with ``MANIFEST``, the manifest's own first, then
    the parts', then the files'; ``first_problem`` is its first entry or None; ``ok`` is True when there is none.
    ``content_sha256`` is recomputed from the content block."""
    out = dict(ok=False, bundle_id=None, content_sha256=None, files_ok=0, files=0,
               parts_ok=None if parts_dir is None else 0, parts=0, first_problem=None, problems=[])
    try:
        doc = load(manifest)
    except BundleError as exc:
        out.update(problems=[f"MANIFEST: {exc}"], first_problem=f"MANIFEST: {exc}")
        return out
    problems = manifest_problems(doc)
    content = doc.get("content") if isinstance(doc.get("content"), dict) else {}
    files = content.get("files") if isinstance(content.get("files"), dict) else {}
    parts = content.get("parts") if isinstance(content.get("parts"), dict) else {}
    out.update(files=len(files), parts=len(parts))
    if isinstance(content.get("bundle_id"), str):
        out["bundle_id"] = content["bundle_id"]
    try:
        out["content_sha256"] = s2io.sha256_json(content) if content else None
    except (TypeError, ValueError, RecursionError):
        pass

    def check(path: Path, label: str, e: dict) -> bool:
        try:  # is_file itself raises where the file cannot be looked at (a mount that went away)
            if not path.is_file():
                problems.append(f"{label}: missing" if not os.path.lexists(path) else f"{label}: not a file")
                return False
            got, n = _hash(path)
        except OSError as exc:
            problems.append(f"{label}: cannot be read ({exc.strerror or exc})")
            return False
        if n != e["bytes"] or got != e["sha256"]:
            problems.append(f"{label}: {n} bytes and sha256 {got[:12]}..., the manifest holds {e['bytes']} and "
                            f"{e['sha256'][:12]}...")
            return False
        return True

    if parts_dir is not None:
        pdir = _anchored(parts_dir)
        for name in sorted(parts, key=str):
            if isinstance(name, str) and _NAME.match(name) and _file_ok(parts[name]):
                out["parts_ok"] += check(pdir / name, name, parts[name])
    base = _anchored(root)
    for rel in sorted(files, key=str):
        if path_problem(rel) is None and _file_ok(files[rel]):
            out["files_ok"] += check(base / rel, rel, files[rel])
    out.update(problems=problems, first_problem=problems[0] if problems else None, ok=not problems)
    return out


# ------------------------------------------------------------------------------------------------
# unpack
# ------------------------------------------------------------------------------------------------
def _dest(root_real: str, rel: str) -> Path:
    """Where ``rel`` goes under the resolved root; ``BundleError`` when its directory resolves outside the root (a
    symbolic link on the way), a directory stands at the file's place or something that is no directory stands
    where one of its directories must be."""
    dest = Path(root_real) / rel
    try:
        parent = os.path.realpath(dest.parent)
        if os.path.commonpath([root_real, parent]) != root_real:
            raise BundleError(f"{rel}: its directory resolves to {parent}, outside the root {root_real} (a symbolic "
                              "link on the way): it would leave the root")
        if dest.is_dir() and not dest.is_symlink():
            raise BundleError(f"{rel}: a directory stands at {dest}")
        above = dest.parent
        while not os.path.lexists(above) and above != above.parent:
            above = above.parent
        if not above.is_dir():
            raise BundleError(f"{rel}: {above} is in its way and is no directory")
    except OSError as exc:
        raise BundleError(f"{rel}: its place {dest} cannot be looked at ({exc.strerror or exc}); nothing is "
                          "unpacked") from None
    return dest


def unpack(manifest, parts_dir, root, *, parts=None) -> dict:
    """Unpack the bundle's parts from ``parts_dir`` to their study paths under ``root``.

    ``manifest``: the document or the path of ``MANIFEST.json``; ``parts`` (keyword only): the part names to unpack,
    default every part of the manifest (rule N4's second fallback brings the model parts alone).

    Before anything is written, every chosen part is read to its end and ``BundleError`` refuses: a manifest with a
    problem (``manifest_problems``), a part that is missing or whose size or sha256 differs, and whatever
    ``scan_part`` refuses in a part (a member that is not one of the part's files in ``content.files``, a link or a
    directory, an absolute path or one with ``..``, a member with other bytes, a missing file); also a file whose
    directory under ``root`` resolves outside it or cannot be made, and a part or a place that cannot be looked at
    (every such refusal is a ``BundleError``, never an ``OSError``).  Then each file is written beside its place as
    ``<name>.tmp``, flushed, compared once more with the manifest and moved into place.  A regular file already
    there with the manifest's sha256 is left untouched, so a second call writes nothing.

    Returns ``{ok: True, bundle_id, content_sha256, root, parts, files, written, replaced, left}``: the part names
    unpacked, the number of their files, and the paths written new, written over another file, and left."""
    doc = load(manifest)
    bad = manifest_problems(doc)
    if bad:
        raise BundleError(f"the manifest has {_count(len(bad), 'problem')}, nothing is unpacked: "
                          + "; ".join(bad[:5]))
    content = doc["content"]
    names = sorted(content["parts"]) if parts is None else sorted(set(parts))
    unknown = [n for n in names if n not in content["parts"]]
    if unknown:
        raise BundleError(f"not parts of bundle {content['bundle_id']}: {', '.join(map(str, unknown))}")
    pdir = _anchored(parts_dir)
    root_real = os.path.realpath(_anchored(root))
    by_part = {name: {} for name in names}
    for rel, e in content["files"].items():
        if e["part"] in by_part:
            by_part[e["part"]][rel] = e

    dests = {}
    for name in names:  # every part and every member, before anything is written
        want, path = content["parts"][name], pdir / name
        try:  # is_file itself raises where the part cannot be looked at (a mount that went away)
            if not path.is_file():
                raise BundleError(f"{name}: missing in {pdir}; nothing is unpacked")
            got, n = _hash(path)
        except OSError as exc:
            raise BundleError(f"{name}: cannot be read in {pdir} ({exc.strerror or exc}); nothing is "
                              "unpacked") from None
        if n != want["bytes"] or got != want["sha256"]:
            raise BundleError(f"{name}: {n} bytes and sha256 {got[:12]}..., the manifest holds {want['bytes']} and "
                              f"{want['sha256'][:12]}...; nothing is unpacked")
        scan_part(path, by_part[name])
        for rel in by_part[name]:
            dests[rel] = _dest(root_real, rel)

    written, replaced, left = [], [], []
    for name in names:
        try:
            with tarfile.open(pdir / name, mode="r:") as tar:
                for info in tar:
                    rel = info.name
                    e, dest = by_part[name].get(rel), dests.get(rel)
                    if e is None or dest is None or info.type not in (tarfile.REGTYPE, tarfile.AREGTYPE):
                        raise BundleError(f"{name}: changed while it was unpacked (member {rel!r})")
                    if dest.is_file() and not dest.is_symlink() and _hash(dest) == (e["sha256"], e["bytes"]):
                        left.append(rel)
                        continue
                    there = os.path.lexists(dest)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dest.with_name(dest.name + ".tmp")  # s2io's convention; no "*.npz" pattern matches it
                    _unlink(tmp)  # what a killed unpack left; created anew below, so a link there is never followed
                    try:
                        h, n, src = hashlib.sha256(), 0, tar.extractfile(info)
                        with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666), "wb") as fh:
                            for block in iter(lambda: src.read(_BLOCK), b""):
                                h.update(block)
                                n += len(block)
                                fh.write(block)
                            fh.flush()
                            os.fsync(fh.fileno())
                        if (h.hexdigest(), n) != (e["sha256"], e["bytes"]):
                            raise BundleError(f"{name}: changed while it was unpacked ({rel} as read is not the "
                                              "manifest's file)")
                        os.replace(tmp, dest)
                    except BaseException:
                        _unlink(tmp)
                        raise
                    (replaced if there else written).append(rel)
        except tarfile.TarError as exc:
            raise BundleError(f"{name}: changed while it was unpacked ({exc})") from None
        except OSError as exc:  # a full disk, a place that cannot be written, another unpack at work here
            raise BundleError(f"{name}: a file could not be written under {root_real} ({exc})") from None
    return dict(ok=True, bundle_id=content["bundle_id"], content_sha256=s2io.sha256_json(content), root=root_real,
                parts=names, files=len(dests), written=sorted(written), replaced=sorted(replaced), left=sorted(left))


# ------------------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------------------
def _count(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else many or one + 's'}"


def _csv(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _mb(text: str):
    x = float(text)
    return int(x) if x.is_integer() else x


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m designlab.stage2.bundle", description=__doc__.split("\n\n")[0],
                                 epilog="A relative path is taken from the study directory "
                                        "(hardware/trex_form_study), not from the working directory.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("build", help="write the parts and MANIFEST.json of a bundle")
    a.add_argument("--bundle-id", required=True)
    a.add_argument("--bodies", required=True, help="Stage 1 body names, comma-separated")
    a.add_argument("--sets", default="V6", help='reference-set names, comma-separated (default V6); "" packs the '
                                                "model files only")
    a.add_argument("--out", default=None, help="default stage2/tb/bundle/<bundle id>")
    a.add_argument("--part-mb", type=_mb, default=20, help="a part's largest size in units of 1,000,000 bytes "
                                                             "(default 20; rule N4's first fallback: 5)")
    a = sub.add_parser("verify", help="re-hash every file under a root (and every part) against a manifest")
    a.add_argument("--manifest", required=True)
    a.add_argument("--root", required=True)
    a.add_argument("--parts", default=None, help="a directory holding the parts: each is hashed too")
    a = sub.add_parser("unpack", help="check the parts and write their files to their study paths under a root")
    a.add_argument("--manifest", required=True)
    a.add_argument("--parts", required=True)
    a.add_argument("--root", required=True)
    a.add_argument("--only", default=None, help="part names to unpack, comma-separated (default: every part)")
    args = ap.parse_args(argv)

    if args.cmd == "build":
        try:
            doc, info = _build(args.bundle_id, _csv(args.bodies), _csv(args.sets), args.out, args.part_mb, None)
        except s2io.S2Error as exc:
            print(f"S2 BUNDLE {args.bundle_id}: NOT BUILT: {exc}", flush=True)
            return 1
        c = doc["content"]
        print(f"S2 BUNDLE {c['bundle_id']}: {_count(len(c['bodies']), 'body', 'bodies')} ({', '.join(c['bodies'])}), "
              + (f"{_count(len(c['sets']), 'set')} ({', '.join(sorted(c['sets']))})" if c["sets"] else "no set")
              + f": {_count(len(c['files']), 'file')} in {_count(len(c['parts']), 'part')}, "
              f"{sum(p['bytes'] for p in c['parts'].values())} bytes", flush=True)
        for name in sorted(c["parts"]):
            p = c["parts"][name]
            print(f"  {name}: {p['bytes']} bytes, {_count(p['files'], 'file')}, sha256 {p['sha256']}", flush=True)
        for note in info["notes"]:
            print(f"  note: {note}", flush=True)
        print(f"S2 BUNDLE {c['bundle_id']}: content sha256 {doc['content_sha256']} (manifest "
              f"{doc['content_sha256'][:12]}); {info['manifest']} "
              + ("written" if info["written"] else "left as it was (the same content)")
              + f"; parts in {info['parts_dir']}", flush=True)
        return 0
    if args.cmd == "verify":
        got = verify(args.manifest, args.root, args.parts)
        line = (f"S2 BUNDLE {got['bundle_id']}: manifest {str(got['content_sha256'])[:12]}, {got['files_ok']} of "
                f"{got['files']} files match their sha256")
        if args.parts is not None:
            line += f", {got['parts_ok']} of {got['parts']} parts"
        print(line + (": OK" if got["ok"] else f": FAILED ({got['first_problem']})"), flush=True)
        for p in got["problems"][1:20]:
            print(f"  also: {p}", flush=True)
        return 0 if got["ok"] else 1
    try:
        got = unpack(args.manifest, args.parts, args.root, parts=_csv(args.only) if args.only is not None else None)
    except s2io.S2Error as exc:
        print(f"S2 BUNDLE: NOT UNPACKED: {exc}", flush=True)
        return 1
    print(f"S2 BUNDLE {got['bundle_id']}: manifest {got['content_sha256'][:12]}, {_count(got['files'], 'file')} of "
          f"{_count(len(got['parts']), 'part')} under {got['root']}: {len(got['written'])} written, "
          f"{len(got['replaced'])} replaced, {len(got['left'])} already there: OK", flush=True)
    return 0


__all__ = ["BundleError", "INPUT_LISTS", "MANIFEST", "MB", "PARTS", "RULE", "SCHEMA", "SETS_DIR", "build",
           "content_sha256", "load", "main", "manifest_problems", "model_paths", "path_problem", "scan_part",
           "tar_bytes", "unpack", "verify"]

if __name__ == "__main__":
    from designlab.stage2 import bundle as _named  # the importable copy owns every class and all module state

    sys.exit(_named.main())
