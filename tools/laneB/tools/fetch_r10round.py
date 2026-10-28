"""T. rex r10 round: the fetch planner and verifier (criteria lean v5 FETCHES).  No Drive access of its own.

It checks and records the copies an agent session makes through the Drive connector (read only), into
<ROUND>/fetch/ (never into a run folder, never onto Drive):
    <ROUND>/fetch/manifest.csv                        every transfer: COPY or TRANSFER-ERROR, and every listing
    <ROUND>/fetch/folders.json                        the run folders' ids (from the ppo folder's listing, by id)
    <ROUND>/fetch/<FETCH>/<RUN_ID>/listings/<key>.json  a listing, every page followed to the last
    <ROUND>/fetch/<FETCH>/<RUN_ID>/copies/<path>.copy<k>  each COPY of a run-relative path
FETCH is F0 (the launch check), TRIGGER (the trigger listing), F2 or REFETCH.

    PY -I -B fetch_r10round.py folders  --ppo-id ID --pages P1.json [P2.json ...]
    PY -I -B fetch_r10round.py listing  --fetch F --run RUN_ID --key KEY --folder-id ID --pages P1.json [...]
    PY -I -B fetch_r10round.py transfer --fetch F --run RUN_ID --path PATH (--b64 FILE | --saved FILE)
    PY -I -B fetch_r10round.py next     --fetch F --run RUN_ID         # the next actions, in the text's order
    PY -I -B fetch_r10round.py status   --fetch F --run RUN_ID
    PY -I -B fetch_r10round.py runlog-listing --run RUN_ID             # the LISTING note line for RUNLOG
    (every subcommand takes --round ROUND; default SP/foot/stance6)

Rules applied (FETCHES): every listing by folder id, followed through its page tokens to the last page; a COPY is
one transfer whose decoded bytes have the size Drive's listing gives; a transfer that fails its own decode or that size
check is a TRANSFER-ERROR, recorded, repeated and not counted as a copy, up to 10 transfers per file in one fetch
(v5 O-1): a file whose 10 transfers give no copy is not fetched (C3 then fails and the re-fetch applies; for PRESENCE
the file is absent); an eleventh transfer is refused and recorded OVER-CAP (no copy); every transfer, error or not, is
recorded in the fetch manifest with its decoded size; no transfer is repaired by hand.  Only sizes and sha256 are
evidence (Drive's modifiedTime is never read).  F0: provenance.json,
zero_action_baseline.json, 01_stance/stage_config.json and, once listed, 01_stance/eval_episodes.csv and
01_stance/progress.json, two agreeing copies (a third on a mismatch).  TRIGGER: the listing of 01_stance/models/ only.
F2 / REFETCH, per run, in this order: (1) the listings of the run folder, 01_stance/, 01_stance/models/,
01_stance/figures/training_history/ (reached through 01_stance/figures/) and 01_stance/replays/; (2) the manifest, two
equal copies (a third on a mismatch); (3) PRESENCE's three files, two copies each (a third on a mismatch); (4) with a
manifest, the other non-zip C3 files and the readout files (v5 A-4: 01_stance/plant_identity.json and
01_stance/task_fingerprint.json among them), each copied until its sha256 equals the manifest's entry, at most three
copies; without one (a LOST run), D-16's small files, two copies each; (5) for a run passing C1, C2, C4
and C6 whose non-zip C3 files match, the pairs C3 names; (6) for a run passing C1-C6, the report-only pairs (the 11
of the 1,048,576 grid and at most 8 D-3 onset pairs).  Each zip passes testzip; its num_timesteps is noted.
Exit codes: 0 done; 1 the transfer was a TRANSFER-ERROR (recorded); 2 input error.

Fixer, 2026-10-26 (review findings):
  R-B1  a saved search_files page must be exactly what the connector returns [measured: 76 real responses in earlier
        sessions' transcripts: a JSON object with "files" (a list) and "nextPageToken" when more pages follow; an
        empty result is {}]: any other shape, an unknown key, an item without id, title and mimeType, or a non-folder
        item without an integer fileSize is refused (exit 2, nothing written).  A page token is read under
        "nextPageToken" or "next_page_token".
  R-B2  a folder absent from its parent's complete listing (the run folder from the ppo folder's, 01_stance/ from the
        run folder's, 01_stance/models/ from 01_stance/'s) is recorded as absent: `next` stops with "done: ... absent"
        and runlog-listing prints the LISTING line with folder=absent:<key> files="" and records that LISTING row in
        the fetch manifest.  v5: "a run folder, 01_stance/ or 01_stance/models/ absent from its parent's complete
        listing ... is a TRIGGER LISTING that holds no file, noted folder=absent:<key>"; the scorer reads it there.
        The run folder counts as absent only on a ppo folder listing made at or after this fetch's RUNLOG note
        (LAUNCH for F0, the trigger for TRIGGER, F2-START, REFETCH-START); an older listing is made again first.
        folders.json keeps every ppo folder listing (_ppo_listings): a later fetch's never replaces the one a trigger
        listing read (the scorer reads that one).
  R-B3  --round, --runlog and --pins are accepted before or after the subcommand.
  R-m5  --saved: the response's id and title must be the listed file's, else the transfer is recorded WRONG-FILE (no
        copy; exit 2).
  R-m8  F0: a TRANSFER-ERROR whose decoded size differs from the listing makes `next` ask for that folder's listing
        again before the next transfer (a live file); every listing is kept (<key>.json.v<n>), the newest is read.
"""
import os
import sys

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import argparse  # noqa: E402
import base64  # noqa: E402
import binascii  # noqa: E402
import csv  # noqa: E402
import fcntl  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
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
CP = _load("complete_r10round")

FETCHES = ("F0", "TRIGGER", "F2", "REFETCH")
LISTING_KEYS = ("root", "01_stance", "01_stance/models", "01_stance/figures", "01_stance/figures/training_history",
                "01_stance/replays")
F0_FILES = ("provenance.json", "zero_action_baseline.json", "01_stance/stage_config.json")
F0_LATER = ("01_stance/eval_episodes.csv", "01_stance/progress.json")
PRESENCE_FILES = ("provenance.json", "01_stance/stage_config.json", "01_stance/eval_episodes.csv")
C3_SMALL_OTHER = ("01_stance/progress.json", "01_stance/gate_verdict.json")      # hop_watch.json when listed
READOUT = ("01_stance/policy_probe.npz", "01_stance/train_episodes.csv", "01_stance/evaluations.npz",
           "01_stance/gate_progress.npz", "01_stance/plant_identity.json", "01_stance/task_fingerprint.json",
           "01_stance/k3_panel.csv", "01_stance/k3_panel.json",
           "01_stance/stance_heading_report.json", "01_stance/stance_heading_report.txt",
           "01_stance/stance_heading_panel_selected.csv")
D16_SMALL = ("01_stance/progress.json", "01_stance/gate_verdict.json", "zero_action_baseline.json",
             "01_stance/hop_watch.json")                  # v5 O-10: exactly these (hop_watch.json when listed)
TRANSFER_CAP = 10                                        # v5 O-1: "up to 10 transfers per file in one fetch"


# ------------------------------------------------------------------------------------------------- the store
def fetch_root(round_dir):
    return Path(round_dir) / "fetch"


def run_dir(round_dir, fetch, run_id):
    return fetch_root(round_dir) / fetch / run_id


def record(round_dir, row: dict):
    """Append one row to <ROUND>/fetch/manifest.csv under an exclusive lock."""
    p = fetch_root(round_dir) / "manifest.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a+", newline="", encoding="utf-8") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0, os.SEEK_END)
        w = csv.DictWriter(fh, fieldnames=C.FETCH_MANIFEST_COLUMNS, extrasaction="ignore")
        if fh.tell() == 0:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in C.FETCH_MANIFEST_COLUMNS})
        fcntl.flock(fh, fcntl.LOCK_UN)


FOLDER_MIME = "application/vnd.google-apps.folder"
PAGE_KEYS = ("files", "nextPageToken", "next_page_token")


class PageFormatError(Exception):
    """A saved page that is not a search_files response as the connector returns it (R-B1)."""


def _page_items(d, where):
    """(items, token) of one saved search_files response; PageFormatError for any other shape (R-B1)."""
    if not isinstance(d, dict):
        raise PageFormatError(f"{where} is a {type(d).__name__}, not the connector's JSON object")
    unknown = sorted(set(d) - set(PAGE_KEYS))
    if unknown:
        raise PageFormatError(f"{where} carries keys {unknown} beside {list(PAGE_KEYS)}")
    items = d.get("files", [])
    if not isinstance(items, list):
        raise PageFormatError(f"{where}: 'files' is not a list")
    toks = [d[k] for k in ("nextPageToken", "next_page_token") if k in d]
    if any(not isinstance(t, str) or not t for t in toks) or len(set(toks)) > 1:
        raise PageFormatError(f"{where}: page token(s) {toks!r} are not one non-empty string")
    for k, it in enumerate(items):
        if not isinstance(it, dict):
            raise PageFormatError(f"{where} item {k} is not an object")
        for key in ("id", "mimeType"):
            if not isinstance(it.get(key), str) or not it[key]:
                raise PageFormatError(f"{where} item {k} has no {key}")
        if not isinstance(it.get("title") or it.get("name"), str):
            raise PageFormatError(f"{where} item {k} has no title")
        if it["mimeType"] != FOLDER_MIME:
            size = it.get("fileSize")
            ok = (isinstance(size, str) and size.isdigit()) or (isinstance(size, int) and not isinstance(size, bool))
            if not ok:
                raise PageFormatError(f"{where} item {k} ({it.get('title')}) has no integer fileSize ({size!r})")
    return items, (toks[0] if toks else None)


def parse_pages(paths):
    """Items of a listing's saved search_files responses, pages in order; the last page must carry no page token
    (every listing followed to its last page).  Returns (files, folders, problems).  Raises PageFormatError for a
    page that is not the connector's response shape (R-B1)."""
    files, folders, probs = [], [], []
    for i, p in enumerate(paths):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise PageFormatError(f"page {i + 1} ({p}) is not JSON: {exc}") from exc
        items, token = _page_items(d, f"page {i + 1} ({p})")
        if i < len(paths) - 1 and not token:
            probs.append(f"page {i + 1} has no nextPageToken but more pages were given")
        if i == len(paths) - 1 and token:
            probs.append(f"the last page given ({p}) still carries a nextPageToken: the listing is not followed to its last page")
        for it in items:
            name = it.get("title") or it.get("name")
            mime = it["mimeType"]
            folder = mime == FOLDER_MIME
            ent = dict(name=name, id=it["id"], mime=mime, size=None if folder else int(it["fileSize"]))
            (folders if folder else files).append(ent)
    names = [f["name"] for f in files] + [f["name"] for f in folders]
    if len(names) != len(set(names)):
        probs.append("a title repeats in this folder: printed; the copies are made by the listed id")
    return files, folders, probs


def listing(round_dir, fetch, run_id, key):
    p = run_dir(round_dir, fetch, run_id) / "listings" / f"{key}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def absent_folder(round_dir, fetch, run_id, runlog=None):
    """R-B2: (key, parent key, parent folder id) of the first folder on the path to 01_stance/models/ that a complete
    parent listing shows absent, or None.  The run folder's parent is the ppo folder (folders.json); v5: it counts as
    absent only on a complete ppo listing made at or after this fetch's RUNLOG note (the trigger for TRIGGER)."""
    fj = fetch_root(round_dir) / "folders.json"
    folders = json.loads(fj.read_text()) if fj.is_file() else {}
    ppo = folders.get("_ppo_listing") or {}
    if run_id not in folders:
        ref = reference_note(runlog, fetch, run_id) if runlog else None
        lt = C.parse_utc(ppo.get("utc"))
        if ppo.get("complete") and run_id not in (ppo.get("folders") or []) and \
                (ref is None or (lt is not None and lt >= ref["time"])):
            return "run_folder", "ppo", folders.get("ppo_folder_id")
        return None
    for parent, child, key in (("root", "01_stance", "01_stance"), ("01_stance", "models", "01_stance/models")):
        lst = listing(round_dir, fetch, run_id, parent)
        if lst is None or not lst.get("complete"):
            return None
        if child not in [f["name"] for f in lst.get("folders", [])]:
            return key, parent, lst.get("folder_id")
    return None


def path_of(key, name):
    return name if key == "root" else f"{key}/{name}"


def listed_files(round_dir, fetch, run_id) -> dict:
    """run-relative path -> (drive id, size) over every listing this fetch holds for the run."""
    out = {}
    for key in LISTING_KEYS:
        lst = listing(round_dir, fetch, run_id, key)
        for f in (lst or {}).get("files", []):
            out[path_of(key, f["name"])] = (f["id"], f["size"])
    return out


def decode_transfer(args):
    """(raw bytes or None, problem, method, (id, title) or None) from the agent's transfer: --b64 (an inline
    response's base64 transcribed exactly) or --saved (the connector's saved response JSON with content/id/title)."""
    if args.b64:
        try:
            text = "".join(Path(args.b64).read_text(encoding="utf-8").split())
            return base64.b64decode(text, validate=True), None, "inline base64", None
        except (OSError, ValueError, binascii.Error) as exc:
            return None, f"base64 does not decode: {exc}", "inline base64", None
    try:
        d = json.loads(Path(args.saved).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"saved response unreadable: {exc}", "saved response", None
    if isinstance(d, list):
        d = next((x for x in d if isinstance(x, dict) and "content" in x), {})
    if not isinstance(d, dict) or "content" not in d:
        return None, "saved response has no content field", "saved response", None
    ident = (d.get("id"), d.get("title"))
    try:
        raw = base64.b64decode(d["content"], validate=True)
    except (ValueError, binascii.Error) as exc:
        return None, f"content does not decode: {exc}", "saved response", ident
    return raw, None, f"saved response (id {d.get('id')}, title {d.get('title')})", ident


# ------------------------------------------------------------------------------------------------- commands
def cmd_folders(a):
    try:
        files, folders, probs = parse_pages(a.pages)
    except PageFormatError as exc:
        print(f"INPUT ERROR (R-B1, nothing written): {exc}")
        return 2
    root = fetch_root(a.round)
    root.mkdir(parents=True, exist_ok=True)
    p = root / "folders.json"
    d = json.loads(p.read_text()) if p.is_file() else {}
    d["ppo_folder_id"] = a.ppo_id
    names = [f["name"] for f in folders]
    twice = sorted({n for n in names if names.count(n) > 1 and C.RUN_TOKEN_RE.fullmatch(n or "")})
    for f in folders:
        if C.RUN_TOKEN_RE.fullmatch(f["name"] or "") and f["name"] not in twice:
            d[f["name"]] = f["id"]
    for n in twice:
        probs.append(f"two folders are titled {n}: not recorded; stop and report")
    now = C.utcnow()
    d["_ppo_listing"] = dict(utc=now, pages=len(a.pages), folders=sorted(names),
                             complete=not any("nextPageToken" in x for x in probs))
    # v5 R-B2: every ppo folder listing is kept (a later fetch's listing never replaces the one a trigger listing read)
    d.setdefault("_ppo_listings", []).append(dict(d["_ppo_listing"]))
    p.write_text(json.dumps(d, indent=1) + "\n")
    record(a.round, dict(utc=now, fetch="FOLDERS", run_id="", path="(ppo folder listing)", drive_id=a.ppo_id,
                         status="LISTING", note=f"{len(folders)} folders, {len(files)} files; problems {probs}"))
    print(json.dumps({k: v for k, v in d.items()}, indent=1))
    for x in probs:
        print("PROBLEM:", x)
    return 2 if any(("nextPageToken" in x) or ("two folders" in x) for x in probs) else 0


def cmd_listing(a):
    if a.key not in LISTING_KEYS:
        print(f"INPUT ERROR: key {a.key!r} not one of {LISTING_KEYS}")
        return 2
    try:
        files, folders, probs = parse_pages(a.pages)
    except PageFormatError as exc:
        print(f"INPUT ERROR (R-B1, nothing written): {exc}")
        return 2
    d = run_dir(a.round, a.fetch, a.run) / "listings"
    d.mkdir(parents=True, exist_ok=True)
    rec = dict(key=a.key, folder_id=a.folder_id, utc=C.utcnow(), pages=len(a.pages), files=files, folders=folders,
               problems=probs, complete=not any("nextPageToken" in x for x in probs))
    lp = d / f"{a.key}.json"
    lp.parent.mkdir(parents=True, exist_ok=True)
    if lp.is_file():                                     # R-m8: a listing made again keeps the earlier ones
        k = 1
        while lp.with_name(f"{lp.name}.v{k}").exists():
            k += 1
        lp.rename(lp.with_name(f"{lp.name}.v{k}"))
    lp.write_text(json.dumps(rec, indent=1) + "\n")
    record(a.round, dict(utc=rec["utc"], fetch=a.fetch, run_id=a.run, path=a.key + "/", drive_id=a.folder_id,
                         status="LISTING", note=f"{len(files)} files, {len(folders)} folders, {len(a.pages)} pages; "
                         + "; ".join(f"{f['name']}:{f['size']}" for f in files)[:20000]))
    print(f"listing {a.key} of {a.run} ({a.fetch}): {len(files)} files, {len(folders)} folders over {len(a.pages)} pages")
    for x in probs:
        print("PROBLEM:", x)
    return 0 if rec["complete"] else 2


def cmd_transfer(a):
    lf = listed_files(a.round, a.fetch, a.run)
    if a.path not in lf:
        print(f"INPUT ERROR: {a.path} is in no listing of {a.run} in {a.fetch}: list its folder first")
        return 2
    did, size = lf[a.path]
    raw, prob, method, ident = decode_transfer(a)
    if ident is not None and (ident[0] != did or ident[1] != a.path.rsplit("/", 1)[-1]):
        record(a.round, dict(utc=C.utcnow(), fetch=a.fetch, run_id=a.run, path=a.path, drive_id=did,
                             listing_size=size, status="WRONG-FILE", method=method,
                             note=f"the saved response is id {ident[0]} title {ident[1]}, the listed file is id {did} "
                                  f"title {a.path.rsplit('/', 1)[-1]}: not a transfer of this file (R-m5)"))
        print(f"INPUT ERROR: the saved response is id {ident[0]} title {ident[1]}, not the listed {did} "
              f"{a.path.rsplit('/', 1)[-1]}; recorded WRONG-FILE, no copy")
        return 2
    rows = [r for r in C.FetchStore(run_dir(a.round, a.fetch, a.run), fetch_root(a.round) / "manifest.csv", a.fetch,
                                    a.run).manifest_rows() if r.get("path") == a.path]
    n_tr = len([r for r in rows if r.get("status") in ("COPY", "TRANSFER-ERROR")]) + 1
    n_cp = len([r for r in rows if r.get("status") == "COPY"])
    row = dict(utc=C.utcnow(), fetch=a.fetch, run_id=a.run, path=a.path, drive_id=did, listing_size=size,
               transfer=n_tr, method=method)
    if n_tr > TRANSFER_CAP:                              # v5 O-1: the cap; the transfer is recorded, never a copy
        row.update(status="OVER-CAP", decoded_size="" if raw is None else len(raw),
                   note=f"transfer {n_tr} of this file in this fetch: beyond the {TRANSFER_CAP} the text allows (O-1); "
                        "not a copy, not stored")
        record(a.round, row)
        print(f"INPUT ERROR: {a.path} has had {TRANSFER_CAP} transfers in {a.fetch}; no more are made (v5 O-1). "
              "Recorded OVER-CAP, no copy; the file has the copies it has.")
        return 2
    if raw is None or len(raw) != size:
        row.update(status="TRANSFER-ERROR", decoded_size="" if raw is None else len(raw),
                   note=prob or f"decoded {len(raw)} bytes, the listing gives {size}")
        record(a.round, row)
        print(f"TRANSFER-ERROR {a.path} (transfer {n_tr}): {row['note']}. Not a copy; repeat the transfer.")
        return 1
    k = n_cp + 1
    dst = run_dir(a.round, a.fetch, a.run) / "copies" / f"{a.path}.copy{k}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    tmp.write_bytes(raw)
    os.replace(tmp, dst)
    h = C.sha256_bytes(raw)
    note = ""
    if a.path.endswith(".zip"):
        nts, zp = C.zip_num_timesteps(raw)
        note = f"testzip ok; num_timesteps {nts}" if zp is None else f"zip: {zp}"
    row.update(status="COPY", decoded_size=len(raw), copy=k, sha256=h, stored_as=str(dst), note=note)
    record(a.round, row)
    print(f"COPY {k} of {a.path}: {len(raw)} bytes, sha256 {h}{'; ' + note if note else ''}")
    return 0


# ------------------------------------------------------------------------------------------------- the plan
def transfers_made(store, path) -> int:
    """COPY and TRANSFER-ERROR rows of path in this fetch (v5 O-1: at most 10)."""
    return len([r for r in store.manifest_rows() if r.get("path") == path
                and r.get("status") in ("COPY", "TRANSFER-ERROR")])


def need_agree(store, path):
    """What the two-agreeing-copies rule still needs for path: None when satisfied or failed for good (v5 O-1: also
    when 10 transfers were made)."""
    cs = store.copies(path)
    if transfers_made(store, path) >= TRANSFER_CAP:
        return None
    if len(cs) < 2:
        return f"copy {len(cs) + 1}"
    if cs[0][1] == cs[1][1]:
        return None
    if len(cs) < 3:
        return "copy 3 (copies 1 and 2 differ)"
    return None


def need_match(store, path, hx, size):
    cs = store.copies(path)[:3]
    if any(C.sha256_bytes(b) == hx and len(b) == size for _k, b, _f in cs):
        return None
    if len(cs) >= 3:
        return None                                  # three copies, none equal: the file fails C3 (at most three)
    if transfers_made(store, path) >= TRANSFER_CAP:
        return None                                  # v5 O-1: 10 transfers gave no matching copy: not fetched
    return f"copy {len(cs) + 1} (until its sha256 equals the manifest's entry)"


def onset_pairs(res, periodic):
    """D-3: per run with a tail-flagged evaluation, every 262,144 checkpoint from the last evaluation before the first
    tail-flagged one that had no tail-saturated episode up to that first tail-flagged evaluation (the 8 nearest)."""
    if not res or not res["tail_flagged"]:
        return []
    first = res["tail_flagged"][0]
    before = [s for s in sorted(res["evals"]) if s < first and res["evals"][s]["tail"] == 0]
    start = before[-1] if before else 0
    ps = [p for p in sorted(periodic) if start <= p <= first]
    return ps[-8:]


def plan_next(round_dir, fetch, run_id, runlog=C.RUNLOG, pins_path=None):
    """The next actions for one run in one fetch, in the text's order, and the state."""
    store = C.FetchStore(run_dir(round_dir, fetch, run_id), fetch_root(round_dir) / "manifest.csv", fetch, run_id)
    acts, state = [], {}
    fj = fetch_root(round_dir) / "folders.json"
    folders = json.loads(fj.read_text()) if fj.is_file() else {}
    if run_id not in folders:
        ppo = folders.get("_ppo_listing") or {}
        ref = reference_note(runlog, fetch, run_id)
        lt = C.parse_utc(ppo.get("utc"))
        if ppo.get("complete") and ref is not None and (lt is None or lt < ref["time"]):
            return [f"list the ppo folder by its id again (folders): its listing ({ppo.get('utc')}) is older than "
                    f"this fetch's RUNLOG note ({ref['kind']} {ref['time']:%Y-%m-%d %H:%M} UTC), so it cannot show "
                    f"the run folder of {run_id} absent (v5 R-B2)"], state
        if ppo.get("complete"):
            return [f"done: the run folder of {run_id} is absent from the ppo folder's complete listing (R-B2"
                    + ("; runlog-listing prints the LISTING line with folder=absent:run_folder: a TRIGGER LISTING "
                       "that holds no file)" if fetch == "TRIGGER" else ")")], state
        return [f"list the ppo folder by its id (folders) to find the run folder of {run_id}"], state
    have = {k: listing(round_dir, fetch, run_id, k) for k in LISTING_KEYS}

    def sub_id(parent_key, name):
        lst = have.get(parent_key)
        for f in (lst or {}).get("folders", []):
            if f["name"] == name:
                return f["id"]
        return None
    order = ["root", "01_stance"] + (["01_stance/models"] if fetch in ("TRIGGER", "F2", "REFETCH") else []) + \
        (["01_stance/figures", "01_stance/figures/training_history", "01_stance/replays"] if fetch in ("F2", "REFETCH") else [])
    absent = set()
    for key in order:
        if have.get(key) and have[key].get("complete"):
            continue
        if key == "root":
            fid = folders[run_id]
        else:
            parent, name = key.rsplit("/", 1) if "/" in key else ("root", key)
            fid = sub_id(parent, name)
            if fid is None:
                if parent in absent or (have.get(parent) and have[parent].get("complete")):
                    state[f"listing {key}"] = "folder absent"
                    absent.add(key)
                    if fetch == "TRIGGER" and key in ("01_stance", "01_stance/models"):
                        return [f"done: {key}/ is absent from its parent's complete listing (R-B2); runlog-listing "
                                f"prints the LISTING line with folder=absent:{key}"], state
                    continue
                return acts or [f"list {parent} first"], state
        acts.append(f"list {key} (folder id {fid}), following every page token: listing --key {key} --folder-id {fid}")
        return acts, state
    if fetch == "TRIGGER":
        return ["done: the trigger listing of 01_stance/models/ is recorded (runlog-listing prints its RUNLOG line)"], state
    lf = listed_files(round_dir, fetch, run_id)

    def want_agree(paths):
        out = []
        for p in paths:
            if p not in lf:
                state[p] = "not listed"
                continue
            n = need_agree(store, p)
            if n:
                out.append(f"transfer {p} ({n}; id {lf[p][0]}, size {lf[p][1]})")
            else:
                b, how = store.agreed(p)
                state[p] = how
        return out
    if fetch == "F0":
        relist = f0_relist(round_dir, run_id, lf)
        if relist:
            key, path, why = relist
            fid = folders[run_id] if key == "root" else sub_id("root", "01_stance")
            return [f"list {key} (folder id {fid}), following every page token: listing --key {key} --folder-id "
                    f"{fid}   [again: {path} {why}; a live file is listed again before its next transfer (R-m8)]"], state
        acts = want_agree(F0_FILES + F0_LATER)
        later = [p for p in F0_LATER if p not in lf]
        return acts or ["done: run complete_r10round.py launch-check"
                        + (f" ({later} not listed yet: list 01_stance again within the 60 minutes to fetch them "
                           "once they exist)" if later else "")], state
    # F2 / REFETCH (2) the manifest
    acts = want_agree(("artifact_manifest.json",))
    if acts:
        return acts, state
    # (3) PRESENCE's three files, every run
    acts = want_agree(PRESENCE_FILES)
    if acts:
        return acts, state
    notes, _o = C.parse_runlog(runlog, C.RUN_IDS)
    pins = C.load_pins(pins_path)
    c2f, man = CP.c2(run_id, store)
    if man is None:
        acts = want_agree([p for p in D16_SMALL if p in lf])
        return acts or [f"done: no valid manifest (C2 {CP.status_of(c2f)}): LOST; D-16's small files held"], state
    entries = C.manifest_entries(man)
    # (4) the other non-zip C3 files, then the readout files (report only)
    small = list(C3_SMALL_OTHER) + (["01_stance/hop_watch.json"] if "01_stance/hop_watch.json" in entries else [])
    extra = [p for p in lf if p.startswith("01_stance/figures/training_history/")] + \
        [p for p in lf if p.startswith("01_stance/replays/") and p.endswith("_stance.csv")]
    for p in small + list(READOUT) + extra:
        if p not in entries:
            state[p] = "not in the manifest"
            continue
        if p not in lf:
            state[p] = "in the manifest, not listed"
            continue
        size, hx = entries[p]
        n = need_match(store, p, hx, size)
        if n:
            acts.append(f"transfer {p} ({n}; id {lf[p][0]}, size {lf[p][1]})")
            if p in small:
                return acts, state
    if acts:
        return acts[:1], state
    # (5) the pairs C3 names, for a run passing C1, C2, C4 and C6 whose non-zip C3 files match
    comp = CP.completeness(run_id, store, notes, pins, CP.PanelLocator(None))
    nonzip_ok = all(comp["C3"]["value"].get(p) == "OK" for p in comp["C3"]["value"] if not p.startswith("01_stance/models/"))
    gate5 = (comp["C1"]["ok"] is True and comp["C2"]["ok"] is True and comp["C4"]["ok"] is True
             and comp["C6"]["ok"] is True and nonzip_ok)
    state["step 5"] = dict(C1=CP.status_of(comp["C1"]), C2=CP.status_of(comp["C2"]), C4=CP.status_of(comp["C4"]),
                           C6=CP.status_of(comp["C6"]), nonzip_C3_match=nonzip_ok)
    if not gate5:
        return [f"done: step 5 not reached ({state['step 5']}); the pairs are not fetched"], state
    pair_paths = [p for p in comp["C3"]["value"] if p.startswith("01_stance/models/")]
    for p in pair_paths:
        if p not in lf:
            state[p] = "in C3, not listed"
            continue
        size, hx = entries.get(p, (None, None))
        if hx is None:
            state[p] = "not in the manifest"
            continue
        n = need_match(store, p, hx, size)
        if n:
            return [f"transfer {p} ({n}; id {lf[p][0]}, size {lf[p][1]})"], state
    # (6) the report-only pairs, for a run passing C1-C6
    comp = CP.completeness(run_id, store, notes, pins, CP.PanelLocator(None))
    if not all(comp[k]["ok"] is True for k in ("C1", "C2", "C3", "C4", "C5", "C6")):
        brief = {k: CP.status_of(comp[k]) for k in ("C1", "C2", "C3", "C4", "C5", "C6")}
        return [f"done: C1-C6 not all held ({brief}); no report-only pairs"], state
    periodic = C.periodic_steps_in(entries)
    rep = [s for s in C.REPORT_GRID_PAIRS if s in periodic] + onset_pairs(comp["m5"], periodic)
    for s in rep:
        for p in C.pair_paths(s):
            if p in comp["C3"]["value"]:
                continue
            if p not in lf or p not in entries:
                state[p] = "not listed / not in the manifest"
                continue
            size, hx = entries[p]
            n = need_match(store, p, hx, size)
            if n:
                return [f"transfer {p} ({n}; report only; id {lf[p][0]}, size {lf[p][1]})"], state
    return ["done: every file of this fetch is held as the rules require"], state


REFERENCE_KINDS = {"F0": ("LAUNCH", "RELAUNCH"), "TRIGGER": ("TRIGGER-a", "TRIGGER-b", "TRIGGER-c", "TRIGGER-d"),
                   "F2": ("F2-START",), "REFETCH": ("REFETCH-START",)}


def reference_note(runlog, fetch, run_id):
    """The RUNLOG note a fetch of run_id is made after (LAUNCH for F0, the first trigger for TRIGGER, F2-START,
    REFETCH-START): a ppo folder listing older than it cannot show the run folder absent (v5 R-B2).  None when RUNLOG
    holds no such note (then the listing's time is not compared)."""
    try:
        notes, _o = C.parse_runlog(runlog, C.RUN_IDS)
    except C.InputError:
        return None
    return C.first_note(notes, run_id, REFERENCE_KINDS.get(fetch, ()))


def f0_relist(round_dir, run_id, lf):
    """R-m8: (folder key, path, why) when the newest transfer of an F0 file is a TRANSFER-ERROR whose decoded size
    differs from its listing and no listing of that folder was made after it; else None."""
    rows = C.FetchStore(run_dir(round_dir, "F0", run_id), fetch_root(round_dir) / "manifest.csv", "F0",
                        run_id).manifest_rows()
    store = C.FetchStore(run_dir(round_dir, "F0", run_id), fetch_root(round_dir) / "manifest.csv", "F0", run_id)
    for path in F0_FILES + F0_LATER:
        if path not in lf or transfers_made(store, path) >= TRANSFER_CAP:      # v5 O-1: no next transfer to list for
            continue
        key = "root" if "/" not in path else path.rsplit("/", 1)[0]
        last_t = max((i for i, r in enumerate(rows) if r.get("path") == path
                      and r.get("status") in ("COPY", "TRANSFER-ERROR")), default=None)
        last_l = max((i for i, r in enumerate(rows) if r.get("status") == "LISTING" and r.get("path") == key + "/"),
                     default=None)
        if last_t is None:
            continue
        r = rows[last_t]
        if (r.get("status") == "TRANSFER-ERROR" and str(r.get("decoded_size", "")).isdigit()
                and r.get("decoded_size") != r.get("listing_size") and (last_l is None or last_l < last_t)):
            return key, path, f"decoded {r.get('decoded_size')} bytes, the listing gives {r.get('listing_size')}"
    return None


def cmd_next(a):
    acts, state = plan_next(a.round, a.fetch, a.run, a.runlog, a.pins)
    print(f"{C.VERSION}\nNEXT for {a.run} in {a.fetch}:")
    for x in acts:
        print("  ", x)
    if a.verbose:
        print(json.dumps(state, indent=1, default=str))
    return 0


def cmd_status(a):
    acts, state = plan_next(a.round, a.fetch, a.run, a.runlog, a.pins)
    print(json.dumps(dict(next=acts, state=state), indent=1, default=str))
    return 0


def cmd_runlog_listing(a):
    lst = listing(a.round, "TRIGGER", a.run, "01_stance/models")
    if lst is None or not lst.get("complete"):
        ab = absent_folder(a.round, "TRIGGER", a.run, a.runlog)
        if ab is not None:
            key, parent, pid = ab
            print(f"<date -u '+%Y-%m-%d %H:%M UTC'> STANCE6 LISTING {a.run} folder=absent:{key} parent={parent} "
                  f"parent_id={pid} files=\"\"")
            # v5 R-B2: "the listing goes to RUNLOG and to the fetch manifest": its LISTING row, recorded once
            fm = fetch_root(a.round) / "manifest.csv"
            rows = C.FetchStore(run_dir(a.round, "TRIGGER", a.run), fm, "TRIGGER", a.run).manifest_rows()
            note = (f"folder=absent:{key} parent={parent} parent_id={pid}; 0 files: a TRIGGER LISTING that holds no "
                    "file (v5 R-B2)")
            if not any(r.get("status") == "LISTING" and r.get("path") == "01_stance/models/"
                       and (r.get("note") or "").startswith(f"folder=absent:{key} ") for r in rows):
                record(a.round, dict(utc=C.utcnow(), fetch="TRIGGER", run_id=a.run, path="01_stance/models/",
                                     drive_id="", status="LISTING", note=note))
            print(f"NOTE: {key} is absent from {parent}'s complete listing: a TRIGGER LISTING that holds no file (v5 "
                  "R-B2); its LISTING row is in the fetch manifest", file=sys.stderr)
            return 0
        print("INPUT ERROR: no complete TRIGGER listing of 01_stance/models/ for this run")
        return 2
    files = ";".join(f"{f['name']}:{f['size']}" for f in lst["files"])
    print(f"<date -u '+%Y-%m-%d %H:%M UTC'> STANCE6 LISTING {a.run} folder={lst['folder_id']} pages={lst['pages']} "
          f"files=\"{files}\"")
    return 0


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)       # R-B3: before or after the subcommand
    common.add_argument("--round", default=argparse.SUPPRESS)
    common.add_argument("--runlog", default=argparse.SUPPRESS)
    common.add_argument("--pins", default=argparse.SUPPRESS)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 parents=[common])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("folders", "listing", "transfer", "next", "status", "runlog-listing"):
        p = sub.add_parser(name, parents=[common])
        if name == "folders":
            p.add_argument("--ppo-id", required=True)
            p.add_argument("--pages", nargs="+", required=True)
        if name in ("listing", "transfer", "next", "status"):
            p.add_argument("--fetch", choices=FETCHES, required=True)
        if name in ("listing", "transfer", "next", "status", "runlog-listing"):
            p.add_argument("--run", required=True)
        if name == "listing":
            p.add_argument("--key", required=True)
            p.add_argument("--folder-id", required=True)
            p.add_argument("--pages", nargs="+", required=True)
        if name == "transfer":
            p.add_argument("--path", required=True)
            g = p.add_mutually_exclusive_group(required=True)
            g.add_argument("--b64")
            g.add_argument("--saved")
        if name == "next":
            p.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)
    for k, v in (("round", C.ROUND), ("runlog", C.RUNLOG), ("pins", None)):
        if not hasattr(a, k):
            setattr(a, k, v)
    print(C.VERSION, file=sys.stderr if a.cmd == "runlog-listing" else sys.stdout)
    handlers = {"folders": cmd_folders, "listing": cmd_listing, "transfer": cmd_transfer, "next": cmd_next,
                "status": cmd_status, "runlog-listing": cmd_runlog_listing}
    return handlers[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
