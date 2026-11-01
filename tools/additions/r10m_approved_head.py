def approved_branch_head(approved: str | None = None, repo: str | None = None) -> dict:
    """PC: the branch HEAD the runs train, read from the approval record (default <PCR>/APPROVED, JSON key
    "branch_head") and checked in *repo* (WT unless given; read only): a 40-hex commit; BASE_COMMIT (8737cd3d,
    STANCE6's commit) is an ancestor of it; the M1 and M3 pinned files are unchanged between BASE_COMMIT and it (the
    criteria: HEAD must leave them unchanged; check_pins then checks their sha256 in the copy).  Refused (Refusal,
    exit 2) otherwise.  Returns the head, the record's path and sha256, the commits BASE..HEAD and the files changed.
    The round's header-commit rule is kept below as approved_branch_head_round (not called)."""
    repo = repo or BR
    path = Path(approved or APPROVED)
    try:
        raw = path.read_bytes()
        rec = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise Refusal(f"no readable approval record at {path} ({type(exc).__name__}: {exc}): the branch HEAD is read "
                      "only from it (PC criteria MEASUREMENT); nothing measured") from exc
    head = rec.get("branch_head") if isinstance(rec, dict) else None
    if not isinstance(head, str) or len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise Refusal(f"{path}: branch_head {head!r} is not a 40-hex commit")
    try:
        kind = _git(repo, "cat-file", "-t", head).strip()
        anc = subprocess.run(["git", "--no-optional-locks", "-C", repo, "merge-base", "--is-ancestor", BASE_COMMIT,
                              head], capture_output=True, text=True).returncode
        commits = _git(repo, "rev-list", "--reverse", f"{BASE_COMMIT}..{head}").split()
        changed = _git(repo, "diff", "--name-only", "--no-renames", BASE_COMMIT, head).split()
    except (subprocess.CalledProcessError, OSError) as exc:
        raise Refusal(f"{path}: branch_head {head} cannot be read in {repo} ({exc})") from exc
    if kind != "commit":
        raise Refusal(f"{path}: branch_head {head} is a {kind}, not a commit")
    if anc != 0:
        raise Refusal(f"branch_head {head} does not descend from the base {BASE_COMMIT} (STANCE6's commit)")
    touched = sorted(p for p in (*M1_PINS, *M3_PINS) if p in changed)
    if touched:
        raise Refusal(f"branch_head {head} changes the pinned measuring files {touched} against {BASE_COMMIT}")
    return {"head": head, "base": BASE_COMMIT, "approved": os.path.realpath(path), "approved_sha256": sha256_bytes(raw),
            "repo": os.path.realpath(repo), "commits_since_base": commits, "files_changed_since_base": changed}
