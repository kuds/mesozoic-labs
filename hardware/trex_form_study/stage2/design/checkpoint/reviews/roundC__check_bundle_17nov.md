Verdict: sound

## Verdict: sound

Bundle s2-b1 (75-M-H-10, set V6, 19 files in 3 parts) is built the way D20 and D24 say. It rebuilds byte for byte, and every part's bytes and sha256 match the manifest. Only MANIFEST.json is committed, and the commit's attribution lines are right. Every claim in the report that I checked held. I changed no tracked file and committed nothing. My scripts and outputs are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/v11/check/`.

## What I ran and saw

**Environment**
- `df -h /` showed 15G free.
- The bit-identity line printed `45719d03…528d c0027992…c1c1`, which matches the expected values.
- The branch's reflog shows the fast-forward from c5d84ae4 to ac231e26, then commit b38be399.

**My own rebuild**
- I ran the same `bundle build --bundle-id s2-b1 --bodies 75-M-H-10 --sets V6 --part-mb 20` from the worktree's study directory, with the required environment and through `tools/lowprio.sh`.
- Output went to `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/v11/check/rebuild/s2-b1`.
- It printed the same three part hashes and content sha256 9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc.
- `cmp` found all three parts identical to the committed build's.
- The content blocks are equal, and a canonical JSON hash I computed independently matches `content_sha256`.
- Only `commit` and `made_at` differ in provenance. Mine ran at b38be399; the original at ac231e26.
- The builder's own scratch rerun (`scratchpad/bundle-rerun/s2-b1`) also matches: same content and parts. It differs in made_at, and in dirty/dirty_paths because the first build's MANIFEST.json was untracked at the time, exactly as reported.

**Parts** (stat and sha256sum agree with the manifest; every part is at most 20,000,000 bytes)
- models-75-M-H-10.tar: 972,800 bytes, 2 files, 2e9a55fc…6917
- V6-75-M-H-10-1.tar: 16,281,600 bytes, 5 files, 11d7bd3e…2387
- V6-75-M-H-10-2.tar: 16,558,080 bytes, 12 files, 6739c8b7…ba39

**Tar format** (checked by parsing the raw headers)
- Members are sorted and equal each part's files in the manifest.
- Every member: mode 0644, uid and gid 0, mtime 0, empty user and group names, regular file, `ustar\0` "00" magic, no pax or GNU headers.
- The trailers are zero and each part is a multiple of 10,240 bytes.
- Every member's bytes equal the file on disk.

**Tool checks**
- `bundle verify --parts`: "19 of 19 files … 3 of 3 parts: OK", exit 0.
- `tests/test_stage2_bundle.py`: 118 passed. The report says the builder did not run these; I did.

**Freeze, models and records**
- **Freeze:** V6/75-M-H-10's `set.json` `freezes[-1]` is index 0, manifest_sha256 2eeaebe8…8def, 17 entries, at 2026-11-17T17:22:04Z, commit 78ecff3f. The hash recomputes. Every entry is packed with its sha256, and nothing outside the freeze is packed.
- **Models:** the XML's sha256 (0517e4d3…1c7c) equals the candidate record's `xml_sha256`, the entry in `stage1_inputs.json` and the file on disk. The JSON's (58f7cc20…1a90) equals the inputs list and the disk.
- **Records:** all 5 are tracked, and their bytes are the same on disk, at HEAD, at ac231e26 and at the working branch's tip 6bfc0128. The tool's file is the same at both commits too. So the content hash would not change if built at the tip.

**Commit b38be399**
- It adds only `stage2/tb/bundle/s2-b1/MANIFEST.json`.
- The message ends with exactly the two required lines, and no AI model is named anywhere else in it.
- The author field "Claude <noreply@anthropic.com>" matches every earlier commit in the repository.
- `git check-ignore -v`: the parts and `parts/` are ignored by `stage2/tb/.gitignore:13`, `.lock` by line 9. MANIFEST.json is not ignored.
- The worktree is clean, before and after my checks.
- The main checkout is clean at 6bfc0128.
- No data branch, `s2-*` tag or UPLOADS.json exists, and no remote branch contains b38be399.

**Merge readiness**
- wt/wp-p42 and the working branch are now one commit apart in each direction. Since ac231e26 the tip changed only DECISION_LOG.md and `drive_route.json`.
- A trial `git merge-tree --write-tree 6bfc0128 HEAD` exited 0, so it merges cleanly. It needs a merge commit, not a fast-forward.

## The report's points for the lead (confirmed)

- **The -X question.** The plan's queue includes both heads (STAGE2_PLAN section 6). D5 asks only that V6 be frozen per body and head. The -X set is frozen: index 0, 77516680…, and its `g9_ref.json` has `pass_` true.
  - The V0 preconditions are open: the design's risk 10 and its owner list want the jaw cover decision before any -X set, and HEAD_X.md lines 111–113 say trimming the cover changes every -X record.
  - `gates/V6/75-M-H-10-X/` holds only `g9_ref.json`, with no G8a.
  - `lr_probe.json` probed only 75-M-H-10.
  - Rule O (plan line 911) wants the -X head probed before any Colab job of that body is queued.
  - Leaving -X out and handing the choice to the lead fits either way the task framed it.
- **Drive route.** `drive_route.json` records about 3.1 MB per connector call; the 5 MB and 20 MB test tars were refused.
  - Four packed files are over 3.1 MB, so no part size gets the arrays through that route.
  - P46 chose the data branch. The notebook's `git:<ref>` path reads `stage2/tb/bundle/<id>/parts/<name>`, which is where these parts sit.
- **P46's "about 50 MB, 8 files" figure.** The plain-body files total 33,776,402 bytes, with 4 files of 3.0 MB or more, so P46's figure presumably counts both heads.
- **D24's "committed rules".** The manifest names them only through `provenance.commit` and the tool's own rule text. Section 3.10's schema has no field for a rules file, so this is not a defect of the build.

## Not done by me

I did not run `unpack`, and I did not rebuild at 6bfc0128; instead I compared the records' and the tool's bytes at that commit. I made no merge, push, upload, data branch, UPLOADS.json or tag.
