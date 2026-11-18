## V11 bundle report: s2-b1, built for 75-M-H-10 on V6 only

Bundle s2-b1 is built, it rebuilds bit-identically, it verifies, and its manifest is committed on wt/wp-p42 as b38be39976b48606874509408b281b50b53314de. Nothing was pushed or uploaded.

**Setup**
- `df -h /` showed 15G free.
- The worktree `/home/user/mesozoic-wt/wp-p42` was fast-forwarded from c5d84ae4 to ac231e26 (the working branch's head at that time).
- The bit-identity line printed `45719d03…528d c0027992…c1c1` before the build and again after the commit. Both match.

**Command** (from the worktree's study directory, with the required environment, through `tools/lowprio.sh`, using the main checkout's venv python):
`python -m designlab.stage2.bundle build --bundle-id s2-b1 --bodies 75-M-H-10 --sets V6 --out stage2/tb/bundle/s2-b1 --part-mb 20`
- The tool has no fixed pattern for bundle ids, so I used `s2-b1`.
- `--part-mb 20` is D20's 20 MB limit, which is also the tool's default; it means 20,000,000 bytes per part.

**Bodies and sets**
- Body: 75-M-H-10.
- Models: `round2/refine/feet_75/models/75-M-H-10.xml` (sha256 0517e4d3…1c7c) and `.json` (58f7cc20…1a90).
- Set: V6, freeze 0, manifest_sha256 2eeaebe8d37cf0f60fda90bd9c71df4e828d35b51fc6411efc072d23e7e38def, 17 files (6 refs, 11 transitions).
- 19 files in 3 parts, 33,812,480 bytes in total.

**Parts** (name, bytes, sha256)
- models-75-M-H-10.tar: 972,800 bytes, 2 files, 2e9a55fcad2ff40b29ca1be706a6e2fddb99f6236d75997423de3264013f6917
- V6-75-M-H-10-1.tar: 16,281,600 bytes, 5 files, 11d7bd3e3749c3a7c519cf0dfeadab228c8b77b7fd586176e73ba696ebea2387
- V6-75-M-H-10-2.tar: 16,558,080 bytes, 12 files, 6739c8b72699b9887fd99f5c4ee84c0df78c246550252661a3258aa6c529ba39

**Content block hash** (what the steps sheet quotes): 9227d8a0bf0bef46769404fe0a5a0443cd2512570bf7b307d2d1633b6527f9dc. Line 2 shows it as `manifest 9227d8a0bf0b`.

**Determinism (D20)**
- I ran the same build a second time with `--out /tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/bundle-rerun/s2-b1`. That copy is left in place.
- `cmp` found all three parts identical, and sha256sum gives the same hashes for both copies.
- The two content blocks are equal and give the same content sha256.
- The provenance differs only in `made_at`, `dirty` and `dirty_paths`. The scratch build saw the first build's manifest as untracked; the committed manifest reads dirty false.

**Checks against the manifest**
- `bundle verify --root . --parts …`: "19 of 19 files match their sha256, 3 of 3 parts: OK", exit 0.
- Independent check with stat and sha256sum: every part's bytes and sha256 equal the manifest's, and every part is at most 20,000,000 bytes.
- Tar members are sorted and equal each part's `content.files`. Every member has mtime 0, uid and gid 0, empty user and group names, mode 0644, is a regular file, and has the raw `ustar\000` header magic, with no pax or GNU extension headers.
- **Freeze:** the manifest's freeze index, manifest hash and file count equal `set.json` `freezes[-1]` (the set's only freeze, at 2026-11-17T17:22:04Z, commit 78ecff3f). Every freeze entry is packed with the freeze's sha256.
- **Models:** the XML's sha256 equals the candidate record's `xml_sha256` and the entry in `stage1/runs/2026-11-08/stage1_inputs.json`. The ledger JSON's sha256 equals its entry in that list.
- **Records:** provenance lists the 5 records the tool read with their sha256; all 5 are tracked and unchanged against HEAD, and `records_committed` is true. `provenance.commit` is ac231e26, `tool_sha256` is ea11c37d…777d, and the tool's own rule text (RULE) is included.

**Commit**
- b38be39976b48606874509408b281b50b53314de on `wt/wp-p42`, holding only `stage2/tb/bundle/s2-b1/MANIFEST.json`, which is the one file the tool marks as tracked.
- `git check-ignore -v`: the three parts and `parts/` are ignored by `stage2/tb/.gitignore:13:parts/`, and `.lock` by line 9. MANIFEST.json is not ignored (exit 1).
- The worktree is clean after the commit.

## For the lead

1. **The -X question: the rules are ambiguous, so 75-M-H-10-X is out and the choice is yours.**
   - **For including it:** STAGE2_PLAN §6's queue puts R5 seed 42 on both heads in the first phase. D5's table asks only that V6 be frozen on each body and head trained. For -X it is (freeze 0, 77516680b662), and G9-ref passed (`pass_` true).
   - **Against it:** D5's V0 paragraph is worded as a condition on "the look work package commits to four bodies". Design §7 item 10 and the "For the owner" list want the jaw cover decision before any -X set. HEAD_X.md says trimming the cover changes every -X record. Neither text says in words that -X stays out of a bundle or a launch until V0.
   - **Also seen:** `stage2/tb/gates/V6/75-M-H-10-X/` holds only `g9_ref.json`, with no G8a. `validation/lr_probe.json` probed only 75-M-H-10.
   - Rule O (plan, around line 911) says "A body's `-X` head is probed before any Colab job of that body is queued". That would apply to the plain 75 cm body's first Colab queue too, so it needs settling before V13.
   - Adding -X later means a new bundle id, because a bundle id names one content. By K6 a run keeps one bundle.
2. **Part sizes against the Drive route.** The working branch moved after my fast-forward, to 6bfc0128 (P45, P46 and `validation/drive_route.json`).
   - That record says the connector carries files of about 3.1 MB only; the 5 MB and 20 MB test tars were refused.
   - Only the models part (0.97 MB) fits. The two set parts (16.3 and 16.6 MB) do not.
   - No part size fixes this: 4 single files are over 3.1 MB (`refs/v0.5.npz` at 3,842,157 bytes, and `v0.35`, `v0.4`, `v0.45` at 3,547,725 each).
   - P46 chose the data branch (D20's third fallback). The parts are ready for that route; I did not create the branch.
   - P46 describes the 75 cm bundle as "about 50 MB with 8 files of 3.0–3.8 MB". This plain-body bundle is 33,776,402 bytes of files, with 4 files of 3.0 MB or more, so that estimate presumably counts both heads.
3. **Branch divergence.** wt/wp-p42 (b38be399, built on ac231e26) is no longer a fast-forward of the working branch (6bfc0128), so bringing it in needs a merge. The commits since ac231e26 changed only DECISION_LOG.md and drive_route.json, which are not records the bundle reads, so the content hash is unaffected. The manifest's provenance names ac231e26.
4. **D24's "committed rules".** The manifest names the rules only through `provenance.commit` (ac231e26, which holds STAGE2_TRACKB_PLAN.md and the design), plus the tool's own rule text and the records it read. Neither it nor the 3.10 schema has a field naming the rules file or its hash; the steps sheet's `<<RULES_FILE>>` would carry that.
5. **Not done:** no push, upload, data branch, UPLOADS.json, C1 or tag, or merge into the working branch. I did not run the bundle tests or `unpack`, and did not rebuild at 6bfc0128. The main checkout was not touched.
