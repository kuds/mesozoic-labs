G9-ref passed on all three 75 cm look sets, each on its freeze 0. Nothing failed, so nothing was marked or frozen a second time, and the procedure of rulings 14 and 32 (A) was not needed. Simulation only: every step was kinematic and no closed-loop episode ran.

**One decision for you:** I froze all three sets with `--allow-mixed`. Verify on each set said "the entries were built from 2 differing source states: a freeze refuses them without --allow-mixed". The P42 build and check reports say the same: P42 rebuilt only the stop and turn groups, so the choice was this flag or a full rebuild. The freeze rule (S2S-freeze) allows it and records `mixed_sources: true` in each freeze. The moves.py condition (A4), which no flag waives, held: the freeze did not refuse. I did not use `--allow-dirty`; every freeze recorded `dirty: false`. If you want unmixed freezes instead, the sets need a full rebuild in an empty directory before anything runs on them.

**Setup.** I fast-forwarded the worktree `/home/user/mesozoic-wt/wp-p42` from 4ad729c5 to dd431e6c with `--ff-only`. The bit-identity check printed `45719d030359ec7d5d3914af3814d837fdc05d77652ba45c64792b511027528d c0027992de21067525cc4489b74acb157feb89f3a4c7618580addc2b341be1c1` before and after. `git diff dd431e6c..HEAD` shows no change under V0, F25F9 or F25F9cl. Disk was at 15G free; I never had more than one process running at a time.

Each set had three commits, in this order: the refcheck record, then the freeze (so the freeze's recorded commit holds the record it cites), then the G9-ref record. Every commit message ends with the two required lines and names no model anywhere else.

## 1. V6 / 75-M-H-10
- **Verify:** "V6 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures". **Status:** "V6 75-M-H-10: not frozen; lead-in start".
- **Refcheck:** exit 0, "refcheck V6 75-M-H-10: pass", every turn and stop item placed at J1's leave point. Record `stage2/sets/V6/75-M-H-10/refcheck.json`, sha256 5efd1045eaabccd1fd74a09fd616bd747f8a654380aa71ef33ebdfd7930c22a1.
- **Freeze:** `sets freeze --allow-mixed --refcheck <that record>`, exit 0.
  - Freeze 0, 17 entries, manifest sha256 2eeaebe8d37cf0f60fda90bd9c71df4e828d35b51fc6411efc072d23e7e38def, made at commit 78ecff3f.
  - It cites the refcheck: pass_ true, failing {}, not_run [].
  - set.json sha256 after the freeze: a901500fd40bad3600386415614e276de3c872ad48856111f1aa01dec398efff.
- **G9-ref:** exit 0, "(a) True, (b) True, (c) True, (f) True: PASS; none", 29.7 s, at a8014b72, dirty False.
  - **(a)** It read the refcheck the freeze cites: files match the freeze, plans placed. Items passing: walks nominal, v0.3, v0.35, v0.4, v0.45; turns t90 0.35 and 0.5 left and right; stops one and two step. Not run: q90, t90_1 and walk/v0.5 (references not available). No failing reference, nothing still available.
  - **(b)** Nominal rung against the ladder: bob −0.053 %, head 0.0049°, tail −0.00061 cycle, toe-off −0.0011 rad (−0.36 %). v040 rung: bob −0.057 %, head 0.0027°, tail 0.00077 cycle, toe-off −0.0015 rad (−2.7 %). Limits are 5 %, 0.1°, 0.01 cycle, 0.02 rad and 10 %; the L1 hashes are equal on both rungs. fr05 is not in the ladder. All six walks are finite with 18 to 25 steps, and every played walk and turn item has at least 6 steps and 7 lift-offs.
  - **(c)** On nominal and v0.4 each degraded playback fails only its own term: bob×0.5 fails bob, neck 3° fails head, tail late fails tail, flat feet fails toe_off, tail tenth fails tail (amplitude 0.0125 against 0.125 rad). The unchanged playback and the 3-step cut both pass.
  - **(f)** v0.5 is not played by the bar; its screen block's sha256 matches the entry and it is within limits. Every available task's joins are within limit; the largest join is 0.00023 rad and the largest jump-to-step ratio 0.0227 (turn_0.35_right, v0.4 into the turn, neck_pitch 4.33e-05 against 0.00191 rad). The 2 s zero action is finite on every task.
  - Record `stage2/tb/gates/V6/75-M-H-10/g9_ref.json`: own sha256 field 862176970ba30e84cda3c8b32998d2235c375ae34d7d64b1ef23b65837b12a6d, file sha256 af627ba90a65ef83e5eb3009d36c44d4425c99be15f5d262ada8a24f58b4cfdc.
- **Commits:** 78ecff3f (refcheck), a8014b72 (freeze), 0a705c32 (G9-ref).

## 2. V6noF5 / 75-M-H-10
- **Verify:** "V6noF5 75-M-H-10: verify ok: 17 references, 11 not available, 0 failures". **Status:** "not frozen; lead-in start".
- **Refcheck:** exit 0, pass, placed. sha256 37389fa9011e1f4810df1c2d8668ad2553a3700459ecfad9e1ac93339e745e31.
- **Freeze:** freeze 0, 17 entries, manifest sha256 ff2e79b3695e0c69bb4f5dbbe0576e79d622923093260f0ed437a1025c82e15a, mixed_sources true, dirty false, at 17253051. It cites the refcheck (pass_ true). set.json sha256 20a4c351656a51e5cd84dfe5d76b82e783a2caeb2913bb2c898f6c5814db4062.
- **G9-ref:** exit 0, "(a) True, (b) True, (c) True, (f) True: PASS; none", at 5708dbf4, dirty False.
  - **(a)** Same items and outcome as V6.
  - **(b)** In the V6noF5 form, compared with V6 / 75-M-H-10's freeze 0: bob, head and tail differ by 0.0 on all six rungs. Toe-off is "not read" (rule L11); the walks' toe-off reads −0.0000 rad.
  - **(c)** Four degraded playbacks plus the tail tenth, each failing only its own term; no flat-feet playback on V6noF5.
  - **(f)** Largest join 0.000282 rad, largest ratio 0.0227; zero action finite.
  - Record `stage2/tb/gates/V6noF5/75-M-H-10/g9_ref.json`: own field d2bca1f17cf120192133100e280a2229ce4ace312870ac202f81b23ba12a31ec, file e8387ba1893c4be8cc5d0a6e33a64afc335eb12b376e0dde717d18da05e590c7.
- **Commits:** 17253051, 5708dbf4, 5de84bcb.

## 3. V6 / 75-M-H-10-X
- **Verify:** "V6 75-M-H-10-X: verify ok: 17 references, 11 not available, 0 failures". **Status:** "not frozen; lead-in start".
- **Refcheck:** exit 0, pass, placed. sha256 1b7308e961cc63ed95e29f23b4c84aecdfe05ebba977deb16a59e45ef9bb210c.
- **Freeze:** freeze 0, 17 entries, manifest sha256 77516680b66203cff886be35fc4527eb47222cc22017dab89d6176f94f5747ed, mixed_sources true, dirty false, at c007de81. set.json sha256 4fd9cebaf35a7eef2c81ea17920b16cc019ada7185bea2b9d32af203e6b2a106.
- **G9-ref:** exit 0, "(a) True, (b) True, (c) True, (f) True: PASS; none", at 84e71235, dirty False.
  - **(a)** Same items and outcome as V6.
  - **(b)** In the -X form, compared with V6 / 75-M-H-10's freeze 0: all four terms differ by 0.0 on all six rungs.
  - **(c)** All five degraded playbacks fail only their own term, flat feet included.
  - **(f)** Largest join 0.000223 rad, largest ratio 0.0227; zero action finite.
  - Record `stage2/tb/gates/V6/75-M-H-10-X/g9_ref.json`: own field 8efa5e7eb72dad59bd9dc9951adda2958d118b073bc7c85c77e3a44852679c56, file 940701f4323ce679ba6d4b4d9708273bc3bee3a93ceeb09e1cc701203b018b19.
- **Commits:** c007de81, 84e71235, 73d828e7.

After the last commit, verify on each set reads "verify ok: 17 references, 11 not available, 0 failures; freeze 0", and status reads "freeze 0; lead-in start".

## What I did not do
- **Nothing pushed or merged:** branch `wt/wp-p42` is at 73d828e7. The main checkout is still at dd431e6c, and `git status` there printed nothing. wp-l is still at 8bc47346.
- **Chain markers:** I ran the commands by hand, not through `run_trackb.sh`, so it has no done-markers for V1. Running the chain would run G9-ref on these three again and rewrite the records.
- **Scope:** no other V-step ran (G8a and later), and I changed no code.

My logs, the summary script and the commit message files are in `/tmp/claude-0/-home-user-mesozoic-labs/2dca1c00-13f3-5853-acf6-8c3e3e04feb6/scratchpad/p42freeze/`.
