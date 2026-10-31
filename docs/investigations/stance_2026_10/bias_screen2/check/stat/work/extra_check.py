"""BS2 statistic lens, extra checks: marker identities behind SEC-G / SEC-MARGIN, the pins the STATUS rests on, the P4
atime snapshot, the paired falls, and where the spread-only chance of complete separation crosses 0.05.
Own code; reads only; stdout captured by the caller into check/stat/."""
import csv
import hashlib
import json

from scipy import integrate, optimize, stats
from pathlib import Path

_BS2 = Path(__file__).resolve().parents[3]                 # work -> stat -> check -> bias_screen2
S2 = str(_BS2.parents[1])                                   # bias_screen2 -> r10round -> the session scratchpad
BS2 = str(_BS2)
S, E = 2_097_152, 3_538_944
SEEDS = (2501, 2511, 2521, 2531)
NAMES = [f"{a}_{s}" for s in SEEDS for a in ("control", "critic_blind_tail3")]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = json.load(open(BS2 + "/APPROVED"))
    print("APPROVED sha256", sha256(BS2 + "/APPROVED"))
    print("criteria pin ok", sha256(BS2 + "/criteria.txt") == ap["criteria_sha256"],
          "; SCREEN pin ok", sha256(BS2 + "/SCREEN.json") == ap["screen_sha256"])
    for t, h in ap["tool_sha256"].items():
        print(f"  tool {t:16s} pin ok {sha256(BS2 + '/tools/' + t) == h}")
    for n in NAMES:
        rm = json.load(open(f"{BS2}/runs/{n}/run_meta.json"))
        mk = json.load(open(f"{BS2}/reads/{n}/markers.json"))
        ck = json.load(open(f"{BS2}/runs/{n}/check.json"))
        led = f"{BS2}/runs/{n}/01_stance/eval_episodes.csv"
        print(f"  {n:24s} run_meta criteria {rm['criteria_sha256'] == ap['criteria_sha256']} approved "
              f"{rm['approved_sha256'] == sha256(BS2 + '/APPROVED')} screen {rm['screen_sha256'] == ap['screen_sha256']} "
              f"tools {all(rm['tool_sha256'].get(k) == v for k, v in ap['tool_sha256'].items())}; markers reader "
              f"{mk['tool_sha256'] == ap['tool_sha256']['read_bs.py']}; check reader "
              f"{ck.get('tool_sha256', ck.get('reader_sha256')) == ap['tool_sha256']['read_bs.py']}; ledger unchanged since "
              f"read {mk['ledger']['sha256'] == sha256(led)}; probe unchanged {mk['probe']['sha256'] == sha256(mk['probe']['file'])}")
        # G = 5.16 x slope at every loop row; OWN seeds 5001-5010
        bad = []
        for r in mk["loop"]:
            P = r["OWN"]["P"]
            if abs(P["G"] - 5.16 * P["slope_per_deg"]) > 1e-12:
                bad.append((r["steps"], P["G"], 5.16 * P["slope_per_deg"]))
            seeds = [e["seed"] for e in r["OWN"]["episodes"]]
            if seeds != list(range(5001, 5011)):
                bad.append((r["steps"], "seeds", seeds))
        print(f"  {'':24s} G == 5.16 x OWN slope and OWN seeds 5001-5010 at all {len(mk['loop'])} loop rows: {not bad} {bad[:2]}")
    # P4 atime snapshot
    p4 = BS2 + "/logs/atimes_at_ALL_DECIDED.tsv"
    with open(p4, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh, delimiter="\t"))
    head = rows[0]
    print("P4 snapshot columns", head, "rows", len(rows) - 1)
    ip, ir = head.index("path"), head.index("rearmed_unread")
    zeros = [r[ip] for r in rows[1:] if r[ir] == "0"]
    queue = tuple(NAMES)

    def of_queue(p, top):
        parts = p.split("/")
        return len(parts) > 1 and parts[0] == top and parts[1] in queue
    runs_all = [r[ip] for r in rows[1:] if of_queue(r[ip], "runs")]
    runs0 = [p for p in zeros if of_queue(p, "runs") and not p.endswith(("/run_meta.json", "/check.json"))]
    print(f"  rearmed_unread 0: {len(zeros)} rows in all; queue runs/ files listed {len(runs_all)}; of them 0 other than "
          f"run_meta.json/check.json: {len(runs0)} {runs0[:5]}")
    tmp_all = [r[ip] for r in rows[1:] if of_queue(r[ip], "tmp")]
    tmp0 = [p for p in zeros if of_queue(p, "tmp")]
    att_all = [r[ip] for r in rows[1:] if r[ip].startswith("logs/") and ".attempt" in r[ip]]
    att0 = [p for p in zeros if p.startswith("logs/") and ".attempt" in p]
    print(f"  queue tmp/ files listed {len(tmp_all)}, with 0: {len(tmp0)} {tmp0[:3]}; attempt logs listed {len(att_all)}, "
          f"with 0: {len(att0)} {att0[:3]}")
    # the paired falls at 2,359,296 (seed 2521)
    for n in ("control_2521", "critic_blind_tail3_2521"):
        with open(f"{BS2}/runs/{n}/01_stance/eval_episodes.csv", newline="", encoding="utf-8") as fh:
            rr = [r for r in csv.DictReader(fh) if float(r["timesteps"]) == 2_359_296]
        print(f"  {n}: evaluation 2,359,296 episodes (episode, env, length, cause): "
              f"{[(r['episode'], r['env'], r['length'], r['cause']) for r in rr if r['length'] != '1000']}")

    def p_sep(ratio):
        f = lambda z: 4 * stats.norm.pdf(z) * stats.norm.cdf(z) ** 3 * stats.norm.sf(z / ratio) ** 4
        return integrate.quad(f, -12, 12, epsabs=1e-13, epsrel=1e-11, limit=400)[0]
    r05 = optimize.brentq(lambda r: p_sep(r) - 0.05, 7.0, 20.0, xtol=1e-6)
    print(f"spread-only, equal means: P(complete separation) reaches 0.05 at an SD ratio of {r05:.2f}")


if __name__ == "__main__":
    main()
