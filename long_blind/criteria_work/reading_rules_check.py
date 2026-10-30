"""A self-check of criteria_v1.txt THE READING (Q1-Q4 and the overall line) on the dry-run sets the text names.
Not the scorer (the scorer adapter writes score_bs.py): this is the designer's own transcription of the counting rules,
run on hand-entered summaries of the three parents' measured classes (MEAS validate reports) and on synthetic sets, to
check that every line and word the text promises is reachable and that the dry-run expectations written in
VALIDATIONS hold.  Read only; prints.  Usage: PY -I -B reading_rules_check.py
"""

# A continuation summary: status COUNTED / STOPPED / LOST; measured (READ TO 12M needs COUNTED and measured);
# tail_class RAIL / UNREAD / PARKED / CLEAR; tail3_plus_alarm (a TAIL ALARM on tail_3 +); driven_tail3 (DRIVEN lists
# tail_3); gate PASS / FAIL / None; elsewhere: {actuator: set of sites among L, A, T, D}; lost_ledger_tail_alarm.


def read_to_12m(c):
    return c["status"] == "COUNTED" and c.get("measured", True)


def railed(c):
    if c["status"] in ("COUNTED", "STOPPED") and c.get("tail_class") == "RAIL":
        return True
    return c["status"] == "LOST" and c.get("lost_ledger_tail_alarm", False)


def habit_elsewhere(c):
    hab = {a: s for a, s in (c.get("elsewhere") or {}).items() if s}
    if c["status"] == "LOST":
        hab = {a: s & {"L", "A"} for a, s in hab.items() if s & {"L", "A"}}
    sat = any(s & {"L", "A", "T"} for s in hab.values())
    return bool(hab), sat


def q1(blind):
    k = c = p = dn = r = u = 0
    for b in blind:
        if railed(b):
            r += 1
        elif read_to_12m(b) and b["tail_class"] in ("CLEAR", "PARKED"):
            k += 1
            c += b["tail_class"] == "CLEAR"
            p += b["tail_class"] == "PARKED"
            dn += bool(b.get("driven_tail3"))
        else:
            u += 1
    if r >= 1:
        word = "NO"
    elif c == 3 and dn == 0:
        word = "YES"
    elif k == 3 and p + dn >= 1:
        word = "YES, WITH A PARK"
    else:
        word = "NOT SHOWN"
    return dict(k=k, c=c, p=p, dn=dn, r=r, u=u, word=word)


def q2(blind):
    g = x = m = 0
    for b in blind:
        if read_to_12m(b) and b.get("gate") == "PASS":
            g += 1
        elif b["status"] == "STOPPED" or (b["status"] == "COUNTED" and b.get("gate") == "FAIL"):
            x += 1
        else:
            m += 1
    word = "YES" if g == 3 else "PARTLY" if g in (1, 2) else "NO" if x >= 1 else "NOT SHOWN"
    return dict(g=g, x=x, m=m, word=word)


def q3(blind):
    h = hs = hd = f = u = 0
    for b in blind:
        has, sat = habit_elsewhere(b)
        if has:
            h += 1
            hs += sat
            hd += not sat
        elif read_to_12m(b):
            f += 1
        else:
            u += 1
    word = "NONE" if f == 3 else "HABIT ELSEWHERE" if hs >= 1 else "DRIVE ONLY" if h >= 1 else "NOT SHOWN"
    return dict(h=h, hs=hs, hd=hd, f=f, u=u, word=word)


def q4(ctrl):
    if ctrl["status"] in ("COUNTED", "STOPPED") and ctrl.get("tail_class") == "RAIL":
        return "YES" if ctrl.get("tail3_plus_alarm") else "RAILED, ANOTHER FORM"
    if ctrl["status"] == "LOST" and ctrl.get("lost_ledger_tail3_plus_alarm"):
        return "YES"
    if read_to_12m(ctrl) and ctrl["tail_class"] in ("PARKED", "CLEAR"):
        return f"NO ({ctrl['tail_class']})"
    return "NOT SHOWN"


def overall(ctrl, blind):
    a, b, c3, d = q1(blind), q2(blind), q3(blind), q4(ctrl)
    railed_ctrl = d in ("YES", "RAILED, ANOTHER FORM")
    if a["r"] >= 1:
        return 1, f"THE TAIL RAIL FORMED IN {a['r']} OF 3 CRITIC-BLIND CONTINUATIONS BY 12M ..."
    if a["u"] >= 1:
        return 2, f"NOT READ TO 12M: NO TAIL RAIL IN THE {a['k']} ... READ TO 12M; {a['u']} NOT READ TO 12M ..."
    if not railed_ctrl:
        return 3, f"NO TAIL RAIL IN 3 OF 3 ..., BUT THE CONTROL {d} ..."
    if a["c"] == 3 and a["dn"] == 0 and b["g"] == 3 and c3["f"] == 3:
        return 4, "BLINDING HELD TO 12M IN 3 OF 3 ..."
    items = []
    if a["p"]:
        items.append(f"{a['p']} PARKED")
    if a["dn"]:
        items.append(f"{a['dn']} WITH TAIL_3 DRIVEN NEGATIVE")
    if b["x"]:
        items.append(f"{b['x']} GATE FAIL")
    if c3["h"]:
        items.append(f"{c3['h']} WITH A {'DRIVE ONLY ' if c3['hs'] == 0 else ''}HABIT ON ANOTHER ACTUATOR")
    return 5, "NO TAIL RAIL IN 3 OF 3 ... WHILE THE CONTROL RAILED, WITH: " + ", ".join(items)


# The three parents as measured (MEAS validate reports; RUNLOG line 711)
S42 = dict(status="COUNTED", tail_class="RAIL", tail3_plus_alarm=True, gate="FAIL", elsewhere={})
S60 = dict(status="COUNTED", tail_class="PARKED", gate="PASS", elsewhere={})
S80 = dict(status="COUNTED", tail_class="CLEAR", gate="PASS",
           elsewhere={"neck_pitch_act": {"L", "A", "T", "D"}, "head_pitch_act": {"D"}})
CLEAN = dict(status="COUNTED", tail_class="CLEAR", gate="PASS", elsewhere={})
sets = {
    "(i)   parent s42 as all four": (S42, [S42, S42, S42]),
    "(ii)  control s42; blind s60, s80, s80": (S42, [S60, S80, S80]),
    "(iii) line 4: control s42; blind clean x3": (S42, [CLEAN, CLEAN, CLEAN]),
    "(iii) line 3: control CLEAR; blind clean x3": (CLEAN, [CLEAN, CLEAN, CLEAN]),
    "(iii) line 2: one blind STOPPED (hop watch) clean": (S42, [CLEAN, CLEAN, dict(status="STOPPED", tail_class="UNREAD",
                                                                             gate="FAIL", elsewhere={})]),
    "(iii) line 2: one blind LOST, no tail alarm": (S42, [CLEAN, CLEAN, dict(status="LOST", elsewhere={})]),
    "(iii) line 1: one blind LOST with a ledger TAIL ALARM": (S42, [CLEAN, CLEAN, dict(status="LOST",
                                                                                   lost_ledger_tail_alarm=True)]),
    "(iii) line 5: drive-only head park and a tail_3 negative drive": (S42, [
        dict(CLEAN, elsewhere={"head_pitch_act": {"D"}}), dict(CLEAN, driven_tail3=True), CLEAN]),
    "(iii) line 3: control LOST without alarm": (dict(status="LOST"), [CLEAN, CLEAN, CLEAN]),
    "(iii) line 4 with control RAILED, ANOTHER FORM": (dict(S42, tail3_plus_alarm=False), [CLEAN, CLEAN, CLEAN]),
    "(iii) Q2 NOT SHOWN: blind COUNTED but not measured": (S42, [dict(CLEAN, measured=False)] * 3),
}
for name, (ctrl, blind) in sets.items():
    line, text = overall(ctrl, blind)
    print(f"{name}\n   line {line}: {text}\n   Q1 {q1(blind)}\n   Q2 {q2(blind)}\n   Q3 {q3(blind)}\n   Q4 {q4(ctrl)}")
