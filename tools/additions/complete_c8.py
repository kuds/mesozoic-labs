    # C8 (PC): THE MANIPULATION CHECK on R2 (the final pair) and every J pair held, each record bound to the label's
    # C3-verified pair as C7's records are (binding_problems); absent or crashed: PENDING (a crash is rerun on the same
    # files and never fails C8); NOT IN FORCE on the run's own pair: FAIL (not of the study's configuration: LOST).
    c8_labels = ([str(s) for s in js] + ["final"]) if E is not None else []
    c8_res, c8_missing, c8_failed, c8_other = {}, [], [], []
    for lab in c8_labels:
        rec, why = panels.manip(lab)
        if rec is not None:
            bad = binding_problems(C.record_binding(rec), lab, files, E, run_id)
            if bad:
                c8_other.append(f"C8 {lab}: recorded on another file or as a validation input: {bad}")
                rec, why = None, f"C8 {lab}: not a check of this run's C3-verified pair ({'; '.join(bad)})"
        c8_res[lab] = None if rec is None else dict(verdict=rec.get("verdict"), checks=rec.get("checks"))
        if rec is None:
            c8_missing.append(why)
        elif rec.get("verdict") != "IN FORCE":
            c8_failed.append(f"C8 {lab}: {rec.get('verdict')} {json.dumps(rec.get('checks'), default=str)[:300]}")
    c8_name = "C8 the manipulation check (R2 and every J pair): phase in force, actor independent, obs 6 read"
    if E is None:
        out["C8"] = fact(c8_name, None, None, "E unknown")
    elif c8_failed:
        out["C8"] = fact(c8_name, False, dict(labels=c8_labels, failed=c8_failed, missing=c8_missing),
                         "FAILS: the phase is not in force or the actor reads it: not of the study's configuration "
                         "(LOST, printed with D-16)")
    else:
        out["C8"] = fact(c8_name, True if not c8_missing else None,
                         dict(labels=c8_labels, results=c8_res, missing=c8_missing),
                         None if not c8_missing else "PENDING: a manipulation check absent (run manip_check.py; a "
                         "crash is rerun on the same files, never a failure)")
    out["c8_other_file"] = c8_other
