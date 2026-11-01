    def manip(self, label):
        """PC C8: manip_check.py's record at <panels>/<label>/c8/manip_check.json, or (None, why) when it is absent,
        unreadable, not a manipulation-check record, or not made with the criteria's seeds 3042, 3043 and bank 500."""
        p = self.path(label, C.MANIP_KIND)
        f = None if p is None else p / C.MANIP_RECORD
        if f is None or not f.is_file():
            return None, f"no manipulation check at {p}"
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return None, f"manipulation check unreadable at {f}: {exc}"
        if not isinstance(rec, dict) or rec.get("schema") != C.MANIP_SCHEMA:
            return None, f"{f}: not a {C.MANIP_SCHEMA} record"
        if list(rec.get("seeds") or []) != list(C.MANIP_SEEDS) or rec.get("bank_limit") != C.MANIP_BANK_LIMIT:
            return None, (f"{f}: seeds {rec.get('seeds')} / bank_limit {rec.get('bank_limit')}, not the criteria's "
                          f"{list(C.MANIP_SEEDS)} / {C.MANIP_BANK_LIMIT}")
        return rec, None

