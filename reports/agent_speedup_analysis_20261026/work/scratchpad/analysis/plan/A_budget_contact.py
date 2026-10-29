"""Plan A: optional contact-cap lever (5 -> 8 or 10 mm) on top of T4 central (ESTIMATE; python3 -I)."""
import json, statistics as st, importlib.util, io, contextlib
P = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/"
spec = importlib.util.spec_from_file_location("b", P + "A_budget.py")
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
csav = {}
for r in b.MM["rows"]:
    if r["prof"] != "contact" or not r["pred"]:
        continue
    d = csav.setdefault(r["run"], {"c8": 0.0, "c10": 0.0})
    for col in ("c8", "c10"):
        d[col] += r["actual"] - r["actual"] * r[col] / r["pred"]
for speed in (1.0, 1.25):
    for col in (None, "c8", "c10"):
        vals = {}
        for run in b.SWEEP:
            d = b.service(run, 4, 1, speed)
            extra = csav.get(run, {}).get(col, 0.0) if col else 0.0
            vals[run] = d["total"] - extra * d["per_step"]
        ranked = sorted((v if b.R[r]["raw"] else float("inf")) for r, v in vals.items())
        print("x%.2f T4 + contact %-5s 5th-best %6.1f median10 %6.1f S300 %d/10" % (
            speed, col, ranked[4], (ranked[4] + ranked[5]) / 2, sum(1 for r, v in vals.items() if b.R[r]["raw"] and v <= 300)))
