"""Plan A: user-decision scenarios on top of tier T4 (ESTIMATES; read-only; python3 -I).
U1 reasoning effort high->medium modelled as -50% reasoning tokens at 0.0392 s/token (uc7 fit, model_io.md 1.2);
U2 first executive call overlapped with (excluded) human review wait; C2x fewer non-first calls."""
import json, statistics as st, importlib.util, sys
P = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/"
spec = importlib.util.spec_from_file_location("b", P + "A_budget.py")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
R, C, SWEEP = b.R, b.C, b.SWEEP
REAS = 0.0392
def scen(run, speed, effort=False, hide_first=False, fewer=0, lvl=1):
    d = b.service(run, 4, lvl, 1.0)
    cs = b.calls_of(run)
    model = d["model"]
    if effort:
        model -= sum(REAS * 0.5 * (c["reasoning_tokens"] or 0) for c in cs if c["k"] not in b.VETO_REPLACEMENT.get(run, []))
    if hide_first:
        model -= max(4.0, cs[0]["latency_s"] - b.LEVELS["prose_s_per_call"][lvl])
    nonfirst = [c["latency_s"] for c in cs if c["k"] != 1]
    model -= fewer * st.mean(nonfirst)
    return d["total"] - d["model"] + speed * model
for speed in (1.0, 1.25, 1.5):
    print("== speed x%.2f" % speed)
    for name, kw in (("T4 central", {}), ("T4 + U1 effort medium", dict(effort=True)),
                     ("T4 + U2 first call during review", dict(hide_first=True)),
                     ("T4 + 2 fewer calls", dict(fewer=2)), ("T4 + 3 fewer calls", dict(fewer=3)),
                     ("T4 + U1 + 2 fewer calls", dict(effort=True, fewer=2)),
                     ("T4 + U1 + U2", dict(effort=True, hide_first=True))):
        vals = {r: scen(r, speed, **kw) for r in SWEEP}
        ranked = sorted((v if R[r]["raw"] else float("inf")) for r, v in vals.items())
        n = sum(1 for r, v in vals.items() if R[r]["raw"] and v <= 300)
        print("  %-36s 5th-best %6.1f  median10 %6.1f  S300 %d/10" % (name, ranked[4], (ranked[4] + ranked[5]) / 2, n))
