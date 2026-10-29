"""Plan A: aggregate statistics over A_calls.json (uc7, 218 calls). Read-only; python3 -I."""
import json, statistics as st, collections

P = "/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/plan/"
C = json.load(open(P + "A_calls.json"))
SUCC = "t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t7_001 t8_001 t9_001".split()


def m(xs):
    xs = [x for x in xs if x is not None]
    return st.mean(xs) if xs else None


print("n calls", len(C))
first = [c for c in C if c["k"] == 1]
rest = [c for c in C if c["k"] > 1]
for name, g in (("first", first), ("rest", rest), ("all", C)):
    print(name, "n", len(g), "lat %.2f" % m([c["latency_s"] for c in g]), "reas %.0f" % m([c["reasoning_tokens"] for c in g]),
          "vis %.0f" % m([c["visible_tokens"] for c in g]), "out_chars %.0f" % m([c["out_chars"] for c in g]),
          "prose %.0f" % m([c["prose_chars"] for c in g]), "exec %.0f" % m([c["exec_chars"] for c in g]),
          "inp %.0f" % m([c["input_tokens"] for c in g]), "obs_text %.0f" % m([c["obs_text_chars"] for c in g]))
tot_out = sum(c["out_chars"] for c in C)
tot_prose = sum(c["prose_chars"] for c in C)
print("prose share of visible output chars: %.3f" % (tot_prose / tot_out))
# triggers
trig = collections.Counter()
for c in C:
    if c["k"] == 1:
        trig["initial"] += 1
        continue
    t = c["trigger"] or "none"
    a, s, r = (t.split("/") + [None, None, None])[:3]
    trig[r] += 1
print("triggers (all 13 runs):", trig.most_common())
trig_s = collections.Counter()
for c in C:
    if c["run"] not in SUCC:
        continue
    if c["k"] == 1:
        trig_s["initial"] += 1
        continue
    t = c["trigger"] or "none"
    trig_s[t.split("/")[2] if t.count("/") >= 2 else t] += 1
print("triggers (11 raw successes):", trig_s.most_common(), sum(trig_s.values()))
# latency after stall vs after complete
by = collections.defaultdict(list)
for c in C:
    if c["k"] > 1 and c["trigger"]:
        by[c["trigger"].split("/")[2]].append(c["latency_s"])
for r, v in by.items():
    print("latency after %-28s n %3d mean %.2f" % (r, len(v), st.mean(v)))
# observation text composition (mean chars per call) for non-first calls
keys = collections.defaultdict(list)
for c in rest:
    for k, v in c["obs_ctx"].items():
        keys["ctx." + k].append(v)
    for k, v in c["obs_top"].items():
        keys[k].append(v)
print("non-first obs text mean chars %.0f" % m([c["obs_text_chars"] for c in rest]))
for k, v in sorted(keys.items(), key=lambda kv: -st.mean(kv[1])):
    print("  %-28s mean %7.0f  share %.3f  n %d" % (k, st.mean(v), st.mean(v) / m([c["obs_text_chars"] for c in rest]), len(v)))
# segments per execute
ex = [c for c in C if c["type"] == "execute"]
print("execute n", len(ex), "segments mean %.2f" % m([c["n_segments"] for c in ex]),
      "dist", collections.Counter(c["n_segments"] for c in ex))
print("types", collections.Counter(c["type"] for c in C))
# reasoning tail
big = [c for c in C if (c["reasoning_tokens"] or 0) > 300]
print("calls with >300 reasoning tokens", len(big), "share of latency %.3f" % (sum(c["latency_s"] for c in big) / sum(c["latency_s"] for c in C)))
# per-run mean visible tokens / prose chars
for r in SUCC + ["t6_009", "t6_011"]:
    g = [c for c in C if c["run"] == r]
    print(r, "calls", len(g), "lat %.1f" % m([c["latency_s"] for c in g]), "vis %.0f" % m([c["visible_tokens"] for c in g]),
          "reas %.0f" % m([c["reasoning_tokens"] for c in g]), "prose %.0f" % m([c["prose_chars"] for c in g]),
          "inp_k %.1f" % (m([c["input_tokens"] for c in g]) / 1000))
