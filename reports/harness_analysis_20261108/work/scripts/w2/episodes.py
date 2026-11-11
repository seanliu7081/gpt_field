import pickle, json, collections, sys
OUT = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
LABELS = ["T04_t100","T04_20mm","T04_default","T08_t100","T08_stoveoff","drawer001","drawer003","H3_T04","H3_T08"]
def phase_of(st):
    f = st.get("frag")
    if st.get("fb_primitive") == "execute_fragment" and f:
        prof = f.get("profile"); sk = f.get("skill")
        routed = f.get("routed")
        sv = None
        if routed and f.get("routing"):
            sv = f["routing"].get("straight_path_valid")
        return f"frag:{sk}/{prof}/{'routed' if routed else 'unrouted'}" + (f"/sv={sv}" if routed else "") + f"/g={f.get('gripper')}"
    return f"{st.get('fb_primitive')}"
allep = []
for lab in LABELS:
    S = pickle.load(open(f"{OUT}/{lab}_steps.pkl","rb"))
    steps = sorted(S)
    eps = []
    cur = None
    for s in steps:
        inc = S[s].get("forb_inc", 0)
        if inc > 0:
            if cur and cur["end"] == s - 1:
                cur["end"] = s; cur["steps"].append(s)
            else:
                cur = {"run": lab, "start": s, "end": s, "steps": [s]}
                eps.append(cur)
    for e in eps:
        cls = collections.Counter(); pairs = collections.Counter(); phases = collections.Counter()
        for s in e["steps"]:
            for k, v in S[s]["inc"].items():
                cls[k] += v
            for (c, p), v in S[s]["pair_inc"].items():
                if c in ("robot_scene","robot_nonfinger_manipulated_object","robot_object","manipulated_object_scene"):
                    pairs[(c, "~".join(x.split(":")[1] for x in p))] += v
            phases[phase_of(S[s])] += S[s]["forb_inc"]
        e["n"] = len(e["steps"]); e["samples"] = sum(cls.values())
        e["class"] = cls.most_common(1)[0][0]
        e["classes"] = dict(cls)
        e["pair"] = pairs.most_common(1)[0][0][1] if pairs else None
        e["phase"] = phases.most_common(1)[0][0]
        e["phases"] = dict(phases)
        del e["steps"]
    allep.extend(eps)
    print(f"{lab}: {len(eps)} episodes, samples {sum(e['samples'] for e in eps)}")
    for e in eps:
        print(f"   {e['start']}-{e['end']} n={e['n']} s={e['samples']} {e['class']} pair={e['pair']} phase={e['phase']}  {'' if len(e['phases'])==1 else e['phases']}")
json.dump(allep, open(f"{OUT}/episodes_w2.json","w"), indent=1)
h1 = [e for e in allep if not e["run"].startswith("H3")]
print("H1 episodes", len(h1), "samples", sum(e["samples"] for e in h1))
import statistics
print("durations: min", min(e["n"] for e in h1), "max", max(e["n"] for e in h1), "median", statistics.median(e["n"] for e in h1))
