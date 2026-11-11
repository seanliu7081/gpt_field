import pickle, json, collections, sys, numpy as np
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
from calls import calls
D = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
H1 = ["T04_t100","T04_20mm","T04_default","T08_t100","T08_stoveoff","drawer001","drawer003"]
SCALE = {"T04_20mm": .05, "T04_default": .05}
res = pickle.load(open(f"{D}/stages.pkl","rb"))
stalls = []
for lab in H1:
    S = pickle.load(open(f"{D}/{lab}_steps.pkl","rb"))
    C = {c["step"]: c for c in calls(lab)}
    for k, f in res[lab]["frags"].items():
        fr = f["frag"]
        if fr["reason"] != "segment_stall": continue
        end = f["last_global"]
        seg = fr["segments"][fr["current_segment_index"]]
        entry = k + seg["entry_at_fragment_step"]       # global step of first action in the segment (approx)
        # contact during the stalled segment
        forb = sum(S[s]["forb_inc"] for s in range(entry, end+1) if s in S)
        allowed = sum(sum(S[s]["inc_allowed"].values()) for s in range(entry, end+1) if s in S)
        # next call
        nxt = min([s for s in C if s > end - 1], default=None)
        stalls.append({"run": lab, "entry": entry, "end": end, "skill": seg["skill"], "profile": seg["request"]["profile"],
                       "routed": bool(seg.get("routing")), "pos_err": seg.get("position_error_m"), "forb": forb, "allowed": allowed,
                       "next_call": nxt, "next_lat": sum(C[nxt]["lat"]) if nxt is not None else None})
print("stall handoffs (H1):", len(stalls))
cat = collections.Counter()
for x in stalls:
    c = ("forbidden" if x["forb"] > 0 else "allowed_contact" if x["allowed"] > 0 else "near_target<10mm" if x["pos_err"] < .01 else "no_contact_far")
    x["cat"] = c; cat[c] += 1
print(cat)
calls_after = [x for x in stalls if x["next_call"] is not None and x["next_call"] - x["end"] <= 1]
print("followed directly by a model call:", len(calls_after), "total latency %.1f s" % sum(x["next_lat"] for x in calls_after))
print("by profile/routed:", collections.Counter((x["profile"], x["routed"]) for x in stalls))
pickle.dump(stalls, open(f"{D}/stalls.pkl","wb"))
