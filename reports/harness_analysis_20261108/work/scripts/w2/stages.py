"""Collect every fragment segment (stage) with routing info + per-step measured poses, from raw traces."""
import json, pickle, collections, sys
RUNS = "/workspace/gpt_field/runs"
OUT = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
sys.path.insert(0, "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts")
from extract import RUNLIST
def collect(label):
    path = f"{RUNS}/{RUNLIST[label]}/trial/trace.jsonl"
    frags = collections.OrderedDict()   # key: global start step of the fragment
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") != "feedback":
                continue
            fb = d["feedback"]; pf = fb.get("primitive_feedback") or {}
            frag = pf.get("fragment")
            if not frag:
                # failed-at-entry fragments are reported via evidence in later 'action' diag; handle below
                continue
            g = d["step"]; k = g - frag["observed_at_fragment_step"] + 1
            frags[k] = {"start_global": k, "last_global": g, "frag": frag, "evidence": pf}
    return frags
def collect_entry_failures(label):
    """Fragments that fail before issuing any action appear only as feedback.evidence in the next action record."""
    path = f"{RUNS}/{RUNLIST[label]}/trial/trace.jsonl"
    out = []
    seen = set()
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            if d.get("event") != "action":
                continue
            fbk = (d.get("diagnostics") or {}).get("feedback") or {}
            ev = fbk.get("evidence") or {}
            if fbk.get("action") == "execute" and ev.get("executed_steps") == 0 and ev.get("segments"):
                key = (fbk.get("observed_at_step"), ev.get("reason"))
                if key in seen: continue
                seen.add(key)
                out.append({"observed_at_step": fbk.get("observed_at_step"), "reason": ev.get("reason"),
                            "segments": ev.get("segments"), "detail": ev.get("detail")})
    return out
if __name__ == "__main__":
    res = {}
    for lab in RUNLIST:
        frags = collect(lab)
        ent = collect_entry_failures(lab)
        res[lab] = {"frags": frags, "entry_fail": ent}
        nseg = sum(len(f["frag"]["segments"]) for f in frags.values())
        routed = [(k, s) for k, f in frags.items() for s in f["frag"]["segments"] if s.get("routing")]
        print(lab, "fragments", len(frags), "segments", nseg, "routed segs", len(routed), "entry-fail fragments", len(ent),
              collections.Counter(e["reason"] for e in ent))
    pickle.dump(res, open(f"{OUT}/stages.pkl", "wb"))
