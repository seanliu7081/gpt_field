"""Re-implementation of the F5 proprioceptive blocked-motion rule: >=K consecutive steps with >=2 mm commanded
translation and <0.3 mm measured progress along the command. Counter resets when the active stage changes."""
import pickle, json, collections, statistics, sys, numpy as np
D = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
H1 = ["T04_t100","T04_20mm","T04_default","T08_t100","T08_stoveoff","drawer001","drawer003"]
SCALE = {"T04_20mm": .05, "T04_default": .05}
K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
CMD, PROG = .002, .0003
eps = [e for e in json.load(open(f"{D}/episodes_w2.json")) if e["run"] in H1]
stalls = pickle.load(open(f"{D}/stalls.pkl", "rb"))
fires_all = {}; fp = collections.Counter(); moving_free = collections.Counter(); fp_pre_stall = 0
for lab in H1:
    S = pickle.load(open(f"{D}/{lab}_steps.pkl", "rb")); sc = SCALE.get(lab, .10)
    cnt, prev_key, fires = 0, None, []
    for s in sorted(S):
        st = S[s]
        na = st.get("norm_action"); md = st.get("measured_delta")
        if na is None or md is None: cnt = 0; continue
        f = st.get("frag") or {}
        key = (st.get("fb_primitive"), f.get("seg_index"), f.get("entry_step"), f.get("target") and tuple(f["target"]))
        pass
        cmd = np.asarray(na[:3]) * sc; n = np.linalg.norm(cmd)
        if n >= CMD:
            prog = float(np.asarray(md) @ (cmd / n))
            cnt = cnt + 1 if prog < PROG else 0
        else:
            cnt = 0
        fired = cnt == K
        if fired: fires.append(s)
        contact_free = st.get("forb_inc", 0) == 0 and sum(st.get("inc_allowed", {}).values()) == 0
        if contact_free and n >= CMD:
            moving_free[lab] += 1
            if cnt >= K: fp[lab] += 1
    fires_all[lab] = fires
fs = {lab: set(v) for lab, v in fires_all.items()}
# (a) episodes
hit, delays, after = 0, [], 0
for e in eps:
    S = pickle.load(open(f"{D}/{e['run']}_steps.pkl", "rb"))
    ins = [s for s in fires_all[e["run"]] if e["start"] <= s <= e["end"]]
    # also count a fire that precedes onset by <= 0? only inside
    if ins:
        hit += 1; delays.append(ins[0] - e["start"])
        after += sum(S[s]["forb_inc"] for s in range(ins[0] + 1, e["end"] + 1))
tot = sum(e["samples"] for e in eps)
print(f"K={K}: fires inside {hit}/{len(eps)} contiguous episodes; median delay {statistics.median(delays) if delays else None} steps; samples after fire {after}/{tot}")
# (b) stalled segments
inside, lead = 0, []
for x in stalls:
    ins = [s for s in fires_all[x["run"]] if x["entry"] <= s <= x["end"]]
    if ins: inside += 1; lead.append(x["end"] - ins[0])
print(f"   fires inside {inside}/{len(stalls)} stalled segments; median lead before stall {statistics.median(lead) if lead else None} steps; total lead {sum(lead)} steps")
print(f"   contact-free commanded-moving steps {sum(moving_free.values())}, steps with rule active (cnt>=K) {sum(fp.values())} -> {100*sum(fp.values())/max(1,sum(moving_free.values())):.1f}%  by run {dict(fp)}")
print(f"   total fires {sum(len(v) for v in fires_all.values())} by run { {k: len(v) for k, v in fires_all.items()} }")
# classify fire events
cls = collections.Counter(); extra = []
for lab, fl in fires_all.items():
    S = pickle.load(open(f"{D}/{lab}_steps.pkl", "rb"))
    ep_ranges = [(e["start"], e["end"]) for e in eps if e["run"] == lab]
    st_ranges = [(x["entry"], x["end"]) for x in stalls if x["run"] == lab]
    for s in fl:
        if any(a <= s <= b for a, b in ep_ranges): c = "in_forbidden_episode"
        elif any(a <= s <= b for a, b in st_ranges): c = "in_stalled_segment_no_forbidden"
        elif sum(S[s].get("inc_allowed", {}).values()) > 0: c = "allowed_contact_only"
        else:
            # does the stage end shortly anyway (arrival/other) ?
            c = "contact_free_not_stalled"; extra.append((lab, s, (S[s].get("frag") or {}).get("skill"), (S[s].get("frag") or {}).get("profile"), S[s].get("fb_primitive")))
        cls[c] += 1
print("   fire events by class:", dict(cls))
print("   contact-free, non-stalled fire events (would add a handoff):", len(extra), collections.Counter(x[0] for x in extra), collections.Counter((x[2], x[3], x[4]) for x in extra))
res = pickle.load(open(f"{D}/stages.pkl", "rb"))
ends = collections.Counter(); lst = []
for lab, s, sk, pr, prim in extra:
    S = pickle.load(open(f"{D}/{lab}_steps.pkl", "rb"))
    f = S[s].get("frag") or {}
    # walk forward until the segment index/entry changes, report the segment end reason
    key = (f.get("seg_index"), f.get("entry_step")); t = s; reason = None
    while t + 1 in S and (S[t+1].get("frag") or {}).get("seg_index") == key[0] and (S[t+1].get("frag") or {}).get("entry_step") == key[1] and S[t+1].get("fb_primitive") == "execute_fragment":
        t += 1
    ff = S[t].get("frag") or {}
    # find the segment record in the next feedback (seg_reason of the segment at t)
    reason = ff.get("seg_reason") or ff.get("reason")
    if reason is None and t + 1 in S:
        nf = S[t+1].get("frag") or {}
        reason = "advanced_to_next_segment" if nf else (S[t+1].get("fb_primitive"))
    ends[reason] += 1; lst.append((lab, s, t, sk, pr, reason, round(1000*(ff.get("pos_err") or 0),1)))
print("   end of segments containing extra fires:", dict(ends))
for x in lst: print("     ", x)
