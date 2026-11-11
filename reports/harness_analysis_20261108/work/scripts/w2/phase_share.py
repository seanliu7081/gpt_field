import pickle, collections
OUT = "/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data"
LABELS = ["T04_t100","T04_20mm","T04_default","T08_t100","T08_stoveoff","drawer001","drawer003"]
tot = collections.Counter(); bycls = collections.defaultdict(collections.Counter)
for lab in LABELS:
    S = pickle.load(open(f"{OUT}/{lab}_steps.pkl","rb"))
    for s, st in S.items():
        inc = st.get("forb_inc", 0)
        if not inc: continue
        f = st.get("frag")
        if st.get("fb_primitive") == "execute_fragment" and f:
            cat = ("routed_transit" if f.get("routed") else
                   f"unrouted_{f.get('profile')}_{f.get('skill')}")
        else:
            cat = f"direct_{st.get('fb_primitive')}"
        tot[cat] += inc
        for k, v in st["inc"].items():
            if v: bycls[cat][k] += v
T = sum(tot.values())
for k, v in tot.most_common():
    print(f"{k:45s} {v:6d} {100*v/T:5.1f}%  {dict(bycls[k])}")
print("total", T)
