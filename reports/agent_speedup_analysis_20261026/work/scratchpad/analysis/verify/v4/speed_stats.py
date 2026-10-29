import json,statistics as st
OUT="/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v4/"
ts="t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
succ=[t for t in ts if t not in ("t6_009","t6_011")]
LIM={"transit":0.02,"contact":0.005}
MOTION=("move_to_pose","move_relative","approach")
def load(tl):
    rows=[]
    for t in tl: rows+=json.load(open(OUT+"speed_rows_%s.json"%t))
    return rows
for label,tl in (("11 raw successes",succ),("all 13",ts)):
    rows=load(tl)
    print("=====",label,"steps",len(rows))
    from collections import Counter
    print("prim",Counter(r["prim"] for r in rows))
    print("skill",Counter(r["skill"] for r in rows))
    print("mtd max diff", max((r["mtd_diff"] or 0) for r in rows if r.get("mtd_diff") is not None))
    mot=[r for r in rows if r["skill"] in MOTION]
    print("motion steps",len(mot), "by profile",Counter(r["profile"] for r in mot))
    path={p:sum(r["real"] for r in mot if r["profile"]==p) for p in LIM}
    tot=sum(path.values())
    n={p:sum(1 for r in mot if r["profile"]==p) for p in LIM}
    print("path m",{p:round(v,3) for p,v in path.items()},"total %.3f"%tot,"contact share of path %.4f"%(path["contact"]/tot),"contact share of motion steps %.4f"%(n["contact"]/len(mot)))
    allpath=sum(r.get("real",0) or 0 for r in rows)
    print("path over ALL steps incl gripper/wait/local_motion %.3f"%allpath)
    for p,lim in LIM.items():
        cap=[r for r in mot if r["profile"]==p and r["cmd"]>=lim*(1-1e-6)]
        over=[r for r in mot if r["profile"]==p and r["cmd"]>lim*(1+1e-6)]
        if not cap: continue
        re=[r["real"] for r in cap]; ra=[r["real_along"] for r in cap]
        print(f"  {p}: capped n={len(cap)} (over-cap n={len(over)}) mean real {1000*st.mean(re):.3f} mm, median {1000*st.median(re):.3f} mm, ratio mean {st.mean(re)/lim:.3f}; along-cmd mean {1000*st.mean(ra):.3f} mm")
        # steady state: segment action index >=k
        for k in (3,5,10):
            cs=[r for r in cap if (r["segi"] or 0)>=k]
            re2=[r["real"] for r in cs]
            if re2: print(f"     segi>={k}: n={len(cs)} mean {1000*st.mean(re2):.3f} mm median {1000*st.median(re2):.3f} ratio {st.mean(re2)/lim:.3f}")
        # steady: previous step in same run & same seg also capped
        idx={(r["t"],r["step"]):r for r in mot}
        cs=[r for r in cap if (r["t"],r["step"]-1) in idx and idx[(r["t"],r["step"]-1)]["seg"]==r["seg"] and idx[(r["t"],r["step"]-1)]["cmd"]>=lim*(1-1e-6) and idx[(r["t"],r["step"]-1)]["fs"]==r["fs"]-1]
        re3=[r["real"] for r in cs]
        if re3: print(f"     prev-also-capped: n={len(cs)} mean {1000*st.mean(re3):.3f} mm median {1000*st.median(re3):.3f} ratio {st.mean(re3)/lim:.3f}")
    # gripper/wait
    print("non-motion steps", Counter((r["prim"],r["skill"]) for r in rows if r["skill"] not in MOTION))
