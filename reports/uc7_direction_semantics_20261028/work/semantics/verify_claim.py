"""Check claim: right=+y, left=-y, front=+x (toward agentview), back/behind=-x, for every BDDL whose language has a direction word."""
import re, sys, json, numpy as np
sys.path.insert(0, "/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/semantics")
from bddl_parse import all_bddls
U = {"right": np.array([0,1.]), "left": np.array([0,-1.]), "front": np.array([1.,0]), "back": np.array([-1.,0]), "behind": np.array([-1.,0])}
CADDY_SITES = {"right": (0.00631,-0.14339), "left": (0.00631,0.14551), "back": (0.03364,-0.00064), "front": (-0.03102,-0.00064)}
def flat(g): return g if isinstance(g,str) else "("+" ".join(flat(x) for x in g)+")"
def centre(d, full):
    regs = {r["target"]+"_"+n: r for n, r in d["regions"].items()}
    r = regs[full]; off = d["offset"] or (0,0,0); x0,y0,x1,y1 = r["ranges"][0]
    return np.array([(x0+x1)/2+off[0], (y0+y1)/2+off[1]]), r.get("yaw")
def init_pos(d, obj):
    for s in d["init"]:
        if s[0].lower()=="on" and s[1]==obj:
            try: return centre(d, s[2])[0]
            except KeyError: return None
    return None
rows=[]
for d in all_bddls():
    lang=d["language"]; L=lang.lower()
    words = re.findall(r"\b(left|right|front|back|behind|middle)\b", L)
    if not words: continue
    goal=flat(d["goal"]); objs=list(d.get("objects",{}))+list(d.get("fixtures",{}))
    verdicts=[]
    # Type C: compartment of the caddy (goal In ... desk_caddy_1_<w>_contain_region)
    m=re.search(r"desk_caddy_1_(\w+?)_contain_region", goal)
    if m and "compartment" in L and re.search(r"in the (\w+) compartment", L):
        w=re.search(r"in the (\w+) compartment", L).group(1)
        c, yaw = centre(d, "study_table_desk_caddy_init_region"); th=yaw[0][0]
        Rz=np.array([[np.cos(th),-np.sin(th)],[np.sin(th),np.cos(th)]]); s=Rz@np.array(CADDY_SITES[m.group(1)])
        ok = (s@U[w])>0 and w==m.group(1)
        verdicts.append(dict(kind="caddy compartment", word=w, goal_site=m.group(1), caddy_yaw_rad=round(th,4), site_offset_world=np.round(s,4).tolist(), ok=bool(ok)))
    # Type A: to the <w> of the <anchor> with table goal region
    m=re.search(r"to the (right|left|front|back) (?:of|compartment of) the ([\w ]+?)$", L)
    gm=re.search(r"\(On (\w+) (\w+_(?:right|left|front|behind|back)_region)\)", goal)
    if m and gm:
        w=m.group(1); reg=gm.group(2)
        rc,_=centre(d, reg)
        anc = [s for s in d["init"] if s[0].lower()=="on" and any(k in s[2] for k in ("plate_init","porcelain_mug_init","desk_caddy_init","stove_region"))]
        ac = centre(d, anc[0][2])[0] if anc else None
        dv = rc-ac
        verdicts.append(dict(kind="side-of-anchor goal region", word=w, region=reg, anchor_init=anc[0][2] if anc else None, delta=np.round(dv,3).tolist(),
                             ok=bool(dv@U[w] > 0 and abs(dv@np.array([-U[w][1],U[w][0]]))<1e-6),
                             note=("language says 'compartment' but goal is a TABLE region beside the caddy" if "compartment" in L else "")))
    # Type B: selection among same-category objects by direction word
    sel = re.findall(r"(?:the (right|left|middle) ([\w ]+?)(?: on| in|$| and))|(?:the ([\w ]+?) (?:on|at|in) the (right|left|front|back|middle))", L)
    for a,b,c_,dw in sel:
        w = a or dw
        if not w: continue
        # find goal-referenced objects and their init region names
        for s in d["init"]:
            if s[0].lower()!="on" or len(s)<3: continue
            if f"_{w}_init_region" in s[2] or f"_{w}_region" in s[2] and "plate" in s[2]:
                p = init_pos(d, s[1])
                sib = [(t[1], init_pos(d,t[1]), t[2]) for t in d["init"] if t[0].lower()=="on" and t[1]!=s[1] and t[1].rsplit("_",1)[0]==s[1].rsplit("_",1)[0]]
                if w=="middle":
                    ok=None
                else:
                    ok = all(((p-q)@U[w])>0 for _,q,_ in sib if q is not None)
                verdicts.append(dict(kind="select by direction", word=w, obj=s[1], region=s[2], pos=np.round(p,3).tolist(),
                                     siblings=[(n,np.round(q,3).tolist(),r) for n,q,r in sib if q is not None], in_goal=(s[1] in goal), ok=ok))
    rows.append(dict(file=d["file"], language=lang, goal=goal, checks=verdicts))
print(len(rows), "BDDLs with left/right/front/back/behind/middle")
for r in rows:
    print("==", r["file"].split("/")[0], r["language"], "|", r["goal"])
    for v in r["checks"]: print("    ", v)
    if not r["checks"]: print("     (no machine check; vertical/drawer or not applicable)")
json.dump(rows, open("/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/semantics/verify_claim.json","w"), indent=1, default=str)
