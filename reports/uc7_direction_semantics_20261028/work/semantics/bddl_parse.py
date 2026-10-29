"""Minimal s-expression BDDL parser (read-only on LIBERO tree)."""
import re, glob, os, json
L = "/workspace/heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files"
OFFS = {  # workspace_offset (x,y,z) per problem class, from envs/problems/*.py
 "LIBERO_Tabletop_Manipulation": (0,0,0.90),
 "LIBERO_Kitchen_Tabletop_Manipulation": (0.0,0,0.90),
 "LIBERO_Living_Room_Tabletop_Manipulation": (0,0,0.41),
 "LIBERO_Study_Tabletop_Manipulation": (-0.2,0,0.867),
 "LIBERO_Floor_Manipulation": None,
}
def tokenize(s):
    s = re.sub(r";[^\n]*", "", s)
    return re.findall(r"\(|\)|[^\s()]+", s)
def parse(tokens):
    stack=[[]]
    for t in tokens:
        if t=="(": stack.append([])
        elif t==")":
            x=stack.pop(); stack[-1].append(x)
        else: stack[-1].append(t)
    return stack[0][0]
def load(f):
    txt=open(f).read()
    lang = re.search(r"\(:language ([^)]*)\)", txt).group(1).strip()
    tree = parse(tokenize(txt))
    d = {"file": os.path.relpath(f, L), "language": lang}
    for item in tree:
        if isinstance(item, list) and item and item[0]=="problem": d["problem"]=item[1]
        if isinstance(item, list) and item and isinstance(item[0], str) and item[0].startswith(":"):
            key=item[0]
            if key==":regions":
                regs={}
                for r in item[1:]:
                    name=r[0]; info={}
                    for sub in r[1:]:
                        k=sub[0]
                        if k==":target": info["target"]=sub[1]
                        elif k==":ranges": info["ranges"]=[[float(v) for v in rr] for rr in sub[1]]
                        elif k==":yaw_rotation": info["yaw"]=[[float(v) for v in rr] for rr in sub[1]]
                        elif k==":rgba": info["rgba"]=sub[1]
                    regs[name]=info
                d["regions"]=regs
            elif key in (":fixtures", ":objects"):
                m={}; toks=item[1:]
                i=0
                while i < len(toks):
                    # pattern: name1 name2 - category
                    names=[]
                    while toks[i]!="-": names.append(toks[i]); i+=1
                    cat=toks[i+1]; i+=2
                    for n in names: m[n]=cat
                d[key[1:]]=m
            elif key==":init": d["init"]=item[1:]
            elif key==":goal": d["goal"]=item[1]
            elif key==":obj_of_interest": d["obj_of_interest"]=item[1:]
    d["offset"]=OFFS.get(d.get("problem"))
    return d
def all_bddls():
    return [load(f) for f in sorted(glob.glob(L+"/*/*.bddl"))]
if __name__=="__main__":
    ds=all_bddls(); print(len(ds))
    print(json.dumps(ds[0], indent=1)[:1500])
