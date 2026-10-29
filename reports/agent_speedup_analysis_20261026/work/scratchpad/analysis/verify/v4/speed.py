import json,math,sys,os
from concurrent.futures import ProcessPoolExecutor
R="/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
OUT="/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v4/"
ts="t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
LIM={"transit":0.02,"contact":0.005}
def norm(v): return math.sqrt(sum(x*x for x in v))
def proc(t):
    rows=[]
    prim_counts={}
    nfb=0
    prev_key=None
    with open(R+t+"/trace.jsonl") as f:
        for line in f:
            if not line.startswith('{"event": "feedback"'): continue
            r=json.loads(line)
            fb=r["feedback"]; nfb+=1
            pf=fb.get("primitive_feedback") or {}
            prim=fb.get("primitive"); prim_counts[prim]=prim_counts.get(prim,0)+1
            frag=pf.get("fragment")
            if frag is None:
                rows.append(dict(t=t,step=fb["step"],prim=prim,skill=None,profile=None)); continue
            s=pf["step"]
            act=None
            for seg in frag["segments"]:
                e=seg["entry_at_fragment_step"]; en=seg.get("end_at_fragment_step")
                if e < s and (en is None or en >= s):
                    act=seg; break
            b=fb["before_state"]; a=fb["after_state"]; na=fb["normalized_action"]
            cmd=[x*0.05 for x in na[:3]]
            real=[a[i]-b[i] for i in range(3)]
            mtd=fb.get("measured_translation_delta_m")
            rec=dict(t=t,step=fb["step"],fs=s,prim=prim,
                     seg=act["index"] if act else None, skill=act["skill"] if act else None,
                     profile=act["request"]["profile"] if act else None,
                     segi=(s-act["entry_at_fragment_step"]) if act else None,
                     cmd=norm(cmd), real=norm(real), mtd_diff=(norm([mtd[i]-real[i] for i in range(3)]) if mtd else None),
                     rot_cmd=norm(na[3:6])*0.5, grip=na[6],
                     seg_reason=act.get("reason") if act else None,
                     done=pf.get("done"), reason=pf.get("reason"))
            # dot product of realized with commanded direction
            c=norm(cmd)
            rec["real_along"]= (sum(cmd[i]*real[i] for i in range(3))/c) if c>0 else 0.0
            rows.append(rec)
    with open(OUT+"speed_rows_%s.json"%t,"w") as g: json.dump(rows,g)
    return t,nfb,prim_counts
if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=13) as ex:
        for res in ex.map(proc,ts): print(res, flush=True)
