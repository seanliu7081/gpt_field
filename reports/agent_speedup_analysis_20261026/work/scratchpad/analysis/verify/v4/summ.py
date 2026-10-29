import json,sys
R="/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
ts="t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
tot={}
rows=[]
for t in ts:
    s=json.load(open(R+t+"/summary.json"))
    tc=s.get("timing_components",{})
    r=dict(t=t,steps=s["steps"],task_success=s["task_success"],v2=s.get("v2_success"),calls=s["api_calls"],
           sim=tc.get("simulator_step_s"),vid=tc.get("video_render_s"),dec=tc.get("decision_s"),model=tc.get("model_pipeline_s"),
           rs=tc.get("run_state_s"),obs=tc.get("observe_transition_s"),ctrl=s.get("control_elapsed_s"),ctrl_nov=s.get("control_without_video_s"))
    rows.append(r)
    print(t, r["steps"], r["task_success"], r["v2"], r["calls"], "sim=%.1f vid=%.1f dec=%.1f model=%.1f rs=%.1f obs=%.1f ctrl=%.1f"%(r["sim"],r["vid"],r["dec"],r["model"],r["rs"],r["obs"],r["ctrl"]),
          "sim/step=%.4f vid/step=%.4f"%(r["sim"]/r["steps"],r["vid"]/r["steps"]))
S=sum(r["steps"] for r in rows)
print("total steps",S,"calls",sum(r["calls"] for r in rows))
print("pooled sim/step %.4f vid/step %.4f"%(sum(r["sim"] for r in rows)/S, sum(r["vid"] for r in rows)/S))
succ=[r for r in rows if r["task_success"]]
print("raw successes",len(succ),[r["t"] for r in rows if not r["task_success"]],"calls in raw succ",sum(r["calls"] for r in succ),"steps in succ",sum(r["steps"] for r in succ))
v2s=[r for r in rows if r["v2"]]
print("v2 successes",len(v2s),[r["t"] for r in v2s])
Ss=sum(r["steps"] for r in succ)
print("succ pooled sim/step %.4f vid/step %.4f"%(sum(r["sim"] for r in succ)/Ss, sum(r["vid"] for r in succ)/Ss))
print("steps per call (all) %.2f, succ %.2f"%(S/sum(r["calls"] for r in rows), Ss/sum(r["calls"] for r in succ)))
