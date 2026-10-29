import json,glob,os,re,collections
R="/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
OUT="/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v4/"
ts="t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
allrows=[]
for t in ts:
    d=R+t+"/requests/"
    reqs=sorted(glob.glob(d+"*.request.json"))
    for f in reqs:
        base=f[:-len(".request.json")]
        r=json.load(open(f))
        meta=r.get("request_meta",{})
        last=[m for m in r["payload"]["input"] if m.get("role")=="user"][-1]
        txt=[p["text"] for p in last["content"] if p.get("type")=="input_text"][0]
        try:
            ctx=json.loads(txt)
        except Exception as e:
            ctx=None
        lf=((ctx or {}).get("context") or {}).get("last_feedback")
        stage_out=(lf or {}).get("stage_outcomes") if isinstance(lf,dict) else None
        summ=(lf or {}).get("summary") if isinstance(lf,dict) and "summary" in lf else lf
        resp=json.load(open(base+".response.json")) if os.path.exists(base+".response.json") else None
        res=json.load(open(base+".result.json")) if os.path.exists(base+".result.json") else None
        server=None; dec=None
        if resp:
            if resp.get("created_at") and resp.get("completed_at"): server=resp["completed_at"]-resp["created_at"]
            for o in resp.get("output",[]):
                if o.get("type")=="message":
                    for c in o.get("content",[]):
                        if c.get("type")=="output_text":
                            try: dec=json.loads(c["text"])
                            except Exception: dec=None
        row=dict(t=t,file=os.path.basename(f),ordinal=r.get("request_ordinal"),attempt=r.get("attempt"),mode=r.get("mode"),
                 step=meta.get("step"),kind=meta.get("kind"),
                 trig_action=(summ or {}).get("action") if isinstance(summ,dict) else None,
                 trig_reason=(summ or {}).get("reason") if isinstance(summ,dict) else None,
                 trig_steps=(summ or {}).get("executed_steps") if isinstance(summ,dict) else None,
                 trig_status=(summ or {}).get("status") if isinstance(summ,dict) else None,
                 trig_obs_step=(summ or {}).get("observed_at_step") if isinstance(summ,dict) else None,
                 lf_keys=list(lf.keys())[:8] if isinstance(lf,dict) else None,
                 server_s=server,result_status=(res or {}).get("status"),
                 dec_type=(dec or {}).get("type"),
                 dec=dec, stage_out=stage_out)
        allrows.append(row)
json.dump(allrows,open(OUT+"calls_rows.json","w"))
c=collections.Counter((r["t"]) for r in allrows)
print("requests per run",dict(c), "total",len(allrows))
print("modes",collections.Counter(r["mode"] for r in allrows),"attempts",collections.Counter(r["attempt"] for r in allrows),"result",collections.Counter(r["result_status"] for r in allrows))
print("lf keys sample",collections.Counter(tuple(r["lf_keys"] or []) for r in allrows).most_common(5))
