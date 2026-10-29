import json,glob,re,collections
R="/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
OUT="/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v4/"
ts="t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
kinds=collections.Counter()
bad=0; total_imgs=0; new_imgs=0
for t in ts:
    acts=json.load(open(OUT+"act_rows_%s.json"%t))
    for a in acts: kinds[a["kind"]]+=1
    call_steps={a["step"] for a in acts if a["calls"]}
    for f in sorted(glob.glob(R+t+"/requests/*.request.json")):
        r=json.load(open(f))
        step=r["request_meta"]["step"]
        labels=[]
        for m in r["payload"]["input"]:
            c=m.get("content")
            if not isinstance(c,list): continue
            for p in c:
                if p.get("type")=="input_text":
                    mm=re.match(r"(External|Wrist) upright RGB, observation step (\d+)$",p["text"])
                    if mm: labels.append(int(mm.group(2)))
        total_imgs+=sum(1 for x in labels)
        new_imgs+=sum(1 for x in labels if x==step)
        b=[x for x in labels if x not in call_steps]
        if b: bad+=1; print("non-decision image step",t,step,b[:5])
print("execution kinds",kinds)
print("image labels in requests:",total_imgs,"labels at current step:",new_imgs,"requests with non-decision-step images:",bad)
