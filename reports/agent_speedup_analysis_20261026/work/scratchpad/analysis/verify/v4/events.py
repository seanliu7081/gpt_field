import json,collections
R="/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_"
ts="t0_001 t1_001 t2_002 t3_003 t3_005 t4_001 t5_001 t5_002 t6_009 t6_011 t7_001 t8_001 t9_001".split()
succ=[t for t in ts if t not in ("t6_009","t6_011")]
T=collections.Counter()
for t in ts:
    s=json.load(open(R+t+"/summary.json"))
    ev=collections.Counter(); frames=0; n=0
    with open(R+t+"/episode.events.jsonl") as f:
        for line in f:
            r=json.loads(line); n+=1
            ev[r.get("event")]+=1; frames+=r.get("frame_count") or 0
    st=s["steps"]
    print(t,"steps",st,"events",n,dict(ev),"events/step %.3f frames/step %.2f"%(n/st,frames/st))
    if t in succ:
        T["steps"]+=st; T["events"]+=n; T["frames"]+=frames
print("succ pooled events/step %.3f frames/step %.3f"%(T["events"]/T["steps"],T["frames"]/T["steps"]))
