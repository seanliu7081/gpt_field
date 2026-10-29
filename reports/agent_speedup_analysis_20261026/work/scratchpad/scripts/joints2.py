import re, glob, json, os
R='/workspace/gpt_field/runs'
pat=re.compile(r'\\?"joint_position_rad\\?":\s*\[([^\]]*)\]')
uc7=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001']
tot=0; tot_near=0
for n in uc7:
    fs=sorted(glob.glob(R+'/agent_speedup_v2_user_check_libero_10_'+n+'/requests/*.request.json'))
    q4s=[]
    for f in fs:
        s=open(f).read()
        ms=pat.findall(s)
        if not ms: q4s.append(None); continue
        vals=[float(x) for x in ms[-1].split(',')]
        q4s.append(vals[3])
    near=[i+1 for i,q in enumerate(q4s) if q is not None and q>-0.0698-0.02]
    tot+=len([q for q in q4s if q is not None]); tot_near+=len(near)
    print(n, 'requests', len(fs), 'with q4', len([q for q in q4s if q is not None]), 'q4 within 0.02 rad of -0.0698 at request:', near)
print('total', tot, 'near-limit', tot_near)
