import re, sys, json
R='/workspace/gpt_field/runs'
pat=re.compile(r'"observation/joint_position": \[([^\]]*)\]')
for n in sys.argv[1:]:
    f=R+'/agent_speedup_v2_user_check_libero_10_'+n+'/trace.jsonl'
    q4=[]
    with open(f) as fh:
        for line in fh:
            if not line.startswith('{"event": "executed"'): continue
            # executed events contain 'observation' (post-step) ; take the last joint_position occurrence
            ms=pat.findall(line)
            if not ms: continue
            vals=[float(x) for x in ms[-1].split(',')]
            q4.append(vals[3])
    lim=-0.0698
    near=[x for x in q4 if x>lim-0.02]
    at=[x for x in q4 if x>lim-0.005]
    print(n, 'steps', len(q4), 'q4 within 0.02 rad of upper limit:', len(near), '(%.1f%%)'%(100*len(near)/max(len(q4),1)), ' within 0.005:', len(at), ' max q4 %.4f'%max(q4))
