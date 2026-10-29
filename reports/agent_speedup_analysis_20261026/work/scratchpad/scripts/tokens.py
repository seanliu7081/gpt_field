import json, glob, os, math
R='/workspace/gpt_field/runs'
E='/workspace/gpt_field/results/speedup_evidence_20261026T055123Z/evaluations'
M=json.load(open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/runs_metrics.json'))
names=['t0_001','t1_001','t2_002','t3_003','t3_005','t4_001','t5_001','t5_002','t6_009','t6_011','t7_001','t8_001','t9_001']
paths={n:R+'/agent_speedup_v2_user_check_libero_10_'+n for n in names}
paths['today_A_trial_002']=E+'/trial_002/trial'; paths['today_B_trial_003']=E+'/trial_003/trial'
allrows=[]
for n,d in paths.items():
    lats=M[n]['runner_lat']
    rows=[]
    for i,f in enumerate(sorted(glob.glob(d+'/requests/*.response.json'))):
        r=json.load(open(f)); u=r.get('usage') or {}
        inp=u.get('input_tokens'); out=u.get('output_tokens'); rea=(u.get('output_tokens_details') or {}).get('reasoning_tokens'); cached=(u.get('input_tokens_details') or {}).get('cached_tokens')
        srv=(r.get('completed_at') or 0)-(r.get('created_at') or 0)
        lat=lats[i] if i<len(lats) else None
        rows.append((i+1,inp,cached,out,rea,srv,lat))
    allrows+= [(n,)+x for x in rows]
    first=rows[0]; rest=rows[1:]
    def mean(xs): xs=[x for x in xs if x is not None]; return sum(xs)/len(xs) if xs else None
    print(f"{n:17s} calls={len(rows):3d} first: in={first[1]} out={first[3]} reas={first[4]} lat={first[6]} | rest mean: in={mean([x[1] for x in rest]):.0f} out={mean([x[3] for x in rest]):.0f} reas={mean([x[4] for x in rest]):.0f} lat={mean([x[6] for x in rest]):.1f} srv={mean([x[5] for x in rest]):.1f} | last in={rows[-1][1]}")
# correlation latency vs reasoning tokens, vs input tokens (excluding first calls)
def corr(a,b):
    n=len(a); ma=sum(a)/n; mb=sum(b)/n
    sa=math.sqrt(sum((x-ma)**2 for x in a)); sb=math.sqrt(sum((y-mb)**2 for y in b))
    return sum((x-ma)*(y-mb) for x,y in zip(a,b))/(sa*sb)
rows=[r for r in allrows if r[7] is not None and r[1] is not None and r[1]>0 and not r[0].startswith('today')]
nf=[r for r in rows if r[1]!=1 and r[1+0]!=None and r[1]!=0 and r[1]]  # all
rest=[r for r in rows if r[1] is not None and r[0] and r[1]>0 and r[1]!=None and r[1] and r[1]]
rest=[r for r in rows if r[1] and r[1]>0 and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1]]
rest=[r for r in rows if r[1] is not None and r[1]>0 and r[1]]
rest=[r for r in rows if r[1]>0 and r[1] and r[1]]
rest2=[r for r in rows if r[1]>0 and r[1] and r[1]]
nonfirst=[r for r in rows if r[1] and r[1]>0 and r[1] and r[1] and r[1] and (r[1]) and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1]]
nonfirst=[r for r in rows if r[1]>0 and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1] and r[1]]
nonfirst=[r for r in rows if r[1]>0 and r[1] and r[1] and r[1]]
nonfirst=[r for r in rows if r[1]>0]
nonfirst=[r for r in nonfirst if r[1]!=None]
nonfirst=[r for r in nonfirst if r[1] and r[1]>0 and r[1] and r[1]]
nonfirst=[r for r in nonfirst if r[1] and r[1]>0]
nonfirst=[r for r in nonfirst if r[1]]
nonfirst=[r for r in nonfirst if r[1]]
nonfirst=[r for r in nonfirst if r[1]]
nf=[r for r in allrows if not r[0].startswith('today') and r[1]!=1]
nf=[r for r in nf if r[7] is not None and r[2] is not None]
nf=[r for r in nf if r[1]!=1]
nf=[r for r in allrows if not r[0].startswith('today') and r[7] is not None and r[2] is not None and r[1]!=1]
print('N non-first calls', len(nf))
print('corr(latency, reasoning_tokens)=%.2f'%corr([r[7] for r in nf],[r[5] for r in nf]))
print('corr(latency, output_tokens)=%.2f'%corr([r[7] for r in nf],[r[4] for r in nf]))
print('corr(latency, input_tokens)=%.2f'%corr([r[7] for r in nf],[r[2] for r in nf]))
first=[r for r in allrows if not r[0].startswith('today') and r[1]==1]
print('first-call latency mean %.1f s (n=%d); non-first mean %.1f s'%(sum(r[7] for r in first)/len(first),len(first),sum(r[7] for r in nf)/len(nf)))
print('reasoning tokens mean non-first %.0f, first %.0f'%(sum(r[5] for r in nf)/len(nf), sum(r[5] for r in first)/len(first)))
# latency per 1000 reasoning tokens - simple regression
xs=[r[5] for r in nf]; ys=[r[7] for r in nf]
mx=sum(xs)/len(xs); my=sum(ys)/len(ys)
b=sum((x-mx)*(y-my) for x,y in zip(xs,ys))/sum((x-mx)**2 for x in xs); a=my-b*mx
print('latency ~ %.2f + %.4f * reasoning_tokens'%(a,b))
json.dump(allrows, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/call_tokens.json','w'))
