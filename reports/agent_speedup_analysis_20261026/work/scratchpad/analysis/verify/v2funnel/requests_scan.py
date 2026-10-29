import json, sys, os, glob
ROOT='/workspace/gpt_field/runs'
RUNS = {
 't1_005':'agent_speedup_v3_t1_005',
 't1_006':'agent_speedup_v3_t1_006',
 't1_007':'agent_speedup_v3_t1_007',
 't1_008':'agent_speedup_v3_t1_008',
 't1_009':'agent_speedup_v3_t1_009',
 't1_010':'agent_speedup_v3_t1_010',
 't7_001':'agent_speedup_v3_t7_001',
 'g5_001':'agent_speedup_v3_g5_001',
 'g5_002':'agent_speedup_v3_g5_002',
 't1_011':'agent_speedup_v3_t1_011',
 't1_012':'agent_speedup_v3_t1_012',
 'base_a3':'v3_completion_baseline_20261026T133633Z/attempts/attempt_003/execution',
 'base_a1':'v3_completion_baseline_20261026T133633Z/attempts/attempt_001/execution',
}
def find_key(d, key, path=''):
    """yield (path, value) for every occurrence of key"""
    if isinstance(d, dict):
        for k,v in d.items():
            p=path+'.'+k
            if k==key: yield p, v
            yield from find_key(v, key, p)
    elif isinstance(d, list):
        for i,v in enumerate(d):
            yield from find_key(v, key, path+f'[{i}]')

def model_output(resp):
    for o in resp.get('output',[]) or []:
        if o.get('type')=='message':
            for c in o.get('content',[]):
                if c.get('type')=='output_text':
                    try: return json.loads(c['text'])
                    except Exception as e: return {'_unparsed': c['text'][:200]}
    return None

which = sys.argv[1:] or list(RUNS)
for k in which:
    rdir=os.path.join(ROOT,RUNS[k],'requests')
    reqs=sorted(glob.glob(rdir+'/*.request.json'))
    print(f'===== {k}  n_requests={len(reqs)}')
    for rq in reqs:
        base=rq[:-len('.request.json')]
        try: req=json.load(open(rq))
        except Exception as e: print('  bad req',rq,e); continue
        ctx=req.get('original_context') or {}
        src='orig'
        if not ctx:
            try:
                last=req['payload']['input'][-1]
                txt=last['content'][0]['text']
                ctx=json.loads(txt); src='wire'
            except Exception as e:
                ctx={}; src='none:'+str(e)[:40]
        step=ctx.get('step')
        mode=req.get('mode')
        eas=list(find_key(ctx,'evidence_acquisition'))
        ea_desc=[]
        for p,v in eas:
            if isinstance(v,dict) and 'candidates' in v:
                ea_desc.append(f"{p.split('.')[-2]}:{len(v['candidates'])}cand/{v.get('reason')}")
            else:
                ea_desc.append(f"{p}:{str(v)[:80]}")
        resp=None; out=None
        if os.path.exists(base+'.response.json'):
            try:
                resp=json.load(open(base+'.response.json')); out=model_output(resp)
            except Exception as e: out={'_err':str(e)}
        res=None
        if os.path.exists(base+'.result.json'):
            try: res=json.load(open(base+'.result.json')).get('status')
            except Exception: res='?'
        t = out.get('type') if isinstance(out,dict) else None
        why = (out.get('why') or '')[:160] if isinstance(out,dict) else ''
        print(f"  #{req.get('request_ordinal')} {os.path.basename(base)[:6]} src={src} step={step} res={res} type={t} EA={ea_desc}")
        if t in ('escalate','acquire_evidence'):
            print('      why:', why)
