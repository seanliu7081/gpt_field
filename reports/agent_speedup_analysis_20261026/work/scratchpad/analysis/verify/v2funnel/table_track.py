import json, sys, os, glob, math
ROOT='/workspace/gpt_field/runs'
RUNS = {f't1_{i:03d}':f'agent_speedup_v3_t1_{i:03d}' for i in range(1,13)}
RUNS['base_a3']='v3_completion_baseline_20261026T133633Z/attempts/attempt_003/execution'
RUNS['base_a1']='v3_completion_baseline_20261026T133633Z/attempts/attempt_001/execution'

def qrot(q, v):
    w,x,y,z=q
    # rotate v by quaternion q (wxyz)
    n=math.sqrt(w*w+x*x+y*y+z*z) or 1
    w,x,y,z=w/n,x/n,y/n,z/n
    R=[[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
       [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
       [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    return [sum(R[i][j]*v[j] for j in range(3)) for i in range(3)], R

def top_z(e):
    pos=e.get('position_m'); q=e.get('quaternion_wxyz') or [1,0,0,0]
    tops=[]
    for b in e.get('boxes') or []:
        c=b.get('center_local_m') or [0,0,0]; h=b.get('half_extents_m') or [0,0,0]
        bq=b.get('quaternion_wxyz') or [1,0,0,0]
        cw,R=qrot(q,c)
        # box orientation in world = q*bq ; compute world z-extent = sum |R_wb[2][j]|*h[j]
        # compose rotations
        _,Rb=qrot(bq,[0,0,0])
        Rwb=[[sum(R[i][k]*Rb[k][j] for k in range(3)) for j in range(3)] for i in range(3)]
        ext=sum(abs(Rwb[2][j])*h[j] for j in range(3))
        tops.append(pos[2]+cw[2]+ext)
    return max(tops) if tops else None

def model_output(resp):
    for o in resp.get('output',[]) or []:
        if o.get('type')=='message':
            for c in o.get('content',[]):
                if c.get('type')=='output_text':
                    try: return json.loads(c['text'])
                    except Exception: return None
    return None

def req_step(req):
    ctx=req.get('original_context')
    if ctx and 'step' in ctx: return ctx['step']
    try:
        return json.loads(req['payload']['input'][-1]['content'][0]['text']).get('step')
    except Exception: return None

which=sys.argv[1:] or list(RUNS)
for k in which:
    rdir=os.path.join(ROOT,RUNS[k],'requests')
    reqs=sorted(glob.glob(rdir+'/*.request.json'))
    print(f'===== {k} n={len(reqs)}')
    for rq in reqs:
        base=rq[:-len('.request.json')]
        if not os.path.exists(base+'.response.json'): continue
        req=json.load(open(rq)); st=req_step(req)
        out=model_output(json.load(open(base+'.response.json')))
        if not out: continue
        su=out.get('scene_update') or {}
        for e in su.get('entities',[]) or []:
            if e.get('role')=='support_surface' or 'table' in str(e.get('id')).lower():
                tz=top_z(e)
                print(f"  #{req.get('request_ordinal')} step={st} type={out.get('type')} id={e.get('id')} role={e.get('role')} pos_z={e.get('position_m',[None]*3)[2]} boxes={[ (b.get('center_local_m'),b.get('half_extents_m')) for b in e.get('boxes') or []]} top_z={tz:.4f} unc={e.get('uncertainty_m')} coverage={e.get('coverage')}")
        inv=su.get('invalidate_ids') or []; rem=su.get('remove_ids') or []
        if any('table' in str(x) for x in inv+rem):
            print(f"  #{req.get('request_ordinal')} step={st} invalidate={inv} remove={rem}")
