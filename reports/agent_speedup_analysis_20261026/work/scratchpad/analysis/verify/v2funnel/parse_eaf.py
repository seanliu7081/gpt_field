import json, sys, os
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
def load_eaf(path):
    txt=open(path).read()
    key='EvidenceAcquisitionFailed: '
    i=txt.rfind(key)
    if i<0: return None, txt[:300]
    js=txt[i+len(key):].strip()
    return json.loads(js), None

def keys_tree(d, depth=0, maxd=2):
    out=[]
    if isinstance(d, dict):
        for k,v in d.items():
            t=type(v).__name__
            extra=''
            if isinstance(v,(list,)): extra=f' len={len(v)}'
            if isinstance(v,(str,int,float,bool)) or v is None: extra=' = '+repr(v)[:160]
            out.append('  '*depth+f'{k}: {t}{extra}')
            if isinstance(v, dict) and depth<maxd:
                out+=keys_tree(v, depth+1, maxd)
    return out

which = sys.argv[1:] or list(RUNS)
for k in which:
    p=os.path.join(ROOT,RUNS[k],'error_traceback.txt')
    if not os.path.exists(p):
        print('==',k,'NO error_traceback'); continue
    d,err=load_eaf(p)
    print('==',k, p)
    if d is None:
        print('  not EAF:', err.splitlines()[0]); continue
    print('\n'.join(keys_tree(d, 0, 1)))
