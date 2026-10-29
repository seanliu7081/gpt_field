import json, sys, collections
for path in sys.argv[1:]:
    prim = collections.Counter(); macro = collections.Counter(); ctrl=collections.Counter()
    steps_by_prim = collections.defaultdict(list)
    exec_kind = collections.Counter()
    n=0
    last_success=None
    with open(path) as f:
        for line in f:
            r=json.loads(line)
            if r.get('event')=='executed':
                n+=1
                p=r.get('primitive'); prim[p]+=1; macro[r.get('macro')]+=1; ctrl[r.get('controller')]+=1
                steps_by_prim[p].append(r.get('step'))
                last_success = r.get('success')
                if 'execution_kind' in r: exec_kind[r['execution_kind']]+=1
    print(path)
    print(' executed rows:', n, 'last success:', last_success)
    for p,c in prim.most_common():
        s=steps_by_prim[p]
        print('  primitive', p, c, '%.1f%%'%(100*c/n), 'steps min/max', min(s), max(s))
        # contiguous ranges
        rngs=[]; start=s[0]; prev=s[0]
        for x in s[1:]:
            if x!=prev+1:
                rngs.append((start,prev)); start=x
            prev=x
        rngs.append((start,prev))
        print('    ranges:', rngs[:30], '...' if len(rngs)>30 else '')
    print('  macro', dict(macro))
    print('  controller', dict(ctrl))
    print('  exec_kind', dict(exec_kind))
