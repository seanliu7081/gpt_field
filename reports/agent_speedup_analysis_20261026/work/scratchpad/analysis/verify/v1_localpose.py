import json, sys, os, collections
for d in sys.argv[1:]:
    p = os.path.join(d, 'agent_stats.json')
    if not os.path.exists(p):
        print(d, 'no agent_stats'); continue
    s = json.load(open(p))
    vals = collections.Counter()
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == 'local_pose_revision':
                    vals[v] += 1
                walk(v)
        elif isinstance(o, list):
            for x in o: walk(x)
    walk(s)
    g = s.get('shared_geometry') or {}
    ents = g.get('entities') or {}
    if isinstance(ents, dict): ents_list = list(ents.values())
    else: ents_list = ents
    lpr = [(e.get('id'), e.get('local_pose_revision')) for e in ents_list if isinstance(e, dict)]
    nonzero = [x for x in lpr if x[1] not in (None, 0)]
    tri = collections.Counter(((e.get('triangulation') or {}).get('status'), (e.get('triangulation') or {}).get('reason')) for e in ents_list if isinstance(e, dict))
    evs = g.get('events') or []
    evk = collections.Counter()
    for e in evs:
        if isinstance(e, dict):
            evk[e.get('kind') or e.get('type') or e.get('event') or str(sorted(e.keys()))[:80]] += 1
    print(os.path.relpath(d, '/workspace/gpt_field'), '| entities', len(ents_list), '| local_pose_revision values (all paths)', dict(vals), '| shared_geometry nonzero', nonzero, '| triangulation', dict(tri))
    print('    shared_geometry events kinds', evk.most_common(15))
