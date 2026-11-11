import json, sys, re
from pathlib import Path

def outputs(run):
    res = []
    for resp in sorted((Path(run)/'trial'/'requests').glob('*.response.json')):
        d = json.load(open(resp))
        text = None
        for item in d.get('output', []):
            if item.get('type') == 'message':
                for c in item.get('content', []):
                    if c.get('type') == 'output_text':
                        text = c['text']
        try:
            obj = json.loads(text) if text else None
        except Exception:
            obj = None
        res.append((resp.name, obj, text))
    return res

if __name__ == '__main__':
    pat = re.compile(sys.argv[2], re.I) if len(sys.argv) > 2 else None
    for name, obj, text in outputs(sys.argv[1]):
        if obj is None:
            print(name[:6], 'NO-OBJ'); continue
        om = obj.get('obstacle_map')
        segs = (obj.get('execution') or {}).get('segments') if isinstance(obj.get('execution'), dict) else None
        print(name[:6], obj.get('type'), '| belief:', obj.get('belief'))
        print('        plan:', obj.get('plan'), '| why:', obj.get('why'))
        if om is not None:
            print('        obstacle_map:', json.dumps(om))
        if segs:
            for s in segs:
                print('        seg:', json.dumps({k: s.get(k) for k in ('skill','xyz_m','delta_m','gripper','profile','max_steps','observation_after','route') if k in s}))
        if pat and text and pat.search(text):
            print('        MATCH:', [m.group(0) for m in pat.finditer(text)])
