import json, sys
path = sys.argv[1]
with open(path, 'rb') as f:
    for l in f:
        if not l.startswith(b'{"event": "action"'):
            continue
        e = json.loads(l)
        dg = e.get('diagnostics') or {}
        if not dg.get("api_calls_in_act"):
            continue
        fb = dg.get('feedback') or {}
        ev = fb.get('evidence') or {}
        segs = ev.get('segments') or []
        seg_info = []
        for s in segs[:6]:
            if isinstance(s, dict):
                seg_info.append((s.get('skill'), s.get('status'), s.get('reason')))
        print('step', e['step'], 'type', (dg['routing'].get('params') or {}).get('type'),
              '| prev fb', fb.get('action'), fb.get('status'), fb.get('reason'), 'exec', fb.get('executed_steps'),
              'gap', None if ev.get('measured_gap_mm') is None else round(ev['measured_gap_mm'], 1),
              'detail', str(ev.get('detail'))[:160], 'segs', seg_info)
