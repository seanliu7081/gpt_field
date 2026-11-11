"""Extract compact per-step records from trace.jsonl (feedback rows) for a run."""
import json, sys, os, math
run = sys.argv[1]
out = sys.argv[2]
tr = os.path.join(run, 'trial', 'trace.jsonl')
rows = []
api_steps = {}
with open(tr) as f:
    for line in f:
        head = line[:30]
        if '"event": "executed"' in head or '"event": "reset"' in head:
            continue
        d = json.loads(line)
        ev = d.get('event')
        if ev == 'action':
            dg = d.get('diagnostics') or {}
            if dg.get('api_calls_in_act'):
                api_steps[d['step']] = dg.get('api_latency_s')
            continue
        if ev != 'feedback':
            continue
        fb = d['feedback']
        pf = fb.get('primitive_feedback') or {}
        frag = pf.get('fragment') or {}
        segs = frag.get('segments') or []
        ci = frag.get('current_segment_index')
        cur = segs[ci] if (ci is not None and ci < len(segs)) else {}
        req = cur.get('request') or {}
        rt = cur.get('route_tracking') or {}
        lm = pf.get('local_motion') or {}
        b = fb.get('before_state'); a = fb.get('after_state')
        disp = math.dist(b[:3], a[:3]) if b and a else None
        na = fb.get('normalized_action') or [0]*7
        rows.append({
            'step': d['step'], 'primitive': fb.get('primitive'),
            'seg_index': ci, 'skill': frag.get('current_skill') or req.get('skill'),
            'profile': req.get('profile'), 'max_steps': req.get('max_steps'),
            'obs_after': req.get('observation_after'),
            'seg_actions': frag.get('current_segment_actions'),
            'frag_status': frag.get('status'), 'frag_reason': frag.get('reason'),
            'pos_err': frag.get('position_error_m'), 'rot_err': frag.get('orientation_error_rad'),
            'cmd_norm_norm': math.hypot(*na[:3]), 'rot_cmd': math.hypot(*na[3:6]), 'grip': na[6],
            'disp': disp, 'gap_mm': fb.get('finger_gap_mm'),
            'route_status': rt.get('status'), 'route_cmd_m': rt.get('command_norm_m'),
            'lm_status': lm.get('status'),
            'z': a[2] if a else None,
        })
json.dump({'rows': rows, 'api_steps': api_steps}, open(out, 'w'))
print(run, 'rows', len(rows), 'api', len(api_steps))
