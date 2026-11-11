"""Classify every execution call by what ended the previous decision (from the request context)."""
import json, glob, sys, os
from collections import Counter
base = '/workspace/gpt_field/runs'
runs = {
 't04_t100': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
 't08_t100': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
 't08_stoveoff': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
 'drawer_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
 'drawer_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
allres = {}
for key, run in runs.items():
    reqs = sorted(glob.glob(os.path.join(base, run, 'trial', 'requests', '*.request.json')))
    resps = [r.replace('.request.json', '.response.json') for r in reqs]
    prev_answer = None
    rows = []
    for i, (rq, rs) in enumerate(zip(reqs, resps)):
        d = json.load(open(rq))
        last = d['payload']['input'][-1]
        txt = [c['text'] for c in last['content'] if c['type'] == 'input_text'][0]
        fields = json.loads(txt)
        ctx = fields['context']
        lf = ctx.get('last_feedback') or {}
        summ = lf.get('summary', lf) if isinstance(lf, dict) else {}
        ev = (summ.get('evidence') or {}) if isinstance(summ, dict) else {}
        status = summ.get('status'); reason = summ.get('reason'); action = summ.get('action'); ex = summ.get('executed_steps')
        # answer
        r = json.load(open(rs))
        # find output text
        out_txt = None
        for item in r.get('output', []) if isinstance(r, dict) else []:
            if item.get('type') == 'message':
                for c in item.get('content', []):
                    if c.get('type') == 'output_text': out_txt = c.get('text')
        if out_txt is None and isinstance(r, dict) and 'response' in r:
            for item in r['response'].get('output', []):
                if item.get('type') == 'message':
                    for c in item.get('content', []):
                        if c.get('type') == 'output_text': out_txt = c.get('text')
        ans = json.loads(out_txt) if out_txt else {}
        segs = ((ans.get('execution') or {}).get('segments')) or []
        # classify trigger using prev answer
        if i == 0:
            cls = 'initial'
        else:
            pa = prev_answer or {}
            psegs = ((pa.get('execution') or {}).get('segments')) or []
            if reason in ('fragment_complete',):
                cls = 'complete_last_segment_checkpoint' if (psegs and psegs[-1].get('observation_after')) else 'complete_no_checkpoint'
            elif reason == 'observation_required':
                cls = 'mid_sequence_checkpoint'
            elif reason in ('segment_stall',):
                cls = 'stall'
            elif reason in ('target_envelope_rejected', 'route_unavailable', 'invalid_action_arguments'):
                cls = 'veto:' + reason
            elif action == 'direct_motion':
                cls = 'direct_motion_' + str(status)
            else:
                cls = str(action) + ':' + str(status) + ':' + str(reason)
        pending = len(ev.get('pending_segment_indices') or []) if isinstance(ev, dict) else None
        rows.append(dict(i=i+1, step=fields.get('step'), trigger=cls, prev_reason=reason, prev_status=status, prev_exec=ex, pending_discarded=pending,
                         type=ans.get('type'), nseg=len(segs), seg_steps=sum(s.get('max_steps') or 0 for s in segs),
                         last_obs_after=(segs[-1].get('observation_after') if segs else None),
                         any_mid_obs=any(s.get('observation_after') for s in segs[:-1]) if segs else None,
                         skills=[(s.get('skill')[:4], (s.get('profile') or '')[:1], s.get('max_steps'), int(bool(s.get('observation_after')))) for s in segs],
                         map_update=ans.get('obstacle_map') is not None, motion=ans.get('motion')))
        prev_answer = ans
    allres[key] = rows
    c = Counter(r['trigger'] for r in rows)
    print('=====', key, 'calls', len(rows), dict(c))
    for r in rows:
        print('  #%2d step %4s trig=%-34s prev_exec=%s pend_disc=%s -> %s nseg=%d steps=%d lastobs=%s %s%s' % (r['i'], r['step'], r['trigger'], r['prev_exec'], r['pending_discarded'], r['type'], r['nseg'], r['seg_steps'], r['last_obs_after'], r['skills'], ' MAP' if r['map_update'] else ''))
json.dump(allres, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/triggers.json', 'w'), indent=1)
