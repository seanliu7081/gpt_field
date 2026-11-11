"""Independent classification of what preceded each executive call (uc7.1 family), from raw trace.jsonl.

For each action row carrying diagnostics.api_calls (an executive call), find the most recent feedback row before it
in file order and read feedback.primitive_feedback.fragment {status, reason}; also record the previous call's step
(to detect zero-motion vetoes: a call followed by another call one action later, with the intervening action
being a framework hold) and feedback.primitive_feedback reason fields.
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

RUNS = Path('/workspace/gpt_field/runs')
UC71 = {
    'H1_T04': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
    'H1_T08': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
    'H3_T04': 'foundation_vision_t04_can_obstacle_20261105_001',
    'H3_T08': 'foundation_vision_t08_bottle_obstacle_20261105_001',
    'T20_T04': 'agent_speedup_v2_uc71_can_obstacle_t04_20261103_001',
    'DEF_T04': 'agent_speedup_v2_uc71_default_routing_t04_20261103_001',
    'SOFF_T08': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
    'DRW_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
    'DRW_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}


def main():
    grand = Counter()
    out = {}
    for key, run in UC71.items():
        rows = [json.loads(l) for l in open(RUNS / run / 'trial' / 'trace.jsonl')]
        last_fb = None
        calls = []
        for r in rows:
            ev = r.get('event')
            if ev == 'feedback':
                last_fb = r
            elif ev == 'action':
                d = r.get('diagnostics') or {}
                api = d.get('api_calls') or []
                if api:
                    fr = None
                    pfr = None
                    if last_fb is not None:
                        pf = (last_fb.get('feedback') or {}).get('primitive_feedback') or {}
                        fr = pf.get('fragment')
                        pfr = (pf.get('status'), pf.get('reason'))
                    calls.append({
                        'step': r['step'],
                        'n_api': len(api),
                        'latency': sum(c.get('latency_s') or 0 for c in api),
                        'prev_fb_step': last_fb.get('step') if last_fb else None,
                        'frag_status': fr.get('status') if fr else None,
                        'frag_reason': fr.get('reason') if fr else None,
                        'frag_executed_steps': fr.get('executed_steps') if fr else None,
                        'prim': pfr,
                        'exec_kind': d.get('execution_kind'),
                    })
        # zero-motion vetoes: call at step s followed by another call at step s+1
        steps = [c['step'] for c in calls]
        for i, c in enumerate(calls):
            c['next_call_gap'] = (calls[i + 1]['step'] - c['step']) if i + 1 < len(calls) else None
        cat = Counter()
        for i, c in enumerate(calls):
            if c['prev_fb_step'] is None:
                k = 'first_after_plan'
            else:
                k = f"{c['frag_status']}|{c['frag_reason']}"
            cat[k] += 1
        vetoes = [(c['step'], calls[i + 1]['step'], calls[i + 1]['frag_status'], calls[i + 1]['frag_reason'], calls[i + 1]['frag_executed_steps'])
                  for i, c in enumerate(calls) if c['next_call_gap'] == 1]
        out[key] = {'n_calls': len(calls), 'categories': dict(cat), 'vetoes': vetoes, 'calls': calls}
        grand.update(cat)
        print(f"== {key} n_calls={len(calls)}")
        for k, v in sorted(cat.items(), key=lambda x: -x[1]):
            print(f"     {v:3d}  {k}")
        print(f"     next-call-at-step+1 (zero-motion candidates): {vetoes}")
    print('== GRAND', sum(grand.values()))
    for k, v in sorted(grand.items(), key=lambda x: -x[1]):
        print(f"     {v:3d}  {k}")
    json.dump(out, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/v_triggers.json', 'w'), indent=1, default=str)


if __name__ == '__main__':
    main()
