"""Plan B (async executive) extraction over the 13 uc7 runs. Read-only.

For every executive decision k it records: step, latency (skill_decision api_latency_s,
cross-checked against runner.log model_call lines), decision segments, the executed
fragment's per-segment outcome (last 'executed' event of that fragment in
episode.events.jsonl), the fragment's executed steps and wall-clock span.
Writes B/decisions.json.
"""
import json, os, sys

R = '/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'decisions.json')
RUNS = ['t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002',
        't6_009', 't6_011', 't7_001', 't8_001', 't9_001']


def runner_latencies(name):
    out = []
    path = R + name + '_launch/runner.log'
    if not os.path.exists(path):
        return None
    with open(path, errors='replace') as f:
        for line in f:
            line = line.strip()
            if line.startswith('{'):
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get('event') == 'model_call':
                    out.append((ev.get('step'), ev.get('latency_s'), ev.get('wall_s')))
    return out


def frag_of(e):
    fb = e.get('feedback') or {}
    pf = fb.get('primitive_feedback') or {}
    return fb.get('primitive'), pf.get('fragment'), pf


def main():
    res = {}
    for name in RUNS:
        d = R + name
        summ = json.load(open(d + '/summary.json'))
        hist = json.load(open(d + '/user_check_plan_history.json'))
        decisions = sorted(hist['model_decisions'], key=lambda x: x['decision'])
        sds, execd = [], {}
        terminal = None
        with open(d + '/episode.events.jsonl') as f:
            for line in f:
                e = json.loads(line)
                ev = e.get('event')
                if ev == 'skill_decision':
                    sds.append({'step': e['step'], 'lat': e.get('api_latency_s') or 0.0,
                                'wall': e['wall_elapsed_s'], 'dec_lat': e.get('decision_latency_s')})
                elif ev == 'executed':
                    prim, frag, pf = frag_of(e)
                    execd[e['step']] = {'wall': e['wall_elapsed_s'], 'prim': prim,
                                        'kind': e.get('execution_kind'),
                                        'frag': frag, 'pf_keys': sorted(pf)[:12]}
                elif ev == 'terminal':
                    terminal = {'step': e['step'], 'status': e.get('status'), 'wall': e['wall_elapsed_s']}
        rl = runner_latencies(name)
        if len(decisions) != len(sds):
            print('MISMATCH decisions vs skill_decision', name, len(decisions), len(sds), file=sys.stderr)
        tc = summ['timing_components']
        steps_total = summ['steps']
        rows = []
        for k, (dec, sd) in enumerate(zip(decisions, sds)):
            assert dec['step'] == sd['step'], (name, k)
            s0 = sd['step']
            s1 = sds[k + 1]['step'] if k + 1 < len(sds) else steps_total
            a = dec['action']
            segs = []
            if a['type'] == 'execute':
                for s in a['execution']['segments']:
                    segs.append({'skill': s['skill'], 'profile': s['profile'], 'gripper': s['gripper'],
                                 'max_steps': s['max_steps'], 'oa': s['observation_after'],
                                 'xyz': s['xyz_m'], 'q': s['quaternion_wxyz'] is not None})
            # executed steps for this decision's primitive: steps s0..s1-1
            frag_steps = [st for st in range(s0, s1) if st in execd]
            last = execd[frag_steps[-1]] if frag_steps else None
            frag = last['frag'] if last else None
            seg_out = []
            frag_status = frag_reason = None
            if frag is not None:
                frag_status, frag_reason = frag.get('status'), frag.get('reason')
                for sg in frag.get('segments') or []:
                    req = sg.get('request') or {}
                    seg_out.append({'index': sg.get('index'), 'skill': sg.get('skill'),
                                    'profile': req.get('profile'), 'oa': req.get('observation_after'),
                                    'actions': sg.get('actions'), 'status': sg.get('status'),
                                    'reason': sg.get('reason'),
                                    'pos_err': sg.get('position_error_m'),
                                    'target': sg.get('target_xyz_m'),
                                    'entry': (sg.get('entry_pose_wxyz') or [None]*3)[:3],
                                    'last': (sg.get('last_pose_wxyz') or [None]*3)[:3]})
            # kinds of executed steps (fragment vs direct/hold)
            kinds = {}
            for st in frag_steps:
                kinds[execd[st]['kind']] = kinds.get(execd[st]['kind'], 0) + 1
            wall_first = execd[frag_steps[0]]['wall'] if frag_steps else None
            wall_last = execd[frag_steps[-1]]['wall'] if frag_steps else None
            rows.append({'k': k, 'step': s0, 'next_step': s1, 'type': a['type'], 'lat': sd['lat'],
                         'wall_decision': sd['wall'], 'wall_first_exec': wall_first,
                         'wall_last_exec': wall_last, 'n_steps': len(frag_steps), 'kinds': kinds,
                         'segs': segs, 'seg_out': seg_out, 'frag_status': frag_status,
                         'frag_reason': frag_reason, 'plan': a['plan'][:600], 'why': a['why'][:400],
                         'belief': a['belief'][:600]})
        res[name] = {'summary': {'status': summ.get('status'), 'task_success': summ.get('task_success'),
                                 'v2_success': summ.get('v2_success'), 'steps': steps_total,
                                 'timing': tc, 'elapsed_s': summ.get('elapsed_s'),
                                 'control_elapsed_s': summ.get('control_elapsed_s'),
                                 'prep_s': (summ.get('user_check') or {}).get('preparation_elapsed_s'),
                                 'approval_to_claim_s': (summ.get('user_check') or {}).get('approval_to_claim_s')},
                     'runner_latencies': rl, 'terminal': terminal, 'rows': rows}
        lat_ev = sum(r['lat'] for r in rows)
        lat_rl = sum(x[1] for x in rl) if rl else None
        print(name, 'decisions', len(rows), 'lat(events)', round(lat_ev, 1), 'lat(runner.log)',
              None if lat_rl is None else round(lat_rl, 1), 'model_pipeline_s', round(tc['model_pipeline_s'], 1))
    json.dump(res, open(OUT, 'w'), indent=1)


if __name__ == '__main__':
    main()
