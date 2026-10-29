"""Counterfactual overlap of model latency with robot execution for uc7 runs.

Read-only over /workspace/gpt_field/runs. Writes JSON next to this script.
For each handoff (fragment k ends -> call k+1 -> decision k+1) we ask how much of
call k+1's latency could have been hidden if the leading part P of decision k+1
had been committed by decision k as a continuation after a checkpoint, so that P
runs while call k+1 is in flight.
"""
import json, os, sys, collections

R = '/workspace/gpt_field/runs'
A = '/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis'
OUT = os.path.join(A, 'plan', 'pipeline_cf.json')
RUNS = ['t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002',
        't6_009', 't6_011', 't7_001', 't8_001', 't9_001']
COMPACT_VIDEO_S_PER_STEP = 25.642 / 1019  # trial_003 measured video_render_s / steps
steps_metrics = json.load(open(os.path.join(A, 'steps_metrics.json')))


def load_run(name):
    d = os.path.join(R, 'agent_speedup_v2_user_check_libero_10_' + name)
    summ = json.load(open(os.path.join(d, 'summary.json')))
    hist = json.load(open(os.path.join(d, 'user_check_plan_history.json')))
    decisions = sorted(hist['model_decisions'], key=lambda x: x['decision'])
    sds, exec_wall, terminal = [], {}, None
    with open(os.path.join(d, 'episode.events.jsonl')) as f:
        for line in f:
            e = json.loads(line)
            ev = e.get('event')
            if ev == 'skill_decision':
                sds.append({'step': e['step'], 'lat': e.get('api_latency_s') or 0.0,
                            'wall': e['wall_elapsed_s'],
                            'log': os.path.basename(e.get('api_request_log') or '')[:6]})
            elif ev == 'executed':
                exec_wall[e['step']] = e['wall_elapsed_s']
            elif ev == 'terminal':
                terminal = e
    per_call = {c['call']: c for c in steps_metrics[name]['per_call']}
    return d, summ, decisions, sds, exec_wall, terminal, per_call


def leading(segs, pred):
    out = []
    for s in segs:
        if not pred(s):
            break
        out.append(s)
    return out


def main():
    res = {}
    agg = collections.Counter()
    for name in RUNS:
        d, summ, decisions, sds, exec_wall, terminal, per_call = load_run(name)
        tc = summ['timing_components']
        steps = summ['steps']
        c_legacy = (tc['simulator_step_s'] + tc['video_render_s'] + tc['run_state_s'] +
                    tc['observe_transition_s'] + tc['policy_compute_s']) / steps
        c_compact = (tc['simulator_step_s'] + tc['observe_transition_s'] + tc['policy_compute_s']) / steps + COMPACT_VIDEO_S_PER_STEP
        if len(decisions) != len(sds):
            print('MISMATCH', name, len(decisions), len(sds), file=sys.stderr)
        rows = []
        for k, (dec, sd) in enumerate(zip(decisions, sds)):
            if dec['step'] != sd['step']:
                print('STEP MISMATCH', name, k, dec['step'], sd['step'], file=sys.stderr)
            act = dec['action']
            segs_req = (act.get('execution') or {}).get('segments') or []
            pc = per_call.get(sd['log'])
            next_step = sds[k + 1]['step'] if k + 1 < len(sds) else steps
            rows.append({'k': k + 1, 'step': sd['step'], 'lat': sd['lat'], 'type': act['type'],
                         'n_steps': next_step - sd['step'],
                         'segs_req': [(s['skill'], s['profile'], s['gripper'], s['max_steps'], s['observation_after'])
                                      for s in segs_req],
                         'segs_exec': (pc or {}).get('segs'), 'reason': (pc or {}).get('reason')})
        # handoffs
        hand = []
        for k in range(len(rows) - 1):
            prev, nxt = rows[k], rows[k + 1]
            L = nxt['lat']
            prev_reason = prev['reason'] if prev['type'] == 'execute' else 'direct_' + prev['type']
            prev_grip = [s[2] for s in prev['segs_req'] if s[0] == 'set_gripper']
            exec_segs = nxt['segs_exec'] or []
            # leading free-space transit segments of the next decision (as executed)
            P_mod = leading(exec_segs, lambda s: s[1] == 'transit' and s[0] in ('move_to_pose', 'move_relative'))
            P_steps_mod = sum(s[2] or 0 for s in P_mod)
            # upper bound: the whole next fragment's executed steps
            P_steps_all = sum(s[2] or 0 for s in exec_segs) if exec_segs else nxt['n_steps']
            eligible = prev_reason in ('fragment_complete', 'observation_required')
            ov = {}
            for label, c in (('legacy', c_legacy), ('compact', c_compact)):
                ov[label] = {
                    'upper': min(L, P_steps_all * c),
                    'moderate': min(L, P_steps_mod * c) if eligible else 0.0,
                    'conservative': min(L, P_steps_mod * c) if (eligible and prev_grip) else 0.0,
                }
            miss = bool(eligible and 'close' in prev_grip and not P_mod)
            hand.append({'k_next': nxt['k'], 'L': L, 'prev_reason': prev_reason, 'prev_grip': prev_grip,
                         'next_first': exec_segs[0][:2] if exec_segs else nxt['type'],
                         'P_steps_mod': P_steps_mod, 'P_steps_all': P_steps_all, 'eligible': eligible,
                         'potential_miss_after_close': miss, 'overlap': ov})
        M = sum(r['lat'] for r in rows)
        first = rows[0]['lat'] if rows else 0.0
        out = {'status': summ['status'], 'raw': summ.get('libero_success'), 'v2': summ.get('v2_success'),
               'steps': steps, 'calls': len(rows), 'model_s': M, 'first_call_s': first,
               'c_legacy': c_legacy, 'c_compact': c_compact,
               'control_s': summ['control_elapsed_s'], 'elapsed_s': summ['elapsed_s'],
               'prep_s': summ['user_check']['preparation_elapsed_s'],
               'approval_to_claim_s': summ['user_check'].get('approval_to_claim_s') or 0.0,
               'video_s': tc['video_render_s'], 'run_state_s': tc['run_state_s'],
               'handoffs': hand, 'rows': rows}
        ctrl_compact = out['control_s'] - tc['video_render_s'] - tc['run_state_s'] + steps * COMPACT_VIDEO_S_PER_STEP
        overhead = out['elapsed_s'] - out['control_s'] + out['prep_s'] + out['approval_to_claim_s']
        out['service_today'] = out['control_s'] + overhead
        out['service_compact'] = ctrl_compact + overhead
        for rule in ('upper', 'moderate', 'conservative'):
            sv = sum(h['overlap']['compact'][rule] for h in hand)
            out['saved_compact_' + rule] = sv
            out['service_compact_pipelined_' + rule] = ctrl_compact + overhead - sv
            sl = sum(h['overlap']['legacy'][rule] for h in hand)
            out['saved_legacy_' + rule] = sl
        out['handoff_reason_counts'] = dict(collections.Counter(h['prev_reason'] for h in hand))
        out['eligible_handoffs'] = sum(h['eligible'] for h in hand)
        out['eligible_with_transit_prefix'] = sum(h['eligible'] and h['P_steps_mod'] > 0 for h in hand)
        out['eligible_after_gripper'] = sum(h['eligible'] and bool(h['prev_grip']) for h in hand)
        out['eligible_after_gripper_with_transit_prefix'] = sum(
            h['eligible'] and bool(h['prev_grip']) and h['P_steps_mod'] > 0 for h in hand)
        out['potential_misses_after_close'] = sum(h['potential_miss_after_close'] for h in hand)
        res[name] = out
        for key in ('calls', 'steps', 'eligible_handoffs', 'eligible_with_transit_prefix', 'eligible_after_gripper',
                    'eligible_after_gripper_with_transit_prefix', 'potential_misses_after_close'):
            agg[key] += out[key]
        agg['handoffs'] += len(hand)
    json.dump(res, open(OUT, 'w'), indent=1)
    print('aggregate', dict(agg))
    hdr = ('run', 'raw', 'calls', 'steps', 'model', 'svc_today', 'svc_cmp', 'cmp_up', 'cmp_mod', 'cmp_cons', 'save_mod', 'save_cons', 'c_leg', 'c_cmp')
    print(' '.join(f'{h:>9s}' for h in hdr))
    for name, o in res.items():
        print(' '.join([f'{name:>9s}', f"{str(o['raw'])[:1]:>9s}", f"{o['calls']:9d}", f"{o['steps']:9d}",
                        f"{o['model_s']:9.1f}", f"{o['service_today']:9.1f}", f"{o['service_compact']:9.1f}",
                        f"{o['service_compact_pipelined_upper']:9.1f}", f"{o['service_compact_pipelined_moderate']:9.1f}",
                        f"{o['service_compact_pipelined_conservative']:9.1f}", f"{o['saved_compact_moderate']:9.1f}",
                        f"{o['saved_compact_conservative']:9.1f}", f"{o['c_legacy']:9.3f}", f"{o['c_compact']:9.3f}"]))
    succ = [o for o in res.values() if o['raw']]
    n = len(succ)
    for key in ('service_today', 'service_compact', 'service_compact_pipelined_upper',
                'service_compact_pipelined_moderate', 'service_compact_pipelined_conservative',
                'saved_compact_upper', 'saved_compact_moderate', 'saved_compact_conservative',
                'saved_legacy_upper', 'saved_legacy_moderate', 'model_s'):
        vals = [o[key] for o in succ]
        print(f'{key:45s} mean over {n} raw successes = {sum(vals)/n:7.1f}  <=300: '
              f"{sum(v <= 300 for v in vals) if key.startswith('service') else '-'}")


if __name__ == '__main__':
    main()
