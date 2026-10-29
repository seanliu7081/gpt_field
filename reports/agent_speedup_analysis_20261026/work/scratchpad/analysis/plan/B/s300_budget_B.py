"""Plan B S300 arithmetic for the 11 uc7 raw-LIBERO successes. Read-only; estimates only.

Scenarios are cumulative and are arithmetic on measured components (summary.json
timing_components, per-call latencies from episode.events.jsonl skill_decision rows),
NOT predictions of model behaviour:
  S0 measured   : preparation_elapsed_s + approval_to_claim_s + elapsed_s
  S1 compactIO  : video -> 0.0252 s/step (trial_003), run_state -> 0 (C1 arm B, verified 897/897 equal actions)
  S2 noVeto     : drop the replacement call after each target_envelope_rejected (+1 hold step)
  S3 actionFirst: subtract per-call prose tail (stream_order.json: prose share x visible tokens x 0.0212 s)
  S4 continue   : post-gripper checkpoints become non-blocking; the next decision's measured transit
                  prefix P runs during the call: saving min(L_eff, E_compact(P)) on hits (handoffs.json)
  S5 asyncVideo : the remaining compact video work runs in a recorder thread during model waits
                  (hidden up to the idle wait left after S4)
A latency multiplier variant (x1.42 = 21.5/15.1 s per call, today's trial_002/003 vs history) is
also reported; per-step costs are left at the historical values there (optimistic).
"""
import json, os, statistics

B = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(B, 'decisions.json')))
H = json.load(open(os.path.join(B, 'handoffs.json')))
SO = json.load(open(os.path.join(B, 'stream_order.json')))
RAW = ['t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002', 't7_001', 't8_001', 't9_001']
VIDEO_COMPACT = 25.642 / 1019


def scenario(name, lat_mult=1.0):
    run = D[name]
    s = run['summary']
    tc = s['timing']
    steps = s['steps']
    rows = run['rows']
    lat = [r['lat'] * lat_mult for r in rows]
    model = sum(lat)
    approval = s['approval_to_claim_s'] or 0.0
    s0 = s['prep_s'] + approval + s['elapsed_s'] + (model - tc['model_pipeline_s'])
    s1 = s0 - tc['video_render_s'] - tc['run_state_s'] + VIDEO_COMPACT * steps
    compact_step = (tc['simulator_step_s'] + tc['observe_transition_s'] + tc['policy_compute_s']) / steps + VIDEO_COMPACT
    # S2: envelope veto replacement calls
    veto_save = 0.0
    vetoed = []
    for k, r in enumerate(rows[:-1]):
        zero_action_veto = (r['type'] == 'execute' and r['frag_reason'] is None and r['n_steps'] <= 1
                            and set(r['kinds']) == {'agent_owned:local_motion'})
        if r['frag_reason'] == 'target_envelope_rejected' or zero_action_veto:
            veto_save += lat[k + 1] + compact_step
            vetoed.append(k)
    s2 = s1 - veto_save
    # S3: action-first early dispatch
    so = [x for x in SO if x['run'] == name]
    assert len(so) == len(rows), (name, len(so), len(rows))
    a3 = [x['save_s'] * lat_mult for x in so]
    a3_total = sum(a3[i] for i in range(len(rows)) if i - 1 not in vetoed)
    s3 = s2 - a3_total
    # S4: continuation overlap on post-gripper hits
    o_total = 0.0
    n_hits = 0
    hidden_steps = 0.0
    for h in H[name]['handoffs']:
        k1 = h['k'] + 1
        if h['class'] == 'post_gripper' and h['P_segments'] > 0:
            l_eff = lat[k1] - a3[k1]
            o_total += min(l_eff, h['P_actions'] * compact_step)
            hidden_steps += min(h['P_actions'], l_eff / compact_step)
            n_hits += 1
    # miss cost: post-gripper checkpoints whose next decision did not start with transit; assume the model
    # had committed a run-out of the median hit length and must come back (one extra run-out of travel)
    hit_steps = sorted(h['P_actions'] for n2 in RAW for h in H[n2]['handoffs']
                       if h['class'] == 'post_gripper' and h['P_segments'] > 0)
    med_steps = hit_steps[len(hit_steps) // 2]
    n_miss = sum(1 for h in H[name]['handoffs'] if h['class'] == 'post_gripper' and h['P_segments'] == 0)
    miss_cost = n_miss * med_steps * compact_step
    s4 = s3 - o_total + miss_cost
    # S5: async compact video hidden within remaining idle waits
    idle = sum(lat[i] - a3[i] for i in range(len(rows)) if i - 1 not in vetoed) - o_total
    video = VIDEO_COMPACT * (steps - hidden_steps)   # only video work of steps still on the critical path
    s5 = s4 - min(video, max(0.0, idle))
    return {'run': name, 'calls': len(rows), 'steps': steps, 'S0': s0, 'S1': s1, 'S2': s2, 'S3': s3, 'S4': s4,
            'S5': s5, 'veto_save': veto_save, 'a3_total': a3_total, 'overlap': o_total, 'hits': n_hits,
            'misses': n_miss, 'miss_cost': miss_cost,
            'model': model, 'idle_after': idle}


def report(lat_mult):
    res = [scenario(n, lat_mult) for n in RAW]
    print(f'--- latency multiplier {lat_mult} ---')
    print('run calls steps S0 S1 S2 S3 S4 S5 | veto a3 overlap(hits)')
    for r in res:
        print(r['run'], r['calls'], r['steps'], *[round(r[k], 1) for k in ('S0', 'S1', 'S2', 'S3', 'S4', 'S5')],
              '|', round(r['veto_save'], 1), round(r['a3_total'], 1), f"{r['overlap']:.1f}({r['hits']})",
              f"miss {r['miss_cost']:.1f}({r['misses']})")
    for k in ('S0', 'S1', 'S2', 'S3', 'S4', 'S5'):
        v = [r[k] for r in res]
        print(k, 'mean', round(statistics.mean(v), 1), 'median', round(statistics.median(v), 1),
              '<=300:', sum(x <= 300 for x in v), [r['run'] for r in res if r[k] <= 300])
    return res


if __name__ == '__main__':
    out = {'x1.0': report(1.0), 'x1.42': report(1.42)}
    # blocking calls that would still have to go at x1.42 to bring the S5 mean to 300 s
    r42 = out['x1.42']
    gap = statistics.mean(r['S5'] for r in r42) - 300
    per_call = statistics.mean((r['model'] - r['a3_total']) / r['calls'] for r in r42)
    print('x1.42: mean gap to 300 s', round(gap, 1), 's; mean effective blocking latency per call', round(per_call, 1),
          's -> about', round(gap / per_call, 1), 'fewer blocking calls per run needed')
    json.dump(out, open(os.path.join(B, 's300_budget_B.json'), 'w'), indent=1)
