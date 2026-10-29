"""Plan B: classify every uc7 handoff and estimate overlap counterfactuals. Read-only.

Input: B/decisions.json (extract.py). Output: B/handoffs.json + printed tables.

Handoff k -> k+1 = fragment k ended, call k+1 was made (blocking), decision k+1 ran.
Classes (from measured fragment outcome, no simulator truth):
  post_gripper : fragment k completed (fragment_complete) and executed a set_gripper
                 segment followed by a final arm motion (test lift / withdraw).
  pre_contact  : fragment k completed with no gripper segment (arrived somewhere).
  stall        : fragment k ended in segment_stall.
  veto         : target_envelope_rejected (zero-action or mid-sequence).
  direct       : decision k was a 'motion'/'finish' direct action.
  other        : observation_required / budget / anything else.
Continuation candidate P_{k+1}: leading segments of fragment k+1 whose profile is
transit and skill is move_to_pose/move_relative and that ended pose_arrived.
Counterfactual (estimate): if decision k had committed P_{k+1} after its checkpoint,
P_{k+1} runs while call k+1 is in flight: saving = min(L_{k+1}, E(P_{k+1})).
E(.) = executed actions x per-step non-model cost (legacy = measured; compact = trial_003 video).
"""
import json, os, statistics, collections

B = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(B, 'decisions.json')))
COMPACT_VIDEO = 25.642 / 1019   # trial_003 video_render_s / steps (C1 paired_results / summary)
RAW_OK = {'t0_001', 't1_001', 't2_002', 't3_003', 't3_005', 't4_001', 't5_001', 't5_002', 't7_001',
          't8_001', 't9_001'}


def per_step_costs(s):
    tc, n = s['timing'], s['steps']
    legacy = (tc['simulator_step_s'] + tc['video_render_s'] + tc['run_state_s'] +
              tc['observe_transition_s'] + tc['policy_compute_s']) / n
    compact = (tc['simulator_step_s'] + tc['observe_transition_s'] + tc['policy_compute_s']) / n + COMPACT_VIDEO
    return legacy, compact


def classify(r):
    if r['type'] != 'execute':
        return 'direct'
    if r['frag_reason'] is None and set(r['kinds']) == {'agent_owned:local_motion'} and r['n_steps'] <= 1:
        return 'veto'   # stage-entry envelope rejection: zero fragment actions, one hold step
    reason = r['frag_reason']
    segs = r['seg_out']
    if reason == 'segment_stall':
        return 'stall'
    if reason == 'target_envelope_rejected':
        return 'veto'
    if reason == 'fragment_complete':
        grip_idx = [i for i, s in enumerate(segs) if s['skill'] == 'set_gripper' and s['status'] == 'succeeded']
        if grip_idx and segs[-1]['skill'] != 'set_gripper' and grip_idx[-1] < len(segs) - 1:
            return 'post_gripper'
        if grip_idx:
            return 'gripper_final'
        return 'pre_contact'
    return 'other:' + str(reason)


def grip_kind(r):
    kinds = [s['gripper'] for s in r['segs'] if s['skill'] == 'set_gripper']
    return kinds[-1] if kinds else None


def transit_prefix(r):
    acts = 0
    nseg = 0
    for s in r['seg_out']:
        if s['profile'] == 'transit' and s['skill'] in ('move_to_pose', 'move_relative') and s['reason'] == 'pose_arrived':
            acts += s['actions']
            nseg += 1
        else:
            break
    return nseg, acts


def main():
    out = {}
    agg = collections.defaultdict(lambda: collections.Counter())
    lat_by_class = collections.defaultdict(list)
    for name, run in D.items():
        rows = run['rows']
        legacy, compact = per_step_costs(run['summary'])
        hand = []
        for k in range(len(rows) - 1):
            r, nx = rows[k], rows[k + 1]
            cls = classify(r)
            nseg, acts = transit_prefix(nx) if nx['type'] == 'execute' else (0, 0)
            first_next = nx['segs'][0]['skill'] + '/' + nx['segs'][0]['profile'] if nx['type'] == 'execute' else nx['type']
            hit = nseg > 0
            e_leg, e_cmp = acts * legacy, acts * compact
            row = {'k': k, 'class': cls, 'grip': grip_kind(r), 'L_next': nx['lat'], 'next_first': first_next,
                   'P_segments': nseg, 'P_actions': acts, 'E_P_legacy': e_leg, 'E_P_compact': e_cmp,
                   'save_legacy': min(nx['lat'], e_leg) if hit else 0.0,
                   'save_compact': min(nx['lat'], e_cmp) if hit else 0.0,
                   'frag_k_steps': r['n_steps'], 'E_k_compact': r['n_steps'] * compact,
                   'E_k_legacy': r['n_steps'] * legacy}
            hand.append(row)
            agg[name][cls] += 1
            lat_by_class[cls].append(nx['lat'])
        out[name] = {'legacy_s_per_step': legacy, 'compact_s_per_step': compact, 'first_call_s': rows[0]['lat'],
                     'handoffs': hand, 'last_decision_reason': rows[-1]['frag_reason'],
                     'raw': name in RAW_OK}
    json.dump(out, open(os.path.join(B, 'handoffs.json'), 'w'), indent=1)

    # tables
    print('per-run handoff classes')
    classes = sorted({c for v in agg.values() for c in v})
    print('run', *classes)
    tot = collections.Counter()
    for name in D:
        print(name, *[agg[name][c] for c in classes])
        tot.update(agg[name])
    print('TOTAL', *[tot[c] for c in classes], 'sum', sum(tot.values()))
    print()
    print('next-call latency by class: n mean median')
    for c in classes:
        v = lat_by_class[c]
        print(c, len(v), round(statistics.mean(v), 1), round(statistics.median(v), 1))
    print()
    # post_gripper detail
    for scope, names in (('raw11', RAW_OK), ('all13', set(D))):
        pg = [h for n in names for h in out[n]['handoffs'] if h['class'] == 'post_gripper']
        hits = [h for h in pg if h['P_segments'] > 0]
        print(scope, 'post_gripper handoffs', len(pg), 'with transit-prefix next (hit)', len(hits),
              'by grip', collections.Counter((h['grip'], h['P_segments'] > 0) for h in pg))
        if hits:
            print('   P actions median', statistics.median(h['P_actions'] for h in hits),
                  'E_P compact median', round(statistics.median(h['E_P_compact'] for h in hits), 1),
                  'E_P legacy median', round(statistics.median(h['E_P_legacy'] for h in hits), 1),
                  'L_next median', round(statistics.median(h['L_next'] for h in hits), 1))
            print('   saving compact sum', round(sum(h['save_compact'] for h in hits), 1),
                  'legacy sum', round(sum(h['save_legacy'] for h in hits), 1))
        miss = [h for h in pg if h['P_segments'] == 0]
        print('   misses next_first:', collections.Counter(h['next_first'] for h in miss))
    print()
    print('per-run (raw successes): calls, first, post_gripper hits, saving_compact, saving_legacy')
    for n in sorted(RAW_OK):
        hs = out[n]['handoffs']
        pg = [h for h in hs if h['class'] == 'post_gripper' and h['P_segments'] > 0]
        print(n, len(hs) + 1, round(out[n]['first_call_s'], 1), len(pg), round(sum(h['save_compact'] for h in pg), 1),
              round(sum(h['save_legacy'] for h in pg), 1))
    # Oracle upper bound: every handoff's next transit prefix overlapped (any class)
    print()
    for scope, names in (('raw11', RAW_OK),):
        allh = [h for n in names for h in out[n]['handoffs']]
        print(scope, 'ORACLE any-class transit-prefix overlap: n', sum(h['P_segments'] > 0 for h in allh),
              'saving compact', round(sum(h['save_compact'] for h in allh), 1),
              'legacy', round(sum(h['save_legacy'] for h in allh), 1))
        # full-fragment overlap upper bound (every call fully overlapped with preceding fragment)
        ub_c = sum(min(h['L_next'], h['E_k_compact']) for h in allh)
        ub_l = sum(min(h['L_next'], h['E_k_legacy']) for h in allh)
        print(scope, 'UPPER BOUND sum min(L_k+1, E_k): compact', round(ub_c, 1), 'legacy', round(ub_l, 1))


if __name__ == '__main__':
    main()
