#!/usr/bin/env python3
"""Read-only: build per-run table + trigger classification from calls.json and run artifacts.
Usage: python3 -I run_table.py <calls.json> <runs_root> <out.json>
"""
import collections
import json
import os
import statistics
import sys

RUNS = [
    # key, run dir (relative to runs root), revision label, review dir (prep)
    ('uc7 t1_001', 'agent_speedup_v2_user_check_libero_10_t1_001', 'uc7', 'agent_speedup_v2_user_check_libero_10_t1_001_review'),
    ('t1_001', 'agent_speedup_v3_t1_001', 'r1', 'agent_speedup_v3_t1_review_001'),
    ('t1_002', 'agent_speedup_v3_t1_002', 'r2', 'agent_speedup_v3_t1_review_002'),
    ('t1_003', 'agent_speedup_v3_t1_003', 'r3', 'agent_speedup_v3_t1_review_003'),
    ('t1_004', 'agent_speedup_v3_t1_004', 'r4', 'agent_speedup_v3_t1_review_004'),
    ('t1_005', 'agent_speedup_v3_t1_005', 'r4.1', 'agent_speedup_v3_t1_review_005'),
    ('t1_006', 'agent_speedup_v3_t1_006', 'r5', 'agent_speedup_v3_t1_review_006_compact'),
    ('t1_007', 'agent_speedup_v3_t1_007', 'r5', 'agent_speedup_v3_t1_review_007'),
    ('t1_008', 'agent_speedup_v3_t1_008', 'r6', 'agent_speedup_v3_t1_review_008'),
    ('t1_009', 'agent_speedup_v3_t1_009', 'r7', 'agent_speedup_v3_t1_009_review'),
    ('t1_010', 'agent_speedup_v3_t1_010', 'r8', 'agent_speedup_v3_t1_010_review'),
    ('t7_001', 'agent_speedup_v3_t7_001', 'r8', 'agent_speedup_v3_t7_001_review'),
    ('g5_001', 'agent_speedup_v3_g5_001', 'r8', 'agent_speedup_v3_g5_001_review'),
    ('g5_002', 'agent_speedup_v3_g5_002', 'r9', 'agent_speedup_v3_g5_002_review'),
    ('t1_011', 'agent_speedup_v3_t1_011', 'r10', 'agent_speedup_v3_t1_011_review'),
    ('t1_012', 'agent_speedup_v3_t1_012', 'r11', 'agent_speedup_v3_t1_012_review'),
    ('base att1', 'v3_completion_baseline_20261026T133633Z/attempts/attempt_001/execution', 'r6copy', None),
    ('base att3', 'v3_completion_baseline_20261026T133633Z/attempts/attempt_003/execution', 'r6copy', None),
]

NATURAL = {'fragment_complete', 'adaptive_local_segment_complete', 'observation_required'}
PHYSICAL = {'segment_stall', 'segment_budget_exhausted'}


def classify(tr, idx):
    if idx == 0:
        return 'initial'
    a, st, r = tr.get('lf_action'), tr.get('lf_status'), tr.get('lf_reason')
    if a == 'acquire_evidence':
        return 'evidence_followup'
    if a == 'rejected_repeat':
        return 'repeat_guard'
    if r in PHYSICAL:
        return 'physical_stall_or_budget'
    if r in ('fragment_complete', 'adaptive_local_segment_complete'):
        return 'natural_completion'
    if r == 'observation_required' and st == 'succeeded':
        return 'natural_completion'  # model-requested observation_after segment completed
    if r and ('geometry' in r):
        return 'geometry_gate'
    return 'other:%s/%s/%s' % (a, st, r)


def pct(vals, q):
    v = sorted(vals)
    if not v:
        return None
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def main():
    calls = json.load(open(sys.argv[1]))
    root = sys.argv[2]
    out = {}
    for key, rd, rev, review in RUNS:
        base = os.path.basename(rd.rstrip('/'))
        cv = calls.get(base)
        if base == 'execution':
            cv = calls.get('execution') if 'attempt_003' in rd else None
        s = json.load(open(os.path.join(root, rd, 'summary.json')))
        tc = s.get('timing_components') or {}
        perf = None
        pp = os.path.join(root, rd, 'performance.json')
        if os.path.exists(pp):
            perf = (json.load(open(pp)).get('completion_target') or {})
        prep = None
        if review and os.path.exists(os.path.join(root, review, 'review.json')):
            prep = (json.load(open(os.path.join(root, review, 'review.json'))).get('preparation') or {}).get('elapsed_s')
        row = {
            'key': key, 'dir': rd, 'revision': rev, 'task': s.get('task_key'), 'episode': s.get('episode_index'),
            'prompt_version': s.get('prompt_version'), 'steps': s.get('steps'), 'api_calls': s.get('api_calls'),
            'status': s.get('status'), 'libero_success': s.get('libero_success'),
            'elapsed_s': s.get('elapsed_s'), 'control_s': s.get('control_elapsed_s'),
            'control_wo_video_s': s.get('control_without_video_s'),
            'model_s': tc.get('model_pipeline_s'), 'transition_s': tc.get('observe_transition_s'),
            'sim_s': tc.get('simulator_step_s'), 'video_s': tc.get('video_render_s'), 'run_state_s': tc.get('run_state_s'),
            'policy_s': tc.get('policy_compute_s'),
            'prep_s': (perf or {}).get('preparation_elapsed_s') or prep,
            'total_service_s': (perf or {}).get('total_service_elapsed_s'),
            'tokens_in': (s.get('budget') or {}).get('input_tokens'), 'tokens_out': (s.get('budget') or {}).get('output_tokens'),
            'tokens_reason': (s.get('budget') or {}).get('reasoning_tokens'), 'tokens_cached': (s.get('budget') or {}).get('cached_input_tokens'),
        }
        if row['total_service_s'] is None and row['prep_s'] is not None and row['elapsed_s'] is not None:
            row['total_service_s_approx'] = row['prep_s'] + row['elapsed_s']
        if cv:
            cl = cv['calls']
            valid = [c for c in cl if c.get('result_status') == 'valid']
            lat = [c.get('transport_elapsed_s') if c.get('transport_elapsed_s') is not None else c.get('budget_interval_s') for c in valid]
            lat = [x for x in lat if x is not None]
            row.update({
                'n_request_files': len(cl), 'n_valid': len(valid),
                'lat_sum': sum(lat), 'lat_mean': statistics.mean(lat) if lat else None,
                'lat_p50': pct(lat, .5), 'lat_p90': pct(lat, .9), 'lat_max': max(lat) if lat else None,
                'lat_first': lat[0] if lat else None,
                'req_bytes_mean': statistics.mean(c['request_bytes'] for c in cl), 'req_bytes_max': max(c['request_bytes'] for c in cl),
                'text_chars_mean': statistics.mean(c['text_chars_total'] for c in cl), 'text_chars_max': max(c['text_chars_total'] for c in cl),
                'images_max': max(c['images_total'] for c in cl),
                'in_tok_mean': statistics.mean(c['input_tokens'] for c in valid) if valid else None,
                'in_tok_max': max(c['input_tokens'] for c in valid) if valid else None,
                'out_tok_mean': statistics.mean(c['output_tokens'] for c in valid) if valid else None,
                'reason_tok_mean': statistics.mean((c['reasoning_tokens'] or 0) for c in valid) if valid else None,
                'out_chars_mean': statistics.mean(c['output_text_chars'] for c in valid) if valid else None,
                'out_chars_first': valid[0]['output_text_chars'] if valid else None,
                'out_tok_first': valid[0]['output_tokens'] if valid else None,
                'decision_types': dict(collections.Counter(c.get('decision', {}).get('type') for c in valid)),
            })
            trig = collections.Counter()
            trig_time = collections.Counter()
            dec_time = collections.Counter()
            for i, c in enumerate(cl):
                if c.get('result_status') != 'valid':
                    continue
                cat = classify(c.get('trigger', {}), i)
                trig[cat] += 1
                l = c.get('transport_elapsed_s') if c.get('transport_elapsed_s') is not None else c.get('budget_interval_s')
                trig_time[cat] += l or 0
                dec_time[c.get('decision', {}).get('type')] += l or 0
            row['triggers'] = dict(trig)
            row['trigger_time_s'] = {k: round(v, 1) for k, v in trig_time.items()}
            row['decision_time_s'] = {k: round(v, 1) for k, v in dec_time.items()}
            steps = [c.get('step') for c in cl if c.get('step') is not None]
            gaps = [b - a for a, b in zip(steps, steps[1:])]
            row['step_gap_median'] = statistics.median(gaps) if gaps else None
            row['step_gap_le4'] = sum(1 for g in gaps if g <= 4)
            row['n_gaps'] = len(gaps)
        out[key] = row
    json.dump(out, open(sys.argv[3], 'w'), indent=1)
    # print compact table
    for k, r in out.items():
        print(k, json.dumps({kk: (round(vv, 1) if isinstance(vv, float) else vv) for kk, vv in r.items() if kk not in ('dir',)}))


if __name__ == '__main__':
    main()
