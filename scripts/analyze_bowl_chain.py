"""Describe a terminal bowl-episode chain from saved evidence only.

No credential access, API client, simulator, alternative policy, or rollout is
created. Historical segments are counted once; reconstruction is not new action.
Use the existing Python environment that contains numpy and matplotlib, e.g.:
PYTHONPATH=src /workspace/venvs/libero-plus/bin/python scripts/analyze_bowl_chain.py \
  --source runs/astra_bowl_continuation_003
The source must already have a terminal run_state and closed trace. Existing
reports are never overwritten. This analysis complements the separate audit;
it does not independently certify API pricing or physical replay correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from gpt_field.schema import GridSpec

TASK = 'pick up the black bowl between the plate and the ramekin and place it on the plate'
TERMINAL = {'success', 'budget_exceeded', 'infrastructure_error', 'user_stop'}
FIT_TOLERANCE = 1e-10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def terminal_row(run):
    state = read(run / 'run_state.json')
    require(state.get('status') in TERMINAL, f'Source is not terminal: {run}')
    rows = [json.loads(line) for line in (run / 'episodes.jsonl').read_text().splitlines() if line.strip()]
    require(len(rows) == 1, f'Expected one episode/segment result: {run}')
    row = rows[0]
    require(row.get('status') == state['status'], f'Terminal result/state disagree: {run}')
    require((row.get('task_prompt'), row.get('suite'), row.get('task_id'), row.get('episode_index'),
             row.get('seed'), row.get('method')) == (TASK, 'libero_spatial', 0, 0, 7, 'field'),
            f'Unexpected task or episode identity: {run}')
    if 'continuation_offset' in row:
        require(state.get('stage') == 'terminal', f'Continuation is not finalized: {run}')
        require(row == read(run / 'summary.json'), f'Summary/result differ: {run}')
        for key in ('continuation_offset', 'new_evaluated_actions', 'cumulative_actions'):
            require(state.get(key) == row.get(key), f'Terminal action counts disagree: {run}')
        completed = row['status'] == 'success'
        require(row.get('evaluation_completed') is completed and row.get('libero_success') is (True if completed else None),
                f'Terminal outcome is inconsistent: {run}')
    return row, state


def discover_chain(source):
    """Reject live sources before inspecting their trace or creating outputs."""
    source = Path(source).resolve()
    row, state = terminal_row(source)
    manifest = read(source / 'manifest.json')
    identity = manifest['configuration']
    require(manifest['identity_hash'] == canonical_hash(identity), 'Source manifest identity changed')
    chain = identity.get('chain')
    require(isinstance(chain, list) and len(chain) >= 2, 'Expected the ordered root and continuation history')
    require(identity.get('source_trace', {}).get('segments') == chain, 'Ordered history bundle differs')
    require(identity['source_trace'].get('available_completed_actions') == row['continuation_offset'],
            'History action count differs from source offset')
    segments = [dict(item) for item in chain]
    segments.append({'run': str(source), 'identity_hash': manifest['identity_hash'],
        'action_start': row['continuation_offset'], 'action_count': row['new_evaluated_actions'],
        'trace': {'trace': row['log']}})
    require(len({str(Path(s['run']).resolve()) for s in segments}) == len(segments), 'Duplicate segment would double-count actions')
    require(Path(segments[0]['run']).resolve() == Path(identity['root_run']).resolve()
            and segments[0]['identity_hash'] == identity['root_identity_hash'], 'Root identity differs')
    cumulative = 0
    previous = None
    for index, segment in enumerate(segments):
        run = Path(segment['run']).resolve()
        result, run_state = terminal_row(run)
        m = read(run / 'manifest.json')
        config = m['configuration']
        require(m['identity_hash'] == segment['identity_hash'] == canonical_hash(config), f'Identity changed: {run}')
        require(type(segment['action_count']) is int and segment['action_count'] >= 0
                and segment['action_start'] == cumulative, f'Discontinuous/invalid action span: {run}')
        expected_count = result.get('new_evaluated_actions', result.get('steps'))
        require(expected_count == segment['action_count'], f'Result/chain action count differs: {run}')
        require(segment['trace']['trace'] == result['log'], f'Trace filename differs: {run}')
        trace_name = Path(result['log'])
        require(trace_name.name == result['log'], f'Trace must be inside its run: {run}')
        if previous is None:
            require(config.get('grid') == GridSpec().to_dict(), 'Unexpected root grid')
        else:
            require(config.get('continuation_offset') == cumulative and result['cumulative_actions'] == cumulative + expected_count,
                    f'Continuation/reconstruction counts differ: {run}')
            require(config.get('parent_identity_hash') == previous['identity_hash']
                    and Path(config.get('parent_run', '')).resolve() == Path(previous['run']).resolve(),
                    f'Parent link differs: {run}')
            require(result.get('reconstruction_actions') == cumulative, f'Reconstruction/action offset differs: {run}')
        segment['run'] = str(run)
        segment['terminal_status'] = result['status']
        segment['manifest_sha256'] = digest(run / 'manifest.json')
        cumulative += expected_count
        previous = segment
    require(cumulative == row['cumulative_actions'], 'Cumulative actions omit or duplicate history')
    return segments, row, state


def load_records(segments):
    records, api_attempts, sources, unexecuted = [], [], [], []
    previous_observation = None
    previous_status = None
    for segment in segments:
        run = Path(segment['run'])
        path = run / segment['trace']['trace']
        before = digest(path)
        expected_hash = segment['trace'].get('snapshot_sha256')
        require(expected_hash is None or expected_hash == before, f'Historical trace hash changed: {path}')
        pending, seen_start, count = None, False, 0
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                require(line.endswith('\n'), f'Partial terminal trace row: {path}')
                entry = json.loads(line)
                event = entry.get('event')
                if not seen_start:
                    expected_event = 'reset' if segment['action_start'] == 0 else 'continuation_start'
                    require(event == expected_event, f'Trace start differs: {path}')
                    require(entry['sim_status']['steps'] == segment['action_start'], 'Trace reset/offset differs')
                    if previous_observation is not None:
                        require(entry['observation'] == previous_observation and entry['sim_status'] == previous_status,
                                f'Physical observation handoff differs: {path}')
                    previous_observation, previous_status = entry['observation'], entry['sim_status']
                    seen_start = True
                elif 'diagnostics' in entry:
                    require(pending is None and entry['step'] == len(records), 'Duplicate/skipped/out-of-order decision')
                    require(not previous_status.get('terminated') and not previous_status.get('success'), 'Decision follows terminal physical state')
                    require(entry['state'] == previous_observation['observation/state'], 'Decision measured state is discontinuous')
                    details, plan = entry['diagnostics'], entry['diagnostics']['plan']
                    require(details['step'] == entry['step'] and details['method'] == 'field'
                            and details['refreshed'] is (entry['step'] % 5 == 0), 'Field protocol/cadence differs')
                    pending = {key: entry[key] for key in ('step', 'state', 'action')}
                    pending.update(stage=details['stage'], delta_xyz=plan['delta_xyz'], planner_status=plan['status'],
                        refreshed=details['refreshed'], field_center=details['field_center'],
                        candidate_count=plan['candidate_count'], valid_count=plan['valid_count'],
                        invalid_reasons=dict(Counter(c.get('invalid_reason') for c in plan['candidates'] if not c['valid'])))
                    if details['refreshed']:
                        require(details['prediction'] == details['field_prediction'], 'Field transformation differs')
                        request_path = Path(details['api_request_log'])
                        require(request_path.resolve().parent == (run / 'api').resolve(), 'Field request is outside its segment')
                        request = read(request_path)
                        require(request.get('request_key') == details['api_request_key'], 'Field request linkage differs')
                        pending['prediction'] = details['prediction']
                        pending['request_ordinal'] = request['request_ordinal']
                elif event == 'executed':
                    require(pending is not None and pending['step'] == entry['step'] == len(records), 'Execution lacks matching decision')
                    require(entry['next_state'] == entry['observation']['observation/state'], 'Executed state/observation differ')
                    require(entry['sim_status']['steps'] == len(records) + 1, 'Simulator/global action count differs')
                    pending.update(next_state=entry['next_state'], simulator_success=entry['success'])
                    records.append(pending)
                    previous_observation, previous_status = entry['observation'], entry['sim_status']
                    pending = None
                    count += 1
                elif event in ('reset', 'continuation_start'):
                    raise ValueError('Unexpected additional reset within one episode')
        require(seen_start and count == segment['action_count'], f'Trace/manifest action count differs: {path}')
        if pending is not None:
            require(pending['step'] == len(records), 'Nonterminal unexecuted decision')
            unexecuted.append({'run': str(run), 'step': pending['step'], 'refreshed': pending['refreshed'],
                               'request_ordinal': pending.get('request_ordinal')})
        require(before == digest(path), f'Trace changed during analysis: {path}')
        sources.append({**segment, 'trace_sha256': before, 'trace_bytes': path.stat().st_size})
        for request_path in (run / 'api').glob('*.request.json'):
            request = read(request_path)
            result_path = request_path.with_name(request_path.name.removesuffix('.request.json') + '.result.json')
            require(result_path.is_file(), f'Terminal request has no settled result: {request_path}')
            result = read(result_path)
            api_attempts.append({'ordinal': request['request_ordinal'], 'status': result['status'], 'run': str(run)})
    api_attempts.sort(key=lambda item: item['ordinal'])
    require([a['ordinal'] for a in api_attempts] == list(range(1, len(api_attempts)+1)), 'Missing/duplicate historical request ordinal')
    require(records, 'No completed actions to analyze')
    require(all(a['step'] == i for i, a in enumerate(records)), 'Global action indices are not contiguous')
    accepted = {a['ordinal'] for a in api_attempts if a['status'] == 'valid'}
    require(all(r.get('request_ordinal') in accepted for r in records if r['refreshed']), 'Executed refresh lacks valid API result')
    return records, sources, api_attempts, unexecuted


def distribution(values):
    a = np.asarray(values, dtype=float)
    return {'count': len(a), 'min': float(a.min()), 'median': float(np.median(a)),
            'mean': float(a.mean()), 'max': float(a.max())} if len(a) else {'count': 0}


def axis(delta):
    delta = np.asarray(delta)
    if np.max(np.abs(delta)) < 1e-12:
        return 'hold'
    require(np.count_nonzero(np.abs(delta) > 1e-12) == 1, 'Unexpected non-axis-aligned command')
    i = int(np.argmax(np.abs(delta)))
    return ('+' if delta[i] > 0 else '-') + 'xyz'[i]


def fit_field(prediction, grid=GridSpec()):
    xyz = grid.coordinates((0,0,0)) / grid.spacing_m
    x,y,z = xyz.T
    designs = {'affine': np.column_stack((np.ones(len(x)),x,y,z)),
        'separable_quadratic': np.column_stack((np.ones(len(x)),x,y,z,x*x,y*y,z*z)),
        'full_quadratic': np.column_stack((np.ones(len(x)),x,y,z,x*x,y*y,z*z,x*y,x*z,y*z)),
        'arbitrary_additive_separable': np.column_stack([np.ones(len(x))] +
            [(a == k).astype(float) for a in (x,y,z) for k in range(-3,5)])}
    values = np.asarray(prediction['values'], dtype=float)
    require(values.shape == (grid.count,) and np.isfinite(values).all(), 'Malformed scalar field')
    valid = np.ones(grid.count, bool)
    valid[prediction['blocked_indices'] + prediction['unknown_indices']] = False
    require(valid.any(), 'No valid scalar samples for field fit')
    fitted_values = values[valid]
    std = float(fitted_values.std())
    fits = {}
    for name, design in designs.items():
        coef, _, rank, _ = np.linalg.lstsq(design[valid], fitted_values, rcond=None)
        residual = design[valid] @ coef - fitted_values
        rms = float(np.sqrt(np.mean(residual**2)))
        fits[name] = {'coefficients': coef.tolist(), 'design_rank': int(rank),
            'max_absolute_residual': float(np.abs(residual).max()), 'rms_residual': rms,
            'rms_residual_over_value_std': rms/std if std else None}
    return fits, valid, xyz


def analyze_fields(records):
    fields, changes = [], []
    grid = GridSpec()
    for record in records:
        if not record['refreshed']:
            continue
        p = record['prediction']
        fits, valid, xyz = fit_field(p, grid)
        values = np.asarray(p['values'])
        minima = np.flatnonzero(valid & (np.abs(values-values[valid].min()) <= FIT_TOLERANCE))
        centroid = grid.coordinates(record['field_center'])[minima].mean(axis=0)
        gradient = np.asarray(fits['separable_quadratic']['coefficients'][1:4]) / grid.spacing_m
        field = {'step': record['step'], 'request_ordinal': record['request_ordinal'],
            'stage': p['stage'], 'gripper_command': p['gripper_command'], 'motion_mode': p['motion_mode'],
            'orientation_target_xyzw': p['orientation_target_xyzw'], 'field_center_xyz_m': record['field_center'],
            'blocked_count': len(p['blocked_indices']), 'unknown_count': len(p['unknown_indices']),
            'mask_sha256': canonical_hash([p['blocked_indices'],p['unknown_indices']]),
            'valid_samples': int(valid.sum()), 'valid_value_min': float(values[valid].min()),
            'valid_value_max': float(values[valid].max()), 'valid_value_std': float(values[valid].std()),
            'fit': fits, 'sampled_minimizer_count': len(minima), 'sampled_minimum_centroid_xyz_m': centroid.tolist(),
            'all_sampled_minimizers_on_grid_boundary': bool(np.all(np.any(np.abs(xyz[minima]) == 4,axis=1))),
            'fitted_gradient_at_field_center_per_m': gradient.tolist(),
            'executed_axes_before_next_refresh': [axis(r['delta_xyz']) for r in records[record['step']:record['step']+5]]}
        if fields:
            old = fields[-1]
            old_gradient = np.asarray(old['fitted_gradient_at_field_center_per_m'])
            changes.append({'from_step': old['step'], 'to_step': field['step'],
                'center_motion_m': float(np.linalg.norm(np.asarray(field['field_center_xyz_m'])-old['field_center_xyz_m'])),
                'sampled_minima_centroid_jump_m': float(np.linalg.norm(centroid-old['sampled_minimum_centroid_xyz_m'])),
                'gradient_sign_reversals_axes': ['xyz'[i] for i in range(3)
                    if old_gradient[i]*gradient[i] < -1e-10 and min(abs(old_gradient[i]),abs(gradient[i])) > 1e-8]})
        fields.append(field)
    return fields, changes


def motion_summary(records, start, end):
    selected = records[start:end]
    if not selected:
        return {'action_start': start, 'action_count': 0}
    positions = np.asarray([selected[0]['state'][:3]] + [r['next_state'][:3] for r in selected])
    delta = np.diff(positions,axis=0)
    commands = np.asarray([r['delta_xyz'] for r in selected])
    path = float(np.linalg.norm(delta,axis=1).sum())
    target_path = float(np.linalg.norm(commands,axis=1).sum())
    by_axis = {}
    for label in sorted({axis(c) for c in commands}):
        mask = np.asarray([axis(c)==label for c in commands])
        by_axis[label] = {'count': int(mask.sum()), 'sum_realized_xyz_m': delta[mask].sum(axis=0).tolist()}
        if label != 'hold':
            projected = delta[mask,'xyz'.index(label[-1])] * (1 if label[0]=='+' else -1)
            by_axis[label]['realized_displacement_along_command_axis_m'] = distribution(projected)
    return {'action_start': start, 'action_count': end-start, 'initial_xyz_m': positions[0].tolist(),
        'final_xyz_m': positions[-1].tolist(), 'position_min_xyz_m': positions.min(axis=0).tolist(),
        'position_max_xyz_m': positions.max(axis=0).tolist(), 'measured_path_m': path,
        'measured_axis_absolute_travel_m': np.abs(delta).sum(axis=0).tolist(),
        'net_displacement_xyz_m': (positions[-1]-positions[0]).tolist(),
        'net_displacement_m': float(np.linalg.norm(positions[-1]-positions[0])),
        'commanded_target_delta_path_m': target_path,
        'measured_path_over_commanded_target_delta_path': path/target_path if target_path else None,
        'measured_displacement_per_step_m': distribution(np.linalg.norm(delta,axis=1)),
        'commanded_axes': dict(Counter(axis(c) for c in commands)), 'realized_motion_by_command_axis': by_axis}


def build_report(source):
    segments, result, state = discover_chain(source)
    records, sources, api_attempts, unexecuted = load_records(segments)
    require(len(api_attempts) == state['budget']['api_calls'], 'Cumulative API ledger/attempt count differs')
    require(records[-1]['simulator_success'] == (result['status']=='success'), 'Terminal outcome/last executed success differ')
    fields, changes = analyze_fields(records)
    reversals = [{'step': b['step'], 'from': axis(a['delta_xyz']), 'to': axis(b['delta_xyz']), 'at_field_refresh': b['refreshed']}
        for a,b in zip(records,records[1:]) if axis(a['delta_xyz']) != 'hold'
        and np.allclose(np.asarray(a['delta_xyz']),-np.asarray(b['delta_xyz']),atol=1e-12,rtol=0)]
    structure = {'accepted_executed_field_count': len(fields), 'fit_exact_threshold_absolute': FIT_TOLERANCE,
        'fits_within_threshold': {name: sum(f['fit'][name]['max_absolute_residual'] < FIT_TOLERANCE for f in fields)
                                 for name in fields[0]['fit']},
        'fit_residual_summary': {name: distribution([f['fit'][name]['max_absolute_residual'] for f in fields])
                                 for name in fields[0]['fit']},
        'quadratic_exception_steps': [f['step'] for f in fields if f['fit']['full_quadratic']['max_absolute_residual'] >= FIT_TOLERANCE],
        'distinct_separable_coefficients_rounded_10dp': len({tuple(np.round(f['fit']['separable_quadratic']['coefficients'],10)) for f in fields}),
        'fields_with_blocked_nodes': sum(f['blocked_count']>0 for f in fields),
        'fields_with_unknown_nodes': sum(f['unknown_count']>0 for f in fields),
        'blocked_counts': distribution([f['blocked_count'] for f in fields]),
        'unknown_counts': distribution([f['unknown_count'] for f in fields]),
        'unique_mask_patterns': len({f['mask_sha256'] for f in fields}),
        'valid_sample_counts': distribution([f['valid_samples'] for f in fields]),
        'fields_all_sampled_minimizers_on_grid_boundary': sum(f['all_sampled_minimizers_on_grid_boundary'] for f in fields),
        'refresh_center_motion_m': distribution([c['center_motion_m'] for c in changes]),
        'refresh_sampled_minima_centroid_jump_m': distribution([c['sampled_minima_centroid_jump_m'] for c in changes]),
        'refreshes_with_gradient_sign_reversal': sum(bool(c['gradient_sign_reversals_axes']) for c in changes),
        'gradient_sign_reversals_by_axis': dict(Counter(a for c in changes for a in c['gradient_sign_reversals_axes'])),
        'complete_five_action_fields': sum(len(f['executed_axes_before_next_refresh'])==5 for f in fields),
        'complete_fields_with_five_identical_command_axes': sum(len(f['executed_axes_before_next_refresh'])==5
            and len(set(f['executed_axes_before_next_refresh']))==1 for f in fields)}
    invalid = Counter()
    for record in records:
        invalid.update(record['invalid_reasons'])
    observations = {'model_gripper_commands': dict(Counter(f['gripper_command'] for f in fields)),
        'model_motion_modes': dict(Counter(f['motion_mode'] for f in fields)),
        'model_stage_labels_field_counts': dict(Counter(f['stage'] for f in fields)),
        'model_stage_labels_action_counts': dict(Counter(r['stage'] for r in records)),
        'executed_gripper_action_values': dict(Counter(str(r['action'][-1]) for r in records)),
        'planner_statuses': dict(Counter(r['planner_status'] for r in records)), 'candidate_invalid_reasons': dict(invalid),
        'candidate_total': sum(r['candidate_count'] for r in records), 'candidate_valid': sum(r['valid_count'] for r in records),
        'immediate_consecutive_command_reversals': len(reversals), 'immediate_reversals_at_refresh': sum(r['at_field_refresh'] for r in reversals),
        'stage_label_transitions': sum(a['stage'] != b['stage'] for a,b in zip(fields,fields[1:])),
        'whole_history_motion': motion_summary(records,0,len(records)),
        'segment_motion': [motion_summary(records,s['action_start'],s['action_start']+s['action_count']) for s in segments],
        'motion_by_window': [motion_summary(records,start,min(start+200,len(records))) for start in range(0,len(records),200)],
        'field_structure': structure}
    report = {'analysis_kind': 'terminal_chain_saved_evidence_field_and_motion_diagnostic',
        'generated_utc': datetime.now(timezone.utc).isoformat(), 'analysis_script_sha256': digest(__file__),
        'source': str(Path(source).resolve()), 'source_segments': sources,
        'api_calls_made_by_analysis': 0, 'simulator_steps_executed_by_analysis': 0, 'scored_evaluation': False,
        'terminal_status': result['status'], 'evaluation_completed': result['evaluation_completed'],
        'libero_success': result['libero_success'], 'included_completed_actions': len(records),
        'accepted_fields_with_executed_actions': len(fields), 'excluded_unexecuted_decisions': unexecuted,
        'historical_api_attempts': len(api_attempts), 'historical_api_result_status_counts': dict(Counter(a['status'] for a in api_attempts)),
        'historical_nonvalid_attempt_ordinals': [a['ordinal'] for a in api_attempts if a['status'] != 'valid'],
        'cumulative_ledger_as_recorded': state['budget'],
        'method': {'grid': GridSpec().to_dict(), 'polynomial_coordinates': '(world_point-recorded_field_center)/0.02; axes -4..4',
            'affine_terms': ['1','x','y','z'], 'separable_quadratic_terms': ['1','x','y','z','x^2','y^2','z^2'],
            'full_quadratic_terms': ['1','x','y','z','x^2','y^2','z^2','xy','xz','yz'],
            'arbitrary_additive_separable_terms': 'Constant + independent level indicators -3..4 on each axis;25parameters, no cross-axis terms.',
            'fit': 'Least squares on unmasked recorded scalar samples only; residuals and ranks retained.',
            'minimum': 'Centroid of all unmasked grid nodes within1e-10 of minimum; no claim of measured target position.',
            'trajectory': 'Sum of consecutive measured world-xyz displacement norms; reconstruction actions excluded.'},
        'observations': observations, 'per_refresh': fields, 'between_refresh': changes, 'command_reversals': reversals,
        'interpretation_limits': [
            'Only completed decision/executed pairs are counted; an unexecuted trailing decision is explicitly excluded.',
            'Stage labels are model claims, not contact, grasp, clearance or measured task-progress labels.',
            'Scalar field values are unitless model outputs, not calibrated distances or true cost-to-go.',
            'Scalar fits exclude masks; masked placeholders and spatial validity are separate outputs.',
            'Trilinear interpolation of quadratic samples is not an analytic quadratic between grid vertices.',
            'Temporal coefficient/minimum changes do not identify target grounding errors; intended stages can change.',
            'OSC target deltas are not guaranteed realized displacements. Small realized motion does not identify contact or controller failure.',
            'No ground-truth object poses, contact labels, alternative rollout, or new inference were inspected.',
            'Incomplete terminal outcomes remain unscored; this report does not invent a success rate.',
            'The recorded cumulative ledger is retained, including unresolved reservations; this helper does not independently audit token pricing.',
        ]}
    return report, records


def plot_report(report, records, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    n = len(records)
    fields = report['per_refresh']
    positions = np.asarray([records[0]['state'][:3]] + [r['next_state'][:3] for r in records])
    fig, axes = plt.subplots(2,2,figsize=(13,9))
    fig.suptitle(f'One bowl episode: {n:,} recorded actions | {report["terminal_status"]}',fontsize=16,y=.975)
    ax=axes[0,0]
    lc=LineCollection(np.stack((positions[:-1,:2],positions[1:,:2]),axis=1),cmap='viridis',
                      norm=plt.Normalize(0,n),linewidths=2)
    lc.set_array(np.arange(n));ax.add_collection(lc);ax.autoscale();ax.set_aspect('equal','box')
    ax.scatter(*positions[0,:2],color='#222',marker='o',s=45,label='Start',zorder=3)
    ax.scatter(*positions[-1,:2],color='#d62728',marker='X',s=65,label=f'After action {n-1}',zorder=3)
    ax.set(xlabel='World x (m)',ylabel='World y (m)',title='Measured hand path, viewed from above')
    ax.legend(loc='best',fontsize=8);fig.colorbar(lc,ax=ax,label='Executed action index',shrink=.82)
    ax=axes[0,1];ax.plot(np.arange(n+1),positions[:,2],color='#2166ac',lw=1.6)
    for segment in report['source_segments'][1:]:
        ax.axvline(segment['action_start'],color='#999',lw=.8,ls='--')
    ax.set(xlabel='Completed actions',ylabel='Measured world z (m)',title='Measured hand height')
    ax.text(.98,.95,'Dashed lines: continuation boundaries\nReconstruction is excluded from action counts',
            transform=ax.transAxes,ha='right',va='top',fontsize=8)
    ax=axes[1,0];ax.plot(np.arange(n),[r['action'][-1] for r in records],color='#555',lw=2,label='Executed command')
    for command,color,marker in [('open','#2166ac','o'),('hold','#de8f05','v'),('close','#b2182b','s')]:
        steps=[f['step'] for f in fields if f['gripper_command']==command]
        if steps:
            ax.scatter(steps,[records[s]['action'][-1] for s in steps],s=16,color=color,marker=marker,
                       label='Model '+command,zorder=3,alpha=.8)
    ax.set(xlabel='Executed action index',ylabel='Normalized gripper action',title='Recorded gripper commands',ylim=(-1.35,1.25))
    ax.set_yticks([-1,1],labels=['−1: open','+1: close']);ax.legend(loc='upper center',fontsize=8,ncol=2)
    commands=report['observations']['model_gripper_commands']
    ax.text(.5,.47,', '.join(f'{commands.get(c,0)} {c}' for c in ('open','hold','close'))+' fields\nHold retains the existing actuator command',
            transform=ax.transAxes,ha='center',va='center',fontsize=10)
    ax=axes[1,1]
    for name,label,color in [('affine','Affine','#b2182b'),('separable_quadratic','Separable quadratic','#2166ac')]:
        selected=[f for f in fields if f['fit'][name]['rms_residual_over_value_std'] is not None]
        ax.plot([f['step'] for f in selected],[max(1e-16,f['fit'][name]['rms_residual_over_value_std']) for f in selected],
                '.',ms=4,label=label,color=color)
    exact=report['observations']['field_structure']['fits_within_threshold']
    ax.set_yscale('log');ax.set_ylim(1e-16,2)
    ax.set(xlabel='Field refresh action index',ylabel='Fit RMS / valid-sample standard deviation',
           title=f'Unitless fields: {exact["separable_quadratic"]}/{len(fields)} quadratic within 1e−10')
    ax.legend(loc='center left',fontsize=8)
    ax.text(.98,.30,f'{exact["arbitrary_additive_separable"]}/{len(fields)} fit f(x)+g(y)+h(z)\nMasks excluded from scalar fits',
            transform=ax.transAxes,ha='right',fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.subplots_adjust(top=.90,bottom=.12,hspace=.42,wspace=.36)
    outcome='Simulator success recorded.' if report['libero_success'] is True else 'Incomplete episode; no success-rate claim.'
    fig.text(.5,.025,'Saved observations only; no target/object ground truth, causal conclusion, or alternative rollout.\n'+outcome+
             ' Field values are unitless model outputs.',ha='center',fontsize=9)
    fig.savefig(path,dpi=180,facecolor='white');plt.close(fig)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output-prefix',help='Default: SOURCE/diagnostic_full_chain; existing files are refused.')
    args=parser.parse_args(argv)
    prefix=Path(args.output_prefix) if args.output_prefix else Path(args.source)/'diagnostic_full_chain'
    json_path,figure_path=prefix.with_suffix('.json'),prefix.with_suffix('.png')
    require(not json_path.exists() and not figure_path.exists(), 'Output already exists; historical reports are not overwritten')
    report,records=build_report(args.source)
    require(prefix.parent.is_dir(), 'Output directory must already exist')
    plot_report(report,records,figure_path)
    json_path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':'written','actions':len(records),'executed_fields':len(report['per_refresh']),
        'terminal_status':report['terminal_status'],'json':str(json_path),'figure':str(figure_path)}))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
