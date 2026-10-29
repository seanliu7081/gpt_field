"""Actual V36 client/policy routing with fake model, simulator and clock."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from test_visual_field_v7_protocol import stream_error, current_record, current_images
from test_visual_field_v8_runner import fixtures, runner, run_args
from test_visual_field_v21_runner import checkpoints
from test_visual_field_v33_runner import gripper_request
from gpt_field.visual_feedback_compact_v35 import expand_current_feedback


@pytest.mark.parametrize('outcome', ['success', 'collision', 'simultaneous', 'interrupt'])
def test_v36_actual_route_outcome_priority_privacy_timing_and_source_freeze(tmp_path, monkeypatch, outcome):
    clock = [100.]
    monkeypatch.setattr(runner.time, 'monotonic', lambda: clock[0])
    snapshots = checkpoints(monkeypatch)
    Sim, Recorder, transport, sims = fixtures(clock, [stream_error(), gripper_request()],
        outcome=outcome, terminal_step=8, request_action_counts=[0, 0])
    args = run_args(tmp_path, method_version='v36')
    class BoundaryStop(Sim):
        def step(self, action):
            result = super().step(action)
            if len(self.actions) == 8:
                (args.output / 'operator_stop.request.json').write_text('{"reason":"offline boundary stop"}')
            return result
    result = runner.execute(args, token_provider=lambda: 'offline-oauth', transport=transport,
                            sim_factory=BoundaryStop, recorder_factory=Recorder)
    assert result['method_version'] == 'v36' and result['steps'] == 8
    assert result['gripper_actions'] == 8 and result['field_actions'] == result['direct_actions'] == 0
    assert result['status'] == {'success': 'success', 'collision': 'collision_failure',
        'simultaneous': 'collision_failure', 'interrupt': 'user_stop'}[outcome]
    assert result['task_success'] is result['completion_time_eligible'] is (outcome == 'success')
    assert len(transport.calls) == result['budget']['subscription_inference_calls'] == 2
    assert result['budget']['unknown_usage_calls'] == 1 and result['budget']['api_key_calls'] == 0
    first, retry = map(current_record, transport.calls)
    assert first['observation_index'] == retry['observation_index'] == 0
    assert first['observation/state'] == retry['observation/state']
    assert current_images(transport.calls[0]) == current_images(transport.calls[1])
    assert all(p['model'] == 'gpt-6-astra' and p['reasoning'] == {'effort': 'medium'}
               and p['text']['format']['name'] == 'visual_field_rgb_joints_v36' for p in transport.calls)
    assert 'V8_EVALUATOR_PRIVATE' not in json.dumps(transport.calls)
    semantics = expand_current_feedback(first['observable_context']['visual_feedback'])['geometry_semantics']
    assert semantics['version'] == 'ongoing-field-objective-review-v36'
    assert 'recovery_to_approach_proposals' not in semantics  # Static prose is supplied in instructions/manifest.
    assert 'TASK RESUMPTION.' in transport.calls[0]['instructions']
    assert 'ONGOING OBJECTIVE REVIEW.' in transport.calls[0]['instructions']
    assert set(first) == {'type', 'observation_index', 'prompt', 'observation/state',
        'observation/joint_position', 'observation/joint_velocity', 'observable_context'}
    current_feedback = expand_current_feedback(first['observable_context']['visual_feedback'])
    assert current_feedback['ordinary_replan_previews'] is None
    assert not current_feedback['ongoing_objective_review']['review_required']
    rows = [json.loads(line) for line in (args.output / 'trace.jsonl').read_text().splitlines()]
    assert [r['event'] for r in rows if r['event'] in ('action', 'executed', 'feedback')] == ['action', 'executed', 'feedback'] * 8
    if outcome == 'interrupt':
        assert result['error']['type'] == 'OperatorStopFile'
        assert rows[-1]['event'] == 'operator_stop' and rows[-1]['completed_actions'] == 8
    else:
        assert not any(row['event'] == 'operator_stop' for row in rows)
    final = json.loads((args.output / 'run_state.json').read_text())
    assert final['finalized'] and final['stage'] == 'finished'
    assert final['elapsed_control_s'] == result['control_elapsed_s'] == 20.
    assert final['control_without_video_s'] == result['control_without_video_s'] == 16.
    assert final['timing_components']['video_render_s'] == 4.
    assert final['timing_components']['model_pipeline_s'] == 14.
    assert len([row for row in snapshots if row['stage'] == 'observed']) == 8
    assert not (args.output / 'run_state.json.tmp').exists()
    manifest = json.loads((args.output / 'manifest.json').read_text())
    assert manifest['method_version'] == 'v36' and manifest['shots'] == 0
    assert all(manifest[key] is None for key in ('max_policy_steps', 'max_api_calls', 'max_wall_time_s'))
    assert manifest['feedback_presentation']['version'] == 'current-feedback-exact-shared-facts-v35-v1'
    assert manifest['hand_outline_runtime_change'].startswith('same-source-coordinate-continuity-only')
    full_semantics = manifest['model_geometry_admission']
    assert full_semantics['recovery_to_approach_proposals']['explicit_model_choice_required']
    assert full_semantics['final_feedback_contract']['observer_and_control_updates_keep_original_order']
    assert not full_semantics['final_feedback_contract']['across_observation_cache']
    assert 'explicit noncontact approach transition' in manifest['task_resumption_contract']
    assert 'No across-observation cache' in manifest['feedback_transaction_contract']
    assert 'original role context is restored' in manifest['prospective_recovery_role_contract']
    review = full_semantics['ongoing_objective_review_contract']
    assert review['review_requires_explicit_model_decision']
    assert review['ordinary_field_replacement_remains_available_subject_to_admission']
    assert not review['review_completes_goal_or_changes_budget']
    assert not review['automatic_action_or_stage_change']
    assert 'before the next intent acknowledges it' in manifest['ongoing_visual_review_contract']
    assert 'not validated visual corrections or executable proposal IDs' in manifest['ordinary_replan_preview_contract']
    feedback_rows = [row['feedback'] for row in rows if row['event'] == 'feedback']
    assert all(not row['ongoing_objective_review']['review_required'] for row in feedback_rows)
    assert all(row['ordinary_replan_previews'] is None for row in feedback_rows)
    assert feedback_rows[-1]['ongoing_objective_review']['observed_transitions'] == 8
    old_sources = manifest['v35_runtime_source_sha256']
    current_sources = manifest['v36_runtime_source_sha256']
    assert set(current_sources) - set(old_sources) == {
        'visual_field_policy_v36.py', 'visual_field_subscription_v36.py',
        'visual_intent_watch_v36.py', 'visual_intent_replan_v36.py'}
    assert set(old_sources) <= set(current_sources)
    assert 'visual_hand_outline_v34.py' not in current_sources
    assert full_semantics['wrist_navigation_contract']['explicit_model_selection_required']
    assert not full_semantics['wrist_navigation_contract']['depth_or_contact_alignment_estimated']
    for name, digest in current_sources.items():
        assert hashlib.sha256((args.output / 'frozen_source' / name).read_bytes()).hexdigest() == digest
        assert digest == manifest['software']['source_sha256'][name]
    assert sims[0].closed and not result['cleanup_errors']


def test_v36_remains_opt_in_and_rejects_demo_or_direct_route():
    default = runner.method_components(SimpleNamespace(controller='hybrid', demo=None))
    assert default[0] is runner.VisualFieldSubscriptionClient
    for values in ({'controller': 'direct', 'demo': None}, {'controller': 'hybrid', 'demo': 'demo.json'}):
        with pytest.raises(ValueError, match='zero-shot Hybrid only'):
            runner.method_components(SimpleNamespace(method_version='v36', **values))
