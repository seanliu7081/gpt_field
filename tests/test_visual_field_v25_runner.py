"""Actual V25 client/policy route with fake inference/simulator and a clock."""
import hashlib
import json
import pytest

from test_visual_field_v7_protocol import stream_error,current_record,current_images
from test_visual_field_v8_runner import fixtures,runner,run_args
from test_visual_field_v21_runner import gripper_request as old_gripper_request,checkpoints

def gripper_request():
    value=old_gripper_request();value["recovery_route"]="normal";return value


@pytest.mark.parametrize('outcome',['success','collision','simultaneous','interrupt'])
def test_v25_route_boundary_priority_retry_privacy_timing_and_frozen_manifest(tmp_path,monkeypatch,outcome):
    clock=[100.];monkeypatch.setattr(runner.time,'monotonic',lambda:clock[0])
    snapshots=checkpoints(monkeypatch)
    Sim,Recorder,transport,sims=fixtures(clock,[stream_error(),gripper_request()],
        outcome=outcome,terminal_step=8,request_action_counts=[0,0])
    config=run_args(tmp_path,method_version='v25')
    class BoundaryStop(Sim):
        def step(self,action):
            value=super().step(action)
            if len(self.actions)==8:
                (config.output/'operator_stop.request.json').write_text('{"reason":"offline boundary stop"}')
            return value
    result=runner.execute(config,token_provider=lambda:'offline-oauth',transport=transport,
        sim_factory=BoundaryStop,recorder_factory=Recorder)
    assert result['method_version']=='v25'and result['steps']==8
    assert result['gripper_actions']==8 and result['field_actions']==result['direct_actions']==0
    assert result['status']=={'success':'success','collision':'collision_failure',
        'simultaneous':'collision_failure','interrupt':'user_stop'}[outcome]
    assert result['task_success']is result['completion_time_eligible']is(outcome=='success')
    assert len(transport.calls)==result['budget']['subscription_inference_calls']==2
    assert result['budget']['unknown_usage_calls']==1 and result['budget']['api_key_calls']==0
    a,b=map(current_record,transport.calls)
    assert a['observation_index']==b['observation_index']==0
    assert a['observation/state']==b['observation/state']
    contract=a['observable_context']['visual_feedback']['recovery_execution_contract']
    assert contract['explicit_model_choice_required'] is True
    assert contract['clearance_certified'] is contract['physical_budgets_reset'] is False
    assert 'uncertainty_egress' in a['observable_context']['visual_feedback']
    assert transport.calls[0]['text']['format']['schema']['properties']['recovery_route']['enum']==['normal','reverse_observed','initial_uncertainty_egress']
    assert current_images(transport.calls[0])==current_images(transport.calls[1])
    assert all(p['model']=='gpt-6-astra'and p['reasoning']=={'effort':'medium'}and
        p['text']['format']['name']=='visual_field_rgb_joints_v25'for p in transport.calls)
    assert 'V8_EVALUATOR_PRIVATE'not in json.dumps(transport.calls)
    rows=[json.loads(s)for s in(config.output/'trace.jsonl').read_text().splitlines()]
    events=[r['event']for r in rows if r['event']in('action','executed','feedback')]
    assert events==['action','executed','feedback']*8
    if outcome=='interrupt':
        assert result['error']['type']=='OperatorStopFile'
        assert rows[-1]['event']=='operator_stop'and rows[-1]['completed_actions']==8
    else:assert not any(r['event']=='operator_stop'for r in rows)
    final=json.loads((config.output/'run_state.json').read_text())
    assert final['finalized']and final['stage']=='finished'
    assert final['elapsed_control_s']==result['control_elapsed_s']==20.
    assert final['control_without_video_s']==result['control_without_video_s']==16.
    assert final['timing_components']['video_render_s']==4.
    assert final['timing_components']['model_pipeline_s']==14.
    assert len([s for s in snapshots if s['stage']=='observed'])==8
    assert not(config.output/'run_state.json.tmp').exists()
    manifest=json.loads((config.output/'manifest.json').read_text())
    assert manifest['method_version']=='v25'and manifest['shots']==0
    assert all(manifest[k]is None for k in('max_policy_steps','max_api_calls','max_wall_time_s'))
    assert manifest['feedback_presentation']['version']=='current-full-older-action-result-v24-v1'
    assert manifest['model_geometry_admission']['heading_constrained_recovery']['debt_reset']is False
    assert manifest['model_geometry_admission']['initial_uncertainty_egress']['contextual_budget_reset_on_rename_or_reseed'] is False
    required={'visual_field_v25.py','visual_field_policy_v25.py','visual_field_subscription_v25.py','robot_translation_response_v25.py',
        'visual_translation_guard_v21.py','visual_reference_track_v21.py','visual_monitor_v21.py',
        'visual_task_progress_v21.py','visual_feedback_compact_v21.py','visual_scene_guard_v17.py',
        'uncertainty_egress_v23.py','visual_egress_evidence_v23.py',
        'visual_translation_guard_v24.py','visual_translation_guard_v25.py','visual_route_exposure_v24.py',
        'visual_reference_candidates_v24.py','visual_scene_guard_v23.py'}
    assert required<=set(manifest['v25_runtime_source_sha256'])
    for name,digest in manifest['v25_runtime_source_sha256'].items():
        assert hashlib.sha256((config.output/'frozen_source'/name).read_bytes()).hexdigest()==digest
        assert digest==manifest['software']['source_sha256'][name]
    assert sims[0].closed and not result['cleanup_errors']
