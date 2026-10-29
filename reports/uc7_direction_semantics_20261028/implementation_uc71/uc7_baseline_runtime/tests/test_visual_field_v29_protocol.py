"""V29 empirical-risk guidance and unchanged observable subscription protocol."""
import copy
import json

import pytest

from gpt_field.astra import APIError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v26 import output_schema as schema_v26
from gpt_field.visual_field_subscription_v29 import (
    VisualFieldSubscriptionClientV29, output_schema, geometry_semantics,
    shared_guidance, VERSION, HISTORY_VERSION, FEEDBACK_VERSION,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v25_protocol import prediction
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def client(*responses):
    transport=FakeTransport(*(r if isinstance(r,BaseException) else envelope(
        r,model='gpt-6-astra',reasoning={'effort':'medium'},id=f'v29_{i}')for i,r in enumerate(responses)))
    return VisualFieldSubscriptionClientV29(token_provider=lambda:'offline-oauth',transport=transport),transport


def test_same_wire_task_generic_conditional_transit_and_no_new_allowance():
    assert output_schema()==schema_v26()
    text=shared_guidance(resolve_distance_task('bowl_plate'))
    for value in ('recovery_route=normal','translation_mode=transit','target_contact=none','every warned object',
                  'prospectively validated heading-specific relative image flow',
                  'Resolved closing still stops','translation_guard.projected_warning_transit',
                  'existing 8 mm heading and 20 mm shared allowances','Warnings/history/current geometry stay active',
                  'Neither view is a clearance certificate'):
        assert value in text
    assert 'cabinet' not in text.lower()
    with pytest.raises(ValueError):shared_guidance(resolve_distance_task('bowl_plate'),True)
    values=geometry_semantics();rule=values['projected_warning_transit']
    assert rule['scope']=='normal_empty_tool_no_contact_fixed_goal_field_transit_only'
    assert rule['current_full_scene_margin_sweep_required']
    assert rule['warned_object_model_clearance_must_not_decrease']
    assert rule['per_warned_object_prospective_validation_required']
    assert rule['boundary_evidence_must_be_ambiguous_not_resolved_closing']
    assert rule['warnings_history_and_measured_debt_retained']
    assert rule['empirical_boundary_evidence_is_not_calibrated_confidence']
    assert rule['maximum_command_m']==.003 and not rule['new_budget_or_clearance_certificate']
    assert values['measurement_representation']['prior_valid_worsening_retained']
    assert not values['neighbor_descriptor_matching']['static_object_or_camera_assumed']


def test_current_proof_evidence_debt_and_unknown_feedback_survive_with_raw_history():
    first,second=prediction(),prediction();instance,transport=client(first,second)
    instance.predict(observation().to_dict());history=copy.deepcopy(instance._history)
    feedback={'translation_guard':{'projected_warning_transit':{'accepted':False,'reason':'resolved_closing',
        'warned_ids':['object_a'],'physical_budget_replenished':False,'clearance_certified':False,
        'boundary_evidence':{'classification':'closing','gap_change_interval_px':[-4.,-1.]},
        'remaining_shared_m':.006,'unknown_extension':{'retain':True}}},
        'transit_reconciliation_geometry':{'object_proofs':{'object_a':{'gap_nondecreasing':False}},
        'current_goal_xyz_m':[.1,.2,.3]},'task_progress':{'review_required':True}}
    instance.observable_context={'visual_feedback':copy.deepcopy(feedback)}
    result=instance.predict(observation(1).to_dict());payload=transport.calls[-1]
    current=current_record(payload)['observable_context']['visual_feedback']
    assert {key:current[key]for key in feedback}==feedback
    assert instance._history[:len(history)]==history and result['raw_wire_prediction']==second
    assistants=[json.loads(p['text'])for m in payload['input']if m.get('role')=='assistant'
        for p in m.get('content',[])if p.get('type')=='output_text']
    assert assistants==[first]
    assert payload['text']['format']['name']=='visual_field_rgb_joints_v29'
    assert instance.prompt_version==VERSION+'-hybrid'
    assert instance.history_stats()['protocol']==HISTORY_VERSION
    assert instance.history_stats()['older_feedback_presentation']==FEEDBACK_VERSION


@pytest.mark.parametrize('order',['stream_first','schema_first'])
def test_same_observation_counted_bounded_retries_keep_medium_subscription(order):
    invalid=prediction();invalid.pop('recovery_route');good=prediction()
    failures=[stream_error(),invalid]if order=='stream_first'else[invalid,stream_error()]
    instance,transport=client(*failures,good)
    assert instance.predict(observation().to_dict())['raw_wire_prediction']==good
    assert len(transport.calls)==instance.budget_snapshot()['subscription_inference_calls']==3
    assert instance.budget_snapshot()['api_key_calls']==0
    assert len({current_record(p)['observation_index']for p in transport.calls})==1
    assert all(current_images(p)==current_images(transport.calls[0])for p in transport.calls)
    assert all(p['model']=='gpt-6-astra'and p['reasoning']=={'effort':'medium'}for p in transport.calls)
    assert json.loads(current_record(transport.calls[-1])['observable_context']['output_repair']['rejected_model_output_json'])==invalid


def test_quota_has_no_retry_or_api_fallback():
    instance,transport=client(stream_error(code='insufficient_quota',status=429),prediction())
    with pytest.raises(APIError):instance.predict(observation().to_dict())
    assert len(transport.calls)==1 and instance.budget_snapshot()['api_key_calls']==0


def test_paid_review_and_unvalidated_acquisition_are_not_permission_or_identity():
    semantics=geometry_semantics()
    assert not semantics['heading_evidence_acquisition']['physical_budget_replenished']
    assert semantics['heading_evidence_acquisition']['finite_unvalidated_acquisition_is_not_validated_response']
    assert semantics['paid_handoff_reference_review']['maximum_frames']==32
    assert not semantics['paid_handoff_reference_review']['automatic_reference_promotion']
    text=shared_guidance(resolve_distance_task('drawer_bowl'))
    assert 'never select direct while leaving direct_actions null' in text.lower()
    assert 'retain the fixed Field subgoal' in text


def test_v29_axis_tracking_contract_is_scoped_and_keeps_exact_live_feedback():
    semantics=geometry_semantics()['constraint_aware_tracking']
    assert semantics['scope']=='normal_empty_tool_noncontact_axis_constrained_field_with_warned_objects'
    assert semantics['only_conflicting_locked_axis_recentering_may_be_omitted']
    assert semantics['original_and_replacement_exact_candidate_proofs_required']
    assert semantics['warned_object_model_gap_nondecrease_retained']
    assert not semantics['clearance_certified'] and not semantics['physical_budget_replenished']
    value=prediction();instance,transport=client(value)
    facts={'field':{'axis_tracking_constraint':{'nominal_delta_m':[.002,0,-.0001],
        'selected_delta_m':[.002,0,0], 'suppressed_axes':[2], 'fixed_goal_unchanged':True,
        'current_failure_reason':'example_current_reason', 'unknown_extension':{'preserve':True}}}}
    instance.observable_context={'visual_feedback':copy.deepcopy(facts)}
    instance.predict(observation().to_dict())
    assert current_record(transport.calls[0])['observable_context']['visual_feedback']['field']==facts['field']
    assert 'field.axis_tracking_constraint' in transport.calls[0]['instructions']


def test_supported_lifecycle_preserves_old_debt_and_never_claims_clearance():
    contract=geometry_semantics()['validated_transit_lifecycle']
    assert contract['fresh_per_object_held_out_response_required']
    assert contract['supported_travel_accounted_separately']
    assert contract['old_acquisition_debt_preserved']
    assert contract['support_loss_falls_back_to_remaining_old_allowance']
    assert not contract['physical_budget_replenished'] and not contract['clearance_certified']
    text=shared_guidance(resolve_distance_task('bowl_plate'))
    assert 'Supported travel is counted separately' in text
    assert 'Losing support falls back to whatever old allowance remains' in text
    assert 'measured travel spends existing 8 mm heading' not in text
