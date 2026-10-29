"""Measurement-epoch guidance and unchanged V25 wire/transport; no live calls."""
import copy
import json

import pytest

from gpt_field.astra import APIError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v25 import output_schema as schema_v25, geometry_semantics as semantics_v25
from gpt_field.visual_field_subscription_v26 import (
    VisualFieldSubscriptionClientV26, output_schema, shared_guidance, geometry_semantics,
    VERSION, HISTORY_VERSION, FEEDBACK_VERSION,
)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v25_protocol import prediction
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def client(*responses):
    transport=FakeTransport(*(r if isinstance(r,BaseException) else envelope(
        r,model='gpt-6-astra',reasoning={'effort':'medium'},id=f'v26_{i}')for i,r in enumerate(responses)))
    return VisualFieldSubscriptionClientV26(token_provider=lambda:'offline-oauth',transport=transport),transport


def test_same_schema_current_risk_not_motion_and_no_permission_from_epoch():
    assert output_schema()==schema_v25()
    text=shared_guidance(resolve_distance_task('bowl_plate'))
    for phrase in ('translation_guard.measurement_representation','not robot motion',
                   'without erasing prior valid worsening','enlarged rectangle can reveal current risk',
                   'new epoch or inconclusive gap alone grants neither clearance nor execution permission',
                   'hold drift consumes','phase=recover'):
        assert phrase in text
    assert 'cabinet' not in text.lower()
    with pytest.raises(ValueError):shared_guidance(resolve_distance_task('bowl_plate'),True)
    semantic=geometry_semantics();measurement=semantic['measurement_representation']
    assert not measurement['accepted_feature_or_extent_revision_is_observed_motion']
    assert measurement['current_revised_extent_and_all_scene_hypotheses_retained']
    assert measurement['prior_valid_worsening_retained']
    assert not measurement['new_epoch_grants_motion_permission']
    assert not measurement['new_epoch_replenishes_physical_allowance']
    assert not measurement['clearance_certified']
    matching=semantic['neighbor_descriptor_matching']
    assert matching['scope']=='opt_in_projected_neighbor_tracker_only'
    assert matching['equality_space']=='exact_bilinear_sampled_grayscale_descriptor_not_whole_RGB'
    assert matching['legacy_texture_uniqueness_forward_backward_and_consensus_checks_retained']
    assert not matching['static_object_or_camera_assumed']
    assert not matching['tool_payload_and_inventory_trackers_changed']
    assert not matching['risk_thresholds_or_physical_allowances_changed']
    assert not matching['clearance_certified']
    for key in ('finite_reverse_route','initial_uncertainty_egress','recent_response_evidence','recovery_route_selection'):
        assert semantic[key]==semantics_v25()[key]


def test_current_epoch_risk_and_debt_feedback_survive_and_raw_history_is_exact():
    first=prediction();second=prediction();instance,transport=client(first,second)
    instance.predict(observation().to_dict());history=copy.deepcopy(instance._history)
    evidence={'epoch':3,'per_neighbor_epoch':{'fixture':2},'last_revision':{'is_observed_motion':False,'tool_changed':True},
        'reverse_segments':[{'baseline_epoch':3,'current_epoch':3,'measured_travel_m':.002}],
        'new_unknown_key':{'keep_exact':True}}
    feedback={'translation_guard':{'measurement_representation':evidence,'historical_worsening_ids':['fixture'],
        'remaining_measured_travel_m':.005,'current_overlap':True,'latest_reason':'current_constraint'},
        'recovery_route_binding':{'recovery_route':'reverse_observed'},'unknown_context_extension':[1,2,3]}
    instance.observable_context={'visual_feedback':copy.deepcopy(feedback)}
    result=instance.predict(observation(1).to_dict());payload=transport.calls[-1]
    assert result['raw_wire_prediction']==second
    assert instance._history[:len(history)]==history
    current=current_record(payload)['observable_context']['visual_feedback']
    assert {k:current[k]for k in feedback}==feedback
    assert set(current)-set(feedback)=={'feedback_representation','presentation_v21'}
    assistants=[json.loads(p['text'])for m in payload['input']if m.get('role')=='assistant'
        for p in m.get('content',[])if p.get('type')=='output_text']
    assert assistants==[first]
    assert payload['text']['format']['name']=='visual_field_rgb_joints_v26'
    assert instance.prompt_version==VERSION+'-hybrid'
    assert instance.history_stats()['protocol']==HISTORY_VERSION
    assert instance.history_stats()['older_feedback_presentation']==FEEDBACK_VERSION


@pytest.mark.parametrize('order',['stream_first','schema_first'])
def test_inherited_repair_and_stream_retry_bound_to_same_observation(order):
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


def test_no_quota_retry_or_fallback():
    instance,transport=client(stream_error(code='insufficient_quota',status=429),prediction())
    with pytest.raises(APIError):instance.predict(observation().to_dict())
    assert len(transport.calls)==1 and instance.budget_snapshot()['api_key_calls']==0
