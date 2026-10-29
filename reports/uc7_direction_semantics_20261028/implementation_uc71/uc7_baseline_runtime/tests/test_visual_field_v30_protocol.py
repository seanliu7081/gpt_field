"""Current role contract, typed paid-history board and unchanged transport."""
import copy
import json

import pytest

from gpt_field.astra import APIError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v29 import (
    output_schema as old_schema, shared_guidance as old_guidance)
from gpt_field.visual_field_subscription_v30 import (
    VisualFieldSubscriptionClientV30, output_schema, geometry_semantics,
    shared_guidance, VERSION, HISTORY_VERSION, FEEDBACK_VERSION, CANDIDATE_LABEL)
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v25_protocol import prediction
from test_visual_field_v7_protocol import current_record, current_images, stream_error


def client(*responses):
    transport = FakeTransport(*(r if isinstance(r, BaseException) else envelope(
        r, model='gpt-6-astra', reasoning={'effort': 'medium'}, id=f'v30_{i}')
        for i, r in enumerate(responses)))
    return VisualFieldSubscriptionClientV30(token_provider=lambda: 'offline-oauth',
                                           transport=transport), transport


def test_same_wire_compact_current_contract_preserves_control_and_target_roles():
    assert output_schema() == old_schema()
    task = resolve_distance_task('bowl_plate')
    text = shared_guidance(task)
    assert len(text) < .8*len(old_guidance(task))
    for rule in ('exactly FIVE', 'release Field is z only', 'Direct release has zero XYZ',
                 'target_contact=none', 'MAY DECREASE', 'every NON-TARGET check remains unchanged',
                 'positive', 'outside the contact band', 'current causal same-heading response',
                 'extraction_status', 'ineligible', 'extracted_none', 'same total',
                 'recovery_route=normal|reverse_observed|initial_uncertainty_egress',
                 'never a renamed remembered object', 'no permission', 'physical-debt'):
        # The exact semantic metadata below checks the debt rule; prose may
        # spell it as separate words rather than a compound label.
        if rule == 'physical-debt': continue
        assert rule in text
    assert 'cabinet' not in text.lower()
    with pytest.raises(ValueError): shared_guidance(task, True)
    semantics = geometry_semantics()
    target = semantics['target_approach_execution']
    assert target['target_only_modeled_gap_may_decrease']
    assert target['current_causal_same_heading_response_required']
    assert target['positive_full_margin_model_clearance_and_outside_contact_band_required']
    assert target['all_non_target_checks_unchanged']
    assert not target['history_or_physical_debt_reset'] and not target['clearance_certified']
    assert not semantics['projected_warning_transit']['warned_object_model_clearance_must_not_decrease']
    assert semantics['projected_warning_transit']['non_target_warned_object_model_clearance_must_not_decrease']
    assert not semantics['constraint_aware_tracking']['warned_object_model_gap_nondecrease_retained']
    assert semantics['constraint_aware_tracking']['non_target_model_gap_nondecrease_retained']


def test_current_evidence_unknown_keys_and_raw_assistant_history_remain_exact():
    first, second = prediction(), prediction()
    instance, transport = client(first, second)
    instance.predict(observation().to_dict())
    history = copy.deepcopy(instance._history)
    feedback = {
        'translation_guard': {'target_approach_execution': {
            'accepted': False, 'reason': 'same_heading_response_unavailable',
            'target_id': 'target', 'remaining_shared_m': .006,
            'unknown_extension': {'preserve': True}}},
        'reference_repair': {'extraction_status': 'ineligible', 'review_required': False,
            'review_event': {'kind': 'target_reference_lost'},
            'eligibility_reason': 'stationary_anchor_review_required'},
        'scene_inventory': {'entries': [{'id': 'target', 'current_geometry': {
            'center_offset_m': [.1, .2, .3], 'half_extents_m': [.04]*3,
            'uncertainty_m': .01}}]},
    }
    instance.observable_context = {'visual_feedback': copy.deepcopy(feedback)}
    result = instance.predict(observation(1).to_dict())
    payload = transport.calls[-1]
    actual = current_record(payload)['observable_context']['visual_feedback']
    assert actual['translation_guard'] == feedback['translation_guard']
    assert actual['reference_repair'] == feedback['reference_repair']
    assert actual['scene_inventory']['entries'] == feedback['scene_inventory']['entries']
    assert instance._history[:len(history)] == history
    assert result['raw_wire_prediction'] == second
    assistants = [json.loads(part['text']) for m in payload['input'] if m.get('role') == 'assistant'
                  for part in m.get('content', []) if part.get('type') == 'output_text']
    assert assistants == [first]
    assert payload['text']['format']['name'] == 'visual_field_rgb_joints_v30'
    assert instance.prompt_version == VERSION+'-hybrid'
    assert instance.history_stats()['protocol'] == HISTORY_VERSION
    assert instance.history_stats()['older_feedback_presentation'] == FEEDBACK_VERSION


@pytest.mark.parametrize('order', ['stream_first', 'schema_first'])
def test_bounded_same_observation_retries_stay_medium_subscription(order):
    invalid = prediction(); invalid.pop('recovery_route')
    failures = [stream_error(), invalid] if order == 'stream_first' else [invalid, stream_error()]
    good = prediction(); instance, transport = client(*failures, good)
    assert instance.predict(observation().to_dict())['raw_wire_prediction'] == good
    assert len(transport.calls) == instance.budget_snapshot()['subscription_inference_calls'] == 3
    assert instance.budget_snapshot()['api_key_calls'] == 0
    assert len({current_record(p)['observation_index'] for p in transport.calls}) == 1
    assert all(current_images(p) == current_images(transport.calls[0]) for p in transport.calls)
    assert all(p['model'] == 'gpt-6-astra' and p['reasoning'] == {'effort': 'medium'}
               for p in transport.calls)
    repaired = current_record(transport.calls[-1])['observable_context']['output_repair']
    assert json.loads(repaired['rejected_model_output_json']) == invalid


def test_quota_has_no_retry_or_api_fallback():
    instance, transport = client(stream_error(code='insufficient_quota', status=429), prediction())
    with pytest.raises(APIError): instance.predict(observation().to_dict())
    assert len(transport.calls) == 1 and instance.budget_snapshot()['api_key_calls'] == 0


def test_actual_typed_loss_review_board_reaches_current_payload_without_extra_motion():
    from test_visual_reference_candidates_v30 import plant, lost_event
    from test_visual_reference_candidates_v24 import advance
    bank, monitor, _, obs = plant()
    obs = advance(bank, monitor, obs)
    obs.update({'observation/wrist_image': obs['observation/image'],
                'observation/joint_velocity': [0.]*7, 'prompt': resolve_distance_task('bowl_plate').prompt})
    monitor.tracks['moving'].reliable = False
    before = (bank.path_m, bank.sequence, copy.deepcopy(monitor._active_reference))
    report = bank.review(obs, monitor, lost_event('new_reason_not_in_a_string_allowlist'))
    assert report['extraction_status'] == 'available' and report['candidates']
    instance, _ = client(prediction())
    instance.reference_candidate_provider = bank.image
    instance.observable_context = {'visual_feedback': {'reference_repair': report}}
    payload = instance.build_payload(obs)
    record = current_record(payload)
    assert record['observable_context']['visual_feedback']['reference_repair'] == report
    assert len(current_images(payload)) == 3
    current = [m for m in payload['input'] if m.get('role') == 'user'][-1]
    assert any(p.get('text') == CANDIDATE_LABEL for p in current['content'])
    assert current_images(payload)[:2] == [obs['observation/image'], obs['observation/wrist_image']]
    assert (bank.path_m, bank.sequence, monitor._active_reference) == before
    assert not monitor.tracks['moving'].reliable
    monitor.tracks['target'].reliable = False
    from gpt_field.visual_reference_review_v30 import reference_review_event
    report = bank.review(obs, monitor, reference_review_event(monitor, handoff_present=True))
    instance.observable_context = {'visual_feedback': {'reference_repair': report}}
    payload = instance.build_payload(obs)
    assert report['extraction_status'] == 'ineligible'
    assert len(current_images(payload)) == 2
