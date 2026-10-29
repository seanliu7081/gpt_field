"""Subscription-only wire and exact current presentation with fake transport."""
import copy
import json

import pytest

from gpt_field.astra import APIError
from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_feedback_compact_v35 import expand_current_feedback
from gpt_field.visual_field_subscription_v34 import VisualFieldSubscriptionClientV34, output_schema as schema34
from gpt_field.visual_field_subscription_v35 import VisualFieldSubscriptionClientV35, output_schema, shared_guidance
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v7_protocol import current_record, current_images, stream_error
from test_visual_field_v32_protocol import prediction


def client(cls, *responses):
    transport = FakeTransport(*(response if isinstance(response, BaseException) else envelope(
        response, model='gpt-6-astra', reasoning={'effort': 'medium'}, id=f'v35_{i}')
        for i, response in enumerate(responses)))
    return cls(token_provider=lambda: 'offline-oauth', transport=transport), transport


def test_current_factoring_is_exact_and_leaves_raw_history_images_and_inputs_unchanged():
    values = [prediction(), prediction()]
    old, old_transport = client(VisualFieldSubscriptionClientV34, *values)
    new, new_transport = client(VisualFieldSubscriptionClientV35, *values)
    facts = {'remaining_m': .01234567890123, 'rejected': True,
             'reason': 'observed current warning ' * 15}
    context = {'visual_feedback': {'repeated': [copy.deepcopy(facts) for _ in range(12)],
        'current_rejection': {'accepted': False, 'allowance_m': 0., 'evidence': facts}}}
    before = copy.deepcopy(context)
    for index in range(2):
        for instance in (old, new):
            instance.observable_context = copy.deepcopy(context)
            instance.predict(observation(index).to_dict())
    for previous, current in zip(old_transport.calls, new_transport.calls):
        a, b = current_record(previous), current_record(current)
        b['observable_context']['visual_feedback'] = expand_current_feedback(
            b['observable_context']['visual_feedback'])
        assert a == b
        assert current_images(previous) == current_images(current)
        assert current['text']['format']['schema'] == schema34() == output_schema()
        assert current['text']['format']['name'] == 'visual_field_rgb_joints_v35'
    raw = lambda p: [json.loads(part['text']) for message in p['input']
        if message.get('role') == 'assistant' for part in message['content'] if part['type'] == 'output_text']
    assert raw(old_transport.calls[-1]) == raw(new_transport.calls[-1]) == values[:1]
    assert context == before and new.observable_context == before
    text = shared_guidance(resolve_distance_task('bowl_plate'))
    assert 'No executable offer means report' not in text
    assert 'wrist_bearing' in text and 'desired UV is a goal coordinate' in text
    assert '$same_v35' in text and 'execution_limit_sources' in text
    with pytest.raises(ValueError): shared_guidance(resolve_distance_task('bowl_plate'), True)


def test_native_repair_keeps_current_observation_medium_and_no_api_fallback():
    valid = prediction()
    instance, transport = client(VisualFieldSubscriptionClientV35, stream_error(), valid)
    assert instance.predict(observation().to_dict())['raw_wire_prediction'] == valid
    assert len(transport.calls) == 2 and instance.budget_snapshot()['api_key_calls'] == 0
    assert all(p['model'] == 'gpt-6-astra' and p['reasoning'] == {'effort': 'medium'} for p in transport.calls)
    assert current_images(transport.calls[0]) == current_images(transport.calls[1])
    assert current_record(transport.calls[0])['observation_index'] == current_record(transport.calls[1])['observation_index']
    blocked, attempts = client(VisualFieldSubscriptionClientV35, stream_error(code='insufficient_quota', status=429), valid)
    with pytest.raises(APIError): blocked.predict(observation().to_dict())
    assert len(attempts.calls) == 1
