"""Native V16 wire stays exact while V17 changes request presentation only."""
import copy
import json

import pytest

from gpt_field.distance_tasks import resolve_distance_task
from gpt_field.visual_field_subscription_v16 import output_schema as schema_v16
from gpt_field.visual_field_subscription_v17 import (
    VisualFieldSubscriptionClientV17, output_schema, shared_guidance, HISTORY_VERSION,
)
from gpt_field.visual_feedback_compact_v17 import compact_feedback
from test_field_history import FakeTransport, envelope
from test_visual_field_protocol import observation
from test_visual_field_v16_protocol import prediction, neighbor
from test_visual_field_v7_protocol import current_record, stream_error


def client(*responses):
    transport = FakeTransport(*(value if isinstance(value, BaseException) else envelope(
        value, model="gpt-6-astra", reasoning={"effort":"medium"}, id=f"v17_{i}")
        for i,value in enumerate(responses)))
    return VisualFieldSubscriptionClientV17(token_provider=lambda:"offline-oauth", transport=transport), transport


def test_native_inventory_wire_history_and_inputs_stay_exact():
    value=prediction(controller="visual_field");value['scene_inventory'].append(neighbor())
    original=copy.deepcopy(value);instance, transport=client(value,prediction())
    obs=observation().to_dict();before=copy.deepcopy(obs)
    result=instance.predict(obs);instance.predict(observation(1).to_dict())
    assert result['raw_wire_prediction']==value==original and obs==before
    p=transport.calls[-1]
    history=[json.loads(c['text'])for m in p['input']if m.get('role')=='assistant'
             for c in m.get('content',[])if c['type']=='output_text']
    assert history==[original]
    assert p['text']['format']['schema']==schema_v16()==output_schema()
    assert p['text']['format']['name']=='visual_field_rgb_joints_v17'
    assert instance.history_stats()['protocol']==HISTORY_VERSION


@pytest.mark.parametrize('omitted',[False,True])
def test_compacted_late_memory_preserves_exact_unbounded_target_fallback(omitted):
    value=prediction(controller='visual_field');identity=value['target_id']
    if omitted:value['scene_inventory']=[]
    else:value['scene_inventory'][0]['geometry']=None
    current={'center_offset_m':[.71,-.82,.93],'half_extents_m':[.02]*3,'uncertainty_m':.03}
    feedback={'scene_inventory':{'entries':[{'id':identity,'current_geometry':copy.deepcopy(current),
        'world_geometry':{'center_xyz_m':[.5,.3,1.0],'half_extents_m':[.02]*3,'uncertainty_m':.03},
        'views':{'external':{'current_roi_uv':None,'partial_current_support':True,
                           'current_visible_roi_uv':[.1,.1,.3,.3],'support_status':'partial'}}}]}}
    instance,_=client(value);instance.observable_context={'visual_feedback':compact_feedback(feedback)}
    result=instance.predict(observation().to_dict())
    assert result['target_geometry']['center_offset_m']==current['center_offset_m']
    assert result['raw_wire_prediction']==value


def test_bounded_same_observation_semantic_and_stream_retries_inherited():
    invalid=prediction();invalid['direct_actions']=None
    valid=prediction(phase='recover');valid['scene_inventory']=[];valid['direct_actions']=[[0.]*7 for _ in range(5)]
    instance,transport=client(stream_error(),invalid,valid)
    assert instance.predict(observation().to_dict())['raw_wire_prediction']==valid
    assert len(transport.calls)==instance.budget_snapshot()['subscription_inference_calls']==3
    assert instance.budget_snapshot()['api_key_calls']==0
    assert len({current_record(p)['observation_index']for p in transport.calls})==1
    assert all(p['reasoning']=={'effort':'medium'}for p in transport.calls)
    text=shared_guidance(resolve_distance_task('bowl_plate'))
    assert 'partial_current_support' in text and 'actual candidate tool sweep' in text
    assert 'NOT complete object visibility or clearance' in text and 'NEVER hand/palm' in text
    assert 'extent_visibility=boundary_limited takes precedence' in text
