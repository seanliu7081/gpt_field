"""Read-only planning video provenance and stale-target checks; no simulator/model."""
from copy import deepcopy
import json

import numpy as np
import pytest

from gpt_field.agent_speedup_v2_user_check_video import (
    AgentSpeedUpV2UserCheckVideoRecorder, control_delta, fragment_from_feedback, make_planning_view)
from gpt_field.agent_speedup_v2_video import AgentSpeedUpV2VideoRecorder
from gpt_field.policy import Decision
from test_direct_history_video import MemoryWriter, observation


def fragment():
    return {"status": "running", "current_segment_status": "running", "handoff_kind": "executing",
            "current_segment_index": 0, "current_skill": "move_to_pose", "target_xyz_m": [.1, .2, .9],
            "segments": [{"index": 0, "entry_pose_wxyz": [0., .1, .9, 1., 0., 0., 0.],
                          "request": {"profile": "transit"}}],
            "pending_segments": [{"skill": "move_to_pose", "xyz_m": [.2, .3, .9]},
                                 {"skill": "move_relative", "xyz_m": [.01, 0., 0.]},
                                 {"skill": "approach", "xyz_m": [0., 0., -.02]},
                                 {"skill": "set_gripper", "xyz_m": None}],
            "observation_requested": False}


def feedback(value):
    return {"primitive_feedback": {"fragment": deepcopy(value)}}


def decision(kind="framework_fragment:execute_fragment"):
    return Decision((.1, .1, 0., 0., 0., 0., -1.),
                    {"step": 0, "refreshed": True, "execution_kind": kind,
                     "primitive": "execute_fragment", "macro": "framework_fragment"})


def test_control_arrows_match_euclidean_translation_clamp():
    np.testing.assert_allclose(control_delta([3.,4.,0.],[0.,0.,0.],.02),[.012,.016,0.])
    np.testing.assert_allclose(control_delta([.001,.002,0.],[0.,0.,0.],.02),[.001,.002,0.])
    np.testing.assert_array_equal(control_delta([0.,0.,0.],[0.,0.,0.],.005),[0.,0.,0.])


def test_active_target_uses_recorded_progress_and_never_resolves_future_relative_motion():
    raw=fragment(); before=deepcopy(raw)
    view=make_planning_view(raw,source="recorded_transition_feedback")
    assert raw==before
    assert view["active"] and view["target_xyz_m"]==[.1,.2,.9]
    assert view["translation_limit_m"]==.02
    assert view["pending_absolute_targets"]==[{"segment_index":1,"xyz_m":[.2,.3,.9],"status":"requested_unexecuted"}]
    assert view["unresolved_relative_segments"]==2
    assert view["scalar_planning_field"] is False and view["obstacle_model"] is False
    raw["target_xyz_m"][0]=999
    assert view["target_xyz_m"][0]==.1
    raw["segments"][0]["request"]["profile"]="contact"
    assert make_planning_view(raw)["translation_limit_m"]==.005


@pytest.mark.parametrize("change",[
    {"status":"failed","reason":"target_envelope_rejected"},
    {"status":"succeeded","reason":"observation_required"},
    {"handoff_kind":"budget"}, {"current_segment_status":"succeeded"},
    {"observation_requested":True}])
def test_handoff_rejected_and_completed_targets_never_emit_arrows(change):
    raw=fragment();raw.update(change)
    view=make_planning_view(raw)
    assert not view["active"] and view["target_xyz_m"] is None and view["pending_absolute_targets"]==[]


def test_malformed_or_missing_progress_stays_unavailable():
    for value in (None,{},"bad",{"status":"running"}):
        assert not make_planning_view(value)["active"]
    raw=fragment();raw["segments"][0]["request"]["profile"]=[]
    assert not make_planning_view(raw)["active"]
    raw=fragment();raw["target_xyz_m"]=[float('nan'),0.,0.]
    assert not make_planning_view(raw)["active"]
    assert fragment_from_feedback({"primitive_feedback":None}) is None


def test_recorder_changes_only_path_panels_and_caption_and_emits_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr(AgentSpeedUpV2VideoRecorder,"writer_class",MemoryWriter)
    raw=fragment()
    old=AgentSpeedUpV2VideoRecorder(tmp_path/'original.mp4')
    new=AgentSpeedUpV2UserCheckVideoRecorder(tmp_path/'planning.mp4',planning_provider=lambda:raw)
    obs=observation();act=decision()
    for recorder in (old,new):
        recorder.observe(obs);recorder.decision(obs,act)
    a=np.asarray(old.writer.last_frame);b=np.asarray(new.writer.last_frame)
    assert a.shape==b.shape==(960,1600,3)
    outside=np.ones((960,1600),dtype=bool)
    outside[443:652,425:1577]=False
    outside[919:960,24:1577]=False
    np.testing.assert_array_equal(a[outside],b[outside])
    assert np.any(a[~outside]!=b[~outside])
    original_action=act.action
    after=observation(.003)
    new.executed(after,step=0,success=False,feedback=feedback(raw))
    events=[json.loads(line) for line in new.events_path.read_text().splitlines()]
    assert events[-1]['planning_debug']['source']=='recorded_transition_feedback'
    assert events[-1]['planning_debug']['target_status']=='active_resolved'
    assert events[-1]['action']==list(original_action)
    assert act.action==original_action and raw==fragment()
    old.close(status='user_stop');new.close(status='user_stop')
    last=json.loads(new.events_path.read_text().splitlines()[-1])
    assert last['planning_debug']['target_xyz_m'] is None
    assert last['planning_debug']['target_status']=='terminal'


def test_new_decision_switch_and_missing_feedback_clear_previous_target(tmp_path,monkeypatch):
    monkeypatch.setattr(AgentSpeedUpV2VideoRecorder,"writer_class",MemoryWriter)
    active=[fragment()]
    video=AgentSpeedUpV2UserCheckVideoRecorder(tmp_path/'planning.mp4',planning_provider=lambda:active[0])
    video.observe(observation());video.decision(observation(),decision())
    assert video.planning_debug['active']
    active[0]=None
    video.decision(observation(),decision())
    assert not video.planning_debug['active']
    active[0]=fragment()
    video.decision(observation(),decision())
    video.executed(observation(),step=0,success=False,feedback=None)
    assert not video.planning_debug['active']
    video.decision(observation(),decision('agent_owned:short_target'))
    assert video.planning_debug['target_status']=='not_fragment_control'
    video.executed(observation(),step=0,success=False,feedback=feedback(fragment()))
    assert not video.planning_debug['active']
    video.close(status='user_stop')


def test_provider_error_is_diagnostic_only(tmp_path,monkeypatch):
    monkeypatch.setattr(AgentSpeedUpV2VideoRecorder,"writer_class",MemoryWriter)
    def unavailable():
        raise RuntimeError('unavailable diagnostic')
    video=AgentSpeedUpV2UserCheckVideoRecorder(tmp_path/'planning.mp4',planning_provider=unavailable)
    video.observe(observation());video.decision(observation(),decision())
    assert not video.planning_debug['active']
    assert video.planning_debug['source']=='provider_unavailable:RuntimeError'
    video.close(status='user_stop')


def test_malformed_provider_snapshot_is_diagnostic_only(tmp_path,monkeypatch):
    monkeypatch.setattr(AgentSpeedUpV2VideoRecorder,"writer_class",MemoryWriter)
    raw=fragment();raw['current_segment_index']=np.array([1,2])
    video=AgentSpeedUpV2UserCheckVideoRecorder(tmp_path/'planning.mp4',planning_provider=lambda:raw)
    video.observe(observation());video.decision(observation(),decision())
    assert not video.planning_debug['active']
    video.executed(observation(),step=0,success=False,feedback=feedback(raw))
    assert not video.planning_debug['active']
    video.close(status='user_stop')


def test_live_renderer_uses_xy_xz_zy_in_that_order(tmp_path, monkeypatch):
    import gpt_field.agent_speedup_v2_user_check_video as module
    monkeypatch.setattr(AgentSpeedUpV2VideoRecorder, "writer_class", MemoryWriter)
    seen=[]
    original=module.draw_planning_panel
    def capture(draw, box, axes, **kwargs):
        seen.append((box, axes))
        return original(draw, box, axes, **kwargs)
    monkeypatch.setattr(module, "draw_planning_panel", capture)
    video=module.AgentSpeedUpV2UserCheckVideoRecorder(tmp_path/'three_planes.mp4')
    video.observe(observation())
    assert [axes for _,axes in seen]==[(0,1),(0,2),(2,1)]
    assert seen[0][0][2] < seen[1][0][0] < seen[1][0][2] < seen[2][0][0]
    assert video.planning_debug['planes']==['XY','XZ','ZY']
    video.close(status='preview_only')
