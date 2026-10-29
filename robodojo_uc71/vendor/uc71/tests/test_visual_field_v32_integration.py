"""Full V32 policy with separate synthetic RGB proxy/hand observations.

The deterministic image plant supplies RGB and measured robot state only.
No tracker status, response fit, guard admission or task-success flag is injected.
"""
import copy

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, image_to_data_url
from gpt_field.visual_field_policy_v32 import VisualFieldPolicyV32
from gpt_field.visual_monitor_v32 import RoleAwareVisualMonitorV32
from gpt_field.visual_translation_guard_v32 import ProjectedTranslationGuardV32
from test_visual_field_v4_policy import ORIGIN, Q
from test_visual_field_v6_policy import Client
from test_visual_field_v20_policy import prediction as base_prediction
from test_visual_field_v25_policy import explicit
from test_visual_rigid_proxy_v32 import images, references, context, roi, A, H, P


def prediction():
    value = explicit(base_prediction())
    value.update(target_id='target', scene_revisions=[], field_proposal_id=None,
                 reference_candidate_id=None, contact_alignment=None)
    value['visual_references'] = references()
    value['motion_context'] = context()
    value['field_intent'].update(command_limit_m=.002, goal_offset_m=[.065, 0., 0.])
    value['target_geometry']['id'] = 'target'
    value['scene_inventory'] = [{'id':'target', 'external_roi_uv':roi(A).tolist(),
        'wrist_roi_uv':roi(A).tolist(),
        'geometry':copy.deepcopy(value['scene_inventory'][0]['geometry'])}]
    value['raw_wire_prediction'] = {'offline_internal_fixture':True,
        'scene_inventory':copy.deepcopy(value['scene_inventory'])}
    return value


def observation(offset=(0., 0., 0.), **image_options):
    offset = np.asarray(offset, float)
    # Distinct camera depths: a rigid proxy translates 2x as many pixels as
    # the hand. This explicitly defeats copying its affine map to the hand.
    hand = int(np.rint(offset[0]*1000))
    proxy = int(np.rint(offset[0]*2000))
    rendered = images(proxy=proxy, hand=hand, **image_options)
    return PolicyObservation(image_to_data_url(rendered['external'], rotate=False),
        image_to_data_url(rendered['wrist'], rotate=False),
        tuple(ORIGIN+offset)+(0.,0.,0.,.04,-.04),
        'Move the open hand toward the target without contact', Q, (0.,)*7)


def step(policy, before, **image_options):
    decision = policy.act(before)
    offset = np.asarray(before.state[:3])-ORIGIN+np.asarray(decision.action[:3])*.05*.5
    after = observation(offset, **image_options)
    feedback = policy.observe_transition(before, decision, after)
    return after, decision, feedback


def test_proxy_full_policy_reuses_one_field_request_with_separate_hand_projection():
    value = prediction(); client = Client(value); policy = VisualFieldPolicyV32(client)
    obs = observation(); original_goal = ORIGIN+np.array([.065,0.,0.])
    try:
        for _ in range(8):
            obs, decision, feedback = step(policy, obs)
            assert decision.diagnostics['selection']['accepted'], decision.diagnostics['selection']
            assert np.linalg.norm(decision.action[:3]) > 0
            assert decision.action[6] == 0.
            assert not policy.refresh_reason, policy.refresh_reason
            assert policy.translation_guard.reference_role_context['reference_role'] == 'rigid_transit_proxy'
            assert policy.translation_guard.tool_track is policy.monitor.tracks['tool']
            assert policy.translation_guard.tool_track is not policy.monitor.tracks['moving']
            assert policy.pending is policy.local.pending is None
            assert policy.translation_guard._proxy_role_pending is None
            assert feedback['rigid_proxy_use']['active']
            assert not feedback['rigid_proxy_use']['anatomical_goal_or_contact_evidence']
            assert not feedback['rigid_proxy_use']['proxy_transform_transports_hand_outline']
            assert policy.local.goal == pytest.approx(original_goal)
        assert client.calls == 1 and policy.field_effective_actions == 8
        assert type(policy.monitor) is RoleAwareVisualMonitorV32
        assert type(policy.translation_guard) is ProjectedTranslationGuardV32
        moved = (obs.state[0]-ORIGIN[0])*1000
        hand = policy.monitor.tracks['tool']; proxy = policy.monitor.tracks['moving']
        assert moved == pytest.approx(8., abs=1e-8)
        assert hand.position[0]-H[0] == pytest.approx(moved, abs=.3)
        assert proxy.position[0]-P[0] == pytest.approx(2*moved, abs=.3)
        source = policy.translation_guard.tool_source
        corners = source@hand.matrix+hand.translation
        assert policy.translation_guard.tool_box == pytest.approx(
            np.r_[corners.min(0),corners.max(0)])
        assert not np.allclose(hand.matrix, proxy.matrix) or not np.allclose(hand.translation, proxy.translation)
        assert policy.visual_corrections == 0
        assert policy.prediction == value and client.value == value
        assert policy.visual_goal_replay['reason'] == 'proxy_has_no_anatomical_goal'
        assert not feedback['task_success_certified']
    finally:
        policy.finish()


@pytest.mark.parametrize('missing', ['missing_hand', 'missing_proxy'])
def test_current_track_loss_requests_review_without_another_moving_command(missing):
    client = Client(prediction()); policy = VisualFieldPolicyV32(client); obs = observation()
    try:
        for _ in range(3):
            obs, decision, _ = step(policy, obs)
            assert decision.diagnostics['selection']['accepted']
        obs, decision, feedback = step(policy, obs, **{missing:True})
        assert decision.diagnostics['selection']['accepted']  # Admitted against the earlier valid RGB.
        assert policy.refresh_reason
        assert policy.monitor.control_limit()[0] == 0.
        assert client.calls == 1  # Loss feedback itself never calls the model.
        assert policy.pending is policy.local.pending is None
        assert not policy.monitor._external_reliable()
        # The real next policy call receives a new response with the same
        # now-invalid seed; it must not substitute the surviving observer.
        client.value['scene_inventory'][0]['geometry'] = None  # Retain the existing world hypothesis.
        client.value['subgoal_update']['action'] = 'retain'
        rejected = policy.act(obs)
        assert not rejected.diagnostics['selection']['accepted']
        assert np.linalg.norm(rejected.action[:3]) <= 1e-9
        assert policy.translation_guard._proxy_role_pending is None
        policy.observe_transition(obs, rejected, obs)
        assert client.calls == 2 and policy.pending is policy.local.pending is None
    finally:
        policy.finish()
