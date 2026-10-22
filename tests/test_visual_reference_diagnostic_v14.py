"""Passive grounding evidence from real RGB support and acknowledged robot state."""
import copy
from dataclasses import replace

import numpy as np
import pytest

from gpt_field.observation import decode_image, image_to_data_url
from gpt_field.visual_field import _gray
from gpt_field.visual_reference_diagnostic_v14 import MovingReferenceDiagnosticV14
from test_visual_contact_alignment_v13 import fixture
from test_visual_field_v13_execution import observation


def setup():
    _,monitor,_,_=fixture()
    monitor.refs['external']['moving_kind']='tool'
    monitor._active_reference={'unvalidated_travel_m':.019,'blocked_reason':None}
    obs=observation();diagnostic=MovingReferenceDiagnosticV14()
    diagnostic.configure(obs,monitor)
    return diagnostic,monitor,obs


def observe(diagnostic,monitor,before,after):
    assert monitor.tracks['moving'].track(_gray(decode_image(after.image)))
    return diagnostic.observe(before,after,monitor)


def test_static_raw_support_reports_inconclusive_review_at_four_mm_not_wrong_identity():
    diagnostic,monitor,obs=setup();original_debt=copy.deepcopy(monitor._active_reference)
    for i in range(4):
        after=replace(obs,state=(obs.state[0]+.001,*obs.state[1:]))
        report=observe(diagnostic,monitor,obs,after);obs=after
        assert report['review_recommended']==(i==3)
    assert report['status']=='reference_support_stasis'
    assert report['measured_travel_m']==pytest.approx(.004)
    assert report['raw_rgb_change']['maximum_absolute']==0.
    assert report['interpretation']=='inconclusive'
    assert 'weak projection' in report['recommended_action']
    assert not report['execution_veto'] and not report['automatic_reference_switch']
    assert not report['semantic_correspondence_certified']
    assert not report['attachment_certified'] and not report['clearance_certified']
    assert not report['control_debt_modified'] and monitor._active_reference==original_debt


def test_real_rgb_motion_is_raw_change_not_a_semantic_or_attachment_certificate():
    diagnostic,monitor,obs=setup()
    for i in range(1,5):
        after=observation((i*.002,0.,0.));report=observe(diagnostic,monitor,obs,after);obs=after
    assert report['status']=='raw_support_changed' and not report['review_recommended']
    assert report['raw_rgb_change']['maximum_changed_pixel_fraction']>.05
    assert not report['semantic_correspondence_certified'] and not report['attachment_certified']


def test_retain_and_whole_envelope_revision_do_not_restart_diagnostic_travel():
    diagnostic,monitor,obs=setup()
    for i in range(3):
        after=replace(obs,state=(obs.state[0]+.001,*obs.state[1:]))
        observe(diagnostic,monitor,obs,after);obs=after
        diagnostic.configure(obs,monitor)
    assert diagnostic.summary()['measured_travel_m']==pytest.approx(.003)
    after=replace(obs,state=(obs.state[0]+.001,*obs.state[1:]))
    report=observe(diagnostic,monitor,obs,after)
    assert report['review_recommended'] and report['transitions']==4
    diagnostic.configure(after,monitor)
    assert diagnostic.summary()['review_recommended']


def test_no_robot_motion_does_not_turn_many_static_frames_into_a_repair_demand():
    diagnostic,monitor,obs=setup()
    for _ in range(20):report=observe(diagnostic,monitor,obs,obs)
    assert report['transitions']==20 and report['measured_travel_m']==0.
    assert report['status']=='accumulating_motion_evidence'
    assert not report['review_recommended']


def test_small_pixel_noise_is_inconclusive_stasis_not_evidence_of_tool_motion():
    diagnostic,monitor,obs=setup();initial=np.array(decode_image(obs.image))
    for i in range(4):
        noise=np.random.default_rng(i).integers(-1,2,initial.shape)
        rgb=np.clip(initial.astype(int)+noise,0,255).astype(np.uint8)
        after=replace(obs,image=image_to_data_url(rgb,rotate=False),
            state=(obs.state[0]+.001,*obs.state[1:]))
        report=observe(diagnostic,monitor,obs,after);obs=after
    assert report['review_recommended']
    assert report['raw_rgb_change']['maximum_mean_absolute']<1.
    assert report['raw_rgb_change']['maximum_changed_pixel_fraction']==0.


def test_changes_elsewhere_in_rgb_cannot_claim_that_the_selected_support_moved():
    diagnostic,monitor,obs=setup();rgb=np.array(decode_image(obs.image))
    rgb[170:210,20:80]=255-rgb[170:210,20:80]
    after=replace(obs,image=image_to_data_url(rgb,rotate=False),state=(obs.state[0]+.004,*obs.state[1:]))
    report=observe(diagnostic,monitor,obs,after)
    assert report['review_recommended'] and report['raw_rgb_change']['maximum_absolute']==0.


def test_return_to_seed_pixels_does_not_erase_previously_observed_support_change():
    diagnostic,monitor,obs=setup();moved=observation((.004,0.,0.))
    observe(diagnostic,monitor,obs,moved)
    report=observe(diagnostic,monitor,moved,obs)
    assert report['raw_rgb_change']['latest_maximum_absolute']==0.
    assert report['raw_rgb_change']['maximum_absolute']>6.
    assert not report['review_recommended'] and report['status']=='raw_support_changed'


def test_lost_or_replaced_correspondence_requires_configuration_not_silent_reacquisition():
    diagnostic,monitor,obs=setup()
    monitor.tracks['moving'].reliable=False
    report=diagnostic.observe(obs,obs,monitor)
    assert report['status']=='moving_tool_correspondence_unavailable'
    assert not report['review_recommended']
    monitor.tracks['moving']=copy.deepcopy(monitor.tracks['target'])
    report=diagnostic.observe(obs,obs,monitor)
    assert report['status']=='reference_correspondence_changed_requires_configuration'
    assert report['transitions']==0


def test_payload_tracking_and_attachment_state_are_not_reinterpreted_as_tool_stasis():
    diagnostic,monitor,obs=setup();monitor.refs['external']['moving_kind']='payload'
    monitor.attachment_evidence=True
    original=copy.deepcopy(monitor._active_reference)
    report=diagnostic.observe(obs,replace(obs,state=(obs.state[0]+.006,*obs.state[1:])),monitor)
    assert report['status']=='not_a_moving_tool_reference' and not report['review_recommended']
    assert monitor.attachment_evidence and monitor._active_reference==original


def test_out_of_order_feedback_never_double_counts_unobserved_travel():
    diagnostic,monitor,obs=setup()
    after=replace(obs,state=(obs.state[0]+.003,*obs.state[1:]))
    observe(diagnostic,monitor,obs,after)
    report=diagnostic.observe(obs,after,monitor)
    assert report['status']=='reference_diagnostic_transition_gap'
    assert report['measured_travel_m']==pytest.approx(.003) and report['transitions']==1
    assert not report['review_recommended']
