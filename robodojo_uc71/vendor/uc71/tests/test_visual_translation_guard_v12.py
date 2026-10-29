"""Task-free RGB risk evidence and archive counterfactuals, no live control."""
import copy,json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_translation_guard_v12 import ObservedRiskTranslationGuardV12
from gpt_field.robot_translation_response import MeasuredTranslationResponse
from gpt_field.visual_field import _gray
from test_visual_translation_guard_v11 import context,pixels,plant as old_plant


def plant(ctx=None):
    _,monitor,state=old_plant(ctx)
    guard=ObservedRiskTranslationGuardV12();im=pixels()
    guard.configure({'external':im,'wrist':im},ctx or context(),state,target_id='target',monitor=monitor)
    return guard,monitor,state


def assess(guard,monitor,command,*,distance=.04,sigma=.03,**kwargs):
    return guard.assess(command,monitor=monitor,nominal_clear=True,
        nominal_regions=[{'id':'neighbor','core_distance_m':distance,'uncertainty_m':sigma}],**kwargs)


def advance(guard,monitor,state,x,*,observed_x=None,common=0,rotation=0.,missing=False):
    im=pixels(x if observed_x is None else observed_x,common=common,missing=missing)
    monitor.tracks['moving'].track(_gray(im));new=state.copy();new[0]=x*.001;new[3]=rotation
    guard.observe({'external':im,'wrist':im},new,[.04,0,0,0,0,0,0],monitor=monitor)
    return new


def test_initial_overlap_admits_bounded_metric_clear_observation_not_occupancy():
    g,m,s=plant(context(tool=(43,53,99,77)))
    r=assess(g,m,[.005,0,0])
    assert r['accepted'] and r['cap_m']==.005
    assert r['reason']=='projected_risk_bounded_nominal_clear_observation'
    assert not r['diagnostics']['clearance_certified']
    assert not g.assess([.005,0,0],monitor=m,nominal_clear=False)['accepted']


def test_remote_model_object_overlap_is_telemetry_not_a_veto_or_clearance_claim():
    g,m,s=plant(context(tool=(43,53,99,77)))
    r=assess(g,m,[.012,0,0],distance=.4,sigma=.03)
    assert r['accepted'] and r['cap_m'] is None
    assert g.summary()['nominal_regions'][0]['model_remote']
    assert not r['diagnostics']['clearance_certified']
    assert g.summary()['conflicting_obstacle_ids']==['neighbor']


def test_free_route_retains_full_field_commands_without_small_caps():
    g,m,s=plant()
    for x in range(1,12):
        r=assess(g,m,[.006,0,0])
        assert r['accepted'] and r['cap_m'] is None
        s=advance(g,m,s,x)
    assert g.observation_travel_m==0


def test_resolved_approach_stops_before_real_projected_contact():
    g,m,s=plant(context(tool=(43,53,94,77)))
    reasons=[]
    for x in range(1,9):
        r=assess(g,m,[.001,0,0]);reasons.append(r['reason'])
        if not r['accepted']:break
        s=advance(g,m,s,x)
    assert r['reason']=='translation_observed_neighbor_approach'
    assert x<=6
    assert 'neighbor' in g.worsening
    assert g.observation_travel_m < .020


def test_static_wrong_feature_exhausts_same_measured_budget_even_with_renames_and_holds():
    ctx=context(tool=(43,53,99,77));g,m,s=plant(ctx)
    for i in range(1,22):
        r=assess(g,m,[.002,0,0])
        if not r['accepted']:break
        s=advance(g,m,s,i,observed_x=0)
        if i%3==0:
            renamed=copy.deepcopy(ctx);renamed['nearby_obstacles'][0]['id']='renamed'+str(i)
            im=pixels();g.configure({'external':im,'wrist':im},renamed,s,target_id='target',monitor=m)
    assert r['reason']=='translation_observation_budget_exhausted'
    assert .019<=g.observation_travel_m<=.0200001
    used=g.observation_travel_m
    # A rejected hold can still have residual measured motion: charge it.
    g.assess([0,0,0],monitor=m)
    s=advance(g,m,s,i+.2,observed_x=0)
    assert g.observation_travel_m > used
    assert not g.cumulative_response_ready


def test_current_envelope_can_correct_top_edge_without_reseeding_or_resetting_debt():
    ctx=context(tool=(43,0,99,77));g,m,s=plant(ctx)
    assert assess(g,m,[.002,0,0])['accepted']
    s=advance(g,m,s,.5,observed_x=0)
    used=g.observation_travel_m;track=g.tool_track
    corrected=context(tool=(43,53,99,77));im=pixels()
    g.configure({'external':im,'wrist':im},corrected,s,target_id='target',monitor=m)
    assert g.tool_box[1]==53 and g.tool_track is track
    assert g.observation_travel_m==used
    assert g.summary()['envelope_revision']['accepted']
    assert not g.summary()['envelope_revision']['is_observed_motion']
    assert not g.improving


def test_envelope_cannot_crop_known_descriptor_support_or_remove_neighbor():
    g,m,s=plant(context(tool=(43,53,99,77)));im=pixels()
    bad=context(tool=(54,64,56,66));g.configure({'external':im,'wrist':im},bad,s,target_id='target',monitor=m)
    assert assess(g,m,[.002,0,0])['reason']=='translation_current_envelope_support_inconsistent'
    corrected=context(tool=(43,53,99,77));corrected.update(obstacle_assessment='none_relevant',nearby_obstacles=[])
    g.configure({'external':im,'wrist':im},corrected,s,target_id='target',monitor=m)
    assert g.summary()['conflicting_obstacle_ids']==['neighbor']


@pytest.mark.parametrize('change',['lost','rotation','camera'])
def test_local_correspondence_or_domain_loss_still_hands_off(change):
    g,m,s=plant(context(tool=(43,53,99,77)));assess(g,m,[.002,0,0])
    s=advance(g,m,s,1,missing=change=='lost',rotation=.03 if change=='rotation' else 0,common=3 if change=='camera' else 0)
    assert not assess(g,m,[.002,0,0])['accepted']


def test_unknown_neighborhood_is_still_strict():
    ctx=context();ctx['obstacle_assessment']='unobservable';g,m,s=plant(ctx)
    assert not assess(g,m,[.002,0,0],distance=100)['accepted']


def test_image_response_is_fit_on_aggregated_measured_displacements():
    g,m,s=plant();previous=s.copy()
    response=MeasuredTranslationResponse()
    for i in range(1,61):
        # 5mm OSC commands produce .2mm measured steps; pixels quantized by
        # this testplant. Fit from2mm aggregates, not eachsubpixel transition.
        x=.2*i;im=pixels(int(x));m.tracks['moving'].track(_gray(im))
        new=s.copy();new[0]=.001*x
        response.observe(s,[.1,0,0,0,0,0,0],new)
        g.observe({'external':im,'wrist':im},new,[.1,0,0,0,0,0,0],monitor=m);s=new
    assert g.cumulative_response_ready
    assert g._response_extent_m>=.002
    predicted=response.predict([.005,0,0],s)
    assert predicted['supported'] and predicted['mean_delta_m'][0] < .001
    flow,_=g._image_prediction(np.array([.005,0,0]),response)
    assert flow is not None and 0<flow[0]<1.


def test_remaining_budget_caps_command_using_measured_actuator_response():
    g,m,s=plant(context(tool=(43,53,99,77)))
    class LearnedPlant:
        def predict(self,command,state):return {'supported':True,'responsive':True,'mean_delta_m':(np.asarray(command)*.2).tolist()}
        def command_cap_for_measured_limit(self,direction,limit,requested,state):
            return {'supported':True,'command_cap_m':min(requested,limit/.25)}
    g.observation_travel_m=.0185
    r=assess(g,m,[.005,0,0],translation_response=LearnedPlant())
    assert r['accepted'] and r['cap_m']==pytest.approx(.005)
    # The floor is on COMMAND metres. A .5mm remaining measured allowance
    # can still support a2mm command under the observed .25 gain upper bound.
    g.observation_travel_m=.0195
    r=assess(g,m,[.005,0,0],translation_response=LearnedPlant())
    assert r['accepted'] and r['cap_m']==pytest.approx(.002)
    g.observation_travel_m=.0199
    assert not assess(g,m,[.005,0,0],translation_response=LearnedPlant())['accepted']


def archive_monitor(start_guard_at,run_name,method):
    from gpt_field.observation import decode_image
    from gpt_field.visual_monitor_v10 import RoleAwareVisualMonitorV10
    p=Path(__file__).parents[1]/'runs'/run_name/'trials'/('bowl_original_zero_hybrid_'+method)/'trace.jsonl'
    if not p.exists():pytest.skip('local archive not present')
    mon=RoleAwareVisualMonitorV10();guard=ObservedRiskTranslationGuardV12();obs=None;active=False;records=[]
    def views(o):return {'external':np.asarray(decode_image(o['observation/image'])),'wrist':np.asarray(decode_image(o['observation/wrist_image']))}
    for line in p.open():
        d=json.loads(line);step=d.get('step',-1)
        if d['event']=='reset':obs=d['observation']
        elif d['event']=='action':
            q=d['diagnostics'].get('prediction')
            if q:
                mon.configure(views(obs),q['visual_references'],obs['observation/state'],intent_id=d['diagnostics']['chunk_index'])
                if step>=start_guard_at:
                    ctx=copy.deepcopy(q['motion_context']);ctx.setdefault('translation_mode','transit');ctx.setdefault('intended_contact_id',None)
                    guard.configure(views(obs),ctx,obs['observation/state'],target_id=q['target_id'],monitor=mon);active=True
            if active:
                command=np.asarray(d['action'][:3])*.05
                if np.linalg.norm(command)<1e-12:command=np.array([0,.002,0])
                # Read-only alternative admission under the original selected
                # nominal-clear geometry; no counterfactual actions are executed.
                r=guard.assess(command,monitor=mon,nominal_clear=True)
                records.append((step,r,copy.deepcopy(guard.summary())))
        elif d['event']=='executed':
            obs=d['observation'];im=views(obs);mon.observe(im,obs['observation/state'],d['action'])
            if active:guard.observe(im,obs['observation/state'],d['action'],monitor=mon)
    return records


def test_v11_actual_first4mm_need_not_exhaust_v12_observation_permission():
    records=archive_monitor(0,'visual_field_v11_zero_shot_001','v11')
    row=next(r for r in records if r[0]==13)
    assert row[1]['accepted'],row[1]['reason']
    assert row[2]['observation_travel_remaining_m']>.015
    corrected=next(r for r in records if r[0]==16)
    assert corrected[2]['envelope_revision']['accepted']
    assert corrected[2]['tool_box_px'][1]>40
    assert corrected[2]['observation_travel_m']>0


def test_v10_late_approach_still_yields_observed_worsening_before_contact():
    records=archive_monitor(89,'visual_field_v10_zero_shot_001','v10')
    worsened=[r for r in records if r[1]['reason']=='translation_observed_neighbor_approach']
    assert worsened and min(r[0] for r in worsened)<138
    assert any('ramekin' in r[2]['risk_progress']['worsening_ids'] for r in worsened)


def test_field_orientation_hold_is_explicit_and_observed_rotation_still_invalidates():
    g,m,s=plant()
    assert not assess(g,m,[.005,0,0],rotation_rad=.02)['accepted']
    assert assess(g,m,[.005,0,0],rotation_rad=.02,orientation_hold=True)['accepted']
    assert not assess(g,m,[.005,0,0],rotation_rad=.021,orientation_hold=True)['accepted']
    s=advance(g,m,s,1,rotation=.03)
    assert not assess(g,m,[.005,0,0],rotation_rad=0,orientation_hold=True)['accepted']


def test_slow_actuator_aggregates_receding_evidence_and_sustains_field_beyond_budget():
    g,m,s=plant(context(tool=(43,53,125,77)))
    response=MeasuredTranslationResponse();supported_steps=0
    for i in range(1,151):
        r=assess(g,m,[-.005,0,0],translation_response=response)
        assert r['accepted'],(i,r['reason'],g.summary()['risk_progress'])
        supported_steps += r['reason']=='projected_risk_observed_separation_supported'
        x=-.2*i;im=pixels(int(x));m.tracks['moving'].track(_gray(im))
        new=s.copy();new[0]=.001*x
        response.observe(s,[-.1,0,0,0,0,0,0],new)
        g.observe({'external':im,'wrist':im},new,[-.1,0,0,0,0,0,0],monitor=m);s=new
    assert supported_steps>0
    assert abs(s[0])>.020 and g.observation_travel_m<.020
    assert g.cumulative_response_ready



def test_current_direct_observation_permits_small_nonzero_free_correction_only():
    g,m,s=plant()
    r=g.assess([.002,0,0],monitor=m,nominal_clear=False,direct_observation=True)
    assert r['accepted'] and r['cap_m']==.002
    assert r['reason']=='projected_direct_current_observation_bounded'
    assert not r['diagnostics']['nominal_segment_clear_reported']
    r=g.assess([.02,0,0],monitor=m,nominal_clear=False,direct_observation=True)
    assert r['accepted'] and r['cap_m']==.002  # Caller rejects; never clips silently.
    g,m,s=plant(context(tool=(43,53,99,77)))
    r=g.assess([.002,0,0],monitor=m,nominal_clear=False,direct_observation=True)
    assert not r['accepted']
    assert r['reason']=='translation_metric_segment_clearance_unavailable'


def test_direct_exact_intended_contact_does_not_exempt_other_neighbor():
    ctx=context(tool=(43,53,99,77));ctx['translation_mode']='interaction';ctx['intended_contact_id']='target'
    ctx['nearby_obstacles'][0]['id']='target'
    g,m,s=plant(ctx)
    assert g.assess([.002,0,0],monitor=m,direct_observation=True)['accepted']
    # Retained aliases cannot be relabeled as intended contact to evade a known
    # independent obstacle, even if the declaration uses the same current ROI.
    ctx['nearby_obstacles'][0]['id']='other';im=pixels()
    g.configure({'external':im,'wrist':im},ctx,s,target_id='target',monitor=m)
    assert not g.assess([.002,0,0],monitor=m,direct_observation=True)['accepted']
