"""Explicit bounded Field egress from initially overlapping uncertainty only."""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from .action import _rotvec_quaternion
from .planner import CandidateRecord,PlanResult
from .visual_field_v10 import MetricFieldExecutorV10
from .visual_field_v22 import MetricFieldExecutorV22
from .visual_monitor_v4 import _orientation_distance
from .uncertainty_egress_v23 import UncertaintyEgressRoute


class _DirectedEgressValidity:
    """Route-only validity; the original grid masks are never modified."""
    def __init__(self,owner):self.owner=owner
    def segment_valid(self,start,end):
        owner=self.owner
        start,end=np.asarray(start,float),np.asarray(end,float)
        if (np.any(start<owner.field.lower) or np.any(start>owner.field.upper)
                or np.any(end<owner.field.lower) or np.any(end>owner.field.upper)
                or owner._outside_corridor(start) or owner._outside_corridor(end)):
            return False
        return owner.egress_geometry.check(start,end)['accepted']
    def query(self,point):
        if not self.segment_valid(point,point):return None
        return float(np.linalg.norm(np.asarray(point)-self.owner.goal))


class MetricFieldExecutorV23(MetricFieldExecutorV22):
    def __init__(self):
        super().__init__()
        self.egress_context=None
        self.egress_geometry=None
        self.uncertainty_egress=None
        self._egress_world_signature=None
        self._committed_egress_geometry=None
        self._committed_egress_world_signature=None

    def _world_geometry_signature(self):
        # Measured orientation changes alter the robot configuration boxes, not
        # the model's world hypotheses. Keep those concepts separate so a small
        # SO(3) change cannot rebase the original episode's halo membership.
        rows=[{'id':r['id'],'kind':r['kind'],'nominal':r.get('nominal'),
               'margins_m':r.get('margins_m')} for r in self.region_geometry]
        def canonical(value):
            if isinstance(value,dict):return {k:canonical(v) for k,v in value.items()}
            if isinstance(value,list):return [canonical(v) for v in value]
            if isinstance(value,float):return round(value,12)
            return value
        return hashlib.sha256(json.dumps(canonical(sorted(rows,key=lambda r:(r['id'],r['kind']))),sort_keys=True).encode()).hexdigest()

    def configure(self,*args,**kwargs):
        prior=self.egress_geometry
        prior_signature=self._egress_world_signature
        self.egress_context=None
        self.uncertainty_egress=None
        result=super().configure(*args,**kwargs)
        self._egress_world_signature=self._world_geometry_signature()
        target=self.target_configuration or {}
        if self.region_geometry and target.get('target_id'):
            fresh=UncertaintyEgressRoute(self.origin_state(),self.region_geometry,
                target_id=target['target_id'],target_band_m=target.get('sigma_m',.03))
            if self._committed_egress_geometry is not None:
                self.egress_geometry=self._committed_egress_geometry
            else:
                self.egress_geometry=prior if prior is not None and prior_signature==self._egress_world_signature else fresh
        else:self.egress_geometry=None
        return self.summary()

    def origin_state(self):
        state=self.expected_state.copy();state[:3]=self.origin
        return state

    def _egress_requested(self):
        c=self.egress_context or {}
        return bool(c.get('phase')=='recover' and c.get('purpose')=='reference_repair'
                    and c.get('translation_mode')=='separate')

    def egress_candidates(self,state8):
        geometry=self.egress_geometry
        if geometry is None or geometry.invalid_reason or self.pending is not None:return []
        if (self._committed_egress_world_signature is not None
                and self._egress_world_signature!=self._committed_egress_world_signature):return []
        state=np.asarray(state8,float)
        # A rejected Field command is replaced and acknowledged by the policy's
        # pose hold, not by this stopped local executor. Its expected_state may
        # therefore predate real residual hold motion. Proposals use the current
        # acknowledged perception boundary; actual select remains strictly bound
        # to the prospective configure and its exact pending acknowledgment.
        guard=self.translation_guard
        if guard is None or not all(self._same(state,getattr(source,'state',None),(8,))
                for source in (guard,self.visual_monitor,getattr(guard,'egress_inventory',None))):return []
        result=[]
        for item in geometry.candidates():
            if not item['full_exit']:continue
            endpoint=geometry.state[:3]+item['goal_offset_m']
            offset=endpoint-state[:3]
            if not .003<np.linalg.norm(offset)<=.04+1e-10:continue
            if geometry.check(state[:3],endpoint,require_progress=True)['accepted']:
                result.append({**item,'goal_offset_m':offset.tolist(),
                    'maximum_contextual_measured_travel_m':.04,'initial_measured_probe_m':.004,
                    'continuation_requires_fresh_positive_image_progress':True})
        return result

    def egress_contract(self,state8,delta_xyz):
        state,delta=np.asarray(state8,float),np.asarray(delta_xyz,float)
        base={'accepted':False,'reason':'egress_not_explicitly_selected',
              'clearance_certified':False,'full_uncertainty_sweep_disjoint':False}
        if not self._egress_requested() or self.egress_geometry is None:return base
        geometry=self.egress_geometry
        proof=geometry.check(state[:3],state[:3]+delta)
        proof['candidate_endpoint_full_halo_disjoint']=proof.get('exit_goal_full_halo_disjoint',False)
        goal_check=geometry.check(state[:3],self.goal)
        proof['exit_goal_full_halo_disjoint']=goal_check.get('exit_goal_full_halo_disjoint',False)
        proof['complete_goal_geometry_accepted']=goal_check['accepted']
        goal_clearance=min(float(np.linalg.norm(np.maximum(np.maximum(lo-self.goal,self.goal-hi),0)))
            for _,_,lo,hi in geometry.parts+geometry.unknown)
        proof['goal_full_halo_clearance_m']=goal_clearance
        proof['goal_arrival_tolerance_m']=self.intent['goal_tolerance_m']
        proof.update(state8=state.tolist(),observation_epoch=getattr(self.translation_guard,'sweep_observation_epoch',None),
            complete_route_goal_xyz=self.goal.tolist(),phase='recover',purpose='reference_repair',translation_mode='separate',
            target_contact=(self.target_configuration or {}).get('effective_allowed_contact'),
            full_finger_range=True,orientation_margin_rad=.04,
            geometry_origin_state8=geometry.state.tolist(),outside_target_contact_band=proof.get('target_core_distance_lower_bound_m',0)>proof.get('target_band_m',.03))
        carried=self.contact_geometry.carried_summary(state) if self.contact_geometry else {}
        proof['empty_tool']=not carried.get('active') and not carried.get('candidate_ids')
        objects=(self._transit_materialization or {}).get('objects') or []
        represented={r['id']for r in self.region_geometry}
        geometry_complete=bool(objects and all(r.get('world_geometry') is not None and r.get('id') in represented for r in objects))
        proof['scene_ids']=[r['id']for r in objects]
        reason=None
        if (self._committed_egress_world_signature is not None
                and self._egress_world_signature!=self._committed_egress_world_signature):reason='egress_committed_world_geometry_changed'
        elif (not self._same(state,self.expected_state,(8,))
                or not self._same(state,getattr(self.translation_guard,'state',None),(8,))):reason='egress_stale_current_robot_state'
        elif _orientation_distance(state[3:6],geometry.state[3:6])>.04:reason='egress_orientation_domain_changed'
        elif not geometry_complete:reason='egress_incomplete_scene_geometry'
        elif not proof['empty_tool'] or proof['target_contact']!='none':reason='egress_contact_or_payload_dependency'
        elif self.contact_refinement_enabled or self.goal_revisions:reason='egress_visual_goal_revision_dependency'
        elif np.linalg.norm(self.goal-geometry.state[:3])>.04+1e-10:reason='egress_goal_outside_context_extent'
        elif not proof['exit_goal_full_halo_disjoint'] or goal_clearance<=self.intent['goal_tolerance_m']+1e-10:
            reason='egress_fixed_goal_does_not_complete_exit_with_arrival_margin'
        if reason:proof.update(accepted=False,reason=reason)
        elif not goal_check['accepted']:proof.update(accepted=False,reason=goal_check['reason'])
        return proof

    def nominal_regions_for_segment(self,start,end):
        records=super().nominal_regions_for_segment(start,end)
        guard=self.translation_guard
        if guard is not None and guard.state is not None:
            proof=self.egress_contract(guard.state,np.asarray(end)-np.asarray(start))
            if not self._same(start,guard.state[:3],(3,)):
                proof.update(accepted=False,reason='egress_proposal_start_mismatch')
            for row in records:row['uncertainty_egress_contract']=copy.deepcopy(proof)
        return records

    def _prepare_egress(self,state):
        proof=self.egress_contract(state,self.goal-state[:3])
        if not proof['accepted']:return proof['reason']
        preliminary=self.egress_geometry.check(state[:3],self.goal)
        check=self.egress_geometry.check(state[:3],self.goal,require_progress=bool(preliminary.get('part_progress')))
        if not check['accepted']:return check['reason']
        if self.uncertainty_egress is None:
            positions=(tuple(self.origin),tuple(self.goal));length=float(np.linalg.norm(self.goal-self.origin))
            route_id=hashlib.sha256(json.dumps({'geometry':self.egress_geometry.geometry_sha256,
                'positions':positions},sort_keys=True).encode()).hexdigest()
            self.uncertainty_egress={'route_id':route_id,'origin_xyz_m':self.origin.tolist(),
                'goal_xyz_m':self.goal.tolist(),'length_m':length,'geometry_sha256':self.egress_geometry.geometry_sha256,
                'initial_halo_parts':proof['initial_halo_parts'],'partial_egress':not proof.get('exit_goal_full_halo_disjoint'),
                'conditional_initial_halo_exception':True,'clearance_certified':False,'grid_masks_modified':False}
            candidate=CandidateRecord(('checked_uncertainty_egress',),positions,length,True,0.,0.)
            self.last_plan=PlanResult(tuple(self.goal-self.origin),'move',1,1,(candidate,),0,length,length,
                search_metadata={'method':'directed_uncertainty_egress_distance','distance_semantics':'remaining_arc_length_on_checked_fixed_route',
                    'cost_unit':'m','cache_hit':False,'route_id':route_id,'full_uncertainty_disjointness_claimed':False})
            self.tracker.reset(positions);self.plan_calls+=1
        self.checked=_DirectedEgressValidity(self)
        return None

    def select(self,state8,joint_position,*,command_cap_m=None,measured_cap_m=None):
        if not self._egress_requested():return super().select(state8,joint_position,command_cap_m=command_cap_m,measured_cap_m=measured_cap_m)
        if self.pending is not None:raise RuntimeError('observe must acknowledge previous Field action')
        if self.stop_reason is not None:return self._result(self.stop_reason)
        state=np.asarray(state8,float)
        if (not self._same(state,self.expected_state,(8,))
                or not self._same(state,getattr(self.translation_guard,'state',None),(8,))):
            return self._stop('egress_stale_current_robot_state')
        # Leave real goal-arrival handling intact rather than requiring further
        # halo progress after a model's bounded partial endpoint is reached.
        if np.linalg.norm(state[:3]-self.goal)<=self.intent['goal_tolerance_m']:
            return self._stop('measured_local_goal_arrival')
        reason=self._prepare_egress(state)
        if reason:return self._stop(reason,{'uncertainty_egress':copy.deepcopy(self.uncertainty_egress)})
        cap=min(.003,self.intent['command_limit_m'])
        if command_cap_m is not None:cap=min(cap,command_cap_m)
        count,rotation=self.orientation_goal_refresh_commands,self.last_orientation_command_rad.copy()
        admission=None
        for _ in range(3):
            self.pending=None;self.orientation_goal_refresh_commands=count;self.last_orientation_command_rad=rotation.copy()
            selected=MetricFieldExecutorV10.select(self,state,joint_position,command_cap_m=cap)
            if not selected['accepted']:return selected
            delta=np.asarray(selected['action'][:3])*.05;size=float(np.linalg.norm(delta))
            if measured_cap_m is not None:
                if self.robot_response is None:
                    allowed=min(size,measured_cap_m)
                else:allowed=self.robot_response.command_cap_for_measured_limit(delta/size,measured_cap_m,size,state,joint_position=joint_position)['command_cap_m']
                if allowed<size-1e-10:
                    cap=min(cap,allowed)
                    if cap<self.MIN_USEFUL_COMMAND_M:admission={'accepted':False,'reason':'egress_measured_allowance_below_useful_command'};break
                    continue
            proof=self.egress_contract(state,delta)
            admission=self.translation_guard.assess(delta,monitor=self.visual_monitor,
                nominal_clear=proof['nominal_core_clear'] and proof['unknown_clear'],
                nominal_away=self._nominal_away(state[:3],delta),
                nominal_regions=self.nominal_regions_for_segment(state[:3],state[:3]+delta),
                egress_contract=proof,translation_response=self.robot_response,
                rotation_rad=float(np.linalg.norm(selected['action'][3:6])*.5),orientation_hold=True)
            if not admission['accepted']:break
            allowed=admission.get('cap_m')
            if allowed is not None and allowed<size-1e-10:
                cap=min(cap,allowed)
                if cap<self.MIN_USEFUL_COMMAND_M:admission={**admission,'accepted':False,'reason':'egress_observed_allowance_below_useful_command'};break
                continue
            scene=self.scene_assessor(state,delta) if self.scene_assessor else {'accepted':False,'reason':'egress_scene_assessor_unavailable'}
            self.last_scene_admission=copy.deepcopy(scene)
            if not scene['accepted']:admission=scene;break
            self.last_translation_admission=copy.deepcopy(admission)
            selected['diagnostics'].update(uncertainty_egress=copy.deepcopy(self.uncertainty_egress),
                egress_contract=proof,translation_admission=copy.deepcopy(admission),scene_coverage_admission=copy.deepcopy(scene))
            return selected
        self.pending=None;self.orientation_goal_refresh_commands=count;self.last_orientation_command_rad=rotation
        admission=admission or {'accepted':False,'reason':'egress_cap_reselection_did_not_converge'}
        self.last_translation_admission=copy.deepcopy(admission)
        return self._stop(admission['reason'],{'translation_admission':copy.deepcopy(admission)})

    def observe(self,before_state8,action,after_state8):
        accepted_egress=self.pending is not None and self._egress_requested()
        geometry,signature=self.egress_geometry,self._egress_world_signature
        result=super().observe(before_state8,action,after_state8)
        # super validates the exact pending state/action before any commitment;
        # preview, downstream rejection holds and wrong acknowledgments cannot
        # create or move this physical episode's original reference geometry.
        if accepted_egress and self._committed_egress_geometry is None:
            self._committed_egress_geometry=geometry
            self._committed_egress_world_signature=signature
        return self.summary()

    def preview_egress(self,state8,joint_position,goal_offset_m,*,goal_tolerance_m=.003,command_limit_m=.003,
                       max_travel_m=.04,command_cap_m=None,measured_cap_m=None,scene_assessor=None):
        guard=self.translation_guard
        if self.intent is None or guard is None or self.pending is not None:
            return {'accepted':False,'status':'egress_preview_unavailable','action':[0.]*7,'diagnostics':{}}
        original,guard_original=self.__dict__,guard.__dict__
        working,guard_working=original.copy(),guard_original.copy()
        guard_working['_command_exhausted_experiments']=set(guard._command_exhausted_experiments)
        guard_working['context']=dict(guard.context or {})
        guard_working['context'].update(translation_mode='separate',purpose='reference_repair',intended_contact_id=None)
        state=np.asarray(state8,float);intent=copy.deepcopy(self.intent)
        shift=self.origin-state[:3]
        for region in intent['unknown_regions']:
            for key in ('lower_offset_m','upper_offset_m'):region[key]=(np.asarray(region[key])+shift).tolist()
        intent.update(goal_offset_m=list(goal_offset_m),goal_tolerance_m=goal_tolerance_m,
            command_limit_m=command_limit_m,max_travel_m=min(max_travel_m,.04),motion_axes='xyz',max_actions=120,
            orientation_target_xyzw=_rotvec_quaternion(state[3:6]).tolist())
        material=copy.deepcopy(self._transit_materialization)
        try:
            self.__dict__,guard.__dict__=working,guard_working
            if scene_assessor is not None:self.scene_assessor=lambda s,d:scene_assessor(self,s,d)
            self.configure(state,intent,scene_assessment=guard.context,inventory_materialization=material)
            self.egress_context={'phase':'recover','purpose':'reference_repair','translation_mode':'separate'}
            selected=copy.deepcopy(self.select(state,joint_position,command_cap_m=command_cap_m,measured_cap_m=measured_cap_m))
        finally:self.__dict__,guard.__dict__=original,guard_original
        selected.update(preview=True,requires_explicit_model_choice=True,physical_debt_replenished=False)
        return selected

    def summary(self):
        result=super().summary();result.update(version='directed-uncertainty-egress-field-v23',
            uncertainty_egress=copy.deepcopy(getattr(self,'uncertainty_egress',None)))
        return result
