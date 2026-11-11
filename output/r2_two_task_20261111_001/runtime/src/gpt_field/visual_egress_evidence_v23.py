"""Finite, observed two-view support for leaving a model uncertainty halo.

This is conditional risk exposure, not depth, segmentation or clearance. The
initial geometry and all spent robot travel survive reference and goal changes.
Assessment is pure; only acknowledgement commits snapshots or image progress.
"""
from __future__ import annotations

import copy
from collections import deque
import numpy as np

from .visual_monitor_v4 import _orientation_distance
from .visual_translation_guard_v11 import _gap


def valid_egress_contract(proof, state8, delta_xyz, epoch=None):
    """Defensive current-candidate binding; geometry is computed by executor."""
    if not isinstance(proof, dict) or proof.get('accepted') is not True:
        return False
    try:
        state, delta = np.asarray(state8, float), np.asarray(delta_xyz, float)
        ids, scene, parts = proof['initial_halo_ids'], proof['scene_ids'], proof['initial_halo_parts']
        return bool(state.shape == (8,) and delta.shape == (3,)
            and np.isfinite(np.r_[state, delta]).all()
            and np.asarray(proof['state8']).shape == (8,)
            and np.asarray(proof['start_xyz_m']).shape == (3,)
            and np.asarray(proof['end_xyz_m']).shape == (3,)
            and np.allclose(proof['state8'], state, atol=1e-10, rtol=0)
            and np.allclose(proof['start_xyz_m'], state[:3], atol=1e-10, rtol=0)
            and np.allclose(proof['end_xyz_m'], state[:3]+delta, atol=1e-10, rtol=0)
            and (epoch is None or proof['observation_epoch'] == epoch)
            and isinstance(proof['geometry_sha256'], str) and bool(proof['geometry_sha256'])
            and np.asarray(proof['geometry_origin_state8']).shape == (8,)
            and np.isfinite(np.asarray(proof['geometry_origin_state8'],float)).all()
            and ids and len(ids) == len(set(ids)) and set(ids).issubset(scene)
            and isinstance(parts,list) and parts
            and all(isinstance(p,dict) and p.get('id') in ids
                    and isinstance(p.get('tool_part'),str) and p['tool_part'] for p in parts)
            and {p['id'] for p in parts}==set(ids)
            and len(scene) == len(set(scene)) and proof['target_id'] in scene
            and all(proof.get(k) is True for k in ('nominal_core_clear', 'unknown_clear',
                'no_new_halo_entry', 'monotone_egress', 'exit_goal_full_halo_disjoint',
                'empty_tool', 'outside_target_contact_band', 'full_finger_range'))
            and proof.get('target_contact') == 'none' and proof.get('phase') == 'recover'
            and proof.get('translation_mode') == 'separate' and proof.get('purpose') == 'reference_repair'
            and float(proof['orientation_margin_rad']) >= .04)
    except (KeyError, TypeError, ValueError):
        return False


def _track_snapshot(track):
    if track is None or not getattr(track, 'reliable', False):
        return None
    points, active = np.asarray(track.points, float), np.asarray(track.active, int)
    if len(points) < 3 or len(active) != len(points) or not np.isfinite(points).all():
        return None
    return {'identity': id(track), 'point': np.asarray(track.position, float).copy(),
        'points': {int(i): p.copy() for i, p in zip(active, points)},
        'noise': max(.08, float(track.last_residual_px)/np.sqrt(max(3, track.last_inliers)))}


def _box(record):
    value = record.get('projected_roi_uv')
    if value is None:
        return None
    value = np.asarray(value, float)
    return value if value.shape == (4,) and np.isfinite(value).all() else None


def _source_signature(value):
    if value is None:
        return None
    return tuple(np.round(np.asarray(value, float).ravel(), 9))


class EgressEvidenceV23:
    total_limit_m = .040
    bootstrap_m = .004
    tranche_m = .004
    command_limit_m = .003
    minimum_command_m = .001
    maximum_orientation_change_rad = .015
    checkpoint_travel_m = .003

    def __init__(self):
        self.spent_m = 0.
        self.egress_travel_m = 0.
        self.permitted_egress_travel_m = self.bootstrap_m
        self.baseline = None
        self.frames = deque(maxlen=64)
        self.last_progress = None
        self.last_reason = 'uncertainty_egress_not_started'
        self.last_charge_egress = False
        self.grants = 0
        self.observations = 0
        self.last_observed_state = None
        self.wrist_risk_origins = {}
        self.committed_geometry_context = None

    @staticmethod
    def _geometry_context(proof):
        return (proof['geometry_sha256'],tuple(proof['geometry_origin_state8']),
                tuple(sorted((p['id'],p['tool_part']) for p in proof['initial_halo_parts'])))

    def _snapshot(self, state8, guard, monitor, inventory, identities):
        track = guard._tool(monitor)
        tool = _track_snapshot(track)
        state = np.asarray(state8, float)
        if (tool is None or track is not guard.tool_track or guard.tool_box is None
                or inventory is None or getattr(monitor,'state',None) is None
                or not np.allclose(monitor.state,state,atol=1e-9,rtol=0)
                or not np.allclose(inventory.state, state, atol=1e-9, rtol=0)):
            return None
        objects = getattr(inventory, 'objects', {})
        regions, stationary = {}, {}
        for identity, item in objects.items():
            view = item['views']['external']
            current = _track_snapshot(view.get('track'))
            if current is not None and view.get('support_observed_at') == inventory.observation_index:
                stationary[identity] = current
        for identity in identities:
            item = objects.get(identity)
            if item is None:
                return None
            row = {}
            for name in ('external', 'wrist'):
                view = item['views'][name]
                current = _track_snapshot(view.get('track'))
                box = _box(view)
                if (current is None or box is None or view.get('current_visible_roi_uv') is None
                        or view.get('support_observed_at') != inventory.observation_index
                        or view.get('support_inlier_count', 0) < 3):
                    return None
                current['box'] = box.copy()
                current['source'] = _source_signature(view.get('source'))
                row[name] = current
            regions[identity] = row
        return {'state8': state.copy(), 'tool': tool, 'tool_box': np.asarray(guard.tool_box).copy(),
            'tool_source': _source_signature(guard.tool_source), 'scale': np.asarray(guard.scale).copy(),
            'regions': regions, 'stationary': stationary, 'travel_m': self.egress_travel_m,
            'observation': inventory.observation_index}

    @staticmethod
    def _compatible(first, last):
        return bool(first['tool']['identity'] == last['tool']['identity']
            and first['tool_source'] == last['tool_source']
            and first['regions'].keys() == last['regions'].keys()
            and all(first['regions'][i][v]['identity'] == last['regions'][i][v]['identity']
                and first['regions'][i][v]['source'] == last['regions'][i][v]['source']
                for i in first['regions'] for v in ('external', 'wrist')))

    @staticmethod
    def _wrist_scale(before, after):
        common = before['points'].keys() & after['points'].keys()
        if len(common) < 3:
            return None
        ids = sorted(common)
        a, b = np.array([before['points'][i] for i in ids]), np.array([after['points'][i] for i in ids])
        pairs = [(i, j) for i in range(len(ids)) for j in range(i) if np.linalg.norm(a[i]-a[j]) >= 5.]
        if len(pairs) < 3:
            return None
        lengths = np.array([np.linalg.norm(a[i]-a[j]) for i,j in pairs])
        ratios = np.array([np.linalg.norm(b[i]-b[j])/length for (i,j),length in zip(pairs,lengths)])
        log_scale = float(np.median(np.log(np.maximum(ratios, 1e-9))))
        noise = float(np.hypot(before['noise'], after['noise'])/max(5.,np.median(lengths)))
        # Correlated pair distances do not create independent extra samples.
        threshold = float(max(.008, 3*np.sqrt(2.)*noise))
        return {'log_scale_change':log_scale, 'threshold':threshold,
                'favorable':bool(log_scale < -threshold), 'worsening':bool(log_scale > threshold),
                'common_features':len(common), 'is_metric_depth':False}

    def _progress(self, current):
        start = self.baseline
        if start is None or not self._compatible(start,current):
            return {'supported':False,'reason':'egress_correspondence_changed','regions':{}}
        path = self.egress_travel_m-start['travel_m']
        common = start['stationary'].keys() & current['stationary'].keys()
        common = [i for i in common if start['stationary'][i]['identity'] == current['stationary'][i]['identity']]
        if not common:
            return {'supported':False,'reason':'egress_stationary_context_unavailable','regions':{}}
        scene_flow = np.median([current['stationary'][i]['point']-start['stationary'][i]['point'] for i in common],axis=0)
        tool_flow = current['tool']['point']-start['tool']['point']-scene_flow
        threshold = max(.35, 3*np.hypot(start['tool']['noise'],current['tool']['noise']))
        samples = [s for s in self.frames if s['travel_m'] >= start['travel_m']-1e-12]
        # A coherent controlled patch must move relative to the scene across
        # actual acknowledged motion; a static seed/common camera flow cannot
        # turn neighbor box noise into positive robot evidence.
        controlled = bool(path >= .001 and len(samples) >= 3 and np.linalg.norm(tool_flow) > threshold)
        regions, worse = {}, False
        for identity in start['regions']:
            a, b = start['regions'][identity], current['regions'][identity]
            first_box = a['external']['box']*np.tile(start['scale'],2)
            last_box = b['external']['box']*np.tile(current['scale'],2)
            change = _gap(current['tool_box'],last_box)-_gap(start['tool_box'],first_box)
            # Whole-box transport magnifies point uncertainty; do not use a
            # subpixel feature residual as whole-tool boundary precision.
            def edge_noise(s, neighbor):
                tool_points=np.array(list(s['tool']['points'].values()))
                neighbor_points=np.array(list(neighbor['points'].values()))
                tool_lever=max(1.,np.linalg.norm(s['tool_box'][2:]-s['tool_box'][:2])/max(8.,np.linalg.norm(np.ptp(tool_points,axis=0))))
                box=neighbor['box']*np.tile(s['scale'],2)
                near_lever=max(1.,np.linalg.norm(box[2:]-box[:2])/max(8.,np.linalg.norm(np.ptp(neighbor_points,axis=0))))
                return np.hypot(s['tool']['noise']*tool_lever,neighbor['noise']*near_lever)
            gap_threshold=max(.75,3*np.hypot(edge_noise(start,a['external']),edge_noise(current,b['external'])))
            wrist=self._wrist_scale(a['wrist'],b['wrist'])
            risk_origin=self.wrist_risk_origins.get((identity,b['wrist']['identity']))
            cumulative_wrist=None if risk_origin is None else self._wrist_scale(risk_origin,b['wrist'])
            external_worse=change < -gap_threshold
            region_worse=(external_worse or (wrist is not None and wrist['worsening'])
                          or (cumulative_wrist is not None and cumulative_wrist['worsening']))
            favorable=controlled and wrist is not None and not region_worse and (change > gap_threshold or wrist['favorable'])
            regions[identity]={'external_gap_change_px':float(change),'external_gap_threshold_px':float(gap_threshold),
                'wrist':wrist,'wrist_from_first_correspondence':cumulative_wrist,
                'favorable':bool(favorable),'worsening':bool(region_worse)}
            worse |= region_worse
        supported=bool(controlled and regions and all(r['favorable'] for r in regions.values()))
        return {'supported':supported,
            'reason':'egress_two_view_worsening' if worse else 'egress_observation_inconclusive',
            'worsening':bool(worse),'controlled_feature_motion_supported':controlled,
            'controlled_relative_flow_px':tool_flow.tolist(),'controlled_motion_threshold_px':threshold,
            'measured_travel_since_checkpoint_m':path,'regions':regions,
            'two_view_evidence_is_clearance_certificate':False}

    def assess(self, proof, state8, delta_xyz, *, guard, monitor, inventory, translation_response=None):
        state,delta=np.asarray(state8,float),np.asarray(delta_xyz,float)
        if not valid_egress_contract(proof,state,delta,guard.sweep_observation_epoch):
            return {'accepted':False,'cap_m':0.,'reason':'uncertainty_egress_contract_unavailable'}
        if (self.committed_geometry_context is not None
                and self._geometry_context(proof)!=self.committed_geometry_context):
            return {'accepted':False,'cap_m':0.,'reason':'uncertainty_egress_initial_geometry_context_changed'}
        current=self._snapshot(state,guard,monitor,inventory,proof['initial_halo_ids'])
        if current is None:
            return {'accepted':False,'cap_m':0.,'reason':'uncertainty_egress_current_two_view_support_unavailable'}
        if self.baseline is not None:
            if (_orientation_distance(self.baseline['state8'][3:6],state[3:6]) > self.maximum_orientation_change_rad
                    or np.max(abs(self.baseline['state8'][6:]-state[6:])) > .002):
                return {'accepted':False,'cap_m':0.,'reason':'uncertainty_egress_pose_domain_changed'}
        if self.last_progress and self.last_progress.get('worsening'):
            return {'accepted':False,'cap_m':0.,'reason':'uncertainty_egress_observed_worsening'}
        spent=max(self.spent_m,float(guard.observation_travel_m))
        remaining=min(self.total_limit_m-spent,self.permitted_egress_travel_m-self.egress_travel_m)
        if remaining <= 1e-9:
            return {'accepted':False,'cap_m':0.,'reason':'uncertainty_egress_fresh_progress_required'}
        norm=float(np.linalg.norm(delta)); requested=min(norm,self.command_limit_m)
        if norm <= 1e-12:
            return {'accepted':False,'cap_m':0.,'reason':'uncertainty_egress_requires_translation'}
        conversion=None
        if translation_response is not None:
            conversion=translation_response.command_cap_for_measured_limit(delta/norm,remaining,requested,state)
            cap=min(requested,float(conversion['command_cap_m']))
        else:
            cap=min(requested,remaining,.002)  # Explicit bounded unit-gain cold assumption.
        if cap < self.minimum_command_m-1e-10:
            return {'accepted':False,'cap_m':0.,'reason':'uncertainty_egress_no_useful_command_within_remaining_budget',
                    'remaining_measured_m':remaining}
        return {'accepted':True,'cap_m':cap,'reason':'uncertainty_egress_bounded_observation',
            'remaining_measured_m':remaining,'conversion':conversion,
            'pending':{'before':current,'proof':copy.deepcopy(proof),'delta':delta.copy(),
                'command_cap_m':cap,'prior_shared_debt_m':float(guard.observation_travel_m)},
            'clearance_certified':False}

    def observe(self, before_state8, action7, after_state8, *, pending, guard, monitor, inventory):
        before,after,action=np.asarray(before_state8,float),np.asarray(after_state8,float),np.asarray(action7,float)
        delta=action[:3]*.05; size=float(np.linalg.norm(delta))
        if self.last_observed_state is not None and not np.allclose(before,self.last_observed_state,atol=1e-9,rtol=0):
            self.last_reason='uncertainty_egress_acknowledgement_gap';self.last_progress={'worsening':True}
            return self.summary()
        matched=bool(pending is not None and size>1e-8
            and np.allclose(before,pending['before']['state8'],atol=1e-9,rtol=0)
            and size<=pending['command_cap_m']+1e-9
            and np.allclose(delta,pending['delta'],atol=1e-10,rtol=0)
            and abs(action[6])<1e-12)
        travel=float(np.linalg.norm(after[:3]-before[:3]))
        charge=matched or (size<=1e-8 and self.last_charge_egress)
        if charge:
            self.spent_m=max(self.spent_m+(travel if self.last_charge_egress else 0.),float(guard.observation_travel_m))
            self.egress_travel_m+=travel
        self.spent_m=max(self.spent_m,float(guard.observation_travel_m))
        if size>1e-8:
            self.last_charge_egress=matched
        self.last_observed_state=after.copy()
        if not matched:
            return self.summary()
        context=self._geometry_context(pending['proof'])
        if self.committed_geometry_context is None:
            self.committed_geometry_context=context
        elif context!=self.committed_geometry_context:
            self.last_reason='uncertainty_egress_initial_geometry_context_changed'
            self.last_progress={'worsening':True,'reason':self.last_reason}
            return self.summary()
        self.observations+=1
        start=pending['before']
        for identity,region in start['regions'].items():
            key=(identity,region['wrist']['identity'])
            if key not in self.wrist_risk_origins:
                if len(self.wrist_risk_origins)>=24:
                    self.last_reason='uncertainty_egress_support_revision_limit'
                    self.last_progress={'worsening':True,'reason':self.last_reason}
                    return self.summary()
                self.wrist_risk_origins[key]=copy.deepcopy(region['wrist'])
        current=self._snapshot(after,guard,monitor,inventory,pending['proof']['initial_halo_ids'])
        if current is None:
            self.last_reason='uncertainty_egress_current_two_view_support_unavailable'
            self.last_progress={'supported':False,'worsening':False,'reason':self.last_reason}
            return self.summary()
        # Repairs change a measurement domain, never its remaining allowance.
        if self.baseline is None or not self._compatible(self.baseline,start):
            self.baseline=copy.deepcopy(start); self.baseline['travel_m']=self.egress_travel_m-travel
            self.frames.clear();self.frames.append(copy.deepcopy(self.baseline))
        current['travel_m']=self.egress_travel_m;self.frames.append(current)
        self.last_progress=self._progress(current)
        self.last_reason=self.last_progress['reason']
        if (self.last_progress['supported']
                and self.egress_travel_m-self.baseline['travel_m'] >= self.checkpoint_travel_m-1e-9):
            self.permitted_egress_travel_m=max(self.permitted_egress_travel_m,self.egress_travel_m+self.tranche_m)
            self.baseline=copy.deepcopy(current);self.frames.clear();self.frames.append(copy.deepcopy(current))
            self.grants+=1;self.last_reason='uncertainty_egress_fresh_two_view_progress'
        return self.summary()

    def summary(self):
        return {'version':'finite-two-view-halo-egress-v23','total_limit_m':self.total_limit_m,
            'spent_measured_m':self.spent_m,'egress_measured_m':self.egress_travel_m,
            'total_remaining_m':max(0.,self.total_limit_m-self.spent_m),
            'observation_tranche_remaining_m':max(0.,self.permitted_egress_travel_m-self.egress_travel_m),
            'bootstrap_m':self.bootstrap_m,'per_command_cap_m':self.command_limit_m,
            'fresh_progress_grants':self.grants,'acknowledged_egress_actions':self.observations,
            'initial_geometry_context_bound':self.committed_geometry_context is not None,
            'reason':self.last_reason,'progress':copy.deepcopy(self.last_progress),
            'geometry_and_image_evidence_are_hypotheses':True,'clearance_certified':False,
            'reseed_renaming_replenishes_budget':False}
