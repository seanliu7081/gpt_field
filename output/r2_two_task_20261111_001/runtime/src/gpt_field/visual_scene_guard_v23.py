"""Narrow scene admission for a current monotone initial-halo exit proof.

Full canonical geometry and the original intersection diagnostics survive.
No intersecting halo is relabeled disjoint and unknown regions remain hard.
"""
from __future__ import annotations

import copy
from .visual_scene_guard_v17 import SceneCoverageGuardV17
from .visual_egress_evidence_v23 import valid_egress_contract


class SceneCoverageGuardV23(SceneCoverageGuardV17):
    version='bounded-initial-halo-egress-scene-v23'

    @staticmethod
    def _part(name):
        return 'declared_tool_or_payload_extent' if name=='model_declared_tool_or_payload_extent' else name

    def assess(self,state8,delta_xyz,*,materialization,field=None,monitor=None,prediction,
               contact_alignment=None,requires_scene_coverage=False,action7=None,rotation_margin_rad=None):
        base=super().assess(state8,delta_xyz,materialization=materialization,field=field,monitor=monitor,
            prediction=prediction,contact_alignment=contact_alignment,
            requires_scene_coverage=requires_scene_coverage,action7=action7,rotation_margin_rad=rotation_margin_rad)
        method=getattr(field,'egress_contract',None)
        if method is None or prediction.get('controller')!='visual_field':
            return base
        proof=method(state8,delta_xyz)
        if proof is None or not proof.get('accepted'):
            return base
        guard=getattr(field,'translation_guard',None)
        if (not valid_egress_contract(proof,state8,delta_xyz,getattr(guard,'sweep_observation_epoch',None))
                or prediction.get('phase')!='recover' or prediction.get('target_contact')!='none'
                or proof['target_id']!=prediction.get('target_id')):
            return base
        details=base['diagnostics']
        # Every canonical object remains checked; no absent/missing geometry
        # issue can be hidden behind the one deliberately intersected halo.
        ids={r['id'] for r in materialization.get('objects',[])}
        if (set(proof['scene_ids'])!=ids or set(details.get('checked_metric_ids',[]))!=ids
                or details.get('unresolved_far_image_ids')
                or details.get('carried_payload_hypothesis_in_sweep') is not False):
            return base
        allowed={(r['id'],self._part(r['tool_part'])) for r in proof.get('initial_halo_parts',[])}
        issues=details.get('coverage_issues',[])
        exempt=[]
        for issue in issues:
            if (issue.get('reason')!='scene_incomplete_image_support_intersects_candidate_uncertainty_sweep'
                    or issue.get('id') not in proof['initial_halo_ids']
                    or not issue.get('intersected_tool_parts')
                    or any((issue['id'],self._part(p)) not in allowed for p in issue['intersected_tool_parts'])):
                return base
            exempt.append(copy.deepcopy(issue))
        if not base['accepted'] and (not exempt or base['reason']!=issues[0]['reason']):
            return base
        result=copy.deepcopy(base)
        result.update(accepted=True,need_refresh=False,reason='scene_conditional_initial_halo_egress')
        result['diagnostics'].update(conditional_egress=True,
            conditional_initial_halo_issues=exempt,coverage_issues=[],
            uncertainty_egress_contract=copy.deepcopy(proof),
            full_uncertainty_sweeps_not_reclassified=True,
            clearance_certified=False)
        return result
