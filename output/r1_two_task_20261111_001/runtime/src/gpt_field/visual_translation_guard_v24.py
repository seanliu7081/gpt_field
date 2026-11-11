"""V23 hard admission plus immutable raw-reference route exposure."""
from __future__ import annotations
import copy
import hashlib
import numpy as np
from .visual_translation_guard_v23 import ProjectedTranslationGuardV23
from .visual_route_exposure_v24 import RouteExposureV24
from .visual_reference_replay_v18 import evidence_digest


class ProjectedTranslationGuardV24(ProjectedTranslationGuardV23):
    version='causal-reference-fixed-route-exposure-v24'
    def __init__(self):
        super().__init__()
        self.egress_evidence=RouteExposureV24()
        self.reference_witness=None
        self.reference_selection=None

    def bind_reference_witness(self,bank):
        self.reference_witness=bank

    def assess(self,delta_xyz,**kwargs):
        result=super().assess(delta_xyz,**kwargs)
        if result.get('accepted') and self._egress_pending is not None:
            result['reason']=self._egress_pending['exposure_mode']
            result['diagnostics']['route_exposure_is_clearance_proof']=False
        return result

    def bind_reference_selection(self,resolved_candidate,state8):
        """Bind an explicitly resolved current candidate, not a new allowance."""
        self.reference_selection=None
        try:
            candidate=resolved_candidate;ev=candidate['evidence'];state=np.asarray(state8,float)
            bank=self.reference_witness;current=bank.current;track=self._tool(self.egress_monitor)
            binding=hashlib.sha256(current.rgb.tobytes()+np.asarray(state,'<f8').tobytes()).hexdigest()
            valid=(state.shape==(8,) and np.isfinite(state).all()
                and np.array_equal(current.state,state) and np.array_equal(self.state,state)
                and candidate['observation_binding']==binding and candidate['evidence_sha256']==evidence_digest(ev)
                and ev['kind']=='retrospective_candidate_conditioned' and ev['validation_is_prospective'] is False
                and np.allclose(ev['current_state8'],state,rtol=0,atol=1e-9)
                and isinstance(candidate['candidate_id'],str) and bool(candidate['candidate_id'])
                and track is not None and track.reliable
                and np.allclose(track.uv,candidate['current_point_uv'],rtol=0,atol=.5/223))
            if not valid:raise ValueError('invalid current candidate binding')
            self.reference_selection={'candidate_id':candidate['candidate_id'],'state8':state.tolist(),
                'observation_binding':binding,'evidence_sha256':candidate['evidence_sha256'],
                'track_identity':id(track),'historical_validation_credit':False}
            return {'accepted':True,'physical_budget_modified':False,'prospective_confirmation_required':True}
        except (KeyError,TypeError,ValueError,AttributeError):
            return {'accepted':False,'reason':'route_reference_selection_binding_invalid','physical_budget_modified':False}

    def summary(self):
        result=super().summary()
        selection=getattr(self,'reference_selection',None)
        result['route_reference_selection']=None if selection is None else {k:copy.deepcopy(v) for k,v in selection.items() if k!='track_identity'}
        return result
