"""Explicit reviewer binding before single-use approval; never automatic approval."""
from copy import deepcopy
from .agent_speedup_v2_user_check import ReviewStore as BaseReviewStore, ReviewError, canonical_sha256

class EvidenceReviewStore(BaseReviewStore):
    @classmethod
    def create(cls,path,**kwargs):
        preparation=deepcopy(kwargs.get('preparation') or {})
        preparation.update(approval_mode='pending_explicit_review',manual_review=False,
            approval_provenance={'reviewer':None,'authorized_by':None,'authorization_note':None,'scope':'this_initial_plan_and_one_execution','automatically_approved':False})
        kwargs['preparation']=preparation
        return super().create(path,**kwargs)
    def bind_reviewer(self,reviewer,note):
        if reviewer not in ('human','codex') or not isinstance(note,str) or not note.strip():
            raise ReviewError('explicit reviewer and authorization note required')
        with self._lock():
            data=self._load()
            if data['status']!='draft' or data['preparation']['approval_mode']!='pending_explicit_review':
                raise ReviewError('reviewer can be bound once, to an unapproved pending draft')
            # Preserve the original pending context. No task, reset, plan, source,
            # runtime configuration or selected step changes are allowed here.
            self._write('pending_review_binding.json',data)
            data['preparation'].update(approval_mode='manual_browser_review' if reviewer=='human' else 'user_delegated_agent_review',manual_review=reviewer=='human',approval_provenance={'reviewer':reviewer,'authorized_by':'user','authorization_note':note,'scope':'this_initial_plan_and_one_execution','automatically_approved':False})
            data['revision']+=1
            data['context_sha256']=canonical_sha256(self._context(data))
            self._write('.review-context.json',{'id':data['id'],'context_sha256':data['context_sha256']})
            self._save(data)
            return self._public(data)
    def approve(self,**kwargs):
        data=self.load()
        if data['preparation'].get('approval_mode')=='pending_explicit_review':
            raise ReviewError('reviewer must be explicitly bound before approval')
        return super().approve(**kwargs)
