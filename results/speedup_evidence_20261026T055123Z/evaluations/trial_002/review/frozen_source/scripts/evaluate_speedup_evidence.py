"""One reserved A/B episode: prepare, explicit review, then single-use run.

A is historical v7 IO; B only reuses compact video and display-status throttling.
No automatic approval, model changes, observation changes, or controller changes.
"""
from __future__ import annotations
import argparse
import ast
import inspect
from dataclasses import replace
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['PYTHONPATH']=str(ROOT/'src')
from gpt_field.agent_speedup_evidence_clock import ServiceClock, EvaluationBudgetExceeded
from gpt_field.agent_speedup_evidence_adapter import adapt_service_budget
from gpt_field.agent_speedup_evidence_review import EvidenceReviewStore
from gpt_field.ordered_transport import ordered_sse_transport
from gpt_field.agent_speedup_v3_performance import adapt_private_evaluator
from gpt_field.agent_speedup_v3_video import AgentSpeedUpV3VideoRecorder

def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def write(path,data):
    target=path.with_suffix(path.suffix+'.tmp');target.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n');target.replace(path)

def runtime(arm,clock=None):
    spec=importlib.util.spec_from_file_location('_speedup_evidence_v7',ROOT/'scripts/evaluate_agent_speedup_v2_user_check.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.ReviewStore=EvidenceReviewStore
    source=inspect.getsource(module._normalize_artifacts)
    old='preparation.get("approval_mode", "manual_browser_review") != "user_authorized_automatic"'
    if source.count(old)!=1:raise RuntimeError("review provenance adapter mismatch")
    source=source.replace(old,'preparation.get("manual_review") is True')
    source=source.replace('user-authorized automatically approved','explicitly delegated Codex-reviewed')
    exec(compile(source,module.__file__+"::explicit_reviewer","exec"),module.__dict__)
    inventory=module.runtime_inventory
    def bound_inventory(*args,**kwargs):
        result=inventory(*args,**kwargs)
        result['scripts/evaluate_speedup_evidence.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        result['@evidence_protocol']=digest({'arm':arm,'clock':ServiceClock.VERSION,'budget_s':900,'display':'historical_v7' if arm=='A' else 'compact_v3_video_and_snapshot_gate'})
        return result
    module.runtime_inventory=bound_inventory
    if clock is not None:
        adapt_service_budget(module.base,clock)
        def instrument(client_class):
            class TimedClient(client_class):
                def __init__(self,*args,**kwargs):
                    original=kwargs.pop('transport',None) or ordered_sse_transport
                    def transport(*values,**options):
                        clock.event('model_request_start')
                        try:return clock.call(original,*values,event='transport',**options)
                        finally:clock.event('model_request_end')
                    if getattr(original,'offline_mock',False):transport.offline_mock=True
                    super().__init__(*args,transport=transport,**kwargs)
            return TimedClient
        module.AgentFirstV2Client=instrument(module.AgentFirstV2Client)
        module.UserCheckV2Client=instrument(module.UserCheckV2Client)
    if arm=='B':
        adapt_private_evaluator(module.base)
        module.base._default_recorder_factory=lambda:AgentSpeedUpV3VideoRecorder
    return module

def reservation(path):
    path=path.resolve()
    result_root=ROOT.parent
    data=json.loads(path.read_text());claimed=data.pop('reservation_sha256',None)
    if not claimed or claimed!=digest(data):raise ValueError('reservation hash mismatch')
    if data['arm'] not in ('A','B') or data['stage'] not in ('P5','P6'):raise ValueError('full-task A/B reservation required')
    if data.get('effective_service_budget_s')!=900 or data.get('score_threshold_s')!=300:raise ValueError('protocol budget mismatch')
    canonical=(result_root/'reservations'/(data['reservation_id']+'.json')).resolve()
    if path!=canonical:raise ValueError('reservation must use canonical experiment path')
    rows=[json.loads(line) for line in (result_root/'budget_ledger.jsonl').read_text().splitlines()]
    matching=[row for row in rows if row.get('event')=='reservation' and row.get('reservation_id')==data['reservation_id']]
    complete={**data,'reservation_sha256':claimed}
    if matching!=[complete]:raise ValueError('reservation missing or mismatched in cumulative ledger')
    return complete

def configs(module,data):
    api=module.base.configuration()
    sim,monitor=module.base.build_sim_config(data['task'])
    sim=replace(sim,seed=data['seed'])
    return api,sim,monitor

def token(args):
    from gpt_field.chatgpt_auth import SubscriptionCredentials
    return SubscriptionCredentials(store_dir=args.store_dir,profile=args.profile).access_token

def summarize(session,clock,summary,review_evidence):
    timing=clock.report();raw=timing['raw_success_observed']
    evidence={'reservation_id':session['reservation_id'],'status':'evaluation_budget_exhausted' if timing['evaluation_budget_exhausted'] else summary.get('status'),
        'raw_libero_success':raw,'time_to_raw_success_s':timing['first_raw_success_service_s'],
        'service_elapsed_s':timing['service_elapsed_s'],'human_review_wait_s':timing['human_review_wait_s'],
        'automatic_review_compute_s':review_evidence.get('automatic_review_compute_s',0.),
        'approved_by':review_evidence.get('approved_by'),'review_modified':review_evidence.get('modified'),
        'autonomous_quality':{'terminal_physical_diagnostics':summary.get('physical_truth'),
            'post_success_release_or_settle_is_separate_diagnostic':True,
            'operator_action_interventions':0,'policy_fallback_counts':'see agent_stats.json'},
        'clock':timing,'arm':session['arm'],'task':session['task'],'episode_index':session['episode_index'],
        'raw_evaluator_summary_status':summary.get('status'),'full_summary':'trial/summary.json',
        'historical_v7_approval_reused':False}
    return evidence

_ACTIVE_CLOCK=None
_ACTIVE_DIRECTORY=None
_ACTIVE_SESSION=None

def main(argv=None):
    global _ACTIVE_CLOCK,_ACTIVE_DIRECTORY,_ACTIVE_SESSION
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('preflight');p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--task',default='libero_10_t1');p.add_argument('--episode-index',type=int,default=0)
    p=sub.add_parser('prepare');p.add_argument('--session-dir',type=Path,required=True);p.add_argument('--reservation-file',type=Path,required=True);p.add_argument('--enroll-reset',action='store_true');p.add_argument('--expected-initial-observation-sha256');p.add_argument('--store-dir',type=Path);p.add_argument('--profile')
    p=sub.add_parser('review-begin');p.add_argument('--session-dir',type=Path,required=True);p.add_argument('--authorization-note',required=True)
    p=sub.add_parser('approve');p.add_argument('--session-dir',type=Path,required=True);p.add_argument('--reviewer',choices=['human','codex'],required=True);p.add_argument('--authorization-note',required=True);p.add_argument('--reviewed-steps',type=int,nargs='+',required=True)
    p=sub.add_parser('run');p.add_argument('--session-dir',type=Path,required=True);p.add_argument('--store-dir',type=Path);p.add_argument('--profile')
    args=parser.parse_args(argv)
    if args.command=='preflight':
        m=runtime(args.arm);api,sim,monitor=configs(m,{'task':args.task,'seed':7})
        result=m.preflight(api,sim,task_key=args.task,episode_index=args.episode_index,monitor_options=monitor,video=True)
        result.update(evidence_arm=args.arm,service_clock=ServiceClock.VERSION,evaluation_budget_s=900,score_threshold_s=300,auto_approval=False)
        print(json.dumps(result,indent=2));return 0
    directory=args.session_dir.resolve()
    if args.command=='prepare':
        if directory.exists():raise FileExistsError(directory)
        data=reservation(args.reservation_file)
        # Atomic one-session binding survives crashes and cannot be reclaimed.
        claim=args.reservation_file.resolve().with_suffix('.claim.json')
        with claim.open('x') as stream:json.dump({'session_dir':str(directory),'reservation_id':data['reservation_id']},stream)
        directory.mkdir(parents=True);write(directory/'session.json',data)
        clock=ServiceClock(directory/'service_clock.json',create=True)
        _ACTIVE_CLOCK,_ACTIVE_DIRECTORY,_ACTIVE_SESSION=clock,directory,data
        m=runtime(data['arm'],clock);api,sim,monitor=configs(m,data)
        if args.enroll_reset:
            if args.expected_initial_observation_sha256:raise ValueError('choose enrollment or expected hash')
            original=m.base.resolve_expected_sha
            def enroll(task,episode,expected=None,**kw):
                if expected is None:return None,'explicit_first_official_reset_enrollment'
                return original(task,episode,expected,**kw)
            m.base.resolve_expected_sha=enroll
        try:
            clock.event('preparation_start')
            clock.check()
            store=m.prepare_review(directory/'review',config=api,sim_config=sim,task_key=data['task'],episode_index=data['episode_index'],monitor_options=monitor,token_provider=token(args),expected_initial_observation_sha256=args.expected_initial_observation_sha256,video=True,auto_approve=False)
            clock.event('preparation_complete');clock.check()
            review=store.load()
            write(directory/'preparation_result.json',{'status':'review_required','initial_observation_sha256':review['initial_observation_sha256'],'review_id':review['id'],'validation':review['validation'],'enrolled':args.enroll_reset})
            clock.pause_human()
            print(json.dumps({'status':'review_required','session_dir':str(directory),'initial_observation_sha256':review['initial_observation_sha256']}));return 0
        except BaseException as exc:
            clock.event('preparation_failed',exception_type=type(exc).__name__);clock.finish()
            summary={'status':'evaluation_budget_exhausted' if isinstance(exc,EvaluationBudgetExceeded) else 'preparation_error'}
            write(directory/'evidence_result.json',summarize(data,clock,summary,{}));raise
    data=json.loads((directory/'session.json').read_text());clock=ServiceClock(directory/'service_clock.json')
    _ACTIVE_CLOCK,_ACTIVE_DIRECTORY,_ACTIVE_SESSION=clock,directory,data
    if args.command=='review-begin':
        clock.resume(reason='delegated_review_compute_begin');clock.event('automatic_review_start',authorization_note=args.authorization_note);clock.save();return 0
    m=runtime(data['arm'],clock);store=m.ReviewStore(directory/'review')
    if args.command=='approve':
        before=store.load()
        if args.reviewer=='codex':
            starts=[e for e in clock.data['events'] if e['event']=='automatic_review_start']
            if not starts or clock.data['pause_start'] is not None:raise ValueError('call review-begin before inspecting delegated draft')
            review_start=starts[-1]['service_elapsed_s']
        else:
            clock.resume(reason='explicit_human_approval');review_start=clock.elapsed()
        clock.check();bound=store.bind_reviewer(args.reviewer,args.authorization_note)
        result=store.approve(expected_revision=bound['revision'],reviewed_steps=args.reviewed_steps)
        evidence={'approved_by':args.reviewer,'authorization_note':args.authorization_note,'human_reviewed_plan':args.reviewer=='human','modified':before['draft_plan']!=result['plan'],'selected_plan_sha256':digest(result['plan']),'automatic_review_compute_s':clock.elapsed()-review_start if args.reviewer=='codex' else 0.}
        write(directory/'review_evidence.json',evidence);clock.event('approval_complete',**evidence);clock.save();print(json.dumps({'status':result['status'],'review_id':result['id']}));return 0
    if store.load()['status']!='approved':raise ValueError('fresh explicitly approved plan required')
    review_evidence=json.loads((directory/'review_evidence.json').read_text())
    if clock.data['pause_start'] is not None:raise ValueError('approval wait not ended')
    clock.check();api,sim,monitor=configs(m,data);clock.event('reviewed_execution_start');clock.save()
    try:
        summary=m.execute_reviewed(directory/'trial',review_dir=directory/'review',config=api,sim_config=sim,task_key=data['task'],episode_index=data['episode_index'],monitor_options=monitor,token_provider=token(args),video=True)
        clock.event('policy_and_evaluator_finished',status=summary['status'])
        # Review provenance is separate from old v7 auto-approval labels.
        summary.update(human_reviewed_plan=review_evidence['human_reviewed_plan'],approved_by=review_evidence['approved_by'],evidence_review=review_evidence)
        write(directory/'trial/summary.json',summary)
        clock.finish();write(directory/'evidence_result.json',summarize(data,clock,summary,review_evidence))
        print(json.dumps(json.loads((directory/'evidence_result.json').read_text())));return 0
    except BaseException as exc:
        clock.event('runner_failed',exception_type=type(exc).__name__);clock.finish()
        path=directory/'trial/summary.json';summary=json.loads(path.read_text()) if path.exists() else {'status':'runner_error'}
        write(directory/'evidence_result.json',summarize(data,clock,summary,review_evidence));raise

def cli(argv=None):
    try:return main(argv)
    except EvaluationBudgetExceeded:
        if _ACTIVE_CLOCK is None:raise
        path=_ACTIVE_DIRECTORY/'evidence_result.json'
        if not path.exists():
            _ACTIVE_CLOCK.finish()
            review_path=_ACTIVE_DIRECTORY/'review_evidence.json'
            review=json.loads(review_path.read_text()) if review_path.exists() else {}
            starts=[e for e in _ACTIVE_CLOCK.data['events'] if e['event']=='automatic_review_start']
            if starts and not review:
                review={'approved_by':None,'automatic_review_compute_s':_ACTIVE_CLOCK.elapsed()-starts[-1]['service_elapsed_s']}
            write(path,summarize(_ACTIVE_SESSION,_ACTIVE_CLOCK,{'status':'evaluation_budget_exhausted'},review))
        print(json.dumps({'status':'evaluation_budget_exhausted','evidence_result':str(path)}))
        return 3

if __name__=='__main__':raise SystemExit(cli())
