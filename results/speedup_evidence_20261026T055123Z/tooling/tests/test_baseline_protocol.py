import importlib.util,json,pathlib,signal,sys,time
from types import SimpleNamespace
import pytest
HERE=pathlib.Path(__file__).resolve().parents[2];BASE=HERE/'baseline';REPO=HERE.parents[1]
sys.path.insert(0,str(BASE/'src'));sys.path.append(str(REPO/'tests'))
from gpt_field.agent_speedup_evidence_clock import ServiceClock,EvaluationBudgetExceeded
from gpt_field.agent_speedup_evidence_review import EvidenceReviewStore
from agent_speedup_v2_fakes import FakeV2Sim
from gpt_field.agent_speedup_v2_tasks import resolve_v2_task
spec=importlib.util.spec_from_file_location('evidence_test_runner',BASE/'scripts/evaluate_speedup_evidence.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)

def test_clock_human_wait_only_and_raw_preserved(tmp_path):
    now=[0.];c=ServiceClock(tmp_path/'clock.json',clock=lambda:now[0],create=True)
    now[0]=30;c.pause_human();now[0]=2030;c.resume(reason='delegated_review_compute_begin')
    now[0]=2040;c.event('automatic_review_end');assert c.elapsed()==40
    now[0]=2290;c.mark_raw(True,step=5);assert c.elapsed()==290
    now[0]=2301;r=c.finish();assert r['service_elapsed_s']==301 and not r['S300']
    assert r['human_review_wait_s']==2000 and r['first_raw_success_service_s']==290

def test_nested_inflight_deadline_cancels_and_restores_timer(tmp_path):
    c=ServiceClock(tmp_path/'clock.json',budget_s=.04,create=True)
    with pytest.raises(EvaluationBudgetExceeded):c.call(lambda:c.call(time.sleep,.3,event='transport'),event='policy_decision')
    assert c.data['budget_exhausted'] and signal.getitimer(signal.ITIMER_REAL)==(0.,0.)
    assert [e['event'] for e in c.data['events']][-2:]==['transport_end','policy_decision_end']

def test_refuse_external_timer(tmp_path):
    c=ServiceClock(tmp_path/'clock.json',create=True);signal.setitimer(signal.ITIMER_REAL,10)
    try:
        with pytest.raises(RuntimeError,match='unrelated'):c.call(lambda:None)
    finally:signal.setitimer(signal.ITIMER_REAL,0)

@pytest.mark.parametrize('arm',['A','B'])
def test_real_private_evaluator_budget_stops_at_complete_feedback(tmp_path,arm):
    now=[0.];clock=ServiceClock(tmp_path/'clock.json',budget_s=3.,clock=lambda:now[0],create=True)
    module=runner.runtime(arm,clock);task=resolve_v2_task('libero_10_t2')
    class Fake(FakeV2Sim):
        def step(self,action):
            value=super().step(action);now[0]+=1.;return value
    sim=Fake(prompt=task.prompt,task_id=task.task_id)
    summary=module.base.execute(tmp_path/'out',config=module.base.configuration(),sim_config=module.base.build_sim_config(task.key)[0],task_key=task.key,episode_index=0,sim_factory=lambda _:sim,video=False,smoke=True,smoke_steps=12)
    assert summary['status']=='evaluation_budget_exhausted' and len(sim.actions)==3
    rows=[json.loads(line) for line in (tmp_path/'out/trace.jsonl').read_text().splitlines()]
    assert sum(r.get('event')=='feedback' for r in rows)==3
    assert sim.closed and not clock.report()['raw_success_observed']

@pytest.mark.parametrize('arm',['A','B'])
def test_first_raw_event_survives_finalization_overrun(tmp_path,arm):
    now=[0.];clock=ServiceClock(tmp_path/'clock.json',budget_s=3.,clock=lambda:now[0],create=True)
    module=runner.runtime(arm,clock);task=resolve_v2_task('libero_10_t2')
    class Fake(FakeV2Sim):
        def step(self,action):
            value=super().step(action);now[0]=2.;return value
        def close(self):
            super().close();now[0]=4.
    sim=Fake(prompt=task.prompt,task_id=task.task_id,success_at=1)
    summary=module.base.execute(tmp_path/'out',config=module.base.configuration(),sim_config=module.base.build_sim_config(task.key)[0],task_key=task.key,episode_index=0,sim_factory=lambda _:sim,video=False,smoke=True,smoke_steps=12)
    report=clock.finish();assert report['first_raw_success_service_s']==2.
    assert report['raw_success_observed'] and report['evaluation_budget_exhausted']
    assert report['service_elapsed_s']==4.

def test_expired_before_approval_writes_terminal_receipt(tmp_path):
    session=tmp_path/'session';session.mkdir();(session/'session.json').write_text(json.dumps({'reservation_id':'fake','arm':'A','task':'libero_10_t1','episode_index':1}))
    clock=ServiceClock(session/'service_clock.json',budget_s=.001,create=True);clock.save();time.sleep(.002)
    # Force budget check at the common command edge without creating a real review.
    original=runner.main
    def expire(_):
        runner._ACTIVE_CLOCK=clock;runner._ACTIVE_DIRECTORY=session;runner._ACTIVE_SESSION=json.loads((session/'session.json').read_text());clock.check()
    runner.main=expire
    try:assert runner.cli([])==3
    finally:runner.main=original
    row=json.loads((session/'evidence_result.json').read_text());assert row['status']=='evaluation_budget_exhausted'
    assert row['raw_libero_success'] is False and row['service_elapsed_s']>0

def test_pending_review_cannot_approve_and_binds_explicit_reviewer(tmp_path):
    historical=json.loads((REPO/'runs/agent_speedup_v2_user_check_libero_10_t1_001/user_check_review.json').read_text())
    initial=json.loads((REPO/'runs/agent_speedup_v2_user_check_libero_10_t1_001/initial_observation.json').read_text())
    store=EvidenceReviewStore.create(tmp_path/'review',task_key=historical['task_key'],episode_index=0,task_prompt=historical['task_prompt'],initial_observation=initial,plan=historical['plan'],runtime_sha256={'source.py':'a'*64},preparation={})
    data=store.load();assert data['preparation']['approval_mode']=='pending_explicit_review'
    with pytest.raises(Exception,match='explicitly bound'):store.approve(expected_revision=0,reviewed_steps=list(range(len(data['plan']['steps']))))
    bound=store.bind_reviewer('codex','offline test only; never execute')
    approved=store.approve(expected_revision=bound['revision'],reviewed_steps=list(range(len(bound['plan']['steps']))))
    assert approved['preparation']['manual_review'] is False and approved['preparation']['approval_provenance']['automatically_approved'] is False
    assert approved['initial_observation_sha256']==data['initial_observation_sha256']
    assert approved['runtime_sha256']==data['runtime_sha256'] and approved['plan']==data['plan']
    assert (store.path/'pending_review_binding.json').exists()

def test_copied_reservation_rejected_even_with_valid_hash(tmp_path):
    row={'event':'reservation','reservation_id':'trial_999','experiment_id':HERE.name,'arm':'A','stage':'P5','task':'libero_10_t1','episode_index':1,'seed':7,'effective_service_budget_s':900,'score_threshold_s':300}
    row['reservation_sha256']=runner.digest(row);path=tmp_path/'copy.json';path.write_text(json.dumps(row))
    with pytest.raises(ValueError,match='canonical'):runner.reservation(path)


def test_reservation_symlink_resolves_to_same_exclusive_claim(tmp_path):
    canonical=tmp_path/'canonical.json';canonical.write_text('{}')
    alias=tmp_path/'alias.json';alias.symlink_to(canonical)
    first=canonical.resolve().with_suffix('.claim.json')
    second=alias.resolve().with_suffix('.claim.json')
    assert first==second
    with first.open('x') as stream:stream.write('{}')
    with pytest.raises(FileExistsError):second.open('x')
