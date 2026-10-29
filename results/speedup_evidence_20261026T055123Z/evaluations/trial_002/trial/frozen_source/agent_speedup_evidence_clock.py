"""One monotonic service clock; only explicitly recorded human review is paused.

The service deadline cancels in-flight model calls, never a simulator step or
required finalization. Any boundary/finalization overshoot is measured, retained,
and scored outside the window. No evaluator truth is exposed to the policy.
"""
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import signal
import threading
import time

class EvaluationBudgetExceeded(BaseException):
    pass

class ServiceClock:
    VERSION='speedup-service-clock-1'
    def __init__(self, path, *, budget_s=900., clock=time.monotonic, create=False):
        self.path=Path(path);self.now=clock;self._deadline_depth=0
        if create:
            if self.path.exists(): raise FileExistsError(self.path)
            if not math.isfinite(budget_s) or budget_s<=0: raise ValueError('positive finite budget required')
            self.data={'version':self.VERSION,'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'started_monotonic':self.now(),'budget_s':float(budget_s),'human_wait_s':0.,
                'pause_start':None,'finished_monotonic':None,'first_raw_success_service_s':None,
                'events':[],'budget_exhausted':False}
            self.event('service_started');self.save()
        else:
            self.data=json.loads(self.path.read_text())
            if self.data['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():
                raise RuntimeError('monotonic clock cannot cross host reboot')
    def elapsed(self):
        end=self.data['finished_monotonic']
        if end is None: end=self.now()
        wait=self.data['human_wait_s']
        if self.data['pause_start'] is not None: wait+=end-self.data['pause_start']
        return max(0.,end-self.data['started_monotonic']-wait)
    def remaining(self): return max(0.,self.data['budget_s']-self.elapsed())
    def event(self,name,**details):
        self.data['events'].append({'event':name,'monotonic_s':self.now(),'service_elapsed_s':self.elapsed(),**details})
    def check(self):
        if self.remaining()<=0:
            if not self.data['budget_exhausted']:
                self.data['budget_exhausted']=True;self.event('evaluation_budget_exhausted')
            raise EvaluationBudgetExceeded('effective service deadline exhausted')
    def pause_human(self):
        if self.data['pause_start'] is not None: raise RuntimeError('human wait already active')
        self.event('human_review_wait_start');self.data['pause_start']=self.now();self.save()
    def resume(self,*,reason='human_approval_received'):
        if self.data['pause_start'] is None: raise RuntimeError('no human wait to resume')
        self.data['human_wait_s']+=self.now()-self.data['pause_start'];self.data['pause_start']=None
        self.event('human_review_wait_end',reason=reason);self.save()
    def mark_raw(self,success,*,step=None):
        if success is True and self.data['first_raw_success_service_s'] is None:
            self.data['first_raw_success_service_s']=self.elapsed();self.event('first_raw_success',step=step)
    def save(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        target=self.path.with_suffix(self.path.suffix+'.tmp')
        target.write_text(json.dumps({**self.data,'service_elapsed_s':self.elapsed()},indent=2)+'\n')
        os.replace(target,self.path)
    @contextmanager
    def deadline(self):
        self.check()
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError('deadline-controlled model work must run on main thread')
        if self._deadline_depth:
            yield
            return
        old_handler=signal.getsignal(signal.SIGALRM);old_timer=signal.getitimer(signal.ITIMER_REAL)
        if old_timer[0]>0: raise RuntimeError('refusing nested/unrelated SIGALRM timer')
        def expired(_signum,_frame):
            self.data['budget_exhausted']=True;self.event('evaluation_budget_exhausted',source='in_flight_model_deadline')
            raise EvaluationBudgetExceeded('in-flight request reached effective service deadline')
        signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,self.remaining())
        self._deadline_depth+=1
        try: yield
        finally:
            self._deadline_depth-=1
            signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old_handler)
    def call(self,function,*args,event='model_request',**kwargs):
        self.event(event+'_start')
        try:
            with self.deadline(): return function(*args,**kwargs)
        finally: self.event(event+'_end')
    def finish(self):
        if self.data['pause_start'] is not None: raise RuntimeError('cannot finish during human review wait')
        self.event('required_artifacts_persisted')
        self.data['finished_monotonic']=self.now()
        if self.elapsed()>self.data['budget_s']: self.data['budget_exhausted']=True
        self.save()
        return self.report()
    def report(self):
        raw=self.data['first_raw_success_service_s'];elapsed=self.elapsed();budget=self.data['budget_s']
        return {'version':self.VERSION,'service_elapsed_s':elapsed,'human_review_wait_s':self.data['human_wait_s'],
            'first_raw_success_service_s':raw,'raw_success_observed':raw is not None,
            'raw_success_within_900':raw is not None and raw<=budget,
            'S300':raw is not None and elapsed<=300.,'evaluation_budget_s':budget,
            'evaluation_budget_exhausted':self.data['budget_exhausted'],
            'deadline_overshoot_s':max(0.,elapsed-budget),
            'boundary':'preparation + automatic review compute + execution + required finalization including synchronous video',
            'excluded':'explicit human review wait only; final accounting receipt serialization follows measured persistence'}
