import copy
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SPEC = importlib.util.spec_from_file_location('bowl_chain_analysis', Path(__file__).parents[1]/'scripts/analyze_bowl_chain.py')
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def write(path, value):
    path.write_text(json.dumps(value)+'\n')


def write_trace(path, rows):
    with gzip.open(path,'wt') as stream:
        for row in rows:
            stream.write(json.dumps(row)+'\n')


def prediction(values=None):
    coords=analysis.GridSpec().coordinates((0,0,0))/.02
    return {'values': (np.sum(coords**2,axis=1) if values is None else values).tolist(),
        'blocked_indices': [], 'unknown_indices': [], 'stage':'approach', 'gripper_command':'open',
        'motion_mode':'move','orientation_target_xyzw':[1,0,0,0]}


def observation(step):
    return {'observation/state':[.001*step,0,1,0,0,0,.04,-.04],
        'observation/image':f'image{step}','observation/wrist_image':f'wrist{step}'}


def sim_status(step, success=False):
    return {'steps':step,'success':success,'terminated':success,'horizon_reached':False}


def manifest(run, config):
    identity=analysis.canonical_hash(config)
    write(run/'manifest.json',{'configuration':config,'identity_hash':identity})
    return identity


def fixture_chain(tmp_path, success=True):
    chain=[]
    previous=None
    root=tmp_path/'root'
    root_identity=None
    for index,(start,count) in enumerate(((0,5),(5,5),(10,2))):
        run=root if index==0 else tmp_path/f'child{index}'
        run.mkdir();(run/'api').mkdir()
        config={'grid':analysis.GridSpec().to_dict()} if index==0 else {
            'parent_run':previous['run'],'parent_identity_hash':previous['identity_hash'],'continuation_offset':start}
        if index==2:
            config.update(root_run=str(root),root_identity_hash=root_identity,chain=copy.deepcopy(chain),
                source_trace={'kind':'ordered_chain','segments':copy.deepcopy(chain),'available_completed_actions':start})
        identity=manifest(run,config)
        root_identity=root_identity or identity
        end=start+count
        terminal='success' if index==2 and success else 'budget_exceeded'
        row={'task_prompt':analysis.TASK,'suite':'libero_spatial','task_id':0,'episode_index':0,
            'seed':7,'method':'field','status':terminal,'log':'trace.jsonl.gz','steps':count}
        if index:
            row.update(continuation_offset=start,reconstruction_actions=start,new_evaluated_actions=count,
                cumulative_actions=end,evaluation_completed=terminal=='success',libero_success=True if terminal=='success' else None)
            write(run/'summary.json',row)
        write(run/'episodes.jsonl',row)
        state={**row,'stage':'terminal','budget':{'api_calls':index+1,'unknown_usage_calls':0,'cost_usd_upper_bound':1.0}}
        write(run/'run_state.json',state)
        rows=[{'event':'reset' if index==0 else 'continuation_start','observation':observation(start),'sim_status':sim_status(start)}]
        for step in range(start,end):
            details={'step':step,'method':'field','refreshed':step%5==0,'stage':'approach',
                'field_center':observation(start)['observation/state'][:3],
                'plan':{'delta_xyz':[.01,0,0],'status':'move','candidate_count':1,'valid_count':1,
                        'candidates':[{'valid':True,'invalid_reason':None}]}}
            if details['refreshed']:
                ordinal=step//5+1
                request_path=run/'api'/f'{ordinal:04d}.request.json'
                write(request_path,{'request_ordinal':ordinal,'request_key':f'key{ordinal}'})
                write(request_path.with_name(f'{ordinal:04d}.result.json'),{'status':'valid'})
                details.update(prediction=prediction(),field_prediction=prediction(),api_request_log=str(request_path),api_request_key=f'key{ordinal}')
            rows.append({'step':step,'state':observation(step)['observation/state'],'action':[.2,0,0,0,0,0,-1.0],
                         'diagnostics':details,'observation':observation(step)})
            succeeded=terminal=='success' and step==end-1
            rows.append({'event':'executed','step':step,'next_state':observation(step+1)['observation/state'],
                'observation':observation(step+1),'success':succeeded,'sim_status':sim_status(step+1,succeeded)})
        write_trace(run/'trace.jsonl.gz',rows)
        item={'run':str(run),'identity_hash':identity,'action_start':start,'action_count':count,
              'trace':{'trace':'trace.jsonl.gz','snapshot_sha256':analysis.digest(run/'trace.jsonl.gz')}}
        chain.append(item);previous=item
    return run


def test_live_source_refused_before_trace_or_output(tmp_path):
    source=tmp_path/'live';source.mkdir();write(source/'run_state.json',{'status':'running'})
    with pytest.raises(ValueError,match='not terminal'):
        analysis.main(['--source',str(source),'--output-prefix',str(tmp_path/'diagnostic_live')])
    assert not list(tmp_path.glob('diagnostic_live.*'))


def test_chain_counts_each_segment_once_and_partial_success_field(tmp_path):
    source=fixture_chain(tmp_path)
    report,records=analysis.build_report(source)
    assert len(records)==report['included_completed_actions']==12
    assert [s['action_count'] for s in report['source_segments']]==[5,5,2]
    assert report['accepted_fields_with_executed_actions']==3
    assert report['observations']['field_structure']['complete_five_action_fields']==2
    assert report['per_refresh'][-1]['executed_axes_before_next_refresh']==['+x','+x']
    assert report['libero_success'] is True and report['evaluation_completed'] is True
    assert report['historical_api_attempts']==3


def test_incomplete_budget_is_not_scored(tmp_path):
    report,_=analysis.build_report(fixture_chain(tmp_path,success=False))
    assert report['terminal_status']=='budget_exceeded'
    assert report['libero_success'] is None and report['evaluation_completed'] is False
    assert 'success_rate' not in report


def test_duplicate_segment_refused(tmp_path):
    source=fixture_chain(tmp_path)
    m=analysis.read(source/'manifest.json');c=m['configuration']
    c['chain'][1]=copy.deepcopy(c['chain'][0]);c['source_trace']['segments']=copy.deepcopy(c['chain'])
    manifest(source,c)
    with pytest.raises(ValueError,match='Duplicate'):
        analysis.build_report(source)


def test_wrong_span_refused(tmp_path):
    source=fixture_chain(tmp_path)
    c=analysis.read(source/'manifest.json')['configuration'];c['chain'][1]['action_start']=6
    c['source_trace']['segments']=copy.deepcopy(c['chain']);manifest(source,c)
    with pytest.raises(ValueError,match='span'):
        analysis.build_report(source)


def test_changed_historical_trace_hash_refused(tmp_path):
    source=fixture_chain(tmp_path)
    path=tmp_path/'root/trace.jsonl.gz'
    with path.open('ab') as stream:stream.write(b'x')
    with pytest.raises(ValueError,match='hash changed'):
        analysis.build_report(source)


def test_discontinuous_state_refused(tmp_path):
    source=fixture_chain(tmp_path)
    path=source/'trace.jsonl.gz'
    with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
    rows[1]['state'][0]+=.1
    write_trace(path,rows)
    with pytest.raises(ValueError,match='state is discontinuous'):
        analysis.build_report(source)


def test_unexecuted_terminal_decision_excluded(tmp_path):
    source=fixture_chain(tmp_path,success=False)
    path=source/'trace.jsonl.gz'
    with gzip.open(path,'rt') as stream:rows=[json.loads(line) for line in stream]
    trailing=copy.deepcopy(rows[-2]);trailing['step']=12;trailing['diagnostics']['step']=12
    trailing['state']=observation(12)['observation/state'];trailing['observation']=observation(12)
    rows.append(trailing);write_trace(path,rows)
    report,records=analysis.build_report(source)
    assert len(records)==12
    assert report['excluded_unexecuted_decisions']==[{'run':str(source),'step':12,'refreshed':False,'request_ordinal':None}]


def test_masked_placeholders_do_not_create_curvature():
    xyz=analysis.GridSpec().coordinates((0,0,0))/.02
    p=prediction(1+xyz[:,0]+2*xyz[:,1]-xyz[:,2]);p['unknown_indices']=[0,10,20]
    for index in p['unknown_indices']:p['values'][index]=1e10
    fits,valid,_=analysis.fit_field(p)
    assert valid.sum()==726
    assert fits['affine']['max_absolute_residual']<1e-10


def test_cross_axis_field_distinguished_from_additive():
    xyz=analysis.GridSpec().coordinates((0,0,0))/.02
    fits,_,_=analysis.fit_field(prediction(xyz[:,0]*xyz[:,1]))
    assert fits['full_quadratic']['max_absolute_residual']<1e-10
    assert fits['separable_quadratic']['max_absolute_residual']>1
    assert fits['arbitrary_additive_separable']['max_absolute_residual']>1


def test_constant_field_and_hold_motion_defined():
    fits,_,_=analysis.fit_field(prediction(np.ones(729)))
    assert fits['affine']['rms_residual_over_value_std'] is None
    records=[{'state':[0,0,1], 'next_state':[0,0,1], 'delta_xyz':[0,0,0]}]
    summary=analysis.motion_summary(records,0,1)
    assert summary['commanded_axes']=={'hold':1}
    assert summary['measured_path_over_commanded_target_delta_path'] is None


def test_outputs_never_overwrite(tmp_path):
    output=tmp_path/'diagnostic_saved.json';output.write_text('original')
    with pytest.raises(ValueError,match='already exists'):
        analysis.main(['--source',str(tmp_path/'does_not_exist'),'--output-prefix',str(output.with_suffix(''))])
    assert output.read_text()=='original'
