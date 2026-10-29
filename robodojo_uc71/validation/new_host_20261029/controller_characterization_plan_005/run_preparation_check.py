"""One isolated preparation-only check; --launch is forbidden."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
BASE=Path('/workspace/gpt_field/robodojo_uc71')
sys.path[:0]=[str(BASE/'scripts'),str(BASE/'src')]
from prepare_controller_characterization import program_files,content_hash,check_authorization,verify_first_probe,validate_diagnostic_plan
from provision_controller_characterization import prepare_worker_git_config
from provision_native_common import pins
from robodojo_uc71.worker_isolation import popen_identity
V=BASE/'validation/new_host_20261029';OUT=V/'controller_characterization_preparation_009'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
assert not OUT.exists()
OUT.mkdir(mode=0o750);os.chown(OUT,0,65534);os.chmod(OUT,0o750)
for name in ('home','cache','config','data','worker','tmp'):
 p=OUT/name;p.mkdir(mode=0o700);os.chown(p,65534,65534)
(OUT/'operator_reviews').mkdir(mode=0o750);os.chown(OUT/'operator_reviews',0,65534)
plan_path=V/'controller_characterization_plan_005/proposal.json';plan=json.loads(plan_path.read_text())
validate_diagnostic_plan(plan,'controller-characterization')
scope=json.loads((V/'controller_characterization_plan_005/scope_009_proposal.json').read_text())
assert scope['preparation_only'] is True and scope['diagnostic_native_actions_authorized'] is False
scope['attempt_output']=str(OUT)
scope['authorization_basis']='Isolated read-only preparation check only. --launch is forbidden; actions and simulator startup are unauthorized.'
scope_path=OUT/'preparation_scope.json';write(scope_path,scope)
programs={str(p):digest(p) for p in program_files()};assert programs==scope['program_sha256']
write(OUT/'program_binding.json',programs)
prior=verify_first_probe(V/'native_characterization_001/launch_result.json',V/'joint_probe_analysis_001/analysis.json',plan)
authorization=check_authorization(scope_path,plan,plan['binding'],first_probe=prior,launch=False,output_parent=OUT,diagnostic_kind='controller-characterization')
old_dir=V/'native_controller_characterization_008';old=json.loads((old_dir/'launch.json').read_text())
env={k:v.replace(str(old_dir),str(OUT)) for k,v in old['environment'].items()}
for key in ('UC71_CONTROLLER_CHARACTERIZATION_AUTHORIZED_CHILD','UC71_OPEN_CLEARANCE_PREVIEW_AUTHORIZED_CHILD'):env.pop(key,None)
env['UC71_CONTROLLER_PARENT_PID']=str(os.getpid())
prepare_worker_git_config(OUT,BASE/'runtime/native/RoboDojo',pins())
oldargv=old['argv'];script=str(BASE/'scripts/prepare_controller_characterization.py')
args=[x.replace(str(old_dir),str(OUT)) for x in oldargv[oldargv.index(script):] if x!='--launch']
for flag,value in {'--authorization':str(scope_path),'--plan':str(plan_path)}.items():args[args.index(flag)+1]=value
args.extend(['--diagnostic-kind','controller-characterization'])
assert '--launch' not in args
wrapper="""import json,os,runpy,sys
from pathlib import Path
sys.path[:0]=['/workspace/gpt_field/robodojo_uc71/scripts','/workspace/gpt_field/robodojo_uc71/src']
from robodojo_uc71.worker_isolation import enforce_child_boundary
proof=enforce_child_boundary(credential_store='/workspace/gpt_field/.private/chatgpt_plan',parent_pid=int(os.environ['UC71_CONTROLLER_PARENT_PID']))
root=Path(os.environ['HOME']).parent/'worker'
with (root/'preparation_boundary.json').open('x') as f:json.dump(proof,f);f.flush();os.fsync(f.fileno())
sys.argv=sys.argv[1:]
assert '--launch' not in sys.argv
try:runpy.run_path(sys.argv[0],run_name='__main__')
finally:
 blocked=sorted(n for n in sys.modules if n.split('.')[0] in {'isaacsim','isaaclab','omni','torch','carb','pxr'})
 audit={'uid':os.getuid(),'pid':os.getpid(),'native_imports':blocked,'launch_argument_present':False,'full_test_authorized':False}
 with (root/'preparation_import_audit.json').open('x') as f:json.dump(audit,f);f.flush();os.fsync(f.fileno())
"""
argv=[str(BASE/'runtime/new_host_env/bin/python'),'-B','-X','faulthandler','-u','-c',wrapper,*args]
write(OUT/'invocation.json',{'argv':argv,'environment':env,'scope_sha256':digest(scope_path),'program_sha256':programs,'simulator_launch_forbidden':True,'existing_permissions_changed':False})
started=time.monotonic()
with (OUT/'stdout_stderr.log').open('xb') as stream:
 proc=subprocess.Popen(argv,env=env,cwd=OUT,stdout=stream,stderr=subprocess.STDOUT,**popen_identity())
 try:code=proc.wait(timeout=120)
 except subprocess.TimeoutExpired:proc.kill();proc.wait();raise
ready=json.loads((OUT/'worker/controller/preparation.json').read_text()) if code==0 else None
audit=json.loads((OUT/'worker/preparation_import_audit.json').read_text())
proof=json.loads((OUT/'worker/preparation_boundary.json').read_text())
unchanged=programs=={str(p):digest(p) for p in program_files()}
passed=bool(code==0 and ready and ready['launch_requested'] is False and ready['open_only'] is False and ready['diagnostic_native_actions_planned']==960 and ready['plan_sha256']==content_hash(plan) and ready['source_file_sha256']==programs and audit['native_imports']==[] and proof['uid']==65534 and proof['access_denied']=={'credential_directory':True,'parent_environment':True} and unchanged)
write(OUT/'result.json',{'schema':'robodojo-controller-characterization-preparation-check-1','passed':passed,'returncode':code,'elapsed_s':time.monotonic()-started,'simulator_starts':0,'native_actions':0,'model_calls':0,'full_test_authorized':False,'scope_sha256':digest(scope_path),'program_sha256':programs,'programs_unchanged':unchanged,'preparation_sha256':digest(OUT/'worker/controller/preparation.json') if ready else None,'boundary':{'path':str(OUT/'worker/preparation_boundary.json'),'sha256':digest(OUT/'worker/preparation_boundary.json')},'import_audit':{'path':str(OUT/'worker/preparation_import_audit.json'),'sha256':digest(OUT/'worker/preparation_import_audit.json')},'stdout_sha256':digest(OUT/'stdout_stderr.log'),'existing_permissions_changed':False})
print(json.dumps({'passed':passed,'returncode':code,'result':str(OUT/'result.json'),'sha256':digest(OUT/'result.json')}))
if not passed:raise SystemExit(1)
