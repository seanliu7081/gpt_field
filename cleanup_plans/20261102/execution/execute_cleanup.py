"""Apply only the user-approved inventory; never recursively remove directories."""
from pathlib import Path,PurePosixPath
from collections import Counter,OrderedDict
import argparse,hashlib,json,os,stat,time
ROOT=Path('/workspace/gpt_field')
PLAN=ROOT/'cleanup_plans/20261102'
OUT=PLAN/'execution'
p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');args=p.parse_args()
m=json.loads((PLAN/'deletion_manifest.json').read_text())
auth=json.loads((OUT/'authorization.json').read_text())
raw=(PLAN/'DELETE_ENTRIES.jsonl').read_bytes();digest=hashlib.sha256(raw).hexdigest()
assert digest==m['inventory_sha256']==auth['inventory_sha256']=='82f73870b0c99f1baf5f5400e125ffc76a88fe26fb17095a5d86febe4fabe7e2'
assert auth['user_message']=='ok delete them all'
assert (OUT/'protected_before.json').exists()
assert json.loads((OUT/'service_stop_verification.json').read_text())['remaining_live_pids']==[]
entries=[json.loads(line) for line in raw.splitlines()]
assert len(entries)==72766
protected=set(m['protected_result_roots'])|set(m['protected_validation_inputs'])
for rel in m['protected_dependencies']:
 if '*' not in rel and (ROOT/rel).exists():protected.add(rel)
protected.update(['src','scripts','tests','configs','data','.git','.venv','.private','cleanup_plans'])
index={e['path']:e for e in entries};assert len(index)==len(entries)
for rel,e in index.items():
 q=PurePosixPath(rel);assert not q.is_absolute() and '..' not in q.parts and '.' not in q.parts
 assert not any(rel==k or rel.startswith(k+'/') for k in protected),(rel,'protected')
 if e['kind']=='directory':assert not any(k.startswith(rel+'/') for k in protected),(rel,'protected descendant')
# Validate each parent and open it using O_NOFOLLOW. Cache only a small number of FDs.
root_fd=os.open(ROOT,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
cache=OrderedDict()
def parent_fd(rel):
 parent=str(PurePosixPath(rel).parent)
 if parent=='.':return root_fd
 if parent in cache:
  cache.move_to_end(parent);return cache[parent]
 fd=os.dup(root_fd)
 try:
  for part in PurePosixPath(parent).parts:
   nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd);os.close(fd);fd=nxt
 except BaseException:
  os.close(fd);raise
 cache[parent]=fd
 if len(cache)>96:
  _,old=cache.popitem(last=False);os.close(old)
 return fd
def state(rel):
 try:return os.stat(PurePosixPath(rel).name,dir_fd=parent_fd(rel),follow_symlinks=False)
 except FileNotFoundError:return None
def kind(s):
 if stat.S_ISREG(s.st_mode):return 'file'
 if stat.S_ISDIR(s.st_mode):return 'directory'
 if stat.S_ISSOCK(s.st_mode):return 'socket'
 if stat.S_ISLNK(s.st_mode):return 'symlink'
 return 'special'
missing=[];drift=[];errors=[];current={}
for e in entries:
 rel=e['path']
 try:s=state(rel)
 except OSError as exc:errors.append({'path':rel,'error':str(exc)});continue
 if s is None:missing.append({'path':rel,'kind':e['kind']});continue
 if kind(s)!=e['kind'] or s.st_dev!=e['device'] or s.st_ino!=e['inode']:
  errors.append({'path':rel,'error':'type or device/inode changed'});continue
 changes=[]
 if s.st_size!=e['size']:changes.append('size')
 if s.st_mtime_ns!=e['mtime_ns']:changes.append('mtime')
 if changes:drift.append({'path':rel,'kind':e['kind'],'fields':changes})
 current[rel]={'device':s.st_dev,'inode':s.st_ino,'size':s.st_size,'mtime_ns':s.st_mtime_ns,'kind':kind(s)}
# Shutdown may update existing logs and remove pid/socket artifacts. Any unrelated drift aborts.
service_dirs={str(PurePosixPath(x['config'].removeprefix(str(ROOT)+'/')).parent) for x in json.loads((OUT/'obsolete_services_before.json').read_text()) if 'config' in x}
def shutdown_artifact(rel):
 pp=PurePosixPath(rel)
 return any(rel.startswith(d+'/') for d in service_dirs) and (pp.suffix in {'.log','.pid','.sock'} or pp.name=='supervisord.log')
for row in drift:
 if row['kind']!='directory' and not shutdown_artifact(row['path']):errors.append({'path':row['path'],'error':'unexpected content drift'})
for row in missing:
 if not shutdown_artifact(row['path']):errors.append({'path':row['path'],'error':'unexpected missing item'})
# Preserve any new path rather than broadening the reviewed deletion list.
unlisted=[]
for rec in m['candidates']:
 start=ROOT/rec['path']
 if not start.is_dir() or start.is_symlink():continue
 for d,dirs,files in os.walk(start,followlinks=False):
  for name in dirs+files:
   rel=str((Path(d)/name).relative_to(ROOT))
   if rel not in index and not any(rel==k or rel.startswith(k+'/') or k.startswith(rel+'/') for k in protected):unlisted.append(rel)
pre={'inventory_sha256':digest,'ready':not errors and not unlisted,'present_entries':len(current),'missing_after_service_shutdown':missing,'drift_after_service_shutdown':drift,'unlisted_paths':unlisted,'errors':errors,'execution_requested':args.execute}
(OUT/'execution_preflight.json').write_text(json.dumps(pre,indent=2)+'\n')
print(json.dumps({'ready':pre['ready'],'present_entries':len(current),'missing':len(missing),'drift':len(drift),'unlisted':len(unlisted),'errors':errors},indent=2),flush=True)
assert pre['ready'],'Review execution_preflight.json; no deletion performed.'
if not args.execute:
 for fd in cache.values():os.close(fd)
 os.close(root_fd)
 raise SystemExit(0)
# Files/sockets first; directories only via rmdir after their approved contents are gone.
counts=Counter();failures=[];t0=time.monotonic()
ledger=OUT/'deletion_ledger.jsonl'
with ledger.open('a',buffering=1) as log:
 for e in entries:
  rel=e['path']
  if e['kind']=='directory':continue
  if rel not in current:counts['already_removed_by_service_shutdown']+=1;continue
  try:
   s=state(rel);c=current[rel]
   if s is None:raise RuntimeError('Path disappeared after preflight')
   if (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,kind(s))!=(c['device'],c['inode'],c['size'],c['mtime_ns'],c['kind']):raise RuntimeError('Path changed after preflight')
   os.unlink(PurePosixPath(rel).name,dir_fd=parent_fd(rel))
   counts['deleted_'+e['kind']]+=1
   log.write(json.dumps({'path':rel,'kind':e['kind'],'status':'deleted'},separators=(',',':'))+'\n')
  except (OSError,RuntimeError) as exc:
   failures.append({'path':rel,'error':str(exc)});break
  if counts['deleted_file'] and counts['deleted_file']%10000==0:print('Deleted files:',counts['deleted_file'],flush=True)
 if not failures:
  for e in sorted((e for e in entries if e['kind']=='directory'),key=lambda e:(e['path'].count('/'),e['path']),reverse=True):
   rel=e['path']
   try:
    s=state(rel)
    if s is None:counts['already_absent_directory']+=1;continue
    if kind(s)!='directory' or (s.st_dev,s.st_ino)!=(e['device'],e['inode']):raise RuntimeError('Directory identity changed')
    os.rmdir(PurePosixPath(rel).name,dir_fd=parent_fd(rel))
    counts['deleted_directory']+=1
    log.write(json.dumps({'path':rel,'kind':'directory','status':'deleted'},separators=(',',':'))+'\n')
   except (OSError,RuntimeError) as exc:failures.append({'path':rel,'error':str(exc)})
for fd in cache.values():os.close(fd)
os.close(root_fd)
remaining=[e['path'] for e in entries if os.path.lexists(ROOT/e['path'])]
sv=os.statvfs('/workspace')
report={'approved_inventory_sha256':digest,'counts':dict(counts),'elapsed_s':time.monotonic()-t0,'failures':failures,'remaining_listed_paths':remaining,'complete':not failures and not remaining,'disk_available_after':sv.f_bavail*sv.f_frsize,'disk_available_before':auth['before_disk']['available'],'disk_available_delta_bytes':sv.f_bavail*sv.f_frsize-auth['before_disk']['available'],'protected_paths_preserved_by_guard':True,'no_unlisted_entries_deleted':True}
(OUT/'deletion_result.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
assert report['complete'],'Review deletion_result.json.'
