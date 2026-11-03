"""Build a read-only deletion inventory. This script never deletes or stops anything."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import csv,hashlib,json,os,stat,subprocess
ROOT=Path('/workspace/gpt_field')
OUT=ROOT/'cleanup_plans/20261102'
uc=json.loads((OUT/'uc7_keep_audit.json').read_text())
astra=json.loads((OUT/'astranomy_preservation_audit.json').read_text())
other=json.loads((OUT/'outside_results_audit.json').read_text())
keep={p:'Verified UC7/UC7.1 run or associated review/preparation/launch' for p in uc['keep_roots']}
keep.update({v['path']:v['reason'] for v in astra['keep_roots']})
keep.update(other['keep_result_trees'])
keep['runs/cleanup_20261025_001']='Historical cleanup audit; not experiment output'
keep['runs/speedup_next_20261026T101438Z']='V7-based request/transport investigation; four planning probes, no robot actions'
protected_files={
 'runs/v3_completion_baseline_20261026T133633Z/attempts/attempt_001/execution/approved_plan.json': 'Input cited by retained UC7.1 direction_evidence.json schema validation',
 'runs/v3_completion_baseline_20261026T133633Z/attempts/attempt_003/execution/approved_plan.json': 'Input cited by retained UC7.1 direction_evidence.json schema validation',
}
protected=keep|protected_files
candidates=[]
for p in sorted((ROOT/'runs').iterdir()):
 rel=str(p.relative_to(ROOT))
 if rel in keep:continue
 n=p.name
 if n.startswith('agent_speedup_v3') or n.startswith('v3_completion_'):group='SpeedUp v3 results and reviews'
 elif n.startswith('agent_speedup_v1'):group='SpeedUp v1 results'
 elif n.startswith('agent_speedup_v2'):group='Older SpeedUp v2 / User Check v1-v6'
 elif n.startswith('robodojo_direct'):group='RoboDojo Direct results (separate method)'
 else:group='Other legacy field/hybrid/direct experiments'
 candidates.append({'path':rel,'category':group,'reason':'Outside the two requested retained method families'})
for c in other['candidates']:
 rel=c['path']
 group='Old scene-field/reconstruction outputs' if rel.startswith('scene_test/') else 'Old v1/hybrid presentation outputs' if rel.startswith(('presentations/','reports/')) else 'Old aggregate results'
 candidates.append({'path':rel,'category':group,'reason':c['reason']})
candidates.sort(key=lambda v:v['path'])
roots=[r['path'] for r in candidates]
assert len(roots)==len(set(roots))
for i,p in enumerate(roots):
 assert not any(p.startswith(q+'/') for q in roots[:i]),('nested candidate',p)
 assert not any(p==q or p.startswith(q+'/') for q in protected),('protected candidate',p)
 assert (ROOT/p).exists() or (ROOT/p).is_symlink(),('missing candidate',p)
for q in other['protected_dependencies']:
 if '*' not in q and (ROOT/q).exists():
  assert not any(q==p or q.startswith(p+'/') for p in roots),('dependency would be removed',q)
tracked=set(subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).decode().split('\0'))
groups=defaultdict(lambda:Counter())
entry_counts=Counter();inode_counts=Counter();inode_meta={};inode_groups=defaultdict(set)
non_dir=[];directories=[];errors=[]
entries_path=OUT/'DELETE_ENTRIES.jsonl'
def excluded(rel):return any(rel==q or rel.startswith(q+'/') for q in protected)
def needed_ancestor(rel):return any(q.startswith(rel+'/') for q in protected)
with entries_path.open('w') as entries:
 for rec in candidates:
  stats=Counter();root=ROOT/rec['path']
  retained=[p for p in protected_files if p.startswith(rec['path']+'/')]
  rec['mode']='listed contents only; keep cited validation inputs' if retained else 'all inventoried contents'
  rec['preserved_descendants']=retained
  def visit(path):
   rel=str(path.relative_to(ROOT))
   if excluded(rel):return
   try:s=path.lstat()
   except OSError as e:errors.append({'path':rel,'error':str(e)});return
   is_dir=stat.S_ISDIR(s.st_mode)
   if is_dir:
    for child in sorted(path.iterdir(),key=lambda v:v.name):visit(child)
    if needed_ancestor(rel):return
    kind='directory'
   elif stat.S_ISLNK(s.st_mode):kind='symlink'
   elif stat.S_ISREG(s.st_mode):kind='file'
   elif stat.S_ISSOCK(s.st_mode):kind='socket'
   else:kind='special'
   entry={'path':rel,'kind':kind,'size':s.st_size,'allocated_bytes':s.st_blocks*512,'mtime_ns':s.st_mtime_ns,'device':s.st_dev,'inode':s.st_ino,'nlink':s.st_nlink,'git_tracked':rel in tracked,'root':rec['path']}
   if kind=='symlink':entry['target']=os.readlink(path)
   entries.write(json.dumps(entry,separators=(',',':'))+'\n')
   count_key='directories' if kind=='directory' else kind+'s'
   stats[count_key]+=1;entry_counts[count_key]+=1
   if kind=='directory':directories.append(rel)
   else:non_dir.append(rel)
   stats['logical_bytes']+=s.st_size;stats['allocated_bytes']+=s.st_blocks*512
   if entry['git_tracked']:stats['git_tracked_entries']+=1
   key=(s.st_dev,s.st_ino);inode_counts[key]+=1;inode_meta[key]=(s.st_nlink,s.st_blocks*512,kind);inode_groups[key].add(rec['category'])
  visit(root)
  rec['inventory']=dict(stats)
  groups[rec['category']].update(stats);groups[rec['category']]['roots']+=1
assert not errors,errors
# Only count file storage as reclaimable when every hardlink is selected.
reclaimable=sum(blocks for key,(links,blocks,kind) in inode_meta.items() if kind=='directory' or inode_counts[key]>=links)
partial_hardlinks=[{'device':k[0],'inode':k[1],'links':v[0],'selected':inode_counts[k]} for k,v in inode_meta.items() if v[2]!='directory' and inode_counts[k]<v[0]]
(OUT/'DELETE_FILES.txt').write_text('\n'.join('/workspace/gpt_field/'+p for p in sorted(non_dir))+'\n')
(OUT/'DELETE_DIRECTORIES.txt').write_text('\n'.join('/workspace/gpt_field/'+p for p in sorted(directories,key=lambda v:(v.count('/'),v),reverse=True))+'\n')
with (OUT/'DELETE_ROOTS.tsv').open('w',newline='') as f:
 w=csv.writer(f,delimiter='\t');w.writerow(['path_relative_to_gpt_field','category','mode','files','allocated_GiB'])
 for r in candidates:w.writerow([r['path'],r['category'],r['mode'],r['inventory'].get('files',0),f"{r['inventory']['allocated_bytes']/2**30:.6f}"])
(OUT/'KEEP_PATHS.txt').write_text('\n'.join(f'{p}\t{why}' for p,why in sorted(protected.items()))+'\n\nInfrastructure/dependencies preserved:\n'+'\n'.join(other['protected_dependencies'])+'\nsrc/ scripts/ tests/ configs/ data/ .git/ .venv/ .private/ cleanup_plans/\nRoot documentation and historical cleanup receipts.\nAll paths outside /workspace/gpt_field.\n')
ancestors=set();pid=os.getpid()
while pid>1:
 ancestors.add(pid)
 try:pid=int(Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[1])
 except (OSError,ValueError):break
live=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name) in ancestors:continue
 try:
  args=proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0');args=[a for a in args if a]
  if not args:continue
  hits=[p for p in roots if any(str(ROOT/p) in a or p+'/' in a or a==p for a in args)]
  if hits:live.append({'pid':int(proc.name),'args':args,'candidate_paths':hits,'required_action':'Inspect then stop only this obsolete result service/monitor before approved deletion'})
 except (OSError,ValueError):continue
# Retain the parent shell of an old log monitor in the stop notice as well.
for row in list(live):
 if Path(row['args'][0]).name not in ('tail','grep','cut'):continue
 try:
  fields=Path(f"/proc/{row['pid']}/stat").read_text().split(') ',1)[1].split();parent=int(fields[1])
  pargs=[a for a in Path(f'/proc/{parent}/cmdline').read_bytes().decode(errors='replace').split('\0') if a]
  if parent not in ancestors and parent not in {v['pid'] for v in live} and any(p in a for p in row['candidate_paths'] for a in pargs):
   live.append({'pid':parent,'args':pargs,'candidate_paths':row['candidate_paths'],'required_action':'Stop the obsolete log-monitor wrapper and its own pipeline children after approval; recheck PID ownership first'})
 except (OSError,ValueError):pass
(OUT/'LIVE_PROCESS_DEPENDENCIES.json').write_text(json.dumps(live,indent=2)+'\n')
manifest={'created_utc':datetime.now(timezone.utc).isoformat(),'workspace':str(ROOT),'status':'PROPOSED_ONLY_AWAITING_USER_REVIEW','results_deleted':False,'processes_stopped':False,'scope':'Remove generated results outside UC7/UC7.1 and Astranomy-family direct control; keep live source/config/tests/docs/dependencies/credentials and protected result evidence','root_count':len(candidates),'counts':dict(entry_counts),'estimated_reclaimable_bytes':reclaimable,'estimated_reclaimable_gib':reclaimable/2**30,'allocated_bytes_note':'Estimate from st_blocks, deduplicated by device/inode; overlay/host accounting may differ.','group_totals':{k:dict(v) for k,v in groups.items()},'protected_result_roots':keep,'protected_validation_inputs':protected_files,'protected_dependencies':other['protected_dependencies'],'candidates':candidates,'live_process_dependencies':live,'partial_hardlinks':partial_hardlinks,'inventory_errors':errors,'execution_rules':['This notice does not execute deletion or stop services.','Preserve all paths outside this exact file/directory inventory.','Before deletion recheck live process ownership and drift; stop only the identified obsolete services/monitors after approval.','Do not follow symlinks; do not recursively remove new unlisted entries.','Preserve the active UC7.1 task0 editable UI and the UC7 manual-batch bridge.','Two tiny V3 approved plans remain in place solely as UC7.1 validation inputs.'],'inventory_sha256':hashlib.sha256(entries_path.read_bytes()).hexdigest()}
(OUT/'deletion_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
lines=['PROPOSED RESULT CLEANUP - NO DELETION PERFORMED','',f'Workspace: {ROOT}',f'Candidate roots/files: {len(candidates)}',f'Regular files: {entry_counts["files"]:,}',f'Directories: {entry_counts["directories"]:,}',f'Estimated recoverable allocation: {reclaimable/2**30:.2f} GiB','', 'CATEGORY TOTALS (GiB allocated):']
for group,s in groups.items():lines.append(f'{group}: {s["roots"]} roots, {s["files"]:,} files, {s["allocated_bytes"]/2**30:.2f} GiB')
lines+=['','EXACT LISTS','DELETE_ROOTS.tsv - each proposed top-level subtree/file, category and size','DELETE_FILES.txt - every proposed non-directory path (absolute)','DELETE_DIRECTORIES.txt - directories removable after listed entries','DELETE_ENTRIES.jsonl - lstat snapshot for drift/hardlink checks','KEEP_PATHS.txt - retained results, validation inputs and dependencies','deletion_manifest.json - full structured proposal','','PRESERVE','All verified SpeedUp v2 User Check v7/v7.1 official, relayout, failed, review and launch records.','Related v7 investigations, releases, PDF guides and official-suite export.','The active editable Task0 UI and UC7 manual-batch bridge.','Astranomy direct control: entire astranomy/ plus runs/astranomy_agentic_* family.','RoboDojo UC7.1 port, validation and runtime under robodojo_uc71/.','Project source, configs, tests, docs, environments, models, data, credentials and cleanup receipts.','Shared scene_test/astra_field_slides/vendor needed by retained PDF builders.','Two V3 approved_plan.json validation inputs named in KEEP_PATHS.txt.','','BEFORE DELETION','Obsolete V3 review services and older-version supervisors/monitors are still running.','Their exact PIDs/config paths are listed in LIVE_PROCESS_DEPENDENCIES.json.','Stopping those services is part of the proposed cleanup, after the user reviews this notice.','No active retained-method service will be stopped.','','NOTES','The term Astranomy direct control preserves both the earlier Responses-based and later persistent-Codex implementations.','Unrelated Direct5/history/demo and RoboDojo Direct result families are proposed for deletion.','Old combined comparison documents that contain UC7 evidence remain as mixed provenance.','Historical documentation may still mention removed experiments.','No files outside /workspace/gpt_field are in this proposal.','No command in build_notice.py deletes files, stops processes or executes experiments.']
(OUT/'NOTICE.txt').write_text('\n'.join(lines)+'\n')
print(json.dumps({k:manifest[k] for k in ['root_count','counts','estimated_reclaimable_gib','group_totals']},indent=2))
print('LIVE_MATCHES',len(live),'PARTIAL_HARDLINKS',len(partial_hardlinks))
