"""Reconstruct immutable v7 evidence into independent inodes; no credentials or sim."""
import argparse, hashlib, json, pathlib, shutil

def reconstruct(repo, destination):
    archive = repo/'runs/agent_speedup_v2_user_check_libero_10_t1_001'
    record = json.loads((archive/'user_check_review.json').read_text())
    destination.mkdir(exist_ok=False)
    rows=[]
    for name, expected in record['runtime_sha256'].items():
        if name.startswith('@'): continue
        source=archive/'frozen_source'/pathlib.Path(name).name
        actual=hashlib.sha256(source.read_bytes()).hexdigest()
        if actual!=expected: raise RuntimeError(f'Archive mismatch: {name}')
        target=destination/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(source,target)
        assert target.stat().st_ino!=source.stat().st_ino
        rows.append({'path':name,'sha256':actual,'source':str(source),'source_nlink':source.stat().st_nlink,'independent_inode':True})
    for name in ['agent_speedup_v3_video.py','agent_speedup_v3_performance.py']:
        source=repo/'src/gpt_field'/name; target=destination/'src/gpt_field'/name
        shutil.copyfile(source,target)
        rows.append({'path':str(target.relative_to(destination)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'source':str(source),'role':'reuse_IO_only'})
    (destination/'reconstruction.json').write_text(json.dumps({'historical_archive':str(archive),'historical_files':len(record['runtime_sha256'])-1,'source_rows':rows,'identity':'reconstructed_baseline; frozen v7 control plus explicit evaluation/IO adapters'},indent=2)+'\n')
    print(json.dumps({'destination':str(destination),'files':len(rows)}))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--repository',type=pathlib.Path,required=True);p.add_argument('--destination',type=pathlib.Path,required=True);a=p.parse_args();reconstruct(a.repository,a.destination)
