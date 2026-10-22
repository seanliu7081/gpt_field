"""Package and upload only the ten completed manual-batch run directories."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path('/workspace/gpt_field')
BATCH = ROOT / 'runs/agent_speedup_v2_user_check_libero_10_manual_batch_001/queue_manifest.json'
EXPORT = ROOT / 'exports/gpt_field_libero10_20261022'
STAGE = EXPORT / 'dataset'
REPO = 'SeanLiu0272/gpt_field_libero10'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def digest(path):
    size = path.stat().st_size
    sha256 = hashlib.sha256()
    blob = hashlib.sha1(f'blob {size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            sha256.update(block)
            blob.update(block)
    return {'size': size, 'sha256': sha256.hexdigest(), 'git_blob_sha1': blob.hexdigest()}


def prepare():
    if STAGE.exists():
        raise RuntimeError('Staging directory already exists; refusing to replace it')
    STAGE.mkdir(parents=True)
    batch = json.loads(BATCH.read_text())
    rows, inventory = [], []
    for task in batch['tasks']:
        source = Path(task['output'])
        if source.parent != ROOT / 'runs' or not source.is_dir():
            raise RuntimeError('Unexpected source directory')
        summary = json.loads((source / 'summary.json').read_text())
        if summary['status'] not in ('success', 'user_stop'):
            raise RuntimeError('Source run is not terminal')
        rows.append({
            'task': task['task'],
            'run_directory': source.name,
            'libero_success': bool(summary['libero_success']),
            'time_without_video_s': summary['control_without_video_s'],
            'model_calls': summary['api_calls'],
        })
        files = sorted(p for p in source.rglob('*') if p.is_file())
        for original in files:
            if original.is_symlink() or any(p.is_symlink() for p in original.parents if p != ROOT):
                raise RuntimeError('Symlinks require explicit inspection')
            relative = Path(source.name) / original.relative_to(source)
            target = STAGE / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(original, target)
            inventory.append({'path': relative.as_posix(), **digest(original)})
        print(json.dumps({'staged_run': source.name, 'files': len(files)}), flush=True)
    with (STAGE / 'results.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    passed = [r for r in rows if r['libero_success']]
    result = {
        'method': batch['version'], 'model': batch['model'],
        'reasoning_effort': batch['reasoning_effort'],
        'episode_index': batch['episode_index'], 'seed': batch['seed'],
        'success_criterion': 'raw LIBERO success',
        'model_calls_scope': 'Execution-stage calls from summary.json api_calls; initial plan generation is excluded.',
        'time_scope': 'Execution control time minus video rendering; excludes initial plan generation and manual review waiting.',
        'successful_tasks': len(passed), 'attempted_tasks': len(rows),
        'total_time_without_video_s': sum(r['time_without_video_s'] for r in rows),
        'successful_tasks_mean_time_without_video_s': sum(r['time_without_video_s'] for r in passed) / len(passed),
        'total_model_calls': sum(r['model_calls'] for r in rows),
        'note': 't06 was stopped by the user; its elapsed time is an attempt duration, not completion latency.',
        'tasks': rows,
    }
    write_json(STAGE / 'results.json', result)
    write_json(STAGE / 'upload_manifest.json', {
        'repository': REPO, 'repo_type': 'dataset',
        'source_batch': BATCH.parent.name,
        'scope': 'Exactly the ten final run output directories named in the batch manifest. Review-server, launch, earlier-run and smoke-test directories are excluded.',
        'source_files_modified': False,
        'run_file_count': len(inventory),
        'run_total_bytes': sum(x['size'] for x in inventory),
        'files': inventory,
    })
    table = '\n'.join(
        f"| {r['task']} | {'Yes' if r['libero_success'] else 'No (user stopped)'} | {r['time_without_video_s']:.3f} | {r['model_calls']} |"
        for r in rows
    )
    card = '''---
language:
- en
tags:
- robotics
- libero
- evaluation
pretty_name: SpeedUp v2 User Check - LIBERO-10
configs:
- config_name: results
  data_files:
  - split: test
    path: results.csv
---

# SpeedUp v2 User Check: LIBERO-10

One manually reviewed episode per task, t00-t09, from the completed manual batch.
Method: `agent-speedup-v2-user-check-7`; model: `gpt-6-astra`; reasoning effort: `high`;
episode index: `0`; seed: `7`.

Each `agent_speedup_v2_user_check_libero_10_*` directory preserves its original
run artifacts, including video, traces, model request logs, approved plan,
evaluation records, and frozen source snapshots. Historical retries, review
servers, launch directories, and smoke tests are not included.

## Requested metrics

Success means the raw LIBERO success flag recorded in `summary.json`.
Time is `control_without_video_s`; model calls are execution-stage `api_calls`.
Initial plan generation and manual approval waiting are excluded.
The t06 run was manually stopped; its time is an incomplete attempt duration.

| Task | LIBERO success | Time without video (seconds) | Model calls |
|---|---|---:|---:|
''' + table + f'''

Official successes: **{len(passed)}/{len(rows)}**. Mean time among successful tasks:
**{result['successful_tasks_mean_time_without_video_s']:.3f} seconds**.
Total execution-stage model calls: **{result['total_model_calls']}**.

The retained physical diagnostics use additional criteria and are distinct from
the raw LIBERO result reported above. These are single-episode records, not a
multi-seed benchmark.

`results.csv` and `results.json` contain the metric index.
`upload_manifest.json` records file sizes and SHA-256 hashes for all original run
files. Original absolute paths inside saved artifacts are provenance records.
'''
    (STAGE / 'README.md').write_text(card)
    print(json.dumps({'prepared': str(STAGE), 'run_files': len(inventory),
                      'bytes': sum(x['size'] for x in inventory),
                      'libero_successes': len(passed), 'model_calls': result['total_model_calls']}), flush=True)


def upload():
    from huggingface_hub import HfApi
    api = HfApi()
    if api.whoami()['name'] != REPO.split('/')[0]:
        raise RuntimeError('Hugging Face account changed')
    if not (STAGE / 'upload_manifest.json').is_file():
        raise RuntimeError('Prepare the upload first')
    marker = EXPORT / 'repository_created.json'
    if not marker.exists():
        url = api.create_repo(REPO, repo_type='dataset', private=True, exist_ok=False)
        write_json(marker, {'repo_id': REPO, 'repo_type': 'dataset', 'private': True, 'url': str(url)})
        print(json.dumps({'repository_created': str(url), 'private': True}), flush=True)
    info = api.repo_info(REPO, repo_type='dataset')
    if not info.private:
        raise RuntimeError('Expected a private repository')
    api.upload_large_folder(
        repo_id=REPO, repo_type='dataset', folder_path=STAGE,
        private=True, num_workers=8, print_report=True, print_report_every=30,
    )
    print('UPLOAD_COMPLETE', flush=True)


def verify():
    from huggingface_hub import HfApi
    from huggingface_hub.hf_api import RepoFile
    api = HfApi()
    info = api.repo_info(REPO, repo_type='dataset')
    revision = info.sha
    manifest = json.loads((STAGE / 'upload_manifest.json').read_text())
    expected = {entry['path']: entry for entry in manifest['files']}
    for name in ('README.md', 'results.csv', 'results.json', 'upload_manifest.json'):
        expected[name] = {'path': name, **digest(STAGE / name)}
    remote = {entry.path: entry for entry in api.list_repo_tree(
        REPO, repo_type='dataset', revision=revision, recursive=True
    ) if isinstance(entry, RepoFile)}
    missing = sorted(set(expected) - set(remote))
    unexpected = sorted(set(remote) - set(expected) - {'.gitattributes'})
    mismatches = []
    for name in sorted(set(expected) & set(remote)):
        local, saved = expected[name], remote[name]
        if saved.size != local['size']:
            mismatches.append({'path': name, 'issue': 'size'})
        if saved.lfs:
            if saved.lfs.sha256 != local['sha256']:
                mismatches.append({'path': name, 'issue': 'sha256'})
        elif saved.blob_id != local['git_blob_sha1']:
            mismatches.append({'path': name, 'issue': 'git_blob_sha1'})
    result = {
        'repo_id': REPO, 'url': f'https://huggingface.co/datasets/{REPO}',
        'private': info.private, 'revision': revision,
        'expected_file_count': len(expected), 'remote_file_count': len(remote),
        'run_file_count': manifest['run_file_count'], 'run_total_bytes': manifest['run_total_bytes'],
        'missing': missing, 'unexpected': unexpected, 'mismatches': mismatches,
        'verified': not missing and not unexpected and not mismatches and info.private,
    }
    write_json(EXPORT / 'upload_verification.json', result)
    print(json.dumps(result), flush=True)
    if not result['verified']:
        raise RuntimeError('Remote files did not match the upload inventory')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'upload', 'verify'))
    args = parser.parse_args()
    {'prepare': prepare, 'upload': upload, 'verify': verify}[args.action]()
