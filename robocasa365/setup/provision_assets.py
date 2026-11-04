"""Download official simulator assets; never create a simulation or load demos."""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = ROOT / 'vendor/robocasa/robocasa/models/assets'
DOWNLOADS = ROOT / 'assets/downloads'
DOWNLOADS.mkdir(parents=True, exist_ok=True)
CHECKS = json.loads((ROOT / 'setup/asset_get_checks.json').read_text())
SELECTED = {'textures': ASSET_ROOT, 'generative_textures': ASSET_ROOT,
            'fixtures_lightwheel': ASSET_ROOT, 'objects_lightwheel': ASSET_ROOT / 'objects',
            'objaverse': ASSET_ROOT / 'objects'}

def provision(item):
    name, destination = item
    info = CHECKS[name]
    size = int(info['content_range'].rsplit('/', 1)[-1])
    archive = DOWNLOADS / (name + '.zip')
    started = time.time()
    if not archive.exists():
        temporary = archive.with_suffix('.zip.partial')
        # Explicit complete range also avoids the server's unsupported HEAD route.
        req = urllib.request.Request(info['url'], headers={'Range': f'bytes=0-{size-1}'})
        with urllib.request.urlopen(req, timeout=60) as response, temporary.open('wb') as output:
            if response.status not in (200, 206):
                raise RuntimeError(f'{name}: unexpected HTTP {response.status}')
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size != size:
            raise RuntimeError(f'{name}: size mismatch {temporary.stat().st_size} != {size}')
        temporary.rename(archive)
    if archive.stat().st_size != size:
        raise RuntimeError(f'{name}: existing archive size mismatch')
    digest = hashlib.file_digest(archive.open('rb'), 'sha256').hexdigest()
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f'{name}: ZIP CRC error at {bad}')
        for member in zf.infolist():
            path = Path(member.filename)
            if path.is_absolute() or '..' in path.parts or not (destination / path).resolve().is_relative_to(destination.resolve()):
                raise RuntimeError(f'{name}: unsafe ZIP member')
        destination.mkdir(parents=True, exist_ok=True)
        zf.extractall(destination)
        result = dict(name=name, url=info['url'], bytes=size, sha256=digest,
                      extracted_file_count=sum(not x.is_dir() for x in zf.infolist()),
                      uncompressed_bytes=sum(x.file_size for x in zf.infolist()),
                      top_level_entries=sorted({Path(x.filename).parts[0] for x in zf.infolist()}),
                      archive=str(archive.relative_to(ROOT)), destination=str(destination.relative_to(ROOT)),
                      elapsed_s=time.time()-started, status='downloaded_crc_verified_extracted')
    (ROOT / 'setup' / f'asset_{name}.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
    return result

if __name__ == '__main__':
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for future in concurrent.futures.as_completed([pool.submit(provision, x) for x in SELECTED.items()]):
            results.append(future.result())
    (ROOT / 'setup/assets_manifest.json').write_text(json.dumps({'assets': results,
        'omitted': {'aigen_objs': 'not in the official default object registries; no download required'},
        'simulator_instances_created': 0, 'model_calls': 0}, indent=2)+'\n')
