import os, re, sys, json
root = '/workspace/gpt_field'
out = []
pat_mode = re.compile(rb'"mode"\s*:\s*"([^"]+)"')
pat_kind = re.compile(rb'"request_meta"\s*:\s*\{\s*"kind"\s*:\s*"([^"]+)"')
for base in ('runs', 'results'):
    for dp, dn, fn in os.walk(os.path.join(root, base)):
        if 'frozen_source' in dp or 'frozen_runtime' in dp:
            continue
        for f in fn:
            if not f.endswith('.request.json'):
                continue
            p = os.path.join(dp, f)
            with open(p, 'rb') as fh:
                head = fh.read(4000)
            m = pat_mode.search(head)
            mode = m.group(1).decode() if m else None
            out.append((p[len(root)+1:], mode))
from collections import Counter
c = Counter(m for _, m in out)
print(c)
plan = [p for p, m in out if m in ('plan', 'replan')]
print(len(plan))
dirs = Counter(os.path.dirname(p) for p in plan)
for d, n in sorted(dirs.items()):
    print(n, d)
json.dump(out, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v5_modes.json', 'w'))
