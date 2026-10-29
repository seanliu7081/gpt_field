import json, glob, os, sys, re
sys.path.insert(0, os.path.dirname(__file__))
from exec_types import decisions
pat = re.compile(r"[^.]*\b(right|left|\+y|-y|\+x|-x|world y|world x|image-right|image right)\b[^.]*\.", re.I)
r = sys.argv[1]
lim = int(sys.argv[2]) if len(sys.argv) > 2 else 40
n = 0
for name, j in decisions(os.path.join(r, 'requests')):
    hits = []
    for key in ('belief', 'plan', 'why'):
        for m in pat.finditer(j.get(key, '')):
            hits.append(f"{key}: {m.group(0).strip()}")
    if hits:
        print('--', name, j['type'])
        for h in hits[:4]:
            print('   ', h[:400])
        n += 1
        if n >= lim:
            break
