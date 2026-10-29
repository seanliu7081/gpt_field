import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))
from exec_types import decisions
pat = re.compile(r"[^.]*wrist[^.]*\b(left|right)\b[^.]*\.|[^.]*\b(left|right)\b[^.]*wrist[^.]*\.", re.I)
for r in sys.argv[1:]:
    hits = []
    for name, j in decisions(os.path.join(r, 'requests')):
        for key in ('belief', 'plan', 'why'):
            for m in pat.finditer(j.get(key, '')):
                hits.append((name, key, m.group(0).strip()[:300]))
    print('==', r.split('runs/')[-1], len(hits))
    for h in hits[:6]:
        print('   ', h)
