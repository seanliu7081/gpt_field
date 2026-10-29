#!/usr/bin/env python3
"""Read-only: tabulate evidence_acquisition proposals in every request of given runs.
Usage: python3 -I evidence_acq.py <run_dir> [...]
"""
import collections
import glob
import json
import sys


def find_ea(j):
    cd = j.get('current_decision') or {}
    if isinstance(cd, dict) and 'evidence_acquisition' in cd:
        return cd.get('evidence_acquisition')
    ne = (j.get('context') or {}).get('native_evidence') or {}
    if isinstance(ne, dict):
        s = ne.get('summary') if isinstance(ne.get('summary'), dict) else ne
        if isinstance(s, dict):
            cd2 = s.get('current_decision')
            if isinstance(cd2, dict):
                return cd2.get('evidence_acquisition')
    return None


def main():
    for run in sys.argv[1:]:
        fs = sorted(glob.glob(run + '/requests/*.request.json'))
        reasons = collections.Counter()
        rej = collections.Counter()
        ncand = []
        rows = []
        for i, f in enumerate(fs):
            d = json.load(open(f))
            t = d['payload']['input'][-1]['content'][0]['text']
            j = json.loads(t)
            ea = find_ea(j)
            if ea is None:
                reasons['<absent>'] += 1
                rows.append((i + 1, '<absent>', None, None))
                continue
            if not isinstance(ea, dict):
                reasons[str(ea)[:60]] += 1
                continue
            r = ea.get('reason')
            reasons[r] += 1
            c = ea.get('candidates')
            nc = len(c) if isinstance(c, list) else 0
            ncand.append(nc)
            rj = ea.get('rejected')
            rsum = None
            if isinstance(rj, list):
                cc = collections.Counter()
                for x in rj:
                    if isinstance(x, dict):
                        cc[x.get('reason') or x.get('status') or '?'] += 1
                    else:
                        cc[str(x)[:40]] += 1
                rsum = dict(cc)
                for k, v in cc.items():
                    rej[k] += v
            elif isinstance(rj, dict):
                rsum = {k: (len(v) if isinstance(v, list) else v) for k, v in rj.items()}
                for k, v in rsum.items():
                    rej[k] += v if isinstance(v, int) else 1
            rows.append((i + 1, r, nc, rsum))
        print('=====', run, 'requests', len(fs))
        print('  reasons:', dict(reasons))
        print('  rejected reasons (summed):', dict(rej))
        for row in rows:
            print('   ', row)


if __name__ == '__main__':
    main()
