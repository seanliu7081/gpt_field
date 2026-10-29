import json, sys, os, glob, collections, re
pat = re.compile(r'local_pose_revision\\*"\s*:\s*(\d+)')
pat2 = re.compile(r'"(local_pose[a-z_]*)"')
for d in sys.argv[1:]:
    vals = collections.Counter()
    for f in glob.glob(os.path.join(d, 'requests', '*.request.json')):
        txt = open(f).read()
        for m in pat.finditer(txt): vals[int(m.group(1))] += 1
    tvals = collections.Counter(); tkeys = collections.Counter()
    with open(os.path.join(d, 'trace.jsonl')) as fh:
        for line in fh:
            if 'local_pose' not in line: continue
            for m in pat.finditer(line): tvals[int(m.group(1))] += 1
            for m in pat2.finditer(line): tkeys[m.group(1)] += 1
    print(os.path.basename(d), 'requests lpr values', dict(vals), '| trace lpr values', dict(tvals), '| trace local_pose* keys', dict(tkeys.most_common(8)))
