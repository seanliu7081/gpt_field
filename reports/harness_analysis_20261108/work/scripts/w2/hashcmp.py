"""Independent hash comparison of H1 policy modules vs every frozen source copy on disk."""
import hashlib, json, os, sys, glob
H1 = '/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm'
MAIN = '/workspace/gpt_field'
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        h.update(f.read())
    return h.hexdigest()
h1 = {}
for p in glob.glob(H1 + '/src/gpt_field/*.py'):
    h1['src/gpt_field/' + os.path.basename(p)] = sha(p)
for p in glob.glob(H1 + '/scripts/*.py'):
    h1['scripts/' + os.path.basename(p)] = sha(p)
POLICY = sorted(k for k in h1 if k.startswith('src/gpt_field/') and (
    'user_check' in k or os.path.basename(k) in (
        'action.py', 'agent_speedup_v2_client.py', 'astranomy_agentic_geometry.py', 'astranomy_agentic_motion.py',
        'contact_skills.py', 'observation.py', 'ordered_transport.py', 'chatgpt_auth.py', 'chatgpt_plan.py',
        'agent_speedup_v2_tasks.py', 'libero.py', 'agent_speedup_v2_sim.py', 'agent_speedup_v2_types.py')))
POLICY += ['scripts/evaluate_agent_speedup_v2.py', 'scripts/evaluate_agent_speedup_v2_user_check.py', 'scripts/launch.py']
def locate(root, rel):
    # root may be source_at_launch (with src/ and scripts/) or frozen_source (flat or nested)
    cands = [os.path.join(root, rel), os.path.join(root, os.path.basename(rel))]
    if rel.startswith('scripts/'):
        cands.append(os.path.join(root, 'scripts', os.path.basename(rel)))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None
roots = []
for pat in ['/workspace/gpt_field/runs/*/source_at_launch', '/workspace/gpt_field/runs/*/trial/frozen_source',
            '/workspace/gpt_field/runs/*/review/frozen_source', '/workspace/gpt_field/results/*/evaluations/*/trial/frozen_source',
            '/workspace/gpt_field/reports/uc7_direction_semantics_20261028/implementation_uc71/v1_planning/*/frozen_source']:
    roots += sorted(glob.glob(pat))
roots.append(MAIN)
out = {}
for r in roots:
    res = {}
    for rel in POLICY:
        p = locate(r, rel)
        if p is None:
            res[rel] = 'absent'
        else:
            res[rel] = 'same' if sha(p) == h1[rel] else 'DIFF'
    out[r] = res
json.dump({'policy_files': POLICY, 'h1_hashes': {k: h1[k] for k in POLICY}, 'cmp': out}, open(sys.argv[1], 'w'), indent=1)
for r, res in out.items():
    diffs = [k for k, v in res.items() if v == 'DIFF']
    absent = [k for k, v in res.items() if v == 'absent']
    print(r.replace('/workspace/gpt_field/', ''))
    print('   DIFF:', [os.path.basename(d) for d in diffs])
    print('   absent:', len(absent), [os.path.basename(a) for a in absent][:8])
