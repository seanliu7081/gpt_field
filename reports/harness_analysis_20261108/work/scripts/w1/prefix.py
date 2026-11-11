import json, glob, sys, os, hashlib
run = sys.argv[1]
req = os.path.join(run, 'trial', 'requests')
files = sorted(glob.glob(os.path.join(req, '*.request.json')))
prev = None
def h(x): return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()[:10]
for f in files:
    d = json.load(open(f))
    p = d.get('payload', {})
    inp = p.get('input', [])
    instr = p.get('instructions', '')
    nimg = sum(1 for it in inp for c in (it.get('content') or []) if isinstance(c, dict) and c.get('type') == 'input_image')
    hs = [h(it) for it in inp]
    # bytes of text
    txt = sum(len(c.get('text','')) for it in inp for c in (it.get('content') or []) if isinstance(c, dict) and c.get('type') in ('input_text','output_text'))
    enc = sum(len(it.get('encrypted_content') or '') for it in inp)
    if prev is not None:
        same_instr = (prev[0] == h(instr))
        n = 0
        for a, b in zip(prev[1], hs):
            if a == b: n += 1
            else: break
        print(os.path.basename(f)[:6], 'items', len(inp), 'imgs', nimg, 'text_chars', txt, 'enc_chars', enc, 'instr_same', same_instr, 'common_prefix_items', n, 'of prev', len(prev[1]), 'kind', d.get('mode'))
    else:
        print(os.path.basename(f)[:6], 'items', len(inp), 'imgs', nimg, 'text_chars', txt, 'enc_chars', enc, 'kind', d.get('mode'))
    prev = (h(instr), hs)
