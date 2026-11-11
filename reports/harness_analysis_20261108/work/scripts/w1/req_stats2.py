import json, sys
from pathlib import Path
import statistics as S

def per_call(run):
    reqdir = Path(run) / 'trial' / 'requests'
    rows = []
    prev = dict(in_tok=0, out_tok=0, reas=0, cached=0, elapsed=0.0)
    for req in sorted(reqdir.glob('*.request.json')):
        stem = req.name[:-len('.request.json')]
        d = json.load(open(req))
        res = json.load(open(reqdir / f'{stem}.result.json')) if (reqdir / f'{stem}.result.json').exists() else {}
        st = json.load(open(reqdir / f'{stem}.stream.json')) if (reqdir / f'{stem}.stream.json').exists() else {}
        p = d['payload']; inp = p['input']
        nimg = 0; text_chars = 0; fv_hist = 0
        for m in inp:
            content = m.get('content')
            if isinstance(content, str):
                text_chars += len(content); continue
            for c in content or []:
                if c.get('type') == 'input_image': nimg += 1
                elif 'text' in c:
                    t = c['text']; text_chars += len(t)
                    if '"foundation_vision"' in t:
                        try:
                            o = json.loads(t); fv = o.get('context', {}).get('foundation_vision')
                            if fv is not None: fv_hist += len(json.dumps(fv, ensure_ascii=False, separators=(',',':')))
                        except Exception:
                            pass
        b = res.get('budget', {})
        cur = dict(in_tok=b.get('input_tokens', 0), out_tok=b.get('output_tokens', 0), reas=b.get('reasoning_tokens', 0), cached=b.get('cached_input_tokens', 0), elapsed=b.get('elapsed_s', 0.0))
        delta = {k: cur[k] - prev[k] for k in cur}
        prev = cur
        rows.append(dict(f=stem[:6], step=d.get('request_meta', {}).get('step'), msgs=len(inp), imgs=nimg,
                         instr=len(p.get('instructions', '')), text=text_chars, fv_hist=fv_hist,
                         in_tok=delta['in_tok'], cached=delta['cached'], out=delta['out_tok'], reas=delta['reas'],
                         budget_dt=round(delta['elapsed'], 1), hdr=st.get('headers_received_s'), ttfv=st.get('first_visible_output_s'),
                         done=st.get('completion_s'), vis=st.get('visible_characters')))
    return rows

def show(run):
    rows = per_call(run)
    keys = list(rows[0].keys())
    print('RUN', run)
    print('\t'.join(keys))
    for r in rows:
        print('\t'.join(str(round(r[k], 1) if isinstance(r[k], float) else r[k]) for k in keys))
    def s(k): return sum((r[k] or 0) for r in rows)
    print(f"N={len(rows)} sum_done={s('done'):.1f} mean_done={s('done')/len(rows):.1f} median_done={S.median([r['done'] for r in rows]):.1f} "
          f"sum_ttfv={s('ttfv'):.1f} sum_gen={s('done')-s('ttfv'):.1f} sum_hdr={s('hdr'):.1f} sum_in={s('in_tok')} sum_cached={s('cached')} sum_out={s('out')} sum_reas={s('reas')} "
          f"mean_in={s('in_tok')/len(rows):.0f} mean_reas={s('reas')/len(rows):.0f} mean_out={s('out')/len(rows):.0f} sum_budget_dt={s('budget_dt'):.1f}")
    print()
    return rows

if __name__ == '__main__':
    for run in sys.argv[1:]:
        show(run)
