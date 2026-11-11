import json, sys, glob, os
from pathlib import Path

def stats(run):
    reqdir = Path(run) / 'trial' / 'requests'
    rows = []
    for req in sorted(reqdir.glob('*.request.json')):
        stem = req.name[:-len('.request.json')]
        d = json.load(open(req))
        res = json.load(open(reqdir / f'{stem}.result.json')) if (reqdir / f'{stem}.result.json').exists() else {}
        st = json.load(open(reqdir / f'{stem}.stream.json')) if (reqdir / f'{stem}.stream.json').exists() else {}
        p = d['payload']
        inp = p['input']
        nmsg = len(inp)
        nimg = 0; text_chars = 0; img_chars = 0; fv_chars = 0; ctx_chars = 0; first_text = None
        last_text_chars = 0
        for m in inp:
            content = m.get('content')
            if isinstance(content, str):
                text_chars += len(content); continue
            for c in content or []:
                if c.get('type') in ('input_image',):
                    nimg += 1; img_chars += len(c.get('image_url', ''))
                elif c.get('type') in ('input_text', 'output_text'):
                    t = c.get('text', '')
                    text_chars += len(t)
        # last user message primary text
        last = inp[-1]
        lt = None
        for c in last['content']:
            if c.get('type') == 'input_text' and c['text'].startswith('{'):
                lt = c['text']; break
        if lt:
            last_text_chars = len(lt)
            try:
                obj = json.loads(lt)
                ctx = obj.get('context', {})
                ctx_chars = len(json.dumps(ctx, ensure_ascii=False, separators=(',',':')))
                fv = ctx.get('foundation_vision')
                if fv is not None:
                    fv_chars = len(json.dumps(fv, ensure_ascii=False, separators=(',',':')))
            except Exception as e:
                pass
        b = res.get('budget', {})
        rows.append(dict(
            file=stem[:6], kind=d.get('request_meta', {}).get('kind'), step=d.get('request_meta', {}).get('step'),
            msgs=nmsg, imgs=nimg, instr=len(p.get('instructions', '')), text=text_chars, last_text=last_text_chars,
            ctx=ctx_chars, fv=fv_chars,
            in_tok=b.get('input_tokens'), cached=b.get('cached_input_tokens'), out_tok=b.get('output_tokens'), reas=b.get('reasoning_tokens'),
            elapsed=b.get('elapsed_s'), hdr=st.get('headers_received_s'), first_vis=st.get('first_visible_output_s'), done=st.get('completion_s'),
            vis_chars=st.get('visible_characters'), status=res.get('status')))
    return rows

if __name__ == '__main__':
    for run in sys.argv[1:]:
        rows = stats(run)
        print('RUN', run)
        keys = ['file','kind','step','msgs','imgs','instr','text','last_text','ctx','fv','in_tok','cached','out_tok','reas','elapsed','hdr','first_vis','done','vis_chars','status']
        print('\t'.join(keys))
        for r in rows:
            print('\t'.join(str(round(r[k],1) if isinstance(r[k], float) else r[k]) for k in keys))
        import statistics as S
        def tot(k): return sum(r[k] or 0 for r in rows)
        print('N', len(rows), 'sum_elapsed', round(tot('elapsed'),1), 'sum_in', tot('in_tok'), 'sum_out', tot('out_tok'), 'sum_reas', tot('reas'), 'sum_cached', tot('cached'),
              'mean_elapsed', round(tot('elapsed')/len(rows),2), 'median_elapsed', round(S.median([r['elapsed'] for r in rows]),2),
              'sum_first_vis', round(sum(r['first_vis'] or 0 for r in rows),1), 'sum_hdr', round(sum(r['hdr'] or 0 for r in rows),1),
              'sum_gen(done-first_vis)', round(sum((r['done'] or 0)-(r['first_vis'] or 0) for r in rows),1), 'sum_vis_chars', tot('vis_chars'))
        print()
