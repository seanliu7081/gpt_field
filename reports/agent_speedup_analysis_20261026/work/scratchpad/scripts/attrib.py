import json, os, sys, glob, re
from collections import defaultdict
S='/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis'
info=json.load(open(f'{S}/runinfo.json'))
out=[]
for run, meta in info.items():
    rq=f'/workspace/gpt_field/runs/{run}/requests'
    if not os.path.isdir(rq): continue
    for p in sorted(glob.glob(rq+'/*.response.json')):
        d=json.load(open(p))
        u=d.get('usage') or {}
        att=(u.get('attribution') or {})
        items=att.get('items') or {}
        rf=att.get('request_fields') or {}
        rec=dict(run=run, version=meta['version'], ordinal=int(os.path.basename(p)[:6]),
                 input_tokens=u.get('input_tokens'), cached=(u.get('input_tokens_details') or {}).get('cached_tokens'),
                 instr=(rf.get('instructions') or {}).get('input_tokens'), fmt=(rf.get('text.format') or {}).get('input_tokens'),
                 user_text=0, user_text_latest=0, image_tok=0, n_img_parts=0, placeholder_tok=0, label_tok=0,
                 reasoning_replay=0, assistant_replay=0, n_user=0, out_reason=0, out_visible=0)
        vals=list(items.values())
        # identify latest user msg: last input item with content len>=3 and output_tokens==0
        user_idx=[i for i,v in enumerate(vals) if v.get('output_tokens',0)==0 and isinstance(v.get('content'),list) and len(v['content'])>=3]
        latest=user_idx[-1] if user_idx else None
        for i,v in enumerate(vals):
            c=v.get('content')
            if v.get('output_tokens',0)>0:
                if c is None: rec['out_reason']+=v['output_tokens']
                else: rec['out_visible']+=v['output_tokens']
                continue
            if i in user_idx:
                rec['n_user']+=1
                parts=c
                # parts: [text, label, image_or_placeholder, label, image_or_placeholder]
                t=parts[0].get('input_tokens',0)
                rec['user_text']+=t
                if i==latest: rec['user_text_latest']+=t
                for j,pp in enumerate(parts[1:],1):
                    tok=pp.get('input_tokens',0)
                    if j in (2,4):
                        if tok>100: rec['image_tok']+=tok; rec['n_img_parts']+=1
                        else: rec['placeholder_tok']+=tok
                    else: rec['label_tok']+=tok
            elif c is None:
                rec['reasoning_replay']+=v.get('input_tokens',0)
            else:
                rec['assistant_replay']+=v.get('input_tokens',0)
        out.append(rec)
with open(f'{S}/attrib.jsonl','w') as f:
    for r in out: f.write(json.dumps(r)+'\n')
print(len(out))
