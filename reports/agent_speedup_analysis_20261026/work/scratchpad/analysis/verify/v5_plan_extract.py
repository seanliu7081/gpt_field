import os, re, sys, json, glob
root = '/workspace/gpt_field'
modes = json.load(open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v5_modes.json'))
plan = sorted(p for p, m in modes if m in ('plan', 'replan'))

def ws_stats(text):
    inside = False; esc = False; run = 0; longest = 0; ws_out = 0; ws_all = 0
    for ch in text:
        if ch in ' \t\r\n':
            ws_all += 1
        if inside:
            run = 0
            if esc: esc = False
            elif ch == '\\': esc = True
            elif ch == '"': inside = False
        elif ch in ' \t\r\n':
            run += 1; ws_out += 1; longest = max(longest, run)
        else:
            run = 0
            if ch == '"': inside = True
    return dict(chars=len(text), ws_all=ws_all, ws_outside=ws_out, longest_run=longest)

rows = []
for rel in plan:
    p = os.path.join(root, rel)
    req = json.load(open(p))
    base = p[:-len('.request.json')]
    res = json.load(open(base + '.result.json')) if os.path.exists(base + '.result.json') else None
    resp = json.load(open(base + '.response.json')) if os.path.exists(base + '.response.json') else None
    retry = json.load(open(base + '.request.transient_retry.json')) if os.path.exists(base + '.request.transient_retry.json') else None
    payload = req.get('payload', {})
    pdump = json.dumps(payload, ensure_ascii=False)
    instr = payload.get('instructions') or ''
    row = dict(path=rel, dir=os.path.dirname(os.path.dirname(rel)), mode=req.get('mode'),
               kind=(req.get('request_meta') or {}).get('kind'),
               prompt_version=req.get('prompt_version'),
               v2_client=req.get('agent_speedup_v2_client_version'),
               v3_client=req.get('agent_speedup_v3_client_version'),
               wire=req.get('wire_transport_version'),
               timeout_s=req.get('timeout_s'),
               before=req.get('budget_elapsed_s_before'),
               attempt=req.get('attempt'),
               ordinal=req.get('request_ordinal'),
               effort=(payload.get('reasoning') or {}).get('effort'),
               fmt=((payload.get('text') or {}).get('format') or {}).get('name'),
               compact_in_payload=('OUTPUT SERIALIZATION' in pdump),
               compact_in_instructions=('OUTPUT SERIALIZATION' in instr),
               compact_full_text=None,
               max_output_tokens_in_payload=('max_output_tokens' in payload),
               payload_keys=list(payload.keys()),
               instr_len=len(instr), instr_head=instr[:60])
    if res is not None:
        b = res.get('budget') or {}
        row.update(status=res.get('status'), error_type=res.get('error_type'), error_code=res.get('error_code'),
                   http_status=res.get('http_status'), after=b.get('elapsed_s'),
                   output_token_limit_sent=b.get('output_token_limit_sent'),
                   v3diag=res.get('v3_transport_diagnostics'))
    else:
        row.update(status=None)
    if row.get('after') is not None and row.get('before') is not None:
        row['client_interval_s'] = row['after'] - row['before']
    if retry is not None:
        row['retry'] = dict(status=retry.get('status'), error_code=retry.get('error_code'), wait_s=retry.get('wait_s'))
    if resp is not None:
        row['server_s'] = (resp.get('completed_at') or 0) - (resp.get('created_at') or 0)
        row['resp_max_output_tokens'] = resp.get('max_output_tokens')
        u = resp.get('usage') or {}
        row['usage'] = dict(inp=u.get('input_tokens'), out=u.get('output_tokens'),
                            reas=(u.get('output_tokens_details') or {}).get('reasoning_tokens'))
        texts = []
        for item in resp.get('output') or []:
            if item.get('type') == 'message':
                for part in item.get('content') or []:
                    if part.get('type') == 'output_text':
                        texts.append(part.get('text') or '')
        t = ''.join(texts)
        row['out_ws'] = ws_stats(t)
        row['out_head'] = t[:30]
        try:
            row['compact_len'] = len(json.dumps(json.loads(t), ensure_ascii=False, separators=(',', ':')))
        except Exception as e:
            row['compact_len'] = 'ERR ' + str(e)[:60]
    rows.append(row)

json.dump(rows, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/analysis/verify/v5_plan_rows.json', 'w'), indent=1)
for r in rows:
    print(r['path'][:110])
    print('   ', r['mode'], r['kind'], r['prompt_version'], r['v2_client'], r['v3_client'], r['wire'], 'eff', r['effort'], 'compact', r['compact_in_payload'], r['compact_in_instructions'], 'mot', r['max_output_tokens_in_payload'], 'to', r['timeout_s'])
    print('    status', r.get('status'), r.get('error_type'), r.get('error_code'), 'interval', None if r.get('client_interval_s') is None else round(r['client_interval_s'], 2), 'server', r.get('server_s'), 'usage', r.get('usage'), 'ws', r.get('out_ws'), 'compact_len', r.get('compact_len'), 'retry', r.get('retry'))
