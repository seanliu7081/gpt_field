import json, os, sys, glob, re

def load(p):
    with open(p, 'rb') as f:
        return json.loads(f.read().decode('utf-8'))

def req_features(p):
    d = load(p)
    pay = d.get('payload') or {}
    inp = pay.get('input') or []
    n_user = n_reason = n_asst = n_img = n_placeholder = 0
    img_b64 = 0; text_chars = 0; last_text_chars = 0; reason_enc = 0; asst_chars = 0
    last_user_idx = max([i for i, it in enumerate(inp) if it.get('role') == 'user'] or [-1])
    img_detail = set()
    for i, it in enumerate(inp):
        if it.get('role') == 'user':
            n_user += 1
            for c in it.get('content') or []:
                if c.get('type') == 'input_image':
                    n_img += 1; img_b64 += len(c.get('image_url') or ''); img_detail.add(c.get('detail'))
                elif c.get('type') == 'input_text':
                    t = c.get('text') or ''
                    if t.startswith('[Older camera image omitted'):
                        n_placeholder += 1
                    text_chars += len(t)
                    if i == last_user_idx: last_text_chars += len(t)
        elif it.get('type') == 'reasoning':
            n_reason += 1; reason_enc += len(it.get('encrypted_content') or '')
        elif it.get('type') == 'message':
            n_asst += 1
            for c in it.get('content') or []:
                asst_chars += len(c.get('text') or '')
    wire = len(json.dumps(pay, ensure_ascii=False, separators=(',', ':')).encode('utf-8')) if pay else None
    meta = d.get('request_meta') or {}
    hs = d.get('history_stats_before_request') or {}
    return dict(
        req_file_bytes=os.path.getsize(p), wire_bytes=wire,
        kind=meta.get('kind') or d.get('mode'), step=meta.get('step'),
        instructions_chars=len(pay.get('instructions') or ''),
        schema_chars=len(json.dumps(pay.get('text'), separators=(',', ':'))) if pay.get('text') else None,
        strict=((pay.get('text') or {}).get('format') or {}).get('strict'),
        fmt_type=((pay.get('text') or {}).get('format') or {}).get('type'),
        max_output_tokens_sent=('max_output_tokens' in pay),
        reasoning_effort=(pay.get('reasoning') or {}).get('effort'),
        n_input_items=len(inp), n_user=n_user, n_reason=n_reason, n_asst=n_asst,
        n_img=n_img, img_detail=sorted(x for x in img_detail if x), n_placeholder=n_placeholder,
        img_b64_chars=img_b64, text_chars=text_chars, last_text_chars=last_text_chars,
        reason_enc_chars=reason_enc, asst_chars=asst_chars,
        budget_before=d.get('budget_elapsed_s_before'), timeout_s=d.get('timeout_s'),
        history_protocol=d.get('history_protocol') or d.get('agent_speedup_v3_history_protocol'),
        image_history=d.get('image_history'),
        hs_retained_pairs=hs.get('retained_image_pairs'), hs_messages=hs.get('messages'),
        prompt_version=d.get('prompt_version'),
    )

def resp_features(p):
    d = load(p)
    u = d.get('usage') or {}
    itd = u.get('input_tokens_details') or {}
    otd = u.get('output_tokens_details') or {}
    out_text = 0; out_ws = 0
    for it in d.get('output') or []:
        if it.get('type') == 'message':
            for c in it.get('content') or []:
                t = c.get('text') or ''
                out_text += len(t); out_ws += sum(1 for ch in t if ch in ' \n\r\t')
    return dict(
        input_tokens=u.get('input_tokens'), cached_tokens=itd.get('cached_tokens'),
        cache_write_tokens=itd.get('cache_write_tokens'),
        output_tokens=u.get('output_tokens'), reasoning_tokens=otd.get('reasoning_tokens'),
        created_at=d.get('created_at'), completed_at=d.get('completed_at'),
        server_s=(d.get('completed_at') - d.get('created_at')) if d.get('completed_at') and d.get('created_at') else None,
        out_text_chars=out_text, out_ws_chars=out_ws, resp_status=d.get('status'),
        prompt_cache_key=d.get('prompt_cache_key'), service_tier=d.get('service_tier'),
        resp_max_output_tokens=d.get('max_output_tokens'),
    )

def res_features(p):
    d = load(p)
    b = d.get('budget') or {}
    td = d.get('v3_transport_diagnostics') or {}
    return dict(
        res_status=d.get('status'), error_code=d.get('error_code'), error_type=d.get('error_type'),
        budget_after=b.get('elapsed_s'), cum_input_tokens=b.get('input_tokens'),
        cum_cached=b.get('cached_input_tokens'), cum_output=b.get('output_tokens'),
        cum_reasoning=b.get('reasoning_tokens'), cum_calls=b.get('api_calls'),
        td_elapsed=td.get('elapsed_s'), td_status=td.get('status'), td_stream_bytes=td.get('stream_bytes'),
        td_visible=td.get('visible_characters'), td_ws_run=td.get('longest_json_whitespace_run'),
        td_deadline_exceeded=td.get('deadline_exceeded'), td_events=td.get('events'),
        request_kind=d.get('request_kind'),
    )

def trace_latencies(run):
    lat = {}
    p = os.path.join(run, 'trace.jsonl')
    if not os.path.exists(p):
        return lat
    rx_log = re.compile(r'"api_request_log": "([^"]+)"')
    with open(p, 'rb') as f:
        for line in f:
            if b'"api_calls"' not in line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            diag = ((r.get('decision') or {}).get('diagnostics')) if isinstance(r.get('decision'), dict) else None
            if diag is None:
                diag = r.get('diagnostics')
            if not isinstance(diag, dict):
                continue
            calls = diag.get('api_calls') or []
            logp = diag.get('api_request_log')
            if calls and logp:
                base = os.path.basename(logp).split('.')[0]
                lat[base] = dict(trace_latency=sum(c.get('latency_s') or 0 for c in calls),
                                 trace_ncalls=len(calls), trace_step=r.get('step'),
                                 trace_kinds=[c.get('kind') for c in calls])
    return lat

def run_rows(run, label):
    rows = []
    rq = os.path.join(run, 'requests')
    if not os.path.isdir(rq):
        return rows
    lat = trace_latencies(run)
    bases = sorted(set(f.split('.')[0] for f in os.listdir(rq) if re.match(r'^\d{6}-[0-9a-f]{32}\.', f)))
    for b in bases:
        row = dict(run=label, base=b, ordinal=int(b[:6]))
        for suf, fn in (('request', req_features), ('response', resp_features), ('result', res_features)):
            p = os.path.join(rq, f'{b}.{suf}.json')
            if os.path.exists(p):
                try:
                    row.update(fn(p))
                except Exception as e:
                    row[suf + '_error'] = repr(e)[:200]
        for suf in ('overload_retry', 'transient_retry'):
            p = os.path.join(rq, f'{b}.request.{suf}.json')
            if os.path.exists(p):
                row[suf] = load(p)
        if row.get('budget_after') is not None and row.get('budget_before') is not None:
            row['client_interval_s'] = row['budget_after'] - row['budget_before']
        row.update(lat.get(b, {}))
        rows.append(row)
    return rows

if __name__ == '__main__':
    out = sys.argv[1]
    runs = sys.argv[2:]
    allrows = []
    for r in runs:
        label = os.path.relpath(r, '/workspace/gpt_field')
        rows = run_rows(r, label)
        allrows += rows
        print(label, len(rows), file=sys.stderr)
    with open(out, 'w') as f:
        for row in allrows:
            f.write(json.dumps(row) + '\n')
