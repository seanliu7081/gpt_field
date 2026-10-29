#!/usr/bin/env python3
"""Read-only extraction of per-call model request metrics from run directories.

Usage: python3 -I extract_calls.py <out_json> <run_dir> [<run_dir> ...]
"""
import glob
import json
import os
import statistics
import sys


def load(p):
    with open(p) as f:
        return json.load(f)


def pct(vals, q):
    if not vals:
        return None
    v = sorted(vals)
    k = (len(v) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def deepest_segment_reason(stage_outcomes):
    reasons = []
    for so in stage_outcomes or []:
        if isinstance(so, dict) and so.get('reason'):
            reasons.append((so.get('path'), so.get('status'), so.get('reason')))
    return reasons


def parse_request(path):
    d = load(path)
    out = {}
    out['request_bytes'] = os.path.getsize(path)
    payload = d.get('payload', {})
    out['instructions_chars'] = len(payload.get('instructions') or '')
    try:
        out['schema_chars'] = len(json.dumps(payload['text']['format']['schema'], separators=(',', ':')))
    except Exception:
        out['schema_chars'] = None
    inp = payload.get('input') or []
    n_img = 0
    img_chars = 0
    user_text = 0
    asst_text = 0
    n_reason = 0
    n_user_msgs = 0
    n_asst = 0
    for m in inp:
        role = m.get('role') or m.get('type')
        if m.get('type') == 'reasoning':
            n_reason += 1
            continue
        c = m.get('content')
        if not isinstance(c, list):
            continue
        if role == 'user':
            n_user_msgs += 1
        if role == 'assistant':
            n_asst += 1
        for x in c:
            t = x.get('type')
            if t == 'input_image':
                n_img += 1
                iu = x.get('image_url')
                if isinstance(iu, str):
                    img_chars += len(iu)
                elif isinstance(iu, dict):
                    img_chars += len(iu.get('url', ''))
            elif t == 'input_text':
                user_text += len(x.get('text') or '')
            elif t == 'output_text':
                asst_text += len(x.get('text') or '')
    out['images_total'] = n_img
    out['image_url_chars'] = img_chars
    out['user_text_chars'] = user_text
    out['assistant_text_chars'] = asst_text
    out['reasoning_items'] = n_reason
    out['user_messages'] = n_user_msgs
    out['assistant_messages'] = n_asst
    out['text_chars_total'] = (out['instructions_chars'] + (out['schema_chars'] or 0) + user_text + asst_text)
    # last user message = current observation
    last = None
    for m in reversed(inp):
        if (m.get('role') == 'user'):
            last = m
            break
    out['last_user_images'] = 0
    out['last_user_text_chars'] = 0
    trig = {}
    if last is not None:
        for x in last.get('content', []):
            if x.get('type') == 'input_image':
                out['last_user_images'] += 1
            elif x.get('type') == 'input_text':
                out['last_user_text_chars'] += len(x.get('text') or '')
        # first input_text is JSON observation
        try:
            t0 = [x for x in last['content'] if x.get('type') == 'input_text'][0]['text']
            j = json.loads(t0)
            ctx = j.get('context') or {}
            lf = ctx.get('last_feedback') or {}
            s = lf.get('summary') if isinstance(lf, dict) else None
            if isinstance(s, dict):
                trig['lf_action'] = s.get('action')
                trig['lf_status'] = s.get('status')
                trig['lf_reason'] = s.get('reason')
                trig['lf_executed_steps'] = s.get('executed_steps')
                ev = s.get('evidence') if isinstance(s.get('evidence'), dict) else {}
                trig['lf_ev_reason'] = ev.get('reason')
                trig['lf_ev_status'] = ev.get('status')
            elif isinstance(lf, dict) and lf:
                trig['lf_action'] = lf.get('action')
                trig['lf_status'] = lf.get('status')
                trig['lf_reason'] = lf.get('reason')
                trig['lf_executed_steps'] = lf.get('executed_steps')
            if isinstance(lf, dict):
                trig['lf_stage_reasons'] = deepest_segment_reason(lf.get('stage_outcomes'))
            cd = j.get('current_decision') or {}
            if cd:
                mer = cd.get('missing_evidence_requests')
                trig['cd_missing_evidence_n'] = len(mer) if isinstance(mer, list) else None
                trig['cd_missing_kinds'] = sorted({(m.get('kind') or m.get('type') or m.get('requirement') or '?') for m in mer if isinstance(m, dict)}) if isinstance(mer, list) else None
                bl = cd.get('blocker')
                trig['cd_blocker'] = (bl.get('reason') or bl.get('status') or bl.get('kind')) if isinstance(bl, dict) else bl
                ep = cd.get('evidence_pause')
                trig['cd_evidence_pause'] = (ep.get('reason') or ep.get('status')) if isinstance(ep, dict) else ep
                ea = cd.get('evidence_acquisition')
                trig['cd_evidence_acq'] = (ea.get('status') or ea.get('state')) if isinstance(ea, dict) else ea
                pi = cd.get('pending_intent')
                trig['cd_pending_kind'] = pi.get('kind') if isinstance(pi, dict) else None
                rag = cd.get('repeated_action_guard')
                trig['cd_repeat_guard'] = (rag.get('status') or rag.get('reason')) if isinstance(rag, dict) else rag
        except Exception as e:
            trig['parse_error'] = repr(e)[:200]
    out['trigger'] = trig
    meta = d.get('request_meta') or {}
    out['step'] = meta.get('step')
    out['kind'] = meta.get('kind')
    out['budget_elapsed_s_before'] = d.get('budget_elapsed_s_before')
    out['prompt_version'] = d.get('prompt_version') or d.get('agent_speedup_v3_prompt_version')
    out['client_version'] = d.get('agent_speedup_v3_client_version') or d.get('agent_speedup_v2_client_version')
    out['history_protocol'] = d.get('history_protocol')
    return out


def parse_response(path):
    d = load(path)
    out = {'response_bytes': os.path.getsize(path)}
    out['status'] = d.get('status')
    out['created_at'] = d.get('created_at')
    out['completed_at'] = d.get('completed_at')
    if d.get('created_at') and d.get('completed_at'):
        out['server_span_s'] = d['completed_at'] - d['created_at']
    u = d.get('usage') or {}
    out['input_tokens'] = u.get('input_tokens')
    out['cached_tokens'] = (u.get('input_tokens_details') or {}).get('cached_tokens')
    out['output_tokens'] = u.get('output_tokens')
    out['reasoning_tokens'] = (u.get('output_tokens_details') or {}).get('reasoning_tokens')
    diag = d.get('v3_transport_diagnostics') or ((d.get('_subscription_transport') or {}).get('v3_bounds')) or {}
    out['transport_elapsed_s'] = diag.get('elapsed_s')
    out['stream_bytes'] = diag.get('stream_bytes')
    out['visible_characters'] = diag.get('visible_characters')
    text = None
    for o in d.get('output') or []:
        if o.get('type') == 'message':
            for c in o.get('content') or []:
                if c.get('type') == 'output_text':
                    text = c.get('text')
    out['output_text_chars'] = len(text) if text else 0
    dec = {}
    if text:
        try:
            j = json.loads(text)
            dec['type'] = j.get('type')
            mc = j.get('motion_context') or {}
            dec['intent'] = mc.get('intent') if isinstance(mc, dict) else None
            dec['target_id'] = mc.get('target_id') if isinstance(mc, dict) else None
            su = j.get('scene_update')
            if isinstance(su, dict):
                ents = su.get('entities')
                dec['scene_entities'] = len(ents) if isinstance(ents, list) else None
                dec['scene_update_chars'] = len(json.dumps(su, separators=(',', ':')))
                dec['scene_update_keys'] = sorted(su.keys())
            ex = j.get('execution')
            if isinstance(ex, dict):
                segs = ex.get('segments')
                dec['segments'] = len(segs) if isinstance(segs, list) else None
                dec['skills'] = [s.get('skill') for s in segs] if isinstance(segs, list) else None
            pr = j.get('program')
            if isinstance(pr, dict):
                st = pr.get('stages')
                ops = pr.get('operations')
                dec['program_stages'] = len(st) if isinstance(st, list) else None
                dec['program_operations'] = len(ops) if isinstance(ops, list) else None
            cont = j.get('continuation')
            if isinstance(cont, dict):
                dec['continuation_resume'] = cont.get('resume')
            dec['macro'] = (j.get('macro') or {}).get('step_index') if isinstance(j.get('macro'), dict) else None
            dec['why'] = (j.get('why') or '')[:300]
            # evidence acquisition fields if any
            for k in ('evidence', 'acquire_evidence', 'evidence_request'):
                if k in j and j[k] is not None:
                    dec['has_' + k] = True
            dec['keys'] = sorted(j.keys())
        except Exception as e:
            dec['parse_error'] = repr(e)[:200]
    out['decision'] = dec
    return out


def parse_result(path):
    d = load(path)
    out = {'result_status': d.get('status')}
    b = d.get('budget') or {}
    out['budget_elapsed_s_after'] = b.get('elapsed_s')
    out['request_kind'] = d.get('request_kind')
    return out


def run(run_dir):
    reqdir = os.path.join(run_dir, 'requests')
    calls = []
    reqs = sorted(glob.glob(os.path.join(reqdir, '*.request.json')))
    for rq in reqs:
        base = rq[:-len('.request.json')]
        rec = {'file': os.path.basename(base)}
        try:
            rec.update(parse_request(rq))
        except Exception as e:
            rec['request_error'] = repr(e)[:200]
        rs = base + '.response.json'
        if os.path.exists(rs):
            try:
                rec.update(parse_response(rs))
            except Exception as e:
                rec['response_error'] = repr(e)[:200]
        else:
            rec['response_missing'] = True
        rr = base + '.result.json'
        if os.path.exists(rr):
            try:
                rec.update(parse_result(rr))
            except Exception as e:
                rec['result_error'] = repr(e)[:200]
        else:
            rec['result_missing'] = True
        if rec.get('budget_elapsed_s_after') is not None and rec.get('budget_elapsed_s_before') is not None:
            rec['budget_interval_s'] = rec['budget_elapsed_s_after'] - rec['budget_elapsed_s_before']
        calls.append(rec)
    # other request-like files (errors)
    others = sorted(set(os.listdir(reqdir)) - {os.path.basename(x) for x in glob.glob(os.path.join(reqdir, '*.request.json'))} - {os.path.basename(x) for x in glob.glob(os.path.join(reqdir, '*.response.json'))} - {os.path.basename(x) for x in glob.glob(os.path.join(reqdir, '*.result.json'))}) if os.path.isdir(reqdir) else []
    lat = [c.get('transport_elapsed_s') if c.get('transport_elapsed_s') is not None else c.get('budget_interval_s') for c in calls]
    lat = [x for x in lat if x is not None]
    summ = {
        'run_dir': run_dir,
        'n_request_files': len(reqs),
        'other_files_in_requests': others,
        'latency_n': len(lat),
        'latency_sum_s': sum(lat) if lat else None,
        'latency_mean_s': statistics.mean(lat) if lat else None,
        'latency_p50_s': pct(lat, 0.5),
        'latency_p90_s': pct(lat, 0.9),
        'latency_max_s': max(lat) if lat else None,
        'first_call_latency_s': lat[0] if lat else None,
    }
    return {'summary': summ, 'calls': calls}


def main():
    out = sys.argv[1]
    res = {}
    for rd in sys.argv[2:]:
        res[os.path.basename(rd.rstrip('/'))] = run(rd)
    with open(out, 'w') as f:
        json.dump(res, f, indent=1)


if __name__ == '__main__':
    main()
