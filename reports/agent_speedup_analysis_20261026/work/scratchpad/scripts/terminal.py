#!/usr/bin/env python3
"""Read-only: parse the EvidenceAcquisitionFailed JSON in error_traceback.txt."""
import json, sys
for run in sys.argv[1:]:
    try:
        t = open(run + '/error_traceback.txt').read()
    except FileNotFoundError:
        print('==', run, 'no error_traceback'); continue
    k = t.find('EvidenceAcquisitionFailed: {')
    if k < 0:
        print('==', run, 'not EAF:', t.strip().splitlines()[-1][:300]); continue
    js = t[k + len('EvidenceAcquisitionFailed: '):].strip()
    # take the first JSON object
    dec = json.JSONDecoder()
    obj, end = dec.raw_decode(js)
    b = obj.get('blocker') or {}
    print('==', run)
    print('   top keys:', sorted(obj.keys()))
    print('   reason:', obj.get('reason')[:400])
    print('   step:', obj.get('step'), '| automatic_retry:', obj.get('automatic_retry'))
    print('   missing_evidence_requests:', obj.get('missing_evidence_requests'))
    print('   blocker: status=%s reason=%s allowed_fraction=%s unknown_fraction=%s stale=%s blocking_reasons=%s' % (b.get('status'), b.get('reason'), b.get('allowed_fraction'), b.get('unknown_fraction'), b.get('stale_evidence_ids'), b.get('blocking_reasons')))
    w = b.get('whole_goal_assessment') or {}
    if w:
        print('   whole_goal: status=%s reason=%s unknown_fraction=%s missing=%s' % (w.get('status'), w.get('reason'), w.get('unknown_fraction'), w.get('missing_evidence_requests')))
    for kk in ('tracking_status', 'limiting_pair', 'semantic_goal', 'payload_geometry_unknown'):
        if kk in b:
            print('   blocker.%s: %s' % (kk, json.dumps(b.get(kk))[:300]))
