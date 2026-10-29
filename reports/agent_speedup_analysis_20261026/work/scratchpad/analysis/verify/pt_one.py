import json, sys
for path in sys.argv[1:]:
    p = json.load(open(path))
    su = p['summary']
    print('==', path)
    print('  summary:', {k: su.get(k) for k in ['raw_libero_success','raw_libero_success_after_settle','physical_success_at_terminal','physical_success_after_settle','v2_success','verdict','terminal_failure_is_hold_only']})
    for state in ['terminal', 'after_release_settle']:
        st = p.get(state, {})
        print('  [%s] steps=%s check_success=%s raw_conj=%s strict_conj=%s' % (state, st.get('steps'), st.get('check_success'), st.get('raw_conjunction'), st.get('strict_conjunction')))
        if state == 'after_release_settle':
            print('     settle:', json.dumps(st.get('settle')))
        for a in st.get('atoms', []):
            n = a.get('numbers', {})
            print('    atom', a['atom'], 'raw', a.get('raw'), 'strict', a.get('strict'), 'why_not', a.get('why_not'))
            print('       libero', json.dumps(a.get('libero')))
            print('       numbers', json.dumps({k: n.get(k) for k in ['origin','com','low_z','high_z','speed_mps','ang_rps','robot_contacts','tilt_deg','window_disp_mm','support_top_z','rim_z','floor_z','lowest_below_rim_mm','origin_minus_low_mm'] if k in n}))
