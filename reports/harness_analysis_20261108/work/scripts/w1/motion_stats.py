import json, sys, numpy as np
from collections import Counter, defaultdict
D = '/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/data/'
for key in ['t04_t100', 't08_t100', 't08_stoveoff', 'drawer_001', 'drawer_003']:
    d = json.load(open(D + key + '.json'))
    rows = d['rows']
    n = len(rows)
    by = defaultdict(list)
    for r in rows:
        k = (r['primitive'], r['skill'], r['profile'])
        by[k].append(r)
    print('=====', key, 'steps', n)
    tot_path = sum(r['disp'] for r in rows)
    print('  total grip-site path %.3f m, mean %.2f mm/step' % (tot_path, tot_path/n*1000))
    for k, rs in sorted(by.items(), key=lambda kv: -len(kv[1])):
        disp = np.array([r['disp'] for r in rs]) * 1000
        cmd = np.array([r['cmd_norm_norm'] for r in rs]) * 100  # normalized*0.10 m -> mm
        print('  %-45s n=%4d (%4.1f%%) path %.3f m (%4.1f%%) disp mean %.2f med %.2f p90 %.2f max %.2f mm | cmd mean %.1f mm' % (
            str(k), len(rs), len(rs)/n*100, disp.sum()/1000, disp.sum()/1000/tot_path*100, disp.mean(), np.median(disp), np.percentile(disp, 90), disp.max(), cmd.mean()))
    # transit steps: realized fraction of command
    tr = [r for r in rows if r['profile'] == 'transit' and r['skill'] in ('move_to_pose', 'move_relative')]
    if tr:
        disp = np.array([r['disp'] for r in tr]); cmd = np.array([r['cmd_norm_norm'] * 0.10 for r in tr])
        moving = cmd > 1e-6
        print('  transit: n=%d, cmd>0: %d, zero-cmd (translation complete, rotating/settling) %d; mean disp %.2f mm; ratio disp/cmd (cmd>0) mean %.3f median %.3f; steps cmd>=50mm %d' % (
            len(tr), moving.sum(), (~moving).sum(), disp.mean()*1000, float(np.mean(disp[moving]/cmd[moving])), float(np.median(disp[moving]/cmd[moving])), int((cmd>=0.05).sum())))
        fast = disp >= 0.01
        print('  transit steps with disp>=10mm: %d (%.1f%%), they cover %.1f%% of transit path; steps < 2 mm: %d (%.1f%%)' % (
            fast.sum(), fast.mean()*100, disp[fast].sum()/disp.sum()*100, (disp < 0.002).sum(), (disp < 0.002).mean()*100))
    ct = [r for r in rows if r['profile'] == 'contact' and r['skill'] in ('approach', 'move_to_pose', 'move_relative')]
    if ct:
        disp = np.array([r['disp'] for r in ct]); cmd = np.array([r['cmd_norm_norm'] * 0.10 for r in ct])
        print('  contact motion: n=%d mean disp %.2f mm (cmd mean %.2f mm) ratio %.3f; path %.3f m' % (len(ct), disp.mean()*1000, cmd.mean()*1000, float(np.mean(disp[cmd>1e-6]/cmd[cmd>1e-6])), disp.sum()))
    gr = [r for r in rows if r['skill'] == 'set_gripper']
    print('  set_gripper steps: %d (%.1f%%)' % (len(gr), len(gr)/n*100))
