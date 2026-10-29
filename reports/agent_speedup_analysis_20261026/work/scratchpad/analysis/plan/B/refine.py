"""Plan B: how much did the model change its target after each pre-contact checkpoint? Read-only.

For pre_contact handoffs (fragment k completed, no gripper stage), compare the xy of
fragment k's final executed target with the xy of decision k+1's first resolved arm
target (segment targets as resolved by the executor, from episode.events.jsonl).
Small xy changes indicate the checkpoint confirmed rather than redirected.
"""
import json, os, math, statistics, collections

B = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(B, 'decisions.json')))
H = json.load(open(os.path.join(B, 'handoffs.json')))


def xy(v):
    return None if v is None or v[0] is None else (v[0], v[1])


def main():
    rows = []
    for name, run in D.items():
        rs = run['rows']
        for h in H[name]['handoffs']:
            if h['class'] != 'pre_contact':
                continue
            k = h['k']
            r, nx = rs[k], rs[k + 1]
            if not r['seg_out'] or nx['type'] != 'execute' or not nx['seg_out']:
                continue
            tgt_k = r['seg_out'][-1]['target']
            # first arm segment of k+1 with a resolved target
            first = next((s for s in nx['seg_out'] if s['skill'] in ('move_to_pose', 'move_relative', 'approach')
                          and s['target'] is not None), None)
            if first is None or tgt_k is None:
                continue
            dxy = math.dist(xy(tgt_k), xy(first['target']))
            dz = first['target'][2] - tgt_k[2]
            nxt_first = nx['segs'][0]['skill'] + '/' + nx['segs'][0]['profile']
            has_grip = any(s['skill'] == 'set_gripper' for s in nx['segs'])
            rows.append({'run': name, 'k': k, 'dxy_mm': dxy * 1000, 'dz_mm': dz * 1000, 'next_first': nxt_first,
                         'next_has_gripper': has_grip, 'L_next': h['L_next'],
                         'last_seg_k': r['seg_out'][-1]['skill'] + '/' + str(r['seg_out'][-1]['profile'])})
    v = [x['dxy_mm'] for x in rows]
    print('pre_contact handoffs with resolvable targets:', len(rows))
    for thr in (2, 5, 10, 20, 50):
        print(f'  dxy <= {thr} mm: {sum(x <= thr for x in v)}')
    print('  median dxy mm', round(statistics.median(v), 1), 'p75', round(sorted(v)[int(.75 * len(v))], 1))
    print('  next decision has gripper stage:', sum(x['next_has_gripper'] for x in rows))
    print('  next first segment kinds:', collections.Counter(x['next_first'] for x in rows).most_common())
    print('  dz (next first target minus k final target) median mm', round(statistics.median(x['dz_mm'] for x in rows), 1))
    small = [x for x in rows if x['dxy_mm'] <= 5 and x['next_has_gripper']]
    print('  dxy<=5mm AND next has gripper (descend+grip after a confirming view):', len(small),
          'their next-call latency sum', round(sum(x['L_next'] for x in small), 1))
    json.dump(rows, open(os.path.join(B, 'refine.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
