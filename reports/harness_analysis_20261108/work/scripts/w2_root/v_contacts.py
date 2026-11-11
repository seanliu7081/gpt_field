"""Independent contact decomposition from raw per-step cumulative collision snapshots.

For each run, read trial/trace.jsonl rows that carry sim_status.collision (uc7.1 'executed', astranomy 'step'),
difference the exact cumulative class counters (counts.points_<class>) and the top-12 per-class pair records
(pairs_by_class, truncated by agent_speedup_v2_contacts.py:346) between consecutive steps, and build per-pair
episodes (contiguous action indices with positive deltas, gaps <= GAP merged).
Checks: per-class sum of pair deltas == class-counter delta (truncation check).
"""
import json, sys
from collections import defaultdict
from pathlib import Path

RUNS = Path('/workspace/gpt_field/runs')
RUNS_ALL = {
    'H1_T04': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
    'H3_T04': 'foundation_vision_t04_can_obstacle_20261105_001',
    'H2_T04': 'foundation_harness_t04_can_obstacle_001',
    'T20_T04': 'agent_speedup_v2_uc71_can_obstacle_t04_20261103_001',
    'DEF_T04': 'agent_speedup_v2_uc71_default_routing_t04_20261103_001',
    'AST_001': 'astranomy_direct_can_obstacle_t04_high_20261104_001',
    'AST_002': 'astranomy_direct_can_obstacle_t04_high_20261104_002',
    'AST_CA': 'astranomy_direct_can_obstacle_t04_collision_avoidance_high_20261105_001',
    'H1_T08': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
    'H3_T08': 'foundation_vision_t08_bottle_obstacle_20261105_001',
    'SOFF_T08': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
    'DRW_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
    'DRW_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
FORBIDDEN = ("robot_scene", "robot_nonfinger_manipulated_object", "robot_object", "manipulated_object_scene")
OBSTACLES = ("tomato_sauce_obstacle_1", "ketchup_obstacle_1")
GAP = 6


def rows_with_collision(path):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get('event') in ('executed', 'step', 'reset'):
                ss = r.get('sim_status') or {}
                c = ss.get('collision')
                if c is not None:
                    out.append((r.get('event'), r.get('step'), c))
    return out


def owner(p):
    return p.split(':', 1)[0]


def main():
    result = {}
    for key, run in RUNS_ALL.items():
        rows = rows_with_collision(RUNS / run / 'trial' / 'trace.jsonl')
        prev_counts = defaultdict(int)
        prev_pairs = {}
        per_action = []  # (action_index, {(cls,pair): delta}, {cls: delta})
        trunc_mismatch = []
        for ev, step, c in rows:
            counts = c.get('counts') or {}
            pairs = {}
            for cls, lst in (c.get('pairs_by_class') or {}).items():
                for rec in lst:
                    pairs[(cls, tuple(rec['pair']))] = rec['points']
            if ev == 'reset':
                prev_counts = defaultdict(int, {k: v for k, v in counts.items()})
                prev_pairs = dict(pairs)
                continue
            # action index: uc7.1 executed rows have step = sim steps after action (1-based);
            # action index = step - 1 for the action just executed.
            ai = step if ev == 'executed' else step - 1
            dcls = {cls: counts.get('points_' + cls, 0) - prev_counts.get('points_' + cls, 0) for cls in FORBIDDEN}
            dpair = {}
            for k, v in pairs.items():
                d = v - prev_pairs.get(k, 0)
                if d:
                    dpair[k] = d
            for cls in FORBIDDEN:
                sp = sum(v for (c2, _), v in dpair.items() if c2 == cls)
                if sp != dcls[cls]:
                    trunc_mismatch.append((ai, cls, sp, dcls[cls]))
            per_action.append((ai, dpair, dcls))
            prev_counts = defaultdict(int, {k: v for k, v in counts.items()})
            prev_pairs = dict(pairs)
        # totals
        tot = defaultdict(int)
        for ai, dpair, dcls in per_action:
            for cls, v in dcls.items():
                tot[cls] += v
        # pair episodes for forbidden classes
        by_pair = defaultdict(list)
        for ai, dpair, dcls in per_action:
            for (cls, pair), d in dpair.items():
                if cls in FORBIDDEN and d > 0:
                    by_pair[(cls, pair)].append((ai, d))
        episodes = []
        for (cls, pair), lst in by_pair.items():
            lst.sort()
            cur = None
            for ai, d in lst:
                if cur and ai - cur['end'] <= GAP + 1:
                    cur['end'] = ai
                    cur['samples'] += d
                    cur['active_actions'] += 1
                else:
                    if cur:
                        episodes.append(cur)
                    cur = {'cls': cls, 'pair': list(pair), 'start': ai, 'end': ai, 'samples': d, 'active_actions': 1}
            if cur:
                episodes.append(cur)
        episodes.sort(key=lambda e: (e['start'], e['cls']))
        # classify obstacle involvement
        for e in episodes:
            owners = [owner(p) for p in e['pair']]
            e['obstacle'] = any(o in OBSTACLES for o in owners)
            e['owners'] = owners
            e['length'] = e['end'] - e['start'] + 1
        obstacle_samples = sum(e['samples'] for e in episodes if e['obstacle'])
        fe = (rows[-1][2].get('first_event') or {}) if rows else {}
        forb_total = sum(tot.values())
        result[key] = {
            'run': run, 'n_rows': len(per_action), 'forbidden_total_from_counters': forb_total,
            'by_class': dict(tot), 'obstacle_samples_from_pairs': obstacle_samples,
            'n_pair_episodes': len(episodes), 'n_obstacle_episodes': sum(1 for e in episodes if e['obstacle']),
            'first_forbidden_action': min((e['start'] for e in episodes), default=None),
            'monitor_first_event_action': fe.get('action_index'), 'monitor_first_event_class': fe.get('class'),
            'episodes': episodes, 'truncation_mismatches': trunc_mismatch[:20], 'n_trunc_mismatch': len(trunc_mismatch),
        }
        print(f"== {key}  forbidden={forb_total} by_class={dict(tot)} obstacle_samples={obstacle_samples} "
              f"pair_eps={len(episodes)} obstacle_eps={result[key]['n_obstacle_episodes']} first={result[key]['first_forbidden_action']} monitor_first={result[key]['monitor_first_event_action']} trunc_mismatch={len(trunc_mismatch)}")
        for e in episodes:
            print(f"     {e['cls']:36s} {e['start']:5d}-{e['end']:5d} len={e['length']:3d} active={e['active_actions']:3d} samples={e['samples']:5d} obst={e['obstacle']} {e['pair']}")
    json.dump(result, open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/v_contacts.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
