"""Offline replay of the proposed measured-state blocked-motion signal over saved traces (independent re-implementation).

Signal: commanded translation > 5 mm AND measured/commanded < 0.1 for 3 consecutive steps; count rising edges.
Commanded translation = |normalized_action xyz| * 0.1 m (declared worker OSC scale, result_summary
command_configuration_audit). Measured = |feedback.measured_translation_delta_m|.
Overlap with forbidden-contact steps from v_contacts.json (per-step forbidden sample deltas > 0).
"""
import json, math
from pathlib import Path

RUNS = Path('/workspace/gpt_field/runs')
SEL = {
    'H1_T04': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
    'H1_T08': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
    'H3_T04': 'foundation_vision_t04_can_obstacle_20261105_001',
    'H3_T08': 'foundation_vision_t08_bottle_obstacle_20261105_001',
    'SOFF_T08': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
    'DRW_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
    'DRW_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
SCALE = 0.1
contacts = json.load(open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/v_contacts.json'))


def main():
    for key, run in SEL.items():
        rows = [json.loads(l) for l in open(RUNS / run / 'trial' / 'trace.jsonl')]
        fb = {r['step']: r['feedback'] for r in rows if r['event'] == 'feedback'}
        forb = set()
        for e in contacts[key]['episodes']:
            forb.update(range(e['start'], e['end'] + 1))
        cond = {}
        for s, f in fb.items():
            na = f.get('normalized_action'); md = f.get('measured_translation_delta_m')
            if not na or md is None:
                continue
            cmd = math.sqrt(sum(x * x for x in na[:3])) * SCALE
            meas = math.sqrt(sum(x * x for x in md)) if isinstance(md, list) else float(md)
            cond[s] = cmd > 0.005 and meas / cmd < 0.1
        fires = []
        steps = sorted(cond)
        run_len = 0
        prev = None
        for s in steps:
            if cond[s]:
                run_len = run_len + 1 if (prev == s - 1 and run_len > 0) else 1
            else:
                run_len = 0
            prev = s
            if run_len == 3:
                fires.append(s)
        with_contact = [s for s in fires if any(t in forb for t in range(s - 3, s + 4))]
        print(f"{key:9s} fires={len(fires):3d} near_forbidden_contact={len(with_contact):3d} first_fires={fires[:8]}")


if __name__ == '__main__':
    main()
