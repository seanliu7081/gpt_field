"""Per-call stream phases from trial/requests/*.stream.json, joined to per-call tokens (v_percall.json order)."""
import json, glob, statistics
from pathlib import Path
RUNS = Path('/workspace/gpt_field/runs')
UC71 = {
    'H1_T04': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
    'H1_T08': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
    'H3_T04': 'foundation_vision_t04_can_obstacle_20261105_001',
    'H3_T08': 'foundation_vision_t08_bottle_obstacle_20261105_001',
    'T20_T04': 'agent_speedup_v2_uc71_can_obstacle_t04_20261103_001',
    'DEF_T04': 'agent_speedup_v2_uc71_default_routing_t04_20261103_001',
    'SOFF_T08': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
    'DRW_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
    'DRW_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}
pc = json.load(open('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/v_percall.json'))
for key, run in UC71.items():
    fs = sorted(glob.glob(str(RUNS / run / 'trial' / 'requests' / '*.stream.json')))
    rows = [r for r in pc if r['run'] == key]
    hdr, pre, vis, rate, spr, started = [], [], [], [], [], []
    for f, r in zip(fs, rows):
        j = json.load(open(f))
        h = j.get('headers_received_s'); fv = j.get('first_visible_output_s'); lv = j.get('last_visible_output_s')
        if h is None or fv is None or lv is None:
            continue
        hdr.append(h); pre.append(fv - h); vis.append(lv - fv)
        if lv > fv: rate.append(j.get('visible_characters', 0) / (lv - fv))
        if r['rsn'] > 0: spr.append((fv - h) / r['rsn'])
        started.append(j.get('started_at', '')[:16])
    print(f"{key:9s} n={len(hdr):2d} start={started[0] if started else ''} headers_med={statistics.median(hdr):5.2f}s pre_visible_med={statistics.median(pre):6.2f}s visible_med={statistics.median(vis):5.2f}s vis_chars/s_med={statistics.median(rate):6.1f} pre_s_per_rsn_tok_med={statistics.median(spr)*1000:6.1f}ms")
