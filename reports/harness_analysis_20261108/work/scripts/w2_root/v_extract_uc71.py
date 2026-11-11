"""Independent re-extraction of the uc7.1-family run table from raw run artifacts.

Reads only files under /workspace/gpt_field/runs/<run>/ (read-only). Does not use any w1 script or data.
"""
import json, glob, os, sys, statistics
from pathlib import Path

RUNS = Path('/workspace/gpt_field/runs')
UC71 = {
    'H1_T04': 'agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001',
    'H3_T04': 'foundation_vision_t04_can_obstacle_20261105_001',
    'T20_T04': 'agent_speedup_v2_uc71_can_obstacle_t04_20261103_001',
    'DEF_T04': 'agent_speedup_v2_uc71_default_routing_t04_20261103_001',
    'H1_T08': 'agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001',
    'H3_T08': 'foundation_vision_t08_bottle_obstacle_20261105_001',
    'SOFF_T08': 'agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001',
    'DRW_001': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001',
    'DRW_003': 'agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003',
}


def load_jsonl(p):
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def result_files(d):
    return sorted(glob.glob(str(d / '*.result.json')))


def main():
    table = {}
    for key, run in UC71.items():
        R = RUNS / run
        T = R / 'trial'
        rows = load_jsonl(T / 'trace.jsonl')
        actions = [r for r in rows if r.get('event') == 'action']
        executed = [r for r in rows if r.get('event') == 'executed']
        summ = json.load(open(T / 'summary.json'))
        tc = summ.get('timing_components', {})
        # calls and latency from action diagnostics
        call_lat = []
        call_steps = []
        for a in actions:
            for c in (a.get('diagnostics') or {}).get('api_calls') or []:
                call_lat.append(c.get('latency_s'))
                call_steps.append(a['step'])
        # trial request results
        trial_results = result_files(T / 'requests')
        review_results = result_files(R / 'review' / 'requests') if (R / 'review' / 'requests').is_dir() else []
        plan_elapsed = []
        plan_reason = []
        for p in review_results:
            j = json.load(open(p))
            b = j.get('budget', {})
            plan_elapsed.append(b.get('elapsed_s'))
            plan_reason.append((j.get('request_kind'), j.get('status'), b.get('reasoning_tokens'), b.get('input_tokens'), b.get('output_tokens')))
        # preparation timing (H3 might store elsewhere)
        rs = None
        if (R / 'result_summary.json').is_file():
            rs = json.load(open(R / 'result_summary.json'))
        an = None
        if (R / 'analysis.json').is_file():
            an = json.load(open(R / 'analysis.json'))
        uc = summ.get('user_check') or {}
        prep_model_s = uc.get('preparation_model_s')
        prep_elapsed_s = uc.get('preparation_elapsed_s')
        last = executed[-1]['sim_status'] if executed else {}
        table[key] = {
            'run': run,
            'status': summ.get('status'),
            'libero_success': summ.get('libero_success'),
            'v2_success': summ.get('v2_success'),
            'steps_summary': summ.get('steps'),
            'n_action_rows': len(actions),
            'n_executed_rows': len(executed),
            'n_exec_calls_in_trace': len(call_lat),
            'n_trial_result_files': len(trial_results),
            'n_review_result_files': len(review_results),
            'plan_results': plan_reason,
            'plan_elapsed_s': plan_elapsed,
            'sum_exec_latency_s': sum(call_lat),
            'median_exec_latency_s': statistics.median(call_lat) if call_lat else None,
            'mean_exec_latency_s': statistics.mean(call_lat) if call_lat else None,
            'first_exec_latency_s': call_lat[0] if call_lat else None,
            'call_steps': call_steps,
            'control_elapsed_s': summ.get('control_elapsed_s'),
            'control_without_video_s': summ.get('control_without_video_s'),
            'elapsed_s': summ.get('elapsed_s'),
            'startup_to_initial_observation_s': summ.get('startup_to_initial_observation_s'),
            'timing_components': tc,
            'user_check_prep_model_s': prep_model_s,
            'user_check_prep_elapsed_s': prep_elapsed_s,
            'rs_timing': rs.get('timing') if rs else None,
            'rs_status': rs.get('status') if rs else None,
            'rs_layout': rs.get('layout') if rs else None,
            'rs_transit': rs.get('transit_command_limit_m') if rs else None,
            'rs_model_calls': {k: rs['model_calls'].get(k) for k in ('initial_plan', 'execution', 'total')} if rs else None,
            'analysis_timing': (an.get('timing_s') if an else None),
            'analysis_calls': (an.get('calls') if an else None),
            'final_sim_steps': last.get('steps'),
            'final_libero_success': last.get('libero_success'),
            'final_counts': (last.get('collision') or {}).get('counts'),
        }
    out = Path('/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/v_uc71_table.json')
    json.dump(table, open(out, 'w'), indent=1, default=str)
    for k, v in table.items():
        tc = v['timing_components']
        print(f"{k:9s} status={v['status']} lib={v['libero_success']} actions={v['n_action_rows']}/{v['n_executed_rows']} steps={v['steps_summary']} "
              f"execcalls={v['n_exec_calls_in_trace']} trialres={v['n_trial_result_files']} reviewres={v['n_review_result_files']} ")
        print(f"          sumlat={v['sum_exec_latency_s']:.2f} model_pipeline={tc.get('model_pipeline_s')} med={v['median_exec_latency_s']:.2f} mean={v['mean_exec_latency_s']:.2f} first={v['first_exec_latency_s']:.2f}")
        print(f"          control={v['control_elapsed_s']} ctl_wo_video={v['control_without_video_s']} video={tc.get('video_render_s')} sim_step={tc.get('simulator_step_s')} run_state={tc.get('run_state_s')} obs={tc.get('observe_transition_s')} pol={tc.get('policy_compute_s')} decision={tc.get('decision_s')}")
        print(f"          plan_elapsed={v['plan_elapsed_s']} prep_model={v['user_check_prep_model_s']} prep_elapsed={v['user_check_prep_elapsed_s']} plans={v['plan_results']}")
        print(f"          rs_timing={v['rs_timing']}")
        print(f"          rs_layout={v['rs_layout']} transit={v['rs_transit']} rs_calls={v['rs_model_calls']}")
        if v['analysis_timing'] is not None:
            print(f"          analysis_timing={json.dumps(v['analysis_timing'])[:800]}")
            print(f"          analysis_calls={json.dumps(v['analysis_calls'])[:600]}")


if __name__ == '__main__':
    main()
