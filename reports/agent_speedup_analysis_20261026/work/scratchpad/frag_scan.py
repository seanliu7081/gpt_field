import json, sys, collections, statistics
runs=sys.argv[1:]
out={}
for r in runs:
    d=f'/workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_{r}'
    rows=[]
    with open(f'{d}/trace.jsonl','rb') as f:
        for line in f:
            if not line.startswith(b'{"event": "feedback"'): continue
            dd=json.loads(line)
            fb=dd['feedback']; pf=fb.get('primitive_feedback') or {}
            if not pf.get('done'): continue
            if pf.get('primitive')=='execute_fragment':
                fr=pf['fragment']; segs=fr['segments']; cur=segs[fr['current_segment_index']]
                rows.append({'step':dd['step'],'kind':'execute','reason':fr['reason'],'executed':fr['executed_steps'],'budget':fr['budget_steps'],
                    'nseg':len(segs)+len(fr['pending_segments']),'completed':len(fr['completed_segment_indices']),
                    'cur_skill':cur['skill'],'cur_profile':cur['request']['profile'],'cur_actions':cur['actions'],'cur_max':cur['request']['max_steps'],
                    'cur_pos_err':cur.get('position_error_m'),'cur_rot_err':cur.get('orientation_error_rad'),
                    'gap':fr['measured_gap_mm'],'grip':fr['gripper_command'],
                    'seg_reasons':[(s['skill'],s['request']['profile'],s['actions'],s['reason']) for s in segs]})
            else:
                rows.append({'step':dd['step'],'kind':pf.get('primitive'),'reason':pf.get('reason')})
    out[r]=rows
json.dump(out, open('/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/frag_rows.json','w'))
print('done', {r: len(v) for r,v in out.items()})
