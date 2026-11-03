"""Exact original MP4 frame/time -> policy action and observed state mapping.

Reads the recorder's existing episode.events.jsonl; never imports a simulator.
Intervals use inclusive start/exclusive end. A state index counts executed actions.
"""
from bisect import bisect_right
from collections import Counter
from pathlib import Path
import json
import math

RUN = Path('/workspace/gpt_field/runs/agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001')
OUTPUT = Path(__file__).resolve().parent
FPS = 20
CROPS_XYXY = {'external': (44, 181, 380, 517), 'wrist': (44, 577, 380, 913)}
COLUMNS = ['frame_start', 'frame_count', 'event_line', 'event', 'action_index', 'state_index']


def read_blocks(path=RUN/'trial/episode.events.jsonl'):
    blocks = []
    final_state_index = 0
    for line_number, line in enumerate(path.open(), 1):
        event = json.loads(line)
        kind, step = event['event'], event['step']
        if kind == 'executed':
            action_index, state_index = step, step+1
            final_state_index = state_index
        elif kind in ('skill_decision', 'agent_action'):
            action_index, state_index = step, step
        elif kind == 'observation':
            action_index, state_index = None, 0
        elif kind == 'terminal':
            action_index, state_index = None, final_state_index
        else:
            raise ValueError('Unexpected recorded event: '+kind)
        blocks.append(dict(zip(COLUMNS, (event['frame_start'], event['frame_count'], line_number,
                                         kind, action_index, state_index))))
    return blocks


def frame_record(frame_index, blocks=None):
    blocks = read_blocks() if blocks is None else blocks
    index = bisect_right([b['frame_start'] for b in blocks], frame_index)-1
    if index < 0:
        raise IndexError(frame_index)
    block = blocks[index]
    if frame_index >= block['frame_start']+block['frame_count']:
        raise IndexError(frame_index)
    return block.copy()


def time_record(time_s, blocks=None):
    return frame_record(math.floor(time_s*FPS+1e-9), blocks)


def main():
    blocks = read_blocks()
    cursor = 0
    for block in blocks:
        assert block['frame_start'] == cursor
        cursor += block['frame_count']
    assert cursor == 3654
    event_counts = Counter(b['event'] for b in blocks)
    frame_counts = Counter()
    for block in blocks: frame_counts[block['event']] += block['frame_count']
    metadata = {
        'source_events': str(RUN/'trial/episode.events.jsonl'),
        'source_video': str(RUN/'trial/episode.mp4'),
        'fps': FPS, 'frame_count': cursor, 'duration_s': cursor/FPS,
        'event_counts': dict(event_counts), 'frames_by_event': dict(frame_counts),
        'policy_actions': event_counts['executed'], 'contiguous_coverage': True,
        'interval_convention': 'frame_start inclusive, frame_start+frame_count exclusive; displayed frame at time t is floor(t*20)',
        'state_index_convention': 'Initial state 0; action t transitions state t to t+1. Pre-action skill_decision/agent_action show state t; executed t shows state t+1. Terminal repeats final state 843, despite original retained event step 842.',
        'api_waits': 'Actual model/API wait time omitted. Each refreshed skill decision is fixed 12 frames / 0.6 s, irrespective of measured latency.',
        'setup': 'Initial 12 frames show the already-settled reset observation; ordinary setup settling actions are not in the MP4.',
        'terminal': 'Last 60 frames repeat policy-terminal state 843. Diagnostic release/settle actions are not in this video.',
        'camera_crops_xyxy_exclusive': CROPS_XYXY,
        'camera_crops_ffmpeg_whxy': {'external':[336,336,44,181], 'wrist':[336,336,44,577]},
        'camera_input': '512x512 legacy observation PNG is flipped left-right to upright, then Lanczos resized to 336x336; MP4 is lossy H.264.',
        'columns': COLUMNS,
        'blocks': [[b[k] for k in COLUMNS] for b in blocks],
        'source_refs': ['src/gpt_field/agent_speedup_v1_video.py:193-238',
                        'src/gpt_field/video.py:227-243',
                        'src/gpt_field/agent_speedup_v2_video.py:193-217'],
    }
    (OUTPUT/'frame_mapping.json').write_text(json.dumps(metadata,separators=(',',':'))+'\n')
    frames = [{'state_index': 0, 'source_frame_index': 0, 'source_time_s': 0.,
               'source_event': 'observation', 'action_index': None}]
    frames += [{'state_index': b['state_index'], 'source_frame_index': b['frame_start'],
                'source_time_s': b['frame_start']/FPS, 'source_event': 'executed',
                'action_index': b['action_index']} for b in blocks if b['event']=='executed']
    assert [x['state_index'] for x in frames] == list(range(844))
    control = {'source_video': str(RUN/'trial/episode.mp4'), 'source_fps': FPS,
               'control_state_count': len(frames),
               'state_convention': 'State 0 is initial settled observation; state n is after control action n-1. Diagnostic release/settle is excluded.',
               'camera_crops_xyxy_exclusive': CROPS_XYXY, 'control_state_frames': frames}
    (OUTPUT/'control_state_frames.json').write_text(json.dumps(control,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in metadata.items() if k!='blocks'},indent=2))


if __name__ == '__main__': main()
