"""Render archived V3 map snapshots and synchronized RGB; no simulator imports."""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1920, 1080, 10
BG, PANEL = '#0c1420', '#142132'
TEXT, MUTED, GRID = '#e5edf7', '#94a8c0', '#2a3b50'
PATH, GOAL, CURRENT, BOX = '#ec78b7', '#ffd369', '#ffffff', '#ff9857'
FONT_ROOT = Path('/usr/share/fonts/truetype/dejavu')
FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(str(FONT_ROOT / (
            'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf')), size)
    return FONTS[key]


def load_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def xyz(values):
    values = np.asarray(values, dtype=float)
    return values.reshape(-1, 3) if values.size else np.empty((0, 3))


def fit_text(draw, text, position, max_width, size=23, fill=TEXT):
    while draw.textlength(text, font=font(size)) > max_width and len(text) > 3:
        text = text[:-4] + '...'
    draw.text(position, text, font=font(size), fill=fill)


def plot(draw, rect, axes, bounds, snapshot, state):
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=16, fill=PANEL)
    title = ''.join('XYZ'[a] for a in axes)
    draw.text((x0+18, y0+14), title + ' / WORLD METRES', font=font(21, True), fill=TEXT)
    boxes = snapshot.get('obstacle_boxes', [])
    draw.text((x0+18, y0+47), f'Recorded obstacles: {len(boxes)}', font=font(18), fill=BOX)
    centre = np.mean(bounds, axis=0)[list(axes)]
    span = max(float(np.max(np.diff(bounds, axis=0)[0, list(axes)])), .16)
    low, high = centre-span/2, centre+span/2
    left, right, top, bottom = x0+70, x1-25, y0+100, y1-60
    side = min(right-left, bottom-top)
    left += (right-left-side)/2
    top += (bottom-top-side)/2
    right, bottom = left+side, top+side

    def project(point):
        value = np.asarray(point)[list(axes)]
        uv = (value-low)/(high-low)
        return (float(left+uv[0]*side), float(bottom-uv[1]*side))

    for i in range(5):
        f = i/4
        px, py = left+side*f, bottom-side*f
        draw.line((px, top, px, bottom), fill=GRID, width=1)
        draw.line((left, py, right, py), fill=GRID, width=1)
        draw.text((px-22, bottom+9), f'{low[0]+span*f:.2f}', font=font(13), fill=MUTED)
        draw.text((x0+12, py-7), f'{low[1]+span*f:.2f}', font=font(13), fill=MUTED)
    draw.text(((left+right)/2-12, y1-25), 'XYZ'[axes[0]]+' m', font=font(15), fill=MUTED)
    draw.text((x0+12, top-25), 'XYZ'[axes[1]]+' m', font=font(15), fill=MUTED)

    for item in boxes:
        lo = item.get('lower_xyz_m')
        hi = item.get('upper_xyz_m')
        if lo is not None and hi is not None:
            a, b = project(lo), project(hi)
            draw.rectangle((min(a[0], b[0]), min(a[1], b[1]),
                            max(a[0], b[0]), max(a[1], b[1])), outline=BOX, width=3)
    for key, color, width in [('measured_path', PATH, 3), ('selected_path', GOAL, 3)]:
        points = xyz(snapshot.get(key, []))
        if len(points) > 1:
            draw.line([project(p) for p in points], fill=color, width=width)
    for pose in snapshot.get('keyposes', []):
        point = np.asarray(pose['position'], float)
        px, py = project(point)
        draw.rectangle((px-6, py-6, px+6, py+6), outline=GOAL, width=3)
        rotation = np.asarray(pose.get('rotation', np.eye(3)), float)
        for i, color in enumerate(('#f27c78', '#75ce90', '#7eaaf5')):
            draw.line([(px, py), project(point+rotation[:, i]*.035)], fill=color, width=3)
    px, py = project(state[:3])
    draw.ellipse((px-5, py-5, px+5, py+5), fill=CURRENT, outline=BG, width=1)


def run(root):
    trial = root/'trial'
    events = load_rows(trial/'episode.events.jsonl')
    traces = load_rows(trial/'trace.jsonl')
    actions = {r['step']:r for r in traces if r['event']=='action'}
    plans = {}
    ordinal = 0
    for row in load_rows(trial/'foundation_vision_harness_V3/events.jsonl'):
        if row['event']=='model_request': ordinal = row['ordinal']
        if row['event']=='model_plan': plans[ordinal] = row['answer']
    result = json.loads((root/'result.json').read_text())
    snapshots = [r['foundation_harness_planning'] for r in events]
    all_points = [xyz([r['state'][:3]]) for r in events]
    all_points += [xyz([p['position'] for p in s.get('keyposes', [])]) for s in snapshots]
    points = np.concatenate(all_points)
    bounds = np.array([points.min(axis=0)-.045, points.max(axis=0)+.045])
    labels = {'libero_10_t4':'T04  CAN OBSTACLE', 'libero_10_t8':'T08  BOTTLE OBSTACLE',
              'libero_10_t7':'T07  ORIGINAL'}
    title = labels.get(result['task'], result['task'])
    out = root/'visualizations'
    out.mkdir(exist_ok=True)
    dest = out/'obstacle_map.mp4'
    if dest.exists(): raise FileExistsError(dest)
    source = trial/'episode.mp4'
    decoder = subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(source),
        '-vf','crop=336:732:44:181','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    encoder = subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast',
        '-crf','20','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(dest)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    origin = events[0]['capture']['monotonic_s']
    poster_index = len(events)//3
    frame_count = 0
    try:
        for index, row in enumerate(events):
            if row['frame_start'] != index or row['frame_count'] != 1:
                raise ValueError('This renderer expects the archived one-frame-per-event format')
            raw = decoder.stdout.read(336*732*3)
            if len(raw) != 336*732*3: raise RuntimeError('Video shorter than saved frame index')
            cameras = Image.frombytes('RGB',(336,732),raw)
            canvas = Image.new('RGB',(W,H),BG)
            draw = ImageDraw.Draw(canvas)
            snap = row['foundation_harness_planning']
            step = row['step']
            terminal = row['event']=='terminal'
            online = result['online_wall_s'] if terminal else row['capture']['monotonic_s']-origin
            calls = result['agent_stats']['model_decisions'] if terminal else row['model_calls']
            plan = plans.get(calls,{})
            draw.text((30,22),title+'  |  V3 OBSTACLE MAP REPLAY',font=font(32,True),fill=TEXT)
            draw.text((30,72),'Archived RGB + exact recorded planning snapshots; no new model or simulator run',font=font(21),fill=MUTED)
            draw.text((35,127),'EXTERNAL RGB',font=font(20,True),fill=TEXT)
            canvas.paste(cameras.crop((0,0,336,336)).resize((360,360)),(35,160))
            draw.text((35,560),'WRIST RGB',font=font(20,True),fill=TEXT)
            canvas.paste(cameras.crop((0,396,336,732)).resize((360,360)),(35,593))
            for i,axes in enumerate(((0,1),(0,2),(1,2))):
                plot(draw,(430+490*i,135,905+490*i,660),axes,bounds,snap,row['state'])
            draw.rounded_rectangle((430,685,1885,987),radius=16,fill=PANEL)
            draw.text((453,704),f'Step {step}/{result["steps"]}     Online capture: {online:.1f} s     Astra requests: {calls}',font=font(24,True),fill=TEXT)
            gap=abs(row['state'][6]-row['state'][7])*1000
            command='CLOSE' if row.get('action') and row['action'][-1]>0 else 'OPEN'
            draw.text((453,747),f'State: {snap.get("status")}    Active: {snap.get("stage") or "none"}    Gripper: {command} / {gap:.1f} mm',font=font(21),fill=MUTED)
            poses=snap.get('keyposes',[])
            target=', '.join(f'{v:+.3f}' for v in poses[0]['position'])+' m' if poses else 'none'
            draw.text((453,785),'Active target XYZ: '+target,font=font(23),fill=GOAL)
            why=plan.get('why','Awaiting first Astra response')
            fit_text(draw,'Astra: '+why,(453,826),1405,size=19)
            collision=actions.get(step,{}).get('evaluator',{}).get('collision',{}).get('counts',{}).get('forbidden_contact_points',0)
            raw_success=actions.get(step,{}).get('evaluator',{}).get('raw_success',False)
            status=f'Forbidden contact samples: {collision}    Raw success: {str(raw_success).lower()}'
            if terminal: status+=f'    Strict terminal: {str(result["strict_success_at_terminal"]).lower()}'
            draw.text((453,863),status,font=font(20),fill=TEXT)
            draw.text((453,906),'Pink: measured EEF path   |   Yellow: planned route / target   |   White: current EEF',font=font(19),fill=TEXT)
            draw.text((453,945),'Small RGB axes at target: target orientation. DA3 measurements were advisory, not an obstacle map.',font=font(17),fill=MUTED)
            draw.text((30,1008),'EMPTY MAP = NO SUPPLIED OBSTACLE GEOMETRY; IT DOES NOT MEAN THE SCENE IS OBSTACLE-FREE.',font=font(22,True),fill=BOX)
            draw.text((30,1043),'Sampled playback at 10 fps; model waits compressed. Online capture time is shown above. No obstacle geometry was fabricated.',font=font(17),fill=MUTED)
            if index==poster_index: canvas.save(out/'obstacle_map_preview.png')
            encoder.stdin.write(canvas.tobytes());frame_count+=1
            if terminal:
                canvas.save(out/'obstacle_map_terminal.png')
                for _ in range(19): encoder.stdin.write(canvas.tobytes());frame_count+=1
            if index and index%250==0: print(json.dumps({'run':root.name,'rendered':index,'source_frames':len(events)}),flush=True)
        if decoder.stdout.read(1): raise RuntimeError('Video has additional unindexed frames')
    finally:
        encoder.stdin.close()
        decoder.stdout.close()
    decode_error=decoder.stderr.read().decode();encode_error=encoder.stderr.read().decode()
    if decoder.wait() or encoder.wait(): raise RuntimeError(decode_error+encode_error)
    manifest={'run':root.name,'source_video':str(source),'source_video_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'source_events':str(trial/'episode.events.jsonl'),'source_events_sha256':hashlib.sha256((trial/'episode.events.jsonl').read_bytes()).hexdigest(),
        'output':str(dest),'width':W,'height':H,'fps':FPS,'source_frames':len(events),'output_frames':frame_count,
        'duration_s':frame_count/FPS,'terminal_extra_frames':19,'actual_obstacle_box_counts':sorted({len(s['obstacle_boxes']) for s in snapshots}),
        'map_source':'recorded Astra obstacle boxes only','keyposes':'exact active resolved snapshot; no inferred future relative poses',
        'rgb_source':'continuous crops from original video, no simulator replay','new_model_calls':0,'new_simulator_steps':0,
        'playback_is_wall_clock':False,'source_files_modified':False}
    (out/'obstacle_map.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('run',type=Path)
    run(parser.parse_args().run.resolve())
