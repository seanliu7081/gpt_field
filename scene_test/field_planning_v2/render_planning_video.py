"""Render sealed drawer planning paths, without planning or dynamics execution.

Actual native robot meshes move through joint-linear saved paths. The adjacent
Astra field slice is a visualization, not a substitute for whole-body checks.
DepthPro refusals are shown as text; no nonexistent DepthPro path is animated.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'field_planning'
sys.path.insert(0, str(OLD))
from robot_geometry import GroundTruthEvaluator, RobotGeometry
from run_planning import sha256, write_json
from diagnose_failures import verify_run
from planning_fields import AstraEnvironmentField
from planning_video_panels import FieldPanel

WIDTH, HEIGHT, FPS = 1280, 900, 15
MOVING_FRAMES, HOLD_FRAMES = 61, 7
BG, TEXT, MUTED = '#101922', '#edf3f8', '#a9bac8'
GREEN, RED, ORANGE = '#48d39d', '#ff665e', '#ffc36c'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def font(size):
    return ImageFont.truetype(FONT, size)


def interpolate_saved_path(path, frame_count=MOVING_FRAMES):
    """Allocate fixed frames to original edges, including every saved vertex."""
    q = np.asarray(path, float)
    if q.ndim != 2 or q.shape[1] != 7 or len(q) < 2 or not np.isfinite(q).all():
        raise ValueError('finite saved Nx7 path with at least two vertices required')
    lengths = np.linalg.norm(np.diff(q, axis=0), axis=1)
    if frame_count < len(q):
        raise ValueError('not enough frames to preserve every saved waypoint')
    extra = frame_count - len(q)
    weights = lengths / lengths.sum() if lengths.sum() else np.ones(len(lengths))/len(lengths)
    ideal = weights * extra
    allocated = np.floor(ideal).astype(int)
    remain = extra - allocated.sum()
    for i in np.argsort(-(ideal-allocated), kind='stable')[:remain]:
        allocated[i] += 1
    intervals = allocated + 1
    frames, vertex_frames = [q[0]], [0]
    for a, b, n in zip(q, q[1:], intervals):
        frames.extend(a + (b-a)*t for t in np.linspace(0, 1, n+1)[1:])
        frames[-1] = b.copy()
        vertex_frames.append(len(frames)-1)
    assert len(frames) == frame_count
    return np.asarray(frames), vertex_frames


class Encoder:
    def __init__(self, path):
        self.path = Path(path)
        self.partial = self.path.with_suffix('.partial.mp4')
        if self.path.exists() or self.partial.exists():
            raise FileExistsError(self.path)
        self.process = subprocess.Popen([
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s:v', f'{WIDTH}x{HEIGHT}',
            '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
            '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(self.partial)], stdin=subprocess.PIPE)
        self.frames = 0

    def append(self, image):
        array = np.asarray(image.convert('RGB'), dtype=np.uint8)
        if array.shape != (HEIGHT, WIDTH, 3):
            raise ValueError('wrong video frame shape')
        self.process.stdin.write(array.tobytes())
        self.frames += 1

    def close(self):
        self.process.stdin.close()
        if self.process.wait() != 0:
            raise RuntimeError('ffmpeg encoding failed')
        self.partial.rename(self.path)


def card(title, lines):
    image = Image.new('RGB', (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    draw.text((70, 140), title, fill=TEXT, font=font(38))
    for i, line in enumerate(lines):
        draw.text((70, 245+i*60), line, fill=MUTED, font=font(24))
    return image


def project(points, calibration, pixels=600):
    p = np.asarray(points, float)
    T = np.asarray(calibration['T_camera_cv_from_world'])
    p = p @ T[:3, :3].T + T[:3, 3]
    K = np.asarray(calibration['K'], float).copy()
    scale = pixels/calibration['width']
    K[0, 0] *= scale; K[1, 1] *= scale
    K[0, 2] = (K[0, 2]+.5)*scale-.5
    K[1, 2] = (K[1, 2]+.5)*scale-.5
    xy = p @ K.T
    return xy[:, :2]/xy[:, 2:3], p[:, 2] > 0


def scene_overlay(rgb, path_xyz, current_xyz, calibration, contacts):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    xy, valid = project(path_xyz, calibration)
    for a, b, ok in zip(xy, xy[1:], valid[:-1] & valid[1:]):
        if ok:
            draw.line([tuple(a), tuple(b)], fill='#e0c983', width=3)
    now, visible = project(np.asarray(current_xyz)[None], calibration)
    if visible[0]:
        x, y = now[0]; draw.ellipse((x-6, y-6, x+6, y+6), fill=GREEN, outline='white', width=2)
    for pos in contacts:
        uv, v = project(np.asarray(pos)[None], calibration)
        if v[0]:
            x, y = uv[0]
            draw.ellipse((x-10, y-10, x+10, y+10), outline=RED, width=3)
    return image


def composed_frame(scene, panel, record, paired_dp, frame_index, current_check, index):
    image = Image.new('RGB', (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)
    state = 'OPEN' if 'open' in record['snapshot'] else 'CLOSED'
    algorithm = 'RRT CONNECT' if record['reason'] == 'rrt_connect' else 'DIRECT JOINT PATH'
    draw.text((24, 15), f'DRAWER {state}  |  {record["case_id"]}  |  seed {record["seed"]}', fill=TEXT, font=font(27))
    draw.text((24, 55), f'Astra: {algorithm}  |  saved query {index+1}/40  |  planning {record["planning_seconds"]:.2f} s', fill=MUTED, font=font(20))
    draw.text((24, 86), 'Original LIBERO scene + robot | rendered saved path', fill=TEXT, font=font(18))
    image.paste(scene, (24, 112))
    image.paste(panel, (656, 112))
    draw.rounded_rectangle((24, 727, 624, 846), radius=8, fill='#1d2b36')
    passed = record['truth']['valid_at_all_samples']
    title = 'PATH: PASSED GT SAMPLES' if passed else 'PATH: GT COLLISION FOUND'
    draw.text((38, 738), title, fill=GREEN if passed else RED, font=font(23))
    current = 'current pose clear' if current_check['valid'] else 'CURRENT POSE COLLIDES'
    draw.text((38, 772), f'{current}  |  replay {frame_index+1}/{MOVING_FRAMES}', fill=MUTED if current_check['valid'] else RED, font=font(18))
    if current_check['contacts']:
        c = current_check['contacts'][0]
        contact = f'{c["geom1"]} / {c["geom2"]}'
        draw.text((38, 802), contact, fill=RED, font=font(14))
    else:
        draw.text((38, 802), f'Full path GT samples: {record["truth"]["sample_count"]} | finite checks', fill=MUTED, font=font(18))
    draw.rounded_rectangle((656, 727, 1256, 846), radius=8, fill='#302b24')
    draw.text((670, 738), 'DEPTHPRO: NO PATH RETURNED', fill=ORANGE, font=font(23))
    counts = paired_dp['goal_check']['field_points']
    unknown = 100*counts['unknown']/counts['points']
    occupied = 100*counts['occupied']/counts['points']
    draw.text((670, 773), f'Target robot samples: {unknown:.1f}% unknown, {occupied:.1f}% occupied', fill=TEXT, font=font(17))
    draw.text((670, 803), 'Same target; no DepthPro motion is synthesized.', fill=MUTED, font=font(18))
    draw.text((24, 867), 'KINEMATIC VISUALIZATION | fixed presentation speed, not control time | no dynamics or task execution', fill=MUTED, font=font(18))
    return image


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                             '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames,duration',
                             '-of', 'json', str(path)], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)['streams'][0]


def concatenate(paths, output, list_path):
    # All paths are generated below this experiment, and cannot contain quotes.
    if any("'" in str(p) or '\n' in str(p) for p in paths):
        raise ValueError('unsupported output path')
    list_path.write_text(''.join(f"file '{Path(p).resolve()}'\n" for p in paths))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
                    '-f', 'concat', '-safe', '0', '-i', str(list_path), '-c', 'copy',
                    '-movflags', '+faststart', str(output)], check=True)


def render(output):
    started = time.perf_counter()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    clips = output/'clips'; clips.mkdir(exist_ok=False)
    previews = output/'previews'; previews.mkdir()
    plan_manifest, consumed = verify_run(HERE/'runs/planning_001')
    _, evaluation_inputs = verify_run(HERE/'runs/evaluation_001')
    consumed.update(evaluation_inputs)
    for source in (Path(__file__).resolve(), HERE/'planning_video_panels.py', Path(FONT)):
        consumed[str(source)] = sha256(source)
    metrics = json.loads((HERE/'runs/evaluation_001/metrics.json').read_text())
    rows = [r for r in metrics['records'] if r['method'] == 'astra']
    dp = {(r['snapshot'], r['case_id'], r['seed']): r for r in metrics['records'] if r['method'] == 'depthpro'}
    timeline, poses = [], []
    prepared = {}
    for snapshot in metrics['settings']['snapshots']:
        robot = RobotGeometry(Path(plan_manifest['preparation'])/snapshot, sample_spacing_m=None)
        truth = GroundTruthEvaluator(HERE.parent/'rgb_scene/references_004/snapshots'/snapshot/'evaluator')
        truth.model.vis.global_.offwidth = 640; truth.model.vis.global_.offheight = 640
        renderer = mujoco.Renderer(truth.model, height=600, width=600)
        option = mujoco.MjvOption(); option.geomgroup[0] = 0; option.sitegroup[:] = 0
        field = AstraEnvironmentField(snapshot)
        panel = FieldPanel(field, robot, size=600)
        calibration = json.loads((truth.source/'calibration.json').read_text())
        prepared[snapshot] = robot, truth, renderer, option, field, panel, calibration
        consumed.update(truth.provenance()['consumed_sha256'])
    for index, record in enumerate(rows):
        key = record['snapshot'], record['case_id'], record['seed']
        paired = dp[key]
        if record['status'] != 'planned' or paired['path_q']:
            raise ValueError('unexpected saved experiment status')
        robot, truth, renderer, option, field, panel, calibration = prepared[record['snapshot']]
        frames_q, vertices = interpolate_saved_path(record['path_q'])
        eef_xyz = np.array([robot.eef_pose(q)[:3, 3] for q in frames_q])
        clip_name = f'{index:02d}_{record["snapshot"]}_{record["case_id"]}_s{record["seed"]}.mp4'
        encoder = Encoder(clips/clip_name)
        frame_collisions = []
        for frame_index, (q, eef) in enumerate(zip(frames_q, eef_xyz)):
            current = truth.check(q, distances=False)
            if not current['valid']:
                frame_collisions.append(frame_index)
            renderer.update_scene(truth.data, camera='agentview', scene_option=option)
            contact_positions = [c.pos.copy() for c in truth.data.contact
                                 if (int(c.geom1) in truth.robot_ids or int(c.geom2) in truth.robot_ids)
                                 and float(c.dist) < -1e-7]
            scene = scene_overlay(renderer.render(), eef_xyz, eef, calibration, contact_positions)
            field_image = panel.render(q, eef_xyz, progress_index=frame_index)
            image = composed_frame(scene, field_image, record, paired, frame_index, current, index)
            repeats = 1 + (HOLD_FRAMES if frame_index in (0, len(frames_q)-1) else 0)
            for _ in range(repeats):
                encoder.append(image)
            if frame_index in (0, len(frames_q)//2, len(frames_q)-1):
                if record['seed'] == 17 and record['case_id'] in ('base_negative', 'wrist_negative', 'wrist_positive'):
                    image.save(previews/f'{index:02d}_{frame_index:03d}.png')
        encoder.close()
        row = {'index': index, 'snapshot': key[0], 'case_id': key[1], 'seed': key[2],
               'clip': str((clips/clip_name).relative_to(output)), 'frames': encoder.frames,
               'duration_s': encoder.frames/FPS, 'saved_vertex_frame_indices': vertices,
               'saved_path_q': record['path_q'], 'frame_contact_indices': frame_collisions,
               'saved_truth_valid': record['truth']['valid_at_all_samples'],
               'planning_seconds': record['planning_seconds'], 'planner_reason': record['reason'],
               'depthpro_status': paired['status'], 'depthpro_reason': paired['reason']}
        if row['saved_truth_valid'] and frame_collisions:
            raise ValueError('new displayed pose contradicts saved finite-sample clearance; review before publication')
        timeline.append(row); poses.append(frames_q)
        write_json(output/'timeline.json', timeline)
        print(json.dumps({'clip': index+1, 'total': len(rows), 'case': key,
                          'gt_valid': row['saved_truth_valid'], 'display_collision_frames': len(frame_collisions)}), flush=True)
    for robot, truth, renderer, option, field, panel, calibration in prepared.values():
        consumed.update(field.provenance(verify=True)['consumed_sha256'])
        renderer.close()
    intro = card('DRAWER003 | Planning replay', [
        'Original LIBERO robot + saved Astra paths + predicted field slice',
        'Left: original scene. Right: Astra geometry field at gripper height.',
        'DepthPro status is paired with the same requested joint target.',
        'Only returned paths are replayed; no search-tree history was saved.',
        '5 seconds per query is presentation speed, not robot control time.'])
    outro = card('Result | Starts fixed; geometry limits remain', [
        'Astra: 40 returned paths; 32 passed GT samples; 8 collide.',
        'DepthPro: 40 target refusals; missing or occupied scene support.',
        'Two Astra RRT detours passed GT checks.',
        'Static snapshots and fixed fingers; no drawer manipulation rollout.',
        'Finite path checks do not certify continuous collision freedom.'])
    for name, image in (('intro', intro), ('outro', outro)):
        writer = Encoder(clips/f'{name}.mp4')
        for _ in range(3*FPS): writer.append(image)
        writer.close()
    picks = [
        ('drawer003_open_step552', 'base_negative', 17),
        ('drawer003_closed_step60', 'base_positive', 17),
        ('drawer003_open_step552', 'wrist_negative', 17),
        ('drawer003_closed_step60', 'wrist_positive', 17)]
    lookup = {(r['snapshot'], r['case_id'], r['seed']): output/r['clip'] for r in timeline}
    selected = [lookup[k] for k in picks]
    concatenate([clips/'intro.mp4', *selected, clips/'outro.mp4'], output/'planning_highlights.mp4', output/'highlights_concat.txt')
    concatenate([clips/'intro.mp4', *(output/r['clip'] for r in timeline), clips/'outro.mp4'], output/'planning_all_queries.mp4', output/'all_concat.txt')
    np.savez_compressed(output/'displayed_joint_poses.npz', q=np.asarray(poses))
    intact = {p: sha256(p) == h for p, h in consumed.items()}
    if not all(intact.values()):
        raise ValueError('source/input changed during video generation')
    metadata = {name: probe(output/name) for name in ('planning_highlights.mp4', 'planning_all_queries.mp4')}
    expected = {'planning_highlights.mp4': 2*3*FPS+len(picks)*75,
                'planning_all_queries.mp4': 2*3*FPS+len(rows)*75}
    for name, meta in metadata.items():
        if int(meta['nb_frames']) != expected[name] or meta['r_frame_rate'] != '15/1':
            raise ValueError('encoded duration/frame mismatch')
    report = {'status': 'complete', 'elapsed_s': time.perf_counter()-started,
              'consumed_sha256': consumed, 'inputs_unchanged': intact,
              'videos': metadata, 'highlight_cases': [list(k) for k in picks],
              'all_saved_astra_queries_shown': len(rows), 'depthpro_paths_synthesized': 0,
              'original_waypoints_preserved': True, 'dynamics_steps': 0,
              'display_speed': '75 frames / 15 fps = 5 seconds per query; not control or planning duration',
              'field_slice': 'Astra environment field XY at instantaneous EEF world Z; saved EEF path projected XY',
              'camera': 'native agentview Renderer upright; geomgroup0 hidden for duplicate collision-visualization removal only',
              'scene_state': 'original frozen snapshot objects and fingers reset for every displayed joint pose',
              'artifacts_sha256': {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}}
    write_json(output/'manifest.json', report)
    print(json.dumps({'status': 'complete', 'videos': metadata, 'output': str(output)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    render(args.output)
