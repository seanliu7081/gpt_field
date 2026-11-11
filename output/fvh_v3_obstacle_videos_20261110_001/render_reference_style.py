"""Reference-style 3D map replay from archived V3 frames and snapshots only."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from render_obstacle_video import font, fit_text, load_rows, xyz
from reference_style_geometry import draw_isometric

W,H,FPS=1920,1080,20
BG='#0e1521'; PANEL='#151f2d'; TEXT='#e9f0f7'; MUTED='#9badc1'; GRID='#2f4052'
ORANGE='#ffa74f'; CYAN='#4bd1ef'; PINK='#ee73ba'; GOLD='#ffdb6d'; RED='#ff6968'


def flat_panel(draw, rect, axes, bounds, snap, state, contacts):
    lo,hi=bounds
    x0,y0,x1,y1=rect
    def project(p):
        return (x0+(p[axes[0]]-lo[axes[0]])/(hi[axes[0]]-lo[axes[0]])*(x1-x0),
                y1-(p[axes[1]]-lo[axes[1]])/(hi[axes[1]]-lo[axes[1]])*(y1-y0))
    draw.rectangle(rect,outline=GRID,width=1)
    for i in range(4):
        f=i/3;px=x0+(x1-x0)*f;py=y1-(y1-y0)*f
        draw.line((px,y0,px,y1),fill=GRID);draw.line((x0,py,x1,py),fill=GRID)
        draw.text((px,y1+8),f'{lo[axes[0]]+(hi[axes[0]]-lo[axes[0]])*f:.2f}',font=font(15),fill=MUTED,anchor='mt')
        draw.text((x0-10,py),f'{lo[axes[1]]+(hi[axes[1]]-lo[axes[1]])*f:.2f}',font=font(15),fill=MUTED,anchor='rm')
    draw.text(((x0+x1)/2,y1+34),'XYZ'[axes[0]]+' (m)',font=font(18),fill=MUTED,anchor='mt')
    draw.text((x0-54,(y0+y1)/2),'XYZ'[axes[1]],font=font(18),fill=MUTED,anchor='mm')
    for box in snap.get('obstacle_boxes',[]):
        a,b=project(box['lower_xyz_m']),project(box['upper_xyz_m'])
        draw.rectangle((min(a[0],b[0]),min(a[1],b[1]),max(a[0],b[0]),max(a[1],b[1])),outline=ORANGE,width=2)
    for name,color,width in [('measured_path',PINK,3),('selected_path',CYAN,4)]:
        points=snap.get(name,[])
        if len(points)>1:draw.line([project(p) for p in points],fill=color,width=width)
    for pose in snap.get('keyposes',[]):
        x,y=project(pose['position']);draw.line((x-8,y,x+8,y),fill=GOLD,width=3);draw.line((x,y-8,x,y+8),fill=GOLD,width=3)
    x,y=project(state[:3]);draw.ellipse((x-6,y-6,x+6,y+6),fill=TEXT,outline=BG,width=1)
    for p in contacts:
        x,y=project(p);draw.ellipse((x-10,y-10,x+10,y+10),outline=RED,width=2);draw.ellipse((x-2,y-2,x+2,y+2),fill=RED)


def run(root, replace=False):
    trial=root/'trial';events=load_rows(trial/'episode.events.jsonl')
    result=json.loads((root/'result.json').read_text())
    trace=load_rows(trial/'trace.jsonl');actions={x['step']:x for x in trace if x['event']=='action'}
    plans={};maps={0:{'revision':0,'clearance':0.}};ordinal=revision=0;clearance=0.
    failures={};contacts={};contact_delta={};last=0;seen=set()
    for row in load_rows(trial/'foundation_vision_harness_V3/events.jsonl'):
        if row['event']=='model_request':ordinal=row['ordinal']
        if row['event']=='model_plan':
            a=row['answer'];plans[ordinal]=a
            if a.get('obstacle_map') is not None:
                revision+=1;clearance=a['obstacle_map']['clearance_m']
            maps[ordinal]={'revision':revision,'clearance':clearance}
        if row['event']=='segment_result' and row['outcome']['status']!='succeeded':
            failures[row['step']]=row['outcome']['reason']
    for step,row in actions.items():
        c=row['evaluator']['collision'];n=c['counts']['forbidden_contact_points']
        contact_delta[step]=n-last;last=n
        for kind,hit in c.get('first_by_class',{}).items():
            if kind in c['forbidden_classes'] and kind not in seen:
                seen.add(kind);contacts.setdefault(step,[]).append({'point':hit['contact_pos'],'kind':kind})
    snapshots=[r['foundation_harness_planning'] for r in events]
    all_points=[xyz([r['state'][:3]]) for r in events]
    all_points += [xyz([p['position'] for p in s.get('keyposes',[])]) for s in snapshots]
    all_points += [xyz([h['point'] for h in hs]) for hs in contacts.values()]
    pts=np.concatenate(all_points);bounds=np.array([np.floor((pts.min(0)-.035)*20)/20,np.ceil((pts.max(0)+.035)*20)/20])
    title={'libero_10_t4':'T04 - Can obstacle map replay','libero_10_t8':'T08 - Bottle obstacle map replay','libero_10_t7':'T07 - Original task map replay'}[result['task']]
    out=root/'visualizations';out.mkdir(exist_ok=True);dest=out/'obstacle_map_3d.mp4'
    if dest.exists() and not replace:raise FileExistsError(dest)
    encoded=out/'obstacle_map_3d.partial.mp4'
    decoder=subprocess.Popen(['ffmpeg','-v','error','-threads','1','-i',str(trial/'episode.mp4'),'-vf','crop=336:732:44:181','-threads','1','-f','rawvideo','-pix_fmt','rgb24','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    encoder=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','pipe:0','-an','-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p','-threads','1','-movflags','+faststart',str(encoded)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    origin=events[0]['capture']['monotonic_s'];frames=states=0;previous_calls=0;preview=False;frame_index=[]
    try:
        for index,row in enumerate(events):
            raw=decoder.stdout.read(336*732*3)
            if len(raw)!=336*732*3:raise RuntimeError('Source frame missing')
            # Decision-boundary states carry the active recorded target and path.
            # Ordinary after-action duplicates are omitted; their next boundary has the same RGB.
            if row['event'] not in ('observation','agent_action','terminal'):continue
            step=row['step'];terminal=row['event']=='terminal';snap=row['foundation_harness_planning'];state=row['state']
            calls=result['agent_stats']['model_decisions'] if terminal else row['model_calls']
            online=result['online_wall_s'] if terminal else row['capture']['monotonic_s']-origin
            mapinfo=maps.get(calls,maps[0]);plan=plans.get(calls,{})
            archived_contacts=[h['point'] for s,hits in contacts.items() if s<=step for h in hits]
            current_hits=contacts.get(step,[]);delta=contact_delta.get(step,0)
            c=actions.get(step,{}).get('evaluator',{}).get('collision',{}).get('counts',{}).get('forbidden_contact_points',0)
            cameras=Image.frombytes('RGB',(336,732),raw);im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
            d.text((32,22),title,font=font(36),fill=TEXT)
            d.text((32,82),'Recorded states at 0.1 s/state; model waits omitted; key events paused',font=font(20),fill=MUTED)
            d.text((1888,28),f'Control step {step:04d} / {result["steps"]}',font=font(27),fill=TEXT,anchor='rt')
            d.text((1888,80),f'Online {online:.1f} s  |  Map revision {mapinfo["revision"]}  |  Boxes {len(snap["obstacle_boxes"])}',font=font(20),fill=MUTED,anchor='rt')
            d.line((32,124,1888,124),fill=GRID)
            d.text((32,143),'External camera',font=font(23),fill=TEXT)
            d.text((32,603),'Wrist camera',font=font(23),fill=TEXT)
            im.paste(cameras.crop((0,0,336,336)).resize((400,400),Image.Resampling.LANCZOS),(32,180))
            im.paste(cameras.crop((0,396,336,732)).resize((400,400),Image.Resampling.LANCZOS),(32,640))
            d.text((485,151),'Obstacle map - World coordinates',font=font(25),fill=TEXT)
            d.text((1410,151),'Top view - XY projection',font=font(24),fill=TEXT)
            d.text((1410,566),'Side view - XZ projection',font=font(24),fill=TEXT)
            draw_isometric(d,(475,227,1372,906),bounds,snap,state,archived_contacts)
            flat_panel(d,(1450,245,1872,502),(0,1),bounds,snap,state,archived_contacts)
            flat_panel(d,(1450,637,1872,880),(0,2),bounds,snap,state,archived_contacts)
            for x,label,col,kind in [(490,'Recorded boxes',ORANGE,'box'),(730,'Planned path',CYAN,'line'),(965,'Measured path',PINK,'line'),(1210,'Target',GOLD,'plus'),(1400,'EEF',TEXT,'dot'),(1540,'First recorded contacts',RED,'ring')]:
                y=198
                if kind=='box':d.rectangle((x,y-7,x+22,y+7),outline=col,width=2)
                elif kind=='line':d.line((x,y,x+25,y),fill=col,width=3)
                elif kind=='plus':d.line((x,y,x+22,y),fill=col,width=2);d.line((x+11,y-10,x+11,y+10),fill=col,width=2)
                elif kind=='dot':d.ellipse((x+6,y-5,x+16,y+5),fill=col)
                else:d.ellipse((x+4,y-8,x+20,y+8),outline=col,width=2)
                d.text((x+33,y-13),label,font=font(17),fill=MUTED)
            heading='Recorded obstacle boxes: 0 - scene geometry was not supplied';color=MUTED
            detail='Astra: '+plan['why'] if plan.get('why') else 'Initial recorded observation'
            if delta:heading=f'Forbidden contact recorded this action: +{delta} samples';color=RED
            if step in failures:heading='Local feedback: '+failures[step].replace('_',' ');color=ORANGE
            if current_hits:heading='First recorded contact: '+current_hits[0]['kind'].replace('_',' ');color=RED
            if terminal:
                heading=f'Autonomous finish | raw {str(result["raw_success_at_terminal"]).lower()} | strict terminal {str(result["strict_success_at_terminal"]).lower()} | after hold {str(result["strict_success_after_same_gripper_hold"]).lower()}'
                color=GOLD
            d.rectangle((472,940,1888,1017),fill=PANEL)
            fit_text(d,heading,(490,945),1370,size=24,fill=color)
            fit_text(d,detail,(490,983),1370,size=18,fill=MUTED)
            x0,x1,y=490,1870,1032;total=result['steps']
            d.line((x0,y,x1,y),fill=GRID,width=5);d.line((x0,y,x0+(x1-x0)*step/total,y),fill=CYAN,width=5)
            for s,n in contact_delta.items():
                if n:xx=x0+(x1-x0)*s/total;d.line((xx,y-5,xx,y+5),fill=RED,width=2)
            for s in failures:
                xx=x0+(x1-x0)*s/total;d.line((xx,y-8,xx,y+8),fill=ORANGE,width=2)
            fit_text(d,f'Empty map is not an obstacle-free scene. Red timeline: contact samples ({c} so far); orange: local interruptions.',(490,1048),1390,size=16,fill=MUTED)
            count=2
            if index==0:count=30
            if calls!=previous_calls:count=max(count,12)
            if step in failures:count=max(count,20)
            if current_hits:count=max(count,30)
            if terminal:count=60
            if not preview and step>=total//3 and snap.get('keyposes'):
                im.save(out/'obstacle_map_3d_preview.png');preview=True
            if current_hits:im.save(out/f'obstacle_map_3d_contact_{step:04d}.png')
            if terminal:im.save(out/'obstacle_map_3d_terminal.png')
            pixels=im.tobytes()
            for _ in range(count):encoder.stdin.write(pixels)
            frame_index.append({'source_frame':index,'step':step,'event':row['event'],'output_start':frames,'output_count':count,'online_s':online,'model_calls':calls})
            frames+=count;states+=1;previous_calls=calls
            if states%150==0:print(json.dumps({'run':root.name,'states_rendered':states}),flush=True)
        if decoder.stdout.read(1):raise RuntimeError('Unindexed source frames')
    finally:
        encoder.stdin.close();decoder.stdout.close()
    de=decoder.stderr.read().decode();ee=encoder.stderr.read().decode();dc=decoder.wait();ec=encoder.wait()
    if dc or ec:raise RuntimeError(de+ee)
    encoded.replace(dest)
    manifest={'video':str(dest),'reference_style':'central isometric 3D + right XY/XZ + left cameras + contact timeline',
        'source_video':str(trial/'episode.mp4'),'source_video_sha256':hashlib.sha256((trial/'episode.mp4').read_bytes()).hexdigest(),
        'fps':FPS,'frames':frames,'duration_s':frames/FPS,'recorded_states_rendered':states,'width':W,'height':H,
        'playback':'0.1s per selected recorded state plus event holds; model waits omitted; online time displayed',
        'obstacle_source':'recorded Astra boxes only','actual_box_counts':sorted({len(s['obstacle_boxes']) for s in snapshots}),
        'contact_source':'first_by_class real recorded contact positions, retained as historical markers; not inferred current contacts',
        'timeline_source':'positive deltas in cumulative forbidden contact samples and recorded segment failure steps',
        'new_model_calls':0,'new_simulator_steps':0,'original_files_modified':False}
    (out/'obstacle_map_3d.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'obstacle_map_3d.frames.json').write_text(json.dumps(frame_index,separators=(',',':'))+'\n')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    p.add_argument('--replace-generated',action='store_true')
    args=p.parse_args();run(args.run.resolve(),replace=args.replace_generated)
