#!/usr/bin/env python3
"""Recorded-only obstacle-map movie. No simulator/model imports or calls."""
from __future__ import annotations
import argparse, json, math, subprocess
from pathlib import Path
from functools import lru_cache
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

ROOT=Path(__file__).resolve().parent
RUN=ROOT.parents[2]/'runs/agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001'
W,H=1920,1080
BG=(14,21,33); PANEL=(21,31,45); FG=(233,240,247); MUTED=(155,173,193)
GRID=(47,64,82); ORANGE=(255,167,79); CYAN=(75,209,239); PINK=(238,115,186); GOLD=(255,219,109); RED=(255,105,104)
FONT='/tmp/t04-browser-fonts/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
@lru_cache(None)
def font(size): return ImageFont.truetype(FONT,size,index=2)
def txt(d,xy,s,size=22,fill=FG,anchor=None): d.text(xy,s,font=font(size),fill=fill,anchor=anchor)
def dash(d,a,b,fill,width=2,on=7,off=5):
 a=np.array(a,float);b=np.array(b,float);v=b-a;n=np.linalg.norm(v)
 if n<.1:return
 for t in np.arange(0,n,on+off):d.line([tuple(a+v*t/n),tuple(a+v*min(n,t+on)/n)],fill=fill,width=width)
def cube(lo,hi):return np.array([[x,y,z] for z in (lo[2],hi[2]) for y in (lo[1],hi[1]) for x in (lo[0],hi[0])],float)
EDGES=[(i,j) for i in range(8) for j in range(i+1,8) if (i^j) in (1,2,4)]
FACES=((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))

class Charts:
 def __init__(self,bounds):
  self.lo,self.hi=np.array(bounds[0]),np.array(bounds[1])
  self.main=(480,204,1360,900)
  self.xy=(1450,226,1872,494);self.xz=(1450,637,1872,905)
  self.matrix=np.array([[.78,-.63,0],[.30,.38,-.875]])
  pr=cube(self.lo,self.hi)@self.matrix.T
  self.prlo=pr.min(0);self.prhi=pr.max(0)
  self.scale=min((self.main[2]-self.main[0]-130)/(self.prhi[0]-self.prlo[0]),(self.main[3]-self.main[1]-100)/(self.prhi[1]-self.prlo[1]))
  self.offset=np.array([(self.main[0]+self.main[2])/2,(self.main[1]+self.main[3])/2])-self.scale*(self.prlo+self.prhi)/2
 def project(self,p):return tuple((np.array(p)@self.matrix.T*self.scale+self.offset).tolist())
 def flat(self,p,axes,rect):
  a,b=axes;x0,y0,x1,y1=rect
  return x0+(p[a]-self.lo[a])/(self.hi[a]-self.lo[a])*(x1-x0),y1-(p[b]-self.lo[b])/(self.hi[b]-self.lo[b])*(y1-y0)
 def base(self,canvas):
  d=ImageDraw.Draw(canvas)
  txt(d,(485,155),'障碍地图 · 世界坐标',26)
  txt(d,(1410,155),'俯视 XY 投影',25)
  txt(d,(1410,566),'侧视 XZ 投影',25)
  # Stable floor grid; bounds are inferred from all recorded poses and boxes.
  for a in (0,1):
   for value in np.arange(math.ceil(self.lo[a]*10)/10,self.hi[a]+1e-7,.1):
    p=self.lo.copy();q=self.lo.copy();p[a]=q[a]=value;q[1-a]=self.hi[1-a]
    d.line([self.project(p),self.project(q)],fill=GRID,width=1)
  origin=self.lo.copy()
  for a,label in enumerate(('X','Y','Z')):
   q=origin.copy();q[a]=self.hi[a]
   d.line([self.project(origin),self.project(q)],fill=MUTED,width=2)
   px,py=self.project(q);txt(d,(px+6,py-5),label+' (m)',19,MUTED)
  for axes,rect in (((0,1),self.xy),((0,2),self.xz)):
   x0,y0,x1,y1=rect
   d.rectangle(rect,outline=GRID,width=1)
   for a in axes:
    ticks=np.linspace(self.lo[a],self.hi[a],4)
    for value in ticks:
     p=self.lo.copy();q=self.hi.copy();p[a]=q[a]=value
     u=self.flat(p,axes,rect);v=self.flat(q,axes,rect)
     d.line([u,v],fill=GRID,width=1)
     if a==axes[0]:txt(d,(u[0],y1+9),f'{value:.2f}',16,MUTED,'mt')
     else:txt(d,(x0-9,u[1]),f'{value:.2f}',16,MUTED,'rm')
   txt(d,((x0+x1)/2,y1+35),'XYZ'[axes[0]]+' (m)',18,MUTED,'mt')
   txt(d,(x0-54,(y0+y1)/2),'XYZ'[axes[1]],18,MUTED,'mm')
 def draw(self,canvas,boxes,clearance,trajectory,pose,goal,path,active,can_index=None):
  overlay=Image.new('RGBA',(W,H),(0,0,0,0));od=ImageDraw.Draw(overlay);labels=[]
  for n,box in enumerate(boxes):
   lo=np.array(box['lower_xyz_m']);hi=np.array(box['upper_xyz_m']);v=cube(lo,hi)
   projected=[self.project(p) for p in v]
   # Translucent cuboids, not a reconstructed physical mesh or SDF.
   for face in FACES:od.polygon([projected[i] for i in face],fill=ORANGE+(16,))
   for i,j in EDGES:od.line([projected[i],projected[j]],fill=ORANGE+(205,),width=2)
   ext=[self.project(p) for p in cube(lo-clearance,hi+clearance)]
   for i,j in EDGES:dash(od,ext[i],ext[j],ORANGE+(95,),1,6,6)
   tx,ty=self.project((lo+hi)/2+np.array([0,0,(hi[2]-lo[2])/2]))
   label=f'B{n+1}'+(' · 罐子估计' if n==can_index else '')
   labels.append(((tx,ty-18),label))
   for axes,rect in (((0,1),self.xy),((0,2),self.xz)):
    p=self.flat(lo,axes,rect);q=self.flat(hi,axes,rect)
    r=(min(p[0],q[0]),min(p[1],q[1]),max(p[0],q[0]),max(p[1],q[1]))
    od.rectangle(r,fill=ORANGE+(35,),outline=ORANGE+(220,),width=2)
    p=self.flat(lo-clearance,axes,rect);q=self.flat(hi+clearance,axes,rect)
    pp=[(p[0],p[1]),(q[0],p[1]),(q[0],q[1]),(p[0],q[1])]
    for i in range(4):dash(od,pp[i],pp[(i+1)%4],ORANGE+(110,),1)
  for pos,label in labels:
   bounds=od.textbbox(pos,label,font=font(18),anchor='mm')
   od.rectangle((bounds[0]-4,bounds[1]-2,bounds[2]+4,bounds[3]+2),fill=BG+(225,))
   txt(od,pos,label,18,ORANGE+(255,),'mm')
  canvas.alpha_composite(overlay)
  d=ImageDraw.Draw(canvas)
  for project in (self.project,lambda p:self.flat(p,(0,1),self.xy),lambda p:self.flat(p,(0,2),self.xz)):
   if len(trajectory)>1:d.line([project(p) for p in trajectory],fill=PINK,width=3)
   if path and active:
    pp=[project(p) for p in path]
    if len(pp)>1:d.line(pp,fill=CYAN,width=4)
    for x,y in pp:d.ellipse((x-3,y-3,x+3,y+3),fill=CYAN)
   if goal is not None:
    x,y=project(goal);d.line((x-8,y,x+8,y),fill=GOLD,width=3);d.line((x,y-8,x,y+8),fill=GOLD,width=3)
   x,y=project(pose);d.ellipse((x-7,y-7,x+7,y+7),fill=FG,outline=BG,width=2)


def read_source_frames(indices,video):
 cap=cv2.VideoCapture(str(video));wanted=set(indices);images={};maxindex=max(indices)
 for n in range(maxindex+1):
  ok=cap.grab()
  if not ok:raise RuntimeError(f'Cannot read source frame {n}')
  if n in wanted:
   ok,bgr=cap.retrieve()
   if not ok:raise RuntimeError(f'Cannot decode source frame {n}')
   # Retain only the two authentic camera panels, not the old dashboard.
   imgs=[]
   for box in ((44,181,380,517),(44,577,380,913)):
    x0,y0,x1,y1=box;imgs.append(Image.fromarray(cv2.cvtColor(bgr[y0:y1,x0:x1],cv2.COLOR_BGR2RGB)))
   images[n]=imgs
 cap.release();return images

TOTAL=549
FAILURE_KINDS={'route_failure','transit_stall','contact_stall','target_rejection','segment_stall','target_envelope_rejected'}

def load_data():
 data=json.loads((ROOT/'obstacle_map_timeline.json').read_text())
 mapping=json.loads((ROOT/'control_state_frames.json').read_text())
 assert len(data['frames'])==len(mapping['control_state_frames'])==TOTAL+1
 maps={m['revision']:m for m in data['maps']};routes={r['id']:r for r in data['routes']}
 points=[f['grip_xyz_m'] for f in data['frames']]
 for m in maps.values():
  for b in m['avoid_boxes']:
   points.extend([np.array(b['lower_xyz_m'])-m['clearance_m'],np.array(b['upper_xyz_m'])+m['clearance_m']])
 for f in data['frames']:
  if f['goal_xyz_m']:points.append(f['goal_xyz_m'])
 points=np.array(points);lo=np.floor((points.min(0)-.015)*20)/20;hi=np.ceil((points.max(0)+.015)*20)/20
 events={}
 for e in data['events']:events.setdefault(e['frame_index'],[]).append(e)
 return data,mapping,maps,routes,Charts((lo,hi)),events

def note_for(n,events,frame,intro=False):
 if intro:return ('初始画面：模型尚未建立障碍地图','10 cm 是每步命令上限；实际位移由控制器响应决定。地图由后续模型输出建立。',MUTED)
 ev=events.get(n,[])
 if n==0:return ('旋转目标超限：1.572 rad > 1.2 rad','本段未执行。平移目标距离约 13.4 cm，在 50 cm 上限内；拒绝原因是旋转。',RED)
 if n==162:return ('路径不可用：6 cm 间隙使起点插值无效','实测点在扩张框外，但插值涉及 2 个被屏蔽顶点；本段未执行，随后间隙调回 4 cm。',RED)
 if n in (196,507):
  error='4.82' if n==196 else '4.88'
  return ('transit 停滞 · 已接近目标，但尚未满足到达阈值',f'位置误差 {error} mm，高于 4 mm 阈值；停止本段并交回模型。不能据此判定碰到障碍。',RED)
 if n==433:return ('contact 下降停滞 · 转交模型调整','位置误差约 8.15 mm；此接触阶段不使用 transit 路径规划。',RED)
 if n==TOTAL:return ('LIBERO 成功 · 第二只杯子仍被夹持','严格终止检查未通过；额外释放、静置诊断后通过。诊断动作不包含在此视频中。',GOLD)
 contact=next((e for e in ev if e['kind']=='first_forbidden_contact'),None)
 if contact:return ('记录到夹爪掌部—杯子接触','红点为评估器记录的接触位置。此轮没有记录到杯子与罐子的接触。',RED)
 update=next((e for e in ev if e['kind']=='map_update'),None)
 if update:
  rev=update['map_revision']
  detail={1:'模型建立 3 个估计障碍框，后续 transit 段复用地图。',2:'安全间隙从 2.5 cm 增至 6 cm；本次更新导致起点不可用。',3:'模型保留 3 个障碍框，将安全间隙从 6 cm 调回 4 cm。',4:'第一只杯子放置后更新完整障碍集合，切换到第二只杯子；间隙为 3 cm。'}[rev]
  return (f'共享障碍地图更新 · 修订 {rev}',detail,ORANGE)
 if frame.get('forbidden_contact_delta',0)>0:return ('当前记录到掌部—杯子接触','接触证据来自实际记录；障碍框仍是模型估计，不表示完整机械臂或杯身包络。',RED)
 if frame['profile']=='contact':return ('contact 操作 · 本段未启用路径规划','每步位移命令上限仍为 0.5 cm；地图保留供后续 transit 复用。',MUTED)
 if frame['route_id']:return ('transit 操作 · 10 cm 命令上限 / 50 cm 目标上限','青线为本段记录的规划路径；本轮可用路径均为直线，A* 绕行 0 次。',CYAN)
 return ('观察 / 保持 · 共享地图保留','粉线为夹爪测得位置轨迹，不代表被搬物体的完整运动包络。',MUTED)

def render(n,cameras,data,maps,routes,charts,events,intro=False):
 f=data['frames'][n];ev=[] if intro else events.get(n,[])
 revision=0 if intro else f.get('decision_boundary_map_revision',f['map_revision'])
 m=maps[revision]
 failure=next((e for e in ev if e['kind'] in FAILURE_KINDS),None)
 goal=(failure.get('goal_xyz_m') if failure else f['goal_xyz_m']) if not intro else None
 route=routes.get(f['route_id']);path=route.get('path_xyz_m',[]) if route else []
 active=not intro and not failure and f['profile']=='transit' and f['status']!='failed' and route is not None
 im=Image.new('RGBA',(W,H),BG+(255,));d=ImageDraw.Draw(im)
 txt(d,(32,22),'t04 · 10 cm / 50 cm 障碍地图回放',38)
 txt(d,(32,80),'同步原始相机画面   /   0.5× 动作回放，省略模型等待；重点事件暂停',21,MUTED)
 txt(d,(1888,28),f'控制步 {n:04d} / {TOTAL}',27,FG,'rt')
 txt(d,(1888,75),f'动作时间 {n/20:05.2f} s   ·   地图修订 {m["revision"]}   ·   间隙 {m["clearance_m"]*100:g} cm',20,MUTED,'rt')
 d.line((32,124,1888,124),fill=GRID,width=1)
 txt(d,(32,142),'外部相机',23);txt(d,(32,601),'腕部相机',23)
 im.paste(cameras[0].resize((400,400),Image.Resampling.LANCZOS),(32,180))
 im.paste(cameras[1].resize((400,400),Image.Resampling.LANCZOS),(32,640))
 charts.base(im)
 trajectory=[q['grip_xyz_m'] for q in data['frames'][:n+1]]
 trajectory=trajectory[::max(1,len(trajectory)//500)]+[f['grip_xyz_m']]
 charts.draw(im,m['avoid_boxes'],m['clearance_m'],trajectory,f['grip_xyz_m'],goal,path,active,m['can_box_index'])
 d=ImageDraw.Draw(im)
 for x,label,col,kind in ((490,'估计障碍框',ORANGE,'box'),(700,'安全间隙',ORANGE,'dash'),(900,'规划路径',CYAN,'line'),(1100,'实际夹爪轨迹',PINK,'line'),(1360,'当前目标',GOLD,'plus'),(1560,'夹爪位置',FG,'dot')):
  if kind=='box':d.rectangle((x,190,x+23,205),outline=col,width=2)
  elif kind=='dash':dash(d,(x,198),(x+26,198),col,2)
  elif kind=='line':d.line((x,198,x+27,198),fill=col,width=3)
  elif kind=='plus':d.line((x,198,x+22,198),fill=col,width=2);d.line((x+11,187,x+11,209),fill=col,width=2)
  else:d.ellipse((x+7,192,x+19,204),fill=col)
  txt(d,(x+34,183),label,18,MUTED)
 hit=next((e for e in ev if e.get('contact_xyz_m')),None)
 if hit:
  for project in (charts.project,lambda p:charts.flat(p,(0,1),charts.xy),lambda p:charts.flat(p,(0,2),charts.xz)):
   x,y=project(hit['contact_xyz_m']);d.ellipse((x-11,y-11,x+11,y+11),outline=RED,width=3);d.ellipse((x-3,y-3,x+3,y+3),fill=RED)
 title,detail,color=note_for(n,events,f,intro)
 d.rectangle((472,940,1888,1016),fill=PANEL)
 txt(d,(490,947),title,25,color);txt(d,(490,982),detail,19,MUTED)
 x0,x1,y=490,1870,1032
 d.line((x0,y,x1,y),fill=GRID,width=5);d.line((x0,y,x0+(x1-x0)*n/TOTAL,y),fill=CYAN,width=5)
 for idx,ee in events.items():
  col=RED if any(e['kind'] in FAILURE_KINDS or e['kind']=='first_forbidden_contact' for e in ee) else ORANGE
  xx=x0+(x1-x0)*idx/TOTAL;d.line((xx,y-5,xx,y+5),fill=col,width=2)
 txt(d,(490,1048),'框＝模型估计；非距离场 / 非全身碰撞保证。XY、XZ 为几何投影；罐子标签为推定。',17,MUTED)
 return im.convert('RGB')

def embed_chapters(output,chapters,duration):
 lines=[';FFMETADATA1','title=t04 obstacle map replay - transit 10 cm / target 50 cm','comment=Recorded obstacle estimates and measured gripper path; no simulator replay.']
 for i,ch in enumerate(chapters):
  end=chapters[i+1]['video_time_s'] if i+1<len(chapters) else duration
  lines.extend(['[CHAPTER]','TIMEBASE=1/1000',f'START={round(ch["video_time_s"]*1000)}',f'END={round(end*1000)}','title='+ch['title']])
 meta=ROOT/'chapters.ffmeta';meta.write_text('\n'.join(lines)+'\n');temp=ROOT/'chaptered.mp4'
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(output),'-i',str(meta),'-map_metadata','1','-c','copy','-movflags','+faststart',str(temp)],check=True)
 temp.replace(output)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
 data,mapping,maps,routes,charts,events=load_data()
 selected=[0,2,115,162,163,196,347,433,507,TOTAL]
 states=selected if args.preview else range(TOTAL+1)
 indices=[mapping['control_state_frames'][n]['source_frame_index'] for n in states]
 print('Decoding recorded camera panels...',flush=True)
 cameras=read_source_frames(indices,RUN/'trial/episode.mp4')
 initial_cameras=cameras[mapping['control_state_frames'][0]['source_frame_index']]
 render(0,initial_cameras,data,maps,routes,charts,events,intro=True).save(ROOT/'preview-initial.png')
 for n in selected:
  src=mapping['control_state_frames'][n]['source_frame_index']
  render(n,cameras[src],data,maps,routes,charts,events).save(ROOT/f'preview-step-{n:04d}.png')
 if args.preview:print('Preview frames ready.',flush=True);return
 output=ROOT/'t04-can-obstacle-map-transit100.mp4'
 command=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',str(output)]
 proc=subprocess.Popen(command,stdin=subprocess.PIPE)
 chapters=[];outframe=0
 holds={0:60,115:40,162:60,163:36,196:46,347:30,433:40,507:46,TOTAL:100}
 try:
  initial=render(0,initial_cameras,data,maps,routes,charts,events,intro=True).tobytes()
  chapters.append({'control_state':0,'initial_observation':True,'video_time_s':0,'duration_s':2,'title':'初始观察 · 10 cm / 50 cm'})
  for _ in range(40):proc.stdin.write(initial)
  outframe=40
  for n in range(TOTAL+1):
   src=mapping['control_state_frames'][n]['source_frame_index'];im=render(n,cameras[src],data,maps,routes,charts,events)
   count=2+holds.get(n,0)
   if n in holds:chapters.append({'control_state':n,'video_time_s':outframe/20,'duration_s':count/20,'title':note_for(n,events,data['frames'][n])[0]})
   raw=im.tobytes()
   for _ in range(count):proc.stdin.write(raw)
   outframe+=count
   if n%100==0:print(json.dumps({'state':n,'encoded_frames':outframe}),flush=True)
 finally:proc.stdin.close()
 code=proc.wait()
 if code:raise RuntimeError(f'Encoder exited {code}')
 duration=outframe/20;embed_chapters(output,chapters,duration)
 meta={'video':output.name,'fps':20,'width':W,'height':H,'frames':outframe,'duration_s':duration,'control_states':TOTAL+1,'playback':'Each control state 0.1s; 0.5x simulator speed; labeled event freezes; API waits omitted','source_video':str(RUN/'trial/episode.mp4'),'mapping':'control_state_frames.json','data':'obstacle_map_timeline.json','chapters':chapters,'embedded_chapters':True,'no_simulator_or_model_calls':True,'terminal_diagnostic_actions_included':False,'event_semantics':'Decision-boundary map shown at exact measured state before next action; failures suppress active route. Initial uninitialized map shown separately before model decision at state0.','transit_command_limit_m':.1,'transit_target_limit_m':.5,'size_bytes':output.stat().st_size}
 (ROOT/'video_manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps({'video':str(output),'duration_s':duration,'frames':outframe}),flush=True)

if __name__=='__main__':main()
