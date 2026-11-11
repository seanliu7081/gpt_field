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
RUN=ROOT.parents[2]/'runs/agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001'
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
   label=f'B{n+1}'
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

TOTAL=843
FAILURE_KINDS={'route_failure','transit_stall','contact_stall','target_rejection','segment_stall','target_envelope_rejected','stage_budget_exhausted'}

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
 if intro:return ('初始观察 · 障碍地图尚未建立','相机来自本轮实际录像；后续显示模型估计框、规划路径和测得的夹爪轨迹。',MUTED)
 ev=events.get(n,[])
 if n==TOTAL:return ('LIBERO 成功 · 释放动作刚开始','终止时仍有持握 / 未静止证据；额外释放静置诊断通过，诊断动作未收入视频。',GOLD)
 reject=next((e for e in ev if e['kind']=='target_rejection'),None)
 if reject:
  return ('旋转目标超限 · 本段未执行','平移目标在 50 cm 范围内；模型随后拆分旋转。此拒绝发生在路径规划之前。',RED)
 if n==38:return ('首次碰瓶 · 夹爪左指接触瓶身','发生于 transit 动作 37；红圈是记录的接触点。直线通过地图检查仍发生了实际接触。',RED)
 if n==82:return ('transit 停滞 · 模型更新地图并退开','瓶子接触已发生；此处更新为地图修订 2，随后执行 5 步短直接控制。',RED)
 if n==87:return ('路径不可用 · 起点插值涉及屏蔽网格','测得起点在膨胀框外，但插值涉及 4 个屏蔽顶点；本段未执行，随后先向上退出。',RED)
 if n==224:return ('瓶子与前方摩卡壶接触','接触发生在闭合夹爪阶段；红圈来自记录的首次接触位置，非规划器预测。',RED)
 if n==368:return ('前方摩卡壶抬起 · 准备升高搬运','模型选择先升高再横向搬运；后续 transit 的有效规划路径仍为直线。',CYAN)
 if n==538:return ('松手片段达到预算 · 更新地图修订 3','张开片段执行 30 步后尚未完成；模型更新地图并抬手。地图间隙始终为 1.8 cm。',ORANGE)
 if n==597:return ('再次起点插值不可用 · 先向上退出','测得起点在膨胀框外，但插值涉及 2 个屏蔽顶点；本段未执行，随后抬高再移动。',RED)
 failure=next((e for e in ev if e['kind'] in FAILURE_KINDS),None)
 if failure:
  if failure['kind']=='route_failure':return ('路径不可用 · 返回模型','本段未执行；起点有效性检查失败不等于已经发生物理碰撞。',RED)
  if failure.get('skill')=='set_gripper':return ('松手片段未完成 · 返回模型','夹爪已开始张开，但保持姿态尚未满足片段完成条件；后续动作未自动执行。',RED)
  if failure.get('profile')=='contact':return ('接触操作停滞 · 返回模型','当前接近 / 下降片段停止；contact 阶段不调用 transit 路径规划。',RED)
  return ('transit 停滞 · 返回模型','当前片段停止，等待模型调整；停滞本身不构成碰撞证明。',RED)
 contact=next((e for e in ev if e['kind']=='first_forbidden_contact'),None)
 if contact:return ('记录到物体接触','红圈为记录的首次接触位置；接触记录独立于模型估计的障碍地图。',RED)
 update=next((e for e in ev if e['kind']=='map_update'),None)
 if update:return (f'共享障碍地图更新 · 修订 {update["map_revision"]}','橙框为本次模型估计；虚线为 1.8 cm 间隙。B1 / B2 / B3 仅是框编号。',ORANGE)
 if frame.get('obstacle_contact_delta',0)>0:
  detail='当前动作记录到瓶子接触；框是模型估计，不能当作真实瓶身或完整夹爪包络。'
  pairs=frame.get('obstacle_contact_pairs',[])
  return ('当前动作记录到瓶子接触',detail,RED)
 if frame.get('forbidden_contact_delta',0)>0:return ('当前动作记录到其他接触','接触位置来自仿真记录；此类接触采用记录模式，不会自行终止任务。',RED)
 if 82<n<=87:return ('短直接控制 · 退出瓶子附近','本次模型选择 5 步短直接控制；单步命令上限 2 cm，不使用 transit 路径规划。',MUTED)
 if frame['profile']=='contact':return ('contact 操作 · 本段未启用路径规划','每步位移命令上限为 0.5 cm；共享地图保留供后续 transit 使用。',MUTED)
 if frame['route_id']:return ('transit 操作 · 10 cm 命令上限 / 50 cm 目标上限','青线是本段记录的规划路径；本轮 11 条有效路径均为直线，A* 绕行 0 次。',CYAN)
 return ('观察 / 保持 · 共享地图保留','粉线为测得的夹爪轨迹；接触与抓取结果需结合相机画面及记录判断。',MUTED)

def render(n,cameras,data,maps,routes,charts,events,intro=False):
 f=data['frames'][n];ev=[] if intro else events.get(n,[])
 revision=0 if intro else f.get('decision_boundary_map_revision',f['map_revision'])
 m=maps[revision]
 failure=next((e for e in ev if e['kind'] in FAILURE_KINDS),None)
 goal=(failure.get('goal_xyz_m') if failure else f['goal_xyz_m']) if not intro else None
 route=routes.get(f['route_id']);path=route.get('path_xyz_m',[]) if route else []
 active=not intro and not failure and f['profile']=='transit' and f['status']!='failed' and route is not None and n<TOTAL
 im=Image.new('RGBA',(W,H),BG+(255,));d=ImageDraw.Draw(im)
 txt(d,(32,22),'t08 · 瓶子障碍地图回放',38)
 txt(d,(32,80),'10 cm / 50 cm   ·   0.5× 动作回放，省略模型等待；关键事件暂停',21,MUTED)
 txt(d,(1888,28),f'控制步 {n:04d} / {TOTAL}',27,FG,'rt')
 txt(d,(1888,75),f'动作时间 {n/20:05.2f} s   ·   地图修订 {m["revision"]}   ·   间隙 {m["clearance_m"]*100:g} cm',20,MUTED,'rt')
 d.line((32,124,1888,124),fill=GRID,width=1)
 txt(d,(32,142),'外部相机',23);txt(d,(32,601),'腕部相机',23)
 im.paste(cameras[0].resize((400,400),Image.Resampling.LANCZOS),(32,180))
 im.paste(cameras[1].resize((400,400),Image.Resampling.LANCZOS),(32,640))
 charts.base(im)
 trajectory=[q['grip_xyz_m'] for q in data['frames'][:n+1]]
 trajectory=trajectory[::max(1,len(trajectory)//500)]+[f['grip_xyz_m']]
 charts.draw(im,m['avoid_boxes'],m['clearance_m'],trajectory,f['grip_xyz_m'],goal,path,active,None)
 d=ImageDraw.Draw(im)
 for x,label,col,kind in ((490,'估计障碍框',ORANGE,'box'),(700,'安全间隙',ORANGE,'dash'),(900,'规划路径',CYAN,'line'),(1100,'实际夹爪轨迹',PINK,'line'),(1360,'当前目标',GOLD,'plus'),(1560,'夹爪位置',FG,'dot'),(1750,'接触点',RED,'contact')):
  if kind=='box':d.rectangle((x,190,x+23,205),outline=col,width=2)
  elif kind=='dash':dash(d,(x,198),(x+26,198),col,2)
  elif kind=='line':d.line((x,198,x+27,198),fill=col,width=3)
  elif kind=='plus':d.line((x,198,x+22,198),fill=col,width=2);d.line((x+11,187,x+11,209),fill=col,width=2)
  elif kind=='contact':d.ellipse((x+5,190,x+21,206),outline=col,width=2)
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
  col=RED if any(e['kind'] in FAILURE_KINDS or e['kind'] in ('first_forbidden_contact','obstacle_contact') for e in ee) else ORANGE
  xx=x0+(x1-x0)*idx/TOTAL;d.line((xx,y-5,xx,y+5),fill=col,width=2)
 txt(d,(490,1048),'显示模型估计框及间隙，并非有符号距离场；轨迹仅为夹爪，接触点来自实际记录。',17,MUTED)
 return im.convert('RGB')

def embed_chapters(output,chapters,duration):
 lines=[';FFMETADATA1','title=t08 bottle obstacle map replay - transit 10 cm / target 50 cm','comment=Recorded obstacle estimates and measured gripper path; no simulator replay.']
 for i,ch in enumerate(chapters):
  end=chapters[i+1]['video_time_s'] if i+1<len(chapters) else duration
  lines.extend(['[CHAPTER]','TIMEBASE=1/1000',f'START={round(ch["video_time_s"]*1000)}',f'END={round(end*1000)}','title='+ch['title']])
 meta=ROOT/'chapters.ffmeta';meta.write_text('\n'.join(lines)+'\n');temp=ROOT/'chaptered.mp4'
 subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','error','-i',str(output),'-i',str(meta),'-map_metadata','1','-c','copy','-movflags','+faststart',str(temp)],check=True)
 temp.replace(output)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args()
 data,mapping,maps,routes,charts,events=load_data()
 selected=[0,1,38,82,87,224,368,508,538,597,767,TOTAL]
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
 output=ROOT/'t08-bottle-obstacle-map-transit100.mp4'
 command=['ffmpeg','-y','-hide_banner','-loglevel','error','-f','rawvideo','-vcodec','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r','20','-i','-','-an','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-threads','4',str(output)]
 proc=subprocess.Popen(command,stdin=subprocess.PIPE)
 chapters=[];outframe=0
 holds={0:40,1:32,38:80,82:60,87:50,224:60,368:28,508:28,538:42,597:50,767:24,TOTAL:100}
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
