#!/usr/bin/env python3
"""Render a source-grounded Chinese deck from existing Astra drawer experiments.
Only writes inside this new presentation directory; no inference/planning rerun.
"""
from pathlib import Path
import sys, json, math, hashlib, re
ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
ASSETS = ROOT / 'assets'
sys.path.insert(0, str(ROOT / 'vendor'))
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.utils import ImageReader
from PIL import Image

W,H=960,540
C={
 'bg':'#F5F7F9', 'ink':'#14293A', 'muted':'#536575', 'line':'#D9E2E8',
 'teal':'#007F82', 'tealsoft':'#E2F2F0', 'blue':'#245FA4', 'bluesoft':'#E7EFF9',
 'orange':'#C45C35', 'orangesoft':'#FCEBE3', 'white':'#FFFFFF',
 'navy':'#132E41', 'gray':'#ECEFF2', 'red':'#BD3F42', 'gold':'#EAC35D'
}
for name,label in [('CN','Regular'),('CNB','Bold')]:
    pdfmetrics.registerFont(TTFont(name,str(ASSETS/f'NotoSansSC-{label}.ttf')))
E=json.loads((ROOT/'evidence_astra.json').read_text())
P=json.loads((ROOT/'evidence_planning.json').read_text())
rows=E['rows']; closed,opened=rows
out=ROOT/'astra_field_explained_zh.pdf'
c=canvas.Canvas(str(out),pagesize=(W,H),pageCompression=1)
c.setTitle('Astra 场景场：生成、几何精度与运动规划')
c.setAuthor('gpt_field experiment review')
c.setSubject('LIBERO drawer003 — source-grounded single RGB CSG field and planning evaluation')
page_count=17
page_no=0
text_records=[]


def color(v): return HexColor(C.get(v,v))
def rect(x,y,w,h,fill='white',stroke=None,r=0):
    c.setFillColor(color(fill)); c.setStrokeColor(color(stroke or fill))
    if r: c.roundRect(x,H-y-h,w,h,r,stroke=bool(stroke),fill=1)
    else: c.rect(x,H-y-h,w,h,stroke=bool(stroke),fill=1)
def line(x1,y1,x2,y2,stroke='line',width=1):
    c.setStrokeColor(color(stroke));c.setLineWidth(width);c.line(x1,H-y1,x2,H-y2)
def circle(x,y,r,fill,stroke=None):
    c.setFillColor(color(fill));c.setStrokeColor(color(stroke or fill));c.circle(x,H-y,r,stroke=bool(stroke),fill=1)
def arrow(x1,y1,x2,y2,stroke='teal',width=2):
    line(x1,y1,x2,y2,stroke,width)
    a=math.atan2(y2-y1,x2-x1); s=7
    line(x2,y2,x2-s*math.cos(a-.45),y2-s*math.sin(a-.45),stroke,width)
    line(x2,y2,x2-s*math.cos(a+.45),y2-s*math.sin(a+.45),stroke,width)
def txt(s,x,y,size=18,fill='ink',bold=False):
    font='CNB' if bold else 'CN'
    c.setFillColor(color(fill));c.setFont(font,size)
    c.drawString(x,H-y-size*.9,str(s))
    text_records.append({'page':page_no,'text':str(s),'x':x,'top':y,'font_size':size,
                         'width':pdfmetrics.stringWidth(str(s),font,size)})
def wrap(s,width,size=17,bold=False):
    font='CNB' if bold else 'CN'; result=[]
    for para in str(s).split('\n'):
        if not para: result.append('');continue
        current=''
        for ch in para:
            if pdfmetrics.stringWidth(current+ch,font,size)>width and current:
                result.append(current.rstrip());current=ch.lstrip()
            else: current+=ch
        result.append(current)
    return result

def boxtext(s,x,y,w,size=17,fill='ink',bold=False,leading=None,max_h=None):
    leading=leading or size*1.48
    ls=wrap(s,w,size,bold)
    h=(len(ls)-1)*leading+size
    if max_h is not None and h>max_h+.1:
        raise ValueError(f'Page {page_no} text too tall {h:.1f}>{max_h}: {s[:50]}')
    for i,l in enumerate(ls): txt(l,x,y+i*leading,size,fill,bold)
    return h

def small_label(s,x,y,fill='teal'):
    txt(s,x,y,11,fill,True)
def image(name,x,y,w,h=None):
    path=Path(name)
    if not path.is_absolute():path=ASSETS/path
    im=Image.open(path);iw,ih=im.size
    h=h or w*ih/iw
    c.drawImage(ImageReader(im),x,H-y-h,w,h,mask='auto')

def bullet(s,x,y,w,size=17,fill='ink',dot='teal',gap=8):
    circle(x+3,y+size*.53,2.5,dot)
    return boxtext(s,x+16,y,w-16,size,fill)+gap

def badge(s,x,y,w,fill='tealsoft',fg='teal'):
    rect(x,y,w,28,fill,r=7);txt(s,x+12,y+6,12,fg,True)

def callout(s,x,y,w,h=48,fill='tealsoft',fg='teal',size=16):
    rect(x,y,w,h,fill,r=8)
    boxtext(s,x+16,y+12,w-32,size,fg,bold=True,max_h=h-20,leading=size*1.35)

def start(title,tag,subtitle=None,refs=''):
    global page_no
    page_no+=1
    rect(0,0,W,H,'bg')
    rect(0,0,W,6,'teal')
    small_label(tag.upper(),40,23)
    txt(title,40,46,28,'ink',True)
    if subtitle:txt(subtitle,40,88,13,'muted')
    line(40,504,920,504)
    txt(refs,40,515,9,'muted')
    txt(f'{page_no:02d} / {page_count}',866,515,9,'muted')
    c.bookmarkPage(f'page{page_no}')
    c.addOutlineEntry(title,f'page{page_no}',level=0,closed=False)

def end():c.showPage()

def metric(value,label,x,y,w=200,fill='teal'):
    txt(value,x,y,36,fill,True)
    boxtext(label,x,y+47,w,14,'muted',max_h=48)

# 01
start('Astra 场景场', 'EXPERIMENT REVIEW · LIBERO DRAWER003',
      '从单张图像，到几何表示，再到机器人路径',refs='证据范围：drawer003 closed step60 / open step552；原有实验复核，未重新推理。 [S1–S8]')
image('open_rgb.png',588,124,332,332)
badge('单张 RGB  →  CSG 隐式场',40,132,282)
txt('三个需要分开回答的问题',40,185,23,'ink',True)
for n,title,body,y in [
 ('01','场从哪里来','Astra 估计几何参数；代码把参数变成可查询的场。',230),
 ('02','几何保留得多准确','网格能保留多数模型表面；真实柜体仍有厘米级偏差。',310),
 ('03','规划怎样使用场','关节空间搜索，候选机器人姿态通过场查询筛选。',390)]:
    txt(n,40,y,19,'teal',True);txt(title,80,y,18,'ink',True)
    boxtext(body,80,y+28,455,15,'muted',max_h=48)
txt('实验图：抽屉打开快照，Astra 的唯一视觉输入',588,466,11,'muted')
end()

# 02
start('先区分：目标距离场与场景几何场','01 / FIELD MEANING',
      '相同的“热图”形式，可以表达完全不同的信息。',refs='[S10] drawer003 冻结规划代码；目标场为到目标区域的图距离，无障碍编码。Astra 语义据 [S1, S2]。')
rect(40,131,424,286,'white',r=12);rect(484,131,436,286,'white',r=12)
small_label('此前 speedup v1 的 waypoint 场',60,151,'blue')
txt('“离目标还有多远”',60,181,24,'blue',True)
boxtext('目标位置决定场值。颜色描述每个空间位置到当前路点目标区域的距离。',60,231,380,18,max_h=85)
boxtext('它本身没有编码柜体、桌面和障碍物的表面。',60,338,380,17,'muted',max_h=62)
small_label('本次 Astra 场景场',504,151)
txt('“哪里被物体占据”',504,181,24,'teal',True)
boxtext('图像估计出的物体形状决定场值。零等值面表示预测表面，正负号区分内外。',504,231,392,18,max_h=85)
boxtext('规划可用它检查机器人是否与预测环境重叠。',504,338,392,17,'muted',max_h=62)
callout('本 PDF 讨论场景几何场；热图颜色由物体几何决定，与目标路点距离无关。',40,437,880,48)
end()

# 03
start('Astra 输出几何描述，代码生成数值场','02 / CREATION PIPELINE',
      '模型没有逐个输出数百万个 voxel，也没有直接输出最终网格模型。',refs='[S1] prompt / schema / field_csg.py；[S2] build_fields.py 与 evaluation。')
steps=[('单张 RGB','512 × 512\n无深度输入','blue'),('Astra JSON','基元、位姿\n尺寸、布尔关系','teal'),('解析场 φ(p)','代码计算任意\n三维点的场值','teal'),('保存数值网格','float32\n4 mm / 2 mm','teal'),('解码表面','φ = 0\nMarching Cubes','blue')]
xs=[40,221,402,583,764]
for i,(title,body,col) in enumerate(steps):
    rect(xs[i],150,156,153,'white',r=10)
    txt(f'{i+1:02d}',xs[i]+16,165,14,col,True)
    boxtext(title,xs[i]+16,195,130,18,'ink',True,max_h=45)
    boxtext(body,xs[i]+16,246,130,13,'muted',max_h=40,leading=19)
    if i<4:arrow(xs[i]+160,226,xs[i]+177,226)
rect(40,334,337,96,'bluesoft',r=10)
txt('模型负责：几何估计',58,350,18,'blue',True)
boxtext('从照片猜测尺寸、位置、方向、开口和不可见部分。',58,383,300,15,max_h=48)
rect(397,334,523,96,'tealsoft',r=10)
txt('本地代码负责：场求值、保存和解码',415,350,18,'teal',True)
boxtext('解码器只读取保存的数值块，不读取 RGB、CSG JSON 或真值几何。',415,383,486,15,max_h=48)
txt('本次生成器：gpt-6-astra / medium；每个快照独立输入、一次有效输出。',40,460,15,'muted')
end()

# 04
start('输入没有真值几何；尺度和形状都由图像估计','03 / PROMPT & OUTPUT',
      '原始提示词要求估计场景几何，同时明确单张图像的尺度与遮挡歧义。',refs='[S1] instructions.txt、schema.json、fields/drawer003_closed_step60.json；未做 GT 尺度对齐。')
rect(40,128,420,341,'white',r=10)
txt('输入约束（原提示词摘要）',58,147,20,'ink',True)
y=192
for s in ['只提供一张正向 RGB；允许常识性的物体尺寸先验。','不提供深度、相机标定、场景资产、任务历史或其他帧。','输出 JSON：box / sphere / cylinder 加布尔组合。','最多 32 个基元、64 个总节点；记录不确定和省略的细节。']:
    y+=bullet(s,58,y,382,16,gap=13)
boxtext('模型还报告焦距估计，但代码直接使用其三维参数建场，不用该焦距再反投影。',58,422,382,12,'muted',max_h=40,leading=18)
rect(480,128,440,341,'navy',r=10)
txt('真实输出片段：closed 快照',500,146,18,'white',True)
code=[('name', 'dark_left_cabinet'),('op','box'),('center_m','[-0.30, 0.09, 0.82]'),('size_m','[0.14, 0.115, 0.17]'),('rotation_xyzw','[0.316228, 0, 0, 0.948683]')]
y=191
for k,v in code:
    txt(k,500,y,12,'#85C7CB',True);txt(v,500,y+19,14,'white');y+=47
boxtext('单位：米；box 的 size_m 是半尺寸。\n相机坐标：X 向右、Y 向下、Z 向前。',500,428,390,12,'#D0DEE6',max_h=38,leading=17)
end()

# 05
start('场保存的是“实体在哪里”的隐式函数','04 / FIELD DEFINITION',
      'CSG = Constructive Solid Geometry：用简单实体和布尔关系组合复杂形状。',refs='[S1] field_csg.py；右侧为空腔概念示意，不是实验重建图。')
rect(40,131,480,328,'white',r=10)
txt('场定义域内每个 p，都能查询 φ(p)',60,151,21,'ink',True)
for y,col,sign,meaning in [(203,'orange','φ(p) < 0','实体内部：占据'),(246,'ink','φ(p) = 0','模型预测的表面'),(289,'teal','φ(p) > 0','实体外部：候选自由空间')]:
    circle(69,y+10,5,col);txt(sign,88,y,18,col,True);txt(meaning,233,y,17)
line(60,330,500,330)
txt('并集：min(a, b)       交集：max(a, b)',60,349,16)
txt('差集：max(a, −b)     → 可以表达空腔',60,385,16)
rect(540,131,380,328,'tealsoft',r=10)
# Illustrative CSG difference cross-section: cavity and closed bottom.
rect(610,199,244,174,'orange',r=5);rect(635,187,194,156,'tealsoft')
txt('实体',617,349,14,'white',True)
txt('空腔 / 自由空间',648,254,16,'teal',True)
arrow(730,181,730,221,'teal')
boxtext('示意：外部实体 − 内部实体\n零等值面同时包含外壁和内壁。',560,397,340,14,'muted',max_h=44,leading=20)
boxtext('注意：布尔组合后的 φ 不保证处处等于到最近表面的精确欧氏距离。',40,476,880,15,'orange',bold=True,max_h=24)
end()

# 06
start('重建测试与规划测试使用不同的读取路径','05 / WHAT IS ACTUALLY QUERIED',
      '两条路径来自同一次 Astra 几何估计，但不能把它们当作完全相同的数值实验。',refs='[S2] 保存场与解码；[S5] planning_fields.py:215–267；planning_001/settings.json。')
rect(40,139,192,268,'navy',r=10)
boxtext('Astra\n原始 CSG JSON',58,162,157,20,'white',True,max_h=77,leading=34)
boxtext('包含场景\n以及可见机器人',58,290,157,17,'#D0DEE6',max_h=60)
arrow(240,209,283,209);arrow(240,341,283,341)
rect(298,139,622,122,'bluesoft',r=10)
small_label('重建精度分支',318,153,'blue')
txt('原始场 → 保存 2 mm 网格 → 解码 mesh → 评分',318,180,18,'blue',True)
boxtext('检验数值表示是否保留模型形状，以及解码场景与真值差多远。',318,219,580,15,max_h=30)
rect(298,285,622,122,'tealsoft',r=10)
small_label('当前 planning v2 分支',318,299)
txt('移除 robot_* → 环境解析场 → 直接查询 φ(p)',318,326,18,'teal',True)
boxtext('原 2 mm 网格仅提供查询域边界；规划没有读取网格值做插值。',318,365,580,15,max_h=30)
callout('解析查询可避免网格插值误差，但不会纠正模型估计错的尺寸、位置或漏建物体。',40,433,880,49,size=15)
end()

# 07
start('准确性必须分两层测量','06 / ERROR DEFINITION',
      '“模型给出的形状被保存得很好”，不等于“模型给出的形状就是真实场景”。',refs='[S2] evaluate_fields.py；comparison_summary.json。所有真值仅用于事后评估。')
rect(40,133,424,216,'bluesoft',r=10)
small_label('A / C：相对于真实场景的误差',60,149,'blue')
boxtext('A：原始解析场 vs LIBERO 真值\nC：2 mm 网格解码表面 vs 真值',60,186,382,18,max_h=76,leading=31)
boxtext('同一真实相机射线，比较第一个交点的相机 Z 深度。',60,275,382,16,max_h=60)
rect(484,133,436,216,'tealsoft',r=10)
small_label('B：场表示本身的保存误差',504,149)
boxtext('原始解析场\nvs 保存网格后解码的表面',504,186,394,18,max_h=76,leading=31)
boxtext('6 个由模型边界生成的外部视角；不使用真值。检验同一预测几何的变化。',504,275,394,16,max_h=60)
rect(40,368,880,66,'white',r=10)
txt('e(u,v) = | z_pred(u,v) − z_GT(u,v) |',60,386,22,'ink',True)
txt('单位：mm',737,390,18,'teal',True)
boxtext('MAE / P95 只统计命中射线；同时报告覆盖率。≤5 mm 比例把漏失计为失败。\n这里不是三维最近表面距离，也没有验证所有隐藏表面；B 的射线集合与 A / C 不同。',40,450,880,14,'muted',max_h=44,leading=21)
end()

# 08
start('2 mm 网格保留了多数已预测表面','07 / REPRESENTATION PRESERVATION',
      'B：只比较“原始 Astra 形状”和“网格解码形状”，不比较真实场景。',refs='[S2] comparison_summary.json / preservation_B_six_views；[S3] grazing-wall diagnostic。')
cols=[40,344,624]; widths=[304,280,296]
for x,w,t in zip(cols,widths,['保存误差 B（2 mm）','抽屉关闭 · step60','抽屉打开 · step552']):
    rect(x,134,w,46,'navy');txt(t,x+16,147,16,'white',True)
data=[('有参考表面的射线数','1,750','1,897'),('命中保留率','100%','98.735%'),('命中 MAE','0.0071 mm','0.0034 mm'),('命中最大误差','8.395 mm','0.539 mm'),('丢失原始交点','0','24')]
for i,vals in enumerate(data):
    y=180+i*44
    for x,w,v in zip(cols,widths,vals):
        rect(x,y,w,44,'white' if i%2==0 else 'gray');txt(v,x+16,y+12,16,'ink',i==2)
boxtext('为什么均值很小？大量平面易于保留；均值还排除了没有命中的射线。',40,419,880,16,'teal',True,max_h=30)
boxtext('打开快照的 24 条漏失都在背景墙上沿的掠射位置，不在抽屉；小均值仍不保证薄结构、隐藏空腔或拓扑完整。',40,454,880,15,'muted',max_h=42,leading=21)
end()

# 09
start('与真值相比：柜体仍有约 71–98 mm 深度误差','08 / SCENE RECONSTRUCTION ACCURACY',
      'C：2 mm 网格解码后，在整块可见目标柜体上评估；不是只测把手。',refs='[S2] comparison_summary.json；MAE / P95 为相机 Z 深度绝对误差。')
cols=[40,360,640]; widths=[320,280,280]
for x,w,t in zip(cols,widths,['相对 LIBERO 真值','关闭 · step60','打开 · step552']):
    rect(x,134,w,43,'navy');txt(t,x+16,146,16,'white',True)
C0=closed['decoded_C_target'];C1=opened['decoded_C_target']
data=[('柜体 MAE',f"{C0['mae_on_hits_m']*1000:.2f} mm",f"{C1['mae_on_hits_m']*1000:.2f} mm"),
      ('柜体 P95',f"{C0['p95_on_hits_m']*1000:.2f} mm",f"{C1['p95_on_hits_m']*1000:.2f} mm"),
      ('柜体误差 ≤ 5 mm','0.830%','1.213%'),('柜体射线命中率','100%','100%'),
      ('整幅图像 MAE','438.15 mm','361.06 mm')]
for i,vals in enumerate(data):
    y=177+i*41
    for x,w,v in zip(cols,widths,vals):
        rect(x,y,w,41,'white' if i%2==0 else 'gray');txt(v,x+16,y+10,17,'orange' if i==0 else 'ink',i==0)
boxtext('A → C：关闭 71.14 → 70.99 mm；打开 97.98 → 98.01 mm。主要偏差已存在于原始几何估计。',40,400,880,15,'teal',True,max_h=45,leading=22)
boxtext('100% 命中只表示射线撞上了某个预测表面，可能是错的墙或桌面；不代表 100% 几何正确。\n2 mm 是网格间距，不是本次重建精度。结论仅覆盖这两个单次预测快照。',40,451,880,14,'muted',max_h=42,leading=21)
end()

# 10
start('同一视角对比：布局可辨，深度与细节有偏差','09 / VISUAL EVIDENCE',
      '从保存场解码出的表面重新投射深度；未做尺度或位姿对齐。',refs='[S2] evaluation_001 保存的数值数组；本页据原始数组重画，见 figures manifest。')
xs=[84,296,508,720]
for x,label in zip(xs,['输入 RGB','LIBERO 真值深度','Astra 场解码深度','绝对深度误差']):txt(label,x,121,14,'ink',True)
for y,prefix,label in [(148,'closed','关闭'),(316,'open','打开')]:
    txt(label,40,y+64,15,'teal',True)
    for x,name in zip(xs,['rgb','gt_depth','decoded_depth','absolute_error']):
        image(f'{prefix}_{name}.png',x,y,156,156)
# Small aligned color scales, vector gradients sourced from matplotlib palette.
import matplotlib
for j,(x,w,cmap,label) in enumerate([(296,368,'viridis','深度 0 → 4 m'),(720,156,'magma','误差 0 → 0.5 m（饱和）')]):
    cm=matplotlib.colormaps[cmap]
    for k in range(128):
        rgba=cm(k/127); c.setFillColor(Color(*rgba[:3]));c.rect(x+k*w/128,H-486,w/128+.2,5,stroke=0,fill=1)
    txt(label,x,489,9,'muted')
end()

# 11
start('规划前先移除“机器人自己的重建”','10 / SELF-GEOMETRY FIX',
      '保留环境几何，再用原生机器人模型查询它，避免把自己当成障碍。',refs='[S5] planning_fields.py；RESULTS.txt；field_provenance.json。')
rect(40,136,420,250,'orangesoft',r=10)
small_label('原接入问题',60,153,'orange')
boxtext('照片中的机器人也被建进了场',60,190,380,23,'orange',True,max_h=65)
boxtext('原生机器人表面与 robot_* 组件重合。所有查询都在起点被拒绝，无法进入规划。',60,279,380,18,max_h=88)
rect(484,136,436,250,'tealsoft',r=10)
small_label('当前 v2 处理',504,153)
boxtext('移除完整的 robot_* 并集组件',504,190,394,22,'teal',True,max_h=65)
boxtext('关闭快照移除 6 个，打开快照移除 14 个。保留其他环境实体、差集和重叠部分。',504,279,394,18,max_h=88)
boxtext('修复后的环境场支持起点检查。但 robot_* 名称是模型判断，不是经过真值验证的语义分割。',40,413,880,18,'ink',max_h=57)
txt('查询范围沿用原场边界；域外或符号不明确的点按未知拒绝。',40,477,15,'muted')
end()

# 12
start('搜索在 7 维关节空间，场负责检查候选姿态','11 / HOW PLANNING WORKS',
      '先尝试关节直连；失败后用 RRT-Connect 搜索。当前没有沿场梯度导航。',refs='[S5] run_fixed_planning.py；[S6] robot_geometry.py / joint_planner.py；[S7] 原生碰撞评分。')
boxes=[(40,182,'候选关节角 q','7 个机械臂关节\n手指固定'),(270,182,'正运动学 FK','原生机器人表面\n变换到相机坐标'),(500,182,'查询 φ_env(p)','任一点占据或未知\n拒绝该姿态'),(730,182,'连接与搜索','关节插值检查边\n返回关节路径')]
for x,y,title,body in boxes:
    rect(x,y,190,131,'white',r=9);txt(title,x+13,y+16,18,'teal',True)
    boxtext(body,x+13,y+56,164,14,'muted',max_h=54,leading=23)
    if x<730:arrow(x+195,y+65,x+221,y+65)
small_label('输入：预测环境场 + 已知机器人模型、关节与相机配准',40,139)
rect(40,340,424,93,'tealsoft',r=8)
txt('13 个原生碰撞几何体',56,353,20,'teal',True)
boxtext('172,411 个表面点；覆盖半径 ≤5 mm。\n同时检查关节限位与机器人自碰撞。',56,386,390,14,max_h=43,leading=21)
rect(484,340,436,93,'bluesoft',r=8)
txt('先保存全部路径，再用真值评分',500,353,20,'blue',True)
boxtext('规划边最大关节步长 0.025 rad；\n事后真值检查细化至 0.005 rad。',500,386,404,14,max_h=43,leading=21)
boxtext('5 mm 是表面采样覆盖尺度，不是安全余量。有限姿态/表面采样不能证明连续路径绝对无碰撞。',40,459,880,15,'muted',max_h=41,leading=21)
end()

# 13
start('返回 40 条路径，其中 32 条通过真值采样','12 / PLANNING RESULTS',
      '两个静态快照 × 每个 10 组起终点 × 两个随机种子；共 20 组不同起终点组合。',refs='[S7] evaluation_001/metrics.json；[S5] RESULTS.txt；仅静态几何规划，没有执行抽屉操作任务。')
legend=[('teal','通过真值采样'),('orange','真值发生碰撞')]
for i,(col,lab) in enumerate(legend):rect(40+i*201,139,12,12,col);txt(lab,61+i*201,136,14)
for y,title,n_ok,n_bad in [(195,'关闭快照',18,2),(280,'打开快照',14,6)]:
    txt(title,40,y,20,'ink',True)
    x=195;totw=560
    rect(x,y-2,totw*n_ok/20,43,'teal',r=0);rect(x+totw*n_ok/20,y-2,totw*n_bad/20,43,'orange')
    txt(str(n_ok),x+12,y+8,20,'white',True);txt(str(n_bad),x+totw*n_ok/20+12,y+8,20,'white',True)
    txt('20 条返回',788,y+9,17,'ink',True)
rect(40,361,880,55,'white',r=8)
txt('38 条关节直连',60,377,20,'teal',True);txt('+',303,377,20,'muted');txt('2 条 RRT 搜索路径',354,377,20,'blue',True)
boxtext('32/40 是这组路径的采样通过数，不是“80% 重建精度”，也不是抽屉任务成功率。\n两种种子下的相同直连路径并不构成独立的几何重建试验。',40,443,880,17,'muted',max_h=50,leading=25)
end()

# 14
start('一个 RRT 案例：场可以参与路径搜索','13 / RRT EXAMPLE',
      '打开快照 · base_negative · seed 17；已保存路径的运动学回放。',refs='[S7] evaluation metrics / plans；[S9] video_001/previews/22_030.png。没有保存搜索树轨迹。')
image('rrt_native_scene.png',40,132,308,308)
image('rrt_field_panel.png',366,132,308,308)
rect(696,132,224,308,'white',r=9)
metric('20.11 s','规划计算时间',712,147,192,'blue')
metric('3','保存的路径顶点',712,240,192,'teal')
metric('4.725 mm','92 个真值姿态样本中的最小间隙',712,333,192,'teal')
boxtext('热图是当前末端高度的环境场水平切片；线是末端 XY 投影，不代表整臂间隙。\n同组真值场中直连也可行，因此这次绕行不能证明必要或最优。',40,453,880,15,'muted',max_h=43,leading=22)
end()

# 15
start('失败案例：场判为自由，原生夹爪却碰到柜体','14 / CONTACT DIAGNOSIS',
      '打开快照 · wrist_negative · seed 17 · 第 33 个真值采样姿态（零基编号）。',refs='[S8] failure_diagnostics_001/report.json；本页按原关节姿态重绘并复核原生接触，见 assets/collision_render_data.json。')
image('collision_native_scene_marked.png',40,133,330,330)
image('collision_contact_zoom.png',389,133,186,186)
boxtext('红圈：原生接触位置投影\n图像为事后诊断，未进入规划。',389,334,186,13,'muted',max_h=65,leading=21)
rect(594,133,326,330,'white',r=9)
small_label('同一位置，两种判断',612,151,'orange')
txt('真实几何',612,186,16,'muted');txt('已穿入 0.200 mm',612,213,23,'orange',True)
txt('Astra 预测环境场',612,265,16,'muted');txt('φ = +150.580 mm',612,292,23,'teal',True)
boxtext('此姿态的全部 172,411 个机器人表面采样点也被预测为空闲。',612,352,290,16,max_h=75)
boxtext('这直接证实接触区域的预测几何有偏差。φ 数值不是实际安全间隙，也不是整场重建误差。',40,478,880,14,'muted',max_h=24)
end()

# 16
start('本轮结论：场可查询、可规划，几何还不够准确','15 / WHAT THE EVIDENCE SUPPORTS',
      '区分“表示是否有效”和“输入图像能否提供足够准确的三维信息”。',refs='综合 [S1–S8]；下方为建议的后续实验，不是已经取得的结果。')
for x,w,tag,title,body in [
 (40,278,'表示能力','可以表达与保存','CSG 能表达实体和空腔；2 mm 网格在多数已预测表面上保留精度较好。'),
 (340,278,'重建精度','仍有厘米级偏差','可见柜体 MAE 约 71 / 98 mm；高网格分辨率不能修正模型猜错的形状。'),
 (640,280,'规划作用','能搜索，尚不能保证安全','原生机器人可用场做可行性检查；8 条返回路径仍在真值中碰撞。')]:
    rect(x,134,w,229,'white',r=10);small_label(tag,x+17,151)
    boxtext(title,x+17,184,w-34,21,'ink',True,max_h=63)
    boxtext(body,x+17,260,w-34,16,'muted',max_h=85,leading=24)
txt('下一轮建议：固定相同查询，逐项改善几何信息',40,395,20,'teal',True)
boxtext('① 提供已知相机标定和尺度，隔离尺度误差。\n② 加入多视角，单独评估抽屉开口、柜体边缘与遮挡区域。\n③ 显式记录未知和不确定区域，同时复测重建误差与真实碰撞。',40,431,880,15,'ink',max_h=65,leading=22)
end()

# 17
start('数据来源与复现入口','APPENDIX / EVIDENCE INDEX',
      '所有结论来自现有 drawer003 实验；本次只新增幻灯片、图表与生成脚本。',refs='项目根目录：/workspace/gpt_field；完整来源哈希保存在本幻灯片目录的 evidence_*.json / deck_manifest.json。')
refs=[
 ('S1','生成约束与场定义','scene_test/astra_scene/inferences_001/instructions.txt','scene_test/astra_scene/field_csg.py'),
 ('S2','重建与保存精度','scene_test/astra_scene/comparison_summary.json','scene_test/astra_scene/evaluate_fields.py'),
 ('S3','掠射边界漏失诊断','scene_test/astra_scene/runs/diagnostic_open_wall_misses_001/report.json',''),
 ('S4','原始 Astra 场与保存网格','scene_test/astra_scene/inferences_001/fields/','scene_test/astra_scene/runs/fields_001/'),
 ('S5','规划接入与自体移除','scene_test/field_planning_v2/RESULTS.txt','scene_test/field_planning_v2/planning_fields.py'),
 ('S6','机器人表面与搜索算法','scene_test/field_planning/robot_geometry.py','scene_test/field_planning/joint_planner.py'),
 ('S7','路径和真值评分','scene_test/field_planning_v2/runs/planning_001/plans.json','scene_test/field_planning_v2/runs/evaluation_001/metrics.json'),
 ('S8','失败接触诊断','scene_test/field_planning_v2/runs/failure_diagnostics_001/report.json',''),
 ('S9','规划演示视频','scene_test/field_planning_v2/video_001/planning_highlights.mp4','scene_test/field_planning_v2/video_001/planning_all_queries.mp4'),
 ('S10','此前 waypoint 目标场','runs/agent_speedup_v1_drawer_zero_shot_003/trial/frozen_source/contact_skills.py','runs/agent_speedup_v1_drawer_zero_shot_003/trial/frozen_source/fine_distance_planner.py')]
y=125
for i,(key,label,p1,p2) in enumerate(refs):
    x=40 if i<5 else 500
    yi=125+(i if i<5 else i-5)*73
    txt(f'[{key}]  {label}',x,yi,13,'teal',True)
    for j,path in enumerate([p1,p2]):
        if path:
            # Keep reference paths as selectable English text, sized to column width.
            sz=min(9,408/pdfmetrics.stringWidth(path,'CN',1))
            txt(path,x,yi+24+j*14,sz,'muted')
            c.linkURL(str(PROJECT/path),(x,H-(yi+24+j*14)-12,x+408,H-(yi+24+j*14)),relative=0,thickness=0)
end()
assert page_no==page_count
c.save()

# Preserve traceability without modifying any consumed experiment sources.
sources={}
for source in E['sources']:
    p=Path(source['path']);sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha==source['sha256'],p
    sources[str(p)]=sha
for rel in ['scene_test/field_planning_v2/planning_fields.py',
            'scene_test/field_planning_v2/runs/planning_001/plans.json',
            'scene_test/field_planning_v2/runs/evaluation_001/metrics.json',
            'scene_test/field_planning_v2/runs/failure_diagnostics_001/report.json',
            'runs/agent_speedup_v1_drawer_zero_shot_003/trial/frozen_source/contact_skills.py',
            'runs/agent_speedup_v1_drawer_zero_shot_003/trial/frozen_source/fine_distance_planner.py']:
    p=PROJECT/rel;sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest={
 'pdf':str(out),'page_count':page_count,'language':'zh-CN','page_size_pt':[W,H],
 'scope':'Existing single RGB Astra drawer003 closed step60/open step552 reconstruction and field_planning_v2; no new inference or planning.',
 'source_sha256':sources,'pdf_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'images_manifest':str(ASSETS/'evidence_figures_manifest.json'),
 'font_license':str(ASSETS/'OFL.txt'),
 'caveats':['Planning directly queries self-excluded analytic CSG, not original saved2mm samples.',
            'Scene MAE is visible-camera-Z depth error, not 3D closest-surface distance.',
            'B uses six model-derived view rays and is not a subtraction of A/C.',
            '32/40 is finite-sampled path count, not reconstruction accuracy or task success.',
            'No continuous collision certificate or complete hidden geometry evaluation.']}
(ROOT/'deck_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(ROOT/'layout_text.json').write_text(json.dumps(text_records,ensure_ascii=False,indent=2)+'\n')
print(out)
print('pages:',page_count,'bytes:',out.stat().st_size)
