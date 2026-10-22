#!/usr/bin/env python3
"""Chinese technical slides from frozen SpeedUp v1 evidence. Writes new presentation artifacts only."""
from pathlib import Path
import sys,json,math,hashlib,re
ROOT=Path(__file__).resolve().parent
PROJECT=ROOT.parent.parent
ASSETS=ROOT/'assets'
sys.path.insert(0,str(PROJECT/'scene_test/astra_field_slides/vendor'))
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor,Color
from reportlab.lib.utils import ImageReader
from PIL import Image
W,H=960,540
C={'bg':'#F5F7F9','ink':'#14293A','muted':'#536575','line':'#D9E2E8',
   'teal':'#007F82','tealsoft':'#E2F2F0','blue':'#245FA4','bluesoft':'#E7EFF9',
   'orange':'#C45C35','orangesoft':'#FCEBE3','white':'#FFFFFF','navy':'#132E41',
   'gray':'#ECEFF2','red':'#BD3F42','gold':'#EAC35D'}
for name,label in [('CN','Regular'),('CNB','Bold')]:
 pdfmetrics.registerFont(TTFont(name,str(ASSETS/f'NotoSansSC-{label}.ttf')))
R=json.loads((ROOT/'evidence_results.json').read_text())
M=json.loads((ROOT/'evidence_methods.json').read_text())
F=json.loads((ROOT/'evidence_framework.json').read_text())
B=R['runs']['bowl_zero_shot_002'];D=R['runs']['drawer_zero_shot_003']
out=ROOT/'speedup_v1_framework_methods_results_zh.pdf'
c=canvas.Canvas(str(out),pagesize=(W,H),pageCompression=1)
c.setTitle('SpeedUp v1：系统框架、技术方法与实测结果')
c.setAuthor('gpt_field experiment review')
c.setSubject('Frozen drawer003 and bowl002 robot skill controller, goal-distance planning, visual servo and contact feedback')
page_count=24
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
        for ch in re.findall(r'[A-Za-z0-9_./+−=<>*:-]+|[^A-Za-z0-9_./+−=<>*:-]',para):
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


def card(x,y,w,h,title,body,accent='teal',size=16):
    rect(x,y,w,h,'white',r=10)
    boxtext(title,x+16,y+15,w-32,19,accent,True,max_h=57,leading=26)
    boxtext(body,x+16,y+76,w-32,size,'muted',max_h=h-86,leading=size*1.45)

def table(headers,rows,x=40,y=133,widths=(310,285,285),row_h=42,size=16):
    xx=x
    for h,w in zip(headers,widths):
        rect(xx,y,w,44,'navy');txt(h,xx+13,y+12,size,'white',True);xx+=w
    for i,vals in enumerate(rows):
        xx=x;yy=y+44+i*row_h
        for j,(v,w) in enumerate(zip(vals,widths)):
            rect(xx,yy,w,row_h,'white' if i%2==0 else 'gray')
            txt(v,xx+13,yy+11,size,'ink',False);xx+=w

# 01: title
start('SpeedUp v1', 'FRAMEWORK · METHODS · RESULTS',
      '模型监督的宏技能 + 局部视觉与状态反馈闭环',refs='技术细节以 drawer003 冻结版本为主；结果另列 bowl002。所有证据来自已保存运行。[S1–S10]')
image('drawer_zero_shot_003_handle_released_agent.png',588,133,332,332)
badge('语言与视觉决策 / 本地技能执行',40,135,315)
boxtext('输入如何变成技能、\n距离场、路径与动作？',40,188,510,32,'ink',True,max_h=95,leading=44)
for y,n,s in [(303,'01','框架：谁负责理解，谁负责执行'),(354,'02','方法：指点、伺服、接触与规划细节'),(405,'03','结果：完成条件、耗时、调用与限制')]:
    txt(n,40,y,18,'teal',True);txt(s,80,y,18)
txt('图：drawer003 把手释放后的真实运行画面',588,476,11,'muted')
end()

# 02: full framework
start('总框架：模型选技能，执行器闭环完成动作','01 / SYSTEM FRAMEWORK',
      '距离场是局部 waypoint 执行模块；抓取接触由另一条反馈链处理。',refs='[S1] agent_speedup_v1.py / client；[S2–S6] 技能、局部规划、视觉与状态反馈。')
rect(40,134,176,214,'white',r=10)
txt('输入',55,151,20,'blue',True)
boxtext('任务语言\n当前外部 RGB\n当前腕部 RGB\n机器人 state8\n压缩结果 / 记忆',55,191,148,16,leading=29,max_h=151)
rect(248,134,234,97,'bluesoft',r=10)
txt('Astra / medium',264,150,21,'blue',True)
boxtext('语义决策：skill + params\n按需指点：cell / point ID',264,187,200,14,max_h=40,leading=20)
arrow(220,179,242,179,'blue')
rect(248,261,234,87,'navy',r=10)
txt('本地 Skill Executor',264,277,19,'white',True)
boxtext('分派、状态机、失败处理',264,312,200,14,'#D0DEE6',max_h=25)
arrow(365,236,365,255,'blue')
rect(526,133,394,91,'tealsoft',r=10)
txt('移动分支',542,146,18,'teal',True)
boxtext('waypoint → 目标距离场 → Planner\n→ PathTracker → 位移与姿态保持',542,180,360,15,max_h=44,leading=22)
rect(526,247,394,101,'orangesoft',r=10)
txt('接触分支',542,260,18,'orange',True)
boxtext('Point + Align → GuardedApproach\n→ Close → VerifyHold / Pull / Release',542,295,360,15,max_h=44,leading=22)
arrow(488,288,516,288,'teal');line(501,288,501,179,'teal',2);arrow(501,179,518,179,'teal')
rect(248,387,672,62,'white',r=9)
txt('7D 控制动作 → LIBERO / OSC → 新 RGB 与机器人状态',267,406,19,'ink',True)
arrow(724,353,724,380,'orange')
arrow(365,381,365,354,'blue');txt('逐步反馈',385,360,11,'blue')
line(924,179,940,179,'teal',2);line(940,179,940,417,'teal',2);arrow(940,417,924,417,'teal')
line(245,417,126,417,'blue',2);arrow(126,417,126,354,'blue')
boxtext('技能结束：outcome / ledger → 下一次模型决策；技能执行中：每动作反馈，按需请求指点。',40,473,880,14,'muted',max_h=24)
end()

# 03: actual inputs
start('模型看到当前双 RGB 与结构化状态摘要','02 / INPUTS',
      '实图来自 drawer003 首次 decision（step89，完成自动标定与探测之后）。',refs='[S1] request000001；输入图逐字节提取，未重新渲染。state8 本地派生摘要；无深度或物体真值。')
txt('外部相机 · 512 × 512',40,132,15,'blue',True);image('drawer003_input_external.png',40,160,237,237)
txt('腕部相机 · 512 × 512',296,132,15,'blue',True);image('drawer003_input_wrist.png',296,160,237,237)
rect(555,132,365,304,'white',r=10)
txt('state8 → state_summary',572,149,19,'teal',True)
boxtext('[x, y, z, roll, pitch, yaw, qf1, qf2]',572,188,330,13,'muted',max_h=42)
y=230
for s in ['末端世界位置、工具轴、指间距','抓持状态、已拉距离、末次结果','最近 12 条技能 ledger；drawer 另有最近 3 次决策摘要','外部图像方向：由自身探测动作在线估计']:
    y+=bullet(s,572,y,328,14,gap=7)
boxtext('还有任务文字、目标外观说明和机器人几何先验。没有示范、深度、物体位姿或模拟器接触真值。\nstateless 表示不保留完整聊天 / 旧图像；显式摘要仍携带任务记忆。',40,451,880,15,'muted',max_h=44,leading=22)
end()

# 04: interface
start('Astra 返回技能 JSON，本地代码负责分派','03 / MODEL CONTRACT',
      'gpt-6-astra · medium · ChatGPT subscription · strict JSON schema。',refs='[S1] frozen client / policy；drawer003 request000008 对应 step248 的成功把手抓取决策。')
rect(40,134,430,315,'navy',r=10)
small_label('真实输出节选（省略 situation 与 null 字段）',58,150,'#85C7CB')
lines=['{','  "skill": "grasp_with",','  "params": {','    "target": "top_drawer_handle",','    "strategy": "frontal_bar_pinch",','    "approach": "world_-y",','    "max_travel_m": 0.12','  }','}']
for i,s in enumerate(lines):txt(s,58,188+i*26,15,'white')
rect(494,134,426,315,'white',r=10)
txt('决策对象的三个部分',513,151,21,'blue',True)
y=195
for t,s in [('situation','对当前视觉关系的简短描述。'),('skill','从六个宏技能中选择一个。'),('params','目标、策略、抓取侧、接近方向、相对 waypoint 或拉动上限。')]:
    txt(t,513,y,17,'teal',True);h=boxtext(s,513,y+29,386,16,'muted',max_h=60);y+=h+52
boxtext('这是结构化输出后的本地技能调用；模型不直接输出低层 7D 控制动作，也不逐点输出距离场。',40,469,880,15,'muted',max_h=25)
end()

# 05: inventory
start('六个宏技能把长动作序列封装为接口','04 / SKILL LIBRARY',
      '自动 calibrate 属于初始化，不是模型可选的第七个宏技能。',refs='[S1, S2] frozen skill schema、start_macro 与技能状态机；多 waypoint / 抓拉合并为 drawer003 版本能力。')
rows=[('reach','相对 waypoint','保持姿态与夹爪，移动末端'),('grasp_with','目标 + 策略 + 侧 / 方向','对齐 → 接近 → 闭合 → 验证'),('pull','总行程 2–16 cm','反向拉动；包含 2 cm 验证拉'),('transport','waypoint + 目的地','携物移动，逐步监测抓持'),('place_release','waypoint + 目的地','移到上方 → 降低 → 松开退让'),('retreat','无','沿接近轴反向退 3 cm')]
table(['技能','模型给出的参数','本地执行内容'],rows,widths=(180,270,430),row_h=42,size=15)
boxtext('waypoint 每轴限制 ±7.5 cm；drawer003 可附加最多 2 个偏移（一次最多 3 段）。\nfrontal_bar_pinch 可附 max_travel_m，将抓住把手和拉开合并在同次决策中。',40,453,880,15,'muted',max_h=44,leading=22)
end()

# 06: event loop
start('模型按事件调用；局部动作按反馈推进','05 / EXECUTION LOOP',
      '20 Hz 是模拟控制频率；模型等待期间模拟器暂停，不代表墙钟实时 20 Hz。',refs='[S1] act / observe_transition；[S7, S8] manifest simulation_pauses_for_api 与请求记录。')
steps=[('初始化','自动标定与探测\n不调用模型'),('选择技能','新 RGB + 摘要\n请求一次 decision'),('执行原语','动作 → 观测 → 更新\n本地状态机循环'),('完成或失败','记录 outcome / ledger\n再请求 decision')]
for i,(title,body) in enumerate(steps):
    x=40+i*226
    rect(x,155,202,144,'white',r=9);txt(f'{i+1:02d}',x+16,170,13,'teal',True)
    txt(title,x+16,200,20,'ink',True);boxtext(body,x+16,241,170,14,'muted',max_h=44,leading=22)
    if i<3:arrow(x+205,226,x+222,226)
line(820,309,820,333,'blue',2);line(820,333,365,333,'blue',2);arrow(365,333,365,305,'blue')
rect(40,367,424,94,'bluesoft',r=10)
txt('技能内部的例外：pointing',57,383,18,'blue',True)
boxtext('Align 需要视觉目标时另发 coarse / fine 请求；不会每个动作都问模型怎么动。',57,416,388,15,max_h=44,leading=22)
rect(484,367,436,94,'tealsoft',r=10)
txt('外部评估器每步检查任务成功',501,383,18,'teal',True)
boxtext('raw LIBERO success 可在原语中途终止。hold_pass 只是局部抓持测试结果。',501,416,400,15,max_h=44,leading=22)
end()

# 07: calibration
start('先测机器人自身响应，再解释图像与接触','06 / ONLINE CALIBRATION',
      '不提供场景深度；利用可观测的手指位置和自身运动建立局部参考。',refs='[S2, S6] Calibrate / external_axes.py；[S8] agent_stats.json，drawer003 在线标定记录。')
card(40,134,424,288,'夹爪标定','空中闭合曲线 + 打开与稳定段。\n记录空闭合指间距、开指噪声。\n\ndrawer003：空闭合底值 1.186 mm；\n接触阈值 max(0.1 mm, 3σ) = 0.1 mm。',size=17)
card(484,134,436,288,'外部图像方向标定','3 次总体回到附近的 XY 探测动作。\n用图像位移与实测世界位移拟合映射。\n\n图像向右 ≈ [0.061, 0.998, 0]\n图像向上 ≈ [−0.996, −0.085, 0]',accent='blue',size=17)
callout('Δs_image ≈ J_ext · Δp_world,XY      （外部相机方向不能直接套用腕部相机轴）',40,442,880,48,size=16)
end()

# 08: pointing
start('视觉指点：先选区域，再选局部接触点','07 / COARSE-TO-FINE POINTING',
      '实图：drawer003 成功把手抓取的请求 9 / 10；模型返回 cell28 → point44。',refs='[S1, S5] request000009 / request000010、mark_prompting.py；图像为实际发送给模型的字节。')
txt('Coarse：8 × 8 区域',40,131,17,'blue',True)
image('drawer003_handle_coarse_success.png',40,164,280,280)
arrow(339,289,377,289,'blue')
txt('Fine：局部裁剪 + 8 × 8 点',398,131,17,'teal',True)
image('drawer003_handle_fine_success.png',398,164,280,280)
rect(704,164,216,280,'white',r=9)
boxtext('横杆版本',720,180,184,20,'teal',True,max_h=50)
boxtext('128 × 128 裁剪\n放大 4 倍\n\n普通碗：96 × 96\n放大 5 倍\n\n不可见可返回 0',720,226,184,16,'muted',max_h=190,leading=26)
boxtext('把手指点图按重力转正，结果映回腕图；NCC 跟踪只读无标记原图。模型选的是像素点，不是 3D 位姿。',40,468,880,14,'muted',max_h=27)
end()

# 09: servo
start('像素点变成毫米级动作：在线图像 Jacobian','08 / WRIST VISUAL SERVO',
      '小幅自身探测 + NCC 跟踪；不把像素误差直接假装成已知的米制深度。',refs='[S5] wrist_servo.py:295–360,462–529,850–857,970–998,1075–1114；drawer003 冻结版本。')
rect(40,135,880,77,'navy',r=10)
txt('Δs ≈ J Δp_plane       J ∈ R^(2×2), 单位 px/m',60,151,21,'white',True)
txt('Δp_cmd ∝ − pinv(J) (s − s*)',60,184,17,'#85C7CB',True)
card(40,235,274,208,'1. 建立局部映射','探测移动、读取实测位移、跟踪像素。普通二维探测目标 2 mm；横杆跨轴探测出 / 回各 6 mm。',size=15)
card(334,235,282,208,'2. 持续更新','普通二维 J 用 Broyden；横杆更新标量 κ。使用干净、小旋转样本，拒绝异常更新。目标 s* 补偿 5 cm 腕相机安装偏移。',accent='blue',size=15)
card(636,235,284,208,'3. 检查收敛','阈值 min(6 px, max(2 px, κ×3 mm))，连续 3 帧满足。横杆只视觉纠正跨杆方向，沿杆由本体位置保持。',accent='orange',size=15)
boxtext('κ 为在线估计的 px/m；3 mm 是伺服阈值的尺度参数，不是经场景真值验证的定位精度。',40,470,880,15,'muted',max_h=25)
end()

# 10: rim skill
start('rim_pinch：用反馈决定接近深度与闭合时机','09 / BOWL GRASP STRATEGY',
      '模型选碗沿左 / 右侧；本地执行器把碗壁放入两指之间。',refs='[S2, S5] GuardedApproach / Grasp / VerifyHold；[S7] bowl002 hold_pass 原视频帧。技术常量为 drawer003。')
image('bowl_zero_shot_002_hold_pass_wrist.png',40,150,270,270)
badge('实录：碗抓持测试通过后',40,432,270)
phases=[('Point + Align','先横向对齐选定的碗沿'),('GuardedApproach','每 20 mm 分段下降并刷新对齐'),('接触 → 回退 → Close','接触后回退 4 mm，再按测量条件闭合'),('VerifyHold','上提 25 mm；检查夹距与运动是否支持抓持')]
for i,(t,s) in enumerate(phases):
    y=137+i*78;rect(342,y,578,67,'white',r=8);txt(f'{i+1:02d}',356,y+13,16,'teal',True)
    txt(t,397,y+11,18,'ink',True);txt(s,397,y+39,14,'muted')
boxtext('接触代理：开指向内偏转超过阈值，或实测 / 命令接近量比值显著下降；未读取模拟器接触身份。',342,466,578,14,'muted',max_h=36,leading=19)
end()

# 11: sensor details and handle
start('frontal_bar_pinch：抓住把手，再验证与拉动','10 / HANDLE GRASP & CONTACT LOGIC',
      '水平接近、夹爪转向、跨杆对齐；避免沿均匀把手纹理误追踪。',refs='[S2, S5] Grasp / VerifyHold / frontal_bar_pinch；[S8] step248–441 的宏结果。')
rect(40,135,880,68,'navy',r=10)
txt('Orient → Point / Align → GuardedApproach → 回退 5 mm → Close → Pull-test',58,157,18,'white',True)
card(40,226,274,211,'接触事件','满足启用与小旋转条件后，近 2 步响应比均值 < 0.6×本次自由响应，或手指偏转超阈值。',size=16)
card(334,226,282,211,'闭合与验证','第 15 步夹距 > 空闭合基线 +0.5 mm；再回拉 20 mm。夹距跌破空闭合底值 +1 mm 则判 hold_lost。',accent='blue',size=16)
card(636,226,284,211,'抓拉合并','模型给 0.12 m 总行程。验证拉动计入其中；成功宏报告实际拉动约 0.1198 m。',accent='orange',size=16)
boxtext('hold_pass 是传感器代理测试；接触可能来自意外遮挡，局部技能并不构成完整碰撞安全证明。',40,466,880,15,'muted',max_h=30)
end()

# 12: field equations
start('距离场来自 waypoint，不来自场景重建','11 / LOCAL FIELD DEFINITION',
      '相对位移由模型给出；本地代码设置目标、局部网格与距离查询。',refs='[S2] contact_skills.py:894–956；[S3] FineCompactField / FineDistancePlanner；实际逻辑网格不是旧客户端33³元数据。')
rect(40,134,880,64,'navy',r=10)
txt('g = p_start + Δp_model',60,153,25,'white',True)
txt('目标固定；后续用新的实测 p 跟踪',513,159,16,'#85C7CB')
card(40,220,424,196,'查询势值 U：无量纲','U(v) = ||v − g|| / 0.08\n格点值按需计算，再三线性插值。\n它不是障碍物表面的 signed distance。',size=18)
card(484,220,436,196,'实际规划距离 D：单位 m','D(p) = max(0, ||p − g|| − ε)\nε = 2.5 mm，表示目标球半径。\n规划器不用 U 的梯度来决定方向。',accent='blue',size=18)
boxtext('161³ 逻辑格点，1 mm 间距，起点周围每轴 ±8 cm；不完整物化 417 万个值。\nrepulsors / blocked_boxes / unknown_boxes 均为空：没有编码抽屉、碗或障碍表面。',40,440,880,16,'muted',max_h=51,leading=24)
end()

# 13: planner detail
start('本次实际走解析最短直线；A* 是备用能力','12 / FINE DISTANCE PLANNER',
      '空障碍、均匀密度且直段有效时，解析求到目标球的路径；不需要搜索整个网格。',refs='[S3] fine_distance_planner.py:180–431；[S9] 已有回放30次局部规划均为 analytic_uniform_goal_distance。')
rect(40,139,520,296,'tealsoft',r=10)
txt('已运行分支',59,155,21,'teal',True)
boxtext('1. 检查实测起点、workspace 与局部域。\n2. 直接朝目标球边界生成路径。\n3. 每段 1 cm，最多 10 段 / 10 cm。',59,200,480,18,leading=34,max_h=124)
txt('p_end = g + ε (p − g) / ||p − g||',59,344,19,'teal',True)
boxtext('距离和局部几何路径按需生成，不保存全域求解距离图。',59,385,480,15,'muted',max_h=42)
rect(584,139,336,296,'white',r=10)
txt('通用备用 A*',603,155,21,'blue',True)
boxtext('1 mm lattice，26 邻接。\n边代价 = 边长 × 端点平均密度。\nρ ≥ 1，可含 Gaussian repulsors。\n启发式：到目标球的欧氏距离。\n只保存已探索节点。',603,200,298,16,leading=30,max_h=181)
boxtext('本次没有障碍项，不能把备用 A* 能力说成已经验证了场景避障；模型需要用中间 waypoint 选路。',40,461,880,16,'muted',max_h=44,leading=23)
end()

# 14: tracking
start('路径执行依赖实测位置，并持续重新规划','13 / PATH TRACKER & OSC',
      '局部规划生成几何参考；PathTracker 把参考转换成有界的控制命令。',refs='[S4] tracking.py；[S2] WaypointReach / HoldingActionAdapter。OSC 为底层位姿增量控制。')
steps=[('实测 p_t','投影到剩余路径'),('参考点 r_t','沿路径前看 3 cm'),('位移命令','方向朝参考点，模长 ≤5 cm'),('OSC 执行','显式保持目标姿态 / 夹爪')]
for i,(t,b) in enumerate(steps):
    x=40+i*226;rect(x,144,202,116,'white',r=9);txt(t,x+15,162,20,'teal',True)
    boxtext(b,x+15,207,171,15,'muted',max_h=43,leading=22)
    if i<3:arrow(x+205,201,x+222,201)
rect(40,286,424,162,'bluesoft',r=10)
txt('重新规划触发',57,303,20,'blue',True)
boxtext('执行 10 个动作，或到达当前路径端点（2.5 mm）。\n固定原 goal / 局部域，以新实测位置重规划。',57,345,390,17,max_h=92,leading=27)
rect(484,286,436,162,'tealsoft',r=10)
txt('到达判据与停止条件',501,303,20,'teal',True)
boxtext('||p − g|| ≤5 mm 且姿态误差 ≤0.05 rad。\n无进展、异常跃迁、场界越出、姿态漂移或抓持丢失可提前返回。',501,345,400,17,max_h=92,leading=27)
txt('5 cm 是命令上限，不是保证每一步真实移动 5 cm；每次都会读回实测反馈。',40,474,15,'muted')
end()

# 15: field actual slices
start('实际局部场：颜色表示到 waypoint 的距离','14 / RECORDED FIELD EXAMPLE',
      'drawer003 · step552 后的 reach：Δp = [−40, +75, +61.7] mm。',refs='[S9] 既有回放 run002 / field007、runtime_plans 与 field_actions；本页只读取保存数据绘图。')
image('waypoint_field_slices.png',40,139,880,278)
boxtext('白虚线：局部规划路径；橙线：实测末端轨迹；星号：目标。三张切片穿过目标，轨迹为投影。',40,433,880,15,'muted',max_h=43,leading=22)
callout('这些密集颜色是事后诊断采样；原规划器解析计算路径，不读取这张热图，也没有从图中识别障碍。',40,466,880,34,size=12)
end()

# 16: transport and place
start('携物与放置也有局部反馈，但不是全场避障','15 / TRANSPORT & PLACE',
      '抓到物体后，模型通过外部图像选目的地与中间 waypoint；局部执行器监测状态。',refs='[S1, S2] transport / place_release；drawer003 frozen prompt-5 与原语实现。')
card(40,135,424,293,'transport：运动中检查抓持','使用 waypoint / PathTracker，保持抓取姿态。\n\n夹距异常下降 → lost_in_transport。\n夹距突增等代理信号 → held_object_contact。\n异常响应提前退出，回传证据给模型。',size=17)
card(484,135,436,293,'place_release：先到上方，再接触式降低','水平 / 向上移到目标上方；降低时根据实测响应寻找支撑，再松开退让。\n\n把手释放是单独分支：松开、退让、抬升、转回朝下。',accent='blue',size=17)
boxtext('若抓取失败，先用有限本地重试；耗尽后模型需改 hover、抓取侧或策略。两次选定成功记录的实际本地重试数均为 0。',40,452,880,16,'muted',max_h=48,leading=24)
end()

# 17: speed mechanism
start('“SpeedUp”的设计重点是减少模型往返','16 / WHERE SPEED COMES FROM',
      '高频动作、对齐和状态判断留在本地；模型负责较低频的语义选择。',refs='[S1, S8] drawer003 step818 实际 transport 参数；两段额外 waypoint 为 prompt-5 新能力。')
card(40,134,274,252,'更长的本地执行','一个宏技能可运行许多动作。视觉追踪、接近、闭合、抓持验证不用逐步生成文字动作。',size=16)
card(334,134,282,252,'压缩模型输入输出','只发当前双图 + 显式摘要。指点只返回 ID。stateless 没有累积的整段对话与图像历史。',accent='blue',size=16)
card(636,134,284,252,'合并语义相邻步骤','drawer003 支持多 waypoint、抓后直接拉、释放后自动复位；减少技能边界的模型请求。',accent='orange',size=16)
rect(40,411,880,73,'navy',r=10)
txt('真实示例：step818 的一次 transport 决策',57,423,17,'white',True)
txt('[0, +60, 0] → [0, 0, +75] → [0, 0, +75] mm（每段相对当前实测位置）',57,454,16,'#85C7CB')
end()

# 18: main results
start('两条已完成记录：11 / 21 次模型请求','17 / RECORDED TASK RESULTS',
      '同为 init0 / seed7；各一条完整记录。bowl002 和 drawer003 使用不同提示修订。',refs='[S7, S8] summary.json / agent_stats.json / manifest.json；成功依据原始 LIBERO 目标。')
rows=[('任务 / 版本','碗放盘子 · prompt-2','开抽屉放碗 · prompt-5'),('raw LIBERO success','是','是'),('控制步数','449','967'),('决策 + 指点 = 模型请求','9 + 2 = 11','15 + 6 = 21'),('去视频控制墙钟','135.41 s','234.54 s'),('非允许接触记录','有','有')]
table(['指标','bowl002','drawer003'],rows,row_h=42,size=16)
boxtext('两次成功均不等于 collision-free success；本协议碰撞只记录，不阻止原始目标成功。\n各一次成功记录不能作为跨初始状态成功率，也不能视为完全相同版本的任务对比。',40,451,880,15,'muted',max_h=44,leading=22)
end()

# 19: drawer story
start('drawer003：从把手失败恢复，到碗进入抽屉','18 / DRAWER EPISODE',
      '真实视频关键帧；step 表示已执行动作数，不是视频秒数。',refs='[S8] episode.mp4 / events / trace；截图来自原视频，方向对齐模型正向视图。')
frames=[('initial','step0','初始场景'),('handle_released','step552','把手已释放，手转回朝下'),('bowl_hold_pass','step818','碗抓持验证通过'),('terminal','step967','LIBERO 目标触发')]
for i,(name,step,caption) in enumerate(frames):
    x=40+i*226;image(f'drawer_zero_shot_003_{name}_agent.png',x,154,202,202)
    txt(step,x,368,16,'teal',True);boxtext(caption,x,398,202,15,max_h=48,leading=22)
boxtext('前期一次 reach 无进展、一次把手 hover 跨杆偏离；模型调整后，成功抓拉约 12 cm。\n终止发生在最后 release 的第 3 个动作；松开与退让宏尚未完整结束。',40,456,880,15,'muted',max_h=44,leading=22)
end()

# 20: bowl story
start('bowl002：成功标志先于完整松开动作','19 / BOWL EPISODE',
      '任务完成判据来自模拟器；本地宏技能和视觉判断不决定何时评分成功。',refs='[S7] episode.mp4 / events / trace；final_macro=place_release，final_primitive=lower，primitive.done=false。')
frames=[('initial','step0 · 初始场景'),('hold_pass','step387 · hold_pass'),('terminal','step449 · LIBERO success')]
for i,(name,label) in enumerate(frames):
    x=40+i*300;image(f'bowl_zero_shot_002_{name}_agent.png',x,145,280,280);txt(label,x,438,16,'teal',True)
boxtext('最终成功发生在 lower 阶段，夹爪仍闭合。可以说“满足 LIBERO 放置目标”，不能据此说完整释放并撤回已经执行。',40,473,880,14,'muted',max_h=27)
end()

# 21 timing
start('控制墙钟、模拟时间、视频长度是不同量','20 / TIMING ACCOUNTING',
      '主要耗时指标：T_net = control_elapsed_s − video_render_s。',refs='[S7, S8] timing_components；model_pipeline_s 已包含在 decision_s，图中未重复计算。')
max_time=350
for y,label,r in [(154,'bowl002',B),(250,'drawer003',D)]:
    txt(label,40,y+2,19,'ink',True)
    model=r['timing_components']['model_pipeline_s'];sim=r['timing_components']['simulator_step_s'];other=r['control_without_video_s']-model-sim;video=r['timing_components']['video_render_s']
    x=184
    for v,col in [(model,'blue'),(sim,'teal'),(other,'orange'),(video,'#CCD6DE')]:
        ww=v/max_time*610;rect(x,y,ww,39,col)
        if ww>38:txt(f'{v:.1f}',x+7,y+10,14,'white' if col!='#CCD6DE' else 'ink',True)
        x+=ww
    txt(f"净耗时 {r['control_without_video_s']:.2f} s",184,y+49,16,'ink',True)
for i,(lab,col) in enumerate([('模型调用','blue'),('模拟器调用墙钟','teal'),('其他控制','orange'),('视频渲染','#CCD6DE')]):
    x=40+i*225;rect(x,342,11,11,col);txt(lab,x+20,338,14,'muted')
rect(40,381,880,107,'white',r=9)
txt('drawer003 的计时口径',57,395,18,'teal',True)
boxtext('去视频控制：234.54 s    ｜    含渲染控制：340.75 s    ｜    总墙钟：348.96 s\n模拟动作时间：967 / 20 = 48.35 s    ｜    注释视频：204.7 s\n135.41 s 属于 bowl002；旧 drawer002 的 24.82 分钟是另一条未完成尝试。',57,428,844,14,'muted',max_h=60,leading=19)
end()

# 22: historical and limits
start('相对历史 Direct，v1 模型调用更少','21 / HISTORICAL REFERENCE',
      '同一命名碗任务与 seed7；提示、历史、控制接口、标定与碰撞监测工作量不同。',refs='[S7, S10] 两条成功历史记录；Direct 净耗时按保存的 control − video 事后计算，未重跑。')
rows=[('控制接口','每次生成 5 个直接动作','模型选宏技能 + 本地闭环'),('模型请求','26','11'),('低层动作数','127','449'),('模型调用墙钟','242.23 s','81.56 s'),('去视频控制墙钟','249.45 s','135.41 s')]
table(['指标','历史 Direct002','SpeedUp v1 bowl002'],rows,widths=(270,305,305),row_h=42,size=15)
boxtext('观察：v1 的模型调用更少，但机器人动作更多；减少模型等待不等于减少所有运动或接触。',40,411,880,17,'teal',True,max_h=50)
boxtext('不能把这两条记录的比值当作受控加速倍数或普遍收益。仍需同版、同输入条件的多初始状态对照。',40,466,880,15,'muted',max_h=32)
end()

# 23 conclusion and scope
start('方法价值与边界：分工明确，几何安全仍待补齐','22 / CONCLUSIONS',
      '把“语义选择”与“动作反馈”分层，是 v1 的主要结构变化。',refs='综合 [S1–S10]。此前试验已用于修订提示与局部技能，zero-shot 仅表示本次无示范输入。')
card(40,134,278,249,'已展示的能力','当前双 RGB → 少量技能决策 → 多步本地执行。\n\n两条记录分别以 11 / 21 次模型请求达到原始任务目标。',size=16)
card(340,134,278,249,'当前能力边界','距离场只指向 waypoint，没有环境 SDF。\n\n接触代理、抓持验证与停止规则减少盲目执行，但两次仍记录非允许接触。',accent='orange',size=16)
card(640,134,280,249,'下一步如何验证','冻结同一版本与输入条件。\n\n分别消融技能、本地伺服、批量 waypoint；跨初始状态测成功、碰撞、请求数与净耗时。',accent='blue',size=16)
boxtext('1 mm：逻辑网格间距   /   3 mm：视觉伺服阈值尺度   /   5 mm：实测到达容差',40,416,880,18,'teal',True,max_h=48)
boxtext('它们是三个不同的工程参数，都不能被解释成“场景重建达到毫米级精度”。',40,469,880,16,'muted',max_h=30)
end()

# 24 refs
start('证据与复现入口','APPENDIX / SOURCE INDEX',
      '技术细节优先使用 drawer003 的 frozen_source；bowl002 差异按各自日志单列。',refs='完整源码 / 请求 / 视频哈希：evidence_*.json 与 deck_manifest.json；仅新增本幻灯片目录，无新实验。')
refs=[
 ('S1','模型输入、契约与调度','agent_speedup_v1.py / agent_speedup_v1_client.py','drawer003/trial/requests/*.request.json'),
 ('S2','技能、接触、抓持与执行','contact_skills.py / agent_speedup_v1.py',''),
 ('S3','场与局部规划','fine_distance_planner.py / planner.py',''),
 ('S4','路径跟踪','tracking.py / contact_skills.py',''),
 ('S5','视觉指点与伺服','mark_prompting.py / wrist_servo.py',''),
 ('S6','在线标定与图像方向','external_axes.py / contact_skills.py',''),
 ('S7','碗任务结果','runs/agent_speedup_v1_bowl_zero_shot_002/trial/','summary.json / agent_stats.json / trace.jsonl / episode.mp4'),
 ('S8','抽屉任务结果','runs/agent_speedup_v1_drawer_zero_shot_003/trial/','summary.json / agent_stats.json / trace.jsonl / episode.mp4'),
 ('S9','已有场诊断回放','/workspace/agent_speedup_v1_drawer003_field_replay/run_002/','fields/field_007.* / runtime_plans.jsonl / field_actions.jsonl'),
 ('S10','历史 Direct 参照','runs/direct_medium_libero_success_002/','summary.json / manifest.json')]
for i,(key,title,a,b) in enumerate(refs):
    x=40 if i<5 else 500;y=128+(i if i<5 else i-5)*67
    txt(f'[{key}] {title}',x,y,14,'teal',True)
    for j,t in enumerate([a,b]):
        if t:txt(t,x,y+25+j*13,min(10,416/max(1,pdfmetrics.stringWidth(t,'CN',1))),'muted')
boxtext('S1–S6 源码目录：runs/agent_speedup_v1_drawer_zero_shot_003/trial/frozen_source/\n项目根目录：/workspace/gpt_field；设计文档 agent_SpeedUp_v1.md 仅作背景，实测以冻结记录为准。',40,470,880,11,'muted',max_h=30,leading=16)
end()

assert page_no==page_count
c.save()
# Recursively collect exact hashes from evidence, including archived model requests.
sources={}
def collect(v):
    if isinstance(v,dict):
        if isinstance(v.get('path'),str) and isinstance(v.get('sha256'),str):sources[v['path']]=v['sha256']
        for x in v.values():collect(x)
    elif isinstance(v,list):
        for x in v:collect(x)
for e in [R,M,F,json.loads((ROOT/'assets_provenance_framework.json').read_text()),json.loads((ASSETS/'field_figure_manifest.json').read_text())]:collect(e)
unchanged={p:Path(p).exists() and hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in sources.items()}
assert all(unchanged.values()),[p for p,v in unchanged.items() if not v]
manifest={'pdf':str(out),'page_count':page_count,'page_size_pt':[W,H],'language':'zh-CN',
 'scope':'Frozen SpeedUpv1 drawer003 method + bowl002 and drawer003 single-run results; no new inference or simulator run.',
 'source_sha256':sources,'all_sources_unchanged':True,'pdf_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),
 'critical_distinctions':['drawer003 prompt5 vs bowl002 prompt2','Local waypoint goal distance, not scene SDF',
 'Runtime analytic goal path; generic Astar not exercised','Raw LIBERO success vs collision-free success',
 'Final place_release interrupted by success in both runs','Historical Direct reference is unmatched; no causal speedup claim',
 'Video duration, simulation time, total wall and video-excluded control time differ']}
(ROOT/'deck_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(ROOT/'layout_text.json').write_text(json.dumps(text_records,ensure_ascii=False,indent=2)+'\n')
print(out);print('pages',page_no,'bytes',out.stat().st_size,'verified sources',len(sources))
