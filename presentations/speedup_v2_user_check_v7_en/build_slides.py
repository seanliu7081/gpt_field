"""Build English v7 slides from saved source and real-task evidence; no runtime calls."""
from pathlib import Path
import sys, json, math
from xml.sax.saxutils import escape
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'scene_test/astra_field_slides/vendor'))
from slide_layout import Deck, NAVY, TEAL, BLUE, GREEN, RED, AMBER, MUTED, LIGHT, WHITE, INK
A = HERE / 'assets'
E = json.loads((HERE / 'evidence.json').read_text())
OUT = HERE / 'speedup_v2_user_check_v7_framework_and_scene_tests_en.pdf'
d = Deck(OUT)

def foot(s): d.footer(s)
def note(y,s,color=MUTED): d.p(38,y,884,s,13,color,max_h=55)
def code(x,y,lines,size=12,color=WHITE,leading=18):
    for i,l in enumerate(lines): d.text(x,y+i*leading,l,size,color,'Mono')
def tag(x,y,w,t,color=TEAL):
    d.rect(x,y,w,25,color,r=4);d.text(x+10,y+6,t,10,WHITE,'Bold')
def scenes(names,captions,top=150):
    for i,(name,caption) in enumerate(zip(names,captions)):
        x=38+i*301
        d.image(A/name,x,top,282,282)
        d.p(x,top+292,282,caption,12,MUTED,max_h=54)
def bar(x,y,w,label,value,maxvalue,color=TEAL,fmt=None):
    d.text(x,y,label,13,INK)
    d.rect(x,y+25,w,21,LIGHT,r=3)
    d.rect(x,y+25,w*value/maxvalue,21,color,r=3)
    d.text(x+w+12,y+25,fmt or str(value),14,color,'Bold')

# 01
DARK_TEXT='#BDCCD9'
d.begin('SpeedUp v2 User Check','VERSION 7  /  FRAMEWORK + SCENE TESTS',dark=True)
d.p(38,143,480,'Visual decisions.<br/>Measured motion fragments.<br/>One persistent task owner.',29,WHITE,font='Bold',max_h=150)
d.p(39,314,480,'A model-led robot executive that delegates repeated control while retaining the ability to recover and revise its plan.',18,DARK_TEXT,max_h=114)
d.p(39,444,480,'English edition · October 21, 2026<br/>gpt-6-astra / high · LIBERO-10 t3 and t6',12,DARK_TEXT,max_h=45)
d.image(A/'t3_initial.png',558,151,177,177)
d.image(A/'t6_initial.png',746,151,177,177)
d.p(560,344,362,'Recorded initial scenes<br/><b>t3:</b> bowl into bottom drawer, then close.<br/><b>t6:</b> mug onto plate; pudding to its right.',15,DARK_TEXT,max_h=119)
foot('Implementation: agent-speedup-v2-user-check-7. Scene images and metrics come from preserved real-task runs.')

# 02
d.begin('What the current evidence shows','RESULTS AT A GLANCE','One matched initialization per task; the prior reference is v6 User Check, not a new Astranomy benchmark.')
d.metric(38,151,272,'2.49×','t3 observed speed at raw goal completion')
d.metric(344,151,272,'99%+','v7 actions executed by generic fragments')
d.metric(650,151,272,'Incomplete','t6 stopped by user; no completion ratio',AMBER)
d.card(38,280,424,178,'t3: faster, with quality caveats','8:29 versus 21:10; 22 versus 87 model decisions.<br/>Raw LIBERO success in both runs.<br/>Strict physical completion fails in both.',TEAL,16)
d.card(490,280,432,178,'t6: execution works; goal still missed','Mug placed; pudding grasped and repositioned.<br/>Stopped at 12:35 and 28 decisions.<br/>Final pudding center lies outside target region.',AMBER,16)
note(470,'These runs demonstrate controller use and a t3 timing improvement, not broad task generalization.',MUTED)
foot('[R1–R4] Saved summaries, agent_stats and physical_truth. Times exclude video rendering and initial plan preparation.')

# 03
d.begin('The model remains responsible for the task','DESIGN PRINCIPLE','Use fast execution where the target is clear; return control when the execution evidence changes.')
d.card(38,154,274,248,'Model responsibility','Interpret the task and scene.<br/>Choose grasp, route and object goal.<br/>Inspect outcomes.<br/>Change method when needed.',BLUE,17)
d.card(343,154,274,248,'Executor responsibility','Track numerical targets from measured state.<br/>Run explicit gripper stages.<br/>Detect local completion or lack of progress.<br/>Return measured evidence.',TEAL,17)
d.card(648,154,274,248,'Independent evaluation','Record benchmark success.<br/>Audit release, support and stability separately.<br/>Record contacts.<br/>Do not feed hidden truth to the policy.',AMBER,17)
d.p(39,425,883,'<b>Connection to Astranomy:</b> retain model-directed numerical control and visual correction; v7 adds longer measured execution sequences between model decisions.',18,INK,max_h=66)
foot('[M1–M4] agent.py, agent_client.py, fragments.py, sim.py. Existing Astranomy and original v2 implementations are unchanged.')

# 04
d.begin('The full control loop','ARCHITECTURE','Visual reasoning between fragments; proprioceptive feedback control within fragments.')
for x,w,t,b,c in [(38,195,'Initial review','Task + draft JSON<br/>Object / goal selectors<br/>One-time approval',BLUE),(260,218,'Persistent executive','Current RGB + robot state<br/>Reviewed goals / constraints<br/>High-effort model decision',BLUE),(507,193,'Execution branch','execute: fragments<br/>motion: short target<br/>macro: optional native skill',TEAL),(727,195,'Measured handoff','Pose / gap / action result<br/>Completed / pending stages<br/>Fresh visual observation',TEAL)]:
    d.card(x,155,w,214,t,b,c,15)
for x1,x2 in [(236,253),(481,500),(703,720)]:d.arrow(x1,263,x2,263)
d.line(824,395,369,395,TEAL,2);d.arrow(369,395,369,375)
d.p(440,410,475,'Arrival, checkpoint, stall or exhausted budget returns control to the same executive.',16,INK,max_h=62)
d.rect(38,408,340,73,LIGHT)
d.p(52,422,312,'Independent benchmark and physical audit observe execution outside this control loop.',13,AMBER,max_h=54)
foot('[M1,M2,M3] A finish response is a model claim plus fresh observation; it cannot set the benchmark success flag.')

# 05
d.begin('The English review interface','BEFORE EACH TEST','Selection-based editing generates the JSON; the user does not need to type a program.')
d.rect(38,153,526,315,LIGHT)
tag(54,168,154,'PRE-TEST PLAN REVIEW')
ui=[('Action  ·  do','Place on a surface'),('Object  ·  obj','C · selected object'),('Grasp part  ·  grip','D · selected part'),('Reference  ·  ref','B · selected support'),('Direction  ·  sense','Executor chooses'),('Constraint  ·  keep','Maintain orientation')]
for i,(k,v) in enumerate(ui):
    y=211+i*35;d.text(56,y,k,12,MUTED);d.rect(263,y-5,281,29,WHITE,r=5);d.text(274,y+2,v,12,INK);d.text(522,y+2,'▾',12,TEAL)
d.p(588,158,329,'<b>English throughout</b><br/>Headings, choices, buttons, status messages, errors and accessibility labels.',18,INK,max_h=131)
d.p(588,297,329,'<b>Review scope</b><br/>Objects, task steps, preserved relations, spatial regions and placement goals.',17,INK,max_h=117)
d.p(588,415,329,'Recent v7 tests used explicitly authorized automatic approval; no manual review occurred.',12,AMBER,max_h=65)
foot('[UI] agent_speedup_v2_user_check.html. Left: schematic of the actual selection fields, not a browser screenshot.')

# 06
d.begin('Three execution paths, different responsibilities','ACTION INTERFACE','The model selects one action type per decision. Optional payloads not used by that type remain null.')
d.table(38,151,[143,335,410],['Path','What it executes','What this does not imply'],[
 ['<b>execute</b>','1–6 generic segments; measured completion; up to 240 control steps per sequence.','No object recognition, grasp certificate or collision-free path guarantee.'],
 ['<b>motion</b>','A numerical pose/gripper target for 1–5 control steps.','Target may not be reached within that short grant.'],
 ['<b>macro</b>','Optional legacy hold, turn, place_on or release, with its own prerequisites.','A semantic plan intent is not proof that a native macro implements it.'],
 ['<b>observe / finish</b>','Fresh observation, or a completion claim followed by observation.','Neither action sets official success.'],
],row_h=71,font=13.5)
note(478,'v7’s measured acceleration in these tests came from execute fragments; native macro actions were zero.',TEAL)
foot('[M1,M2,M3,M5] Legacy macro grants are bounded and may require calibration / grounding; they are optional.')

# 07
d.begin('One decision can close, test-lift and observe','REAL EXECUTION EXAMPLE','t6_009, decision 3: the model supplies the sequence; the same generic executor handles both stages.')
d.rect(38,151,521,328,NAVY)
code(54,168,[
 '"execution": {', '  "segments": [',
 '    {"skill": "set_gripper",',
 '     "xyz_m": null, "quaternion_wxyz": null,',
 '     "gripper": "close", "profile": "contact",',
 '     "max_steps": 40, "observation_after": false},',
 '    {"skill": "move_relative",',
 '     "xyz_m": [0, 0, 0.03],',
 '     "quaternion_wxyz": null, "gripper": "keep",',
 '     "profile": "contact", "max_steps": 55,',
 '     "observation_after": true}',
 '  ]', '}'],size=11,leading=22)
d.p(591,155,323,'<b>Explicit stages</b><br/>Close until local measured conditions pass, then lift from the actual segment-entry pose.',17,INK,max_h=133)
d.p(591,302,323,'<b>Visual confirmation belongs to the model</b><br/>After the lift, inspect whether the mug moved with the gripper. Stable gap alone is insufficient.',17,INK,max_h=145)
d.p(591,450,323,'Shown: execution payload excerpt.',11,MUTED,max_h=24)
foot('[R4 request 3; M3] The 3 cm lift is a model-selected value from this run, not a task-specific constant in the executor.')

# 08
d.begin('Measured completion and explicit return points','LOCAL CONTROL','The controller converts pose error into bounded OSC commands at 20 Hz.')
d.table(38,149,[287,285,316],['Design bound','Transit profile','Contact profile'],[
 ['Target translation from entry','Up to 30 cm','Up to 8 cm'],
 ['Per-action translation cap','2 cm','5 mm'],
 ['Position arrival tolerance','4 mm','2 mm'],
 ['Target rotation / action cap','1.2 rad / 0.1 rad','0.5 rad / 0.1 rad'],
],row_h=43,font=14)
d.card(38,377,424,111,'Advance on measured completion','Relative goals use actual entry pose; gripper open/close is a separate stage.',TEAL,13)
d.card(490,377,432,111,'Return on uncertainty or boundary','Stall, gap change, envelope rejection, budget or observation checkpoint returns control.',AMBER,13)
foot('[M3] Max 120 steps/segment, 240/sequence. Contact is a slower profile—not contact sensing or a safety certificate.')

# 09
d.begin('What is reusable—and what still requires judgment','GENERALITY AND INFORMATION BOUNDARY','Task-independent code does not establish that all tasks or spatial meanings are supported correctly.')
d.card(38,151,424,303,'Reusable implementation','Five fragments: move_to_pose, move_relative, approach, set_gripper, wait.<br/><br/>Measured state and explicit handoffs.<br/>No task IDs, object names or predefined task coordinates in the fragment executor.<br/><br/>Model-selected route, grasp and checkpoints.',TEAL,16)
d.card(490,151,432,303,'Limits of the current interface','Input: RGB, robot pose/gap/joints, camera and tool geometry, action feedback.<br/><br/>No hidden object poses, depth, contact identities or goal predicates.<br/><br/>Fragment constraints rely on model interpretation; there is no whole-arm collision planner.',AMBER,16)
note(469,'Local feedback is proprioceptive. Fresh visual reasoning happens at model decision boundaries.',BLUE)
foot('[M1–M4] Six recent image pairs plus continuous textual execution history. Geometry calibration does not provide object depth.')

# 10
d.begin('Real-task evaluation protocol','EXPERIMENT SETUP','All tests use existing LIBERO scenes; no synthetic trajectory or replay is substituted for a task result.')
d.table(38,150,[240,648],['Setting','Recorded value'],[
 ['Model / effort','gpt-6-astra / high; subscription route; zero API-key calls'],
 ['Matched conditions','LIBERO-10 t3 or t6; episode 0; seed 7; matching initial observation hashes within each task'],
 ['Reference','Prior v6 User Check: t3_002 and t6_007; not a fresh pure-Astranomy comparison'],
 ['Current trials','v7 t3_003; t6_009 after user-requested restart of the stalled t6_008 preparation'],
 ['Approval','Prior runs manually reviewed; v7 initial plans automatically approved with user authorization'],
 ['Outcome / timing','Raw LIBERO goal; separate strict physical audit; contacts recorded. Control time excludes video and preparation.'],
],row_h=50,font=13)
foot('[R1–R5] Frozen run/review records. Local implementation validation was limited; no repeated smoke rollout was performed.')

# 11
d.begin('t3: bowl into the bottom drawer, then close','RECORDED SCENE  /  t3_003','The agent changes its numerical route and hand configuration while fragments execute the repeated control.')
scenes(['t3_initial.png','t3_mid.png','t3_final.png'],[
 '<b>Step 0</b> · Initial scene.<br/>The bowl starts outside the drawer.',
 '<b>Step 393</b> · Bowl in the cavity.<br/>Model reports a likely slip during transfer.',
 '<b>Step 1202</b> · Near-final push.<br/>Raw success follows at step 1207.'
])
foot('[R2] Unaltered saved external-camera observations. The transfer should not be described as a verified controlled placement.')

# 12
d.begin('t3: fewer calls, faster raw-goal completion','MEASURED RESULT','Net control time falls 59.9%, while the physical action count grows 2.88×.')
d.text(38,150,'Video-excluded control time',17,INK,'Bold')
bar(38,188,284,'v6 / t3_002',1270.233049,1270.233049,BLUE,'21:10')
bar(38,255,284,'v7 / t3_003',509.164051,1270.233049,TEAL,'8:29')
d.text(486,150,'Model decisions',17,INK,'Bold')
bar(486,188,310,'v6 / t3_002',87,87,BLUE,'87')
bar(486,255,310,'v7 / t3_003',22,87,TEAL,'22')
d.card(38,343,424,137,'Official goal succeeds in both','Actions: 419 → 1,207.<br/>Single-run observed time ratio: 2.49×.<br/>No multi-seed reliability claim.',TEAL,14)
d.card(490,343,432,137,'Manipulation quality is unresolved','Strict final-state success: false in both.<br/>Contact points: 671 → 1,488.<br/>Final motion/contact has not settled.',AMBER,14)
foot('[R1,R2] Contact points are accumulated physics samples, not distinct collisions. Counterfactual settle checks are not agent actions.')

# 13
d.begin('t6: mug placement and pudding recovery','RECORDED SCENE  /  t6_009','The mug phase finishes in six model decisions; subsequent work is spent on pudding grasp and placement.')
scenes(['t6_initial.png','t6_mug_placed.png','t6_final.png'],[
 '<b>Step 0</b> · Initial scene.<br/>White mug, plate and pudding.',
 '<b>Step 376</b> · Mug released on plate.<br/>Gripper has withdrawn upward.',
 '<b>Step 1543</b> · Before user stop.<br/>Pudding released; target relation still false.'
])
foot('[R4] Saved external-camera observations. Final physical audit confirms the mug goal; pudding region membership fails.')

# 14
d.begin('t6: useful recovery, incomplete task','EXECUTION EVIDENCE','Stopped by the user at 1,544 steps and 28 model decisions; net elapsed control time 12:35.')
items=[('1','Mug transfer','Approach, close, test-lift, carry, lower, release and withdraw in six decisions.'),('2','Pudding descent blocked','Fingers contact the tabletop. The fragment yields on stall; the model initially suspects a joint limit.'),('3','Grasp recovered','After another blocked descent, close at the reached height and test-lift. The pudding follows.'),('4','Placement revised','Regrasp and reposition; change wrist pitch to improve reach. Final center still misses the goal region.')]
for i,(n,t,b) in enumerate(items):
    y=151+i*81;tag(38,y,29,n,TEAL if i!=3 else AMBER);d.p(82,y,830,t,17,INK,font='Bold',max_h=30);d.p(82,y+29,830,b,14,MUTED,max_h=43)
foot('[R4] 1,208 forbidden contact-point samples, from fingers/table. Prior t6_007 completed raw goal in 21:29 / 84 calls; no speed ratio here.')

# 15
d.begin('t6 exposes a coordinate-frame mismatch','GENERAL FRAMEWORK LIMITATION','Framework right_of = world +x is not the same as the benchmark’s named plate-right region.')
# World XY schematic based on evaluator-only region bounds and terminal COM.
px,py,pw,ph=83,175,385,270
xmin,xmax=.04,.32; ymin,ymax=-.03,.22
xy=lambda x,y:(px+(x-xmin)/(xmax-xmin)*pw,py+ph-(y-ymin)/(ymax-ymin)*ph)
d.rect(px,py,pw,ph,WHITE,stroke='#D8E2EA',r=0)
for x in [.10,.15,.20,.25,.30]:
    sx,_=xy(x,0);d.line(sx,py,sx,py+ph,'#E5EBF0',.6);d.text(sx-13,py+ph+6,f'{x:.2f}',10,MUTED)
for y in [0,.05,.10,.15,.20]:
    _,sy=xy(.1,y);d.line(px,sy,px+pw,sy,'#E5EBF0',.6);d.text(px-35,sy-4,f'{y:.2f}',10,MUTED)
x0,y0=xy(.10,.15);x1,y1=xy(.20,.05)
d.rect(x0,y0,x1-x0,y1-y0,'#DDEFE8',stroke=GREEN,r=0)
d.p(x0+7,y0+32,x1-x0-14,'Official<br/>target region',12,GREEN,max_h=45)
sx,sy=xy(.1874,.1704);d.c.setFillColor(__import__('reportlab.lib.colors',fromlist=['HexColor']).HexColor(RED));d.c.circle(sx,540-sy,5,fill=1,stroke=0)
d.text(sx-75,sy-28,'Final pudding COM',11,RED,'Bold')
d.text(px+pw-115,py+ph+28,'world x (m) →',11,MUTED)
d.text(41,147,'world y (m) ↑',11,MUTED)
d.p(511,154,407,'<b>Semantic issue</b><br/>Shared instructions tie “right” to +x. Initial placements follow that assumption, and later recovery must work around it.',17,INK,max_h=143)
d.p(511,302,407,'<b>Terminal result</b><br/>Pudding COM ≈ (0.1874, 0.1704) m.<br/>x is within the official range.<br/>y is about <b>20.4 mm beyond</b> its upper edge.',17,RED,max_h=139)
d.p(511,448,407,'Directions need an explicit, task-grounded reference frame.',13,TEAL,max_h=40)
foot('[R4; M6; BDDL] Evaluator-only post-run diagram, not policy input. Region x=[0.10,0.20], y=[0.05,0.15] m.')

# 16
d.begin('Where the framework actually did the work','CONTROL ATTRIBUTION','The model still selects targets. The fragment controller supplies almost all per-step motor commands.')
for y,title,frag,direct,total,grants in [(166,'t3_003',1197,10,1207,20),(287,'t6_009',1538,6,1544,26)]:
    d.text(38,y,title,18,INK,'Bold');w=570;x=204
    d.rect(x,y,w,27,LIGHT,r=3);d.rect(x,y,w*frag/total,27,TEAL,r=3);d.rect(x+w*frag/total,y,w*direct/total,27,BLUE,r=0)
    d.text(794,y+2,f'{frag/total*100:.2f}%',17,TEAL,'Bold')
    d.p(204,y+40,704,f'<b>{frag:,}</b> fragment actions / <b>{direct}</b> direct actions / <b>0</b> native macro actions<br/>{grants} execute grants; {total:,} total control steps.',15,MUTED,max_h=61)
d.rect(38,415,884,71,LIGHT)
d.p(54,430,852,'<b>The useful combination:</b> model-level perception, planning and recovery + reusable measured execution. These percentages measure command attribution, not the fraction of intelligence supplied by either side.',15,INK,max_h=52)
foot('[R2,R4 agent_stats] Generic v7 fragments are distinct from old semantic macros. Direct counts can include observation/finish holds.')

# 17
d.begin('What to retain, and what to improve next','DESIGN CONCLUSIONS','These are conclusions from the recorded tests, not new changes or validated future performance.')
d.card(38,151,424,310,'Retain','Model ownership of the full task.<br/><br/>Longer measured fragments where targets are clear.<br/><br/>Explicit gripper stages, checkpoints and cause-preserving feedback.<br/><br/>Separate task outcome, physical quality and execution timing.',TEAL,16)
d.card(490,151,432,310,'Next general improvements','Ground directional relations in an explicit frame instead of a universal axis convention.<br/><br/>Verify whole-object placement from new observations.<br/><br/>Improve contact/cause feedback and route quality.<br/><br/>Evaluate multiple tasks and resets before claiming generality.',AMBER,15.5)
note(475,'The English UI is a language-only update. The recorded v7 controller and evaluator were not changed for these slides.',MUTED)
foot('[M1–M6; R1–R5] t3 is a raw-goal timing gain with quality caveats. t6 remains a stopped, incomplete attempt.')

# 18
d.begin('Evidence, timing and reproducibility','APPENDIX','Saved recordings only. No new simulator episode, model request or smoke test was used to build this deck.')
d.table(38,150,[181,223,232,252],['Run','Outcome','Control / calls','Draft call / review'],[
 ['t3_002 · v6','Raw success; strict false','21:10 / 87 decisions','1 request; manual review'],
 ['t3_003 · v7','Raw success; strict false','8:29 / 22 decisions','21.4 s; auto-approved'],
 ['t6_007 · v6','Raw success; strict false','21:29 / 84 decisions','1 request; manual review'],
 ['t6_009 · v7','User stop; raw/strict false','12:35 / 28 decisions','24.3 s; auto-approved'],
],row_h=48,font=12.5)
d.p(38,390,884,'<b>Preparation interruption:</b> t6_008 was stopped by the user while the first draft request was still pending, before policy actions. It is preserved separately and excluded from control-time ratios.',13,AMBER,max_h=61)
d.p(38,455,884,'Source index, exact metrics, reset hashes and frame provenance: <b>evidence.json</b>.<br/>Method files: agent.py, agent_client.py, fragments.py, agent_macros.py, spatial.py and the user-check runner.',11.5,MUTED,max_h=44)
foot('Full preparation: t3_003 27.1 s; t6_009 32.3 s. Draft-call latency is a subset. Exact timing and source paths: evidence.json.')


# 19
d.begin('Source map','APPENDIX  /  AUDIT TRAIL','All claims are grounded in current source or preserved task artifacts; frame provenance is recorded in evidence.json.')
d.rect(38,151,424,330,LIGHT)
d.rect(490,151,432,330,LIGHT)
d.p(55,166,390,'<b>Method source IDs</b>',18,TEAL,max_h=30)
d.p(55,207,390,'Directory: src/gpt_field/<br/>Common prefix: agent_speedup_v2_user_check',11,MUTED,max_h=44)
for i,line in enumerate([
    'M1  _agent.py — executive and shared state',
    'M2  _agent_client.py — model interface',
    'M3  _fragments.py — measured execution',
    'M4  _sim.py — observations and evaluation bridge',
    'M5  _agent_macros.py — optional native macros',
    'M6  _spatial.py — directional semantics',
    'UI   .html — English plan review interface']):
    d.p(55,267+i*27,390,line,11.5,INK,max_h=24)
d.p(507,166,398,'<b>Recorded run IDs</b>',18,TEAL,max_h=30)
d.p(507,207,398,'Directory: runs/<br/>Prefix: agent_speedup_v2_user_check_libero_10_',10.5,MUTED,max_h=44)
for i,line in enumerate([
    'R1  t3_002 — prior v6 reference',
    'R2  t3_003 — current v7 task result',
    'R3  t6_007 — prior v6 reference',
    'R4  t6_009 — current v7 user-stopped attempt',
    'R5  t6_008_review — interrupted preparation']):
    d.p(507,267+i*27,398,line,11.5,INK,max_h=24)
d.p(507,414,398,'Artifacts: summary.json, agent_stats.json,<br/>physical_truth.json, review.json and requests/.',11,MUTED,max_h=50)
foot('BDDL: LIBERO-10 LIVING_ROOM_SCENE6 white-mug / pudding task, plate_right_region (lines 16–24). See evidence.json for its full path.')

d.save()
(HERE/'slide_manifest.json').write_text(json.dumps({'title':'SpeedUp v2 User Check v7','language':'en','date':'2026-10-21','pages':d.pages,'output':OUT.name,'source_evidence':'evidence.json'},indent=2)+'\n')
(HERE/'layout_bounds.json').write_text(json.dumps(d.bounds,indent=2)+'\n')
print(json.dumps({'pdf':str(OUT),'pages':d.n}))
