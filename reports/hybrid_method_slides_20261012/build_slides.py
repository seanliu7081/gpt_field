"""Build the English research slide PDF from verified source/run evidence."""
import json
import hashlib
from pathlib import Path
from xml.sax.saxutils import escape
from slide_layout import Deck, NAVY, TEAL, BLUE, GREEN, RED, AMBER, MUTED, LIGHT, WHITE, INK

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNS=ROOT/'runs'
FA=HERE/'field_assets'
EVID=HERE/'deck_assets/zero_shot_frames'
OLD=HERE/'deck_assets/one_demo_frames'
NOTES=json.loads((HERE/'results_notes.json').read_text())
ROWS=NOTES['slide_table_rows']
SNAP=json.loads((HERE/'deck_assets/snapshot_201.json').read_text())
OUT=HERE/'hybrid_method_current_en.pdf'
d=Deck(OUT)

def foot(t): d.footer(t)
def fmt(t):
    s=int(round(t));return f'{s//60}:{s%60:02d}'
def tag(x,y,w,label,color=TEAL):
    d.rect(x,y,w,25,color,r=4);d.text(x+9,y+6,label,10,WHITE,'Bold')
def note(y,text,color=MUTED):d.p(38,y,884,text,13,color,max_h=44)

# 01
d.begin('Hybrid robot control with distance fields','GPT_FIELD  /  METHOD + EVIDENCE',dark=True)
d.p(38,142,530,'Use GPT for local task intent.<br/>Use a local planner for repeated motion.',30,WHITE,font='Bold',max_h=130)
d.p(39,278,505,'Compact scene description → graph distance-to-go → measured-pose tracking, with Direct control near contact.',20,'#BDCCD9',max_h=112)
d.p(39,410,505,'Current Dijkstra version · Astra medium · ChatGPT subscription<br/>Source and recorded experiments reviewed on October 12, 2026',12,'#BDCCD9',max_h=52)
d.image(EVID/'hybrid_reset_initial_external.png',640,152,270,270)
d.p(640,435,270,'Real obstacle-scene observation.<br/>No synthetic scene or trajectory is used in the data figures.',11,'#BDCCD9',max_h=50)
foot('Scope: current distance-field implementation; historical comparisons retain their original evaluation criteria.')

# 02
d.begin('The division of work','OVERVIEW','The acceleration hypothesis is fewer model calls during free-space motion.')
boxes=[(38,150,202,'Observe','Task + two RGB views<br/>Measured robot pose<br/>Observable history'),
       (265,150,202,'GPT decision','Route + local goal<br/>Predicted obstacles<br/>Orientation + gripper'),
       (492,150,202,'Local planning','Build valid grid<br/>Solve distance-to-go<br/>Extract one path'),
       (719,150,202,'Robot control','Track measured pose<br/>Execute bounded action<br/>Read new observation')]
for x,y,w,t,b in boxes:d.card(x,y,w,210,t,b,size=16)
for x in [243,470,697]:d.arrow(x,248,x+17,248)
d.line(820,380,140,380,TEAL,2);d.arrow(140,380,140,365)
d.p(320,390,580,'Local feedback continues every action. A model refresh occurs at a controller boundary or an observable event.',17,INK,max_h=80)
foot('[M1–M4] distance_subscription.py, distance_policy.py, distance_planner.py, tracking.py; current code, not a learned rollout model.')

# 03
d.begin('What the model sees—and what it predicts','POLICY INTERFACE','Zero-shot means no demonstration episode; common hand-written task guidance is still present.')
d.card(38,147,424,300,'Input','Task instruction<br/>External camera + wrist RGB<br/>8D robot state: end-effector xyz, axis-angle, two finger joints<br/>Recent live images + summarized action feedback<br/><br/>Optional one-demo prefix only in marked experiments.',TEAL,16)
d.card(490,147,432,300,'Output','<b>Field branch:</b> compact geometry/cost parameters, pose/gripper intent and control context.<br/><br/><b>Direct branch:</b> five normalized 7D actions.<br/><br/>No ground-truth object pose, depth, contact or success signal enters the model.',BLUE,16)
foot('[M1,M5] distance_subscription.py, zero_shot_direct_subscription.py, observation.py; latest two live image pairs retained.')

# 04
d.begin('A real compact Field output','FIELD GENERATION  1 / 3','Snapshot A · zero-shot obstacle Hybrid · decision before action index 201 (indices are zero-based).')
d.rect(38,149,508,302,NAVY)
code=[('goal_offset_m', '[-0.070, 0.000, +0.045]'),('attraction_weights','[1, 1, 1]'),('repulsors','[]'),('blocked lower','[+0.015, -0.080, -0.080]'),('blocked upper','[+0.080, +0.080, -0.015]'),('unknown lower','[-0.080, -0.080, -0.080]'),('unknown upper','[+0.015, +0.080, -0.050]'),('gripper / phase','open / recover')]
for i,(k,v) in enumerate(code):
    d.text(54,171+i*31,k,12,'#78DAD1','Mono');d.text(225,171+i*31,v,11.7,WHITE,'Mono')
d.p(576,158,332,'<b>Small structured output</b><br/>Goal, up to four Gaussian repulsors, up to four blocked boxes and four unknown boxes.',18,INK,max_h=133)
d.p(576,299,332,'All offsets are in meters along world axes, relative to the end-effector position when this Field was generated.',17,MUTED,max_h=106)
d.p(576,410,332,'GPT does not emit 35,937 scalar distances.',16,TEAL,font='Bold',max_h=55)
foot('[S1] trace.jsonl action 201 / effective_compact_field; raw snapshot saved in deck_assets/snapshot_201.json.')

# 05
d.begin('Expand locally into a bounded planning domain','FIELD GENERATION  2 / 3','The grid is centered once per Field refresh. Reuse does not expand known space.')
d.metric(38,149,272,'33 × 33 × 33','35,937 grid vertices')
d.metric(344,149,272,'5 mm','Spacing along each grid axis')
d.metric(650,149,272,'16 cm','Span per axis: ±8 cm about the center')
d.rect(38,278,550,179,LIGHT)
d.p(56,295,510,'<b>Coordinates:</b> x<sub>ijk</sub> = c + 0.005 · (i−16, j−16, k−16)<br/><br/><b>Validity:</b> inside workspace AND outside predicted blocked / unknown regions.<br/>Check graph edges and tracking chords against masks and closed boxes.',17,INK,max_h=149)
d.p(626,283,286,'<b>Unknown is not free.</b><br/><br/>Repulsors add soft cost; they do not create collision geometry.<br/><br/>5 mm resolution is a sampling choice, not 5 mm perception accuracy.',17,TEAL,max_h=193)
foot('[M2,M3,M6] compact_field.py:118; distance_planner.py:67,128; schema.py GridSpec. Grid/mask geometry is model-predicted.')

# 06
d.begin('Definition: graph distance-to-go','FIELD GENERATION  3 / 3','Distance is to a local goal region through predicted valid space—not distance to an obstacle surface.')
d.rect(38,146,590,322,LIGHT)
eqs=[('Traversal density','ρ(x) = 1 + Σ<sub>k</sub> β<sub>k</sub> exp(−‖x−r<sub>k</sub>‖² / (2σ<sub>k</sub>²))'),
      ('Cost of a valid edge','w(u,v) = ‖x<sub>v</sub>−x<sub>u</sub>‖ · [ρ(x<sub>u</sub>)+ρ(x<sub>v</sub>)] / 2'),
      ('Distance-to-go','D(v) = min<sub>P: v→G</sub> Σ<sub>(u,z)∈P</sub> w(u,z)')]
for i,(label,eq) in enumerate(eqs):
    y=163+i*93;d.text(56,y,label.upper(),10,TEAL,'Bold');d.p(56,y+25,551,eq,21,INK,max_h=51)
d.p(658,154,254,'<b>26-neighbor graph</b><br/>Reverse multi-source Dijkstra stores D and a successor toward the goal.<br/><br/><b>Units</b><br/>Meters with no repulsors; weighted meters with repulsors.<br/><br/>Invalid / unreachable: +∞.<br/>Goal endpoints: D = 0.',16,INK,max_h=302)
foot('[M3] G: valid grid/virtual goal-region endpoints; terminal-edge costs included. Enclosing corners can have D > 0. Tolerances: slide 9.')

# 07
d.begin('Three fields that should not be conflated','DEFINITION','The current path objective is D. The compatibility potential U is still stored but does not drive Dijkstra.')
d.table(38,151,[190,288,410],['Quantity','Meaning','Role / limitation'],[
 ['<b>Legacy potential U</b>','Weighted goal attraction + Gaussian bumps; dimensionless.','Expanded into GridField for compatibility. Current Dijkstra ignores U and attraction_weights.'],
 ['<b>Current distance D</b>','Minimum weighted route length to the local goal region.','Computed locally on the valid graph. Exact for that finite augmented graph; not exact continuous geodesic distance.'],
 ['<b>Conventional SDF</b>','Signed distance to the nearest obstacle surface.','Different quantity. This implementation does not reconstruct a surface-distance field from sensor geometry.'],
 ],row_h=91,font=15,colors={(0,0):MUTED,(1,0):TEAL,(2,0):BLUE})
note(470,'No claim of a full-scene metric, a Riemannian neural field, or a certified robot configuration-space planner.',AMBER)
foot('[M2,M3] compact_field.py:118; distance_planner.py:28,228. Definitions distinguish representation, objective and geometry.')

# 08
d.begin('Real data: predicted masks and solved distance','SNAPSHOT A','A slice of the full 3D solve at the recorded start Y; the selected path lies in this XZ plane.')
d.image(FA/'field_cost_slice_wide.png',30,144,900,296)
d.p(40,446,884,'Snapshot A has no Gaussian repulsors: ρ = 1. The plot shows the model’s local masks and graph distances, not recovered physical obstacle geometry.',13,MUTED,max_h=43)
foot('[S1] zero-shot Hybrid action 201. Frozen-code reconstruction; axes/units and exact numerical arrays are in field_assets/.')

# 09
d.begin('Planning returns one route, not a trajectory bank','PATH EXTRACTION','35,937 possible grid vertices are not 35,937 candidate trajectories.')
items=[('1','Connect the measured start','Validate connections to enclosing grid corners; choose start-edge cost + D.'),
       ('2','Follow stored successors','Extract a shortest graph route. Reuse cached D while the field and goal are unchanged.'),
       ('3','Keep a short executable prefix','Merge only collinear edges. At most 10 segments, each ≤1 cm; at most 10 cm of geometric path.')]
for i,(n,t,b) in enumerate(items):
    y=151+i*94;tag(40,y,31,n);d.p(88,y,814,t,19,INK,font='Bold',max_h=28);d.p(88,y+34,814,b,15,MUTED,max_h=49)
d.rect(39,438,881,52,'#E2F2EF')
d.p(54,447,850,'<b>Current tolerances:</b> 2.5 mm goal-region radius + 2.5 mm tracking endpoint tolerance; outer arrival check = 5 mm and 0.05 rad. Invalid/out-of-map goals request a new observation.',12.5,INK,max_h=40)
foot('[M3,M4] distance_planner.py:262–337; distance_policy.py:28–35,104,423. Success: one CandidateRecord; hold/failure: zero.')

# 10
d.begin('Convert the geometric path into robot actions','EXECUTION','Geometric segments, control ticks and model calls are different quantities.')
d.card(38,150,423,276,'Measured-pose path tracking','Project actual end-effector position onto the remaining path.<br/><br/>Choose a checked point up to <b>3 cm ahead</b>.<br/><br/>Command position error, bounded to <b>5 cm norm</b>. Shorten the reference if its chord crosses a predicted mask.',TEAL,16)
d.rect(489,150,432,276,LIGHT)
d.p(507,169,396,'δx = bounded(x<sub>reference</sub> − x<sub>measured</sub>)',20,INK,max_h=65)
d.p(507,245,396,'a = [δx / 0.05, δθ / 0.5, gripper]',18,BLUE,max_h=56)
d.p(507,310,396,'Rotation error is capped at 0.15 rad.<br/>Gripper: open −1, close +1, hold previous.<br/>Actions go to the 20 Hz OSC controller.',15,MUTED,max_h=99)
note(445,'Track for at most 10 control ticks per local plan, or replan earlier. A “10-step path” does not guarantee 10 cm of actual robot travel.',TEAL)
foot('[M4] tracking.py:63–145; action.py:40–96; distance_policy.py:104. Commands are bounded setpoint increments, not measured displacements.')

# 11
d.begin('How stages and controller switches work','CLOSED LOOP','GPT selects phase and branch. Typical uses are shown below; phase names do not impose a fixed controller mapping.')
d.card(38,152,420,245,'Field branch','Approach / transport / alignment.<br/>Local path tracking; initially up to <b>30 actions</b> per model response.<br/><br/>Reliable visual progress may extend reuse to 60, 90, then 120 actions.',TEAL,17)
d.card(500,152,422,245,'Direct branch','Grasp / release / contact recovery.<br/><b>Five model-generated actions</b> per response, executed unchanged.<br/><br/>Task termination can interrupt the final chunk.',BLUE,17)
d.arrow(461,268,496,268);d.arrow(496,291,461,291)
d.p(40,423,882,'<b>Refresh early:</b> measured arrival, invalid path/reference, boundary, low realized motion, lost tracking, or placement verification concerns. A phase label or “near_target” prediction is not proof of contact or success.',15,MUTED,max_h=69)
foot('[M4,M5] distance_policy.py:218,340,426; hybrid_policy.py:149,193. Cadence is event-driven; 30 × 1-step replans is not the current loop.')

# 12
d.begin('Ground the goal and correct it when evidence allows','LOCAL FEEDBACK','These mechanisms are conditional. Their existence in code does not mean they activated in a rollout.')
d.card(38,147,424,233,'Account for the held bowl','g = c + d − R<sub>target</sub> R<sub>current</sub><super>T</super> r<br/><br/>c: Field center; d: desired bowl offset.<br/>r: current bowl offset from the gripper.<br/><br/>Subtract the rotated grasp offset from the bowl goal. Assumes a visually estimated, rigid grasp.',TEAL,15)
d.card(490,147,432,233,'Optional image-based XY correction','Track model-specified image points.<br/>Fit a local 2×2 image Jacobian from measured XY motion.<br/>Apply a validated correction ≤1 cm.<br/><br/>No depth reconstruction; masks and original grid center stay fixed.',BLUE,16)
d.rect(38,402,884,83,'#FFF0E1')
d.p(55,417,850,'<b>Latest zero-shot run:</b> all 13 Field outputs had no visual anchor; all 310 feedback records reported anchor unavailable. No local visual goal corrections or reuse extensions occurred.',16,AMBER,max_h=64)
foot('[M4,M7,S2] field_grounding.py:91; local_visual_feedback.py:294; distance_policy.py:297. Visual agreement never certifies task success.')

# 13
d.begin('Real data: a plan is not the measured motion','SNAPSHOT A EXECUTION','Recorded actions 201–214, including local replanning; no simulated trajectory was substituted.')
d.image(FA/'field_tracking_wide.png',32,144,896,304)
d.p(39,454,884,'A 10-segment geometric prefix can be shorter than 10 cm. Tracking uses feedback to reduce error; commanded displacement is not equal to realized displacement.',13,MUTED,max_h=41)
foot('[S1] planned paths and measured end-effector states, trace.jsonl. Reconstruction metadata and CSV/NPZ are saved with the figure.')

# 14
d.begin('Evaluation protocol and the timing metric','MEASURED RESULTS','Do not mix different success criteria or count failed-attempt durations as completed-task speedups.')
d.table(38,146,[230,658],['Common condition','Setting'],[
 ['Model / channel','gpt-6-astra · medium · ChatGPT subscription OAuth · zero API-key calls'],
 ['Within each paired condition','Same task, initial state/scene, demo if present, shared guidance and observable feedback'],
 ['Timing','T = recorded control_elapsed_s − video_render_s'],
 ['Earlier completed pairs','Original LIBERO goal only; release/stability and collision freedom were not separately certified'],
 ['Latest zero-shot obstacles','LIBERO goal AND no forbidden contact; immediate failure at first forbidden physics contact'],
 ],row_h=54,font=14)
note(462,'One trial per arm and condition. Video subtraction is retrospective accounting, not a recording-disabled rerun.',AMBER)
foot('[R1–R5] frozen summaries/manifests/audits. Timings include model waiting, local control, simulator/observation and loop overhead.')

# 15
d.begin('Earlier matched pairs reached the LIBERO goal','GOAL-ONLY RESULTS','One demonstration per method. These runs did not use the later collision-free success criterion.')
short=['Drawer + bowl','Bowl: original layout','Bowl: relocated objects']
table=[]
for label,r in zip(short,ROWS[:3]):
    table.append([label,fmt(r['hybrid_s_excluding_video']),fmt(r['direct_s_excluding_video']),f"{r['hybrid_percent_less_time']:.1f}%",f"{r['hybrid_requests']} / {r['direct_requests']}"])
d.table(38,149,[272,145,145,170,156],['Task / condition','Hybrid','Direct','Less time','Calls H / D'],table,row_h=65,font=17,colors={(i,3):TEAL for i in range(3)})
d.p(42,397,877,'Times are min:sec after removing measured video overhead. Hybrid used fewer model requests in all three pairs; the gain ranged from 1.9% to 25.6%.',17,INK,max_h=56)
d.p(42,458,877,'Single trials—not a reliability estimate. The 1.9% result is small; raw recorded control time actually favored Direct before video subtraction.',12.5,AMBER,max_h=35)
foot('[R1–R3] results_notes.json, paired_conditions. Both arms stopped on the original goal predicate; post-release stability not verified.')

# 16
d.begin('A completed bowl pair illustrates the mechanism','BOWL / ORIGINAL LAYOUT / ONE DEMO','Less model interaction, even with more executed actions. This is one observed pair.')
r=NOTES['paired_conditions'][1]
for j,arm in enumerate(['hybrid','direct']):
    a=r[arm];x=38+j*456;color=TEAL if arm=='hybrid' else BLUE
    d.rect(x,149,428,227,LIGHT);d.text(x+18,168,arm.upper(),13,color,'Bold')
    d.text(x+18,203,f"{a['control_excluding_video_s']:.1f} s",37,color,'Bold')
    d.p(x+18,263,390,f"<b>{a['requests_started']} requests</b> · {a['executed_actions']} actions<br/>{a['field_actions']} Field actions / {a['direct_actions']} Direct actions<br/>Model/client pipeline: {a['model_pipeline_s']:.1f} s",16,INK,max_h=97)
    barw=385*a['model_pipeline_s']/a['control_excluding_video_s']
    d.rect(x+18,393,385,15,'#DDE6ED',r=2);d.rect(x+18,393,barw,15,color,r=2)
    d.text(x+18,418,f"Model/client pipeline: {100*a['model_pipeline_s']/a['control_excluding_video_s']:.1f}% of adjusted time",12,MUTED)
note(457,'Pipeline time includes client/transport work and local expansion. It is not pure GPU inference latency; timer components must not be added blindly.',MUTED)
foot('[R2] bowl_distance_medium_one_demo_001 and bowl_direct_medium_one_demo_001 / summary.json; 25.6% adjusted time reduction.')

# 17
d.begin('Obstacle tasks expose a different failure mode','ONE-DEMO OBSTACLE RUNS','Both reached the original placement predicate, but both collided with the ketchup bottle: both are failures.')
for x,title,img in [(38,'Hybrid',OLD/'hybrid_after_action_132_external.png'),(342,'Direct',OLD/'direct_after_action_100_external.png')]:
    d.text(x,148,title,18,INK,'Bold');d.image(img,x,177,266,266)
d.p(644,153,275,'<b>Adjusted attempt duration</b><br/>Hybrid: 6:41<br/>Direct: 8:50<br/><br/><b>No valid task speedup.</b><br/>A goal-only success flag did not enforce obstacle preservation.<br/><br/>Review corrected both outcomes to collision failure.',17,RED,max_h=290)
note(457,'Recorded RGB excerpts show the bottle displaced/tilted. Earlier Hybrid: all eight Field responses had empty obstacle/unknown/repulsor lists.',MUTED)
foot('[R4] corrected comparison v2 + collision_evidence/. Images are real recorded frames; these times are not times to first contact.')

# 18
d.begin('Latest zero-shot test: no successful pair','STRICTER OBSTACLE EVALUATION','Same frozen obstacle scene; explicit no-collision guidance; actual physics-substep contact monitoring.')
d.table(38,145,[170,313,135,120,150],['Method','Outcome','Time*','Actions','Requests'],[
 ['Hybrid','Collision failure: left finger–ramekin','9:52','310','44'],
 ['Direct','User-stopped; no goal reached','25:07','505','102†'],
 ],row_h=67,font=16,colors={(0,1):RED,(1,1):AMBER})
d.image(EVID/'hybrid_after_action_0309_terminal_collision_external.png',40,334,148,148)
d.image(EVID/'direct_after_action_0504_terminal_user_stop_external.png',203,334,148,148)
d.text(42,486,'Hybrid / collision stop',9,MUTED)
d.text(205,486,'Direct / user stop',9,MUTED)
d.p(379,341,537,'Neither reached LIBERO success. Direct remained incomplete and unscored, with no forbidden contact before the user stop.<br/><br/><b>Allowed:</b> finger–target bowl grasp; bowl–table/plate support.<br/><b>Excluded:</b> robot self/internal and scene-scene-only contact.<br/>Monitoring detects contact; it does not prevent first contact.',13.5,INK,max_h=141)
d.p(41,299,879,'* Video-excluded attempt duration, not completion time.  † 101 valid responses + one interrupted request.',11.5,MUTED,max_h=31)
foot('[R5,S2] comparison.json, task_evaluation.json and collision.json. Hybrid stopped at zero-based action 309, physics substep 7.')

# 19
d.begin('What the evidence supports—and what is missing','INTERPRETATION','Fewer model requests do not by themselves establish successful or collision-free execution.')
d.card(38,146,425,303,'Supported by these recordings','Fewer model requests in three earlier completed goal-only pairs.<br/><br/>Compact outputs avoid writing a dense 35,937-value response.<br/><br/>Current local planning is a well-defined shortest-path computation over predicted valid space.',TEAL,17)
d.card(490,146,432,303,'Not demonstrated yet','Reliable collision-free manipulation in the harder scene.<br/><br/>Correct target identification from zero-shot RGB.<br/><br/>Whole-arm / payload swept-volume safety, or consistent activation of local visual correction.',RED,17)
note(462,'Priority: verifiable target localization → usable visual anchors → gripper/payload geometry checks → repeated matched trials.',AMBER)
foot('[M1–M7,R1–R5,S2] These are implementation properties and single-run observations, not generalized performance guarantees.')

# 20
d.begin('Appendix: exact potential and distance semantics','TECHNICAL DETAIL','The code retains the legacy potential representation alongside the new distance objective.')
d.rect(38,148,884,129,LIGHT)
d.p(56,165,848,'U(x) = √[Σ<sub>a</sub> α<sub>a</sub>((x<sub>a</sub>−g<sub>a</sub>)/ℓ)²] + Σ<sub>k</sub> β<sub>k</sub> exp(−‖x−r<sub>k</sub>‖² / (2σ<sub>k</sub>²))',21,INK,max_h=75)
d.p(57,241,844,'ℓ = 0.08 m. U is dimensionless. Attraction weights α affect U, but not current Dijkstra edge weights.',13,TEAL,max_h=31)
d.bullet(40,303,876,'Distance is goal-conditioned','D is a cost-to-go to one terminal region. It is not a scalar clearance to nearby obstacle surfaces.',TEAL,17)
d.bullet(40,380,876,'Optimality has a precise scope','Reverse Dijkstra is exact on the constructed finite graph with endpoint-averaged density and valid terminal connections. It is not a continuous Eikonal/PDE solve.',BLUE,17)
foot('[M2,M3] compact_field.py:118–146; distance_planner.py:128–201. A Gaussian cost penalty alone never creates a hard obstacle.')

# 21
d.begin('Appendix: all paired conditions at a glance','RESULTS LEDGER','Same current Dijkstra family; criterion and demonstration count are explicit for every row.')
labels=['Drawer','Bowl / original','Bowl / relocated','Bowl / obstacles','Bowl / obstacles']
outcomes=['Goal / goal','Goal / goal','Goal / goal','Collision / collision','Collision / incomplete']
rows=[]
for i,r in enumerate(ROWS):
    rows.append([labels[i],str(r['demo_count']),outcomes[i],f"{r['hybrid_s_excluding_video']:.1f}",f"{r['direct_s_excluding_video']:.1f}"])
d.table(38,145,[210,90,278,155,155],['Condition','Demos','Outcome H / D','H seconds*','D seconds*'],rows,row_h=52,font=14)
d.p(40,450,880,'* Video-excluded time. First three rows: earlier goal-only completion. Last two rows: failed/incomplete attempts, excluded from completion-speed comparisons. No pooled average is meaningful across these criteria.',12.5,AMBER,max_h=44)
foot('[R1–R5] Exact timing subtraction, raw timings, request/actions and source hashes are included in results_notes.json.')

# 22
d.begin('Appendix: source and reproducibility map','EVIDENCE','All visuals and numbers are derived from local source or recorded experiment artifacts. No new robot test was run for this deck.')
d.table(38,146,[126,762],['Reference','Where to verify'],[
 ['M1–M7','src/gpt_field/: distance_subscription.py, compact_field.py, distance_planner.py, distance_policy.py, tracking.py, action.py, field_grounding.py, local_visual_feedback.py'],
 ['R1–R3','runs/{drawer_one_demo, bowl_one_demo, bowl_relayout_one_demo}_matched_comparison_001/ and paired raw summaries'],
 ['R4','runs/bowl_obstacle_one_demo_matched_comparison_001/comparison.json — corrected collision-aware report'],
 ['R5 / S2','runs/bowl_obstacle_zero_shot_matched_comparison_001/{comparison.json, analysis.json, evidence/}'],
 ['S1','runs/bowl_obstacle_distance_medium_zero_shot_001/trace.jsonl — snapshot 201 and measured execution'],
 ],row_h=53,font=12.5)
d.p(40,460,877,'Delivered with build source, numerical field/trajectory assets, provenance hashes, method_notes.json and results_notes.json.',13,TEAL,max_h=30)
foot('Prepared October 12, 2026. “Audit passed” means record/protocol integrity; it is not a claim that the robot task succeeded.')

d.save()
(HERE/'slide_manifest.json').write_text(json.dumps({'pdf':str(OUT),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'pages':d.pages,'layout_boxes':d.bounds},indent=2))
print(json.dumps({'pdf':str(OUT),'pages':len(d.pages),'bytes':OUT.stat().st_size},indent=2))
