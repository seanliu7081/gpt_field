"""Build the current UC7.1 technical guide using preserved source/run evidence only."""
from pathlib import Path
import json,sys,math
from xml.sax.saxutils import escape
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'scene_test/astra_field_slides/vendor'))
from slide_layout import Deck,NAVY,TEAL,BLUE,GREEN,RED,AMBER,MUTED,LIGHT,WHITE,INK
from reportlab.lib.colors import HexColor
A=HERE/'assets'
OUT=HERE/'speedup_v2_user_check_v71_libero10_technical_framework_en.pdf'
d=Deck(OUT)

def page(title,kicker,sub=None,dark=False):d.begin(title,kicker,sub,dark=dark)
def foot(source):d.footer(source)
def note(y,text,color=MUTED,size=13):return d.p(38,y,884,text,size,color,max_h=499-y)
def codebox(x,y,w,h,lines,size=11.5,leading=18):
    d.rect(x,y,w,h,NAVY)
    for i,line in enumerate(lines):d.text(x+16,y+16+i*leading,line,size,WHITE,'Mono')
def two_cards(left,right,top=155,height=280,size=16):
    d.card(38,top,424,height,*left,size=size)
    d.card(490,top,432,height,*right,size=size)
def three_cards(items,top=155,height=270,size=15):
    for x,item in zip((38,344,650),items):d.card(x,top,272,height,*item,size=size)
def tag(x,y,w,text,color=TEAL):
    d.rect(x,y,w,25,color,r=4);d.text(x+9,y+6,text,10,WHITE,'Bold')
def scene_strip(names,captions,top=153):
    for i,(name,caption) in enumerate(zip(names,captions)):
        x=38+i*301;d.image(A/name,x,top,282,282)
        d.p(x,top+291,282,caption,12,MUTED,max_h=51)
def table(headers,rows,widths=(245,643),top=151,row_h=47,font=13.5):
    return d.table(38,top,list(widths),headers,rows,row_h=row_h,font=font)

# 01
page('SpeedUp v2 User Check 7.1','TECHNICAL FRAMEWORK / LIBERO-10',dark=True)
d.p(38,146,495,'Human-reviewed goals.<br/>Persistent visual decisions.<br/>Measured motion fragments.',28,WHITE,font='Bold',max_h=150)
d.p(39,314,492,'Architecture, data contracts, calibrated direction semantics, execution control and independent evaluation.',18,'#BDCCD9',max_h=105)
d.p(39,440,480,'English technical guide · October 30, 2026<br/>Frozen UC7.1 Version A / A9 · gpt-6-astra / high',12,'#BDCCD9',max_h=48)
d.image(A/'t0_initial.png',565,155,174,174);d.image(A/'t1_initial.png',748,155,174,174)
d.p(565,351,357,'Recorded worked examples<br/><b>t0:</b> soup + tomato sauce into basket.<br/><b>t1:</b> cream cheese + butter into basket.',15,'#BDCCD9',max_h=121)
foot('Release: agent-speedup-v2-user-check-7.1 | Evidence: approved October 30 custom-layout runs; no new rollout for this guide.')

# 02
page('Scope, release and reading map','DOCUMENT CONTRACT','This guide describes the frozen implementation used in the two approved LIBERO-10 attempts.')
table(['Item','Meaning'],[
 ['Release identity','agent-speedup-v2-user-check-7.1; direction_semantics = external_view_v1.'],
 ['Framework chapters','Model/executor boundary, plans and approval, direction semantics, actions, feedback and deployment protocol.'],
 ['Evaluation chapters','Official goal, terminal physical truth, diagnostic settling, timing and recorded contact accounting.'],
 ['Worked examples','One rearranged layout each for t0 and t1; official init 0 / seed 7; actual browser approval.'],
 ['Evidence boundary','These two runs are not a ten-task sweep, multi-seed study, baseline ablation or measured UC7.1 speedup.'],
 ['Source policy','Runtime, prompts and scoring were unchanged during both runs. Old v7 deck remains a separate historical report.'],
 ],row_h=48,font=13)
foot('[P] UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md; [L] frozen release manifest; [R0,R1] approved run artifacts.')

# 03
page('Three responsibilities, one control loop','ARCHITECTURE','The model owns task decisions; the executor owns repeated motor control; evaluation remains separate.')
three_cards([
 ('Task executive','Interpret images and task.<br/>Choose target, grasp and route.<br/>Inspect visual outcomes.<br/>Revise method from feedback.',BLUE),
 ('Measured executor','Track numeric pose targets.<br/>Run explicit gripper stages.<br/>Check arrival and progress.<br/>Return control at a boundary.',TEAL),
 ('Independent evaluator','Check the official goal.<br/>Audit support and stability.<br/>Record physical contacts.<br/>Keep hidden truth out of policy.',AMBER),
 ],height=260,size=16)
d.arrow(314,276,338,276);d.arrow(620,276,644,276)
note(439,'The measured feedback loop is local to the robot. Visual reasoning and task-level recovery happen at model-decision boundaries.',size=16)
foot('[M1] agent policy; [M2] agent client; [M3] fragments; [M6] simulator and independent physical-truth evaluator.')

# 04
page('From preparation to one authorized attempt','LIFECYCLE','A draft is generated before policy actions; approval is consumed once before execution requests.')
for x,w,t,b,c in [(38,192,'Prepare','Validated layout<br/>Frozen reset<br/>One model draft',BLUE),(269,191,'Review','Images + selectors<br/>Check every step<br/>Approve plan',BLUE),(500,191,'Execute','Observe / decide<br/>Measured fragments<br/>Fresh feedback',TEAL),(731,191,'Finalize','Official terminal<br/>Physical diagnosis<br/>Video + audit',AMBER)]:
    d.card(x,159,w,191,t,b,c,14)
for x1,x2 in ((234,263),(463,494),(694,725)):d.arrow(x1,254,x2,254)
d.rect(38,382,884,103,LIGHT)
d.p(55,397,850,'<b>Continuous execution:</b> fresh RGB + measured robot state → model decision → bounded local sequence → measured result → next observation.<br/><b>Authorization boundary:</b> one approved task/layout/configuration, one attempt; no automatic episode restart.',15,INK,max_h=79)
foot('[P §§4–6] preparation, approval and supervision; [M4] ReviewStore; [M5] user-check runner.')

# 05
page('What enters the policy—and what does not','OBSERVATION CONTRACT','Calibration is allowed in UC7.1. It does not supply object depth or hidden task answers.')
two_cards(
 ('Policy-visible input','Two upright RGB views at 512×512.<br/>Measured robot pose / gripper state and seven arm joints.<br/>Camera and fixed tool geometry for execution.<br/>Official task text, reviewed goals, action history and measured feedback.',TEAL),
 ('Evaluator-only information','Simulator object poses and identities.<br/>BDDL numeric goal regions and success predicates.<br/>Physical contact, support and stability truth.<br/>Reset validation and post-terminal diagnosis stay outside model input.',AMBER),height=284,size=15.5)
note(460,'Initial plan/replan requests use images and task context without camera calibration; execution requests carry calibrated geometry.',BLUE,13)
foot('[M2,M6] calibrated observation / executive request path; [M7] plan client; [P §2] allowed input boundary.')

# 06
page('The reviewed plan is a semantic contract','PLAN DATA MODEL','Object references and goals are reviewed; a plan is not a fixed trajectory or a grasp certificate.')
table(['Field','Role'],[
 ['parts','Stable IDs + visual descriptions; x = external-grid cell, w = wrist-grid cell; move = free/fixed/hinged/sliding/unknown.'],
 ['steps','do, obj, ref, grip, sense, keep, until define an ordered task intent and selected constraints.'],
 ['hold_true','Relations that the plan asks to preserve across the task.'],
 ['regions','Support / anchor / relation / frame / bounds_xy_m define spatial targets. Directional bounds may remain null.'],
 ['placement_goals','Object/reference relation and optional tolerance; independent evaluation still owns task success.'],
 ['Grid convention','8×8, 1–64 row-major from top-left; 0 means invisible. A cell identifies a region, not a metric contact point.'],
 ],row_h=48,font=13)
foot('[M4,M7,M8] review schema, selector-edit validation and spatial extension; actual t0/t1 plan copies are preserved.')

# 07
page('Actual t0 plan: two objects, one container','WORKED PLAN / NO MANUAL EDITS','The browser-approved draft has two place_in steps and two inside goals; extra constraints are empty.')
codebox(38,153,492,330,[
 'parts: A=soup, B=tomato, C=basket',
 'cells (external / wrist):',
 '  A: 34 / 48',
 '  B: 36 / 53',
 '  C: 40 / 25',
 '',
 'steps:',
 '  place_in(A, C), sense="any"',
 '  place_in(B, C), sense="any"',
 '',
 'placement_goals:',
 '  inside(A, C), tolerance=null',
 '  inside(B, C), tolerance=null',
 '',
 'hold_true=[], regions=[]',
 'grip=[], keep=[], until=[] per step',
 ],size=11.5,leading=18)
d.image(A/'t0_external_grid.png',565,153,330,278)
d.p(565,443,340,'Saved external planning grid.<br/>No extra pose or motion trajectory was inserted by the reviewer.',12,MUTED,max_h=51)
foot('[R0] planning_review/review.json and review/review.json; plan unchanged, both steps checked; shorthand rendering of exact fields.')

# 08
page('Browser review, edits and approval','USER CHECK INTERFACE','The existing UI edits structured selectors and validates the resulting plan before approval.')
d.rect(38,153,510,327,LIGHT);tag(54,168,181,'PLAN REVIEW / SCHEMATIC')
for i,(k,v) in enumerate([('Object','A · alphabet soup'),('Action','place_in'),('Reference','C · basket'),('Grasp part','object itself / default'),('Movement sense','sense = any'),('Step checked','✓ reviewed')]):
    y=209+i*34;d.text(55,y,k,12,MUTED);d.rect(218,y-5,310,28,WHITE,r=4);d.text(230,y+2,v,12,INK)
d.p(575,154,342,'<b>Review order</b><br/>Inspect both views and object cells.<br/>Check task steps and constraints.<br/>Save allowed selector edits.<br/>Check every step and approve.',16,INK,max_h=178)
d.p(575,343,342,'<b>t0 / t1 provenance</b><br/>Actual browser approval; revision 2; both steps checked; plans unchanged.<br/>Original model drafts retained separately.',15,INK,max_h=132)
foot('[UI,M4] actual field semantics; left panel is a labeled vector schematic, not a browser screenshot. Both runs used manual_browser_review.')

# 09
page('Approval binds more than the JSON plan','INTEGRITY / SINGLE USE','A change to the task, initial observation, runtime, configuration or view frame invalidates the bound context.')
for x,w,t,b,c in [(38,265,'Frozen context','Task + episode + initial RGB/state hash<br/>Runtime SHA-256 map<br/>Execution configuration<br/>Preparation + view_frame',BLUE),(347,265,'Approval record','Context hash + reviewed plan<br/>Revision + checked steps<br/>Approval timestamp<br/>Plan hash',TEAL),(656,266,'Consumption','Fresh output required<br/>Claim marker written once<br/>Approval cannot be replayed<br/>Even zero-action failure consumes it',AMBER)]:
    d.card(x,154,w,282,t,b,c,14)
d.arrow(307,284,339,284);d.arrow(615,284,648,284)
note(458,'runtime_sha256 includes @configuration. A git commit alone is insufficient because the release also froze previously uncommitted files.',size=13)
foot('[M4] _context, _approval_binding, approve, claim; [M5] runtime_inventory / execution_options; [L] source manifest.')

# 10
page('UC7.1 defines directions from the external view','DIRECTION SEMANTICS / external_view_v1','The same frozen reference is used in planning language, approval, execution requests and region solving.')
codebox(38,154,465,180,[
 'R = world-from-external-camera rotation',
 'r = normalize(R[:2, 0])',
 'a = (-r_y, r_x)',
 '',
 'right_of = +r       left_of  = -r',
 'behind   = +a       front_of = -a',
 ],size=12,leading=24)
d.p(38,359,465,'For these recorded LIBERO scenes:<br/><b>r ≈ (0, +1)</b>; <b>a ≈ (-1, 0)</b>.<br/>World +z remains up. Numeric motion targets and region bounds still use world metres.',16,INK,max_h=124)
# Vector diagram: the frozen basis is illustrative; no task target truth is drawn.
d.rect(549,154,373,280,LIGHT);cx,cy=733,301
d.arrow(cx,cy,cx+123,cy,TEAL,3);d.arrow(cx,cy,cx,cy-103,BLUE,3)
d.arrow(cx,cy,cx-115,cy,TEAL,2);d.arrow(cx,cy,cx,cy+85,BLUE,2)
d.text(775,314,'right = +r',12,TEAL,'Bold');d.text(562,314,'left = -r',12,TEAL)
d.text(672,174,'behind = +a',12,BLUE,'Bold');d.text(674,401,'front = -a',12,BLUE)
d.text(701,278,'anchor',11,MUTED)
d.p(549,450,373,'Global horizontal convention, not a perspective Jacobian. Wrist images never redefine these words.',12,MUTED,max_h=45)
foot('[M8] build_view_frame / region_directions; [L] A1–A3; formula uses calibration, not hard-coded LIBERO axes.')

# 11
page('One direction source across the stack','DIRECTION PROPAGATION AND LIMITS','The full task contract is sent initially; compact region_directions is included in every executive request.')
table(['Stage','Implementation contract'],[
 ['Preparation / approval','Compute and freeze preparation.view_frame; retain calibration, right_xy, away_xy, roll and unavailable reason.'],
 ['PlanLock / policy context','Carry the frozen frame through approval consumption and the persistent episode context.'],
 ['Execution request','Supply the compact direction table even when the reviewed regions list is empty.'],
 ['Directional solver','For direction d, keep candidate points p with p·d greater than the anchor polygon’s maximum projection.'],
 ['Unavailable / display','Missing calibration does not fall back to legacy world-axis directions in the solver. Axis-wording highlights are diagnostic only.'],
 ],row_h=53,font=13)
note(461,'Large roll/inverted views remain a limitation. UC7.1 records them; it adds no image rotation, E1 preference rule or UC8 changes.',AMBER,12.5)
foot('[M8] spatial solver; [M1,M2,M4,M5] propagation; [L] A0–A9 release. t0/t1 contain no directional regions, so do not validate that feature.')

# 12
page('Persistent model decisions between sequences','EXECUTIVE REQUEST / RESPONSE','The agent keeps textual task history and a bounded visual history instead of restarting after every fragment.')
two_cards(
 ('Request context','Task and reviewed goal contract at initialization.<br/>Fresh paired RGB + measured robot state.<br/>Latest execution feedback and history.<br/>Frozen region_directions every request.<br/>Up to six recent visual pairs.',BLUE),
 ('Decision contract','belief: explicitly model-estimated state.<br/>plan / why: intended action and reason.<br/>type: execute, motion, macro, observe or finish.<br/>Only the selected payload is active.<br/>Completed responses are validated before use.',TEAL),height=292,size=15)
note(468,'A model “finish” claim cannot set official task success. Stable gripper gap is measured feedback, not object-attachment proof.',AMBER,13)
foot('[M1,M2] continuous task executive and persistent history; [R0,R1] actual request/result logs with model gpt-6-astra and effort high.')

# 13
page('Action branches and their responsibilities','EXECUTION API','Generic fragments handle repeated local control; optional native macros keep their own prerequisites.')
table(['Branch','Executed behavior','Boundary'],[
 ['<b>execute</b>','Sequence of 1–6 generic segments; at most 240 control steps per grant.','Measured completion and explicit observation checkpoints.'],
 ['<b>motion</b>','Short numerical pose/gripper target; 1–5 local control steps.','A short grant may end before target arrival.'],
 ['<b>macro</b>','Optional hold / turn / place_on / release path.','Its own grounding / calibration / feasibility requirements apply.'],
 ['<b>observe / finish</b>','One pose-hold step retaining gripper command; finish also makes a model completion claim.','Neither replaces the independent benchmark predicate.'],
 ],widths=(125,382,381),row_h=73,font=13)
note(483,'An unavailable macro returns a reason to the model; legacy grasp verification is not a universal gate before direct execution.',size=10)
foot('[M1,M2,M3,M9] policy dispatch, executive schema, measured fragments and optional macro adapters.')

# 14
page('Five reusable measured-motion fragments','FRAGMENT CONTRACT','Coordinates refer to the robot grip site in world metres; orientation is the hand’s absolute world quaternion (wxyz).')
table(['Skill','Semantics'],[
 ['move_to_pose','Absolute world xyz target; quaternion_wxyz or null to retain measured entry orientation.'],
 ['move_relative','Resolve xyz displacement once from the measured segment-entry pose.'],
 ['approach','World displacement using the contact profile; a stall is not a contact certificate.'],
 ['set_gripper','Hold entry pose; open or close as an explicit stage; require pose arrival, minimum duration and stable measured gap.'],
 ['wait','Hold the entry pose and command for its declared max_steps, then return.'],
 ['observation_after','Yield after that segment. Pending stages do not autoplay or resume implicitly after a handoff.'],
 ],row_h=48,font=13)
foot('[M3] fragment_capabilities and execution engine. Skill names are generic; the model selects task-specific numerical values online.')

# 15
page('A real grasp-and-test-lift request','EXECUTION EXAMPLE / t0 DECISION 3','The first descent stalled; the model chose to close at the reached height and inspect a short lift.')
codebox(38,154,535,329,[
 '"execution": {"segments": [',
 '  {"skill": "set_gripper",',
 '   "xyz_m": null, "quaternion_wxyz": null,',
 '   "gripper": "close", "profile": "contact",',
 '   "max_steps": 30, "observation_after": false},',
 '  {"skill": "move_relative",',
 '   "xyz_m": [0, 0, 0.04],',
 '   "quaternion_wxyz": null, "gripper": "keep",',
 '   "profile": "contact", "max_steps": 60,',
 '   "observation_after": true}',
 ']}',
 '',
 '# The 4 cm lift is model-selected,',
 '# not a task-specific executor constant.',
 ],size=11.5,leading=21)
d.image(A/'t0_soup_lift.png',603,154,294,251)
d.p(603,422,294,'Next saved observation (step 195): soup can visibly lifted with the gripper. Attachment is judged from subsequent evidence.',12,MUTED,max_h=72)
foot('[R0] recorded decision 3 at step 149; request 4 image at step 195. Payload excerpt reproduced without a new model request.')

# 16
page('Control bounds are local, not episode caps','MEASURED CONTROL / 20 Hz OSC_POSE','Targets are tracked from measured pose with bounded Cartesian commands.')
table(['Parameter','Transit profile','Contact profile'],[
 ['Target translation from entry','≤ 0.30 m','≤ 0.08 m'],
 ['Per-action translation cap','0.020 m','0.005 m'],
 ['Position arrival tolerance','0.004 m','0.002 m'],
 ['Target rotation / per-action cap','1.2 rad / 0.1 rad','0.5 rad / 0.1 rad'],
 ['Rotation arrival tolerance','0.06 rad','0.04 rad'],
 ],widths=(330,279,279),row_h=44,font=13)
d.p(38,422,884,'≤120 steps per segment; ≤240 per execution grant. Arrival needs two stable samples. Gripper stages require at least 8 open or 12 close actions. These local bounds do not impose a task time/action/call limit.',14,INK,max_h=67)
foot('[M3] frozen fragment capabilities; [M5] episode limits are None. Contact profile means slower control, not guaranteed collision avoidance.')

# 17
page('Return control when execution evidence changes','FEEDBACK AND HANDOFF','Local measurements determine completion or yield; they do not decide physical task success.')
table(['Event','Measured condition'],[
 ['Pose arrival','Position and rotation within profile tolerances for 2 consecutive samples; acknowledge completion only after a real action.'],
 ['Gripper completion','At least 8 open / 12 close actions; 5 recent gaps span ≤0.6 mm; pose arrival also required.'],
 ['Motion stall','After ≥16 actions, error improves by &lt;0.8 mm over 12 samples. Weighted error = position error + 0.05 × rotation error.'],
 ['Maintained-grip gap change','Outside set_gripper while maintaining close: gap rises ≥6 mm above its observed minimum for 3 consecutive samples.'],
 ['Other handoff boundaries','observation_after, completed wait, step budget or target-envelope rejection. Pending segments never autoplay.'],
 ],widths=(256,632),row_h=52,font=12.8)
note(459,'A stall does not certify contact identity. Pose/gap completion does not certify grasp or placement. Continuation requires a new model decision from current evidence.',size=12)
foot('[M1,M3] measured predicates and execution handoff. Thresholds are local fragment rules; they do not impose an episode-wide cap.')

# 18
page('LIBERO-10: the complete task registry','BENCHMARK SCOPE','Zero-based task IDs. Only t0 and t1 are execution examples in this October 30 campaign.')
eval_notes=json.loads((HERE/'evaluation_notes.json').read_text())
rows=[[t['task_key'].replace('libero_10_',''),escape(t['official_prompt'])] for t in eval_notes['tasks']]
table(['ID','Official task language'],rows,widths=(55,833),row_h=30,font=11.2)
foot('[T] src/gpt_field/agent_speedup_v2_tasks.py:53–72; all ten exact official prompts. This guide does not claim a ten-task evaluation.')

# 19
page('Rearranged scenes keep the task unchanged','LAYOUT PREPARATION','Current protocol starts from official init 0 / seed 7 and freezes a validated custom object layout.')
d.image(A/'t0_layout_comparison.png',38,151,525,265)
d.p(38,431,525,'Actual saved t0 reset comparison: original on the left, rearranged on the right. Image content is unmodified.',12,MUTED,max_h=55)
d.p(590,154,332,'<b>Preparation checks</b><br/>Move only free-object XY in the input state; retain robot, fixed objects, Z, orientation and velocity.<br/><br/>Preserve official task and goal logic.<br/>Check stable support and initial goal=false.<br/>Repeat real resets; match observation hashes.<br/><br/>Archive state files, change list, physical validation and both camera previews.',14.3,INK,max_h=332)
foot('[P §3]; [R0,R1] layout_changes.json, preview_manifest.json, physical_validation.json and repeated-reset evidence.')

# 20
page('Supervise one attempt without silent restarts','OPERATING CONTRACT','Each task has a separate approval, fresh output directory and one-use launch marker.')
three_cards([
 ('Before launch','Check frozen runtime hashes.<br/>Verify observation + review binding.<br/>Use an exclusive launch lock.<br/>Keep the approved configuration.',BLUE),
 ('During execution','Supervisor runs the wrapper.<br/>No automatic process restart.<br/>Track requests and feedback.<br/>Contacts are record-only.',TEAL),
 ('At termination','Preserve raw outcome and logs.<br/>Run separate physical diagnosis.<br/>Verify source and video artifacts.<br/>Close private processes/sockets.',AMBER),
 ],height=269,size=14.5)
note(442,'No task-wide time, action or call cap. Official success, existing error/conflict paths and user stop remain termination conditions. User stop is handled at the feedback boundary.',size=14)
foot('[P §§5–6] supervisor autostart=false, autorestart=false, startretries=0; [R0,R1] launch and shutdown records.')

# 21
page('Request retry is different from task retry','TRANSPORT / SUBSCRIPTION ONLY','The 180-second deadline covers HTTP open and the full SSE response after authentication.')
table(['Condition','Frozen behavior'],[
 ['Overload','server_is_overloaded with HTTP 503 or absent status: at most 2 overload retries.'],
 ['Transient transport','Missing completion, truncated event, stream interruption or network error: at most 2 retries in total for this category.'],
 ['Retry boundary','Same observation, no response envelope and no history commit. Partial output is never executed.'],
 ['Independent limits','Overload and transient counters are separate. Mixed errors can exceed 3 total sends; accepted backoff is at most 60 s.'],
 ['Non-retryable / task restart','HTTP 520 / unknown_error and deadline exceeded do not retry. A new full attempt needs new approval and output.'],
 ],row_h=57,font=13)
note(480,'No model downgrade or API-key fallback. t0/t1 recorded zero request errors and zero retries.',size=11)
foot('[M2] client; [X] retry classification and transport deadline; [P §6]; [R0,R1] request logs and terminal audit.')

# 22
page('Report four outcomes, not one SUCCESS label','EVALUATION CONTRACT','The runner stops at official success; physical truth examines the state at that same boundary.')
table(['Layer','Meaning'],[
 ['Official LIBERO goal','task_success / libero_success use the original task predicate; an object may still be held.'],
 ['Strict terminal truth','Physical checks at termination: containment, release, support and rest. Rest uses the recent 10-step displacement/rotation window when available.'],
 ['Post-stop diagnosis','38 diagnostic actions: 10 open + 8 lift + 20 settle. Inspect possible release/settle effects separately.'],
 ['Framework v2_success','raw_terminal AND [strict_terminal OR (hold_only AND strict_after AND raw_after)]. Read the saved frozen result; do not rescore a diagnostic pass.'],
 ],row_h=66,font=13.5)
note(460,'Hold-only: each failing atom has only permitted hold/rest/orientation failures and also fails not_held or robot_not_touching_part. Support/below-rim failures are excluded.',AMBER,13)
foot('[M6] independent physical-truth evaluator; [P §7]; [E] strict checks, HELD_CHECKS and terminal_failure_is_hold_only.')

# 23
page('Completion time and model-call accounting','MEASUREMENT','Report official completion time only when official success and completion_time_eligible are true.')
codebox(38,151,884,88,[
 'official completion time = control_elapsed_s - video_render_s',
 '                         = control_without_video_s',
 ],size=16,leading=27)
table(['Recorded component','t0','t1'],[
 ['Control-loop elapsed','532.04 s','450.99 s'],
 ['Video render overhead','137.52 s','109.83 s'],
 ['Official completion, video excluded','394.52 s','341.16 s'],
 ['Requests: execution + reused draft','10 + 1 = 11','12 + 1 = 13'],
 ],widths=(480,204,204),top=255,row_h=41,font=13)
note(463,'Time includes model wait, control computation, simulator actions and logging; excludes setup, planning, approval and post-stop diagnosis. It is not video playback length.',size=12)
foot('[P §8]; [R0,R1] summary.json and timing_components; retrospective removal of measured video-render overhead.')

# 24
page('t0: soup and tomato sauce into the basket','RECORDED EXECUTION / 885 POLICY ACTIONS','One approved rearranged attempt; 10 accepted execution decisions, all using measured fragments.')
scene_strip(['t0_initial.png','t0_soup_lift.png','t0_soup_placed_wrist.png'],[
 'Initial external view.<br/>Two target objects and the basket.',
 'External view, step 195.<br/>Soup is lifted after a short test lift.',
 'Wrist view, step 506.<br/>Soup placement visible in the basket.',
 ])
foot('[R0] actual saved initial and request images. Policy execution proceeds next to the tomato sauce; no generated scene imagery.')

# 25
page('t0: official success, incomplete physical placement','TERMINAL STATE / DIAGNOSIS','At termination, soup passes strict checks; tomato sauce remains held and fails support/rest checks.')
d.image(A/'t0_terminal_dashboard.png',38,151,548,329)
d.p(613,152,309,'<b>Official: true</b><br/><b>Strict terminal: false</b><br/><b>After-settle strict: false</b><br/><b>Framework v2: false</b><br/><br/>After diagnostic release, tomato becomes unheld and at rest, but still fails admissible container support / rim-related checks.<br/><br/>Its lowest point is about 15.9 mm above the container floor.',14,INK,max_h=332)
foot('[R0] terminal_frame.png and trial/physical_truth.json. Dashboard SUCCESS denotes the official predicate, not strict physical completion.')

# 26
page('t1: cream cheese and butter into the basket','RECORDED EXECUTION / 715 POLICY ACTIONS','One approved rearranged attempt; 12 accepted execution decisions, all using measured fragments.')
scene_strip(['t1_initial.png','t1_cheese_placed_wrist.png','t1_butter_lift.png'],[
 'Initial external view.<br/>Cheese, butter and the basket.',
 'Wrist view, step 317.<br/>Cream cheese placement is visible.',
 'External view, step 529.<br/>Butter is lifted for transfer.',
 ])
foot('[R1] actual saved initial and request images; terminal evaluation remains separate from these visual milestones.')

# 27
page('t1: settling helps, but does not change v2 failure','TERMINAL STATE / HOLD-ONLY RULE','The official predicate fires during opening, after 5 opening actions and before planned withdrawal/settling.')
d.image(A/'t1_terminal_dashboard.png',38,151,548,329)
d.p(613,152,309,'<b>Official: true</b><br/><b>Strict terminal: false</b><br/><b>After-settle strict: true</b><br/><b>Framework v2: false</b><br/><br/>Butter fails release, rest, support and below-rim checks at termination. Cheese also fails rest.<br/><br/>These failures exceed the per-object hold-only exception. The later diagnostic pass cannot relabel the attempt.',14,INK,max_h=332)
foot('[R1] terminal_frame.png, physical_truth.json and summary.v2_success; frame step 0714 is zero-based, corresponding to 715 policy actions.')

# 28
page('Two attempts: audited outcome comparison','RESULTS / NO BASELINE OR SWEEP CLAIM','Both tasks reached the official predicate; neither passed the frozen framework success rule.')
table(['Measure','t0','t1'],[
 ['Official success','true','true'],
 ['Strict at terminal','false','false'],
 ['Strict after diagnostic settling','false','true'],
 ['Framework v2_success','false','false'],
 ['Policy actions / execution requests','885 / 10','715 / 12'],
 ['Official completion, video excluded','394.52 s','341.16 s'],
 ['Flagged contact samples (record-only)','11,943','4,035'],
 ],widths=(480,204,204),row_h=40,font=13)
note(480,'Contact samples are repeated physics observations, not counts of distinct collisions. Different tasks do not establish a speedup ratio.',size=10.8)
foot('[R0,R1] 23 terminal audit checks each passed; one reused planning draft each; no retry, model downgrade or API-key call.')

# 29
page('Read the video as control telemetry','PLANNING VIDEO / 1600×960 AT 20 fps','Saved active frame with XY, XZ and ZY projections and an embedded legend.')
d.image(A/'t1_active_dashboard.png',38,151,548,329)
d.p(613,151,309,'<b>Blue arrows:</b> bounded direction toward the active fragment target.<br/><b>White point:</b> gripper.<br/><b>Yellow cross:</b> active target.<br/><b>Pink trail:</b> measured motion.<br/><b>Dashed yellow:</b> entry-to-target reference.<br/><br/>This is a target-error control field. It does not represent obstacle clearance or certify collision safety.',14,INK,max_h=332)
foot('[P §9]; [R1] active_frame.png. No stale target arrows at termination. API wait is compressed; playback length is not completion time.')

# 30
page('Validation coverage and known limits','EVIDENCE QUALITY','Release tests, planning probes and the current task attempts answer different questions.')
table(['Evidence','What it establishes—and its limit'],[
 ['V0 release checks','350 passed, 59 pre-existing failures; 71 new checks passed, no new failures. The full regression suite was not green.'],
 ['V1 planning probes','13 scenes, 13 planning calls, zero policy actions. Raw draft legality 12/13; direction/part checks 38/38.'],
 ['Current campaign checks','65 focused offline checks passed; source hashes matched; 23 terminal audit checks passed for each attempt. These are integrity checks, not task success.'],
 ['Current execution scope','Two single attempts on custom layouts. Both plans have regions=[]; they do not test directional-region behavior.'],
 ['Unmeasured claims','No ten-task execution sweep, multi-seed robustness, matched baseline timing, collision-free guarantee or measured speedup is supplied here.'],
 ],row_h=58,font=12.5)
foot('[L] release validation; [R0,R1] prelaunch / terminal audits; [E] evaluation_notes.json preserves source locations and caveats.')

# 31
page('Reproduce the workflow with fresh approval','OPERATOR REFERENCE / COMMAND SHAPE','These are command templates, not actions performed while generating this document.')
codebox(38,151,884,213,[
 'cd /workspace/gpt_field',
 'RUNNER=scripts/evaluate_agent_speedup_v2_user_check.py',
 '# Fresh TASK_DIR; verified initial_state.npy + expected observation hash.',
 '.venv/bin/python "$RUNNER" preflight [task/layout/config flags]',
 '.venv/bin/python "$RUNNER" prepare   [same flags] --review-dir "$TASK_DIR/review"',
 '.venv/bin/python "$RUNNER" review --review-dir "$TASK_DIR/review" \\',
 '  --host 127.0.0.1 --port 8765',
 '# After valid approval, invoke once through the supervised wrapper:',
 '.venv/bin/python "$RUNNER" run --review-dir "$TASK_DIR/review" \\',
 '  --output "$TASK_DIR/trial"',
 ],size=11,leading=19)
d.p(38,382,884,'<b>Required preparation settings:</b> --reasoning-effort high --planning-video --request-deadline-s 180 --stream-progress; official episode index 0, verified custom state and observation hash. Use the protocol’s environment and simulator Python. The runner config selects Astra; there is no --model flag.',13.5,INK,max_h=77)
note(471,'The exact executable templates are in [P §5]. start --auto-approve also executes; it is not a preparation command.',AMBER,11.5)
foot('[P §§4–6] command templates, environment, browser loopback access and one-attempt supervised launch. Never reuse a consumed review.')

# 32
page('Source map and reproducible document assets','TECHNICAL REFERENCES','Project-relative source IDs used in footers; detailed line ranges and file hashes accompany the PDF.')
source_rows=[
 ['P / L','UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md<br/>reports/uc7_direction_semantics_20261028/implementation_uc71/'],
 ['M1 / M2 / M3','src/gpt_field/agent_speedup_v2_user_check_agent.py<br/>agent_speedup_v2_user_check_agent_client.py / agent_speedup_v2_user_check_fragments.py'],
 ['M4 / M5','src/gpt_field/agent_speedup_v2_user_check.py<br/>scripts/evaluate_agent_speedup_v2_user_check.py'],
 ['UI / E','src/gpt_field/agent_speedup_v2_user_check.html / agent_speedup_v2_user_check_server.py<br/>src/gpt_field/agent_speedup_v2_truth.py'],
 ['M6 / M7','src/gpt_field/agent_speedup_v2_user_check_sim.py<br/>src/gpt_field/agent_speedup_v2_user_check_client.py'],
 ['M8 / M9 / X / T','In src/gpt_field/: …_user_check_spatial.py / …_user_check_agent.py + agent_speedup_v2.py<br/>…_client.py + …_user_check_transport.py / …_tasks.py; ellipsis = agent_speedup_v2'],
 ['R0 / R1','runs/agent_speedup_v2_uc71_relayout_t0_t1_20261030_001/task_0/ and task_1/<br/>Saved trial, planning/review, approval, video, terminal audit and shutdown artifacts'],
 ]
table(['ID','Source or retained artifact'],source_rows,widths=(130,758),row_h=40,font=10.6)
d.p(38,469,884,'Companion files: builder, evidence.json, technical_notes.json, evaluation_notes.json, exact scene images, extracted PDF text and rendering QA. Full source paths and line ranges are in the notes.',10.4,MUTED,max_h=30)
foot('Frozen archive SHA-256: b3e8f84f3bf1b7603fad0ae0d8c4bee8a5a125b50ab08e320394b4069af57a68')

d.save()
(HERE/'page_manifest.json').write_text(json.dumps(d.pages,indent=2)+'\n')
(HERE/'layout_bounds.json').write_text(json.dumps(d.bounds,indent=2)+'\n')
print(f'Wrote {OUT} ({d.n} pages)')
