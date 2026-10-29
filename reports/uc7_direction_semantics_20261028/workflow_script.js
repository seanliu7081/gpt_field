export const meta = {
  name: 'uc7-direction-semantics-fix',
  description: 'Verify the t6 right_of=+x root cause, audit every direction-semantics code site, design and adversarially judge a general fix (read-only, no code changes)',
  phases: [
    { title: 'Investigate & design', detail: 'code-path audit, LIBERO/RoboCasa semantics check, two independent designs, adversarial critique of the lead proposal' },
    { title: 'Judge', detail: 'reconcile everything into one prioritized suggestion list' },
  ],
}

const SCRATCH = '/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem'

const CONSTRAINTS = `HARD CONSTRAINTS FOR YOU:
- READ-ONLY on /workspace: do not edit, create or delete anything under /workspace (src, runs, reports, LIBERO). The user explicitly said "do not change the code yet".
- You may write helper scripts/outputs ONLY under ${SCRATCH}/<your-label>/ (create that subfolder).
- Do NOT launch the simulator with rendering and do NOT use GPUs (other jobs run there). Parsing XML/meshes/JSON with python3 is fine. Do NOT make any LLM/model API calls (the model subscription is shared with the user's live runs).
- Some run files are huge (trace.jsonl ~700 MB, episode.events.jsonl ~20 MB). Never cat them; use grep -m / streaming python with targeted keys.
- Cite evidence as file:line or file + JSON key. Say "unverified" when you could not check something. Be concrete and terse.`

const RULES = `USER'S FRAMEWORK RULES (binding on any fix):
1. One frozen code version and one threshold table run every task. No task key reaches the agent; the agent gets only the task sentence, images, robot state and camera calibration.
2. No change may be justified by a single task. Every change must be a general capability (relation, primitive, observer, measurement rule) and must not break other tasks' recorded runs.
3. A failure on any task is a framework gap with a general fix; never patch per task.
4. Prefer removing veto checks over adding thresholds.
5. Organise fixes by mechanisms that recur on 2+ tasks or by structural capability gaps, not by one run's timeline. Acceptance is an all-task sweep on one frozen build.`

const FACTS = `CONTEXT. /workspace/gpt_field is an LLM-driven robot-control framework evaluated on LIBERO-10. The current line is "agent_speedup_v2 user-check" revision 7 (uc7): a planner model call writes a JSON plan (parts, steps, regions, placement_goals) from 4 images; then a continuous "executive" model-call loop controls a Franka arm with world-frame targets (it gets external+wrist images, measured robot state and camera calibration). Source: /workspace/gpt_field/src/gpt_field/agent_speedup_v2_user_check*.py (+ .html review UI). The latest run's frozen source equals current src for the spatial module.

FAILURE (2026-10-28, run dir R=/workspace/gpt_field/runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6/ ; see R/direction_semantics_audit.json):
- LIBERO-10 task 6 sentence: "put the white mug on the plate and put the chocolate pudding to the right of the plate". Official goal: And(On porcelain_mug_1 plate_1, On chocolate_pudding_1 living_room_table_plate_right_region). plate_right_region = world x in [0.10,0.20], y in [0.05,0.15] (living-room table offset (0,0,0.41)); plate centre ~(0.149, 0.006). BDDL copy: R/original_task.bddl. This run used a custom layout (mug/pudding/red mug moved; plate kept at its official pose): R/layout_changes.json.
- The planning prompt appends SPATIAL_INSTRUCTIONS (src/gpt_field/agent_speedup_v2_user_check_spatial.py:51-75) which says "Relations use WORLD axes: right_of=+x, left_of=-x, behind=+y, front_of=-y; they do not mean screen directions." (lines 59-60). It also appends METRIC_INSTRUCTIONS (src/gpt_field/agent_speedup_v2_user_check_client.py:132-138): "World +z is up; do not assume image left/right is world -x/+x." The plan request carries NO camera calibration, only 4 images (external clean/grid, wrist clean/grid; the wrist image is turned so its left/right match the external image). Full first plan request: R/review/requests/000001-*.request.json.
- Region resolver _ground_observed_region (spatial.py:166-263; side tests at 210-214): right_of -> cells with x > anchor polygon max x; left_of -> x < min x; behind -> y > max y; front_of -> y < min y. Null bounds => the whole observed support half-plane beyond the anchor (minus other objects); region centre = mean of those cells (spatial.py:247).
- The model's approved plan (R/trial/approved_plan.json): region E = {support observed_plane, anchor C (plate), relation right_of, frame world, bounds null}; part E text "table: surface to the plate's world +x side", pointed external grid cell x=55; plate C at cell 61 (8x8 grid, cells 1-64 row-major from top-left, 512 px images).
- Ray-casting the cell-55 centre (u,v)=(416,416) onto plane z=0.44 with the agentview calibration gives world ~(0.113, +0.180): the +y side of the plate = LIBERO's "right". Sanity: cell 35 (pudding) -> (-0.138,-0.140) vs true (-0.16,-0.14); cell 38 (mug) -> (-0.138,+0.142) vs true (-0.12,0.18). So the planner's own visual pointing was on the correct side; its relation word and text followed the prompt's +x rule.
- Executive calibration (R/trial/requests/000001-*.request.json, key rotation_world_from_camera): agentview position (0.6066, 0, 0.96), R[:,0] = (0,1,0) -> external image-right = world +y; view direction -R[:,2] = (-0.849, 0, -0.529). Wrist camera (robot0_eye_in_hand) at the start pose has R[:,0] = (-0.0015,-1.0,-0.0005).
- The executive released the pudding at ~(0.29,-0.02) (+x side), declared finish, then spent ~10 calls re-centering the mug; the user stopped at step 1454. Final pudding COM (0.276,-0.020).
- Earlier t6 runs, all with the same +x rule (runs/agent_speedup_v2_user_check_libero_10_t6_0{06,07,09,11}/): t6_007 succeeded (pudding COM (0.145,0.146); part E text "table: bare top surface"); t6_009 first released at (0.30,0.02), later corrected toward +y but overshot (final COM (0.187,0.170), 2 cm past y_max=0.15; part E text "surface region to world +x side of plate"); t6_006 and t6_011 never moved the pudding.
- LIBERO convention evidence: every LIBERO problem class places the "agentview" camera on +x at y=0 looking toward -x (LIBERO/libero/libero/envs/problems/*.py set_camera). Parsing all BDDL tasks whose language has left/right/front/back (libero/libero/bddl_files) shows right/left regions at +y/-y of their anchor and front/back at +x/-x (front = toward the camera). LIBERO root: /workspace/heading_flow_pub/third_party/LIBERO/libero/libero/. Parser used: ${SCRATCH}/bddl_regions.py.
- A prior analysis (2026-10-26) already flagged this and proposed, but never implemented, "viewer-centric spatial words" for a future uc8: /workspace/gpt_field/reports/agent_speedup_analysis_20261026/plan/final_plan.md section 1.5(a) and plan/C_generality_reliability.md section S4(a); verification.md has the t6_009 recomputation. That proposal's "adjacency default" is a pending user decision (D9) the user may drop.
- The user also runs this framework on RoboCasa: /workspace/gpt_field/runs/robocasa_coffee_user_check_001 .
- Recorded uc successes on other tasks that a fix must not break include runs/agent_speedup_v2_user_check_libero_10_t4_001 (left/right plates), t5_001/t5_002 (back compartment of the caddy), t6_007, and the latest relayout task_7 (R/../task_7).`

const P0 = `LEAD PROPOSAL P0 (from the main session; critique it, do not assume it is right):
P0.1 Single source of truth: a helper derives support-plane unit vectors from the EXTERNAL camera calibration: right = normalize(R[:2,0]) (camera +X projected on the support plane), away = normalize(-R[:2,2]) (view direction projected). right_of=+right, left_of=-right, behind=+away (farther from the external camera), front_of=-away (nearer to it). LIBERO agentview gives right=(0,+1), away=(-1,0).
P0.2 Planning prompt: replace spatial.py:59-60 with a viewer-frame definition ("as seen in image 1, the external camera; behind = farther from that camera; front_of = nearer"), and inject this scene's numeric mapping (e.g. "image-right on the table = world (+0.00,+1.00); farther from the camera = world (-1.00,+0.00)") computed from calibration at request time; ask region part text to state both (e.g. "right of plate (world +y)"). Replace the negative METRIC_INSTRUCTIONS sentence (client.py:135) with the same positive mapping.
P0.3 Grounding: replace the axis tests at spatial.py:210-214 with projections onto the relation vector d: keep cells whose (xy - anchor_centre).d exceeds the anchor polygon's max projection on d. Bounds semantics unchanged.
P0.4 Executive prompt (agent_speedup_v2_user_check_agent_client.py calibration paragraph ~179-190): one sentence with the same definition + numbers, and "direction words refer to the external image, not the wrist image".
P0.5 Zero-call consistency check at plan validation: ray-cast each region part's pointed external cell to the support plane and check it lies on the relation's side of the anchor (sign test, no threshold). Start diagnostic-only (logged + shown in review), optionally a plan issue later.
P0.6 Optional (user decision D9): a null-bounds "<side>_of" region means beside the anchor: laterally within the anchor's extent, nearest fitting free spot just past the anchor edge, instead of the centroid of a half-plane.
P0.7 Verification before live: unit test for P0.1 on the LIBERO camera quaternions; offline harness test that for every LIBERO BDDL '<dir> of <anchor>' region the anchor->region offset aligns with the derived vector; plan-only calls on t4/t5/t6; then live t6 (official layout + relayout) and t4/t5 regression.`

const AUDIT_SCHEMA = {
  type: 'object',
  properties: {
    sites: { type: 'array', items: { type: 'object', properties: {
      file: { type: 'string' }, line: { type: 'string' },
      kind: { type: 'string', enum: ['plan_prompt', 'executive_prompt', 'grounding', 'aiming', 'validation', 'review_ui', 'image_pipeline', 'calibration', 'other'] },
      encodes: { type: 'string' }, must_change_for_fix: { type: 'boolean' }, note: { type: 'string' } },
      required: ['file', 'line', 'kind', 'encodes', 'must_change_for_fix', 'note'] } },
    image_orientation: { type: 'string', description: 'what the planner and the executive see: external mirrored or not; wrist turned or not for each; what image-right means in world for each' },
    executive_region_inputs: { type: 'string', description: 'whether/how the executive receives grounded region geometry or relation meaning' },
    null_bounds_behaviour: { type: 'string', description: 'what region cells/centre a null-bounds right_of produces, what aim_target and _measure_goal would do with a far placement on the correct side' },
    calibration_available_at_plan_time: { type: 'string', description: 'can the external camera calibration be computed before the plan call; where; what code path' },
    review_ui_regions: { type: 'string' },
    other_findings: { type: 'array', items: { type: 'string' } },
  },
  required: ['sites', 'image_orientation', 'executive_region_inputs', 'null_bounds_behaviour', 'calibration_available_at_plan_time', 'review_ui_regions', 'other_findings'],
}

const SEMANTICS_SCHEMA = {
  type: 'object',
  properties: {
    camera_frames: { type: 'array', items: { type: 'object', properties: {
      problem_class: { type: 'string' }, image_right_world: { type: 'string' }, view_dir_world: { type: 'string' } },
      required: ['problem_class', 'image_right_world', 'view_dir_world'] } },
    tasks_checked: { type: 'integer' },
    counterexamples: { type: 'array', items: { type: 'object', properties: {
      task: { type: 'string' }, word: { type: 'string' }, finding: { type: 'string' } },
      required: ['task', 'word', 'finding'] } },
    object_intrinsic_cases: { type: 'array', items: { type: 'object', properties: {
      task: { type: 'string' }, word: { type: 'string' }, finding: { type: 'string' } },
      required: ['task', 'word', 'finding'] } },
    side_of_anchor_regions: { type: 'array', items: { type: 'object', properties: {
      task: { type: 'string' }, region: { type: 'string' }, anchor: { type: 'string' },
      along_offset_m: { type: 'number' }, lateral_offset_m: { type: 'number' },
      region_half_extent_m: { type: 'string' }, anchor_footprint_radius_m: { type: 'string' }, note: { type: 'string' } },
      required: ['task', 'region', 'anchor', 'along_offset_m', 'lateral_offset_m', 'region_half_extent_m', 'anchor_footprint_radius_m', 'note'] } },
    libero_on_region_test: { type: 'string', description: 'exactly what LIBERO tests for On(obj, region), with file:line' },
    t6_band: { type: 'string', description: 'plate radius, pudding footprint, the band of pudding-centre positions inside plate_right_region and whether it overlaps the plate; compare t6_007 and t6_009' },
    robocasa: { type: 'string' },
    summary: { type: 'string' },
  },
  required: ['camera_frames', 'tasks_checked', 'counterexamples', 'object_intrinsic_cases', 'side_of_anchor_regions', 'libero_on_region_test', 't6_band', 'robocasa', 'summary'],
}

const DESIGN_SCHEMA = {
  type: 'object',
  properties: {
    design_name: { type: 'string' },
    summary: { type: 'string' },
    changes: { type: 'array', items: { type: 'object', properties: {
      file: { type: 'string' }, where: { type: 'string' }, what: { type: 'string' } },
      required: ['file', 'where', 'what'] } },
    new_model_inputs: { type: 'array', items: { type: 'string' } },
    extra_model_calls: { type: 'string' },
    handles: { type: 'object', properties: {
      official_t6: { type: 'string' }, relayout_t6: { type: 'string' }, t4_left_right_plates: { type: 'string' },
      t5_back_compartment: { type: 'string' }, robocasa: { type: 'string' }, wrist_view: { type: 'string' } },
      required: ['official_t6', 'relayout_t6', 't4_left_right_plates', 't5_back_compartment', 'robocasa', 'wrist_view'] },
    rule_compliance: { type: 'string' },
    risks: { type: 'array', items: { type: 'string' } },
    tests_before_live: { type: 'array', items: { type: 'string' } },
  },
  required: ['design_name', 'summary', 'changes', 'new_model_inputs', 'extra_model_calls', 'handles', 'rule_compliance', 'risks', 'tests_before_live'],
}

const CRITIQUE_SCHEMA = {
  type: 'object',
  properties: {
    items: { type: 'array', items: { type: 'object', properties: {
      item: { type: 'string' }, verdict: { type: 'string', enum: ['keep', 'modify', 'drop'] },
      reason: { type: 'string' }, evidence: { type: 'string' }, modification: { type: 'string' } },
      required: ['item', 'verdict', 'reason', 'evidence', 'modification'] } },
    missing_from_p0: { type: 'array', items: { type: 'string' } },
    regression_risks: { type: 'array', items: { type: 'string' } },
    rule_conflicts: { type: 'array', items: { type: 'string' } },
  },
  required: ['items', 'missing_from_p0', 'regression_risks', 'rule_conflicts'],
}

const JUDGE_SCHEMA = {
  type: 'object',
  properties: {
    root_cause_verdict: { type: 'string' },
    suggestions: { type: 'array', items: { type: 'object', properties: {
      id: { type: 'string' }, priority: { type: 'string', enum: ['must', 'should', 'optional'] },
      title: { type: 'string' }, change: { type: 'string' }, files: { type: 'string' },
      evidence: { type: 'string' }, effort: { type: 'string' }, risks: { type: 'string' } },
      required: ['id', 'priority', 'title', 'change', 'files', 'evidence', 'effort', 'risks'] } },
    rejected_or_deferred: { type: 'array', items: { type: 'object', properties: {
      idea: { type: 'string' }, reason: { type: 'string' } }, required: ['idea', 'reason'] } },
    open_questions_for_user: { type: 'array', items: { type: 'string' } },
    verification_plan: { type: 'array', items: { type: 'string' } },
    corrections_to_facts: { type: 'array', items: { type: 'string' }, description: 'any fact in the brief that the investigators found wrong or imprecise' },
  },
  required: ['root_cause_verdict', 'suggestions', 'rejected_or_deferred', 'open_questions_for_user', 'verification_plan', 'corrections_to_facts'],
}

phase('Investigate & design')

const auditPrompt = `${FACTS}

${CONSTRAINTS}

YOUR TASK (label: audit): find EVERY place in the current uc7 runtime where the meaning of direction words (left/right/front/behind/back/side), an image-vs-world frame mapping, or region side selection is defined, assumed, displayed or checked, so that a fix can be complete and consistent. Scope: /workspace/gpt_field/src/gpt_field/agent_speedup_v2_user_check*.py and agent_speedup_v2_user_check.html; base modules they import for region grounding/aiming/validation (agent_speedup_v2_planner.py aim_target/_region_cells/_centre_xy; agent_speedup_v2_relations.py; agent_speedup_v2_scene.py; agent_speedup_v2_client.py plan content builder); the runner scripts under /workspace/gpt_field/scripts/ that build user-check plan requests. Answer:
(1) each site with file:line and what it encodes;
(2) does the executive receive grounded region geometry (centre/cells/relation) or the relation meaning, and in what form (check the executive context builder in agent_speedup_v2_user_check_agent.py / _agent_client.py and a saved executive request in R/trial/requests/);
(3) image orientation actually shown to the planner and to the executive: is the external image mirrored? Is the wrist image turned for the planner? For the executive? What does image-right mean in world for each (use the saved calibration and the code that builds external_rgb_upright / wrist_rgb_upright / the turned wrist image, e.g. agent_speedup_v2_sim.py ~550-640);
(4) what the review UI shows the user for regions (would a human reviewer see the world-axis meaning or a region overlay?);
(5) null-bounds behaviour: what region cells/centre a right_of with null bounds produces, what aim_target (spatial.py:299-351) and _measure_goal (spatial.py:433-496) would do with a placement far from the anchor but on the correct side, and whether these paths are even used in uc7 runs (macro grants vs direct motion);
(6) whether the external camera calibration is available before the plan call (prepare_review / initial observation / sim worker) and via which code path, so calibration-derived axes could be injected into the plan request;
(7) any other module with left/right/front/behind semantics (macros, skills observe_spatial_region, fragments, transport).`

const semanticsPrompt = `${FACTS}

${CONSTRAINTS}

YOUR TASK (label: semantics): adversarially verify the claim "LIBERO direction words are in the agentview viewer frame: right = image-right = world +y, left = -y, front = toward the camera (+x), behind/back = away from the camera (-x)", find any counterexample, and quantify the geometry of '<side> of <anchor>' goal regions.
(1) Re-derive image-right (R[:,0]) and view direction (-R[:,2]) for the agentview camera of every LIBERO problem class from the set_camera quaternions (wxyz) in libero/libero/envs/problems/*.py.
(2) For every BDDL under libero/libero/bddl_files/* whose language has left/right/front/back/behind/middle, compare the referenced object/region geometry with the claim (the parser ${SCRATCH}/bddl_regions.py prints region centres; reuse it but do not trust it blindly; region ranges are (x_min, y_min, x_max, y_max) in the table frame).
(3) Object-intrinsic words: desk_caddy compartments (front/back/left/right contain regions): find the caddy object XML sites and the caddy init yaw (BDDL :yaw_rotation) and compute the compartments' world directions; libero_goal "front of the stove" (find the stove position); LIBERO-10 t5 "back compartment of the caddy". Report whether each matches the viewer frame.
(4) Read LIBERO's predicate/object-state code (libero/libero/envs/predicates, envs/object_states, envs/regions or similar) and state exactly what On(obj, region) tests (origin in box? z band? contact?) with file:line.
(5) For every '<side> of <anchor>' goal region (e.g. plate_right_region/plate_left_region in LIVING_ROOM_SCENE6 and KITCHEN_SCENE7, porcelain_mug_front_region, desk_caddy_right_region, red_coffee_mug_behind_region, stove_front_region) compute the along-direction and lateral offset of the region centre from the anchor's init region centre and the region half-extents, and the anchor footprint radius from the object's XML/mesh bounding box (parse .obj/.stl vertices with numpy; no simulator).
(6) For t6: plate footprint radius and chocolate pudding footprint; the band of pudding-centre positions inside plate_right_region, whether a pudding there must overlap the plate rim; compare with t6_007 COM (0.145,0.146, success) and t6_009 COM (0.187,0.170, failure).
(7) RoboCasa (/workspace/gpt_field/runs/robocasa_coffee_user_check_001): which external camera is used, its rotation (from saved requests if present), whether a hard-coded world-axis convention would be wrong there and whether a calibration-derived viewer frame is meaningful.`

const designBrief = (angle) => `${FACTS}

${RULES}

${CONSTRAINTS}

YOUR TASK (label: design): design a fix for this failure class. Do not write code; specify changes at file/function level, grounded in the actual source (read the files named above). DESIGN ANGLE: ${angle}
Cover: what definition of direction words the model gets and why; how the plan request and the executive request change; how region grounding converts words to geometry; how the system would notice a frame mismatch in the future (if at all) without adding vetoes; what happens for the official t6 layout, the relayout, t4 (left/right plates), t5 (back compartment of the caddy), RoboCasa and the wrist view; which tests to run before a live run. Stay inside the user's rules.`

const critiquePrompt = `${FACTS}

${RULES}

${P0}

${CONSTRAINTS}

YOUR TASK (label: critique): try to REFUTE or improve each P0 item using the source code and run artifacts; default to skepticism. Check in particular:
(a) P0.5: is the pointed-cell cross-check sound? Consider grid coarseness (64 px cells), the anchor's own cell, ray-plane depth error, region parts pointed at x=0 (t6_009 and t6_011 had x=0 for part E), and whether it is effectively a new veto (user rule 4).
(b) P0.6: risks of an adjacency default (evidence base, interaction with aim_target / place_on macros, objects larger than the gap, whether LIBERO's region boxes overlap the anchor).
(c) wrist-view confusion for the executive; object-intrinsic words (caddy compartments, "front of the stove"); "middle"; vertical words (top/bottom drawer) that must NOT be converted.
(d) regressions for recorded successes (t4_001, t5_001, t5_002, t6_007, relayout task_7): read their approved_plan.json and check whether any used a directional region relation or would read differently under P0.
(e) anything P0 misses: e.g. bounds_xy_m semantics, the 'frame' enum value 'world', review UI, the plan-correction loop, executive context, html display, other prompts.
For each P0 item give keep/modify/drop with evidence.`

const [audit, semantics, designGeneral, designMinimal, critique] = await parallel([
  () => agent(auditPrompt, { label: 'audit:code-paths', phase: 'Investigate & design', schema: AUDIT_SCHEMA }),
  () => agent(semanticsPrompt, { label: 'verify:libero-semantics', phase: 'Investigate & design', schema: SEMANTICS_SCHEMA }),
  () => agent(designBrief('the most robust GENERAL design for the long term: correct for any external camera pose and any benchmark the framework will meet (e.g. RoboCasa), minimal reliance on the model doing frame conversions, and a principled representation of spatial relations. Consider alternatives such as grounding regions from the model\'s own visual pointing, explicit viewer-frame relation names, or model-written direction vectors, and pick one with reasons.'),
    { label: 'design:general', phase: 'Investigate & design', schema: DESIGN_SCHEMA }),
  () => agent(designBrief('the SMALLEST SAFE change to the frozen uc7 runtime that fixes this failure class on all tasks with no regressions: roughly 1-2 hours of work, zero extra model calls, no new vetoes or thresholds. Be explicit about what you deliberately leave out.'),
    { label: 'design:minimal', phase: 'Investigate & design', schema: DESIGN_SCHEMA }),
  () => agent(critiquePrompt, { label: 'critique:P0', phase: 'Investigate & design', schema: CRITIQUE_SCHEMA }),
])

const got = { audit, semantics, designGeneral, designMinimal, critique }
const missing = Object.entries(got).filter(([, v]) => !v).map(([k]) => k)
if (missing.length) log(`Missing outputs (agent failed or skipped): ${missing.join(', ')}`)

phase('Judge')

const judgePrompt = `${FACTS}

${RULES}

${P0}

${CONSTRAINTS}

You are the judge. Below are five independent outputs: a code-path audit, a LIBERO/RoboCasa semantics verification, two independent fix designs (general and minimal), and an adversarial critique of P0. Reconcile them into ONE prioritized list of fix SUGGESTIONS for the user (the user asked for suggestions only; no code is to be changed yet). Rules for you:
- Prefer claims supported by evidence from the audit/semantics outputs; where outputs disagree, check the source yourself (read-only) and decide.
- Mark each suggestion must/should/optional. "must" = required to fix the root cause correctly and consistently. Keep the list short and concrete (file:line, what changes).
- Respect the user's rules; flag any suggestion that adds a veto or a threshold, or that is justified by only one task.
- Include a verification plan (offline first, then live) and open questions only where the user's decision is genuinely needed.
- List any fact in the brief that the investigators found wrong or imprecise under corrections_to_facts.

AUDIT:
${JSON.stringify(audit, null, 1)}

SEMANTICS:
${JSON.stringify(semantics, null, 1)}

DESIGN (general):
${JSON.stringify(designGeneral, null, 1)}

DESIGN (minimal):
${JSON.stringify(designMinimal, null, 1)}

CRITIQUE OF P0:
${JSON.stringify(critique, null, 1)}`

const judged = await agent(judgePrompt, { label: 'judge:synthesis', phase: 'Judge', schema: JUDGE_SCHEMA })

return { judged, audit, semantics, designGeneral, designMinimal, critique }
