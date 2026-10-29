# Final plan (DRAFT for verification): uc7 direction-semantics fix

All paths are relative to /workspace/gpt_field. "spatial.py" = src/gpt_field/agent_speedup_v2_user_check_spatial.py; "uc client" = src/gpt_field/agent_speedup_v2_user_check_client.py; "agent_client.py" = src/gpt_field/agent_speedup_v2_user_check_agent_client.py; "agent.py" = src/gpt_field/agent_speedup_v2_user_check_agent.py; "user_check.py" = src/gpt_field/agent_speedup_v2_user_check.py; "runner" = scripts/evaluate_agent_speedup_v2_user_check.py; "robocasa runner" = scripts/evaluate_robocasa_user_check.py; "html" = src/gpt_field/agent_speedup_v2_user_check.html; "base client" = src/gpt_field/agent_speedup_v2_client.py.

## Goal and scope
- Direction words (left/right/front/behind) are interpreted in the EXTERNAL camera's viewer frame, computed from calibration, frozen at review, and used identically by planner text, executive contract, region resolver, review UI and approval binding.
- Out of scope: t6 mug-handle grasp / elbow-stall recovery; uc8 speed items; v3 (frozen).
- Rules: one frozen build for all tasks; no task key reaches the agent; no new vetoes or tuned thresholds; acceptance = all-task sweep on one frozen build.

## Build A (one frozen build, ship together)

A1 Direction-frame helper (S1)
- New pure helpers in spatial.py near _polygon/_centre (:139-153); bump SPATIAL_VERSION (:23).
- Uses only the external camera calibration. right = unit(R[:2,0]); away = (-right_y, right_x) (= world-up x right). right_of=+right, left_of=-right, behind=+away, front_of=-away.
- Checks without tuned thresholds: values finite; R[:2,0] horizontal norm 0 -> unavailable; when the view is not vertical and away . (-R[:2,2]) < 0 (upside-down image) -> unavailable (reason view_frame_upside_down); top-down camera -> away = image-up (convention). Record roll = asin(R[2,0]) as evidence only.
- No image flip: R already describes the displayed upright, non-mirrored image (contact_skills.py:283 applies [:, ::-1] to undo the policy pipeline's [::-1, ::-1]).
- Documented scope: exact for any camera position / yaw / elevation; under roll the four words still classify correctly up to ~89 deg but the vector can be off by up to ~20 deg (roll 30-60 deg); an anchor-local planar projection is the later generalization.

A2 Planner text (S2)
- Replace spatial.py:59-60 with: "left_of/right_of/front_of/behind (and these words in the task) are as seen in image 1, the external camera: right/left = toward image 1's right/left; behind/front_of = farther from/nearer to that camera along the support. Not the robot's own left/right; never judged in wrist images. Use the same viewer words in plan text; never restate them as world axes. Vertical words mean world up/down. A word naming part of an object (back compartment, left plate) selects that part by pointing; it is not a region."
- spatial.py:60-63: frame "world" only means bounds are world metres relative to the anchor centre; keep bounds null for the four directional relations (the plan request carries no calibration).
- uc client METRIC_INSTRUCTIONS (:131-138, sentence at :134-135): REWRITE, do not delete: "Image left/right/up/down are not world axes; direction words are judged as seen in the external image (image 1) and the robot converts them to world coordinates with calibration. World +z is up." Self-contained, so it stays valid in plan, replan, repair and review (METRIC is appended to all four at :381-387). Keep "State the frame and units" for metric quantities; direction words use viewer words.
- robocasa runner :43-44: the plan-mode context says calibration is supplied with each image; the plan request carries none -> remove for plan mode.
- Bump PROMPT_VERSION (uc client :24).

A3 Executive contract (S3) - the change that alters uc7's t6 behaviour
- agent.py _task_contract (:115-126): add region_directions = {direction_semantics: "external_view_v1", frame: {camera, right_xy, away_xy, calibration_sha256}, regions: {id: {relation, anchor, direction_world_xy}}} from the frame frozen in A5. Beside regions, not inside, so PlanLock's semantic hash (uc client :235-321) is unchanged.
- AGENT_INSTRUCTIONS camera paragraph (agent_client.py ~:179-191): one sentence with the definition, plus "task_contract.region_directions is the execution basis and overrides direction wording in reviewed text; directions are never read from the wrist image".
- Stay under the request-size bounds (agent_client.py:483-494: compaction above 45000 chars, error above 56000). Bump agent PROMPT_VERSION (agent_client.py:41).

A4 Region resolver (S4)
- spatial.py:211-214 -> projection test: keep cells p with p.d > max over anchor-polygon vertices v of v.d, d from A5's frozen frame. 'on', bounds, object exclusion and centroid unchanged.
- No frame -> not-ready reason view_frame_unavailable; record frame and direction_world_xy in the evidence (result.update at :235).
- This IS a semantic change: the same right_of now selects what the old code called behind (intended). A6 tags the semantics.
- The macro path is dormant in uc7 (0 grants); this keeps _measure_goal and future macro use consistent.

A5 Freeze and bind to approval (S6, now must)
- runner prepare_review (:238-275) and robocasa runner prepare (:180-212): compute the A1 frame from observation.geometry's external camera; store preparation.view_frame = {direction_semantics, camera, pose_source, right_xy, away_xy, roll_deg, calibration_sha256 over R, position and intrinsics}. preparation is in user_check.py _CONTEXT_FIELDS (:35-39), hashed by _context (:236-237) into context_sha256 (:260) -> automatically bound to approval.
- Trial start: for fixed cameras (pose_source fixed_world_camera, LIBERO) recompute the live external calibration digest and refuse to start on mismatch, like the existing reset-identity check (integrity check, not a behaviour threshold). For robot-mounted cameras (RoboCasa) goal directions always come from the frozen frame; live calibration only interprets current images.
- Pass the frozen frame through approval/PlanLock to A3 and A4.
- Review UI shows each region's computed direction ("right_of C -> world +y (image-right)") and an arrow on image 1.

A6 Semantics tag and historical plans
- preparation.view_frame and region_directions carry direction_semantics "external_view_v1". Offline tools (validators, reports, replays) interpret plans without the tag as "world_axes_legacy"; never silently apply the new meaning.
- Old approvals cannot execute on new code (runtime hash check, user_check.py:396-397) - keep.
- validate_spatial_spec keeps accepting all recorded approved plans (regression test).
- v3: AgentSpeedUpV3PlanClient inherits UserCheckV2Client; do not rerun v3 on the new source without a snapshot.

A7 Review UI labels (S5): html :104-105 labels and :244 note use the external-view definition.

A8 False wrist-image text (S7 text part)
- At plan time the wrist image is not turned (runner :246 frame_alignment_turns(view.R, None) -> 0; review.json quarter_turns 0), but text says it is: base client :199 (plan instructions), :505 (image label), and :1097, :1159-1160 (other request kinds, only where the uc client uses them).
- Override in the uc client with the existing text-replacement mechanism (uc client :374-389) instead of editing the base v2 file. Per request, state the actual relation computed from calibration (e.g. "image 3's right is roughly image 1's left").

A9 Optional, display-only (S9): cell-order cue (region cell vs anchor cell on image 1) and world-axis-wording highlight in the review UI and trial log; never a plan issue.

## Validation (in order)

V0 Offline (no simulator, GPU or model calls)
1. A1 unit tests: recorded calibrations (relayout t6, t4_001, t5_001 -> right (0,1), away (-1,0); RoboCasa coffee -> (0.979,-0.202)/(0.202,0.979)); synthetic cameras (position/yaw/elevation/top-down exact; roll <= 89 deg words correct; upside-down -> unavailable). Prototypes: reports/uc7_direction_semantics_20261028/work/judge/axes_check.py and scratchpad rotation_check.py.
2. Pipeline consistency (guards against a double flip): project known object positions of the relayout layout through the calibration onto the displayed image; they must fall in the planner's pointed cells (pudding 35, mug 38, plate 61).
3. BDDL convention harness (developer test only): every anchored directional region agrees with its problem class's agentview frame (expect 41/41 decidable, 0 counterexamples; two LIBERO labelling errors listed as known).
4. Resolver on a synthetic heightfield with the t6 plate polygon: new right_of == old behind (intended change); RoboCasa direction; missing frame -> view_frame_unavailable.
5. Text snapshots: no right_of=+x / WORLD axes; rewritten METRIC present in all four kinds; AGENT_INSTRUCTIONS has the definition; html has no "(world +/-x/y)"; versions bumped.
6. Executive payload with a stub transport: relayout plan -> region_directions.regions.E.direction_world_xy == [0,1]; size < 45000.
7. Approval binding: editing preparation.view_frame breaks the review integrity check.
8. Regression: tests/test_agent_speedup_v2_user_check_*.py and tests/test_robocasa_user_check_*.py (skip simulator/GPU tests); every recorded approved_plan still validates.

V1 Plan-only (about 11 plan calls): all LIBERO-10 tasks + libero_goal_g5. t6 region E = right_of with viewer words and review shows world +y; g5 front_of the stove shows world +x; t3/t4/t5/t9 gain no regions; t4 left/right plate pointing unchanged.

V2 Targeted live (4 runs): t6 official layout, t6 relayout, g5, RoboCasa coffee smoke. t6: first pudding release on the plate's +y side (post-hoc truth); g5: first push target on the stove's +x side; RoboCasa: no breakage; log release/push offset from the anchor edge (input for E1). Judge direction criteria first; t6 may still fail on the mug grasp, g5 on pushing.

V3 Acceptance: full LIBERO-10 sweep (t0-t9 incl. t6 official and relayout) + g5 on the frozen Build A; no regression vs t4_001, t5_001/002, relayout t7.

## Separate experiments after Build A is accepted

E1 Placement preference (S8 / D9)
- Semantics unchanged: null bounds means "that side" (half-plane).
- Preference, ranking only (no mask band): aim_target prefers fitting free cells nearest the anchor edge and laterally aligned with the anchor centre; the same preference goes into region_directions for the executive ("with no distance given, place it at the nearest free spot just past the anchor's edge, aligned with the anchor across the direction").
- No new constants (existing M16 clearance, measured object radius); no veto.
- Compare Build A vs A+E1 on t6 (official + relayout) and g5: offset from anchor edge and in-box rate (post-hoc truth). Known tension: nearest-past-edge lands in 6/7 LIBERO side boxes, but g5's box starts 7.5 cm past the stove edge.
- Explicit adjacency not added; if ever needed, design relation-frame bounds (along/lateral or max_gap_m) and validate separately.

E2 Wrist image rotation (S7 image part)
- k from the current calibration every time (frame_alignment_turns with an external-camera adapter), never a suite constant.
- Keep in sync: images, grid numbering and w-cell back-transform (merge_plan quarter_turns), fingertip edge label (FINGERTIP_EDGES), and text "roughly aligned (nearest quarter turn)".
- Plan-only A/B first (11 tasks x k=0 vs calibrated k, ~22 plan calls): plan validity and wrist-cell pointing accuracy; live only after.

## Decisions for the user
1. D9 = "null bounds means that side only + placement preference" (E1).
2. Upside-down camera -> unavailable (recommended) rather than reading behind as image-up.
3. Ship Build A as a narrow build (e.g. uc7.1), not bundled with uc8 speed items, for clean attribution.
4. Include S9 (display-only) in Build A or not.

## Effort and risks
- Build A ~8-10 h (A1 0.5-1, A2 1, A3 1-1.5, A4 1, A5 2, A6 0.5-1, A7 0.25, A8 0.5, offline tests 1-2). V1 ~11 plan calls; V2 4 live runs; V3 ~11 live runs. E1 2-3 h; E2 1-2 h + ~22 plan calls.
- Risks: the plan text applies to every task, so only the sweep accepts it; the model may still write world-axis prose (region_directions overrides); t6's mug grasp and g5's pushing are separate gaps.
