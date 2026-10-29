# User constraints for hybrid experiments

- agent_SpeedUp_v1 DRAWER LIVE RUN 002 ANALYSED + FIXED (2026-10-17), user request
  "how do you think about the next improve" -> "in this order" (1 blocked bowl
  grasp, 2 speed, 3 multi-init measurement): runs/agent_speedup_v1_drawer_zero_shot_002
  (user stop, 4702 actions, 98 calls). Drawer opened (hold_pass, 14.1 cm pull);
  all 13 bowl descents stalled with the palm on the open drawer front 14 cm above
  the rim, then closed empty and retried toward the drawer. Fixes (prompt-5):
  blocked_above_rim (contact before the ladder setpoint cutoff -> no close, one
  2.5 cm outward retry), away-before-lift detail, orientation_restored no longer a
  failure, overhang-aware hover + pull-length prompt text; speed (prompt-4):
  more_waypoints_m, grasp_with+max_travel_m, handle release lifts+turns top-down.
  See agent_SpeedUp_v1.md §17.3. Validated on real LIBERO with the live model's recorded rim
  pointing: from live states 799/1316, one blocked attempt -> hold_pass -> (oracle
  carry) place -> LIBERO success with the bowl on the drawer floor; bowl smokes 020-023 identical to
  018/019. Held contact while rising backs off out from under the overhang. Pass --target-description for the bowl. The USER starts live runs.

- agent_SpeedUp_v1 DRAWER LIVE RUN 001 ANALYSED + FIXED (2026-10-17), user request
  "i stopped it. now apply your fix": runs/agent_speedup_v1_drawer_zero_shot_001
  (user stop, 867 actions, 61 calls, 12/12 handle grasps failed in Align). Causes
  (verified by 3 analysis lenses + 3 adversarial verifiers on bitwise sim replays):
  pointing model saw a 90-deg turned wrist view with "horizontal bar"/"UPRIGHT"
  cues and hit the top handle in 1/11 rounds; NCC cannot localize along a uniform
  bar; hovers off the bar length; from step 436 the arm sat at the elbow
  singularity (full stretch, unrecoverable). Fixes: gravity-turned pointing,
  bar orientation check + coarse re-ask, wide bar crop, Align axes="across_bar"
  (along position set once from the pointed pixel, held proprioceptively), bar
  descent holds, guards (align_blocked/approach_drift/hover_off_bar_across),
  finger_contact, neutral reach/orient texts, per-target loop warning, prompt-3.
  See agent_SpeedUp_v1.md §17.2. Validated on real LIBERO from live states
  335/361/393/555 (grasp + pull) and smokes drawer_011-016; bowl rim + place
  regressions pass. User then GRANTED ruling R-5 (robot base/shoulder position,
  2026-10-17): frontal full-stretch guard (wrist-to-shoulder 0.77 m refuse /
  0.785 m stop) added and validated (§17.2). The USER starts live runs.

- agent_SpeedUp_v1 LIVE RUN 001 ANALYSED + FIXED (2026-10-16), user request "do the
  full anlysis and fixes": runs/agent_speedup_v1_bowl_zero_shot_001 (user Ctrl+C,
  2835 actions, 108 calls). Root cause: model mapped EXTERNAL-image directions to
  world with the wrist axes (both signs inverted, median 153.5 deg), pushing past
  the top-down reach envelope (~0.79 m from base) where the hand pitched up to 61
  deg and the bowl was banged loose; transport only checked the hold after its
  move; 15 cm place bound too short; post-loss restore-in-place loop. Fixes in new
  modules only (policy-2, contact-skills-2, wrist-servo-2, prompt-2, new
  external_axes.py; runner finalization/setsid/stuck alert): see
  agent_SpeedUp_v1.md §17.1. ASSUMED ruling R-1x (external-image direction map
  from the robot's own probe motion; --external-axes off disables). Real-LIBERO
  scripted smokes 009-014 (zero model calls): grasp hold_pass, scripted
  carry+place reached LIBERO success, reach_limit + guard at the workspace edge.
  The USER starts live runs.

- agent_SpeedUp_v1 IMPLEMENTED (2026-10-16), user request "implement the
  agent_SpeedUp_v1.md ... give me the command and i will start the test myself";
  model config unchanged (gpt-6-astra medium, ChatGPT subscription). New files
  only: src/gpt_field/{agent_speedup_v1,agent_speedup_v1_client,contact_skills,
  wrist_servo,mark_prompting,ordered_transport,agent_speedup_v1_video}.py,
  scripts/evaluate_agent_speedup_v1.py (loads, never edits, the hybrid runner),
  tests/{agent_speedup_fakes,test_agent_speedup_v1_*,test_contact_skills,
  test_wrist_servo,test_mark_prompting}.py; 217 new tests pass; full suite
  5951 pass, 1 pre-existing failure (v36 provenance seal vs user's uncommitted
  src edits). Rulings ASSUMED per spec: R-1 online image Jacobian yes, R-2
  wrist-mount facts yes, R-3 evaluator-tuned constants no, R-4 no. Real-LIBERO
  scripted smokes (zero model calls) runs/agent_speedup_v1_sim_smoke_00*:
  bowl rim_pinch hold_pass (palm-rim contact recorded, record-only); drawer
  handle grasp hold_pass + 10 cm pull opened drawer + released_handle
  (runs/agent_speedup_v1_sim_smoke_drawer_004_grasp, scripted ORACLE hover). The USER starts
  live full tasks; no run launched by the agent. Commands: agent_SpeedUp_v1.md
  §17 and the session reply.

- Contact-region APPLY original-drawer full task is TERMINAL USER_STOP
  (2026-10-15): runs/contact_region_drawer_zero_shot_001. User explicitly chose
  "停止本轮并分析（推荐）"; stop requested16:19:43UTC, stopped at a complete
  feedback boundary.697 actions,41 valid Astra medium subscription responses,
  API-key0,rawLIBEROfalse. Control1271.130402s-video148.278941s=1122.851460s
  (18m42.85s) INCOMPLETE duration, not completion latency. Drawer opened/released;
  bowl remained on table. Handle26requests460actions; bowl15requests237actions,
  all12move+3recover, no bowlpregrasp or closure. One handlepregrasp request20
  omitted reference citing occluded tool/handle and unknown desired relation.
  All41compact pregrasp refsnull: epoch0/seeds0/tracked0/proposals0/applied0/
  rejections0/errors0. Five old body-ROI monitors are separate from new module.
  35Fieldresponses supplied no obstacle terms;96localplans0.160004s; model
  pipeline936.376493s=83.39%ofnet.633Field/35gripper/20ordinaryDirect/9verify.
  New correction was never exercised; do not attribute outcome to tracker
  rejection/6mmbound or claim benefit/speedup. Final RGB, video,operationhistory,
  independent audits and frozen sources retained. Runner3238882 exited, private
  supervisor/socket shut down; no retry/other task/queue authorized.

- Latest approval (2026-10-15), "run a test", authorizes ONE original-drawer
  zero-shot full task using contact-region APPLY and the existing Field planner.
  Run: runs/contact_region_drawer_zero_shot_001. Keep Astra medium subscription
  only, native512 RGB/state8, 1mm lazy field, prior robot/target guidance,
  task3/init0/seed7 and frozen initial observation. Raw LIBERO completion only;
  collisions record-only, no episode time/action/call cap. Explain changes
  before launch (done); do not ask redundant launch approval. Freeze runtime
  sources; no manual offline ROI or demo input. No automatic retry, other task,
  Direct or shadow trial queue. Preserve old results and report video-excluded
  time only as completion latency if raw LIBERO succeeds.

- Contact-region implementation/offline preparation is COMPLETE (2026-10-15).
  Evidence: runs/contact_region_preparation_001/implementation_validation.json.
  New contact_region_feedback.py/contact_region_policy.py and runner opt-in
  --contact-region-feedback apply|shadow retain existing Field and Direct5.
  344 offline tests passed; four side-effect-free task/mode preflights preserve
  subscription medium, raw LIBERO completion and no full-task caps. Per-task
  apply/shadow initial payloads are identical. New wrist-tool-only fixed-anchor
  fallback checks frozen visible edge/appearance per frame and measured robot
  geometry; explicitly conditional rigidity, not optical flow or contact proof.
  Same frozen eight historical windows: init8/8, tracking47/240 observations,
  validated13 observations across5 windows, two UNEXECUTED handle corrections
  1.28/1.49mm; bowl corrections0. Active tracking median15.62ms. Manual regions
  and initial-relative-offset objectives are offline-only, not GPT grounding or
  certified grasp goals. Development replay does not prove physical benefit or
  speedup. Earlier negative replays and pre-fix sources are preserved. No live
  model/simulator/full-task test ran; no trial/automatic queue is authorized.
  Explain final changes and obtain explicit approval before any full-task test.

- Latest user request (2026-10-15), "ok implement this", authorizes implementing
  the compact semantic-region contact feedback plan and offline validation.
  Preparation: runs/contact_region_preparation_001. Keep the v4 Field planner,
  Astra medium subscription protocol and RGB/state8 boundary. New opt-in
  --contact-region-feedback apply|shadow modes share model inputs; only apply
  revises Field goals, within total6mm of each accepted nominal goal. Direct5,
  normal-control fallback, no automatic probes/waits/closure/extra model calls.
  References are image regions and desired relations, never physical contacts;
  response models require later held-out validation and applied prediction checks.
  Preserve prior trials. Current work has no full-task launch authorization;
  full tasks remain subject to explicit user approval after explaining changes.
  Manual historical RGB fixtures are offline-only, not zero-shot inputs or proof
  of model grounding. Keep initial negative replays when implementation revisions
  are evaluated; do not change fixtures or call replay evidence physical benefit.

- Contact-routing diagnostic001 is TERMINAL COMPLETE (2026-10-15).
  runs/contact_routing_diagnostic_001 contains all eight SHORT trials from four
  frozen precontact states (two handle, two bowl), current-v4 versus Direct5
  contact correction.52 Astra medium subscription requests,741 actions,API-key0;
  all paired RGB/state8 and full physics/controller fingerprints match. All
  sources/inputs stayed frozen; terminal integrity and policy audits passed.
  No full task was launched; raw LIBERO completion was not the short endpoint.
  Strict sustained-engagement proxy0/8 within declared bounds DOES NOT mean
  zero physical progress: handle_b Direct5 opened15.9323mm with intermittent
  bilateral contact; bowl_a Direct5 lifted6.30mm and ended with five unsupported
  bilateral-pad samples. Neither met the frozen continuous10mm criterion.
  handle_a both arms had palm/handle contact without finger capture; bowl_b
  Direct5 had two empty closes. Precise pregrasp/placement references and local
  metric corrections were0;14bodyROI references produced70unknown/4contradicted/
  0supporting check observations. Only one contradicted outcome reached another
  model request before a trial cap. Fourteen close/check requests comprise11
  checks started,2unsettled skips,1action-interrupted request, not14 independent
  grasp attempts. Local action/request/attempt caps are diagnostic-only.
  Total net time1672.794611s includes incomplete trials and is NOT completion
  latency; model1495.823151s,planning0.027447s. No completion-speed ratio eligible.
  Positive historical bowl replay established the evaluator can detect a grasp;
  diagnostic states were deliberately selected, with fresh model histories.
  Main evidence: analysis.json; preparation/final_policy_audit.json,
  terminal_integrity_audit.json,evaluator_terminal_review.json; phase evidence
  under preparation/live_contact_frames/. Private supervisor shut down and no
  diagnostic processes remain (shutdown.json). Original tests and failed initial
  pre-inference startup are preserved. No retry/full-task/other queue authorized.

- Latest approval (2026-10-15): "ok do the step 1 and step 2 to identify the
  problem" authorizes a contact-routing ablation and paired SHORT diagnostics,
  not another full-task episode. Preparation/run root:
  runs/contact_routing_diagnostic_001. Freeze two drawer-handle and two original
  bowl pre-contact checkpoints before observing new results; replay identical
  action prefixes for both arms. Compare current v4 routing with explicit
  Direct five-action contact correction, retaining identical free-space Field,
  gripper settle, model-selected verification, RGB/state interface and history.
  Eight trials, alternating arm order; each ends at independent engagement or
  two completed close/check attempts / eight model requests / 100 actions.
  These finite diagnostic bounds do not change full-task LIBERO/no-cap rules.
  Astra medium, subscription only, zero-shot; no demo, depth/calibration or
  simulator truth in policy. Evaluator-only bilateral contact plus target lift
  or mechanism displacement measures LOCAL engagement, never task completion.
  Replay/setup/video excluded from control timing. Preserve old runs; no full
  task, automatic retry or other task queue is authorized.

- Operation Hybrid v4 DRAWER trial is TERMINAL USER_STOP (2026-10-15).
  User explicitly selected "停止本轮并分析（推荐）"; stop requested12:39:38UTC.
  Run: runs/operation_v4_drawer_zero_shot_001.668 complete actions,41 valid
  Astra medium subscription responses,API-key0,rawLIBEROfalse.385Field tracking,
  241localgripper,42explicitcloseverification; ordinaryDirect0. Six close/check
  operations ran without intervening model calls, but drawer did not open and
  bowl stage was never reached.14pregrasp intents,all precise refsnull/corrections0;
  six generic tracking references supplied;42verification outcomesunknown.
  No extended Field requests;65plans/no obstacle terms. Net excludingvideo
  1068.449645s (17m48.45s) is INCOMPLETE duration, notcompletiontime.
  Model924.702809s(86.55%net),planning0.063322s,visual2.426430s.
  333prelaunchtests,27live/26terminalauditchecks passed.280runtime sources/five
  helpers frozen throughterminal; exact runtime_sources.tar.gz archived. Runner
  2834050 exited; private supervisor shut down12:40:55UTC; video saved.
  Post-run ONLY, fixed operation_visual's axis-angle pi-wrap diagnostic using
  quaternion geodesic angle;96focused tests passed. The old guard caused12false
  rejection reasons, but those samples still lacked fixed-view paired evidence,
  and no verification actions were cancelled. Do not attribute task failure or
  hypothetical speedup to this diagnostic bug. No retry/Direct/waypoint live
  comparison/other task queue. Further full tasks require fresh user approval
  after explaining changes. Preserve archives; post-run patch is not live-tested.

- Current approval: "implement this and do the full tast test" authorizes the
  operation-level redesign described in the preceding plan and ONE fresh original
  drawer zero-shot full-task Hybrid test after offline checks and a prelaunch
  explanation. Preparation: runs/operation_v4_preparation_001; trial root:
  runs/operation_v4_drawer_zero_shot_001. Introduce explicit generic operation
  contracts, dated task state, local visual outcomes, bounded feedback-driven
  continuation and model-requested verification motion; include field/waypoint
  ablation support. Preserve legacy default-off behavior and all old archives.
  Astra medium subscription-only; two native512 RGB + state8, no depth, scene
  calibration, demo or object truth. Raw LIBERO success only, collisions recorded,
  no episode cap. Explain changes before this authorized test; do not ask a second
  launch approval. No automatic retry, Direct comparison or other task queue.

- Pregrasp v2 DRAWER test is TERMINAL USER_STOP (2026-10-15).
  User explicitly selected stop/analyze. Run:
  runs/pregrasp_feedback_v2_drawer_zero_shot_001. 836 complete actions,
  61 accepted Astra medium subscription responses; API-key calls0. Raw LIBERO
  false, completion_time_eligible=false.531Field/190localgripper/115ordinaryDirect.
  Control1180.670765s minus video145.641948s=1035.028817s (17m15.03s);
  this is incomplete elapsed time, NOT completion latency. Model873.786004s
  (84.42%net),84plans0.104902s. ALL61pregrasp_reference=null,0align phases,
  0seeds/tracking/fits/corrections/errors. Model-described rim alignment was
  phase=approach; competing old held-object/new pregrasp align prompt semantics
  are a possible cause, not proven. Do not claim tracker rejection or benefit.
  Drawer opened, first bowl grasp missed, second grasp/lifts carried bowl;
  drawer had retracted, so bowl was staged and drawer reopened. Final836RGB:
  drawer open, bowl on table, beginning another approach; task incomplete.
  228offline tests and dual705action/59response compatibility replays passed;
  savedRGB4/4initialization but only4tracked transitions,0validatedfits/proposals.
  Prelaunch30/live17/terminal20checks passed;277runtime/fivehelpers unchanged.
  Original RGB evidence, analysis.json, trial/summary.json, episode.mp4 and
  shutdown.json saved. Runner2579214 exited and private supervisor/socket closed.
  No retry/other task/Direct queue. Any further full-task test requires fresh
  explicit user approval after changes explained. Preserve all run archives.

- Latest user approval: "we just make a full revision, and test on the full
  task" authorizes the complete pregrasp feedback revision AND ONE fresh
  original drawer zero-shot full-task test. Implement independent view support,
  observed wrist tool reference, nearby same-body support and adaptive small
  texture tracking. Keep current Field planner, <=6mm TOTAL nominal correction,
  natural-motion learning and normal fallback; no probe/hold/close gates.
  Explain final changes BEFORE starting this already-approved test; do not ask
  for redundant launch permission. New run:
  runs/pregrasp_feedback_v2_drawer_zero_shot_001. Retain Astra medium subscription
  only, native512 RGB+state8,1mm grid, Field30/plan10/Direct5, prior geometry,
  target identity and attempt-memory guidance, task3/init0/seed7. Raw LIBERO
  completion only, collision record-only, no episode cap. Preserve old archives;
  freeze sources and helpers before launch. No extra retry, task or Direct queue.
  Offline preparation: runs/pregrasp_feedback_v2_preparation_001.

- Pregrasp implementation and OFFLINE validation are complete. Artifacts:
  runs/pregrasp_feedback_offline_001/analysis.json. Default-off
  --pregrasp-feedback adds two-view visible-material tracking, measured local
  XYZ response and <=6mm TOTAL nominal Field-goal revision within4cm. No
  probes/automaticclose/action gate, no extra dedicated GPT calls. Existing
  Direct5/Field30, subscriptionmedium, sensors, LIBERO/collision/timing rules
  unchanged.210distinct tests passed (209combined, then24integration including
  one added test). Both disabled and enabled-null saved705action/59response
  replays passed; archive unchanged. IMPORTANT: realRGB0/4 eligible contact
  seeds,0realcorrections: three smallpads, fourthoccluded. Synthetic RGB-to-Field
  success and44ms mediantracking do NOT demonstrate realgraspimprovement or
  taskspeedup. Addedprompt/schema/preflight are review artifacts, NOT a sealed
  futuretrial. No live model, real simulator, new fulltask, retry or queue ran.
  Future fulltask still requires explicit userapproval after changes explained.

- Latest approval: "ok, implement this" authorizes the pregrasp local visual
  feedback extension and offline validation ONLY. No full task, simulator
  rollout or live model call is authorized. Keep current working v3+memory
  baseline default-off compatible; add opt-in --pregrasp-feedback using two
  uncalibrated RGB views/measured robot state, fixed-orientation local XYZ
  response and bounded Field retargeting. No depth, camera/environment data,
  hidden object truth, automatic probe/close or action admission gate. Failed
  tracking falls back to normal control. Preserve all existing run archives.
  Validate synthetic behavior and saved705action/59response replay; manual
  historical RGB annotations are offline diagnostics only, never live inputs.
  Explain changes/results before seeking approval for any future full test.

- Attempt-memory DRAWER retry002 is TERMINAL USER_STOP (2026-10-15).
  User explicitly chose stop/analyze. Run: runs/attempt_memory_drawer_zero_shot_002.
  705completeactions/59accepted Astra medium subscription responses/API-key0;
  300Field/260localgripper/145ordinaryDirect. Raw LIBERO false. IMPORTANT:
  third handle closure248 led to visually open drawer by283, maintained open.
  Firstthree bowl closures425/485/535 missed; changed jaw orientation580,
  fourth close629–643 plus lift644–648 led to Field carry649onward. Final705
  RGB supports bowl held beside open drawer, not placed/released inside.
  Do NOT report nevergrasped, or eventual task failure; stopped during transport.
  Control1141.452810s minus video115.549888s=1025.902922s(17m05.90s),
  incomplete duration notcompletiontime. Model893.837220s=87.13%net;
  planning0.070254s;58analyticplans/0obstacles/0visualrefs/0localcorrections.
  All59requests had currentmemory/review; chronology checks passed. Real
  position/rotationchanges occurred but no causal memory or speed benefit
  established. Runtime276sources/fivehelpers exactly001 and unchanged.
  Root24/prelaunch32/live39/terminal23checks passed; fullsavedactualreview
  replay705actions/59responses passed with0extra model/simulatorcalls.
  112offline tests/old dualreplay reused with explicitprovenance, notrerun.
  Stopfile08:54:26UTC;runner2312371 exited andprivate supervisor/socket/pidfile
  closed08:55:19UTC. analysis/video/attempt evidence/audit/shutdown saved.
  No restart/other task/Direct queue. Future fulltasktest requires fresh
  userapproval after changes explained; preserve archived inputs/results.

- Latest user approval (2026-10-15): "retry" authorizes ONE fresh reset
  original drawer zero-shot attempt-memory retry002 after trial001 network
  interruption. Run: runs/attempt_memory_drawer_zero_shot_002. No method,
  prompt, schema, model, controller, sensor or stopping-rule change from001.
  Same276runtime sources; reuse112passed tests and dual1584action/128response
  saved replay with explicit provenance. Astra medium subscription-only;
  task3/init0/seed7,native512/state8,1mm,Field30/plan10/Direct5. Raw LIBERO
  completion only,collisionsrecordonly,noepisodecap. No API-key fallback.
  Seal before launch; freeze runtime/helpers through terminal audit. This
  retry has approval; no further retry,other task or Direct queue authorized.

- Attempt-memory DRAWER trial001 is TERMINAL SUBSCRIPTION_ERROR (2026-10-15).
  Run: runs/attempt_memory_drawer_zero_shot_001. Request12 returned
  subscription_network_error with no response after a saved180.199s interval;
  exact underlying network/timeout cause is not retained. No overload claim.
  220 complete actions:215Field/5ordinaryDirect/0localgripper;11accepted
  medium subscription responses;API-key0. Raw LIBERO false, drawer still
  closed in final RGB, no explicit grasp closure. This is an infrastructure
  interrupted incomplete attempt, not a completed policy evaluation.
  Control442.087461s minus video37.671955s=404.415507s(6m44.42s), INCLUDING
  final failed request wait; accepted model-pipeline179.766287s excludes it.
  No completion-speed or grasp-recovery benefit can be inferred. Memory and
  reviews active in saved requests;12causal memories checked without mismatch.
  26plans all analytic,visual_reference allnull,obstacleterms0,localupdates0.
  First evaluator contact26finger/wine-bottle,record-only as approved.
  112 offline tests passed; old1584action/128response dual regression passed
  (enabled old replay uses synthetic neutral reviews, not live evidence).
  Terminal integrity23/23 and full actual-review replay220actions/11responses
  passed with0extra model/simulator calls;276runtime/fivehelpers unchanged.
  Runner2173689 exited; private supervisor/socket/pidfile closed08:13:21UTC.
  analysis.json,diagnostics,video,shutdown and terminal monitoring saved.
  No retry/other-task/Direct queue started. Every future full-task test needs
  fresh user approval after explaining changes; preserve these run artifacts.

- Latest user approval (2026-10-15): "ok, implement this and then start the
  full task test" authorizes attempt/outcome memory implementation and ONE
  original drawer zero-shot trial: runs/attempt_memory_drawer_zero_shot_001.
  Explain final changes and offline validation before launching this approved
  run; no second approval is needed for this one. Retain geometry + target
  identity guidance, Astra medium subscription-only, task3/init0/seed7, native
  512 RGB/state8, 1mm grid, Field30/plan10/Direct5/local-gripper unchanged.
  New capability is compact attempt-level memory from accepted observations,
  robot feedback and explicitly attributed model assessments; same-call short
  outcome/next-change/prerequisite review, no new inference call or action gate.
  Preserve continuation closures, uncertain outcomes and dated prerequisites;
  no depth, scene calibration, object truth, demos, automatic test lift or new
  tracker. Validate offline and freeze runtime through terminal audit.
  Raw LIBERO-only completion, collisions record-only, no episode time/step cap;
  subtract measured video only. No retry/Direct/other-task queue authorized.

- Target-identity DRAWER trial001 is TERMINAL USER_STOP (2026-10-15).
  Run: runs/target_identity_drawer_zero_shot_001. User explicitly chose stop
  and analysis; runner 1952807 exited and private supervisor/socket closed.
  Raw LIBERO false after 1,584 complete actions and 128 Astra medium
  subscription requests; API-key calls 0. Control 1,941.785636 s minus video
  258.874864 s = 1,682.910773 s (28m02.91s). This is incomplete attempt time,
  NOT task completion latency. 546 Field / 727 local-gripper / 311 ordinary
  Direct actions. Prompt-only target-identity-v1 retained correct patterned
  bowl in inspected RGB/explanations, but stable grasp remained unresolved.
  Drawer visibly opened at 232, retracted substantially by 376. Brief actual
  bowl lift/transport at 1054-1082, then drop reported 1083 and visible 1112.
  26 bowl CLOSE operations include continuations, not 26 independent attempts.
  33 Field blocks: 24 goal reached / 6 no progress / 2 budget / 1 invalid.
  85 plan records: 84 analytic, one zero-segment already-within-tolerance
  return at 180 labelled lazy_fine_grid_astar, with no obstacle graph search.
  All 128 visual references null, local corrections 0, obstacle terms 0.
  Model pipeline 1,400.079142 s (83.19% net); planning 0.094280 s. Collision
  at action 43 recorded only. Do not claim success, speedup, or causal prompt
  benefit from this single incomplete trial. Both prior/current drawer runs
  were stopped at different times, so completion-speed comparison unavailable.
  69 offline tests and old 730-action/56-response replay passed before launch.
  Terminal integrity 22/22 passed; full saved-only replay 1,584 actions/128
  responses passed with zero additional model/simulator calls. 275 sources,
  five launch helpers and frozen inputs unchanged. Initial audit shutdown
  metadata race retained in history, resolved by verified shutdown/rerun.
  analysis.json, terminal diagnostics, raw RGB evidence, video and shutdown
  saved. No further implementation, repeat, Direct or other-task run started;
  every next full-task test needs fresh approval after explaining changes.

- Latest user approval: "ok implement this and test it on the drawer task"
  authorizes a prompt-only target identity improvement and ONE new original
  drawer zero-shot trial: runs/target_identity_drawer_zero_shot_001.
  Keep geometry-guidance v3, medium subscription-only, task3/init0/seed7,
  native512/state8,1mm grid, Field30/plan10/Direct5/localgripper unchanged.
  Add opt-in human RGB appearance description + generic object identity rules
  to existing requests; no fixed coordinates, hidden scene data, demonstration,
  extra inference, cropped reference, memory/history or controller changes.
  Description approved in discussion: separate round bowl, visible rim/concave
  opening/light grey patterned interior, distinguished from stove/heating ring/
  dark appliance components. Validate offline and state changes before starting
  the already-approved one trial. Preserve previous profiles and artifacts.
  LIBERO-only completion, record-only collisions, no episode cap, video-excluded
  timing. No Direct/other task/retry queue or follow-on full test authorized.

- Stage1 DRAWER trial001 is TERMINAL USER_STOP (2026-10-15). User explicitly
  selected stop/analyze after opened drawer but repeated missed bowl attempts.
  Run: runs/robot_geometry_stage1_drawer_zero_shot_001. Raw LIBERO false,
  730completeactions/56Astra medium subscriptionrequests/API-key0;410Field,
  184localgripper,136ordinaryDirect. Control811.685393s minusvideo111.238581s
  =700.446813s(11m40.45s), incomplete attemptduration, NOTcompletionlatency.
  One15actionhandleclose143–157,20pullactions158–177,open178–197; drawer
  visuallyopened, bowlstagebegan198. Sixbowl-targetcloseoperations include
  continuationblocks; no transport/placementphase. RecordedRGB/model439support
  confusion of darkburner-adjacentcomponent with actualpatternedbowl visible
  besidehand. Attachment/displacementnotverified; no hiddenstatepolicyinputs.
  Modelpipeline579.276916s(82.70%net),planning0.059020s;56plansallanalytic,
  all56visualrefsnull/localupdates0/obstacleterms0.20Fieldblocks13arrived,
  2stalled,5budget. Firstrecordedcontact45hand-vs-cabinettop,record-only.
  Existingblack_bowllabel matchesofficialakita_black_bowlclass; cannotattribute
  confusioncausallyto thelabel. Same274runtime asstage1bowl, unchangedinputs.
  Root17/prelaunch29/live36/terminal23checks passed; fullsavedpolicyreplay
  730actions/56responsespassed, realmodel/sim0. LastDirectchunkstoppedafter1/5
  atcompletefeedbackboundary;runner1886185/private supervisorclosed06:26:13UTC.
  Analysis/video/RGBevidence/audit/shutdownsaved. No furthertrial or method
  change implemented; discuss targetidentity/failed-attemptmemory next.
  Each futurefulltaskteststillneedsfreshapproval afterexplainingchanges.

- Approved stage1 DRAWER trial001 IS LIVE as of 2026-10-15T06:12:01UTC,
  runner1886185 under its private supervisor. Run:
  runs/robot_geometry_stage1_drawer_zero_shot_001. Root17seal and independent
  29prelaunch checks passed;274sources identical to completedstage1bowl,
  5helpers/official drawer inputs sealed. Initial reset validation passed.
  Do not edit runtime until terminal audit. This is one authorized drawer
  episode only, with no automatic retries or other-task queue.

- Latest user approval (2026-10-15): "try the drawer task" authorizes ONE fresh
  original-drawer zero-shot stage1 geometry-guidance v3 test. Directory:
  runs/robot_geometry_stage1_drawer_zero_shot_001. Same runtime as completed
  stage1 bowl; same drawer task/init as prior v3 drawer: drawer_bowl,
  libero_goal/task3/init0/seed7, Astra medium subscription-only, native512/state8,
  1mm lazy field, Field30/plan10/ordinaryDirect5/localgripper unchanged.
  Geometry guidance remains the only policy change from prior drawer v3.
  No depth/calibration/objecttruth/evaluator feedback in policy. LIBERO-only
  completion, collisions record-only, no episode cap, subtract measured video.
  Freeze runtime/inputs before launch. No retry/Direct/other scene/later stage
  is authorized by this request. Preserve historical trial artifacts.

- Stage 1 original-bowl trial001 is TERMINAL SUCCESS (2026-10-15), superseding
  the historical live/awaiting-approval entries below. Run:
  runs/robot_geometry_stage1_bowl_zero_shot_001. Raw LIBERO true after 521 actions
  (283 Field / 163 local gripper / 75 ordinary Direct), 39 Astra medium
  subscription requests, API-key calls 0. Control 548.181256 s minus measured
  video 80.203822 s = 467.977433 s (7m47.98s). Previous v3: 269.728379 s,
  254 actions / 24 requests; current is 198.249055 s (73.50%) longer.
  Seven explicit CLOSE operations; these are not seven independent attempts.
  First model-selected transport action428 vs212 previously. One monitored
  finger/ramekin collision at action81; record-only under approved protocol.
  Final phase align, gripper +1: raw LIBERO success, no post-release validation.
  All 39 visual_reference=null, local corrections0; all32plans analytic with
  no obstacle terms. No evidence of speed benefit, local visual correction
  benefit, or 1mm obstacle-planning benefit from this single episode.
  Root launch17 / independent prelaunch26 / live33 / terminal22 checks passed.
  Full saved-only policy replay521actions/39responses passed, extra model/sim0;
  274sources/5helpers and frozen inputs unchanged. Private runner1815744 exited,
  supervisor closed05:41:41UTC; analysis/audit/shutdown and video saved.
  No repeat, drawer, Direct or later-stage full-task test has approval; each
  needs fresh user approval after explaining changes. Do not launch next test.

- Stage1 original-bowl trial001 IS LIVE, launched2026-10-15T05:31:09UTC,
  private supervisor runner1815744. Run: robot_geometry_stage1_bowl_zero_shot_001.
  Root17seal checks and independent26prelaunch checks passed.274runtime sources,
  5helpers sealed;233prep003 sources match. No runtime edits until terminal audit.
  Initial reset passed identical-observation check; now inference/execution.
  User authorized this one round only, no new task/repeat/Direct/later stage.

- Latest user approval (2026-10-15): "ok then test this version" authorizes ONE
  fresh original-bowl zero-shot stage1 geometry-guidance v3 test, following the
  explained review_plan from robot_geometry_stage1_preparation_003. New run:
  runs/robot_geometry_stage1_bowl_zero_shot_001. Use --speed-first-v3 plus
  --robot-geometry-guidance; otherwise identical medium subscription-only,
  libero_spatial/task0/init0/seed7, native512/state8, 1mm grid, Field30/plan10,
  Direct5/localgripper and LIBERO-only completion. Collisions record-only,
  no episodecap, video-excluded timing. Freeze runtime through terminal audit.
  Do not queue/restart another bowl, drawer, Direct or later-stage test; each
  requires a separate future approval. Preserve all historical artifacts.

- Stage 1 implementation/offline preparation COMPLETE, awaiting user approval
  (2026-10-15). New --robot-geometry-guidance is default off, v3-only. It appends
  audited Panda grip_site/right_hand/pad-axis definitions; prompt version suffix
  +panda-grasp-frame-v1 and metadata/hash are logged. No control, history, visual
  feedback, grid, schema or gripper behavior change. Files: robot_geometry_guidance.py,
  speed_first_v3.py, evaluate_bowl_hybrid_subscription.py; offline helper
  scripts/prepare_robot_geometry_stage.py has NO live mode. Final review artifacts:
  runs/robot_geometry_stage1_preparation_003/{review_plan.json,added_prompt.txt,
  validation.json,offline_tests.json}; per-task prompt_after.txt/preflight.json.
  144 unique offline tests passed; saved bowl254actions/24replies and drawer703/
  66 replay with matching actions/selected diagnostics/feedback. Initial payloads
  exactly old except appended instructions. Later full request payloads were NOT
  compared byte-for-byte. Historical artifacts unchanged; real model/sim calls0.
  Preparation001 aborted before replay (missing initial local-feedback in preview
  builder);002 replay passed but proposal suite typo corrected in003, which
  derives and checks official suite. Proposed first live test: original bowl,
  libero_spatial/task0/init0/seed7, same v3 medium subscription zero-shot protocol.
  This is only a proposal: no approval, process, queue or task result exists.
  Each full-task run, including later drawer/repeats, needs separate approval.
  Stage 2 memory and stage 3 general interaction feedback remain unimplemented.

- Latest user instruction (2026-10-15): implement the Astranomy-inspired plan
  STAGE BY STAGE. EVERY full-task test requires fresh user approval before
  starting, after explaining the specific implemented changes. Implementation,
  offline unit/fixture checks, saved-data replay and prompt/preflight preparation
  are authorized now; no live model or simulator episode is currently approved.
  Stage 1 is isolated known-robot geometry/frame prompt guidance on current v3,
  default off. Do not bundle history, local alignment or controller changes.
  Later stages are attempt/outcome memory, then general interaction feedback;
  review each stage's result before proceeding. Keep medium subscription-only,
  zero-shot, RGB+allowed robot state, no depth/env calibration/object truth,
  LIBERO-only completion, collision record-only and video-excluded timing.
  Preserve old profiles and historical runs. Do not interpret prior "test it"
  entries below as authorization to launch any new round.

- Color heatmap request COMPLETED (2026-10-15). V3 video recorder now
  displays three161x161 reconstructed analytic distance slices for current
  uniform box-free Field plans, teal-low/orange-high with meter colorbar;
  weighted/masked/missing plans honestly retain path-only view. No policy or
  planner changes. Only speed_first_v3_video.py differs from the271source
  pretest seal; historical frozen_source/audits remain original. New script:
  scripts/render_speed_first_v3_video.py (offline-only, no policy/model/sim).
  Drawer output: runs/speed_first_v3_drawer_zero_shot_001/video_color/
  episode_color.mp4 + episode_color.events.jsonl + preview_field.png +
  provenance.json. All703actions/66savedreplies,20fps/4228frames/211.4s retained;
  originaleventmetadata/latencies/results and7originalartifacthashes unchanged.
  Focusedrenderer tests3passed; no newinference/simulator/trial. Original
  netattempttime remains679.810599s. No further live trial authorized.

- Speed-first v3 ORIGINAL DRAWER zero-shot trial001 is TERMINAL USER_STOP
  (2026-10-15), explicitly authorized by user after repeated missed handle
  grasps. runs/speed_first_v3_drawer_zero_shot_001/trial/summary.json:
  703 complete actions / 66 medium subscription requests, raw LIBERO false,
  332 Field / 201 local-gripper / 170 ordinary Direct. Control789.514075s
  minus video109.703476s =679.810599s (11m19.81s), attempt duration only,
  not completion latency. Private runner1693204 exited; supervisor closed
  04:40:04UTC, shutdown.json saved. No queue/restart/new trial authorized.
  Terminal audit23/23 and recorded-only replay703actions/66responses passed;
  271sources/4helpers/input seals unchanged, extra model/sim calls0. See
  analysis.json and terminal_rollout_diagnostic.json. Six complete20-step
  closures missed handle; seventh interruptedafter1action. All66decisions
  targethandle; no bowlstage. Field21blocks:14reachedgoal3.31-4.89mm,5stall,
  2budget; all68plansanalytic, obstacleterms0, visualrefs/corrections0. Model
  pipeline558.730697s (82.19%net), planning0.066247s. Current localvisual
  branch covers held-object planarplacement, not gripper-to-handle grasp.
  No next implementation or trial authorized; current task complete.

- Latest user authorizes one fresh drawer task using the SAME current v3 test
  protocol: runs/speed_first_v3_drawer_zero_shot_001, drawer_bowl/LIBERO-Goal
  task3/init0/seed7, zero-shot, gpt-6-astra medium subscription OAuth only.
  Preserve the271runtime sources from the completedbowlv3; native512pairedRGB,
  lazy161^3/.001m/16cm grid, optionalheld-objectlocalfeedback, Field30/plan10,
  ordinaryDirect5 and explicitlocalgripper operations unchanged. Task-specific
  prompt/schema useexisting drawerconfiguration. RawLIBERO successonly,
  collisionsrecord-only, noepisodecap/restart/other-scene/Directqueue. Exclude
  measuredvideooverhead butincludealllocalvision/planning inreportedtime.
  Prepare/freezeallowedinitialRGB+state andofficialtaskinputs beforelaunch.
  No depth/calibration/objecttruth/evaluatorsignals inpolicy. Existingbowlrun
  andhistoricaldrawerresults remainimmutable.

- Speed-first v3 trial001 is TERMINAL SUCCESS (2026-10-15). See
  runs/speed_first_v3_bowl_zero_shot_001/analysis.json. RawLIBEROtrue after254
  actions:109Field/85explicitlocalgripper/60unchangedDirect,24Astra medium
  subscriptioncalls/API-key0. Control308.533742s minusvideo38.805363s =269.728379s
  (4m29.73s). No monitored forbiddencontact. Finalgripper+1; noextra release or
  postrelease stabilitytest. 271sources/4helpers unchanged,32live/22terminal
  auditchecks passed; full offlinepolicyreplay254actions/24recordedresponses
  passes numeric tolerances1e-12absolute/1e-11relative, excludingwalltime, with
  zero realmodel/simcalls. Direct60/localgripper85 serializedactions exact.
  Runner1658614/private supervisor closed04:12:58UTC, shutdownsaved. Noqueue.
  Criticallimitation: all24responses visual_reference=null, epoch/fit/updates0.
  Alignment included15Field+10Directactions, but optionalvisualfeedback never
  activated. All13plans clearuniformanalytic,total.015958s;actual1mmgrid
  configuredlazy161^3, but no obstacle/weightedlatticesearch inthisliveepisode.
  Thus neitherlocalfeedbackbenefit nor1mmresolutionbenefit established. Prior
  v2success344.946861s/current21.81%less;historicalDirect415.420830s/current35.07%
  less;earlyHybrid257.384130s remains12.34squicker(newtakes4.80%longer). Historicalsingleepisode
  references, notfreshmatchedpair/componentablation/successrate. Preserveall
  priorincompleteattempts. No furthertrial or implementation queued.

- Speed-first v3 original-bowl zero-shot trial001 IS LIVE, launched2026-10-15
  04:07:00UTC under private supervisor, runner1658614. Directory:
  runs/speed_first_v3_bowl_zero_shot_001.271runtime sources/4helpers sealed;
  no runtime changes until terminal audit.19seal/24independentprelaunch checks
  passed. Native512 initial state matches priorv2;161^3/.001m/16cm lazyfield,
  analytic clear paths else actualfine26neighborA*. Optional held-object RGB
  localfeedback only nearplanaralignment; no global admission/probing guards.
  Direct5 unchanged; localgripperoperations retained. Medium Astra subscription
  only, LIBERO-only completion, record-only collisions, no episodecap/restart
  or other-scene/Directqueue. Videoexcludedtime includesalllocalvision/planning.
  Track whether visualreferences/fit/corrections actually activate before
  attributing outcomes. Combined variant, not isolated resolution ablation.

- Latest authorization: implement/test native high-resolution RGB plus narrow
  local object-relative correction, with 1 mm field spacing over the same 16 cm
  cube (161^3). New opt-in Speed-first v3; preserve prior profiles/artifacts.
  One fresh original-bowl zero-shot trial only, task0/init0/seed7, Astra medium
  ChatGPT subscription only, raw LIBERO completion, collisions record-only,
  no episode cap, video-excluded timing including local vision/planning costs.
  No depth, supplied camera calibration, hidden object poses or evaluator inputs.
  Fine grid is evaluated lazily, not expanded into millions of JSON values.
  Native render/input512; local feedback is optional near planar alignment,
  not a global Field admission gate or automatic probing/recovery controller.
  Freeze tested runtime and inputs before launch. No Direct/other scene queue.
  This combined change is not an isolated resolution or feedback ablation.

- Speed-first v2 unchanged repeat004 is TERMINAL USER_STOP (2026-10-15).
  User explicitly chose stop/analyze after six placement-release attempts.
  runs/speed_first_v2_bowl_zero_shot_004/analysis.json:695complete transitions,
  244Field/285localgripper/166ordinaryDirect,71validmediumsubscription replies,
  API-key0,15operations,LIBERO false. FinalDirect block stopped after1of5 at a
  complete-feedback boundary; no service error or episode timeout. Control
  774.590560s minus109.145394s video =665.445167s (11m05.45s), attempt time only.
  Model608.810513s (91.49%net),planning6.625285s. First collision57hand-vs-bowl
  record-only. Runner1571457/private supervisor closed;shutdown saved.267sources,
  4helpers,226frozen files unchanged;launch19/terminal26checks passed, including
  166Direct actions exact,285localop action/diagnostic replay and71request inputs.
  No new policy trial/Direct/scene queued. Post-terminal recorded-action evaluator
  replay VALIDATED695actions/696state+RGBpairs exactly,maxstate/pixelerror0,
  zero model calls, simulator closed. Goal misses XYdistance<3cm: six release
  endpoints7.119/6.842/5.725/3.662/3.414/3.757cm, allcontact/height true. Minimum
  XY3.057625cm(step468, contactfalse);final3.757469cm. Diagnostic groundtruth
  never entered policy; no new scored episode.18movingField blocks reached their
  EEgoals within2.64–4.94mm. Both004/002 used16requests before firstrelease;
  fromfirstrelease onward55vs18.629–633 measured+Y1.43cm reversed634–638−1.48cm.
  Centering improved but never passed tolerance and last release worsened;
  don't claim zero progress, impossible completion, or planner execution failure.

- Latest user request "test it again" authorizes one fresh unchanged Speed-first
  v2 original-bowl zero-shot repeat004 after003's HTTP503 interruption. Use
  runs/speed_first_v2_bowl_zero_shot_004, same267sources/prompt/task0/init0/seed7,
  Astra medium subscription only, raw LIBERO completion, collisions record-only,
  no episode cap, video-excluded time. Freeze runtime during execution; preserve
  all earlier attempts. No Direct/other-scene queue or automatic episode restart.

- Speed-first v2 unchanged repeat003 is TERMINAL INCOMPLETE (2026-10-15).
  runs/speed_first_v2_bowl_zero_shot_003/analysis.json: subscription_error,
  HTTP503 unknown_error on request54;53valid predictions,745complete actions,
  255Field/405local gripper/85ordinary Direct,16operations,LIBERO false.
  Control640.066918s minus106.989519s video =533.077400s (8m53.08s), attempt
  duration only. Model returned-prediction pipeline470.613169s;planning7.422705s.
  First release/retract191-220 did not satisfy LIBERO; repeated regrasp followed.
  Long consecutive closures412-496(85actions),522-639(118actions) did not establish
  attachment. First collision122 hand-vs-target bowl is record-only. Runtime and
  initial full request exactly002; preserve sampling variability and distinguish
  service interruption from preceding policy difficulty. Runner1541262 exited,
  private supervisor closed; no automatic retry/new scene/Direct queued.
  Launch audit19checks and terminal audit27checks passed. All745transitionchains,
  85ordinaryDirect actions exact,405localgripper actions/diagnostics replayexact,
  all54request boundaries and267source/4helper/226frozen hashes verified.
  Full Field replay was not performed. Failed54 had no response or action.

- Latest user authorization: "test it again" authorizes one fresh unchanged
  Speed-first v2 original-bowl zero-shot repeat003. Preserve successful002.
  Same task0/init0/seed7, Astra medium subscription only, raw LIBERO completion,
  record-only collisions, no episode cap, and video-excluded timing. Verify and
  freeze identical runtime/prompt/input hashes; no Direct/other-scene queue or
  automatic episode restart. Directory: runs/speed_first_v2_bowl_zero_shot_003.

- Speed-first v2 original-bowl zero-shot test002 is TERMINAL SUCCESS (2026-10-15).
  runs/speed_first_v2_bowl_zero_shot_002/analysis.json: raw LIBERO true after390
  complete actions (159Field,161explicit local gripper,70ordinary Direct),34
  Astra medium subscription calls/API-key0,7local operations. Control408.428780s
  minus63.481919s measured video =344.946861s (5m44.95s). Historical Direct net
  415.420830s/46calls:16.96% less time,26.09% fewer calls; early successful
  Hybrid257.384130s/23calls remains faster: new takes34.02% longer than early
  (new/early=1.3402). These are single historical whole-system references,
  not a fresh matched pair or isolated field-planning causal improvement.
  First nonfinger-hand/bowl contact61 is record-only. Final LIBERO success
  occurred with gripper command+1; no post-final-release stability was tested.
  Input tokens max8455/last8294; planning4.368615s,model pipeline311.801149s.
  Runtime267sources/4helpers remained sealed, no other scene/Direct queued.
  Runner1503472 and private supervisor closed; shutdown.json/video saved.
  Independent terminal audit25checks allpassed, including390completechains,
  all70Direct actions exact,161local operations actions/diagnostics replayexact,
  all34request state/image boundaries and video-excluded timing. Full Field
  action replay was not run and is not claimed.
  Predecessor v2_001 remains infrastructure_error after54complete actions:
  real recorder expected prediction.actions for explicit gripper intent.
  It failed before close action54; no closure executed.21audit checks passed.
  Renderer-only repair added operation panel/events;64regressions and5actual
  recorder/render tests passed before002. Initial v2 validation230tests passed.
  Preserve both attempts and frozen sources; no new live trial is authorized.

- Latest user authorization (2026-10-15): implement the proposed improvements
  and test them. New opt-in --speed-first-v2 restores full interaction guidance,
  adds explicit model-selected local open/close-and-settle operations (ordinary
  Direct5 unchanged), allows explicit free-space Field recovery, bounds model
  history with a local digest, and normalizes small quaternion norm errors with
  raw/effective provenance. Keep old profiles and immutable run artifacts.
  One fresh ORIGINAL bowl zero-shot trial is authorized, task0/init0/seed7,
  Astra medium through ChatGPT subscription only, raw LIBERO completion,
  collision record-only, video-excluded control timing and no episode cap.
  runs/speed_first_v2_bowl_zero_shot_001 is being prepared; no launch until
  focused integration tests and protocol/source/input seal pass. Freeze runtime
  through the experiment. Do not queue Direct, other scenes or automatic retries
  of an episode. If incomplete, analyze this original-bowl attempt first.
  Local finger stability does not certify grasp/release/task completion.
  This combined test is not a component ablation or a fresh matched Direct pair.

- Retry002 terminal audit completed: independent_audit.json21checks passed;
  all578action/executed/feedback chains,93Direct5chunks,100request-boundary
  observation/history payloads,264source/4helper/224frozenruntime hashes,
  official inputs, video subtraction and shutdown verified. Fullpolicyreplay
  was not run and is not claimed. Final analysis.json links this audit.

- Retry002 is TERMINAL INCOMPLETE. User approved stop/analyze on2026-10-15;
  it had already automatically ended Oct14~19:16UTC with SchemaError, so preserve
  raw subscription_error rather than relabeling user_stop. Root closed private
  supervisor02:02:51UTC Oct15; runner696092gone, workersnone, control.sockabsent.
  Final runs/speed_first_bowl_zero_shot_002/analysis.json:578complete actions,
  113Field/465Direct,100requests/98validresponses,LIBEROfalse/API-key0.
  Control1246.958883s minusvideo88.060198s =1158.898685s (19m18.899s), NOT
  completionlatency. Localplanning1.962429s. Collisionrecord-only detected
  nonfingergripperhand-vs-bowl first159; not episode termination criterion.
  Overload34 recovered via identicalrequest35 after2.918s. Finalrequest100
  quaternion[.9728,-.2292,.0125,.0865]norm1.003250208 violates.001unitnorm
  tolerance; exactSchemaError reproduced offline withzeroactions/network.
  At548modeljudgedrimattachment and switched to30Fieldtransport actions;
  do NOT claim wholeepisode nevergrasped or nevertransported. Placementfalse.
  Earlybaseline184actions/23requests/net257.384s vs this prolongedrecovery;
  same reset/planner but promptchanged (8697->4488chars) and omitted explicit
  multi-five-action-chunk closurewarning. Initialcurrent5stationaryclosecommands
  beforelift vsold15first/20successful; handoffEEF~8.29cmdifferent. Plausible
  interactioncauses, not isolatedcausality. No productioncode/promptchanged
  this analysis turn. Proposed restorefullhistoricalinteractionguidance plus
  separate robustquaternionhandling are NOT implemented. No newtrial/restart.

- Retry002 IS STILL RUNNING under private supervisor, runner PID696092, launched
  2026-10-14T18:55:44Z; verify /proc/run_state each continuation. No runtime edits.
  runs/speed_first_bowl_zero_shot_002/progress_snapshot.json is NONTERMINAL.
  User stop decision pending via async question call asking stop this prolonged
  incomplete round/analyze or keep waiting LIBERO. NO reply/stop authorization
  received yet; no operator_stop file created, no automatic cap. At~19minutes,
  >518actions,83Field then repeatedDirect grasp/recovery, LIBERO false.
  First forbidden contact159 was gripper nonfinger hand versus target bowl;
  collision remains record-only. Request34 overloaded; identical-payload retry35
  succeeded after2.918s. Prior001 remains unchanged. Early successful behavior
  comparison: first Field-to-Direct at83vs77, EEF endpoint~8.29cm apart; initial
  current closure5stationary commands beforelift vs15earlyfirst/20earlysuccessful.
  Simplified prompt omitted explicit multi-five-action-block closure warning,
  though retained measured-finger settling guidance. This is plausible evidence,
  not isolated causality. On user reply stop, create graceful operator_stop file
  at trial/ then wait for complete-feedback terminal summary, audit and close only
  this private supervisor. Do not restart/queueanothertrial. Independent audit
  agent retry002_audit verified264src/4helpers/inputs/initialobs/subscription-only.

- Latest user authorization (2026-10-14): explain differences from early versions
  FIRST, then retry the previous round. Comparison delivered before launch.
  One fresh original-bowl zero-shot Speed-first trial002 is authorized:
  runs/speed_first_bowl_zero_shot_002. Identical task0/init0/seed7, policy prompt,
  medium Astra subscription settings and simulator as001; only tested bounded
  server-overload retry added. Raw LIBERO completion, collision record-only,
  video-excluded timing, no episode cap, automatic episode restart or other task.
  Preserve001 and all earlier records.264sources/4helpers/official inputs sealed
  before launch; no runtime changes until terminal audit. If incomplete, analyze
  this original-bowl round before any further episode or implementation.

- Latest request (2026-10-14): debug the subscription overload; no fresh robot
  episode was launched or resumed. Debug artifacts: runs/speed_first_overload_debug_001.
  The exact failed trial001 request9 payload was re-sent through Astra medium
  subscription OAuth only and returned a validated complete response on its FIRST
  attempt in5.986642s. No simulator/actions/API-key calls. No session restart is
  currently indicated. Original trial001 remains terminal subscription_error.
  Speed-first alone now retries exact server_is_overloaded up to twice for the
  SAME observation, 2/4s +0..1s jitter, honoring HTTP Retry-After up to60s (longer
  hint stops rather than retrying early), existing budgets rechecked. Failed
  responses never commit history or execute actions; waits/calls remain counted.
  Other profiles retain single-attempt behavior. HTTP diagnostic metadata now
  preserves sanitized retry_after_s.120tests+69subtests passed;25new overload
  cases include Field/Direct exactly-once, payload/history equality and limits.
  Keep historical frozen artifacts intact. Debug success is not task completion
  or evidence of speedup; no robot trial is currently running.

- Speed-first original-bowl zero-shot trial001 is TERMINAL INCOMPLETE after
  subscription service error server_is_overloaded on request9. See
  runs/speed_first_bowl_zero_shot_001/analysis.json.78complete actions:53Field,
  25Direct; first close action56, short lift then model-assessed missed grasp
  and Direct reopen/lower recovery. LIBERO false, collision false. Full control
  133.871173s minus17.789099s video =116.082074s (1m56s), NOT completion latency.
  No retry/API-key fallback/new scene; runner and worker closed and private
  supervisor shut down. Preserve raw subscription_error and all frozen inputs.
  This service interruption does not establish successful speedup or a policy
  inability to complete the task. No new trial is currently running.

- LATEST USER AUTHORIZATION: "test it" now authorizes one fresh original-bowl
  zero-shot speed-first trial, starting at task0/init0/seed7. Use medium Astra
  through the subscription only, raw LIBERO completion, collision record-only,
  no episode cap, and video-excluded timing. Freeze runtime/inputs through the
  run; preserve V36 and all historical artifacts. Run directory is
  runs/speed_first_bowl_zero_shot_001 (trial/ holds runner artifacts). Do not
  restart a stopped episode, queue other tasks or rerun Direct automatically.
  Analyze an incomplete original-bowl result before further implementation.

- LATEST USER AUTHORIZATION (2026-10-14): Implement an opt-in return to the
  earlier successful Distance Hybrid as the speed-first baseline: Field for
  navigation, unchanged Direct5 for interaction and recovery. This authorizes
  implementation and offline validation, superseding the prior no-source-edit
  pause for this work only. It does not launch a fresh task trial or restart a
  stopped episode/queue. Preserve V36 and all historical artifacts. New profile
  --speed-first uses zero-shot Astra medium via ChatGPT subscription only,
  original two-RGB/state8 interface, 33 cubed / 5 mm, Dijkstra and measured-pose
  tracking, 30-action Field refresh, ten-segment/ten-action local plans. No pixel
  objective, visual-evidence admission, automatic upward recovery or V36 guard
  inheritance. Optional predicted masks remain planner constraints, with uncertain
  obstacles represented by model-chosen soft costs. Actual LIBERO success is
  completion; collisions are recorded evaluator-only and do not stop/disqualify
  this new profile (LiberoConfig.stop_on_collision=False). Existing strict modes
  retain their defaults. Time excludes measured video overhead. Offline archived
  response/observation replay is behavior regression, not a fresh success or
  inference-speed benchmark. No new speedup or success claim until a fresh trial.
  Implementation is complete: src/gpt_field/speed_first_hybrid.py plus opt-in
  runner wiring and evaluator record-only mode. Final saved-response regression
  runs/speed_first_implementation_001/baseline_replay_final.json matches184/184
  actions exactly (106Field/78Direct), all feedback and23refresh boundaries.
  Focused policy/distance120 tests, runner101 tests and simulator196 tests passed
  in separate overlapping suites; do not add counts. No fresh task trial ran.

- V36 trial001 is TERMINAL INCOMPLETE and work is PAUSED AT USER REQUEST.
  runs/visual_field_v36_zero_shot_001/analysis.json.181complete transitions,
  169effectiveField/171selections,10Directholds,0gripper/LIBERO/forbidden collision;
  16Astra medium subscription replies,0API-key. Full790.426555s minus4.450189s
  video=785.976366s (13m05.976s), NOT completion latency. Model512.805279s;
  local decision/planning153.831140s; feedback102.456247s.223frozen sources/
  7helpers/inputs matched; complete-feedback boundary stop/audit passed.
  Runner/worker/private supervisor closed17:54:16Z; shutdown.json saved.
  Watch21 and explicit ordinary+y correction22 worked; wrist remained reliable
  all181steps. But final EEF netup51.480mm while pixel error106.667->89.965;
  no descending approach/grasp/contact or scene geometry revisions. After125,
  upward waypoints optimize perspective/bearing; response180 explicitly admits
  this does not reduce physical target distance yet chooses another up20mm.
  Root stopped for reviewed persistent task stagnation/proxy-objective drift,
  raw user_stop, not timeout. User asked PAUSE after unsuccessful round and
  discuss architecture/simplification before any more implementation or test.
  Future user priority is completion speed; collisions/attempts may be accepted
  as separately recorded metrics, but no new protocol/code/test is authorized
  before discussion. Do not resume source edits or experiments automatically.
  Active20-50% speed objective unachieved; do not mark complete.

- LATEST USER STEERING DURING V36 (2026-10-14): Finish only this already-live
  original-bowl round. If it remains unsuccessful, PAUSE the active goal after
  saving/auditing the result and discuss next design before any further source
  implementation or trial. User now prioritizes task-completion speed and is
  willing to accept collisions/attempts for a future simplified version; do not
  silently change this sealed round's criterion. Summarize current architecture
  first. No next version/scene/Direct/restart has authorization before discussion.
- V36 trial001 IS LIVE under private supervisor, runner PID546868, launched
  2026-10-14T17:39:20Z. Verify /proc and live run_state each continuation; do not
  infer completion from polling timeouts.223runtime sources/7helpers/plan/inputs
  sealed. Actual watch triggered after21, model replaced -Y with ordinary+Y after22;
  references held and bearing improved; six ordinary preview candidates reached
  the actual request. Later +Y passes and cabinet uncertainty blocks alternate,
  followed by upward recovery; checked wrist task offer selected93. No grasp or
  LIBERO success as of104actions/10completed responses. No code changes while live.

- V36 original-bowl zero-shot trial001 is READY FOR SEAL/LAUNCH:
  runs/visual_field_v36_zero_shot_001. Four opt-in modules add ongoing wrist
  visibility review after completed observers, and current ordinary noncontact
  Field replan previews. Native schema, medium subscription-only inputs,
  physical geometry/debt and LIBERO/no-forbidden-contact completion retained.
  Integrated actual V35 prefix19 transitions matches, then requests review after18
  while wrist remains reliable;6axis20mm cloned full-route/first-command previews
  pass, live state unchanged. No old outcome after the new decision is consumed.
  Initial raw-track-summary adapter omitted monitor current-point fields; fixed
  to final row.visual.tracking and actual prefix repeated. Final45runtime checks,
  10V36+V35runner and16supervision checks pass (overlapping). Prior V35 raw/frozen
  records and inherited sources remain unchanged; runner only adds V36 routing.
  Freeze all runtime/helpers/plan/inputs before launch, and keep frozen until
  terminal audit. No episode cap/restart/queue/Direct/other scene; on original
  incomplete outcome stop and analyze. This is offline readiness, not task
  success or whole-task speedup. Active20-50% goal remains unachieved.

- V35 original-bowl zero-shot trial is TERMINAL INCOMPLETE:
  runs/visual_field_v35_zero_shot_001/analysis.json. Raw user_stop after reviewed
  blocked80/90/95 despite reference-only85;97complete transitions,75Field
  selections/71effective,22Direct holds,0gripper/grasp/LIBERO/forbidden collision.
  9valid Astra medium subscription replies, API-key0. Full371.622372s minus
  2.472832s video =369.149541s (6m09.150s), NOT completion latency. Model281.710836s;
  planning39.670128s;feedback40.005160s. All219sources/7helpers/plan/inputs frozen
  through terminal audit; worker/runner/private supervisor closed17:19:32Z,
  shutdown.json saved. No other scene/Direct/restart/queue. Post-terminal edits
  may proceed; historical frozen sources and raw trial artifacts remain intact.
  Initial reply supported53effectiveField/~61.207mm descent, but wrist target
  error worsened with validated response by18 and lost at37. Ongoing intent did
  not use this objective until refresh54. Cabinet warned52/refused53: complete
  model sweep disjoint but warned-object model gap33.409->30.434mm decreased;
  this composite current-geometry rejection is NOT missing/old geometry.
  Hand-envelope mismatch54 corrected55, continuity exercised60; recovery60..78
  spent8.002mm heading including drift. Fresh wrist only2training/0heldout;
  fixed3mm aggregates need at least15mm for3training+2heldout, impossible in8mm
  recovery. Analyze ongoing visual objective/visibility and noise-aware learning
  cadence before a new original-only test. Active20-50% speed goal unachieved.
  Post-terminal causal1/2/3mm replay:1mm first useful upward7.77mm goal only78,
  with~0.227mm heading allowance left;2/3mm do not fit after55. Smaller aggregates
  alone do not establish executable continuation. Prioritize keeping tracked
  task target in view and reviewing conflicting ongoing intent before loss,
  not more recovery-only credit or unobserved reverse. Analysis artifact:
  visual_field_v35_design_001/v35_wrist_visibility_cadence_analysis.json.
  No fully input/collision-qualified completed Direct comparator exists; see
  visual_field_v35_design_001/direct_comparator_audit.json. V35 real tracker ident
  mismatch was repaired before seal: production tracker regression plus90runtime
  checks passed; integrated61 actual V34 transitions match, with20.32mm wrist
  offers passing full-route/first-command at58/59 and unchanged live state.

- V35 preparation evidence (completed before the terminal trial above):
  runs/visual_field_v35_design_001/design.json and prepared unsealed harness
  runs/visual_field_v35_zero_shot_001. Separate wrist-bearing objective, stable
  same-observation hand-source coordinates, exact current-feedback factoring
  and explicit effective-limit provenance are opt-in only. Root added a V31
  preview configure hook whose default behavior is unchanged; V35 binds the
  prospective role AFTER clone configure. Frozen historical sources stay intact.
  First integrated61-transition replay matched actions but exposed actual track
  identity field ident versus the logged identity adapter: wrist proposals were
  absent. Repaired actual ident use and repeated integration before sealing;
  pure logged replay alone was insufficient. Preserve all actual geometry/contact/debt,
  zero-shot medium subscription-only inputs, LIBERO/no-forbidden-contact stop and
  video-excluded timing. Original bowl only; analyze failure before other scenes.

- V34 repaired retry002 is terminal and incomplete:
  runs/visual_field_v34_zero_shot_002/analysis.json. Raw user_stop after reviewed
  blocked stationary responses79/89/94/99 and reference review84;105complete
  transitions,64Field selections/61effective,41Direct holds,0gripper/grasp/LIBERO/
  forbidden collision.17valid Astra medium subscription calls,API-key0. Full
  632.721634s minus2.607649s video =630.113985s (10m30.114s). Model519.180648s;
  planning58.899343s;feedback43.713532s. All212runtime sources/7helpers/plan/inputs
  stayed frozen; terminal audit passes, worker/runner/private supervisor closed.
  Prospective role repair worked for ordinary previews after Direct proxy16;
  actual21 retained earlier goal,26 manually proposed approach, so no causal
  whole-task speedup claim. Initial cabinet uncertainty halo includes hand by
  0.593mm despite nominal core separation. Model26 chose15mm for6cm goal from
  partial visibility while runtime scene_execution_limits=null. Scalar cabinet
  worsening59 triggers reverse; full model margin increases but is not physical
  proof. Reverse heading allowance exhausts78; no validated anatomical goal or
  grasp. Analyze generic reliable visual objective, consistent projected evidence
  and excessive current-feedback duplication before a fresh original-only test.
  Preserve all runtime/history; no other scene/Direct/resume/queue. Active20-50%
  speed goal remains unachieved. No live test now; post-terminal edits may proceed.

- V34 original-bowl trial001 is terminal and incomplete:
  runs/visual_field_v34_zero_shot_001/analysis.json. Raw user_stop after reviewed
  blocked stationary responses52/57/62;68complete transitions,46effectiveField,
  21Direct holds,0gripper/grasp/LIBERO/forbidden collision.7valid Astra medium
  subscription replies,API-key0. Full306.613067s minus1.736515s video equals
  304.876552s. Model246.324019s;local planning26.613942s;feedback26.691101s.
  All210runtime sources/7helpers/inputs stayed frozen during the trial; terminal
  audit and shutdown passed. Final feedback dependency fix worked. Qualified
  visual task-goal transitions0. Typed reference review and proxy choice worked,
  but recovery previews inherited old Direct role binding after reference-only
  response47, falsely rejecting all Field candidates. Exact52-transition replay
  changing only prospective role context admits a2mm first command on8mm reverse
  without changing geometry/debt; not proof of full-route or task completion.
  Post-terminal guard/executor34 repair is now implemented for reverse and
  ordinary recovery previews;42related offline checks passed. Final runner
  manifest/inner-exception verification and a fresh original-only retry002 are
  in preparation. Historical trial001 sources/helpers/plan remain immutable.
  No current live test, no other scene/Direct/queue/resume. Keep medium
  subscription-only, zero demo, RGB+robot inputs, actual LIBERO plus no forbidden
  contact, full video-excluded timing and no automatic cap/restart. On an
  incomplete original result, stop and analyze. Active20-50% speed goal unproven.

- V34 is validated and being prepared for one fresh original-bowl zero-shot
  trial: runs/visual_field_v34_zero_shot_001. V34 explicitly offers currently
  qualified visual task goals from recover, dispatches paid-history reference
  repair at health-required handoffs, and assembles complete transition feedback
  once after all observers. Fixed the V18-summary/V24-witness cache order before
  sealing; five actual saved62 comparisons preserve actions/control/feedback
  except Python instance IDs. Observer median0.659670->0.383597s is component-only,
  not task speedup.135 saved V33 transitions matched; current62 offers up to20.213mm
  pass preview but retain15mm travel cap;119/134 extract6/5 candidates with no
  control/debt mutation.55 prior combined checks,18 post-order-fix related checks,
  22 prior helper/protocol checks passed (overlapping). Independent readiness
  review found no new blocker. Distributed whole-hand witness is OFFLINE ONLY,
  not installed in production; legacy outline warnings remain possible. Freeze
  all runtime/helpers/inputs before launch. Keep Astra medium subscription only,
  zero demo, permitted RGB/robot inputs, LIBERO plus no forbidden contact,
  full video-excluded time; no cap/restart/queue/Direct/other scene. On incomplete
  original result, stop and analyze. No successful-task speedup established.

- V33 original-bowl zero-shot trial001 is terminal and incomplete:
  runs/visual_field_v33_zero_shot_001/analysis.json. Raw user_stop after reviewed
  repeated blocked handoffs269/274/279, not a timeout.285 complete transitions,
  249 Field selections/247 effective,36 Direct zero holds,0gripper/grasp/LIBERO/
  forbidden collision.27 valid Astra medium subscription replies,no API-key
  calls. Full1097.946775s minus7.012174s video =1090.934602s (18m10.935s).
  Model691.787118s;local planning180.583531s;feedback193.636068s. Effective Field
  path201.381mm; longest one-reply block50actions/57.781mm. Proxy role and three
  candidate selections exercised; ordinary reverse executed but V33-specific
  precheckpoint-abort events0. No successful-task speedup established.
  Concrete evidence: request62 has four qualified visual goals up to20.213mm,
  but recover-phase filter excludes them before geometry preview, leaving8mm
  recovery offers with3mm arrival tolerance. These longer goals were not proved
  executable. At119/134 paid reference review is required but reliable-track
  budget handoff does not dispatch extraction. External point plateau coexists
  with real improving wrist bearing; it was not itself task stagnation.
  At243 whole-tool box width108.30->138.29px with~1.77px point shift triggers
  scalar cabinet warning; comparable whole-outline interval remains inconclusive.
  Current modeled sweeps are disjoint but warning constrains continuation.
  Reverse consumes8.155mm including residual movement;4 paid reference candidates
  still existed at268, though executable motion offers were empty. Model269
  supplies null external/wrist references; the subsequent board becomes empty.
  Do not conflate absent motion offers with absent reference hypotheses or claim
  natural total correspondence loss at269. Runner/worker/private supervisor are
  closed; shutdown.json saved. No other scene/Direct/queue/resume.204 runtime
  sources/7helpers/plan/inputs stayed sealed. Offline validation59combined tests,
  16helper+protocol checks passed; exact69-transition old replay and final-source
  saved56/68 checks admit2mm reverse with unchanged debt, but no real V33 abort
  executed in this trial. Analyze phase/offer lifecycle, current-reference review
  and stable whole-hand projected geometry before another fresh original-only
  test. Active20-50% speed goal remains unachieved; preserve historical evidence.

- V32 original-bowl retry002 is terminal and incomplete after reviewed admission
  deadlock: runs/visual_field_v32_zero_shot_002/analysis.json. Raw user_stop,
 75 complete transitions,54 Field selections/49 effective,21 Direct zero holds,
 0gripper/grasp/LIBERO/forbidden collision.10valid Astra medium subscription
  replies, no repairs/API-key calls. Full346.296811s minus1.864052s video
  =344.432759s (5m44.433s). Firstreply49effective actions/55.509mm. V32 selected
  anatomy candidate training import was exercised; proxy and learned-goal
  selection were not. Clone fix did not fail. All200 sources/7helpers/input
  hashes stayed sealed; terminal protocol38/38passed; runner/worker/private
  supervisor closed. No other scenes/Direct or implicit episode resume.
  At55 model copied an older rejected reverse endpoint after hold drift:
 8.080125mm exceeded8mm. Corrected56/68 exact reverse requests still failed
  direction_change_requires_review: previous downward acquisition had only
 2.409928/2.532615mm charged (<4mm checkpoint), with shared allowance remaining.
  Direction was observed adverse, so continuing toward checkpoint was blocked.
  Lateral62 had all8 full-margin sweeps clear, cabinet gap28.401->30.401mm,
  but mature prior closing latch vetoed normal transit. Terminal shared17.408864mm
  remained. Opt-in V33 is under implementation to permit explicit reviewed
  finite reverse to abort a precheckpoint adverse heading, preserving paid
  debt/current physical checks; no budget refund or stale preview authority.
  Validate offline actual56/68 states and then a fresh original-only task
  before claiming improvement. No successful-task speedup established; active
 20-50% goal remains unachieved.

- V32 original-bowl zero-shot trial001 ended with local implementation error:
  runs/visual_field_v32_zero_shot_001/analysis.json. Raw infrastructure_error is
  preserved.27 complete transitions,24 effective Field actions,19.857mm Field
  path,5 valid Astra medium subscription replies,0grasp/LIBERO/forbidden contact.
  Full179.111365s minus0.650902s video =178.460463s (2m58.460s); model145.937359s.
  Last reply supported13 effective Field actions; all replies used tool_anatomy,
  so new rigid-transit-proxy mode was not exercised. Selected candidate had
  only0.185px relative motion over16.044mm; four rank1 training aggregates but
  sole prospective SSE0.33245 exceeded zero-motion0.16405, no validated goal.
  Before request6, FieldProposalBankV31._clone followed checked.owner back into
  the live executor, then scene_assessor bound policy/client and a thread lock;
  copy.deepcopy raised TypeError. This is not an inference/auth failure or
  evidence of successful task speedup. All200 runtime sources/7helpers/inputs
  matched their seals during the trial; independent terminal audit38/38passed.
  Runner/worker/private supervisor closed; no other scene or Direct ran. A
  post-terminal isolated cloning repair passed15 focused checks+8 runner
  checks. Exact27transition saved replay reproduced the frozen crash and fixed
  preparation produced2 checked offers without new model/simulator calls.
  Historical frozen code and raw results stay intact. No fresh trial ran after
  this repair. Original-first failure branch and
  active speed goal remain; do not resume this stopped episode or its queues.

- V31 original-bowl zero-shot trial001 is terminal and incomplete:
  runs/visual_field_v31_zero_shot_001/analysis.json. Root boundary-stopped after
  four repeated stationary five-action replies and explicit blocked/incomplete
  statements.71 complete transitions,50 Field selections/47 effective actions,
  21 Direct holds,0 gripper/grasp/LIBERO/forbidden collision.9 valid Astra medium
  subscription replies,no repairs/API-key calls. Full334.965734s minus1.795717s
  video =333.170017s (5m33.170s);model252.675094s (75.84%). Firstreply30 effective
  Field actions; effectiveFieldpath35.981mm. No V31 visual-goal offers/selections
  or explicit scene revisions. Earlier epoch1 response DID fit8 aggregates,
  but5 held-out SSE2.672751 exceeds zero-motion2.229956; no validated descent.
  Reference changes31/44 restart objective evidence; terminalepoch3 has0samples.
  Independent trace/provenance audit passed:192sources/7helpers/inputs frozen,
  all71 transitions complete. Runner/worker/private supervisor closed; no other
  scene or Direct ran.125 focused tests and9 harmless helper tests passed before
  launch. Offline archived242-prefix checkpoint admits a new6mm visualgoal with
  V31 exact connectors, but this did not occur in the fresh trial and is not a
  task-speed result. Preserve raw user_stop; no successful speedup established.
  Analyze reference grounding and useful checked evidence acquisition before
  any further task; original-first failure branch remains in effect.

- V30 original-bowl zero-shot full trial001 is terminal and incomplete:
  runs/visual_field_v30_zero_shot_001/analysis.json. Root boundary-stopped after
  repeated stationary handoffs and explicit model requests for intervention.
  279 complete transitions,253 Field selections/250 effective Field actions,
  26 Direct holds,0 gripper/grasp/LIBERO/forbidden collision.26 valid Astra
  medium subscription replies,no retries/API-key calls. Full911.935296s minus
  6.698376s video =905.236921s (15m05.237s);model665.232529s (73.49%). Maximum
  47 effective Field actions/reply; effective Field path274.566mm. Typed review
  now dispatches candidate extraction (11 boards,10 explicit selections), but
  reference identity/correspondence remains unreliable. Fresh full-trial target
  approach qualification and supported-intent execution both0. All186 runtime
  sources/7 helpers/inputs frozen;32 independent protocol checks passed. Runner,
  worker and private supervisor closed; shutdown.json saved. No other scene or
  Direct ran; no successful-task speedup established. Preserve raw user_stop.
  Positive-Y block195..241 improves image error39.530->25.922px, while retained
  observation0 metric bowl hypothesis remains205.663mm in negative Y. Further
  positive-Y goal242 enters retained cabinet full uncertainty halo; nominal
  hand-core gap remains4.066mm. At257 intent uncertainty15->25mm expands the
  combined cabinet margin55->65mm and puts unchanged measured start1.969mm
  inside its halo (nominal gap63.031mm). This is model-geometry inconsistency
  and uncertainty-admission deadlock, not evidence of physical cabinet contact.
  Six hold-request blocks cost146.999 model seconds. Actual component test
  separately matched243 archived transitions exactly, then21 new Field moves/
  15.170mm without model calls, stopping on immutable-reference loss. Component
  success is not task completion or speed evidence. Analyze metric/image
  reconciliation, stable references and recovery admission before another
  original-only trial; active speed objective remains unachieved.

- V29 original-bowl zero-shot full trial001 is terminal and incomplete:
  runs/visual_field_v29_zero_shot_001/analysis.json. Root boundary-stopped after
  repeated stationary handoffs and explicit model requests for intervention.
  276complete transitions,255Field selections/250effective Field actions,
  21Direct holds,0gripper/grasp/LIBERO/forbidden collision.21valid Astra medium
  subscription replies, no retries/API-key calls. Full774.679110s minus7.066711s
  video =767.612399s (12m47.612s);model558.354963s (72.74%). Maximum40actions
  perreply; totalEEFpath378.310487mm, net258.377647mm. Supported-intent lifecycle
  actions0.181sources/7helpers/inputs remained frozen;32independent protocol
  checks passed. Runner/worker/private supervisor closed, shutdown.json saved.
  No other scene/Direct ran; no successful-task speedup established.
  At243 the ONLY current warning is the target bowl.3mm downward candidate has
  all8full-model sweeps disjoint, but target gap37.5823->34.5870mm violates
  retained nondecrease. This is intended approach conflicting with the guard,
  not evidence of actual collision. A historical target closing latch from
  epoch3 also persists despite current comparable/inconclusive boundary data.
  Reverse244..253 adds4.496113mm; other-bowl same-epoch scalar gap decreases
  4.438011px vs4.381771px threshold (excess0.056240px), while full-outline
  interval remains inconclusive. Opposing warning headings then block reverse.
  Separate reference audit: handoffs40/42/61 had31retained frames but extraction
  count0 because their reasons do not dispatch candidate review; empty boards
  do not prove no candidates exist. Preserve independent audits in this run.
  V29 tracking/guard implemented; focused validation passed. Real controller
  diagnostic matched207prefix transitions exactly then16newactions/7.568921mm,
  no model calls or collision. It stopped before qualification because fresh
  same-heading zero-flow SSE0.088823 failed>0.25; supported lifecycle0actions.
  Component continuation and qualified unit fixtures are not task benchmarks.
  Analyze generic target-approach semantics, evidence-consistent recovery and
  reference-review dispatch before another original-only experiment. Active
  speed objective remains unachieved; do not advance other scenes on this result.

- V28 original-bowl zero-shot trial001 is terminal and incomplete:
  runs/visual_field_v28_zero_shot_001/analysis.json. Root boundary-stopped after
  persistent lateral-command/height-correction conflict and an exhausted useful
  reverse route; no grasp/LIBERO/forbidden collision.228complete transitions,
 223Field selections/211effective Field actions,5Direct holds,0gripper actions.
  Twenty valid medium subscription replies, no API-key calls. Full782.717879s
  minus5.884887s video =776.832992s (12m56.833s);model564.126884s(72.62%).
  A single reply supported44actions; totalEEFpath358.959mm, net235.286mm.
  This is real reuse but no successful-task speedup. All177sources/7helpers
  remained sealed; full transition/provenance audit passed. Runner/worker and
  private supervisor closed; no other scene/Direct ran. Preserve raw user_stop.
  Independent207/209 audit verifies current exact geometry and positive full
  gaps; only nondecrease clauses fail. After lateral1mm commands, upward EEF
  drift leads fixed-Z tracking to command downward0.295253mm/0.019969mm. Both
  reduce modeled gaps to warned ramekin/otherbowl despite remaining66+mm apart
  under that model and having heading/shared budget. Repeats221/222,223/224
  reproduce the incompatibility. This is a route/tracking/separation contract
  conflict, not numeric binding error or proven physical collision. Next design
  should admit useful route progress and bounded tracking error consistently,
  with retained hard geometry/current contradictions/debt; do not merely clear
  warnings or add travel. Reference audit separately found feature-domain loss
  and one repeated static-background selection, no new tracker defect. V28 real
  archived-output component14exact+4newactions/3.403mm stopped on actual new
  image boundaryclosing; no model/taskbenchmark claim. Active speed goal remains
  unachieved; analyze this basic-task result before any further scene.

- V28 is being validated for one fresh original-bowl zero-shot test. New guard
  gathers finite unsupported-heading evidence under the original current full
  model-geometry/reference/boundary gates and existing8mm heading/20mm shared
  measured allowances;1mm cold commands grow only with empirical robot response
  to at most3mm. No fictional held-out samples or debt reset. Explicit normal
  model-separating transit can use the same physical heading rows, retaining
  distinct-direction/checkpoint rules. Reverse temporal evidence uses comparable
  whole-outline sensitivity; inconclusive is neither opening nor clearance,
  and unavailable/current robust closing stops that route. Candidate boards now
  trigger at these actual handoffs using paid RGB history, never injected or
  automatically selected. See runs/visual_field_v28_design_001/design.json.
  Offline prefix0..13 matches V27 retry002 exactly. First validate frozen actual
  continuation with ONLY archived output0, then decide the fresh full trial.
  Keep all runtime/helpers/inputs frozen while either experiment runs. No other
  scene/Direct queue, no automatic cap/restart; actual LIBERO plus no forbidden
  contact and whole-task video-excluded time remain the outcome. No task-speed
  gain has been established. Existing3mm-before1mm construction can still cause
  a conservative geometry refusal on an unused tail; it grants no exemption.

- V27 repaired original-bowl retry002 is terminal and incomplete:
  `runs/visual_field_v27_zero_shot_002/analysis.json`. First reply supported
  14 effective Field actions/12.832mm. Final46 complete transitions include31
  effective Field actions/19.900mm path, no close/grasp/LIBERO/forbidden contact.
  Root stopped after repeated stationary handoff and explicit intervention
  requests. Eight medium subscription calls:7valid and1same-observation schema
  repair;API-key0. Full272.057927s minus1.129129s video =270.928798s;
  model221.675024s. All171sources/7helpers/inputs remained frozen,25terminal
  checks passed, runner/worker/private supervisor closed. No other task ran.
  Independently measured next bottlenecks: new planar heading28 has0samples
  within15degrees (closest28.19), so support-before-motion blocks evidence
  collection; same-epoch reverse32 expands the whole-tool box8.335px and
  crosses inherited6.20005px threshold despite V27 boundary interval
  [-24.816,+8.276] remaining inconclusive; reference candidate extraction
  count0/last_handoffnull despite24retained frames because these new refusal
  reasons do not trigger the V24 candidate handoff. Empty board is not failed
  candidate search, nor does projected expansion prove physical danger.
  Next design must unify uncertainty treatment, surface paid-history repairs,
  and allow finite useful heading evidence acquisition under retained scene
  geometry and existing physical budgets. Do not just extend reverse allowance
  or lower thresholds. No successful-task speedup is established; goal active.
  Earlier immutable-goal fix remains:14new+66affected+4runner+18helper tests
  passed; frozen retained-goal diagnostic matched9prefix transitions and then
  executed5new actions/3.4999mm to local arrival. This component is not a task
  benchmark. Preserve both full attempts and all frozen diagnostic artifacts.

- V27 original-bowl zero-shot full trial001 is terminal and incomplete:
  `runs/visual_field_v27_zero_shot_001/analysis.json`. Root boundary-stopped
  after repeated stationary handoff and a verified numerical route-binding
  defect:21 complete transitions,8 effective Field actions/4.051mm actual path,
  no grasp/LIBERO/forbidden contact. Seven medium subscription requests include
  six valid responses and one same-observation schema repair; API-key0.
  Net control213.495245s excludes0.648917s video; model190.149091s. All170
  sources/7helpers stayed frozen, final24-check audit passed, worker/runner
  and private supervisor closed. At reference repair9, identical geometry
  and reconstructed fixed goal differed only9.72e-13m, but round(goal,12)
  exact tuple equality falsely rejected the route. Fix V27 only using an
  immutable first-acknowledged raw goal and finite serialization tolerance,
  with strict remaining context and all original debt. Verify real archived
  continuation before a fresh original-only retry; no Direct/other scenes.
  Earlier approach diagnostic002 executed22 actions and found a legitimate
  current-direction handoff; it is not a completed task/speed result. Preserve
  both diagnostics and this full attempt. The active speed goal is unachieved.

- V27 is being validated for the active zero-shot Field speed objective:
  `runs/visual_field_v27_design_001/design.json`. Preserve projected warnings,
  full geometry and all physical debt. Use empirical whole-outline sensitivity,
  exact candidate/object full-margin model distance, and prospectively validated
  same-heading relative image response to distinguish ambiguous extent evidence
  from a directional contradiction. Conditional normal empty-tool transit is
  bounded by existing 8mm heading/20mm shared measured travel and fixed intent,
  with <=3mm commands; no contact/payload/stale/unknown/robust-closing exemption.
  Fitted support on a new axis cannot borrow old-axis held-out validation.
  Validate actual policy and frozen archived-output real-controller continuation
  before a fresh original-bowl-only medium subscription trial. No other scene
  or Direct queue, automatic episode cap/restart, hidden policy inputs, or live
  runtime edits. Only LIBERO plus no forbidden contact and full video-excluded
  task time can support the task-speed objective; component results cannot.

- V26 follow-up approach diagnostic is terminal and still incomplete:
  `runs/visual_field_v26_design_001/analysis.json`. Two generic measurement
  fixes are implemented: compare motion within representation epochs, and
  suppress subpixel neighbor drift only for exactly unchanged sampled
  descriptors after the original matcher has passed all checks. Final combined
  validation is 153 tests (not additive with the earlier 136), plus 14 approach
  helper checks. Corrected-neighbor real simulation used archived model outputs
  at 0/8 only, matched the first 15 transitions bit-for-bit and executed 16 normal
  Field actions / 13.121466mm actual path. Selection16 was refused after a new
  cumulative projected-gap warning; no LIBERO, collision or live model calls.
  Its 15.622673s is component time, not successful task time or inference speedup.
  All 166 sources/helpers and 6 inputs remained frozen; worker is closed.
  The cabinet box is now stable. Independent public-observation replay shows
  cabinet-specific full-uncertainty model gap grows 41.744819 to 45.815956mm;
  actual15 increases it 0.402008mm and refused16 would increase it 1.767436mm.
  These model-hypothesis distances do not certify physical clearance. Whole-tool
  image transport relies on six features spanning 6x13px, with no calibrated
  corner uncertainty. Analyze projected/metric disagreement and extent transport
  before fresh full model testing; do not just inflate a warning threshold or
  erase risk/history/debt. Preserve both V26 diagnostics and their source seals.
  Full V26 remains unsealed/unlaunched, no Direct/other scene queued, goal active.

- V26 component diagnostic001 is terminal, not a model/task benchmark:
  `runs/visual_field_v26_design_001/analysis.json`. 136 combined regressions,
  21 helper checks and 15 independent accounting checks passed. Real simulation
  matched all 25 archived prefix transitions bit-for-bit, then admitted the old
  refused action and two more (1.233581mm extra measured path). It stopped at
  selection28 on new same-epoch worsening after 1.627306mm fresh segment travel:
  gap -6.415812px versus 5.994988px threshold. No collision/LIBERO/model calls;
  worker closed and all frozen sources/inputs unchanged. The epoch fix works,
  but sustained execution/task speedup remains unproven. Audit finds a cabinet
  box jump despite a byte-identical interior RGB crop; inspect local feature
  matching and whole-extent extrapolation before launching a fresh model test.
  Keep diagnostic001's frozen source/input evidence. The full V26 plan remains
  unsealed and deferred; no other scene/Direct is queued. Do not label image
  estimator drift as proven physical obstacle motion or clearance.

- V26 preparation follows the V25 measurement-provenance defect:
  `runs/visual_field_v26_design_001/design.json`. Keep the V25 fixed-goal
  Field executor, response estimator, explicit route families and physical
  allowances. Segment temporal image-risk comparisons by actual tool/reference
  and neighbor representation changes, including same-ID extent revisions.
  Retain current enlarged geometry, valid prior adverse latches and all shared,
  heading and route debt; an epoch change supplies no clearance or travel.
  Uncharged intervening motion ends a reverse temporal segment without refund.
  The saved V25 25-transition prefix and old rejection have been reproduced
  offline with no model/simulator calls. Complete regression and independent
  review, freeze code/inputs, then run the archived-output real-controller
  diagnostic before deciding whether a fresh original-only zero-shot medium
  subscription test is warranted. Other scenes/Direct remain unqueued, and
  only actual LIBERO plus no forbidden contact qualifies task completion.
  Preserve full video-excluded timing and incomplete results; do not turn
  component continuation or an earlier stop into a successful speed claim.

- V25 original-bowl zero-shot test is terminal and incomplete:
  `runs/visual_field_v25_zero_shot_001/analysis.json`. Root reviewed three
  complete five-action stationary Direct blocks and explicit model requests
  for intervention; a boundary stop ended at 42 complete transitions: 24
  effective Field commands, 2 Field rejection holds, 16 Direct holds, and no
  grasp, LIBERO success or forbidden collision. Eight valid Astra medium
  subscription calls, API-key 0; full control 264.260184s minus 1.064401s video
  = 263.195783s. Model pipeline 227.183406s (86.32%). Sources stayed frozen,
  initial inputs exact, terminal provenance passed; worker/runner and private
  supervisor are closed. No other scene or Direct run started.
  The verified new defect is measurement provenance: at step 24, accepted
  reference/whole-tool ROI revision changed the representation (epoch 2 to 3),
  but reverse progress kept the pre-16 gap baseline +6.046097px. New pre-action
  gap was -4.747862px and post-action gap -5.414076px: only -0.666214px within
  the new representation, yet the old/new comparison reported -11.460173px
  against an 8.172456px threshold and latched further_worsening. Action 25 was
  refused. Preserve expanded current risk, prior valid worsening and all motion
  debt; reference/ROI changes are not observed motion or new travel allowance.
  Segment comparable temporal measurements before another original-only test.
  Earlier archived-output controller diagnostic003 improved reverse motion to
  +3.873862mm net over 21 actions/7.159117mm path, but stopped short of its goal
  at the original finite budget/minimum command boundary. It used 0 model calls
  and is not a completed-task benchmark. Neither result establishes speedup.
  Preserve all artifacts; no live edits, automatic queue/cap/restart or new
  Direct test. The active zero-shot Field speed objective remains unachieved.

- V24 original-bowl zero-shot test is terminal and incomplete:
  `runs/visual_field_v24_zero_shot_001/analysis.json`. Root reviewed repeated
  recovery refusal and two complete5-action stationary Direct blocks, with the
  model explicitly requesting controller intervention; a boundary stop ended
  at152 complete transitions,136 effective Field actions,0grasp/LIBERO/forbidden
  collision. Ten valid Astra medium subscription calls,API-key0; full control
  373.344409s minus3.779060s video =369.565349s. Model270.357995s (73.16%).
  First request supported83 effective actions/97.178mm measured travel, but no
  successful-task speedup follows. The ordinary8mm recovery at132 was previewed
  through V22 but mis-dispatched to V23 egress against a161.15mm old-origin
  extent; the model bypassed it at133. Distinct137 heading and140 response-cap
  refusals then blocked recovery. At140 the4mm checkpoint still had1.9068mm;
  low median longitudinal response, not exhausted budget, set command cap zero.
  Preserve this trial and all frozen sources. No scene/Direct advance occurred.
  Analyze explicit recovery dispatch and measured reversal transients before
  another original-only test; no live edits, automatic queue/cap/restart.
  Separate V24 cached-command diagnostic001 had0modelcalls and no task success:
  real candidate binding worked, but unchanged old tool outline excluded current
  descriptor support, so no new-reference confirmation action executed. Its
  15.421499s component time is not a task or inference result.

- V24 preparation addresses V23 diagnostic003's measured observability failure.
  Reinspection shows the selected raw21x21 RGB patch was byte-identical over
  3.9846mm actual movement; legacy controlled-motion support was tracker drift,
  not valid robot-reference evidence. Preserve the historical diagnostic and
  its original report, with this correction in the V24 analysis. Separate
  immutable, prospectively measured robot-reference motion from obstacle
  clearance. Expose current candidates from paid RGB at a blocked handoff even
  just below4mm; model selection remains explicit and semantic identity remains
  unproved. Finite model-conditioned routes retain original geometry/origin/full
  goal, nominal-core nondecrease, unknown/target/no-new-halo checks, all physical
  debt and current contradiction/domain stops. Total40mm includes initial4mm
  bootstrap and at most one4mm prospective confirmation after explicit reference
  repair; no renaming/reseeding renews this allowance. Favorable2D gap is no
  longer a travel-credit mechanism and inconclusive vision is not clearance.
  Compact only older request contexts; retain full current context and raw logs.
  Validate whole-policy continuation and real controller before freezing fresh
  original-bowl-only zero-shot Astra medium subscription test. Actual LIBERO
  plus no forbidden contact and full video-excluded timing remain required;
  no new Direct/scene queue, automatic cap/restart or edits during live testing.

- V23 preparation now has a measured original-bowl controller diagnostic result:
  `runs/visual_field_v23_design_001/analysis.json`. This is deterministic archived
  prediction plus one explicit checked offer, not a fresh zero-shot model/task
  benchmark. After a real initial hold exposed stale proposal-state handling,
  candidates now use the acknowledged guard/monitor/inventory state; actual
  command selection and acknowledgement remain exact. Production hold-orientation
  parsing is reproduced in the diagnostic helper. Offline validation is staged:
  70 initial checks plus13 affected executor/policy checks,71 unique tests.
  Diagnostic003 executed8 Field actions,3.984625mm measured egress,0 collision,
  0LIBERO and0modelcalls; continuation refused without a fresh visual grant.
  Controlled movement is detected, but cabinet gap/scale separation remains
  inconclusive within the4mm bootstrap. Control10.132299s is component time,
  not task completion time or speedup. All150 runtime/helper snapshots and6
  inputs unchanged;19 independent audit checks pass, worker cleaned up. Preserve
  all three diagnostics (001 startup error,002 hold-state defect,003 actual
  observability limit). Fresh V23 subscription plan remains unsealed/unlaunched.
  Analyze actual visual observability and execution assumptions before spending
  another full trial; no other scene/Direct test has started and goal is active.

- V23 preparation continues the active zero-shot Field speed objective after
  V22 initial uncertainty-halo deadlock. Preserve all historical results. Add
  explicit model-selected bounded Field egress only when all nominal known-tool
  cores and unknown regions are clear, initially overlapping uncertainty halos
  monotonically recede and no new halo is entered. Geometry remains intact.
  A finite initial probe must obtain fresh RGB evidence for further travel;
  contextual allowance includes already spent physical uncertainty debt and is
  never replenished by reference/goal renaming. Validate actual archived start
  admission, measured multi-action progress and changed-scene/poor-evidence
  refusals before freezing a fresh original-bowl-only medium subscription trial.
  Retain permitted RGB/robot inputs, zero demo, Direct5, actual LIBERO plus no
  forbidden contact, full video-excluded timing, no automatic cap/restart/queue
  or new Direct test. Offline gains do not establish successful-task speedup.

- V22 original-bowl zero-shot is terminal and incomplete:
  `runs/visual_field_v22_zero_shot_001/analysis.json`. Root reviewed repeated
  initial-geometry admission stagnation and stopped at23 complete transitions:
  two rejected Field selections,21 zero Direct holds,0 effective Field/gripper,
  LIBERO or forbidden collision. Seven valid medium subscription requests,
  API-key0; video-excluded196.719450s, model186.118579s (94.61%). Identical
  initial RGB/state to V21 produced a different model cabinet hypothesis whose
  52mm uncertainty halo contains the starting tool, despite27.894mm nominal
  separation. Both Field attempts failed before planning; the V22 recovery
  heading repair was not exercised. This is uncertain model geometry, not proof
  of physical collision or no escape. All143 sources/7 helpers stayed frozen,
  30/30 final protocol checks passed; cleanup/private shutdown verified. Preserve
  this attempt; no other scene or Direct ran. Analyze evidence-producing response
  to initially infeasible geometry before another original-only trial. Do not
  remove scene hypotheses or claim task speedup from offline/unfinished results.

- V22 preparation continues the active zero-shot Field speed goal after V21's
  verified recovery implementation defect. Preserve V21 terminal evidence.
  Use a continuous checked Field segment only for explicitly chosen short
  observed-motion reversals; preview through the same actual planner/tracker.
  Retain normal distance planning, current scene/contact/reference admission,
  direction limits and all physical budgets. Validate the recorded V21 heading
  and arbitrary directions/sub-grid endpoints, including changed-geometry and
  preview-purity counterexamples. Freeze then fresh original-bowl-only medium
  subscription trial, zero demos, permitted RGB/robot inputs, LIBERO plus no
  forbidden contact, video-excluded time. No Direct rerun, queue, automatic
  episode cap or restart. Full task speedup remains unproven.

- V21 original-bowl zero-shot is terminal and incomplete:
  `runs/visual_field_v21_zero_shot_001/analysis.json`. Root reviewed repeated
  offer/planner mismatch and stopped at232 complete transitions,226 effective
  Field commands,0 gripper/LIBERO/forbidden collision. Fifteen valid medium
  subscription requests,0 API-key calls; video-excluded attempt515.553507s,
  model383.121377s. Model copied the offered8mm reverse exactly, but the graph
  path produced +Z as its first command,36.54degrees from the checked diagonal
  and outside the30degree guard cone. Field/Direct retries remained vetoed.
  The single-heading offline positive missed this topology-dependent mismatch.
  Preserve this attempt; no other scene/Direct started. All140 sources/7 helpers
  frozen,29/29 final protocol checks passed, cleanup/private shutdown confirmed.
  Future recovery must preview the actual planner/tracker or use an explicitly
  checked heading-constrained route, with arbitrary-heading/sub-grid regression;
  preserve all scene/contact checks and debts. No completion/speedup established.

- V21 preparation continues the active zero-shot Field speed goal under
  `runs/visual_field_v21_zero_shot_001`. Maintain bounded causal appearance
  templates, compare consistent image objectives with metric waypoint progress,
  and allow only explicitly chosen, freshly checked finite reversals of observed
  worsening. Preserve physical debts, scene/contact/attachment constraints and
  all original hypotheses. Compact only request presentation, keeping raw logs.
  Validate and freeze before a fresh original-bowl-only trial with medium Astra,
  subscription only, permitted RGB/robot inputs, actual LIBERO plus no forbidden
  contact, and full video-excluded timing. No new Direct, scene queue, automatic
  episode cap or restart. Analyze an unsuccessful original result before any
  other scene; offline component gains do not establish task completion/speedup.

- V20 original-bowl zero-shot is terminal and incomplete:
  `runs/visual_field_v20_zero_shot_001/analysis.json`. Root reviewed four complete
  all-hold Direct blocks and requested a boundary stop; in-flight work ended at
  184 complete transitions:155 effective Field commands,3 rejection holds and
  26 Direct holds,0 gripper/LIBERO/forbidden collision. Video-excluded636.288169s;
  21 medium subscription requests (20 valid,1 same-observation stream retry),
  API-key0. Reference repairs repeatedly interrupted reuse. After178.7mm of
  Field travel, projected-neighbor worsening blocked every recovery candidate;
  six final hold replies cost125.932919s. Whole-tool rectangle expansion was a
  substantial contributor, not proof of physical collision. Metric waypoint
  improvement also disagreed with some matched-reference image objectives.
  The new weak-reference exception passed80 offline checks but never activated
  in this fresh run. All132 sources/7 helpers stayed sealed,25/25 final protocol
  checks passed, cleanup and private-supervisor shutdown verified. Preserve this
  attempt; no other scene or Direct run started. No successful speedup claim.
  Analyze causal template maintenance, consistent task-progress evidence and
  direction-dependent recovery before another original-only trial; no live edits.

- V20 preparation continues under `runs/visual_field_v20_zero_shot_001`.
  Only weak-image-response allowance exhaustion may be deferred for empty-tool
  fixed-world-goal tracking, with reliable references and no image retarget,
  contact refinement, payload or grasp candidate. Every actual <=5mm command
  must independently pass a fresh full-margin whole-scene sweep and remain
  outside the target contact band. Preserve monitor validity/debt, independently
  reject masked cumulative contradictions, and retain observed anchor-shift
  latches across reference retain. Validate archived candidate admission and
  one-request measured travel beyond30mm before freezing a fresh original-bowl
  zero-shot trial. Keep medium subscription only, permitted RGB/robot inputs,
  actual LIBERO plus no forbidden collision, video-excluded full timing,
  no new Direct, queue, automatic episode cap or restart. Analyze an incomplete
  original outcome before another scene. No speedup claim without completion.

- V19 original-bowl zero-shot is terminal and incomplete:
  `runs/visual_field_v19_zero_shot_001/analysis.json`. First model request drove
  28 effective Field actions /29.991mm through the exact full-margin model-sweep
  branch, preserving zero uncertain-geometry debt. A separate global30mm weak
  visual-response limit then vetoed Field before actual geometric selection,
  although the block used fixed-goal proprioceptive tracking and no visual
  retarget/contact alignment. Six later decisions produced only holds. Root
  reviewed stagnation and stopped at34 complete transitions,7 valid medium
  subscription requests,0 API-key calls, no grasp/LIBERO/forbidden collision.
  Video-excluded attempt228.380329s, model204.916736s. Preserve this measured
  component improvement; it is not task completion or speedup. Cleanup and
  private-supervisor shutdown succeeded. Analyze the actual control dependency
  before another fresh original-only trial; no queued scene or Direct rerun.

- V19 is being prepared under `runs/visual_field_v19_zero_shot_001` for the
  active zero-shot Field speed objective. Use exact current full-additive-margin
  whole-tool/payload candidate sweeps to distinguish model-remote projected
  overlaps from genuinely uncertain near regions. A declared spatial margin is
  not a measured statistical standard deviation; no physical-clearance claim
  follows from model separation. Retain all obstacle/unknown hypotheses, hard
  core/target/contact/attachment checks, observed worsening, reference repair
  and measured physical debts. Bind sweep evidence to exact current state,
  proposal and observation epoch; do not reuse it for another action or image.
  Validate real archived admission and changed-obstacle rejection, freeze, then
  fresh original bowl only with medium subscription and no demonstration.
  Actual LIBERO plus no forbidden contact remains completion; report full
  video-excluded control time. No new Direct, queued scene, automatic episode
  cap or restart; analyze any unsuccessful original result before continuing.

- V18 original-bowl zero-shot is terminal and incomplete:
  `runs/visual_field_v18_zero_shot_001/analysis.json`. Root reviewed four
  consecutive complete all-hold decisions and requested a boundary stop; the
  pending ninth request completed with one final hold. Fifty complete transitions,
  26 effective Field actions, three rejected Field holds, 21 zero Direct actions,
  no grasp/LIBERO/forbidden collision. Video-excluded attempt 262.919710 s;
  nine valid medium subscription calls, zero API-key calls. Early reference
  review selected a genuinely changing candidate and warm-started both models;
  fresh point-response validation occurred without resetting physical debt.
  The cumulative predictor required by risk admission never validated, so all
  translation spent finite uncertain-travel allowance and then stagnated.
  This is local evidence/admission exhaustion, not proven physical impossibility
  or successful-task speedup. Preserve V18; do not resume or advance scenes
  before analysis. Sources/helpers were frozen and private supervision stopped.

- V18 is being prepared for the active zero-shot Field speed objective in
  `runs/visual_field_v18_zero_shot_001`. Use passive observed-motion candidates
  for early model review of an uninformative controlled reference, bind the
  model-selected candidate to exact current pixels, and warm-start response
  training from recorded RGB/robot transitions. Historical candidate-conditioned
  samples are training only; fresh actual validation and all physical debts,
  scene/contact constraints remain. Freeze and test original bowl first with
  medium Astra subscription-only and no demo, depth, calibration or object truth.
  Require actual LIBERO plus no forbidden contact; include all inference/local
  work in video-excluded time. No automatic cap/restart/queue or Direct rerun.
  Preserve V17 and all previous outcomes; analyze an unsuccessful original-bowl
  outcome before other scenes and never infer speedup from incomplete time.

- V17 original-bowl zero-shot test is terminal and incomplete:
  `runs/visual_field_v17_zero_shot_001/analysis.json`. Root reviewed persistent
  admission stagnation and stopped at 71 complete transitions: 27 effective
  Field actions, three Field rejection holds and 41 zero Direct fallback actions;
  no gripper closure, LIBERO success or forbidden collision. Video-excluded
  attempt time is 368.1342 s; 13 medium subscription requests (one semantic
  repair), zero API-key calls. Partial scene support and actual candidate sweep
  checks worked, but the controlled visual response stayed unvalidated and
  both explored directions exhausted their allowance; remaining shared travel
  could not admit another current candidate. This is not a physical-path
  impossibility, timeout or successful-task speedup. All 121 source and seven
  helper hashes stayed unchanged; 34 final protocol checks passed with two
  explicitly logged historical presentation-label additions, no action-value
  changes. Cleanup succeeded and private supervision is shut down. Preserve
  this attempt; do not resume or advance other scenes before failure analysis.

- V17 preparation continues the authorized zero-shot Field speed objective.
  Preserve V16_001 numerical-defect and V16_002 incomplete artifacts. Distinguish
  current visible RGB support from complete object extent; keep all projected
  and world hypotheses and use full uncertainty in planning for partial objects.
  Replace whole-cube coverage with exact proposed known-tool/payload sweeps,
  preserve image guards/contact rules and log refused proposals. Compress only
  model-feedback presentation, preserving raw logs and current geometry/state.
  Validate useful Field reuse and obstacle refusals, seal all sources/helpers,
  then fresh original-bowl-only `runs/visual_field_v17_zero_shot_001`. Keep Astra
  medium subscription-only, zero demos, permitted RGB/robot inputs, Direct5,
  actual LIBERO plus no forbidden collision, video-excluded full timing,
  no automatic cap/restart/queue or new Direct trial. Analyze an unsuccessful
  original result before other scenes; offline evidence does not prove speedup.

- Corrected V16 original-bowl zero-shot attempt002 is terminal and incomplete:
  `runs/visual_field_v16_zero_shot_002/analysis.json`. Root reviewed repeated
  coverage interruption and stopped at24 complete transitions,17 effective Field,
  seven valid medium subscription requests,0 API-key calls, no grasp/LIBERO or
  forbidden collision. Video-excluded attempt210.7283s; model185.9695s (88.25%).
  Full inventory reached both planning and image checks, but reliable partial
  cabinet tracking triggered five whole-external-rectangle coverage vetoes.
  The quarter-pixel numerical repair is validated; enlarging its tolerance does
  not address the structural whole-cube coverage requirement. Preserve all full
  projected/world geometry and analyze actual candidate sweeps with partial-view
  support. A static tool-reference repair and final nominal-separation refusal
  also remain relevant. All114 source/input/route checks and seven helper hashes
  passed; cleanup succeeded and private supervisor shut down. No other scene or
  new Direct trial ran. Do not resume or infer task-speedup from incomplete time.

- V16 attempt001 is preserved as an implementation-defect boundary stop:
  `runs/visual_field_v16_zero_shot_001/analysis.json`. Ten complete transitions,
  two effective Field actions, no closure/LIBERO/collision;110.6578s excluding
  video, four medium subscription calls. All114 sources and seven helpers were
  frozen and audited; private supervision shut down. Cabinet tracking remained
  reliable but a0.027984 pixel image-edge overrun incorrectly invalidated its
  whole rectangle and starved Field. Correct only this numerical boundary case,
  preserving full projected/world geometry and rejecting meaningful out-of-view
  extents; validate then freeze a fresh original-only attempt002. Never overwrite
  attempt001 or infer completion/speedup from its truncated attempt time.

- V16 is being prepared under the continuing zero-shot Field speed objective.
  Use one persistent RGB-declared inventory for metric and image constraints,
  preserve omitted identities/world hypotheses, and require live two-view contact
  descriptors for near-target Field approach. Known robot parts, primary-target
  palm/contact and payload rules remain. Inventory completeness remains a model
  claim, not a detector or clearance certificate. Validate useful clear Field
  reuse plus missing-coverage/obstacle rejection, freeze sources/helpers/inputs,
  then run a fresh original-bowl-only trial in `runs/visual_field_v16_zero_shot_001`.
  Preserve V15 failure and all earlier artifacts. Keep medium Astra subscription
  only, allowed RGB/robot inputs, Direct5, actual LIBERO plus no forbidden contact,
  no automatic cap/restart, private supervision and video-excluded full timing.
  Analyze any unsuccessful original-bowl outcome before other scenes; no new
  Direct run or successful-speedup claim follows from offline tests alone.

- V15 original-bowl zero-shot is terminal `collision_failure`:
  `runs/visual_field_v15_zero_shot_001/analysis.json`. At action235 the left
  finger contacted neighboring ramekin;236 complete transitions,232 effective
  Field actions,0 Direct/gripper actions, no LIBERO success. Attempt347.9394s
  excluding video,15 medium subscription requests and0 API-key calls; model
  pipeline281.8610s. Motion diagnostic helped repair a stationary reference and
  enabled sustained Field approach, but every prediction declared no relevant
  neighbors and empty obstacle/unknown lists. Only the primary-target geometry
  entered the planner, so the ramekin had no representation and the final
  no-near-conflict admission did not establish clearance. All15 responses left
  contact_alignment null. Directional recovery debt stayed unused. All108
  sources and7 helpers remained frozen; cleanup passed and private supervisor
  shut down. Preserve this scored failure; do not resume or automatically advance
  scenes. Analyze obstacle inventory/coverage and near-contact identity before
  another trial; no successful-task speedup has been established.

- V15 is being prepared under `runs/visual_field_v15_zero_shot_001` for a fresh
  original-bowl zero-shot test after validation. Separate physical-direction
  experiment accounting from visual-reference revisions without resetting debt,
  preview candidate eligibility through the actual guard, and show causal recent
  RGB change as ambiguous evidence for model reference repair. Preserve V14
  artifacts and all robot/contact/input constraints, medium subscription-only,
  Direct5, qualified LIBERO success, no automatic episode cap or restart, private
  supervision and video-excluded timing. Freeze source/helper/input hashes before
  launch. No other scene or Direct rerun until original-bowl outcome analysis.

- V14 original-bowl zero-shot is terminal and incomplete:
  `runs/visual_field_v14_zero_shot_001/analysis.json`. Operator-reviewed recovery
  stagnation stopped at89 complete transitions:68 effective Field actions,
  5 rejected Field holds,6 Direct actions and10 open-gripper actions, with no
  closure, LIBERO success or forbidden collision. Attempt327.7484s excluding
  video;13 medium subscription requests (11 valid decisions,2 semantic repairs),
  0 API-key calls. Model pipeline294.9703s (90%). Changed-direction Field recovery
  executed, but a reference revision split +Z into two trial records: three
  experiment slots covered only two physical directions, blocking another
  nominally offered direction despite remaining shared travel allowance. Final
  visual response was unvalidated. All104 source and7 helper hashes unchanged;
  cleanup passed, private supervisor shut down. Do not resume or advance scenes
  automatically. Analyze separation of reference repair from spatial experiment
  accounting and align offered candidates with actual admission. No successful
  task-speedup is established; preserve this attempt and all prior versions.

- V14 is being prepared for one fresh original-bowl zero-shot Hybrid test under
  `runs/visual_field_v14_zero_shot_001`. It reserves persistent uncertain-travel
  allowance for bounded directional evidence, adds passive RGB reference-support
  diagnostics and compacts repeated feedback. These are conditional model/RGB
  hypotheses, not calibrated clearance. Validate positive evidence-producing
  recovery as well as vetoes, then seal all runtime/helper/input files before
  launch. Preserve V13 results, medium Astra subscription-only, allowed RGB/robot
  inputs, Direct5, actual LIBERO success plus no forbidden contact, no automatic
  episode cap/restart, private supervision and full video-excluded control timing.
  Run original bowl first; analyze any unsuccessful outcome before other scenes.
  No new Direct trial or measured task-speedup claim is authorized by preparation.

- Corrected V13 original-bowl zero-shot is terminal and incomplete:
  `runs/visual_field_v13_zero_shot_002/analysis.json`. The operator stopped a
  reviewed admission deadlock at31 complete transitions:20 effective Field
  actions,5 rejected Field holds,6 unchanged zero Direct holds, no gripper
  closure, no LIBERO success and no forbidden collision. Seven medium Astra
  subscription requests,0 API-key calls; video-excluded attempt196.8692s,
  model pipeline179.7372s (91.3%). Initial Field reuse worked, but exhausted
  observation allowance plus projected cabinet ambiguity refused upward and
  horizontal recovery; later static holds could not acquire directional evidence.
  All7 responses left contact_alignment null, so contact refinement and carried
  geometry were not exercised. All99 source/input/route/command checks and7
  helper hashes passed; cleanup succeeded and the private supervisor is shut
  down. Do not resume or automatically advance other scenes. No completion
  speedup is established. Next design must resolve evidence-seeking recovery
  admission without deleting scene hypotheses or bypassing collision evaluation.

- V13 attempt001 is separately preserved as an implementation-defect stop:
  `runs/visual_field_v13_zero_shot_001/analysis.json` (4 complete Field actions,
  119.9974s video-excluded; no LIBERO success/collision). Target-mask rebuilding
  had incorrectly copied axis-only blocked nodes into the underlying obstacle
  grid. The narrow repair unwraps axis decorators, preserves real obstacle and
  unknown masks, then reapplies both corridors. Recorded17.682micrometre lateral
  drift now continues within the3mm allowance. Corrected final198 offline checks
  passed before attempt002; attempt001 frozen sources and results remain intact.

- V13 preparation is under `runs/visual_field_v13_zero_shot_001`; it is not
  launched or sealed. This is the authorized original-bowl zero-shot Hybrid
  retry only, with no queued scene or new Direct baseline. V13 requires the
  primary target's model geometry independently of nearby obstacles, preserves
  palm constraints under finger contact, and adds optional two-view physical
  contact-to-pinch Field refinement. Keep medium Astra subscription-only, the
  existing allowed RGB/robot inputs, Direct5, actual LIBERO success plus no
  forbidden contact, and full video-excluded control timing. Complete offline
  validation and seal sources/helpers before launch. Preserve V12 artifacts;
  do not claim V13 task success or speedup from offline tests.

- V12 original-bowl zero-shot is terminal `collision_failure`:
  `runs/visual_field_v12_zero_shot_001/analysis.json`. All704 transitions are
  complete; action703 touched target bowl with `gripper0_hand_collision`, a
  forbidden nonfinger contact. LIBERO false. There were616 effective Field
  actions,67 gripper actions,10 unchanged Direct actions, two closures and no
  validated attachment/transport. Video-excluded attempt time742.6961s; model
  pipeline619.9184s. All94 runtime/source/input/route checks passed; seven helper
  hashes unchanged, cleanup succeeded, private supervisor shut down. Preserve
  this scored failure; do not resume or automatically advance scenes. Field
  reuse improved, but final target-bowl contact geometry was absent from both
  metric regions and image-neighbor checks. Analyze contact-specific robot-part
  constraints and grasp targeting before another trial; no task speedup is proven.

- V12 is prepared for the user's fresh test request under
  `runs/visual_field_v12_zero_shot_001`. Validate and freeze sources before one
  original-bowl zero-shot Hybrid trial. V12 separates command metres from measured
  travel using an empirical robot response, accumulates causal RGB evidence over
  distinguishable displacement, and permits finite nominal-clear observation of
  projected ambiguity while rejecting observed approach. Current-observation
  bounded Direct contact corrections remain available without stale Field geometry.
  Model geometry and image checks are hypotheses, not clearance certificates.
  Keep medium Astra subscription-only, allowed RGB/robot inputs, Direct5, actual
  LIBERO success plus no forbidden contact, no automatic episode cap/restart,
  private UNIX-socket supervision, and video-excluded full control timing.
  Do not advance scenes after an unsuccessful original-bowl outcome without
  analysis; preserve historical attempts and do not rerun Direct.

- V11 original-bowl zero-shot test is terminal and incomplete:
  `runs/visual_field_v11_zero_shot_001/analysis.json`. Operator-reviewed admission
  deadlock stopped it at 24 complete transitions / 9 valid medium subscription
  decisions. Eleven accepted Field actions moved only 4.0778mm; eight Field
  rejections and five zero Direct holds followed, with no gripper action,
  LIBERO success or forbidden contact. Video-excluded attempt time254.3576s,
  of which model time239.2243s. Projected hotplate/cabinet ambiguity persisted
  after the finite cold-separation allowance became smaller than1mm; same-track
  envelope union also delayed image-box correction. Final reseeding removed the
  outside-view condition but not the unresolved ambiguity/consumed allowance.
  All87 source/input/route/command checks and6 helper hashes passed; cleanup
  succeeded and private supervisor is shut down. Preserve the attempt and do not
  resume or automatically advance scenes. Analyze revision and informative-motion
  design before a fresh trial. Audit also found a command/response scale mismatch:
  largest actual cold step0.4361mm gives a supported prediction horizon0.6542mm,
  below the executor's1mm command floor; held-out predictive improvement was real
  but below the declared gate. No successful-task speedup has been demonstrated.

- The user requests testing the improved method on the existing tasks. V11 is
  prepared under `runs/visual_field_v11_zero_shot_001`: known robot tool-part
  configuration obstacles, live RGB whole-tool/neighbor translation screening,
  persistent Field subgoals and compact observable feedback. These are model
  hypotheses and empirical image checks, not calibrated 3D clearance. Run one
  frozen, privately supervised original-bowl zero-shot trial first, medium Astra
  through subscription only. Keep permitted RGB/robot inputs, Direct5, actual
  LIBERO plus no forbidden contact, full video-excluded control timing, and no
  automatic episode cap/restart. Analyze any unsuccessful original-bowl outcome
  before other scenes; preserve previous attempts and do not rerun Direct.

- V10 original-bowl zero-shot test is terminal collision_failure:
  `runs/visual_field_v10_zero_shot_001/analysis.json`. At the139th action the
  right finger touched the neighboring ramekin; LIBERO remained false. All139
  actions were effective Field motion, with9 medium subscription responses,
  no Direct/gripper actions and218.1156s video-excluded attempt time. Full81
  source/input/route/transition checks passed; cleanup and video finalization
  succeeded; its private supervisor is shut down. Persistent goals worked,
  but the model obstacle/tool envelope and finite soft-risk cost admitted a
  vertical route through uncertainty. Attachment changes were not exercised.
  Preserve this scored failure; do not resume it or advance other scenes
  automatically. Analyze geometry and execution permission before another
  fresh original-bowl test. Goal remains active; no task-speedup is established.

- V10 is being validated for a fresh original-bowl zero-shot test under
  `runs/visual_field_v10_zero_shot_001`. Keep persistent world subgoals across
  repairs, displacement-adaptive fresh attachment evidence, and explicit
  below-1mm command handoff. Previously observed attachment permits only a
  finite 5--15mm measured uncertainty allowance estimated from fresh image
  response; requests/renaming/holds never renew it. Preserve V9 risk geometry,
  hard unknowns, permitted RGB/robot inputs, medium subscription-only, Direct5,
  actual LIBERO plus no forbidden contact, and no automatic episode cap.
  Run one frozen original-bowl trial under private UNIX-socket supervision with
  no automatic restart; persist atomic timing checkpoints. Do not resume V9 or
  advance other scenes after an incomplete outcome without analysis. Offline
  regressions and archive replay do not establish task success or speedup.

- V9 original-bowl trial is terminal and incomplete:
  `runs/visual_field_v9_zero_shot_001/analysis.json`. The durable prefix has657
  complete action/execution/feedback triples,349 effective moving Field actions,
  57 valid subscription responses and request58 without a response/result. No
  recorded LIBERO success or forbidden contact. Operator requested a boundary
  stop after reviewing repeated attachment-monitor command starvation; however
  the execution session reported143 and no summary or operator-stop acknowledgement
  was finalized. Do not claim a clean user_stop, timeout, exact elapsed completion
  time or speedup. Sources77/input/wire/commands match; runner/workers are absent.
  Exact video-excluded time is unavailable; recorded disjoint model+local decision
  components supply only a981.9263s lower bound. Raw state remains stale running;
  original partial video is preserved and a labeled posthoc replay was recovered.
  No other scene or Direct trial ran. Next work should address displacement-aware
  attachment evidence, ineffective tiny-command loops and persistent world subgoals
  across repairs; preserve V9 risk geometry and all actual success/collision rules.
  Do not resume this episode or advance scenes automatically. Goal remains active.

- V9 is opt-in model-risk planning, not a collision certificate. Localized visible
  objects keep nominal boxes plus tool radius as hard cores; their full declared
  position uncertainty contributes nonnegative path cost. Genuinely unknown
  regions remain fully hard inflated. Missing/unobservable/inconsistent current
  RGB assessment falls back to strict full inflation; no automatic reclassification
  or carving is allowed. V8 measured orientation, correspondence/attachment guards,
  Direct5 and explicit gripper control remain. Keep original-bowl-first zero-shot,
  medium subscription-only, permitted RGB/robot inputs, LIBERO plus no forbidden
  contact, no fixed episode cap and video-excluded full control timing. A manually
  created operator_stop.request.json now stops at a full transition boundary after
  evaluator checks. Pending fresh test: runs/visual_field_v9_zero_shot_001. Offline
  risk-field regression does not establish task success or 20–50% acceleration.

- V8 original-bowl zero-shot trial is stopped and incomplete:
  `runs/visual_field_v8_zero_shot_001/analysis.json`. Thirteen valid subscription
  decisions produced 48 executed actions: four rejected Field holds and 44 zero
  Direct holds, with no grasp, effective Field motion, LIBERO success or forbidden
  collision. Video-excluded attempt time is 241.2079 seconds. The operator
  stopped reviewed planning stagnation, not an episode timeout: every Field
  start was enclosed by a model cabinet uncertainty envelope, before graph
  search. All 74 source/input/route checks passed. SIGINT arrived after the last
  execution log and before its feedback record, so preserve 48 executions and
  47 complete transition triples; do not synthesize feedback or label the full
  transition audit complete. Cleanup passed. Do not resume or advance scenes.
  Next work must address metric-hypothesis feasibility/uncertainty representation,
  preserving obstacle evidence rather than deleting regions to force a path.
  The separately validated V8 pose-hold and weak-response fixes do not establish
  successful-task acceleration; the full speed goal remains active and unproven.

- The continuing speed-improvement goal authorizes opt-in V8 and a fresh
  original-bowl zero-shot test after validation. V8 distinguishes weak visual
  response from contradiction, explicitly replaces stale OSC orientation goals
  with bounded measured-pose holding, and screens model-directed rotation using
  declared RGB tool/neighbor rectangles. These are uncertain image hypotheses,
  not calibrated geometry or collision certificates. Every named Field neighbor
  must enter its obstacle/unknown hypotheses; unobservable scenes cap reuse at
  15 mm measured travel and 5 mm per command. Accepted Direct5 is unchanged;
  rejected blocks execute only a logged current-pose orientation-target refresh.
  Preserve all prior trials. Keep medium subscription-only, zero-shot, allowed
  RGB/robot inputs, actual LIBERO plus no forbidden contact, no episode cap and
  video-excluded full control timing. Test original bowl only; analyze an
  unsuccessful result before any other scene. Controller probes and archive
  replay do not establish the successful-task speed objective.

- V7 original-bowl zero-shot trial is terminal with a forbidden-contact failure:
  `runs/visual_field_v7_zero_shot_001/analysis.json`. It executed 388 actions
  (323 Field selections, 310 effective moving Field, 65 Direct reorientation,
  zero gripper closures). At zero-based action 387 the right finger contacted
  the neighboring ramekin. LIBERO stayed false. Video-excluded attempt time is
  731.7144 seconds; 46 subscription requests produced 45 valid decisions and
  one missing-completion retry. All 69 source hashes and original inputs
  matched; cleanup passed. Preserve this failure and do not resume it or
  automatically advance other scenes. Actual approach motion occurred, but
  repeated feature repair and orientation recovery prevented grasping. Analyze
  weak-evidence rejection and measured orientation holding before another
  implementation/test; the successful-task speed goal remains unproven.

- The continuing active speed-improvement goal authorizes opt-in V7 and a fresh
  original-bowl zero-shot retry after meaningful validation. Address V6's wrong
  controlled-feature correspondence with recent motion evidence, persistent
  per-correspondence unvalidated travel, and derived RGB reference diagnostics.
  Distinguish navigation alignment/stall from grasp-feature engagement. Keep
  Field as the bulk-motion planner, V6 axis constraints and measured gripper
  settling, Direct5 for necessary contact correction, and all collision/input
  boundaries. V7 may retry stream_missing_completion once on the SAME observation
  before any action, with every attempt timed/logged/counted; no model/API-key
  fallback or broad auth/quota retry. Preserve the stopped V6 trial. Start only
  original bowl, analyze any incomplete outcome before other scenes, retain no
  fixed episode cap and actual LIBERO plus no forbidden contact. Neither offline
  regression nor an interrupted attempt proves the 20–50% task-speed objective.

- V6 original-bowl zero-shot trial is now terminal and incomplete:
  `runs/visual_field_v6_zero_shot_001/analysis.json`. At 383 actions (338 effective
  moving Field, 5 Direct, 34 gripper-settle), the 26th subscription request ended
  with APIError `stream_missing_completion`. Preserve raw subscription_error;
  this is not a local invalid-response/schema failure or a completed task.
  Video-excluded attempt time is 414.3314 seconds, LIBERO false and no monitored
  forbidden contact. First grasp was unsupported by its lift test; recovery
  had begun. Early RGB tracking followed stationary robot-base features rather
  than the gripper; later reseeding enabled longer Field execution. All 65
  source hashes and original inputs matched, cleanup passed. Do not resume this
  run or automatically advance other scenes. The current branch is analysis;
  successful-task speed improvement remains unproven.

- The fresh V6 original-bowl zero-shot test is authorized after offline validation.
  Use `runs/visual_field_v6_zero_shot_001`, with frozen original inputs and sources.
  V6 adds persistent explicit visual references, windowed payload/tool evidence,
  axis-constrained Field motion, and explicit model-commanded local gripper
  settling. The latter is a separate controller, not Direct5 or a grasp certificate.
  Keep Astra medium via subscription only, no API fallback, RGB plus measured
  robot state/joints, no depth/calibration/object truth in policy, and actual
  LIBERO success with no forbidden contact. Report full video-excluded control
  time. Start original bowl only; analyze an unsuccessful outcome before other
  scenes, without an automatic queue or resuming any old episode. There is no
  fixed episode cap. A reviewed persistent-stagnation stop remains incomplete.
  Offline regressions and archived RGB replay do not establish task speedup.

- The repaired V5 original-bowl zero-shot retry is stopped and incomplete:
  `runs/visual_field_v5_zero_shot_002/analysis.json`. It executed 423 actions
  (192 effective moving Field actions, 225 Direct actions), reached observed
  payload/tool co-motion, Field transport and release, but LIBERO stayed false
  without monitored forbidden contact. Video-excluded attempt time is 973.4511
  seconds. The operator stopped reviewed post-release task stagnation: repeated
  model claims of correct support and changed views/holds without corrective
  object manipulation. Actual view changes occurred, so this is not a proven
  zero-information timeout. Preserve raw user_stop and both V5 attempts; do not
  resume the episode or advance other scenes automatically. Posthoc evaluator
  replay is analysis only and must never feed hidden state into the policy.
  The role-contract repair worked; remaining issues include fragile image
  attachment evidence, landmark refresh continuity, and false visual placement
  completion. The overall successful-task speedup remains unproven.

- The first V5 original-bowl test is finished and incomplete at 207 actions,
  `runs/visual_field_v5_zero_shot_001/analysis.json`: 512.76 seconds excluding
  video, no LIBERO success and no detected forbidden contact. The final local
  SchemaError came from conflicting visual roles: a payload test-lift needed
  a stationary external anchor different from its interaction target. Preserve
  the raw broad subscription_error status; this was a local protocol defect,
  not evidence of authentication/quota failure. The continuing improvement goal
  permits repairing this general role contract, meaningful regression tests,
  and a fresh original-bowl V5 retry. Do not resume the old episode or advance
  other scenes without original-bowl success. Keep Field navigation-target
  identity checks, independent payload/tool evidence, all collision constraints
  and medium subscription-only settings. A retry remains a separate result and
  does not establish a successful-task speedup unless actually completed.

- The continuing active speed-improvement goal authorizes a new opt-in v5
  implementation and fresh zero-shot tests after validation. Address the v4
  near-contact plateau with bounded multi-feature RGB tracking, model-directed
  Direct5 interaction handoff, locally computed measured orientation hold, and
  useful Field motion up to 120 actions within unchanged spatial/obstacle
  limits. These are general robot-control changes, not task-specific poses.
  Preserve all frozen runs; do not resume stopped episodes or old scene queues.
  Start from the original bowl again after meaningful offline validation, and
  analyze failure before further scenes. Keep medium subscription-only, allowed
  RGB/robot inputs, actual LIBERO plus no forbidden contact, and video-excluded
  full-task timing. Success and a legitimate Direct comparison remain required
  before claiming the overall 20–50% speed objective is achieved.

- The repaired fresh v4 original-bowl test is stopped and incomplete:
  `runs/visual_field_v4_zero_shot_002/analysis.json`. It executed 606 actions,
  including 553 effective Field actions (12 blocks with at least 30, longest 46),
  but never closed the gripper. Video-excluded attempt time was 854.63 seconds;
  LIBERO false, no monitored forbidden contact. Repeated near-contact external
  tracking loss and retreats prevented transition into grasp. The operator
  stopped after reviewing persistent task stagnation; raw `user_stop` is kept.
  Some descent and numerical evidence occurred, so this is not a proven
  zero-information timeout. All tested sources stayed frozen, cleanup passed,
  and no other scene or Direct trial was started. Do not resume this episode or
  its prepared scenes implicitly. The overall speed-improvement goal is not met.

- The user requested testing the improved method on the existing tasks. Repair
  the confirmed v4 SO(3) discontinuity, uncertainty-budget reset and model box
  representation defects before a fresh v4 retry; preserve prior frozen runs.
  Start zero-shot Hybrid on the original bowl. Advance to the other prepared
  scenes only after qualified success; stop and analyze a basic-task failure.
  Keep medium Astra via subscription only, allowed RGB/robot inputs, evaluator
  LIBERO success and forbidden-contact checks, and video-excluded timing. No
  Direct trial is needed for this test request. Keep the speed-improvement goal
  active until verified; implementation tests alone do not establish speedup.

- The authorized v4 original-bowl trial ended incomplete at 115 actions:
  `runs/visual_field_v4_zero_shot_001/analysis.json`. The tenth returned model
  response had reversed obstacle bounds and failed local schema validation;
  raw `subscription_error` is preserved but this was not quota/auth/network
  failure. It executed 105 effective Field actions, including two 30-action
  blocks, without LIBERO success or monitored forbidden contact. The frozen
  visual monitor also has a confirmed near-pi rotation-vector subtraction bug
  that resets evidence/bootstrap accounting. No further scenes or Direct runs
  were launched, no episode was resumed, and tested sources stayed unchanged.
  Analyze/repair before another fresh test; this attempt is not a completed
  task speedup and must not be relabeled successful.

- The user now authorizes live testing of v4 on the existing tasks. Run fresh
  zero-shot Hybrid, original bowl first, then the other prepared scenes only if
  the basic task succeeds. Preserve the prior stop-and-analyze branch on failure
  or reviewed persistent task stagnation; no implicit retries, no resumed old
  queues and no Direct reruns. Freeze source/input hashes, use medium Astra via
  subscription only, evaluator LIBERO plus no forbidden contact, and exclude
  video recording from reported control time. A supervised stagnation stop is
  incomplete, not official benchmark timeout or collision failure.

- The latest request authorizes improving the method to reduce GPT robot-control
  time through field planning. Implement opt-in v4 with bounded metric scene
  hypotheses, cached distance planning and measured-pose tracking; external RGB
  validates the controlled feature while wrist RGB observes target bearing.
  Preserve uncertainty and obstacle constraints, Direct5 contact control, medium
  subscription-only, RGB plus measured robot state/joints, and evaluator-only
  LIBERO/collision outcomes. Validate meaningful Field reuse and real archived
  RGB before claiming improvements. Do not resume the stopped v3 trial or the
  canceled multi-scene queues. Offline fake-inference counts are not measured
  real-task speedups; retain historical code, traces and results.

- The authorized v3 original-bowl zero-shot test is now stopped and incomplete:
  `runs/visual_field_v3_zero_shot_001/analysis.json`. It executed 200 Direct
  actions, zero Field actions, with no LIBERO success or monitored forbidden
  contact. The operator stopped persistent task/method stagnation; raw
  `user_stop` is preserved. Numerical information gains did occur, so this is
  not a mechanically proven zero-information timeout. No other scenes or Direct
  trials were launched. Do not resume this trial or advance its prepared inputs
  automatically; the user's failure branch now calls for analysis.

- The latest user authorizes live ZERO-SHOT Hybrid v3 testing, starting with the
  original bowl. If that trial fails, stop and analyze rather than continue to
  other scenes. Preserve medium Astra/subscription-only, the allowed RGB/robot
  inputs, actual LIBERO-only completion, and collision failure. Record any
  externally observed persistent-stagnation stop separately from benchmark
  collision failure; never call model-declared completion success. Do not resume
  canceled v2 queues or rerun Direct. Freeze the tested v3 source and new outputs.

- The user now authorizes implementing the next reliability improvement as
  opt-in Visual Field v3: feasibility preflight, separate obstacle/visibility
  constraints, persistent per-view response learning, bounded local observation
  intents, and evidence-based recovery progress. Validate with saved traces and
  offline tests first. Do not launch or resume any live task trial or queue in
  this implementation turn. Preserve v1/v2 and frozen historical artifacts;
  retain zero-shot Hybrid, medium subscription-only, RGB plus measured joints,
  actual LIBERO completion, collision failure and video-excluded timing rules.

- The user subsequently canceled continuation to other tasks and requested
  method-improvement analysis focused on the basic bowl failure. Stop the active
  relocated-bowl v2 trial and its continuation queue; do not start obstacle or
  drawer trials. Preserve both interrupted attempts and frozen artifacts. This
  is a discussion/analysis request, not authorization to launch new experiments.

- The user explicitly stopped ONLY the original-bowl Visual Field v2 trial and
  authorized continuing the remaining three zero-shot Hybrid v2 scenes. Preserve
  its `user_stop` outcome as incomplete. The relocated bowl, obstacle bowl and
  original drawer retain no episode cap, actual LIBERO-only completion, and the
  existing forbidden-contact failure rule. Keep frozen policy/input hashes;
  record this one-trial continuation authorization without relabeling outcomes.

- The latest request authorizes implementing opt-in Visual Field v2 reliability
  improvements, then testing ONLY zero-shot Hybrid on original bowl, relocated
  bowl, obstacle bowl and original drawer (four fresh trials). Do not rerun
  Direct: retain existing Direct records as historical references and disclose
  differences in interface/prompt rather than claim a fresh matched comparison.
  Only actual LIBERO success marks completion; model claims/verification labels
  never stop an episode. Preserve the existing forbidden-contact failure rule,
  no episode caps, medium Astra/subscription-only, RGB plus measured robot joints
  without supplied depth/calibration/object truth, and video-excluded timing.
  Preserve v1 and all previous trial artifacts. Local task-stagnation detection
  must trigger observable recovery/reobservation, not an automatic episode stop.

- After the four missing historical cells were tested, the user requested one
  fresh retry of new-layout bowl Zero-shot Hybrid. Reuse the exact old distance
  method, frozen layout/reset, prompts and medium subscription-only settings from
  `historical_zero_shot_missing_001/relayout_bowl_zero_hybrid`. Keep LIBERO-only
  stopping without an episode cap. Save the retry separately and preserve the
  earlier user-stopped attempt and the completed Direct reference; do not resume
  other trials or the new visual-field matrix.

- The user stopped the new visual-field matrix, then authorized ONLY filling the
  four missing cells in the previous 3D distance-method results table: original
  bowl zero-shot Direct, relocated bowl zero-shot Hybrid and Direct, and original
  drawer zero-shot Direct. Preserve all filled cells and do not resume the new
  visual-field matrix. Reuse the frozen relayout artifact and historical RGB+8D
  interface, medium Astra/subscription-only, five unchanged Direct actions,
  shared original no-demo task guidance, LIBERO-only completion and no episode
  caps. Report video-excluded time. The opt-in `--historical-zero-shot-direct`
  interface supplies the missing pure Direct baselines without changing the
  previous Hybrid or strict obstacle protocols. These supplementary runs are
  recorded later than the existing original-scene Hybrid runs.

- The user now authorized implementing generic RGB visual distance/risk-field
  control and rerunning original bowl, relocated bowl, obstacle bowl and original
  drawer in BOTH zero-shot and one-shot modes, Hybrid versus Direct (16 fresh
  trials). Measured robot arm joint positions/velocities are allowed; depth,
  supplied camera/environment calibration, privileged object poses and evaluator
  contact/success signals remain excluded from policy inputs. New opt-in runner:
  `scripts/evaluate_visual_field_subscription.py`; matrix:
  `runs/visual_field_matrix_v1/matrix.json`. Both arms share observable tracking
  feedback; label the baseline grounded Direct5, whose actions execute unchanged.
  Use medium Astra, subscription OAuth only, frozen paired inputs/demonstrations,
  no episode cap, LIBERO plus no forbidden contact, and video-excluded timing
  including all local perception/planning. Original scenes are distribution
  controls; do not label their small demo/test initialization differences OOD.
  Preserve all older protocols/artifacts. New image fields have pixel/cost units,
  not calibrated metric obstacle geometry or certified 3D collision safety.

- The latest request is a fresh ZERO-SHOT Hybrid-versus-Direct pair on the same
  frozen ketchup/milk obstacle scene. Remove all external demonstration input and
  explicitly state zero-shot and no unintended collisions in BOTH prompts. Keep
  Astra medium, subscription OAuth only, Direct five actions, shared task guidance,
  live feedback summaries and video-excluded timing. Intentional finger/bowl grasp
  and bowl/table or bowl/plate support contacts are allowed; robot/held-bowl contact
  with other scene geometry is forbidden. Use evaluator-only physics-substep
  contact monitoring, terminate and score failure on the first forbidden contact,
  and require the LIBERO goal with no collision for task success. Contact identities
  and hidden scene state must not enter policy input. Preserve historical protocols
  and do not impose a time/action cap or count failed attempt times as speedups.

- The user explicitly clarified that BOTH new bowl-obstacle rollouts count as
  failures because they collided with the ketchup bottle. For these obstacle
  evaluations, task success requires the original LIBERO goal AND no collision
  with the ketchup obstacle during execution. Preserve raw LIBERO flags and
  immutable rollout evidence, but give collision-aware reviewed outcomes priority
  in result/comparison reports. Failed runs' elapsed times are attempt durations,
  not successful completion times; exclude this pair from successful-task speed
  comparisons. Retrospective video review is not an automatic contact detector.

- The user requested a harder bowl scene with different distractor objects and
  selected path obstacles as the added difficulty. Preserve the bowl, target
  plate, original language/goal, language anchor ramekin, robot initial state,
  and frozen original demonstration. Validate a deterministic custom BDDL and
  matching joint-mapped state before inference; freeze both artifacts identically
  for fresh Hybrid and Direct runs. Keep obstacles out of model input except via
  RGB. Label results custom-scene evaluations. All current medium/subscription,
  no-cap LIBERO-only stopping and video-excluded timing rules still apply.

- For Hybrid-versus-Direct timing comparisons, the user now explicitly excludes
  video-recording overhead. The primary reported metric is
  `control_elapsed_s - timing_components.video_render_s`. Preserve raw recorded
  wall times and video timings; label retrospective subtraction as adjusted
  accounting, not a new rollout with recording disabled. Use completed matched
  task/layout pairs and keep incomplete or unmatched attempts separate.

- The user subsequently requested a new bowl-task layout with the same objects
  at changed positions, then fresh one-demonstration Hybrid and Direct tests.
  Use one deterministic, validated custom initial-state artifact derived from
  the original task; freeze the exact same layout in both arms. Keep the original
  bowl task language and LIBERO goal predicate, unchanged robot initial state,
  and the previous frozen official demonstration (do not regenerate it for the
  new layout). Label this as a custom-layout evaluation rather than an official
  initial-state benchmark. Layout/object-state inspection is evaluator-only;
  model input remains the demonstration, RGB and observable robot state/history.
  Keep Astra medium, subscription-only inference, Direct five actions, shared
  guidance, and LIBERO-success-only stopping without an episode cap.

- The user also requested one-demonstration tests of BOTH distance-field Hybrid
  and pure Direct on the original bowl-to-plate task: bowl_plate,
  LIBERO-Spatial task 0 / initial state 0 / seed 7. Use the same independently
  selected official demonstration in both fresh runs. Keep medium effort,
  subscription-only inference, shared observable task guidance and feedback
  summaries, Direct five actions, and actual LIBERO-success-only stopping with
  no episode cap. Preserve historical drawer and no-demonstration protocols.

- The user requested a pure Direct retry under the same one-demonstration drawer
  settings. Reuse the exact frozen reference, task/init/seed, Astra medium,
  subscription-only route, observable feedback summaries and shared task guidance
  from the completed Hybrid run. Execute five model-generated actions per request
  without local field planning or action replay; stop only on LIBERO success,
  user interruption or infrastructure/subscription failure, with no episode cap.
  Compare measured control-loop time; incomplete attempts remain unscored.

- The user authorized adding one official LIBERO demonstration as model input
  alongside current observations, then retrying drawer task D. Freeze one
  independent dataset episode with only paired RGB, robot state and actions;
  distinguish post-action demonstration observations from live observations.
  Keep the reference separate from live history and never auto-replay its actions
  or use its completion as evaluation success. Preserve Astra medium, subscription
  OAuth only, current field/Direct settings and LIBERO-success-only stopping.
  The earlier drawer attempt was explicitly stopped by the user and remains
  incomplete; it is not a successful timing baseline.

- The user selected task D and authorized adapting and running the next distance-field
  controller on `libero_goal`, task 3, initial state 0: "open the top drawer and put
  the bowl inside". Use explicit `--task drawer_bowl` task-aware prompts/routes and
  truthful task metadata; keep the historical bowl task as the default. Preserve
  Astra medium, subscription OAuth only, Direct five actions, 33³ / 5 mm and
  LIBERO-success-only stopping without an episode cap. For this cabinet task,
  request model-directed recovery instead of automatic upward reobservation.
  Opening is part of the requested sequence; do not add drawer-closing or release
  requirements to the official evaluator's containment predicate.

- The user subsequently authorized implementing the next opt-in `--distance-field-control`
  version: cached Dijkstra goal-distance planning, observable paired-RGB feedback with
  calibrated bounded XY goal updates, conditional Field reuse (30 up to 120 actions),
  and placement observation/recovery. This refresh limit is not an episode cap.
  Keep Astra medium, subscription OAuth only, Direct five actions, 33³ / 5 mm,
  evaluator-only LIBERO success and no episode cap. Preserve historical protocols.
  RGB tracking/calibration may be unavailable; never present image agreement as
  task success or hidden object measurements. One bounded open-gripper upward view
  recovery may be used inside predicted-valid space, separately logged from model
  intent. Implementation validation is not a successful full-task timing result.

- The user subsequently authorized implementing the opt-in `--field-improvements` variant: observable RGB goal-consistency checks, rotation-compensated held-object placement targets, and variable-length analytic paths for pure goal fields. This supersedes fixed ten-segment/thirty-action cadence only for this variant: paths have at most ten segments, local tracking at most ten actions, and model refresh at most thirty Field actions or earlier on measured arrival/boundary/visual contradiction/invalid direction/no progress. An early refresh is not an episode stop. Keep Astra medium, ChatGPT subscription OAuth only, Direct five actions, 33³ / 5 mm grid and actual LIBERO-success-only stopping with no episode cap. Preserve old protocols and historical artifacts. Implementation/offline validation does not establish a new full-task performance result.

- The user explicitly rejected the proposed 15-minute cap and chose to keep unlimited time while waiting for actual LIBERO success. Do not add a timeout, stagnation stop, or model-completion stop to the fair pair. Subscription/infrastructure errors still end an affected run and remain incomplete; preserve those attempts separately from fresh retries.
- The latest user request is a fair Hybrid-versus-Direct comparison. Run both arms fresh from the same initial state, with identical centered-placement and grasp-settling guidance, Astra medium, subscription OAuth only, feedback summaries and actual LIBERO-success stopping. Preserve the method differences (Hybrid field tracking/routing and Direct five-action control). Verify the shared guidance in actual requests and compare complete control-loop wall time. Do not reuse a selected historical successful Direct episode as the new paired baseline or report incomplete arms as completed timing results.
- Use only the user's ChatGPT subscription plan through its OAuth login for hybrid tests. Do not use `OPENAI_API_KEY` or any API-key billing path, including fallback after subscription authentication, quota, or model-access failures.
- Use `scripts/evaluate_bowl_hybrid_subscription.py`, whose configured model is `gpt-6-astra`. Subscription failures must stop the run rather than switch credentials or models.
- Hybrid defaults to `reasoning.effort="high"`. The user subsequently authorized a fresh Compact Hybrid test with explicit `reasoning.effort="medium"`, and clarified that either LIBERO success or Astra's task-completion signal is an acceptable stopping condition: stop on whichever occurs first. Record the requested setting in the manifest and request logs; do not silently change it if rejected or if the response reports a different setting. For this opt-in completion test, use a versioned explicit model completion flag, stop before executing any action from that terminal response, and distinguish model-declared completion from LIBERO success.
- Keep Direct mode at five generated and executed actions per model request. Field mode retains a 30-action model refresh interval and a 33×33×33 grid at 0.005 m spacing over 16 cm. Its local planner looks ahead ten 1 cm steps (10 cm total commanded path), using exact lattice-state merging, and executes all ten selected translation primitives before replanning. Each 30-action field block has three local planning/execution cycles, not 30 cycles or 300 actions. Collect feedback and adapt orientation from the actual state after each action. Route again after the active controller's block (30 field actions or five direct actions).
- Historical API-key baseline runners and recorded results are separate experiments; they are not authorized substitutes for this hybrid test.
- The latest user request retries Direct and requires actual LIBERO success. For this new Direct test, disable model-completion stopping: only the evaluator's LIBERO success signal marks task completion. Keep `gpt-6-astra`, medium effort, subscription OAuth only, five-action Direct chunks and the current feedback-summary mode. Preserve previous OR-stop runs as historical results; do not relabel a model declaration as LIBERO success or inject hidden object states into the policy.
- The user subsequently authorized implementing and testing an optimized Compact Hybrid: track the selected ten-segment geometric path from the measured pose with bounded look-ahead commands (instead of fixed primitive replay), deterministically summarize feedback blocks for model input while preserving full local logs, and skip redundant segment traversal only on fully valid grids. Keep the legacy execution path available. The matched full-task timing comparison uses `gpt-6-astra`, medium effort, subscription OAuth only, identical task/initial state/seed, and the first of LIBERO success or explicit Astra completion. Both arms use the same feedback summarization; Direct still outputs five actions. This authorization supersedes fixed primitive execution for the opt-in tracking experiment.
- Current live hybrid field output uses the compact `analytic-goal-rbf-v1` representation: GPT predicts a relative goal, attraction weights, at most four Gaussian repulsors, and blocked/unknown boxes; local code expands these into the same 33³ grid. Retain the raw compact output in model history, never the expanded dense values. The legacy dense subscription client remains available for reproducing older protocols.
- The user explicitly authorized a timing-only comparison through the hybrid runner's `--timing-comparison` option. Compare forced compact-field control against forced Direct control on equal executed-action counts, using the same model, high effort, task, initial state, and subscription credentials. This finite timing benchmark is separate from full-task success evaluation. Record model transport time, local expansion/planning time, control-loop time, and startup/reset overhead separately; do not present mismatched or incomplete arms as completed comparisons.
- The `api_calls` log field counts inference requests, including subscription requests. Use `billing`, `subscription_inference_calls`, and `api_key_calls` to distinguish the route; this test requires `billing="chatgpt_subscription"` and `api_key_calls=0` for live execution.
