# agent SpeedUp v3

Version: `agent-speedup-v3-11`. This is a separate entry point extending the current v2 user-check agent. Existing v2 code and historical run content remain unchanged by this implementation.

## Lean execution evaluation outcome (revisions 10–11)

This implementation round is complete, including cleanup, focused regression checks, explicit delegated plan review, and two frozen real t1 evaluations. **The five-minute successful-completion goal is not achieved.** Both durations below include preparation and full execution/finalization, excluding review wait; they are failed-attempt durations.

| Run | Revision | Steps | Execution model calls | Total seconds | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| t1_011 | 10 | 194 | 3 | 214.799 | Grasp descent stalled; tracking recovery could not supply a legacy coverage witness |
| t1_012 | 11 | 431 | 9 | 469.394 | Closed on the first object, but no accepted hold; stale neighboring milk geometry prevented lift-goal construction and targeted viewing |

Revision 11 carries actual tracking failure into the view request even when nominal route geometry admits motion, and allows a target/observed robot endpoint to supply an observation focus without a coverage witness. The 10 observation regressions and 2 policy/approval regressions passed. Its real test exposed remaining limitations: a support-volume center was chosen as a viewing focus and all candidates failed predicted visibility; a later missing semantic goal stopped view planning before candidate generation. A useful executed automatic observation recovery is therefore still not demonstrated.

For t1_012, execution model calls consumed 354.685 s, transition feedback 17.764 s, simulator steps 50.070 s and video 11.188 s. Its 419 open and 12 close actions reached first-object closure/contact, not verified transport grasp or release. The independent contact audit recorded table contact and later palm contact with the milk carton; this post-run evidence never entered policy input. Do not weaken neighbor invalidation merely to hide an actual possible disturbance. Both runs had zero transient/overload retries and unchanged runtime hashes, and both private supervisors were shut down.

The archived request used for offline context comparison shrank from 213,934 to 91,777 text characters including instructions, accepted output and schema (57.1%); all 12 images and 5 opaque reasoning entries remained. This is a payload-size measurement, not a measured model-latency speedup. Full outcomes and causal analyses are in `runs/agent_speedup_v3_local_execution_001/final_summary.json`, `t1_011_analysis.json` and `t1_012_analysis.json`.

## Revision 10: goal-local execution

The current policy uses `LocalGeometryMemory`; the older strict-coverage `GeometryMemory` remains available for historical reproduction. A missing free-space box is advisory, not an automatic execution veto. The evidence status remains `unknown`; separate `execution_allowed`, `blocking_reasons` and `advisories` describe the bounded execution decision. Neither permission nor successful robot tracking certifies physical clearance.

The hand and possible payload are checked against nominal obstacle hypotheses with a short continuous rigid sweep. Conflicts extending into the obstacle core after subtracting its declared uncertainty require correction; the uncertainty shell influences approach speed. These are uncalibrated model-consistency checks, not measured collision truth. Whole-route advice is cached between local checkpoints, while every emitted coupled command is independently checked. Current visual changes, controller bounds, missing carried shapes, finite attachment-probe limits, operation identity, and the progress watchdog remain enforced. Newly relevant objects consume their current RGB watch before the first action. Explicit moving gripper commands cannot bypass attachment verification.

Semantic grasp candidates now intersect nominal whole-hand/support clearance with the real public finger-pad closing range. The correction is derived from current geometry and controller tolerance. If the intersection is empty, the operation reports inconsistent geometry; no task-specific offset is substituted. Supported local grasp/release evidence and cumulative recovery budgets remain separate from geometry estimates.

The v10 executive prompt and request projection remove repeated capabilities, scene diagnostics and historical per-step detail from model input. Original accepted responses and complete local audit records are retained. Approval is still explicit and single-use; delegated review is labelled as Codex review. Model and reasoning remain `gpt-6-astra`, high, subscription only.

The 300-second objective includes initial preparation, model requests, execution, perception, logging, recording and finalization; human/agent review waiting is excluded. It is not an episode timeout. Only independent raw LIBERO success counts as task completion. Revisions 10–11 are recorded separately from earlier unsuccessful revisions.

Before implementation, `runs/cleanup_20261025_001/manifest.json` recorded removal of 75 inactive Python caches and lossless hardlink deduplication of 21,738 frozen copies, reclaiming 395,235,328 bytes. All non-cache paths and contents were retained. Frozen hardlinked copies must be copied to a new inode before editing; active runs, unique logs, images and videos were excluded.

## Staged evaluation outcome (revisions 7–9)

Implementation and the authorized automatic evaluations are finished. **None of the new runs succeeded; the five-minute successful-execution objective and a full real locally recovered object operation remain unproven.** The following durations include preparation and execution/finalization, exclude approval waiting, and are failed-attempt durations rather than completion latency. Execution-call counts exclude the one draft-plan request per run.

| Gate / run | Revision | Steps | Execution calls | Total seconds | Result |
| --- | --- | ---: | ---: | ---: | --- |
| A / t1_009 | 7 | 743 | 25 | 935.953 | Failed: release provenance erased by geometry replacement; old possible payload blocked next object |
| B / t1_010 | 8 | 190 | 7 | 535.930 | Failed before close: grasp descent coverage/clearance unresolved |
| C1 / t7_001 | 8 | 176 | 8 | 463.192 | Failed before close: can grasp palm/finger path unsupported |
| C2 / g5_001 | 8 | 21 | 5 | 201.183 | Failed: empty-hand pushing posture incorrectly created a plate payload |
| C2 fix / g5_002 | 9 | 71 | 6 | 242.435 | False payload removed; task still failed on unresolved palm coverage |

Revision9's causal retest recorded zero attachment probes, zero interaction references and zero active probe actions while maintaining closed empty-hand posture. Its 50 affected regressions passed in 1.26 seconds. This verifies removal of the false-payload behavior in that run, not pushing completion or a complete grasp/place loop. The earlier phase passes were 238 and 234 cases, with only failed cases rerun after corrections; these overlapping passes must not be summed as distinct tests.

All model requests returned valid results without HTTP503, transport retries or schema failures. Every runtime remained frozen during its episode. Gate B/C1/C2 used identical inventories of 319 runtime source files. The independent record-only audit counted 280 forbidden contact points in gate A and zero in the other new runs; these are physics-substep contact points, not independent collisions. Most runs stopped before task contact, so zero recorded contacts does not demonstrate successful contact manipulation.

The remaining evidence bottleneck is concrete: no accepted local metric pose updates were produced; stereo references were often unavailable/ambiguous and the explicit B anchor in t1_010 was rejected. Repeated missing whole-hand/payload coverage or predicted overlap led to finite recovery attempts and escalation. In t1_010, model calls consumed 358.349 seconds and transition feedback 103.772 seconds; dense boundary sweep refinement and overlapping-region checks were associated with the elevated feedback cost. Feedback includes more than pure geometry computation. Smaller call counts from early-failing runs cannot be presented as successful speedups over the 1,626.176-second successful t1_008 baseline.

Detailed components, source checks, causal analyses and videos are indexed by `runs/agent_speedup_v3_staged_001/final_summary.json` and `gate_results.json`. No additional simulator smoke tests were run. Optional asynchronous planning remains deferred because the prerequisite reliable local closed loop has not been demonstrated.

## Revisions 7–8: persistent operations and compact composition

The operation runtime now survives program, ordinary fragment, short-motion and local-recovery route changes. Actual acknowledged close/open commands establish observation epochs. The bound manipulated object, two-view references, accepted attachment/release evidence, finite probe window and cumulative recovery allowance persist across these routes. Model geometry updates cannot erase an actual pending release or certify release by declaring `attachment=none`; old/new carried and stationary extents remain conservative alternatives until fresh two-view evidence is accepted. A conflicting next-object close is stopped before its decision reaches the simulator.

Ordinary execution and semantic programs share the same finite grasp-probe authority after an actually acknowledged, settled close. Both stationary and hand-following payload hypotheses remain checked, including the final emitted command. Unresolved coverage is still unresolved. Accepted release removes the carried interpretation while retaining uncertain world occupancy. No task ID or task-specific movement distance selects these behaviors.

Local RGB geometry can establish calibrated feature references and consume credible two-view material displacement. An explicitly corresponding feature with a defensible shape-local anchor may translate an initial shape estimate after calibrated cross-view validation. These observations do not certify complete object shape, empty space, contact or task success. Unavailable/occluded references remain unknown. Accepted local pose evidence may revise a semantic goal while retaining consumed control budget. Local recovery has finite per-object attempts and must use supported candidates; it does not repeatedly renew an unknown path.

The executive can submit `program={operations:[...]}` with one to three generic `pick_place` operations and explicit dependencies, source/destination inventories, route obstacles and phase-specific contact permissions. Each compiles into the existing eight-stage controller. The next operation requires accepted release of its predecessor. Explicit semantic stages and existing native/ordinary interfaces remain available for other operations. This response-batch bound is not an episode cap. The compact contract contains no XYZ waypoints or distance offsets. Current decision context avoids repeating expanded stages, while complete runtime state, actions and accepted model history remain archived.

The prompt now describes the current design once, removing the old mandatory-model-call-at-phase-change and unconditional contact-speed statements. `gpt-6-astra`, high reasoning effort and the ChatGPT subscription transport remain unchanged. The target remains 300 seconds including preparation and finalization, excluding approval wait; it is not a stop condition.

Staged evaluation work is recorded under `runs/agent_speedup_v3_staged_001/implementation_plan.json`. Stage 1–3 regressions passed 238 cases, including the one corrected pre-recompute expectation. Recorded public RGB replay at steps 0/29/58 of t1_008 produced no triangulated material reference, so it does not validate metric local recovery on real observations. Revision7 gate A (`t1_009`) failed after 743 steps and 25 execution requests, taking 935.953 seconds including preparation/finalization. Shared tracking accepted one hold operation; 14 raw release-support samples were rejected after a model entity replacement erased release provenance. The stale possible payload then blocked the next object. No HTTP503, retry, or runtime source mutation occurred. This failed attempt is not a successful speedup comparison.

Revision8 fixes the observed provenance/probe interfaces and adds compact composition. Its targeted integration pass has 234 passing cases after correcting an obsolete release expectation and fixing the legacy monitor's second-operation release identity. Gate B (`t1_010`) failed before closing at 190 steps and seven requests, taking 535.930 seconds including preparation/finalization. Two compact operations were submitted and resumed, but unresolved fingertip/table and palm coverage blocked the saved grasp descent. It did not exercise the grasp/release fixes; no successful speedup is established. The same frozen revision8 source was then evaluated on t7 (176 steps, eight requests, 463.192 seconds, failure before closing) and g5 pushing (21 steps, five requests, 201.183 seconds, failure). All three revision8 source inventories matched and remained unchanged. g5 exposed a distinct bug: closing the empty hand as a pusher incorrectly created a possible-plate payload. Revision9 fixes the command-intent distinction without changing geometry tolerances or prompt content; only g5 is retested for this causal fix. Revision9 has not thereby established a complete pick/place closed loop. No simulator smoke tests were added.

## Revision 6: continuous geometric programs

The executive can authorize up to 24 semantic stages in one `program`: approach, grasp, lift clear, transfer, place, hold, and retreat. Each stage retains its own inventory and contact permissions. Stage targets are computed from current estimated object/support/destination geometry and calibrated robot geometry; the program has no waypoint or distance-offset fields. Local stage transitions do not require another executive request. Geometry pauses retain the remaining program, the resolved target when available, and its remaining watchdog. Missing geometry before target derivation remains a semantic requirement rather than a fabricated pose.

Gripper settling uses the controller's pose tolerance while checking the actual corrective sweep. A closed gripper alone never proves a grasp. The local interaction monitor tracks real RGB features in both calibrated camera views and compares stationary-world and hand-attached motion hypotheses. Supported co-motion permits conditional attachment; supported stationary release while the hand retreats permits the next stage. Missing views, occlusion or ambiguous motion remain unknown. These hypotheses do not update object poses from simulator truth or certify task success.

Geometry checks reuse unchanged projected masks and geometry within their validity conditions, while preserving the final command check and all margins. Local framework snapshots avoid repeatedly copying the complete executive history; full state is rebuilt for executive decisions. State files are sampled at most once per second and video reuses existing observation frames. All executed actions, model requests/responses, measurements and independent contact records remain retained.

The 300-second performance target includes preparation and reviewed execution through video draining and artifact finalization, excluding approval waiting. It is neither an episode cutoff nor a success predicate. `performance.json` records this metric separately from inherited control and model timings.

The frozen revision was evaluated as `runs/agent_speedup_v3_t1_008`, official LIBERO-10 t1 episode 0, seed 7, with the same initial-observation SHA as t1_007. Codex inspected all four fresh review images and both plan steps, explicitly approved under the user's delegation, and monitored to terminal success. The combined offline regression pass was **207 passed in 4.79 s**; no simulator smoke tests were run. Raw LIBERO and task success were true at **873 steps**, with **40 valid execution model requests**, no HTTP503, retry or runtime error, and unchanged runtime hashes. The private supervisor was shut down after both preparation and execution processes exited.

The five-minute target **was not achieved**: preparation plus complete reviewed execution was **1,626.176 s (27 min 06 s)**, comprising 20.302 s preparation and 1,605.875 s reviewed execution including finalization. Comparable evaluation wall time fell from 1,877.140 s to 1,601.897 s (14.66%); this is one paired episode, not a general benchmark result.

| Control component | t1_007, seconds | t1_008, seconds |
| --- | ---: | ---: |
| Model pipeline | 1398.966 | 1336.477 |
| Transition feedback, including geometry | 170.560 | 97.794 |
| Simulator steps | 104.708 | 110.091 |
| Video path | 110.080 | 37.492 |
| Live state files | 72.536 | 0.222 |
| Other policy decision computation | 9.187 | 8.795 |

Model calls fell from 52 to 40, but average model latency increased from 26.903 s to 33.412 s; the first program/geometry response took 119.56 s. The program advanced from approach to grasp locally and resumed through two geometry patches. At step148, driven translation genuinely stalled; the executive then replaced the program with ordinary `execute` fragments. The new local two-view interaction monitor recorded **zero observations**, because the program was abandoned before its close/probe stage. This run therefore validates program dispatch, local phase transition and continuation, but does not validate its complete grasp/transfer/release chain or establish that this chain accelerated the task.

The independent record-only contact audit counted **251 forbidden contact points** across physics substeps: 250 robot–scene and one manipulated-object–scene point. These are not independent collision events. Raw success does not establish collision-free execution.

Remaining limitations identified during the frozen run include repeated geometry handoffs after fallback to ordinary fragments and a contact profile cap applied throughout a long descent. Static review also found two unexercised program issues: release verification currently chooses an object from the current stage context, which can refer to the destination after the held-object field is cleared; and the program does not inspect the acceptance return from geometry's attachment-evidence confirmation. These were not reached and are not explanations for this run's observed stall. They remain explicit follow-up defects; no code was changed during or after this evaluation. See `performance.json`, `summary.json`, the detailed comparison report, and `v3_runtime_integrity.json` in the run directory.

The final ordinary-fragment release handoffs have a different observed cause: action868 was blocked at gripper-segment entry by estimated obstacle overlap, and action869 by `nearby_support_surface_for:A`. Both had `gripper_settling_reference=null` and zero executed steps in that rejected segment. Thus these attempts never received settling authorization; they do not demonstrate recurrence of the old one-micrometre correction gate. After the unchanged-blocker rejection at870, observe/wait responses retained the existing open command, and raw success arrived at873. This exposes a release-entry geometry/intent transition limitation.

## Implemented scope

Revision 5 repairs the t1_005 failure mechanisms with task-independent changes. Tracking evaluates driven translation and rotation separately, requires sufficient samples and persistent mismatch, and excludes in-tolerance holding corrections from response learning. Controller/risk limits are retained separately from residual-sized commands; the final coupled command is checked before emission. Whole-goal coverage, a continuously supported execution prefix, and spatial missing-evidence witnesses remain distinct. Bounded sweep refinement reduces interpolation padding only, preserving model uncertainty.

Geometry patches are previewed before repeat rejection, including new regions intersecting the pending sweep; accepted patches update the live visual monitor. Unrelated edits and duplicate renamed regions do not reset the guard. `acquire_evidence` proposes bounded camera-calibrated viewing moves supported by the current hand/payload geometry, preserves the pending goal/contact rules/watchdog and gripper state, then requests interpretation of fresh images before explicit continuation. It does not infer free space from predicted visibility.

The v3-only subscription transport enforces a total per-request deadline (180 s by default) including active streams, and rejects excessive JSON syntax whitespace or output growth without accepting partial actions. It requires the Linux main thread and an unused real-time signal timer. Terminal policy exceptions retain measured call and computation timings; normalized evaluator artifacts include those timings while preserving the original raw records.

Revision 5.1 of the prompt explicitly requests compact JSON without changing the schema. In preparation for t1_006, the new stream guard rejected a 2,049-character syntax-whitespace run after 24.85 s, before any policy action. Compact formatting resolved that observed preparation failure. During t1_006, a null uncertainty in a `no_free_region` witness exposed a view-proposal bug at action 90; it was fixed by retaining unavailable uncertainty and excluding it only from view-baseline sizing, without reducing geometry margins. All failed preparation/run artifacts remain available.

The final frozen build was evaluated on official LIBERO-10 t1, episode 0, seed 7, original reset, as `runs/agent_speedup_v3_t1_007`. Codex reviewed and approved the fresh plan under explicit user delegation and monitored the run to completion. Independent raw LIBERO and task success were both true at 880 actions. All 52 execution requests completed, with no HTTP 503, retries, output-guard failure or runtime exception; runtime hashes were unchanged. Evaluation wall time was 1,877.140 s; control excluding video was 1,758.153 s, of which model pipeline time was 1,398.966 s (79.57%). Preparation was separately 24.749 s (model 17.832 s). There were 177 distinct passing focused regression cases across the combined pass and targeted fixes; no simulation smoke runs were used.

Success does not establish a speed improvement over v2. Long supported paths continued without fixed 4/6-step handoffs, but contact/appearance invalidations still triggered many model calls. A remaining settling limitation was observed: the exact-command check treats arm translation above 1 micrometre as nonstationary, while the gripper controller makes small corrections to hold the same target. That mismatch interrupted some closure/release sequences. No behavioral changes were made during the final run or after it; this remains documented follow-up work. The single successful episode does not establish cross-task success rate. See the run's `summary.json`, `retest_analysis.json`, `v3_runtime_integrity.json` and `episode.mp4`.

The successful task was **not collision-free**. The independent record-only contact audit counted 433 forbidden contact points across physics substeps: 302 robot–scene points and 131 manipulated-object–scene points. These are accumulated contact points, not 433 independent collision events; they do not change the benchmark's raw success predicate. The first robot–scene contact was action 151, consistent in time with the later measured downward stall, while the first manipulated-object–scene contact was action 395 during basket placement. This contact evidence is evaluator-only and was not supplied to the policy.

The model keeps task ownership and proposes the next action, a bounded `scene_update`, and a `motion_context` in the same executive request. No separate reconstruction call is added. Inference remains `gpt-6-astra`, high reasoning effort, through the ChatGPT subscription; there is no API-key or model fallback.

`GeometryMemory` retains task-relevant objects, parts, supports, destination/free-space regions, multiple oriented boxes, identity/dependency links, source, age, validity, and uncertainty. Geometry from model vision is explicitly an uncalibrated hypothesis. Public robot measurements update the tool and conditionally attached payload; suspected contact, release, or slip invalidates affected geometry and dependencies. An available native visual held offset can supplement an existing entity; it does not create a shape or certify attachment.

For `execute` fragments and short direct `motion`, the local planner checks the proposed palm, separate fingers, and estimated payload sweep against this memory. It chooses a prefix of the model's requested path, per-action limits, and an observation horizon. A committed prefix may shrink during execution; it cannot continuously extend itself using the same old visual evidence. The original requested goal remains visible in feedback. Revision 4 distinguishes local progress checkpoints, geometry-only repair, and a new operation intent.

Revision 2 removed the fixed 4/6/24-control-step handoff for paths fully supported by still-valid geometry. Revision 3 also removes the mechanical 80-step expiry: age remains diagnostic, and relevant geometry is invalidated by observed appearance change, suspected disturbance, release/slip. A tracking mismatch revokes the current motion commitment without declaring all supporting scene entities stale. A lightweight fixed external-camera RGB watcher checks projected regions against their explicit model-update references, masking both reference and current hand/payload silhouettes. Consistent appearance does not refresh geometry timestamps or certify empty volume. Expected self-occlusion can retain a conditional static prediction; unexpected monitoring loss after a usable reference makes the relevant evidence unknown. Featureless or unavailable images remain explicitly unverified. Held objects use the attachment, slip and tracking checks rather than a fixed image region.

Every acknowledged step still checks the remaining sweep. Unknown or invalidated relevant geometry imposes a bounded remaining continuation, and explicit observation checkpoints, arrival, stalls and predicted blockage still return control. Unknown paths never renew their own distance/step allowance. Validity uses evidence supporting the current sweep, including relevant obstacles and payload, rather than every stored region. Newly relevant entities are checked before a new segment can use them. Measured robot progress does not refresh visual observation age.

Revision 4 checks a finite set of local translation/rotation commands against the current hand/payload sweep and its uncertainty margins, choosing supported predicted progress under the controller limits. A transit-profile motion can recover toward 20 mm per command; the contact profile stays within 5 mm. Normal command-response samples estimate progress; low response during contact or stalls is never inverted to amplify commands. Safety padding is not applied a second time as a global speed multiplier. Recovery cannot extend an unknown prefix or its deadline.

Each model response also includes `motion_context.scene_inventory`: `reviewed`, `support_ids`, `nearby_obstacle_ids`, and `unresolved`. Before the final approach/grasp/place interaction or gripper closure, the runtime requires a current target, a nearby modeled solid support and a resolved inventory with valid referenced entities. Missing evidence blocks that final interaction and returns specific `missing_evidence_requests`; it cannot be bypassed by crawling through successive unknown prefixes. The prompt requires the table/support and visible neighboring objects to be modeled along the whole hand/payload route. A free-region box is not a support surface. This is a model audit backed by consistency checks, not independent proof that every real obstacle was detected.

The first version checks model-selected routes and adaptively divides execution. It does not search for obstacle detours. Optional native macros retain their existing local planners; their measured transitions feed the shared memory. They are not all rerouted through the new sweep checker.

The existing hard outer target limits remain: fragment `transit` is at most 0.30 m / 1.2 rad and `contact` at most 0.08 m / 0.5 rad, with the original actuator limits. In revision 4, supported moving segments can continue past their local chunk budget under a measured-progress watchdog; gripper settling and explicit observation boundaries retain their own limits. Geometry may shorten these. Outside the final interaction guard, unknown geometry can permit a short explicitly uncertain model-directed move; it is never reported as known free space. Predicted overlap can return control to the model.

This checks the hand and estimated payload, not the full robot arm. Clearance estimates and model-specified uncertainty do not certify physical collision freedom. Cached geometry is not a new observation. Manipulation intent/contact permission does not prove that contact occurred or that grasping succeeded.

## Manual initial-plan approval

The existing selection-based review interface is retained: initial images, object/part selections, task steps, constraints, per-step checks, and a final approval. Every episode requires its own reviewed approval. Browser review by the user remains the default. If the user explicitly delegates review to Codex for a run, `prepare --delegated-review-note "<authorization and scope>"` records that delegation in the immutable review context. Codex must inspect the prepared images and plan before approving its steps; the flag itself does not approve or execute anything. Delegated runs are labelled separately and never reported as human-reviewed plans. The model may subsequently revise execution while retaining the reviewed task contract, as in user-check v7.

The v3 review has a separate version identity. A v2 approval cannot run v3. Approval binds the exact initial reset, task/episode, draft and selected plan, execution configuration, every package Python source, both relevant runners, helper chain, and review UI assets. Claiming approval is single-use; changed code/configuration or a different reset requires a fresh review. There is no `--auto-approve` option.

## Running the requested tasks

The requested official LIBERO-10 tasks, episode 0 / seed 7, are:

| Key | Official task |
| --- | --- |
| `libero_10_t1` | put both the cream cheese box and the butter in the basket |
| `libero_10_t2` | turn on the stove and put the moka pot on it |
| `libero_10_t3` | put the black bowl in the bottom drawer of the cabinet and close it |

Prepare each review sequentially. Preparation resets the simulator and requests the draft, then closes the simulator with zero policy actions:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_agent_speedup_v3.py prepare \
  --task libero_10_t1 --episode-index 0 \
  --review-dir runs/agent_speedup_v3_t1_review_001
```

Then serve the prepared review and run only after the user approves it:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_agent_speedup_v3.py run \
  --review-dir runs/agent_speedup_v3_t1_review_001 \
  --output runs/agent_speedup_v3_t1_001 \
  --wait-for-approval --port 8771 \
  --execution-lock /workspace/gpt_field/runs/agent_speedup_v3_execution.lock
```

Use separate review/output paths and ports for t2/t3, with the same execution-lock path. The URL is printed and flushed immediately after the review server starts. The shared lock is acquired only after manual approval and before claiming the review or starting execution; approved tasks wait without consuming their approval. Run managed processes under supervisor, as required by this instance. The review server binds loopback and retains its private token and origin checks; use SSH forwarding for remote access.

`start` combines preparation, review and execution. `review` serves an existing review without executing. Plain `run` requires an already approved review. `preflight` describes the active schema/configuration without inference, credentials or simulator work. The CLI exposes no smoke-test, automatic retry, or full-episode cap option.

## Evidence and evaluation

The unchanged evaluator determines completion from raw LIBERO success; the model, local arrival, geometry checks, and native holding labels cannot declare benchmark success. Contacts remain independently recorded, not episode failure gates. The diagnostic `v2_success` field remains in the inherited records; v3's reported `v3_success` uses raw LIBERO task completion.

The runner retains full model requests/responses, action/measurement traces, review/plan history, video, and independent evaluation. `active_agent.json` identifies v3 during execution. Original evaluator summaries are preserved in `base_v2_artifacts`; the exact pre-execution source tree is retained in `frozen_runtime`, with `v3_runtime_integrity.json` reporting changes. The prepared review also keeps its complete bound source tree.

Report control time with measured video overhead removed. An interrupted/incomplete attempt's duration is not completion latency. Implementation and offline contract checks do not establish task success or speedup; those require the requested real-task results.

## Revision 3 implementation (2026-10-24)

This revision implements event-driven geometry validity, local RGB change monitoring, clearance/distance/tracking-based speed, and the explicit support/neighbor inventory guard described above. Actual contact/release/attachment updates occur after action acknowledgement and before local reassessment or a following segment; a per-step idempotent update prevents the policy's fallback from applying the same transition twice. The manual/delegated review contract is retained, with a new code version requiring a freshly bound plan.

An already checked stationary gripper operation may finish its original bounded settling period through expected target contact/release or self-occlusion. This runtime-only permission is tied to its entry geometry, identities, hold pose and budget; the model cannot supply it and the next segment cannot inherit it. It does not clear stale evidence, extend a movement, excuse an unrelated scene change or waive palm/neighbor collision checks.

Validation: the five focused geometry/visual-watch/execution/client/review suites passed together (91 tests). After the transition-order and bounded-settling integration fixes, the two affected geometry/execution suites passed again (69 tests, including the added regressions). These are synthetic offline checks; no model or simulator calls were made by them.

The implementation checks launched no LIBERO episode or simulator smoke test. The user subsequently requested the separate revision 3 t1 retest recorded below. Visual change and tracking thresholds are heuristics; missed visual objects, depth/shape errors, unobserved motion and full-arm collisions remain outside any guarantee.

## Revision 2 t1 retry (2026-10-23)

The user explicitly delegated initial-plan review to Codex and authorized one t1 retry. `runs/agent_speedup_v3_t1_review_002/codex_plan_review.json` records the image/selector and two-step review; the bound approval records `human_reviewed_plan=false` and `approved_by=codex`. The original official t1/init0/seed7 reset hash was enforced. All 52 focused v3 geometry/execution/client/review tests passed in one invocation; no simulator smoke test was run.

| Recorded run | LIBERO success | Model calls | Control steps | Control minus measured video rendering | Total elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| v2 user check t1_001 | true | 13 | 897 | 325.391 s | 416.323 s |
| v3 t1_001 | true | 79 | 772 | 1651.311 s | 1739.202 s |
| v3 revision 2 t1_002 | true | 41 | 1568 | 1105.565 s | 1285.084 s |

The single retry reduced model calls by 48.1% and video-adjusted control time by 33.0% versus the preceding v3 attempt. It remains slower than the historical matched-reset v2 run; these individual trials do not establish an average success rate or speedup. Supported motion continued beyond the old fixed horizons, while coverage gaps and relevant evidence age still caused handoffs. Slow approach commands and recovery from unintended milk contact increased control steps. Fingers also contacted the table; collision-free execution was not achieved. Raw LIBERO and the post-release settling diagnostic both passed. Runtime integrity stayed unchanged throughout preparation and execution, and no model-request retry or additional episode was launched. Full evidence and video are in `runs/agent_speedup_v3_t1_002`.

## Revision 3 t1 retest (2026-10-24): subscription interruption

The user's request, "re test it on t1", launched one fresh episode in `runs/agent_speedup_v3_t1_003`, retaining their delegated t1 plan-review preference. Codex inspected the new draft and initial grid images before approving both steps. The reset matched t1/init0/seed7 and the earlier trials exactly; source integrity remained unchanged throughout the run.

The subscription endpoint returned HTTP 500 `unknown_error` on execution request 43 after 1,531 actions. There were 42 accepted execution responses and one separate initial-plan request. The runner terminated with `subscription_error`, with no automatic request retry, credential/model fallback or additional episode. Raw LIBERO success was false; evaluation was incomplete and completion timing was ineligible. Butter had been released into the basket, while cream-cheese transport was still pending. The private supervisor was shut down after the runner exited.

Consumed control time was 1,413.970 s, including 147.032 s of measured video rendering; the adjusted **attempt duration** was 1,266.937 s (21 min 6.937 s). Model pipeline time was 898.123 s, and transition observation time (including geometry work) was 115.439 s. These are interrupted-attempt timings, not successful completion latency or a speedup result. The contact diagnostic recorded 166 forbidden-class contact points on the table; these are physics-substep point samples, not separate collision counts, and this incomplete trial cannot establish a safety improvement.

Supported fragments continued beyond 80 steps, confirming removal of mechanical age expiry. However, the clearance/uncertainty speed rule often produced 2 mm commands and approximately 0.5 mm of measured progress. Model-created reciprocal B/table dependencies amplified invalidation across the scene until the model rebuilt them. Some compressed execute-fragment feedback omitted the specific missing-evidence lists, delaying those repairs. These observed limitations remain unresolved in revision 3; the runtime was not changed during this evaluation. See `retest_analysis.json`, the preserved requests, and `../agent_speedup_v3_t1_003_launch/live_geometry_diagnosis.json` for evidence.


## Revision 4 implementation and evaluation scope (2026-10-24)

The user authorized implementing the geometry-validity, adaptive-control and geometry-repair/intent-continuation changes together, delegated the new initial-plan review to Codex, and requested only one fresh t1 trial. The configuration remains task independent: object poses, movement distances and intermediate goals come from current model observations and robot feedback; no task/suite-name dispatch or task-specific waypoints are added. Existing camera calibration and robot/controller specifications remain allowed inputs.

Semantic `related_ids` are separated from directional parent-to-child pose dependencies. Target disturbance invalidates the target and affected local free space, not every semantically associated support or neighbor. Tracking anomalies revoke motion support. Visual references use nominal object shapes independently of collision-risk inflation, and expected hand occlusion remains a loss of observability rather than proof that the object moved.

Local motion uses relevant padded body–obstacle clearance and bounded candidates, informed by a measured normal command-response estimate. It does not divide commands by a small response ratio during contact or stalls, and it does not apply the largest unrelated scene uncertainty as a second global speed penalty. Supported progress checkpoints can continue the same absolute goal; retained watchdog accounting, explicit observations and gripper transitions still constrain continuation.

The current decision core preserves complete missing-evidence requests. A same-endpoint geometry patch can repair evidence and explicitly resume a stored intent without replacing its absolute targets or protected contact semantics. Repeating a blocked action without relevant new evidence routes to evidence acquisition or an explicit diagnostic stop. The complete accepted transcript and visual-history protocol remain unchanged in this revision.

The fresh review, run and supervisor directory are `runs/agent_speedup_v3_t1_review_004`, `runs/agent_speedup_v3_t1_004` and `runs/agent_speedup_v3_t1_004_launch`. Implementation checks and real-task results are recorded below only after they actually finish.

Revision 4 validation: 120 focused offline checks passed on the first invocation; one new side-wall fixture genuinely overlapped the uncertainty-padded palm. Correcting that synthetic fixture and rerunning only that case passed (121 total checks). No production change was needed after the initial invocation, and no simulator smoke test was run. Runtime sources are now frozen for the new review and one real t1 trial.


## Revision 4 t1 result (2026-10-24)

Codex inspected the prepared external/wrist grid images and approved both placement steps under the user's explicit delegation. The official init0/seed7 reset hash matched the historical v2/v3 comparisons. The one real `agent_speedup_v3_t1_004` attempt ended after **4 actions** because executive request 2 returned **HTTP 503 / subscription_sharing_user_unavailable** (1 accepted execution response). There was no automatic retry, API-key fallback or model substitution.

The attempt consumed 39.285 s of control time excluding the recording wrapper, 48.842 s total execution time. Task completion was not reached and completion-time comparisons are ineligible. Initial unknown geometry used a bounded 4-step continuation and normal measured tracking was established; the run ended before geometry-patch or supported local-continuation behavior could be assessed. All frozen runtime hashes remained unchanged. See `runs/agent_speedup_v3_t1_004/retest_analysis.json`, `summary.json` and `v3_runtime_integrity.json`.


## Revision 4.1 subscription availability recovery (2026-10-24)

Following the user's request to fix the t1_004 HTTP 503 and retest, v3 adds narrow recovery for `subscription_sharing_user_unavailable` and `subscription_sharing_usage_unavailable` at HTTP 503. Official recovery guidance identifies these as temporary subscriber/workspace or usage lookup failures: https://developers.openai.com/siwc/token-sharing-open-source/errors-and-recovery . Existing credentials are preserved. There is no account/model/API-key fallback or automatic episode restart.

The v3-only retry mixin covers both initial-plan preparation and execution. It reuses the existing same-observation retry loop: at most two additional transient attempts, 2/4-second backoff plus jitter, Retry-After respected, and the existing 60-second single-wait/budget bound. No complete response or committed observation may be replayed; robot actions remain paused while requesting a decision. Errors outside the exact status/code allowlist retain their previous handling. Planning, geometry, control, prompts and accepted-history semantics remain revision 4.

Every failed request and backoff is logged and included in wall/control timing. The normalized summary exposes transient retry counts/codes/waits and retains preparation history. The fresh review and run use suffix `005`; the previous interrupted attempt remains intact.

Revision 4.1 validation: all 48 affected client/review checks passed in one invocation. These include both preparation and execution recovery, same-payload/history-once semantics, Retry-After and retry limits, and terminal nonmatching errors. No simulator smoke or extra inference probe was used. Sources are frozen for the fresh t1_005 review and execution.

## Revision 4.1 t1 result (2026-10-24)

Codex inspected both grid images and the new two-step placement plan, approved it under the user's delegation, and ran one fresh official t1/init0/seed7 episode in `runs/agent_speedup_v3_t1_005`. The original observation hash matched; frozen runtime integrity remained unchanged. The private supervisor was shut down after both programs exited. No second episode or source change was introduced during this trial.

Preparation took **1682.111 s**, including **1675.190 s** in the plan client. The first request ended after about 900.5 s with `stream_missing_completion`; the existing transient recovery waited 2.877 s and the second request succeeded after about 771.7 s. The accepted plan text contained 152,567 characters, of which 151,950 were JSON whitespace outside strings; its compact JSON was only 617 characters. This observed output expansion explains the successful request's unusually large output and long wait. The failed first stream's contents were not retained, so its cause cannot be equated with the second response's whitespace. The specific new subscription-availability 503 branch did not trigger in this live trial.

Execution ended with **`evidence_acquisition_failed`**, **210 control steps**, and **25 accepted model responses**. No execution HTTP 503 or transport retry occurred. The final response explicitly escalated because the model could not establish the partly occluded basket-side and milk-side boundaries for the whole lower-hand sweep. There were zero gripper-close actions. Independent LIBERO success was false, evaluation remained incomplete, and successful-completion timing comparisons are ineligible. The runner records 24 normally returned policy decisions because response 25 raised the evidence-stop exception.

Execution elapsed time was **579.736 s**; control time was **571.942 s**, or **549.586 s** excluding 22.356 s measured in the recording wrapper. Recorded model-pipeline time was 484.762 s, policy compute 1.128 s, transition observation 13.385 s, and simulator stepping 25.483 s. The terminal policy call is excluded from the normal model-pipeline accumulator but remains in control time; transition observation includes geometry and feedback work, and run-state timing is not an exclusive component of the control window. Preparation time is separate from these execution figures.

Supported motion did continue for longer stretches (including 44-step alignment and 26-step descent). However, geometry-patch decisions at steps 110 and 114 still found approximately 21–23% of the original full sweep uncovered. Their locally supported assessment applied only to a shortened 15 mm prefix; the fixed unknown-geometry horizon therefore returned to the model after four steps, despite remaining watchdog budget. Tracking transients and repeated partial coverage repairs also consumed decisions. This run verifies that the revised interfaces execute, but does not establish faster task completion. See `retest_analysis.json`, `summary.json`, and `../agent_speedup_v3_t1_005_launch/transport_diagnosis.json` for the separate transport and task outcomes.
