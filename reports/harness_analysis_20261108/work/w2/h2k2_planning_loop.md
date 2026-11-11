# H2 keypose-2: planning loop, execution and attempt economy (W2, key `h2k2_planning_loop`)

Independent second opinion on revision `foundation-harness-keypose-2`. Scope: `motion.py`, `executor.py`, `client.py`, `feedback.py`, `scripts/evaluate_foundation_harness.py`, `tests/test_foundation_harness_integration.py`, plus the parts of `perception.py`, `surface.py`, `robot_geometry.py` and `foundation_vision/geometry.py` that the planning loop depends on. The W1 H2 reports were not read.

Abbreviations:
- **SNAP** = `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/h2_keypose2_snapshot`. File:line references such as `motion.py:416` are to `SNAP/src/gpt_field/foundation_harness/`.
- **RUN1** = `/workspace/gpt_field/runs/foundation_harness_t04_can_obstacle_001`.
- **WORK** = `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/h2k2_planning_loop_work`. It holds the scripts `aN_*.py` and their outputs `data/aN_*`.

All checks are offline and CPU-only on saved artifacts: no simulator, no model calls, no GPU.

## 0. Snapshot check

- **All 11 changed files match.** Every file listed in `SNAP/prep/implementation.json` `changed_source_sha256` has the recorded sha256 in SNAP. The live tree `/workspace/gpt_field` has the same hashes, re-checked at 2026-11-08 00:41 UTC. The unchanged `__init__.py` and `perception.py` are identical in SNAP and live. No live divergence was seen.
- **Last edits happened before the prep checks.** `motion.py` was last modified at 23:58:34. The final prep check (`saved_input_check.json`) ran at 23:58:53 and `final_changed_checks.xml` at 23:58:47, so both ran on the final code.
- **The `debug` JSONs are intermediate-revision diagnostics.** `initial_configuration_debug.json` (23:44) and `previous_astra_direct_debug*.json` (23:46 and 23:50) predate the final self-filter: they report method `current_robot_first_ray_surface_agreement`, while the final one is `current_robot_silhouette_ray_occlusion`.
- **Upstream vision code is unchanged.** `foundation_vision/*.py` is identical between `RUN1/source_at_launch` and live, and `perception.py` is identical between RUN1 and keypose-2.
- **My offline harness reproduces the prep results exactly.** WORK/overlay is a copy of `src/gpt_field` with the SNAP harness files. Both prep cases match: 5 and 11 path points, 0 and 220 uncertain samples, 264 and 583 unknown visits, 72,738 self points removed (`data/a14_reproduce_prep.json`).
- **The 5 integration tests pass** against SNAP in a scratch copy: `WORK/tests`, `5 passed in 2.83s`.

## 1. How keypose-2 works (planning loop)

**Astra calls.** Each call (`client.py:389-419`) returns targets plus operations. The JSON schema is strict (`client.py:155-177`) and validated semantically (`client.py:210-266`). A violation raises `SchemaError`, which ends the episode as `subscription_error` with no re-ask (`hybrid_subscription.py:471-486`, `scripts/evaluate_agent_speedup_v2.py:1637-1638`).

Every operation carries 0-12 keyposes. Each keypose is an absolute world grip-site position plus a wxyz quaternion, with `gripper` (open/close/keep), `profile` (transit/contact) and `observe_after` (`client.py:89-113`).

**Per-keypose planning.** The executor pops one operation (`executor.py:380-414`) and plans **only the keypose at `keypose_index`** (`motion.py:556-607`):

- **IK.** Damped least squares, seeded only with the measured joints, 80 iterations (`motion.py:468`, `robodojo_direct_sim.py:226-264`).
- **Endpoint veto.** If either the start or the goal configuration is unsafe, the plan fails at once (`motion.py:414-426`).
- **Direct edge.** A straight joint edge sampled every 0.055 rad (`motion.py:335-354`).
- **Transit fallback.** Transit keyposes only fall back to an RRT: 100 iterations, step 0.18 rad, uniform joint samples, deterministic `default_rng(0)`, 12 s compute budget (`motion.py:23-33, 434-460`). Contact keyposes get no fallback (`motion.py:432-433`).

**Configuration check** (`motion.py:224-333`), in order:

1. Joint limits.
2. GJK self-collision on convex mesh hulls.
3. Upright tilt (5°) for `keep_upright` payloads.
4. Payload vs non-finger robot parts.
5. Scene points within 6 mm of any link or payload hull. The fixed base, link group 0, is skipped (`motion.py:271-272`).

**Strong vs uncertain points** (`motion.py:110-130`).
- A point is **strong** if it lies on a target's coherent planning surface, or has cross-view support ≥2 with zero contradictions.
- Everything else is **uncertain**. Uncertain points, and unobserved "unknown" swept space, are counted and **allowed** (`allow_unknown_space=True`, `motion.py:31, 294-306, 330-333`).

**Exemptions by keypose name** (`motion.py:518-521`). They apply only to points *owned* by the named ids:
- grasp/close/lift: finger/pad parts may touch the object or grasp region.
- place/release: the payload may touch the target.
- Every other name: no exemption.

**Held object.** It is the convex hull of the object's close-time `planning_surface`, expressed in the gripper frame at Astra's close pose. If that surface has fewer than 4 points, the payload is **absent** and the held object is not checked at all (`motion.py:532-551, 252-256`).

**Execution.**
- `JointPathStage` (`executor.py:43-113`) tracks the forward-kinematics (FK) poses of the joint path, densified to 0.025 rad (`motion.py:407-412`), with Cartesian OSC delta actions.
- Steps are capped at 20 mm (transit) or 3 mm (contact). The tracker advances to the next waypoint only within 4 mm / 0.055 rad, and counts as arrived within 3 mm / 0.05 rad on 2 consecutive steps.
- The budget is 180 actions per stage. A stall is declared after 18 steps with less than 0.8 mm progress.

**Per-step re-checks.** Every control step re-checks the joint edge from the measured joints to the current waypoint against the newest scene (`executor.py:452-461`). At gripper keyposes it also checks the next 3 mm of finger travel (`executor.py:462-473`, `motion.py:372-405`).

**Perception refresh.** SAM and DA3 refresh whenever the frame is older than `max_age_s*0.5` = 2.5 s, or every 10 observations (`perception.py:181-183`). `act()` calls the update every step (`executor.py:439`).

**Failure routing.**
- `observe_after`, or the last keypose, ends the operation and triggers a new call (`executor.py:361-372`).
- Planning failures call Astra immediately, with structured evidence (`executor.py:283-289`).
- Motion and stage failures get one local replan per operation (`executor.py:298-311`).
- There is no call cap and no give-up. A `finish` answer keeps `need_model` true (`executor.py:268`).

## 2. Run-001 defects and their status in keypose-2

| # | Run-001 defect (evidence) | Status | Keypose-2 evidence |
|---|---|---|---|
| D1 | **Start configuration rejected.** The fixed-base `link0_collision` hit table/mount points with 5.0-5.7 mm residual, inside the 6 mm margin. Both moves failed before leaving q (`RUN1/analysis.json` finding 1; `RUN1/analysis/reconstructed_rejections.json` `initial_configuration`). | **replaced by new risk** | Group 0 is skipped (`motion.py:271-272`) and the current-robot silhouette self-mask removes 72,738 self points. The 5 cm move now plans in 1.6-2.0 s (`prep/saved_input_check.json`; `data/a14`). But the start-endpoint veto is now explicit (`motion.py:416-426`) and also hits contact states: H3's own contact-free white-mug grasp cannot be left by any next keypose (§4 K3). |
| D2 | **Five pick_place requests produced 0 placement candidates** (`no_supported_payload_footprint`). The local generator tested raw whole-object XY extrema, handle included, against the visible plate (`analysis.json` finding 2). | **replaced by new risk** | The generator is gone: Astra authors place keyposes (`motion.py:503-554`). The coherent patch shrinks the yellow mug's extent from 26.3×23.4 to 12.3×14.2 cm (`prep/saved_input_check.json` `surface`). But placement metric accuracy is now Astra's (H1 F3: 3.4-5.7 cm), with no local support or height check (K5). |
| D3 | **Lower-level reasons were lost.** `str(exc)` became `no_complete_operation_path` (`analysis.json` finding 3; RUN1 `source_at_launch/.../executor.py:282`). | **fixed** | `executor.py:283-289` and `motion.py:598-605` pass the reason, evidence (part, point, endpoint, IK errors) and keypose index. The test asserts it (`tests/...:213-220`). Caveat: blocking points carry no semantic label, and uncertain counts are cumulative (K9). |
| D4 | **Handle grasp rejected** with `no_observed_opposing_contact_pair` (1 rejection). | **replaced by new risk** | Local grasp synthesis was removed; Astra owns grasp poses. Handle and side grasps now collide with the convex-hull payload model: `payload_robot_collision` between `hand_collision` and the hull (`data/a9_*`, `data/a10`). |
| D5 | **Hard veto `payload_geometry_insufficient`** (old `motion.py:176-177`). | **replaced by new risk** | Payload absence is now only reported (`motion.py:256, 545-548`), so the held object is not collision-checked at all. The white-mug planning surface is empty in 9/34 saved revisions (`data/a8`). |
| D6 | **Unknown swept space was a blanket prohibition** (old `allow_unknown_space=False`). | **replaced by new risk** | Unknown space is allowed (`motion.py:31, 330-333`). The "bounded step with observation" uses the same 20 mm / 3 mm caps as everywhere else (`executor.py:31-33, 80-83`). There is no slow-down, no required view of the swept volume, and no automatic `observe_after` (K4). |
| D7 | **Robot self-surfaces and phantom depth acted as obstacles.** The old filter removed only points within 3 mm inside the envelope. Intermediate keypose-2 revisions: start unsafe at link6 z 0.839 (`prep/initial_configuration_debug.json`); goal and direct path hit link5 at z 0.81-0.82 (`prep/previous_astra_direct_debug_v2.json`). | **partially fixed** | The link6 point is a robot self-surface. It sits 5.9 mm outside the link6 hull and is gone after the final silhouette filter: the nearest retained point is 6.7 cm away (`data/a2`). The z 0.81-0.82 points are phantom background depth (§3). They remain in the cloud, only demoted to non-blocking "uncertain", together with 90% of the scene, including real obstacles (K4, K9). |
| D8 | **Attempt economy.** 12 calls, 0 operations. Every planning failure meant a new call, and Astra resent `pick_place` 6 times (`RUN1/trial/foundation_harness/events.jsonl`). | **partially fixed** | First motions now plan. Planning failures still go straight to Astra (`executor.py:288`). There is still no budget or give-up, and new failures Astra cannot fix appear (K3, K8). |
| D9 | **Slow loop.** A forced SAM/DA3 refresh on every step (frames 1-19 at steps 0-18). Non-model steps took 3.0-6.5 s (`RUN1/trial/trace.jsonl` `decision_latency_s`); 71 s of local compute (cross_eval §2). | **still present (worse when moving)** | Refresh fires whenever the frame is older than 2.5 s (`perception.py:181`). Each refresh now also re-runs the 1.07 s silhouette self-mask (`motion.py:96-208`; `data/a1`) and the per-step motion check costs 0.26 s. Net ≈3.5-3.7 s per control step (K1). |

## 3. What the "observed obstacles" at z ≈ 0.81-0.84 m are

T04 truth (layout manifest, scoring only):
- table top z = 0.437 m; can top 0.513 m; mug tops about 0.54 m;
- nothing but the robot exists above about 0.55 m.

Planner cloud at RUN1 frame 14 (`data/a1`, `data/a2`, `data/a3_overlay_external_step13.png`):
- **Phantom layer above z 0.60 m.** 39,738 points are retained there, all from the external view and all uncertain (z quantiles 0.62 / 0.72 / 0.80). 88% lie outside the current robot silhouette; they are the dark background at the top of the external image, which DA3 puts 0.6-0.8 m high around the arm. The 12% inside the silhouette lie 0.22-0.30 m in front of the robot surface. This is **phantom depth, not robot and not objects**.
- **`initial_configuration_debug` link6 point** (−0.113, −0.007, 0.839) is a **robot self-surface point**: hull residual +5.9 mm. The final filter removes it.
- **The `previous_astra_direct_debug_v2` link5 points and the final `saved_input_check` "uncertain" examples** (z 0.814-0.825) project to external image rows v ≈ 2-10, next to the robot silhouette. They are 7-10 cm from the link5/link6 hulls, and they are still in the final cloud. They are **phantom depth**.
- **The final revision makes them passable only by demoting all single-view geometry.** The table plane is 92.4% uncertain: 2,672 of 34,979 points within ±1 cm are strong. Wrist depth puts the table 6.5 cm low (median z 0.372; 82% of wrist points lie more than 2 cm below the table) (`data/a15`). This matches the H3 vision report: +18% wrist table depth error and unstable per-frame cross-view agreement (`w1/h3_vision_engine.md:154-164`).

## 4. Keypose-2 findings (ranked)

### K1 (critical, FAST). Every control step re-runs SAM/DA3 plus the self-mask: about 3.5 s per step, so T04 needs about an hour of control

**Mechanism.**
- `act()` refreshes perception when the frame is older than 2.5 s (`executor.py:439`, `perception.py:181-183`).
- A refreshing step costs:

  | Component | Time | Source |
  |---|---|---|
  | SAM + DA3 + geometry | 1.83 s median over 20 full refreshes (1.44-3.40) | RUN1 `snapshots.jsonl` `timings_s` |
  | npz save | 0.18 s | measured |
  | new `_prepare` silhouette self-mask | 1.07 s | `data/a1` |
  | per-step `check_motion` (2 configurations at about 110 ms) | 0.26 s | `data/a5`, `data/a6` |
  | runner/video | 0.28-0.40 s | H1 §1.2, same runner base |
  | **Total** | **≈3.5-3.7 s** | |

- 3.5-3.7 s is more than 2.5 s, so the next step refreshes again: the steady state is a refresh on every step. This is an inference from measured components; the robot never moved in RUN1.

**Step count.** A first-order OSC model (realized 22-30% of the command, H1 §3) runs over IK and densified paths for an H3-like T04 keypose plan. It needs **1,042-1,404 actions**, 493-672 for the first mug (`data/a7`). That is consistent with uc7.1-T20's 1,276 actions at the same 20 mm cap.

**Total.** About **60-85 min of control**, against a 3-minute target.

**Improvement.**
- Decouple perception from control: a background refresh at keypose boundaries plus a fixed cadence of 1 Hz or less.
- Cache the self-mask per frame and down-sample the planning cloud by voxel.
- Drop the per-configuration free-space query while unknown space is always allowed (`motion.py:313-333`).

**Expected effect.** About 0.4-0.6 s per step; about 10-14 min for the same action count (with K7, far less).

**Validation.** Offline per-step timing replay on saved frames; live per-step timing distribution; all-task sweep.

### K2 (critical, COLLISION + FEW-ATTEMPTS). The held payload is either invisible or blocked by its own surface

**Mechanism.** The payload is the convex hull of the close-time planning surface (`motion.py:532-551`). The held object's points are exempted only if they are **owned** *and* a payload exists (`motion.py:280-282`).

Ownership drops mask-boundary points that are cross-view conflicting (`perception.py:454-461`). Those points are exactly the strong-but-unowned ones.

**Evidence** (`data/a8`, `data/a9_*`):
- **Self-blocking is near-universal.** In every saved revision with a coherent surface (white mug 25/25, yellow-white mug 14/14), the hull plus 6 mm contains the object's **own** strong unowned points: 151-7,299 for the white mug, 197-205 for the yellow mug. The count excludes the table band and the can.
- **The payload is often missing.** The white-mug planning surface is **empty in 9/34** revisions (`ambiguous_components`). In those cases there is no held-object collision check at all, and the owned points of the mug block the hand or fingers instead.
- **Direct check at a white-mug rim grasp (frame 20).** `lift` and `transport` fail at the start with `held_visible_surface_envelope` against a strong point next to the can.
- **Direct check at a yellow-mug handle-side grasp.** `payload_robot_collision` with `hand_collision` (hull top 0.528 m vs hand bottom 0.520 m).

**Improvement.** Remove the mask-ownership dependence:
- define the payload as everything inside its close-time envelope (geometric membership that moves with the gripper);
- use a conservative fallback envelope (raw object points or mask frustum) when `planning_surface` is empty;
- exempt by membership, not by SAM ownership.

**Expected effect.** Lift and transport become plannable, and the held object is always checked.

**Validation.** Offline: a8/a9 re-run after the change, requiring zero self-hits and zero missing payloads across all 44 revisions. Live: a T04/T08 sweep counting `held_visible_surface_envelope` start failures and `manipulated_object_scene` contacts.

### K3 (critical, FEW-ATTEMPTS/ZERO-SHOT). Once in contact, the robot cannot leave: start-endpoint veto plus name-based exemptions

**Mechanism.**
- `_connect` fails immediately if the start is unsafe (`motion.py:416-426`), and every per-step check includes the current q (`motion.py:338`).
- `retreat`, `transport` and `preplace` have no contact exemption (`motion.py:518-521`).
- A start "collision" is neither fixable by the local replan nor by Astra, because the start is the measured state.

**Evidence** (`data/a10`). At H3's actual white-mug grasp pose — grip (−0.085, −0.10, 0.51), top-down, executed in H3 with **0 forbidden contacts** (cross_eval §1b row 2) — every next keypose is unsafe at the start in both saved frames:
- frame 14: `hand_collision` with a strong unowned rim point at z 0.558;
- frame 20: `hand_collision` with an owned white-mug point, or `payload_robot_collision`.

The same grasp would already be vetoed as a *goal*. A too-low grasp is likewise vetoed through `hand_collision` (`data/a5` S6/S7).

**Improvement.** Replace the endpoint veto with a "clearance must not decrease" rule: allow paths whose contact or penetration set shrinks monotonically from the start. This removes a veto instead of adding a threshold.

**Expected effect.** No dead ends after close or release; fewer calls that cannot help.

**Validation.** Offline a10 sweep over grasp poses and frames; live count of `endpoint: start` failures.

### K4 (high, COLLISION). Blocking evidence is transient: 90% of the scene is "uncertain" and there is no memory

**Mechanism.** Only strong points block (`motion.py:285-306`). Strength is recomputed from the newest two-view frame (`motion.py:96-130`), with no fusion across frames and no support-plane model. Unknown space is allowed.

**Evidence.**
- **The scene is mostly non-blocking.** At frame 14, 10% of 448,483 points are strong and the table is 92% uncertain (`data/a1`).
- **The table does not stop descents.** Contact descents driving the fingertips about 2 cm *into the table* plan at 5 of 6 free-table sites, with 1,447-3,004 "uncertain" finger-table intersections reported (`data/a12`).
- **The can's strength comes and goes.** From H3's saved geometry, same pipeline (`data/a4`):

  | Frame | Robot situation | Can points strong |
  |---|---|---|
  | 0 | home | 95% |
  | 17 | above white mug | **0%** (0/11) |
  | 86, 167 | white-mug grasp | no can points at all |
  | 201, 252 | after the transfer | 95% / 79% |

- **Without the wrist view, the can stops blocking.** With the wrist depth invalidated, transport keyposes over the can plan at both z 0.65 and z 0.59 (`data/a5`). The white-mug payload is empty in that frame, so this run also shows K2's invisible payload.
- **Model-written obstacle geometry is also unreliable.** H3 F2: model obstacle boxes covered 0-30% of the true obstacle on the first try (H1 runs).

**Improvement.**
- A persistent fused obstacle map: keep strong evidence until it is contradicted or carved by the robot's own measured swept body.
- A support-plane estimate as a strong obstacle for non-contact parts.
- A per-view depth affine self-calibration from robot-visible anchors (H3 vision report §depth).

**Expected effect.** The can and the table stay blocking during approach, grasp and transport. This targets the T04 mug-can contacts of uc7.1/astranomy (cross_eval §1b: 33-634 samples).

**Validation.** Offline re-run of a4/a5/a12; live forbidden samples by class on the T04/T08 obstacle variants plus all tasks.

### K5 (high, ZERO-SHOT/FEW-ATTEMPTS). Astra's metric keyposes are used verbatim

**Mechanism.**
- The pose comes straight from Astra's JSON (`motion.py:509-512`).
- There is no snapping to observed surfaces or support planes.
- `approach_direction` and `placement_relation` are validated but never read by the planner (`client.py:261-263`; not referenced in `motion.py`).
- There is no guarded or contact-detecting descent; only an 18-step stall rule (`executor.py:107-110`).
- Planning-surface statistics only reach Astra as text (`types.py:39-57`).
- Local protection is both leaky (K4) and over-strict: the 6 mm margin plus DA3 rim error vetoes H3's physically clean grasp (K3).

**Evidence.**
- H1 F3: first estimates 3.4-5.7 cm off along the external camera's depth axis; 28% of calls were stall-triggered, mostly height/depth misestimates.
- cross_eval §1b: palm-vs-target dominates T04 forbidden contacts (H1 3,372; uc7.1-T20 1,292).

**Improvement.**
- Refine Astra keyposes locally: project grasp and place points onto the observed target surface or support plane along the approach axis.
- Execute contact segments as guarded moves that stop on stall and record the contact height.
- Give Astra a `locate` triangulation tool (H1 F3).

**Expected effect.** Fewer stall/miss calls (H1 F3: 1-2 fewer per pick-and-place) and fewer palm contacts.

**Validation.** Offline: triangulation error against layout truth. Live A/B sweep.

### K6 (high, FAST/FEW-ATTEMPTS). The RRT fallback cannot work within its budget

**Mechanism.** One configuration check costs 107-136 ms; 82% of that is `CollisionShape.contains` on dense points, much of it the phantom layer near the upper links (`data/a6`). The 12 s budget therefore buys about 110 checks, which is about 12 RRT iterations. The seed is fixed, so a local replan repeats the same search.

**Evidence.** A transit keypose whose direct path crosses the strong can at z 0.50 fails with `planning_compute_budget` after 12.03 s, 111 checks and iteration 12 (`data/a11_rrt_budget_z0.50.json`).

**Improvement.**
- Vectorize checks with per-frame per-link point subsets.
- Replace the uniform joint-space RRT with short Cartesian detours (lift-over or sidestep).
- Drop the free-space query that never blocks.

**Expected effect.** Detours found in under 1 s; no 12 s dead waits.

**Validation.** Offline a11 plus a batch of blocked pairs; live count of `planning_compute_budget`.

### K7 (medium, FAST). Stop-and-go tracking at 20 mm / 3 mm caps

**Mechanism.** The tracker chases the next densified waypoint, about 1-2 cm ahead, and only advances within 4 mm (`executor.py:67-88`). The OSC realizes about 25% of each command (H1 §3).

**Evidence.**
- Steps per keypose from the step model (`data/a7`):

  | Keypose | Profile | Actions |
  |---|---|---|
  | Retreat, 7.5 cm | contact | 81-110 |
  | Pregrasp, 30 cm | transit | 104-140 |

- A 15 cm contact segment would exceed the 180-action stage budget (`executor.py:105-106`).
- H1 at a 100 mm transit cap needed 57% fewer T04 actions than at 20 mm.

**Improvement.**
- Track a lookahead target along the path instead of gating every waypoint.
- Use H1's 100 mm transit cap; run contact as guarded moves (K5).
- Tell Astra the profile costs.

**Expected effect.** About 2× fewer actions.

**Validation.** a7 model plus live actions per keypose.

### K8 (medium, FEW-ATTEMPTS). Failures route to Astra even when it cannot help, and nothing ends a futile episode

**Mechanism.**
- Planning failures go straight to Astra (`executor.py:288`), and each operation gets a single local replan (`executor.py:304-311`).
- There is no call cap, give-up or recovery-exhausted event (`scripts/evaluate_foundation_harness.py:100-117`; runner `STOPPING_RULE`).
- A `finish` answer keeps `need_model` true, so Astra is called every step until LIBERO succeeds (`executor.py:268`).
- Text history is never trimmed (`client.py:359`). Per-call context is about 9-27 KB of scene plus about 5.7 KB per failure record, an estimated ~103 KB after 12 failures, close to the uncaught 120,000-character limit (`client.py:410-411`; `data/a13`).
- `SchemaError` ends the episode with no re-ask.
- A region box beyond the raster raises an uncaught `ValueError` (`perception.py:394-395`; `client.py:221-224` checks only the lower bounds).
- A `displacement` move is re-based on the current pose at every local replan, so it overshoots (`motion.py:569-579`).

**Minimum calls for two mugs.** About 5-7 if nothing fails: one identity observe, one approach/grasp/lift and one transport/place/release per mug, and a finish.

**Improvement.**
- Handle start-collision, self-block and budget failures locally.
- Label blocking points semantically: held object, target, support plane, phantom.
- Add a call budget, a finish rule and history compaction.
- Re-ask once on a schema error and clamp region boxes.
- Store the operation-entry pose.

**Validation.** Replay a failure taxonomy on traces; live calls per task.

### K9 (medium, COLLISION/FAST). The phantom layer is kept and drives both cost and Astra's evidence

**Mechanism.** About 39.7k external-only background points above z 0.6 m (§3) are kept as "uncertain".
- They make up much of the ball-query load in each configuration check (K6).
- They inflate the `uncertain_obstacle_samples` sent to Astra: 220-3,004 per plan (`data/a5`, `data/a12`; cumulative semantics, `motion.py:498`).
- The INSTRUCTIONS tell Astra to choose clearance and observations from this evidence (`client.py:49-52`).

**Improvement.** Remove space the robot's measured body has swept through (robot-body carving). Report uncertain intersections per part once, with nearest-owner labels.

**Validation.** Cloud size, check cost and the count of observation-only calls.

### K10 (medium, test coverage). Tests and prep checks miss every live risk above

**Evidence.**
- 5 tests: client protocol, perception timestamps, the planner on an **empty** scene (`tests/...:131-158`), a policy with a **mocked** planner (`tests/...:161-220`), and video.
- 2 saved-input checks: first transit keyposes from home only (`prep/saved_input_check.json`). The implementation notes record that no broad matrix was run (`implementation.json` `checks_note`).

**Not covered.**
- Strong/uncertain classification with points.
- The self-mask.
- Payload, close, release and lift/retreat starts.
- RRT and IK failure paths.
- Stage completion and stall behavior.
- Refresh cadence and step time.
- `finish`/observe loops, schema-invalid answers and context limits.

**Improvement.** Promote a5/a8-a12 and a14 into saved-input regression tests with pass criteria.

### K11 (low, COLLISION). The joint path is checked but Cartesian OSC is executed

**Mechanism.** The planner checks joint-space paths, but `JointPathStage` commands Cartesian deltas toward FK poses (`executor.py:76-88`). Elbow and nullspace motion therefore need not follow the checked joints, and joint-space RRT detours are not reproduced for arm links. Per-step checks start from the measured q, but the target is the *planned* waypoint q.

**Improvement.** Execute joint paths with a joint-space or impedance controller, or check the Cartesian-OSC predicted configuration instead.

## 5. Readiness for the pending T04 can-obstacle launch (`SNAP/prep/pending_launch.json`)

**Will operations execute?** Yes, at least the first approach.
- The first transit keyposes plan in 1.6-2.1 s on the saved T04 frame (`prep/saved_input_check.json`, reproduced in `data/a14`).
- An H3-style pregrasp and a grasp descent to z 0.555 also plan (`data/a5` S1/S2).
- The old blockers (link0 start, local placement generation, opposing-pair grasp, unknown-space prohibition) are gone.

**Expected course.**
1. **Slow approach.** Each keypose takes about 25-110 actions at about 3.5 s per step (K1, K7): roughly 1.5-6.5 min per keypose.
2. **Grasp vetoes are likely.** A grasp at a realistic height is likely vetoed by `hand_collision` with a strong rim point: H3's own clean grasp at z 0.51 is (K3). Each veto costs a call.
3. **Closure can be flagged.** The fingers may hit unowned mug points, giving `gripper_sweep_blocked`; the gripper keeps closing during the hold steps (`executor.py:116-131`).
4. **Lift and transport are likely to stall at the start.** Causes: own unowned strong points inside the payload hull, the hull touching the adjacent can (white mug), or hand vs hull for handle grasps (K2). These are repeated `planning_unavailable` events that Astra cannot fix, so a user stop is the probable end.
5. **If transport does proceed, the can is not reliably avoided.** Can blocking depends on that frame's wrist view (0% strong during H3's approach frames), and the payload is missing in 26% of white-mug frames. Descents and placements are not stopped by the table (5/6 penetrations accepted).

**Calls and time.**
- If everything worked: about 5-7 calls, 1,040-1,400 actions, about 60-85 min of control. The first mug alone takes about 29-40 min.
- More likely: 8-20 calls within the first 15-30 min, with 0-1 mugs moved.
- GPU0, the default vision device, was 78% busy during this analysis, so refresh will probably be slower than RUN1.

**Collision risk.**
- Palm-mug contact is blocked only where the rim is strong; the same check also blocks clean grasps.
- Finger-table and held mug vs table/plate contacts are unprotected.
- Mug-can contact is protected intermittently.

**Recommendation.** Do not expect a useful T04 completion or speed signal from this build. If it is launched as a diagnostic, watch per-step `decision_latency_s` and start-endpoint failures, and keep the stop-file workflow ready.

## 6. Limits of this analysis

- **The robot never moved in RUN1.** All frames are at home, so the post-contact checks (a8-a10) use home-pose images with the mug on the table. They are proxies, not in-hand frames.
- **The can strength series is a proxy.** It comes from H3's 2048-point subsample, not from keypose-2 frames.
- **Step-time and step-count figures are models.** They are built from measured components (RUN1, offline timing on a loaded host) and H1's OSC calibration, not from a keypose-2 execution.
- **Real keyposes may differ.** H3-like keyposes stand in for real keypose-2 Astra outputs.
- **Truth was used only for scoring.** Layout truth comes from `output/layouts/libero_10_t4_can_obstacle_20261103_001/layout_manifest.json`.
