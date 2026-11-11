# H2 `foundation_harness`: planning, execution, agent loop and client

Key: `h2_planning_execution` · analyst w1 · written 2026-11-08 00:0x UTC · read-only analysis (no simulator, model or GPU use)

## 0. Which code this covers (important)

H2 was rewritten while this analysis ran.

| Version | Where | Status |
|---|---|---|
| **H2-v1 `foundation-harness-1`** (ran T04) | `runs/foundation_harness_t04_can_obstacle_001/source_at_launch/src/gpt_field/foundation_harness/` (md5-identical to the tree I first read at 23:2x) | Evidence-backed. All `file:line` below without a prefix refer to this frozen copy. |
| **H2-v2 `foundation-harness-keypose-2`** (in flight) | `src/gpt_field/foundation_harness/` modified 23:39 → 23:58 (client, executor, feedback, motion, robot_geometry, types, video, new `surface.py`, runner, doc, tests) | Snapshots kept in `SCRATCH/snap_234017`, `snap_234822`, `snap_000244`. Doc says "not validated in a full task". I ran the 23:48 snapshot offline on the saved arrays and code-read the 00:02 snapshot. |

v2 changes the division of labour: Astra now writes absolute world keyposes (≤12 per op), the planner checks only the next segment, unknown space and "uncertain" (non-cross-view) observed points may be traversed, the fixed base is skipped in scene checks, failures keep structured detail, and object clouds go through a density/normal/connected-component filter (`surface.py`). The 00:02 snapshot also starts projecting the current robot into each camera for self-masking, which is the direction my counterfactuals below support.

SCRATCH = `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1`. The scripts are in `SCRATCH/scripts/` and the outputs are in `SCRATCH/out/`. The repo was not modified.

## 1. How H2-v1 works as implemented

1. **Perception** (`perception.py`). Each refresh runs SAM 3.1 on the bootstrap or model queries and DA3 depth on both cameras. `build_geometry` produces about 521k scene points per frame. Points are attributed to objects by mask and cross-view support. Normals come from depth gradients and are flipped to face the camera.
2. **Client** (`client.py`). One strict JSON contract per call: `status/why/targets/operations/completion_conditions`. Targets carry SAM candidate IDs per view and optional region boxes. Operations are semantic (`pick_place/grasp/place/move/contact_move/release/observe`). The text history is persistent, plus the six most recent RGB pairs.
3. **Planner** (`motion.py`).
   - Grasp candidates come from pairs of observed points with opposing normals (≤96 points sampled, 5 approach axes).
   - Placement candidates must fit the payload's whole visible XY bounding box onto upward-normal support points: 9 probes within 12 mm and a height span ≤15 mm.
   - The pick-place chain has 9 keyposes, each solved by damped least-squares IK and connected by a straight joint edge or a 100-iteration RRT.
   - Every 0.055 rad sample runs a full collision and free-space check: convex hulls of every link and finger, the 6 mm margin, KD-tree ball queries on the scene cloud, and ≤160 probes that must be `estimated_free` (`allow_unknown_space=False`).
   - Each `plan()` call has a 12 s budget.
4. **Executor** (`executor.py`).
   - `act()` refreshes perception every step and asks the model when needed, then holds for one step.
   - It plans, then tracks the joint path as Cartesian OSC deltas: 2 cm per action in transit, 3 mm in contact, advancing waypoints only within 4 mm.
   - Before each action it re-checks the next segment against collisions.
   - Grasp and release are gated on RGB feature tracking plus PnP.
   - A planning failure clears the remaining operations and asks the model again. There is no attempt budget.

## 2. Run evidence (T04 can obstacle, `runs/foundation_harness_t04_can_obstacle_001`)

| Metric | Value | Source |
|---|---|---|
| Outcome | user stop; 19 actions, **0 keyposes executed, 0 grasp attempts**, path length 0.26 mm | `analysis.json`, `trial/summary.json` |
| Model calls | 12, all schema-valid; latency 125.0 s total, mean 10.42 s | `analysis.json` timing |
| Tokens | input 348,179 (4,383 → 51,893 per call); cached 3,584 (1.0%); output 4,173; reasoning 335 | `trial/summary.json` budget, `trial/requests/*.response.json` |
| Local time | vision 45.4 s, planning 21.6 s; control 206.6 s | `trial/agent_stats.json` timing |
| Cycle | 11 model-to-model gaps, mean **17.5 s** (10.5 s model + 7.0 s local) | `trial/foundation_harness/events.jsonl` (computed) |
| Perception refresh | 20 full refreshes, median **1.83 s** (1.44–3.39 s) | `vision/snapshots.jsonl` timings (computed) |
| Rejections | 8: 1× `no_observed_opposing_contact_pair`; 7× `no_complete_operation_path` (5 placement-zero pick_place, 2 moves) | events + `analysis/reconstructed_rejections.json` |
| Calls by purpose | 3 `status=observe` (queries/identity at steps 0, 5, 14), 6 pick_place variants, 2 viewpoint moves, 1 split grasp (withheld at stop) | events (`semantic_plan` answers) |

## 3. The v1 constraint stack: why every operation failed

The run's own analysis found two blockers: the start state, and placement. My counterfactuals (`SCRATCH/out/cf_planner.json`, `cf_ghost.json`) show **at least five independent, stacked blockers**. Each one alone was sufficient to prevent any motion.

| # | Blocker (v1) | Evidence | v2 status (23:48 / 00:02) |
|---|---|---|---|
| B1 | The fixed base (link0) is always "in collision" with its own depth points. Self-filter is 3 mm (`motion.py:121`) but the collision margin is 6 mm (`motion.py:24`). | Step 18: link0 has 67 points inside the hull, 61 at 0–3 mm, **74 at 3–6 mm**, 428 at 6–20 mm, 1,306 at 20–50 mm. All 8 reconstructed starts are unsafe at `link0_collision` (residuals 3.9–5.7 mm). Link0 never moves, so *every* configuration fails, including `observe`/`release` ops, which also go through `plan()` (`motion.py:541-549`). | Fixed: group 0 is skipped in scene checks. The start configuration is now `within_measured_robot_volume`. |
| B2 | Unknown-space veto. `estimated_free` needs ≥2 independent views terminating at cross-view-supported depth (`foundation_vision/geometry.py:34,368`). | In a tabletop grid (0.44×0.80×0.34 m at 2 cm, 16,031 points, frame 19): **43 (0.27%) estimated_free**, 79.9% insufficient rays, 10.4% behind surface, 9.4% surface. With B1 removed, the move [0,−0.15,0.05] still fails with `unknown_swept_space` (81 unknown probes). `_connect` raises without trying RRT (`motion.py:277`). | Flipped to allowed (`allow_unknown_space=True`). At 00:02, "uncertain" observed points are also traversable (see F7). |
| B3 | The arm's own depth "ghost" points 6–50 mm outside its hulls are kept as obstacles. | With B1 removed and the free check off, **all IK-feasible grasp configurations are blocked**: 14/14 (yellow-white mug, step 4), 15/15 and 8/8 (white mug, steps 16/18). Example: link5 has 17,046 hit points, of which 14,049 are unowned and within 5 cm of the *current* arm. The move at step 13 is blocked by a link6 ghost; the RRT burns 12.02 s and hits `planning_compute_budget`. | 23:48: still blocking. The 6 mm-shell ray filter removes only 100–113 points; the pregrasp hits a link5 ghost 6.7–11.6 mm outside the hull and burns the 12 s budget. 00:02: silhouette projection added (untested by me). |
| B4 | Non-pad finger and hand hulls are not exempt for the grasped object (`motion.py:197`; finger kinds `robot_geometry.py:173`). | finger1_collision has 2,036 hit points, 2,026 of them owned by the target mug. | Fixed for grasp/close/lift (finger+pad contact parts). |
| B5 | Placement gate uses the whole visible XY extent. | Payload extents are 23.9–26.3 cm because of outliers; robust p5–p95 is 6.1–9.4 cm. Support is 12.5×8.6–9.3 cm. Result: 0 placements → 5/5 pick_place rejected before any motion. | Local placement removed (model chooses); `surface.py` shrinks the white mug to 8.2×10.3×12.5 cm. |
| (B6) | Opposing-normal grasp rule with camera-facing normals. | Handle: `no_observed_opposing_contact_pair` (1/8). | Local grasp synthesis removed. |
| (B7) | Planner compute | One configuration check costs 0.12–0.13 s (v1) or 0.17 s mean / 0.24 s max (v2). One 12 s plan is enough for only about 70–100 checks. | Unchanged |

**Conclusion (a).** v1 was over-constrained by design, not by tuning. Self-geometry was not removed robustly (B1/B3). The free-space rule could almost never be satisfied with one external camera plus a narrow wrist camera (B2). The grasp and placement gates were all-or-nothing (B4/B5). All of this was checked before the first motion. Any one of B1, B2 or B3 alone guaranteed `no_complete_operation_path` for every manipulation.

## 4. Findings

### F1 — Robot self-geometry is treated as scene obstacles (critical, verified; v1 and v2-23:48)

**Mechanism.** DA3 depth on the robot body is centimetres off the true surface. Projected collision hulls align with the RGB (visual check in `SCRATCH/out/overlay/overlay_step18_{external,wrist}.png`). Even so, external-view robot pixels lie a median ~10 cm behind the hull entry along the camera ray (IQR 8.6–10.9 cm). That figure is an upper bound because it includes hull slack. About 24% of silhouette pixels are foreground in front of the robot, such as the red mug. Shell filters in 3-D (v1: inside hull + 3 mm; v2-23:48: inside hull + 6 mm, then ray occlusion) only see points within millimetres of the hull. The thousands of ghost points 6–50 mm away remain as hard obstacles around the current arm, so any motion that sweeps the arm through its own former position is rejected.

**Evidence.**
- Band counts per link, from E1 in `SCRATCH/out/cf_planner.json`.
- Grasp-config hits attributed to points within 5 cm of the current robot (`cf_ghost.json`).
- v2-23:48 pregrasp blocked by a link5 ghost 6.7 / 11.6 mm outside the hull (`cf_v2b.json`).
- Image-space masking catches 83–89% (no dilation) and 92–95% (6 px dilation) of near-robot external points, and 100% in the wrist view (`cf_selfmask.json`).

**Improvement (general).**
- Build the self-mask in image space. Render the current robot (FK + meshes + public calibration) into each camera.
- Drop depth pixels inside the silhouette unless they are clearly in front of the rendered robot depth.
- Add a flying-pixel filter at silhouette edges, and apply the same mask to the free-space evidence (`scene.geometry`), not only to the obstacle cloud.
- Add temporal consistency: points that move with the robot across frames are self.

The 00:02 snapshot starts this (`motion.py:135-195`, `visual_robot` envelopes, camera-inside-envelope skip). It still lacks a depth-error-aware tolerance and edge filtering.

**Counterfactual on v2-23:48** with a depth-tested mask applied upstream (silhouette dilated 6 px, 3 cm in-front tolerance; `cf_v2d.json`):
- Saved moves plan in 1.15 s and 0.66 s.
- Pregrasp 12 cm above the white mug plans in **1.72 s** (it previously hit the 12 s budget).
- Pregrasp 6 cm still fails: the goal configuration hits unowned points 2.6 cm from the arm at z≈0.81 (`cf_v2e.json`). The mask alone is therefore necessary but not sufficient; edge and temporal filtering are needed too.

**Validate** with replay on the archived T04 frames (see F13): the start configuration is safe, both saved moves plan, and pregrasps at 6/12 cm above each mug plan in under 3 s.

### F2 — Free-space policy: from "everything unknown is forbidden" (v1) to "unknown is allowed" (v2) (critical, verified)

**Mechanism.** In v1, a probe was free only if both cameras saw past it to cross-view-supported depth. Only 0.27% of the tabletop grid qualifies (max 1.3% at any height), so v1 could not move. The probe check also uses raw `scene.geometry`, which is not self-filtered: the arm's own surface is a "surface" for moved-arm probes. v2 removes the veto (`allow_unknown_space=True`, snapshot `motion.py:31`), so collision avoidance in unknown space now rests only on the points that happen to be observed in the current frame.

**Improvement.**
- Keep a persistent world-frame occupancy/TSDF map fused over all frames and both cameras, with free-space carving along rays.
- Mark the robot's executed swept volume as free, since the robot has just passed through it.
- Plan with a cost on unknown volume plus a speed and re-observation rule, instead of a veto or a blanket allowance.

**Validate** by checking the free-fraction growth of the grid over a recorded trajectory, and recording unknown-volume exposure per executed segment.

### F3 — v1 local grasp synthesis is gripper-blind and its candidates are not robust (high, verified)

**Mechanism** (`motion.py:308-358`).
- Pairs of index-sampled points must have opposing *camera-facing* normals.
- Approach options are projected off the closing axis and can point upward. Step 4 includes `approach=[0.72,0.03,0.69]` (approach from below the table), and step 16 includes `[0.73,0.39,0.57]`.
- There is no hand, finger or table collision filtering before IK.
- Every IK-feasible candidate collided at the grasp pose (B3/B4 plus table points at z≈0.41–0.44).

v2 removes local grasp synthesis entirely, so Astra must now write metric grasp poses (see F9).

**Improvement.**
- Run gripper-aware antipodal sampling on the coherent surface (`surface.py`), with tabletop-consistent approach sets (z-component ≤ 0, or horizontal), full gripper-model collision filtering, and ranking.
- Give the model the top-K candidates as numbered marks on both RGB images to choose from.

**Validate offline**: share of candidates that pass the gripper and finger collision check, and their IK; then measure grasp success live.

### F4 — v1 placement: an all-or-nothing whole-extent footprint gate (high, verified)

**Mechanism** (`motion.py:360-419`). The payload's full visible XY bounding box, including outliers and the handle, must sit within 12 mm of upward-normal support points at 9 probes. The 23.9–26.3 cm extents against a 12.5×9 cm visible support produced 0 placements in 5/5 ops. The height also uses the payload's min z: object points sit 4.3–7.4 cm *below* the support (min z 0.357–0.383 vs support median 0.426–0.431), so a found placement would have been 4–7 cm too high.

v2 removes local placement. Astra chooses the place pose, and `surface.py` reduces the extents (white mug 8.2×10.3×12.5 cm; yellow-white mug 12.3×14.2×9.0 cm).

**Improvement.** Keep a local placement *proposer and verifier* rather than a gate:
- Use the payload's robust bottom contact patch (lowest coherent slice) against a support polygon (alpha shape of upward points), tolerating partial visibility.
- Set height from the support plane plus the bottom patch.
- Never gate the approach on placement (v2 already does not).

**Validate** on replay: placements found for both mugs onto both plates in frames 5–19.

### F5 — Failure feedback was collapsed and is still not actionable (high, verified)

**Mechanism.** `executor.py:282` passes `detail=str(exc)`. The model received `no_complete_operation_path` seven times, and the evidence was dropped: 16 placement rejections, and the link0 hit with its point. Even the dropped evidence lacked quantities: placement rejections store only `{stage, reason}` (`motion.py:500`). Astra's 11 follow-ups varied semantic knobs (handle vs body, upright on/off, viewpoint shifts, split grasp), none of which could affect B1–B5.

v2 passes `{reason, evidence, keypose_index}` (snapshot `executor.py:285`).

**Improvement.** Add a failure taxonomy with `addressable_by ∈ {model, harness, perception}`. Include the blocking part, the owner class (self / table / target / other / unknown), the distance or overlap, and the nearest feasible alternative (for example, reachable z, or a free approach side). For harness-internal causes, do not ask the model at all.

**Validate**: every `operation_interrupted` row carries these fields, and the share of model calls spent on non-addressable failures is near 0.

### F6 — No attempt budget, no failure-signature de-duplication, no escalation (high, verified)

**Mechanism.** Every planning failure triggers `_fail(..., allow_local=False)` → `operations.clear()` → a new model call (`executor.py:282,306`). `local_replans=1` applies only to stage failures. There is no episode cap by design (configuration `episode_model_call_cap: null`). With a mean cycle of 17.5 s and 12 calls, the run spent 206.6 s with zero motion and would have continued until the user stopped it. v2 is the same.

**Improvement.**
- Compute a failure signature (reason, part class, owner class, op kind/object).
- After the same signature appears twice: run a local escalation (fresh perception with self-mask diagnostics, alternative candidate, viewpoint step), call the model only when new evidence exists, and after N no-progress calls stop with a diagnostic.

**Expected effect.** In this failure mode, ≤3 calls instead of 12 (≈ −9 × 17.5 s ≈ **−157 s**).

**Validate** in replay with synthetic repeated failures, then in live runs: calls per executed keypose.

### F7 — v2 collision-avoidance regression: weak or unknown points are traversable (high, likely)

**Mechanism.** In snapshot 00:02, only *strong* points reject motion (`motion.py:286-305`). Strong means owned by a coherent object surface, or per-frame cross-view support ≥2 without contradiction (`motion.py:128`). Unnamed added obstacles such as the T04 can and T08 bottle are unowned. Even with both cameras seeing the can (start pose), only **60–72%** of its depth points are strong (5 frames) (`cf_can_strength.json`). When the wrist camera turns away, support drops to 1 and the whole can becomes passable.

The 6 mm margin is also far below the cm-level depth error measured on the robot body (F1). Collision-avoidance claims for v2 therefore rest on per-frame luck.

**Improvement.**
- Persistent obstacle memory (F2): a voxel stays occupied once it has been cross-view supported in any frame, until rays carve it free.
- Inflate obstacles by a margin calibrated online from the robot-body depth error (the robot is known geometry in both views every frame).
- Treat uncertain points as a cost plus slow, re-observed motion, not as free.

**Validate.** Forbidden-contact samples on T04 and T08. Offline: obstacle persistence of the can across frames as the wrist camera moves.

### F8 — Even when unblocked, execution is ~10–20× slower per action than H1 (high, likely; unit costs verified)

**Mechanism** (all components measured):
- One configuration check costs 0.12–0.17 s, and `check_motion` runs at least 2 samples per action, so ≥0.25–0.34 s per action.
- A KD-tree and self-filter rebuild costs 0.43–0.83 s per new frame.
- A full perception refresh has a median of 1.83 s. It triggers whenever snapshot age exceeds 2.5 s (`perception.py:181`), and age is measured from capture start (`perception.py:192`). The refresh plus checks plus the sim step (0.125–0.16 s) already exceed 2.5 s, so a refresh happens every 1–2 actions.
- Result: **≈1.3–2.8 s per action, against 0.133 s (H1) and 0.137 s (H3)** non-model time per step (computed from `timing_components`).

Action counts are also inflated:
- The contact profile moves 3 mm per action (`executor.py:33,81`).
- Approach distance is `clip(0.5·|extent|, 3.5, 12) cm` (`motion.py:470`), which gives 12 cm with the inflated extents, i.e. ≥40 actions per approach and per place descent.
- Close needs ≥12 actions and release ≥8.
- Transit is stop-and-go: waypoints every 0.025 rad (`motion.py:269`) with a 4 mm arrival tolerance (`executor.py:72`).

Estimate: ≥150 actions per pick-place, i.e. ≥200–400 s per object before any model time. The prior uc7 analysis (`reports/agent_speedup_analysis_20261026/README.md`) measured that the arm moves only ~24% of each commanded OSC step at a 20 mm cap. H2's 4 mm waypoint-arrival rule will therefore need several actions per 0.025 rad waypoint, which makes this estimate optimistic. T04 has two objects, so the 180 s target is missed by construction.

**Improvement.**
- (a) A voxel/ESDF plus sphere-approximated robot model for ~ms checks, RRT-Connect with shortcutting, and checking only the segment ahead.
- (b) Asynchronous perception off the control loop: depth-only refresh for collision, mask tracking, full refresh only at contact transitions, and an age policy that does not count its own processing time.
- (c) Look-ahead (carrot) tracking with 5–10 cm transit steps (H1 runs a 0.10 m cap) and contact speed only for the last 2–3 cm.

**Expected effect.** Per-action non-model time ≈0.15–0.3 s, i.e. −80–90%.

**Validate.** Per-step timing breakdown in `summary.json` on the next full run.

### F9 — Division of labour: v1 gave the model knobs that could not matter; v2 gives it metric poses it may not get right (medium-high, likely/hypothesis)

v1 is verified: 3/12 calls were perception bookkeeping, and 8 re-asked semantic variants against geometric blockers the model could not see or change.

v2 (hypothesis) asks Astra for absolute world grip-site positions and right-hand quaternions from RGB plus calibration (snapshot `client.py:39,105,172`). Each keypose has 7 floats, there can be up to 12 per op, and nothing local proposes or corrects grasp or placement geometry. Earlier uc7 direction-semantics work showed frame and direction errors in model metric outputs.

**Improvement.** The harness *proposes* verified, ranked options (grasp candidates, placement spots, viewpoints, approach sides) as numbered marks on both images. The model *selects* and may nudge them (e.g. "grasp #3, place B, lift 8 cm"). The harness re-verifies and executes.

**Expected effect.** Fewer invalid first plans, much shorter outputs (F10), and errors become selections rather than metric guesses.

**Validate.** First-plan validity rate and calls per completed pick across the all-task sweep.

### F10 — Model latency is driven by output length; the context never caches (medium, verified)

**Latency.** Per-call latency ≈ **3.65 s + 19.5 ms × output tokens** (r=0.92). Correlation with input tokens is r=0.20. Answers are 245–442 output tokens because the strict schema forces every optional field (nulls) and the model re-lists targets with frame-specific candidate IDs each call. v2 adds about 45–60 tokens per keypose, so a 9-keypose pick-place call is likely ~15–20 s.

**Caching.** Input grows 4.4k → 51.9k tokens per call, and only 1.0% was cached. The prefix is identical for requests 1–6, yet nothing was cached. From request 7 the six-pair image window strips images from message 0, so the prefix changes every call. The other six-pair protocols show the same pattern (H1 2.5%, H3 0%), whereas astranomy-direct reached 36.6%.

**Improvement.**
- Per-kind compact schema (`oneOf`, no null-filled fields), omit unchanged targets, cap `why` at about 120 characters, and use parametric or candidate-index outputs instead of 7-float poses.
- A fixed, cacheable prefix (instructions + task + calibration), a rolling text summary, and only the current (and at most one previous) image pair.

**Expected effect.** −3 to −8 s per call, and −60–80% input tokens (this matters for subscription limits more than for latency). The same prior analysis found that a compact-output instruction removed long or hanging plan requests (0 of 12 with it, against 11 of 38 without). H2 has only "Return one compact JSON object".

**Validate** with per-call output tokens and latency in `requests/*.response.json`.

### F11 — Bootstrap and perception bookkeeping cost a call per task (medium, verified)

**Mechanism.** Generic splitting on "and" breaks coordinated adjectives. "yellow and white mug" became the seeds `yellow` (missing) and a duplicate of `white mug`. Other LIBERO sentences give `both moka pots`, `both the alphabet soup`, and `it`. The first call is therefore always a query fix. Calls at steps 0, 5 and 14 were `status=observe`. Step 14 corrected an automatic "unique candidate" wrist selection that had picked the wrong mug.

**Improvement.**
- Make the first call combined: queries plus intended operation. The harness segments locally, proceeds when each target resolves uniquely, and asks back only on real ambiguity.
- Split clauses on verbs ("… and put …") rather than on every "and".

**Expected effect.** −1 to −2 calls per task (−17 to −35 s).

### F12 — Plans are made in joint space but executed by Cartesian OSC (medium, likely)

**Mechanism.** `JointPathStage` converts the planned joint path to forward-kinematics poses and tracks them with `OSC_POSE` delta actions (`executor.py:46-89`; controller `libero.py:862-870`). OSC resolves redundancy through its own nullspace, so the executed joint trajectory, and hence the swept arm volume, need not match the collision-checked joint path. The per-step check tests a hypothetical straight joint segment from the measured q to the planned waypoint q.

**Improvement.** Plan in Cartesian space with an OSC-consistent redundancy model, or log measured versus planned joints at every waypoint and re-check the swept volume with measured joints. Shorter look-ahead segments also reduce divergence.

**Validate.** Joint-deviation statistics on the first executed motions (no motion has executed yet, so the magnitude is unknown).

### F13 — Tests could not catch any of the live blockers, and the v2 tests still cannot (medium, verified)

**What the tests cover.**
- `tests/test_foundation_harness_integration.py` (frozen, 5 tests) plans only in an **empty scene** (`np.empty((0,3))`), with a geometry stub that returns all `estimated_free` or all `unknown`.
- The executor test stubs out the planner (`SimpleNamespace(plan=…, check_motion=lambda…: safe)`).
- There is no pick_place, grasp, placement, failure-detail or real-perception test.

The in-flight file (225 lines, md5 `471fa4c2…`) keeps the empty-scene/stub pattern (lines 135-158), so it now asserts that unknown space is *allowed*. Every live blocker was untestable by construction: self points, the two-view free rule, the footprint gate, the opposing-pair rule and the string collapse.

**Improvement.** Add replay regression checks on 2–3 archived frames from this run (≈2.7 MB each; 44 are available). Use the `SCRATCH/scripts` style checks:
- the start configuration is safe;
- both saved moves plan in <3 s;
- pregrasps above each target plan;
- grasp keyposes respect finger/pad contact;
- planning failures carry structured detail;
- the can stays an obstacle.

They run on CPU in ~1–2 min with no model or simulator. Use them as a development aid, not a launch gate, consistent with the user's live-first rule.

### F14 — Older verified defects (low, verified)

- **Image mirroring.** Not observed. Astra's "left plate" is the external-image-left candidate, which sits at world y=−0.31, consistent with the uc7.1 viewer frame (right=+y).
- **`sort_keys` schema reordering.** Not applicable. H2 goes through the ordered transport (`agent_speedup_v2_user_check_transport.py:480-485`), and the wire schema order is preserved (`status, why, targets, operations, completion_conditions`).
- **Zero-rotation stale OSC goal.** Mitigated by `HoldingActionAdapter` (`contact_skills.py:180-192`).
- **Axis-angle sign jump.** Present. state8 flips at step 9, and the `robot_pose_wxyz` sent to Astra flips sign between requests 6 and 7 with no motion: [0.0,−0.9996,0.0004,0.0287] → [0.0,0.9996,−0.0005,−0.0287] (`astranomy_agentic_motion.py:89-99` does not canonicalise). This is harmless for v1 planning (rotation matrices), but v2 asks the model to write absolute quaternions, so canonicalise to w≥0 before sending.

## 5. Ordered fix plan

**Stage A: first executed grasp.** Without these, nothing moves or every motion is unsafe.
1. Image-space robot self-mask with depth test, edge filtering, and application to free-space evidence (F1).
2. Persistent occupancy with carving; unknown and uncertain space as a cost, not a veto or free (F2, F7).
3. Contact semantics for finger, pad and payload by stage. v2 has these; keep them.
4. Structured, classified failure feedback plus a signature-based attempt budget, so unblocking work does not burn calls (F5, F6).
5. Replay checks on archived frames for each item above before the next live run (F13).

**Stage B: speed.** Target ≤0.3 s non-model per action and ≤6 s per call.
6. Fast collision representation (voxel/ESDF plus robot spheres) and RRT-Connect (F8a).
7. Asynchronous perception with depth-only updates and mask tracking; fix the age rule (F8b).
8. Look-ahead tracking; transit 5–10 cm per action; contact speed only for the last 2–3 cm; approach distance from robust extents (F8c).
9. Compact contract and cacheable prefix (F10); combined first call (F11).

**Stage C: few attempts and robustness.**
10. Harness-proposed, verified candidates with model selection (F9); local placement verifier using the bottom patch (F4); gripper-aware grasp candidates (F3).
11. Execution-space consistency logging and re-checks (F12); quaternion canonicalisation (F14).

## 6. Expected performance after Stage A+B (estimate, to be measured)

For each T04 object: about 2 model calls (combined plan plus one checkpoint) at 6–10 s each, about 120–180 actions at 0.15–0.3 s each, and planning under 1 s per segment. That gives ~40–75 s per object, i.e. ~100–150 s for T04. This is an estimate only. v1 never moved, and the per-action unit costs are taken from this run, H1 and H3.

## 7. Artifacts (all under SCRATCH)

- `scripts/cf_planner.py` → `out/cf_planner.json`: start-state bands, moves without the base, free-space grid, grasp geometry, check costs.
- `scripts/cf_ghost.py` → `out/cf_ghost.json`: grasp-config hit attribution (self ghost vs object vs table).
- `scripts/cf_selfmask.py` → `out/cf_selfmask.json`: silhouette capture of near-robot points.
- `scripts/cf_v2.py`, `cf_v2b.py`, `cf_v2d.py`, `cf_v2e.py` → `out/cf_v2*.json`: v2-23:48 on saved arrays, with and without masking.
- `scripts/cf_can_strength.py` → `out/cf_can_strength.json`: per-frame strong fraction of can points.
- `scripts/cf_depth_error.py` → `out/cf_depth_error.json`: robot-pixel depth offset along rays. This is an upper bound because it includes hull slack.
- `scripts/overlay.py` → `out/overlay/overlay_step18_{external,wrist}.png`: kinematics/calibration alignment check.
- `snap_234017/`, `snap_234822/`, `snap_000244/`: in-flight H2 source snapshots; `pkg_v2/`: scratch package used for v2 replays.

**Not covered.** Perception internals beyond what the planner consumes (H3 and other analysts). The 00:02 v2 silhouette implementation is code-read only, not executed. No live or simulated motion exists for H2, so F8, F12 and the v2 pose-accuracy items are derived rather than measured.
