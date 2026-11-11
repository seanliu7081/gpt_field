# H2 `foundation-harness-keypose-2`: perception and geometry (key `h2k2_perception_geometry`)

Date: 2026-11-08. Independent second opinion. I did not read the W1 H2 reports.

**Method.** Read-only. No simulator, Astra or GPU use. All numbers come from offline CPU replays on saved run-001 public inputs: RGB, calibration, saved DA3 depth, saved object points and evidence, measured robot state. The replays use the keypose-2 classes.

**Truth use.** Truth was used **only for scoring**, never as a planner input. Sources:
- table height and can bounds from the layout manifest. It is byte-identical for the H2 and H3 T04 runs: `output/foundation_harness_preparation_001/inputs/layout_manifest.json`, sha256 `9f5d5661…`.
- object collision AABBs from the LIBERO asset XMLs. These are written to `w2/truth_boxes.json`.

**Where things are.**
- Scripts: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts/k2_*.py` (plus `truth_boxes.py`).
- Outputs: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/k2_*.json` and `*.png`.
- Every script calls `k2_common.verify_snapshot()`. It aborts if the live tree differs from the snapshot.

Code citations are `file:line` in the snapshot `h2_keypose2_snapshot/src/gpt_field/foundation_harness/`, unless prefixed `fv/` (= `/workspace/gpt_field/src/gpt_field/foundation_vision/`).

## 0. Snapshot check

- **Recorded hashes match.** All 11 files in `prep/implementation.json` `changed_source_sha256` match the snapshot. They also match the live tree, re-checked at 2026-11-08T00:37Z.
- **Unchanged since run 001.** `perception.py` and `__init__.py` are not in the changed list. Both are byte-identical across three places: the snapshot, the live tree, and run 001's `source_at_launch/`.
- **foundation_vision unchanged.** All six `foundation_vision` modules in the live tree equal run 001's frozen copy.
- **No divergence.** The live `foundation_harness` has not diverged from the snapshot. Last modification was 23:58:34Z on motion.py.
- **What keypose-2 changed in my area:** `surface.py` (new), `robot_geometry.py` (visual envelopes), `types.py` (summary), `video.py` (strong/uncertain colours), and how `motion.py` consumes perception. Every perception input is identical to run 001.

## 1. How it works now (as implemented)

### Perception (unchanged)

On each refresh, `HarnessPerception.update` (perception.py:168-234) does the following:
1. Runs SAM 3.1 for every target query on both views. Masks with score > 0.5 are kept (fv/backends.py:239-260).
2. Runs DA3 on both views. Metric scale is one ratio per frame: known baseline / predicted baseline (fv/backends.py:315-354).
3. Back-projects every valid pixel of both views, about 521k points (perception.py:212).
4. For each point, counts other-view support and contradictions inside a ±(2 cm + 2 %·depth) band. This count ignores masks (fv/geometry.py:32-34, 464-474).

Target masks come from one of three sources:
- an Astra candidate id;
- a histogram/IoU or world-point re-association;
- the only candidate for the query in that view (perception.py:308-367).

Target points are the mask points with no scene contradiction and no cross-view mask conflict (perception.py:446-463). Ownership marks exactly those points. Points claimed by two non-related targets become unowned (perception.py:582-596).

### Surfaces (new, surface.py)

The steps are:
1. Eligible points are finite and have either a valid local normal or cross-view support (:147-159).
2. Per view, 8-NN radius ≤ max(2·median, median + 3·1.4826·MAD) (:177-210).
3. Normals agree: |cos| ≥ 0.5 with at least half of the usable neighbours (:200-204).
4. A 16-NN graph links points within 2× the local scale when their normals agree. The graph is then split into connected components (:218-228).
5. The dominant component is chosen with cross-view-supported points weighted ×2. It needs ≥ 60 % of the mass and the second component < 50 %. Otherwise the output is **empty** (`ambiguous_components`, :250-255).

The output is cached by content hash (:97-140).

The patch is used in four places:
- as `planning_surface_*` in Astra's scene summary (types.py:39-57; client.py:48 tells Astra to prefer it);
- to mark owned points as strong (motion.py:111-120);
- as the payload convex hull at a close keypose (motion.py:532-551) and for confirmed attachment (feedback.py:191-195);
- as tracker seeds (feedback.py:71).

### Planner view of the scene (motion.py `_prepare`, :96-208)

**Strong** means one of two things:
- an owned point that lies in its object's patch; or
- support ≥ 2 with zero contradictions (:110-130).

Everything else is **uncertain**. Uncertain intersections are counted and *allowed* (`allow_unknown_space=True`, :31, :294-306).

**Self-filter.** It applies to unowned points only. Per view and per robot body:
1. The current-pose visual envelope is projected to a convex silhouette. The envelope is the merged convex hull of the body's visual geoms with 98 supporting planes (robot_geometry.py:218-254).
2. Points inside the silhouette whose depth is at or behind the ray entry − 6 mm are removed (:141-195).

Base shapes (group 0) are skipped in scene checks (:271-272). Collision checks use the collision-mesh hulls (:72).

## 2. Run-001 defects: status in keypose-2

| # | Run-001 defect | Status | Evidence |
|---|---|---|---|
| D1 | Every start/move configuration rejected. Observed points sat 3.9-5.7 mm inside link0's 6 mm dilation, outside the 3 mm self-removal (run-001 `analysis/geometry_explanation.json`). | **Fixed** | All 4 run-001 link0 blocking points are now removed by `link0_visual_envelope`. About 70 points are kept inside link0's hull + 6 mm and none is strong (`k2_link0_points.json`). Base shapes are also exempt (motion.py:271-272). |
| D2 | 0 placement candidates. The raw XY extrema footprint was 26.3×23.4 cm for the mug against a 12.5×9.3 cm plate. | **Replaced by a new risk** | The local placement generator is gone; Astra proposes place poses. The surface.py patch is 12.3×14.2×9.0 cm against a true AABB of 10.6×14.0×10.3 cm, with 98.5 % of patch points inside truth + 1 cm (`k2_surface_all.json`). But the same patch now forms a payload hull that blocks its own lift (F1). Support patches are also DA3-biased (F4). |
| D3 | Handle grasp failed: "no observed opposing contact pair". | **Replaced by a new risk** | There is no local grasp synthesis now; Astra writes metric grasp poses. These can be self-rejected (F5) and carry DA3/model error (F4). |
| D4 | Structured reasons were lost (`str(exc)` became `no_complete_operation_path`). | **Fixed** | The structured reason and evidence now reach feedback (executor.py:283-288; motion.py:598-605). |
| D5 | Robot and background surfaces stayed in the obstacle cloud. The prep debug showed link6/link5 hits at z 0.81-0.84 m. | **Replaced by a new risk** | Near links are removed (72.7k points per frame). Far links and the background are kept. They no longer block only because single-view points became "uncertain" (F3, F2). |
| D6 | DA3 background bias and an unstable per-frame scale. | **Still present** | Table depth error: external +2.7-2.9 %, wrist +19.6-20.3 % (`k2_da3_table_bias.json`). The static external scene drifts by up to 12 cm between frames (`k2_h3_scale_drift.json`). |
| D7 | SAM prompt form and identity problems. "white mug" and the seed "yellow" returned 0 candidates. 3 of 12 calls went to identity. A wrong wrist mug was auto-bound at step 14. | **Still present** | perception.py and `bootstrap_targets` are unchanged (client.py:269-290). |
| D8 | Synchronous perception on every step: 20 full refreshes for 19 actions, 1.83 s median. | **Still present, worse** | A new `_prepare` costs about 1.1 s per new frame (F7). |
| D9 | 12 calls, 0 executed operations. | **Partially fixed** | The prep shows Astra keyposes connecting in about 2 s with 0 strong hits. New blockers appear at grasp and lift (F1, F5). |
| D10 | Obstacles exist only as unowned depth. The can was never a target. | **Replaced by a new risk** | Before, every observed point blocked. Now the can blocks only while it is two-view consistent (F2). |

**Answers to the specific questions:**
- **link0 phantom collisions:** removed by the silhouette filter, and the base is exempt anyway.
- **link5/link6 phantom collisions: not removed.** The points at z 0.81-0.84 m (`prep/previous_astra_direct_debug_v2.json`; `prep/saved_input_check.json` example points) trace to these pixels:
  - external view, rows 1-22 and columns 181-198. These are upper-arm (link4) pixels at the top image edge, grey RGB 120-181.
  - DA3 put them at about 0.76 m, while the robot model's ray entry is 0.272 m farther away. The filter therefore treats them as "predicted foreground".
  - support is 1 and there are no contradictions, so they are uncertain.

  The table top is z 0.437 m and the tallest real object is the red mug at 0.569 m. These points are robot self-surfaces with phantom depth, not objects (`k2_phantom_trace.json`; overlay `external_zoom_self_filter_step13.png`).

## 3. Findings

### F1 (critical). The held object's own unowned surface blocks every post-close lift. A DA3-raised rim also makes the payload hit the hand. *Latent since run 001; now on the critical path.*

**Goals:** few attempts, fast, collision avoidance.

**Mechanism.**
1. A close keypose stores `payload_after`. Its points are `planning_surface(obj)` expressed in the gripper frame (motion.py:532-551).
2. The next keypose (lift) is planned with this payload (executor.py:340-347, 416-429).
3. `_configuration` checks two things:
   - payload against every robot shape except fingers and pads, with a 6 mm margin (motion.py:259-264);
   - payload against strong scene points, exempting **only points owned by the object** (motion.py:278-293).
4. Ownership leaves out part of the object's own visible surface:
   - mask-conflict points (perception.py:446-463);
   - points from views with no bound mask.

   Those points are still strong, because support counting ignores masks (fv/geometry.py:464-474). They also lie inside the object's own hull.
5. The endpoint check runs before any search (motion.py:418-426). So the lift fails at its start configuration.
6. The confirmed `held` payload is built from the same points (feedback.py:191-195).

**Evidence** (offline):
- **Self-blockers in every frame.** Every saved frame with a usable mug payload (18/18) has 211-9,650 unowned strong points inside the payload hull + 6 mm. Between 99 % and 100 % of them lie inside the **true target box** (`k2_payload_selfblock_all.json`).
- **Ownership coverage.** Owned share of the target's strong points is 0.88-0.97 with both views bound and 0.28-0.29 with the external view only.
- **Yellow-white mug (frame 5):** the close configuration and a 5 cm lift are both blocked: `observed_obstacle_intersection`, part `held_visible_surface_envelope`, owner `''` (`k2_payload_test.json`).
- **White mug (frames 16 and 20):**
  - the payload hull top is at 0.563-0.568 m, against a visual-mesh rim at 0.549 m;
  - the result is `payload_robot_collision` with `hand_collision` at any grasp depth ≥ 0.5 cm (frame 16) or ≥ 0 cm (frame 20) below the collision rim;
  - even at a 0 mm margin it collides from 1-2 cm depth (`k2_payload_detail.json`).
- **Paradox.** When the patch is empty, the payload is unknown and payload checks are skipped entirely (motion.py:256, 267). This happened in 3 of 18 frames.

**Severity / confidence:** critical; likely. The mechanism is verified on saved frames. No post-close frame exists, because run 001 never grasped.

**Improvement** (general; removes a veto rather than adding one):
- Treat the payload like the robot. At closure, every point inside the payload hull at the closure pose *is* payload. Carry that set rigidly with the gripper.
- Add the payload hull to the per-view silhouette self-filter (motion.py:135-195).
- Judge payload-vs-hand overlap relative to the closure configuration. Overlap that already exists at closure is allowed, using the same logic as the fixed-base rule (motion.py:268-272).
- Never let "payload unknown" disable payload collision checks.

**Expected effect:** 0 self-blockers by construction on the 18 frames. It removes a dead end that Astra cannot fix by changing poses, because the failing configuration is the current one.

**Validation:**
1. Re-run `k2_payload_selfblock_all.py` and `k2_payload_detail.py`.
2. In the next live run, log the first lift check after closure.
3. Run the all-task sweep.

### F2 (critical, collision). Only two-view-consistent surfaces block. The table, the red mug, the right plate and anything outside the wrist view are non-blocking. *New in keypose-2.*

**Goals:** collision avoidance.

**Mechanism.**
- Strong means an owned patch point, or support ≥ 2 with no contradiction (motion.py:110-130).
- Uncertain intersections are allowed (motion.py:31, 303-306). The old revision blocked all observed points (old motion.py:33).
- Support comes only from the *other* view's depth (fv/geometry.py:464-474). So an obstacle must sit in the narrow, moving wrist view at the same time, at a consistent DA3 scale.

**Evidence:**
- **Strong share.** 44,817 strong against 403,666 uncertain at frame 14 (10.0 %). Across 20 frames the share is 10.0-11.5 % (`k2_scene_audit.json`).
- **What is strong, by truth region** (`k2_strong_composition_step13.json`):

  | Region | Strong share |
  |---|---|
  | can | 84 % |
  | white mug | 81 % |
  | yellow-white mug | 62 % |
  | left plate (owned) | 91 % |
  | table band | 1.3 % (3,585 of 283,965) |
  | red mug | 2.5 % |
  | right plate | 0 % |

- **Intrusion test.** The grip site was placed via IK inside true objects, and `_configuration` was called (`k2_intrusion_test_step13.json`):
  - **Red mug:** allowed at 25 %, 50 % and 90 % of its height, with 3,049-6,079 uncertain samples.
  - **Table:** fingers 2.7 cm into the table are allowed.
  - **Right plate:** allowed at every height.
  - **Shallow intrusions:** the top centimetre of the can and of the yellow-white mug is allowed.
  - **Deep intrusions** into the can and both mugs are rejected.
- **Other robot poses** (H3 T04 snapshots, same SAM/DA3/geometry code) (`k2_h3_pose_strength.json`):
  - the scene strong share swings from 0.8 % to 47.5 %;
  - the can is 34/34 strong at frame 0, 0/2 at frame 17, and has no sampled points at its true location in 5 of 11 frames.

**Severity / confidence:** critical; verified for the classification, likely for live contacts.

**Improvement** (general):
1. Keep a persistent fused occupancy map. The static external camera fuses trivially, and each wrist pose adds coverage. An obstacle seen once stays blocking until motion evidence clears it.
2. Single-view surfaces become obstacles inside a per-view along-ray uncertainty band, after depth calibration (F4).
3. Model the support plane explicitly. It blocks non-finger links, and fingers may touch it only at contact keyposes.
4. Discover instances class-agnostically, so non-role objects (can, red mug) become owned obstacles.
5. Keep "uncertain" for unobserved space only.

**Expected effect:** true obstacles are enforced wherever the wrist looks.

**Validation:**
1. Turn `k2_intrusion_test.py` into a truth-scored offline gate over all saved H1/H2/H3 frames. It should report the rejection rate inside each object and false blocks on free-air probes.
2. Run the all-task sweep with forbidden-contact counts.

### F3 (high). The silhouette self-filter removes near links only. Far links and the room background survive as "foreground" floating 0.6-0.84 m above the table. *New filter; the phantoms are now hidden rather than removed.*

**Goals:** collision avoidance, few attempts.

**Mechanism.**
- A point inside the robot silhouette is removed only if its DA3 depth is at or behind the modeled entry − 6 mm (motion.py:141-195).
- DA3 compresses the depth of the far links (base column, upper arm) toward the gripper.
- Wall and TV pixels sit at about 0.9 m, which lands them inside the workspace.

**Evidence:**
- **Removal counts.** 72,738 points removed per frame:
  - external: 22,753;
  - wrist: 49,985, including all 41,391 pixels of wrist rows 430-511 (`k2_region_probe.json`).
  - The wrist filter works well (`wrist_selfremoved_red_strong_green_frame14.png`).
- **Kept far-robot points.** External points kept inside the silhouette are 0.279 / 0.289 / 0.325 m (p10 / p50 / p90) in front of the modeled surface.
- **Base column** (rows 120-195, columns 200-300): 5,086 kept against 2,414 removed. The kept median position is (-0.18, -0.01, 0.60).
- **Floating cloud.** 27k-38k external points float above z 0.60 m over the table in every frame, and 0 of them are strong (`k2_scene_audit_all.log`).
- **Overlay.** Red = removed, blue = kept as foreground, magenta = floating background (`external_zoom_self_filter_step13.png`).

**Severity / confidence:** high; verified. These points stop blocking only because of F2. Making "uncertain" blocking (needed for F2) would bring back the run-001 self-blocking.

**Improvement:**
- Use self-pixels at FK-known depth as metric anchors for a per-view affine depth correction (H3 F4).
- Then label every pixel inside the projected robot silhouette as robot, unless the other view confirms foreground there.
- This replaces a fixed 6 mm depth gate with evidence.

**Expected effect:** the floating cloud of about 36k points per frame drops toward 0. That is a precondition for F2.

**Validation:** count kept robot pixels against FK-rendered silhouettes on saved frames (`k2_region_probe.py`).

### F4 (high). DA3 bias and per-frame scale drift still feed both the planner and Astra's keyposes. Astra is now told to prefer `planning_surface_*`. *Inherited.*

**Goals:** collision avoidance, few attempts, zero-shot.

**Mechanism.**
- One baseline-ratio scale per frame (fv/backends.py:333). There is no per-view correction.
- perception.py is unchanged.
- Astra now authors every metric keypose, and client.py:48 points it at the patches.

**Evidence:**
- **Table bias (open table, ray-plane truth).** These numbers match H3's +3.5 % / +18.1 % (`k2_da3_table_bias.json`):

  | View | Relative depth error | Height error |
  |---|---|---|
  | external | +2.7 % to +2.9 % | about -1.5 cm |
  | wrist | +19.6 % to +20.3 % | -6.7 to -7.0 cm |

  About 35 % of external points are contradicted.
- **Left-plate support patch.** It sits 1.5-2.5 cm *below* the table top, about 3 cm further along the external camera axis, and only 15-66 % of its points lie inside the true plate box + 1 cm (`k2_surface_all.json`).
- **White mug.** Its patch top is 1.4-2.5 cm above the true rim.
- **Scale drift.** Between H3 frames, as the wrist moved, the table height of the *static* external view changed from -0.5 cm to -12.3 cm (`k2_h3_scale_drift.json`).

**Severity / confidence:** high; the bias is verified, the effect on keyposes is likely. The H1 baseline already shows model-geometry errors of 3.4-5.7 cm (h1_zero_shot.md F3).

**Improvement:**
- Feed ≥ 3 views to DA3 (add a previous wrist frame).
- Fit a per-view affine depth correction from robot self-pixels and the shared support plane.
- Fix the static external scale once and re-estimate only the wrist.
- Give Astra uncertainty bands, not raw extrema.
- Add two-view triangulation of named pixels (H1 F3 "locate").

**Expected effect:** table bias ≤ about 3 % in both views, and a stable static scene. Sub-cm keypose grounding is a hypothesis.

**Validation:** truth-scored offline replay (table z, AABB errors). Re-run DA3 when GPUs are free.

### F5 (high). Targets block their own grasps: unowned target points and a DA3-raised rim become vetoes. *Latent; now on the critical path.*

**Goals:** few attempts, fast.

**Mechanism.**
- Fingers and pads may touch only *owned* points of allowed contacts (motion.py:283-284).
- The hand (kind `tool`) is checked against all strong points, including owned ones (motion.py:267-293).
- Unowned target points come from mask conflicts and from wrong or unbound views. One source is the per-view unique-candidate auto-binding (perception.py:353-355).

**Evidence** (`k2_grasp_selfblock.json`): a top-down rim grasp at the true mug centre, with the planner's grasp allowances.

| Case | Ownership | Result |
|---|---|---|
| Yellow-white mug, frame 5 | owned in both views | allowed at 0, 1.5 and 3 cm depth |
| White mug, frame 14 | wrong wrist identity, owned share 0.28 | blocked at *every* depth by unowned strong points |
| White mug, frames 16 and 20 | both views owned | blocked at ≥ 1.5 cm depth: the hand hits owned points at z 0.556-0.561 m, 0.7-1.2 cm above the visual rim |

In run 001, the call at step 14 was spent fixing a wrongly auto-bound wrist mug (`trial/foundation_harness/events.jsonl`).

**Severity / confidence:** high; verified offline, likely live.

**Improvement:**
- Grow ownership geometrically: assign every strong point within the target's coherent patch neighbourhood to the target. Mark overlaps as shared, not unowned.
- Erode masks by 1-2 px and use robust upper envelopes (H3 F7), so rims are not raised.
- Do not auto-bind a unique candidate that conflicts with the instance bound in the other view.

**Expected effect:** valid grasps are no longer self-rejected, and fewer identity calls are needed.

**Validation:** run `k2_grasp_selfblock.py` over all frames, then a live run.

### F6 (medium). surface.py's comparable-components veto returns an empty surface, flips between frames, and silently disables payload checks and tracking. *New.*

**Goals:** few attempts, collision avoidance.

**Mechanism.**
- surface.py:250-255 returns empty when the dominant component has < 60 % of the mass or the second component is ≥ 50 % of it.
- An empty patch has three effects:
  - the payload becomes unknown (motion.py:545-548, 256), so payload checks are skipped;
  - the tracker cannot seed (feedback.py:72-76);
  - owned points lose patch-based strength.

**Evidence:** white_mug at frames 14 and 15 had 3 components with mass fractions 0.621 and 0.585, and both returned empty (`k2_surface_all.json`). 3 of 18 mug-payload frames were "unknown".

**Severity / confidence:** medium; verified.

**Improvement:** remove the veto. Return the union, or labelled patches, of supported or mask-adjacent components. Use the union hull for the payload.

**Expected effect:** no frame-to-frame flips, and the payload is always modelled.

**Validation:** re-run `k2_surface_all.py`.

### F7 (high, FAST). Synchronous SAM + DA3 on effectively every control step, plus the new self-filter, makes each 2 cm / 3 mm step cost about 3.5-4.5 s. *Inherited, worsened by `_prepare`.*

**Goals:** fast.

**Mechanism.**
1. A refresh triggers when the snapshot is older than `max_age_s × 0.5` = 2.5 s, or every 10 observations (perception.py:181-183). The runner keeps the 5 s default (evaluate_foundation_harness.py:54).
2. Each refresh re-runs, for all targets: SAM (perception.py:198-203), DA3, `build_geometry`, and a 2.3 MB npz save (perception.py:233, 610-642).
3. Every new frame re-runs `_prepare` (motion.py:96-208).
4. Refresh plus `_prepare` takes about 3 s, which is more than 2.5 s. So the next step is always "old" and refreshes again.

**Evidence:**
- **Run-001 refresh cost:** median 1.83 s (SAM 0.87, DA3 0.34, geometry 0.32); first frame 3.40 s (`snapshots.jsonl`).
- **`_prepare`:** 1.07-1.37 s of CPU (`k2_scene_audit.json`).
- **npz save:** 0.19 s, 2.27 MB per refresh.
- **surface.py:** 0.02-0.13 s per object.
- **Measured step cost:** run-001 non-model wall time per step was a median 3.56 s (mean 4.07 s).
- **Step sizes:** transit 2 cm per step, contact 3 mm per step (executor.py:32-33).

**Estimate.** A two-mug T04 needs about 210-220 steps under stated assumptions:
- 13 + 22 + 14 + 7 + 12 + 19 + 10 + 7 steps per mug (pregrasp, grasp, close, lift, transport, place, release, retreat);
- plus hold steps after model calls.

At about 3.5-4.5 s per step that is roughly 13-17 min of control, before model time.

**Severity / confidence:** high (critical for the 3-minute goal). The components are verified; the episode total is an estimate.

**Improvement:**
- Run perception asynchronously and decide staleness by robot or object motion events, not wall clock (drop the 2.5 s trigger).
- Cache the static external geometry and re-run only the wrist.
- Track masks with SAM 3.1 video mode instead of re-prompting.
- Limit `_prepare` to points near the next segment's swept volume.
- Save asynchronously.

**Expected effect:** per-step overhead falls to about 0.1-0.3 s (hypothesis).

**Validation:** timing replay of recorded observations, then live T04 timing.

### F8 (high). Grammar seeds and text-query identity still cost calls, miss non-task obstacles, and lose the held object once it moves. *Inherited.*

**Goals:** zero-shot, few attempts.

**Mechanism.**
- `bootstrap_targets` splits the sentence on conjunctions and prepositions (client.py:269-290).
- Each target is a text prompt with a 0.5 score threshold.
- Re-association needs a histogram match plus IoU in a fixed camera, or the old world points (perception.py:288-306). A moved object therefore becomes `selected_region_lost`. Its points turn unowned and feed F1 during transport (likely mechanism, not observed).
- Obstacles exist only if Astra names them.

**Evidence:**
- **T04 seeds** are `white mug`, `left plate`, `yellow` and `right plate`. Two of them, "yellow" and "white mug", had 0 candidates in every frame where they were requested.
- **"textured ceramic mug"** returned 3 external candidates in all 15 frames.
- **3 of 12 run-001 calls** (steps 0, 5, 14) were perception-identity calls.
- **7 of 10 LIBERO-10 sentences** yield malformed seeds: "it" in 4 tasks, "both the alphabet soup", "right of the plate" (`k2_bootstrap_seeds_libero10.json`).
- **The can** was never a target in run 001.

**Severity / confidence:** high; verified for seeds and identity, hypothesis for losing the moved object.

**Improvement:**
- Build one category-level instance inventory, plus a class-agnostic object sweep.
- Give instances stable ids, tracked on the static external stream; associate the wrist using its known pose.
- Astra binds roles once, by instance id.
- All non-role instances become obstacles automatically.
- Once grasped, the held object becomes the payload.

**Expected effect:** about 25 % fewer calls, and obstacles handled zero-shot.

**Validation:** offline SAM replay of saved RGB pairs (when a GPU is free), scored against truth.

### F9 (low-medium, FAST). The scene summary roughly doubled per object and is kept in history. *New.*

**Evidence:**
- Each object is 4.0-5.4 kB, of which `planning_surface_evidence` is 2.1-2.65 kB. It includes a 276-character static `rules` block.
- Raw extrema are sent twice (types.py:44-55; `k2_summary_size.json`).
- Run 001 used 348k input tokens over 12 calls.

**Improvement:**
- Send compact per-object facts.
- Move the static rules into the instructions.
- Drop the duplicate extrema.
- Replace superseded summaries in history.

**Cost / effect:** small cost; saves about 2-3 kB per object per call.

## 4. Readiness of the pending T04 launch (`prep/pending_launch.json`)

**Will operations execute?** Probably the first ones.
- Every run-001 blocker in my area is gone (D1-D4).
- The prep connects the formerly rejected 5 cm move in 1.98 s, and the old-success first keypose in 2.07 s.
- That second connection passed 220 "uncertain" samples. Their two examples are the link5/link6 phantom points of F3.
- So a pregrasp or viewpoint transit with `observe_after` should run.

**Most likely failure modes, in order:**
1. **Lift dead end after the first close (F1).** The lift-start check fails on the mug's own unowned strong points (seen in every saved frame), or on payload-vs-hand overlap if the rim is reconstructed high.
   - Astra receives `held_visible_surface_envelope` or `payload_robot_collision`. No new pose can fix a failing start configuration.
   - Expect repeated calls, or release-and-regrasp loops.
2. **Grasp descent rejected (F5).** This happens with partial or wrong ownership, or a raised rim. The white mug cannot be grasped deeper than about 1-1.5 cm below the rim.
3. **Identity calls (F8).** Expect about 1 call in 4.
4. **Pace (F7).** About 3.5-4.5 s per step. The pregrasp alone takes about 1 min.

**Collision risks.**
- **Allowed intrusions (F2).** The planner allows intrusion into the red mug, the right plate, the table (2.7 cm finger penetration was allowed) and the top centimetre of the mugs and the can.
- **The can** is enforced only while the wrist sees it consistently.
- **Biased poses (F4).** Astra's poses come from DA3-biased patches: plate 1.5-2.5 cm low, rim 1.4-2.5 cm high, static drift up to 12 cm. Descents may reach the plate, table or mug early. 3 mm contact steps limit the impact but not the forbidden-sample count.

**Expected calls and time** (estimate):
- First call to a planned pregrasp: 1-2 calls.
- Grasp chain: 2-4 calls.
- Then a likely dead end at the lift.
- Most likely outcome: more than 10 calls and more than 10 minutes without completing T04, ending in a user stop.
- Even with F1 fixed, perception pacing alone puts a successful T04 near 13-17 min, versus the 3-minute target.

**Recommendation.** Before the live T04 run:
1. Fix F1 (small) and F6 (small).
2. Gate F2/F3 with the offline intrusion benchmark.

Otherwise the run will mostly measure these known defects.

## 5. Artefacts

- **Scripts:** `scripts/k2_common.py` (snapshot assert, restore), `k2_scene_audit.py` (`_prepare` replica and audit), `k2_phantom_trace.py`, `k2_region_probe.py`, `k2_overlay_zoom.py`, `k2_object_strength.py`, `k2_can_ownership.py`, `k2_surface_all.py`, `k2_summary_size.py`, `k2_intrusion_test.py`, `k2_grasp_selfblock.py`, `k2_payload_test.py`, `k2_payload_detail.py`, `k2_payload_selfblock_all.py`, `k2_da3_table_bias.py`, `k2_h3_pose_strength.py`, `k2_h3_scale_drift.py`, `k2_link0_points.py`, `truth_boxes.py`.
- **Replica check:** the replica asserts equality with the real `_prepare` (72,738 removed and 44,817 strong at frame 14).
- **Data:** `k2_*.json`, `k2_scene_audit_all.log`, `truth_boxes.json`, `k2_bootstrap_seeds_libero10.json`, `k2_strong_composition_step13.json`.
- **Images:** `external_zoom_self_filter_step13.png`, `external_full_self_filter_step13.png`, `external_selfremoved_red_strong_green_frame14.png`, `wrist_selfremoved_red_strong_green_frame14.png`, `external_kept_above_z060_frame14.png`.
- **Can ownership check:** a separate cylinder test (`k2_can_ownership.json`) found 0 can points owned by any target. An earlier box-based count of "can points owned by white_mug" was an artefact of overlapping AABBs.

**Limits.**
- run 001 never moved the robot, so all H2 evidence is at the initial pose. Other poses come from H3's 2,048-point subsamples.
- Saved depth and object points are float32 (ownership re-associated within 6.6e-8 m).
- Live post-close behaviour is inferred from code and saved frames, not observed.
