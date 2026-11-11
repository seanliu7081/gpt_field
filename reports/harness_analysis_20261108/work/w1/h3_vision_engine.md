# H3 `foundation_vision` perception engine: analysis (key `h3_vision_engine`)

Date: 2026-11-07. Read-only analysis. No simulator, Astra, or GPU work was run. Offline numpy/scipy only ran on saved artifacts (CPU), using `/workspace/gpt_field/.venv-foundation-vision/bin/python`. Reproduction scripts are in
`/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/scripts/`.
Mask overlays (`t04_frame0_sam_overlay.png`, `t04_f201.png`, `t08_f0.png`, `t08_f24.png`, `t08_f243.png`, `t08_f328.png`) are in the same `w1/` folder.

Evaluation-only truth was used **only to score perception**, never as a policy input. It came from: the table height and can collision bounds in `runs/foundation_vision_t04_can_obstacle_20261105_001/inputs/layout_manifest.json` (table top z = 0.43676 m; can x∈[-0.0545, 0.0077], y∈[-0.2454, -0.1837], z∈[0.437, 0.513]); object positions in `inputs/initial_state.npy`.

## 1. How the engine works (as implemented)

1. **Bridge** (`bridge.py:53-94`). Public StepView upright RGB (external agentview plus wrist) and calibration are converted to OpenCV `world_to_camera`. The wrist pose is cross-checked against state8.
2. **SAM 3.1 detector, per frame** (`backends.py:205-261`). Each `TargetSpec.text` is run as an independent text prompt on each view. Every mask with score > 0.5 is kept (`backends.py:139,258-260`). There is no video tracking (`backends.py:5,176`).
   - In live mode the targets are the reviewed plan's `parts[].text` verbatim, plus `--vision-query` strings (`live.py:74-82,182-184`).
3. **Selection** (`engine.py:118-155`).
   - Masks under 8 px are dropped.
   - If any view has more than 1 candidate, the target is `identity_ambiguous` and all of its masks are dropped (`:144-153`).
   - `visible` requires exactly one mask in **every** camera.
4. **DA3-LARGE-1.1 two-view depth** (`backends.py:470-521`). It is pose-conditioned on the known K/E. For exactly 2 views, metric scale = known baseline / predicted baseline (`backends.py:315-354,389-395`). Upstream Umeyama Sim3 is rank-deficient for 2 views, which is why `da3_real_001` failed.
5. **Geometry** (`geometry.py:435-476`).
   - Every valid pixel of both views is back-projected.
   - Each point is projected into the other view and labelled surface, occluded or contradicted, using a ±(2 cm + 2 %·depth) band (`:121-133,464-474`).
6. **Target 3D** (`geometry.py:199-266`).
   - Mask points of view A that are depth-consistent in view B must land inside B's mask for ≥ 90 % of them, in **both directions**. Otherwise the target is `rejected_mixed_instance` (`:213-235,240-241`).
   - Only points with support ≥ 2 views and zero scene contradictions survive (`:236-238`).
   - Centroid and bounds are the mean and min/max of the surviving points (`:252-253`).
7. **Identity** (`engine.py:200-249`).
   - A track continues only if the frame gap ≤ `max_age_s`, the appearance histogram is close in ≥ 2 views, the centroid moved ≤ 0.10 m, and mask IoU > 0.05 in a stationary camera.
   - Attachment, slip and release are always `unknown` (`:268-269`).
8. **Queries** (`geometry.py:330-432`).
   - `query`: `surface` if inside any view's depth band; `estimated_free` only if ≥ 2 independent views see the point in front of cross-view-supported, uncontradicted depth; otherwise `unknown`.
   - `assess_path` blocks on **all** retained points, including robot pixels.
   - Neither is called anywhere in the live loop. A grep of `src/` finds no caller outside `geometry.py`/`engine.py`.
9. **Live adapter** (`live.py:238-299`). Vision runs synchronously inside every executive request. A bounded summary (objects, per-view candidate boxes, 64 sampled scene points; `live.py:28,142-143`) is inserted as `context.foundation_vision`. There is no automatic obstacle use and no action gate.

## 2. Headline numbers

| Metric | T04 can | T08 bottle | Source |
|---|---|---|---|
| Target-frames (targets × executive frames) | 77 (7×11) | 66 (6×11) | `trial/foundation_vision/frame_*_snapshot.json` |
| Usable 3D target centroids | **0** | **0** | same; `analysis.json` |
| No candidate in either view | 34 | 42 | recomputed (matches `analysis.json`) |
| Candidate in one view only | 26 | 17 | recomputed |
| >1 candidate in a view (ambiguous) | 11 | 3 | recomputed |
| Cross-view association rejected | 6 | 4 | recomputed |
| Target-frames with a **unique external** detection | 32 | 20 | recomputed |
| Target-frames with **any** wrist detection | 11 | 6 | recomputed |
| Target-frames unique in both views (max that could ever pass) | 6 (all rejected) | 4 (all rejected) | recomputed |
| Per-frame total vision time, median (first frame) | 1.159 s (2.837 s) | 1.066 s (2.310 s) | snapshot `timing_s` |
| SAM / DA3 / geometry medians | 0.746 / 0.228 / 0.188 s | 0.687 / 0.216 / 0.165 s | snapshot `timing_s` |
| Vision time per episode | 14.64 s | 13.11 s | sum over 11 frames |
| `context.foundation_vision` size per request | 7,666-7,921 chars | ~7.6-8.1 kB | `trial/requests/*.request.json` |
| Input-token growth per call, H3 vs matched H1 transit100 | 9.5k vs 6.6k | 9.0k vs 6.2k | `trial/requests/*.response.json` usage |

## 3. Findings

### F1 (critical). Plan part-descriptions are used as SAM prompts, and there is no instance inventory or role binding. This is the dominant cause of 0 usable centres.

**Mechanism.**
- `targets_from_plan` passes `part["text"]` verbatim (`live.py:79`), e.g. "white mug: loop handle on its side", "left plate: round top with striped rim", "stove: rectangular top surface around circular burner".
- Each prompt is grounded independently per view (`backends.py:239-260`).
- SAM 3.1 detects noun-phrase concepts. It cannot resolve "left/right/near", and the colon-qualified part phrases rarely fire.
- When a category legitimately has 2 instances (two plates), the engine drops the target as ambiguous (`engine.py:144-153`).
- Nothing prevents one mask from being claimed by several targets.

**Evidence.**

| Observation | T04 | T08 |
|---|---|---|
| Targets with 0 candidates in all 22 view-frames | A "white mug: textured cylindrical body", B "white mug: loop handle…", E "yellow and white mug: loop handle…" | A "near moka pot: whole faceted body…", B "near moka pot: curved side handle…", E "stove: rectangular top surface…" |
| These objects visible in RGB? | Yes. The white mug is large in both views at frame 0 (`t04_frame0_sam_overlay.png`). | Yes. The stove and silver pot are visible in both views (`t08_f0.png`, `t08_f24.png`). |
| Simple-noun extra query, external unique detections | "can" 11/11 (score 0.82-0.86) | "bottle" 10/11 (0.71-0.79) |
| Same mask claimed by 2 prompts (IoU > 0.8) | C/F (left/right plate) 11/11 external, 5/5 wrist | C/D external 6; D/bottle external 3, wrist 1 |
| Relational prompt grounded on wrong instance | wrist "right plate" = the left plate under the gripper (f201 overlay); 0 % cross-view agreement ×5 | wrist "right moka pot: curved side handle…" = **the bottle** (0.935 f0, 0.927 f24; overlays) |

- 66 of the 76 "no candidate" target-frames come from these 6 never-detected targets.
- All 14 ambiguity cases and all 7 wrong-instance rejections come from relational or compound prompts.
- Together that is 97/143 (68 %) of the target-frame failures.

**Improvement (general).**
- Split each plan part into a short category noun phrase ("mug", "plate", "moka pot", "stove", "can", "bottle"), obtained by stripping the qualifier after ":" and spatial words. This is a text rule, not per-task text.
- Run one prompt per **category**, de-duplicate masks across prompts, and keep **all** instances as numbered candidates instead of vetoing on more than 1.
- Bind task roles to instances once:
  - either deterministically in world/viewer frame ("left/right" = viewer frame; uc7.1 already has direction semantics),
  - or with one set-of-marks choice by Astra on the external image at plan time.
- Then keep the binding by tracking (F8).
- Keep parts (handles) as sub-masks searched *inside* the bound instance mask rather than as global prompts.

**Expected effect.** Whole-object detection should approach the simple-noun rate already observed (21/22 external frames). That removes the 97/143 prompt-caused failures. Prompts also fall from 6-7 to about 3 per scene, which feeds F9. Confidence: likely; SAM was not re-run because GPUs are busy.

**Validation.** Offline SAM replay on the 22 saved RGB pairs embedded in `trial/requests/*.request.json` (last user turn, two `input_image`s). Count unique correct-instance detections per object, then run an all-task frozen-build sweep.

### F2 (high). 3D target geometry requires a unique mask in **both** views, so the narrow, moving wrist camera is a single point of failure. Single-view evidence is discarded.

**Mechanism.**
- `min_crossview_support = 2` (`geometry.py:34`).
- Support only increments from the other view's mask (`geometry.py:213-222`).
- `visible` requires every camera (`engine.py:150-151`), and `observable` requires `visible` (`engine.py:200`).

**Evidence.**
- The wrist produced a detection in only 11/77 and 6/66 target-frames: 6 and 5 distinct wrist instances over 11 frames each.
- So at most 6 + 4 target-frames could ever qualify. All 10 were rejected (F3).
- Meanwhile 32/77 and 20/66 target-frames had a unique external detection and got nothing (26 + 17 "single view only").
- The discarded external evidence is good. Ray-cast truth on the saved T04 frame (`da3_raycast.py`) scores the external DA3 depth on the 3,552 can-mask pixels that hit the true can:
  - median error +1.3 mm,
  - |err| p50 4.3 mm, p90 10.7 mm,
  - 100 % within the engine's own tolerance.
- The external SAM can mask has IoU 0.907 with the true silhouette.

**Improvement.**
- Emit single-view target hypotheses with an explicit along-ray depth band, labelled `single_view_depth_hypothesis`, instead of `unknown`.
- Upgrade to multi-view when the wrist (or several wrist poses with known state8 poses) confirms.
- Add triangulation across ≥ 2 wrist frames during approach.

**Expected effect.** 52/143 target-frames (36 %) would have yielded a centroid with today's prompts, and most visible objects would with F1. Expected error is about 1 cm for compact objects at ~0.8 m external range (one sample; must be confirmed, F10). Confidence: verified counts, likely accuracy.

### F3 (high). The bidirectional ≥ 0.9 mask-agreement rule is unattainable for partial or border-truncated wrist masks. It rejected the same physical object 3 times.

**Mechanism.** `pair_rejected = comparable > 0 and ratio < 0.9` is applied in **each** direction (`geometry.py:36,224-229,240-241`). When the wrist sees only part of an object, external points that are visible to the wrist but outside SAM's partial wrist mask count as disagreement.

**Evidence: the 10 both-view cases.**

| Case | ext→wrist agreement (comparable) | wrist→ext agreement (comparable) | Same instance? |
|---|---|---|---|
| T04 f0 can | 0.638 (2460) | **0.913** (2946) | yes |
| T08 f328 bottle | 0.832 (357) | **0.847** (1376) | yes (overlay) |
| T08 f243 bottle | 0.042 (48) | 1.0 (1) | yes, tiny overlap |
| T04 F ×5 ("right plate") | 0 comparable | **0.0** (34-8209) | no: wrist mask is the left plate |
| T08 D ×2 | 0.0 / 0 | **0.0** (2270, 2962) | no: wrist mask is the bottle |

Offline reconstruction of the can case (`can_assoc.py`, same saved frame):
- Calibration is correct: the true can centre projects at (87.9, 316.8), inside the external SAM mask (centroid 86.3, 316.0).
- The wrist SAM mask (score 0.509) covers only **53.2 %** of the true can silhouette (IoU 0.52, precision 0.96). The can is cut by the wrist image border (true silhouette x up to 511).
- Even with **ray-cast true geometry**, only **73.8 %** of external can points visible from the wrist fall in the wrist mask. 0.9 is therefore impossible.
- The non-matching points lie a median 10.3 px (p90 26.3 px) from the mask. An 8 px dilation only reaches 0.771.
- The existing unit tests use full-image masks only (`tests/test_foundation_vision_geometry.py:50-51`), so this case was never exercised.

**Improvement.**
- Replace the symmetric veto with a containment test on the **smaller** mask (max over directions ≥ ~0.8) plus a minimum comparable count (e.g. ≥ 100 points).
- Optionally add projected-mask IoU as a soft score.
- Carry the score instead of vetoing.

**Expected effect.** On this data, same-instance cases sit at 0.85-0.91 and different-instance cases at 0.0, so the separation is clean. The can and the f328 bottle flip to accepted; f243 (1 point) falls back to single view (F2); the 7 wrong-instance cases stay rejected. Confidence: verified.

### F4 (high). DA3 two-view baseline-ratio scale gives surface-dependent metric bias. Background planes are 3-6 cm too deep while objects are within ~1 cm. Most "contradictions" are background.

**Mechanism.**
- One scalar per frame, `known_baseline / predicted_baseline` (`backends.py:333`), scales both views.
- The predicted relative rotation is off by 0.043 rad and is ignored (`:334-343`).
- Per-view and per-surface errors are not corrected (`"depth_errors_from_prediction_are_not_corrected": true`).
- Target eligibility also needs zero *scene-level* contradictions (`geometry.py:236-238`), so background bias leaks into target acceptance. At T04 f0, 409 can points were contradicted.

**Evidence.** Offline on `runs/foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz` (`da3_metric.py`, `da3_raycast.py`). The report numbers reproduce exactly: support 12,363 / 29,555; contradicted 91,607 / 54,919.

| Per-view scale (ext, wrist) | Table depth rel. error (ext / wrist) | Can depth error median (ext / wrist) | Cross-view supported px (ext / wrist) | External contradicted |
|---|---|---|---|---|
| shipped 0.878 / 0.878 | **+3.5 % / +18.1 %** (≈ +3.0 / +6.3 cm) | +1.3 mm / +8.0 mm | 4.75 % / 11.3 % | 35.2 % |
| self-calibrated by max agreement, 0.84 / 0.72 | -1.1 % / -3.1 % | -3.2 cm / -4.6 cm | **37.1 % / 68.7 %** | 3.2 % |
| table-calibrated (truth), 0.848 / 0.744 | ~0 % / ~0 % | -2.5 cm / -3.8 cm | 36.9 % / 65.4 % | 3.2 % |

How to read this:
- Over the pixels actually comparable at the shipped scale, external→wrist is 10.3 % agree / **76.1 % contradicted** / 13.6 % occluded; wrist→external is 13.9 % / 25.8 % / **60.3 %**. This is the signature of a wrist background that is too deep.
- A per-view self-calibration that maximises cross-view agreement recovers the true table plane **without any truth**, to within 1-3 %. That is a usable zero-shot mechanism.
- But it moves the can 3-5 cm. A single scale per view cannot fit both object and background: DA3 exaggerates depth relief, worst in the near-field wrist view.
- Live per-frame agreement varied by two orders of magnitude: external 0.1-26.2 % (T04) and 0.3-22.9 % (T08); wrist 1.4-82.6 % and 3.2-62.7 %. That indicates an unstable per-frame scale.

**Improvement.**
1. Feed **≥ 3 views** to DA3 (external + current wrist + a previous wrist frame at a different known pose). This makes upstream Sim3 non-degenerate and avoids the baseline-ratio shortcut.
2. Fit a per-view **affine (scale + shift)** depth correction from metric anchors available from calibration only:
   - robot self-pixels at known depth (hand/fingers from state8 and the fixed mount; arm links if joint state is available),
   - the support plane shared by both views (agreement maximisation, as above).
3. Carry per-view, per-depth uncertainty instead of a fixed 2 cm + 2 % band.

**Expected effect.** Background bias goes from 3.5 % / 18 % to ≤ ~3 %, and cross-view support from ~5-11 % to ~37-69 %, which makes free-space evidence possible (F6). Object-level accuracy under the affine model is a hypothesis to validate (F10).

**Cross-harness note.** H2 reuses these backends and `build_geometry` (`foundation_harness/perception.py:19-24,202-212`).

### F5 (high). Robot self-surfaces and the held object stay in the obstacle cloud. Any path that starts at the gripper is "blocked".

**Mechanism.**
- All valid pixels become surface hypotheses (`geometry.py:140-168`).
- `assess_path` blocks on all retained points of all views (`:393-410`), and probe points inside any depth band are `surface` (`:361-362,427-428`).
- Robot removal is listed only as a limitation (`engine.py:306`).

**Evidence** (`query_eval.py`, `selfblock.py`, saved T04 frame):
- A 10 cm straight descent from the grip site is `blocked`. All 4 blocking probes project to wrist rows 450-493 at 0.10-0.12 m depth, i.e. the fingers and hand.
- 10,332 retained hypotheses lie within 5 cm of the grip site, and 39,491 within 10 cm.
- Wrist rows ≳ 430 (≈ 16 % of the raster, plus finger tips) are robot body in every inspected wrist frame.
- In T08 f243 the held moka pot fills most of the wrist image.
- H2 independently failed 2/8 operations on "observed robot surface at link0". Plane residuals were 3.9-5.5 mm: outside its 3 mm self-removal but inside its 6 mm dilation (`runs/foundation_harness_t04_can_obstacle_001/analysis/geometry_explanation.json`).

**Improvement.**
- Render or project known robot geometry (hand + fingers from state8; full arm from joint state with H2's `robot_geometry.py` FK) into each view.
- Label those pixels `robot`: not obstacle and not free evidence.
- Use a removal tolerance tied to the per-view depth uncertainty from F4.
- Apply the same label to the bound held-object mask once it co-moves with the hand (F8).

**Expected effect.** No self-blocking: 4/4 blocking probes in the test path were robot pixels. Fewer spurious contradictions. This is a prerequisite for any vision-based clearance.

### F6 (high). Query semantics give no usable free space, the evidence is not fused over time, and the live loop never queries it.

**Evidence** (shipped scale, `query_eval.py`, `frustum.py`).

| Probe set (truth label) | surface | unknown | estimated_free |
|---|---|---|---|
| inside true can (occupied) | 79.3 % | 20.7 % | 0 % |
| inside table slab (occupied) | 58.8 % | 41.2 % | 0 % |
| free air 2-4 cm above table | 16.3 % | 83.7 % | **0 %** |
| free air z 0.56-0.64 | 0 % | 100 % | **0 %** |

- The "surface" shell is ±(2 cm + 2 %·depth): ±3.3-3.7 cm at external range.
- Even after F4 self-calibration, only about 6 % of free probes become `estimated_free`.
- Structurally, at the initial pose only 23 % of the workspace volume (z 0.44-0.75) and 19 % of the z 0.56-0.64 band lies in **both** camera frusta. The rest can never be free under the 2-view rule, because a static single view never certifies free space.
- The executive receives 64 of ~521k points (≈ 1 per 8,100 px; `live.py:28,142-143`).
- `query` / `query_path` have no caller in the live loop.

**Improvement.**
- Accumulate a persistent world-frame occupancy/TSDF. The external camera is static, so its evidence fuses trivially; each wrist pose adds coverage. Apply self-masking (F5) and calibrated uncertainty (F4).
- Extract object-level primitives (box or cylinder per bound instance, F1/F7).
- Hand those to the **existing** transit router as avoid boxes, replacing Astra-authored boxes. This is a new obstacle source, not a new veto, consistent with the user's "prefer removing vetoes" rule.

**Expected effect.** Removes the need for Astra to hand-write obstacle boxes. In T04, 2 of 11 accepted responses wrote one non-empty map, and 1 response was spent on a self-inflicted "start blocked near the expanded white-mug box". Gives the router geometry for added obstacles in every task. Effect on forbidden contacts is a hypothesis to measure in a sweep.

### F7 (medium). Target bounds are min/max of back-projected mask pixels, so boundary misregistration inflates extents 1.5-2.8×.

**Mechanism.** `bounds_world_m` = `points.min/max` (`geometry.py:252-253`). Mask-edge pixels take the background depth.

**Evidence** (`extent.py`; true can 0.062 × 0.062 × 0.076 m).

| View | Raw min/max extent | p2-p98 extent | 2-px eroded mask extent |
|---|---|---|---|
| External | **0.175** × 0.108 × 0.088 | 0.064 × 0.067 × 0.078 | 0.060 × 0.073 × 0.078 |
| Wrist | 0.059 × 0.103 × **0.135** | 0.052 × 0.056 × 0.098 | 0.051 × 0.062 × 0.083 |

DA3 has no flying pixels here (0.0 % of mask pixels exceed a 5×5-median deviation of 3 cm), so the cause is mask/depth boundary misregistration. The same mechanism is visible in H2: the yellow_white_mug "visible extent" was 0.268 × 0.230 × 0.168 m, against a 0.13 × 0.10 m visible plate, behind 5/8 `placement_candidates_zero` rejections.

**Improvement.** Erode masks by 1-2 px before back-projection, use robust percentiles, or fit primitives with RANSAC.

**Expected effect.** Extent error drops from ≤ 2.8× to ≈ 1.0-1.1×. That prevents false blocking and false "does not fit" outcomes.

### F8 (medium). No tracking: IDs cannot persist under the live cadence, and held objects are unhandled.

**Mechanism.**
- Detection runs per frame only (`backends.py:5`).
- Continuity needs a frame gap ≤ `max_age_s`, appearance in ≥ 2 views, displacement ≤ 0.10 m, and a stationary-view IoU (`engine.py:205-228`).
- Snapshots exist only at executive requests (`live.py:258`).
- Interaction is always `unknown`.

**Evidence.**
- Inter-snapshot gaps were 24.0-66.3 s (T04) and 24.8-85.8 s (T08), covering 1-109 actions, against `max_age_s` = 5 s.
- Every track would expire even if it were observable. Any transported object (moves > 0.10 m) would get a new ID.
- No `track_id` was ever assigned in either run.
- The engine default `max_age_s` = 1.0 s (`engine.py:23`) is below every observed processing age (1.12-2.94 s). The wrapper overrides it to 5 s (`scripts/evaluate_foundation_vision_task.py:45`).
- Freshness is wall-clock at payload build, not at action time (13-79 s later).

**Improvement.**
- Bind instances once (F1), then propagate masks with SAM 3.1's video/multiplex tracker on the static external stream (async, a few Hz).
- Associate the wrist by robot motion compensation (known wrist pose).
- Replace wall-clock staleness with event staleness: robot motion near the object, grasp/release, contact.
- Mark `held` when the mask co-moves with the hand over k frames, and treat it as attached robot geometry (F5).

**Expected effect.** Persistent IDs across the episode, held-object state, and incremental updates. Confidence: hypothesis; tracker latency not measured.

### F9 (medium, FAST). Vision costs about 1.1 s per call and about 2.9k input tokens per call, for zero usable output.

**Evidence.**
- Vision runs synchronously in `build_payload` (`live.py:281`): median 1.16 / 1.07 s per call. First call is 2.84 / 2.31 s because of SAM warm-up (1.93 / 1.46 s).
- Totals are 14.6 / 13.1 s per episode. SAM is 64 % of each frame.
- Estimated SAM cost is ≈ 0.17 s image encoding per view plus ≈ 0.03 s per prompt per view. That is inferred from the 7- vs 6-prompt medians, so it is approximate.
- Model preload is 24.6 s per process (SAM 19.7 s + DA3 4.9 s; `runner.log`); SAM peak GPU memory is 6.0 GB.
- The 7.7-7.9 kB summary (all objects `not_visible` / `unknown`) is retained in the conversation history:
  - user-turn text is ~29k vs ~21k chars in H1;
  - input tokens grow 9.5k/call vs 6.6k/call (T04) and 9.0k vs 6.2k (T08);
  - at call 11 that is 106,092 vs 75,454 tokens (+41 %).
- Running vision every 10 actions synchronously would cost ≈ 60 frames × 1.1 s ≈ 66 s per 600-action episode, about 37 % of a 180 s budget. It has to be asynchronous.

**Improvement.**
- Deduplicate prompts by category (F1, about 3 prompts → SAM ≈ 0.5 s).
- Warm up at preload, and run inference asynchronously (overlap with motion settle or the previous model call).
- Keep models resident across tasks in a sweep.
- Send only usable objects and deltas, and drop the 64 scene points.
- Strip superseded vision summaries from history.

**Expected effect.**
- Saves ≈ 2.5-2.9k input tokens per call, ≈ 30k by call 11.
- Removes ≈ 1.1 s per call from the critical path and 1.6 s from the first call.
- Removes the 24.6 s preload per task in a resident server.

The latency gain from fewer tokens is likely but was not measured directly.

### F10 (medium). Perception was never validated against truth. A truth-scored offline replay benchmark is cheap and already feasible.

**Evidence.**
- `validation_report.json`: `geometry_accuracy_validated: false`, `joint_models_validated: false`. All 82-105 tests use synthetic constant depth and full-image masks.
- The offline scripts here reproduced the report numbers exactly and produced metric errors in seconds of CPU time from saved artifacts plus evaluation-only truth: table height, object bounds, and initial_state.

**Improvement.** Build a frozen perception benchmark from saved runs:
- RGB in request JSONs, calibration, and saved masks;
- re-run SAM/DA3 offline when a GPU is free;
- score detection per bound instance, centroid and extent error, association decisions, table bias, and free-space precision/recall.

Gate perception changes on that benchmark before spending live episodes.

**Expected effect.** Turns perception iteration from ~10 min plus 12 Astra calls per live episode into minutes of offline GPU time. This directly serves "few attempts" and avoids tuning to one run.

## 4. Older defects, re-checked for H3

| Defect | Status in H3 vision path | Evidence |
|---|---|---|
| Policy images L-R mirrored | **Not present.** The upright raster and calibration are consistent. | True can centre projects inside the external SAM mask (87.9, 316.8 vs centroid 86.3, 316.0). Wrist true silhouette bbox x 423-511 vs SAM bbox 422-508. `bridge.py:76-89,106-107`. |
| `astra.py _canonical` sort_keys | Not relevant to the engine. `live.py:61` sorts keys only for logging and hashing. | Wire order is owned by the parent client and outside this scope. |
| Zero rotation keeps a stale OSC goal / axis-angle sign jumps | Not applicable to perception. state8 axis-angle → matrix is sign-invariant, and wrist calibration is cross-checked against state8. | `bridge.py:69-72,91`. Control is uc7.1's. |

## 5. Suggested order of work

Every step is general (no per-task code) and is validated on the F10 benchmark before an all-task sweep.

1. F1 category prompts + instance inventory + role binding.
2. F3 containment association and F2 single-view hypotheses. These remove vetoes and carry uncertainty.
3. F5 robot/held self-masking and F7 robust extents.
4. F4 ≥ 3-view DA3 + per-view affine calibration from robot self-pixels and the shared support plane.
5. F6 fused occupancy → instance primitives → existing router avoid boxes (replacing Astra-written boxes).
6. F8 tracking and event staleness, and F9 async + compact payload.

## 6. Scripts

All scripts are in `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/scripts/`:

- `snap_stats.py`: per-frame rejection causes and timings.
- `cross_prompt.py`: duplicate masks across prompts.
- `da3_metric.py`: reproduction, table bias, scale sweep.
- `da3_raycast.py`: ray-cast truth depth errors.
- `can_assoc.py`: association diagnosis, SAM coverage, true-geometry bound.
- `query_eval.py`, `selfblock.py`, `frustum.py`: query semantics, self-blocking, coverage.
- `extent.py`: extent inflation.
- `overlay.py`, `overlay_req.py`: mask overlays from request images.
