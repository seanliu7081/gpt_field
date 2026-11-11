# H2 foundation_harness — perception and geometry layer (key `h2_perception_geometry`)

Analyst scope: `src/gpt_field/foundation_harness/{perception.py, robot_geometry.py, types.py}`, the H3
`foundation_vision` pieces H2 imports (`backends.py` SAM 3.1 / DA3, `geometry.py`, `bridge.py`), and the
geometry consumers in `motion.py` (self-filter, free space, grasp/placement generation) as far as they
interpret perception output. Evidence: the single H2 T04 run
`runs/foundation_harness_t04_can_obstacle_001` (user-stopped, 19 actions, 0 keyposes, 12 calls), plus the two
H3 runs for recurrence. Read-only; no simulator, model or GPU job was started.

## 0. Which code version is analysed (important: the tree changed during this analysis)

| Module | At T04 run (`source_at_launch`) | Current tree (snapshot 2026-11-08 00:02 UTC) |
|---|---|---|
| `perception.py` (642 lines) | v1 | **identical** (sha256 8dc5f72c…) |
| `foundation_vision/{geometry,backends,bridge,engine,types}.py`, `robodojo_direct_sim.py` | v1 | identical |
| `robot_geometry.py` | collision hulls only | + `include_visual` body envelopes + `ray_entry_distances` (in-progress) |
| `motion.py` | local grasp + placement generation, 3 mm self-removal, `allow_unknown_space=False` | **"foundation-harness-keypose-2"** (in-progress, unvalidated): Astra keyposes, silhouette ray-occlusion self-mask, link0 skipped, strong/uncertain obstacles, `allow_unknown_space=True`; `_grasps`/`_placements` deleted |
| `surface.py` | absent | new: density + normal-consistency + dominant-component filter |
| `client.py`, `executor.py`, `types.py` | semantic-1 | keypose-2 (edited 23:37–23:58 UTC Nov 7, while this analysis ran) |

Snapshot with hashes: `SCRATCH/h2pg/current_src_snapshot/`. Every finding below states whether it holds for
(a) the run-time build and (b) the in-progress keypose-2 tree. The keypose-2 tree has never run a task;
`output/foundation_harness_keypose_v2_preparation_001/implementation.json` says
`implementation_complete_awaiting_full_task_approval`.

Method: offline replays of the saved public inputs (`trial/foundation_harness/vision/*.npz`: RGB,
calibration, float32 DA3 depth, SAM masks, object points) through the frozen run code and through a private
copy of the in-progress tree (`SCRATCH/h2pg/pkg_cur`). The run's own `physical_truth.json` (table top
z = 0.4368 m; mug heights) is used **only for diagnosis**, never as a proposed policy input. Scripts and
JSON results: `SCRATCH/h2pg/*.py`, `SCRATCH/h2pg/*_results.json`.

## 1. How the layer works as implemented

1. `bridge.frame_from_step_view` builds a `VisionFrame` from the two public 512×512 RGB views and calibration.
   I verified the rasters are geometrically consistent with calibration: the Panda collision hulls,
   projected with the measured joints, overlay the robot in both images (`SCRATCH/h2pg/overlay_*.png`), so the
   old "mirrored policy image" defect does not affect H2 perception.
2. `Sam31Segmenter.segment` runs every text query on both views and keeps masks with score > 0.5
   (`backends.py:139,260`); sub-threshold candidates are discarded.
3. `DA3DepthEstimator.estimate` runs DA3-LARGE-1.1 pose-conditioned on both views, rescales **both** depth maps
   by one factor `known_baseline / predicted_baseline` (`backends.py:315-353`, applied at 344), and bilinearly
   resamples to the original rays (`backends.py:264-313`). The DA3 confidence map is resampled but used only
   for reporting (`geometry.py:294`).
4. `build_geometry` back-projects every valid pixel (≈260 k per view) and cross-checks each view against the
   other with tolerance 2 cm + 2 %·z (`geometry.py:32-33,97-133,464-474`): on the other view's surface → support;
   in front of it → contradiction; behind it → "occluded" (kept; `geometry.py:130` "a point behind a visible
   surface is occluded, not contradicted").
5. `HarnessPerception._select` binds one candidate per (target, view): an Astra-selected ID, the **only**
   candidate of that query (`perception.py:353-355`, auto-binding), or appearance/IoU re-association.
6. `_object` keeps uncontradicted points that do not conflict with the other view's mask (`perception.py:445-463`),
   labels them `crossview_supported` or `single_view_clean_patch` ("graded" support), and sets
   `status` supported/partial/conflict/ambiguous/missing (`:491-507`). Normals are camera-facing central
   differences, zeroed at depth edges and mask borders (`:48-77, 509-521`). Objects are visible shells; nothing
   is completed (`:4-6`).
7. `_assemble` returns **all valid points of both views** as the scene obstacle cloud, explicitly including
   conflicting depth and robot surfaces (`perception.py:566-571, 597-599`).
8. Run-time `motion.py` (path `runs/foundation_harness_t04_can_obstacle_001/source_at_launch/src/gpt_field/foundation_harness/motion.py`):
   removes scene points within **3 mm** of the current robot collision hulls (`:114-124`), rejects any
   configuration with a scene point within **6 mm** (`:24, 186-202`), requires `estimated_free` from ≥ 2
   independent views on cross-view-supported depth for every swept hull probe (`:203-221`,
   `geometry.py:330-374`), raises immediately on unknown space without RRT (`:277-278`), generates grasps only
   from observed opposing-normal pairs (`:308-358`) and placements only where a 3×3 probe grid over the payload's
   **whole visible XY bounding box** lies within 12 mm of upward-normal support (`:360-419`).

## 2. The T04 blocker chain (each blocker hid the next)

| # | Blocker | Where | Evidence (verified offline) | Fixing only this exposes |
|---|---|---|---|---|
| B0 | `no_supported_payload_footprint` (5 of 8 rejections) | `motion.py:383-394` (run) | 0 of 96 centre×yaw samples pass in every pick-place; payload 26.25×23.35 cm vs visible plate 12.5×9.3 cm | B1 (start pose) |
| B1 | start configuration collides with own `link0` | `motion.py:119-121` (3 mm) vs `:24` (6 mm) | 74–83 unowned points in the 3–6 mm shell of link0 (+7–17 at link6) | B2 |
| B2 | **`unknown_swept_space` on every move** (not reported by `analysis.json`) | `motion.py:203-221, 277-278`; `geometry.py:347-368` | with a consistent 6.5 mm or 2 cm self-margin, or with image-silhouette masking, all 5 test moves (Astra's two viewpoint moves, ±5 cm z, +5 cm x) fail with 32–88 of 160 probes unknown | B3 |
| B3 | robot "ghost" depth in the descent path | DA3 robot depth error (F4) | with `allow_unknown_space=True` + 3D margins, a 5 cm descent still fails (direct edge: link5 vs an unowned ghost point 6.9 cm behind/6.7 cm above the hand; RRT exhausts 12 s); with image-silhouette masking all 5 moves plan | B4 |
| B4 | grasp generation | `motion.py:308-358` | handle: 0 opposing pairs in every variant; whole-mug "grasps" disappear (16 → 0) once depth is made cross-view consistent (F7) | — |

Masking order matters for "few attempts": Astra spent 12 calls (125 s) trying variants of operations whose
failure was fixed geometry it could not influence; the executor reported only `no_complete_operation_path`
(already noted in `analysis.json`; keypose-2 claims structured reasons).

## 3. Findings (ranked)

### F1 — Per-view metric depth is inconsistent; one global DA3 scale cannot fix it (critical)
**Mechanism.** `_align_two_view_depth` multiplies both views by one baseline-ratio scale
(`backends.py:333,344`). The two views carry different, stable biases, so the cross-view rules in
`build_geometry` mostly measure bias, not visibility.

**Evidence (verified).** Table-plane check on all 20 fresh DA3 inferences (`SCRATCH/h2pg/depth_scale_results.json`):

| View | predicted / true depth on table | median error | true depth |
|---|---|---|---|
| wrist | 1.194 – 1.212 | +6.5 … +7.1 cm | 0.334 m |
| external | 1.033 – 1.040 | +2.2 … +2.7 cm | 0.655 m |

Both exceed the 6 mm collision margin, by about 4× (external) and 11–12× (wrist). Consequences in frame 5 (`support_stats_results.json`):
external points 5.0 % cross-view supported, **35.1 % contradicted**; wrist 11.4 % supported, **49.3 %
"occluded"** (biased far, so they pass as occluded geometry). The yellow-white mug's points reach
z = 0.357 m, 8 cm below the table (true low/high z 0.4368/0.540).
Recurs across tasks and runs with the same backend (H3 `trial/foundation_vision/frame_*_snapshot.json`):
T08 frame 0 wrist 90.6 % occluded, external contradicted up to 28.6 %; H3 T04 frame 0 identical to H2
(35.2 % / 49.3 %).

**Improvement (general).** Estimate a per-frame, per-view affine depth correction (scale + shift, robust
least squares) from the region both cameras see, with the more reliable view as reference, before
`build_geometry`. Longer term: feed DA3 several wrist frames with known measured poses (known-motion
parallax) and propagate a per-pixel σ(z) instead of the fixed 2 cm + 2 % gate.

**Expected effect (offline, verified).** `SCRATCH/h2pg/align_whatif.py` / `crossview_align.py` on frames 3, 5,
10, 19: wrist table error 6.6–7.1 → 1.6 cm; external supported 5 → 39 %, contradicted 35 → 1.5 %; wrist
supported 11 → 72–73 %, occluded 49 → 5 %; mug min z 0.357 → 0.435 m. Cost: one ~87 k-pair least-squares fit
per frame (milliseconds). Raises the cross-view-supported part of keypose-2 "strong obstacles" about
6.5–8× (F8). Validation: replay
gate on saved frames (table error, contradiction %), then the all-task sweep.

### F2 — Hard, asymmetric per-point cross-view voting deletes good evidence (high)
**Mechanism.** In front of the other view → removed; behind it → kept as occlusion (`geometry.py:129-132`).
A mask disagreement in the other view deletes the point (`perception.py:454,460`). With a far-biased wrist
view this keeps the wrong wrist points and deletes the right external ones. The deletion is also driven by
identity errors.

**Evidence (verified).** The yellow handle's external mask yields 698 points, of which 694 are "contradicted"
by the wrist view and dropped, leaving 4 (frames 2–4, snapshots `crossview_pairs`). The whole yellow mug loses
2 528–2 625 external points per frame the same way. At obs 13 the wrist auto-binding chose the yellow mug for
"textured ceramic mug" (`f14:a676…`, bbox [39,210,233,348], score 0.508). The mask-conflict rule then cut the
correct external white-mug cloud from 5 691 to 1 555 points (−73 %). After a pure 1/1.2 wrist rescale,
silhouette misregistration still produced 12 481 wrist contradictions (`scale_whatif_results.json`), so
the rule is fragile even with better calibration. Every object in all 44 snapshots is `partial` (never
`supported`), so the status carries no information for Astra.

**Improvement.** Replace per-point hard votes with uncertainty-weighted fusion (a σ(z)-aware TSDF or
log-odds voxel map across views and frames). Treat behind-surface points as occluded only when they are
beyond the other ray's surface by more than its σ. Treat mask disagreement as identity ambiguity that is
reported, not deleted. Grade objects by supported fraction, not "any single-view point → partial".
**Expected effect:** no more good-evidence deletion. Object point retention, scored against saved frames, is
the validation metric. Confidence: mechanism verified; magnitude of benefit beyond F1 is likely.

### F3 — Mixed ("flying") pixels at mask/depth edges inflate payload and support geometry (high)
**Mechanism.** No edge, confidence or discontinuity filter exists before back-projection. Bilinear ray
resampling (`backends.py:264-313`) interpolates across foreground/background edges. The DA3 confidence map
is ignored (`geometry.py:294` reporting only). `_normals` only zeroes normals, it does not drop points
(`perception.py:67-73`).

**Evidence (verified, `payload_points_results.json`).** All four whole-cloud xy extremes lie ≤ 1 px from
the SAM mask edge. x-min (−0.227, external pixel (344,253)), x-max (0.035) and y-max (0.249) are mixed pixels
with depth gradients of 3.8–8.3 cm/px. All per-view extremes of the single-view patches are edge pixels
(0.95 px). External single-view patch: 202 points with 22.9 cm x-extent,
51 % within 2 px of the mask edge. Whole cloud 26.25×23.35×17.4 cm. The runtime placement height uses the
lowest visible point (`motion.py:384`), which is 8 cm low.

| Payload variant (yellow mug, frame 5) | XY extent (cm) | z range (m) |
|---|---|---|
| runtime (all points) | 26.3 × 23.4 | 0.357–0.531 |
| in-progress `surface.py` | 12.3 × 14.2 | 0.438–0.528 |
| mask-edge > 2 px & depth-gradient < 1 cm/px + 2–98 % trim | 10.5 × 9.8 | 0.449–0.508 |
| true (diagnosis) | height 10.3 cm | 0.437–0.540 |

**Improvement.** Drop pixels within 1–2 px of a mask boundary or above a depth-gradient or low-DA3-confidence
threshold *for object/support geometry*, and keep them only as uncertain obstacle evidence. Use
discontinuity-aware (nearest or min-depth) resampling at edges. Validation: extents on saved frames versus
evaluator object sizes in an offline gate.

### F4 — DA3 is grossly wrong on the robot; a 3-D hull self-filter cannot work (critical in run build; partially fixed)
**Mechanism.** Self-removal kept only points within 3 mm of the current collision hulls (`motion.py:119-121`,
run), while collision used 6 mm (`:24`). More fundamentally, DA3's robot depth is off by decimetres.
The model covers link0–7, hand, both fingers and pads (13 hulls, `robot_geometry.py:153`); visual-only geoms
and the mount are not modelled (the mount is not visible in T04).

**Evidence (verified; `ray_error_results.json`, `self_filter_results.json`).** Ray-cast against the hulls,
DA3 minus hull-entry depth, median per part:

| View / part | median signed error | n px |
|---|---|---|
| external hand / link7 / link6 | +9.3 / +10.3 / +10.2 cm (behind) | 12.6 k / 6.6 k / 2.0 k |
| external link0 / link1 / link2 | −29.9 / −25.9 / −26.8 cm (pulled toward camera) | 5.3 k / 0.3 k / 1.6 k |
| wrist hand / fingers | +8.0 / +9.4…+13.2 cm | 42.4 k / 7.4 k |

The 3 mm rule removed 98–110 of ≈29.8 k external robot-silhouette points and 0 of ≈49.9 k wrist ones. The
start pose collides at link0 (74–83 points in the 3–6 mm shell) and link6 (7–17). A 2 cm hull margin removes only
5.9–9.5 k points and still blocks descents (B3).

**In-progress status (verified on saved frames, `current_selffilter_results.json`).** Silhouette
ray-occlusion masking (`motion.py:135-195`) removes ≈72.7 k points; wrist robot pixels kept: 0–145. But
**6.9 k external robot pixels remain**, all > 6 mm in front of the hulls (the pulled-forward base). They are
retained as uncertain obstacles. link0 is skipped in obstacle checks (`:271`). Cost: `_prepare` 1.19–1.26 s
per new scene (528 k per-point ray tests).

**Improvement.** Image-space self-filter: z-buffer-render the robot model (known q + calibration) per view,
dilate 2–4 px, and **replace** robot pixels' depth with the rendered depth instead of deleting them. Rays in
front of the robot then certify free space, the region behind it is occluded or unknown, and foreground ghosts
disappear. Rasterised, this is milliseconds instead of 1.2 s. **Expected:** removes the remaining 6.9 k
ghosts. Offline, silhouette masking let 5/5 small moves plan versus 0/5 (B2/B3). Validation: robot-pixel
residual count = 0 on all saved frames.

### F5 — Free space requires two views on cross-view-supported depth: impossible with this rig (critical in run build; veto removed in keypose-2)
**Mechanism.** `estimated_free` needs ≥ 2 independent views. In each, the probe must lie in front of depth that
is itself cross-view supported (`geometry.py:347-368`). With one static and one wrist camera, and 5 %/11 %
supported pixels (F1), arm-link probes are almost never free. `_connect` gives up on unknown without RRT
(`motion.py:277-278`, run).

**Evidence (verified, `motion_unblock_results.json`, `unknown_diag_results.json`, `free_relax_results.json`).**
After fixing B1 in any way, every tested move fails with 32–82 of 160 probes unknown (88 at the worst
sampled configuration). Even after image-silhouette masking, 36/38 (up 5 cm) and 80/88 (Astra's step-9
move) unknown probes have **zero** views with single-view free evidence: they project onto the robot's own
masked image region or outside the wrist frustum.

**In-progress status.** `allow_unknown_space=True` (`motion.py:31`) removes the veto, as the user's
"remove vetoes" rule suggests. The flip side, not handled: visible obstacles are only shells
(`perception.py:4-6`), so the occluded back halves of the can or bottle are now traversable unknown.

**Improvement (general).** (i) Shadow-volume completion: space behind an observed obstacle surface along the
camera ray, down to the support plane, is presumed occupied (inflated by σ). (ii) Robot-body carving: space
the robot's measured body has swept is free. (iii) Temporal accumulation across frames. (iv) Unknown as a
path cost, not a veto. **Expected:** keeps motion feasible (as keypose-2 does) while restoring conservatism
exactly where obstacles hide volume. Validation: count of probes inside obstacle shadow volumes along executed
paths, and the all-task contact sweep. Confidence: diagnosis verified; benefit hypothesis.

### F6 — Placement support uses the payload's whole visible XY bbox on a truncated support (high in run build; check removed in keypose-2)
**Mechanism.** `_placements` centres the payload's full XY extrema, including handle and flying pixels, and
requires all 9 bbox probes within 12 mm of upward-normal support (`motion.py:383-394`, run). This conflates
the collision envelope with the contact footprint. Support came from a single external patch cut by the image
border: right plate bbox [447,312,512,382], left [0,308,58,375]. The wrist never saw a plate because no move
was ever allowed.

**Evidence (verified, `placement_variants_results.json`).** 0 placements for raw, crossview-only,
edge-filtered and trimmed payloads (best gaps 13.0 / 5.0 / 4.7 / 3.0 cm vs 1.2 cm allowed). 6 placements for
both mugs when the footprint is the **bottom 2 cm slice** (4.8×7.7 cm and 3.6×6.4 cm, gaps 2 mm and 1 mm).

**In-progress status.** Local placement generation is deleted; Astra chooses place poses with no
"support truncated / single-view" flag in the summary.

**Improvement.** If local placement checking is kept or reinstated as a non-veto ranker, it should test the
contact footprint (lowest slice) and COM projection inside the support polygon, with the remaining shape used
only for clearance. In both builds, flag supports whose mask touches the image border or that have no
cross-view support. That should trigger an active wrist look before placing (one bounded move, around
2–4 s) instead of extra model calls.

### F7 — "Observed opposing contact pair" grasp generation is structurally broken (high in run build; removed in keypose-2)
**Mechanism.** Normals are flipped toward their own camera (`perception.py:74-75`). Two cameras both above
the table rarely see truly opposing faces. Thin parts (handles) show one face at most (`motion.py:323-326, 355-357`).

**Evidence (verified, `grasp_audit_results.json`, `align_whatif_results.json`).** The handle has 0 pairs in
every variant: its 404–542 points are wrist top-surface points, 79 % with normal z > 0.85. The run's
top-ranked whole-mug "grasps" (alignment 0.82, 4 of 16) pair an external front-face point
(n = (0.93,−0.26,−0.25), z 0.448) with an upward-facing wrist point inside the mug footprint
(n = (−0.15,0,0.99), z 0.472), 3.4 cm apart diagonally through the wall. The rest pair similar cross-view
points 6.8 cm apart. After the F1 alignment makes depth
consistent, the count drops 16 → 0. The generator was driven by inconsistency, not geometry.

**Improvement.** Use the gripper's own geometry: sample closing axes, require the object's mask width
across the axis (from the top view) to be under the max opening, and require finger-volume clearance on both
sides. Alternatively, run a generic learned 6-DoF grasp proposer on the partial cloud. Use either as a
ranking aid for Astra keyposes, not a veto. Validation: grasp success on the all-task sweep, plus offline
proposal counts on saved frames of all tasks.

### F8 — keypose-2 "strong vs uncertain" obstacles make single-view obstacles advisory (high, collision goal; in-progress only)
**Mechanism.** Strong = owned coherent target surface, or support ≥ 2 without contradiction. Uncertain
intersections are counted and allowed when `allow_unknown_space=True` (`motion.py:110-130, 286-309`).

**Evidence (verified, `current_selffilter_results.json`).** Frames 5/10/19: 45–49 k strong vs ≈400 k
uncertain points. Strong share of non-robot pixels in an image box: the **can** 57–61 % (external) and 82 %
(wrist), because it lies in the overlap. The **red-mug distractor next to the yellow mug: 3.5–10 %**. The
6.9 k foreground robot ghosts are uncertain, and so are flying pixels.

**Improvement.** Fix F1 first: whole-view strong coverage rises from 5 % to 39 % (external) and 11 % to 72 %
(wrist). Then treat a single-view surface as an obstacle inflated along its viewing ray by σ(z), rather than
traversable. Keep "uncertain" for pixels that fail the edge/robot filters (F3/F4). **Expected:** collision
vetoes keep covering non-target obstacles without re-creating the B2 deadlock. Confidence: fractions
verified; collision outcome is a hypothesis until the sweep.

### F9 — Zero-shot target grounding wastes calls (medium)
**Evidence (verified).** The bootstrap grammar split (`client.py:208` run, `:274` current) turns "yellow
and white mug" into seeds "yellow" and "white mug". In frame 1, seeds were "white mug", "left plate",
"yellow" and "right plate"; both "white mug" and "yellow" were `missing`. "white mug" returned no candidate
above 0.5 in either view twice. "textured ceramic mug" returned all three mugs. Unique-candidate
auto-binding picked the wrong wrist instance (F2). **3 of 12 calls (25 %, 27.9 s of model time) were
observe-only identity or segmentation calls** (steps 0, 5, 14).

**Improvement.** (i) Chunk noun phrases, not words: never split inside a modifier list ("X and Y mug"), or
let Astra's first call author queries only. (ii) Prompt 2–4 phrasings per target in one pass: the image
encoder is cached and the cost is ≈70 ms per prompt-view (frame 1: 8 new prompt-views in 0.58 s). (iii)
Return the top-k candidates with scores even below the threshold. (iv) Accept a single-view auto-binding only
when the 3-D centroid agrees with the other view's binding. **Expected:** −2 to −3 calls (−20 to −30 s) per
task with similar clutter.

### F10 — Perception cadence: re-inference on a static scene sits on the critical path (medium)
**Evidence (verified).** Control 206.6 s = model 125.0 s (61 %) + vision 45.4 s (22 %) + planning 21.6 s
(10 %), for 0 motion. The EEF moved 0.25 mm in total (max joint drift 5.9 mrad), yet there were **20 full
refreshes (38.1 s: SAM 20.2 s, DA3 7.9 s, geometry 7.1 s)** plus 24 re-selections (5.9 s). Causes: refresh
whenever the snapshot is older than 0.5·max_age = 2.5 s (`perception.py:181`) versus about 10.4 s per model
call; a forced refresh before every call; the post-call observation. The in-progress `_prepare` adds 1.2 s
per new scene. The verbose per-call scene summary (4.6–12.6 k chars) stays in history.

**Improvement.** Key freshness on change, not wall time: skip re-inference when joint motion is below ε
and the image difference is below δ (static-world assumption). Run perception asynchronously, overlapped
with model latency. Rasterise the self-mask (F4). Trim the summary to decision fields.
**Expected:** about 30–35 s saved in this episode; 1.5–3.4 s per model call in general; 1.2 s per keypose
segment saved in keypose-2.

### F11 — No saved-observation regression gate; tests are synthetic (medium, cheap)
**Evidence (verified).** `tests/test_foundation_harness_integration.py:90-158` uses a flat 2 m synthetic
depth and stubbed `query`. Nothing exercised self-filtering, real DA3 bias, flying pixels or footprint, so
B0–B4 all shipped to a paid run. Every blocker above reproduces offline in 1–5 s per frame from saved `.npz`
files.
**Improvement.** A frozen replay suite of saved public observations across tasks, run before any launch,
checking:
- start-pose safety;
- robot-pixel residual near 0;
- small ±5 cm moves feasible;
- object extent sanity and contradiction %;
- non-zero grasp/placement candidates (where applicable).

It fits the user's one-frozen-build acceptance process. **Expected:** catches geometry deadlocks before
spending calls; zero runtime cost.

## 4. Would anything have unblocked T04?

No single fix. A first pick-place in the run-time architecture needed all of F6 (contact footprint),
F4 (image-space self-mask), F5 (unknown not vetoed) and a different grasp source than F7. F1 and F3 are
needed for any of these to be metrically trustworthy. Offline, the combination of image-silhouette masking,
`allow_unknown_space=True` and the bottom-slice footprint yields feasible viewpoint moves (5/5) and 6
placement poses per mug. Grasp generation remains the open piece, and keypose-2 hands it to Astra. The
keypose-2 tree addresses B0, B1, B2 and B4 mostly by removing local checks. B3 is mostly addressed (6.9 k
foreground ghosts remain). The perception root causes F1–F3 and F9–F10 are untouched (`perception.py`
unchanged), and F8 is a new collision-avoidance risk.

## 5. Older defects re-checked for this layer

- Mirrored images: **not present** in H2 perception (hull overlay matches both views).
- Axis-angle sign jump in state8 (steps 8→9: (−3.1403, …) → (3.1403, …)): **no effect**. The base pose stays
  (−0.51, 0, 0.42), identity rotation, because H2 uses rotation matrices (`SCRATCH/h2pg/flip_check.py`).
- `astra.py` `sort_keys` and the all-zero-rotation OSC goal: outside this layer, not checked.

## 6. Limits

- One episode; robot static for the whole run, so closer-range or moving-camera perception is untested in H2.
- Saved depth and points are float32, not a bitwise replay.
- Hulls are convex approximations, so per-part depth errors are bounds by part.
- The affine wrist alignment is a diagnostic: the fit scale ≈0.54 with offset ≈0.13 m means DA3's wrist depth
  is range-compressed, not merely scaled. It must be validated on objects and other tasks.
- The keypose-2 tree is unvalidated and was edited during this analysis.

## 7. Artifacts (all under `SCRATCH/h2pg/`)

| Script | Result file | Content |
|---|---|---|
| `common.py` | — | Restores saved frames with the frozen run code |
| `self_filter.py` | `self_filter_results.json`, `robot_silhouette_*.png` | Self-removal residuals; 3–6 mm shell; colliding shapes |
| `ray_error.py` | `ray_error_results.json` | Signed DA3 depth error per robot part |
| `overlay.py` | `overlay_external.png`, `overlay_wrist.png` | Calibration check |
| `depth_scale.py` | `depth_scale_results.json` | Per-view table-plane depth ratio, 20 inferences |
| `support_stats.py` | `support_stats_results.json` | Cross-view support, contradiction and occlusion fractions |
| `crossview_align.py`, `align_whatif.py` | `crossview_align_results.json`, `align_whatif_results.json` | Affine wrist alignment what-if |
| `scale_whatif.py` | `scale_whatif_results.json` | Pure-rescale what-if (shows the fragility) |
| `payload_points.py` | `payload_points_results.json` | Payload extremes and edge/gradient provenance |
| `placement_variants.py` | `placement_variants_results.json` | Runtime `_placements` on robust payload variants |
| `motion_unblock.py`, `unknown_diag.py`, `free_relax.py` | `motion_unblock_results.json`, `unknown_diag_results.json`, `free_relax_results.json` | The B1→B2→B3 chain |
| `grasp_audit.py` | `grasp_audit_results.json` | Provenance of opposing-pair grasps |
| `current_selffilter.py`, `time_current.py` | `current_selffilter_results.json`, `time_current_results.json` | In-progress keypose-2 self-filter, strong/uncertain split, timings |
| `flip_check.py` | — | Axis-angle sign-flip check |
| `current_src_snapshot/` | — | Hashed copy of the in-progress tree |
| `pkg_cur/` | — | Private import overlay |

`SCRATCH` = `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1`
