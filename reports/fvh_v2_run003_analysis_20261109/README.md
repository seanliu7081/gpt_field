# foundation_vision_harness_V2 run 003, V3 first look, and the next step (2026-11-10)

Scope: `runs/foundation_vision_harness_V2_t04_can_obstacle_003` (Codex's V2 revision `evidence_recovery_r3`, T04 can-obstacle scene, launched 2026-11-09 19:59:42 UTC), analysed from the saved logs, arrays and hash-pinned source only. No simulator, model or GPU was used. Section 9 adds a first look at the three `foundation_vision_harness_V3` runs that Codex launched after this run, because the next-step discussion depends on them.

How this was made: six analysts (timeline and speed, code trace, geometry against ground truth, model protocol, lineage against the plan, audit of Codex's own write-ups) worked independently from raw data; four adversarial verifiers then tried to refute their claims; three more analysts took the V3 first look. Section 11 lists the corrections the verifiers forced. All reports and scripts are under `work/`.

---

## 0. Bottom line

**Run 003 ended `blocked` at 208 s with 0 grasps, and nothing physical happened**: the arm moved 4.9 cm sideways at 25 cm height, never came closer than 27 cm to the target mug, closed the gripper 0 times, and the evaluator recorded 0 forbidden contacts. Every "block" in the run was a harness prediction.

The terminal cause was a bookkeeping bug (a wall-clock scene expiry was fingerprinted and de-duplicated as if it were a geometric blocker), but three deeper mechanisms each independently prevented progress and would have remained after that bug was fixed:

1. **A permanent slow mode.** Every one of the 51 executed steps ran under the "stale scene" rule because perception needs 4.4 to 7.4 s to publish a frame and the freshness threshold is 5 s. Commands were capped at 5 mm, the OSC realised 1.26 mm per step, and the robot moved at 1.6 mm/s while moving. One 50 mm move took 39 steps and 29 s. On top of that the first command after every model call saw a scene whose age already included the model's latency, so any call longer than about 15 s produced an expired (> 20 s) scene.
2. **The obstacles that vetoed the approach were the robot's own upper arm.** Every conflict point behind the step-4, step-46 and step-49 vetoes except one is an external-camera pixel on the robot, which DA3 places 27 to 33 cm too near (shoulder and base) or 8 to 10 cm too far (hand links). The 6 mm self-filter band removed only 17 to 41 points per frame, most of them table pixels, and kept 333 robot-pixel points as "external" obstacles in frame 0.
3. **The one metric grounding was correct and was rejected.** The model's handle pixel lifted through the external depth lands 2.9 mm from the true handle mesh. The harness rejected it because the wrist camera's depth disagreed by 44 mm; the wrist depth is biased by +3 to +7 cm on this scene and is not usable as a metric check.

Codex's own reports on the run are accurate in their numbers (82 of 95 checkable statements confirmed, 10 partly, 0 refuted, 3 unverifiable) and identify the terminal bug correctly. They omit the absolute speed, that the vetoing points are robot pixels mis-measured by DA3, that the rejected grounding was right, and that the lineage has now produced six runs, 78 calls and about 52 minutes of control with zero gripper closures.

**V3 changes the picture.** Within two hours of run 003 Codex replaced V2 with `foundation_vision_harness_V3`: H1's fragment loop and model-written obstacle boxes, plus a batched action protocol, an on-demand depth tool the model may call, and a joint-limit/self-collision guard. It completed all three tasks it was given (T04 can 244 s, T08 bottle 261 s, official T07 358 s online), raw success 3/3, strict success at the finish instant 1/3 and after a 10-step hold 3/3, with 0 forbidden contacts on T04 but a pot slip on T08 and palm-on-basket and finger-on-table contacts on T07. See section 9 for the verified first look and section 10 for what this means for the next step.

---

## 1. What happened in run 003

| Item | Value | Source |
|---|---|---|
| Outcome | `blocked`, reason `recovery_exhausted_without_changed_evidence`, at step 61 | `result.json` |
| Online wall / total | 208.39 s / 242.29 s | `result.json` |
| Actions | 61: 51 fragment control steps, 10 observation holds | `trace.jsonl` |
| Model requests | 6: 5 accepted plans, 1 `ProtocolRepairError` | `events.jsonl` |
| Gripper closures / grasp attempts | 0 / 0 | `trace.jsonl` |
| Forbidden contacts (evaluator) | 0 points in 1,587 checks; objects moved 0 mm | `terminal_evaluator_status.json`, `truth_terminal.json` |
| Completed segments | 1 (nominal −Y 50 mm; 47.4 mm path in 39 steps, 2.7 mm final error) | `events.jsonl` `segment_result` |
| Net end-effector displacement | 48.6 mm; height above the table 25.0 cm at start and end | `trace.jsonl` |
| Grip-site distance to the white mug origin | 0.293 m at start, 0.272 m at the end (xy 149 → 102 mm) | `trace.jsonl`, `truth_terminal.json` |
| Source | 362 files unchanged during the run; live tree identical to the launch snapshot at analysis time | `terminal_source_audit.json`, hash check |

**Timeline (online seconds).**

| t | Step | Event |
|---|---|---|
| 0–26.5 | 0 | Call 1 (26.5 s): a coherent 5-segment grasp plan: grounded approach to the handle pixel [174,263] at standoff 0.14 m, then 0.04 m, contact approach, close, lift 9 cm with a `held` checkpoint. |
| 30.6 | 1 | Source binding waited 3.9 s (the scene bound was already 30.7 s old). Grounding of the handle rejected: `contact_depth_crossview_conflict`. All candidates unavailable → handoff. |
| 30.7–45.3 | 2 | Call 2 (14.6 s) rejected by the harness validator: the model put an object on an `observe` checkpoint, which the instructions permit (§6). |
| 45.3–54.0 | 3 | Call 3 (8.7 s, 0 reasoning tokens): the same plan with the object removed: move above the handle, observe. |
| 59.3–65.3 | 4 | Binding wait 5.0 s. First command rejected: `observed_collision_risk`, link5 against "external" points that are robot pixels (§4). Forced refresh, then handoff. |
| 65.3–81.8 | 5 | Call 4 (16.4 s): −Y 50 mm, then an observe move. |
| 87.0–92.5 | 6 | Binding wait 4.8 s for a relative move that never reads the binding. Scene expired (21.5 s old) → forced refresh 5.4 s. |
| 92.5–121.9 | 7–45 | The −Y 50 mm segment: 39 steps, every command capped at 5 mm, 1.26 mm realised per step, 1.6 mm/s. |
| 122–136 | 46 | Next segment (descent toward the handle) rejected: `local_candidate_blocked`, link2/link5 against external points (one robot pixel, one unexplained); two refreshes of byte-identical frames; handoff. |
| 136–151 | 47 | Call 5 (14.8 s): −Y 45 mm with two candidates, then observe. |
| 156–169 | 48–49 | Binding wait 4.8 s. Scene expired (19.8 s) → refresh. Candidate 1 rejected (`unresolved_swept_region`, all eight points robot pixels); candidate 2 selected. |
| 169–177 | 50–59 | Eleven steps: 1.1 mm of motion in total; steps 52–59 are eight accepted null actions (rotation-first option, 0 mm translation). The scene reaches 20.05 s → `scene_expired`; the fingerprint equals step 49's, so no refresh: handoff to the model although frame 59 would have published 4.55 s later. |
| 177.8–203.4 | 60 | Call 6 (25.6 s): the plan returns to plan 1's strategy (lateral move with the jaws rotated, then a grounded observe). |
| 208.4 | 61 | Binding wait 4.8 s; the bound scene is 30.6 s old (model latency + perception). Filter returns `scene_expired` in 11.5 ms without any geometric check; the fingerprint repeats → `blocked`. |

---

## 2. Why it ended: the terminal chain (verified line by line)

- `motion.py:302-304` returns `reobserve`/`scene_expired` whenever the scene is older than `unavailable_scene_s` = 20 s, before any contact or collision assessment (`_contact_key` at 305, `_assess` at 343).
- `executor.py:61-135` builds the recovery fingerprint from `{reason, causes, object, contact_region, view, pixel_patch}` plus the local robot state (tolerances 10 mm, 0.1 rad, 0.05 rad on joints). Scene age and frame id are excluded by design (docstring at 62-67). The expiries at steps 49, 59 and 61 therefore hashed to one key (`42435e…`); the robot moved 0.845 mm between steps 49 and 59 and 0.000 mm between 59 and 61.
- `_recover` (`executor.py:493-523`) refreshes perception once per fingerprint (`recoveries`, line 496). At step 59 the key was already present, so no refresh: `candidate_unavailable`, then `_handoff` (477-491), which adds the key to `failed_segments`. At step 61 the key is in `failed_segments` → `_terminal('blocked', …)` (484-485).
- At step 59 frame 59's job was in flight and published 4.55 s after the handoff (`snapshots.jsonl` revision 17). The handoff instead issued call 6 (25.6 s), and the next bound frame (60) was 30.6 s old at binding.
- `_consume_binding` (`executor.py:270-271`) sets both the immutable `source_scene` and the navigation `scene` to the bound frame, so the stale frame also became the frame the filter aged.

Why bound frames are always stale: `SceneService.submit` (`V1/scene.py:293-304`) stamps a frame with the time of the FIRST submit for that observation object, and that first submit happens before the model call (a cadence-skipped submit still registers the stamp, 303-304). The post-model binding submit (`executor.py:376-379`) reuses it. Measured ages at binding: 30.7, 13.9, 21.5, 19.8, 30.6 s = model latency + about 4.6 s of perception + 0.4 s of saving. With a 20 s expiry, any model call longer than about 15 s delivered an expired first command.

Codex's account of this chain (REPORT.md "为什么提前结束", DEEP_ANALYSIS §1) is correct. Its proposed fix 2 ("the navigation scene takes the latest observation") would not have saved step 61 by itself: the latest frame (59) was also 30.8 s old; a new submit at activation is needed.

---

## 3. Why it was slow: three independent mechanisms

**3.1 The 5 mm slow mode was always on.** `motion.py:308-310`: when scene age > `stale_slow_s` (5 s) the command is limited to `unknown_command_m` = 5 mm. Perception publishes a fresh frame 4.44 to 7.35 s after capture (`snapshots.jsonl`), and the age at the first filter call adds activation overhead, so the age at all 51 dispatched commands was 5.40 to 19.81 s (median 12.4 s). 37 of 39 commands in the completed segment were exactly 5.000 mm. The OSC's steady-state gain under the fitted plant (a = 0.711, b = 0.0732, so b/(1−a) = 0.253) realises 1.27 mm per 50 ms step; measured median 1.256 mm. Independently, `unresolved_nearby_slowdown` (`motion.py:253-257`) would have capped 44 of the 51 steps at 5 mm even with a fresh scene, because robot-pixel "unresolved" points lay within 1.5 cm of the moving links.

| Quantity | Run 003 | Reference |
|---|---|---|
| Commanded translation per step | 5.0 mm (37/39 steps) | H1 transit cap 100 mm |
| Realised translation per step | 1.26 mm median | H1 ≈ 25 mm at full command |
| Speed while moving (steps 8–45) | 1.6 mm/s | |
| Net motion over the run | 48.6 mm in 208 s | H1 T04: 549 actions, two mugs moved, in 287 s of control |

**3.2 Each control step cost 0.77 s.** On the 48 pure control steps (no model call, no wait): period 0.765 s mean (p50 0.69), of which the command filter took 0.505 s (p50 0.455), the simulator step 0.158 s, feedback/recording/status/trace about 0.10 s. The filter is a per-OSC-step convex-hull and KD-tree check over 25 to 45 k points; it doubled (0.32 → 0.58 s) while the perception worker thread was busy, consistent with GIL contention (inferred from timing coincidence, not profiled). The uc7.2 plan's M0 target is 0.05 to 0.06 s per step; H1 spent 0.28 to 0.40 s per step with inline video and per-step PNGs, 0.16 s per action of non-model time by its own summary.

**3.3 Perception waits on the critical path: 57.1 s (27.4 %).** Five source-binding waits (3.9, 5.0, 4.8, 4.8, 4.8 s); the three at steps 6, 48 and 61 (14.4 s) blocked `move_relative` candidates that never read the bound anchor (`_needs_source_binding`, `executor.py:56-58`, is true for any anchor; `_resolve`'s anchor correction at 438 is guarded on the original skill being `move_to_pose`). Seven forced refreshes inside `_recover` cost about 33.8 s, of which about 13.3 s re-processed byte-identical frames (SAM and DA3 cache hits that still took 3.2 s of assembly plus 0.43 s of `npz` saving each). The perception worker's 94.8 s of processing is 19.1 s SAM + 4.1 s DA3 + 4.5 s geometry + 67.1 s of untimed assembly and saving.

**3.4 Model time: 106.7 s (51.2 %) for 6 calls**, mean 17.8 s. The latencies match the H1 latency model (6.6–6.9 s + 0.051 s per reasoning token + 0.007 s per visible token + 0.019 s per 1k input tokens) within −3.5 to +3.0 s, so the service window was normal. Four of the six calls were failure handoffs; one was a validator repair (14.6 s + 8.7 s = 23.4 s, 11 % of the run).

---

## 4. Why it never approached the mug

**4.1 The geometry vetoes were the robot's own arm.** Truth depth was rendered from the robosuite Panda visual meshes at the logged joints and from the LIBERO object meshes at their true poses (the pipeline validates itself: truth-mesh silhouettes overlap the SAM masks with IoU 0.91–0.99).

| Veto | Rule and numbers | What the points are |
|---|---|---|
| Step 4, link5 `observed_collision_risk` | Point from external pixel (290, 6): 36.0 mm outside the link5 hull, inflated by 41.9 mm (3 mm margin + 7.2 mm motion band + 20 mm single-view error + 11.75 mm voxel radius) → "overlap" 5.9 mm | Robot pixel. True depth 1.002 m, DA3 0.735 m (−267 mm). Classified `external` because no SAM robot query existed before step 5 and the point sat 289 mm in front of the FK envelope (`scene.py:228,234`: "foreground" protection). |
| Step 46, link2/link5 `local_candidate_blocked` | Points 6.4 and 10.5 mm outside the hulls, inflated by 29–30 mm | (299, 4) is a robot pixel (−309 mm). (299, 50) is unexplained (not robot, object or table in the truth model). |
| Step 49, link6 `unresolved_swept_region` | Links approached "unresolved" points by 0.13–0.18 mm against a 0.1 mm tolerance | All eight points are robot pixels (+84…+91 mm or −316 mm). The option ladder then accepted eight null actions (0 mm, 0.00015 rad) until the scene expired. |

The self filter cannot work with this depth error: `self` requires DA3 to agree with the FK envelope within 6 mm (`self_surface_band_m`), while the measured DA3 error on robot pixels is +76 to +103 mm (hand links) and −250 to −330 mm (shoulder and base); 0.03 % of robot pixels fall within ±20 mm. Result per frame: 17–41 `self` points (14–30 of them table pixels), 2,028–2,874 robot-pixel points left `unresolved` (the slow-down zones), and 333 / 339 / 129 / 6 robot-pixel points admitted as `external` obstacles at frames 0 / 4 / 46 / 60 (the drop coincides with the model naming a robot target at step 5, which enabled SAM robot queries).

**4.2 The grounding that was rejected was right.** Along the external ray through the model's pixel [174, 263] the true porcelain-mug handle surface is at 0.8152 m and DA3 gave 0.8128 m (−2.4 mm); the lifted point is 2.9 mm from the nearest handle vertex, 65 mm from the mug axis (the handle spans 50–81 mm), and inside the handle's height range. The crossview test reprojected it into the wrist image 2 px off the thin handle onto table background and compared with wrist DA3; at the true handle pixel the wrist DA3 is +30.6 mm off against a 26.1 mm tolerance, so even a perfect lift would have been rejected. After this the model stopped using grounding (plans 3–5 used metric relative moves) and returned to it only in plan 6.

**4.3 What the model was told.** The feedback after each veto carried reason codes, part names, anonymous world points and overlaps, but not that a slow mode existed, not the 5 mm cap, not the scene-age regime, and no free direction. The model's replies ("uncertain arm geometry", "arm-region occupancy", "scene expired") show it understood the labels and could do nothing with them.

---

## 5. How good is the harness's own metric geometry

This matters beyond run 003: the agreed direction takes metric estimation away from the model and gives it to the harness. Measured against rendered truth on this scene (frames 0, 4, 46, 47, 60):

| Surface | External camera (DA3) | Wrist camera (DA3) |
|---|---|---|
| Table plane | median +13 to +17 mm, RMS 19–34 mm; depth compressed with slope 0.84–0.92 (near table +20–30 mm too far, far table up to −80 mm too near); fitted plane within 7 mm of the true height, tilted 2.4–4.4° | median +64 to +71 mm, slope 0.05–0.26: essentially flat; the DA3 table lies 64–70 mm below the real one |
| Object surfaces (frame 0) | can +3, white mug +1, yellow mug +25, plates +23 to +29, red mug −38 mm | can +3, white mug −34, yellow mug +29, red mug −127 mm |
| Robot pixels | +76 to +103 mm (hand links), −250 to −330 mm (shoulder, base); 0.03 % within ±20 mm | fingers +50 to +80 mm (ratio 2.0–2.5) |
| Per-object point clouds | mugs within ~1 cm of the true surface (p50 3.8 and 8.3 mm); plates sink 10–14 mm below the table with 31–34 mm centroid error; the red mug floats 42–46 mm | |
| A model-chosen object pixel lifted by DA3 (the "model points, harness measures" scheme) | 3-D error p50 26 mm, p90 47 mm; 35 % within 20 mm | p50 32 mm, p90 82 mm |
| Affine calibration against the known table plane | object median +14 → −1 mm, lift p50 26 → 19 mm, p90 unchanged (46 mm), robot +103 → +79 mm | destructive (slope 0.28 → lift p50 405 mm) |
| SAM 3.1 masks | IoU 0.91–0.99 against truth silhouettes (mugs, both cameras); the segmentation is not the problem | |

Consequences. V2's constants (20 mm single-view position error, 3 mm collision margin, 6 mm self band, 0.1 mm "deepening" tolerance) are all below the measured noise of every surface class. The crossview arbitration keeps the worse measurements: accepted external table points err −35 mm median (far table, outside the wrist view) while the "disputed" ones (near table, contradicted by the biased wrist) err +20 mm. For the plan's M3 this means: the external camera's DA3 is metric to 2–3 cm on objects and can carry approach authority only with a measured, per-view, per-surface σ; the wrist camera's DA3 must not be used as a metric arbiter on this scene; robot pixels must be owned by FK before any veto; a table-plane calibration helps the external bias but not object-specific offsets.

---

## 6. The model side

- The instructions are byte-identical across calls (984 tokens) plus a 557-token schema and one image pair (642 tokens); input grew only from 3.5k to 7.0k tokens with the harness's feedback evidence. H1's context grew from 7.9k to 87k tokens over 13 calls; this is a real improvement.
- Plan 1 was a complete, sensible grasp plan (approach the upper handle from above with the jaws across the handle, close, lift 9 cm, check `held`). It died on the grounding rejection before any command. Plans 3–5 were repositioning moves written in reaction to harness labels; plan 6 returned to plan 1. Zero grasp attempts.
- Call 2 was rejected by `protocol.py:250-251` ("only held/released checkpoints have an object reference") although the instructions (`client.py:92-95`) say the checkpoint object is the state-check object and "name it when needed", and the schema allows it for every state. Runs 001 (released checkpoint with null object), 002 (parent_id on a robot target, three times, 67 s) and 003 are one class: harness cross-field strictness on auxiliary metadata that rejects a whole response; r3 fixed the first two rules and not its own.
- Hygiene holds: no task or benchmark nouns in the instructions or schema beyond the task sentence and the model's own words; no evaluator data in any context; 0 API-key calls; the images the model saw are byte-identical to the perception raster (the raw observation is their left-right mirror; the harness handles the convention correctly).
- Attribution of the six calls: five lost to harness-side gates (grounding veto, validator, two geometry vetoes, expiry), none to the service, none to a model error. The model's single-candidate segments made every veto a full round trip, which is a protocol-design cost rather than a model mistake.

---

## 7. Codex's own reports, audited

95 checkable statements in `REPORT.md`, `DEEP_ANALYSIS.md`, `analysis.json`, `analysis_geometry.json` and `analysis_detail/*.json` were re-derived: 82 confirmed, 10 partly, 0 refuted, 3 unverifiable. Every timing, count, fingerprint and geometry number reproduces, including the FK residuals (−288.57, −331.45, +103.81 mm), the handle reprojection (−44.21 mm vs a 26.15 mm tolerance) and the 5×5 depth span (26.28 vs 34.77 mm). The partly-correct items: "no SAM robot mask covers the point" is vacuous at frame 4 (no robot target existed yet); "disputed isolation worked" is true of the implementation only (no veto came from disputed points, and no outcome changed); frame 60 was 30.17 s old at publish and 30.60 s at binding, not 30.61; "51 checks passed" names 51 test items whose pass status is not recorded with the run; "even without the bug the speed cannot meet three minutes" is an extrapolation the run's data support but do not prove. Omitted facts: 0.24 mm/s average and 1.6 mm/s moving speed; the gripper never closer than 272 mm to the target; the veto points being DA3 phantoms of the robot's own arm; the rejected grounding being correct; 79 numeric parameters that can veto, scale or terminate motion, of which only the four OSC plant constants (taken from the uc7.2 plan's EXC-0 fit) have a documented measured source; the zero-closure streak across six lineage runs.

---

## 8. V2 against the agreed direction, and the lineage record

**Where V2 stands.** V2 implemented parts of the plan's layers 3–5 (a resident asynchronous SAM/DA3 worker, pixel-to-metre grounding, an FK hull body model, a committed segment queue, honest `finish`/`blocked`, bounded context, no model-written metric obstacle boxes) and skipped layers 1–2 (an equivalence baseline, honest measurement, the fast loop). It inverted the plan's central rule: predicted geometry held veto power over every OSC step, perception waits sat on the critical path, and staleness was treated as motion failure. Of the earlier analysis's shared root causes, R2 (point-only or veto-only geometry), R3 (robot self-surfaces), R4 (vetoes), R7–R9 recur; R1 (model-written metres) persists as the fallback once grounding fails; R6 (honest termination) improved but produced a false give-up.

**The lineage on this scene** (all init 0, seed 7, Astra high):

| Run | Start (UTC) | Duration | Calls | Actions | Closures | Outcome | Terminal class |
|---|---|---|---|---|---|---|---|
| foundation_harness 001 (v1) | 11-07 23:15 | 202 s | 12 | 19 holds | 0 | user stop | geometry veto at start |
| foundation_harness 002 (keypose-2) | 11-08 03:28 | 1,534 s | 21 | 306 | 0 | user stop | self-surfaces as obstacles; perception per step |
| vision V1 001 | 11-09 12:31 | 33 s | 1 | 1 | 0 | blocked | planning budget after 40 configurations |
| vision V2 001 | 11-09 17:15 | 118 s | 1 | 0 | 0 | error | validator |
| vision V2 002 | 11-09 18:14 | 1,053 s | 37 | 176 | 0 | user stop | geometry vetoes + perception stall |
| vision V2 003 | 11-09 19:59 | 208 s | 6 | 61 | 0 | blocked | expiry fingerprint (this report) |
| **H1 T04 reference** | 11-02 23:48 | 402 s wall, 287 s control + 17.5 s plan | 14 | 549 | 2 | raw success (oracle-stopped; strict at terminal false, hold-only) | |

Six H2-lineage runs: 78 calls, about 52 minutes of control, 0 closures, 0 grasp attempts, 0 forbidden contacts (nothing was ever near an object). The two autonomous `blocked` claims (V1 001, V2 003) were both harness-internal conditions, not task impossibility.

---

## 9. V3 first look

(Filled in after the V3 analysts and verifiers report; see the end of the document for the current status.)

## 10. The next step

(Filled in after section 9.)

## 11. Verification

(Filled in after the adversarial verifiers report.)

## 12. Files

- `work/w1/`: the six analysts' reports (`timeline_speed.md`, `code_trace.md` + `code_trace_inventory_table.md`, `geometry_truth.md`, `model_protocol.md` + `model_protocol_requests.md`, `lineage_vs_plan.md`, `codex_claims_audit.md`).
- `work/w2/`: the four verifiers' reports.
- `work/w3/`: the V3 first-look reports.
- `work/scripts/`, `work/results/`, `work/tables/`, `work/figures/`: every script, JSON result and figure cited above (run with `.venv-foundation-vision/bin/python -I <script>`; all paths absolute and read-only).
- Codex's own write-ups stay in the run folder: `runs/foundation_vision_harness_V2_t04_can_obstacle_003/{REPORT.md, DEEP_ANALYSIS.md, analysis.json, analysis_geometry.json, analysis_detail/}`.
