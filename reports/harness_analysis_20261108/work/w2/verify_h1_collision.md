# Verification of `w1/h1_collision.md` (key: verify_h1_collision)

I tried to refute each finding F1–F10. I re-read the cited code and recomputed every number from the raw artifacts (`runs/*/trial/trace.jsonl`, `collision.json`, `requests/*.request.json`, `inputs/layout_manifest.json`, `physical_truth.json`, `result_summary.json`). I did not reuse the analyst's tables. I read the analyst's scripts only to explain differences.

Rules followed:
- Nothing under `/workspace/gpt_field` was edited.
- I made no simulator, model or GPU calls.
- The frozen H1 router was replayed from a byte-copy of `agent_speedup_v2_obstacle_harness_100mm/src/gpt_field` placed in `w2/pkg/`, run with `python -B`.
- Truth geometry was used only for diagnosis.

Paths:
- Scripts: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/scripts/` (`extract.py`, `episodes.py`, `phase_share.py`, `stages.py`, `route_table.py`, `start_blocked.py`, `t08_cf.py`, `t08_map82.py`, `box_acc.py`, `carry_replay.py`, `calls.py`, `stalls.py`, `guard.py`, `profile_speed.py`, `standoff.py`, `model_box_sweep.py`, `req_text_search.py`).
- Numbers: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/data/verify_h1_collision_numbers.json`.

Code paths are relative to `agent_speedup_v2_obstacle_harness_100mm/src/gpt_field/`.

## 0. Baseline facts re-derived

- **Totals reproduce exactly.** Per-run forbidden totals from the cumulative `sim_status.collision.counts` are 3,372 / 1,325 / 726 / 3,826 / 1,650 / 4,715 / 2,068, for 17,682 in all. They match `collision.json → generic.counts`. The class totals also match: palm (`robot_nonfinger_manipulated_object`) 9,739 (55.1%), `manipulated_object_scene` 3,313, `robot_scene` 1,126, `robot_object` 3,504.
- **"Samples" are contact points.** The counters are contact *points* summed over physics substeps (`agent_speedup_v2_contacts.py:257-287`, the increment is at `:287`), not substeps. One substep can add several points. This does not change any ratio.
- **Episode count is 27, not 29.** Strictly contiguous runs of steps with a forbidden increment give **27** H1 episodes (median 13 steps, range 1–99). The analyst's 29 comes from splitting two episodes that overlap in time by class:
  - T04 20mm, palm 120–153 and mug–can 141–144;
  - drawer 001, hand–cabinet 826–841 and bowl–cabinet 841–853.
- **Two of the "seven H1 runs" are not the frozen H1 build.** T04 20mm and T04 default ran `user-check-measured-fragments-3` and `agent-first-prompt-5`. Their `frozen_source` differs from H1 in `_routing.py`, `_fragments.py`, `_agent_client.py` and `tracking.py`, and they use a 0.02 m transit cap (`TRANSIT_ACTION_M='0.02'`).
  - The differences are tracker cap and lookahead only, so the collision mechanisms are unchanged.
  - Speed and transit numbers from these two runs are not frozen-H1 numbers.
  - The other five H1 runs and both H3 runs are byte-identical to H1 for those files (`cmp`).

## 1. Verdicts

### F1. Point-only router, transit stages only. Verdict: confirmed_with_corrections. Severity: critical.

**Confirmed:**
- The scope matches the code:
  - `_routing.py:3-4` ("checks the grip-site path … not whole-robot collision");
  - box expansion at `:77-90`;
  - start and goal refusal at `:105-108`;
  - routing only when `profile=="transit"` with `move_to_pose` or `move_relative` (`_fragments.py:144-145`, `:293`; `approach` requires the contact profile, `:159-160`).
- Routed-transit stages account for 4,304 / 17,682 samples (24.3%). The rest:
  - unrouted contact approach: 8,570;
  - unrouted `set_gripper`: 4,291;
  - unrouted contact `move_relative`: 84;
  - direct `local_motion`: 433.
- Four episodes are dominated by routed stages, all `straight_path_valid=True`: T08 37–83, and drawer 001 at 1159–1180, 1227–1232 and 2217–2231.
- **T08.** The stage from global step 35 to 81 is `move_to_pose`, routed, `straight_path_valid=True`, clearance 0.018. Its first contact is `ketchup_obstacle_1_g4` against `gripper0_finger1_collision` at (0.0100, −0.0630, 1.0456), which is the bottle top. In the hand frame that point is R1 = +52.7 mm, R2 ≈ +16 mm from the grip site, so it is the fingertip edge, not the grip site.
- drawer 001 link6, link7 and hand against the cabinet: 414 samples (287 + 2 + 2 + 123).

**Refuted sub-claim: "even the TRUE bottle box + 0.018 m passes the point check".** That was tested with an idealized point-in-box sampler (`w1/.../truth_sweep.py: seg_hits_aabb`). The frozen `TransitRoute`, with its 8-corner lattice rule (`fine_distance_planner.py:101-106`), does something else for that stage with the true box and 0.018 m clearance:
- It returns **`straight_path_valid=False`** and runs `lazy_fine_grid_astar` (1 call).
- The A* path is just the same descent snapped 1 mm onto a lattice line: maximum lateral deviation 0.0010 m.
- Measured grip-site tracking would also have failed `field.query` at steps 69–81.

The conclusion still holds: even with perfect geometry, a point router would have let the finger hit the bottle. The stated mechanism ("accepted as straight") does not.

**Body-sweep sub-claim.** With my two-finger model (from `robot_facts_v2.json`) and the true box:
- With the 0.018 m clearance, it flags the stage at t = 0.1–0.8 under a synchronized slerp.
- With 0 clearance it does **not** flag the planned slerp sweep. It flags only the *measured* profile, where the hand arrives first and then rotates in place (finger hits at steps 37–45; see missed issue M1).

So the sweep catches this case only if it models the controller's real rotation timing, or adds a clearance.

**Correction: finger extent.** "Open fingers reach about ±47 mm" is the pad outer face. `robot_facts_v2.json` gives:
- finger body `finger_and_pad_R1_mm` up to ±66.4 mm at q = 40 mm;
- fingertip landing band ±54.3 mm.

The contact at +52.7 mm lies outside the analyst's single ±47.5 mm finger box. Their sweep under-models the fingers.

**Naive-fix counts reproduce.** Sweeping against the model's own boxes:
- with the analyst's hand model, 21 / 111 accepted stages are flagged, 14 at the start pose;
- with my wider fingers, 23 / 111;
- the T08 t100 colliding stage is flagged by neither, because the model's T08 boxes are wrong (F2).

**Improvement assessment:**
- **Direction is right and general.** It uses robot geometry only and works as a local planner rather than a veto.
- **Gap 1: the largest class is not covered as specified.** Palm-on-target (55%) is not covered, because the prompt tells the model to remove the manipulated object from the avoid set before grasping (`_agent_client.py:101-102`). The map has no object identity or per-part permissions, so this needs F9 (see M3).
- **Gap 2: interpolated orientation is not what executes.** A check at interpolated orientations is unsound unless the controller synchronizes rotation with translation (M1).
- **Gap 3: escaping into the obstacle.** "Escape" from a start inside an inflated box (14 of 21 hits) can itself move into the object just grasped or released.
- **Gap 4: speed cost is unmeasured.**
  - "Well under 10 ms per stage" is an estimate.
  - Reusing `foundation_harness/robot_geometry.RobotCollisionModel` and `motion.CandidatePlanner.check_motion` (both exist: `robot_geometry.py:151`, `motion.py:356`) is costly. H2's checker costs **0.12–0.24 s per configuration** (`w1/h2_planning_execution.md:60`), which is seconds per stage.
  - My untuned pure-Python full-arm box sweep (10 parts × 4 boxes × 41 samples) took **355 ms**.
  - A vectorized box-only implementation with a broad phase is needed to make the cost negligible against 0.155–0.198 s per control step.
- **Direct motions.** Direct `motion` actions must be checked too: 135 of the 414 drawer link samples happened in `local_motion`.

### F2. Model boxes miss added obstacles by 5–12 cm. Verdict: confirmed_with_corrections. Severity: high.

**Confirmed** (`box_acc.py` against `layout_manifest.json`):

| Map | IoU | Centre error (m) | Note |
|---|---|---|---|
| T04 t100 can, maps 0/162/163/347 | 0.116 | (+0.048, 0.000, −0.025) | same can box in all four maps; never revised |
| T04 20mm, map 0 | 0.124 | — | |
| T04 20mm, map 155 | 0.301 | — | after the contact |
| T08 t100, map 0 | 0.000 | (−0.085, −0.009, −0.083) | |
| T08 t100, map 82 | 0.196 | — | |
| T08 stove-off, map 0 | 0.000 | (−0.077, −0.024, −0.093) | |
| T08 stove-off, map 161 | 0.352 | — | |

- First-map box bottoms in T08 are at z 0.79–0.80 m; the table top is 0.90 m.
- After close-up views, centre errors were 0.020–0.036 m, and the expanded boxes contained the obstacle in 3/3 cases.

**Correction: the camera-ray claim holds for T08 only.**
- T08: the error is ≥98% along the external ray (along +0.119 / +0.122 m, perpendicular 0.001 / 0.020 m), i.e. a depth error.
- T04: the 5.4 cm error is about 90% **perpendicular** to the ray (along −0.022 m, perpendicular 0.050 m).
- Box bottoms below the table also occur in T04 (0.40 m against a table at 0.437 m), which F2 and F9 do not mention.

**Refuted: expected effect.** F2 says step-82-quality boxes before motion "would have removed the T08 bottle contact (3,504) and the T04 20mm mug–can contact (33)". Neither follows:
- Replaying the colliding T08 stage against the **step-82 map** gives A* with ≤1.0 mm deviation from the colliding descent, so the router would still allow the finger sweep.
- The T04 20mm mug–can contact occurred in an unrouted `set_gripper` close, which boxes cannot affect.
- Any benefit would come from the model choosing a different approach, which is speculative.

**Improvement assessment:**
- Moving metric estimation out of the LLM (triangulation from pointed pixels, support-plane fit, overlays) is general and sound.
- Missed supporting evidence: H3's perception context roughly halved the initial box error even without "usable 3-D centres" (M6).
- Costs not estimated: a fixed wrist survey adds control time (not quantified; my guess is about 5–15 s per task), and overlays add image tokens (+1–3 s per call, unmeasured).
- This is a prerequisite for F1, not a fix on its own.

### F3. Palm-on-object is the largest class; the palm envelope is never given. Verdict: confirmed_with_corrections. Severity: high.

**Confirmed:**
- 10 episodes, 9,739 samples (55%), in 5/7 runs. All are `gripper0_hand_collision` against a mug or moka part.
- Request text contains no "palm", "footprint", "fingertip" or "31.04". I checked all requests of four runs with images stripped (raw greps hit base64). The context gives only `hand_to_grip_site_m`, pad midpoint, pad size and inner gap (`_agent_client.py:228-233`; `geometry.gripper`).
- `robot_facts_v2.json:11,18,440-447` holds the values.
- The prior plan recommended adding them (`reports/agent_speedup_analysis_20261026/plan/final_plan.md:92-93`).
- drawer 001 had 74 closed-gripper lift steps with palm contact (the analyst said 73).
- T04 t100's approach ended `pose_arrived` (1.9 mm error) after 10 steps of palm-on-rim contact.

**Corrections:**
- **White-yellow rim height.** The rim on the table is about 0.542–0.545 m (`physical_truth` high_z − low_z of 0.105–0.109 m, base at 0.4365 m). 0.552 m is its height on the plate. So the T04 t100 step-369 commanded depth is **≈0.059–0.062 m**, not 0.069 m. Contact began with the grip site about 0.050 m (not 0.06 m) below that rim.
- **Porcelain-mug numbers** (0.051, 0.053, 0.055 m) are right.
- **Depth range.** Commanded depths are 5.1–6.2 cm, not "4–7 cm".
- **"The model *therefore* chooses deep grasps" is not established.** H3 T04 had the same prompt and also no palm information (text search: 0 hits). It still targeted 15 mm shallower (36–45 mm below the rim) and had 0 palm contacts. Grasp depth is driven by the model's handle localisation, not only by the missing envelope.

**Improvement assessment:**
- **Giving the envelope** is sound, general (an allowed robot-geometry input) and costs 0 calls.
- **The depth bound "top − palm offset + margin" is weaker:**
  1. It depends on a model-estimated feature top, which carries F2's 2–12 cm errors.
  2. The conflict is with the rim *under the palm footprint* (±31.6 × ±100 mm), not with the grasped feature (the handle). A footprint height map from perception or depth is more robust.
  3. A rim − 31 mm bound is 5–14 mm shallower than H3's successful grasps, which risks weaker grasps and regrasps.
- **Expected effect.** In the four T04 episodes, the bound lies 4–21 mm above the contact-onset grip height (6.5–25 mm above the deepest point reached, not 6–38 mm). With an accurate rim it would have prevented them.

### F4. A* never runs; start refusals; hidden 0.02 m inflation. Verdict: confirmed_with_corrections. Severity: medium.

**Confirmed:**
- 117 H1 routing attempts: 111 straight, **0 planner calls**, 0 detours; 6 `route_start_blocked` plus 1 in H3.
- One tracking `invalid_measured_start` (T04 20mm step 538, 9.5 s).
- Recovery calls after the 7 refusals: 10.7, 10.5, 12.2, 10.3, 10.3, 15.4 and 15.0 s, **84.4 s** in total.
- The 8-corner rule exists in `fine_distance_planner.py:101-106` and `query`, `:117-130`, and is used at `tracking.py:83`.
- T04 t100's maps at 162 and 163 changed only the clearance (boxes identical).
- The models themselves noticed it ("avoids a coarse-grid start blockage", step 163; "local routing discretization issue", T04 20mm, step 538).

**Refuted in part: "7 refusals were spurious (start outside by 6–20 mm)".** Only **5/7** fit. The start's signed distance to the nearest expanded box (negative means inside):

| Case | Signed distance | Start outside? |
|---|---|---|
| T04 t100 @162 | +20.3 mm | yes |
| T08 @87 | +6.3 mm | yes |
| T08 @597 | +13.3 mm | yes |
| drawer 001 @1582 | +6.8 mm | yes |
| H3 T04 @349 | +6.4 mm | yes |
| T04 20mm @395 | **−0.4 mm** | no |
| T08 stove-off @179 | **−52.4 mm** | no; the hand was inside the model's own box |

Exact distances would fix 5 of 7; the other 2 need "escape first".

**Missed trade-off.** The lattice rule is also a hidden 0–20 mm per-axis margin. It alone invalidated the straight path in the true-box T08 replay. Removing it lowers effective clearance under 5–12 cm box errors, so it should be replaced by an explicit, reported margin or by the body sweep (M4).

**Expected effect.** About 6 calls (about 70 s) across the 9 runs, not "about 1 per obstacle task".

### F5. No contact sensing; late stall; a 3-step guard. Verdict: confirmed_with_corrections. Severity: medium.

**Confirmed:**
- The monitor is record-only (`agent_speedup_v2_contacts.py:1-4`; `agent_speedup_v2_sim.py:86-87`, as cited) and feedback always reports cause "unknown" (`_fragments.py:482`).
- The stall rule is at `_fragments.py:404-419`. Correction: unrouted stages measure `position_error_m`, not `route_remaining_m`.
- **T08.** z stayed pinned near 1.049–1.054 m while the jaw axis turned from (0.725, −0.689) to (1.0, −0.007); the stall fired at step 81.
- **Stalls.** 46 stall handoffs, with follow-up calls totalling 939 s (analyst 896 s). Categories: 15 forbidden, 10 near-target, 8 allowed contact, 13 none-far (analyst 14/10/8/14).
- **Guard re-implementation** (≥2 mm commanded, <0.3 mm progress, 3 steps):
  - fires inside 15/27 contiguous episodes, median 4 steps after onset;
  - 12,351 samples come after the fire point;
  - fires inside 46/46 stalls, a median of 7 steps before the recorded stall (559 steps, about 100 s);
  - active on 218/4,126 contact-free moving steps (5.3%), 195 of them in the drawer runs.
  - These reproduce the analyst's numbers. Their rule counts any active step inside an episode, which is why they report 18/29.

**Correction: the drawer 001 diagnosis was right.** "The model blamed a near-singular arm configuration" implies misdiagnosis, but the arm **was** at joint 4's limit:
- q4 = −0.070 against −0.0698 at steps 1156, 1175, 1180, 1185, 1190 and 1253;
- the first stall (1122–1155) had no contact and a 92 mm error.

The failure is reachability or joint-limit blindness, and the recovery moves then hit the cabinet with unmodelled links (M2). It is not a misdiagnosis.

**Improvement assessment:**
- **Collision.** The guard shortens contacts; it does not prevent them (it fires after onset). The user's goal is zero contacts. It cannot see `set_gripper` contacts (no commanded motion), and 12 of 27 episodes go undetected.
- **Speed: net negative as written.**
  - Of 83 fire events, 24 occur in contact-free segments that went on to arrive or advance normally (21 of them in the drawer runs, many as near-target creep). Each would become an extra handoff.
  - At about 12–15 s per call, that is roughly 290–360 s added, against about 100 s of steps saved.
  - To be net-positive it must: treat in-band creep as arrival, resolve contact-free blocks (joint limits) locally, and report joint margins. Otherwise it adds calls.
- **Thresholds.** The 2 mm / 0.3 mm / 3-step values are new thresholds, which the user's rule disfavours. Expressing them relative to the measured OSC realization (≈0.24 of the command) keeps them general.

### F6. Rotation unchecked; the 1.2 rad veto. Verdict: partially_refuted. Severity: medium.

**Confirmed:**
- The veto is at `_fragments.py:28` and `:288-291`.
- Both first decisions were rejected at 1.572 / 1.569 rad and split into 45° then 90°.
- No orientation check exists.

**Refuted:**
1. **Cost.** "Wasting about 30 s" is wrong. The 30.8 s and 32.4 s calls produced the original decisions, which are needed anyway. The veto added replacement calls of **7.2 s and 7.7 s** plus one hold step. The prior plan measured 5 replacement calls for 42.4 s, about 8.5 s each (`final_plan.md:80`).
2. **Causation.** "The split put finger1 over the bottle" is not established. Without the veto, the single stage (0.23 m and 90°) would also arrive with rotation incomplete. Free-space stages finish translation about 3× before rotation (M1), so the hand would still rotate in place beside the bottle. The cause is unsynchronized rotation plus unmodelled fingers.

**Improvement assessment.** Removing the veto is consistent with the user's rule and saves about 7–8 s per occurrence. It is only safe together with M1, i.e. a controller split that couples rotation to translation, checked over the executed profile.

### F7. Held objects not modelled. Verdict: confirmed_with_corrections. Severity: medium.

**Confirmed:**
- `_routing.py:28-29` ("does not … infer object motion").
- **Carry replay.** With the T04 t100 step-163 map (clearance 0.04), the router accepts a straight carry at z ≥ 0.55 m and refuses z ≤ 0.54 m only as `route_goal_blocked`. With the true can + 0.04 it accepts z ≥ 0.56 m.
- The mug is grasped about 0.06 m above its base, so it hangs below the can top (0.513 m).
- Measured carry heights: T04 max 0.61–0.64 m; T08 1.22–1.26 m.
- Held-object contacts match: 697, 611 and 111 samples.

**Corrections:**
- **Vertical share depends on the definition:** 47–61% of the Euclidean path, 35–50% of L1 (analyst 39–53%).
- **Most vertical travel is mandatory** grasp and place descents and lifts, not extra carry height.
- **Speed saving is overstated.** "20–40% less vertical, 5–25 s per run" is too high. The excess carry height is about 5–8 cm, which is about 1–3 s per carry at the measured 7–13 mm/step transit speed. The value is safety, not speed.

**Improvement assessment.** Sound and general. The attached box needs the in-gripper offset (the handle grasp offsets the mug body by about 6 cm), and that offset is a model or perception estimate subject to F2.

### F8. Open-loop close and place. Verdict: confirmed. Severity: medium.

**Confirmed:**
- 11 `manipulated_object_scene` episodes and 3,313 samples in 5/7 runs, plus H3 T08's 2 episodes and 400 samples.
- The quote "Avoids forcing the stalled descent" is at T08 stove-off step 564.
- `set_gripper` only holds the pose and monitors the gap (`_fragments.py:387-394`).

**Improvement assessment.** A guarded place descent cannot tell rim contact from support contact using proprioception alone. Both block the descent, so "release on measured contact" could release a pot tilted on a burner rim. It needs perception of the support footprint and xy precision, as F8 itself notes.

### F9. Unlabelled, capped scratchpad map. Verdict: confirmed. Severity: medium.

**Confirmed:**
- 36 updates; the 4-box cap was hit in 10; clearances 0.012–0.08 m; 0 empty maps.
- The schema has no IDs (`_fragments.py:54-59`).
- Validation is at `_fragments.py:90-110`.
- The obstacle appears as a plan part or constraint in 0/4 plans. The one "can" mention is the scene sentence in the T04 20mm plan.
- 0 native-macro actions in all 9 runs.

**Improvement assessment.** Sound and general. It is also the missing prerequisite for F1 to cover palm-on-target and for F7 to attach held objects (M3).

### F10. The slow contact profile dominates actions. Verdict: confirmed_with_corrections. Severity: medium.

**Confirmed:**
- Contact-profile *move* actions are 57–67% of all actions (311–1,424 per run). Including `set_gripper` raises this to 63–74%.
- Realized/commanded ratio is 0.24–0.26.
- Non-model time is 0.155–0.198 s per action, computed as (`control_without_video_s` − model latency) / actions.
- 172–858 contact steps per run were more than 20 mm from their target.

**Corrections:**
- **Speeds.** Mean contact motion is 1.06–1.24 mm/action (not 1.2–1.4); transit is 4.0–12.9 mm/action.
- **"Does not make motion safer" is unsupported.** Contacts happen in contact stages because that is where the hand is near objects; this is confounded by proximity.
- **Realistic saving.** The 22–141 s/run upper bound assumes a 20 mm standoff. F10's own condition is "standoff > geometry uncertainty", and the post-close-up error is 20–37 mm. At a 40 mm standoff the saving is 58–374 steps, about 10–67 s per run. With pre-close-up errors (5–12 cm) it mostly disappears.
- **Stopping distance.** A guard firing about 3 steps after contact at transit speed (7–13 mm/step) means 2–4 cm of pushing, so the standoff must also cover stopping distance.

## 2. Missed issues

- **M1 (high). OSC rotation lags translation by about 3×.**
  - Free space: T08 segment 0 is 92% translated but only 27% rotated at step 10; T04 is 93% vs 20% at step 8.
  - So stages execute as "translate, then rotate in place". This is exactly the T08 finger sweep at the low pose.
  - Fix: couple per-step translation to rotation progress, or reorient at a cleared pose, and check the executed profile.
  - Cost: about 0 s when rotation already bounds stage completion (T08 segment 0 took 34 steps either way).
- **M2 (high). No arm joint-limit or reachability awareness.**
  - The model receives joint positions but no arm limits (only `finger_joint_ranges_m` is in the request text).
  - The Cartesian router does no IK feasibility check.
  - drawer 001: joint 4 at its limit led to a contact-free stall (92 mm error), then 9 recovery calls (142.8 s), during which link6, link7 and the hand touched the cabinet (414 samples).
  - Fix: joint-limit margins in the feedback, an IK feasibility check of stage targets, and local posture escape.
- **M3 (high). F1 cannot reach the 55% class without object identity.**
  - The prompt removes the target from the avoid set before grasping (`_agent_client.py:101-102`), and boxes carry no ID or permissions.
  - The fix needs F9's object IDs with per-robot-part contact permissions (fingers allowed on the target; palm and links forbidden).
- **M4 (medium). F4's exact-distance fix removes a hidden margin.**
  - The lattice rule adds 0–20 mm per axis, depending on grid alignment.
  - Replace it explicitly, and validate on contacts, not only on refusals.
- **M5 (medium). Finger geometry is under-modelled in the analyst's sweep.**
  - The sweep uses ±47.5 mm; the actual finger body reaches ±54–66 mm.
  - The T08 contact at R1 = +52.7 mm lies outside the analyst's finger box.
- **M6 (medium). H3's perception context already halved the initial box error.**
  - T08: IoU 0.146 vs 0.0, error 6.2 vs 11.9 cm. T04: IoU 0.284 vs 0.116, error 2.8 vs 5.4 cm.
  - F2 dismissed H3 ("0 usable 3-D centres"); this is the cheapest path toward F2's goal.

## 3. Bottom line

- **Robust:** the core diagnosis. The only collision mechanism is a model-box, grip-site point router on about 24% of the contact samples. The largest classes (palm-on-target, held object, rim) sit in unrouted stages, and there is no contact feedback.
- **Wrong or overstated:**
  - the "true box passes the point check" mechanism;
  - the "7/7 spurious refusals" claim;
  - the ~30 s veto cost (actually ~7.5 s);
  - the drawer "misdiagnosis";
  - F2's expected contact removals;
  - several speed estimates: F1 compute, the F5 net effect, the F7 saving and the F10 saving.
- **Proposed fixes:**
  - F1 and F3 need M1 (synchronized rotation), M3 (object-ID permissions) and a measured, vectorized checker.
  - F5 must avoid adding handoffs.
