# H1 (uc7.1 obstacle harness, transit100): collision avoidance and its effect on speed

Key: `h1_collision`. This was a read-only analysis. It made no edits under `/workspace/gpt_field`, started no simulator, made no model calls and used no GPU. All numbers come from saved run artifacts and offline Python. The scripts and intermediate data are in
`/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w1/h1_collision_work/`
(`scripts/*.py`, `data/*.json`).

**Code paths.** Paths are given relative to `agent_speedup_v2_obstacle_harness_100mm/src/gpt_field/`. I checked with `cmp` that `agent_speedup_v2_user_check_routing.py`, `_fragments.py`, `_agent_client.py` and `fine_distance_planner.py` are byte-identical to the main tree. The router, fragment and prompt findings therefore also apply to H3, which runs uc7.1's low-level control. The two H3 runs show the same behaviour (see F4 and F8).

**Runs analysed.** These are the seven H1-family runs, with paths under `/workspace/gpt_field/runs/`:

| Label | Run dir | Scene | Result | Actions | Calls | Forbidden samples |
|---|---|---|---|---|---|---|
| T04 t100 | agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001 | T04 + can | success | 549 | 14 | 3,372 |
| T04 20mm | agent_speedup_v2_uc71_can_obstacle_t04_20261103_001 | T04 + can | success | 1,276 | 27 | 1,325 |
| T04 default | agent_speedup_v2_uc71_default_routing_t04_20261103_001 | official T04 (**no can**) | success | 801 | 11 | 726 |
| T08 t100 | agent_speedup_v2_uc71_bottle_obstacle_transit100_t08_20261103_001 | T08 + bottle | success | 843 | 22 | 3,826 |
| T08 stove-off | agent_speedup_v2_uc71_stove_off_bottle_obstacle_transit500_cap100_t08_20261103_001 | T08 stove off + bottle | success | 889 | 17 | 1,650 |
| drawer 001 | agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_001 | 5-stage ordered | success | 2,364 | 43 | 4,715 |
| drawer 003 | agent_speedup_v2_uc71_drawer_bowl_moka_transit500_cap100_20261103_003 | 5-stage ordered | success | 2,131 | 37 | 2,068 |

For cross-checks I also used two H3 runs: `foundation_vision_t04_can_obstacle_20261105_001` and `foundation_vision_t08_bottle_obstacle_20261105_001`.

---

## 1. How H1 collision handling actually works

- **Model calls.** One initial plan call produces `parts`, `steps` and `keep` (upright/clear_of). It contains no geometry. After that, every handoff triggers one executive call, which returns `belief`/`plan`/`why`, one action (`execute | motion | macro | observe | finish`) and `obstacle_map`.
- **Obstacle map.** It holds at most 4 world-axis-aligned boxes and one `clearance_m` (0–0.20 m) (`_fragments.py:54-59`). The model writes the boxes from RGB and calibration (`_agent_client.py:92-105`). `null` reuses the cached map and an object replaces it completely (`_agent.py:314`; `_routing.py:25-48`). Validation only checks that values are finite, that lower is below upper, the box count and the clearance range (`_fragments.py:90-110`).
- **Execute stages.** An `execute` runs 1–6 measured stages, each under one of two profiles:

  | Profile | Envelope | Per-action translation cap | Where |
  |---|---|---|---|
  | transit | 0.50 m / 1.2 rad | 0.10 m | `_fragments.py:27-34` |
  | contact | 0.08 m / 0.5 rad | 0.005 m | `_fragments.py:27-34` |

- **Which stages are routed.** Only transit `move_to_pose` and `move_relative` (`_fragments.py:144-145`, `:293-317`). Contact-profile approaches, lifts, descents and retreats are never routed. Neither are `set_gripper`, `wait`, direct `motion` (`_agent.py:377-378`) or native macros.
- **How routing works.** At stage entry, `TransitRoute`:
  1. expands every box by the single clearance;
  2. builds a lattice with 0.02 m spacing centred between the start and the goal;
  3. refuses the stage if the cell around the start or the goal touches a box (`_routing.py:105-108`);
  4. accepts the straight segment if every cell it crosses is free;
  5. otherwise runs lattice A* (`_routing.py:111-143`);
  6. then tracks the polyline with a lookahead of at least 0.10 m. Each control step re-checks the measured point and stops if its cell touches a box (`tracking.py:83-84`).

  The module states its own scope: "checks the grip-site path against them, not whole-robot collision" (`_routing.py:3-4`).
- **What is not modelled.** The OSC controller interpolates orientation on its own, and no orientation is ever checked. The hand, the fingers, the held object and the arm links are not modelled.
- **When a stage ends.** A stage ends on measured arrival, a stable gap, a stall, an exhausted budget or a rise in finger gap. A stall means `route_remaining + 0.05·orientation_error` improved by less than 0.8 mm over 12 steps, checked after at least 16 steps (`_fragments.py:404-419`). The feedback always says `contact_or_collision_cause: "unknown"` (`_fragments.py:482`).
- **Contact monitoring.** The contact monitor is evaluator-only and `record_only` (`agent_speedup_v2_contacts.py:1-4`; `agent_speedup_v2_sim.py:86-87`). Contacts never reach the policy.
- **Native macros.** The v2 macro, keepout and observer machinery exists but was never used: all 7 runs have 0 native macro actions.

---

## 2. Where the forbidden contacts came from

Counts are physics-substep samples. I grouped them into **contiguous episodes**: runs of consecutive control steps with a nonzero forbidden increment, built from `trace.jsonl`'s cumulative `sim_status.collision.counts` and `pairs_by_class`. Each episode is joined to the active fragment stage, profile and gripper command, and to the motion direction (`scripts/contact_phases.py`, `scripts/episode_table.py`).

**Totals: 29 episodes and 17,682 samples. Episodes last 1–99 control steps; the median is 13.**

| Mechanism | Episodes | Samples | Share | Runs |
|---|---|---|---|---|
| A. Palm/hand (`gripper0_hand_collision`) vs the object being grasped, placed or carried | 10 | 9,739 | **55%** | 5/7 (all three T04 runs, both drawer runs) |
| B. Robot (finger, hand, link6, link7) vs scene during **routed transit** or direct motion | 5 | 3,918 | 22% | 2/7 (T08 t100, drawer 001) |
| C. Finger or hand vs fixture in contact-profile approaches | 3 | 712 | 4% | 2/7 (drawer 001, drawer 003) |
| D. Held object pushed into a neighbour during closing or lift | 6 | 1,452 | 8% | 4/7 |
| E. Placed object vs support rim (burner) during descent, release or retreat | 5 | 1,861 | 11% | 2/7 (+ 2 episodes / 400 samples in H3 T08) |

- Only **4/29 episodes (4,267 samples, 24%)** were dominated by routed transit stages. All four were on routes the planner had **accepted as straight-valid**:
  - T08 t100: the stage ending at step 82;
  - drawer 001: the stages ending at steps 1175, 1232 and 2233.
- The other 25 episodes (76% of samples) occurred in stages the router never sees.

<details><summary>Per-episode table (all 29)</summary>

| Run | Steps | n | Samples | Class | Mechanism | Dominant phase | Main pair |
|---|---|---|---|---|---|---|---|
| T04 t100 | 114-132 | 19 | 473 | robot_nonfinger_manipulated_object | A | grasp_close | porcelain_mug_1_g13~hand |
| T04 t100 | 261-300 | 40 | 865 | robot_nonfinger_manipulated_object | A | release_open | porcelain_mug_1_g13~hand |
| T04 t100 | 420-466 | 47 | 2034 | robot_nonfinger_manipulated_object | A | lift_closed | hand~white_yellow_mug_1_g3 |
| T04 20mm | 120-153 | 34 | 1292 | robot_nonfinger_manipulated_object | A | grasp_close | porcelain_mug_1_g13~hand |
| T04 20mm | 141-144 | 4 | 33 | manipulated_object_scene | D | grasp_close | porcelain_mug_1_g8~tomato_sauce_obstacle_1_g11 |
| T04 default | 137-152 | 16 | 726 | robot_nonfinger_manipulated_object | A | contact_hold_open | porcelain_mug_1_g13~hand |
| T08 t100 | 37-83 | 47 | 3504 | robot_object | B | free_transit (routed) | ketchup_obstacle_1~finger1 |
| T08 t100 | 223-232 | 10 | 110 | manipulated_object_scene | D | grasp_close | ketchup_obstacle_1_g10~moka_pot_2_g14 |
| T08 t100 | 362 | 1 | 1 | manipulated_object_scene | D | lift_closed | ketchup_obstacle_1_g10~moka_pot_2_g14 |
| T08 t100 | 510-513 | 4 | 149 | manipulated_object_scene | E | release_open | flat_stove_1_burner~moka_pot_2 |
| T08 t100 | 540 | 1 | 62 | manipulated_object_scene | E | retreat_open | flat_stove_1_burner~moka_pot_2 |
| T08 stove-off | 548-565 | 18 | 1220 | manipulated_object_scene | E | place descent / hold | flat_stove_1_burner~moka_pot_2 |
| T08 stove-off | 626-632 | 7 | 428 | manipulated_object_scene | E | release_open | flat_stove_1_burner~moka_pot_2 |
| T08 stove-off | 888 | 1 | 2 | manipulated_object_scene | E | release_open | flat_stove_1_burner~moka_pot_1 |
| drawer 001 | 148-160 | 13 | 293 | robot_scene | C | contact approach (open) | finger1~wooden_cabinet_1_g9 |
| drawer 001 | 826-841 | 16 | 402 | robot_scene | C | contact approach (open) | hand~wooden_cabinet_1_g14 |
| drawer 001 | 841-853 | 13 | 673 | manipulated_object_scene | D | grasp_close/lift | akita_black_bowl_1~wooden_cabinet_1_g16 |
| drawer 001 | 865-866 | 2 | 24 | manipulated_object_scene | D | lift_closed | akita_black_bowl_1~wooden_cabinet_1_g16 |
| drawer 001 | 1159-1180 | 22 | 287 | robot_scene | B | free_transit (routed) + direct | robot0_link6~wooden_cabinet_1_g9 |
| drawer 001 | 1182 | 1 | 2 | robot_scene | B | direct motion | robot0_link7~wooden_cabinet_1_g11 |
| drawer 001 | 1184-1185 | 2 | 2 | robot_scene | B | direct motion | robot0_link7~wooden_cabinet_1_g11 |
| drawer 001 | 1227-1232 | 6 | 123 | robot_scene | B | free_transit (routed) | robot0_link6~wooden_cabinet_1_g9 |
| drawer 001 | 2075-2173 | 99 | 2556 | robot_nonfinger_manipulated_object | A | grasp + 73 lift steps | moka_pot_1_g13~hand |
| drawer 001 | 2217-2231 | 15 | 353 | robot_nonfinger_manipulated_object | A | carry_transit (routed) | moka_pot_1_g13~hand |
| drawer 003 | 223 | 1 | 17 | robot_scene | C | grasp_close | hand~wooden_cabinet_1_g29 |
| drawer 003 | 667-676 | 10 | 611 | manipulated_object_scene | D | grasp_close/lift | akita_black_bowl_1~wooden_cabinet_1_g16 |
| drawer 003 | 1732-1749 | 18 | 590 | robot_nonfinger_manipulated_object | A | grasp approach | moka_pot_1_g13~hand |
| drawer 003 | 1813-1825 | 13 | 325 | robot_nonfinger_manipulated_object | A | grasp approach | moka_pot_1_g13~hand |
| drawer 003 | 1960-1981 | 22 | 525 | robot_nonfinger_manipulated_object | A | grasp approach/close | moka_pot_1_g13~hand |

</details>

**The labelled numbers in the task brief, explained:**
- **T04 t100, 3,372 samples.** Three palm-on-rim episodes during grasps and one placement. There were 0 can contacts because the model carried at z 0.62–0.64 m, not because of routing.
- **T08 t100.** The 3,504 `robot_object` samples are open `finger1` on the bottle top during a routed transit. There are 111 pot-vs-bottle samples during closing and lift, and 211 pot-vs-burner samples during release.
- **T08 stove-off, 1,650 samples.** All pot-vs-burner, during place descent and release.
- **drawer 001.** The 1,109 `robot_scene` samples are finger and hand vs cabinet in contact approaches (695), plus link6/link7/hand vs cabinet in routed transits and direct recovery moves (414).
- **drawer 003.** Mostly palm-vs-moka over three grasp attempts, plus bowl-vs-cabinet.
- **T04 default (no can), 726 samples.** All palm.
- **T04 20mm, 1,325 samples.** 1,292 palm and 33 mug-vs-can.

---

## 3. Findings (ranked)

### F1 (critical): the router checks one point, on one stage type

**Mechanism.** `TransitRoute` checks only the grip-site polyline against the boxes after one scalar clearance is added (`_routing.py:3-4`, `:77-90`, `:105-115`). It runs only on transit `move_to_pose` and `move_relative` (`_fragments.py:144-145`, `:293`).

The robot's own geometry is much larger than any clearance the models used (0.012–0.08 m across 36 map updates):
- the hand spans R1 −104…+100 mm and R0 ±31.6 mm;
- the palm underside sits 31 mm behind the grip site (`robot_facts_v2.json:18`, `:440-447`);
- the open fingers reach about ±47 mm.

The prompt admits the gap: "The planner does not discover missing obstacles or check the full robot" (`_agent_client.py:150-151`).

**Evidence.**
- Only 4/29 episodes were in routed stages, and every one was on a route the router had approved (Section 2).
- **T08 t100, steps 37–83.** `finger1` touched the bottle for 47 steps (3,504 samples). This happened in a routed `move_to_pose` with `straight_path_valid=True` and clearance 0.018.
- **drawer 001.** link6, link7 and the hand touched the cabinet in routed transits and recovery moves (414 samples).
- **Diagnostic counterfactual** (`scripts/truth_sweep.py`; truth is used for diagnosis only). With the *true* bottle box (`inputs/layout_manifest.json collision_bounds`) plus the model's 0.018 m clearance:
  - the grip-site point check still **passes** the T08 stage that collided;
  - a palm+finger box sweep built from the `robot_facts_v2` envelope, with orientation interpolated, **flags** it.

  So point-only routing fails even with perfect geometry.
- **Cost of a naive fix** (`scripts/hand_sweep.py`). The same hand sweep against the model's *own* boxes would have refused 21 of the 110 stages the router accepted (29/110 with a nominal 8×8×7 cm held box). Of those 21 hits, 14 occur at the start pose, 4 at the goal and 3 mid-path. With the model's T08 boxes it flags 0 stages, so a swept check alone does not help without better geometry (F2).

**Improvement.**
- Replace the point check with a **body-swept check** on every arm-motion stage, transit and contact alike, including lifts, descents and retreats. It should cover:
  - hand and finger boxes at the interpolated orientation;
  - the held object's box, attached at grasp;
  - arm-link hulls via forward kinematics from measured joints.
- Reuse existing code: `foundation_harness/robot_geometry.RobotCollisionModel` and `foundation_harness/motion.CandidatePlanner.check_motion` already provide robot-only convex envelopes and FK.
- Run it as a *local planner, not a veto*:
  - **Escape** when the start is inside an inflated region, by retreating along −approach or upward.
  - **Detour** around hits mid-path.
  - Treat a goal touching a box as the handoff point to a contact stage.
  - Return to the model only when no route exists.

  This also matches the user's rule against adding veto checks.

**Expected effect.**
- With accurate geometry, it catches the one routed obstacle collision (3,504 samples) and the link-on-cabinet contacts.
- With escape and goal handling, only about 3/110 stages would need a real detour. As a hard veto it would instead cost roughly 3 extra calls per run (about 40 s).
- Computing it should cost well under 10 ms per stage (an estimate, not measured), against 0.155–0.198 s of non-model time per control step.

**Validation.**
- Offline: replay every recorded stage of the 7 H1 runs and 2 H3 runs against model boxes and against diagnostic truth. Report recall and precision against the actual contact episodes, and the escape and detour rates.
- Live: run the all-task sweep on one frozen build. Pass if there are 0 forbidden episodes in routed stages and the call count is no higher than baseline.

### F2 (critical): the model's boxes miss the added obstacles by 5–12 cm

**Mechanism.** The only geometry source is the LLM's own two-view reasoning ("Choose the destination and relevant obstacles using RGB and camera calibration", `_agent_client.py:139`). There is no support-plane check, no reprojection, and boxes are never shown back to the model.

**Evidence** (`scripts/box_accuracy.py`, compared against `layout_manifest.json` truth, diagnostic only):

| Run | Map step | Clearance | IoU | Centre error (m) | Extra inflation needed to contain truth | Expanded box contains obstacle? |
|---|---|---|---|---|---|---|
| T04 t100 (can) | 0 / 162 / 163 / 347 | .025/.06/.04/.03 | **0.116** (never revised) | (+0.048, 0.000, −0.025) | 0.039 | only at clearance ≥0.04 |
| T04 20mm (can) | 0 | .025 | 0.124 | (+0.048, −0.005, −0.020) | 0.039 | no |
| T04 20mm (can) | 155 (after mug–can contact at 141–144) | .045 | 0.301 | (−0.017, −0.013, −0.020) | 0.008 | yes |
| T08 t100 (bottle) | 0 | .018 | **0.0** | (−0.085, −0.009, −0.083) | 0.083 | no |
| T08 t100 (bottle) | 82 (after the 47-step finger contact) | .018 | 0.196 | (+0.033, −0.014, −0.001) | 0.013 | yes |
| T08 stove-off (bottle) | 0 | .025 | **0.0** | (−0.077, −0.024, −0.093) | 0.076 | no |
| T08 stove-off (bottle) | 161 (after close wrist views) | .025 | 0.352 | (+0.015, −0.011, +0.007) | 0.013 | yes |

- In both T08 runs, every box in the first map had its bottom at z 0.79–0.80 m. The real table top is at 0.90 m.
- The errors run mostly along the external camera's viewing ray (x and z).
- Boxes were corrected only after a contact or after the gripper reached close range. The T04 t100 can box was never corrected.

**Improvement.** Move metric estimation out of the LLM.
- The model names or points at each relevant object in both images. The harness triangulates with the synchronized calibration, or uses a calibrated perception module once it produces usable 3-D centres (H3 produced 0 in its runs), and fits each box onto a support plane.
- Render the current map, and the planned swept volume, as overlays on both images in every request.
- Flag boxes whose bottoms sit far from the estimated support plane.
- Add a short fixed wrist-camera survey before the first `execute`.
- The model labels and edits; it does not estimate metres.

**Expected effect.** After a close-up view the model's own boxes were 0.020–0.037 m off, and their expanded boxes contained the obstacle in 3/3 corrected cases. Reaching that accuracy *before* the first motion would have removed the T08 bottle contact (3,504 samples) and the T04 20mm mug–can contact (33). Overlays add image tokens, estimated at +1–3 s per call; triangulation takes milliseconds.

**Validation.** Offline, measure IoU and centre error on the saved initial observations of the 4 obstacle layouts and the LIBERO-10 resets (diagnostic truth only). Live, pass when 0 contacts with added obstacles occur in the sweep.

### F3 (high): palm-on-object is the largest forbidden class, and the model is never told the palm envelope

**Mechanism.** The executive gets the pad midpoint, pad size and inner gap only (`_agent_client.py:228-233`). I searched T04 t100 request payloads 000001 and 000005 for "palm", "footprint" and "fingertip_reach": 0 hits. `robot_facts_v2.json` already holds the palm offset (31.04 mm), the footprint and the fingertip reach (9.25 mm; `:11`, `:18`, `:440-447`).

The model therefore chooses top-down grasp depths of 4–7 cm below the rim. The unrouted contact approach then puts the palm on the rim.

**Evidence.**
- 10 episodes and 9,739 samples (55%), in 5 of 7 runs.
- In the T04 runs, the commanded grasp depth below the mug rim was 0.051 m (t100, step 70), 0.069 m (t100, step 369) and 0.053 m (default, step 69). The palm offset is 0.031 m. Contact began with the grip site 0.037–0.06 m below the rim (rims 0.546 and 0.552 m, from layout poses plus `physical_truth` heights).
- Contacts lasted up to 99 steps, 73 of them during the lift (drawer 001, steps 2075–2173). drawer 003 needed 3 moka grasps, each with palm contact.
- H3 T04 has the same scene and the same control code. It targeted about 15 mm shallower and had **0** palm contacts.
- The 2026-10-26 analysis (`final_plan.md` §1.5(b)) recommended adding this envelope. It is still missing from H1.

**Improvement.**
- Add the full tool envelope as fixed robot geometry.
- In contact approaches, the model states the estimated top of the feature it is entering. The local controller limits descent to top − palm offset + margin, unless the stage is declared an insertion, and reports the remaining depth.
- Pair this with the guarded stop in F5.

**Expected effect.** This targets the 55% class directly. In the 4 T04 episodes with known rims, a depth bound would have stopped 6–38 mm higher. It costs 0 calls and about 0.3 KB of prompt. It also reduces regrasps.

### F4 (medium): A* never runs; the router refuses instead, made worse by a hidden 0.02 m inflation

**Mechanism.**
- Start/goal validity is tested before any search (`_routing.py:105-108`).
- `FineCompactField._point_valid` needs all 8 corners of the enclosing 0.02 m lattice cell to be free (`fine_distance_planner.py:101-106`). That adds a hidden inflation of up to one cell per axis, and the amount depends on where the grid happens to fall.
- The same rule is applied to the measured pose during tracking (`tracking.py:83-84`).
- One global clearance has to cover "hand/carried-object extent and estimation error" at once (`_agent_client.py:95-96`).

**Evidence.**
- Across all 7 runs, 117 stages attempted routing: 111 were straight, with **0 A* calls and 0 detours**. The H3 runs also had 0 non-straight routes.
- The models avoided obstacles with their own high waypoints instead. Example: T04 t100 step 162, "The elevated transfer avoids sweeping the mug body into the can".
- There were 7 `route_start_blocked` refusals:
  - T04 t100 at step 162;
  - T04 20mm at step 395;
  - T08 t100 at steps 87 and 597;
  - T08 stove-off at step 179;
  - drawer 001 at step 1582;
  - H3 T04 at step 349.

  Each forced a recovery call: 84.3 s in total, about 12 s each.
- There was also 1 tracking `invalid_measured_start` (T04 20mm, step 538, 9.5 s).
- Replaying the frozen `TransitRoute` offline (`scripts/route_replay.py`, transit cap 0.10) confirms the refusals were spurious. The start was outside the nearest expanded box in each case: by 0.020 m (T04, step 162), 0.0063 m (T08, step 87) and 0.0133 m (T08, step 597).
- In T04 t100 the model raised the clearance from 0.025 to 0.06 to cover the carried mug. That blocked the start, and it then lowered the clearance to 0.04: two map updates that changed only the clearance.

**Improvement.**
- Use exact point-to-box distances for start, goal and tracking validity.
- Never refuse because the current pose is inside an inflated region: escape first, then route.
- Use per-object margins plus an attached held volume instead of one global clearance (F7).
- Report the effective clearance.

**Expected effect.** About 1 recovery call fewer per obstacle task (8 events across 8 runs, roughly 85–100 s in total). The router, not the model, picks the detours.

### F5 (high): the robot never senses contact; the only proxy is a late stall check that rotation can hide

**Mechanism.**
- The contact monitor is record-only, and feedback always says "cause unknown".
- The stall predicate needs at least 16 steps and less than 0.8 mm of improvement over 12 steps, measured on `route_remaining + 0.05·orientation_error` (`_fragments.py:404-419`).
- So a translation blocked by contact goes unflagged while the wrist is still rotating. An OSC push that still reaches the target is never flagged at all.

**Evidence.**
- **Episode length.** Median 13 steps, maximum 99.
- **T08 t100.** Translation was pinned at z ≈ 1.049 m for 45 steps while the jaw axis rotated from (0.72, −0.69) to (1.0, −0.01). The stall fired only after the rotation finished.
- **T04 t100.** The approach reported `pose_arrived` although the palm had been on the rim since step 114.
- **Stall handoffs.** There were 46. The 44 model calls they triggered took 896 s (`scripts/stall_causes.py`, `scripts/failure_calls.py`). The handoffs break down as:
  - 14 with forbidden contact;
  - 8 with allowed contact;
  - 10 near the target (under 10 mm);
  - 14 with no contact and far from the target.
- **drawer 001.** The model blamed a "near-singular arm configuration" and spent 9 recovery calls on it (steps 1156–1275, 142.8 s). Meanwhile its recovery moves pushed link6, link7 and the hand into the cabinet.
- **Offline counterfactual** using proprioception only (`scripts/guarded_abs.py`, `scripts/guard_vs_stall.py`). The rule: at least 3 consecutive steps with at least 2 mm commanded and less than 0.3 mm of progress along the command.
  - It fires inside 18 of the 29 episodes, a median of 4 steps after onset.
  - 12,610 of the 17,682 samples come after its fire point.
  - It fires inside all 46 stalled segments, a median of 8 steps before the recorded stall (607 steps, about 111 s).
  - It fires falsely on 217 of 4,057 contact-free moving steps: 194 of these are in the drawer runs, and 121 precede an existing stall by 15 steps or less.

**Improvement.**
- Add **guarded motion** to every arm stage: measure translation progress separately from rotation.
- On blockage, hold or back off 5 mm and hand off `blocked_motion` with:
  - the blocked direction;
  - the depth reached;
  - the gripper state;
  - the robot part nearest each box, via FK.
- Contact stages declare the contact they expect, so a guarded stop counts as arrival when that contact is reached.
- On a real robot, use external-torque estimates where the robot provides them.

**Expected effect (upper bound).** About 70% less pressing time, about 110 s of wasted steps saved across 7 runs, and fewer misdiagnosed recovery loops. The cost is about 5% false fires, mostly near joint limits.

### F6 (medium): rotations are never collision-checked, and the 1.2 rad veto forces split rotations

**Mechanism.** The route covers translation only. Stage entry rejects rotations over 1.2 rad (`_fragments.py:28`, `:288-291`). The model then splits the rotation, and the intermediate pose is never checked.

**Evidence.**
- The first decision in both T04 t100 and T08 t100 was rejected at 1.572 and 1.569 rad, wasting about 30 s of calls each.
- In T08 the split (45° and then 90°) put open `finger1` at about (0.007, −0.054) over the bottle's top face at contact onset (bottle y −0.090…−0.053, top 1.046). That is the 3,504-sample contact.
- The prior uc7 analysis recorded the same veto in t6_011.

**Improvement.** Remove the rotation veto. Let the local controller split rotations internally, and include every interpolated orientation in the swept check (F1).

**Expected effect.** One fewer call (about 30 s) where the first grasp needs more than 1.2 rad (2/7 runs here), and no more rotation-induced finger sweeps.

### F7 (medium): held objects are not modelled, so carry safety depends on the model lifting high

**Mechanism.** The only provision for a held object is the global clearance. The map does "not infer object motion from gripper commands" (`_routing.py:28-29`). Lifts out of containers and place descents are unrouted.

**Evidence.**
- **Offline replay.** The router accepts the T04 t100 carry at z = 0.56 m. At that height the mug (grasped about 0.06 m above its base) hangs below the can top (0.513 m). The replay refuses only z ≤ 0.54, because the goal is blocked.
- **What the models did instead.** They carried at 0.62–0.65 m in T04 and 1.22–1.25 m in T08.
- **Cost of lifting high.** Vertical travel is 39–53% of the grip-site path in all 7 runs.
- **Held-object contacts during unrouted lifts.**
  - bowl vs cabinet: drawer 001 (697 samples) and drawer 003 (611);
  - pot vs bottle: T08 (111).

**Improvement.** At grasp, attach the object's pre-grasp box, or a perception box, to the grip frame. Sweep it during lift, carry and place, and let the router choose the minimum safe height and lift-out direction.

**Expected effect.** Carry safety no longer depends on the model's altitude choice. Estimated 20–40% less vertical transit, about 5–25 s per run.

### F8 (medium): grasp closing and placement run open-loop to a commanded depth

**Mechanism.** `set_gripper` closes without checking neighbours. Placement descends to a fixed depth and then opens, with no support-contact signal and no check that the object is seated.

**Evidence.**
- 11 `manipulated_object_scene` episodes (3,313 samples) in 5 of 7 H1 runs, plus 2 in H3 T08 (400 samples):
  - mug pushed into the can during closing;
  - pot pushed into the bottle during closing;
  - bowl vs cabinet on extraction;
  - pot vs burner rim during descent and release.
- T08 stove-off step 564, "Avoids forcing the stalled descent": the rim was reported only through a stall.

**Improvement.**
- Guarded place descent until measured support contact, then release and retreat along the approach axis.
- Before closing, check the jaw sweep and the likely push direction against the map.
- Use support geometry from perception.

**Expected effect.** Pressing on rims shrinks to a few steps. Removing these episodes entirely still needs xy placement precision (F2).

### F9 (medium): the map is an unlabelled, capped, unvalidated scratchpad, and plans ignore added obstacles

**Mechanism and evidence.**
- Each update replaces the whole map. It has no IDs and no per-box uncertainty or source, and it is frozen for the length of a fragment.
- The 4-box cap was hit in 10 of 36 updates. Clearances ranged from 0.012 to 0.08 m.
- The T08 boxes that sat below the table were accepted without any warning.
- The reviewed plan lists the can or bottle as a part or constraint in **0 of 4** obstacle-task plans.
- The keepout machinery in `_constraints.py` sits idle: 0 macro actions in any run.

**Improvement.** A harness-maintained, object-centric map:
- IDs taken from the plan's parts plus "other obstacles";
- source, uncertainty and timestamp for each object;
- automatic attach and detach on grasp and release;
- plausibility checks;
- edits by ID instead of full replacement;
- a plan review that lists every non-target object.

### F10 (high, for speed): the slow contact profile takes most actions but does not make motion safer

**Mechanism.** The contact profile caps commands at 5 mm per step. The OSC realizes about 24% of each command per step: the prior analysis measured this, and onset ratios here were 0.23–0.27. Actual motion is therefore 1.2–1.4 mm per action.

**Evidence.**
- Contact-profile actions are **57–67%** of all actions in every run (311–1,424 per run). They move 1.2–1.4 mm per action, against 4.5–14.9 mm in transit.
- Non-model time is 0.155–0.198 s per action, so contact stages cost about 49 s (T04 t100) up to about 265 s (drawer 001).
- 25 of the 29 forbidden episodes happened in these slow stages.
- Between 172 and 858 contact steps per run were still more than 20 mm from their own target. At each run's measured transit speed they would have needed 30–146 steps, so the upper-bound saving is **22–141 s per run** (`scripts/contact_speed.py`).

**Improvement.** A guarded fast approach: transit speed down to a standoff set by the swept check (larger than the geometry uncertainty), then a guarded contact segment. Keep 5 mm only inside the standoff.

**Expected effect.** 22–52 s saved on T04/T08-type tasks and 128–141 s on the drawer task. This is only safe together with F5 and a standoff.

---

## 4. Prior findings re-checked against H1

| Prior finding | Status in H1 |
|---|---|
| Tool envelope missing: pads only (2026-10-26) | **Still present** (`_agent_client.py:228-233`); it is now the largest forbidden class (F3). |
| Contact profile ~24% tracking, 48.5% of motion steps (uc7) | **Worse share**: 57–67% of actions, because transit got faster. |
| Stall cause never recorded (`contact_or_collision_cause: unknown`) | **Still present** (`_fragments.py:482`), now with 14 contact-caused stalls (F5). |
| Rotation-envelope rejections (t6_011) | **Still present**: first call in T04 t100 and T08 t100 (F6). |
| Table-height estimates 17–42 mm low (v3) | H1 T08 boxes were about 100 mm low (F2). |
| Direction words / viewer frame | Fixed in uc7.1 (`region_directions` plus prompt text). Not a collision factor here. |
| Mirrored policy images | Not in H1: the executive uses `view.*_rgb_upright` and `encode_png_data_url` (`_agent_client.py:559-563`). |
| All-zero rotation keeps a stale OSC goal | Not seen: `ActionAdapter` sends a nonzero rotation unless the error is below 1e-12 (`action.py:86-91`). |

## 5. Suggested sweep metrics for collision work

- Report contiguous forbidden **episodes** per task, broken down by phase and class, with their durations.
- Count route refusals and escapes, guarded stops (true vs false), and recovery calls that follow a stall or blocked motion.
- Report box IoU, using truth only as an offline diagnostic.
- Report contact-profile action share and seconds per task.

## 6. Not covered

- Contact positions after the first event of each class are not saved, so the exact link6/cabinet locations could not be checked.
- drawer 002 (`initial_plan_failed`) was not analysed.
- No videos were reviewed.
- H2 internals were read only at the level of their function lists.
- The guarded-motion and hand-sweep results are log-replay counterfactuals: they do not simulate what would have happened after a stop.
