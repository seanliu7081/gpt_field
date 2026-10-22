# agent_SpeedUp_v2 specification: relation plans, measured contact, observed completion

Date: 2026-10-18, revision 2 (after the critic review; the first version is kept at `scratchpad/agent_SpeedUp_v2_spec.rev1.md`). Status: specification only. Nothing under `/workspace/gpt_field` was written and no model call was made.

Computations behind this revision (all diagnosis only, zero model calls):
- The critic's two simulator probes (`CP`, below) were re-read in full.
- One new simulator probe on libero_10 task 2 (`SP`, below): knob twists from a fresh close in both sense orders, and a closed empty hand yawing in free air.
- An `NCCTrack` versus `cv2.matchTemplate` timing on `SURVEY/tasks/frames/t2_wrist.png`, and a count of the grid patches that pass `NCCTrack`'s texture test.
- Compact-JSON lengths of the example plans.
- The first version's palm-hull computation (§4.4 R6).


> **User decisions (2026-10-18).** Rulings **R-1v granted**, **R-2f granted**, **R-5b granted**. Build order: **task-2 slice first**. That means M0, M1 for task-2 parts, M2 lever and loop grasps, M3 place_on, M4 turn, M5, and a minimal M6, gated on a zero-call task-2 chain before the user's first live run. The bowl_plate and drawer_bowl regressions follow after that. The eval protocol stays as in v1, with no speed-item protocol changes (§6.8 deferred).


> **Scope rule (user, 2026-10-20).** v2 is ONE general framework. There are no per-task skills or harness. One frozen code version and one threshold table run every task. No change is justified by a single task. Failures are framework gaps that get general fixes. The headline metric is the number of LIBERO-10 tasks completed by the same code. Order:
> 1. finish the general grasp-robustness round;
> 2. run all 10 LIBERO-10 tasks with the same frozen code;
> 3. build only the missing general relations (place_in, push_to/slide) and re-run all 10.

## 0. Conventions

**Labels.**
- **[F]** fact: read in code (`file:line`; paths are relative to `/workspace/gpt_field`, with `src/gpt_field/` omitted for modules), or measured in a recorded run, survey probe, oracle run or the probes listed below (file cited).
- **[I]** interpretation or design judgement.
- **[P]** projection, not measured.
- **[Q]** needs a user ruling or a milestone measurement before it is relied on.

**Threshold classes** (§5):
- **ROBOT**: a robot asset fact or a ruling fact.
- **ONLINE**: measured in this episode by the robot's own motion or sensing.
- **DD**: a labelled design default. It is k × an ONLINE quantity, or a floor justified from robot-only traces.
- **RULING**: set by a user ruling.

There is no TUNED-evaluator class; R-3 stays refused.

**Evidence shorthand.**
- `SURVEY/` = `/tmp/claude-0/-workspace/2e071f26-a38c-4a1c-b40e-bd363cbcfe54/impl/v2/survey`.
- `PA[k]` = element k of `SURVEY/phase_a_results.json`. [F] The file order is: **PA[0]** LIBERO-10 task survey, **PA[1]** eval-protocol infrastructure, **PA[2]** predicate groundings and truth checker.
- `D003` = the drawer run-003 analysis, `wf_43adc074-537/journal.jsonl`. Its result entries are at 0-based lines 3 (truth lens), 4 (audit lens) and 6 (skeptic).
- `RECIPE` = the critic-checked v1 skill analysis, `wf_416060a7-821/journal.jsonl`, result entries at 0-based lines 8 and 10.
- `CP` = the critic's probes, `/tmp/claude-0/-workspace/2e071f26-a38c-4a1c-b40e-bd363cbcfe54/scratchpad/critic_probe/knob_twist_probe.json` and `pot_lower_probe.json` (scripts beside them).
- `SP` = this revision's probe, `/tmp/claude-0/-workspace/2e071f26-a38c-4a1c-b40e-bd363cbcfe54/scratchpad/spec_rev_probe/knob_sense_probe.json` (script `knob_sense_probe.py`, run with `/workspace/venvs/libero-plus/bin/python`, `MUJOCO_EGL_DEVICE_ID=5`, 128 px renders).
- `RC`, `VS`, `MP` = the three input designs under `impl/v2/design/`.
- `GF`, `FR`, `ES` = the judges' scorecards.
- `spec v1` = `agent_SpeedUp_v1.md`.
- Appendix A = the critic findings, applied or rejected.

**Image convention.** v1's upright images: u right, v down (`contact_skills.py:259-303`). Wrist image right = R1, up = R0, principal point at the centre (`wrist_servo.py:56-66`).

---

## 1. Goal and acceptance criteria

### 1.1 Goal

The user's goal: *"Using the same robot arm and RGB image and robot-state inputs, there is no longer a need to modify control code or thresholds for each individual manipulation task when changing objects, layouts, or task combinations."*

Operationally, the interface changes from "a program that invokes an object" to "establish a specific physical relation". Each task becomes a short plan of relation steps. Each step names:
1. the object and the visible part that is manipulated (by text and pointing);
2. the physical relation to establish (held, turned, slid open, supported on, inside, pushed to);
3. the constraints kept throughout (moves with the hand, stays upright, stays clear of a part, relations already established stay true);
4. the observable criteria that track progress and decide completion.

The executor turns every step into motion using only robot facts and quantities measured in the episode. In particular it controls the **object–reference relation**, not hand coordinates. In "place a cup onto a plate", it measures where the cup sits in the hand and where the plate is, aims the cup's footprint at the plate, and accepts the release only when the cup is supported at the plate's measured height.

**Inputs (unchanged zero-shot rules).**
- Upright agentview and wrist RGB at 512 px, state8 (grip xyz, rotvec, two finger joints), and the task sentence.
- Rulings in force: R-1 (online image Jacobian), R-2 (wrist mount facts), R-1x (external-image axes from own probe moves), and R-5 (robot base, verified for the tabletop/kitchen MountedPanda mount (−0.66, 0, 0.912), gated by §5.3).
- R-3 (evaluator-tuned constants) and R-4 (continuation implies incomplete) stay refused.
- No BDDL or privileged geometry is used at run time. Evaluator truth is used only in diagnosis and tests.

### 1.2 Acceptance criteria

| ID | Criterion | How it is measured |
|---|---|---|
| AC-1 | **Same code, same thresholds.** One `frozen_source`, one constants table and one instruction text serve every task. | (a) The manifest's `constants_sha256` and `instructions_sha256` are identical for every registry key (a test builds the manifest for all keys). (b) The policy and client constructors take no task key. The client's only task input is the episode's official sentence, which the runner reads from the reset observation and checks against `list_tasks()`. A test asserts the payload task text equals that sentence for every key and that v1's bowl default never reaches a non-bowl task (§4.2). (c) The T0 lint passes: every numeric module-level constant in a v2 control module has a `CONSTANT_LABELS` entry (ROBOT / ONLINE / DD / RULING with its derivation), and no token from the lint vocabulary (§5.1) appears in a v2 control module. |
| AC-2 | **Task success = raw LIBERO success AND strict physical truth.** | Raw LIBERO success stays the only completion signal of the episode (`stop_on_success` True). After finalization, the diagnosis-only truth check (PA[2] strict verdicts, run through the PA[1] worker ops) must report every BDDL goal atom satisfied **at the terminal state**. When the only failing sub-check at the terminal state is "held" (LIBERO fired before the release), the atoms must be satisfied **after the release-and-settle counterfactual**. Both are reported. A run counts as a v2 success only when raw and physical are both true. A raw success with a failed physical check (the drawer-003 perch) is reported as `lenient_success` and counts as a v2 failure. |
| AC-3 | **Tasks.** libero_10 task 2 ("turn on the stove and put the moka pot on it", episode 0) is the primary task. v1's `bowl_plate` (libero_spatial 0) and `drawer_bowl` (libero_goal 3) are regressions. libero_10 task 7 is the backup. | Before the first live run, zero-model-call scripted chains (ORACLE pointing stub, test-only) succeed under AC-2 on the primary task and both regression tasks (gate G7, §8.2). Live runs are started by the user; the first is task 2. |
| AC-4 | **Speed targets** (nominal = no repair; ceiling = at most 2 repairs). | From `summary.json`: model calls, `model_pipeline_s`, output tokens, steps, `control_without_video_s`. Table below. |
| AC-5 | **The model never outputs a number, direction or completion claim.** | Schema-level test over all v2 schemas: no numeric field except grid cell or point ids (0–64); no direction enum; no "done" field. The review answer `hold_still` is logged as an unverified model claim; it never ends the episode and never overrides an observer (§3.7). |
| AC-6 | **Diagnosability.** | Every outcome record carries its reason code, the measured evidence and every threshold it was compared against, with label and derivation (§4.3 `Threshold`). An observer-verdict-versus-truth confusion table is produced offline for every scripted and live run (§6.5). |

**Speed targets** [P targets; v1 values are F, from the runs' `summary.json`; re-budgeted in this revision for Calibrate+ ≈ 200 steps, dense-JSON tokenization and plan-call reasoning, §3.8]:

| Task | v1 reference run | v2 nominal target | v2 ceiling |
|---|---|---|---|
| libero_10 task 2 | — (v1 cannot run it) | ≤ 4 calls, ≤ 50 s model time, ≤ 900 steps, `control_without_video_s` ≤ 160 s | ≤ 8 calls, ≤ 100 s, ≤ 1,300 steps, ≤ 260 s |
| drawer_bowl | run 003: 21 calls, 132.2 s, 2,998 output tokens, 967 steps, 234.5 s | ≤ 7 calls, ≤ 60 s, ≤ 1,200 steps, ≤ 200 s | ≤ 11 calls, ≤ 110 s, ≤ 1,650 steps, ≤ 300 s |
| bowl_plate | bowl run 002: 11 calls, 81.6 s, 1,798 output tokens, 449 steps, 135.4 s | ≤ 3 calls, ≤ 30 s, ≤ 650 steps, ≤ 110 s | ≤ 7 calls, ≤ 70 s, ≤ 1,000 steps, ≤ 180 s |

[F] The simulator costs about 0.098–0.111 s per step with two 512 px renders (D003 audit 97.6 ms/step; bowl 002: 49.8 s for 449 steps). The generic monitor adds about 5 ms per step (PA[1]).

[I] v2 moves time from the model to the simulator: probes (Calibrate+, parallax probes, sense and DOF probes, rest checks) replace model-driven motion decisions. [P] On bowl_plate v2 is expected to cost about as much `control_without_video_s` as v1 (fewer model seconds, more steps); the gain is in robustness, not speed. Missing a speed target is analysed; it is not a task failure.

---

## 2. Decision and rationale

### 2.1 Judge scores

| Design | goal_fit (GF) | feasibility_risk (FR) | eval_speed (ES) | Mean | Ranks (GF / FR / ES) |
|---|---|---|---|---|---|
| relation_constraints (RC) | 7.5 | 7.0 | 7.4 | **7.30** | 1 / 1 / 1 |
| model_plans_executor_observes (MP) | 7.4 | 6.2 | 7.2 | 6.93 | 2 / 2 / 3 |
| visual_servoing (VS) | 7.2 | 5.7 | 7.3 | 6.73 | 3 / 3 / 2 |

### 2.2 Backbone: relation_constraints

The backbone is RC. The choice rests on evidence, not only on the unanimous first rank:
- **Verified claims.** FR checked 14 RC code and evidence claims and found none wrong (one slip: the pot hang, §2.5).
- **Task-2 contact analysis, corrected.** RC was right that a palm stop on the overhanging moka handle would tip the free pot (rule P13). Its knob analysis was wrong, and FR's confirmation of it was wrong too.
  - RC took the 34 mm half-size of the knob's central collision box (`heading_flow_pub/third_party/LIBERO/libero/libero/assets/articulated_objects/flat_stove.xml:41`) as the disk radius.
  - [F] Ring slabs extend the disk to about 47.6 mm (`flat_stove.xml:42-54`). The button's collision AABB is 9.5 cm across (`SURVEY/tasks/t2_dump.json`, `per_body_col_aabb`).
  - [F] So a full-opening fingertip landing does land on the disk: a CP descent stalled at grip 0.9272, with the fingertips on the 0.9178 disk top.
  - VS and MP had planned that landing. The first version's rejection of it is withdrawn (Appendix A, R1).
- **Run-003 perch.** Three independent barriers: measured aim, a drift invariant, and a release gated on support height against the measured floor and rim. The floor is surveyed empty-handed (GF 9; FR: "alone catches the perch").
- **Structure.**
  - A RelationType library: preconditions, controller, every-step invariants and code-owned completion.
  - `stays`/`already` semantics.
  - A name-free GraspPlanner, from which v1's two strategies fall out.
  - A `Threshold{label, derivation}` record in every outcome (GF 8; ES 9 for debuggability).
- **Cheapest failure loop**: a ≤ 80-token repair call (ES).

RC's weaknesses, which this spec fixes by grafting:
- its generality rested on an untested dense parallax depth map with no working fallback for the knob and loop (GF 5.5, FR 5);
- batched start-view fine pointing (FR, ES);
- per-arena mount constants (GF);
- optimistic token and effort numbers (ES, FR).

### 2.3 Grafted ideas

| Idea | From | Why (judge evidence) | Where here |
|---|---|---|---|
| A plan with fixed enum part ids A–H and inline constructs only (no `$ref`). It carries a `move` mechanism prior; step-level `until` criteria `effect(p, txt)`, `stop`, `fits(p)`, `quarter_turn`/`half_turn`; constraint `clear_of(p)` with an operand; and already-true relations. | MP | GF 8.5 expressiveness; the schema checks references for free; v1-proven constructs (`agent_speedup_v1_client.py:65-112`) | §3.3 |
| Coarse cells for every part inside the plan; **hover-time single fine pointing** per contact part, with the crop centred on the predicted pixel | MP | Live-proven protocol (D003: 2/2 handle pointings correct, about 3.9 s per pointing call). RC's start-view re-find would need a 2–3× template scale change, beyond `REACQUIRE_SCALES` max 1.25 (`wrist_servo.py:130`) (FR). | §3.6 |
| Geometric checks on pointing answers (with the "surroundings" defined, §3.6) | MP | FR, ES | §3.6 |
| In-hand offset re-measured after the close and before the lift; Touchdown measuring the lowest held point by touch; four-way SupportClassify with an `ambiguous` class and an empty-hand floor touch probe | MP | GF 9 perch; captures D003's 2.6 cm close drag | §4.7, §4.9 |
| BDDL-like plan rendering; `parse_check.json`; V5 distributions of robot-only quantities; ORACLE stub asserted unavailable in live mode; enumerated trace events | MP | ES | §6, §8 |
| **Sparse κ-at-feature depth** (multi-feature probe, v1 machinery) as the primary depth source; looming as an f-free cross-check; touch heights overriding; dense flow only as an optional layer | VS | GF, FR and ES all ask for a fallback that does not collapse to v1's per-strategy defaults | §4.5 |
| 2×3 external Jacobian with a vertical probe, and the n̂/v̂ parallax decomposition (the v̂ row only with a measured height difference) | VS | Removes D003's external-image parallax aiming at its root (FR) | §4.5, §4.9 |
| Width-consistency check at the close; post-close mobility (DOF) probe, now decided by part observables (§4.9) | VS | FR: "catches pinching the wrong thing"; DOF measured, not named | §4.6, §4.9 |
| **T0 constants lint** (labelled numbers, no object or task vocabulary in control modules) | VS | GF, ES: makes "no per-task code" enforceable | §5.1 |
| Effect threshold normalized by the false-change rate measured in that region during Calibrate+ | VS | GF | §5.2 O1 |
| Zero-call re-acquisition by pose prediction with rotated and κ-scaled templates | VS | GF | §3.6 |
| Lateral error live during lowering; `lateral_unreachable` after 4 saturated steps (v1 `BLOCKED_STEPS`, `wrist_servo.py` Align) | VS | ES | §4.9 |
| `FakeTwoViewSim` (two cameras, held offset and tilt, hinge with colour change and lever backlash, drawer with stops) | VS | ES | §8.3 |

### 2.4 Rejected ideas

| Idea | From | Reason |
|---|---|---|
| Model-chosen approach side `from` (ext_left / ext_right / ext_near / ext_far) | VS | A model directional judgement with no measured fallback. v1's world-direction choice was wrong in 3 of 6 pull checks (spec v1 §15 item 4) (GF, FR, ES). Replaced by the measured `mount_side`. |
| A decision loop of up to 3 chained procedural steps as the whole interface | VS | No declarative constraints, no already-true relations, no keep-out for distractors (GF 6.5). |
| Nested `$defs`/`$ref` strict schema | VS | Untested on the subscription endpoint; a rejection would first appear in a user live run (ES, FR). |
| Palm-on-loop depth rule for the moka handle; pad window `z_grip − 11.6 ≤ z_top` | VS | Physically wrong (FR; §4.4). The fingertip landing on the knob disk at full opening, rejected in the first version, is reinstated (Appendix A, R1). |
| Signed world vectors in model-facing evidence (`in_hand_offset_mm [0,37,-37]`) | VS | They invite the direction reasoning v1 mis-signed (median command error 153.5°, spec v1 §17.1, `agent_SpeedUp_v1.md:628`) (ES). Model evidence carries magnitudes and named relations; vectors stay in the trace. |
| Surveying the destination floor while the held object blocks the wrist view | VS | v1's own prompt says the wrist view is mostly blocked while holding (`agent_speedup_v1_client.py:181-182`) (FR). Destinations are registered empty-handed (§4.8). |
| `verify` call: the model judges whether an effect happened | MP | A bounded model-completion channel at step level (GF). Effects are decided by the observer; an unseen effect falls back to rotation, and that is reported. |
| Full replan as the only escalation | MP | Costs 15–30 s per failure (ES, §3.8). Kept as one repair action; the default repair is ≤ 80 tokens. |
| Floorless lateral tolerance and drift guard; footprint-inside test shrunk by σ_offset | MP | At the run-003 opening (1.6 mm geometric clearance in y) it would never release (FR). Floors added (M4, M5) and the `inside` rule made axis-specific (§4.8). |
| Batched start-view fine pointing; dense parallax depth as the primary path | RC | §2.3. Dense flow is kept as an optional layer whose failure only makes plans more conservative (§4.5). |
| Per-arena robot-mount table (R-5 option a) | RC | A scene-keyed constant registry (GF). Replaced by the start-pose consistency rule (§5.3). |
| `side` words (`left_of`, `in_front_of`) | RC | Replaced by a pointed spot part, so the model uses fewer direction words. |
| Pre-shaped apertures (`ApertureSet`, `ZeroGripAdapter`) on the critical path | RC/FR (first version) | No target task needs them once the knob landing is correct (Appendix A, MUST 1). Deferred (§9.3). |
| Hand realized/commanded yaw or translation ratios as the turn-sense, end-stop or DOF decision | first version (K1, K4) | [F, SP] With the knob gripped, the hand yaws −1.08…−1.17°/step in the blocked sense and +1.16…+1.25°/step in the free sense (cap 0.1 rad). Hand-only motion is not part motion. Replaced by part observables (§4.9). |

### 2.5 Flaws found by the judges and the critic, and how they are fixed

| Flaw | Found by | Fix |
|---|---|---|
| Pad-span sign (VS gate, VS/MP walk-throughs) | FR | One convention (§4.4): pads span −11.6…+4.4 mm along the approach about the grip site. Top-down, the pad world z ∈ [g − 4.4, g + 11.6] mm. |
| Fingertip reach (11.6 / 12.5 / 9.2 mm) | FR, ES, critic | **Pinned at 9.3 mm** (R5): finger STL hull 9.2 mm; simulated stall 9.4 mm (CP). The 12.5 mm AABB value is not used; it would bias every touch height by about 3 mm. |
| Palm underside (26 / 31 / 39–45 mm) | RC, GF, FR, critic | **Pinned at 31.0 mm** (R6), the collision hull MuJoCo actually collides with (critic: 31.04 mm). The 26.1 mm geom-AABB bound and RECIPE's 39–45 mm visual mesh are not used. |
| Knob disk radius taken as 34 mm, so the full-opening landing was rejected | critic (R1) | Radius ≈ 47.6 mm, centre 4.6 mm from the hinge. The full-opening landing on the ring is the first-ranked plan, with `height_stop` second (§7.1). P4 accepts a landing on the highest surface in a finger column that straddles an edge (P16). |
| Turn sense and end stop from hand yaw ratios | critic (R2), SP | Wedge observer on the signed finger-gap rate, plus wrist-view part rotation, plus the effect (§4.9 Turn; K1w, K4w). G4 includes both sense orders and a deliberately wrong first sense. |
| `inside` precondition "opening ≥ extent + 2·max(5 mm, σ_aim)" can never hold in the run-003 drawer (1.6 mm clearance) | critic (R3) | Axis-specific fit rule with compliant entry on tight axes, validated on D003's counterfactual drops (§4.8, M18). |
| LookAt assumed the part lies under the hand once the hand covers it in the external view | critic (R4) | The part lies about Δh·cot 39° beyond the hand along the camera's viewing direction. Replaced by HandLocator + RayServo + RayDescent (§3.6 item 5, §4.9). |
| The client would be built with v1's bowl prompt | critic (R5) | The client is constructed with the episode's official sentence; tested (§4.2). |
| Palm stop on a free object | RC, FR | Forbidden by rule P13. |
| 63 mm soup can vs 66 mm pot body: a knife-edge width rule | GF, FR, critic | Three-way width rule (P2) plus a finger-descent-column check (P16). The pot body is refused by the column check along the spout–handle axis, and is a feasible-or-marginal candidate across it (§7.1 step 12). |
| Tolerances and drift guard without floors (MP) | FR | Floors on every lateral tolerance (M4, M5). |
| Destination surveyed while holding (VS) | FR, GF | Empty-hand destination registration is a precondition of every place (§4.8). |
| Pot hang 0.12–0.14 m (RC) | FR | b_low = g_close − support z at pick: [F] 0.1219 m in CP (close at 1.0219); 0.115 m in the oracle (close at 1.0146). |
| Token budgets | ES, critic | Restated at 2.5–3.0 compact-JSON characters per token, plus medium-effort reasoning (§3.8). |
| Plate zone "±2 cm" (MP) | GF | [F] The goal-5 zone is ±4 cm (zone half-size = half the BDDL range; PA[2] On(table zone)). |
| Burner effect visibility conflict | RC, VS, MP | [F] `SURVEY/tasks/stove_vis_check.py:17-22` sets the joint and steps 3 times: red fraction 0.208 / 0.201 at q 0.6, 0.0 at 0.49 (`probes/stove_vis_t2.json`). [I] PA[2]'s "not visible" probe changed the joint without stepping. M0 re-measures under a real scripted twist; the effect stays secondary to the part observables. |
| PNG level 1 "protocol-neutral" (MP) | ES | [F] The reset hash is the sha256 of `obs.to_dict()`, which contains the PNG data URLs (`scripts/evaluate_agent_speedup_v1.py:999-1001`). Deferred (§6.8). |
| D7 × P7 rejects the handle grasp once σ_h ≳ 2.2 mm | critic | P7 excludes the grasped part and the finger slot. Unknown cells beside a fitted support plane with the plane's appearance are filled from the plane (D7). |
| NCC compute and Calibrate+ step count not in the speed model | critic | v2 FeatureTracker on `cv2.matchTemplate`; Calibrate+ trimmed and re-budgeted (§4.5, §3.8). |
| Effort (41 developer-days) | critic | Restated as about 60–80 developer-days (§8.4). |

---

## 3. Model interface

### 3.1 Division of labour

| The model decides (semantics) | The executor measures and decides (physics) |
|---|---|
| Which objects and visible parts matter; their text descriptions; one coarse cell per image per part | Where each part is in 3-D: its height, width per jaw yaw, elongation, circular footprint, cavity floor and rim, mounting side |
| Which relation steps to establish and in what order; the preferred grip part(s) | Grasp family, jaw yaw, aperture, depth rule, hover, approach, clearance retreat |
| Constraints to keep (`upright`, `clear_of`) and relations it claims are already true | Keep-out volumes, carry height and path, the in-hand pose of the held object |
| The completion criteria per step (`effect` of a named part, `stop`, `fits` a named part, quarter or half turn) | Turn sense, hinge axis, slide axis, travel, when to stop, whether support and release are valid |
| After a failure: one repair action from a closed list | Local retries before any repair; every measurement in the repair request |

**The model never emits:** coordinates, offsets, waypoints, heights, travel lengths, angles (except the qualitative quarter or half turn), directions, approach axes, jaw choices, gripper commands, strategy names, or completion claims.

[F] Each of v1's model-emitted metric parameters (`waypoint_offset_m`, `more_waypoints_m`, `approach`, `side`, `max_travel_m`; `agent_speedup_v1_client.py:90-106`) was behind a recorded live failure:
- mis-signed external directions (spec v1 §17.1);
- "low in the wrist view" wrong 5 of 8 times (`agent_SpeedUp_v1.md:688`);
- the +7.0 / +2.75 cm parallax overshoot in drawer 003 (D003);
- a layout-specific pull window in drawer 002 (spec v1 §17.3).

### 3.2 Request kinds

| Kind | Sent when | Images and text | Output schema | Visible-token budget [P] | Latency [P] |
|---|---|---|---|---|---|
| `plan` | Once, in the first `act()` after Calibrate+ (§4.2) | External clean + 8×8 grid; wrist (frame-aligned, §3.6) clean + grid; the official task sentence | §3.3 | 160–340 (examples: 834 chars → 280–335; 741 → 250–300) | 15–32 s with medium reasoning (§3.8) |
| `point_fine` | (a) At the executor's hover above a contact part, before its grasp. (b) After a failed answer check (one re-ask). (c) In the external image, for a part with `w` = 0, before LookAt (§3.6 item 5) | 128 px crop ×4 with 8×8 numbered points, centred on the part's predicted pixel (wrist, aligned) or on its plan cell (external); the part's `text` | `{"point_id": 0..64}` | 3–6 | 3–8 s |
| `point_coarse` | A needed part has no wrist cell (`w` = 0) after LookAt, or a mechanism step exposed it; or `point_fine` answered 0 | Full aligned wrist image with the 8×8 grid and, after LookAt, the predicted ray hit marked; the part's `text` | `{"cell_id": 0..64}` | 3–6 | 3–8 s |
| `repair` | A step failed after its local retries; or every step is done, every observer reports its relation holding, and the episode continues (review, at most 2 per episode) | Current images; the plan rendered compactly with step statuses; the last outcome (reason code, evidence text ≤ 200 chars in cm and degrees, thresholds with names and values); holding state | §3.7 | 25–80 (≤ 300 if it carries a new step) | 5–12 s |
| `replan` | Only when a repair answer is `replan` | As `plan`, plus the status block | §3.3 | 160–340 | 15–32 s |

**Call rules.**
- At most 2 model calls per `act()` (v1 `MAX_DECISIONS_PER_ACT = 2`, `agent_speedup_v1.py:37`).
- No model call inside a primitive or controller chain; pointing is a precondition satisfied at phase boundaries. [F] v1's `Align` defaults to `allow_pointing=True` and calls the pointer on a lost mark (`wrist_servo.py:531`, `:621`, `:796`). v2 always constructs it with `allow_pointing=False`; a lost mark ends the primitive with v1's reason and the StepRunner decides (tested, M6).
- Reasoning effort stays hard-set to medium (the v1 client passes `reasoning_effort="medium"`, `agent_speedup_v1_client.py:409`).
- Wire order is insertion order through v1's `ordered_sse_transport` (`ordered_transport.py:29`). `scene` (plan) and `why` (repair) come first, so the reasoning precedes the actions.

### 3.3 Plan schema (exact)

`PID` below stands for the inline object `{"type": "string", "enum": ["A", "B", "C", "D", "E", "F", "G", "H"]}`. The builder writes it inline at every use; the wire schema contains no `$ref`, `$defs` or `pattern`.

Constructs used: closed objects with every key required, `anyOf` with null, string enums, bounded integers, and arrays of closed objects. [F] v1 used the first four on the live endpoint (`agent_speedup_v1_client.py:65-112`). Arrays of closed objects are new [Q, checked in M5 and by the optional schema-check call in M8].

```json
{
  "type": "object", "additionalProperties": false,
  "required": ["scene", "parts", "hold_true", "steps"],
  "properties": {
    "scene": {"type": "string", "minLength": 1, "maxLength": 80},
    "parts": {"type": "array", "minItems": 1, "maxItems": 8, "items": {
      "type": "object", "additionalProperties": false,
      "required": ["id", "text", "move", "w", "x"],
      "properties": {
        "id":   PID,
        "text": {"type": "string", "minLength": 3, "maxLength": 60},
        "move": {"type": "string", "enum": ["free", "fixed", "hinged", "sliding", "unknown"]},
        "w":    {"type": "integer", "minimum": 0, "maximum": 64},
        "x":    {"type": "integer", "minimum": 0, "maximum": 64}}}},
    "hold_true": {"type": "array", "maxItems": 3, "items": {
      "type": "object", "additionalProperties": false, "required": ["rel", "a", "b"],
      "properties": {
        "rel": {"type": "string", "enum": ["on", "in", "open", "closed", "effect"]},
        "a":   PID,
        "b":   {"anyOf": [PID, {"type": "null"}]}}}},
    "steps": {"type": "array", "minItems": 0, "maxItems": 6, "items": {
      "type": "object", "additionalProperties": false,
      "required": ["do", "obj", "grip", "ref", "sense", "keep", "until"],
      "properties": {
        "do":    {"type": "string", "enum": ["hold", "turn", "slide", "place_on", "place_in", "push_to", "release"]},
        "obj":   PID,
        "grip":  {"type": "array", "minItems": 0, "maxItems": 2, "items": PID},
        "ref":   {"anyOf": [PID, {"type": "null"}]},
        "sense": {"type": "string", "enum": ["any", "open", "close"]},
        "keep":  {"type": "array", "maxItems": 3, "items": {
          "type": "object", "additionalProperties": false, "required": ["c", "p"],
          "properties": {"c": {"type": "string", "enum": ["upright", "clear_of"]}, "p": PID}}},
        "until": {"type": "array", "maxItems": 3, "items": {
          "type": "object", "additionalProperties": false, "required": ["o", "p", "txt"],
          "properties": {
            "o":   {"type": "string", "enum": ["effect", "stop", "fits", "quarter_turn", "half_turn"]},
            "p":   {"anyOf": [PID, {"type": "null"}]},
            "txt": {"anyOf": [{"type": "string", "maxLength": 40}, {"type": "null"}]}}}}}}}
  }
}
```

`scene` is capped at 80 characters in this revision (120 before) to save about 12 visible tokens per plan (§3.8).

**Parts.**
- `text` = "<object>: <visible part>", with distinguishing appearance or position when similar objects exist ("black bowl between the plate and the ramekin: rim"). [F] v1 drawer 002 hunted a phantom "black" bowl when the description was wrong (`agent_speedup_v1_client.py:43-44` comment).
- `w` and `x` are coarse cells in the frame-aligned wrist grid and the upright external grid, 1–64 row-major from the top left; 0 = not visible in that image.
- `move` is a prior only. The executor verifies it with the DOF or sense probe (§4.9); the measurement wins, and `wrong_mechanism` is logged as evidence.

**Step relations** (`do`). The relation is the physical claim the executor checks.

| `do` | Relation established | `obj` / `grip` / `ref` | Completion when `until` is empty | Allowed `until` | Executor expansion (§4.8) |
|---|---|---|---|---|---|
| `hold` | `obj` is lifted and moves with the hand | obj; grip parts; – | held verified (§4.10) | – | register → grasp → load test |
| `turn` | a rotary part, held at `grip`, is rotated about its own axis | the rotary part; grip (default obj); – | (required) | effect, stop, quarter_turn, half_turn | register → grasp → twist with the wedge observer (sense found by trying) → release |
| `slide` | a sliding part, held at `grip`, moves along its own track; `open` = away from the body it slides in, `close` = into it | the moving part; grip; – | (required) | stop, fits, effect | register → grasp → DOF probe → follow axis → release → survey |
| `place_on` | `obj` rests on `ref`'s top surface, released and at rest | obj; grip parts; surface | support at the surface height, released, at rest | effect | register ref (empty hand) → grasp → in-hand → carry → aim → lower → classify → release → rest |
| `place_in` | `obj` rests inside `ref`'s cavity, below its rim, released and at rest | obj; grip parts; cavity | support at the floor, bottom below the rim, released, at rest | effect | as `place_on`, with the axis-specific fit rule and cavity checks |
| `push_to` | `obj` slides on its support to the pointed spot `ref` without lifting | obj; –; spot | object at the spot, at rest after withdrawal | – | register → contact beside → push → withdraw → rest |
| `release` | the hand is open and clear of `obj` | obj; –; – | released (§4.10) | – | open → retreat |

The executor inserts an implicit grasp before `turn`, `slide`, `place_*` and `push_to` when the hand does not hold a `grip` part, and an implicit release when a step needs a free hand. `grip` lists up to 2 parts in preference order; the planner may skip one by measurement.

**Criteria** (`until`; several are ORed, and the first one met ends the motion).

| `o` | Meaning | Observer |
|---|---|---|
| `effect` (p, txt) | region part `p` visibly changes as `txt` says ("burner rings turn red") | region-change observer O1, valid only after the region checks O7 and the minimum part rotation O8 |
| `stop` | the mechanism reached its end stop | wedge onset or part stop after the segment was classified free (K4w), or travel stall with part co-motion (K8) |
| `fits` (p) | the opening made by the slide is long enough for free part `p` | measured travel ≥ measured width of `p` + aim margin (M15); re-checked by the post-slide survey (M18) |
| `quarter_turn`, `half_turn` | π/2 or π of **part** rotation | hand yaw accumulated while the segment is classified free (K1w) |

**Constraints** (`keep`).
- `upright(p)`: p keeps its grasp-time in-hand pose within tolerance (O4).
- `clear_of(p)`: the hand and the held object stay outside p's measured keep-out volume (O5).
- Implicit, never written: `with_hand` while holding; `stays` for every completed step and every `hold_true` relation (re-checked at every relation boundary; a lost relation is reopened); a keep-out around every articulated part after it has been operated.

**Already-true relations** (`hold_true`). These are relations the model claims the images show as already true, and that must stay true.
- They are **unverified model claims** [I]. The change-based observers have no "before" state for them, so they can only check "unchanged since the episode start" (the baseline is the first image).
- Each becomes a `stays` invariant against that baseline and adds `clear_of(b)`.
- [F] KITCHEN_SCENE8 starts with the stove on, q = 1.438 (PA[2] TurnOn). Its plan would hold `{"rel": "effect", "a": <burner>, "b": <knob>}` and write no turn step.
- If LIBERO never fires and every observer holds, the claim is the first item named in the review request.

### 3.4 Plan validation (executor side)

An invalid plan gets one re-ask carrying the error text, and v1's rejection-storm damping applies (`REJECTION_STORM_LIMIT`, `agent_speedup_v1.py:38`). A second invalid plan ends with `invalid_plan`, the stuck alert, and holding still.

Rules:
- ids are unique; every reference is defined; `obj ≠ ref`;
- `turn` needs a criterion and a part with `move` ∈ {hinged, unknown};
- `slide` needs `sense` ∈ {open, close} and a part with `move` ∈ {sliding, unknown};
- `place_*` and `push_to` need `ref` and a free `obj`;
- `fits(p)` needs a free `p`;
- `upright` only on a free object; `clear_of.p ≠ obj`;
- `quarter_turn`/`half_turn` only on `turn`;
- an `effect` part must differ from the step's `obj` and `grip` parts (O7 checks the regions later).

Whether a grip part belongs to `obj` is checked later by measurement: connected regions in the height field, and co-motion of `obj` during the load test.

### 3.5 Prompt structure

**Fixed instructions** (one text for every task; versioned; sha256 in the manifest identity):
1. *Robot facts:* one parallel-jaw gripper with about 79 mm maximum opening, starting with the hand pointing straight down. The executor measures every distance, height, direction and angle itself by moving, touching and looking.
2. *Images:* both upright. The wrist image is turned so that its left and right match the external image. The edge where the robot's own fingertips appear is named (v1's `FINGERTIP_EDGE`, `mark_prompting.py:154`).
3. *What to write:* parts, steps, keeps, criteria and already-true relations, with the vocabulary definitions of §3.3.
4. *Rules:*
   - name only things visible in at least one image;
   - as the first grip, give a part thin enough to fit between the fingers (for example a thin protrusion, a ridge or a thin wall), and a second choice if one exists;
   - include enabling steps (make an opening before placing into it);
   - describe completion only from the criteria list;
   - list a relation as already true only when the images clearly show it;
   - **never give directions, distances, heights, angles or coordinates**.
5. *One worked example* using objects absent from every LIBERO BDDL file (the T0 lint checks this against the lint vocabulary, §5.1).
6. *Repair semantics* (repair instructions only): the actions of §3.7 and what each does.

[I] The first version's example part words ("a handle, rim, lever, knob, edge") mirrored the test tasks; the neutral wording above names shapes, not object parts.

There is no world-axis text and no image-to-world mapping; with no metric output there is nothing to convert. No sentence says or implies that a continuing episode means the task is incomplete (R-4 off, `scripts/evaluate_agent_speedup_v1.py:105-110`).

**Request text** (plan): `{"request": "plan", "task": "<official sentence>", "holding": null, "images": "1 external clean, 2 external grid, 3 wrist clean (fingertips at the <edge>), 4 wrist grid; cells 1-64 row-major from the top left"}`.

### 3.6 Grounding object.part by pointing

1. **Plan cells → seeds.** Each nonzero `w` cell centre seeds a region in the start wrist view. The start survey's κ values give it a 3-D estimate (§4.5), about ±½ cell, ±3.7 cm at table depth [P: 64 px cells at κ ≈ 864 px/m]. Region parts (a burner plate, a plate, a cavity, a spot) are registered from the seed by segmentation. A region whose seed fails its geometric check gets one `point_coarse` re-ask.
2. **Frame alignment.** Before any wrist image is shown, it is turned by k quarter turns so that its image-right best matches the external image-right in world terms. This uses the measured external axes (R-1x, `external_axes.py:79-113`) and state8 R. Answers are mapped back with `unrotate_pixel` (`mark_prompting.py:182`). Frontal (horizontal-approach) hands use v1's gravity turn instead (`gravity_quarter_turns`, `mark_prompting.py:157`).
   - [F] At the task-2 start pose, R0 = +x (image up) and R1 = −y (image right), and the agentview image right is +y (`t2_dump.json` camera `xmat`; critic-confirmed). So k = 2 there.
   - k is always computed, never assumed. [I] This removes bowl-live-001's cause (external-view directions converted with the wrist axes).
3. **Hover-time fine pointing (contact parts).** Every part whose grasp point is a pointed feature (lever, loop, bar, neck, box face) gets one `point_fine` call at its hover.
   - The crop is 128 px ×4, centred on the part's predicted pixel (its 3-D estimate reprojected through the R-2 mount, §4.5).
   - It uses v1's overlay drawing and `fine_point_pixel` mapping (`mark_prompting.py:144`, `:226`). v2 adds `fine_overlay_at(uv)` in its own module.
   - Parts whose grasp point comes from a fitted footprint (the rim point of a circular container, the centre line of a box) skip fine pointing when the fit passes D11, and get one call otherwise.
4. **Answer checks** (generalizing `bar_point_check`, `wrist_servo.py:83-114`). "Surroundings" is defined per check:
   - *Grasp part:* the pointed point must stand ≥ 3σ_h above **its finger-landing columns** (P16). These are the two cells-sets where the fingers would go along the planned jaw, so there must be room for the fingers on both sides. It is not compared with every neighbour. [I] A level neighbour, such as the moka body top 1.0331 beside the 1.0345 handle top, therefore cannot reject a correct handle point.
   - *Cavity point:* it must lie at least max(1 cm, 3σ_h) below the boundary ring of its connected low region (D12).
   - *Surface:* flat within D8 over the held object's footprint.
   - *Elongated part:* it must pass the structure-tensor check with its **measured** long axis (agree 0.8, coherence 0.5, rms 2: `wrist_servo.py:77-80`).
   - *LookAt answer:* it must lie in the projected ray band (L5).
   - One re-ask carries the measured reason. A second failure ends the step with `pointed_not_<kind>` → repair.
5. **LookAt** (a part with `w` = 0 and `x` ≠ 0; §4.9 gives the controller).
   - [F, geometry] The first version's rule fails. Servoing the hand onto the part's external pixel at the hand's height leaves the part Δh·cot 39° farther along the camera's horizontal viewing direction (agentview pitch 38.9°, from `t2_dump.json` `xmat`). That is about 0.31 m for Δh = 0.25 m, outside the 37.5° wrist half-FOV.
   - [F] `hand_image_motion` returns only a median flow vector, not an absolute hand pixel (`external_axes.py:42-62`).
   - v2 therefore does the following:
     - (a) `point_fine` in the **external** image on a crop centred on the plan cell, because an 8×8 external cell is 64 px, ≈ 11 cm at the scene;
     - (b) an absolute **HandLocator**;
     - (c) a **RayServo** that puts the grip on the part's external viewing ray at a safe height;
     - (d) a **RayDescent** along the ray (the null direction of the 2×3 J_ext), during which the wrist camera keeps the ray points within about 8.5 cm below the grip inside its inner 30° FOV [geometry: atan(1.235·h / (h + 0.097)) ≤ 30° ⇔ h ≤ 0.085 m];
     - (e) a stop at the first measured surface along the ray, then one `point_coarse` with the predicted hit marked and the ray-band answer check (L5), plus a ray–depth consistency check (L4).
   - LookAt needs R-1v (the vertical J_ext column). With R-2f only, a horizontal scan along the measured image-up direction is used, with one `point_coarse` per view and ≤ 3 views.
6. **Re-acquisition** (zero calls):
   - predict the pixel from the 3-D estimate and the current pose;
   - rotate the stored template by the measured hand-yaw change and scale it by the κ ratio, plus v1's `REACQUIRE_SCALES` (`wrist_servo.py:130`);
   - search ±48 px and accept a peak ≥ 0.6 within 3σ of the prediction.
   - Otherwise re-point (≤ 2 per part per view, V4), then `part_not_found` → repair.
7. **Persistence.** A part keeps its registration across repairs and replans while its id and text are unchanged.

### 3.7 Repair and review schema (exact)

`STEP` stands for the identical inline step-item object of §3.3; `PID` is as above.

```json
{
  "type": "object", "additionalProperties": false,
  "required": ["why", "action", "part", "step"],
  "properties": {
    "why":    {"type": "string", "minLength": 1, "maxLength": 100},
    "action": {"type": "string",
               "enum": ["retry", "repoint", "use_part", "insert_step", "skip_step", "swap_next", "replan", "hold_still"]},
    "part":   {"anyOf": [PID, {"type": "null"}]},
    "step":   {"anyOf": [STEP, {"type": "null"}]}
  }
}
```

| Action | Effect |
|---|---|
| `retry` | rerun the failed step from its last satisfied precondition |
| `repoint` | drop `part`'s registration and point it again |
| `use_part` | grasp `part` instead (it must be an existing part) |
| `insert_step` | put `step` before the failed step (e.g. clear an obstacle) |
| `skip_step` | the failed step becomes a `stays` relation if its observer currently holds; otherwise it is dropped |
| `swap_next` | exchange the failed step with the next step |
| `replan` | send one `replan` request (§3.2) |
| `hold_still` | hold position; the only answer besides the above in a review |

**Evidence rules.** Evidence is executor-written and ≤ 200 characters. It carries measured magnitudes in cm and degrees, named relations ("the object met a support 5.6 cm above the measured drawer floor, 0.2 cm below the measured rim") and the thresholds used. It never carries world axes or signed vectors.

**Review.** A review is sent only when every step is done, **every observer currently reports its relation holding**, and the episode continues.
- It is the same request with every step "done" and the `hold_true` claims listed.
- The instructions say only: "If the planned relations hold in the images, answer hold_still." Nothing says the episode continues because of failure (R-4).
- `hold_still` is logged as an **unverified model claim** (`model_claim: hold_still`). It is not a completion: the episode still ends only on raw LIBERO success, an operator stop or an error.
- If any observer later reports a relation lost, the executor reopens it without asking.
- At most 2 reviews per episode. After that the policy holds still, and the stuck alert stays an alert (PA[1] runner architecture; v1 `_stuck_reason`, `scripts/evaluate_agent_speedup_v1.py:326`).

### 3.8 Call budget and token projections

**Rates.** [F] The example plans are 834 (task 2) and 741 (drawer) compact-JSON characters (computed for this revision). [I] Dense JSON tokenizes at about 2.5–3.0 characters per visible token, against v1's prose-heavy 3.4 (critic).

**Latency fits** [F fits from D003]: decisions 0.51 s + 0.0376 s per output token; pointing 1.56 s + 0.0397 s per output token. Output tokens include reasoning.

**Reasoning** [P]: medium effort adds 150–500 reasoning tokens for a plan, about 30–100 for pointing, and about 60–250 for a repair. [F] v1 averaged 52 reasoning tokens per call over D003's 21 calls (1,093 in total), but a plan is a harder request.

**Compute** [F, measured for this revision on `t2_wrist.png`]: `NCCTrack.match` costs 1.4–1.7 ms per track per frame (numpy), and `cv2.matchTemplate` (TM_CCOEFF_NORMED, 21 px template, 85 px window) costs 0.16 ms. v2's FeatureTracker therefore uses cv2 (§4.5): ≈ 25 ms per tracked frame at 150 features, only during probes [P ≈ 3–5 s per episode].

| Task | Nominal calls | Visible output | Model time [P] | Conditional calls |
|---|---|---|---|---|
| libero_10 task 2 | plan + fine(knob lever) + fine(handle loop) = 3 | ≈ 280–335 + 2×5 | plan 15–32 s + 2 × 3–6 s = 21–44 s | +1–2 re-points; +1–2 repairs (5–12 s each) |
| drawer_bowl | plan + external fine(handle) + coarse(handle, after LookAt) + fine(handle, top-down) + fine(handle, frontal) + coarse(drawer interior) = 6; the bowl rim skips fine pointing when its circle fit passes | ≈ 250–300 + 5×5 | 14–28 s + 5 × 3–6 s = 29–58 s | +1 fine(bowl rim); +1–2 repairs |
| bowl_plate | plan (+ fine(rim) only if the circle fit fails) = 1–2 | ≈ 150–190 + 5 | 10–28 s | +1–2 repairs |

[F] v1 for comparison: drawer 003, 21 calls and 132.2 s; bowl 002, 11 calls and 81.6 s (runs' `summary.json`).

[I] The plan call is now the largest single model cost. If M8's first live plan call exceeds 32 s, the deferred asynchronous plan call overlapped with Calibrate+ (§6.8) is the first remedy to propose to the user.

### 3.9 Example plans

Cells are illustrative. Task-2 wrist cells refer to the frame-aligned image (2 quarter turns of `SURVEY/tasks/frames/t2_wrist.png`); external cells refer to `t2_agentview.png`.

**libero_10 task 2, "turn on the stove and put the moka pot on it"** (834 compact characters, ≈ 280–335 visible tokens; the `scene` below is 99 characters and would be shortened under the new 80-character cap):
```json
{"scene": "stove off; black knob with lever below the stove; moka pot at top edge, handle loop toward the frypan handle",
 "parts": [
  {"id": "A", "text": "stove knob: lever on top of the knob", "move": "hinged", "w": 31, "x": 30},
  {"id": "B", "text": "stove: ring burner plate", "move": "fixed", "w": 47, "x": 39},
  {"id": "C", "text": "moka pot: black handle loop", "move": "free", "w": 59, "x": 36},
  {"id": "D", "text": "moka pot: whole pot", "move": "free", "w": 60, "x": 44},
  {"id": "E", "text": "frying pan: long handle", "move": "free", "w": 43, "x": 35}],
 "hold_true": [],
 "steps": [
  {"do": "turn", "obj": "A", "grip": ["A"], "ref": null, "sense": "any", "keep": [],
   "until": [{"o": "effect", "p": "B", "txt": "burner rings turn red"}, {"o": "stop", "p": null, "txt": null}]},
  {"do": "place_on", "obj": "D", "grip": ["C"], "ref": "B", "sense": "any",
   "keep": [{"c": "upright", "p": "D"}, {"c": "clear_of", "p": "E"}], "until": []}]}
```

**drawer_bowl, "open the top drawer and put the bowl inside"** (741 compact characters, ≈ 250–300 tokens):
```json
{"scene": "cabinet with three closed drawers; white bowl with dark pattern in front of it beside a plate",
 "parts": [
  {"id": "A", "text": "cabinet: handle bar of the TOP drawer", "move": "sliding", "w": 0, "x": 21},
  {"id": "B", "text": "white bowl with dark pattern: whole bowl", "move": "free", "w": 28, "x": 45},
  {"id": "C", "text": "white bowl with dark pattern: rim", "move": "free", "w": 28, "x": 45},
  {"id": "D", "text": "cabinet: inside of the top drawer", "move": "sliding", "w": 0, "x": 21}],
 "hold_true": [],
 "steps": [
  {"do": "slide", "obj": "A", "grip": ["A"], "ref": null, "sense": "open", "keep": [],
   "until": [{"o": "fits", "p": "B", "txt": null}, {"o": "stop", "p": null, "txt": null}]},
  {"do": "place_in", "obj": "B", "grip": ["C"], "ref": "D", "sense": "any", "keep": [{"c": "upright", "p": "B"}], "until": []}]}
```

**bowl_plate** (≈ 150–190 tokens): parts A "black bowl between the plate and the ramekin: whole bowl", B "same black bowl: rim", C "plate: centre"; one step `place_on(A, grip [B], ref C, keep upright A)`.

**BDDL-like rendering** (logs and video only, generated from the JSON):
```
(:parts (A stove-knob-lever hinged) (B stove-burner-plate fixed) (C moka-pot-handle-loop free) (D moka-pot free) (E frying-pan-handle free))
(:goal  (and (effect B A) (on D B)))
(:steps (turn A :grip A :until (or (effect B "burner rings turn red") (stop)))
        (place_on D :grip C :ref B :keep (and (upright D) (clear_of E))))
```

---

## 4. Executor

### 4.1 Modules and data flow

```
            ┌──────────────── model (stateless calls, §3.2) ───────────────────┐
            │ plan / replan          point_fine / point_coarse          repair  │
            └────▲──────────────┬───────────────────────▲──────────────▲──────┘
     status, evidence│   plan JSON│              pixel ids│      outcome record│
            ┌────────┴───────┐ ┌──▼──────────┐  ┌─────────┴────────┐          │
            │ StepRunner     │◄┤PlanValidator│  │ GroundingService │          │
            │ (graph, order, │ └─────────────┘  │ seeds, crops,    │          │
            │ stays, ledger, │                  │ answer checks,   │          │
            │ escalation)    │──────────────────│ LookAt, reacquire│          │
            └──┬──────────▲──┘                  └─────────▲────────┘          │
     expand +  │          │ verdicts                      │ registrations    │
     schedule  ▼          │                               │                  │
            ┌─────────────┴────────┐   measured   ┌───────┴──────────────┐    │
            │ RelationTypes:       │◄────────────►│ SceneModel           │    │
            │ Grasp, Turn, Slide,  │   geometry   │  RobotModel (R)      │    │
            │ PlaceOn/In, PushTo,  │              │  CameraModel (f, f̂)  │    │
            │ Release, Register    │              │  FeatureTracker (cv2)│    │
            │ + GraspPlanner,      │              │  FeatureDepth, Touch │    │
            │   TransitPlanner,    │              │  HeightField         │    │
            │   AimSolver          │              │  PartRecords         │    │
            └──┬───────────────────┘              │  HeldFrame, KeepOuts │    │
     primitive │ factories (generators)           │  ExternalModel J_ext │    │
            ┌──▼────────────────────┐             │  HandLocator         │    │
            │ Primitives, 20 Hz:    │ before/after└───────▲──────────────┘    │
            │ v1 (unchanged) + v2   ├────────────►┌───────┴───────────┐      │
            └──┬────────────────────┘   views     │ ObserverBus +     │──────┘
               │                                  │ InvariantMonitor  │
               │                                  │ (WedgeObserver …) │
               ▼ 7-D OSC_POSE action              └───────────────────┘
                 (v1 HoldingActionAdapter)
      V2LiberoWorker + GenericContactMonitor (record-only) ──obs──► StepView
```

### 4.2 Step protocol

The runner contract is v1's: `reset()`, `act(obs)` returning one 7-D action, and `observe_transition(before, decision, after)`, with v1's pending-action and expected-observation invariants (`agent_speedup_v1.py:308-421`). These are re-implemented, because the v1 methods are bound to `AgentSpeedUpV1Policy`, whose constructor rejects unknown tasks (`agent_speedup_v1.py:241-242`).

```text
reset():      clear plan, registrations, ledger; phase = calibrate
act(obs):
    view = StepView(obs); scene.update(view)          # reprojection, held-frame update, height-field stamps
    if runner.needs_model(): <= 2 calls (plan | point batch of coarse+fine | repair | replan)
    action = active_primitive.next_action(view)       # generator-driven, as v1 macros (agent_speedup_v1.py:637-693)
observe_transition(before, action, after):
    primitive.observe_transition(...)
    invariants.observe(before, after)  -> may preempt the controller with a reason
    observers.update(); primitive done -> controller advances; relation verdict -> StepRunner + ledger + trace
```

Episode order: Calibrate+ (§4.5) → plan call → validation → steps.

**Client construction (corrected).**
- [F] The base client chain requires the episode's prompt at construction. `DirectHistoryAstraClient.__init__` defaults `task_prompt` to v1's bowl `TASK` (`direct_history.py:68`). Every payload goes through `_allowed()`, which raises when the observation prompt differs (`direct_history.py:37-42`, `:114`). v1 passes `task_prompt=self.task_spec.prompt` (`agent_speedup_v1_client.py:411`).
- The v2 client (`AgentSpeedUpV2Client(ChatGPTPlanHybridClient)`) therefore takes a **required** keyword `task_prompt` (no default). The runner gets the sentence from the reset observation after checking it against `list_tasks()` (PA[1] v1_eval_protocol: prompt matching at `scripts/evaluate_agent_speedup_v1.py:990-998`, `:1038-1039`).
- The client and policy take no task key and no per-task text.
- Tests (M5): constructing the v2 client without `task_prompt` raises; for every registry key, the plan payload's `task` equals the official prompt and `_allowed` accepts the reset observation; v1's bowl `TASK` string appears in no payload of any non-bowl key.

### 4.3 Representations

```python
@dataclass(frozen=True)
class Threshold:                 # every number the executor compares against; logged in evidence
    name: str
    value: float
    label: Literal["ROBOT", "ONLINE", "DD", "RULING"]
    derivation: str              # e.g. "max(0.010, 3*sqrt(sd_dest**2 + sd_b**2 + sd_touch**2))"

@dataclass
class Meas:                      # a measured scalar
    value: float; sigma: float; source: Literal["kappa", "loom", "touch", "dense", "plane", "prior"]

@dataclass
class PartRecord:                # measured; never derived from names
    pid: str; text: str; move_prior: str
    seeds: list                  # (camera, uv, step, source=plan_cell|fine|coarse|lookat, quantization_px)
    xyz: np.ndarray | None; cov: np.ndarray | None        # world point of the seed (3x3 covariance)
    track: NCCTrack | None                                # wrist_servo.py:117
    top_z: Meas | None; support_z: Meas | None
    region: Polygon | None       # world-xy footprint of the connected region above support + D9
    width: Callable[[float], Meas] | None                 # cross-section width along a jaw yaw, in the pad band
    elongation: tuple[np.ndarray, float] | None           # (world horizontal axis, coherence)
    circle: tuple[np.ndarray, float, float] | None        # (centre, radius, rms) when D11 passes
    cavity: dict | None          # floor_z, rim_z (Meas), free polygon, principal axes and widths, centre
    mount_side: np.ndarray | None                         # horizontal unit vector toward the taller structure (D13)
    dof: dict | None             # after probing: free | fixed | prismatic(axis, sign_open) | revolute(axis, centre, sense, backlash)
    rigid_patch: NCCTrack | None # a textured patch of the same rigid part beside the grip point (DOF/rotation observable)
    registered_empty_hand: bool; view_pose: np.ndarray    # state8 at registration

@dataclass
class HeightField:               # world grid, 5 mm cells, fused from all depth sources
    z: np.ndarray; sigma: np.ndarray; known: np.ndarray; source: np.ndarray; stamp: np.ndarray

@dataclass
class HeldFrame:                 # the held object relative to the hand
    pid: str; grip_pid: str; grasp_pose: np.ndarray       # state8 at hold_pass
    offset_h: np.ndarray; sigma_offset: float              # footprint centre in the hand frame
    b_low: Meas                  # grip z minus the object's lowest point
    r_obj: float; extent_axes: np.ndarray | None           # footprint extents along its principal axes
    tilt_deg: Meas | None; wrist_visible: bool
    silhouette: np.ndarray | None; ext_template: NCCTrack | None
    gap_ref_mm: float; sigma_carry_mm: float | None

@dataclass
class StepState:
    i: int; do: str; operands: dict; keep: list; until: list
    status: Literal["pending", "active", "done", "failed", "stays"]
    reason: str | None; evidence: dict; calls: dict; steps: int

class Observer(Protocol):
    name: str
    def start(self, view: StepView, ctx) -> None: ...
    def update(self, before: StepView, action, after: StepView, ctx) -> "ObsReport": ...

@dataclass
class ObsReport:
    verdict: Literal["pending", "met", "violated", "failed"]
    progress: float | None; reason: str | None; evidence: dict; thresholds: list[Threshold]
```

**Graph semantics.**
- Steps run in model order.
- Each step expands into preconditions: registered parts; held grip; destination registered with an empty hand; the fit rule for `place_in`.
- A precondition may be satisfied earlier than its step: destinations are registered before the grasp of the object to be placed.
- A done step becomes a `stays` invariant re-checked at every relation boundary; a lost one is reopened after the current relation.
- Raw LIBERO success, checked by the runner at every boundary, ends the episode at any moment, as in v1.

### 4.4 Robot model (pinned ROBOT facts and one contact convention)

| ID | Fact | Source |
|---|---|---|
| R1 | grip site = hand + 0.097·R2 | `robot_tool_geometry.py:49` |
| R2 | wrist camera = hand + 0.05·R0, optical axis R2; image right = R1, up = R0; principal point at the centre | `wrist_servo.py:29`, `:56-66`; R-2 |
| R3 | f ≈ 333.6 px at 512 px | brief, under R-2 [Q R-2f, §5.3]; cross-checked by the ONLINE f̂ (D3) |
| R4 | **Pads span −11.6 … +4.4 mm along the approach about the grip site** (finger body 0.0524 + pad centre 0.041 ± 0.008 against grip 0.097). Pad thickness 8 mm along the jaw; pad length **16 mm along R0** (±8 mm). **Top-down: pad world z ∈ [g − 4.4, g + 11.6] mm.** | `robot_tool_geometry.py:49-52`; `gripper_geom.json` (`finger_joint*_tip`) |
| R5 | **Fingertip reach t_f = 9.3 mm beyond the grip** (pinned). The finger STL hull reaches 9.2 mm (`robot_tool_geometry.py:163` vertices: max finger-vertex z 0.0538 + 0.0524 − 0.097). [F, CP] A full-opening descent stalled at grip 0.9272 with the fingertips on the 0.9178 knob-disk top (9.4 mm). The 12.5 mm sim AABB is a box bound and is not used. Finger width ±10.9 mm along R0. | as cited; critic-verified |
| R6 | **Palm underside p_min = 31.0 mm behind the grip** (pinned), from the convex hull of `HAND_COLLISION_VERTICES_M` (`robot_tool_geometry.py:57`): the vertices with \|jaw\| > 8 cm and the flat hull between them at hand z 0.066. MuJoCo collides with the mesh's convex hull. [I] The 26.1 mm value of `gripper_geom.json` is the geom's bounding box, not the surface. The 39–45 mm visual mesh is not used. | computed in the first version; critic: 31.04 mm |
| R7 | Hand footprint R1 −104.0 … +100.4 mm, R0 ±31.6 mm (hull vertices, `robot_facts_v2.json`); R1 = the finger-separation direction. [F, M0 review TLR-6] The earlier values (−102.5 … +105.7 mm, ±31.7 mm) were the hand mesh's grip-site AABB: box corners projected on the grip-site x axis, which is −R1 (sign flipped, and inflated by the mesh frame's 1.47° tilt). | `robot_facts_v2.json` (`hand_footprint_R1_mm`); `gripper_geom.json` (old AABB) |
| R8 | Pad inner faces at full opening ≥ 38.2 mm from the grip axis, outer pad faces 46.2 mm; finger bodies to 67.6 mm; finger joints 0–40 mm each | `gripper_geom.json`; `robot_tool_geometry.py:53` |
| R9 | Pad separation ≈ finger gap − 1.0 mm (cross-checks: the 16.3 mm bar reads 17.45–17.48 mm; [F, CP/SP] the 24.3 mm knob lever (2 × 0.01214, `flat_stove.xml:55`) reads 25.39 mm) | spec v1 §9; RECIPE line 10 §2.A |
| R10 | Command limits: 0.05 m per unit translation, rotation cap 0.15 rad/step, 20 Hz | `action.py:16-17`, `:48` |
| R11 | Gripper ramp: `current_action += [−1, 1]·0.01·sign(a)` per substep; action 0 holds | robosuite `panda_gripper.py` `format_action` (critic-confirmed) |
| R12 | Net hand-yaw budget about the approach: ±(2.8973 − π/4 − 0.15) ≈ ±1.96 rad from the start orientation. Joint 7 range ±2.8973, nominal init q7 = π/4; 0.15 rad covers init perturbation (task 2 episode 0 starts at q7 = 0.8177). | robosuite `panda/robot.xml:225`; LIBERO `mounted_panda.py:35-38`; `SURVEY/tasks/t2_dump.json` |
| R13 | Kitchen/tabletop MountedPanda base (−0.66, 0, 0.912); shoulder z 1.245; wrist point = grip − 0.204·R2; full stretch ≈ 0.804 m | `agent_speedup_v1.py:92-96`; R-5 |
| R14 | Robot start pose relative to its base: Δ0 = (0.4537, 0.0002, 0.2655) m; spread ≤ (1.1, 1.9, 1.8) cm = (0.011, 0.019, 0.018) m over 10 resets: the measured maxima (1.08, 1.85, 1.72) cm, rounded up (`START_OFFSET_SPREAD_M`; the earlier 1.7 cm z bound was a rounded-down maximum, M0 review TLR-11). [F] Base distances from the start pose are 2.25 cm (task 2 ep 0), 0.98 cm (drawer 003), 1.27 cm (bowl 002) (critic). [F] The other LIBERO mounts are the study MountedPanda (−0.75, 0, 0.912), 9.0 cm away, and the living-room OnTheGroundPanda (−0.51, 0, 0.42) (`SURVEY/tasks/t{0..9}_dump.json`). | PA[0] rationale; dumps [Q R-5b] |

M0 writes `robot_facts_v2.json` from the simulator's own robot collision geoms and pins it in a test: fingertip reach, pad span, the palm profile p(ξ) along the jaw, and finger landing footprints per aperture. Reading the robot's own asset is a ROBOT fact, not evaluator truth. The test fails if R5 or R6 differs from the asset by more than 0.5 mm.

### 4.5 Measured scene geometry

**Depth sources, in order of authority.**
1. **Touch** (ONLINE). A stall or touch event of `AnchoredMove` (`contact_skills.py:577-730`) gives surface z = grip z − t_f for a fingertip landing, or the matching robot offset for other contacts. σ_touch = the realized motion of one stall window ([F, CP] ≈ 0.7–2.6 mm at a 10 mm/step command).
2. **Sparse κ at features** (ONLINE, R-1; the primary visual source).
   - Features: the pointed and seeded parts; up to 150 auto-selected textured patches on a grid (patch std ≥ `min_std` 4 grey, the `NCCTrack` default); and VS's relief probes beside each contact part (on its principal axes at ±(w/2 + 15 mm), the highest-contrast patch within ±8 px; v1's `_textured_default` idea, `wrist_servo.py:1263-1276`).
   - [F, measured for this revision] At the task-2 start view only 54 of 143 grid patches (21 px, 36 px spacing) pass the std test; the plain wood table has patch std ≈ 3 grey. The critic counted 64 of 156. So most table cells start unknown, and D7's plane fill matters.
   - Tracking uses the v2 **FeatureTracker**: `cv2.matchTemplate` (TM_CCOEFF_NORMED) with `NCCTrack`'s quadratic subpixel peak. It is validated against `NCCTrack` on recorded frames (|Δuv| ≤ 0.1 px, M1 test).
   - It tracks only during probe moves and hover surveys. The search window is centred on the shift predicted from the commanded motion and the prior κ, because [F, critic] a 4 cm Calibrate move shifts features by about 35 px, beyond `NCCTrack`'s ±32 px search (`wrist_servo.py:134`). Probe frames are tracked every step. `Align` keeps v1's `NCCTrack` for its 1–3 marks.
   - Probe set: the hand moves out and back along R1 and R0, after v1's quiet settle (Align `SETTLE_*`). The start survey reuses Calibrate's three 4 cm moves (`PROBE_MOVES`, `agent_speedup_v1.py:72`); hovers use ±2 cm (D2).
   - Each feature's κ comes from the similarity model J = −κP fitted to its displacements (`ImageJacobian` math, `wrist_servo.py:295-432`), with v1's validity rules (shift ≥ 6 px, residual < 0.1·shift; D1).
   - Depth Z = f/κ (R3 or D3). The world point is C + Z·(R2 + ((u − c_u)/f)·R1 − ((v − c_v)/f)·R0), with C = the camera position (R2).
3. **Looming** (ONLINE, [Q R-1v]). Calibrate+ adds a vertical probe of ±4 cm. For a top-down hand this is motion along the optical axis, so tracked features scale about the principal point by s = Z₁/(Z₁ − d), giving Z₁ = d·s/(s − 1) without f (the radial-scaling idea of v1's κ transfer, `wrist_servo.py:893-930`). It gives f̂ = the median over features of κ_i·Z_i, the f cross-check (D3). The same vertical moves supply the third column of J_ext.
4. **Dense flow** (optional layer, DD, decided by gate G1). Farneback flow with v1's call (`external_axes.py:51`), with the rotation residual compensated by K·ΔR·K⁻¹. It is valid where forward-backward flow agrees within 0.5 px and the gradient is ≥ 2 grey/px (D6). [I] It fills free-space evidence between sparse features; its failure on dark or thin parts can only leave cells unknown.

**HeightField.**
- 5 mm world cells, each holding z, σ, source and stamp. Touch overrides κ, κ overrides plane fill, and plane fill overrides dense.
- **Unknown cells** (D7, revised):
  - (a) A cell within 2 cm of a fitted support plane's known cells, with no region edge (D9) between them, and whose colour lies within the plane's measured colour distribution (|ΔLab| ≤ 3σ_plane), is filled at the plane height + 3σ_plane (source `plane`).
  - (b) Every other unknown cell is an obstacle at the maximum of known neighbours within 2 cm, plus 3σ.
  - [I] A depth failure still makes plans conservative except where the cell looks like the measured support. A textureless black object beside a light table is never filled from the table.
- Cells are re-stamped whenever a later view covers them.
- Cells inside the held-object mask or the fingertip band are never updated. The fingertip band is measured in Calibrate+ from the free-air close difference image, replacing v1's fixed `FINGERTIP_BAND_ROW_FRACTION` 410/512 (`agent_speedup_v1.py:48`).

**PartRecord registration from a seed** (all DD × ONLINE):
1. lift the seed to 3-D;
2. grow the region (same height band D8, connected, edge D9);
3. support plane = the dominant plane in an annulus around the region;
4. top_z = the 95th percentile of region heights;
5. width per yaw from image edges and depth steps in the pad band;
6. elongation axis and coherence from the structure tensor (v1's tensor of `bar_point_check`, `wrist_servo.py:103-112`, re-implemented to return the orientation, about 10 lines);
7. circle fit (D11), cavity with principal axes and widths (D12), mount side (D13);
8. a `rigid_patch`: the most textured patch of the same connected region at least 15 mm from the grip point, used as the part's own motion observable (§4.9).

**ExternalModel and HandLocator** (R-1x extended by the vertical probe [Q R-1v]).
- *J_ext.* The 5 Calibrate+ moves (3 horizontal v1 moves + up + down) and `hand_image_motion` (`external_axes.py:42-62`) give a least-squares 2×3 J_ext in px per metre of hand xyz, with v1's residual test (`MAX_RESIDUAL_DEG` 30°, `external_axes.py:31`). Broyden secants on clean non-rotating steps refine it.
- *Directions.* From J_ext: v̂ = the image direction of world vertical, n̂ ⟂ v̂, and the ray direction d = the unit null vector of J_ext with d_z < 0.
- *Parallax decomposition.* For a hand-attached feature S and a static feature D, the **n̂ row** n̂·(s_S − s_D) depends only on (S − D)_xy, so it is parallax-free in the affine model. The **v̂ row** is used only with a measured height difference (anchors or a contact). [I] This is the parallax that misled drawer 003's decisions 19–20 (D003 audit).
- *Frame alignment.* The 2×2 horizontal part also feeds v1's `fit_external_axes` summary (`external_axes.py:79-113`).
- *HandLocator* (new) gives the grip's **absolute** external pixel.
  - On each probe frame pair, take the fastest-decile changed region (v1's `hand_image_motion` mask). The fingertip pixel u_tip is its extreme point along +v̂ (image-down of world vertical), and u_grip = u_tip + J_ext·(0, 0, t_f).
  - [I] With a top-down hand the fingertips are about 10 cm below the hand body, so they are that extreme for a camera pitched 39°.
  - Validation: across the Calibrate+ moves, the displacement of u_grip must match J_ext·Δp within L1. The residual sets σ_loc. M1 checks the locator against the evaluator's grip projection (diagnosis only, ±6 px).
  - [I] A "motion-mask centroid plus a calibrated offset" (critic) cannot be calibrated without an absolute reference, so the extreme point is used instead.

**CameraModel.** f_used = R3 when R-2f is confirmed and |f̂/R3 − 1| ≤ max(0.05, 3σ_f̂/R3); otherwise f̂ with its σ. A disagreement is logged as `camera_model_disagreement`, and the larger σ is propagated.

**Calibrate+** (episode start, zero calls, ≈ 180–200 steps [P]; v1's Calibrate plus its probes took 89 steps in D003). It wraps v1 `Calibrate` (open hold, empty close; `contact_skills.py:497-553`) with:
1. while v1's empty close is held: a ±5° closed-hand yaw probe at the twist cap K2. This gives the free yaw rate and σ_g, the robot-only gap-rate noise under yaw. [F, SP] The empty closed gap stayed at 0.997–1.000 mm with |Δ| ≤ 0.003 mm/step while yawing 1.31°/step.
2. one extra empty close (≈ 50 steps). The two gap(15) values must agree within the G5 floor; σ_close = |Δ|/√2, floored. If they disagree, a third close is added (+50).
3. the 3 v1 external-axes probe moves, with their wrist frames kept (start survey);
4. the vertical ±4 cm probe (looming, the J_ext column, HandLocator validation);
5. static frames for σ_static and the per-region false-change rates (O1);
6. the fingertip-band mask.

The first version's two extra closes would have cost ≈ 250 steps (critic).

### 4.6 Choosing the contact primitive from measured part geometry

**Inputs:** the PartRecord (top, support, width per yaw, elongation, circle, cavity, mount side, σ), the HeightField, the RobotModel, the calibration, keep-outs, and the `move` prior (used only to allow palm stops on fixtures). Never the part's name.

**Candidates** = family × jaw yaw. All apertures are full opening; pre-shaped apertures are deferred (§9.3).
- Families:
  - `top_down`;
  - `frontal`: horizontal approach opposite the measured `mount_side`, jaw perpendicular to the part's long axis and to the approach;
  - `frontal_pitched_45`: the frontal hand pitched 45° about the jaw axis, for low frontal parts.
  - Frontal families need a measured `mount_side` and, outside the verified arena, R-5b (§5.3).
- Yaws:
  - perpendicular to the measured long axis when the part is elongated (D10); otherwise every 15° within the yaw budget (R12); smaller turns are preferred.
  - Of the two yaws 180° apart (equivalent for a parallel-jaw grasp), the planner prefers the one that keeps the object about to be held in the wrist view (P17).

**Finger-descent columns (P16, new).** For a candidate, each finger's descent column is its lowest-point footprint at full opening (R8; until M0, the pad footprint 38.2–46.2 mm from the axis × ±8 mm along R0, inflated by e_lat). The column runs from the hover down to the part's support. Its **landing height** is the highest measured surface in the column, including D7 obstacles. A column may straddle an edge; the higher side is the landing.

**Depth rules** (preference order):

| Rule | Condition | Stop |
|---|---|---|
| `touch_stop` | (1) both columns have a landing (P4); (2) after the clearance retreat P10, the pads overlap the part's band [part_low, top_z] by ≥ 8 mm (P3); (3) the closing sweep from each landing to the part edge has no height above the landing + 3σ_h (P5) | guarded descent to the stall or touch on the landing; expected contact band P8 |
| `face_stop` | frontal family; the fingertips meet the measured mounting face behind the part | guarded approach to the face stall (v1 frontal bar step 4, spec v1 §7.2) |
| `height_stop` | target g* = top_z − 11.6 mm − max(2 mm, 1.5σ_h), so the pad top is below the part top; requires the palm clearance g* + p_min ≥ top_z + m_palm (P6), and both columns' landings below the pad bottom (g* − 4.4 mm) − max(3 mm, 3σ_h) | guarded descent to g*; any earlier contact is classified (P8) |
| `palm_stop` | only on parts with `move` ≠ free and verified fixed after contact | descent to the palm touch on the part top |

**Rule P13: never plan a palm stop on a free object.** [F] The moka handle protrudes about 6 cm beyond the pot body and the pot weighs 68 g (PA[0] task 2); pressing its top tips the pot before a stall.

**Checks on every candidate:**
- *Width, three-way* (P2). Pad separation sep = the aperture gap − 1.0 mm (R9); e_lat = 2σ_align + σ_w + 1.0 mm.
  - **feasible** if w + 2·e_lat ≤ sep;
  - **infeasible** if w − 2σ_w ≥ sep − 2 mm;
  - otherwise **marginal**: ranked after every feasible candidate and executed only with contact classification armed (try-and-classify).
  - [F] The 63 mm soup can closed at 63.0 mm (`oracle_t7_a.json`).
  - [I, critic-corrected] The 66 mm moka body is feasible for e_lat ≤ 5.75 mm (sep 77.5 mm). Whether a body pinch is refused depends on the columns (P16), not on P2 (§7.1 step 12).
- *Hand-footprint clearance* (P7, revised). The maximum height in the hand footprint (R7) at the planned grip must lie below grip z + p_min (31.0 mm) − max(5 mm, 3σ_h). The footprint **excludes** the part being grasped (its registered region) and the finger slot (the cells between the two columns, within the pad width). The interaction of those cells with the hand is governed by P6, P3 and P5.
- *Turned-hand clearance* (P12): the hand volume swept by any Orient to the candidate orientation must clear the HeightField by max(1 cm, 3σ_h). [F] Drawer 003's low handle hover swept the bowl and caused 70% of the 4,078 forbidden points (D003 audit).
- *Reach:* v1's `wrist_to_shoulder_m` stretch guard for frontal hands (`agent_speedup_v1.py:168-171`, with §5.3's base rule) and v1's reach-limit memory.
- *Keep-outs:* `clear_of` volumes and articulated-part keep-outs are obstacles (vertical extent per O5).

**Expected first contacts.** Every plan lists the contacts it expects, with heights: the fingertips on the column landings (`touch_stop`), the face (`face_stop`), none before g* (`height_stop`), or the palm on top (`palm_stop`).

**Score** = min(width margin / σ, pad overlap / 8 mm, clearance / margin) + 0.5 if touch-terminated − 0.2 per 90° of turn − 1 if marginal + 0.3 if P17 holds. [I] The weights only order feasible candidates; they never make an infeasible candidate feasible. DD, logged.

```python
def plan_contact(part, hf, robot, cal, keepouts, yaw_now, next_held=None) -> Plan | Infeasible:
    plans, reasons = [], []
    for family in families_for(part):                   # frontal* need part.mount_side
        for yaw in jaw_yaws(part, family, yaw_now):     # D10 axis or 15-deg grid within R12; both 180-deg twins
            c = evaluate(part, hf, robot, family, yaw, keepouts, next_held)  # P2-P8, P12, P16, P17
            (plans if c.feasible else reasons).append(c)
    if not plans:
        return Infeasible(reasons)                      # e.g. part_too_wide, column_blocked, footprint_blocked
    return max(plans, key=lambda c: (not c.marginal, c.score))
```

**Contact classification** (generalizes v1's rim-only `blocked_above_rim`, `agent_speedup_v1.py:1274-1293`). When a descent stops at grip z_c, it is compared with the plan's expected contacts (band P8 = ±max(5 mm, 2σ_h)):
- Inside a band: the expected contact.
- Above every band: `blocked_above_part` → **never close**; return to the hover and try the next candidate, with no model call.
- Below every band (it descended past where contact should have occurred): `no_contact_where_expected` → re-register the part (re-point).
- No contact within the bound P9: `no_contact_within_bound`.

**v1's two strategies are outputs of this planner** [I]:
- `rim_pinch` = `top_down`, jaw radial to the measured circle at the chosen rim point, with a wall cross-section of a few mm. It is a `touch_stop`: the inner column lands on the container floor and the outer one on the support.
  - The rim point and side come from footprint clearance (D11 centre + r·direction), so a drawer front overhanging the bowl is rejected by P7 before descending.
  - [F] In drawer 002, 13 descents stalled on the drawer front at grip z 1.092–1.104 (`agent_speedup_v1.py:80` comment; spec v1 §17.3).
- `frontal_bar_pinch` = `frontal`, approach opposite `mount_side`, jaw vertical for a horizontal bar, `face_stop`.
  - Top-down on the bar is rejected by P7: the 204.4 mm hand footprint (R7 hull; the earlier 208 mm was the AABB) over the cabinet top (1.128) sits below the hand underside at bar height [rationale only: PA[0] task 3, D003 cabinet AABBs].

### 4.7 In-hand pose estimation (HeldFrame)

1. **Prior at the close:** offset_h⁰ = (registered footprint centre − grip) in the hand frame; b_low⁰ = g_close − support_z(pick). The object rested on its support at grasp time, so its lowest point was at the measured support height. [F, CP] For the moka handle grasp, b_low = 1.0219 − 0.900 = 0.1219 m. At the support stall, this predicts a bottom of 0.9297 against the burner top 0.9305 (−0.8 mm).
2. **Visibility (P17).**
   - [F, critic] With the handle grasp, the pot centre is 8 cm from the grip (pot − grip = (−0.0001, +0.0795), CP).
   - A yaw putting the camera (hand + 0.05·R0) on the pot's side leaves the lid about 8 cm from the camera and filling the wrist view. The opposite yaw puts the pot about 59° off-axis, outside the 37.5° half-FOV.
   - The planner therefore prefers the yaw with R0·(footprint centre − grip) ≥ 0, within R12 and P12.
   - When no such yaw is feasible, `wrist_visible` = false, and then:
     - re-localization and the silhouette use the external view only (external template and n̂ row);
     - σ_offset = max(measured, 0.5·r_obj) (M17);
     - Touchdown is always run (b_low by touch);
     - AimCheck uses the external n̂ row only.
3. **Re-localization after the close, before the lift** (hand static; MP): re-segment the object in the wrist view, seeded at its projected centre, and update offset_h. [F] Drawer 003's close dragged the bowl 2.6 cm along +y (D003 audit).
4. **Lift silhouette:** during VerifyHold's 2.5 cm lift (`LIFT_TEST_M`, `agent_speedup_v1.py:52`), the held object has zero flow while the scene looms. The zero-flow mask gives the silhouette and centroid (RC's SilhouetteProbe without extra motion). If it is ambiguous near the principal point, one 1.5 cm lateral move is added.
5. **External co-motion:** the object's external template must move by J_ext·Δp ± 3σ. This confirms the hold; a principal-axis change indicates tilt.
6. **Touchdown trigger** (M14): a gap change during the lift > max(1 mm, 6σ_close), or silhouette IoU < 0.8 against the pre-lift mask, or a template shift > max(4 px, 3σ_track), or `wrist_visible` = false.
   - Touchdown lowers the object back onto the pick support until the support event (≤ lift + 1 cm) and sets b_low = g_touch − support_z(pick) by touch (MP).
   - [F] In drawer 003 the VerifyHold gap fell from 11.86 to 4.06 mm (RECIPE line 10 §1). The bowl later hung about 15° tilted, with its lowest point 3.7 cm below the grip instead of the 1.3 cm at pick time (D003 truth lens).
   - A pivot during Touchdown (silhouette change) is logged as `touchdown_pivot` and inflates σ_b.
7. **σ:** σ_offset from the centroid scatter and κ σ; with IoU < 0.8, σ_offset = 0.5·r_obj (M17). The footprint extents along its principal axes (`extent_axes`) feed the fit rule M18.

Metric conversion of wrist pixels near the grip plane uses κ at the object's measured depth (Z ≈ 0.097 m + the measured top-to-grip distance), or the finger-pad κ measured in Calibrate+ (pad pixel separation against the gap).

### 4.8 Relation types

| Relation | Preconditions | Controller chain (§4.9) | Invariants while active | Completion | Local recovery (no call) |
|---|---|---|---|---|---|
| `registered(p)` (implicit) | – | Survey: hover (P11) → ParallaxProbe → pointing if needed (§3.6) → PartRecord; LookAt when `w` = 0 | – | record valid; σ meets the next controller's need | lower the hover (doubles κ); re-point ≤ 2; LookAt variant |
| `held(obj via grip)` | registered(grip); hand empty | Grasp: plan → hover → [Orient] → Align → descent → classify → clearance → close → load test → DOF/mobility → HeldFrame | gap ≥ hold floor G6; co-motion O3 | present (G5) + load test + co-motion of `obj` | empty close → re-align toward the measured centre line by ½ the measured width, ≤ 2; blocked → next candidate; all candidates → next grip part |
| `turned(p)` | held(p) | Turn: twist segment with the wedge observer → (reverse if blocked) → HingeTwist → release in place → post-check | gap ≥ floor G6; WedgeObserver; part-rotation observer (K7); yaw budget (R12) | criterion met; after release the part stays (K9) and the effect persists | blocked → other sense (once per sense); slip → re-grasp at the same part and continue from the measured angle, ≤ 2 |
| `slid(p, open\|close)` | held(p) | Slide: DOF probe (part observables) → FollowAxis → release (v1 handle path) → survey | gap ≥ floor; no gap jump (G10); co-motion of the part's rigid patch / external template; travel monotone | criterion met; part static after release | stall before expected → posture variant ≤ 2 |
| `supported_on(obj, ref)` | held(obj); **registered(ref) with an empty hand**; a free target spot (M16) | Carry → AimCheck → GuardedLower → SupportClassify → release → retreat → rest | held, upright, keep-outs, drift (M5) | support at ref's height (M8), released, at rest (M11) | perch/miss → lift, re-measure, re-aim, ≤ 2; drift → re-aim, ≤ 2 |
| `inside(obj, ref)` | as above; cavity measured (floor, rim, free polygon with principal axes); **axis-specific fit rule M18** (below); if rim − floor < 2·tol_h, an empty-hand floor touch probe first (M10) | as above, with compliant entry on tight axes | as above | as above, plus bottom ≤ rim − M9 | as above; a slide-axis shortfall → re-grasp the slide part and pull more (no call) |
| `pushed_to(obj, spot)` | registered(obj, spot); both on one measured support | PushTo | co-motion of obj with the hand | obj centre at the spot (O6), at rest after withdrawal | lost contact → re-contact ≤ 2 |
| `released(obj)` | – | open settle → retreat 3 cm along −R2 | – | gap ≥ g_open − 2 mm; obj does not follow the hand | – |
| keep `upright(p)` | held | – | O4 | – | violation → restore the grasp orientation; tilt beyond O4 → Touchdown, release, re-grasp once |
| keep `clear_of(p)` | – | keep-out obstacle in every plan | no unexpected contact within the keep-out (O5) | – | back off (v1 rule) |
| `stays(rel)` | from done or `hold_true` | – | the relation's observer at every boundary (`hold_true`: unchanged since the start baseline) | – | reopen the relation |

**Axis-specific fit rule M18** (replaces "opening ≥ extent + 2·max(5 mm, σ_aim)", which can never hold in the run-003 drawer).
- For each horizontal principal axis a of the cavity's free polygon:
  - clearance c_a = W_open,a − W_obj,a, where W_obj,a is the held object's measured extent along a;
  - σ_c = √(σ_open,a² + σ_obj,a²).
- Classes:
  - **Free axis**, c_a ≥ 2·max(5 mm, σ_aim): the normal aim tolerance M4 applies.
  - **Tight axis**, −2σ_c ≤ c_a < 2·max(5 mm, σ_aim): **compliant entry**. Aim at the opening's mid-line on that axis with the floor tolerance (5 mm), lower at the slow speed, and let SupportClassify decide by height. A rim support is `perched` → lift 1.5 cm, re-measure, re-aim, ≤ 2; a floor support is `on_floor` → release.
  - **Blocked axis**, c_a < −2σ_c (the object is measurably larger than the opening): if a is the slide axis of a just-opened mechanism, pull more (no call); otherwise `opening_too_small` → repair.
- [F] Evidence (D003 skeptic): the run-003 drawer gives a flat drop 1.6 mm of total clearance in y, yet scripted drops ±1.1 cm off-centre in y still landed flat on the floor, because the tilted hanging bowl slides in. x was decisive: at the run's x 0.118 even a centred-y drop tipped 41° on g14 and failed In(); drops centred or within ±3 cm in x landed on the floor.
- [I] So the y axis is tight, which means compliant entry, not a refusal. The x axis (≈ 20 cm opening for a 10.7 cm bowl) is free.
- Validated in M3 (G3) on those counterfactual aims.

### 4.9 Controllers

**Grasp(part).**
1. Plan (§4.6).
2. Hover: fingertips at max(5 cm, 3σ_h + 2 cm) above the part top (P11), turned-hand clearance checked (P12).
3. Fine pointing at the hover if the part needs it (§3.6), then ParallaxProbe ±2 cm (D2) and re-registration.
4. `Orient` if the plan's yaw or family differs (`contact_skills.py:775-847`), then re-acquire by prediction.
5. `Align` (constructed with `allow_pointing=False`) in mode `across_bar` when the part is elongated, else `both` (`wrist_servo.py:462-1237`), on the plan's grip point; the setpoint comes from v1's `setpoint` (`wrist_servo.py:62-66`).
6. Descent: v1's ladder pattern (20 mm segments, 3-step settle, Align refresh; `agent_speedup_v1.py:45-46`, `:1144-1198`), re-written and bounded by P9 instead of v1's 0.300 m (`agent_speedup_v1.py:50`). `height_stop` plans end at g*.
7. Contact classification (§4.6).
8. Clearance retreat centring the pad band on [contact surface, part top], clipped 0–8 mm (P10). This replaces `RIM_CLEARANCE_M` 4 mm and `BAR_CLEARANCE_M` 5 mm (`agent_speedup_v1.py:42-43`).
9. `GripperSettlePrimitive("close")` (`contact_skills.py:403-445`) and `classify_close` with the online margin (`contact_skills.py:556`, margin G5); width consistency G13.
10. Load test along the next motion: the `VerifyHold` lift for carried objects (`contact_skills.py:732-772`, floor G6); the first twist segment for `turn`; the DOF probe for `slide`.
11. **DOF/mobility probe** (VS; decided by **part observables**, K10).
    - `free` if the lift passes with object co-motion (O3).
    - A lift stall with the gap held (`hold_jam`) = constrained. Then closed-grip probes of ±1.5 cm along the approach, the long axis and z classify each direction by:
      - (a) the part's `rigid_patch` in the wrist view: static relative to the hand ⇒ the part moved with the hand; shifted by ≥ 50% of the hand's image motion ⇒ the hand slid over or along the part;
      - (b) the part's external template moving by J_ext·Δp ± 3σ;
      - (c) the finger-gap trend (a sustained rise = wedging against a constraint).
    - The hand realized/commanded ratio is logged as evidence only.
    - [I] This is what separates "pads sliding along a pinched bar" from "lateral free" (critic). A uniform bar is not its own observable, but the drawer face beside it is.
12. HeldFrame (§4.7).

**Turn(part)** (re-written; sense, end stop and slip from part observables).
1. Grasp. [F, CP] For the task-2 lever, both a full-opening `touch_stop` on the disk (CP stall at grip 0.9272) and `height_stop` (CP close at 25.39 mm from grip 0.9359) work (§7.1).
2. **Twist segments with the WedgeObserver.** Each step commands a yaw δ ≤ 0.1 rad (K2) about the approach, through v1's `HoldingActionAdapter` (`contact_skills.py:180-192`), with a **soft** position hold.
   - [F, CP] With a fixed hold target the hand orbited the hinge by compliance, 12 mm off the hinge, and turned the knob 1.2 rad in 60 steps.
   - [F, CP] A 0.5 rad cap lost the lever: the gap fell to 2.45 mm and the grip was dragged about 10 cm. K2 therefore stays ≤ 0.1.
   - The first segment's sense is arbitrary (+ about the approach) and is recorded. There is no separate probe: the first segment is the probe.
3. **Sense decision** (K1w; both classes are decided at step 12, extended to 16 if undecided; [F, SP] evidence):

   | From a fresh close | Hand yaw per step | Finger gap over 12 steps | Knob q (diagnosis) |
   |---|---|---|---|
   | free sense first | +1.16…+1.19° | 25.39 → 27.10 mm; rate 0.29 → 0.004 mm/step, decaying | 0 → 0.134 rad |
   | blocked sense first | −1.08…−1.17° | 25.39 → 29.58 mm; rate 0.34–0.51 mm/step, sustained | stays −0.005 (limit) |

   - Let r(k) be the signed gap rate, *early* the mean of r over steps 1–4 of the segment, and *late* the mean over its last 4 steps. ℓω is the rigid-wedge rate: ℓ = the pad length along R0 (16 mm, R4) and ω = the measured hand yaw rate in rad/step.
   - **free** if late ≤ max(0.1·ℓω, 4σ_g, 0.25·early);
   - **blocked** if late ≥ max(0.5·ℓω, 0.75·early) in the two windows ending at steps 11 and 12. [I] An earlier blocked decision is not allowed: at steps 6–7 the free sense's decaying transient still averages 0.20–0.23 mm/step [F, SP], which reaches the blocked threshold;
   - otherwise extend the segment to 16 steps, then use the secondary observable (step 5). If still undecided: `sense_undecided` → the other sense.
   - [F, SP] Late = 0.045 (free) against 0.445 mm/step (blocked); 0.1·ℓω ≈ 0.032 and 0.5·ℓω ≈ 0.16 mm/step.
   - The hand yaw ratio does not separate the senses. [F, SP] The closed hand in free air yaws 1.31°/step at this cap, so both senses read 0.82–0.95× free (Appendix A, R2).
   - On **blocked**: reverse. [F, SP] After a reversal the gap first falls (unwedging, 2–13 steps), then may rise again while the backlash is taken up (up to 0.62 mm/step for about 10 steps), then flattens if free. The sense decision for the reversed segment therefore starts its *early* window at the first non-negative rate after the fall.
   - Both senses blocked → `not_rotatable` (evidence: both wedge traces).
4. **Axis.** The prior centre c₀ is the D11 circle centre around the gripped part when it passes (the knob disk), else the grip point. σ_c = max(fit σ, 5 mm, the measured close drag).
   - [F] The disk centre lies 4.6 mm from the hinge (`flat_stove.xml:35` joint at the body origin; `t2_dump.json` button AABB centre (−0.2008, 0.2074) against the hinge (−0.1964, 0.2061)). [F, CP] The close dragged the grip a further 3.7 mm.
   - c is refit from the realized grip path every 5 steps once the segment is classified free.
   - Orbit-follow (p_k = c + Rz(θ_k)(p₀ − c)) is commanded only when the fitted centre is > K3 from the grip.
   - `axis_tug` = the realized-versus-commanded lateral deviation > max(3 mm, 2·sin(θ/2)·σ_c + 3σ_pos) for 4 steps → re-centre once, then repair.
5. **Part-rotation observer** (K7 machinery, secondary).
   - It tracks the part's `rigid_patch` orientation in the wrist view (structure tensor of the patch, or the lever's measured long axis) relative to the hand.
   - A part rotation in the image ≥ max(3°, 3σ_θ) while the hand yaws means the part is not co-rotating (blocked or slipping). An apparent rotation of ≈ 0 while the hand yaws means the part co-rotates.
   - Where the patch is textureless (a black knob may be), this observer reports `unmeasurable` and the wedge observer decides alone (`sense_by_wedge_only` in the trace).
   - [I] For a round knob gripped on its cylinder (no lever), a blocked twist makes the pads slip without wedging, so this observer is the primary one there [Q, M4 fake round knob].
6. **Monitors while twisting the free sense:**
   - hold floor G6;
   - slip: a gap fall ≤ −0.5·ℓω for ≥ 2 steps outside an expected post-reversal unwedge, or the part-rotation observer, or G6;
   - EffectWatch (O1) on the `until` effect part, in the external view (hand mask from `hand_image_motion`'s fastest decile, dilated) and in the wrist view when visible, valid only under O7 and O8;
   - the end stop K4w: a wedge onset (the K1w blocked test) or a part stop (step 5) after the segment was classified free;
   - the yaw budget R12.
   - [F, SP] After a reversal the knob reached its lower limit and the gap rate jumped from ≤ 0.004 to 0.18, 0.49 and 0.44 mm/step, so K4w fires 3 steps after the limit.
   - The first version's K4 (hand-yaw progress < 0.5° for 12 steps) is kept only as `hand_yaw_stall` (the robot's own joint limit). [F, CP/SP] It never fires at a knob limit, because the hand keeps yawing by compliance.
7. **Part angle.** θ_part = the hand yaw accumulated since the segment was classified free (the backlash transient is not counted) [I, conservative; F, SP: the steady knob rate 0.0207–0.0212 rad/step against a hand rate of 1.19–1.23°/step]. It is used for `quarter_turn`/`half_turn` and for θ_e.
8. **Stop:**
   - the effect onset θ_e, then continue to θ_e + max(0.1 rad, 0.25·θ_e) (K5);
   - or the end stop (K4w);
   - or a quarter or half turn;
   - or the yaw budget (`effect_not_observed` / `yaw_budget`).
9. **Release in place:** open, then retreat 3 cm along −R2. Post-check: no spring-back (K9) and the effect persists. Register `stays` and a keep-out around the part.

**Slide(part, sense).**
1. Grasp (frontal or top-down, as planned).
2. The DOF probe (part observables, K10) gives the free axis û. `open` = away from `mount_side`, `close` = toward it (both measured).
3. FollowAxis: an `AnchoredMove`-style guarded move along û (`contact_skills.py:577-730`). û is re-estimated every 5 steps from the part's own displacement (the external template of the part's rigid region; the wrist rigid patch must stay static relative to the hand). The lateral cap is relaxed so the mechanism steers the hand.
4. Stop at the first of:
   - the end stop K8: a hand stall (G12) after travel > 2× the probe, **with** the part's external template stopped as well;
   - `fits(p)` (M15);
   - a slip gap jump (G10) or a wrist-patch shift (the hand sliding off the part);
   - a travel budget of 0.30 m.
5. A stall before the expected travel with no finger touch means the arm, not the hand, is blocked: one retry pitched 45° (`frontal_pitched_45`), ≤ 2. [F] In KITCHEN_SCENE4 the forearm hits the cabinet top, and a 45°-pitched push closes the drawer (PA[0] rationale).
6. Release with v1's handle sequence: open, back off, lift 5 cm, restore top-down (`agent_speedup_v1.py:1002-1018`, `:1098-1133`).
7. Post-check: the part is static. Survey the exposed region. For `fits`, M18 is evaluated on the measured opening; a slide-axis shortfall re-grasps and pulls more (no call).

**LookAt(part)** (new controller for `w` = 0, `x` ≠ 0; [Q R-1v]).
1. `point_fine` (external), centred on the plan cell → u_p with ±8 px quantization.
2. **RayServo.** At a safe height z_s = max(current, the maximum known height + P11 clearance), move the hand horizontally (chained `WaypointReach` free segments ≤ 7.5 cm with v1's reach guards) until |u_grip − u_p| ≤ L2.
   - It uses J_ext's horizontal block with Broyden updates and HandLocator readings after every segment.
   - [I] The closed loop removes the affine-model error, which at 30 cm from the calibration pose is about 20% of the predicted shift (perspective at ≈ 1.1 m camera distance) [P].
3. **Local ray.** A ±3 cm vertical probe at the servo point refits J_ext locally; d = its null direction (downward).
4. **RayDescent.** A guarded descent along d (`AnchoredMove` along d; stall, touch and finger-touch events armed; P12 checked against the HeightField for the hand's swept volume).
   - Every 2 cm the FeatureTracker updates the HeightField along the ray from the wrist view. [geometry: ray points within ≈ 8.5 cm below the grip lie in the inner 30° wrist FOV]
   - It stops at the first of:
     - the ray meets a measured surface (HF ≥ ray z − 3σ_h; L3);
     - a contact event;
     - a P12 conflict;
     - the ray reaching the table plane + 5 cm.
   - [I] The external camera sees the first surface along its ray, and that ray is free of obstacles up to the part. The descent stays near the line of sight; only the hand body above it can meet overhangs, and those are guarded.
5. `point_coarse` on the aligned wrist image with the predicted hit marked. Answer checks: the answer must lie in the projected ray band (L5), and its κ depth must agree with the ray hit within L4.
6. A P12 conflict before the hit falls back to the **two-pose survey**: up to 2 top-down survey poses whose inner FOVs cover the remaining ray segment, with one `point_coarse` each.
7. Without R-1v but with R-2f: a horizontal scan along the measured image-up direction (`fit_external_axes`), with ≤ 3 views and one `point_coarse` each.

**Carry(held → above the target).**
1. Target: the destination target minus R·offset_h. The destination target is the cavity or surface centre, the mid-line on tight axes (M18), or the centre of the largest free disk of radius r_obj + max(5 mm, σ_offset) in the destination region when it is occupied (M16).
2. Path (TransitPlanner):
   - if the held footprint lies under measured overhang cells, move horizontally out first;
   - then lift so that the lowest held point (grip − b_low) clears the maximum height of the swept hand ∪ object footprint plus max(2 cm, 3σ_h) (M1);
   - then move horizontally; keep-outs are obstacles.
3. Moves: chained `WaypointReach` held segments of ≤ 7.5 cm (M2; the planning grid is ±8 cm, `contact_skills.py:945`). They use v1's held checks (`reach_limit`, `held_object_contact`, `hold_lost`; `contact_skills.py:850-1100`) and v1's held-orientation rule (≤ 15° reorientation, `agent_speedup_v1.py:65`). [F] Drawer 003's model-chosen carry reached 37.7 cm above the pick (D003 audit).

**AimCheck (before lowering).** The proprioceptive aim must agree with the image:
- the external n̂ row between the held object's external template and the destination region's centroid;
- when the destination outline is visible around the held object in the wrist view (`wrist_visible`), the wrist two-operand row: the destination servoed to c + κ_d·P·offset, using v1 `Align` with `offset_m` = the held offset and `allow_pointing=False`.
- A disagreement > 3σ re-runs the silhouette measurement once. A persistent disagreement is resolved in favour of the n̂ row for its component (closed-loop correction).

**GuardedLower.** v1's two-speed lowering (`agent_speedup_v1.py:1019-1068`), re-written:
- fast to 3 cm + 3σ_h above the expected support (M6), then slow; bound M7;
- support cues: a stall, a touch, or a gap rise (G8). The hand stall law is valid here because the held object moves rigidly with the hand (co-motion O3 verified). [F, CP] The free lowering ratio was 0.27; at the moka pot's support contact it fell to 0.154 and then 0.059, so v1's stall law fires within 2 steps.
- **drift invariant**: a lateral error (realized grip against the aim, plus the n̂ row when visible) beyond max(5 mm, 0.5·m_free) aborts, lifts 1 cm and re-aims (M5, ≤ 2); `lateral_unreachable` after 4 saturated correction steps.
- [F] Drawer 003's `lower_fast` drifted from x 0.0853 to 0.1167, with the correction pinned at its cap for 34 steps and no reaction (D003 skeptic).
- [F, CP] Continuing past the support stall tilted the moka pot to 17.7° and opened the gap to 18.95 mm, so the stop on the support cue is required.

**SupportClassify** (at the support event; bottom = g_event − b_low):

| Class | Test | Action |
|---|---|---|
| `on_surface` / `on_floor` | \|bottom − S_dest\| ≤ tol_h (M8); for `place_in`, S_dest = floor_z and bottom ≤ rim_z − M9; lateral error ≤ the axis tolerance (M4 on free axes, 5 mm on tight axes) + σ_offset | release |
| `perched` / `blocked_above` | bottom > S_dest + tol_h (for `place_in`, typically ≈ rim_z) | **do not open**; lift 1.5 cm, re-measure the in-hand pose, re-aim, ≤ 2; then repair `support_above_destination` |
| `missed_surface` | bottom < S_dest − tol_h | do not open; lift to clearance, re-aim, ≤ 2 |
| `ambiguous` | rim − floor < 2·tol_h | prevented by the M10 floor touch precondition; if it still occurs, release only if the lateral test passes, else repair |
| `no_support_within_bound` | bound M7 reached | lift, re-survey the destination, repair |

**Release, retreat, rest.** v1's `_release_and_back_off` pattern (open settle, retreat 3 cm along −R2; `agent_speedup_v1.py:986-997`), then the rest observer (M11, 10 frames) and the follow check: the object must not move with the retreating hand.

**PushTo(obj → spot).**
1. Touch the support beside obj, on the side opposite the spot, with closed fingers; the stall measures the support.
2. Lift 5 mm.
3. Push along (spot − obj centre), with co-motion as the contact invariant. Both lie on one support, so the external v̂ row has Δz = 0 and the external error is fully parallax-free (VS).
4. Stop at arrival (O6) or on lost contact; withdraw; rest.

**New primitives** (subclasses of v1's `Primitive`, `contact_skills.py:312-401`): `ParallaxProbe`, `LoomingProbe`, `Touchdown`, `SilhouetteProbe`, `DOFProbe`, `TwistSegment`, `HingeTwist`, `FollowAxis`, `GuardedLower` (an `AnchoredMove` subclass with the drift invariant), `PushAlong`, `ReleaseVerify`, `RayServo`, `RayDescent`. New observers: `WedgeObserver`, `PartRotationObserver`, `EffectWatch`.

### 4.10 Observers

All observers use RGB and state8 only.

| Relation | Progress (logged, drawn in video) | Completion decided by | Failure reasons |
|---|---|---|---|
| held | Align residual; grip height against the expected contacts; gap(k) during the close | G5 present; load test with gap ≥ G6; co-motion O3 of the object (part template in the wrist view, object template in the external view) | `empty_close`, `blocked_above_part`, `column_blocked`, `no_contact_where_expected`, `no_contact_within_bound`, `hold_lost`, `hold_jam`, `width_mismatch` (evidence), `grip_not_on_obj` (obj does not co-move), `precondition_unmet` (pose change > 2 mm / 2° before the close, spec v1 §6.5) |
| turned | θ_part; the signed gap rate and its early/late windows; part rotation in the wrist view; effect fraction | criterion met (effect onset + K5 under O7/O8, end stop K4w, quarter or half turn of θ_part); after release no spring-back K9 and the effect persists | `not_rotatable`, `sense_undecided`, `turn_slip`, `axis_tug`, `effect_not_observed`, `effect_region_invalid`, `yaw_budget`, `hand_yaw_stall`, `spring_back` |
| slid | travel along û; part co-motion (external template, wrist rigid patch) | criterion met; static after release | `no_free_axis`, `hand_slides_on_part`, `stall_before_expected`, `pull_gap_jump`, `spring_back` |
| supported_on | aim error of the object reference point (grip + R·offset_h) to the target; bottom height above S_dest | SupportClassify `on_surface`; then released and at rest | `support_above_destination`, `missed_surface`, `no_support_within_bound`, `lateral_drift`, `lateral_unreachable`, `held_object_contact`, `lost_in_transport`, `moved_after_release`, `object_follows_hand` |
| inside | as above, plus the per-axis clearance classes (M18) | SupportClassify `on_floor` with bottom ≤ rim − M9; released and at rest | `perched_on_rim`, `opening_too_small` (blocked axis), `blocked_at_opening`, `ambiguous_depth` |
| registered / aligned | image error to the setpoint | ≤ V2 for 3 frames with the hand still | `align_no_convergence`, `mark_lost`, `align_blocked` (v1 reasons), `part_not_found`, `pointed_not_<kind>`, `lookat_ray_blocked`, `lookat_inconsistent` |
| released | gap | gap ≥ g_open − 2 mm (M12); retreat with no touch; obj static | `object_follows_hand`, `touch_on_retreat` |
| at rest | frame difference over the object region | M11 for 10 frames | `object_moved_after_release` |
| upright (keep) | silhouette IoU and centroid; hand tilt against the grasp; external principal axis | O2 and O4 throughout | `object_rotated_in_hand` |
| clear_of (keep) | distance of planned swept volumes to the keep-out | no stall, touch or gap rise within the keep-out margin (O5); the part's external region unchanged at boundaries (hand pixels masked) | `keepout_contact` |
| stays | the relation's own observer (`hold_true`: change from the start baseline) | re-run at each boundary | `kept_relation_lost` → reopen |
| effect | changed-pixel fraction in the region in each visible camera | O1 under O7 and O8 | `effect_occluded` (> 50% of the region masked), `effect_region_invalid` |
| pushed_to | object centroid track; co-motion ratio | O6 and at rest | `object_not_following`, `object_rotating`, `overshoot` |

### 4.11 Escalation

Every failure first runs its local recovery (§4.8 table), with 2 local retries per relation (v1 `local_retries = 2`, `agent_speedup_v1.py:233`). Then one `repair` call carries the reason, the measured evidence and the thresholds.

Stuck alert: at least 10 of the last 12 relation outcomes failed, or a relation once met failed to re-establish more than 5 times. It is an alert only, with no cap (PA[1]).

### 4.12 v1 components: reused unchanged, wrapped, re-written, not reused

**Imported unchanged (read-only):**

| Component | Where | v2 use |
|---|---|---|
| `StepView`, `Primitive` | `contact_skills.py:259-303`, `:312-401` | every phase |
| `GripperCalibration` (touch threshold `:226-227`), `Calibrate`, `classify_close` | `:209-251`, `:497-553`, `:556-574` | wrapped by Calibrate+; online margin passed in |
| `GripperSettlePrimitive`, `HoldPrimitive` | `:403-445`, `:447-495` | close, open, settles |
| `AnchoredMove` (stall, touch, gap jump, gap rise) | `:577-730` | ladder, clearance, lowering, Touchdown, FollowAxis base, pushes, RayDescent |
| `VerifyHold` | `:732-772` | load test (floor G6 passed as `hold_margin_mm`) |
| `Orient` (gripper argument, rotation stall, pose drift) | `:775-847` | yaw and family changes, restores |
| `WaypointReach` | `:850-1100` | free and held segments, RayServo |
| `axes_to_quaternion_xyzw`, `quaternion_angle_rad`, `finger_gap_mm`, `HoldingActionAdapter` | `:147`, `:163`, `:170`, `:180-192` | geometry; every action, including twists |
| `setpoint`, `inplane_displacement`, `bar_point_check`, `NCCTrack`, `ImageJacobian`, `Align` (`both`, `across_bar`; always with `allow_pointing=False`), `WristPatchMonitor` (with an explicit `uv`) | `wrist_servo.py:62-66`, `:68-75`, `:83-114`, `:117-293`, `:295-432`, `:462-1237` (`allow_pointing` `:531`, `:621`, `:796`), `:1240-1290` | marks, κ math, alignment, co-motion |
| `encode_png_data_url`, `coarse_cell_center`, `fine_point_pixel`, `gravity_quarter_turns`, `rotate_quarter`, `rotate_pixel`, `unrotate_pixel`, `coarse_overlay`, `fine_overlay` | `mark_prompting.py:84`, `:99`, `:144`, `:157`, `:168`, `:173`, `:182`, `:211`, `:226` | pointing |
| `hand_image_motion`, `fit_external_axes`, `image_shift_for` | `external_axes.py:42-62`, `:79-113`, `:116-121` | J_ext probes, HandLocator mask, frame alignment |
| `jsonable`, `world_up_in_wrist_image`, `wrist_to_shoulder_m`, `tilt_from_straight_down_deg` | `agent_speedup_v1.py:123`, `:158`, `:168-171`, `:181-183` | summaries, stretch guard |
| `ordered_dumps`, `ordered_sse_transport` | `ordered_transport.py:24`, `:29` | wire order |
| `ChatGPTPlanHybridClient` | `hybrid_subscription.py:166` (import only) | base of the v2 client, constructed with `task_prompt` (§4.2) |
| v1 runner helpers (`_FinalizationSignals`, `redact_secrets`, `_write_traceback`, `NullRecorder`, `check_render_environment`, `termination_as_interrupt`, `_simulator_without_credentials`, `_safe_error`, `_append`, `configuration`, `digest`, `write_json`, `source_manifest`, `_validate_task_configuration`) | `scripts/evaluate_agent_speedup_v1.py`, loaded read-only (PA[1] runner architecture §5) | runner |
| `FakeWristSim`, `FakeLiberoSim` | `tests/agent_speedup_fakes.py:49`, `:239` | base of the v2 fakes |

**Re-written in v2 from v1 patterns** (they are methods bound to v1 policy state): `_calibrate` (`agent_speedup_v1.py:826-852`), `_descend_ladder` (`:1144-1198`), `_grasp_attempt_close` (`:1200-1210`), `_release_and_back_off` (`:986-997`), `_place_release` (`:999-1068`), `_restore_start_orientation` (`:1098-1133`), and the held-move back-off (`:903-940`).

**Not reused (task-specific)** [F]:
- `AgentSpeedUpV1Policy` and `AgentSpeedUpV1Client`: the closed registry `TASK_TARGETS` (`agent_speedup_v1_client.py:42-45`); the constructor check (`agent_speedup_v1.py:241-242`); the name rule `(strategy == "rim_pinch") == ("handle" in target)` (`:799-800`); `resolve_distance_task` in the client (`agent_speedup_v1_client.py:398`).
- `_grasp_rim`, `_grasp_bar` (`agent_speedup_v1.py:1226-1494`).
- `rim_target_phrase`, `bar_target_phrase` and the two-name appearance routing (`:190-209`).
- The fixed constants `RIM_CLEARANCE_M`, `BAR_CLEARANCE_M`, `RETRY_SHIFT_M`, `BLOCKED_OUTWARD_SHIFTS_M`, `PLACE_*`, `APPROACH_BOUND_M`, `SETPOINT_ROW_FRACTION`, `FINGERTIP_BAND_ROW_FRACTION` (`:42-63`, `:85`, `:104`).
- The pull range [0.02, 0.16] and the four world approaches (`agent_speedup_v1_client.py:41`, `:47`).

---

## 5. Threshold policy

### 5.1 The rule and its enforcement

1. **No task-specific constant.** For every number, ask (RECIPE line 8 §3.2): *would this number be the same for an unseen object in an unseen scene?* If not, it must be measured in the episode.
2. Every number the executor compares against is a `Threshold` with label ROBOT, ONLINE, DD or RULING and a derivation. It is logged in every outcome record that used it and in the repair request.
3. **DD values** are k × an ONLINE quantity, k × a ROBOT quantity, or a floor justified from **robot-only** traces: empty closes, open holds, closed-empty-hand motions, free motion, own-probe flow, static frames. They are never justified from object success labels.
   - V5 (§8.3) collects robot-only distributions on initial states 1–4 of the gate tasks **and** on held-out tasks (libero_goal 1, 6, 8; libero_10 tasks 3, 8).
   - Task outcomes are never an input to a DD change.
4. **T0 lint** (test):
   - every numeric module-level constant in v2 control modules appears in `CONSTANT_LABELS` with its class and derivation;
   - no token of the **lint vocabulary** appears in identifiers or strings of v2 control modules;
   - the prompt's worked example uses none of them;
   - nothing branches on the task sentence except passing it to the client.
   - *Lint vocabulary* = the underscore- and digit-split tokens of every object category, fixture category and region name in all LIBERO BDDL files (all suites), plus the words of every official task sentence, minus a checked-in allowlist of generic words that the executor legitimately uses (e.g. top, bottom, middle, side, front, back, left, right, region, open, close, on, in, turn, put, the, and). Both lists and their sha256 are in the manifest.
   - **D2 (decided 2026-10-18; lead default; user may override; M0 review TLR-5).** The vocabulary also contains a checked-in `PART_NOUNS` list (button, burner, door, handle, knob, lever, lid, spout): object-part nouns that no BDDL name contains, so that a task-shaped controller (`twist_knob`, `pull_handle`) fails the lint. `cap` is not listed: it is this spec's generic word for a command limit (R10 rotation cap, K2), and banning it would rename that ROBOT fact. The allowlist adds the generic programming and geometry words `box`, `bin`, `flat`, `layer` (bounding box, histogram bin, flat plane fill, HeightField layer); the object nouns around them (cream, cheese, stove, …) stay banned. `rim` stays allowlisted because it names the measured top edge of any cavity, a quantity the generic place-in code compares against (§4.8 inside, M9, M10, SupportClassify `perched_on_rim`, `rim_pinch`). The vocabulary, allowlist and part-noun lists and their sha256 are in the manifest (`agent_speedup_v2.lint`). The lint skips the definitions of these two lists in `agent_speedup_v2_thresholds.py` itself, and only there.
5. One constants table, hashed into the manifest; equality across all registry keys is tested (AC-1).
6. **Changes.** Any change to a DD value is a version bump.
   - Its justification must be a robot-only quantity (V5 or M0 traces).
   - It must pass the full gate-G7 scripted suite **and** the held-out scripted tasks of item 3 before a live run.
   - "DD violated → bump and rerun on the gate tasks" is not allowed (critic SHOULD 7).
7. Evaluator truth appears only in diagnosis and tests. The ORACLE pointing stub is test-only and asserted unavailable in live mode.

### 5.2 Threshold table

**ROBOT facts:** R1–R14 (§4.4).

**Gripper and contact (G).**

| ID | Threshold | Value / rule | Label | Origin |
|---|---|---|---|---|
| G1 | open gap g_open | Calibrate open hold (78.5 mm recorded on real physics; [F, CP] 79.95 mm at the knob hover) | ONLINE | `contact_skills.py:497-553` |
| G2 | empty-close curve gap(k) | Calibrate | ONLINE | same |
| G3 | finger touch | max(0.10 mm, 3σ_in) | ONLINE | `contact_skills.py:226-227` |
| G4 | σ_close | \|Δ gap(15)\|/√2 over 2 empty closes (a third if they disagree by more than the G5 floor) | ONLINE | Calibrate+ |
| G5 | object present | gap(15) > empty(15) + max(0.5 mm, 4σ_close) | ONLINE + DD floor | floor = v1's margin, set from the empty-close spread 0.35 mm (robot-only; spec v1 §6.5); replaces the TUNED-initial label (`contact_skills.py:556`) |
| G6 | hold floor | empty floor + max(0.5 mm, 6σ_close); co-motion O3 is required as well | ONLINE + DD | replaces TUNED-initial 1.0 mm (`agent_speedup_v1.py:106`); [F] drawer 003's bowl cleared v1's floor by 0.87 mm (D003) |
| G7 | held contact while carrying | rise ≥ max(1.0 mm, 6σ_carry) in total, or ≥ max(0.4 mm, 4σ_carry) per step; σ_carry from this held move's first 5 free steps; plus v1's stall and no-progress events | ONLINE + DD floors (provisional) | Floors = v1's values (`contact_skills.py:886-887`): 1.8× the recorded 0.55 mm clean-carry deviation and 2.7× the 0.149 mm/step noise. Those recordings are held-object traces, so **M0 re-derives the floors from robot-only traces** (closed-empty and open hands through the same free motions): floor = max(2 × the robot-only p99, 1.5 × the recorded values). |
| G8 | support cue while lowering | per-step rise ≥ max(0.3 mm, 4σ_lower) (σ from the lowering's first 5 free steps), or a stall, or a touch | ONLINE + DD floor (provisional) | floor = v1 `PLACE_GAP_RISE_MM` (`agent_speedup_v1.py:62`), 3.3× the recorded 0.09 mm drift; re-derived in M0 like G7. [F, CP] At the moka support contact the gap rose only 0.12 then 0.06 mm/step, so the stall fired first. |
| G9 | open-finger contact during reaches | \|Δgap\| ≥ max(1.0 mm, 10σ_in) from a settled open gap | ONLINE + DD (v1) | `contact_skills.py:890-891` (robot-only: settled open fingers drift ≤ 0.55 mm on free reaches) |
| G10 | slip gap jump (slides only) | \|Δgap\| ≥ max(0.2 mm, 5σ_carry) | ONLINE + DD (v1) | `AnchoredMove` `gap_jump_mm` 0.2 (`contact_skills.py:605`, `:698`). **Not used during twists**: [F, SP] the free-sense backlash transient alone gives 0.29–0.30 mm/step. |
| G12 | stall | 2-step realized/commanded < 0.6 × this move's free ratio, clipped 0.18–0.40 | ONLINE × DD (v1) | `contact_skills.py:584-597`; valid for the hand or a rigidly held object, never for an articulated part's motion |
| G13 | width consistency at the close | gap − 1.0 mm within [w_min − max(3σ_w, 3 mm), w_max + max(3σ_w, 3 mm)] of the planned cross-section; outside = `width_mismatch` (evidence; a failure only together with no co-motion) | ROBOT + ONLINE | R9; VS |

The first version's G11 (aperture levels) is removed with the deferred pre-shaped apertures.

**Depth and geometry (D).**

| ID | Threshold | Value / rule | Label | Origin |
|---|---|---|---|---|
| D1 | κ probe validity | shift ≥ 6 px, residual < 0.1·shift (floor 0.6 px), leg agreement 0.35 | DD (v1) | `wrist_servo.py:515-519` |
| D2 | hover probe amplitude | ±2 cm out and back along R1, then R0 | DD | disparity κ·0.02 ≥ 40 px for Z ≤ 0.17 m; v1's 2–6 mm probes gave 10–20% κ error (`wrist_servo.py:515` comment) |
| D3 | f̂ by looming | vertical ±4 cm; Z = d·s/(s − 1); f̂ = median(κ_i·Z_i); agreement with R3 within max(5%, 3σ) | ONLINE [Q R-1v] | §4.5 |
| D4 | σ of a κ height | σ_Z/Z = σ_κ/κ ⊕ σ_f/f, from fit residuals and `PIXEL_NOISE_PX` 1.5 (`wrist_servo.py:33`) | ONLINE | — |
| D5 | touch height | surface = grip z − t_f (R5, 9.3 mm); σ_touch = one stall window of realized motion | ROBOT + ONLINE | — |
| D6 | dense flow validity | forward-backward ≤ 0.5 px; gradient ≥ 2 grey/px | DD | G1 decides use |
| D7 | unknown cells | (a) plane fill at plane z + 3σ_plane when the cell is within 2 cm of the plane's known cells, has no D9 edge between them, and has \|ΔLab\| ≤ 3σ_plane against the plane colour; (b) otherwise an obstacle at max(known neighbours within 2 cm) + 3σ | DD × ONLINE | critic SHOULD 1 |
| D8 | same plane | \|Δh\| ≤ max(3 mm, 3σ_h) | DD × ONLINE | — |
| D9 | region edge | \|∇I\| > 4 × the local median, or a depth step > 3σ_h | DD × ONLINE | — |
| D10 | elongated | structure-tensor coherence ≥ 0.5 and w∥/w⊥ ≥ 1.5 | DD | coherence = v1 `BAR_CHECK_MIN_COHERENCE` (`wrist_servo.py:79`) |
| D11 | circle fit accepted | rms ≤ max(1.5 px, 3%·r) | DD | — |
| D12 | cavity | a low region bounded on ≥ 270° by cells ≥ floor + max(1 cm, 3σ_h); rim_z = the lowest boundary top; floor_z = the median inner height; principal axes and widths of the free polygon | DD × ONLINE | — |
| D13 | mount side | an adjacent region (within 3 cm) whose top ≥ part top + max(2 cm, 3σ_h) | DD × ONLINE | — |
| D14 | J_ext residual | 30° | DD (v1) | `external_axes.py:31` |
| D15 | texture test for features | patch std ≥ 4 grey (21 px) | DD (v1 `NCCTrack` `min_std`, `wrist_servo.py:134`) | [F] 54/143 grid patches pass at the task-2 start view |

**Contact planning (P).**

| ID | Threshold | Value / rule | Label |
|---|---|---|---|
| P1 | lateral error budget e_lat | 2σ_align + σ_w + 1.0 mm (σ_align = the Align residual std) | ONLINE + ROBOT |
| P2 | width, three-way | feasible: w + 2e_lat ≤ sep; infeasible: w − 2σ_w ≥ sep − 2 mm; else marginal (try-and-classify); sep = aperture gap − 1.0 mm | ROBOT + ONLINE |
| P3 | pad overlap | ≥ 8 mm (half the 16 mm pad) | ROBOT-derived |
| P4 | landing | each finger column (P16) has ≥ 70% known cells, and its highest measured surface is the landing; straddling an edge is allowed (the higher side) | DD |
| P5 | closing sweep | no height above landing + 3σ_h between each landing and the part edge | ROBOT + ONLINE |
| P6 | height-stop target and palm margin | g* = top − 11.6 mm − max(2 mm, 1.5σ_h); g* + p_min (31.0 mm) ≥ top + m_palm, with m_palm = max(5 mm, 2σ_h) for free objects and 0 for fixtures; both column landings ≤ g* − 4.4 mm − max(3 mm, 3σ_h) | ROBOT + ONLINE |
| P7 | hand-footprint clearance | max height in the footprint, **excluding the grasped part's region and the finger slot**, ≤ grip z + p_min − max(5 mm, 3σ_h) | ROBOT + ONLINE |
| P8 | contact classification band | expected height ± max(5 mm, 2σ_h) | ONLINE |
| P9 | approach bound | expected contact + max(3 cm, 3σ_h) | ONLINE (replaces 0.300 m) |
| P10 | clearance retreat | centres the pad band on [contact surface, part top], clipped 0–8 mm | ROBOT-derived |
| P11 | hover | fingertips ≥ max(5 cm, 3σ_h + 2 cm) above the part top | DD (v1 hovers of 5–10 cm worked, spec v1 §17) |
| P12 | turned-hand clearance | the swept hand volume clears the HeightField by max(1 cm, 3σ_h) | ROBOT + ONLINE |
| P13 | palm stop on a free object | never | rule |
| P14 | ladder | 20 mm segments, 3 settle steps | DD (v1, `agent_speedup_v1.py:45-46`) |
| P15 | guarded command / lateral cap | 10 mm / 3 mm per step | DD (v1 `AnchoredMove` defaults) |
| P16 | finger-descent column | each finger's lowest-point footprint at the planned aperture (robot_facts_v2.json; until M0, 38.2–46.2 mm × ±8 mm) inflated by e_lat, from the hover to the support | ROBOT + ONLINE |
| P17 | held-object visibility | prefer the 180°-twin yaw with R0·(held footprint centre − grip) ≥ 0, within R12 and P12; score +0.3 | ROBOT (camera mount R2) + rule |

**Motion and placement (M).**

| ID | Threshold | Value / rule | Label |
|---|---|---|---|
| M1 | transit clearance | lowest held point ≥ swept max height + max(2 cm, 3σ_h) | DD × ONLINE |
| M2 | held segment length | ≤ 7.5 cm | from the code's ±8 cm grid (`contact_skills.py:945`) |
| M3 | held reorientation | ≤ 15° | DD (v1, `agent_speedup_v1.py:65`) |
| M4 | aim tolerance (free axes) | max(5 mm, 0.5·m_free), m_free = r_free − r_obj (both measured) | ONLINE + DD floor |
| M5 | lowering drift | abort at max(5 mm, 0.5·m_free) (5 mm on tight axes); ≤ 2 re-aims | ONLINE + DD floor |
| M6 | fast-to-slow switch | 3 cm + 3σ_h above the expected support | ONLINE |
| M7 | lowering bound | down to S_dest + b_low − max(1 cm, 2·tol_h) | ONLINE (replaces clip 0.15–0.30, `agent_speedup_v1.py:1032-1037`) |
| M8 | support height | tol_h = max(1 cm, 3·√(σ_dest² + σ_b² + σ_touch²)) | ONLINE + DD floor |
| M9 | inside | bottom ≤ rim − max(1 cm, 2σ_rim) | ONLINE + DD floor |
| M10 | ambiguous cavity | rim − floor < 2·tol_h → empty-hand floor touch probe first | ONLINE |
| M11 | at rest | frame difference over the object region ≤ max(3 grey, 3σ_static) for 10 frames | DD × ONLINE |
| M12 | released | gap ≥ g_open − 2 mm | DD (v1 `OPEN_TOLERANCE_MM`, `agent_speedup_v1.py:105`) |
| M13 | retreat | 3 cm along −R2 | DD (v1 `RETREAT_M`) |
| M14 | Touchdown trigger | lift gap change > max(1 mm, 6σ_close), or IoU < 0.8, or template shift > max(4 px, 3σ_track), or `wrist_visible` false | ONLINE + DD |
| M15 | `fits(p)` (travel stop) | travel ≥ w_p + 2·max(5 mm, σ_aim), σ_aim = σ_offset ⊕ σ(dest) | ONLINE + DD floor |
| M16 | free target spot | centre of the largest free disk of radius r_obj + max(5 mm, σ_offset) in the destination region | ONLINE |
| M17 | offset σ fallback | IoU < 0.8 or `wrist_visible` false → σ_offset = max(measured, 0.5·r_obj) | DD |
| M18 | axis-specific fit | per cavity axis: free c_a ≥ 2·max(5 mm, σ_aim); tight −2σ_c ≤ c_a < that → compliant entry; blocked c_a < −2σ_c | ONLINE + DD floors |

**Mechanisms (K).** ℓ = pad length along R0 (16 mm, R4); ω = the measured hand yaw rate (rad/step); σ_g = the robot-only gap-rate noise under closed-hand yaw (Calibrate+).

| ID | Threshold | Value / rule | Label |
|---|---|---|---|
| K1w | twist sense (WedgeObserver) | 12-step segment, decided at step 12 (extend to 16); early = mean gap rate over steps 1–4 (after a reversal: from the first non-negative rate), late = mean over the last 4 steps; **free** if late ≤ max(0.1·ℓω, 4σ_g, 0.25·early); **blocked** if late ≥ max(0.5·ℓω, 0.75·early) in the windows ending at steps 11 and 12 | ROBOT × ONLINE × DD factors [F, SP validation: late 0.045 vs 0.445 mm/step; 0.1ℓω ≈ 0.032, 0.5ℓω ≈ 0.16; the free windows ending at steps 11–12 are 0.057 and 0.045 against a 0.195 blocked threshold] |
| K2 | twist increment | ≤ 0.1 rad/step command | ROBOT cap (0.15) + DD; [F, CP] cap 0.5 lost the lever |
| K3 | orbit follow | fitted hinge offset > max(3 mm, the Align tolerance in mm) | DD × ONLINE |
| K4w | rotation end stop | after a free classification: two consecutive 4-step windows with mean gap rate ≥ max(0.5·ℓω, 5 × the median rate since the free classification), or a part stop from K7 | ROBOT × ONLINE × DD; [F, SP] the steady rate was ≤ 0.004 mm/step, then 0.18 → 0.49 mm/step at the limit; fires 3 steps after it |
| K5 | turn margin after the effect onset | + max(0.1 rad, 0.25·θ_e) | DD |
| K6 | turn cap | the yaw budget R12 | ROBOT |
| K7 | part rotation / slip | part rotation in the wrist view relative to the hand ≥ max(3°, 3σ_θ); or a gap fall ≤ −0.5·ℓω for ≥ 2 steps outside an expected unwedge; or the gap reaching G6 | ONLINE + DD |
| K8 | slide end stop | stall (G12) after travel > 2 × the probe travel, **and** the part's external template stopped | DD × ONLINE |
| K9 | spring-back | part orientation or position change ≤ max(3°, 3 mm, 3σ) over 10 frames after release | DD × ONLINE |
| K10 | DOF probe (part observables) | rigid-patch shift relative to the hand ≤ max(2 px, 3σ_track) ⇒ moved with the hand; ≥ 0.5 × the hand's image shift ⇒ the hand slid on the part; external co-motion J_ext·Δp ± 3σ; ±1.5 cm probes | DD × ONLINE |
| K11 | axis tug | realized-versus-commanded lateral deviation > max(3 mm, 2·sin(θ/2)·σ_c + 3σ_pos) for 4 steps; σ_c = max(fit σ, 5 mm, measured close drag) | DD × ONLINE |

**Observers (O).**

| ID | Threshold | Value / rule | Label |
|---|---|---|---|
| O1 | effect | a pixel is changed if \|ΔLab\| > max(12 grey, 4σ_px); effect when the changed fraction outside the hand mask ≥ max(2%, 10 × the false-change rate measured in that region during Calibrate+) for 3 consecutive frames | ONLINE + DD (12 = v1 `DIFF_THRESHOLD`, `external_axes.py:25`) |
| O2 | rigid hold | silhouette IoU ≥ 0.8; centroid shift ≤ max(4 px, 3σ_track) | DD |
| O3 | co-motion | part template within 10 px in the wrist view (`WristPatchMonitor` default) and the external template moving by J_ext·Δp ± 3σ | DD × ONLINE |
| O4 | upright | hand orientation within M3 of the grasp and an in-hand tilt change ≤ 20° | DD ([F] the run-003 bowl hung 15° and was placeable, D003) |
| O5 | keep-out | the part region inflated horizontally by max(2 cm, 3σ_loc); vertically from its support to top + max(2 cm, 3σ_h) | DD × ONLINE |
| O6 | push arrival | object centre within max(1 cm, 3σ_track) of the spot | DD × ONLINE |
| O7 | effect region validity | the effect region must not overlap the turned or gripped part's region or the hand's projected swept volume in that camera, and its Calibrate+ false-change rate must be < 1%; otherwise `effect_region_invalid` and the criterion is dropped (rotation and stop criteria remain) | DD |
| O8 | minimum part rotation before an onset counts | θ_part ≥ 5° | DD |

**Pointing and tracking (V).**

| ID | Threshold | Value / rule | Label |
|---|---|---|---|
| V1 | NCC lost / re-template | peak < 0.6 for 2 frames; re-template 0.70 ≤ peak < 0.85 | DD (v1, `wrist_servo.py:129`; Align constants) |
| V2 | Align converged | ≤ min(6 px, 3 mm·κ), ≥ 2 px, 3 frames | DD (v1, `wrist_servo.py:850-857`) |
| V3 | re-acquisition | ±48 px around the prediction, peak ≥ 0.6, within 3σ | DD |
| V4 | re-points | 2 per part per view | DD |
| V5 | answer checks | §3.6 item 4 (finger-column surroundings, cavity ring, plane, structure tensor, ray band) | DD |
| V6 | fine crop | 128 px ×4, 8×8 points | DD (v1 mark protocol, `mark_prompting.py:20`) |
| V7 | FeatureTracker equivalence | \|Δuv\| ≤ 0.1 px against `NCCTrack` on recorded frames | test |

**LookAt (L).**

| ID | Threshold | Value / rule | Label |
|---|---|---|---|
| L1 | HandLocator consistency | displacement residual against J_ext·Δp ≤ max(3 px, 3σ) over the Calibrate+ moves; σ_loc = residual RMS | ONLINE + DD |
| L2 | RayServo converged | \|u_grip − u_p\| ≤ max(4 px, 2σ_loc) | ONLINE + DD |
| L3 | first surface on the ray | HF(ray point) ≥ ray z − 3σ_h | ONLINE |
| L4 | ray–depth consistency | the answer's κ 3-D point within 3σ (combined locator, pointing and κ σ) of the ray | ONLINE |
| L5 | answer ray band | the projected ray ± (external pointing quantization + σ_loc), mapped into the wrist image | ONLINE |

**Budgets (B).**

| ID | Value | Label |
|---|---|---|
| B1 | ≤ 2 model calls per `act()` | DD (v1, `agent_speedup_v1.py:37`) |
| B2 | 2 local retries per relation | DD (v1, `agent_speedup_v1.py:233`) |
| B3 | ≤ 2 reviews per episode | DD |
| B4 | settle quietness ≤ 0.03°/step and ≤ 0.3 mm/step for 2 steps, ≤ 14 steps | DD (v1 Align `SETTLE_*`) |
| B5 | stuck alert (alert only) | DD (v1 runner `_stuck_reason`, `scripts/evaluate_agent_speedup_v1.py:326`) |

### 5.3 Rulings: what changes for R-5, and the precise requests

**R-5 in v2.** The base is used only for v1's frontal stretch guard (`wrist_to_shoulder_m`, `agent_speedup_v1.py:168-171`): refuse frontal hovers beyond 0.77 m, and stop frontal moves at 0.785 m (`:95-96`). The model never receives a base direction. Rule:
1. At episode start, compute base_est = start_grip − Δ0 (R14).
2. If |base_est − R13| ≤ **4.5 cm**, the arena is the verified MountedPanda tabletop/kitchen mount, and R13 is used exactly (R-5 as granted; guards unchanged).
   - The gate is half the 9.0 cm distance to the nearest other LIBERO mount, the study MountedPanda (−0.75, 0, 0.912) [F, `t5_dump.json`]. That is 1.6× the 2.8 cm start-pose spread bound and 2× the largest measured distance (2.25 cm, task 2 ep 0).
   - [F] This covers task 2, bowl_plate, drawer_bowl and every libero_goal task (PA[0]: R-5 holds in the 4 kitchen tasks; libero_goal shares the tabletop scene, RECIPE line 8 §2.0).
   - [F] The study task 5's start-pose estimate is 9.2 cm from R13 and is correctly rejected. The first version's 3 cm gate left 0.2 cm of margin (critic).
3. Otherwise frontal families are infeasible (`frontal_needs_r5b`) unless R-5b is granted. Top-down plans keep v1's reactive `reach_limit` handling, which needs no base. [I] The backup task 7 needs only top-down grasps.

[F] Tightening the guard by the start-pose spread would have refused drawer live 001's working 0.759 m hover (`agent_SpeedUp_v1.md:723`). So R-5b below keeps a separate, tighter guard only for unverified arenas.

**Ruling requests** (to the user):

> **R-1v (needed for the primary depth path unless R-2f is confirmed, and always for LookAt's ray).** May the executor add ±4 cm vertical moves to its own probe set? It would use them, together with the lateral probes already allowed under R-1/R-1x, to (a) estimate per-feature depth by looming (image scaling about the principal point under motion along the optical axis), (b) estimate the wrist camera's focal length f̂ online from its own motion, and (c) fit the vertical column of the external-image Jacobian, whose null direction is the external camera's viewing ray. Nothing is supplied in advance; every value is measured from the robot's own motion.

> **R-2f (confirmation).** The brief lists f ≈ 333.6 px at 512 px under R-2, while v1's spec excluded projection through fovy (tier T1; `agent_SpeedUp_v1.md` §10 R-2 row and §14). May v2 use f as a ROBOT fact for metric depth Z = f/κ? With R-1v granted and R-2f refused, v2 uses f̂ only. With R-2f granted, f is cross-checked against f̂ (D3).

> **R-5b (needed only for frontal hands outside the verified mount).** In arenas where the start-pose-derived base disagrees with the granted (−0.66, 0, 0.912) by more than 4.5 cm, may v2 use base = start_grip − (0.4537, 0.0002, 0.2655) m and shoulder = base + (0, 0, 0.333) m? This is the robot's own start pose relative to its base: identical init_qpos in the LIBERO robot files, spread ≤ (1.1, 1.9, 1.7) cm over 10 resets. The frontal guards would be tightened by 2.8 cm (refuse beyond 0.742 m, stop at 0.757 m). Without it, frontal grasps are disabled there, and top-down grasps run unchanged.

**G0 is a hard stop unless R-1v or R-2f is granted** (critic SHOULD 8). Without either there is no metric depth from images. Scaling κ by touch would be the same measurement and would need the same ruling. Touch alone cannot register parts before contact. With R-2f but not R-1v, depth works but LookAt degrades to the horizontal scan (§4.9), and the drawer handle must be visible in the start wrist view or found by that scan.

---

## 6. Eval protocol and runner

### 6.1 Unchanged from v1

Taken from PA[1] (v1_eval_protocol, v2_runner_architecture), all [F] for v1:
- one fresh LIBERO episode per process; a new output directory; no resume;
- seed 7, settle 10, native 512 render and policy images, state8, `max_steps` None, `stop_on_success` True, overrides refused;
- raw LIBERO success is the only completion; the model never declares completion;
- record-only collisions (`stop_on_collision` False);
- manifest identity hash (configuration, rulings, constant labels, prompt and schema sha256, lint vocabulary sha256) and `frozen_source`;
- the frozen source covers all `src/gpt_field/*.py`, the v2 runner, v1's runner, and v1's `HELPER_SCRIPTS` (`scripts/evaluate_agent_speedup_v1.py:52-53`, copied at `:756-761`), re-read until the hashes are stable. **The manifest identity also lists the sha256 of each file in the user-modified helper chain** (`scripts/evaluate_bowl_hybrid_subscription.py`, `scripts/evaluate_bowl_subscription_close.py`, `scripts/evaluate_bowl_field_history.py`, `hybrid_subscription.py`, `direct_history.py`, `chatgpt_plan.py`, `observation.py`, `policy.py`, `libero.py`), so a change to any of them changes the identity (critic COULD 2);
- the printed operator stop command (`touch <out>/operator_stop.request.json`), checked at every complete-feedback boundary;
- a credential-free simulator worker in its own session; SIGINT/SIGTERM → user_stop; the same `subscription_error` / `infrastructure_error` mapping, deferred-signal finalization order and exit codes;
- official-prompt matching (`list_tasks()`, at reset and every step) and a fail-closed reset hash; the matched sentence is the client's `task_prompt` (§4.2);
- trace, video, manifest and summary artifacts; the same timing components, including `control_without_video_s`;
- the stuck alert, re-expressed over relation outcomes; R-4 off.

### 6.2 Runner architecture

`scripts/evaluate_agent_speedup_v2.py` loads `scripts/evaluate_agent_speedup_v1.py` read-only (as `SURVEY/infra/subprocess_route_test.py` did) for the helpers of §4.12, and replaces everything task- or collision-specific.

CLI: `--task` (v2 keys), `--episode-index 0..49`, `--expected-initial-observation-sha256`, `--no-video`, `--preflight | --sim-smoke [--smoke-script plumbing|relation] [--smoke-steps]`, `--store-dir/--profile`, `--sim-python`, `--libero-root`.

Simulator: `agent_speedup_v2_sim.py`, promoted from `SURVEY/infra/v2_worker_prototype.py`.
- `V2LiberoWorker(_LiberoWorker)` and `V2SimProcess(SimProcess)` reuse v1's `_SessionSimProcess` semantics through `command=` (PA[1] verified the JSONL route and the reset hash).
- Extra evaluator-only ops: `truth`, `settle_truth`, `answer_key`.
- The worker never edits `libero.py`. [F] PA[1]: `LiberoConfig` refuses `task_manipulation_v1` for libero_10 (`libero.py:83-84`), so the v2 worker uses `collision_check_mode='none'` and attaches its own monitor.

### 6.3 Task registry and reset hashes

`agent_speedup_v2_tasks.py`: `V2TaskSpec(key, suite, task_id, prompt, scene, frozen_reset_sha256: {episode: sha})`, with `resolve_v2_task()` raising on unknown keys. It holds no targets, regions or mount facts. The policy sees nothing from it; the client sees only the prompt, through the runner.

| Key | Suite, id | Episode-0 reset sha256 [F] |
|---|---|---|
| `bowl_plate` | libero_spatial 0 | `8626d3e51bf8618a2c2793dbac9121dc8c87106197b95a7a6e7f483ffed65835` (`scripts/evaluate_agent_speedup_v1.py:142-145`) |
| `drawer_bowl` (= libero_goal 3) | libero_goal 3 | `c6853a2d01f24753fce9fe3b06dd05f78d5d667c258fd048144c908f29321eb8` (same) |
| `libero_10_t0` … `t9` | libero_10 0–9 | measured by PA[1] (`SURVEY/infra/out/t{0..9}_v2.json`); task 2: `7b854d9bd19a0d91a843980142f15b6586416ff51790e447b044d6c89d1eceef`; task 7: `5ed858a8309893828ce7646d9187890c96345bfd9f5354e0f2815fe6ff565c2c` |
| `libero_goal_g0` … `g9` (g3 aliases `drawer_bowl`) | libero_goal 0–9 | measured in M0 |

Other episodes stay fail-closed until measured. [F] PA[1] did not reproduce reset-count nondeterminism in its worker for episode 0, but it measured only episode 0.

### 6.4 Generic contact monitor (record-only)

`agent_speedup_v2_contacts.py` promotes `SURVEY/infra/generic_contact_monitor.py`.
- Roles come from `objects_dict`, `fixtures_dict`, the gripper's `important_geoms` and the MuJoCo body/joint tree, with no names. [F] There were 0 unmapped contacts in all 10 libero_10 tasks.
- Classes:
  - forbidden: `robot_scene`, `robot_nonfinger_manipulated_object`, `robot_object`, `manipulated_object_scene` (normal > 35° from vertical);
  - allowed: `finger_object`, `finger_articulated_part`, `manipulated_object_support`;
  - ignored: internal and passive.
- An optional evaluator-only goal-aware tier is built from BDDL atoms.
- Mode `generic_contact_v2`. `_collision_record_v2` keeps v1's invariants: enabled; bool detected/libero_success; `stop_on_collision` False; `record_only`; success == libero_success; a collision never terminates.
- For `bowl_plate` and `drawer_bowl`, the v1 monitor also runs chained in record-only mode, for comparability. [F] PA[1]: replays agree on first events 340 / 129, with 41/41 and 99/101 steps.
- Cost ≈ +5 ms per step [F].

### 6.5 Diagnosis-only physical truth

At finalization, after the terminal status is fixed and before `sim.close`, the runner calls `truth` (terminal) → `settle_truth` (release-and-settle counterfactual: open 10, lift 8, settle 20) → `answer_key`. Each runs through `guarded()`, so a failure is a `cleanup_error`.

Verdicts use PA[2]'s strict kinds (`SURVEY/grounding/truth_lib.py`, promoted to `agent_speedup_v2_truth.py`):
- `inside`: not held, at rest, supported from below by the cavity or its contents, lowest point ≤ rim − 10 mm, not perched, below-rim vertices inside the inner walls;
- `on_object`, `on_fixture_region`, `on_table_zone`;
- `articulation`: the LIBERO threshold, no robot contact on the part, the joint at rest.

**D1 (decided 2026-10-18; lead default; user may override; M0 review TRUTH-1 / TRUTH-4).** Orientation and perches:
- strict `inside` does **not** require the object's reset orientation. An object resting in the cavity with floor (or contents) support counts even when it leans on a wall or the rim, as long as it is not perched. *Not perched* = at least one **support** contact (below the COM, normal within 35° of vertical) from the cavity or its contents lies ≥ 5 mm below the rim, or the cavity has a ceiling. The run-003 corner perch (all contacts at rim height, no floor contact) stays `lenient_success`; the front-wall lean with a floor contact is `physical_success`, and so is an inverted bowl on the drawer floor;
- strict `on` (`on_object`, `on_fixture_region`, `on_table_zone`) **does** require `orientation_kept`: the object's body z axis within 30° of its reset z axis. A tipped, side-lying or upside-down pot on the burner fails (`lenient_success` when LIBERO's raw On holds); an upright pot passes. A held object's tilt is deferred to the release-and-settle, like the other hold consequences;
- the evidence always carries `tilt_change_from_reset_deg` and `orientation_required`, and for `inside` also the share of the container normal force carried at rim height and the support contacts below the rim. The switch is `STRICT_ORIENTATION_KINDS` (`agent_speedup_v2_truth.py`). Recorded cases: `scripts/agent_speedup_v2_truth_probes.py` → `runs/agent_speedup_v2_m0close_truth_probes/`. Drawer 003 → `lenient_success` and bowl 002 → `physical_success` (`runs/agent_speedup_v2_m0close_truth_{drawer003,bowl002}/`).

Outputs:
- `physical_truth.json`;
- `summary.physical_truth = {at_terminal, after_release_settle, physical_success_at_terminal, physical_success_after_settle, diagnosis_only: true, affects_completion: false}`;
- `answer_key.json`;
- `parse_check.json`: plan relations and parts against the BDDL atoms (`turn`+`effect` ↔ TurnOn, `place_on` ↔ On, `place_in` ↔ In or On(object), `slide open/close` ↔ Open/Close, `push_to` ↔ On(table zone)).

[F] PA[1] validated these: drawer 003 `perched_or_partial`; bowl 002 `held` → `resting_on_target`; teleport controls `resting_inside` and `outside`.

**Observer-vs-truth confusion table** (offline). Every relation verdict's step is in the trace. `SURVEY/grounding/truth_check.py` (promoted as a script) replays the run bitwise and evaluates truth at those steps, giving v2-verdict × truth counts per relation kind. The WedgeObserver's sense and end-stop verdicts are scored against the knob joint, and LookAt's hit against the part's geometry. It never runs in the control loop.

### 6.6 Modes and test tools

- `--preflight`: a synthetic 512 observation; prints the plan, fine, coarse and repair payloads for any key, with one instruction text and the key's official `task_prompt`. No credentials, network or simulator.
- `--sim-smoke plumbing`: the real simulator with `ScriptedV2Transport` envelopes, zero model calls, unscored.
- `--sim-smoke relation`: a scripted plan plus the ORACLE pointing stub. The stub is test-only: it reads evaluator geometry only to choose scripted cells and points, and the runner refuses it in live mode (tested).
- `scripts/agent_speedup_v2_replay.py` (promoted from `SURVEY/infra/replay_v1_trace.py`): a bitwise replay of a v1 or v2 `trace.jsonl` to step k (state and observation hash asserted), then one v2 relation with ORACLE or recorded pointing. It is a diagnosis and test tool, never a scored run.

### 6.7 Trace, video, timing

- **Trace events:** v1's names plus `relation_plan`, `plan_validation`, `step_start`, `observer`, `wedge` (gap-rate windows and verdicts), `lookat`, `step_end`, `pointing`, `repair`, `review`, `model_claim`, and `threshold_set` (the constants table, once per run).
- **Video overlays** (`agent_speedup_v2_video.py`): the active step and relation; observer values and verdicts; reprojected parts; keep-outs; the planned contact and its expected heights; the ray during LookAt.
- **Timing:** v1's components, with `model_pipeline_s` split by request kind, and a `policy_compute_s` component.

### 6.8 Speed items outside the policy (deferred, need a user decision)

| Item | Saving | Why deferred |
|---|---|---|
| Video rendered off the control loop | [F] about 106 s in drawer 003 (D003 audit) | protocol-neutral for observations, but it changes timing semantics; user decision |
| PNG encoding level 1 | [F] about 38 s in drawer 003 | changes the observation bytes and therefore every frozen reset hash; the encoding lives in user-modified `observation.py`; needs re-freezing and a user decision |
| Plan call overlapped with Calibrate+ (asynchronous call) | [P] about 15–30 s per episode (the whole plan call) | needs runner support for overlapping call accounting; protocol decision |

---

## 7. Walk-throughs

Scene facts marked [F, rationale] come from evaluator geometry, oracle runs or the CP/SP probes. They are used here only to show the size of the checks and are never policy input.

### 7.1 libero_10 task 2, "turn on the stove and put the moka pot on it"

**[F, rationale]** (PA[0] task 2; oracle runs; CP; SP):
- **Knob.**
  - Hinge about +z at (−0.1964, 0.2061, 0.905); range [−0.005, 2.1]; damping 1.0; no return spring; TurnOn at q ≥ 0.5 (`flat_stove.xml:33-36`).
  - Disk ≈ 47.6 mm radius (9.5 cm AABB), top 0.918, centred (−0.2008, 0.2074), 4.6 mm from the hinge (`flat_stove.xml:41-54`; `t2_dump.json`).
  - Lever 24.3 mm wide, spanning the disk along world x, top 0.955 (`flat_stove.xml:55`; PA[0]).
  - The knob stands on the table (0.900) beside the stove body (`frames/t2_wrist.png`).
- **Burner plate:** top 0.9305, 19 × 19 cm, centred (−0.046, 0.206); cook region ±7.5 cm.
- **Moka pot:**
  - At (0.030, −0.007), 68 g.
  - Handle loop 1.3 cm thick in x, z 0.956–1.035, about 6 cm toward −y. At the grasp y the outer loop segment spans z 0.967–1.0345 (critic, `t2_dump.json` moka geoms g13/g14).
  - Body top 1.0331. The handle root and spout tops are 1.032–1.033 (g9, g11, g12; critic).
- **Frypan:** handle reaching y −0.013.
- **Start:** grip (−0.2014, 0.0187, 1.1893), q7 0.8177 (`t2_dump.json`). The camera is about 0.386 m above the table.
- **Oracle:**
  - Lever closed at gap 25.3 mm from a scripted grip of 0.9325 (`oracle_t2_a.json`).
  - Handle closed at gap 14.2 mm at grip (0.0335, −0.0886, 1.0146); pot − grip = (−0.002, +0.080); over-burner grip (−0.043, 0.130); lowered to grip 1.0481; success at the release, step 467 (`oracle_t2_b.json`).
  - A body pinch with the start jaw (fingers along y) closed empty at 1.5 mm, with the fingers stalled on the handle root and spout (critic; `oracle_t2_a.json` grip 1.0488).

| # | Calls | Executor | Observers and decisions [P unless marked] | Steps [P] |
|---|---|---|---|---|
| 1 | 0 | Calibrate+ | g_open; empty curve; σ_close from 2 closes; closed-hand yaw probe → σ_g and the free yaw rate ([F, SP] 1.31°/step at K2, gap noise ≤ 0.003 mm/step); J_ext 2×3 (5 moves); f̂ by looming; HandLocator σ_loc; σ_static and false-change rates; fingertip band. Frame alignment k = 2. Start HeightField: table ≈ 0.900 (with large plane-filled areas, D7a), knob disk ≈ 0.918, lever ridge ≈ 0.955, burner plate ≈ 0.93, pot top ≈ 1.05 at the image edge (larger σ) | ≈ 190 |
| 2 | **1** plan | §3.9 example: `turn(A until effect(B) or stop)`, `place_on(D via C → B, keep upright D, clear_of E)` | validator passes; parts A–E seeded from the plan cells; B registered as a flat plate region (empty hand, start view); O7 for B: B's region is disjoint from A's and from the hand's projected twist volume in the external view → valid | — |
| 3 | 0 | Hover above A (fingertips ≥ 5 cm above the lever top) | — | ≈ 25 |
| 4 | **2** fine(A) | ParallaxProbe ±2 cm; registration | lever elongated along world x (D10), width ≈ 24 mm, top ≈ 0.955 (σ_h ≈ 2–3 mm at the hover); disk circle fit (black disk on light wood) → radius ≈ 47.6 mm, centre c₀ with σ_c = max(fit σ, 5 mm) (the visual centre is 4.6 mm from the hinge [F]); `rigid_patch` = the most textured disk or lever patch ≥ 15 mm from the grip point, or `unmeasurable` | ≈ 30 |
| 5 | 0 | **Plan the lever grasp** | `top_down`, jaw ⟂ lever = the start jaw (no Orient). *Full opening, `touch_stop`* (ranked first): the finger columns (P16, 35–49 mm from the axis with e_lat) straddle the disk edge; the highest surface in each is the disk/ring top ≈ 0.918 (P4) → expected landing at grip ≈ 0.918 + 0.0093 ≈ 0.927 ([F, CP] stall at 0.9272); P10 retreat ≈ 5.6 mm → grip ≈ 0.933, pads [0.928, 0.944] → 16 mm overlap with [0.918, 0.955] (P3 ✓); the closing sweep runs over the flat disk top, 5.7 mm under the fingertips (P5 ✓); palm 31.0 mm above the grip, far above the lever (fixture, palm margin 0). *Full opening, `height_stop`* (second): g* = 0.955 − 0.0116 − max(2, 1.5σ_h) ≈ 0.939; pads [0.935, 0.951] ✓; column landings 0.918 ≤ 0.939 − 0.0044 − 0.003 ✓ ([F, CP] closed at 25.39 mm from 0.9359). Grip point along the lever = the projection of c₀ | — |
| 6 | 0 | Align `across_bar` (`allow_pointing=False`) → descent → touch on the disk (inside band P8) → retreat → close | close gap ≈ 25.4 mm [F, CP/SP] = 24.3 mm lever + 1.0 mm (R9; G13 ✓) ≫ empty curve (G5) | ≈ 60 |
| 7 | 0 | **First twist segment** (+ about the approach) with the WedgeObserver | [F, SP] the knob starts at its lower limit, so + is the free sense: the gap rises 25.39 → 27.10 mm with the rate decaying to 0.004 mm/step; late 0.045 ≤ max(0.032, 0.25·0.26) → **free** at step 12 (q 0.134 rad, diagnosis). Wrong-first-sense branch (the trace records the sense; G4 forces it): windows 0.42 and 0.445 ≥ max(0.16, 0.23) → **blocked** at step 12 (gap +4.2 mm, lever held) → reverse; unwedge ≈ 13 steps (gap back to ≈ 25.3 mm), then the + segment decides free 12 steps after the unwedge ends (q ≈ 0.22, diagnosis) | 12 (free) / ≈ 37 (wrong first) |
| 8 | 0 | HingeTwist in the free sense | soft hold; the hand orbits by compliance ([F, CP] 12 mm off the hinge, no tug at 1.2 rad); circle refit every 5 steps; θ_part counted from the free classification; steady knob rate ≈ 0.021 rad/step [F, SP]; EffectWatch on B (external, hand-masked): [F] red fraction 0% at q 0.49 and ≈ 20% at q 0.6 → onset near q 0.5, valid after θ_part ≥ 5° (O8); θ_e (conservative, the transient not counted) ≈ 0.37 rad → stop at θ_e + 0.1 ≈ q 0.6. K4w and K7 armed; yaw budget R12 | ≈ 25 |
| 9 | 0 | Release in place, retreat 3 cm | no spring-back; the effect persists → `turned`; `stays` registered (the effect region still changed against its pre-twist state) plus a keep-out around the knob | ≈ 25 |
| 10 | 0 | Hover above C | re-acquire by prediction; Orient to put the jaw ⟂ the loop's long axis (world y), i.e. the jaw along x. Of the two 180°-twin yaws, P17 picks R0 → +y, which puts the camera on the pot's side ([F, CP] pot − grip = (−0.0001, +0.0795)). From the post-twist hand yaw (≈ +40–45°: the part angle plus the backlash transient) that is a turn of about +45–50°; the twin (R0 → −y) would put the pot ≈ 59° off-axis (critic) | ≈ 55 |
| 11 | **3** fine(C) | ParallaxProbe; registration | loop top ≈ 1.035 (σ_h ≈ 2 mm at ≈ 0.15 m camera distance); width ≈ 13 mm; elongated along y; answer check: the loop stands ≥ 3σ_h above its finger columns (table 0.900 at ±40 mm in x) ✓, although the pot body beside it is level (1.0331) | ≈ 30 |
| 12 | 0 | **Plan the handle grasp** | grip list [C]. *touch_stop:* the columns land on the table 13.5 cm below the loop → no pad overlap → rejected. *palm_stop:* free object → forbidden (P13). *height_stop:* g* = 1.0345 − 0.0116 − 0.003 ≈ 1.020 (oracle 1.0146; [F, CP] closed at 14.11 mm from 1.0219); palm (31.0 mm hull) at ≈ 1.051 ≥ 1.0345 + 5 mm ✓ (P6, 11 mm margin). [I] The feasible window is g ∈ [1.0085, 1.031] (pads overlapping the 0.967–1.0345 loop by ≥ 8 mm, palm ≥ top + 5 mm), so it tolerates σ_h up to ≈ 7 mm. P7: the footprint ±31.6 mm along y (R0) stays ≈ 15 mm outside the pot body; the handle and the finger slot are excluded; textureless table cells beside the handle are plane-filled (D7a) → ✓; frypan handle below; keep-out E ✓. [I] *If the model had listed D as the grip:* with the jaw along y, the columns land on the spout and handle-root tops (1.032–1.033), so P3 fails → refused before any descent (`column_blocked`); with the jaw along x (66 mm), P2 is feasible for e_lat ≤ 5.75 mm and the columns reach the table → a valid `touch_stop` candidate (untested; M2 smoke) | — |
| 13 | 0 | Align → descent to g* → close → VerifyHold lift 2.5 cm | gap ≈ 14 mm (G5, G13); co-motion: the pot (D) moves with the hand in both views (O3), so the grip is on the pot (`grip_not_on_obj` otherwise); [F, CP] the pot is held rigidly | ≈ 50 |
| 14 | 0 | HeldFrame | `wrist_visible` true (P17); prior offset_h ≈ (0, +0.080) m; b_low ≈ g_close − 0.900 ≈ 0.120 m; the lift shows no gap change and IoU ≥ 0.8 → no Touchdown | ≈ 10 |
| 15 | 0 | Carry | swept obstacles: the knob keep-out (0.955 + 2 cm), burner 0.9305 → lowest held point ≥ ≈ 0.995 → grip ≥ ≈ 1.115 (oracle carried at 1.16); target grip = burner centre − R·offset ≈ (−0.046, 0.126) (oracle (−0.043, 0.130)); path clear of E and the knob | ≈ 35 |
| 16 | 0 | AimCheck → GuardedLower | n̂ row consistent; m_free ≈ 9.5 − 4 cm ≫ M4; [F, CP] stall within 2 steps of contact (ratio 0.27 → 0.154 → 0.059) at grip ≈ 1.0516; bottom ≈ 1.0516 − 0.1219 = 0.9297 against 0.9305 → `on_surface` (M8) | ≈ 30 |
| 17 | 0 | Release → retreat → rest | LIBERO TurnOn ∧ On fires at the support contact or the release ([F, CP] On() turned true at the contact step, before the stall). The pot's hover allowance is 10.9 mm by PA[0]'s band with CP's resting origin 0.9966, against 16.4 mm in PA[2] [F, the sources disagree; M0 re-measures]. The truth check reports `on_fixture_region` at the terminal state or after release-and-settle | ≈ 20 |

**Totals [P]:** 3 calls (≈ 21–44 s model time), ≈ 650–800 steps (≈ 700–830 in the wrong-first-sense branch), `control_without_video_s` ≈ 100–150 s. Within the AC-4 nominal target.

**Failure branches** (no task code):
- Fine pointing hits the frypan handle near the pot handle: the load test shows no co-motion of D → `grip_not_on_obj` → release and `repoint` (local), then repair.
- The touch lands outside the band (the disk fit was off): `no_contact_where_expected` → re-register, then `height_stop`.
- The effect is never seen (occlusion, lighting): the twist continues to the yaw budget (part rotation ≈ 1.7 rad ≥ 0.5) and reports `effect_not_observed`. TurnOn is established if, and only if, K1w chose the free sense. A wedge misread ends as `turn_slip` or `not_rotatable` evidence, never as a silent "turned".
- Slip during the twist (K7): re-grasp at the same part and continue from the measured angle (≤ 2).
- A later motion bumps the knob below q 0.5: the `stays` check at the next boundary sees the effect region revert → the turn is reopened.

### 7.2 Drawer run 003 replayed through v2: how the perch is prevented

**[F] What v1 did** (D003):
- 967 steps, 15 decisions + 6 pointing calls, 132.2 s of model time.
- The handle grasp worked (bar gap plateau 17.48 mm; pull 0.1198 m, drawer q −0.122). But the low model-chosen hover swept the bowl (70% of 4,078 forbidden points).
- The rim pinch passed hold_pass. The close dragged the bowl 2.6 cm along +y, and the bowl hung ≈ 15° with its origin ≈ 3.7 cm in −y of the grip.
- Five transports reached 37.7 cm above the pick. Decisions 19–20 added +7.0 and +2.75 cm in x from external-image parallax. `lower_fast` drifted +3.15 cm.
- A gap rise at grip z 1.1575 released the bowl. It settled on the wall tops g12/g14 (≈ 1.1202), 56 mm above the floor (1.0642). LIBERO In() passed by a 1.16 mm margin.
- Counterfactual drops centred, within ±3 cm in x, or ±1.1 cm in y of the opening centre landed flat on the floor; a flat drop has only 1.6 mm of total y clearance.

| # | Calls | Executor | What changes against run 003 |
|---|---|---|---|
| 1 | 0 | Calibrate+ (with HandLocator); start HeightField: table, cabinet as a tall block, the bowl as a ≈ 5 cm-tall circular footprint (r ≈ 5.3 cm) | — |
| 2 | **1** plan | §3.9 drawer example (A and D have `w` = 0) | no metric fields; `fits(B)` couples the opening to the measured bowl |
| 3 | **2** fine(A, external) | crop centred on cell 21 → the handle's external pixel ± 8 px | the external image gives only a viewing ray, never a metric target |
| 4 | 0 | LookAt: RayServo at a safe height onto the handle's ray (HandLocator + J_ext), local vertical probe → ray direction, RayDescent along the line of sight until the first measured surface on the ray (L3) | the hand reaches a view of the handle without any model direction or distance |
| 5 | **3** coarse(A) | the aligned wrist image with the predicted hit marked; ray band (L5) and ray–depth (L4) checks | a wrong answer off the ray is refused geometrically |
| 6 | **4** fine(A) at the top-down hover | registration: bar elongated (coherence ≈ 0.96 in v1's bar check), top ≈ 1.09, `mount_side` toward the taller cabinet block (D13); `rigid_patch` = a textured patch of the drawer face beside the bar | the approach side is measured, not chosen by the model |
| 7 | 0 | Plan: `top_down` rejected (P7: the hand footprint over the cabinet top); `frontal`, jaw vertical, `face_stop`; frontal hover computed level with the bar; the turned-hand sweep (P12) checked against the bowl's measured top (≈ 0.952): lowest hand point ≈ 1.089 − 0.105 ≈ 0.984 clears it | **removes the bowl sweep** (the run-003 hover at z 0.982 puts the hand's lowest point near 0.877, below the bowl top, and is refused) |
| 8 | **5** fine(A, frontal) | Orient frontal (gravity-turned view) → fine pointing with the bar check (measured direction) → Align `across_bar` → face stall → clearance (P10) → close (≈ 17.5 mm, G13 consistent with the bar width) → DOF probe by part observables (K10): on the pull probe the drawer-face patch stays fixed relative to the hand and the external face template co-moves → prismatic; on the lateral probe along the bar the face patch shifts → `hand_slides_on_part` = lateral not free | v1 needed 3 reaches and a `hover_off_bar_across` for this; the first version's ratio-based DOF probe could misread pad sliding as free |
| 9 | 0 | FollowAxis open until `fits(B)`: travel ≥ 10.7 + 2·max(0.5, σ_aim) cm ≈ 11.7–12.5 cm, or K8 (stall with the face template stopped) | the travel comes from the measured bowl, not a layout window |
| 10 | 0 | Release (v1 handle path: open, back off, lift 5 cm, restore top-down) | — |
| 11 | **6** coarse(D) | survey over the opened drawer with an **empty hand**: cavity floor ≈ 1.064, rim ≈ 1.120, free polygon with principal axes; rim − floor ≈ 5.6 cm ≫ 2·tol_h, so no floor touch is needed. **M18:** along the slide axis the opening (≈ 20 cm) is free for the 10.7 cm bowl; across it the clearance (true 1.6 mm, measured ± a few mm) is **tight** → compliant entry, not `opening_too_small` | the first version's precondition would have refused this drawer forever |
| 12 | 0 (+1 if the circle fit fails) | Grasp the rim: the planner picks the rim point and side by clearance (P7) against the now-open drawer front and the neighbouring objects; P17 picks the yaw that keeps the bowl in view; `touch_stop` with the inner column on the bowl floor (expected band P8) | a rim point under the overhanging drawer front is refused before descending (drawer 002's 13 stalls) |
| 13 | 0 | HeldFrame: re-localization after the close captures the 2.6 cm drag; the lift's gap change (11.9 → 4.1 mm in run 003) exceeds M14 → **Touchdown** measures b_low by touch (≈ 3.7 cm, including the 15° hang); σ_offset inflated | v1 measured none of these |
| 14 | 0 | Carry: lowest held point ≥ max(front 1.1224, rim 1.12) + 2 cm → grip ≈ 1.18 (run 003: 1.29); target = cavity centre (mid-line across, centre along) − R·offset_h | one planned path instead of 5 model transports |
| 15 | 0 | AimCheck: the external n̂ row between the bowl template and the cavity centroid is parallax-free; the v̂ row is used only with the measured height difference | run 003's +9.75 cm parallax overshoot shows up as a large n̂/v̂ residual and is corrected before lowering |
| 16 | 0 | GuardedLower with the drift invariant at max(5 mm, 0.5·m_free) on the free axis and 5 mm on the tight axis | run 003's 3.15 cm drift aborts within a few steps and is re-aimed |
| 17 | 0 | **SupportClassify.** A rim contact at grip 1.1575 gives bottom ≈ 1.1575 − 0.037 ≈ 1.120, against the floor 1.064 ± tol_h (≈ 1–2 cm) → `perched` → **no release**; lift 1.5 cm, re-measure, re-aim (≤ 2). A floor contact gives bottom ≈ 1.064 → `on_floor`, bottom ≤ rim − M9 → release | the decisive barrier |
| 18 | 0 | Release → retreat → rest | LIBERO In() fires; truth `resting_inside` |

**Four run-003 causes, four barriers:**
- unmeasured in-hand offset and tilt → re-localization and Touchdown;
- model aiming from the external image → measured cavity centre, proprioceptive aim and the parallax-free n̂ check;
- unchecked lowering drift → a drift invariant with a floor;
- no check of where support was met → SupportClassify against the measured floor and rim.

The last barrier alone refuses run 003's release. M18 makes sure the tight drawer is entered, not refused.

**Totals [P]:** 6–7 calls (≈ 29–58 s), ≈ 1,000–1,150 steps, `control_without_video_s` ≈ 150–195 s (run 003: 21 calls, 132.2 s, 967 steps, 234.5 s).

### 7.3 bowl_plate (libero_spatial 0)

Plan: `place_on(A whole bowl, grip [B rim], ref C plate, keep upright A)`. The identity "between the plate and the ramekin" is resolved by the model's cells.

The executor:
1. registers the plate (empty hand);
2. grasps the rim by the planner (`touch_stop` on the bowl floor; P17 yaw);
3. measures the in-hand pose;
4. carries the bowl over the plate;
5. accepts support at the plate's measured height (19 mm above the table).

[I] Height alone barely separates "on the plate" from "on the table", so the aim tolerance M4 (plate 13.75 cm against bowl 10.7 cm: m_free ≈ 1.5 cm) and the AimCheck do the work. [F] LIBERO's On(object) needs the origins within 3 cm (PA[2]).

Projected: 1–2 calls, ≈ 470–550 steps, `control_without_video_s` ≈ 80–100 s.

### 7.4 Coverage: the 10 libero_goal tasks (one scene: wooden cabinet, flat stove, wine rack; bowl, plate, cream cheese, wine bottle)

| goal | Sentence (BDDL atom) | Plan | Measured primitive | Completion | Coverage |
|---|---|---|---|---|---|
| 0 | open the middle drawer (Open, q < −0.14, 87.5% of travel) | `slide(handle, open, until stop)` | `frontal` fails P12/P7 near the table (the handle is ≈ 10–12 cm above the table, inside the turned hand's ±10.4 cm extent; RECIPE line 10 §2.E) → `frontal_pitched_45`; DOF by part observables | travel to K8; static after release | **Untested**: the pitched family needs an M7 smoke |
| 1 | put the bowl on the stove (On cook region) | `place_on(bowl via rim → burner)` | rim `touch_stop`; place on the burner height | SupportClassify on_surface | Covered ([F] LIBERO fires early: 84 mm hover allowance, PA[2]; the truth check judges) |
| 2 | put the wine bottle on top of the cabinet | `place_on(bottle via neck → cabinet top, keep upright)` | top_down neck pinch (`height_stop` or touch on the shoulder); transit above 1.128 + clearance | support at the measured top | Partial: a tall load near the top-down reach envelope (v1 reach_limit); the top must be surveyed by LookAt |
| 3 | open the top drawer and put the bowl inside | §7.2 | §7.2 | §7.2 | Covered |
| 4 | put the bowl on top of the cabinet | `place_on(bowl via rim → cabinet top)` | rim pinch; high carry | support at the measured top | Covered, with reach and visibility risk |
| 5 | push the plate to the front of the stove (On table zone, ±4 cm, no contact test) | `push_to(plate → spot "table in front of the stove")` | PushTo; parallax-free external error on one support | at the spot, at rest | Mechanically covered; success depends on the model's reading of "front" ([F] "front" = +x, toward the camera, PA[2]) |
| 6 | put the cream cheese in the bowl (graded On, 3 cm) | `place_in(cheese → bowl interior)` | top_down pinch across the ≈ 4.3 cm width, `touch_stop` on the table; cavity floor and rim measured; M18 per axis | on_floor of the bowl, bottom below the rim | Covered |
| 7 | turn on the stove | `turn(knob lever until effect(burner) or stop)` | as §7.1 steps 3–9 (the same knob) | effect + part rotation | Covered |
| 8 | put the bowl on the plate (On, 3 cm) | `place_on(bowl via rim → plate)` | as §7.3 | the aim tolerance decides | Covered |
| 9 | put the wine bottle on the rack (slot tilted 62.5°) | — | — | — | **Not covered**: needs orientation-constrained insertion; v2 lowers only along −z with the grasp orientation held (`agent_speedup_v1.py:65`) |

**Other libero_10 tasks** [I, from PA[0]]:
- Tasks 0, 1, 7 (baskets): `place_in` twice. The inside observer accepts support by contents, and the free-spot target M16 handles the second object. Task 7 is the backup: top-down only; living-room arena, where R-5 rule 3 leaves top-down grasps unchanged.
- Tasks 4, 6 (left/right plates, "right of the plate"): resolved by the model's pointing in the external frame; the frame-aligned wrist image keeps left and right consistent.
- Task 3 (bowl into the bottom drawer, then close it): `place_in` + `slide close`. The KS4 forearm block needs the pitched posture variant (untested).
- Task 8 (two pots): two `place_on` with M16 and `clear_of(first pot)`, plus `hold_true` for the stove-on start (a model claim).
- Tasks 5 (book into the caddy) and 9 (mug into the microwave, close the door): not covered (§9.3).

---

## 8. Migration path from v1

### 8.1 Module map

| v1 | v2 | Relation |
|---|---|---|
| `agent_speedup_v1.py` `AgentSpeedUpV1Policy`, macros | `agent_speedup_v2.py` `AgentSpeedUpV2Policy`, StepRunner; `agent_speedup_v2_relations.py` RelationTypes and controllers | replaced; the contract and macro patterns are re-written (§4.12) |
| `agent_speedup_v1_client.py` | `agent_speedup_v2_client.py` (subclass of `ChatGPTPlanHybridClient`, required `task_prompt`); `agent_speedup_v2_plan.py` (schema builder, dataclasses, `PlanValidator`, BDDL-like renderer) | replaced; the transport and pointing schema construct are reused |
| `contact_skills.py` | imported unchanged; `agent_speedup_v2_primitives.py` adds the new primitives and `CalibratePlus` (wraps `Calibrate`) | reused + wrapped |
| `wrist_servo.py` | imported unchanged; `agent_speedup_v2_geometry.py` adds `FeatureTracker` (cv2), `structure_axis()`, multi-feature κ, looming | reused |
| `mark_prompting.py` | imported unchanged; `fine_overlay_at(uv)` and the ray-hit marker in `agent_speedup_v2_client.py` | reused |
| `external_axes.py` | imported unchanged; 2×3 J_ext, n̂/v̂, `HandLocator`, the ray in `agent_speedup_v2_geometry.py` | reused + extended |
| `robot_tool_geometry.py` | imported unchanged; `agent_speedup_v2_robot.py` holds `robot_facts_v2.json` and `RobotModel` | reused |
| (none) | `agent_speedup_v2_planner.py` (GraspPlanner, finger columns, ContactClassifier, TransitPlanner, AimSolver, fit rule, keep-outs); `agent_speedup_v2_observers.py` (observers, `WedgeObserver`, `PartRotationObserver`, InvariantMonitor, EffectWatch); `agent_speedup_v2_thresholds.py` (`Threshold`, `CONSTANT_LABELS`, the lint vocabulary) | new |
| `agent_speedup_v1_video.py` | `agent_speedup_v2_video.py` | new (may import v1 drawing helpers) |
| `scripts/evaluate_agent_speedup_v1.py` | `scripts/evaluate_agent_speedup_v2.py` | new; the v1 runner is loaded read-only for helpers |
| `distance_tasks.py` registry (frozen) | `agent_speedup_v2_tasks.py` | new |
| `libero.py` `_LiberoWorker`, collision monitors (user-modified) | `agent_speedup_v2_sim.py` (`V2LiberoWorker` subclass), `agent_speedup_v2_contacts.py` | new, from the PA[1] prototypes |
| (none) | `agent_speedup_v2_truth.py`, `scripts/agent_speedup_v2_truth_check.py`, `scripts/agent_speedup_v2_replay.py` | new, from the PA[2] / PA[1] prototypes |
| `tests/agent_speedup_fakes.py` | `tests/agent_speedup_v2_fakes.py` (`FakeTwoViewSim`) | extended by import |

No user-modified tracked file and no v1 file is edited. The PA[1] list covers `libero.py`, `observation.py`, `policy.py`, `hybrid_*`, `chatgpt_plan.py`, `distance_*.py`, `scripts/evaluate_bowl_hybrid_subscription.py`, and v1's own files.

### 8.2 Milestones

All tests are ours and make zero model calls; live runs are the user's. Effort per milestone is [I] (§8.4).

| M | Deliverables | Tests | Gate |
|---|---|---|---|
| **M0** Protocol, infra, robot facts, rulings (≈ 5 d) | tasks registry (+ libero_goal hashes); V2 worker and process; generic monitor; truth module; runner skeleton (`--preflight`, `--sim-smoke plumbing`, finalization truth); replay script; `robot_facts_v2.json`; scripted stove-twist probe recording the burner red fraction against q; **robot-only gap traces** (closed-empty and open hands through free carries, lowerings and yaws) for G7, G8 and σ_g; moka hover allowance re-measured; manifest with helper-chain sha256 and lint vocabulary; ruling requests sent (§5.3) | runner tests with a fake SimProcess (pattern `tests/test_agent_speedup_v1_runner.py`); `_collision_record_v2` invariants; monitor regression (first events 340 / 129; 41/41, 99/101); truth verdicts (bowl 002 held → resting; drawer 003 perched); registry hash and prompt tables; **client `task_prompt` tests (§4.2)**; T0 lint scaffold; robot-facts pin (R5, R6 within 0.5 mm) | **G0**: all pass; replays bitwise; robot facts pinned; effect probe recorded; robot-only floors derived; **hard stop unless R-1v or R-2f is granted** |
| **M1** Depth and LookAt (≈ 10 d) | RobotModel, CameraModel (f, f̂), FeatureTracker, FeatureDepth, LoomingProbe, ParallaxProbe, TouchAnchor, HeightField (plane fill D7a; optional dense layer), PartRecord registration, ExternalModel (J_ext, n̂/v̂, ray), HandLocator, RayServo, RayDescent, frame alignment | fake: textured planes and boxes at known depths, dark low-texture parts, thin bars, plane-fill negatives (a black box on wood is not filled); FeatureTracker against `NCCTrack` (V7); real-sim survey smokes (zero calls, ORACLE seeds) on task 2, bowl_plate, drawer_bowl and task 7, plus held-out libero_goal 1, 6 parts, against evaluator truth (diagnosis only) | **G1 (go/no-go)**, per part class (knob lever top, handle loop top, bowl rim, drawer bar, burner plate, empty-hand drawer floor and rim), with **≥ 20 hover-probe measurements per class** (episodes 1–5 × 4 hover offsets of ±1 cm): median \|Δh\| ≤ 3 mm and 90th percentile ≤ 5 mm; widths within max(3 mm, 15%); f̂ within 5% of 333.6 (diagnosis). **G1b (LookAt)**: on the drawer_bowl handle and two other parts outside the start wrist view, with the ORACLE external pixel ± 8 px, the part lies in the inner wrist FOV and passes L4 in ≥ 9 of 10 trials per part; HandLocator within ±6 px of the evaluator projection. A failing part class is listed with its fallback (lower hover, touch-first plans, or `geometry_unmeasured` → repair). A failure on the loop or lever blocks task 2 and stops the build for a design review |
| **M2** Grasp (≈ 9 d) | GraspPlanner (finger columns P16, P7 exclusions, P17), ContactClassifier, Grasp controller, DOF probe with part observables (K10), HeldFrame | fake planner truth tables (wall, bar on a face, lever on a disk with a straddling column, loop, box, 63 mm can marginal, overhang → blocked with no close, a column on a spout → `column_blocked`); real-sim smokes with ORACLE pointing: task-2 lever (`touch_stop` and `height_stop`), task-2 loop, bowl rim (bowl_plate), drawer bar (frontal), cream cheese (goal 6), soup can (task 7); forced pot-body attempts with the jaw along y and along x; DOF probe on the drawer bar with a lateral probe | **G2**: hold_pass on all six; the pot body along y is refused before any descent, and along x it ends classified (hold or `blocked_above_part`, with no close above the band); the drawer bar plateau ≈ 17.5 mm is reproduced; the DOF probe reports prismatic along the pull and `hand_slides_on_part` laterally |
| **M3** Place (≈ 8 d) | Carry/TransitPlanner, AimCheck, GuardedLower, SupportClassify, Touchdown, Release/rest, PlaceOn/PlaceIn, fit rule M18 | fake: perch ring + floor (no release), drift (abort), miss, ambiguous cavity (touch probe inserted), tight-axis and blocked-axis cavities; real-sim: task-2 pot via the handle onto the burner (CP numbers reproduced), bowl_plate place; **drawer 003 live-state replays** from steps 818 (bowl held) and 925 (place hover), including the D003 counterfactual aims (centred; ±1.1 cm y; ±3 cm x; the run's x 0.118); drawer 002 from steps 799 and 1316 (overhang refused before descending) | **G3**: all pass under AC-2 where a full relation completes; M18 never refuses the run-003 geometry; release only on `on_floor`; no release on any perch |
| **M4** Mechanisms (≈ 8 d) | Turn (TwistSegment, WedgeObserver, PartRotationObserver, HingeTwist with orbit-follow, EffectWatch with O7/O8, budget), Slide (FollowAxis, fits, K8, posture variant), PushTo | fake hinge with damping, limits, lever backlash and a colour-change region; **fake round knob without a lever** (slip without wedging); fake slide with stops; real-sim: task-2 knob and goal 7, each over episodes 0–4 **in both first-sense orders** (scripted first sense); the deliberately wrong first sense; an end stop after a reversal; goal 3 open to `fits`; goal 0 attempt | **G4**: the sense decided correctly in 20/20 real-sim runs (2 tasks × 5 episodes × 2 orders) with the lever kept (G6 held, no `turn_slip`); the end stop after a reversal detected ≤ 5 steps after the joint reaches its limit (diagnosis); knob q ≥ 0.5 with the effect onset within ±0.05 rad of the q = 0.5 crossing, no spring-back; the round-knob fake's blocked sense detected by the part-rotation observer; the drawer opened to `fits` with DOF from part observables |
| **M5** Client (≈ 5 d) | plan / point / repair / replan schemas, `PlanValidator`, one instruction text, `ScriptedV2Transport`, preflight payloads | schema strictness and insertion order; reference validation and one re-ask; example plans within the §3.8 budgets; no numeric or direction fields (AC-5); identical instruction sha for all keys; the prompt example free of lint-vocabulary tokens; `task_prompt` required and equal to the official sentence for every key | **G5** |
| **M6** Integration (≈ 7 d) | `AgentSpeedUpV2Policy`, StepRunner (expansion, stays, escalation, reviews), ObserverBus, video overlays, stuck alert, timing | fake end-to-end with scripted plans; "no model call inside a step" (pattern `tests/test_agent_speedup_v1_policy.py:134-150`), including `Align` with `allow_pointing=False`; ≤ 2 calls per act; no task key reaches the policy or client; `hold_still` logged as a claim and never ends an episode; T0 lint complete | **G6** |
| **M7** Scripted chains and robustness (≈ 10 d) | `--sim-smoke relation` full runs; perturbed-pointing runs; V5 collection (robot-only); live-state replays with recorded model pointing | zero-call full chains with ORACLE pointing on task 2, bowl_plate and drawer_bowl (AC-2); perturbed pointing (ORACLE pixel ±6 px, one item wrong) per task; V5 robot-only distributions on initial states 1–4 of the gate tasks and on the held-out tasks (§5.1 item 3); drawer 002/003 and bowl 001/002 replays with recorded pointing; constants hash identical across all keys | **G7 (live readiness)**: 3/3 tasks raw ∧ physical on episode 0; ≥ 3 of 4 perturbed runs per task; every DD floor ≥ 1.5 × the 99th percentile of its robot-only V5 distribution (a violation is re-derived from robot-only data as a version bump and validated on the held-out tasks, §5.1 item 6); scripted step counts within the AC-4 targets |
| **M8** Live (user) | optional one-call schema check on the live endpoint (§9.1 risk 6); task 2 episode 0 live; lens-by-lens analysis as in spec v1 §17; then bowl_plate and drawer_bowl live regressions; then task 7 | — | AC-1…AC-6 |

[I] A vertical slice (M0; M1 on the task-2 parts; M2 lever and loop; M3 place_on; M4 turn; M5; a minimal M6) reaches a zero-call task-2 chain at about day 40. The first live run waits for G7, which includes the v1 regressions as the brief requires.

### 8.3 Test inventory

- **T0** constants and vocabulary lint (§5.1).
- **T1** fake-sim (`FakeTwoViewSim`):
  - per relation: the happy path, every failure reason, and the negative gates (no close on `blocked_above_part` or `column_blocked`; no release on `perched`);
  - the perch geometry (rim 5.6 cm above the floor) and a tight-axis cavity (M18);
  - a parallax case (destination 16 cm above the pick support);
  - an in-hand offset of 3.7 cm with 15° tilt, and a `wrist_visible` false case;
  - lever backlash with a wrong first sense, a round knob that slips, an off-axis twist with orbit-follow;
  - LookAt with an external camera ray and an overhang;
  - no model call inside a step.
- **T2** scripted real-sim smokes with zero model calls (ORACLE stub test-only; truth check after every smoke).
- **T3** live-state replays: drawer 003 steps 818, 899 and 925, with the counterfactual aims; drawer 002 steps 799 and 1316; bowl 001 and 002 states.
- **T4** runner protocol tests (PA[1] list), plus the client `task_prompt` tests.
- **V5** robot-only distributions (empty closes, open holds, closed-empty-hand gap under yaw, carry and lowering motions, static frames, probe flow) on initial states 1–4 and on held-out tasks.

### 8.4 Effort

[I] About 60–80 developer-days to G7: M0 5, M1 10, M2 9, M3 8, M4 8, M5 5, M6 7, M7 10 = 62 days, plus a ≈ 25% contingency. That is about 10–13k lines including tests.
- The first version's 41 days underestimated M1 and M2 by about 2× (critic).
- It also omitted LookAt, the wedge observer and the robot-only trace collection.
- Expect at least one live iteration cycle per new relation kind after M8.

---

## 9. Risks, open questions, out of scope

### 9.1 Risks

1. **Depth on dark, thin and low-texture parts.** The black knob and lever, the 13 mm black loop, the dark stove and drawer interiors are all affected. [F] Only about 38–44% of the start-view grid patches pass the texture test. This is the largest risk. Mitigations: sparse features plus touch as the primary path; hover probes at short range; plane fill only where the appearance matches (D7a); unknown-as-obstacle elsewhere. G1 decides go/no-go per part class.
2. **Height stop on the free pot handle.** [I] With the pinned 31.0 mm palm and the loop's 0.967–1.0345 extent, the feasible window is ≈ 22 mm wide (g ∈ [1.0085, 1.031]), so it tolerates σ_h up to ≈ 7 mm. [F, CP] g* ≈ 1.020 closed at 14.11 mm, holding the pot rigidly. The risk is lower than the first version stated.
3. **Wedge observer generality.** It is validated on one lever knob (CP, SP). Round knobs without a lever slip instead of wedging and rely on the part-rotation observer, which may be unmeasurable on a black part. Backlash transients (≈ 10–14° of hand yaw) lengthen every sense decision [Q, M4].
4. **Pointing on small or adjacent parts** (the moka handle next to the frypan handle; drawer interiors): untested live. Partly caught by the answer checks and the co-motion check; costs re-points.
5. **Plan-level errors** (wrong grip part, missing enabling step, wrong order): measured rejections (`part_too_wide`, `column_blocked`, `opening_too_small`, `grip_not_on_obj`) reach the model as repairs, each costing 5–32 s.
6. **Strict schema with arrays of closed objects** is new on the subscription endpoint; only a live call can reveal a rejection. Mitigation: M5 offline checks plus an optional one-call schema check the user can run before the first full live run. Fallback: flatten `keep` and `until` into fixed nullable fields (≈ +20 tokens per step).
7. **In-hand estimation:** the zero-flow silhouette near the principal point; Touchdown may pivot a handle-held object; tilt is captured mainly through b_low; an off-centre held object leaves the wrist view under the wrong yaw (P17 chooses; `wrist_visible` false falls back to the external view).
8. **Effect observer:** arm occlusion and shadows; the visibility conflict is resolved in M0; the effect is valid only under O7 and O8; part rotation remains the fallback.
9. **Twist mechanics:** fast twists lose the lever ([F, CP] cap 0.5), so K2 ≤ 0.1. The visual disk centre is 4.6 mm off the hinge; compliance absorbs this in the probe, and `axis_tug` uses σ_c. A joint-7 limit stall (`hand_yaw_stall`) cannot be seen in state8. The yaw budget R12 limits single-grasp turns to ≈ 1.96 rad.
10. **LookAt:** affine J_ext error far from the calibration pose (closed-loop servo); the HandLocator's extreme-point assumption (G1b); overhangs above the line of sight (guarded descent, two-pose fallback); it depends on R-1v.
11. **LIBERO leniency:** raw success can fire on contact or mid-recovery (84 mm hover allowance for the bowl on the stove; a held bowl pressed on a rim scores, D003 skeptic; [F, CP] the moka On() fires at the support contact). The protocol accepts it; AC-2 judges physically.
12. **Generic-monitor counts** are not numerically comparable with v1's (bowl 002 +7 steps, drawer 003 −2 steps, PA[1]); v1 keys report both.
13. **Speed:** Calibrate+ ≈ 190 steps and the probes add ≈ 250–350 steps per task; the plan call at medium effort is 15–32 s [P]; on bowl_plate v2 may not beat v1's `control_without_video_s`.
14. **Complexity:** ≈ 60–80 developer-days; M1, M2 and M4 validation is the critical path.
15. **R-5 gate:** 4.5 cm against a ≤ 2.8 cm start-pose spread and a mount 9 cm away. A future arena with a mount within 9 cm of the kitchen mount would need the gate re-derived (version bump).

### 9.2 Open questions for the user

1. Rulings R-1v, R-2f, R-5b (§5.3). Without R-1v or R-2f the build stops at G0.
2. Whether video off the control loop, PNG level 1 (with re-frozen hashes) and an asynchronous plan call overlapping Calibrate+ are acceptable protocol changes (§6.8).
3. Whether reviews (at most 2 per episode, only when every observer holds) are wanted, or whether the policy should hold still as soon as every step is done.
4. Whether task 7 (living room, no articulation) is the right backup, or whether task 3 (kitchen, bowl into the bottom drawer and close it) should be, keeping R-5 exact.
5. Whether an unseen effect should continue the turn to the yaw budget (the default here) or stop at a quarter turn.

### 9.3 Out of scope for v2.0

- Orientation-constrained insertion: the wine rack slot (libero_goal 9, 62.5° tilt) and the book into the caddy (libero_10 task 5; its In predicate also fails an upright book, PA[0]).
- Horizontal insertion into an occluded cavity, and hinged-door arcs: the microwave (libero_10 task 9). FollowAxis can in principle follow an arc but is untested on doors.
- Ratchet twists beyond one grasp's yaw budget; spring-loaded or push-and-turn knobs.
- Stacking and spacing constraints beyond the free-spot rule (libero_10 task 8 is partially covered).
- Push-to precision against BDDL zones beyond what the model's pointing provides.
- Dense parallax depth as the primary depth source (optional layer only).
- **Pre-shaped apertures** (`ApertureSet`, a `ZeroGripAdapter` emitting gripper action 0; R11 says action 0 holds). No target task needs them. They return only with a part whose finger columns fit no full-opening landing.
- High-effort model calls, model-judged completion (`verify`), asynchronous calls, PNG re-encoding.

---

## Appendix A. Critic findings: applied / rejected

Verdicts: **Applied**; **Applied, corrected** (the change is made, but part of the critic's statement was wrong, with the correction given); **Rejected** (with the reason). Every finding was checked against the code, the dumps, CP, and SP where noted.

### A.1 Refuted or overstated claims (critic §2)

| ID | Finding | Verdict | Check and action |
|---|---|---|---|
| R1 | The knob disk radius is ≈ 47.6 mm, not 34 mm; the "6 mm overlap, rejected" analysis is wrong; the disk centre is 4.6 mm from the hinge; ApertureSet/ZeroGripAdapter/G11 are unnecessary | **Applied** | [F] `flat_stove.xml:41` (central box half-size 0.03396), `:42-54` (ring slabs to ≈ 0.043–0.052 m), `:33-36` (hinge at the body origin); `t2_dump.json` button AABB x [−0.2484, −0.1531], y [0.1598, 0.2549]; wrist image ≈ 87 px across at Z ≈ 0.365 m; CP stall 0.9272. Action: §2.2, §2.4, §2.5, §4.4 R5, §4.6 P4/P16, §7.1 steps 4–8, §9.3. Minor correction: the lever is 24.3 mm wide (2 × 0.01214, `flat_stove.xml:55`), not 24.7 mm, so 25.39 mm = 24.3 + 1.0 (R9) fits better. |
| R2 | The hand yaw-ratio sense probe cannot find the free sense; K4 cannot see an end stop; "TurnOn still established" is false in the wrong-sense branch | **Applied, corrected** | Confirmed and strengthened by SP. The critic compared cap-0.1 knob rates against a cap-0.15 free-air rate (1.87°/step). At the same cap the closed hand yaws 1.31°/step in free air, and with the knob gripped −1.08…−1.17 (blocked) against +1.16…+1.25°/step (free): a ratio of ≈ 0.9, so even less separable. Correction to "\|Δgap\| jump tests miss this": v1's `gap_jump` is \|Δgap\| ≥ 0.2 mm (`contact_skills.py:698`). It would fire in the blocked sense (0.24–0.51 mm/step), but also in the free sense from a fresh close (0.29–0.30 mm/step) and after every reversal. It is non-discriminating rather than blind. Action: K1w/K4w/K7 from the signed gap trend (rigid-wedge rate ℓω), the part-rotation observer, the effect under O7/O8; G10 not used in twists; G4 with both orders and a forced wrong first sense; §7.1 failure branch restated. |
| R3 | The `inside` precondition blocks drawer_bowl (1.6 mm y clearance); ±1.1 cm off-centre drops still land on the floor | **Applied** | [F] D003 skeptic (0-based line 6): "the 1.6 mm total clearance in y for a flat drop is correct geometry … drops at y −0.095 and −0.117 … ended flat on the floor"; x was decisive. Action: M18 axis-specific fit with compliant entry (§4.8), G3 replays with those counterfactual aims. |
| R4 | LookAt is not implementable as written: `hand_image_motion` gives no absolute pixel; the part lies Δh·cot 39° away | **Applied, corrected** | [F] `external_axes.py:42-62` returns a median flow vector; the agentview pitch is 38.9° (`t2_dump.json` `xmat`). Correction to the critic's remedy: a "motion-mask centroid with a grip offset calibrated in Calibrate+" needs an absolute reference that Calibrate+ lacks. v2 uses the mask's extreme point along the measured image-down direction (the fingertips), validated against J_ext over the probe moves (L1). It also adds an external `point_fine`, because an 8×8 external cell is ≈ 11 cm wide at the scene. Action: §3.6 item 5, §4.5 HandLocator, §4.9 LookAt (RayServo, RayDescent, two-pose fallback, R-2f-only scan), L1–L5, G1b. |
| R5 | The base client rejects non-bowl prompts unless the prompt is passed at construction | **Applied** | [F] `direct_history.py:37-42`, `:68`, `:114`; v1 passes it at `agent_speedup_v1_client.py:411`. Action: §4.2 (required `task_prompt`), AC-1(b), M0/M5 tests. |
| R6 | The pot-body pinch reasoning is wrong: the oracle fingers landed on the handle root and spout; P2 rates 66 mm feasible; a finger-descent-column check is missing | **Applied, corrected** | Consistent with `oracle_t2_a.json`: the descent ended at grip 1.0401 (`at_pot`), which puts the fingertips at ≈ 1.031, on the 1.032–1.033 tops; the grip then rose to 1.0488 during the empty close. Correction: the e_lat bound is 5.75 mm (sep = 78.5 − 1.0 mm, R9), not 6.25 mm. Added consequence: across the spout–handle axis (jaw along x) the body pinch is a *valid* candidate whose columns reach the table, so the spec no longer claims the body is refused in every yaw. Action: P16, §7.1 step 12, M2 forced attempts along both yaws. |
| R7 | Slips: 834 not 856 characters; hand footprint ≈ 15 mm from the body and palm ≈ 7.6 mm above it; moka hover allowance 10.9 vs 16.4 mm; t_f 12.5 mm biases touches by ≈ 3 mm | **Applied, corrected** | 834 confirmed by computation. The 15 mm footprint clearance is confirmed. The 7.6 mm palm figure uses the oracle grip 1.0146 with the 26.1 mm AABB palm; at g* ≈ 1.020 with the pinned 31.0 mm hull palm, the margin is ≈ 18 mm above the body top (≈ 16.5 mm above the loop top); both are stated in §7.1. The hover allowance sources disagree; both are reported and M0 re-measures. t_f pinned at 9.3 mm. |

### A.2 Missing pieces and interpretation risks (critic §3)

| Finding | Verdict | Check and action |
|---|---|---|
| Texture: only 64 of 156 patches pass `min_std` 4; D7 inflates unknown cells beside the handle; P7 counts the grasped part; P7 rejects the handle once σ_h ≳ 2.2 mm | **Applied** | [F] Re-measured: 54/143 patches pass; the table patch std ≈ 2.95 grey. Action: D7a plane fill with an appearance test, P7 exclusions, p_min 31.0 mm (§4.5, §4.6, §5.2). |
| Compute: NCC 1.74 ms per track per frame; 4 cm moves shift features ≈ 35 px, beyond the 32 px search | **Applied** | [F] Re-measured 1.4–1.7 ms (numpy) against 0.16 ms (cv2); search ±32 px (`wrist_servo.py:134`). Action: FeatureTracker on cv2, predicted-shift search, per-step tracking only during probes (§4.5, §3.8). |
| Calibrate+ ≈ 250 steps, not 150 | **Applied, corrected** | Accepted for the first version's content. This revision trims Calibrate+ to one extra close (a third only on disagreement) and puts the yaw probe inside v1's close: ≈ 180–200 steps [P] (§4.5). |
| The plan token estimate is low (2.5–3 chars per token; reasoning; 15–30 s) | **Applied** | §3.8 re-derived (15–32 s); AC-4 re-budgeted; `scene` capped at 80 characters. |
| Held-object visibility depends on the yaw sense | **Applied** | [F, CP] pot − grip = (−0.0001, +0.0795) with the 0.05·R0 camera offset (R2). Action: P17, `wrist_visible` fallback, M14/M17 (§4.6, §4.7). |
| The G7/G8 floors are too tight and come from held-object traces | **Applied** | [F] `contact_skills.py:886-887` comments (live 001 clean carries ≤ 0.55 mm; noise ≤ 0.149 mm/step) and `agent_speedup_v1.py:62` (lowering drift ≤ 0.09 mm) are held-object recordings. Action: provisional floors 1.0 mm / 0.4 mm per step / 0.3 mm per step (1.8×, 2.7×, 3.3× those values), re-derived in M0 from robot-only traces. |
| The R-5 gate is near the edge; "the nearest other mount is more than 15 cm away" | **Applied, corrected** | Distances 2.25 / 0.98 / 1.27 cm confirmed. The nearest-mount claim is **wrong**: the study MountedPanda (−0.75, 0, 0.912) is 9.0 cm from the kitchen mount (`SURVEY/tasks/t5_dump.json`); the living room is ≈ 51 cm away. The gate is set to half of 9.0 cm, i.e. 4.5 cm (§5.3); the study start-pose estimate (9.2 cm) is still rejected. |
| A DOF probe along a pinched bar can read "lateral blocked" as free | **Applied** | [I] Accepted: a uniform bar is not its own observable. Action: K10 with the part's `rigid_patch` (the drawer face beside the bar) and external co-motion; `hand_slides_on_part` (§4.9, M2). |
| `Align` can call the model | **Applied** | [F] `wrist_servo.py:531` (`allow_pointing=True` default), `:621`, `:796`. Action: always `allow_pointing=False`; M6 test (§3.2, §4.9). |
| The answer check on the handle ("≥ 3σ above surroundings") can reject a correct point | **Applied** | Surroundings are defined as the finger-landing columns (§3.6 item 4). |
| Effect region validation is missing | **Applied** | O7 (region disjoint from the turned part and the hand's projected volume, low false-change rate) and O8 (≥ 5° of part rotation before an onset counts). |
| The V5 tuning loop tunes on the evaluation tasks | **Applied** | §5.1 items 3 and 6: robot-only inputs only, held-out task validation; the G7 wording changed. |
| `hold_true` effects cannot be verified by the change-based O1; `hold_still` is a step-level completion claim | **Applied** | Both are logged as unverified model claims. `hold_true` is monitored as "unchanged since the start baseline". `hold_still` never ends an episode, and reviews are sent only when every observer holds (§3.3, §3.7, AC-5). |

### A.3 Required changes (critic §4)

| Item | Verdict | Where |
|---|---|---|
| MUST 1: correct the knob facts; full-open touch landing or height_stop; remove ApertureSet/ZeroGripAdapter/G11/Q-3 from the critical path; the P4 straddle rule; axis_tug relative to the disk-centre prior error | **Applied** | §4.4, §4.6 (P4, P16), §4.9 Turn step 4 (σ_c, K11), §5.2, §7.1, §9.3; Q-3 removed |
| MUST 2: turn sense, end stop and DOF from part observables; G4 with blocked-sense detection in both senses and a wrong first sense | **Applied** (evidence extended by SP) | §4.9 Turn and DOF, K1w/K4w/K7/K10, M4/G4 |
| MUST 3: replace the `inside` precondition; validate on the D003 counterfactual drops | **Applied** | M18, §4.8, §7.2 step 11, M3/G3 |
| MUST 4: specify LookAt properly, or give drawer_bowl another way | **Applied, corrected** (extreme-point locator instead of centroid + offset; external fine pointing added) | §3.6 item 5, §4.5, §4.9, L1–L5, G1b |
| MUST 5: construct the client with the official prompt; test that the bowl default never reaches a non-bowl task | **Applied** | §4.2, AC-1(b), M0/M5 |
| SHOULD 1: D7 × P7 handle interaction; exclude the grasped part and finger slot; plane fill; hull palm 31.0 and t_f 9.3 pinned; finger-descent column | **Applied** (the plane fill carries an appearance test so textureless objects are not filled) | D7, P7, P16, R5, R6 |
| SHOULD 2: re-derive the G7/G8 floors from robot-only traces with margin | **Applied** | G7, G8, M0 |
| SHOULD 3: widen the R-5 gate to about half the distance to the nearest other mount | **Applied, corrected** (that distance is 9 cm, not > 15 cm) | §5.3 |
| SHOULD 4: re-budget speed (NCC compute, Calibrate+ steps, plan reasoning tokens) | **Applied** | AC-4, §3.8, §4.5 |
| SHOULD 5: choose the yaw that keeps an off-centre held object in view; specify the fallback | **Applied** | P17, §4.7 |
| SHOULD 6: define "surroundings"; validate effect parts; minimum rotation before an onset | **Applied** | §3.6, O7, O8 |
| SHOULD 7: V5-driven changes from robot-only inputs only, validated on held-out tasks | **Applied** | §5.1 |
| SHOULD 8: G0 a hard stop unless R-2f or R-1v is granted | **Applied** | §5.3, M0 |
| COULD 1: neutralize the prompt's part list; define the T0 lint vocabulary | **Applied** | §3.5, §5.1 item 4 |
| COULD 2: freeze the sha256 of the user-modified helper chain in the manifest | **Applied** (v1 already copies `HELPER_SCRIPTS` into `frozen_source`, `scripts/evaluate_agent_speedup_v1.py:52-53`, `:756-761`; v2 also lists their hashes in the identity) | §6.1 |
| COULD 3: set G1 sample sizes | **Applied** | G1: ≥ 20 per part class |
| COULD 4: define the vertical extent of keep-outs | **Applied** | O5 |
| COULD 5: treat `hold_true` and `hold_still` as unverified claims | **Applied** | §3.3, §3.7 |
| COULD 6: re-estimate effort | **Applied** | §8.4: 60–80 developer-days, 10–13k lines |

### A.4 Confirmed strengths (critic §1), noted

The citation spot-checks, the palm and finger figures, k = 2 for task 2, the moka handle height_stop, place_on with SupportClassify, the twist at K2 and the knob height_stop were all accepted as stated. CP numbers are cited in §7.1. One correction: the cap-0.5 twist dragged the grip ≈ 10 cm ((−0.2068, 0.1961) → (−0.2619, 0.2853)), not ≈ 8 cm. The interface and call-count assessments needed no change.
