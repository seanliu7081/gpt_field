# Agent-control harness analysis: problems and improvements (2026-11-08)

Scope: the three harness versions in `/workspace/gpt_field`, judged against the four goals: zero-shot, fast, collision-avoiding, few attempts.

| Short name | Location | What it is |
|---|---|---|
| **H1** | `agent_speedup_v2_obstacle_harness_100mm/` (git `3cc84ac`) | The frozen uc7.1 agent-first loop: model-written obstacle boxes, a local box router and a 0.10 m transit cap. |
| **H2** | `src/gpt_field/foundation_harness/` | SAM 3.1 + DA3 perception → Astra operation contracts → local IK/collision planning. **The current revision is `foundation-harness-keypose-2` (K2).** It finished at 2026-11-07 23:59:57 UTC, has never run a task, and its T04 launch is awaiting approval. The only live H2 run (`runs/foundation_harness_t04_can_obstacle_001`) used the older revision **v1**. |
| **H3** | `src/gpt_field/foundation_vision/` | SAM 3.1 + DA3 perception whose summary is injected into uc7.1's executive prompt. Control is H1's code; there is no automatic obstacle use. |

**Method.** The analysis was read-only and in three rounds:
- 8 first-round analysts.
- 5 adversarial verifiers (H1 speed, H1 collision, H1 zero-shot, H3, cross-harness numbers). They re-derived the cited numbers from raw run artifacts and the cited code lines.
- 2 independent analysts of the hash-pinned K2 snapshot, a cross-report critic, and a final fact-check plus completeness check of this report.

No simulator, model call or GPU job was run. All reports and scripts are in `work/`.

**What was checked.** Every H1 and H3 finding below survived adversarial verification, with corrections applied. The H2-v1 findings (§4.1) come from the first-round reports and were re-derived only in part by the two K2 analysts. The K2 findings (§4.2–4.3) come from two independent analyses; where they disagree, both are given.

---

## 0. Bottom line

| Goal | H1 obstacle harness (uc7.1 transit100) | H2 foundation_harness (K2 current; v1 ran) | H3 foundation_vision |
|---|---|---|---|
| **Zero-shot** | One frozen build, but never run on an official task. Its 5 evidence runs are custom obstacle or composite scenes. Success is scored by an evaluator oracle. | Not demonstrated: v1 executed 0 operations, and K2 has not run. | Breaks the rule: the obstacle is perceived only because the operator passes `--vision-query can/bottle`. |
| **Fast** | T04: 402 s wall, of which 304.8 s is control without video + plan. Model time is 71–73 % of control + plan (64–70 % in the stove-off and drawer runs). Harness overhead is 0.28–0.40 s per 0.05 s step. | v1: no progress. K2: both analyses predict about 3.5 s per control step. T04 then needs **about 1 h of control** (60–85 min, a step model anchored to uc7.1-T20's 1,276 actions at the same 20 mm cap), or 13–17 min under the other analysis's full-step-realisation model. Far above 3 min either way. | T04 is slower than H1 (451 vs 305 s); T08 is about equal (501 vs 490 s). Most of the T04 gap is a slower service window (≈ +59 s) and 2× reasoning tokens (≈ +42 s); vision itself cost ≈ 15 s. n = 1. |
| **Collision** | T04: 0 obstacle contact (the model carried high), but 3 palm-on-mug episodes. T08: a 47-step finger press on the bottle. The router checks one point. | K2: about 90 % of the scene is "uncertain" and traversable, so the table, a distractor mug and the plate can be entered. | Same controller and grip-site router as H1, and its vision geometry is never consumed (0 usable centres). Yet its first boxes were better (full coverage with clearance in 2/2 runs vs 0/4 for H1-family) and it had 0 obstacle contacts. The likely channels are the operator-named obstacle's 2D box or the extra instructions. Untested. |
| **Few attempts** | 14–43 calls per task. 54–58 % of calls follow a fragment that completed. Vetoes cost extra calls. No give-up. | v1: 12 calls, 0 motion. K2 is predicted to dead-end after the first grasp. | 12 calls, but nothing ties that to vision (n = 1). |

**Three things to act on first**

1. **The pending K2 T04 launch.**
   - **What K2 fixes:** it removes all five v1 blockers (the link0 start veto, the unknown-space veto, blocking arm-ghost points, non-exempt finger hulls and the placement gate) and forwards structured failure reasons. Both analyses expect its first approach keyposes to execute; the prep connects them in 1.6–2.1 s. It would be the first H2 run to move.
   - **Speed:** SAM, DA3 and the new self-mask re-run on every control step, at about 3.5 s per step.
   - **Lift dead end:** after the first close, the lift is blocked by the mug's own surface points (19 of 19 saved frame checks with a usable payload).
   - **Contact trap:** once in contact, the robot cannot leave, because the start configuration itself is vetoed.
   - **Forecast:** 8–20 calls in the first 15–30 min with 0–1 mugs moved (planning-loop analysis), or more than 10 calls and more than 10 min without completing T04 (perception analysis).
   - **Recommendation:** before launching, make the two small fixes the perception analysis names: payload by geometric membership with no skipped held-object check, and no empty surfaces from `surface.py`'s component veto. Ideally also take perception off the per-step path. Otherwise run it as a diagnostic with a manual stop; the runner has no call cap and no give-up.
2. **The largest cross-harness problem is that the LLM writes metric geometry.**
   - **Size of the error:** first obstacle boxes are 5–12 cm off. In T08 the error lies almost entirely along the external camera's viewing ray (a depth error). In T04 it is mostly perpendicular to the ray: the box was drawn at the can's base row.
   - **It repeats:** errors recur within 1–2 cm across runs, so resampling the model will not fix them.
   - **What it caused:** it set the target of the T08 bottle press. Astra aimed at a box 8.5 cm off in x and 6.6 cm low; the contact itself came from OSC rotation lag during a combined translate-and-rotate stage. It also contributes to over-deep grasp targets, together with the missing palm envelope, and to some stall-triggered calls.
   - **Not enough on its own:** better boxes are necessary but not sufficient. With the improved post-contact map, the point router still plans within 1 mm of the colliding descent.
   - **K2 makes it worse:** it asks the model for even more metric output, as absolute keyposes.
3. **Current success numbers are not zero-shot measurements.**
   - Every known uc7.1-module success (12/12) stopped at the first raw-LIBERO step with an object still held.
   - LIBERO's site regions accept an object up to 10 cm above the site (`site_object.py:63-80`). In 2 of those 12, raw success was not a valid completion: in the t6 relayout the pudding was still being carried about 5 cm up, and in t0 the sauce would perch on the basket rim even after release. The honest success rate of the 7 retained runs is anywhere from 0/7 to 6/7.
   - The agent is told it is "not done" while the episode continues.
   - Every configuration is n = 1, and model service speed varied 31.8–61.1 tok/s between runs.

**Recommended direction.** H1 is the base because it is the only harness that moves and completes tasks, not because a fair comparison favoured it: every result is n = 1 and oracle-scored. On top of H1:
- turn **H3's perception into an executor service**, not prompt text;
- reuse **H2's robot model** (FK/IK, collision hulls) for body-swept checks and reachability;
- move metric grounding out of the LLM.

Prerequisites:
- (a) H3's prompt, two-view and 0.9-gate fixes must ship together and pass a truth-scored offline benchmark. Fixing the prompts alone still gave 0/22 usable centres.
- (b) Everything must run in one interpreter. H1's `.venv` (numpy 2.5.3, no scipy or torch) cannot import H2's `robot_geometry` or H3's perception, so use `.venv-foundation-vision` or a service process.
- (c) H2's per-configuration collision check (0.11–0.24 s) needs a vectorised broad phase before it can run on every stage.
- (d) Fix measurement first (§6).

---

## 1. Evidence base (single episodes; init 0, seed 7; GPT-6 Astra high on the subscription)

| Run | Result | Calls (plan + exec) | Actions | Time | Forbidden contact |
|---|---|---|---|---|---|
| H1 T04 + can, transit100 | raw ✓, strict at terminal ✗ | 1 + 13 | 549 | 304.8 s (control w/o video + plan); **402.1 s wall** | 0 obstacle; 3 palm-on-mug episodes (3,372 samples) |
| H1 T08 + bottle | raw ✓, strict ✗ | 1 + 21 | 843 | 489.7 s; 634 s wall | 47-step finger-on-bottle press (3,504) + pot–bottle + pot–burner |
| H1-config stove-off T08 / drawer 001 / 003 | raw ✓ | 17 / 43 / 37 | 889 / 2,364 / 2,131 | 459 / 1,301 / 1,259 s | Stove-off: pot–burner rim (1,650). Drawers: mostly palm-on-moka (2,909 / 1,440) plus cabinet contacts, including arm links on the static cabinet in 001. |
| H1-config drawer 002 | plan call failed (HTTP 401 `token_expired`) | 1 | 0 | – | – |
| H3 T04 / T08 | raw ✓ | 1 + 11 / 1 + 11 | 604 / 590 | 451.2 / 500.5 s | 0 / 400 (pot–burner, at touchdown and release) |
| H2 v1 T04 | user stop | 0 + 12 | 19 (holds) | 202 s, 0 motion | – |
| Astranomy direct T04 (baseline) | raw ✓ (2nd attempt) | 63 | 306 | 740 s | hit the can in 3 of 3 attempts |

Notes:
- Two runs often labelled uc7.1 are older builds with a 20 mm cap, and one of them (`uc71_default_routing_t04`) is the **official layout without the can**.
- Contact counts are physics-substep contact points, not events.

---

## 2. Shared root causes (fix once, help every harness)

| # | Root cause | Where | Evidence | General fix |
|---|---|---|---|---|
| R1 | **The LLM writes metric geometry** (boxes, targets, keyposes) from images plus calibration. | H1, H3 (inherited), K2, astranomy | First boxes miss the added obstacles by 5–12 cm. The T08 bottle box had IoU 0 and its bottom 10 cm below the table; that error is ≥ 98 % along the external camera ray. The T04 can error is mostly perpendicular to the ray, and that box also sat below the table. The first plate estimate was 5.7 cm off. Errors repeat within 1–2 cm across runs. | The harness measures. Triangulate pixels the model points to in ≥ 2 views, or intersect one view's ray with a support plane, or use automatic perception boxes. The model labels and selects; it never writes unanchored metres. |
| R2 | **The robot is modelled partially, or wrongly, in collision reasoning.** | H1 router (one grip-site point); H2 (own depth treated as obstacle); astranomy | 76 % of H1 forbidden samples happen in stages the router never sees. Palm-on-target is 55 % of samples. OSC rotation lags translation about 3×, so stages run as "translate, then rotate in place". | One body + held-object swept model (FK hulls or spheres) over the *executed* orientation profile, used by every stage as a planner (escape, detour, contact handoff), not as a veto. Object IDs with per-part contact permissions. |
| R3 | **The shared perception backend.** | H2, H3 (`foundation_vision`) | DA3 two-view scale leaves a per-view bias: wrist table +18–20 % (6.5–7 cm), external +2.7–3.5 % and range-dependent. Robot-body depth is off by 8–30 cm. Two-view and 0.9-agreement gates discard good evidence. Plan part phrases are used as SAM prompts. | Category prompts plus instance binding. Uncertainty instead of vetoes. A rendered-robot mask with depth replacement. Per-view affine depth anchored to the external view, robot pixels and the support plane. A truth-scored offline benchmark. |
| R4 | **A serial, blocking model loop.** | H1, H3, H2, astranomy | Model time is 71–76 % of H1/H3 control + plan, and about 87 % of policy-intrinsic time. 58 % of calls follow a successful fragment. The robot is frozen during inference. | Commit whole sequences with measured continuation predicates; call the model only at semantic decisions; run perception asynchronously. |
| R5 | **No contact sensing.** | H1, H3, astranomy; K2 descents end only by stall | Contact shows up only through a late stall: median 13-step episodes, and a 47-step press. | Guarded motion (a translation-progress guard) plus "until contact" as a declared stage terminator. |
| R6 | **No honest termination; unbounded attempts.** | all | 12/12 uc7.1 successes stopped while holding the object. There is no `give_up` and no cap. H2 v1 spent 12 zero-progress calls; astranomy spent a 12-call hold loop. | Terminal `finish` plus `give_up`, an evaluation cap recorded as a failure, and failure-signature de-duplication. |
| R7 | **Tolerances smaller than the errors they guard.** | H1 (hidden 0–20 mm lattice inflation); H2 (3 mm vs 6 mm vs cm-level depth error); H3 (2 cm + 2 % band; 0.9 gate) | Spurious start-blocked refusals; the v1 deadlock; correct detections rejected. | Derive tolerances per frame from measured uncertainty. |
| R8 | **Tests cannot catch live blockers.** | H2, H3 (synthetic depth, full-image masks); H1 (deterministic replay only) | Every H2 v1 blocker reached a paid run. | One shared saved-observation replay suite: minutes on CPU, no model calls. A development aid before launches, not a hard gate (the user works live-first). |
| R9 | **Simulator-only privileges are load-bearing.** | all | The prompts say "Simulation pauses during inference", so 13–79 s calls cost no world change. The camera validator accepts only an ideal MuJoCo pinhole and the exact Panda gripper (H1, H3). Success is scored by evaluator oracles. SAM and DA3 ran only on rendered images. H2 spends 3.7 s of local compute per action, against 1.2–1.7 ms per step for H1/H3 policy compute. | Accept real calibrations. Re-localise in the executor at the moment of use and stamp capture time against action start (step- or event-indexed freshness is safe only in a paused simulator). Keep per-action local compute inside the control period. |

---

## 3. H1: `agent_speedup_v2_obstacle_harness_100mm` (uc7.1, transit100)

### 3.1 Problems

**FAST**
1. **Per-step harness work dominates (critical).**
   - Each 0.05 s step costs 0.28–0.40 s: inline planning video (0.12–0.20 s), camera render plus PNG encode and decode of both views (0.11–0.13 s), and `run_state.json` rewrites. The cost grows with episode length.
   - T04 wall was 402.1 s, of which motion was only 27.5 s.
   - Code: `scripts/evaluate_agent_speedup_v2.py:1036-1053, 1515-1555`; `observation.py:44, 114-115`; `agent_speedup_v2_user_check_sim.py:118-125`.
2. **A checkpoint after every phase (critical).**
   - Every fragment end returns control to the model (`_fragments.py:422-434`).
   - The prompt mandates a blocking visual check after each grasp and each placement (`_agent_client.py:161-164`).
   - Hard caps of 6 segments, 120 steps per segment and 240 steps per fragment (`_fragments.py:24-26`) force at least 2 calls per pick-and-place.
   - The result is 6.5–10.5 calls per object, and about 2–3 s of motion per 15 s call.
3. **The 0.10 m cap overshoots and creates false stalls (medium).**
   - The carrot lookahead overshoots by 10–28 mm. The stall test (`_fragments.py:404-419`) then fires about 11 steps after the error minimum, often at the overshoot apex or early in the return. That accounts for 15–17 of 41 stalls; the 20 mm runs had none.
   - The cap also raised p99 transit speed from 0.12 to 0.53 m/s. Grip-site overshoot never exceeded the clearance in use (0/104 stages). But hand, finger and held-object contacts, which the router does not model, now begin faster: the T08 bottle contact started at 0.17–0.19 m/s, above the 20 mm ceiling.
   - `PathTracker` commands zero translation inside the 4 mm arrival tolerance (`tracking.py:90-95`). An arm that is still rotating therefore hovers at the boundary and drifts into a stall.
4. **Contact counts as failure (medium).** An approach that stops on its intended contact ends as a failed `segment_stall`, and the pending close and lift are dropped. The verified gain from fixing this is small: only 16 of 46 stall handoffs had pending segments.
5. **Slow primitives (medium).**
   - The contact profile is 51–67 % of all actions, at about 1.1–1.3 mm per action, including free-space lifts and retreats.
   - Rotation realises about 0.023 rad per step, about 3× slower than translation.
6. **Context cost (medium, second-order).**
   - Input grows by about 6.5k tokens per call.
   - Each 1k input tokens costs 18–20 ms once reasoning tokens are controlled for: about 6 % of T04 model time and 14 % of drawer 001's.
   - Caching is 1–2.5 %. No stable `prompt_cache_key` is sent (each request gets a new server key), and even strictly append-only calls 2–6 got 0 cached tokens in 4 of 5 cases. From call 7 the sliding image window also rewrites the prefix, and the transport rejects `prompt_cache_retention` and `max_output_tokens`.
   - About 15 % of each request is the dead macro layer.
   - Latency model: L ≈ 6.6–6.9 s + 0.051 s per reasoning token + 0.007 s per visible token + 0.019 s per 1k input tokens (R² ≈ 0.94).
7. **Serial startup (low/medium).** The plan call takes 14–52 s before the first action; its macro content is never used (0 macro grants in 7/7 runs). A second simulator start adds 5–9 s.

**COLLISION**

8. **The router checks one point on one stage type (critical).**
   - It checks only the grip-site point, and only on transit `move_to_pose` / `move_relative` stages (`_routing.py:3-4, 105-108`; `_fragments.py:144-145, 293`). Hand, fingers, held object and arm links are never checked.
   - 76 % of the 17,682 forbidden samples (27 episodes) happened in unrouted stages: contact-profile stages 73 % (approaches 48 %, gripper close/open 24 %) and direct motion 2 %.
   - The T08 finger press happened on a routed, straight-valid stage. The fingertip edge sat 52.7 mm from the grip site, and the open fingers reach ±54–66 mm.
   - Even with the *true* bottle box, the frozen router's A* path is the same descent to within 1 mm. A point router cannot prevent this contact.
9. **Model-written boxes are wrong (critical).**
   - T04 can box: IoU 0.116, never revised.
   - T08 bottle box: IoU 0, with its top 6.6 cm below the bottle top and its bottom 10 cm below the table.
   - Boxes were corrected only after a contact or a close-up view.
   - A* never ran: 0 detours in 117 routed stages.
   - Better boxes are necessary but not sufficient. Replaying with the improved post-contact T08 map (step 82, IoU 0.196), the router still plans within 1 mm of the colliding descent (see item 8).
10. **The palm envelope is never given (high).**
    - The prompt gives pad size and gap only (`_agent_client.py:228-233`), although `robot_facts_v2.json` has the 31 mm palm offset.
    - Commanded grasp depths are 5.1–6.2 cm below the rim.
    - Palm-on-target is 55 % of samples, in 5 of 7 runs.
    - Grasp depth also depends on the model's localisation. H3 used the same prompt, also without palm information, yet grasped 15 mm shallower and had 0 palm contacts.
11. **No contact sensing (high).**
    - The monitor is record-only, and feedback says "cause unknown" (`_fragments.py:482`).
    - A stall needs at least 16 actions in the segment with < 0.8 mm of weighted improvement over the last 12, and rotation can mask it.
    - Replayed offline, a 3-step proprioceptive guard fires inside 15 of 27 contact episodes, a median 4 steps after onset. It shortens contacts but cannot prevent them, and it cannot see contacts during `set_gripper`.
12. **Unmodelled parts of the motion (medium).**
    - The held object is not modelled. The router accepts a T04 carry at z 0.55–0.56 m, where the mug hangs below the can top; T04's 0 can contacts came from the model's own high carry (z 0.62–0.64 m), not from routing.
    - Rotations are not checked.
    - Closing and placing run open-loop.
    - The map is 4 unlabelled boxes, and the cap was hit in 10 of 36 updates.
    - The 8-corner lattice rule (`fine_distance_planner.py:101-106`) adds a hidden 0–20 mm margin. It made 5 of 7 start refusals spurious; in the other 2 the start was inside the model's own box.

**FEW ATTEMPTS**

13. **Vetoes waste calls (medium).**
    - The 1.2 rad stage rotation envelope rejected the first fragment in both T04 and T08 (1.572 / 1.569 rad), so the first plan was invalid in both.
    - `route_start_blocked` refusals came mid-episode, 5 of 7 with the start 6–20 mm outside the expanded box.
    - Each veto costs a repair call: 7.2–7.7 s after a rotation veto, 10.3–15.4 s after a start-blocked refusal.
14. **The initial plan is an unrecoverable gate (high).** Auto-approve validation aborts on any issue (`agent_speedup_v2_user_check.py:121-122, 367-371`). A real uc7.1 t9 draft fails it, and there is no re-ask.
15. **No reachability or joint-limit model (medium).** In drawer 001 the elbow sat at its limit, followed by a 9-call recovery (143 s).
16. **The transport ends episodes on recoverable errors (medium/high).** 401 `token_expired`, 503 `unknown_error`, 520 and deadline errors are not retried (`agent_speedup_v2_client.py:903-914`). This cost 2 launches.

**ZERO-SHOT**

17. **A two-sided evaluator oracle (high).**
    - The episode stops at the first raw success, with an object still held in 12/12 known successes.
    - LIBERO's site regions accept an object up to 10 cm above the site, which credited a mid-carry state (t6 relayout) and a rim-perched one (t0).
    - Two prompt sentences tell the agent it is not done (`_agent_client.py:205-209`): "Only the independent evaluator terminates a successful task" and "If observation continues…". T04-20 mm used that cue after its `finish`, for 9 more calls and 482 actions.
    - There is no `give_up` and no cap.
18. **Never swept (high).**
    - The frozen build has never run any of the 21 registered official tasks.
    - As packaged, it reproduces only 1 of its 5 evidence runs: the composite runs needed main-tree evaluator code.
19. **Closed registries and hard-coded facts (medium).**
    - New sentences or layouts need code edits.
    - The camera validator accepts only an ideal MuJoCo pinhole (`astranomy_agentic_geometry.py:79-128`), so a real calibration is rejected.
    - The gripper must be the Panda.
    - The prompts contain LIBERO-derived examples.
20. **No metric grounding tool (high).**
    - The prompt says "Pixels specify rays; infer depth" (`_agent_client.py:221-226`).
    - 28 % of calls follow a stall. That is an upper bound for grounding-caused calls: 22 of the 46 stalls were free-space transit moves, and 37–41 % of stalls are overshoot artefacts.
    - Region extent and "on" precision are undefined. LIBERO's `on` needs an origin-to-origin distance under about 3 cm.

Older defects:
- **Fixed:** direction words (now the viewer frame), image mirroring, schema reordering.
- **Still present:** the measured-pose quaternion sent to the model flips sign between requests. This is harmless for H1 control.

### 3.2 Improvements, ordered for safety

| Step | Change | Expected effect (estimate unless verified) | Caution |
|---|---|---|---|
| A1 | Terminal `finish` + `give_up` + an evaluation cap recorded as failure. Score strict success at the agent's own finish, and make release part of the completion contract. Delete both oracle cues in `_agent_client.py:205-209`. | An honest baseline, at +1–2 calls (+20–40 s) per success. | Do this first. Keep LIBERO first-success and time-to-finish distributions as secondary metrics: a 180 s failure cap would fail every current H1 run. |
| A2 | Move the video out of the loop (separate process, or post-hoc re-simulation). Render raw frames at a low fixed rate into shared memory, PNG-encode only the frames sent to the model, and let perception subscribe to the raw stream. Drop the redundant decodes. Write `run_state` only at decisions. | T04 −117…−131 s (402 → ~275 s); larger on long tasks. | Rendering *only* at decision steps would starve perception tracking and D1's asynchronous checks. The replay validator compares encoded RGB; switch it to decoded pixels. |
| A3 | Refresh and resend on 401. Treat 5xx, 520 and deadline errors as transient: same observation, nothing committed. | Removes infrastructure losses (2 seen). | — |
| A4 | Relax the macro-only plan validation and re-ask once on an invalid draft. | No t9-type aborts. | Folding the plan into the first executive call saves 14–52 s, partly offset by a longer first call. It must keep the reviewed goal contract (`approved_plan`, `reviewed_goal_contract`), the frozen view-frame and region semantics, and the object nouns and cells perception needs. It also drops the human-review option, so it is the user's decision. |
| A5 | Make the build sweepable. Pass the task as data (sentence + BDDL/layout) and record a build digest. Package the obstacle variants. Accept a real calibration file instead of the ideal-pinhole validator. Run a no-model reset-hash preflight over the 21 official tasks. Then run a baseline all-task sweep. | Turns "never swept" into a measured zero-shot baseline for every later change. | Reset equality is validated only for T04 + can today. Record the reset hash and refuse only on a mismatch with a known hash. |
| B1 | Replace the 1.2 rad veto with executor sequencing: rotate at a cleared pose, or couple rotation to translation. Check the *executed* orientation profile. | About −7…−8 s per occurrence; prevents T08-type finger sweeps. | Removing the veto alone would not prevent the T08 contact (rotation lag). |
| B2 | Exact point-to-box distances + an explicit, reported margin + a checked escape whose exit leg strictly increases distance from the unexpanded box. | Removes the spurious start refusals (5/7). | The lattice rule was a hidden margin, so replace it rather than just deleting it. Don't escape vertically by default. |
| B3 | Define a stall by a windowed test of realised progress along the commanded direction. Make the tracker's zero-translation deadband smaller than the 4 mm arrival tolerance. Add braking-aware tracking (offset ≤ k·√remaining). Add `stop_on: contact` as a stage terminator. | −5 directly avoidable calls in these runs; correct feedback. | "Speed ≈ 0" alone still fires at overshoot apexes and misses an arm moving away from its target. Many contact stalls are protective, so continue only when contact was declared. |
| C1 | Guarded motion: a translation-progress guard → hold or back off, then `blocked_motion` feedback (direction, depth, nearest part). | About 70 % less pressing time. | It fires *after* onset: it shortens contacts but does not prevent them. As a naive model handoff it would add about 24 calls across these runs, a net slowdown. Treat in-band creep as arrival, resolve contact-free blocks (joint limits) locally, and express thresholds relative to the measured OSC realisation (about 0.24 of the command). |
| C2 | Give the tool envelope (palm, fingertips). Bound contact descents by a measured height map under the palm footprint. | Targets the 55 % palm class; in T04 the bound sits 4–21 mm above the contact onset. | A bound from a model-estimated top inherits R1's errors. A rim − 31 mm bound is 5–14 mm shallower than H3's successful grasps, which risks weak grasps. |
| C3 | Metric grounding outside the LLM. Add a `locate` tool (triangulation of pointed pixels; one view + support plane). Add automatic perception boxes for non-target objects from a fixed H3 locator. Overlay the boxes on the images. | First boxes and targets within about 1–3 cm. | Triangulation needs unique masks in both views (3/22 frames today). One view needs a measured support height. |
| C4 | Body + held-object swept check on all arm stages, using H2's `robot_geometry` FK hulls, run as a planner (escape, detour, goal-as-contact). Object IDs with per-part contact permissions. Attach the held object at grasp. Once this is in, allow cruise speed only on stages with body-swept clearance, scaled to that clearance. | Catches the routed finger/link contacts and the carry-height risk. | With the model's own boxes it would refuse 21/111 stages, so add it only together with C3. Without object IDs it cannot cover the palm class: the prompt removes the target from the avoid set. It must sweep the controller's executed rotation timing (B1). H2's checker costs 0.11–0.24 s per configuration (107–136 ms in K2), so a vectorised checker with a broad phase is needed. |
| C5 | IK and joint-limit feasibility at stage entry; margins in the feedback; a posture-recovery primitive. | Avoids recoveries of about 9 calls. | — |
| D1 | Commit whole pick-and-place sequences with measured continuation predicates: a finger-gap band after close, and no gap collapse while lifting. Raise the fragment caps. Verify asynchronously. Keep blocking checks only for placement relations and articulated engagement. | About −4 calls (−60 s) on T04. | Only after C1: otherwise contacts are detected only by stalls. Grasp failures in these runs were gap-detectable. |
| D2 | Distance-scheduled approach speed. The slow zone is the measured uncertainty plus braking distance. | About −10…−67 s per task at a 4 cm standoff. | Only after C1–C3. 3 of 15 contact onsets began 2.2–6.8 cm early, and a guard firing 3 steps late at transit speed means 2–4 cm of pushing. |
| D3 | A stable per-episode cache key, if the transport allows it. Append-only history with images stripped at insertion. A cacheable static prefix. De-duplicated feedback with reasons kept verbatim. Trim static boilerplate. | −10…−30 s on T04; larger on long tasks. | Compaction already truncates measured evidence (e.g. `measured_pose` reduced to xyz). Don't cap the belief field without testing: it is the model's cross-call memory. |

On the current raw-success metric, T04 drops to about 170 s only if A2 and D1 both work together with B1–B3; A2 + D1 alone give about 190 s. It drops to about 130 s if A4, D2 and D3 also land. A1's finish scoring adds back about 20–40 s, and a slow service window like 11-05 adds about 60–90 s. **Under the honest metric, the 3-minute target needs nearly every lever.** T08 is about 3.5–4.5 min and drawer about 6–9 min. All call-count effects are hypotheses until an all-task sweep on one frozen build measures them.

---

## 4. H2: `foundation_harness` (v1 ran; keypose-2 is current)

### 4.1 Why v1 executed nothing (run 001, 2026-11-07)

Stacked blockers. Each of the first three alone was enough to block every operation; the last two blocked every grasp and every placement.
1. **The fixed base collided with its own depth points.** The self-filter used 3 mm and the collision margin 6 mm.
2. **Unknown space was vetoed.** Only 0.27 % of the tabletop grid could ever be "estimated free".
3. **Arm "ghost" depth points** 6–50 mm outside the hulls blocked every IK-feasible grasp.
4. **Finger hulls were not exempted** for the grasped object.
5. **The placement gate used the whole visible XY extent**, inflated to 24–26 cm by edge pixels.

The executor collapsed 7 of 8 rejections to `no_complete_operation_path`. Astra spent 12 calls (17.5 s cycles) varying semantic knobs that could not change the outcome. There was no attempt budget.

### 4.2 Status of the v1 defects in keypose-2

| Defect | Status in K2 |
|---|---|
| Start config / link0 | **Fixed for link0** (silhouette self-mask; base exempt). The start-endpoint veto now traps contact states instead (K3), so one analysis rates this "replaced by a new risk". |
| Structured failure reasons | **Fixed.** Reason, evidence and keypose index reach Astra. |
| Placement / grasp synthesis gates | **Replaced by a new risk.** Removed; Astra writes the metric grasp and place poses. |
| Unknown-space veto | **Replaced by a new risk.** Unknown space and every single-view ("uncertain") point are now traversable. |
| Robot/phantom depth | **Partially fixed.** Near links are removed; far links and background are demoted to "uncertain", not removed. |
| DA3 bias, SAM prompt and identity, synchronous perception | **Still present.** `perception.py` is unchanged, and per-step cost is now higher. |

### 4.3 Keypose-2 problems

**FAST**
- **K1 (critical): perception re-runs on every control step.**
  - A refresh fires when the frame is older than 2.5 s (`perception.py:181-183`; `executor.py:439`).
  - A step costs SAM + DA3 (1.83 s) + the new self-mask (1.07 s) + motion checks (0.26 s) + runner work, about **3.5–3.7 s**. That exceeds 2.5 s, so in steady state every step refreshes.
  - With 20 mm / 3 mm caps and stop-and-go waypoint gating (`executor.py:31-33, 67-88`), the planning-loop analysis models 1,040–1,400 actions for T04, i.e. **about 60–85 min of control**. That action count matches uc7.1-T20's 1,276 at the same 20 mm cap.
  - The perception analysis assumes full step realisation and estimates 13–17 min.
- **K6 (high): the RRT cannot work within its budget.** One configuration check costs 107–136 ms, so the 12 s budget buys about 12 iterations. The fixed seed repeats the same search.

**COLLISION**
- **K4 / F2 (critical): only two-view-consistent "strong" points block.**
  - About 90 % of points are "uncertain" and allowed (`motion.py:31, 110-130, 294-306`).
  - Offline intrusion tests:
    - fingers 2.7 cm into the table are allowed;
    - the red distractor mug is 2.5 % strong and allowed at every height tested;
    - the right plate is 0 % strong.
  - The can was 57–95 % strong at the home pose (depending on frame and analysis) and 0 % at an approach frame. There is no memory across frames.
- **F3 (high): phantom robot and background depth.** Far links and the wall survive as "foreground" at z 0.6–0.84 m. These are the link5/link6 "obstacles" in the prep debug files. They are demoted, not removed.
- **K2 / F1 (critical): the held object is either self-blocked or invisible.**
  - In all 19 saved frame checks with a usable payload, the payload hull contains the mug's own unowned strong points (211–9,650 points). Lift and transport therefore fail at their start configuration.
  - When the coherent surface is empty (3 of 22 frame-object checks in one analysis; 9 of 34 revisions in the other), the payload is absent and the held object is not collision-checked at all.
- **K11 (low).** The joint path is checked, but Cartesian OSC deltas execute it.

**FEW ATTEMPTS / ZERO-SHOT**
- **F5 (high): targets block their own grasps.**
  - Fingers may touch only *owned* target points, and the hand is checked against all strong points (`motion.py:267-293`).
  - Unowned target points (mask conflicts, a wrongly auto-bound wrist mug) and a DA3-raised rim veto valid grasps.
  - The white mug is blocked at every depth when the wrist identity is wrong, and from 1.5 cm below the rim even when it is owned in both views.
  - Each veto costs a call.
- **K3 (critical): once in contact, the robot cannot leave.**
  - The start configuration itself is vetoed (`motion.py:416-426`), and contact exemptions depend on the keypose name (`:518-521`).
  - From H3's real, contact-free grasp pose, every next keypose is unsafe at its start.
- **K5 (high): Astra's metric keyposes are used verbatim.**
  - Nothing snaps them to measured surfaces; `approach_direction` and `placement_relation` are validated but never used.
  - This re-imports H1's 3–6 cm errors.
  - The planning surfaces carry DA3 bias: plate 1.5–2.5 cm low, mug rim 1.4–2.5 cm high.
- **K8 (medium): failures go to Astra even when it cannot fix them.**
  - There is no call cap and no give-up, and `finish` keeps calling.
  - A schema error ends the episode with no re-ask.
  - A region box beyond the raster raises an uncaught `ValueError`.
  - A displacement is re-based on every replan.
- **F8 (high): target grounding still costs calls.** The bootstrap grammar splits "yellow and white mug" (7 of 10 LIBERO-10 sentences give malformed seeds). Identity costs about 1 call in 4. There is no class-agnostic obstacle discovery.
- **Quaternion sign.** The measured-pose quaternion sent to Astra (`robot_pose_wxyz`) is not sign-canonicalised and flips sign between requests (1 of 12 H2-v1 transitions).
- **Tests.** There are 5: client protocol, perception timestamps, the planner on an empty scene, a policy with a mocked planner, and video. None exercises real scene points. The prep checks cover only the first transit keyposes from home.

**Readiness for the pending T04 launch.** The first approach keyposes will probably execute. The likely course: very slow motion → grasp keyposes vetoed by rim or unowned points → closing flagged → lift and transport failing at the start configuration → user stop.

The two analyses give different estimates:
- **Planning-loop analysis:** 8–20 calls in the first 15–30 min, with 0–1 mugs moved. If everything worked, about 5–7 calls and 60–85 min of control.
- **Perception analysis:** more than 10 calls and more than 10 min without completing T04. About 13–17 min of control even with the lift dead end fixed.

**Recommendation.** Before the launch, make the two small fixes (payload by geometric membership with no skipped held-object check; no empty surfaces from `surface.py`) and check F2/F3 on the offline intrusion test. If it is launched as is, treat it as a diagnostic with a manual call/time stop, and watch per-step `decision_latency_s` and `endpoint: start` failures.

### 4.4 Improvements, ordered

1. **Perception off the critical path.**
   - Judge freshness by scene events, not wall time.
   - Refresh asynchronously at keypose boundaries, at 1 Hz or less.
   - Cache the self-mask per frame and voxel-downsample the cloud.
   - Drop the always-allowed free-space query.
   - K1: about 0.4–0.6 s per step.
2. **A rendered-robot image mask with depth replacement.**
   - Robot pixels get the FK-rendered depth; dilate 2–4 px; filter edges.
   - Then use robot pixels as metric anchors for a per-view affine depth correction. Offline, that took the wrist table error from 6.6–7.1 cm to 1.6 cm and external cross-view support from 5 % to 39 %.
3. **A persistent fused occupancy map.**
   - Ray carving, carving of the robot's measured swept body, shadow volumes behind observed surfaces, and an explicit support plane.
   - Single-view surfaces block within their along-ray uncertainty. "Uncertain" is reserved for unobserved space.
4. **Payload by geometric membership at closure.** Use a conservative fallback envelope, and never skip the held-object check.
5. **Replace the start-endpoint veto with "clearance must not decrease".** This removes a veto without adding a threshold.
6. **Failure handling.** Failure-signature de-duplication, local escalation for unfixable classes, a call budget + `give_up`, a re-ask on schema error, and clamped region boxes.
7. **A saved-frame replay suite as a cheap pre-launch check.** Minutes on CPU, no model calls; a development aid, not a hard gate. It should check: start is safe; ±5 cm moves plan; the lift after a close plans; the can and the table block; per-step time.
8. **Speed.**
   - A fast collision representation (voxels plus robot spheres, vectorised) and Cartesian detours.
   - Lookahead tracking with braking-aware deceleration, and cruise speed scaled to the swept clearance. H1's plain 0.10 m cap overshoots by 10–28 mm and raised p99 transit speed from 0.12 to 0.53 m/s.
   - Guarded contact moves.
   - A compact keypose contract.
9. **The harness grounds metric poses.** Snap Astra's grasp and place points to measured surfaces. Propose gripper-aware grasp candidates and support-polygon place candidates as numbered marks; Astra selects or nudges.
10. **Ownership by geometry, not SAM masks (F5).**
    - Assign every strong point in the target's coherent-patch neighbourhood to the target, and mark overlaps as shared.
    - Erode masks by 1–2 px and use robust upper envelopes so rims are not raised.
    - Never auto-bind a unique candidate that conflicts with the other view's instance.
11. **No empty surfaces (F6).** Replace `surface.py`'s comparable-components veto with the union, or labelled patches, of the supported components, so payload checks and tracking never switch off.
12. **An instance inventory instead of grammar seeds (F8).** One category-level inventory plus a class-agnostic object sweep with stable IDs. Astra binds roles once by ID, and every non-role instance becomes an obstacle automatically.

Even after items 1–8, the estimates for T04 range widely:
- **About 100–150 s (first round).** This assumes 120–180 actions per object, whereas H1 needed about 275 per mug.
- **Several minutes (K2 planning analysis).** After item 1 a step costs 0.4–0.6 s, which is about 10–14 min at today's action count, and roughly half that with lookahead tracking.

No H2 motion has ever executed, so neither estimate is validated.

---

## 5. H3: `foundation_vision` (vision-conditioned uc7.1)

### 5.1 Problems

- **V1 (high, zero-shot): plan part phrases are the SAM prompts.**
  - `live.py:79` passes text such as "white mug: loop handle on its side" verbatim. Six such prompts (three per run) returned 0 candidates in every one of their run's 11 frames, in both views.
  - Compound or relational prompts cause 61–67 % of target-frame failures. The simple nouns `can` and `bottle` were detected in 10–11 of 11 frames.
  - Two-instance categories are dropped as ambiguous (`engine.py:144-153`).
  - The plan's grid cells, which resolve every observed ambiguity, are discarded.
- **V2 (high, zero-shot): the obstacle is named by the operator.** It is perceived only via `--vision-query can/bottle` (`live.py:182-184`), a per-scene privileged hint. The T08 plan never mentions the bottle.
- **V3 (critical, collision): advisory only.**
  - `automatic_depth_routing = False`, and `query_path` is never called (`scripts/evaluate_foundation_vision_task.py:102-109`). Astra still writes the boxes, and routing is H1's grip-site router.
  - 2 of 22 responses mention vision, both to say it is unavailable.
  - Yet H3's first obstacle boxes covered the true obstacle fully with clearance in 2/2 runs (H1-family: 0/4; IoU 0.28 and 0.15 against 0.12 and 0), and H3 had 0 obstacle contacts. The plausible channels are the operator-named obstacle's SAM 2D box (privileged) and H3's extra obstacle-map instructions (`live.py:34-55`). Neither has been tested.
  - The T08 400 samples (pot–burner, at touchdown and release) are the same structural class as in H1.
- **V4 (high): the gates discard correct evidence.**
  - A target needs a unique mask in *both* views plus ≥ 0.9 bidirectional mask agreement (`geometry.py:34-36, 213-241`).
  - That is unattainable for partial wrist masks: even with true geometry the bound is 0.72–0.74.
  - The external view's own depth on the can was accurate to about 4 mm median, but it is dropped as single-view.
  - The result: 0 usable 3D target centres in 22 requests.
- **V5 (high): depth.**
  - The two-view baseline-ratio scale (`backends.py:315-354`) leaves wrist depth 18 % too deep and external depth with a range-dependent tilt. Most "contradictions" are background.
  - Robot self-surfaces stay in the cloud, so offline a gripper-width path from the hand is "blocked" (tube radius ≥ 2 cm). `assess_path` has no live caller.
  - Free air is almost never `estimated_free`.
  - Fixing the prompts alone would still give zero usable geometry; the gates must change too.
- **V6 (high, fast): vision sits in the request path.**
  - It runs synchronously: +1.1–1.3 s per call, about 14 s per episode, plus 23.5–24.6 s of model preload per process.
  - The about 2.9k-token summary is re-sent in the history forever, about 30 % of all input tokens.
  - The ~2× per-call latency against H1 is **not** vision compute. It is mostly a slower service window (+57–90 s) plus 2× reasoning tokens of unexplained cause; H3's added instruction text is an untested confound.
- **V7 (medium): wall-clock staleness in a paused simulator.** Frames are 24–86 s apart against `max_age` 5 s, so tracking never engages. A same-step re-request blanks the geometry.
- **V8 (medium): fail-closed.** A vision exception aborts the episode (`live.py:257-262`).
- **V9 (medium): never validated against truth.** All tests use synthetic depth and full masks.

### 5.2 Improvements, ordered

1. **The cheapest next test.** Run same-window, same-seed arms: H1; H1 + H3's instruction text only; H1 + SAM 2D boxes only. If the boxes help, feed automatic perception boxes, not longer prompts.
2. **Category prompts and instance binding.**
   - Strip `: part` qualifiers and spatial words.
   - One prompt per category; de-duplicate masks; keep every instance.
   - Bind roles once, using plan cells, viewer-frame directions or a set-of-marks pick.
   - Segment parts inside the bound instance.
3. **Uncertainty instead of vetoes.**
   - Prefer DA3-free epipolar ray-gap association: same-instance gaps are 0.5–2.1 cm, wrong-instance gaps 22–49 cm. A containment test on the smaller mask would be a veto re-tuned on only 10 cases.
   - Single-view hypotheses (ray ∩ estimated support plane) with along-ray uncertainty.
   - Triangulation across wrist poses.
4. **Depth and self.** FK-rendered robot mask with depth replacement; per-view affine correction anchored to the external view and the support plane.
5. **Consume perception in the executor, not the prompt.**
   - Automatic boxes for non-target objects into H1's router, excluding the held object, the target and the robot.
   - Class-agnostic discovery (connected components above the support plane) instead of `--vision-query`.
   - A held-object model and placement verification.
6. **Asynchronous and compact.** Run vision in parallel with the model call. Send only the current compact facts; drop the scene points; strip superseded summaries; keep models resident across tasks.
7. **Event-based freshness; fail open.** Forward `need_observation`.
8. **A truth-scored offline perception benchmark.** Build it from saved request RGB plus calibration, and gate changes on it.

---

## 6. Measurement fixes needed before comparing harnesses again

1. **Score at the policy's own `finish`** with strict physical checks, and have the policy (not the evaluator) do the release. Keep the raw-success step as a diagnostic.
2. **Report contact episodes by class and phase, not sample totals.** Sample totals inverted the T04 ranking: H1 at 3,372 samples never touched the can, while T20 at 1,325 did.
   - Separate palm contact with the current target from contact with other goal objects. `robot_nonfinger_manipulated_object` merges them today.
   - Decide how to count intended palm pushes on articulated targets (classified forbidden today).
   - Tighten the strict check: it accepted an astranomy mug resting against the obstacle.
3. **Log server tok/s per call and interleave arms in the same service window.** Decode speed ranged 31.8–61.1 tok/s across runs.
4. **Repeats.** At 14 scenes × 3 repeats, a 20 pp success difference is detected only 26–38 % of the time, and even a 30 pp difference only 62–77 %. Plan for more repeats, or treat differences under about 30 pp as unresolved.
5. **Frozen, hashed configuration.**
   - The transit cap is an environment variable whose default is 0.02.
   - No per-task flags.
   - Relaunch infrastructure failures and report them separately.
   - A seeded on-path obstacle variant for every relayout.
6. **Pre-register a decision rule across the four goals.** For example: zero-shot gate first, then collision-free strict success, then time, then calls. Budget about 20–27 h serial and 2,500–3,000 calls for about 95 episodes.
7. **Offline checks first.** H1's deterministic replay, compared on decoded pixels, plus an H2/H3 saved-frame replay.

## 7. Files

- `work/w1/`: first-round reports (h1_speed_attempts, h1_collision, h1_zero_shot, h2_perception_geometry, h2_planning_execution, h3_vision_engine, h3_live_integration, cross_eval_methodology).
- `work/w2/`:
  - verifiers: verify_h1_speed, verify_h1_collision, verify_h1_zero_shot, verify_h3, verify_cross_numbers;
  - K2 analyses: h2k2_perception_geometry, h2k2_planning_loop;
  - cross_critic: contradictions, improvement conflicts, gaps, cross-harness patterns.
- `work/scripts/`: the offline scripts behind the numbers. They expect the scratch layout, so adjust paths before re-running.
- `work/h2_keypose2_snapshot/`: the hash-pinned K2 source that was analysed.
