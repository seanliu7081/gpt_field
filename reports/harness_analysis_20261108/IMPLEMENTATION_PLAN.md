# Implementation plan: uc7.2 measured-geometry harness (2026-11-08)

**Direction you approved:** build on the obstacle harness's loop (H1, uc7.1 transit100), run foundation_vision's perception as a service inside the controller, reuse foundation_harness's robot geometry for path checks, and take metric estimation away from the model.

**How this plan was made.**
- The starting point is the verified analysis, `README.md` in this folder.
- Six designs, one per workstream, were written against H1's code at `3cc84ac`, with every cited line checked.
- An integrator merged them into one plan, `plan/INTEGRATED_PLAN.md` (≈ 200 KB, with every interface schema).
- Two adversarial reviewers then checked that plan:
  - the feasibility reviewer found it buildable;
  - the rules-and-safety reviewer found it mostly faithful to your rules and the safe order, with 7 must-fix defects, 4 of which left unsafe interim states.
- Together the reviews raised **10 must-fix and about 35 should-fix items**. A final consistency and actionability check of this document added further corrections. Everything is folded in; §9 lists each correction and the integrated-plan section it supersedes.
- Nothing has been implemented or run: no simulator, model or GPU job.

**How to read it.**

| Sections | Content |
|---|---|
| §0–§2 | what gets built and why |
| §3 | the six milestones |
| §4 | measurement |
| §5 | effort and staffing |
| §6 | your decisions |
| §7 | the first weeks |
| `plan/INTEGRATED_PLAN.md` | the detailed specification: schemas, file lists, tests. Read it together with §9 here; where they conflict, this document wins. |

---

## 0. In one page

**What gets built.** A new repository `agent_speedup_v2_uc72_harness`, cloned from H1. H1 stays byte-identical as the A/B baseline. On top of H1's agent-first loop come five layers, each behind config flags:

| Layer | What changes | Fixes (analysis refs) |
|---|---|---|
| 1. Honest measurement + fast evaluation loop | Scoring at the policy's own `finish`/`give_up`; contact *episodes*; per-call service speed; a shared-memory frame ring instead of per-step PNG round trips; video out of the loop; one simulator; transport retries | A1–A5, R6, H1 #1, #7, #16–19 |
| 2. A measured executor | Fitted OSC plant; braking tracker; a progress guard with `stop_on: contact`; exact box distances with a checked escape; rotation sequencing instead of the 1.2 rad veto; reachability; grasp-gap predicates; compact, append-only context | B1–B3, C1, C5, D3, R5, H1 #3–6, #11, #13, #15 |
| 3. Perception as a resident, asynchronous, fail-open GPU service | SAM 3.1 + DA3 kept loaded; FK-rendered robot mask; measured support plane; association by geometry (no 0.9 two-view gate); an instance inventory bound from plan cells; class-agnostic obstacle discovery; held-object tracking | H3 V1–V9, R3, C3, K2 F2–F4 |
| 4. Metric grounding in the harness | One harness object map with IDs and σ. The model references objects (`o3`, plan part `A`) and points at pixels (`locate`); the harness resolves metres at the moment of use. Model-written obstacle boxes are removed. | R1, C3, H1 #9, #20, K5 |
| 5. Body + held-object swept planner, then fewer calls | Numpy-only, built from H2's robot geometry (372–600 spheres; FK matches H2's to 2e-16). Runs on every arm stage as a *planner* (escape, sequencing, detours, standoff handoff), never as a veto. Then committed sequences, asynchronous verification and measured approach speed. | R2, R4, H1 #2, #8, #10, #12, C2, C4, D1–D2 |

**Milestones.** Each milestone M1–M5 ends with a sweep of one frozen build, run in the same service window:
- all tasks at S1, S3 and S5; `regression12` at S2 and S4 (UD-11);
- S1 compares pristine H1 with measured H1; S2–S4 compare against the previous build; S5 compares against the M1 baseline.

| M | Name | Ends with | Effort (est.) |
|---|---|---|---|
| M0 | Foundation and equivalence | Offline proof that the flags-off build replays H1's T04 reference exactly; reset preflight of every scene; one ≤ 3-call smoke | 28–31 d |
| M1 | Honest measurement baseline | **S1**: all-task coverage map with honest scoring, pristine H1 vs measured H1 | 25–27 d |
| M2 | Veto replacements, sensing, compact context | **S2** | 41–44 d |
| M3 | Perception service + measured geometry | **S3** | 81–86 d |
| M4 | Body + held-object swept planner | **S4** | 24–27 d |
| M5 | Committed sequences, asynchronous verification, approach speed | **S5 = acceptance** (frozen build vs M1 baseline, 3 repeats) | 17–19 d |
| | **Total** | | **≈ 215–235 engineer-days** |

**Calendar and cost.**
- **Critical path (est.):** ≈ 70–80 working days, i.e. ≈ 14–16 weeks (≈ 3.2–3.7 months), with five parallel tracks. It is the feasibility review's 65–72 d plus the must-fixes on the perception and planner path. It runs through perception.
- **Sweep cost (est., ±30 %):**
  - `full34`: ≈ 10,000–16,700 subscription calls and 64–113 h of serial sweep time for S1–S5.
  - Recommended `full44` at S1/S3/S5: ≈ 12,600–21,100 calls and ≈ 81–143 h.
  - Not included: the model-only A/Bs and calibrations (§4), and the options of a third S5 arm (S-M7, V-S14), each ≈ 2,700–4,400 calls and 17–30 h.

**Expected end state (est.).**
- Honest T04 policy time ≈ 150–230 s. Today H1 takes 402 s wall, of which 304.8 s is control + plan without video. S1 measures H1's honest policy time on the same clock, so "402 → 150–230 s" would overstate the gain.
- Calls per object ≤ 3 (H1: 6.5–10.5).
- Forbidden contacts in routed stages and with added obstacles → 0 (target).
- Model-written metres are limited to relative moves, direct motions and offsets from measured anchors, flagged beyond an anchor's extent + kσ. Absolute targets are removed at the M5 freeze if S3 shows ≥ 95 % reference use (UD-27).
- **The 3-minute target needs nearly every lever, including the plan-call fold that this plan defers (§8), and a normal service window.** A slow window like 11-05 adds 60–90 s. All effects are estimates until sweeps measure them.

**Early value and a stopping point (decide before Day 0).**
- **M0–M2 (≈ 94–102 engineer-days) stand on their own.** They deliver:
  - one simulator;
  - ≈ 0.04–0.07 s per-step harness cost instead of 0.28–0.40 s;
  - honest finish scoring with transport retries;
  - the first all-task coverage map (S1);
  - the measured executor (S2; honest T04 ≈ 240–290 s, est.).
- **Perception maps, the body planner and the 3-minute target come only in M3–M5 (≈ 122–132 d).** S2 plus the offline perception scorecard (PS-12) is the natural go/no-go point for M3–M5.
- **Staffing choice.** With five tracks from day 1, most of the perception and geometry effort (E4/E5, ≈ 75 d) is spent before S1/S2 report. Staffing E1–E3 first and adding E4/E5 after S2 defers that spend, but lengthens the calendar by roughly the M3 perception chain.

---

## 1. Ground rules (binding on every workstream)

1. **One frozen general framework.**
   - No per-task code, prompt words, thresholds or inputs (no `--vision-query`).
   - A task is data: sentence + scene + initial state + evaluator predicate.
   - A shared test greps policy, prompt and perception-query code for benchmark nouns.
2. **Every numeric parameter has a recorded source.**
   - The config loader rejects any number without one. The allowed sources are `inherited:<H1 file:line>`, `measured:<artifact>`, `derived:<formula over measured quantities>` and `user_decision:<UD-nn>`.
   - Vetoes are replaced by planner behaviour, not by new thresholds.
3. **Acceptance is an all-task sweep on one frozen build**, presented as a per-task coverage map. Any code, prompt or config change after a freeze creates a new tag.
4. **Live-first.** Offline checks are development aids: their results go into the tag message and never block you. Claude may run capped smokes within your UD-44 budget. You start full tasks and sweeps, and the plan gives the commands.
5. **ChatGPT subscription only** (GPT-6 Astra, reasoning high). The launcher pops `OPENAI_API_KEY`; there is no fallback.
6. **Hands off what isn't ours.**
   - H1 (`3cc84ac`), the main tree's tracked files and the live `foundation_harness` (edited by your Codex session) are never touched.
   - Needed code is vendored with hash pins: H2 `robot_geometry.py`, `robodojo_direct_sim.py`, `foundation_vision` modules, robosuite Panda assets.
   - H1's venvs are used read-only through symlinks and are fingerprint-checked before every pristine episode. The same applies to the main tree's `.venv-foundation-vision` and the editable `sam3`/`depth_anything_3` sources it loads.
   - The new controller adds no package dependencies.
7. **Evaluator-only data never reaches the policy.** This covers truth, contacts, object poses, BDDL fields and obstacle asset names. It is enforced by an import guard plus a leakage test that greps payload strings.

---

## 2. Architecture

```
                 ┌────────────────────────── controller (.venv, py3.12, numpy only) ───────────────────────────┐
 task spec ───▶  │ uc72 runner ─ executive client (Astra, ordered SSE) ─ policy:                                │
 (data)          │   SceneMap (object IDs, σ, permissions) ◀─ PerceptionClient ◀──── unix socket ─────────────┐ │
                 │   reference resolver / locate ─▶ stage planner (point_v2 ▶ body planner) ─▶ fragment executor │ │
                 │   (plant model, braking tracker, guard, predicates) ─▶ OSC delta actions                    │ │
                 └──────────┬───────────────────────────────▲──────────────────────────────────────────────────┘ │
                   step IPC │ (JSON lines; geometry in the  │ /dev/shm FrameRing v1 (raw RGB + calib + state)   │
                   step reply; one IPC per step)            │                                                   │
                 ┌──────────▼──────────────┐      ┌─────────┴─────────────────────────────────────────────────┐ │
                 │ sim worker (.venv-sim,   │─────▶│ perception service (.venv-foundation-vision, 1 fixed GPU) │─┘
                 │ py3.10, MuJoCo/robosuite)│ ring │ SAM 3.1 + DA3 resident per sweep; async; fail-open        │
                 └──────────┬──────────────┘      └───────────────────────────────────────────────────────────┘
                            └─▶ evaluator-only: contacts.jsonl, truth, object poses ─▶ post-hoc video, coverage map
```

**Processes.**
- **Controller:** H1's `.venv`, numpy only. FK/IK, the sphere checker and the geometry math are numpy/stdlib.
- **Simulator worker:** H1's `.venv-sim`. It writes the frame ring.
- **Perception service:** `.venv-foundation-vision` on one fixed GPU, ≈ 8 GB.
  - At writing only GPU 7 was idle (≈ 24 GB free); GPU 6 had ≈ 15 GB free but was busy.
  - S4 runs two resident services (≈ 2 × 8 GB, est.), which fit together only on GPU 7.
- **Video:** rendered post hoc by deterministic re-simulation.

**Shared interfaces (20, one owner each, frozen at the milestone that first needs them; full schemas in `plan/INTEGRATED_PLAN.md` §4):**

| Owner | Interfaces |
|---|---|
| Runner | I-01 config/flags/invariants/digest; I-02 task spec; I-03 FrameRing v1; I-04 service protocol; I-18 termination and scoring; I-19 timing and telemetry |
| Geometry | I-05 calibration; I-08 harness object map; I-09 geometry reference grammar |
| Perception | I-06 perception snapshot; I-20 height query |
| Protocol | I-07 plan scene inventory; I-14 decision/segment schema; I-16 outcome record |
| Swept planner | I-10 robot model API |
| Executor | I-11 executed profile + OSC plant; I-12 stage-planner seam; I-13 robot events; I-15 failure taxonomy; I-17 committed-sequence API |

**The config loader enforces the safety order** with the invariants INV-1…INV-14 and INV-D. Each INV-n enforces the cross-critic's ICn; IC8, IC11 and IC15 are structural. Examples:

| Rule | Enforces |
|---|---|
| Any capability flag requires finish-scoring | IC6 |
| The body planner requires perception + rotation sequencing + guard | IC12 |
| Cruise speed requires the body planner + braking | IC2 |
| Approach speed requires guard + `stop_on` + perception + an active depth bound | IC1 |
| Committed sequences require guard + grasp predicates | IC5 |

This plan adds **INV-DM** (S-M3): `perception.mode=on ⇒ executor.degraded_mode`. The review keyed it to `swept.speed ≠ off`. Keying it to perception also covers arms with speed scheduling off, where an outage would still leave an empty map at H1's 0.10 m transit cap.

---

## 3. Milestones

Each milestone lists what is built (designs' component IDs), the review corrections that apply (§9 IDs), checks, acceptance metrics and expected effects (est.). Exact files, signatures and tests are in `plan/INTEGRATED_PLAN.md` §6.

### M0: Foundation and equivalence (offline, plus one ≤ 3-call capped smoke; ≈ 28–31 d)

**Build.**
- **R1:** clone + tags + worktrees; config loader with sourced values; flag registry + invariants; build digest. The digest also hashes the editable `sam3`/`depth_anything_3` sources and all three interpreters.
- **R2:** a new runner that keeps H1's loop order and uses one simulator for plan + execution, with a service supervisor.
- **R3:** task-as-data for 20 official tasks, 3 obstacle variants, the composite and 10 relayouts, plus the seeded on-path obstacle generator (UD-10). Reset identity uses both PNG and pixel hashes. A no-model reset preflight runs on the new and the pristine code. The task spec gets an `evaluator.collision_base_mode` field (V-mig).
- **R7:** FrameRing v1 with the corrected seqlock (V-M1); lazy observations; geometry in the step reply; compact trace; throttled state writes.
- **R8:** post-hoc video.
- **R9:** decoded-pixel zero-call replay.
- **INT-7:** GPU index → EGL minor resolver (minors are reversed on this host).
- **INT-5:** truth exporter, moved forward from M3 (V-S9; 3.5 d). Its replays build the env through the worker's `reset()` (S-S18).

**Review corrections.**
- **V-S3:** the `action.py` cut-over happens only after the new replay passes.
- **V-S5:** a differential zero-call replay with scripted `finish`, `observe`, `motion`, `move_relative`, `wait` and re-ask answers. The T04 anchor only has `execute` decisions.
- **V-S6:** no-op seams in the shared H1 files, each with one named owner, landed *before* any other track edits them.
- **V-S7:** the worker no longer stops at the first raw success.
- **V-S11:** outputs go outside the repo.
- **S-S2:** off-box backup (UD-00).
- **S-S3:** venv fingerprints.
- **S-S13:** one ≤ 3-call smoke on `h1_equivalent`.
- **S-S18:** replay exactness on the reset-count-sensitive scenes t3/t8/t9.

**Start with a 2-day equivalence spike (§7).**

**Checks (development aids).**
- The `h1_equivalent` replay reproduces T04 exactly: 549/549 actions, state8, contact counters, decoded pixels, and all 13 request images (byte-exact data URLs).
- The differential replay matches the pristine worktree.
- Overhead: median ≤ 0.06 s/step and flat across the episode (H1: 0.158 s without video, 0.282 s with).
- Deterministic frame-ring interleaving tests.
- The preflight table, with every mismatch resolved (UD-03).

**Expected effect (est.).**
- Per-step harness cost 0.28–0.40 → ≈ 0.04–0.07 s.
- T04 wall 402 → ≈ 275 s.
- −5…−9 s per episode from a single simulator.
- Drawer-type tasks −700…−820 s.

### M1: Honest measurement baseline (sweep S1; ≈ 25–27 d)

**Build.**
- **PC1 + R4, honest termination:**
  - terminal `finish` with release in the completion contract, plus `give_up(reason)`;
  - scoring at the policy's own finish after a 10-step hold;
  - LIBERO first-success kept as a diagnostic only;
  - all oracle cues deleted, including the `task_success`/evaluator keys in model-facing payloads and the "Simulation pauses during inference" sentence (S-S14).
- **R5:** contact episodes by subclass and phase, plus obstacle displacement. `intended` is computed evaluator-side (S-S11).
- **R6:** timing schema, per-call tok/s, service-speed index, a canary call per block.
  - Policy time = simulated motion (steps × 0.05 s) + blocking waits (plan and model calls, perception waits, planner overruns) + compute beyond the control period (I-19, V-S17).
- **PC2:**
  - forced token refresh on 401;
  - transient 5xx/520/deadline errors retried on the same observation;
  - one shared credential store for every arm and tool (S-S1, V-S2).
- **PC3a:** relaxed plan gate (macro-only validation removed, one re-ask). The S1 arm keeps H1's plan prompt otherwise; the scene inventory (PC3b) comes in M3.
- **R11:** sweep harness: interleaved ABBA blocks, caps, censoring, the coverage map.
- **Pristine-arm adapter (V-M2, V-S1):**
  - calls H1's evaluator directly, with launch.py's environment, the EGL minor, the store and the pristine-code preflight hashes;
  - the runner records the cap reason *before* touching H1's stop file, so pristine failures are scored and censored, not "user stop".

**Live.**
- A capped T04 smoke (≤ 3 calls).
- A g8 run to `finish`, which is a full task and needs your go-ahead.
- **S1, which you start:** the UD-10 scene set × 1 repeat, comparing pristine H1 (oracle metric) with `measured_h1` (honest metric).

**Acceptance.**
- Every cell has a terminal glyph, and every infrastructure failure is explained.
- Per task, report:
  - strict success at finish, raw at finish, and first success;
  - policy time (median and P(≤ 180/300/600 s));
  - contact episodes by subclass;
  - calls per success;
  - 0 post-finish calls;
  - 0 infrastructure-terminated episodes after retries.
- S1 *sets* the honest baseline, which is unknown today (between 0/7 and 6/7 for the retained runs).

### M2: Veto replacements, sensing, compact context (sweep S2; ≈ 41–44 d)

Model-written boxes are still used in M2, so the geometry effect is not mixed in.

**Executor.**
- **EXC-0:** measured OSC plant, s_t = a·s_{t−1} + b·c_t, with a = 0.711, b = 0.0732, realisation ρ = b/(1−a) = 0.254 (per run 0.248–0.262), R² 0.973. `osc_response.json` is the one source.
- **EXC-B3:** stalls by realised progress along the command; no deadband inside the arrival tolerance; braking-aware tracking; `stop_on: contact`.
  - The **contact window is derived by the executor** (S-M6). In M2 (no measured σ, no object map) it is the plant-derived guard detection distance + tolerance. From M3, for a segment with a `target` reference, it adds k·σ along the motion.
  - A model-written window may only shrink it.
  - From M3, a guarded stop away from the intent anchor is reported as `unexpected_contact_other_object` and handled as undeclared.
  - Offline: replaying the 15 recorded contact onsets, the 3 early ones classify as `unexpected_contact_early`.
- **EXC-B1:** rotation sequencing replaces the 1.2 rad veto. A runtime finger/pad keypoint sweep picks the rotation site (S-S8).
- **EXC-B2:** exact point-to-box distances + an explicit margin + a checked escape (S-S7):
  - the last contact anchor may be left without increasing penetration;
  - −approach is the preferred exit;
  - no horizontal escape while holding an object on a support;
  - `route_goal_blocked` becomes a standoff handoff.
- **EXC-C5:** reachability and joint margins.
- **EXC-C1:** guard response resolved locally, with `blocked_motion` feedback.
- **EXC-D1a:** grasp-gap predicates.
- **INT-1:** `src/gpt_field/vendor/` + MANIFEST + `robot/kinematics.py`. The vendor tree mirrors robosuite's asset layout, and npz files are saved without pickles (V-S12). It may start on day 1 and is frozen with M2 (I-10).

**Geometry.**
- **GT8:** tool-envelope text (palm 31.04 mm behind the grip site, fingertip reach 9.25 mm, open-finger reach ±66 mm), rendered from one robot description.
- Until the body planner is on, it says plainly that **only the grip-site point is checked**, and that the model remains responsible for the hand, fingers and carried object (S-M2).

**Context (PC4).**
- A static cacheable prefix.
- An append-only transcript with the current image pair only (UD-20).
- The macro layer removed.
- Verbatim outcome reasons.
- `why` dropped (UD-21).
- The quaternion aligned to the hemisphere of the episode's first measured orientation. w ≥ 0 alone does not remove the flips; this removes all 33.

**A/B harness (PC7), model-only replays of saved payloads, including stove-off and drawer requests, reported per scene family (S-S17).** Each needs your go-ahead, because each exceeds the smoke limit (UD-44):
- the context A/B (≤ 40 calls; UD-20, UD-21);
- the cache-key probe (≤ 14 calls; decides UD-40 for M5);
- the H3 channel ablation (first requests of T04/T08, 3 arms × 10 samples, ≤ 60 calls; informs UD-26).

**Live.**
- Capped smokes of ≤ 12 calls.
- **S2, which you start:** `regression12` × 1, b002 vs b001.

**Acceptance (S2 vs b001).**
- Rotation-envelope rejections = 0; spurious start-blocked = 0.
- Overshoot-stall calls ≈ 0.
- Median pressing duration, onset → stop, ≤ 4 steps (b001 ≈ 13).
- 0 contact episodes during escape legs.
- Pending discards after a declared contact = 0.
- Joint-limit streaks ≤ 2 calls.
- Input tokens at later calls −60 %, with reasoning tokens per call not above b001.
- Contact episodes not above b001.

**Expected effect (est.).**
- −18…−19 s per T04/T08 from the rotation veto.
- ≈ −6 repair calls across 9 runs from exact distances + escape.
- −5 overshoot calls and −2…−3 calls from `stop_on`.
- Up to −70 % pressing time.
- Honest T04 ≈ 240–290 s.

### M3: Perception service + measured geometry (sweep S3; ≈ 81–86 d; critical path)

**Perception service (PS-1…PS-12).**
- **Service:** resident, asynchronous, fail-open.
- **Detection and identity:**
  - category-noun prompts from the plan inventory, with mask de-duplication;
  - an instance inventory with stable IDs, bound from plan cells, then direction hints, then prompt hits;
  - remaining ambiguity is resolved by the model with `scene_edits bind` on the overlay's o-ids (no numbered marks; X8, X35).
- **Geometry:**
  - FK-rendered robot mask with depth replacement;
  - support plane measured by plane-sweep parallax, with a DA3 prior and contact anchors;
  - DA3-free epipolar ray-gap association (same instance 0.5–2.1 cm vs wrong instance 22–49 cm);
  - single-view hypotheses with along-ray σ (the multi-view primitive fit is an M4 refinement).
- **Discovery and tracking:**
  - class-agnostic discovery + height field (PS-8);
  - held-object tracking (PS-9);
  - validity by events, with re-localisation at use.
- **Benchmark (PS-12):** a truth-scored offline benchmark built from saved runs.
  - σ, and UD-28's k, are fitted on historical scenes and validated per family on S1 decision frames re-rendered through INT-5.
  - They are never refit on S5 conditions; any refit is a new tag (S-S4).

**Protocol.** **PC3b:** the plan's scene inventory, `objects` (I-07): nouns + cells for every visible object, ≤ 12 (UD-30), enabled by `protocol.plan_inventory` from b003.

**Geometry tools (GT1–GT7, GT9–GT11) + INT-4.**
- **INT-4:** `camera-calibration-1` emission. The ideal-pinhole validator stays only on `h1_equivalent`.
- **SceneMap** replaces the model's 4 boxes: objects with IDs, source, σ, automatic attach/detach and per-part contact permissions.
  - The model may *dispute* an obstacle: its σ is inflated and it is restored automatically when re-observed.
  - The model may remove only single-view detections (S-S10).
- **Target references** (`target-ref-1`):
  - anchor = an o-id, plan part, inventory id, `locate` point L1–L4 or support S1–S4;
  - `point` = center/top/…/inside;
  - plus `offset_m`, `frame` and `tool_point`.
  - Pointed pixels enter through `locate`. Offsets beyond the anchor's extent + kσ are flagged (S-S9).
- **`locate`:** XYZ ± σ with no extra round trip.
- **Regions:** metric meaning for `on`/`beside`/`inside`.
- **Overlays:** drawn on the current image pair.
- **`held_identity_mismatch`** (S-S15): a barrier yield when PS-9's co-moving instance differs from the SceneMap held record. The record is re-attached, and the event counts among S5's late-detected failures.

**Swept planner (SP1–SP5, SP8 report-only, SP11 shadow, SP12)** runs in shadow on live maps. The depth bound is logged, not applied (UD-34).

**Review corrections in M3.**
- **V-M3, perception ordering.**
  - Request 1 waits, bounded, for the plan-bound snapshot. The timeout comes from the service's measured latency model, and the wait is charged to `Timers.block("perception")`.
  - For later requests, UD-50 chooses:
    - (a) a bounded wait for the decision-frame refresh, ≈ 1.0–1.9 s per decision (est.), counted in policy time. This is the safe default for M3/M4.
    - (b) a 0 s wait, with `observed_step` stamped and no overlay boxes for records not re-observed since the last robot event.
  - `scene_snapshot_frame` and `request_frame` are logged per call.
- **S-M1.** Never exclude an object for meeting the robot at the current joints.
  - Self-detection is tested only at the record's own observed joints, in 3D, with near-full containment. The containment bound comes from PS-3's measured raster recall.
  - The test never applies to an object previously observed static and clear of the robot.
  - Otherwise the object is kept, with `near_robot` and σ inflation.
  - Offline: the T08 bottle box (actions 37–83) and both T04 mugs at grasp stay in the obstacle set.
- **S-M2.** From M3 (model boxes removed) until `swept.mode=on`, `GripSiteBoxOracle` inflates each box by a **derived robot term**: the hand/finger half-extent per axis at the stage orientation and commanded gap, plus the held record's below-grip extent and footprint (or the jaw-column fallback envelope).
- **S-M3, degraded motion mode (INV-DM).**
  - At reset: a bounded, measured wait for the first `ok` snapshot. Otherwise the episode is `infra_error/service`: relaunched, unscored.
  - Mid-episode: σ grows with age, transit is capped at contact-safe speed, and cruise/approach scheduling is disabled.
  - Outage cells are flagged in the coverage map.
- **S-S16.** `unknown_volumes` are populated from the height field's shadow cells near task objects. SP9 lowers the speed cap there: logged in shadow in M3, active from M4.
- **V-S4.** GPU headroom is measured through the service's allocator. Placement is fixed per sweep, with a leak guard between blocks.
- **V-S15.** Event and decision frames are copied to a per-episode event store.

**Live.**
- Needs your go-ahead (UD-44), because each exceeds the smoke limit:
  - a pointing calibration (≤ 20 calls);
  - the overlay A/B for UD-26 (≤ 40 calls).
- Needs you to start or approve the GPU perception service: the shadow/on capped smokes on T04 and T08. The bottle must appear without `--vision-query`.
- **S3, which you start:** the UD-10 scene set (`full44` recommended) × 1, b003 vs b002.

**Acceptance (S3 vs b002).**
- **Zero-shot:** obstacle scenes run with no operator naming; perception availability ≥ 99 % of decision frames.
- **First boxes and targets:** first-call obstacle coverage with kσ in 2/2 obstacle scenes (H1-family first boxes: 0/4); first-target error by source.
- **Calls and tokens:** identity-only calls = 0; input tokens ≤ +0.6k per call (H3 added +2.9k).
- **Perception wait** per decision, reported by UD-50 mode:
  - under (a), it equals the decision-frame refresh (≈ 1.0–1.9 s, est.), checked against the service's measured latency;
  - the ≤ 50 ms p95 target applies only under (b).
- **Collision and safety checks:**
  - shadow-planner recall and precision against contact episodes;
  - obstacle contact episodes, and hand/finger/held-object episodes in routed stages, not above b002; carry margin against truth (S-M2);
  - real objects excluded as `inside_robot` (by post-hoc truth) = 0 (S-M1);
  - arm results with and without outage cells (S-M3).

**Expected effect (est.).**
- First boxes and targets within ≈ 1–3 cm, instead of 5–12 cm.
- −2…−3 identity calls per cluttered task; 1–2 fewer calls per pick-and-place.
- Added time per decision:
  - ≈ 0 s only under UD-50 (b);
  - the default (a) adds ≈ 1.0–1.9 s per decision (≈ 12–23 s on a 13-call T04), close to H3's 1.1–1.3 s per call but without H3's 24.6 s preload per task.
  - Settle UD-50 for M5 from the S3/S4 stale-box evidence.
- Collision gains stay limited until M4, because the point router still checks only the grip site.

**Lean option, if M3 is too large.** Defer discovery (PS-8) and tracking (PS-9).
- **Saving.** About 6.5 d of M3's 81–86 d (PS-8 3.5 d + PS-9 3.0 d), and about as many critical-path days. The perception design's "≈ 24 d minimum slice" is on its own 36.5 d scale, not M3's. Geometry (≈ 17 d), the swept shadow (≈ 15 d) and integration are unchanged.
- **Without PS-8:**
  - obstacles reach the map only through the plan inventory (≤ 12 objects, named by the model, never by the operator) and `locate`;
  - there is no height field, so S-S16's unknown volumes and S-M4's walls/rim container representation are unavailable, and containers need the scoped `allow_contact: inside` (UD-45);
  - SP8 stays report-only (UD-34), and by INV-1 that also disables M5's approach speed.
- **Without PS-9:** M4 uses the SceneMap held record (GT3) with its jaw-column fallback envelope, without co-motion confirmation.

### M4: Body + held-object swept planner (sweep S4; ≈ 24–27 d)

**Build.**
- **SP6:** `LocalBodyPlanner`: escape → direct (with sequencing mode) → Cartesian detours → standoff handoff → `no_route`.
- **SP7:** body-aware routing and tracking, superseding `TransitRoute`.
- **SP8:** contact-descent depth bound from measured heights per palm-footprint cell.
- **SP9:** clearance-scaled cruise, at most H1's 0.10 m cap.
- **SP10:** held-object spheres with the in-gripper offset.
- **SP11:** flags off/shadow/on.
- PS-5/PS-6 refinement: the multi-view primitive fit.

**Review corrections in M4.**
- **S-M4, containers and fixtures.**
  - Containers and front-opening fixtures (drawer, microwave, caddy) must not be filled boxes. With filled boxes, arm links never permitted and the palm permitted only for push/insert, LIBERO-10 t3 and t9, probably t5, and LIBERO-Goal g3 (3–4 of 20 official scenes) would be scored as framework failures caused by the representation. If baskets are included, the basket tasks t0, t1 and t7 need the same check; the review did not assess them.
  - **Preferred representation:** observed walls and rim from the height field, plus the opening volume seen in the external view for front-opening fixtures.
  - **Minimum alternative:** a scoped `allow_contact: inside` that permits the palm and distal links inside the anchor's box on `inside`/articulated stages, with the guard as judge (UD-45).
  - Until either lands, container stages run on the point planner + guard, with the body plan in shadow.
  - **Validation:** an SP12 replay of drawer 001/003 on live-style perception boxes, plus a t9 truth fixture.
- **S-M5, no new vetoes.**
  - A blocked jaw sweep opens or closes only to the clear width, or inserts a short guarded retreat first.
  - Direct `motion` inherits the last fragment's contact intent and is truncated only for forbidden pairs outside that intent (UD-36 revised).
- **S-S12.** Planners use a deterministic evaluation-count budget; wall time is only reported.

**Live.**
- Capped smokes (≤ 12 calls) on T08 + bottle, T04 + can and the drawer composite.
- **S4, which you start:** `regression12` × 1, b004 vs b003.

**Acceptance (S4 vs b003).**
- **Collision targets:**
  - forbidden episodes in routed stages → 0;
  - obstacle episodes in obstacle variants → 0.
- **Collision reductions:**
  - palm-on-target and held-object episodes well below b003;
  - impact speed at onset ≤ 1.5·s_contact;
  - carry height − required height ≥ 0.
- **Routing:**
  - `route_start_blocked` = 0;
  - `no_route` ≤ 2 % of stages.
- **Reported:**
  - stages handed off at a container boundary (S-M4);
  - `jaw_sweep_*` outcomes, direct-motion truncations and the calls they caused (S-M5).
- **Grasp quality:** grasp retries rise by ≤ 1 per 5 grasps.
- **Planner cost:** p95 ≤ 150 ms per stage and ≤ 3 ms per step.

**Expected effect (est.).**
- With truth-quality geometry, the offline replay prevents the T08 finger press (3,504 samples) and the arm-link vs cabinet contacts (414).
- The 21/111 would-be veto stages become local actions, with 0 added calls.
- The T04 carry window 0.54–0.57 m is closed.
- Cost: +5–15 s per task near objects.

### M5: Committed sequences, asynchronous verification, approach speed (sweep S5 = acceptance; ≈ 17–19 d)

**Build.**
- **PC5 + EXC-D1b:** committed sub-tasks of up to 12 segments with measured continuation predicates. Blocking checks remain only for placement relations and articulated engagement.
- **PC6:** asynchronous verification on a virtual clock:
  - staleness rules and barriers;
  - a cancellable in-flight call;
  - per-call metadata returned in a `CallResult` across the 9-level client chain;
  - costed at 7–8 d (V-S8).
- **SP9 (approach):** measured approach speed, only on `stop_on: contact` stages with a measured contact expectation (UD-41). The slow zone is measured σ + braking + guard distance; the executor's EXC-D2 hook (M4) applies the cap.
- **PC2:** cache key, if the M2 probe showed it is honoured (UD-40).

**Review correction (S-M7).** Whether async verification ships (UD-39) is decided from an interleaved, capped, **user-started** A/B:
- ≥ 3 task families, e.g. T04, T08, the drawer composite, one LIBERO-Goal task and one multi-object LIBERO-10 task;
- ≥ 30 verify calls per arm, reported with confidence intervals;
- async passes if late-detected failures do not rise and stale rejections stay ≤ 1 per 10 verify calls.

The alternative is to run both a blocking and an async arm in S5, with the acceptance arm pre-registered. That third S5 arm costs ≈ 2,700–4,400 calls and ≈ 17–30 h.

**S5, which you start.**
- Frozen b005 vs b001, the UD-10 scene set × 3 repeats, on **held-out conditions** (UD-47).
- The decision rule is pre-registered (UD-09, UD-48):
  1. zero-shot gate;
  2. collision-free strict success at finish (scene-clustered test, with margins);
  3. time (P(strict ≤ 180/300/600 s) and a cap-imputed restricted mean);
  4. calls.
- A pre-registered rule says how to proceed when a level is unresolved. Differences the design cannot resolve are labelled "unresolved", never "tie".

**Expected effect (est.).**
- Committed sequences: −4 calls on T04, −8 on T08, −10 on drawers. The net saving is ≈ 10 s per avoided call, because committed answers are longer.
- Async verification: ≈ −7 s.
- Approach speed: −10…−67 s, only if the depth bound is active (INV-1, UD-34).
- Overall honest times: T04 ≈ 150–230 s, T08 ≈ 3.5–4.5 min, drawer ≈ 6–9 min.

---

## 4. Measurement protocol (all sweeps)

**Scene sets** (frozen at M1):

| Set | Content | Use |
|---|---|---|
| `full34` | 20 official + 3 obstacle variants + composite + 10 relayouts | minimum full sweep |
| `full44` | `full34` + 10 seeded on-path obstacle relayouts | recommended full sweep |
| `regression12` | 12-scene subset | S2 and S4 |

- Under S-S4 the 3 obstacle variants are historical. With `full34`, S5 would therefore have no held-out obstacle scene; `full44`'s fresh seeds are the only held-out obstacle evidence.
- Every previously used scene is flagged *historical*. Held-out and historical subtotals are reported separately.

**Interleaving.**
- Each (task, repeat) is one block. Arms run back-to-back in ABBA order with a recorded seed, and a canary call opens each block.
- A quota pause stops all arms.
- Every episode re-verifies its arm's digest.

**Outcomes.**

| Outcome | Treatment |
|---|---|
| `finish` / `give_up` | scored |
| Caps (1,800 s policy time / 80 calls, UD-04) | scored and censored |
| User stops | unscored |
| Infrastructure failures | one relaunch; a second failure is `I` |
| `cap_wall` | an infra hang |
| Simulator crashes right after a forbidden contact | `sim_instability`, scored (S-S6) |

The counted calls are plan + decide + verify + re-asks. Canary calls and infra retries are excluded (UD-49).

**Statistics (S-S5).**
- Scene-clustered tests: a sign-flip permutation over per-scene differences, or stratified CMH. Not pooled McNemar.
- Pre-registered margins per decision level, plus a pre-registered rule for proceeding past an unresolved level. Both are proposed with the power re-run (below) and approved under UD-48 before S1.
- Survivorship-free fast metrics.
- One-repeat sweeps (S1–S4) are coverage maps plus high-count mechanism metrics. Success-rate claims are made only at S5.
- Power, from the integrated plan's paired-McNemar simulation with *homogeneous* effects:

  | Scenes × repeats | 20 pp | 30 pp |
  |---|---|---|
  | 34 × 3 | ≈ 0.80–0.94 | ≥ 0.99 |
  | 34 × 1 | — | only 0.67–0.82 |

  S-S5 shows that model is anti-conservative for clustered repeats, so treat these as optimistic. The scene-clustered re-run reports its own power and type-I error. Confirm the 3 S5 repeats after it.

**Cost (est., ±30 %):**

| Sweep | Calls | Serial wall |
|---|---|---|
| S1 | 1,800–2,950 | 12–20 h |
| S2 | 580–1,020 | 4–7 h |
| S3 | 1,620–2,900 | 10–19 h |
| S4 | 580–1,020 | 4–7 h |
| S5 (two arms) | 5,400–8,800 | 35–60 h |
| **Total with `full34`** | 10,000–16,700 | 64–113 h |
| **Total with `full44` at S1/S3/S5** | ≈ 12,600–21,100 | ≈ 81–143 h |

- **Optional extra S5 arm.** Keeping the pristine arm at 3 repeats in S5, or running both async and blocking arms (S-M7), adds ≈ 2,700–4,400 calls and ≈ 17–30 h each. Recommendation: drop the pristine arm from S5, or keep it at 1 repeat (V-S14).
- **Not in the table.** Each of these exceeds the smoke limit and needs your go-ahead (UD-44):
  - the PC7 A/Bs (≤ 40–60 calls each);
  - the H3 channel ablation (≤ 60);
  - the cache probe (≤ 14);
  - the pointing calibration (≤ 20);
  - the overlay A/B (≤ 40);
  - the UD-39 multi-family A/B (≥ 30 verify calls per arm).
- **Paired-concurrent execution** would cut wall time ≈ 40–50 %, if the subscription allows two streams.

---

## 5. Effort, staffing and critical path

| Engineer-days | M0 | M1 | M2 | M3 | M4 | M5 | Total |
|---|---|---|---|---|---|---|---|
| Integrated plan | 18.5 | 19.5 | 36.5 | 67.5 | 18.0 | 13.0 | 173 |
| Feasibility review re-baseline | 24–27 | 21–23 | 38–41 | 75–80 | 19–21 | 16–18 | 193–210 |
| **This plan, with review fixes (est.)** | **28–31** | **25–27** | **41–44** | **81–86** | **24–27** | **17–19** | **≈ 215–235** |

The deltas over the feasibility row (≈ +4 / +4 / +3 / +6 / +5–6 / +1 d) are this plan's estimate of the review fixes.
- **M0:** INT-5 moved in (V-S9), the no-op seams (V-S6, booked in M2 by the feasibility review), deterministic frame-ring tests, the differential replay, backup and fingerprints.
- **M1:** the pristine adapter, symmetric scoring, evaluator-side `intended`, statistics.
- **M2:** the staged contact window, escape semantics, the keypoint site choice. INT-1 stays in M2.
- **M3:** degraded mode, the robot-term inflation, the `inside_robot` rule, unknown volumes, GPU accounting, the event-frame store, PC3b.
- **M4:** container representation, jaw-sweep planning, deterministic budgets.
- **M5:** the multi-family A/B.

**Five parallel tracks** (engineers or coding agents):

| Track | Scope | Load |
|---|---|---|
| E1 | runner + integration | ≈ 45 d |
| E2 | protocol + telemetry + sweep harness | ≈ 35 d |
| E3 | executor, then the swept planner `on` | ≈ 40 d |
| E4 | perception, plus the truth exporter early | ≈ 35 d |
| E5 | geometry tools + swept shadow | ≈ 40 d |
| (unassigned) | review fixes and milestone operations; assign them before M2 | ≈ 20–40 d |

- The five loads sum to ≈ 195 d, about the feasibility re-baseline. The remaining ≈ 20–40 d is mostly the safety-review fixes:
  - on E3: S-M5, S-M6, S-S7, S-S8, parts of S-M2/S-M3/S-M4;
  - on E5: S-M1, S-S9, S-S10, parts of S-M4/S-S15/S-S16.
- E3's M4 work is on the critical path.

**Critical path: the perception track.**
- The chain: config/digest → frame ring + service protocol → PS-1…PS-11 → geometry integration → S3 → planner `on` → S4 → committed/async → S5.
- Length: ≈ 70–80 working days with five tracks (est.).
- With fewer people, total effort bounds the calendar: ≈ 22–24 weeks for two engineers, ≈ 10–11 months for one.
- Coding agents can shorten the build. Sweeps, quota, GPU availability and your decisions then bound the calendar: S1–S5 alone are 64–143 h of serial runs, depending on the scene set.

**What can start on day 1** (offline): the truth exporter and benchmark dataset, vendoring and the SP1 robot body model, the EXC-0 plant fit from logs, the GT1/GT8 libraries, the PC7 dry-run harness, and R1/R9.

---

## 6. Decisions needed from you

UD-01…UD-42 come from `plan/INTEGRATED_PLAN.md` §10. The reviews raised new decisions, which are not in that table:

| New decision | Raised by | Needed by |
|---|---|---|
| UD-00 | S-S2 | M0 |
| UD-43 | S-S1/V-S2 | M0 |
| UD-44 | S-S13 | M0 |
| UD-45 | S-M4 | M4 |
| UD-46 | S-M3 | M3 |
| UD-47 | S-S4 | M1 |
| UD-48 | S-S5 | M1 |
| UD-49 | S-S6 | M1 |
| UD-50 | V-M3 | M3 |

### A. Before any work starts

| ID | Decision | Recommendation |
|---|---|---|
| UD-00 | **Off-box backup.** `/workspace` is not a persistent volume on this instance. | You create two private targets and provide push credentials before Day 0:<br>• a git remote for the new repo (never H1's public origin), with tags pushed at each freeze;<br>• an artefact target for runs, sweeps and benchmarks, and for the inputs this plan is built on, which also exist only here: the fixture runs under `/workspace/gpt_field/runs` (≈ 12 GB for the folder) and this analysis folder.<br>Request logs hold full model payloads, so the artefact target must be private. Exclude `.private/` and tokens. |
| UD-01 | Location | Local clone at `/workspace/gpt_field/agent_speedup_v2_uc72_harness` + worktrees. H1's tree and `.git` stay untouched. Staging it as a gitlink in the main tree is your choice. |
| UD-02 | Standing approval for no-model GPU jobs | Approve both kinds:<br>• **simulator jobs on a free GPU** (reset preflight, pixel fixture, replays, truth export, scripted executor benchmark, post-hoc video), about 1–3 GPU-hours per milestone;<br>• **perception development on one fixed GPU** (service skeleton, benchmark replays on saved frames; ≈ 8 GB). Reserve GPU 7 from M3 (UD-22).<br>Without this, the perception track waits for a UD-44 go-ahead. |
| UD-03 | Official reset hashes that differ only in PNG bytes | Adopt the pixel hash after you glance at the preflight images. |
| UD-43 | Credential store | One absolute store, `/workspace/gpt_field/.private/chatgpt_plan`, passed to every arm and tool and never copied. Refresh tokens rotate, so copies break other arms. Its concurrency test runs against a fake token endpoint and a scratch store. |
| UD-44 | **Subscription load and who runs what** | Claude may run capped smokes of **≤ 12 calls**, only when none of your sweeps or other agents (astranomy, Codex) are running, within a daily budget you set (e.g. ≤ 150 calls/day).<br>Everything else needs your go-ahead: runs to `finish`, runs of ≥ 13 calls (a full T04), the model-only A/Bs above that size, starting the GPU perception service for live runs, and all sweeps. |

### B. Evaluation protocol, needed before S1

These define what "success", "collision-free" and "fast" mean.

| ID | Decision | Recommendation |
|---|---|---|
| UD-04 | Evaluation caps | 1,800 s policy time + 80 calls (no historical success is censored). Report P(≤ 180/300/600 s). |
| UD-05 | Scoring instant | After a 10-step hold settle, also reporting the finish instant. |
| UD-06 | **Contact taxonomy** | Report collision-free success both with and without palm-on-current-target counted as forbidden. That class is 55 % of H1's forbidden samples, and the analysis brief treated robot-body contact as forbidden, so **you choose which definition is the primary endpoint** at level 2 of the S5 rule; the other is secondary.<br>Also:<br>• articulated pushes are forbidden unless declared (checked evaluator-side);<br>• held-object support outside the goal region is forbidden;<br>• resting contact with other goal objects is reported. |
| UD-09 / UD-48 | Repeats, decision rule, statistics | 1 repeat for S1–S4, 3 for S5. Lexicographic rule: zero-shot → collision-free strict success → time → calls, with a scene-clustered test.<br>The margins and the rule for unresolved levels are **not written yet**. E2 proposes them with the heterogeneous-effect power re-run (offline, weeks 1–2), and you approve them before S1. Confirm the 3 S5 repeats after that re-run. |
| UD-10 / UD-11 | Scene set and sweep scope (budget) | `full44`: full sweeps at S1/S3/S5 and `regression12` at S2/S4, ≈ 12,600–21,100 calls and ≈ 81–143 h. With `full34`, S5 has no held-out obstacle scene. **Decide in week 1**, because R3 builds the seeded-obstacle generator in M0. |
| UD-12 | Quota mid-episode | Pause and resume the same request; the pause is excluded from policy time. |
| UD-14 | Baseline arms | Pristine H1 + measured H1 in S1. Drop pristine from S5, or keep it at 1 repeat. |
| UD-15 | Plan call | Keep the separate plan call (human review possible; perception gets the inventory first). A/B the folded variant after acceptance. |
| UD-47 | Held-out acceptance | Pre-register the mapping: for official scenes, repeat r runs init state r (r = 1–3), with hashes adopted through the M0 preflight and UD-03. For single-init scenes, a repeat is a new model sample at the same reset. Fresh obstacle seeds exist only in `full44`. |
| UD-49 | Counted calls | plan + decide + verify + re-asks (not canary or infra retries), for both caps and metrics. |

### C. Technical defaults: approve as a block unless you object

**Needed by:**

| Milestone | Decisions |
|---|---|
| Before S1 | UD-07, UD-08, UD-13 |
| Before M2 | UD-16…UD-21 |
| Before M3 | UD-22…UD-33, UD-46, UD-50 |
| Before M4 | UD-34…UD-38, UD-45 |
| Before M5 | UD-39…UD-42 |

**Eight of them change behaviour you may care about:**

| ID | Decision | Recommendation |
|---|---|---|
| UD-16 | May the executor insert short motions you did not request (escape legs, back-off, one local replan)? Each is guarded, distance-increasing and reported. | Allow. |
| UD-45 (new) | Container/fixture representation | Walls/rim from the height field; the scoped `allow_contact: inside` is the fallback. |
| UD-25 | Perception boxes vs model boxes at M3 | Perception boxes *replace* model boxes, with the derived robot inflation (S-M2). |
| UD-34 | Palm depth bound | Report-only in M3. Active in M4 only where measured cell heights have σ < pad height. Revert if grasp retries rise by > 1 per 5 grasps.<br>Reverting also switches off M5's approach speed (UD-41; −10…−67 s per task), because INV-1 requires an active depth bound (S-S12). |
| UD-39 | Asynchronous verification | Ships only if the user-started multi-family A/B passes: no rise in late-detected failures, ≤ 1 stale rejection per 10 verify calls (S-M7). |
| UD-27 | Raw absolute world targets | Accepted with telemetry in M3. Absolute `xyz` for `move_to_pose`/`approach` is removed at the M5 freeze if references cover ≥ 95 % in S3 (S-S9). |
| UD-50 | Per-request perception wait | Bounded wait in M3/M4. **Not low-stakes:** ≈ +1.0–1.9 s per decision (est.). Set the M5 value from the S3/S4 stale-box evidence before the M5 freeze. |
| UD-46 | Degraded motion mode on perception outage | On. |

**The rest are low-stakes defaults:**

| ID | Topic | Default |
|---|---|---|
| UD-07 | video | post-hoc |
| UD-08 | A/B execution | sequential blocks |
| UD-13 | canary | on |
| UD-17 | interim margin | plant-derived margin only |
| UD-18 | step budget | remove `MAX_EXECUTION_STEPS` |
| UD-19 | jaw-symmetric orientation | when unreachable, reported |
| UD-20 | image retention | current pair only, if the PC7 A/B (incl. stove-off/drawer, per family) and the M2 smokes show no regression |
| UD-21 | `why` field | drop, unless PC7 shows reasoning tokens rising |
| UD-22 | service GPU | fixed per sweep |
| UD-23 | "look" motion | only on step-0 support failure (σ > pad height) |
| UD-24 | discovery | DA3 components + exemplar |
| UD-26 | overlays | ids + boxes + last path, confirmed by the PC7 overlay A/B and the H3 channel ablation across families |
| UD-28 | k | from benchmark coverage |
| UD-29 | obstacle edits | dispute, not remove |
| UD-30 | inventory | ≤ 12 objects |
| UD-31 | plane sweep | gated by benchmark |
| UD-32 | event frames | non-blocking |
| UD-33 | sphere resolution | 4/10 mm |
| UD-35 | cruise | enable in M4 |
| UD-36 | direct motion | inherit intent |
| UD-37 | posture recovery | report-only first |
| UD-38 | body planner | perception map only |
| UD-40 | cache key | if the M2 probe shows it honoured |
| UD-41 | approach gate | measured only; requires an active depth bound (INV-1) |
| UD-42 | hold steps during waits | pause + idle accounting |

---

## 7. First weeks (after the A-decisions)

### Day 0
1. Create the backup remote and the private artefact sync (UD-00).
2. Clone H1 → `agent_speedup_v2_uc72_harness`. Rename the origin, tag `h1-baseline-3cc84ac`, and create `uc72/dev` and the pristine worktree `arms/h1-baseline-3cc84ac`.
3. Symlink the venvs read-only. Record fingerprints of all three interpreters, including the main-tree `.venv-foundation-vision` and the editable `sam3`/`depth_anything_3` sources, which other sessions share (S-S3).
   - If those change before M3, snapshot a private service venv. Under `--require-frozen`, a change mid-sweep would abort every perception-arm episode.
4. Point every tool at the one credential store (UD-43).

### Days 1–2: the equivalence spike (E1)
- **Prerequisites:**
  - INT-7's EGL-minor resolver;
  - the T04 pixel fixture built from the pristine code (R9 step 1, ≈ 100 s, a UD-02 simulator job).
- **The spike:** frame observation + one simulator + the T04 zero-call replay through a new-runner skeleton.
- **Exit:**
  - exact on 549/549 actions, state8 and the 13 request images → continue;
  - otherwise log the first divergence, tell you, and re-estimate R2/R7 before E1 continues. The other tracks are unaffected.

### Days 1–10, in parallel
| Track | Work |
|---|---|
| E1 | The spike. R1 config/digest/flags/invariants. The V-S6 no-op seams in `agent.py`, `fragments.py` and `agent_client.py`, landed before any other track edits those files. R7 frame ring with the corrected seqlock and deterministic paused-writer/paused-reader tests. |
| E2 | R3 first: task pack, importer, the no-model reset preflight on new and pristine code (V-S1). Then PC2 transport with its fault-injection tests, landing through E1's seams. Then the power re-run and the proposed margins (UD-48). |
| E3 | EXC-0 plant fit from the 9 saved runs. The EXC-B3 progress monitor on replayed traces. |
| E4 | Truth exporter, a UD-02 simulator job that builds each env through the worker's `reset()` (S-S18). The PS-12 benchmark dataset, static truth first. The PS-1 service skeleton against a stub backend on CPU, moving to GPU 7 under UD-02. |
| E5 | INT-1 vendoring (robosuite asset layout). SP1 body-model build in `.venv-sim`. GT1 metric core and GT8 envelope. |

### End of week 2 (est.)
- The spike verdict, with the first divergence if it was not exact.
- R1 config, flags, invariants and digest, plus the V-S6 seams.
- FrameRing v1, with its deterministic tests passing.
- The T04 pixel fixture.
- The reset preflight table (new and pristine code), ready for your UD-03 glance.
- Off the M0 path:
  - PC2's fault-injection tests;
  - the EXC-0 plant fit;
  - INT-1 vendoring;
  - the static-truth benchmark dataset;
  - the power re-run.

### Weeks 3–4 (est.)
- R2 and R9 bring the full `h1_equivalent` replay to bit-exact.
- Then: the differential replay, the overhead benchmark and the capped ≤ 3-call smoke on `h1_equivalent` (within your UD-44 budget).
- M0 is 28–31 engineer-days, mostly on E1 and E2; R1 + R2 + R7 + R9 alone are ≈ 16 d. **M0 cannot close in two weeks.**

---

## 8. Deferred (with reasons)

| Item | Why deferred |
|---|---|
| Folding the plan call into the first executive call | It removes human review, and the first decision would have no inventory-seeded perception. A/B it after acceptance. Note: the 3-minute target likely needs it. |
| A native tool loop for `locate` | The transport commits one output text per decision, so a round trip costs as much as an `observe`. |
| A joint-space posture controller | It changes the control interface; only if the Cartesian candidates fail. |
| A higher rotation cap | Speed only; it needs a plant refit. |
| A persistent fused occupancy map with ray carving | The height field + discovery + conservative boxes cover the measured failure classes. |
| Harness-generated grasp/place candidates as numbered marks | References + depth bound + envelope come first; revisit if regrasps dominate S5. |
| Real-robot items (maximum frame age, checkerboard calibration) | Out of scope for this build. |
| Paired-concurrent sweeps | They need a two-stream subscription smoke first. |
| A K2 (`foundation-harness-keypose-2`) launch | Outside this build. If it runs, treat it as a diagnostic, per analysis §4.3. |

---

## 9. Review corrections applied (these supersede the cited sections of `plan/INTEGRATED_PLAN.md`)

**Sources.**
- `plan/reviews/review_feasibility.md`: V-Mn / V-Sn are its MF-n / SF-n. V-mig collects its §6 migration items 8 and 10.
- `plan/reviews/review_rules_safety.md`: S-Mn / S-Sn are its MF-n / SF-n.
- C- marks corrections from the final consistency and actionability check of this document. M = must fix, S = should fix.

| ID | Correction | Supersedes |
|---|---|---|
| V-M1 | FrameRing seqlock. **Writer:** `seq_begin` → payload → `seq_end` → header. **Reader:** `seq_end` first → copy → `seq_begin` last; accept only if both are equal and match the expected value. Deterministic paused-writer/paused-reader tests; the 10⁴-frame run is a smoke test only. | I-03; M0 check 3 |
| V-M2 | The sweep runner records the cap reason before touching H1's stop file. The adapter maps H1 `user_stop` + cap reason to scored, censored `cap_*` and computes pristine policy time post hoc. | §2.7 item 1, §7.2, §7.3 |
| V-M3 | Bounded wait for the plan-bound snapshot before request 1, charged to `Timers.block("perception")`. Per-request wait policy (UD-50) frozen in config; frames logged per call; a stale-overlay test. | §3.2, M3 |
| V-S1/S2, S-S1 | Pristine arm via H1's evaluator with launch.py's environment, the EGL minor and preflight hashes. One absolute `--store-dir` everywhere. A concurrent-refresh test, run against a fake token endpoint and a scratch store, never the real one. | §2.7, §7.2, all commands |
| V-S3 | `action.py` hard failure only after `replay_runner.py` passes. The env var stays the carrier, set from config and verified in both processes. Port or retire H1's tests. | R1 |
| V-S4 | GPU headroom via the service allocator; fixed placement; leak guard. | §3.4, §7.2 |
| V-S5 | Differential zero-call replay with scripted answers, plus per-milestone golden fixtures. | R9, §2.7 |
| V-S6 | M0 no-op seams in `agent.py` / `fragments.py` / `agent_client.py`, landed first, with one owner per shared H1 file. | §2.8 |
| V-S7 | Worker `stop_on_success = not runner.terminal_actions`, tested past the first raw success. | R2/R3 |
| V-S8 | Async client: `CallResult` metadata, cancellable in-flight call, SIGTERM test, loopback HTTPS/SSE deadline test. PC6 = 7–8 d. | PC6, M5 |
| V-S9 | Truth exporter (INT-5) moved to M0, weeks 1–3, on the perception track. | §8, M3 |
| V-S10 | Effort re-baselined (§5); 2-day equivalence spike first, with prerequisites and an exit criterion (§7). | §12 |
| V-S11 | Outputs outside the repo; the dirty check ignores untracked files. | §2.1 |
| V-S12 | Vendor tree mirrors robosuite's asset layout; npz without pickles (INT-1 stays in M2, may start on day 1). | I-10, INT-1 |
| V-S13, S-S3 | No new controller dependencies (hand-written validators). Read-only fingerprints for all three interpreters and the editable SAM/DA3 sources, hashed into the digest. | §2.1, §2.3 |
| V-S14 | S5 cost counts the pristine arm, or that arm is dropped or kept at 1 repeat. | §7.5 |
| V-S15 | Per-episode event-frame store filled by the controller. | I-03, PS-2 |
| V-S16 | Weight hashes taken once from the service `hello`, then cached. | §2.3 |
| V-S17 | Router enum derived from `swept.mode`. Plan call timed as model idle time inside policy time. Decision/event marks set controller-side. | §2.4, §3.5 |
| V-mig | `evaluator.collision_base_mode` in the task spec (g3, bowl_plate); `python -I` scripts insert `src`. | R3 |
| S-M1 | No `inside_robot` exclusion at the current joints. 3-D self-test at the observed joints only, with a containment bound from PS-3's measured recall, never on objects previously observed static and clear. Otherwise `near_robot` + σ inflation. Offline test on the T08 bottle and the T04 mugs. S3 metric: real objects excluded = 0. | I-08, X16 |
| S-M2 | **From M2** (when GT8 ships) until `swept.mode=on`: the envelope text says only the grip-site point is checked and the model owns hand/finger/held clearance. **From M3** (model boxes removed): box inflation by the derived hand/finger half-extent at stage orientation and gap, plus the held record's below-grip extent and footprint (or the jaw-column fallback). S3 metrics: carry margin vs truth; hand/finger/held episodes in routed stages vs b002. | X20, I-08, GT8 |
| S-M3 | Degraded motion mode (INV-DM, keyed to `perception.mode=on`): bounded reset wait, contact-safe speed when stale, outage cells flagged. S3 reports results with and without outage cells. | §3.6, SP9 |
| S-M4 | Containers and front-opening fixtures as walls/rim/opening, or a scoped `allow_contact: inside` (UD-45). Container stages on point planner + guard until then. Validated by an SP12 drawer replay plus a t9 truth fixture. Boundary handoffs reported in S4. | I-08, SP5, SP6, M4 |
| S-M5 | Jaw sweep as a planner step (clear-width open/close or guarded retreat). Direct motion inherits intent (UD-36 revised). `jaw_sweep_*`, truncations and calls caused are reported in S4. | SP6, UD-36, X17 |
| S-M6 | Contact window derived by the executor and staged: plant-derived detection distance + tolerance in M2; + k·σ along the motion for referenced targets from M3. The model may only shrink it. From M3, `unexpected_contact_other_object`. Offline: the 3 early onsets classify as `unexpected_contact_early`. | I-14, EXC-B3 |
| S-M7 | UD-39 decided by a user-started, interleaved, capped A/B: ≥ 3 families, ≥ 30 verify calls per arm, with CIs. Pass = no rise in late-detected failures and ≤ 1 stale rejection per 10 verify calls. Alternative: both arms in S5, ≈ +2,700–4,400 calls. | UD-39, M5 |
| S-S2 | Off-box backup decision (UD-00), including the fixture runs and this analysis folder. | §10 |
| S-S4 | Historical scenes flagged; held-out S5 conditions (UD-47). σ/k fitted on historical scenes, validated on S1 frames, never refit on S5 conditions. | §7.1, M3, M5 |
| S-S5 | Scene-clustered test; margins and an unresolved-level rule (UD-48); heterogeneous-effect power re-run with type-I error; survivorship-free fast metrics. | §7.4, M5 |
| S-S6 | `cap_wall` = infra hang; pristine caps scored; counted calls defined (UD-49); `sim_instability`; async accounting during failed attempts and quota pauses. | §7.3, I-18 |
| S-S7 | Escape: the last anchor can be left without increasing penetration; −approach first; no horizontal escape while holding on a support; `route_goal_blocked` becomes a standoff handoff. | EXC-B2 |
| S-S8 | Runtime finger/pad keypoint sweep picks the rotation site. | EXC-B1 |
| S-S9 | Absolute targets planned for removal by the M5 freeze; `offset_beyond_anchor` flag. | UD-27, I-09 |
| S-S10 | "Dispute" as the default; "remove" only for single-view detections. | UD-29 |
| S-S11 | Palm-on-target reported both ways, with a primary endpoint chosen (UD-06). `intended` computed evaluator-side. | UD-06, R5 |
| S-S12 | β as a user decision with evidence; k from benchmark coverage; look trigger σ_support > pad height; deterministic planner budgets. INV-1 couples the M5 approach speed to an active depth bound (UD-34 ↔ UD-41). | I-01, SP6, UD-23/28/33/34/41 |
| S-S13 | M0 capped smoke. Checks recorded, never blocking. Runs of ≥ 13 calls or to `finish`, the model-only A/Bs above that size, and GPU-service starts need a go-ahead (UD-44). | M0–M3 live checks |
| S-S14 | `task_success`/evaluator keys banned in model payloads; pause sentence removed in M1. | PC1 |
| S-S15 | `held_identity_mismatch` barrier and re-attachment in **M3** (needs GT3 + PS-9); counted among S5's late-detected failures. | GT3, PS-9 |
| S-S16 | `unknown_volumes` from height-field shadow cells; SP9 lowers the speed cap there (shadow in M3, active in M4). | I-08, SP9 |
| S-S17 | Prompt/context A/Bs include the stove-off and drawer requests, reported per family; UD-20/21/26 stay conditional on them. | PC7, M2 |
| S-S18 | Replays build the env through the worker's `reset()`; exactness check on t3/t8/t9. | INT-5, R8, M0 |
| C-M | PC3b (scene inventory) belongs to M3, not M1 (S1 keeps H1's plan prompt). The UD-50 default (bounded wait) costs ≈ 1.0–1.9 s per decision, so the S3 wait target and the "≈ 0 s added" effect hold only under option (b). The lean option saves ≈ 6.5 d (not 24) and loses the height-field-dependent items. The first-weeks list no longer promises M0 closure in two weeks. | this document |
| C-S | EXC-0 constants (a = 0.711, ρ = 0.254); GT8 and INT-1 in M2; INT-4 in M3; binding without numbered marks; M3 swept components include SP8 (report-only) and SP11 (shadow); approach speed is SP9; cost totals for `full44`; power caveat; INV-DM naming; decision sources and needed-by milestones; smoke limit ≤ 12 calls. | this document |

---

## 10. Files

| Path | Content |
|---|---|
| `IMPLEMENTATION_PLAN.md` | this document |
| `plan/INTEGRATED_PLAN.md` | the detailed specification: interfaces I-01…I-20 with schemas, flags and invariants, file lists, tests, commands, traceability. §9 here supersedes the sections it cites. |
| `plan/design/ws_*.md` | the six workstream designs, with their offline prototype measurements: OSC plant fit, sphere-cover body model, FK parity, swept-check timings, pointing and support-plane studies |
| `plan/prototypes/` | the designs' offline scripts and small result files |
| `plan/reviews/review_feasibility.md`, `plan/reviews/review_rules_safety.md` | the two adversarial reviews |
| `README.md` + `work/` | the verified analysis this plan implements |
