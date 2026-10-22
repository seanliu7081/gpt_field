# agent_speedup_v2: why the stove + moka pot run failed, and the fix plan

- **Run:** `runs/agent_speedup_v2_fast_libero_10_t2_001`
  - libero_10 task 2, "turn on the stove and put the moka pot on it", episode 0.
  - Build: fast build, v2 sha `a5eb3a16`, thresholds v8, prompt-4.
- **Result:** reached the 4000-step cap. LIBERO success false. 44 model calls (plan 1, replan 4, repair 12, point_fine 18, point_coarse 9).
- **Method:**
  - The run was replayed bitwise from simulator truth (0 action mismatches, 0 state error).
  - Three analysts covered: planner refusals, grasp execution, and observers/replans/policy.
  - Each key claim was checked against simulator truth, with forks from saved states. The headline fork was re-checked for this report.
- **Date:** 2026-10-20.

---

## 1. Summary

**t2 was one false refusal away from success.**
- The agent turned the stove on with its first knob grasp (step 596).
- It then grasped the moka pot by its handle on the first try (step 1469). The hold was physically good.
- During the short test lift, the pot pivoted about 10° in the hand and stayed there. The in-hand rotation check (O-1, limit 5°) called the grip unstable and let the pot go.
- **Proof:** a fork of the same run with only that limit raised (5° → 20°), keeping the agent's own policy and its recorded model answers:
  - held the pot (step 1498);
  - reached LIBERO success at step 1679;
  - placed the pot (step 1866);
  - used **3 model calls** in total;
  - LIBERO success then stayed true for every logged step to 2986 (`runs/agent_speedup_v2_t2fa_grasp_o1_764/cv5_sim.json`).

**Everything after that refusal was a cascade.**
- The refusal removed the handle's only two grasp directions, and the model was told only "no feasible candidate".
- Putting the pot back moved it 3.3 cm and turned it 9°. From then on almost every refusal was a measurement artefact on the disturbed pot. One was a false "burner changed" alarm.
- Nothing in the recovery logic ended the loop, so the run spent the remaining ~2500 steps and 40 calls getting nowhere.

**The fix that gets t2 working is small:**
- a better load test (F1);
- an honest failure message (F2);
- a corrected "keep the burner glowing" check (F3).

That is about 7–8 hours of work before the next live t2 run. Other fixes make retries and the other tasks work.

---

## 2. What happened, step by step

| Steps | Event | Real or artefact? |
|---|---|---|
| 0–596 | Calibrated. The model planned (1 call, valid). Knob held on the first try (478). The burner was seen turning on (596). | Worked |
| 977–1194 | Handle D pointed at and registered. The planner admitted a top-down handle grasp on the first try. | Worked |
| 1469 | Both finger pads closed on the handle (gap 15 mm), and the hold check passed. | Worked (truth: both pads on the handle) |
| 1475 | **O-1 latched `grip_unstable`.** The wrist estimate was 5.95° and 5.64° on 2 frames, against the 5° limit. | **False refusal.** Truth: the tilt went 1.5 → 10.5° and then stayed flat. A raw fork kept lifting 7.5 cm, carried and placed with no slip. |
| 1539 | `no_feasible_candidate`, with nothing listed. The repair switched the grip to the pot body. | Dead end. Both handle directions were excluded, and the model never heard "the pot rotated". |
| 1863–2006 | Pot "body" grasp: the record was actually the **lid knob**. Finger 2 dragged on the rim, giving `early_finger_contact`. | Artefact. The 0.544 mm shortfall barely crossed the 0.5 mm floor; the finger still made 82% of its travel. A fork shows the knob grip holds and places. |
| 2334–2510 | D was re-registered. Align failed with `ill_conditioned_probes`, identically on 3 tries. | Artefact. Hand rotation during the probe (0.0013–0.0047 rad) exceeded the 0.0035 rad clean-frame limit, while the mark was tracked fine. |
| 2510 | Replan 1 added the claim "keep the burner glowing". | The claim is true, but it is checked wrongly (next row). |
| 2670 | **`kept_relation_lost`: "burner changed"** (36% of its pixels, against a 2% limit). | **Artefact.** The check compares against the episode-start image, taken when the stove was off. The share was exactly 0.3613 at all 8 checks: glowing coil against cold coil. Truth: stove on to the end. |
| 2671–3292 | More handle attempts. D's region collapsed (`geometry_unmeasured`, 6 of 7 registrations), then `footprint_blocked` at 3292. | Artefacts. Truth puts every point on the handle. The 3292 blockers are inferred fills on cells that are table in truth, and full truth admits the plan. |
| 3292–3314 | Replan 3: "move the frying pan aside first". The destination was out of reach (0.815 m against 0.77 m), so it failed twice; then the step was skipped. | The model's guess had no evidence behind it: the refusal listed no blockers. |
| 3314–3991 | Back to the handle; `geometry_unmeasured`, then the cap. | Artefact (collapsed region on the disturbed pot) |

**Counts:**
- Grasp attempts on the pot: 3 closes in the whole episode (the knob and two pot closes). Both pot closes held in truth and were refused by safety rules.
- Model calls: 2 for the stove, 42 for the pot. None of the 12 repair answers led to a grasp.

---

## 3. Root causes

### 3.1 The load test confuses "pivoted once, then stable" with "slipping" (the deciding failure)

- **Where:** `agent_speedup_v2_observers.py`, CoMotionObserver (~L2360, L2438–2446), `O3_ROTATION_MAX_DEG = 5`.
- **What the 5° rule predicts:** that the hold will tilt more than 15° if lifted further. But it never looks at the further lift.
- **How the new rule was chosen:** the RS-7 calibration data (465 holds, 7 runs, truth) were re-scored offline.
  - **The simple "it stopped growing" rule is unsafe.** 66 of 245 holds that ended badly tilted were flat over their last 3 lift frames. A 4-frame "settled" rule would admit 174 holds that later reach 20° or more.
  - **What separates the holds is creep over a further 6 cm lift**, the same diagnostic the calibration labels came from. With less than 2° allowed:
    - 109 of the 403 tripping holds pass;
    - none of them reaches 20° in truth;
    - clean holds creep at most 0.27°;
    - t2's hold creeps about 0.4°.

### 3.2 A refused grasp becomes a silent dead end

- **Where:** `agent_speedup_v2_relations.py:3407–3411` and `agent_speedup_v2_planner.py:2449–2450`.
- **What happens:** after `grip_unstable`, the candidate and its 180° twin are excluded. For a single-axis appendage like a handle, those are the only two candidates. The planner returns `no_feasible_candidate` with no rejections.
- **The consequence:** the model repairs blind and switches to the pot body. The body is about 80–92 mm wide, larger than the 78.5 mm open gripper, so that grip is a real dead end.

### 3.3 Once the pot was disturbed, measurement artefacts blocked every retry

| Artefact | Where | Seen at |
|---|---|---|
| Handle region collapses after the object moves (S-13) | `agent_speedup_v2_scene.py` (S-13 ~L2718–2730; MC-R2a ~L3407, L3749) | 2334, 2865, 2949, 3991 |
| Phantom inferred fills from high-σ measured parents block P7; the outcome lists no blockers, so nothing is re-measured | planner `_blocking` / `needs_cell_refinement`; scene fill-parent rule | 3292 |
| Phantom fills also made the *first* grip 5.1 mm shallower (1.0219 vs 1.0168 with fills set to truth). Whether this caused the pivot is unproven | planner `_height_stop` (IF-HS raise) | 1194 |
| Touch guard counts a fingertip dragging on its landing surface as a touch (fixed 0.5 mm floor) | `agent_speedup_v2_primitives.py` GuardedClose (L130, L1801–1806) | 2006 |
| Align probe fit becomes degenerate when the hand rotates slightly during the probe | `wrist_servo.py` `Align._fit_probes` (L488, L1114–1126) | 2510 |
| "Whole pot" registered on the lid knob (top 1.0525 against the body top of 1.033) | scene registration | 1863–2441 |

### 3.4 Observer and policy bugs that wasted the rest of the episode

- **Effect claims are checked against the wrong baseline.** In `hold_true_check` (`agent_speedup_v2.py:1108–1140`), "keep the burner glowing" is checked as *unchanged since the episode start*. At the start the burner was off, so the check is guaranteed to read "violated".
  - It cost 3 calls.
  - It took over the step's own repair (L1241–1251).
  - It blocked reviews for the rest of the episode.
  - The turn step's own effect check, `stays:turned:A`, correctly read "met" the whole time.
- **The escalation ladder never climbed.** Every adopted replan resets all counters (`adopt`, L945–947). The model replanned every 1–3 failures, so `recovery_exhausted` was unreachable. The (place_on pot, burner) step failed 9 times.
- **Replans run blind.** Replan requests carry no triggering outcome. Two of the four returned the same step unchanged. One invented the pan detour without evidence.
- **Part ids can be reused.** The 3292 plan reused id A for the pan while the live check `stays:turned:A` still referred to the knob.
- **A repoint doesn't move the hover target.** `_seed_lift` (`relations.py:713–731`) keeps the first seed, so the repoint of F was a no-op.

### 3.5 Not a cause for t2

The "free spot / beside" relation (eval gap 8) is not needed for t2:
- the successful fork never touches the pan;
- the 3292 blockers were phantom fills, not the pan;
- the model's own next repair said the handle was accessible.

That relation is still required for t6.

---

## 4. Fix plan

**Classes:**
- **P:** prerequisite for t2 success.
- **R:** robustness, so retries and other tasks work.
- **E:** efficiency.

All fixes are general: none has moka, stove or pan logic.

| # | Class | Fix | Owner | Size | Effect |
|---|---|---|---|---|---|
| **F1** | **P** | **Extended load test.** When O-1 trips 5° on a passing, gap-stable hold, don't latch yet:<br>1. Lift a further 6 cm (`O1_EXTEND_M`, DD from RS-7).<br>2. `grip_unstable` only if creep over the extension is ≥ 2° (`O1_CREEP_MAX_DEG`, DD) or the tilt passes 20° (O4).<br>3. Otherwise pass, and write the measured in-hand rotation into the held pose, so the carry aim and the upright check use it. | observers (CoMotionObserver) and the load step in relations | M, 3–4 h | Removes the deciding failure (fork: success at 1679, 3 calls). Likely matters for t8 (same handles). Bowl and mug holds keep the calibration protection. |
| **F2** | **P** | **Honest failure reason.** When exclusions empty the candidate set, list the excluded candidates (`excluded:<cause>`). The grasp then fails with the real reason (`grip_unstable` / `early_finger_contact`) and the measured value. | planner L2449–2450; relations L3407–3411, L3189 | S, 1 h | The model stops switching to an impossible body grip. Helps t8/t6, which had 8/10 and 4/5 no-op repairs. |
| **F3** | **P-guard** | **Correct effect claims.**<br>- Baseline = the image at claim adoption, with the chroma statistic for effects.<br>- Reuse an overlapping live effect check.<br>- Mask the step's object and grip part; pending when < 50% is visible.<br>- Escalate only after 2 violations in a row.<br>- A claim loss is folded into the step's own repair, never replaces it. | `agent_speedup_v2.py` `hold_true_check`, `add_hold_true`, L1241–1251 | S, 2 h | Removes the false "burner changed" alarm, and the risk of a repair turning the knob off to "fix" it. Without the visibility rule, placing the pot on the burner would trip it again. |
| **F5** | R | **No shallow grips from phantom fills.** If the height-stop raise is driven only by inferred-fill blockers, re-measure those cells (the existing refine path) and re-plan before accepting the raise. Do not just put IF-PV first: it hides real obstacles in stored plans. | planner `_height_stop`, `needs_cell_refinement` | S–M, 2–3 h | Deeper first grip on t2 (it may avoid the pivot entirely; see V0). **Large for t6/t8**, where inferred fills blocked P7 in all three tasks. |
| **F6** | R | **Phantom-fill blockers become re-measurable.** A fill whose parent is a single high-σ measured cell, and that parent itself, count as unverified. The refusal exports its blockers, and one refine round runs before a P7 refusal is final. | planner `_blocking`, `needs_cell_refinement`; scene fill-parent rule | S, 2 h | Admits the 3292-type plan. t6 (rim grasp) and t8 (P7 refusals at 467/2691/3468). |
| **F10** | E | **Ladder, replan evidence, seed lift.**<br>- A replan with the same step and grips is `no_change` and keeps the counters.<br>- Cap failed attempts per step across replans.<br>- Replan requests carry the triggering outcome, blockers and history.<br>- Refuse reused part ids that a live check still names.<br>- `_seed_lift` uses the latest seed. | `agent_speedup_v2.py` (`adopt`, `_ladder_trigger`); plan/client replan request; relations `_seed_lift` | S, 3 h | Saves about 9 of 44 calls, ends hopeless episodes early, removes evidence-free detours. |
| **F7** | R | **Guarded close for touch_stop grasps.** Back off about 3 mm after the landing stall, keeping the P3 overlap, before closing. Count a touch only on a real stop: shortfall ≥ max(floor, 0.5 × the finger's free motion). | primitives GuardedClose; close step in relations | M, 3 h | Opens the top-feature grip route (fork: the knob grip holds and places). Real side hits must still abort. |
| **F8** | R | **Align survives small probe rotation.** Correct each probe frame for the measured rotation, or keep the least-rotated frame per leg. On an identical repeated failure, fall back to the mount-model Jacobian. | `wrist_servo.py` `Align._fit_probes` | S–M, 2 h | Removes the 2510-type failure. |
| **F9** | R (gates F4) | **Handle region collapse after a disturbance.** Diagnose first: replay D's registrations against the truth handle cells. Then fix the attribution of region members, or re-survey the whole record after a known disturbance. Do *not* loosen the collapse ratio. | scene S-13 / MC-R2a | L, 1–2 days | Needed for any retry after a failed lift; also t8 (A collapsed 6×) and t6 (rim). |
| **F4** | R | **Re-grip the same handle better** after `grip_unstable`: re-register, then re-plan the same jaw line with the grip point moved toward the object's centre and the deepest admissible stop. Exclusions are keyed on (family, yaw, axial offset, depth). | relations grip_unstable branch; planner candidate generation | M, 4–5 h | A second-attempt route, but useless until F9 makes the disturbed handle register. |
| F11 | t6 track | **Free spot / "beside" destination**, flagged in the plan schema: it registers the base-band patch, and the runner picks a free, reachable cell near the hint. | scene, relations `place_on`, planner, plan schema | M–L, 1 day | **Not needed for t2**; required for t6. |

**Latent risk, no fix yet.** The pot body was measured 70.7–75 mm wide, against a true 80–92 mm and a 78.5 mm gripper opening. P2 would rate it marginal, not infeasible. This belongs in F9's diagnosis.

---

## 5. Validation (cheap first, then one live run)

| Fix | Check | Pass criterion |
|---|---|---|
| **V0** (decides F4/F5 first, 30 min) | 4 raw sim forks at the 1454 state: grip height {1.0219, 1.0168} × grip point {recorded, 1 cm toward the pot centre}. Close, lift 2.6 + 6 cm. | If a deeper or inner grip stays under 5°, then F5 also cures t2 and becomes a safety net for F1. |
| F1 | Unit tests: trip-and-flat passes; trip-and-creep +3° fails; past 20° fails; a clean series doesn't extend. RS-7 re-score in sim with the new observer (episodes in parallel, ~6.5 min each). t2 policy fork from 764. | Every hold reaching ≥ 20° is flagged, and 0 clean holds are flagged. Fork: hold ~1498, placed, LIBERO success ~1679, 0 new calls. |
| F2 | Unit test, plus a zero-call policy replay to 1540 | Outcome n=11 reads `grip_unstable` with the measured rotation. |
| F3 | Unit tests (adoption on and now on: met; now off: violated; 60% covered: pending; joint repair), plus a replay with `t2fa/obs/hook_holdtrue.py` | Met at all 8 checks. |
| F5/F6 | Offline `cf_plan` on planner calls 2 (1194) and 17 (3292); re-score the t6/t8 eval3post P7 refusals | 1194 plans at 1.0168; 3292 admits at about 1.0114 with blockers listed; t6/t8 refusals become admissions or refine requests. |
| F10 | `ladder_sim2` on this run; unit test; replay to 3294 | Forced replan around 2949, forced skip, `recovery_exhausted` before the cap; the hover target moves after a repoint. |
| F7 | Fork at 1539; rerun the candidate_v5 t8keepc/t2keepc branches | The knob close holds; real side hits still abort. |
| F8 | Replay hook on 2476–2509 | The fit succeeds on the recorded probe frames. |
| F9 | Replay hook on D's registrations at 2334/2865/2949 against truth; `sal_collapse.py` on all 44 S-13 tests | No new false negatives on the 4 real collapses. |

**Then one live t2 run** (`runs/agent_speedup_v2_t2fa_live_001`) with F1+F2+F3, plus whichever of F5/F6/F10 are ready, followed by the truth check.
- **Target:** LIBERO success, with the nominal path under 10 calls.
- **Caveat:** live model answers can differ from the recorded ones the fork replayed. A different first grip point on the handle is the main source of variance.

---

## 6. Order and time

| Phase | Work | Time |
|---|---|---|
| 0 | V0 grip fork | 0.5 h |
| 1 (must) | F1 + RS-7 re-score + t2 fork; F2; F3 | ~7 h |
| **Live t2** | F1+F2+F3 (+F5 if V0 says so, +2–3 h) | run ~15–30 min, plus the truth check |
| 2 (before t8/t6) | F6 → F5, F10 | ~7 h |
| 3 (robust retries) | F9 diagnosis → F4; F7; F8 | 1–2 days |
| t6 track | F11 free spot / beside | ~1 day |

**Recommendation:**
1. Ship F1+F2+F3 and run live t2. This is the only path with direct evidence of success.
2. Add F5 first if V0 shows a deeper grip avoids the pivot.
3. F5/F6 and F10 next, because they carry over to t8 and t6.
4. F9+F4 make a second grasp attempt possible after any disturbance. Today no retry path works on t2.

---

## 7. Evidence

All scratch paths below are under `/tmp/claude-0/-workspace/2e071f26-a38c-4a1c-b40e-bd363cbcfe54/`.

**Runs:**
- Run: `/workspace/gpt_field/runs/agent_speedup_v2_fast_libero_10_t2_001`. The final frame is `scratchpad/t2fast_end.png`, and the outcome timeline is `scratchpad/t2fast_outcomes.txt`.
- Deciding fork (O3 5° → 20°, switched at step 764): `/workspace/gpt_field/runs/agent_speedup_v2_t2fa_grasp_o1_764/cv5_sim.json`. Outcomes: hold_pass 1498, placed 1866, 3 calls, LIBERO success true on the final 1308 logged steps.
- Touch-rule fork: `/workspace/gpt_field/runs/agent_speedup_v2_t2fa_grasp_gc_1539/cv5_sim.json`.

**Analyses** (in `impl/v2/t2fa/`):
- `analysis_planner_refusals.md`
- `analysis_grasp_execution.md`
- `analysis_observers_policy.md`
- the full plan with re-checks, `fix_plan.md` (also `fix_plan_result.md`)

**Truth capture and scripts:**
- Truth capture: `impl/v2/t2fa/truth/` (per-step poses, 22 planner calls with truth heights) and `impl/v2/t2fa/grasp/truth_capture/`.
- Load-test re-score on RS-7: `scratchpad/rs7_settle.py`.
- Planner counterfactuals: `impl/v2/t2fa/planner/scripts/cf_plan.py`. Run it with `/venv/fm_dno/bin/python` and `PYTHONPATH=impl/v2/t2fa/truth/capture/pkg/src`.
- Observer replays: `impl/v2/t2fa/obs/` (`hook_holdtrue.py`, `replay_log_holdtrue.json`, `replay_log_F.json`).
