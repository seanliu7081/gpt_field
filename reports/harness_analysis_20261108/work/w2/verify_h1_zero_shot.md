# W2 adversarial verification of `w1/h1_zero_shot.md` (key `verify_h1_zero_shot`)

2026-11-08. Read-only. No simulator, model, Astra or GPU work; nothing under `/workspace/gpt_field` was edited.
Every number below was recomputed from raw run artifacts with my own scripts (not the W1 scripts):

| Script (in `w2/scripts/`) | What it recomputes | Output (in `w2/data/`) |
|---|---|---|
| `hashcmp.py` | SHA-256 of H1 policy/runner modules vs every `source_at_launch`, `trial/frozen_source`, `review/frozen_source`, V1 planning drafts, `results/` trials and the main tree | `hashcmp.json` |
| `calls.py` | per-call table from `trial/requests/*.request.json`/`.response.json` + `episode.events.jsonl` (step, latency, type, belief, trigger incl. compacted `last_feedback.summary`, joints, tokens) | `calls.json` |
| `stall_continuation.py` | for each stall handoff: stalled skill/profile, discarded pending segments, whether the next call re-issued them | `stall_continuation.json` |
| `joint_margins.py` | joint-limit margins per request (limits read from robosuite `panda/robot.xml`) | `joint_margins.json` |
| `req_sizes.py` | request size, compaction flag, macro share per request | `req_sizes.json` |
| `t9_check.py` | H1's own `validate_review_plan`, run under H1's `.venv` with `-I`, on 13 V1 drafts + 7 approved plans | `t9_check.json` |
| `grasps.py` | open→close transitions of the executed gripper command | stdout |

Paths: code is relative to `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/`; runs to `/workspace/gpt_field/runs/`.

## Headline

- The core of W1 holds. **F1, F2, F4, F7, F10 and F11 are confirmed.** For F1, F2 and F10 the uncounted evidence makes the case stronger than W1 wrote.
- **Numbers that are wrong or over-attributed:**
  - F5's projected call savings are not supported by the logs.
  - F6's veto cost is counted on the wrong call: it is 93.7 s, not 213.4 s.
  - F3 attributes all 46 stall handoffs to metric grounding, which is too many.
  - F9's proposed "on" semantics is looser than LIBERO's real predicate.
- **One claim is refuted.** F12 says compaction hid the route sub-reason in T08 call 15. In fact the sub-reason is in that request, and the model named it.
- **F11 holds after an exhaustive hash search.** No run anywhere on disk with H1's policy modules is on an official task. The only official-scene uc7.1-family run is `default_routing_t04` (layout `official`), and it is an older prompt-5/fragments-3/20 mm build.
- **On the F1 implication.** The reported uc7.1 successes are oracle-terminated: the raw-success stop truncates the run, and the "episode continues" cue tells the agent it is not done. They are not honest finish-time measurements.
  - Every known uc7.1-module success (12/12) ended with an object still held.
  - In two of them the raw success was not a valid completion at all:
    - **t6 relayout:** the pudding was still being carried about 5 cm above the table.
    - **t0 relayout:** after a scripted release the sauce would perch on the basket rim.
  - The honest success rate of the 7 retained runs is unknown, anywhere from 0/7 to 6/7. W1 implies 6/7, which overstates it.

## Per-finding verdicts

### F1 — two-sided success oracle — CONFIRMED WITH CORRECTIONS (high)

**Checked:**
- **Code:**
  - `scripts/evaluate_agent_speedup_v2.py:213-216` requires `stop_on_success` and forbids `max_steps`.
  - The loop exits at `:1437-1472`: success, operator file, operator step cap, `recovery_stop`, smoke, or an exception. The agent-first user-check modules never emit `recovery_exhausted` (grep), so in H1 the loop ends only on success, an operator stop or an error.
  - `finish` is a 1-step zero motion plus a logged claim (`src/gpt_field/agent_speedup_v2_user_check_agent.py:366-372`).
  - Prompt text is at `src/gpt_field/agent_speedup_v2_user_check_agent_client.py:205-209`.
- **Raw artifacts (`result_summary.json` → `physical_truth` / `physical_audit`):** all 7 retained successes have `terminal_failing_checks` equal to `not_held`/`at_rest` only, `terminal_failure_is_hold_only=true`, and `diagnostic_settle.actions=38`.
- **Final stages:**
  - T04-100: `approach` contact, no pending stages (the final call 13 = transit + descent).
  - T08 and stove-off T08: `set_gripper open` 3 actions in, with a +0.17 m retreat pending.
  - Composite 001: approach −0.018 m with open + 0.07 m withdraw pending.
- **T04-20 mm (`can_obstacle_t04`):** `finish` at call 17, step 794. `sim_status.success=false` at step 794. Calls 18–26 = 9 calls, 159.06 s API latency, 1276−794 = 482 actions. The call-18 belief starts "The episode continues, so at least one final relation may be insufficient…". Re-grasps happen at calls 21 and 26.

**Corrections:**
1. **Scope.** "7/7 uc7.1 successes" means the 7 *retained* runs. The pattern also holds for the uc7.1 runs W1 did not count:
   - relayout t0/t1 (2026-10-30; `presentations/speedup_v2_user_check_v71_libero10_en/evidence.json`, `tasks.t0/t1.result`);
   - relayout t6 (`reports/uc7_direction_semantics_20261028/implementation_uc71/README.md:80`);
   - the two H3 runs, which use byte-identical user-check modules (`runs/foundation_vision_t0{4,8}_*/trial/physical_truth.json`; T08's `analysis.json` says `second_pot_still_held_at_raw_success: true`).

   That makes 12/12 known uc7.1-module successes.
2. **"Hold-only" is not universal.**
   - t0 relayout: terminal failing checks add `supported_by_container_or_contents` and `not_perched_on_rim`, and *after* the scripted release they still fail (`strict_after_release_settle_success=false`).
   - t1 relayout: `terminal_failure_is_hold_only=false`.
   - t6 relayout: raw success fired while the pudding was still gripped, COM z 0.504 m against 0.4505 m after release, 0.66 mm inside the region edge.

   The mechanism is LIBERO's site-region predicate, which accepts an object up to 0.10 m above the site (`third_party/LIBERO/libero/libero/envs/objects/site_object.py:63-80`). The positive oracle can therefore credit mid-carry or invalid states, not only "hold-only" ones.
3. **"Stops before the agent's own release" is not exact.** In T08 and stove-off T08 the agent's release had started (3 actions into `set_gripper open`). The stop cut it off *during* the release.
4. **The expected effect cannot be rescored as stated.** "1 of the 7 would be scored as a failure at its own finish" is true for T04-20 mm only. The other 6 never issued `finish`. Their honest outcome is unknown, so rescoring "at their finish steps" applies to 1/7 runs. Even then only raw success is reconstructible: `trace.jsonl` stores `sim_status.libero_success` per step but no object poses, so strict-at-finish cannot be recomputed offline.
5. **The leak is broader than the sentence W1 cites.** Line 207 also says "Only the independent evaluator terminates a successful task", so every new request tells the agent "not yet successful". Deleting only the "If observation continues" sentence would not close the negative oracle.

**Improvement assessment:**
- **Soundness:** finish-terminal + `give_up` + scoring without `stop_on_success` is sound and general, and it removes both oracles together.
- **Additions needed:**
  - Remove the "Only the independent evaluator terminates…" wording too.
  - Report both the standard LIBERO first-success metric and the finish metric, so results stay comparable with published LIBERO numbers.
- **Risks:**
  - The agent may later disturb a completed relation. That is a legitimate failure, but it will lower the headline number.
  - A 180 s bound scored as failure makes every current H1 run a failure (all are ≥287 s). Report time-to-finish distributions alongside it.

### F2 — mandatory, unrecoverable plan gate — CONFIRMED WITH CORRECTIONS (high for the sweep)

**Checked:**
- **Code:**
  - Runner `scripts/evaluate_agent_speedup_v2_user_check.py:307-309` (single `request_plan`, no `reask_error`).
  - Auto-approve path at `:940-944` reaches `store.approve`, which calls `validate_review_plan` at `src/gpt_field/agent_speedup_v2_user_check.py:367-371`.
  - `ReviewError` becomes `parser.error` at runner `:965-966`.
  - `reask_error` exists in `src/gpt_field/agent_speedup_v2_client.py:1040-1047`.
  - The validator rule is at `agent_speedup_v2_user_check.py:121-122`.
  - The live plan instructions (T04-100 `review/requests/000001*.request.json`) say "sense is any, open or close". Open/close semantics are defined only for `slide`, so the prompt is under-specified rather than contradictory, but it does permit `turn … close`.
- **t9 draft.** Running H1's own validator in H1's `.venv` (`t9_check.py`): V1 t9 = `(place_in A any C)`, `(turn D close)` → `ReviewError: sense must be any except for slide`.
- **Other plans.** The other 12 V1 drafts and all 7 approved uc7.1 plans validate. V1 mean draft time is 28.78 s (`v1_planning/summary.json`).
- **Live plan time:** `initial_plan_model_s` ranges 14.26–52.16 s.
- **Macros:** `agent_stats.json` shows `optional_macro.status = not_requested` in all 7 runs.
- **Plan structure:** `PIDS` A–H (`src/gpt_field/agent_speedup_v2_types.py:58`); both composite plans use A–H.
- **History:** the stale sentence ("The robot measures every distance…") is present in the live plan instructions. The historical uc7 t9 sense close→any hand edit is recorded in `reports/agent_speedup_analysis_20261026/history_intent.md:299`.

**Corrections:**
- **Obstacle-scene count.** There are 4 uc7.1 obstacle-scene plans in `runs/` (T04-20, T04-100, T08, stove-off T08), or 6 with the H3 runs, not 5. None names the can or bottle.
- **Approved-plan count.** There are 7 approved uc7.1 plans, not 8: composite 002 never produced a draft.
- **"Barely uses" overstates it.** Plan `regions` feed `region_directions` in every executive request, and the plan travels as `reviewed_goal_contract`.

**Improvement assessment:**
- **(a) Removing the macro-only `ignored_selection` rules from approval** is a sound veto removal; t9 would pass.
- **(b) A one-shot re-ask** is sound but adds a call. With (a) in place it should almost never trigger.
- **(c) Merging the plan into the first executive call** is a structural design change. It must keep the frozen view-frame and region semantics, and it drops the human-review option that gives the "user check" its name. That is a user decision, not a cheap fix.

### F3 — metric grounding is model mental arithmetic — CONFIRMED WITH CORRECTIONS (mechanism high, effect medium)

**Checked:**
- **Code:** the prompt sentence is at `agent_client.py:224`. The schema at `:237-260` has no locate/triangulate field.
- **Recomputed errors** against `inputs/layout_manifest.json` (`custom_after_passive_hold_object_poses`):

  | Estimate (T04-100 call) | Truth comparison | Error |
  |---|---|---|
  | Left plate, call 5: (−0.055, −0.35, 0.41) | plate_1 (−0.0045, −0.3232, 0.4393) | dx −5.05, dy −2.68 cm → **5.72 cm** horizontal |
  | Left plate after wrist view, call 7 | plate_1 | 2.38 cm |
  | Right plate, call 12 | plate_2 | 3.41 cm horizontal, 2.93 cm low |
  | Can box x-range [−0.015, 0.065] (call 1 `obstacle_map`) | can x −0.0233 | centre 4.83 cm too far toward the camera |

  The can box overlaps only the rear 2.3 cm of the 6.2 cm-deep can (collision bounds x [−0.0545, 0.0077]).
- **Recomputed call costs:**
  - Composite 003 calls 25–35: 11 calls, 297.1 s.
  - Composite 001 and 003 calls 1–9: 174.6 s and 230.8 s.
  - All runs: 46/164 calls follow a `segment_stall`, totalling 938.7 s of 3071.3 s.

**Corrections:**
- **Error axis.** The right-plate error is lateral (+y), not depth. Only the left plate's first estimate and the can box are mostly depth-axis errors.
- **Attribution of the 46 stall handoffs.** By the stalled segment's own profile:
  - 22 are transit-profile moves in free space;
  - 24 are contact-profile (22 approaches and 2 knob turns).

  Many transit stalls are kinematic or tracking problems: composite 001 calls 17–25, and composite 003 calls 26–27 ("Simultaneous rotation and translation are producing poor tracking"). Some are near-arrival stalls (see missed issue M4). So "28% of calls caused by grounding" is an upper bound, not a measurement. The 11-call composite 003 span is mixed: calls 26–28 are kinematic, calls 29–34 are grasp-height misestimates.
- **H3 cross-check.** The H3 runs use byte-identical uc7.1 policy modules plus a SAM+DA3 vision summary, so they are a natural test of "better grounding → fewer calls":
  - **T04:** 13 → 11 executive calls, but control time 287 → 432 s.
  - **T08:** 21 → 11 executive calls; control time 467 → 464 s.

  These are single episodes (`runs/foundation_vision_*/analysis.json`). Fewer calls did not translate into speed.

**Improvement assessment:**
- **Soundness:** a `locate` triangulation field is general and uses only allowed inputs.
- **Risks:**
  - Cross-view correspondence errors, especially between the wrist and external views.
  - Pixel-pointing precision.
  - Earlier pointing-call designs (uc1–uc6) were slow. Keeping `locate` inside the same response avoids that.
- **Alternative:** H3's DA3 depth already exists and is a ready-made alternative.
- **Expected effect:** treat "1–2 fewer calls per pick-and-place" and "1–2 cm" as unvalidated.

### F4 — no kinematic or reach model in the loop — CONFIRMED (medium-high)

**Checked:**
- **Request contents:** the executive `geometry` keys are exactly `schema_version`, `frame`, `image`, `gripper` and `cameras` (`astranomy_agentic_geometry.py:81` enforces this). `measured_state` has joint positions and velocities but no limits or base pose.
- **Limit hits** (limits from `.venv-sim/.../robosuite/models/assets/robots/panda/robot.xml`):
  - Composite 001: requests 15, 17, 18, 19, 20, 21 and 24 have joint 4 at or past its −0.0698 rad upper limit (margins −0.0005 to 0.0).
  - Composite 003: requests 5 and 9 are at the joint-5 upper limit (2.8973 rad).
  - No other H1 run comes within 0.02 rad of a limit.
- **Recovery cost:** composite 001 calls 17–27 = 11 calls, 169.0 s; the beliefs match the quotes in W1.

**Corrections:**
- **Recurrence.** Only one H1 run (composite 001) had a long recovery. The "~10-call recovery" recurs across builds (uc7 t6_011, per the earlier report), not within H1's own runs.
- **Discovery.** The model also read joint angles and named the limit itself (composite 003 calls 5 and 9). Limits are discovered partly by reasoning, not only by stalling.

**Improvement assessment:**
- **Sound:** reporting limits, margins and stage-entry IK feasibility. This is general and robot-description based.
- **Risky:** auto-inserting via-postures can sweep the arm through obstacles, because the router checks only the grip-site path. Keep it report-only, or route the via-posture through the same obstacle check.

### F5 — "until contact" / "until end stop" cannot be expressed — MECHANISM CONFIRMED, EFFECT REFUTED AS STATED (medium)

**Checked:**
- **Code:** the stall predicate is `src/gpt_field/agent_speedup_v2_user_check_fragments.py:404-419` (constants `:42-45`). Pending segments end via `_end`. The schema `:60-69` has no termination predicate.
- **Quotes:** composite 001 call 9 and composite 003 call 20 match W1's quotes. So does T04-100 call 11.

**Corrections — `stall_continuation.py`:**
- Of 46 stall handoffs, only **16** had pending segments. The stalled segment was the fragment's last in 30, so the model would have been called anyway.
- Only **9** follow-up calls simply re-issued the discarded pending skills, and only **5** of those were contact approaches:
  - T08 call 13;
  - T04-20 calls 4 and 22;
  - composite 003 calls 27 and 34.
- Composite 001 has **0** such contact cases and composite 003 has 2, against W1's "about 8–12 fewer calls on the composite".

**Safety risk W1 missed.** Several contact stalls were *protective* — unexpected contacts that the stall handoff caught:

| Run, call | Belief at the stall |
|---|---|
| T04-20 call 24 | "open fingers overlap the mug rim … Closing and lifting did not run" (a close was pending) |
| Composite 001 call 5 | finger compressed on the cabinet lip |
| Composite 001 call 11 | finger deflected by the drawer side |
| Composite 003 call 11 | finger over the drawer wall |

With `stop_on: contact`, these would have continued into a close at the wrong place.

**Improvement assessment:**
- The idea needs a model-declared expected-contact window, or a check of where the stall happened.
- The compliant profile for mechanisms is reasonable.
- The expected savings are small: at most about 5 of 164 calls on this data.

### F6 — rotation-envelope and start-blocked vetoes; router never detours — CONFIRMED WITH CORRECTIONS (medium-low)

**Checked:**
- **Code:** envelope `fragments.py:285-291`; 0.1 rad adapter cap `:234` and `action.py:88-92`; start-blocked refusal at `src/gpt_field/agent_speedup_v2_user_check_routing.py:105-106`.
- **Rotation rejections:** T04-100 1.5724 rad and T08-100 1.5691 rad on call 1 (32.4 s and 30.8 s). These are 2 of the 5 H1-configuration runs; the other three never requested more than 1.2 rad.
- **Start-blocked:** `route_start_blocked` = 6 events in 5 of 7 runs (`routing_metrics.json` `stage_records`). The seventh unavailable route was `invalid_measured_start`.
- **Router behaviour:** 88 routed stages succeeded, with 0 detours and 0 search calls.
- **Calls whose output was vetoed:** 9, with 213.4 s of latency.

**Corrections:**
- **Cost accounting is wrong.** Removing a veto saves the *follow-up* call, not the vetoed one, which is still needed to produce the motion.
  - Follow-up latency is **93.7 s in total (3.1% of model time)**.
  - For the H1 reference it is **18.4 s** (7.7 + 10.7 s), not 52.1 s.
- **Not all vetoes are zero-progress.** 4 of the 9 happened mid-fragment, after 18–59 executed actions.

**Improvement assessment:**
- **Rotation envelope.** Splitting large rotations into sub-stages is fine. Deleting the envelope outright lets a wrist sweep while carrying an object go unchecked, because the router checks only the grip-site point, not the carried object.
- **Start-blocked.** The start-blocked boxes were the model's own boxes around the knob, the grasped object or the just-placed object. Rather than a "least-penetration" escape, which can push through the very object the box stands for, exempt the box that contains the start position for a vertical exit, or let the map mark a box as the manipulated object.

### F7 — closed task registry — CONFIRMED (medium)

**Checked:**
- **Registry and entry checks:** runner `:831` (`choices`) and `:280-282`; `launch.py:35-37`; `libero.py:203-218`; registry `agent_speedup_v2_tasks.py:97-124` (2 + 10 + 9 = 21 entries plus 1 alias).
- **Main-tree additions:** the main tree adds `COMPOSITE_TASK_KEY`, `CUSTOM_TASK_KEYS`, hash-pinned layouts, a composite contract with a 70° external camera, and the task-specific `drawer_bowl_moka_predicates` evaluator (`/workspace/gpt_field/src/gpt_field/ordered_task_evaluation.py:118-131`). The composite runs also differ in `_sim.py` and in both runner scripts (`hashcmp.json`).
- **Version label:** `trial/summary.json` reports `agent-speedup-v2-user-check-7.1` for both prompt-5 and prompt-6 builds.

**Correction (strengthens F7):** "7.1" covers at least three code generations: the 10-28 V1/relayout build, 11-03 prompt-5 and 11-03 prompt-6. The V1 frozen sources differ from H1 in `action.py`, `_agent.py`, `_agent_client.py`, `_control.py`, `_fragments.py` and `_sim.py`.

**Improvement assessment:** passing the task as data plus recording the build digest is sound. However, "compute the reset hash instead of requiring it" weakens the guard against LIBERO's reset nondeterminism. Record it, and refuse only on mismatch with a known hash.

### F8 — robot, camera and gripper facts hard-coded — CONFIRMED WITH CORRECTIONS (medium-low)

**Checked:**
- **Geometry module:** `astranomy_agentic_geometry.py:23` (camera names), `:33-48` and `:90-91` (exact Panda dict check), `:87-88` (square images), `:99-102` (external must be `fixed_world_camera`).
- **Prompt text:** `agent_client.py:53` and `:228-231`; `agent_speedup_v2_client.py:194-195`.
- **Benchmark wording:** `spatial.py:69-71` ("back compartment, left plate").
- **Leftovers in requests:** the DROID/15 Hz `adaptations` from `astranomy_agentic_motion.py:34-35,122` appear in request 1's `capabilities.direct_motion`.
- **Task-keyed file:** `robot_facts_v2.json` has `start_poses` for `libero_10_t2`, `bowl_plate` and `drawer_bowl`, read via `agent_speedup_v2_robot.py:32`.

**Correction:** the text does not "silently contradict" metadata on a new robot. `validate_geometry_metadata` hard-rejects any non-Panda gripper or other camera names, so a new robot fails loudly.

**Improvement assessment:** sound. Prompt edits change model behaviour, so make them before the sweep, not after.

### F9 — directions fixed; region extent and "on" precision undefined — CONFIRMED; IMPROVEMENT PARTLY UNSOUND (medium)

**Checked:**
- **Direction fix:**
  - `spatial.py:198-211`: right = R[:2,0], away = (−r_y, r_x).
  - Live frames: T04 `right_xy=(−1.7e−6, 1.0)`, `away_xy=(−1.0, −1.7e−6)`; composite (70° FOV) `right_xy=(−7e−8, 1.0)`.
- **Region resolver:** half-plane at `spatial.py:334`.
- **t6 relayout:** facts from `implementation_uc71/README.md:66-80`.

**Correction to the improvement.** LIBERO's object-on-object predicate is origin-to-origin xy distance < 0.03 m (`third_party/LIBERO/libero/libero/envs/object_states/base_object_states.py:87-94`; also logged in `physical_truth.json` → `libero.tested`). The plate footprint radius is about 7.9 cm (from `xy_dist_mm` + `com_margin_target_mm`), so "centre inside the support's inner half" allows about 4 cm. It would *not* have prevented T04-20's 3–4 cm miss. A default of "centred as closely as measurable" works better, with no fixed threshold.

### F10 — transport retry classes end the episode — CONFIRMED, STRONGER THAN STATED (medium-high)

**Checked:**
- **Code:**
  - `agent_speedup_v2_client.py:903-914` and `:149-150` (retry codes);
  - `chatgpt_plan.py:150`;
  - `agent_speedup_v2_user_check_transport.py:46,117`;
  - `chatgpt_auth.py:450-480` (refresh only when local expiry is within 60 s).
- **Composite 002:** HTTP 401 `token_expired`, 0 actions, 14.04 s wrapper time (`result_summary.json` → `failure`; `initial_plan_failure_audit.json`; `runner.log`).

**Missed evidence:**
- The uc7.1 t8 relayout's first attempt died on **HTTP 520 at its first execution request, with 0 actions** (`UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md:245`). The protocol itself notes that 520, `unknown_error` and `request_deadline_exceeded` are not retried (`:165`).
- The same-day V1 frozen sources have byte-identical `agent_speedup_v2_client.py`, `_transport.py`, `chatgpt_plan.py` and `chatgpt_auth.py` to H1.
- So 2 of the known uc7.1 launches died on non-retried transport errors.

**Improvement assessment:**
- **Sound:** force a refresh and resend once on 401; treat 5xx and deadline errors as bounded transient retries.
- **Caveat on resume:** "resume from the last committed observation" must be reported, and not counted as a fresh episode. That keeps it compatible with "no episode retries".

### F11 — the frozen H1 build never ran an official task — CONFIRMED (high)

**Checked (`hashcmp.py`, 46 source roots):**
- Every root with H1-identical user-check modules is on a custom scene:
  - 7 uc7.1 runs (T04+can, T08+bottle, stove-off T08, composite ×2, plus the two 20 mm T04 runs, which are prompt-5 and *not* identical);
  - 2 H3 runs (T04+can, T08+bottle);
  - 3 astranomy runs (T04+can).
- A disk-wide grep for `user-check-agent-first-prompt-6` finds no other run.
- H1 has no `runs/` directory. `validation/VERIFICATION.md:95` confirms the replay is not a success-rate estimate.

**Missing context:**
- The only official-scene uc7.1-family run is `default_routing_t04_20261103_001` (`run.py:1` "One authorized official LIBERO t4 trial"; `result_summary.json` `layout: official`). It is a prompt-5/fragments-3/20 mm build: raw ✓, strict ✗, 1 + 10 calls, 338.1 s.
- The deleted 11-02 transit50 runs on t00 (a retry) and t01 have unverifiable configuration and a 0.05 m cap (`robocasa365/setup/uc71_baseline_audit.json:86-122`).

**Improvement assessment:** the sweep is the right acceptance test, with two prerequisites:
1. Run a no-model reset-hash preflight over all 21 official tasks in H1's own environment. Reset equality has been validated only for T04+can (`validation/results/reset_mismatch_diagnosis.json`), and any mismatch aborts before the model call (runner `:287-289`).
2. The obstacle variants other than T04+can are not packaged in H1. T08+bottle needs its scene files added; stove-off T08 and the composite need main-tree code.

### F12 — dead macro layer and compaction — PARTIALLY REFUTED (low)

**Confirmed:**
- 155/164 requests carry `context_detail_compaction` (`agent_client.py:545-553`).
- The largest compacted request is 26.3 k characters.
- The macro-related fields (`macro` + `native_evidence` + `capabilities.native_macros`) are about 3.16 k characters per request.
- Input tokens grow from 8.5 k to 260.6 k in composite 001.
- `measured_pose` is reduced to xyz plus `{"more_in_audit": 5}`.

**Corrections:**
- **Macro share range.** The per-request share is 12.0–20.8% (median 15.1%). W1's 15.2–16.4% is the per-run aggregate.
- **"Hidden sub-reason" is refuted.** T08 request 15 contains `route_start_blocked` in `context.execution.stage_outcomes[1|3].detail`. The model's belief names it: "The long transit was rejected at its start…". In all 6 start-blocked handoffs, the sub-reason is present in the request and the model identified the start rejection.

## Missed issues

1. **The evidence base is incomplete, and the positive oracle is worse than "hold-only" (high).** Missing from W1's tables:
   - relayout t0, t1, t6 and t8;
   - transit50 t00 (a retry) and t01;
   - 2 H3 runs with H1-identical modules.

   Across all of them, 12/12 successes ended with an object held. LIBERO site regions accept an object up to 10 cm above the site (`site_object.py:63-80`), which is how t6 counted as success mid-carry. t0's raw success fails strict checks even after release.
2. **A second transport death (medium-high).** HTTP 520, 0 actions, uc7.1 t8 relayout, under the same retry code (`UC7_1_ASTRA_HIGH_TEST_PROTOCOL.md:165,245`).
3. **H1-as-packaged can reproduce only 1 of its 5 evidence runs (medium).**
   - `tasks/` holds only `t04_can_obstacle`.
   - T08+bottle used main-tree override files.
   - Stove-off T08 and the composite need main-tree registry code, the task-specific ordered evaluator (`ordered_task_evaluation.py:118-131`) and a 70° camera contract.
4. **Stall classes W1 did not count (medium).**
   - 22/46 stall handoffs are transit-profile.
   - Some are near-arrival convergence stalls just outside the fixed tolerances: H1-reference calls 7 and 13 at 4.8/4.9 mm against the 4 mm transit tolerance; stove-off call 5 at 2.09 mm against the 2 mm contact tolerance (`fragments.py:27-34,383-419`).
   - These report healthy arrivals as `failed`, and when segments are pending they drop them.
5. **F5's continuation upside is small, and protective stalls would be bypassed (medium).** 16/46 stalls had pending segments; 9 were resumed verbatim; 5 of those were contact approaches. The protective stall cases are those in the F5 table above.
6. **F6 veto cost is overstated by about 2.3× overall and 2.8× on the reference (medium-low).** 93.7 s versus 213.4 s overall; 18.4 s versus 52.1 s on the reference.
7. **H3 cross-check against F3 (medium-low).** Calls fall, wall time does not (T04 287 → 432 s).
8. **Reset-hash portability risk for the sweep (low-medium).** See F11.
9. **F9's "inner half" default is looser than LIBERO's 3 cm predicate (medium).**
