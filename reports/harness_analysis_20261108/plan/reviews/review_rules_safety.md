# Review of the integrated plan: user rules, safety order, evaluation validity, coverage, decisions

Key: `review_rules_safety`. Date: 2026-11-08. Read-only review. Nothing under `/workspace/gpt_field` was edited, and no simulator, model call or GPU job was run.

**What I read.**
- `INTEGRATED_PLAN.md` in full (2,173 lines).
- The analysis `README.md` §0–§6 and `work/w2/cross_critic.md` §2 (IC1–IC15).
- The six designs, wherever the plan defers detail to them.
- `ws_runner_scripts/power.py`.
- The H1 code at the lines the plan cites: `launch.py`, `run.sh`, `.gitignore`, `chatgpt_auth.py`, `agent_client.py`, `agent.py`, `fragments.py`, `libero.py`, `agent_speedup_v2_tasks.py`.
- The H1 and service venv `site-packages` path hooks; `vast-capabilities` (workspace persistence).
- The user's memory notes on live-first runs, task-general rules, subscription sharing and LIBERO reset nondeterminism.

Line references `:N` are `INTEGRATED_PLAN.md` unless a file is named.

---

## Verdict

The plan is mostly faithful to the user's rules and to the cross_critic safe order.
- Measurement comes first, and INV-6 enforces it.
- H1 stays byte-identical: it is cloned, tagged and never edited.
- Every behaviour switch is a sourced config flag, and the loader enforces IC rules as invariants (INV-1…INV-D).
- Subscription-only use is kept: the launcher pops `OPENAI_API_KEY`, the model is fixed to GPT-6 Astra at high reasoning, and the canary also runs at high reasoning.
- Every critical or high report item maps to a component.

Seven defects must be fixed before the affected milestone:
- **Four leave unsafe interim states** that the IC order was meant to prevent:
  - an obstacle-exclusion rule that drops the objects the robot is touching (MF-1);
  - an M3 clearance regression (MF-2);
  - full-speed motion on an empty map when perception fails (MF-3);
  - contact-window semantics that let early forbidden contacts count as declared (MF-6).
- **Two hurt the acceptance build directly:**
  - new zero-motion vetoes in M4 (MF-5);
  - filled container boxes with arm links never permitted, which block drawer and microwave placements (MF-4).
- **One fits an acceptance-build switch to a single 6-call smoke on one task** (MF-7).

All seven have general, cheap fixes. Eighteen should-fix items follow. They cover credentials, backup, environment pinning, held-out evaluation, statistics, scoring symmetry, escape semantics, parameter provenance and live-first process.

---

## Must fix

### MF-1. `inside_robot` drops obstacles the robot is near or touching
- **Where.** I-08 `excluded … "inside_robot"` (`:667`); X16 (`:1144`, "Geometry's `inside_robot` uses `robot.body.silhouette`"); `ws_geometry_tools.md:331` and `:1019`.
- **Evidence.**
  - GT3 says: "`inside_robot`: the box intersects the robot FK silhouette **at the current joints**. It is excluded from obstacles, because it is likely self-detection." `:1019` gives `robot_silhouette(joints, finger_q, camera) -> polygon list: for inside_robot`.
  - A static object's box meets the robot's silhouette exactly when the hand approaches it, presses it or occludes it. That is when collision reasoning matters: the T08 bottle press (47 actions), palm-on-mug grasps, and drawer-001 link-on-cabinet contacts.
  - The rule is listed under "Plausibility (warnings, never vetoes)" but acts as a hard exclusion.
  - Self-detection is already removed upstream by PS-3 (FK raster plus depth replacement), so this second filter adds risk and no protection.
- **Fix.**
  - Never exclude an object because it meets the robot at the current joints.
  - Test self-detection only at the record's own `observed_step` joints, in 3D against the dilated body model. Require near-full containment, with the bound derived from PS-3's measured raster recall.
  - Never apply it to an object previously observed static and clear of the robot.
  - Otherwise keep the object, with a `near_robot` warning and σ inflation.
  - Offline test: the T08 bottle box during actions 37–83 and both T04 mugs at grasp stay in `ObstacleSet.objects`.
  - S3 metric: objects excluded as `inside_robot` that are real by post-hoc truth, target 0.

### MF-2. M3 removes the only hand and held-object allowance before the body sweep exists, and the prompt overstates path checking
- **Where.**
  - X20 (`:1148`): `obstacle_map` leaves the schema at M3.
  - I-08 `:671`: `inflate_m = k·σ`.
  - X14 (`:1142`): `PointStagePlanner` with `GripSiteBoxOracle` in M2–M3.
  - `ws_executor_control.md:433-435`, `:460`: `model_clearance` becomes per-object uncertainty.
  - GT8 text, `ws_geometry_tools.md:769`.
- **Evidence.**
  - H1 defines the model's `clearance_m` as covering "hand/carried-object extent and estimation error" (`agent_speedup_v2_user_check_agent_client.py:95-96`). Runs used 0.04–0.06 m (cross_critic C3).
  - In M3 the grip-site router's margin becomes kσ (≈ 1–2 cm) + `m_track`, with nothing for:
    - the open fingers (±54–66 mm);
    - the hand footprint (−104…+100 mm along the jaw);
    - a held mug hanging ≈ 6 cm below the grip.
  - cross_critic C3 already shows the point router approving a colliding T04 carry band (0.54–0.57 m) even with the model's 0.04 m clearance. M3 makes the margin smaller.
  - GT8 tells the model "The harness applies these numbers when it resolves targets and checks paths". Until `swept.mode=on` (M4) only the grip-site point is checked. That sentence can erode the self-chosen high carries that gave H1 T04 its 0 can contacts (cross_critic C3).
- **Fix.**
  - From M3, `GripSiteBoxOracle` inflates each box by a derived robot term:
    - the hand and finger half-extent per world axis at the stage orientation and commanded gap (`robot_facts_v2.json` / `panda_robot_model_v1`);
    - when an object is held, the SceneMap held record's below-grip extent and footprint (or the fallback jaw-column envelope).
  - These come from robot geometry and measured held state, so they satisfy rule 2 and add no task constant.
  - Make GT8 truthful per milestone. While `swept.mode≠on`, say that only the grip-site point is checked and that the model remains responsible for the hand, fingers and carried object.
  - S3 metrics: carry margin against truth, and hand/finger/held contact episodes in routed stages, vs b002.

### MF-3. A perception outage leaves the robot moving fast through an empty map
- **Where.** §3.6 `:339` (fail-open); INV-13 `:178` (model boxes leave the schema once perception is on); SP9 (`ws_swept_planner.md`, cap from clearance, clipped at 0.10 m); `:70` (7 of 8 GPUs busy at writing).
- **Evidence.**
  - If the service is down or out of memory at reset, the SceneMap holds no objects for the whole episode. Mid-episode it keeps stale records.
  - From M3 the model cannot write boxes; it can only use `locate` + `add_from_locate`.
  - The point or body planner then sees free space, and SP9's clearance-scaled cap grants the full 0.10 m wherever clearance is "large".
  - Fail-open for perception thus becomes fail-fast for motion, and the scored episode measures a different policy than the arm claims.
  - On this shared host an out-of-memory failure is plausible.
- **Fix.** Define a degraded motion mode in the executor:
  1. At reset, wait a bounded, measured time for the first `ok` snapshot. Otherwise end with `infra_error/service`: relaunch, unscored.
  2. Mid-episode, inflate kept records' σ with age. Cap transit at the contact-safe speed (SP9 with unknown clearance → `s_contact`, or the 20 mm cap). Disable cruise and the approach schedule.
  3. Flag cells with any outage in the coverage map, and report arm results with and without them.
  - Add an invariant: `swept.speed≠off ⇒ executor.degraded_mode`.

### MF-4. Containers and articulated fixtures are filled boxes, and arm links are never permitted, so the acceptance build cannot place into drawers or the microwave
- **Where.** I-08 permission table (`:675-684`); SP5 table and `ws_swept_planner.md:637` ("object-map OBBs … fills hollow interiors"); PS-6 primitives `cylinder|box`; `allow_contact` values `palm/fingers/held_object` only (`:741`); backlog `:1971`; M4 SP6 and SP11-on; INV-12.
- **Evidence.**
  - Several official scenes need the hand, and for the microwave also link 7, inside the fixture's bounding volume (`agent_speedup_v2_tasks.py:50-95`):
    - LIBERO-10 t3 ("put the black bowl in the bottom drawer of the cabinet and close it");
    - t9 ("put the yellow and white mug in the microwave and close it");
    - probably t5 (caddy back compartment);
    - LIBERO-Goal g3 ("open the top drawer and put the bowl inside").
  - With a filled box, links forbidden on every object, and the palm allowed only for `push`/`insert`, SP6 stops at the box boundary (`standoff_handoff`, `no_route`). The model's only exits are:
    - `allow_contact palm`, which still leaves the links forbidden;
    - `remove`, which deletes the fixture, so drawer 001's 414 link-on-cabinet samples come back.
  - The acceptance sweep would score representation artefacts as framework failures on 3–4 of 20 official scenes.
  - The plan's mitigation ("contact permissions; kind edits", `:1605`) does not cover links.
- **Fix.** Before M4 turns `swept.mode=on`, add one general mechanism and a user decision. Preferred:
  - represent objects with an observed open interior (`kind ∈ {fixture, container}`) by their observed walls and rim instead of a filled primitive, using PS-8 height-field cells (2.5-D: walls and floor seen from above);
  - for front-opening fixtures, add the opening volume observed in the external view.
  - Minimum alternative: a scoped `allow_contact inside` that permits palm and distal links inside the intent anchor's box on `inside`/articulated stages, with the guard as judge, reported.
  - Validate with an SP12 R-B replay of drawer 001/003 on live-style perception boxes plus a t9 truth fixture.
  - Add a coverage-map column "stages handed off at a container boundary".
  - Until then, run container-intent stages on the point planner + guard, with the body plan in shadow.

### MF-5. M4 adds vetoes the plan says it does not add
- **Where.**
  - M4 goal `:1627` ("never as a new veto");
  - `body_plan` statuses `:895`;
  - SP6 step 7 (`ws_swept_planner.md:465`: "If it is blocked, return `jaw_sweep_blocked` with the pair; **no motion**");
  - UD-36 `:2020` (truncate direct motions);
  - X17 `:1145` (ContactIntent overrides scoped to "the next `execute` fragment");
  - SP5 ("Without [a declaration], every contact is forbidden").
- **Evidence.**
  - A gripper open or close whose finger sweep touches a kσ-inflated forbidden box is refused with zero motion. Examples: releasing beside a container wall, or closing next to a neighbour.
  - Direct `motion` has no target, `stop_on` or contact field, and edits do not reach it. So every fine adjustment near the target is truncated at the last clear sample, and the model has no lever.
  - H1's prompt prescribes `motion` exactly for "uncertain fine adjustment" (`agent_client.py:166`).
  - Each refusal costs a call. Rule 2 says to prefer removing vetoes; IC12 says to use the sweep as a planner, not a veto.
- **Fix.**
  - Make the jaw sweep a planner step:
    - open or close only to the clear width (a partial `set_gripper` width); or
    - first insert a short guarded retreat along −approach by the measured interference + band;
    - report the pair either way.
  - Direct motions inherit the last fragment's `ContactIntent` (anchor, parts, overrides). Truncate them only for forbidden pairs outside that intent; otherwise execute under the guard.
  - Add `jaw_sweep_*`, truncations and "calls caused" to the M4 coverage map.

### MF-6. A null `contact_window_m` makes every contact "declared"
- **Where.** I-14 `segment-uc72-1` `:962` (nullable); `ws_executor_control.md:251` ("either `contact_window_m` is null or remaining ≤ window + tol → `contact_reached` … pending segments continue"); `:867` ("declared by the model").
- **Evidence.**
  - cross_critic IC1: 3 of 15 forbidden contact onsets began 2.2–6.8 cm before the commanded endpoint, and 5 more at 1.7–2.2 cm.
  - Under a null window these become successes, and the pending close and lift continue.
  - The gap predicate then reports "object present" for whatever the fingers closed on. From M5 a 12-segment commit carries on, which is exactly IC5's failure mode.
  - The window is also a model-written metric, against the approved direction.
- **Fix.**
  - The executor derives the window:
    - with a `target` reference: `k·sigma_along_motion_m` (ResolvedTarget) + guard detection distance (`osc_response.json` guard) + tolerance;
    - without measured σ: the plant-derived detection distance + tolerance only.
  - A model-written window may only shrink the derived one.
  - From M3, also compare the `guarded_stop` position with the intent anchor (the SceneMap's nearest object). Report `unexpected_contact_other_object` and handle it as undeclared.
  - Offline: replay the 15 onsets; the 3 early ones must classify as `unexpected_contact_early`.

### MF-7. The acceptance build's asynchronous-verification switch (UD-39) rests on one task's 6-call smoke
- **Where.** UD-39 `:2023`; M5 smokes `:1754-1755` (T04 only, `--max-exec-calls 6` per arm); `ws_protocol_context.md` UD3.
- **Evidence.**
  - Rule 1 says no change may be fitted to one run. The user rejected exactly that pattern on 2026-10-21 (memory "v2-scope-general-framework").
  - "≤ 1 stale rejection per 10 verify calls" cannot be estimated from ≤ 6 calls, and late-detected failures are rare events.
- **Fix.** Either option:
  - Decide UD-39 from an interleaved, capped, user-started A/B over ≥ 3 task families (for example T04, T08, the drawer composite, one LIBERO-Goal and one multi-object LIBERO-10 task), with ≥ 30 verify calls per arm. Report the rates with confidence intervals.
  - Or run both `m5_blocking` and `m5_candidate` in S5 and pre-register which one is the acceptance arm.

---

## Should fix

**SF-1. Use one credential store for every arm and tool.**
- **Evidence.**
  - H1's launcher defaults `--store-dir` to `ROOT/.private/chatgpt_plan` (`launch.py:77-78`), and `chatgpt_auth.py` defaults to `parents[2]/.private/chatgpt_plan`.
  - Refresh tokens rotate under a per-directory `fcntl` lock (`chatgpt_auth.py:450-478`).
  - H1 has no `.private/`; the working store is `/workspace/gpt_field/.private/chatgpt_plan`.
  - Worktrees `arms/<tag>` default to empty stores. The plan never names a store; only the protocol design uses a relative `.private/chatgpt_plan` (`ws_protocol_context.md:919`).
  - If stores are copied, one arm's rotation invalidates another's refresh token. H1 does not retry 401, so pristine-arm episodes would end as arm-correlated auth infra failures.
- **Fix.**
  - Pass one absolute `--store-dir` to every arm (including H1's launcher) and every tool (PC7, canary, pointing smoke, sweep runner). Never copy credential files.
  - Record the store path, not its contents, in `build.json`.
  - Add an M1 offline check: two processes refresh concurrently through the lock.

**SF-2. Back the build up off the box (missing decision).**
- **Evidence.** `vast-capabilities` reports `workspace_is_volume=false`. §2.1 renames the origin and adds no remote. 173 engineer-days of code and every sweep artifact would exist only in this container.
- **Fix.** Add a decision "UD-00":
  - a private remote for the new repository (never H1's public origin), with tags pushed at each freeze;
  - an artifact sync for runs, sweeps and benchmarks, with an exclusion list (`.private/`, tokens).

**SF-3. Pin the shared environments as part of the frozen build (rule 6).**
- **Evidence.**
  - §2.1 `:92-93` symlinks H1's `.venv` and `.venv-sim`, so any `pip install` in the new build changes H1's environment.
  - H1's `.venv` has no `jsonschema`, yet I-06 specifies a Draft 2020-12 JSON Schema; installing it into the shared venv would be the obvious shortcut.
  - The service venv loads SAM 3.1 and DA3 from editable installs under `/workspace/gpt_field/artifacts/foundation_vision/vendor/` (`__editable__.sam3-0.1.0.pth`, `_editable_impl_depth_anything_3.pth`). Source edits there change neither a pip-freeze package hash nor the weight hashes.
- **Fix.**
  - Treat the symlinked venvs as read-only. Before each pristine episode, test that the H1 venv package hash equals its M0 fingerprint.
  - If a package is ever needed, build separate venvs from H1's `requirements-controller.txt` and `provenance/simulator_resolved.lock.txt`.
  - Validate schemas with stdlib code, or only in tests.
  - Hash the editable `sam3` and `depth_anything_3` source trees into `hello.code_sha256` and the digest.

**SF-4. Keep a held-out acceptance condition, and calibrate perception beyond four scene types.**
- **Evidence.**
  - §7.1 `:1811` flags only relayouts t00, t01 and t06 as historical. But prior development used several official scenes (memory notes; `agent_speedup_v2_tasks.py:105-117`):
    - LIBERO-10 t2 (v2 primary task), t3 and t6 (user-check v6), t6 and t8 (eval3);
    - the t4 and t8 scenes (H1's obstacle variants);
    - g3 (the `drawer_bowl` regression task).
  - All six designs' offline checks use T04, T08, stove-off and drawer data, and S1–S4 then iterate on the same 34 scenes.
  - "Repeat" is never defined, and the registry holds only init-0 hashes (`agent_speedup_v2_tasks.py:46-47`).
  - PS-12/GT11 σ models and binding targets (11/11 T04, 3/3 T08) come from four scene types.
- **Fix.**
  - (a) Flag every previously used scene as historical, and report held-out and historical subtotals in every coverage map.
  - (b) Pre-register S5 conditions not used in S1–S4: official init states 1–3 (hashes adopted through the preflight and UD-03) and fresh seeds for the generated obstacle relayouts. Define "repeat" explicitly.
  - (c) Add a PS-12/GT11 split made by re-rendering S1's decision frames for all 34 scenes through INT-5. Fit σ on the historical set, validate it per family on S1 frames, and never refit on S5 conditions; any refit is a new tag.

**SF-5. Fix the statistics of the acceptance decision.**
- **Evidence.**
  - `ws_runner_scripts/power.py` simulates one constant logit shift for every scene, then applies McNemar to pooled (task, repeat) pairs. If per-scene effects differ, repeats within a scene are correlated and McNemar becomes anti-conservative.
  - The lexicographic rule (`:1772-1782`) has no non-inferiority margins, so an "unresolved" level 2 has no defined path to level 3.
  - Level 3's geometric-mean time over "pairs where both succeeded" (`:1775`) is survivorship-biased.
  - S5 lists the pristine arm (`:1764`), but its cost row (`:1857`) leaves it out.
- **Fix.**
  - Use a scene-clustered test: a sign-flip permutation over per-scene success differences, or a stratified CMH test.
  - Re-run the power simulation with heterogeneous effects and report its type-I error.
  - Pre-register a margin per level, and the rule for proceeding when a level is unresolved.
  - Make P(strict success ≤ t) and a cap-imputed restricted mean the primary fast metrics.
  - Fix the S5 cost row.

**SF-6. Make scoring and caps symmetric, classify infrastructure failures by cause, and define async accounting.**
- **Evidence.**
  - `cap_wall_s` is described as a "hang guard" (`ws_runner_measurement.md:108`), yet every cap is scored (I-18 `:1065`).
  - The pristine arm is capped by touching H1's operator stop file at the call or wall cap (`ws_runner_measurement.md:684`). That records `operator_cap`, which is unscored (`:1065`). The cap is also wall-based, while H1 spends ≈ 0.158 s of overhead per step.
  - Nothing says whether verify calls (M5), schema re-asks or the plan call count toward `cap_exec_calls` or "calls per success".
  - Simulator crashes are treated as infra and relaunched (§3.6; `ws_runner_measurement.md:237`) even when a violent contact causes them.
  - In async mode the arrival latency L is wall latency (§3.5), so failed attempts and quota pauses could move the robot or inflate virtual time.
- **Fix.**
  - Classify `cap_wall` as `infra_error/hang`, relaunched once.
  - Score the adapter's pristine-arm caps as `cap_*` (censored), computed from H1's logs in policy-time terms (steps·dt + model latency).
  - Define the counted calls once (plan + decide + verify + re-asks; excluding the canary and infra retries) for both caps and metrics.
  - Score simulator errors that follow a forbidden contact within a few steps, or MuJoCo instability warnings, as `sim_instability`. Report infra rates per arm.
  - In async mode, pause the simulator during failed attempts and `QuotaPause`, and use the successful attempt's latency for virtual arrival.

**SF-7. Fix escape and goal semantics in M2–M3 (IC4, IC13, K3).**
- **Evidence.**
  - The EXC-B2 escape (`ws_executor_control.md:437-443`) checks grip-site box distances only. Candidates include horizontal gradients, and "+z gets no preference".
  - Nothing exempts the object just grasped or released until SP6 rule (ii) arrives in M4.
  - After a release, the open fingers straddle the placed object, which is now `moved_unobserved` with inflated σ_z. A horizontal escape drags it and fails the strict check.
  - `route_goal_blocked` stays a zero-motion veto (`:444`) just when automatic boxes multiply near destinations (IC13).
- **Fix.**
  - Port SP6 rule (ii) into `TransitRoute2`: the last contact anchor is `permitted_nonincreasing` for the next stage, and the preferred exit from inside it is −approach.
  - Drop horizontal escape directions while the gripper is closed on an object that rests on a support.
  - Turn `route_goal_blocked` into SP6's standoff handoff.
  - Add the evaluator metric "object displacement during escape legs".

**SF-8. In M2–M3, choose the rotation site by the finger sweep (IC3).**
- **Evidence.**
  - The EXC-B1 interim rule picks the site with the largest grip-site clearance.
  - The finger keypoint sweep (±66.4 mm, tip 9.25 mm) appears only in an offline test, "not at runtime".
  - IC3 requires the realised orientation profile to be in the swept check.
- **Fix.** Use `robot.kinematics` (INT-1, M2) to sweep the finger and pad keypoints along each candidate `ExecutedProfile` at runtime. Choose the site by keypoint clearance (planner, no veto) and report it.

**SF-9. Bring UD-27 in line with the approved direction.**
- **Evidence.**
  - The direction (`:5`) is "take metric estimation away from the model".
  - UD-27 (`:2011`) keeps raw absolute targets through M5, and the backlog (`:1974`) decides removal from S3–S5 shares, which arrive after the acceptance freeze.
  - `target-ref-1.offset_m` is unbounded (`:714`), so `anchor: S1` plus a large offset is an absolute target in disguise.
- **Fix.**
  - Recommend removing absolute `xyz_m` for `move_to_pose`/`approach` at M3, or at the latest at the M5 freeze, decided from S3's share.
  - Keep `move_relative`, direct motion and anchored offsets.
  - Flag `offset_beyond_anchor` (offset larger than the anchor's extent + kσ) as a warning, and count it with raw targets in the share.

**SF-10. Default UD-29 to "dispute", not "remove".**
- **Evidence.**
  - UD-29 (`:2013`) lets the model delete perception obstacles. `removed_by_model` objects are excluded from the `ObstacleSet` (`:667`), and nothing restores them on re-observation.
  - That hands geometric authority back to the model, whose errors are reproducible bias (G4). An example: deleting a real obstacle to clear a refusal.
- **Fix.**
  - Make "dispute" the default: σ inflated, the object kept for speed scaling and guard attribution, scoped to the next fragment, and auto-restored when perception re-confirms it.
  - Allow "remove" only for single-view detections.
  - Count contacts with disputed or removed objects.

**SF-11. Justify the contact taxonomy (UD-06) and keep it independent of the policy.**
- **Evidence.**
  - UD-06 (`:1990`) recommends counting palm-on-current-target contacts as "reported, not forbidden" without giving a reason. This class is 55 % of H1's forbidden samples and is the M4 depth bound's target.
  - `intended` is true when the active stage declared contact with that part (`ws_runner_measurement.md:433`). So the model's own declarations and `allow_contact` edits move contacts out of the forbidden count.
- **Fix.**
  - Justify (i), or report collision-free success both with and without palm-on-target.
  - Compute `intended` on the evaluator side:
    - the contacted MuJoCo body must map to the declared anchor (M3+), or to an articulated body named in the BDDL goal (M1–M2);
    - "current grasp target" comes from truth (the object next held).

**SF-12. Correct parameter provenance and make planning deterministic (rule 2).**
- **Evidence.**
  - I-01 `:396-397` labels β "derived: transit arrival tolerance" (gripper, 4 mm) and "measured: SP2 p90 arm prediction error" (arm, 10 mm). The swept design defines β as a build-time sphere-cover resolution (`ws_swept_planner.md:154`), and the arm prediction band is a separate quantity.
  - k = 2 is "a stated confidence convention" (`ws_perception_service.md:560`).
  - The look-motion trigger is "σ above a target such as 1 cm" (`:1302`).
  - SP6's 150 ms wall-clock budget (`ws_swept_planner.md:479`) makes `no_route_budget` handoffs depend on host load and breaks replay determinism.
- **Fix.**
  - Label β `user_decision:UD-33`, with the coverage and bulge measurement as evidence.
  - Derive k from the GT11 coverage curve (the smallest k with ≥ 95 % per-axis coverage).
  - Set the look trigger to UD-34's bound (σ_support > pad height).
  - Give planners a deterministic evaluation-count budget and only report wall time.
  - Note that INV-1 makes the M5 approach schedule (UD-41) impossible while UD-34 keeps the depth bound report-only.

**SF-13. Keep the process live-first and respect who starts which runs.**
- **Evidence.**
  - M0 acceptance requires the bit-exact replay and the preflight (`:1217`), and merges require green offline checks (`:98`). The first model call therefore comes only in M1, in weeks 5–6.
  - The M1 g8 "smoke" (`:1270-1276`) is meant to reach `finish`, i.e. a full task; the protocol design itself notes "a finish needs a complete task".
  - The M2 smokes use 20, 20 and 30-call caps (`:1388-1390`), above H1 T04's 13 calls.
  - "Claude may start" the GPU perception service (`:1521`) on GPUs shared with training jobs.
  - The subscription is shared with the user's own astranomy agent; the memory note says to keep v2's call load low while theirs runs.
- **Fix.**
  - Add a capped live smoke (≤ 3 calls) on `h1_equivalent` at M0.
  - Record replay and merge checks in the tag instead of blocking on them.
  - Mark the g8 finish run and every smoke with ≥ 13 calls "tell the user first", or hand them to the user.
  - Add decisions on (a) whether Claude may start the GPU service, and (b) a subscription load policy: no Claude smokes during user sweeps or the user's other agents, plus a daily call budget.

**SF-14. Remove the remaining oracle-adjacent wording.**
- **Evidence.**
  - PC1's banned list (`ws_protocol_context.md:198`) covers phrases, but not the `task_success` keys that M1 still emits:
    - `agent.py:166`, `:248` ("unknown_to_policy"), `:372`, `:396`, `:420` ("independent evaluator decides task success"), `:608`;
    - `fragments.py:482`.
  - Test T1.2 renders only the first and second payloads.
  - "Simulation pauses during inference" is removed by PC1 in the traceability table (`:2042`) but by PC6 in the protocol design (`:32`, `:59`, `:744`).
- **Fix.**
  - Ban `task_success`, `evaluator` and `evaluator_success` in every model-facing payload, and run T1.2 on one payload per feedback type.
  - Remove the pause sentence in M1 and say so in one place.

**SF-15. Detect wrong-object grasps.**
- **Evidence.**
  - The held record attaches to "the last approach stage's anchor" (`:646`).
  - Perception's `held_confirmed` carries the co-moving instance, but no rule compares it with the attached anchor, and gap predicates cannot tell which object is held.
- **Fix.**
  - Add a `held_identity_mismatch` barrier yield, and re-attach the record to the co-moving instance.
  - Count it among S5's late-detected failures.

**SF-16. Make the IC11 compliance claim substantive.**
- **Evidence.**
  - §2.5 `:191` says IC11 is met because no "unknown space is traversable" rule exists. In fact every unmapped volume is free at full speed.
  - `unknown_volumes` (`:668`) is never populated (SP9: "if I1 provides them").
- **Fix.** Either:
  - populate `unknown_volumes` from the PS-8 height-field `shadow` cells near task objects and let SP9 slow down there; or
  - restate the claim as "H1 semantics kept", and add an S3 metric for contacts with undiscovered objects.

**SF-17. Base prompt and context decisions on more than two obstacle scenes (rule 1).**
- **Evidence.** The PC7 A/Bs behind UD-20, UD-21 and UD-26, and the H3 channel ablation, use only the T04 and T08 transit100 runs (`:1377-1383`). Saved stove-off and drawer 001/003 requests already exist.
- **Fix.** Add those runs to every A/B that feeds a user decision, and report results per scene family.

**SF-18. Make replays deterministic on reset-count-sensitive scenes.**
- **Evidence.**
  - LIBERO KITCHEN_SCENE4/6/8 (t3, t9, t8) initial states depend on the env's hard-reset count (memory "libero-corrected-init-nondeterminism").
  - H1's worker rebuilds the env on every `reset()` (`libero.py:818-872`), so only tools that reuse an env are exposed.
  - M0 check 7 tests re-simulation on T04 only (LIVING_ROOM_SCENE5).
- **Fix.**
  - INT-5 (including its `--sweep` mode), R8's post-hoc video and the preflight must build the env through the worker's `reset()` for each episode.
  - Add a replay-exactness check on t3, t8 and t9.

---

## IC1–IC15 compliance

| IC | Plan mechanism | Verdict |
|---|---|---|
| IC1 | INV-1: guard + depth bound + perception before `cruise_approach`; slow zone = measured σ + braking + guard distance | Order respected; MF-6 weakens the guard's declared-contact semantics |
| IC2 | braking (M2); cruise clipped at 0.10 m and scaled by body clearance (M4); INV-2 | Respected. The inherited 0.10 m cap persists through M3 (baseline, not a speedup); MF-3 (outage) |
| IC3 | veto → sequencing (M2), INV-3; executed profile in the M4 sweep | Partial: M2–M3 site choice ignores the finger sweep (SF-8) |
| IC4 | exact distances, then a checked escape (M2), INV-4 | Partial: the escape is grip-site only and can drag a just-released object (SF-7) |
| IC5 | guard + predicates (M2) before commits (M5), INV-5; late failures counted | Order respected; MF-6 and SF-15 open holes in the predicates |
| IC6 | finish scoring in M1, INV-6 | Respected |
| IC7 | inventory in the plan call; fold deferred; INV-7 | Respected |
| IC8 | references resolved by the harness | Partial: raw targets and unbounded offsets remain (SF-9) |
| IC9 | overlays replace images; append-only; INV-9; PC7 A/B | Respected; A/B evidence base thin (SF-17) |
| IC10 | frame ring; INV-10 | Respected |
| IC11 | no occupancy map; "unknown affects speed only" | Claim hollow (SF-16); equals H1 semantics |
| IC12 | body planner only with a perception map, INV-12 | Respected in order; MF-4 and MF-5 make it block or veto |
| IC13 | exclude → escape → automatic boxes → sweep | Order respected; MF-1 excludes the wrong objects; `route_goal_blocked` veto (SF-7) |
| IC14 | relocalise at stage entry, `age_at_use`, PC6 staleness and barriers | Respected for the simulator; async accounting gaps (SF-6) |
| IC15 | verbatim reasons, enforced by test | Respected |

## User rules

| Rule | Status | Open items |
|---|---|---|
| 1. No per-task code, prompts, thresholds or inputs; general mechanisms only | Structurally met: CLI is pointers only; noun lint (`test_no_task_words.py`); task as data; leakage test | MF-7; SF-4 (historical scenes, calibration on 4 scene types); SF-17 |
| 2. Prefer removing vetoes; derive parameters from measurement | Mostly met: lattice, stall constants, rotation envelope, start refusal and the 4-box cap are all removed | MF-5 (new vetoes); SF-12 (β, k, look trigger, wall budget) |
| 3. Acceptance = all-task sweep on one frozen build, coverage map | Met: S5 on frozen `uc72-b005`, digest checked per episode, glyph coverage map | SF-4 (held-out conditions); SF-5 (test) |
| 4. Live-first; the user starts full tasks and sweeps | Mostly met: offline checks labelled development aids, the user starts sweeps | SF-13 (M0 offline gating, merge gate, near-full smokes, GPU service) |
| 5. Subscription only | Met: launcher pops `OPENAI_API_KEY`; GPT-6 Astra, high reasoning; canary at high reasoning; no fallback | SF-1 (one store); SF-13 (shared load) |
| 6. Do not modify H1, main-tree tracked files or foundation_harness | Met: local clone, tags, worktrees, vendoring with hash pins, import guard | SF-3 (shared venvs, editable SAM/DA3 sources); SF-2 (no backup) |

## Evaluation validity checklist

| Requirement | Plan | Assessment |
|---|---|---|
| Honest finish scoring | `finish` + `give_up`; strict truth after a 10-step hold with the last gripper command; counterfactual only as a diagnostic; oracle cues deleted | Sound; SF-14 (remaining `task_success` keys) |
| Contact episodes, not samples | `contacts.jsonl`, episodes by subclass and phase, 6-action merge gap, rescoring checks against verified counts | Sound; SF-11 (self-declared `intended`, unjustified palm rule) |
| Same-window interleaving | (task, repeat) blocks, ABBA with a seed, canary per block, quota pauses all arms, `service_speed_index` | Sound; SF-1 (auth infra could correlate with the pristine arm) |
| Power | 34 × 3 for the S5 A/B; differences under 30 pp "unresolved" at one repeat | SF-5: power model assumes homogeneous effects; McNemar on clustered pairs |
| Censoring | caps scored and censored; P(≤ t) reported | SF-5 (survivorship in the time ratio); SF-6 (`cap_wall`, pristine caps) |
| Infrastructure failures | one relaunch, both runs kept, `I` after a second failure, sweep pause on auth | SF-6 (policy-caused sim crashes, async waits); MF-3 (outage episodes) |
| Not gameable by oracle cues | leakage test, evaluator-only task fields, truth unimportable by policy code | Sound for oracle cues; SF-11 (self-declared contact exemptions); SF-4 (no held-out conditions) |

## Coverage of report problems

| Report item | Status | Note |
|---|---|---|
| H1 #1, #2, #9, #14, #16, #17, #18; R4, R6, R8 | covered | R7/R8/R9 (M0); PC5 + EXC-D1b; GT3 + PS; PC3a; PC2; PC1 + R4; R3/R11/S1 |
| H1 #8 (router checks one point) | partially covered | M4 body planner; containers and fixtures not representable (MF-4); gripper close/open class handled by a veto (MF-5) |
| H1 #10 (palm envelope) | partially covered | GT8 in M2; SP8 report → active only if σ < pad height (UD-34); GT8 overclaims checking (MF-2) |
| H1 #11 / R5 (contact sensing) | partially covered | guard + `stop_on` in M2; a null window turns any contact into a declared one (MF-6) |
| H1 #12 (held object, rotations, lattice) | partially covered | lattice removed and rotation sequenced in M2; M3 drops the carried-object allowance before SP10 (MF-2) |
| H1 #20 / R1 / K5 (metric grounding) | partially covered | references, `locate` and regions in M3; raw targets and unbounded offsets kept (SF-9) |
| R2 (partial robot model) | partially covered | body sweep in M4; filled fixtures (MF-4); `inside_robot` exclusion (MF-1) |
| R9 (simulator-only privileges) | partially covered | calibration-1 and virtual clock in place; every sweep uses exact MuJoCo calibration; real-robot maximum age deferred with reason |
| H3 V1–V7, V9 | covered | PS-1…PS-12, GT1, INT-5 |
| H3 V8 (fail-closed) | partially covered | fail-open adopted; no safe degraded motion mode (MF-3) |
| K3 (contact trap) | partially covered | SP6 monotone clearance in M4; the M2–M3 point escape lacks the last-anchor rule (SF-7) |
| K2/F1, K6, F3, F5, F8, K1, K8 | covered | GT3 + PS-9 + SP10; SP3; PS-3; SP5 + PS-7; PS-7 + PC3b; PS-11; PC1/R4 |
| H1 #5 (slow primitives) | partially covered | approaches sped up only in M5; contact-profile lifts and retreats stay at 1.1–1.3 mm/action; rotation-cap raise deferred with reason |
| §6 item 2 (contact taxonomy) | partially covered | evaluator classes exist; `intended` comes from policy declarations (SF-11) |
| §6 items 4 and 6 (repeats, decision rule) | partially covered | no non-inferiority margins, homogeneous-effect power, "repeat" undefined (SF-4, SF-5) |
| H2 §4.4 items 3 and 9; plan fold; K2 launch | covered (deferred with reason) | backlog `:1956-1975` |

## User decisions

**Missing.**
- Off-box backup (SF-2).
- Credential-store sharing (SF-1).
- Subscription load policy alongside the user's own agents (SF-13).
- Which runs Claude may start, and whether Claude may start the GPU service (SF-13).
- Held-out S5 conditions and the definition of a repeat (SF-4).
- Container and fixture representation, plus link permissions (MF-4).
- Degraded mode on perception outage (MF-3).
- Non-inferiority margins and the statistical test (SF-5).
- Which calls count toward the cap (SF-6).

**Recommendations to revise or justify.**
- UD-06 (i): no reason given (SF-11).
- UD-27: conflicts with the approved direction (SF-9).
- UD-28: k should come from GT11 coverage (SF-12).
- UD-29: dispute, not remove (SF-10).
- UD-33: β provenance mislabelled (SF-12).
- UD-36: truncation is a veto (MF-5).
- UD-39: evidence base (MF-7).
- UD-20/21/26: two-scene A/Bs (SF-17).
- UD-23: trigger value (SF-12).
- UD-41: depends on UD-34 through INV-1 (SF-12).
- UD-12: async-mode quota handling (SF-6).
- UD-16: add the escape constraints (SF-7).

## Verified correct (no action)

- **Line references.** These match H1 `3cc84ac`:
  - fragment caps 6/120/240 and stall constants (`fragments.py:23-46`);
  - envelope veto (`:288-291`) and `contact_or_collision_cause: "unknown"` (`:482`);
  - the `obstacle_map` duty and "remove the object you will manipulate" (`agent_client.py:92-105`);
  - the test-lift mandate (`:161-163`), "Simulation pauses" (`:182-183`), and the oracle cues plus "no give_up" (`:205-209`).
- **Pristine arm.** H1's launcher has `--no-video` and an operator stop file (`launch.py:83`, `:135-159`), so the pristine arm can be capped without H1 code changes. It sets `MUJOCO_EGL_DEVICE_ID=--gpu` and `LIBERO_CONFIG_PATH` per root (`:97-103`).
- **No silent H1 code.** No editable install in H1's `.venv` points at H1 `src`. `PYTHONPATH=src` makes each worktree run its own code.
- **Vendoring.** The vendored foundation_vision modules have no absolute `gpt_field` imports, so the import guard can stop live-tree imports.
- **Cruise cap.** SP9's cruise cap is clipped to the configured 0.10 m, so M4 slows down near objects and never exceeds H1's top speed.
