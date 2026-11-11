# Cross-report critic (W2, key `cross_critic`)

Date: 2026-11-08. This was a read-only analysis.

- I read all eight W1 reports in full.
- I re-checked every resolution below against the saved run artifacts, using small offline scripts (numpy only, run with `python -I -B`).
- I started no simulator, made no model or Astra call and used no GPU.
- Nothing under `/workspace/gpt_field` was written. My scripts never wrote into the W1 folders either; I checked that the one `.pyc` in `w1/scripts/__pycache__` predates this work.

Paths:
- Scripts: `/tmp/claude-0/-workspace/ba83318d-1178-4524-a098-31607ddcba01/scratchpad/harness_analysis/w2/cross_critic_work/scripts/`
- Outputs: `…/w2/cross_critic_work/data/`
- Runs: under `/workspace/gpt_field/runs/`
- Code: `H1/` = `/workspace/gpt_field/agent_speedup_v2_obstacle_harness_100mm/src/gpt_field/`; `K2/` = the hash-pinned keypose-2 snapshot.

Short names:
- **H1**: uc7.1 with the 0.10 m transit cap.
- **H3**: foundation_vision.
- **H2-v1**: the run that actually happened. **K2**: the keypose-2 snapshot.
- **T20**: uc7.1 with the 20 mm cap.
- **AST**: the astranomy direct baseline.

## Bottom line

1. **H3's roughly 2× per-call latency is mostly the service, not the vision context.**
   - A latency model fitted on 164 same-code calls from 11-02/03 predicts H3's own token counts at 253.8 s (T04) and 250.5 s (T08). The observed totals were 311.1 s and 340.8 s.
   - Visible decode speed does not depend on prompt content, and it was 31.8–36.4 tok/s against the 46–52 tok/s predicted for that context size and time of day.
   - What remains is that H3 used 2× the reasoning tokens. Its cause is unknown, and H3's added instruction text is an untested confound.
2. **Input size does cost latency.** At fixed reasoning and visible tokens it costs **19 ± 3 ms per 1k tokens** (t ≈ 6.6). The "R² = 0.009, no effect" result comes from a single-variable fit that leaves out the dominant variable.
3. **H3's better obstacle outcomes do not come from usable vision geometry; it produced 0 usable centres. They are not shown to be unrelated to the H3 arm either.**
   - H3's first obstacle boxes covered the true obstacle fully with clearance in 2/2 runs; the H1-family first boxes did in 0/4.
   - The H1-family box errors reproduce within about 1 cm across runs, so they are bias, not sampling noise.
   - The likely active ingredients are two things: the 2D box of the operator-named obstacle, which is a privileged input, and H3's extra instructions about `obstacle_map`.
4. **Three proposed speedups collide measurably with collision avoidance.**
   - A "fast until the last 2 cm" contact approach would have met **3 of 15** observed contact-approach onsets at transit speed (onsets at 2.2–6.8 cm before the endpoint), plus 5 more at 1.7–2.2 cm.
   - Removing the 1.2 rad rotation veto alone would not have prevented the H1 T08 finger–bottle contact. The cause there is OSC rotation lag during a combined translate-and-rotate stage.
   - The 0.10 m cap never pushed the grip site past the clearance in use (0/104 transit stages). It did raise top speed from 0.12 to 0.53 m/s, and the only robot–obstacle contact began at 0.17–0.19 m/s.
5. **Async or committed verification is safer for grasps than h1_speed feared.**
   - Both observed object-grasp failures are visible in the finger gap alone: 1.4 mm settled in one, and 13.3 mm collapsing by 12.3 mm during the lift in the other.
   - All 19 visually confirmed object grasps settled at 9.3–21.5 mm and lost at most 4.4 mm during the lift.
6. **Astranomy's 63–87 calls are fixed by its tool contract.** Each call allows at most 0.05 m and 5 steps, and the runs averaged 4.7–4.9 actions per call.
   - It hit the can in 3/3 runs because it grasped the white mug on the can-facing side and has no held-object model.
   - Runs 001 and 002 struck the identical can-rim point (−0.048, −0.195, 0.512) with the same mug geom.
   - It has no done tool, so the collision-avoidance run (CA) re-issued a near-identical hold target for 12 calls until the user stopped it.
7. **Simulator-only privileges are load-bearing.**
   - H1's camera validator (shared by H3) rejects any calibration other than an ideal centred pinhole derived from MuJoCo fovy (`H1/astranomy_agentic_geometry.py:105-111`).
   - Success relies on evaluator oracles.
   - H1/H3 policy compute is 1.2–1.7 ms per step, which is real-time capable; the rest of the 0.16–0.40 s/step is simulator and evaluator work. H2's local compute is 3.7 s per action, which is not real-time capable.

---

## 1. Contradictions and how I resolved them

### C1. Why H3's per-call latency was about 2× H1's
- **cross_eval_methodology** (summary 6, B5): "H3's vision context roughly doubled per-call latency"; "consistent with 2× reasoning tokens".
- **h3_live_integration** (F3): about +59 s came from a slower service day (decode 31–39 against 48–57 tok/s, and the vision-free plan request was also slower on 11-05), and about +42 s from 2× reasoning tokens.

**My check** (`scripts/calls_all.py` → `data/calls_all.json`, 390 calls; `scripts/latency_day.py`, `scripts/latency_resid.py`):

| run (UTC window) | median visible decode tok/s | exec latency mean s | reasoning per call (mean / later-call median) |
|---|---|---|---|
| T20 official, 11-02 21:56 | 54.0 | 19.6 | 190 / 115 |
| T20 T04, 22:53 | 51.1 | 17.7 | 129 / 67 |
| H1 T04, 23:40 | 61.1 | 15.3 | 123 / 54 |
| H1 T08, 11-03 00:42 | 53.4 | 15.9 | 142 / 53 |
| H1 stove-off, 07:49 | 51.4 | 16.2 | 111 / 64 |
| H1 drawer 001, 10:34 | 47.6 | 18.4 | 151 / 93 |
| H1 drawer 003, 11:33 | 46.5 | 22.9 | 219 / 158 |
| **H3 T04, 11-05 12:53** | **36.4** | 28.3 | 259 / 183 |
| **H3 T08, 13:18** | **31.8** | 31.0 | 254 / 110 |
| H2 T04, 11-07 23:15 | 55.3 | 10.3 | 28 / – |

- **Residual check.** A model fitted on the 11-02/03 calls only is `L = 6.89 + 0.0510·reasoning + 0.0071·visible + 0.0192·input_k`.
  - With H3's own token counts it predicts 253.8 s (T04) and 250.5 s (T08), against 311.1 s and 340.8 s observed. The residuals are **+57.3 s and +90.3 s** (+5.2 and +8.2 s per call).
  - The same model gives H1 residuals of −9.3 s and −24.4 s. This reproduces h3_live's +59 / +92 s.
- **Decode speed is the vision-independent measure.**
  - Within 11-02/03 it is flat in context size (−0.01 tok/s per 1k tokens) and falls 0.57 ± 0.15 tok/s per hour after 21:00 UTC.
  - Extrapolated to H3's hour it predicts about 46.5 tok/s, against 31.8–36.4 observed.
  - H2 on 11-07 fits the trend: 54.2 predicted, 55.3 observed.
- **Identical plan requests also decoded slower in the H3 window.**
  - T04 plan: 74.1 and 89.8 tok/s on 11-02, against 61.6 for H3.
  - T08 plan: 82.8 against 63.7.
  - h3_live's "84–89 tok/s on 11-03" quotes only the two fastest plans. The 11-03 07:48 stove-off plan decoded at 62.6 tok/s, as slowly as H3's.
- **It is a service window, not a calendar day.** The AST collision-avoidance run on the same calendar day (11-05 02:51–03:10 UTC) was not slow: the day term in its own fit is +0.58 ± 0.38 s, not significant.

**Resolution.** h3_live's decomposition is supported; cross_eval's causal claim is not.
- About 60–90 s of H3's extra model time is a slower service window, which shows up as a per-call fixed penalty of +5–8 s plus slower decode.
- The 2× reasoning per call is real but unexplained. Later-call median reasoning in H3 T04 was 182.5 tokens against 53.5 in H1 T04, but H1 drawer 003 reached 158 without any vision.
- An untested confound: H3's +1.4k characters of instructions explicitly ask the model to use evidence for its own `obstacle_map` and to treat unknown space as not clear (`/workspace/gpt_field/src/gpt_field/foundation_vision/live.py:34-55`, line 49). That can induce reasoning on its own, without any perception.

### C2. Does input or context size affect latency?
- **cross_eval:** `21.8 − 0.018 s/1k`, R² = 0.009, "does not depend on input or history tokens".
- **h2_planning:** r = 0.20, "latency is driven by output length".
- **h1_speed:** 0.0179 s per 1k tokens in a multivariate fit.
- **h3_live:** time to first token (TTFT) ≈ 3.89 + 0.049·reasoning + 0.0196 s/1k input.

**My check** (n = 186 uc7.1 + H3 calls):

| fit | input coefficient |
|---|---|
| single variable | −0.016 ± 0.014 s/1k (R² 0.008) |
| + reasoning + visible | +0.0121 ± 0.0043 |
| + an 11-05 term | +0.0178 ± 0.0036 |
| 11-02/03 calls only (n = 164) | **+0.0192 ± 0.0029** (R² 0.939) |
| H2 calls (n = 12) | +0.028 ± 0.007 s/1k (R² 0.93) |

Input size is nearly uncorrelated with reasoning tokens, the dominant driver (r = −0.16), so a single-variable fit cannot see the effect.

**Resolution.** The multivariate estimates agree on about 18–20 ms per 1k input tokens.
- That is +1.7 s per call at the 87k tokens of H1 T04's last call, and +5 s at drawer's 260k.
- In total it is about 6 % of T04 model time and about 14 % of drawer 001's.
- History trimming is therefore a real lever, though a secondary one; cross_eval's ranking stands but its premise does not.
- Astranomy shows a larger input cost (0.069 s/1k) because its input is mostly retained images; see G8.

### C3. Why H1 T04 had 0 can contacts
- **h1_collision:** the model carried high (z 0.62–0.64); routing did not cause it.
- **cross_eval:** the router never detoured.
- The brief asked whether routing deserves credit.

**My check** (`scripts/dump_exec.py`):
- Call 5 commanded a lift to z 0.64 and a carry at 0.62, with clearance 0.06, which hit `route_start_blocked`.
- Call 6 resent **identical waypoints** with only the clearance changed (0.04).
- The model box top was 0.50; plus the 0.04 clearance that gives 0.54, so the router accepts any grip-site carry above 0.54 m. The held mug's base is about 6 cm below the grip and the can top is at 0.513 m, so a safe carry needs a grip height of about 0.57 m or more.

**Resolution.** The model's altitude explains it, as h1_collision said.
- The router's only effect in T04 was one wasted call (10.6 s).
- The router would have approved a colliding carry between 0.54 and 0.57 m, consistent with h1_collision's offline replay (F7).

### C4. How low was the first T08 bottle box?
- **h3_live** F2: "6.6 cm too low".
- **h1_collision:** about 100 mm low.

**My check** (`scripts/box_truth.py`): against layout truth, the box **top** was 6.6 cm below the bottle top, the **bottom** 10.0 cm below the table, and the centre was off by (−8.5, −0.9, −8.3) cm.

**Resolution.** Both are right, measured against different reference surfaces. For an approach from above the top is what matters. The expanded top was at 0.998 m against a bottle top of 1.046 m, a 4.8 cm shortfall.

### C5. What caused the H1 T08 finger–bottle contact (3,504 samples)?
- **h1_collision** F6: "the split (45° and then 90°) put open finger1 over the bottle's top face" (the envelope veto).
- **h3_live** F10: the "approach at grip z = 1.035 was below the bottle top at 1.046".

**My check** (trace steps 28–45):
- Steps 28–34: the grip site rotated in place at z 1.077 (0.1 mm/step).
- Step 35: it began descending at 8.5–9.3 mm/step (0.17–0.19 m/s) toward 1.035. Contact started at step 37 at z 1.062, then it pinned at 1.051 while the jaw axis finished turning, (0.72, −0.69) → (1.0, −0.01) per h1_collision.
- Realized rotation is about 0.023 rad/step, so 90° needs about 68 steps while the translation needs about 15.

**Resolution.** Both reports are partly right.
- The mechanism is the **rotation lag inside a combined translate-and-rotate stage**, ending at a target 1.1 cm below the bottle top and 3 cm from its edge.
- Astra chose that target because its box put the bottle 8.5 cm further in −x and 6.6 cm lower than it was.
- An unsplit 90° stage would lag the same way, so removing the veto would not have prevented the contact (see IC3).
- H3 avoided it: it hovered at 1.075 m (its box top 1.025 plus 0.025 clearance) and descended only in the next call, after the rotation had finished.

### C6. How much does removing the vetoes save?
- **h1_zero_shot** F6: −52 s for the H1 T04 reference (2/13 calls, 52.1 s).
- **h1_collision** F6: "about 30 s of calls each".
- **h1_speed** F5 / **cross_eval** §3: T04 −2 calls, −18.2 / 18.5 s.

**My check:**

| run | vetoed calls | repair calls (reasoning tokens) |
|---|---|---|
| H1 T04 | 32.3 s, 19.5 s | 7.6 s (0), 10.6 s (16) |
| H1 T08 | 30.8 s, 38.1 s | 7.1 s (0), 12.1 s (0) |

**Resolution.** h1_speed and cross_eval are right.
- Without the veto the vetoed answer would have executed, so the saving is the repair call plus one hold step: about 18–19 s per run.
- h1_zero_shot and h1_collision count the vetoed call itself, which is still needed. They overstate the saving by about 1.5–3×.

### C7. Model share of time
- **cross_eval:** 71 % for H1 T04 (control + plan, excluding video).
- **h1_speed:** 54 % (process wall, including inline video and two simulator starts).

**Resolution.** Both are right with different denominators. I add the share relevant to a real robot: model / (model + motion + policy compute + observe) = 217.8 / (217.8 + 27.45 + 0.65 + 3.98) = **87 %**.
- The simulator render and PNG step (124.6 ms/step), `run_state` (23.7 ms/step) and video (123.8 ms/step) are evaluator artefacts.
- H1's own policy compute is 1.18 ms/step (`trial/summary.json` `timing_components`).

### C8. What causes the stalls
- **h1_zero_shot** F3: 25 of 46 stall calls are "height/depth misestimate", by belief keywords. It proposes a `locate` tool.
- **h1_speed** F3/F4: of 41 stalls, 17 are transit-overshoot false stalls, 19 contact stops and 5 genuine. It proposes controller and stall-predicate fixes.
- **h1_collision** F5: 14 forbidden contact, 8 allowed contact, 10 near the target, 14 no contact and far. It proposes guarded motion.

**My check** (`scripts/stall_join.py`, joining on run and step; h1_collision's script re-run read-only → `data/stall_causes_out.txt`):
- 11 of h1_speed's 15 overshoot false stalls are h1_collision's "no contact, far": the rebound leaves them more than 10 mm out.
- h1_speed's 19 contact stalls split into 8 forbidden, 4 allowed, 6 near the target and 1 far.
- A re-implementation of the belief-keyword rule gives 9 height/depth stalls instead of 25. The keyword attribution is unstable, and the model misreads overshoot stalls as "poor tracking" (drawer 003 call 27).

**Resolution.** The trace- and contact-based taxonomies are consistent with each other; the keyword split is weak evidence.
- About 37–41 % of stalls are a controller artefact: 15 overshoot false stalls in `w1/data/stall_class.json`, 17 counting T04's two near-stalls the way h1_speed's text does.
- About 46 % involve contact: of the 41 stalls h1_speed classified, 19 are contact stalls, 8 of them forbidden.
- Geometry error is a plausible upstream cause of over-deep contact targets (see G9), but not of the overshoot stalls.

### C9. Are prompt quaternions sign-normalized?
- **h1_speed** §5: harmless because "control and prompts use quaternions with sign normalization (`action.py:84-86`)".
- **h1_zero_shot** §4 and **h2_planning** F14: `robot_pose_wxyz` flips sign between requests.

**My check:**
- `H1/action.py:85` normalizes only the control error (`if error[3] < 0`).
- `H1/astranomy_agentic_motion.py:89-97`, `measured_pose`, emits the rotation-vector quaternion with no sign convention.
- Counted flips between consecutive requests: 3 in 13 (H1 T04) and 1 in 12 (H2).

**Resolution.** h1_zero_shot and h2_planning are right. The flip is harmless for H1 control but matters for K2, where Astra writes absolute quaternions.

### C10. Which DA3 correction works
- **h2_perception** F1: a per-view affine alignment (wrist → external) fixes the wrist table error from 6.6–7.1 cm to 1.6 cm and raises cross-view support.
- **h3_vision_engine** F4: a per-view self-calibrated scale fixes the table but moves the can 3–5 cm; "a single scale per view cannot fit both".

**My check** (`scripts/affine_can.py` → `data/affine_can.json`): I ray-cast the true can and table on H2 T04 frames 5, 10 and 19. Diagnostic truth only.

| view and transform | can depth error | table error |
|---|---|---|
| external, shipped | +0.64 – 0.80 cm | +3.3 – 3.5 % |
| wrist, shipped | +0.70 – 1.01 cm | +18.1 – 18.7 % |
| wrist after h2's affine (scale ≈ 0.54, offset ≈ 0.13 m) | +0.90 – 1.32 cm | +3.3 – 3.5 % |

**Resolution.** Both are right for their own transform.
- The affine with the external view as reference keeps object depth at about 1 cm. A pure per-view scale is what distorts objects.
- Neither removes the external reference's own +3.3 % background bias, nor the decimetre-level robot-body depth errors (h2_perception F4).
- At the shipped scale DA3's object depth was already about 1 cm in both views. H3's 0 usable centres and H2's blockers came from gates, association and self-geometry, not from object depth.

### C11. Are H3's fewer contacts and calls attributable to vision?
- **cross_eval** (§4) and **h3_live** (F10): not attributable (0 usable centres); "may be chance".
- **h1_collision** F3: H3 T04 had no palm contacts because it grasped about 15 mm shallower.

**My check** (`scripts/box_truth.py`, `scripts/bbox_channel.py`). First-box coverage of the true obstacle, raw / with clearance:

| scene | T20 | H1 | H1 stove-off | H3 |
|---|---|---|---|---|
| T04 can | 0.34 / 0.77 | 0.30 / 0.77 | – | **0.83 / 1.0** |
| T08 bottle | – | 0.00 / 0.00 | 0.00 / 0.10 | **0.51 / 1.0** |

- The H1-family first boxes reproduce each other: T04 x error +4.8 cm in both T20 and H1; T08 errors of (−8.5, −8.3) cm and (−7.7, −9.3) cm.
- In T04, H3's box centre projects 11 px from the SAM box of `can`; the H1 and T20 boxes sit 37–40 px lower, at the can's base row.
- In T08, H1 already had the right pixel. H3's gain there was depth along the camera ray, which a 2D box cannot supply.
- H3's first two calls in both scenes were "hover high, rotate later" moves. Of H1 T08's 10 extra calls, 4–5 trace to the veto plus the bottle-contact chain (calls 2–5, plus the call-6 re-approach) and 2 to an unrelated missed grasp (calls 8–9).

**Resolution.** It is not attributable to vision geometry, but the H3 arm is a plausible cause, through two untested channels:
- the operator-named obstacle's 2D box, which is privileged;
- the instruction text.

The T04 palm-contact difference is grasp depth, unrelated to vision. An instruction-only ablation (H1 plus H3's instruction text, no vision) is the missing test.

### C12. Astranomy's "per-call floor of 5–10 s on the subscription transport" (cross_eval §2)
**My check:**
- In astranomy, the time outside the server's created → completed window grows 0.11 s per MB of request (R² 0.59).
- Requests reach 35–48 MB at 126–174 retained images.
- A 512² image costs about 893 input tokens (fit R² = 1.0).
- H2's tiny-reasoning calls fit an intercept of 2.3–3.2 s.

**Resolution.** Part of astranomy's floor is self-inflicted by its unbounded image history, not a property of the transport. The transport floor itself is about 3–5 s plus upload time.

### Count reconciliation (definitions, not errors)

| quantity | values in reports | why they differ | value to use |
|---|---|---|---|
| H1 T04 calls | 13 (cross, speed, zero) vs 14 (collision) | the plan call included | 1 plan + 13 exec |
| stall-triggered calls | 41 (speed, 5 runs) / 46 (zero, collision, 7 runs) / 51 of 186 (cross, 9 runs) | run sets | per set |
| vetoes | 6 zero-motion (cross) / 7 (speed) / 9 of 164 (zero) / 7 route + 2 envelope (collision, with H3) | zero-motion vs any veto; whether H3 is included | — |
| drawer 001 recovery streak | 9 calls, 141–143 s (speed, collision) vs 11 calls, 169 s (zero) | calls 26–27 are successful completions of the recovery | 9 failure-triggered (142.7 s) + 2 |
| per-step non-model time, H1 | 0.158 s (cross, speed, live) vs 0.133 s (h2_planning) | h2_planning excludes `run_state` (23.7 ms) | 0.158 s; policy alone 1.2 ms |
| drawer 001 episodes | 10 (cross text, collision) vs 9/7 (cross `normalised_table.md`) | pair vs class; merge gap | 10 pair-episodes |
| cached share | "0–2.5 % in most runs" (cross) | astranomy left out | AST 25 / 37 / 42 % |
| vision tokens per call | 2.3–2.5k (live) / 2.9k (engine) / ~3k (cross) | summary size vs per-call growth | summary 2.3–2.5k; growth +2.9k |
| strict success | 0/10 and 0/11 (cross, internally) / 0/5 (zero) | the AST CA run | 0/10 strict at terminal |
| "bottle tilted" | h3_live F2 | it is Astra's belief (H1 T08 call 3), not a measured pose | visual claim only, like "can tipped" |

---

## 2. Improvement conflicts and sequencing

**IC1. Faster contact approaches vs collision.** Proposals: h1_speed F6 (fast until about 2 cm before contact), h1_collision F10 (guarded fast approach to a standoff), h2_planning F8c (contact speed only for the last 2–3 cm).
- Evidence (`scripts/contact_onset.py`): of 15 forbidden-contact onsets inside contact-profile approaches, 3 began 2.2–6.8 cm before the commanded endpoint (T08 at action 540, drawer 001 at 148, drawer 003 at 1732). Five more began at 1.7–2.2 cm.
- Model target errors before close-up views are 2.4–5.7 cm (h1_zero_shot F3). Approach segments are only 4.2–7.5 cm long, so a safe slow zone would remove most of the saving.
- **Sequence:**
  1. guarded translation-progress stops (h1_collision F5) and a palm/fingertip envelope depth bound (h1_collision F3);
  2. measured geometry (C10, G4);
  3. only then distance-scheduled speed, with the slow zone set to measured uncertainty plus braking distance.

  Validate with the onset-distance statistics.

**IC2. A 0.10 m cruise and a higher rotation cap vs collision.** Proposals: h1_speed F3 and F6.
- The p99 step is 25–26 mm (0.53 m/s), against 5.4 mm (0.12 m/s) at 20 mm. The T08 contact began at 0.17–0.19 m/s, above the 20 mm ceiling.
- Grip-site overshoot never exceeded the clearance in use (0/104 stages; `scripts/overshoot.py`). The risk is impact speed at hand, finger and held-object contacts that the router does not model.
- **Sequence:** braking-aware tracking first. Then allow cruise speed only on stages with body-swept clearance, with speed scaled to the measured swept clearance. That is a mechanism, not a per-task threshold.

**IC3. Removing the rotation-envelope veto vs finger sweeps.** Proposals: h1_speed F5, h1_zero_shot F6, h1_collision F6.
- An unsplit 90° stage lags about 68 steps of realized rotation against about 15 of translation, so the fingers sweep at the target, as in C5.
- **Replace the veto with executor sequencing:** rotate in place at the stage start or at a cleared altitude, then translate. Include the realized orientation profile in the swept check. This removes the zero-motion veto without removing safety.

**IC4. Turning `route_start_blocked` into an escape vs pushing into neighbours.** Proposals: h1_speed F5.2, h1_zero_shot F6, h1_collision F4.
- In all replayed refusals the start was outside the true expanded box, by 6–20 mm (h1_collision F4). The beliefs say "vertical space ... clear".
- **Sequence:** exact point-to-box distances first, which removes the spurious refusals. Then a vertical-first escape, checked by the body sweep and reported.

**IC5. Committed pick-and-place with async verification vs undetected failures.** Proposal: h1_speed F2.
- Grasp failures are gap-detectable (bottom line 5; `scripts/grasp_gap.py`).
- Contacts are detected only via stalls: median 13 steps, and the T08 press lasted 47 steps. A committed carry would push through a contact longer.
- **Sequence:**
  1. guarded motion and gap/lift continuation predicates, with the expected gap band stated by the model;
  2. commit sequences;
  3. keep blocking visual checks only for placement relations (on/in) and articulated engagement;
  4. count late-detected failures in the sweep.

**IC6. Finish-scoring, `give_up`, and deleting the "episode continues" cue vs headline speed and success.** Proposals: h1_zero_shot F1, h1_speed F9, cross_eval M4.
- It adds 1–2 calls (+20–40 s) per success, and T20 T04 becomes a failure at its own `finish`.
- **Do it first.** Every later speedup must be measured against finish-scored baselines, otherwise oracle-dependent shortcuts get credit.

**IC7. Folding the plan call into the first executive call vs H3 and H2 target bootstrapping.** Proposals: h1_zero_shot F2c, h1_speed F8.
- H3 builds its SAM prompts from plan parts and disambiguates with the plan's cells (h3_live F7). H2 bootstraps queries from the sentence.
- **Requirement:** the merged first call must emit object noun phrases plus external-image cells or points per task object. Perception keys off that output.

**IC8. Harness-proposed candidates vs Astra-owned metric keyposes (K2).** Proposals: h2_planning F9 and F3, h2_perception F6/F7, h3 set-of-marks.
- K2 asks Astra for up to 12 absolute 7-float poses per operation (`K2/src/gpt_field/foundation_harness/client.py:89-103`).
- Model metric estimates are biased and reproducible (C11, G4), so repeated sampling does not fix them.
- **Resolution:** Astra owns the operation, sequence, contact intent and identity. The harness grounds the metric pose (snaps to a measured top or support patch, or proposes grasp and place candidates), and Astra selects or nudges.
- Run K2 as-is only as a baseline, after cross_eval's S0 offline feasibility gate.

**IC9. Overlays, numbered marks and extra images vs latency, context and caching.** Proposals: h1_collision F2, h2_planning F3/F9, h3_vision F1 vs h1_speed F7, h3_live F4, h2_planning F10.
- An image costs about 893 tokens, about 17 ms of prefill. The real costs are unbounded histories (astranomy: +0.11 s per MB, 48 MB requests) and the sliding window, which rewrites the prefix so caching is 0–2.5 %.
- **Resolution:**
  - Draw overlays onto the current pair; replace, do not add.
  - Keep history append-only, stripping images at insertion time.
  - Put static robot and camera text in a cacheable prefix.
  - A/B test reasoning tokens, since H3's extra context coincided with 2× reasoning.

**IC10. Images only at decision steps vs perception tracking and async checks.** Proposals: h1_speed F1.2 vs h3_vision F8, h2_perception F10, h1_speed F2.
- **Resolution:** render raw frames at a low fixed rate into shared memory, PNG-encode only frames sent to the model, and let perception subscribe to the raw stream.

**IC11. K2 veto removals vs collision.** K2 removals: `allow_unknown_space=True`, uncertain points traversable, link0 skipped. Concerns: h2_perception F5/F8, h2_planning F7.
- The can is only 57–72 % "strong"; its occluded back half becomes traversable.
- **Sequence:** persistent occupancy with shadow-volume completion and robot-body carving, then unknown space as a cost with slow speed. Only then lift the veto.

**IC12. Body-swept checks vs the "remove vetoes" rule and few attempts.** Proposal: h1_collision F1.
- With the model's own boxes, a hand sweep refuses 21/110 accepted stages, 14 of them at the start, which would add about 3 calls per run.
- **Resolution:** build it only as a planner (escape, detour, goal-as-contact handoff), and only together with better geometry.

**IC13. Automatic perception boxes vs route refusals.** Proposals: h3_live F2, h3_vision F6.
- More boxes near the gripper mean more start and goal refusals.
- **Sequence:** exclude held, target and robot geometry, add escape semantics, then add automatic boxes, then the body sweep.

**IC14. Async perception vs freshness on a real robot.** Proposals: h3_live F3/F6, h2_perception F10.
- Step-indexed validity is safe only in a paused simulator. Real model calls take 13–79 s, so a snapshot taken at request time is stale at execution time.
- **Resolution:** re-localize in the executor at the moment of use, and stamp capture time against action start.

**IC15. History compaction vs evidence loss.** Proposals: h1_speed F7, h1_zero_shot F12.
- Compaction already hid the `route_start_blocked` sub-reason (T08 call 15).
- **Resolution:** compact into structured outcome records that keep reasons verbatim, and trim static boilerplate first.

**Recommended order, on one frozen build, followed by the all-task sweep:**
1. **Measurement.** Finish-scoring, `give_up`, an evaluation cap, per-call throughput logging, same-window interleaved arms, contact-episode metrics, offline replay gates (H1 deterministic replay; H2/H3 saved frames).
2. **Remove spurious vetoes safely.** Exact box distances; a checked vertical-first escape; executor rotation sequencing; removal of macro-only plan validation; 401 refresh and 5xx retry; vision failing open.
3. **Sensing.** Guarded motion with "until contact"; the palm envelope; gap and lift predicates.
4. **Geometry out of the LLM.** Mask triangulation, or one view plus a support plane; automatic boxes; harness-proposed candidates; a body and held-object sweep used as a planner.
5. **Speed.** Committed sequences with async checks; evaluator overhead moved off the loop; distance-scheduled approach; braking-aware cruise; append-only cacheable context.

---

## 3. Gaps none of the reports covered (investigated offline)

**G1. Astranomy's own failure modes** (high; goals: few attempts, collision).
- **Call count.** The prompt allows at most 0.05 m and 5 steps per call, with no done or give_up tool (`/workspace/gpt_field/src/gpt_field/astranomy_agentic_prompt.py:27,31`). Each 5-step call ends about 23 mm short of its target (`runner.log` `position_error_m`). The runs executed 161, 306 and 409 actions over 34, 63 and 87 calls.
- **Contacts.**
  - 001: action 45, open fingers descending at (−0.058, −0.171, 0.55) between the mug and the can.
  - 002: action 66, during close and lift.
  - Both hit the can rim at **(−0.048, −0.195, 0.512)** with mug geom g8.
  - CA: action 126, placement descent at y −0.36; the hanging mug's far edge at y −0.24 hit the can top.
- **Prompt suffix.** The CA wording raised the carry (0.63–0.64 m against 0.58–0.60 m) but did not change the grasp side.
- **Hold loop.** CA calls 76–87 were 12 near-identical hold targets (calls 77–87 exactly identical; about 170 s) until the user stopped it.
- **Context and tokens.** It retains every image (174 images, 48 MB) and uses 3.7–7.0 M input tokens per episode, against 0.65 M for H1 T04.

**G2. Service throughput as a confound for "fast"** (high; goal: fast).
- Decode speed ranged 31.8–61.1 tok/s across runs, with a −0.57 tok/s per hour trend in UTC plus an extra-slow 11-05 window (C1).
- There were no 429s and no overload or transient retries (`history_stats` all 0). The only infrastructure losses were AST 001 (503 `unknown_error`, no retry) and drawer 002 (401 `token_expired`).
- At H3-window speed, H1 T04's 14 calls would take about 88 s longer; h1_speed's "about 172 s after F1+F2" becomes about 235 s.
- **Fix:** log tok/s, interleave arms in the same window, and normalise or censor by service speed.

**G3. Simulator-only privileges** (high; goals: zero-shot, fast).
- **Ideal-camera validator.** `validate_geometry_metadata` requires frame `simulation_world`, square images, focal length = H/(2·tan(fovy/2)) to 1e-10, the principal point exactly at the centre, MuJoCo camera names and the exact Panda gripper dict (`H1/astranomy_agentic_geometry.py:79-128`). The wrist pose is FK times an exact mount (`:131-142`). A real calibration would be rejected, and triangulation accuracy (h3_live's 0.4–1.7 cm) assumes perfect calibration.
- **Synchronous stepping.** Only about 1.2 ms/step is policy compute in H1/H3. H2-v1's local compute is 3.7 s per action and would break 20 Hz control.
- **Frozen world.** The H1 prompt (`H1/agent_speedup_v2_user_check_agent_client.py:182`) and the AST prompt both state "Simulation pauses during inference".
- **Evaluator oracles.** `stop_on_success`; "the episode continues"; "Only the environment evaluator determines success".
- **Rendered images.** DA3 and SAM ran on rendered images; real-camera error profiles are unknown.

**G4. Model-authored geometry is reproducible bias, not noise** (medium-high; goals: collision, zero-shot).
- First-box errors repeat within 1–2 cm across H1-family runs (C11).
- The first handle targets in T04 are (−0.075…−0.09, −0.105…−0.12) in 3/3 runs.
- AST's grasp side and contact point repeated in 2/3 runs.
- So self-consistency or resampling will not fix geometry; measured grounding will. H3's deviation lies outside the H1 spread.

**G5. Safety of the 0.10 m cap near obstacles** (medium; goal: collision).
- Grip-site overshoot never exceeded the clearance (0/104). The closest grip-site approach to a true obstacle was 27 mm, during T08's contact stage.
- The tracker validates its lookahead chord (`H1/tracking.py:116`), but only for the grip-site point.
- A* never ran, so lookahead behaviour on real detours is untested.
- The cap's main effect is a 4–5× higher top speed at unmodelled contacts (IC2).

**G6. Can the harnesses share components?** (medium; all goals).
- The H1 tree has no `foundation_harness`, `foundation_vision` or `robodojo_direct_sim`.
- H1's controller venv (`/workspace/gpt_field/.venv`: numpy 2.5.3, no scipy or torch) cannot import H2's `robot_geometry` (scipy `ConvexHull`, `cKDTree`) or H3's perception.
- H3 shows the uc7.1 runner runs under `.venv-foundation-vision` (numpy 1.26.4, scipy 1.17.1, torch 2.10). A merged build is feasible there or as a service process.
- H2's per-configuration check costs 0.12–0.17 s, fine per stage but not per step. Its FK reads the MJCF via `robot_kinematics.json`.
- Note: the H1 ↔ H3 comparison is code-controlled but not interpreter-controlled.

**G7. A measured grasp-continuation predicate exists in the data** (medium; goals: fast, few attempts). See bottom line 5 and `data/grasp_gap.json`. Gap readings near 1.4 mm in drawer runs were deliberate closed-finger pushes, so the predicate must key on operation intent.

**G8. Image and request-byte costs** (medium; goal: fast). About 893 tokens per image, and +0.11 s per request MB outside the server window. This bounds the overlay proposals (IC9).

**G9. Where contacts start along an approach** (medium; goal: collision). 8 of 15 onsets were within 1 cm of the commanded endpoint, which says the targets were too deep (palm offset missing). 3 of 15 began more than 2 cm early (IC1).

**G10. The instruction confound between H1 and H3** (medium; goals: few attempts, collision). H3's instructions add obstacle-map and clearance mandates (`live.py:49-55`). No ablation separates prompt text from perception.

---

## 4. Cross-harness patterns and one general fix each

| pattern | harnesses | evidence | general fix |
|---|---|---|---|
| Model-authored metric geometry | H1, H3 (inherited), K2, AST | boxes 5–12 cm off in 4/4 H1-family first maps; targets 2.4–5.7 cm off; reproducible; AST grasped the obstacle side | Harness-side metric grounding (mask triangulation across views and frames, support-plane fit). The model labels and selects; it never emits metres without a measured anchor. |
| Point or partial robot model in collision reasoning | H1 router, H3, AST; H2-v1 the inverse (self-depth treated as obstacle) | 76 % of H1 forbidden samples in unrouted stages; all 4 routed contacts were on routes approved as straight; T08 grip site ≥ 27 mm from the bottle while a finger hit it | One shared body + held-object swept model (FK hulls or spheres), used by every stage as a planner |
| DA3 per-view bias and robot self-surfaces in depth | H2, H3 (shared backends) | wrist table +18 %; robot depth decimetres off; 0 usable centres; H2 B1–B3 | Rendered-robot image mask with depth replacement; affine alignment to the external view; DA3-free triangulation for objects |
| Serial, blocking model loop | H1, H3, H2, AST | model is 87 % of policy-intrinsic time; 58 % of calls follow a successful fragment; AST 4.7–4.9 actions per call | Committed sequences with measured continuation predicates; calls only at semantic decisions; async perception |
| No honest termination and unbounded attempts | H1, H3, H2 (`episode_model_call_cap: None`, K2 `scripts/evaluate_foundation_harness.py:108`), AST (no done tool) | H2 12 zero-progress calls; AST CA 12-call hold loop; T20 used the "episode continues" cue | Terminal `finish` + `give_up` + evaluation wall-clock or call cap recorded as failure + failure-signature dedupe |
| Evaluation oracle | all | 10/10 raw successes stopped while still holding the object | Score at the policy's own `finish` with strict physical checks; log raw success as a diagnostic only |
| Wall-clock staleness in a paused simulator | H2 (2.5 s refresh, `K2/…/perception.py:181`, unchanged), H3 (`max_age_s`) | 20 refreshes on a static scene; H3 tracks can never persist | Event- or step-indexed validity; capture time checked at action time |
| Sliding image window defeats caching | H1, H2, H3 vs AST | cached 0–2.5 % vs AST 25–42 % (append-only, but 48 MB) | Append-only history with images stripped at insertion; stable cacheable prefix |
| Contact handled as stall or failure | H1, H3, AST | 19 contact stalls; T08 press 47 steps; "until contact" cannot be expressed | Guarded motion plus contact as a declared stage terminator, reported to the model |
| Fixed tolerances below the error they guard | H1 (hidden lattice inflation), H2 (3 vs 6 mm vs cm-level depth error), H3 (2 cm + 2 % band, 0.9 gate) | spurious start-blocked refusals; H2 deadlock; H3 rejected correct detections | Derive tolerances each frame from measured uncertainty (robot self-pixels and the support plane) |
| Tests that cannot catch live blockers | H2, H3 (synthetic depth, full-image masks); H1 only a deterministic replay | B0–B4 reached a paid run | One shared saved-observation replay suite across harnesses |
| Quaternion double cover in prompts | H1, H2 (shared `astranomy_agentic_motion.py:89-97`) | 3 flips in 13 H1 T04 requests | Canonicalize w ≥ 0 before sending |

---

## Appendix: scripts and outputs (all under `w2/cross_critic_work/`)

| script | output | content |
|---|---|---|
| `scripts/calls_all.py` | `data/calls_all.json` | 390 calls: server usage, stream timing, images, request bytes |
| `scripts/latency_day.py`, `scripts/latency_resid.py` | `data/latency_day.json`, stdout | per-run throughput, multivariate fits, H3 residuals, decode vs context |
| `scripts/dump_exec.py` | stdout | executive segments and obstacle maps per run |
| `scripts/box_truth.py` | `data/box_truth.json` | box IoU and coverage against layout truth |
| `scripts/bbox_channel.py` | stdout | SAM 2D box vs model box placement |
| `scripts/overshoot.py` | `data/overshoot.json` | transit overshoot vs clearance; distance to true obstacles |
| `scripts/grasp_gap.py` | `data/grasp_gap.json` | finger gap at close and during lift vs model verdict |
| `scripts/contact_onset.py` | `data/contact_onset.json` | contact onset vs the approach's remaining distance |
| `scripts/affine_can.py` | `data/affine_can.json` | DA3 can and table error before and after h2's affine |
| `scripts/stall_join.py` | stdout, plus `data/stall_causes_out.txt` | join of the three stall taxonomies |
| inline | — | time-of-day decode fit, veto repair latencies, astranomy contacts and prompt, validator read |
