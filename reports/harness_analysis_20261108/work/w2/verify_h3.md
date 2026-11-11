# verify_h3: adversarial check of the H3 (foundation_vision) W1 reports

Reports checked:
- `w1/h3_vision_engine.md` (engine findings E-F1 to E-F10)
- `w1/h3_live_integration.md` (live findings L-F1 to L-F11)
- the latency claim in `w1/cross_eval_methodology.md` (§1 item 6 and B5)

How this was checked:
- **Read-only.** No simulator, model or GPU work. Only offline numpy work on saved artifacts.
- **Same code as the runs.** All `foundation_vision/*.py` files, the wrapper, and the uc7.1 agent client, fragments, routing, agent and history modules hash-identical to both runs' `source_at_launch`. Both `terminal_source_audit.json` files report `passed: true`. So every cited line is the code that executed.
- **Independent recomputation.** Every number below was recomputed from raw artifacts with my own scripts in `w2/scripts/` (`v_*.py`). The raw artifacts were:
  - `trial/foundation_vision/frame_*_{snapshot,sam_candidates}.{json,npz}`
  - `trial/requests/*.{request,response,result,stream}.json`
  - `review/requests/*` (the initial plans)
  - `trial/trace.jsonl` and `trial/collision.json`
  - `inputs/layout_manifest.json`
  - `foundation_vision_preparation_001/da3_real_003/cold_native_prediction.npz`

## Bottom line

1. **The core H3 diagnosis holds.**
   - There were 0 usable 3D targets in 143 target-frames, caused by prompt form, the two-view requirement and the 0.9 bidirectional gate.
   - Consumption is advisory only: no live caller of `query_path`/`assess_path`, and routing runs only for transit segments against Astra's own boxes.
   - The two runs used a per-scene `--vision-query` (`can`, `bottle`).
   - Stale vision text is replayed in every later call.
   - All of these reproduce exactly.
2. **Latency attribution (the decision requested).** The evidence supports the live report, not cross_eval's "vision context roughly doubled per-call latency".
   - The identical, vision-free initial-plan requests decoded at **61.6 / 63.7 tok/s on 11-05** against **89.8 / 82.8 tok/s on 11-02/03**. The SHA of each payload is equal across days.
   - Executive per-call visible decode had a median of **36.2 tok/s** for H3 (IQR 31.7–39.0), against **50.3** for 164 earlier uc7.1 calls (p25 43.7).
   - The earlier runs include contexts twice as long that still decoded at about 48 tok/s, so context length does not explain the slowdown.
   - The latency model refits exactly: TTFT = 3.89 + 0.049·rt + 0.0196·in_k, R² 0.937. It leaves server residuals of **+59.0 s (T04) and +91.8 s (T08)**.
   - The rest of the increase is about 2× reasoning tokens per call (259/254 vs 123/142), with an unidentified cause.
   - The vision context's directly measured cost is about 1.2–1.3 s of synchronous vision plus about 0.2 s of prefill per call. cross_eval B5 should be corrected.
3. **Material corrections.**
   - **E-F1 arithmetic.** 66 + 14 + 7 = **87/143 (61 %)**, not 97/143 (68 %). Also, the simple-noun prompts (the "F1-fixed" case) still produced **0/22** usable centres, so F1 is not the dominant cause on its own.
   - **E-F4.** The external DA3 table error is a **range-dependent tilt**: about +3.7 % at 0.65 m and −4.7 % at 1.3 m, with a median of −0.35 % once objects are excluded. The "+3.5 %" is the histogram mode. Per-view self-calibration worsens object geometry.
   - **E-F7 does not describe the engine's output.** The engine's own supported-point bounds for the can would be 0.86–1.14× the true size. The 1.5–2.8× figures are raw single-view mask extents.
   - **L-F4 token share.** The vision JSON tokenizes at about 2.6 chars/token (about 2.9k tokens per summary). That makes it about **31 % / 29 %** of input tokens, not 26 % / 24 %.
   - **L-F8 timing.** The 400 samples did **not** accrue during about 20 pressing actions. The count went 0→278 in actions 277–282, stayed flat through 283–298 (classified as support), then rose 278→400 in 299–301, after the open command.
   - **Triangulation scope.** It works on only **3/22** frames. Errors there were 0.4–1.7 cm xy, with a z error of −2.5 cm at T08 f243. The ≥ 21/22 coverage needs single view plus a plane height. With the true height that gives 1.3–1.5 cm (can) and 0.6–0.7 cm (bottle); with a ±2 cm plane error it gives 1.3–3.3 cm.

## Verdicts: engine report (`h3_vision_engine.md`)

| Ref | Verdict | Severity after review |
|---|---|---|
| E-F1 prompts | confirmed with corrections | high (not critical alone) |
| E-F2 two-view requirement | confirmed | high |
| E-F3 0.9 bidirectional gate | confirmed with corrections | high |
| E-F4 DA3 scale bias | confirmed with corrections | medium |
| E-F5 robot self-surfaces | confirmed with corrections | medium |
| E-F6 no free space | confirmed with corrections | medium |
| E-F7 extent inflation | partially refuted | low |
| E-F8 no tracking | confirmed | medium |
| E-F9 vision cost | confirmed with corrections | medium |
| E-F10 no truth validation | confirmed | medium |

**E-F1.**
- What reproduces:
  - The six prompts T04 A/B/E and T08 A/B/E have **no SAM rows at all**, even before the 8 px filter (`frame_*_sam_candidates.json`).
  - They cause 66/76 of the no-candidate frames.
  - Ambiguity: 14 frames (T04 C ×11, T08 D ×3). Wrong instance: 7 frames (T04 F ×5, T08 D ×2).
  - `can` was found externally in 11/11 frames (scores 0.821–0.857) and `bottle` in 10/11 (0.712–0.789).
  - Duplicate masks: C/F ext 11 (IoU ≥ 0.998) and wrist 5; C/D ext 6; D/bottle ext 3 and wrist 1.
- Wrong-instance claims, verified by projecting truth:
  - The T04 wrist "right plate" masks at f17/f201 contain plate_1, which is the *left* plate.
  - The T08 wrist D masks at f0/f24 (scores 0.935/0.927) contain the ketchup bottle.
- Corrections:
  - 97/143 → 87/143 (61 %). 97 adds all 76 none-frames, including one `bottle` frame.
  - "SAM cannot resolve left/right" is overstated. "right plate" and "right moka pot: whole faceted body" were grounded correctly in the external view.
  - Handle prompts were 0/4 correct, but 4/7 whole-object part phrases did fire.
  - The simple-noun prompts got 0/22 usable centres: 19 single-view frames and 3 rejected at 0.9.

**E-F2.**
- The two-view rule is real: `geometry.py:34,213-222`; `engine.py:150-151,200`.
- Wrist detections were 11/77 and 6/66; unique external detections were 32/77 and 20/66.
- External DA3 on the can, from my cylinder ray-cast: median +1.2 mm, |err| p50 4.3 mm, p90 10.5 mm. The external SAM mask has IoU 0.911 with the true silhouette.
- Caveat: the can sits at 0.77 m, where the external error is near zero. Single-view hypotheses at other ranges inherit an error of up to ±3–6 cm along the ray (see E-F4), plus raw-extent inflation (see E-F7).

**E-F3.**
- The numbers are exact: 0.638/0.913, 0.832/0.847, 0.042/1.0, and 0.0 for the 7 wrong-instance pairs.
- With true cylinder geometry, ext→wrist agreement would be **0.717**, still below 0.9. The wrist SAM mask has coverage 0.529, precision 0.961 and IoU 0.518.
- Correction: the overlay (`w2/t04_f0_can_masks_vs_truth.png`) shows SAM **excluded the visible metal lid**, and the white mug's rim clips the lower edge. Border truncation is not the main cause.
- On the proposed fix: the 0.8 max-direction plus ≥ 100 comparable-point rule is a veto re-tuned on n = 10 cases.
- A DA3-free epipolar check separates the cases by about 10×:
  - same-instance mask-centroid ray gaps are 0.5–2.1 cm;
  - wrong-instance ray gaps are 22–49 cm.
  - This is the better replacement (`v_raygap.py`).

**E-F4.**
- Reproduces exactly: support 12,363 / 29,555 and contradicted 91,607 / 54,919. Self-calibrated support is 37.1 % / 68.7 %. The rotation error is 0.043 rad.
- The wrist table bias is confirmed: **+18.1 % (+6.2 cm)**.
- The external table error by true plane depth:

  | Plane depth (m) | 0.63–0.68 | 0.68–0.80 | 0.80–1.02 | 1.02–1.17 | 1.17–1.38 |
  |---|---|---|---|---|---|
  | Error | +3.7 % | +3.1 % | −0.4 % | −2.6 % | −4.7 % (−6 cm) |

- So the external far background is too *shallow*, not too deep. A per-view scalar cannot fit this.
- Self-calibrating to 0.84 / 0.72 hurts. The engine-style can centroid error goes from 2.0 cm to **4.0 cm xy and +3.6 cm z**, and the x-extent shrinks to 0.3×. Near-table free air becomes 79 % false "surface".
- The "unstable per-frame scale" reading is supported. Co-visible agreement varies from 0.3 % to 92 % per frame, so the variation is not just co-visibility.

**E-F5.**
- 10,332 / 39,491 hypotheses within 5 / 10 cm: exact.
- The 10 cm descent is blocked for tube radius ≥ 2 cm. At 3 cm the first blocker is wrist pixel (258, 431), depth 0.155 m, i.e. the fingers.
- At radius ≤ 1.5 cm the result is `unknown`, not blocked.

**E-F6.**
- At the shipped scale, 0 % of free-air probes are `estimated_free`: 17 % surface / 83 % unknown at 2–4 cm above the table, and 0.3 % / 99.7 % at z 0.56–0.64.
- Frusta overlap 24 % / 19 %. There is no live caller.
- Correction: self-calibration gives about 9 % free, not 6 %, while turning 79 % of near-table probes into false surface.

**E-F7 (partially refuted).**
- The raw extents reproduce: 0.175×0.108×0.088 and 0.059×0.103×0.135.
- But `bounds_world_m` only takes points that pass the support/contradiction filter (`geometry.py:236-238,252-253`). For the can at f0 those bounds would be **0.054×0.058×0.087 m (0.86–1.14×)**.
- Per-axis raw ratios are 0.95–2.8×.
- The finding is relevant to single-view hypotheses (E-F2) and to H2's "partial" points, not to current H3 output.

**E-F8.**
- Snapshot gaps were 24.0–66.3 s and 24.8–85.8 s.
- 0 track IDs were assigned. `need_observation` fired 77/66 times and was never forwarded.
- The engine default `max_age_s` is 1.0 and the wrapper sets 5.0.

**E-F9.**
- Timings are exact: median 1.159 / 1.066 s, first frame 2.837 / 2.310 s, totals 14.64 / 13.11 s, SAM 64 % of each frame.
- Preload was 24.6 s for T04 but **23.5 s for T08**. SAM peak memory was 5.98 GB.
- Token figures are exact: 9.5k vs 6.6k and 9.0k vs 6.2k per call; 106,092 vs 75,454 at call 11.
- Correction: the T08 summary was **6,946–7,337 chars**, not 7.6–8.1 kB.
- The per-prompt SAM cost (two medians from different scenes) is weak evidence.

**E-F10.**
- `validation_report.json` has `geometry_accuracy_validated: false`, with 82/87 synthetic tests. The proposed benchmark is sound.

## Verdicts: live report (`h3_live_integration.md`)

| Ref | Verdict | Severity after review |
|---|---|---|
| L-F1 four gates | confirmed with corrections | critical |
| L-F2 advisory only | confirmed | critical |
| L-F3 latency attribution | confirmed (minor corrections) | medium |
| L-F4 stale summaries re-sent | confirmed with corrections | medium |
| L-F5 operator obstacle name | confirmed with corrections | high |
| L-F6 wall-clock staleness | confirmed | medium |
| L-F7 plan cells resolve ambiguity | confirmed | medium |
| L-F8 T08 400 samples | confirmed with corrections | medium |
| L-F9 fail-closed | confirmed | medium |
| L-F10 better first boxes | confirmed (hypothesis) | low |
| L-F11 scene-point noise | confirmed | low |

**L-F1.**
- The gates and counts are exact.
- Triangulation against my can truth (collision-bounds centre) gives 0.9 cm; against the body origin it gives 1.2 cm.
- The bottle triangulates at 1.7 cm xy (f243) and 0.4 cm (f328). The f243 z error is −2.5 cm, which the report omits.
- Scope: both-view masks exist in only 3/22 frames. Single view plus the true plane gives 1.28–1.48 cm (can) and 0.60–0.67 cm (bottle) in every frame.
- The analyst's `single_view2.py` processed only the first two frames, but the camera is static, so the result extends to all frames.
- T08 D ambiguity was pot plus **bottle**, not "both pots".

**L-F2.**
- Settings are at `evaluate_foundation_vision_task.py:106-108`. Routing applies only to transit segments (`fragments.py:293`) and uses only the grip-site path (`routing.py:1-5`).
- `query_path`/`assess_path` are called only in tests.
- 2/22 outputs mention vision, both saying it is unavailable. Each box holds 0–5 of the sampled points, at most 2 of them supported.

**L-F3.**
- The table numbers are exact.
- My refit reproduces the 252.1 s and 249.0 s predictions.
- The reasoning rate is 15.3 vs 20.4 tok/s.
- The H1 T08 plan decoded at 82.8 tok/s, not 84.
- **Supported over the cross_eval attribution.**

**L-F4.**
- The mechanism is confirmed: `direct_history.py:43-56`, `agent_client.py:417`. Chars are exact (86,135 / 78,099) and caching is near zero (0 / 3,328 / 16,000).
- Correction: per-message token attribution gives about 2.6 chars/token for the vision JSON, about 2.9k tokens per summary.
- Totals are about 202k / 184k tokens, **31 % / 29 %**. The last call carries about 33.6k / 30.5k vision tokens.

**L-F5.**
- `run.py:48` in both runs.
- Nuance: the T04 plan's scene text *does* say "a small can nearby", so plan nouns would have found the can but not the T08 bottle.

**L-F6.**
- Payload ages were 1.14–2.94 s, and accepted responses arrived 12.8–78.7 s after capture. The same-step reuse goes stale (test lines 158-175). Events are not forwarded.

**L-F7.**
- Plan cells resolve 11/11 and 3/3 ambiguous frames and are consistent with 31/31 single detections.
- Limits: they work only on the external view, at 64 px resolution, and only from plan time.

**L-F8.**
- The geometry is exact: local x = −78.2 mm against a 75 + 2 mm limit (`agent_speedup_v2_contacts.py:233-236`, `REGION_TOLERANCE_M` = 0.002). The stall at 296 and the open at 297 are confirmed.
- **The accrual timeline is wrong** (`trace.jsonl`):

  | Actions | 277–282 | 283–298 | 299–301 |
  |---|---|---|---|
  | Forbidden samples added | +278 | +0 (contact classified as support) | +122 (after the open command) |

- Stopping at touchdown would not remove the release burst. Placing inside the support footprint is the main lever.
- The same pot-vs-burner class appeared in H1 T08 (211 samples), so this is a recurring, structural issue.

**L-F9.**
- `agent.py:271-288` catches only `SchemaError`. The behaviour is asserted by tests 209-218. The largest observation was 34,003 chars, against the 56,000 limit.

**L-F10.**
- First-box coverage is 0.83 / 0.51 (H3) vs 0.30 / 0.00 (H1); with clearance 1.00 / 1.00 vs 0.76 / 0.00.
- Missing confound: `VISION_INSTRUCTIONS` adds obstacle-map guidance ("account for unobserved extents and uncertainty"). The proposed ablation cannot isolate that.

**L-F11.**
- The mechanism is confirmed (`live.py:143`, `geometry.py:291`).

## Missed issues, with evidence

1. **cross_eval B5 attributes the latency to vision.** The evidence above shows a server throughput drop on 11-05. The decision-driving text should be corrected.
2. **Fixing prompts alone gives 0 usable geometry.** `can`/`bottle` got 0/22 under the two-view rule and the 0.9 gate. The prompt, two-view and gate fixes must ship together, or be replaced by the DA3-free locator.
3. **Epipolar association.** A DA3-free ray-gap check separates same from wrong instance by about 10× on this data. It can replace the mask-agreement veto, in keeping with "remove vetoes".
4. **The external DA3 tilt makes "self-calibration by agreement" harmful for objects.** Use robot self-pixels (from FK) as affine anchors, or avoid DA3 for compact objects.
5. **Engine label.** Association-rejected single-candidate targets are labelled `identity_ambiguous` (`engine.py:195-196`), so Astra is told the can was "ambiguous".
6. **Collision accounting.** The H1 T08 forbidden set also contains a held-pot-vs-bottle contact (111 samples, action 223) and 211 pot-vs-burner samples. cross_eval's "only robot-vs-obstacle collision" undercounts violations of the user's held-object rule.
7. **Plane height is the crux of single-view localisation.** A ±2 cm plane-height error yields 2.4–3.3 cm; DA3's external plane is tilted by ±4 %. A metric plane source is needed: robot contact, or FK-anchored wrist depth.

## Scripts (all under `w2/scripts/`)

- `v_frames.py`, `v_cands.py`, `v_assoc.py`: per-frame recount, raw SAM rows, association evidence.
- `v_triang.py`, `v_single.py`, `v_raygap.py`: DA3-free localisation and association.
- `v_can_raycast.py`, `v_can_depth.py`, `v_crop.py`: cylinder truth, silhouettes, depth error, overlay.
- `v_da3_table.py`, `v_geom_repro.py`, `v_free.py`, `v_extent_engine.py`, `v_frustum.py`: DA3 bias, self-blocking, free space, engine bounds.
- `v_latency.py`, `v_latmodel.py`, `v_tokens.py`: latency attribution and token share.
- `v_boxes.py`, `v_box_points.py`, `v_outputs.py`, `v_cells.py`, `v_wrong_inst.py`: obstacle boxes, Astra outputs, plan cells, wrong instances.
