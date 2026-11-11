The requested `foundation_vision_harness_V3 / astra_tools_r2_obstacle_map` tests are complete. T04 did not meet the full task criterion; T08 passed. Each ran once, sequentially on GPU 7, using the same frozen obstacle inputs, episode 0 / seed 7, Astra high subscription, on-demand vision, 100 mm command cap, and video settings as the immediately preceding r1 tests.

| Metric | T04 can obstacle / 009 | T08 bottle obstacle / 005 |
|---|---:|---:|
| Online wall time | 366.802 s | 449.776 s |
| Actions | 700 | 936 |
| Astra requests | 17 | 15 |
| Raw LIBERO at terminal | False | True |
| Physical checks at terminal | True | True |
| Physical checks after unchanged-gripper hold | True | True |
| Combined task success | False | True |
| Forbidden contact samples | 18 | 874 |

T04: both mugs were released, rested, and supported by their plates, but the yellow-and-white mug origin was 38.31 mm from the right plate center, outside raw LIBERO’s strictly-less-than-30-mm requirement. The white mug distance was 18.35 mm. Raw failure persisted after the ten-step unchanged-gripper hold. The 366.802 s is an incomplete attempt duration, not completion latency. All 18 prohibited samples were yellow-mug/right-plate contact during placement descent; no can-obstacle contact pair was recorded.

T08: both pots were released on the stove, and raw plus both physical checks passed. Completion including terminal hold/scoring was 450.883 s. All 874 prohibited samples were pot/burner contacts (655 first handled pot, 219 second), with no forbidden bottle or robot pair. Contact counts are physics-substep samples, not independent collision events.

| Task | r1 online / result | r2 online / result | Time difference | Contacts r1 → r2 |
|---|---|---|---:|---:|
| T04 | 279.129 s / pass | 366.802 s / fail | +87.673 s (31.4%) | 4 → 18 |
| T08 | 240.114 s / pass | 449.776 s / pass | +209.662 s (87.3%) | 565 → 874 |

These single episodes used identical inputs and settings, but their model decisions differ. T04 does not provide a successful-completion speed comparison, and neither comparison isolates a causal effect of the map prompt.

The r2 runtime differs from original r1 in two source files: the revision declaration and obstacle-map instructions. Its routing/executor code is unchanged. In these tests, T04 used 3–4 boxes and T08 used 3 boxes across recorded transit routes. One T04 and two T08 transit starts were refused by the obstacle-field query; Astra selected contact-profile withdrawals/recoveries. T04 step 87 and T08 step 304 were directly inside expanded boxes. T08 step 25 was 2.55 mm outside the expanded box but rejected because four corners of its 20 mm interpolation cell were blocked. No detour graph search ran. Contact motions do not use the transit box route. Model request time was 67.5% and 69.7% of online time; route construction totaled only 0.114 s and 0.106 s.

All 370 source hashes and both initial observation hashes were verified. Both independent saved-artifact audits passed 39 checks, videos finalized, and sources remained unchanged. Both runners exited, their private supervisor was shut down, and GPU 7 returned to idle. No automatic retry or remaining task queue exists.

Artifacts:
- T04: [report](../../runs/foundation_vision_harness_V3_t04_can_obstacle_009/REPORT.md), [video](../../runs/foundation_vision_harness_V3_t04_can_obstacle_009/trial/episode.mp4), [result](../../runs/foundation_vision_harness_V3_t04_can_obstacle_009/result.json), [independent audit](t04_terminal_audit.json).
- T08: [report](../../runs/foundation_vision_harness_V3_t08_bottle_obstacle_005/REPORT.md), [video](../../runs/foundation_vision_harness_V3_t08_bottle_obstacle_005/trial/episode.mp4), [result](../../runs/foundation_vision_harness_V3_t08_bottle_obstacle_005/result.json), [independent audit](t08_terminal_audit.json).
- [Machine-readable summary](summary.json), [source provenance](source_provenance.json), [shutdown evidence](shutdown.json).
