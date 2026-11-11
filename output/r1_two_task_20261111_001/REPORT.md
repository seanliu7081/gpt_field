Both requested obstacle tests completed successfully using the exact archived `foundation_vision_harness_V3 / astra_tools_r1` runtime. Each ran once, sequentially on GPU 7, with Astra high, subscription-only authentication, on-demand vision, episode 0 / seed 7, original frozen obstacle inputs, and video recording.

| Metric | T04 can obstacle / 008 | T08 bottle obstacle / 004 |
|---|---:|---:|
| Online wall time | 279.129 s | 240.114 s |
| Completion including terminal hold/scoring | 280.430 s | 241.230 s |
| Actions | 474 | 504 |
| Astra requests | 16 | 12 |
| Raw LIBERO at terminal | True | True |
| Strict at terminal | True | True |
| Strict after unchanged-gripper hold | True | True |
| Combined success | True | True |
| Forbidden contact samples | 4 | 565 |

Both policies finished autonomously after release and retreat. Primary time includes online model/tool waits, simulator work, logging, and recording submission. The additional ten-step terminal check held the existing gripper command and pose; it did not release or reposition objects for the policy.

T04 recorded four white-mug/obstacle-can contact samples. T08 recorded 565 pot/stove-burner samples (492 for the first manipulated pot, 73 for the second), with no recorded bottle contact. These are physics-substep samples, not counts of separate collisions. Both tasks passed their frozen success criteria, but neither run was collision-free.

T04 used empty obstacle-box collections on its 13 transit routes. T08 supplied one approximate obstacle box, used by all 14 transit routes; all initial straight routes passed, so no detour search ran. T08 demonstrates that the supplied box constraint was active, but these single episodes do not isolate a causal benefit from obstacle routing or optional vision tools. Model waits accounted for 65.1% and 68.0% of online time, respectively.

The exact 370 r1 source files and both initial observation hashes were verified. Sources remained unchanged, both saved-artifact audits passed 38 checks, videos finalized, and both runners exited cleanly. The private supervisor was shut down. No automatic retry was started and no task remains queued.

Artifacts:
- T04: [report](../../runs/foundation_vision_harness_V3_t04_can_obstacle_008/REPORT.md), [video](../../runs/foundation_vision_harness_V3_t04_can_obstacle_008/trial/episode.mp4), [result](../../runs/foundation_vision_harness_V3_t04_can_obstacle_008/result.json), [independent audit](t04_terminal_audit.json).
- T08: [report](../../runs/foundation_vision_harness_V3_t08_bottle_obstacle_004/REPORT.md), [video](../../runs/foundation_vision_harness_V3_t08_bottle_obstacle_004/trial/episode.mp4), [result](../../runs/foundation_vision_harness_V3_t08_bottle_obstacle_004/result.json), [independent audit](t08_terminal_audit.json).
- [Machine-readable summary](summary.json), [source provenance](source_provenance.json), [shutdown evidence](shutdown.json).
