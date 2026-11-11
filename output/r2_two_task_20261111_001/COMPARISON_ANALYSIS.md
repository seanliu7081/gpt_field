The r2 runs added obstacle constraints, more slow contact motion, and longer model decisions, while final placement accuracy remained unresolved. This explains the observed regressions; it is not an estimate of either version's general success rate.

Only two frozen source files differ between r1 and r2: the revision declaration and the V3 client's obstacle-map instructions. The router, contact executor, robot guard, and simulator code are identical. Both task resets and run settings were matched.

| Measured quantity | T04 r1 → r2 | T08 r1 → r2 |
|---|---:|---:|
| Online wall time | 279.129 → 366.802 s | 240.114 → 449.776 s |
| Model wait | 181.850 → 247.471 s | 163.314 → 313.276 s |
| Simulator action time | 60.060 → 87.556 s | 52.668 → 98.034 s |
| Total actions | 474 → 700 | 504 → 936 |
| Contact motion actions, excluding gripper | 218 → 481 | 188 → 645 |
| Transit motion actions | 193 → 151 | 257 → 230 |
| Requests / average latency | 16 / 11.37 s → 17 / 14.56 s | 12 / 13.61 s → 15 / 20.89 s |
| Route construction | 0.018 → 0.114 s | 0.087 → 0.106 s |
| Combined task success | true → false | true → true |

The profile totals omit the separately counted gripper operations and stationary tool/refusal hold actions; complete counts are in comparison_diagnosis.json. Model, guard, and vision timings sit inside the decision timing and must not be added to decision time again.

The maps created hard start refusals. The r2 prompt asks Astra to provide relevant boxes, retain them, and inflate clearance for the hand/carried object and estimation error. The unchanged router refuses transit when the start-position field query is blocked. T04 step 87 and T08 step 304 had measured grip positions directly inside expanded boxes. T08 step 25 was different: the grip point was 2.55 mm outside the expanded bottle box, but four corners of its 20 mm interpolation cell were blocked; the field query rejects any blocked interpolation corner. Astra explicitly responded with contact-profile upward recovery. These are direct recorded sequences, not hypothetical explanations. The geometry is approximate; a route refusal does not itself establish physical collision. No detour graph search ran in either r2 episode.

The slow profile then appeared beyond those recoveries. Transit permits a 100 mm translation command and uses 4 mm positional arrival tolerance; contact permits 5 mm and uses 2 mm tolerance. These are controller command limits, not guaranteed displacement or a fixed 20-fold speed ratio. T04's five upward verification/withdrawal motions used transit in r1 (50 actions for 525 mm requested total travel) versus contact in r2 (289 actions for 350 mm). This subset alone added 239 actions. In T08, withdrawing after the first release took 12 transit actions for 160 mm in r1, versus 115 contact actions for 135 mm in r2. The second grasp verification took 11 versus 66 actions, and final withdrawal 10 versus 62. These later choices were not each preceded by a route refusal. More cautious profile selection is observed; attributing every such choice to the added prompt remains an inference.

Longer model decisions dominate added wall time. Extra model wait accounts for 65.621 of T04's 87.673 extra seconds (74.8%) and 149.962 of T08's 209.662 extra seconds (71.5%). Server-reported reasoning tokens rose 1,607 → 2,297 and 1,688 → 3,746; output tokens rose 4,421 → 6,579 and 3,748 → 7,740. This is consistent with more involved planning/map responses, but the logs cannot separate prompt complexity from sampling or service-latency variation. It would be unsupported to blame all model latency on the extra map text. Per-step simulator time was essentially unchanged. Route construction contributed less than 0.1 extra second per episode. T04 optional vision was actually faster in r2 (9.17 versus 23.02 s).

T04's failure is an additional placement problem. r2 grasped the other side of the yellow mug, compensated with a different grip-site target, then stalled on the final descent with about 8.05 mm grip-site error. Astra released and finished on visible support without a subsequent lateral centering correction. The final mug-origin/plate-center distance was 38.31 mm versus 21.31 mm in r1; raw LIBERO requires strictly less than 30 mm. Both runs' selected plate-pixel DA3 Y estimates differed by only about 1.44 mm. The evidence therefore does not isolate plate localization as the cause, nor prove the map directly pushed the mug off-center. Grasp offset, tilt, tracking residual, and settling may all contribute. A grip-site route does not verify object-to-target placement precision.

The extra contacts also occur outside the map's useful coverage. r2 T04's 18 forbidden samples were yellow-mug/plate contacts; T08's 874 were pot/burner contacts. There was no recorded can or bottle obstacle-contact pair in those r2 runs. Contact-phase manipulation does not use the transit box router, so a better transit obstacle map alone cannot resolve those contact interactions. Counts are physics-substep samples, not distinct collision incidents or impact severity.

My interpretation is that r2 exposed an incomplete interface between approximate obstacle estimates and execution. It adds hard constraints and recovery work, while still depending on Astra to choose fast versus slow motion and to verify precise placement. The next changes worth evaluating are explicit phase-appropriate profile selection, principled handling of starts near uncertain clearance margins (including interpolation-corner rejection), and object-relative placement verification. Do not simply erase obstacles or bypass constraints whenever they block movement. Repeated matched trials would be needed to estimate success rate or attribute the entire latency difference to the prompt.

No runtime sources were edited and no additional episode or replay was run for this analysis.

Evidence: [r2 combined report](REPORT.md), [r1 combined report](../r1_two_task_20261111_001/REPORT.md), [machine-readable diagnosis](comparison_diagnosis.json). Frozen source: [r2 prompt](runtime/src/gpt_field/foundation_vision_harness_V3/client.py), [motion profiles](runtime/src/gpt_field/agent_speedup_v2_user_check_fragments.py), [route admission](runtime/src/gpt_field/agent_speedup_v2_user_check_routing.py).
