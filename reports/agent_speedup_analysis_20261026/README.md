# Agent SpeedUp analysis: v2 user-check v7 and v3 (2026-10-26)

This folder answers three questions about v2 user-check (latest version: v7) and v3 (latest version: r11):
1. What is the framework of the working version and of the latest version?
2. What is the current bottleneck?
3. What should happen next?

The analysis was read-only. It made no code changes, ran no simulator and made no model calls. The work had three stages:
- Seven analysts read the code and run artifacts in parallel.
- Six skeptics tried to refute 18 key claims by recomputing them from the raw run files.
- Three plans written from different angles were scored by a judge and merged into one plan.

## Reading order

| File | Contents |
|---|---|
| [plan/final_plan.md](plan/final_plan.md) | **Start here.** The merged next-step plan: uc8 → uc9 → uc10, an S300 time budget, the v3 decision and the decisions you need to make. |
| [verification.md](verification.md) | Skeptic verdicts on the 18 key claims: 11 confirmed, 7 partly confirmed with corrected numbers. |
| [uc7_framework.md](uc7_framework.md) | How v7 actually runs, from the code: diagram, one control cycle, module, limit and veto tables, and the v6 → v7 diff. |
| [uc7_runs.md](uc7_runs.md) | v7 per-run table, time breakdown, the gap to 300 s, motion efficiency, the t6 failures, and success taken mid-release. |
| [v3_framework.md](v3_framework.md) | v3 r11 against r6: the layers v3 adds to v7, a full inventory of the gates that can stop motion, and what ended each failed run. |
| [v3_runs.md](v3_runs.md) | v3 per-run table: why it is 3–4× slower than v7 on the same t1 reset, the failure mechanism of each run, and what drove the r6 success. |
| [model_io.md](model_io.md) | The model-call channel: what drives latency, the hanging plan requests (whitespace), how the history grows, the serial loop, and a size estimate for each lever. |
| [campaigns_today.md](campaigns_today.md) | Today's three Codex campaigns: what ran, the results, the harness defects found, and where the 4 hours went. |
| [history_intent.md](history_intent.md) | Timeline from v1 to v3 r11, your stated goals and constraints, decisions still open, and recurring patterns. |
| [plan/A_speed_first_uc7.md](plan/A_speed_first_uc7.md), [plan/B_async_executive.md](plan/B_async_executive.md), [plan/C_generality_reliability.md](plan/C_generality_reliability.md) | The three candidate plans the judge scored. C won narrowly over A; B came last. |

## Main conclusions

### v7 works; v3 does not

- **v7 is the working framework.** 11 of 13 runs succeeded (raw LIBERO), covering 9 of 10 LIBERO-10 tasks; t6 failed both times. The mean successful service time is 449 s. Only t5_001 finished under 300 s, at about 270 s.
- **v3 should be frozen at r11.**
  - Its gates create most of its model calls: 36 of 39 follow-up calls in the r6 success came after a gate stopped motion.
  - Each call takes 2.4× longer than in v7.
  - Its four successes were all on t1 and ran ≥ 81.6% through v7-style execute fragments.
  - Since r6 it has 0 successes in 7 runs, and 0 in 3 more today.

### Where v7's time goes

| Share | Cause |
|---|---|
| Model calls, ~50% | The robot stands still during every call, and every finished fragment triggers a new one. |
| Per-step overhead, ~42% | Rendering, PNG encode/decode, video and state-file writes cost ~0.23 s per step, for 0.05 s of simulated time. |
| Slow motion | The arm moves ~24% of each commanded step. The 20 mm per-step cap is a framework choice, not a hardware limit. |

### Framework bugs confirmed in the code

- **Spatial words are mapped to the wrong axes.** `src/gpt_field/agent_speedup_v2_user_check_spatial.py:47-49` hard-codes `right_of=+x`. The camera calibration has image-right = world +y. This is why t6's "right of the plate" region was grounded on the wrong side.
- **Plan-draft requests can hang.** The v7 plan prompt lacks v3's compact-output instruction and has no total deadline. 11 of 38 such requests ran long or failed, against 0 of 12 with the instruction. One request that eventually completed was 99.6% whitespace.

## Paths inside the reports

The reports cite helper scripts and intermediate data under `/tmp/claude-0/-workspace/02fb8e60-2ddf-4353-88bc-336305aea415/scratchpad/...`. That session directory may be deleted. The same files are copied here under [work/scratchpad/](work/scratchpad/) with the same layout. Workflow results are in `work/scratchpad/analysis/wf1_results.json` and `work/scratchpad/analysis/wf2_results.json`.
