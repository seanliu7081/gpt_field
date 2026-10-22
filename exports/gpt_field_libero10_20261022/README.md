---
language:
- en
tags:
- robotics
- libero
- evaluation
pretty_name: SpeedUp v2 User Check - LIBERO-10
configs:
- config_name: results
  data_files:
  - split: test
    path: results.csv
---

# SpeedUp v2 User Check: LIBERO-10

One manually reviewed episode per task, t00-t09, from the completed manual batch.
Method: `agent-speedup-v2-user-check-7`; model: `gpt-6-astra`; reasoning effort: `high`;
episode index: `0`; seed: `7`.

Each `agent_speedup_v2_user_check_libero_10_*` directory preserves its original
run artifacts, including video, traces, model request logs, approved plan,
evaluation records, and frozen source snapshots. Historical retries, review
servers, launch directories, and smoke tests are not included.

## Requested metrics

Success means the raw LIBERO success flag recorded in `summary.json`.
Time is `control_without_video_s`; model calls are execution-stage `api_calls`.
Initial plan generation and manual approval waiting are excluded.
The t06 run was manually stopped; its time is an incomplete attempt duration.

| Task | LIBERO success | Time without video (seconds) | Model calls |
|---|---|---:|---:|
| libero_10_t0 | Yes | 412.442 | 14 |
| libero_10_t1 | Yes | 325.391 | 13 |
| libero_10_t2 | Yes | 296.217 | 15 |
| libero_10_t3 | Yes | 346.722 | 21 |
| libero_10_t4 | Yes | 342.182 | 16 |
| libero_10_t5 | Yes | 253.581 | 10 |
| libero_10_t6 | No (user stopped) | 593.175 | 27 |
| libero_10_t7 | Yes | 323.589 | 13 |
| libero_10_t8 | Yes | 313.602 | 13 |
| libero_10_t9 | Yes | 424.167 | 19 |

Official successes: **9/10**. Mean time among successful tasks:
**337.544 seconds**.
Total execution-stage model calls: **161**.

The retained physical diagnostics use additional criteria and are distinct from
the raw LIBERO result reported above. These are single-episode records, not a
multi-seed benchmark.

`results.csv` and `results.json` contain the metric index.
`upload_manifest.json` records file sizes and SHA-256 hashes for all original run
files. Original absolute paths inside saved artifacts are provenance records.
