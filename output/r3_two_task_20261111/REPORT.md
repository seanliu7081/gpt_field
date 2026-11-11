# V3 r3：T04 与 T08 obstacle 两轮汇总

本次用户授权分别运行一次 T04 can obstacle 与 T08 bottle obstacle，均使用 `foundation_vision_harness_V3 / astra_tools_r3_grid5mm` 的同一封存源码、各自的 init 0/seed 7 输入、5 mm 网格、120 步 transit 预算和 100 mm 命令上限。两轮顺序运行，源码未修改。

| 项目 | T04 / 007 | T08 / 003 |
|---|---:|---:|
| 结束方式 | Astra finish | 用户停止 |
| 在线尝试耗时 | 511.297 秒（8分31秒） | 489.136 秒（8分9秒） |
| 扣已测录像提交／日志 | 503.220 秒 | 480.441 秒 |
| 动作／模型请求 | 832／18 | 935／20 |
| 模型耗时及占比 | 354.661秒／69.4% | 354.237秒／72.4% |
| Raw LIBERO | false | false |
| 自定义终态物理条件 | true | false |
| 最终组合成功 | false | false |
| 禁止接触采样点 | 0 | 4612 |

**T04：两杯均已释放并由对应盘支撑，但白杯原点与盘中心相距34.31 mm，超过raw判定要求的30 mm以内。** 黄白杯距离13.18 mm，raw成立；它第一次空抓后通过第二次尝试抓起。模型最终没有识别白杯的偏心。物理条件与raw条件并非同一个判据，因此不能将物理通过当作完整成功。

**T08：近壶闭爪搬运时滑脱，恢复空抓后改抓另一壶。** 用户停止时，第一壶横躺炉面，第二壶已抓起但未放置；模型刚返回的搬运命令被拦下。此次触发方式不同于上一轮的旋钮区域放置错误，不能沿用上一轮因果解释。

两轮均未达到任务成功；时间是尝试时长。两轮路线计算分别仅0.290与0.284秒，模型占比约七成。当前限制仍是定位与放置精度、抓持保持和失持后的恢复，而非网格计算吞吐。

两进程均正常退出、源码保持一致、视频完成，没有自动重试或剩余任务队列。

- [T04完整报告](../../runs/foundation_vision_harness_V3_t04_can_obstacle_007/REPORT.md) · [T04视频](../../runs/foundation_vision_harness_V3_t04_can_obstacle_007/trial/episode.mp4)
- [T08完整报告](../../runs/foundation_vision_harness_V3_t08_bottle_obstacle_003/REPORT.md) · [T08视频](../../runs/foundation_vision_harness_V3_t08_bottle_obstacle_003/trial/episode.mp4)
