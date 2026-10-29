# P0 仓库与版本核验

本轮真实仓库为 `/workspace/gpt_field`，Git HEAD 为 `59f4b0f`。已有工作树修改包括 README、v2 user-check 客户端和模拟器；v3 代码、测试、设计与历史文档未被 Git 跟踪。当前文件内容必须独立封存，不能仅凭 Git commit 重建，也不能覆盖用户修改。

读取真实常量与运行冻结清单后确认：当前 v3 policy/client/review 为 11，prompt 为 10；并非从历史标题推定。当前源码与 `runs/agent_speedup_v3_t1_012_review/review.json` 绑定的全部 runtime 文件哈希一致。运行编号 `t1_011` 对应 policy 10，`t1_012` 对应 policy 11。逐模块哈希、常量和授权原文见 `p0_verification.json`。

| 真实模块 | 当前职责 | 本轮建议修改位置 |
| --- | --- | --- |
| `scripts/evaluate_agent_speedup_v2_user_check.py` | v7 审核、执行入口和原评估器适配 | 在独立复制树中建立 A/B 公共计时与预算适配 |
| `agent_speedup_v2_user_check_agent.py` | v7 持续模型决策、普通实测片段和可选旧宏 | 保留行为作为 A |
| `agent_speedup_v2_user_check_agent_client.py` | v7 请求、历史和结构化响应 | 仅选定上下文候选时改独立 B 副本 |
| `agent_speedup_v2_user_check_fragments.py` | `move_to_pose/move_relative/approach/set_gripper/wait` | 保留动作逻辑；stall 结束片段并明确原因未知，未自动 close |
| `scripts/evaluate_agent_speedup_v3.py` | v3 审核、私有 runtime 装配、冻结与完整执行计时 | 参考，不将完整 v3 移植进 v7 |
| `agent_speedup_v3_performance.py` | 状态文件写入节流、记录成本、300 秒报告 | 可隔离复用；目前没有900秒有效服务终止器 |
| `agent_speedup_v3_video.py` | 现有 RGB 的紧凑展示录像，每真实动作一帧 | 可隔离验证后复用；编码排空仍同步 |
| `agent_speedup_v3.py` | 路由普通执行、程序、几何补丁、观察与模型重决策 | 本轮默认只读审计 |
| `agent_speedup_v3_operation.py` | 对象绑定、真实开闭事件、抓持/释放和恢复预算 | 已有能力先核验 |
| `agent_speedup_v3_program.py`、`fragments.py` | 语义程序及自适应片段 | 状态/恢复候选才改 |
| `agent_speedup_v3_observation.py` | 标定几何的观察候选 | 必须审计完整目标/旧见证依赖 |
| `agent_speedup_v3_perception.py`、`visual_watch.py` | 双目/时序局部估计与外观有效性 | 无有效新证据不能宣称恢复 |
| `agent_speedup_v3_local_geometry.py`、`interaction.py` | 局部运动检查与视觉交互证据 | 不将估计当物理接触真值 |
| `agent_speedup_v2_user_check_sim.py` | 公共可观测接口与独立仿真评估 | 使用冻结 v7 版本；当前工作树存在用户修改 |

## 历史 v7 的重建

基线来源：`runs/agent_speedup_v2_user_check_libero_10_t1_001`。
`user_check_review.json.runtime_sha256` 除配置外共有300个文件；全部可在 `frozen_source/` 平铺目录按 basename 找到，300/300 SHA256 一致，basename 无冲突。含6个 helper runner、292个其他 Python/相关文件以及审核 HTML 和 robot facts；实际完整清单以 JSON 为准。

平铺目录本身不是可独立运行的包布局。按原清单 relative_path 用普通复制重建 `scripts/` 和 `src/gpt_field/`，随后将 `PYTHONPATH` 指向该副本的 `src`；所有 helper 通过其 `__file__` 派生 ROOT，因此正确目录结构能够切断对当前工作树的模块依赖。不要直接运行平铺 runner，不要使用硬链接。冻结 v7 agent 文件的硬链接计数为23；当前 v3.py 与 t1_012 两处冻结文件则各为1。

v7 policy 为 `agent-speedup-v2-user-check-agent-2`，client/prompt 为 `user-check-agent-first-client-2` / `user-check-agent-first-prompt-2`，审核版本为 `agent-speedup-v2-user-check-7`。v7 t1 历史成功897动作、13请求；416.323秒 evaluator elapsed 未包含单独 prepare，不是本轮完整服务时间。

## 活动运行与授权

两个存活进程1237967/1237968由旧 `agent_speedup_v3_t1_t2_t3_001_launch` supervisor管理，分别为 t2/t3 审核服务。两份 review 都是 v3-1、`status=draft`、`approval_mode=manual_browser_review`，对应执行输出目录不存在：它们正在等待审核，未执行 episode。本轮不批准旧草案，也不停止这些服务。

用户本轮已授权阶段内实现和仿真，旧“每阶段重新确认”不适用。但初始计划代审的具体范围不能扩大：最新 t1_012 委托原文明确只限 `libero_10_t1, episode 0, agent_speedup_v3_t1_012`，且已消费；t7/g5 委托分别绑定旧 gate C1/C2 单轮。更早 t1_003 记录保留“just approve the plan yourself”的 t1 偏好，但仍写单个新 t1 episode，不证明 t1/t5/t7 新初态矩阵已全部授权代审。

本轮拟议 P5 t1/t5/t7 episode1–3、P6 episode4–6，需要用户对每份新计划人工审核，或明确授权 Codex 检查这些条件中 A/B 的每份新图像与计划后分别批准。该权限不应被解释为自动点击或省略检查。独立审计、复制重建、离线验证及已授权代码工作不依赖此答复。

已轻量检查184份去重复运行 summary，以及对应 review 元数据；可读 t1/t5/t7 记录全部 episode0、seed7，没有发现拟预留 episode1–6 的已有运行。该结论限于已读取的运行元数据，不是从模拟器真值推出；未读取任何留出真值。

## 实际 CLI 与计时边界

真实入口支持 `prepare/run/start/preflight/review`；v3 `prepare` 有 `--delegated-review-note`，其说明为明确授权本次草案、不会批准草案。v3 CLI 没有 `--auto-approve`、episode time cap 或 retry 参数；不能猜造900秒参数。v7 `prepare` 有 `--stop-after-steps`，动作数上限不是900秒服务预算。

v3现有 `completion_target(summary, measured_execution_s)` 用 preparation elapsed加外层 execution elapsed；它是300秒报告、不触发停止，不含最终计时报告自身序列化。它未单独计入 Codex 自动审核时间，也不能记录首次 raw 成功到完整收尾间隔。本轮应将单调时钟和900秒有效服务预算作为公共评估工程，不能把自动审核耗时排成人工等待。`decision_s`包含模型管线，状态写入/记录子项可能嵌套，不能直接相加。
