# uc7.1 + Astra high 测试协议

版本：1.0；日期：2026-10-28。适用于 `agent_speedup_v2_user_check` 的冻结版本 **uc7.1（版本 A，包含 A9）**。

本协议记录已确认的测试方式，不构成新的任务启动或重试授权。当前 LIBERO-10 测试采用重排布局；每次完整任务开始前，先展示原始与新布局对比，再取得用户对具体任务和次数的批准。写文档、准备预览、生成规划草稿与执行完整任务分别记录。

依据：[最终实施方案](reports/uc7_direction_semantics_20261028/final_plan.md)、[uc7.1 交付记录](reports/uc7_direction_semantics_20261028/implementation_uc71/README.md)、[当前测试入口](scripts/evaluate_agent_speedup_v2_user_check.py)。后续用户明确修改的要求优先，变更必须记录在对应运行中。

## 1. 固定测试配置

| 项目 | 要求 |
|---|---|
| 框架版本 | `agent-speedup-v2-user-check-7.1` |
| 方向语义 | `external_view_v1` |
| 模型 | `gpt-6-astra` |
| 推理强度 | `high`；初始规划和执行均相同，不使用 `xhigh` 或静默降级 |
| 推理通道 | ChatGPT subscription / OAuth；`api_key_calls=0`，不切换 API-key 计费路径 |
| 审核属性 | 保留框架的 `human_reviewed_plan=true`、`zero_shot=false`；实际审批方式另行如实记录 |
| 主入口 | `scripts/evaluate_agent_speedup_v2_user_check.py` |
| 任务编号 | `libero_10_t0` 至 `libero_10_t9`，采用从 0 开始的编号 |
| 初始状态 | 默认官方初始状态索引 `0`、seed `7`，在其基础上生成并冻结自定义布局 |
| 模拟器 | `/workspace/heading_flow_pub/third_party/LIBERO`；Panda / `OSC_POSE`，20 Hz |
| Python | 主进程 `/workspace/gpt_field/.venv/bin/python`；模拟器 `/workspace/venvs/libero-plus/bin/python` |
| 图像 | 外部相机和腕部相机，原生 512×512；保留冻结版本的图像变换 |
| 重置静置 | 10 次环境准备动作；不计为任务策略动作 |
| 视频 | 启用 `--planning-video`，1600×960、20 fps、XY/XZ/ZY 和内嵌解释字幕 |
| 请求诊断 | `--request-deadline-s 180 --stream-progress` |
| 完整任务上限 | 不设置任务总时间、动作数或调用数上限，不使用 `--stop-after-steps` 或 smoke 模式 |
| 每次授权 | 默认一个指定任务、指定布局、一次完整尝试；额外重试需再次批准 |

模型由 runner 的配置函数提供，CLI **没有 `--model` 参数**。启动前及实际请求中都要核对 `gpt-6-astra` 与 `reasoning.effort=high`，不能仅根据命令名称推断模型。

180 秒是认证完成后，单次 HTTP 打开与完整 SSE 流的截止时间，**不是整个任务的时间限制**。底层 socket 超时与该截止时间不同。局部运动片段的步数限制同样不是 episode 上限。

## 2. 版本冻结与允许的输入

冻结依据为 [uc71_release_manifest.json](reports/uc7_direction_semantics_20261028/implementation_uc71/uc71_release_manifest.json) 和 [源码快照](reports/uc7_direction_semantics_20261028/implementation_uc71/uc71_release_source.tar.gz)。快照 SHA-256：

```text
b3e8f84f3bf1b7603fad0ae0d8c4bee8a5a125b50ab08e320394b4069af57a68
```

启动前核对 manifest 中的运行文件摘要，结束后再次核对。保存本次 runner 的完整 `runtime_sha256`，包括 `@configuration`；不只记录 git commit，因为基线包含未提交文件。运行中不改源码、提示词、模型强度、执行参数或布局。发现不一致时先解释差异，不能直接更新摘要使其通过。

本版本包含外部相机方向计算、规划提示、每次执行请求的方向契约、求解器投影、审批冻结、审核标签、腕部图像文字修正和 A9 显示高亮。**不加入 E1 选点偏好、E2 实际腕图旋转或 uc8 提速改动。** `bounds=null` 的方向区域仍表示对应一侧，不自动缩成任务目标框。

模型使用冻结框架允许的 RGB、机器人状态/关节、相机与夹爪几何、任务文字、审核计划和执行历史。世界方向从外部相机标定计算，并保存在 `preparation.view_frame`、绑定审批，执行请求持续携带同一方向。腕部图像不用于重新定义左右。

BDDL 数值目标框、模拟器物体真值、碰撞/支撑真值和成功判据仅用于环境准备或独立评估，不作为额外答案送给模型。不要套用旧 Hybrid 的“无标定 RGB-only”输入协议，也不添加演示或人工动作轨迹。

旧 uc7/v3 复现使用[改动前快照](reports/uc7_direction_semantics_20261028/implementation_uc71/uc7_baseline_runtime/)，不能从已升级的共享工作区直接运行旧版本并称为基线。

## 3. 布局生成与预览

1. 为任务和尝试创建新目录，保留已有成功、失败和中止记录，不覆盖历史试验。
2. 从官方初始状态生成自定义状态。默认只移动自由物体 XY，保留机器人、固定物体、关节、物体 Z、朝向及速度的原始输入。保存原始和自定义 `.npy`、修改索引与坐标、官方 BDDL 及摘要。
3. 保留官方任务语言和目标逻辑。注意目标区可能固定在场景中，不能假定它会随被移动的锚点自动移动。Task 8 保持炉子初始开启；Task 9 保持微波炉门初始打开。
4. 用实际模拟器重置检查：初始目标为 false，物体稳定且有桌面支撑，没有初始物体间、机器人或非桌面固定物体接触，没有明显穿透或位姿漂移。采用现有布局准备 helper 的检查并保存结果，不把这些准备检查变成新的策略否决规则。
5. 至少重复重置两次，确认初始观测摘要一致；保存 `.npy` 的文件 SHA-256 和 `initial_observation_sha256`，二者不可混用。
6. 渲染实际外部与腕部图像。对比图明确标注 **左侧原始布局、右侧新布局**，列出移动对象与位置。图像来自真实重置，不使用生成图或人为移动图像内容。
7. 目视检查目标可见性，说明腕图裁切等限制。初始无碰撞不代表后续开门、抽屉或机器人运动路径已经被证明无碰撞。
8. 向用户展示对比图，然后取得本次任务启动批准。布局、计划或实质执行设置发生变化时，重新展示变化并取得对应批准。

现有参考：[Task 8/9 布局 helper](runs/agent_speedup_v2_uc71_relayout_t8_t9_20261028_001/prepare_layouts.py)、[Task 8 对比](runs/agent_speedup_v2_uc71_relayout_t8_t9_20261028_001/task_8/layout_comparison.png)、[Task 9 对比](runs/agent_speedup_v2_uc71_relayout_t8_t9_20261028_001/task_9/layout_comparison.png)。这些历史产物不是新任务的自动授权。

## 4. 规划、审核和审批

### 4.1 仅准备规划

`preflight` 不读凭据、不调用模型、不启动模拟器。`prepare` 重置环境并生成规划草稿，但不执行策略动作，也不批准任务。默认每个新布局请求一次 Astra high 规划；保留原始输出和验证结果，不能反复调用模型挑选有利草稿。

核对官方任务、目标物体、图像指向、步骤、约束和计划合法性。不同布局必须重新生成与其观测绑定的计划，不能把旧布局的格子位置直接搬过来。

允许在同一冻结版本、同一观测、同一布局和同一执行设置下复用已有草稿；记录来源 review ID、文件/计划摘要和已发生的规划调用。复用草稿表示本轮新增规划调用为 0，不表示这份计划从未产生模型成本。

### 4.2 审核修正

修正必须走既有审核允许的编辑范围，保留原始模型输出、逐字段差异、修正后校验结果，并在任务批准前说明。不得修改校验器以放行计划。

已知 Task 9 例子：模型可能给 `do=turn` 填 `sense=close`。JSON Schema 接受这个枚举值，但现有审核语义校验要求非 `slide` 的 `sense=any`。可单独提出 `close → any` 的字段修正；官方“放入杯子并关门”的任务要求仍然保留。该原始草稿仍计为原始验证失败，不能把修正后的有效性回写为模型原稿成功。

### 4.3 每次尝试的授权

- 记录用户原文、任务编号、布局/观测摘要、计划摘要、版本、模型、次数和批准时间。
- 用户可以明确一次批准多个指定任务；只批准 Task 8 不包括 Task 9。批准后的同一尝试不重复询问。
- 浏览器审核与聊天中委托批准均需如实记录。聊天批准不得标注为 `manual_browser_review=true`。
- 每次执行使用新的审批记录和输出目录。审批在模型首个执行请求之前被消费；即使随后零动作就失败，也不能复用已消费的审批。
- 复用原始草稿时，可建立独立执行 review，保留原始草稿不变，记录来源，并将准确的审批方式纳入新的上下文绑定。
- 冻结发布 manifest 中的“当时尚未授权”是历史事实；后续实际授权保存在对应 run 中，不改写历史发布记录。

## 5. 命令模板

以下模板仅适用于 **LIBERO**。先完成布局准备，将 `TASK_DIR` 指向新目录；其中必须已有经过验证的 `initial_state.npy` 和 `preview_manifest.json`。不要把历史已消费的 review 当作新 review。

```bash
cd /workspace/gpt_field
TASK_KEY="libero_10_t8"
TASK_DIR="/workspace/gpt_field/runs/REPLACE_WITH_NEW_CAMPAIGN/task_8"
EXPECTED_SHA="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["expected_initial_observation_sha256"])' "$TASK_DIR/preview_manifest.json")"

env -u OPENAI_API_KEY -u CUDA_VISIBLE_DEVICES \
  PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python scripts/evaluate_agent_speedup_v2_user_check.py preflight \
  --task "$TASK_KEY" --episode-index 0 \
  --initial-state-override "$TASK_DIR/initial_state.npy" \
  --expected-initial-observation-sha256 "$EXPECTED_SHA" \
  --planning-video --reasoning-effort high \
  --request-deadline-s 180 --stream-progress \
  > "$TASK_DIR/preflight.json"
```

核对 preflight 后，只生成草稿：

```bash
env -u OPENAI_API_KEY -u CUDA_VISIBLE_DEVICES \
  MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=3 \
  PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  .venv/bin/python scripts/evaluate_agent_speedup_v2_user_check.py prepare \
  --task "$TASK_KEY" --episode-index 0 \
  --review-dir "$TASK_DIR/review" \
  --initial-state-override "$TASK_DIR/initial_state.npy" \
  --expected-initial-observation-sha256 "$EXPECTED_SHA" \
  --planning-video --reasoning-effort high \
  --request-deadline-s 180 --stream-progress \
  > "$TASK_DIR/prepare.log" 2>&1
```

`MUJOCO_EGL_DEVICE_ID=3` 是本机既有设置，换机器需在冻结前验证实际渲染设备。模型通道保持 subscription，不打印或保存 token。

如果使用浏览器审核，在单独终端运行以下命令；逐项查看并批准后才可以执行。服务只绑定 loopback，需要远程访问时使用 SSH 转发。

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_agent_speedup_v2_user_check.py review \
  --review-dir "$TASK_DIR/review" --host 127.0.0.1 --port 8765
```

**只有获得本次任务批准并存在有效 approved review 后**，单次执行入口为：

```bash
/workspace/gpt_field/.venv/bin/python -u \
  /workspace/gpt_field/scripts/evaluate_agent_speedup_v2_user_check.py run \
  --review-dir "$TASK_DIR/review" --output "$TASK_DIR/trial"
```

该命令应由独立 supervisor 的单次启动 wrapper 执行，沿用准备时的环境并删除 API-key 环境变量；`run` 恢复已审批的模型强度、视频及传输设置。不要绕过配置匹配强行启动。`start --auto-approve` 会连接准备、批准和执行，**不能当作仅准备命令使用**。

## 6. 运行监督、停止与错误处理

每个已批准尝试使用独立 supervisor 配置：`autostart=false`、`autorestart=false`、`startretries=0`。wrapper 使用排他锁及只创建一次的 attempt 标记，核对绑定文件并拒绝已存在的输出目录。不要让服务管理器自动重启任务，也不要运行未获批的后续队列。

Unix socket 使用简短且唯一的本地路径，例如 `/tmp/uc71-t8-ATTEMPT.sock`，权限 0600。较长 run 路径可能在 supervisor 添加临时后缀后超过系统限制。此类**尚未启动任务**的服务配置修正要保存原始失败日志、差异和更新后的绑定；不能把已开始的失败任务当作“尚未尝试”。

正常结束依据官方 LIBERO 成功；模型说“完成”不能代替成功信号。现有碰撞监视为 **record-only**，不会自动终止任务或新增成功否决；报告观察到的碰撞，不宣称无碰撞保证。保留框架现有的计划冲突、基础设施错误和用户中止路径，不新增停滞/15 分钟等自动 episode 截止条件。

用户要求停止时通过既有 `operator_stop.request.json`/运行状态提供的停止方式，在完整反馈边界结束并保留未完成结果。运行中应持续说明有证据的进展或错误；长时间没有回复不等于在等待用户审批。

### 请求重试与任务重跑必须区分

冻结客户端已有同一请求的有限重试，不能笼统写成“所有自动请求重试都关闭”：

| 类别 | 冻结行为 |
|---|---|
| `server_is_overloaded`，HTTP 503 或无 HTTP 状态 | overload 类最多 2 次重试，即该类单独出现时最多 3 次发送 |
| `stream_missing_completion`、`stream_truncated_event`、`subscription_stream_interrupted`、`subscription_network_error` | transient 类合计最多 2 次重试，不是每个错误码各 2 次 |
| HTTP 520 / `unknown_error`、`request_deadline_exceeded` | 不属于上述可重试码；本次尝试按错误结束 |
| supervisor/整个任务重跑 | 不自动进行，另需一次明确任务批准和新的执行 review/output |

请求重试仅适用于尚未提交响应、观测没有推进的同一请求；不丢弃已经收到的有效答案再重新采样。两个重试类别独立计数，混合错误时不能宣称总发送数一定不超过 3。每次退避等待最多接受 60 秒，实际以日志和冻结客户端为准；每次发送、错误与等待均保留。任何重试均不切换模型或 API-key。

错误结束后保存原始记录并关闭私有进程。已批准的其他任务是否继续，按用户明确批准的范围执行；本协议不补充授权。

## 7. 成功判定与验收

同时报告以下字段，不用一个“SUCCESS”覆盖所有含义：

| 层次 | 依据与含义 |
|---|---|
| 官方任务结果 | `summary.task_success`、`summary.libero_success`；沿用官方任务判据 |
| 终止瞬间严格物理状态 | `physical_truth.physical_success_at_terminal`、失败检查及终止物体状态 |
| 框架严格结果 | `summary.v2_success`，保留冻结的严格真值规则 |
| 结束后诊断 | `physical_success_after_settle` 等；与实际终止状态分开报告 |
| 方向检查 | 冻结方向契约、模型目标和实际轨迹是否一致；方向正确本身不是任务成功 |

`v2_success` 不是简单等同于“结束后静置成功”。现有唯一相关例外是：所有失败子检查都属于已定义的持握及其后果，并满足 `terminal_failure_is_hold_only`，且诊断释放静置后的严格和 raw 判据都通过。支撑失败不属于该例外。直接采用 runner 保存的结果，并给出终止/诊断两份证据，不人工重算为成功。

官方目标有可能在物体仍被夹持时成立，runner 会按原有官方规则停止。需要明确写“官方成功、严格物理未通过”，不能写成完整物理完成，也不能自动补执行动作后覆盖原结果。程序退出码 0 只表示 runner 的评估完成条件，不证明 `v2_success=true`。

版本 A 验收要求实际任务成功，不能只因方向正确而把其他原因的失败算作通过。当前用户逐项批准的 relayout 测试，与[最终方案](reports/uc7_direction_semantics_20261028/final_plan.md)中的 V3 官方布局/定向场景矩阵分开统计；单次或少数 relayout 成功不代表完成全部 sweep。官方布局、g5、RoboCasa 扩展仍需各自的明确授权及适用 runner。

## 8. 时间与模型调用统计

### 8.1 任务完成时间

```text
task_completion_time_s = summary.control_without_video_s
                       = summary.control_elapsed_s
                         - summary.timing_components.video_render_s
```

只有 `completion_time_eligible=true` 且官方任务成功时才报告为“任务完成时间”。它包含执行期间模型等待、局部计算、模拟器动作及控制循环记录开销；不包含初始规划、审批等待、环境启动/重置、结束后的物理诊断及清理。保留原始控制时长与视频渲染时长，注明这是记录开销的回溯扣除。

失败、用户中止或零动作接口错误的完成时间写 **N/A/null**；其已消耗时间只能标为未完成尝试时长。视频播放时长和 `steps / 20` 的模拟时间都不能代替任务完成时间。只有完成的、版本及任务/布局/配置可比的样本才能比较完成速度，不将未完成基线用于提速率。

### 8.2 模型调用

分别记录：

- 初始规划发送数、有效/无效草稿数。
- 本次执行请求实际发送数，包含失败和有限重试；按 `agent/replan/repair/...` 分项。
- 被接受的执行回复数、错误数、重试次数及等待时间。
- 本次新增规划调用与复用的历史规划调用。
- `billing`、`api_key_calls`、服务端返回的 token 用量和 unknown-usage 次数。

以 `requests/` 记录、`summary.model_calls.requests_sent` 和 budget 交叉核对。**不要只读 `run_state.model_calls`**：第一次请求就失败时，动作诊断驱动的计数可能仍为 0，实际已有一次请求发出。无服务端用量的零 token 字段不证明未消耗额度。

建议同时报告：`本次新增请求 = 本次规划发送 + 本次执行发送`；`本任务使用的请求 = 本次新增请求 + 复用的历史规划请求`。汇总整个 campaign 时，历史规划请求按请求 ID 去重，不能把同一份复用草稿重复计入实际新增调用总数。

## 9. 视频与产物要求

视频格式沿用 `runs/agent_speedup_v2_user_check_libero_10_t7_001` 的仪表板，开启 `--planning-video`。必须有 **XY、XZ、ZY** 三个投影和内嵌短说明，解释箭头、末端、目标与轨迹。

当前显示的是由活动片段目标推导的**目标误差控制场**，不是障碍物距离场、完整规划代价场或碰撞安全证明。字幕应保留这一限制。蓝箭头表示受限的目标方向运动，白点为夹爪，黄色十字为目标，粉线为实测轨迹；黄色虚线表示片段入口至目标的参考线。终止时没有活动目标，不应继续画旧目标箭头。API 等待不按真实等待时长播放，视频时长不作耗时指标。

任务结束后用 ffprobe 核对分辨率、帧率、帧数和时长；查看活动帧及末帧。零动作错误只会产生初始/终止画面，应明确说明没有执行轨迹。

每次运行至少保留：

| 类别 | 产物 |
|---|---|
| 布局 | 原始/自定义状态、原始任务 BDDL、`layout_changes.json`、实际预览及对比图、`physical_validation.json`、`preview_manifest.json` |
| 规划 | 原始 request/response/result/stream、草稿、初始观测、方向坐标系、任何审核差异与来源摘要 |
| 审批 | 用户原文与范围、review、计划/上下文/运行文件绑定、`approval.json` |
| 启动 | 单次 wrapper、supervisor 配置、`launch_manifest.json`、`attempt_claimed.json`、`launch_state.json` |
| 执行 | `trial/summary.json`、`run_state.json`、`trace.jsonl`、完整请求日志、框架/计划历史、源文件与输入快照 |
| 评估 | 官方结果、`physical_truth.json`、时间/调用汇总、视频与核验记录 |
| 结束 | 源码完整性和结果核对、退出码、进程/socket 清理记录；保留失败产物 |

对外结果至少包含：任务与布局、版本/模型、官方成功、严格物理结果、完成时间或 N/A、执行请求数与规划调用数、视频链接、未完成原因及是否重试。

## 10. 已有记录示例与当前边界

以下仅作为正确报告口径的例子，不授权继续执行：

- [uc7.1 Task 6 relayout](runs/agent_speedup_v2_uc71_relayout_t6_20261028_001/README.md)：官方成功，严格物理未通过；完成时间 604.86 秒，14 次执行调用加 1 次复用规划。诊断释放静置通过没有覆盖终止结果。
- [uc7.1 Task 8 首次尝试](runs/agent_speedup_v2_uc71_relayout_t8_t9_20261028_001/task_8/README.md)：HTTP 520，1 次失败执行请求、0 动作；复用 1 次规划，完成时间 N/A。再次尝试需要新批准。
- [Task 9 准备](runs/agent_speedup_v2_uc71_relayout_t8_t9_20261028_001/README.md)：布局与修正候选已准备，尚未获准启动。

截至本协议日期，V0 为 350 通过、59 个原有失败，无新增失败；V1 原始草稿合法性为 12/13，Task 9 原始格式问题仍如实保留。不能称整个回归、V1 或完整验收已全部通过。后续结果写入新的运行记录，不改写历史证据。
