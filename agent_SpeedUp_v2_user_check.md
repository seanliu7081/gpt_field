# speed up v2 user check

当前版本：`agent-speedup-v2-user-check-7`。模型持续负责任务，通用动作片段负责连续闭环执行。
初始计划可由用户在浏览器中逐项选择、审核，也可在明确授权后使用 `start --auto-approve` 自动确认；
自动确认单独记录，`human_reviewed_plan=false`。执行和自动调整使用 `gpt-6-astra` / `high`。
原始 v2、Astranomy 和已完成运行的冻结代码不变。本版本没有任务编号、对象名称或预设任务坐标分支。

## 执行方式

1. `start` 按指定任务和 episode 重置，模型生成完整初始计划。准备阶段不执行策略动作。
2. 浏览器显示初始图像、对象、步骤、约束及空间目标。用户通过选择控件修改并检查每一步。
3. 用户确认后，审批绑定任务、初始观测、代码及运行配置，只能使用一次。
4. 模型进入连续的“观察 → 短动作或可选宏技能 → 实测反馈 → 再观察”循环。
   已审核 JSON 表达目标、对象身份及约束，执行顺序和具体方法可以由模型根据新观察调整。
5. 模型能直接完成接近、抓取、搬运、对齐、释放和重新观察，不要求先通过旧验证器才能继续运动。
   只有选择某项宏技能时，才需要满足该宏技能自身的执行条件；宏技能不可用会把原因交回模型。
6. 模型关于抓持或完成的判断是带有来源的信念，不会伪装成测量证明。
   最终成功仍由原有独立 LIBERO 评估器判定。

这版没有把旧的 `verify_grasp → resume_macro` 作为必经路径。
“未验证”“观察器不可用”和“已测得失败”分别保留；未知不自动等同于物理失败。
原有控制器和测量仍可为可选宏技能提供证据，但不接管模型的整个任务决策。

## 模型输入与动作

版本 7 新增 `type=execute`，模型一次可委托 1–6 个通用动作片段，不依赖审核步骤编号：
`move_to_pose`、`move_relative`、`approach`、`set_gripper`、`wait`。
执行器每步依据实测状态重算 OSC 命令，到达、夹爪稳定、停滞、预算结束或指定观察点时返回。
后续片段仅在前一片段的测量完成条件满足后开始；观察点后的片段不会自动执行。

每段包含 `skill`、`xyz_m`、`quaternion_wxyz`、`gripper`、`profile`、`max_steps`、`observation_after`。
坐标均为 world；相对目标在该片段开始时根据**实际位姿**解析一次。
移动片段保持夹爪命令；开闭必须使用独立 `set_gripper`，不隐式释放或重新抓取。
`transit` 目标距离最多 30 cm、旋转 1.2 rad；`contact` 为 8 cm、0.5 rad，
每步平移分别最多 2 cm、5 mm，旋转最多 0.1 rad。每段最多 120 步，总预算最多 240 步。
这些是设计边界，不代表路径已验证无碰撞。模型需要根据当前 RGB 选择路线及观察点。
本地位姿到达、夹爪稳定均不等于抓持成功或任务完成。

共享状态分别保留实测机器人数据、夹爪命令、带来源/时刻的模型判断和原生控制器估计。
新的动作片段从当前状态直接接手，不要求旧宏的全场景校准/注册，也不自动回到原抓取位置。
旧宏仍按需执行其自身校准；约 190 步是名义准备成本，60 步授权暂停不表示失败。
反馈保留已完成、当前和剩余准备阶段，模型可选择继续同一暂停宏。

`agent_stats.json` 的 `fragment_actions`、`direct_actions`、`native_macro_actions` 分别记录实际控制步数；
`execution_kind` 区分 `framework_fragment`、`agent_owned` 和 `native_macro`。
应结合任务完成时间和结果判断加速，初始化动作不能当作抓取/放置贡献。

执行模型保留最近六组图像以及连续的文本、动作反馈和推理历史，接收当前两路原生 RGB、实测末端位姿与夹爪状态，
以及同一传感器时刻的七个机械臂关节位置、速度，Astranomy 的相机标定和固定夹爪几何。
腕部相机位姿从同一幅 RGB 对应的 state8 推导，
避免把较晚的模拟器端点与旧图像混用。现有 v2 的 state8 序列化与初始观测散列保持不变。

模型可以给出方向、距离、高度和世界坐标，并通过 Astranomy 的短目标接口执行。
每个直接目标重复执行 1–5 个控制步，再取得新图像及实测反馈。
末端位置采用世界坐标 `grip_site`；姿态采用 `right_hand` 的 `[qw,qx,qy,qz]`。
每个直接目标距当前实测位置最多 5 cm、姿态变化最多 0.35 rad；
每个底层 OSC 步的命令限制仍为 2 cm / 0.1 rad，控制频率 20 Hz。
目标执行结束不表示已经到达目标，夹爪闭合也不表示已经确认抓住物体。

宏技能只在模型明确选择后运行，并受动作预算与返回边界约束。
宏技能的失败、受阻、预算结束和实测结果都交回同一个模型会话。
模型可以换方法、缩短动作、改变观察位置，或直接继续完成该子任务。
该方式保留快速执行片段，同时允许模型持续处理那些无法由固定几何规则可靠判断的情况。

输入不含模拟器物体位姿、深度、接触真值、任务谓词或评估结果。
相机标定使用独立的 `agent_geometry` 操作；原有 evaluator-only `geometry`、
`truth`、`answer_key` 保留用于评估与终止后的诊断，不进入模型请求。
碰撞监视和成功判定沿用 v2，短动作幅度限制本身不是无碰撞保证。

## 审核与结果

审核页面继续提供 `parts`、`steps`、`do`、`obj`、`grip`、`ref`、`sense`、
`keep`、`until`、`hold_true`、`regions`、`placement_goals` 的选择式编辑。
对象和效果描述来自本轮草案，图像格点不是用户编辑项。
计划表单允许所有已有 schema 意图，包括 `slide`、`place_in`、`push_to`；这些意图由模型直接控制执行。
可选原生宏仍只有 `hold`、`turn`、`place_on`、`release`。审核检查语义与引用，宏调用另行检查其实现能力。
空间目标描述与自然语言任务共同进入模型上下文，不再要求旧整物体验证器签发后才允许继续。

`review.json` 保存草案、选择、逐步审核与单次领取记录；
`approved_plan.json` 始终保存用户审核的原始版本。
`user_check_framework.json` 在执行中更新当前模型控制状态与反馈，
请求日志保存持续会话，`trace.jsonl`、`episode.mp4` 与终止后的评估保留完整执行记录。
已审核的旧版本审批不能用于新版代码，历史 `_001`–`_006` 不覆盖。

这个方法仍属于有人类初始审核的评估，不是纯自主零样本基线。
宏技能加速效果应按完整任务是否成功及端到端耗时判断，不能仅按模型调用次数判断。

## 本轮真实任务

版本 7 使用 `libero_10_t3` episode 0 的 `_003` 和 `libero_10_t6` episode 0 的 `_008`，
明确授权后自动确认模型计划，串行运行以避免资源竞争；模型、初态和独立成功标准沿用前轮。
只执行一次六项接口测试，无模拟 smoke 或额外任务队列。
可单独启动一个新目录：

```bash
cd /workspace/gpt_field
env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=3 PYTHONPATH=src \
  .venv/bin/python -u scripts/evaluate_agent_speedup_v2_user_check.py start \
  --task libero_10_t3 --episode-index 0 --auto-approve \
  --review-dir runs/agent_speedup_v2_user_check_libero_10_t3_003_review \
  --output runs/agent_speedup_v2_user_check_libero_10_t3_003
```

以下为版本 6 的历史手动审核启动示例：

不重复运行 smoke 或测试套件；实现后直接启动一轮真实 t06，等待用户审核。
使用新的 `_007` 目录及本实例原来的 GPU 3 渲染配置：

```bash
cd /workspace/gpt_field
env -u CUDA_VISIBLE_DEVICES MUJOCO_EGL_DEVICE_ID=3 PYTHONPATH=src \
  .venv/bin/python -u scripts/evaluate_agent_speedup_v2_user_check.py start \
  --task libero_10_t6 --episode-index 0 \
  --review-dir runs/agent_speedup_v2_user_check_libero_10_t6_007_review \
  --output runs/agent_speedup_v2_user_check_libero_10_t6_007 \
  --host 127.0.0.1 --port 8765
```

命令请求真实模型草案后打印审核链接。通过 IDE Ports 或 SSH 转发本地 8765 端口，
打开包含会话令牌的完整链接；逐项确认后自动执行。
运行期间不再要求用户逐步批准动作，也不接受在线修改已领取的审核计划。
已存在的目录不能重用；需要重跑时换新编号。

长期运行使用独立 supervisor 配置，实例管理服务不受影响。
本轮配置位于 `runs/agent_speedup_v2_user_check_libero_10_t6_007_launch/supervisord.conf`。
请求停止运行可在输出目录创建 `operator_stop.request.json`，
执行器在完整动作反馈边界保存状态并结束；不要删除运行目录来停止任务。

## 主要入口

| 文件 | 职责 |
| --- | --- |
| `scripts/evaluate_agent_speedup_v2_user_check.py` | 草案、审核、单次执行及独立结果标识 |
| `src/gpt_field/agent_speedup_v2_user_check.py` | 审核校验、版本与一次性审批绑定 |
| `src/gpt_field/agent_speedup_v2_user_check_agent_client.py` | 持续模型会话及任务控制请求 |
| `src/gpt_field/agent_speedup_v2_user_check_agent.py` | 模型主导策略、短动作与可选宏技能调度 |
| `src/gpt_field/agent_speedup_v2_user_check_fragments.py` | 独立通用片段、测量闭环、阶段完成与交接 |
| `src/gpt_field/agent_speedup_v2_user_check_sim.py` | 保留 v2 评估的独立相机几何扩展 |
| `src/gpt_field/agent_speedup_v2_user_check_client.py` | 初始审核草案与既有计划/记录适配 |
| `src/gpt_field/agent_speedup_v2_user_check_server.py` | 本地审核服务 |
| `src/gpt_field/agent_speedup_v2_user_check.html` | 选择式审核界面 |

其他 user-check 技能、空间与约束模块保留，按可选宏技能的需要复用。
版本 7 的加速效果以本轮真实任务结果为准。
