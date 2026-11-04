# UC7.1 transit50 接入基线审计

本审计只读取当前工作区源码，未启动模拟器、未调用模型、未读取凭据。它冻结选中的策略、适配器和传输源码摘要，并不替代最终运行的完整依赖封存。详见 [JSON](uc71_baseline_audit.json)。

## 本次明确采用的配置

用户所称 `transit50` 暂按 **每个控制步平移命令最多 50 mm** 配置：`GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M=0.05`。单段目标距离上限另为 **0.50 m**；两者不可混淆。当前仓库默认是每步 20 mm，后来的 `transit500_cap100` 试验是每步 100 mm，本次均不沿用。此命名解释是本次设置的明确假设，不声称已复现一个缺失的历史 launch manifest。

| 项目 | 本次配置 |
|---|---|
| 策略版本 | `agent-speedup-v2-user-check-7.1` |
| 模型/推理强度 | `gpt-6-astra` / `high` |
| 计费通道 | ChatGPT subscription；无 API-key fallback |
| transit 每步/单段目标 | 0.05 m / 0.50 m |
| contact 每步/单段目标 | 0.005 m / 0.08 m |
| 每步旋转命令 | 0.1 rad |
| OSC XYZ/旋转尺度 | 0.05 m / 0.5 rad |
| 控制频率 | 20 Hz |
| 方向语义 | `external_view_v1`，审批绑定外部视角坐标系 |
| 当前执行器 | fragments-4、agent-4、agent-first-prompt-6 |
| 当前局部规划 | 模型提供最多 4 个障碍盒；2 cm 网格；本配置 5 cm 前视 |

命令限幅不等于实际位移或机械臂速度。每段最多 120 步、每次 1–6 段且总预算最多 240 步，是动作片段预算，不是完整任务终止上限。当前 UC7.1 已包含默认 transit 局部规划及共享障碍缓存；保留这条当前分支，不回退旧冻结版本。

## 历史证据的限制

`output/analysis/uc71_transit50_t00_t01_distances_20261102/summary.json` 仍保留，但其中引用的 T00/T01 两个原始 run/trace 路径当前不存在。`astranomy/` 下未找到对应 transit50 manifest。较早的离线方案把 transit50 写为 0.050 m；现存 2026-11-03 `transit500_cap100` manifest 明确是 0.1 m。因而不把历史分析摘要当作本次 50 mm launch 参数的直接证明。

## 可复用部分与接入缺口

现有 `scripts/evaluate_robocasa_user_check.py` 与 `src/gpt_field/robocasa_user_check_sim.py` 只支持 `CoffeePressButton`。任务名、官方语言校验、成功器标签及 snapshot 中的 `coffee_machine` 均有专用假设。不能只换 task 参数就启动 `RestockPantry`。

旧适配器使用 PandaOmron 的 arm-only 模式：base 零速度、torso 保持 reset 位置，12 维原生控制经已检查的索引接收 7 维手臂动作，模式槽为 -1。原生宏因 PandaOmron 几何未验证而禁用；模型持续控制、短动作和 monitored fragments 可以复用。它的旧默认安装路径和 `RoboCasa-0.2-arm-only` 标签也不能代表新 RoboCasa365 环境。

可复用的数据边界为两幅 512 RGB、state8、七个手臂关节的位置/速度、同步相机标定及固定夹爪几何。物体位姿、深度、分割、接触身份、reward、成功谓词及 physics snapshot 留在 evaluator。正式结果只使用新任务的官方 `_check_success()`。

## 原生环境必须检查的契约

1. 固定 RoboCasa/robosuite commit 与资源摘要；核对实际 import 路径、MuJoCo、渲染器和资源可用性。
2. 核对实际机器人/夹爪的 body、site、相机名、腕部安装位置与旋转、图像翻转约定。旧几何 schema 对 Panda 的固定事实进行严格比对。
3. 核对 live OSC 为 world-frame delta、6 维、输入 [-1,1]，输出 XYZ ±0.05 m、旋转 ±0.5 rad；核对实际 controller-part 索引，不硬猜 base/torso 槽位。
4. 保证 RGB、state8、关节及相机外参来自同一控制端点；reset 指纹应包含官方任务随机化元数据。
5. 移除 coffee 专用 snapshot，按 `RestockPantry` 的官方语言和成功器接线；检查任务真值没有流入规划或执行请求。
6. 若扩展移动底盘/torso，单独声明接口与观测，并验证 world arm target、base-local velocity、torso absolute/delta 的不同单位和语义。

移动底盘会改变机器人安装的外部相机位姿。UC7.1 的审批 `view_frame` 冻结语言方向，而每帧标定用于当下视觉度量，不能随 base yaw 偷换已批准的空间含义。现有 transit 规划仅检查夹爪参考点路径，并不验证整臂、底盘 footprint 或完整携物几何。arm-only 接通一项任务也不代表覆盖 RoboCasa365 全部任务。

旧 runner 的 `prepare` 会 reset 模拟器并进行一次订阅模型规划调用，因此不能把它当作本轮离线 preflight。此次用户要求设置完成后再批准测试；本审计没有消耗任何测试授权。
