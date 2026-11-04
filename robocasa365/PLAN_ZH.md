# UC7.1 transit50 接入 RoboCasa365 的计划

日期：2026-11-04（北京时间）。范围：先准备独立环境、通用移动接口和首个 `RestockPantry` 试验，再由用户批准测试。当前不启动任何真实模型或仿真任务。用户已选择支持底盘和升降柱。

## 1. 方法与范围

保留 agent SpeedUp v2 UC7.1 的持续任务会话、世界坐标机械臂目标、实测反馈片段执行、局部 transit 障碍规划、观察点、独立夹爪动作和 UC7.1 方向语义。新增 `RoboCasa365` 的传感器/控制器适配层及 `mobile` 指令。原 `src/`、原 runner、LIBERO 结果和旧 RoboCasa 安装不改动。

这是 **UC7.1 的 RoboCasa 移动适配版本**，不是声称原 LIBERO 版本无需修改就能完成 365 项任务。目录清单覆盖365项，当前运行入口只开放首个 RestockPantry；后续任务仍需验证与授权。

| 项目 | 本次固定值 |
|---|---|
| 模型、推理与计费 | `gpt-6-astra` / `high` / ChatGPT subscription；无 API-key 或模型回退 |
| UC版本 | `agent-speedup-v2-user-check-7.1`；冻结当前源码摘要 |
| transit50 | 每控制步平移命令范数最多 **0.05m**；单段目标距离最多 **0.50m** |
| 接触动作 | 每步最多0.005m；保留原UC7.1接触/姿态边界 |
| 摄像头 | 原生512×512 external-left + wrist RGB；无深度、分割或物体真值 |
| 机器人 | PandaOmron：Panda机械臂、平行夹爪、移动底盘和torso |
| 控制频率 | 20Hz；模型推理期间仿真不前进 |
| 新移动片段 | 原生归一化base3、torso1，分别限±0.2；每次1–5控制步后返回观察 |
| 原生宏 | LIBERO专用宏禁用；保留通用arm fragments和短动作 |

历史T00/T01 transit50清单引用的原运行路径当前不存在，因此不能从旧目录名证明完整参数或位级复现。本次明确固定上述两个距离，避免把50cm目标上限、50mm命令上限及100mm变体混淆。审计见 `setup/uc71_baseline_audit.md`。

## 2. 首个任务的准确含义

用户给出的 `arranging_cabinets/restock_pantry.py` 路径在当前上游已失效。固定版本的实际源码为 [restocking_supplies/restock_pantry.py](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/environments/kitchen/composite/restocking_supplies/restock_pantry.py)。

官方语言为：`Pick the cans from the counter and place them in their designated side in the cabinet.` 初始柜门已打开。任务要把台面的两只罐头放进柜内已有罐头所在一侧；不是开柜任务，也不要求关门。需要观察柜内原有罐头和另一侧干扰物，保持两次抓取的对象身份，并在释放后移开夹爪。

给模型的目标只来自官方语言及本轮RGB。源码中的对象名称、随机左右侧、真值坐标、距离判据和成功布尔值不进入模型。精确评估逻辑留在 `setup/research_report.md`，用于独立评估审计；不得把它当作策略目标坐标或预写动作脚本。

## 3. 已完成的环境准备

- 新目录：`/workspace/gpt_field/robocasa365`。
- 独立Python3.11环境：`.venv/`；RoboCasa1.0.1、robosuite1.5.2、MuJoCo3.3.1和NumPy2.2.5。
- 固定RoboCasa提交 `3443cdf3dd20f9ae8ed41adba442059b506891d0`，robosuite提交 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`。
- 官方默认资产textures、generative textures、fixtures_lightwheel、objects_lightwheel、objaverse已下载；ZIP总计5.213GB，解压11.083GB、64503文件，保留来源、SHA256、长度、CRC记录。
- 未下载演示、训练集和非默认aigen包；未安装不用于本次仿真的lerobot/tianshou训练/数据工具。`pip_check.log`保留这两项依赖缺失声明，环境不是完整训练工具安装。
- `RestockPantry` 类、原生PandaOmron控制器配置的导入检查通过。导入成功不等于渲染或机器人运动已经验证。

安装证据见 `setup/provisioning_report.json`、`setup/requirements-runtime.lock` 和 `setup/assets_manifest.json`。新旧环境隔离，原模拟器和根工作区策略源码均未修改。

## 4. 接口设计与已实现内容

`rc365/sim.py` 在单独进程里运行RoboCasa。普通机械臂动作保持world-frame OSC delta；底盘移动时使用原生 `HYBRID_MOBILE_BASE` 的base-frame desired-goal模式，让手臂跟随底盘。模式切换根据实测位姿重建目标，不写机器人qpos，不传送机器人。回到arm模式后保持最新torso高度。

`rc365/policy.py` 复用UC7.1执行器并增加严格 `mobile` schema。移动与机械臂操作分开，移动时arm delta严格为零，夹爪保留前一命令。新移动动作提供短时原生速度/torso增量控制，**尚不包含自动导航规划或完整机器人避碰**。模型需要根据RGB选择可见路线和重新观察；已有手部transit盒子规划不能当作底盘或全臂安全证明。

每个观察只包含两路RGB、state8、七个机械臂关节位置/速度、机器人底盘/torso实测状态，以及同步相机/固定工具标定。物体位姿、接触标识、reward、任务谓词和真值成功标志留在worker status与独立日志，策略不接收。

`rc365/runner.py` 复用原草案—审核—单次执行流程。审核绑定代码、配置、初始RGB/传感器、物理初态与XML；第二次reset不一致则在执行模型调用前终止。资产按完整内容树摘要绑定，避免把11GB资产复制进每个trial。每次动作保留arm7、移动命令、实际native12和反馈；完整反馈边界支持用户停止。

## 5. 首轮待批准的具体测试

| 参数 | 拟定值 |
|---|---|
| task | RestockPantry |
| 数量 | 一次新试验；不自动重试，不排其他任务 |
| seed | 7 |
| split | `target`厨房/对象分割；不固定/挑选有利layout |
| 场景 | 原生生成，原物体注册表，无演示、手工ROI或oracle动作 |
| 成功 | 只认 `RestockPantry._check_success()` |
| 碰撞 | endpoint contact记录；不声称“无碰撞成功” |
| 停止 | 原始成功、用户停止、基础设施或订阅错误 |
| episode上限 | 探索性首轮沿用无动作/时间上限；局部动作预算不变 |
| 官方参考horizon | 2550，单独记录是否在范围内成功 |
| 记录 | 初态、代码/资产摘要、请求响应、完整trace、视频、终态、计时、清理状态 |

`split=target`表示厨房和对象的分割；RestockPantry本身属于pretrain任务集合，**不属于target50**。该首轮是迁移调试的探索性pilot，不能当作标准target50成绩，超2550步的成功也不能当作官方限步成功。

批准后的顺序：先做同一试验的真实reset并检查控制器类别/尺度/索引、机器人关节轴、工具几何与RGB同步；通过后生成初始模型计划并按现有UC7.1流程审核，再执行一次任务。若运行前兼容性检查失败，保留故障报告，先修复并报告；不暗中切换任务、锁住底盘或使用旧环境。新的额外smoke rollout不是当前已授权内容。

## 6. 从一个任务扩展到365项

| 阶段 | 要解决的问题 | 通过依据 |
|---|---|---|
| A：本轮准备 | 依赖、资产、观测边界、arm/mobile模式切换、审批/记录 | 导入、静态审计、纯fake测试；当前已完成，仍待真实联调 |
| B：RestockPantry首轮 | 看清指定柜侧、两次抓取、移动/升降、放入后退出 | 完整视频/trace及raw success；失败按感知、控制、基础设施分类 |
| C：同任务多初态 | 排除单次偶然成功和单厨房可达性 | 预先固定若干种子/布局，不根据结果删掉困难初态；另行批准 |
| D：技能覆盖 | 抓放、门抽屉、按钮旋钮、容器水/热状态、跨区域移动 | 每类代表任务单独分析，补通用接口而非任务坐标脚本 |
| E：365项矩阵 | 批量复现、版本冻结、错误隔离、统计 | 固定任务×种子×split清单，独立trial与终态，无隐式retry；另行批准 |

`configs/task_catalog.json`静态解析65 atomic +300 composite实际类，合计365。上游dataset registry / all_tasks仅包含317项；不能把这一列表误当成365项。剩余48项的horizon保留 `null`。标准限步评估必须先确定这48项的官方规则，或为它们另列预先声明的探索协议，不能随意补值后混入同一分数。

批量化时先确定所有种子、split和版本，再产生执行清单。成功率同时给任务平均和episode汇总，并公开每项成功/未完成/基础设施错误的数量；基础设施错误单列，不靠重复运行挑选成功。任何改动使用新版本，前后结果不无标记混合。模型调用量和耗时预算用首轮实测估算后再讨论批量资源，不在现在承诺“365项全能完成”。

## 7. 评估指标与计时

主结果保留raw success、控制步、模型调用数、mobile/arm/contact/gripper分类、是否超官方参考horizon、终止原因、碰撞记录。`control_elapsed_s - video_render_s`只在实际成功且完整日志/代码一致时称为完成耗时；未完成只报尝试时长。另列初态准备、模型初始计划、模型等待、本地规划、仿真步进及视频开销。单个任务成功不证明加速；需要固定初态/设置的后续比较才能判断收益。

## 8. 入口与当前边界

可重复运行的无仿真preflight：

```bash
cd /workspace/gpt_field
robocasa365/.venv/bin/python robocasa365/run.py preflight
```

完整使用说明见 `README.md`。当前 `test_authorized=false`，没有真实review目录、trial目录或运行队列；没有真实模型调用、仿真创建、reset、step。离线测试中的假对象不计为真实测试。**准备完成后通知用户，等待测试批准。**
