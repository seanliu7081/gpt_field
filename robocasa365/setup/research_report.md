# RoboCasa365 / RestockPantry 接入核对（2026-11-04）

本文是准备阶段的源码与接口调查，不是运行结果。仅阅读官方网页、下载/固定的官方源码，并以 Python AST 静态解析任务清单；本报告作者没有导入模拟器、创建环境、reset、step、播放 demonstration 或调用模型。依赖/资产实际安装进度以同目录 provisioning 记录为准。

## 版本与来源

RoboCasa365 是 RoboCasa v1.0 之后的版本系列；本次固定 RoboCasa **1.0.1**、commit `3443cdf3dd20f9ae8ed41adba442059b506891d0`，配套官方 robosuite master commit `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`。代码与 assets 各有版本，不能把工作区过去的 RoboCasa v0.2 环境当作 365 环境。官方 1.0.1 曾将任务 horizon 增加 1.5 倍，评估时应读取固定版本配置。[官方版本说明](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/README.md)

用户提供的 `composite/arranging_cabinets/restock_pantry.py` 在当前 main 返回 404；同名正式任务位于 `composite/restocking_supplies/restock_pantry.py`。文档仍将它列在 Arranging Cabinets 类别，目录名称差异不构成另一个任务。[固定任务源码](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/environments/kitchen/composite/restocking_supplies/restock_pantry.py)、[官方任务文档](https://robocasa.ai/docs/build/html/tasks/composite_tasks.html)

## RestockPantry 的自然语言目标

把台面上的两罐食品放进柜内已有罐头的同一侧。官方初始化会把双门柜完全打开；柜内另一侧有一个非罐头物品，台面也有干扰物。已有罐头位于左侧还是右侧会随机变化。任务没有要求先开柜或最后关柜。因此运行时应让模型依据当前 RGB 识别罐头与目标侧，不把位置、对象 ID、随机侧或预写坐标交给策略。官方任务语言来自 `get_ep_meta()["lang"]`；不要把其余 episode 元信息整包传给模型。[固定任务源码](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/environments/kitchen/composite/restocking_supplies/restock_pantry.py)

### 仅供 evaluator / 离线核对的成功条件

以下细节只用于核对原始环境成功判定，不应被导入运行时策略配置、动作规划或模型提示：

- `obj1`、`obj2` 都通过官方 `obj_inside_of(..., cab)`。
- 每一罐到柜内已有罐头的距离乘以 2，严格小于它到柜内非罐头物品的距离；源码使用三维位置距离。
- 夹爪末端 site 到两罐各自物体中心的距离都大于 `gripper_obj_far` 默认阈值 0.25 m。
- 由原始 `_check_success()` 的布尔值统一判断成功，不用模型说“完成”、抓到一罐、放入一罐、视觉相似或自定义阶段代理代替。

来源：[RestockPantry._check_success](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/environments/kitchen/composite/restocking_supplies/restock_pantry.py)、[object_utils](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/utils/object_utils.py)。`obj_inside_of` 检查对象包围盒点与柜内区域；不要将其容差值直接解释成统一的物理厘米容差。

## 环境与控制适配

官方 `create_env` 默认机器人是 `PandaOmron`，包括机械臂、夹爪、移动底座和躯干。默认控制频率 20 Hz。默认相机为 `robot0_agentview_left`、`robot0_agentview_right`、`robot0_eye_in_hand`。Gym 默认 RGB 为 256×256；现有 UC7.1 的图像尺寸、显示方向与 state 边界必须在适配器明确保留，并记录相对于官方接口的差异。[env_utils](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/utils/env_utils.py)、[Kitchen](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/environments/kitchen/kitchen.py)

官方 Gym 动作是结构化字典：末端平移 3、旋转 3、夹爪 1、base_motion 4（底座 3 + 躯干 1）、control_mode 1，共 12 个标量。不能直接将 LIBERO 的 7 维动作交给原始环境。默认右臂 `OSC_POSE` 是 **base frame 的 delta 输入**；归一化输入 ±1 分别缩放为平移 ±0.05 m、旋转 ±0.5 rad。要核对 UC7.1 的世界/底座坐标变换、旋转约定、夹爪符号和模式位，而不是凭数组长度补零。[Gym wrapper](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/wrappers/gym_wrapper.py)、[PandaOmron controller](https://github.com/ARISE-Initiative/robosuite/blob/5ce6643f3092639d08f7b0f90ed1c6a84f50552c/robosuite/controllers/config/robots/default_pandaomron.json)

Gym 观测包含底座位置/四元数、相对于底座的末端位姿、夹爪关节位置、RGB 与自然语言任务；原始环境启用 object observations，适配层必须白名单过滤，不能把对象真值状态整体泄漏给策略。Gym RGB 会上下翻转，直接 raw observation 和 Gym 路径不能重复翻转。Gym 的 `info["success"]` 来自原始 `_check_success()`；不要误把 `done` 等同成功。[Gym wrapper](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/wrappers/gym_wrapper.py)

首个 RestockPantry 试验可用于验证现有手臂控制路径，但即使成功，也不能证明已支持 365 项任务。涉及跨厨房导航、躯干调节、炉钮、开关、抽屉、插入和容器盖的任务应逐类核对能力与动作接口；冻结底座的适配必须明确记录为受限协议。环境初始化可使用官方 reset 规则，策略不得读取其隐藏位置作为导航提示。

## 安装约束

官方建议 Python 3.11，安装 robosuite master 与 RoboCasa 后配置 macros、下载 kitchen assets。当前源码硬性要求 `numpy==2.2.5`、`mujoco==3.3.1`，并检查 `robosuite>=1.5.2`；setup 同时固定 `numba==0.61.2`、`scipy==1.15.3`。应使用 `robocasa365` 内独立环境，避免替换现有 LIBERO 环境。官方将资产下载规模描述为约 10 GB；下载大小和解压后的实际占用应单独记录，不需要下载示范数据。[官方安装说明](https://robocasa.ai/docs/build/html/introduction/installation.html)、[依赖源码](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/setup.py)

## 365 任务清单与评估边界

`../configs/task_catalog.json` 已静态枚举 **65 个 atomic + 300 个 composite = 365** 个唯一任务，逐项保存源码路径/链接/SHA256、类别、官方 dataset 分组和已知 horizon。方法是 atomic registry 的 65 个名称加包入口导入的 300 个 composite 类，并验证每一类在对应文件中存在；未 import 模拟器。

一个关键区别：当前 `dataset_registry.py` 只有 65 + 252 = **317** 个带数据集配置的任务；`TASK_SET_REGISTRY["all_tasks"]` 也只有 317 个。另有 **48** 个 composite 类存在于源码并被导入，但该 registry 没有其 horizon。清单将这些 horizon 留为 `null`，完整名称见 `unknown_horizon_tasks`，不能自行填默认值并声称沿用官方评测。[dataset_registry](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/utils/dataset_registry.py)、[环境导入入口](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/__init__.py)

RestockPantry 的 registry horizon 为 **2550 actions**；它属于 pretrain200/pretrain300，不在当前 target50 里。`split="pretrain"` 与 `split="target"` 控制厨房/对象分布，与模型是否 zero-shot 是不同概念。官方 multi-task 基准在 pretrain scenes 上比较 atomic_seen/composite_seen/composite_unseen；一次自选 RestockPantry episode 不能称作官方排行榜结果。[官方多任务评测](https://robocasa.ai/docs/build/html/benchmarking/multitask_learning.html)、[场景 split 实现](https://github.com/robocasa/robocasa/blob/3443cdf3dd20f9ae8ed41adba442059b506891d0/robocasa/utils/env_utils.py)

现有用户要求的 uncapped 探索试验与官方 horizon 评测应分开记录。首轮若沿用不设 episode cap 的用户协议，可以额外记录第 2550 action 时是否已成功，但不能把超出 horizon 的成功混入官方 horizon 内成功率。后续 365 任务评测需先冻结 episode seeds、场景/对象分布、每任务次数、允许的传感器/控制动作、horizon 缺项处理和分母，单独记录基础设施错误、用户停止及 unsupported，不自动重试或挑选成功样本。

## 准备完成的判断边界

本报告与 catalog 完成只证明源码合同与任务库存已核对。依赖导入、资产完整性、适配器离线单测和启动清单应在各自准备记录中给出证据。原生环境 reset/render、动力学动作与 live 模型完整任务结果仍必须按用户本轮要求等待批准；不应将静态通过写成模拟成功或任务成功。
