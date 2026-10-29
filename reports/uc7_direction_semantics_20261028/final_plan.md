# 最终方案：uc7 方向语义修复（uc7.1）

**一句话：** 方向词（左/右/前/后）一律按外部相机（图 1）的视角解释。世界方向由代码从标定算出，在审核时冻结并绑定审批。规划文本、执行契约、区域求解器、审核输出都用同一个来源。

**已确认的实施决定（2026-10-28）：** 版本 A 独立交付为 uc7.1，包含 A9 的世界轴措辞显示高亮；E1 偏好句和 E2 图像旋转留到独立实验。颠倒相机的额外处理延后，本版本保留 A1 公式并记录滚转/颠倒状态，不加否决。A0 采用 tar 快照。V3 以实际任务成功验收，不能将非方向原因的任务失败计为通过。完整任务实跑须先获得用户批准。

- **范围外：** t6 的杯柄抓取与肘部卡住后的恢复；uc8 的提速项；v3（已冻结）。
- **原则：**
  - 所有任务用同一个冻结版本；
  - agent 不接收任务键；
  - 不新增否决或调参阈值；
  - 以全量 sweep 验收。

文件简称（都在 `gpt_field/` 下）：

| 简称 | 路径 |
|---|---|
| spatial | `src/gpt_field/agent_speedup_v2_user_check_spatial.py` |
| uc client | `src/gpt_field/agent_speedup_v2_user_check_client.py` |
| agent | `src/gpt_field/agent_speedup_v2_user_check_agent.py` |
| agent client | `src/gpt_field/agent_speedup_v2_user_check_agent_client.py` |
| user_check | `src/gpt_field/agent_speedup_v2_user_check.py` |
| html | `src/gpt_field/agent_speedup_v2_user_check.html` |
| base client | `src/gpt_field/agent_speedup_v2_client.py` |
| runner | `scripts/evaluate_agent_speedup_v2_user_check.py` |
| robocasa runner | `scripts/evaluate_robocasa_user_check.py` |

## 一、版本 A（uc7.1，一次冻结，一起上线）

### A0 改代码之前先冻结基线

- **原因：**
  - uc7 的源文件处于"已修改、未提交"状态，v3 和 RoboCasa 运行器是未跟踪文件；
  - v3 运行器直接读取工作区的源码；
  - 所以版本 A 的改动会覆盖它们唯一的副本。
- **做法：**
  - 给 `src/`、`scripts/` 和两个 html 打一个 tar 快照（先例是 relayout 运行目录里的 `frozen_runtime_high.tar.gz`），或者经你同意后提交到一个分支或 tag；
  - 此后 v3 的重跑和 uc7 的基线回放都只用这个快照。
- **同时**记录一次改动前的 pytest 基线。`tests/test_agent_speedup_v3_client_prompt.py` 预计现在就会失败：它期望 client-10，代码里是 -11。

### A1 方向坐标系 helper（唯一来源）

- **位置：** 在 spatial 的 `_polygon`（:139-144）和 `_centre`（:147-152）附近新增纯函数；升级 `SPATIAL_VERSION`（:23）。
- **定义：** 只用外部相机标定。
  - `right = unit(R[:2,0])`；
  - `away = (-right_y, right_x)`，即世界上方 × right；零滚转时它就是相机的水平视线方向；
  - right_of = +right，left_of = −right，behind = +away，front_of = −away。
- **检查（不设调参阈值）：**
  - `|R[:2,0]|` 小于数值 eps（1e-9，与 `geometry._unit` 相同）时，返回 unavailable（camera_x_vertical）；
  - `R[2,1] < 0` 表示世界的"上"投到了图像下方，即画面颠倒，按"决定 2"处理。这个条件与 `away·(-R[:2,2]) < 0` 完全等价；
  - 俯视相机由公式自然得到 away = 图像上方，不需要特判；
  - 标定的有限性和正交性已经由现有的几何校验保证，不重复检查。
- **作为证据记录：** `roll_deg = atan2(R[2,0], R[2,1])`。注意 `asin(R[2,0])` 不是滚转角。
- **不做任何图像翻转：**
  - `observation.py:31` 的 `[::-1, ::-1]` 被 `contact_skills.py:283` 的 `[:, ::-1]` 抵消，合起来只是标定声明的竖直翻转；
  - 所以 R 描述的就是模型看到的正立、非镜像图像。
- **适用范围（写进 docstring，数值已核实）：**
  - **零滚转**时，不论相机换位、水平转向还是改变仰角：
    - left/right 在整幅图上都精确；
    - front/behind 取相机视线方向，与 LIBERO 的 ±x 完全一致；在图像边缘它和局部的"图像上方"最多差约 17–22°，但四个词都判对。
  - **有滚转**时：
    - ≤15° 时全桌都判对；
    - 30° 时约 7%、45° 时约 20% 的可见桌面位置至少有一个词判错；
    - 方向向量最多偏约 23°。
  - 已记录相机的滚转是 0°（LIBERO）和 0.9°（RoboCasa）。以后要支持大滚转，可改用锚点处的局部平面投影。

### A2 规划提示

- **只替换 spatial :59-60 中间的那一句**（"Relations use WORLD axes: … they do not mean screen directions."），保留前后文字，包括"anchor 为 null 时 bounds 是绝对世界坐标"。新句：

  ```
  left_of/right_of/front_of/behind (and these words in the task) are as seen in image 1, the external
  camera: right/left = toward image 1's right/left; behind/front_of = farther from/nearer to that camera
  along its viewing direction (toward the top/bottom of image 1). Not the robot's own left/right; never
  judged in the wrist images. Use the same viewer words in plan text; never restate them as world axes.
  Vertical words mean world up/down. A word naming part of an object (back compartment, left plate)
  selects that part by pointing; it is not a region.
  ```

- **spatial :60-63：** frame "world" 只表示 bounds 是世界米数；方向关系的 bounds 保持 null。这条只写在提示里，不加校验规则，否则就是新否决。
- **METRIC 提示（uc client :131-138，原句在 :134-135）：** 改写，不删除。
  - repair/review 用世界坐标的 local_motion：
    "Image left/right/up/down are not world axes; direction words are judged as seen in the external image (image 1); convert to world coordinates with calibration. World +z is up."
  - plan/replan 的请求里没有标定：
    "Image left/right are not world axes and you have no calibration: give directions only in image-1 viewer words; distances and heights may be given in metres. World +z is up."
- **RoboCasa：**
  - 把 `ROBOT_CONTEXT`（:38-45）拆成两份：plan 用的 `PLAN_ROBOT_CONTEXT`（:51）去掉"每张图附带标定"这句，执行用的保持原样（:59）；
  - 保留 `tests/test_robocasa_user_check_runner.py:370-372` 断言的三个短语。
- **升级** `PROMPT_VERSION`（uc client :24）。

### A3 执行契约（改变 uc7 在 t6 上行为的关键一步）

- **一个纯函数**（放在 spatial）：输入 view_frame 和计划里的 regions，输出 `region_directions`：
  `{direction_semantics: "external_view_v1", available, frame: {right_xy, away_xy}, regions: {id: {relation, anchor, direction_world_xy}}}`。
  A3、A4、审核输出和 V1 检查都调用它。
- **放进 agent 的 `_task_contract`（:115-126）：**
  - 即使计划里没有区域也给出 frame，因为 t4 的左右盘和 t5 的 back compartment 也要用；
  - 放在 regions 旁边而不是里面，PlanLock 的语义哈希不变。
- **每个执行请求都带一份紧凑版本（几百字符）：**
  - 原因：task_contract 只在第一次请求里发送（agent client :472-480），而 t6 的错误决定发生在第 11 次；
  - 做法：把新键加入 `CONTEXT_KEYS`（agent client :45，在 :456 校验）。
- **AGENT_INSTRUCTIONS**（agent client :51-192，相机段在 ~:179-191）加入：

  ```
  Direction words in the task and in reviewed regions are as seen in the external image (image 1):
  right/left = toward its image-right/left; front/behind = nearer to/farther from the external camera
  along its viewing direction. Never judge them in the wrist image. region_directions gives each
  region's world-xy direction computed from calibration; use it over any direction wording in reviewed
  text. If region_directions is unavailable, judge the words in image 1 as defined.
  ```

- **升级** agent client 的 `PROMPT_VERSION`（:41）和 agent 的 `POLICY_VERSION`（:30）。
- 请求大小没问题：第一次请求约 13–14k 字符，压缩阈值是 45k。

### A4 区域求解器

- **改判断：** spatial :211-214 改为投影判断。保留满足 `p·d > max_v(v·d)` 的格子，其中 d 来自冻结的坐标系，v 是锚点多边形的顶点。on、bounds、障碍排除和质心都不变。
- **载体放在 ctx 上，不放在 scene 上：**
  - 在 `AgentFirstV2Policy.reset()`（agent :66-77）之后设置 `ctx.user_check_view_frame = plan_lock.view_frame`；
  - 原因：ctx 每个 episode 只建一次，而 scene 会在中途重建。
- **读取和记录：** `_ground_observed_region` 读取这个载体；取不到就返回 `view_frame_unavailable`，并在证据（:235）里记录 frame 和 `direction_world_xy`。
- **旧路径：** `ReviewedV2Policy`（uc client :786-797）也设置这个载体，或者明确规定它返回 unavailable。
- **注意这是真实的语义变化：** 同一个 right_of 现在会选中旧代码里 behind 的区域，这是预期的。uc7 里宏路径 0 次授权，所以它不影响已记录的 uc7 行为，只保证一致性和 `_measure_goal` 的正确。

### A5 冻结并绑定审批（S6，必须做）

- **计算：** runner 的 prepare_review（:193-305，preparation 字典在 :266-281）和 RoboCasa 的 prepare（:164-228，字典在 :199-214）从 `observation.geometry` 的外部相机算出 A1 坐标系，写入 `preparation.view_frame`：
  - direction_semantics、available/reason；
  - 外部相机的完整标定（mujoco_name、pose_source、position、R、fovy/焦距/主点、图像尺寸）及其摘要；
  - right_xy、away_xy、roll_deg、四个关系的方向表。

  没有标定时（例如测试 fake），写 `available=false, reason=no_external_calibration`，不抛错。
- **绑定：** `preparation` 已在 user_check 的 `_CONTEXT_FIELDS`（:35-39）里，经 `_context`（:236-237）进入 `context_sha256`，再进入审批绑定（:240-246）。所以它自动被绑定，不需要新的哈希机制。
- **传递：**
  - 给 `PlanLock` 增加 `view_frame` 字段（uc client :234-240）：在 `__post_init__` 里深拷贝并校验，在 `snapshot()`（:346）里记录；
  - 在 runner :567 和 robocasa runner :256 用 `review["preparation"].get("view_frame")` 构造；
  - 不给 policy/client 工厂加新参数，因为测试 fake 的签名是 `(client, *, lock)`。
- **不加试验开始时的标定比对：**
  - LIBERO 的相机一变，reset 图像就变，而 reset 图像已被哈希；几何代码也在 runtime_sha256 里；
  - RoboCasa 在 reset 时已经比对含标定的传感器记录（:286-287）；
  - RoboCasa 目前底座和躯干不动，所以冻结方向等于实时方向。以后底座会动时，目标方向仍以冻结值为准，实时标定只用来解读当前图像。
- **可见性：**
  - 自动审批不经过界面（这次的 t6 就是自动审批）。所以在 prepare 的 JSON 输出、`plan_automatically_approved` 事件和 RoboCasa 的 approve 输出里，打印 view_frame，并为每个区域打印一行，如 `E: right_of C → world (0,+1) [image-right]`；
  - 审核界面在 relation 或 anchor 被编辑时，用 `preparation.view_frame` 的方向表重新算出这一行；
  - 不画箭头：一是有 CSP 限制，二是没有滚转时箭头就是图像的坐标轴。

### A6 语义标签与历史计划

- **升级 `VERSION`**（user_check :31）到 `agent-speedup-v2-user-check-7.1`。这样旧审核在新界面里会加载失败，而不是被悄悄套上新标签。旧审核用它们 `frozen_source` 里的 html 或原始 JSON 查看。
- **在 `framework_settings()`（runner :58）加入** `direction_semantics = "external_view_v1"`。它会随 execution_options 进入审批，并写进每个 manifest 和 summary；没有这个标签的计划即为 `world_axes_legacy`。RoboCasa 的 manifest 同样处理。
- **保持现状：** 旧审批不能在新代码上执行。runtime_sha256 覆盖全部 src 文件、runner 和 html（user_check :396-397）。
- **v3：** 它继承 `UserCheckV2Client` 和 `AgentFirstV2Policy`，运行器又直接加载工作区里的 uc 运行器，所以版本 A 会改变 v3 的行为。v3 只从 A0 的快照运行。

### A7 审核界面标签

- html :104-105 的标签和 :244 的备注改成外部视角的定义。
- 只改 `agent_speedup_v2_user_check.html`；v3 的 html 已冻结，不动。

### A8 腕部图像的不实文字（S7 的文字部分）

规划时腕部图像并没有旋转（k=0），但文字说已旋转对齐。逐处处理：

- **base client :199-200**（plan/replan 的指令文本）：用 uc client 现有的替换机制（:374-389）换成与 k 无关的句子：
  "Images 3/4 are the wrist camera, turned by the number of quarter turns stated in the request; a quarter turn is coarse and never a perspective match; judge direction words only in image 1."
- **base client :505**（每次请求的图像标签）：在 `UserCheckV2Client` 里覆盖 `_plan_request`，参照 `ReviewedV2Client` 已有的覆盖（uc client :467-486），按 quarter_turns 改写图 3 的标签；k=0 时写 "not turned; not aligned with image 1"。
- **base client :1097、:1159-1161、:291**：不改。它们属于 uc7 不发送的请求类型；在宏路径上 k 按标定计算，"turned"的说法属实。

计算"图 3 与图 1 的实际关系"需要新的输入路径，放到 E2。

### A9（已纳入版本 A，只做显示）

- 在审核界面和 trial 日志里，高亮部件文字中的世界轴措辞（如 "world +x"）。
- 格子顺序提示（比较区域格子和锚点格子）在 t6_006 上有误报。要加的话也只显示，绝不作为 plan issue。

## 二、验证（按顺序）

### V0 离线（不用仿真、GPU 或模型调用）

1. **A1 单测：**
   - 已记录的标定：relayout t6、t4_001、t5_001、t6_007、v3 g5_001 应得到 right (0,1)、away (−1,0)、roll 0°；RoboCasa coffee 应得到 (0.979,−0.202)、(0.202,0.979)、roll 0.9°。
   - 合成相机：
     - 换位、改仰角、俯视：四个词都判对；
     - 滚转：只在图像中心检查；
     - `R[2,1]<0`：走"决定 2"；
     - X 轴竖直：返回 unavailable。
   - 原型：`work/rotation/rotation_check.py`、`work/judge/axes_check.py`。
2. **图像管线一致性（防止重复翻转）：**
   - (a) 确定性测试：对任意原始渲染 A，断言 `StepView.external_rgb_upright == A[::-1, :]`，LIBERO 和 RoboCasa 各一条；
   - (b) 用 relayout 的稳定后位置（`physical_validation.json`）投影：布丁应落在格子 35，杯子落在 38；镜像之后两者互换（35↔38），这对物体就是镜像检测器；
   - 盘子中心投在格子 53，所以只检查规划模型指的格子 61 的中心落在盘子范围（r 0.069 m）内。
3. **BDDL 约定测试（只作开发测试，不喂给 agent）：**
   - 41/41 个可判定的检查通过：21 个按方向选物体、11 个 caddy 隔间、9 个侧向目标区；4 个 "middle" 不可判定；
   - 31 个锚定侧区的符号都正确；
   - 这只证明坐标系约定，不证明每句任务语言都和目标一致。两处 LIBERO 标注错误列为已知。
4. **求解器：** 用合成高度场加 t6 的盘子多边形，检查：
   - 新 right_of 等于旧 behind（预期的变化）；
   - RoboCasa 的方向正确；
   - 没有坐标系时返回 `view_frame_unavailable`。
5. **文本快照：**
   - 四种请求里都没有 `right_of=+x` 和 `WORLD axes`，改写后的 METRIC 句子都在；
   - AGENT_INSTRUCTIONS 含方向定义；
   - k=0 的规划请求里没有 "turned so that its left and right match"；
   - html 只检查 `agent_speedup_v2_user_check.html`，正则要能匹配 Unicode 负号（`/world [+\-−][xy]/`），并检查 "follow world axes"；
   - 所有版本号都已升级。
6. **端到端（fake）：** 带标定的 fake prepare → 审批 → `execute_reviewed`，断言：
   - `PlanLock.view_frame == preparation.view_frame`；
   - 第一个执行请求的 `reviewed_goal_contract.task_contract.region_directions` 正确，之后的每个请求都带紧凑版本；
   - 修改 `preparation.view_frame` 会使审核完整性检查失败。
7. **其他路径：**
   - RoboCasa 的 execute 携带 view_frame（fake 的单位 R 应得到 right (1,0)、away (0,1)）；
   - 没有 view_frame 的旧审核，方向为 unavailable；
   - k=0 时的腕部图像标签正确。
8. **回归：** 跑 `tests/test_agent_speedup_v2_user_check_*.py` 和 `tests/test_robocasa_user_check_*.py`（跳过需要仿真或 GPU 的），与 A0 的基线对比；所有已记录的 approved_plan 仍能通过 `validate_spatial_spec`。

### V1 只调规划（约 13 次调用）

- **范围：** LIBERO-10 全部、`libero_goal_g5`、t6 relayout、RoboCasa prepare；与已记录的 uc7 草稿（`runs/..._t*_review/review.json`）对比。
- **通过标准：**
  - t6 的区域是 right_of，方向 (0,1)；
  - g5 是 front_of 炉子，方向 (1,0)；
  - 部件文字里没有世界轴方向（用正则检查）；
  - t3/t4/t5/t9 的 regions 仍为空；
  - t4 两个盘子的格子与 t4_001 相同或相邻。

### V2 定向实跑（4 次）

只用 t6 relayout 和 g5 判断方向修复是否生效。t6 官方布局不用来判断：布丁一开始就在 +y 半平面内，执行模型可能认为目标已满足而不移动它，这时的失败属于 E1 的问题，不能归因到版本 A。

- **t6 relayout：**
  - 方向：在响应文件里，抓起布丁之后、第一次 `gripper=open` 之前的 `move_to_pose` 目标应满足 y > 0.079（盘子 y 0.0057 + 碰撞半径 0.073）；
  - 结果：看 physical_truth 中布丁 com 的 y 和 `com_margin_region_mm`。
- **g5：**
  - 方向：推动目标在炉子的 +x 侧；
  - 结果：看盘子的 `com_margin_region_mm`。
- **两者都要检查：** 第一个执行请求里的 region_directions，t6 的 E 应为 [0,1]，g5 的前方区域应为 [1,0]。
- **t6 官方布局：** 报告结果，但不用来判断方向语义。
- **RoboCasa coffee：** 只检查管线（任务里没有方向词）：view_frame 的 pose_source 为 `synchronized_robot_mounted_camera`，状态不是 infrastructure_error。
- **另外记录：** 释放点或推动点到锚点边缘的距离，作为 E1 的输入。

t6 仍可能因为杯柄抓取失败，g5 仍可能因为推动失败。

### V3 验收（同一个冻结版本）

- **范围：** t0–t9 官方布局、t6 relayout、t7 relayout、g5、RoboCasa coffee，共 14 次。V2 的 4 次只在 runtime_sha256 相同时才算进来，否则要重跑。
- **通过标准：**
  - 所有方向标准都满足；
  - 没有任何一次失败的证据指向方向原因（走错侧、文字与向量冲突）或计划无效；
  - 按用户最终决定，要求实际任务成功；其他原因的失败同样不计为通过，并记录原因。原始 benchmark 成功、严格物理诊断和是否满足验收分别报告，不用模型声明代替任务成功。
  - V2/V3 完整任务启动前须经用户批准；本轮交付先完成实现、V0 和 V1。

## 三、版本 A 通过后的独立实验

### E1 选点偏好（D9）

- **语义不变：** bounds 为 null 仍只表示"那一侧"（半平面）；`_measure_goal` 和区域格子都不变。
- **求解器：** 对没有 bounds 的方向区域：
  - 目标点 `p* = c + (max_v(v·d) − c·d + r_obj + M16 间隙)·d`，即越过锚点边缘、横向对齐锚点中心的点；
  - 选离 p* 最近、放得下的空格；这是单一距离，不需要权重或新常量；
  - 这一步要放在原生"中心优先"的提前返回（spatial :306-313）之前。
- **执行契约（只是偏好，不是命令）：** "with no distance given, prefer a free spot just past the anchor's edge, roughly aligned with the anchor; any spot on that side satisfies the relation."
- **测量与对照：**
  - uc7 里宏 0 次授权，所以测到的主要是执行模型文字的效果；
  - 在 t6（官方布局和 relayout）和 g5 上对比版本 A 与 A+E1，比较落点到锚点边缘的距离和落进目标框的比例。
- **已知情况：**
  - 之前的复核核实了"最近空位"能落进 4/5 个拾放类锚点目标框；
  - g5 的框从炉子边缘外 7.5 cm 才开始，紧挨着放的盘子中心在框内约 4 mm，正好处于边界上；而且 g5 是推动，求解器不控制终点。
- **采用前**需要单独跑一次全量 sweep。
- **不做显式"紧挨"。** 以后需要时，再设计沿方向/横向的 bounds 或 `max_gap_m` 字段，单独验证。

### E2 腕部图像旋转（S7 的图像部分）

- **k 每次从当前标定计算**（`frame_alignment_turns` 加一个外部相机适配器），不是套件常量。
- **以下内容要同步改：**
  - 图像；
  - 格子编号和 w 的反变换（`merge_plan` 的 quarter_turns）；
  - 指尖边缘标注（`FINGERTIP_EDGES`）；
  - 文字改为"大致对齐（最接近的四分之一圈）"。

  "图 3 与图 1 的实际关系"的计算和输入路径也在这一步加入。
- **验证顺序：**
  - 先只调规划做 A/B：13 个任务，k=0 与标定 k 各一次，约 26 次调用；比较计划有效性和腕部格子的指向精度；
  - 再实跑；采用前同样需要单独跑一次全量 sweep。

## 四、已确认的决定（覆盖原草案选项）

- E1 偏好句留在 E1；版本 A 不添加。
- 颠倒相机的特殊处理延后；保留 `away = up × right` 并记录诊断。
- 版本 A 独立命名 uc7.1。
- A9 世界轴措辞高亮纳入；不增加格子顺序否决。
- V3 要求任务成功；方向正确但任务失败不算验收通过。
- A0 使用已校验的 tar 快照，未进行 git 提交。

以下保留原讨论作为决策背景，不再是待决项：

1. **E1 的执行偏好句放进版本 A，还是留在 E1？**
   - 放进 A：t6 官方布局成为有意义的测试，但版本 A 多一处行为变化；
   - 留在 E1：归因更干净，t6 官方布局在 V2 中只作参考。

   我倾向留在 E1。
2. **画面颠倒的相机怎么处理？**（目前没有任何相机是这样）
   - (a) 返回 unavailable：等于新增一个永远不会触发的否决；
   - (b) 保持定义：right = 图像右侧，behind = 沿相机视线更远（`R[2,1]<0` 时把 away 取反），并记录 roll。

   我倾向 (b)。
3. **版本 A 作为 uc7.1 单独发布**（VERSION 升级），不和 uc8 的提速项捆绑。
4. **A9 是否随版本 A 上线：**
   - 世界轴措辞高亮：只做显示，成本低；
   - 格子顺序提示：在 t6_006 上有误报。
5. **V3 中单次 episode 的回退标准：** 按上面的"只拦方向原因导致的失败"，还是要求已记录成功的任务必须再次成功？
6. **A0 用哪种方式：** tar 快照，还是提交到分支或 tag？提交需要你同意。

## 五、工作量与风险

- **版本 A：** 约 10–12 小时。

  | 步骤 | 工时 |
  |---|---|
  | A0 | 0.5 h |
  | A1 | 1 h |
  | A2 | 1 h |
  | A3 | 1.5 h |
  | A4 | 1 h |
  | A5 | 2.5 h |
  | A6 | 0.5 h |
  | A7 | 0.25 h |
  | A8 | 1 h |
  | 离线测试 | 2 h |

- **验证：** V1 约 13 次规划调用；V2 4 次实跑；V3 另需约 10 次实跑。
- **独立实验：** E1 约 2–3 小时；E2 约 2 小时加约 26 次规划调用。E1 和 E2 采用前各需一次全量 sweep。
- **风险：**
  - 规划文本对所有任务都生效，只有全量 sweep 能证明它可以接受；
  - 模型仍可能写世界轴措辞，由 region_directions 覆盖；
  - t6 的杯柄抓取和 g5 的推动是独立问题；
  - v3 如果不从快照运行，会被版本 A 改变。

## 六、本轮核查纠正了哪些说法

- **滚转：** 之前说"滚转 5°–89° 四个方向词都判对"，那只是在盘子所在的像素上。看整张桌面，滚转 ≥30° 时部分位置会判错（30° 约 7%，45° 约 20%）。已记录相机的滚转约为 0°，目前没有影响。另外，`asin(R[2,0])` 并不是滚转角。
- **g5：** 之前说紧挨着放"差一点够不到"。计入间隙后，盘子中心其实在框内约 4 mm 处，正好在边界上；而且 g5 是推动。"6/7 个框"的说法也没有依据，核实到的是拾放类的 4/5 个。
- **图像管线测试：** 之前的设计在正确代码上也会失败，因为盘子中心在格子 53，不是 61。现已改成确定性的图像变换测试，加上用布丁和杯子做的镜像检测。
- **冻结方向的传递路径：** 之前没有写明冻结方向如何到达试验运行（PlanLock）、RoboCasa 的试验路径和区域求解器。A3–A5 现在都写明了具体载体。
- **试验开始时的标定比对：** 它是多余的，已删除，以免多出一条拒绝路径。

依据：本目录的 [README.md](README.md)、[workflow_result.json](workflow_result.json)、[plan_check_result.json](plan_check_result.json)（对草稿 [work/final_plan_draft.md](work/final_plan_draft.md) 的三项只读核查），以及 [work/rotation/](work/rotation/) 和 [work/finalcheck/](work/finalcheck/) 中的原型脚本。
