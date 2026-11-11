# foundation_vision_harness_V3

当前修订：`astra_tools_r4_grid1mm_uncapped_transit`。Astra 输出批量动作和按场景变化维护的粗障碍图，原 speedup v2 的测量反馈片段执行，SAM/DA3 作为按需测量工具。r4 使用 1 mm transit 网格、取消 transit 动作步数预算；T04 第 004 轮已严格成功，在线 557.969 秒，记录 41 个禁止接触采样点。此前 r3（5 mm）的第 003 轮在线 373.181 秒，r2 的第 002 轮在线 405.467 秒。第 001 轮结果和三轮离线视频均来自原 `astra_tools_r1`。

## 数据流与控制权限

1. Astra 接收当前外部/腕部 RGB、公开标定、实测关节与夹爪状态、执行反馈。
2. `execute` 给出 1–24 段动作，每段 1–3 个候选；支持绝对位姿、相对移动、接近、开合与等待。XYZ 必须显式提供。局部完成一段后自动进入下一段。
3. `obstacle_map` 使用 Astra 指定的最多四个世界坐标盒与 clearance。原 `TransitObstacleMap/TransitRoute` 在每段 transit 入口规划，随后跟踪路线；contact 不使用该盒体路由。空值保留原图，空盒列表显式清空。
4. 每条片段动作检查已知机器人关节限制及当前/短时预测构型的自碰撞。使用缓存的静态机器人碰撞模型，不使用视觉模型点云或场景年龄。预测仅采样一个控制周期，不能证明连续或全臂环境无碰撞。
5. 本地判断到位、夹爪稳定和运动停滞；`observe/held/released` 检查点暂停批次，把新的 RGB 和剩余动作交回 Astra。稳定间隙不等于抓牢，模型明确安排抓取后的抬升和释放后的退出。
6. `measure` 请求分割、材料像素三维位置或局部表面范围。结果带来源、年龄、不确定性或不可用原因，交回 Astra；不自动改写目标，不产生障碍门禁。
7. `finish/blocked` 自主结束。raw LIBERO 与严格物理检查独立记录，不进入策略、不提前结束策略；终态检查保持夹爪，不替策略释放物体。

普通动作没有 `SceneService`、来源图绑定、DA3 场景 TTL 或载物点云存在性门禁。相对位移在入段时解析一次；只有实际下发并收到匹配反馈的动作计入片段进度。首条动作不可用可尝试模型给出的备用候选；已有实际运动后受阻则返回实际状态，由 Astra 改计划。

## 障碍图维护（r2）

恢复 Astra 的建图与维护提示职责，沿用已经接通的地图缓存、transit 路由和录像快照：

- 第一次 `execute` 根据当前 RGB 和标定，在同一条动作响应内给出相关障碍盒。`source=uninitialized` 的初始空图不代表已确认无障碍；只有未见与后续 transit 相关的障碍时才显式给空集合。
- 场景未变时用 `null` 保留地图及其原始更新步数。保留相关障碍，即使当前直线路径已经绕开它；不能仅为绕过规划阻断而清空地图。
- 抓取／搬运前，由 Astra 替换地图，移除将操作的对象，保留其他相关障碍，并在 clearance 中考虑手部、载物尺寸和估计误差。
- 移动／放置物体、切换目标或发现几何错误后，在下一次正常动作响应里重新判断并按需替换地图；新放下的物体仍可能成为后续障碍。
- 地图在整批动作内固定，不会随物体自动移动。若后续动作依赖变化后的场景，在适当的现有视觉检查点交回 Astra，再随下一批更新。

没有新增专门建图请求、强制 SAM/DA3 调用或初始地图门禁。额外开销来自响应中的地图内容和必要的路线计算，实际延迟需要完整任务测量。地图仍是模型估计；transit 只检查末端路径，contact 不使用盒体路由，不能据此保证全臂无碰撞。

## 1 mm 网格与无步数预算 transit（r4）

`TRANSIT_GRID_SPACING_M=0.001`，`TRANSIT_MAX_STEPS=None`。V3 协议、执行器和启动配置均使用同一设置；Astra 提示也明确说明 transit 无动作计数预算。每段 transit 的 `max_steps` 和进度中的 `budget_steps` 均为 null，同时取消该执行片段原先隐藏的累计 240 步上限。

执行仍按实测反馈运行：到位确认后完成；停滞、路由不可用、持物间隙变化、机器人约束或用户停止仍可触发交接／结束。100 mm 单步平移命令上限、4 mm 到达距离、姿态误差阈值及两次稳定采样不变。contact 保持 80 步，open／close 保持 16／20 步，显式 wait 保持模型指定的有限步数。旧版调用共享执行器时仍使用原预算，只有 V3 显式开启无上限 transit。

1 mm 改的是规划网格，不是动作幅度或障碍几何精度；插值角点有效性规则仍然存在。此前 T04 第 003 轮没有 transit 步数耗尽事件，因此取消预算本身不解决那轮的网格拒绝和普通停滞。

实现时只运行一次必要的 CPU 检查：3 项通过（1.92 秒；[记录](output/foundation_vision_harness_V3_grid1mm_uncapped_r4/integration.xml)）。覆盖实际 V3 使用 1 mm、保存位置的路线判定，以及慢速但持续进展的 transit 超过 120／240 步后正常到位；同一检查确认停滞仍会交接、旧执行器仍在 120 步预算耗尽。该检查无真实模型或模拟器调用；随后经用户批准完成第 004 轮完整测试，见下文。没有批准后续重试或其他任务。

## 5 mm 路由网格（r3，历史修订）

r3 显式传入 `TRANSIT_GRID_SPACING_M=0.005`，经片段执行器进入真实 `TransitRoute`；启动配置记录 `transit_grid_spacing_m`，每条路线诊断记录实际网格。共享执行器的旧调用仍默认 20 mm。障碍盒、clearance、控制动作幅度及到达阈值保持原值，5 mm 是搜索网格间距，不是视觉定位精度。

更细的网格缩小了插值角点造成的额外屏蔽范围，但没有修改“所有插值角点必须有效”的规则或到达判断顺序。因此这是一项有针对性的缓解，不能保证所有贴近障碍的情况都消失。同体积逻辑节点约增至 64 倍；当前直线优先、惰性搜索，不会固定分配全部节点，实际耗时仍需测量。

离线检查使用 r2 保存的第 276、369 步地图和实测位置，只重建路线和计算命令，不启动模型或模拟器；另在已有 CPU 贯通检查中确认 V3 实际采用 5 mm。随后经用户批准完成一次第 003 轮完整测试，见下文；没有批准后续任务或重试。

结果：两项检查通过（2 passed，1.28 秒；[记录](output/foundation_vision_harness_V3_grid5mm_r3/integration.xml)）。20 mm 在两处保存位置均重现 `invalid_measured_start`；5 mm 下分别正常跟踪、识别为到达，且障碍盒内的实测点仍被拒绝。这不等于重新执行任务，也不证明完整运行一定更快。

## 可选视觉

`--vision on-demand` 为默认值；启动时不加载视觉权重。工具首次调用才加载所需 SAM/DA3，加载和测量时间均计入在线时间。默认发现仓库已有权重，也可显式提供 `--sam-checkpoint`、`--da3-model`。缺失权重作为工具不可用报告。

`--vision off` 可在没有 SAM/DA3 的环境下运行普通 RGB 操作。分割与深度只是预测：单视角测量不会被包装成已验证几何，过期结果不会重新盖时间戳。按需工具需要额外模型往返，应只用于会影响下一步动作的疑问。

`--vision-max-age-s` 默认 60 秒，针对模型选中来源图的可选测量，不是运动场景 TTL。原始观测进入执行器、等待 Astra 和首次权重加载的时间全部计入年龄；超过有效期返回不可用。它不会自动换一帧图后继续使用旧像素，也不会因工具过期结束操作。

模型保持 Astra high、ChatGPT 订阅，不使用 API-key fallback。不增加任务专用技能、固定任务坐标或演示输入。模型障碍盒、机器人自身检查和 record-only 物理碰撞统计是不同来源，日志与录像分别标记。

## 第 001 轮 T04 can obstacle 启动配置

第 001 轮已获用户批准并在 GPU 7 上执行，以下保留其参数。该输出目录已经存在，不应重用；任何后续完整测试需要新批准与新目录。启动时按资源配置 `CUDA_VISIBLE_DEVICES` / `MUJOCO_EGL_DEVICE_ID`。

```bash
PYTHONPATH=src GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M=0.10 \
  .venv-foundation-vision/bin/python scripts/evaluate_foundation_vision_harness_V3.py run --approved \
  --task libero_10_t4 --episode-index 0 --vision on-demand \
  --output runs/foundation_vision_harness_V3_t04_can_obstacle_001 \
  --task-bddl-override output/foundation_harness_preparation_001/inputs/custom_task.bddl \
  --initial-state-override output/foundation_harness_preparation_001/inputs/initial_state.npy \
  --expected-initial-observation-sha256 06f623f31ee2ea2e8ef600ce29d299413d81a8647d8f59663f3aff2e660da187 \
  --store-dir .private/chatgpt_plan
```

可用 `inspect --task libero_10_t4 --vision off` 只查看配置；它不请求模型、不加载权重、不启动模拟器。完整运行没有整轮时间、动作或请求上限，也没有自动重试或后续任务队列。三分钟是性能目标。

停止文件：`trial/operator_stop.request.json`。终态产物包括 `result.json`、`trial/summary.json`、逐步 `trace.jsonl`、模型请求、工具结果和 `trial/episode.mp4`。录像显示模型障碍盒、目标位姿、规划路线及实测路径；没有 DA3 地图时不展示伪造点云。计时包含在线日志/录像提交/工具等待，加载与收尾单列。

## 实现与最小检查

- `client.py/protocol.py`：独立订阅接口与批量/工具协议。
- `executor.py`：实测动作队列、真实 Astra 障碍盒、视觉检查点反馈。
- `vision_tools.py`：按需推理与带来源的可选测量。
- `robot_guard.py`：不依赖视觉的机器人自身约束。
- `runner.py/recording.py/telemetry.py`：完整运行、异步录像与计时。
- `scripts/evaluate_foundation_vision_harness_V3.py`：独立入口，无强制视觉预加载。

最小检查集中在 `tests/test_fvh_v3_integration.py`。CPU 贯通检查使用离线订阅传输、模拟反馈和合成视觉后端，覆盖实际协议、执行队列、障碍盒路由、过期工具反馈、检查点与自主结束。r2 在同一检查中覆盖首次提示传递、非空地图绕行、null 复用、移动对象后整图替换并保留其他障碍，以及录像快照获得更新后的盒子。r3 增加保存位置的网格回归检查，r4 增加超过 240 步的执行与保留停滞反馈检查。它们不证明真实 Astra 建图质量、几何精度或任务成功；真实表现由获批后的完整任务测量。

旧版源文件与历史运行保持原样。准备结果保存于 `output/foundation_vision_harness_V3_preparation_001/`。

r2 最小检查：上述单项测试通过（1 passed，1.28 秒），记录见 [`integration.xml`](output/foundation_vision_harness_V3_obstacle_map_r2/integration.xml)。没有运行真实模型、模拟器任务或新增完整测试。

## T04 第 004 轮结果（r4，1 mm，无 transit 步数预算）

[本轮报告](runs/foundation_vision_harness_V3_t04_can_obstacle_004/REPORT.md)：两杯通过杯口抓取后完成放置、释放和自主结束，raw 与两次 strict 均通过。757 步、21 次 Astra 请求，在线 557.969 秒，比第 003 轮慢 184.788 秒；模型等待占约 73%。自主结束时累计 41 个白杯与障碍罐接触采样点。

14 条路线确实使用 1 mm，transit 预算均为 null；规划合计 1.154 秒，无图搜索。最长 transit 仅 35 步，取消预算尚未展示实际任务收益。本轮没有插值起点拒绝，但有一次 `route_start_blocked`：55 mm clearance 膨胀框直接包含实测起点，随后 contact 上撤恢复。进程正常退出、源码未变；[视频](runs/foundation_vision_harness_V3_t04_can_obstacle_004/trial/episode.mp4) 已保存。

## T04 第 003 轮结果（r3，5 mm）

[本轮报告](runs/foundation_vision_harness_V3_t04_can_obstacle_003/REPORT.md)：两杯各一次抓取并释放，raw、终态 strict、保持后 strict 及严格总分全部通过。649 步、15 次 Astra 请求，在线 373.181 秒；扣除已测录像提交／日志为 367.858 秒。比第 002 轮快 32.286 秒，但两轮动作和模型决策不同，不能将全部提速归因于网格细化。

10 条路线均实际使用 5 mm，初始直线全部通过，路线构建合计 0.266 秒。第 270 步仍因插值角点屏蔽触发 `invalid_measured_start`，后续 contact 上撤恢复，因此边界问题尚未根治。自主结束时记录 13 个白杯与障碍罐接触采样点，发生在闭爪阶段；本轮不是无碰撞成功。进程正常退出，源码未变；[视频](runs/foundation_vision_harness_V3_t04_can_obstacle_003/trial/episode.mp4) 已保存。

## T04 第 002 轮结果（r2）

[本轮报告](runs/foundation_vision_harness_V3_t04_can_obstacle_002/REPORT.md)：两杯各一次抓取，完成搬运、释放和自主结束，raw 与两次严格物理检查全部通过。757 步、18 次 Astra 请求，在线 405.467 秒；扣除已测主线程录像提交／日志写入为 399.223 秒。禁止接触记录为 0。

首次动作响应恢复 3 个障碍盒，之后按放置和换目标更新。10 条 transit 路由均使用非空地图，其中一次触发绕障搜索，总路线构建 0.122 秒。两次 `invalid_measured_start` 通过后续 contact 动作恢复；没有重复阻断循环。Astra 请求共 263.400 秒，仍是主要耗时，三分钟目标未达到。源码未变，进程正常退出；[带障碍图的视频](runs/foundation_vision_harness_V3_t04_can_obstacle_002/trial/episode.mp4) 已保存。

## T04 第 001 轮结果

运行：[`runs/foundation_vision_harness_V3_t04_can_obstacle_001`](runs/foundation_vision_harness_V3_t04_can_obstacle_001/REPORT.md)。在线 243.757 秒、459 个动作、15 次 Astra 请求。两杯均完成抓取、搬运和释放，禁止接触记录为 0；未达到三分钟目标。

终止时 raw=true、strict=false，唯一未满足项为黄杯 `at_rest`。随后 10 步保持原张爪命令和手部位姿，两个杯子的物理条件全部通过；没有替策略释放或移动目标。当前严格总分要求两个时刻都通过，因此仍为 false，不把本轮时长标为严格完成延迟。

三批按需测量共九个点，只调用 DA3，累计约 10.08 秒，SAM 未加载。Astra 请求累计 152.36 秒，占在线时间 62.5%。五次片段中断（三次停滞、两次张爪预算结束）造成额外交接；机器人自身检查没有拒绝动作。Astra 本轮未提供非空障碍盒，不能据此声称非空障碍路由或视觉工具的因果收益已验证。源码 370 文件保持不变，进程已退出、视频已完成，无自动重试。

## T08 第 001 轮结果

运行：[`runs/foundation_vision_harness_V3_t08_bottle_obstacle_001`](runs/foundation_vision_harness_V3_t08_bottle_obstacle_001/REPORT.md)。在线 260.617 秒、476 个动作、13 次 Astra 请求，自主结束。两个壶最终均在炉面、夹爪已退出，raw=true；终止时第二壶静置窗口未满足，strict=false，保持原张爪状态 10 步后全部物理条件通过，冻结的严格总分仍为 false。

第一壶在第 197–198 步附近水平搬运时滑脱，随后落在目标炉面，不能记为受控放置。最后撤手阶段记录 53 个第二壶与炉头的禁止接触采样点，没有记录到瓶子碰撞；不能报告为零禁止接触。

Astra 累计 178.52 秒，占 68.5%；一次按需 DA3 测量约 9.84 秒，SAM 未调用。三次停滞与两次张爪片段预算结束导致模型交接。障碍盒均为空，机器人自身检查无拒绝。源码 370 文件未变，进程正常退出、录像已完成；未追加测试或修改控制器。

## 原始 T07 第 001 轮结果

运行：[`runs/foundation_vision_harness_V3_t07_original_001`](runs/foundation_vision_harness_V3_t07_original_001/REPORT.md)。官方 init 0、seed 7，没有场景覆盖。两件物体均一次抓取后搬入篮子、主动释放，Astra 自主结束；raw、终止时 strict、保持后 strict 与严格总分全部为 true。

在线 358.161 秒、510 个动作、17 次 Astra 请求；记录的完成时间含终态保持与评分为 359.483 秒。模型请求累计 269.00 秒，占在线 75.1%，未达到三分钟目标。六次停滞和一次张爪预算结束造成交接；机器人自身检查没有拒绝动作。

禁止接触采样点 3,065：手掌与篮子 1,778、手指与桌面 587、汤罐与篮子 700。因此任务成功不等于无碰撞完成。一次按需 DA3 测量约 9.03 秒，SAM 未调用，模型障碍盒为空。源码 370 文件未变，进程正常退出，录像已完成，无自动重试。

## 三轮障碍图视频

使用原录像中的连续双视角 RGB 与逐帧规划快照离线生成，1080p、10 fps。右侧放大显示固定坐标范围的 XY／XZ／YZ 视图：黄色为当前目标与规划路径，粉色为实测末端轨迹，白点为当前末端位置，目标处三色轴为姿态。画面显示原始在线时间，播放压缩模型等待，不能用视频时长作为任务耗时。

三轮的全部记录均为空障碍盒集合，视频明确显示 `Recorded obstacles: 0`。这表示没有提供障碍几何，不表示真实场景没有障碍。没有将 DA3 测量补造为运行时障碍图，也没有重跑模拟器或调用模型。

- [T04 can obstacle：93.9 秒](runs/foundation_vision_harness_V3_t04_can_obstacle_001/visualizations/obstacle_map.mp4)
- [T08 bottle obstacle：97.3 秒](runs/foundation_vision_harness_V3_t08_bottle_obstacle_001/visualizations/obstacle_map.mp4)
- [原始 T07：104.1 秒](runs/foundation_vision_harness_V3_t07_original_001/visualizations/obstacle_map.mp4)

每个视频同目录保存预览、终态图和来源清单 `obstacle_map.json`。离线生成脚本：`output/fvh_v3_obstacle_videos_20261110_001/render_obstacle_video.py`。

按用户指定旧版视频样式重做的版本采用中央三维透视图、右侧 XY／XZ 投影、左侧双相机和底部事件时间轴，1080p、20 fps。红圈表示日志中各类别首次接触的历史位置；红色时间轴表示当步新增禁止接触采样，橙色标记表示片段中断。地图仍只显示本轮实际记录的障碍盒，不复制旧版运行中的方框。

- [T04 三维参考样式：59.4 秒](runs/foundation_vision_harness_V3_t04_can_obstacle_001/visualizations/obstacle_map_3d.mp4)
- [T08 三维参考样式：61.5 秒](runs/foundation_vision_harness_V3_t08_bottle_obstacle_001/visualizations/obstacle_map_3d.mp4)
- [T07 三维参考样式：70.5 秒](runs/foundation_vision_harness_V3_t07_original_001/visualizations/obstacle_map_3d.mp4)

这些版本每个记录状态播放 0.1 秒，关键事件额外暂停，模型等待省略；在线时间另行显示。来源清单与帧索引为 `obstacle_map_3d.json`、`obstacle_map_3d.frames.json`。离线脚本为 `output/fvh_v3_obstacle_videos_20261110_001/render_reference_style.py`，没有模型或模拟器调用。
