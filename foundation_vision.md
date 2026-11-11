# Foundation vision v1

这一版把通用视觉模型接成独立的、只读的感知层：SAM 3.1 提供文本指定对象的候选区域，DA3-LARGE-1.1 提供深度假设，已知相机标定把它们转换到世界坐标。新增的可选 live 接入将这些输出送入 Astra 的实际请求，由既有执行链执行动作；基础几何模块本身不生成或执行机器人动作。

## 输入边界

- 同时刻的至少两个 RGB 视角，以及各自的内参和世界到相机外参。
- 可选的机器人手部位姿，用于输出目标相对手的位置。
- `TargetSpec(key, text, seed_pixels={})`：由上层给出的短文本目标描述；可选像素点只用于初次消歧，不能当成持续跟踪依据。
- 不接受物体真值位姿、模拟器深度、接触、奖励或任务成功标志。

相机坐标约定是 OpenCV：右为 +X、下为 +Y、前为 +Z；长度单位为米。`bridge.frame_from_step_view()` 和 `frame_from_observation()` 负责现有公开观测的图像方向和 OpenGL→OpenCV 转换，并验证腕部相机标定与同一帧机器人状态一致。这需要当前已公开的相机标定，不能直接用于旧版未标定 RGB 协议。

## 输出与失效行为

`VisionEngine.process(frame, targets)` 返回 `VisionSnapshot`：

- `summary()` 给上层：观测时间、年龄、目标状态、候选 ID、世界坐标和相对手坐标、目标丢失/歧义/几何变化等事件。
- `to_dict()` 另外包含每视角质量统计、部分目标表面和场景点云的有界序列化结果。
- `geometry.query(points, now_s, max_age_s)` 返回 `surface`、`estimated_free`、`unknown` 或 `stale`。
- `engine.query_path(polyline, radius_m)` 检查局部扫掠管道，返回 `blocked`、`estimated_clear`、`unknown` 或 `stale`；默认查询时钟随真实等待时间前进。

目标区域在多个视角中不一致时拒绝融合；多个候选不会按最高分偷偷选一个。持续 ID 需要跨视角几何、外观以及至少一个固定相机中的区域重叠。丢失、过期或长时间没有观测后，重新出现的候选获得新 ID。全部相机都移动时，目前不足以建立同一物体的持续对应关系。

场景障碍来自所有有效深度像素，与目标是否被 SAM 找到无关。低分或相互矛盾的表面仍保留为可能的障碍。只对 JSON 点云抽样，路径检查使用完整内部点集。遮挡后方、未看到的空间、无效深度保持未知。模型异常会清除当前可用快照，不复用旧结果冒充新推理。

这些是模型估计，不是测量误差界或无碰撞证明。默认 2 cm + 2% 的跨视角一致性容差是可配置的工程阈值，没有由该值推导精度保证。目标边界只覆盖可见表面；局部路径查询没有检查机械臂全身、动力学、夹持物完整形状或连续体积。机器人自身的可见表面也暂未从场景中去除。

## 模型适配

`backends.py` 延迟加载模型，常驻复用；导入模块不会下载模型。

- SAM 3.1：使用官方 multiplex 模型的 **3.1 detector 与 tri-neck backbone**，严格加载 detector 权重；没有把 3.1 权重宽松加载到旧 SAM 3 架构。CUDA 推理在局部 BF16 autocast 上下文中执行，保留 FP32 权重；图像编码和文本分割均在该上下文内。当前按帧检测并缓存同帧图像特征，不是 SAM multiplex 视频跟踪。
- DA3-LARGE-1.1：输入同步 RGB 与已知 K/E；将模型原始输出按实际内参映射回原始 RGB 射线，保留无效边缘，不把图像缩放误当成深度尺度变化。
- 官方 DA3 的相机中心 Sim(3) 对齐对恰好两个视角会退化。适配器针对双视角使用“标定基线长度 / 预测基线长度”缩放 camera-Z depth，再用输入 K/E 反投影；三个及以上视角沿用上游方法。该替代步骤、尺度和预测相对旋转/基线方向误差写入 provenance；它不能消除模型深度误差。原始预测保存用于复核。
- 模型仓库 revision 固定；实际 checkpoint SHA-256 写入结果。缺少权重或依赖会明确报错，没有其他模型或伪结果回退。

CoTracker3 尚未接入。本版 `attachment`、`slip`、`release` 始终是 `unknown`：区域共移或跟踪 ID 本身不证明抓持成立。后续加入材料点跟踪时，应单独验证这些证据。

## 安装与离线运行

```bash
bash scripts/setup_foundation_vision.sh
PYTHONPATH=src .venv-foundation-vision/bin/python scripts/evaluate_foundation_vision.py inspect
```

安装脚本使用独立 `.venv-foundation-vision` 和固定版本的上游源码，不修改已有 `.venv`。它安装依赖，不主动下载权重。模型文件和 vendor 源码不提交到仓库。

DA3 可以独立验证，完全不构造 SAM：

```bash
PYTHONPATH=src .venv-foundation-vision/bin/python scripts/evaluate_foundation_vision.py run-depth \
  --manifest runs/foundation_vision_preparation_001/input/manifest.json \
  --output runs/foundation_vision_depth_local_001 \
  --device cuda:0
```

输出目录必须是新的；已有结果不会覆盖。首次模型加载可能下载权重，`--local-files-only` 禁止下载。冷启动与持续推理应分开报告。

联合管线需要获授权的 SAM 3.1 checkpoint，以及非空 `targets`：

```bash
PYTHONPATH=src .venv-foundation-vision/bin/python scripts/evaluate_foundation_vision.py run \
  --manifest /path/to/target_manifest.json \
  --sam-checkpoint /path/to/sam3.1_multiplex.pt \
  --output runs/foundation_vision_joint_local_001 \
  --device cuda:0
```

也可以通过已有的、具备仓库访问权限的 Hugging Face 登录环境下载 checkpoint。此前访问受限的问题已于 2026-11-05 解决：官方固定版本权重已下载并校验，路径为 `artifacts/foundation_vision/models/sam3.1/sam3.1_multiplex.pt`。SAM 单独分割验证已通过；真实联合验证尚未完成。

## Manifest 与程序集成

输入 manifest 格式：

```json
{
  "schema_version": "foundation-vision-input-1",
  "source_kind": "saved_public_observations",
  "targets": [{"key": "target", "text": "short visual object description"}],
  "frames": [{
    "frame_id": 0,
    "captured_at_s": 0.0,
    "observation_path": "public_observation.json",
    "geometry_path": "public_geometry.json"
  }]
}
```

路径相对 manifest 所在目录解析。也可直接给出 `cameras`，每个相机含 `name`、`rgb_path`、`intrinsics`、`world_to_camera`、`observed_at_s`；顶层帧可带 `hand_world`。具体严格校验见 `bridge.py`。深度单独验证允许 `targets: []`，不能将它报告为目标检测成功。

```python
from gpt_field.foundation_vision.backends import Sam31Segmenter, DA3DepthEstimator
from gpt_field.foundation_vision.bridge import frame_from_step_view
from gpt_field.foundation_vision.engine import VisionEngine
from gpt_field.foundation_vision.types import TargetSpec

vision = VisionEngine(Sam31Segmenter(), DA3DepthEstimator())
frame = frame_from_step_view(
    view, geometry, frame_id=frame_id, captured_at_s=sensor_time_s)
snapshot = vision.process(frame, [TargetSpec("target", target_description)], now_s=now_s)
executive_feedback = snapshot.summary()
corridor_evidence = vision.query_path(world_waypoints, radius_m=tool_radius_m)
```

传感器时间和 `now_s` 必须使用同一个时钟。默认 1 秒 freshness 是可配置的运行条件；模型冷加载不应被算作一条可立即使用的新鲜观测。`estimated_clear` 仅表示采样位置存在一致的可见射线证据，不能直接触发机器人执行。

## 验证边界

运行测试：

```bash
.venv/bin/python -m pytest -q tests/test_foundation_vision_*.py
```

测试涵盖输入隔离、坐标/内参映射、跨视角关联、遮挡/未知空间、过期和丢失、路径障碍、模型失败和 CLI 结果保护。注入预测的测试只验证软件契约，不是模型能力证据。

最终合并回归：**82 项通过**。机器可读结果为 `runs/foundation_vision_preparation_001/validation_tests.xml`；汇总、来源哈希和验证范围见同目录 `validation_report.json`。

本次真实 DA3 输入来自已保存的公开双 RGB 和同帧标定；提取时验证了机器人状态、图像像素和腕相机一致性，来源哈希在 `runs/foundation_vision_preparation_001/input/extraction_evidence.json`。没有手工目标区域或真值深度。失败运行和后续修正运行分别保留。

离线 CLI 不导入模拟器、不执行机器人动作、不调用 Astra，不修改现有控制器。本次结果不能证明三分钟内完成任务、真实抓持改善、无碰撞或 OOD 泛化；这些仍需模型联合验证和后续独立评测。

## 本次真实模型结果

`runs/foundation_vision_preparation_001/da3_real_003/` 使用一帧已保存的 512×512 双视角 RGB，在 CUDA GPU 上运行固定 checkpoint。该样本没有深度真值：

| 项目 | 结果 |
| --- | --- |
| 冷加载权重 | 9.451 s |
| 冷运行整帧（含加载、深度、几何） | 10.944 s |
| 同一帧重复热运行的 DA3 adapter（含预处理、推理、射线重采样） | 0.405 s |
| 热运行网络 forward | 0.251 s |
| 外部视角跨视角支持 / 有效像素 | 12,363 / 260,100（4.75%） |
| 腕部视角跨视角支持 / 有效像素 | 29,555 / 261,121（11.32%） |
| 外部 / 腕部视角矛盾点数 | 91,607 / 54,919 |

支持比例的分母包含不重叠或被遮挡区域，不能解释为深度准确率。矛盾统计说明不能仅凭模型成功返回就接受几何；目前没有建立厘米级精度或可靠空闲空间的证据。热运行只是对同一样本重复测时，不是第二个测试样本，也不包含 SAM 或完整机器人控制。

双视角基线尺度为 0.878077 m/model-unit；预测相对旋转和基线方向与输入标定均相差约 0.043 rad。适配器记录这些差异，没有据此改阈值。原始深度、相机预测和标定保存在 `cold_native_prediction.npz` 与 `warm_native_prediction.npz`，分项耗时在 `timings.json`。

后续 CLI 运行会逐帧自动保存 `native_prediction_<frame_id>.npz`，在 JSONL 记录其相对路径、SHA-256 和后端分项耗时，便于复核深度尺度和射线重采样。

最初的 SAM 检查只验证了 detector/processor 构造 API；当时权重不可访问，`sam_adapter_load.json` 保留 GatedRepoError。以下新增结果取代这一访问阻塞状态，旧记录保留。

## SAM 授权与真实分割验证（2026-11-05）

结果目录：`runs/foundation_vision_sam_validation_001/`。官方 checkpoint 共 3,502,755,717 字节，SHA-256 为 `0567debeec80ba4ac6369540c6c248025283cb3ff2b92827509e57e2b3541cb6`，与官方元数据一致。

首次严格加载通过，但分割时遇到 BF16 激活与 FP32 线性层的 dtype 不匹配。已按照上游调用方式补充局部 CUDA BF16 autocast，并将分数转换为 FP32 后导出 NumPy。失败记录及修复前源码保存在 `sam_inference_001/`；原来的提示词、阈值和 RGB 均未改变。

修复后 `sam_inference_002/` 使用同一保存双视角帧、固定提示词 `can`、阈值 0.5：外部和腕部视角各返回一个候选，分数分别约 0.835 和 0.509。加载约 14.79 秒；加载完成后的首次双视角分割约 1.17 秒（含 CUDA 同步，不是同帧缓存返回）。保存了所有候选 masks、分数、输入 RGB 和来源哈希。

人工目视检查支持外部掩码与罐体对齐；腕部只覆盖可见的部分罐体，分数接近阈值。这里只有一个独立样本，没有真值分割准确率、时序跟踪或几何精度验证。修复后的完整模块回归 **87 项通过**。该次授权验证没有运行机器人，T04 当时保持暂停；用户随后授权的新完整任务记录如下。

## 视觉辅助 Astra 的 live 接入

新增 `foundation_vision/live.py` 和 `scripts/evaluate_foundation_vision_task.py`。后者加载原 UC7.1 runner，在两个本地视觉模型预加载成功后，通过 client factory 注入 `FoundationVisionAgentClient`。原 runner 的任务定义、订阅通道、初始模型规划和低层控制器继续使用；源码和视觉配置纳入同一轮冻结校验。

每次执行决策前，SAM 按新初始计划中的物体/部件描述，以及可选 `--vision-query TEXT`，生成候选区域；DA3 估计同步双视角深度。受限大小的摘要进入实际 `context.foundation_vision`，Astra据此结合RGB选择目标、动作和自己的障碍盒。完整候选掩码、原始快照、实际请求摘要及接受计数保存到 `trial/foundation_vision/`。

该模式为 `vision_conditioned_executive`：**没有自动把 DA3 点云变成规划障碍，也没有自动视觉动作门禁**。低层路由仍检查 Astra 输出的障碍盒；direct/native 动作和机械臂全身不由本模块检查。不能把一次成功归因于自动三维避障。

live 默认观测有效期为 5 秒，通过 `--vision-max-age-s` 显式配置。时间戳取公开观测进入适配器的时刻，包含前置校验和本地推理；同 step 缓存保留原时间戳。摘要中的年龄是构建请求时的年龄，不等于等待 Astra 返回后执行动作时的年龄。过期、未知和身份歧义不会被改写成有效几何。

任务 wrapper 使用原 runner 的命令与参数，额外要求 `--sam-checkpoint`、`--da3-model` 为本地文件/目录，支持 `--vision-device` 和重复的 `--vision-query`。独立视觉环境需要订阅认证依赖；本次使用与原环境相同的版本：

```bash
.venv-foundation-vision/bin/python -m pip install 'PyJWT[crypto]==2.15.1'
```

## T04 can obstacle 完整任务结果（2026-11-05）

运行：`runs/foundation_vision_t04_can_obstacle_20261105_001/`。本次保留额外 can 障碍、原始初始状态、Astra high 订阅通道和 100 mm transit 命令上限，使用 `--vision-query can`。启动前 **105 项测试通过**；SAM+DA3 保存 RGB 联合检查通过。新初始计划和完整任务各执行一次，没有自动重试。

| 指标 | 结果 |
| --- | --- |
| 原始 LIBERO 结果 | 成功 |
| 控制动作 | 604（603 fragment、1 direct、0 native macro） |
| Astra 调用 | 初始规划 1 次＋执行 11 次；API key 0 |
| 控制耗时，扣除视频 | 432.422 s |
| 初始模型规划＋控制，扣视频、不含启动 | 451.173 s |
| 完整 wrapper 墙钟，含加载/启动/reset/视频/收尾 | 569.751 s |
| 禁止接触点记录 | 0；允许的抓持和支撑接触存在 |
| 视觉处理累计 | 14.734 s，11 次完成并进入被接受的请求 |
| 可用三维目标中心 | 0 |
| 源码终态检查 | 396 个文件未改变 |

本次**没有达到三分钟目标**。77 个目标帧中，34 个双视图都没有候选，26 个仅单视图有候选，11 个存在多候选歧义，6 个跨视图关联被拒绝；请求构建时无过期帧。算法拒绝关联不证明实际对象不同，也可能由分割、深度或投影误差引起。Astra两次返回非空障碍图更新；本次任务成功和零禁止接触记录不能证明新视觉模块带来了因果性的速度或避障收益。

最终视频为 `trial/episode.mp4`，分析为 `analysis.json`。运行已正常退出，专属 supervisor/socket 已关闭，未启动其他任务或重试。

## T08 bottle obstacle 完整任务结果（2026-11-05）

运行：`runs/foundation_vision_t08_bottle_obstacle_20261105_001/`。使用原炉灶已开启的瓶子障碍布局，任务为 `put both moka pots on the stove`，init0/seed7；额外视觉查询为 `bottle`。运行源码与上述 T04 相同，复用其 105 项测试并保存来源，T08 无副作用预检查通过。仅启动这一轮，无自动重试。

| 指标 | 结果 |
| --- | --- |
| 原始 LIBERO 结果 | 成功，590 个策略动作 |
| Astra 调用 | 初始规划 1 次＋执行 11 次，均 high 订阅；API key 0 |
| 控制耗时，扣除视频 | 464.082 s |
| 初始模型规划＋控制，扣视频、不含启动 | 500.508 s |
| 完整 wrapper 墙钟，含加载/启动/reset/视频/收尾 | 614.065 s |
| 策略阶段禁止接触记录 | 400 个接触点采样，全部为第一把壶与炉头 |
| 视觉处理累计 | 13.208 s，11 次完成并进入被接受的请求 |
| 可用三维目标中心 | 0 |
| 源码终态检查 | 396 个文件未改变；9 项独立终态检查通过 |

**未达到三分钟或无禁止接触目标。** 第 277 步首次记录壶—炉头 `manipulated_object_scene` 接触，不是瓶子接触。400 是接触点采样数，不是独立碰撞次数。分类要求接触法向朝上且位于目标区域；原事件没有保存触发分支，不能进一步认定为边缘撞击、区域外支撑或误报，也没有力/冲量证据判断严重度。详见 `monitoring/collision_classification_review.json`。

成功边界也有明确限制：第 590 个策略动作触发原始成功时，第二把壶仍被夹持，策略尚未自主释放。原评估器随后执行了独立的 38 步释放/稳定诊断（开爪 10、抬升 8、稳定 20），诊断通过，且 `affects_completion=false`。这些不是策略动作，不能据此声称策略已自主释放；诊断后的禁止接触累计为 1013，与控制期的 400 分开报告。

66 个目标帧中，无候选 42、仅单视图 17、多候选歧义 3、跨视图拒绝 4。11 个请求在构建时均未过期（年龄 1.143–2.398 s），所以缺失目标几何不能归因于过期。Astra 返回 3 次非空障碍图更新，但本次仍不能证明新视觉模块的因果性速度或避障收益。

视频为 `trial/episode.mp4`，汇总为 `analysis.json`，终态视觉诊断为 `monitoring/vision_evidence_terminal.json`。运行进程和专用 supervisor/socket 均已关闭，未追加其他任务或重试。
