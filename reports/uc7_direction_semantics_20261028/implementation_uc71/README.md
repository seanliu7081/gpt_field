# uc7.1 版本 A 交付

实现范围为 [final_plan.md](../final_plan.md) 的 A0–A9。版本为 `agent-speedup-v2-user-check-7.1`，方向语义标签为 `external_view_v1`。2026-10-28 经用户批准完成一次 Task 6 relayout：官方成功、方向标准通过，严格物理验证未通过。其余 V2/V3 任务仍须先获得用户批准。

## 已实现

- 外部相机标定生成唯一的水平方向表；完整标定、图像约定及摘要保存在 `preparation.view_frame`，由现有审批上下文哈希绑定。
- LIBERO 和 RoboCasa 的 prepare → approval → PlanLock → policy ctx → 执行请求使用同一冻结坐标系。执行请求每次都带方向；没有区域的计划也带坐标系。
- 实际区域求解器改为沿冻结方向投影。缺失标定时方向区域返回 `view_frame_unavailable`，旧世界轴解释不会作为后备。
- 规划、执行、审核界面采用外部视角词；世界米数仍用于数值 bounds 和动作目标。
- 审核输出显示各区域的世界方向；部件文字中的 `world +x` 等措辞在界面及 trial 日志中标出，仅作显示，不新增 plan issue 或否决。
- 腕部图像描述如实说明实际 quarter turns；没有旋转或翻转图像。
- uc7.1 拒绝加载旧版本审核；历史计划仍可用原始 JSON 或旧快照查看。

E1 的选点偏好和 E2 的图像旋转均未加入。颠倒画面的特殊策略延后：本版保留 `away = up × right`，仅记录 `roll_deg` 和 `inverted`。没有新增模型决策阶段、动作探测、任务专用分支或回合停止上限。

## 冻结物

| 物件 | 用途 |
|---|---|
| [uc7_baseline_source.tar.gz](uc7_baseline_source.tar.gz) | A0：改动前所有 src、scripts、tests 及项目配置，845 个文件 |
| [baseline_manifest.json](baseline_manifest.json) | A0 逐文件摘要和原始 git 状态 |
| [uc7_baseline_runtime/](uc7_baseline_runtime/) | 已展开、逐文件验证的旧版本；后续 v3/uc7 基线复现只使用此目录 |
| [uc71_release_source.tar.gz](uc71_release_source.tar.gz) | 通过 V0 后冻结的 uc7.1 源码和测试 |
| [uc71_release_manifest.json](uc71_release_manifest.json) | 发布摘要、运行文件摘要和冻结当时的未授权实跑状态；后续授权单独保存在各次运行中 |
| [uc71_changes.patch](uc71_changes.patch) | 相对 A0 的准确改动，不混入本轮以前的未提交改动 |

基线 tar SHA-256：`31acb3fde93a2bbb05a472bacf3def1f7a6e17495db23e2b5fea31eee4526f7e`。

uc7.1 tar SHA-256：`b3e8f84f3bf1b7603fad0ae0d8c4bee8a5a125b50ab08e320394b4069af57a68`。

未提交 git；v3 本身的源码未改。工作区共享模块已是 uc7.1，不能从工作区直接启动旧 v3。获得对应试验授权后，应使用上述基线目录中的 runner，并将 `PYTHONPATH` 指向该目录的 `src`；Python 解释器仍使用 `/workspace/gpt_field/.venv/bin/python`。快照不复制外部模拟器或依赖环境。

## V0 离线结果

| 项目 | 结果 |
|---|---|
| 改动前回归 | 279 通过、59 失败 |
| uc7.1 相同回归 + 新测试 | 350 通过、59 失败 |
| 新增测试 | 71/71 通过 |
| 新增失败 / 丢失原测试 | 0 / 0 |
| BDDL 可判定方向约定 | 41/41；另有 4 个 middle 不可判定 |
| 锚定侧区符号 | 31/31 |
| 历史批准计划的空间 schema | 48/48；不代表旧审批可以在新版本执行 |

证据见 [pytest_comparison_final.json](pytest_comparison_final.json)、[uc71_pytest_final.log](uc71_pytest_final.log)、[direction_evidence.json](direction_evidence.json)。59 个失败均在 A0 已存在，涉及旧夹具/接口预期以及已知 v3 prompt 版本预期；本轮没有改写这些旧测试或宣称整个仓库测试全绿。

新增验证包含：真实求解器的 LIBERO/斜向 RoboCasa 掩码、缺失方向后的缓存失效、合成相机、标定及审批篡改检查、11 次连续执行请求、两条 runner 的 fake 全流程、实际图像转换分支及镜像检测。真实 t6 方向约为 `(-0.00000172, 1)`，按标定保留，未硬编码为整数轴。

方案中数值 epsilon 的出处有一个小误差：当前 `geometry._unit` 实际用 `1e-12`，实现保留方案明确指定的 `1e-9`，没有据结果调参。

## V1 只调规划

使用冻结版本、Astra high 和 ChatGPT subscription，覆盖 LIBERO-10、g5、t6 relayout、RoboCasa coffee，共 13 个准备场景。准备过程只重置环境、生成审核图和草稿，不批准或启动任务。

13/13 份草稿均已生成，共 **13 次规划调用**，执行调用、任务动作、审批和 API-key 调用均为 **0**。392 个运行文件在全过程中保持冻结。13 个初始观测均与所选历史基线一致。

方向与部件指向检查 **38/38 通过**：t6 官方及重排布局均为 `right_of → ≈(0,1)`；g5 为 `front_of → ≈(1,0)`；t3/t4/t5/t9 没有新增区域；t4 盘子仍为格子 41/48；t5 指向实体后侧隔间，未替换为 behind 区域；部件文字中无带符号的世界 X/Y 轴措辞。

**原始草稿合法性为 12/13，因此 V1 整体尚未通过。** Task 9 再次生成 `do=turn, sense=close`，违反既有的“非 slide 时 sense=any”规则。历史 uc7 原始草稿已有同一错误，后来经审核修改；它不是方向检查失败，但本次仍记为无效草稿，没有重复调用模型来覆盖这个结果。

已保存一份[未应用的审核修正建议](task9_review_correction.json)：仅将该字段改为 `any`，经现有校验通过；正式任务仍要求关上微波炉。此建议不修改原始草稿、不批准或执行任务，待 Task 9 审核时处理。

详细证据：[V1 对比报告](v1_draft_comparison.md)、[V1 对比数据](v1_draft_comparison.json)、[完整性审计](v1_integrity_audit.json)、[原始记录](v1_planning/summary.json)、[准备日志](v1_planning.log)。runner 退出码 0 仅表示 13 次准备完成，不代表 V1 全部标准通过。

## V2 已批准的 Task 6 relayout

用户以“ok retry task6”批准一次测试，使用已展示的同一重排布局、冻结的 uc7.1 和 Astra high。原始 V1 草稿保持不变，复制到独立审核目录后批准并消费；复用其一次规划调用。本轮只启动一次任务，无自动重试、无新规划调用、无 API-key 调用。冻结源码未改变，测试进程已退出。

| 项目 | 结果 |
|---|---|
| 官方任务结果 | `task_success=true`、`libero_success=true` |
| 严格物理结果 | `v2_success=false` |
| 任务完成时间 | **604.86 秒（10 分 4.86 秒）**，`766.73` 秒控制耗时减去 `161.86` 秒视频渲染；不含准备、原 V1 规划、审批和结束后诊断 |
| 模型调用 | 本轮执行 **14 次**；加上复用的规划 1 次，共 **15 次**；均为 Astra high / subscription |
| 任务动作 | 1187 |
| 首次布丁释放方向 | 第 10 次调用释放前目标 `(0.15, 0.21, 0.446)`；`y > 0.079`，方向标准通过 |
| 后续修正 | 首次放置超出有限目标框；模型随后自行重抓，第 14 次调用移向 `(0.15, 0.145, 0.50)`，此时触发官方成功 |

方向修复在本次布局中生效，但官方成功不代表最终物理状态完整：终止时布丁仍被夹持在桌面上方，COM 约 `(0.1510, 0.1493, 0.5042)`，距区域边缘仅 0.66 mm；`not_held`、`at_rest` 和支撑检查失败。白杯的严格检查通过。已有的结束后释放静置诊断使两者均通过，布丁 COM 约 `(0.1507, 0.1482, 0.4505)`、边缘余量 1.83 mm；**诊断不改变本次终止时的 `v2_success=false`**。没有加入 E1 偏好、修改代码或向执行模型提供隐藏目标框。

视频已核验为 1600×960、20 fps，包含 XY/XZ/ZY、活动控制场及解释字幕，终止帧已清除活动目标。248 秒视频时长不是任务完成时间；画面 `SUCCESS` 表示官方判定。

证据：[本次运行与结果](../../../runs/agent_speedup_v2_uc71_relayout_t6_20261028_001/README.md)、[结果 JSON](../../../runs/agent_speedup_v2_uc71_relayout_t6_20261028_001/results_summary.json)、[视频](../../../runs/agent_speedup_v2_uc71_relayout_t6_20261028_001/trial/episode.mp4)。旧 uc7 同布局运行提前停止，不能用它的耗时计算本次提速率。

## 其余完整验证

- V2：t6 relayout 已完成上述一次；g5、t6 官方布局、RoboCasa coffee 等待批准。前两者判定方向修复；后两者分别作参考与接线检查。
- V3：按最终方案的 14 个场景在同一冻结版本验收；V2 结果仅在源码摘要相同的情况下合并。按用户最终决定，各任务须实际成功，即现有 `summary.task_success == true`，不能只凭方向正确判通过。严格物理结果 `v2_success` 继续单独报告。
- 每次任务启动前提供布局预览并等待用户批准。任务完成时间沿用 `control_without_video_s`，不包含准备/审批/视频渲染；未完成任务不报告为完成耗时。

当前仅有一次 Task 6 relayout 结果，尚无完整 sweep 成功率或提速结论。
