# uc7.1 V1 规划草稿对比

状态：**failed**；已生成 13/13 份草稿。

本报告只检查规划与冻结方向。没有审批、执行动作或任务成功结论；V2/V3 仍需用户批准，并以任务成功验收。

方向及部件身份检查：13/13 个已完成案例通过，38/38 项检查通过。草稿有效性：12/13。整体 V1 状态保留为 failed。

| 案例 | 状态 | 与历史初始观测相同 | 方向区域 |
|---|---|---|---|
| libero_10_t0 | 通过 | True | — |
| libero_10_t1 | 通过 | True | — |
| libero_10_t2 | 通过 | True | — |
| libero_10_t3 | 通过 | True | — |
| libero_10_t4 | 通过 | True | — |
| libero_10_t5 | 通过 | True | — |
| libero_10_t6 | 通过 | True | E: right_of C → world (-1.72339e-06,+1) [image-right] |
| libero_10_t7 | 通过 | True | — |
| libero_10_t8 | 通过 | True | — |
| libero_10_t9 | 有偏差 | True | — |
| libero_goal_g5 | 通过 | True | C: front_of B → world (+1,+7.26818e-08) [toward external camera] |
| libero_10_t6_relayout | 通过 | True | E: right_of C → world (-1.72339e-06,+1) [image-right] |
| robocasa_coffee | 通过 | True | — |

Task 4 按左右盘子的物体描述匹配 ID，再比较外部 8×8 网格；相邻包含对角格。Task 5 检查书本指向收纳盒的实体后侧隔间，不能用 behind 区域代替。

原始布局基线按目录顺序选择最早的 revision 7 草稿，不按成功结果挑选；t2/t3/t6 分别为 002/003/009，t4 固定为 001。g5 的历史对照为 v3 保存的规划草稿；其执行结果不作为 uc7 基线。完整前后部件、步骤引用、逐项证据与草稿哈希见 [v1_draft_comparison.json](v1_draft_comparison.json)。人工核对原始网格图像的记录见 [v1_visual_comparison.json](v1_visual_comparison.json)。

## 偏差

- libero_10_t9: plan_schema_and_semantics_valid（历史原始草稿已有相同校验问题；旧审核计划后来已修改） — {"valid": false, "issues": [{"code": "ignored_selection", "step": 1, "pid": "D", "text": "sense must be any except for slide"}]}
