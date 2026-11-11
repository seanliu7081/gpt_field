V3 / astra_tools_r1 与 agent_speedup_v2_obstacle_harness_100mm 的区别

两者共用核心动作执行和路由代码，但不是同一个完整 harness。这里的 r1 指最近两轮测试使用的封存版本，而非当前工作树中的 r4。

逐字节验证一致的文件：agent_speedup_v2_user_check_fragments.py、agent_speedup_v2_user_check_routing.py、contact_skills.py、fine_distance_planner.py、action.py、robot_facts_v2.json。两者的 transit 100 mm / contact 5 mm 命令上限、transit 4 mm / contact 2 mm 到位容差及底层 20 mm 网格相同。 不过 V3 的 RoutedFragment 会将原 ActionAdapter 替换为 HoldingActionAdapter：当旋转命令恰好为零时，加入1 nrad的姿态刷新量以刷新OSC目标；另外还套有每步 RobotGuard。因此共享源文件不等于整条控制调用链或下发动作逐位相同。T04 障碍场景 BDDL 和初始状态也完全相同。哈希记录见 standalone_comparison.json。

| 方面 | 独立 100 mm 包 | V3 r1 |
|---|---|---|
| 活跃策略 | 初始 UserCheckV2Client 计划；随后 AgentFirstV2Client / AgentFirstV2Policy | FoundationVisionHarnessClient / FoundationVisionHarnessPolicy |
| 初始计划 | 单独模型规划、自动批准、再启动执行阶段 | 无单独审阅计划阶段，直接进入自主控制循环 |
| 执行接口 | execute、短 motion、可选 native macro、observe、finish | execute、measure、finish、blocked |
| 批次 | 1–6 个片段，模型给步数预算，累计预算≤240 | 1–24 个片段，每段1–3候选，封装器分配片段预算 |
| 历史 | 保留整轮文字、已接受输出、测量结果和不透明推理；最近6组图像 | 当前双视角RGB，最近6条压缩契约摘要及执行器当前反馈 |
| 视觉工具 | 无 SAM/DA3 工具 | 可按需请求分割、点和表面测量；普通动作无需预加载 |
| 障碍图提示 | 已要求首次建图、保留相关障碍、载物clearance、换目标后更新 | 主要说明地图缓存/替换/清空语义，没有完整维护提示 |
| 机器人保护 | 没有 V3 的统一 RobotGuard 包装层 | 每步增加已知关节极限和采样自碰撞检查；不等于环境避障保证 |
| 停止条件 | raw LIBERO 成功立即结束；模型 finish 只是再观察 | raw 成功不会提前结束；由模型 finish/blocked 结束 |
| 物理评价 | 另报物理诊断/v2_success，特定情况允许释放并静置的反事实诊断 | 终态与保持原夹爪10步后均检查 raw+strict；不替策略松爪 |
| 计时/录像 | 同步录像计时扣除；初始模型计划单列 | 有界异步录像；在线时间含模型/工具等待、记录与提交开销 |
| 部署 | 独立双环境及打包LIBERO代码/资产 | 最近运行依赖现有模拟器、LIBERO和可选视觉模型路径 |

关键关系：r2 的地图维护提示部分恢复了这个独立包已有的要求。因此 r1 不能描述为该包“仅增加 SAM/DA3”；它同时改变了策略提示、记忆、动作协议、每步保护和终止/评分流程。r2 与 r1 的比较也不等于独立包与 r1 的比较。

成功判定差异尤其重要：旧包 raw=true 时可能仍由夹爪持物；它会结束控制并另外做物理诊断。V3 要求模型自行完成任务并声明结束，组合成功还检查终态和不改变夹爪状态的保持后结果。不能把两种 endpoint 的成功率或完成时间直接当成同一指标。

历史包参考：549 动作，14 次模型调用（包含1次初始规划）；287.321秒为扣录像后的控制时间，另有17.484秒初始规划，总计304.805秒仍不含启动/重置及录像开销。最近 V3 r1 T04 的279.129秒是完整在线时间，并以自主结束和额外严格物理评分评价。两者没有相同计时边界或停止条件，不能仅凭这两个数宣称速度提升。

任务名也不同：包内 t04 或 t04_can_obstacle 选择加罐场景，libero_10_t4 选择官方无额外罐场景。该包只内置 T04 自定义障碍场景；libero_10_t8 是官方任务，并不会自动选择我们测试的 bottle-obstacle 覆盖场景。

本次只进行了文件/代码比较，没有启动模型或模拟器，也没有修改运行源代码。

证据入口：
- ../../agent_speedup_v2_obstacle_harness_100mm/README.md
- ../../agent_speedup_v2_obstacle_harness_100mm/scripts/launch.py
- ../../agent_speedup_v2_obstacle_harness_100mm/src/gpt_field/agent_speedup_v2_user_check_agent_client.py
- ../../agent_speedup_v2_obstacle_harness_100mm/validation/VERIFICATION.md
- runtime/src/gpt_field/foundation_vision_harness_V3/{client,protocol,executor,runner,robot_guard}.py
