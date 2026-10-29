# 已核对接口的复现命令

> 本轮已按重复准备预算失败停止新增运行，见 `online_stop.json`。以下在线命令保留实际复现接口，不构成继续消耗剩余额度的自动队列；不要重新执行已消费预约。离线汇总可直接重跑。

所有命令使用独立副本，不修改历史 `frozen_source` 或用户工作树。以下离线命令没有新模型或仿真。重复输出使用新目录／文件名，保留已有结果；报告汇总可重跑，模型准备和策略执行不能因此重跑。

```bash
R=/workspace/gpt_field/results/speedup_evidence_20261026T055123Z
PY=/workspace/gpt_field/.venv/bin/python
export PYTHONPATH="$R/baseline/src"

"$PY" "$R/baseline/scripts/evaluate_speedup_evidence.py" preflight --arm A
"$PY" "$R/baseline/scripts/evaluate_speedup_evidence.py" preflight --arm B
"$PY" "$R/tooling/baseline_replay.py" --archive /workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_t1_001 --output "$R/baseline_replay_reproduction.json"
"$PY" "$R/tooling/joint_audit.py" --root /workspace/gpt_field --output "$R/audit_reproduction"
"$PY" "$R/tooling/benchmark_display_io.py" --baseline "$R/baseline" --trace /workspace/gpt_field/runs/agent_speedup_v2_user_check_libero_10_t1_001/trace.jsonl --output "$R/display_io_reproduction" --steps 120
"$PY" -m pytest -q "$R/tooling/tests/test_baseline_protocol.py"
(cd "$R/tooling" && "$PY" -m unittest -v test_evaluation_summary.py)
"$PY" "$R/tooling/evaluation_summary.py" --root "$R" summarize
"$PY" "$R/tooling/live_results_audit.py" --root "$R"
```

## 授权与已有预约

用户已授权 Codex 对冻结矩阵逐份代审：`libero_10_t1/t5/t7`，seed 7；P5 为 episode 1–3 的 A/B 共 18 次原始运行；仅当 P5 实测通过并冻结后，才能进入 episode 4–6 的 P6，最多 18 次。授权证据为 `delegated_review_authorization.json`。每份仍须实际查看 RGB、对象、目标及全部步骤；不包括盲批、其他任务、真实机器人或预算重置。

`trial_001` 保留为 A 基础设施准备失败；`trial_002` 是唯一关联重试，`trial_003` 是对应 B 原始运行。后二者已逐份批准、消费批准并完成策略执行，不能再次启动。其后新增预约的状态以 `reservations/`、`budget_ledger.jsonl`、receipt 和实际进程为准。已有预约不重复 reserve，已有准备或已消费批准不重复 prepare／run；失败不退款。

## 成对调度和 launch-only 包装

`paired_schedule_protocol.json` 在首对动作前固定：配对条件依次进行，同一对两组在 RTX 4090 槽位 1、2 上重叠执行；首对 A 先发／GPU 1，B 后发／GPU 2；以后每对交换先发组和槽位。因此第 2 对 B 先发／GPU 1，A 后发／GPU 2。每进程 OMP／OpenBLAS 线程数为 1。共享 OS 和模型服务竞争属于固定评测环境，需保留比较限制。

`launches/prepare_then_begin_review.py` 是纯编排包装，不是 launch 生成器：它核对本轮 P5 任务、seed、episode 及授权记录，在同进程调用冻结 runner 的 prepare，成功立即调用真实 review-begin。它只开始自动审核服务计时，**不会 approve、run 或自动重试**。它仅接受 P5 episode 1–3，不能直接用于 P6。

以下是尚未预约条件的接口模板。先按冻结矩阵和账本填写 TASK、EPISODE、ARM；不要照抄已消费条件或一次启动全部矩阵。seed 由 manifest 固定为 7。

```bash
"$PY" "$R/tooling/evaluation_summary.py" --root "$R" reserve \
  --stage P5 --task "$TASK" --episode-index "$EPISODE" --arm "$ARM"
# SPEEDUP_RESERVATION 使用实际返回的reservation文件，不能猜trial编号。
# 同对先发组登记reset：
"$PY" "$R/tooling/build_launch.py" --root "$R" \
  --reservation-file "$SPEEDUP_RESERVATION" \
  --store-dir /workspace/gpt_field/.private/chatgpt_plan --enroll-reset

# 后发组等先发组preparation_result.json有效后，使用其中确切hash，替代上面的build命令：
# "$PY" "$R/tooling/build_launch.py" --root "$R" \
#   --reservation-file "$SPEEDUP_RESERVATION" \
#   --store-dir /workspace/gpt_field/.private/chatgpt_plan \
#   --expected-initial-observation-sha256 "$INITIAL_OBSERVATION_SHA256"
```

build_launch 默认两个命令都使用 GPU 1，不会自动启动。仅对新生成、从未启动的 launch 做以下编排替换：prepare 使用包装，run 保留原 runner；先发组 SPEEDUP_SLOT=1，后发组 SPEEDUP_SLOT=2。实际生效配置是 supervisord.conf；commands.json 是 build_launch 生成时的原命令记录。

```bash
"$PY" - "$R" "$SPEEDUP_RESERVATION" "$SPEEDUP_SLOT" <<'PY'
import json
from pathlib import Path
import sys
root = Path(sys.argv[1])
reservation = json.loads(Path(sys.argv[2]).read_text())
slot = int(sys.argv[3])
assert slot in (1, 2)
trial_id = reservation['reservation_id']
assert not (root / 'evaluations' / trial_id).exists(), '已有session，不重写launch'
launch = root / 'launches' / trial_id
assert not (launch / 'supervisord.pid').exists(), 'supervisor已启动，不重写launch'
config = launch / 'supervisord.conf'
text = config.read_text()
runner = str(root / 'baseline/scripts/evaluate_speedup_evidence.py')
wrapper = str(root / 'launches/prepare_then_begin_review.py')
assert text.count(runner + ' prepare ') == 1
assert text.count(runner + ' run ') == 1
assert text.count('MUJOCO_EGL_DEVICE_ID=1') == 2
text = text.replace(runner + ' prepare ', wrapper + ' prepare ', 1)
text = text.replace('MUJOCO_EGL_DEVICE_ID=1', f'MUJOCO_EGL_DEVICE_ID={slot}')
config.write_text(text)
print(config)
PY

# SPEEDUP_TRIAL填写实际reservation_id；以下只对该新launch执行一次：
supervisord -c "$R/launches/$SPEEDUP_TRIAL/supervisord.conf"
supervisorctl -c "$R/launches/$SPEEDUP_TRIAL/supervisord.conf" start prepare
supervisorctl -c "$R/launches/$SPEEDUP_TRIAL/supervisord.conf" status
```

核对 preparation_result、review/review.json 和 service_clock 的审核计时事件。包装已经开始 review-begin，不要再次调用。后发组绑定同对先发组的初始观测 hash，执行时还会重新校验；不得为通过校验替换 hash 或 reset。

## 实际逐份审核后批准

从该份 review 存储的真实初始观测取得双视图及审核网格，实际查看图像、对象身份、目标和 plan.steps 的每一步，保存审核记录。不能只看文本、复用另一份批准或脚本生成“已看过”。两份图像字节相同可记录证据，但每份计划仍须分别审阅。修改计划须保留前后版本，对两组统一协助条件。

下面 REVIEWED_STEPS 数组只填写实际审阅过的步骤编号，不得预设每份都有两步。AUTHORIZATION_NOTE 写本份审核结论及授权范围，不能用“已授权”代替真实审阅。

```bash
# 例：实际看完第0、1步后，Bash可设置 REVIEWED_STEPS=(0 1)。
"$PY" "$R/baseline/scripts/evaluate_speedup_evidence.py" approve \
  --session-dir "$R/evaluations/$SPEEDUP_TRIAL" --reviewer codex \
  --reviewed-steps "${REVIEWED_STEPS[@]}" --authorization-note "$AUTHORIZATION_NOTE"
# 仅该份批准成功且尚未消费时运行；两组按固定顺序重叠执行：
supervisorctl -c "$R/launches/$SPEEDUP_TRIAL/supervisord.conf" start run
supervisorctl -c "$R/launches/$SPEEDUP_TRIAL/supervisord.conf" status
# 终局receipt已持久化、私有supervisor全部程序退出后关闭它：
supervisorctl -c "$R/launches/$SPEEDUP_TRIAL/supervisord.conf" shutdown
"$PY" "$R/tooling/evaluation_summary.py" --root "$R" summarize
"$PY" "$R/tooling/live_results_audit.py" --root "$R"
```

对未使用包装、且确需进入已授权代审的准备结果，原 runner 计时接口如下。应在草案就绪后立即进入真实自动审核，不把自动排队或计算算成人工等待；已开始的审核不重复调用。

```bash
"$PY" "$R/baseline/scripts/evaluate_speedup_evidence.py" review-begin \
  --session-dir "$R/evaluations/$SPEEDUP_TRIAL" --authorization-note "$AUTHORIZATION_NOTE"
```

P6 必须 P5 实测闸门通过并冻结后才可 reserve，仍使用相同服务时钟和逐份批准接口。当前 P5 包装不接受 P6，不能直接套用或提前读取留出真值。live_results_audit.py 目前只审 P5 终局，不读取 P6 summary／truth。

## 时钟、重试和保留

900 秒使用同一 monotonic 有效服务时钟：准备、自动审核／调度、执行和必要同步收尾。只有真实人工审核等待单列排除；300 秒是完整服务成绩门槛。到点取消模型等待，不中断仿真步或必要持久化；边界／收尾超额仍计入完整服务并单列，不能据此扩展窗口。

基础设施重试必须有证据；reserve 保留原任务／episode／arm，显式加 `--retry-of 原reservation_id --reason 证据说明`。全包最多 4 次、每原始运行 1 次，原始结果不删除。模型轨迹不理想不是基础设施故障。所有修复和重测共享预算；同一缺陷两轮有依据修复仍未达验收时冻结候选，预算用尽停止新增运行。

启动配置显式使用 `--store-dir /workspace/gpt_field/.private/chatgpt_plan`；不读取或复制令牌到副本。只管理本轮私有 supervisor，不操作既有 t2／t3 审核服务或实例管理服务。

初始冻结运行源码不变。freeze_manifest_launch_fix.json 与 freeze_manifest_final.json 记录先前启动参数和报告工具修正，旧清单保留；新 launch-only 编排另外记录实际配置及 hash，不改写旧冻结清单。
