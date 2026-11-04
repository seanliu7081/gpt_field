# RoboCasa365 × UC7.1 transit50

本目录包含中文接入计划、独立仿真环境与资产、365项任务清单，以及首个RestockPantry的arm+mobile适配代码。

**状态：测试前准备完成，等待用户批准。** 仅导入、静态preflight和纯fake测试；尚未验证真实场景渲染、模式切换或任务表现。

- [完整计划](PLAN_ZH.md)
- [首轮配置](configs/restock_pantry.json)
- [365任务清单](configs/task_catalog.json)
- [环境/资产记录](setup/README.md)
- [UC7.1基线审计](setup/uc71_baseline_audit.md)
- [官方任务和评估调查](setup/research_report.md)（评估资料，不传给策略）

## 当前可执行的准备检查

```bash
cd /workspace/gpt_field
robocasa365/.venv/bin/python robocasa365/run.py preflight
env PYTHONPATH=robocasa365:src GPT_FIELD_USER_CHECK_TRANSIT_ACTION_M=0.05 \
  robocasa365/.venv/bin/python -m pytest -q robocasa365/tests
```

preflight只读取代码/配置并散列资产。模型固定Astra/high、subscription-only；`run.py`在任何策略模块导入前固定transit命令0.05m，并拒绝冲突环境变量。单段transit目标仍最多0.50m。

## 获得测试授权后才使用

以下是待批准的调用示例，不是已执行记录。`prepare`会创建真实环境并调用模型生成计划；它不属于无仿真setup。`run`还要求已有UC7.1逐步审核，通过一次性claim绑定代码、配置和初态。模型计划审核与“允许开始测试”不是同一件事；本轮尚未进行这两项操作。

```bash
# 仅在用户批准后：生成首轮真实初态和模型草案
robocasa365/.venv/bin/python robocasa365/run.py prepare --test-approved \
  --review-dir robocasa365/runs/restock_pantry_001_review --seed 7

# 查看本地审核页；保持127.0.0.1，通过SSH/IDE端口转发访问
robocasa365/.venv/bin/python robocasa365/run.py review \
  --review-dir robocasa365/runs/restock_pantry_001_review --port 8765

# 完成计划审核后，执行一次；目录必须全新
robocasa365/.venv/bin/python robocasa365/run.py run --test-approved \
  --review-dir robocasa365/runs/restock_pantry_001_review \
  --output robocasa365/runs/restock_pantry_001 --seed 7
```

也可使用 `approve --revision N --reviewed-steps ...` 记录实际完成的逐步审核，不能伪造人工审核。CLI的 `--test-approved` 只是授权声明，必须有用户的真实同意。本轮没有自动确认、启动或后台队列。未来长运行沿用工作区独立supervisor流程，`autorestart=false`，不使用自动重试。

运行中在对应trial目录写入 `operator_stop.request.json` 可在完整反馈边界停止。保存用户停止、订阅错误、基础设施错误的原始结果，不能改标为任务完成。

## 目录

```text
robocasa365/
  PLAN_ZH.md / README.md
  configs/       首轮配置、365任务清单
  rc365/         观测/动作适配、移动策略扩展、review/run入口
  tests/         纯离线fake测试
  setup/         安装、依赖锁、资产校验、审计与验证证据
  vendor/        固定版本官方RoboCasa、robosuite与解压资产
  assets/        保留的官方下载ZIP
  .venv/         独立Python3.11
  runs/          获批后新建；本轮未创建真实trial
```

当前runner只接首个任务。移动能力是短时原生控制，不含自动全场景导航或全身碰撞规划。365项清单不是已验证的任务覆盖率。首轮采用target厨房/对象split的无上限探索pilot，官方参考horizon2550单独报告；RestockPantry不属于target50，不能据此声称target50或完整365评测成绩。
