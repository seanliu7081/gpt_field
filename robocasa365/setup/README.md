# RoboCasa365 独立环境准备记录

状态：`PROVISIONED_NO_SIMULATOR_RUN`。环境与默认资产准备完成；没有创建、reset、step 任何仿真环境，没有模型调用、演示下载或任务测试。实际渲染、控制器及策略联调留待用户批准测试后完成。

- Python：`../.venv/bin/python`，3.11.16。
- RoboCasa：1.0.1，官方提交 `3443cdf3dd20f9ae8ed41adba442059b506891d0`。
- robosuite：1.5.2，官方提交 `5ce6643f3092639d08f7b0f90ed1c6a84f50552c`。
- MuJoCo 3.3.1、NumPy 2.2.5、Numba 0.61.2、SciPy 1.15.3 使用 RoboCasa 上游要求。
- Pillow 12.3.0 满足 gpt_field 的 `<13` 约束；OpenCV 使用现有工程版本 4.13.0.92。
- 官方 `RestockPantry` 类与 PandaOmron 的 `HYBRID_MOBILE_BASE` 控制器配置均可导入。
- `vendor` 中两份上游仓库 tracked files 未修改。仅生成官方私有 macros、安装元数据与下载资产。

已从官方 Box 地址下载五类默认资产：textures、generative_textures、fixtures_lightwheel、objects_lightwheel、objaverse。共 5,213,425,799 字节 ZIP，解压 11,082,628,022 字节、64,503 个文件。每个 ZIP 均检查实际长度、全部 ZIP CRC，并保存 SHA-256。下载包保存在 `../assets/downloads/`，解压到官方包预期的 `models/assets/`。没有缩减默认 `objaverse + lightwheel` 对象注册表。

可选 `aigen_objs` 不属于默认注册表，未下载。`lerobot==0.3.3` 与 `tianshou==0.4.10` 仅在上游数据转换/训练工具和 bench_speed 工具中引用，本环境面向仿真与 UC7.1，因此未安装；`pip_check.log` 如实保留这两个缺失声明，不能把本环境称作完整上游训练依赖安装。环境导入不需要它们。GR1/mink、额外 robosuite_models、mimicgen 的提示也不影响内置 PandaOmron 的配置导入。

主要证据：

- `provisioning_report.json`：源码提交、资产总量及未执行测试声明。
- `import_check.json`：实际解释器、包版本、类与控制器配置导入结果。
- `requirements-runtime.in` / `requirements-runtime.lock`：运行依赖与精确安装版本。
- `assets_manifest.json` 与 `asset_*.json`：每份 ZIP 的来源、长度、SHA-256、CRC/解压结果。
- `provision_assets.py`：可重用下载与校验工具，不创建仿真环境。
- `asset_heads.json` 保留初次 HEAD 探测 404 的原始结果；`asset_get_checks.json` 说明相同官方地址的 GET 正常，因此不能把 HEAD 返回误写为资产缺失。

没有修改旧 `/workspace/past_action_robocasa`、共享 Python 环境、根 `src` 或历史运行。
