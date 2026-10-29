# uc7.1 × RoboDojo Make Toast Standard

The new host's **Isaac Sim 5.1 rendering check passed on an RTX 4090 with driver 580.82.09**. Eight real 640×480 captures were saved and inspected. NVIDIA's default fast shutdown exited with code 0 and returned GPU memory to its 17 MiB baseline. A monitor race incorrectly marked the raw final result as failed; the independent assessment preserves that raw record and explains the correction. See [new-host evidence](validation/new_host_20261029/).

The GPU has 24 GB VRAM with approximately 23.5 GiB free at idle. A model-free native layout-0 reset produced all three camera views, used a sampled peak of **5,726 MiB (5.59 GiB)**, and left at least **18,356 MiB (17.93 GiB)** free. It exited with code 0 and released its GPU allocation. See the [native diagnostic review](validation/new_host_20261029/native_preview001_review.json) and [head view](validation/new_host_20261029/native_preview_001/preview/cam_head.png).

An isolated compatibility overlay repaired the `hid` and `pkg_resources` imports. A second native reset passed with clean extension startup and the same memory use. A **300-second physics-paused wait** preserved all joint positions, physics time and step counts. Both fresh resets produced identical robot state; rendered images have small recorded pixel differences. See the [second native review](validation/new_host_20261029/native_preview002_review.json).

The published assets contain a missing camera-stand normal texture and invalid robot material references. Original assets are preserved, and successful rendering does not certify full asset fidelity. Worker integration, motion calibration and approval preparation remain pending. The supported stock fast-shutdown mode does not prove full plugin teardown.

**No Make Toast policy test or model call has run. No start approval has been consumed.** The diagnostic preview is available; validated control, a planning draft and the remaining readiness checks are required before requesting test approval. NVIDIA licence acceptance permits setup only.

- [New-host migration instructions](MIGRATION.md) and [migration inventory](migration_manifest.json)
- [Current reviewed plan](PLAN.md), [model-input contract](INPUT_CONTRACT.md) and [adapter differences](docs/adapter_diff.md)
- [Implementation decisions](docs/implementation_decisions.md) and [pinned sources](configs/source_pins.json)
- [Original independent review](review_20261028/REVIEW.md)
- [Unmodified original planning archive](planning_archive_20261029/)
- [Implementation status](IMPLEMENTATION_STATUS.md) and [machine-readable status](implementation_status.json)
- [Provisioning evidence](validation/provisioning/)

The intended pilot uses native `make_toast`, seed-directory 0 / layout ID 0, dual ARX arms, three 640×480 cameras and Astra high through subscription transport. The official horizon is 1,400 native actions (56 simulated seconds). No extended continuation or live execution smoke is authorized.

The earlier GPU 4 / driver 610.43.02 failures belong to the previous host and remain preserved in the [A1 report](validation/provisioning/A1_REPORT.md). Their unique cause was not established. Current setup uses GPU 0, UUID `GPU-a9e034f8-a424-d7fe-6eaa-45667dd2c6b2`. The frozen uc7.1 core passed all 851 file hashes; all adaptations and runtime files stay here.

Latest combined offline validation: **209 tests passed** ([record](validation/new_host_20261029/offline_tests_native_components_002.xml)); the original [84-test record](validation/offline_validation.json) is preserved. Synthetic tests do not validate physical robot tracking or task execution.
