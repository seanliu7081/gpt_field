# uc7.1 × RoboDojo Make Toast Standard

The default for newly prepared runtime bundles/review packets is now
**official RoboDojo end-effector control**: UC71 tool targets → native link6
targets → official cuRobo IK → native joint interpolation and actuators.
The previous Jacobian controller remains available only as the explicit
`legacy_joint` backend or in historical bundles. See the
[control contract and validation status](docs/official_control.md) and
[dependency setup](docs/official_control_setup.md).
Official native motion/gripper qualification remains required before issuing
an executable full-task packet; older controller results cannot satisfy it.
Current control validation: **118 official-backend tests passed**; the full
offline run has **690 passed / 33 root-permission fixture failures**. Real native
cuRobo initialization/FK/IK passed on GPU 0. See the
[validation record](validation/official_control_20261030/README.md).

Current local installation: [Conda setup and verification](docs/conda_setup.md).
On 2026-10-30, the Conda environment, GPU-0 camera diagnostic, subscription sign-in
and isolated worker permissions passed verification. Supervisor is installed.
Before this control change, the privileged offline suite passed **593 tests**; **103 targeted checks** also
passed after the final launcher changes. See the [current-host record](validation/conda_host_20261030/setup_result.json).
Fresh native Make Toast qualification and a new review packet remain before
a full task test. This environment setup launched no model request or task episode.

The host/run reports below are preserved historical evidence from before migration.

The new host's **Isaac Sim 5.1 rendering check passed on an RTX 4090 with driver 580.82.09**. Eight real 640×480 captures were saved and inspected. NVIDIA's default fast shutdown exited with code 0 and returned GPU memory to its 17 MiB baseline. A monitor race incorrectly marked the raw final result as failed; the independent assessment preserves that raw record and explains the correction. See [new-host evidence](validation/new_host_20261029/).

The GPU has 24 GB VRAM with approximately 23.5 GiB free at idle. A model-free native layout-0 reset produced all three camera views, used a sampled peak of **5,726 MiB (5.59 GiB)**, and left at least **18,356 MiB (17.93 GiB)** free. It exited with code 0 and released its GPU allocation. See the [native diagnostic review](validation/new_host_20261029/native_preview001_review.json) and [head view](validation/new_host_20261029/native_preview_001/preview/cam_head.png).

An isolated compatibility overlay repaired the `hid` and `pkg_resources` imports. A second native reset passed with clean extension startup and the same memory use. A **300-second physics-paused wait** preserved all joint positions, physics time and step counts. Both fresh resets produced identical robot state; rendered images have small recorded pixel differences. See the [second native review](validation/new_host_20261029/native_preview002_review.json).

The published assets contain a missing camera-stand normal texture and invalid robot material references. Original assets are preserved, and successful rendering does not certify full asset fidelity. The supported stock fast-shutdown mode does not prove full plugin teardown.

The native worker, parent execution loop, single-use approval, private supervisor preparation, three-view video and both-arm trajectory reporting are implemented. Robot geometry, Jacobian reference corrections and moving camera projections have passed recorded checks. The complete model-free controller diagnostic passed **960 actions and all 39 stages**, including fresh camera reviews and both empty-gripper closing/opening measurements. It exited cleanly and released its GPU allocation. Measured parameter selection and independent review passed; the runtime bundle and final approval packet passed root and isolated-worker checks. The complete model-driven launch remains unexercised.

The preceding 260-action open-only preview established a clearer empty-closing pose under an explicitly reviewed 0.50-radian diagnostic home-joint excursion bound, with all other guards unchanged. Its sampled memory peak was **5,741 MiB (5.61 GiB)**, leaving **18,341 MiB (17.91 GiB)** free. See the [complete controller diagnostic](validation/new_host_20261029/native_controller_characterization_010/launch_result.json).

**The archived host recorded a single approved Make Toast attempt and consumed its approval. Its copied records do not contain a verified terminal result. That approval cannot be reused on this host.** One Astra-high planning-only request produced the original validated plan, which was preserved without edits and independently reviewed. Automatic restart and retry remain disabled. Review the [archived proposed attempt](validation/new_host_20261029/FULL_TEST_REVIEW.md), including its published-asset defects and limited controller qualification. NVIDIA licence acceptance permits setup only.

- [New-host migration instructions](MIGRATION.md) and [migration inventory](migration_manifest.json)
- [Current reviewed plan](PLAN.md), [model-input contract](INPUT_CONTRACT.md) and [adapter differences](docs/adapter_diff.md)
- [Implementation decisions](docs/implementation_decisions.md) and [pinned sources](configs/source_pins.json)
- [Original independent review](review_20261028/REVIEW.md)
- [Unmodified original planning archive](planning_archive_20261029/)
- [Implementation status](IMPLEMENTATION_STATUS.md) and [machine-readable status](implementation_status.json)
- [Provisioning evidence](validation/provisioning/)

The intended pilot uses native `make_toast`, seed-directory 0 / layout ID 0, dual ARX arms, three 640×480 cameras and Astra high through subscription transport. The official horizon is 1,400 native actions (56 simulated seconds). No extended continuation or live execution smoke is authorized.

The earlier GPU 4 / driver 610.43.02 failures belong to the previous host and remain preserved in the [A1 report](validation/provisioning/A1_REPORT.md). Their unique cause was not established. The archived single-GPU setup used UUID `GPU-a9e034f8-a424-d7fe-6eaa-45667dd2c6b2`; the current host uses GPU 0, UUID `GPU-5e601970-3bfa-34f4-d99c-98b2ff594e91`. The frozen uc7.1 core passed all 851 file hashes; all adaptations and runtime files stay here.

Archived combined offline validation: **577 tests passed** ([record](validation/new_host_20261029/offline_tests_integrated_007.xml)); all earlier records are preserved. Offline tests do not establish task performance, loaded-contact behavior or workspace-wide control accuracy.
