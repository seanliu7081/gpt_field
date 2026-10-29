# uc7.1 × RoboDojo Make Toast Standard

Implementation for **layout 0 remains blocked at the renderer compatibility check after local repairs**. Removing the CUDA mask and installing the complete official Isaac Sim 5.1 package set did not produce a camera frame. The latest standard-application diagnostic segfaulted during viewport startup. See the [repair conclusion and next step](validation/provisioning/A1_REPORT.md); driver compatibility is the leading hypothesis, not a proven unique cause.

**No Make Toast test or model call has run. No start approval has been consumed.** A real layout preview and plan are still required before requesting test approval. NVIDIA licence acceptance permits setup only.

- [New-host migration instructions](MIGRATION.md) and [migration inventory](migration_manifest.json)
- [Current reviewed plan](PLAN.md), [model-input contract](INPUT_CONTRACT.md) and [adapter differences](docs/adapter_diff.md)
- [Implementation decisions](docs/implementation_decisions.md) and [pinned sources](configs/source_pins.json)
- [Original independent review](review_20261028/REVIEW.md)
- [Unmodified original planning archive](planning_archive_20261029/)
- [Implementation status](IMPLEMENTATION_STATUS.md) and [machine-readable status](implementation_status.json)
- [Provisioning evidence](validation/provisioning/)

The intended pilot uses native `make_toast`, seed-directory 0 / layout ID 0, dual ARX arms, three 640×480 cameras and Astra high through subscription transport. The official horizon is 1,400 native actions (56 simulated seconds). No extended continuation or live execution smoke is authorized.

The isolated GPU 4 camera gate remains unsuccessful; all diagnostic processes and allocations have been released. Task assets have not been installed. The frozen uc7.1 core is verified in `vendor/uc71`; all adaptations and runtime files stay here. The review's requested fixes are implementation gates, not claims of completed validation.

Offline validation: **84 tests passed** ([record](validation/offline_validation.json)). This does not validate native rendering, robot tracking or task execution.
