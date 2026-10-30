# Official cuRobo control dependencies

The official EEF backend uses the repository-pinned RoboDojo `CuroboPlanner`
and cuRobo submodule. Dependencies are installed in a task-local overlay at
`runtime/official_control/site-packages`; the shared Conda environment, PyTorch
2.7.0+cu128, host driver and frozen UC7.1 core are not changed.

`scripts/setup_official_control.py` builds a wheel from a `git archive` of the
pinned cuRobo commit and downloads the additional wheels in
`configs/official_control_dependencies.lock` with SHA-256 verification. The local
wheel uses the explicit provenance version `0.0.0+gd17b54ce32cb`; this is a local
build label, not a claimed upstream release number. Its source commit remains
`d17b54ce32cba095c0b000c4c58777075d11de0e`.

The pinned wheel does not include cuRobo's `content/` configuration and mesh
files. The installer copies their exact bytes from the same pinned source,
matching the files available in the upstream recommended editable install.
This is package-data completion; it does not modify the solver or native source.

The official X5 configuration is generated from the already pinned dataset file
`Assets/Robots/x5/curobo_tmp.yml` (SHA-256
`5eea08bf7ef276bf9dfd078ef9f69221e9a67c8cf6cd19fce5ef66d42e96b3f2`). Only
`${ASSETS_PATH}`/`$ASSETS_PATH` is expanded, following upstream
`utils/update_embodiment_config_path.py`. The installer does not substitute the
different configuration bundled under `third_party/curobo/robot/x5_v2`.
Existing authored assets and a conflicting generated configuration are never
overwritten.

To reproduce into a fresh directory with the existing project interpreter:

```bash
python scripts/setup_official_control.py \
  --install --prepare-assets --output runtime/official_control_fresh
```

The installation produces a setup JSON with source/asset hashes, complete overlay
file hashes, wheel hashes and base-package versions. `verify_setup_record(path)`
performs static verification without importing cuRobo or starting CUDA. It reuses
`VerifiedFileHashCache` while continuing to check file identity and unexpected
files. Subsequent checks validate every file identity and directory names; on
this host they take about 11–12 ms after the first verification. Unbound Python
bytecode is rejected, and installation disables bytecode generation.
`activate_setup_record(path, cache_root)` must run before Isaac Sim imports;
it selects the verified overlay and directs Warp, CUDA, cuRobo and XDG caches into the
worker's writable runtime directory. It rejects an already loaded cuRobo, Warp
or CUDA module from a different source. It imports only the lightweight pinned
cuRobo runtime configuration modules to set their `cache_dir`, verifies their
provenance, and starts no CUDA/solver. The overlay itself is not a cache.

The current setup record is `runtime/official_control/setup_v5.json`. On this
host, the real official planner constructed and warmed up successfully, and the
zero-joint FK/IK roundtrip returned `Success` in 35.65 seconds. See
`runtime/official_control/ik_check_003/result.json`, bound to that final setup
record and using the same activation helper as production. Both GPUs returned to 0 MiB
after the process exited. Fourteen static integrity/activation tests also passed. The earlier
package-data failure is retained below.

Setup verification is separate from native control qualification. The bounded
`scripts/check_official_ik.py` test imports and constructs the actual native
planner, then attempts an FK/IK roundtrip at the zero-joint pose on an explicitly
selected GPU. It starts no simulator, performs no robot action, and makes no
model request. This successful check establishes dependency/solver integration
only; it does not certify camera/TCP agreement, inactive-arm holding, loaded
contact or task success.

```bash
python scripts/check_official_ik.py \
  --setup runtime/official_control/setup_v5.json \
  --output runtime/official_control/ik_check_new \
  --gpu-uuid GPU-5e601970-3bfa-34f4-d99c-98b2ff594e91
```

Earlier setup records and failed check outputs are preserved. In particular,
`ik_check_001/result.json` records the missing-wheel-package-data failure before
the exact source data was included. Use the latest verified setup record selected
by the runtime bundle; an older record is not silently rewritten after file
changes. No setup record authorizes a task attempt.
