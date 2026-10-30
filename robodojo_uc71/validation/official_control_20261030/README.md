# Official controller implementation validation

Final offline run: **690 passed, 33 failed, 0 errors** in 219 seconds.
All **118 tests in the official-backend test modules passed**.
Every remaining failure is an `Operation not permitted` error from the existing
root-owned runtime/supervisor isolation fixtures. This account cannot use
passwordless sudo. Their permission checks were not bypassed or mocked away.

The real pinned native `CuroboPlanner` initialized and completed FK→IK on GPU 0
using production dependency activation. The final record is
[`ik_check_003/result.json`](../../runtime/official_control/ik_check_003/result.json),
bound to `runtime/official_control/setup_v5.json`. Both GPUs returned to 0 MiB.
All 851 frozen UC7.1 core files match their release hashes.

[`validation_summary.json`](validation_summary.json) binds the test reports,
source files and native IK result. The initial report is preserved separately;
it collected an intermediate fixture name and omitted Conda's ffmpeg from PATH.
Both issues are resolved in the final run.

No simulator or model episode ran. Native simulated motion/gripper qualification
and a fresh full-task review remain outstanding; CUDA-only IK success does not
certify those behaviors.
