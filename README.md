# GPT field

Test Astra-generated spatial fields as interfaces for local robot planning. The primary field contains unitless lower-is-better potentials. Metric distance fidelity has a separate command and reference format. Read [the implementation plan](IMPLEMENTATION_PLAN.md) for hypotheses, baselines, and evaluation gates.

Read [the implementation protocol](IMPLEMENTATION_PROTOCOL.md) for the exact direct, field-only, and hybrid control loops, history/feedback contracts, and reproduction commands. Read [the evaluation report](EVALUATION_REPORT.md) for recorded outcomes, videos, grasp/placement analysis, and comparison limits (evidence through 2026-10-11).

Read [the method and architecture document](METHOD_AND_ARCHITECTURE.md) for equations, algorithm pseudocode, model interfaces, class relationships, and runtime data flow.

The new opt-in `--field-improvements` variant adds RGB goal-consistency checks,
held-object placement offsets, and analytic paths that can end before ten segments.
Preview the exact subscription request without credentials or execution:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py \
  --field-improvements --preflight
```

For a fresh full-task run, replace `--preflight` with
`--output runs/grounded_compact_medium_next`. This makes live ChatGPT subscription
requests. The preset uses Astra medium, tracking, block summaries, and the existing
grasp-settling/centered-placement guidance. Direct remains five actions; the grid
remains 33³ at 0.005 m. Only actual LIBERO success completes the evaluation; there
is no episode timeout, stagnation stop, model-completion stop, or API-key fallback.

The model's versioned `goal_reference` can describe the visually estimated offset
between a held bowl and the gripper. Local code compensates that offset for the
requested orientation before computing the gripper goal and expanding the grid.
An optional external-camera image anchor enables local patch tracking. Two reliable
observations moving farther from its projected target, accompanied by measured
robot motion, request a new model decision. Missing, textureless, ambiguous or lost
anchors are reported as unavailable/inconclusive, not as successful tracking.

For pure goal fields, local paths contain at most ten 1 cm segments, with a shortened
last segment and a 5 mm measured arrival tolerance. Goal-triggered model refresh
also requires measured orientation error at most 0.05 rad, so unfinished rotation
can continue locally after translation arrives. Paths remain inside the current
grid/workspace; repulsors or masks retain lattice planning. A path guides at most
ten control actions and can replan sooner on measured endpoint arrival. Field
blocks refresh after at most thirty actions, or earlier on goal arrival, local
boundary arrival, an invalid direction, visual contradiction, or five commanded
actions totaling at least 5 mm but realizing at most 0.5 mm. These events request
fresh RGB-based routing; they never declare task completion or force a grasp.
Early blocks retain full feedback locally and explicit `field_control_event` records;
their closed histories can be summarized for the next model request.

This implementation has offline behavioral tests. No new full-task success or
speed improvement is established by those tests. Previous comparison results below
belong to the earlier fixed-cadence protocols; the comparison script retains that
protocol rather than silently enabling this new variant.

The opt-in optimized Compact Hybrid keeps the 33³ / 0.005 m field, ten 1 cm geometric segments,
ten control actions per local plan, and thirty actions per model refresh. `--execution-mode tracking`
follows the path from each measured pose with a 3 cm look-ahead and a 5 cm command-norm bound;
the actual motion is measured rather than assumed equal to geometric path length. Predicted masks,
grid bounds and workspace still constrain commands. `--feedback-history block_summary` replaces
completed feedback blocks in model input with deterministic measured-motion summaries; complete
per-action records remain in `trace.jsonl`. Legacy `fixed` / `full` defaults remain available.
The planner skips redundant segment traversal only when the entire grid is valid, retaining its
original candidate ranking and counts.

Run a fresh, sequential full-task comparison with the same Astra medium subscription route,
initial state, feedback summaries and first-of-LIBERO-or-model completion rule:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_optimized_compact.py \
  --output runs/optimized_compact_medium_next
```

The Direct arm always emits five actions and uses no local field planning. Both arms retain
videos, request logs, source snapshots and detailed timing. `comparison.json` reports wall-clock
control time including model waits; video playback duration is not a timing measurement.
This command makes live subscription requests and has no API-key or model fallback.

For a fresh pair with identical grasp/placement guidance and actual LIBERO-success
stopping in both arms, use:

```bash
PYTHONPATH=src .venv/bin/python scripts/compare_optimized_compact.py \
  --matched-libero-guidance --output runs/fair_compact_medium_pair_next
```

Both arms restart from the same task, initial state and seed, using Astra medium,
subscription OAuth and block feedback summaries. The same two operational guidance
paragraphs are included verbatim in both prompts; controller-specific routing, field
representation and action-schema instructions remain appropriate to each method.
The report verifies actual request guidance as well as configuration, source hashes,
initial observations and evaluator success. It reports a completed-task speed ratio
only when both arms achieve actual LIBERO success and the matching checks pass.
This remains one sequential pair, subject to model behavior and service-latency variation.

The completed [matched-guidance pair 004](runs/fair_compact_medium_pair_004/comparison.json)
used this protocol without time, action, or stagnation caps. Both arms attained actual
LIBERO success and all 23 matching checks passed:

| Controller | Control wall time | Model pipeline time | Model calls | Actions |
| --- | ---: | ---: | ---: | ---: |
| Tracking Compact Hybrid | 445.662 s | 368.150 s | 37 | 335 |
| Five-action Direct | 443.232 s | 403.347 s | 46 | 227 |

Hybrid took 2.430 s longer (0.55%); this pair does not demonstrate an end-to-end
speed advantage. Its model pipeline used 35.197 s less time (8.73%), but more
actions, simulation and inline video work offset that saving. Inline video rendering
took 54.618 s for Hybrid and 27.811 s for Direct; local Hybrid planning took 2.999 s.
Model pipeline time includes request/history processing and local field expansion
(0.357 s in Hybrid), not just server inference. Both arms stopped on the first
evaluator-success action while still commanding gripper closure, so these results
do not establish released-and-settled placement. Full audits passed 17,441 Hybrid
and 10,396 Direct checks, including complete video decoding.

The [attempt ledger](runs/fair_compact_medium_attempts.json) also retains all three
preceding incomplete pairs: server overload after 125.561 s, missing stream completion
after 5,465.491 s, and an invalid model quaternion after 2,415.380 s. Each ended in
the Hybrid arm before Direct started, with no LIBERO success. Their combined
8,006.433 s is separate retry cost and is not erased by pair 004. The completed-pair
times above are conditional on that pair succeeding; they are not average time to
success across attempts or evidence of a statistically stable speed difference.

The recorded [optimized comparison](runs/optimized_compact_medium_comparison_001/comparison.json)
used identical initial images/state, `gpt-6-astra` medium, subscription OAuth, block summaries,
and the first LIBERO-success or explicit-model-completion signal:

| Controller | Control wall time | Model calls | Stop signal |
| --- | ---: | ---: | --- |
| Tracking Compact Hybrid | 433.436 s | 39 | LIBERO success |
| Five-action Direct, fresh retry | 724.784 s | 75 | Astra completion; LIBERO success false |

In this completed timing comparison Hybrid used 291.348 s less time (40.20%; 1.67× speedup).
The first Direct attempt was interrupted by HTTP 503 after 589.071 s and 350 actions; it is
retained separately in the report and excluded from completed-trial timing. This is one
comparison with different model-generated trajectories and different allowed stop signals,
so it does not establish a stable average speedup or equal physical success.

The completed runs passed 33,487 offline audit checks, including request/model/effort/route,
history reconstruction, commands, stopping, source snapshots and full video decoding.
[Offline validation](runs/optimized_compact_validation_001/verification.json) also preserves
the exact 45-plan comparison (7.83× local speedup), real-simulator tracking calibration,
and deterministic history compression measurements. Tracking's commands respect predicted
grid validity, but actual dynamics can overshoot the fixed field boundary: this Hybrid run
had 20 subsequent zero-translation actions while waiting for a valid plan or field refresh.

For a fresh Direct retry requiring actual LIBERO success, use:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py \
  --reasoning-effort medium --controller-mode direct \
  --feedback-history block_summary --centered-placement-guidance \
  --output runs/direct_medium_libero_success_next
```

This opt-in prompt asks for visual centering within the plate rim and alignment checks
from both cameras before release and after retracting. It adds no hidden object state,
success feedback, local controller or supplied action directions. Five-action chunks and
subscription-only inference remain unchanged. The prompt is versioned separately; model
completion stopping is disabled. Previous model-declared completions remain historical
results and are not reclassified as LIBERO success.

Add `--grasp-settle-guidance` to this command for the separate v5 prompt: it explains
the 20 Hz control clock (five actions are 0.25 s, and model waiting does not advance
physics), asks the model to hold its grasp pose until measured finger motion settles,
and to check a short lift in both views before transport. This remains model-generated
Direct control, with no scripted grasp sequence or hidden object feedback. The v4
prompt remains available without this flag.

The [v5 Direct retry](runs/direct_medium_libero_success_002/results.json) attained actual
LIBERO success in **268.433 s (4 min 28 s)**, with 127 actions and 26 Astra medium
subscription calls. It stopped at the first evaluator success during lowering onto the
plate, while still commanding closure; release and retract were not executed. The run
passed 5,890 offline audit checks. The preceding v4 attempt was interrupted after
856.203 s and 435 actions without success; both attempts remain recorded. This v5
successful episode took 165.003 s less than the previous successful Hybrid episode,
but its added grasp/placement guidance means this is not a strict matched comparison.

## Repository contents and local experiment artifacts

This repository contains the controller source, tests, configuration examples, method documentation, and a [curated evaluation summary](results/summary.json). The full `runs/` directory, videos, raw model requests/responses, private credentials, virtual environments, and caches remain outside Git. Links into `runs/` in the historical documents refer to the originating experiment workspace; they are not downloadable repository assets. Links into `../heading_flow_pub/` likewise identify that workspace's external LIBERO source checkout.

The code is a standalone Python package, but simulation dependencies are installed separately. On another machine, pass your own `--sim-python /path/to/simulator/python` and `--libero-root /path/to/LIBERO` to the dedicated runners. Their historical defaults point to the original `/workspace` instance. Annotated video generation also requires `ffmpeg` with `libx264` and `ffprobe` on PATH. The lock file covers the historical core environment, not the simulator or all subscription extras.

## Setup

Python 3.10 or later is required. The simulator may use a separate Python environment.

```bash
# From the cloned repository root
python -m venv .venv
.venv/bin/pip install -e '.[dev,subscription]'
.venv/bin/python -m pytest -q
.venv/bin/python -m gpt_field doctor
```

This instance already has a project-local `.venv`. `requirements.lock.txt` records the packages used for validation; it does not pin the separate simulator environment. To install that exact core environment, install the lock file, then install this package with `--no-deps`.

## Offline experiments

```bash
.venv/bin/python -m gpt_field demo --output runs/my_demo
.venv/bin/python -m gpt_field report runs/my_demo --output runs/my_demo/report.json
.venv/bin/python -m gpt_field fidelity configs/distance_fidelity.example.json
```

Output directories must be new. The demo writes two synthetic scenes across six methods, with JSON, PNGs, and GIFs. Its fields are generated by a numerical oracle for software checks. They are not Astra results. The fidelity example also uses illustrative numbers, not measurements.

`runs/offline_demo` contains the executed fixtures. On the diagonal task, the full/one-step/affine/waypoint methods reached in 43 actions, the minimum-point method in 47, and shuffled values timed out. On the detour, full/one-step/affine reached in 35 actions; the other methods timed out. These examples show no measured advantage for the full field over its affine approximation or for three-step over one-step planning.

## Simulator smoke without API

Set an official LIBERO source checkout and a Python runtime containing compatible simulation dependencies:

```bash
.venv/bin/python -m gpt_field sim-smoke \
  --libero-root /workspace/heading_flow_pub/third_party/LIBERO \
  --sim-python /workspace/venvs/libero-plus/bin/python \
  --suite libero_spatial --output runs/my_sim_smoke
```

The runtime path above is the compatible environment available on this instance. The benchmark source is explicitly loaded from the separate official-layout LIBERO checkout; no LIBERO-plus task substitution is performed. Elsewhere, replace both paths with your own. Source hashes, dependency versions, and controller details are recorded; the vendored checkout has no standalone upstream Git revision available.

The worker renders 256 by 256, converts both cameras to 224 by 224, loads an official initial state, waits ten actions, executes one neutral action, and closes. `observation.json` is suitable for the next command. The policy and evaluator never exchange the full simulator observation dictionary.

## Inspect a request or call Astra

Build the exact payload without credentials or a network call:

```bash
.venv/bin/python -m gpt_field infer runs/my_sim_smoke/observation.json \
  --dry-run --output runs/request_preview
```

For a live call, set `OPENAI_API_KEY` in your shell and omit `--dry-run`:

```bash
.venv/bin/python -m gpt_field infer runs/my_sim_smoke/observation.json \
  --api-config configs/astra_pilot.json --output runs/first_prediction
.venv/bin/python -m gpt_field replay runs/first_prediction/field.json \
  --output runs/first_prediction/ablation_actions.json
```

The key is never written into logs. No `.env` file is loaded automatically. A missing key produces an explicit error; the model is never silently substituted. The first live attempt was rejected for exhausted credits. The authorized retry produced three valid fields from three calls; see `runs/astra_live_pilot_002/RESULTS.md` and the validation record.

## Development and benchmark runs

```bash
.venv/bin/python -m gpt_field libero \
  --libero-root /workspace/heading_flow_pub/third_party/LIBERO \
  --sim-python /workspace/venvs/libero-plus/bin/python \
  --suite libero_spatial --tasks 0 --episodes 0 --method field \
  --api-config configs/astra_pilot.json --output runs/development_field
```

The pilot config allows only three requests and is expected to stop early in a long episode. Configure an appropriate call/time budget after inspecting real usage. `configs/astra_cost_capped.example.json` illustrates explicit pricing and a dollar cap; verify [current official rates](https://developers.openai.com/api/docs/models/gpt-6-astra) before using it. At the recorded example rates, a full-context reservation is $26.8644 per outstanding request, including long-context premiums; measured usage usually releases most of that reservation.

Available methods are `field`, `field_one`, `affine`, `shuffled`, `minimum`, `waypoint`, `direct`, and `pi05`. Options include grid size/spacing, motion step, search horizon, and update cadence. Direct actions require five-step cadence. Defaults are 9 cubed grid points, 2 cm spacing, 1 cm motion, horizon three, and refresh every five actions.

Full held-out evaluation uses `--suite libero_10 --tasks all --episodes all --phase frozen` after development settings are frozen, with an explicitly chosen budget. A run can need 52,000 calls per method before retries. Running this command is a separate experimental step; the code does not start it automatically.

Use the identical command plus `--resume` after an interruption. Completed terminal outcomes remain immutable; interrupted episodes restart from the same official initial state. Changed source, settings, budgets, or simulator metadata reject resumption. Budget-exhausted episodes remain recorded; raising a cap requires a new run identity. A command ending at a budget cap exits with code 3; setup/runtime errors exit with code 2.

For `pi05`, install the official `openpi_client` into the policy environment and run the checkpoint server externally. Supply `--reference-manifest` with nonempty `checkpoint`, `server_revision`, and `config` entries. This preserves the original checkpoint configuration. Server/checkpoint validation, including its actual state-input behavior, remains required; no pi0.5 server was launched here.

Compare completed methods on the same suite:

```bash
.venv/bin/python -m gpt_field report runs/development_field runs/development_waypoint \
  --output runs/development_comparison.json
```

## Artifacts and limits

Runs contain `manifest.json`, `simulator.json`, `episodes.jsonl`, `summary.json`, `run_state.json`, raw API logs, compressed per-step decisions/candidate scores, and optional annotated MP4s. MP4 encoding requires `ffmpeg` with `libx264`. The manifest records code and dependency hashes. `--no-video` reduces storage. Simulation pauses while awaiting Astra; videos use slowed event playback and show measured API latency without reproducing the wait. The analytic offline demo still writes GIFs.

## Ten episodes of the black-bowl task

The dedicated runner freezes LIBERO-Spatial task 0, official initial states 0–9,
220 control steps per episode, a 9³ field every five steps, and three-step local
planning from the measured end-effector position on every step. It stops each
episode on the simulator's task-success predicate or the horizon. Credentials
are read from `OPENAI_API_KEY` or a hidden terminal prompt; they are not saved.

```bash
.venv/bin/python scripts/evaluate_bowl10.py \
  --output runs/bowl10_live \
  --sim-python /workspace/venvs/libero-plus/bin/python \
  --libero-root /workspace/heading_flow_pub/third_party/LIBERO \
  --workers 10
```

Each episode has an independent simulator and API ledger. The supplied config
allows 88 API attempts (44 refreshes plus at most one schema retry each), 18,000
seconds, and a $50 conservative accounting ceiling **per episode**. The ceiling
includes headroom for a full-context reservation; it is not expected usage.
At the earlier pilot's usage, 440 successful calls would cost approximately $90
across all ten episodes before retries, using the recorded official rates.

The run directory contains `index.html` with all ten videos, `summary.json` with
the fixed denominator and infrastructure outcomes, and one directory per initial
state. Each MP4 shows the two observed RGB images, actual model field slices,
candidate and selected paths, action commands, measured execution, and outcome.
Its `.events.jsonl` sidecar maps displayed frames back to refresh/plan/execution
events. Raw API records and compressed traces preserve the underlying evidence.
Videos omit API waiting time and explicitly label their slowed playback.

Use `--resume` with identical settings after an interruption; terminal records
are preserved. A final task success rate is emitted only after all ten episodes
end in success or a complete 220-step timeout. API, simulator, or recording
failures remain visible and do not become fabricated task failures.

The fixed-center field interpolator rejects any active masked corner. Segment validation checks every crossed interpolation cell. This validates only model-predicted masks. It cannot establish real arm collision clearance or grasp feasibility. Stage, orientation, and gripper failures can dominate full-task results independently of field quality.

The completed closed-loop batch achieved **0/10 successes** on the specified black-bowl task: all ten initial states reached 220 actions. It used 440 real Astra field predictions and 2,200 executed actions, with no infrastructure failures. The model never requested gripper closure. See [results](runs/astra_bowl10_001/RESULTS.md), [all ten videos](runs/astra_bowl10_001/offline_gallery.html), and [the video ZIP](runs/astra_bowl10_001_videos.zip).

Earlier evidence includes the offline tests, analytic fixtures, simulator smoke checks, a synthetic six-step probe, and three Astra fields from one repeated observation. Those static fields matched their own affine baselines and showed no dense-field advantage. The new single-task rollout establishes task performance under the recorded configuration; full LIBERO-10 comparisons and metric distance fidelity remain unmeasured. See [validation details](VALIDATION.md).

## One episode without an action limit

The dedicated single-episode runner removes the action horizon at both the
evaluator and simulator. It uses the same task, policy, seed, and initial state 0
as the first episode above. Field refresh remains every five executed actions.

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_until_success.py \
  --output runs/bowl_one_live
```

The simulator stops on LIBERO success. User interruption, API budget exhaustion,
or infrastructure errors end an incomplete run and preserve its video and logs.
The existing $50 per-episode conservative API budget remains in force; there is
no action, total API-call, or total elapsed-time limit. Per-request timeouts and
output-token limits still apply. Cost reservations may stop the next request
before measured usage reaches $50. `runner_summary.json` distinguishes success
from incomplete execution; an interrupted run does not establish a failure rate.
Recorded terminal outcomes are immutable. Resume can restart an unrecorded
attempt from its initial state; it does not restore a simulator checkpoint.

The live retry reached **565 actions and 113 Astra fields** without success,
then stopped at the API budget reservation. The gripper never closed. This is an
incomplete evaluation, not a finite-horizon failure rate. See the
[full result](runs/astra_bowl_unbounded_002/RESULTS.md) and
[annotated video](runs/astra_bowl_unbounded_002/libero_spatial_t00_i00_field_attempt00.mp4).

After explicit approval of a higher finite total API budget, the separate
continuation runner can reconstruct this same physical episode from its saved
actions, verify every state and camera image, and continue in that simulator.
Historical API calls are not repeated and their accounting is carried forward.
The user approved a **$100 total API budget**, including prior usage, for
`runs/astra_bowl_continuation_001`. It executed ten new actions and two new Astra
fields before HTTP 503 stopped it at action 575. Its unresolved request retains
the full reservation. See the [continuation result](runs/astra_bowl_continuation_001/RESULTS.md).
`runs/astra_bowl_continuation_002` reached 1,115 cumulative actions and 223
cumulative valid fields without success before its $100 reservation guard
stopped it. See its [result](runs/astra_bowl_continuation_002/RESULTS.md).

```bash
# The user approved this $100 total budget for this continuation.
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/continue_bowl_episode.py \
  --source runs/astra_bowl_unbounded_002 \
  --output runs/astra_bowl_continuation_001 \
  --max-total-cost-usd 100
```

This runner only accepts the original episode's terminal budget stop at a field
refresh boundary with an open-gripper history. Generic resume is unsupported.

The separate `scripts/recover_bowl_continuation.py` handles the recorded HTTP 503
handoff. It verifies the complete ordered history, replays all 575 actions before
new inference, carries all 116 API attempts and the unresolved reservation, and
rejects a prior descendant ledger or competing coordinator. It does not reset
the budget. Recovery uses a fresh output beside its source:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/recover_bowl_continuation.py \
  --source runs/astra_bowl_continuation_001 \
  --output runs/astra_bowl_continuation_002
```
After a continuation spends money, further continuation must reconcile that
child's ledger; starting another child from the original ledger would omit its
spending. The read-only `scripts/replay_bowl_state.py` helper can independently
verify a recorded prefix without inference or new evaluated actions.

The user subsequently approved a **$150 total API budget**, including all prior
usage and unresolved reservations. The $100 segment kept its
frozen configuration through its terminal budget stop. The separate
`scripts/continue_bowl_chain_budget.py` then verifies the complete three-segment
history and changes only the total cost cap for a fresh continuation. It requires
terminal source accounting and exact reconstruction before new paid inference.

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/continue_bowl_chain_budget.py \
  --source runs/astra_bowl_continuation_002 \
  --output runs/astra_bowl_continuation_003 \
  --max-total-cost-usd 150 \
  --authorization runs/astra_bowl_budget_150_authorization.json
```

This is the same single episode without an action limit. Reconstruction actions
do not count as new evaluation actions; earlier model calls are not repeated.

`runs/astra_bowl_continuation_003` was interrupted after 1,865 cumulative actions.
Its last persisted `running` status is stale. Request 375 has no recorded response;
the old episode remains incomplete and its unsettled request reservation must be
retained. A new policy evaluation must not be presented as a continuation of this
frozen field episode.

## Direct actions with history on the same bowl task

`scripts/evaluate_bowl_direct_history.py` implements a separate `direct_history`
baseline. It uses official LIBERO-Spatial task 0, initial state 0, seed 7, the same
two RGB cameras and eight robot-state values, ten settling steps, and the same
20 Hz OSC controller as the field episode. Live evaluation has no action limit.

Astra returns five normalized seven-dimensional actions per request. All five
actions are executed before asking for the next chunk. Each actual transition
adds the command, measured robot-state change, finger-joint change, and rolling
motion statistics to the next request. Only these observable values reach the
policy; simulator success, object poses, contact signals and reward remain in
the evaluator. The baseline has no field planner, hard-coded grasp sequence,
forced closure or automatic stall recovery.

Manual Responses history retains accepted model output items and all textual
observations/feedback. Only the newest two model-observation image pairs remain
in later requests. The client uses `store=false`; it preserves encrypted
reasoning items and assistant phase when returned, without interpreting them.
Invalid responses are not added to history. Episode reset clears history but
never resets the client's API ledger. There is no silent text compaction: a
context-limit error ends the run as incomplete. This follows the
[official conversation-state interface](https://developers.openai.com/api/docs/guides/conversation-state).

Preview the request contract without a simulator, credentials or API call:

```bash
.venv/bin/python scripts/evaluate_bowl_direct_history.py --preflight
```

Run a ten-action integration diagnostic in real LIBERO with two scripted mock
API responses. This checks command execution, feedback, history and video. It
does not measure Astra or task success:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_bowl_direct_history.py \
  --offline-smoke --output runs/direct_history_smoke
```

Live execution requires a fresh output directory, a configured `OPENAI_API_KEY`
(or the hidden terminal prompt), and an explicit budget for this new run:

```bash
# Set BASELINE_BUDGET_USD to the separately authorized cap before running.
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_direct_history.py \
  --max-cost-usd "$BASELINE_BUDGET_USD" --output runs/astra_direct_history_001
```

The previous field episode's $150 authorization is not automatically transferred
to a fresh experiment. At the configured rates, the inherited conservative
reservation requires $26.8644 of unreserved budget before each request. Reported
usage settles that reservation; missing usage retains it. The runner writes
source/configuration provenance, complete request/response logs, a flushed
`trace.jsonl`, a summary and `episode.mp4`. Videos show observation, action-chunk
generation, issued actions and measured feedback. Mock videos are labeled on
every frame. Budget and infrastructure stops remain incomplete, unscored trials.
Existing outputs cannot be overwritten or resumed by this runner.

The earlier `--method direct` remains the stateless baseline. `direct_history`
also changes memory and execution feedback relative to the old field run, so
their comparison alone cannot isolate the causal effect of field representation.
It can choose any valid normalized seven-dimensional action, while the field
planner uses 1 cm axial translation primitives and a 0.15 rad rotation cap.
Direct actions run open-loop within each five-action chunk; the field planner
replans translation from measured position at every step. A controlled
representation comparison must match history and account for these action and
feedback differences.

The first live history baseline used a separately authorized **$150** cap and
ran one episode without an action horizon. It executed **450 actions from 90
real Astra responses**, including 82 close commands, but never reached LIBERO
success. The conservative API cost estimate was **$123.5201**; another full
request reservation would exceed $150, so the runner stopped before sending it.
All usage was reported, with no unresolved calls. This is a budget-limited,
incomplete episode (`success_rate=null`), not a measured 0% benchmark result.
The last 260 actions request zero arm motion with the gripper open. See the
[terminal summary](runs/astra_direct_history_001/summary.json) and
[full annotated video](runs/astra_direct_history_001/episode.mp4).
The user subsequently raised this same episode's cumulative cap to **$300**.
The first segment remains immutable; continuation must recover its exact state,
model history, and $123.5201 ledger. A matching diagnostic replay showed that
the target bowl was lifted but placed 3.95018 cm from the plate centre, outside
the strict 3 cm goal threshold; see the
[segment result and diagnosis](runs/astra_direct_history_001/RESULTS.md).

`scripts/continue_bowl_direct_history.py` supports this fresh, budget-stopped
history run at a complete five-action boundary. It reconstructs every recorded
model request and feedback record offline, then checks every historical robot
state and camera image in the simulator that will continue. Only after exact
agreement does it restore the full conversation and cumulative ledger. It
changes only the authorized total cost cap. It refuses missing usage, partial
actions, retries, incompatible parents, or another already-spent child.

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/continue_bowl_direct_history.py \
  --parent runs/astra_direct_history_001 \
  --output runs/astra_direct_history_continuation_001 \
  --max-cost-usd 300 \
  --authorization runs/astra_direct_history_300_authorization.json
```

The command records the existing approval and uses a new sibling output; it is
not reusable once that child has attempted a new request. Add `--preflight` to
check history and accounting without credentials, simulation, API calls, or
writes. The continuation video contains new actions only and uses cumulative
step/chunk numbering; reconstruction is not another scored evaluation.

The authorized $300 continuation has now stopped at the budget guard. Across
the same episode it executed **625 actions and 125 real Astra calls**, with no
LIBERO success. Cumulative conservative cost is **$280.3785** and unknown usage
is zero. All 175 added actions requested zero arm motion with the gripper open;
actions 191–625 form a 435-action suffix without active position correction.
The complete attempt remains incomplete and unscored. See the
[combined result](runs/astra_direct_history_continuation_001/RESULTS.md) and
[full episode video](runs/astra_direct_history_continuation_001_full_episode/full_episode.mp4).

## Field with history and execution feedback

The `field_history` variant adds the same local conversation history and measured
execution feedback as `direct_history` to the original field planner. It retains
all observation text, accepted model output items, and per-action feedback;
only the latest two model-observation camera pairs remain in each request.
Feedback reports commanded versus actual robot motion and finger-joint changes.
Reward, benchmark success, object poses, and contact predicates are not model
inputs. This adds no automatic grasp, phase switch, or recovery action.

The task is the same LIBERO-Spatial black-bowl task, initial state 0, seed 7.
Astra generates a 9³ unitless potential field every five actions. The original
three-step planner replans from the actual position on every action, using 1 cm
axis primitives and executing only the first move. Orientation and gripper
control are unchanged. Historical fields retain their original grid origins.

Read-only preflight and a scripted, zero-API real-simulator smoke:

```bash
.venv/bin/python scripts/evaluate_bowl_field_history.py --preflight --max-cost-usd 300
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_bowl_field_history.py \
  --offline-smoke --output runs/field_history_smoke
```

The smoke uses two scripted fields and ten diagnostic actions. It verifies
field/plan/action/feedback plumbing and is not an Astra capability result.

The user authorized one new live episode with a **$300 API ceiling and no
action-step limit**. The authorization is recorded in
[`runs/astra_field_history_300_authorization.json`](runs/astra_field_history_300_authorization.json).
Run with the configured credential or the hidden terminal prompt:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_field_history.py \
  --max-cost-usd 300 --output runs/astra_field_history_001
```

The ledger starts at zero and includes schema retries and unknown-usage
reservations. There is no action, API-call, or episode-time horizon. Each request
must fit the remaining budget with its conservative reservation; this can stop
the episode before the settled cost reaches $300. Stops for budget, interruption,
or infrastructure errors remain incomplete and unscored. Existing runs are
never overwritten. The annotated video shows both cameras, actual generated
field slices, candidate and selected local plans, actions, and measured feedback.

The first live field-history episode has completed its authorized attempt:
**350 actions, 70 real field generations, 40 close commands, and no LIBERO
success**. It stopped at a cost upper estimate of **$276.37565**, because the
next $26.8644 request reservation would exceed $300. There was no action horizon.
This remains an incomplete, unscored episode (`success_rate=null`).

Unlike the older stateless field run, the model issued closing, trial-lift,
reopening and alignment-correction decisions. Exact offline reconstruction
confirmed three closing attempts but no stable pickup: contact occurred briefly
during the second attempt, and the bowl's maximum body-origin height increase
was 7.662 mm before returning to the table. Neither trial lift maintained contact.
The independent audit passed 29,284 checks, with all 350 plans and actions
matching the original planner on recorded inputs. See the
[result and diagnosis](runs/astra_field_history_001/RESULTS.md),
[full video](runs/astra_field_history_001/episode.mp4), and
[video viewer](runs/astra_field_history_001/index.html).

## Try ChatGPT subscription access

The subscription entry point is separate from the historical API-key runners.
It uses the official [Sign in with ChatGPT flow](https://developers.openai.com/siwc/token-sharing-open-source/sign-in)
and the public Responses endpoint. It never reads an API key or falls back to
API-key billing, and never substitutes a different model for `gpt-6-astra`.
Availability must be verified for the signed-in account; a model catalog entry
alone does not establish successful access.

Install the optional login dependency, then start the temporary local login
helper from this project directory:

```bash
.venv/bin/python -m pip install -e '.[subscription]'
.venv/bin/python scripts/chatgpt_plan_login.py login
```

On a remote IDE connection, privately forward port **1455** from this instance
to local port **1455**, then open `http://127.0.0.1:1455/login` on your computer.
The helper binds only to loopback; no public service or tunnel is needed. Complete
the OpenAI login and consent screen yourself. Review the app's plan/credit
permissions there; subscription use remains subject to your account's limits.
Do not paste credentials or callback URLs into chat. Protected credentials live
under `.private/chatgpt_plan/`, outside the experiment artifacts and ignored by
source control. Existing Codex credentials are not extracted or repurposed.

After login, run the initial access probe:

```bash
.venv/bin/python scripts/chatgpt_plan_login.py status
.venv/bin/python scripts/chatgpt_plan_field_probe.py \
  --output runs/chatgpt_plan_field_probe_next
```

Choose a fresh output directory for every probe; existing results are preserved.

The probe uses the first recorded observation of `astra_field_history_001`,
checks the account model catalog, requests **one real Astra field**, and runs
the original local planner. It executes **zero robot actions** and reports no
task success rate. It cannot resume or alter the prior episode. This one-request
access check does not introduce an action limit into the field experiment.
Full subscription-driven episode evaluation is a separate integration step.

Subscription requests preserve the field schema, instructions and history
representation, but use streaming and omit unsupported parameters, including
`max_output_tokens`, as required by the
[subscription API contract](https://developers.openai.com/siwc/token-sharing-open-source/preview-limitations).
Responses are accepted only at `response.completed`; quota failures, errors,
incomplete streams and invalid fields stop the probe. Logs report token counts
and subscription billing provenance rather than converting them into an API
dollar bill. Check [ChatGPT usage](https://chatgpt.com/settings/usage) for your
actual subscription or credit consumption.

The observed subscription response path can label its SSE body as `text/plain`.
The transport accepts this label only through the same strict SSE completion
parser, records the discrepancy separately, and continues to reject plain JSON,
HTML and incomplete streams. This compatibility handling does not alter the
model, prompt, field schema or planner.

Some completed streams leave the terminal response's `output` array empty.
The reader then assembles the exact complete `response.output_item.done` items
in index order, only after `response.completed`. Missing, unfinished, duplicate
or conflicting items are rejected. Assembly provenance is recorded separately
from the model's output and history.

Live verification on 2026-10-10 succeeded with the signed-in account:
[`chatgpt_plan_field_probe_004`](runs/chatgpt_plan_field_probe_004/summary.json)
generated one real `gpt-6-astra` field and one local plan using subscription
credentials. It made zero API-key calls and executed zero robot actions.
This proves access and the static field/planning path; a complete subscription
episode has not yet been evaluated. Historical failed probes remain available,
and [aggregate usage](runs/chatgpt_plan_setup_001/live_result.json) explicitly
retains unknown usage from the first two attempts.

## Subscription episode until the first close action

`scripts/evaluate_bowl_subscription_close.py` runs a fresh initial-state-zero
episode of the same bowl task with the subscription client and original
field/history policy. It stops after the simulator acknowledges the first
positive gripper action and the resulting observation and execution feedback
are recorded. No subsequent robot action is sent. The measured finger motion
is retained; this stop establishes a closing command, not full finger closure
or a successful grasp. This diagnostic episode is unscored.

```bash
.venv/bin/python scripts/evaluate_bowl_subscription_close.py --preflight
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_subscription_close.py \
  --output runs/chatgpt_subscription_close_next
```

Use a fresh output directory. `--offline-smoke` exercises the same execution and
stopping logic with scripted fields and real LIBERO, without subscription calls.
Live execution uses `gpt-6-astra` through the existing ChatGPT login and never
switches credentials or models. There is no action, request-count or episode-time
horizon. Subscription/authentication failures, simulator errors, user interruption
or earlier LIBERO termination also stop the runner. Subscription consumption is
reported in tokens, without applying API-key dollar prices.

The run saves `run_state.json`, `summary.json`, request/response records,
`trace.jsonl`, and an annotated `episode.mp4` showing field generation, local
planning, execution and measured feedback.

The first live subscription execution test completed its requested stop:
**241 actions, 49 real Astra fields, and exactly one closing action at the end**.
The simulator acknowledged the first close command at zero-based step 240;
no later action was sent. Finger-joint aperture proxy decreased by **0.877 mm**
in that 50 ms control cycle, which does not establish full closure or a grasp.
The run used zero API-key calls and remains unscored. See the
[video viewer](runs/chatgpt_subscription_close_001/index.html),
[83.65-second MP4](runs/chatgpt_subscription_close_001/episode.mp4), and
[execution summary](runs/chatgpt_subscription_close_001/summary.json).

## Subscription hybrid: coarse fields and direct contact control

`scripts/evaluate_bowl_hybrid_subscription.py` starts a fresh episode of the
same official LIBERO-Spatial bowl task (task 0, initial state 0, seed 7) and
attempts the complete bowl-on-plate task. It continues through the first close
command. There is no action, model-call, or episode-time horizon. Actual LIBERO
success completes the episode; user interruption, subscription errors, or
infrastructure failures leave it incomplete and unscored.

At each mode-specific block boundary, one `gpt-6-astra` subscription response selects
either `coarse_field` or `contact_direct` and supplies that controller's payload.
The current compact hybrid uses a **33×33×33 grid at 0.005 m (0.5 cm) spacing**,
retaining the original **16 cm cube** centered at the measured end-effector pose.
GPT returns a relative goal, three attraction weights, up to four Gaussian
repulsors, and up to four blocked and four unknown boxes. Local NumPy code
expands these parameters into **35,937 potentials** and vertex masks. The raw
compact response stays in model history; expanded values are local artifacts.
This analytic representation restricts the field shapes GPT can express and
does not turn 0.5 cm sampling into 0.5 cm visual localization accuracy.
The local planner uses **1 cm primitives
and a ten-step search horizon** (10 cm total commanded path), executes all ten
selected translation primitives in order, then replans from the actual measured
position. The same field supplies **three local plans per 30-action block** at
block-relative steps 0, 10, and 20. Each action still records measured feedback
and adapts orientation from the latest state. Remaining translation commands are
fixed during each ten-action window; measured tracking error does not rewrite
them into waypoint corrections. This reduces planning work but permits more
tracking error to accumulate before the next local replan.
Field control refreshes the model after **30 actions**; direct control
returns exactly **five normalized seven-value OSC actions**, executed before the
next model request. At 20 Hz these blocks span **1.5 seconds** and **0.25 seconds**
of simulated time, respectively.
Hybrid requests explicitly set **`reasoning.effort="high"`**, using the existing
ChatGPT subscription OAuth route. The manifest records the requested effort;
an explicitly different effort in response metadata stops execution without
retry or API-key fallback. Missing response metadata is not proof of the resolved effort.
Near-target status is Astra's visual judgment about the active bowl or plate
interaction region, not a simulator contact signal or a calibrated distance.
The choice can change after 30 field actions or five direct actions, including
a return to field planning during free-space transport. For example, field →
direct → field requests occur at zero-based steps 0, 30, and 35.

The field remains fixed for all 30 actions. Longer action blocks do not expand
its coverage: near the cube boundary, feasible candidates decrease; an invalid
start or no valid candidates gives zero translation until the next scheduled
refresh. There is no automatic early refresh. Unlike the legacy dense v5
protocol, the compact protocol does not ask GPT to emit all grid values.
The v1 results below do not evaluate the compact method.

The ten-step planner uses exact dynamic programming over lattice positions,
first moves, and last moves, merging equivalent partial paths. It preserves
terminal-potential ranking, immediate-potential tie breaking, and fixed primitive
order without enumerating all **6¹⁰ = 60,466,176** sequences. Logs and videos show
at most six terminal-band finalists, while `plan.search` separately records the
represented valid sequence count and search state counts. The ±8 cm field limits
every segment: a 10 cm path may turn, but cannot travel straight 10 cm from the
center. A complete valid ten-step path is required; there is no shorter-horizon
fallback. Hold or no valid full path commits zero translation for that execution
window, while orientation and gripper still operate. Episode success or an
interruption can stop partway through a window. Standalone baselines retain their
three-step exhaustive planner with one action executed before replanning.

This is a new prompt and routing condition. Both modes share all prior textual
observations, complete accepted model outputs, opaque reasoning items, and
measured execution feedback; only the latest two observation image pairs are
retained. The policy receives no object poses, reward, or benchmark-success
feedback. Returning to field mode generates a fresh field centered at the
current measured pose. Field `hold` preserves the last executed nonzero gripper
intent across both modes; a direct zero command is executed unchanged.

```bash
.venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py --preflight
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_hybrid_subscription.py \
  --output runs/chatgpt_hybrid_next
```

The existing ChatGPT login supplies credentials; there is no API-key or model
fallback. Choose a new output directory. `--offline-smoke` exercises 65 real
simulator actions with three scripted responses (30 field → five direct → 30 field),
including closure and retention of close intent, without model calls or a
performance score. Live runs save shared request/response records, routing
decisions, controller switches, actions, feedback, summary, and one annotated
video. The video shows heatmaps and local plans for field control, and action
chunks for direct control. Routing labels are explicitly model predictions.

For the authorized timing-only comparison, run:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_hybrid_subscription.py \
  --timing-comparison --blocks 3 --actions-per-block 30 \
  --output runs/compact_field_timing_next
```

Each pair resets both controllers to the same official initial state and clears
their histories. Order alternates between pairs. The compact-field arm makes one
request and three local plans per 30 actions; the Direct arm makes six requests
for five actions each. Controller selection is fixed by the experiment in these
arms, while the normal hybrid runner still lets GPT select it. Both arms use
`gpt-6-astra`, explicit high effort, the existing subscription login, and actual
simulator execution with feedback. Startup/reset time is separated from model
transport, field expansion, local planning, and control-loop time. This compares
time for equal action counts, not time to task completion. Incomplete arms are
reported as such, and no model or credential fallback is used.

The completed [compact timing run](runs/compact_field_timing_001/comparison.md)
contains three paired 30-action trials per controller (90 actions each), all with
matching reset images/state and `gpt-6-astra` / `high` response metadata. Mean
control time per 30 actions was **28.49 s for compact Field versus 66.75 s for
Direct**: **57.31% less time**, or a **2.34×** ratio. Mean model transport time
was **21.13 s versus 65.00 s**; Field additionally spent **0.061 s expanding the
grid** and **5.07 s on three local plans** per trial. Requests totaled 3 versus
18. Startup/reset are excluded from those controller timings. This is a timing
result for equal action counts, not a task-completion comparison. Exact totals
and definitions are in [summary.json](runs/compact_field_timing_001/summary.json).

The subsequent [compact full-task attempt](runs/compact_hybrid_full_001/analysis.json)
was stopped by the user after **657 acknowledged actions, 47 subscription
requests, and 854.08 s**, with no LIBERO success observed. It remains incomplete
and unscored. The correct bowl was lifted, transported, and released in contact
with the plate, but its final horizontal body-origin distance from the plate
was **4.669 cm**, outside the goal's strict **<3 cm** threshold. The model
declared completion and held for the final 177 acknowledged actions. An isolated
diagnostic replay matched every recorded state and both camera images exactly;
the [artifact audit](runs/compact_hybrid_full_001_audit/audit.json) passed 30,874
checks. The final additional action proposal lacked simulator acknowledgement
and was excluded from the replay. See the
[episode video](runs/compact_hybrid_full_001/episode.mp4). This single interrupted
attempt does not measure time to successful completion against Direct control.

For an explicit model-completion timing test with the user's requested medium
effort, run a fresh episode:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_hybrid_subscription.py \
  --reasoning-effort medium --stop-on-model-complete \
  --output runs/compact_hybrid_medium_next
```

This opt-in completion v2 protocol adds `task_complete` and `completion_reason`
to Astra's compact response. A validated completion declaration stops before
local planning or execution of any action from that response. Temporary holds
do not stop the episode. Either LIBERO success or Astra's completion declaration
stops this mode, whichever occurs first; the two signals remain separate. The summary records
`model_declared_complete`, the terminal response, and `control_elapsed_s` from
the initial observation through that response, excluding startup and cleanup.
The default high-effort v1 protocol and the equal-action timing benchmark remain
available; the new completion instruction is an additional experimental change.

The [medium completion run](runs/compact_hybrid_medium_001/timing_comparison.json)
stopped on Astra's explicit completion declaration after **550 executed actions
and 36 subscription requests**, with no action issued from the terminal response.
LIBERO success was not observed. Control time to the declaration was **658.98 s**;
the comparable video-event interval was **659.23 s**, versus **736.91 s** for the
prior high-effort run's first completion judgment (**77.68 s / 10.54% shorter**).
The medium rollout included a placement correction. This is a single-record
comparison with different trajectories and an added completion instruction,
not an isolated estimate of effort's effect. See the
[video](runs/compact_hybrid_medium_001/episode.mp4) and
[summary](runs/compact_hybrid_medium_001/summary.json).

The historical **v1 (9×9×9, 2 cm, five-action blocks)** live hybrid episode reached **LIBERO success: 1/1**, after **448
executed actions** and **90 real subscription requests** (51 fields and 39
direct-action chunks). It ran field approach → direct grasp/retries → field
transport → direct placement, switching at action ordinals 226, 386 and 416.
There were 255 field-planned actions and 193 direct actions. All 448 transitions
have measured feedback; API-key calls were zero. Runtime was 39.68 minutes.

Watch the [177.95-second video](runs/chatgpt_hybrid_001/episode.mp4) or the
[viewer with stage jump buttons](runs/chatgpt_hybrid_001/index.html). The
[summary](runs/chatgpt_hybrid_001/summary.json) and
[independent audit](runs/chatgpt_hybrid_001_audit/audit.json) preserve the evidence.
The audit passed 18,605 checks, including full video decoding, exact history,
direct commands and recomputed original field plans. This is one episode, not
an estimate of general reliability. The final gripper command remained closed
(+1) when LIBERO reported success; release and post-release stability were not
tested.

## Distance-field control v7 (opt-in)

The `--distance-field-control` preset selects Astra medium, subscription OAuth,
tracking, block feedback summaries and the shared grasp/placement guidance. It
keeps Direct chunks at five actions and the local grid at 33³ / 5 mm. Historical
protocols and runs remain available. Preview the exact request without credentials,
inference or a simulator:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py \
  --distance-field-control --preflight
```

A fresh full-task run uses the same flag with `--output runs/distance_field_next`.
Only actual LIBERO success completes the evaluation. The preset has no episode
time/action cap and no model-completion stop or API-key fallback.

Select `--task drawer_bowl` to run LIBERO-Goal task 3, initial state 0, seed 7:
"open the top drawer and put the bowl inside". The default remains `bowl_plate`.
Preview with `--preflight`, then use a fresh output directory for live execution:

```bash
env -u OPENAI_API_KEY PYTHONPATH=src .venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py \
  --distance-field-control --task drawer_bowl --output runs/drawer_distance_next
```

This task adds `drawer_handle` and `drawer_interior` targets and explicit handle
grasp/pull/release phases. Handle contact and constrained insertion use five-action
Direct chunks; free-space movements use the same distance planner. The model
infers the pull direction from RGB and measured response. Automatic upward
reobservation is disabled around the cabinet: recovery movements require a new
model decision. Task metadata and prompts are recorded and checked by the same
offline auditor. The official containment predicate determines success; no
extra drawer-closing or released-stability requirement is added to it.

The model supplies compact scene guidance and explicit `control_context.phase`
(`approach`, `grasp`, `transport`, `align`, `release`, `verify`, `recover`). Local
Dijkstra solves a 26-neighbor cost-to-go graph with Euclidean edge lengths and
optional Gaussian traversal density >= 1. Unknown/blocked vertices, diagonal
corner cutting and continuous compact boxes are excluded. Costs are meters in
the graph when density is one, or weighted meters otherwise; the old attraction
values do not rank paths. The guarantee is for this finite predicted graph, not
true scene geometry, a continuous geodesic or whole-arm collision avoidance.
Each extracted path has at most ten segments, each <= 1 cm. A cached solve can
serve multiple measured-position replans. The planner reserves 2.5 mm for its
goal region and 2.5 mm for tracking within the 5 mm measured arrival threshold.

Paired external-camera patch tracking follows an explicitly supplied moving
point and its desired projected location. Planar XY corrections (<= 1 cm) require
enough actual motion in both axes to identify a well-conditioned local image
Jacobian. Height, orientation, finger motion, match ambiguity/loss and prediction
residual checks invalidate unsupported calibration. The desired image location
must represent the same rigid point at the intended height/orientation; matching
a bowl rim to a plate center is not generally sufficient. Missing anchors,
texture or two-axis excitation produce an explicit fallback. This is local RGB
guidance, not a learned object detector, depth estimator or 3D placement proof.
Proposed corrected goals must also remain inside the calibrated XY region and
height tolerance before adoption. Already-overlapping paired anchors cannot
initialize independent patch tracking; they request unverified/reobservation
fallback rather than certify alignment.

Fields may be reused beyond 30 actions, in increments of 30 up to 120, only with
reliable visual tracking, measured movement and non-worsening image error.
Arrival, lost tracking, invalid paths or lack of progress request an earlier
model decision. Reuse and goal corrections preserve the original world-frame
grid and obstacle coordinates; the implementation does not enlarge known space
or recenter a grid into unobserved geometry. Outside-grid goals request a nearer
goal/new observation. A 120-action field limit is an inference refresh bound,
not an episode stopping rule.

Verification holds are checked after five observations. Consecutive stable
image agreement is reported only as observational evidence. If evidence is
missing/misaligned, an open-gripper Field hold may perform one logged upward
reobservation move of at most 2 cm within the same predicted-valid region, for
at most ten control ticks, then request a recovery decision. A new release or
model-directed recovery re-arms this allowance; repeated holds alone do not.
The observation move uses a separately logged vertical path with hard geometry
checks; soft repulsors cannot turn it into a lateral/downward detour. It is
re-anchored at each measured pose without correcting lateral drift.
Direct actions are never altered or truncated. Local code never certifies task
success, and a stubborn model can still repeat recovery requests.

Traces separate raw model fields, locally revised goals, local visual feedback,
distance-control boundary events and computed distance maps. The new video
renderer displays the local cost-to-go map rather than labeling the old compact
attraction potential as a computed distance. The dedicated offline auditor is
`scripts/audit_distance_control.py`; reconstruction uses the controller's shared
algorithms and does not reconstruct physical object state. Offline tests and
solver microbenchmarks do not establish full-task success or speedup.
Video local metadata describes the pre-action decision that generated the
displayed command; the trace separately records post-action visual feedback.
