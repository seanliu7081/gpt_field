# Astra field planning implementation plan

The first experiment tests whether Astra's spatial predictions help a fixed planner choose useful movements. Metric distance accuracy is a separate auxiliary experiment. The executable project in this folder implements that distinction; full LIBERO results require live model calls and are not implied by the offline fixtures.

## Research claims

The primary hypothesis is that a predicted spatial field supplies useful route information beyond one waypoint or a local direction. We will compare task success, movement choices, stalls, path length, latency, and inference cost. A positive result requires a benefit against a relevant baseline, not merely sensitivity to shuffled field values.

The secondary hypothesis is that a model can predict calibrated remaining distances in meters. It needs an explicitly defined target, motion domain, and reference distance. The primary model interface produces unitless potentials, so those outputs are not eligible for metric error claims without a separately defined experiment.

The original [Riemannian distance-field paper](https://arxiv.org/html/2412.05197v3) connects a specified metric, its Eikonal equation, and a distance function. This prototype does not solve that equation. Eikonal correction, model-based trajectory selection, and learned-policy takeover remain later experimental extensions.

## Policy boundary and field representation

The policy receives only two RGB images, the task instruction, and eight robot-state values: end-effector xyz, orientation axis-angle xyz, and two finger positions. A separate simulator process filters and serializes observations. Object states, rewards, contacts, BDDL predicates, depth, and future simulator rollouts never enter the policy. Evaluator success and provenance remain outside the policy call.

Images are rendered at 256 by 256, rotated 180 degrees, and resized with padding to 224 by 224 using the openpi bilinear and integer-rounding conventions. Coordinates use the simulator world frame. The action adapter checks the installed OSC configuration and records its sources.

The default field has 9 points per axis, 0.02 meter spacing, and a fixed center at the measured end-effector position when generated. Index order is `i*N*N+j*N+k`. Its 729 values are finite, unitless potentials; lower is better. Negative values are permitted. Separate blocked and unknown masks invalidate interpolation. Every active interpolation corner and every crossed cell along a candidate segment must be valid.

Auxiliary predictions are the current stage, a unit xyzw target quaternion, gripper open/close/hold, and translation move/hold. Hold is an explicit stage decision. These predictions do not establish real geometric feasibility or task success.

The scene and inferred stage are treated as fixed while evaluating hypothetical positions in one field. This approximation omits changing contact and object dynamics. Re-observation and replanning limit its duration but do not remove that limitation.

## Corrected planner

For the main usefulness experiment, compare trajectories with exactly three nonzero 0.01 meter axis displacement targets. They have equal nominal translation length. Score their terminal potential, then immediate potential, then a fixed direction order. Reject immediate reversals, workspace violations, and invalid field segments. Execute only the first move and replan from the measured position. Refresh the field every five simulator steps; its center stays fixed between updates.

Candidate positions assume ideal geometric integration of those targets. OSC tracking is not instantaneous: the initial six-step controller probe moved about 1.075 cm in x under six nominal 1 cm x targets. Development must log commanded versus realized motion and assess whether the idealized rollout horizon is informative. Measured path length, rather than the sum of commands, is used in reports.

This avoids the earlier objective `path_length + terminal_distance`: a true continuous distance makes no-op optimal, and axis primitives generally cannot follow an exact diagonal geodesic prefix. Removing just the all-zero candidate would still allow repeated waiting prefixes.

The new planner evaluates 216 sequences before filtering. Flat or non-improving fields produce an explicit `plateau_move` diagnostic. Explicit model hold and absence of valid candidates produce zero translation. The episode step limit bounds persistence. Predicted masks are not a collision certificate for the arm or a held object.

The planner consumes potential values through interpolation. Positive rescaling is numerically tested; arbitrary nonlinear remapping of grid samples can change interpolated rankings and is not claimed to be invariant.

## Comparisons

| Method | Change | Question |
|---|---|---|
| Field with three steps | Main method | Can the predicted field support short-horizon motion? |
| Field with one step | Shorter search | Does lookahead help? |
| Affine field | Fit valid values to a plane, retain masks and auxiliaries | Is spatial structure beyond a direction useful? |
| Shuffled field | Permute only valid values with a recorded seed | Does the method depend on spatial arrangement? |
| Minimum-point field | Distance to lowest valid grid point | Is the full field better than its simplest local target? |
| Predicted waypoint | Separate Astra waypoint output with the same auxiliary schema and planner | Does the dense representation earn its output cost? |
| Direct action | Astra returns five normalized actions per call | Does the field interface improve over action prediction? |
| Direct action with history | Five direct actions plus prior observations, model outputs and measured execution feedback | Can a model with memory manage grasping and recover progress on the same task? |
| Field with history | Original field planner plus the same observation/output history and measured execution feedback as direct-history | Does memory and execution feedback improve field-guided behavior without changing the planner? |
| External pi0.5 reference | Official openpi websocket policy, documented checkpoint | Does the simulator support a trained VLA reference? |

Cached-field comparisons reuse the same original auxiliary outputs and masks. All-invalid predictions follow the same no-valid fallback across field transformations. Independent closed-loop methods receive different observations after trajectories diverge; their auxiliary commands cannot be described as identical. The waypoint call is a separate prediction. Direct actions use the same five-step call cadence, but token costs and controller authority differ and must be reported.

The pi0.5 checkpoint must be identified as LIBERO-trained. Its server revision/configuration and state-input behavior require separate validation; this project does not alter a checkpoint to improve comparability.

## Execution stages and acceptance criteria

| Stage | Implementation and evidence | Advance when |
|---|---|---|
| A Interface and controller | Allowlisted serialization, simulator subprocess, action scaling, source manifests | Offline tests and real reset/action smoke pass |
| B Planner diagnostics | Diagonal reaching and obstacle detour using explicitly labeled analytic oracle fields | No artificial wait bug; masks and motion traces match the fixture |
| C Model field inspection | `infer` on saved permitted observations, strict schema and raw API logs | Outputs parse reliably; failures and usage are measurable |
| D Development closed loop | Use spatial/object/goal suites; compare fields, waypoint, and direct actions | Failure analysis separates spatial ranking, stage, orientation, gripper, and execution |
| E Freeze | Save exact code hashes, prompts, API settings, dependency versions, field/planner settings, and task/init list | All settings and run budgets are fixed before held-out evaluation |
| F LIBERO-10 | Ten tasks, fifty official initial states each, seed 7, ten settling steps, 520 control actions | All attempted episodes and failures are accounted for; incomplete runs are labeled |
| G Distance fidelity | Separate known-reference inputs with `semantics=distance_m` | Metric assumptions and reference construction are documented independently |

The supplied three-call API configuration is a pipeline pilot, not a complete episode or benchmark budget. At five-step updates, a 520-step episode can use 104 calls before retries; 500 episodes can use 52,000 calls for one method. Measure actual token use and latency before selecting the larger budget. Every schema retry counts as another call.

## Logging and reproducibility

Each run saves an immutable configuration identity, code/package hashes, simulator source/controller metadata, task list, unique episode records, compressed decisions and candidate scores, raw requests/responses, and optional GIFs. The GIF frame interval reflects simulator time; API waiting time is excluded. Simulated success does not establish real-time control.

The CLI distinguishes success, timeout, API failure, schema failure, simulator error, policy error, and budget exhaustion. A model stage label cannot mark an episode successful. Official environment success is checked after initialization and each executed action.

Run resumption requires identical source and settings. It skips recorded terminal outcomes and restarts interrupted episodes from their official initial state. It restores the API budget and conservatively accounts for request logs newer than the last snapshot. It does not reuse historical model answers as fresh samples. A crash-truncated final episode append is preserved as an artifact before recovery; earlier corruption is rejected.

Reports include per-task counts and Wilson intervals, a task-balanced mean with a within-task bootstrap, and paired differences on matching suite, seed, task, initial state, and simulator protocol. They reject mixed-suite pooling and duplicate episode identities. All-success or all-failure bootstrap intervals can degenerate; per-task Wilson intervals remain available. These intervals characterize sampled initial states for the observed tasks, not generalization to unseen tasks.

## Budget and API behavior

The client uses the official Responses endpoint, explicit `gpt-6-astra`, image inputs, and strict JSON Schema. It stores no conversation and requests the default service tier. At most one schema retry is allowed; refusals, incomplete output, and transport errors end that rollout without fallback.

Call and cumulative active-wall-time caps are available. A hard dollar cap additionally requires explicit prices. The conservative reservation covers a full documented input context, the configured output maximum, and applicable long-context premiums. The supplied cost example is intentionally conservative: a small nominal dollar cap can reject a small request because the client does not guess image-token usage. Unknown billing after transport failure retains the full reservation. Dollar figures remain bounds under the supplied pricing configuration, not verified invoices.

## Deliverables and current boundary

The code includes the planner, transformations, actor loop, optional official simulator bridge, live Astra client, dry request generation, resume support, diagnostics, and reports. Automated tests exercise numerical behavior, masks, input isolation, controller transforms, lifecycle, budgets, failures, and resumption.

Analytic fixtures validate software only. The initial fixture results show useful detour guidance but no advantage of three-step search over one-step search or of the full field over its affine approximation. Live Astra capability, full-task success, and the external pi0.5 reference remain experimental work requiring credentials and the chosen model/checkpoint budget.

## Field history and execution feedback experiment

The next controlled variant adds memory to the original field policy. The task
is LIBERO-Spatial task 0, official initial state 0, seed 7: pick up the black
bowl between the plate and ramekin and place it on the plate. The user authorized
a fresh **$300 API budget with no action-step limit** for one episode. This
ledger is separate from the previous field and direct-history experiments.

The spatial interface remains a 9³ grid with 0.02 m spacing, refreshed after
every five executed actions. Every action uses the original three-step local
planner with 0.01 m axis primitives, then executes its first move. Orientation,
gripper interpretation, masks, field interpolation, and controller scaling are
unchanged. Field values remain unitless preferences, not metric distances.

Each new field request retains all previous textual observations, accepted
Responses output items, and execution feedback. Only the latest two model
observation image pairs are retained. Previous field arrays stay associated
with their original observation origins; they are not recentered on the current
robot. Opaque encrypted reasoning items are replayed without interpretation.

After each executed action, the policy records the issued normalized action,
before/after eight-value robot state, desired versus measured displacement,
finger-joint changes, and the same 25-action motion statistics used by the
direct-history baseline. A subsequent action is rejected until its predecessor's
feedback has been recorded. No reward, success flag, task-finished hint, object
pose, contact predicate, or evaluator-only placement criterion enters the model
input. There is no new automatic recovery or forced grasp/phase transition.

Validation must establish identical plans and actions to the original field
policy under identical predictions, correct transcript retention and feedback,
budget/error handling, and a zero-API real-simulator smoke with clearly scripted
fields. The live video must show observation, field, local plan, issued action,
and measured execution feedback. Success ends the episode; user interruption,
insufficient request-budget headroom, or infrastructure errors leave it
incomplete and unscored. The $300 ceiling reserves the full possible next
request before sending it, so a budget stop can occur below $300 settled usage.

## Primary implementation references

- [Official openpi LIBERO evaluation](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)
- [Official openpi image conversion](https://github.com/Physical-Intelligence/openpi/blob/main/packages/openpi-client/src/openpi_client/image_tools.py)
- [Astra model interface and pricing](https://developers.openai.com/api/docs/models/gpt-6-astra)
- [Responses structured outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [Image input format](https://developers.openai.com/api/docs/guides/images-vision)
