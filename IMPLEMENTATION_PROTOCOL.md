# Implementation protocol: direct actions, field planning, and hybrid control

This document specifies the code that was actually implemented in `gpt_field`, including the information supplied to Astra, the actions executed by the simulator, and the rules used to stop and score an episode. It describes the current experiments; it is not a claim that a learned metric distance field, a contact detector, or a complete grasp controller has been implemented.

Recorded outcomes, videos, and the limits of comparison are collected in [EVALUATION_REPORT.md](EVALUATION_REPORT.md). Mathematical definitions, pseudocode, and code/model diagrams are in [METHOD_AND_ARCHITECTURE.md](METHOD_AND_ARCHITECTURE.md).

The opt-in `--field-improvements` protocol introduced on 2026-10-12 uses the
grounded compact schema and event-triggered field refresh described in the
[README](README.md). It requires compact Hybrid, tracking, medium effort and
LIBERO-only completion. Its geometric path has at most ten segments; its field
refresh interval is at most thirty actions. Direct still executes all five actions.
The fixed-cadence tables and historical runs below describe the older protocols.
`field_control_event` trace rows are emitted immediately after the acknowledged
feedback that caused a refresh; their `step` is the next action index. Raw compact
predictions, effective gripper goals, grounding diagnostics and source snapshots
remain available for reconstruction. Offline tests are not full-task timing results.

The immediate objective is **planning usefulness**: can model-produced spatial potentials guide useful robot actions? Metric distance fidelity is a separate question. The primary `values` output is a finite, arbitrary-scale, lower-is-better potential, with no claimed distance unit. The separate fidelity command and its illustrative reference format do not turn these manipulation experiments into distance-estimation evaluations. See [the original implementation plan](IMPLEMENTATION_PLAN.md), [schema](src/gpt_field/schema.py), and [README](README.md).

## 1. Experimental conditions and scope

The three current control interfaces are direct actions with history, field planning with history, and hybrid field/direct control. The earlier field condition without model history is a distinct historical condition.

| Condition | Model output at each refresh | Between refreshes | Model history and execution feedback | Implemented episode route |
|---|---|---|---|---|
| Original field, `field` | One local potential grid plus orientation/gripper commands | Replan translation from the current measured position on every action | Current model observation only; the host still maintains the active field and gripper intent | Historical API-key runner |
| Direct with history, `direct_history` | Exactly five normalized 7D actions | Execute the returned actions in order | Full retained textual history and per-action measured feedback; latest two model-observation camera pairs | API-key full-task runner |
| Field with history, `field_history` | Same field contract as the original field | Same original local planner and action adapter | Same history/feedback policy as the direct baseline | API-key full-task runner; subscription access probe and first-close diagnostic |
| Compact hybrid, `hybrid_field_direct` | One routing decision and either compact field parameters or five direct actions | Expand to 33³ locally; execute 30 field actions or five direct actions before the next model request | One shared history and feedback stream across both controllers | Subscription full-task runner |

“Plan-only” means the **field-planning control condition** here. It still sends executed robot actions and controls orientation and the gripper. It is different from the static subscription probe, which generates a field and a local plan but executes no robot action.

The exact task is:

> pick up the black bowl between the plate and the ramekin and place it on the plate

The current single-episode bowl experiments use **official LIBERO-Spatial task 0, official initial-state index 0, seed 7**, with a Panda robot and `OSC_POSE` control. The earlier stateless-field batch used ten fixed official initial states, indices 0–9. These are not ten random starts or a completed LIBERO-10 benchmark. The project’s broader LIBERO-10 evaluation support does not change the identity of this task.

The simulator source is explicitly loaded from `/workspace/heading_flow_pub/third_party/LIBERO`. The compatible runtime is `/workspace/venvs/libero-plus/bin/python`; that runtime’s name does not mean the tasks were substituted with LIBERO-plus. The worker rejects a LIBERO-plus source checkout and records the loaded source tree, task-file hashes, controller sources, and package versions. The vendored LIBERO checkout has no independent upstream Git revision available, so recorded hashes are the relevant local provenance. See [simulator implementation](src/gpt_field/libero.py) and [recorded hybrid simulator metadata](runs/chatgpt_hybrid_001/simulator.json).

## 2. Reset, observation, and controller contract

### 2.1 Reset and simulation timing

Each fresh episode in the specialized single-episode runners follows this sequence (the earlier ten-episode batch substitutes initial-state indices 0–9):

1. Construct official LIBERO-Spatial task 0 with the verified `OSC_POSE` controller and Panda robot.
2. Seed the environment and random-number generators with 7.
3. Load official initial state 0 from the task’s stored set of 50 initial states.
4. Execute ten settling actions `[0, 0, 0, 0, 0, 0, -1]`.
5. Record the initial policy observation and begin action numbering at zero.

Settling actions occur before the evaluated policy trajectory and are not counted as policy actions. The simulator runs at **20 Hz**: one policy action advances **0.05 seconds** of simulated control time. A five-action standalone or hybrid-direct block spans **0.25 seconds**; a 30-action hybrid-field block spans **1.5 seconds**. The simulator pauses while waiting for a model response. Wall-clock runtime therefore includes inference and rendering latency and is different from simulated motion time.

For the unbounded episode runners, `max_steps=None` is serialized and the worker uses an infinite simulator horizon with `ignore_done=True`. The worker still explicitly checks task success. There is no concealed 520-action limit from the generic `LiberoConfig` default in these runs. Per-request HTTP and simulator-response timeouts remain infrastructure safeguards, not an episode action horizon.

### 2.2 Allowed model observation

The policy boundary accepts exactly these four keys:

| Key | Content |
|---|---|
| `observation/image` | External `agentview` RGB camera |
| `observation/wrist_image` | `robot0_eye_in_hand` RGB camera |
| `observation/state` | Eight finite robot-state values |
| `prompt` | Exact task instruction |

Both cameras render at **256 × 256**, are rotated **180 degrees**, and are converted to **224 × 224 RGB PNGs** using bilinear resize with centered padding when necessary. Requests contain inline images at `detail="high"`. No depth or segmentation channel is supplied.

The eight-value state is:

```text
[eef_x, eef_y, eef_z,
 orientation_axis_angle_x, orientation_axis_angle_y, orientation_axis_angle_z,
 left_finger_qpos, right_finger_qpos]
```

End-effector position is in world-frame meters; orientation is a rotation vector in radians derived from the observed end-effector quaternion. The last two values are the measured gripper joint positions. These are **not eight arm joint angles** and are not an object pose.

The worker discards extra simulator observation fields before returning a `PolicyObservation`. Object poses, object identities beyond the task text, contact flags, reward, benchmark predicates, and task success are not model inputs. Success and other simulator status remain evaluator-side records. See [observation conversion and allowlist](src/gpt_field/observation.py).

### 2.3 Executed action

Every executed action contains exactly seven finite values in `[-1, 1]`:

```text
[dx, dy, dz, rx, ry, rz, gripper]
```

| Component | Meaning |
|---|---|
| `dx, dy, dz` | World-frame translation delta targets; `1` corresponds to **0.05 m per axis** |
| `rx, ry, rz` | World-frame axis-angle rotation delta targets; `1` corresponds to **0.5 rad per axis** |
| `gripper = +1` | Close command |
| `gripper = -1` | Open command |
| `gripper = 0` | Zero incremental gripper command; not proof of closure or a grasp |

Each arm target is relative to the current measured pose, not the previous requested target. Controller targets are not guaranteed realized displacements. Zero arm motion requests the current pose; contacts and dynamics can still produce measured movement. The worker checks controller scaling against the installed robosuite implementation. See [action adapter](src/gpt_field/action.py) and [simulator checks](src/gpt_field/libero.py).

## 3. Shared history and execution feedback

The direct-history, field-history, and hybrid conditions share the same retained-history policy:

- Preserve every textual model observation, every accepted model output item, and every execution-feedback record for the episode.
- Preserve complete accepted Responses output items, including IDs and assistant `phase` when present. Encrypted reasoning is retained only as opaque protocol data; the implementation does not interpret or render it.
- Keep images from only the **latest two model-observation messages**, each containing the external and wrist cameras. Older image contents are replaced with explicit omission markers; their text is retained.
- Use `store=false` and resend the locally maintained transcript. There is no hidden server conversation or automatic text summarization.
- Commit a model output to history only after response completion and schema validation. Invalid outputs do not become an accepted turn.
- Clear episode history on reset without resetting the client’s lifetime usage ledger. Context exhaustion ends the run; it does not silently remove textual history.

“Latest two observations” refers to observations at model refresh boundaries. The simulator returns images after every action, and those observations are recorded, but the model is queried only at block boundaries (five actions for the standalone baselines and hybrid direct; 30 for hybrid field). It does not receive a fresh visual turn after each individual action inside a block.

After every acknowledged simulator action, the runner calls `observe_transition(before, decision, after)` before requesting another action. The policy enforces both the exact pending decision and equality between the next observation and the preceding measured `after` observation. This prevents skipping feedback or associating it with a different action.

Each feedback record contains:

- Global zero-based `step`, chronological `chunk_index`, and within-block `chunk_step` (30 actions for hybrid field; five for hybrid direct and the standalone baselines). Hybrid block indices count accepted model decisions; global step division cannot recover them when block lengths differ.
- Measured before/after eight-value states and the exact executed normalized action.
- Desired translation `0.05 × action[:3]` and desired world rotation `0.5 × action[3:6]`.
- Measured translation `after_xyz − before_xyz`, its norm, the commanded translation norm, and translation tracking error `||measured − desired||`.
- Both finger-joint changes and the executed gripper command.
- Statistics over the latest **25 actions**: commanded/measured path length, measured net translation, close-command count, and command/low-motion counts.

The rolling statistics classify a translation command when its requested norm is at least **0.001 m**, and low motion when such a command produces a measured norm at most **0.0001 m**. These are measured robot-motion descriptors, not reward, object progress, grasp labels, or a trigger for automatic recovery. Measured rotational tracking error is not separately computed by this feedback schema.

The exact feedback method is shared from [DirectHistoryPolicy](src/gpt_field/direct_history.py); [FieldHistoryPolicy](src/gpt_field/field_history.py) reuses it, and [HybridFieldDirectPolicy](src/gpt_field/hybrid_policy.py) adds cross-controller gripper-memory bookkeeping after it succeeds.

## 4. Direct-action baseline

`DirectHistoryAstraClient` uses prompt version `astra-direct-history-v1`. Astra returns a strict object with one key:

```text
{"actions": [action_0, action_1, action_2, action_3, action_4]}
```

Each action is the seven-value normalized command described above. The validator requires exactly a **5 × 7** array, finite numbers, and values within `[-1, 1]`.

`DirectHistoryPolicy` executes each returned action unchanged. It does not call the field planner or the field orientation adapter, clip a valid model action into a more restricted action set, infer a grasp phase, force closure, or replace repeated stationary actions. The model receives the accumulated execution feedback and new images when the five-action chunk is exhausted. Task success, an interruption, or an error can terminate the episode before all five actions are used.

Thus the direct baseline is open-loop at the **model-command level within each five-action chunk**, while the low-level OSC controller continues to use its normal feedback. The measured transitions are logged immediately and inform the next model request; they do not rewrite the remaining actions in the current chunk. See [direct history client and policy](src/gpt_field/direct_history.py) and [full-task runner](scripts/evaluate_bowl_direct_history.py).

## 5. Field-planning condition

### 5.1 Model output and fixed world grid

This section describes the original and standalone field configuration. Hybrid v5 uses the same field semantics with a 33×33×33, 0.005 m grid, a ten-step exact lattice planner, and a 30-action field refresh interval; see §6.

The original field interface uses prompt version `astra-potential-v1`; the history extension uses `astra-field-history-v1`. The nested field schema is the same in both:

| Field | Meaning |
|---|---|
| `stage` | Nonempty descriptive label used in logs; not an executable phase switch |
| `orientation_target_xyzw` | Unit quaternion target, tolerance `0.001` on norm |
| `gripper_command` | `open`, `close`, or `hold` |
| `motion_mode` | `move` or `hold` |
| `values` | Exactly **729 finite potentials** |
| `blocked_indices` | Predicted invalid grid-vertex indices |
| `unknown_indices` | Predicted unknown grid-vertex indices, also invalid for planning |

The grid has **9 × 9 × 9 vertices**, **0.02 m spacing**, and world-aligned axes. It is centered at the measured end-effector position in the observation that generated the field. Each axis spans offsets `[-0.08, +0.08]` m, giving a **0.16 m side length**. Flattening uses `81*i + 9*j + k`: x varies slowest and z fastest.

The grid center remains fixed for the whole five-action block. Local replanning uses the latest measured position inside that original coordinate frame. Historical fields keep their own original centers; they are not reinterpreted around the latest end-effector position. A new field is generated on the next five-action boundary.

`values` are task-conditioned potentials. They can be negative, have arbitrary finite scale, and need not be monotonic with Euclidean distance. There is no Eikonal loss, Riemannian metric estimation, signed-distance guarantee, gradient training, or conversion from a scalar field value to physical proximity in this implementation. See [field schema](src/gpt_field/schema.py).

### 5.2 Interpolation and predicted validity

`GridField.query` uses trilinear interpolation. Every corner with nonzero interpolation weight must be valid. An exact valid grid vertex or grid-aligned face does not require off-face corners with zero weight. Outside-grid or malformed queries return no valid value.

Both blocked and unknown masks invalidate vertices. Their index sets must be unique, disjoint, and in range; masked placeholder potentials must still be finite. `segment_valid` checks the endpoints, every grid-plane crossing, and a midpoint in each intervening interval. These points cover all interpolation stencils traversed by a segment, including short cell crossings that fixed-spacing samples could miss.

This validates motion only against the **model’s predicted masks**. It is not robot-link collision checking, a swept-volume test of the gripper/object, contact feasibility, or a physical collision guarantee. See [field implementation](src/gpt_field/field.py).

### 5.3 Exact local planning objective

The default planner uses six nonzero translation primitives:

```text
+x, -x, +y, -y, +z, -z; each primitive is 0.01 m
```

At every policy action it enumerates **6³ = 216** horizon-three primitive sequences. Every sequence has **0.03 m total commanded travel length**. It rejects an immediate axis reversal, a waypoint outside workspace bounds `[-1, -1, 0]` to `[1, 1, 1.5]` m, or a segment crossing an invalid field stencil. The start must also be inside the workspace and valid in the field.

For the remaining candidates, with intermediate points `p1, p2, p3`, selection is:

1. Minimize terminal potential `V(p3)`.
2. Among numerical terminal ties, minimize immediate potential `V(p1)`.
3. Break remaining ties by the fixed primitive ordering shown above.

There is **no weighted path-length penalty, integrated field cost, learned trajectory cost, or gradient-descent optimizer**. Equal commanded travel length is built into the candidate set. The tie tolerance is `64 × float64_epsilon × max(abs(valid_vertex_potentials))`; masked values do not determine that tolerance.

Only the **first 1 cm primitive** of the chosen horizon is executed. After the measured transition, the planner recomputes a horizon from the new actual position. `predicted_improvement = V(start) − V(p3)` classifies the action as `move` or `plateau_move`, with default plateau tolerance zero. A flat or non-improving field does **not** automatically cause waiting: the selected nonzero primitive is still executed and the plateau is logged.

Explicit `motion_mode="hold"`, an invalid start, or no valid candidate produces zero **translation**. The orientation adapter and gripper command still operate, so this does not necessarily produce a zero seven-dimensional action. See [LocalPlanner](src/gpt_field/planner.py).

### 5.4 Orientation and gripper adaptation

The field specifies one orientation target per refresh. On each action, `ActionAdapter` computes the shortest world-frame rotation error from the current measured orientation to that target, caps its rotation-vector norm at **0.15 rad**, and divides by the OSC scale of `0.5 rad`. Translation is divided by `0.05 m`; the standard 1 cm primitive therefore becomes a normalized single-axis command of `±0.2`.

Field gripper commands set `open → -1` or `close → +1`. `hold` repeats the adapter’s remembered open/close intent, initially open. It is distinct from direct action `gripper=0`. Translation above the verified per-axis physical limit is rejected before adaptation rather than silently changed into a different planned translation.

The field constrains translational search only; it is not a joint SE(3), gripper, and contact trajectory optimizer. Orientation and gripper intent are separate model outputs applied through this adapter. See [ActionAdapter](src/gpt_field/action.py).

## 6. Hybrid routing and execution

The current live hybrid uses the compact `analytic-goal-rbf-v1` field representation. The legacy dense client retains prompt version `astra-hybrid-field-direct-history-v5`. After **30 field actions or five direct actions**, **one** Astra response selects a controller and supplies that controller’s payload. There is no separate routing-model request followed by a second control-model request.

The strict envelope has exactly these keys:

```text
{
  "controller": "coarse_field" | "contact_direct",
  "target_region": "black_bowl" | "plate",
  "near_target": boolean,
  "routing_reason": nonempty string, at most 500 characters,
  "field": compact field parameters | null,
  "direct": {"actions": five normalized 7D actions} | null
}
```

The validator requires `near_target=false` for `coarse_field` and `near_target=true` for `contact_direct`; the inactive payload must be `null`. `routing_reason` is a brief observable explanation, not a required chain-of-thought transcript.

The compact field contains `stage`, `orientation_target_xyzw`, `gripper_command`,
`motion_mode`, `goal_offset_m`, `attraction_weights`, `repulsors`, `blocked_boxes`,
and `unknown_boxes`. All positions are offsets in world axes from the measured
end-effector position at generation. With offset `r`, goal `g`, weights `w`, and
repulsors `(c_j, radius_j, weight_j)`, local expansion computes:

```text
V(r) = sqrt(sum_i w_i * ((r_i - g_i) / 0.08)^2)
       + sum_j weight_j * exp(-0.5 * ||r - c_j||^2 / radius_j^2)
```

There are at most four repulsors and four boxes of each mask type. Vertices
inside inclusive axis-aligned boxes receive the corresponding mask, with blocked
taking precedence over unknown. Local expansion uses no simulator geometry.
The 33³ expanded field enters the existing planner; only the raw compact response
is replayed in GPT history. This is a restricted analytic potential family, not a
metric distance field or a claim of 0.5 cm perception accuracy.

The routing instruction asks Astra to use fields for free-space approach or transport, and direct actions for interaction near the active bowl/plate region, including alignment, grasping, immediate lift clearance, placement, or release. **Near-target is a visual/model judgment.** It is not a measured contact flag, hidden object-distance threshold, field-value threshold, or parsed `stage` string.

The selected controller remains active for 30 actions in field mode or five actions in direct mode, subject to task success or interruption. Switching is reversible at every block boundary, measured from that block's actual start. For field → direct → field, requests occur at zero-based steps 0, 30, and 35. Holding the bowl does not permanently lock the policy into direct mode: it may return to field planning for transport and later choose direct placement. There is no hard-coded approach/grasp/lift/place finite-state machine or automatic recovery sequence.

Execution rules:

- **Coarse field:** feed the already validated 33×33×33 field (0.005 m spacing) to `FieldPolicy` through a cache that makes no model calls. The grid still covers 16 cm per axis. Plan ten 1 cm primitives with exact lattice dynamic programming and execute all ten in order before replanning from the latest measured position. A full field block plans at relative steps 0, 10, and 20, then requests a new model decision at step 30. Interpolation, masks, terminal/immediate/lexicographic selection, and action scaling retain their existing meanings.
- **Contact direct:** consume exactly the validated five-action array, unchanged, without field planning or the field rotation cap.
- **Return to coarse field:** request a new field centered at the current measured pose; never reactivate an old field centered before direct motion.
- **Shared gripper memory:** after an acknowledged action and valid transition feedback, remember its last nonzero open/close intent across both controllers. Direct `0` is executed as `0` and leaves this remembered intent unchanged. A later field `hold` repeats the remembered intent, initially open.
- **Shared history:** mode changes do not reset the transcript, step numbering, feedback window, or image-retention rule.

A field retains its original center for the complete 30-action block. Coverage stays ±8 cm, so a ten-step path may turn but cannot go straight 10 cm from the center. Every predicted segment must remain valid inside the field and workspace when planned. An invalid start or absence of any complete ten-step path commits zero translation for that execution window, while orientation and gripper commands still apply; the planner does not shorten the requested horizon. There is no early-refresh or mid-window replanning rule. Flat/non-improving valid fields retain the original `plateau_move` behavior. Expanded grid values remain local; model history contains the compact parameters.

During execution, translation is the next fixed world-frame primitive from the selected path, not the first primitive repeated and not a correction from measured position to a stored waypoint. Orientation is still adapted against each actual measured state, and every action requires transition feedback before the next action. Predicted path validity does not certify a tracking-disturbed realized trajectory. Success, interruption, and errors may end a window early. Reset, controller handoff, and new field generation discard any queued moves. For explicitly configured refresh intervals that are not divisible by the execution window, the last window is truncated at the model boundary.

The ten-step search is exact dynamic programming, not beam pruning. At each depth it merges prefixes with identical integer displacement, first direction, and last direction, retaining the lexicographically first prefix and summing represented path counts. It caches lattice-node potentials and edge validity. After all ten layers it finds the global terminal-potential minimum and its numerical tie band, keeps one lexicographic finalist per first direction, then applies the immediate-potential and lexicographic tie breaks. No running path-cost term is added. Only horizons above six use this search; the original horizon-three baseline remains exhaustive.

`plan.candidates`, `candidate_count`, and `valid_count` describe the at-most-six logged terminal-band finalists for this planner. They are not the number of all searched paths. `plan.search` identifies `exact_lattice_dp` and records `nominal_candidate_count=60,466,176`, `represented_valid_sequence_count`, layer state counts, and transition/cache counts. The video labels these paths as finalists. The selected finalist contains eleven positions (start plus ten moves) and ten primitives, with 0.10 m total commanded travel.

The recorded `chatgpt_hybrid_001` success used **v1: 9×9×9, 0.02 m, five-action blocks**, and does not establish performance for the compact method. The dense client preserves v1–v5 for protocol verification. The live runner now defaults to compact fields with both planning horizon and execution window ten; its scripted smoke retains the dense fixture.

The trace records `routing_decision` at each refresh and `controller_switch` when the controller changes after an earlier selection. Both are labeled as model predictions. Field action diagnostics distinguish `plan_refreshed`, global `plan_start_step`, within-plan `plan_step`, committed `plan_execution_steps`, and the current `planned_delta_xyz`. The cached full `plan` remains unchanged during its execution window; `plan.delta_xyz` remains that plan's first move. `local_plan_count` counts actual searches. The video emits `local_replan` only when planning occurs and `field_action` for cached moves, highlighting the current primitive. Contact-direct frames show the returned action chunk and measured motion, with no stale field or invented local plan. See [hybrid schema/client](src/gpt_field/hybrid_subscription.py), [hybrid policy](src/gpt_field/hybrid_policy.py), [runner](scripts/evaluate_bowl_hybrid_subscription.py), and [video recorder](src/gpt_field/hybrid_video.py).

## 7. Model transport, settings, and credential separation

### 7.1 API-key experiments

The historical stateless-field, direct-history, and full-task field-history runners use the official Responses endpoint with an explicit `OPENAI_API_KEY` or a hidden terminal prompt. They use model `gpt-6-astra`, strict JSON schemas, `store=false`, a configured **180-second socket timeout**, and a transmitted **8,192-output-token limit**. The base payload requests `service_tier="default"`.

The socket timeout bounds blocking connection/read operations; it is not an end-to-end inference deadline. In particular, a subscription stream can continue beyond 180 seconds while data or heartbeats keep arriving. An episode-level wall-time budget, when configured in other runners, is separate.

These runners require an explicit finite API dollar cap. The ledger counts real attempts, known usage, and conservative reservations for requests whose usage is unknown. A request is not sent unless its full configured reservation fits; consequently a budget stop can occur before settled cost equals the cap. Historical pricing/configuration is part of the recorded experiment, not a claim about future prices.

The API-key client allows at most one fresh sample after a schema-validation failure. Transport failures, refusals, and incomplete responses are not generally retried. Schema retries and unresolved attempts count toward usage. See [Astra client/configuration](src/gpt_field/astra.py).

### 7.2 Subscription experiments

The subscription implementation uses the app’s own official Sign in with ChatGPT registration and protected local OAuth credentials. It does not read Codex’s credential files, inherit billing from the IDE, use an API key, or silently substitute another model. Authentication uses PKCE/state/nonce validation, signed ID-token validation, an explicit direct-use grant, separate account registrations, and serialized token refresh. Credentials are stored under `.private/chatgpt_plan/`, outside run artifacts, with owner-only permissions. See [authentication implementation](src/gpt_field/chatgpt_auth.py).

`ChatGPTPlanFieldClient` and `ChatGPTPlanHybridClient` use the public Responses endpoint with `stream=true` and `store=false`. Unsupported parameters, including `max_output_tokens`, are omitted from the actual request. The inherited configuration object still contains an `8192` default, but it is **not a transmitted or enforced subscription output cap**; artifacts explicitly record `output_token_limit_sent=null`.

The transport accepts output only after `response.completed`. If the terminal `output` array is empty, it can assemble the exact complete `response.output_item.done` items in index order after completion. It rejects unfinished, missing, duplicate, conflicting, or model-mismatched outputs. An observed `text/plain` response label is accepted only through the same strict SSE parser, with the discrepancy recorded. This is transport compatibility handling, not partial-output execution.

Subscription requests have no automatic retry or API-key/model fallback. Quota, authentication, context, malformed-output, and stream failures stop the episode. The ledger reports actual subscription request/token counts and unknown usage; it does not convert them into API-key dollars or claim a zero-dollar monetary bill. Account plan and credit settings remain external limits. See [subscription transport](src/gpt_field/chatgpt_plan.py).

### 7.3 Reasoning effort and service tier actually used

The standalone baselines omit a `reasoning` configuration object. Current hybrid requests explicitly include **`reasoning={"effort":"high"}`** through `ChatGPTPlanHybridClient`, and the runner records `policy.reasoning_effort` in its preflight and manifest. This setting uses the same ChatGPT subscription OAuth transport; it does not enable API-key billing. An explicitly conflicting effort in response metadata is rejected before actions/history are accepted, with no retry or fallback. Missing effort metadata does not establish the resolved setting. Explicit `reasoning_effort=None` is available on the client to reproduce historical omitted-effort requests. Runner CLIs expose no effort/tier selector, and the shared payload still carries `service_tier="default"`.

For the completed hybrid run, **all 90 recorded responses** report `model="gpt-6-astra"`, `reasoning.effort="medium"`, and `service_tier="default"`. Thus “medium” describes the observed resolved model setting; it was not a user-selected `--reasoning-effort` experiment. “Default” is the recorded standard/default service route. Neither “maximum effort,” “high effort,” nor priority service was requested. This statement concerns public response metadata only, not opaque reasoning content. See [configuration implementation](src/gpt_field/astra.py), [hybrid manifest](runs/chatgpt_hybrid_001/manifest.json), and [recorded response metadata](runs/chatgpt_hybrid_001/requests/000001-ad37697575064384b00b5112255f16a1.response.json).

## 8. Task success, stopping, and scoring

The authoritative evaluator is `OffScreenRenderEnv.check_success()`, checked after reset/settling and after each actual action. The bowl task’s goal is `On(akita_black_bowl_1, plate_1)`.

In the installed official LIBERO implementation, this goal delegates to the plate’s `check_ontop(bowl)`, which requires:

1. Bowl body-origin height at or above the plate body-origin height.
2. Physical bowl–plate contact according to the simulator.
3. XY body-origin separation **strictly less than 0.03 m**.

It does **not** additionally require an open gripper, absence of gripper contact, or post-release settling. The hybrid run therefore correctly satisfies its recorded benchmark criterion, while release and stable unsupported placement remain separate untested criteria. The final hybrid action still commanded gripper closure. Any future release-and-settle requirement must be an explicitly separate evaluator condition rather than a retrospective reinterpretation of the recorded success.

Predicate sources: [task BDDL](../heading_flow_pub/third_party/LIBERO/libero/libero/bddl_files/libero_spatial/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate.bddl), [On predicate](../heading_flow_pub/third_party/LIBERO/libero/libero/envs/predicates/base_predicates.py), and [object-state predicate implementation](../heading_flow_pub/third_party/LIBERO/libero/libero/envs/object_states/base_object_states.py).

| Runner type | Normal completion | Other stops | Scoring |
|---|---|---|---|
| Full-task API-key direct/field | Actual LIBERO success | API budget, user interruption, infrastructure/API failure | Only successful completion scores; incomplete attempts keep `success_rate=null` |
| Full-task subscription hybrid | Actual LIBERO success | User interruption, subscription/auth/quota/context failure, infrastructure failure | Successful episode scores; incomplete attempts keep `success_rate=null` |
| Compact Field / Direct timing comparison | Equal requested action counts in every paired arm (default 3 × 30 per controller) | User interruption, subscription failure, infrastructure failure, early simulator termination | Timing only; no success score and no overall speed ratio for incomplete pairs |
| Subscription first-close diagnostic | First actual positive gripper action acknowledged, followed by saved observation/feedback | User/API/infrastructure stop or earlier simulator termination | Always unscored; not a full task evaluation |
| Static subscription field probe | One validated field and local plan | Access/protocol/validation failure | Zero executed robot actions; always unscored |
| Scripted smoke | Fixed diagnostic action count, or earlier termination/error | Infrastructure failure | Always unscored and visibly labeled scripted |

The full-task direct, field-history, and hybrid runners have no policy-step, model-call, or episode-wall-time horizon. A close command does not end a full-task episode. An action block can end early on task success, and no further action or model request is sent after the successful transition is saved. Errors in required feedback or finalization do not produce a false completed score.

The separate timing comparison fixes the controller rather than asking the model
to route. Its Field schema contains only compact parameters; its Direct schema
contains only five actions. Each arm starts with empty model history and the same
official initial state/seed; both state and image digests must match. Replicate
order alternates Field/Direct and Direct/Field. Every proposed action must receive
simulator acknowledgement and feedback. An early terminal state is recorded as an
incomplete arm, never filled with invented actions. The test does not compare
success rates, path quality, or time to task completion.

The ledger measures model transport (network, generation, stream parsing), local
field expansion, local planning, the complete policy decision and feedback, and
simulator stepping separately. Transport, expansion, and planning are subsets of
policy time and must not be added to it again. `control_loop_s` is policy plus
simulator time; startup, reset, external trace writes, and finalization are excluded.
Request logging inside policy calls is included. Simulator startup/reset and full
run wall time are also recorded. Actual holds remain counted as executed actions.

The current completed hybrid result is **one success in one episode**, not a statistically reliable estimate across initial states or tasks. Budget-limited baselines must not be silently converted to completed 0% evaluations or pooled with setup probes and smoke tests.

## 9. Reproduction commands

Run commands from `/workspace/gpt_field`. Use a **new output directory** for every fresh run; the specialized runners refuse to overwrite an existing experiment. These commands document execution paths; reading this protocol itself does not launch an experiment.

### 9.1 Environment and dependency checks

The project already has a local `.venv`. Core setup and tests use Python 3.10 or later; the simulator runs in its separately specified compatible environment.

```bash
cd /workspace/gpt_field
.venv/bin/python -m pip install -e '.[dev,subscription]'
.venv/bin/python -m gpt_field doctor
.venv/bin/python -m pytest -q
```

`requirements.lock.txt` records the validated core environment but does not pin the separate simulator runtime. For historical reproduction, inspect each run’s `manifest.json` and `simulator.json` before replacing installed versions. Preserve the simulator virtual-environment executable path; resolving its symlink to the base Python can select the wrong environment.

`MUJOCO_EGL_DEVICE_ID=2` was the rendering device selected on this instance. Change it only to an available rendering device on another machine. The BLAS thread limits below keep the recorded execution setup reproducible.

### 9.2 Direct actions with history: full task through API-key transport

Preflight does not call the model, authenticate, start a simulator, or write run artifacts:

```bash
.venv/bin/python scripts/evaluate_bowl_direct_history.py --preflight
```

The scripted smoke uses two synthetic responses and ten real simulator actions:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_bowl_direct_history.py \
  --offline-smoke --output runs/protocol_direct_smoke
```

A fresh live episode uses an intentionally selected API-key budget; `300` below reproduces the final historical budget scale, not a subscription limit. Supply the credential through the environment or the runner’s hidden prompt, never as a literal in the command:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_direct_history.py \
  --max-cost-usd 300 --output runs/protocol_direct_live
```

The historical 625-action direct episode spans [its initial segment](runs/astra_direct_history_001/summary.json) and [a specifically validated continuation](runs/astra_direct_history_continuation_001/summary.json). Those are one episode, not two independent trials. Fresh execution is the appropriate default; specialized historical continuation utilities are not generic resume mechanisms.

### 9.3 Field with history: full task through API-key transport

```bash
.venv/bin/python scripts/evaluate_bowl_field_history.py --preflight --max-cost-usd 300
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_bowl_field_history.py \
  --offline-smoke --output runs/protocol_field_smoke
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_field_history.py \
  --max-cost-usd 300 --output runs/protocol_field_live
```

The smoke uses two scripted fields and ten actions. The live reference is [astra_field_history_001](runs/astra_field_history_001/summary.json).

For the distinct original field **without model history**, the historical runner is:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_until_success.py \
  --episode-index 0 --api-config configs/astra_unbounded_episode.json \
  --output runs/protocol_original_field_live
```

That file supplies its own finite historical API budget and price configuration. It is not interchangeable with the history runner, and its historical multi-segment records must not be counted as additional reset episodes.

### 9.4 Subscription login and bounded access/execution diagnostics

```bash
.venv/bin/python scripts/chatgpt_plan_login.py login
.venv/bin/python scripts/chatgpt_plan_login.py status
```

For this remote workspace, privately forward local port 1455 to the instance’s loopback port 1455 and open `http://127.0.0.1:1455/login`. Complete the OpenAI consent yourself and review the app’s subscription/credit permissions. The temporary helper does not need a public endpoint. Account selection is also available through `--profile` and the login helper’s `select` command; use `--help` for exact placement.

The optional access probe performs at most one inference attempt from the previous field run’s saved initial observation, followed by local planning and **zero robot actions**:

```bash
.venv/bin/python scripts/chatgpt_plan_field_probe.py \
  --output runs/protocol_subscription_probe
```

The field-only subscription execution diagnostic deliberately stops after the first acknowledged close command:

```bash
.venv/bin/python scripts/evaluate_bowl_subscription_close.py --preflight
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_subscription_close.py \
  --output runs/protocol_subscription_first_close
```

The recorded references are [the successful access probe](runs/chatgpt_plan_field_probe_004/summary.json) and [the first-close diagnostic](runs/chatgpt_subscription_close_001/summary.json). Neither is a full-task field-only subscription evaluation.

### 9.5 Hybrid: full task through subscription transport

```bash
.venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py --preflight
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py \
  --offline-smoke --output runs/protocol_hybrid_smoke
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_hybrid_subscription.py \
  --output runs/protocol_hybrid_live
```

The isolated smoke uses exactly **three scripted envelopes and 65 actions**: field/open for 30 actions, direct/close for five, and field/hold for 30. It verifies return to field mode and retention of close intent. That 65-action cap exists only in smoke mode. The live runner has no such cap and continues through closing to task success or an allowed stop.

The historical v1 completed reference is [chatgpt_hybrid_001](runs/chatgpt_hybrid_001/summary.json), with its [full video](runs/chatgpt_hybrid_001/episode.mp4) and [stage-indexed viewer](runs/chatgpt_hybrid_001/index.html).

Run the compact-versus-Direct timing comparison through the same authorized
subscription entry point:

```bash
MUJOCO_EGL_DEVICE_ID=2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u scripts/evaluate_bowl_hybrid_subscription.py \
  --timing-comparison --blocks 3 --actions-per-block 30 \
  --output runs/compact_field_timing_next
```

This uses `gpt-6-astra` and explicit high effort in both arms, with no API-key
fallback or automatic request retry. `summary.json`, `comparison.md`, per-arm
traces, and raw request/response artifacts retain the timing evidence. The normal
full-task hybrid defaults to compact fields; `--field-representation dense`
explicitly restores the legacy dense v5 protocol.

### 9.6 Independent offline audit

```bash
.venv/bin/python scripts/audit_hybrid_subscription.py \
  --run runs/protocol_hybrid_live --output runs/protocol_hybrid_live_audit
```

Use `--allow-offline-smoke` explicitly to audit a scripted smoke. An optional `--frozen-sources PATH` checks a recorded source-hash snapshot. The auditor reconstructs input histories, checks accepted output identity without inspecting opaque content, recomputes field plans, verifies exact direct actions and feedback, reconciles token accounting, checks source provenance and video event ranges, and decodes the complete MP4. It makes no model requests or simulator actions. It does not independently prove a grasp or re-simulate object dynamics. See [audit implementation](scripts/audit_hybrid_subscription.py) and [completed reference audit](runs/chatgpt_hybrid_001_audit/audit.json).

## 10. Artifacts and interpretation boundaries

Each fresh runner stores configuration/source provenance in `manifest.json`, simulator details in `simulator.json`, current/terminal bookkeeping in `run_state.json`, flushed decision/action/observation/feedback records in `trace.jsonl`, and terminal outcomes in `summary.json`, `runner_summary.json`, and `episodes.jsonl`. Model records are under `api/` for the older API-key history runners and `requests/` for subscription runners. Request keys link decisions to their exact model input and response records.

Annotated `episode.mp4` files and `episode.events.jsonl` sidecars show recorded intermediate products. They omit API waiting time and hold events for legibility; video length is not wall-clock duration or simulator elapsed time. Field mode shows real field slices/candidates and the current selected primitive; direct mode shows real generated action chunks. Every smoke frame is labeled scripted. Success labels come from the evaluator, not a policy reward input.

This implementation already provides reversible visual routing, shared history, per-action feedback, ten-step exact field planning in the current hybrid, and direct control near model-selected interaction regions. It does **not** yet provide a matched full-task subscription evaluation of all three interfaces, repeated-initial-state reliability estimates, explicit reasoning-effort/tier experiments, oracle contact-based routing, automatic stall recovery, or a release-and-settle success criterion. Those would be new experimental conditions with their own frozen configurations and artifacts.

The generic code also includes `field_one`, `affine`, `shuffled`, `minimum`, and `waypoint` alternatives and a separate fidelity command. Their existence, and the synthetic offline demo, are not completed real-Astra manipulation results. Comparisons should distinguish implemented software paths from actually executed experiments.

Direct and field interfaces differ in action expressiveness as well as representation: direct commands may coordinate several translation/rotation axes within the full normalized limits, whereas the default field planner executes one 1 cm axial translation and caps adapted rotation at 0.15 rad. Hybrid changes the prompt and routing condition as well as switching controllers. The original field lacks the later model history/feedback. API-key and subscription conditions also differ in output-limit handling and budget termination. These differences must remain explicit when interpreting the observed runs.
