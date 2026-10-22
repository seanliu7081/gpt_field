# GPT field

## Contact-region feedback (opt-in)

`--contact-region-feedback apply` selects the new operation client and local
contact observer. `--contact-region-feedback shadow` uses the identical model
prompt and schema, learns the same kind of response and records proposed goals,
but executes the original nominal Field goals. Both imply `--operation-control`.
Add `--preflight` to inspect the complete configuration without credentials,
model calls, simulator startup or a rollout. Full-task runs still require the
user's explicit approval.

During pregrasp, GPT selects tight visible tool/target feature regions and their
desired relative image offset. Local optical flow chooses texture within those
regions and transports their image anchors. A visible wrist-tool edge lacking
flow corners can instead use a conditional fixed anchor: its frozen appearance
and edge pattern must match every frame while measured finger width and
orientation remain stable. This is a rigid-hand hypothesis, not measured optical
flow; target anchors still require tracking. The anchors are not measured contact
points, object poses or certified grasp locations. Unavailable views may remain
null; no missing measurement is manufactured. The local response is learned from
normal measured robot translations and validated on subsequent samples excluded
from the active fit. Only validated observable directions can revise a Field
goal, within a total 6 mm neighborhood of the model's immutable nominal goal and
a 4 cm tracking domain. Applied-motion prediction failures restore the nominal
goal; they do not introduce probes, waiting actions, automatic closure or extra
model requests. Direct retains its five exact model-authored actions.

New implementation: `contact_region_feedback.py` and `contact_region_policy.py`.
The original v4/legacy protocols remain available. Region diagnostics distinguish
tracking, held-out prediction, proposed goals, actually applied goals and physical
grasp evidence (which remains unknown to this observer). `apply` and `shadow`
are an ablation switch, not a claim that the new visual objective improves grasping.
Offline evidence is saved under `runs/contact_region_preparation_001/`; manual
RGB fixtures there are evaluation-only and never become zero-shot policy input.

Implementation validation: 344 offline tests passed, and four preflights
(bowl/drawer, apply/shadow) preserved medium/subscription and raw-LIBERO rules.
Each task's apply/shadow initial model payloads were identical. On eight frozen
historical windows from four contact checkpoints, final development replay
initialized 8/8 windows, retained tracking for 47 of 240 observations, and had
validated response models at 13 observations. It produced two unexecuted
handle-region corrections (1.28 and 1.49 mm); bowl windows produced none.
Active tracking computation had a median of 15.62 ms. Manual desired relations
preserve initial image offsets and are not certified grasp goals. Reusing these
fixtures during development does not establish held-out performance, GPT
grounding, physical grasp benefit or task speedup. Initial negative replays are
preserved. See `implementation_validation.json` and `development_comparison.json`
in the preparation directory. No live model request or new simulator task ran.

The subsequent authorized original-drawer zero-shot APPLY trial is saved under
`runs/contact_region_drawer_zero_shot_001/` and ended by explicit user stop:
697 actions, 41 medium subscription responses, raw LIBERO false. The drawer was
opened and released, but the bowl remained on the table. Video-excluded control
time was 1122.85 s, an incomplete duration rather than task-completion latency.
All compact pregrasp references were null; the new observer never initialized
or applied a correction. The bowl phase comprised 12 move and 3 recover requests,
without bowl pregrasp or closure. Five older body-region monitoring references
do not constitute contact-goal corrections. Model pipeline time was 936.38 s;
96 local plans took 0.160 s. This run therefore exposes an unresolved approach
and reference-generation problem and cannot establish the new correction's
physical benefit. No automatic retry or further task is scheduled.

## Paired contact-routing diagnostics

`scripts/evaluate_contact_routing_diagnostic.py` provides a separate diagnostic
runner for current v4 contact routing versus model-authored five-action Direct
contact correction. Free-space Field execution, gripper settling, verification,
RGB/state input and feedback remain shared. It replays identical historical
action prefixes, verifies both observable and full physics/controller state,
then starts fresh model histories. Prefix actions and simulator contact truth
are never policy input. This is a local experiment, not a full-task benchmark.

The approved eight trials in `runs/contact_routing_diagnostic_001` are complete:
52 Astra medium subscription requests and741 actions, with exact paired starts
and frozen sources verified. All eight reached a diagnostic bound before the
strict sustained-engagement criterion. This does not mean zero physical progress:
Direct5 at handle_b opened the drawer15.93mm with intermittent capture; Direct5
at bowl_a lifted the bowl6.30mm with final unsupported bilateral pad contact.
Neither demonstrated the required continuous10mm co-motion interval. No matched
completion-latency ratio is eligible. Setup/replay/video are excluded from net
control time; incomplete durations are retained as costs, not completion times.

Precise pregrasp/placement references and actual metric visual corrections were
zero. Body-ROI monitoring did run:14 references,70 unknown and4 contradicted
check observations, with no supporting observations. Contact-mode substitution
alone did not consistently resolve incorrect capture geometry or motion goals.
Independent contact/target-motion evidence and phase images are retained under
`preparation/live_contact_frames/`; aggregate results are in `analysis.json` and
the three terminal audits under `preparation/`. The close/check cap counts bounded
request outcomes, including an unsettled close whose check was skipped; it must
not be interpreted as a count of independent realigned grasp attempts.

## Operation-directed Hybrid v4 (opt-in)

`--operation-control --operation-planner field` selects a standalone operation
protocol instead of the legacy phase wrapper. It enables native512 RGB/state8,
the 1 mm lazy Field planner, attempt memory and local pregrasp feedback. The
model explicitly selects operation kind, target, intended outcome, prerequisites
and a finite local action budget. A bounded ledger retains dated model claims
and measured robot transitions separately; neither is hidden object-state truth.

Ordinary Direct remains exactly five actions. Field normally executes up to30;
explicit permission permits up to60 for the same goal only if both cumulative
and recent measured goal progress support continuation. Existing arrival,
invalidity and stall returns remain. Explicit close may include a model-selected
world translation of at most3 cm: after measured finger settling the local
controller performs that check from the fresh pose for at most30 additional
actions, within the total60 budget. No verification direction is inferred.

`visual_intent` separates pregrasp, held-object placement and generic body-ROI
tracking. Independent-view sparse optical flow supplies image-motion evidence;
fixed-view object/tool co-motion during an explicit check can support or
contradict attachment, but never certifies a grasp or task completion. Optional
precise visual corrections retain a total6 mm nominal-goal bound. Missing or
failed visual feedback falls back to normal model-selected execution without
probing, waiting or an extra dedicated model call. Optional dependency:
`pip install -e '.[operation-visual]'`.

`--operation-planner waypoint` uses the same operation controller and feedback
with straight waypoint tracking and validity checks, without field-cost search
or obstacle detours. This supports a future controlled planner ablation; it is
not a measured speed comparison or a new matched Direct baseline.

Preparation evidence: `runs/operation_v4_preparation_001/`.333 offline tests
passed. Saved RGB diagnostics retained at least one region for55/64 transitions;
manual initial ROIs are offline evidence only, never live input. These results
do not establish a valid metric response model or task improvement. The single
user-approved original-drawer zero-shot trial is recorded under
`runs/operation_v4_drawer_zero_shot_001/`; consult its terminal summary when
available. Astra medium subscription-only, raw LIBERO completion, collision
record-only, no episode cap, and video-excluded primary timing are retained.

That trial ended by explicit user stop:668 actions,41 model responses,raw LIBERO
false; net time excluding video1,068.45 seconds (incomplete duration). Six local
close/check operations ran without intermediate GPT calls, but the drawer never
opened. All14 pregrasp intents omitted precise references; no local corrections
or Field continuation requests occurred. Six body-ROI references were supplied,
but all42 verification observations remained unknown. No ordinary Direct actions
were selected. Model requests consumed86.55% of net time; planning took0.063s.
This run establishes neither task improvement nor speedup. See `analysis.json`,
`policy_terminal_analysis.json`, `terminal_audit.json`, and `trial/episode.mp4`.
The exact tested sources are archived in `runtime_sources.tar.gz`.

After the run, the ROI observer's rotation guard was corrected to use quaternion
angle rather than subtracting axis-angle vectors across the pi branch.96 focused
offline tests passed. The erroneous reasons had not cancelled verification
actions; sufficient fixed-view object/tool evidence was still absent in those
samples. This diagnostic correction has not received another full-task test.

## Revised pregrasp feedback v2 (opt-in)

`--speed-first-v3 --pregrasp-feedback` now uses independently available external
and wrist views. Each reference view can be null; at least one supplied view is
required. A wrist `tool_reference_mode="fixed_wrist"` retains a currently visible
finger reference at fixed opening. It checks available texture and expires after
12 distinct observations without a successful check, increasing uncertainty in
the meantime. It cannot initialize an unseen contact point.

Adaptive 5/7/11 pixel texture supports remain inside the selected body ROI. Nearby
visible same-body supports may transport an initially visible contact point using
a bounded local affine approximation. Such transported points are labeled local
inferences. Partial-view response fits remain independent; combining them uses
only commonly excited robot-motion directions, then finds observable image
directions. Missing measurements are never interpreted as zero response. If one
view loses tracking, another valid view can continue. Credible fixed-camera target
motion invalidates both stationary-target models; wrist-only feedback cannot
always distinguish target movement from robot-induced image motion.

The revised module preserves the original Field planner and action budgets,
ordinary Direct5, <=6 mm total nominal-goal revision and normal-control fallback.
It adds no probes, waiting actions, closure gates or dedicated model calls.
Dropping a view starts a new comparable error baseline instead of recording a
smaller partial error as alignment improvement. No image estimate certifies a
grasp, clearance or LIBERO success.

Offline evidence is in `runs/pregrasp_feedback_v2_preparation_001/`: 228 tests
passed, as did disabled and enabled-null saved compatibility replays (each 705
actions and 59 responses). Four historical RGB clips initialized successfully,
but only four transitions tracked and no response model validated. This establishes
neither reliable sustained real tracking nor a grasp improvement.

The authorized original-drawer zero-shot test in
`runs/pregrasp_feedback_v2_drawer_zero_shot_001/` ended by explicit user stop:
836 actions, 61 responses, raw LIBERO success false. Control time excluding video
was 1,035.03 seconds, an incomplete duration rather than completion latency.
All 61 pregrasp references were null and no `align` phase was selected, so no new
tracking or corrections ran. The model described rim-alignment movements under
`approach`; existing held-object alignment and appended pregrasp alignment wording
may contribute to this routing gap, but causation is unproven. The robot opened
the drawer and later carried the bowl; the drawer retracted, leading to temporary
bowl staging and another drawer-opening sequence. At stop, the bowl was on the
table and the drawer was open. See the run's `analysis.json`, `trial/summary.json`
and `trial/episode.mp4`. No speedup or correction benefit is established.

Manual offline reference annotations were never supplied to the live policy.
Astra medium subscription-only, 512 RGB/state8, 1 mm grid, raw LIBERO completion,
collision record-only and video-excluded timing were preserved. Runtime sources
remained frozen; the trial and its private supervisor are stopped. Further
full-task tests require fresh user approval after explaining changes.

## Pregrasp feedback v1: historical offline implementation

`--speed-first-v3 --pregrasp-feedback` adds local **open-gripper contact
alignment** feedback. It is off by default and can be combined with
`--attempt-memory` and `--robot-geometry-guidance`. GPT still selects the contact
strategy, orientation, target and Field/Direct controller. The optional
`control_context.pregrasp_reference` describes visible material contact points,
their textured body ROIs and the desired tool-minus-target image offset in both
RGB views. A null reference keeps the normal controller behavior.

Frozen texture supports track those model-selected points; they do not establish
semantic correctness. Natural measured end-effector translations identify a local
image response at fixed orientation and finger opening. Corrections are available
only in motion directions supported by both the observations and subsequent
prediction checks. A correction revises the existing Field XYZ goal by at most
6 mm **total from its original nominal goal**, within a 40 mm local neighborhood.
Repeated updates do not accumulate that allowance. The existing distance planner
and measured-pose path follower execute the revised goal within the original
Field action budget. Ordinary Direct action arrays remain unchanged.

Target motion in the external view, rotation, finger-width change, occlusion,
ambiguous texture or insufficient motion disables the optional correction. If a
corrected Field goal was active, execution returns to the model's nominal goal.
The controller does not add probes, holds, automatic closure or model calls. A
bounded current feedback message preserves measured image-error changes and the
last epoch's invalidation reason; pixel convergence does not certify a grasp or
task success. Depth, camera calibration and simulator object poses are excluded.

Inspect the opt-in payload without starting inference or the simulator:

```bash
.venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py \
  --speed-first-v3 --attempt-memory --pregrasp-feedback --preflight
```

Offline validation passed 210 distinct tests, including a synthetic RGB-to-Field
correction. Two saved replays each preserved 705 actions and 59 responses: exact
legacy payloads with the feature disabled, and unchanged actions/feedback with
explicitly null references when enabled. No full-task trial has been started.

The real-image diagnostic is a limitation, not a positive tracking result:
**0 of 4 reviewed historical pregrasp clips supplied usable contact seeds**.
Three visible material pads were too small for contained texture supports; the
fourth was occluded. Manual annotations are diagnostic fixtures, never runtime
inputs or demonstrations. Native-512 synthetic active updates took about 44 ms
median, excluding image construction, while enabled-null requests added about
5,875 characters on average. These measurements establish neither real grasp
improvement nor whole-task speedup. See
`runs/pregrasp_feedback_offline_001/analysis.json` for evidence and limitations.

## Speed-first v2: explicit local operations (opt-in)

`--speed-first-v2` keeps the early distance planner and pose follower, with Astra
medium through the ChatGPT subscription. It restores the full grasp-settling and
placement guidance, permits explicit free-space Field recovery waypoints, bounds
request history, and adds an explicitly selected local gripper operation. Ordinary
Direct remains five unchanged model actions. A `gripper_operation` response instead
sets `direct=null` and requests `open` or `close`; local feedback holds the measured
start pose and repeats the intent for 10–30 open actions or 15–30 close actions.
Observed finger motion followed by four stable feedback transitions can end the
operation. Settling is not a grasp, release or task-success certificate.

Field output quaternions with norm error at most 0.01 are normalized, with raw and
effective values logged; larger errors remain invalid. Input history retains four
recent observation blocks and two current/recent RGB pairs plus a bounded local
digest of older decisions and feedback. Complete raw logs remain local. These
changes are opt-in and do not alter the old profile or frozen experiments.

Run `--speed-first-v2 --preflight` to inspect the protocol without inference.
Live evaluation still completes only on raw LIBERO success, records collisions
separately, has no automatic episode cap, and subtracts measured video overhead.
Operation actions are included in `contact_direct_actions` for compatibility and
also reported separately as `local_gripper_actions`; this is a different controller
contract from pure Direct5. No performance improvement is assumed before testing.

## Speed-first Distance Hybrid (opt-in)

`--speed-first` restores the small controller architecture used by the earlier
successful original-bowl distance experiment. It is a separate profile; V36 and
the historical results remain available. This version is implemented and checked
offline, **not yet evaluated in a fresh model-driven rollout**.

GPT selects nearby end-effector waypoints and routes between local navigation
and interaction. Local navigation uses cached 26-neighbor Dijkstra cost-to-go on
a 33×33×33 grid at 5 mm spacing, then measured-pose path tracking. One local path
has at most ten 1 cm segments and is reconsidered within ten control actions.
GPT refreshes within 30 Field actions, earlier on measured arrival, invalid path,
boundary or stall. Direct executes five unchanged model actions for interaction
and recovery, unless an external episode stop occurs first. A Field hold returns
to GPT after five actions. Arrival is a local waypoint event, not task completion.

The profile disables local pixel objectives, visual-reference qualification and
automatic upward reobservation. `visual_reference` and `goal_reference` must be
null; the model chooses end-effector goals directly from the current images and
state, including the held-object offset where relevant. It does not inherit the
V36 execution guards. Model-provided blocked/unknown boxes still exclude paths;
nonnegative Gaussian repulsors provide soft obstacle costs. Neither constitutes
measured 3D clearance. The task definitions select language and benchmark IDs,
not fixed grasp poses or prerecorded trajectories.

The preset uses **zero-shot, gpt-6-astra, medium effort, subscription OAuth only**,
two RGB views and the original 8D robot state, with feedback block summaries.
It accepts both registered tasks (`bowl_plate`, `drawer_bowl`) and the existing
explicit custom-layout/scene arguments. Raw LIBERO success alone marks completion.
Collision monitoring records secondary evaluator-only metrics and continues after
contact; it does not feed contacts to the policy. Existing strict collision modes
retain their original stop-and-disqualify behavior. Attempts are allowed, but
gripper command counts are not claimed to be a validated attempt counter.

The primary timing field is `control_without_video_s`, subtracting measured
recording overhead only. It is a completion latency only when
`completion_time_eligible` is true. There is no automatic episode time/action cap.
An operator can create `operator_stop.request.json` in the new run directory to
stop at a completed action/feedback boundary and save an incomplete result.

The speed-first client retries only `server_is_overloaded`, at most twice for
the same observation. Backoff is 2 then 4 seconds plus up to 1 second of jitter;
an HTTP `Retry-After` can lengthen either wait. If that wait exceeds 60 seconds
or an existing request budget would be exceeded, it stops instead. Other errors
remain terminal. Every attempt is logged and counted, and waits stay in control
time. A failed response never advances physics or enters model history. No model
or API-key fallback is used. Other experiment profiles retain their old behavior.

Entry point (starts a new live trial; implementation validation does not run it):

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py \
  --speed-first --task bowl_plate --output runs/speed_first_bowl_zero_shot_001
```

The offline regression tool replays saved model outputs and observations, and
compares each newly computed action **before** supplying the next archived
observation. It makes no model or simulator calls:

```bash
PYTHONPATH=src .venv/bin/python scripts/replay_speed_first_baseline.py \
  --output runs/speed_first_replay_check_001.json
```

See [implementation](src/gpt_field/speed_first_hybrid.py),
[offline replay evidence](runs/speed_first_implementation_001/baseline_replay_final.json),
and [earlier real successful run](runs/distance_field_medium_task_001/summary.json).
The historical success and exact replay do not establish success or speedup with
the new prompt in a fresh trial.

## Historical visual-field development

The corrected V16 original-bowl zero-shot test was **incomplete**.
It recorded24 complete transitions and17 effective Field actions, with7 medium
subscription calls, no grasp, LIBERO success or forbidden contact. The operator
stopped reviewed repeated coverage interruptions; video-excluded attempt time
was210.728s. The shared inventory included surrounding objects, but partial
cabinet visibility repeatedly prevented reuse despite reliable texture tracking.
This is not a successful completion time or evidence of speedup. Other scenes
and Direct were not rerun. See [result and diagnosis](runs/visual_field_v16_zero_shot_002/analysis.json),
[audit](runs/visual_field_v16_zero_shot_002/independent_audit.json) and
[video](runs/visual_field_v16_zero_shot_002/trials/bowl_original_zero_hybrid_v16/episode.mp4).
The earlier image-boundary numerical-defect attempt is separately preserved in
[runs/visual_field_v16_zero_shot_001](runs/visual_field_v16_zero_shot_001/analysis.json).

The V15 original-bowl zero-shot test **failed on forbidden contact**:
the left finger touched the neighboring ramekin at action235. It executed236
actions (232 effective Field), with15 medium subscription calls,0 API-key calls
and347.939s excluding video. No grasp or LIBERO success occurred. Motion
diagnostics helped repair a reference and sustain approach motion, but all15
predictions omitted neighboring obstacles; only target-bowl geometry entered
the planner. This attempt does not establish completed-task acceleration.
All sources stayed frozen, cleanup passed, and no other scene or Direct trial
was started. See [result and diagnosis](runs/visual_field_v15_zero_shot_001/analysis.json),
[audit](runs/visual_field_v15_zero_shot_001/independent_audit.json) and
[video](runs/visual_field_v15_zero_shot_001/trials/bowl_original_zero_hybrid_v15/episode.mp4).

The V14 original-bowl zero-shot test is **incomplete**: no grasp, LIBERO
success or forbidden collision. It executed68 effective Field actions and took
327.748s excluding video, with13 medium subscription requests (11 valid decisions
and2 semantic repairs). Changed-direction recovery ran, but reference revisions
consumed separate experiment slots for the same direction and later candidate
feedback disagreed with the remaining execution permissions. The operator stopped
reviewed recovery stagnation; no other scene or Direct trial was started.
See [result and diagnosis](runs/visual_field_v14_zero_shot_001/analysis.json),
[audit](runs/visual_field_v14_zero_shot_001/independent_audit.json) and
[video](runs/visual_field_v14_zero_shot_001/trials/bowl_original_zero_hybrid_v14/episode.mp4).
The attempt does not establish completed-task acceleration.

V13 original-bowl zero-shot test is stopped and **incomplete**. The corrected
attempt executed20 effective Field actions, then reached a reviewed visual
admission deadlock; no grasp, LIBERO success or forbidden contact occurred.
Video-excluded elapsed time was196.869s across7 medium subscription requests.
This is not a task-completion time or evidence of speedup. Other scenes and
Direct were not rerun. See [result and diagnosis](runs/visual_field_v13_zero_shot_002/analysis.json),
[complete audit](runs/visual_field_v13_zero_shot_002/independent_audit.json) and
[video](runs/visual_field_v13_zero_shot_002/trials/bowl_original_zero_hybrid_v13/episode.mp4).
The earlier grid-reconstruction defect attempt remains preserved in
[runs/visual_field_v13_zero_shot_001](runs/visual_field_v13_zero_shot_001/analysis.json).

Test Astra-generated spatial fields as interfaces for local robot planning. The primary field contains unitless lower-is-better potentials. Metric distance fidelity has a separate command and reference format. Read [the implementation plan](IMPLEMENTATION_PLAN.md) for hypotheses, baselines, and evaluation gates.

The opt-in `--method-version v13` is prepared for an original-bowl zero-shot
Hybrid trial. It adds primary-target geometry independently of obstacle lists,
finger-only contact permission that never exempts the palm, and optional
two-view contact-to-pinch alignment for bounded local Field correction.
The [V13 plan](runs/visual_field_v13_zero_shot_001/plan.json) retains Astra medium
through the subscription, RGB and measured robot inputs, actual LIBERO success
without forbidden contact, and video-excluded control timing. It has not been
launched; validation and a final source/helper seal precede the single private
supervised trial. No task-performance or speedup result is claimed.

The opt-in `--method-version v12` visual Field variant learns the local relation
between commanded and measured robot translation, accumulates causal visual
response over resolvable displacement, and uses observed tool/neighbor motion
to constrain finite exploration through uncertain image overlap. Metric Field
planning retains model obstacle hypotheses and known robot geometry. It also
uses a shorter protocol and exact duplicate-feedback references. These changes
address V11's command-scale and admission deadlock; they do not establish
collision safety, task success, or faster completed tasks. The fresh original-bowl
zero-shot test is tracked in [the V12 plan](runs/visual_field_v12_zero_shot_001/plan.json).

That [V12 original-bowl test](runs/visual_field_v12_zero_shot_001/analysis.json)
ended in a forbidden palm-to-target-bowl contact at action704, with LIBERO false.
It executed616 effective Field actions and took742.696s excluding video.
Continuous Field reuse worked, but repeated grasp recovery and missing target
contact geometry prevented completion. No later scene was run and no successful
task speedup is established.

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

An opt-in one-demonstration retry adds `--demonstration path/to/reference.json`
to that command. Prepare the bundle with `scripts/prepare_libero_demonstration.py`
from a single official HDF5 episode. It retains all normalized actions and
post-action robot states, with 24 deterministic paired-camera keyframes (or all
frames for shorter episodes). No object poses, articulation state, rewards or
completion flags enter the reference. The one-demo prompt explicitly distinguishes
the dataset's post-action observations from current live observations.

The same immutable reference is included once per model request, outside the
live history's two-image-pair retention rule. Its actions are never automatically
executed. The run freezes `demonstration.json` and records hashes, source episode,
preprocessing, selected frame indices and a separate prompt version. Existing
no-demonstration runs remain the default. Dataset preparation/download time is
outside the evaluation control loop; model processing of the reference is included
in the measured control time.

For a matched pure Direct retry, use the same reference with
`--task drawer_bowl --demonstration path/to/reference.json --controller-mode direct`.
This selects medium effort, fixed five-action execution, block feedback summaries
and the same drawer/demo operational guidance. It disables field generation and
planning and still stops only on actual LIBERO success (or user/transport failure),
without an episode cap. The raw model response includes actions, target region,
observable routing evidence and control context; only its five actions are
executed. `scripts/audit_demo_direct_control.py` audits this arm and can compare
its frozen reference and initial observation against a recorded Hybrid run.

The same one-demonstration comparison supports the original `bowl_plate` task
(LIBERO-Spatial task 0). Prepare its matching HDF5 episode using
`scripts/prepare_libero_demonstration.py --task bowl_plate`, then use the identical
bundle for both fresh runs:

```bash
PYTHONPATH=src .venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py --task bowl_plate --demonstration data/demo_references/bowl_one_demo_001.json --distance-field-control --output runs/bowl_distance_medium_one_demo_001
PYTHONPATH=src .venv/bin/python scripts/evaluate_bowl_hybrid_subscription.py --task bowl_plate --demonstration data/demo_references/bowl_one_demo_001.json --controller-mode direct --output runs/bowl_direct_medium_one_demo_001
```

Both use initial state 0, seed 7, medium subscription inference and actual LIBERO
success stopping. A task-mismatched reference is rejected before execution. Bowl
Direct without `--demonstration` retains its historical behavior and defaults.

`--episode-index N` selects one of the 50 official initial states (default 0).
For a custom rearrangement of the same task objects, pass
`--initial-state-override path/to/layout.npy` to both arms. This is an evaluator-only
finite flat MuJoCo state, derived and validated for the same task model. The runner
validates it before authentication, freezes its exact bytes into the run directory,
and records its hash separately from the base official state and BDDL hashes.
The task language and success predicate stay unchanged, and the custom state is
never included in model input. Reports label this as a custom-layout evaluation,
not an official initial-state benchmark. Keep the demonstration unchanged and
verify identical reset observations and layout hashes for the paired runs.

To replace distractor assets as well as positions, also pass
`--task-bddl-override path/to/task.bddl`. It requires a matching custom initial
state built for that model (map joints by name when preparing it; do not reuse a
flat state solely because its length matches). The worker validates that the
original task language, goal and target asset classes are preserved, then checks
the state against the custom model. The runner archives both custom and original
BDDL files before the first model request. Audits compare the exact scene and
state hashes across arms. These runs are labeled `custom_scene`; scene definitions
and hidden object coordinates remain evaluator-only. Validate initial stability,
initial failure of the goal predicate, and identical resets before live tests.

For the registered ketchup/milk bowl obstacle scene, add
`--zero-shot-collision-free` with the custom BDDL/state and omit `--demonstration`.
Use `--distance-field-control` for Hybrid or `--controller-mode direct` for the
matched five-action Direct arm. This opt-in protocol explicitly requests zero-shot
control and avoidance of unintended contacts in both prompts, defaults to Astra
medium with feedback summaries, and rejects any demonstration bundle.

The evaluator checks MuJoCo contact geometry at every physics substep and the
final control endpoint. It stops on the first forbidden robot/scene or held-bowl/
scene contact and records a scored `collision_failure`, even if the original
LIBERO goal is also true. Named finger/bowl grasp and bowl/table or bowl/plate
support contacts are allowed. Robot-internal contacts and static scene resting
contacts are outside this monitor's scope; it is not a full swept-volume guarantee.
Raw `libero_success`, qualified `task_success`, contact evidence and first action/
substep are stored separately in the summary and `collision.json`. Hidden contact
metadata never enters model input. Initial forbidden contacts invalidate setup
before any model request. Only collision-free successes qualify for task-completion
time comparisons; failed attempts retain their measured costs for diagnostics.

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

## Opt-in RGB visual field v2

`scripts/evaluate_visual_field_subscription.py --method-version v2` selects the
new zero-shot Hybrid controller. The default remains v1; earlier 3D distance
controllers and their recorded runs are unchanged. V2 consumes two RGB views,
the measured EEF/finger state, and measured arm joints. It receives no depth,
camera calibration, object poses, contacts, or evaluator success information.

The model supplies a physical `target_anchor_uv` separately from the virtual
`target_point_uv`. Local feature banks track the physical landmark and preserve
the desired image offset. Each view has a weighted image-distance field with
predicted blocked/unknown image regions. These fields have pixel/cost units,
not metric 3D clearance. Action-to-measured-EEF and EEF-to-relative-image models
are fitted online; independent response checks permit control only in supported
directions. Joint-response predictions reject local joint-limit violations.
Missing evidence requests a model-directed short observation probe. The local
controller never invents an unobserved clear direction.

Payload transport requires observable object/tool co-motion. Loss/reseeding of
the attachment landmarks invalidates old evidence. Post-release alignment,
separation and stability are reported as uncertain image evidence, never task
success. Repeated uninformative holds trigger a recovery request, not an episode
timeout. Only a persistently stalled, entirely zero Direct chunk may have its
unused zero suffix skipped; moving and gripper-contact Direct chunks remain
five unchanged model actions. Model history retains two recent complete turns
plus a deterministic summary; complete raw records remain on disk.

The v2 evaluation was prepared for four fresh zero-shot Hybrid trials:

```bash
.venv/bin/python scripts/prepare_visual_field_v2_trials.py
.venv/bin/python scripts/run_visual_field_v2_trials.py
```

The preparation freezes the original bowl, relocated bowl, obstacle bowl and
original drawer inputs, with exact initial RGB/state checks. The queue freezes
source before execution. It uses Astra medium through subscription OAuth only,
without an episode cap. Actual LIBERO success marks completion; forbidden
contact terminates a failed trial. User interruption or infrastructure errors
pause the queue. Older Direct results are archived references, not a fresh
matched comparison. Reported time excludes measured video overhead and includes
local tracking/planning. Reports distinguish accepted local visual descent from
bootstrap, rejected Field proposals, and attachment-gated holds.

That queue was subsequently canceled by the user. The original and relocated
bowl attempts remain `user_stop`; obstacle and drawer trials were not started.
The commands above describe the historical workflow, not a pending continuation.

## Opt-in RGB visual field v3

V3 implements feasibility checks and local evidence acquisition after the v2
original-bowl failure. It is selected with `--method-version v3` in
`scripts/evaluate_visual_field_subscription.py`; the default is still v1.
Only zero-shot Hybrid is enabled for v3. Its subsequent original-bowl live test
stalled without LIBERO completion: all 200 actions were Direct and none Field.
See `runs/visual_field_v3_zero_shot_001/analysis.json`; no completion-time
improvement was established.

Before using a field, the controller checks the moving footprint, start, goal
and typed image regions. The model supplies the observed asymmetric footprint
instead of an isotropic radius. A footprint extending beyond the image remains
an explicit coverage problem; it is never silently clipped. Obstacle regions,
intended contact regions, and visibility uncertainty have distinct semantics.
An intended contact goal can be represented, but approaching contact hands off
to Direct. Self-occlusion is scoped to the visible moving body; hidden background
remains unknown. Semantic contradictions request a reference/goal correction,
not repeated motion-response probes.

Per-view landmark and response models persist independently. A missing patch
on a distant obstacle no longer disables every candidate in both views. Its
last bounds and uncertainty remain available to candidate-local checks; unknown
regions do not need a textured interior and are not treated as free space.
When only one view supports a direction, restricted local commands carry
explicit missing-view uncertainty. These are empirical image checks, not a
metric obstacle map or a 3D collision certificate.

A model-issued `observation_intent` can authorize several bounded local probes
under the same Field intent. It specifies actual normalized translation
commands, per-view image corridors, a step budget (at most 10), and a measured
EEF displacement budget (at most 2 cm). The local controller must also have
measured predictive support and pass the candidate checks. Unsupported initial
directions still require explicit model-generated Direct5 actions. These local
budgets request a fresh observation/decision when exhausted; they do not stop
the episode.

Persistent recovery memory distinguishes comparable image-goal descent,
visibility changes and response validation from merely moving the robot or
renaming a hypothesis. Transport still requires payload/tool co-motion evidence.
Only the evaluator's actual LIBERO success completes a live trial; the existing
forbidden-contact failure rule remains. The runner separately counts accepted
local descent, local information probes, rejected Field actions, and attachment
holds, and excludes video overhead from control time. Astra medium, subscription
OAuth only, RGB plus measured robot state/joints, and the ban on supplied
depth/calibration/object truth are unchanged.

The offline audit is available through `scripts/audit_visual_field_v3.py`. Give
it a fresh `--output` directory; it only reads an archived v2 run and compares
its preserved references with the v3 preflight. Legacy radii become unclipped
square footprints, all old blocked regions retain obstacle roles, and unknown
regions remain unknown. This conservative conversion does not infer new contact
permissions or reconstruct a successful rollout. The implementation validation
is recorded in `runs/visual_field_v3_offline_validation_001/validation.json`.

## Opt-in metric hypothesis field v4

V4 addresses the v3 run in which two-view footprint/response prerequisites kept
every action in Direct mode. Select `--method-version v4` in
`scripts/evaluate_visual_field_subscription.py`. This is zero-shot Hybrid only;
v1–v3 and their frozen results remain available. The model remains Astra medium
over subscription OAuth, with no API-key fallback. Inputs remain two RGB views,
EEF/finger state, and measured arm joints. No supplied depth, camera calibration,
object poses, contact identities or evaluator outcomes enter the policy.

GPT supplies a **bounded metric hypothesis**: a relative local goal, predicted
obstacle and unknown boxes, tool/payload radius, position uncertainty, command
limit and action/travel limits. These quantities are visual estimates, not
reconstructed or measured environment geometry. Each box is inflated by the
tool radius plus global and per-box uncertainty. All inflated obstacle and
unknown boxes remain hard exclusions throughout that intent, including checks
on segments thinner than one grid cell. An invalid start or goal is reported;
the planner never carves a route through a box to make an intent executable.

The local planner solves a 26-neighbor graph on a 33×33×33 grid at 5 mm spacing.
The field value is shortest graph path length to the goal region, in meters,
within the hypothesized traversable space. It is a goal-distance field, not an
SDF to an observed surface. Diagonal moves check interpolation corners. The
distance grid is cached, a path of up to ten segments is extracted, and a
bounded look-ahead controller follows it from the **measured** EEF pose. It
replans locally when the path horizon ends. Inference is reused for the supplied
10–60 action budget (prompt default 30), subject to arrival, tracking/observation
contradiction, measured stall, and travel/domain limits. No automatic episode
timeout or model-completion stop is introduced.

External RGB tracks the controlled point and a real target anchor. A bounded
initial Field motion does not require a prelearned camera Jacobian: commands
start at at most 5 mm, with at most 15 mm of unvalidated observed travel.
Actual measured motion must produce a consistent feature response; a stationary
or lost feature requests a new observation. Held-out response checks are
required before relaxing this initial command limit. A validated planar response
can adjust an alignment goal by at most 3 mm at once and 10 mm cumulatively;
the planner rechecks the corrected goal against the same geometry and budgets.
These numerical checks do not prove semantic identity or physical clearance.

Wrist RGB tracks a target bearing; it does not require an image of the whole
gripper or a second learned tool-response model to permit every Field command.
The bearing is reported for the next model decision. It is not silently used
as a metric pose estimate or a collision certificate. Intended contact, grasp,
release, drawer interaction, and larger orientation changes remain unchanged
Direct5 actions. Field emits zero rotation and zero incremental gripper motion.
Payload transport requires independently tracked tool/payload co-motion;
changing the destination preserves this evidence only if payload/tool tracking
and robot-state continuity remain valid. Opposing motion or release invalidates
it. Only actual evaluator LIBERO success with no forbidden contact completes a
live trial; video overhead remains excluded from its reported control time.

Validation records distinguish **effective moving Field actions**, raw selected
actions, requests per measured centimeter, and local processing time. The
offline audit in `scripts/audit_visual_field_v4.py` uses a deterministic feedback
fixture plus the prior run's allowlisted real RGB/state/actions. Fake inference
counts demonstrate reuse, not a measured GPT latency reduction. Archived frames
can diagnose tracking and branch eligibility; they cannot establish a
counterfactual successful rollout or collision-free motion. A new completed
live task is needed to establish actual completion-time improvement.

The v4 implementation check passed 459 relevant tests. Its frozen audit and
validation manifest are in `runs/visual_field_v4_offline_validation_001/`.
In the synthetic matched-feedback fixture, 20 moving Field actions cover
4.375 cm from one fake model response; identical Direct5 commands require four
fake responses. No new live trial or real-model latency measurement is included.

The subsequent live zero-shot original-bowl attempt is recorded separately in
`runs/visual_field_v4_zero_shot_001/analysis.json`. It ended incomplete after
115 actions and 221.79 seconds excluding video. There were 105 effective Field
actions, including two 30-action reused blocks, but no grasp or LIBERO success.
The tenth model response contained reversed obstacle bounds and failed schema
validation. Independent auditing also found near-pi rotation-vector subtraction
resets in the visual monitor, affecting evidence and bootstrap accounting.
Tested sources were preserved; other scenes were not started. This is evidence
of live Field participation, not a valid completed-task speed comparison.

## Opt-in V5: sustained Field motion and contact handoff

The repaired V4 attempt (`runs/visual_field_v4_zero_shot_002/analysis.json`)
executed 553 moving Field actions but never grasped. Its 854.63 seconds excluding
video were dominated by model requests. Near the target, repeated loss of a
single external feature caused reseeding and retreats. V5 is available through
`--method-version v5` in the same runner, with unchanged zero-shot, medium OAuth,
robot/RGB inputs, independent LIBERO/contact evaluation and timing rules.

V5 tracks a fixed bundle of local image corners around each declared landmark,
optionally inside a model-declared ROI. Forward/backward matching, bounded
similarity consensus and original appearance checks reject inconsistent tracks.
It does not introduce new corners after initialization or turn image regions
into metric obstacles. Motion validation uses the estimated image noise and
requires held-out predictions to outperform a static-image baseline. Bootstrap
travel stays finite; static background cannot authorize indefinite Field reuse.

The model expresses orientation as `hold`; local code uses the measured robot
orientation. Field can reuse an intent for up to 120 actions (prompt default80),
with useful clear-space command limits rather than many tiny intents. The
existing 33³/5mm distance grid, spatial bounds, inflated obstacle/unknown boxes,
segment checks, arrival and measured-motion checks remain. Normal action-budget
boundaries are progress refreshes, not reported grasp or recovery failures.
For a predicted-clear straight route, sustained image-goal divergence triggers
a fresh model decision; the check is not applied to a required obstacle detour.

Terminal approach and contact alignment are explicitly assigned to Direct5.
An unavailable Field landmark does not automatically cause retreat or closure:
GPT chooses an interaction or a changed observation from the live views. Grasp,
release and drawer actions remain exactly five model-generated actions. Small
planar corrections are also available during approach when the same existing
response, height, orientation and geometry conditions hold.

A parsed response rejected by local semantic validation can receive one explicit
same-observation correction request. Its rejected JSON and validator message
are identified as rejected model output; no invalid action executes. Both calls
remain logged, timed and counted. Authentication, model/effort, transport and
budget failures never use this retry or any fallback credentials/model.

Visual identity roles are distinct. In a Direct grasp/test-lift response, the
top-level target and wrist target identify the interaction object, while the
external target may identify a different stationary reference. Payload and
independent tool tracks can then test co-motion against that reference. In a
Field response, the external target must again match the navigation target so
image-goal corrections cannot use an unrelated anchor. A payload is never its
own stationary reference, and naming it does not certify attachment.

The first V5 live attempt (`runs/visual_field_v5_zero_shot_001/analysis.json`)
ended incomplete after 207 actions and 512.76 seconds excluding video, without
LIBERO success or detected forbidden contact. It reached grasp interaction, but
the original protocol incorrectly required the independent stationary reference
to be the interaction object too. A bounded repair could not resolve those
contradictory roles. That frozen result is retained; the corrected role contract
requires a fresh test. Terminal failed-decision time is now separately reported
within total decision/control time, rather than omitted from component timings.

The repaired retry (`runs/visual_field_v5_zero_shot_002/analysis.json`) reached
grasp, Field transport and release without invalid returned model outputs or
monitored forbidden contact. It still did not satisfy LIBERO, and was stopped
after persistent post-release task stagnation: 423 actions, 192 effective Field
actions, 973.45 seconds excluding video. An evaluator-only replay matched all
424 recorded dual-camera pairs exactly and all robot state/joint values, with
zero inference calls. The bowl touched the plate, but their final horizontal
body-center separation was 4.947 cm; this checkout's `On` predicate requires
strictly less than 3 cm. Repeated visual claims of stable support did not fix
the centering error. Field alignment ended at 2.148 cm horizontal error while
the bowl was still unsupported. Direct placement then passed through 0.766 cm
error before overshooting to 4.807 cm before release; terminal corrections and
completion verification therefore need joint review. Fragile per-frame attachment thresholds and model landmark
refreshes also interrupted transport. This remains an incomplete task result,
not evidence of faster successful completion; no other scene was launched.

Archived replay and implementation checks are recorded separately under
`runs/visual_field_v5_design_001/`. They are not counterfactual task successes or
measured completion-time improvements. Fresh successful task runs and a matched
Direct comparison are still required to establish a speed advantage.

## Opt-in V6: preserve alignment and reduce repeated contact decisions

`--method-version v6` retains the same zero-shot RGB/robot interface and medium
subscription transport. The model explicitly seeds or retains visual references.
Retaining a reliable reference reuses its tracked coordinates rather than a new
coordinate guess. Attachment evidence uses several transitions of independent
payload/tool motion relative to a stationary reference. Uncertain evidence is
not certified attachment: travel is bounded, while observed slip, opening,
identity changes and actual track loss invalidate it.

Field intents select `xyz`, `xy` or `z` motion. Horizontal alignment can therefore
precede a vertical approach that preserves the starting horizontal position.
The 5 mm distance grid, obstacle checks and measured-pose feedback remain; locked
axis drift beyond 3 mm ends local execution. Neither proximity nor image overlap
causes automatic opening or establishes task success.

A third, explicit `gripper` controller settles a model-selected open/close command
locally using finger measurements. It sends zero pose commands, checks pose drift,
and stops on settling or its finite 10–30 action budget. Settled fingers do not
prove a grasp. Other contact/reorientation actions still use model-generated
Direct5. Direct release commands cannot translate: translational placement
approach uses Field, followed by an explicit gripper decision.

Implementation and archived RGB checks are separate from live task results.
The first original-bowl trial (`runs/visual_field_v6_zero_shot_001/analysis.json`)
ended incomplete after 383 actions: 338 effective moving Field actions, 5 Direct
actions and 34 local gripper actions. The 26th subscription request ended without
its terminal completion event (`stream_missing_completion`). LIBERO remained
false and no monitored forbidden contact occurred; video-excluded attempt time
was 414.33 seconds. The first grasp was unsupported by its lift test, and recovery
had started. Early tracked features were on the stationary robot base rather than
the gripper; after reseeding, Field blocks reached 32 actions (31 effective moves).
Each explicit close/open block settled in 17 actions. These demonstrate executed
components, not successful task speedup. No other scene or retry was launched.
Timing includes inference, perception, planning and physics, excluding video.

## Opt-in V7: actionable correspondence repair

The V6 RGB audit found that early "gripper" tracks actually followed a fixed
robot-base structure. A small historical image excursion kept the reference
apparently responsive, while repeated model intents renewed short unvalidated
movement. V7 checks recent controlled-feature motion and carries unvalidated
travel across intents and reuse of the same local appearance. A new name or
nearly identical seed cannot erase a rejected correspondence's evidence.

When repair is needed, an additional derived copy of current external RGB marks
the tracker's moving/anchor/tool claims. Original external and wrist images are
preserved. The labels are estimates, not segmentation or semantic truth; healthy
references incur no extra image. The model can select a different visible moving
part rather than repeatedly issuing the same rejected local motion.

Generic contact guidance distinguishes navigation-target centering and descent
stall from graspable-feature engagement. Field handles bulk motion and constrained
placement; the model can use Direct5 for a necessary terminal contact correction,
then explicit local gripper settling. A tiny inconclusive attachment test is
distinct from an observed grasp miss. No task-specific positions are introduced.

V7 allows one missing-completion stream retry at the same observation before any
action. It shares an overall maximum of three attempts with the single semantic
repair allowance. Every attempt remains logged, counted and timed. Other
transport, authorization, quota or model/effort errors do not use this retry, and
there is no API-key or model fallback. Fresh evaluation uses
`--method-version v7`; archived validation alone does not establish task speedup.

The fresh original-bowl zero-shot V7 trial ended in **collision failure**:
`runs/visual_field_v7_zero_shot_001/analysis.json`. It ran 388 actions without
closing the gripper; action 387 (zero-based) contacted the neighboring ramekin
with the right finger during Direct yaw recovery. LIBERO remained false.
Video-excluded attempt duration was 731.7144 s, including 689.1016 s in the model
pipeline and 7.4907 s in local planning. There were 46 subscription requests,
45 valid decisions, and one recovered missing-completion stream. The 323 Field
selections included 310 effective moving actions; 65 Direct actions were used
for reorientation. Real approach progress occurred, but repeated reference and
orientation repair prevented transition into grasp. All 69 frozen-source checks
and initial-observation checks passed. No later scene was launched. This failed
attempt is not a successful completion time or evidence of task speedup.

## Opt-in V8: retain useful evidence and hold measured orientation

V8 separates weak visual response from a distinguishable prediction error.
Weak motion below image noise consumes the same finite correspondence-specific
travel allowance, but does not itself require a new feature or a yaw recovery.
Actual contradiction, tracking loss and exhausted uncertainty still stop reuse.
Archived RGB replay retains the useful V7 references at actions 339 and 367,
while still rejecting the V6 fixed-base reference after 6.59 mm of robot motion.

The installed OSC controller retains its old orientation goal on a zero rotation
command. V8 explicitly replaces that goal and computes bounded world-frame
rotation feedback toward the orientation measured at intent start. The existing
0.04 rad drift guard stays in place; each correction is at most 0.02 rad. A
logged 1e-9 rad refresh handles exact alignment and rejected-action holds.
Field path/axis constraints and model-commanded gripper settling are preserved.
In an isolated paired simulator probe, both yaw directions stopped the previous
zero-rotation Field after three actions; V8 ran all 40 bounded probe actions with
maximum drift below 0.014 rad and no monitored forbidden contact. This is robot
controller validation, not a task-completion timing result.

Every response also declares its motion purpose, a whole-tool external RGB
rectangle, and neighboring obstacle rectangles or an explicit uncertainty
assessment. Every listed Field neighbor must map to a metric obstacle/unknown
hypothesis, including known neighbors in an otherwise unobservable scene.
Unobservable Field scenes preserve goal/regions but cap measured travel at
0.015 m and each command at 0.005 m. Direct rotation solely for reference repair
is rejected. Other Direct rotations undergo a conservative projected-envelope
risk screen; accepted five-action chunks are unchanged, while rejected chunks
are discarded and a measured-current-pose hold is logged. The image screen
cannot certify 3D clearance or detect objects omitted by the model.

Use `--method-version v8` for a fresh zero-shot Hybrid trial. The original-bowl
plan is `runs/visual_field_v8_zero_shot_001/plan.json`; there is no queued scene
continuation. Successful-task speed improvement remains to be demonstrated.

The V8 original-bowl trial was operator-stopped after reviewed planning
stagnation, with no LIBERO success or monitored forbidden contact. It took
241.2079 s excluding video and made 13 subscription requests. All four Field
intents failed `invalid_measured_start` before graph search: uncertainty inflation
of the model's cabinet box enclosed the current EEF. The other 44 executed
actions were zero Direct holds; no grasp or effective Field motion occurred.
The nominal cabinet box did not contain the origin, but the first combined
tool/uncertainty inflation was 0.155 m. Changing goals did not resolve that
starting-state inconsistency. This exposes a geometry-representation deadlock,
not a measured task speedup. See `runs/visual_field_v8_zero_shot_001/analysis.json`.
The interruption preserved 48 execution records and 47 feedback records; source,
input and route checks passed, but complete transition auditing remains partial.

### Opt-in V9 uncertainty-cost planning

`--method-version v9` keeps the V8 measured-pose executor and visual monitors,
while changing explicitly localized visible object hypotheses to a hard nominal
box plus tool radius and a separate uncertainty cost. Dijkstra integrates
`1 + 8 * sum(exp(-0.5 * (distance_to_tool_core / sigma)^2))`, with
`sigma = intent_uncertainty_m + region_uncertainty_m`. The cost is not a calibrated
collision probability. Unknown regions retain their fully inflated hard masks;
missing current tool/object image support reverts to the strict field. Continuous
hard-box checks, 5 mm grid spacing, axis constraints and measured travel limits
remain active. This permits a model-risk route through uncertainty without
claiming verified free space or deleting obstacle evidence.

The fresh original-bowl zero-shot test is tracked under
`runs/visual_field_v9_zero_shot_001`. It must establish LIBERO success without
forbidden contact before any task-speed improvement can be claimed. A manual
`operator_stop.request.json` in a V9 trial directory requests a stop after the
current full transition; collision and LIBERO outcomes take precedence. It is
not an automatic episode limit.

V9 live original-bowl outcome is **incomplete**, with657 complete logged actions,
349 effective Field actions,57 valid responses and one unfinalized request. No
recorded LIBERO success or forbidden contact occurred. The requested operator
stop was not acknowledged before the execution session exited143; no raw summary
was finalized. The exact video-excluded total is unavailable. Returned model and
local-decision durations alone give a981.9263s lower bound. See
`runs/visual_field_v9_zero_shot_001/analysis.json`; the labeled reconstructed replay
uses only recorded RGB. Real Field motion resolved the prior geometry deadlock,
but low-motion attachment windows reduced commands until motion became too small
to verify, and repeated repairs moved the lift target. No task-speedup is claimed.

### Opt-in V10 persistent goals and adaptive evidence

`--method-version v10` retains V9's hypothesized geometry and weighted distance
planner. A local ledger stores each explicitly identified world goal across
Field, Direct and gripper repair. Measured robot motion may reach that goal
during repair; retaining it never adds a fresh relative displacement. Explicit
replacement changes a model hypothesis. Validated local visual corrections are
logged and bounded to 3mm per update and 10mm cumulatively per subgoal.

Independent payload/tool tracking now accumulates fresh displacement rather
than using four frames regardless of robot speed. Each supported co-motion
window is consumed once. Only that fresh support may estimate a finite 5--15mm
uncertain-travel allowance; it cannot certify a grasp, and lost references,
opening, finger changes or contradictory motion invalidate the evidence.
Below a 1mm effective command cap the executor hands off rather than generating
an indefinite microscopic-motion loop. Cold, indistinguishable image response
uses the same finite 15mm correspondence budget without being called a resolved
contradiction. Neither change supplies calibrated obstacle geometry.

The fresh original-only test is prepared in
`runs/visual_field_v10_zero_shot_001`, with private UNIX-socket supervision,
automatic restart disabled and atomic timing checkpoints. All model/perception/
planning/physics time counts; video recording is excluded. Historical Direct runs remain references,
not an automatically matched successful speedup denominator.

The V10 original-bowl test ended in **collision_failure** at action139: the right
finger contacted the neighboring ramekin before any grasp. All139 actions were
effective Field movement across9 subscription decisions, with218.1156s excluding
video. Persistent goals survived repairs, but the final z-only path crossed the
softened uncertainty halo of a model obstacle/tool-envelope hypothesis that did
not cover the real contact. Low tracking error did not prevent that geometry
failure. Full81-source/input/route/action auditing and cleanup passed; the saved
video has140frames at20Hz. See `runs/visual_field_v10_zero_shot_001/analysis.json`.
No other scene or Direct trial ran, and no completed-task speedup is claimed.

V11 adds known robot tool-part configuration obstacles, a tracked RGB
tool/neighbor translation guard, and deterministic feedback compression to the
persistent Field controller. Scene boxes and projected envelopes remain uncertain
model hypotheses; neither is a calibrated clearance certificate. Accepted Direct5
actions remain unchanged, and the allowed policy inputs are unchanged.

The frozen original-bowl zero-shot V11 test was stopped after reviewed execution
stagnation, **not completed**. It used9 medium subscription responses and24 complete
transitions:11 effective Field actions moved4.0778mm,8 Field commands were rejected,
and5 Direct commands were exactly zero. No grasp, LIBERO success or forbidden
contact occurred. Video-excluded attempt time was254.3576s, including239.2243s of
model time. Projected hotplate/cabinet overlap outlasted the finite cold separation
allowance before image response became predictive. Persistent envelope union also
impeded a proposed image-bound correction; final reseeding fixed the outside-view
condition but did not resolve the exhausted allowance and projection ambiguity.
All87 source/input/route/command checks passed, full transitions were preserved,
and private supervision was shut down. Other scenes and Direct were not run.
See `runs/visual_field_v11_zero_shot_001/analysis.json` and its independent audits.
This failed-to-complete attempt supplies no successful-task speedup measurement.
