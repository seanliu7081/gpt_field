# Astranomy Agentic drawer retry

Preparation only. **The full test requires the user's approval and has not started.**
All additions are in this folder. Existing `src/`, `scripts/`, configurations,
tests and historical runs are reused read-only.

The proposed single episode is **LIBERO-Goal task 3, initial state 0, seed 7**:
“open the top drawer and put the bowl inside.” It starts from a fresh reset,
checked against the existing historical initial-observation hash.

## Method

The [Astranomy report](https://research.nvidia.com/labs/srl/projects/astranomy/)
describes a persistent Codex agent with short absolute end-effector targets,
history, planning and measured motion feedback. Its final configuration disables
optional tools. This implementation follows that interaction pattern using
`gpt-6-astra`, medium reasoning, and the existing ChatGPT subscription OAuth login.

Each Codex turn receives two upright RGB images, synchronized robot state,
camera/gripper geometry and the previous action's measured error. Astra proposes
one absolute pose and a hold of 1–5 steps. The host acknowledges the proposal,
waits for the turn to complete, validates it, and then executes it. Failed turns,
duplicate proposals and malformed actions execute no motion. Actual feedback
returns with the next observation in the same thread.

The installed Codex build supplies extra tools by default. `model_catalog.json`
derives its Astra entry from the installed 0.153.4 bundled catalog, retaining the
model identity and context limits while selecting native function calls and
removing optional tool hints. `transport.py` additionally enforces a robot-only
tool list at the provider boundary and rejects any other returned tool call
before Codex receives it. It forwards subscription requests to the official
`https://api.openai.com/v1/responses` endpoint; it never uses an API key.

The loopback gateway is private and authenticated with a fresh local capability.
Only it receives the subscription token. Codex retains its thread in its normal
local session store; this code does not change the user's Codex configuration.
On OAuth token renewal, the adapter restarts its app-server and resumes the same
thread. Full text/image history is retained without a rolling window. Unexpected
compaction or context exhaustion ends the attempt as an infrastructure issue.

## Adaptations and evaluation

- This is a LIBERO/Panda adaptation, not an exact reproduction of DROID/RoboLab.
  It uses two 512×512 cameras and native 20 Hz OSC, instead of the report's three
  cameras and 15 Hz joint controller. Existing motion and calibration modules
  are imported without edits.
- Targets use world grip-site position and right-hand orientation. Target limits
  are 5 cm / 0.35 rad; requested per-step increments are bounded to 2 cm / 0.1 rad.
  These are controller limits, not collision guarantees.
- Camera calibration and fixed robot geometry are supplied as in the report's
  expanded interface. No demonstrations, hidden object poses, depth, segmentation,
  contact identities, rewards or subtask progress reach the policy.
- Only the raw LIBERO predicate counts as success. Collision information is
  recorded for evaluation. No drawer-closing or extra post-release criterion is
  added. There is no whole-episode action, call or wall-time cap, and no automatic
  retry. Operator interruption and infrastructure/subscription errors end the run.
- The summary separates Codex turns from actual inference requests counted by
  the gateway. Video overhead is measured separately. An incomplete attempt's
  elapsed time is never labeled task-completion time.

## Review and validation

Run from `/workspace/gpt_field` with the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m astranomy.run
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m pytest -q -p no:cacheprovider astranomy/test_policy.py astranomy/test_transport.py tests/test_astranomy_agentic_motion.py tests/test_astranomy_agentic_prompt.py
```

The preflight command prints the proposed configuration and performs no login,
model inference or simulator rollout. `prepare.py` saves the plan, complete
prompt, tool schema and source hashes. `offline_check.py` tests the actual installed
Codex against a fake local provider, using synthetic images and robot feedback.
This verifies transport/history behavior, not physical task performance or live
subscription entitlement. Earlier failed mock checks remain preserved.

## Launch only after approval

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. .venv/bin/python -m astranomy.launch --approved-full-test --name drawer_bowl_001
```

The launcher creates a private supervisor with automatic restart disabled.
It does not change the instance's services or expose an internet port. Results,
video, source snapshots, request logs and measured motion feedback go to
`astranomy/runs/drawer_bowl_001/`. Supervisor logs go to
`astranomy/launch/drawer_bowl_001/`. Existing directories are never overwritten.

To request a graceful stop after launch:

```bash
touch astranomy/runs/drawer_bowl_001/operator_stop.request.json
```

After the episode exits, shut down its private supervisor:

```bash
supervisorctl -c astranomy/launch/drawer_bowl_001/supervisord.conf shutdown
```

App-server integration follows the [official Codex protocol](https://learn.chatgpt.com/docs/app-server)
and [subscription provider configuration](https://developers.openai.com/siwc/token-sharing-open-source/codex-app-server).
