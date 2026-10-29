# GPT field from an iPad

Use ChatGPT's Codex/Remote view to talk to the Codex server on the Vast instance.
Experiments and files stay in `/workspace/gpt_field`. Ask for a private HTTPS link
when you want to watch a rollout in Safari. An always-on personal computer is not
part of this setup; the Vast instance must stay running.

## Connect ChatGPT

In the iPad ChatGPT app, sign in to the same account/workspace and open
Codex/Remote. Add a computer using a manual pairing code. The current server name
is `f0a181e29c0f`; choose `/workspace/gpt_field` as the project. Open the existing
project conversation if it is offered. Otherwise start a conversation with:

> Read MOBILE_WORKFLOW.md and AGENTS.md. Help me manage gpt_field experiments.
> Start by checking the latest recorded runs and any live experiment processes.

Generate a fresh, short-lived code from the server when needed:

```bash
cd /workspace/gpt_field
python3 scripts/ipad_codex.py status
python3 scripts/ipad_codex.py pair
```

This instance runs Codex CLI 0.153.4 and an existing app server with a private
Unix socket. The helper uses that installed version's experimental remote-control
protocol. Remote access has been enabled in place with the existing ChatGPT login.
It does not expose the app-server socket on the internet. Pairing on the iPad still
has to finish in the app. If the relay is disconnected, `enable` reconnects it;
`disable` turns remote access off. Neither command starts or stops the app server.

The [CLI reference](https://learn.chatgpt.com/docs/cli/reference) documents
experimental remote control and pairing. The general
[mobile guide](https://learn.chatgpt.com/docs/mobile) primarily describes desktop
pairing; availability and button labels may differ. This helper is specific to the
installed server protocol and may need updating with Codex.

## Talk to Codex

Use ordinary requests such as:

- “Check the latest t1 run. Give me the outcome, steps, requests, and timing.”
- “Compare t1_002 and t1_003 and explain why the last run stopped.”
- “Show me the rollout video for t1_003.”
- “Prepare the next t1 test and explain the changes before execution.”
- “Stop run `<exact run ID>` gracefully and summarize the attempt.”

For a compact recorded-status report, Codex can run:

```bash
python3 scripts/experiment_status.py --limit 5
python3 scripts/experiment_status.py agent_speedup_v3_t1_003 --json
```

The helper only reads immediate run directories. A recorded `running` state does
not establish process liveness; check the relevant private supervisor/process
before describing a run as active. It reports raw evaluator success separately
from timing. Incomplete attempts do not count as successful completion latency.

Run launches, retries, plan approvals, and changes still follow `AGENTS.md` and
the applicable runner's review contract. This mobile setup launches no experiment.
Keep subscription authentication, model settings, and archived evidence intact.
For explicit stop requests, verify the exact run and create its existing
`operator_stop.request.json`; the runner checks it at a complete-feedback boundary.
This ends the attempt; it is not a resumable pause. Do not execute an arbitrary
shell command copied out of a run record. Existing t2/t3 review processes may hold
older drafts; inspect their runtime/review binding before acting on them.

## Watch rollout videos

The media helper copies a selected finalized `episode.mp4`, verifies iPad-compatible
H.264 video, and remuxes the copy for fast startup. It never changes the original.
The curated copies live under ignored `.private/mobile_media/public/`.

```bash
python3 scripts/mobile_media.py publish agent_speedup_v3_t1_003
python3 scripts/mobile_media.py start
python3 scripts/mobile_media.py link agent_speedup_v3_t1_003
```

`publish` returns a local path until the service is ready. Share only the resulting
HTTPS link in the chat once `status` reports `ready: true`. A filesystem path or
`localhost` URL on the Vast instance is not an iPad-accessible video link.

A dedicated private supervisor manages a loopback-only Caddy file server and a
Cloudflare quick tunnel. The Caddy edge requires a random query token and serves
only exported videos, with directory browsing disabled. It cannot control runs or
read the rest of the workspace. The full link grants access to its holder; keep
tokens and pairing codes out of Git and experiment logs.

External sharing requires approval of the selected videos and the Cloudflare
destination if automatic approval review asks for it. Local preparation can happen
before that approval; starting the external tunnel must wait. No public app-server
listener is needed.

```bash
python3 scripts/mobile_media.py status
python3 scripts/mobile_media.py stop
```

`stop` shuts down only this media service. A later `start` rotates the token and
gets a new temporary Cloudflare URL, so request new links. To remove one export,
delete its exact copy under `.private/mobile_media/public/`, leaving `runs/` intact.
An active recording is not published; wait for its finalized `episode.mp4`.

## Availability and checks

The current Codex server's startup is managed outside these helpers. Its automatic
recovery after an instance restart has not been verified. The media service runs
under its own supervisor while the instance is up; run `start` after an instance
restart if needed. Cloudflare quick-tunnel availability is best effort.

This instance currently has no persistent workspace volume. Recycling or
destroying it removes local experiments and this setup; stopping/starting it
preserves the disk. Keep any required backups off the instance.

Offline workflow checks (no model calls or robot execution):

```bash
/usr/bin/python3 -m pytest -q tests/test_experiment_status.py tests/test_mobile_media.py tests/test_ipad_codex.py
```
