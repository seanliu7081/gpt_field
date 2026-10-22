Astranomy persistent Codex agentic / Astra high / LIBERO-10 tasks 2,6,8 (zero-based).
Each task: one fresh episode, init0, seed7, native512 RGB, fullhistory, native20Hz/1-5steps.
Task sentence varies; original motion/prompt body/history/transport preserved.
LIBERO-10 uses the existing generic record-only contact monitor; only raw LIBERO success stops normally.
No task-specific recipes, demonstrations, object geometry, truth, depth, segmentation or contact labels reach the policy.
132 offline tests, three real reset/geometry checks, and three mock Codex protocol checks passed.
Original project code and prior runs are unchanged.
Three independent episodes run in parallel, with separate threads/processes/render GPUs; model service is shared.
No automatic retries or episode caps. For a graceful stop create operator_stop.request.json in that run output.
