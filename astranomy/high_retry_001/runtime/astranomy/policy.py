"""Policy contract shared by offline preparation and the live Codex adapter."""
from __future__ import annotations

from gpt_field.astranomy_agentic_prompt import instructions
from gpt_field.astranomy_agentic_client import tool_definitions

MODEL = "gpt-6-astra"
EFFORT = "high"
TASK = "drawer_bowl"
PROVIDER = "astranomy_chatgpt_plan"
REFERENCE = "https://research.nvidia.com/labs/srl/projects/astranomy/"
VERSION = "astranomy-codex-libero-v1"


def policy_instructions():
    return instructions(TASK) + """

PERSISTENT CODEX SESSION
This episode uses one persistent Codex thread. Each observation starts a turn.
The execute_eef_target tool submits a PROPOSAL ONLY. Its immediate acknowledgement
does not mean the robot moved. Submit exactly one proposal, then end your turn.
The host executes it only after the entire turn completes successfully. The next
observation contains the actual execution feedback and synchronized new images.
Never submit a second target in the same turn. Do not infer movement from the
proposal acknowledgement. A failed or interrupted turn executes no proposal.
Maintain your subgoal plan and revise it from measured outcomes within this thread.
Do not request additional tools, delegates, filesystem access, or outside information.
"""


def dynamic_tools():
    tool = tool_definitions()[0]
    return [{"type": "namespace", "name": "robot",
             "description": "Submit one short robot target for host execution after this turn.",
             "tools": [{"type": "function", "name": tool["name"],
                        "description": "Propose (do not execute) one absolute EEF target held for 1-5 control steps.",
                        "inputSchema": tool["parameters"], "deferLoading": False}]}]


def config_overrides():
    # Do not inherit programming tools, user memories, project guides, or plugins
    # into the robot policy. No scene data or historical experiment is readable.
    disabled = ("shell_tool", "unified_exec", "view_image", "apps", "plugins",
                "multi_agent", "multi_agent_v2", "browser_use", "computer_use",
                "image_generation", "code_mode", "code_mode_host", "goals",
                "memories", "hooks", "shell_snapshot", "sleep_tool", "skill_search",
                "tool_suggest", "workspace_dependencies", "unbounded_connection_retries",
                "compaction_image_budget", "remote_compaction_v2")
    result = {"model": MODEL, "model_provider": PROVIDER,
              "model_reasoning_effort": EFFORT, "web_search": "disabled",
              "project_doc_max_bytes": 0, "notify": [], "mcp_servers": {},
              "tools.update_plan.enabled": False,
              "tools.experimental_request_user_input.enabled": False,
              "agents.enabled": False,
              "skills.bundled.enabled": False,
              "skills.include_instructions": False,
              "include_environment_context": False,
              "include_collaboration_mode_instructions": False,
              "include_apps_instructions": False,
              "include_permissions_instructions": False,
              "features.enable_request_compression": False,
              "history.persistence": "none",
              # Avoid automatic history compaction; context exhaustion is an
              # infrastructure outcome, not permission to drop visual history.
              "model_auto_compact_token_limit": 2**60,
              "model_post_turn_compact_threshold_percent": 100,
              "features.skip_host_skill_discovery": True}
    result.update({"features." + name: False for name in disabled})
    return result


def review_plan():
    return {"version": VERSION, "reference": REFERENCE,
            "status": "prepared_awaiting_user_approval", "full_test_started": False,
            "task": "open the top drawer and put the bowl inside",
            "suite": "libero_goal", "task_id": 3, "episode_index": 0, "seed": 7,
            "model": MODEL, "reasoning_effort": EFFORT,
            "billing": "chatgpt_subscription", "api_key_fallback": False,
            "runtime": "persistent_codex_app_server_thread",
            "action": "one absolute EEF pose; 1-5 native control steps",
            "inputs": ["two upright 512x512 RGB images", "measured robot pose and arm joints",
                       "fixed gripper geometry and synchronized camera calibration",
                       "prior observations, plans, proposals and measured motion feedback"],
            "excluded_inputs": ["demonstrations", "object poses", "depth", "segmentation",
                                "contact identities", "reward", "subtask progress"],
            "optional_tools": False, "episode_caps": None,
            "success": "raw LIBERO goal predicate only", "collisions": "record only",
            "stopping": ["LIBERO success", "operator stop", "infrastructure/subscription error"],
            "adaptations": ["LIBERO Panda instead of DROID/Robotiq",
                            "two 512x512 cameras instead of three 384x216 cameras",
                            "20 Hz OSC instead of 15 Hz damped-least-squares joint control",
                            "world grip-site position/right-hand orientation instead of robot-root flange pose",
                            "no privileged subtask progress; no episode cap"],
            "approval_scope": "one fresh drawer_bowl episode only; no automatic retry or other task"}
