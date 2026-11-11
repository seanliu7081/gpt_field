"""Loopback-only v3 review UI, using the unchanged selection and approval routes."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_path = Path(__file__).with_name("agent_speedup_v2_user_check_server.py")
_spec = importlib.util.spec_from_file_location("gpt_field._speedup_v3_review_server", _path)
_server = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _server
_spec.loader.exec_module(_server)
_server.HTML_PATH = Path(__file__).with_name("agent_speedup_v3.html")
ReviewServer = _server.ReviewServer
review_schema = _server.review_schema
