import importlib.util
import os
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("host_worker_setup",
    Path(__file__).resolve().parents[1]/"scripts/configure_host_worker.py")
setup = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(setup)


@pytest.mark.parametrize("kind",["symlink_file","symlink_directory","hardlink","fifo"])
def test_credential_ownership_transfer_rejects_nonplain_tree(tmp_path,kind):
    store=tmp_path/"store"; store.mkdir()
    profiles=store/"profiles"; profiles.mkdir()
    outside=tmp_path/"outside"; outside.write_text("nonsecret fixture")
    target=profiles/"profile.json"
    if kind == "symlink_file": target.symlink_to(outside)
    if kind == "symlink_directory": (store/"external").symlink_to(tmp_path,target_is_directory=True)
    if kind == "hardlink": os.link(outside,target)
    if kind == "fifo": os.mkfifo(target)
    with pytest.raises(ValueError,match="plain_single_link"):
        setup.validate_store_tree(store)
    assert outside.read_text() == "nonsecret fixture"
