import json
from pathlib import Path

import pytest

from robodojo_uc71 import host_runtime as runtime


@pytest.fixture
def installation(tmp_path,monkeypatch):
    monkeypatch.setattr(runtime,"BASE",tmp_path)
    monkeypatch.setattr(runtime,"LOCKS",("base.lock","native.lock","compat.lock","tools.lock"))
    for name,text in zip(runtime.LOCKS,(
        "setuptools==84.2.1\ntorch==2.7.0+cu128\n","hidapi==0.15.0\n",
        "setuptools==80.9.0\n","supervisor==4.3.0\n")):
        (tmp_path/name).write_text(text)
    (tmp_path/"runtime").mkdir()
    python = tmp_path/"env/bin/python"; python.parent.mkdir(parents=True); python.write_text("interpreter fixture")
    purelib = tmp_path/"env/site-packages"; purelib.mkdir()
    (purelib/"robodojo_uc71_source.pth").write_text(str(tmp_path/"src")+"\n")
    packages = runtime.pinned_versions(runtime.combined_lock())
    value = {"python":"fixture","python_version":[3,11,16],"purelib":str(purelib),
        "packages":[{"name":name,"version":version} for name,version in packages.items()]}
    monkeypatch.setattr(runtime,"inventory",lambda _:value)
    monkeypatch.setattr(runtime.subprocess,"run",lambda *_,**__:None)
    record = runtime.prepare_conda_record(python,tmp_path/"record")
    return python,record,value


def test_conda_record_preserves_compatibility_override(installation):
    python,path,_ = installation
    info = runtime.verify_conda_record(python,path)
    assert info["layout"] == "consolidated_conda"
    assert runtime.pinned_versions(runtime.combined_lock())["setuptools"] == "80.9.0"


@pytest.mark.parametrize("change",["package","inventory","lock","interpreter","pth","overlay","schema"])
def test_conda_runtime_rejects_drift(installation,change):
    python,path,value = installation
    if change == "package": value["packages"][0]["version"] = "0.0.0"
    if change == "inventory": path.with_name("package_inventory.json").write_text("{}")
    if change == "lock": path.with_name("requirements.lock").write_text("unreviewed")
    if change == "interpreter": python.write_text("changed executable")
    if change == "pth": (Path(value["purelib"])/"robodojo_uc71_source.pth").write_text("unreviewed")
    if change == "overlay": (runtime.BASE/"runtime/conda_imports/torch.py").write_text("unreviewed")
    if change == "schema":
        record=json.loads(path.read_text()); record["schema"]="unreviewed"; path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        runtime.verify_conda_record(python,path)
