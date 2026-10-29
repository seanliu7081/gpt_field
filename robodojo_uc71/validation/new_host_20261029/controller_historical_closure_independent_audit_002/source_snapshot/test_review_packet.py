import importlib.util
import json
from pathlib import Path

import pytest

from client_fixtures import plan
from robodojo_uc71.approval import file_sha256
from robodojo_uc71.planning import plan_validation


@pytest.fixture
def packet(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("packet_preparation_test",scripts/"prepare_review_packet.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def original_plan_files(directory):
    draft = plan()
    response = {"status":"completed","output":[{"type":"message","content":[
        {"type":"output_text","text":json.dumps(draft)}]}]}
    for name,value in (("original_draft.json",draft),("original_response.json",response),
                       ("validation.json",plan_validation(draft))):
        (directory/name).write_text(json.dumps(value))
    return draft


def test_original_response_and_draft_verified_offline(packet,tmp_path):
    draft = original_plan_files(tmp_path)
    actual, validation = packet.verified_original_plan(tmp_path)
    assert actual == draft and validation == plan_validation(draft)


def test_edited_draft_cannot_claim_no_edits(packet,tmp_path):
    draft = original_plan_files(tmp_path)
    draft["parts"][0]["x"] = (draft["parts"][0]["x"] + 1) % 64
    (tmp_path/"original_draft.json").write_text(json.dumps(draft))
    with pytest.raises(ValueError,match="differs_from_original_model_response"):
        packet.verified_original_plan(tmp_path)


def test_saved_validation_must_equal_recomputed_validation(packet,tmp_path):
    original_plan_files(tmp_path)
    (tmp_path/"validation.json").write_text('{"valid":true,"forged":true}')
    with pytest.raises(ValueError,match="validation_differs"):
        packet.verified_original_plan(tmp_path)


def test_camera_relative_evidence_is_frozen_with_source_repair(packet,tmp_path,monkeypatch):
    monkeypatch.setattr(packet,"BASE",tmp_path)
    (tmp_path/"camera_003").mkdir()
    raw = tmp_path/"camera_003/rgb.png"; raw.write_bytes(b"fixture")
    source = tmp_path/"repair.py"; source.write_text("# fixture\n")
    assessment = tmp_path/"assessment.json"
    assessment.write_text(json.dumps({"schema":"robodojo-new-host-camera-assessment-1",
        "raw_provenance":{"camera_003/rgb.png":{"sha256":file_sha256(raw),"bytes":raw.stat().st_size}},
        "source_repair":{"path":"repair.py","sha256":file_sha256(source)}}))
    files = packet.FileBindings(); files.document_closure(assessment)
    assert files.values[str(raw)] == file_sha256(raw)
    assert files.values[str(source)] == file_sha256(source)
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError,match="source_binding_changed"):
        packet.FileBindings().document_closure(assessment)


def test_external_json_reference_cannot_read_parent_credentials(packet,tmp_path,monkeypatch):
    approved = tmp_path/"approved"; approved.mkdir(); monkeypatch.setattr(packet,"BASE",approved)
    external = tmp_path/"outside.json"; external.write_text("{}")
    document = approved/"document.json"
    document.write_text(json.dumps({"path":str(external),"sha256":file_sha256(external)}))
    with pytest.raises(ValueError,match="external_evidence_reference"):
        packet.FileBindings().document_closure(document)


def test_sent_planning_payload_matches_prepared_without_copying_metadata(packet,tmp_path):
    payload = {"model":"fixture","input":[]}
    (tmp_path/"planning_payload.json").write_text(json.dumps(payload))
    logs = tmp_path/"subscription_logs"; logs.mkdir()
    record = {"payload":payload,"request_key":"fixture-key","request_ordinal":1,
              "configuration":{"parent_only_fixture":"not copied"}}
    (logs/"1.request.json").write_text(json.dumps(record))
    audit = packet.verified_planning_payload(tmp_path)
    assert audit["sent_structured_payloads_match_prepared"] is True
    assert audit["transport_request_count"] == 1
    assert "parent_only_fixture" not in json.dumps(audit)
    record["payload"] = {"model":"different","input":[]}
    (logs/"1.request.json").write_text(json.dumps(record))
    with pytest.raises(ValueError,match="sent_planning_payload_differs"):
        packet.verified_planning_payload(tmp_path)


def test_prepared_payload_alone_is_not_evidence_of_sent_input(packet,tmp_path):
    (tmp_path/"planning_payload.json").write_text("{}")
    with pytest.raises(ValueError,match="actual_planning_request_log_required"):
        packet.verified_planning_payload(tmp_path)


def test_real_client_sanitized_log_is_compatible_with_packet_audit(packet,tmp_path):
    from client_fixtures import TASK, OfflineTransport, observation
    from robodojo_uc71.client import NativePlanClient
    transport = OfflineTransport(plan())
    client = NativePlanClient(task_prompt=TASK,transport=transport,log_dir=tmp_path/"subscription_logs")
    images = observation()["images"]
    payload = client.build_payload(client.planning_request(images))
    (tmp_path/"planning_payload.json").write_text(json.dumps(payload))
    assert client.request_plan(images) == plan()
    audit = packet.verified_planning_payload(tmp_path)
    assert audit["transport_request_count"] == 1 and len(transport.calls) == 1


def historical_sources_fixture(packet, tmp_path, monkeypatch):
    monkeypatch.setattr(packet, "BASE", tmp_path)
    scripts = tmp_path / "scripts"; scripts.mkdir()
    live = scripts / "controller.py"; live.write_text("# current program\n")
    analysis = tmp_path / "analysis"; analysis.mkdir()
    snapshot = analysis / "source_snapshot"; snapshot.mkdir()
    archived = snapshot / live.name; archived.write_text("# historical program\n")
    comparison = analysis / "comparison.json"
    comparison.write_text(json.dumps({"source_files": [{"path":str(live), "sha256":file_sha256(archived)}]}))
    companion = analysis / "archived_sources.json"
    record = {"schema":"robodojo-offline-analysis-source-snapshots-1",
        "comparison_sha256":file_sha256(comparison),
        "sources":[{"original_path":str(live),"archived_path":str(archived),"sha256":file_sha256(archived)}],
        "note":"Fixture: retain original report bytes."}
    companion.write_text(json.dumps(record))
    review = tmp_path / "review.json"

    def bind_review(reverse=False, extra=None):
        rows = [("comparison", comparison), ("archived_sources", companion)]
        if reverse: rows.reverse()
        value = {name:{"path":str(path),"sha256":file_sha256(path)} for name,path in rows}
        if extra is not None: value["extra"] = extra
        review.write_text(json.dumps(value))

    bind_review()
    return live, archived, comparison, companion, review, record, bind_review


@pytest.mark.parametrize("reverse", [False, True])
def test_historical_source_closure_preserves_report_and_current_program(packet,tmp_path,monkeypatch,reverse):
    live, archived, comparison, companion, review, record, bind_review = historical_sources_fixture(packet,tmp_path,monkeypatch)
    original = comparison.read_bytes()
    bind_review(reverse)
    files = packet.FileBindings(); files.document_closure(review)
    assert str(live) not in files.values
    # Current executable code is a separate binding, as in prepare's program sweep.
    files.add(live)
    assert files.values[str(live)] == file_sha256(live) != record["sources"][0]["sha256"]
    assert files.values[str(archived)] == record["sources"][0]["sha256"]
    assert files.values[str(companion)] == file_sha256(companion)
    assert comparison.read_bytes() == original
    assert {str(live),str(archived),str(comparison),str(companion),str(review)} == set(files.mapping().values())


def test_unreferenced_companion_never_overrides_live_hash_mismatch(packet,tmp_path,monkeypatch):
    _, _, comparison, _, _, _, _ = historical_sources_fixture(packet,tmp_path,monkeypatch)
    with pytest.raises(ValueError,match="source_binding_changed"):
        packet.FileBindings().document_closure(comparison)


@pytest.mark.parametrize("changed", ["companion", "comparison", "archive"])
def test_historical_closure_rejects_changed_bound_bytes(packet,tmp_path,monkeypatch,changed):
    _, archive, comparison, companion, review, _, _ = historical_sources_fixture(packet,tmp_path,monkeypatch)
    target = {"companion":companion,"comparison":comparison,"archive":archive}[changed]
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError,match="source_binding_changed"):
        packet.FileBindings().document_closure(review)


@pytest.mark.parametrize("defect", ["wrong_report_hash", "wrong_old_hash", "unknown_original", "duplicate", "extra_key", "wrong_archive_directory", "symlink_archive"])
def test_historical_alias_rejects_malformed_or_unrelated_substitution(packet,tmp_path,monkeypatch,defect):
    live, archived, _, companion, review, record, bind_review = historical_sources_fixture(packet,tmp_path,monkeypatch)
    row = record["sources"][0]
    if defect == "wrong_report_hash": record["comparison_sha256"] = "0" * 64
    elif defect == "wrong_old_hash": row["sha256"] = file_sha256(live)
    elif defect == "unknown_original": row["original_path"] = str(live.with_name("other.py"))
    elif defect == "duplicate": record["sources"].append(dict(row))
    elif defect == "extra_key": row["ignore_mismatch"] = True
    elif defect == "wrong_archive_directory":
        other = tmp_path / archived.name; other.write_bytes(archived.read_bytes()); row["archived_path"] = str(other)
    elif defect == "symlink_archive":
        other = tmp_path / "actual_archive.py"; other.write_bytes(archived.read_bytes())
        archived.unlink(); archived.symlink_to(other)
    companion.write_text(json.dumps(record)); bind_review()
    with pytest.raises(ValueError,match="historical_"):
        packet.FileBindings().document_closure(review)


def test_historical_alias_does_not_apply_to_another_document(packet,tmp_path,monkeypatch):
    _, _, comparison, _, review, _, bind_review = historical_sources_fixture(packet,tmp_path,monkeypatch)
    other = tmp_path / "unrelated.json"; other.write_bytes(comparison.read_bytes())
    bind_review(extra={"path":str(other), "sha256":file_sha256(other)})
    with pytest.raises(ValueError,match="source_binding_changed"):
        packet.FileBindings().document_closure(review)


def test_historical_alias_cannot_hide_direct_current_program_requirement(packet,tmp_path,monkeypatch):
    live, _, _, _, review, record, _ = historical_sources_fixture(packet,tmp_path,monkeypatch)
    files = packet.FileBindings(); files.document_closure(review)
    with pytest.raises(ValueError,match="source_binding_changed"):
        files.add(live,record["sources"][0]["sha256"])


def explicit_alias_fixture(packet,tmp_path,monkeypatch):
    fixture = historical_sources_fixture(packet,tmp_path,monkeypatch)
    live, archived, comparison, _, review, record, bind_review = fixture
    owner = tmp_path / "partial_report.json"; owner.write_bytes(comparison.read_bytes())
    explicit = tmp_path / "explicit_aliases.json"
    aliases = {"schema":"robodojo-historical-evidence-aliases-1", "documents":[{
        "document":{"path":str(owner),"sha256":file_sha256(owner)},
        "sources":record["sources"]}]}
    explicit.write_text(json.dumps(aliases))
    bind_review(extra={"path":str(owner),"sha256":file_sha256(owner)})
    return live, archived, owner, explicit, review, aliases


def test_explicit_companion_and_discovered_comparison_aliases_have_separate_owners(packet,tmp_path,monkeypatch):
    live, archived, owner, explicit, review, _ = explicit_alias_fixture(packet,tmp_path,monkeypatch)
    original = owner.read_bytes()
    files = packet.FileBindings(); files.load_historical_aliases(explicit,file_sha256(explicit))
    files.document_closure(explicit); files.document_closure(review); files.add(live)
    assert {str(explicit),str(owner),str(archived),str(live)} <= set(files.values)
    assert files.values[str(live)] != files.values[str(archived)]
    assert owner.read_bytes() == original


def test_explicit_companion_requires_supplied_hash_before_any_alias(packet,tmp_path,monkeypatch):
    _, _, _, explicit, _, _ = explicit_alias_fixture(packet,tmp_path,monkeypatch)
    files = packet.FileBindings()
    with pytest.raises(ValueError,match="source_binding_changed"):
        files.load_historical_aliases(explicit,"0"*64)
    assert files.historical_aliases == {}


@pytest.mark.parametrize("changed", ["companion", "owner", "archive"])
def test_explicit_bindings_rechecked_when_closure_starts(packet,tmp_path,monkeypatch,changed):
    _, archived, owner, explicit, review, _ = explicit_alias_fixture(packet,tmp_path,monkeypatch)
    files = packet.FileBindings(); files.load_historical_aliases(explicit,file_sha256(explicit))
    target = {"companion":explicit,"owner":owner,"archive":archived}[changed]
    target.write_bytes(target.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="source_binding_changed"):
        files.document_closure(review)


@pytest.mark.parametrize("defect", ["owner_hash", "old_source_hash", "duplicate_owner", "unknown_schema", "extra_row_key", "external_archive", "external_owner"])
def test_explicit_aliases_reject_invalid_bindings(packet,tmp_path,monkeypatch,defect):
    live, archived, _, explicit, _, aliases = explicit_alias_fixture(packet,tmp_path,monkeypatch)
    entry = aliases["documents"][0]
    if defect == "owner_hash": entry["document"]["sha256"] = "0"*64
    elif defect == "old_source_hash": entry["sources"][0]["sha256"] = file_sha256(live)
    elif defect == "duplicate_owner": aliases["documents"].append(entry)
    elif defect == "unknown_schema": aliases["schema"] = "arbitrary_aliases"
    elif defect == "extra_row_key": entry["sources"][0]["ignore_hash"] = True
    elif defect == "external_archive":
        entry["sources"][0]["archived_path"] = str(Path(__file__).resolve())
    elif defect == "external_owner":
        external = tmp_path.parent / (tmp_path.name + "_external.json")
        external.write_text("{}")
        entry["document"]["path"] = str(external)
        real_hash = packet.file_sha256
        def checked_hash(path):
            assert Path(path) != external, "external document must not be read"
            return real_hash(path)
        monkeypatch.setattr(packet,"file_sha256",checked_hash)
    explicit.write_text(json.dumps(aliases))
    with pytest.raises(ValueError):
        packet.FileBindings().load_historical_aliases(explicit,file_sha256(explicit))


def test_explicit_alias_is_not_a_global_old_hash_replacement(packet,tmp_path,monkeypatch):
    _, _, owner, explicit, _, _ = explicit_alias_fixture(packet,tmp_path,monkeypatch)
    other = tmp_path / "unbound_report.json"; other.write_bytes(owner.read_bytes())
    files = packet.FileBindings(); files.load_historical_aliases(explicit,file_sha256(explicit))
    with pytest.raises(ValueError,match="source_binding_changed"):
        files.document_closure(other)


def test_conflicting_explicit_companions_are_rejected(packet,tmp_path,monkeypatch):
    _, _, _, explicit, _, _ = explicit_alias_fixture(packet,tmp_path,monkeypatch)
    other = tmp_path / "second_aliases.json"; other.write_bytes(explicit.read_bytes())
    files = packet.FileBindings(); files.load_historical_aliases(explicit,file_sha256(explicit))
    with pytest.raises(ValueError,match="duplicate_historical_source_alias"):
        files.load_historical_aliases(other,file_sha256(other))


@pytest.mark.parametrize("changed", ["companion", "archive", "current_program"])
def test_packet_mapping_rejects_changes_after_verified_closure(packet,tmp_path,monkeypatch,changed):
    live, archived, _, explicit, review, _ = explicit_alias_fixture(packet,tmp_path,monkeypatch)
    files = packet.FileBindings(); files.load_historical_aliases(explicit,file_sha256(explicit))
    files.document_closure(review); files.add(live)
    target = {"companion":explicit,"archive":archived,"current_program":live}[changed]
    target.write_bytes(target.read_bytes()+b"\n")
    with pytest.raises(ValueError,match="source_binding_changed"):
        files.mapping()
