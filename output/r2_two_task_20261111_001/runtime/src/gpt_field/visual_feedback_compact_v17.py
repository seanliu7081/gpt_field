"""V17 request presentation only: canonical inventory and exact duplicate references.

The complete local trace remains authoritative. Current offsets, every ID, live
support/uncertainty, vetoes and budgets survive; unrecognized fields survive.
Raw model messages, observation state/joints and images are never rewritten.
"""
from __future__ import annotations

import copy
import json

from .astra import _canonical

VERSION = "observable-feedback-inventory-actionable-v17-v1"
_REVISION_KEYS = frozenset(("anchored_eef_xyz_m", "evidence", "new_world_geometry",
    "previous_world_geometry", "observation_index", "update_index"))


def _inventory(value):
    result = copy.deepcopy(value)
    if not isinstance(result, dict):
        return result
    for entry in result.get("entries", []):
        current, world = entry.get("current_geometry"), entry.get("world_geometry")
        if isinstance(current, dict) and isinstance(world, dict) and all(
                current.get(k) == world.get(k) for k in ("half_extents_m", "uncertainty_m")):
            # Runtime already rebased the fixed world box using this observation's
            # measured EEF. Preserve that exact offset for the inherited validator.
            remaining = {k:v for k,v in world.items()
                         if k not in ("center_xyz_m", "half_extents_m", "uncertainty_m")}
            if remaining:
                entry["world_geometry"] = remaining
            else:
                entry.pop("world_geometry")
            entry["world_geometry_from_current_measured_eef"] = True
        revisions = entry.get("geometry_revisions")
        if isinstance(revisions, list):
            # Counts and current geometry remain. Keep unknown extension fields,
            # including unknown keys nested within old geometry records.
            extensions = []
            for revision in revisions:
                if not isinstance(revision, dict):
                    extensions.append(copy.deepcopy(revision)); continue
                extra = {k:copy.deepcopy(v) for k,v in revision.items() if k not in _REVISION_KEYS}
                for name in ("previous_world_geometry", "new_world_geometry"):
                    old = revision.get(name)
                    if isinstance(old, dict):
                        unknown = {k:copy.deepcopy(v)for k,v in old.items()
                                   if k not in ("center_xyz_m", "half_extents_m", "uncertainty_m")}
                        if unknown: extra[name] = unknown
                if extra: extensions.append(extra)
            if extensions:
                entry["geometry_revision_extensions"] = extensions
            entry.pop("geometry_revisions")
    result["geometry_presentation"] = "Exact current_geometry is relative to this observation's measured EEF; full fixed world boxes and old revision records remain in local traces. Omission is not deletion."
    return result


def _reference_subset(value, source, path):
    """Represent only exactly equal snapshot keys by a same-feedback reference."""
    if not isinstance(value, dict) or not isinstance(source, dict) or "feedback_ref" in value:
        return copy.deepcopy(value)
    shared = [key for key,item in value.items()
              if key in source and _canonical(item) == _canonical(source[key])]
    if not shared:
        return copy.deepcopy(value)
    result = {"feedback_ref": {"path":path,"keys":shared},
              "overrides":{key:copy.deepcopy(item)for key,item in value.items() if key not in shared}}
    return result if len(_canonical(result))+100 < len(_canonical(value)) else copy.deepcopy(value)


def _part_arrays(value):
    if isinstance(value,list): return [_part_arrays(item)for item in value]
    if not isinstance(value,dict): return copy.deepcopy(value)
    result={key:_part_arrays(item)for key,item in value.items()}
    for key in ('core_parts','uncertainty_halo_parts','tool_core_parts','full_halo_parts'):
        parts=result.get(key)
        if not isinstance(parts,list): continue
        # Only known metric robot configuration coordinates are omitted. New
        # per-part diagnostics survive alongside names, count and provenance.
        if not all(isinstance(part,dict)for part in parts): continue
        result[key+'_summary']={'part_count':len(parts),'parts':[
            {name:copy.deepcopy(item)for name,item in part.items()
             if name not in ('lower_xyz','upper_xyz','offset_min_xyz','offset_max_xyz','min_xyz','max_xyz')}
            for part in parts], 'expanded_coordinates_in_local_trace':True}
        result.pop(key)
    return result


def _difference(before,base,path=()):
    if _canonical(before)==_canonical(base): return []
    replace=[{'path':list(path),'value':copy.deepcopy(before)}]
    changes=[]
    if isinstance(before,dict) and isinstance(base,dict):
        for key in sorted(base.keys()-before.keys()):changes.append({'path':list(path+(key,)),'delete':True})
        for key,item in before.items():
            changes += (_difference(item,base[key],path+(key,)) if key in base else
                        [{'path':list(path+(key,)),'value':copy.deepcopy(item)}])
    elif isinstance(before,list) and isinstance(base,list) and len(before)==len(base):
        for index,(item,other)in enumerate(zip(before,base)):
            changes += _difference(item,other,path+(index,))
    else:return replace
    return changes if len(_canonical(changes))<len(_canonical(replace)) else replace


def restore_feedback_delta(value,base):
    """Offline inverse of an exact same-request older-feedback delta."""
    result=copy.deepcopy(base)
    for operation in value['feedback_patch']:
        path=operation['path']
        if not path:
            result=copy.deepcopy(operation['value']);continue
        target=result
        for component in path[:-1]:target=target[component]
        if operation.get('delete') is True:del target[path[-1]]
        else:target[path[-1]]=copy.deepcopy(operation['value'])
    return result


def _deduplicate_older_feedback(payload):
    observations=[]
    for message in payload.get('input',[]):
        if message.get('role')!='user':continue
        for part in message.get('content',[]):
            if part.get('type')!='input_text':continue
            try:record=json.loads(part['text'])
            except (ValueError,TypeError,KeyError):continue
            if isinstance(record,dict) and record.get('type')=='observation':
                feedback=(record.get('observable_context')or{}).get('visual_feedback')
                if isinstance(feedback,dict)and 'feedback_ref'not in feedback:
                    observations.append((part,record,feedback))
    if len(observations)<2:return payload
    _,latest,base=observations[-1]
    index=latest.get('observation_index')
    if type(index)is not int:return payload
    for part,record,feedback in observations[:-1]:
        patch={'feedback_ref':{'observation_index':index,'path':'observable_context.visual_feedback'},
               'feedback_patch':_difference(feedback,base),
               'patch_semantics':'Exact older snapshot: start from named latest feedback; apply path value replacements/deletions. No new evidence.'}
        if len(_canonical(patch))+100<len(_canonical(feedback)):
            assert restore_feedback_delta(patch,base)==feedback
            record['observable_context']['visual_feedback']=patch
            part['text']=_canonical(record)
    return payload


def compact_feedback(value):
    if not isinstance(value, dict) or "feedback_ref" in value:
        return copy.deepcopy(value)
    result = _part_arrays(value)
    if "scene_inventory" in result:
        result["scene_inventory"] = _inventory(result["scene_inventory"])
    field, guard = result.get("field"), result.get("translation_guard")
    if isinstance(field, dict):
        admission = field.get("translation_admission")
        if isinstance(admission, dict) and isinstance(admission.get("diagnostics"), dict):
            admission["diagnostics"] = _reference_subset(admission["diagnostics"], guard, "translation_guard")
        for local, canonical in (("scene_coverage_admission", "scene_coverage"),
                                  ("target_contact_geometry", "target_contact_geometry")):
            if local in field and canonical in result:
                field[local] = _reference_subset(field[local], result[canonical], canonical)
    # Exact references only; differing old snapshot values and newly added fields
    # remain in overrides. No rounding, semantic equivalence or cross-turn reuse.
    current_guard = result.get("translation_guard")
    if isinstance(current_guard, dict) and isinstance(current_guard.get("last_admission"), dict):
        admission = current_guard["last_admission"]
        if isinstance(admission.get("diagnostics"), dict):
            admission["diagnostics"] = _reference_subset(admission["diagnostics"],
                {k:v for k,v in current_guard.items() if k != "last_admission"}, "translation_guard")
    result["feedback_representation"] = VERSION
    return result


def _record(value):
    if isinstance(value, list):
        return [_record(item) for item in value]
    if not isinstance(value, dict):
        return copy.deepcopy(value)
    return {key:compact_feedback(item) if key in ("visual_feedback","end_visual_feedback") else _record(item)
            for key,item in value.items()}


def compact_payload_feedback(payload):
    result = copy.deepcopy(payload)
    for message in result.get("input", []):
        if message.get("role") != "user": continue
        for part in message.get("content", []):
            if part.get("type") != "input_text": continue
            try: record = json.loads(part["text"])
            except (ValueError,TypeError,KeyError): continue
            if isinstance(record,dict) and record.get("type") in ("observation","visual_execution_block","visual_history_summary"):
                compact = _record(record)
                if compact != record: part["text"] = _canonical(compact)
    return _deduplicate_older_feedback(result)
