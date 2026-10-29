"""Auditable scene estimates and explicit, observation-bound Field proposals.

The model chooses revisions/proposals. Resolution changes only the internal
validated prediction; wire logs and assistant history keep the original JSON.
Neither operation supplies environmental truth or resets motion allowance.
"""
from __future__ import annotations

import copy
import json

from .astra import SchemaError, _canonical
from .visual_field_subscription import _object, _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v30 import (
    VisualFieldSubscriptionClientV30, output_schema as schema_v30,
    shared_guidance as guidance_v30, geometry_semantics as semantics_v30,
    FEEDBACK_VERSION as BASE_FEEDBACK_VERSION, CANDIDATE_LABEL, compact_older_contexts, RECOVERY_ROUTES,
)

VERSION = 'visual-field-rgb-joints-v31-explicit-scene-revision-field-proposal-v1'
HISTORY_VERSION = 'visual-v31-current-scene-estimate-bound-proposal-v1'
FEEDBACK_VERSION = 'current-scene-estimate-compact-revision-provenance-v31-v1'
SCENE_REVISION_EVIDENCE_KINDS = (
    'current_two_view', 'measured_motion_conflict', 'observed_object_motion',
)


def output_schema(controller='hybrid'):
    result = schema_v30(controller)
    result['properties']['scene_revisions'] = {
        'type': 'array', 'maxItems': 12, 'items': _object({
            'id': {'type': 'string', 'minLength': 1, 'maxLength': 80},
            'reason': {'type': 'string', 'minLength': 1, 'maxLength': 500},
            'evidence_kind': {'type': 'string', 'enum': list(SCENE_REVISION_EVIDENCE_KINDS)},
        }),
    }
    result['properties']['field_proposal_id'] = {'anyOf': [
        {'type': 'string', 'minLength': 1, 'maxLength': 120}, {'type': 'null'}]}
    result['required'].extend(('scene_revisions', 'field_proposal_id'))
    return result


def geometry_semantics():
    result = semantics_v30()
    result['version'] = 'explicit-scene-estimate-revision-bound-field-proposal-v31'
    result['scene_estimate_revision'] = {
        'wire_field': 'scene_revisions', 'evidence_kinds': list(SCENE_REVISION_EVIDENCE_KINDS),
        'existing_changed_geometry_requires_explicit_revision': True,
        'current_visible_geometry_and_measured_eef_anchor_required': True,
        'current_two_view_requires_both_current_rois': True,
        'null_geometry_or_new_roi_refreshes_metric_estimate': False,
        'old_objects_or_physical_debt_deleted': False,
        'nominal_estimate_and_uncertainty_margin_are_distinct': True,
        'revision_is_calibrated_geometry_or_clearance': False,
    }
    result['field_proposal_selection'] = {
        'wire_field': 'field_proposal_id', 'selected_wire_field_intent': None,
        'exact_current_observation_resolver_required': True,
        'exact_role_route_context_contract_required': True,
        'simultaneous_scene_revision_or_reference_reseed_allowed': False,
        'raw_wire_and_history_unchanged': True,
        'actual_candidate_admission_still_required': True,
        'permission_or_physical_allowance_reset': False,
    }
    result['scene_revision_presentation'] = {
        'version': FEEDBACK_VERSION, 'base': BASE_FEEDBACK_VERSION,
        'current_geometry_freshness_and_warning_values_preserved': True,
        'complete_revision_bindings_kept_in_local_trace': True,
        'raw_assistant_and_stored_history_unchanged': True,
    }
    return result


_REVISION_LOG_KEYS = frozenset((
    'id', 'kind', 'previous_world_geometry', 'new_world_geometry', 'revision',
    'observation_binding', 'current_declared_regions', 'evidence_status',
    'geometry_measured_or_certified', 'motion_allowances_or_evidence_changed',
))


def _revision_extensions(record):
    """Unknown future facts survive; only understood historical detail shrinks."""
    if not isinstance(record, dict):
        return copy.deepcopy(record)
    result = {key: copy.deepcopy(value) for key, value in record.items()
              if key not in _REVISION_LOG_KEYS}
    known_nested = {
        'observation_binding': {'state8', 'observation_index', 'update_index', 'rgb_sha256'},
        'previous_world_geometry': {'center_xyz_m', 'half_extents_m', 'uncertainty_m'},
        'new_world_geometry': {'center_xyz_m', 'half_extents_m', 'uncertainty_m'},
        'current_declared_regions': {'external_roi_uv', 'wrist_roi_uv'},
        'revision': {'id', 'reason', 'evidence_kind'},
    }
    for key, known in known_nested.items():
        value = record.get(key)
        if isinstance(value, dict):
            extra = {name: copy.deepcopy(item) for name, item in value.items() if name not in known}
            if extra:
                result[key] = extra
    return result


def _last_revision(record):
    if not isinstance(record, dict):
        return copy.deepcopy(record)
    result = _revision_extensions(record)
    for key in ('id', 'kind', 'revision', 'evidence_status', 'geometry_measured_or_certified',
                'motion_allowances_or_evidence_changed'):
        if key in record:
            result[key] = copy.deepcopy(record[key])
    binding = record.get('observation_binding') or {}
    for key in ('observation_index', 'update_index'):
        if key in binding:
            result[key] = binding[key]
    previous, current = record.get('previous_world_geometry'), record.get('new_world_geometry')
    if isinstance(previous, dict) and isinstance(current, dict):
        if previous.get('center_xyz_m') != current.get('center_xyz_m'):
            result['previous_center_xyz_m'] = copy.deepcopy(previous.get('center_xyz_m'))
        result['changed_estimate_components'] = [key for key in ('center_xyz_m', 'half_extents_m', 'uncertainty_m')
                                                  if previous.get(key) != current.get(key)]
    return result


def _compact_scene_revision_inventory(inventory):
    result = copy.deepcopy(inventory)
    if not isinstance(result, dict):
        return result
    log = result.pop('revision_log', None)
    if isinstance(log, list):
        result['revision_log_summary'] = {
            'record_count': len(log),
            'explicit_revision_count': sum(isinstance(row, dict) and row.get('kind') == 'explicit_metric_revision' for row in log),
            'complete_log_and_observation_bindings_in_local_trace': True,
        }
        extensions = [_revision_extensions(row) for row in log]
        extensions = [row for row in extensions if row]
        if extensions:
            result['revision_log_extensions'] = extensions
    elif log is not None:
        result['revision_log'] = log  # Do not interpret an unknown representation.
    if 'revision_log_complete' in result:
        result['local_revision_log_complete'] = result.pop('revision_log_complete')
    for entry in result.get('entries', []):
        if isinstance(entry, dict) and 'last_explicit_revision' in entry:
            entry['last_explicit_revision'] = _last_revision(entry['last_explicit_revision'])
    result['revision_presentation'] = FEEDBACK_VERSION
    return result


def compact_scene_revision_payload(payload):
    """Presentation only; no raw assistant JSON, state/joints or RGB is touched."""
    result = copy.deepcopy(payload)
    def walk(value):
        if isinstance(value, list):
            return [walk(item) for item in value]
        if not isinstance(value, dict):
            return copy.deepcopy(value)
        return {key: _compact_scene_revision_inventory(item) if key == 'scene_inventory' else walk(item)
                for key, item in value.items()}
    for message in result.get('input', []):
        if message.get('role') != 'user':
            continue
        for part in message.get('content', []):
            if part.get('type') != 'input_text':
                continue
            try:
                record = json.loads(part['text'])
            except (ValueError, TypeError, KeyError):
                continue
            if isinstance(record, dict) and record.get('type') in (
                    'observation', 'visual_execution_block', 'visual_history_summary'):
                part['text'] = _canonical(walk(record))
    return result


def shared_guidance(task, demonstration=False):
    text = guidance_v30(task, demonstration)
    text = text.replace(
        'Choose exactly one nonnull branch: visual_field/field_intent for bulk translation;',
        'Choose exactly one controller branch: visual_field for bulk translation with either '
        'nonnull field_intent or a selected field_proposal_id (then field_intent=null);')
    text = text.replace(
        'Do not alter, rename, shrink or delete a hypothesis to evade rejection; '
        'reference-only repair should retain unchanged geometry as null.',
        'Estimated scene geometry can be wrong: explicitly CORRECT its same-ID nominal position '
        'or extent when current RGB and acknowledged robot motion contradict it. Do not freeze '
        'an old estimate merely because it was previously retained. Keep geometry=null only when '
        'there is no justified new estimate, including a reference-only repair. Never delete, '
        'rename, shrink or reduce a margin merely to make a blocked path pass.')
    marker = '\n\nNominal cores constrain known tool parts;'
    assert marker in text
    text = text.replace(marker, '''

SCENE REVISION. Every response includes scene_revisions, normally []. For an EXISTING object's changed nonnull geometry, add {id,reason,evidence_kind}, with evidence_kind=current_two_view|measured_motion_conflict|observed_object_motion. Describe the actual visible discrepancy or acknowledged displacement; current_two_view requires that object's current ROI in BOTH views. Initial estimates need no revision entry. Revisions refer to this observation and the same physical ID; null geometry, an omitted object or an unchanged estimate is not a revision. A new ROI refreshes image evidence, not the age of a retained metric estimate. Read scene_review and the separate estimate freshness in scene_inventory. current_geometry is the current-EEF representation of a remembered estimate, not necessarily a newly observed pose.

Nominal center/extent and uncertainty margin are different claims. A changed center does not by itself justify smaller uncertainty, and raising a margin is not observed object motion. State evidence for each changed quantity in the revision reason. All estimates remain uncalibrated RGB hypotheses; revisions supply no collision certificate. Local code anchors new offsets at measured EEF, archives prior estimates, rechecks current paths and preserves physical travel/history. You may revise geometry and select an ordinary new/replaced/retained Field intent in one response; no extra confirmation call is required solely for a scene update.

FIELD PROPOSALS. field_proposals contains optional locally generated, current-observation-bound intents from already paid observations, or exact checked recovery offers. Choose one only if its direction and role serve the task. To select, set field_proposal_id to its exact ID and field_intent=null; keep controller=visual_field and copy its phase, target_id, recovery_route, subgoal_update, scene_declarations (as scene_inventory) and required motion_context values exactly. The offered new/replace update selects its fixed goal; retain never creates a new goal. Do not rewrite its margin, command limits or unknown regions. Keep target_contact=none, reference_candidate_id=null, scene_revisions=[], all scene_inventory geometry=null, external references retained, wrist references null/retained and contact_alignment=null. A fresh reference, scene estimate or contact descriptor requires an ordinary response with field_proposal_id=null, after which later offers use the new evidence. IDs cannot survive a changed observation or binding. Resolution supplies the exact internal intent; actual geometry, reference, contact and budget admission still runs. A proposal is neither an automatic action nor permission or calibrated clearance. If no appropriate current offer exists, ordinary explicit Field planning remains available.

Nominal cores constrain known tool parts;''')
    return text


class VisualFieldSubscriptionClientV31(VisualFieldSubscriptionClientV30):
    history_protocol = HISTORY_VERSION
    field_proposal_resolver = None
    scene_revision_validator = None
    _v31_history_payload = None

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v31', schema=output_schema())
        for message in payload['input']:
            record = _text_record(message)
            if record and record.get('type') == 'visual_history_summary':
                record['protocol'] = HISTORY_VERSION
                message['content'][0]['text'] = _canonical(record)
        compact = compact_scene_revision_payload(payload)
        # The transport logs exactly what was sent. In-memory history retains
        # the pre-V31 presentation, including complete local revision records.
        self._v31_history_payload = (id(compact), payload)
        return compact

    def predict(self, observation, grid=None, mode='hybrid'):
        with self._history_lock:
            self._v31_history_payload = None
            try:
                return super().predict(observation, grid, mode)
            finally:
                self._v31_history_payload = None

    def _history_input(self, payload):
        cached = self._v31_history_payload
        original = cached[1] if cached is not None and cached[0] == id(payload) else payload
        return super()._history_input(original)

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        revisions = data['scene_revisions']
        revision_ids = [revision['id'] for revision in revisions]
        if len(revision_ids) != len(set(revision_ids)):
            raise SchemaError('Scene revisions require unique physical object IDs')
        entries = {entry['id']: entry for entry in data['scene_inventory']}
        for revision in revisions:
            entry = entries.get(revision['id'])
            if not revision['id'].strip() or not revision['reason'].strip():
                raise SchemaError('Scene revision identity and evidence reason must be nonempty')
            if entry is None or entry['geometry'] is None:
                raise SchemaError('Scene revision requires the same current inventory ID and nonnull geometry')
            if (revision['evidence_kind'] == 'current_two_view'
                    and (entry['external_roi_uv'] is None or entry['wrist_roi_uv'] is None)):
                raise SchemaError('A current_two_view scene revision requires current object ROIs in both views')
        if callable(self.scene_revision_validator):
            if self._candidate_observation is None:
                raise SchemaError('Scene revision validation is unavailable for this exact observation')
            try:
                self.scene_revision_validator(copy.deepcopy(self._candidate_observation),
                    copy.deepcopy(data['scene_inventory']), copy.deepcopy(revisions))
            except (ValueError, KeyError):
                raise SchemaError('Scene revision does not match current persistent inventory: changed existing geometry needs an exact current evidence record; unchanged, initial or retained geometry does not') from None
        # The persistent inventory validates existing/changed versus initial
        # estimates atomically using its exact world memory, not compact text.
        identity = data['field_proposal_id']
        legacy = copy.deepcopy(data)
        legacy.pop('scene_revisions')
        legacy.pop('field_proposal_id')
        resolved = None
        if identity is not None:
            if not identity.strip() or data['controller'] != 'visual_field' or data['field_intent'] is not None:
                raise SchemaError('Field proposal selection requires visual_field and null wire field_intent')
            if data['reference_candidate_id'] is not None or revisions or any(
                    entry['geometry'] is not None for entry in data['scene_inventory']):
                raise SchemaError('Field proposal selection cannot also revise scene geometry or select a reference candidate')
            refs = data['visual_references']
            if refs['external'] is None or refs['external']['update'] != 'retain':
                raise SchemaError('Field proposal selection requires retained external references')
            if refs['wrist'] is not None and refs['wrist']['update'] != 'retain':
                raise SchemaError('Field proposal selection cannot reseed wrist references')
            if data['contact_alignment'] is not None or data['target_contact'] != 'none':
                raise SchemaError('Field proposal selection requires no contact alignment or target contact permission')
            if self._candidate_observation is None or not callable(self.field_proposal_resolver):
                raise SchemaError('Field proposal resolution is unavailable for this exact observation')
            try:
                resolved = copy.deepcopy(self.field_proposal_resolver(
                    identity, copy.deepcopy(self._candidate_observation)))
            except (ValueError, KeyError):
                raise SchemaError('Field proposal is unknown, stale or unavailable for this exact observation') from None
            if not isinstance(resolved, dict) or resolved.get('proposal_id') != identity:
                raise SchemaError('Field proposal resolver did not return the exact selected identity')
            if not isinstance(resolved.get('binding'), dict) or not resolved['binding']:
                raise SchemaError('Field proposal resolver requires an explicit current observation binding')
            if any(resolved.get(key) != data[key] for key in ('phase', 'target_id', 'recovery_route')):
                raise SchemaError('Selected Field proposal phase, target and recovery route must match the wire exactly')
            if resolved.get('controller', 'visual_field') != 'visual_field':
                raise SchemaError('Field proposal resolver must describe the visual_field controller')
            overrides = resolved.get('motion_context_overrides')
            if not isinstance(overrides, dict) or any(
                    key not in data['motion_context'] or data['motion_context'][key] != value
                    for key, value in overrides.items()):
                raise SchemaError('Selected Field proposal motion context must match the supplied wire exactly')
            if resolved.get('subgoal_update') != data['subgoal_update']:
                raise SchemaError('Selected Field proposal subgoal update must match the supplied wire exactly')
            if resolved.get('scene_declarations') != data['scene_inventory']:
                raise SchemaError('Selected Field proposal current scene declarations must match the supplied wire exactly')
            if resolved.get('visual_references') != data['visual_references']:
                raise SchemaError('Selected Field proposal retained visual references must match the supplied wire exactly')
            field = resolved.get('field_intent')
            if field is None:
                raise SchemaError('Field proposal resolver requires a complete native Field intent')
            _validate(field, _runtime_schema(schema_v30()['properties']['field_intent']['anyOf'][0]),
                      'resolved_field_proposal.field_intent')
            legacy['field_intent'] = copy.deepcopy(field)
        result = super()._validate_v5_response(legacy, grid)
        result.update(scene_revisions=copy.deepcopy(revisions), field_proposal_id=identity,
                      field_proposal_resolution=resolved, raw_wire_prediction=copy.deepcopy(data))
        return result

    def history_stats(self):
        result = super().history_stats()
        result['protocol'] = HISTORY_VERSION
        result['scene_revision_presentation'] = FEEDBACK_VERSION
        return result
