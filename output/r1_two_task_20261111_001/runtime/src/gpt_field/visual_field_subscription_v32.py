"""Explicit rigid transit support, separate from hand/contact geometry."""
from __future__ import annotations

import copy
import json

from .astra import SchemaError, _canonical
from .visual_field_subscription import _array, _validate
from .visual_field_subscription_v2 import _text_record
from .visual_field_subscription_v4 import _runtime_schema
from .visual_field_subscription_v5 import _validate_region
from .visual_field_subscription_v8 import _ordered_roi
from .visual_field_subscription_v15 import MOTION_LABEL as OLD_MOTION_LABEL
from .visual_field_subscription_v18 import CANDIDATE_LABEL as OLD_CANDIDATE_LABEL
from .visual_field_subscription_v31 import (
    VisualFieldSubscriptionClientV31, output_schema as schema_v31,
    geometry_semantics as semantics_v31, FEEDBACK_VERSION as BASE_FEEDBACK_VERSION,
)

VERSION = 'visual-field-rgb-joints-v32-explicit-rigid-transit-reference-v1'
HISTORY_VERSION = 'visual-v32-separate-reference-contact-clearance-v1'
FEEDBACK_VERSION = 'current-guard-exact-debug-snapshot-delta-v32-v1'
REFERENCE_ROLES = ('tool_anatomy', 'rigid_transit_proxy')
CANDIDATE_LABEL = (
    'Current observation-bound reference candidates: exact external-image patches, not semantic labels. '
    'Inspect the original RGB views. Choose tool_anatomy on the actual hand/tool, or explicitly declare '
    'rigid_transit_proxy on housing rigidly attached to the EEF for fixed-orientation noncontact transit. '
    'A proximal articulated link, fixed base, shadow or merely moving patch is not such a proxy. '
    'Keep the true whole-hand clearance ROI separate from the proxy support. '
    'Selection certifies neither identity, grasp alignment nor clearance and refunds no motion allowance.'
)
MOTION_LABEL = (
    'Acknowledged RGB motion diagnostic. Changed pixels and tracked patches are evidence, not robot '
    'segmentation or identity. Explicitly distinguish actual hand anatomy from a rigid EEF transit '
    'proxy; keep true hand clearance and contact anatomy separate. Shadows, background and articulated '
    'links can move. A new support does not certify clearance or refill a budget.'
)


def output_schema(controller='hybrid'):
    result = schema_v31(controller)
    external = result['properties']['visual_references']['properties']['external']['anyOf'][0]
    external['properties']['reference_role'] = {'type': 'string', 'enum': list(REFERENCE_ROLES)}
    external['required'].append('reference_role')
    return result


def geometry_semantics():
    result = semantics_v31()
    result['version'] = 'separate-rigid-transit-support-and-hand-geometry-v32'
    result['reference_role_contract'] = {
        'wire_field': 'visual_references.external.reference_role',
        'roles': list(REFERENCE_ROLES),
        'proxy_scope': 'empty_tool_noncontact_fixed_orientation_field_or_exact_zero_direct_hold',
        'proxy_support_is_hand_collision_envelope': False,
        'proxy_is_anatomical_contact_or_visual_goal': False,
        'proxy_candidate_uses_exact_current_resolver_patch': True,
        'model_semantic_claim_is_certified_identity': False,
        'reference_change_refunds_physical_allowance': False,
    }
    result['feedback_presentation'] = {
        'version': FEEDBACK_VERSION, 'base': BASE_FEEDBACK_VERSION,
        'scope': 'Known nested guard debug snapshots only; exact same-observation deltas against complete current translation_guard.',
        'current_geometry_reference_limits_choices_preserved': True,
        'full_local_trace_and_raw_assistant_unchanged': True,
    }
    return result


def _structural_changes(base, value, path=()):
    if isinstance(base, dict) and isinstance(value, dict):
        result = []
        for key in sorted(base.keys() - value.keys()):
            result.append({'path': [*path, key], 'remove': True})
        for key, item in value.items():
            if key not in base:
                result.append({'path': [*path, key], 'value': copy.deepcopy(item)})
            else:
                result.extend(_structural_changes(base[key], item, (*path, key)))
        return result
    if base == value:
        return []
    return [{'path': list(path), 'value': copy.deepcopy(value)}]


def restore_guard_debug_snapshot(value, current_guard):
    """Exact decoder; inputs and the authoritative current guard are untouched."""
    if not isinstance(value, dict) or value.get('presentation') != FEEDBACK_VERSION:
        return copy.deepcopy(value)
    if value.get('base_path') != 'observable_context.visual_feedback.translation_guard':
        raise ValueError('Unknown guard diagnostic base')
    result = copy.deepcopy(current_guard)
    for change in value['changes']:
        path = change['path']
        if not path:
            result = copy.deepcopy(change['value'])
            continue
        parent = result
        for key in path[:-1]:
            parent = parent[key]
        if change.get('remove'):
            del parent[path[-1]]
        else:
            parent[path[-1]] = copy.deepcopy(change['value'])
    return result


def compact_guard_debug_payload(payload):
    """Only known duplicate admission snapshots; no across-request references."""
    result = copy.deepcopy(payload)
    for message in result.get('input', []):
        if message.get('role') != 'user':
            continue
        for part in message.get('content', []):
            if part.get('type') != 'input_text':
                continue
            try:
                record = json.loads(part['text'])
            except (TypeError, ValueError, KeyError):
                continue
            if not isinstance(record, dict) or record.get('type') != 'observation':
                continue
            feedback = (record.get('observable_context') or {}).get('visual_feedback')
            if not isinstance(feedback, dict):
                continue
            guard = feedback.get('translation_guard')
            if not isinstance(guard, dict):
                continue
            def visit(value):
                if isinstance(value, list):
                    for item in value:
                        visit(item)
                elif isinstance(value, dict):
                    admission = value.get('translation_admission')
                    if isinstance(admission, dict):
                        original = admission.get('diagnostics')
                        snapshot = original
                        if (isinstance(original, dict) and set(original) == {'feedback_ref', 'overrides'}
                                and isinstance(original['feedback_ref'], dict)
                                and original['feedback_ref'].get('path') == 'translation_guard'):
                            keys = original['feedback_ref'].get('keys')
                            if isinstance(keys, list) and all(key in guard for key in keys):
                                snapshot = {key: copy.deepcopy(guard[key]) for key in keys}
                                snapshot.update(copy.deepcopy(original['overrides']))
                        if (isinstance(snapshot, dict) and 'translation_mode' in snapshot
                                and 'configured' in snapshot and 'clearance_certified' in snapshot):
                            compact = {
                                'presentation': FEEDBACK_VERSION,
                                'base_path': 'observable_context.visual_feedback.translation_guard',
                                'semantics': 'Exact prospective/debug snapshot: apply changed/removed keys to the complete current guard in this SAME observation; this is not current permission.',
                                'changes': _structural_changes(guard, snapshot),
                            }
                            if len(_canonical(compact)) < .85 * len(_canonical(original)):
                                admission['diagnostics'] = compact
                    for key, item in value.items():
                        if key != 'translation_admission':
                            visit(item)
            # Complete canonical guard stays intact; it never refers back to
            # the compressed children, so there are no reference cycles.
            for key, value in feedback.items():
                if key != 'translation_guard':
                    visit(value)
            part['text'] = _canonical(record)
    return result


def shared_guidance(task, demonstration=False):
    if demonstration:
        raise ValueError('V32 evaluation is zero-shot')
    return f'''Complete this task: {task.prompt}
Use only both RGB views, measured EEF xyz/axis-angle/fingers, joints and acknowledged history. No demo, depth, calibration, object truth or evaluator input. RGB geometry is an estimate; local arrival and appearance do not establish task success.

CONTROL. Select one branch, others null: visual_field for bulk translation with field_intent, or field_proposal_id with field_intent=null; direct with exactly FIVE 7D actions for necessary contact/articulation/reorientation; gripper with explicit open/close, max_actions10..30, only grasp/release. Direct normalized [xyz,rotation xyz,grip] is in[-1,1], translation1=.05m, rotation1=.5rad, world axes, +grip close/-open/0hold,20Hz; measured motion differs. Accepted Direct is unchanged, rejected blocks discarded. Free Direct allows<=2mm commands,<=5mm with supported response; current risk can refuse. Direct release has zero XYZ; place through Field align/xy then release/z and explicitly open after observed support. No rotation just to fix weak tracking. A truly blocked response may return five exact zero actions; holds create no evidence.

FIELD. The local33^3,5mm grid is weighted goal/risk cost, not measured SDF or clearance. World-axis goal offsets are relative to current EEF, components within±.08m. orientation_command=hold and grip held. motion_axes xyz|xy|z requires zero goal offset on locked axes; release is z only. Prefer useful5–8cm free-space goals,max_actions80–120; valid10..120,max_travel_m<=.16,command_limit_m .012–.02 free/.003–.006 near. Current runtime limits can be smaller. Fixed path tracking may omit conflicting locked-axis correction within its existing corridor; actual commands still pass geometry/evidence/budget checks.

SCENE. scene_inventory is the sole object source: stable IDs for ALL visible nonrobot objects, current external/wrist ROIs, approximate center_offset_m/half_extents_m/uncertainty_m. The12-entry limit is not a required count. Null geometry retains a world estimate; omission never deletes it or makes it fresh. Image-only/lost/partial objects remain uncertainty. current_geometry is a rebased estimate, not newly measured truth. Correct a contradicted same-ID estimate using current RGB and measured motion: changed existing nonnull geometry requires scene_revisions={{id,reason,evidence_kind}}, kind current_two_view|measured_motion_conflict|observed_object_motion. current_two_view requires both current ROIs. Initial estimates need no revision. State evidence separately for nominal position/extent and margin; moving a center does not justify smaller uncertainty. Never rename/delete/shrink/reduce margin merely to pass a refusal. Reference-only repair normally keeps geometry=null. A justified revision and ordinary Field intent may share one response.

unknown_regions means genuinely unlocalized/unseen space, with IDs distinct from remembered objects. Do not output legacy obstacles/nearby_obstacles/target_geometry. Full world/projected regions persist; tracked texture or boundary_limited visibility does not show an entire object. Nominal cores constrain known robot parts, eligible uncertainty adds cost, unknown regions remain hard/full-margin. uncertainty_m is a margin applied once, not statistical sigma. Actual candidate sweeps use full tool/payload geometry; no image or model estimate certifies physical clearance. Unobservable scene caps remain15mm travel/5mm command. motion_context.tool_external_roi_uv always bounds the true visible hand/open fingers/carried extent, never merely a support patch or an enlarged box to include a remote proxy.

REFERENCES. Every nonnull external reference declares reference_role=tool_anatomy|rigid_transit_proxy. UV is top-left origin,u right/v down,[0,1]; seed boxes are ordered and contain their points. tool_anatomy is the ordinary anatomy branch: moving_kind=tool means an actual hand/tool feature; moving_kind=payload still means the separately observed carried object, NOT robot anatomy, with independent hand support. rigid_transit_proxy is visible housing rigidly attached to the EEF, NOT an articulated proximal link or base; use only empty-tool, noncontact, fixed-orientation Field approach/align/recover, or exact five-zero Direct reobservation. Proxy supports motion response, not pinch position, grasp readiness, payload attachment or the anatomical image objective. Do not aim the proxy at the target or interpret zero proxy-target error as alignment. Runtime posture/reference evidence can end its domain; switching roles refunds no travel. Keep the true hand clearance ROI separate from the proxy support ROI.

External moving tool/payload differs from its stationary anchor. Field anchor target_id matches the navigation target; Direct/gripper may use another stationary anchor. Wrist tracks the actual interaction target when visible, never invented image center. tool_anatomy goal_offset_uv is desired moving-minus-anchor; changing anatomy requires a consistent offset. retain preserves live IDs/role with seed=null, not a lost correspondence. Payload uses independent actual-tool co-motion evidence. Proxy mode has target_contact=none, contact_alignment=null, intended_contact_id=null and translation_mode=transit|separate; no grip/rotation/contact operation. The measured posture domain and actual empty-tool state are checked locally.

REFERENCE REVIEW. reference_repair distinguishes not_requested,ineligible,extracted_none,available. Lost external anchor can invalidate candidate history; lost wrist alone is not lost external anchor. Weak response alone is inconclusive, while actual loss/static support despite other motion merits inspection. Candidate boards use paid history, not extra calibration actions. For a candidate, set reference_candidate_id and external update=seed,moving_kind=tool; moving_point_uv/moving_roi_uv null so exact current coordinates are resolved locally. Anatomy candidate also has tool_point_uv/tool_roi_uv null. Every proxy seed instead has NONNULL independent actual-hand tool_point_uv/tool_roi_uv, with the hand point inside the true whole-hand ROI; proxy support may lie outside it. Retain preserves both tracks. Unknown/stale IDs fail. Null candidate uses explicit seed coordinates. No automatic identity promotion, clearance, attachment, validation or budget refill; do not blindly repeat a failed feature.

TARGET/CONTACT. target_contact=none forbids contact; fingers exempts only fingers/pads, never palm; payload_support allows separately tracked payload support on a different destination during release, never robot contact. Target geometry is required for approach/closure. Carried geometry persists until observed separation. interaction has exact intended_contact_id=target_id; transit/separate has null. Stable physical IDs survive role changes. Empty-tool normal Field approach/align with transit/task_translation/none may reduce TARGET modeled gap only with positive full-margin sweep outside contact band and current causal same-heading response. Every NON-TARGET check remains. Inside max(30mm,target uncertainty), approach needs current two-view contact_alignment: actual graspable patch/edge/handle within jaw span and actual open-pad midpoint, not image center. Only Field approach/align may seed both view descriptors or retain with both null on same anatomy tracks. Other phases/proxy use null. Missing evidence permits honest handoff/checked away recovery; strong measured misalignment blocks closure. Necessary Direct5 terminal correction precedes explicit grip. Tiny noisy lift is inconclusive: extend one useful bounded test if views/fingers support grasp; visible slip/miss needs changed grasp.

GOALS/PROPOSALS. subgoal_update=new creates ID/phase/target/axes/worldgoal; retain keeps these and recovery_route, ignoring new relative offsets. Direct/gripper repair retains active ID. replace uses that ID and evidence reason; new cannot discard unfinished work. finish needs measured local arrival; none does not clear active work. Reference change alone does not require replacing useful goal. If anatomy-image error contradicts metric progress, review estimate/offset and explicitly revise a wrong goal. field_proposal_id selects an OPTIONAL current offer: copy phase,target,recovery_route,subgoal_update,scene_declarations as scene_inventory,visual_references and motion_context contract EXACTLY; field_intent=null,target_contact=none,contact_alignment=null,reference_candidate_id=null,scene_revisions=[],all geometry=null. Local resolution supplies checked intent, not automatic action. Changed reference/scene requires ordinary response, later fresh offers. No suitable offer means ordinary Field remains available.

ADMISSION/RECOVERY. Read current reasons, remaining allowances and exact offers. Ambiguous projected overlap is not collision and nonworsening is not clearance. Normal empty-tool transit retains full current checks; non-target modeled gap must not decrease and contrary/lost/stale evidence can stop it. Initial heading acquisition spends existing8mm heading/20mm shared debt and intent remaining. Fresh held-out response plus responsive robot may qualify validated_intent_execution under fixed intent limits; old debt stays, loss falls back to its remainder. Sole weak-image-allowance exhaustion can qualify proprioceptive_transit only with current reliable support, exact full-margin geometry, fixed goal, no contact/payload/refinement; it is not visual validation.

Every response has recovery_route normal|reverse_observed|initial_uncertainty_egress; Direct/gripper normal does not erase retained Field route. Non-normal Field requires recover/separate. Select current offers, copying exact goal/tolerance/command/travel; phase/purpose alone does not select a route. reverse_observed uses the bounded checked reversal, normally<=8mm route,<=3mm command, original8mm heading/20mm shared debt;4mm observation is telemetry, not credit. initial_uncertainty_egress additionally uses purpose=reference_repair,empty tool/none: nominal-clear route out of initial uncertainty only,total<=40mm INCLUDING earlier uncertain debt,initial<=4mm probe then required current causal evidence. A selected reference can get one<=4mm confirmation within same total. Current further worsening/domain/reference/unknown/contact failures still stop. No rename,reseed,new goal or revision refunds physical travel. Recovery arrival means resume task, not completion. Latest feedback is authoritative; older engineering summaries are historical, raw assistant outputs and recent two RGB/state/joint observations remain. Do not repeat static confirmation loops as progress.'''


def _external(value):
    return (value.get('visual_references') or {}).get('external')


class VisualFieldSubscriptionClientV32(VisualFieldSubscriptionClientV31):
    history_protocol = HISTORY_VERSION

    @property
    def prompt_version(self):
        return VERSION + '-hybrid'

    def build_payload(self, observation, grid=None, mode='hybrid'):
        payload = super().build_payload(observation, grid, mode)
        payload['instructions'] = shared_guidance(self.task_spec)
        payload['text']['format'].update(name='visual_field_rgb_joints_v32', schema=output_schema())
        # Replace inherited derived-image labels only, never assistant JSON,
        # observation RGB, model coordinates or historical output.
        for message in payload['input']:
            if message.get('role') != 'user':
                continue
            record = _text_record(message)
            if record and record.get('type') == 'visual_history_summary':
                record['protocol'] = HISTORY_VERSION
                message['content'][0]['text'] = _canonical(record)
            for part in message.get('content', []):
                if part.get('type') == 'input_text':
                    if part.get('text') == OLD_CANDIDATE_LABEL:
                        part['text'] = CANDIDATE_LABEL
                    elif part.get('text') == OLD_MOTION_LABEL:
                        part['text'] = MOTION_LABEL
        # Keep the object identity used by V31's full-history cache while
        # changing only request presentation, not stored feedback or raw output.
        compact = compact_guard_debug_payload(payload)
        payload.clear()
        payload.update(compact)
        return payload

    def _validate_v5_response(self, data, grid):
        _validate(data, _runtime_schema(output_schema()))
        legacy = copy.deepcopy(data)
        ext = _external(legacy)
        role = None if ext is None else ext.pop('reference_role')
        proxy = role == 'rigid_transit_proxy'
        if proxy:
            if (ext['moving_kind'] != 'tool' or data['target_contact'] != 'none'
                    or data['contact_alignment'] is not None
                    or data['motion_context']['intended_contact_id'] is not None
                    or data['motion_context']['translation_mode'] not in ('transit', 'separate')):
                raise SchemaError('Rigid transit proxy requires empty-tool noncontact translation without contact descriptors')
            if data['controller'] == 'direct':
                if data['direct_actions'] is None or len(data['direct_actions']) != 5 or any(
                        any(value != 0 for value in action) for action in data['direct_actions']):
                    raise SchemaError('Rigid transit proxy Direct is limited to exactly five all-zero reobservation actions')
            elif data['controller'] != 'visual_field' or data['phase'] not in ('approach', 'align', 'recover'):
                raise SchemaError('Rigid transit proxy only supports Field approach, align or recovery')
            if data['field_intent'] is not None and data['field_intent']['orientation_command'] != 'hold':
                raise SchemaError('Rigid transit proxy requires held orientation')
            if data['motion_context']['tool_external_roi_uv'] is None:
                raise SchemaError('Rigid transit proxy requires the separate current whole-hand clearance ROI')
            seed = ext['seed']
            if seed is not None:
                if seed['tool_point_uv'] is None or seed['tool_roi_uv'] is None:
                    raise SchemaError('Rigid transit proxy seed requires an independent actual-hand point and ROI')
                _ordered_roi(seed['tool_roi_uv'])
                _validate_region(seed, 'tool_roi_uv', 'tool_point_uv')
                _validate_region({'point': seed['tool_point_uv'],
                    'roi': data['motion_context']['tool_external_roi_uv']}, 'roi', 'point')
        candidate = None
        identity = data['reference_candidate_id']
        if proxy and identity is not None:
            seed = ext['seed']
            if not identity.strip() or ext['update'] != 'seed' or seed is None:
                raise SchemaError('Proxy candidate requires explicit current tool seed mode')
            if any(seed[name] is not None for name in ('moving_point_uv', 'moving_roi_uv')):
                raise SchemaError('Proxy candidate coordinates must be null and resolved from the current board')
            if self._candidate_observation is None or not callable(self.reference_candidate_resolver):
                raise SchemaError('Proxy candidate resolution is unavailable for this exact observation')
            try:
                candidate = copy.deepcopy(self.reference_candidate_resolver(
                    identity, copy.deepcopy(self._candidate_observation)))
            except (ValueError, KeyError):
                raise SchemaError('Proxy candidate is unknown, stale or unavailable for this exact observation') from None
            if not isinstance(candidate, dict) or candidate.get('candidate_id') != identity:
                raise SchemaError('Proxy candidate resolver must return the exact selected identity')
            point, roi = candidate.get('current_point_uv'), candidate.get('current_roi_uv')
            _validate(point, _array(2, 0, 1), 'proxy_candidate.current_point_uv')
            _validate(roi, _array(4, 0, 1), 'proxy_candidate.current_roi_uv')
            _ordered_roi(roi)
            _validate_region({'point': point, 'roi': roi}, 'roi', 'point')
            seed['moving_point_uv'], seed['moving_roi_uv'] = copy.deepcopy(point), copy.deepcopy(roi)
            # Only the compatibility validator sees an explicit seed. The
            # physical hand ROI is NEVER altered to contain a proxy candidate.
            legacy['reference_candidate_id'] = None
        if proxy and ext['seed'] is not None:
            seed = ext['seed']
            if seed['moving_roi_uv'] is None:
                raise SchemaError('Rigid transit proxy requires a bounded moving support ROI')
            if seed['moving_point_uv'] == seed['tool_point_uv']:
                raise SchemaError('Rigid transit proxy and actual-hand support must be distinct')
        resolver = self.field_proposal_resolver
        resolved_original = []
        if data['field_proposal_id'] is not None and callable(resolver):
            def role_checked_proposal(identity, observation):
                resolved = copy.deepcopy(resolver(identity, observation))
                resolved_ext = _external(resolved)
                if resolved_ext is None or resolved_ext.get('reference_role') != role:
                    raise ValueError('proposal_reference_role_mismatch')
                resolved_original.append(copy.deepcopy(resolved))
                resolved_ext.pop('reference_role')
                return resolved
            self.field_proposal_resolver = role_checked_proposal
        try:
            result = super()._validate_v5_response(legacy, grid)
        finally:
            self.field_proposal_resolver = resolver
        if _external(result) is not None:
            _external(result)['reference_role'] = role
        if candidate is not None:
            result['reference_candidate_id'] = identity
            result['reference_candidate_resolution'] = candidate
        if resolved_original:
            result['field_proposal_resolution'] = resolved_original[-1]
        result['raw_wire_prediction'] = copy.deepcopy(data)
        return result

    def history_stats(self):
        result = super().history_stats()
        result['protocol'] = HISTORY_VERSION
        result['guard_debug_presentation'] = FEEDBACK_VERSION
        return result
