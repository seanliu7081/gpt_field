"""Route-family and transient-drift regressions; no live model or simulator."""
import copy
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from gpt_field.observation import PolicyObservation, decode_image
from gpt_field.visual_field_v25 import MetricFieldExecutorV25, constrained_recovery_delta
from gpt_field.visual_field_v24 import MetricFieldExecutorV24
from test_visual_field_v22 import HEADINGS, bind_history, configure_reverse, snapshot, assert_unchanged
from test_visual_translation_guard_v21 import plant, setup, triggered, pixels
from test_visual_field_v4 import Q


def promote(field):
    defaults = MetricFieldExecutorV25().__dict__
    field.__class__ = MetricFieldExecutorV25
    for key, value in defaults.items():
        if key not in field.__dict__:
            field.__dict__[key] = value
    field.recovery_route_binding = None
    return field


@pytest.mark.parametrize('direction', HEADINGS)
def test_reverse_preview_and_actual_dispatch_match_despite_reference_repair_labels(direction):
    guard, monitor, field, measured = plant(); promote(field)
    bind_history(guard, field, direction, measured)
    guard.context['purpose'] = 'reference_repair'
    saved = snapshot(field, guard)
    preview = field.preview_recovery(measured, Q, direction*.008)
    assert preview['accepted'], preview['status']
    assert_unchanged(saved, field, guard)
    configure_reverse(field, guard, measured, direction)
    field.egress_context = {'phase': 'recover', 'purpose': 'reference_repair', 'translation_mode': 'separate'}
    assert field.bind_recovery_route('reverse_observed', measured)['accepted']
    selected = field.select(measured, Q)
    assert selected['accepted'], selected['status']
    assert selected['action'] == pytest.approx(preview['action'])
    assert selected['diagnostics']['recovery_route_binding']['recovery_route'] == 'reverse_observed'
    assert not field._egress_requested()
    wrong = np.array(selected['action']); wrong[0] += .01
    with pytest.raises(ValueError):
        field.observe(measured, wrong, measured)
    after = measured.copy(); after[:3] += np.array(selected['action'][:3])*.05*.3
    field.observe(measured, selected['action'], after)
    assert field.actions == 1 and field.pending is None


def test_normal_graph_is_identical_to_v24_and_never_selected_by_purpose_labels():
    ga, ma, old, sa = plant(); gb, mb, new, sb = plant()
    # Existing ordinary geometry/normal action remains bit-for-bit identical.
    promote(old); old.__class__ = MetricFieldExecutorV24
    promote(new); assert new.bind_recovery_route('normal', sb)['accepted']
    old.egress_context = new.egress_context = {'phase': 'approach', 'purpose': 'task_translation', 'translation_mode': 'free'}
    a, b = old.select(sa, Q), new.select(sb, Q)
    assert a['accepted'] and b['accepted'] and a['action'] == b['action']
    assert new.last_plan.search_metadata['method'] == 'dijkstra_distance_field'


@pytest.mark.parametrize('direction', HEADINGS)
def test_arbitrary_headings_and_lateral_drift_stay_inside_existing_cone_without_amplification(direction):
    lateral = np.cross(direction, [0., 0., 1.])
    if np.linalg.norm(lateral) < .1:
        lateral = np.cross(direction, [0., 1., 0.])
    lateral /= np.linalg.norm(lateral)
    requested = (direction*.8+lateral*.6)*.002
    delta = constrained_recovery_delta(requested, direction, [direction], np.pi/6)
    assert np.linalg.norm(delta) <= np.linalg.norm(requested)+1e-12
    assert np.arccos(np.clip(delta@direction/np.linalg.norm(delta), -1., 1.)) <= np.pi/6
    assert delta@lateral > 0.  # Retain correction instead of silently discarding it.
    axes = [direction, (direction+.1*lateral)/np.linalg.norm(direction+.1*lateral)]
    delta = constrained_recovery_delta(requested, direction, axes, np.pi/6)
    assert all(delta@axis/np.linalg.norm(delta) >= np.cos(np.pi/6)-1e-12 for axis in axes)


def archive_v24():
    path = Path(__file__).resolve().parents[1]/'runs/visual_field_v24_zero_shot_001/trials/bowl_original_zero_hybrid_v24/trace.jsonl'
    if not path.exists():
        pytest.skip('Optional V24 saved trial unavailable')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return ({r['step']: r for r in rows if r['event'] == 'action'},
            {r['step']: r['feedback'] for r in rows if r['event'] == 'feedback'},
            {r['step']: PolicyObservation.from_dict(r['observation']) for r in rows if r['event'] == 'executed'})


def archived_reverse_fixture(decision_step=132):
    actions, feedback, observations = archive_v24()
    obs = observations[decision_step-1]
    pred = copy.deepcopy(actions[decision_step]['diagnostics']['prediction'])
    material = copy.deepcopy(actions[decision_step]['diagnostics']['scene_materialization'])
    fb = feedback[decision_step-1]
    refs = copy.deepcopy(pred['visual_references']); track = fb['visual']['tracking']
    refs['external']['update'] = 'seed'
    refs['external']['seed'] = {'moving_point_uv': track['moving']['tracked_point_uv'],
        'moving_roi_uv': track['moving']['current_roi_uv'], 'target_anchor_uv': track['target']['tracked_point_uv'],
        'target_roi_uv': track['target']['current_roi_uv'], 'tool_point_uv': None, 'tool_roi_uv': None}
    context = copy.deepcopy(actions[decision_step]['diagnostics']['feedback']['effective_motion_context'])
    images = {'external': np.asarray(decode_image(obs.image)), 'wrist': np.asarray(decode_image(obs.wrist_image))}
    guard, monitor, field = setup(obs.state, images, refs, context, pred['field_intent'], material)
    promote(field)
    # Current full geometry and real recorded historical worsening; no hidden
    # pose or fabricated successful image response enters this counterfactual.
    guard.worsening = {'other_bowl'}
    guard.worsening_motion = copy.deepcopy(fb['translation_guard']['directional_retreat']['actual_worsening_motion'])
    field.egress_context = {'phase': pred['phase'], 'purpose': context['purpose'], 'translation_mode': context['translation_mode']}
    assert field.bind_recovery_route('reverse_observed', obs.state)['accepted']
    return field, guard, monitor, obs, pred, actions, feedback, observations


def test_recorded132_reverse_offer_does_not_enter_old_origin_egress_dispatch():
    field, guard, monitor, obs, pred, actions, feedback, observations = archived_reverse_fixture()
    selected = field.select(obs.state, obs.joint_position)
    assert selected['accepted'], selected['status']
    offer = next(c for c in feedback[131]['recovery_options']['candidates'] if c['source'] == 'reverse_observed_worsening_motion')
    assert np.asarray(selected['action'][:3])*.05 == pytest.approx(offer['preview_first_command_delta_m'], abs=1e-11)
    assert field.heading_recovery is not None and field.uncertainty_egress is None
    assert field.goal == pytest.approx(np.asarray(obs.state[:3])+pred['field_intent']['goal_offset_m'])


def test_recorded137_actual_state_drift_gets_checked_cone_command_instead_of_heading_veto():
    field, guard, monitor, obs, pred, actions, feedback, observations = archived_reverse_fixture(133)
    initial_goal = field.goal.copy()
    for n in range(133, 137):
        choice = field.select(obs.state, obs.joint_position)
        assert choice['accepted'], (n, choice['status'])
        after = observations[n]
        # Actual archived feedback exposes transient drift. This is a
        # counterfactual controller check, not a claimed task rollout.
        field.observe(obs.state, choice['action'], after.state)
        images = {'external': np.asarray(decode_image(after.image)), 'wrist': np.asarray(decode_image(after.wrist_image))}
        monitor.observe(images, after.state, choice['action'])
        guard.observe(images, after.state, choice['action'], monitor=monitor)
        obs = after
    choice = field.select(obs.state, obs.joint_position)
    assert choice['accepted'], choice['status']
    constraint = choice['diagnostics']['tracking']['recovery_heading_constraint']
    assert constraint['changed']
    assert max(constraint['heading_errors_rad']) <= guard.reverse_tolerance_rad
    assert np.linalg.norm(np.asarray(choice['action'][:3])*.05) <= .002+1e-12
    assert field.goal == pytest.approx(initial_goal)
    actual = choice['diagnostics']['translation_admission']['diagnostics']['checked_proposal']['delta_xyz_m']
    assert actual == pytest.approx(np.asarray(choice['action'][:3])*.05)
    assert field.pending[1] == pytest.approx(choice['action'])


def test_preview_exception_and_family_binding_preserve_all_original_state():
    guard, monitor, field, state = plant(); promote(field)
    direction = HEADINGS[8]; bind_history(guard, field, direction, state)
    assert field.bind_recovery_route('normal', state)['accepted']
    saved = snapshot(field, guard); binding = copy.deepcopy(field.recovery_route_binding)
    def broken(*args):
        raise RuntimeError('deliberate scene error')
    with pytest.raises(RuntimeError, match='deliberate'):
        field.preview_recovery(state, Q, direction*.008, scene_assessor=broken)
    assert_unchanged(saved, field, guard)
    assert field.recovery_route_binding == binding
    assert not field.bind_recovery_route('reverse_observed', state)['accepted']
    field.configure(state, field.intent, scene_assessment=guard.context,
                    inventory_materialization=field._transit_materialization)
    assert field.recovery_route_binding is None
    selected = field.select(state, Q)
    assert not selected['accepted'] and selected['status'] == 'explicit_recovery_route_unbound'


def test_initial_egress_preview_actual_route_and_debt_are_unchanged():
    from test_visual_field_v24 import upgraded_plant
    field, guard, monitor, inventory, response, plant_, state, observe, obs, bank = upgraded_plant()
    promote(field)
    goal = field.goal.copy(); debt = pickle.dumps(guard.egress_evidence.__dict__)
    preview = field.preview_egress(state, obs.joint_position, goal-state[:3])
    assert preview['accepted'], preview['status']
    assert pickle.dumps(guard.egress_evidence.__dict__) == debt
    assert field.bind_recovery_route('initial_uncertainty_egress', state)['accepted']
    selected = field.select(state, obs.joint_position)
    assert selected['accepted'], selected['status']
    assert selected['action'] == pytest.approx(preview['action'])
    assert field.heading_recovery is None and field.uncertainty_egress is not None
    assert field.goal == pytest.approx(goal) and guard.egress_evidence.spent_m == 0.


def finite_contract_fixture():
    guard, monitor, field, state = plant(); promote(field)
    direction = HEADINGS[8]
    bind_history(guard, field, direction, state)
    configure_reverse(field, guard, state, direction)
    assert field.bind_recovery_route('reverse_observed', state)['accepted']
    return field, guard, state, direction


def test_finite_reverse_contract_is_exact_and_spends_no_allowance():
    field, guard, state, direction = finite_contract_fixture()
    before = snapshot(field, guard)
    proof = field._retreat_geometry_contract(state, state[:3], state[:3]+direction*.002)
    finite = proof['finite_reverse_route']
    assert finite['route_family'] == 'reverse_observed'
    assert finite['state8'] == pytest.approx(state)
    assert finite['end_xyz_m'] == pytest.approx(state[:3]+direction*.002)
    assert finite['origin_xyz_m'] == pytest.approx(field.origin)
    assert finite['goal_xyz_m'] == pytest.approx(field.goal)
    assert finite['max_measured_travel_m'] == .008
    assert finite['remaining_measured_travel_m'] == .008
    assert finite['geometry_sha256'] == proof['geometry_sha256']
    assert finite['observation_epoch'] == guard.sweep_observation_epoch
    assert not finite['physical_budget_replenished'] and not finite['clearance_certified']
    assert_unchanged(before, field, guard)
    action = field.select(state, Q)
    assert action['accepted'], action['status']
    assert field.heading_recovery['route_id'] == finite['route_id']


@pytest.mark.parametrize('fault', ['normal', 'unbound', 'state', 'guard_state',
    'origin', 'goal', 'extent', 'budget', 'exhausted', 'geometry', 'scene', 'contact'])
def test_finite_reverse_extension_never_licenses_stale_swapped_or_normal_contract(fault):
    field, guard, state, direction = finite_contract_fixture()
    start = state[:3].copy(); end = start+direction*.002
    if fault == 'normal': field.recovery_route_binding['recovery_route'] = 'normal'
    elif fault == 'unbound': field.recovery_route_binding = None
    elif fault == 'state': state = state.copy(); state[0] += .0001
    elif fault == 'guard_state': guard.state = state.copy(); guard.state[0] += .0001
    elif fault == 'origin': field.origin = field.origin+direction*.001
    elif fault == 'goal': field.goal = field.goal+direction*.001
    elif fault == 'extent':
        field.goal = field.origin+direction*.009
        field.recovery_route_binding['goal_xyz'] = field.goal.tolist()
    elif fault == 'budget': field.intent['max_travel_m'] = .009
    elif fault == 'exhausted': field.measured_travel_m = .008
    elif fault == 'geometry':
        assert field._prepare_heading_route(state) is None
        field.heading_recovery['geometry_sha256'] = 'changed'
    elif fault == 'scene':
        field.scene_assessor = lambda *_: {'accepted': False, 'diagnostics': {}}
    elif fault == 'contact': field.contact_refinement_enabled = True
    proof = field._retreat_geometry_contract(state, start, end)
    assert 'finite_reverse_route' not in proof


def test_actual_reverse_executor_runs_beyond4mm_checkpoint_without_renewing_allowance():
    guard, monitor, field, measured, pixel = triggered(); promote(field)
    direction = np.array([-1., 0., 0.])
    configure_reverse(field, guard, measured, direction)
    assert field.bind_recovery_route('reverse_observed', measured)['accepted']
    initial = measured.copy(); goal = field.goal.copy()
    for n in range(1, 6):
        choice = field.select(measured, Q)
        assert choice['accepted'], (n, choice['status'])
        delta = np.asarray(choice['action'][:3])*.05
        after = measured.copy(); after[:3] += delta*.5
        field.observe(measured, choice['action'], after)
        images = {'external': pixels(pixel-n), 'wrist': pixels(pixel-n)}
        monitor.observe(images, after, choice['action']); guard.observe(images, after, choice['action'], monitor=monitor)
        measured = after
        assert field.goal == pytest.approx(goal)
    assert field.actions == 5 and np.linalg.norm(measured[:3]-initial[:3]) > .004
    assert guard.retreat_progress['all_trigger_gaps_improved']
    assert len(guard.experiments) == 1
    assert guard.observation_travel_m == pytest.approx(field.measured_travel_m)
    arrived = field.select(measured, Q)
    assert not arrived['accepted'] and arrived['status'] == 'measured_local_goal_arrival'


def test_actual_v25_guard_executes_fixed_reverse_past4mm_without_claiming_gap_progress():
    from gpt_field.visual_translation_guard_v25 import ProjectedTranslationGuardV25
    guard, monitor, field, measured, pixel = triggered(); promote(field)
    defaults = ProjectedTranslationGuardV25().__dict__
    guard.__class__ = ProjectedTranslationGuardV25
    for key, value in defaults.items():
        if key not in guard.__dict__:
            guard.__dict__[key] = value
    # A genuine current reference refresh changes image response coordinates;
    # measured history and projected-worsening debt remain in the same guard.
    refs = {'external': {'update': 'seed', 'moving_id': 'tool', 'moving_kind': 'tool',
        'target_id': 'neighbor', 'goal_offset_uv': [0., 0.], 'seed': {
            'moving_point_uv': [(55+pixel)/223, 65/223],
            'moving_roi_uv': [(43+pixel)/223, 53/223, (67+pixel)/223, 77/223],
            'target_anchor_uv': [110/223, 65/223],
            'target_roi_uv': [98/223, 53/223, 122/223, 77/223],
            'tool_point_uv': None, 'tool_roi_uv': None}}, 'wrist': None}
    initial_images = {'external': pixels(pixel), 'wrist': pixels(pixel)}
    history = copy.deepcopy(guard.worsening_motion)
    debt = guard.observation_travel_m
    monitor.configure(initial_images, refs, measured, intent_id=2)
    guard.configure(initial_images, guard.context, measured, target_id='neighbor', monitor=monitor)
    assert guard.worsening_motion == history and guard.observation_travel_m == debt
    direction = np.array([-1., 0., 0.])
    configure_reverse(field, guard, measured, direction)
    assert field.bind_recovery_route('reverse_observed', measured)['accepted']
    initial = measured.copy(); original_worsening = set(guard.worsening)
    for n in range(5):
        choice = field.select(measured, Q)
        assert choice['accepted'], (n, choice['status'])
        finite = choice['diagnostics']['translation_admission']['diagnostics']['finite_reverse_exposure']
        assert finite['accepted'] and finite['finite_measured_limit_m'] == .008
        after = measured.copy(); after[:3] += np.asarray(choice['action'][:3])*.05*.5
        field.observe(measured, choice['action'], after)
        # A moving tool feature goes vertically in the image. Its box still
        # overlaps the neighbor vertically, so their horizontal gap stays the
        # same. Actual image motion is present without a clearance claim.
        image = pixels(pixel)
        x = 55+pixel
        texture = image[55:76, x-10:x+11].copy()
        image[55:76, x-10:x+11] = 60
        image[56+n:77+n, x-10:x+11] = texture
        images = {'external': image, 'wrist': image}
        monitor.observe(images, after, choice['action'])
        guard.observe(images, after, choice['action'], monitor=monitor)
        measured = after
    assert np.linalg.norm(measured[:3]-initial[:3]) > .004
    assert not guard.retreat_progress['all_trigger_gaps_improved']
    assert guard.worsening == original_worsening
    assert guard.information_checkpoint_m == .004
    assert len(guard.experiments) == 1
    assert guard.observation_travel_m == pytest.approx(field.measured_travel_m)
    assert guard.experiments[0]['measured_travel_m'] == pytest.approx(field.measured_travel_m)


@pytest.mark.parametrize('fault', ['state', 'geometry', 'payload', 'contact', 'unknown', 'scene_missing'])
def test_explicit_reverse_never_overrides_current_constraints(fault):
    guard, monitor, field, measured = plant(); promote(field)
    direction = HEADINGS[9]; bind_history(guard, field, direction, measured)
    if fault == 'state':
        measured = measured.copy(); measured[0] += .001
    elif fault == 'geometry':
        guard.worsening_motion['neighbor']['geometry_sha256'] = 'changed'
    elif fault == 'payload':
        field.contact_geometry._grasp_candidates['payload'] = {'attachment_observed': False}
    elif fault == 'contact':
        field.contact_geometry._record['geometry']['allowed_contact'] = 'fingers'
    elif fault == 'unknown':
        field.intent['unknown_regions'] = [{'id': 'unseen', 'lower_offset_m': [-.002, -.002, .001],
            'upper_offset_m': [.002, .002, .003], 'uncertainty_m': 0.}]
    else:
        field.scene_assessor = lambda s, d: {'accepted': True, 'diagnostics': {}}
    saved = snapshot(field, guard)
    chosen = field.preview_recovery(measured, Q, direction*.008)
    assert not chosen['accepted']
    assert_unchanged(saved, field, guard)
    assert guard.observation_travel_m == 0. and guard.experiments == []


def test_changed_cone_command_is_continuously_checked_not_licensed_by_original_chord():
    from gpt_field.visual_field_v25 import _RecoveryConeTracker
    tracker = _RecoveryConeTracker(lookahead_m=.002, max_command_m=.002, endpoint_tolerance_m=.0001)
    tracker.reset([[0., 0., 0.], [.008, 0., 0.]])
    tracker.route_direction = np.array([1., 0., 0.])
    tracker.reverse_axes = np.array([[1., 0., 0.]])
    tracker.reverse_tolerance_rad = np.pi/6
    start = np.array([0., .0015, 0.])
    # The original lookahead correction is 36.87 degrees. Place a thin
    # hypothetical obstacle across the projected30degree command, while the
    # original desired chord remains clear. Every check is exact, no sampling.
    calls = []
    class Field:
        def query(self, point): return 0.
        def segment_valid(self, a, b):
            calls.append((np.asarray(a).copy(), np.asarray(b).copy()))
            delta = np.asarray(b)-a
            if delta[0] <= 0.: return True
            fraction = .0016/delta[0]
            if not 0. <= fraction <= 1.: return True
            y = a[1]+fraction*delta[1]
            return not .00055 < y < .00065
    delta, detail = tracker.command(start, Field(), [-1.]*3, [1.]*3)
    assert delta == (0., 0., 0.)
    assert detail['status'] == 'recovery_constrained_segment_blocked'
    assert len(calls) >= 3
