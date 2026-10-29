"""Actual RGB tracking, measurable motion, escape debt and archive regressions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field.visual_field import _gray
from gpt_field.visual_monitor_v5 import LocalFeatureBundle
from gpt_field.visual_translation_guard_v11 import ProjectedTranslationGuardV11


TEXTURES = [np.random.default_rng(130+i).integers(0, 256, (21, 21, 3), dtype=np.uint8) for i in range(2)]


def pixels(x=0, *, neighbor_x=0, common=0, missing=False):
    im = np.full((224, 224, 3), 60, np.uint8)
    for index, (cx, cy) in enumerate([(55+int(x)+int(common), 65), (110+int(neighbor_x)+int(common), 65)]):
        if index == 0 and missing:
            continue
        im[cy-10:cy+11, cx-10:cx+11] = TEXTURES[index]
    return im


def context(*, mode="transit", tool=(43, 53, 67, 77), obstacle=(98, 53, 122, 77), identity="neighbor"):
    return {"purpose": "task_translation", "translation_mode": mode,
            "intended_contact_id": "target" if mode == "interaction" else None,
            "obstacle_assessment": "present", "tool_external_roi_uv": (np.array(tool)/223).tolist(),
            "nearby_obstacles": [{"id": identity, "external_roi_uv": (np.array(obstacle)/223).tolist()}]}


def plant(ctx=None):
    im = pixels();state = np.zeros(8)
    track = LocalFeatureBundle(_gray(im), [55/223, 65/223], "distal_tool", roi_uv=np.array([43, 53, 67, 77])/223)
    monitor = SimpleNamespace(refs={"external": {"moving_kind": "tool"}}, tracks={"moving": track})
    guard = ProjectedTranslationGuardV11()
    guard.configure({"external": im, "wrist": im}, ctx or context(), state, target_id="target", monitor=monitor)
    return guard, monitor, state


def advance(guard, monitor, state, x, *, observed_x=None, common=0, rotation=0., missing=False):
    image = pixels(x if observed_x is None else observed_x, common=common, missing=missing)
    monitor.tracks["moving"].track(_gray(image))
    new = state.copy();new[0] = x*.001;new[3] = rotation
    guard.observe({"external": image, "wrist": image}, new, [0., 0., 0., 0., 0., 0., 0.], monitor=monitor)
    return new


def test_continuous_far_field_is_not_cut_into_model_calls():
    guard, monitor, state = plant()
    for i in range(1, 19):
        result = guard.assess([.001, 0, 0], monitor=monitor)
        assert result["accepted"] and result["cap_m"] is None
        state = advance(guard, monitor, state, i)
    assert guard.response_ready
    assert guard.events == 0


def test_supported_lookahead_halts_before_projected_intersection():
    guard, monitor, state = plant()
    for i in range(1, 32):
        decision = guard.assess([.001, 0, 0], monitor=monitor)
        if not decision["accepted"]:
            break
        state = advance(guard, monitor, state, i)
    assert i <= 30
    assert decision["reason"] == "translation_projected_neighbor_conflict"
    assert guard.summary()["neighbors"][0]["gap_px"] > 0
    assert not decision["diagnostics"]["clearance_certified"]


def test_same_overlap_supports_away_but_never_a_growing_direction():
    guard, monitor, state = plant()
    for i in range(1, 30):
        state = advance(guard, monitor, state, i)
    ctx = context(mode="separate", tool=(72, 53, 96, 77))
    im=pixels(29);guard.configure({"external":im,"wrist":im},ctx,state,target_id="target",monitor=monitor)
    assert guard.assess([-.001, 0, 0], monitor=monitor)["reason"] == "projected_separation_supported"
    result=guard.assess([.001, 0, 0], monitor=monitor)
    assert not result["accepted"] and result["reason"] == "translation_separation_not_supported_away"
    cap=guard.assess([-.01,0,0],monitor=monitor)
    assert cap["accepted"] and .001 < cap["cap_m"] <= .00150001
    assert guard.assess([-cap["cap_m"],0,0],monitor=monitor)["accepted"]


def test_interaction_target_does_not_exempt_an_unintended_neighbor():
    ctx=context(mode="interaction",tool=(43,53,99,77))
    guard, monitor, _=plant(ctx)
    assert not guard.assess([.001,0,0],monitor=monitor)["accepted"]
    assert guard.summary()["conflicting_obstacle_ids"] == ["neighbor"]


def test_cold_separation_prior_uses_finite_measured_budget_not_request_count():
    ctx=context(mode="separate",tool=(43,53,99,77))
    guard, monitor, state=plant(ctx)
    for i in range(4):
        decision=guard.assess([-.002,0,0],monitor=monitor,nominal_away=True)
        assert decision["accepted"] and decision["cap_m"] <= .002
        # Wrong/static feature has no predictive response despite real motion.
        state=advance(guard,monitor,state,-(i+1)*1.2,observed_x=0)
        changed=copy.deepcopy(ctx);changed["nearby_obstacles"][0]["id"]="rename"+str(i)
        im=pixels();guard.configure({"external":im,"wrist":im},changed,state,target_id="target",monitor=monitor)
    used=guard.cold_separation_travel_m
    assert used == pytest.approx(.0048)
    result=guard.assess([-.002,0,0],monitor=monitor,nominal_away=True)
    assert not result["accepted"] and result["reason"] == "translation_cold_separation_budget_exhausted"
    assert result["cap_m"] == 0
    assert not guard.response_ready


def test_cold_prior_worsening_is_contradicted_without_reset_by_new_context():
    ctx=context(mode="separate",tool=(43,53,99,77));guard,monitor,state=plant(ctx)
    assert guard.assess([.001,0,0],monitor=monitor,nominal_away=True)["accepted"]
    state=advance(guard,monitor,state,1)
    im=pixels(1);guard.configure({"external":im,"wrist":im},ctx,state,target_id="target",monitor=monitor)
    result=guard.assess([.001,0,0],monitor=monitor,nominal_away=True)
    assert not result["accepted"] and result["reason"] == "translation_cold_separation_contradicted"


def test_no_prior_means_no_cold_separation_permission():
    guard,monitor,_=plant(context(mode="separate",tool=(43,53,99,77)))
    assert guard.assess([-.001,0,0],monitor=monitor)["reason"] == "translation_separation_direction_unobserved"


@pytest.mark.parametrize("kind", ["lost", "rotation", "camera"])
def test_lost_tool_domain_rotation_and_camera_flow_require_new_evidence(kind):
    guard,monitor,state=plant()
    state=advance(guard,monitor,state,1,missing=kind=="lost",rotation=.03 if kind=="rotation" else 0,
                  common=3 if kind=="camera" else 0)
    result=guard.assess([.001,0,0],monitor=monitor)
    assert not result["accepted"]
    expected={"lost":"translation_tool_correspondence_lost", "rotation":"translation_tool_configuration_changed", "camera":"translation_scene_motion_unresolved"}
    assert result["reason"] == expected[kind]


def test_reconfigure_cannot_delete_or_rename_away_existing_conflict():
    guard,monitor,state=plant(context(tool=(43,53,99,77)))
    ctx=context();ctx.update(obstacle_assessment="none_relevant",nearby_obstacles=[])
    image=pixels();guard.configure({"external":image,"wrist":image},ctx,state,target_id="target",monitor=monitor)
    assert not guard.assess([.001,0,0],monitor=monitor)["accepted"]
    assert guard.summary()["conflicting_obstacle_ids"] == ["neighbor"]


def test_missing_texture_is_not_obstacle_disappearance_near_tool():
    guard,monitor,_=plant(context(tool=(43,53,99,77)))
    guard.neighbors[0]["track"].reliable=False
    assert guard.assess([.001,0,0],monitor=monitor)["reason"] == "translation_nearby_obstacle_track_unavailable"


def test_v10_saved_rgb_blocks_projected_approach_before_contact():
    from gpt_field.observation import decode_image
    from gpt_field.visual_monitor_v10 import RoleAwareVisualMonitorV10
    path=Path(__file__).parents[1]/"runs/visual_field_v10_zero_shot_001/trials/bowl_original_zero_hybrid_v10/trace.jsonl"
    if not path.exists():pytest.skip("local archived run not available")
    monitor=RoleAwareVisualMonitorV10();guard=ProjectedTranslationGuardV11();obs=None;blocked=[]
    def views(o):return {"external":np.asarray(decode_image(o["observation/image"])),"wrist":np.asarray(decode_image(o["observation/wrist_image"]))}
    for line in path.open():
        row=json.loads(line);step=row.get("step",-1)
        if row["event"]=="reset":obs=row["observation"]
        elif row["event"]=="action":
            pred=row["diagnostics"].get("prediction")
            if pred:
                monitor.configure(views(obs),pred["visual_references"],obs["observation/state"],intent_id=row["diagnostics"]["chunk_index"])
                ctx=copy.deepcopy(pred["motion_context"]);ctx.update(translation_mode="transit",intended_contact_id=None)
                guard.configure(views(obs),ctx,obs["observation/state"],target_id=pred["target_id"],monitor=monitor)
            decision=guard.assess(np.asarray(row["action"][:3])*.05,monitor=monitor)
            if decision["reason"]=="translation_projected_neighbor_conflict":blocked.append(step)
        elif row["event"]=="executed":
            obs=row["observation"]
            # Recorded continuation AFTER counterfactual veto is analysis only.
            monitor.observe(views(obs),obs["observation/state"],row["action"])
            guard.observe(views(obs),obs["observation/state"],row["action"],monitor=monitor)
    assert blocked and min(blocked) < 119
    assert 119 in blocked and 138 in blocked


def test_large_unsupported_transit_command_is_capped_before_jumping_a_narrow_gap():
    guard,monitor,state=plant()
    for i in range(1,25):state=advance(guard,monitor,state,i)
    assert guard.response_ready
    decision=guard.assess([.02,0,0],monitor=monitor)
    assert decision["accepted"] and .001 < decision["cap_m"] <= .00150001
    assert decision["reason"] == "projected_translation_response_scale_cap"
    # Admission of the capped command requires a second actual-size check.
    checked=guard.assess([decision["cap_m"],0,0],monitor=monitor)
    assert checked["cap_m"] is None
    far=plant()[0]
    assert not far.response_ready


def test_fresh_context_after_configuration_change_discards_old_motion_mapping():
    guard,monitor,state=plant()
    for i in range(1,9):state=advance(guard,monitor,state,i)
    assert guard.response_ready
    state=advance(guard,monitor,state,9,rotation=.03)
    assert guard.domain_invalid == "translation_tool_configuration_changed"
    im=pixels(9)
    guard.configure({"external":im,"wrist":im},context(tool=(52,53,76,77)),state,target_id="target",monitor=monitor)
    assert not guard.response_ready and not guard.response.samples
    assert guard.cold_separation_travel_m == 0


def test_tool_disappearing_during_cold_escape_spends_debt_and_hands_off():
    guard,monitor,state=plant(context(mode="separate",tool=(43,53,99,77)))
    assert guard.assess([-.001,0,0],monitor=monitor,nominal_away=True)["accepted"]
    monitor.tracks["moving"] = None
    state[0] = -.001
    im=pixels()
    guard.observe({"external":im,"wrist":im},state,[0.]*7,monitor=monitor)
    assert guard.cold_separation_travel_m == pytest.approx(.001)
    assert guard.assess([-.001,0,0],monitor=monitor,nominal_away=True)["reason"] == "translation_tool_correspondence_lost"


def test_declared_whole_tool_box_must_contain_actual_tracked_tool_reference():
    guard,monitor,_=plant(context(tool=(130,120,150,150)))
    result=guard.assess([.001,0,0],monitor=monitor)
    assert not result["accepted"]
    assert result["reason"] == "translation_tool_reference_outside_declared_envelope"
