"""agent_SpeedUp_v2 generic record-only contact monitor on a fake MuJoCo-like model (no simulator)."""

from types import SimpleNamespace

import numpy as np
import pytest

from gpt_field import agent_speedup_v2_contacts as contacts

BODIES = ["world", "robot0_base", "gripper0_root", "gripper0_finger", "table", "bowl_1_main", "plate_1_main",
          "mug_1_main", "cabinet_1_main", "cabinet_1_drawer"]
PARENT = [0, 0, 1, 2, 0, 0, 0, 0, 0, 8]
JOINTS = [0, 0, 0, 1, 0, 1, 1, 1, 0, 1]
# geom: (name, body)
GEOMS = [("hand_collision", 2), ("left_pad", 3), ("right_pad", 3), ("table_top", 4), ("bowl_g", 5), ("plate_g", 6),
         ("mug_g", 7), ("cabinet_static", 8), ("drawer_handle", 9), ("link3", 1)]
G = {name: i for i, (name, _) in enumerate(GEOMS)}


class FakeModel:
    ngeom = len(GEOMS)
    geom_contype = [1] * len(GEOMS)
    geom_conaffinity = [1] * len(GEOMS)
    geom_bodyid = [b for _, b in GEOMS]
    body_jntnum = JOINTS
    body_parentid = PARENT
    site_size = [np.array([0.05, 0.05, 0.01])]

    def body_name2id(self, name):
        return BODIES.index(name)

    def geom_id2name(self, i):
        return GEOMS[i][0]

    def body_id2name(self, i):
        return BODIES[i]

    def site_name2id(self, name):
        return 0


class FakeSim:
    def __init__(self):
        self.model = FakeModel()
        self.data = SimpleNamespace(ncon=0, contact=[], site_xpos=[np.zeros(3)], site_xmat=[np.eye(3).ravel()],
                                    qacc_warmstart=np.array([1.0, 2.0, 3.0]))
        self.raw_steps = 0
        self.step = self._step
        self.forwards = 0

    def _step(self):
        self.raw_steps += 1

    def forward(self):
        self.forwards += 1
        self.data.qacc_warmstart[:] = -7.0      # MuJoCo's forward rewrites the solver warm start


def make_env(goal=(("on", "bowl_1", "plate_1"),), sites=None):
    sim = FakeSim()
    obj = lambda root: SimpleNamespace(root_body=root)
    gripper = SimpleNamespace(root_body="gripper0_root",
                              important_geoms={"left_fingerpad": ["left_pad"], "right_fingerpad": ["right_pad"],
                                               "left_finger": [], "right_finger": []})
    core = SimpleNamespace(
        objects_dict={"bowl_1": obj("bowl_1_main"), "plate_1": obj("plate_1_main"), "mug_1": obj("mug_1_main")},
        fixtures_dict={"cabinet_1": obj("cabinet_1_main")},
        robots=[SimpleNamespace(robot_model=SimpleNamespace(root_body="robot0_base"), gripper=gripper)],
        parsed_problem={"goal_state": [list(a) for a in goal]}, object_sites_dict=dict(sites or {}))
    return SimpleNamespace(sim=sim, env=core)


def contact(a, b, normal=(0.0, 0.0, 1.0), dist=-0.001, pos=(0.0, 0.0, 0.0), margin=0.0):
    return SimpleNamespace(geom1=G[a], geom2=G[b], dist=dist, frame=np.array([*normal, 0, 0, 0, 0, 0, 0], float),
                           pos=np.array(pos, float), includemargin=margin)


def classify(monitor, a, b, **kwargs):
    c = contact(a, b, **kwargs)
    return monitor.classify(monitor.geometry[c.geom1], monitor.geometry[c.geom2], c)


def test_roles_from_registries_only():
    monitor = contacts.GenericContactMonitor(make_env())
    roles = {record["name"]: record["role"] for record in monitor.geometry.values()}
    assert roles == {"hand_collision": "robot", "left_pad": "robot_finger", "right_pad": "robot_finger",
                     "table_top": "arena", "bowl_g": "object", "plate_g": "object", "mug_g": "object",
                     "cabinet_static": "fixture_static", "drawer_handle": "fixture_part", "link3": "robot"}
    assert monitor.goal_objects == {"bowl_1", "plate_1"}


def test_robot_contact_classes():
    m = contacts.GenericContactMonitor(make_env())
    assert classify(m, "hand_collision", "link3") == "robot_internal"
    assert classify(m, "left_pad", "bowl_g") == "finger_object"
    assert classify(m, "left_pad", "mug_g") == "robot_object"                  # goal-aware: non-goal object
    assert classify(m, "hand_collision", "mug_g") == "robot_object"            # never-touched distractor
    assert classify(m, "hand_collision", "bowl_g") == "robot_nonfinger_manipulated_object"   # goal object
    assert classify(m, "left_pad", "drawer_handle") == "finger_articulated_part"
    assert classify(m, "hand_collision", "drawer_handle") == "robot_scene"
    assert classify(m, "left_pad", "cabinet_static") == "robot_scene"
    assert classify(m, "link3", "table_top") == "robot_scene"
    blind = contacts.GenericContactMonitor(make_env(), goal_aware=False)
    assert classify(blind, "left_pad", "mug_g") == "finger_object"
    assert classify(blind, "hand_collision", "bowl_g") == "robot_object"      # not yet touched


def test_manipulated_object_support_and_scene():
    m = contacts.GenericContactMonitor(make_env())
    assert classify(m, "bowl_g", "table_top", normal=(0, 0, -1)) == "passive_scene"   # not manipulated
    m.touched["bowl_1"] = m._substep_clock
    assert classify(m, "bowl_g", "table_top", normal=(0, 0, -1)) == "manipulated_object_support"
    assert classify(m, "table_top", "bowl_g", normal=(0, 0, 1)) == "manipulated_object_support"
    tilt = np.radians(40)
    assert classify(m, "bowl_g", "table_top", normal=(-np.sin(tilt), 0, -np.cos(tilt))) == "manipulated_object_scene"
    assert classify(m, "bowl_g", "mug_g", normal=(0, 0, -1)) == "manipulated_object_scene"   # on a distractor
    assert classify(m, "bowl_g", "plate_g", normal=(0, 0, -1)) == "manipulated_object_support"
    m._substep_clock += contacts.MANIPULATED_MEMORY_SUBSTEPS + 1
    assert classify(m, "bowl_g", "table_top", normal=(0, 0, -1)) == "passive_scene"   # memory expired


def test_inspect_counts_first_events_and_hold():
    env = make_env()
    m = contacts.GenericContactMonitor(env)
    env.sim.data.contact = [contact("left_pad", "bowl_g"), contact("right_pad", "bowl_g"),
                            contact("link3", "table_top"), contact("hand_collision", "link3", dist=0.002)]
    env.sim.data.ncon = len(env.sim.data.contact)
    m.inspect(phase="initial_state")
    snap = m.snapshot()
    assert snap["held_objects"] == ["bowl_1"] and snap["manipulated_objects"] == ["bowl_1"]
    assert snap["detected"] and snap["first_event"]["class"] == "robot_scene"
    assert snap["counts"]["points_finger_object"] == 2 and snap["counts"]["points_robot_scene"] == 1
    assert snap["counts"]["positive_distance_contacts_ignored"] == 1
    assert snap["counts"]["forbidden_contact_points"] == 1
    assert snap["mode"] == "generic_contact_v2" and snap["collision_policy"] == "record_only"
    assert snap["stop_on_collision"] is False and snap["success_qualification"] == "raw_libero_goal"
    with pytest.raises(ValueError):
        env.sim.data.contact = [contact("left_pad", "bowl_g", dist=float("nan"))]
        env.sim.data.ncon = 1
        m.inspect(phase="x")


def test_step_hook_protocol_and_record_only():
    env = make_env()
    m = contacts.GenericContactMonitor(env)
    m.attach()
    with pytest.raises(RuntimeError):
        m.attach()
    with pytest.raises(RuntimeError):
        env.sim.step()                       # outside an acknowledged action
    m.begin_action(0)
    with pytest.raises(RuntimeError):
        m.finish_action()                    # no physics substep observed
    for _ in range(25):
        env.sim.step()
    m.finish_action()
    assert env.sim.raw_steps == 25 and m.counts["physics_substeps"] == 25 and env.sim.forwards == 1
    # CW-1: the endpoint refresh does not perturb the next step's solver warm start
    assert env.sim.data.qacc_warmstart.tolist() == [1.0, 2.0, 3.0]
    m.begin_action(1)
    env.sim.step()
    m.finish_action(refresh=False)           # a chained monitor already refreshed
    assert env.sim.forwards == 1 and m.counts["checks"] == 25 + 1 + 1 + 1
    assert m._action_index is None


def test_unmapped_contacts_counted():
    env = make_env()
    m = contacts.GenericContactMonitor(env)
    env.sim.data.contact = [SimpleNamespace(geom1=99, geom2=0, dist=-0.001, frame=np.zeros(9), pos=np.zeros(3),
                                            includemargin=0.0)]
    env.sim.data.ncon = 1
    m.inspect(phase="x")
    assert m.counts["unmapped_contact_points"] == 1 and not m.snapshot()["detected"]


def test_forward_keeping_warmstart():
    sim = FakeSim()
    contacts.forward_keeping_warmstart(sim)
    assert sim.forwards == 1 and sim.data.qacc_warmstart.tolist() == [1.0, 2.0, 3.0]


def test_margin_contacts_are_active_up_to_includemargin():
    """CW-2: a margin geom (microwave class, 1 mm) touches at a small positive distance; it counts, as in truth."""
    env = make_env()
    m = contacts.GenericContactMonitor(env)
    env.sim.data.contact = [contact("hand_collision", "cabinet_static", dist=0.0005, margin=0.001),
                            contact("link3", "table_top", dist=0.0015, margin=0.001),
                            contact("link3", "table_top", dist=0.0005)]
    env.sim.data.ncon = 3
    m.inspect(phase="x")
    counts = m.snapshot()["counts"]
    assert counts["points_robot_scene"] == 1 and counts["active_positive_distance_contacts"] == 1
    assert counts["positive_distance_contacts_ignored"] == 2


def test_finger_on_distractor_never_makes_it_manipulated():
    """CW-6: under the goal-aware tier a finger on a non-goal object is robot_object and the distractor stays
    passive; a palm contact uses the same memory window as support."""
    env = make_env()
    m = contacts.GenericContactMonitor(env)
    env.sim.data.contact = [contact("left_pad", "mug_g")]
    env.sim.data.ncon = 1
    m.inspect(phase="x")
    assert "mug_1" not in m.touched and m.counts["points_robot_object"] == 1
    assert classify(m, "mug_g", "table_top", normal=(0, 0, -1)) == "passive_scene"
    blind = contacts.GenericContactMonitor(make_env(), goal_aware=False)
    blind.touched["mug_1"] = blind._substep_clock
    assert classify(blind, "hand_collision", "mug_g") == "robot_nonfinger_manipulated_object"
    blind._substep_clock += contacts.MANIPULATED_MEMORY_SUBSTEPS + 1
    assert classify(blind, "hand_collision", "mug_g") == "robot_object"      # released long ago: a distractor


def test_goal_regions_cover_in_and_hosted_on_regions():
    """CW-4: In regions (3-D box) and On regions hosted by a registered fixture/object (xy footprint); table
    zones (host = arena) are not goal regions of the tier."""
    sites = {"cabinet_1_top_region": SimpleNamespace(parent_name="cabinet_1"),
             "cabinet_1_cook_region": SimpleNamespace(parent_name="cabinet_1"),
             "table_zone": SimpleNamespace(parent_name=None)}
    objects, regions, predicates = contacts.goal_objects_and_regions(make_env(
        goal=(("on", "bowl_1", "cabinet_1_cook_region"), ("on", "plate_1", "table_zone")), sites=sites).env)
    assert regions == {"cabinet_1_cook_region": "cabinet_1"} and predicates == {"cabinet_1_cook_region": "on"}
    assert objects == {"bowl_1", "plate_1"}
    env = make_env(goal=(("on", "bowl_1", "cabinet_1_cook_region"),), sites=sites)
    m = contacts.GenericContactMonitor(env)
    m.touched["bowl_1"] = m._substep_clock
    # site box half-size (0.05, 0.05, 0.01) at the origin: an On region tests the xy footprint only
    assert classify(m, "bowl_g", "cabinet_static", normal=(0, 0, -1), pos=(0.0, 0.0, 0.2)) == \
        "manipulated_object_support"
    assert classify(m, "bowl_g", "cabinet_static", normal=(0, 0, -1), pos=(0.2, 0.0, 0.0)) == \
        "manipulated_object_scene"
    env = make_env(goal=(("in", "bowl_1", "cabinet_1_top_region"),), sites=sites)
    m = contacts.GenericContactMonitor(env)
    m.touched["bowl_1"] = m._substep_clock
    assert classify(m, "bowl_g", "cabinet_static", normal=(0, 0, -1), pos=(0.0, 0.0, 0.2)) == \
        "manipulated_object_scene"                                   # In: the 3-D box (above the cavity)
    assert classify(m, "bowl_g", "cabinet_static", normal=(0, 0, -1), pos=(0.0, 0.0, 0.0)) == \
        "manipulated_object_support"
