from pathlib import Path
import sys

import mujoco
import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from native_distance_evaluation import NativeDistanceGroundTruthEvaluator, OriginalGroundTruthEvaluator
from robot_geometry import collision_mesh

CASES = [
    ("drawer003_closed_step60", [-0.0067498633465514195, 0.1433218678017, -0.011506449273037388, -2.4210588150882306, -0.0019494314421893962, 2.2188079722269975, 0.7710865825626639], ["gripper0_finger1_collision", "wine_bottle_1_g14"]),
    ("drawer003_open_step552", [-0.39243176590893236, 0.5169359511551379, -0.0043780579924078905, -1.5997556967619508, 0.005287650813237378, 2.065451833739141, 0.7816515284884364], ["gripper0_finger1_collision", "wooden_cabinet_1_g9"]),
]


@pytest.mark.parametrize("snapshot,q,names", CASES)
def test_native_distance_consistent_with_independent_exact_separation(snapshot, q, names):
    source = HERE.parents[1] / "scene_test/rgb_scene/references_004/snapshots" / snapshot / "evaluator"
    truth = NativeDistanceGroundTruthEvaluator(source)
    original_flags = int(truth.model.opt.enableflags)
    truth._set_q(q)
    ids = [mujoco.mj_name2id(truth.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in names]
    vertices = []
    for gid in ids:
        local, _ = collision_mesh(truth.model, gid)
        vertices.append(local @ truth.data.geom_xmat[gid].reshape(3, 3).T + truth.data.geom_xpos[gid])
    gaps = np.maximum(vertices[1].min(0) - vertices[0].max(0), vertices[0].min(0) - vertices[1].max(0))
    assert gaps.max() > .004
    distances = []
    for cap in [.05, .1, 1.]:
        truth.distance_cap_m = cap
        distance, pair = truth._minimum_distance([tuple(ids)])
        assert pair == names
        assert distance >= gaps.max() - 1e-8
        assert int(truth.model.opt.enableflags) == original_flags
        distances.append(distance)
    np.testing.assert_allclose(distances, distances[0], atol=1e-10, rtol=0)
    before = truth.check(q, distances=False)
    after = truth.check(q, distances=True)
    assert before["valid"] and after["valid"]
    assert before["contacts"] == after["contacts"] == []
    assert int(truth.model.opt.enableflags) == original_flags


def test_original_flags_restored_even_if_distance_query_raises(monkeypatch):
    source = HERE.parents[1] / "scene_test/rgb_scene/references_004/snapshots" / CASES[0][0] / "evaluator"
    truth = NativeDistanceGroundTruthEvaluator(source)
    original = int(truth.model.opt.enableflags)
    def fail(self, pairs):
        assert int(self.model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_NATIVECCD)
        raise RuntimeError("injected query failure")
    monkeypatch.setattr(OriginalGroundTruthEvaluator, "_minimum_distance", fail)
    with pytest.raises(RuntimeError, match="injected"):
        truth._minimum_distance([])
    assert int(truth.model.opt.enableflags) == original


def test_native_beyond_cutoff_retains_declared_cap():
    source = HERE.parents[1] / "scene_test/rgb_scene/references_004/snapshots" / CASES[0][0] / "evaluator"
    truth = NativeDistanceGroundTruthEvaluator(source, distance_cap_m=.001)
    truth._set_q(CASES[0][1])
    pair = tuple(mujoco.mj_name2id(truth.model, mujoco.mjtObj.mjOBJ_GEOM, name) for name in CASES[0][2])
    distance, closest = truth._minimum_distance([pair])
    assert distance == .001 and closest is None
