from pathlib import Path
import sys
import numpy as np
import pytest
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ray_evaluation import raycast_depth,depth_hit_scores,independent_first_hit


def plane(z,size=1):
    return trimesh.Trimesh(vertices=[[-size,-size,z],[size,-size,z],[size,size,z],[-size,size,z]],faces=[[0,1,2],[0,2,3]],process=False)


def test_first_hit_not_far_hit_and_z_not_ray_length():
    mesh=trimesh.util.concatenate([plane(2),plane(1)])
    K=np.array([[2.,0,1],[0,2.,1],[0,0,1.]])
    depth,report=raycast_depth(mesh,K,(3,3))
    assert np.allclose(depth,1,atol=1e-12)
    assert report['independent_validation']['passed']


def test_single_ray_and_miss_are_explicit():
    depth,_=raycast_depth(plane(.75),np.eye(3),(1,1))
    assert depth[0,0]==pytest.approx(.75,abs=1e-12)
    K=np.array([[1.,0,100],[0,1.,100],[0,0,1.]])
    missed,_=raycast_depth(plane(.75),K,(1,1))
    assert np.isnan(missed[0,0])


def test_conditional_accuracy_cannot_hide_misses():
    pred=np.array([[1.,np.nan],[1.002,2.]])
    truth=np.ones((2,2));mask=np.ones((2,2),bool)
    result=depth_hit_scores(pred,truth,mask)
    assert result['coverage']==.75
    assert result['within_threshold_all_reference_pixels']['1mm']==.25
    assert result['miss_pixels']==1


def test_independent_analytic_slanted_triangle():
    tri=np.array([[[-1,-1,1],[1,-1,2],[0,1,1.5]]],float)
    assert independent_first_hit(tri,np.array([0.,0.,1.]))==pytest.approx(1.5)


def test_native_slanted_plane_reversed_winding_and_empty_geometry():
    tri=np.array([[-1,-1,1],[1,-1,2],[0,1,1.5]],float)
    for face in ([[0,1,2]],[[2,1,0]]):
        mesh=trimesh.Trimesh(vertices=tri,faces=face,process=False)
        depth,report=raycast_depth(mesh,np.eye(3),(1,1))
        assert depth[0,0]==pytest.approx(1.5,abs=2e-7)
        assert report['independent_validation']['oracle_hit_count']==1
    empty=trimesh.Trimesh(vertices=np.empty((0,3)),faces=np.empty((0,3),int),process=False)
    depth,report=raycast_depth(empty,np.eye(3),(3,3))
    assert np.isnan(depth).all() and report['hits']==0
