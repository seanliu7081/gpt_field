"""Independent geometry/metric and provenance checks for new RGB stages."""
from pathlib import Path
import sys
import numpy as np
import pytest
HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
from field_rgb import visible_mesh
from evaluate_rgb import mask_cells,restrict_to_image_region,depth_metrics
from rgb_metrics import scores,compare_meshes
from build_fields import validate_prediction_manifest


def test_exact_pixel_region_clipping_preserves_slanted_plane_area():
    depth=np.full((9,9),.4)
    K=np.array([[20.,0,4],[0,20.,4],[0,0,1.]])
    mesh,_=visible_mesh(depth,K)
    # An L-shaped region, exact pixel-cell area known independently.
    cells=np.zeros((8,8),bool);cells[:4,:]=True;cells[:,0:2]=True
    clipped=restrict_to_image_region(mesh,K,cells)
    assert clipped.area==pytest.approx(cells.sum()*(.4/20)**2,abs=1e-12)
    assert np.allclose(clipped.vertices[:,2],.4)
    # Retriangulate plane coarsely: clipping must give same physical region.
    import trimesh
    coarse=trimesh.Trimesh(vertices=mesh.bounds[[0,1,1,0],:] if False else [[-.08,-.08,.4],[.08,-.08,.4],[.08,.08,.4],[-.08,.08,.4]],faces=[[0,1,2],[0,2,3]],process=False)
    assert restrict_to_image_region(coarse,K,cells).area==pytest.approx(clipped.area,abs=1e-12)


def test_mask_domain_identical_for_all_resolutions():
    mask=np.ones((5,5),bool);mask[2,2]=False
    cells=mask_cells(mask)
    assert cells.sum()==12
    assert not cells[1:3,1:3].any()


def test_empty_reconstruction_has_zero_recall_not_missing_entry():
    result=scores(np.empty(0),np.full(100,np.inf))
    assert result['thresholds']['1mm']=={'precision':0.,'recall':0.,'f1':0.}
    assert result['reference_to_prediction']['sample_count']==100
    assert result['symmetric_mean_m'] is None


def test_known_offset_surface_distances_and_both_oracle_checks():
    import trimesh
    from rgb_metrics import mesh_distance,validate_distance
    reference=trimesh.Trimesh(vertices=[[0,0,0],[1,0,0],[0,1,0]],faces=[[0,1,2]],process=False)
    points=np.array([[.2,.2,.003],[.1,.1,.005]])
    assert np.allclose(mesh_distance(reference,points),[.003,.005],atol=1e-12)
    assert validate_distance(reference,points)['passed']


def test_oracle_scale_diagnostic_does_not_mutate_primary_depth():
    predicted=np.full((3,3),2.);reference=np.ones((3,3));before=predicted.copy()
    K=np.eye(3);valid=np.ones((3,3),bool)
    result=depth_metrics(predicted,reference,valid,K,K,.5)
    assert result['mae_m']==1
    assert result['oracle_scale_depth_only']['mae_m']==0
    assert np.array_equal(predicted,before)


def test_prediction_manifest_fails_empty_or_false_integrity_and_duplicate_names():
    groups=('sources_unchanged','model_files_unchanged','frozen_artifacts_unchanged','inputs_unchanged','predictions_unchanged')
    manifest={'status':'complete','predictions':[{'name':name} for name in ('bowl002_initial','drawer003_closed_step60','drawer003_open_step552')],**{g:{'one':True} for g in groups}}
    validate_prediction_manifest(manifest)
    for bad in ({},{'one':False}):
        with pytest.raises(ValueError):
            validate_prediction_manifest({**manifest,'sources_unchanged':bad})
    duplicate={**manifest,'predictions':manifest['predictions'][:2]+[manifest['predictions'][0]]}
    with pytest.raises(ValueError):
        validate_prediction_manifest(duplicate)


def test_independent_block_workers_match_serial_saved_arrays_and_mesh(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from build_fields import initialize_worker,build_worker
    from field_rgb import plan_blocks,ProjectiveField
    depth=np.tile(np.linspace(.295,.335,9),(9,1))
    K=np.array([[30.,0,4],[0,30.,4],[0,0,1.]])
    blocks=plan_blocks(depth,K,tiles=1)
    results=[]
    for workers in (1,2):
        out=tmp_path/f'w{workers}';out.mkdir();fields=out/'fields';fields.mkdir()
        with ProcessPoolExecutor(max_workers=workers,initializer=initialize_worker,initargs=(depth,K,.004,1000000,fields,out)) as pool:
            results.append(list(pool.map(build_worker,blocks)))
    for index,block in enumerate(blocks):
        first=ProjectiveField.load(tmp_path/'w1'/'fields'/f'{block["id"]}.npz')
        second=ProjectiveField.load(tmp_path/'w2'/'fields'/f'{block["id"]}.npz')
        assert np.array_equal(first.values,second.values)
        assert np.array_equal(first.observed,second.observed)
        assert np.array_equal(first.cell_valid,second.cell_valid)
        assert np.array_equal(results[0][index][1].vertices,results[1][index][1].vertices)
        assert np.array_equal(results[0][index][1].faces,results[1][index][1].faces)


def test_evaluator_requires_upstream_integrity_and_exact_rgb_state_gates():
    from evaluate_rgb import validate_upstream
    import copy
    field={'status':'complete','integrity':{'sources':[True],'inputs':[True],'artifacts':[True],'settings':True}}
    metrics={'status':'complete','snapshots':[{'name':'bowl002_initial'}],'configurations':[{'name':'bowl002_initial_h4mm','status':'ok'}]}
    references={'status':'complete','script_unchanged':True,'capture_unchanged':True,'artifacts_unchanged':{'one':True},'snapshots':[]}
    for name in ('bowl002_initial','drawer003_closed_step60','drawer003_open_step552'):
        references['snapshots'].append({'name':name,'status':'complete','sources_unchanged':True,'rgb_match':{'pixels_exact':True},'state_match':{'qpos_exact':True,'qvel_exact':True,'max_geom_position_error_m':0,'max_geom_rotation_element_error':0}})
    validate_upstream(field,metrics,references)
    bad=copy.deepcopy(field);bad['integrity']['sources']=[]
    with pytest.raises(ValueError):validate_upstream(bad,metrics,references)
    bad=copy.deepcopy(references);bad['snapshots'][0]['rgb_match']['pixels_exact']=False
    with pytest.raises(ValueError):validate_upstream(field,metrics,bad)


def test_reference_geometry_boundaries_do_not_create_connecting_surface():
    from evaluate_rgb import reference_visible_mesh,same_geom_cells
    depth=np.full((4,4),.4);depth[:,2:]=.45
    ids=np.ones((4,4),int);ids[:,2:]=2
    K=np.array([[10.,0,1.5],[0,10.,1.5],[0,0,1.]])
    mesh,info=reference_visible_mesh(depth,K,ids,np.ones_like(ids,bool))
    z=mesh.triangles[:,:,2]
    assert np.all(np.ptp(z,axis=1)<1e-12)
    assert not same_geom_cells(ids)[:,1].any()
