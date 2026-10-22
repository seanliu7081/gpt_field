"""Pixel-center first-hit evaluation of saved 3D geometry, independent of GT meshing."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'libero_scene'/'_deps'))


def camera_rays(shape,K):
    v,u=np.indices(shape)
    return np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones(shape)),axis=-1).reshape(-1,3)


def independent_first_hit(triangles,ray):
    """Double-precision exhaustive Moller-Trumbore first-positive intersection."""
    best=np.inf
    for start in range(0,len(triangles),100000):
        tri=triangles[start:start+100000]
        a,b,c=tri[:,0],tri[:,1],tri[:,2]
        e1,e2=b-a,c-a
        p=np.cross(np.broadcast_to(ray,e2.shape),e2)
        determinant=np.einsum('ij,ij->i',e1,p)
        scale=np.linalg.norm(e1,axis=1)*np.linalg.norm(e2,axis=1)*np.linalg.norm(ray)
        valid=np.abs(determinant)>64*np.finfo(float).eps*scale
        inverse=np.divide(1.,determinant,out=np.zeros(len(determinant)),where=valid)
        tvec=-a
        u=np.einsum('ij,ij->i',tvec,p)*inverse
        q=np.cross(tvec,e1)
        v=(q@ray)*inverse
        t=np.einsum('ij,ij->i',e2,q)*inverse
        valid &= (u>=-1e-12)&(v>=-1e-12)&(u+v<=1+1e-12)&(t>0)
        if valid.any():best=min(best,float(t[valid].min()))
    return best


def raycast_depth(mesh,K,shape,validate=True):
    """Return camera-Z of first hit (NaN for miss), with independent spot checks."""
    rays=camera_rays(shape,np.asarray(K,float))
    if not len(mesh.faces):
        return np.full(shape,np.nan),{'status':'empty_mesh','rays':len(rays),'hits':0}
    import point_cloud_utils as pcu
    query=np.repeat(rays,2,axis=0) if len(rays)==1 else rays
    ids,bc,_=pcu.ray_mesh_intersection(np.ascontiguousarray(mesh.vertices,np.float64),np.ascontiguousarray(mesh.faces,np.int64),np.zeros(3),np.ascontiguousarray(query,np.float64))
    ids,bc=np.asarray(ids).reshape(-1)[:len(rays)],np.asarray(bc).reshape(-1,3)[:len(rays)]
    hits=ids>=0
    result=np.full(len(rays),np.nan)
    if hits.any():
        points=np.einsum('ni,nij->nj',bc[hits],np.asarray(mesh.triangles)[ids[hits]])
        result[hits]=points[:,2]
        if not np.all(np.isfinite(result[hits])&(result[hits]>0)):
            raise ValueError('invalid native ray intersection')
    report={'rays':len(rays),'hits':int(hits.sum()),'misses':int((~hits).sum()),
            'backend':'PCU_first_hit_barycentric_camera_z_with_singleton_repeat',
            'depth_convention':'camera_z_m_not_normalized_ray_distance'}
    if validate:
        rng=np.random.default_rng(20261020)
        selected=[]
        for available in (np.flatnonzero(hits),np.flatnonzero(~hits)):
            if len(available):selected.extend(rng.choice(available,min(8,len(available)),replace=False).tolist())
        indices=np.array(sorted(set(selected)),int)
        expected=np.array([independent_first_hit(np.asarray(mesh.triangles),rays[i]) for i in indices])
        got=result[indices]
        native_hit=np.isfinite(got);expected_hit=np.isfinite(expected)
        if not np.array_equal(native_hit,expected_hit):
            raise ValueError('native/independent first-hit visibility differs')
        error=float(np.max(np.abs(got[native_hit]-expected[native_hit]))) if native_hit.any() else 0.
        if error>2e-6:raise ValueError(f'native/independent ray Z differs by {error}m')
        report['independent_validation']={'passed':True,'rays':len(indices),'oracle_hit_count':int(expected_hit.sum()),'oracle_miss_count':int((~expected_hit).sum()),'selection':'fixed_seed_stratified_native_hits_and_misses','max_absolute_error_m':error,'tolerance_m':2e-6}
    return result.reshape(shape),report


def depth_hit_scores(decoded_depth,reference_depth,mask):
    valid=np.asarray(mask,bool)&np.isfinite(reference_depth)&(reference_depth>0)
    hits=valid&np.isfinite(decoded_depth)&(decoded_depth>0)
    total=int(valid.sum());count=int(hits.sum())
    error=np.abs(decoded_depth[hits]-reference_depth[hits])
    result={'reference_pixels':total,'hit_pixels':count,'miss_pixels':total-count,
            'coverage':count/total if total else None,
            'mae_on_hits_m':float(error.mean()) if count else None,
            'rmse_on_hits_m':float(np.sqrt(np.mean(error**2))) if count else None,
            'p95_on_hits_m':float(np.quantile(error,.95)) if count else None,
            'max_on_hits_m':float(error.max()) if count else None,
            'misses_are_failures_in_all_reference_denominator':True,'within_threshold_all_reference_pixels':{}}
    for t in (.001,.002,.005,.010):
        result['within_threshold_all_reference_pixels'][f'{t*1000:g}mm']=float(np.sum(error<=t)/total) if total else None
    return result
