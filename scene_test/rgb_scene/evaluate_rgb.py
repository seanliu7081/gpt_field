"""Separate evaluator for completed RGB-only fields: perception A and end-to-end C.

Ground truth is read ONLY here after the prediction-only field run completes.
Primary geometry is reloaded from the decoded-field mesh artifact. Native metric
scale and predicted K remain untouched; oracle scale is a depth-only diagnostic.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback
import numpy as np
import trimesh
from field_rgb import visible_mesh, backproject, clip_mesh_planes, frustum_planes, depth_triangles
from ray_evaluation import raycast_depth, depth_hit_scores
from rgb_metrics import (sha256, write_json, sample_mesh, compare_meshes, save_mesh, load_mesh, stats)

HERE = Path(__file__).resolve().parent


def checked(root, record):
    path = (root/record['file']).resolve()
    if not path.is_file() or sha256(path) != record['sha256']:
        raise ValueError(f'artifact hash mismatch: {path}')
    return path


def mask_cells(mask):
    """Scoring domain: pixel-cell square retained only if all4 GT labels match.

    The same fixed 2D domain clips source, reference and decoded geometry at
    every resolution. GT semantic masks select scoring regions, never fields.
    """
    mask = np.asarray(mask, bool)
    return mask[:-1, :-1] & mask[1:, :-1] & mask[:-1, 1:] & mask[1:, 1:]


def restrict_to_image_region(mesh, K, cells):
    """Exact triangle clipping to the union of selected pixel-cell frusta.

    Whole-interior faces pass vectorized integral-image classification. Only
    boundary faces are clipped, so masking cannot create artificial cap faces.
    """
    triangles = np.asarray(mesh.triangles)
    if not len(triangles) or not np.any(cells):
        return trimesh.Trimesh(vertices=np.empty((0,3)), faces=np.empty((0,3),int), process=False)
    z = triangles[:, :, 2]
    if np.any(z <= 0):
        raise ValueError('camera geometry must have positive z')
    u = K[0,0]*triangles[:,:,0]/z+K[0,2]
    v = K[1,1]*triangles[:,:,1]/z+K[1,2]
    h, w = cells.shape
    x0 = np.floor(u.min(1)+1e-8).astype(int)
    x1 = np.maximum(x0+1, np.ceil(u.max(1)-1e-8).astype(int))
    y0 = np.floor(v.min(1)+1e-8).astype(int)
    y1 = np.maximum(y0+1, np.ceil(v.max(1)-1e-8).astype(int))
    intersects = (x1 > 0) & (y1 > 0) & (x0 < w) & (y0 < h)
    x0c, x1c = np.clip(x0, 0, w), np.clip(x1, 0, w)
    y0c, y1c = np.clip(y0, 0, h), np.clip(y1, 0, h)
    integral = np.pad(cells.astype(np.int32), ((1,0),(1,0))).cumsum(0).cumsum(1)
    total = integral[y1c,x1c]-integral[y0c,x1c]-integral[y1c,x0c]+integral[y0c,x0c]
    box_cells = (x1-x0)*(y1-y0)
    whole = intersects & (total == box_cells)
    pieces = [triangles[whole]]
    for index in np.flatnonzero(intersects & (total > 0) & ~whole):
        one = trimesh.Trimesh(vertices=triangles[index], faces=[[0,1,2]], process=False)
        for iy in range(y0c[index], y1c[index]):
            for ix in range(x0c[index], x1c[index]):
                if cells[iy,ix]:
                    part = clip_mesh_planes(one, frustum_planes([ix,ix+1,iy,iy+1], K), True)
                    if len(part.faces):
                        pieces.append(np.asarray(part.triangles))
    triangles = np.concatenate(pieces)
    return trimesh.Trimesh(vertices=triangles.reshape(-1,3), faces=np.arange(len(triangles)*3).reshape(-1,3), process=False)


def reference_visible_mesh(depth,K,geom_id,valid):
    """Evaluator reference triangles stay within one visual geometry identity."""
    vertices=backproject(depth,K).reshape(-1,3)
    faces,keep=depth_triangles(depth)
    ids=np.asarray(geom_id).ravel()[faces]
    keep &= np.all(np.asarray(valid).ravel()[faces],axis=-1)
    keep &= (ids[...,0]>=0)&(ids[...,0]==ids[...,1])&(ids[...,0]==ids[...,2])
    mesh=trimesh.Trimesh(vertices=vertices,faces=faces[keep],process=False)
    mesh.remove_unreferenced_vertices()
    return mesh,{'candidate_triangles':int(keep.size),'same_geom_retained_triangles':int(keep.sum()),
                 'surface_area_m2':float(mesh.area),'scope':'secondary visible-geometry interior approximation; primary evaluation uses exact GT rays'}


def same_geom_cells(geom_id):
    ids=np.asarray(geom_id)
    a=ids[:-1,:-1]
    return (a>=0)&(a==ids[1:,:-1])&(a==ids[:-1,1:])&(a==ids[1:,1:])


def depth_metrics(predicted, reference, valid, K_pred, K_gt, oracle_scale):
    valid = np.asarray(valid, bool) & np.isfinite(reference) & (reference > 0) & np.isfinite(predicted) & (predicted > 0)
    if not valid.any():
        return {'status': 'empty_visible_mask', 'pixels': 0}
    p, r = predicted[valid].astype(float), reference[valid].astype(float)
    delta = p-r
    absolute = np.abs(delta)
    ratio = np.maximum(p/r, r/p)
    pp, rr = backproject(predicted, K_pred)[valid], backproject(reference, K_gt)[valid]
    return {'pixels': int(valid.sum()), 'mae_m': float(absolute.mean()), 'rmse_m': float(np.sqrt(np.mean(delta**2))),
            'p95_absolute_m': float(np.quantile(absolute,.95)), 'signed_bias_m': float(delta.mean()),
            'absolute_relative': float(np.mean(absolute/r)), 'delta_1_25': float(np.mean(ratio < 1.25)),
            'matched_pixel_3d_error': stats(np.linalg.norm(pp-rr,axis=1)),
            'oracle_scale_depth_only': {'scale': oracle_scale, 'mae_m': float(np.mean(np.abs(oracle_scale*p-r))),
                 'rmse_m': float(np.sqrt(np.mean((oracle_scale*p-r)**2))),
                 'warning': 'fitted to full-image GT; diagnostic only; never applied to primary field or meshes'}}


def validate_upstream(field_manifest, field_metrics, references):
    if field_manifest.get('status') != 'complete' or field_metrics.get('status') != 'complete' or references.get('status') != 'complete':
        raise ValueError('field build and evaluator reference must both be complete')
    integrity = field_manifest.get('integrity', {})
    for group in ('sources', 'inputs', 'artifacts'):
        flags = integrity.get(group)
        if not isinstance(flags, list) or not flags or not all(value is True for value in flags):
            raise ValueError(f'field integrity group {group} is absent/empty/failed')
    if integrity.get('settings') is not True:
        raise ValueError('field settings integrity failed')
    for key in ('script_unchanged', 'capture_unchanged'):
        if references.get(key) is not True:
            raise ValueError(f'reference {key} failed')
    flags = references.get('artifacts_unchanged', {})
    if not flags or not all(value is True for value in flags.values()):
        raise ValueError('reference artifact integrity failed')
    expected = {'bowl002_initial', 'drawer003_closed_step60', 'drawer003_open_step552'}
    names = [item['name'] for item in references['snapshots']]
    if len(names) != 3 or set(names) != expected:
        raise ValueError('reference names are not exactly the three declared snapshots')
    field_names = [item['name'] for item in field_metrics['snapshots']]
    if len(set(field_names)) != len(field_names) or not set(field_names).issubset(expected):
        raise ValueError('invalid/duplicate field snapshot names')
    config_names = [item['name'] for item in field_metrics['configurations']]
    if not config_names or len(config_names) != len(set(config_names)) or any(item['status'] != 'ok' for item in field_metrics['configurations']):
        raise ValueError('invalid/duplicate/failed field configurations')
    for item in references['snapshots']:
        state = item.get('state_match', {})
        if item.get('status') != 'complete' or item.get('sources_unchanged') is not True or item.get('rgb_match', {}).get('pixels_exact') is not True:
            raise ValueError(f'reference {item["name"]} RGB/source gate failed')
        if state.get('qpos_exact') is not True or state.get('qvel_exact') is not True or state.get('max_geom_position_error_m') != 0 or state.get('max_geom_rotation_element_error') != 0:
            raise ValueError(f'reference {item["name"]} captured-state gate failed')


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name in ('source','inputs','geometry','samples','distances'):
        (output/name).mkdir()
    fields = Path(args.fields).resolve()
    field_manifest = json.loads((fields/'manifest.json').read_text())
    field_metrics = json.loads((fields/'metrics.json').read_text())
    ref_manifest_path = Path(args.references).resolve()
    references = json.loads(ref_manifest_path.read_text())
    validate_upstream(field_manifest, field_metrics, references)
    if sha256(fields/'metrics.json') != field_manifest['metrics_sha256']:
        raise ValueError('field metrics hash mismatch')
    settings = {'stage': 'evaluation_only_after_prediction_only_field_build', 'secondary_surface_samples': args.surface_samples,
                'secondary_continuous_surfaces_enabled': args.secondary_surfaces, 'primary_ray_samples': 'every valid reference image pixel; no subsampling',
                'test_mode': field_metrics['settings']['test_mode'],
                'seed': args.seed, 'primary_scale': 'native metric output, no alignment',
                'regions': 'full_visible/nonrobot_visible/target_bowl_visible/target_cabinet_visible where nonempty',
                'region_domain': 'secondary continuous surface uses fixed cells with all4 same visual geom_id and region label; primary ray score uses ALL valid GT pixels',
                'primary_A_C': 'first-hit depth of predicted or decoded mesh along GT-calibrated pixel-centre rays vs exact GT depth; misses remain failures',
                'primary_B': 'full predicted visible continuous surface in prediction-only builder; additional ray diagnostic here',
                'geometry_reference': 'rendered first-visible visual geometry, not collision proxy or hidden solid',
                'perception_A': 'predicted visible surface vs GT visible surface',
                'representation_B': 'saved-field decoded surface vs same predicted visible surface',
                'end_to_end_C': 'saved-field decoded surface vs GT visible surface',
                'reference_samples': 'fixed area-uniform continuous-surface points, identical across voxel resolutions',
                'unknown': 'missing geometry reduces recall; unseen backfaces/occluded solids are outside scope',
                'oracle': 'one least-squares scalar fit on all valid depth pixels, diagnostic only; no field modification'}
    write_json(output/'settings.json', settings)
    source_files = [HERE/name for name in ('evaluate_rgb.py','ray_evaluation.py','field_rgb.py','rgb_metrics.py')]
    source_files += [HERE.parent/name for name in ('reconstruct.py','mesh_queries.py','sdf_grid.py')]
    source_records=[]
    for path in source_files:
        copy=output/'source'/path.name
        shutil.copy2(path,copy)
        source_records.append({'path':str(path),'sha256':sha256(path),'copy':str(copy.relative_to(output)),'copy_sha256':sha256(copy)})
    inputs=[]
    for path in (fields/'manifest.json',fields/'metrics.json',ref_manifest_path):
        copy=output/'inputs'/('fields_'+path.name if path.parent==fields else 'reference_manifest.json')
        shutil.copy2(path,copy)
        inputs.append({'path':str(path),'sha256':sha256(path),'copy':str(copy.relative_to(output)),'copy_sha256':sha256(copy)})
    manifest={'status':'running','created_utc':datetime.now(timezone.utc).isoformat(),'fields_run':str(fields),
              'reference_manifest':str(ref_manifest_path),'settings_sha256':sha256(output/'settings.json'),'sources':source_records,'inputs':inputs}
    metrics={'status':'running','settings':settings,'snapshots':[],'configurations':[]}
    write_json(output/'manifest.json',manifest)
    write_json(output/'metrics.json',metrics)
    start=time.perf_counter()
    artifacts=[]
    try:
        ref_by_name={item['name']:item for item in references['snapshots']}
        prediction_manifest_path = fields/'inputs'/'prediction_manifest.json'
        bindings = [item for item in field_manifest['inputs'] if item.get('copy') == 'inputs/prediction_manifest.json']
        if len(bindings) != 1 or sha256(prediction_manifest_path) != bindings[0]['copy_sha256'] or bindings[0]['copy_sha256'] != bindings[0]['sha256']:
            raise ValueError('field frozen prediction manifest binding failed')
        copy = output/'inputs'/'field_prediction_manifest.json'
        shutil.copy2(prediction_manifest_path, copy)
        inputs.append({'path':str(prediction_manifest_path),'sha256':sha256(prediction_manifest_path),'copy':str(copy.relative_to(output)),'copy_sha256':sha256(copy)})
        prediction_manifest=json.loads(prediction_manifest_path.read_text())
        model_inputs={item['name']:item for item in prediction_manifest['inputs']}
        for scene_index,snapshot in enumerate(field_metrics['snapshots']):
            name=snapshot['name']
            ref=ref_by_name[name]
            reference_path=checked(ref_manifest_path.parent,ref['evaluator_reference'])
            calibration_path=checked(ref_manifest_path.parent,ref['calibration'])
            rgb_path=checked(ref_manifest_path.parent,ref['input_rgb'])
            for label,path in (('reference',reference_path),('calibration',calibration_path),('rgb',rgb_path)):
                copy=output/'inputs'/f'{name}_{label}{path.suffix}'
                shutil.copy2(path,copy)
                inputs.append({'path':str(path),'sha256':sha256(path),'copy':str(copy.relative_to(output)),'copy_sha256':sha256(copy)})
            if sha256(rgb_path)!=model_inputs[name]['rgb_sha256']:
                raise ValueError(f'{name}: evaluator RGB hash differs from actual model input')
            calibration=json.loads(calibration_path.read_text())
            K_gt=np.asarray(calibration['K'],float)
            with np.load(reference_path,allow_pickle=False) as data:
                gt_depth=data['depth_z_m'].copy()
                gt_valid=data['valid_depth'].copy()
                geom_id=data['geom_id'].copy()
                masks={key:data[key].copy() & gt_valid for key in ('nonrobot_visible','target_bowl_visible','target_cabinet_visible')}
            masks={'full_visible':gt_valid,**masks}
            prediction_path=fields/'inputs'/f'{name}_prediction.npz'
            if sha256(prediction_path)!=snapshot['prediction_sha256']:
                raise ValueError('frozen field prediction changed')
            inputs.append({'path':str(prediction_path),'sha256':sha256(prediction_path)})
            with np.load(prediction_path,allow_pickle=False) as data:
                predicted_depth,K_pred=data['depth_m'].copy(),data['K_pred'].copy()
            if predicted_depth.shape != gt_depth.shape or list(gt_depth.shape) != [calibration['height'],calibration['width']] or any(mask.shape != gt_depth.shape for mask in masks.values()):
                raise ValueError('prediction/reference/calibration/mask shape mismatch')
            if not np.array_equal(K_pred,np.asarray(snapshot['K_pred'])):
                raise ValueError('field and frozen prediction intrinsics differ')
            if not np.all(np.isfinite(gt_depth[gt_valid]) & (gt_depth[gt_valid] > 0)):
                raise ValueError('invalid GT metric depth')
            source_mesh_path=checked(fields,snapshot['source_mesh'])
            inputs.append({'path':str(source_mesh_path),'sha256':sha256(source_mesh_path)})
            pred_mesh=load_mesh(source_mesh_path)
            gt_mesh,gt_info=reference_visible_mesh(gt_depth,K_gt,geom_id,gt_valid)
            interior_cells=same_geom_cells(geom_id)&mask_cells(gt_valid)
            perceived_depth,perception_ray_check=raycast_depth(pred_mesh,K_gt,gt_depth.shape)
            ray_path=output/'distances'/f'{name}_perception_ray_depth.npz'
            with ray_path.open('xb') as file:
                np.savez_compressed(file,predicted_mesh_first_hit_z_m=perceived_depth,reference_z_m=gt_depth,valid_reference=gt_valid)
            artifacts.append({'file':str(ray_path.relative_to(output)),'sha256':sha256(ray_path)})
            common=gt_valid & np.isfinite(predicted_depth) & (predicted_depth>0) & np.isfinite(gt_depth) & (gt_depth>0)
            p,r=predicted_depth[common].astype(float),gt_depth[common].astype(float)
            oracle_scale=float(np.dot(p,r)/np.dot(p,p))
            scene={'name':name,'K_pred':K_pred.tolist(),'K_gt':K_gt.tolist(),'gt_visible_surface':gt_info,
                   'oracle_scale_depth_only':oracle_scale,'perception_ray_validation':perception_ray_check,
                   'secondary_geometry_interior_cells':int(interior_cells.sum()),'excluded_cross_geom_or_invalid_cells':int(interior_cells.size-interior_cells.sum()),'regions':{}}
            metrics['snapshots'].append(scene)
            region_cache={}
            for region_index,(region,mask) in enumerate(masks.items()):
                cells=mask_cells(mask)&interior_cells
                ray_A=depth_hit_scores(perceived_depth,gt_depth,mask)
                if not args.secondary_surfaces:
                    scene['regions'][region]={'status':'ok','perception_A_rays':ray_A,
                        'depth_diagnostic':depth_metrics(predicted_depth,gt_depth,mask,K_pred,K_gt,oracle_scale),
                        'secondary_continuous_surface_status':'disabled_by_explicit_run_setting',
                        'secondary_interior_cells':int(cells.sum()),
                        'excluded_from_secondary_surface_cells':int(mask_cells(mask).sum()-cells.sum())}
                    continue
                if not cells.any():
                    scene['regions'][region]={'status':'no_visible_reference_region','pixels':int(mask.sum()),'perception_A_rays':ray_A}
                    continue
                pmesh=restrict_to_image_region(pred_mesh,K_pred,cells)
                rmesh=restrict_to_image_region(gt_mesh,K_gt,cells)
                if not len(rmesh.faces):
                    scene['regions'][region]={'status':'no_valid_reference_triangles','pixels':int(mask.sum()),'perception_A_rays':ray_A}
                    continue
                base_seed=args.seed+scene_index*1000+region_index*10
                ref_points=sample_mesh(rmesh,args.surface_samples,np.random.default_rng(base_seed))
                pred_points=sample_mesh(pmesh,args.surface_samples,np.random.default_rng(base_seed+1)) if len(pmesh.faces) else np.empty((0,3))
                sample_path=output/'samples'/f'{name}_{region}.npz'
                with sample_path.open('xb') as file:
                    np.savez_compressed(file,reference_points_m=ref_points,prediction_points_m=pred_points,scoring_cells=cells)
                sample_record={'file':str(sample_path.relative_to(output)),'sha256':sha256(sample_path)}
                artifacts.append(sample_record)
                for label,mesh in (('gt',rmesh),('prediction',pmesh)):
                    path=output/'geometry'/f'{name}_{region}_{label}.npz'
                    save_mesh(path,mesh)
                    artifacts.append({'file':str(path.relative_to(output)),'sha256':sha256(path)})
                A,distances=compare_meshes(pmesh,rmesh,ref_points,args.surface_samples,np.random.default_rng(base_seed+1))
                path=output/'distances'/f'{name}_{region}_A.npz'
                with path.open('xb') as file:
                    np.savez_compressed(file,**distances)
                artifacts.append({'file':str(path.relative_to(output)),'sha256':sha256(path)})
                scene['regions'][region]={'status':'ok','scoring_pixels':int(mask.sum()),'scoring_cells':int(cells.sum()),
                       'reference_samples':sample_record,'perception_A':A,'perception_A_rays':ray_A,
                       'excluded_from_secondary_surface_cells':int(mask_cells(mask).sum()-cells.sum()),
                       'depth_diagnostic':depth_metrics(predicted_depth,gt_depth,mask,K_pred,K_gt,oracle_scale)}
                region_cache[region]=(cells,pmesh,rmesh,ref_points,pred_points,base_seed)
                write_json(output/'metrics.json',metrics)
            for source_config in [c for c in field_metrics['configurations'] if c['snapshot']==name]:
                config={'name':source_config['name'],'snapshot':name,'spacing_m':source_config['spacing_m'],'status':'running','regions':{}}
                metrics['configurations'].append(config)
                try:
                    mesh_path=checked(fields,source_config['decoded_mesh'])
                    inputs.append({'path':str(mesh_path),'sha256':sha256(mesh_path)})
                    decoded=load_mesh(mesh_path)
                    config['source_decoded_mesh']=source_config['decoded_mesh']
                    end_depth,end_check=raycast_depth(decoded,K_gt,gt_depth.shape)
                    repr_depth,repr_check=raycast_depth(decoded,K_pred,predicted_depth.shape)
                    config['ray_validation']={'end_to_end_C':end_check,'representation_B':repr_check}
                    ray_path=output/'distances'/f'{config["name"]}_field_ray_depth.npz'
                    with ray_path.open('xb') as file:
                        np.savez_compressed(file,decoded_gt_rays_z_m=end_depth,decoded_predicted_rays_z_m=repr_depth)
                    artifacts.append({'file':str(ray_path.relative_to(output)),'sha256':sha256(ray_path)})
                    config['ray_regions']={region:{'end_to_end_C_rays':depth_hit_scores(end_depth,gt_depth,mask),
                          'representation_B_rays':depth_hit_scores(repr_depth,predicted_depth,mask)} for region,mask in masks.items()}
                    config['regions']={region:dict(values) for region,values in config['ray_regions'].items()}
                    for region,(cells,pmesh,rmesh,ref_points,pred_points,seed) in region_cache.items():
                        restricted=restrict_to_image_region(decoded,K_pred,cells)
                        B,bdata=compare_meshes(restricted,pmesh,pred_points,args.surface_samples,np.random.default_rng(seed+2))
                        C,cdata=compare_meshes(restricted,rmesh,ref_points,args.surface_samples,np.random.default_rng(seed+2))
                        config['regions'][region]={'representation_B':B,'end_to_end_C':C,**config['ray_regions'][region]}
                        path=output/'distances'/f'{config["name"]}_{region}_BC.npz'
                        with path.open('xb') as file:
                            np.savez_compressed(file,**{f'B_{k}':v for k,v in bdata.items()},**{f'C_{k}':v for k,v in cdata.items()})
                        artifacts.append({'file':str(path.relative_to(output)),'sha256':sha256(path)})
                        del restricted
                    config['status']='ok'
                    del decoded
                except Exception:
                    config['status']='failed'
                    config['error']=traceback.format_exc()
                write_json(output/'metrics.json',metrics)
                print(json.dumps({'evaluated':config['name'],'status':config['status']}),flush=True)
        status='complete' if len(metrics['configurations'])==len(field_metrics['configurations']) and all(c['status']=='ok' for c in metrics['configurations']) else 'failed'
    except BaseException:
        status='failed'
        manifest['error']=traceback.format_exc()
    integrity={'sources':[sha256(Path(r['path']))==r['sha256'] and sha256(output/r['copy'])==r['copy_sha256']==r['sha256'] for r in source_records],
               'inputs':[sha256(Path(r['path']))==r['sha256'] and (not r.get('copy') or sha256(output/r['copy'])==r['copy_sha256']==r['sha256']) for r in inputs],
               'artifacts':[sha256(output/r['file'])==r['sha256'] for r in artifacts],
               'settings':sha256(output/'settings.json')==manifest['settings_sha256']}
    if not all(integrity['sources']+integrity['inputs']+integrity['artifacts']+[integrity['settings']]):
        status='integrity_failed'
    metrics['status']=status
    write_json(output/'metrics.json',metrics)
    manifest.update(status=status,elapsed_s=time.perf_counter()-start,metrics_sha256=sha256(output/'metrics.json'),integrity=integrity,artifacts=artifacts)
    write_json(output/'manifest.json',manifest)
    return 0 if status=='complete' else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fields',required=True)
    parser.add_argument('--references',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--surface-samples',type=int,default=100000)
    parser.add_argument('--seed',type=int,default=20261020)
    parser.add_argument('--secondary-surfaces',action='store_true',help='optional same-geom-interior continuous A/C; primary A/C always use all GT rays')
    args=parser.parse_args()
    if args.surface_samples<=0:
        parser.error('positive sample count required')
    # Preflight failures still leave a readable NEW-run terminal status. Never
    # touch an existing output directory when refusal-to-overwrite is triggered.
    output=Path(args.output).resolve()
    preexisting=output.exists()
    try:
        code=run(args)
    except BaseException:
        if not preexisting and output.is_dir() and not (output/'manifest.json').exists():
            write_json(output/'manifest.json',{'status':'failed','phase':'preflight','error':traceback.format_exc()})
        raise
    raise SystemExit(code)

if __name__=='__main__':
    main()
