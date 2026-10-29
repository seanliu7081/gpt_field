"""Offline robot-only projected full-finger sweep ranking. Never clearance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import trimesh

BASE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(BASE / 'validation/new_host_20261029/controller_clearance_wider_analysis_001/source_snapshot'))
from analyze_joint_probe import URDF, pose_matrix

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--grid', action='append', nargs=2, required=True, metavar=('PATH','SHA256'))
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
assert not args.output.exists()
sources = {}


def bind(path, expected=None):
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError('Input hash changed: '+str(path))
    if str(path) in sources and sources[str(path)] != digest:
        raise ValueError('Conflicting source bytes')
    sources[str(path)] = digest
    return {'path':str(path),'sha256':digest}


def read(path, expected=None):
    binding = bind(path, expected)
    return json.loads(Path(binding['path']).read_text())


validation = BASE / 'validation/new_host_20261029'
geometry = read(validation / 'tool_geometry_001/candidate.json')
for row in geometry['source_files']: bind(row['path'],row['sha256'])
asset_paths = {Path(row['path']).name:Path(row['path']) for row in geometry['source_files']}
urdf = URDF(asset_paths['X5A.urdf'])
bind(Path(sys.modules[URDF.__module__].__file__), '3254117d0f8ed7c7fd49a332436da278a8621ebd6a55021d0a6810daa0c91ddf')
meshes = {name:trimesh.load_mesh(asset_paths[name+'.STL'],process=False) for name in ('link7','link8')}
assert all(np.isfinite(mesh.vertices).all() for mesh in meshes.values())
data = {}
fk_checks = []
controller = validation / 'native_controller_characterization_008/worker/controller'
for arm, stage in (('left',27),('right',34)):
    request = read(controller / f'clearance_{arm}/request.json')
    capture = request['capture']
    assert request['stage_index'] == stage and capture['physics_unchanged_during_capture'] is True
    records = {name:read(controller / capture[name]['path'],capture[name]['sha256'])
               for name in ('robot_measurement','camera_inventory')}
    image = next(row for row in capture['images'] if row['camera']=='cam_head')
    image_binding = bind(controller/image['path'],image['sha256'])
    state, cameras = records['robot_measurement'], records['camera_inventory']
    assert state['native_physics_time_s']==cameras['native_physics_time_s']==capture['native_physics_time_s']
    robot = next(row for row in state['raw_robot_inventory']['robots'] if row['arm_name']==arm+'_arm')
    q = dict(zip(robot['joint_names'],robot['joint_positions']['values'][0]))
    poses = np.asarray(robot['body_link_pose_w']['values'])[0]
    root = pose_matrix(poses[robot['body_names'].index('base_link')])
    frames,_ = urdf.forward(q,root)
    for link in ('link6','link7','link8'):
        actual = pose_matrix(poses[robot['body_names'].index(link)])
        error = float(np.max(np.abs(actual-frames[link])))
        assert error<1e-5
        fk_checks.append({'arm':arm,'link':link,'maximum_matrix_error':error})
    data[arm] = {'q':q,'root':root,'camera':cameras['cameras']['head'],'image':image_binding,
                 'stage_index':stage,'state':capture['robot_measurement'],'cameras':capture['camera_inventory']}


def sweep(arm, q, root):
    """Positive-depth projected triangle-prism extrema occur at endpoint vertices.

    Every finger is a rigid visual mesh translating with its one prismatic DOF.
    The sweep of each triangle is its endpoint convex prism. Perspective images
    of positive-depth convex prisms have extrema of any linear image functional
    at projected endpoint vertices. This does not compare scene depth/objects.
    """
    camera = data[arm]['camera']
    optical = np.asarray(camera['optical_from_world'])
    intrinsics = np.asarray(camera['intrinsics'])
    uv = []; depth = []; by_link = {}
    for endpoint in ('open','physical_zero'):
        state = dict(q)
        for name in ('joint7','joint8'):
            state[name] = data[arm]['q'][name] if endpoint=='open' else 0.
            assert 0<=state[name]<=.044001
        frames,_ = urdf.forward(state,root)
        for link,mesh in meshes.items():
            world = mesh.vertices@frames[link][:3,:3].T+frames[link][:3,3]
            xyz = world@optical[:3,:3].T+optical[:3,3]
            assert np.isfinite(xyz).all() and (xyz[:,2]>0).all()
            homogeneous = xyz@intrinsics.T
            pixels = homogeneous[:,:2]/homogeneous[:,2,None]
            uv.append(pixels);depth.append(xyz[:,2])
            by_link.setdefault(link,{})[endpoint]={'u_min':float(pixels[:,0].min()),'u_max':float(pixels[:,0].max()),
                'v_min':float(pixels[:,1].min()),'v_max':float(pixels[:,1].max())}
    uv = np.concatenate(uv);depth=np.concatenate(depth)
    functional = ((1 if arm=='right' else -1)*uv[:,0]+uv[:,1])/np.sqrt(2)
    return {'directional_minimum_px':float(functional.min()),
            'projected_bounds_px':{'u_min':float(uv[:,0].min()),'u_max':float(uv[:,0].max()),
                                   'v_min':float(uv[:,1].min()),'v_max':float(uv[:,1].max())},
            'minimum_camera_depth_m':float(depth.min()),'all_mesh_vertices_positive_depth':True,
            'whole_sweep_positive_depth':True,'unclipped_vertex_count':len(uv),
            'some_vertices_outside_rgb':bool(((uv[:,0]<-.5)|(uv[:,0]>639.5)|(uv[:,1]<-.5)|(uv[:,1]>479.5)).any()),
            'per_link_endpoint_bounds_px':by_link}


baseline = {arm:sweep(arm,row['q'],row['root']) for arm,row in data.items()}
fields = ('lift_m','retreat_m','outward_m','outward_yaw_rad')
results = []; grids = []; rejected = []
for path,digest in args.grid:
    grid = read(path,digest);grids.append(bind(path,digest))
    assert grid['schema']=='robodojo-clearance-lift-retreat-outward-yaw-grid-1'
    assert grid['execution_authorized'] is False and grid['native_actions_executed']==0
    assert grid['hypothetical_analysis_only'] is True
    assert grid['production_guards_unchanged']['joint_distance_from_home_rad']==.35
    assert grid['unchanged_guards']==dict(grid['production_guards_unchanged'],joint_distance_from_home_rad=.50)
    for row in grid['source_files']: bind(row['path'],row['sha256'])
    grouped = {}
    for row in grid['results']:
        key = tuple(row[name] for name in fields)
        assert row['arm'] not in grouped.setdefault(key,{})
        grouped[key][row['arm']] = row
    for key,arms in grouped.items():
        guards = grid['unchanged_guards']
        feasible = set(arms)=={'left','right'} and all(
            row['failure'] is None and row['numerical_updates']==180
            and row['max_command_lead_rad']<=.05+1e-12
            and row['minimum_native_position_margin_rad']>=0 and row['minimum_native_velocity_margin_rad_s']>=0
            and max(row['max_joint_home_rad'],row['max_command_home_rad'])<=guards['joint_distance_from_home_rad']
            and row['max_tool_home_m']<=guards['tool_distance_from_home_m']
            and row['max_orientation_home_rad']<=guards['tool_orientation_from_home_rad']
            and row['max_kinematic_joint_rate_rad_s']<=guards['measured_joint_velocity_rad_s']
            for row in arms.values())
        if not feasible:
            rejected.append({'grid_sha256':digest,'parameters':dict(zip(fields,key)),
                             'reason':'one_or_both_nominal_paths_incomplete_or_outside_declared_guards'})
            continue
        entry={'grid_sha256':digest,'parameters':dict(zip(fields,key)),'arms':{}}
        for arm,row in arms.items():
            endpoint = row['candidate_endpoint']
            root = np.asarray(endpoint['world_from_base_link'])
            assert np.allclose(root,data[arm]['root'],rtol=0,atol=1e-6)
            metrics = sweep(arm,dict(zip(endpoint['all_joint_names'],endpoint['full_q_rad'])),root)
            metrics['directional_minimum_gain_px']=metrics['directional_minimum_px']-baseline[arm]['directional_minimum_px']
            metrics['candidate_endpoint']=endpoint
            metrics['nominal_joint_margin_rad']=row['joint_margin_rad']
            for name in ('production_035_joint_margin_rad','violates_production_035_joint_envelope','minimum_joint_home_ceiling_required_by_nominal_path_rad','tool_margin_m','orientation_margin_rad','velocity_ceiling_margin_rad_s','max_command_lead_rad','minimum_native_position_margin_rad','minimum_native_velocity_margin_rad_s'):
                metrics[name]=row[name]
            entry['arms'][arm]=metrics
        entry['worst_arm_directional_gain_px']=min(row['directional_minimum_gain_px'] for row in entry['arms'].values())
        entry['minimum_joint_home_ceiling_required_by_either_arm_rad']=max(row['minimum_joint_home_ceiling_required_by_nominal_path_rad'] for row in entry['arms'].values())
        entry['production_035_would_accept_both_nominal_paths']=all(not row['violates_production_035_joint_envelope'] for row in entry['arms'].values())
        entry['nominal_reserve_filter'] = all(row['nominal_joint_margin_rad']>=.05 and row['tool_margin_m']>=.005 and row['orientation_margin_rad']>=.01 and row['velocity_ceiling_margin_rad_s']>=.5 for row in entry['arms'].values())
        results.append(entry)
results.sort(key=lambda row:row['worst_arm_directional_gain_px'],reverse=True)
for index,row in enumerate(results,1):row['rank']=index
record={'schema':'robodojo-offline-complete-finger-directional-ranking-1',
        'hypothetical_analysis_only':True,'production_joint_home_ceiling_rad':.35,'hypothetical_joint_home_ceiling_rad':.50,
        'nominal_reserve_filter_definition':{'joint_below_hypothetical_ceiling_rad':.05,'tool_below65mm_ceiling_m':.005,'orientation_below012rad_ceiling_rad':.01,'velocity_below2rads_ceiling_rad_s':.5},
        'production_source_or_limits_changed':False,'execution_authorized':False,
        'metric':{'right':'min((u+v)/sqrt(2))','left':'min((-u+v)/sqrt(2))',
                  'comparison':'candidate minimum minus actual008 clearance minimum',
                  'paired_rank':'descending minimum gain across both arms','units':'image pixels'},
        'scope':'robot_only_predicted_visual_mesh_sweeps_not_object_distance_or_clearance',
        'geometry_method':sweep.__doc__,'grid_inputs':grids,'baseline':baseline,
        'baseline_captures':{arm:{key:row[key] for key in ('image','stage_index','state','cameras')} for arm,row in data.items()},
        'measured_open_finger_positions_m':{arm:{name:row['q'][name] for name in ('joint7','joint8')} for arm,row in data.items()},
        'fk_checks':fk_checks,'ranked_candidates':results,'rejected_nominal_candidates':rejected,
        'native_close_target_control_fact_m':-.01,'negative_physical_positions_projected':False,
        'limitations':['Ranking uses unclipped visual meshes, not native collision geometry.',
                       'Positive diagonal gains are not clearance or contact tests.',
                       'Nominal path guards are kinematic checks, not measured future dynamics.',
                       'Some full-finger vertices may be outside the saved RGB; bounds are not clipped.',
                       'Each arm uses its own recorded008 clearance pose and camera, not the other arm’s later pose.'],
        'simulator_starts':0,'model_calls':0,'native_actions':0,'clearance_approved':False,
        'full_test_authorized':False,'standard_ready':False,'model_input_eligible':False}
bind(__file__)
record['source_files']=[{'path':path,'sha256':digest} for path,digest in sorted(sources.items())]
assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest for path,digest in sources.items())
args.output.mkdir(parents=False,exist_ok=False)
result=args.output/'ranking.json';result.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
print(json.dumps({'output':str(result),'sha256':hashlib.sha256(result.read_bytes()).hexdigest(),
                  'paired_feasible':len(results),'rejected':len(rejected),
                  'top':[{'rank':row['rank'],'parameters':row['parameters'],
                          'gain_px':{arm:v['directional_minimum_gain_px'] for arm,v in row['arms'].items()},
                          'bounds_px':{arm:v['projected_bounds_px'] for arm,v in row['arms'].items()},'required_joint_ceiling_rad':row['minimum_joint_home_ceiling_required_by_either_arm_rad'],'reserve_filter':row['nominal_reserve_filter']}
                         for row in results[:5]]},indent=2),flush=True)
