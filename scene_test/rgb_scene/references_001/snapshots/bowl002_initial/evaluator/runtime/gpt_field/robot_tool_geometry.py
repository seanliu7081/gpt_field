"""Frozen Panda hand/finger collision envelopes from known ROBOT assets only.

The observation has a mixed frame convention: state8[:3] is the world position
of ``grip_site``, while state8[3:6] is the world axis-angle orientation of the
robot ``right_hand`` body, NOT the orientation of grip_site. The installed
robosuite ``SingleArm.setup_observables`` defines this convention. The gripper
root's -90-degree Z rotation and the mesh/finger bodies' +90-degree Z rotations
cancel relative to right_hand. grip_site is 0.097 m along this local Z axis.

This module embeds the unique vertices of the two fixed collision STL meshes,
not the visual meshes. It never loads a simulator, task, camera calibration,
object pose, contact, RGB, or runtime asset file. The per-part AABBs enclose the
collision mesh (and its convex hull) but are conservative outside its surface.
Their union preserves more gripper free space than a single ball or whole-tool
box. Robot assets establish robot geometry, NOT obstacle geometry or clearance.

Only the hand, two fingers and two pads are included. Payload, arm links,
actuator dynamics, contacts and translation swept volumes are NOT included.
A caller using these in motion planning must bound the entire commanded and
measured motion, add payload geometry separately, and retain obstacle uncertainty.
"""
from __future__ import annotations

import math
from numbers import Real

import numpy as np

GEOMETRY_VERSION = "panda-known-tool-parts-v1"
# Source paths are relative to the installed robosuite package. Hashes include
# original binary STL bytes and XML/source line endings. No runtime reads occur.
ROBOT_ASSET_PROVENANCE = ({'path': 'models/assets/grippers/panda_gripper.xml',
  'sha256': 'd9d68123909d6910488310d0fae4e358ed4ac831c6d0957f3e30893a5412baac',
  'size_bytes': 4600},
 {'path': 'models/assets/grippers/meshes/panda_gripper/hand.stl',
  'sha256': '94493e94f30fe940f2c8ca2f155c3bbe67bbff406d3edf5e261670d2f0f6e2ed',
  'size_bytes': 10084},
 {'path': 'models/assets/grippers/meshes/panda_gripper/finger.stl',
  'sha256': '2d07a740392f3b9b0816f65d64fff9927d3d57c897870fc4b6ff9c56fff3a0c8',
  'size_bytes': 1684},
 {'path': 'robots/single_arm.py',
  'sha256': '38237d7b68c876b413f0df811b135ca3b2fb91061c011696baab07db5e87fa14',
  'size_bytes': 19730},
 {'path': 'models/assets/robots/panda/robot.xml',
  'sha256': '263c0f9abacbc8686027ec5545bee144481cbcf5cc74aa421cbe3227030cd95e',
  'size_bytes': 21536})

# Exact XML transforms after the two fixed yaw rotations cancel.
GRIP_SITE_Z_M = 0.097
FINGER_BODY_Z_M = 0.0524
PAD_CENTER_FROM_FINGER_M = (0.0, 0.0035, 0.041)
PAD_HALF_SIZE_M = (0.008, 0.004, 0.008)
FINGER_JOINT_RANGES_M = ((0.0, 0.04), (-0.04, 0.0))
NUMERICAL_PADDING_M = 1e-12

# hand.stl: 102 unique binary-float32 vertices, in metres.
HAND_COLLISION_VERTICES_M = (
    (-0.031635917723178864, -0.0005787328700534999, 0.005973074585199356),
    (-0.031597502529621124, -0.001986504066735506, 0.001152931246906519),
    (-0.028010865673422813, 0.04178430140018463, -0.0063964021392166615),
    (-0.025980986654758453, 0.07576971501111984, -0.005082232877612114),
    (-0.025270359590649605, 0.08206511288881302, 0.0051970407366752625),
    (-0.024650247767567635, 0.07666800171136856, -0.0069321258924901485),
    (-0.024495253339409828, 0.08589234948158264, -0.0018327292054891586),
    (-0.023555783554911613, 0.09116631746292114, 0.004607920069247484),
    (-0.02302609011530876, 0.09289686381816864, 0.009876862168312073),
    (-0.022761128842830658, 0.08017436414957047, -0.006909539457410574),
    (-0.022732777521014214, 0.08543447405099869, -0.004120331723242998),
    (-0.021822143346071243, 0.09544170647859573, 0.021756364032626152),
    (-0.02132495865225792, 0.09092690050601959, 0.0009798021055758),
    (-0.020657643675804138, 0.09466791152954102, 0.009537984617054462),
    (-0.0192292220890522, 0.09726057201623917, 0.05058842897415161),
    (-0.019066793844103813, 0.09769713878631592, 0.024627480655908585),
    (-0.018448680639266968, 0.09201552718877792, 0.058393169194459915),
    (-0.018192637711763382, 0.09951794147491455, 0.043814171105623245),
    (-0.017510375007987022, -0.09790761768817902, 0.05166616290807724),
    (-0.0172546599060297, -0.09070031344890594, 0.0630820095539093),
    (-0.017053667455911636, -0.09552478045225143, 0.02212120220065117),
    (-0.01699484884738922, 0.09936285763978958, 0.05411478504538536),
    (-0.016812236979603767, -0.0875329002737999, -0.0012064524926245213),
    (-0.0168041680008173, -0.0919061228632927, 0.006295484956353903),
    (-0.016767239198088646, 0.09025220572948456, 0.06433220207691193),
    (-0.016632145270705223, -0.09855398535728455, 0.05473393574357033),
    (-0.016532693058252335, -0.08223627507686615, -0.005118109285831451),
    (-0.016045523807406425, 0.09320925921201706, 0.06361687183380127),
    (-0.01601313054561615, -0.09962863475084305, 0.04930143058300018),
    (-0.01547170802950859, -0.09070632606744766, 0.06519009917974472),
    (-0.014623098075389862, 0.10016821324825287, 0.055734917521476746),
    (-0.01461702585220337, -0.09344169497489929, 0.06450004875659943),
    (-0.014421956613659859, 0.09127645939588547, 0.06568973511457443),
    (-0.014017116278409958, -0.1000911220908165, 0.056026946753263474),
    (-0.01298792939633131, -0.09249575436115265, 0.06571194529533386),
    (-0.012566710822284222, 0.09371249377727509, 0.06516847014427185),
    (-0.00932386890053749, -0.09998264163732529, 0.058489199727773666),
    (-0.004494388122111559, 0.10042587667703629, 0.058119092136621475),
    (-0.0036368153523653746, -0.08358591049909592, -0.024606257677078247),
    (-0.003616770962253213, -0.10115781426429749, -0.012610324658453465),
    (-0.003429556032642722, -0.10259784758090973, -0.00896429643034935),
    (-0.0032150913029909134, -0.08776598423719406, -0.023884393274784088),
    (-0.00161672942340374, 0.09747105836868286, 0.06230733543634415),
    (-0.0015001483261585236, -0.0458698496222496, -0.02592480182647705),
    (-0.0008678294834680855, -0.08254846930503845, -0.025870200246572495),
    (-0.0008337707840837538, 0.08025997132062912, -0.007321717217564583),
    (-0.000626994005870074, -0.08797474205493927, -0.024961603805422783),
    (-0.0005056393565610051, -0.09294531494379044, 0.06595310568809509),
    (-0.00033732273732312024, 0.08656162023544312, -0.004163281060755253),
    (-0.00031413143733516335, 0.09461534768342972, 0.007574658375233412),
    (-0.00017340583144687116, 0.09099503606557846, 0.0005767836119048297),
    (-0.00011256894504185766, 0.09280849248170853, 0.06596215069293976),
    (-4.1084393160417676e-05, -0.10398972034454346, 0.005690876394510269),
    (-1.7273325283895247e-05, -0.10374269634485245, -0.00919930450618267),
    (-1.5947141946526244e-05, 0.09849606454372406, 0.027081793174147606),
    (3.1809715437702835e-05, -0.09945143014192581, -0.017161205410957336),
    (0.00046960156760178506, -0.09665588289499283, 0.06307835876941681),
    (0.0011905885767191648, -0.10030148178339005, 0.05862313136458397),
    (0.0018427898176014423, -0.0840023085474968, -0.02552490122616291),
    (0.0019174363696947694, -0.04551994055509567, -0.025659434497356415),
    (0.0030760017689317465, -0.08797289431095123, -0.02397872507572174),
    (0.0034947695676237345, -0.1027156412601471, -0.008029340766370296),
    (0.0036021259147673845, -0.10119467973709106, -0.012627636082470417),
    (0.0038103379774838686, -0.08296618610620499, -0.024449335411190987),
    (0.00994154717773199, 0.10003473609685898, 0.05834079533815384),
    (0.012505684047937393, -0.1001875177025795, 0.056894149631261826),
    (0.013138734735548496, -0.09406288713216782, 0.06468318402767181),
    (0.013424810953438282, -0.09192700684070587, 0.06573130935430527),
    (0.013539580628275871, 0.09267213940620422, 0.0656878799200058),
    (0.015396896749734879, 0.09290558844804764, 0.06455627083778381),
    (0.015550924465060234, 0.10019978880882263, 0.05467259883880615),
    (0.015718378126621246, -0.09177957475185394, 0.06491253525018692),
    (0.01580490544438362, -0.09941626340150833, 0.05497027188539505),
    (0.016218384727835655, 0.10028521716594696, 0.04586976766586304),
    (0.016230108216404915, -0.09953752160072327, 0.051584091037511826),
    (0.016558218747377396, -0.0851321741938591, -0.0038727361243218184),
    (0.016734907403588295, 0.09012892097234726, 0.06443186104297638),
    (0.016955528408288956, -0.09160291403532028, 0.005562833044677973),
    (0.017033163458108902, -0.0968022570014, 0.030322037637233734),
    (0.017129452899098396, 0.09906721860170364, 0.05430515855550766),
    (0.017134251073002815, -0.08927122503519058, 0.0636143907904625),
    (0.01758510246872902, -0.09740705043077469, 0.051984645426273346),
    (0.01760716363787651, -0.09384757280349731, 0.058448486030101776),
    (0.018555859103798866, 0.09460194408893585, 0.056617286056280136),
    (0.01926882565021515, 0.09849356859922409, 0.0449417382478714),
    (0.019802046939730644, 0.09701105952262878, 0.020913975313305855),
    (0.020670875906944275, 0.09369184076786041, 0.04330829158425331),
    (0.02134259045124054, 0.09433312714099884, 0.008942007087171078),
    (0.021482903510332108, 0.0917946845293045, 0.0026841284707188606),
    (0.0223142858594656, 0.08824997395277023, -0.0019331998191773891),
    (0.022376978769898415, 0.08335519582033157, -0.005526112858206034),
    (0.022707801312208176, 0.09373555332422256, 0.014340022578835487),
    (0.02295909821987152, 0.07790137827396393, -0.007399038877338171),
    (0.02386915683746338, 0.09040140360593796, 0.004368236754089594),
    (0.024515166878700256, 0.0817025899887085, -0.005051423329859972),
    (0.02472938597202301, 0.0857401043176651, -0.0012316935462877154),
    (0.02481674961745739, 0.07437754422426224, -0.007155700121074915),
    (0.025447942316532135, 0.08118050545454025, 0.0035006047692149878),
    (0.02603430487215519, 0.07529757916927338, -0.005030847620218992),
    (0.028015080839395523, 0.04204234853386879, -0.006349603645503521),
    (0.03158979117870331, -0.0005585176404565573, 0.006214227061718702),
    (0.03161578252911568, -0.001860857242718339, 0.0011797059560194612),
)

# finger.stl: 18 unique binary-float32 vertices, in metres.
FINGER_COLLISION_VERTICES_M = (
    (-0.010479453019797802, 0.01610156148672104, 0.018988166004419327),
    (-0.010401265695691109, 0.026309410110116005, 0.00016650975157972425),
    (-0.01038885023444891, 0.025220269337296486, 0.019187619909644127),
    (-0.010387184098362923, 0.002534183207899332, 0.00013169624435249716),
    (-0.008841174654662609, 0.013917580246925354, 0.05058935657143593),
    (-0.008730395697057247, -2.3525208234786987e-05, 0.036164846271276474),
    (-0.008622935973107815, -5.680193498847075e-05, 0.05095282196998596),
    (-0.0077783796004951, 0.014218212105333805, 0.052365921437740326),
    (-0.005481416825205088, -9.112076077144593e-05, 0.053724709898233414),
    (-0.0052779195830225945, 0.014293111860752106, 0.05384903401136398),
    (0.005839826073497534, 0.014274293556809425, 0.053803451359272),
    (0.00613801833242178, -2.0602645236067474e-05, 0.053577620536088943),
    (0.008616075851023197, 0.013988715596497059, 0.051327913999557495),
    (0.00869277399033308, -0.0001326427300227806, 0.05016622692346573),
    (0.010359962470829487, 0.026403382420539856, 0.00015462865121662617),
    (0.010400470346212387, 0.025253409519791603, 0.019036589190363884),
    (0.010448622517287731, 0.002583304885774851, 0.00014680107415188104),
    (0.010494818910956383, 0.015102642588317394, 0.018435589969158173),
)

def geometry_contract() -> dict:
    """Return independent, JSON-friendly provenance and interpretation metadata."""
    return {
        "version": GEOMETRY_VERSION,
        "robot": "robosuite Panda with PandaGripper, fixed assets listed below",
        "units": "metres and radians",
        "state8": "grip_site world XYZ; right_hand world rotation vector; finger qpos1, qpos2",
        "orientation_frame": "robot right_hand body, not grip_site",
        "eef_reference": "grip_site, local [0, 0, 0.097] from right_hand",
        "fixed_yaw_cancellation": "gripper root -pi/2; hand mesh and finger bodies +pi/2",
        "finger_joint_ranges_m": [list(pair) for pair in FINGER_JOINT_RANGES_M],
        "parts": ["hand_collision", "finger1_collision", "finger2_collision",
                  "finger1_pad_collision", "finger2_pad_collision"],
        "mesh_unique_vertex_counts": {"hand": len(HAND_COLLISION_VERTICES_M),
                                      "finger": len(FINGER_COLLISION_VERTICES_M)},
        "assets": [dict(item) for item in ROBOT_ASSET_PROVENANCE],
        "numerical_padding_m": NUMERICAL_PADDING_M,
        "environment_inputs": [],
        "payload_included": False,
        "arm_links_included": False,
        "certifies_environment_clearance": False,
        "scope": "instantaneous per-part AABBs; optional bounded orientation/finger sweep",
        "orientation_margin_definition": "SO(3) angle from supplied measured right_hand orientation",
        "finger_sweep_definition": "nominal joint interval union measured endpoint; never clip feedback",
        "padding_definition": "nonnegative outward per-part margin, never a model obstacle estimate",
    }


def _state(value) -> np.ndarray:
    # Reject booleans and coerced text rather than interpreting them as geometry.
    raw = np.asarray(value, dtype=object)
    if raw.shape != (8,) or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Real) for v in raw):
        raise ValueError("state8 must contain exactly eight finite real numbers")
    result = np.asarray(raw, dtype=float)
    if not np.isfinite(result).all():
        raise ValueError("state8 must contain representable finite numbers")
    # Measured q is used as supplied, including small joint-limit overshoot.
    # This module does not clip robot feedback or infer whether it is plausible.
    return result


def _nonnegative(value, name, maximum=None) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    checked = float(value)
    if not math.isfinite(checked) or checked < 0 or (maximum is not None and checked > maximum):
        interval = f"[0, {maximum}]" if maximum is not None else "[0, infinity)"
        raise ValueError(f"{name} must be finite and in {interval}")
    return checked


def _rotation_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = math.hypot(*(float(v) for v in rotvec))
    if not math.isfinite(angle):
        raise ValueError("state8 orientation magnitude must be finite")
    if angle == 0:
        return np.eye(3)
    x, y, z = rotvec / angle
    cross = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    # sin(theta/2)^2 avoids cancellation in 1-cos(theta) near zero.
    return np.eye(3) + math.sin(angle)*cross + (2*math.sin(angle/2)**2)*(cross @ cross)


def _local_parts(state: np.ndarray, full_finger_range: bool):
    hand = np.asarray(HAND_COLLISION_VERTICES_M) - [0., 0., GRIP_SITE_Z_M]
    base_finger = np.asarray(FINGER_COLLISION_VERTICES_M)
    pad_corners = np.asarray([(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    pad_corners = pad_corners * PAD_HALF_SIZE_M
    pieces = [("hand_collision", hand, None)]
    pads = []
    for index, sign in enumerate((1., -1.)):
        q = float(state[6+index])
        nominal = FINGER_JOINT_RANGES_M[index]
        interval = (min(q, nominal[0]), max(q, nominal[1])) if full_finger_range else (q, q)
        # Vertex positions vary affinely with the slider. Its two endpoint
        # meshes suffice to enclose every joint position in the interval.
        offsets = np.array([[0., qi, FINGER_BODY_Z_M-GRIP_SITE_Z_M] for qi in interval])
        finger = base_finger * [sign, sign, 1.]
        finger = (finger[None, :, :] + offsets[:, None, :]).reshape(-1, 3)
        pad_center = np.array([0., sign*PAD_CENTER_FROM_FINGER_M[1], PAD_CENTER_FROM_FINGER_M[2]])
        pad = (pad_corners[None, :, :] + (offsets+pad_center)[:, None, :]).reshape(-1, 3)
        pieces.append((f"finger{index+1}_collision", finger, interval))
        pads.append((f"finger{index+1}_pad_collision", pad, interval))
    return pieces + pads


def world_aabbs(state8, *, orientation_margin_rad=0.0,
                full_finger_range=False, padding_m=0.0) -> list[dict]:
    """Enclose five known robot collision parts using only measured robot state.

    ``min_xyz`` and ``max_xyz`` are world-space bounds. ``offset_min_xyz`` and
    ``offset_max_xyz`` are bounds in WORLD axes relative to measured EEF XYZ;
    they are not coordinates in the rotating hand frame. For fixed orientation,
    a caller can Minkowski-inflate a world obstacle [lo, hi] for this part as
    [lo-offset_max_xyz, hi-offset_min_xyz], then check the entire EEF segment.
    Do not replace this asymmetric expression with a scalar tool radius.

    By default only measured finger positions and orientation are enclosed.
    ``full_finger_range`` additionally covers the nominal independent slider
    ranges including any measured overshoot (no feedback clipping).
    ``orientation_margin_rad`` additionally encloses any orientation within
    the specified SO(3) angle of the measured one, using the exact displacement
    upper bound 2*r*sin(angle/2) per part. It does not authorize such rotation.
    ``padding_m`` is a separate nonnegative outward margin. A fixed 1e-12 m
    outward numeric guard is logged independently. No geometry is silently
    removed or reduced, and payload is not substituted by this robot model.
    """
    state = _state(state8)
    if not isinstance(full_finger_range, (bool, np.bool_)):
        raise ValueError("full_finger_range must be a boolean")
    angle = _nonnegative(orientation_margin_rad, "orientation_margin_rad", math.pi)
    padding = _nonnegative(padding_m, "padding_m")
    rotation = _rotation_matrix(state[3:6])
    output = []
    for name, vertices, interval in _local_parts(state, bool(full_finger_range)):
        offsets = vertices @ rotation.T
        radius = float(np.max(np.linalg.norm(vertices, axis=1)))
        orientation_margin = 2*radius*math.sin(angle/2)
        margin = orientation_margin + padding + NUMERICAL_PADDING_M
        lower, upper = offsets.min(axis=0)-margin, offsets.max(axis=0)+margin
        low_world, high_world = lower+state[:3], upper+state[:3]
        if not np.isfinite(np.r_[lower, upper, low_world, high_world, radius]).all():
            raise ValueError("robot geometry exceeds finite numeric range")
        output.append({
            "name": name,
            "min_xyz": low_world.tolist(), "max_xyz": high_world.tolist(),
            "offset_min_xyz": lower.tolist(), "offset_max_xyz": upper.tolist(),
            "max_radius_m": radius,
            "orientation_margin_rad": angle, "orientation_margin_m": orientation_margin,
            "padding_m": padding, "numerical_padding_m": NUMERICAL_PADDING_M,
            "finger_range_m": None if interval is None else list(interval),
            "full_finger_range": bool(full_finger_range),
            "geometry_version": GEOMETRY_VERSION,
        })
    return output
