"""Prototype of the offline build step: Panda MJCF collision geoms -> convex hulls -> covering spheres.

Runs in an interpreter WITH scipy (e.g. .venv-sim). Output: npz + json manifest with hashes.
Read-only on the robosuite assets. Usage:
  python -I build_body_model.py ASSET_ROOT OUT_PREFIX
"""
import hashlib, json, struct, sys, time
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import ConvexHull

ASSET = Path(sys.argv[1])
OUT = Path(sys.argv[2])
ROBOT = ASSET / "robots/panda/robot.xml"
GRIPPER = ASSET / "grippers/panda_gripper.xml"
RNG = np.random.default_rng(0)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def quat_mat(wxyz):
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def tf(el):
    T = np.eye(4)
    T[:3, 3] = np.asarray(el.get("pos", "0 0 0").split(), float)
    T[:3, :3] = quat_mat(np.asarray(el.get("quat", "1 0 0 0").split(), float))
    return T


def stl_vertices(path):
    data = path.read_bytes()
    n = struct.unpack_from("<I", data, 80)[0]
    assert len(data) == 84 + 50 * n, path
    rows = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), offset=84)
    return np.unique(rows["v"].reshape(-1, 3).astype(float), axis=0)


def collect(xml, category):
    root = ET.parse(xml).getroot()
    meshdir = xml.parent
    assets = {m.get("name"): m for m in root.findall("./asset/mesh")}
    bodies = []

    def visit(body, parent):
        idx = len(bodies)
        joint = body.find("joint")
        rec = {"name": body.get("name"), "parent": parent, "T": tf(body).tolist(),
               "joint": None if joint is None else {k: joint.get(k) for k in ("name", "type", "axis", "pos", "range")},
               "geoms": []}
        bodies.append(rec)
        for gi, geom in enumerate(body.findall("geom")):
            if geom.get("contype", "1") == "0" and geom.get("conaffinity", "1") == "0":
                continue  # visual only
            name = geom.get("name", f"{body.get('name')}_{gi}")
            kind = geom.get("type", "sphere")
            if kind == "mesh":
                a = assets[geom.get("mesh")]
                path = (meshdir / a.get("file")).resolve()
                v = stl_vertices(path) * np.asarray(a.get("scale", "1 1 1").split(), float)
                src = {"file": str(path), "sha256": sha(path)}
            elif kind == "box":
                s = np.asarray(geom.get("size").split(), float)
                v = np.array([[x, y, z] for x in (-s[0], s[0]) for y in (-s[1], s[1]) for z in (-s[2], s[2])])
                src = {"box_half_size": s.tolist()}
            else:
                raise ValueError(kind)
            Tg = tf(geom)
            v = v @ Tg[:3, :3].T + Tg[:3, 3]
            part_kind = "pad" if "_pad_" in name else "finger" if "finger" in name else (
                "hand" if name.startswith("hand") else category)
            rec["geoms"].append({"name": name, "kind": part_kind, "verts": v, "src": src})
        for child in body.findall("body"):
            visit(child, idx)

    for b in root.findall("./worldbody/body"):
        visit(b, -1)
    return bodies



def fit(verts, beta, grid_mm=None):
    """Greedy cover: candidate centres on an interior grid, radius = beta + depth(c).

    Each ball's outward error w.r.t. every hull face plane is exactly <= beta (closed form),
    so the union's plane-bulge is <= beta; coverage of hull vertices, dense surface samples
    and an interior grid is verified afterwards.
    """
    hull = ConvexHull(verts)
    V = verts[hull.vertices]
    eq = hull.equations
    tri = verts[hull.simplices]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    lo, hi = V.min(0), V.max(0)

    def surface(n):
        idx = RNG.choice(len(tri), size=n, p=area / area.sum())
        r1, r2 = RNG.random(n), RNG.random(n)
        s = np.sqrt(r1)
        return (1 - s)[:, None] * tri[idx, 0] + (s * (1 - r2))[:, None] * tri[idx, 1] + (s * r2)[:, None] * tri[idx, 2]

    def grid(step):
        g = np.stack(np.meshgrid(*[np.arange(lo[i], hi[i] + step, step) for i in range(3)], indexing="ij"), -1).reshape(-1, 3)
        return g[np.all(g @ eq[:, :3].T + eq[:, 3] <= 0, axis=1)]

    ext = float(np.min(hi - lo))
    step = grid_mm / 1000 if grid_mm else max(0.002, min(0.006, ext / 8))
    cand = np.concatenate([grid(step), V * 0.999 + V.mean(0) * 0.001])
    depth = -np.max(cand @ eq[:, :3].T + eq[:, 3], axis=1)  # >= 0 inside
    rad = beta + np.maximum(depth, 0.)
    targets = np.concatenate([V, surface(6000), grid(max(step * 1.5, 0.004))])
    cov = (np.linalg.norm(targets[None, :, :] - cand[:, None, :], axis=2) <= rad[:, None])
    uncovered = np.ones(len(targets), bool)
    chosen = []
    while uncovered.any():
        counts = cov[:, uncovered].sum(1)
        j = int(np.argmax(counts))
        if counts[j] == 0:
            raise RuntimeError("cover failed")
        chosen.append(j)
        uncovered &= ~cov[j]
    C, R = cand[chosen], rad[chosen].copy()
    val = np.concatenate([V, surface(40000), grid(max(step, 0.003))])
    grown = 0.
    for _ in range(4):
        d = np.linalg.norm(val[:, None] - C[None], axis=2) - R[None]
        bad = d.min(1) > 0
        if not bad.any():
            break
        near = np.argmin(d[bad], axis=1)
        for j in np.unique(near):
            need = float(np.max(np.linalg.norm(val[bad][near == j] - C[j], axis=1))) + 1e-6
            grown = max(grown, need - R[j]); R[j] = need
    d = np.linalg.norm(val[:, None] - C[None], axis=2) - R[None]
    covered = bool((d.min(1) <= 1e-9).all())
    bulge = float(np.max(np.max(C @ eq[:, :3].T + eq[:, 3], axis=1) + R))
    return {"centers": C, "radii": R, "hull_vertices": V, "hull_equations": eq, "covered": covered,
            "bulge_plane_m": bulge, "extent_m": (hi - lo).tolist(), "n_val": len(val), "grown_m": grown}


BETA = {"arm": float(sys.argv[3]) / 1000, "hand": float(sys.argv[4]) / 1000, "finger": float(sys.argv[4]) / 1000, "pad": float(sys.argv[4]) / 1000}
K_BY_KIND = BETA

started = time.perf_counter()
out = {"bodies": [], "parts": []}
for category, xml in (("arm", ROBOT), ("tool", GRIPPER)):
    bodies = collect(xml, "arm")
    offset = len(out["bodies"])
    for b in bodies:
        rec = {k: v for k, v in b.items() if k != "geoms"}
        rec["xml"] = category
        rec["parent"] = -1 if b["parent"] < 0 else b["parent"] + offset
        out["bodies"].append(rec)
        for g in b["geoms"]:
            res = fit(g["verts"], BETA[g["kind"]])
            out["parts"].append({"name": g["name"], "kind": g["kind"], "body": len(out["bodies"]) - 1,
                                 "src": g["src"], **res})
elapsed = time.perf_counter() - started
arrays = {}
manifest = {"robot_xml": str(ROBOT), "robot_xml_sha256": sha(ROBOT), "gripper_xml": str(GRIPPER),
            "gripper_xml_sha256": sha(GRIPPER), "builder_sha256": sha(__file__), "rng_seed": 0,
            "beta_by_kind_m": BETA, "build_s": elapsed, "bodies": [], "parts": []}
for i, b in enumerate(out["bodies"]):
    manifest["bodies"].append({"index": i, "name": b["name"], "parent": b["parent"], "xml": b["xml"],
                               "T": b["T"], "joint": b["joint"]})
for i, p in enumerate(out["parts"]):
    for key in ("centers", "radii", "hull_vertices", "hull_equations"):
        arrays[f"p{i}_{key}"] = np.asarray(p[key])
    manifest["parts"].append({"index": i, "name": p["name"], "kind": p["kind"], "body": p["body"],
                              "body_name": out["bodies"][p["body"]]["name"], "src": p["src"],
                              "spheres": int(len(p["radii"])), "covered_validation": p["covered"],
                              "validation_points": p["n_val"], "bulge_plane_mm": round(1000 * p["bulge_plane_m"], 2),
                              "max_radius_mm": round(1000 * float(np.max(p["radii"])), 2),
                              "hull_extent_mm": [round(1000 * x, 1) for x in p["extent_m"]]})
np.savez(OUT.with_suffix(".npz"), **arrays)
manifest["npz_sha256"] = sha(OUT.with_suffix(".npz"))
OUT.with_suffix(".json").write_text(json.dumps(manifest, indent=1))
for p in manifest["parts"]:
    print(f"{p['name']:24s} {p['kind']:6s} body={p['body_name']:18s} spheres={p['spheres']:2d} covered={p['covered_validation']} "
          f"bulge={p['bulge_plane_mm']:6.2f}mm rmax={p['max_radius_mm']:6.2f}mm extent={p['hull_extent_mm']}")
print("build_s", round(elapsed, 2), "total spheres", sum(p["spheres"] for p in manifest["parts"]))
