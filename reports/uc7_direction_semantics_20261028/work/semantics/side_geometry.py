"""Quantify '<side> of <anchor>' goal/aux regions vs anchor init region + footprint (no simulator)."""
import sys, json, numpy as np
sys.path.insert(0, "/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/semantics")
from bddl_parse import load, L
from meshes import OBJ, object_points, qmat
np.set_printoptions(precision=4, suppress=True)
def footprint_pts(name):
    xml, q, _ = OBJ[name]
    P = object_points(xml, "collision" if name == "flat_stove" else "visual")
    return (qmat(q) @ P.T).T[:, :2]
CASES = [
 # (bddl, region full name, anchor object, anchor init region full name, side word, unit dir)
 ("libero_10/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate.bddl",
  "living_room_table_plate_right_region", "plate", "living_room_table_plate_init_region", "right", (0,1)),
 ("libero_90/LIVING_ROOM_SCENE6_put_the_chocolate_pudding_to_the_left_of_the_plate.bddl",
  "living_room_table_plate_left_region", "plate", "living_room_table_plate_init_region", "left", (0,-1)),
 ("libero_90/KITCHEN_SCENE7_put_the_white_bowl_to_the_right_of_the_plate.bddl",
  "kitchen_table_plate_right_region", "plate", "kitchen_table_plate_init_region", "right", (0,1)),
 ("libero_90/KITCHEN_SCENE6_put_the_yellow_and_white_mug_to_the_front_of_the_white_mug.bddl",
  "kitchen_table_porcelain_mug_front_region", "porcelain_mug", "kitchen_table_porcelain_mug_init_region", "front", (1,0)),
 ("libero_90/STUDY_SCENE1_pick_up_the_yellow_and_white_mug_and_place_it_to_the_right_of_the_caddy.bddl",
  "study_table_desk_caddy_right_region", "desk_caddy", "study_table_desk_caddy_init_region", "right", (0,1)),
 ("libero_90/STUDY_SCENE3_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy.bddl",
  "study_table_red_coffee_mug_behind_region", "red_coffee_mug", "study_table_red_coffee_mug_init_region", "behind(unused)", (-1,0)),
 ("libero_goal/push_the_plate_to_the_front_of_the_stove.bddl",
  "main_table_stove_front_region", "flat_stove", "main_table_stove_region", "front", (1,0)),
]
rows = []
for f, reg, anchor, ainit, word, u in CASES:
    d = load(f"{L}/{f}"); off = d["offset"] or (0,0,0)
    regs = {r["target"]+"_"+n: r for n, r in d["regions"].items()}
    def box(name):
        x0,y0,x1,y1 = regs[name]["ranges"][0]
        return np.array([(x0+x1)/2+off[0], (y0+y1)/2+off[1]]), np.array([(x1-x0)/2, (y1-y0)/2])
    rc, rh = box(reg); ac, ah = box(ainit)
    u = np.array(u, float); v = np.array([-u[1], u[0]])
    dvec = rc - ac
    along, lateral = float(dvec @ u), float(dvec @ v)
    fp = footprint_pts(anchor) + ac           # footprint at nominal anchor pose
    # anchor extent along u measured from anchor origin; region inner/outer edge along u
    a_ext = float(((fp - ac) @ u).max())
    r_half_along = float(abs(rh @ np.abs(u))); r_half_lat = float(abs(rh @ np.abs(v)))
    inner = along - r_half_along; outer = along + r_half_along
    # fraction of region area whose points lie inside the anchor footprint (convex hull approx via radial test for disks/boxes)

    rows.append(dict(task=f.split("/")[1][:70], region=reg, anchor=anchor, side=word,
        anchor_init_centre=[round(x,3) for x in ac], anchor_init_half=[round(x,3) for x in ah],
        region_centre=[round(x,3) for x in rc], region_half=[round(x,3) for x in rh],
        along_m=round(along,3), lateral_m=round(lateral,3),
        anchor_extent_along_from_origin_m=round(a_ext,4),
        region_inner_edge_from_anchor_origin_m=round(inner,3), region_outer_edge_m=round(outer,3),
        gap_anchor_edge_to_region_inner_m=round(inner - a_ext,4)))
for r in rows: print(json.dumps(r))
json.dump(rows, open("/tmp/claude-0/-workspace/10fe2541-a89e-49c2-8e8c-939d1a05ad07/scratchpad/dirsem/semantics/side_geometry.json","w"), indent=1)
