import json,glob,sys
import numpy as np
R="/workspace/gpt_field/runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6"
p=sorted(glob.glob(R+"/trial/requests/000001-*.request.json"))[0]
d=json.load(open(p))
txt=[c["text"] for c in d["payload"]["input"][0]["content"] if c["type"]=="input_text"][0]
g=json.loads(txt)["geometry"]["cameras"]
def cell_uv(cell, W=512, n=8):
    r,c=divmod(cell-1,n); s=W/n
    return ((c+0.5)*s-0.5, (r+0.5)*s-0.5)
def rotate_pixel(uv,k,W=512):
    last=W-1.0; u,v=uv
    for _ in range(k%4): u,v=v,last-u
    return (u,v)
def unrotate_pixel(uv,k,W=512):
    last=W-1.0; u,v=uv
    for _ in range(k%4): u,v=last-v,u
    return (u,v)
def ray(cam, uv, z0):
    fx,fy=cam["focal_length_px"]; cx,cy=cam["principal_point_px"]
    Rm=np.array(cam["rotation_world_from_camera"]); o=np.array(cam["position_world_m"])
    u,v=uv
    dc=np.array([(u-cx)/fx, -(v-cy)/fy, -1.0])
    dw=Rm@dc
    t=(z0-o[2])/dw[2]
    return o+t*dw
truth={"A mug":(-0.12,0.18),"C plate":(0.149,0.006),"D pudding":(-0.16,-0.14),"E region(plate +y, LIBERO right)":(0.15,0.10)}
cells={"A mug":(49,38),"C plate":(12,61),"D pudding":(55,35),"E region":(29,55)}
for z0 in (0.42,0.44):
    print("plane z=",z0)
    for name,(w,x) in cells.items():
        pe=ray(g["external"], cell_uv(x), z0)
        pw=ray(g["wrist"], cell_uv(w), z0)
        # if the wrist image had been turned by k=2 (180 deg) the shown cell w would map to upright pixel unrotate(k=2)
        pw2=ray(g["wrist"], unrotate_pixel(cell_uv(w),2), z0)
        print(f"  {name:10s} ext cell {x:2d} -> ({pe[0]:+.3f},{pe[1]:+.3f}) | wrist cell {w:2d} (k=0 as shown) -> ({pw[0]:+.3f},{pw[1]:+.3f}) | same cell if shown with k=2 -> ({pw2[0]:+.3f},{pw2[1]:+.3f})")
print("truth", truth)
# image axes
for name in ("external","wrist"):
    Rm=np.array(g[name]["rotation_world_from_camera"])
    print(name, "image-right world", np.round(Rm[:,0],3), "image-up world", np.round(Rm[:,1],3), "forward", np.round(-Rm[:,2],3))
