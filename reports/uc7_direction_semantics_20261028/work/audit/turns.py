import numpy as np, json, glob
def rotvec_to_R(rv):
    rv=np.asarray(rv,float); th=np.linalg.norm(rv)
    if th<1e-12: return np.eye(3)
    k=rv/th; K=np.array([[0,-k[2],k[1]],[k[2],0,-k[0]],[-k[1],k[0],0]])
    return np.eye(3)+np.sin(th)*K+(1-np.cos(th))*K@K
state8=[-0.053896835941528554,-0.007751438647052829,0.6959845859205318,-3.139805653715165,0.0023759174191390884,0.08660449720186457]
Rh=rotvec_to_R(state8[3:6])
print("hand R columns: x",np.round(Rh[:,0],3),"y",np.round(Rh[:,1],3),"z",np.round(Rh[:,2],3))
R="/workspace/gpt_field/runs/agent_speedup_v2_uc7_relayout_t6_t7_20261028_001/task_6"
p=sorted(glob.glob(R+"/trial/requests/000001-*.request.json"))[0]
d=json.load(open(p))
txt=[c["text"] for c in d["payload"]["input"][0]["content"] if c["type"]=="input_text"][0]
g=json.loads(txt)["geometry"]["cameras"]
Rw=np.array(g["wrist"]["rotation_world_from_camera"])
print("wrist cam R columns: right",np.round(Rw[:,0],3),"up",np.round(Rw[:,1],3))
print("hand^T * wristcam:\n",np.round(Rh.T@Rw,3))
ext_right=np.array([0.0,1.0])   # from agentview calibration R[:,0]
cands=[Rh[:2,1],-Rh[:2,0],-Rh[:2,1],Rh[:2,0]]
scores=[float(c@ext_right) for c in cands]
k=int(np.argmax(scores))
print("frame_alignment_turns candidates (turned image-right per k):",[np.round(c,3).tolist() for c in cands],"scores",np.round(scores,3),"-> k",k)
# verify: after rotating the wrist image CCW by k quarter turns, what's image-right in world?
# rot90 CCW once: new right = old up ; new up = old left (=-old right)
right, up = Rw[:,0], Rw[:,1]
for kk in range(4):
    print(" k",kk,"turned wrist image-right =",np.round(right,3), " image-up =",np.round(up,3))
    right, up = up, -right
