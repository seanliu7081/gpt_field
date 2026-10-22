from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path(__file__).resolve().parent
SRC=Path('/workspace/agent_speedup_v1_drawer003_field_replay/run_002')
field_id=7
meta=json.loads((SRC/f'fields/field_{field_id:03d}.json').read_text())
a=np.load(SRC/f'fields/field_{field_id:03d}.npz')
axes=[a[k] for k in ['x_m','y_m','z_m']];center=a['center_m'];goal=a['goal_m']
plans=[json.loads(l) for l in (SRC/'runtime_plans.jsonl').open() if json.loads(l).get('field_id')==field_id]
actions=[json.loads(l) for l in (SRC/'field_actions.jsonl').open() if json.loads(l).get('field_id')==field_id]
traj=np.array([r['state_before'][:3] for r in actions]+[actions[-1]['state_after'][:3]])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,3,figsize=(12.8,4.05),layout='constrained')
for ax,(i,j,k,name) in zip(axs,[(0,1,2,'XY'),(0,2,1,'XZ'),(1,2,0,'YZ')]):
 idx=int(np.argmin(abs(axes[k]-goal[k])))
 v=np.take(a['distance_to_goal_region_m'],idx,axis=k).T*1000
 ext=[(axes[i][0]-center[i])*1000,(axes[i][-1]-center[i])*1000,(axes[j][0]-center[j])*1000,(axes[j][-1]-center[j])*1000]
 im=ax.imshow(v,origin='lower',extent=ext,cmap='viridis',vmin=0,vmax=220,interpolation='nearest')
 for n,r in enumerate(plans):
  path=np.array(r['path_xyz_m']);ax.plot((path[:,i]-center[i])*1000,(path[:,j]-center[j])*1000,'w--',lw=2,label='Planned path' if n==0 else None)
 ax.plot((traj[:,i]-center[i])*1000,(traj[:,j]-center[j])*1000,color='#ff8259',lw=2.4,label='Measured path')
 ax.scatter([0],[0],c='white',s=34,edgecolors='#14293a',zorder=5)
 ax.scatter([(goal[i]-center[i])*1000],[(goal[j]-center[j])*1000],marker='*',c='#ff4a87',s=135,edgecolors='white',linewidths=.7,zorder=5)
 ax.set_title(f'{name}: slice through goal',fontweight='bold')
 ax.set_xlabel(f'{"XYZ"[i]} offset from start (mm)');ax.set_ylabel(f'{"XYZ"[j]} offset (mm)');ax.set_aspect('equal')
axs[0].legend(loc='upper left',fontsize=9,framealpha=.9)
fig.colorbar(im,ax=axs,shrink=.8,label='Distance to goal region (mm)',pad=.02)
fig.savefig(BASE/'assets/waypoint_field_slices.png',dpi=180,facecolor='white');plt.close(fig)
fig,ax=plt.subplots(figsize=(7.1,3.0),layout='constrained')
s=[r['step']+1 for r in actions];err=[r['goal_error_after_mm'] for r in actions]
ax.plot(s,err,'o-',color='#007f82',lw=2,ms=4);ax.axhline(5,color='#c45c35',linestyle='--',label='Arrival threshold 5 mm')
for r in plans:ax.axvline(r['step'],color='#536575',alpha=.55,linestyle=':')
ax.set_xlabel('Recorded action boundary (step)');ax.set_ylabel('Measured goal error (mm)');ax.legend(fontsize=10)
ax.grid(alpha=.2);fig.savefig(BASE/'assets/waypoint_tracking.png',dpi=170);plt.close(fig)
paths=[SRC/f'fields/field_{field_id:03d}.json',SRC/f'fields/field_{field_id:03d}.npz',SRC/'runtime_plans.jsonl',SRC/'field_actions.jsonl',SRC/'summary.json']
manifest={'field_id':field_id,'start_action':meta['created_before_action'],'phase':meta['phase'],
 'goal_offset_m':meta['compact_definition']['goal_offset_m'],'end_action':actions[-1]['step'],
 'last_error_mm':err[-1],'plans':len(plans),'source':'Existing frozen-controller recorded-action replay run002; no new simulator or model calls.',
 'semantics':'Saved diagnostic materialization of runtime analytic distance, not original policy dense input. Slice through goal; paths projected, may be at other heights.',
 'replay_note':'State and WaypointReach commands match original exactly; a single wrist pixel at step891 differs by1 outside this plotted segment.',
 'sources':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
(BASE/'assets/field_figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest['last_error_mm'])
