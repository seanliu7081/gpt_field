# Idealized estimate of _ground_observed_region null-bounds centre (spatial.py:197-247), assuming EVERY
# table cell inside the HeightField window is observed (upper bound on coverage; real coverage is partial).
import numpy as np
cell=0.005
hand=np.array([-0.0539,-0.0078])            # start grip xy (state8) -> HeightField.empty centre (scene.py:1003)
ox,oy=hand-0.5                               # HF_HALF_EXTENT_M=0.5 (types.py:128)
n=int(round(1.0/cell))+1
gy,gx=np.indices((n,n)); xs=ox+gx*cell; ys=oy+gy*cell
table=(np.abs(xs)<=0.35)&(np.abs(ys)<=0.80)  # living_room_table_full_size (0.70,1.6) at offset (0,0,0.41)
plate_c=np.array([0.149,0.006]); r_plate=0.075
plate_hi=plate_c+r_plate; plate_lo=plate_c-r_plate
def disc(c,r): return (xs-c[0])**2+(ys-c[1])**2<=r**2
others=[((-0.12,0.18),0.06),((0.06,-0.20),0.05),((-0.16,-0.14),0.04)]   # mug, red mug, pudding (approx radii)
mask_obj=np.zeros_like(table)
for c,r in others: mask_obj|=disc(c,r+cell)
box=lambda x0,x1,y0,y1:(xs>=x0)&(xs<=x1)&(ys>=y0)&(ys<=y1)
libero_right=(0.10,0.20,0.05,0.15)
for name,side in (("right_of as +x (current)",xs>plate_hi[0]),("right_of as +y (LIBERO/viewer)",ys>plate_hi[1])):
    m=table&side&~mask_obj
    c=np.array([xs[m].mean(),ys[m].mean()])
    inside=box(*libero_right)
    print(f"{name}: cells={int(m.sum())} centre=({c[0]:+.3f},{c[1]:+.3f}) centre_in_plate_right_region={bool(libero_right[0]<=c[0]<=libero_right[1] and libero_right[2]<=c[1]<=libero_right[3])} region_cells_overlapping_LIBERO_box={int((m&inside).sum())}")
