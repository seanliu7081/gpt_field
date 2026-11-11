"""Compose ten exact RGB pairs in five columns and two task rows."""
from pathlib import Path
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path('/workspace/gpt_field')
OUT=Path(__file__).resolve().parent
OLD=ROOT/'output/visualizations/libero10-relayout-comparison-20261105'
REMAINING=ROOT/'output/layouts/libero10_remaining_relayouts_20261105_001'
TITLES={0:'Soup + tomato sauce → basket',1:'Cream cheese + butter → basket',
        2:'Stove on + moka pot',3:'Bowl → drawer; close drawer',
        4:'Two mugs → left/right plates',5:'Book → caddy back compartment',
        6:'White mug → plate; pudding → right',7:'Soup + cream cheese → basket',
        8:'Both moka pots → stove',9:'Yellow/white mug → microwave; close'}
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(n):return ImageFont.truetype(FONT,n)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def folder(t):
 if t in (0,1,6):return OLD/f'task_{t:02d}'
 if t==2:return ROOT/'output/layouts/libero_10_t2_moka_relayout_20261105_001'
 return REMAINING/f'task_{t:02d}'
margin=28;gap=20;row_gap=28;card_w=536;card_h=1176;header=104
width=margin*2+5*card_w+4*gap
height=header+2*card_h+row_gap+margin
canvas=Image.new('RGB',(width,height),'#0d141d');draw=ImageDraw.Draw(canvas)
draw.text((margin,16),'LIBERO-10 | Original vs relayout — all 10 tasks',font=font(34),fill='#f1f5f9')
draw.text((margin,64),'Top task row: T00–T04     Bottom task row: T05–T09     Each task: original above, relayout below',font=font(22),fill='#b8c5d3')
records=[]
for task in range(10):
 row,col=divmod(task,5);x=margin+col*(card_w+gap);y=header+row*(card_h+row_gap)
 draw.rounded_rectangle((x,y,x+card_w-1,y+card_h-1),radius=10,fill='#18232e',outline='#334354',width=1)
 draw.text((x+12,y+10),f'T{task:02d}',font=font(29),fill='#f1f5f9')
 draw.text((x+12,y+48),TITLES[task],font=font(19),fill='#cbd5e1')
 record={'task_id':task,'task_row':row,'task_column':col,'images':{}}
 for label,filename,dy,colour in [('ORIGINAL','original_external.png',100,'#b8c5d3'),('RELAYOUT','relayout_external.png',650,'#5eead4')]:
  draw.text((x+12,y+dy-27),label,font=font(17),fill=colour)
  source=folder(task)/filename;im=Image.open(source).convert('RGB');assert im.size==(512,512)
  px,py=x+12,y+dy;canvas.paste(im,(px,py))
  record['images'][label.lower()]={'source':str(source),'source_sha256':sha(source),'box_xyxy':[px,py,px+512,py+512]}
 records.append(record)
path=OUT/'compare_all_10.png';canvas.save(path)
loaded=Image.open(path).convert('RGB');checks=[]
for rec in records:
 for kind,item in rec['images'].items():
  actual=np.asarray(loaded.crop(item['box_xyxy']));expected=np.asarray(Image.open(item['source']).convert('RGB'))
  checks.append({'task_id':rec['task_id'],'kind':kind,'exact_pixels':bool(np.array_equal(actual,expected))})
assert all(c['exact_pixels'] for c in checks)
manifest={'output':str(path),'dimensions_px':[width,height],'task_rows':[[0,1,2,3,4],[5,6,7,8,9]],
 'within_task_order':'original above, relayout below','all20_frames_unscaled_pixel_exact':True,'tasks':records,
 'output_sha256':sha(path),'validation':checks,'new_simulator_calls':0,'new_model_calls':0}
(OUT/'image_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'path':str(path),'size_px':[width,height],'task_rows':manifest['task_rows'],'pixel_checks_passed':len(checks)},indent=2))
