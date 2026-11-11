"""Compose exact saved initial RGB pairs for the six remaining task relayouts."""
from pathlib import Path
import json,hashlib
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parent
TITLES={3:'T03 | Black bowl into bottom drawer; close drawer',
        4:'T04 | White mug to left plate; yellow/white mug to right plate',
        5:'T05 | Book into back compartment of caddy',
        7:'T07 | Alphabet soup and cream cheese into basket',
        8:'T08 | Both moka pots onto stove (stove starts ON)',
        9:'T09 | Yellow/white mug into microwave; close door'}
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
large=ImageFont.truetype(font,25);normal=ImageFont.truetype(font,20)
records={}
for task in TITLES:
 folder=ROOT/f'task_{task:02d}'
 data=json.loads((folder/'layout_manifest.json').read_text());assert data['passed']
 records[str(task)]={'task_id':task,'task_prompt':data['task_prompt'],'layout_directory':str(folder),
   'initial_state_sha256':data['initial_state_sha256'],'settled_state_sha256':data['settled_state_sha256'],
   'bddl_sha256':data['bddl_sha256'],'original_observation_sha256':data['original_observation_sha256'],
   'relayout_observation_sha256':data['relayout_observation_sha256'],'checks':data['checks'],
   'comparison_image':str(folder/'layout_comparison.png'),'changed_objects':[x['object'] for x in data['changes']]}
for tasks in [(3,4,5),(7,8,9)]:
 im=Image.new('RGB',(1060,1820),'#101820');d=ImageDraw.Draw(im)
 d.text((18,13),'LIBERO-10 | Original vs proposed relayout',font=large,fill='#f1f5f9')
 d.text((18,53),'ORIGINAL',font=normal,fill='#cbd5e1');d.text((546,53),'RELAYOUT',font=normal,fill='#5eead4')
 for row,task in enumerate(tasks):
  y=90+row*574
  d.text((18,y),TITLES[task],font=normal,fill='#f1f5f9')
  for x,name in [(18,'original_external.png'),(546,'relayout_external.png')]:
   frame=Image.open(ROOT/f'task_{task:02d}'/name).convert('RGB');assert frame.size==(512,512)
   im.paste(frame,(x,y+34))
 path=ROOT/('comparison_'+'_'.join(f'T{t:02d}' for t in tasks)+'.png');im.save(path)
manifest={'scope':'New free-object XY relayouts for the remaining sixLIBERO10tasks: T03,T04,T05,T07,T08,T09.',
 'original_goal_and_fixtures_preserved':True,'added_objects':False,'model_calls':0,'policy_actions':0,
 'one_initial_layout_per_task':True,'tasks':records,
 'comparison_sheets':[str(ROOT/'comparison_T03_T04_T05.png'),str(ROOT/'comparison_T07_T08_T09.png')]}
(ROOT/'batch_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'tasks':list(records),'sheets':manifest['comparison_sheets'],'model_calls':0,'policy_actions':0},indent=2))
