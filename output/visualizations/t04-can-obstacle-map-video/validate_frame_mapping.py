"""Read-only video/trace validation; writes derived checks and sample PNGs here."""
from pathlib import Path
import base64
from io import BytesIO
import json
import numpy as np
import cv2
from PIL import Image
from frame_mapping import RUN, OUTPUT, CROPS_XYXY, read_blocks

cv2.setNumThreads(1)
blocks=read_blocks()
initial=json.loads((RUN/'trial/initial_observation.json').read_text())
sample_states={0,1,142,396,539,795,1276}
states={0:initial['observation/state']}
images={0:initial}
for line in (RUN/'trial/trace.jsonl').open():
 row=json.loads(line)
 if row.get('event')=='executed':
  state_index=row['step']+1
  states[state_index]=row['next_state']
  if state_index in sample_states:images[state_index]=row['observation']
assert len(states)==1277
mismatches=[]; max_state_error=0.
for block,line in zip(blocks,(RUN/'trial/episode.events.jsonl').open()):
 event=json.loads(line)
 error=float(np.max(np.abs(np.asarray(event['state'])-np.asarray(states[block['state_index']]))))
 max_state_error=max(max_state_error,error)
 if error>1e-12:mismatches.append({'event_line':block['event_line'],'max_abs_error':error})
video=cv2.VideoCapture(str(RUN/'trial/episode.mp4'))
metadata={'frame_count':int(video.get(cv2.CAP_PROP_FRAME_COUNT)),'fps':video.get(cv2.CAP_PROP_FPS),'width':int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),'height':int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))}
frame_by_state={0:0,**{b['state_index']:b['frame_start'] for b in blocks if b['event']=='executed'}}
image_checks=[];samples=OUTPUT/'source_frame_samples';samples.mkdir(exist_ok=True)
for state_index in sorted(sample_states):
 frame_index=frame_by_state[state_index]
 assert video.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
 ok,bgr=video.read();assert ok
 reported_next_frame=int(video.get(cv2.CAP_PROP_POS_FRAMES));assert reported_next_frame==frame_index+1
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); shown=Image.fromarray(rgb)
 filename=f'state-{state_index:04d}-source-frame-{frame_index:04d}.png'
 shown.save(samples/filename)
 comparisons={}
 for camera,box in CROPS_XYXY.items():
  key='observation/image' if camera=='external' else 'observation/wrist_image'
  source=Image.open(BytesIO(base64.b64decode(images[state_index][key].split(',',1)[1]))).convert('RGB')
  expected=source.transpose(Image.Transpose.FLIP_LEFT_RIGHT).resize((336,336),Image.Resampling.LANCZOS)
  actual=np.asarray(shown.crop(box)).astype(np.float32)
  delta=np.abs(actual-np.asarray(expected).astype(np.float32))
  comparisons[camera]={'mae_rgb_0_255':float(delta.mean()),'p95_abs_error':float(np.quantile(delta,.95)),'max_abs_error':float(delta.max())}
 image_checks.append({'state_index':state_index,'source_frame_index':frame_index,'source_time_s':frame_index/20.,'file':str(Path('source_frame_samples')/filename),'camera_comparison':comparisons})
video.release()
checks={'exact_contiguous_frame_coverage':sum(b['frame_count'] for b in blocks)==metadata['frame_count']==5436,'fps_20':metadata['fps']==20,'dashboard_1600x960':(metadata['width'],metadata['height'])==(1600,960),'event_state_matches_trace_state':not mismatches,'source_camera_crops_match_saved_observation_with_lossy_encoding':all(c['mae_rgb_0_255']<6 for s in image_checks for c in s['camera_comparison'].values())}
report={'checks':checks,'passed':all(checks.values()),'video':metadata,'matched_event_blocks':len(blocks),'max_state_absolute_error':max_state_error,'state_mismatches':mismatches,'representative_frames':image_checks,'camera_comparison_note':'Compare decoded MP4 crop to saved512x512 PNG flipped left-right and Lanczos resized336x336. Small RGB differences are expected from lossy H.264.','tools':{'python':'/workspace/gpt_field/.venv/bin/python','Pillow':True,'OpenCV':cv2.__version__,'ffmpeg':'/usr/bin/ffmpeg','ffprobe':'/usr/bin/ffprobe'}}
(OUTPUT/'frame_mapping_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'checks':checks,'max_state_absolute_error':max_state_error,'samples':[(s['state_index'],s['source_frame_index'],{k:round(v['mae_rgb_0_255'],3) for k,v in s['camera_comparison'].items()}) for s in image_checks]},indent=2))
