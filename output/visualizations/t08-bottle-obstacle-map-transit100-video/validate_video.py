"""Validate the encoded artifact against its source-derived frames; no replay."""
from pathlib import Path
import hashlib,json,subprocess
import cv2
import numpy as np
from PIL import Image

out=Path(__file__).resolve().parent
manifest=json.loads((out/'video_manifest.json').read_text())
data=json.loads((out/'obstacle_map_timeline.json').read_text())
video=out/manifest['video']
p=subprocess.run(['ffprobe','-v','error','-show_streams','-show_format','-show_chapters','-of','json',str(video)],capture_output=True,text=True,check=True)
probe=json.loads(p.stdout)
(out/'video_probe.json').write_text(json.dumps(probe,ensure_ascii=False,indent=2)+'\n')
stream=next(s for s in probe['streams'] if s['codec_type']=='video')
assert (stream['width'],stream['height'])==(manifest['width'],manifest['height'])
assert int(stream['nb_frames'])==manifest['frames']
assert abs(float(probe['format']['duration'])-manifest['duration_s'])<.06
assert len(probe['chapters'])==len(manifest['chapters'])
assert not manifest['terminal_diagnostic_actions_included']
cap=cv2.VideoCapture(str(video));checks=[]
selected={0,38,82,87,224,538,597,843}
for chapter in manifest['chapters']:
 state=chapter['control_state']
 if state not in selected:continue
 n=round(chapter['video_time_s']*manifest['fps'])+1
 cap.set(cv2.CAP_PROP_POS_FRAMES,n);ok,bgr=cap.read();assert ok
 rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
 name='preview-initial.png' if chapter.get('initial_observation') else f'preview-step-{state:04d}.png'
 expected=np.array(Image.open(out/name).convert('RGB'))
 mae=float(np.mean(np.abs(rgb.astype(float)-expected.astype(float))))
 assert mae<5,(state,mae)
 checks.append({'control_state':state,'initial_observation':bool(chapter.get('initial_observation')),'video_frame':n,'rgb_mean_absolute_error_vs_preview':round(mae,4)})
cap.release()
subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
run=Path(data['source_run'])
if not run.is_absolute():run=out.parents[2]/run
trace=run/'trial/trace.jsonl'
assert hashlib.sha256(trace.read_bytes()).hexdigest()==data['source_trace_sha256']
result={'passed':True,'video':video.name,'frames':manifest['frames'],'duration_s':manifest['duration_s'],'dimensions':[stream['width'],stream['height']],'chapters':len(probe['chapters']),'decoded_keyframes':checks,'source_trace_unchanged':True,'no_simulator_or_model_calls':True,'full_video_decode':'passed'}
(out/'video_qa.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['passed','frames','duration_s','chapters','full_video_decode']},indent=2))
