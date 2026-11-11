"""Repackage preserved T00/T01/T06 original-relayout images; no simulation."""
from pathlib import Path
import base64, hashlib, io, json, shutil
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
PRES=ROOT/'presentations/speedup_v2_user_check_v71_libero10_en'
PLANS=ROOT/'reports/uc7_direction_semantics_20261028/implementation_uc71/v1_planning'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def relative(p): return str(p.relative_to(ROOT))
def record(p): return {'path':relative(p),'sha256':sha(p),'bytes':p.stat().st_size}
def font(s): return ImageFont.truetype(FONT,s)
def panel(task, directory, caption):
    original=Image.open(directory/'original_external.png').convert('RGB')
    relayout=Image.open(directory/'relayout_external.png').convert('RGB')
    assert original.size==relayout.size==(512,512)
    canvas=Image.new('RGB',(1060,634),'#111820')
    d=ImageDraw.Draw(canvas)
    d.text((18,12),f'T{task:02d}  |  Original and relayout',font=font(24),fill='#f3f6fa')
    d.text((18,48),caption,font=font(17),fill='#b9c4d1')
    d.text((18,81),'ORIGINAL',font=font(17),fill='#c4ceda')
    d.text((546,81),'RELAYOUT',font=font(17),fill='#65dec8')
    canvas.paste(original,(18,108)); canvas.paste(relayout,(546,108))
    canvas.save(directory/'comparison.png')
    # Guarantee picture pixels are preserved; text lives outside them.
    assert canvas.crop((18,108,530,620)).tobytes()==original.tobytes()
    assert canvas.crop((546,108,1058,620)).tobytes()==relayout.tobytes()
    return record(directory/'comparison.png')|{'dimensions_px':[1060,634], 'image_resampling':False}

evidence=json.loads((PRES/'evidence.json').read_text())
for task in (0,1):
    directory=OUT/f'task_{task:02d}'; directory.mkdir(exist_ok=True)
    entries={}
    for kind,name in [('original',f't{task}_original_external.png'),('relayout',f't{task}_initial.png')]:
        source=PRES/'assets'/name
        meta=evidence['assets'][name]
        assert sha(source)==meta['sha256']==meta['source']['sha256']
        output=directory/f'{kind}_external.png'; shutil.copyfile(source,output)
        assert sha(source)==sha(output)
        entries[kind]={'source':record(source),'preserved_source_record':meta,'output':record(output),'method':'byte-for-byte copy; no transform','dimensions_px':[512,512]}
    caption=['Alphabet soup and tomato sauce into basket','Cream cheese and butter into basket'][task]
    provenance={'task_id':task,'task_prompt':evidence['tasks'][f't{task}']['config']['task'],'scene_kind':'relayout; object positions rearranged; not obstacle variant','evidence_manifest':record(PRES/'evidence.json'),'images':entries,'comparison':panel(task,directory,caption),'generated_by':relative(Path(__file__)),'new_simulator_actions':0,'new_model_calls':0}
    (directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

task=6
directory=OUT/f'task_{task:02d}'; directory.mkdir(exist_ok=True)
entries={}
for kind,name in [('original','libero_10_t6'),('relayout','libero_10_t6_relayout')]:
    source=PLANS/name/'review.json'; obs_source=PLANS/name/'initial_observation.json'
    review=json.loads(source.read_text()); obs=json.loads(obs_source.read_text())
    clean=next(item for item in review['preparation']['review_images'] if item['label']=='external clean')
    png=base64.b64decode(clean['data_url'].split(',',1)[1])
    rawpng=base64.b64decode(obs['observation/image'].split(',',1)[1])
    raw=Image.open(io.BytesIO(rawpng)).convert('RGB'); image=Image.open(io.BytesIO(png)).convert('RGB')
    assert raw.transpose(Image.Transpose.FLIP_LEFT_RIGHT).tobytes()==image.tobytes()
    output=directory/f'{kind}_external.png'; output.write_bytes(png)
    entries[kind]={'source':record(source),'source_json_field':'preparation.review_images[label="external clean"].data_url','observation_source':record(obs_source),'output':record(output),'method':'Decode exact saved PNG bytes from external clean review image; no transform to review image. Pixel-verified equal to saved observation/image after horizontal flip.','dimensions_px':list(image.size),'review_created_at':review['created_at'],'review_initial_observation_sha256':review['initial_observation_sha256'],'review_state_sha256':review['state_sha256'],'review_clean_matches_horizontally_flipped_observation':True}
provenance={'task_id':6,'task_prompt':review['task_prompt'],'scene_kind':'relayout; object positions rearranged; not obstacle variant','images':entries,'comparison':panel(6,directory,'White mug onto plate; chocolate pudding to right of plate'),'generated_by':relative(Path(__file__)),'new_simulator_actions':0,'new_model_calls':0}
(directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps({'generated_tasks':[0,1,6],'output_root':str(OUT),'image_dimensions':[512,512],'comparison_dimensions':[1060,634],'all_source_hash_and_pixel_checks_passed':True},indent=2))
