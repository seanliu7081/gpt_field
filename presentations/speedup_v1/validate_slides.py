"""Render all pages and verify layout, embedded text and source integrity."""
from pathlib import Path
import sys, json, hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent.parent/'scene_test/astra_field_slides/vendor'))
import pymupdf
from PIL import Image,ImageOps,ImageDraw
pdf=ROOT/'speedup_v1_framework_methods_results_zh.pdf'
doc=pymupdf.open(pdf)
previews=ROOT/'previews';previews.mkdir(exist_ok=True)
issues=[]; text=[];fonts=set()
for i,p in enumerate(doc):
    pix=p.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5),alpha=False)
    pix.save(str(previews/f'slide_{i+1:02d}.png'))
    text.append(p.get_text())
    for f in p.get_fonts(full=True):fonts.add((f[3],f[1],f[2]))
    for block in p.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<-1 or y0<-1 or x1>p.rect.width+1 or y1>p.rect.height+1:
                    issues.append({'page':i+1,'text':span['text'],'bbox':[x0,y0,x1,y1]})
    if '\ufffd' in text[-1]:issues.append({'page':i+1,'kind':'replacement-glyph'})
for start in range(0,len(doc),6):
    end=min(start+6,len(doc));sheet=Image.new('RGB',(1200,3*366),'#E2E6EA');d=ImageDraw.Draw(sheet)
    for k,idx in enumerate(range(start,end)):
        im=Image.open(previews/f'slide_{idx+1:02d}.png').convert('RGB')
        im.thumbnail((576,324))
        x=(k%2)*600+12;y=(k//2)*366+28
        sheet.paste(im,(x,y));d.text((x,y-18),f'Page {idx+1}',fill='#182837')
    sheet.save(previews/f'contact_sheet_{start//6+1:02d}.jpg',quality=92)
manifest=json.loads((ROOT/'deck_manifest.json').read_text())
source_checks={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in manifest['source_sha256'].items()}
records=json.loads((ROOT/'layout_text.json').read_text())
layout=[r for r in records if r['x']+r['width']>960 or r['top']+r['font_size']>540]
report={'pages':len(doc),'size_bytes':pdf.stat().st_size,'all_sources_unchanged':all(source_checks.values()),
        'layout_outside_page':issues,'builder_text_outside_page':layout,'embedded_fonts':sorted(fonts),
        'extractable_text_characters':sum(map(len,text)),'expected_chinese_text_present':all(s in '\n'.join(text) for s in ['技能','距离场','规划','135','234']),
        'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest()}
(ROOT/'validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
(ROOT/'slides_text.txt').write_text('\n\n'.join(f'--- Page {i+1} ---\n{t}' for i,t in enumerate(text)))
print(json.dumps(report,ensure_ascii=False,indent=2))
assert len(doc)==24 and not issues and not layout and all(source_checks.values())
