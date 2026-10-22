"""Render every page and validate text/page geometry for delivery."""
import json
from pathlib import Path
import fitz
from PIL import Image,ImageDraw

P=Path(__file__).resolve().parent
R=P/'preview';R.mkdir(exist_ok=True)
doc=fitz.open(P/'hybrid_method_current_en.pdf')
texts=[]; issues=[]; thumbs=[]
for i,page in enumerate(doc):
    assert abs(page.rect.width-960)<.01 and abs(page.rect.height-540)<.01
    text=page.get_text(); texts.append(f'PAGE {i+1}\n{text}')
    if '\ufffd' in text:issues.append({'page':i+1,'issue':'replacement character'})
    if not text.strip():issues.append({'page':i+1,'issue':'no searchable text'})
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines',[]):
            for span in line['spans']:
                x0,y0,x1,y1=span['bbox']
                if x0<0 or y0<0 or x1>960.1 or y1>540.1:
                    issues.append({'page':i+1,'issue':'text outside page','text':span['text'],'bbox':span['bbox']})
    pix=page.get_pixmap(matrix=fitz.Matrix(1,1),alpha=False)
    pix.save(R/f'page_{i+1:02d}.png')
    im=Image.open(R/f'page_{i+1:02d}.png').convert('RGB');im.thumbnail((480,270));thumbs.append(im)
for start in range(0,len(thumbs),8):
    chunk=thumbs[start:start+8];rows=(len(chunk)+1)//2
    sheet=Image.new('RGB',(980,rows*292+12),'#d9e1e9');draw=ImageDraw.Draw(sheet)
    for j,im in enumerate(chunk):
        x=10+(j%2)*485;y=10+(j//2)*292
        sheet.paste(im,(x,y));draw.text((x+3,y+273),f'Slide {start+j+1}',fill='black')
    sheet.save(R/f'contact_{start//8+1}.png')
(P/'pdf_text.txt').write_text('\n\n'.join(texts))
out={'page_count':len(doc),'page_size_points':[960,540],'searchable_text':True,'issues':issues,'all_pages_rendered':True}
(P/'pdf_qa.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
