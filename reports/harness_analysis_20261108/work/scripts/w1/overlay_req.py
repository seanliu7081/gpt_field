import sys, json, glob, base64, io
import numpy as np
from PIL import Image, ImageDraw
run, req_idx, step, out = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
f = sorted(glob.glob(run + '/trial/requests/*.request.json'))[req_idx - 1]
d = json.load(open(f))
last = [x for x in d['payload']['input'] if isinstance(x, dict) and x.get('role') == 'user'][-1]
labels = [c.get('text', '')[:80] for c in last['content'] if c.get('type') == 'input_text']
imgs = [c['image_url'] for c in last['content'] if c.get('type') == 'input_image']
print('labels', [l for l in labels[1:]], 'n images', len(imgs))
ctx = json.loads(last['content'][0]['text'])
print('request step', ctx.get('step') or ctx.get('fields', {}).get('step'))
arr = [np.array(Image.open(io.BytesIO(base64.b64decode(u.split(',', 1)[1]))).convert('RGB')) for u in imgs]
cand = json.load(open(run + '/trial/foundation_vision/frame_%06d_sam_candidates.json' % step))
mk = np.load(run + '/trial/foundation_vision/frame_%06d_sam_candidates.npz' % step)
pal = [(255, 0, 0), (255, 255, 0), (0, 255, 255), (255, 0, 255), (0, 255, 0), (255, 128, 0), (128, 128, 255)]
keys = sorted({c['target_key'] for c in cand['candidates']})
tiles = []
for vi, view in enumerate(('external', 'wrist')):
    img = arr[vi].astype(float)
    for c in cand['candidates']:
        if c['view'] != view: continue
        col = np.array(pal[keys.index(c['target_key']) % len(pal)], float)
        msk = mk[c['array_key']]; img[msk] = 0.5*img[msk] + 0.5*col
    im = Image.fromarray(img.astype(np.uint8)); dr = ImageDraw.Draw(im)
    for c in cand['candidates']:
        if c['view'] != view: continue
        x0, y0, x1, y1 = c['bbox_xyxy_exclusive']; col = pal[keys.index(c['target_key']) % len(pal)]
        dr.rectangle([x0, y0, x1-1, y1-1], outline=col); dr.text((x0+2, max(0, y0-10)), '%s %.2f' % (c['target_key'], c['score']), fill=(255, 255, 255))
    dr.text((5, 5), view + ' step %d' % step, fill=(255, 255, 255)); tiles.append(im)
W = Image.new('RGB', (1024, 512)); W.paste(tiles[0], (0, 0)); W.paste(tiles[1], (512, 0)); W.save(out)
print('saved', out)
