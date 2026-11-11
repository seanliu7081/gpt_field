import json, base64, io, time, sys
from PIL import Image
import PIL, zlib
p = '/workspace/gpt_field/runs/agent_speedup_v2_uc71_can_obstacle_transit100_t04_20261103_001/trial/initial_observation.json'
obs = json.load(open(p))
urls = [obs['observation/image'], obs['observation/wrist_image']]
imgs = [Image.open(io.BytesIO(base64.b64decode(u.split(',', 1)[1]))).convert('RGB') for u in urls]
print('Pillow', PIL.__version__, 'zlib', zlib.ZLIB_VERSION, 'sizes', [i.size for i in imgs])
def enc(im):
    b = io.BytesIO(); im.save(b, format='PNG'); return 'data:image/png;base64,' + base64.b64encode(b.getvalue()).decode('ascii')
def dec(u):
    im = Image.open(io.BytesIO(base64.b64decode(u.split(',', 1)[1]))); im.load(); return im
N = 30
for name, fn in [('2x encode+b64', lambda: [enc(i) for i in imgs]), ('2x decode', lambda: [dec(u) for u in urls])]:
    fn(); t = time.perf_counter()
    for _ in range(N): fn()
    print(name, '%.1f ms/step' % ((time.perf_counter() - t) / N * 1000))
