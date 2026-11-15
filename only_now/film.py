"""Render 《此刻 · ONLY NOW》.
usage:  python3 film.py still <t> <out.png>
        python3 film.py sheet <out.png> t1 t2 ...        (contact sheet, 1/4 size)
        python3 film.py frames <f0> <f1> <out.mp4>
"""
import sys, importlib, subprocess
import numpy as np
from PIL import Image
import core
from core import W, H, FPS
import timeline as TL

_mods = {}
def module(name):
    if name not in _mods:
        try: _mods[name] = importlib.import_module(name)
        except ImportError: _mods[name] = None
    return _mods[name]

def placeholder(shot, lt):
    img = np.empty((H, W, 3), np.float32); img[:] = core.VOID
    core.blit(img, core.text_sprite(f'[{shot}  {lt:5.2f}s]', 'mono_l', 30, (0.5, 0.5, 0.5)), core.CX, core.CY)
    return img

def shot_at(t):
    for s in TL.SHOTS:
        if s[0] <= t < s[1]: return s
    return TL.SHOTS[-1]

def render(t):
    s0, s1, mod, shot, src = shot_at(t)
    m = module(mod)
    k = src / (s1 - s0)                      # slowed shots play their source timing stretched
    img = m.render(shot, (t - s0) * k, src) if m is not None else placeholder(shot, t - s0)
    return core.finish(img, t, TL.SUBS, cards_abs=[(TL.at(k, 0.6), txt) for k, txt in TL.CARDS])

def main():
    mode = sys.argv[1]
    if mode == 'still':
        Image.fromarray(render(float(sys.argv[2]))).save(sys.argv[3])
    elif mode == 'sheet':
        out = sys.argv[2]; ts = [float(x) for x in sys.argv[3:]]
        tiles = [np.asarray(Image.fromarray(render(t)).resize((W // 4, H // 4), Image.BILINEAR)) for t in ts]
        cols = 4
        while len(tiles) % cols: tiles.append(np.zeros_like(tiles[0]))
        rows = [np.concatenate(tiles[i:i + cols], 1) for i in range(0, len(tiles), cols)]
        Image.fromarray(np.concatenate(rows, 0)).save(out)
    elif mode == 'frames':
        f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
                              '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium', '-crf', '16',
                              '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
        for f in range(f0, f1):
            p.stdin.write(render(f / FPS).tobytes())
            if f % 48 == 0: print(out, f, flush=True)
        p.stdin.close(); p.wait()

if __name__ == '__main__':
    main()
