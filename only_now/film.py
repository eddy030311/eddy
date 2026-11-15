"""Render 《此刻 · ONLY NOW》.
usage:  python3 film.py still <t> <out.png>
        python3 film.py sheet <out.png> t1 t2 ...        (contact sheet, 1/4 size)
        python3 film.py frames <f0> <f1> <out.mp4>
"""
import sys, importlib, subprocess
import numpy as np, cv2
from PIL import Image
import core
from core import W, H, FPS
import timeline as TL

_mods = {}
def module(name):
    if name not in _mods:
        try: _mods[name] = importlib.import_module(name)
        except ImportError: _mods[name] = None
        if _mods[name] is not None and hasattr(_mods[name], 'set_raw'):
            _mods[name].set_raw(raw)          # the finale replays other shots inside its windows
    return _mods[name]

def placeholder(shot, lt):
    img = np.empty((H, W, 3), np.float32); img[:] = core.VOID
    core.blit(img, core.text_sprite(f'[{shot}  {lt:5.2f}s]', 'mono_l', 30, (0.5, 0.5, 0.5)), core.CX, core.CY)
    return img

# exposure lift for the shots that measured too dark (mean luma < 8 on 0–255)
GAIN = {'clocks': 1.55, 'line': 1.6, 'reveal': 2.0,
        'pushin': lambda lt: 1.0 + 0.6 * max(0.0, 1 - lt / 9.0)}

def shot_at(t):
    for s in TL.SHOTS:
        if s[0] <= t < s[1]: return s
    return TL.SHOTS[-1]

def key_of(s0):
    return [k for k in TL.START if abs(TL.START[k] - s0) < 1e-6][0]

def raw_shot(s, t):
    """Module image of shot s at absolute time t (t may run past the shot's end, for crossfades)."""
    s0, s1, mod, shot, src, off = s
    m = module(mod)
    k = src / (s1 - s0)
    img = m.render(shot, (t - s0) * k + off, src + off) if m is not None else placeholder(shot, t - s0)
    g = GAIN.get(shot)
    if g is not None: img = img * (g((t - s0) * k) if callable(g) else g)
    return img

def raw(t):
    return raw_shot(shot_at(t), t)

# chapter crossfades: incoming shot key -> seconds (the outgoing shot keeps moving underneath)
XFADE = {'drift': 1.0, 'montage': 1.0, 'line': 1.6}

def grade_flashback(img):
    lum = img.mean(axis=2, keepdims=True)
    warm = lum * np.array([1.12, 0.92, 0.70], np.float32)
    out = img * 0.45 + warm * 0.75
    out = cv2.GaussianBlur(out, (0, 0), 1.3)
    gx, gy = core.grid(W, H)
    v = (1 - 0.55 * np.clip(np.hypot((gx - core.CX) / 1000, (gy - core.CY) / 520), 0, 1) ** 2)[..., None]
    return out * v * 1.05

def compose(t):
    s = shot_at(t)
    key = key_of(s[0])
    img = raw_shot(s, t)
    # crossfade from the previous shot
    d = XFADE.get(key)
    if d and t - s[0] < d:
        i = [x[0] for x in TL.SHOTS].index(s[0])
        prev = TL.SHOTS[i - 1]
        w = core.ease((t - s[0]) / d)
        img = raw_shot(prev, t) * (1 - w) + img * w
    # flashbacks inside her chapter
    for (fs, fd, src_key, src_rel) in TL.FLASHBACKS:
        a0 = TL.at('her', fs)
        if a0 - 0.01 <= t < a0 + fd:
            u = t - a0
            a = min(1.0, u / 0.08) * min(1.0, (fd - u) / 0.15)
            fb = grade_flashback(raw(TL.at(src_key, src_rel) + u))
            flash = max(0.0, 1 - u / 0.07) * 0.35        # a breath of light as the memory cuts in
            img = img * (1 - a) + fb * a + flash
    # the heartbeat in the dark: after the last light dies, one light keeps time until the earth appears
    g0 = TL.BRIDGE_G0
    t_end = TL.at('reveal', 3.2)
    if g0 - 0.2 <= t < t_end:
        nb = np.floor((t - g0) / TL.BEAT)
        beat = (t - g0) - nb * TL.BEAT if t >= g0 else 9.0
        pulse = np.exp(-beat * 4.5)
        a = core.sstep(g0, g0 + 1.2, t) * (1 - core.sstep(TL.at('reveal', 1.0), t_end, t))
        core.glow_spot(img, core.CX, core.CY, 3.5, core.AMBER_HOT, (0.5 + 1.1 * pulse) * a)
        core.glow_spot(img, core.CX, core.CY, 26, core.AMBER, (0.08 + 0.18 * pulse) * a)
    return img

def render(t):
    return core.finish(compose(t), t, TL.SUBS, cards_abs=[(TL.at(k, 0.6), txt) for k, txt in TL.CARDS])

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
