"""Shared engine for 《此刻 · ONLY NOW》: frame geometry, fonts, compositing, letterbox, subtitles."""
import math, functools
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

FPS = 24
W, H = 1920, 1080
BAR = 140                      # 2.39:1 letterbox -> active picture is y in [BAR, H-BAR)
TOP, BOT = BAR, H - BAR
AH = BOT - TOP                 # 800
CX, CY = W / 2, (TOP + BOT) / 2

NOTO = '/usr/share/fonts/opentype/noto/'
JB = '/usr/share/fonts/truetype/jetbrains-mono/'
FONTS = {
    'serif_xl': (NOTO + 'NotoSerifCJK-ExtraLight.ttc', 2),
    'serif_l': (NOTO + 'NotoSerifCJK-Light.ttc', 2),
    'serif': (NOTO + 'NotoSerifCJK-Regular.ttc', 2),
    'serif_m': (NOTO + 'NotoSerifCJK-Medium.ttc', 2),
    'sans_l': (NOTO + 'NotoSansCJK-Light.ttc', 2),
    'sans': (NOTO + 'NotoSansCJK-Regular.ttc', 2),
    'sans_ja': (NOTO + 'NotoSansCJK-Regular.ttc', 0),
    'mono_l': (JB + 'JetBrainsMono-Light.ttf', 0),
    'mono': (JB + 'JetBrainsMono-Regular.ttf', 0),
    'mono_xl': (JB + 'JetBrainsMono-ExtraLight.ttf', 0),
}

@functools.lru_cache(maxsize=None)
def font(name, size):
    path, idx = FONTS[name]
    return ImageFont.truetype(path, size, index=idx)

# --------------------------------------------------------------------------- math helpers
def clamp(x, a=0.0, b=1.0): return min(max(x, a), b)
def lerp(a, b, t): return a + (b - a) * t
def sstep(e0, e1, x):
    if isinstance(x, np.ndarray):
        t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)
    t = clamp((x - e0) / (e1 - e0)); return t * t * (3 - 2 * t)
def ease(u):
    u = clamp(u); return u * u * u * (u * (u * 6 - 15) + 10)
def ease_in(u): u = clamp(u); return u * u * u
def ease_out(u): u = clamp(u); return 1 - (1 - u) ** 3
def fade(t, t0, t1, fin=0.4, fout=0.4):
    return sstep(t0, t0 + fin, t) * (1 - sstep(t1 - fout, t1, t))
def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255

AMBER = hexc('#F2A65A')
AMBER_HOT = hexc('#FFD9A8')
VOID = hexc('#05060D')
INK = hexc('#E9E1D6')

_grids = {}
def grid(w, h):
    if (w, h) not in _grids:
        _grids[(w, h)] = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return _grids[(w, h)]

# --------------------------------------------------------------------------- sprites / compositing
@functools.lru_cache(maxsize=4096)
def text_sprite(text, fname, size, color=(1.0, 1.0, 1.0), tracking=0, glow=0.0):
    """Premultiplied RGBA float sprite of a text line; anchor = left baseline-ish box."""
    f = font(fname, size)
    if tracking:
        widths = [f.getlength(ch) for ch in text]
        total = int(sum(widths) + tracking * (len(text) - 1)) + 4
    else:
        total = int(f.getlength(text)) + 4
    asc, desc = f.getmetrics()
    pad = 8 + int(glow * 3)
    w, h = total + 2 * pad, asc + desc + 2 * pad
    im = Image.new('L', (w, h), 0); d = ImageDraw.Draw(im)
    if tracking:
        x = pad
        for ch, cw in zip(text, widths):
            d.text((x, pad), ch, font=f, fill=255); x += cw + tracking
    else:
        d.text((pad, pad), text, font=f, fill=255)
    a = np.asarray(im).astype(np.float32) / 255
    col = np.array(color, np.float32)
    rgba = np.zeros((h, w, 4), np.float32)
    if glow > 0:
        g = cv2.GaussianBlur(a, (0, 0), glow) * 0.9
        rgba[..., :3] = col * g[..., None]; rgba[..., 3] = g * 0.0   # additive halo, no occlusion
    rgba[..., :3] = rgba[..., :3] * (1 - a[..., None]) + col * a[..., None]
    rgba[..., 3] = np.maximum(rgba[..., 3], a)
    return rgba

def blit(dst, spr, x, y, alpha=1.0, anchor='c', add=False):
    """Composite premultiplied RGBA sprite. anchor: 'c' center, 'l' left-center, 'r' right-center, 'tl'."""
    if alpha <= 0.002 or spr is None: return
    h, w = spr.shape[:2]
    if anchor == 'c': x, y = x - w / 2, y - h / 2
    elif anchor == 'l': y = y - h / 2
    elif anchor == 'r': x, y = x - w, y - h / 2
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0: return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    if add:
        d += s[..., :3] * alpha
    else:
        d *= 1 - s[..., 3:4] * alpha; d += s[..., :3] * alpha

def transform_sprite(spr, scale=1.0, angle=0.0):
    if scale == 1.0 and angle == 0.0: return spr
    h, w = spr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    nw, nh = int(w * scale * 1.45) + 4, int(h * scale * 1.45) + 4
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(spr, M, (nw, nh), flags=cv2.INTER_LINEAR)

def glow_spot(img, x, y, r, col, strength, add=True):
    """Gaussian light; only evaluated in a local window for speed."""
    R = int(r * 3)
    x0, y0, x1, y1 = max(int(x - R), 0), max(int(y - R), 0), min(int(x + R), img.shape[1]), min(int(y + R), img.shape[0])
    if x1 <= x0 or y1 <= y0: return
    gx, gy = grid(img.shape[1], img.shape[0])
    d2 = ((gx[y0:y1, x0:x1] - x) ** 2 + (gy[y0:y1, x0:x1] - y) ** 2) / (r * r)
    g = (np.exp(-d2) * strength)[..., None]
    if add: img[y0:y1, x0:x1] += col * g
    else: img[y0:y1, x0:x1] += (col - img[y0:y1, x0:x1]) * np.clip(g, 0, 1)

def bloom(img, thresh=0.75, sigma=18, strength=0.6, scale=4):
    """Cheap bloom: blur the highlights at low resolution and add back."""
    small = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale), interpolation=cv2.INTER_AREA)
    hi = np.clip(small - thresh, 0, None)
    b = cv2.GaussianBlur(hi, (0, 0), sigma / scale) + 0.5 * cv2.GaussianBlur(hi, (0, 0), sigma * 3 / scale)
    return img + cv2.resize(b, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR) * strength

def tonemap(img):
    """Soft filmic shoulder so highlights roll off instead of clipping."""
    k = 0.8
    over = np.maximum(img - k, 0)
    return np.minimum(img, k) + (1 - k) * (1 - np.exp(-over / (1 - k)))

# --------------------------------------------------------------------------- finishing
_rng = np.random.default_rng(1)
_GRAIN = [(_rng.standard_normal((AH // 2, W // 2)).astype(np.float32)) for _ in range(8)]
_yy, _xx = np.mgrid[0:AH, 0:W].astype(np.float32)
_VIG = (1 - 0.28 * np.clip(np.hypot((_xx - W / 2) / (W * 0.62), (_yy - AH / 2) / (AH * 0.85)), 0, 1.4) ** 2.4)[..., None].astype(np.float32)

SUB_ZH_Y, SUB_EN_Y = BOT + 44, BOT + 94

def subtitles(frame, t, subs):
    for sub in subs:
        t0, t1, en, zh = sub[:4]
        hard = len(sub) > 4 and sub[4] == 'cut'      # forgotten mid-word: vanishes with the picture
        if t0 - 0.1 <= t < (t1 if hard else t1 + 0.1):
            a = fade(t, t0, t1 + (1.0 if hard else 0.0), 0.18, 0.25)
            if zh: blit(frame, text_sprite(zh, 'serif', 34, (0.93, 0.90, 0.86)), W / 2, SUB_ZH_Y, alpha=a)
            if en: blit(frame, text_sprite(en, 'serif_l', 28, (0.74, 0.71, 0.67)), W / 2, SUB_EN_Y, alpha=a)

def cards(frame, t, cards_abs):
    for (t0, text) in cards_abs:
        if t0 <= t <= t0 + 3.4:
            a = fade(t, t0, t0 + 3.4, 0.8, 1.0) * 0.42
            blit(frame, text_sprite(text, 'sans_l', 20, (0.86, 0.80, 0.72), tracking=2), 112, BOT - 52, alpha=a, anchor='l')

def finish(img, t, subs, grain=0.022, vignette=True, bars=True, cards_abs=()):
    """img: float RGB full frame. Applies tonemap, vignette, grain, letterbox, subtitles. Returns uint8."""
    act = img[TOP:BOT]
    act = tonemap(act)
    if vignette: act = act * _VIG
    if grain > 0:
        g = _GRAIN[int(t * FPS) % len(_GRAIN)]
        g = cv2.resize(g, (W, AH), interpolation=cv2.INTER_LINEAR)[..., None]
        lum = act.mean(axis=2, keepdims=True)
        act = act + g * grain * (0.35 + 0.65 * np.sqrt(np.clip(lum, 0, 1)))
    out = np.zeros((H, W, 3), np.float32)
    out[TOP:BOT] = act
    if cards_abs: cards(out, t, cards_abs)
    if not bars:
        out[:TOP] = img[:TOP]; out[BOT:] = img[BOT:]
    subtitles(out, t, subs)
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)
