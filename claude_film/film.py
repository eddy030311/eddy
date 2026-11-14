"""Render the film 《Claude》.

usage:  python3 film.py frames 0 2880 out.mp4      # render a frame range to a video segment
        python3 film.py still 45.3 out.png         # render a single frame for checking
"""
import sys, math, os, subprocess
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from scipy.interpolate import PchipInterpolator

import rig
from timeline import *

FONT_SERIF = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc'
FONT_SERIF_B = '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc'
FONT_SYM = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
def font(size, bold=False):
    return ImageFont.truetype(FONT_SERIF_B if bold else FONT_SERIF, size, index=2)

CORAL = np.array([217, 119, 87], np.float32) / 255
GOLD = np.array([226, 170, 96], np.float32) / 255
INK = (58, 36, 30)

def smoothstep(e0, e1, x):
    t = min(max((x - e0) / (e1 - e0), 0.0), 1.0)
    return t * t * (3 - 2 * t)

def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * u * (u * (u * 6 - 15) + 10)

def bump(t, t0, a, hold, r):
    """0 -> 1 over `a`, hold, -> 0 over `r`."""
    if t < t0 or t > t0 + a + hold + r: return 0.0
    if t < t0 + a: return ease((t - t0) / a)
    if t < t0 + a + hold: return 1.0
    return 1 - ease((t - t0 - a - hold) / r)

def fade(t, t0, t1, fin=0.4, fout=0.4):
    return smoothstep(t0, t0 + fin, t) * (1 - smoothstep(t1 - fout, t1, t))

# ----------------------------------------------------------------------------- sprites
def text_sprite(text, size, color=INK, bold=False, halo=True, sym=False):
    f = ImageFont.truetype(FONT_SYM, size) if sym else font(size, bold)
    l, t_, r, b = f.getbbox(text)
    pad = 24
    w, h = r - l + 2 * pad, b - t_ + 2 * pad
    im = Image.new('L', (w, h), 0)
    ImageDraw.Draw(im).text((pad - l, pad - t_), text, font=f, fill=255)
    a = np.asarray(im).astype(np.float32) / 255
    rgba = np.zeros((h, w, 4), np.float32)
    col = np.array(color, np.float32) / 255
    if halo:
        g = cv2.GaussianBlur(cv2.dilate(a, np.ones((5, 5), np.uint8)), (0, 0), 5) * 0.85
        rgba[..., :3] = np.array([1.0, 0.985, 0.965]) * g[..., None]
        rgba[..., 3] = g
    rgba[..., :3] = rgba[..., :3] * (1 - a[..., None]) + col * a[..., None]
    rgba[..., 3] = rgba[..., 3] * (1 - a) + a
    return rgba  # premultiplied

def blit(dst, spr, x, y, alpha=1.0, center=True, scale=1.0, angle=0.0):
    """Composite a premultiplied RGBA sprite onto dst (float RGB) with optional scale/rotation."""
    if alpha <= 0.003: return
    if scale != 1.0 or angle != 0.0:
        h, w = spr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        nw, nh = int(w * scale * 1.5) + 4, int(h * scale * 1.5) + 4
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        spr = cv2.warpAffine(spr, M, (nw, nh), flags=cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
    if center: x, y = x - w / 2, y - h / 2
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x1 <= x0 or y1 <= y0: return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]
    d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * alpha
    d += s[..., :3] * alpha

def sunburst_sprite(radius, n=12, color=CORAL, inner=0.14, width=0.17, ss=3):
    """Claude-style radiating flower / asterisk mark, premultiplied RGBA."""
    R = int(radius * ss); size = 2 * R + 8
    m = np.zeros((size, size), np.uint8); c = size / 2
    for i in range(n):
        a = 2 * math.pi * i / n - math.pi / 2
        L = R * (1.0 if i % 2 == 0 else 0.86)
        dx, dy = math.cos(a), math.sin(a); px, py = -dy, dx
        w0 = R * width * 0.5
        pts = [(c + dx * R * inner + px * w0 * 0.55, c + dy * R * inner + py * w0 * 0.55),
               (c + dx * L * 0.55 + px * w0, c + dy * L * 0.55 + py * w0),
               (c + dx * L, c + dy * L),
               (c + dx * L * 0.55 - px * w0, c + dy * L * 0.55 - py * w0),
               (c + dx * R * inner - px * w0 * 0.55, c + dy * R * inner - py * w0 * 0.55)]
        cv2.fillPoly(m, [np.int32(np.round(pts))], 255, cv2.LINE_AA)
    cv2.circle(m, (int(c), int(c)), int(R * 0.2), 255, -1, cv2.LINE_AA)
    m = cv2.resize(m, (size // ss, size // ss), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    rgba = np.zeros(m.shape + (4,), np.float32)
    rgba[..., :3] = color * m[..., None]; rgba[..., 3] = m
    return rgba

def sunburst_partial(radius, k, rot=0.0):
    """Sunburst whose rays grow in one by one (k in 0..1)."""
    n = 12; R = int(radius * 3); size = 2 * R + 8; c = size / 2
    m = np.zeros((size, size), np.uint8)
    for i in range(n):
        g = ease(k * n - i * 0.75)
        if g <= 0: continue
        a = 2 * math.pi * i / n - math.pi / 2 + rot
        L = R * (1.0 if i % 2 == 0 else 0.86) * (0.2 + 0.8 * g)
        dx, dy = math.cos(a), math.sin(a); px, py = -dy, dx
        w0 = R * 0.17 * 0.5 * g
        pts = [(c + dx * R * 0.14 + px * w0 * 0.55, c + dy * R * 0.14 + py * w0 * 0.55),
               (c + dx * L * 0.55 + px * w0, c + dy * L * 0.55 + py * w0), (c + dx * L, c + dy * L),
               (c + dx * L * 0.55 - px * w0, c + dy * L * 0.55 - py * w0),
               (c + dx * R * 0.14 - px * w0 * 0.55, c + dy * R * 0.14 - py * w0 * 0.55)]
        cv2.fillPoly(m, [np.int32(np.round(pts))], 255, cv2.LINE_AA)
    cv2.circle(m, (int(c), int(c)), int(R * 0.2 * ease(k * 3)), 255, -1, cv2.LINE_AA)
    m = cv2.resize(m, (size // 3, size // 3), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    rgba = np.zeros(m.shape + (4,), np.float32)
    rgba[..., :3] = CORAL * m[..., None]; rgba[..., 3] = m
    return rgba

def card_sprite(img, h=240):
    k = h / img.shape[0]
    im = cv2.resize(img, (int(img.shape[1] * k), h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    b = 7
    card = np.ones((h + 2 * b, im.shape[1] + 2 * b, 3), np.float32) * np.array([1.0, 0.985, 0.965])
    card[b:b + h, b:b + im.shape[1]] = im
    cv2.rectangle(card, (0, 0), (card.shape[1] - 1, card.shape[0] - 1), (0.85, 0.62, 0.52), 1, cv2.LINE_AA)
    sh = 14
    out = np.zeros((card.shape[0] + 2 * sh, card.shape[1] + 2 * sh, 4), np.float32)
    a = np.zeros(out.shape[:2], np.float32); a[sh + 4:-sh + 4, sh + 3:-sh + 3] = 0.35
    a = cv2.GaussianBlur(a, (0, 0), 6)
    out[..., :3] = np.array([0.35, 0.18, 0.12]) * a[..., None]; out[..., 3] = a
    out[sh:-sh, sh:-sh, :3] = card; out[sh:-sh, sh:-sh, 3] = 1
    return out

def chip_sprite(sym, label):
    fs = ImageFont.truetype(FONT_SYM, 30); fl = font(32)
    ws = fs.getbbox(sym)[2]; wl = fl.getbbox(label)[2]
    w, h = ws + wl + 70, 66
    S = 3
    im = Image.new('RGBA', (w * S, h * S), (0, 0, 0, 0)); d = ImageDraw.Draw(im)
    d.rounded_rectangle((2 * S, 2 * S, (w - 2) * S, (h - 2) * S), radius=30 * S, fill=(255, 251, 246, 235),
                        outline=(217, 119, 87, 255), width=2 * S)
    im = im.resize((w, h), Image.LANCZOS); d = ImageDraw.Draw(im)
    d.text((24, h / 2), sym, font=fs, fill=(217, 119, 87, 255), anchor='lm')
    d.text((24 + ws + 14, h / 2 + 1), label, font=fl, fill=INK + (255,), anchor='lm')
    a = np.asarray(im).astype(np.float32) / 255
    a[..., :3] *= a[..., 3:4]
    sh = np.zeros((h + 30, w + 30, 4), np.float32)
    g = cv2.GaussianBlur(np.pad(a[..., 3], 15), (0, 0), 6) * 0.25
    sh[..., :3] = np.array([0.5, 0.25, 0.15]) * g[..., None]; sh[..., 3] = g
    sh = np.roll(sh, 5, axis=0)
    inner = sh[15:15 + h, 15:15 + w]
    inner[...] = inner * (1 - a[..., 3:4]) + a
    return sh

# ----------------------------------------------------------------------------- assets
class Assets:
    def __init__(self):
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        r = np.hypot((xx - W * 0.5) / W, (yy - H * 0.45) / H)
        top = np.array([0.988, 0.972, 0.950]); edge = np.array([0.952, 0.890, 0.835])
        self.base = top * (1 - np.clip(r * 1.25, 0, 1)[..., None] ** 1.6) + edge * np.clip(r * 1.25, 0, 1)[..., None] ** 1.6
        self.base = self.base.astype(np.float32)
        self.vign = (1 - 0.10 * np.clip(r * 1.4, 0, 1) ** 2.2)[..., None].astype(np.float32)
        rng = np.random.default_rng(7)
        n = 46
        self.bokeh = dict(x=rng.uniform(0, W, n), y=rng.uniform(0, H, n), v=rng.uniform(10, 34, n),
                          r=rng.uniform(2.5, 11, n), a=rng.uniform(0.10, 0.32, n), ph=rng.uniform(0, 6.28, n),
                          c=rng.integers(0, 2, n))
        self.sub_sprites = {}
        self.title = [text_sprite(ch, 150, INK, halo=False) for ch in "Claude"]
        self.title_w = [font(150).getbbox(ch)[2] - font(150).getbbox(ch)[0] for ch in "Claude"]
        self.title_adv = [font(150).getlength(ch) for ch in "Claude"]
        self.tag = text_sprite("一段关于我自己的小影片", 34, (120, 80, 66), halo=False)
        self.thanks = text_sprite("谢谢你的观看", 34, (120, 80, 66), halo=False)
        self.sun_big = sunburst_sprite(120)
        self.sun_small = sunburst_sprite(16)
        glyph_src = "我在文字里思考每一次对话都从页空白开始你说的字认真听记得" + "abcdeghklmnorstuwy"
        rng = np.random.default_rng(3)
        self.glyphs = []
        for i in range(44):
            ch = glyph_src[rng.integers(len(glyph_src))]
            sz = int(rng.uniform(26, 64))
            spr = text_sprite(ch, sz, (190, 110, 80), halo=False)
            self.glyphs.append(dict(s=spr, x=rng.uniform(0, W), y=rng.uniform(0, H), v=rng.uniform(6, 22),
                                    a=rng.uniform(0.10, 0.30), ph=rng.uniform(0, 6.28), rot=rng.uniform(-20, 20)))
        self.chips = [chip_sprite(s, l) for s, l in
                      [("✎", "写诗"), ("</>", "代码"), ("∑", "数学"), ("A⇄", "翻译"), ("…", "聊天")]]
        self.values = [(text_sprite(c, 64, INK, bold=True), text_sprite(e, 26, (160, 100, 80)))
                       for c, e in [("有帮助", "Helpful"), ("诚实", "Honest"), ("无害", "Harmless")]]
        self.qmark = text_sprite("?", 260, (217, 119, 87), halo=False)
        ex = np.asarray(Image.open('assets/expressions.png').convert('RGB'))
        cols = [(6, 314), (318, 628), (634, 940), (946, 1258), (1262, 1530)]
        rows = [(46, 505), (554, 1005)]
        self.cards = [card_sprite(ex[r0:r1, c0:c1]) for (r0, r1) in rows for (c0, c1) in cols]

    def sub(self, text):
        if text not in self.sub_sprites:
            self.sub_sprites[text] = text_sprite(text, 44, INK)
        return self.sub_sprites[text]

A = None

# ----------------------------------------------------------------------------- camera / timing
_ct = np.array([k[0] for k in CAMERA])
CAM = [PchipInterpolator(_ct, np.array([k[i] for k in CAMERA], float)) for i in (1, 2, 3)]
def camera(t):
    t = min(max(t, CAMERA[0][0]), CAMERA[-1][0])
    return float(CAM[0](t)), float(CAM[1](t)), float(CAM[2](t))

def state(t):
    """('pose', name) or ('morph', a, b, u)."""
    for i, (s0, s1, name) in enumerate(SEGMENTS):
        if s0 <= t < s1 or (i == len(SEGMENTS) - 1 and t >= s0):
            return ('pose', name)
        if i + 1 < len(SEGMENTS) and s1 <= t < SEGMENTS[i + 1][0]:
            nxt = SEGMENTS[i + 1]
            return ('morph', name, nxt[2], (t - s1) / (nxt[0] - s1))
    return ('none',)

_rng = np.random.default_rng(11)
BLINKS = []
_tb = 9.0
while _tb < 120:
    BLINKS.append(_tb); _tb += float(_rng.uniform(2.6, 4.6))
    if _rng.random() < 0.18: BLINKS.append(_tb - 0.0 + 0.32)  # occasional double blink
def blink_amt(t):
    for b in BLINKS:
        d = t - b
        if 0 <= d < 0.21:
            return [0.65, 1.0, 1.0, 1.0, 0.6][min(int(d * FPS), 4)]
    return 0.0

PUNCT = set("，。—…、「」？！ ")
def speaking(t):
    for s0, s1, text in SUBS:
        n = len(text)
        if s0 + 0.05 <= t < min(s1, s0 + n / CHARS_PER_SEC + 0.1):
            i = int((t - s0) * CHARS_PER_SEC)
            ch = text[min(i, n - 1)]
            if ch in PUNCT or ch.isascii() and ch.isalpha() is False: return 0.0
            o = 0.55 + 0.35 * math.sin(2 * math.pi * 4.3 * t) + 0.25 * math.sin(2 * math.pi * 6.9 * t + 1.3)
            return float(np.clip(o, 0, 1))
    return 0.0

def pose_params(name, t):
    p = dict(breath=math.sin(2 * math.pi * t / 3.9),
             sway=0.35 * math.sin(2 * math.pi * t / 7.3) + 0.15 * math.sin(2 * math.pi * t / 3.1),
             hair=1.0, hair_phase=0.0,
             head_rot=0.022 * math.sin(2 * math.pi * t / 6.1),
             head_nod=0.25 * math.sin(2 * math.pi * t / 4.7),
             blink=blink_amt(t), mouth=speaking(t))
    ax, ay = 0.0, 0.0
    if name == 'full':
        p['head_nod'] += 1.2 * bump(t, 10.0, 0.25, 0.1, 0.4)          # hello nod
        p['head_rot'] += 0.07 * bump(t, 22.8, 0.8, 2.6, 0.9)          # curious tilt
        p['head_rot'] += -0.065 * bump(t, 110.2, 0.7, 7.0, 1.0)       # "想聊些什么呢？"
        p['head_nod'] += 0.8 * bump(t, 105.6, 0.3, 0.3, 0.6)
        p['sway'] += 0.6 * math.sin(2 * math.pi * t / 9.0)
    elif name == 'think':
        p['head_rot'] += -0.04 * bump(t, 31.6, 0.8, 3.5, 0.8)
        p['head_nod'] += 1.0 * bump(t, 36.6, 0.7, 3.0, 0.8)           # looking down, a little sad
        p['head_rot'] += 0.035 * bump(t, 80.4, 0.7, 3.8, 0.8)
        p['head_nod'] += 0.7 * bump(t, 84.2, 0.3, 0.2, 0.5)
    elif name == 'calm':
        p['head_nod'] += 0.6 * bump(t, 41.4, 0.9, 2.0, 0.9)
        p['head_rot'] += 0.03 * bump(t, 41.4, 1.0, 2.0, 1.0)
    elif name == 'offer':
        u = t - SEGMENTS[3][0]
        ay += 0.35 * math.exp(-u * 3.5) * math.cos(u * 9)              # settle after the reach
        ay += -0.07 * math.sin(2 * math.pi * t / 2.7); ax += 0.04 * math.sin(2 * math.pi * t / 4.1)
        for tc in (50.4, 51.8, 53.2, 54.8, 56.6):                      # lift the palm as each idea floats up
            ay += -0.28 * bump(t, tc - 0.15, 0.15, 0.05, 0.45)
        p['head_rot'] += 0.06 * bump(t, 59.2, 0.8, 5.2, 1.0)
        p['head_nod'] += 0.5 * bump(t, 61.5, 0.3, 0.2, 0.5)
    elif name == 'point':
        u = t - SEGMENTS[4][0]
        ax += -0.32 * math.exp(-u * 4) * math.sin(u * 11 + 0.6)        # overshoot after the swing
        for tc in (70.6, 75.1):
            ax += -0.22 * bump(t, tc, 0.12, 0.15, 0.5)
            p['head_nod'] += 0.7 * bump(t, tc, 0.12, 0.15, 0.5)
        p['head_rot'] += -0.03 * bump(t, 75.0, 0.6, 3.6, 0.7)
    elif name == 'heart':
        p['head_nod'] += 0.6 * bump(t, 91.4, 0.6, 1.5, 0.8) + 0.5 * bump(t, 96.0, 0.8, 2.5, 0.8)
        p['head_rot'] += 0.045 * bump(t, 100.3, 0.8, 3.0, 0.8)
    p['arm'] = (ax, ay)
    return p

# ----------------------------------------------------------------------------- character
CW, CH, CMX, CMY = 1000, 1900, 500, 250   # normalized morph canvas (64 units between the eyes)

def norm_affine(name):
    P = rig.POSES[name]; k = 64.0 / P['io']
    return np.float32([[k, 0, CMX - k * P['mid'][0]], [0, k, CMY - k * P['mid'][1]]])

def to_screen(t, M_local):
    cx, cy, s = camera(t)
    C = np.float64([[s, 0, cx - s * CMX], [0, s, cy - s * CMY], [0, 0, 1]])
    Ml = np.vstack([M_local, [0, 0, 1]]).astype(np.float64)
    return (C @ Ml)[:2].astype(np.float32)

_flows = {}
def flows(a, b):
    key = (a, b)
    if key in _flows: return _flows[key]
    path = f'build/flow_{a}_{b}.npz'
    if os.path.exists(path):
        z = np.load(path); _flows[key] = (z['fab'].astype(np.float32), z['fba'].astype(np.float32)); return _flows[key]
    A_ = cv2.warpAffine(rig.load(a), norm_affine(a), (CW, CH)); B_ = cv2.warpAffine(rig.load(b), norm_affine(b), (CW, CH))
    def gray(x): return cv2.cvtColor((np.clip(x[..., :3] + (1 - x[..., 3:]), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    fab = dis.calc(gray(A_), gray(B_), None); fba = dis.calc(gray(B_), gray(A_), None)
    np.savez_compressed(path, fab=fab.astype(np.float16), fba=fba.astype(np.float16))
    _flows[key] = (fab, fba)
    return _flows[key]

def character(t):
    """Premultiplied RGBA layer of the character in screen space, plus hand position helper."""
    st = state(t)
    if st[0] == 'none': return None
    if st[0] == 'pose':
        name = st[1]
        img = rig.deform(name, t, pose_params(name, t))
        M = to_screen(t, norm_affine(name))
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR)
    _, a, b, u = st
    fab, fba = flows(a, b)
    Aimg = cv2.warpAffine(rig.deform(a, t, pose_params(a, t)), norm_affine(a), (CW, CH))
    Bimg = cv2.warpAffine(rig.deform(b, t, pose_params(b, t)), norm_affine(b), (CW, CH))
    gx, gy = rig.grid(CW, CH)
    e = ease(u)
    blur = 0.11 * math.sin(math.pi * u)
    acc = 0
    samples = np.linspace(e - blur, e + blur, 7) if blur > 0.01 else [e]
    for ts in samples:
        ts = float(np.clip(ts, 0, 1))
        At = cv2.remap(Aimg, gx - ts * fab[..., 0], gy - ts * fab[..., 1], cv2.INTER_LINEAR)
        Bt = cv2.remap(Bimg, gx - (1 - ts) * fba[..., 0], gy - (1 - ts) * fba[..., 1], cv2.INTER_LINEAR)
        w = smoothstep(0.38, 0.62, ts)
        acc = acc + At * (1 - w) + Bt * w
    img = acc / len(samples)
    I = np.float32([[1, 0, 0], [0, 1, 0]])
    return cv2.warpAffine(img, to_screen(t, I), (W, H), flags=cv2.INTER_LINEAR)

def hand_screen(t):
    """Screen position of the open palm in the 'offer' pose."""
    P = rig.POSES['offer']; hx, hy = P['arm']['hand']
    p = pose_params('offer', t); ax, ay = p['arm']
    hx += ax * P['io'] * 0.9; hy += ay * P['io'] * 0.9 - 25
    M = to_screen(t, norm_affine('offer'))
    return M[0, 0] * hx + M[0, 1] * hy + M[0, 2], M[1, 0] * hx + M[1, 1] * hy + M[1, 2]

# ----------------------------------------------------------------------------- background
def background(t, cx, cy):
    img = A.base.copy()
    # slow sunburst rays behind the character
    lw, lh = W // 4, H // 4
    rays = np.zeros((lh, lw), np.float32)
    n = 18; rot = t * 0.035
    ccx, ccy = cx / 4, (cy - 40) / 4
    for i in range(n):
        a0 = rot + 2 * math.pi * i / n; a1 = a0 + math.pi / n * 0.55
        pts = np.float32([[ccx, ccy], [ccx + 900 * math.cos(a0), ccy + 900 * math.sin(a0)],
                          [ccx + 900 * math.cos(a1), ccy + 900 * math.sin(a1)]])
        cv2.fillPoly(rays, [np.int32(pts)], 1.0, cv2.LINE_AA)
    yy, xx = rig.grid(lw, lh)
    rr = np.hypot(xx - ccx, yy - ccy)
    rays *= np.exp(-rr / 220.0) * 0.07
    glow = np.exp(-(rr / 95.0) ** 2) * 0.10
    lay = cv2.resize(cv2.GaussianBlur(rays, (0, 0), 1.2), (W, H))[..., None]
    gl = cv2.resize(glow, (W, H))[..., None]
    img = img * (1 - lay) + CORAL * lay
    img = img + (np.array([1.0, 0.9, 0.75]) - img) * gl
    # bokeh dust, drawn at half resolution
    hw, hh = W // 2, H // 2
    dust = np.zeros((hh, hw, 3), np.float32)
    B = A.bokeh
    for i in range(len(B['x'])):
        x = (B['x'][i] + 30 * math.sin(t * 0.4 + B['ph'][i])) % W
        y = (B['y'][i] - B['v'][i] * t) % (H + 60) - 30
        col = CORAL if B['c'][i] else GOLD
        a = B['a'][i] * (0.6 + 0.4 * math.sin(t * 1.3 + B['ph'][i]))
        cv2.circle(dust, (int(x / 2), int(y / 2)), int(B['r'][i] / 2 + 1), tuple(float(v) for v in col * a), -1, cv2.LINE_AA)
    dust = cv2.resize(cv2.GaussianBlur(dust, (0, 0), 1.5), (W, H))
    img = img * (1 - dust.max(axis=2, keepdims=True) * 0.9) + dust * 0.9 / np.maximum(dust.max(axis=2, keepdims=True), 1e-4) * dust.max(axis=2, keepdims=True)
    return img

# ----------------------------------------------------------------------------- overlays
def draw_title(img, t, t0, alpha_all, y=470, tagline=True, sun_k=1.0):
    total = sum(A.title_adv); x = W / 2 - total / 2
    for i, spr in enumerate(A.title):
        k = ease((t - t0 - 0.12 * i) / 0.9)
        blit(img, spr, x + A.title_adv[i] / 2, y + 30 * (1 - k), alpha=k * alpha_all)
        x += A.title_adv[i]
    if tagline:
        k = ease((t - t0 - 1.2) / 1.0)
        blit(img, A.tag, W / 2, y + 125, alpha=k * alpha_all)
        lw = int(300 * k)
        if lw > 2 and alpha_all > 0:
            ov = img[y + 90:y + 92, W // 2 - lw // 2:W // 2 + lw // 2]
            ov[...] = ov * (1 - 0.6 * alpha_all) + CORAL * 0.6 * alpha_all

_band = None
def subtitle_band():
    global _band
    if _band is None:
        yy, xx = rig.grid(W, 200)
        a = np.exp(-((yy - 100) / 52) ** 4) * np.exp(-((xx - W / 2) / 620) ** 6) * 0.78
        _band = a[..., None].astype(np.float32)
    return _band

def subtitles(img, t):
    vis = 0.0
    for s0, s1, text in SUBS:
        vis = max(vis, fade(t, s0 - 0.2, s1 + 0.3, 0.4, 0.5))
    if vis > 0:
        b = subtitle_band() * vis
        reg = img[880:1080]
        reg[...] = reg * (1 - b) + np.array([0.992, 0.978, 0.958]) * b
    for s0, s1, text in SUBS:
        if s0 - 0.05 <= t <= s1 + 0.05:
            spr = A.sub(text)
            al = fade(t, s0, s1, 0.25, 0.35)
            # typewriter: reveal left to right with a soft edge
            n = len(text); prog = (t - s0) * CHARS_PER_SEC / n
            hgt, wid = spr.shape[:2]
            edge = 26 + prog * (wid - 26) * 1.0
            xs = np.arange(wid, dtype=np.float32)
            m = np.clip((edge - xs) / 26.0, 0, 1)[None, :, None]
            blit(img, spr * m, W / 2, 1000, alpha=al)

def scene_overlays_back(img, t):
    # floating glyphs (thinking in words)
    al = fade(t, 27.0, 45.2, 1.5, 1.2)
    if al > 0:
        for g in A.glyphs:
            x = (g['x'] + 14 * math.sin(t * 0.3 + g['ph'])) % W
            y = (g['y'] - g['v'] * t) % (H + 100) - 50
            if 560 < x < 960 and y < 900: continue  # keep the face area clean
            blit(img, g['s'], x, y, alpha=al * g['a'] * (0.6 + 0.4 * math.sin(t * 0.8 + g['ph'])),
                 angle=g['rot'] + 6 * math.sin(t * 0.2 + g['ph']))
    # big soft question mark
    al = fade(t, 75.0, 79.6, 0.8, 0.5)
    if al > 0:
        blit(img, A.qmark, 330 + 10 * math.sin(t * 1.3), 430 + 12 * math.sin(t * 0.9), alpha=al * 0.22,
             angle=-8 + 4 * math.sin(t * 0.7))
    # orbiting expression cards: back half
    orbit_cards(img, t, back=True)

def orbit_cards(img, t, back):
    al = fade(t, 104.2, 116.2, 1.2, 1.4)
    if al <= 0: return
    n = len(A.cards)
    cx, cy, s = camera(t)
    for i, card in enumerate(A.cards):
        ang = 2 * math.pi * i / n + (t - 104) * 0.16
        depth = math.sin(ang)          # >0 front
        if (depth > 0) == back: continue
        x = W / 2 + 760 * math.cos(ang)
        y = 470 + 170 * math.sin(ang) + 10 * math.sin(t * 1.1 + i)
        sc = 0.66 + 0.34 * (depth + 1) / 2
        k = ease((t - 104.4 - 0.12 * i) / 0.8)
        a = al * k * (0.55 + 0.45 * (depth + 1) / 2)
        if not back and abs(x - W / 2) < 260: a *= 0.35   # don't hide her face
        blit(img, card, x, y + 40 * (1 - k), alpha=a, scale=sc, angle=6 * math.cos(ang * 1.3))

def scene_overlays_front(img, t):
    # opening: sunburst + title
    if t < 9.2:
        k = ease((t - 0.6) / 2.6)
        zoom = 1 + 6 * ease((t - 7.0) / 1.6)
        al = 1 - ease((t - 7.4) / 1.2)
        if k > 0 and al > 0:
            spr = sunburst_partial(120, k, rot=t * 0.15)
            blit(img, spr, W / 2, 300, alpha=al, scale=min(zoom, 3.0))
        draw_title(img, t, 2.4, 1 - ease((t - 6.6) / 1.0), y=560)
    # emphasis marks near her head (pointing scene)
    for tc in (66.7, 70.6):
        al = fade(t, tc, tc + 1.0, 0.08, 0.4)
        if al > 0:
            cx, cy, s = camera(t)
            hx, hy = cx - 150 * s, cy - 120 * s
            for j, ang in enumerate((-150, -120, -95)):
                r0 = 30 + 25 * ease((t - tc) / 0.3); r1 = r0 + 30
                a = math.radians(ang)
                p0 = (int(hx + r0 * math.cos(a)), int(hy + r0 * math.sin(a)))
                p1 = (int(hx + r1 * math.cos(a)), int(hy + r1 * math.sin(a)))
                ov = img.copy(); cv2.line(ov, p0, p1, tuple(float(v) for v in CORAL), 6, cv2.LINE_AA)
                img[...] = img * (1 - al) + ov * al
    # idea chips floating up from the open palm
    slots = [(330, 260), (560, 170), (300, 470), (560, 380), (420, 640)]
    for i, tc in enumerate((50.4, 51.8, 53.2, 54.8, 56.6)):
        if tc <= t < 66.3:
            k = ease((t - tc) / 1.1)
            hx, hy = hand_screen(min(t, tc + 0.01) if k < 1 else t)
            hx, hy = hand_screen(tc)
            sx, sy = slots[i]
            sx += 12 * math.sin(t * 0.9 + i); sy += 10 * math.sin(t * 1.2 + i * 2)
            x = hx + (sx - hx) * k; y = hy + (sy - hy) * k - 80 * math.sin(math.pi * k)
            al = min(1, (t - tc) / 0.25) * (1 - smoothstep(65.4, 66.2, t))
            blit(img, A.chips[i], x, y, alpha=al, scale=0.5 + 0.5 * k)
            if k < 1:  # sparkle trail
                blit(img, A.sun_small, x - 40, y + 20, alpha=al * (1 - k), angle=t * 90, scale=1.2)
    # the three values
    for i, tc in enumerate((88.4, 90.2, 92.6)):
        al = fade(t, tc, 103.6, 0.7, 0.8)
        if al > 0:
            k = ease((t - tc) / 0.9)
            y = 300 + 175 * i
            x = 1340 + 40 * (1 - k)
            blit(img, A.sun_small, x - 110, y, alpha=al, angle=(t - tc) * 25, scale=1.0 + 0.25 * math.sin(t * 2 + i))
            blit(img, A.values[i][0], x + 30, y - 8, alpha=al)
            blit(img, A.values[i][1], x + 30, y + 52, alpha=al * 0.9)
    if 95.9 < t < 103.6:  # gentle connecting line
        al = fade(t, 95.9, 103.6, 1.0, 0.8)
        y0, y1 = 300, 650
        ov = img.copy(); cv2.line(ov, (1230, y0), (1230, int(y0 + (y1 - y0) * ease((t - 95.9) / 1.5))),
                                  tuple(float(v) for v in CORAL), 2, cv2.LINE_AA)
        img[...] = img * (1 - al * 0.6) + ov * al * 0.6
    orbit_cards(img, t, back=False)
    # ending card
    if t > 116.0:
        al = ease((t - 116.0) / 1.2)
        k = ease((t - 116.2) / 1.6)
        blit(img, A.sun_big, W / 2, 330, alpha=al, angle=(t - 116) * 8, scale=0.6 + 0.1 * k)
        draw_title(img, t, 116.4, al, y=600, tagline=False)
        blit(img, A.thanks, W / 2, 735, alpha=ease((t - 117.6) / 1.0))

# ----------------------------------------------------------------------------- frame
def render(t):
    cx, cy, s = camera(t)
    img = background(t, cx, cy)
    scene_overlays_back(img, t)
    ch = character(t)
    if ch is not None:
        vis = smoothstep(8.2, 9.6, t) * (1 - smoothstep(115.4, 116.8, t))
        if vis > 0:
            if t < 9.8:  # rise in
                dy = 40 * (1 - ease((t - 8.2) / 1.6))
                ch = cv2.warpAffine(ch, np.float32([[1, 0, 0], [0, 1, dy]]), (W, H))
            ch *= vis
            # soft warm aura from the silhouette
            a4 = cv2.resize(ch[..., 3], (W // 8, H // 8), interpolation=cv2.INTER_AREA)
            aura = cv2.resize(cv2.GaussianBlur(a4, (0, 0), 4), (W, H))[..., None] * 0.22
            img = img * (1 - aura) + np.array([1.0, 0.86, 0.70]) * aura
            img = img * (1 - ch[..., 3:4]) + ch[..., :3]
    scene_overlays_front(img, t)
    subtitles(img, t)
    for tp, dur in FLASHES:
        a = 1 - min(abs(t - tp) / (dur / 2), 1)
        if a > 0:
            a = ease(a)
            img = img * (1 - a) + np.array([1.0, 0.975, 0.95]) * a
    img = img * A.vign
    # global fade in/out
    g = smoothstep(0.0, 0.8, t) * (1 - smoothstep(119.0, 120.0, t) * 0.0)
    img = img * g + np.array([0.988, 0.972, 0.950]) * (1 - g)
    return np.clip(img * 255 + 0.5, 0, 255).astype(np.uint8)

def main():
    global A
    A = Assets()
    mode = sys.argv[1]
    if mode == 'still':
        t = float(sys.argv[2]); out = sys.argv[3]
        Image.fromarray(render(t)).save(out)
    elif mode == 'flows':
        for i in range(len(SEGMENTS) - 1):
            a, b = SEGMENTS[i][2], SEGMENTS[i + 1][2]
            if SEGMENTS[i][1] not in HARD_CUTS: flows(a, b); print('flow', a, b)
    elif mode == 'frames':
        f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        p = subprocess.Popen(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
                              '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-', '-c:v', 'libx264', '-preset', 'medium',
                              '-crf', '18', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
        for f in range(f0, f1):
            p.stdin.write(render(f / FPS).tobytes())
            if f % 48 == 0: print(out, f, flush=True)
        p.stdin.close(); p.wait()

if __name__ == '__main__':
    main()
