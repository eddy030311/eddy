"""S11 'her': the character, seen only in fragments — lashes, a hand on the book, blank pages, the open palm."""
import math, os, sys
import numpy as np, cv2
from core import *
from ui import cursor_alpha

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'claude_film'))
import rig   # pose layers are read from ./build/<pose>.png

def _clean(name):
    """Remove the white fringe left by the original white background (visible at close-up scales)."""
    if getattr(rig, '_cleaned', None) is None: rig._cleaned = set()
    if name in rig._cleaned: return
    src = rig.load(name)
    a = src[..., 3]
    a2 = cv2.erode(a, np.ones((3, 3), np.uint8), iterations=2)
    a2 = cv2.GaussianBlur(a2, (0, 0), 0.7)
    rgb = src[..., :3] / np.maximum(a[..., None], 1e-4)
    # pixels near the edge that are pale (fringe) are pulled toward the dark outline colour
    edge = np.clip(a - cv2.erode(a, np.ones((3, 3), np.uint8), iterations=3), 0, 1)[..., None]
    pale = np.clip((rgb.min(axis=2, keepdims=True) - 0.6) / 0.3, 0, 1)
    rgb = rgb * (1 - edge * pale) + np.array([0.18, 0.10, 0.08]) * edge * pale
    out = np.dstack([rgb * a2[..., None], a2]).astype(np.float32)
    rig._cache[name] = np.ascontiguousarray(out)
    rig._cleaned.add(name)

PALM_LIGHT = (96, 318)          # offer-pose local point where the light hovers above the palm
END_POINT = (960, 560)          # where the light must sit at the cut to 'relight'

def params(t):
    return dict(breath=math.sin(2 * math.pi * t / 4.2), sway=0.25 * math.sin(2 * math.pi * t / 8.1),
                hair=0.8, head_rot=0.012 * math.sin(2 * math.pi * t / 7.0), head_nod=0.15 * math.sin(2 * math.pi * t / 5.3),
                blink=0.0, mouth=0.0, arm=(0.0, -0.05 * math.sin(2 * math.pi * t / 3.3)))

_bg = None
def room(lt, warm=1.0):
    """Warm, dim, out-of-focus room: a lamp pool and drifting dust."""
    global _bg
    if _bg is None:
        gx, gy = grid(W, H)
        r = np.hypot((gx - W * 0.62) / W, (gy - H * 0.42) / H)
        base = np.array([0.20, 0.115, 0.06]) * np.exp(-(r / 0.42) ** 2)[..., None] + np.array([0.022, 0.016, 0.02])
        _bg = base.astype(np.float32)
    img = _bg * warm
    rng = np.random.default_rng(9)
    dust = np.zeros((H // 2, W // 2, 3), np.float32)
    for i in range(70):
        x = (rng.uniform(0, W) + 9 * lt * rng.uniform(-1, 1) + 20 * math.sin(lt * 0.3 + i)) % W
        y = (rng.uniform(0, H) - 6 * lt * rng.uniform(0.2, 1)) % H
        r_ = rng.uniform(1.5, 9); a = rng.uniform(0.04, 0.16) * (0.6 + 0.4 * math.sin(lt * 0.9 + i))
        cv2.circle(dust, (int(x / 2), int(y / 2)), max(1, int(r_ / 2)), tuple(float(v) for v in AMBER_HOT * a), -1, cv2.LINE_AA)
    return img + cv2.resize(cv2.GaussianBlur(dust, (0, 0), 2.5), (W, H))

def place(img, name, t, focus, s, center, dof=1.0, focus_r=260, light=1.0):
    """Render pose `name` so that pose-local `focus` lands on screen `center` at scale s, with shallow focus."""
    _clean(name)
    p = params(t)
    lay = rig.deform(name, t, p)
    M = np.float32([[s, 0, center[0] - s * focus[0]], [0, s, center[1] - s * focus[1]]])
    lay = np.clip(cv2.warpAffine(lay, M, (W, H), flags=cv2.INTER_CUBIC), 0, None)
    # warm key light from upper right, cooler falloff
    gx, gy = grid(W, H)
    key = (0.55 + 0.6 * np.exp(-(((gx - W * 0.68) / 900) ** 2 + ((gy - H * 0.3) / 700) ** 2)))[..., None]
    tint = np.array([1.0, 0.86, 0.72], np.float32)
    rgb = lay[..., :3] * key * tint * light
    a = lay[..., 3:4]
    # shallow depth of field around the focus point
    if dof > 0:
        d = np.hypot(gx - center[0], gy - center[1])
        wb = (sstep(focus_r, focus_r * 2.4, d) * dof)[..., None]
        small = (W // 2, H // 2)
        rb = cv2.resize(cv2.GaussianBlur(cv2.resize(rgb, small), (0, 0), 5), (W, H))
        ab = cv2.resize(cv2.GaussianBlur(cv2.resize(a[..., 0], small), (0, 0), 5), (W, H))[..., None]
        rgb = rgb * (1 - wb) + rb * wb; a = a * (1 - wb) + ab * wb
    img *= 1 - a
    img += rgb
    return img

def blank_pages(lt):
    """Insert: an open book, both pages blank, lamp-lit; a faint caret on the right page."""
    img = room(lt, 0.6)
    S = 2
    m = np.zeros((H * S // 2, W * S // 2), np.float32)
    shade = np.zeros_like(m)
    cx, cy = W * S // 4, int(H * S // 4 + 20)
    pw, ph = 250 * S, 300 * S
    for side in (-1, 1):
        pts = []
        for k in np.linspace(0, 1, 40):
            x = cx + side * k * pw
            y = cy - ph / 2 + 18 * S * math.sin(k * math.pi) * 0.6 - (1 - k) ** 6 * 30 * S
            pts.append((x, y))
        for k in np.linspace(1, 0, 40):
            x = cx + side * k * pw
            y = cy + ph / 2 + 10 * S * math.sin(k * math.pi) * 0.4 - (1 - k) ** 6 * 12 * S
            pts.append((x, y))
        cv2.fillPoly(m, [np.int32(pts)], 1.0, cv2.LINE_AA)
    gx, gy = grid(m.shape[1], m.shape[0])
    gutter = np.exp(-((gx - cx) / (60 * S)) ** 2) * 0.45
    m = cv2.resize(m, (W, H), interpolation=cv2.INTER_AREA)
    gutter = cv2.resize(gutter.astype(np.float32), (W, H))
    paper = np.array([0.93, 0.86, 0.74], np.float32) * (1 - gutter[..., None])
    gxx, gyy = grid(W, H)
    lamp = (0.45 + 0.75 * np.exp(-(((gxx - W * 0.6) / 700) ** 2 + ((gyy - H * 0.35) / 520) ** 2)))[..., None]
    shadow = cv2.GaussianBlur(m, (0, 0), 18)[..., None] * 0.6
    img = img * (1 - shadow)
    cover = cv2.dilate(m, np.ones((13, 13), np.uint8))[..., None]          # the book's cover under the pages
    img = img * (1 - cover) + np.array([0.16, 0.07, 0.05]) * lamp * cover
    img = img * (1 - m[..., None]) + paper * lamp * m[..., None] * 0.95
    a = cursor_alpha(lt + 0.3)
    x0, y0 = int(CX + 70), int(CY - 120)
    img[y0:y0 + 26, x0:x0 + 3] = img[y0:y0 + 26, x0:x0 + 3] * (1 - a * 0.8) + AMBER * 0.7 * a * 0.8
    return img

def palm_light(img, x, y, lt, scale=1.0, strength=1.0, roundness=0.0):
    a = (0.55 + 0.45 * cursor_alpha(lt)) * (1 - roundness) + roundness
    glow_spot(img, x, y, 70 * scale, AMBER, 0.55 * a * strength)
    glow_spot(img, x, y, 22 * scale, AMBER_HOT, 1.1 * a * strength)
    hw, hh = max(1, int(3 * scale)), int(20 * scale * (1 - roundness) + 3 * scale * roundness)
    y0, y1 = max(int(y - hh), 0), min(int(y + hh), H); x0, x1 = max(int(x - hw), 0), min(int(x + hw), W)
    if y1 > y0 and x1 > x0:
        img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a * strength) + AMBER_HOT * 1.6 * a * strength

def render(shot, lt, dur):
    t = lt + 190.0
    if lt < 6.5:                               # F1 lashes: closed eyes, listening
        u = lt / 6.5
        img = room(lt)
        P = rig.POSES['heart']
        focus = (P['mid'][0] + 6, P['mid'][1] + 4)
        return place(img, 'heart', t, focus, 3.7 + 0.35 * ease(u), (960 + 40 * (1 - u), 470), focus_r=360)
    if lt < 10.6:                              # F2 the hand resting on the book
        u = (lt - 6.5) / 4.1
        img = room(lt)
        return place(img, 'heart', t, (300, 512), 2.15 + 0.15 * u, (1000 - 40 * u, 560), focus_r=330)
    if lt < 11.9:                              # insert: blank pages
        return blank_pages(lt)
    # F3 the open palm, a blinking light above it; push into the light
    u = (lt - 11.9) / (dur - 11.9)
    push = ease(sstep(0.55, 1.0, u))
    s = 2.1 + 0.2 * u + 9.0 * push ** 2.2
    cx0, cy0 = 1060 - 60 * u, 600
    center = (lerp(cx0, END_POINT[0], push), lerp(cy0, END_POINT[1], push))
    img = room(lt)
    img = place(img, 'offer', t, PALM_LIGHT, s, center, focus_r=280 * (1 + 3 * push), dof=1 - 0.6 * push)
    # everything except the light falls away into darkness at the end
    dark = sstep(0.55, 0.88, u)
    img = img * (1 - dark) + VOID * dark
    palm_light(img, center[0], center[1], lt, scale=1.0 + 1.2 * push, strength=1.0, roundness=sstep(0.85, 1.0, u))
    return img
