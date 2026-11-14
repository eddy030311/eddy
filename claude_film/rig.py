"""Pose library + puppet deformation for the Claude character."""
import numpy as np, cv2
from PIL import Image

# landmarks in each cutout's own pixel space
POSES = {
    'think': dict(mid=(254, 165), io=62, mouth=(260, 205), mw=15,
                  eyes=[(260, 310, 140, 168), (202, 245, 163, 190)], head_r=150, neck=(262, 240)),
    'point': dict(mid=(280, 155), io=64, mouth=(290, 197), mw=15, mouth_open=True,
                  eyes=[(292, 334, 134, 157), (230, 268, 153, 177)], head_r=150, neck=(285, 232),
                  arm=dict(shoulder=(200, 326), hand=(17, 241))),
    'calm':  dict(mid=(253, 160), io=62, mouth=(260, 202), mw=15, closed=True, head_r=150, neck=(262, 238)),
    'offer': dict(mid=(467, 175), io=75, mouth=(474, 222), mw=18, closed=True, head_r=180, neck=(478, 275),
                  arm=dict(shoulder=(310, 440), hand=(85, 370))),
    'heart': dict(mid=(315, 178), io=74, mouth=(322, 222), mw=18, closed=True, head_r=180, neck=(325, 275)),
    'full':  dict(mid=(472, 147), io=55, mouth=(478, 186), mw=13,
                  eyes=[(482, 519, 129, 147), (430, 461, 150, 164)], head_r=130, neck=(478, 215)),
}

_cache = {}
_grids = {}
def grid(W, H):
    if (W, H) not in _grids:
        _grids[(W, H)] = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return _grids[(W, H)]

def load(name):
    if name not in _cache:
        rgba = np.asarray(Image.open(f'build/{name}.png')).astype(np.float32) / 255.0
        rgba[..., :3] *= rgba[..., 3:4]          # premultiplied alpha from here on
        _cache[name] = np.ascontiguousarray(rgba)
    return _cache[name]

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

_hair = {}
def hair_weight(name):
    """Smooth mask of 'loose hair' — orange hue regions, increasing below the head."""
    if name in _hair: return _hair[name]
    rgba = load(name); P = POSES[name]
    hsv = cv2.cvtColor((rgba[..., :3] * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    hairish = ((h < 18) | (h > 170)) & (s > 110) & (v > 120)
    m = hairish.astype(np.float32) * rgba[..., 3]
    m = cv2.GaussianBlur(m, (0, 0), 18)
    m = np.clip(m * 2.2, 0, 1)
    H, W = m.shape
    yy = np.arange(H, dtype=np.float32)[:, None]
    below = smoothstep(P['mid'][1], P['mid'][1] + 4 * P['io'], yy)
    _hair[name] = (m * below).astype(np.float32)
    return _hair[name]

def deform(name, t, p):
    """Return displaced RGBA of pose `name` at time t with params p (dict)."""
    rgba = load(name); P = POSES[name]
    if 'eyes' in P and p.get('blink', 0) > 0.02:
        ce, _ = closed_eyes(name); b = float(np.clip(p['blink'], 0, 1))
        rgba = rgba * (1 - b) + ce * b
    if p.get('mouth', 0) > 0.03 and not P.get('mouth_open'):
        rgba = draw_mouth(rgba, P, p['mouth'], (0.0, 0.0))
    H, W = rgba.shape[:2]
    gx, gy = grid(W, H)
    dx = np.zeros_like(gx); dy = np.zeros_like(gy)
    io = P['io']; mx, my = P['mid']
    # --- breathing: chest rises, everything above shifts a bit up
    br = p.get('breath', 0.0)  # -1..1
    w_chest = smoothstep(H, my + 1.5 * io, gy)  # 1 at top region .. 0 at bottom
    dy += -br * io * 0.035 * w_chest
    # --- head tilt / nod around neck pivot
    ang = p.get('head_rot', 0.0); nod = p.get('head_nod', 0.0); hx_ = p.get('head_dx', 0.0)
    if ang or nod or hx_:
        nx, ny = P['neck']; R = P['head_r']
        r = np.hypot(gx - mx, gy - (my - 0.3 * io))
        w_head = (1 - smoothstep(R * 0.8, R * 1.55, r)) * (1 - smoothstep(ny - 0.2 * io, ny + 1.0 * io, gy))
        c, s_ = np.cos(ang), np.sin(ang)
        rx, ry = gx - nx, gy - ny
        dx += (rx - (c * rx + s_ * ry)) * w_head
        dy += (ry - (-s_ * rx + c * ry)) * w_head
        dy += nod * io * 0.12 * w_head
        dx += hx_ * io * 0.1 * w_head
    # --- hair sway (travelling wave)
    hw = hair_weight(name)
    amp = p.get('hair', 1.0) * io * 0.09
    if amp:
        depth = np.clip((gy - my) / (6 * io), 0, 2)
        dx += amp * hw * depth ** 1.3 * np.sin(t * 1.7 - depth * 2.2 + p.get('hair_phase', 0.0))
        dx += amp * 0.35 * hw * depth * np.sin(t * 3.1 + gy / 70.0)
        dy += amp * 0.15 * hw * depth * np.sin(t * 1.3 + gx / 90.0)
    # --- arm gesture: displacement weighted along the arm capsule, strongest at the hand
    arm = P.get('arm'); ag = p.get('arm', (0.0, 0.0))
    if arm is not None and (ag[0] or ag[1]):
        sx_, sy_ = arm['shoulder']; hx, hy = arm['hand']
        vx, vy = hx - sx_, hy - sy_; L2 = vx * vx + vy * vy
        u = np.clip(((gx - sx_) * vx + (gy - sy_) * vy) / L2, 0, 1.15)
        px, py = sx_ + u * vx, sy_ + u * vy
        d = np.hypot(gx - px, gy - py)
        w_arm = (1 - smoothstep(io * 0.9, io * 1.9, d)) * smoothstep(0.05, 0.9, u)
        dx += ag[0] * io * w_arm; dy += ag[1] * io * w_arm
    # --- global sway (whole figure leaning, pivot at bottom)
    sway = p.get('sway', 0.0)
    dx += sway * io * 0.25 * (1 - gy / H)
    mapx = gx - dx; mapy = gy - dy
    out = cv2.remap(rgba, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out

_closed = {}
def closed_eyes(name):
    """Open-eyed pose with the closed eyes of the 'calm' pose transplanted in."""
    if name in _closed: return _closed[name]
    P = POSES[name]; C = POSES['calm']; src = load(name); cal = load('calm')
    k = P['io'] / C['io']
    # affine: calm -> this pose, scale about mid point
    M = np.float32([[k, 0, P['mid'][0] - k * C['mid'][0]], [0, k, P['mid'][1] - k * C['mid'][1]]])
    H, W = src.shape[:2]
    warped = cv2.warpAffine(cal, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = np.zeros((H, W), np.float32)
    for (x0, x1, yt, yb) in P['eyes']:
        cx, cy = (x0 + x1) / 2, (yt + yb) / 2
        cv2.ellipse(mask, (int(cx), int(cy)), (int((x1 - x0) / 2 + 9), int((yb - yt) / 2 + 9)), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 4.5)[..., None]
    out = src * (1 - mask) + warped * mask
    out[..., 3] = src[..., 3]
    _closed[name] = (out.astype(np.float32), mask)
    return _closed[name]

def draw_mouth(img, P, o, off):
    """Paint a small open mouth (o in 0..1) using 4x supersampled shapes."""
    out = img.copy()
    mx, my = P['mouth']; mx += off[0]; my += off[1]
    w = P['mw'] * (0.75 + 0.25 * o); hgt = P['mw'] * 0.55 * o + 1.2
    S = 4; pad = 12
    x0, y0 = int(mx - w / 2 - pad), int(my - 3 - pad)
    pw, ph = int(w + 2 * pad), int(hgt + 3 + 2 * pad)
    region = out[y0:y0 + ph, x0:x0 + pw]
    canvas = np.zeros((ph * S, pw * S), np.uint8)
    inner = np.zeros_like(canvas)
    cx, cy = (mx - x0) * S, (my - y0 + hgt * 0.35) * S
    # skin cover for the original line
    skin = img[int(my + 6), int(mx)][:3] if 0 <= int(my + 6) < img.shape[0] else np.array([0.97, 0.88, 0.86])
    cover = np.zeros_like(canvas)
    cv2.ellipse(cover, (int(cx), int((my - y0) * S)), (int(w * 0.62 * S), int(3.2 * S)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (int(cx), int(cy)), (int(w / 2 * S), int(hgt / 2 * S)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(inner, (int(cx), int(cy + hgt * 0.25 * S)), (int(w * 0.32 * S), int(hgt * 0.28 * S)), 0, 0, 360, 255, -1, cv2.LINE_AA)
    def ds(m): return cv2.resize(m, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)[..., None] / 255
    cov, mm, im = ds(cover), ds(canvas), ds(inner)
    reg = region.copy()
    reg[..., :3] = reg[..., :3] * (1 - cov) + skin * cov
    dark = np.array([0.42, 0.16, 0.16]); tongue = np.array([0.86, 0.45, 0.45])
    reg[..., :3] = reg[..., :3] * (1 - mm) + dark * mm
    reg[..., :3] = reg[..., :3] * (1 - im * mm) + tongue * im * mm
    out[y0:y0 + ph, x0:x0 + pw] = reg
    return out
