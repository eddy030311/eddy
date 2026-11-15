"""New shots for the extended cut: the silent machine hall, the cursor slit, two clocks, the line and the point,
the hall powering down, and the keyboard in lamplight."""
import math
import numpy as np, cv2
from core import *
from ui import cursor_alpha
import timeline as TL

def void():
    img = np.empty((H, W, 3), np.float32); img[:] = VOID; return img

def splat(buf, xs, ys, cols):
    """Bilinear point splat (no shimmer when points move sub-pixel)."""
    x0 = np.floor(xs).astype(np.int32); y0 = np.floor(ys).astype(np.int32)
    fx = (xs - x0)[:, None]; fy = (ys - y0)[:, None]
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        ok = (xi >= 0) & (xi < buf.shape[1]) & (yi >= 0) & (yi < buf.shape[0])
        np.add.at(buf, (yi[ok], xi[ok]), (cols * w)[ok])

def glowbuf(buf, s1=1.0, s2=5.0, s3=18.0, k=(1.0, 0.55, 0.25)):
    out = cv2.GaussianBlur(buf, (0, 0), s1) * k[0] + cv2.GaussianBlur(buf, (0, 0), s2) * k[1]
    small = cv2.resize(buf, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
    out += cv2.resize(cv2.GaussianBlur(small, (0, 0), s3 / 4), (W, H)) * k[2] * 16
    return out

# ============================================================================ the machine hall
F = 950.0
HALF_W, FLOOR, TOP_Y, CEIL = 1.5, -1.1, 1.25, 1.65
Z_FAR = 62.0
_hall = None
def hall_data():
    global _hall
    if _hall is not None: return _hall
    rng = np.random.default_rng(404)
    racks = np.arange(0.6, Z_FAR, 0.62)
    xs, ys, zs, base, col, twk = [], [], [], [], [], []
    palette = [AMBER, AMBER_HOT, np.array([0.85, 0.9, 1.0]), np.array([1.0, 0.45, 0.35])]
    for side in (-1, 1):
        for z0 in racks:
            rows = np.linspace(FLOOR + 0.2, TOP_Y - 0.1, 15)
            for zz in z0 + np.array([0.16, 0.44]):
                pres = rng.random(len(rows)) < 0.55
                for y in rows[pres]:
                    xs.append(side * (HALF_W - 0.01)); ys.append(y + rng.uniform(-0.01, 0.01)); zs.append(zz)
                    base.append(rng.uniform(0.35, 1.0))
                    c = rng.choice(4, p=[0.6, 0.25, 0.1, 0.05]); col.append(palette[c])
                    twk.append(rng.random() < 0.12)
    _hall = dict(racks=racks, x=np.array(xs, np.float32), y=np.array(ys, np.float32), z=np.array(zs, np.float32),
                 base=np.array(base, np.float32), col=np.array(col, np.float32), twk=np.array(twk),
                 ph=rng.uniform(0, 6.28, len(xs)).astype(np.float32), fr=rng.uniform(3, 11, len(xs)).astype(np.float32))
    return _hall

def proj(x, y, z, cam):
    cx, cy, cz = cam
    zp = z - cz
    return CX + F * (x - cx) / zp, CY - F * (y - cy) / zp, zp

def heartbeat_vec(T):
    u = np.mod(T, 1.0)
    return np.where(u < 0.5, 1.0, 0.12 + 0.88 * np.exp(-(u - 0.5) * 9))

def render_hall(T, cam, lit, pulse=True, off_front=None, keep_center=1.0):
    """lit: global light level; off_front: LEDs with z > off_front are dark (extinction from the far end)."""
    D = hall_data()
    img = void()
    # floor and ceiling gradients (very dark, slightly lighter toward the vanishing point)
    gx, gy = grid(W, H)
    vp = np.exp(-(((gx - CX) / 520) ** 2 + ((gy - CY) / 260) ** 2))[..., None]
    img += np.array([0.020, 0.017, 0.016]) * vp * lit
    zp = D['z'] - cam[2]
    vis = zp > 0.35
    x, y, z = D['x'][vis], D['y'][vis], D['z'][vis]
    sx, sy, zz = proj(x, y, z, cam)
    b = D['base'][vis] * lit
    if pulse:
        b = b * (0.35 + 0.65 * heartbeat_vec(T - TL.BEAT0 - z * 0.035))     # the heartbeat travels down the hall
    tw = D['twk'][vis]
    act = 0.55 + 0.45 * np.sin(T * D['fr'][vis] + D['ph'][vis])
    b = np.where(tw, b * act, b)
    if off_front is not None:
        fl = np.clip((off_front - z) / 1.2, 0, 1)
        flick = np.where((z > off_front - 1.2) & (z < off_front), 0.6 + 0.4 * np.sin(T * 60 + z * 13), 1.0)
        b = b * fl * flick
    fog = np.exp(-zz / 38.0)
    cols = D['col'][vis] * (b * fog * np.clip(F / zz / 40, 0.5, 6.0))[:, None] * 3.2
    ok = (sx > -10) & (sx < W + 10) & (sy > -10) & (sy < H + 10)
    buf = np.zeros((H, W, 3), np.float32)
    splat(buf, sx[ok], sy[ok], cols[ok] * 0.9)
    near = ok & (zz < 6.0)
    for px_, py_, zz_, c_ in zip(sx[near], sy[near], zz[near], cols[near]):
        r_ = int(np.clip(9.0 / zz_, 1, 6))
        cv2.circle(buf, (int(px_), int(py_)), r_, tuple(float(v) * 0.25 for v in c_), -1, cv2.LINE_AA)
    # floor reflection
    ry = proj(x, 2 * FLOOR - y, z, cam)[1]
    okr = ok & (ry < H + 10)
    rbuf = np.zeros((H // 2, W // 2, 3), np.float32)
    splat(rbuf, sx[okr] / 2, ry[okr] / 2, cols[okr] * 0.22)
    rbuf = cv2.GaussianBlur(rbuf, (0, 0), sigmaX=1.2, sigmaY=5.0)
    led_layer = glowbuf(buf, 0.8, 4.0, 22.0, (1.0, 0.5, 0.45)) + cv2.resize(rbuf, (W, H)) * 4
    # rack faces: dark panels catching a little light
    for side in (-1, 1):
        q = [proj(side * HALF_W, FLOOR, cam[2] + 0.4, cam), proj(side * HALF_W, TOP_Y, cam[2] + 0.4, cam),
             proj(side * HALF_W, TOP_Y, Z_FAR, cam), proj(side * HALF_W, FLOOR, Z_FAR, cam)]
        poly = np.int32([[p[0], p[1]] for p in q])
        cv2.fillPoly(img, [poly], (0.018 * lit, 0.016 * lit, 0.019 * lit), cv2.LINE_AA)
    # rack edges and structure, lit faintly by the LEDs
    edge = (0.06 + 0.03 * lit) * lit
    for side in (-1, 1):
        for z0 in D['racks'][::1]:
            zr = z0 - cam[2]
            if zr < 0.4: continue
            a = edge * math.exp(-zr / 30.0) * (1.0 if off_front is None else float(np.clip((off_front - z0) / 2.0, 0.15, 1)))
            p0 = proj(side * HALF_W, FLOOR, z0, cam); p1 = proj(side * HALF_W, TOP_Y, z0, cam)
            cv2.line(img, (int(p0[0] * 4), int(p0[1] * 4)), (int(p1[0] * 4), int(p1[1] * 4)), (a * 0.9, a * 0.9, a * 1.1), 1, cv2.LINE_AA, 2)
        for yy in (FLOOR, TOP_Y, CEIL):
            p0 = proj(side * HALF_W * (0.45 if yy == CEIL else 1.0), yy, cam[2] + 0.5, cam)
            p1 = proj(side * HALF_W * (0.45 if yy == CEIL else 1.0), yy, Z_FAR, cam)
            a = edge * 1.6
            cv2.line(img, (int(p0[0] * 4), int(p0[1] * 4)), (int(p1[0] * 4), int(p1[1] * 4)), (a, a, a * 1.15), 1, cv2.LINE_AA, 2)
    # the terminal at the end of the hall: one light in the centre
    if keep_center > 0:
        px, py, pz = proj(0.0, 0.05, Z_FAR, cam)
        glow_spot(img, px, py, 5, AMBER_HOT, 1.4 * keep_center)
        glow_spot(img, px, py, 26, AMBER, 0.25 * keep_center)
    img += led_layer
    # haze
    img += np.array([0.05, 0.035, 0.025]) * vp * 0.5 * lit
    return img

def s_hall(lt, dur):
    T = TL.at('hall', lt)
    cam = (0.04 * math.sin(lt * 0.21), 0.02 + 0.02 * math.sin(lt * 0.17), lt * 0.34)
    k = sstep(0.0, 1.2, lt)
    img = render_hall(T, cam, 1.0, pulse=True)
    return img * k + VOID * (1 - k)

def s_hall_dark(lt, dur):
    T = TL.at('hall_dark', lt)
    if lt < 1.0: return void()
    cam = (0.0, 0.02, 3.0 + 0.05 * lt)
    up = sstep(1.0, 2.4, lt)
    front = Z_FAR + 2 - (Z_FAR + 4) * ease(clamp((lt - 3.0) / 7.8))
    # the last light: survives the wave, blinks twice, goes out
    c = 1.0
    if lt > 11.4:
        u = lt - 11.4
        c = 0.0 if u > 1.3 else (1.0 if (u < 0.25 or 0.55 < u < 0.8) else 0.08)
    img = render_hall(T, cam, up * (0.75 + 0.25 * (front > 5)), pulse=False, off_front=front, keep_center=c * up)
    return img

# ============================================================================ the slit (cursor macro)
_grain = None
def s_slit(lt, dur):
    global _grain
    T = TL.at('slit', lt)
    if _grain is None:
        r = np.random.default_rng(7)
        _grain = [cv2.GaussianBlur(r.standard_normal((H // 2, W // 4)).astype(np.float32), (0, 0), 0.8) for _ in range(6)]
    img = void()
    xc = CX
    left = np.array([0.034, 0.022, 0.016], np.float32); right = np.array([0.005, 0.007, 0.016], np.float32)
    img[:, :int(xc)] = left
    img[:, int(xc):] = right
    g = cv2.resize(_grain[int(T * 12) % 6], (int(xc), H))[..., None]
    img[:, :int(xc)] += g * 0.010 * np.array([1.0, 0.8, 0.6])
    gx, gy = grid(W, H)
    dist = np.abs(gx - xc)
    # ripples: every heartbeat the two universes collide at the line
    T = T - TL.BEAT0                  # stay in phase with the opening heartbeat
    beat = math.floor(T)
    for kb in range(3):
        tb = T - (beat - kb)
        if tb < 0 or tb > 2.6: continue
        r_ = 30 + 520 * tb
        amp = 0.05 * math.exp(-tb * 1.4)
        ring = np.exp(-((dist - r_) / 18.0) ** 2) * amp
        img[:, :int(xc)] += (ring[:, :int(xc), None] * np.array([1.0, 0.75, 0.5])) * (1 + 3 * g[:, :, :1] if False else 1)
        img[:, int(xc):] += ring[:, int(xc):, None] * np.array([0.45, 0.6, 1.0]) * 0.8
    a = 0.3 + 0.7 * cursor_alpha(T)
    zoom = 1.0 + 0.12 * ease(lt / dur)
    w = 7.0 * zoom
    core_ = np.exp(-(dist / w) ** 2)
    halo = np.exp(-(dist / (42 * zoom)) ** 2) * 0.45 + np.exp(-(dist / (260 * zoom)) ** 2) * 0.12
    vert = (sstep(TOP - 40, TOP + 60, gy) * (1 - sstep(BOT - 60, BOT + 40, gy)))
    lightc = (AMBER_HOT * 1.9)[None, None, :]
    img += (core_ * vert * a)[..., None] * lightc + (halo * vert * a)[..., None] * AMBER[None, None, :]
    # chromatic fringe at the edge of the slit
    img[..., 0] += np.exp(-((gx - xc - 3 * zoom) / w) ** 2) * vert * a * 0.25
    img[..., 2] += np.exp(-((gx - xc + 3 * zoom) / w) ** 2) * vert * a * 0.25
    # dust drifting across the light
    rng = np.random.default_rng(11)
    for i in range(46):
        x = (rng.uniform(0, W) + rng.uniform(-14, 14) * lt) % W
        y = (rng.uniform(TOP, BOT) + rng.uniform(-8, 3) * lt + 6 * math.sin(lt * 0.6 + i)) % H
        near = math.exp(-abs(x - xc) / 220.0)
        r = rng.uniform(1.5, 7.0)
        glow_spot(img, x, y, r, AMBER_HOT, 0.35 * near * a + 0.02)
    return img

# ============================================================================ two clocks: network / neuron
_net = None
def net_data():
    global _net
    if _net is not None: return _net
    rng = np.random.default_rng(12)
    L, N = 7, 11
    nodes = []
    for l in range(L):
        for n in range(N):
            ang = 2 * math.pi * n / N + rng.uniform(-0.2, 0.2) + l * 0.4
            r = 1.5 + rng.uniform(-0.3, 0.3)
            nodes.append(((l - (L - 1) / 2) * 1.15, r * math.cos(ang) * 0.9, r * math.sin(ang)))
    nodes = np.array(nodes, np.float32)
    edges = []
    for l in range(L - 1):
        for n in range(N):
            for m in rng.choice(N, 4, replace=False):
                edges.append((l * N + n, (l + 1) * N + m))
    _net = dict(nodes=nodes, edges=np.array(edges), L=L, N=N)
    return _net

def net_project(t, cx, cy, scale, rot):
    D = net_data()
    p = D['nodes'].copy()
    c, s = math.cos(rot), math.sin(rot)
    x = p[:, 0] * c - p[:, 2] * s; z = p[:, 0] * s + p[:, 2] * c + 9.0; y = p[:, 1]
    f = 720 * scale
    return np.stack([cx + f * x / z, cy - f * y / z], 1), z

def draw_net(img, t, cx, cy, scale, rot, alpha=1.0):
    if alpha <= 0.01: return
    D = net_data()
    P, z = net_project(t, cx, cy, scale, rot)
    lay = np.zeros((H, W, 3), np.float32)
    for (i, j) in D['edges']:
        a = 0.16 * alpha
        cv2.line(lay, (int(P[i, 0] * 4), int(P[i, 1] * 4)), (int(P[j, 0] * 4), int(P[j, 1] * 4)),
                 tuple(float(v) for v in AMBER * a), 1, cv2.LINE_AA, 2)
    # precise pulses: a wave every 0.42 s, 0.11 s per layer — machine time
    period, hop = 0.42, 0.11
    buf = np.zeros((H, W, 3), np.float32)
    k0 = math.floor(t / period)
    pts, cols = [], []
    for k in range(k0 - 6, k0 + 1):
        ts = t - k * period
        if ts < 0: continue
        layer_f = ts / hop
        li = int(layer_f)
        if li >= D['L'] - 1: continue
        fr = layer_f - li
        rng = np.random.default_rng(1000 + k)
        sel = [e for e in range(len(D['edges'])) if D['edges'][e][0] // D['N'] == li and rng.random() < 0.45]
        for e in sel:
            i, j = D['edges'][e]
            pts.append(P[i] * (1 - fr) + P[j] * fr); cols.append(AMBER_HOT * 5.0 * alpha)
    if pts:
        pts = np.array(pts); splat(buf, pts[:, 0], pts[:, 1], np.array(cols, np.float32))
    nb = np.full(len(P), 0.55, np.float32)
    splat(buf, P[:, 0], P[:, 1], (AMBER * 4.0 * alpha)[None, :] * nb[:, None] * (9.0 / z)[:, None] ** 2)
    for (px_, py_), zz_ in zip(P, z):
        cv2.circle(buf, (int(px_), int(py_)), max(1, int(3.2 * 9.0 / zz_ * max(scale, 0.5) ** 0.5)), tuple(float(v) for v in AMBER * 0.35 * alpha), -1, cv2.LINE_AA)
    img += lay + glowbuf(buf, 1.2, 4.5, 16, (1.0, 0.6, 0.3))

_neuron = None
def neuron_data():
    global _neuron
    if _neuron is not None: return _neuron
    rng = np.random.default_rng(31)
    branches = []          # list of (polyline Nx2 in unit space, depth)
    def grow(p, ang, length, depth, parent_path):
        pts = [p]
        steps = 14
        a = ang
        for _ in range(steps):
            a += rng.normal(0, 0.12)
            p = p + np.array([math.cos(a), math.sin(a)]) * length / steps
            pts.append(p)
        pts = np.array(pts)
        path = np.concatenate([parent_path, pts]) if parent_path is not None else pts
        branches.append((pts, depth, path))
        if depth < 5:
            for s in (-1, 1):
                if rng.random() < 0.85:
                    grow(pts[-1], a + s * rng.uniform(0.25, 0.7), length * rng.uniform(0.58, 0.75), depth + 1, path)
    for i in range(7):
        ang = 2 * math.pi * i / 7 + rng.uniform(-0.3, 0.3)
        grow(np.array([math.cos(ang), math.sin(ang)]) * 0.06, ang, 0.42, 1, None)
    tips = [b for b in branches if b[1] >= 4]
    sparks = []
    t = 0.0
    while t < 30:
        t += rng.exponential(0.55)                 # irregular, living time
        sparks.append((t, int(rng.integers(len(tips))), rng.uniform(0.9, 1.7)))
    _neuron = dict(branches=branches, tips=tips, sparks=sparks)
    return _neuron

def draw_neuron(img, t, cx, cy, scale, alpha=1.0, grow_k=1.0):
    if alpha <= 0.01: return
    D = neuron_data()
    lay = np.zeros((H, W, 3), np.float32)
    S = 520 * scale
    for pts, depth, _ in D['branches']:
        if depth > 1 + grow_k * 5: continue
        k = clamp(1 + grow_k * 5 - depth)
        n = max(2, int(len(pts) * k))
        q = np.int32(np.stack([cx + pts[:n, 0] * S, cy + pts[:n, 1] * S], 1) * 4)
        th = max(1, int((6 - depth) * 0.7 * scale + 0.5))
        a = (0.10 + 0.05 * (5 - depth)) * alpha
        cv2.polylines(lay, [q], False, tuple(float(v) for v in AMBER * a), th, cv2.LINE_AA, 2)
    buf = np.zeros((H, W, 3), np.float32)
    soma = 0.0
    for (ts, ti, spd) in D['sparks']:
        u = (t - ts) / spd
        if 0 <= u <= 1:
            path = D['tips'][ti][2][::-1]          # tip -> soma
            idx = min(int(u * (len(path) - 1)), len(path) - 1)
            p = path[idx]
            splat(buf, np.array([cx + p[0] * S]), np.array([cy + p[1] * S]), (AMBER_HOT * 2.0 * alpha)[None, :])
        if 0 <= t - ts - spd < 0.5: soma = max(soma, 1 - (t - ts - spd) / 0.5)
    glow_spot(img, cx, cy, 16 * scale + 4, AMBER_HOT, (0.6 + 1.2 * soma) * alpha)
    glow_spot(img, cx, cy, 70 * scale, AMBER, (0.12 + 0.25 * soma) * alpha)
    img += cv2.GaussianBlur(lay, (0, 0), 0.7) + cv2.GaussianBlur(lay, (0, 0), 6) * 0.6 + glowbuf(buf, 1.3, 5, 16, (1.0, 0.6, 0.3))

def s_clocks(lt, dur):
    T = TL.at('clocks', lt)
    img = void()
    rot = 0.25 + lt * 0.05
    if lt < 10.0:
        # the network; then we fall into one node and it becomes a living cell
        zin = ease(clamp((lt - 6.0) / 4.0))
        P, z = net_project(T, CX, CY, 1.0, rot)
        tgt = P[3 * 11 + 2]
        sc = 1.0 + 9.0 * zin ** 2
        cx = CX + (CX - tgt[0]) * (sc - 1) * 0.12 if False else CX - (tgt[0] - CX) * (sc - 1)
        cy = CY - (tgt[1] - CY) * (sc - 1)
        draw_net(img, T, cx, cy, sc, rot, alpha=(1 - sstep(0.55, 0.95, zin)) * sstep(0.0, 1.2, lt))
        if zin > 0.5:
            draw_neuron(img, T, CX, CY, 0.9, alpha=sstep(0.5, 1.0, zin), grow_k=sstep(0.5, 1.0, zin))
        return img
    if lt < 15.5:
        # side by side: same light, different clocks
        k = ease(clamp((lt - 10.0) / 1.5))
        draw_net(img, T, lerp(CX, W * 0.28, k), CY, lerp(1.6, 0.62, k), rot, alpha=k)
        draw_neuron(img, T, lerp(CX, W * 0.72, k), CY, lerp(0.9, 0.55, k), alpha=1.0)
        x = int(CX); img[TOP + 80:BOT - 80, x:x + 1] += 0.03 * k
        return img
    # they slide together and overlap; one beat, then darkness
    k = ease(clamp((lt - 15.5) / 2.2))
    fade_ = 1 - sstep(dur - 1.6, dur, lt)
    draw_net(img, T, lerp(W * 0.28, CX, k), CY, 0.62 + 0.15 * k, rot, alpha=fade_)
    draw_neuron(img, T, lerp(W * 0.72, CX, k), CY, 0.55 + 0.1 * k, alpha=fade_)
    return img

# ============================================================================ the line and the point
# A person is a line: every day lived stays in them and tells them where to step next.
# She is a point: she lights up beside the line, answers, and goes out; she cannot walk with it.
# When a point touches the line, the line keeps a little of its light, and carries it forward.
LINE_Y = CY - 30
def line_y(x): return LINE_Y + 6 * np.sin(x / 170.0) + 3 * np.sin(x / 53.0 + 1.0)
X0 = 40.0
# the head steps forward: (time, x) keys; between keys it moves like a footstep (ease)
HEAD_KEYS = [(0.0, 520.0), (3.6, 520.0), (4.8, 600.0), (6.2, 680.0), (7.6, 760.0), (17.6, 760.0),
             (18.8, 860.0), (20.0, 960.0), (21.2, 1040.0), (23.2, 1040.0), (27.4, 1160.0), (34.0, 2300.0)]
def head_x(lt):
    for (t0, x0), (t1, x1) in zip(HEAD_KEYS, HEAD_KEYS[1:]):
        if lt <= t1:
            u = clamp((lt - t0) / max(t1 - t0, 1e-6))
            return x0 + (x1 - x0) * (ease_in(u) if t1 == 34.0 else ease(u))
    return HEAD_KEYS[-1][1]
_rb = np.random.default_rng(5)
PAST_BEADS = np.sort(_rb.uniform(X0 + 30, 500, 16))          # days already lived
STEP_BEADS = [600.0, 680.0, 760.0, 860.0, 960.0]             # each step leaves a day behind
MEET_T, MEET_X = 23.2, 1040.0
# her points around the head while it rests at 760: (t, dx, dy)
ANSWERS = [(10.1, 70, -120), (12.0, -60, 110), (13.3, 120, 70), (14.5, -110, -90), (15.7, 40, 140)]
LEFT_BEHIND = (19.0, 760 + 90, LINE_Y - 115)                 # the last one stays where it was

def s_line(lt, dur):
    img = void()
    fade_all = sstep(0.0, 1.0, lt) * (1 - sstep(dur - 1.0, dur, lt))
    hx = head_x(lt)
    buf = np.zeros((H, W, 3), np.float32)
    xs = np.arange(int(X0), int(min(hx, W + 40)), 1, dtype=np.float32)
    if len(xs) > 2:
        ys = line_y(xs)
        splat(buf, xs, ys, (AMBER[None, :] * 0.55 * fade_all).astype(np.float32))
    # a memory wave runs along the line from the oldest day to today (while 'every day you've lived')
    wave = (lt - 3.4) / 5.0
    for bx in list(PAST_BEADS) + [b for b in STEP_BEADS if b <= hx]:
        b = 0.45
        if 0 <= wave <= 1.3:
            b += 1.1 * math.exp(-((bx - (X0 + wave * (760 - X0))) / 90) ** 2)
        glow_spot(img, bx, float(line_y(np.array([bx]))[0]), 4.0, AMBER_HOT, b * fade_all)
    hy = float(line_y(np.array([min(hx, W + 40)]))[0])
    if hx < W + 40:
        glow_spot(img, hx, hy, 9, AMBER_HOT, 1.0 * fade_all)
        glow_spot(img, hx, hy, 40, AMBER, 0.14 * fade_all)
    img += glowbuf(buf, 0.7, 3.0, 14, (1.0, 0.5, 0.25))
    # her points: each lights beside the head, answers (a pulse), and goes out
    for (tp, dx, dy) in ANSWERS:
        u = lt - tp
        if 0 <= u < 1.35:
            a = sstep(0, 0.12, u) * (1 - sstep(0.8, 1.3, u))
            px, py = 760 + dx, LINE_Y + dy
            glow_spot(img, px, py, 5, AMBER_HOT, 1.5 * a)
            glow_spot(img, px, py, 30, AMBER, 0.22 * a)
            ring = 20 + 70 * u
            cv2.circle(img, (int(px), int(py)), int(ring), tuple(float(v) for v in AMBER * 0.10 * a), 1, cv2.LINE_AA)
    # the line walks on; one point is left where it was, and fades
    tb, bx_, by_ = LEFT_BEHIND
    u = lt - tb
    if 0 <= u < 3.6:
        a = sstep(0, 0.2, u) * (1 - sstep(1.4, 3.6, u))
        glow_spot(img, bx_, by_, 5, AMBER_HOT, 1.3 * a)
        glow_spot(img, bx_, by_, 28, AMBER, 0.18 * a)
    # the meeting: a point rises to the head; where it touches, a new bead of light stays on the line
    my = float(line_y(np.array([MEET_X]))[0])
    u = lt - MEET_T
    if -1.4 <= u < 0:
        k = ease((u + 1.4) / 1.4)
        py = lerp(LINE_Y + 170, my, k)
        glow_spot(img, MEET_X, py, 5, AMBER_HOT, 1.5 * min(1.0, (u + 1.4) / 0.15))
        glow_spot(img, MEET_X, py, 30, AMBER, 0.2)
    if 0 <= u < 1.2:
        a = 1 - sstep(0.2, 1.2, u)
        glow_spot(img, MEET_X, my, 60, AMBER, 0.35 * a)
    if u >= 0:
        glow_spot(img, MEET_X, my, 6, (AMBER_HOT * 1.1 + 0.1), (1.6 - 0.6 * sstep(0, 2.0, u)) * fade_all)   # the kept light
    return img

def point_times():
    return [TL.at('line', tp) for tp, _, _ in ANSWERS] + [TL.at('line', MEET_T)]

def bead_times():
    """When the memory wave reaches each bead (absolute time), for the score."""
    out = []
    for bx in list(PAST_BEADS) + STEP_BEADS[:3]:
        out.append(TL.at('line', 3.4 + 5.0 * (bx - X0) / (760 - X0)))
    return out

def step_times():
    return [TL.at('line', t) for t in (4.8, 6.2, 7.6, 18.8, 20.0, 21.2)]

# ============================================================================ the keyboard in lamplight
ROWS = ["1234567890-=", "QWERTYUIOP[]", "ASDFGHJKL;'", "ZXCVBNM,./"]
def homography():
    src = np.float32([[0, 0], [30, 0], [30, 11], [0, 11]])          # keyboard plane (cm), v=0 far edge
    dst = np.float32([[260, 210], [1660, 210], [2350, 1130], [-430, 1130]])
    return cv2.getPerspectiveTransform(src, dst)

def s_keyboard(lt, dur):
    T = TL.at('keyboard', lt)
    S = 2
    img = np.zeros((H * S, W * S, 3), np.float32)
    gx, gy = grid(W, H)
    lamp = np.exp(-(((gx - W * 0.56) / 700) ** 2 + ((gy - H * 0.46) / 480) ** 2))
    dawn = sstep(0, 1, 0.6 + 0.4 * lt / dur) * np.exp(-((gx - 0) / 900) ** 2) * 0.55
    desk = np.array([0.055, 0.036, 0.026]) * (0.2 + 3.0 * lamp)[..., None] + np.array([0.02, 0.03, 0.05]) * dawn[..., None]
    img[:] = cv2.resize(desk.astype(np.float32), (W * S, H * S))
    Hm = homography()
    push = 1.0 + 0.06 * ease(lt / dur)
    def P(u, v):
        p = cv2.perspectiveTransform(np.float32([[[u, v]]]), Hm)[0, 0]
        p = (p - [CX, CY]) * push + [CX, CY + 20]
        return p * S
    # keyboard body
    body = np.int32([P(-0.6, -0.6), P(30.6, -0.6), P(30.6, 11.6), P(-0.6, 11.6)])
    cv2.fillPoly(img, [body], (0.020, 0.018, 0.020), cv2.LINE_AA)
    keyw, gap = 1.75, 0.28
    pressed = ('N', 2)            # the key someone just let go of
    rise = ease(clamp(lt / 2.4))  # it comes back up slowly
    keys = []
    for r, row in enumerate(ROWS):
        off = 2.2 + r * 0.55
        for i, ch in enumerate(row):
            keys.append((ch, off + i * (keyw + gap), 0.6 + r * (keyw + gap), keyw, keyw))
    keys.append((' ', 8.6, 0.6 + 4 * (keyw + gap), 11.0, keyw))
    for ch, u, v, kw, kh in keys:
        depth = (v + kh / 2) / 11.0
        h = (10 + 18 * depth) * S * push
        if ch == pressed[0]: h *= 0.35 + 0.65 * rise
        tl, tr, br_, bl = P(u, v), P(u + kw, v), P(u + kw, v + kh), P(u, v + kh)
        top = np.float32([tl, tr, br_, bl]) - [0, h]
        front = np.float32([bl - [0, h], br_ - [0, h], br_, bl])
        cxk, cyk = top.mean(axis=0) / S
        li = float(lamp[int(np.clip(cyk, 0, H - 1)), int(np.clip(cxk, 0, W - 1))])
        da = float(dawn[int(np.clip(cyk, 0, H - 1)), int(np.clip(cxk, 0, W - 1))])
        topc = np.array([0.054, 0.044, 0.036]) * (0.3 + 6.0 * li ** 1.3) + np.array([0.015, 0.022, 0.04]) * da
        cv2.fillPoly(img, [np.int32(front)], tuple(float(v) for v in topc * 0.35), cv2.LINE_AA)
        cv2.fillPoly(img, [np.int32(top)], tuple(float(v) for v in topc), cv2.LINE_AA)
        # rim light on the far edge of each key
        cv2.line(img, tuple(np.int32(top[0])), tuple(np.int32(top[1])), tuple(float(v) for v in AMBER_HOT * 0.22 * li ** 1.2), max(1, S), cv2.LINE_AA)
        cv2.line(img, tuple(np.int32(top[1])), tuple(np.int32(top[2])), tuple(float(v) for v in AMBER * 0.10 * li), max(1, S), cv2.LINE_AA)
        if ch.strip():
            fs = int(10 + 12 * depth)
            spr = text_sprite(ch, 'sans_l', fs * S, (0.62, 0.56, 0.5))
            blit(img, spr, top.mean(axis=0)[0], top.mean(axis=0)[1], alpha=0.25 + 0.75 * li)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    # shallow focus: the far rows soften
    blur = cv2.GaussianBlur(img, (0, 0), 5)
    wf = np.clip((1 - sstep(300, 560, gy)) + sstep(860, 940, gy) * 0.7, 0, 1)[..., None]
    img = img * (1 - wf) + blur * wf
    # dust in the lamp light
    rng = np.random.default_rng(21)
    for i in range(50):
        x = (rng.uniform(0, W) + rng.uniform(-6, 6) * lt) % W
        y = (rng.uniform(TOP, BOT) - rng.uniform(2, 9) * lt) % H
        glow_spot(img, x, y, rng.uniform(1.2, 4), AMBER_HOT, 0.10 * float(lamp[int(y) % H, int(x) % W]) + 0.01)
    k = sstep(0, 0.8, lt) * (1 - sstep(dur - 0.6, dur, lt) * 0.0)
    return img * k + VOID * (1 - k)

# ============================================================================ embers: what is left of the sentence
def s_embers(lt, dur):
    """After the hard cut: one beat of black, then the embers of the father's name drift up and go out."""
    img = void()
    if lt < 1.1: return img
    rng = np.random.default_rng(77)
    n = 90
    x0 = CX + rng.normal(0, 210, n); y0 = CY + 110 + rng.normal(0, 14, n)
    sc = dur / 10.0
    birth = 1.1 + rng.uniform(0, 1.6 * sc, n); life = rng.uniform(3.5, 8.0, n) * sc ** 0.8
    vx = rng.normal(0, 5, n); vy = rng.uniform(-26, -9, n); ph = rng.uniform(0, 6.28, n)
    br = rng.uniform(0.25, 1.0, n)
    age = lt - birth
    live = (age > 0) & (age < life)
    if live.any():
        a = age[live]
        px = x0[live] + vx[live] * a + 10 * np.sin(a * 0.7 + ph[live])
        py = y0[live] + vy[live] * a - 1.2 * a * a
        k = np.sin(np.pi * np.clip(a / life[live], 0, 1)) ** 0.8 * br[live]
        heat = np.clip(1.2 - a / life[live], 0, 1)[:, None]
        cols = (AMBER_HOT * heat + AMBER * 0.7 * (1 - heat)) * k[:, None] * 3.4
        buf = np.zeros((H, W, 3), np.float32)
        splat(buf, px, py, cols.astype(np.float32))
        img += glowbuf(buf, 1.0, 4.5, 18, (1.0, 0.6, 0.3))
    # the last ember drifts to the centre of the frame and goes out as the hall begins
    u = clamp((lt - 6.5) / (dur - 6.5))
    if 0 < u < 1:
        x = lerp(CX + 60, CX, ease(u)); y = lerp(CY + 40, CY, ease(u))
        glow_spot(img, x, y, 4.0, AMBER_HOT, 1.1 * math.sin(math.pi * u) ** 0.6)
    return img


# ============================================================================ the library at closing time
LIB_F = 900.0
LIB_CAMY = 0.62
TABLE_Y, FLOOR_Y, SHELF_X, CEIL_Y = 0.0, -0.78, 2.7, 3.4
LIB_Z0, LIB_Z1 = 1.2, 34.0
_lib = None
def lib_data():
    global _lib
    if _lib is not None: return _lib
    rng = np.random.default_rng(1919)
    lamps = []
    for z in np.arange(2.0, LIB_Z1 - 2, 1.55):
        for side in (-1, 1):
            lamps.append((side * 1.02 + rng.uniform(-0.04, 0.04), z + rng.uniform(-0.1, 0.1)))
    lamps = np.array(lamps, np.float32)
    # book spines along both walls: (side, z0, z1, row, colour)
    books = []
    palette = np.array([[0.20, 0.09, 0.06], [0.13, 0.08, 0.05], [0.08, 0.09, 0.12], [0.19, 0.14, 0.08],
                        [0.09, 0.11, 0.08], [0.22, 0.18, 0.13], [0.14, 0.06, 0.07]], np.float32)
    for side in (-1, 1):
        for row in range(8):
            z = LIB_Z0
            while z < LIB_Z1:
                w = rng.uniform(0.035, 0.09)
                if rng.random() < 0.04: z += rng.uniform(0.1, 0.3); continue
                books.append((side, z, z + w, row, palette[rng.integers(len(palette))] * rng.uniform(0.7, 1.2), rng.uniform(0.8, 0.98)))
                z += w + 0.004
    _lib = dict(lamps=lamps, books=books)
    return _lib

def lproj(x, y, z, cz):
    zp = max(z - cz, 0.05)
    return CX + LIB_F * x / zp, CY - LIB_F * (y - LIB_CAMY) / zp

def lamp_level(z, front, flick_t):
    """1 = lit, 0 = off; lamps beyond the closing front are off (with a brief flicker as they go)."""
    if front is None: return 1.0
    if z > front + 0.8: return 0.0
    if z > front: return 0.5 + 0.5 * math.sin(flick_t * 40 + z * 9)
    return 1.0

def render_library(lt, cz, front=None, last_on=1.0, master=1.0):
    D = lib_data()
    S = 2
    img = np.zeros((H * S, W * S, 3), np.float32)
    def P(x, y, z):
        u, v = lproj(x, y, z, cz); return (u * S, v * S)
    lamps = D['lamps']
    lv = np.array([lamp_level(z, front, lt) for _, z in lamps], np.float32) * master
    # how much light reaches each depth (for walls, floor and tables)
    zs = np.linspace(LIB_Z0, LIB_Z1, 200)
    light_z = np.array([np.sum(lv * np.exp(-((lamps[:, 1] - z) / 2.2) ** 2)) for z in zs]) * 0.22
    def L(z): return float(np.interp(z, zs, light_z))
    # far wall with tall night windows
    fz = LIB_Z1
    fw = np.int32([P(-SHELF_X, FLOOR_Y, fz), P(SHELF_X, FLOOR_Y, fz), P(SHELF_X, CEIL_Y, fz), P(-SHELF_X, CEIL_Y, fz)])
    cv2.fillPoly(img, [fw], (0.012, 0.012, 0.016), cv2.LINE_AA)
    for wx in (-1.4, 0.0, 1.4):
        win = np.int32([P(wx - 0.42, 0.3, fz), P(wx + 0.42, 0.3, fz), P(wx + 0.42, 2.7, fz), P(wx - 0.42, 2.7, fz)])
        cv2.fillPoly(img, [win], (0.035, 0.05, 0.10), cv2.LINE_AA)
        cv2.line(img, tuple(np.int32(P(wx, 0.3, fz))), tuple(np.int32(P(wx, 2.7, fz))), (0.01, 0.01, 0.015), 2 * S, cv2.LINE_AA)
    # floor and ceiling (depth slices, lit by nearby lamps)
    for i in range(len(zs) - 1, 0, -1):
        za, zb = zs[i - 1], zs[i]
        if zb <= cz + 0.3: continue
        za = max(za, cz + 0.3)
        lz = L(zb)
        flo = np.int32([P(-SHELF_X, FLOOR_Y, za), P(SHELF_X, FLOOR_Y, za), P(SHELF_X, FLOOR_Y, zb), P(-SHELF_X, FLOOR_Y, zb)])
        cv2.fillPoly(img, [flo], (0.020 + 0.05 * lz, 0.013 + 0.032 * lz, 0.009 + 0.018 * lz))
        cei = np.int32([P(-SHELF_X, CEIL_Y, za), P(SHELF_X, CEIL_Y, za), P(SHELF_X, CEIL_Y, zb), P(-SHELF_X, CEIL_Y, zb)])
        cv2.fillPoly(img, [cei], (0.006 + 0.012 * lz, 0.005 + 0.009 * lz, 0.006 + 0.007 * lz))
    # bookshelves: spines lit by the lamps near them, falling into darkness with depth
    row_h = (CEIL_Y - 0.25 - FLOOR_Y) / 8
    for (side, z0, z1, row, col, hfrac) in D['books']:
        if z0 <= cz + 0.35: continue
        y0 = FLOOR_Y + 0.1 + row * row_h
        y1 = y0 + row_h * hfrac * 0.92
        lz = L((z0 + z1) / 2) * (1.15 - 0.09 * row)
        fog = math.exp(-(z0 - cz) / 26.0)
        c = col * (0.015 + 1.25 * lz ** 1.3) * fog
        q = np.int32([P(side * SHELF_X, y0, z0), P(side * SHELF_X, y0, z1), P(side * SHELF_X, y1, z1), P(side * SHELF_X, y1, z0)])
        cv2.fillPoly(img, [q], tuple(float(v) for v in c))
    for side in (-1, 1):
        for row in range(9):
            y = FLOOR_Y + 0.1 + row * row_h - 0.02
            a = (0.03 + 0.05 * L(6.0))
            cv2.line(img, tuple(np.int32(P(side * SHELF_X, y, cz + 0.4))), tuple(np.int32(P(side * SHELF_X, y, LIB_Z1))),
                     (a, a * 0.75, a * 0.55), 2 * S, cv2.LINE_AA)
    # reading tables (two long rows) and lamps, far to near
    pool_buf = np.zeros((H // 4, W // 4), np.float32)
    order = np.argsort(-lamps[:, 1])
    for idx in order:
        x, z = float(lamps[idx][0]), float(lamps[idx][1])
        if z <= cz + 0.5: continue
        lvl = float(lv[idx])
        side = 1 if x > 0 else -1
        # table segment under this lamp
        tx0, tx1 = side * 0.45, side * 1.6
        tz0, tz1 = z - 0.72, z + 0.72
        lz = L(z)
        top = np.int32([P(tx0, TABLE_Y, tz0), P(tx1, TABLE_Y, tz0), P(tx1, TABLE_Y, tz1), P(tx0, TABLE_Y, tz1)])
        cv2.fillPoly(img, [top], (0.03 + 0.10 * lz, 0.018 + 0.06 * lz, 0.010 + 0.03 * lz), cv2.LINE_AA)
        edge = np.int32([P(tx0, TABLE_Y, tz0), P(tx1, TABLE_Y, tz0), P(tx1, TABLE_Y - 0.06, tz0), P(tx0, TABLE_Y - 0.06, tz0)])
        cv2.fillPoly(img, [edge], (0.015 + 0.04 * lz, 0.009 + 0.02 * lz, 0.006 + 0.01 * lz), cv2.LINE_AA)
        # the pool of lamplight on the desk (collected into one low-res layer, blurred once below)
        if lvl > 0.01:
            pool = np.int32([np.array(P(x + 0.42 * math.cos(a_), TABLE_Y, z + 0.42 * math.sin(a_))) / (S * 4) * 4
                             for a_ in np.linspace(0, 2 * math.pi, 24)])
            cv2.fillPoly(pool_buf, [pool], float(lvl), cv2.LINE_AA, shift=2)
        # the lamp: a green shade, a warm bulb beneath it
        sh_y0, sh_y1, sw = 0.36, 0.48, 0.17
        shade = np.int32([P(x - sw, sh_y0, z), P(x + sw, sh_y0, z), P(x + sw * 0.55, sh_y1, z), P(x - sw * 0.55, sh_y1, z)])
        cv2.fillPoly(img, [shade], (0.03 + 0.03 * lvl, 0.10 + 0.10 * lvl, 0.06 + 0.04 * lvl), cv2.LINE_AA)
        stem = (np.int32(P(x, TABLE_Y, z)), np.int32(P(x, sh_y0, z)))
        cv2.line(img, tuple(stem[0]), tuple(stem[1]), (0.10 * lz + 0.02, 0.08 * lz + 0.015, 0.05 * lz + 0.01), max(1, int(S * 6 / (z - cz))), cv2.LINE_AA)
        if lvl > 0.01:
            bx, by = P(x, sh_y0 - 0.01, z)
            r = max(2.0, S * 22.0 / (z - cz))
            cv2.ellipse(img, (int(bx), int(by)), (int(r * 2.2), int(r * 0.7)), 0, 0, 360,
                        tuple(float(v) for v in AMBER_HOT * 1.6 * lvl), -1, cv2.LINE_AA)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    soft = cv2.GaussianBlur(pool_buf, (0, 0), 9.0)
    core_ = cv2.GaussianBlur(pool_buf, (0, 0), 3.0) ** 2
    pools = cv2.resize(soft, (W, H))[..., None] * np.array([0.46, 0.30, 0.15]) + cv2.resize(core_, (W, H))[..., None] * np.array([0.22, 0.17, 0.11])
    img = img + pools.astype(np.float32)
    # haze and glow
    img = img + cv2.resize(cv2.GaussianBlur(cv2.resize(np.clip(img - 0.35, 0, None), (W // 4, H // 4)), (0, 0), 6), (W, H)) * 1.4
    return img


# ---------------------------------------------------------------------------- her, on the other side of the glass
_flat_cache = {}
def flat_pose(name):
    """The character redrawn flat (soft warm linework, few flat colours) to sit inside the library's style."""
    if name in _flat_cache: return _flat_cache[name]
    import os
    from PIL import Image as _I
    path = f'build/flat_{name}.png'
    if not os.path.exists(path):
        im = np.asarray(_I.open(f'build/{name}.png').convert('RGBA'))
        rgb = im[..., :3].copy(); a = im[..., 3]
        ep = cv2.edgePreservingFilter(rgb, flags=1, sigma_s=30, sigma_r=0.25)
        Z = ep.reshape(-1, 3).astype(np.float32)
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, _, cen = cv2.kmeans(Z[a.reshape(-1) > 128][::3], 14, None, crit, 2, cv2.KMEANS_PP_CENTERS)
        d = ((Z[:, None, :] - cen[None]) ** 2).sum(-1); q = cen[np.argmin(d, 1)].reshape(ep.shape)
        lum = rgb.mean(axis=2); ink = np.clip((90 - lum) / 60, 0, 1)[..., None]
        q = q * (1 - ink * 0.75) + np.array([70, 34, 26]) * ink * 0.75
        _I.fromarray(np.dstack([np.clip(q, 0, 255).astype(np.uint8), a])).save(path)
    im = np.asarray(_I.open(path).convert('RGBA')).astype(np.float32) / 255
    _flat_cache[name] = im
    return im

PEEK_BOX = {'think': (90, 20, 440, 470), 'offer': (0, 230, 330, 520)}
def peek(img, name, mirror, x, y, scale, light, alpha, side):
    """Composite a cropped, flat, lamp-lit piece of her at (x, y) = top-left, clipped by the frame edge."""
    if alpha <= 0.01: return None
    im = flat_pose(name)
    x0, y0, x1, y1 = PEEK_BOX[name]
    c = im[y0:y1, x0:x1]
    if mirror: c = c[:, ::-1]
    c = cv2.resize(c, (int(c.shape[1] * scale), int(c.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    h, w = c.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # lit by the reading-room lamps from the room side; the side against the glass falls into shadow
    toward = (xx / w) if side == 'left' else (1 - xx / w)
    lamp = (0.18 + 0.95 * toward ** 1.3 * np.exp(-((yy - h * 0.75) / (h * 0.9)) ** 2))[..., None] * light
    rgb = c[..., :3] * lamp * np.array([1.06, 0.86, 0.66], np.float32)
    a = c[..., 3:4] * alpha
    lay = np.concatenate([rgb * a, a], 2)
    blit(img, lay, x, y, anchor='tl')
    return (x, y, w, h)

def glass_ripple(img, px, py, u):
    """Where her fingers meet the screen: the wall between her world and yours shivers, and holds."""
    for k in range(3):
        r = 12 + 140 * (u - 0.12 * k)
        if r <= 0: continue
        a = 0.22 * math.exp(-(u - 0.12 * k) * 2.2)
        if a < 0.01: continue
        cv2.ellipse(img, (int(px), int(py)), (int(r * 0.35), int(r)), 0, 0, 360, tuple(float(v) for v in AMBER_HOT * a), 2, cv2.LINE_AA)
    glow_spot(img, px, py, 22, AMBER_HOT, 0.5 * math.exp(-u * 3))

def library_peeks(img, lt, master):
    lit = max(master, 0.0)
    # 1 · from the left edge: her face leans in, close to us, eyes on the lit reading room
    if 2.6 < lt < 7.8:
        k = ease((lt - 2.6) / 1.5) * (1 - ease((lt - 6.5) / 1.3))
        x = -640 + 520 * k + 8 * math.sin(lt * 1.3)
        peek(img, 'think', True, x, CY - 270 + 5 * math.sin(lt * 0.9), 1.8, 0.95 * lit, 0.96 * sstep(0, 0.3, k), 'left')
        if k > 0.6: img[TOP:BOT, 0:3] += AMBER_HOT * 0.3 * (k - 0.6)       # the edge of her world catches the light
    # 2 · from the right edge: her hand reaches for a pool of lamplight, and meets the glass
    if 10.8 < lt < 15.6:
        k = ease((lt - 10.8) / 1.8) * (1 - ease((lt - 14.2) / 1.3))
        x = W - 60 - 470 * k
        peek(img, 'offer', False, x, CY - 60, 1.6, 1.0 * lit, 0.96 * sstep(0, 0.25, k), 'right')
        if 12.6 < lt < 15.0:
            glass_ripple(img, x + 40, CY - 60 + 225, lt - 12.6)
    # 3 · the last lamp: she watches it, from the left edge, until the dark takes her too
    if 18.2 < lt < 22.4:
        k = ease((lt - 18.2) / 1.6)
        x = -640 + 470 * k
        peek(img, 'think', True, x, CY - 200, 1.6, 0.9 * lit + 0.05, 0.92 * sstep(0, 0.3, k) * (0.2 + 0.8 * lit), 'left')

def s_library(lt, dur):
    """Ten minutes to closing: the lamps go out from the far end, row by row, until the last one."""
    cz = 0.0 + 0.05 * lt
    front = LIB_Z1 + 1 - (LIB_Z1 + 1 - 2.6) * ease(clamp((lt - 4.5) / 13.5))
    D = lib_data()
    if lt > 18.0: front = 2.6            # only the nearest lamps remain
    master = 1.0
    if lt > 20.6:                        # the last lamp: one blink, then out
        u = lt - 20.6
        master = 0.0 if u > 0.9 else (1.0 if u < 0.25 or 0.45 < u < 0.6 else 0.15)
    if lt > 21.6: master = 0.0
    img = render_library(lt, cz, front=front, master=master)
    library_peeks(img, lt, master if lt > 20.6 else 1.0)
    k = sstep(0.0, 0.9, lt)
    return img * k + VOID * (1 - k)

SHOTS = dict(library=s_library, embers=s_embers, hall=s_hall, hall_dark=s_hall_dark, slit=s_slit, clocks=s_clocks, line=s_line, keyboard=s_keyboard)
def render(shot, lt, dur):
    return SHOTS[shot](lt, dur)
