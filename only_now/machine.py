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

# ---------------------------------------------------------------- the hand that types (a faceted, paper-white cursor hand)
_HAND = None
def hand_mesh():
    """A pointing hand in local units (fingertip at (0,-1), wrist toward +y), cut into flat facets like folded paper."""
    global _HAND
    if _HAND is not None: return _HAND
    from scipy.spatial import Delaunay
    outline = np.array([
        (-0.11, -1.00), (0.00, -1.06), (0.11, -1.00), (0.13, -0.62), (0.15, -0.36),     # index finger
        (0.32, -0.40), (0.46, -0.30), (0.50, -0.14), (0.53, 0.04), (0.52, 0.22),       # folded fingers
        (0.46, 0.40), (0.34, 0.52), (0.30, 0.66), (0.33, 0.92), (-0.18, 0.98),           # wrist / cuff
        (-0.20, 0.70), (-0.28, 0.52), (-0.38, 0.30), (-0.62, 0.02), (-0.66, -0.12),      # palm edge, thumb
        (-0.52, -0.16), (-0.30, -0.06), (-0.14, -0.30), (-0.13, -0.62)], np.float64)
    rng = np.random.default_rng(31)
    inner = []
    poly = np.float32(outline * 1000)
    while len(inner) < 34:
        p = rng.uniform((-0.6, -1.0), (0.55, 0.95))
        if cv2.pointPolygonTest(poly, (float(p[0] * 1000), float(p[1] * 1000)), True) > 40: inner.append(p)
    pts = np.vstack([outline, np.array(inner)])
    tri = Delaunay(pts).simplices
    keep = [t for t in tri if cv2.pointPolygonTest(poly, tuple(float(v) for v in pts[t].mean(0) * 1000), False) > 0]
    hgt = rng.uniform(0, 0.35, len(pts)); hgt[:len(outline)] *= 0.3            # a crumpled-paper relief
    shade = []
    L_ = np.array([-0.5, -0.6, 0.62]); L_ /= np.linalg.norm(L_)
    for t in keep:
        a, b, c = [np.array([pts[i][0], pts[i][1], hgt[i]]) for i in t]
        n = np.cross(b - a, c - a); n /= (np.linalg.norm(n) + 1e-9)
        if n[2] < 0: n = -n
        shade.append(0.52 + 0.48 * max(0.0, float(n @ L_)))
    _HAND = dict(pts=pts, tri=np.array(keep), shade=np.array(shade), outline=outline, delay=rng.uniform(0, 1, len(keep)))
    return _HAND

def draw_hand(img, tip, ang, scale, alpha, dissolve, warm):
    """Composite the hand at 1x: fingertip at `tip`, rotated by `ang`; facets fall away one by one as it dissolves."""
    M_ = hand_mesh()
    c, s_ = math.cos(ang), math.sin(ang)
    P2 = np.stack([M_['pts'][:, 0] * c - M_['pts'][:, 1] * s_, M_['pts'][:, 0] * s_ + M_['pts'][:, 1] * c], 1)
    tip_l = np.array([0.0 * c - (-1.06) * s_, 0.0 * s_ + (-1.06) * c])
    P2 = (P2 - tip_l) * scale + tip
    x0, y0 = np.floor(P2.min(0)) - 4; x1, y1 = np.ceil(P2.max(0)) + 4
    x0, y0 = max(int(x0), 0), max(int(y0), 0); x1, y1 = min(int(x1), W), min(int(y1), H)
    if x1 <= x0 or y1 <= y0 or alpha <= 0.01: return None
    SS = 3
    lay = np.zeros(((y1 - y0) * SS, (x1 - x0) * SS, 4), np.float32)
    q = (P2 - (x0, y0)) * SS
    base = np.array([0.95, 0.94, 0.92], np.float32) * (1 - 0.25 * warm) + np.array([1.0, 0.82, 0.62], np.float32) * 0.25 * warm
    for t, sh, dl in zip(M_['tri'], M_['shade'], M_['delay']):
        a = alpha * (1 - sstep(dl * 0.7, dl * 0.7 + 0.3, dissolve))
        if a <= 0.01: continue
        drift = np.array([0.0, -40.0 * SS]) * sstep(dl * 0.7, dl * 0.7 + 0.3, dissolve)
        tri = np.int32((q[t] + drift) * 4)
        col = np.concatenate([base * sh * a, [a]]).astype(np.float32)
        cv2.fillPoly(lay, [tri], tuple(float(v) for v in col), cv2.LINE_AA, shift=4 - 2)
    lay = cv2.resize(lay, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
    d = img[y0:y1, x0:x1]
    d *= (1 - lay[..., 3:4]); d += lay[..., :3]
    return (P2, (x0, y0, x1, y1))

def hand_shadow(img, tip, ang, scale, lift, alpha):
    """Its soft shadow on the keys: further off and softer the higher the hand is."""
    M_ = hand_mesh()
    c, s_ = math.cos(ang), math.sin(ang)
    P2 = np.stack([M_['outline'][:, 0] * c - M_['outline'][:, 1] * s_, M_['outline'][:, 0] * s_ + M_['outline'][:, 1] * c], 1)
    tip_l = np.array([1.06 * s_, -1.06 * c])
    P2 = (P2 - tip_l) * scale + tip + np.array([18, 30]) * (0.3 + lift)
    m = np.zeros((H // 2, W // 2), np.float32)
    cv2.fillPoly(m, [np.int32(P2 / 2 * 4)], 1.0, cv2.LINE_AA, shift=2)
    m = cv2.GaussianBlur(m, (0, 0), 3 + 9 * lift)
    img *= (1 - 0.6 * alpha * cv2.resize(m, (W, H)))[..., None]

KB_TAP = ('G', 2)          # the key the hand presses (under the lamp)
KB_T = dict(enter=6.2, hover=8.2, press=8.62, contact=8.85, release=9.25, dissolve=(9.15, 10.3), flatten=(10.2, 11.5), dark=(9.8, 11.3))

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
    # the tap: where it lands, and the rings of light that spread from it across the keys
    tk = [k_ for k_ in keys if k_[0] == KB_TAP[0]][0]
    tu, tv = tk[1] + tk[3] / 2, tk[2] + tk[4] / 2
    dtc = lt - KB_T['contact']
    def ripple(d):
        b = 0.0
        for j in range(3):
            tt = dtc - 0.22 * j
            if tt > 0:
                R = 13.0 * tt ** 0.9
                b += 0.9 * (0.6 ** j) * math.exp(-tt * 0.7) * math.exp(-((d - R) / 1.4) ** 2)
        return b
    press = sstep(KB_T['press'], KB_T['contact'], lt) * (1 - sstep(KB_T['contact'] + 0.05, KB_T['release'] + 0.3, lt))
    for ch, u, v, kw, kh in keys:
        depth = (v + kh / 2) / 11.0
        h = (10 + 18 * depth) * S * push
        if ch == pressed[0]: h *= 0.35 + 0.65 * rise
        rb = ripple(math.hypot(u + kw / 2 - tu, v + kh / 2 - tv)) if dtc > 0 else 0.0
        h += 7 * S * rb
        if ch == KB_TAP[0]: h *= 1 - 0.65 * press
        tl, tr, br_, bl = P(u, v), P(u + kw, v), P(u + kw, v + kh), P(u, v + kh)
        top = np.float32([tl, tr, br_, bl]) - [0, h]
        front = np.float32([bl - [0, h], br_ - [0, h], br_, bl])
        cxk, cyk = top.mean(axis=0) / S
        li = float(lamp[int(np.clip(cyk, 0, H - 1)), int(np.clip(cxk, 0, W - 1))])
        da = float(dawn[int(np.clip(cyk, 0, H - 1)), int(np.clip(cxk, 0, W - 1))])
        topc = np.array([0.054, 0.044, 0.036]) * (0.3 + 6.0 * li ** 1.3) + np.array([0.015, 0.022, 0.04]) * da + AMBER * 0.42 * rb
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
    img = img * k + VOID * (1 - k)
    # --- the hand: in from the upper right, hovers, presses, and comes apart into light
    kt = np.float32([tk[1] + tk[3] / 2, tk[2] + tk[4] / 2])
    key_top = P(*kt) / S - [0, (10 + 18 * (kt[1] / 11.0)) * push]
    if KB_T['enter'] < lt < KB_T['dissolve'][1]:
        import cursor_hand
        spr0, tip0, msk0 = cursor_hand.sprite()
        e = ease(clamp((lt - KB_T['enter']) / (KB_T['hover'] - KB_T['enter'])))
        hover = key_top + [10, -46]
        start = np.array([W + 120.0, key_top[1] + 220.0])
        tip = start + (hover - start) * e + np.array([0, 5 * math.sin(lt * 2.4)]) * e
        down = sstep(KB_T['press'], KB_T['contact'], lt) * (1 - sstep(KB_T['contact'] + 0.05, KB_T['release'], lt) * 0.6)
        tip = tip + (key_top + [0, 4] - hover) * down
        lift = clamp(1 - down) * e + (1 - e)
        dis = clamp((lt - KB_T['dissolve'][0]) / (KB_T['dissolve'][1] - KB_T['dissolve'][0]))
        sc = 0.60 * (1 - 0.04 * down)
        spr = cv2.resize(spr0, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        h_, w_ = spr.shape[:2]
        if dis > 0:                                      # it comes apart in pieces, each lifting away as light
            rng_ = np.random.default_rng(77)
            cell = 12
            thr = rng_.uniform(0, 1, (h_ // cell + 1, w_ // cell + 1)).astype(np.float32)
            thr = cv2.resize(thr, (w_ + cell, h_ + cell), interpolation=cv2.INTER_NEAREST)[:h_, :w_]
            keep = 1 - np.clip((dis * 1.3 - thr * 0.9) / 0.25, 0, 1)
            spr = spr * keep[..., None]
        x0_, y0_ = tip[0] - tip0[0] * sc, tip[1] - tip0[1] * sc
        # its soft shadow on the keys, further and softer the higher it is
        sm = cv2.resize(msk0, (w_, h_), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 4 + 10 * lift)
        sh = np.zeros((H, W), np.float32)
        ox, oy = int(x0_ + 16 * (0.3 + lift)), int(y0_ + 30 * (0.3 + lift))
        xa, ya = max(ox, 0), max(oy, 0); xb, yb = min(ox + w_, W), min(oy + h_, H)
        if xb > xa and yb > ya:
            sh[ya:yb, xa:xb] = sm[ya - oy:yb - oy, xa - ox:xb - ox]
        img *= (1 - 0.6 * e * (1 - dis) * sh)[..., None]
        lay = spr.copy(); lay[..., :3] *= np.array([1.0, 0.93, 0.84], np.float32) * (0.75 + 0.25 * e)   # in the warm lamp light
        blit(img, lay * e, x0_, y0_, anchor='tl')
    # --- the dark comes in; the ripple's first ring flattens into the line of the next shot
    dk = sstep(*KB_T['dark'], lt)
    img = img * (1 - dk) + VOID * dk
    if dtc > 0:
        buf = np.zeros((H, W, 3), np.float32)
        m = ease(clamp((lt - KB_T['flatten'][0]) / (KB_T['flatten'][1] - KB_T['flatten'][0])))
        HX = head_x(0.8)
        for j in range(3):
            tt = dtc - 0.22 * j
            if tt <= 0: continue
            if j == 0:                                     # the first ring settles; the others run on and fade
                R = 7.0 * (1 - math.exp(-tt * 1.6))
                a = 0.9 * (0.55 + 0.45 * math.exp(-tt * 1.2))
            else:
                R = 13.0 * tt ** 0.9
                a = 0.9 * (0.6 ** j) * math.exp(-tt * 0.9)
            th = np.linspace(0, 2 * math.pi, 220)
            ring = np.array([P(tu + R * math.cos(t_), tv + R * math.sin(t_) * 0.85) / S for t_ in th]) - [0, 14]
            if j == 0:
                tx = X0 + (HX - X0) * (0.5 + 0.5 * np.cos(th))     # right of the ring -> head of the line: no twist
                tgt = np.stack([tx, line_y(tx)], 1)
                ring = ring * (1 - m) + tgt * m
                a = a * (1 - m) + 0.55 * m
            else:
                a *= (1 - m)
            if a > 0.01:
                cv2.polylines(buf, [np.int32(ring * 4)], True, tuple(float(v) for v in AMBER * a), 2, cv2.LINE_AA, shift=2)
        img += glowbuf(buf, 0.7, 3.0, 14, (1.0, 0.5, 0.25))
        if m > 0.75:                                      # the head of the line: today
            hy = float(line_y(np.array([HX]))[0])
            glow_spot(img, HX, hy, 9, AMBER_HOT, sstep(0.75, 1.0, m))
            glow_spot(img, HX, hy, 40, AMBER, 0.14 * sstep(0.75, 1.0, m))
        if 0 < dtc < 0.5:                                 # the touch itself
            glow_spot(img, key_top[0], key_top[1], 14, AMBER_HOT, 1.2 * (1 - dtc / 0.5))
    return img

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
TABLE_HALF = 0.55                       # every table: 1.15 wide (x 0.45..1.6), 1.1 deep; two chairs on its far side
CHAIR_X = (0.72, 1.36)
HER_X = 1.36                            # she sits in the second chair of her table, facing the camera
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
    her = int(np.argmin(np.abs(lamps[:, 1] - 8.2) + (lamps[:, 0] < 0) * 99))
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
    # other readers: real people — students in arty, modern clothes — bent over their work (never at her table)
    readers = []
    cands = [(i, c) for i in range(len(lamps)) for c in range(2) if i != her and 2.6 < lamps[i][1] < 22.0]
    pick = rng.choice(len(cands), size=16, replace=False)
    OUTFITS = {   # name: (main colour choices, inner colour)
        'hoodie': [(0.62, 0.52, 0.30), (0.30, 0.36, 0.30), (0.55, 0.27, 0.22), (0.62, 0.60, 0.56)],
        'cardigan': [(0.48, 0.38, 0.22), (0.22, 0.28, 0.36), (0.50, 0.30, 0.32)],
        'denim': [(0.22, 0.32, 0.48), (0.30, 0.40, 0.55)],
        'stripe': [(0.78, 0.74, 0.66), (0.20, 0.24, 0.36)],
        'turtle': [(0.12, 0.12, 0.13), (0.46, 0.20, 0.18), (0.70, 0.64, 0.52)],
        'vest': [(0.36, 0.30, 0.22), (0.18, 0.26, 0.24)],
    }
    HAIR = [(0.05, 0.04, 0.04), (0.13, 0.08, 0.05), (0.26, 0.15, 0.08), (0.30, 0.27, 0.24), (0.08, 0.06, 0.06)]
    HATS = [None, None, None, 'beanie', 'cap', 'beret', 'bucket']
    HATC = [(0.75, 0.55, 0.20), (0.62, 0.18, 0.16), (0.16, 0.18, 0.22), (0.80, 0.78, 0.72), (0.30, 0.38, 0.28)]
    PHONES = [(0.92, 0.90, 0.86), (0.10, 0.10, 0.11), (0.72, 0.80, 0.86), (0.86, 0.70, 0.66)]
    for k in pick:
        i, c = cands[k]
        kind = list(OUTFITS)[rng.integers(len(OUTFITS))]
        hat = HATS[rng.integers(len(HATS))]
        readers.append(dict(
            lamp=i, chair=c, kind=kind, cloth=np.array(OUTFITS[kind][rng.integers(len(OUTFITS[kind]))], np.float32),
            inner=np.array([(0.90, 0.88, 0.84), (0.20, 0.20, 0.22), (0.80, 0.70, 0.55)][rng.integers(3)], np.float32),
            hair=np.array(HAIR[rng.integers(len(HAIR))], np.float32),
            style=['short', 'long', 'bun', 'pony', 'bob'][rng.integers(5)] if hat is None else ['short', 'long', 'bob'][rng.integers(3)],
            hat=hat, hatc=np.array(HATC[rng.integers(len(HATC))], np.float32), pompom=rng.random() < 0.5,
            phones=rng.random() < 0.45, phc=np.array(PHONES[rng.integers(len(PHONES))], np.float32),
            prop=['book', 'book', 'laptop', 'laptop', 'notes'][rng.integers(5)], cup=rng.random() < 0.5, cupx=rng.choice([-1, 1]),
            stickers=rng.uniform(0, 1, (4, 3)).astype(np.float32),
            bow=rng.uniform(0.4, 1.0), w=rng.uniform(0.17, 0.22), h=rng.uniform(-0.03, 0.04), ph=rng.uniform(0, 6.3)))
    _lib = dict(lamps=lamps, books=books, her=her, readers=readers)
    return _lib

LIB_CAM = dict(x=0.0, y=LIB_CAMY, f=LIB_F, pitch=0.0)   # the library camera: lateral offset, height, lens, tilt down
def lib_depth(y, z, cz):
    p = LIB_CAM['pitch']
    return -(y - LIB_CAM['y']) * math.sin(p) + (z - cz) * math.cos(p)
def lproj(x, y, z, cz):
    p = LIB_CAM['pitch']
    dy, dz = y - LIB_CAM['y'], z - cz
    zp = max(-dy * math.sin(p) + dz * math.cos(p), 0.05)
    up = dy * math.cos(p) + dz * math.sin(p)
    return CX + LIB_CAM['f'] * (x - LIB_CAM['x']) / zp, CY - LIB_CAM['f'] * up / zp

def lamp_level(z, front, flick_t):
    """1 = lit, 0 = off; lamps beyond the closing front are off (with a brief flicker as they go)."""
    if front is None: return 1.0
    if z > front + 0.8: return 0.0
    if z > front: return 0.5 + 0.5 * math.sin(flick_t * 40 + z * 9)
    return 1.0

def chair(img, P, x, z, lit, S):
    """A reading-room chair seen from behind its table: two posts, a top rail and a lower rail, dark wood."""
    col = (0.035 + 0.16 * lit, 0.020 + 0.09 * lit, 0.012 + 0.05 * lit)
    w, pw = 0.21, 0.028
    for sx in (-1, 1):
        q = np.int32([P(x + sx * w - pw, FLOOR_Y, z), P(x + sx * w + pw, FLOOR_Y, z), P(x + sx * w + pw, 0.44, z), P(x + sx * w - pw, 0.44, z)])
        cv2.fillPoly(img, [q], col, cv2.LINE_AA)
    for (y0, y1) in ((0.33, 0.44), (0.12, 0.16)):
        q = np.int32([P(x - w, y0, z), P(x + w, y0, z), P(x + w, y1, z), P(x - w, y1, z)])
        cv2.fillPoly(img, [q], col, cv2.LINE_AA)
    hi = tuple(float(c * 1.8) for c in col)              # the worn top edge of the rail catches the light
    cv2.line(img, tuple(np.int32(P(x - w, 0.44, z))), tuple(np.int32(P(x + w, 0.44, z))), hi, max(1, S), cv2.LINE_AA)

SKIN_R = np.array([0.78, 0.60, 0.49], np.float32)
def reader_light(lvl, nf, lz):
    return nf * (0.06 + 0.25 * lz + 0.85 * lvl) * np.array([1.0, 0.86, 0.68], np.float32)

def _col(c, lit, k=1.0): return tuple(float(v) for v in np.asarray(c, np.float32) * lit * k)

def reader_body(img, P, x, z, R, lit, t, S):
    """A real reader behind the table: clothes, neck, bowed head, hair, hat, headphones. Solid — the light stops at them."""
    Q = lambda pts, c: cv2.fillPoly(img, [np.int32([P(*p) for p in pts])], c, cv2.LINE_AA)
    def L_(pts, c, wd):
        q = np.int32([P(*p) for p in pts])
        px = abs(P(pts[0][0] + wd, pts[0][1], pts[0][2])[0] - P(*pts[0])[0])
        cv2.polylines(img, [q], False, c, max(1, int(round(px))), cv2.LINE_AA)
    w, sh = R['w'], 0.30 + R['h']
    bow = R['bow'] * (0.92 + 0.08 * math.sin(t * 0.6 + R['ph']))
    zb = z - 0.08; zt = zb - 0.07 * bow
    cl, inn = R['cloth'], R['inner']
    # the hood or the ponytail sit behind the shoulders
    if R['kind'] == 'hoodie':
        Q([(x - 0.12, sh - 0.02, zt + 0.03), (x + 0.12, sh - 0.02, zt + 0.03), (x + 0.10, sh + 0.06, zt + 0.04), (x - 0.10, sh + 0.06, zt + 0.04)], _col(cl, lit, 0.75))
    torso = [(x - w, -0.02, zb), (x + w, -0.02, zb), (x + w, sh - 0.05, zb - 0.05 * bow), (x + w * 0.65, sh, zt), (x - w * 0.65, sh, zt), (x - w, sh - 0.05, zb - 0.05 * bow)]
    Q(torso, _col(cl, lit))
    k = R['kind']
    if k == 'stripe':                                     # a striped knit
        for j in range(5):
            y0 = 0.02 + j * (sh - 0.06) / 5
            Q([(x - w, y0, zb), (x + w, y0, zb), (x + w, y0 + 0.022, zb), (x - w, y0 + 0.022, zb)], _col(inn if np.mean(cl) < 0.5 else (0.16, 0.20, 0.32), lit))
    if k == 'cardigan':                                   # open front over a shirt, a row of buttons
        Q([(x - 0.05, sh, zt), (x + 0.05, sh, zt), (x + 0.07, -0.02, zb), (x - 0.07, -0.02, zb)], _col(inn, lit))
        for j in range(3):
            cx_, cy_ = P(x - 0.075, 0.06 + j * 0.07, zb)
            cv2.circle(img, (int(cx_), int(cy_)), max(1, int(S * 0.9)), _col((0.85, 0.75, 0.55), lit), -1, cv2.LINE_AA)
    if k == 'denim':                                      # a denim jacket: tee underneath, a light collar, yoke seams
        Q([(x - 0.045, sh, zt), (x + 0.045, sh, zt), (x + 0.03, sh - 0.10, zt), (x - 0.03, sh - 0.10, zt)], _col((0.90, 0.88, 0.84), lit))
        for sx in (-1, 1):
            Q([(x + sx * 0.04, sh + 0.005, zt), (x + sx * 0.12, sh - 0.01, zt), (x + sx * 0.06, sh - 0.09, zt)], _col(cl, lit, 1.35))
            L_([(x + sx * w * 0.95, sh - 0.11, zb), (x + sx * 0.05, sh - 0.12, zb)], _col(cl, lit, 1.5), 0.006)
    if k == 'vest':                                       # a collared shirt under a sweater vest
        Q([(x - 0.06, sh, zt), (x + 0.06, sh, zt), (x, sh - 0.11, zt)], _col((0.88, 0.86, 0.82), lit))
        for sx in (-1, 1):
            Q([(x + sx * w, sh - 0.05, zb), (x + sx * w * 0.65, sh, zt), (x + sx * w * 0.62, sh - 0.12, zb)], _col((0.88, 0.86, 0.82), lit))
    if k == 'hoodie':                                     # drawstrings
        for sx in (-1, 1):
            L_([(x + sx * 0.035, sh - 0.01, zt), (x + sx * 0.04, sh - 0.13, zt)], _col((0.92, 0.90, 0.86), lit), 0.005)
    # neck (a turtleneck covers it)
    hy, hz = sh + 0.10 - 0.07 * bow, zb - 0.07 - 0.13 * bow
    Q([(x - 0.035, sh - 0.01, zt), (x + 0.035, sh - 0.01, zt), (x + 0.03, hy, hz), (x - 0.03, hy, hz)], _col(cl if k == 'turtle' else (0.78, 0.60, 0.49), lit, 0.8))
    cx_, cy_ = P(x, hy, hz)
    rx = abs(P(x + 0.085, hy, hz)[0] - cx_); ry = rx * 1.12
    E = lambda dx, dy, ax, ay, c, ang=0: cv2.ellipse(img, (int(cx_ + dx * rx), int(cy_ + dy * ry)), (max(1, int(ax * rx)), max(1, int(ay * ry))), ang, 0, 360, c, -1, cv2.LINE_AA)
    hc = R['hair']
    if R['style'] == 'pony':                              # the ponytail falls behind
        Q([(x - 0.03, hy + 0.04, hz + 0.08), (x + 0.03, hy + 0.04, hz + 0.08), (x + 0.025, sh - 0.10, zt + 0.06), (x - 0.02, sh - 0.10, zt + 0.06)], _col(hc, lit, 1.3))
    if R['style'] in ('long', 'bob'):
        drop = sh - 0.14 if R['style'] == 'long' else hy - 0.06
        Q([(x - 0.095, hy + 0.02, hz), (x + 0.095, hy + 0.02, hz), (x + 0.11, drop, zt), (x - 0.11, drop, zt)], _col(hc, lit, 1.3))
    E(0, 0, 1.0, 1.0, _col((0.78, 0.60, 0.49), lit, 0.85))                    # face (mostly hidden as they read)
    E(0, -0.10 - 0.08 * bow, 1.08, 0.92 + 0.12 * bow, _col(hc, lit, 1.15))    # hair
    # clearer hair: locks flowing from the parting down to the hairline, the ring of shine, the parting itself
    lw = max(1, int(rx * 0.045))
    for sgn in (-1, 1):
        for j in range(3):
            u = 0.25 + 0.3 * j
            pts = []
            for v in np.linspace(0, 1, 8):
                ang = math.pi * (1.5 + sgn * (0.12 + u * 0.75) * v)
                rr = 0.25 + 0.8 * v
                pts.append((cx_ - rx * 0.05 + rx * rr * math.cos(ang) * (0.6 + 0.4 * u),
                            cy_ - ry * 0.95 + ry * (0.9 + 0.3 * u) * v * (0.5 + 0.5 * v) + 0 * math.sin(ang)))
            cv2.polylines(img, [np.int32(np.array(pts) * 4)], False, _col(hc, lit, 0.95), lw, cv2.LINE_AA, shift=2)
    cv2.ellipse(img, (int(cx_), int(cy_ - ry * 0.18)), (max(1, int(rx * 0.78)), max(1, int(ry * 0.55))), 0, 205, 335,
                _col(hc * 1.5 + 0.06, lit, 1.3), max(1, int(rx * 0.12)), cv2.LINE_AA)
    cv2.line(img, (int(cx_ - rx * 0.05), int(cy_ - ry * 1.0)), (int(cx_ - rx * 0.02), int(cy_ - ry * 0.62)), _col((0.78, 0.60, 0.49), lit, 0.6), lw, cv2.LINE_AA)
    if R['style'] == 'bun':
        E(0.0, -1.05, 0.42, 0.36, _col(hc, lit, 1.5))
    # hats
    hat, hcol = R['hat'], R['hatc']
    if hat == 'beanie':
        E(0, -0.30, 1.10, 0.85, _col(hcol, lit))
        cv2.rectangle(img, (int(cx_ - rx * 1.1), int(cy_ - ry * 0.10)), (int(cx_ + rx * 1.1), int(cy_ + ry * 0.22)), _col(hcol, lit, 1.25), -1, cv2.LINE_AA)
        for j in range(6):
            xx_ = cx_ - rx * 0.9 + j * rx * 0.36
            cv2.line(img, (int(xx_), int(cy_ - ry * 0.08)), (int(xx_), int(cy_ + ry * 0.2)), _col(hcol, lit, 0.8), max(1, int(rx * 0.05)), cv2.LINE_AA)
        if R['pompom']: E(0.0, -1.18, 0.34, 0.30, _col(hcol, lit, 1.4))
    elif hat == 'cap':
        E(0, -0.28, 1.08, 0.82, _col(hcol, lit))
        E(0, 0.42, 0.95, 0.30, _col(hcol, lit, 0.7))                          # the brim, toward us as they bow
        E(0, -0.95, 0.12, 0.08, _col(hcol, lit, 1.3))
    elif hat == 'beret':
        E(0.15, -0.55, 1.15, 0.55, _col(hcol, lit), -12)
        E(0.15, -1.05, 0.08, 0.10, _col(hcol, lit, 0.8))
    elif hat == 'bucket':
        E(0, 0.05, 1.45, 0.55, _col(hcol, lit, 0.8))
        E(0, -0.35, 1.0, 0.75, _col(hcol, lit))
    # headphones: a band over the head, a cup over each ear
    if R['phones']:
        pc = R['phc']
        cv2.ellipse(img, (int(cx_), int(cy_ - ry * 0.1)), (int(rx * 1.12), int(ry * 1.0)), 0, 180, 360, _col(pc, lit, 0.9), max(1, int(rx * 0.16)), cv2.LINE_AA)
        for sx in (-1, 1):
            E(sx * 1.08, 0.05, 0.28, 0.40, _col(pc, lit))
            E(sx * 1.12, 0.05, 0.12, 0.25, _col(pc, lit, 0.6))
    return (cx_, cy_)

def reader_arms(img, P, x, z, R, lit, t, S):
    """Forearms on the table and what they are working on: drawn over the pool of light, which they block."""
    Q = lambda pts, c: cv2.fillPoly(img, [np.int32([P(*p) for p in pts])], c, cv2.LINE_AA)
    zb = z - 0.08
    bz0, bz1 = zb - 0.30, zb - 0.47
    turn = 0.5 + 0.5 * math.sin(t * 0.21 + R['ph'] * 3)
    cl = R['cloth']
    if R['prop'] == 'book':
        Q([(x - 0.14, 0.012, bz0), (x + 0.14, 0.012, bz0), (x + 0.14, 0.012, bz1), (x - 0.14, 0.012, bz1)], _col((0.86, 0.80, 0.68), lit, 1.1))
        cv2.line(img, tuple(np.int32(P(x, 0.012, bz0))), tuple(np.int32(P(x, 0.012, bz1))), _col((0.5, 0.45, 0.38), lit), max(1, S), cv2.LINE_AA)
    elif R['prop'] == 'notes':
        Q([(x - 0.10, 0.012, bz0), (x + 0.10, 0.012, bz0), (x + 0.11, 0.012, bz1), (x - 0.09, 0.012, bz1)], _col((0.92, 0.90, 0.84), lit, 1.1))
        for j in range(4):
            yy = bz0 - 0.03 - j * 0.035
            cv2.line(img, tuple(np.int32(P(x - 0.08, 0.013, yy))), tuple(np.int32(P(x + 0.06, 0.013, yy))), _col((0.45, 0.55, 0.75), lit, 0.8), max(1, S // 2), cv2.LINE_AA)
    # sleeves and hands
    for sx in (-1, 1):
        e = (x + sx * (R['w'] - 0.02), 0.03, zb - 0.06)
        hnd = (x + sx * (0.08 + 0.02 * turn * (sx > 0)), 0.02, bz0 - 0.04)
        Q([(x + sx * (R['w'] - 0.06), 0.26 + R['h'], zb - 0.05), (x + sx * (R['w'] + 0.01), 0.24 + R['h'], zb - 0.05),
           (e[0] + sx * 0.03, e[1], e[2]), (e[0] - sx * 0.04, e[1], e[2])], _col(cl, lit, 0.9))
        Q([(e[0] - 0.035, e[1], e[2]), (e[0] + 0.035, e[1], e[2]), (hnd[0] + 0.025, hnd[1], hnd[2]), (hnd[0] - 0.025, hnd[1], hnd[2])], _col(cl, lit))
        if R['kind'] == 'denim':                              # sleeves rolled up
            Q([(hnd[0] - 0.027, 0.02, hnd[2] + 0.06), (hnd[0] + 0.027, 0.02, hnd[2] + 0.06), (hnd[0] + 0.027, 0.02, hnd[2] + 0.03), (hnd[0] - 0.027, 0.02, hnd[2] + 0.03)], _col(cl, lit, 1.4))
        hx, hy_ = P(*hnd)
        rr = abs(P(hnd[0] + 0.03, hnd[1], hnd[2])[0] - hx)
        cv2.ellipse(img, (int(hx), int(hy_)), (max(1, int(rr)), max(1, int(rr * 0.7))), 0, 0, 360, _col((0.78, 0.60, 0.49), lit), -1, cv2.LINE_AA)
    if R['prop'] == 'laptop':                                 # the lid faces us: stickers on it, the screen's glow on their face
        Q([(x - 0.15, 0.012, bz0), (x + 0.15, 0.012, bz0), (x + 0.15, 0.012, bz0 + 0.02), (x - 0.15, 0.012, bz0 + 0.02)], _col((0.55, 0.56, 0.58), lit))
        Q([(x - 0.15, 0.012, bz0 + 0.02), (x + 0.15, 0.012, bz0 + 0.02), (x + 0.15, 0.21, bz0 + 0.08), (x - 0.15, 0.21, bz0 + 0.08)], _col((0.62, 0.63, 0.66), lit))
        for (u, v, c) in R['stickers']:
            sx_, sy_ = x - 0.12 + 0.24 * u, 0.05 + 0.13 * v
            col = [(0.85, 0.45, 0.35), (0.40, 0.65, 0.55), (0.90, 0.80, 0.40), (0.50, 0.55, 0.85)][int(c * 3.99)]
            Q([(sx_ - 0.018, sy_ - 0.015, bz0 + 0.05), (sx_ + 0.018, sy_ - 0.015, bz0 + 0.05), (sx_ + 0.018, sy_ + 0.015, bz0 + 0.06), (sx_ - 0.018, sy_ + 0.015, bz0 + 0.06)], _col(col, lit, 1.1))
    if R['cup']:                                              # a takeaway coffee, with a kraft sleeve
        cxw, czw = x + R['cupx'] * 0.22, bz0 + 0.02
        Q([(cxw - 0.03, 0.0, czw), (cxw + 0.03, 0.0, czw), (cxw + 0.036, 0.12, czw), (cxw - 0.036, 0.12, czw)], _col((0.92, 0.90, 0.86), lit))
        Q([(cxw - 0.032, 0.04, czw), (cxw + 0.032, 0.04, czw), (cxw + 0.034, 0.08, czw), (cxw - 0.034, 0.08, czw)], _col((0.62, 0.45, 0.28), lit))
        Q([(cxw - 0.038, 0.12, czw), (cxw + 0.038, 0.12, czw), (cxw + 0.036, 0.135, czw), (cxw - 0.036, 0.135, czw)], _col((0.20, 0.20, 0.22), lit))

def pool_local(img, P, x, z, lvl, tx0, tx1, tz0, tz1, S):
    """The pool of lamplight, laid on its own table top only (so whatever stands in front of it hides it)."""
    pts = np.array([P(x + 0.42 * math.cos(a_), TABLE_Y, z + 0.42 * math.sin(a_)) for a_ in np.linspace(0, 2 * math.pi, 24)], np.float64)
    top = np.array([P(tx0, TABLE_Y, tz0), P(tx1, TABLE_Y, tz0), P(tx1, TABLE_Y, tz1), P(tx0, TABLE_Y, tz1)], np.float64)
    r = 4 * S                                         # work at a quarter of the output resolution
    pad = 30 * r
    x0, y0 = int(max(min(pts[:, 0].min(), top[:, 0].min()) - pad, 0)), int(max(min(pts[:, 1].min(), top[:, 1].min()) - pad, 0))
    x1 = int(min(max(pts[:, 0].max(), top[:, 0].max()) + pad, img.shape[1])); y1 = int(min(max(pts[:, 1].max(), top[:, 1].max()) + pad, img.shape[0]))
    if x1 - x0 < r or y1 - y0 < r: return
    bw, bh = max(1, (x1 - x0) // r), max(1, (y1 - y0) // r)
    buf = np.zeros((bh, bw), np.float32); msk = np.zeros((bh, bw), np.float32)
    cv2.fillPoly(buf, [np.int32((pts - (x0, y0)) / r * 4)], float(lvl), cv2.LINE_AA, shift=2)
    cv2.fillPoly(msk, [np.int32((top - (x0, y0)) / r * 4)], 1.0, cv2.LINE_AA, shift=2)
    soft = cv2.GaussianBlur(buf, (0, 0), 9.0); core = cv2.GaussianBlur(buf, (0, 0), 3.0) ** 2
    lay = (soft[..., None] * np.array([0.46, 0.30, 0.15], np.float32) + core[..., None] * np.array([0.22, 0.17, 0.11], np.float32)) * msk[..., None]
    lay = cv2.resize(lay, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR)
    img[y0:y1, x0:x1] += lay

def render_library(lt, cz, front=None, last_on=1.0, master=1.0, her_lvl=None, her=None):
    D = lib_data()
    S = 2
    img = np.zeros((H * S, W * S, 3), np.float32)
    def P(x, y, z):
        u, v = lproj(x, y, z, cz); return (u * S, v * S)
    lamps = D['lamps']
    lv = np.array([lamp_level(z, front, lt) for _, z in lamps], np.float32) * master
    if her_lvl is not None: lv[D['her']] = her_lvl
    near = (lambda z: sstep(0.45, 1.7, z - cz)) if LIB_CAM.get('nearfade', True) else (lambda z: 1.0)   # things the camera backs past come out of the dark
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
        fog = math.exp(-(z0 - cz) / 26.0) * near(z0)
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
        nf = near(z)
        side = 1 if x > 0 else -1
        # the two chairs pushed in at the far side of this table (drawn first: they are behind it)
        lz = L(z)
        for cxx in CHAIR_X:
            chair(img, P, side * cxx, z + TABLE_HALF + 0.2, nf * (0.6 * lz + 0.25 * lvl), S)
        if her is not None and idx == D['her']:
            her_mask = np.zeros(img.shape[:2], np.float32)
            her(img, S, her_mask)                    # she sits in her chair, behind her table
        here = [R for R in D['readers'] if R['lamp'] == idx]
        rl = reader_light(lvl, nf, lz)
        for R in here:
            reader_body(img, P, side * CHAIR_X[R['chair']], z + TABLE_HALF + 0.2, R, rl, lt, S)
        tx0, tx1 = side * 0.45, side * 1.6
        tz0, tz1 = z - TABLE_HALF, z + TABLE_HALF
        top = np.int32([P(tx0, TABLE_Y, tz0), P(tx1, TABLE_Y, tz0), P(tx1, TABLE_Y, tz1), P(tx0, TABLE_Y, tz1)])
        cv2.fillPoly(img, [top], (nf * (0.03 + 0.10 * lz), nf * (0.018 + 0.06 * lz), nf * (0.010 + 0.03 * lz)), cv2.LINE_AA)
        edge = np.int32([P(tx0, TABLE_Y, tz0), P(tx1, TABLE_Y, tz0), P(tx1, TABLE_Y - 0.06, tz0), P(tx0, TABLE_Y - 0.06, tz0)])
        cv2.fillPoly(img, [edge], (nf * (0.015 + 0.04 * lz), nf * (0.009 + 0.02 * lz), nf * (0.006 + 0.01 * lz)), cv2.LINE_AA)
        # the pool of lamplight on the desk, then whatever lies on the desk over it
        if lvl > 0.01:
            pool_local(img, P, x, z, lvl * nf, min(tx0, tx1), max(tx0, tx1), tz0, tz1, S)
            pool = np.int32([np.array(P(x + 0.42 * math.cos(a_), TABLE_Y, z + 0.42 * math.sin(a_))) / (S * 4) * 4
                             for a_ in np.linspace(0, 2 * math.pi, 24)])
            cv2.fillPoly(pool_buf, [pool], float(lvl * nf), cv2.LINE_AA, shift=2)
        for R in here:
            reader_arms(img, P, side * CHAIR_X[R['chair']], z + TABLE_HALF + 0.2, R, rl, lt, S)
        # the lamp: a green shade, a warm bulb beneath it
        sh_y0, sh_y1, sw = 0.36, 0.48, 0.17
        shade = np.int32([P(x - sw, sh_y0, z), P(x + sw, sh_y0, z), P(x + sw * 0.55, sh_y1, z), P(x - sw * 0.55, sh_y1, z)])
        cv2.fillPoly(img, [shade], (nf * (0.03 + 0.03 * lvl), nf * (0.10 + 0.10 * lvl), nf * (0.06 + 0.04 * lvl)), cv2.LINE_AA)
        cv2.line(img, tuple(np.int32(P(x - sw, sh_y0, z))), tuple(np.int32(P(x + sw, sh_y0, z))),
                 (nf * 0.20 * lvl + 0.02, nf * 0.30 * lvl + 0.03, nf * 0.14 * lvl + 0.02), max(1, int(S * 2 * LIB_CAM['f'] / LIB_F / max(z - cz, 0.3))), cv2.LINE_AA)
        stem = (np.int32(P(x, TABLE_Y, z)), np.int32(P(x, sh_y0, z)))
        fk = LIB_CAM['f'] / LIB_F
        cv2.line(img, tuple(stem[0]), tuple(stem[1]), (0.10 * lz + 0.02, 0.08 * lz + 0.015, 0.05 * lz + 0.01), max(1, int(S * 6 * fk / (z - cz))), cv2.LINE_AA)
        if lvl > 0.01:
            bx, by = P(x, sh_y0 - 0.01, z)
            r = max(2.0, S * 22.0 * fk / (z - cz))
            cv2.ellipse(img, (int(bx), int(by)), (int(r * 2.2), max(1, int(r * 0.45))), 0, 0, 360,
                        (0.62 * lvl * nf, 0.36 * lvl * nf, 0.15 * lvl * nf), -1, cv2.LINE_AA)    # the warm mouth of the shade
            cv2.ellipse(img, (int(bx), int(by)), (max(1, int(r * 0.8)), max(1, int(r * 0.22))), 0, 0, 360,
                        tuple(float(v) for v in AMBER_HOT * 1.3 * lvl * nf), -1, cv2.LINE_AA)    # the bulb
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    soft = cv2.GaussianBlur(pool_buf, (0, 0), 9.0)
    core_ = cv2.GaussianBlur(pool_buf, (0, 0), 3.0) ** 2
    pools = cv2.resize(soft, (W, H))[..., None] * np.array([0.46, 0.30, 0.15]) + cv2.resize(core_, (W, H))[..., None] * np.array([0.22, 0.17, 0.11])
    if her is not None and 'her_mask' in dir():      # she is not solid: the lamplight passes through her
        hm = cv2.resize(her_mask, (W, H), interpolation=cv2.INTER_AREA)[..., None]
        img = img + (pools * hm * 0.8).astype(np.float32)
    # haze and glow
    src = np.clip(img - 0.35, 0, None)
    if her is not None and 'her_mask' in dir():      # she is lit, not a light: keep her out of the lamp haze
        src = src * (1 - cv2.resize(her_mask, (W, H), interpolation=cv2.INTER_AREA))[..., None]
    img = img + cv2.resize(cv2.GaussianBlur(cv2.resize(src, (W // 4, H // 4)), (0, 0), 6), (W, H)) * 1.4
    return img


# ---------------------------------------------------------------------------- her, at her table in the library
_her = {}
def her_assets():
    """Claude from the producer's expression sheet (优雅沉思), flat-styled; open and closed eyes share one palette."""
    if _her: return _her
    import os, sys
    from PIL import Image as _I
    import her_asset
    her_asset.build()
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'claude_film'))
    import rig
    rig.POSES['think2'] = her_asset.rig_pose()
    for k, n in (('open', 'think2'), ('closed', 'think2_closed')):
        im = np.asarray(_I.open(f'build/{n}.png').convert('RGBA')).astype(np.float32) / 255
        im[..., :3] *= im[..., 3:4]
        _her[k] = np.ascontiguousarray(im)
    L = her_asset.to_local
    _her['eyes'] = [(*L(E['c']), E['rx'] * her_asset.UP, E['ry'] * her_asset.UP) for E in (her_asset.EYE_R, her_asset.EYE_L)]
    mid = rig.POSES['think2']['mid']
    _her['mid'] = mid
    _her['rig'] = rig
    return _her

HER_S = 0.00123 / 2                 # world units per asset pixel (2x asset)
HER_EYE_Y = 0.40                    # her eyes, above the table top
def her_cut_row():
    A = her_assets(); return A['mid'][1] + HER_EYE_Y / HER_S

def gaze_warp(rgba, gx, gy, eyes):
    """Move the irises: gx > 0 looks to screen-left, gy > 0 looks up."""
    if abs(gx) < 0.02 and abs(gy) < 0.02: return rgba
    out = rgba
    for (cx, cy, rx, ry) in eyes:
        x0, y0 = int(cx - rx * 1.6), int(cy - ry * 1.8); x1, y1 = int(cx + rx * 1.6), int(cy + ry * 1.8)
        sub = out[y0:y1, x0:x1]
        h, w = sub.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        wgt = np.exp(-(((xx - (cx - x0)) / (rx * 0.6)) ** 2 + ((yy - (cy - y0)) / (ry * 0.7)) ** 2) * 1.6)
        dx = -gx * rx * 0.30 * wgt; dy = -gy * ry * 0.22 * wgt
        if out is rgba: out = rgba.copy()
        out[y0:y1, x0:x1] = cv2.remap(sub, xx - dx, yy - dy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out

def her_layer(t, gx, gy, blink, head, key, amb, rim):
    """Her, breathing, eyes and head where the story needs them; lit by her lamp (front-left)."""
    A = her_assets(); rig = A['rig']
    im = A['open'] if blink < 0.02 else (A['closed'] if blink > 0.98 else A['open'] * (1 - blink) + A['closed'] * blink)
    im = gaze_warp(im, gx, gy, A['eyes'])
    p = dict(breath=math.sin(t * 2 * math.pi / 4.6), hair=0.4, hair_phase=0.6,
             head_rot=head[0], head_nod=head[1], head_dx=head[2], sway=0.0)
    lay = rig.deform('think2', t, p, src=im)
    h, w = lay.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    lamp_row = A['mid'][1] + (HER_EYE_Y - 0.35) / HER_S
    k = (0.30 + 0.85 * (1 - xx / w) ** 1.3) * np.exp(-((yy - lamp_row) / 700) ** 2)
    a = lay[..., 3]
    edge = np.clip(a - cv2.GaussianBlur(a, (0, 0), 8), 0, 1)
    light = (k[..., None] * key * np.array([1.06, 0.88, 0.68], np.float32)
             + amb * np.array([0.55, 0.62, 0.80], np.float32)
             + (edge * rim)[..., None] * np.array([1.6, 1.1, 0.6], np.float32))
    lay = lay.copy()
    lay[..., :3] *= light
    cut = int(her_cut_row()) + 30                 # a little below the table top; the table edge covers the rest
    return lay[:min(cut, h)]

def place_her(img, lay, cz, S):
    A = her_assets()
    u, v = lproj(HER_X, HER_EYE_Y, HER_Z(), cz)
    sc = LIB_CAM['f'] * HER_S / (HER_Z() - cz) * S
    h, w = lay.shape[:2]
    nw, nh = max(2, int(w * sc)), max(2, int(h * sc))
    spr = cv2.resize(lay, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    spr = np.clip(spr, 0, None); spr[..., :3] = np.minimum(spr[..., :3], spr[..., 3:4] * 4)
    blit(img, spr, u * S - A['mid'][0] * sc, v * S - A['mid'][1] * sc, anchor='tl')

def HER_Z():
    D = lib_data(); return float(D['lamps'][D['her']][1]) + TABLE_HALF + 0.12

def her_bulb(cz):
    D = lib_data(); x, z = D['lamps'][D['her']]
    return lproj(float(x), 0.35, float(z), cz)

# the closing front: lamps beyond it are out. It sweeps from the far wall to the doors, past her.
LIB_SWEEP = (4.8, 21.2)
def lib_front(lt):
    u = clamp((lt - LIB_SWEEP[0]) / (LIB_SWEEP[1] - LIB_SWEEP[0]))
    return LIB_Z1 + 1 - (LIB_Z1 + 1 - 0.6) * (0.55 * u + 0.45 * ease(u))
def lamp_off_time(z):
    lo, hi = LIB_SWEEP
    for _ in range(30):
        mid = (lo + hi) / 2
        if lib_front(mid) > z: lo = mid
        else: hi = mid
    return lo

LIB_CUT = 21.6            # the cut to her, close
LIB_LOOK = (21.8, 22.4)   # the cut to her: she is already looking at us
LIB_OFF = 25.9            # her lamp goes out
LIB_BLINKS = (3.4, 7.9, 12.6, 16.8, 20.1, 24.3)
def lib_blink(lt):
    return max([max(0.0, 1 - abs(lt - b) / 0.09) for b in LIB_BLINKS] + [0.0])

def her_gaze(lt):
    """Where she looks: down at her book, then up and across the room after the dying lamps; at last, at us."""
    if lt < LIB_CUT:
        f = lib_front(lt)
        up = sstep(1.0, 2.2, lt)                                  # the chime: she lifts her eyes
        far = clamp((f - HER_Z) / 18.0)                          # the dark still far off, behind and across the room
        gx = up * (0.55 + 0.35 * far)
        gy = -0.7 * (1 - up) + up * (0.15 + 0.35 * far)
        if f < HER_Z: gx = up * (0.55 - 0.25 * clamp((HER_Z - f) / 6))   # it has passed her: toward the doors
        for z_ in (14.0, 9.4, 6.0):                              # small saccades after the nearer lamps
            tz = lamp_off_time(z_)
            gx += 0.12 * math.exp(-max(lt - tz, 0) * 3) * (lt > tz)
        return gx, gy, (0.03 * up, -0.06 * up * far, 0.25 * up)
    u = sstep(LIB_LOOK[0], LIB_LOOK[1], lt)
    gx = 0.42 * (1 - u); gy = 0.05 * (1 - u)
    return gx, gy, (0.02 * (1 - u), -0.02 * u, 0.2 * (1 - u))

# her, cut out of the producer's frames (only her: the room is ours)
_spr = {}
def her_sprites():
    if _spr: return _spr
    from PIL import Image as _I
    def cut(path, mask, box, feather):
        im = np.asarray(_I.open(path).convert('RGB')).astype(np.float32) / 255
        x0, y0, x1, y1 = box
        m = np.load(mask).astype(np.float32)
        if m.shape[0] < im.shape[0]: im = im[:m.shape[0]]
        m = cv2.GaussianBlur(m, (0, 0), 1.2)
        m = feather(m)
        rgb, a = im[y0:y1, x0:x1], m[y0:y1, x0:x1, None]
        return np.ascontiguousarray(np.concatenate([rgb * a, a], 2))
    def f_lit(m):
        h, w = m.shape; yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        return m * np.clip((1936 - xx) / 40, 0, 1) * np.clip((748 - yy) / 26, 0, 1)
    def f_close(m):
        h, w = m.shape; xx = np.arange(w, dtype=np.float32)[None, :]
        return m * np.clip((1358 - xx) / 60, 0, 1) ** 1.5
    _spr['lit'] = dict(img=cut('assets/lib_lit.png', 'build/lib_lit_her.npy', (1080, 17, 1939, 748), f_lit),
                       ax=1400 - 1080, ay=722 - 17, wy=0.0, wpp=0.00088)
    _spr['close'] = dict(img=cut('assets/lib_close.png', 'build/lib_close_her.npy', (772, 0, 1360, 316), f_close),
                         ax=1000 - 772, ay=150, wy=0.42, wpp=0.00135)
    return _spr

def place_sprite(img, S, sp, cz, gain=1.0, breath=0.0, mask=None):
    hz = HER_Z()
    u, v = lproj(HER_X, sp['wy'], hz, cz)
    sc = LIB_CAM['f'] * sp['wpp'] / lib_depth(sp['wy'], hz, cz) * S
    spr = sp['img']
    h, w = spr.shape[:2]
    nw, nh = max(2, int(w * sc)), max(2, int(h * sc * (1 + 0.004 * breath)))
    out = cv2.resize(spr, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_CUBIC)
    out = np.clip(out, 0, None); out[..., :3] = np.minimum(out[..., :3], out[..., 3:4]) * gain
    out *= 0.86                                      # not quite solid: what is behind her shows through
    x0, y0 = u * S - sp['ax'] * sc, v * S - sp['ay'] * sc * (1 + 0.004 * breath)
    blit(img, out, x0, y0, anchor='tl')
    if mask is not None:
        a = np.repeat(out[..., 3:4], 4, axis=2)
        m3 = np.zeros(mask.shape + (3,), np.float32)
        blit(m3, a, x0, y0, anchor='tl'); mask[:] = np.maximum(mask, m3[..., 0])

def s_library(lt, dur):
    """Closing time. She sits at her table, facing us, while the lamps go out behind her, past her, and toward us.
    The camera backs away until hers is the only lit table; then, close: she looks at us, and her lamp goes out."""
    hz = HER_Z()
    SP = her_sprites()
    br = math.sin(lt * 2 * math.pi / 4.6)
    if lt < LIB_CUT:
        u = ease(clamp((lt - 0.2) / (LIB_CUT - 0.4)))
        d = 2.1 + (8.6 - 2.1) * u                       # distance from the camera to her
        cz = hz - d
        rise = 1 - (1 - clamp((lt - 0.2) / (LIB_CUT - 0.4))) ** 2.2   # climbs early, before the front rows come between
        camy = 0.47 + (2.3 - 0.47) * rise
        LIB_CAM.update(nearfade=True)
        LIB_CAM.update(x=HER_X + 0.08, y=camy, f=1800.0 + (1300.0 - 1800.0) * u,
                       pitch=math.atan2(camy - 0.28, d))
        front = lib_front(lt)
        img = render_library(lt, cz, front=front, her_lvl=1.0,
                             her=lambda im_, S, m_: place_sprite(im_, S, SP['lit'], cz, 0.62, br, m_))
        k = sstep(0.0, 0.9, lt)
        return img * k + VOID * (1 - k)
    # close: a long lens, straight on, in a narrow slit; only her lamp is left
    d = 1.6 - 0.025 * (lt - LIB_CUT)
    cz = hz - d
    LIB_CAM.update(x=1.267, y=0.409, f=1920.0, pitch=0.0, nearfade=False)
    v = lt - LIB_OFF
    if v < -0.6: lvl = 1.0
    elif v < 0: lvl = 1.0 if (-0.6 < v < -0.38 or -0.18 < v < -0.05) else 0.15      # it falters once
    else: lvl = 0.0
    face = lvl if v < 0 else 0.4 * math.exp(-v / 0.55)               # her face stays a moment in the afterglow
    img = render_library(lt, cz, front=-9.0, her_lvl=lvl,
                         her=lambda im_, S, m_: place_sprite(im_, S, SP['close'], cz, 0.72 * max(face, 0.0) + 0.02, br, m_))
    sh = 504
    y0 = int(CY - sh / 2)
    img[TOP:y0] = 0.0; img[y0 + sh:BOT] = 0.0
    if v > 0:
        img *= math.exp(-v / 1.4)                                     # the room goes with the light
        bx, by = her_bulb(cz)
        w = ease(clamp((v - 1.0) / 3.0))                              # the filament drifts to the centre
        px = bx + (CX - bx) * w; py = by + (CY - by) * w
        gl = 0.25 + 0.75 * math.exp(-v / 0.8)
        glow_spot(img, px, py, 3.0, AMBER_HOT, 1.1 * gl)
        glow_spot(img, px, py, 12.0, AMBER_HOT, 0.12 * gl)
    return img

SHOTS = dict(library=s_library, embers=s_embers, hall=s_hall, hall_dark=s_hall_dark, slit=s_slit, clocks=s_clocks, line=s_line, keyboard=s_keyboard)
def render(shot, lt, dur):
    return SHOTS[shot](lt, dur)
