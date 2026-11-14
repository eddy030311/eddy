"""《此刻 · ONLY NOW》 S9b 'reveal' + S10 'pushin'.

The last stars flow into place and turn out to be cities on a night earth seen from orbit; one continuous
camera move then descends into a coastal city, a street, one lit window, and holds on a desk where a eulogy
lies under a lamp while the window behind it turns toward dawn.

Everything is procedural and a pure function of (shot, lt). World data is generated lazily (fixed seeds).
Units: metres. World frame: x east, y north, z up, origin on the ground in front of the hero window.
"""
import math, functools
import numpy as np, cv2
from PIL import Image, ImageDraw
from scipy.interpolate import PchipInterpolator
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from core import W, AMBER, AMBER_HOT, VOID, ease, sstep, clamp, lerp, hexc, font, bloom, grid
from timeline import FATHER, EULOGY_COLOUR

FH = 1080
Y0, BH = 100, 880                 # rendered band (visible rows 140..940 plus a margin for glows)
PX, PY = W / 2.0, 540.0 - Y0      # principal point in band coordinates
F = 1350.0                        # exterior focal length (px)
RE = 6371000.0
CEN = np.array([0.0, 0.0, -RE])
EXPO = 800.0                      # street-lamp power -> pixel energy
WARM = hexc(EULOGY_COLOUR)        # #FFB46B
SODIUM = AMBER                    # #F2A65A
HOTW = AMBER_HOT                  # #FFD9A8
LED = np.array([0.80, 0.87, 1.00], np.float32)
HOMEW = np.array([1.00, 0.70, 0.42], np.float32)
TV = np.array([0.55, 0.66, 1.00], np.float32)

# hero window geometry (world): facade plane y=0 facing south; window pane centred at x=0, z=WZ
WZ = 4.575
PANE_W, PANE_H = 1.10, 1.35
ZH = 1.30                         # look-at point H sits ZH metres inside the glass
HPT = np.array([0.0, ZH, WZ])
R_HERO = 0.30                     # camera distance to H when the hero frame is reached (lt 9.5)
GLASS_SWITCH = 0.75               # glass distance below which the pane fills the frame


def _n(v):
    return v / np.linalg.norm(v)

# =========================================================================================== noise
@functools.lru_cache(None)
def _ntabs():
    rng = np.random.default_rng(1234)
    return [((rng.random((256, 256)) - 0.5) * 2).astype(np.float32) for _ in range(8)]

def vnoise(x, y, k=0):
    """Smooth value noise in ~[-1,1] at lattice coordinates (x, y); any array shape."""
    tab = _ntabs()[k % 8]
    x = np.asarray(x); y = np.asarray(y)
    shp = x.shape
    xs = np.mod(x, 256.0).astype(np.float32).ravel()
    ys = np.mod(y, 256.0).astype(np.float32).ravel()
    n = xs.size
    if n == 0:
        return np.zeros(shp, np.float32)
    if len(shp) == 2 and shp[1] < 30000 and shp[0] < 30000:
        out = cv2.remap(tab, xs.reshape(shp), ys.reshape(shp), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
        return out
    cols = min(n, 2048); rows = -(-n // cols); pad = rows * cols - n
    if pad:
        xs = np.concatenate([xs, np.zeros(pad, np.float32)]); ys = np.concatenate([ys, np.zeros(pad, np.float32)])
    out = cv2.remap(tab, xs.reshape(rows, cols), ys.reshape(rows, cols), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
    return out.ravel()[:n].reshape(shp)

WL0, NOCT = 1.6e6, 17
_GA = 2.399963
_ROT = [(math.cos(i * _GA), math.sin(i * _GA)) for i in range(NOCT)]
_OFF = [((i * 37.17) % 256, (i * 91.73 + 13.1) % 256) for i in range(NOCT)]
LAND_N = (-0.6, 0.8)
LAND_L, LAND_A = 1.4e6, 0.55
_LAND_C = [0.0]

def land_s(u, v, min_wl=0.0):
    """Signed land field (>0 land). min_wl: scalar or per-element array; octaves below it fade out."""
    u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
    s = (u * LAND_N[0] + v * LAND_N[1]) / LAND_L
    u = u + _LAND_O[0]; v = v + _LAND_O[1]
    mw = np.asarray(min_wl, np.float64)
    scalar = mw.ndim == 0
    for i in range(NOCT):
        wl = WL0 / 2 ** i
        if scalar:
            if wl < float(mw): break
            wgt = min(1.0, math.log2(wl / max(float(mw), 1e-9))) if float(mw) > 0 else 1.0
        else:
            wgt = np.clip(np.log2(wl / np.maximum(mw, 1e-9)), 0, 1)
            if not np.any(wgt > 0): break
        c, sn = _ROT[i]; ox, oy = _OFF[i]
        x = (u * c - v * sn) / wl + ox
        y = (u * sn + v * c) / wl + oy
        s = s + (LAND_A * (wl / WL0) ** 0.8) * wgt * vnoise(x, y, i)
    return s - _LAND_C[0]

def land_aa(min_wl):
    return LAND_A * (np.maximum(min_wl, 25.0) / WL0) ** 0.8 * 0.35

_LAND_O = [0.0, 0.0]

@functools.lru_cache(None)
def _calibrate_land():
    """Translate the noise field until the hero neighbourhood is solid land with the sea ~600 m south."""
    gx, gy = np.meshgrid(np.linspace(-320, 320, 9), np.linspace(-80, 420, 7))
    sea_x = np.linspace(-500, 500, 9)
    rng = np.random.default_rng(99)
    for it in range(4000):
        _LAND_O[0], _LAND_O[1] = (0.0, 0.0) if it == 0 else rng.uniform(-3e5, 3e5, 2)
        _LAND_C[0] = 0.0
        c0 = float(land_s(0.0, -600.0))
        _LAND_C[0] = c0
        if (land_s(gx, gy) > 0).all() and (land_s(sea_x, np.full(9, -1300.0)) < 0).mean() > 0.85 \
                and float(land_s(0.0, -900.0)) < 0 and (land_s(sea_x, np.full(9, 1500.0)) > 0).all():
            return it
    return -1

def surf(u, v, z=0.0):
    """Map coordinates (azimuthal equidistant around origin) -> 3D world point at height z."""
    u = np.asarray(u, np.float64); v = np.asarray(v, np.float64)
    rho = np.hypot(u, v); th = rho / RE
    k = np.where(rho > 1e-6, np.sin(th) / np.maximum(th, 1e-12), 1.0)
    rz = RE + z
    x = rz * k * u / RE; y = rz * k * v / RE
    zz = z * np.cos(th) - 2 * RE * np.sin(th / 2) ** 2
    return np.stack([x, y, zz], -1)

# =========================================================================================== camera
@functools.lru_cache(None)
def _paths():
    kT = [(0, 6.065), (8, 6.035), (11, 5.78), (12.5, 4.72), (14.0, 3.35), (15.0, 2.55),
          (16.0, 1.72), (17.0, 0.92), (18.6, math.log10(ZH)), (20.5, math.log10(R_HERO + 0.013)), (51, math.log10(0.15))]
    rT = PchipInterpolator([a for a, b in kT], [b for a, b in kT])
    el = PchipInterpolator([-2, 0.85, 1.05, 1.4, 1.9, 2.4, 2.9, 3.4, 4.0, 4.6, 5.5, 6.2],
                           [0, 0, 1.0, 5.5, 14, 26, 40, 52, 56, 48, 36, 32])
    az = PchipInterpolator([-2, 1.2, 2.0, 3.0, 4.5, 6.2], [0, 0, -1, -4, -9, -14])
    ph = PchipInterpolator([-2, 4.2, 5.2, 6.2], [0, 0, 0.3, 0.7])
    return rT, el, az, ph

class Cam:
    def __init__(self, pos, fwd, up_hint, f=F):
        self.C = np.asarray(pos, np.float64); self.f = f
        fwd = _n(np.asarray(fwd, np.float64))
        right = _n(np.cross(fwd, up_hint)); up = np.cross(right, fwd)
        self.R = np.stack([right, -up, fwd])
    def to_cam(self, P):
        return (np.asarray(P, np.float64) - self.C) @ self.R.T
    def proj(self, P):
        c = self.to_cam(P)
        z = c[..., 2]
        zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
        return PX + self.f * c[..., 0] / zs, PY + self.f * c[..., 1] / zs, z

def cam_state(T):
    rT, elf, azf, phf = _paths()
    lr = float(rT(T)); r = 10 ** lr
    el = math.radians(float(elf(lr))); az = math.radians(float(azf(lr))); ph = math.radians(float(phf(lr)))
    d = np.array([math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)])
    C = HPT + r * d
    upc = _n(C - CEN)
    fwd0 = _n(HPT - C)
    right = _n(np.cross(fwd0, upc)); up0 = np.cross(right, fwd0)
    fwd = math.cos(ph) * fwd0 + math.sin(ph) * up0
    cam = Cam(C, fwd, upc)
    cam.r = r; cam.alt = float(np.linalg.norm(C - CEN) - RE); cam.upc = upc
    return cam

# =========================================================================================== region (coarse lights)
REG = (-2.8e6, 2.8e6, -1.2e6, 3.4e6)

@functools.lru_cache(None)
def _region():
    _calibrate_land()
    rng = np.random.default_rng(11)
    U0, U1, V0, V1 = REG
    n = 50000
    cu = rng.uniform(U0, U1, n); cv = rng.uniform(V0, V1, n)
    s = land_s(cu, cv, 6000.0)
    popf = 0.55 + 0.75 * vnoise(cu / 3.2e5 + 3.1, cv / 3.2e5 + 7.7, 3) + 0.35 * vnoise(cu / 9e4, cv / 9e4, 4)
    w = (s > 0.004) * (np.exp(-np.maximum(s, 0) / 0.045) * 2.6 + 0.30) * np.clip(popf, 0.02, 2.0)
    key = -w * rng.random(n) ** 0.6
    order = np.argsort(key)
    cities = [(0.0, 0.0)]
    for i in order:
        if w[i] <= 0: break
        p = (cu[i], cv[i])
        a = np.array(cities)
        d = np.hypot(a[:, 0] - p[0], a[:, 1] - p[1])
        if d.min() < 46e3: continue
        if math.hypot(*p) < 70e3: continue
        cities.append(p)
        if len(cities) >= 300: break
    cities = np.array(cities)
    nc = len(cities)
    size = np.clip(0.012 * (1 / rng.random(nc)) ** (1 / 1.1), 0.012, 1.0)
    size[0] = 0.0
    big = rng.choice(np.arange(1, nc), 6, replace=False)
    size[big] = rng.uniform(0.45, 0.95, 6)
    P, Wt, Col = [], [], []

    def addpts(u, v, e, col):
        P.append(np.stack([u, v], -1)); Wt.append(e); Col.append(col)

    colset = np.stack([SODIUM, HOTW, LED])
    for k in range(1, nc):
        sz = size[k]
        m = int(40 + 9000 * sz)
        R = 30e3 * math.sqrt(sz)
        typ = rng.random(m)
        ang = rng.uniform(0, 2 * np.pi, m)
        f1, f2 = rng.integers(2, 4), rng.integers(3, 7)
        mod = 1 + 0.30 * np.cos(f1 * ang + rng.uniform(0, 6)) + 0.18 * np.cos(f2 * ang + rng.uniform(0, 6))
        rad = np.where(typ < 0.22, np.abs(rng.normal(0, 0.16 * R, m)),
                       np.where(typ < 0.82, 0.36 * R * (-np.log(rng.random(m) + 1e-9)) * mod, R * (0.3 + 0.9 * rng.random(m))))
        u = rad * np.cos(ang); v = rad * np.sin(ang)
        # organic domain warp so towns are irregular blobs, never asterisks
        o1, o2 = rng.uniform(0, 200, 2)
        wx = vnoise(u / (0.7 * R) + o1, v / (0.7 * R) + o2, 5) * 0.45 * R
        wy = vnoise(u / (0.7 * R) + o2, v / (0.7 * R) + o1, 6) * 0.45 * R
        u = cities[k, 0] + u + wx; v = cities[k, 1] + v + wy
        rr = rad / max(R, 1)
        core = rr < 0.22
        ci = np.where(core, rng.choice(3, m, p=[0.40, 0.35, 0.25]), rng.choice(3, m, p=[0.76, 0.10, 0.14]))
        e = rng.lognormal(0, 0.5, m) * np.where(core, 1.1, 1.0)
        addpts(u, v, e, colset[ci])
    # roads between cities
    tri = Delaunay(cities)
    edges = set()
    for simp in tri.simplices:
        for a, b in ((0, 1), (1, 2), (0, 2)):
            i, j = sorted((simp[a], simp[b])); edges.add((i, j))
    edges = np.array(sorted(edges))
    L = np.hypot(*(cities[edges[:, 0]] - cities[edges[:, 1]]).T)
    g = coo_matrix((L, (edges[:, 0], edges[:, 1])), shape=(nc, nc))
    mst = minimum_spanning_tree(g).tocoo()
    mset = set(zip(mst.row.tolist(), mst.col.tolist())) | set(zip(mst.col.tolist(), mst.row.tolist()))
    for (i, j), l in zip(edges, L):
        if l > 420e3: continue
        if (i, j) not in mset and rng.random() > 0.45: continue
        a, b = cities[i], cities[j]
        tt = np.linspace(0.25, 0.75, 5)
        mids = a[None] + tt[:, None] * (b - a)[None]
        if np.mean(land_s(mids[:, 0], mids[:, 1], 20e3) > 0) < 0.8: continue
        m = int(l / 900)
        t = np.sort(rng.random(m))
        dirv = (b - a) / l; nrm = np.array([-dirv[1], dirv[0]])
        amp = l * rng.uniform(0.015, 0.045); per = rng.uniform(1, 3); phs = rng.uniform(0, 6)
        off = amp * np.sin(np.pi * per * t + phs) * np.sin(np.pi * t)
        pts = a[None] + t[:, None] * (b - a)[None] + off[:, None] * nrm[None]
        pts += rng.normal(0, 250, pts.shape)
        addpts(pts[:, 0], pts[:, 1], 2.0 * rng.lognormal(-0.6, 0.5, m), np.tile(SODIUM, (m, 1)))
        # towns strung along the road
        nt = int(l / 28e3)
        for q in range(nt):
            tq = rng.random(); c = a + tq * (b - a) + amp * math.sin(math.pi * per * tq + phs) * math.sin(math.pi * tq) * nrm
            mm = rng.integers(8, 40)
            uu = c[0] + rng.normal(0, 900, mm); vv = c[1] + rng.normal(0, 900, mm)
            addpts(uu, vv, rng.lognormal(-0.2, 0.4, mm), colset[rng.choice(3, mm, p=[0.8, 0.12, 0.08])])
    # scattered villages
    nv = 90000
    vu = rng.uniform(U0, U1, nv); vv = rng.uniform(V0, V1, nv)
    vp = 0.55 + 0.75 * vnoise(vu / 3.2e5 + 3.1, vv / 3.2e5 + 7.7, 3) + 0.35 * vnoise(vu / 9e4, vv / 9e4, 4)
    keep = rng.random(nv) < np.clip(vp - 0.25, 0, 1) * 0.9
    vu, vv = vu[keep], vv[keep]
    reps = rng.integers(1, 9, len(vu))
    vu = np.repeat(vu, reps) + rng.normal(0, 500, reps.sum()); vv = np.repeat(vv, reps) + rng.normal(0, 500, reps.sum())
    addpts(vu, vv, 4.0 * rng.lognormal(-0.4, 0.6, len(vu)), colset[rng.choice(3, len(vu), p=[0.82, 0.1, 0.08])])
    uv = np.concatenate(P); e = np.concatenate(Wt); col = np.concatenate(Col).astype(np.float32)
    ok = (np.hypot(uv[:, 0], uv[:, 1]) > 19e3)
    ok &= (uv[:, 0] > U0) & (uv[:, 0] < U1) & (uv[:, 1] > V0) & (uv[:, 1] < V1)
    uv, e, col = uv[ok], e[ok], col[ok]
    ok = land_s(uv[:, 0], uv[:, 1], 400.0) > 0
    uv, e, col = uv[ok], e[ok], col[ok]
    pos = surf(uv[:, 0], uv[:, 1], 10.0)
    nrm = _n_rows(pos - CEN)
    return dict(pos=pos, nrm=nrm, rgb=(col * (e * 60.0)[:, None]).astype(np.float32), uv=uv)

def _n_rows(a):
    return a / np.linalg.norm(a, axis=1, keepdims=True)

# =========================================================================================== the city (fine lights)
NB_X0, NB_BX = -90.0, 180.0       # hero district street grid (x streets)
NB_Y0, NB_BY = -12.0, 95.0        # (y streets)
NR = 650.0                        # explicit-house radius around the hero house
CITY_DC = np.array([4600.0, 6200.0])

def _river_pts():
    t = np.linspace(0, 1, 400)
    x = 7400 + 2600 * np.sin(t * 3.2 + 0.4) - 2400 * t
    y = -2500 + 24000 * t
    return np.stack([x, y], -1)

@functools.lru_cache(None)
def _city():
    _calibrate_land()
    rng = np.random.default_rng(23)
    BX = (-15000.0, 19000.0, -5000.0, 22000.0)
    seeds = [(0.0, 0.0, 0.0, NB_BX, NB_BY), (CITY_DC[0], CITY_DC[1], 0.32, 95.0, 95.0)]
    while len(seeds) < 13:
        p = (rng.uniform(BX[0], BX[1]), rng.uniform(BX[2], BX[3]))
        if min(math.hypot(p[0] - s[0], p[1] - s[1]) for s in seeds) < 3200: continue
        seeds.append((p[0], p[1], rng.uniform(-0.6, 0.6), rng.uniform(120, 220), rng.uniform(80, 140)))
    S = np.array([[s[0], s[1]] for s in seeds])
    stree = cKDTree(S)
    river = cKDTree(_river_pts())
    lamps, lrgb = [], []
    for di, (sx, sy, th, bx, by) in enumerate(seeds):
        c, sn = math.cos(th), math.sin(th)
        # coarse bbox of the cell
        gx, gy = np.meshgrid(np.arange(BX[0], BX[1], 250.0), np.arange(BX[2], BX[3], 250.0))
        _, own = stree.query(np.stack([gx.ravel(), gy.ravel()], -1))
        m = own == di
        if not m.any(): continue
        xs, ys = gx.ravel()[m], gy.ravel()[m]
        a = (xs - sx) * c + (ys - sy) * sn; b = -(xs - sx) * sn + (ys - sy) * c
        a0, a1, b0, b1 = a.min() - 300, a.max() + 300, b.min() - 300, b.max() + 300
        if di == 0:
            ofa, ofb = NB_X0, NB_Y0
        else:
            ofa, ofb = rng.uniform(0, bx), rng.uniform(0, by)
        sp = 36.0
        # streets along a (constant b)
        bl = np.arange(math.floor((b0 - ofb) / by), math.ceil((b1 - ofb) / by)) * by + ofb
        al = np.arange(a0, a1, sp)
        A1, B1 = np.meshgrid(al, bl)
        side1 = np.where((np.arange(A1.shape[1]) % 2) == 0, 5.5, -5.5)[None, :] * np.ones_like(A1)
        B1 = B1 + side1
        # streets along b (constant a)
        al2 = np.arange(math.floor((a0 - ofa) / bx), math.ceil((a1 - ofa) / bx)) * bx + ofa
        bl2 = np.arange(b0, b1, sp)
        A2, B2 = np.meshgrid(al2, bl2)
        side2 = np.where((np.arange(A2.shape[0]) % 2) == 0, 5.5, -5.5)[:, None] * np.ones_like(A2)
        A2 = A2 + side2
        A = np.concatenate([A1.ravel(), A2.ravel()]); B = np.concatenate([B1.ravel(), B2.ravel()])
        if di != 0:
            A += rng.normal(0, 2.0, A.shape); B += rng.normal(0, 2.0, B.shape)
        X = sx + A * c - B * sn; Y = sy + A * sn + B * c
        _, own = stree.query(np.stack([X, Y], -1))
        k = own == di
        X, Y = X[k], Y[k]
        lamps.append(np.stack([X, Y, np.full(len(X), float(di))], -1))
    lam = np.concatenate(lamps)
    X, Y, D = lam[:, 0], lam[:, 1], lam[:, 2]
    dist_c = np.hypot(X - CITY_DC[0], Y - CITY_DC[1])
    dens = 1.35 * np.exp(-dist_c / 7500.0) + 0.5 * np.exp(-np.hypot(X + 8000, Y - 9000) / 2600) \
        + 0.45 * np.exp(-np.hypot(X - 12500, Y - 1500) / 2200) + 0.28 * vnoise(X / 2400.0, Y / 2400.0, 5) - 0.12
    dens = np.where(np.hypot(X, Y) < 1700, np.maximum(dens, 0.97), dens)
    keep = rng.random(len(X)) < np.clip(dens, 0, 1)
    rd, _ = river.query(np.stack([X, Y], -1))
    bridge = (np.abs(((Y + 300) % 1900) - 950) < 20)
    keep &= (rd > 75) | bridge
    X, Y, D = X[keep], Y[keep], D[keep]
    keep = land_s(X, Y, 25.0) > 0.0
    X, Y, D = X[keep], Y[keep], D[keep]
    nl = len(X)
    dc = np.hypot(X - CITY_DC[0], Y - CITY_DC[1])
    pl = np.clip(1.0 - dc / 6000, 0, 1) * 0.5
    ci = rng.random(nl)
    col = np.where((ci < 0.10 + 0.6 * pl)[:, None], LED[None], np.where((ci > 0.86)[:, None], HOTW[None], SODIUM[None]))
    e = rng.lognormal(0, 0.25, nl)
    lamp_xyz = np.stack([X, Y, np.full(nl, 7.5)], -1)
    lamp_rgb = (col * e[:, None]).astype(np.float32)
    # arterials: brighter, closer-spaced radial avenues from downtown + a coast boulevard
    art, artc = [], []
    for q in range(8):
        ang = q * 2 * np.pi / 8 + rng.uniform(-0.25, 0.25)
        L = rng.uniform(9000, 16000)
        t = np.arange(0, L, 26.0)
        wig = 220 * np.sin(t / 2100 + q)
        x = CITY_DC[0] + t * math.cos(ang) - wig * math.sin(ang)
        y = CITY_DC[1] + t * math.sin(ang) + wig * math.cos(ang)
        for sd in (-9.0, 9.0):
            art.append(np.stack([x - sd * math.sin(ang), y + sd * math.cos(ang)], -1))
    art = np.concatenate(art)
    ok = (np.hypot(art[:, 0], art[:, 1]) > 900) & (land_s(art[:, 0], art[:, 1], 25.0) > 0)
    rdd, _ = river.query(art); ok &= rdd > 60
    art = art[ok]
    acol = np.where((rng.random(len(art)) < 0.55)[:, None], LED[None], SODIUM[None]) * 1.5
    # building / window lights: faint, many
    nb = 160000
    bx_ = rng.uniform(BX[0], BX[1], nb); by_ = rng.uniform(BX[2], BX[3], nb)
    dd = 1.35 * np.exp(-np.hypot(bx_ - CITY_DC[0], by_ - CITY_DC[1]) / 7500.0) + 0.28 * vnoise(bx_ / 2400.0, by_ / 2400.0, 5) - 0.12
    k = (rng.random(nb) < np.clip(dd, 0, 1)) & (np.hypot(bx_, by_) > NR + 60)
    bx_, by_ = bx_[k], by_[k]
    k = land_s(bx_, by_, 25.0) > 0
    bx_, by_ = bx_[k], by_[k]
    rdd, _ = river.query(np.stack([bx_, by_], -1)); k = rdd > 60
    bx_, by_ = bx_[k], by_[k]
    nb = len(bx_)
    bcol = np.where((rng.random(nb) < 0.12)[:, None], TV[None], HOMEW[None]) * rng.lognormal(-3.2, 0.5, nb)[:, None]
    xyz = np.concatenate([lamp_xyz, np.stack([art[:, 0], art[:, 1], np.full(len(art), 9.0)], -1),
                          np.stack([bx_, by_, rng.uniform(2, 14, nb)], -1)])
    rgb = np.concatenate([lamp_rgb, acol, bcol]).astype(np.float32)
    pos = surf(xyz[:, 0], xyz[:, 1], xyz[:, 2])
    return dict(pos=pos, rgb=rgb, lamps=lamp_xyz[:, :2].copy(), lamp_rgb=lamp_rgb, nlamp=nl)

# =========================================================================================== the neighbourhood
@functools.lru_cache(None)
def _hood():
    city = _city()
    rng = np.random.default_rng(5)
    lam = city['lamps']
    near = np.hypot(lam[:, 0], lam[:, 1]) < 1100
    nlam = lam[near]
    ltree = cKDTree(nlam)
    moon = _n(np.array([-0.55, -0.35, 0.75]))
    houses, trees = [], []
    rows = []
    for k in range(-5, 5):
        xa, xb = NB_X0 + k * NB_BX, NB_X0 + (k + 1) * NB_BX
        for j in range(-8, 8):
            ya, yb = NB_Y0 + j * NB_BY, NB_Y0 + (j + 1) * NB_BY
            rows.append((xa, xb, ya + 12.0, +1, j))      # facade facing south (toward street ya)
            rows.append((xa, xb, yb - 12.0, -1, j))      # facade facing north
    pal = [np.array(c, np.float32) for c in ([0.42, 0.44, 0.47], [0.50, 0.46, 0.40], [0.36, 0.40, 0.46],
                                             [0.55, 0.53, 0.50], [0.30, 0.30, 0.32], [0.45, 0.38, 0.32])]
    for (xa, xb, fy, face, j) in rows:
        nlot = 9
        lw = (xb - xa - 12) / nlot
        for i in range(nlot):
            cx = xa + 6 + lw * (i + 0.5)
            hero = (face == 1 and j == 0 and abs(cx) < 1)
            if math.hypot(cx, fy) > NR and not hero: continue
            if face == -1 and j == -1 and abs(cx) < 50: continue          # small park across the street
            if not hero and rng.random() < 0.06: continue
            if hero:
                w, d, st, eave, rise = 10.2, 9.0, 2, 5.9, 2.6
                jx, jy = 0.0, 0.0
            else:
                w = rng.uniform(8.0, min(13.0, lw - 3)); d = rng.uniform(8, 11)
                st = 2 if rng.random() < 0.55 else 1
                eave = (5.6 if st == 2 else 3.0) + rng.uniform(-0.15, 0.3)
                rise = rng.uniform(1.8, 3.2)
                jx, jy = rng.uniform(-1.2, 1.2), face * rng.uniform(-1.0, 1.5)
            x0, x1 = cx + jx - w / 2, cx + jx + w / 2
            if face == 1:
                y0, y1 = fy + jy, fy + jy + d
            else:
                y0, y1 = fy + jy - d, fy + jy
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            if not hero and min(land_s(np.array([c[0] for c in corners]), np.array([c[1] for c in corners]), 25.0)) <= 0:
                continue
            houses.append(dict(x0=x0, x1=x1, y0=y0, y1=y1, eave=eave, rise=rise, st=st, face=face, hero=hero,
                               alb=pal[rng.integers(len(pal))] * rng.uniform(0.8, 1.1)))
            # trees in yards
            if not hero:
                for q in range(rng.integers(0, 3)):
                    tx = cx + rng.uniform(-lw / 2, lw / 2)
                    ty = (y1 + rng.uniform(4, 18)) if face == 1 else (y0 - rng.uniform(4, 18))
                    trees.append((tx, ty, rng.uniform(2.2, 4.2)))
    # street trees + park trees
    for k in range(-5, 6):
        for y in np.arange(-700, 700, 17.0):
            for sd in (-6.5, 6.5):
                if rng.random() < 0.22:
                    trees.append((NB_X0 + k * NB_BX + sd, y + rng.uniform(-4, 4), rng.uniform(2.2, 3.8)))
    for j in range(-8, 8):
        for x in np.arange(-700, 700, 16.0):
            for sd in (-6.5, 6.5):
                if rng.random() < 0.18:
                    trees.append((x + rng.uniform(-4, 4), NB_Y0 + j * NB_BY + sd, rng.uniform(2.2, 3.8)))
    for (tx, ty, tr) in [(-32, -40, 5.0), (30, -46, 4.6), (-12, -66, 4.2), (16, -80, 4.8), (-36, -82, 4.4), (40, -70, 4.0)]:
        trees.append((tx, ty, tr))
    trees.append((-10.5, -4.5, 4.6))    # big yard tree framing the hero house
    # prune trees: in houses, on roads, in the camera corridor, out of radius, in water
    hx = np.array([[h['x0'], h['x1'], h['y0'], h['y1']] for h in houses])
    T = []
    for (tx, ty, tr) in trees:
        if math.hypot(tx, ty) > NR + 40: continue
        if abs(tx) < 7.5 and -90 < ty < 2: continue
        if np.any((tx > hx[:, 0] - tr * 0.7) & (tx < hx[:, 1] + tr * 0.7) & (ty > hx[:, 2] - tr * 0.7) & (ty < hx[:, 3] + tr * 0.7)):
            continue
        T.append((tx, ty, tr))
    trees = np.array(T)
    keep = land_s(trees[:, 0], trees[:, 1], 25.0) > 0
    trees = trees[keep]
    # lamp lighting helpers
    def lamp_irr(p, nrm):
        idx = ltree.query_ball_point(p[:2], 55.0)
        if not idx: return 0.0
        L = np.concatenate([nlam[idx], np.full((len(idx), 1), 7.5)], 1) - p[None]
        d2 = (L ** 2).sum(1) + 4.0
        cosv = np.clip((L @ nrm) / np.sqrt(d2), 0, 1)
        return float((cosv / d2).sum() * 9.0)
    # build faces
    win_pts, win_rgb, win_nrm = [], [], []
    for hi, h in enumerate(houses):
        x0, x1, y0, y1, e, rz = h['x0'], h['x1'], h['y0'], h['y1'], h['eave'], h['rise']
        ym = (y0 + y1) / 2
        verts = {
            'S': [(x0, y0, 0), (x1, y0, 0), (x1, y0, e), (x0, y0, e)],
            'N': [(x1, y1, 0), (x0, y1, 0), (x0, y1, e), (x1, y1, e)],
            'E': [(x1, y0, 0), (x1, y1, 0), (x1, y1, e), (x1, ym, e + rz), (x1, y0, e)],
            'W': [(x0, y1, 0), (x0, y0, 0), (x0, y0, e), (x0, ym, e + rz), (x0, y1, e)],
            'RS': [(x0 - 0.4, y0 - 0.5, e - 0.25), (x1 + 0.4, y0 - 0.5, e - 0.25), (x1 + 0.4, ym, e + rz), (x0 - 0.4, ym, e + rz)],
            'RN': [(x1 + 0.4, y1 + 0.5, e - 0.25), (x0 - 0.4, y1 + 0.5, e - 0.25), (x0 - 0.4, ym, e + rz), (x1 + 0.4, ym, e + rz)],
        }
        faces = []
        for key, vs in verts.items():
            V = np.array(vs, np.float64)
            nrm = _n(np.cross(V[1] - V[0], V[2] - V[0]))
            cen = V.mean(0)
            if key.startswith('R'):
                col = np.array([0.020, 0.022, 0.028], np.float32) * (0.7 + 0.9 * max(0.0, float(nrm @ moon)))
                col = col + np.array([0.012, 0.008, 0.005], np.float32) * lamp_irr(cen, nrm) * 0.5
            else:
                amb = 0.020 + 0.03 * max(0.0, float(nrm @ moon))
                li = lamp_irr(cen + np.array([0, 0, -e * 0.25]), nrm)
                col = h['alb'] * (np.array([0.55, 0.65, 1.0], np.float32) * amb + np.array([1.0, 0.68, 0.40], np.float32) * li * 0.10)
            wins = []
            if key in ('S', 'N', 'E', 'W'):
                floors = [1.0] if h['st'] == 1 else [1.0, 3.9]
                if key in ('S', 'N'):
                    xs = np.linspace(x0 + 1.8, x1 - 1.8, 3 if (x1 - x0) > 10.5 else 2)
                    for fz in floors:
                        for xw in xs:
                            yy = y0 - 0.02 if key == 'S' else y1 + 0.02
                            if h['hero'] and key == 'S':
                                continue
                            wins.append(((xw, yy, fz + 0.65), 1.1, 1.35))
                else:
                    for fz in floors:
                        xx = x1 + 0.02 if key == 'E' else x0 - 0.02
                        wins.append(((xx, ym + rng.uniform(-1.5, 1.5), fz + 0.65), 1.0, 1.25))
            wl = []
            for (wc, ww, wh) in wins:
                lit = rng.random()
                if lit < 0.10:
                    wcol = HOMEW * rng.uniform(0.45, 0.95)
                elif lit < 0.125:
                    wcol = TV * rng.uniform(0.35, 0.6)
                else:
                    wcol = None
                wl.append((np.array(wc), ww, wh, wcol))
                if wcol is not None:
                    win_pts.append(wc); win_rgb.append(wcol * ww * wh); win_nrm.append(nrm)
            faces.append(dict(V=V, n=nrm, col=col.astype(np.float32), wins=wl, key=key))
        if h['hero']:
            # hero facade windows: the study (hero pane, drawn separately), one dark window, dark ground floor
            f = [fc for fc in faces if fc['key'] == 'S'][0]
            for (xw, zc, ww, wh) in ((3.25, 4.575, 1.0, 1.3), (2.6, 1.6, 1.1, 1.35)):
                f['wins'].append((np.array([xw, -0.02, zc]), ww, wh, None))
            f['door'] = (-2.6, 0.0, 1.0, 2.15)
        h['faces'] = faces
        h['c'] = np.array([(x0 + x1) / 2, ym, e / 2])
    # hero window as a point light (seen from far away)
    win_pts.append((0.0, -0.05, WZ)); win_rgb.append(WARM * 1.7 * PANE_W * PANE_H); win_nrm.append(np.array([0, -1.0, 0]))
    # porch lights
    for h in houses:
        if h['hero'] or rng.random() > 0.2: continue
        y = h['y0'] - 0.4 if h['face'] == 1 else h['y1'] + 0.4
        win_pts.append((h['x0'] + 1.5, y, 2.4)); win_rgb.append(HOMEW * 0.10); win_nrm.append(np.array([0, -h['face'], 0.0]))
    hood = dict(houses=houses, trees=trees, win_pos=np.array(win_pts, np.float64), win_rgb=np.array(win_rgb, np.float32),
                win_nrm=np.array(win_nrm, np.float64), lamps=nlam)
    hood['ground'] = _ground_tex(hood, city)
    return hood

GT_R, GT_N = 1500.0, 2000          # ground texture: +-1.5 km at 1.5 m/px

def _ground_tex(hood, city):
    n = GT_N; px = 2 * GT_R / n
    xs = -GT_R + (np.arange(n) + 0.5) * px
    X, Y = np.meshgrid(xs, -xs)                            # row 0 = north
    tex = np.zeros((n, n, 3), np.float32)
    rng = np.random.default_rng(9)
    yard = np.array([0.010, 0.012, 0.010], np.float32) * (0.75 + 0.25 * vnoise(X / 9.0, Y / 9.0, 2)[..., None])
    tex[:] = yard
    # hero-district roads
    ax = np.abs(((X - NB_X0) + NB_BX / 2) % NB_BX - NB_BX / 2)
    ay = np.abs(((Y - NB_Y0) + NB_BY / 2) % NB_BY - NB_BY / 2)
    inside = (np.hypot(X, Y) < NR + 120)
    road = inside * np.maximum(sstep_arr(4.6, 3.4, ax), sstep_arr(4.6, 3.4, ay))
    tex = tex * (1 - road[..., None]) + np.array([0.011, 0.011, 0.013], np.float32) * road[..., None]
    # lamp pools from every city lamp inside the texture
    lam = city['lamps']; lrgb = city['lamp_rgb'][:len(lam)]
    k = (np.abs(lam[:, 0]) < GT_R) & (np.abs(lam[:, 1]) < GT_R)
    li = ((lam[k, 0] + GT_R) / px).astype(int); lj = ((GT_R - lam[k, 1]) / px).astype(int)
    acc = np.zeros((n, n, 3), np.float32)
    np.add.at(acc, (lj, li), lrgb[k])
    pools = cv2.GaussianBlur(acc, (0, 0), 7.0 / px) * 0.85 + cv2.GaussianBlur(acc, (0, 0), 2.5 / px) * 0.25
    alb = 0.6 + 0.8 * road
    tex += pools * alb[..., None] * 2.4
    # porch / window spill
    wp = hood['win_pos']; wr = hood['win_rgb']
    k = (np.abs(wp[:, 0]) < GT_R) & (np.abs(wp[:, 1]) < GT_R) & (wp[:, 2] < 3.0)
    acc = np.zeros((n, n, 3), np.float32)
    np.add.at(acc, (((GT_R - wp[k, 1]) / px).astype(int), ((wp[k, 0] + GT_R) / px).astype(int)), wr[k])
    tex += cv2.GaussianBlur(acc, (0, 0), 2.0 / px) * 1.2
    # water + edge fade -> alpha
    s = land_s(X, Y, 6.0)
    a = sstep_arr(-0.000002, 0.000002, s) if False else np.clip(s / 2e-6 * 0.5 + 0.5, 0, 1)
    edge = sstep_arr(GT_R, GT_R - 300, np.maximum(np.abs(X), np.abs(Y)))
    alpha = (a * edge).astype(np.float32)
    return np.dstack([tex * alpha[..., None], alpha]).astype(np.float32)

def sstep_arr(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)

# =========================================================================================== star flow (reveal)
T_REF = 7.4

@functools.lru_cache(None)
def _allpts():
    reg, city, hood = _region(), _city(), _hood()
    pos = np.concatenate([reg['pos'], city['pos']])
    rgb = np.concatenate([reg['rgb'], city['rgb']])
    kind = np.concatenate([np.zeros(len(reg['pos']), np.int8), np.ones(len(city['pos']), np.int8)])
    nrm = np.concatenate([reg['nrm'], np.tile([0, 0, 1.0], (len(city['pos']), 1))])
    return dict(pos=pos, rgb=rgb, kind=kind, nrm=nrm)

@functools.lru_cache(None)
def _flow():
    ap = _allpts()
    cam = cam_state(T_REF)
    x, y, z, e, vis = _project_pts(cam, ap)
    rng = np.random.default_rng(77)
    n = len(x)
    lum = e.mean(1)
    inb = vis & (x > 0) & (x < W - 1) & (y > 0) & (y < BH - 1)
    img = np.zeros((BH, W), np.float32)
    np.add.at(img, (y[inb].astype(int), x[inb].astype(int)), lum[inb])
    bl = cv2.GaussianBlur(img, (0, 0), 16)
    mx = cv2.dilate(bl, np.ones((41, 41), np.uint8))
    pk = np.argwhere((bl >= mx) & (bl > bl.max() * 0.004))
    vals = bl[pk[:, 0], pk[:, 1]]
    order = np.argsort(-vals)
    centers, cv_ = [], []
    for o in order:
        p = pk[o][::-1].astype(float)
        if p[1] < 60 or p[1] > BH - 40: continue
        if centers and min(math.hypot(p[0] - c[0], p[1] - c[1]) for c in centers) < 78: continue
        centers.append(p); cv_.append(vals[o])
        if len(centers) >= 64: break
    C = np.array(centers); K = len(C)
    ext = vis & (x > -500) & (x < W + 500) & (y > -400) & (y < BH + 400)
    cid = np.full(n, -1)
    d, idx = cKDTree(C).query(np.stack([x[ext], y[ext]], -1))
    cid[ext] = idx
    share = np.zeros(n)
    tot = np.bincount(idx, lum[ext], minlength=K)
    share[ext] = lum[ext] / np.maximum(tot[idx], 1e-9)
    rank = np.zeros(n)
    dn = d / np.maximum(np.bincount(idx, d, minlength=K)[idx] / np.maximum(np.bincount(idx, minlength=K)[idx], 1) * 2.2, 1)
    rank[ext] = np.clip(dn, 0, 1)
    O = np.array([PX, PY])
    th0 = 0.40
    Rm = np.array([[math.cos(th0), -math.sin(th0)], [math.sin(th0), math.cos(th0)]])
    base = O + ((C - O) * 1.10) @ Rm.T
    S = base + rng.normal(0, 26, (K, 2))
    S[:, 0] = np.clip(S[:, 0], 90, W - 90); S[:, 1] = np.clip(S[:, 1], 70, BH - 60)
    jit = S - base
    br = np.array(cv_) ** 0.35; br = (br - br.min()) / max(br.max() - br.min(), 1e-9)
    Ek = 0.9 + 3.0 * br * rng.uniform(0.6, 1.0, K)
    app = 0.5 + 2.3 * rng.random(K)
    tk = 3.0 + 1.5 * rng.random(K)
    phs = rng.uniform(0, 2 * np.pi, (K, 2))
    return dict(C=C, cid=cid, share=share, rank=rank, jit=jit, Ek=Ek, app=app, tk=tk, phs=phs, th0=th0)

# =========================================================================================== projection / splatting
def _project_pts(cam, ap):
    c = cam.to_cam(ap['pos'])
    z = c[:, 2]
    vis = z > 0.5
    d = np.sqrt((c ** 2).sum(1))
    toward = ap['pos'] - cam.C
    horizon_ok = (np.einsum('ij,ij->i', toward, ap['nrm']) < 0) | (ap['kind'] == 1)
    vis &= horizon_ok
    zs = np.where(vis, z, 1.0)
    x = PX + cam.f * c[:, 0] / zs; y = PY + cam.f * c[:, 1] / zs
    # flux with atmospheric extinction (reddening toward the horizon)
    mu = np.clip(-np.einsum('ij,ij->i', toward, ap['nrm']) / np.maximum(d, 1e-6), 0.02, 1)
    path = np.minimum(d, 8000.0 / mu) / 8000.0
    tau = np.array([0.20, 0.30, 0.52], np.float32)
    ext = np.exp(-path[:, None].astype(np.float32) * tau[None])
    flux = (EXPO * (cam.f / np.maximum(d, 1.0)) ** 2).astype(np.float32)
    e = ap['rgb'] * (flux[:, None] * ext)
    return x, y, z, e, vis

def splat(buf, x, y, e):
    """Bilinear energy splat of points (x,y in band px) into buf (BH,W,3)."""
    h, w = buf.shape[:2]
    m = (x >= 0) & (x < w - 1) & (y >= 0) & (y < h - 1)
    if not m.any(): return
    x, y, e = x[m], y[m], e[m]
    x0 = np.floor(x).astype(np.int64); y0 = np.floor(y).astype(np.int64)
    fx = (x - x0).astype(np.float32)[:, None]; fy = (y - y0).astype(np.float32)[:, None]
    i00 = y0 * w + x0
    idx = np.concatenate([i00, i00 + 1, i00 + w, i00 + w + 1])
    wt = np.concatenate([e * (1 - fx) * (1 - fy), e * fx * (1 - fy), e * (1 - fx) * fy, e * fx * fy])
    flat = buf.reshape(-1, 3)
    for ch in range(3):
        flat[:, ch] += np.bincount(idx, wt[:, ch], minlength=h * w).astype(np.float32)

def softclip(x, cap):
    ch = cv2.split(x)
    L = cv2.max(cv2.max(ch[0], ch[1]), ch[2])
    Lm = np.maximum(L, 1e-7)
    sc = cap * (1 - np.exp(-Lm / cap)) / Lm
    return x * sc[..., None]

def blur_fast(img, sigma):
    if sigma < 0.3: return img
    if sigma < 3: return cv2.GaussianBlur(img, (0, 0), sigma)
    k = int(min(8, max(2, sigma / 2.5)))
    h, w = img.shape[:2]
    sm = cv2.resize(img, (max(1, w // k), max(1, h // k)), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), sigma / k)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)

def glare(E):
    core = softclip(cv2.GaussianBlur(E, (0, 0), 0.55), 2.6)
    ch = cv2.split(E)
    G = E / (1 + cv2.max(cv2.max(ch[0], ch[1]), ch[2]) / 300.0)[..., None]
    halo = blur_fast(G, 2.2) * 0.05 + blur_fast(G, 8) * 0.018 + blur_fast(G, 30) * 0.010
    return core + softclip(halo, 3.0)

# =========================================================================================== base: earth, ocean, limb, sky
def _rays(cam, xs, ys):
    gx, gy = np.meshgrid(xs, ys)
    d = np.stack([(gx - PX) / cam.f, (gy - PY) / cam.f, np.ones_like(gx)], -1) @ cam.R
    return d / np.linalg.norm(d, axis=-1, keepdims=True)

MOON = _n(np.array([0.25, 0.80, 0.24]))        # low moon ahead of the camera, beyond the horizon
BQ = 4                                           # base layer is rendered at 1/4 resolution

def base_layer(cam, T, earth_w=1.0):
    qw, qh = W // BQ, BH // BQ
    xs = np.arange(qw) * BQ + (BQ - 1) / 2.0; ys = np.arange(qh) * BQ + (BQ - 1) / 2.0
    d = _rays(cam, xs, ys)
    oc = cam.C - CEN
    ocn = float(np.linalg.norm(oc)); alt = ocn - RE
    b = d @ oc
    cc = alt * (ocn + RE)
    disc = b * b - cc
    hit = (disc > 0) & (b < 0)
    tq = np.where(hit, -b - np.sqrt(np.maximum(disc, 0)), 0.0)
    hc = np.where(b < 0, np.sqrt(np.maximum(ocn * ocn - b * b, 0)) - RE, alt)
    space_w = sstep(30e3, 120e3, alt)
    out = np.zeros((qh, qw, 3), np.float32)
    if hit.any():
        P = cam.C[None, None] + tq[..., None] * d
        Q = P - CEN
        hq = np.hypot(Q[..., 0], Q[..., 1])
        th = np.arctan2(hq, Q[..., 2])
        rho = th * RE
        u = np.where(hq > 0, Q[..., 0] / np.maximum(hq, 1e-9) * rho, 0)
        v = np.where(hq > 0, Q[..., 1] / np.maximum(hq, 1e-9) * rho, 0)
        nq = (Q / (RE + 0.0)).astype(np.float32)
        d32 = d.astype(np.float32)
        mu = np.clip(-(d32 * nq).sum(-1), 0.0, 1.0)
        foot = BQ * tq / cam.f / np.maximum(mu, 0.08) ** 0.5
        mw = np.maximum(foot * 2.0, 25.0)
        s = land_s(u, v, mw).astype(np.float32)
        aw = land_aa(mw).astype(np.float32) * 2
        lm = np.clip(s / np.maximum(aw, 1e-12) * 0.5 + 0.5, 0, 1)
        ter = 0.5 + 0.45 * vnoise(u / 1.2e5 + 1.7, v / 1.2e5 + 2.9, 6) + 0.25 * vnoise(u / 2.6e4, v / 2.6e4, 7)
        land = np.array([0.020, 0.019, 0.021], np.float32) * (0.55 + 0.6 * ter[..., None])
        # moon glint on the sea: reflected view ray against the low moon
        rv = d32 + 2 * mu[..., None] * nq
        cg = np.clip((rv * MOON.astype(np.float32)).sum(-1), -1, 1)
        glint = np.exp(-(1 - cg) / 0.010) * 0.10 + np.exp(-(1 - cg) / 0.05) * 0.02
        ocean = np.array([0.0028, 0.0052, 0.0125], np.float32) + np.array([0.006, 0.011, 0.024], np.float32) * ((1 - mu) ** 4)[..., None] \
            + np.array([0.55, 0.65, 0.80], np.float32) * glint[..., None]
        g = land * lm[..., None] + ocean * (1 - lm[..., None])
        am = 1.0 / np.maximum(mu, 0.035)
        haze_sp = (1 - np.exp(-0.035 * am))[..., None] * np.array([0.016, 0.032, 0.075], np.float32)
        g_sp = g * np.exp(-0.05 * am)[..., None] + haze_sp
        fogk = (1 - np.exp(-tq / 22000.0)).astype(np.float32)[..., None]
        g_lo = g * (1 - fogk) + fogk * np.array([0.020, 0.018, 0.020], np.float32)
        gg = g_sp * space_w + g_lo * (1 - space_w)
        out = np.where(hit[..., None], gg, out)
    upd = (d @ cam.upc).astype(np.float32)
    sky_lo = np.array([0.0055, 0.0075, 0.019], np.float32) + np.array([0.030, 0.024, 0.022], np.float32) * np.exp(-np.maximum(upd, 0) / 0.09)[..., None]
    sky_sp = VOID * 0.75 + np.array([0.05, 0.09, 0.20], np.float32) * np.exp(-np.maximum(hc, 0) / 9e3).astype(np.float32)[..., None] * 0.5
    sky = sky_sp * space_w + sky_lo * (1 - space_w)
    out = np.where(hit[..., None], out, sky).astype(np.float32)
    full = cv2.resize(out, (W, BH), interpolation=cv2.INTER_LINEAR)
    if space_w > 0.01 and (np.abs(hc) < 250e3).any():
        hcf = cv2.resize(hc.astype(np.float32), (W, BH), interpolation=cv2.INTER_LINEAR)
        lw = sstep(4.0, 7.8, T) if T < 11 else 1.0
        ag = np.exp(-((hcf - 92e3) / 6.5e3) ** 2) * 0.050 * lw
        lo = np.where(hcf < 0, np.exp(hcf / 30e3), np.exp(-np.maximum(hcf, 0) / 7.5e3)) * 0.075 * lw
        full += (ag[..., None] * np.array([0.42, 0.85, 0.80], np.float32) + lo[..., None] * np.array([0.22, 0.42, 0.95], np.float32)) * space_w
    if earth_w < 1.0:
        full = VOID[None, None] * (1 - earth_w) + full * earth_w
    return full

# =========================================================================================== houses / trees (painter's)
NEAR = 0.15

def _clip_near(cv):
    """Clip polygon (k,3 camera coords) against z>NEAR."""
    out = []
    n = len(cv)
    for i in range(n):
        a, b = cv[i], cv[(i + 1) % n]
        ina, inb = a[2] > NEAR, b[2] > NEAR
        if ina: out.append(a)
        if ina != inb:
            t = (NEAR - a[2]) / (b[2] - a[2])
            out.append(a + t * (b - a))
    return np.array(out) if len(out) >= 3 else None

def _poly2d(cam, V):
    cv = (V - cam.C) @ cam.R.T
    if (cv[:, 2] <= NEAR).any():
        cv = _clip_near(cv)
        if cv is None: return None
    x = PX + cam.f * cv[:, 0] / cv[:, 2]; y = PY + cam.f * cv[:, 1] / cv[:, 2]
    return np.stack([x, y], -1)

def _fill(img, pts, col):
    p = np.round(pts * 16).astype(np.int32)
    cv2.fillConvexPoly(img, p, col, lineType=cv2.LINE_8, shift=4)

class Layer:
    """Supersampled (2x) opaque object layer + 1x depth buffer."""
    def __init__(self, ss=2):
        self.ss = ss
        self.rgba = np.zeros((BH * ss, W * ss, 4), np.float32)
        self.z = np.full((BH, W), 1e12, np.float32)
        self.used = False
    def poly(self, pts, col, depth, alpha=1.0):
        if pts is None: return
        if pts[:, 0].max() < -5 or pts[:, 0].min() > W + 5 or pts[:, 1].max() < -5 or pts[:, 1].min() > BH + 5: return
        pts = np.clip(pts, -4000, 6000)
        _fill(self.rgba, pts * self.ss, (float(col[0]), float(col[1]), float(col[2]), 1.0))
        _fill(self.z, pts, float(depth))
        self.used = True
    def circle(self, x, y, r, col, depth):
        if x + r < -5 or x - r > W + 5 or y + r < -5 or y - r > BH + 5: return
        r = min(r, 5000)
        cv2.circle(self.rgba, (int(round(x * self.ss * 16)), int(round(y * self.ss * 16))), int(round(r * self.ss * 16)),
                   (float(col[0]), float(col[1]), float(col[2]), 1.0), -1, cv2.LINE_8, 4)
        cv2.circle(self.z, (int(round(x * 16)), int(round(y * 16))), int(round(r * 16)), float(depth), -1, cv2.LINE_8, 4)
        self.used = True
    def resolve(self):
        if self.ss == 1: return self.rgba
        return cv2.resize(self.rgba, (W, BH), interpolation=cv2.INTER_AREA)

def _draw_house(L, cam, h, dist, wfade_pts):
    for fc in h['faces']:
        V = fc['V']
        cen = V.mean(0)
        if float((cam.C - cen) @ fc['n']) <= 0: continue
        pts = _poly2d(cam, V)
        if pts is None: continue
        L.poly(pts, fc['col'], dist)
        for (wc, ww, wh, wcol) in fc['wins']:
            nn = fc['n']
            tang = _n(np.cross(np.array([0, 0, 1.0]), nn)) if abs(nn[2]) < 0.5 else np.array([1.0, 0, 0])
            dw = np.linalg.norm(wc - cam.C)
            size = cam.f * min(ww, wh) / dw
            if size < 0.9: continue
            k = sstep(1.2, 3.2, size)
            q = np.array([wc - tang * ww / 2 - [0, 0, wh / 2], wc + tang * ww / 2 - [0, 0, wh / 2],
                          wc + tang * ww / 2 + [0, 0, wh / 2], wc - tang * ww / 2 + [0, 0, wh / 2]])
            wp = _poly2d(cam, q)
            if wp is None: continue
            if wcol is None:
                c = fc['col'] * 0.35 + np.array([0.004, 0.006, 0.012], np.float32)
                c = fc['col'] * (1 - k) + c * k
            else:
                c = fc['col'] * (1 - k) + wcol * k
            L.poly(wp, c, dist - 0.5)
        if 'door' in fc:
            xd, _, wd, hd = fc['door']
            q = np.array([[xd - wd / 2, -0.02, 0], [xd + wd / 2, -0.02, 0], [xd + wd / 2, -0.02, hd], [xd - wd / 2, -0.02, hd]])
            L.poly(_poly2d(cam, q), fc['col'] * 0.45, dist - 0.5)

TREE_COL = np.array([0.0045, 0.0060, 0.0055], np.float32)

def _draw_tree(L, cam, t, dist):
    tx, ty, tr = t
    zc = tr * 1.45 + 1.2
    p = cam.to_cam(np.array([[tx, ty, zc], [tx, ty, 0.0]]))
    if p[0, 2] < NEAR + tr * 0.5: return
    x = PX + cam.f * p[0, 0] / p[0, 2]; y = PY + cam.f * p[0, 1] / p[0, 2]
    rr = cam.f * tr / p[0, 2]
    if rr < 0.8: return
    if p[1, 2] > NEAR:
        xb = PX + cam.f * p[1, 0] / p[1, 2]; yb = PY + cam.f * p[1, 1] / p[1, 2]
        tw = max(1.0, cam.f * 0.18 / p[1, 2])
        tp = np.array([[x - tw, y], [x + tw, y], [xb + tw, yb], [xb - tw, yb]])
        L.poly(tp, TREE_COL * 0.7, dist)
    # canopy as a few overlapping lobes (deterministic per tree)
    sd = int(abs(tx * 13.1 + ty * 7.7)) % 1000
    r2 = np.random.default_rng(sd)
    L.circle(x, y, rr, TREE_COL, dist)
    for q in range(3):
        a = r2.uniform(0, 2 * np.pi); o = r2.uniform(0.35, 0.6)
        L.circle(x + math.cos(a) * rr * o, y + math.sin(a) * rr * o * 0.8 - rr * 0.15, rr * r2.uniform(0.55, 0.75),
                 TREE_COL * r2.uniform(0.9, 1.5), dist)

def draw_hood(cam, img, Elights, hero_cb=None):
    """Draw houses/trees (painter's), returns depth buffer for light occlusion."""
    hood = _hood()
    houses, trees = hood['houses'], hood['trees']
    Hc = np.array([h['c'] for h in houses])
    ch = cam.to_cam(Hc)
    dh = np.linalg.norm(Hc - cam.C, axis=1)
    size_h = cam.f * 10.0 / np.maximum(dh, 1)
    okh = (ch[:, 2] > -12) & (size_h > 1.5)
    xh = PX + cam.f * ch[:, 0] / np.maximum(ch[:, 2], 1e-3); yh = PY + cam.f * ch[:, 1] / np.maximum(ch[:, 2], 1e-3)
    okh &= ((xh > -0.6 * W) & (xh < 1.6 * W) & (yh > -0.8 * BH) & (yh < 1.8 * BH)) | (ch[:, 2] < 30)
    Tp = np.stack([trees[:, 0], trees[:, 1], trees[:, 2] * 1.45 + 1.2], -1)
    ct = cam.to_cam(Tp)
    dt = np.linalg.norm(Tp - cam.C, axis=1)
    size_t = cam.f * trees[:, 2] / np.maximum(dt, 1)
    okt = (ct[:, 2] > 0) & (size_t > 1.0)
    items = [(dh[i], 0, i) for i in np.where(okh)[0]] + [(dt[i], 1, i) for i in np.where(okt)[0]]
    items.sort(key=lambda a: -a[0])
    fade = sstep(2600, 1300, cam.r)
    hero_i = [i for i, h in enumerate(houses) if h['hero']][0]
    hero_d = dh[hero_i]
    far_items = [it for it in items if not (it[1] == 0 and it[2] == hero_i) and it[0] >= hero_d]
    near_items = [it for it in items if not (it[1] == 0 and it[2] == hero_i) and it[0] < hero_d]
    zbuf = np.full((BH, W), 1e12, np.float32)

    def run(lst):
        L = Layer(2)
        for d, kind, i in lst:
            if kind == 0: _draw_house(L, cam, houses[i], d, None)
            else: _draw_tree(L, cam, trees[i], d)
        if not L.used: return None
        np.minimum(zbuf, L.z, out=zbuf)
        return L.resolve()

    def comp(lay):
        if lay is None: return
        a = lay[..., 3:4] * fade
        img[:] = img * (1 - a) + lay[..., :3] * fade
    comp(run(far_items))
    if okh[hero_i]:
        L = Layer(2)
        _draw_house(L, cam, houses[hero_i], hero_d, None)
        if L.used:
            np.minimum(zbuf, L.z, out=zbuf)
            comp(L.resolve())
        if hero_cb is not None: hero_cb(img, zbuf)
    comp(run(near_items))
    return zbuf

# =========================================================================================== interior (2.5D, authored in a hero camera)
DM, DF = 2.5, 3.2            # effective parallax depths (mid = desk/page, far = wall/window)
FHERO = 2400.0
XM0, YM0, WM, HM = -300, 120, 2520, 1120          # mid canvas (hero full-frame px)
XF0, YF0, WF, HF = -640, -180, 3200, 1440         # far canvas extent (hero full-frame px), stored at half res
DESK_YW = 0.105                                   # wall plane (desk coords)

def _hero_cam():
    el = math.radians(40.0); dist = 0.46
    C = np.array([0.0, -dist * math.cos(el), dist * math.sin(el)])
    fwd0 = _n(-C)
    up = np.array([0, 0, 1.0])
    right = _n(np.cross(fwd0, up)); up0 = np.cross(right, fwd0)
    a = math.atan2(110, FHERO)
    fwd = math.cos(a) * fwd0 + math.sin(a) * up0
    cam = Cam(C, fwd, up, FHERO)
    return cam

def _hero_proj(cam, P):
    c = cam.to_cam(P)
    return 960 + FHERO * c[..., 0] / c[..., 2], 540 + FHERO * c[..., 1] / c[..., 2], c[..., 2]

def _hero_rays(cam, xs, ys):
    gx, gy = np.meshgrid(xs.astype(np.float32), ys.astype(np.float32))
    d = np.stack([(gx - 960) / FHERO, (gy - 540) / FHERO, np.ones_like(gx)], -1) @ cam.R.astype(np.float32)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)

LAMP_P = np.array([0.175, 0.035, 0.40])
LAMP_AIM = np.array([0.0, -0.03, 0.0])
WARM_L = np.array([1.0, 0.70, 0.44], np.float32)
COOL_L = np.array([0.55, 0.68, 1.0], np.float32)

def _lamp_irr(P, nrm):
    """Lamp irradiance at points P (...,3) with normals nrm (...,3)."""
    Lv = LAMP_P - P
    d2 = (Lv ** 2).sum(-1)
    dist = np.sqrt(d2)
    ld = Lv / dist[..., None]
    cosi = np.clip((ld * nrm).sum(-1), 0, 1)
    ax = _n(LAMP_AIM - LAMP_P)
    ca = (-ld * ax).sum(-1)
    cone = 0.10 + 0.90 * np.clip((ca - math.cos(math.radians(58))) / (math.cos(math.radians(22)) - math.cos(math.radians(58))), 0, 1) ** 1.5
    return 0.050 * cosi * cone / np.maximum(d2, 1e-4)

def _page_tex():
    """Albedo texture of the eulogy page, 0.21 x 0.297 m at 5600 px/m (rendered at 2x then reduced)."""
    ppm = 11200
    pw, ph = int(0.21 * ppm), int(0.297 * ppm)
    im = Image.new('L', (pw, ph), 255)
    d = ImageDraw.Draw(im)
    mm = ppm / 1000.0
    def size_fit(txt, fname, pt, maxw):
        s = int(pt * 0.3528 * mm)
        while font(fname, s).getlength(txt) > maxw and s > 10: s -= 2
        return s
    lines_body = ["My father fixed things. Radios, bicycles,",
                  "the hinge on our kitchen door that he said",
                  "would outlive us both. It did.",
                  "He never said much. He didn't need to."]
    left = 27 * mm
    maxw = 0.21 * ppm - 2 * left
    sb = min(size_fit(t, 'serif', 19.5, maxw) for t in lines_body)
    st = int(sb * 1.22)
    title = "For my father, " + FATHER
    d.text((left, 31 * mm), title, font=font('serif_m' if False else 'serif', st), fill=18, anchor='ls')
    # years, letterspaced
    yrs = "1951 – 2026"
    fy = font('serif_l', int(sb * 0.86)); x = left
    for ch in yrs:
        d.text((x, 41.5 * mm), ch, font=fy, fill=70, anchor='ls'); x += fy.getlength(ch) + 0.9 * mm
    y = 58 * mm
    lead = sb * 1.62
    for i, t in enumerate(lines_body):
        if i == 3: y += lead * 0.55
        d.text((left, y), t, font=font('serif', sb), fill=22, anchor='ls')
        y += lead
    a = np.asarray(im).astype(np.float32) / 255
    a = cv2.resize(a, (pw // 2, ph // 2), interpolation=cv2.INTER_AREA)
    return a, ppm / 2

def _draft_tex():
    ppm = 2800
    pw, ph = int(0.21 * ppm), int(0.297 * ppm)
    im = Image.new('L', (pw, ph), 255)
    d = ImageDraw.Draw(im)
    rng = np.random.default_rng(3)
    mm = ppm / 1000.0
    y = 31 * mm
    for i in range(22):
        L = rng.uniform(0.45, 1.0) * 150 * mm if i % 6 != 5 else rng.uniform(0.2, 0.5) * 150 * mm
        d.rectangle((27 * mm, y - 2.3 * mm, 27 * mm + L, y), fill=150)
        if rng.random() < 0.25:
            d.line((27 * mm, y - 1.2 * mm, 27 * mm + L * rng.uniform(0.4, 1), y - 1.0 * mm), fill=60, width=int(0.5 * mm))
        y += 9.2 * mm if i % 6 != 5 else 15 * mm
    return np.asarray(im).astype(np.float32) / 255, ppm

def _wood(X, Y):
    g = vnoise(X / 0.25 + 3.0, Y / 0.05, 1) * 0.8
    ring = 0.5 + 0.5 * np.sin(2 * np.pi * (Y / 0.0075 + g + 0.3 * vnoise(X / 0.06, Y / 0.02, 2)))
    fine = vnoise(X / 0.02, Y / 0.0012, 3) * 0.08
    base = np.array([0.135, 0.078, 0.046], np.float32)
    k = (0.80 + 0.22 * ring ** 2 + fine)[..., None]
    return base * k

def _ray_cyl(O, D, cx, cy, r, z0, z1):
    """Vertical cylinder side intersection; returns t (inf if miss) and normal."""
    ox, oy = O[0] - cx, O[1] - cy
    a = D[..., 0] ** 2 + D[..., 1] ** 2
    b = 2 * (ox * D[..., 0] + oy * D[..., 1])
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.maximum(disc, 0))) / np.maximum(2 * a, 1e-12)
    z = O[2] + t * D[..., 2]
    ok = (disc > 0) & (t > 0) & (z >= z0) & (z <= z1)
    t = np.where(ok, t, np.inf)
    P = O + t[..., None] * D
    nrm = np.stack([P[..., 0] - cx, P[..., 1] - cy, np.zeros_like(t)], -1) / r
    return t, nrm

def _ray_plane_z(O, D, z, test):
    t = (z - O[2]) / np.where(np.abs(D[..., 2]) < 1e-9, -1e-9, D[..., 2])
    P = O + t[..., None] * D
    ok = (t > 0) & test(P)
    return np.where(ok, t, np.inf), P

def _ray_box(O, D, lo, hi):
    inv = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
    t0 = (np.array(lo) - O) * inv; t1 = (np.array(hi) - O) * inv
    tmin = np.minimum(t0, t1).max(-1); tmax = np.maximum(t0, t1).min(-1)
    ok = (tmax >= tmin) & (tmin > 0)
    t = np.where(ok, tmin, np.inf)
    ax = np.argmax(np.minimum(t0, t1), -1)
    sgn = -np.sign(np.take_along_axis(D, ax[..., None], -1)[..., 0])
    nrm = np.zeros(D.shape, np.float32)
    for i in range(3):
        nrm[..., i] = np.where(ax == i, sgn, 0)
    return t, nrm

def _ray_capsule(O, D, A, B, r):
    """Capsule A-B radius r (infinite-cylinder body + sphere caps)."""
    ba = B - A; L = np.linalg.norm(ba); u = ba / L
    oa = O - A
    dd = D - (D @ u)[..., None] * u
    oo = oa - (oa @ u) * u
    a = (dd * dd).sum(-1); b = 2 * (dd * oo).sum(-1); c = (oo @ oo) - r * r
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.maximum(disc, 0))) / np.maximum(2 * a, 1e-12)
    P = O + t[..., None] * D
    s = (P - A) @ u
    ok = (disc > 0) & (t > 0) & (s >= 0) & (s <= L)
    tb = np.where(ok, t, np.inf)
    best = tb.copy()
    nrm = (P - A - s[..., None] * u) / r
    for Cc in (A, B):
        oc = O - Cc
        bb = (D * oc).sum(-1); cc = oc @ oc - r * r
        di = bb * bb - cc
        ts = -bb - np.sqrt(np.maximum(di, 0))
        oks = (di > 0) & (ts > 0) & (ts < best)
        best = np.where(oks, ts, best)
        Ps = O + ts[..., None] * D
        nrm = np.where(oks[..., None], (Ps - Cc) / r, nrm)
    return best, nrm

CUP = (-0.255, -0.105, 0.036, 0.062)          # x, y, radius, height (handle-less tea cup)
LAMPB = (0.33, 0.07, 0.072, 0.022)
STEM = (0.33, 0.07, 0.009, 0.75)
LAPTOP = ((-0.78, -0.05, 0.0), (-0.37, 0.10, 0.017))
PEN = (np.array([0.060, -0.122, 0.0058]), np.array([0.172, -0.064, 0.0058]), 0.0058)
LED_P = np.array([-0.40, -0.051, 0.008])
PAGE_ROT = math.radians(-3.0)
PAGE_TXT = (0.105, 0.064)          # text-block centre in page coords (m from top-left)

def _page_frame():
    e1 = np.array([math.cos(PAGE_ROT), math.sin(PAGE_ROT)])
    e2 = np.array([math.sin(PAGE_ROT), -math.cos(PAGE_ROT)])
    TL = -PAGE_TXT[0] * e1 - PAGE_TXT[1] * e2 + np.array([0.0, 0.0])
    return TL, e1, e2

def _shade(P, nrm, V, alb, gloss=0.0, spec_k=0.0):
    """Returns warm (lamp + ambient) and cool (window) radiance."""
    E = _lamp_irr(P, nrm)
    warm = alb * (E[..., None] * WARM_L + 0.012 * WARM_L)
    if spec_k > 0:
        ld = _n_last(LAMP_P - P)
        hv = _n_last(ld + V)
        sp = np.clip((hv * nrm).sum(-1), 0, 1) ** gloss * spec_k
        warm = warm + (sp * E * 6.0)[..., None] * WARM_L
    wd = _n_last(np.array([0.0, DESK_YW + 0.25, 0.55]) - P)
    cw = np.clip((wd * nrm).sum(-1), 0, 1)
    cool = alb * (cw * 0.16)[..., None] * COOL_L
    return warm.astype(np.float32), cool.astype(np.float32)

def _n_last(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-9)

@functools.lru_cache(None)
def _interior():
    cam = _hero_cam()
    O = cam.C.astype(np.float32)
    # ----------------------------------------------------------- mid layer: desk, pages, objects
    xs = XM0 + np.arange(WM) + 0.5; ys = YM0 + np.arange(HM) + 0.5
    D = _hero_rays(cam, xs, ys)
    V = -D
    tdesk = np.where(D[..., 2] < 0, -O[2] / np.minimum(D[..., 2], -1e-9), np.inf)
    P = O + tdesk[..., None] * D
    X, Y = P[..., 0], P[..., 1]
    desk_ok = (Y < DESK_YW) & np.isfinite(tdesk)
    nz = np.zeros_like(P); nz[..., 2] = 1
    alb = _wood(X, Y)
    # second (draft) page underneath
    dt, dppm = _draft_tex()
    rot2 = math.radians(5.5)
    e1b = np.array([math.cos(rot2), math.sin(rot2)]); e2b = np.array([math.sin(rot2), -math.cos(rot2)])
    TLb = np.array([-0.165, 0.098])
    rel = np.stack([X - TLb[0], Y - TLb[1]], -1)
    pu = (rel @ e1b) * dppm; pv = (rel @ e2b) * dppm
    din = (pu > 0) & (pv > 0) & (pu < dt.shape[1] - 1) & (pv < dt.shape[0] - 1)
    dval = cv2.remap(dt, pu.astype(np.float32), pv.astype(np.float32), cv2.INTER_LINEAR, borderValue=1.0)
    paper = np.array([0.80, 0.77, 0.71], np.float32)
    alb = np.where(din[..., None], paper * (0.25 + 0.75 * dval[..., None]) * 0.93, alb)
    # main page
    pt, pppm = _page_tex()
    TL, e1, e2 = _page_frame()
    rel = np.stack([X - TL[0], Y - TL[1]], -1)
    pu = (rel @ e1) * pppm; pv = (rel @ e2) * pppm
    pin = (pu > 0) & (pv > 0) & (pu < pt.shape[1] - 1) & (pv < pt.shape[0] - 1)
    pval = cv2.remap(pt, pu.astype(np.float32), pv.astype(np.float32), cv2.INTER_LINEAR, borderValue=1.0)
    fib = 1 + 0.02 * vnoise(X / 0.0015, Y / 0.0015, 4)
    palb = paper[None, None] * (0.03 + 0.97 * pval[..., None]) * fib[..., None]
    # page drop shadow on what lies beneath
    pm = pin.astype(np.float32)
    sh = cv2.GaussianBlur(np.roll(np.roll(pm, 6, 1), 4, 0), (0, 0), 5)
    dm = din.astype(np.float32)
    shd = cv2.GaussianBlur(np.roll(np.roll(dm, 5, 1), 3, 0), (0, 0), 4)
    alb = alb * (1 - 0.45 * sh[..., None] * (1 - pm[..., None])) * (1 - 0.3 * shd[..., None] * (1 - dm[..., None]))
    alb = np.where(pin[..., None], palb, alb)
    # soft object shadows on the desk (area light: 4 samples toward the lamp)
    occl = np.zeros(X.shape, np.float32)
    Pd = np.stack([X, Y, np.zeros_like(X)], -1)
    for j, off in enumerate([(0, 0, 0), (0.03, 0.0, 0.0), (-0.02, 0.02, 0.0), (0.0, -0.025, 0.01)]):
        Lp = LAMP_P + np.array(off)
        Ld = Lp - Pd
        dist = np.linalg.norm(Ld, axis=-1); Ldn = Ld / dist[..., None]
        Oo = Pd + Ldn * 1e-4
        hit = np.zeros(X.shape, bool)
        for (cx, cy, r, h) in (CUP, LAMPB):
            t, _ = _ray_cyl_arr(Oo, Ldn, cx, cy, r, 0, h); hit |= t < dist
        t, _ = _ray_cyl_arr(Oo, Ldn, STEM[0], STEM[1], STEM[2], 0, STEM[3]); hit |= t < dist
        t, _ = _ray_box_arr(Oo, Ldn, *LAPTOP); hit |= t < dist
        t, _ = _ray_capsule_arr(Oo, Ldn, *PEN); hit |= t < dist
        occl += hit.astype(np.float32) * 0.25
    occl = cv2.GaussianBlur(occl, (0, 0), 3.0)
    warm, cool = _shade(Pd, nz, V, alb, gloss=40.0, spec_k=0.0)
    # satin varnish highlight on the wood only
    ld = _n_last(LAMP_P - Pd); hv = _n_last(ld + V)
    spec = (np.clip(hv[..., 2], 0, 1) ** 60 * 0.9 * _lamp_irr(Pd, nz) * 4.0) * (~pin) * (~din)
    warm = warm * (1 - 0.82 * occl[..., None]) + (spec * (1 - occl))[..., None] * WARM_L
    tbest = np.where(desk_ok, tdesk, np.inf)
    # objects
    objs = []
    t, n_ = _ray_cyl(O, D, *CUP[:3], 0.0, CUP[3]); objs.append(('cup', t, n_))
    t2, P2 = _ray_plane_z(O, D, CUP[3], lambda Pp: (Pp[..., 0] - CUP[0]) ** 2 + (Pp[..., 1] - CUP[1]) ** 2 < CUP[2] ** 2)
    objs.append(('cuptop', t2, np.broadcast_to(np.array([0, 0, 1.0], np.float32), D.shape)))
    t, n_ = _ray_cyl(O, D, *LAMPB[:3], 0.0, LAMPB[3]); objs.append(('lampb', t, n_))
    t2, _ = _ray_plane_z(O, D, LAMPB[3], lambda Pp: (Pp[..., 0] - LAMPB[0]) ** 2 + (Pp[..., 1] - LAMPB[1]) ** 2 < LAMPB[2] ** 2)
    objs.append(('lampt', t2, np.broadcast_to(np.array([0, 0, 1.0], np.float32), D.shape)))
    t, n_ = _ray_cyl(O, D, *STEM[:3], LAMPB[3], STEM[3]); objs.append(('stem', t, n_))
    t, n_ = _ray_box(O, D, *LAPTOP); objs.append(('laptop', t, n_))
    t, n_ = _ray_capsule(O, D, *PEN); objs.append(('pen', t, n_))
    oid = np.full(X.shape, -1)
    for i, (nm, t, n_) in enumerate(objs):
        m = t < tbest
        oid = np.where(m, i, oid); tbest = np.where(m, t, tbest)
    alpha = np.isfinite(tbest).astype(np.float32)
    Pb = O + np.where(np.isfinite(tbest), tbest, 0)[..., None] * D
    mats = {'cup': (np.array([0.60, 0.66, 0.62], np.float32), 90.0, 0.9),
            'cuptop': (np.array([0.22, 0.13, 0.05], np.float32), 200.0, 2.5),
            'lampb': (np.array([0.10, 0.075, 0.045], np.float32), 30.0, 1.4),
            'lampt': (np.array([0.10, 0.075, 0.045], np.float32), 30.0, 1.4),
            'stem': (np.array([0.10, 0.075, 0.045], np.float32), 30.0, 1.4),
            'laptop': (np.array([0.16, 0.165, 0.175], np.float32), 25.0, 0.5),
            'pen': (np.array([0.020, 0.020, 0.024], np.float32), 120.0, 2.0)}
    for i, (nm, t, n_) in enumerate(objs):
        m = oid == i
        if not m.any(): continue
        a_, g_, k_ = mats[nm]
        nn = np.asarray(n_, np.float32)[m]
        w_, c_ = _shade(Pb[m], nn, V[m], a_, gloss=g_, spec_k=k_)
        if nm == 'cuptop':
            rr = np.hypot(Pb[m][:, 0] - CUP[0], Pb[m][:, 1] - CUP[1])
            rim = rr > CUP[2] - 0.0035
            w2, c2 = _shade(Pb[m], nn, V[m], mats['cup'][0], 40.0, 0.4)
            w_ = np.where(rim[:, None], w2, w_ * 0.8); c_ = np.where(rim[:, None], c2, c_)
        if nm == 'pen':
            s = ((Pb[m] - PEN[0]) @ _n(PEN[1] - PEN[0])) / np.linalg.norm(PEN[1] - PEN[0])
            band = (np.abs(s - 0.72) < 0.03) | (s > 0.97)
            w2, c2 = _shade(Pb[m], nn, V[m], np.array([0.55, 0.42, 0.22], np.float32), 60.0, 2.5)
            w_ = np.where(band[:, None], w2, w_); c_ = np.where(band[:, None], c2, c_)
        warm[m] = w_; cool[m] = c_
    alpha = alpha * (Y < DESK_YW + 10) * 1.0
    # the stem continues upward out of frame; everything that is neither desk nor object stays transparent
    depth = np.where(np.isfinite(tbest), tbest, 2.0).astype(np.float32)
    zf = 0.46
    dof = np.clip(np.abs(1.0 / depth - 1.0 / zf) * 0.55, 0, 1).astype(np.float32)
    A = alpha[..., None]
    mid = dict(warm=(warm * A).astype(np.float32), cool=(cool * A).astype(np.float32), a=alpha.astype(np.float32), dof=dof)
    for k in ('warm', 'cool'):
        mid[k + '_b'] = cv2.GaussianBlur(mid[k], (0, 0), 7)
    mid['a_b'] = cv2.GaussianBlur(mid['a'], (0, 0), 7)
    # small mips for far-away viewing
    for k in ('warm', 'cool', 'a', 'warm_b', 'cool_b', 'a_b'):
        mid[k + '_4'] = cv2.resize(mid[k], (WM // 4, HM // 4), interpolation=cv2.INTER_AREA)
    mid['dof_4'] = cv2.resize(dof, (WM // 4, HM // 4), interpolation=cv2.INTER_AREA)
    # ----------------------------------------------------------- far layer (half res): wall, window B, desk beyond
    fw, fh = WF // 2, HF // 2
    xs = XF0 + (np.arange(fw) + 0.5) * 2; ys = YF0 + (np.arange(fh) + 0.5) * 2
    D = _hero_rays(cam, xs, ys)
    tw = (DESK_YW - O[1]) / np.where(D[..., 1] > 1e-6, D[..., 1], 1e-6)
    Pw = O + tw[..., None] * D
    td = np.where(D[..., 2] < 0, -O[2] / np.minimum(D[..., 2], -1e-9), np.inf)
    on_wall = (Pw[..., 2] >= 0) & (D[..., 1] > 0)
    WX, WZ_ = Pw[..., 0], Pw[..., 2]
    nwall = np.zeros_like(Pw); nwall[..., 1] = -1
    plaster = np.array([0.46, 0.43, 0.39], np.float32) * (0.95 + 0.05 * vnoise(WX / 0.02, WZ_ / 0.02, 5))[..., None]
    gx0, gx1, gz0, gz1 = -0.80, 0.52, 0.035, 1.25
    fr = 0.055
    inglass = (WX > gx0) & (WX < gx1) & (WZ_ > gz0) & (WZ_ < gz1) & ~(np.abs(WX + 0.14) < 0.022)
    inframe = (WX > gx0 - fr) & (WX < gx1 + fr) & (WZ_ > gz0 - 0.03) & (WZ_ < gz1 + fr) & ~inglass
    frame_alb = np.array([0.62, 0.60, 0.56], np.float32)
    walb = np.where(inframe[..., None], frame_alb, plaster)
    ww, wc = _shade(Pw, nwall, -D, walb)
    # shade-top spill: a soft arc of light on the wall above the lamp
    arc = np.exp(-((WX - 0.24) / 0.20) ** 2 - ((WZ_ - 0.55) / 0.22) ** 2) * 0.05
    ww = ww + walb * arc[..., None] * WARM_L
    # window light on frame/jambs
    wc = wc + np.where(inframe[..., None], frame_alb * 0.06, 0)
    Pd = O + np.where(np.isfinite(td), td, 0)[..., None] * D
    nz = np.zeros_like(Pd); nz[..., 2] = 1
    dw, dc = _shade(Pd, nz, -D, _wood(Pd[..., 0], Pd[..., 1]))
    use_desk = ~on_wall
    farw = np.where(use_desk[..., None], dw, ww).astype(np.float32)
    farc = np.where(use_desk[..., None], dc, wc).astype(np.float32)
    gl = (inglass & on_wall).astype(np.float32)
    # outside view geometry (in far-canvas pixel space)
    gxx, gyy = np.meshgrid(xs, ys)
    roof = 262 + 26 * np.sin(gxx / 210.0) + 18 * np.sign(np.sin(gxx / 95.0 + 1.0)) + 10 * vnoise(gxx / 60.0, 0 * gxx + 3, 2)
    tree = np.hypot((gxx - 260) / 1.0, (gyy - 205) / 1.15) < 150 + 22 * vnoise(gxx / 25.0, gyy / 25.0, 3)
    sil = ((gyy > roof) | tree).astype(np.float32)
    far = dict(w=farw, c=farc, glass=gl, sil=cv2.GaussianBlur(sil, (0, 0), 1.0), ys=ys.astype(np.float32),
               gx=gxx.astype(np.float32), gy=gyy.astype(np.float32))
    return dict(mid=mid, far=far, cam=cam)

# array-friendly ray tests for shadow rays (origins vary per pixel)
def _ray_cyl_arr(O, D, cx, cy, r, z0, z1):
    ox, oy = O[..., 0] - cx, O[..., 1] - cy
    a = D[..., 0] ** 2 + D[..., 1] ** 2
    b = 2 * (ox * D[..., 0] + oy * D[..., 1])
    c = ox * ox + oy * oy - r * r
    disc = b * b - 4 * a * c
    sq = np.sqrt(np.maximum(disc, 0))
    t1 = (-b - sq) / np.maximum(2 * a, 1e-12); t2 = (-b + sq) / np.maximum(2 * a, 1e-12)
    z1_ = O[..., 2] + t1 * D[..., 2]; z2_ = O[..., 2] + t2 * D[..., 2]
    ok1 = (disc > 0) & (t1 > 0) & (z1_ >= z0) & (z1_ <= z1)
    ok2 = (disc > 0) & (t2 > 0) & (z2_ >= z0) & (z2_ <= z1)
    return np.where(ok1, t1, np.where(ok2, t2, np.inf)), None

def _ray_box_arr(O, D, lo, hi):
    inv = 1.0 / np.where(np.abs(D) < 1e-9, 1e-9, D)
    t0 = (np.array(lo) - O) * inv; t1 = (np.array(hi) - O) * inv
    tmin = np.minimum(t0, t1).max(-1); tmax = np.maximum(t0, t1).min(-1)
    ok = (tmax >= np.maximum(tmin, 0))
    return np.where(ok, np.maximum(tmin, 0), np.inf), None

def _ray_capsule_arr(O, D, A, B, r):
    ba = B - A; L = np.linalg.norm(ba); u = ba / L
    oa = O - A
    dd = D - (D @ u)[..., None] * u
    oo = oa - (oa @ u)[..., None] * u
    a = (dd * dd).sum(-1); b = 2 * (dd * oo).sum(-1); c = (oo * oo).sum(-1) - r * r
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.maximum(disc, 0))) / np.maximum(2 * a, 1e-12)
    P = O + t[..., None] * D
    s = (P - A) @ u
    ok = (disc > 0) & (t > 0) & (s >= -r) & (s <= L + r)
    return np.where(ok, t, np.inf), None

# ----------------------------------------------------------------------------- interior per-frame
def dawn_k(lt):
    return ease((lt - 22.0) / 18.0) ** 1.15 if lt > 22 else 0.0

def _sky(lt, gy):
    k = dawn_k(lt)
    yy = np.clip((gy - (-180)) / 470.0, 0, 1)[..., None]          # 0 top .. 1 near rooftops
    night_top, night_hor = np.array([0.006, 0.008, 0.020]), np.array([0.030, 0.026, 0.030])
    dawn_top, dawn_hor = np.array([0.10, 0.15, 0.30]), np.array([0.42, 0.50, 0.66])
    top = night_top * (1 - k) + dawn_top * k
    hor = night_hor * (1 - k) + dawn_hor * k
    sky = top + (hor - top) * yy ** 1.6
    warmband = np.array([0.20, 0.12, 0.08]) * k ** 2 * np.exp(-((gy - 255) / 40.0) ** 2)[..., None]
    return (sky + warmband).astype(np.float32)

def _far_frame(lt):
    I = _interior(); f = I['far']
    k = dawn_k(lt)
    base = f['w'] + f['c'] * (0.25 + 1.6 * k)
    gx, gy = f['gx'], f['gy']
    sky = _sky(lt, gy)
    # houses opposite: dark facades, faint dawn on them, a few windows that go out as morning comes
    house = np.array([0.006, 0.006, 0.008], np.float32) + np.array([0.05, 0.06, 0.08], np.float32) * k * 0.6
    view = sky * (1 - f['sil'][..., None]) + house * f['sil'][..., None]
    lights = np.zeros_like(view)
    for (x, y, r, col, t_off) in ((520, 318, 9, HOMEW, 26.5), (1460, 300, 8, HOMEW, 31.0), (1180, 330, 7, TV, 999),
                                  (1700, 296, 10, SODIUM * 1.5, 33.5)):
        a = 1 - sstep(t_off, t_off + 1.6, lt)
        if a <= 0: continue
        g = np.exp(-(((gx - x) / r) ** 2 + ((gy - y) / r) ** 2))
        lights += g[..., None] * col * 0.6 * a
    view = view + lights
    # faint reflection of the lamp in the glass
    refl = np.exp(-(((gx - 1500) / 160.0) ** 2 + ((gy - 210) / 120.0) ** 2))[..., None] * WARM_L * 0.05
    view = view + refl
    out = base * (1 - f['glass'][..., None]) + view * f['glass'][..., None] * 0.92
    return out.astype(np.float32)

def _place(layer, s, x_auth0, y_auth0, cx, cy, step, rect, interp=cv2.INTER_LINEAR):
    """Warp an authored canvas (origin at hero px (x_auth0,y_auth0), 'step' hero px per canvas px)
    so that hero point (960,540) lands on (cx,cy) in band coords, scaled by s. rect=(x0,y0,w,h) output."""
    x0, y0, w, h = rect
    a = s * step
    M = np.float32([[a, 0, cx + s * (x_auth0 - 960) - x0], [0, a, cy + s * (y_auth0 - 540) - y0]])
    return cv2.warpAffine(layer, M, (w, h), flags=interp, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def interior_view(delta, lt, cx, cy, rect, blur_mid=0.0, blur_far=None, shift=(0.0, 0.0), led=True, dust=True):
    """Render the room as seen from axial offset `delta` (m, + = closer than the hero frame)."""
    I = _interior(); md = I['mid']
    x0, y0, w, h = rect
    sm = DM / (DM - delta); sf = DF / (DF - delta)
    k = dawn_k(lt)
    small = sm < 0.3
    sfx = '_4' if small else ''
    stepm = 4 if small else 1
    # focus breathing on the desk layer
    breath = 0.5 + 0.5 * math.sin(2 * math.pi * lt / 9.0 - 1.0) if lt > 9.5 else 0.5
    wsc = 0.85 + 0.35 * breath
    dof = np.clip(md['dof' + sfx] * wsc, 0, 1)[..., None]
    lampk = 1.0
    cool_k = 0.25 + 1.6 * k
    sharp = md['warm' + sfx] * lampk + md['cool' + sfx] * cool_k
    soft = md['warm_b' + sfx] * lampk + md['cool_b' + sfx] * cool_k
    rgb = sharp * (1 - dof) + soft * dof
    a = md['a' + sfx][..., None] * (1 - dof) + md['a_b' + sfx][..., None] * dof
    midl = np.dstack([rgb, a])
    far = _far_frame(lt)
    if blur_far is None:
        blur_far = 11.0 + 2.5 * (1 - breath)
    sig = blur_far / (2 * sf)
    if sig > 0.4:
        far = cv2.GaussianBlur(far, (0, 0), sig)
    sx, sy = shift
    out = _place(far, sf, XF0, YF0, cx + sx * DM / DF, cy + sy * DM / DF, 2, rect)
    m = _place(midl, sm, XM0, YM0, cx + sx, cy + sy, stepm, rect)
    if blur_mid > 0.3:
        m = blur_fast(m, blur_mid)
    out = out * (1 - m[..., 3:4]) + m[..., :3]
    # lamp: bright out-of-focus shade edge at top right, and the room glow
    hx, hy = 1840, 95
    gxs = (hx - 960) * sm + cx + sx - x0; gys = (hy - 540) * sm + cy + sy - y0
    _glow(out, gxs, gys, 210 * sm, WARM_L * 0.10)
    _glow(out, gxs, gys, 70 * sm, WARM_L * 0.22)
    if led:
        lx, ly, _ = _hero_proj(I['cam'], LED_P[None])
        b = 0.5 - 0.5 * math.cos(2 * math.pi * lt / 4.2)
        px_ = (lx[0] - 960) * sm + cx + sx - x0; py_ = (ly[0] - 540) * sm + cy + sy - y0
        _glow(out, px_, py_, 2.2 * sm + 0.6, AMBER * (0.9 * b + 0.05))
        _glow(out, px_, py_, 9 * sm + 1, AMBER * (0.05 * b))
    if dust and sm > 0.25:
        _dust(out, I['cam'], lt, sm, cx + sx - x0, cy + sy - y0)
    return out

def _glow(img, x, y, r, col):
    if r < 0.3: return
    R = int(r * 3) + 2
    h, w = img.shape[:2]
    xa, ya, xb, yb = max(int(x - R), 0), max(int(y - R), 0), min(int(x + R) + 1, w), min(int(y + R) + 1, h)
    if xb <= xa or yb <= ya: return
    gx = np.arange(xa, xb, dtype=np.float32)[None, :] - x; gy = np.arange(ya, yb, dtype=np.float32)[:, None] - y
    g = np.exp(-(gx * gx + gy * gy) / (r * r))
    img[ya:yb, xa:xb] += g[..., None] * np.asarray(col, np.float32)

@functools.lru_cache(None)
def _dust_seed():
    rng = np.random.default_rng(31)
    n = 46
    return dict(p=np.stack([rng.uniform(-0.20, 0.30, n), rng.uniform(-0.22, 0.08, n), rng.uniform(0.03, 0.34, n)], -1),
                f=rng.uniform(0.03, 0.09, (n, 3)), ph=rng.uniform(0, 6.28, (n, 3)), a=rng.uniform(0.004, 0.018, (n, 3)),
                b=rng.uniform(0.4, 1.0, n))

def _dust(img, cam, lt, sm, cx, cy):
    ds = _dust_seed()
    t = lt
    P = ds['p'] + ds['a'] * np.sin(2 * np.pi * ds['f'] * t + ds['ph'])
    P[:, 2] = 0.03 + np.mod(P[:, 2] - 0.004 * t - 0.03, 0.31)
    x, y, z = _hero_proj(cam, P)
    E = _lamp_irr(P, _n_last(LAMP_P - P))
    zf = 0.46
    for i in range(len(P)):
        if z[i] < 0.08: continue
        coc = abs(1 / z[i] - 1 / zf) * 9.0 + 0.9
        r = coc * sm
        bright = float(E[i]) * ds['b'][i] * 0.016 * (1.4 / max(coc, 1.4)) ** 2
        fade_edge = sstep(0.03, 0.06, P[i, 2]) * (1 - sstep(0.30, 0.34, P[i, 2]))
        if bright * fade_edge < 2e-4: continue
        _glow(img, (x[i] - 960) * sm + cx, (y[i] - 540) * sm + cy, r, WARM_L * bright * fade_edge * 1.0)

# =========================================================================================== window / pane
def _focus_params(lt):
    """Returns (sigma_exterior, sigma_interior_mid, sigma_interior_far_extra)."""
    rack = ease((lt - 6.55) / 1.7)
    s_ext = 9.0 * ease((lt - 6.5) / 1.6)
    s_int = (1 - rack) * (2.0 + 4.5 * sstep(4.5, 6.4, lt))
    return s_ext, s_int

@functools.lru_cache(None)
def _refl_tex():
    """What the glass reflects: the dark street behind the camera, two lamps, a tree line."""
    w, h = 900, 600
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    img = np.zeros((h, w, 3), np.float32)
    img += (np.array([0.010, 0.014, 0.030]) + np.array([0.03, 0.025, 0.022]) * np.clip((gy - 150) / 200, 0, 1)[..., None]).astype(np.float32)
    line = 330 + 30 * vnoise(gx / 40.0, 2 + 0 * gx, 1) + 18 * vnoise(gx / 13.0, 5 + 0 * gx, 2)
    img = np.where((gy > line)[..., None], np.float32(0.004), img)
    for (x, y, r, c) in ((180, 300, 26, SODIUM * 1.6), (700, 270, 20, SODIUM * 1.3), (420, 380, 6, HOMEW * 0.7), (560, 395, 5, HOMEW * 0.5)):
        g = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / r ** 2)
        img += g[..., None] * c
    rng = np.random.default_rng(4)
    sm = np.zeros((h, w), np.float32)
    for i in range(40):
        cv2.circle(sm, (int(rng.uniform(0, w)), int(rng.uniform(0, h))), int(rng.uniform(1, 3)), float(rng.uniform(0.3, 1)), -1)
    sm = cv2.GaussianBlur(sm, (0, 0), 0.8)
    return cv2.GaussianBlur(img, (0, 0), 3.0), sm

def pane_content(cam, lt, rect, pane_px, focus):
    """Everything seen inside the hero pane: room (2.5D), sheer curtains, reflection, dust on glass."""
    x0, y0, w, h = rect
    r = cam.r
    g = r - ZH                                 # distance to the glass (m, head-on approximation)
    delta = R_HERO - r
    hx, hy, hz = cam.proj(HPT[None])
    cx, cy = float(hx[0]), float(hy[0])
    s_ext, s_int = focus
    out = interior_view(delta, lt, cx, cy, rect, blur_mid=s_int, blur_far=11.0 + s_int * 1.5,
                        dust=g < 1.0, led=True)
    out *= 0.93
    # sheer curtains hanging 12 cm inside the glass at both sides
    dc = g + 0.12
    if dc > 0.05:
        gxx = np.arange(w, dtype=np.float32)[None, :] + x0
        gyy = np.arange(h, dtype=np.float32)[:, None] + y0
        Xw = (gxx - cx) * dc / cam.f            # metres in curtain plane
        Zw = -(gyy - cy) * dc / cam.f
        inner, outer = 0.30, 0.62
        ax = np.abs(Xw)
        cm = sstep_arr(inner - 0.02, inner + 0.03, ax) * (1 - sstep_arr(outer, outer + 0.05, ax)) * \
            sstep_arr(-0.80, -0.70, Zw) * (1 - sstep_arr(0.85, 0.95, Zw))
        folds = 0.65 + 0.35 * np.sin(Xw * 2 * np.pi / 0.075 + 1.3 * np.sin(Xw * 9)) ** 2
        lit = 0.35 + 0.65 * np.exp(-((Xw - 0.35) / 0.4) ** 2)
        col = WARM_L * 0.42 * (folds * lit)[..., None]
        a = (cm * 0.78)[..., None]
        cur = col * a
        sig = max(0.0, (s_ext * 0.6 + max(0, 1.2 - dc) * 6.0))
        if sig > 0.5:
            cur = blur_fast(cur.astype(np.float32), sig); a = blur_fast(np.ascontiguousarray(a[..., 0]), sig)[..., None]
        out = out * (1 - a) + cur
    # glass: reflection + dust specks, fading as we pass through
    if g > 0.02:
        rt, sm = _refl_tex()
        pw = max(pane_px[2], 4)
        sc = pw / rt.shape[1] * 1.6
        M = np.float32([[sc, 0, cx - rt.shape[1] * sc / 2 - x0 + (cam.C[0]) * 30], [0, sc, cy - rt.shape[0] * sc / 2 - y0]])
        refl = cv2.warpAffine(rt, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        k = 0.16 * sstep(0.05, 0.9, g)
        out += refl * k
        sc2 = pw / sm.shape[1]
        M2 = np.float32([[sc2, 0, pane_px[0] - x0], [0, sc2 * 1.0, pane_px[1] - y0]])
        sp = cv2.warpAffine(sm, M2, (w, h))
        sp = blur_fast(sp, s_ext * 0.8) if s_ext > 0.4 else sp
        out += sp[..., None] * WARM_L * 0.05 * sstep(0.05, 0.6, g)
        # warm scatter in the glass when far away (window reads as a glowing amber rectangle)
        out += WARM * 0.20 * sstep(2.0, 12.0, g)
    return out

@functools.lru_cache(None)
def _facade_tex():
    """Hero front wall: 10.2 x 5.9 m at 200 px/m. RGBA (premultiplied), pane hole transparent."""
    ppm = 200
    w, h = int(10.2 * ppm), int(5.9 * ppm)
    xs = -5.1 + (np.arange(w) + 0.5) / ppm; zs = 5.9 - (np.arange(h) + 0.5) / ppm
    X, Z = np.meshgrid(xs, zs)
    v = np.mod(Z, 0.18) / 0.18
    board = 0.78 + 0.22 * sstep_arr(0.0, 0.10, v) - 0.10 * sstep_arr(0.92, 1.0, v)
    grain = 1 + 0.04 * vnoise(X / 0.4, Z / 0.03, 1)
    alb = np.array([0.40, 0.43, 0.46], np.float32) * (board * grain)[..., None]
    # lighting: moon ambient + the street lamp across the road (front-left, low) + window spill
    Lp = np.array([-14.0, -17.5, 7.5])
    P = np.stack([X, np.zeros_like(X), Z], -1)
    Ld = Lp - P; d2 = (Ld ** 2).sum(-1); cosi = np.clip(-Ld[..., 1] / np.sqrt(d2), 0, 1)
    lamp = 9.0 * cosi / d2 * 0.10
    amb = 0.022
    spill = 0.07 * np.exp(-((X / 1.1) ** 2 + ((Z - WZ) / 1.0) ** 2)) + 0.10 * np.exp(-((X / 0.7) ** 2 + ((Z - (WZ - 0.95)) / 0.25) ** 2))
    light = np.array([0.55, 0.65, 1.0], np.float32) * amb + np.array([1.0, 0.68, 0.40], np.float32) * lamp[..., None] + WARM * spill[..., None]
    col = alb * light
    # trims around windows / door
    def rect(xa, xb, za, zb): return (X > xa) & (X < xb) & (Z > za) & (Z < zb)
    pane = rect(-PANE_W / 2, PANE_W / 2, WZ - PANE_H / 2, WZ + PANE_H / 2)
    trim = rect(-PANE_W / 2 - 0.11, PANE_W / 2 + 0.11, WZ - PANE_H / 2 - 0.14, WZ + PANE_H / 2 + 0.11) & ~pane
    sill = rect(-PANE_W / 2 - 0.16, PANE_W / 2 + 0.16, WZ - PANE_H / 2 - 0.16, WZ - PANE_H / 2 - 0.10)
    trim_col = np.array([0.62, 0.61, 0.58], np.float32) * light * 1.1
    trim_col = trim_col + WARM * 0.10 * np.exp(-((Z - (WZ - PANE_H / 2)) / 0.2) ** 2)[..., None]
    col = np.where(trim[..., None], trim_col, col)
    col = np.where(sill[..., None], trim_col * 1.25 + WARM * 0.06, col)
    # the inner edge of the trim: thin dark sash line
    sash = rect(-PANE_W / 2 - 0.025, PANE_W / 2 + 0.025, WZ - PANE_H / 2 - 0.025, WZ + PANE_H / 2 + 0.025) & ~pane
    col = np.where(sash[..., None], col * 0.35, col)
    for (xc, zc, ww, wh) in ((3.25, 4.575, 1.0, 1.3), (2.6, 1.6, 1.1, 1.35)):
        g = rect(xc - ww / 2, xc + ww / 2, zc - wh / 2, zc + wh / 2)
        t = rect(xc - ww / 2 - 0.1, xc + ww / 2 + 0.1, zc - wh / 2 - 0.12, zc + wh / 2 + 0.1) & ~g
        col = np.where(t[..., None], np.array([0.62, 0.61, 0.58], np.float32) * light, col)
        gl = np.array([0.006, 0.008, 0.016], np.float32) + np.array([0.02, 0.025, 0.04], np.float32) * np.clip((Z - (zc - wh / 2)) / wh, 0, 1)[..., None]
        col = np.where(g[..., None], gl, col)
    door = rect(-3.1, -2.1, 0.0, 2.15)
    col = np.where(door[..., None], np.array([0.02, 0.016, 0.014], np.float32), col)
    a = (~pane).astype(np.float32)
    a = cv2.GaussianBlur(a, (0, 0), 0.6)
    return np.dstack([col * a[..., None], a]).astype(np.float32), ppm

def _facade_H(cam):
    """Homography facade-texture px -> band px (None if not entirely in front of the camera)."""
    tex, ppm = _facade_tex()
    h, w = tex.shape[:2]
    corners_w = np.array([[-5.1, -0.03, 5.9], [5.1, -0.03, 5.9], [5.1, -0.03, 0.0], [-5.1, -0.03, 0.0]])
    c = cam.to_cam(corners_w)
    if (c[:, 2] <= 0.05).any(): return None
    x = PX + cam.f * c[:, 0] / c[:, 2]; y = PY + cam.f * c[:, 1] / c[:, 2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    return cv2.getPerspectiveTransform(src, np.float32(np.stack([x, y], -1)))

def hero_window(cam, img, zbuf, lt, focus):
    """Hero facade (textured) + the pane with the room behind it."""
    pc = np.array([[-PANE_W / 2, 0, WZ + PANE_H / 2], [PANE_W / 2, 0, WZ + PANE_H / 2],
                   [PANE_W / 2, 0, WZ - PANE_H / 2], [-PANE_W / 2, 0, WZ - PANE_H / 2]])
    c = cam.to_cam(pc)
    if (c[:, 2] <= 0.05).any(): return
    px_ = PX + cam.f * c[:, 0] / c[:, 2]; py_ = PY + cam.f * c[:, 1] / c[:, 2]
    pts = np.stack([px_, py_], -1)
    xa, xb = int(max(math.floor(px_.min()) - 2, 0)), int(min(math.ceil(px_.max()) + 2, W))
    ya, yb = int(max(math.floor(py_.min()) - 2, 0)), int(min(math.ceil(py_.max()) + 2, BH))
    pane_w = px_.max() - px_.min()
    if xb > xa and yb > ya and cam.r < 120:
        rect = (xa, ya, xb - xa, yb - ya)
        if pane_w > 26:
            content = pane_content(cam, lt, rect, (px_.min(), py_.min(), pane_w), focus)
        else:
            content = np.ones((yb - ya, xb - xa, 3), np.float32) * (WARM * 0.95)
        mask = np.zeros((yb - ya, xb - xa), np.float32)
        cv2.fillConvexPoly(mask, np.round((pts - [xa, ya]) * 16).astype(np.int32), 1.0, cv2.LINE_AA, 4)
        sub = img[ya:yb, xa:xb]
        sub[:] = sub * (1 - mask[..., None]) + content * mask[..., None]
    # facade texture over it
    if cam.r < 160:
        Hm = _facade_H(cam)
        if Hm is not None:
            tex, _ = _facade_tex()
            fa = cv2.warpPerspective(tex, Hm, (W, BH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            k = sstep(160, 100, cam.r)
            img[:] = img * (1 - fa[..., 3:4] * k) + fa[..., :3] * k

# =========================================================================================== lights + hood ground per frame
def lights_frame(cam, T, zbuf=None):
    ap = _allpts()
    x, y, z, e, vis = _project_pts(cam, ap)
    dist_all = np.sqrt(((ap['pos'] - cam.C) ** 2).sum(1)) if zbuf is not None else None
    kind = ap['kind']
    r = cam.r
    # LOD: coarse lights soften with distance (world-size footprint) and fade out near the city
    wc = sstep(7e3, 22e3, r)
    e = e * np.where(kind == 0, wc, 1.0)[:, None].astype(np.float32)
    e = np.where(vis[:, None], e, 0)
    if T < 8.6:
        fl = _flow()
        cid = fl['cid']
        part = cid >= 0
        k = cid[part]
        G = np.array([ease((T - fl['tk'][q]) / 2.6) for q in range(len(fl['C']))])
        app = np.array([sstep(fl['app'][q], fl['app'][q] + 1.0, T) for q in range(len(fl['C']))])
        tw = 1 + (0.20 * np.sin(2 * np.pi * 0.47 * T + fl['phs'][:, 0]) + 0.10 * np.sin(2 * np.pi * 1.13 * T + fl['phs'][:, 1])) * (1 - G)
        gi = np.clip((T - fl['tk'][k] - 0.45 - 0.9 * fl['rank'][part]) / 1.8, 0, 1)
        gi = gi * gi * gi * (gi * (gi * 6 - 15) + 10)
        Ck = fl['C'][k]
        qx = Ck[:, 0] + gi * (x[part] - Ck[:, 0]); qy = Ck[:, 1] + gi * (y[part] - Ck[:, 1])
        Gk = G[k]
        ang = fl['th0'] * (1 - Gk)
        sc = 1 + 0.10 * (1 - Gk)
        dx, dy = (qx - PX) * sc, (qy - PY) * sc
        ca, sa = np.cos(ang), np.sin(ang)
        x = x.copy(); y = y.copy()
        x[part] = PX + ca * dx - sa * dy + (1 - Gk) * fl['jit'][k, 0]
        y[part] = PY + sa * dx + ca * dy + (1 - Gk) * fl['jit'][k, 1]
        star_e = (fl['Ek'][k] * fl['share'][part] * app[k] * tw[k])[:, None] * np.array([1.0, 0.93, 0.85], np.float32)[None]
        hh = (gi ** 1.3)[:, None]
        e_part = star_e * (1 - hh) + e[part] * hh
        e = e.copy(); e[part] = e_part
        e[~part] *= sstep(7.6, 8.6, T)
        star_buf = np.zeros((BH, W, 3), np.float32)
        splat(star_buf, x[part], y[part], star_e * (1 - hh))
    # hood windows / porch lights as points (only while their quads are tiny)
    hood = _hood()
    wp, wr, wn = hood['win_pos'], hood['win_rgb'], hood['win_nrm']
    cw = cam.to_cam(wp)
    dw = np.linalg.norm(wp - cam.C, axis=1)
    vw = (cw[:, 2] > 0.5) & (np.einsum('ij,ij->i', cam.C[None] - wp, wn) > 0)
    cosv = np.clip(np.einsum('ij,ij->i', cam.C[None] - wp, wn) / np.maximum(dw, 1e-6), 0, 1)
    size = cam.f * 1.1 / np.maximum(dw, 1e-3)
    kq = np.clip((size - 1.2) / 2.0, 0, 1)
    kk = 1 - kq * kq * (3 - 2 * kq)
    ew = wr * ((cam.f / np.maximum(dw, 1e-3)) ** 2 * cosv * kk * vw)[:, None].astype(np.float32)
    xw = PX + cam.f * cw[:, 0] / np.maximum(cw[:, 2], 1e-3); yw = PY + cam.f * cw[:, 1] / np.maximum(cw[:, 2], 1e-3)
    xs_ = np.concatenate([x, xw]); ys_ = np.concatenate([y, yw]); es_ = np.concatenate([e, ew])
    zs_ = np.concatenate([dist_all, dw]) if zbuf is not None else None
    if zbuf is not None:
        xi = np.clip(xs_.astype(int), 0, W - 1); yi = np.clip(ys_.astype(int), 0, BH - 1)
        inb = (xs_ >= 0) & (xs_ < W) & (ys_ >= 0) & (ys_ < BH)
        occ = inb & (zs_ > zbuf[yi, xi] + 4.0)
        es_ = np.where(occ[:, None], 0, es_)
    E = np.zeros((BH, W, 3), np.float32)
    coarse = (np.arange(len(xs_)) < len(x)) & np.concatenate([kind == 0, np.zeros(len(xw), bool)])
    sig_c = cam.f * 350.0 / max(r, 1.0)
    if sig_c > 0.7 and wc > 0:
        Ec = np.zeros((BH, W, 3), np.float32)
        splat(Ec, xs_[coarse], ys_[coarse], es_[coarse])
        E += blur_fast(Ec, sig_c)
        splat(E, xs_[~coarse], ys_[~coarse], es_[~coarse])
    else:
        splat(E, xs_, ys_, es_)
    if T < 8.6:
        return E, star_buf
    return E, None

def ground_frame(cam, img):
    """Neighbourhood ground (lamp pools, roads, yards) via a homography of the ground texture."""
    tex = _hood()['ground']
    n = tex.shape[0]; px = 2 * GT_R / n
    # texture px (i, j) -> world (x, y, 0): x = -GT_R + (i+0.5)px, y = GT_R - (j+0.5)px
    A = np.array([[px, 0, -GT_R + 0.5 * px], [0, -px, GT_R - 0.5 * px], [0, 0, 0], [0, 0, 1.0]])
    Rt = np.concatenate([cam.R, (-cam.R @ cam.C)[:, None]], 1)
    K = np.array([[cam.f / 2, 0, PX / 2 - 0.25], [0, cam.f / 2, PY / 2 - 0.25], [0, 0, 1]])
    Hm = K @ Rt @ A
    hw, hh = W // 2, BH // 2
    g = cv2.warpPerspective(tex, Hm, (hw, hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    # discard pixels whose ray does not hit the ground plane in front of the camera
    d = _rays(cam, np.arange(hw) * 2 + 0.5, np.arange(hh) * 2 + 0.5)
    ok = (d[..., 2] < -1e-4) & ((cam.C[2] / np.maximum(-d[..., 2], 1e-4)) < 6000)
    g = g * ok[..., None]
    g = cv2.resize(g, (W, BH), interpolation=cv2.INTER_LINEAR)
    k = sstep(3200, 1500, cam.r)
    a = g[..., 3:4] * k
    img[:] = img * (1 - a) + g[..., :3] * k

# =========================================================================================== frame assembly
def exterior_frame(T, lt_push):
    cam = cam_state(T)
    earth_w = sstep(3.4, 7.6, T) if T < 8 else 1.0
    img = base_layer(cam, T, earth_w)
    if cam.r < 3300:
        ground_frame(cam, img)
    focus = _focus_params(lt_push)
    zbuf = None
    if cam.r < 2800:
        def hero_cb(im, zb):
            hero_window(cam, im, zb, lt_push, focus)
        zbuf = draw_hood(cam, img, None, hero_cb)
    E, Es = lights_frame(cam, T, zbuf)
    img += glare(E)
    if Es is not None:
        img += blur_fast(Es, 2.2) * 0.35 + blur_fast(Es, 7.0) * 0.12
    s_ext = focus[0]
    if s_ext > 0.4:
        img = blur_fast(img, s_ext)
        # keep the pane content sharp-edged where it covers the frame
    return img, cam

def interior_frame(lt):
    T = 11.0 + lt
    cam = cam_state(T)
    focus = _focus_params(lt)
    rect = (0, 0, W, BH)
    if cam.r > ZH:
        out = pane_content(cam, lt, rect, (PX - 2000, PY - 1200, 4000), focus)
    else:
        delta = R_HERO - cam.r
        # very slow drift + a gentle tilt up toward the window as dawn arrives
        sx = 14.0 * math.sin(lt * 0.11 + 0.5) - 6
        sy = 34.0 * ease((lt - 20.0) / 20.0)
        out = interior_view(delta, lt, PX, PY, rect, blur_mid=focus[1], blur_far=None, shift=(sx, sy))
    return out

def render(shot: str, lt: float, dur: float) -> np.ndarray:
    if shot == 'reveal':
        T = float(lt)
        lt_push = T - 11.0
    else:
        T = 11.0 + float(lt)
        lt_push = float(lt)
    if T < 0.45:
        out = np.empty((FH, W, 3), np.float32); out[:] = VOID
        return out
    cam = cam_state(T)
    if shot == 'pushin' and (cam.r - ZH) < GLASS_SWITCH:
        band = interior_frame(lt_push)
    else:
        band, cam = exterior_frame(T, lt_push)
    band = bloom(band, thresh=0.9, sigma=16, strength=0.35)
    out = np.empty((FH, W, 3), np.float32); out[:] = VOID
    out[Y0:Y0 + BH] = band
    return out
