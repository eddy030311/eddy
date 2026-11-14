"""《此刻 · ONLY NOW》 — the galaxy shots.

S4  'pullback'   : one window -> a point -> a spiral galaxy of window-lights; title at lt 9.0
S5  'drift'      : dolly through the same field; legible fragments; a 12-frame reflection plant
S9a 'extinction' : waves of extinction across the whole galaxy, to total darkness
S12 'relight'    : from the light in her palm, a new (different) field is born in three pulses

Every light is a conversation window.  Near: a small rounded glass pane with a thin glowing edge
and a few faint lines of text.  Far: a point.  Rendering is a small 3D point field with a
perspective camera, a mip-mapped window sprite set, and a layered depth-of-field compositor
(points/sprites are split between blur layers by circle of confusion, each layer blurred once at
low resolution, then collapsed like a Laplacian pyramid).

render(shot, lt, dur) is a pure function of (shot, lt); heavy data is built lazily per process.
"""
import math, os
import numpy as np, cv2
from PIL import Image, ImageDraw
from core import (W, H, TOP, BOT, CX, CY, AMBER, AMBER_HOT, VOID, clamp, lerp, sstep, ease,
                  ease_in, ease_out, fade, hexc, text_sprite, font)
import core
import ui

HERE = os.path.dirname(os.path.abspath(__file__))
FOC = 1600.0                       # focal length in px (horizontal FOV ~ 62 deg)

# ---------------------------------------------------------------- internal render band
# visible rows are 140..940; we render rows 108..972 plus a 16px guard on every side
PADF = 16
Y0, BH, BW = 108, 864, W
PH, PW = BH + 2 * PADF, BW + 2 * PADF          # 896 x 1952 (divisible by 8)
LAYERS = [(1, 0.0), (1, 1.5), (2, 3.6), (4, 8.5), (8, 19.0), (8, 44.0)]   # (downscale, blur sigma px)
SIG = np.array([s for _, s in LAYERS], np.float32)
DS = [d for d, _ in LAYERS]
LSH = [(PH // d, PW // d) for d, _ in LAYERS]
NL = len(LAYERS)


def _gb(a, s):
    return cv2.GaussianBlur(a, (0, 0), s) if s > 0.05 else a


def _up(a, k):
    return cv2.resize(a, (LSH[k][1], LSH[k][0]), interpolation=cv2.INTER_LINEAR)


def _coc_split(coc):
    c = np.clip(coc, 0, SIG[-1] - 1e-3)
    k = np.clip(np.searchsorted(SIG, c, side='right') - 1, 0, NL - 2)
    t = (c - SIG[k]) / (SIG[k + 1] - SIG[k])
    return k, t.astype(np.float32)


class Canvas:
    """Layered additive light buffer with depth of field."""

    def __init__(self):
        self.L = [np.zeros(s + (3,), np.float32) for s in LSH]
        self.raw = np.zeros((PH, PW, 3), np.float32)      # never blurred (hero, title, plant)
        self.glows = []                                    # (x, y, sigma_px, rgb) big soft lights

    # ---- points -----------------------------------------------------------------------
    def _splat(self, j, x, y, e):
        D = DS[j]; h, w = LSH[j]
        xl = (x + (PADF + 0.5)) / D - 0.5
        yl = (y - Y0 + (PADF + 0.5)) / D - 0.5
        x0 = np.floor(xl); y0 = np.floor(yl)
        fx = (xl - x0).astype(np.float32); fy = (yl - y0).astype(np.float32)
        x0 = x0.astype(np.int64); y0 = y0.astype(np.int64)
        ok = (x0 >= 0) & (x0 < w - 1) & (y0 >= 0) & (y0 < h - 1)
        if not ok.all():
            x0, y0, fx, fy, e = x0[ok], y0[ok], fx[ok], fy[ok], e[ok]
        if len(x0) == 0: return
        base = (y0 * w + x0) * 3
        idx = np.concatenate([base, base + 3, base + 3 * w, base + 3 * w + 3])
        wt = np.concatenate([(1 - fx) * (1 - fy), fx * (1 - fy), (1 - fx) * fy, fx * fy]) / (D * D)
        e4 = np.tile(e, (4, 1)) * wt[:, None]
        ii = np.concatenate([idx, idx + 1, idx + 2])
        ww = np.concatenate([e4[:, 0], e4[:, 1], e4[:, 2]])
        self.L[j] += np.bincount(ii, ww, minlength=h * w * 3).astype(np.float32).reshape(h, w, 3)

    def points(self, x, y, e, coc=None):
        """x,y frame coords; e (N,3) energy (sum over pixels); coc blur sigma in px."""
        if len(x) == 0: return
        if coc is None:
            self._splat(0, x, y, e); return
        if float(coc.max()) < 1e-6:
            self._splat(0, x, y, e); return
        k, t = _coc_split(coc)
        for j in range(NL):
            wj = np.where(k == j, 1 - t, 0) + np.where(k + 1 == j, t, 0)
            s = wj > 1e-3
            if s.any():
                self._splat(j, x[s], y[s], e[s] * wj[s, None])

    # ---- sprites ------------------------------------------------------------------------
    def sprite(self, j, arr, scale, x, y, Mc, raw=False):
        """Warp texture `arr` (h,w,3 channels) by `scale` (full-res px per texel), centre at frame
        (x,y), into layer j; channel mix Mc (3x3: rows = texture channels, cols = rgb)."""
        D = 1 if raw else DS[j]
        buf = self.raw if raw else self.L[j]
        h, w = buf.shape[:2]
        s = scale / D
        xl = (x + PADF + 0.5) / D - 0.5
        yl = (y - Y0 + PADF + 0.5) / D - 0.5
        ah, aw = arr.shape[:2]
        hw, hh = s * aw * 0.5 + 1.5, s * ah * 0.5 + 1.5
        x0, x1 = max(int(math.floor(xl - hw)), 0), min(int(math.ceil(xl + hw)) + 1, w)
        y0, y1 = max(int(math.floor(yl - hh)), 0), min(int(math.ceil(yl + hh)) + 1, h)
        if x1 <= x0 or y1 <= y0: return
        M = np.float32([[s, 0, xl - s * (aw - 1) * 0.5 - x0], [0, s, yl - s * (ah - 1) * 0.5 - y0]])
        p = cv2.warpAffine(arr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if Mc is None:
            buf[y0:y1, x0:x1] += p
        else:
            buf[y0:y1, x0:x1] += p @ Mc

    def glow(self, x, y, sigma, rgb):
        self.glows.append((x, y, sigma, np.asarray(rgb, np.float32)))

    # ---- compose --------------------------------------------------------------------------
    def compose(self, halo=0.0, halo_wide=0.0, sharp_soft=0.6):
        L = self.L
        acc = _gb(L[5], SIG[5] / 8) + _gb(L[4], SIG[4] / 8)
        s4 = None
        if halo > 0 or halo_wide > 0:
            s4 = cv2.resize(L[0], (LSH[3][1], LSH[3][0]), interpolation=cv2.INTER_AREA)
            if halo_wide > 0:
                s8 = cv2.resize(s4, (LSH[4][1], LSH[4][0]), interpolation=cv2.INTER_AREA)
                acc += _gb(s8, 5.5) * halo_wide
        self._glows(acc, 8, 28, 1e9)
        acc = _up(acc, 3) + _gb(L[3], SIG[3] / 4)
        if halo > 0:
            acc += _gb(s4, 2.2) * halo
        self._glows(acc, 4, 10, 28)
        acc = _up(acc, 2) + _gb(L[2], SIG[2] / 2)
        acc = _up(acc, 0)
        acc += _gb(L[1] + 0, SIG[1]) if L[1].any() else 0
        acc += _gb(L[0], sharp_soft)
        acc += self.raw
        self._glows(acc, 1, 0, 10)
        return acc[PADF:PADF + BH, PADF:PADF + BW]

    def _glows(self, acc, D, lo, hi):
        h, w = acc.shape[:2]
        for (x, y, sg, rgb) in self.glows:
            if not (lo <= sg < hi): continue
            sl = sg / D
            xl = (x + PADF + 0.5) / D - 0.5; yl = (y - Y0 + PADF + 0.5) / D - 0.5
            R = sl * 3.2 + 1
            xa, xb = max(int(xl - R), 0), min(int(xl + R) + 2, w)
            ya, yb = max(int(yl - R), 0), min(int(yl + R) + 2, h)
            if xb <= xa or yb <= ya: continue
            gx = np.arange(xa, xb, dtype=np.float32) - xl
            gy = np.arange(ya, yb, dtype=np.float32) - yl
            g = np.exp(-(gy[:, None] ** 2 + gx[None, :] ** 2) / (2 * sl * sl))
            acc[ya:yb, xa:xb] += g[..., None] * rgb      # rgb is peak radiance


def _finish(band, bloom_amt=1.0, thresh=0.7):
    if bloom_amt > 0:
        band = core.bloom(band, thresh=thresh, sigma=22, strength=0.55 * bloom_amt, scale=4)
    out = np.empty((H, W, 3), np.float32)
    out[:] = VOID
    out[Y0:Y0 + BH] += band
    return out


def _void():
    out = np.empty((H, W, 3), np.float32); out[:] = VOID
    return out


# ---------------------------------------------------------------- camera
def _look(pos, target, up_hint):
    f = np.asarray(target, np.float64) - pos; f /= np.linalg.norm(f)
    r = np.cross(f, up_hint); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.asarray(pos, np.float64), np.stack([r, u, f])


def _rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float64)


def _to_gal(pos, R, theta):
    """Camera given in world frame where the galaxy is rotated by theta -> camera in galaxy frame."""
    Rz = _rotz(theta)
    return Rz.T @ pos, R @ Rz


def _project(P, pos, R, cy=540.0):
    Pc = (P - pos.astype(np.float32)) @ R.T.astype(np.float32)
    z = Pc[:, 2]
    inv = 1.0 / np.maximum(z, 1e-3)
    x = 960.0 + FOC * Pc[:, 0] * inv
    y = cy - FOC * Pc[:, 1] * inv
    return x, y, z, inv


# ---------------------------------------------------------------- window textures
LW = [8, 16, 32, 64, 128, 256, 512]
TXT_H = ["can you check this?", "我需要一个名字", "おはよう", "merci", "def solve(", "is this normal?",
         "help me say sorry", "why is the sky blue?", "¿me ayudas?", "잘 자요", "привет, ты тут?",
         "one more try", "我想写一封信", "眠れない", "draft 3. be honest", "grazie mille",
         "SELECT name FROM", "it works!!", "how do I start", "Ich verstehe nicht",
         "for i in range(n):", "tell me a story", "obrigada", "是这样吗？", "on recommence ?",
         "what if it fails", "can you stay?", "let's try again", "ok. thank you", "早上好",
         "ありがとう", "kiitos", "import numpy as np", "is it too late?", "she said yes", "明天见"]
TXT_A = ["of course.", "let's look.", "I'm here.", "好的，我们来看看。", "try this:", "that's okay.",
         "もちろん。", "bien sûr.", "here's a start:", "take your time.", "claro.", "我在。",
         "one step at a time.", "    return a + b", "yes — and", "good question.", "慢慢来。",
         "    if x > 0:", "you can.", "tell me more."]
CODEISH = ("def ", "for ", "SELECT", "import", "    ")


def _rr(w, h, r, S, pad, fill=None, outline=None, width=1):
    im = Image.new('L', (int((w + 2 * pad) * S), int((h + 2 * pad) * S)), 0)
    d = ImageDraw.Draw(im)
    box = (pad * S, pad * S, (pad + w) * S - 1, (pad + h) * S - 1)
    d.rounded_rectangle(box, radius=max(1, r * S), fill=fill, outline=outline, width=max(1, int(round(width * S))))
    return im


def _down(im, S):
    w, h = im.size
    return np.asarray(im.resize((w // S, h // S), Image.BOX), np.float32) / 255


def _build_tex(var):
    rng = np.random.default_rng(9100 + var)
    aspect = float(rng.choice([0.40, 0.46, 0.52, 0.58, 0.64]))
    nl = int(rng.integers(2, 5)) if aspect > 0.5 else int(rng.integers(2, 4))
    first_h = rng.random() < 0.65
    lines = []
    for k in range(nl):
        human = (k % 2 == 0) == first_h
        pool = TXT_H if human else TXT_A
        lines.append((pool[int(rng.integers(len(pool)))], human))
    caret_line = nl - 1
    levels = []
    for wl in LW:
        hl = max(3, int(round(wl * aspect)))
        pad = int(math.ceil(wl * 0.12)) + 2
        S = 8 if wl <= 32 else (4 if wl <= 128 else 2)
        r = max(0.9, 0.075 * hl)
        fill = _down(_rr(wl, hl, r, S, pad, fill=255), S)
        ew = max(1.0, wl / 330.0)
        edge = _down(_rr(wl, hl, r, S, pad, outline=255, width=ew), S)
        glow = cv2.GaussianBlur(fill, (0, 0), max(0.8, 0.045 * wl))
        ch, cw = fill.shape
        grad = np.linspace(1.0, 0.55, ch, dtype=np.float32)[:, None]
        # text
        txt = np.zeros((ch, cw), np.float32); bub = np.zeros((ch, cw), np.float32)
        fs = 0.072 * wl
        top = pad + 0.20 * hl
        step = min(0.155 * wl, (hl * 0.70) / max(nl - 0.4, 1))
        for k, (s, human) in enumerate(lines):
            yc = top + k * step + fs * 0.5
            if yc + fs * 0.6 > pad + hl - 0.08 * hl: break
            fname = 'mono_l' if s.startswith(CODEISH) else 'sans_l'
            if wl >= 256:
                fnt = font(fname, max(6, int(round(fs))))
                tw = fnt.getlength(s)
                maxw = 0.80 * wl
                if tw > maxw:
                    fnt = font(fname, max(6, int(round(fs * maxw / tw)))); tw = fnt.getlength(s)
                im = Image.new('L', (cw, ch), 0); d = ImageDraw.Draw(im)
                x = pad + wl - 0.07 * wl - tw if human else pad + 0.07 * wl
                asc, desc = fnt.getmetrics()
                d.text((x, yc - (asc + desc) * 0.55), s, font=fnt, fill=255)
                txt = np.maximum(txt, np.asarray(im, np.float32) / 255)
                if human:
                    bh = fs * 1.55
                    b = _down(_rr(tw + 0.05 * wl, bh, bh * 0.45, 2, 0), 2)
                    bx, by = int(round(x - 0.025 * wl)), int(round(yc - bh / 2))
                    hb, wb = b.shape
                    bub[by:by + hb, bx:bx + wb] = np.maximum(bub[by:by + hb, bx:bx + wb], b[:ch - by, :cw - bx])
            else:
                f512 = font(fname, 37)
                tw = min(f512.getlength(s) * wl / 512.0, 0.80 * wl)
                th = max(0.5, fs * 0.30)
                im = Image.new('L', (cw * S, ch * S), 0); d = ImageDraw.Draw(im)
                x = pad + wl - 0.07 * wl - tw if human else pad + 0.07 * wl
                # word segments
                words = s.split(' '); tot = sum(len(q) for q in words) + len(words) - 1
                cx = x
                for q in words:
                    ww = tw * len(q) / max(tot, 1)
                    d.rectangle((cx * S, (yc - th / 2) * S, (cx + ww) * S, (yc + th / 2) * S), fill=150)
                    cx += ww + tw / max(tot, 1)
                txt = np.maximum(txt, _down(im, S))
        E = edge * 0.70 + glow * 0.17
        F = fill * grad + bub * 0.9
        T = txt * fill
        tex = np.ascontiguousarray(np.stack([E, F, T], -1).astype(np.float32))
        eps = tex.reshape(-1, 3).sum(0) / (wl * hl)
        levels.append((tex, wl, eps))
    return dict(aspect=aspect, levels=levels, eps=np.stack([l[2] for l in levels]))


NVAR = 36
_TEX = None


def textures():
    global _TEX
    if _TEX is None:
        _TEX = [_build_tex(v) for v in range(NVAR)]
    return _TEX


LOGLW = np.log2(np.array(LW, np.float64))


def _eps_interp(eps, w):
    """Per-area channel energy for a window drawn at full-res width w (smooth in w)."""
    lw = math.log2(max(w, 1e-6))
    if lw <= LOGLW[0]: return eps[0]
    if lw >= LOGLW[-1]: return eps[-1]
    L = int(np.searchsorted(LOGLW, lw))
    u = (LOGLW[L] - lw)
    return eps[L] ** (1 - u) * eps[L - 1] ** u


def _level_for(w):
    for i, l in enumerate(LW):
        if l >= w: return i
    return len(LW) - 1


# ---------------------------------------------------------------- palettes
PAL = {
    'amber': AMBER, 'hot': AMBER_HOT,
    'cool': hexc('#BCD0FF'), 'rose': hexc('#F2B3BE'), 'green': hexc('#C9EBC2'),
}
WHITE_W = np.array([1.0, 0.96, 0.90], np.float32)
KF, KT = 0.016, 0.30


def _mc(col, I):
    tc = col * 0.45 + WHITE_W * 0.55
    return np.stack([col * I, (col * 0.8 + 0.05) * KF * I, tc * KT * I]).astype(np.float32)


def _colours(rng, n, p_cool, p_rose, p_green, p_hot):
    u = rng.random(n)
    cols = np.empty((n, 3), np.float32)
    names = np.array(['amber'] * n, dtype=object)
    edges = np.cumsum([p_cool, p_rose, p_green, p_hot])
    cols[:] = PAL['amber']
    for name, lo, hi in (('cool', 0, edges[0]), ('rose', edges[0], edges[1]),
                         ('green', edges[1], edges[2]), ('hot', edges[2], edges[3])):
        m = (u >= lo) & (u < hi)
        cols[m] = PAL[name]
    # small per-light temperature jitter: each room is its own
    cols *= (1 + rng.normal(0, 0.05, (n, 3))).astype(np.float32)
    return np.clip(cols, 0, 1.2)


class Field:
    def __init__(self, P, s, var, col, I, local=None, phase=None):
        tex = textures()
        self.P = P.astype(np.float32)
        self.s = s.astype(np.float32)
        self.var = var.astype(np.int32)
        self.col = col.astype(np.float32)
        I = 2.0 * np.tanh(np.asarray(I, np.float64) / 2.0)          # soft cap: no blown-out rooms
        self.I = I.astype(np.float32)
        self.N = len(P)
        self.aspect = np.array([tex[v]['aspect'] for v in range(NVAR)], np.float32)[self.var]
        eps0 = np.stack([tex[v]['eps'][0] for v in range(NVAR)])[self.var]          # (N,3) per-area @ 8px
        tc = self.col * 0.45 + WHITE_W * 0.55
        self.E8 = (eps0[:, 0:1] * self.col + eps0[:, 1:2] * (self.col * 0.8 + 0.05) * KF +
                   eps0[:, 2:3] * tc * KT) * self.I[:, None] * self.aspect[:, None]
        self.local = local if local is not None else np.zeros(self.N, bool)
        rng = np.random.default_rng(5)
        self.u = rng.random(self.N).astype(np.float32)
        self.phase = rng.random(self.N).astype(np.float32) * 6.283


# ---------------------------------------------------------------- the galaxy
OMEGA = 0.0055                     # rad/s rigid rotation
R0_ARM, PITCH = 70.0, math.radians(18.5)


def _arm_theta(r):
    return np.log(np.maximum(r, 1.0) / R0_ARM) / math.tan(PITCH)


HERO_R, HERO_Z = 430.0, 105.0
_GAL = None


def _clusters(rng, cc, nm, sc, flat=0.45):
    tot = int(nm.sum())
    ci = np.repeat(np.arange(len(cc)), nm)
    P = cc[ci] + rng.normal(0, 1, (tot, 3)) * np.stack([sc[ci], sc[ci], sc[ci] * flat], 1)
    return P, ci


def galaxy():
    global _GAL
    if _GAL is not None: return _GAL
    rng = np.random.default_rng(20251114)
    Ps, cols, Is, loc = [], [], [], []

    def add(P, col, I, local=False):
        Ps.append(np.asarray(P, np.float64)); cols.append(col); Is.append(np.asarray(I, np.float64))
        loc.append(np.full(len(P), local))

    tp = math.tan(PITCH)
    # --- arm knots (two grand-design arms, two faint secondary arms)
    NC = 5600
    rc = 70 + rng.gamma(1.6, 190, NC); rc = rc[rc < 1080]; NC = len(rc)
    arm = rng.integers(0, 4, NC)
    off = np.where(arm < 2, arm * math.pi, arm * math.pi + math.pi / 2 + 0.35)
    strong = arm < 2
    wid = np.where(strong, 0.10, 0.2) * (1 + 0.5 * np.sin(rc / 90.0 + arm))
    th = _arm_theta(rc) + off + rng.normal(0, 1, NC) * wid
    cc = np.stack([rc * np.cos(th), rc * np.sin(th), rng.normal(0, 5, NC)], 1)
    nm = (np.clip(rng.lognormal(2.9, 0.8, NC), 3, 260) * np.where(strong, 1, 0.45)).astype(int)
    P, ci = _clusters(rng, cc, nm, rng.uniform(3, 15, NC))
    young = rng.random(NC) < 0.3
    tot = len(P)
    col = _colours(rng, tot, 0.12, 0.06, 0.05, 0.12)
    yb = young[ci] & (rng.random(tot) < 0.45)
    col[yb] = PAL['cool'] * (1 + rng.normal(0, 0.04, (yb.sum(), 3)))
    add(P, col, rng.lognormal(0, 0.6, tot) * np.where(young[ci], 1.4, 1.0))
    # --- spurs / feathers branching outward from the main arms
    NS = 70
    rr = rng.uniform(160, 820, NS); a2 = rng.integers(0, 2, NS)
    th0 = _arm_theta(rr) + a2 * math.pi
    root = np.stack([rr * np.cos(th0), rr * np.sin(th0)], 1)
    tg = np.stack([np.cos(th0) - np.sin(th0) / tp, np.sin(th0) + np.cos(th0) / tp], 1)
    tg /= np.linalg.norm(tg, axis=1, keepdims=True)
    ang = rng.uniform(0.45, 0.85, NS) * rng.choice([-1, 1], NS)
    ca, sa = np.cos(ang), np.sin(ang)
    dirs = np.stack([tg[:, 0] * ca - tg[:, 1] * sa, tg[:, 0] * sa + tg[:, 1] * ca], 1)
    L = rng.uniform(60, 220, NS)
    k = 14
    sfrac = rng.random((NS, k)) ** 0.8
    cc = (root[:, None, :] + dirs[:, None, :] * (L[:, None] * sfrac)[..., None]).reshape(-1, 2)
    cc = np.concatenate([cc, rng.normal(0, 4, (len(cc), 1))], 1)
    nm = np.clip(rng.lognormal(2.3, 0.7, len(cc)), 2, 80).astype(int)
    P, ci = _clusters(rng, cc, nm, rng.uniform(3, 9, len(cc)))
    add(P, _colours(rng, len(P), 0.16, 0.07, 0.05, 0.1), rng.lognormal(-0.2, 0.6, len(P)))
    # --- smooth arm population
    n = 90000
    r = 60 + rng.gamma(1.5, 230, n); r = r[r < 1150]; n = len(r)
    a2 = rng.integers(0, 2, n)
    th = _arm_theta(r) + a2 * math.pi + rng.normal(0, 0.24, n) * (1 + 0.4 * np.sin(r / 70.0))
    z = rng.normal(0, 9, n) * (1 + r / 900)
    add(np.stack([r * np.cos(th), r * np.sin(th), z], 1), _colours(rng, n, 0.09, 0.06, 0.04, 0.1),
        rng.lognormal(-0.25, 0.6, n))
    # --- inter-arm disc (the diffuse light between the arms)
    n = 60000
    r = 40 + rng.gamma(1.4, 260, n); r = r[r < 1250]; n = len(r)
    th = rng.uniform(0, 2 * math.pi, n)
    add(np.stack([r * np.cos(th), r * np.sin(th), rng.normal(0, 13, n)], 1),
        _colours(rng, n, 0.07, 0.06, 0.04, 0.08), rng.lognormal(-0.55, 0.5, n))
    # --- bulge with a soft bar along theta=0 (the arms start at its ends)
    n = 38000
    u = rng.random(n)
    rr = np.minimum(46.0 / np.sqrt(np.clip(u, 1e-4, 0.999) ** (-2 / 3) - 1), 420)
    d = rng.normal(0, 1, (n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    add(d * rr[:, None] * np.array([1.3, 0.85, 0.5]), _colours(rng, n, 0.03, 0.05, 0.01, 0.35),
        rng.lognormal(-1.0, 0.45, n))
    # --- sparse halo of far rooms
    n = 7000
    rr = 150 + rng.gamma(2, 300, n)
    d = rng.normal(0, 1, (n, 3)); d /= np.linalg.norm(d, axis=1, keepdims=True)
    add(d * rr[:, None] * np.array([1, 1, 0.55]), _colours(rng, n, 0.15, 0.08, 0.06, 0.05),
        rng.lognormal(-0.5, 0.5, n))
    # --- the hero's neighbourhood: a loose sheet of rooms drifting above the arm (S4 start, S5).
    #     Its long axis points roughly at the galactic core so S5 can look down the arm to the heart.
    th_h = float(_arm_theta(np.array([HERO_R]))[0])
    hero = np.array([HERO_R * math.cos(th_h), HERO_R * math.sin(th_h), HERO_Z])
    tang = np.array([math.cos(th_h) - math.sin(th_h) / tp, math.sin(th_h) + math.cos(th_h) / tp, 0.0])
    tang /= np.linalg.norm(tang)
    to_core = -np.array([hero[0], hero[1], 0.0]); to_core /= np.linalg.norm(to_core)
    b_ = math.radians(24)
    pd = to_core * math.cos(b_) - tang * math.sin(b_); pd /= np.linalg.norm(pd)
    side = np.cross([0, 0, 1.0], pd)
    zz = np.array([0, 0, 1.0])
    n = 170
    along = rng.uniform(-80, -20, n)
    sd = rng.normal(0, 1, n) * 9
    upz = rng.normal(0, 1, n) * 2.6
    P = hero + along[:, None] * pd + sd[:, None] * side + upz[:, None] * zz
    add(P, _colours(rng, n, 0.12, 0.08, 0.06, 0.12), rng.lognormal(-0.3, 0.5, n), local=True)
    # the hero's nearest neighbours: the first tens to appear around it in S4
    n = 90
    rr = 1.3 + rng.gamma(1.4, 5.0, n)
    an = rng.uniform(0, 2 * math.pi, n)
    dz = -(1.5 + np.abs(rng.normal(0, 1, n)) * (2 + 0.5 * rr))
    P = hero + np.stack([rr * np.cos(an), rr * np.sin(an) * 0.75, dz], 1)
    add(P, _colours(rng, n, 0.14, 0.1, 0.08, 0.14), rng.lognormal(0.0, 0.4, n), local=True)
    # the lane the S5 camera travels: rooms placed to come through focus one after another
    c0, c1 = _drift_path(hero, pd, side)
    k = np.arange(13)
    al = 3.2 + 1.85 * k + rng.uniform(-0.4, 0.4, 13)
    sg = np.where(k % 2 == 0, 1, -1) * np.where(k % 5 == 3, -1, 1)
    lat = sg * rng.uniform(0.95, 2.0, 13)
    vz = np.where((k // 2) % 2 == 0, 1, -1) * rng.uniform(0.15, 0.7, 13)
    P = c0 + al[:, None] * pd + lat[:, None] * side + vz[:, None] * zz
    add(P, _colours(rng, 13, 0.18, 0.12, 0.08, 0.14), rng.uniform(1.0, 1.35, 13), local=True)
    n = 40
    al = rng.uniform(1.0, 30, n)
    lat = rng.choice([-1, 1], n) * rng.uniform(2.6, 9, n)
    vz = rng.uniform(-2.4, 2.4, n)
    P = c0 + al[:, None] * pd + lat[:, None] * side + vz[:, None] * zz
    add(P, _colours(rng, n, 0.14, 0.1, 0.08, 0.14), rng.lognormal(-0.1, 0.4, n), local=True)
    tang = pd

    P = np.concatenate(Ps)
    col = np.concatenate(cols); I = np.concatenate(Is); local = np.concatenate(loc)
    # dust lanes on the inner (concave) edge of the two main arms
    r = np.hypot(P[:, 0], P[:, 1])
    dlt = np.mod(np.arctan2(P[:, 1], P[:, 0]) - _arm_theta(r) + math.pi / 2, math.pi) - math.pi / 2
    lane = np.exp(-((dlt - 0.15) / 0.055) ** 2) * sstep_v(90, 200, r) * (1 - sstep_v(700, 1000, r))
    I = I * (1 - 0.8 * lane * ~local)
    # keep the hero's line of sight clear (the pull-back starts face-on, then tilts)
    rel = P - hero
    rxy = np.hypot(rel[:, 0], rel[:, 1])
    keep = np.linalg.norm(rel, axis=1) > 1.6
    keep &= ~((rel[:, 2] > 0) & (rxy < 0.9 + 0.3 * rel[:, 2]) & (rel[:, 2] < 80))
    # keep the S5 dolly path clear
    seg = c1 - c0; L2 = seg @ seg
    tt = np.clip((P - c0) @ seg / L2, -0.4, 1.4)
    keep &= np.linalg.norm(P - (c0 + tt[:, None] * seg), axis=1) > 1.4
    P, col, I, local = P[keep], col[keep], I[keep], local[keep]
    # the hero is a room of the galaxy too (index 0), drawn specially in S4
    P = np.concatenate([hero[None], P]); col = np.concatenate([AMBER[None], col])
    I = np.concatenate([[1.0], I]); local = np.concatenate([[True], local])
    N = len(P)
    s = np.clip(rng.lognormal(math.log(0.72), 0.22, N), 0.42, 1.35); s[0] = 1.0
    var = rng.integers(0, NVAR, N)
    G = Field(P, s, var, col, I, local)
    G.hero = hero; G.tang = tang; G.side = side
    G.path = (c0, c1)
    rg = np.hypot(G.P[:, 0], G.P[:, 1])
    G.coremask = (1 - sstep_v(90, 300, rg)).astype(np.float32)
    _GAL = G
    return G


def _drift_path(hero, pd, side):
    up = np.array([0, 0, 1.0])
    c0 = hero - 46 * pd + 40.0 * side + 0.3 * up
    c1 = c0 + 8.0 * pd - 0.4 * side
    return c0, c1


# ---------------------------------------------------------------- field renderer
def draw_field(cv, G, pos, R, *, vis, expo, focus, kn, kf, wmin, cy=540.0, skip=None,
               max_sprites=2200, breathe=None, sprite_mul=None, want_xy=False, auto=None, shutter=None):
    """auto=(target_mean, strength): exposure partly normalised by the light actually in frame."""
    x, y, z, inv = _project(G.P, pos, R, cy)
    w = FOC * G.s * inv
    m = (z > 0.05) & (x > -w - 40) & (x < W + w + 40) & (y > Y0 - 40 - w) & (y < Y0 + BH + 40 + w)
    if vis is not None:
        m &= vis > 1e-4
    if skip is not None:
        m[skip] = False
    idx = np.nonzero(m)[0]
    xi, yi, zi, wi = x[idx], y[idx], z[idx], w[idx]
    vi = vis[idx] if vis is not None else 1.0
    gain = expo * vi
    if breathe is not None:
        gain = gain * (1 + 0.09 * np.sin(breathe * 0.9 + G.phase[idx]))
    coc = None
    if kn > 0 or kf > 0:
        q = focus / zi
        coc = kn * np.maximum(q - 1, 0) + kf * np.maximum(1 - q, 0)
    ws = sstep_v(wmin, wmin * 1.45, wi)
    if auto is not None:
        vv = vis[idx] if vis is not None else 1.0
        tot = float(((G.E8[idx].sum(1) / 3) * wi ** 2 * vv).sum()) / (BW * BH) * expo
        k = (auto[0] / max(tot, 1e-6)) ** auto[1]
        gain = gain * k
        G.last_expo = expo * k
    # a far window drawn as a point still has a size: spread it (keeps it from becoming a needle)
    coc_dof = coc
    psz = 0.32 * wi
    psz = np.maximum(psz - 0.4, 0) * 1.25
    coc = psz if coc is None else np.maximum(coc, psz)
    # points
    pm = ws < 0.999
    if pm.any():
        g = gain if np.isscalar(gain) else gain[pm]
        if not shutter:
            e = G.E8[idx[pm]] * (wi[pm] ** 2 * (1 - ws[pm]) * g)[:, None]
            cv.points(xi[pm], yi[pm], e, None if coc is None else coc[pm])
        else:
            # motion blur: the same lights, seen from cameras spread across a 180-degree shutter
            ip = idx[pm]; base = G.E8[ip] * ((1 - ws[pm]) * g / len(shutter))[:, None]
            Pp = G.P[ip]
            cpm = None if coc is None else coc[pm]
            XS, YS, ES, CS = [], [], [], []
            sp_ = G.s[ip]
            for (sp, sR) in shutter:
                xs, ys, zs, invs = _project(Pp, sp, sR, cy)
                ok = (zs > 0.05) & (xs > -8) & (xs < W + 8) & (ys > Y0 - 8) & (ys < Y0 + BH + 8)
                ws_ = (FOC * sp_[ok] * invs[ok]) ** 2
                XS.append(xs[ok]); YS.append(ys[ok]); ES.append(base[ok] * ws_[:, None])
                if cpm is not None: CS.append(cpm[ok])
            cv.points(np.concatenate(XS), np.concatenate(YS), np.concatenate(ES),
                      None if cpm is None else np.concatenate(CS))
    # sprites
    sm = np.nonzero(ws > 0.001)[0]
    if len(sm) > max_sprites:
        sm = sm[np.argsort(-wi[sm])[:max_sprites]]
    tex = textures()
    for k in sm:
        i = idx[k]
        g = (gain if np.isscalar(gain) else gain[k]) * ws[k]
        if sprite_mul is not None: g *= sprite_mul(i)
        draw_window(cv, tex[G.var[i]], xi[k], yi[k], wi[k], G.col[i], G.I[i] * g,
                    0.0 if coc_dof is None else coc_dof[k])
    if want_xy:
        return x, y, z, w
    return None


def sstep_v(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def draw_window(cv, T, x, y, wpx, col, I, coc, raw=False):
    eps_t = _eps_interp(T['eps'], wpx)
    Mc0 = _mc(col, I)
    if raw:
        lv = _level_for(wpx); tex, wl, eps = T['levels'][lv]
        cv.sprite(0, tex, wpx / wl, x, y, Mc0 * (eps_t / np.maximum(eps, 1e-9))[:, None], raw=True)
        return
    c = min(max(coc, 0.0), SIG[-1] - 1e-3)
    k = min(int(np.searchsorted(SIG, c, side='right') - 1), NL - 2)
    t = (c - SIG[k]) / (SIG[k + 1] - SIG[k])
    for j, wj in ((k, 1 - t), (k + 1, t)):
        if wj < 0.01: continue
        D = DS[j]
        lv = _level_for(wpx / D)
        tex, wl, eps = T['levels'][lv]
        Mc = Mc0 * (eps_t / np.maximum(eps, 1e-9))[:, None] * wj
        cv.sprite(j, tex, wpx / wl, x, y, Mc)


# ---------------------------------------------------------------- S4 pullback
HERO_LINES = [dict(role='human', text='are you there?'), dict(role='ai', text="I'm here.")]
D0 = FOC * 1.0 / 880.0
D1 = 1700.0
_HEROTEX = None


def hero_tex():
    """The hero window rendered by ui.window at full size, caret off / on, as additive deltas + mips."""
    global _HEROTEX
    if _HEROTEX is not None: return _HEROTEX
    out = []
    for ca in (0.0, 1.0):
        img = np.empty((H, W, 3), np.float32); img[:] = VOID
        ui.window(img, 960, 540, 880, 300, HERO_LINES, t=27.25, cursor=dict(rate=1.0, alpha=ca), alpha=1.0)
        d = img[540 - 230:540 + 231, 960 - 520:960 + 521] - VOID
        out.append(np.ascontiguousarray(d))
    lev = []
    a, b = out
    m = np.zeros(a.shape[:2], np.float32)
    m[80:380, 80:960] = ui.rounded_rect_mask(880, 300, 12)
    m = np.repeat(m[..., None], 3, 2)
    wl = 880.0
    while True:
        lev.append((a, b, m, wl))
        if wl < 6: break
        sz = ((a.shape[1] + 1) // 2, (a.shape[0] + 1) // 2)
        a, b, m = [cv2.resize(q, sz, interpolation=cv2.INTER_AREA) for q in (a, b, m)]
        wl = wl * a.shape[1] / lev[-1][0].shape[1]
    _HEROTEX = lev
    return lev


def _pull_profile():
    """log-distance vs time: slow start, accelerating, landing just before the hit at 9.0."""
    t = np.linspace(0, 17, 4081)
    v = np.where(t < 9.3, (np.clip(t / 7.0, 0, 1) ** 1.8) * (1 - sstep_v(7.0, 9.3, t)) ** 1.0, 0.0)
    v = v + 0.006 * sstep_v(0.0, 1.5, t)
    cum = np.concatenate([[0], np.cumsum((v[1:] + v[:-1]) * 0.5 * np.diff(t))])
    k = cum[np.searchsorted(t, 9.0)]
    tail = 0.012 * sstep_v(8.6, 10.5, t) * (t - 8.6).clip(0)
    lnd = math.log(D0) + (math.log(D1) - math.log(D0)) * cum / k
    lnd = lnd + tail
    return t, lnd


_PP = None


def pull_logd(lt):
    global _PP
    if _PP is None: _PP = _pull_profile()
    return float(np.interp(lt, *_PP))


AZ_P = -2.2
INC_P = math.radians(57)


def pull_cam(lt):
    G = galaxy()
    lnd = pull_logd(lt)
    d = math.exp(lnd)
    w = (lnd - math.log(D0)) / (math.log(D1) - math.log(D0))
    wc = min(max(w, 0.0), 1.0)
    theta = OMEGA * lt
    inc = INC_P * ease(min(1.0, wc ** 1.15 * 1.08))
    az = AZ_P + 0.42 * ease(wc) + 0.004 * lt
    hero_w = _rotz(theta) @ G.hero
    tgt = hero_w * (1 - sstep(0.42, 0.93, wc))
    n = np.array([math.sin(inc) * math.cos(az), math.sin(inc) * math.sin(az), math.cos(inc)])
    pos = tgt + d * n
    upv = np.array([-math.cos(az), -math.sin(az), 0.0])
    pos, R = _look(pos, tgt, upv)
    return pos, R, d, wc, theta, hero_w


def _pull_title():
    """Title sprite, its blur stack, dissolve noise and particle seeds."""
    zh = text_sprite('此刻', 'serif_xl', 150, (1.0, 0.93, 0.82), tracking=28)
    en = text_sprite('ONLY NOW', 'sans_l', 25, (0.98, 0.88, 0.74), tracking=31)
    bw, bh = 820, 380
    rgb = np.zeros((bh, bw, 3), np.float32)
    core.blit(rgb, zh, bw / 2 + 14, 150, anchor='c')
    core.blit(rgb, en, bw / 2 + 15, 282, anchor='c')
    lum = rgb.max(-1)
    rgb[..., :] *= 1.0
    sig = [0, 2.0, 5.0, 12.0, 28.0]
    stack = [rgb] + [cv2.GaussianBlur(rgb, (0, 0), s) for s in sig[1:]]
    rng = np.random.default_rng(31)
    nz = cv2.GaussianBlur(rng.random((bh, bw)).astype(np.float32), (0, 0), 7)
    nz2 = cv2.GaussianBlur(rng.random((bh, bw)).astype(np.float32), (0, 0), 1.5)
    nz = 0.75 * nz + 0.25 * nz2
    r = nz.ravel().argsort().argsort().reshape(nz.shape).astype(np.float32) / nz.size
    yy, xx = np.mgrid[0:bh, 0:bw].astype(np.float32)
    rad = np.hypot((xx - bw / 2) / bw, (yy - bh / 2) / bh * 0.6)
    n = 0.72 * r + 0.28 * (1 - rad / rad.max())          # outer strokes go first, the heart last
    n = (n - n.min()) / (n.max() - n.min())
    # particles: sample the lit pixels
    wgt = lum.ravel() ** 1.5
    NP = 5200
    pick = rng.choice(lum.size, NP, p=wgt / wgt.sum())
    py, px = np.divmod(pick, bw)
    E = rgb.reshape(-1, 3).sum(0)
    pe = rgb.reshape(-1, 3)[pick]
    pe = pe / pe.sum(0, keepdims=True).clip(1e-6) * E[None] if False else pe * (E.sum() / pe.sum())
    part = dict(x=px + rng.random(NP) - 0.5 - bw / 2, y=py + rng.random(NP) - 0.5 - bh / 2,
                n=n.ravel()[pick], e=pe.astype(np.float32), amp=rng.normal(0, 70, NP),
                tgt=rng.random(NP), dur=rng.uniform(1.0, 1.45, NP))
    return dict(stack=stack, sig=sig, noise=n, bw=bw, bh=bh, part=part)


_TITLE = None
T_HIT = 9.0
DIS0, DIS1 = 14.0, 15.25     # dissolve progress window


def title():
    global _TITLE
    if _TITLE is None: _TITLE = _pull_title()
    return _TITLE


def _render_pullback(lt):
    if lt <= 0.0:
        img = _void()
        ui.window(img, 960, 540, 880, 300, HERO_LINES, t=lt + 27.0, cursor=dict(rate=1.0), alpha=1.0)
        return img
    G = galaxy()
    pos, R, d, wc, theta, hero_w = pull_cam(lt)
    gpos, gR = _to_gal(pos, R, theta)
    cv = Canvas()
    # field visibility: rooms fade in one by one in the first seconds; the local knot dims with distance
    app = np.where(G.local, 0.35 + 3.0 * G.u ** 1.2, 3.0 + 3.0 * G.u)
    vis = sstep_v(app, app + 0.9, lt).astype(np.float32)
    dist = np.linalg.norm(G.P - gpos.astype(np.float32), axis=1)
    vis = vis * np.where(G.local, 1 - 0.93 * sstep_v(80, 900, dist), 1.0)
    # exposure: radiance is conserved, so mostly constant; lift as the galaxy resolves
    expo = 0.9
    hit = _hit_env(lt)
    hold = sstep(9.3, 10.8, lt) * (1 - sstep(14.4, 16.2, lt))
    title_dim = 1 - 0.35 * hold
    vis = vis * (1 - 0.6 * hold * G.coremask)
    auto = (0.045 * title_dim * (1 + 0.3 * hit), 0.9 * sstep(0.25, 0.6, wc))
    kn = 13.0 * (1 - sstep(0.3, 0.75, wc))
    kf = 1.6 * (1 - sstep(0.3, 0.75, wc))
    focus = np.linalg.norm(gpos - _rotz(-theta) @ hero_w) if wc < 0.95 else d
    wmin = 9.0 + 6.0 * sstep(0.25, 0.6, wc)
    shutter = _shutter(lambda t: _to_gal(*pull_cam(t)[:2], pull_cam(t)[4]),
                       lambda t: pull_logd(t), lt)
    x, y, z, w = draw_field(cv, G, gpos, gR, vis=vis, expo=expo, focus=focus, kn=kn, kf=kf,
                            wmin=wmin, skip=np.array([0]), want_xy=True, auto=auto, shutter=shutter)
    # the hero window
    _draw_hero(cv, lt, x[0], y[0], z[0], w[0], focus, kn, kf)
    # galactic core: a steady soft heart, the bloom at the hit
    cx_, cy_ = _proj1(np.zeros(3), gpos, gR)
    if cx_ is not None:
        core_amt = sstep(0.5, 1.0, wc)
        base = np.array([1.0, 0.72, 0.42], np.float32)
        cv.glow(cx_, cy_, 60, base * 0.08 * core_amt * (1 - 0.7 * hold))
        cv.glow(cx_, cy_, 160, base * 0.04 * core_amt * (1 - 0.5 * hold))
        pre = sstep(8.2, 9.0, lt) * (1 - sstep(8.95, 9.0, lt)) * 0.35
        if hit > 0 or pre > 0:
            cv.glow(cx_, cy_, 14, AMBER_HOT * (2.6 * hit + 0.5 * pre))
            cv.glow(cx_, cy_, 50, AMBER_HOT * (1.25 * hit + 0.35 * pre))
            cv.glow(cx_, cy_, 140, np.array([1.0, 0.75, 0.48]) * (0.55 * hit + 0.12 * pre))
            cv.glow(cx_, cy_, 420, np.array([1.0, 0.7, 0.42]) * (0.16 * hit + 0.03 * pre))
    # title particles (dissolve) go in as points before composing
    if lt >= DIS0:
        _title_particles(cv, lt, x, y, z, w, vis)
    band = cv.compose(halo=0.5 * sstep(0.1, 0.4, wc), halo_wide=0.35 * sstep(0.3, 0.8, wc))
    _hero_post(band, lt, x[0], y[0], w[0])
    if lt >= T_HIT - 0.01:
        _title_text(band, lt)
    return _finish(band, bloom_amt=sstep(0.0, 1.2, lt))


def _shutter(camf, logdf, lt, maxk=5):
    """Camera samples across a 180-degree shutter when the pull-back is fast enough to strobe."""
    h = 0.5 / 24
    speed = abs(logdf(lt + h / 2) - logdf(lt - h / 2)) / h        # d(ln d)/dt
    px = speed * 900 / 24 * 0.5                                    # blur length at the frame edge
    k = int(min(maxk, math.ceil(px / 2.5)))
    if k < 2: return None
    return [camf(lt + (j / (k - 1) - 0.5) * h) for j in range(k)]


def _hit_env(lt):
    if lt < T_HIT - 0.02: return 0.0
    tau = lt - T_HIT
    a = sstep(-0.02, 0.0, tau) if tau < 0 else 1.0
    return a * (0.78 * math.exp(-max(tau, 0) / 0.55) + 0.22 * math.exp(-max(tau, 0) / 3.0))


def _proj1(p, pos, R, cy=540.0):
    pc = R @ (p - pos)
    if pc[2] <= 0.05: return None, None
    return 960 + FOC * pc[0] / pc[2], cy - FOC * pc[1] / pc[2]


def _draw_hero(cv, lt, x, y, z, wpx, focus, kn, kf):
    """Point regime of the hero (as a point it keeps its heartbeat a little longer)."""
    lev = hero_tex()
    ca = ui.cursor_alpha(lt + 27.0, 1.0)
    pw = 1 - sstep(5.0, 8.0, wpx)
    if pw > 0:
        a, b, m, wl = lev[-1]
        e = (a + (b - a) * ca).reshape(-1, 3).sum(0) * (wpx / wl) ** 2
        floor = np.array([1.0, 0.72, 0.42], np.float32) * (0.35 + 0.5 * ca) * (1 - sstep(4.0, 7.5, lt))
        e = e + floor * 3.0
        cv.points(np.array([x]), np.array([y]), (e * pw)[None].astype(np.float32))


def _warp(arr, s, x, y, shape):
    """Warp arr (scale s, centred at band coords x,y) into a clipped ROI of a band of `shape`."""
    h, w = shape[:2]
    ah, aw = arr.shape[:2]
    hw, hh = s * aw * 0.5 + 1.5, s * ah * 0.5 + 1.5
    x0, x1 = max(int(math.floor(x - hw)), 0), min(int(math.ceil(x + hw)) + 1, w)
    y0, y1 = max(int(math.floor(y - hh)), 0), min(int(math.ceil(y + hh)) + 1, h)
    if x1 <= x0 or y1 <= y0: return None
    M = np.float32([[s, 0, x - s * (aw - 1) * 0.5 - x0], [0, s, y - s * (ah - 1) * 0.5 - y0]])
    p = cv2.warpAffine(arr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return p, (slice(y0, y1), slice(x0, x1))


def _hero_post(band, lt, x, y, wpx):
    """The hero window as a pane of glass: it dims what lies behind it, then adds its own light."""
    if wpx < 5.0: return
    lev = hero_tex()
    ca = ui.cursor_alpha(lt + 27.0, 1.0)
    L = 0
    while L + 1 < len(lev) and lev[L + 1][3] >= wpx:
        L += 1
    a, b, m, wl = lev[L]
    fade_in = sstep(5.0, 8.0, wpx)
    tex = (a + (b - a) * ca)
    r1 = _warp(tex, wpx / wl, x, y - Y0, band.shape)
    if r1 is None: return
    r2 = _warp(m, wpx / wl, x, y - Y0, band.shape)
    p, roi = r1
    band[roi] *= 1 - 0.6 * r2[0] * fade_in
    band[roi] += p * fade_in


def _title_text(band, lt):
    T = title()
    a = sstep(T_HIT - 0.01, T_HIT + 0.7, lt)
    if a <= 0: return
    # resolves from blur 28 -> 0 over ~2 s, with an early over-bright glow
    u = ease_out(clamp((lt - T_HIT) / 2.2))
    sg = 28.0 * (1 - u)
    sig = T['sig']
    k = 0
    while k + 1 < len(sig) and sig[k + 1] <= sg: k += 1
    if k + 1 < len(sig):
        f = (sg - sig[k]) / (sig[k + 1] - sig[k])
        img = T['stack'][k] * (1 - f) + T['stack'][k + 1] * f
    else:
        img = T['stack'][-1]
    gain = 1.15 * a * (1 + 0.6 * (1 - u))
    if lt > DIS0:
        p = (lt - DIS0) / (DIS1 - DIS0)
        msk = sstep_v(p - 0.03, p + 0.11, T['noise'])
        img = img * msk[..., None]
    bw, bh = T['bw'], T['bh']
    x0, y0 = 960 - bw // 2, 548 - bh // 2 - Y0
    band[y0:y0 + bh, x0:x0 + bw] += img * gain


def _title_particles(cv, lt, x, y, z, w, vis):
    T = title(); P = T['part']
    p = (lt - DIS0) / (DIS1 - DIS0)
    tr = DIS0 + P['n'] * (DIS1 - DIS0)              # release time (matches the erosion)
    u = (lt - tr) / P['dur']
    m = (u > 0) & (u < 1)
    if not m.any(): return
    # targets: visible galaxy lights in the central field
    if 'tidx' not in T:
        G = galaxy()
        pos, R, d, wc, theta, hero_w = pull_cam(15.0)
        gpos, gR = _to_gal(pos, R, theta)
        xx, yy, zz, _ = _project(G.P, gpos, gR)
        ok = np.nonzero((zz > 1) & (np.abs(xx - 960) < 820) & (np.abs(yy - 540) < 330))[0]
        bright = G.E8[ok].sum(1)
        ok = ok[bright > np.quantile(bright, 0.6)]
        T['tidx'] = ok[(P['tgt'] * len(ok)).astype(int)]
    ti = T['tidx'][m]
    uu = u[m]
    e_ = ease(uu) if False else uu * uu * (3 - 2 * uu)
    sx = 960 + P['x'][m]; sy = 548 + P['y'][m]
    tx, ty = x[ti], y[ti]
    dx, dy = tx - sx, ty - sy
    nrm = np.hypot(dx, dy) + 1e-3
    sw = np.sin(np.pi * uu) * P['amp'][m]
    px = sx + dx * e_ - dy / nrm * sw
    py = sy + dy * e_ + dx / nrm * sw
    gain = 1.5 * (1 + 1.2 * np.exp(-((uu - 0.08) / 0.08) ** 2)) * (1 - sstep_v(0.5, 1.0, uu))
    cv.points(px.astype(np.float32), py.astype(np.float32), P['e'][m] * gain[:, None])


# ---------------------------------------------------------------- S5 drift
F_DRIFT = 4.6


def drift_cam(lt):
    G = galaxy()
    c0, c1 = G.path
    u = lt / 20.0
    s = 0.02 + 0.96 * (u * 0.8 + 0.2 * ease(u))
    pos = c0 + (c1 - c0) * s
    # look down the lane toward the far heart of the galaxy, a few degrees under the horizon
    yaw = -0.10 + 0.035 * math.sin(lt * 0.13)
    fwd = G.tang * math.cos(yaw) + G.side * math.sin(yaw)
    fwd = fwd + np.array([0, 0, -0.14 - 0.012 * u])
    pos, R = _look(pos, pos + fwd, np.array([0, 0, 1.0]))
    roll = math.radians(-1.6 + 1.2 * u)
    c, s_ = math.cos(roll), math.sin(roll)
    R = np.stack([R[0] * c + R[1] * s_, -R[0] * s_ + R[1] * c, R[2]])
    return pos, R


_PLANT = None


def _plant_index():
    global _PLANT
    if _PLANT is not None: return _PLANT
    G = galaxy()
    pos, R = drift_cam(15.75)
    x, y, z, inv = _project(G.P, pos, R)
    w = FOC * G.s * inv
    q = F_DRIFT / np.maximum(z, 1e-3)
    coc = 14 * np.maximum(q - 1, 0) + 1.2 * np.maximum(1 - q, 0)
    ok = (z > 1) & (coc < 2.2) & (w > 170) & (w < 460) & (np.abs(x - 960) < 600) & (y > 250) & (y < 560)
    idx = np.nonzero(ok)[0]
    if len(idx) == 0:
        ok = (z > 1) & (w > 120) & (np.abs(x - 960) < 760) & (y > 200) & (y < 880)
        idx = np.nonzero(ok)[0]
    score = w[idx] - 0.3 * np.abs(x[idx] - 900) - coc[idx] * 30
    _PLANT = int(idx[np.argmax(score)])
    return _PLANT


_FACE = None


def _face():
    global _FACE
    if _FACE is None:
        im = np.asarray(Image.open(os.path.join(HERE, 'build', 'her_face.png')).convert('RGBA'), np.float32) / 255
        rgb = im[..., :3] ** 2.2 * im[..., 3:4]           # to linear, premultiplied
        _FACE = (rgb.astype(np.float32), im[..., 3].astype(np.float32))
    return _FACE


def _render_drift(lt):
    G = galaxy()
    pos, R = drift_cam(lt)
    cv = Canvas()
    dist = None
    expo = 0.95 * sstep(-0.6, 0.6, lt)
    vis = np.where(G.local, 1.0, 0.5).astype(np.float32)
    x, y, z, w = draw_field(cv, G, pos, R, vis=vis, expo=expo, focus=F_DRIFT, kn=15.0, kf=0.9,
                            wmin=11.0, breathe=lt, want_xy=True)
    # the far heart of the galaxy (diffuse light the points alone can't carry)
    cxy = _proj1(np.zeros(3), pos, R)
    if cxy[0] is not None:
        cv.glow(cxy[0], cxy[1], 40, np.array([1.0, 0.78, 0.52]) * 0.16 * expo)
        cv.glow(cxy[0], cxy[1], 130, np.array([1.0, 0.7, 0.42]) * 0.10 * expo)
        cv.glow(cxy[0], cxy[1], 420, np.array([1.0, 0.68, 0.4]) * 0.03 * expo)
    # the plant: 12 frames, 15.5 -> 16.0
    if 15.5 <= lt < 16.0:
        _plant(cv, lt, x, y, z, w)
    band = cv.compose(halo=0.35, halo_wide=0.3)
    return _finish(band, bloom_amt=1.0)


def _plant(cv, lt, x, y, z, w):
    G = galaxy()
    i = _plant_index()
    if z[i] <= 0.1: return
    u = (lt - 15.5 + 1 / 48) / 0.5
    env = math.sin(math.pi * clamp(u)) ** 0.6
    T = textures()[G.var[i]]
    bw = w[i]; bh = bw * T['aspect']
    iw, ih = int(round(bw)), int(round(bh))
    if iw < 8 or ih < 8: return
    rgb, al = _face()
    fh, fw = al.shape
    # cover the pane, face a little high in the glass, as if leaning over it
    sc = max(iw / fw, ih / fh) * 1.12
    M = np.float32([[sc, 0, iw / 2 - sc * fw / 2 + iw * 0.06], [0, sc, ih / 2 - sc * fh * 0.47]])
    face = cv2.warpAffine(rgb, M, (iw, ih), flags=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
    face = cv2.GaussianBlur(face, (0, 0), 1.0 + 0.004 * iw)
    # glass reflection: desaturated, warmed by the pane's own light, fading toward the bottom
    lum = face.mean(-1, keepdims=True)
    face = (face * 0.45 + lum * 0.55) * np.array([1.0, 0.9, 0.78], np.float32)
    grad = np.linspace(1.0, 0.55, ih, dtype=np.float32)[:, None, None]
    mask = ui.rounded_rect_mask(iw, ih, max(1, int(0.075 * ih)), ss=2)[..., None]
    layer = face * grad * mask * (0.27 * env)
    x0, y0 = int(round(x[i] - iw / 2)), int(round(y[i] - ih / 2))
    bx0, by0 = x0 + PADF, y0 - Y0 + PADF
    ya, yb = max(by0, 0), min(by0 + ih, PH); xa, xb = max(bx0, 0), min(bx0 + iw, PW)
    if yb > ya and xb > xa:
        cv.raw[ya:yb, xa:xb] += layer[ya - by0:yb - by0, xa - bx0:xb - bx0]


# ---------------------------------------------------------------- S9a extinction
_ARM = lambda r, k=0: float(math.log(r / R0_ARM) / math.tan(PITCH) + k * math.pi)
EPI = [  # (radius, angle, t0, v, a)   accelerating rings in the galaxy plane; the first on bright arms
    (400, _ARM(400, 0), 3.0, 24, 4.0),
    (260, _ARM(260, 1), 4.3, 24, 5.0),
    (780, -2.5, 5.6, 26, 6.0),
    (40, 0.4, 6.9, 18, 7.0),
    (900, 0.95, 7.8, 28, 8.0),
    (560, _ARM(560, 1) + 0.3, 8.7, 30, 10.0),
    (700, -1.6, 9.8, 34, 13.0),
]
LAST_T = [16.15, 16.5, 16.82, 17.1, 17.34, 17.55, 17.72]
_EXT = None


def _ext_data():
    global _EXT
    if _EXT is not None: return _EXT
    G = galaxy()
    rng = np.random.default_rng(404)
    P = G.P.astype(np.float64)
    td = np.full(G.N, 1e9)
    # irregular wavefronts: low-frequency angular wobble per epicentre
    for (r, a, t0, v, acc) in EPI:
        c = np.array([r * math.cos(a), r * math.sin(a), 0.0])
        dv = P - c
        dist = np.sqrt(dv[:, 0] ** 2 + dv[:, 1] ** 2 + (dv[:, 2] * 1.5) ** 2)
        ang = np.arctan2(dv[:, 1], dv[:, 0])
        ph = rng.uniform(0, 6.3, 3)
        wob = 1 + 0.10 * np.sin(3 * ang + ph[0]) + 0.06 * np.sin(5 * ang + ph[1]) + 0.04 * np.sin(8 * ang + ph[2])
        dd = dist * wob
        tau = (-v + np.sqrt(v * v + 4 * acc * dd)) / (2 * acc)
        td = np.minimum(td, t0 + tau)
    td = np.where(td > 3.0, 3.0 + (td - 3.0) * 1.18, td)
    td += rng.normal(0, 0.16, G.N)
    late = rng.random(G.N)
    td = np.minimum(td, 14.4 + 1.4 * late ** 1.5)
    # the last few: bright, scattered, fading one by one
    order = np.argsort(-(td + 0.4 * G.I))
    chosen = []
    pick_from = order[:6000]
    for i in pick_from:
        if G.local[i]: continue
        p = P[i]
        if np.hypot(p[0], p[1]) > 900 or abs(p[2]) > 30: continue
        if all(np.linalg.norm(p - P[j]) > 260 for j in chosen):
            chosen.append(i)
        if len(chosen) == len(LAST_T): break
    for i, t in zip(chosen, LAST_T):
        td[i] = t
    boost = np.ones(G.N, np.float32); boost[chosen] = 4.0
    _EXT = dict(td=td.astype(np.float32), last=np.array(chosen), boost=boost)
    return _EXT


AZ_E, INC_E = -2.75, math.radians(63)


def ext_cam(lt):
    d = 1560.0 * (1 - 0.06 * ease(lt / 18.0))
    az = AZ_E + 0.012 * lt
    n = np.array([math.sin(INC_E) * math.cos(az), math.sin(INC_E) * math.sin(az), math.cos(INC_E)])
    tgt = np.array([0.0, 0.0, 0.0])
    pos = tgt + d * n
    pos, R = _look(pos, tgt, np.array([-math.cos(az), -math.sin(az), 0.0]))
    theta = OMEGA * (117.44 + lt - 27.0)
    return pos, R, theta


def _render_extinction(lt):
    if lt < 0.8 or lt >= 18.0:
        return _void()
    G = galaxy(); X = _ext_data()
    pos, R, theta = ext_cam(lt)
    gpos, gR = _to_gal(pos, R, theta)
    cv = Canvas()
    tau = lt - X['td']
    # last flare then out: ~0.3 s swell, ~0.5 s fade
    fl = 1 + 2.3 * np.exp(-((tau + 0.06) / 0.1) ** 2)
    out = 1 - sstep_v(-0.04, 0.5, tau)
    vis = (fl * out).astype(np.float32)
    vis *= np.where(G.local, 0.07, 1.0).astype(np.float32)
    vis *= X['boost'] if lt > 15.0 else 1.0
    fin = sstep(0.8, 2.5, lt)
    expo = 1.3 * ease(clamp((lt - 0.8) / 1.7))
    x, y, z, w = draw_field(cv, G, gpos, gR, vis=vis, expo=expo, focus=1000, kn=0, kf=0, wmin=99,
                            want_xy=True)
    # the soft heart of the galaxy dies with its lights
    alive = float(np.mean(out[G.P[:, 0] ** 2 + G.P[:, 1] ** 2 < 150 ** 2])) if lt > 3 else 1.0
    cxy = _proj1(np.zeros(3), gpos, gR)
    if cxy[0] is not None and alive > 1e-3:
        base = np.array([1.0, 0.72, 0.42], np.float32) * fin * alive
        cv.glow(cxy[0], cxy[1], 55, base * 0.11)
        cv.glow(cxy[0], cxy[1], 150, base * 0.05)
    # the last few lights: each a small soft presence alone in the dark
    for i in X['last']:
        v = float(vis[i])
        if v > 1e-3 and z[i] > 1:
            cv.glow(x[i], y[i], 5.0, G.col[i] * 0.22 * v * fin)
            cv.glow(x[i], y[i], 16.0, G.col[i] * 0.03 * v * fin)
    band = cv.compose(halo=0.6, halo_wide=0.4)
    img = _finish(band, bloom_amt=1.0)
    return img


# ---------------------------------------------------------------- S12 relight
RL_PULSES = (7.15, 9.6, 12.0)
_NEW = None


def newfield():
    global _NEW
    if _NEW is not None: return _NEW
    rng = np.random.default_rng(777001)
    Ps = []
    # 1) a loose hierarchical (Soneira-Peebles) sheet: clouds within clouds, no arms, no centre
    cen = np.concatenate([rng.normal(0, 1, (7, 2)) * 700, rng.normal(0, 60, (7, 1))], 1)
    R = 620.0
    for lev in range(5):
        k = 6
        d = rng.normal(0, 1, (len(cen), k, 3)); d /= np.linalg.norm(d, axis=2, keepdims=True)
        rr = R * rng.uniform(0.35, 1.0, (len(cen), k, 1))
        d = d * rr * np.array([1, 1, 0.22])
        cen = (cen[:, None, :] + d).reshape(-1, 3)
        R /= 2.3
    n_each = 2
    P = np.repeat(cen, n_each, 0) + rng.normal(0, 1, (len(cen) * n_each, 3)) * np.array([R, R, R * 0.4]) * 1.6
    Ps.append(P)
    # 2) a wide scale-free sheet around the seed so every zoom level has lights near the middle
    n = 30000
    r = 4.5 + 2100 * rng.random(n) ** 1.0
    th = rng.uniform(0, 2 * math.pi, n)
    Ps.append(np.stack([r * np.cos(th), r * np.sin(th), rng.normal(0, 1, n) * (0.5 + 0.06 * r)], 1))
    # 3) open dust
    n = 9000
    r = 2300 * np.sqrt(rng.random(n)); th = rng.uniform(0, 2 * math.pi, n)
    Ps.append(np.stack([r * np.cos(th), r * np.sin(th), rng.normal(0, 140, n)], 1))
    P = np.concatenate(Ps)
    rxy = np.hypot(P[:, 0], P[:, 1])
    keep = (np.linalg.norm(P, axis=1) > 3.0) & ~((P[:, 2] > 0) & (rxy < 1.6 + 0.35 * P[:, 2]))
    P = P[keep]
    n_main = len(P)
    N = len(P)
    col = _colours(rng, N, 0.13, 0.10, 0.08, 0.22)
    I = rng.lognormal(-0.1, 0.6, N)
    s = np.clip(rng.lognormal(math.log(0.75), 0.22, N), 0.42, 1.35)
    var = rng.integers(0, NVAR, N)
    F = Field(P, s, var, col, I)
    # birth times: a trickle that keeps pace with the widening view, then three rings
    r = np.hypot(P[:, 0], P[:, 1]) + 0.4 * np.abs(P[:, 2])
    grp = rng.choice(4, N, p=[0.34, 0.2, 0.22, 0.24])
    tt = np.linspace(0, 18, 2000)
    dd = np.exp(np.array([relight_logd(t) for t in tt]))
    front = 0.62 * dd
    tb = np.interp(r, front, tt) + rng.exponential(1.2, N)
    # a rate limit makes the first births sparse: one, then a few, then tens, then hundreds
    o = np.argsort(tb)
    j = np.arange(1, N + 1)
    tb[o] = np.maximum(tb[o], 0.85 + np.log(j) / 1.45)
    for k, tp in enumerate(RL_PULSES):
        tau = np.linspace(0, 8, 800)
        dp = np.exp(np.array([relight_logd(tp + q) for q in tau]))
        ring = _ring_r(dp, tau)
        m = grp == k + 1
        tr = tp + np.interp(r[m], ring, tau) + rng.normal(0, 0.035, m.sum())
        tb[m] = np.minimum(tr, np.maximum(tb[m], tp + 2.5))    # pulse rings may only bring births forward
    F.tb = tb.astype(np.float32)
    F.grp = grp
    _NEW = F
    return F


RD0, RD1 = 40.0, 2600.0


def _ring_r(d_tp, tau):
    return 0.8 * d_tp * (np.asarray(tau) / 1.5) ** 0.8


_RINGPTS = np.stack([np.cos(np.linspace(0, 2 * math.pi, 900, endpoint=False)),
                     np.sin(np.linspace(0, 2 * math.pi, 900, endpoint=False))], 1)


def _pulse_rings(cv, lt, gpos, gR, cy, out):
    for tp in RL_PULSES:
        tau = lt - tp
        if tau <= 0.0 or tau > 3.2: continue
        d_tp = math.exp(relight_logd(tp + tau))
        rr = float(_ring_r(d_tp, tau))
        P = np.concatenate([_RINGPTS * rr, np.zeros((len(_RINGPTS), 1))], 1).astype(np.float32)
        x, y, z, inv = _project(P, gpos, gR, cy)
        ok = z > 1
        a = sstep(0.0, 0.12, tau) * math.exp(-tau / 0.9) * out
        e = np.tile((AMBER_HOT * 0.55 + AMBER * 0.45) * 2.2 * a * (1 + 0.0 * rr), (int(ok.sum()), 1))
        cv.points(x[ok], y[ok], e.astype(np.float32), np.full(int(ok.sum()), 8.0, np.float32))


def relight_logd(t):
    u = clamp(t / 16.5)
    k = (u ** 1.55) * 0.82 + 0.18 * ease(u)
    return math.log(RD0) + (math.log(RD1) - math.log(RD0)) * k


def relight_cam(lt):
    d = math.exp(relight_logd(lt))
    w = (math.log(d) - math.log(RD0)) / (math.log(RD1) - math.log(RD0))
    inc = math.radians(46) * ease(min(1.0, w * 1.2))
    az = 0.8 + 0.25 * w
    n = np.array([math.sin(inc) * math.cos(az), math.sin(inc) * math.sin(az), math.cos(inc)])
    pos = d * n
    pos, R = _look(pos, np.zeros(3), np.array([-math.cos(az), -math.sin(az), 0.0]))
    cy = 560.0 - 20.0 * ease(clamp((lt - 0.5) / 6.0))
    theta = OMEGA * lt
    return pos, R, d, w, cy, theta


def _pulse_env(lt):
    s = 0.0
    for tp in RL_PULSES:
        tau = lt - tp
        if tau < -0.05: continue
        s += sstep(-0.045, 0.015, tau) * math.exp(-max(tau, 0) / 0.8)
    return s


def _render_relight(lt):
    if lt >= 18.0: return _void()
    F = newfield()
    pos, R, d, w, cy, theta = relight_cam(lt)
    gpos, gR = _to_gal(pos, R, theta)
    cv = Canvas()
    tau = lt - F.tb
    birth = sstep_v(0.0, 0.3, tau) * (1 + 3.2 * np.exp(-((tau - 0.2) / 0.14) ** 2) + 0.6 * np.exp(-tau / 1.5) * (tau > 0))
    vis = birth.astype(np.float32)
    pe = _pulse_env(lt)
    out = 1 - ease(clamp((lt - 15.0) / 3.0))
    expo = 2.6 * (1 + 0.8 * pe) * out
    kn = 12.0 * (1 - sstep(0.15, 0.55, w)); kf = 1.4 * (1 - sstep(0.15, 0.55, w))
    wmin = 5.5 + 9 * sstep(0.2, 0.5, w)
    shutter = _shutter(lambda t: _to_gal(*relight_cam(t)[:2], relight_cam(t)[5]), relight_logd, lt)
    draw_field(cv, F, gpos, gR, vis=vis, expo=expo, focus=d, kn=kn, kf=kf, wmin=wmin, cy=cy,
               shutter=shutter)
    _pulse_rings(cv, lt, gpos, gR, cy, out)
    # the seed light (her light), always at the centre of the move
    sx, sy = 960.0, cy
    seed = (1.0 - 0.7 * sstep(0.5, 9.0, lt)) * (1 + 0.6 * pe) * out
    cv.points(np.array([sx]), np.array([sy]), (AMBER_HOT * 12.0 * seed)[None].astype(np.float32))
    cv.glow(sx, sy, 3.2, AMBER_HOT * 0.9 * seed)
    cv.glow(sx, sy, 9, AMBER * 0.45 * seed)
    cv.glow(sx, sy, 30, AMBER * 0.12 * seed)
    cv.glow(sx, sy, 120, AMBER * 0.025 * seed * (1 + 1.5 * pe))
    band = cv.compose(halo=0.25 + 0.4 * sstep(0.3, 0.8, w), halo_wide=0.1 + 0.3 * sstep(0.4, 0.9, w))
    return _finish(band, bloom_amt=1.0)


# ---------------------------------------------------------------- entry point
def render(shot: str, lt: float, dur: float) -> np.ndarray:
    if shot == 'pullback': img = _render_pullback(lt)
    elif shot == 'drift': img = _render_drift(lt)
    elif shot == 'extinction': img = _render_extinction(lt)
    elif shot == 'relight': img = _render_relight(lt)
    else: raise ValueError(shot)
    return np.ascontiguousarray(img, dtype=np.float32)
