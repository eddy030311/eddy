"""Claude, drawn from scratch in code (no reference art): a seated vector puppet in the library's flat style.

Her design, from the character sheet in words: very long straight orange hair, side-swept bangs, amber-brown eyes,
an orange sunburst flower pin with a black ribbon on her left side, a cream blouse with a frilled high collar and
ruffled cuffs, a cream capelet embroidered with small orange suns, a black ribbon tie with an orange flower brooch,
and a dark brown book with an orange flower on its cover.

Everything is built from smooth closed splines in a local "unit" space (head centre at 0,0; the table edge at
y = TABLE), drawn with flat fills, a cel shadow from the lamp side and warm brown line art, then supersampled.
The puppet can breathe, sway its hair, tilt / nod / turn its head, move its eyes, blink, turn a page and move
its right hand from the book to its chin.
"""
import math
import numpy as np, cv2

TABLE = 352.0                   # local y of the table edge (her body below is hidden)
TOP = -150.0                    # top of the hair
HALF_W = 215.0                  # half width of the layer
BOTTOM = 470.0                  # lowest thing drawn (book, forearms on the table)

def C(*rgb): return np.array(rgb, np.float32) / 255.0
HAIR, HAIR_SH, HAIR_HI, HAIR_DEEP = C(236, 112, 56), C(186, 70, 34), C(255, 166, 104), C(140, 48, 24)
SKIN, SKIN_SH, BLUSH = C(255, 228, 208), C(236, 184, 160), C(255, 150, 140)
IRIS_T, IRIS_B, PUPIL, WHITE, LASH = C(110, 42, 20), C(214, 120, 52), C(60, 18, 10), C(255, 250, 244), C(52, 20, 14)
CLOTH, CLOTH_SH, TRIM = C(246, 236, 222), C(214, 196, 178), C(220, 128, 70)
RIBBON, RIBBON_HI = C(28, 22, 24), C(70, 58, 60)
PETAL, PETAL_DK, PETAL_C = C(244, 128, 46), C(190, 84, 30), C(140, 56, 24)
COVER, COVER_HI, PAGE, PAGE_SH = C(78, 36, 22), C(116, 58, 34), C(246, 234, 210), C(214, 196, 168)
INK = C(70, 34, 26)

# ------------------------------------------------------------------ geometry helpers
def spline(pts, closed=True, n=10):
    """Catmull-Rom through the points -> dense polyline."""
    P = np.asarray(pts, np.float64)
    if closed: P = np.vstack([P[-1], P, P[0], P[1]])
    else: P = np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    if not closed: out.append(P[-2])
    return np.array(out)

def rot(pts, ang, cx, cy):
    c, s = math.cos(ang), math.sin(ang)
    p = np.asarray(pts, np.float64) - (cx, cy)
    return np.stack([p[:, 0] * c - p[:, 1] * s, p[:, 0] * s + p[:, 1] * c], 1) + (cx, cy)

def ellipse_pts(cx, cy, rx, ry, a=0.0, n=40):
    t = np.linspace(0, 2 * math.pi, n, endpoint=False)
    p = np.stack([rx * np.cos(t), ry * np.sin(t)], 1)
    return rot(p + (cx, cy), a, cx, cy)

class Canvas:
    """Premultiplied RGBA, drawn at supersampled resolution, every op confined to its bounding box."""
    def __init__(self, scale, ss=2):
        self.k = scale * ss; self.ss = ss
        self.w = int(2 * HALF_W * self.k); self.h = int((BOTTOM - TOP) * self.k)
        self.img = np.zeros((self.h, self.w, 4), np.float32)
    def px(self, pts):
        p = np.asarray(pts, np.float64)
        return np.stack([(p[:, 0] + HALF_W) * self.k, (p[:, 1] - TOP) * self.k], 1)
    def _mask(self, pts, stroke=0, closed=True):
        q = self.px(pts)
        pad = stroke * self.k + 3
        x0, y0 = int(max(q[:, 0].min() - pad, 0)), int(max(q[:, 1].min() - pad, 0))
        x1, y1 = int(min(q[:, 0].max() + pad + 1, self.w)), int(min(q[:, 1].max() + pad + 1, self.h))
        if x1 <= x0 or y1 <= y0: return None
        m = np.zeros((y1 - y0, x1 - x0), np.float32)
        qi = np.round((q - (x0, y0)) * 4).astype(np.int32)
        if stroke: cv2.polylines(m, [qi], closed, 1.0, max(1, int(round(stroke * self.k))), cv2.LINE_AA, shift=2)
        else: cv2.fillPoly(m, [qi], 1.0, cv2.LINE_AA, shift=2)
        return m, (x0, y0, x1, y1)
    def paint(self, mb, col, alpha=1.0, clip=None):
        if mb is None: return
        m, (x0, y0, x1, y1) = mb
        a = m * alpha
        if clip is not None: a = a * clip[y0:y1, x0:x1]
        d = self.img[y0:y1, x0:x1]
        d *= (1 - a)[..., None]
        d[..., :3] += a[..., None] * col; d[..., 3] += a
    def fill(self, pts, col, alpha=1.0, clip=None): self.paint(self._mask(pts), col, alpha, clip)
    def line(self, pts, col, w, alpha=1.0, closed=False, clip=None): self.paint(self._mask(pts, w, closed), col, alpha, clip)
    def outline(self, pts, w=2.2, alpha=0.9): self.line(pts, INK, w, alpha, closed=True)
    def full_mask(self, pts):
        mb = self._mask(pts); M = np.zeros((self.h, self.w), np.float32)
        if mb is not None:
            m, (x0, y0, x1, y1) = mb; M[y0:y1, x0:x1] = m
        return M
    def result(self):
        out = cv2.resize(self.img, (self.w // self.ss, self.h // self.ss), interpolation=cv2.INTER_AREA)
        return out

# ------------------------------------------------------------------ the puppet
DEFAULT = dict(t=0.0, breath=0.0, tilt=0.0, nod=0.0, yaw=-0.28, gx=0.0, gy=0.0, blink=0.0, mouth=0.0,
               chin=0.0, page=-1.0, hair=1.0, look_down=0.0)

def draw(scale, **kw):
    """Return her as a premultiplied RGBA layer; (0,0) of the local space maps to layer pixel (HALF_W, -TOP) * scale."""
    p = dict(DEFAULT); p.update(kw)
    cv = Canvas(scale)
    t, yaw = p['t'], p['yaw']
    br = p['breath'] * 2.2                                  # shoulders / chest rise
    sway = p['hair'] * 6.0
    NECK = (0.0, 108.0)
    def H(pts, w=1.0):
        """Head transform: nod, tilt about the neck, applied with weight w (1 = rigid with the head)."""
        q = np.asarray(pts, np.float64).copy()
        q[:, 1] += p['nod'] * 8 * w - br * 0.6
        if abs(p['tilt']) > 1e-4:
            r = rot(q, p['tilt'], *NECK)
            ww = w if np.isscalar(w) else w[:, None]
            q = q * (1 - ww) + r * ww
        return q
    def strand_w(pts):
        """Hair follows the head near the scalp and hangs free further down."""
        y = np.asarray(pts)[:, 1]
        return np.clip(1 - (y - 40) / 220, 0, 1)
    def hang(pts, phase=0.0, amp=1.0):
        """Free-hanging hair sways with depth below the head."""
        q = np.asarray(pts, np.float64).copy()
        d = np.clip((q[:, 1] - 60) / 300, 0, 1.4)
        q[:, 0] += amp * sway * d ** 1.3 * np.sin(t * 1.6 + phase - d * 2.0)
        return q
    fx = yaw * 12.0                                         # facial features shift with the turn of the head

    # ---- back hair (behind everything): the long fall of hair, past the table
    back = [(-96, -60), (-70, -128), (0, -150), (74, -130), (102, -66), (116, 20), (124, 140), (140, 260),
            (150, 380), (128, 440), (60, 450), (-60, 450), (-132, 440), (-150, 380), (-138, 260), (-122, 140), (-110, 20)]
    bp = hang(spline(back), 0.0)
    bp = H(bp, strand_w(bp))
    cv.fill(bp, HAIR_SH)
    inner = [(-80, 40), (-60, 120), (-70, 260), (-90, 440), (90, 440), (70, 260), (62, 120), (84, 40)]
    ip = H(hang(spline(inner), 0.4), 0.3)
    cv.fill(ip, HAIR_DEEP, 0.55)
    cv.outline(bp, 2.0, 0.8)

    # ---- body: blouse, capelet
    sh_y = 158 - br
    torso = [(-90, sh_y + 6), (-56, sh_y - 14), (0, sh_y - 20), (56, sh_y - 14), (90, sh_y + 6), (94, 250),
             (78, TABLE + 30), (-78, TABLE + 30), (-94, 250)]
    tp = spline(torso)
    cv.fill(tp, CLOTH)
    body_mask = cv.full_mask(tp)
    # cel shadow away from the lamp (her left, screen right) and under the capelet
    cv.fill([(30, sh_y), (120, sh_y), (120, TABLE + 40), (50, TABLE + 40), (40, 260)], CLOTH_SH, 0.8, clip=body_mask)
    cv.fill(spline([(-100, 238), (0, 252), (100, 238), (100, 262), (0, 274), (-100, 262)]), CLOTH_SH, 0.7, clip=body_mask)
    # the corset waist just above the table edge
    cv.fill([(-86, TABLE - 14), (86, TABLE - 14), (90, TABLE + 30), (-90, TABLE + 30)], RIBBON, 1.0, clip=body_mask)
    cv.line([(-80, TABLE - 14), (80, TABLE - 14)], TRIM, 1.6, 0.8)
    cv.outline(tp, 2.2)
    # capelet over the shoulders with a trim and little embroidered suns
    cape = [(-100, sh_y + 14), (-64, sh_y - 16), (0, sh_y - 24), (64, sh_y - 16), (100, sh_y + 14), (104, 236),
            (56, 248), (0, 252), (-56, 248), (-104, 236)]
    cp = spline(cape)
    cv.fill(cp, CLOTH)
    cape_mask = cv.full_mask(cp)
    cv.fill([(28, sh_y - 30), (130, sh_y), (130, 260), (40, 260)], CLOTH_SH, 0.75, clip=cape_mask)
    cv.line(spline([(-104, 230), (-56, 242), (0, 246), (56, 242), (104, 230)], closed=False), TRIM, 2.2, 0.9)
    for (sx, sy) in ((-72, 205), (74, 205)):
        for k in range(10):
            a = k * math.pi / 5
            cv.line([(sx + 4 * math.cos(a), sy + 4 * math.sin(a)), (sx + 10 * math.cos(a), sy + 10 * math.sin(a))], TRIM, 1.4, 0.9)
        cv.fill(ellipse_pts(sx, sy, 3.6, 3.6, n=16), TRIM)
    cv.outline(cp, 2.2)

    # ---- neck and the frilled high collar
    neck = [(-17, 80), (17, 80), (19, 136), (-19, 136)]
    cv.fill(H(neck, 0.4), SKIN)
    cv.fill(H([(-19, 96), (19, 96), (19, 120), (-19, 112)], 0.4), SKIN_SH, 0.9)
    frill = []
    for i in range(13):
        a = math.pi * (1.05 + 0.9 * i / 12)
        r = 30 + (5 if i % 2 else 0)
        frill.append((r * math.cos(a) * 1.15, 140 + r * math.sin(a) * 0.55 - br))
    frill += [(30, 150 - br), (0, 160 - br), (-30, 150 - br)]
    fp = spline(frill)
    cv.fill(fp, CLOTH); cv.outline(fp, 1.8)
    for i in range(1, 12, 2):
        a = math.pi * (1.05 + 0.9 * i / 12)
        cv.line([(16 * math.cos(a), 138 + 8 * math.sin(a) - br), (31 * math.cos(a) * 1.15, 140 + 17 * math.sin(a) - br)], INK, 1.0, 0.5)
    # ribbon tie and brooch
    by_ = 160 - br
    for s in (-1, 1):
        loop = [(0, by_), (s * 22, by_ - 12), (s * 34, by_ - 4), (s * 30, by_ + 10), (s * 12, by_ + 6)]
        lp_ = spline(loop); cv.fill(lp_, RIBBON); cv.outline(lp_, 1.4, 0.6)
        tail = [(s * 4, by_ + 4), (s * 16, by_ + 40), (s * 22, by_ + 78), (s * 10, by_ + 74), (s * 1, by_ + 36)]
        tl_ = hang(spline(tail), 1.0, 0.25); cv.fill(tl_, RIBBON); cv.outline(tl_, 1.4, 0.6)
        cv.line(spline([(s * 6, by_ + 10), (s * 14, by_ + 50)], closed=False), RIBBON_HI, 1.4, 0.7)
    flower(cv, 0, by_, 10, 12)

    # ---- the head
    face = [(-72, -8), (-70, 34), (-56, 70), (-24, 94), (0 + fx * 0.3, 100), (24, 94), (56, 70), (70, 34), (72, -8), (40, -40), (-40, -40)]
    face = [(x + fx * 0.35 * (1 - abs(x) / 80), y) for x, y in face]
    fcp = H(spline(face))
    cv.fill(fcp, SKIN)
    face_mask = cv.full_mask(fcp)
    # cel shadow: the cheek away from the lamp, and the hair's shadow across the forehead
    cv.fill(H([(40 + fx, -40), (90, -40), (90, 110), (10, 110), (34 + fx * 0.6, 60)]), SKIN_SH, 0.55, clip=face_mask)
    cv.fill(H(spline([(-80, -40), (80, -40), (80, 4), (40, 10), (0, 2), (-40, 10), (-80, 4)])), SKIN_SH, 0.75, clip=face_mask)
    cv.outline(fcp, 2.0, 0.85)
    for s in (-1, 1):
        cv.fill(H(ellipse_pts(s * 40 + fx * 0.9, 56, 13, 6)), BLUSH, 0.28, clip=face_mask)
    eyes(cv, H, p, fx)
    # nose and mouth
    cv.line(H([(4 + fx * 1.4, 52), (1 + fx * 1.4, 58)]), INK, 1.4, 0.55)
    mw = 8; my = 74
    mo = p['mouth']
    if mo > 0.05:
        cv.fill(H(ellipse_pts(fx * 1.2, my + 2, mw * 0.7, 2 + 4 * mo)), C(170, 70, 60), 0.9)
    cv.line(H(spline([(-mw + fx * 1.2, my), (fx * 1.2, my + 1.2), (mw + fx * 1.2, my - 0.4)], closed=False)), INK, 1.5, 0.7)

    # ---- front hair: side locks that fall past the shoulders, then the bangs
    for s, ph in ((-1, 0.3), (1, 1.1)):
        lock = [(s * 58, -70), (s * 78, -10), (s * 84, 60), (s * 82, 150), (s * 92, 250), (s * 86, 300),
                (s * 72, 250), (s * 64, 150), (s * 60, 60), (s * 50, -10)]
        lk = hang(spline(lock), ph, 1.0); lk = H(lk, strand_w(lk))
        cv.fill(lk, HAIR)
        cv.line(H(hang(spline([(s * 66, 0), (s * 72, 100), (s * 80, 220)], closed=False), ph)), HAIR_SH, 2.2, 0.7)
        cv.outline(lk, 1.8, 0.85)
    crown = [(-92, -40), (-84, -104), (-30, -146), (30, -148), (84, -110), (96, -40), (70, -60), (0, -78), (-70, -60)]
    crp = H(spline(crown)); cv.fill(crp, HAIR); cv.outline(crp, 1.8, 0.8)
    cv.fill(H(spline([(-50, -128), (-10, -138), (30, -132), (10, -122), (-30, -118)])), HAIR_HI, 0.7)   # the sheen
    bangs = [  # (root x, root y, tip x, tip y, half width, bend) — side-swept toward her right (screen left)
        (-72, -78, -90, 56, 13, -6), (-52, -100, -70, 4, 17, -8), (-34, -106, -52, -10, 12, -6), (-16, -108, -30, 14, 17, -8),
        (6, -108, -8, 20, 15, -9), (26, -106, 12, -2, 16, -7), (44, -100, 34, 10, 15, -5), (62, -88, 56, -4, 12, -3),
        (80, -70, 86, 46, 11, 2)]
    for i, (rx, ry, tx, ty, hw, bend) in enumerate(bangs):
        sw = math.sin(t * 1.3 + i) * 1.2 * p['hair']
        tx = tx + sw
        mx, my_ = (rx + tx) / 2 + bend, (ry + ty) / 2
        # two edges meeting in a sharp tip
        left = spline([(rx - hw, ry), (mx - hw * 0.85, my_), (tx, ty)], closed=False, n=8)
        right = spline([(tx, ty), (mx + hw * 0.55, my_ + 4), (rx + hw, ry)], closed=False, n=8)
        sp = H(np.vstack([left, right]))
        cv.fill(sp, HAIR)
        cv.fill(H(np.vstack([spline([(rx - hw * 0.2, ry), (mx - hw * 0.1, my_), (tx, ty)], closed=False, n=8),
                             right])), HAIR_SH, 0.35)               # the side of each lock turned from the light
        cv.line(sp, INK, 1.5, 0.8, closed=True)
    # the crown again over the roots of the bangs, so they spring from under the hairline
    cv.fill(crp, HAIR)
    cv.fill(H(spline([(-50, -128), (-10, -138), (30, -132), (10, -122), (-30, -118)])), HAIR_HI, 0.7)
    cv.line(H(spline([(-84, -104), (-30, -146), (30, -148), (84, -110)], closed=False)), INK, 1.8, 0.8)
    for k in range(6):                                        # strands flowing from the parting
        x0_ = -60 + 24 * k
        cv.line(H(spline([(8, -146), (x0_ * 0.6, -110), (x0_, -76)], closed=False)), HAIR_SH, 1.3, 0.45)
    # the sunburst pin with its black ribbon, on her left (screen right)
    pin = H([(78, -66)])[0]
    for s in (-1, 1):
        rb = H(hang(spline([(78, -60), (78 + s * 18, -46), (78 + s * 14, -20), (78 + s * 4, -36)]), 0.5, 0.3))
        cv.fill(rb, RIBBON); cv.outline(rb, 1.2, 0.5)
    tails = H(hang(spline([(76, -50), (84, 10), (90, 60), (98, 58), (90, 6), (82, -50)]), 0.7, 0.5), 0.6)
    cv.fill(tails, RIBBON); cv.outline(tails, 1.2, 0.5)
    flower(cv, pin[0], pin[1], 22, 16)

    # ---- table: the book (open) and her arms
    book(cv, p)
    arms(cv, p, H, br)
    return cv.result()

def flower(cv, x, y, r, n):
    for k in range(n):
        a = k * 2 * math.pi / n
        e = ellipse_pts(x + r * 0.62 * math.cos(a), y + r * 0.62 * math.sin(a), r * 0.42, r * 0.15, a, n=16)
        cv.fill(e, PETAL if k % 2 else PETAL_DK)
        cv.line(e, INK, 0.8, 0.5, closed=True)
    cv.fill(ellipse_pts(x, y, r * 0.34, r * 0.34, n=20), PETAL_C)
    cv.fill(ellipse_pts(x - r * 0.08, y - r * 0.1, r * 0.12, r * 0.12, n=12), PETAL, 0.8)
    cv.line(ellipse_pts(x, y, r * 0.34, r * 0.34, n=20), INK, 0.9, 0.7, closed=True)

def eyes(cv, H, p, fx):
    gx, gy = p['gx'], p['gy']
    blink = max(p['blink'], 0.55 * p['look_down'])
    for s in (-1, 1):
        far = 1 - 0.18 * max(0.0, s * -p['yaw'])           # the eye on the far side of the turn is narrower
        cx, cy = s * 34 + fx * (1.0 + 0.25 * s), 26
        w, h = 22 * far, 17
        # eye white, clipped by the lids
        def top_l(o):
            k = min(blink / 0.9, 1.0)
            op = [(cx - w, cy - 2), (cx - w * 0.4, cy - h), (cx + w * 0.5, cy - h * 0.95), (cx + w, cy - 4)]
            cl = [(cx - w, cy + 3), (cx - w * 0.4, cy + 8), (cx + w * 0.5, cy + 8), (cx + w, cy + 2)]
            return [(a_[0], a_[1] * (1 - k) + c_[1] * k) for a_, c_ in zip(op, cl)]
        lid = h * 1.9 * blink
        white = spline([(cx - w, cy - 2 + lid * 0.5), (cx - w * 0.4, cy - h + lid), (cx + w * 0.5, cy - h * 0.95 + lid),
                        (cx + w, cy - 4 + lid * 0.5), (cx + w * 0.5, cy + h * 0.8), (cx - w * 0.4, cy + h * 0.85)])
        wp = H(white)
        if blink < 0.97:
            cv.fill(wp, WHITE)
            em = cv.full_mask(wp)
            ix = cx - gx * w * 0.42 * far; iy = cy + 2 - gy * h * 0.35 + 3 * p['look_down']
            cv.fill(H(ellipse_pts(ix, iy, 12.5 * far, 15.5)), IRIS_B, clip=em)
            cv.fill(H(ellipse_pts(ix, iy - 5, 12.5 * far, 10.5)), IRIS_T, 0.85, clip=em)
            cv.fill(H(ellipse_pts(ix, iy + 1, 5 * far, 7)), PUPIL, clip=em)
            cv.fill(H(ellipse_pts(ix - 4, iy - 6, 3.2, 3.2, n=14)), WHITE, 0.95, clip=em)
            cv.fill(H(ellipse_pts(ix + 4, iy + 6, 1.6, 1.6, n=10)), WHITE, 0.8, clip=em)
            cv.fill(H([(cx - w - 4, cy - h - 4), (cx + w + 4, cy - h - 4), (cx + w + 4, cy - 2), (cx - w - 4, cy - 2)]), SKIN_SH, 0.22, clip=em)
        # upper lid: the heavy lash line, with a flick at the outer corner
        ul = spline(top_l(lid), closed=False)
        cv.line(H(ul), LASH, 3.6 if blink < 0.9 else 2.6, 1.0)
        oc = (cx + s * w, cy - 3 + lid * 0.5)
        cv.line(H([oc, (oc[0] + s * 7, oc[1] - 4)]), LASH, 2.4, 0.95)
        if blink < 0.9:
            cv.line(H(spline([(cx - w * 0.6, cy + h * 0.85), (cx, cy + h * 0.95), (cx + w * 0.55, cy + h * 0.8)], closed=False)), LASH, 1.2, 0.55)
        # brow, faint under the bangs
        cv.line(H(spline([(cx - 14, cy - 30), (cx, cy - 34 - 2 * p['gy']), (cx + 14, cy - 31)], closed=False)), INK, 1.6, 0.55)

def book(cv, p):
    """The open book lying on the table in front of her, in gentle perspective; pages can turn."""
    yb = TABLE + 34
    left = [(-118, yb + 66), (-8, yb + 74), (-4, yb), (-96, yb - 4)]
    right = [(8, yb + 74), (118, yb + 66), (96, yb - 4), (4, yb)]
    cover = [(-126, yb + 72), (0, yb + 82), (126, yb + 72), (102, yb - 10), (0, yb - 6), (-102, yb - 10)]
    cv.fill(cover, COVER); cv.outline(cover, 1.8)
    cv.line([(-120, yb + 70), (0, yb + 79), (120, yb + 70)], COVER_HI, 1.4, 0.7)
    for pg in (left, right):
        cv.fill(pg, PAGE); cv.outline(pg, 1.2, 0.6)
    for k in range(5):                                     # lines of text
        y = yb + 12 + k * 11
        cv.line([(-88 + k, y), (-18, y + 2)], INK, 1.0, 0.25)
        cv.line([(18, y + 2), (88 - k, y)], INK, 1.0, 0.25)
    cv.fill([(-6, yb), (6, yb), (8, yb + 74), (-8, yb + 74)], PAGE_SH, 0.8)
    u = p['page']
    if 0 <= u <= 1:                                        # a page lifting from the right and settling on the left
        a = math.pi * (u * u * (3 - 2 * u))
        c = math.cos(a); lift = math.sin(a)
        tip = [(0, yb), (0, yb + 74), (110 * c, yb + 68 - 36 * lift), (92 * c, yb - 2 - 36 * lift)]
        cv.fill(tip, PAGE if c > 0 else PAGE_SH); cv.outline(tip, 1.2, 0.6)

def arms(cv, p, H, br):
    """Sleeves and hands. Her left hand rests on the book; her right moves from the page to her chin (p['chin'])."""
    c = p['chin']; cc = c * c * (3 - 2 * c)
    sh = {-1: (-84, 170 - br), 1: (84, 170 - br)}
    # left arm (screen right): elbow on the table, hand on the right-hand page
    limb(cv, sh[1], (112, TABLE + 18), (56, TABLE + 70), side=1)
    hand_flat(cv, (54, TABLE + 70), 1)
    # right arm (screen left): from the page (cc=0) to under the chin (cc=1)
    elbow = (-112 + 48 * cc, TABLE + 18 - 4 * cc)
    wrist0, wrist1 = np.array((-54.0, TABLE + 72)), np.array((-22.0, 106.0))
    mid = (wrist0 + wrist1) / 2 + (-46, 0)                  # the wrist travels on an arc, out and up
    w = (1 - cc) ** 2 * wrist0 + 2 * (1 - cc) * cc * mid + cc * cc * wrist1
    if cc > 0.02: w = H([w], cc ** 2)[0]
    limb(cv, sh[-1], elbow, tuple(w), side=-1, front=cc > 0.35)
    if cc < 0.6: hand_flat(cv, tuple(w), -1, alpha=1 - sstep_(0.35, 0.6, cc))
    if cc > 0.35: hand_chin(cv, tuple(w), H, alpha=sstep_(0.35, 0.6, cc))

def sstep_(a, b, x):
    u = min(max((x - a) / (b - a), 0.0), 1.0); return u * u * (3 - 2 * u)

def limb(cv, shoulder, elbow, wrist, side, front=False):
    S, E, W_ = np.array(shoulder, np.float64), np.array(elbow, np.float64), np.array(wrist, np.float64)
    def seg(a, b, r0, r1):
        d = b - a; n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        return [tuple(a + n * r0), tuple(b + n * r1), tuple(b - n * r1), tuple(a - n * r0)]
    up = spline(seg(S, E, 22, 17), n=4); fo = spline(seg(E, W_, 17, 13), n=4)
    for poly in (up, fo):
        cv.fill(poly, CLOTH)
        cv.fill(poly, CLOTH_SH, 0.35 if side < 0 else 0.6)
        cv.outline(poly, 2.0, 0.85)
    cv.fill(ellipse_pts(E[0], E[1], 17, 16), CLOTH); cv.fill(ellipse_pts(E[0], E[1], 17, 16), CLOTH_SH, 0.35 if side < 0 else 0.6)
    # ruffled cuff at the wrist
    d = W_ - E; d /= (np.linalg.norm(d) + 1e-9); n = np.array([-d[1], d[0]])
    ruff = []
    for i in range(9):
        u = i / 8 * 2 - 1
        r = 16 + (4 if i % 2 else 0)
        ruff.append(tuple(W_ - d * 4 + n * u * r))
    ruff += [tuple(W_ + d * 8 + n * 14), tuple(W_ + d * 8 - n * 14)]
    rp = spline(ruff[:9] + [ruff[10], ruff[9]])
    cv.fill(rp, CLOTH); cv.outline(rp, 1.4, 0.7)

def hand_flat(cv, wrist, side, alpha=1.0):
    """A hand resting on the page: palm down, fingers together, soft knuckles."""
    x, y = wrist
    s = -side
    pts = [(x, y - 10), (x + s * 18, y - 8), (x + s * 34, y - 2), (x + s * 40, y + 6), (x + s * 34, y + 12), (x + s * 14, y + 14), (x - s * 2, y + 10)]
    hp = spline(pts)
    cv.fill(hp, SKIN, alpha); cv.fill(hp, SKIN_SH, 0.25 * alpha); cv.line(hp, INK, 1.6, 0.8 * alpha, closed=True)
    for k in range(3):
        cv.line([(x + s * (24 + 4 * k), y - 4 + 2 * k), (x + s * (34 + 2 * k), y + 4 + 2 * k)], INK, 1.0, 0.4 * alpha)

def hand_chin(cv, wrist, H, alpha=1.0):
    """Her right hand curled under her chin, the index finger resting up along her jaw."""
    x, y = wrist
    pts = [(x - 14, y + 8), (x - 12, y - 8), (x + 2, y - 16), (x + 18, y - 14), (x + 24, y - 2), (x + 16, y + 12), (x, y + 16)]
    hp = spline(pts)
    cv.fill(hp, SKIN, alpha); cv.fill(hp, SKIN_SH, 0.3 * alpha); cv.line(hp, INK, 1.6, 0.85 * alpha, closed=True)
    fing = spline([(x - 8, y - 10), (x - 12, y - 24), (x - 12, y - 36)], closed=False)
    cv.line(fing, SKIN, 7.0, alpha); cv.line(fing + (3.5, 0), INK, 1.1, 0.6 * alpha); cv.line(fing - (3.5, 0), INK, 1.1, 0.6 * alpha)
    for k in range(2):
        cv.line([(x + 4 + 7 * k, y - 10), (x + 6 + 7 * k, y + 6)], INK, 1.0, 0.45 * alpha)

def origin(scale):
    """Pixel position of the local origin (head centre) inside the layer, and of the hair top."""
    return HALF_W * scale, -TOP * scale

if __name__ == '__main__':
    import sys
    from PIL import Image
    poses = [dict(), dict(chin=1.0, gx=0.6, gy=0.2), dict(look_down=1.0, page=0.5), dict(chin=1.0, yaw=-0.1, blink=1.0),
             dict(chin=0.5, gx=-0.4, tilt=0.06)]
    tiles = []
    for q in poses:
        lay = draw(1.4, **q)
        bg = np.zeros_like(lay[..., :3]); bg[:] = (0.12, 0.10, 0.09)
        rgb = lay[..., :3] + bg * (1 - lay[..., 3:4])
        tiles.append((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    Image.fromarray(np.concatenate(tiles, 1)).save(sys.argv[1])
