"""序 · before the first word: someone at a desk at night, a laptop, a blinking caret, a cassette by the keyboard.

The camera looks over their shoulder and pushes into the screen until the caret on it is the caret the film
opens on (same place, same size, same blink): the film's first shot is now inside this room.
Flat style, like the readers in the library: no face, a silhouette lit only by the screen.
"""
import math
import numpy as np, cv2
from core import *
from ui import window, draw_caret, cursor_alpha

DUR = 15.0
SC = np.array([1010.0, 410.0])          # the laptop screen's centre (base layout)
K0 = 0.36                               # the on-screen window is the film's hero window at this scale
WIN = (960, 540, 880, 300)              # = interface.WIN
CARET_FILM = np.array([WIN[0] - WIN[2] / 2 + 44, WIN[1] + 5])
CARET = SC + K0 * (CARET_FILM - np.array([WIN[0], WIN[1]]))
VOID_OFF = 1.5                          # the 'void' shot starts 1.5 s into its own clock

def push(lt):
    return ease(clamp((lt - 8.6) / (DUR - 8.6 - 0.4)))

def xf(lt, layer=1.0):
    """Base-layout point -> frame point. layer > 1: nearer the camera, so it grows faster and leaves first."""
    e = push(lt)
    zs = 1.0 + 0.045 * ease(clamp(lt / 8.6))                     # the slow drift-in before the push
    z = zs * (1 + (1 / K0 - 1) * e)
    zl = 1 + (z - 1) * layer
    c = SC + (np.array([CX, CY]) - SC) * e
    def T(p):
        p = np.asarray(p, np.float64)
        return c + (p - SC) * zl
    return T, zl

def poly(img, T, pts, col, a=1.0):
    q = np.int32(np.round(T(np.array(pts)) * 8))
    if a >= 0.999:
        cv2.fillPoly(img, [q], tuple(float(v) for v in col), cv2.LINE_AA, shift=3)
    else:
        m = np.zeros(img.shape[:2], np.float32); cv2.fillPoly(m, [q], 1.0, cv2.LINE_AA, shift=3)
        img[:] = img * (1 - a * m[..., None]) + np.asarray(col, np.float32) * a * m[..., None]

def ellipse(img, T, c, ax, ay, col, ang=0.0, z=1.0):
    p = T(np.array([c]))[0]
    cv2.ellipse(img, (int(p[0] * 8), int(p[1] * 8)), (max(1, int(ax * z * 8)), max(1, int(ay * z * 8))), ang, 0, 360,
                tuple(float(v) for v in col), -1, cv2.LINE_AA, shift=3)

def city(img, T, lt, z, glow):
    """The window on the right: a night city, a few windows still lit."""
    x0, y0, x1, y1 = 1500, 170, 1880, 560
    poly(img, T, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], (0.020, 0.026, 0.050))
    rng = np.random.default_rng(5)
    for i in range(9):                                            # building silhouettes
        bx = x0 + i * 44 + rng.uniform(-6, 6); bh = rng.uniform(80, 260)
        poly(img, T, [(bx, y1), (bx + 46, y1), (bx + 46, y1 - bh), (bx, y1 - bh)], (0.012, 0.014, 0.024))
        for j in range(int(bh // 22)):
            if rng.random() < 0.22:
                on = 1.0 if (rng.random() > 0.15 or math.sin(lt * 0.7 + i + j) > -0.6) else 0.0
                wx, wy = bx + 8 + rng.integers(0, 3) * 12, y1 - 16 - j * 22
                col = np.array([0.55, 0.40, 0.20]) * rng.uniform(0.4, 1.0) * on
                poly(img, T, [(wx, wy), (wx + 7, wy), (wx + 7, wy + 9), (wx, wy + 9)], col)
    for xx in (x0, (x0 + x1) / 2, x1):                            # the frame
        poly(img, T, [(xx - 5, y0), (xx + 5, y0), (xx + 5, y1), (xx - 5, y1)], (0.03, 0.028, 0.03))
    poly(img, T, [(x0, (y0 + y1) / 2 - 4), (x1, (y0 + y1) / 2 - 4), (x1, (y0 + y1) / 2 + 4), (x0, (y0 + y1) / 2 + 4)], (0.03, 0.028, 0.03))

def cassette(img, T, z, light):
    """The mixtape: a smoky shell, two reels, a hand-written label."""
    c = np.array([1330.0, 705.0]); w, h = 150, 94; ang = math.radians(-9)
    R = np.array([[math.cos(ang), -math.sin(ang)], [math.sin(ang), math.cos(ang)]])
    P = lambda pts: [tuple(c + R @ np.array(p)) for p in pts]
    poly(img, T, P([(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]), np.array([0.10, 0.09, 0.09]) * light)
    poly(img, T, P([(-w / 2 + 8, -h / 2 + 8), (w / 2 - 8, -h / 2 + 8), (w / 2 - 8, h / 2 - 30), (-w / 2 + 8, h / 2 - 30)]),
         np.array([0.80, 0.72, 0.58]) * light)                                            # the label
    poly(img, T, P([(-w / 2 + 30, -6), (w / 2 - 30, -6), (w / 2 - 30, 12), (-w / 2 + 30, 12)]), np.array([0.05, 0.045, 0.045]) * light)
    for sx in (-1, 1):
        ellipse(img, T, c + R @ np.array([sx * 26.0, 3.0]), 9, 9, np.array([0.22, 0.20, 0.18]) * light, z=z)
        ellipse(img, T, c + R @ np.array([sx * 26.0, 3.0]), 4, 4, np.array([0.03, 0.03, 0.03]) * light, z=z)
    poly(img, T, P([(-w / 2 + 34, h / 2 - 22), (w / 2 - 34, h / 2 - 22), (w / 2 - 26, h / 2), (-w / 2 + 26, h / 2)]), np.array([0.08, 0.075, 0.075]) * light)
    spr = text_sprite('mix for you — L.', 'serif_l', max(6, int(15 * z)), tuple(np.array([0.25, 0.18, 0.14])))
    p = T(np.array([c + R @ np.array([0.0, -h / 2 + 22])]))[0]
    rs = cv2.warpAffine(spr, cv2.getRotationMatrix2D((spr.shape[1] / 2, spr.shape[0] / 2), 9, 1.0), (spr.shape[1], spr.shape[0]))
    blit(img, rs * light, p[0], p[1])

DRAFT = "hey. can I ask you something"
TYPE_T = (2.4, 4.9)        # she types a first line...
DEL_T = (6.2, 7.0)         # ...looks at it, and deletes it
def draft_shown(lt):
    n = len(DRAFT)
    if lt < TYPE_T[0]: return 0
    if lt < TYPE_T[1]: return int(n * (lt - TYPE_T[0]) / (TYPE_T[1] - TYPE_T[0]))
    if lt < DEL_T[0]: return n
    if lt < DEL_T[1]: return int(n * (1 - (lt - DEL_T[0]) / (DEL_T[1] - DEL_T[0])))
    return 0

def _spline(pts, closed=True, n=8):
    P = np.asarray(pts, np.float64)
    P = np.vstack([P[-1], P, P[0], P[1]]) if closed else np.vstack([P[0], P, P[-1]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n, endpoint=False):
            out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.array(out)

HEAD = np.array([700.0, 520.0])      # her head centre in the base layout
HS = 1.05                            # local units -> base layout

def person(lt, T, z, T1=None, z1=1.0):
    """A slender girl at the desk, seen from behind (after the producer's reference): long wavy hair down her back,
    a ribbon bow on each side, small puff sleeves, slim arms to the keyboard (hidden by her back, elbows at her sides).
    Pure silhouette: everything she is lives in the outline the screen catches. Local units: head centre (0,0)."""
    m = np.zeros((H, W), np.float32)
    typing = (TYPE_T[0] < lt < TYPE_T[1]) or (DEL_T[0] < lt < DEL_T[1])
    br = 1.2 * math.sin(lt * 2 * math.pi / 4.2)
    nod = (0.8 * math.sin(lt * 5.5) if typing else 0.0)
    def L(pts, dy=0.0):
        p = np.asarray(pts, np.float64) * HS + HEAD + np.array([0.0, dy])
        return T(p)
    def fill(pts, dy=0.0):
        cv2.fillPoly(m, [np.int32(np.round(L(pts, dy) * 8))], 1.0, cv2.LINE_AA, 3)
    def ell(c, ax, ay, ang=0.0, dy=0.0):
        p = L([c], dy)[0]
        cv2.ellipse(m, (int(p[0] * 8), int(p[1] * 8)), (max(1, int(ax * HS * z * 8)), max(1, int(ay * HS * z * 8))), ang, 0, 360, 1.0, -1, cv2.LINE_AA, 3)
    # body: slim neck, sloping shoulders, a narrow back to the waist
    fill(_spline([(-14, 50), (14, 50), (20, 112), (62, 128), (96, 146), (104, 200), (92, 260), (84, 330), (86, 420),
                  (-86, 420), (-84, 330), (-92, 260), (-104, 200), (-96, 146), (-62, 128), (-20, 112)]), dy=br * 0.4)
    # small puff sleeves at the shoulders, a few soft folds on top
    for sd in (-1, 1):
        ell((sd * 104, 158), 30, 26, sd * 12, dy=br * 0.5)
        ell((sd * 96, 138), 13, 9, sd * 25, dy=br * 0.5)
        ell((sd * 120, 144), 11, 8, sd * -15, dy=br * 0.5)
    # slim upper arms down to the elbows at her sides (the forearms go forward, hidden by her back)
    tap = 1.4 * max(0.0, math.sin(lt * 11.0)) if typing else 0.0
    for sd in (-1, 1):
        dy = tap * (sd > 0)
        fill(_spline([(sd * 96, 170), (sd * 120, 176), (sd * 122, 240), (sd * 116, 292), (sd * 100, 296), (sd * 98, 240)]), dy=dy)
    # the head and the long wavy hair, one outline: around the crown, down the sides, flowing down her back
    yy = np.linspace(20, 430, 34)
    def edge(y, sd):
        w = 62 + 6 * sstep(20, 70, y) + 16 * sstep(110, 250, y) + 6 * sstep(250, 400, y)
        wave = 4.0 * math.sin(y / 26 + (1.3 if sd > 0 else 0.2) + lt * 0.5) + 2.0 * math.sin(y / 9.0 + (2.0 if sd > 0 else 0.7))
        return sd * (w + wave)
    crown = [(70 * math.cos(a_), -6 - 78 * math.sin(a_)) for a_ in np.linspace(0.05 * math.pi, 0.95 * math.pi, 13)]
    right = [(edge(y, 1), y) for y in yy]
    left = [(edge(y, -1), y) for y in yy[::-1]]
    hair = np.vstack([np.array(right), np.array(left), np.array(crown[::-1])])
    fill(_spline(hair, n=4), dy=nod)
    # a few loose curls breaking the outline, low on the back
    for (y0, sd) in ((230, -1), (290, 1), (350, -1)):
        x0 = edge(y0, sd)
        pts = [(x0 + sd * (5 * math.sin(k / 4 * math.pi) + 1.5 * k), y0 + 6 * k) for k in range(6)]
        cv2.polylines(m, [np.int32(np.round(L(pts, nod) * 8))], False, 1.0, max(1, int(round(3.0 * z))), cv2.LINE_AA, 3)
    # a ribbon bow on each side, tied on the hair just behind the ears: two loops like small wings, a knot, two tails
    for sd in (-1, 1):
        bx, by = sd * 60, 26
        loop_up = _spline([(bx, by), (bx + sd * 22, by - 26), (bx + sd * 48, by - 22), (bx + sd * 44, by - 4), (bx + sd * 14, by - 2)])
        loop_dn = _spline([(bx, by + 2), (bx + sd * 16, by + 6), (bx + sd * 40, by + 20), (bx + sd * 30, by + 32), (bx + sd * 10, by + 14)])
        fill(loop_up, dy=nod); fill(loop_dn, dy=nod)
        ell((bx + sd * 4, by + 2), 9, 10, dy=nod)
        sw = 3 * math.sin(lt * 0.8 + sd)
        for (dx_, ln, w_) in ((sd * 20, 120, 9), (sd * 8, 100, 8)):
            tx = bx + dx_ + sw
            fill([(bx + sd * 2 - w_ / 2, by + 8), (bx + sd * 2 + w_ / 2, by + 8), (tx + w_ / 2, by + ln), (tx, by + ln - 8), (tx - w_ / 2, by + ln)], dy=nod)
    return m

def render(shot, lt, dur):
    img = np.empty((H, W, 3), np.float32); img[:] = np.array([0.018, 0.019, 0.026], np.float32)
    T, z = xf(lt, 1.0)
    Tf, zf = xf(lt, 2.2)                       # the person, nearest the camera
    Tb, zb = xf(lt, 0.85)                      # the far wall and window
    e = push(lt)
    blink = cursor_alpha(lt - DUR + VOID_OFF)  # the same blink the film opens on
    screen = 0.8 + 0.2 * math.sin(lt * 0.9) * 0.1
    city(img, Tb, lt, zb, screen)
    # the desk lamp, switched off; the mug
    poly(img, Tb, [(250, 700), (330, 700), (300, 470), (290, 470)], (0.03, 0.028, 0.03))
    poly(img, Tb, [(220, 470), (380, 470), (340, 400), (260, 400)], (0.035, 0.04, 0.035))
    # the desk, lit in a pool by the screen
    poly(img, T, [(-400, 690), (2400, 690), (2400, 1400), (-400, 1400)], (0.050, 0.034, 0.024))
    gx, gy = grid(W, H)
    pc = T(np.array([[1010.0, 690.0]]))[0]
    pool = np.exp(-(((gx - pc[0]) / (620 * z)) ** 2 + ((gy - pc[1]) / (180 * z)) ** 2))
    deskm = (gy > T(np.array([[0.0, 690.0]]))[0][1]).astype(np.float32)
    img += (pool * deskm)[..., None] * np.array([0.17, 0.14, 0.11], np.float32) * screen
    ellipse(img, T, (460.0, 690.0), 44, 12, np.array([0.12, 0.10, 0.09]) * screen, z=z)
    poly(img, T, [(416, 690), (504, 690), (500, 610), (420, 610)], np.array([0.10, 0.085, 0.075]) * screen)
    ellipse(img, T, (460.0, 610.0), 40, 10, np.array([0.05, 0.04, 0.035]), z=z)
    # the laptop: base and keys, the lid, the screen and what is on it
    poly(img, T, [(700, 612), (1320, 612), (1390, 690), (630, 690)], (0.085, 0.085, 0.092))
    for r in range(4):
        for c in range(14):
            u0 = 0.06 + c * 0.064; v0 = 0.14 + r * 0.19
            xa = 700 + (630 - 700) * v0 + (620 + 140 * v0) * u0; ya = 612 + 78 * v0
            poly(img, T, [(xa, ya), (xa + 34, ya), (xa + 34, ya + 11), (xa, ya + 11)], np.array([0.045, 0.045, 0.05]) + 0.05 * screen)
    poly(img, T, [(700, 222), (1320, 222), (1320, 612), (700, 612)], (0.05, 0.05, 0.056))
    poly(img, T, [(716, 236), (1304, 236), (1304, 596), (716, 596)], np.array([0.075, 0.070, 0.090]) * (0.9 + 0.1 * screen))
    wc = T(np.array([[SC[0], SC[1]]]))[0]
    sk = K0 * z
    n_ = draft_shown(lt)
    lines = [dict(role='human', text=DRAFT, shown=n_)] if n_ > 0 else []
    window(img, wc[0], wc[1], WIN[2] * sk, WIN[3] * sk, lines, 0.0, cursor=None, alpha=1 - sstep(0.85, 1.0, e), border=0.8, scale_text=sk)
    cp = T(np.array([CARET]))[0]
    if n_ == 0:
        draw_caret(img, cp[0], cp[1], max(1, 3 * sk), 32 * sk, blink)
    # the screen's light falling on the room
    glow_spot(img, wc[0], wc[1], 380 * z, np.array([0.55, 0.48, 0.42]), 0.18 * screen * (1 - e))
    cassette(img, T, z, screen * 1.15)
    # them: a silhouette, rim-lit by the screen on the side that faces it
    if zf < 6:
        m = person(lt, Tf, zf, T, z)
        sh = np.float32([[1, 0, -7 * zf], [0, 1, 5 * zf]])
        inner = cv2.warpAffine(m, sh, (W, H))
        rim = np.clip(m - inner, 0, 1)
        rim = cv2.GaussianBlur(rim, (0, 0), 1.2 * zf)
        img *= (1 - m)[..., None]
        img += (m * 0.010)[..., None] + (rim * 0.55 * screen)[..., None] * np.array([0.75, 0.68, 0.62], np.float32)
    # into the caret: the room and the window give way to the dark the film begins in
    out = sstep(0.82, 1.0, e)
    if out > 0:
        base = np.empty_like(img); base[:] = VOID
        draw_caret(base, CARET_FILM[0], CARET_FILM[1], 3, 32, blink)
        img = img * (1 - out) + base * out
    k = sstep(0.0, 1.4, lt)
    return img * k + VOID * (1 - k)
