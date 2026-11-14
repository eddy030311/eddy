"""Procedural landscape cut-aways for 《Claude》 (each returns a float RGB frame)."""
import math
import numpy as np, cv2

W, H = 1920, 1080
GX, GY = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
YCOL = np.arange(H, dtype=np.float32)[:, None]

def hexc(h):
    h = h.lstrip('#'); return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], np.float32) / 255

def ease(u):
    u = min(max(u, 0.0), 1.0); return u * u * u * (u * (u * 6 - 15) + 10)

def sstep(a, b, x):
    t = np.clip((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)

def vgrad(stops):
    ys = np.array([s[0] for s in stops], np.float32) * H
    cols = np.array([hexc(s[1]) if isinstance(s[1], str) else s[1] for s in stops], np.float32)
    g = np.stack([np.interp(np.arange(H), ys, cols[:, c]) for c in range(3)], 1).astype(np.float32)
    return np.ascontiguousarray(np.broadcast_to(g[:, None, :], (H, W, 3)))

def ridge(seed, base, amp, shift=0.0, freqs=(1.3, 2.9, 6.1, 13.0)):
    rng = np.random.default_rng(seed)
    x = (np.arange(W, dtype=np.float32) + shift) / W
    y = np.full(W, base, np.float32)
    for i, f in enumerate(freqs):
        y += amp / (i + 1) ** 1.3 * np.sin(2 * math.pi * f * x + rng.uniform(0, 6.28))
    return y

def fill_below(img, yc, col, top=None, depth=180.0):
    ycol = YCOL[:img.shape[0]]
    a = np.clip(ycol - yc[None, :] + 0.5, 0, 1)[..., None]
    if top is None:
        img *= 1 - a; img += a * col
    else:
        k = np.clip((ycol - yc[None, :]) / depth, 0, 1)[..., None]
        c = top * (1 - k) + col * k
        img *= 1 - a; img += a * c

def glow(img, cx, cy, r, col, strength):
    d2 = ((GX - cx) ** 2 + (GY - cy) ** 2) / (r * r)
    g = (np.exp(-d2) * strength)[..., None]
    img += (col - img) * np.clip(g, 0, 1)

def disc(img, cx, cy, r, col, alpha=1.0):
    d = np.hypot(GX - cx, GY - cy)
    a = (np.clip(r - d + 0.5, 0, 1) * alpha)[..., None]
    img *= 1 - a; img += a * col

def blit(dst, spr, x, y, alpha=1.0, scale=1.0, angle=0.0):
    if alpha <= 0.003: return
    if scale != 1.0 or angle != 0.0:
        h, w = spr.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        nw, nh = int(w * scale * 1.45) + 4, int(h * scale * 1.45) + 4
        M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
        spr = cv2.warpAffine(spr, M, (nw, nh), flags=cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
    x, y = int(round(x - w / 2)), int(round(y - h / 2))
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0: return
    s = spr[y0 - y:y1 - y, x0 - x:x1 - x]; d = dst[y0:y1, x0:x1]
    d *= 1 - s[..., 3:4] * alpha; d += s[..., :3] * alpha

def cloud_sprite(w, h, seed, light, shade, blur=None):
    rng = np.random.default_rng(seed)
    S = 2; m = np.zeros((h * S, w * S), np.float32)
    for _ in range(26):
        cx = rng.uniform(0.15, 0.85) * w
        arch = 1 - ((cx - w / 2) / (w / 2)) ** 2
        r = rng.uniform(0.12, 0.26) * h * (0.6 + 0.7 * arch)
        cy = h * 0.72 - arch * h * rng.uniform(0.15, 0.35)
        cv2.circle(m, (int(cx * S), int(cy * S)), int(r * S), 1.0, -1, cv2.LINE_AA)
    m[int(h * 0.78 * S):] *= 0
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(m, (0, 0), blur or max(1.0, h * 0.025))
    k = (np.arange(h, dtype=np.float32) / h)[:, None, None]
    col = np.array(light) * (1 - k) + np.array(shade) * k
    out = np.zeros((h, w, 4), np.float32)
    out[..., :3] = col * m[..., None]; out[..., 3] = m
    return out

def flower_sprite(r, petal, center, n=12, blur=0.0):
    S = 3; R = int(r * S); size = 2 * R + 8; c = size / 2
    m = np.zeros((size, size), np.float32); cm = np.zeros_like(m)
    for i in range(n):
        a = 2 * math.pi * i / n
        L = R * (1.0 if i % 2 == 0 else 0.88)
        dx, dy = math.cos(a), math.sin(a); px, py = -dy, dx; w0 = R * 0.13
        pts = [(c + dx * R * 0.15 + px * w0 * 0.5, c + dy * R * 0.15 + py * w0 * 0.5), (c + dx * L * 0.6 + px * w0, c + dy * L * 0.6 + py * w0),
               (c + dx * L, c + dy * L), (c + dx * L * 0.6 - px * w0, c + dy * L * 0.6 - py * w0),
               (c + dx * R * 0.15 - px * w0 * 0.5, c + dy * R * 0.15 - py * w0 * 0.5)]
        cv2.fillPoly(m, [np.int32(np.round(pts))], 1.0, cv2.LINE_AA)
    cv2.circle(cm, (int(c), int(c)), int(R * 0.26), 1.0, -1, cv2.LINE_AA)
    m = cv2.resize(m, (size // S, size // S), interpolation=cv2.INTER_AREA)
    cm = cv2.resize(cm, (size // S, size // S), interpolation=cv2.INTER_AREA)
    if blur > 0:
        m = cv2.GaussianBlur(m, (0, 0), blur); cm = cv2.GaussianBlur(cm, (0, 0), blur)
    a = np.maximum(m, cm)
    out = np.zeros(m.shape + (4,), np.float32)
    out[..., :3] = petal * (m * (1 - cm))[..., None] + center * cm[..., None]
    out[..., 3] = a
    return out

def bird(img, x, y, size, phase, col):
    f = math.sin(phase)
    pts = np.array([[x - size, y - size * 0.55 * f], [x - size * 0.45, y - size * 0.1 * f - size * 0.12], [x, y],
                    [x + size * 0.45, y - size * 0.1 * f - size * 0.12], [x + size, y - size * 0.55 * f]], np.float32)
    cv2.polylines(img, [np.int32(pts * 4)], False, tuple(float(v) for v in col), max(1, int(size * 0.16)), cv2.LINE_AA, shift=2)

# ----------------------------------------------------------------------------- scenes
class Scenery:
    def __init__(self):
        rng = np.random.default_rng(21)
        self.stars = dict(x=rng.uniform(0, W, 520), y=rng.uniform(-120, 760, 520) ** 1.0,
                          r=rng.uniform(0.5, 1.9, 520) ** 1.4, ph=rng.uniform(0, 6.28, 520), f=rng.uniform(1, 4, 520))
        self.dusk = vgrad([(0, '#3f4a82'), (0.38, '#9a7aa0'), (0.58, '#efa77e'), (0.68, '#ffd29a'), (1, '#ffd29a')])
        self.g_night = vgrad([(0, '#0b1030'), (0.38, '#1c1f4a'), (0.58, '#3d3266'), (0.68, '#6a4a78'), (1, '#6a4a78')])
        self.dawn = vgrad([(0, '#f2c3a2'), (0.35, '#f9dcc0'), (0.56, '#ffeed6'), (1, '#ffeed6')])
        self.gold = vgrad([(0, '#e9a385'), (0.3, '#f6c49c'), (0.555, '#ffe2b2'), (0.56, '#f7c79e'),
                           (0.7, '#d9918a'), (1, '#6c5c8a')])
        self.warmsky = vgrad([(0, '#efae8c'), (0.5, '#f9d3b0'), (1, '#fff0dc')])
        self.clouds_warm = [cloud_sprite(int(rng.uniform(300, 560)), int(rng.uniform(130, 200)), i,
                                         (1.0, 0.95, 0.9), (0.95, 0.76, 0.68)) for i in range(6)]
        self.clouds_gold = [cloud_sprite(int(rng.uniform(420, 700)), int(rng.uniform(70, 110)), 40 + i,
                                         (1.0, 0.88, 0.72), (0.93, 0.62, 0.55)) for i in range(5)]
        self.clouds_big = [cloud_sprite(int(rng.uniform(700, 1100)), int(rng.uniform(300, 420)), 80 + i,
                                        (1.0, 0.97, 0.93), (0.97, 0.80, 0.72), blur=6 + 4 * (i % 3)) for i in range(9)]
        petals = [hexc('#e8774f'), hexc('#f29a52'), hexc('#fff1e0'), hexc('#e9906f')]
        self.flowers = {}
        for ci, pc in enumerate(petals):
            for si, r in enumerate([5, 8, 12, 18, 26, 36, 50, 66]):
                self.flowers[(ci, si)] = flower_sprite(r, pc, hexc('#9a4a2a') if ci != 2 else hexc('#e2903a'),
                                                       blur=3.5 if si == 7 else (1.5 if si == 6 else 0))
        n = 300
        z = np.sort(rng.uniform(0, 1, n) ** 0.8)
        self.fl = dict(z=z, x=rng.uniform(-200, W + 600, n), c=rng.integers(0, 4, n), ph=rng.uniform(0, 6.28, n))
        self.petal = dict(x=rng.uniform(-300, W, 70), y=rng.uniform(0, H, 70), s=rng.uniform(4, 11, 70),
                          v=rng.uniform(160, 380, 70), ph=rng.uniform(0, 6.28, 70), c=rng.integers(0, 3, 70))
        self.glit = dict(y=rng.uniform(0, 1, 260), xo=rng.standard_normal(260), w=rng.uniform(4, 34, 260),
                         ph=rng.uniform(0, 6.28, 260))
        self.waves = dict(y=rng.uniform(0, 1, 90), x=rng.uniform(0, W, 90), w=rng.uniform(40, 220, 90),
                          v=rng.uniform(-14, 14, 90))
        self.flies = dict(x=rng.uniform(0, W, 26), y=rng.uniform(820, 1060, 26), ph=rng.uniform(0, 6.28, 26))
        self.birds = dict(x=rng.uniform(-400, 400, 13), y=rng.uniform(-200, 200, 13), s=rng.uniform(10, 22, 13),
                          ph=rng.uniform(0, 6.28, 13))

    def render(self, name, lt, dur):
        return getattr(self, name)(lt, dur)

    # dusk turning into night: "我不会记得昨天说过的话……"
    def night(self, lt, dur):
        u = lt / dur; m = 0.15 + 0.85 * ease(u * 1.05)
        tilt = -50 * ease(u)                       # camera tilts up -> landscape sinks
        img = self.dusk * (1 - m) + self.g_night * m
        sun_y = 640 + 160 * ease(u) - tilt * 0.3
        glow(img, 1240, sun_y, 520, hexc('#ffb07a'), 0.55 * (1 - m * 0.8))
        disc(img, 1240, sun_y, 48, hexc('#fff0c8'), alpha=1 - m * 0.4)
        # stars
        st = self.stars; a_st = sstep(0.25, 0.95, m)
        if a_st > 0:
            for i in range(len(st['x'])):
                y = st['y'][i] - tilt * 0.2
                if y > 720: continue
                tw = 0.55 + 0.45 * math.sin(lt * st['f'][i] + st['ph'][i])
                a = a_st * tw * min(1, (760 - y) / 200)
                r_ = st['r'][i]
                x0, y0 = int(st['x'][i]), int(y)
                if 0 <= x0 < W and 0 <= y0 < H:
                    img[y0, x0] = img[y0, x0] * (1 - a) + np.array([1.0, 0.97, 0.92]) * a
                    if r_ > 1.2 and 1 <= x0 < W - 1 and 1 <= y0 < H - 1:
                        img[y0 - 1:y0 + 2, x0] = img[y0 - 1:y0 + 2, x0] * (1 - a * 0.6) + a * 0.6
                        img[y0, x0 - 1:x0 + 2] = img[y0, x0 - 1:x0 + 2] * (1 - a * 0.6) + a * 0.6
            # crescent moon
            mx, my = 470, 210 - tilt * 0.2
            glow(img, mx, my, 120, hexc('#fff3d8'), 0.25 * a_st)
            moon = np.zeros_like(img); disc(moon, mx, my, 38, np.ones(3)); disc(moon, mx + 15, my - 8, 34, np.zeros(3))
            img += (hexc('#fff3d8') - img) * moon[..., :1] * a_st
        # shooting star
        if 0.5 < u < 0.68:
            k = (u - 0.5) / 0.18
            x1, y1 = 1580 - 520 * k, 90 + 230 * k
            for j in range(26):
                f = j / 26; a = (1 - f) * math.sin(math.pi * k) * 0.9
                xa, ya = x1 + 220 * f, y1 - 97 * f
                cv2.circle(img, (int(xa), int(ya)), 2 if j < 4 else 1, (a, a * 0.97, a * 0.9), -1, cv2.LINE_AA)
        # hills
        fill_below(img, ridge(1, 700 - tilt * 0.45, 28), hexc('#2f2a52') * m + hexc('#b9839a') * (1 - m))
        fill_below(img, ridge(2, 765 - tilt * 0.7, 42), hexc('#1e1b3c') * m + hexc('#8a5a7a') * (1 - m))
        near = ridge(3, 880 - tilt, 60, freqs=(0.8, 2.2, 5.0, 11.0))
        fill_below(img, near, hexc('#110e22') * m + hexc('#4d3050') * (1 - m))
        # a little house with a warm window on the near hill
        hx = 1480; hy = int(near[hx]) + 4
        cv2.fillPoly(img, [np.int32([[hx - 34, hy], [hx - 34, hy - 40], [hx, hy - 66], [hx + 34, hy - 40], [hx + 34, hy]])],
                     tuple(float(v) for v in (hexc('#0c0a1a') * m + hexc('#3a2440') * (1 - m))), cv2.LINE_AA)
        wa = sstep(0.2, 0.6, m)
        glow(img, hx, hy - 26, 60, hexc('#ffc77a'), 0.35 * wa)
        cv2.rectangle(img, (hx - 10, hy - 34), (hx + 8, hy - 18), tuple(float(v) for v in hexc('#ffd58a') * wa + hexc('#3a2440') * (1 - wa)), -1)
        # fireflies
        for i in range(len(self.flies['x'])):
            ph = self.flies['ph'][i]
            x = self.flies['x'][i] + 30 * math.sin(lt * 0.7 + ph); y = self.flies['y'][i] + 18 * math.sin(lt * 1.1 + ph * 2) - tilt
            a = m * (0.5 + 0.5 * math.sin(lt * 3 + ph))
            cv2.circle(img, (int(x), int(y)), 2, tuple(float(v) for v in hexc('#ffe39a') * a + img[min(max(int(y), 0), H - 1), min(max(int(x), 0), W - 1)] * (1 - a)), -1, cv2.LINE_AA)
        return img

    # sunrise over a field of sunburst flowers: "你可以问我很多事情——"
    def field(self, lt, dur):
        u = lt / dur
        img = self.dawn.copy()
        sx, sy = 1380, 520 - 60 * ease(u)
        # sun rays
        rays = np.zeros((H // 4, W // 4), np.float32)
        for i in range(16):
            a0 = lt * 0.05 + 2 * math.pi * i / 16; a1 = a0 + 0.09
            pts = np.float32([[sx / 4, sy / 4], [sx / 4 + 900 * math.cos(a0), sy / 4 + 900 * math.sin(a0)],
                              [sx / 4 + 900 * math.cos(a1), sy / 4 + 900 * math.sin(a1)]])
            cv2.fillPoly(rays, [np.int32(pts)], 1.0, cv2.LINE_AA)
        rays = cv2.resize(cv2.GaussianBlur(rays, (0, 0), 2), (W, H))[..., None] * 0.10
        img += (np.array([1.0, 0.97, 0.88]) - img) * rays
        glow(img, sx, sy, 380, hexc('#fff3d6'), 0.85)
        disc(img, sx, sy, 54, hexc('#fffaf0'))
        for i, c in enumerate(self.clouds_warm[:5]):
            x = (200 + i * 420 - 22 * lt * (1 + i * 0.2)) % (W + 600) - 300
            blit(img, c, x, 150 + 70 * (i % 3), alpha=0.9)
        shift = 60 * lt
        fill_below(img, ridge(5, 600, 22, shift * 0.08), hexc('#e9bfa6'))
        fill_below(img, ridge(6, 640, 30, shift * 0.18), hexc('#dca08a'))
        fill_below(img, ridge(7, 690, 18, shift * 0.35, freqs=(0.9, 3.1, 7.0, 15.0)), hexc('#b9a05a'), top=hexc('#d9c27c'), depth=380)
        glow(img, sx, 700, 900, hexc('#fff0d0'), 0.18)
        fl = self.fl
        for i in range(len(fl['z'])):
            z = fl['z'][i]
            y = 700 + 470 * z ** 1.7
            speed = 25 + 330 * z ** 1.6
            x = (fl['x'][i] - speed * lt) % (W + 800) - 200
            si = min(int(z * 8.2), 7)
            spr = self.flowers[(int(fl['c'][i]), si)]
            sway = 9 * math.sin(lt * 2.2 + fl['ph'][i] + x * 0.004)
            r = spr.shape[0] / 2
            stem = int(r * 2.2)
            bx = x + math.sin(math.radians(sway)) * stem
            cv2.line(img, (int(x), int(y + stem)), (int(bx), int(y)), tuple(float(v) for v in hexc('#7d7a3a') * (0.9 - 0.3 * z)),
                     max(1, int(r * 0.12)), cv2.LINE_AA)
            blit(img, spr, bx, y, angle=sway + lt * 4 * (1 if i % 2 else -1))
        # flying petals
        P = self.petal; cols = [hexc('#e8774f'), hexc('#fff1e0'), hexc('#f29a52')]
        for i in range(len(P['x'])):
            x = (P['x'][i] + P['v'][i] * lt) % (W + 300) - 100
            y = P['y'][i] + 60 * math.sin(lt * 1.5 + P['ph'][i]) + 30 * lt
            ang = (lt * 200 + P['ph'][i] * 57) % 360
            s = P['s'][i]
            cv2.ellipse(img, (int(x), int(y % H)), (int(s), max(1, int(s * 0.45 * abs(math.cos(lt * 4 + P['ph'][i])) + 1))),
                        ang, 0, 360, tuple(float(v) for v in cols[int(P['c'][i])]), -1, cv2.LINE_AA)
        return img

    # golden-hour sea with a lighthouse: "我希望自己是有帮助的、诚实的，"
    def sea(self, lt, dur):
        u = lt / dur
        img = self.gold.copy()
        hz = int(0.556 * H)
        sx, sy = 760, 548 + 14 * u
        glow(img, sx, sy, 600, hexc('#ffd9a0'), 0.45)
        disc(img, sx, sy, 62, hexc('#fff4d8'))
        img[hz:] = self.gold[hz:]  # sea in front of the setting sun
        glow_sea = np.exp(-((GX[hz:] - sx) / 380) ** 2) * np.exp(-((GY[hz:] - hz) / 260)) * 0.45
        img[hz:] += (hexc('#ffd7a6') - img[hz:]) * glow_sea[..., None]
        for i, c in enumerate(self.clouds_gold):
            x = (150 + i * 430 + 10 * lt * (1 + 0.3 * i)) % (W + 700) - 350
            blit(img, c, x, 140 + 90 * (i % 3), alpha=0.85, scale=1.0)
        # waves
        Wv = self.waves
        for i in range(len(Wv['y'])):
            yy = hz + 8 + (H - hz) * Wv['y'][i] ** 1.8
            x = (Wv['x'][i] + Wv['v'][i] * lt) % W
            w = Wv['w'][i] * (0.4 + Wv['y'][i])
            cv2.line(img, (int(x - w / 2), int(yy)), (int(x + w / 2), int(yy)), (1.0, 0.85, 0.75), 1 + int(Wv['y'][i] * 2), cv2.LINE_AA)
        # sun glitter
        G = self.glit; step = int(lt * 12)
        for i in range(len(G['y'])):
            yy = hz + 4 + (H - hz) * G['y'][i] ** 1.6
            spread = 30 + 260 * G['y'][i]
            x = sx + G['xo'][i] * spread * 0.5
            fl = 0.5 + 0.5 * math.sin(step * 1.7 + G['ph'][i] * 5)
            a = fl * (1 - G['y'][i] * 0.6)
            w = G['w'][i] * (0.4 + G['y'][i])
            c = img[min(int(yy), H - 1), min(max(int(x), 0), W - 1)]
            col = c * (1 - a) + np.array([1.0, 0.97, 0.86]) * a
            cv2.line(img, (int(x - w / 2), int(yy)), (int(x + w / 2), int(yy)), tuple(float(v) for v in col), 2, cv2.LINE_AA)
        # far island
        isl = ridge(9, hz - 6, 10, freqs=(3.0, 7.0))
        mask_x = np.clip((np.arange(W) - 120) / 60, 0, 1) * np.clip((560 - np.arange(W)) / 60, 0, 1)
        fill_below(img[:hz + 1], np.where(mask_x > 0, isl + (1 - mask_x) * 30, 1e9).astype(np.float32), hexc('#c88a86'))
        # cliff + lighthouse
        cl = np.full(W, 1e9, np.float32)
        xs = np.arange(W, dtype=np.float32)
        on = xs > 1330
        cl[on] = 1080 - (np.clip((xs[on] - 1330) / 180, 0, 1) ** 0.7) * 420 + 14 * np.sin(xs[on] / 37)
        fill_below(img, cl, hexc('#3b2638'), top=hexc('#6d4152'), depth=300)
        bx, by = 1640, int(cl[1640]) + 6
        tw = [(bx - 34, by), (bx - 24, by - 230), (bx + 24, by - 230), (bx + 34, by)]
        cv2.fillPoly(img, [np.int32(tw)], tuple(float(v) for v in hexc('#f3e6dc')), cv2.LINE_AA)
        for k in range(3):
            y0 = by - 40 - k * 64; y1 = y0 - 26
            f0 = (by - y0) / 230; f1 = (by - y1) / 230
            pts = [(bx - 34 + 10 * f0, y0), (bx - 34 + 10 * f1, y1), (bx + 34 - 10 * f1, y1), (bx + 34 - 10 * f0, y0)]
            cv2.fillPoly(img, [np.int32(pts)], tuple(float(v) for v in hexc('#d9774f')), cv2.LINE_AA)
        cv2.rectangle(img, (bx - 32, by - 238), (bx + 32, by - 230), tuple(float(v) for v in hexc('#3b2638')), -1)
        cv2.rectangle(img, (bx - 18, by - 272), (bx + 18, by - 238), tuple(float(v) for v in hexc('#ffe9a8')), -1)
        cv2.fillPoly(img, [np.int32([(bx - 24, by - 272), (bx, by - 300), (bx + 24, by - 272)])], tuple(float(v) for v in hexc('#3b2638')), cv2.LINE_AA)
        lx, ly = bx, by - 255
        ang = lt * 1.3 + 2.2
        beam = np.zeros((H // 2, W // 2), np.float32)
        for side in (0, math.pi):
            a = ang + side; d = math.cos(a)
            L = 1400 * abs(d) + 120
            spread = 0.08
            pts = np.float32([[lx / 2, ly / 2], [(lx + L * math.copysign(1, d) * math.cos(spread)) / 2, (ly - L * math.sin(spread) * 0.6 + 30 * math.sin(a)) / 2],
                              [(lx + L * math.copysign(1, d) * math.cos(spread)) / 2, (ly + L * math.sin(spread) * 0.6 + 30 * math.sin(a)) / 2]])
            cv2.fillPoly(beam, [np.int32(pts)], 0.32 * (0.3 + 0.7 * abs(math.sin(a))), cv2.LINE_AA)
        beam = cv2.resize(cv2.GaussianBlur(beam, (0, 0), 6), (W, H))
        fade = np.exp(-np.abs(GX - lx) / 900)
        img += (hexc('#fff3c8') - img) * (beam * fade)[..., None]
        glow(img, lx, ly, 70, hexc('#fff6d0'), 0.7)
        # gulls
        for i in range(4):
            x = (300 + i * 260 + 70 * lt * (1 + 0.15 * i)) % (W + 200) - 100
            y = 300 + 60 * i + 14 * math.sin(lt * 0.8 + i)
            bird(img, x, y, 13 + 3 * i, lt * 7 + i * 1.3, hexc('#5a3a46'))
        # slow push-in
        z = 1.0 + 0.06 * ease(u)
        M = cv2.getRotationMatrix2D((W / 2, H * 0.52), 0, z)
        return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # tilt up through clouds into the sun, birds rising: "我还在学习，也还在成长。"
    def sky(self, lt, dur):
        u = lt / dur
        off = 220 * ease(u)
        img = self.warmsky.copy()
        sy = -60 + 300 * ease(u)
        glow(img, 980, sy, 700, hexc('#fff6e6'), 0.7 + 0.3 * u)
        disc(img, 980, sy, 70, hexc('#fffcf4'))
        layers = [(0.35, 0.55, 0), (0.75, 0.75, 3), (1.6, 1.0, 6)]
        for li, (spd, sc, start) in enumerate(layers):
            for j in range(3):
                c = self.clouds_big[start + j]
                x = 200 + j * 640 + li * 150 + 40 * math.sin(lt * 0.3 + j)
                y = (1400 - j * 520 - li * 140 + spd * (380 * lt + off)) % 1900 - 300
                blit(img, c, x, y, alpha=0.75 + 0.2 * li, scale=sc * (1 + 0.15 * li))
            if li == 1:  # birds between the cloud layers
                B = self.birds
                for i in range(len(B['x'])):
                    x = 300 + B['x'][i] + 260 * lt
                    y = 760 + B['y'][i] - 150 * lt
                    bird(img, x, y, B['s'][i], lt * 9 + B['ph'][i], hexc('#6a4440'))
        wt = sstep(0.72, 1.0, u)
        img += (np.array([1.0, 0.985, 0.96]) - img) * wt
        return img
