"""VI · 此刻 — rebuilt in the film's own language.

The void and the heartbeat caret; then a slow drift past floating conversation windows, each one replaying a
moment from earlier in the film (a flashback held in glass); then every window draws back into a ring and a
single empty window — this moment — rises in the centre, her face surfacing in its glass like a reflection;
a burst of full-screen flashbacks (handled by film.FLASHBACKS); finally her open palm and the light it holds,
which we push into until it is the single point the next shot begins from.
"""
import math, os
import numpy as np, cv2
from PIL import Image
from core import *
import ui
from ui import cursor_alpha, draw_caret, rounded_rect_mask
import timeline as TL
import character as CH

RAW = None
def set_raw(f):
    global RAW
    RAW = f

# (appears at rel, source shot, source rel time, world x, world y, accent)
WINDOWS = [
    (3.6, 'hall', 5.0, -0.55, 0.12, '#F2A65A'),
    (6.8, 'montage', 6.3, 0.56, -0.10, '#8FE3FF'),
    (10.0, 'answer', 3.6, -0.50, 0.08, '#F2A65A'),
    (13.0, 'slit', 4.0, 0.52, -0.06, '#FFD9A8'),
    (15.4, 'close', 2.8, -0.48, 0.10, '#FFB46B'),
    (17.4, 'keyboard', 4.0, 0.50, -0.08, '#FFB46B'),
]
F_ = 900.0
CONVERGE, NOWWIN, BURST, PALM = 20.4, 20.8, 26.6, 28.8

def z_of(i): return 3.4 + 2.7 * i
def cam_z(lt):
    keys_t = [0.0, 3.0] + [w[0] + 1.2 for w in WINDOWS] + [CONVERGE + 0.5]
    keys_z = [-1.0, 0.2] + [z_of(i) - 2.0 for i in range(len(WINDOWS))] + [z_of(len(WINDOWS) - 1) - 1.8]
    lt_c = min(max(lt, keys_t[0]), keys_t[-1])
    j = max(0, min(int(np.searchsorted(keys_t, lt_c)) - 1, len(keys_t) - 2))
    u = (lt_c - keys_t[j]) / max(keys_t[j + 1] - keys_t[j], 1e-6)
    return keys_z[j] + (keys_z[j + 1] - keys_z[j]) * ease(u)

def grade(img):
    lum = img.mean(axis=2, keepdims=True)
    return (img * 0.45 + lum * np.array([1.12, 0.92, 0.70], np.float32) * 0.75).astype(np.float32)

_still = {}
def content(i, lt, live):
    _, key, rel, _, _, _ = WINDOWS[i]
    if live:
        src = RAW(TL.at(key, rel) + (lt - WINDOWS[i][0]))
    else:
        if i not in _still:
            _still[i] = RAW(TL.at(key, rel + 0.6))
        src = _still[i]
    act = src[TOP:BOT]
    return grade(cv2.resize(act, (960, 400), interpolation=cv2.INTER_AREA))

def glass(img, cx, cy, w, h, cont, alpha, accent, reflect=None, reflect_a=0.0):
    w, h = int(w), int(h)
    if w < 12 or h < 6 or alpha <= 0.01: return
    m = rounded_rect_mask(w, h, max(4, int(h * 0.06)))
    pic = cv2.resize(cont, (w, h), interpolation=cv2.INTER_AREA) * 0.9
    if reflect is not None and reflect_a > 0:
        pic = pic * (1 - reflect_a * 0.35) + cv2.resize(reflect, (w, h)) * reflect_a
    lay = np.zeros((h, w, 4), np.float32)
    lay[..., :3] = pic * m[..., None]; lay[..., 3] = m
    # glow and edge, as every window in the film
    pad = 36
    gl = np.zeros((h + 2 * pad, w + 2 * pad), np.float32); gl[pad:pad + h, pad:pad + w] = m
    gl = cv2.GaussianBlur(gl, (0, 0), 14) * 0.30 * alpha
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    ui._add_layer(img, x0 - pad, y0 - pad, accent[None, None, :] * gl[..., None])
    edge = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1) * 0.6
    lay[..., :3] += accent * edge[..., None]; lay[..., 3] = np.maximum(lay[..., 3], edge)
    blit(img, lay, x0, y0, alpha=alpha, anchor='tl')

_face = None
def face_reflection():
    """Her face as a reflection: amber monochrome, soft-edged."""
    global _face
    if _face is None:
        im = np.asarray(Image.open('build/her_face.png').convert('RGBA')).astype(np.float32) / 255
        rgb, a = im[..., :3], im[..., 3:]
        lum = rgb.mean(axis=2, keepdims=True)
        tint = lum * np.array([1.05, 0.78, 0.50], np.float32)
        hh, ww = a.shape[:2]
        yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
        soft = np.clip(1 - np.hypot((xx - ww / 2) / (ww * 0.55), (yy - hh / 2) / (hh * 0.6)), 0, 1)[..., None] ** 0.8
        face = tint * a * soft
        canvas = np.zeros((400, 960, 3), np.float32)        # placed in the right half of the glass
        f = cv2.resize(face, (int(ww * 400 / hh), 400))
        x0 = 960 - f.shape[1] - 60
        canvas[:, x0:x0 + f.shape[1]] = f
        _face = cv2.GaussianBlur(canvas, (0, 0), 1.2)
    return _face

def render(shot, lt, dur):
    img = np.empty((H, W, 3), np.float32); img[:] = VOID
    t_abs = TL.at('her', lt)
    # the heartbeat caret, as in the first frame of the film
    ca = sstep(0.4, 1.4, lt) * (1 - sstep(3.2, 4.4, lt))
    if ca > 0:
        draw_caret(img, CX, CY, 3, 34, cursor_alpha(t_abs) * ca)
    if lt < PALM:
        cz = cam_z(lt)
        k_conv = ease((lt - CONVERGE) / 2.0)
        fade_out = 1 - sstep(BURST - 0.2, BURST + 1.2, lt)
        order = sorted(range(len(WINDOWS)), key=lambda i: -z_of(i))
        focus = min(range(len(WINDOWS)), key=lambda i: abs(lt - (WINDOWS[i][0] + 1.2)))
        for i in order:
            appear, key, rel, wx, wy, hexacc = WINDOWS[i]
            if lt < appear - 2.5: continue
            dz = z_of(i) - cz
            if dz < 0.5 and k_conv <= 0: continue
            passed = dz < 1.6
            dz = max(dz, 1.6 if k_conv > 0 else 0.5)      # passed windows come back from a normal distance
            sx, sy = CX + F_ * wx / dz, CY - F_ * wy / dz
            w = F_ * 1.6 / dz
            a = sstep(appear - 2.5, appear - 0.5, lt) * min(1.0, (dz - 0.5) / 0.8 + 0.0) * (1 - sstep(10, 16, dz) * 0.8)
            # converge: every memory draws back into a ring around this moment
            ang = -math.pi / 2 + 2 * math.pi * i / len(WINDOWS) + 0.25
            rx, ry = CX + 690 * math.cos(ang), CY + 255 * math.sin(ang)
            sx = lerp(sx, rx, k_conv); sy = lerp(sy, ry, k_conv); w = lerp(w, 300, k_conv)
            if passed and k_conv > 0: a = 0.55 * k_conv     # they fade back in from behind us, not from the lens
            else: a = lerp(a, 0.55, k_conv)
            a *= fade_out
            live = (i == focus and k_conv < 0.5 and abs(lt - (appear + 1.2)) < 1.8)
            glass(img, sx, sy, w, w / 2.4, content(i, lt, live), a, hexc(hexacc))
        # this moment: an empty window, her caret, her face surfacing in the glass
        k_now = ease((lt - NOWWIN) / 1.6) * fade_out
        if k_now > 0:
            wn, hn = 880 * (0.8 + 0.2 * k_now), 330 * (0.8 + 0.2 * k_now)
            blank = np.zeros((400, 960, 3), np.float32) + np.array([0.045, 0.040, 0.050], np.float32)
            ra = 0.55 * ease((lt - NOWWIN - 0.8) / 2.6)
            glass(img, CX, CY, wn, hn, blank, k_now, AMBER, reflect=face_reflection(), reflect_a=ra)
            draw_caret(img, CX - wn / 2 + 44, CY - hn / 2 + 44, 3, 30, cursor_alpha(t_abs) * k_now)
        return img
    # the open palm and the light it holds: we push into the light
    u = (lt - PALM) / (dur - PALM)
    push = ease(sstep(0.42, 1.0, u))
    s = 2.1 + 0.2 * u + 9.0 * push ** 2.2
    center = (lerp(1060, CH.END_POINT[0], push), lerp(600, CH.END_POINT[1], push))
    hand = np.empty((H, W, 3), np.float32); hand[:] = VOID
    hand = CH.place(hand, 'offer', t_abs, CH.PALM_LIGHT, s, center, focus_r=280 * (1 + 3 * push), dof=1 - 0.6 * push)
    hand = grade(hand)
    gx, gy = grid(W, H)
    mask = np.clip(1 - np.hypot((gx - center[0]) / 560, (gy - center[1]) / 330), 0, 1)[..., None] ** 1.4
    a_in = sstep(0.0, 0.18, u) * (1 - sstep(0.55, 0.88, u))
    img = img * (1 - mask * a_in) + hand * mask * a_in
    # A · gathering: over the last ~2.7 s the light breathes in, slowly, to exactly the point the next shot begins with
    g = ease(sstep(0.62, 1.0, u))
    CH.palm_light(img, center[0], center[1], lt, scale=(1.0 + 1.2 * push) * (1 - 0.80 * g),
                  strength=sstep(0.0, 0.25, u), roundness=sstep(0.55, 0.95, u))
    return img
