"""The conversation window: the film's central object (cursor = heartbeat)."""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw
from core import *

def cursor_alpha(t, rate=1.0, phase=0.0):
    """Heartbeat blink: lit for half a beat, then a quick decay (not a hard square wave)."""
    u = ((t + phase) * rate) % 1.0
    return 1.0 if u < 0.5 else 0.12 + 0.88 * math.exp(-(u - 0.5) * 9)

def beat_times(t0, t1, rate=1.0, phase=0.0):
    """Times at which the cursor lights (for the tick sound)."""
    k0 = math.ceil((t0 + phase) * rate); out = []
    k = k0
    while (k / rate - phase) < t1:
        out.append(k / rate - phase); k += 1
    return out

def rounded_rect_mask(w, h, r, ss=3):
    m = Image.new('L', (w * ss, h * ss), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss, fill=255)
    return np.asarray(m.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255

def window(img, cx, cy, w, h, lines, t, title='', footer='', alpha=1.0, accent=AMBER, glass=0.55,
           cursor=None, scale_text=1.0, border=1.0, fontname='sans_l'):
    """Draw a conversation window centered at (cx,cy).

    lines: list of dicts {role:'human'|'ai', text:str, shown:int(chars typed)}.
    cursor: None or dict(role, t_phase) — a blinking caret after the last shown text (or alone).
    """
    w, h = int(w), int(h)
    if alpha <= 0.002 or w < 8 or h < 6: return
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    r = max(4, int(min(w, h) * 0.04))
    m = rounded_rect_mask(w, h, r)
    # outer glow
    pad = 40
    gl = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
    gl[pad:pad + h, pad:pad + w] = m
    gl = cv2.GaussianBlur(gl, (0, 0), 16) * 0.35 * border
    _add_layer(img, x0 - pad, y0 - pad, accent[None, None, :] * gl[..., None] * alpha)
    # glass body
    body = np.zeros((h, w, 4), np.float32)
    k = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    col = np.array([0.055, 0.050, 0.065]) * (1 - k) + np.array([0.030, 0.028, 0.040]) * k
    body[..., :3] = col * m[..., None]; body[..., 3] = m * glass
    edge = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1) * 0.55 * border
    body[..., :3] += accent * edge[..., None]; body[..., 3] = np.maximum(body[..., 3], edge)
    blit(img, body, x0, y0, alpha=alpha, anchor='tl')
    s = scale_text
    if title:
        fs = max(8, int(17 * s))
        spr = text_sprite(title, 'mono_l', fs, tuple(accent * 0.8 + 0.2), tracking=int(4 * s))
        blit(img, spr, cx, y0 + 24 * s, alpha=alpha * 0.85)
        yl = int(y0 + 46 * s)
        if 0 <= yl < img.shape[0]:
            img[yl, x0 + 16:x0 + w - 16] += accent * 0.12 * alpha
    # messages
    fs = max(8, int(30 * s))
    y = y0 + (78 if title else 40) * s
    caret_xy = None
    last_role = None
    caret_idx, last_human_idx = -1, -1
    for li_, ln in enumerate(lines):
        txt = ln['text'][:ln.get('shown', len(ln['text']))]
        human = ln['role'] == 'human'
        colr = (0.95, 0.92, 0.88) if human else tuple(np.clip(accent * 1.05 + 0.08, 0, 1))
        if txt:
            spr = text_sprite(txt, fontname, fs, colr)
            if human:
                bx = x0 + w - 36 * s - spr.shape[1] + 8
                bub_w, bub_h = spr.shape[1] + int(10 * s), int(fs * 1.75)
                bubble = rounded_rect_mask(max(bub_w, 8), max(bub_h, 8), int(bub_h * 0.45))
                bl = np.zeros(bubble.shape + (4,), np.float32)
                bl[..., :3] = np.array([0.16, 0.15, 0.17]) * bubble[..., None]; bl[..., 3] = bubble * 0.9
                blit(img, bl, x0 + w - 28 * s - bub_w, y - bub_h / 2 + fs * 0.1, alpha=alpha, anchor='tl')
                blit(img, spr, bx - 4 * s, y, alpha=alpha, anchor='l')
                last_human_idx = li_
            else:
                blit(img, spr, x0 + 34 * s, y, alpha=alpha, anchor='l')
                caret_xy = (x0 + 34 * s + spr.shape[1] - 6, y); caret_idx = li_
        else:
            if not human and caret_idx < last_human_idx or (not human and caret_xy is None):
                caret_xy = (x0 + 44 * s, y); caret_idx = li_
        if txt: last_role = 'human' if human else 'ai'
        y += fs * 2.3
    if last_role == 'human' and caret_idx < last_human_idx:
        caret_xy = (x0 + 44 * s, y)      # Claude's caret waits on a new line below the human's message
    if cursor is not None:
        ph = cursor.get('phase', 0.0)
        ca = cursor_alpha(t, cursor.get('rate', 1.0), ph) * alpha * cursor.get('alpha', 1.0)
        if caret_xy is None:
            caret_xy = (x0 + 44 * s, y0 + (78 if title else 40) * s)
        cxp, cyp = caret_xy
        cw, chh = max(2, int(3 * s)), int(fs * 1.05)
        draw_caret(img, cxp + 6 * s, cyp, cw, chh, ca, accent)
    if footer:
        spr = text_sprite(footer, 'mono_l', max(8, int(17 * s)), tuple(accent * 0.55 + 0.15))
        blit(img, spr, x0 + 26 * s, y0 + h - 26 * s, alpha=alpha * 0.9, anchor='l')

def draw_caret(img, x, y, w, h, a, accent=AMBER, glow=True):
    if a <= 0.01: return
    x0, y0 = int(x - w / 2), int(y - h / 2)
    x1, y1 = x0 + max(1, int(w)), y0 + max(1, int(h))
    if x1 <= 0 or y1 <= 0 or x0 >= img.shape[1] or y0 >= img.shape[0]: return
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, img.shape[1]), min(y1, img.shape[0])
    img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - a) + (accent * 0.4 + AMBER_HOT * 0.75) * a
    if glow:
        glow_spot(img, x, y, max(h * 0.9, 6), accent, 0.35 * a)

def _add_layer(img, x, y, layer):
    h, w = layer.shape[:2]
    x, y = int(x), int(y)
    x0, y0 = max(x, 0), max(y, 0); x1, y1 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    if x1 <= x0 or y1 <= y0: return
    img[y0:y1, x0:x1] += layer[y0 - y:y1 - y, x0 - x:x1 - x]

import zlib
def key_times(text, t0, cps=14.0, rhythm=None):
    """Deterministic human typing: per-person rhythm seed (same person = same rhythm)."""
    seed = zlib.crc32((rhythm or text).encode('utf-8'))
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.55, 1.6, 64)            # this person's personal key-gap pattern
    gaps = np.array([base[i % 64] for i in range(len(text))]) / cps
    for i, ch in enumerate(text):
        if ch == ' ': gaps[i] *= 1.4
        if ch in ',.?!': gaps[i] *= 2.2
    return list(t0 + np.cumsum(gaps))

def typed(text, t, t0, cps=14.0, rhythm=None):
    """Number of characters of `text` visible at time t when typing starts at t0."""
    if t < t0: return 0
    return int(np.searchsorted(np.array(key_times(text, t0, cps, rhythm)), t, side='right'))
