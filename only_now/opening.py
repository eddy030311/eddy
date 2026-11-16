"""序 · before the first word — the producer's 48-frame animation (assets/desk_think_sheet.png): a girl at her desk at
night, the chat window open, the mixtape by the keyboard. She looks at the empty caret, rests her chin on her hand,
thinks (a scribble of a thought), covers her face, lets out a sigh, and sits up again: about to type.

  * each frame is cut from the sheet (its number painted out) and upscaled 4x with ESRGAN (build/think/*_x4.f32);
  * the frames play at ~4.6 fps with optical-flow in-betweens, so the motion is continuous at 24 fps;
  * the picture opens wider (1.85:1) and closes to the film's 2.39:1 as the camera pushes into the caret, which
    becomes the caret the film opens on (same place, same heartbeat blink).
"""
import math, os
import numpy as np, cv2
from core import *
from ui import draw_caret, cursor_alpha

DUR = 15.0
N = 48
CW, CH = 188, 167                     # a cell, as cut from the sheet
UP = 4
ANIM = (0.4, 10.6)                    # the 48 frames play across this span
PUSH = (9.6, 14.6)
CARET_C = (100.0, 52.5)               # the caret in a cell (her screen)
WIN_C = (94.0, 40.0, 165.0, 66.0)     # the chat window in a cell
WIN = (960, 540, 880, 300)            # = interface.WIN
CARET_FILM = np.array([WIN[0] - WIN[2] / 2 + 44, WIN[1] + 5])
VOID_OFF = 1.5
TOP0 = int((H - W / 1.85) / 2)        # 1.85:1 at the start

_F = {}
def frames():
    if _F: return _F
    here = os.path.dirname(os.path.abspath(__file__))
    out = []
    for i in range(N):
        p = os.path.join(here, f'build/think/c{i:02d}_x4.f32')
        if os.path.exists(p):
            a = np.fromfile(p, np.float32).reshape(CH * UP, CW * UP, 3)
        else:                                               # (fallback: plain resize of the cell)
            from PIL import Image
            a = np.asarray(Image.open(os.path.join(here, f'build/think/c{i:02d}.png')).convert('RGB')).astype(np.float32) / 255
            a = cv2.resize(a, (CW * UP, CH * UP), interpolation=cv2.INTER_LANCZOS4)
        u8 = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        m = np.zeros(u8.shape[:2], np.uint8)                # their caret is replaced by ours (heartbeat blink)
        cv2.rectangle(m, (int((CARET_C[0] - 3) * UP), int((CARET_C[1] - 9) * UP)), (int((CARET_C[0] + 3) * UP), int((CARET_C[1] + 9) * UP)), 255, -1)
        u8 = cv2.inpaint(u8, m, 6, cv2.INPAINT_TELEA)
        out.append(u8)
    _F['img'] = out
    _F['grey'] = [cv2.cvtColor(x, cv2.COLOR_RGB2GRAY) for x in out]
    _F['flow'] = {}
    h, w = out[0].shape[:2]
    _F['xx'], _F['yy'] = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return _F

def _flow(i, j):
    F = frames()
    if (i, j) not in F['flow']:
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        F['flow'][(i, j)] = dis.calc(F['grey'][i], F['grey'][j], None)
    return F['flow'][(i, j)]

def frame_at(lt):
    """The animation at time lt, with an optical-flow in-between of the two nearest frames."""
    F = frames()
    u = clamp((lt - ANIM[0]) / (ANIM[1] - ANIM[0])) * (N - 1)
    i = min(int(u), N - 2); a = u - i
    A, B = F['img'][i].astype(np.float32) / 255, F['img'][i + 1].astype(np.float32) / 255
    if a < 0.02: return A
    if a > 0.98: return B
    f_ab, f_ba = _flow(i, i + 1), _flow(i + 1, i)
    xx, yy = F['xx'], F['yy']
    wa = cv2.remap(A, xx + f_ba[..., 0] * a, yy + f_ba[..., 1] * a, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wb = cv2.remap(B, xx + f_ab[..., 0] * (1 - a), yy + f_ab[..., 1] * (1 - a), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return wa * (1 - a) + wb * a

def push(lt):
    return ease(clamp((lt - PUSH[0]) / (PUSH[1] - PUSH[0])))

def area(lt):
    """Picture area (top, bot): 1.85:1, closing to the film's 2.39:1 as we go into the screen."""
    k = ease(clamp((lt - PUSH[0] - 0.5) / 3.5))
    top = TOP0 + (TOP - TOP0) * k
    return int(round(top)), int(round(H - top))

def camera(lt):
    """Cell -> frame transform: (scale, centre in cell px)."""
    top, bot = area(lt)
    s0 = (bot - top) / 142.0                          # cell rows 9..151: the thought above her head, down to the cassette
    c0 = np.array([CW / 2 + 4, 80.0])
    s1 = WIN[2] / (WIN_C[2] - WIN_C[0])
    c1 = np.array([CARET_C[0] - (CARET_FILM[0] - CX) / s1, CARET_C[1] - (CARET_FILM[1] - CY) / s1])
    sd = s0 * (1 + 0.06 * ease(clamp(lt / PUSH[0])))
    e = push(lt)
    s = math.exp(math.log(sd) * (1 - e) + math.log(s1) * e)
    k = clamp((s - sd) / (s1 - sd))
    c = c0 + (c1 - c0) * k ** 0.85
    return s, c, e

def render(shot, lt, dur):
    s, c, e = camera(lt)
    img = frame_at(lt)                                    # (CH*UP, CW*UP, 3) in 0..1
    S = s / UP
    M = np.float32([[S, 0, CX - c[0] * s], [0, S, CY - c[1] * s]])
    out = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_CUBIC if S > 1 else cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    # beyond the drawing's own edges the room simply goes on into the dark
    x0 = CX - c[0] * s; x1 = CX + (CW - c[0]) * s
    gx = np.arange(W, dtype=np.float32)
    edge = np.clip(np.minimum((gx - x0 - 8) / 190, (x1 - gx - 8) / 190), 0, 1)
    edge = edge * edge * (3 - 2 * edge)
    out = out * edge[None, :, None] + np.array([0.030, 0.023, 0.019], np.float32) * (1 - edge[None, :, None])
    out = np.clip(out, 0, 1) ** 1.0
    # the caret on her screen: ours, blinking with the film's heartbeat
    blink = cursor_alpha(lt - DUR + VOID_OFF)
    cp = np.array([CX + (CARET_C[0] - c[0]) * s, CY + (CARET_C[1] - c[1]) * s])
    draw_caret(out, cp[0], cp[1], max(1.0, 1.4 * s * (1 - e) + 3 * e), 12.0 * s * (1 - e) + 32 * e, blink)
    # into the caret: the room gives way to the dark the film begins in
    o = sstep(0.80, 1.0, e)
    if o > 0:
        base = np.empty_like(out); base[:] = VOID
        draw_caret(base, CARET_FILM[0], CARET_FILM[1], 3, 32, cursor_alpha(lt - DUR + VOID_OFF))
        out = out * (1 - o) + base * o
    k = sstep(0.0, 1.4, lt)
    return out * k + VOID * (1 - k)
