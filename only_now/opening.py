"""序 · before the first word — built on the producer's frame (assets/desk_plate.png): a girl at her desk at night,
typing to Claude, the mixtape by the keyboard.

  * the hands move along the producer's 30-frame typing sheet (assets/desk_typing_sheet.png): each hand's track was
    measured frame by frame (build/typing_track.npy) and drives a soft local warp of the hands in the full-res frame;
  * she breathes; the caret on her screen blinks with the film's heartbeat; she types a line, looks at it, deletes it;
  * the camera drifts in, then pushes into the caret until it is the caret the film opens on (same place, same blink).
"""
import math, os
import numpy as np, cv2
from PIL import Image
from core import *
from ui import draw_caret, cursor_alpha

DUR = 15.0
PLATE_ROWS = (123, 819)                 # the picture inside the plate's own letterbox
WIN_P = (737.0, 309.0, 1026.0, 414.0)   # the chat window in the plate
CARET_P = (750.5, 361.0)
HANDS_P = [((742.0, 655.0), 52, 30, -20), ((842.0, 655.0), 56, 30, -14)]
CELL_TO_PLATE = 2.3                     # sheet-cell px -> plate px (keyboard width ratio)
WIN = (960, 540, 880, 300)              # = interface.WIN
CARET_FILM = np.array([WIN[0] - WIN[2] / 2 + 44, WIN[1] + 5])
VOID_OFF = 1.5                          # the 'void' shot starts 1.5 s into its own clock
DRAFT = "hey. can I ask you something"
TYPE_T = (2.4, 4.9)                     # she types a first line...
DEL_T = (6.2, 7.0)                      # ...looks at it, and deletes it

_C = {}
def plate():
    if _C: return _C
    here = os.path.dirname(os.path.abspath(__file__))
    im = np.asarray(Image.open(os.path.join(here, 'assets/desk_plate.png')).convert('RGB'))
    im = np.ascontiguousarray(im[PLATE_ROWS[0]:PLATE_ROWS[1]])
    # the plate's own caret is replaced by ours (it must blink, and move as she types)
    m = np.zeros(im.shape[:2], np.uint8)
    cx, cy = CARET_P[0], CARET_P[1] - PLATE_ROWS[0]
    cv2.rectangle(m, (int(cx - 4), int(cy - 10)), (int(cx + 5), int(cy + 10)), 255, -1)
    im = cv2.inpaint(im, m, 5, cv2.INPAINT_TELEA)
    P = im.astype(np.float32) / 255.0
    h, w = P.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    hands = []
    for (c, rx, ry, ang) in HANDS_P:
        mk = np.zeros((h, w), np.float32)
        cv2.ellipse(mk, (int(c[0]), int(c[1] - PLATE_ROWS[0])), (rx, ry), ang, 0, 360, 1.0, -1, cv2.LINE_AA)
        hands.append(cv2.GaussianBlur(mk, (0, 0), 9))
    body = np.zeros((h, w), np.float32)                      # her (for breathing): head, hair, shoulders, back
    cv2.fillPoly(body, [np.int32([(300, 696), (330, 330), (460, 210), (600, 210), (700, 330), (720, 520), (640, 696)])], 1.0)
    body = cv2.GaussianBlur(body, (0, 0), 14) * np.clip((560 - yy) / 400, 0, 1)
    tp = os.path.join(here, 'build/typing_track.npy')
    tr = np.load(tp) if os.path.exists(tp) else np.zeros((30, 4))
    tr = tr - tr.mean(0)
    _C.update(P=P, hands=hands, body=body, xx=xx, yy=yy, track=tr * CELL_TO_PLATE)
    return _C

def track_at(f):
    tr = plate()['track']; n = len(tr)
    i = int(math.floor(f)) % n; j = (i + 1) % n; u = f - math.floor(f)
    return tr[i] * (1 - u) + tr[j] * u

def typing_state(lt):
    """(typing amount, backspace amount, draft characters on screen)."""
    n = len(DRAFT)
    ramp = sstep(TYPE_T[0] - 0.25, TYPE_T[0] + 0.1, lt) * (1 - sstep(TYPE_T[1] - 0.1, TYPE_T[1] + 0.3, lt))
    dele = sstep(DEL_T[0] - 0.15, DEL_T[0] + 0.05, lt) * (1 - sstep(DEL_T[1] - 0.05, DEL_T[1] + 0.25, lt))
    if lt < TYPE_T[0]: shown = 0
    elif lt < TYPE_T[1]: shown = int(n * (lt - TYPE_T[0]) / (TYPE_T[1] - TYPE_T[0]))
    elif lt < DEL_T[0]: shown = n
    elif lt < DEL_T[1]: shown = int(n * (1 - (lt - DEL_T[0]) / (DEL_T[1] - DEL_T[0])))
    else: shown = 0
    return ramp, dele, shown

def animated(lt):
    """The plate with her hands moving along the sheet's track, and her breathing."""
    C = plate()
    P = C['P']
    ramp, dele, _ = typing_state(lt)
    t4 = track_at(lt * 24.0)
    dx = np.zeros(P.shape[:2], np.float32); dy = np.zeros(P.shape[:2], np.float32)
    # both hands type while she writes; the right one alone taps backspace
    for k, (hm, amp) in enumerate(zip(C['hands'], (ramp, max(ramp, 0.55 * dele)))):
        if amp > 1e-3:
            dx += hm * float(t4[2 * k]) * amp
            dy += hm * float(t4[2 * k + 1]) * amp
    br = 1.3 * math.sin(lt * 2 * math.pi / 4.2)                # a slow breath lifts her shoulders and head
    dy -= C['body'] * br
    x0, x1, y0, y1 = 280, 960, 180, 696                        # only where something moves
    out = P.copy()
    out[y0:y1, x0:x1] = cv2.remap(P, C['xx'][y0:y1, x0:x1] - dx[y0:y1, x0:x1], C['yy'][y0:y1, x0:x1] - dy[y0:y1, x0:x1],
                                  cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    return out

def camera(lt):
    """Plate -> active-area transform: (scale, centre in plate px, push 0..1)."""
    h = PLATE_ROWS[1] - PLATE_ROWS[0]
    s0 = AH / h
    c0 = np.array([1672 / 2.0, h / 2.0])
    wc = np.array([(WIN_P[0] + WIN_P[2]) / 2, (WIN_P[1] + WIN_P[3]) / 2 - PLATE_ROWS[0]])
    s1 = WIN[2] / (WIN_P[2] - WIN_P[0])
    sd = s0 * (1 + 0.05 * ease(clamp(lt / 8.6)))
    e = ease(clamp((lt - 8.6) / (DUR - 8.6 - 0.4)))
    s = math.exp(math.log(sd) * (1 - e) + math.log(s1) * e)
    k = clamp((s - sd) / (s1 - sd))
    c = c0 + (wc - c0) * k ** 0.8
    half = np.array([W / 2 / s, AH / 2 / s])
    c = np.clip(c, half, np.array([1672.0, h]) - half)
    return s, c, e

def to_film(p, s, c):
    return np.array([CX + (p[0] - c[0]) * s, CY + (p[1] - PLATE_ROWS[0] - c[1]) * s])

def render(shot, lt, dur):
    s, c, e = camera(lt)
    P = animated(lt)
    M = np.float32([[s, 0, W / 2 - c[0] * s], [0, s, AH / 2 - c[1] * s]])
    act = cv2.warpAffine(P, M, (W, AH), flags=cv2.INTER_CUBIC if s > 1.3 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    img = np.empty((H, W, 3), np.float32); img[:] = VOID
    img[TOP:BOT] = np.clip(act, 0, 1)
    # what she types, and the caret (ours: it blinks with the film's heartbeat)
    _, _, shown = typing_state(lt)
    blink = cursor_alpha(lt - DUR + VOID_OFF)
    cp = to_film(CARET_P, s, c)
    fs = max(6, int(10.5 * s))
    x_end = cp[0]
    if shown > 0:
        spr = text_sprite(DRAFT[:shown], 'sans_l', fs, (0.93, 0.86, 0.78))
        blit(img, spr * 0.85, cp[0] + 2, cp[1], anchor='l')
        x_end = cp[0] + spr.shape[1] - 2
        blink = 1.0                                                  # the caret stays lit while typing
    draw_caret(img, x_end + 2, cp[1], max(1, 0.9 * s), 15 * s, blink)
    # into the caret: the room gives way to the dark the film begins in, the caret exactly where it will be
    out = sstep(0.80, 1.0, e)
    if out > 0:
        base = np.empty_like(img); base[:] = VOID
        draw_caret(base, CARET_FILM[0], CARET_FILM[1], 3, 32, cursor_alpha(lt - DUR + VOID_OFF))
        img = img * (1 - out) + base * out
    k = sstep(0.0, 1.4, lt)
    return img * k + VOID * (1 - k)
