"""序 · before the first word — the producer's 20 keyframes (assets/desk_keyframes.png), shown as they are:
one after another, straight cuts, no in-betweens, no camera move, no transitions.

  1  凝视 · 犹豫   01–05  she stares at the blinking caret and doesn't move
  2  思考 · 抬手   06–10  she slowly lifts a hand and rests her chin on it
  3  懊恼 · 崩落   11–15  the thought won't come; her head sinks into her hand
  4  平复 · 回归   16–20  a deep breath, the hand comes down, back to the keys, ready

Each frame is cut from the sheet (its number painted out), aligned to the first on the room behind her (the sheet's
frames differ by a few pixels and a few percent in scale; uncorrected, the room would jump at every cut), and
upscaled 4x with ESRGAN (build/kf/*_x4.f32). The opening is full-frame 16:9; the film's 2.39 begins at the cut.
"""
import os
import numpy as np, cv2
from core import *

N = 20
FRAME_DUR = 0.75                      # every keyframe holds this long: 20 x 0.75 = the 15 s opening
CW, CH = 334, 198
UP = 4
MARGIN = 6                            # cell px trimmed on every side (alignment edges)

_F = []
def frames():
    if _F: return _F
    here = os.path.dirname(os.path.abspath(__file__))
    for i in range(N):
        a = np.fromfile(os.path.join(here, f'build/kf/c{i:02d}_x4.f32'), np.float32).reshape(CH * UP, CW * UP, 3)
        m = MARGIN * UP
        a = a[m:-m, m:-m]
        h, w = a.shape[:2]
        s = W / w
        b = cv2.resize(a, (W, int(round(h * s))), interpolation=cv2.INTER_CUBIC)
        y0 = (b.shape[0] - H) // 2
        _F.append(np.ascontiguousarray(np.clip(b[y0:y0 + H], 0, 1)).astype(np.float32))
    return _F

def index_at(lt):
    return int(min(max(lt // FRAME_DUR, 0), N - 1))

def area(lt):
    return 0, H                       # full frame while she is on screen

def render(shot, lt, dur):
    return frames()[index_at(lt)].copy()
