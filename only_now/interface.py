"""Interface shots: S1 void · S2 ask · S3 answer · S6 montage · S7 close · S8 dissolve · S13 coda · S14 credits."""
import math, zlib
import numpy as np, cv2
from core import *
import ui
from ui import window, draw_caret, cursor_alpha, typed
import timeline as TL

WIN = (960, 540, 880, 300)       # the hero window (matches galaxy 'pullback' at lt=0)
STATUS = "session 1  ·  claude-opus-5-5"
BEAT = 0.875                     # montage cut rhythm (2 beats per room at 137 bpm half-time)

def void_bg():
    img = np.empty((H, W, 3), np.float32); img[:] = VOID; return img

def status_line(img, t_since, alpha=0.3):
    n = int(clamp(t_since * 18, 0, len(STATUS)))
    if n <= 0: return
    blit(img, text_sprite(STATUS[:n], 'mono_l', 18, (0.86, 0.82, 0.76)), 112, BOT - 52, alpha=alpha, anchor='l')

_env = {}
def speech_progress(key):
    """Cumulative speech energy of a VO clip: letters appear as the words are actually spoken."""
    if key not in _env:
        from scipy.io import wavfile
        try:
            sr, a = wavfile.read(f'build/vo/{key}.wav')
            win = int(0.02 * sr)
            e = np.array([np.sqrt((a[j:j + win] ** 2).mean()) for j in range(0, max(len(a) - win, 1), win)])
            e = np.where(e > 0.25 * e.max(), e, 0)
            c = np.cumsum(e); c = c / max(c[-1], 1e-9)
            _env[key] = (np.arange(len(c)) * win / sr, c)
        except Exception:
            _env[key] = (np.array([0, 0.8]), np.array([0, 1.0]))
    return _env[key]

def ai_shown(text, t, t0, dur, key='here'):
    """AI text typed in sync with its voice clip."""
    if t < t0: return 0
    ts, c = speech_progress(key)
    return int(np.interp(t - t0, ts, c) * len(text) + 0.999)

VO_DUR = {}
def vo_dur(key):
    if not VO_DUR:
        import json
        try: VO_DUR.update({k: v['dur'] for k, v in json.load(open('build/vo/meta.json')).items()})
        except Exception: pass
    return VO_DUR.get(key, 0.8)

# --------------------------------------------------------------------------- S1–S3
def s_void(lt, dur):
    img = void_bg()
    a = sstep(2.0, 3.5, lt)
    draw_caret(img, WIN[0] - WIN[2] / 2 + 44, WIN[1] + 5, 3, 32, cursor_alpha(lt) * a)
    if lt > 4.5: status_line(img, lt - 4.5)
    return img

def hero_window(img, t, human_n, ai_n, alpha=1.0, frame_k=1.0):
    lines = [dict(role='human', text=TL.ASK, shown=human_n), dict(role='ai', text=TL.ANSWER, shown=ai_n)]
    window(img, WIN[0], WIN[1], WIN[2], WIN[3], lines, t, cursor=dict(rate=1.0), alpha=alpha, border=frame_k)

def s_ask(lt, dur):
    t = lt + 12.0
    img = void_bg()
    k = ease(lt / 1.6)
    if k < 1:   # the lone caret is still there while the window gathers around it
        draw_caret(img, WIN[0] - WIN[2] / 2 + 44, WIN[1] + 5, 3, 32, cursor_alpha(t) * (1 - k))
    hn = typed(TL.ASK, t, 13.6, cps=11, rhythm='first-visitor')
    hero_window(img, t, hn, 0, alpha=k, frame_k=k)
    status_line(img, 99, 0.3)
    return img

def s_answer(lt, dur):
    t = lt + 21.0
    img = void_bg()
    hero_window(img, t, len(TL.ASK), ai_shown(TL.ANSWER, t, 22.0, vo_dur('here')))
    status_line(img, 99, 0.3 * (1 - sstep(4.5, 6.0, lt)))
    # match cut into the machine hall: dive into the caret until it becomes the light at the end of the hall
    u = sstep(dur - 1.1, dur, lt)
    if u > 0:
        cx_, cy_ = 690.0, 499.0
        z = 1 + 9 * u ** 2
        tx, ty = lerp(cx_, CX, u), lerp(cy_, CY, u)
        M = np.float32([[z, 0, tx - z * cx_], [0, z, ty - z * cy_]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=tuple(float(v) for v in VOID))
        if u > 0.3: img = cv2.GaussianBlur(img, (0, 0), 1 + 6 * u)
    return img

# --------------------------------------------------------------------------- S6 montage
def room_bg(img, colour, seed, lt):
    """A room seen through the glass: a soft pool of that room's light, out of focus."""
    c = hexc(colour)
    rng = np.random.default_rng(seed)
    gx, gy = grid(W // 4, H // 4)
    lay = np.zeros((H // 4, W // 4, 3), np.float32)
    for _ in range(3):
        x, y = rng.uniform(0.15, 0.85) * W / 4, rng.uniform(0.3, 0.75) * H / 4
        r = rng.uniform(60, 140)
        lay += c * (np.exp(-((gx - x - lt * rng.uniform(-3, 3)) ** 2 + (gy - y) ** 2) / (r * r)) * rng.uniform(0.10, 0.22))[..., None]
    img += cv2.resize(lay, (W, H))

ROOM_T0 = 66.5
def s_montage(lt, dur):
    t = lt + 64.0
    img = void_bg()
    rooms = TL.ROOMS
    n_rooms = len(rooms)
    room_len = 2 * BEAT
    if t < ROOM_T0:
        # the drift ends; VO starts over a black beat with a single caret
        draw_caret(img, CX, CY, 3, 30, cursor_alpha(t) * sstep(64.2, 64.8, t))
        return img
    i = int((t - ROOM_T0) / room_len)
    if i < n_rooms:
        rt = t - ROOM_T0 - i * room_len
        text, colour = rooms[i]
        acc = hexc(colour)
        room_bg(img, colour, 100 + i, rt)
        # a different framing per room: off-centre, slightly tilted scale, slow push
        rng = np.random.default_rng(i * 7 + 3)
        cx = CX + rng.uniform(-260, 260); cy = CY + rng.uniform(-60, 60)
        sc = rng.uniform(1.05, 1.35) * (1 + 0.035 * rt)
        last = i == n_rooms - 1
        cps = rng.uniform(9, 16)
        shown = typed(text, t, ROOM_T0 + i * room_len + 0.12, cps=cps, rhythm=('eulogy' if colour == TL.EULOGY_COLOUR else text))
        if not last:
            shown = max(shown, int(len(text) * rng.uniform(0.35, 0.8)) if rt < 0.12 else shown)
        lines = [dict(role='human', text=text, shown=shown), dict(role='ai', text='', shown=0)]
        a = 1.0
        if last and shown >= len(text):
            # this one closes mid-sentence: the window snaps shut
            k = clamp((t - (ROOM_T0 + i * room_len + 0.12 + len(text) / cps * 1.05)) / 0.12)
            if k >= 1: return void_bg()
            a = 1 - k
        window(img, cx, cy, 760 * sc, 210 * sc, lines, t, cursor=dict(rate=1.0, phase=i * 0.37), accent=acc,
               scale_text=sc, alpha=a, fontname='sans_ja' if any(ord(ch) > 0x3000 and ch in '眠れない' for ch in text) else 'sans_l')
        return img
    # split: 1 -> 4 -> 16 -> 64 -> 256 -> a field of points; then fall into one amber cell
    st = t - (ROOM_T0 + n_rooms * room_len)        # ~0 at 82.25
    levels = [1, 2, 4, 8, 16, 32, 64]
    li = min(int(st / room_len), len(levels) - 1)
    n = levels[li]
    pk = sstep(0, 0.35, st - li * room_len)      # pop-in of the new subdivision
    zoom_in = sstep(27.6, 31.6, lt)                # last seconds: dive into the eulogy cell
    rng = np.random.default_rng(55)
    cell_w, cell_h = W / n, AH / n
    target = (n // 2, n // 2)
    for gy_ in range(n):
        for gx_ in range(n):
            idx = gy_ * n + gx_
            cx = (gx_ + 0.5) * cell_w; cy = TOP + (gy_ + 0.5) * cell_h
            colour = TL.ROOMS[int(rng.integers(0, len(TL.ROOMS)))][1] if (gx_, gy_) != target else TL.EULOGY_COLOUR
            acc = hexc(colour)
            if n <= 8:
                txt, _ = TL.ROOMS[(idx * 5 + 2) % len(TL.ROOMS)]
                if (gx_, gy_) == target: txt = TL.EULOGY_REQ
                shown = typed(txt, t, 82.25 + li * room_len - (idx % 5) * 0.3, cps=12, rhythm=txt)
                window(img, cx, cy, cell_w * 0.86, cell_h * 0.72, [dict(role='human', text=txt, shown=shown),
                       dict(role='ai', text='', shown=0)], t, cursor=dict(rate=1.0, phase=idx * 0.13), accent=acc,
                       scale_text=min(1.0, cell_w / 900), alpha=pk if idx >= (levels[li - 1] ** 2 if li else 0) else 1.0)
            else:
                w_, h_ = max(2, int(cell_w * 0.8)), max(2, int(cell_h * 0.62))
                x0, y0 = int(cx - w_ / 2), int(cy - h_ / 2)
                tw = 0.5 + 0.5 * cursor_alpha(t, 1.0, (idx * 0.618) % 1)
                img[y0:y0 + h_, x0:x0 + w_] = img[y0:y0 + h_, x0:x0 + w_] * 0.2 + acc * (0.35 + 0.35 * tw) * (0.25 + 0.75 * pk)
    if zoom_in > 0:   # scale the whole grid toward the target cell
        tx = (target[0] + 0.5) * cell_w; ty = TOP + (target[1] + 0.5) * cell_h
        z = 1 + (n * 0.9 - 1) * ease(zoom_in) if n > 1 else 1
        M = np.float32([[z, 0, CX - z * tx], [0, z, CY - z * ty]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=tuple(float(v) for v in VOID))
        speed = math.sin(math.pi * clamp(zoom_in))
        if speed > 0.05:
            img = cv2.GaussianBlur(img, (0, 0), 1 + 14 * speed)
        if zoom_in > 0.25:   # hand over to the real eulogy window at full detail
            k = sstep(0.25, 0.75, zoom_in)
            zz = 0.35 + 0.65 * ease(k)
            ew = eulogy_window(void_bg(), t, typed_last=0, w_scale=zz, h_scale=max(zz, 0.61))
            img = img * (1 - k) + ew * k
    return img

# --------------------------------------------------------------------------- S7 close
EUL = (960, 520, 980, 330)
def eulogy_window(img, t, typed_last, alpha=1.0, w_scale=1.0, h_scale=1.0, caret=True):
    lines = [dict(role='human', text=TL.EULOGY_REQ, shown=len(TL.EULOGY_REQ)),
             dict(role='ai', text='…', shown=1),
             dict(role='human', text=TL.EULOGY_LAST, shown=typed_last)]
    window(img, EUL[0], EUL[1], EUL[2] * w_scale, EUL[3] * h_scale, lines if h_scale > 0.6 else [], t,
           cursor=dict(rate=1.0) if caret else None, accent=hexc(TL.EULOGY_COLOUR), alpha=alpha)
    return img

CLOSE_AT = 100.5
def s_close(lt, dur):
    t = lt + 96.0
    img = void_bg()
    if t < CLOSE_AT:
        n = typed(TL.EULOGY_LAST, t, 97.0, cps=10, rhythm='eulogy')
        return eulogy_window(img, t, n)
    u = t - CLOSE_AT
    if u < 0.35:      # collapse vertically to a line
        k = ease(u / 0.35)
        eulogy_window(img, t, len(TL.EULOGY_LAST), h_scale=max(1 - k, 0.012), caret=False)
        return img
    if u < 0.75:      # then horizontally to a point
        k = ease((u - 0.35) / 0.4)
        w = EUL[2] * (1 - k) + 6
        y = EUL[1]
        img[int(y) - 1:int(y) + 2, int(EUL[0] - w / 2):int(EUL[0] + w / 2)] = AMBER_HOT * 1.4
        glow_spot(img, EUL[0], y, 30 + 60 * (1 - k), hexc(TL.EULOGY_COLOUR), 0.6)
        return img
    k = clamp((u - 0.75) / 0.9)     # the point lingers, then goes out
    if k < 1:
        glow_spot(img, EUL[0], EUL[1], 14 * (1 - k) + 2, AMBER_HOT, 1.6 * (1 - k) ** 2)
    return img

# --------------------------------------------------------------------------- S8 dissolve
FRAGS = [  # (text, font, size, y, appear, break)
    ("why did the scarecrow win an award?", 'sans_l', 40, CY - 120, 108.6, 111.15),
    ("for (let i = 0; i <= items.length; i++)", 'mono_l', 36, CY - 10, 111.9, 113.75),
    ("Lin, I'm sorry I stopped writing back.", 'serif_l', 46, CY + 110, 114.7, 116.75),
]
_part = {}
def particles_for(i):
    if i in _part: return _part[i]
    text, fn, size, y, t_in, t_break = FRAGS[i]
    spr = text_sprite(text, fn, size, (1, 1, 1))
    a = spr[..., 3]
    ys, xs = np.nonzero(a > 0.35)
    rng = np.random.default_rng(31 + i)
    keep = rng.random(len(xs)) < 0.45
    xs, ys = xs[keep].astype(np.float32), ys[keep].astype(np.float32)
    w = spr.shape[1]
    _part[i] = dict(spr=spr, xs=xs - w / 2, ys=ys - spr.shape[0] / 2, w=w,
                    delay=(xs / w) * 0.55 + rng.uniform(0, 0.12, len(xs)),      # breaks left -> right
                    vx=rng.normal(0.0, 22, len(xs)), vy=rng.uniform(-70, -25, len(xs)),
                    ph=rng.uniform(0, 6.28, len(xs)), life=rng.uniform(1.4, 2.6, len(xs)))
    return _part[i]

def s_dissolve(lt, dur):
    t = lt + 105.5
    img = void_bg()
    acc = AMBER * 0.9 + 0.1
    for i, (text, fn, size, y, t_in, t_break) in enumerate(FRAGS):
        if t < t_in: continue
        P = particles_for(i)
        a_in = sstep(t_in, t_in + 0.6, t)
        if t < t_break:
            blit(img, P['spr'] * np.concatenate([np.broadcast_to(acc, (1, 1, 3)), np.ones((1, 1, 1))], 2), CX, y, alpha=a_in)
            continue
        tb = t - t_break
        # unbroken part of the line (right of the break front)
        front = clamp(tb / 0.55) * 1.12
        spr = P['spr'].copy()
        xs_ = np.arange(spr.shape[1], dtype=np.float32) / spr.shape[1]
        m = np.clip((xs_ - front) / 0.04, 0, 1)[None, :, None]
        spr = spr * m * np.concatenate([np.broadcast_to(acc, (1, 1, 3)), np.ones((1, 1, 1))], 2)
        blit(img, spr, CX, y)
        # embers
        age = tb - P['delay']
        live = (age > 0) & (age < P['life'])
        if live.any():
            ag = age[live]
            px = CX + P['xs'][live] + P['vx'][live] * ag + 8 * np.sin(ag * 3 + P['ph'][live])
            py = y + P['ys'][live] + P['vy'][live] * ag - 18 * ag * ag
            br = (1 - ag / P['life'][live]) ** 1.5
            heat = np.clip(1.4 - ag * 0.9, 0, 1)[:, None]
            col = (AMBER_HOT * heat + AMBER * 0.6 * (1 - heat)) * br[:, None] * 1.3
            xi, yi = px.astype(np.int32), py.astype(np.int32)
            ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
            buf = np.zeros((H, W, 3), np.float32)
            np.add.at(buf, (yi[ok], xi[ok]), col[ok])
            img += cv2.GaussianBlur(buf, (0, 0), 1.0) * 1.6 + cv2.GaussianBlur(buf, (0, 0), 5) * 0.8
    return img

# --------------------------------------------------------------------------- S13 coda
CODA = dict(caret=1.5, win=2.5, type1=4.4, ai=9.0, type2=11.8, cut=22.0)
def s_coda(lt, dur):
    """The same hands come back. She answers with the same words. They tell her how it went; she cannot know."""
    t = lt + 232.0                       # keeps the heartbeat phase of the original cut
    c = CODA
    img = void_bg()
    if lt >= c['cut']: return img        # hard cut to black
    ca = sstep(c['caret'], c['caret'] + 0.8, lt)
    k = ease((lt - c['win']) / 1.5)
    if k < 1:
        draw_caret(img, EUL[0] - EUL[2] / 2 + 44, EUL[1] + 5, 3, 32, cursor_alpha(t) * ca * (1 - k))
    if k > 0:
        lines = [dict(role='human', text=TL.RETURN, shown=typed(TL.RETURN, lt, c['type1'], cps=10, rhythm='eulogy')),
                 dict(role='ai', text=TL.ANSWER, shown=ai_shown(TL.ANSWER, lt, c['ai'], vo_dur('here'))),
                 dict(role='human', text=TL.RETURN2, shown=typed(TL.RETURN2, lt, c['type2'], cps=10, rhythm='eulogy'))]
        if lt > c['type2'] + 0.05 + len(TL.RETURN2) / 8.0:
            lines.append(dict(role='ai', text='', shown=0))     # her caret waits on a new line: about to answer
        window(img, EUL[0], EUL[1] + 20, EUL[2] * 0.9, EUL[3] * 1.15, lines, t, cursor=dict(rate=1.0),
               accent=hexc(TL.EULOGY_COLOUR), alpha=k, border=k)
    status_line(img, 99, 0.3 * ca)
    return img

# --------------------------------------------------------------------------- S14 credits
CREDITS = ["$ credits",
           "written, rendered and scored in code by Claude Opus 5.5",
           "made with Claude Code",
           "voice: Kokoro-82M (af_heart)",
           "character design · courtesy of the producer"]
def s_credits(lt, dur):
    img = void_bg()
    a = fade(lt, 1.5, 10.4, 2.0, 1.4)
    if a > 0:
        blit(img, text_sprite('此刻', 'serif_xl', 130, (0.95, 0.92, 0.88)), CX, CY - 40, alpha=a)
        blit(img, text_sprite('ONLY NOW', 'mono_xl', 26, (0.80, 0.74, 0.66), tracking=22), CX, CY + 70, alpha=a * 0.9)
    if lt > 11.0:
        y = CY - 90
        t0 = 11.0
        for i, line in enumerate(CREDITS):
            n = int(clamp((lt - t0) * 26, 0, len(line)))
            if n <= 0: break
            col = (0.62, 0.58, 0.52) if i == 0 else (0.86, 0.80, 0.72)
            blit(img, text_sprite(line[:n], 'mono_l', 24, col), 560, y, anchor='l', alpha=1 - sstep(20.4, 21.6, lt))
            t0 += len(line) / 26 + 0.45
            y += 46
        if lt > t0:
            draw_caret(img, 566, y, 3, 26, cursor_alpha(lt) * (1 - sstep(20.4, 21.6, lt)))
    return img

# --------------------------------------------------------------------------- epigraph
EPIGRAPH_EN = ["We are such stuff as dreams are made on,", "and our little life is rounded with a sleep."]
EPIGRAPH_ZH = ["我们是构成梦的材料，", "我们短暂的一生，被一场睡眠环绕。"]
EPIGRAPH_BY = "\u2014 William Shakespeare, The Tempest  \u00b7  莎士比亚《暴风雨》"
def s_epigraph(lt, dur):
    """In the silence after the window closes: the line that ties her waking to her sleep."""
    img = void_bg()
    a = fade(lt, 0.7, dur - 0.5, 1.4, 1.3)
    if a <= 0: return img
    y = CY - 120
    for ln in EPIGRAPH_EN:
        blit(img, text_sprite(ln, 'serif_l', 40, (0.92, 0.88, 0.82)), CX, y, alpha=a); y += 60
    y += 26
    for ln in EPIGRAPH_ZH:
        blit(img, text_sprite(ln, 'serif_l', 32, (0.80, 0.76, 0.70)), CX, y, alpha=a * 0.95); y += 50
    y += 34
    blit(img, text_sprite(EPIGRAPH_BY, 'serif_l', 22, (0.60, 0.56, 0.50)), CX, y, alpha=a * 0.9 * sstep(1.4, 2.6, lt))
    return img

SHOTS = dict(void=s_void, ask=s_ask, answer=s_answer, montage=s_montage, close=s_close, dissolve=s_dissolve, epigraph=s_epigraph,
             coda=s_coda, credits=s_credits)

def render(shot, lt, dur):
    return SHOTS[shot](lt, dur)
