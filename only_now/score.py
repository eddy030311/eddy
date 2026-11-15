"""Original score, sound design and VO mix for 《此刻 · ONLY NOW》 -> build/mix.wav (48 kHz stereo)."""
import json, math
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt, resample_poly
from scipy.io import wavfile
import timeline as TL
import ui
import interface as IF

SR = 48000
N = int(SR * TL.DURATION)
rng = np.random.default_rng(2026)
BEAT = IF.BEAT                      # 0.875 s
BAR = 4 * BEAT

class Bus:
    def __init__(self): self.L = np.zeros(N); self.R = np.zeros(N)
    def add(self, sig, t0, gain=1.0, pan=0.0, new=False):
        if not new: t0 = TL.remap(t0)        # cues written on the original cut move with their shot
        i = int(round(t0 * SR))
        if i >= N or len(sig) == 0: return
        if i < 0: sig = sig[-i:]; i = 0
        sig = sig[:N - i] * gain
        if sig.ndim == 2:
            self.L[i:i + len(sig)] += sig[:, 0]; self.R[i:i + len(sig)] += sig[:, 1]
        else:
            self.L[i:i + len(sig)] += sig * math.sqrt(0.5 * (1 - pan)); self.R[i:i + len(sig)] += sig * math.sqrt(0.5 * (1 + pan))
    def st(self): return np.stack([self.L, self.R], 1)

music, sfx, vo, amb = Bus(), Bus(), Bus(), Bus()
def mtof(m): return 440.0 * 2 ** ((m - 69) / 12)
NOTE = {n: i for i, n in enumerate(['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B'])}
def m(name):  # 'A4' -> midi
    return 12 * (int(name[-1]) + 1) + NOTE[name[:-1]]
def T(n): return np.arange(int(SR * n)) / SR
def lp(x, fc, order=2): return sosfilt(butter(order, fc, 'low', fs=SR, output='sos'), x)
def hp(x, fc, order=2): return sosfilt(butter(order, fc, 'high', fs=SR, output='sos'), x)
def bp(x, lo, hi, order=2): return sosfilt(butter(order, [lo, hi], 'band', fs=SR, output='sos'), x)

# ------------------------------------------------------------------ instruments
def piano(n, dur=4.0, vel=0.5):
    f = mtof(n); t = T(dur); s = np.zeros_like(t)
    for k in range(1, 10):
        fk = f * k * math.sqrt(1 + 0.00035 * k * k)
        dec = (0.6 + 0.45 * k) * (f / 330) ** 0.35
        s += (np.sin(2 * np.pi * fk * t + rng.uniform(0, 6)) + 0.6 * np.sin(2 * np.pi * fk * 1.0012 * t)) / k ** 1.5 * np.exp(-t * dec)
    hammer = hp(rng.standard_normal(len(t)), 1800) * np.exp(-t * 90) * 0.04
    env = np.minimum(t / 0.003, 1) * np.minimum(1, (dur - t) / 0.4)
    s = lp(s, 3000 + 2500 * vel) + hammer
    return s * env * vel * 0.12

def strings(notes, dur, vel=0.5, att=1.5, rel=2.0, bright=2200):
    t = T(dur); s = np.zeros_like(t)
    for n in notes:
        f = mtof(n)
        for d in np.linspace(-0.12, 0.12, 5):
            ph = 2 * np.pi * f * 2 ** (d / 12) * t + 0.004 * np.sin(2 * np.pi * 5.2 * t + rng.uniform(0, 6)) * f / 50 + rng.uniform(0, 6)
            s += ((ph / np.pi) % 2 - 1)          # saw
    s = lp(lp(s, bright), bright * 1.3) / (5 * len(notes))
    env = np.minimum(t / att, 1) ** 1.5 * np.minimum(1, np.maximum(dur - t, 0) / rel)
    return s * env * vel * 0.5

def shimmer(notes, dur, vel=0.4, att=2.0, rel=3.0):
    t = T(dur); s = np.zeros_like(t)
    for n in notes:
        for k, a in ((1, 1.0), (2, 0.35), (3, 0.15)):
            s += a * np.sin(2 * np.pi * mtof(n) * k * t + rng.uniform(0, 6)) * (0.7 + 0.3 * np.sin(2 * np.pi * 0.13 * k * t + n))
    env = np.minimum(t / att, 1) * np.minimum(1, np.maximum(dur - t, 0) / rel)
    return s * env * vel * 0.05

def braam(dur=6.0, root=m('D1'), vel=1.0, major=False):
    t = T(dur); s = np.zeros_like(t)
    for n in (root, root + 12, root + 19, root + 24, root + (28 if major else 27)):
        for d in (-0.1, 0.0, 0.1):
            ph = 2 * np.pi * mtof(n) * 2 ** (d / 12) * t
            s += (ph / np.pi) % 2 - 1
    s = np.tanh(s * 0.6)
    # filter sweep: open fast, close slow (block-wise)
    out = np.zeros_like(s); blk = 1024
    for i in range(0, len(s), blk):
        u = i / SR
        fc = 180 + 2200 * math.exp(-u * 1.1) * min(u / 0.08, 1)
        out[i:i + blk] = lp(s[max(0, i - 4096):i + blk], fc)[-len(out[i:i + blk]):]
    sub = np.sin(2 * np.pi * (52 * t - 6 * t * t / 2).clip(None) ) * np.exp(-t * 0.5)
    env = np.minimum(t / 0.02, 1) * np.exp(-t * 0.45)
    return (out * 0.5 + sub * 0.7) * env * vel * 0.5

def boom(vel=1.0):
    t = T(3.0)
    f = 60 * np.exp(-t * 2.5) + 32
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.6)
    s += lp(rng.standard_normal(len(t)), 400) * np.exp(-t * 12) * 0.5
    return s * vel * 0.8

def tick(vel=1.0):
    t = T(0.03)
    s = np.sin(2 * np.pi * 2700 * t) * np.exp(-t * 900) + hp(rng.standard_normal(len(t)), 3000) * np.exp(-t * 1500) * 0.4
    return s * vel * 0.25

def key(vel=1.0, space=False):
    t = T(0.09)
    click = bp(rng.standard_normal(len(t)), 1500, 7000) * np.exp(-t * rng.uniform(380, 520))
    thock = np.sin(2 * np.pi * (rng.uniform(150, 210) * (0.7 if space else 1)) * t) * np.exp(-t * 70)
    s = click * 0.5 + thock * (0.55 if space else 0.35)
    s2 = np.zeros_like(s); d = int(rng.uniform(0.018, 0.03) * SR)    # key release
    s2[d:] = s[:-d] * 0.25
    return (s + s2) * vel * 0.3

def clock_tick(vel=1.0, tock=False):
    t = T(0.12)
    f = 1150 if not tock else 900
    s = np.sin(2 * np.pi * f * t) * np.exp(-t * 60) + bp(rng.standard_normal(len(t)), 1500, 5000) * np.exp(-t * 300) * 0.3
    return s * vel * 0.12

def crackle(dur, density=120, vel=1.0):
    n = int(SR * dur); s = np.zeros(n)
    k = int(density * dur)
    pos = rng.integers(0, n - 200, k)
    for p in pos:
        L = rng.integers(20, 160); s[p:p + L] += rng.standard_normal(L) * np.exp(-np.arange(L) / (L / 4)) * rng.uniform(0.2, 1)
    env = np.minimum(np.arange(n) / (0.05 * SR), 1) * np.exp(-np.arange(n) / SR * 1.2)
    return hp(s, 1500) * env * vel * 0.25

def noise_bed(dur, lo, hi, vel):
    s = bp(rng.standard_normal(int(SR * dur)), lo, hi)
    s /= np.abs(s).max() + 1e-9
    t = T(dur); env = np.minimum(1, np.minimum(t / 1.0, (dur - t) / 1.0)).clip(0)
    return s * env * vel

def whoosh(dur=1.2, up=True, vel=0.3):
    n = int(SR * dur); x = rng.standard_normal(n); out = np.zeros(n); blk = 1024
    for i in range(0, n, blk):
        u = i / n; fc = 300 + 5000 * (u if up else 1 - u) ** 1.4
        out[i:i + blk] = bp(x[max(0, i - 4096):i + blk], fc * 0.6, min(fc * 1.7, SR / 2 - 100))[-len(out[i:i + blk]):]
    return out * np.sin(np.pi * np.arange(n) / n) ** 1.5 * vel

def chirp(vel=0.2):
    t = T(0.09); f0 = rng.uniform(3200, 4800)
    f = f0 * (1 + 0.35 * np.sin(np.pi * t / 0.09))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * t / 0.09) ** 2 * vel * 0.1

# ------------------------------------------------------------------ harmony
Dm = [m('D3'), m('A3'), m('F4')]; Bb = [m('Bb2'), m('F3'), m('D4')]; F = [m('F3'), m('C4'), m('A4')]; C = [m('C3'), m('G3'), m('E4')]
Gm = [m('G2'), m('D3'), m('Bb3')]; A_ = [m('A2'), m('E3'), m('C#4')]
PROG = [Dm, Bb, F, C]
MOTIF = [(m('F4'), 0.0, 0.5), (m('A4'), 0.5, 0.5), (m('E5'), 1.0, 1.5), (m('D5'), 2.5, 1.5)]   # "on - ly now ..."

def motif(t0, vel=0.5, transpose=0, pan=0.15):
    for n, b, l in MOTIF:
        music.add(piano(n + transpose, dur=l * BEAT + 3, vel=vel), t0 + b * BEAT, pan=pan)

def chords(t0, t1, prog, vel=0.4, piano_arp=True, string_vel=0.0, shimmer_vel=0.0):
    t = t0; i = 0
    while t < t1 - 0.5:
        ch = prog[i % len(prog)]
        dur = min(BAR, t1 - t)
        if piano_arp:
            for j, n in enumerate([ch[0], ch[1], ch[2], ch[1] + 12, ch[2], ch[1]]):
                if j * BEAT * 2 / 3 < dur:
                    music.add(piano(n, dur=3.0, vel=vel * (1.0 if j == 0 else 0.7)), t + j * BEAT * 2 / 3 + rng.uniform(0, 0.01), pan=-0.25 + 0.1 * j)
        if string_vel: music.add(strings(ch, dur + 1.8, vel=string_vel), t)
        if shimmer_vel: music.add(shimmer([ch[1] + 24, ch[2] + 12], dur + 2.5, vel=shimmer_vel), t, pan=0.2)
        t += BAR; i += 1

# ------------------------------------------------------------------ cue sheet
# S1–S3: near-silence, room tone, heartbeat ticks
amb.add(noise_bed(27.5, 40, 300, 0.012), 0.0)
amb.add(np.sin(2 * np.pi * 41 * T(27)) * 0.006 * np.minimum(T(27) / 3, 1), 0.0)
TICK_ON = [(2.0, 100.5, 1.0)]
for t0, t1, v in TICK_ON:
    for bt in ui.beat_times(t0, t1):
        if t0 >= 200: g = 1.0
        elif bt < 27: g = 1.0 * min(1, (bt - 1.5) / 2.0)
        elif bt < 64: g = 0.32
        else: g = 0.4
        sfx.add(tick(), bt, gain=g * 0.9)
for i, kt in enumerate(ui.key_times(TL.ASK, 13.6, cps=11, rhythm='first-visitor')):
    sfx.add(key(space=TL.ASK[i] == ' '), kt - 0.01, pan=0.25, gain=0.9)
# S4 pull-back: rising air, then the title hit at 36.0
sfx.add(whoosh(8.5, up=False, vel=0.25), 27.4)
music.add(strings([m('D2'), m('A2')], 9.5, vel=0.55, att=6.0, rel=0.3, bright=900), 26.8)
music.add(braam(8.0, vel=1.0), 36.0)
sfx.add(boom(1.0), 36.0)
music.add(shimmer([m('D6'), m('A6'), m('F6')], 9.0, vel=0.6, att=0.4, rel=4.0), 36.05)
music.add(strings([m('D3'), m('A3'), m('D4')], 9.0, vel=0.3, att=1.0, rel=4.5, bright=1600), 37.0)
# S5 drift: the motif is born
chords(45.5, 64.0, PROG, vel=0.32, string_vel=0.18, shimmer_vel=0.25)
for k in range(4): motif(45.5 + 1.0 * BAR + k * 2 * BAR, vel=0.42)
# S6 montage: a pulse on every cut, typing in every room, the split multiplies everything
for i in range(len(TL.ROOMS)):
    tc = IF.ROOM_T0 + i * 2 * BEAT
    sfx.add(boom(0.25), tc)
    text, colour = TL.ROOMS[i]
    rng_i = np.random.default_rng(i * 7 + 3); rng_i.uniform(); rng_i.uniform(); rng_i.uniform(); cps = rng_i.uniform(9, 16)
    for j, kt in enumerate(ui.key_times(text, tc + 0.12, cps=cps, rhythm=('eulogy' if colour == TL.EULOGY_COLOUR else text))):
        if kt < tc + 2 * BEAT: sfx.add(key(vel=0.7, space=text[j] == ' '), kt, pan=rng.uniform(-0.5, 0.5), gain=0.75)
chords(64.0, 96.0, [Dm, Bb, F, C], vel=0.38, string_vel=0.22)
for k in range(int((96 - 66.5) / (2 * BAR))): motif(66.5 + k * 2 * BAR, vel=0.45, transpose=12 if k % 2 else 0)
for k in range(1500):          # the split: thousands of rooms typing at once (texture swells, then thins)
    tt = 82.25 + 9.5 * (k / 1500) ** 0.6
    sfx.add(key(vel=0.25) if k % 3 else tick(0.4), tt + rng.uniform(0, 0.05), pan=rng.uniform(-1, 1), gain=0.35 * (1 - k / 1800))
sfx.add(whoosh(2.4, up=True, vel=0.25), 93.4)
# S7 close: one held note under the last typing; then true digital silence
music.add(strings([m('A2')], 5.2, vel=0.28, att=1.0, rel=0.6, bright=700), 96.0)
for i, kt in enumerate(ui.key_times(TL.EULOGY_LAST, 97.0, cps=10, rhythm='eulogy')):
    sfx.add(key(vel=0.9, space=TL.EULOGY_LAST[i] == ' '), kt - 0.01, pan=0.2)
t_ = T(0.6); sfx.add(np.sin(2 * np.pi * (220 * np.exp(-t_ * 5)) * t_) * np.exp(-t_ * 7) * 0.25, IF.CLOSE_AT + 0.3)
SILENT = [(101.5, 105.5)]
# S8 dissolve (slowed): low strings; embers crackle as each memory breaks; HARD CUT at its end
DS = lambda t_old: TL.src_time('dissolve', t_old - 105.5)
music.add(strings([m('D2'), m('A2'), m('F3')], TL.END['dissolve'] - TL.START['dissolve'], vel=0.25, att=4.0, rel=0.05, bright=800), TL.START['dissolve'] + 0.1, new=True)
music.add(piano(m('A3'), 5.0, vel=0.22), TL.at('dissolve', 3.0), new=True)
music.add(piano(m('F3'), 5.0, vel=0.2), TL.at('dissolve', 6.8), new=True)
for (_, _, _, _, t_in, t_break) in IF.FRAGS:
    sfx.add(crackle(3.6, density=170, vel=0.9), DS(t_break), new=True)
    sfx.add(shimmer([m('A5')], 1.6, vel=0.5, att=0.05, rel=1.2), DS(t_in), new=True)
# the bridge: after one beat of silence, embers drift; a cello holds, the world keeps turning
EB = lambda r: TL.at('embers', r)
music.add(strings([m('D2'), m('A2')], 9.5, vel=0.24, att=3.0, rel=2.5, bright=500), EB(1.1), new=True)
sfx.add(crackle(7.5, density=40, vel=0.5), EB(1.2), new=True)
music.add(shimmer([m('D5')], 8.0, vel=0.22, att=3.0, rel=3.0), EB(2.0), new=True)
HARD_CUTS = [TL.END['dissolve']]
SILENT = [(TL.remap(a), TL.remap(b)) for a, b in SILENT]
SILENT.append((TL.END['dissolve'], TL.END['dissolve'] + 1.0))
# S9b reveal: one high note, then the earth breathes
music.add(strings([m('A5')], 12.0, vel=0.22, att=4.0, rel=2.0, bright=3000), 140.0)
music.add(shimmer([m('F4'), m('C5'), m('A5')], 12.0, vel=0.35, att=5.0, rel=3.0), 145.0)
# S10 push-in: the motif in major, the city, the room, a wall clock (human time), dawn birds
amb.add(noise_bed(9.0, 80, 900, 0.03), 150.5)
amb.add(noise_bed(31.0, 60, 400, 0.010), 159.0)
for k in range(26):
    amb.add(chirp(vel=0.25 + 0.4 * k / 26), 178.0 + k * 0.45 + rng.uniform(0, 0.3), pan=0.6)
# S11 her: intimate felt piano; a page turns at the insert
for k, ch in enumerate([Dm, Bb, F, C, Dm, Bb, F]):
    music.add(piano(ch[0], 4.0, vel=0.28), 190.2 + k * BAR)
    music.add(piano(ch[2] + 12, 3.0, vel=0.2), 190.2 + k * BAR + 1.5 * BEAT)
music.add(shimmer([m('F5'), m('C6')], 24.0, vel=0.3, att=4.0, rel=4.0), 190.0)

motif(203.0, vel=0.38)
# S12 relight: "Now." and two wordless pulses, then the music lets go
chords(214.0, 229.5, [Dm, Bb, F, C], vel=0.4, string_vel=0.3, shimmer_vel=0.3)
for i, tp in enumerate(TL.NOW_PULSES):
    music.add(braam(6.0, root=m('F1'), vel=0.55 + 0.2 * i, major=True), tp, new=True)
    sfx.add(boom(0.5 + 0.2 * i), tp, new=True)
    music.add(shimmer([m('F5'), m('A5'), m('C6')], 5.0, vel=0.45, att=0.05, rel=3.5), tp, new=True)
music.add(strings([m('F2'), m('C3'), m('A3'), m('E4')], 8.0, vel=0.35, att=0.5, rel=5.0), 226.0)
# S13 coda: room tone, the heartbeat returns, the same hands typing, one piano note; hard cut
CO = lambda r: TL.at('coda', r)
C_ = IF.CODA
amb.add(noise_bed(C_['cut'] - 0.3, 40, 300, 0.012), CO(0.3), new=True)
for bt in ui.beat_times(CO(C_['caret']), CO(C_['cut']), phase=-(TL.START['coda'] % 1.0)):
    sfx.add(tick(), bt, gain=0.9, new=True)
for txt, t0_ in ((TL.RETURN, C_['type1']), (TL.RETURN2, C_['type2'])):
    for i, kt in enumerate(ui.key_times(txt, t0_, cps=10, rhythm='eulogy')):
        sfx.add(key(vel=0.9, space=txt[i] == ' '), CO(kt) - 0.01, pan=0.2, new=True)
music.add(piano(m('F4'), 4.0, vel=0.22), CO(C_['type2'] + 4.6), new=True)
music.add(piano(m('A4'), 4.0, vel=0.2), CO(C_['type2'] + 4.6 + BEAT), new=True)
music.add(piano(m('D5'), 5.0, vel=0.26), CO(C_['type2'] + 4.6 + 2 * BEAT), new=True)
HARD_CUTS.append(CO(C_['cut']))
SILENT.append((CO(C_['cut']), TL.at('credits', 1.3)))
# S14 credits: the title, the motif one last time
music.add(strings([m('D3'), m('A3'), m('F4')], 9.0, vel=0.25, att=2.5, rel=4.0, bright=1200), 249.3)
motif(250.4, vel=0.4)
music.add(piano(m('D4'), 6.0, vel=0.3), 250.4 + 4 * BEAT)
for k in range(130):
    sfx.add(key(vel=0.25), 259.0 + k * 0.05 + rng.uniform(0, 0.02), pan=-0.3, gain=0.4)


# ================================================================== extended-cut scenes (absolute times)
import machine as MC
def A(key, rel): return TL.at(key, rel)
def fan_bed(dur, vel):
    x = bp(rng.standard_normal(int(SR * dur)), 70, 900); x /= np.abs(x).max() + 1e-9
    t = T(dur); hum = sum(np.sin(2 * np.pi * 120 * k * t) / k for k in (1, 2, 3)) * 0.12
    return (x + hum) * vel
def spin_down(dur, vel):
    n = int(SR * dur); t = T(dur); x = rng.standard_normal(n); out = np.zeros(n); blk = 2048
    for i in range(0, n, blk):
        u = i / n; fc = 900 * (1 - u) ** 2 + 40
        out[i:i + blk] = lp(x[max(0, i - 8192):i + blk], fc)[-len(out[i:i + blk]):]
    f = 120 * (1 - t / dur) ** 1.6 + 1
    hum = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.35 + np.sin(2 * np.pi * np.cumsum(2 * f) / SR) * 0.15
    env = (1 - t / dur) ** 0.8
    return (out / (np.abs(out).max() + 1e-9) * 0.8 + hum) * env * vel
def collide(vel=1.0):
    t = T(0.6); return (np.sin(2 * np.pi * 30 * t) * np.exp(-t * 7) + np.sin(2 * np.pi * 61 * t) * np.exp(-t * 11) * 0.4) * vel * 0.9
def click(f=1000, vel=1.0):
    t = T(0.02); return np.sin(2 * np.pi * f * t) * np.exp(-t * 600) * vel * 0.15
def shepard(dur, vel):
    t = T(dur); s = np.zeros_like(t)
    rate = 1 / 26.0                     # one octave over the whole shot, never arriving
    for k in range(7):
        oc = (k + t * rate) % 7
        f = 55 * 2 ** oc
        amp = np.exp(-((oc - 3.5) / 1.6) ** 2)
        s += np.sin(2 * np.pi * np.cumsum(f) / SR) * amp
    env = np.minimum(1, np.minimum(t / 2.0, (dur - t) / 2.5)).clip(0)
    return s * env * vel * 0.05

# I · the machine hall: fans, drives, the heartbeat made enormous
amb.add(fan_bed(16.3, 0.035) * np.minimum(T(16.3) / 1.2, 1), A('hall', 0), new=True)
for k in range(28):
    sfx.add(click(rng.uniform(2500, 4200), 0.5), A('hall', rng.uniform(0.5, 15.5)), pan=rng.uniform(-0.8, 0.8), new=True)
for bt in ui.beat_times(A('hall', 0.6), A('hall', 16.0), phase=-TL.BEAT0):
    sfx.add(tick(1.0), bt, gain=0.8, new=True); sfx.add(boom(0.035), bt, new=True)
music.add(strings([m('D2'), m('A2')], 17.0, vel=0.22, att=4.0, rel=2.0, bright=600), A('hall', 0.5), new=True)
# I · the slit: each blink is a collision of two universes
for bt in ui.beat_times(A('slit', 0.0), A('slit', 12.0), phase=-TL.BEAT0):
    sfx.add(collide(0.45), bt, new=True); sfx.add(tick(0.45), bt, new=True)
amb.add(np.sin(2 * np.pi * 7400 * T(12)) * 0.0025 * np.minimum(1, np.minimum(T(12) / 2, (12 - T(12)) / 2)), A('slit', 0), new=True)
# III · two clocks: machine time is exact; living time is not
for k in range(int(20 / 0.42)):
    tt = k * 0.42
    if tt < 10.0 or (10.0 <= tt < 18.5):
        sfx.add(click(1000, 0.6), TL.src_time('clocks', tt), pan=-0.35 if tt > 10 else 0.0, new=True)
for (ts, ti, spd) in MC.neuron_data()['sparks']:
    ta = ts + spd
    if 7.0 < ta < 19.0:
        sfx.add(crackle(0.35, density=60, vel=0.5), TL.src_time('clocks', ta), pan=0.35, new=True)
music.add(shimmer([m('D5'), m('A5')], 22.0, vel=0.25, att=3.0, rel=3.0), A('clocks', 0), new=True)
# III · the line and the point
music.add(shepard(TL.END['line'] - TL.START['line'], 1.0), A('line', 0), new=True)
for tp in MC.point_times()[:-1]:
    sfx.add(tick(0.9), tp, new=True); sfx.add(shimmer([m('A6')], 0.8, vel=0.25, att=0.01, rel=0.7), tp, new=True)
tm = MC.point_times()[-1]
music.add(shimmer([m('F5'), m('A5'), m('E6')], 4.0, vel=0.5, att=0.05, rel=3.0), tm, new=True)
music.add(piano(m('E5'), 4.0, vel=0.35), tm, new=True)
# IV · the library at closing time: the chime, the air going quiet, a click as each lamp goes out
LB = lambda r: TL.at('library', r)
amb.add(fan_bed(16.0, 0.02) * np.minimum(T(16.0) / 1.0, 1), LB(0.4), new=True)
amb.add(spin_down(6.0, 0.02), LB(16.2), new=True)
for j, n_ in enumerate((m('E5'), m('C5'), m('G4'))):                     # "the library will close in ten minutes"
    sfx.add(shimmer([n_], 1.6, vel=0.55, att=0.01, rel=1.4) + piano(n_, 1.6, vel=0.25), LB(1.0 + 0.5 * j), pan=0.0, new=True)
def lamp_off_time(z):
    lo, hi = 4.5, 18.0
    for _ in range(30):
        mid = (lo + hi) / 2
        fr = MC.LIB_Z1 + 1 - (MC.LIB_Z1 + 1 - 2.6) * ease_(min(max((mid - 4.5) / 13.5, 0), 1))
        if fr > z: lo = mid
        else: hi = mid
    return lo
def ease_(u): return u * u * u * (u * (u * 6 - 15) + 10)
for (x_, z_) in MC.lib_data()['lamps']:
    if z_ > 2.7:
        sfx.add(click(rng.uniform(1700, 2400), 0.45), LB(lamp_off_time(z_)), pan=0.6 * np.sign(x_) * min(1, 6 / z_), new=True)
for dt in (0.0, 0.25, 0.45):
    sfx.add(click(2000, 0.6), LB(20.6 + dt), new=True)
for k in range(16):
    sfx.add(lp(rng.standard_normal(int(0.08 * SR)), 300) * np.exp(-T(0.08) * 40) * 0.12, LB(6.0 + k * 0.62), pan=-0.4 + 0.05 * k, new=True)
music.add(strings([m('D1'), m('A1')], 18.0, vel=0.3, att=2.0, rel=3.0, bright=250), LB(1.2), new=True)

# IV · closing time: an accelerating ostinato that builds to the last light — then nothing
iv = Bus()                     # the closing-time build gets its own bus so it can sink away after the climax
CLIMAX = TL.src_time('extinction', 18.0)
B0 = LB(2.0)
def pluck(n, dur, cutoff, vel):
    t = T(dur); f = mtof(n)
    ph = 2 * np.pi * f * t
    x = ((ph / np.pi) % 2 - 1) * 0.6 + np.sin(ph) * 0.5
    x = lp(x, cutoff) * np.exp(-t * 9) * np.minimum(t / 0.004, 1)
    return x * vel
OST = [m('D2'), m('D2'), m('F2'), m('D2'), m('Eb2'), m('D2'), m('C2'), m('D2')]
tt, i = B0, 0
span = CLIMAX - B0
while tt < CLIMAX - 0.02:
    u = (tt - B0) / span
    bpm = 84 + (156 - 84) * u ** 1.5
    step = 60.0 / bpm / 2                       # eighth notes, ever faster
    cutoff = 500 + 4200 * u ** 1.4              # the filter opens: urgency
    iv.add(pluck(OST[i % 8], 0.35, cutoff, 0.16 + 0.22 * u), tt, pan=-0.15 + 0.3 * (i % 2), new=True)
    iv.add(pluck(OST[i % 8] + 12, 0.25, cutoff, (0.05 + 0.12 * u) * (u > 0.35)), tt, pan=0.25, new=True)
    if i % 2 == 0:
        iv.add(boom(0.10 + 0.45 * u ** 1.3), tt, new=True)
    elif u > 0.4:
        iv.add(hp(rng.standard_normal(int(0.04 * SR)), 6000) * np.exp(-T(0.04) * 120) * 0.08 * u, tt, pan=rng.uniform(-0.4, 0.4), new=True)
    tt += step; i += 1
E0 = TL.at('extinction', 1.0)
iv.add(strings([m('D2'), m('A2'), m('Eb3'), m('Bb3'), m('E4')], CLIMAX - E0 + 4.0, vel=0.65, att=CLIMAX - E0, rel=3.5, bright=1600), E0, new=True)
iv.add(strings([m('D1'), m('D2')], CLIMAX - B0 + 4.0, vel=0.4, att=CLIMAX - B0, rel=3.5, bright=300), B0, new=True)
trem_len = 9.0
trem = strings([m('F5'), m('E6')], trem_len + 3.0, vel=0.35, att=trem_len, rel=2.8, bright=5000)
trem *= (0.6 + 0.4 * np.sin(2 * np.pi * 9 * T(len(trem) / SR)))      # bowed tremolo
iv.add(trem, CLIMAX - trem_len, new=True)
iv.add(shepard(12.5, 2.2), CLIMAX - 12.0, new=True)
iv.add(whoosh(7.0, up=True, vel=0.35), CLIMAX - 7.0, new=True)
iv.add(braam(6.0, vel=1.0), CLIMAX, new=True)          # the peak...
iv.add(boom(1.2), CLIMAX, new=True)
# ...then the whole build sinks away (see the mix: a falling low-pass and a long tail), and the heartbeat rises out of it
sfx.add(whoosh(3.0, up=False, vel=0.14), CLIMAX + 0.3, new=True)
IV_RANGE = (B0, CLIMAX)

# V · the keyboard: room tone, the clock, the key coming back up
amb.add(noise_bed(10.0, 60, 400, 0.010), A('keyboard', 0), new=True)
sfx.add(key(vel=0.35), A('keyboard', 2.2), new=True)
music.add(strings(F, 10.5, vel=0.14, att=2.0, rel=3.0), A('keyboard', 0), new=True)
music.add(piano(m('F4'), 3.0, vel=0.25), A('keyboard', 2.4), new=True)
music.add(piano(m('A4'), 3.0, vel=0.22), A('keyboard', 2.4 + BEAT), new=True)
music.add(piano(m('E5'), 4.0, vel=0.22), A('keyboard', 2.4 + 2 * BEAT), new=True)
for k in range(10):
    amb.add(chirp(vel=0.35), A('keyboard', 1.0 + k * 0.85 + rng.uniform(0, 0.3)), pan=-0.6, new=True)

# ================================================================== pacing pass: every chapter gets an arc and synced events
# 序 — anticipation: a breath that grows under the lone cursor and stops dead on the first keystroke
first_key = ui.key_times(TL.ASK, 13.6, cps=11, rhythm='first-visitor')[0]
fk = TL.remap(first_key - 0.01)
swell_len = fk - 2.0
sw = strings([m('D2'), m('A2')], swell_len, vel=0.16, att=swell_len, rel=0.02, bright=420)
music.add(sw, 2.0, new=True)
music.add(shimmer([m('A5'), m('E6')], 3.5, vel=0.25, att=0.05, rel=3.0), TL.at('answer', 1.0), new=True)
# III · clocks — exact machine time against living time; they resolve together
CL = lambda r: TL.src_time('clocks', r)
arp = [m('D4'), m('A4'), m('D5'), m('A4')]
k = 0; tt = 0.4
while tt < 18.0:
    if tt < 10.0 or tt >= 10.0:
        pan = 0.0 if tt < 10.0 else -0.45
        v = 0.10 if tt < 15.5 else 0.10 * (1 - (tt - 15.5) / 2.5)
        if v > 0.01: music.add(pluck(arp[k % 4], 0.3, 2600, v), CL(tt), pan=pan, new=True)
    tt += 0.21; k += 1
pent = [m('D4'), m('F4'), m('G4'), m('A4'), m('C5'), m('D5')]
for j, (ts, ti, spd) in enumerate(MC.neuron_data()['sparks']):
    ta = ts + spd
    if 7.0 < ta < 17.5:
        music.add(piano(pent[(ti + j) % len(pent)], 2.5, vel=0.20), CL(ta), pan=0.4, new=True)
music.add(piano(m('D3'), 4.0, vel=0.3), CL(18.0), new=True)
for n_ in (m('F4'), m('A4'), m('D5')):
    music.add(piano(n_, 4.0, vel=0.22), CL(18.0), new=True)
music.add(strings([m('D3'), m('A3'), m('F4')], 3.5, vel=0.18, att=0.6, rel=2.5), CL(17.6), new=True)
# III · the line — every memory it reaches is a note; together they are the theme
mel = [m('F4'), m('A4'), m('E5'), m('D5'), m('C5'), m('A4'), m('G4'), m('A4')]
for j, bt in enumerate(MC.bead_times()):
    music.add(piano(mel[j % len(mel)], 2.8, vel=0.18 + 0.06 * (j % 4 == 0)), bt, pan=-0.6 + 1.2 * j / 30, new=True)
tm2 = MC.point_times()[-1]
music.add(strings([m('F3'), m('C4'), m('A4'), m('E5')], 5.5, vel=0.25, att=0.8, rel=4.0), tm2 - 0.2, new=True)
# IV · forgetting — a falling low note as each memory breaks
for (_, _, _, _, t_in, t_break), n_ in zip(IF.FRAGS, (m('A3'), m('F3'), m('D3'))):
    music.add(piano(n_, 5.0, vel=0.32), DS(t_break), new=True)
    music.add(piano(n_ - 12, 5.0, vel=0.2), DS(t_break), new=True)
# V · the desk — the dawn theme: strings open as the page appears; the theme blooms as day breaks
PI = lambda r: TL.at('pushin', r)
music.add(strings([m('F3'), m('A3'), m('C4'), m('E4')], 26.0, vel=0.22, att=8.0, rel=6.0, bright=1800), PI(8.0), new=True)
G0 = TL.BRIDGE_G0
snap = lambda t: G0 + math.ceil((t - G0) / TL.BEAT) * TL.BEAT
dawn = snap(TL.src_time('pushin', 22.0))
for j, (n_, b, l) in enumerate(MOTIF):
    music.add(piano(n_ + 12, l * BEAT + 3.5, vel=0.32), dawn + b * BEAT, pan=0.2, new=True)
    music.add(piano(n_, l * BEAT + 3.5, vel=0.22), dawn + b * BEAT, pan=-0.1, new=True)
for j, n_ in enumerate((m('F3'), m('E3'), m('D3'), m('C3'))):
    music.add(strings([n_ - 12, n_], BAR + 0.6, vel=0.20, att=1.0, rel=1.2, bright=700), dawn + j * BAR, new=True)
music.add(shimmer([m('C6'), m('F6')], 12.0, vel=0.3, att=4.0, rel=4.0), dawn + 1.0, new=True)

# ================================================================== chapter bridges
# IV -> V: one pulse grid (a resting heartbeat, BEAT) carries us from the dark into the room,
#          where the heartbeat becomes the wall clock. The music of the room lands on the same grid.
G0 = TL.BRIDGE_G0
snap = lambda t: G0 + math.ceil((t - G0) / TL.BEAT) * TL.BEAT
def lubdub(vel=1.0):
    t = T(0.5)
    a = np.sin(2 * np.pi * 52 * t) * np.exp(-t * 18)
    b = np.zeros_like(t); d = int(0.19 * SR)
    b[d:] = np.sin(2 * np.pi * 44 * t[:-d]) * np.exp(-t[:-d] * 22) * 0.6
    return (a + b + lp(rng.standard_normal(len(t)), 200) * np.exp(-t * 30) * 0.2) * vel * 0.7
end_pulse = TL.END['keyboard'] - 0.3
k = 0; tp = G0
while tp < end_pulse:
    w_heart = 1.0 - min(1.0, max(0.0, (tp - TL.at('pushin', 2.0)) / 5.0))
    w_clock = min(1.0, max(0.0, (tp - TL.at('pushin', 2.0)) / 5.0))
    fade_in = min(1.0, (tp - G0) / 2.5 + 0.25)
    if w_heart > 0.01: sfx.add(lubdub(0.55 * w_heart * fade_in), tp, new=True)
    if w_clock > 0.01: amb.add(clock_tick(tock=k % 2 == 1), tp, gain=0.95 * w_clock, new=True)
    tp += TL.BEAT; k += 1
# a pedal D in the dark that opens into F major as the earth appears (relative major: the same notes, a new light)
music.add(strings([m('D2'), m('A2')], TL.at('reveal', 4.0) - G0 + 1.5, vel=0.22, att=2.5, rel=2.0, bright=500), G0, new=True)
music.add(strings([m('F2'), m('C3'), m('A3')], 10.0, vel=0.22, att=3.0, rel=3.5, bright=900), snap(TL.at('reveal', 3.0)), new=True)
# the room's music, on the grid
chords(snap(TL.at('pushin', 4.5)), TL.END['keyboard'] - 0.5, [F, C, Dm, Bb], vel=0.28, piano_arp=True, string_vel=0.14, shimmer_vel=0.12)
motif(snap(TL.at('pushin', 8.0)), vel=0.38)
motif(snap(TL.at('pushin', 15.5)), vel=0.33, transpose=-2)
# III -> II: the line's rising tone climbs straight into the montage's first downbeat
sfx.add(whoosh(3.2, up=True, vel=0.22), TL.at('montage', 2.5) - 3.2, new=True)
music.add(shepard(4.0, 1.4), TL.END['line'] - 1.0, new=True)
# II -> III and V -> VI crossfades: a sustained tone carries across each dissolve
music.add(shimmer([m('D5'), m('A5')], 4.0, vel=0.3, att=1.0, rel=2.5), TL.START['clocks'] - 1.2, new=True)
music.add(strings([m('F3'), m('C4')], 5.0, vel=0.16, att=1.0, rel=3.0), TL.START['her'] - 1.0, new=True)
# I: the dive into the caret lands on the hall's first heartbeat
sfx.add(whoosh(1.2, up=True, vel=0.2), TL.START['hall'] - 1.1, new=True)
# VI: her chapter is longer now; two more bars, and a breath for every flashback
for k2, ch in enumerate([Dm, Bb, F]):
    music.add(piano(ch[0], 4.0, vel=0.26), TL.at('her', 24.2) + k2 * BAR, new=True)
    music.add(piano(ch[2] + 12, 3.0, vel=0.18), TL.at('her', 24.2) + k2 * BAR + 1.5 * BEAT, new=True)
music.add(shimmer([m('F5'), m('C6')], 8.0, vel=0.25, att=1.0, rel=4.0), TL.at('her', 22.0), new=True)
for (fs, fd, sk, sr_) in TL.FLASHBACKS:
    rev = shimmer([m('A5') + (2 if fd < 0.5 else 0)], 0.6 + fd, vel=0.35, att=0.5 + fd * 0.5, rel=0.08)
    sfx.add(rev, TL.at('her', fs) - 0.4, pan=0.2, new=True)
    sfx.add(whoosh(0.5 + fd * 0.4, up=True, vel=0.12), TL.at('her', fs) - 0.25, new=True)

# VI rebuilt: a soft chime as each window of memory arrives; the heartbeat ticks again under 'this moment'
import finale as FN
for (ap, key_, rel_, wx_, wy_, acc_) in FN.WINDOWS:
    music.add(shimmer([m('A5') if wx_ < 0 else m('E6')], 2.0, vel=0.28, att=0.4, rel=1.6), TL.at('her', ap + 0.6), pan=0.5 * np.sign(wx_), new=True)
for bt in ui.beat_times(TL.at('her', FN.NOWWIN + 0.5), TL.at('her', FN.BURST)):
    sfx.add(tick(0.5), bt, new=True)
for bt in ui.beat_times(TL.at('her', 0.4), TL.at('her', 4.0)):
    sfx.add(tick(0.7), bt, new=True)
music.add(strings([m('F3'), m('C4'), m('A4')], 7.0, vel=0.2, att=1.5, rel=4.0), TL.at('her', FN.NOWWIN), new=True)

# ------------------------------------------------------------------ voice-over
meta = json.load(open('build/vo/meta.json'))
vo_env = np.zeros(N)
for t0, k in TL.VO:
    sr, a = wavfile.read(f'build/vo/{k}.wav')
    a = resample_poly(a.astype(np.float64), SR, sr)
    a = hp(a, 90) * 0.9
    vo.add(a, t0, new=True)
    i0 = int(t0 * SR); vo_env[i0:i0 + len(a)] = 1.0

# ------------------------------------------------------------------ mix
def reverb(st, secs, wet, seed=0):
    r = np.random.default_rng(seed); t = T(secs)
    irL = r.standard_normal(len(t)) * np.exp(-t * 6.9 / secs); irR = r.standard_normal(len(t)) * np.exp(-t * 6.9 / secs)
    irL = lp(irL, 6000); irR = lp(irR, 6000)
    irL /= np.sqrt((irL ** 2).sum()); irR /= np.sqrt((irR ** 2).sum())
    return st + wet * np.stack([fftconvolve(st[:, 0], irL)[:N], fftconvolve(st[:, 1], irR)[:N]], 1)

ivs = iv.st()
i0 = int((CLIMAX + 0.25) * SR); i1 = min(N, int((CLIMAX + 6.5) * SR))
seg = ivs[i0:i1].copy(); tt_ = np.arange(len(seg)) / SR
sunk = np.zeros_like(seg); blk_ = 1024
for j in range(0, len(seg), blk_):
    fc = 220 + 7500 * np.exp(-(j / SR) / 0.9)          # the build recedes, as if sinking under water
    sos = butter(2, fc, 'low', fs=SR, output='sos')
    sunk[j:j + blk_] = sosfilt(sos, seg[max(0, j - 8192):j + blk_], axis=0)[-len(sunk[j:j + blk_]):]
ivs[i0:i1] = sunk * np.exp(-tt_ / 1.6)[:, None]
ivs[i1:] = 0
mus = reverb(music.st() + ivs, 4.2, 0.5, 1)
fx = reverb(sfx.st(), 1.4, 0.18, 2)
vox = reverb(vo.st(), 0.9, 0.10, 3)
am = amb.st()
# duck music under the voice (smooth)
k = int(0.25 * SR)
duck = np.convolve(vo_env, np.ones(k) / k, mode='same')
depth = np.full(N, 0.38); depth[int(IV_RANGE[0] * SR):int(IV_RANGE[1] * SR)] = 0.55   # keep the words clear over the build
mus *= (1 - depth * duck)[:, None]
mix = mus * 0.9 + fx * 0.75 + vox * 1.25 + am
# hard cuts and true digital silence
for tc in HARD_CUTS:
    i = int(tc * SR); mix[i:i + int(0.9 * SR)] = 0
for a, b in SILENT:
    mix[int(a * SR):int(b * SR)] = 0
# soft-clip and normalize to -1 dBFS
mix = np.tanh(mix * 1.2) / 1.2
mix *= 0.89 / max(np.abs(mix).max(), 1e-9)
for a, b in SILENT:
    mix[int(a * SR):int(b * SR)] = 0
wavfile.write('build/mix.wav', SR, (mix * 32767).astype(np.int16))
print('mix ok', mix.shape, 'peak', float(np.abs(mix).max()))
