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
TICK_ON = [(2.0, 100.5, 1.0), (233.5, 247.2, 1.0)]
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
# S8 dissolve: low strings; embers crackle as each memory breaks; HARD CUT at 117.44
music.add(strings([m('D2'), m('A2'), m('F3')], 12.0, vel=0.25, att=3.0, rel=0.05, bright=800), 105.6)
for (_, _, _, _, t_in, t_break) in IF.FRAGS:
    sfx.add(crackle(2.4, density=220, vel=0.9), t_break)
    sfx.add(shimmer([m('A5')], 1.6, vel=0.5, att=0.05, rel=1.2), t_in)
HARD_CUTS = [TL.END['dissolve']]
SILENT = [(TL.remap(a), TL.remap(b)) for a, b in SILENT]
SILENT.append((TL.END['dissolve'], TL.END['dissolve'] + 1.0))
# S9 extinction: a dissonant cluster that swells and is cut off; then near-nothing
music.add(strings([m('D2'), m('A2'), m('Eb3'), m('Bb3'), m('E4')], 17.5, vel=0.6, att=12.0, rel=0.15, bright=1400), 118.3)
music.add(strings([m('D1'), m('D2')], 20.5, vel=0.45, att=14.0, rel=4.0, bright=300), 118.3)
for k in range(40):
    sfx.add(shimmer([m('A6') + int(rng.integers(-5, 5))], 1.2, vel=0.25, att=0.01, rel=1.1), 120.5 + k * 0.35 + rng.uniform(0, 0.2), pan=rng.uniform(-0.8, 0.8))
# S9b reveal: one high note, then the earth breathes
music.add(strings([m('A5')], 12.0, vel=0.22, att=4.0, rel=2.0, bright=3000), 140.0)
music.add(shimmer([m('F4'), m('C5'), m('A5')], 12.0, vel=0.35, att=5.0, rel=3.0), 145.0)
# S10 push-in: the motif in major, the city, the room, a wall clock (human time), dawn birds
amb.add(noise_bed(9.0, 80, 900, 0.03), 150.5)
chords(155.6, 190.0, [F, C, Dm, Bb], vel=0.30, piano_arp=True, string_vel=0.16, shimmer_vel=0.15)
motif(159.5, vel=0.4, transpose=0)
motif(170.0, vel=0.35, transpose=-2)
amb.add(noise_bed(31.0, 60, 400, 0.010), 159.0)
for k in range(30):
    amb.add(clock_tick(tock=k % 2 == 1), 159.6 + k, gain=0.9)
for k in range(26):
    amb.add(chirp(vel=0.25 + 0.4 * k / 26), 178.0 + k * 0.45 + rng.uniform(0, 0.3), pan=0.6)
# S11 her: intimate felt piano; a page turns at the insert
for k, ch in enumerate([Dm, Bb, F, C, Dm, Bb, F]):
    music.add(piano(ch[0], 4.0, vel=0.28), 190.2 + k * BAR)
    music.add(piano(ch[2] + 12, 3.0, vel=0.2), 190.2 + k * BAR + 1.5 * BEAT)
music.add(shimmer([m('F5'), m('C6')], 24.0, vel=0.3, att=4.0, rel=4.0), 190.0)
sfx.add(whoosh(0.6, up=True, vel=0.12), 200.6)
motif(203.0, vel=0.38)
# S12 relight: "Now." and two wordless pulses, then the music lets go
chords(214.0, 229.5, [Dm, Bb, F, C], vel=0.4, string_vel=0.3, shimmer_vel=0.3)
for i, tp in enumerate(TL.NOW_PULSES):
    music.add(braam(6.0, root=m('F1'), vel=0.55 + 0.2 * i, major=True), tp, new=True)
    sfx.add(boom(0.5 + 0.2 * i), tp, new=True)
    music.add(shimmer([m('F5'), m('A5'), m('C6')], 5.0, vel=0.45, att=0.05, rel=3.5), tp, new=True)
music.add(strings([m('F2'), m('C3'), m('A3'), m('E4')], 8.0, vel=0.35, att=0.5, rel=5.0), 226.0)
# S13 coda: room tone, the heartbeat returns, the same hands typing, one piano note; hard cut
amb.add(noise_bed(15.0, 40, 300, 0.012), 232.3)
for i, kt in enumerate(ui.key_times(TL.RETURN, 236.4, cps=10, rhythm='eulogy')):
    sfx.add(key(vel=0.9, space=TL.RETURN[i] == ' '), kt - 0.01, pan=0.2)
music.add(piano(m('D5'), 3.0, vel=0.3), 244.4)
HARD_CUTS.append(TL.remap(247.2))
SILENT.append((TL.remap(247.2), TL.remap(249.3)))
# S14 credits: the title, the motif one last time
music.add(strings([m('D3'), m('A3'), m('F4')], 9.0, vel=0.25, att=2.5, rel=4.0, bright=1200), 249.3)
motif(250.4, vel=0.4)
music.add(piano(m('D4'), 6.0, vel=0.3), 250.4 + 4 * BEAT)
for k in range(130):
    sfx.add(key(vel=0.25), 257.3 + k * 0.05 + rng.uniform(0, 0.02), pan=-0.3, gain=0.4)


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
for bt in ui.beat_times(A('hall', 0.6), A('hall', 16.0)):
    sfx.add(tick(1.0), bt, gain=0.8, new=True); sfx.add(boom(0.035), bt, new=True)
music.add(strings([m('D2'), m('A2')], 17.0, vel=0.22, att=4.0, rel=2.0, bright=600), A('hall', 0.5), new=True)
# I · the slit: each blink is a collision of two universes
for bt in ui.beat_times(A('slit', 0.0), A('slit', 12.0)):
    sfx.add(collide(0.45), bt, new=True); sfx.add(tick(0.45), bt, new=True)
amb.add(np.sin(2 * np.pi * 7400 * T(12)) * 0.0025 * np.minimum(1, np.minimum(T(12) / 2, (12 - T(12)) / 2)), A('slit', 0), new=True)
# III · two clocks: machine time is exact; living time is not
for k in range(int(20 / 0.42)):
    tt = k * 0.42
    if tt < 10.0 or (10.0 <= tt < 18.5):
        sfx.add(click(1000, 0.6), A('clocks', tt), pan=-0.35 if tt > 10 else 0.0, new=True)
for (ts, ti, spd) in MC.neuron_data()['sparks']:
    ta = ts + spd
    if 7.0 < ta < 19.0:
        sfx.add(crackle(0.35, density=60, vel=0.5), A('clocks', ta), pan=0.35, new=True)
music.add(shimmer([m('D5'), m('A5')], 20.0, vel=0.25, att=3.0, rel=3.0), A('clocks', 0), new=True)
# III · the line and the point
music.add(shepard(26.0, 1.0), A('line', 0), new=True)
for tp in MC.point_times()[:-1]:
    sfx.add(tick(0.9), tp, new=True); sfx.add(shimmer([m('A6')], 0.8, vel=0.25, att=0.01, rel=0.7), tp, new=True)
tm = MC.point_times()[-1]
music.add(shimmer([m('F5'), m('A5'), m('E6')], 4.0, vel=0.5, att=0.05, rel=3.0), tm, new=True)
music.add(piano(m('E5'), 4.0, vel=0.35), tm, new=True)
# IV · the hall powers down: the only mechanical death in the film
amb.add(fan_bed(2.2, 0.035) * np.minimum(T(2.2) / 1.0, 1), A('hall_dark', 1.0), new=True)
amb.add(spin_down(9.5, 0.045), A('hall_dark', 3.0), new=True)
for k in range(40):
    sfx.add(click(rng.uniform(1800, 3000), 0.45), A('hall_dark', 3.0 + 7.8 * (k / 40) ** 0.8 + rng.uniform(0, 0.15)), pan=rng.uniform(-0.9, 0.9), new=True)
for dt in (0.0, 0.55):
    sfx.add(tick(1.0), A('hall_dark', 11.4 + dt), new=True)
music.add(strings([m('D1'), m('A1')], 12.0, vel=0.3, att=2.0, rel=3.0, bright=250), A('hall_dark', 1.2), new=True)
# V · the keyboard: room tone, the clock, the key coming back up
amb.add(noise_bed(10.0, 60, 400, 0.010), A('keyboard', 0), new=True)
for k in range(10):
    amb.add(clock_tick(tock=k % 2 == 1), A('keyboard', 0.6 + k), gain=0.9, new=True)
sfx.add(key(vel=0.35), A('keyboard', 2.2), new=True)
music.add(strings(F, 10.5, vel=0.14, att=2.0, rel=3.0), A('keyboard', 0), new=True)
music.add(piano(m('F4'), 3.0, vel=0.25), A('keyboard', 2.4), new=True)
music.add(piano(m('A4'), 3.0, vel=0.22), A('keyboard', 2.4 + BEAT), new=True)
music.add(piano(m('E5'), 4.0, vel=0.22), A('keyboard', 2.4 + 2 * BEAT), new=True)
for k in range(10):
    amb.add(chirp(vel=0.35), A('keyboard', 1.0 + k * 0.85 + rng.uniform(0, 0.3)), pan=-0.6, new=True)

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

mus = reverb(music.st(), 4.2, 0.45, 1)
fx = reverb(sfx.st(), 1.4, 0.18, 2)
vox = reverb(vo.st(), 0.9, 0.10, 3)
am = amb.st()
# duck music under the voice (smooth)
k = int(0.25 * SR)
duck = np.convolve(vo_env, np.ones(k) / k, mode='same')
mus *= (1 - 0.38 * duck)[:, None]
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
