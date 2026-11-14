"""Synthesize an original soft piano/pad score + transition SFX for 《Claude》 (120 s)."""
import numpy as np
from scipy.signal import fftconvolve, butter, sosfilt
from scipy.io import wavfile
from timeline import SEGMENTS, FLASHES, HARD_CUTS, DURATION

SR = 44100
N = int(SR * DURATION)
L = np.zeros(N); R = np.zeros(N)
rng = np.random.default_rng(5)

def midi(n): return 440.0 * 2 ** ((n - 69) / 12)

def add(sig, t0, pan=0.0, gain=1.0):
    i = int(t0 * SR)
    if i >= N: return
    sig = sig[:N - i] * gain
    L[i:i + len(sig)] += sig * np.sqrt(0.5 * (1 - pan))
    R[i:i + len(sig)] += sig * np.sqrt(0.5 * (1 + pan))

def piano(n, dur=3.0, vel=0.5):
    f = midi(n); t = np.arange(int(SR * dur)) / SR
    s = np.zeros_like(t)
    for k in range(1, 9):
        fk = f * k * (1 + 0.0004 * k * k)
        s += np.sin(2 * np.pi * fk * t + rng.uniform(0, 6)) / k ** 1.4 * np.exp(-t * (0.9 + 0.55 * k) * (f / 400) ** 0.3)
    att = np.minimum(t / 0.004, 1)
    rel = np.minimum(1, (dur - t) / 0.3)
    return s * att * rel * vel * 0.22

def bell(n, dur=4.0, vel=0.4):
    f = midi(n); t = np.arange(int(SR * dur)) / SR
    s = np.sin(2 * np.pi * f * t) * np.exp(-t * 1.2) + 0.35 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t * 3) \
        + 0.15 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t * 6)
    return s * np.minimum(t / 0.002, 1) * vel * 0.12

def pad(notes, dur, vel=0.5):
    t = np.arange(int(SR * dur)) / SR
    s = np.zeros_like(t)
    for n in notes:
        for d in (-0.08, 0.0, 0.07):
            f = midi(n) * 2 ** (d / 12)
            ph = 2 * np.pi * f * t + rng.uniform(0, 6)
            s += (np.sin(ph) + 0.3 * np.sin(2 * ph) + 0.12 * np.sin(3 * ph)) / 3
    env = np.minimum(t / 1.2, 1) * np.minimum(1, (dur - t) / 1.5)
    sos = butter(2, 1800, 'low', fs=SR, output='sos')
    return sosfilt(sos, s) * env * vel * 0.05

# progression (D major, 72 bpm, 4/4 -> bar = 3.333 s)
BAR = 60 / 72 * 4
CH = {
    'D':   [50, 57, 62, 66, 69, 73],     # Dmaj7 voicing
    'Bm':  [47, 54, 59, 62, 66, 69],
    'G':   [43, 50, 55, 59, 62, 66],
    'A':   [45, 52, 57, 61, 64, 69],
    'Asus': [45, 52, 57, 62, 64, 69],
    'F#m': [42, 49, 54, 57, 61, 64],
    'Em':  [40, 47, 52, 55, 59, 62],
}
prog = ['D', 'Bm', 'G', 'Asus'] * 3 + ['G', 'A', 'F#m', 'Bm'] * 2 + ['D', 'Bm', 'G', 'A'] * 2 + \
       ['Em', 'A', 'F#m', 'Bm', 'G', 'A', 'D', 'Bm', 'G', 'Asus', 'D', 'D']
melody_by_bar = {  # sparse melody (midi, beat offset, length beats)
    8: [(78, 0, 2), (76, 2, 1), (74, 3, 1)], 9: [(73, 0, 3)], 10: [(71, 0, 2), (74, 2, 2)], 11: [(76, 0, 4)],
    12: [(78, 0, 2), (81, 2, 2)], 13: [(78, 0, 3), (76, 3, 1)], 14: [(73, 0, 2), (76, 2, 2)], 15: [(74, 0, 4)],
    26: [(81, 0, 2), (78, 2, 2)], 27: [(76, 0, 3), (73, 3, 1)], 28: [(78, 0, 2), (76, 2, 1), (74, 3, 1)],
    29: [(74, 0, 4)], 30: [(79, 0, 2), (78, 2, 2)], 31: [(76, 0, 4)], 32: [(78, 0, 2), (74, 2, 2)], 33: [(74, 0, 4)],
}
beat = BAR / 4
for b, name in enumerate(prog):
    t0 = b * BAR
    if t0 >= DURATION: break
    notes = CH[name]
    section_gain = 1.0
    # pad everywhere
    add(pad(notes[1:4], BAR + 1.6, vel=0.9 if 26 <= b <= 31 else 0.7), t0, pan=0.0)
    if b >= 3:  # piano arpeggio
        arp = [notes[0], notes[2], notes[3], notes[4], notes[5], notes[4], notes[3], notes[2]]
        for i, n in enumerate(arp):
            v = 0.42 + 0.12 * (i == 0)
            if 20 <= b <= 25: v *= 0.75   # quieter, honest section
            add(piano(n, dur=2.5, vel=v), t0 + i * beat / 2 + rng.uniform(0, 0.008), pan=-0.3 + 0.08 * i)
    if b >= 14:  # bass
        add(piano(notes[0] - 12, dur=3.2, vel=0.55 if b < 26 else 0.7), t0, pan=0.0)
        if 14 <= b <= 19 or 26 <= b <= 31:
            add(piano(notes[0] - 12, dur=1.6, vel=0.35), t0 + 2.5 * beat, pan=0.0)
    for (n, off, ln) in melody_by_bar.get(b, []):
        add(piano(n, dur=ln * beat + 1.5, vel=0.55), t0 + off * beat, pan=0.25)
        add(bell(n + 12, vel=0.18), t0 + off * beat, pan=0.4)

# opening sparkle: bells as the sunburst blooms, and as the title appears
for i, n in enumerate([81, 85, 88, 90, 93, 90, 88, 85]):
    add(bell(n, vel=0.35), 0.7 + i * 0.22, pan=-0.6 + i * 0.17)
for i, n in enumerate([74, 78, 81, 86]):
    add(bell(n, vel=0.4), 2.4 + i * 0.12, pan=-0.2 + 0.15 * i)
# ending chime
for i, n in enumerate([74, 78, 81, 85, 88]):
    add(bell(n, dur=5, vel=0.4), 116.2 + i * 0.18, pan=-0.4 + 0.2 * i)

def whoosh(dur=0.7, up=True, gain=0.25):
    n = int(SR * dur); t = np.arange(n) / SR
    noise = rng.standard_normal(n)
    out = np.zeros(n); blk = 512
    for i in range(0, n, blk):
        u = i / n
        fc = 400 + 3800 * (u if up else 1 - u) ** 1.3
        sos = butter(2, [fc * 0.6, min(fc * 1.6, SR / 2 - 100)], 'band', fs=SR, output='sos')
        out[i:i + blk] = sosfilt(sos, noise[max(0, i - 2048):i + blk])[-len(out[i:i + blk]):]
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    return out * env * gain

for i in range(len(SEGMENTS) - 1):
    t1 = SEGMENTS[i][1]
    if t1 in HARD_CUTS: continue
    add(whoosh(0.75, up=i % 2 == 0, gain=0.22), t1 - 0.15, pan=-0.3 if i % 2 else 0.3)
for tp, d in FLASHES:
    add(whoosh(1.4, up=True, gain=0.18), tp - 0.9)
    for j, n in enumerate([86, 90, 93]):
        add(bell(n, vel=0.3), tp - 0.05 + j * 0.08, pan=0.3 * (j - 1))
# chip pops and value dings
for tc in (50.4, 51.8, 53.2, 54.8, 56.6):
    add(bell(88 + int(rng.integers(0, 3)) * 2, dur=1.5, vel=0.22), tc, pan=-0.5)
for tc, n in ((88.4, 81), (90.2, 85), (92.6, 88)):
    add(bell(n, dur=3, vel=0.3), tc, pan=0.5)

# reverb
ir_t = np.arange(int(SR * 2.6)) / SR
irL = rng.standard_normal(len(ir_t)) * np.exp(-ir_t * 2.6); irR = rng.standard_normal(len(ir_t)) * np.exp(-ir_t * 2.6)
irL /= np.sqrt((irL ** 2).sum()); irR /= np.sqrt((irR ** 2).sum())
wetL = fftconvolve(L, irL)[:N]; wetR = fftconvolve(R, irR)[:N]
outL = L * 0.8 + wetL * 0.35; outR = R * 0.8 + wetR * 0.35
t = np.arange(N) / SR
master = np.minimum(t / 1.0, 1) * np.minimum(1, (DURATION - t) / 3.0)
out = np.stack([outL, outR], 1) * master[:, None]
out /= np.abs(out).max() / 0.85
wavfile.write('build/music.wav', SR, (out * 32767).astype(np.int16))
print('ok', out.shape)
