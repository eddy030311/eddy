import numpy as np, json, sys, parselmouth
from parselmouth.praat import call
from scipy.io import wavfile
from scipy.signal import stft
import voice as V, timeline as TL

def metrics(y, sr):
    y = np.asarray(y, np.float64)
    snd = parselmouth.Sound(y, sr)
    p = snd.to_pitch_ac(time_step=0.005, pitch_floor=75, pitch_ceiling=600)
    f = p.selected_array['frequency']; v = f > 0
    st = 12 * np.log2(f[v] / np.median(f[v])) if v.sum() > 3 else np.zeros(1)
    jumps = int(np.sum(np.abs(np.diff(st)) > 5))                  # octave-ish glitches between adjacent 5 ms frames
    pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jit = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3) * 100
    shim = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6) * 100
    hnr = call(snd.to_harmonicity_cc(), "Get mean", 0, 0)
    fr, t, Z = stft(y, sr, nperseg=1024, noverlap=768); P = np.abs(Z) ** 2; e = P.sum(0); act = e > e.max() * 1e-3
    hum = 10 * np.log10(P[fr < 70][:, act].sum() / (P[:, act].sum() + 1e-12) + 1e-12)
    B = P[(fr > 2000) & (fr < 8000)]
    flat = float(np.median((np.exp(np.mean(np.log(B + 1e-14), 0)) / (B.mean(0) + 1e-14))[act]))
    return dict(f0=float(np.median(f[v])) if v.any() else 0, jumps=jumps, jit=jit, shim=shim, hnr=hnr, hum=hum, flat=flat)

if __name__ == '__main__':
    from kokoro_onnx import Kokoro
    k = Kokoro('build/tts/kokoro-q8.onnx', 'build/tts/voices.bin.npz')
    keys = sorted(set(c for _, c in TL.VO) - {'father'})
    rows = []
    for key in keys:
        sr, y = wavfile.read(f'build/vo/{key}.wav')
        m1 = metrics(y, sr)
        text = V.LINES[key]
        a, sr2 = k.create(text, voice='af_heart', speed=0.88, lang='en-us')
        m0 = metrics(np.asarray(a), sr2)
        rows.append((key, V.MOOD.get(key, 'calm'), m0, m1))
    print(f"{'key':13s} {'mood':8s} | RAW f0 jumps jit% hnr | FINAL f0 jumps jit% shim% hnr hum flat")
    for key, mood, a, b in rows:
        print(f"{key:13s} {mood:8s} | {a['f0']:4.0f} {a['jumps']:3d} {a['jit']:4.2f} {a['hnr']:4.1f} | {b['f0']:4.0f} {b['jumps']:3d} {b['jit']:4.2f} {b['shim']:5.2f} {b['hnr']:4.1f} {b['hum']:6.1f} {b['flat']:.3f}")
    json.dump([(r[0], r[1], r[2], r[3]) for r in rows], open('build/vo_audit.json', 'w'))
