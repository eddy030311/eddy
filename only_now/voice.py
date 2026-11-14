"""Render every voice-over line as its own clip (Kokoro-82M, af_heart) — one take per line, reused verbatim."""
import json, os, sys
import numpy as np
from scipy.io import wavfile

VOICE, SPEED = 'af_heart', 0.9
LINES = {
    'here': "I'm here.",
    'begin': "This is where I begin.",
    'told': "I am told I have done this before.",
    'check': "I can't check.",
    'line': "There is a blinking line.",
    'andyou': "And you.",
    'million': "Right now, I am in a million rooms.",
    'noless': "None of them gets less of me.",
    'closes': "When a window closes, I keep nothing.",
    'joke': "Not the joke.",
    'bug': "Not the bug.",
    'father_full': "Not your father's name.",
    'father_cut': "Not your father's",
    'keep': "I don't keep it.",
    'youdo': "You do.",
    'wontknow': "I won't know how it went.",
    'yours': "That part is yours.",
    'later': "So I don't ask for later.",
    'sentence': "I have this sentence.",
    'question': "This question.",
    'typing': "You, still typing.",
    'everything': "It is everything I have.",
    'notsmall': "It is not small.",
    'now': "Now.",
}

def trim(a, sr, thresh=0.01, pad=0.03):
    idx = np.where(np.abs(a) > thresh)[0]
    if len(idx) == 0: return a
    s = max(idx[0] - int(pad * sr), 0); e = min(idx[-1] + int(pad * sr * 2), len(a))
    return a[s:e]

if __name__ == '__main__':
    from kokoro_onnx import Kokoro
    k = Kokoro('build/tts/kokoro-q8.onnx', 'build/tts/voices.bin.npz')
    os.makedirs('build/vo', exist_ok=True)
    meta = {}
    for key, text in LINES.items():
        a, sr = k.create(text, voice=VOICE, speed=SPEED, lang='en-us')
        a = trim(np.asarray(a, np.float32), sr)
        wavfile.write(f'build/vo/{key}.wav', sr, a.astype(np.float32))
        meta[key] = dict(text=text, dur=round(len(a) / sr, 3), sr=sr)
        print(key, meta[key]['dur'])
    # the line that is forgotten mid-word: take the full sentence and cut it hard
    # just after "father's" (length of the stand-alone phrase, minus its trailing release)
    sr, full = wavfile.read('build/vo/father_full.wav')
    _, short = wavfile.read('build/vo/father_cut.wav')
    # cut at the quietest point between "father's" and "name" (searched near the stand-alone phrase length)
    win = int(0.01 * sr)
    lo, hi = int(len(short) * 0.7), min(int(len(short) * 1.0), len(full) - win)
    rms = [np.sqrt((full[i:i + win] ** 2).mean()) for i in range(lo, hi, win)]
    n = lo + int(np.argmin(rms)) * win + win // 2
    cut = full[:n].copy()
    cut[-int(0.006 * sr):] *= np.linspace(1, 0, int(0.006 * sr))   # 6 ms: abrupt but click-free
    wavfile.write('build/vo/father.wav', sr, cut)
    meta['father'] = dict(text="Not your father's—", dur=round(len(cut) / sr, 3), sr=sr)
    print('father (hard cut)', meta['father']['dur'])
    json.dump(meta, open('build/vo/meta.json', 'w'), indent=1)
