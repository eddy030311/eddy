import json, subprocess, sys, os
take = sys.argv[1]; out = sys.argv[2]
m = json.load(open(os.path.join(take, 'meta.json')))
fr = m['frames']; t0 = fr[0][1]; tend = fr[-1][1]
cuts = sorted([c for c in m.get('cuts', []) if len(c) == 2 and c[1] > c[0]])
def vt(t):  # real time -> edited timeline (cut intervals removed)
    s = 0
    for a, b in cuts:
        if t >= b: s += b - a
        elif t > a: s += t - a
    return t - t0 - s
incut = lambda t: any(a <= t < b for a, b in cuts)
keep = [f for f in fr if not incut(f[1])]
with open(os.path.join(take, 'list.txt'), 'w') as f:
    for i, (fn, ts) in enumerate(keep):
        d = (vt(keep[i + 1][1]) - vt(ts)) if i + 1 < len(keep) else 1 / 30
        f.write(f"file 'f/{fn}'\nduration {max(d, 0.001):.5f}\n")
    f.write(f"file 'f/{keep[-1][0]}'\n")
dur = vt(tend)
# audio: keep the same intervals (audio time = real time - aStart), short fades at each splice
segs, s0 = [], t0
for a, b in cuts: segs.append((s0, a)); s0 = b
segs.append((s0, tend))
A = m['aStart']; parts = []; fl = []
for i, (a, b) in enumerate(segs):
    sa, sb = max(0, a - A), b - A
    pad = f"adelay={int((A - a) * 1000)}:all=1," if i == 0 and A > a else ""
    parts.append(f"[1:a]atrim=start={sa:.3f}:end={sb:.3f},asetpts=PTS-STARTPTS,{pad}afade=t=in:d=0.04,afade=t=out:st={max(0,(sb-sa)+(A-a if i==0 and A>a else 0)-0.06):.3f}:d=0.06[a{i}]")
    fl.append(f"[a{i}]")
af = ";".join(parts) + f";{''.join(fl)}concat=n={len(segs)}:v=0:a=1,afade=t=in:st=0:d=0.8,afade=t=out:st={dur-2.2:.2f}:d=2.2,loudnorm=I=-15:TP=-1.5:LRA=11[a]"
cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0', '-i', os.path.join(take, 'list.txt'), '-i', os.path.join(take, 'audio.webm'),
       '-filter_complex', f"[0:v]fps=30,format=yuv420p,fade=t=out:st={dur-0.6:.2f}:d=0.6[v];{af}",
       '-map', '[v]', '-map', '[a]', '-t', f"{dur:.2f}", '-c:v', 'libx264', '-preset', 'slow', '-crf', '18', '-tune', 'animation', '-pix_fmt', 'yuv420p',
       '-c:a', 'aac', '-ar', '48000', '-b:a', '192k', '-movflags', '+faststart', out]
subprocess.run(cmd, check=True)
print('ok', out, f"{dur:.1f}s", 'cuts', [round(b - a, 2) for a, b in cuts])
