"""Refresh android/fonts so the offline APK has every glyph the game text uses.
Downloads Google Fonts CSS (Latin display fonts + sliced Noto Serif SC) and keeps only the
Noto slices whose unicode-range intersects the characters in ../index.html."""
import os, re, subprocess, urllib.parse
HERE = os.path.dirname(os.path.abspath(__file__)); FD = os.path.join(HERE, 'fonts')
UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'
html = open(os.path.join(HERE, '..', 'index.html'), encoding='utf-8').read()
need = {ord(c) for c in html if ord(c) > 127} | set(range(32, 127))
def get_css(q): return subprocess.run(['curl', '-sS', '-A', UA, 'https://fonts.googleapis.com/css2?' + q + '&display=swap'], capture_output=True, text=True, check=True).stdout
def cps_of(block):
    m = re.search(r'unicode-range:\s*([^;]+);', block)
    if not m: return None
    out = set()
    for part in m.group(1).split(','):
        part = part.strip().replace('U+', '')
        if '-' in part: a, b = part.split('-'); out.update(range(int(a, 16), int(b, 16) + 1))
        elif '?' in part: out.update(range(int(part.replace('?', '0'), 16), int(part.replace('?', 'F'), 16) + 1))
        else: out.add(int(part, 16))
    return out
for f in os.listdir(FD):
    if f.endswith('.woff2'): os.remove(os.path.join(FD, f))
css_out = []; n = 0; seen = {}
for q in ['family=UnifrakturMaguntia', 'family=Cinzel:wght@400;600;700;900', 'family=Noto+Serif+SC:wght@400;700;900']:
    for block in re.findall(r'@font-face\s*\{[^}]*\}', get_css(q)):
        cps = cps_of(block)
        if cps is not None and not (cps & need): continue
        url = re.search(r'url\((https://[^)]+)\)', block).group(1)
        if url not in seen:
            seen[url] = fn = f'f{n}.woff2'; n += 1
            subprocess.run(['curl', '-sS', '-o', os.path.join(FD, fn), url], check=True)
        fn = seen[url]
        css_out.append(block.replace(url, fn))
open(os.path.join(FD, 'fonts.css'), 'w').write('\n'.join(css_out) + '\n')
print('font files:', n)
