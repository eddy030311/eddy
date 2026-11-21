"""Assemble the offline web bundle and launcher icons for the APK into <build>/assets and <build>/res."""
import os, re, shutil, sys
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); GAME = os.path.dirname(HERE); B = sys.argv[1]
www = os.path.join(B, 'assets', 'www'); os.makedirs(www, exist_ok=True)
html = open(os.path.join(GAME, 'index.html'), encoding='utf-8').read()
html = re.sub(r'<link rel="preconnect"[^>]*>\n?', '', html)
html = re.sub(r'<link rel="stylesheet" href="https://fonts\.googleapis\.com[^>]*>', '<link rel="stylesheet" href="fonts/fonts.css">', html)
assert 'googleapis' not in html
open(os.path.join(www, 'index.html'), 'w', encoding='utf-8').write('<!doctype html>\n<html lang="zh-CN"><meta charset="utf-8">\n' + html)
shutil.copytree(os.path.join(GAME, 'assets'), os.path.join(www, 'assets'), dirs_exist_ok=True)
shutil.copytree(os.path.join(HERE, 'fonts'), os.path.join(www, 'fonts'), dirs_exist_ok=True)
# launcher icon: the drawn app icon (android/icon/icon512.png)
icon = Image.open(os.path.join(HERE, 'icon', 'icon512.png')).convert('RGBA')
for d, px in {'mdpi': 48, 'hdpi': 72, 'xhdpi': 96, 'xxhdpi': 144, 'xxxhdpi': 192}.items():
    os.makedirs(os.path.join(B, 'res', 'mipmap-' + d), exist_ok=True)
    icon.resize((px, px), Image.LANCZOS).save(os.path.join(B, 'res', 'mipmap-' + d, 'ic_launcher.png'))
print('bundle ready')
