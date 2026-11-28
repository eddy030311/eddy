#!/usr/bin/env bash
# Build dist/AveMujica-Windows-x64.zip (portable Windows build) from ../../index.html and the offline assets.
# Needs: curl, unzip, zip, python3-pil, node (npm package "resedit" for the exe icon).
set -euo pipefail
cd "$(dirname "$0")"
V=33.2.1; GAME=../..; CACHE=.cache; OUT=build/AveMujica
mkdir -p "$CACHE"
Z="$CACHE/electron-v$V-win32-x64.zip"
if [ ! -f "$Z" ]; then
  curl -sSL -o "$Z" "https://github.com/electron/electron/releases/download/v$V/electron-v$V-win32-x64.zip"
  curl -sSL -o "$CACHE/SHASUMS256.txt" "https://github.com/electron/electron/releases/download/v$V/SHASUMS256.txt"
  (cd "$CACHE" && grep " \*\?electron-v$V-win32-x64.zip" SHASUMS256.txt | sha256sum -c -)
fi
rm -rf build; mkdir -p "$OUT"
unzip -q "$Z" -d "$OUT"
rm -f "$OUT/resources/default_app.asar"
mv "$OUT/electron.exe" "$OUT/AveMujica.exe"
APP="$OUT/resources/app"; mkdir -p "$APP"
cp main.js package.json updater.js preload.js "$APP/"
# same offline bundle as the Android build (local fonts, assets, vendor)
python3 "$GAME/android/prepare.py" build/bundle >/dev/null
cp -r build/bundle/assets/www "$APP/game"
python3 - <<'P'
from PIL import Image
im=Image.open('../../android/icon/icon512.png').convert('RGBA')
im.resize((256,256),Image.LANCZOS).save('build/AveMujica/resources/app/icon.png')
im.save('build/icon.ico',sizes=[(16,16),(24,24),(32,32),(48,48),(64,64),(128,128),(256,256)])
P
[ -d node_modules/resedit ] || npm install --silent --no-audit --no-fund resedit@2.0.3 >/dev/null
node seticon.js "$OUT/AveMujica.exe" build/icon.ico
mkdir -p "$GAME/dist"; rm -f "$GAME/dist/AveMujica-Windows-x64.zip"
(cd build && zip -qr -9 "../$GAME/dist/AveMujica-Windows-x64.zip" AveMujica)
echo "built $GAME/dist/AveMujica-Windows-x64.zip"
