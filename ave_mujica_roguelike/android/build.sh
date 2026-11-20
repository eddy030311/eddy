#!/usr/bin/env bash
# Build dist/AveMujica.apk (Android, landscape, offline) from ../index.html and ../assets.
# Needs (Ubuntu): android-sdk-platform-23 aapt apksigner dalvik-exchange zipalign, a JDK, python3-pil.
set -euo pipefail
cd "$(dirname "$0")"
JAR=/usr/lib/android-sdk/platforms/android-23/android.jar
B=build; rm -rf "$B"; mkdir -p "$B/classes"
python3 prepare.py "$B"
javac -nowarn -source 8 -target 8 -bootclasspath "$JAR" -d "$B/classes" src/com/avemujica/masque/MainActivity.java 2>&1 | grep -v "warning" || true
dalvik-exchange --dex --output="$B/classes.dex" "$B/classes"
aapt package -f -M AndroidManifest.xml -S "$B/res" -A "$B/assets" -I "$JAR" -F "$B/app.unsigned.apk"
(cd "$B" && aapt add app.unsigned.apk classes.dex >/dev/null)
zipalign -f 4 "$B/app.unsigned.apk" "$B/app.aligned.apk"
KS="${KEYSTORE:-$HOME/.ave_mujica_release.keystore}"
if [ ! -f "$KS" ]; then
  keytool -genkeypair -keystore "$KS" -alias masque -keyalg RSA -keysize 2048 -validity 10000 \
    -storepass "${KS_PASS:-masquecorridor}" -keypass "${KS_PASS:-masquecorridor}" -dname "CN=Masque Corridor" >/dev/null
fi
mkdir -p ../dist
apksigner sign --v4-signing-enabled false --ks "$KS" --ks-pass "pass:${KS_PASS:-masquecorridor}" --key-pass "pass:${KS_PASS:-masquecorridor}" --out ../dist/AveMujica.apk "$B/app.aligned.apk"
apksigner verify ../dist/AveMujica.apk && echo "built ../dist/AveMujica.apk"
