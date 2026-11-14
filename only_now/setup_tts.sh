#!/bin/bash
# Fetch the offline Kokoro-82M TTS (weights + voices) from npm and assemble it under build/tts.
set -e
cd "$(dirname "$0")"; mkdir -p build/tts build/dl
pip install -q onnxruntime kokoro-onnx
for p in kokoro-js kokoro-q8-shards; do
  url=$(curl -s https://registry.npmjs.org/$p/latest | python3 -c "import json,sys;print(json.load(sys.stdin)['dist']['tarball'])")
  mkdir -p build/dl/$p && curl -sL "$url" | tar xz -C build/dl/$p
done
cat build/dl/kokoro-q8-shards/package/kokoro-q8.part{0,1,2,3,4,5}.bin > build/tts/kokoro-q8.onnx
echo "fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478  build/tts/kokoro-q8.onnx" | sha256sum -c
python3 -I -c "
import numpy as np, glob, os
v={os.path.basename(f)[:-4]: np.fromfile(f,dtype=np.float32).reshape(510,1,256) for f in glob.glob('build/dl/kokoro-js/package/voices/*.bin')}
np.savez('build/tts/voices.bin', **v)"
