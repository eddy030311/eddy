#!/bin/bash
# Render 《此刻 · ONLY NOW》: voice -> score/mix -> picture in parallel chunks -> mux.
set -e
cd "$(dirname "$0")"
[ -f build/tts/kokoro-q8.onnx ] || ./setup_tts.sh
[ -f build/think.png ] || cp ../claude_film/build/{think,point,calm,offer,heart,full}.png build/
[ -n "$SKIP_VO" ] || python3 -I voice.py
python3 score.py
FPS=24; TOTAL=$(python3 -c "import timeline;print(int(timeline.DURATION*$FPS))"); J=${JOBS:-4}
STEP=$(( (TOTAL + J - 1) / J ))
rm -f build/seg_*.mp4 build/segs.txt
for i in $(seq 0 $((J-1))); do
  s=$((i*STEP)); e=$(( s+STEP < TOTAL ? s+STEP : TOTAL ))
  python3 film.py frames $s $e build/seg_$i.mp4 > build/seg_$i.log 2>&1 &
  echo "file 'seg_$i.mp4'" >> build/segs.txt
done
wait
ffmpeg -y -loglevel error -f concat -safe 0 -i build/segs.txt -i build/mix.wav \
  -c:v copy -c:a aac -b:a 256k -shortest -movflags +faststart ONLY_NOW.mp4
echo done
