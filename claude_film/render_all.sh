#!/bin/bash
# Render 《Claude》 in parallel chunks, then mux with the score.
set -e
cd "$(dirname "$0")"
python3 prep_cutouts.py
python3 film.py flows
python3 music.py
TOTAL=2880; J=${JOBS:-4}; STEP=$(( (TOTAL + J - 1) / J ))
rm -f build/seg_*.mp4 build/segs.txt
for i in $(seq 0 $((J-1))); do
  s=$((i*STEP)); e=$(( s+STEP < TOTAL ? s+STEP : TOTAL ))
  python3 film.py frames $s $e build/seg_$i.mp4 > build/seg_$i.log 2>&1 &
  echo "file 'seg_$i.mp4'" >> build/segs.txt
done
wait
ffmpeg -y -loglevel error -f concat -safe 0 -i build/segs.txt -i build/music.wav \
  -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart Claude.mp4
echo done
