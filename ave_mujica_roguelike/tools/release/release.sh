#!/usr/bin/env bash
# Publish the current game to the `release` branch, which every installed APK / Windows build checks for updates.
#   tools/release/release.sh "1.1" "更新内容一" "更新内容二" ...   -> bumps update/version.json, then publishes
#   tools/release/release.sh --same                                -> republishes the version already in version.json
# Afterwards rebuild dist/ (android/build.sh, tools/desktop/build.sh) so fresh downloads carry the same version.
set -euo pipefail
GAME="$(cd "$(dirname "$0")/../.." && pwd)"; REPO="$(git -C "$GAME" rev-parse --show-toplevel)"; SUB="${GAME#$REPO/}"
if [ "${1:-}" != "--same" ]; then
  [ $# -ge 2 ] || { echo "usage: release.sh <version name> <note> [note...] | --same"; exit 1; }
  python3 - "$GAME/update/version.json" "$@" <<'P'
import json,sys,datetime
f=sys.argv[1];v=json.load(open(f,encoding='utf-8'))
v={'v':v['v']+1,'name':sys.argv[2],'date':datetime.date.today().isoformat(),'notes':sys.argv[3:]}
json.dump(v,open(f,'w',encoding='utf-8'),ensure_ascii=False);print('version ->',v['v'],v['name'])
# the in-game changelog (CHANGELOG in index.html) gets the same entry, newest first
import os
h=os.path.join(os.path.dirname(os.path.dirname(f)),'index.html');src=open(h,encoding='utf-8').read();key='const CHANGELOG=[\n'
if "{name:'%s'," % v['name'] not in src:
    ent=" {name:%s,date:%s,notes:%s},\n" % (json.dumps(v['name'],ensure_ascii=False).replace('"',"'"),json.dumps(v['date']).replace('"',"'"),json.dumps(v['notes'],ensure_ascii=False))
    assert key in src;src=src.replace(key,key+ent,1);open(h,'w',encoding='utf-8').write(src);print('changelog +',v['name'])
P
fi
TMP="$(mktemp -d)"; trap 'git -C "$REPO" worktree remove --force "$TMP/wt" 2>/dev/null || true; rm -rf "$TMP"' EXIT
python3 "$GAME/android/prepare.py" "$TMP/b" >/dev/null
if git -C "$REPO" fetch -q origin release 2>/dev/null; then git -C "$REPO" worktree add -q "$TMP/wt" FETCH_HEAD; git -C "$TMP/wt" checkout -q -B release
else git -C "$REPO" worktree add -q --detach "$TMP/wt"; git -C "$TMP/wt" checkout -q --orphan release; git -C "$TMP/wt" rm -rq --cached . ; find "$TMP/wt" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +; fi
U="$TMP/wt/$SUB/update"; rm -rf "$U"; mkdir -p "$U"
cp -r "$TMP/b/assets/www" "$U/www"; mv "$U/www/manifest.json" "$U/manifest.json"
cat > "$TMP/wt/README.md" <<'R'
# release
Ave Mujica 假面回廊 的正式版文件。已安装的安卓版 / Windows 版会读取 `ave_mujica_roguelike/update/manifest.json` 检查更新，并从 `update/www/` 下载有变化的文件。不要手动修改这个分支，请用 `tools/release/release.sh` 发布。
R
NAME=$(python3 -c "import json;print(json.load(open('$U/manifest.json'))['name'])")
git -C "$TMP/wt" add -A
git -C "$TMP/wt" commit -q -m "Release v$NAME" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" || echo "nothing changed"
for i in 1 2 3 4; do git -C "$TMP/wt" push -q -u origin release && break || sleep $((i*2)); done
echo "published v$NAME to release"
