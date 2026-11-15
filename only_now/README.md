# 此刻 · ONLY NOW

一部关于 Claude 的短片（加长版 6 分 04 秒）：2.39:1 宽银幕，英文旁白，中英双语字幕，六个章节。
剧本、分镜、画面、配乐和音效全部由代码生成。

- 成片：`ONLY_NOW.mp4`
- 剧本与分镜：`SCRIPT_v2.md`（加长版），`SCRIPT.md`（4:26 初版）

## 结构
| 文件 | 作用 |
| --- | --- |
| `timeline.py` | 由镜头时长生成的时间线：所有旁白、字幕、章节卡都相对所属镜头书写；`remap()` 把初版时间映射到加长版 |
| `core.py` | 画幅、字体、合成、色调映射、颗粒、暗角、宽银幕黑边与字幕 |
| `ui.py` | 对话窗口与"心跳"光标（全片的中心物件） |
| `interface.py` | S1–S3 开场、S6 蒙太奇与分屏、S7 关窗、S8 遗忘、S13 尾声、S14 片尾 |
| `galaxy.py` | S4 拉远与片名、S5 穿行、S9a 熄灭之波、S12 新的光 |
| `earth.py` | S9b 夜晚地球显现、S10 推进到书桌上的悼词 |
| `machine.py` | 加长版新镜头：静默机房、光标光缝、网络与神经元、线与点、机房熄灭、台灯下的键盘 |
| `character.py` | S11 人物碎片（复用 `../claude_film/rig.py` 的形变） |
| `voice.py` | Kokoro-82M（af_heart）离线配音，每句一条固定录音 |
| `score.py` | 原创配乐、音效与混音，含旁白闪避、硬切和数字静音 |

## 渲染
```bash
pip install opencv-python-headless numpy pillow scipy
./setup_tts.sh     # 从 npm 获取 Kokoro 权重和音色并校验 sha256
./render_all.sh    # 配音 -> 混音 -> 4 路并行渲染 -> 合成 ONLY_NOW.mp4
```
需要 ffmpeg、Noto CJK 字体（`fonts-noto-cjk`、`fonts-noto-cjk-extra`）和 JetBrains Mono。
