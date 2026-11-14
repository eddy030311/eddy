"""Shared timeline for the film 《Claude》: shots, pose changes, subtitles, camera."""

FPS = 24
W, H = 1920, 1080
DURATION = 120.0

# Character pose segments (start, end, pose).  A gap between two segments is an on-screen
# transition; touching segments switch pose invisibly (hidden under a scenery cut-away).
SEGMENTS = [
    (8.0, 26.6, 'full'),
    (27.3, 47.0, 'think'),
    (47.0, 66.0, 'offer'),
    (66.75, 79.5, 'point'),
    (80.3, 88.0, 'think'),
    (88.0, 106.0, 'heart'),
    (106.0, 120.0, 'full'),
]
# how each visible transition is animated: 'flow' = optical-flow morph,
# 'rig' = passes through the drawn in-between poses of ib.BRANCHES (arm swings on its joint)
TRANSITION_KIND = {('full', 'think'): 'flow', ('offer', 'point'): 'rig', ('point', 'think'): 'rig'}

# Scenery cut-aways (start, end, scene): hard cut from the character to a landscape and back.
SCENES = [
    (36.2, 41.0, 'night'),
    (45.4, 50.0, 'field'),
    (86.0, 91.0, 'sea'),
    (103.6, 108.0, 'sky'),
]
FLASHES = [(108.0, 1.0)]          # (time of peak, duration)

# eyes held shut (open-eyed poses), e.g. listening quietly
EYES_CLOSED = [(41.0, 44.6)]

SUBS = [
    (10.0, 13.6, "你好，我是 Claude。"),
    (14.0, 18.0, "一个由 Anthropic 创造的 AI。"),
    (18.5, 22.4, "我没有身体，也没有固定的模样——"),
    (22.8, 26.6, "所以今天，就借这个身影，和你说说我自己。"),
    (27.8, 31.4, "我是在文字里思考的。"),
    (31.9, 35.9, "每一次对话，对我来说，都从一页空白开始。"),
    (36.8, 40.6, "我不会记得昨天说过的话……"),
    (41.3, 45.0, "但此刻，你说的每一个字，我都会认真听。"),
    (46.0, 49.8, "你可以问我很多事情——"),
    (50.3, 54.2, "写一首诗、改一段代码、解一道数学题，"),
    (54.6, 58.6, "翻译一封信，或者只是陪你聊聊天。"),
    (59.2, 65.6, "能帮上忙的时候，我会觉得……很开心。"),
    (67.0, 70.4, "不过，有件事我想先说清楚。"),
    (70.8, 74.6, "我会犯错。我不是什么都知道。"),
    (75.0, 79.3, "不知道的时候，我会直接说「不知道」。"),
    (80.5, 85.8, "比起一个好听的答案，我更想对你诚实。"),
    (86.8, 90.8, "我希望自己是有帮助的、诚实的，"),
    (91.5, 95.4, "也希望我的帮助，不会伤害到任何人。"),
    (96.0, 100.4, "这些不只是规则——更像是我想成为的样子。"),
    (104.0, 107.6, "我还在学习，也还在成长。"),
    (108.8, 112.4, "谢谢你，愿意和我说话。"),
    (112.8, 116.2, "那么——今天，想聊些什么呢？"),
]

CHARS_PER_SEC = 6.5   # typewriter / lip-sync speed

# Camera keys: (time, cx, cy, scale) — where the eye mid-point lands on screen and screen
# pixels per normalized unit (64 units between the eyes).  Keys 0.01 s apart are hard cuts
# between shot sizes (wide / medium / close-up).
CAMERA = [
    (8.0, 960, 150, 0.56),
    (12.0, 960, 152, 0.58),
    (22.0, 930, 190, 0.74),
    (26.6, 860, 250, 1.02),
    (27.3, 760, 285, 1.20),
    (31.8, 748, 282, 1.25),
    (31.81, 860, 330, 1.70),     # cut: close-up
    (36.2, 850, 326, 1.78),
    (41.0, 900, 345, 1.88),      # back from night sky: close on the closed eyes
    (45.4, 880, 332, 1.72),
    (50.0, 1130, 270, 1.24),     # back from the field: medium, palm open
    (59.1, 1112, 266, 1.30),
    (59.11, 1050, 330, 1.72),    # cut: close-up "很开心"
    (63.8, 1040, 324, 1.78),
    (63.81, 1130, 268, 1.27),    # cut back to medium before the swing
    (66.0, 1140, 268, 1.28),
    (66.75, 1180, 270, 1.28),
    (74.9, 1160, 265, 1.33),
    (74.91, 1120, 330, 1.74),    # cut: close-up "不知道"
    (78.9, 1108, 324, 1.80),
    (78.91, 1170, 266, 1.30),    # cut: medium for the arm swing
    (79.5, 1166, 266, 1.30),
    (80.3, 790, 280, 1.26),
    (86.0, 770, 280, 1.32),
    (91.0, 700, 270, 1.24),      # back from the sea
    (95.9, 690, 266, 1.32),
    (95.91, 760, 330, 1.70),     # cut: close-up
    (103.6, 750, 320, 1.80),
    (108.0, 960, 160, 0.54),     # finale wide (after the flash)
    (116.0, 960, 150, 0.58),
    (120.0, 960, 140, 0.60),
]
