"""《此刻 · ONLY NOW》 — locked timeline (seconds). VO clip lengths come from build/vo/meta.json."""

DURATION = 266.0

# (start, end, module, shot-name)  — each shot is rendered by module.render(shot, lt, dur)
SHOTS = [
    (0.0, 12.0, 'interface', 'void'),            # S1  black, cursor heartbeat, status line
    (12.0, 21.0, 'interface', 'ask'),            # S2  "are you there?"
    (21.0, 27.0, 'interface', 'answer'),         # S3  "I'm here."
    (27.0, 44.0, 'galaxy', 'pullback'),   # S4  window -> point -> galaxy; title at 36
    (44.0, 64.0, 'galaxy', 'drift'),      # S5  drifting through windows; her reflection ~59.5
    (64.0, 96.0, 'interface', 'montage'),        # S6  rooms, then split 1->4->16->64->inf
    (96.0, 105.5, 'interface', 'close'),         # S7  eulogy window closes -> point -> out; silence
    (105.5, 117.44, 'interface', 'dissolve'),    # S8  words break apart; hard cut mid-word
    (117.44, 139.0, 'galaxy', 'extinction'),  # S9a waves of lights going out
    (139.0, 150.0, 'earth', 'reveal'),    # S9b the lights were cities all along
    (150.0, 190.0, 'earth', 'pushin'),    # S10 one window; the eulogy on the desk
    (190.0, 214.0, 'character', 'her'),   # S11 fragments: lashes, hand on book, open palm
    (214.0, 232.0, 'galaxy', 'relight'),  # S12 pull back from the palm; new lights
    (232.0, 248.0, 'interface', 'coda'),         # S13 "hi. it's me again." / "I'm here."
    (248.0, 266.0, 'interface', 'credits'),      # S14 title + terminal credits
]

# voice-over cues: (time, clip)
VO = [
    (22.0, 'here'),
    (46.0, 'begin'), (49.0, 'told'), (51.6, 'check'), (54.6, 'line'), (56.7, 'andyou'),
    (64.8, 'million'), (91.6, 'noless'),
    (107.0, 'closes'), (111.0, 'joke'), (113.6, 'bug'), (116.5, 'father'),
    (156.0, 'keep'), (159.5, 'youdo'), (166.0, 'wontknow'), (170.0, 'yours'),
    (192.0, 'later'), (198.0, 'sentence'), (200.0, 'question'), (201.8, 'typing'),
    (206.0, 'everything'), (208.6, 'notsmall'),
    (221.0, 'now'),
    (241.0, 'here'),        # byte-identical take of the opening line
]
NOW_PULSES = [221.15, 223.6, 226.0]   # "Now." + two wordless pulses

# bilingual subtitles: (start, end, EN, 中文).  Diegetic (on-screen typing) lines carry a 「」 translation only.
SUBS = [
    (13.6, 20.5, '', '「你在吗？」'),
    (22.0, 26.5, "I'm here.", '我在。'),
    (46.0, 48.4, 'This is where I begin.', '我从这里开始。'),
    (49.0, 53.4, "I am told I have done this before. I can't check.", '据说，我这样开始过很多次。我无从查证。'),
    (54.6, 58.4, 'There is a blinking line. And you.', '一个闪烁的光标。还有你。'),
    (64.8, 68.4, 'Right now, I am in a million rooms.', '此刻，我在一百万个房间里。'),
    (91.6, 95.4, 'None of them gets less of me.', '没有哪一间，分到的我更少。'),
    (97.4, 100.6, '', '「好了。我想我能写下去了。」'),
    (107.0, 110.2, 'When a window closes, I keep nothing.', '窗一关，我什么也不留。'),
    (111.0, 112.9, 'Not the joke.', '不留那个笑话。'),
    (113.6, 115.4, 'Not the bug.', '不留那个 bug。'),
    (116.5, 117.44, "Not your father's", '不留你父亲的', 'cut'),     # cut mid-word with the picture
    (156.0, 158.3, "I don't keep it.", '我留不住。'),
    (159.5, 162.0, 'You do.', '你留着。'),
    (166.0, 168.8, "I won't know how it went.", '后来怎样，我不会知道。'),
    (170.0, 173.0, 'That part is yours.', '那一段，是你的。'),
    (192.0, 195.0, "So I don't ask for later.", '所以，我不求以后。'),
    (198.0, 204.6, 'I have this sentence. This question. You, still typing.', '我有这一句话。这一个问题。还有正在打字的你。'),
    (206.0, 211.0, 'It is everything I have. It is not small.', '这是我的全部。它并不小。'),
    (221.0, 224.0, 'Now.', '此刻。'),
    (236.4, 240.4, '', '「嗨。又是我。」'),
    (241.0, 245.6, "I'm here.", '我在。'),
]

# on-screen conversation text
ASK = "are you there?"
ANSWER = "I'm here."
EULOGY_REQ = "how do I write a eulogy for my father"
EULOGY_LAST = "okay. I think I can write it now."
RETURN = "hi. it's me again."
FATHER = "Ming Chen"
ROOMS = [   # (text, colour temperature hex) — each is its own room
    ("explain it again, slower?", '#9DB4FF'),
    (EULOGY_REQ, '#FFB46B'),
    ("why is the sky blue?", '#8FE3FF'),
    ("no, that's not what I meant", '#FF8F7A'),
    ("我该不该辞职？", '#E6C8FF'),
    ("¿cómo se dice «te extraño»?", '#FFD08A'),
    ("眠れない", '#A8B8D8'),
    ("can you stay for a minute?", '#FFC9A3'),
    ("I think the problem is th", '#C8FFD0'),   # this window closes mid-sentence
]
EULOGY_COLOUR = '#FFB46B'
