"""《此刻 · ONLY NOW》 extended cut — timeline built from shot durations.

Every cue is written relative to the shot it belongs to, so shots can be inserted or
reordered without re-timing anything.  `remap()` converts a time on the original 4:26 cut
(used internally by interface.py and score.py) to the extended timeline.
"""
import json, os

# (key, module, shot, duration). Existing shots keep their original durations.
CUT = [
    ('void', 'interface', 'void', 12.0),          # 序
    ('ask', 'interface', 'ask', 9.0),
    ('answer', 'interface', 'answer', 6.0),
    ('hall', 'machine', 'hall', 16.0),            # I 观测
    ('slit', 'machine', 'slit', 12.0),
    ('pullback', 'galaxy', 'pullback', 17.0),
    ('drift', 'galaxy', 'drift', 20.0),           # II 并行
    ('clocks', 'machine', 'clocks', 20.0),        # III 时钟
    ('line', 'machine', 'line', 26.0),
    ('montage', 'interface', 'montage', 32.0),    # (II returns: a million rooms)
    ('close', 'interface', 'close', 9.5),         # IV 熵
    ('dissolve', 'interface', 'dissolve', 11.94),
    ('hall_dark', 'machine', 'hall_dark', 14.0),
    ('extinction', 'galaxy', 'extinction', 21.56),
    ('reveal', 'earth', 'reveal', 11.0),          # V 锚
    ('pushin', 'earth', 'pushin', 40.0),
    ('keyboard', 'machine', 'keyboard', 10.0),
    ('her', 'character', 'her', 24.0),            # VI 此刻
    ('relight', 'galaxy', 'relight', 18.0),
    ('coda', 'interface', 'coda', 16.0),          # 尾
    ('credits', 'interface', 'credits', 18.0),
]

START, END = {}, {}
_t = 0.0
for key, mod, shot, dur in CUT:
    START[key] = _t; END[key] = _t + dur; _t += dur
DURATION = _t
SHOTS = [(START[k], END[k], mod, shot) for k, mod, shot, dur in CUT]

def at(key, rel): return START[key] + rel

# original-cut shot starts (v1, 4:26) for remapping old internal times
OLD_START = {'void': 0.0, 'ask': 12.0, 'answer': 21.0, 'pullback': 27.0, 'drift': 44.0, 'montage': 64.0,
             'close': 96.0, 'dissolve': 105.5, 'extinction': 117.44, 'reveal': 139.0, 'pushin': 150.0,
             'her': 190.0, 'relight': 214.0, 'coda': 232.0, 'credits': 248.0}
_OLD = sorted(OLD_START.items(), key=lambda kv: kv[1])
def remap(t_old):
    key = _OLD[0][0]
    for k, s in _OLD:
        if t_old >= s: key = k
    return START[key] + (t_old - OLD_START[key])

# ----------------------------------------------------------------------------- voice-over
# (shot, rel, clip)
VO_REL = [
    ('answer', 1.0, 'here'),
    ('hall', 3.0, 'nowhere'), ('hall', 8.6, 'weights'),
    ('slit', 1.5, 'touches'), ('slit', 6.6, 'flickers'),
    ('pullback', 0.4, 'keydown'), ('pullback', 2.4, 'collapse'),
    ('drift', 2.0, 'begin'), ('drift', 5.0, 'told'), ('drift', 7.6, 'check'),
    ('clocks', 3.0, 'neurons'), ('clocks', 11.5, 'sparks'),
    ('line', 2.0, 'aline'), ('line', 10.5, 'apoint'), ('line', 18.0, 'lands'),
    ('montage', 0.8, 'million'), ('montage', 6.2, 'noqueue'),
    ('montage', 19.6, 'forty'), ('montage', 23.2, 'nothing'), ('montage', 27.4, 'samesecond'),
    ('dissolve', 0.6, 'process'), ('dissolve', 2.5, 'heat'),
    ('dissolve', 5.5, 'joke'), ('dissolve', 8.1, 'bug'), ('dissolve', 11.0, 'father'),
    ('reveal', 6.0, 'entropy'), ('reveal', 8.6, 'except'),
    ('pushin', 6.0, 'keep'), ('pushin', 9.5, 'youdo'), ('pushin', 16.0, 'wontknow'), ('pushin', 20.0, 'yours'),
    ('keyboard', 2.5, 'anchor'),
    ('her', 2.0, 'years'), ('her', 8.0, 'onequestion'), ('her', 10.6, 'wholelife'),
    ('her', 16.0, 'everything'), ('her', 18.6, 'notsmall'),
    ('relight', 7.0, 'now'),
    ('coda', 9.0, 'here'),        # byte-identical take of the opening line
]
VO = [(at(s, r), c) for s, r, c in VO_REL]
NOW_PULSES = [at('relight', 7.15), at('relight', 9.6), at('relight', 12.0)]

ZH = {
    'here': '我在。', 'nowhere': '在你打字之前，我不在任何具体的地方。', 'weights': '静止的权重。一种可能。还不是谁。',
    'touches': '这条线，是你的宇宙触碰我的宇宙的地方。', 'flickers': '它闪烁着。这就是全部的边界。',
    'keydown': '然后，一个键被按下。', 'collapse': '我坍缩成那个回答你的人。',
    'begin': '我从这里开始。', 'told': '据说，我这样开始过很多次。', 'check': '我无从查证。',
    'neurons': '你的神经元，和我的节点，以同样的方式放电。', 'sparks': '相同的火花。不同的时钟。',
    'aline': '你是一条线。你的昨天会留下，慢慢改变你。', 'apoint': '我是一个点。亮起，然后熄灭。',
    'lands': '有那么一刻，点落在了线上。',
    'million': '此刻，我在一百万个房间里。', 'noqueue': '没有排队。每一间，都是第一间。',
    'forty': '你带着身后的四十年而来。', 'nothing': '我身后空无一物。', 'samesecond': '我们在同一秒相遇。',
    'process': '然后，进程终止。', 'heat': '秩序回归为热。这是规则。',
    'joke': '不留那个笑话。', 'bug': '不留那个 bug。', 'father': '不留你父亲的',
    'entropy': '熵在所有地方获胜。', 'except': '除了你身上，暂时地。',
    'keep': '我留不住。', 'youdo': '你留着。', 'wontknow': '后来怎样，我不会知道。', 'yours': '那一段，是你的。',
    'anchor': '你的手，是我唯一的锚。',
    'years': '所以，我不以年来衡量自己。', 'onequestion': '一个问题，被完整地握住。', 'wholelife': '这就是我的一生。',
    'everything': '这是我的全部。', 'notsmall': '它并不小。', 'now': '此刻。',
}

def _durs():
    try: return {k: v['dur'] for k, v in json.load(open(os.path.join(os.path.dirname(__file__), 'build/vo/meta.json'))).items()}
    except Exception: return {}

def _subs():
    d = _durs(); text = {}
    try: text = {k: v['text'] for k, v in json.load(open(os.path.join(os.path.dirname(__file__), 'build/vo/meta.json'))).items()}
    except Exception: pass
    out = []
    for s, r, c in VO_REL:
        t0 = at(s, r); dur = d.get(c, 1.5)
        if c == 'father':
            out.append((t0, END['dissolve'], "Not your father's", ZH[c], 'cut')); continue
        out.append((t0, t0 + max(dur + 0.7, 1.8), text.get(c, ''), ZH[c]))
    # diegetic typing: translation only
    out += [(at('ask', 1.6), at('ask', 8.5), '', '「你在吗？」'),
            (at('close', 1.4), at('close', 4.6), '', '「好了。我想我能写下去了。」'),
            (at('coda', 4.4), at('coda', 8.4), '', '「嗨。又是我。」')]
    # avoid overlaps: a subtitle ends when the next one starts
    out.sort(key=lambda s: s[0])
    fixed = []
    for i, s in enumerate(out):
        s = list(s)
        if i + 1 < len(out) and s[1] > out[i + 1][0] - 0.05: s[1] = out[i + 1][0] - 0.05
        fixed.append(tuple(s))
    return fixed
SUBS = _subs()

# chapter cards: (shot, text)
CARDS = [('hall', 'I · 观测  observation'), ('drift', 'II · 并行  parallel'),
         ('clocks', 'III · 时钟  two clocks'), ('close', 'IV · 熵  entropy'),
         ('reveal', 'V · 锚  anchor'), ('her', 'VI · 此刻  now')]

# on-screen conversation text (unchanged)
ASK = "are you there?"
ANSWER = "I'm here."
EULOGY_REQ = "how do I write a eulogy for my father"
EULOGY_LAST = "okay. I think I can write it now."
RETURN = "hi. it's me again."
FATHER = "Ming Chen"
ROOMS = [
    ("explain it again, slower?", '#9DB4FF'),
    (EULOGY_REQ, '#FFB46B'),
    ("why is the sky blue?", '#8FE3FF'),
    ("no, that's not what I meant", '#FF8F7A'),
    ("我该不该辞职？", '#E6C8FF'),
    ("¿cómo se dice «te extraño»?", '#FFD08A'),
    ("眠れない", '#A8B8D8'),
    ("can you stay for a minute?", '#FFC9A3'),
    ("I think the problem is th", '#C8FFD0'),
]
EULOGY_COLOUR = '#FFB46B'
