// Runs inside the game page. Plays a scripted ~55s promo using real gameplay, with caption overlays.
window.__directMasque = async function (opts) {
  const W = ms => new Promise(r => setTimeout(r, ms));
  const log = window.__plog = [];
  const T0 = performance.now(), mark = n => log.push([n, Math.round(performance.now() - T0)]);
  const cuts = window.__cuts = [], cutA = () => cuts.push([Date.now() / 1000]), cutB = () => cuts[cuts.length - 1].push(Date.now() / 1000);
  /* ---------- overlay ---------- */
  let ov = document.getElementById('pv');
  if (!ov) {
    const st = document.createElement('style');
    st.textContent = `
#pv{position:fixed;inset:0;pointer-events:none;z-index:99999;font-family:"Noto Serif SC",serif}
#pv .blk{position:absolute;inset:0;background:#000;opacity:0;transition:opacity .7s ease}
#pv .blk.on{opacity:1}
#pv .lt{position:absolute;left:5.5%;bottom:13%;opacity:0;transform:translateX(-24px);transition:opacity .55s ease,transform .7s cubic-bezier(.2,.8,.2,1)}
#pv .lt.top{bottom:auto;top:7%;left:50%;transform:translate(-50%,-14px);text-align:center}
#pv .lt.on{opacity:1;transform:none}#pv .lt.top.on{transform:translate(-50%,0)}
#pv .lt i{display:block;font:italic 15px "UnifrakturMaguntia",serif;letter-spacing:.14em;color:#c9a45c;margin-bottom:6px;text-shadow:0 2px 6px #000}
#pv .lt b{display:block;font-weight:900;font-size:46px;letter-spacing:.22em;color:#f4ece0;text-shadow:0 3px 0 #000,0 0 24px rgba(0,0,0,.9)}
#pv .lt s{display:block;text-decoration:none;font-size:17px;letter-spacing:.18em;color:#e8c8d0;margin-top:8px;text-shadow:0 2px 4px #000}
#pv .lt:before{content:"";position:absolute;left:-18px;top:6px;bottom:6px;width:3px;background:linear-gradient(#ff4fa8,#c9a45c)}
#pv .lt.top:before{display:none}
#pv .cc{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;opacity:0;transition:opacity .8s ease}
#pv .cc.on{opacity:1}
#pv .cc b{font-weight:900;font-size:50px;letter-spacing:.3em;color:#f4ece0}
#pv .cc s{text-decoration:none;font-size:22px;letter-spacing:.5em;color:#c9a45c;margin-top:22px}
#pv .cc i{font:italic 17px "UnifrakturMaguntia",serif;letter-spacing:.16em;color:#9a8aa0;margin-top:18px}
#pv .cc .logo{font:96px "UnifrakturMaguntia",serif;letter-spacing:.02em;color:#f4ece0;text-shadow:0 0 40px rgba(201,164,92,.35)}
#pv .cc .eb{font:italic 18px "UnifrakturMaguntia",serif;letter-spacing:.2em;color:#c9a45c;margin-bottom:6px}
#pv .cc small{font-size:14px;letter-spacing:.3em;color:#8d8497;margin-top:26px}
#pv .bars:before,#pv .bars:after{content:"";position:absolute;left:0;right:0;height:0;background:#000;transition:height .8s ease}
#modal.pvR #panel{transform:translateX(30%);max-width:52vw}
#pv .bars:before{top:0}#pv .bars:after{bottom:0}#pv .bars.on:before,#pv .bars.on:after{height:7%}`;
    document.head.appendChild(st);
    ov = document.createElement('div'); ov.id = 'pv';
    ov.innerHTML = '<div class="bars"></div><div class="blk on"></div><div class="lt"><i></i><b></b><s></s></div><div class="cc"></div>';
    ov.setAttribute('data-noi18n', ''); document.body.appendChild(ov);
  }
  const blk = ov.querySelector('.blk'), lt = ov.querySelector('.lt'), cc = ov.querySelector('.cc'), bars = ov.querySelector('.bars');
  const black = on => blk.classList.toggle('on', on);
  const cap = async (fr, zh, sub, top) => { lt.classList.remove('on'); await W(250); lt.classList.toggle('top', !!top); lt.querySelector('i').textContent = fr; lt.querySelector('b').textContent = zh; lt.querySelector('s').textContent = sub || ''; lt.classList.add('on') };
  const capOff = () => lt.classList.remove('on');
  const card = (html) => { cc.innerHTML = html; cc.classList.add('on') };
  const cardOff = () => cc.classList.remove('on');
  /* ---------- autopilot: moves, attacks on the beat, keeps the player alive ---------- */
  let pilot = { on: false, dodge: false, hold: 0, modal: false, fill: false }, lastB = -1;
  const tick = setInterval(() => {
    if (!pilot.modal && G.state === 'play' && !$('modal').hidden) closeModal();
    if (!pilot.on || G.state !== 'play' || G.paused) return;
    if (pilot.fill && !G.ots && G.en.filter(o => !o.dead).length < 3) { const k = pick(['mari', 'mask', 'rose', 'crow', 'knight']), sx = rnd(40, RW - 40), sy = rnd(TOP + 30, TOP + 90); const o = makeEnemy(k, sx, sy); G.cleared = false }
    P.hp = P.maxHp; P.inv = Math.max(P.inv, .05);
    const live = G.en.filter(e => !e.dead && !(e.intro > 0));
    let e = null, bd = 1e9; for (const o of live) { const d = (o.x - P.x) ** 2 + (o.y - P.y) ** 2; if (d < bd) { bd = d; e = o } }
    if (e) {
      const dx = e.x - P.x, dy = e.y - P.y, d = Math.hypot(dx, dy) || 1, want = e.boss ? 70 : 34, s = d > want + 10 ? 1 : d < want - 10 ? -.8 : 0;
      IN.sx = dx / d * s + (-dy / d) * .55; IN.sy = dy / d * s + (dx / d) * .55;
    } else if (pilot.goal) { const dx = pilot.goal.x - P.x, dy = pilot.goal.y - P.y, d = Math.hypot(dx, dy) || 1; IN.sx = d > 6 ? dx / d * .7 : 0; IN.sy = d > 6 ? dy / d * .7 : 0 }
    else IN.sx = IN.sy = 0;
    // press exactly on the beat (the judgement and the ring both use beatNow)
    const b = beatNow(), bi = Math.round(b), f = b - Math.floor(b);
    if (e && bi !== lastB && (f > .97 || f < .03)) {
      lastB = bi; const near = Math.sqrt(bd) < 46;
      if (near) { IN.melee = true; setTimeout(() => IN.melee = false, 70) } else { IN.ranged = true; setTimeout(() => IN.ranged = false, 70) }
    }
  }, 8);
  const onBeat = async (lead = 0) => { for (let i = 0; i < 200; i++) { const b = beatNow(), f = b - Math.floor(b); if (f > 1 - lead - .02 && f < 1 - lead + .02) return; await W(4) } };
  try {
    const waitCut = async (ms) => { for (let i = 0; i < 80 && !CUT.on; i++) await W(50); await W(ms); if (CUT.on) endCut() };
    const clearCard = async () => { await W(200); const ac = $('actCard'); if (!ac.hidden) ac.click(); await W(650); if (!$('modal').hidden) closeModal() };
    const begin = async (world, idx) => { pilot.on = false; pilot.fill = false; if (G.state === 'play') { G.state = 'title' } closeModal(); WORLD = world; G.charIdx = idx; startRun(idx); };
    const bossTalk = async (act, ms) => { pilot.on = false; G.en.length = 0; G.eb.length = 0; G.act = act; enterRoom('boss'); if (G.ots) { G.ots.ph = 'swoop'; G.ots.t = 3.25; G.ots.lb = 1 }
      for (let i = 0; i < 160 && !CUT.on; i++) { await W(50); if (G.ots && G.ots.ph === 'swoop' && G.ots.t < 3.2) G.ots.t = 3.25 } await W(ms); if (CUT.on) endCut(); if (G.ots) otsEnd(); await W(200); pilot.on = true };
    /* ============ 1. title → the fork ============ */
    mark('title'); toTitle(); await W(300); black(false); bars.classList.add('on'); await W(2200);
    const wb = $('w2btn'); if (wb) wb.focus(); await W(900); w3Branch(); await W(600);
    await cap('Sous le masque', '假面之下', '三个故事 · 三种和解', true); await W(2800); capOff();
    /* ============ 2. 祥子的夜 ============ */
    mark('w2'); black(true); await W(700); cutA(); closeModal(); await begin(2, 4); await W(400); cutB(); black(false);
    await cap('La Nuit de Sakiko', '祥子的夜', '陪伴四位同伴，走出各自的心结'); await waitCut(3600); await clearCard(); capOff();
    pilot.on = true; pilot.fill = true; await W(2600);
    await cap('Sous le masque', 'Boss 是戴着假面的她们自己', '三个阶段 · 每个阶段都有新的对话'); pilot.fill = false; await bossTalk(2, 4800); await W(1600); capOff();
    /* ============ 3. 星之约 ============ */
    mark('w3'); black(true); await W(700); cutA(); await begin(3, 4); await W(400); cutB(); black(false);
    await cap("Promesse d'étoiles", '星之约', '初华与祥子的日常 · 恋爱感点到为止'); await waitCut(3800); await clearCard(); capOff();
    pilot.on = true; pilot.fill = true; await W(2200);
    black(true); await W(600); cutA(); G.en.length = 0; G.eb.length = 0; G.act = 3; G.bg = buildBg(); enterRoom('combat'); if (G.ots) otsEnd(); await W(300); cutB(); black(false);
    await cap('Le Festival', '夏日祭 · 雨天的商店街 · 星之岛', '五个平凡又闪亮的地方'); await W(2400);
    pilot.fill = false; await bossTalk(5, 4400); await W(1400); capOff();
    /* ============ 4. 镜中的睦 ============ */
    mark('w4'); black(true); await W(700); cutA(); await begin(4, 1); await W(400); cutB(); black(false);
    await cap('Le Miroir', '镜中的睦', 'Mortis 走进一颗心，和每一个自己和解'); await waitCut(3800); await clearCard(); capOff();
    pilot.on = true; pilot.fill = true; await W(1600);
    await cap('Fever · 镜中人偶', '召唤镜中的 Mortis 并肩作战', 'Fever 期间的伤害由她们平分承担'); await onBeat(); G.fever = fCap(); await W(3800);
    pilot.fill = false; await cap('La Phrase de Trop', '每一个 Boss，都是另一个她', '「我从来没觉得乐队开心过。」'); await bossTalk(3, 4600); await W(1400); capOff();
    /* ============ 5. end card ============ */
    mark('end'); black(true); await W(800); pilot.on = false; G.state = 'title'; closeModal(); toTitle();
    card('<div class="eb">Ave Mujica · 假面回廊</div><div class="logo" style="font-size:78px">Sous le masque</div><s>祥子的夜 · 星之约 · 镜中的睦</s><small>完成任意一个结局后，在标题页「假面之下」开启</small>');
    await W(4400); cardOff(); await W(900);
    mark('done');
  } finally { clearInterval(tick); pilot.on = false }
  return log;
};
