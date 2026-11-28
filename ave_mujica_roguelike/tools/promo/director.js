// Runs inside the game page. Plays a scripted ~55s promo using real gameplay, with caption overlays.
window.__direct = async function (opts) {
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
    /* ============ 1. title ============ */
    mark('title'); G.rc = 0; toTitle(); await W(300); black(false); bars.classList.add('on');
    await W(3800);
    /* ============ 2. five dolls card ============ */
    mark('card1'); black(true); await W(700);
    card('<b>五具人偶，五张假面。</b><s>悲伤 · 死亡 · 恐惧 · 爱 · 忘却</s><i>Cinq poupées, cinq masques.</i>');
    // set up the run behind the card
    G.charIdx = 0; startRun(0); await W(200); if (CUT.on) endCut(); await W(250);
    const ac = $('actCard'); if (!ac.hidden) ac.click(); await W(600); if (!$('modal').hidden) closeModal();
    for (const id of ['mortis', 'timoris']) addAlly(id); placeAllies();
    pilot.on = true; pilot.fill = true; await W(700);
    /* ============ 3. rhythm combat ============ */
    mark('combat'); cardOff(); await W(300); black(false);
    await cap('Le rythme est une arme', '踩着节拍战斗', '按在拍点上 · PERFECT 伤害 ×1.5'); await W(4800);
    /* ============ 4. fever ============ */
    mark('fever'); await cap('Fever', '攒满 Fever · 自动发动专属技能', '五名人偶 · 五种独奏'); await W(900);
    await onBeat(); G.fever = fCap(); await W(4300);
    /* ============ 5. perfect dodge ============ */
    mark('dodge'); await cap('Esquive parfaite', '踩拍闪避', '在拍点上躲开攻击 · 慢动作反击');
    for (let k = 0; k < 2; k++) {
      await onBeat(.42); const n = 9; for (let i = 0; i < n; i++) { const a = i / n * TAU; eshoot(P.x + Math.cos(a) * 62, P.y - 8 + Math.sin(a) * 62, a + Math.PI, 150, { dmg: 1 }) }
      await onBeat(); IN.dashQ = true; await W(1500);
    }
    /* ============ 6. boss ============ */
    mark('boss'); capOff(); black(true); await W(750); cutA(); feverEnd(); G.fever = 0;
    pilot.fill = false; G.en.length = 0; G.eb.length = 0; G.act = 3; enterRoom('boss'); if (G.ots) { G.ots.ph = 'swoop'; G.ots.t = 0; G.ots.lb = 1 } await W(600); cutB(); black(false);
    for (let i = 0; i < 90 && (G.ots || CUT.on); i++) { await W(100); if (CUT.on) endCut() }
    await cap('Le boss danse avec la musique', 'Boss 随节拍出招', '每一招都落在拍点上'); await W(3600);
    /* ============ 7. the cage ============ */
    mark('cage'); pilot.goal = null; pilot.on = false; IN.sx = IN.sy = 0; IN.melee = IN.ranged = false; P.inv = 9; P.hp = P.maxHp; G.eb.length = 0; await cap('Libère-la', '击败 Boss · 救出被囚禁的同伴', '她就在舞台一角的金色鸟笼里');
    if (G.cage) { G.cage.y = Math.max(G.cage.y, TOP + 70); P.x = G.cage.x + 46; P.y = G.cage.y + 6 }
    await W(2600);
    if (G.cage) { G.cage.broken = 1; burst(G.cage.x, G.cage.y - 20, 30, ['#ffe9a0', '#fff', '#c9a45c'], 90, .7); S.p('shbreak') }
    await W(1600);
    /* ============ 8. doors ============ */
    mark('doors'); capOff(); black(true); await W(750); cutA(); pilot.on = false; pilot.fill = false;
    G.act = 2; enterRoom('combat'); otsEnd(); G.boss = null; G.en.length = 0; G.eb.length = 0; G.waves = []; G.pending = 0; G.cleared = true;
    G.doors = [{ x: RW / 2 - 90, t: 'elite', open: 1, aff: ['swift'] }, { x: RW / 2, t: 'rest', open: 1 }, { x: RW / 2 + 90, t: 'treasure', open: 1 }]; P.x = RW / 2; P.y = TOP + 95;
    await W(900); otsEnd(); cutB(); black(false); pilot.on = true; pilot.goal = { x: RW / 2 + 90, y: TOP + 52 };
    await cap('Sept portes', '每一扇门，都是一次选择', '演出 · 精英 · 化妆间 · 商店 · 衣柜 · 邂逅 · 终幕'); await W(3600);
    /* ============ 9. melodies ============ */
    mark('ups'); pilot.on = false; IN.sx = IN.sy = 0; capOff(); pilot.modal = true; $('modal').classList.add('pvR');
    const cs = ['fvExt', 'kiss', 'maestro', 'symbol'].map(id => UPS.find(u => u.id === id)).filter(Boolean);
    openModal('reward', `<h3>选择一段旋律</h3><p class="sub">节拍 · 狂热 · 演奏 · 舞台</p><div class="cards">${cs.map((u, i) => upCard(u, i)).join('')}</div>`);
    await cap(UPS.length + ' mélodies', '四个流派 · ' + UPS.length + ' 段旋律', '每一局都是不同的构筑'); await W(3300);
    /* ============ 10. grade ============ */
    mark('grade'); capOff(); closeModal(); G.jcA = { P: 0, GR: 0, G: 0, B: 0, M: 0 }; G.jc = { P: 58, GR: 4, G: 1, B: 0, M: 0 }; G.rcA = 63; G.act = 2;
    showJRes(() => { }); $('fade').classList.remove('on');
    await cap('Évaluation', '每一章都有演出评价', 'S+ 评价奖励稀有旋律与回忆'); await W(3300);
    /* ============ 11. memories ============ */
    mark('mem'); capOff(); closeModal(); G.state = 'title'; G.rc = 0; showMem('mortis');
    await cap('Souvenirs', '收集 36 片回忆', '读懂假面之下的她们'); await W(2000);
    memRead('mortis', 4); await W(4300);
    /* ============ 12. end card ============ */
    mark('end'); capOff(); black(true); await W(800); G.rc = 0; closeModal(); $('modal').classList.remove('pvR'); pilot.modal = false; toTitle();
    card('<div class="eb">Welcome to the world of</div><div class="logo">Ave Mujica</div><s>假 面 回 廊</s><small>同人像素肉鸽 · Android / Web</small>');
    await W(4200); cardOff(); await W(900);
    mark('done');
  } finally { clearInterval(tick); pilot.on = false }
  return log;
};
