const {chromium}=require('playwright');const FN=process.env.DIRFN||'__direct';const fs=require('fs');const path=require('path');
const OUT=process.argv[2]||path.join(__dirname,'take');const W=+(process.argv[3]||1280),H=+(process.argv[4]||720);const WARM=process.argv[5]!=='nowarm';
(async()=>{fs.rmSync(OUT,{recursive:true,force:true});fs.mkdirSync(path.join(OUT,'f'),{recursive:true});
const b=await chromium.launch({args:['--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader','--autoplay-policy=no-user-gesture-required']});
const ctx=await b.newContext({viewport:{width:W,height:H},deviceScaleFactor:1,isMobile:true,hasTouch:true});const p=await ctx.newPage();const errs=[];p.on('pageerror',e=>errs.push(e.stack));
await p.goto('file:///home/user/eddy/ave_mujica_roguelike/index.html#skip');
await p.evaluate(()=>{localStorage.setItem('amj_unlock',JSON.stringify({timoris:1,amoris:1,mortis:1,oblivionis:1}));localStorage.setItem('amj_best',JSON.stringify({act:5,wins:3,endings:{a:1,b:1}}));
  localStorage.setItem('amj_mem',JSON.stringify({got:{mortis:6,doloris:4,oblivionis:3,timoris:2,amoris:2},sc:{},nw:{}}));localStorage.setItem('amj_settings',JSON.stringify({music:true,sfx:true,fps:60}))});
await p.reload();await p.waitForSelector('#start',{state:'visible',timeout:180000});await p.waitForFunction(()=>R3.iconsDone&&UI.upsN>69,null,{timeout:300000}).catch(()=>{});
await p.addScriptTag({path:path.join(__dirname,process.env.DIRECTOR||'director.js')});
await p.evaluate(m=>{if(m)WORLD=2},!!process.env.MASQUE);await p.evaluate(()=>new Promise(r=>prepWorld(r)));
if(WARM){console.log('warm-up pass');await p.evaluate(fn=>window[fn](),FN);}
await p.evaluate(()=>{auInit();localStorage.setItem('amj_mem',JSON.stringify({got:{mortis:6,doloris:4,oblivionis:3,timoris:2,amoris:2},sc:{},nw:{}}));MS.got={mortis:6,doloris:4,oblivionis:3,timoris:2,amoris:2};
  G.rc=0;G.rcMax=0;G.jc={P:0,G:0,B:0,M:0};toTitle();document.querySelector('#pv .blk').classList.add('on')});await p.waitForTimeout(1500);
// audio tap on the master bus
await p.evaluate(()=>{const d=A.ctx.createMediaStreamDestination();A.comp.connect(d);const rec=new MediaRecorder(d.stream,{mimeType:'audio/webm;codecs=opus',audioBitsPerSecond:192000});window.__ach=[];rec.ondataavailable=e=>__ach.push(e.data);window.__rec=rec});
const cdp=await ctx.newCDPSession(p);const frames=[];let n=0;
cdp.on('Page.screencastFrame',async f=>{const fn=path.join(OUT,'f',String(n++).padStart(6,'0')+'.jpg');fs.writeFileSync(fn,Buffer.from(f.data,'base64'));frames.push([fn,f.metadata.timestamp]);try{await cdp.send('Page.screencastFrameAck',{sessionId:f.sessionId})}catch(e){}});
await cdp.send('Page.startScreencast',{format:'jpeg',quality:93,maxWidth:W,maxHeight:H,everyNthFrame:1});
await p.waitForTimeout(300);
const aStart=await p.evaluate(()=>{__rec.start(200);return Date.now()/1000});
const log=await p.evaluate(fn=>window[fn](),FN);const cuts=await p.evaluate(()=>window.__cuts||[]);
await p.waitForTimeout(400);
await cdp.send('Page.stopScreencast');
const audio=await p.evaluate(()=>new Promise(r=>{__rec.onstop=async()=>{const bl=new Blob(__ach,{type:'audio/webm'});const buf=new Uint8Array(await bl.arrayBuffer());let s='';for(let i=0;i<buf.length;i+=32768)s+=String.fromCharCode.apply(null,buf.subarray(i,i+32768));r(btoa(s))};__rec.stop()}));
fs.writeFileSync(path.join(OUT,'audio.webm'),Buffer.from(audio,'base64'));
fs.writeFileSync(path.join(OUT,'meta.json'),JSON.stringify({aStart,cuts,frames:frames.map(f=>[path.basename(f[0]),f[1]]),log,errs}));
console.log('frames',frames.length,'dur',(frames.at(-1)[1]-frames[0][1]).toFixed(1),'audioOffset',(aStart-frames[0][1]).toFixed(3),'errs',errs.length,errs[0]||'');console.log(JSON.stringify(log));
await b.close()})();
