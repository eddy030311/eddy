// Online update for the desktop build. The release branch carries update/manifest.json (every file with its sha256)
// and update/www/<file>. Unchanged files are copied from the current bundle, changed ones are downloaded and verified;
// the finished folder replaces the old one. Kept free of Electron so it can be tested with plain Node.
const fs = require('fs'), path = require('path'), crypto = require('crypto');
const HOSTS = [
  'https://raw.githubusercontent.com/eddy030311/eddy/release/ave_mujica_roguelike/update/',
  'https://cdn.jsdelivr.net/gh/eddy030311/eddy@release/ave_mujica_roguelike/update/'];
const SAFE = /^[A-Za-z0-9_-][A-Za-z0-9_.-]*(\/[A-Za-z0-9_-][A-Za-z0-9_.-]*)*$/;
const readJson = f => { try { return JSON.parse(fs.readFileSync(f, 'utf8')) } catch (e) { return null } };
const sha = b => crypto.createHash('sha256').update(b).digest('hex');

function liveDir(bundled, store) {
  const s = readJson(path.join(store, 'state.json')) || {}, b = readJson(path.join(bundled, 'manifest.json'));
  if (!s.dir || !(s.v > (b ? b.v : 0))) return bundled;
  return fs.existsSync(path.join(s.dir, 'index.html')) && fs.existsSync(path.join(s.dir, 'manifest.json')) ? s.dir : bundled;
}
function currentFiles(bundled, store) { const m = readJson(path.join(liveDir(bundled, store), 'manifest.json')); return (m && m.files) || {} }

async function runUpdate({ bundled, store, host, fetchFn, progress, hosts = HOSTS }) {
  const base = hosts[Math.max(0, Math.min(hosts.length - 1, host | 0))];
  const get = async u => { const r = await fetchFn(u, { cache: 'no-store' }); if (!r.ok) throw new Error('HTTP ' + r.status); return Buffer.from(await r.arrayBuffer()) };
  const m = JSON.parse((await get(base + 'manifest.json?t=' + Date.now())).toString('utf8'));
  const cur = liveDir(bundled, store), curM = readJson(path.join(cur, 'manifest.json')) || { v: 0, files: {} };
  if (!(m.v > (curM.v || 0))) throw new Error('已经是最新版本');
  fs.mkdirSync(store, { recursive: true });
  const tmp = path.join(store, 'www_tmp'); fs.rmSync(tmp, { recursive: true, force: true }); fs.mkdirSync(tmp, { recursive: true });
  try {
    const keys = Object.keys(m.files); let done = 0;
    for (const p of keys) {
      if (!SAFE.test(p) || p.includes('..')) throw new Error('bad path ' + p);
      const want = m.files[p].sha256; let data = null;
      if (curM.files && curM.files[p] && curM.files[p].sha256 === want) { try { data = fs.readFileSync(path.join(cur, p)); if (sha(data) !== want) data = null } catch (e) { data = null } }
      if (!data) { data = await get(base + 'www/' + p); if (sha(data) !== want) throw new Error('校验失败 ' + p) }
      const out = path.join(tmp, p); fs.mkdirSync(path.dirname(out), { recursive: true }); fs.writeFileSync(out, data);
      done++; if (done % 4 === 0 || done === keys.length) progress(done, keys.length);
    }
    fs.writeFileSync(path.join(tmp, 'manifest.json'), JSON.stringify(m));
    const dst = path.join(store, 'www_' + m.v); fs.rmSync(dst, { recursive: true, force: true }); fs.renameSync(tmp, dst);
    const old = (readJson(path.join(store, 'state.json')) || {}).dir;
    fs.writeFileSync(path.join(store, 'state.json'), JSON.stringify({ dir: dst, v: m.v }));
    if (old && old !== dst && old.startsWith(store)) fs.rmSync(old, { recursive: true, force: true });
    return dst;
  } catch (e) { fs.rmSync(tmp, { recursive: true, force: true }); throw e }
}
module.exports = { HOSTS, liveDir, currentFiles, runUpdate };
