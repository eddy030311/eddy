// Ave Mujica · 假面回廊 — desktop shell (Electron). Loads the offline game bundle from ./game.
const { app, BrowserWindow, Menu, shell, ipcMain, net } = require('electron');
const path = require('path');
const upd = require('./updater');
const BUNDLED = path.join(__dirname, 'game');
const STORE = () => path.join(app.getPath('userData'), 'update');
let updating = false;
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
app.setName('Ave Mujica');
if (!app.requestSingleInstanceLock()) app.quit();
let win = null;
function create() {
  Menu.setApplicationMenu(null);
  win = new BrowserWindow({
    width: 1600, height: 900, minWidth: 960, minHeight: 540, show: false, backgroundColor: '#000000',
    title: 'Ave Mujica · 假面回廊', icon: path.join(__dirname, 'icon.png'), autoHideMenuBar: true,
    webPreferences: { preload: path.join(__dirname, 'preload.js'), contextIsolation: true, sandbox: true, backgroundThrottling: false, spellcheck: false }
  });
  win.once('ready-to-show', () => { win.maximize(); win.show() });
  win.webContents.setWindowOpenHandler(({ url }) => { if (/^https?:/.test(url)) shell.openExternal(url); return { action: 'deny' } });
  win.webContents.on('will-navigate', e => e.preventDefault());
  win.webContents.on('before-input-event', (e, i) => {
    if (i.type !== 'keyDown') return;
    if (i.key === 'F11' || (i.alt && i.key === 'Enter')) { win.setFullScreen(!win.isFullScreen()); e.preventDefault() }
  });
  win.on('page-title-updated', e => e.preventDefault());
  win.loadFile(path.join(upd.liveDir(BUNDLED, STORE()), 'index.html'));
}
ipcMain.on('upd:files', e => { e.returnValue = JSON.stringify(upd.currentFiles(BUNDLED, STORE())) });
ipcMain.on('upd:run', async (e, host) => {
  if (updating || !win) return; updating = true;
  const js = c => { if (win && !win.isDestroyed()) win.webContents.executeJavaScript(c).catch(() => {}) };
  try {
    const dir = await upd.runUpdate({ bundled: BUNDLED, store: STORE(), host, fetchFn: (u, o) => net.fetch(u, o), progress: (d, t) => js(`window.updProg&&updProg(${d},${t})`) });
    js('window.updDone&&updDone(true)');
    setTimeout(() => { if (win && !win.isDestroyed()) win.loadFile(path.join(dir, 'index.html')) }, 900);
  } catch (err) { js(`window.updDone&&updDone(false,${JSON.stringify(String(err && err.message || err))})`) }
  finally { updating = false }
});
app.on('second-instance', () => { if (win) { if (win.isMinimized()) win.restore(); win.focus() } });
app.whenReady().then(create);
app.on('window-all-closed', () => app.quit());
