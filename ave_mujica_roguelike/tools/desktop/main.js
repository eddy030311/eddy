// Ave Mujica · 假面回廊 — desktop shell (Electron). Loads the offline game bundle from ./game.
const { app, BrowserWindow, Menu, shell } = require('electron');
const path = require('path');
app.commandLine.appendSwitch('autoplay-policy', 'no-user-gesture-required');
app.setName('Ave Mujica');
if (!app.requestSingleInstanceLock()) app.quit();
let win = null;
function create() {
  Menu.setApplicationMenu(null);
  win = new BrowserWindow({
    width: 1600, height: 900, minWidth: 960, minHeight: 540, show: false, backgroundColor: '#000000',
    title: 'Ave Mujica · 假面回廊', icon: path.join(__dirname, 'icon.png'), autoHideMenuBar: true,
    webPreferences: { contextIsolation: true, sandbox: true, backgroundThrottling: false, spellcheck: false }
  });
  win.once('ready-to-show', () => { win.maximize(); win.show() });
  win.webContents.setWindowOpenHandler(({ url }) => { if (/^https?:/.test(url)) shell.openExternal(url); return { action: 'deny' } });
  win.webContents.on('will-navigate', e => e.preventDefault());
  win.webContents.on('before-input-event', (e, i) => {
    if (i.type !== 'keyDown') return;
    if (i.key === 'F11' || (i.alt && i.key === 'Enter')) { win.setFullScreen(!win.isFullScreen()); e.preventDefault() }
  });
  win.on('page-title-updated', e => e.preventDefault());
  win.loadFile(path.join(__dirname, 'game', 'index.html'));
}
app.on('second-instance', () => { if (win) { if (win.isMinimized()) win.restore(); win.focus() } });
app.whenReady().then(create);
app.on('window-all-closed', () => app.quit());
