// Bridge for the in-game updater (window.AMNative), mirroring the Android shell.
const { contextBridge, ipcRenderer } = require('electron');
contextBridge.exposeInMainWorld('AMNative', {
  platform: () => 'windows',
  files: () => ipcRenderer.sendSync('upd:files'),
  update: host => ipcRenderer.send('upd:run', host | 0)
});
