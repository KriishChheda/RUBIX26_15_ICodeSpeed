const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electron', {
  platform: process.platform,
  
  // Example IPC methods - add more as needed for your app
  send: (channel, data) => {
    // Whitelist channels for security
    const validChannels = ['toMain'];
    if (validChannels.includes(channel)) {
      ipcRenderer.send(channel, data);
    }
  },
  
  receive: (channel, func) => {
    const validChannels = [
      'fromMain',
      'proctoring:output',
      'proctoring:stopped',
      'proctoring:alert',
      'proctoring:notification'
    ];
    if (validChannels.includes(channel)) {
      // Strip event as it includes `sender`
      ipcRenderer.on(channel, (event, ...args) => func(...args));
    }
  },
  
  invoke: (channel, ...args) => {
    const validChannels = [
      'getVersion', 
      'saveFile', 
      'openFile',
      // Proctoring channels
      'proctoring:start',
      'proctoring:stop',
      'proctoring:status',
      'proctoring:getLogs',
      'proctoring:setParticipantImage',
      'proctoring:getAlertsJSON'
    ];
    if (validChannels.includes(channel)) {
      return ipcRenderer.invoke(channel, ...args);
    }
  }
});
