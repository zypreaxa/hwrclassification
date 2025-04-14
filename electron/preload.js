const { contextBridge } = require("electron");

const API = {
  
};

contextBridge.exposeInMainWorld("api", API);

