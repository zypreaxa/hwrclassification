const { app, BrowserWindow } = require("electron");
const { join } = require("path");
const path = require('node:path');

app.whenReady().then(main);

function main () {
    const window = new BrowserWindow({
        width: 800, height: 700,
        autoHideMenuBar: true,

        show: false,
        webPreferences: {
            preload: join(__dirname, "preload.js"),
        }
    });
    
    window.loadFile(join(__dirname, "../model/public/index.html"));
    
    window.on("ready-to-show", window.show);

    window.on("resized", () => {
        console.log("window resized")
        
    })

    const resizeEvents = window.listeners("resized");
    console.log(resizeEvents)
     
}
