const { app, BrowserWindow, ipcMain } = require("electron");
const { join } = require("path");

let mainWindow; // Use a different name to avoid shadowing

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 800, 
        height: 700,
        resizable: false,
        autoHideMenuBar: true,
        show: false, 
        frame: false,
        webPreferences: {
            preload: join(__dirname, "./preload.js"),
        }
    });

    mainWindow.loadFile(join(__dirname, "../public/index.html"));
    mainWindow.on("ready-to-show", () => mainWindow.show());

    // Setup IPC handlers
    setupIPC();
}

function setupIPC() {
    ipcMain.on('app/close', () => {
        // Use getFocusedWindow() for safety
        const window = BrowserWindow.getFocusedWindow();
        window?.close(); // Close current window
    });

    ipcMain.on('app/minimize', () => {
        const window = BrowserWindow.getFocusedWindow();
        window?.minimize();
    });
}

app.whenReady().then(createWindow);