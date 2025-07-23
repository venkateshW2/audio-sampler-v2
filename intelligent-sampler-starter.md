# Intelligent Sampler DAW - Electron Starter Template

## Project Structure
```
intelligent-sampler-daw/
├── package.json
├── main.js                  # Electron main process
├── preload.js              # Bridge between main and renderer
├── src/
│   ├── index.html          # Main UI
│   ├── styles.css          # Styling
│   ├── app.js              # Main application logic
│   ├── components/
│   │   ├── timeline.js     # Waveform timeline component
│   │   ├── browser.js      # Smart file browser
│   │   ├── analyzer.js     # Analysis integration
│   │   └── regions.js      # Region management
│   └── utils/
│       ├── audioLoader.js  # Audio file handling
│       └── api.js          # Backend communication
└── backend/
    └── (your existing Python analyzer)
```

## 1. package.json
```json
{
  "name": "intelligent-sampler-daw",
  "version": "1.0.0",
  "description": "DAW with intelligent audio analysis",
  "main": "main.js",
  "scripts": {
    "start": "electron .",
    "dev": "electron . --dev",
    "build": "electron-builder",
    "backend": "cd backend && python main.py"
  },
  "devDependencies": {
    "electron": "^27.0.0",
    "electron-builder": "^24.0.0"
  },
  "dependencies": {
    "peaks.js": "^3.0.0",
    "waveform-playlist": "^4.3.0",
    "tone": "^14.7.0"
  },
  "build": {
    "appId": "com.yourname.intelligentsampler",
    "productName": "Intelligent Sampler",
    "directories": {
      "output": "dist"
    },
    "mac": {
      "category": "public.app-category.music"
    },
    "win": {
      "target": "nsis"
    },
    "linux": {
      "target": "AppImage"
    }
  }
}
```

## 2. main.js (Electron Main Process)
```javascript
const { app, BrowserWindow, ipcMain, dialog, Menu } = require('electron');
const path = require('path');
const fs = require('fs').promises;
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

// Start Python backend
function startPythonBackend() {
  if (process.env.NODE_ENV !== 'production') {
    pythonProcess = spawn('python', ['backend/main.py'], {
      cwd: __dirname
    });
    
    pythonProcess.stdout.on('data', (data) => {
      console.log(`Python: ${data}`);
    });
    
    pythonProcess.stderr.on('data', (data) => {
      console.error(`Python Error: ${data}`);
    });
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1600,
    height: 1000,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false
    }
  });

  mainWindow.loadFile('src/index.html');
  
  // Open DevTools in development
  if (process.env.NODE_ENV !== 'production') {
    mainWindow.webContents.openDevTools();
  }

  // Create application menu
  createMenu();
}

function createMenu() {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'Open Audio File',
          accelerator: 'CmdOrCtrl+O',
          click: () => mainWindow.webContents.send('menu-open-file')
        },
        {
          label: 'Open Folder',
          accelerator: 'CmdOrCtrl+Shift+O',
          click: () => mainWindow.webContents.send('menu-open-folder')
        },
        { type: 'separator' },
        {
          label: 'Export Regions',
          accelerator: 'CmdOrCtrl+E',
          click: () => mainWindow.webContents.send('menu-export-regions')
        },
        { type: 'separator' },
        { role: 'quit' }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' }
      ]
    }
  ];

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

// IPC Handlers for file operations
ipcMain.handle('select-audio-file', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile'],
    filters: [
      { name: 'Audio Files', extensions: ['wav', 'mp3', 'flac', 'aiff', 'm4a'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  if (!result.canceled) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('select-folder', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openDirectory']
  });
  
  if (!result.canceled) {
    return result.filePaths[0];
  }
  return null;
});

ipcMain.handle('list-audio-files', async (event, folderPath) => {
  try {
    const files = await fs.readdir(folderPath);
    const audioExtensions = ['.wav', '.mp3', '.flac', '.aiff', '.m4a'];
    
    const audioFiles = files
      .filter(file => {
        const ext = path.extname(file).toLowerCase();
        return audioExtensions.includes(ext);
      })
      .map(file => ({
        name: file,
        path: path.join(folderPath, file)
      }));
    
    return audioFiles;
  } catch (error) {
    console.error('Error listing files:', error);
    return [];
  }
});

ipcMain.handle('read-file', async (event, filePath) => {
  try {
    const buffer = await fs.readFile(filePath);
    return buffer;
  } catch (error) {
    console.error('Error reading file:', error);
    throw error;
  }
});

// App event handlers
app.whenReady().then(() => {
  startPythonBackend();
  createWindow();
});

app.on('window-all-closed', () => {
  if (pythonProcess) {
    pythonProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});
```

## 3. preload.js (Context Bridge)
```javascript
const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  selectAudioFile: () => ipcRenderer.invoke('select-audio-file'),
  selectFolder: () => ipcRenderer.invoke('select-folder'),
  listAudioFiles: (folderPath) => ipcRenderer.invoke('list-audio-files', folderPath),
  readFile: (filePath) => ipcRenderer.invoke('read-file', filePath),
  
  // Menu events
  onMenuAction: (callback) => {
    ipcRenderer.on('menu-open-file', callback);
    ipcRenderer.on('menu-open-folder', callback);
    ipcRenderer.on('menu-export-regions', callback);
  },
  
  // Remove listeners
  removeAllListeners: () => {
    ipcRenderer.removeAllListeners('menu-open-file');
    ipcRenderer.removeAllListeners('menu-open-folder');
    ipcRenderer.removeAllListeners('menu-export-regions');
  }
});
```

## 4. src/index.html
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Intelligent Sampler DAW</title>
  <link rel="stylesheet" href="styles.css">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
</head>
<body>
  <div id="app">
    <!-- Header -->
    <header class="app-header">
      <div class="logo">
        <i class="fas fa-wave-square"></i>
        Intelligent Sampler
      </div>
      <div class="header-controls">
        <button id="analyze-all" class="btn btn-primary">
          <i class="fas fa-brain"></i> Analyze All
        </button>
        <div class="backend-status">
          <span class="status-indicator" id="backend-status"></span>
          <span>Backend</span>
        </div>
      </div>
    </header>

    <!-- Main Layout -->
    <div class="main-layout">
      <!-- File Browser -->
      <aside class="file-browser" id="file-browser">
        <div class="browser-header">
          <h3>Files</h3>
          <button id="open-folder" class="btn btn-sm">
            <i class="fas fa-folder-open"></i>
          </button>
        </div>
        <div class="search-box">
          <input type="text" id="search-input" placeholder="Search: tempo:120 key:C">
        </div>
        <div class="file-list" id="file-list">
          <!-- Files will be populated here -->
        </div>
      </aside>

      <!-- Timeline Area -->
      <main class="timeline-area">
        <!-- Transport Controls -->
        <div class="transport-controls">
          <button id="play-pause" class="btn">
            <i class="fas fa-play"></i>
          </button>
          <button id="stop" class="btn">
            <i class="fas fa-stop"></i>
          </button>
          <button id="record" class="btn">
            <i class="fas fa-circle"></i>
          </button>
          <div class="time-display">
            <span id="current-time">00:00:00</span> / 
            <span id="total-time">00:00:00</span>
          </div>
          <div class="tempo-display">
            <label>BPM:</label>
            <span id="tempo-value">--</span>
          </div>
          <div class="key-display">
            <label>Key:</label>
            <span id="key-value">--</span>
          </div>
        </div>

        <!-- Waveform Display -->
        <div id="waveform-container" class="waveform-container">
          <div id="overview-container"></div>
          <div id="zoomview-container"></div>
        </div>

        <!-- Analysis Overlay -->
        <div id="analysis-overlay" class="analysis-overlay">
          <!-- Region markers will be added here -->
        </div>

        <!-- Region List -->
        <div class="region-list">
          <h3>Detected Regions</h3>
          <div id="regions-container">
            <!-- Regions will be listed here -->
          </div>
        </div>
      </main>

      <!-- Properties Panel -->
      <aside class="properties-panel">
        <h3>Properties</h3>
        <div id="properties-content">
          <p class="empty-state">Select a file or region</p>
        </div>
      </aside>
    </div>

    <!-- Loading Overlay -->
    <div id="loading-overlay" class="loading-overlay" style="display: none;">
      <div class="loading-content">
        <div class="spinner"></div>
        <p id="loading-text">Analyzing...</p>
      </div>
    </div>
  </div>

  <!-- Scripts -->
  <script src="https://cdn.jsdelivr.net/npm/peaks.js@3.0.0/dist/peaks.js"></script>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/tone/14.8.49/Tone.js"></script>
  <script type="module" src="app.js"></script>
</body>
</html>
```

## 5. src/styles.css
```css
/* Reset and Base Styles */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: #1a1a1a;
  color: #e0e0e0;
  overflow: hidden;
}

/* App Layout */
#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.app-header {
  background: #2a2a2a;
  padding: 10px 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #3a3a3a;
}

.logo {
  font-size: 20px;
  font-weight: bold;
  display: flex;
  align-items: center;
  gap: 10px;
}

.logo i {
  color: #4CAF50;
}

.header-controls {
  display: flex;
  align-items: center;
  gap: 20px;
}

.backend-status {
  display: flex;
  align-items: center;
  gap: 5px;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #ff5252;
}

.status-indicator.connected {
  background: #4CAF50;
}

/* Main Layout */
.main-layout {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* File Browser */
.file-browser {
  width: 250px;
  background: #252525;
  border-right: 1px solid #3a3a3a;
  display: flex;
  flex-direction: column;
}

.browser-header {
  padding: 15px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #3a3a3a;
}

.search-box {
  padding: 10px;
}

.search-box input {
  width: 100%;
  padding: 8px;
  background: #1a1a1a;
  border: 1px solid #3a3a3a;
  border-radius: 4px;
  color: #e0e0e0;
}

.file-list {
  flex: 1;
  overflow-y: auto;
  padding: 10px;
}

.file-item {
  padding: 8px 12px;
  cursor: pointer;
  border-radius: 4px;
  margin-bottom: 2px;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.file-item:hover {
  background: #333;
}

.file-item.selected {
  background: #4CAF50;
  color: white;
}

.file-item.analyzed::after {
  content: '✓';
  color: #4CAF50;
  margin-left: 5px;
}

/* Timeline Area */
.timeline-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: #1e1e1e;
}

.transport-controls {
  background: #2a2a2a;
  padding: 10px 20px;
  display: flex;
  align-items: center;
  gap: 15px;
  border-bottom: 1px solid #3a3a3a;
}

.time-display, .tempo-display, .key-display {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 0 10px;
}

.waveform-container {
  flex: 1;
  position: relative;
  overflow: hidden;
}

#overview-container {
  height: 85px;
  border-bottom: 1px solid #3a3a3a;
}

#zoomview-container {
  flex: 1;
}

.analysis-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
}

.region-marker {
  position: absolute;
  background: rgba(76, 175, 80, 0.3);
  border: 1px solid #4CAF50;
  pointer-events: all;
  cursor: pointer;
}

.region-marker:hover {
  background: rgba(76, 175, 80, 0.5);
}

.region-list {
  height: 200px;
  background: #252525;
  border-top: 1px solid #3a3a3a;
  padding: 15px;
  overflow-y: auto;
}

.region-item {
  background: #333;
  padding: 10px;
  margin-bottom: 8px;
  border-radius: 4px;
  cursor: pointer;
}

.region-item:hover {
  background: #3a3a3a;
}

/* Properties Panel */
.properties-panel {
  width: 300px;
  background: #252525;
  border-left: 1px solid #3a3a3a;
  padding: 15px;
  overflow-y: auto;
}

.property-group {
  margin-bottom: 20px;
}

.property-group h4 {
  margin-bottom: 10px;
  color: #4CAF50;
}

.property-item {
  display: flex;
  justify-content: space-between;
  padding: 5px 0;
}

.property-label {
  color: #999;
}

.property-value {
  color: #e0e0e0;
  font-weight: 500;
}

/* Buttons */
.btn {
  background: #333;
  border: 1px solid #444;
  color: #e0e0e0;
  padding: 8px 16px;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
  transition: all 0.2s;
}

.btn:hover {
  background: #444;
  border-color: #555;
}

.btn:active {
  transform: translateY(1px);
}

.btn-primary {
  background: #4CAF50;
  border-color: #4CAF50;
  color: white;
}

.btn-primary:hover {
  background: #45a049;
  border-color: #45a049;
}

.btn-sm {
  padding: 4px 8px;
  font-size: 12px;
}

/* Loading Overlay */
.loading-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.loading-content {
  text-align: center;
}

.spinner {
  width: 50px;
  height: 50px;
  border: 3px solid #333;
  border-top-color: #4CAF50;
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 20px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

/* Scrollbar Styling */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: #1a1a1a;
}

::-webkit-scrollbar-thumb {
  background: #555;
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #666;
}

/* Empty States */
.empty-state {
  color: #666;
  text-align: center;
  padding: 20px;
}
```

## 6. src/app.js (Main Application)
```javascript
// Import components
import { Timeline } from './components/timeline.js';
import { FileBrowser } from './components/browser.js';
import { AnalysisEngine } from './components/analyzer.js';
import { RegionManager } from './components/regions.js';
import { AudioLoader } from './utils/audioLoader.js';
import { API } from './utils/api.js';

class IntelligentSamplerApp {
  constructor() {
    this.currentFile = null;
    this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    
    // Initialize components
    this.api = new API('http://localhost:8000'); // Your Python backend
    this.audioLoader = new AudioLoader(this.audioContext);
    this.timeline = new Timeline('zoomview-container', 'overview-container');
    this.browser = new FileBrowser('file-list', this.onFileSelect.bind(this));
    this.analyzer = new AnalysisEngine(this.api);
    this.regionManager = new RegionManager('regions-container', 'analysis-overlay');
    
    // Initialize UI
    this.initializeUI();
    this.checkBackendConnection();
    
    // Set up menu handlers
    this.setupMenuHandlers();
  }
  
  initializeUI() {
    // Transport controls
    document.getElementById('play-pause').addEventListener('click', () => this.togglePlayback());
    document.getElementById('stop').addEventListener('click', () => this.stop());
    
    // File operations
    document.getElementById('open-folder').addEventListener('click', () => this.openFolder());
    document.getElementById('analyze-all').addEventListener('click', () => this.analyzeAll());
    
    // Search
    document.getElementById('search-input').addEventListener('input', (e) => {
      this.browser.search(e.target.value);
    });
  }
  
  setupMenuHandlers() {
    window.electronAPI.onMenuAction((event, action) => {
      switch(action) {
        case 'menu-open-file':
          this.openFile();
          break;
        case 'menu-open-folder':
          this.openFolder();
          break;
        case 'menu-export-regions':
          this.exportRegions();
          break;
      }
    });
  }
  
  async checkBackendConnection() {
    try {
      const status = await this.api.checkHealth();
      if (status.ok) {
        document.getElementById('backend-status').classList.add('connected');
      }
    } catch (error) {
      console.error('Backend not connected:', error);
      this.showNotification('Backend not connected. Please start the Python server.', 'error');
    }
  }
  
  async openFile() {
    const filePath = await window.electronAPI.selectAudioFile();
    if (filePath) {
      await this.loadAndAnalyzeFile(filePath);
    }
  }
  
  async openFolder() {
    const folderPath = await window.electronAPI.selectFolder();
    if (folderPath) {
      const files = await window.electronAPI.listAudioFiles(folderPath);
      this.browser.loadFiles(files);
    }
  }
  
  async onFileSelect(file) {
    this.currentFile = file;
    await this.loadAndAnalyzeFile(file.path);
  }
  
  async loadAndAnalyzeFile(filePath) {
    this.showLoading('Loading audio file...');
    
    try {
      // Load audio
      const audioBuffer = await this.audioLoader.loadFile(filePath);
      
      // Display waveform
      await this.timeline.loadAudio(audioBuffer, filePath);
      
      // Update UI
      this.updateTimeDisplay(audioBuffer.duration);
      
      // Check if already analyzed
      const existingAnalysis = await this.api.getAnalysis(filePath);
      
      if (existingAnalysis) {
        this.displayAnalysis(existingAnalysis);
      } else {
        // Analyze
        this.showLoading('Analyzing audio...');
        const analysis = await this.analyzer.analyzeFile(filePath, audioBuffer);
        this.displayAnalysis(analysis);
      }
      
      this.hideLoading();
      
    } catch (error) {
      console.error('Error loading file:', error);
      this.showNotification('Error loading file: ' + error.message, 'error');
      this.hideLoading();
    }
  }
  
  displayAnalysis(analysis) {
    // Update transport displays
    document.getElementById('tempo-value').textContent = 
      analysis.tempo ? `${Math.round(analysis.tempo)}` : '--';
    document.getElementById('key-value').textContent = 
      analysis.key || '--';
    
    // Display regions
    this.regionManager.clearRegions();
    if (analysis.regions) {
      analysis.regions.forEach(region => {
        this.regionManager.addRegion(region);
        this.timeline.addRegion(region);
      });
    }
    
    // Update properties panel
    this.updatePropertiesPanel(analysis);
    
    // Mark file as analyzed
    this.browser.markFileAnalyzed(this.currentFile.path);
  }
  
  updatePropertiesPanel(data) {
    const container = document.getElementById('properties-content');
    
    if (!data) {
      container.innerHTML = '<p class="empty-state">Select a file or region</p>';
      return;
    }
    
    let html = '';
    
    // File properties
    if (data.file_info) {
      html += `
        <div class="property-group">
          <h4>File Info</h4>
          <div class="property-item">
            <span class="property-label">Duration:</span>
            <span class="property-value">${this.formatTime(data.file_info.duration)}</span>
          </div>
          <div class="property-item">
            <span class="property-label">Sample Rate:</span>
            <span class="property-value">${data.file_info.sample_rate} Hz</span>
          </div>
        </div>
      `;
    }
    
    // Musical properties
    if (data.tempo || data.key) {
      html += `
        <div class="property-group">
          <h4>Musical Analysis</h4>
          ${data.tempo ? `
          <div class="property-item">
            <span class="property-label">Tempo:</span>
            <span class="property-value">${Math.round(data.tempo)} BPM</span>
          </div>
          ` : ''}
          ${data.key ? `
          <div class="property-item">
            <span class="property-label">Key:</span>
            <span class="property-value">${data.key}</span>
          </div>
          ` : ''}
        </div>
      `;
    }
    
    // Classification
    if (data.classification) {
      html += `
        <div class="property-group">
          <h4>Classification</h4>
          ${Object.entries(data.classification).map(([type, confidence]) => `
          <div class="property-item">
            <span class="property-label">${type}:</span>
            <span class="property-value">${Math.round(confidence * 100)}%</span>
          </div>
          `).join('')}
        </div>
      `;
    }
    
    container.innerHTML = html;
  }
  
  async analyzeAll() {
    const files = this.browser.getAllFiles();
    if (files.length === 0) {
      this.showNotification('No files to analyze', 'info');
      return;
    }
    
    this.showLoading(`Analyzing ${files.length} files...`);
    
    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      this.updateLoadingText(`Analyzing ${i + 1} of ${files.length}: ${file.name}`);
      
      try {
        const audioBuffer = await this.audioLoader.loadFile(file.path);
        const analysis = await this.analyzer.analyzeFile(file.path, audioBuffer);
        this.browser.markFileAnalyzed(file.path);
      } catch (error) {
        console.error(`Error analyzing ${file.name}:`, error);
      }
    }
    
    this.hideLoading();
    this.showNotification(`Analyzed ${files.length} files`, 'success');
  }
  
  async exportRegions() {
    if (!this.currentFile || !this.regionManager.regions.length) {
      this.showNotification('No regions to export', 'info');
      return;
    }
    
    // Implementation for exporting regions
    // This would create individual audio files for each region
    this.showNotification('Export functionality coming soon', 'info');
  }
  
  togglePlayback() {
    if (this.timeline.isPlaying()) {
      this.timeline.pause();
      document.querySelector('#play-pause i').className = 'fas fa-play';
    } else {
      this.timeline.play();
      document.querySelector('#play-pause i').className = 'fas fa-pause';
    }
  }
  
  stop() {
    this.timeline.stop();
    document.querySelector('#play-pause i').className = 'fas fa-play';
  }
  
  updateTimeDisplay(duration) {
    document.getElementById('total-time').textContent = this.formatTime(duration);
  }
  
  formatTime(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }
  
  showLoading(text = 'Loading...') {
    document.getElementById('loading-overlay').style.display = 'flex';
    document.getElementById('loading-text').textContent = text;
  }
  
  updateLoadingText(text) {
    document.getElementById('loading-text').textContent = text;
  }
  
  hideLoading() {
    document.getElementById('loading-overlay').style.display = 'none';
  }
  
  showNotification(message, type = 'info') {
    // Simple notification system - could be enhanced with a toast library
    console.log(`[${type.toUpperCase()}] ${message}`);
  }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  window.app = new IntelligentSamplerApp();
});
```

## 7. src/components/timeline.js
```javascript
export class Timeline {
  constructor(zoomviewId, overviewId) {
    this.zoomviewId = zoomviewId;
    this.overviewId = overviewId;
    this.peaks = null;
    this.isPlaying = false;
  }
  
  async loadAudio(audioBuffer, sourceUrl) {
    // Destroy existing instance
    if (this.peaks) {
      this.peaks.destroy();
    }
    
    // Initialize Peaks.js
    const options = {
      containers: {
        overview: document.getElementById(this.overviewId),
        zoomview: document.getElementById(this.zoomviewId)
      },
      mediaElement: document.createElement('audio'),
      webAudio: {
        audioBuffer: audioBuffer,
        scale: 128,
        multiChannel: false
      },
      keyboard: true,
      showPlayheadTime: true,
      zoomLevels: [128, 256, 512, 1024, 2048, 4096],
      waveformColor: '#4CAF50',
      playedWaveformColor: '#2E7D32',
      playheadColor: '#fff',
      playheadTextColor: '#fff',
      axisGridlineColor: '#555',
      axisLabelColor: '#aaa',
      segmentOptions: {
        markers: false,
        overlay: true,
        waveformColor: '#ff0000',
        overlayColor: 'rgba(255, 0, 0, 0.3)',
        overlayBorderColor: '#ff0000',
        overlayBorderWidth: 2
      }
    };
    
    return new Promise((resolve, reject) => {
      Peaks.init(options, (err, peaks) => {
        if (err) {
          reject(err);
          return;
        }
        
        this.peaks = peaks;
        
        // Set up event handlers
        this.peaks.on('segments.enter', (segment) => {
          console.log('Entered segment:', segment);
        });
        
        resolve(peaks);
      });
    });
  }
  
  addRegion(region) {
    if (!this.peaks) return;
    
    const segment = {
      startTime: region.start_time,
      endTime: region.end_time,
      labelText: region.label || region.type,
      color: this.getColorForType(region.type),
      editable: true
    };
    
    this.peaks.segments.add(segment);
  }
  
  getColorForType(type) {
    const colors = {
      'drums': '#FF5722',
      'melody': '#2196F3',
      'bass': '#9C27B0',
      'vocals': '#4CAF50',
      'texture': '#FF9800',
      'silence': '#607D8B'
    };
    
    return colors[type] || '#757575';
  }
  
  play() {
    if (this.peaks) {
      this.peaks.player.play();
      this.isPlaying = true;
    }
  }
  
  pause() {
    if (this.peaks) {
      this.peaks.player.pause();
      this.isPlaying = false;
    }
  }
  
  stop() {
    if (this.peaks) {
      this.peaks.player.seek(0);
      this.peaks.player.pause();
      this.isPlaying = false;
    }
  }
  
  isPlaying() {
    return this.isPlaying;
  }
  
  destroy() {
    if (this.peaks) {
      this.peaks.destroy();
      this.peaks = null;
    }
  }
}