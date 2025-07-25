// Audio Intelligence Sampler v2 - Main Application
// Standalone architecture ready for component modularization

console.log('🚀 Loading Audio Sampler v2...');

// Main application class
class AudioSamplerApp {
    constructor() {
        console.log('🚀 AudioSamplerApp constructor started...');
        this.apiBase = '';  // Will be determined automatically
        this.isInitialized = false;
        this.isAnalyzing = false;
        this.currentFile = null;
        this.currentFileAnalysis = null;
        this.currentAudioBuffer = null;
        this.statusCheckInterval = null;
        this.peaksInstance = null;
        
        console.log('📋 Collecting UI elements...');
        
        // UI elements with error checking
        const elements = {
            uploadArea: 'uploadArea',
            uploadInput: 'fileInput', 
            uploadPathInput: 'filePath',
            loadingOverlay: 'loadingOverlay',
            loadingText: 'loadingText',
            statusIndicator: 'statusIndicator',
            statusText: 'statusText',
            databaseBrowserBtn: 'database-browser',
            databaseModal: 'databaseModal',
            databaseModalClose: 'closeModal',
            databaseSearch: 'searchQuery',
            databaseResults: 'databaseContent',
            playPauseBtn: 'playPause',
            stopBtn: 'stop',
            tempoDisplay: 'tempoValue',
            keyDisplay: 'keyValue',
            propertiesPanel: 'propertiesContent'
        };
        
        const missing = [];
        
        // Collect elements and track missing ones
        for (const [prop, id] of Object.entries(elements)) {
            const element = document.getElementById(id);
            this[prop] = element;
            if (!element) {
                missing.push(id);
            }
        }
        
        if (missing.length > 0) {
            console.warn('⚠️ Missing DOM elements:', missing);
            console.warn('⚠️ This may cause initialization issues');
        } else {
            console.log('✅ All DOM elements found');
        }
        
        console.log('🔧 Starting app initialization...');
        console.log('🔧 Constructor completed, calling initializeApp...');
        this.initializeApp();
    }
    
    async initializeApp() {
        try {
            console.log('🔧 Setting up components...');
            
            // Determine API base URL (same origin) 
            this.apiBase = window.location.origin;
            console.log(`🌐 API base URL: ${this.apiBase}`);
            
            // Initialize components with detailed logging
            console.log('📦 Initializing AudioLoader...');
            this.audioLoader = new AudioLoader(this.apiBase);
            console.log('✅ AudioLoader initialized');
            
            console.log('📦 Initializing RegionManager...');
            this.regionManager = new RegionManager('segmentsList', null);
            console.log('✅ RegionManager initialized');
            
            console.log('📦 Initializing WaveformPlayer...');
            this.waveformPlayer = new WaveformPlayer();
            console.log('✅ WaveformPlayer initialized');
            
            // Set up event listeners
            console.log('🎧 Setting up event listeners...');
            this.setupEventListeners();
            console.log('✅ Event listeners setup complete');
            
            // Check backend status
            console.log('🌐 Checking backend status...');
            await this.checkBackendStatus();
            console.log('✅ Backend status checked');
            
            console.log('⏰ Starting status monitoring...');
            this.startStatusChecking();
            console.log('✅ Status monitoring started');
            
            this.isInitialized = true;
            console.log('🎉 App initialization complete - READY!');
        } catch (error) {
            console.error('❌ App initialization failed at step:', error.message);
            console.error('❌ Full error:', error);
            console.error('❌ Stack trace:', error.stack);
            this.showError('Initialization failed: ' + error.message);
        }
    }
    
    async checkBackendStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            console.log('Backend status:', data);
            
            // Update status indicators
            if (this.statusIndicator && this.statusText && data && data.status) {
                this.statusIndicator.classList.toggle('connected', data.status === 'healthy');
                this.statusText.textContent = data.status === 'healthy' ? 'Connected' : 'Disconnected';
            }
            
            // Update engine status in properties panel
            const engineStatus = document.getElementById('engineStatus');
            const pluginStatus = document.getElementById('pluginStatus');
            
            if (data && data.engine) {
                if (engineStatus) engineStatus.textContent = 'READY';
                if (pluginStatus) pluginStatus.textContent = `${data.engine.plugins_registered || 0} LOADED`;
            }
            
            return data && data.status === 'healthy';
        } catch (error) {
            console.error('Backend status check failed:', error);
            
            if (this.statusIndicator && this.statusText) {
                this.statusIndicator.classList.remove('connected');
                this.statusText.textContent = 'Disconnected';
            }
            
            return false;
        }
    }

    setupEventListeners() {
        // Upload area events
        if (this.uploadArea) {
            this.uploadArea.addEventListener('dragover', e => {
                e.preventDefault();
                this.uploadArea.classList.add('drag-over');
            });
            
            this.uploadArea.addEventListener('dragleave', () => {
                this.uploadArea.classList.remove('drag-over');
            });
            
            this.uploadArea.addEventListener('drop', e => {
                e.preventDefault();
                this.uploadArea.classList.remove('drag-over');
                
                if (e.dataTransfer.files.length > 0) {
                    this.uploadInput.files = e.dataTransfer.files;
                    this.handleFileUpload();
                }
            });
            
            this.uploadArea.addEventListener('click', () => {
                this.uploadInput.click();
            });
        }
        
        // Upload form events - handled by individual elements
        
        // Upload input events
        if (this.uploadInput) {
            this.uploadInput.addEventListener('change', () => {
                this.handleFileUpload();
            });
        }
        
        // Path analyze button
        const analyzeBtn = document.getElementById('analyzePathBtn');
        if (analyzeBtn) {
            analyzeBtn.addEventListener('click', () => {
                this.handleFileUpload();
            });
        }
        
        // Database browser button
        if (this.databaseBrowserBtn) {
            this.databaseBrowserBtn.addEventListener('click', () => {
                this.openDatabaseBrowser();
            });
        }
        
        // Database modal close button
        if (this.databaseModalClose) {
            this.databaseModalClose.addEventListener('click', () => {
                this.databaseModal.style.display = 'none';
            });
        }
        
        // Database search input
        if (this.databaseSearch) {
            this.databaseSearch.addEventListener('input', e => {
                this.searchDatabase(e.target.value);
            });
        }
        
        // Database search button
        const searchBtn = document.getElementById('searchBtn');
        if (searchBtn) {
            searchBtn.addEventListener('click', () => {
                const query = this.databaseSearch ? this.databaseSearch.value : '';
                this.searchDatabase(query);
            });
        }
        
        // Transport controls
        if (this.playPauseBtn) {
            this.playPauseBtn.addEventListener('click', () => {
                this.togglePlayback();
            });
        }
        
        if (this.stopBtn) {
            this.stopBtn.addEventListener('click', () => {
                this.stopPlayback();
            });
        }
        
        // Volume controls
        const muteBtn = document.getElementById('muteBtn');
        const volumeSlider = document.getElementById('volumeSlider');
        
        if (muteBtn) {
            muteBtn.addEventListener('click', () => {
                this.toggleMute();
            });
        }
        
        if (volumeSlider) {
            volumeSlider.addEventListener('input', (e) => {
                this.setVolume(e.target.value / 100);
            });
        }
        
        // Window click to close modal
        window.addEventListener('click', e => {
            if (e.target === this.databaseModal) {
                this.databaseModal.style.display = 'none';
            }
        });
        
        // Custom events
        document.addEventListener('regionSelected', event => {
            this.onRegionSelected(event.detail.region);
        });
        
        document.addEventListener('timelineSegmentSelected', event => {
            this.onTimelineSegmentSelected(event.detail.segment);
        });
    }
    
    
    startStatusChecking() {
        // Check status every 30 seconds
        this.statusCheckInterval = setInterval(() => {
            this.checkBackendStatus();
        }, 30000);
    }
    
    async handleFileUpload() {
        if ((!this.uploadInput || !this.uploadInput.files.length) && (!this.uploadPathInput || !this.uploadPathInput.value)) {
            this.showError('Please select a file or enter a file path');
            return;
        }
        
        this.showLoading('Uploading audio file...');
        
        try {
            let fileData;
            let filePath;
            
            if (this.uploadInput.files.length) {
                // File upload from input
                const file = this.uploadInput.files[0];
                fileData = new FormData();
                fileData.append('file', file);
                filePath = file.name;
                
                this.currentFile = {
                    name: file.name,
                    size: file.size,
                    type: file.type
                };
            } else if (this.uploadPathInput.value) {
                // File path input
                filePath = this.uploadPathInput.value;
                
                // Path-based upload
                fileData = new FormData();
                fileData.append('file_path', filePath);
                
                this.currentFile = {
                    name: filePath.split('/').pop(),
                    path: filePath
                };
            }
            
            let response;
            
            if (this.uploadInput.files.length) {
                // File upload
                response = await fetch(`${this.apiBase}/api/analyze/upload`, {
                    method: 'POST',
                    body: fileData
                });
            } else {
                // Path-based analysis
                response = await fetch(`${this.apiBase}/analyze`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ file_path: filePath })
                });
            }
            
            if (!response.ok) {
                throw new Error(`Upload failed: ${response.status} ${response.statusText}`);
            }
            
            const apiResponse = await response.json();
            console.log('API Response:', apiResponse);
            
            // Extract the actual analysis result from the API wrapper
            const result = apiResponse.analysis_result || apiResponse;
            console.log('Analysis result:', result);
            
            // Check if we have waveform data from backend
            const waveformData = result.waveform_data;
            if (waveformData) {
                this.updateLoadingText('Initializing waveform visualization...');
                
                // Use processed audio file if available (for timeline alignment)
                const audioElement = new Audio();
                
                if (result.upload_metadata?.processed_file_path) {
                    // Use processed file from backend (ensures timeline alignment)
                    audioElement.src = `${this.apiBase}/api/processed_audio/${encodeURIComponent(result.upload_metadata.processed_file_path)}`;
                    console.log(`🎵 Using processed audio: ${result.upload_metadata.processed_file_path}`);
                } else if (this.uploadInput.files.length) {
                    // Fallback to original file for uploads
                    audioElement.src = URL.createObjectURL(this.uploadInput.files[0]);
                    console.log('🎵 Using original uploaded file');
                } else {
                    // Fallback to file path for path-based analysis
                    audioElement.src = result.file_path || '';
                    console.log('🎵 Using file path');
                }
                
                audioElement.preload = 'auto';
                await this.initializeWaveformWithData(audioElement, waveformData);
            } else if (this.uploadInput.files.length) {
                // Fallback to old audio loading method
                this.updateLoadingText('Loading audio data...');
                await this.loadAudioFile(this.uploadInput.files[0]);
            }
            
            // Show analysis results
            this.displayAnalysisResults(result);
            
            // Hide loading overlay
            this.hideLoading();
        } catch (error) {
            console.error('File upload/analysis failed:', error);
            this.showError(`Analysis failed: ${error.message}`);
            this.hideLoading();
        }
    }
    
    async loadAudioFile(file) {
        try {
            // Use AudioLoader to get audio buffer from file
            const audioBuffer = await this.audioLoader.getAudioBufferFromFile(file);
            this.currentAudioBuffer = audioBuffer;
            
            // Initialize timeline with audio
            await this.initializeTimeline(audioBuffer);
            
            return audioBuffer;
        } catch (error) {
            console.error('Audio loading failed:', error);
            // Don't throw - continue without audio visualization
            console.warn('Continuing without audio visualization');
        }
    }
    
    async initializeWaveformWithData(audioElement, waveformData) {
        try {
            console.log('🎵 Initializing waveform with backend data:', {
                waveformData: !!waveformData,
                duration: waveformData?.duration,
                overviewSamples: waveformData?.overview?.data?.length,
                zoomSamples: waveformData?.zoom?.data?.length
            });
            
            // Check if Peaks.js is available
            if (typeof peaks === 'undefined') {
                console.warn('⚠️ Peaks.js not available, skipping waveform visualization');
                console.log('Debug: peaksAvailable =', window.peaksAvailable, ', peaks =', typeof peaks);
                this.showEmptyWaveformState();
                return null;
            }
            
            // Show waveform container and hide empty state
            console.log('📺 Showing waveform container...');
            this.showWaveformContainer();
            
            // Wait a moment for DOM updates
            await new Promise(resolve => setTimeout(resolve, 100));
            
            // Debug: Check container dimensions before Peaks.js init
            const overviewContainer = document.getElementById('overview-container');
            const zoomContainer = document.getElementById('zoom-container');
            const waveformContainer = document.getElementById('waveformContainer');
            
            console.log('🔍 Container existence check:');
            console.log(`   - waveformContainer exists: ${!!waveformContainer}`);
            console.log(`   - overviewContainer exists: ${!!overviewContainer}`);
            console.log(`   - zoomContainer exists: ${!!zoomContainer}`);
            
            if (waveformContainer) {
                console.log(`   - waveformContainer display: ${getComputedStyle(waveformContainer).display}`);
                console.log(`   - waveformContainer visibility: ${getComputedStyle(waveformContainer).visibility}`);
            }
            
            if (overviewContainer && zoomContainer) {
                console.log('🔍 Container dimensions check:');
                console.log(`   - Overview: ${overviewContainer.offsetWidth}x${overviewContainer.offsetHeight}`);
                console.log(`   - Zoom: ${zoomContainer.offsetWidth}x${zoomContainer.offsetHeight}`);
                console.log(`   - Overview visible: ${overviewContainer.offsetParent !== null}`);
                console.log(`   - Zoom visible: ${zoomContainer.offsetParent !== null}`);
                
                // Force dimensions if they're zero
                if (overviewContainer.offsetWidth === 0 || overviewContainer.offsetHeight === 0) {
                    console.log('⚠️ Overview container has zero dimensions, forcing size...');
                    overviewContainer.style.cssText = 'width: 100% !important; height: 80px !important; display: block !important;';
                }
                
                if (zoomContainer.offsetWidth === 0 || zoomContainer.offsetHeight === 0) {
                    console.log('⚠️ Zoom container has zero dimensions, forcing size...');
                    console.log(`   - Current computed style: ${getComputedStyle(zoomContainer).height}`);
                    console.log(`   - Current inline style: ${zoomContainer.style.height}`);
                    
                    // Force entire layout chain
                    const waveformSection = document.getElementById('waveformContainer');
                    const centerPanel = document.querySelector('.center-panel');
                    const mainLayout = document.querySelector('.main-layout');
                    
                    console.log('🔧 Forcing entire layout chain...');
                    
                    // Force main layout height
                    if (mainLayout) {
                        mainLayout.style.cssText += ' height: 100vh !important; min-height: 600px !important;';
                        console.log(`   - Main layout: ${mainLayout.offsetHeight}px`);
                    }
                    
                    // Force center panel height  
                    if (centerPanel) {
                        centerPanel.style.cssText += ' height: 100% !important; min-height: 500px !important;';
                        console.log(`   - Center panel: ${centerPanel.offsetHeight}px`);
                    }
                    
                    // Force waveform section height
                    if (waveformSection) {
                        waveformSection.style.cssText += ' height: 400px !important; min-height: 400px !important;';
                        console.log(`   - Waveform section: ${waveformSection.offsetHeight}px`);
                    }
                    
                    // Force parent container height
                    const parentContainer = zoomContainer.parentElement;
                    if (parentContainer) {
                        parentContainer.style.cssText += ' height: 300px !important; min-height: 300px !important;';
                        console.log(`   - Parent container: ${parentContainer.offsetHeight}px`);
                    }
                    
                    // Finally force zoom container
                    zoomContainer.style.cssText = 'width: 100% !important; height: 200px !important; display: block !important; min-height: 200px !important; flex: none !important; background: red !important;';
                }
                
                // Wait for DOM updates
                await new Promise(resolve => setTimeout(resolve, 100));
                
                // Re-check dimensions after forcing
                console.log('🔍 Post-fix dimensions:');
                console.log(`   - Overview: ${overviewContainer.offsetWidth}x${overviewContainer.offsetHeight}`);
                console.log(`   - Zoom: ${zoomContainer.offsetWidth}x${zoomContainer.offsetHeight}`);
            } else {
                console.error('❌ Containers not found in DOM!');
                this.showEmptyWaveformState();
                return null;
            }
            
            // Clean up previous instance
            if (this.peaksInstance) {
                this.peaksInstance.destroy();
                this.peaksInstance = null;
                console.log('🗑️ Destroyed previous Peaks instance');
            }
            
            const options = {
                zoomview: {
                    container: document.getElementById('zoom-container')
                },
                overview: {
                    container: document.getElementById('overview-container')
                },
                mediaElement: audioElement,
                webAudio: {
                    audioContext: new (window.AudioContext || window.webkitAudioContext)(),
                    scale: 128,
                    multiChannel: false
                },
                keyboard: true,
                showPlayheadTime: true,
                zoomWaveformColor: '#4ECDC4',
                overviewWaveformColor: '#45B7D1',
                pointMarkerColor: '#FF6B6B'
            };
            
            return new Promise((resolve, reject) => {
                peaks.init(options, (err, peaksInstance) => {
                    if (err) {
                        console.error('❌ Peaks initialization failed:', err.message);
                        this.showEmptyWaveformState();
                        reject(err);
                        return;
                    }
                    
                    console.log('✅ SUCCESS! Peaks.js initialized with waveform display!');
                    this.peaksInstance = peaksInstance;
                    
                    // Set up time tracking for transport controls
                    audioElement.addEventListener('timeupdate', () => {
                        this.updateTimeDisplay();
                    });
                    
                    // Set up playback state tracking
                    audioElement.addEventListener('play', () => {
                        this.playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
                    });
                    
                    audioElement.addEventListener('pause', () => {
                        this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
                    });
                    
                    audioElement.addEventListener('ended', () => {
                        this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
                        this.updateTimeDisplay(0);
                    });
                    
                    // Load audio and generate waveform
                    audioElement.addEventListener('loadedmetadata', () => {
                        console.log(`✅ Audio loaded: ${audioElement.duration}s`);
                        console.log('🎵 Waveform should now be visible!');
                        
                        // Set initial volume
                        audioElement.volume = 0.5;
                        
                        // Update total time display
                        const totalTime = document.getElementById('totalTime');
                        if (totalTime) {
                            totalTime.textContent = this.formatTime(audioElement.duration);
                        }
                    });
                    
                    audioElement.addEventListener('error', (e) => {
                        console.log(`⚠️ Audio loading error (continuing anyway): ${e.message || 'Unknown error'}`);
                    });
                    
                    audioElement.load();
                    resolve(peaksInstance);
                });
            });
            
        } catch (error) {
            console.error('❌ Waveform initialization failed:', error);
            console.warn('Continuing without waveform visualization');
            this.showEmptyWaveformState();
            return null;
        }
    }

    async initializeTimeline(audioBuffer) {
        try {
            console.log('🎵 Initializing timeline with audio buffer:', {
                audioBuffer: !!audioBuffer,
                duration: audioBuffer?.duration,
                sampleRate: audioBuffer?.sampleRate,
                numberOfChannels: audioBuffer?.numberOfChannels
            });
            
            // Check if Peaks.js is available using global flag and either export name
            const PeaksAPI = window.Peaks || window.peaks;
            if (!window.peaksAvailable || !PeaksAPI) {
                console.warn('⚠️ Peaks.js not available, skipping waveform visualization');
                console.log('Debug: peaksAvailable =', window.peaksAvailable, ', Peaks =', typeof window.Peaks, ', peaks =', typeof window.peaks);
                // Hide waveform container and show empty state
                this.showEmptyWaveformState();
                return null;
            }
            
            // Show waveform container and hide empty state
            this.showWaveformContainer();
            
            // Create new Timeline with container IDs
            this.timeline = new Timeline('overviewContainer', 'zoomviewContainer');
            
            // Load audio into timeline
            await this.timeline.loadAudio(audioBuffer);
            
            // Connect timeline to player
            this.waveformPlayer.connectTimeline(this.timeline);
            
            console.log('✅ Timeline initialized successfully');
            return this.timeline;
        } catch (error) {
            console.error('❌ Timeline initialization failed:', error);
            console.warn('Continuing without waveform visualization');
            this.showEmptyWaveformState();
            return null;
        }
    }
    
    showWaveformContainer() {
        const waveformContainer = document.getElementById('waveformContainer');
        const emptyWaveform = document.getElementById('emptyWaveform');
        
        if (waveformContainer) waveformContainer.style.display = 'block';
        if (emptyWaveform) emptyWaveform.style.display = 'none';
    }
    
    showEmptyWaveformState() {
        const waveformContainer = document.getElementById('waveformContainer');
        const emptyWaveform = document.getElementById('emptyWaveform');
        
        if (waveformContainer) waveformContainer.style.display = 'none';
        if (emptyWaveform) emptyWaveform.style.display = 'flex';
    }
    
    displayAnalysisResults(result) {
        this.currentFileAnalysis = result;
        console.log('Display analysis results:', result);
        
        // Extract file metadata from the result structure
        const fileInfo = result.file_metadata || {};
        const duration = result.duration_seconds || 0;
        
        // Update transport controls duration
        const totalTime = document.getElementById('totalTime');
        if (totalTime) {
            totalTime.textContent = this.formatTime(duration);
        }
        
        // Update musical properties from region analyses
        this.updateMusicPropertiesFromRegions(result);
        
        // Update region display
        this.updateRegionDisplay(result);
        
        // Add timeline segments to Peaks.js waveform
        this.addTimelineSegments();
        
        // Update properties panel with correct data structure
        this.updatePropertiesPanel(result);
    }
    
    updateMusicPropertiesFromRegions(result) {
        let tempo = null;
        let key = null;
        
        // Extract musical properties from region analyses - look inside 'data' field
        if (result.region_analyses && result.region_analyses.length > 0) {
            result.region_analyses.forEach(regionAnalysis => {
                // Look for tempo_meter_detector results
                if (regionAnalysis.plugin_results && regionAnalysis.plugin_results.tempo_meter_detector 
                    && regionAnalysis.plugin_results.tempo_meter_detector.data) {
                    const tempoResults = regionAnalysis.plugin_results.tempo_meter_detector.data;
                    if (tempoResults.tempo_bpm !== undefined) {
                        tempo = tempoResults.tempo_bpm;
                    }
                }
                
                // Look for key_finder results
                if (regionAnalysis.plugin_results && regionAnalysis.plugin_results.key_finder
                    && regionAnalysis.plugin_results.key_finder.data) {
                    const keyResults = regionAnalysis.plugin_results.key_finder.data;
                    if (keyResults.key && keyResults.key !== 'Unknown') {
                        key = keyResults.key + (keyResults.mode ? ' ' + keyResults.mode : '');
                    }
                }
            });
        }
        
        // Update tempo display
        if (tempo !== null && tempo !== undefined) {
            this.tempoDisplay.textContent = Math.round(tempo);
            this.tempoDisplay.parentElement.classList.remove('empty');
        } else {
            this.tempoDisplay.textContent = '--';
            this.tempoDisplay.parentElement.classList.add('empty');
        }
        
        // Update key display
        if (key) {
            this.keyDisplay.textContent = key;
            this.keyDisplay.parentElement.classList.remove('empty');
        } else {
            this.keyDisplay.textContent = '--';
            this.keyDisplay.parentElement.classList.add('empty');
        }
    }
    
    updateRegionDisplay(result) {
        // Clear existing regions
        this.regionManager.clearAll();
        
        // Add new regions
        if (result.regions && result.regions.length) {
            this.regionManager.addRegionLayer(result.regions);
        }
        
        // Add timeline segments if available - convert to classification format
        if (result.timeline_segments && result.timeline_segments.length) {
            const timelineClassifications = {};
            result.timeline_segments.forEach((segment, index) => {
                timelineClassifications[`timeline_seg_${index + 1}`] = {
                    classification: segment.content_type || segment.label || 'Unknown',
                    start_time: segment.start_time,
                    end_time: segment.end_time,
                    confidence: segment.confidence || 0.5
                };
            });
            this.regionManager.addClassificationLayer(timelineClassifications);
        }
    }
    
    addTimelineSegments() {
        console.log('🚀 addTimelineSegments() called!');
        
        if (!this.peaksInstance || !this.currentFileAnalysis) {
            console.log('⚠️ Cannot add segments: missing Peaks instance or analysis data');
            return;
        }
        
        // Clear existing segments
        this.peaksInstance.segments.removeAll();
        
        // Use EXACT same data extraction logic as properties panel
        let timelineClassifications = null;
        if (this.currentFileAnalysis.region_analyses && this.currentFileAnalysis.region_analyses.length > 0) {
            const classifierAnalysis = this.currentFileAnalysis.region_analyses.find(r => 
                r.plugin_results && r.plugin_results.classifier && r.plugin_results.classifier.data);
            
            if (classifierAnalysis) {
                timelineClassifications = classifierAnalysis.plugin_results.classifier.data;
            }
        }
        
        // Extract timeline segments using same logic as properties panel
        if (!timelineClassifications || !timelineClassifications.timeline_classifications) {
            console.log('⚠️ No timeline_classifications found');
            return;
        }
        
        const timelineSegs = timelineClassifications.timeline_classifications;
        console.log(`🎯 Found ${Object.keys(timelineSegs).length} timeline segments`);
        
        // TODO: Fix timeline interpolation - currently causing misalignment
        console.log('🔍 Timeline segments found:', Object.keys(timelineSegs).length);
        
        // For now, use original timing (without interpolation) to avoid random colors
        const segments = Object.entries(timelineSegs).map(([segmentId, segmentData]) => ({
            startTime: segmentData.time || 0,
            endTime: (segmentData.time || 0) + (segmentData.duration || 0),
            classification: segmentData.classification || 'Unknown',
            confidence: segmentData.confidence || 0,
            id: segmentId
        })).sort((a, b) => a.startTime - b.startTime);
        
        console.log('⚠️ Using original segment timing (interpolation disabled)');

        console.log(`🎯 Processing ${segments.length} raw segments for grouping...`);

        // Group consecutive segments of same type into regions
        const regions = [];
        let currentRegion = null;

        segments.forEach(segment => {
            if (!currentRegion || currentRegion.classification !== segment.classification) {
                // Start new region
                if (currentRegion) {
                    regions.push(currentRegion);
                }
                currentRegion = {
                    startTime: segment.startTime,
                    endTime: segment.endTime,
                    classification: segment.classification,
                    confidence: segment.confidence,
                    segmentCount: 1
                };
            } else {
                // Extend current region
                currentRegion.endTime = segment.endTime;
                currentRegion.confidence = Math.max(currentRegion.confidence, segment.confidence);
                currentRegion.segmentCount++;
            }
        });

        if (currentRegion) {
            regions.push(currentRegion);
        }

        console.log(`🎨 Grouped into ${regions.length} meaningful regions:`);
        regions.forEach(region => {
            console.log(`  📍 ${region.classification}: ${region.startTime}s-${region.endTime}s (${region.segmentCount} segments)`);
        });

        // Add regions as visual segments
        regions.forEach((region, index) => {
            const segmentConfig = {
                startTime: parseFloat(region.startTime),
                endTime: parseFloat(region.endTime),
                labelText: `${region.classification} (${this.formatTime(region.endTime - region.startTime)})`,
                color: this.getColorForContentType(region.classification.toLowerCase()),
                id: `region_${index}`,
                editable: false
            };

            try {
                this.peaksInstance.segments.add(segmentConfig);
                console.log(`✅ Region: ${region.classification} (${region.startTime}s-${region.endTime}s)`);
            } catch (error) {
                console.error(`❌ Failed to add region ${index}:`, error);
            }
        });

        // Clear existing points first
        if (this.peaksInstance.points) {
            this.peaksInstance.points.removeAll();
        }

        // Add transition markers between different content types
        for (let i = 1; i < regions.length; i++) {
            const transitionTime = regions[i].startTime;
            const fromType = regions[i-1].classification;
            const toType = regions[i].classification;
            
            try {
                this.peaksInstance.points.add({
                    time: transitionTime,
                    labelText: `${fromType} → ${toType}`,
                    color: '#FF6B6B',
                    id: `transition_${Date.now()}_${i}`  // Unique ID with timestamp
                });
                console.log(`🎯 Transition marker: ${fromType} → ${toType} at ${transitionTime}s`);
            } catch (error) {
                console.error(`❌ Failed to add transition marker:`, error);
            }
        }
        
        console.log(`🎨 Timeline segments added to waveform`);
        
        // Add click handler for segments
        this.peaksInstance.on('segments.click', (event) => {
            const segment = event.segment;
            console.log(`🎵 Clicked segment: ${segment.labelText}`);
            this.playSegment(segment.startTime, segment.endTime);
        });
    }
    
    playSegment(startTime, endTime) {
        if (!this.peaksInstance) return;
        
        console.log(`▶️ Playing segment: ${startTime}s - ${endTime}s`);
        
        // Seek to start time
        this.peaksInstance.player.seek(startTime);
        
        // Play the audio
        this.peaksInstance.player.play();
        
        // Stop at end time
        const stopTimer = setTimeout(() => {
            this.peaksInstance.player.pause();
            console.log('⏸️ Segment playback finished');
        }, (endTime - startTime) * 1000);
        
        // Store timer to clear if user stops manually
        this.currentSegmentTimer = stopTimer;
    }
    
    updatePropertiesPanel(result) {
        if (!this.propertiesPanel) return;
        
        let html = '';
        
        // File properties section - use correct data structure
        const fileMetadata = result.file_metadata || {};
        html += `
            <div class="properties-section">
                <h3>File Properties</h3>
                <div class="property-list">
                    <div class="property-item">
                        <span class="property-label">Path</span>
                        <span class="property-value">${result.file_path?.split('/').pop() || 'Unknown'}</span>
                    </div>
                    <div class="property-item">
                        <span class="property-label">Duration</span>
                        <span class="property-value">${this.formatTime(result.duration_seconds || 0)}</span>
                    </div>
                    <div class="property-item">
                        <span class="property-label">Regions</span>
                        <span class="property-value">${result.total_regions || 0}</span>
                    </div>
                    <div class="property-item">
                        <span class="property-label">Success</span>
                        <span class="property-value">${result.success ? 'Yes' : 'No'}</span>
                    </div>
                </div>
            </div>
        `;
        
        // Timeline Classification section - show timeline segments instead of generic classification
        let timelineClassifications = null;
        if (result.region_analyses && result.region_analyses.length > 0) {
            const classifierAnalysis = result.region_analyses.find(r => 
                r.plugin_results && r.plugin_results.classifier && r.plugin_results.classifier.data);
            
            if (classifierAnalysis) {
                timelineClassifications = classifierAnalysis.plugin_results.classifier.data;
            }
        }
        
        // Show timeline classification segments if available
        if (timelineClassifications && timelineClassifications.timeline_classifications) {
            html += `
                <div class="properties-section">
                    <h3>Timeline Classification</h3>
                    <div class="timeline-segments">
            `;
            
            // Show timeline segments with their classifications
            const timelineSegs = timelineClassifications.timeline_classifications;
            Object.entries(timelineSegs).forEach(([segmentId, segmentData]) => {
                const startTime = this.formatTime(segmentData.time || 0);
                const endTime = this.formatTime((segmentData.time || 0) + (segmentData.duration || 0));
                const classification = segmentData.classification || 'Unknown';
                const confidence = Math.round((segmentData.confidence || 0) * 100);
                const color = this.getColorForContentType(classification.toLowerCase());
                
                html += `
                    <div class="timeline-segment-item" data-segment-id="${segmentId}">
                        <div class="segment-color" style="background-color: ${color}"></div>
                        <div class="segment-info">
                            <span class="segment-type">${this.formatLabel(classification)}</span>
                            <span class="segment-time">${startTime} - ${endTime}</span>
                            <span class="segment-confidence">${confidence}%</span>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        } else if (timelineClassifications && timelineClassifications.classifications) {
            // Fallback to generic classification if no timeline segments
            html += `
                <div class="properties-section">
                    <h3>Classification</h3>
                    <div class="property-list">
            `;
            
            const classifications = timelineClassifications.classifications || [];
            const confidenceScores = timelineClassifications.confidence_scores || [];
            
            classifications.slice(0, 3).forEach((label, index) => {
                const confidence = confidenceScores[index] || 0;
                const percent = Math.round(confidence * 100);
                html += `
                    <div class="property-item">
                        <span class="property-label">${this.formatLabel(label)}</span>
                        <div class="confidence-bar">
                            <div class="confidence-value" style="width: ${percent}%"></div>
                            <span class="confidence-text">${percent}%</span>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }
        
        // Musical properties section - extract from region analyses
        let tempo = null;
        let key = null;
        let meter = null;
        
        if (result.region_analyses && result.region_analyses.length > 0) {
            result.region_analyses.forEach(regionAnalysis => {
                if (regionAnalysis.plugin_results) {
                    // Tempo from tempo_meter_detector - look inside 'data' field
                    if (regionAnalysis.plugin_results.tempo_meter_detector && regionAnalysis.plugin_results.tempo_meter_detector.data) {
                        const tempoResults = regionAnalysis.plugin_results.tempo_meter_detector.data;
                        if (tempoResults.tempo_bpm !== undefined) tempo = tempoResults.tempo_bpm;
                        if (tempoResults.meter) meter = tempoResults.meter;
                    }
                    
                    // Key from key_finder - look inside 'data' field
                    if (regionAnalysis.plugin_results.key_finder && regionAnalysis.plugin_results.key_finder.data) {
                        const keyResults = regionAnalysis.plugin_results.key_finder.data;
                        if (keyResults.key && keyResults.key !== 'Unknown') {
                            key = keyResults.key + (keyResults.mode ? ' ' + keyResults.mode : '');
                        }
                    }
                }
            });
        }
        
        if (tempo || key || meter) {
            html += `
                <div class="properties-section">
                    <h3>Musical Properties</h3>
                    <div class="property-list">
            `;
            
            if (tempo) {
                html += `
                    <div class="property-item">
                        <span class="property-label">Tempo</span>
                        <span class="property-value">${Math.round(tempo)} BPM</span>
                    </div>
                `;
            }
            
            if (meter) {
                html += `
                    <div class="property-item">
                        <span class="property-label">Meter</span>
                        <span class="property-value">${meter}</span>
                    </div>
                `;
            }
            
            if (key) {
                html += `
                    <div class="property-item">
                        <span class="property-label">Key</span>
                        <span class="property-value">${key}</span>
                    </div>
                `;
            }
            
            html += `
                    </div>
                </div>
            `;
        }
        
        // Regions summary
        if (result.regions && result.regions.length > 0) {
            html += `
                <div class="properties-section">
                    <h3>Audio Regions</h3>
                    <div class="timeline-segments">
            `;
            
            result.regions.forEach((region, index) => {
                const duration = region.end_time - region.start_time;
                html += `
                    <div class="timeline-segment-item" data-region-id="${region.id || index}">
                        <div class="segment-color" style="background-color: ${this.getColorForContentType(region.type)}"></div>
                        <div class="segment-info">
                            <span class="segment-type">${region.type || region.label || 'Region'}</span>
                            <span class="segment-time">${this.formatTime(region.start_time)} - ${this.formatTime(region.end_time)}</span>
                            <span class="segment-duration">${this.formatTime(duration)}</span>
                        </div>
                    </div>
                `;
            });
            
            html += `
                    </div>
                </div>
            `;
        }
        
        // System status section
        html += `
            <div class="properties-section">
                <h3>System Status</h3>
                <div class="property-list">
                    <div class="property-item">
                        <span class="property-label">Engine</span>
                        <span class="property-value" id="engineStatus">READY</span>
                    </div>
                    <div class="property-item">
                        <span class="property-label">Plugins</span>
                        <span class="property-value" id="pluginStatus">5 LOADED</span>
                    </div>
                    <div class="property-item">
                        <span class="property-label">Processing Time</span>
                        <span class="property-value">${result.duration_seconds?.toFixed(2) || 0}s</span>
                    </div>
                </div>
            </div>
        `;
        
        this.propertiesPanel.innerHTML = html;
        
        // Add event listeners to timeline segment items
        document.querySelectorAll('.timeline-segment-item').forEach(item => {
            item.addEventListener('click', () => {
                const segmentId = item.dataset.segmentId;
                const regionId = item.dataset.regionId;
                if (segmentId) {
                    this.selectTimelineSegment(segmentId);
                } else if (regionId) {
                    this.selectRegion(regionId);
                }
            });
        });
    }
    
    selectRegion(regionId) {
        console.log('Region selected:', regionId);
        
        // Highlight selected region
        document.querySelectorAll('.timeline-segment-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.regionId === regionId);
        });
        
        // Could add more region selection logic here
    }
    
    selectTimelineSegment(segmentId) {
        if (!this.currentFileAnalysis || !this.currentFileAnalysis.timeline_segments) return;
        
        const segment = this.currentFileAnalysis.timeline_segments.find(s => s.id === segmentId);
        if (!segment) return;
        
        // Highlight in UI
        document.querySelectorAll('.timeline-segment-item').forEach(item => {
            item.classList.toggle('selected', item.dataset.segmentId === segmentId);
        });
        
        // Scroll timeline to segment
        if (this.timeline) {
            this.timeline.scrollToTime(segment.start_time);
        }
        
        // Optionally play segment
        // this.playSegment(segment.start_time, segment.end_time);
    }
    
    onRegionSelected(region) {
        console.log('Region selected:', region);
        // Implementation depends on your region structure
    }
    
    onTimelineSegmentSelected(timelineSegment) {
        console.log('Timeline segment selected:', timelineSegment);
        
        // Find corresponding segment in data
        if (this.currentFileAnalysis && this.currentFileAnalysis.timeline_segments) {
            const segmentId = timelineSegment.id.replace('timeline_', '');
            this.selectTimelineSegment(segmentId);
        }
    }
    
    togglePlayback() {
        // Clear any segment timer
        if (this.currentSegmentTimer) {
            clearTimeout(this.currentSegmentTimer);
            this.currentSegmentTimer = null;
        }
        
        if (!this.peaksInstance) {
            this.showError('Player not initialized');
            return;
        }
        
        if (this.peaksInstance.player.isPlaying()) {
            this.peaksInstance.player.pause();
            this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        } else {
            this.peaksInstance.player.play();
            this.playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
        }
    }
    
    stopPlayback() {
        // Clear any segment timer
        if (this.currentSegmentTimer) {
            clearTimeout(this.currentSegmentTimer);
            this.currentSegmentTimer = null;
        }
        
        if (!this.peaksInstance) {
            this.showError('Player not initialized');
            return;
        }
        
        this.peaksInstance.player.pause();
        this.peaksInstance.player.seek(0);
        this.playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
        
        // Update time display
        this.updateTimeDisplay(0);
    }
    
    toggleMute() {
        if (!this.peaksInstance) return;
        
        const audioElement = this.peaksInstance.options.mediaElement;
        const muteBtn = document.getElementById('muteBtn');
        const volumeSlider = document.getElementById('volumeSlider');
        
        if (audioElement.muted) {
            audioElement.muted = false;
            muteBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
            volumeSlider.disabled = false;
        } else {
            audioElement.muted = true;
            muteBtn.innerHTML = '<i class="fas fa-volume-mute"></i>';
            volumeSlider.disabled = true;
        }
    }
    
    setVolume(volume) {
        if (!this.peaksInstance) return;
        
        const audioElement = this.peaksInstance.options.mediaElement;
        audioElement.volume = Math.max(0, Math.min(1, volume));
        
        // Update mute button based on volume
        const muteBtn = document.getElementById('muteBtn');
        if (volume === 0) {
            muteBtn.innerHTML = '<i class="fas fa-volume-mute"></i>';
        } else if (volume < 0.5) {
            muteBtn.innerHTML = '<i class="fas fa-volume-down"></i>';
        } else {
            muteBtn.innerHTML = '<i class="fas fa-volume-up"></i>';
        }
    }
    
    updateTimeDisplay(currentTime = null) {
        const currentTimeDisplay = document.getElementById('currentTime');
        if (currentTimeDisplay) {
            const time = currentTime !== null ? currentTime : 
                        (this.peaksInstance ? this.peaksInstance.player.getCurrentTime() : 0);
            currentTimeDisplay.textContent = this.formatTime(time);
        }
    }
    
    async openDatabaseBrowser() {
        this.databaseModal.style.display = 'block';
        this.loadDatabaseContents();
    }
    
    async loadDatabaseContents(searchQuery = '') {
        try {
            this.databaseResults.innerHTML = '<div class="loading">Loading database...</div>';
            
            const url = searchQuery
                ? `${this.apiBase}/database/files?query=${encodeURIComponent(searchQuery)}`
                : `${this.apiBase}/database/files`;
            
            const response = await fetch(url);
            
            if (!response.ok) {
                throw new Error(`Database query failed: ${response.status} ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (!data || !data.files || !data.files.length) {
                this.databaseResults.innerHTML = '<div class="empty-state">No files found</div>';
                return;
            }
            
            // Render files
            let html = '';
            
            data.files.forEach(file => {
                const fileDate = new Date(file.date_analyzed || file.date_added || null);
                const formattedDate = fileDate instanceof Date && !isNaN(fileDate)
                    ? fileDate.toLocaleDateString() + ' ' + fileDate.toLocaleTimeString()
                    : 'Unknown';
                
                html += `
                    <div class="db-file-item" data-path="${file.path}">
                        <div class="db-file-header">
                            <span class="db-file-name">${file.filename || file.path.split('/').pop()}</span>
                            <span class="db-file-date">${formattedDate}</span>
                        </div>
                        <div class="db-file-details">
                            <span class="db-file-duration">${this.formatTime(file.duration || 0)}</span>
                            <span class="db-file-key">${file.key || '-'}</span>
                            <span class="db-file-tempo">${file.tempo ? Math.round(file.tempo) + ' BPM' : '-'}</span>
                        </div>
                    </div>
                `;
            });
            
            this.databaseResults.innerHTML = html;
            
            // Add click handlers
            document.querySelectorAll('.db-file-item').forEach(item => {
                item.addEventListener('click', () => {
                    const path = item.dataset.path;
                    this.loadDatabaseFile(path);
                });
            });
        } catch (error) {
            console.error('Database loading failed:', error);
            this.databaseResults.innerHTML = `<div class="error-state">Error: ${error.message}</div>`;
        }
    }
    
    searchDatabase(query) {
        if (query.length === 0) {
            this.loadDatabaseContents();
            return;
        }
        
        // Debounce search
        clearTimeout(this._searchTimeout);
        this._searchTimeout = setTimeout(() => {
            this.loadDatabaseContents(query);
        }, 300);
    }
    
    async loadDatabaseFile(path) {
        try {
            this.databaseModal.style.display = 'none';
            this.showLoading(`Loading ${path.split('/').pop()}...`);
            
            // Set file path in upload form
            this.uploadPathInput.value = path;
            
            // Process the file
            await this.handleFileUpload();
        } catch (error) {
            console.error('Database file loading failed:', error);
            this.showError(`Failed to load file: ${error.message}`);
            this.hideLoading();
        }
    }
    
    getColorForContentType(type) {
        const colors = {
            'speech': '#4285F4',  // Google Blue
            'music': '#34A853',   // Google Green
            'thunder': '#FBBC05', // Google Yellow
            'noise': '#EA4335',   // Google Red
            'silence': '#9AA0A6', // Google Gray
            'rain': '#4285F4',    // Blue
            'wind': '#5F6368',    // Dark Gray
            'ambient': '#34A853', // Green
            'sfx': '#FBBC05',     // Yellow
            'drums': '#EA4335',   // Red
            'melody': '#4285F4',  // Blue
            'bass': '#34A853'     // Green
        };
        
        return colors[type?.toLowerCase()] || '#9AA0A6'; // Default to gray
    }
    
    showLoading(message = 'Loading...') {
        this.loadingOverlay.style.display = 'flex';
        this.loadingText.textContent = message;
    }
    
    updateLoadingText(message) {
        this.loadingText.textContent = message;
    }
    
    hideLoading() {
        this.loadingOverlay.style.display = 'none';
    }
    
    showError(message) {
        alert(message); // Simple error display, could be enhanced
    }
    
    formatTime(seconds) {
        if (!seconds && seconds !== 0) return '--:--';
        
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins}:${secs.toString().padStart(2, '0')}`;
    }
    
    formatLabel(label) {
        return label
            .replace(/_/g, ' ')
            .replace(/\b\w/g, c => c.toUpperCase());
    }
}

// Export the AudioSamplerApp class for external initialization
export { AudioSamplerApp };

// Stub component classes (minimal implementations)
class Timeline {
    constructor() {
        this.isInitialized = false;
    }
}

class RegionManager {
    constructor(containerId) {
        this.containerId = containerId;
    }
    
    clearAll() {
        console.log('RegionManager: clearAll called');
    }
    
    addRegionLayer(regions) {
        console.log('RegionManager: addRegionLayer called with', regions?.length || 0, 'regions');
    }
    
    addClassificationLayer(classifications) {
        console.log('RegionManager: addClassificationLayer called');
    }
}

class AudioLoader {
    constructor(apiBase) {
        this.apiBase = apiBase;
    }
    
    async getAudioBufferFromFile(file) {
        console.log('AudioLoader: getAudioBufferFromFile called');
        return null; // Not needed for current implementation
    }
}

class WaveformPlayer {
    constructor() {
        this.playing = false;
    }
    
    connectTimeline(timeline) {
        console.log('WaveformPlayer: connectTimeline called');
    }
    
    isPlaying() {
        return this.playing;
    }
    
    play() {
        this.playing = true;
    }
    
    pause() {
        this.playing = false;
    }
    
    stop() {
        this.playing = false;
    }
}
