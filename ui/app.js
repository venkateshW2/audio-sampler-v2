// Audio Intelligence Sampler v2 - Frontend JavaScript

class AudioSamplerUI {
    constructor() {
        this.apiBase = '';  // Same origin
        this.init();
    }

    async init() {
        this.setupEventListeners();
        await this.checkSystemStatus();
        await this.loadDatabaseStats();
    }

    setupEventListeners() {
        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => fileInput.click());
        uploadArea.addEventListener('dragover', this.handleDragOver.bind(this));
        uploadArea.addEventListener('dragleave', this.handleDragLeave.bind(this));
        uploadArea.addEventListener('drop', this.handleDrop.bind(this));
        
        fileInput.addEventListener('change', this.handleFileSelect.bind(this));
        
        // Path analysis
        document.getElementById('analyzePathBtn').addEventListener('click', this.analyzeFilePath.bind(this));
        document.getElementById('filePath').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.analyzeFilePath();
        });
        
        // Database search
        document.getElementById('searchBtn').addEventListener('click', this.searchFiles.bind(this));
        document.getElementById('searchQuery').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.searchFiles();
        });
    }

    async checkSystemStatus() {
        try {
            const response = await fetch(`${this.apiBase}/health`);
            const data = await response.json();
            
            if (response.ok) {
                this.updateStatusIndicator('healthy', 'System Ready');
                this.updateSystemStatus(data);
            } else {
                this.updateStatusIndicator('error', 'System Error');
                this.updateSystemStatus(null, data.detail);
            }
        } catch (error) {
            this.updateStatusIndicator('error', 'Connection Failed');
            this.updateSystemStatus(null, 'Unable to connect to server');
        }
    }

    updateStatusIndicator(status, text) {
        const indicator = document.getElementById('statusIndicator');
        const dot = indicator.querySelector('.status-dot');
        const textEl = document.getElementById('statusText');
        
        dot.className = `status-dot ${status}`;
        textEl.textContent = text;
    }

    updateSystemStatus(data, error = null) {
        const elements = {
            engineStatus: document.getElementById('engineStatus'),
            pluginStatus: document.getElementById('pluginStatus'),
            databaseStatus: document.getElementById('databaseStatus'),
            gpuStatus: document.getElementById('gpuStatus')
        };

        if (error) {
            Object.values(elements).forEach(el => {
                el.textContent = 'Error';
                el.style.color = '#f44336';
            });
            return;
        }

        if (data) {
            elements.engineStatus.textContent = data.engine ? 
                `Ready (${data.engine.plugins_registered} plugins)` : 'Not Ready';
            elements.engineStatus.style.color = data.engine ? '#4CAF50' : '#f44336';

            elements.pluginStatus.textContent = data.engine ? 
                `${data.engine.plugins_registered} registered` : 'Not loaded';
            elements.pluginStatus.style.color = data.engine?.plugins_registered > 0 ? '#4CAF50' : '#ff9800';

            elements.databaseStatus.textContent = data.engine?.components_ready ? 'Connected' : 'Unavailable';
            elements.databaseStatus.style.color = data.engine?.components_ready ? '#4CAF50' : '#f44336';

            elements.gpuStatus.textContent = data.system?.gpu_available ? 'Available' : 'Not Available';
            elements.gpuStatus.style.color = data.system?.gpu_available ? '#4CAF50' : '#ff9800';
        }
    }

    async loadDatabaseStats() {
        try {
            const response = await fetch(`${this.apiBase}/database/stats`);
            const stats = await response.json();
            
            const statsEl = document.getElementById('databaseStats');
            if (response.ok) {
                statsEl.innerHTML = `
                    <strong>Database Statistics:</strong><br>
                    Files: ${stats.total_files} | 
                    Regions: ${stats.total_regions} | 
                    Analysis Results: ${stats.total_analysis_results}
                `;
            } else {
                statsEl.innerHTML = '<span style="color: #f44336;">Database unavailable</span>';
            }
        } catch (error) {
            document.getElementById('databaseStats').innerHTML = 
                '<span style="color: #f44336;">Failed to load database statistics</span>';
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.add('dragover');
    }

    handleDragLeave(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
    }

    handleDrop(e) {
        e.preventDefault();
        document.getElementById('uploadArea').classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    handleFileSelect(e) {
        const files = e.target.files;
        if (files.length > 0) {
            this.processFile(files[0]);
        }
    }

    async processFile(file) {
        // Validate file type
        const validTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/mp4', 'audio/x-m4a'];
        const validExtensions = ['.wav', '.mp3', '.flac', '.m4a'];
        
        const isValidType = validTypes.some(type => file.type.includes(type)) ||
                           validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
        
        if (!isValidType) {
            this.showError('Please select a valid audio file (.wav, .mp3, .flac, .m4a)');
            return;
        }

        this.showProgress('Uploading file...');

        try {
            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch(`${this.apiBase}/api/analyze/upload`, {
                method: 'POST',
                body: formData
            });

            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            let result;
            
            if (contentType && contentType.includes('application/json')) {
                result = await response.json();
            } else {
                // If not JSON, get text response (likely an error page)
                const textResponse = await response.text();
                console.error('Non-JSON response:', textResponse);
                throw new Error(`Server returned non-JSON response (${response.status})`);
            }

            if (response.ok) {
                this.hideProgress();
                this.displayResults(result.analysis_result, `Upload: ${file.name}`);
                await this.loadDatabaseStats(); // Refresh stats
            } else {
                this.hideProgress();
                this.showError(result.detail || `Upload failed with status ${response.status}`);
            }
        } catch (error) {
            this.hideProgress();
            console.error('Upload error details:', error);
            this.showError(`Upload error: ${error.message}`);
        }
    }

    async analyzeFilePath() {
        const filePath = document.getElementById('filePath').value.trim();
        
        if (!filePath) {
            this.showError('Please enter a file path');
            return;
        }

        this.showProgress('Analyzing file...');

        try {
            const response = await fetch(`${this.apiBase}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ file_path: filePath })
            });

            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            let result;
            
            if (contentType && contentType.includes('application/json')) {
                result = await response.json();
            } else {
                // If not JSON, get text response (likely an error page)
                const textResponse = await response.text();
                console.error('Non-JSON response:', textResponse);
                throw new Error(`Server returned non-JSON response (${response.status})`);
            }

            if (response.ok) {
                this.hideProgress();
                this.displayResults(result.analysis_result, filePath);
                await this.loadDatabaseStats(); // Refresh stats
            } else {
                this.hideProgress();
                this.showError(result.detail || `Analysis failed with status ${response.status}`);
            }
        } catch (error) {
            this.hideProgress();
            console.error('Analysis error details:', error);
            this.showError(`Analysis error: ${error.message}`);
        }
    }

    async searchFiles() {
        const query = document.getElementById('searchQuery').value.trim();
        
        try {
            const url = new URL(`${window.location.origin}/database/files`);
            if (query) url.searchParams.set('query', query);
            url.searchParams.set('limit', '20');

            const response = await fetch(url);
            const result = await response.json();

            const filesList = document.getElementById('filesList');

            if (response.ok && result.files.length > 0) {
                filesList.innerHTML = result.files.map(file => `
                    <div class="file-item" onclick="ui.viewFileDetails('${file.path}')">
                        <div class="file-name">${file.path.split('/').pop()}</div>
                        <div class="file-meta">
                            <span>Duration: ${file.duration_seconds?.toFixed(1)}s</span>
                            <span>Regions: ${file.region_count}</span>
                            <span>Status: ${file.status}</span>
                            <span>Added: ${new Date(file.date_added).toLocaleDateString()}</span>
                        </div>
                    </div>
                `).join('');
            } else if (response.ok) {
                filesList.innerHTML = '<p>No files found</p>';
            } else {
                filesList.innerHTML = '<p style="color: #f44336;">Search failed</p>';
            }
        } catch (error) {
            document.getElementById('filesList').innerHTML = 
                '<p style="color: #f44336;">Search error</p>';
        }
    }

    async viewFileDetails(filePath) {
        try {
            const response = await fetch(`${this.apiBase}/database/files/${encodeURIComponent(filePath)}`);
            const result = await response.json();

            if (response.ok) {
                this.displayDatabaseResults(result, filePath);
            } else {
                this.showError('Failed to load file details');
            }
        } catch (error) {
            this.showError(`Error loading file details: ${error.message}`);
        }
    }

    showProgress(text) {
        const progressSection = document.getElementById('progressSection');
        const progressText = document.getElementById('progressText');
        
        progressSection.style.display = 'block';
        progressText.textContent = text;
        
        // Animate progress bar
        const progressFill = document.getElementById('progressFill');
        progressFill.style.width = '0%';
        setTimeout(() => progressFill.style.width = '90%', 100);
    }

    hideProgress() {
        const progressSection = document.getElementById('progressSection');
        const progressFill = document.getElementById('progressFill');
        
        progressFill.style.width = '100%';
        setTimeout(() => {
            progressSection.style.display = 'none';
            progressFill.style.width = '0%';
        }, 500);
    }

    showError(message) {
        const resultsPanel = document.getElementById('resultsPanel');
        const resultsContent = document.getElementById('resultsContent');
        
        resultsContent.innerHTML = `<div class="error-message">${message}</div>`;
        resultsPanel.style.display = 'block';
    }

    displayResults(result, filename) {
        const resultsPanel = document.getElementById('resultsPanel');
        const resultsContent = document.getElementById('resultsContent');
        
        if (!result.success) {
            this.showError('Analysis failed: ' + (result.errors.join(', ') || 'Unknown error'));
            return;
        }

        let html = `
            <div class="success-message">
                Analysis completed for: <strong>${filename}</strong>
            </div>
            
            <div class="result-category">
                <h3>File Information</h3>
                <div class="result-item">
                    <span class="result-label">Duration:</span>
                    <span class="result-value">${result.file_metadata?.final_duration?.toFixed(1) || 'Unknown'}s</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Sample Rate:</span>
                    <span class="result-value">${result.file_metadata?.final_sample_rate || 'Unknown'} Hz</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Regions Found:</span>
                    <span class="result-value">${result.total_regions}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Processing Time:</span>
                    <span class="result-value">${result.duration_seconds?.toFixed(2)}s</span>
                </div>
            </div>
        `;

        // Display results for each region
        result.region_analyses?.forEach((region, index) => {
            html += `<div class="result-category">
                <h3>Region ${index + 1} (${region.region_metadata.duration.toFixed(1)}s)</h3>`;

            Object.entries(region.plugin_results).forEach(([pluginName, pluginResult]) => {
                if (pluginResult.success) {
                    html += this.formatPluginResult(pluginName, pluginResult.data);
                }
            });

            html += '</div>';
        });

        resultsContent.innerHTML = html;
        resultsPanel.style.display = 'block';
        
        // Scroll to results
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }

    displayDatabaseResults(result, filename) {
        const resultsPanel = document.getElementById('resultsPanel');
        const resultsContent = document.getElementById('resultsContent');
        
        let html = `
            <div class="success-message">
                Database results for: <strong>${filename}</strong>
            </div>
            
            <div class="result-category">
                <h3>File Summary</h3>
                <div class="result-item">
                    <span class="result-label">Total Regions:</span>
                    <span class="result-value">${result.analysis_summary.total_regions}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Plugins Used:</span>
                    <span class="result-value">${result.analysis_summary.plugins_used.join(', ')}</span>
                </div>
                <div class="result-item">
                    <span class="result-label">Processing Time:</span>
                    <span class="result-value">${(result.analysis_summary.total_processing_time_ms / 1000).toFixed(2)}s</span>
                </div>
            </div>
        `;

        // Display region results
        result.regions?.forEach(region => {
            html += `<div class="result-category">
                <h3>${region.region_id}</h3>`;

            Object.entries(region.analysis_results).forEach(([pluginName, pluginResult]) => {
                if (pluginResult.success) {
                    html += this.formatPluginResult(pluginName, pluginResult);
                }
            });

            html += '</div>';
        });

        resultsContent.innerHTML = html;
        resultsPanel.style.display = 'block';
        resultsPanel.scrollIntoView({ behavior: 'smooth' });
    }

    formatPluginResult(pluginName, data) {
        let html = `<h4 style="margin: 16px 0 8px 0; color: #667eea;">${pluginName}</h4>`;

        switch (pluginName) {
            case 'classifier':
                html += `
                    <div class="result-item">
                        <span class="result-label">Top Classification:</span>
                        <span class="result-value">${data.top_prediction}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Confidence:</span>
                        <span class="confidence">${(data.top_confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Speech Probability:</span>
                        <span class="result-value">${(data.speech_probability * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Music Probability:</span>
                        <span class="result-value">${(data.music_probability * 100).toFixed(1)}%</span>
                    </div>
                `;
                break;

            case 'tempo_meter_detector':
                html += `
                    <div class="result-item">
                        <span class="result-label">Tempo:</span>
                        <span class="result-value">${data.tempo_bpm?.toFixed(1)} BPM</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Meter:</span>
                        <span class="result-value">${data.meter?.time_signature || 'Unknown'}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Confidence:</span>
                        <span class="confidence">${((data.tempo_confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                `;
                break;

            case 'key_finder':
                html += `
                    <div class="result-item">
                        <span class="result-label">Key:</span>
                        <span class="result-value">${data.key} ${data.mode}</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Confidence:</span>
                        <span class="confidence">${(data.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div class="result-item">
                        <span class="result-label">Tonality:</span>
                        <span class="result-value">${data.tonality?.is_tonal ? 'Tonal' : 'Atonal'}</span>
                    </div>
                `;
                break;

            default:
                html += `<div class="result-item">
                    <span class="result-label">Result:</span>
                    <span class="result-value">${JSON.stringify(data, null, 2)}</span>
                </div>`;
        }

        return html;
    }
}

// Initialize the UI when the page loads
const ui = new AudioSamplerUI();