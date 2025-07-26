
## Project Overview
**Audio Intelligence Sampler v2** is a professional audio analysis tool for musicians to intelligently sample and catalog found sounds, field recordings, and musical content. Currently in **Phase 1: Core Backend Development** with a complete PyTorch-only audio processing pipeline.

## Architecture Philosophy
- **Sacred Architecture**: Never-crash plugin system with region-based processing
- **Region-based Processing**: Audio is split into silence-detected regions, each processed independently
- **Plugin System**: Modular architecture where plugins can fail safely without crashing the engine
- **GPU-Optimized**: Designed for GTX 1060 with 6GB VRAM limit
- **Dual Environment**: Solves NumPy 2.x vs Essentia compatibility via subprocess bridge

## Dual Environment Architecture

### Environment 1: Main Environment (audio-sampler-v2)
- **Location**: `/mnt/2w12-data/audio-sampler-v2/miniconda/envs/audio-sampler-v2/`
- **Purpose**: Core PyTorch stack with NumPy 2.x
- **Contains**: PyTorch 2.5.1, librosa, madmom, PaSST, FastAPI, NumPy 2.x
- **Limitation**: Cannot use Essentia (NumPy 2.x incompatibility)

### Environment 2: Experimental Environment (audio-sampler-experimental)
- **Location**: `/mnt/2w12-data/audio-sampler-v2/miniconda/envs/audio-sampler-experimental/`
- **Purpose**: Essentia-only environment with NumPy 1.x
- **Contains**: Essentia 2.1b6.dev374 + NumPy 1.24.3
- **Access**: Via subprocess bridge only
- **Accuracy**: 89.5% key detection vs 65% LibROSA fallback

## Technology Stack

### Backend Stack
- **ML Framework**: PyTorch 2.5.1 with CUDA 12.1 support
- **Audio Processing**: librosa 0.10.1, madmom 0.16.1, pedalboard 0.9.0, basic-pitch
- **Backend Framework**: FastAPI 0.104.1 with uvicorn[standard] 0.24.0
- **Database**: SQLite with SQLAlchemy ORM
- **GPU Libraries**: nnAudio 0.2.6 for GPU-accelerated audio processing
- **Essentia Integration**: Subprocess bridge for 89.5% accuracy key detection

### Frontend Stack
- **Waveform Visualization**: Peaks.js with Konva.js and WaveformData.js
- **UI Framework**: Vanilla ES6 modules with flexbox layout
- **Styling**: CSS Grid/Flexbox with Google Fonts (Inter, JetBrains Mono)
- **Icons**: Font Awesome 6.0.0

## Core Components

### 1. Engine (`core/engine.py`)
- **AudioSamplerEngine**: Main orchestrator that never crashes
- **Region-based Processing**: Splits audio into content regions via silence detection
- **Plugin Lifecycle**: Safe plugin registration and execution
- **Database Integration**: Automatic storage of analysis results

### 2. File Handler (`core/file_handler.py`)
- **Smart Loading**: Handles WAV, MP3, FLAC, M4A formats
- **Silence Detection**: Automatically splits audio into meaningful regions
- **Resampling**: Standardizes to 22.05kHz mono for processing
- **Memory Management**: Efficient handling of large audio files

### 3. Waveform Generator (`core/waveform_generator.py`)
- **Backend Generation**: Creates visualization data using librosa
- **Dual Resolution**: 1000 samples (overview) + 8000 samples (zoom)
- **Peak Preservation**: Maintains visual fidelity with downsampling
- **RMS Analysis**: Provides contrast for better visualization

### 4. Database Models (`database/models.py`)
- **File**: Audio file metadata and processing status
- **Region**: Silence-detected content regions
- **RegionAnalysis**: Plugin results per region (JSON storage)
- **FileSummary**: Aggregated file-level results
- **Tags**: User and automatic tags

## Plugin System

### Core Plugins (5 registered in order):
1. **FeatureExtractorPlugin** - Extracts audio features for shared use
2. **ContentAnalysisPlugin** - Timeline-based content segmentation
3. **ClassifierPlugin** - PaSST-based audio classification with GTX 1060 optimization
4. **TempoMeterDetectorPlugin** - BPM and time signature detection
5. **KeyFinderPlugin** - **DUAL ENVIRONMENT** key detection (Essentia + LibROSA)

### Plugin Capabilities:
- **Audio Classification**: Music vs speech, genres, instruments
- **Timeline Analysis**: 19-segment timeline with confidence scores
- **Musical Properties**: Tempo (BPM), key, time signature
- **Content Detection**: Thunder, singing, music transitions
- **Key Detection**: 89.5% accuracy via Essentia subprocess bridge

## API Endpoints

### Main Endpoints (`main.py`):
- `GET /` - Root info and status
- `GET /health` - Health check for monitoring
- `GET /status` - Detailed engine status
- `POST /analyze` - Analyze file by path
- `GET /plugins` - List registered plugins
- `GET /database/files` - Search database files
- `GET /database/files/{path}` - Get specific file results
- `GET /database/stats` - Database statistics

### Extended API (`api/server.py`):
- `POST /api/analyze/upload` - Upload and analyze audio file
- `GET /api/processed_audio/{path}` - Serve processed audio files (universal support)
- `GET /api/audio/{file_id}` - Serve database audio files by ID (Phase 1 addition)
- `GET /api/system/resources` - System resource status
- `POST /api/system/cleanup` - Force resource cleanup

## Database Schema

### Tables:
- **files**: Audio file metadata and processing status
- **regions**: Silence-detected content regions
- **region_analysis**: Plugin results per region (JSON storage)
- **file_summary**: Aggregated file-level results
- **tags**: User and automatic tags

## Dual Environment Setup

### Installation Requirements:

#### **Main Environment Setup**:
```bash
# Create main environment
./miniconda/bin/conda create -n audio-sampler-v2 python=3.9
source ./miniconda/bin/activate audio-sampler-v2

# Install PyTorch stack
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
````

#### __Experimental Environment Setup__:

```bash
# Create experimental environment for Essentia
./miniconda/bin/conda create -n audio-sampler-experimental python=3.9
source ./miniconda/envs/audio-sampler-experimental/bin/activate

# Install Essentia with NumPy 1.x compatibility
pip install essentia numpy==1.24.3
```

### Python Dependencies:

__Main Environment (NumPy 2.x + PyTorch)__:

```bash
torch==2.5.1
torchaudio==2.5.1
librosa==0.10.1
soundfile>=0.12.1
scipy>=1.11.4
numpy>=2.0.0
fastapi==0.104.1
uvicorn[standard]==0.24.0
sqlalchemy>=1.4.0
pedalboard>=0.9.0
basic-pitch
madmom==0.16.1
nnAudio>=0.2.6
```

__Experimental Environment (NumPy 1.x + Essentia)__:

```bash
essentia==2.1b6.dev374
numpy==1.24.3
```

## Subprocess Bridge Architecture

### __Key Detection Process Flow__:

1. __Plugin Initialization__: KeyFinderPlugin checks both environments
2. __Primary Attempt__: Experimental environment via subprocess
3. __Fallback__: LibROSA in main environment if Essentia fails
4. __Communication__: Pickle-serialized numpy arrays between environments

### __Communication Format__:

#### __Input to Experimental Environment__:

```python
{
    'audio_data': np.ndarray,  # Audio samples
    'sample_rate': int         # Sample rate
}
```

#### __Output from Experimental Environment__:

```python
{
    'success': bool,
    'key': str,           # Detected key (C, C#, D, etc.)
    'mode': str,          # major/minor/atonal
    'confidence': float,  # 0.0-1.0
    'analysis_metadata': {
        'method': 'Essentia-NNLSChroma',
        'environment': 'experimental',
        'accuracy_rate': 0.895,
        'essentia_version': '2.1b6.dev374'
    }
}
```

## Current Status - Phase 1 Complete

### ✅ Completed Features:
- **Dual environment architecture** fully operational
- **Core backend engine** with region processing
- **All 5 core plugins** working (including Essentia bridge)
- **Database integration** with full CRUD
- **Waveform generation** backend
- **Peaks.js frontend** integration
- **Timeline visualization** with silence offset correction
- **File upload** and path-based analysis
- **Database browser** with search
- **Backend audio serving** endpoints for universal file access

### ⚠️ Known Issues (Phase 1.5 Targets):
- **Plugin Communication**: ContentAnalysisPlugin → ClassifierPlugin data handoff issues
- **Classification Processing**: Timeline vs traditional 10-second fallback confusion
- **Performance**: Large file processing optimization needed
- **Error Handling**: Plugin failure recovery improvements needed

## Usage Examples

### Basic Analysis:

```bash
# Start server
python main.py

# Upload via web UI
http://localhost:8000/ui/

# API analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"file_path": "/path/to/audio.wav"}'
```

### Testing Dual Environment:

```bash
# Test the complete pipeline
python test_dual_environment.py

# Test Essentia subprocess directly
python test_experimental_key_detection.py

# Test with real audio
python test_real_audio.py
```

### Database Operations:

```bash
# Search files
curl http://localhost:8000/database/files?query=music

# Get file results
curl http://localhost:8000/database/files/path/to/audio.wav

# Get database stats
curl http://localhost:8000/database/stats
```

## Architecture Summary

The **Audio Intelligence Sampler v2** successfully implements a **dual environment architecture** that resolves the NumPy 2.x vs Essentia compatibility issue while maintaining the sacred "never crash" principle. Phase 1 completed with 89.5% accuracy key detection via Essentia subprocess bridge and comprehensive audio analysis capabilities.

---

# PHASE 1.5: BACKEND REFINEMENT ROADMAP

## 🎯 **MISSION: FROM WORKING TO PRODUCTION-GRADE**

Phase 1 achieved "working" status. Phase 1.5 transforms this into a **bulletproof, production-grade backend** by systematically refining each component based on real-world usage patterns and identified issues.

## 🔍 **CRITICAL ISSUES IDENTIFIED**

### **1. Classification Region Processing Issue** 🚨
- **Problem**: ClassifierPlugin may appear to only process "first region"
- **Root Cause**: Timeline-based classification vs traditional 10-second fallback confusion
- **Impact**: Users expect multi-region classification but see inconsistent behavior
- **Priority**: HIGH - Core functionality perception issue

**Investigation Questions for Implementation:**
- Q1: How does ContentAnalysisPlugin pass timeline segments to ClassifierPlugin?
- Q2: What triggers the fallback to 10-second traditional classification?
- Q3: Are timeline segments being created properly for all audio content types?

### **2. ContentAnalysisPlugin → ClassifierPlugin Data Handoff** ⚠️
- **Problem**: Timeline segments not properly passed between plugins
- **Root Cause**: Feature cache vs plugin result data structure mismatch
- **Impact**: Classifier falls back to 10-second chunks instead of intelligent segments
- **Priority**: HIGH - Reduces classification accuracy

**Investigation Questions for Implementation:**
- Q1: What data structure does ContentAnalysisPlugin store in feature cache?
- Q2: What format does ClassifierPlugin expect for timeline segments?
- Q3: Are there version/format mismatches in the plugin communication protocol?

### **3. Long File Processing Performance** ⚠️
- **Problem**: Large files (>5 minutes) may have performance issues
- **Root Cause**: Memory management and processing pipeline inefficiencies
- **Impact**: User experience degradation for real-world content
- **Priority**: MEDIUM - Scalability concern

**Investigation Questions for Implementation:**
- Q1: At what file size/duration do we see performance degradation?
- Q2: Which plugins consume the most memory for large files?
- Q3: Can we implement streaming/chunked processing for large files?

### **4. Database Integration Robustness** ⚠️
- **Problem**: Database format vs engine format transformation complexity
- **Root Cause**: Multiple data format conversions and potential inconsistencies
- **Impact**: Timeline visualization and data integrity issues
- **Priority**: MEDIUM - Data reliability concern

**Investigation Questions for Implementation:**
- Q1: Are there data loss issues in engine→database→frontend transformations?
- Q2: Which format conversions are most error-prone?
- Q3: Can we standardize on a single data format throughout the pipeline?

## 🏗️ **SYSTEMATIC REFINEMENT PLAN**

### **Phase 1.5.1: Plugin Communication Architecture** (Week 1)
**Goal**: Perfect plugin-to-plugin data handoff and eliminate fallback scenarios

#### **Week 1 Sprint Questions:**
- **Day 1**: Can you reproduce the classification issue with specific test files?
- **Day 2**: What exactly is stored in the feature cache by ContentAnalysisPlugin?
- **Day 3**: How should we standardize the plugin communication protocol?
- **Day 4**: What integration tests should we write to prevent regressions?
- **Day 5**: How can we make plugin communication failures more visible to users?

### **Phase 1.5.2: Database & Storage Reliability** (Week 2)
**Goal**: Bulletproof data persistence and retrieval

#### **Week 2 Sprint Questions:**
- **Day 1**: Which database queries are slowest with large datasets?
- **Day 2**: Are there data integrity issues in the current schema?
- **Day 3**: How should we handle database schema migrations?
- **Day 4**: What backup and recovery strategy should we implement?
- **Day 5**: How can we monitor database performance in production?

### **Phase 1.5.3: Audio Processing Pipeline Hardening** (Week 3)
**Goal**: Rock-solid audio handling for any input

#### **Week 3 Sprint Questions:**
- **Day 1**: What audio formats/edge cases cause processing failures?
- **Day 2**: How can we improve silence detection for different content types?
- **Day 3**: What's the optimal memory management strategy for large files?
- **Day 4**: How should we handle corrupted or unusual audio files?
- **Day 5**: Can we parallelize audio processing safely?

### **Phase 1.5.4: Resource Management & Performance** (Week 4)
**Goal**: Optimal resource utilization and scalability

#### **Week 4 Sprint Questions:**
- **Day 1**: Where are the memory leaks and inefficiencies?
- **Day 2**: How can we better utilize GPU resources?
- **Day 3**: What's the optimal caching strategy for features?
- **Day 4**: How should we handle concurrent processing requests?
- **Day 5**: What performance monitoring should we implement?

## 🧪 **TESTING & VALIDATION FRAMEWORK**

### **Test File Collection Needed:**
- **Short files** (<30s): Various formats and content types
- **Medium files** (1-5 min): Typical user content
- **Long files** (>10 min): Stress testing scenarios
- **Edge cases**: Silence, noise, corrupted files, unusual formats

### **Performance Benchmarks:**
- **Processing speed**: Target <0.3x real-time for typical files
- **Memory usage**: Target <2GB peak for 500MB audio files
- **Database performance**: Target <100ms for typical queries
- **Error rate**: Target <0.1% processing failures

## 📋 **IMPLEMENTATION CHECKLIST**

### **Week 1: Plugin Communication (Ready to Start)**
- [ ] Reproduce classification issue with test files
- [ ] Profile ContentAnalysisPlugin → ClassifierPlugin data flow
- [ ] Document current plugin communication patterns
- [ ] Identify exact data handoff failures
- [ ] Implement standardized plugin communication protocol
- [ ] Write integration tests for plugin communication
- [ ] Validate fix with original problematic files

### **Development Notes for Implementation:**
1. **Start with specific test cases**: Use the files that showed the issue
2. **Add extensive logging**: Track data flow between plugins
3. **Write tests first**: Define expected behavior before fixing
4. **Incremental changes**: Small, testable improvements
5. **Performance monitoring**: Track impact of each change

## 🎯 **SUCCESS CRITERIA FOR PHASE 1.5**

### **Functional Requirements:**
- ✅ All regions properly classified (no "first region only" issues)
- ✅ 99.9% successful processing rate for valid audio files
- ✅ Zero database inconsistencies or data loss
- ✅ Graceful handling of all error scenarios

### **Performance Requirements:**
- ✅ <0.3x real-time processing for files up to 30 minutes
- ✅ <2GB peak memory usage for 500MB audio files  
- ✅ <100ms average database query response time
- ✅ <10s full system startup time

### **Quality Requirements:**
- ✅ >90% test coverage for core components
- ✅ 100% API endpoint documentation
- ✅ Zero memory leaks in 24-hour stress testing
- ✅ Comprehensive error logging and diagnostics

---

This document serves as the complete technical foundation and refinement roadmap for Audio Intelligence Sampler v2. Each implementation session should reference this document for context, current status, and specific questions to address.
