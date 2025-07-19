# Audio Intelligence Sampler v2

A professional audio analysis tool for musicians to intelligently sample and catalog found sounds, field recordings, and musical content.

## Phase 1: Core Backend Development

Building a stable PyTorch-only audio processing pipeline with plugin architecture for analyzing audio files and extracting musical information.

### Features (Planned)
- Audio classification (music, speech, ambience)
- Tempo and beat detection
- Key and scale analysis
- Chord progression detection
- Never-crash plugin architecture
- GPU-accelerated processing (GTX 1060+ support)
- Self-contained distribution

### Technology Stack
- **ML Framework**: PyTorch 2.5.1 with CUDA 12.1
- **Audio Processing**: librosa, madmom, pedalboard, basic-pitch
- **Backend**: FastAPI + SQLite
- **Environment**: Self-contained conda environment

### Installation

1. Clone the repository:
```bash
git clone https://github.com/venkateshw2/audio-sampler-v2.git
cd audio-sampler-v2
```

2. Download and install miniconda locally:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ./miniconda
```

3. Create and activate environment:
```bash
./miniconda/bin/conda create -n audio-sampler-v2 python=3.9 -y
source ./miniconda/bin/activate audio-sampler-v2
```

4. Install dependencies:
```bash
# Install PyTorch with CUDA support first
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Install Cython (required for madmom)
pip install Cython

# Install all other requirements
pip install -r requirements.txt
```

### Development Status

**Phase 1 Foundation Complete:**
- ✅ Self-contained conda environment
- ✅ All dependencies installed and tested
- ✅ GPU support verified (GTX 1060)
- ✅ Project structure following sacred architecture
- ✅ Git repository with clean file structure

**Currently Implementing:**
- 🔄 Core engine with audio processing pipeline
- 🔄 Plugin system architecture
- 🔄 Resource management for GPU operations

### Architecture

Following a strict plugin-based architecture where:
- **Core engine** never crashes and orchestrates all processing
- **Plugins** handle specific audio analysis tasks and can fail safely
- **Resource manager** prevents GPU memory conflicts
- **Database** stores all analysis results with full-text search

### Contributing

This is currently a single-developer project for a collective of ~10 musicians. The focus is on stability and reliability over features.

### License

TBD

---

🤖 Generated with [Claude Code](https://claude.ai/code)