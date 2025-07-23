"""
Audio Intelligence Sampler v2 - Centralized Feature Extraction Plugin

Hybrid nnAudio + librosa feature extraction to eliminate redundancy across plugins.
This plugin runs FIRST and provides shared features to all downstream plugins.

Based on nnAudio compatibility testing results:
- nnAudio: GPU spectrograms, mel spectrograms (0.00% difference vs librosa)
- librosa: Chromagrams, onset detection (API compatibility issues with nnAudio)

Architecture principles:
- Centralized feature extraction eliminates redundancy
- GPU-accelerated spectrograms via nnAudio (when available)
- librosa fallback for full compatibility
- Memory-based feature caching during region processing
- SACRED: Never crashes, always returns features
"""

import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import torch

from ..base_plugin import BasePlugin, PluginRequirements

logger = logging.getLogger(__name__)

# Global feature cache shared across all FeatureExtractor instances
_GLOBAL_FEATURE_CACHE = {}

class FeatureExtractorPlugin(BasePlugin):
    """
    Centralized feature extraction with hybrid nnAudio + librosa approach.
    
    Features provided:
    - GPU-accelerated spectrograms (nnAudio) with librosa fallback
    - Mel spectrograms for classification models (nnAudio)
    - Chromagrams for key detection (librosa - API compatibility)
    - Onset detection and strength (librosa)
    - MFCC features for tempo/meter (librosa)  
    - Shared preprocessing (mono, resample, normalize)
    
    Memory caching:
    - Features cached in memory during region processing
    - Downstream plugins request features by name
    - Eliminates redundant computations across plugins
    """
    
    def __init__(self):
        self.plugin_name = "feature_extractor"
        self.version = "1.0.0-hybrid-nnaudio-librosa"
        self._nnaudio_available = False
        self._librosa_available = False
        self._device = None
        # Use global cache instead of instance cache
        self._check_dependencies()
        super().__init__()
        
    def get_name(self) -> str:
        """Return plugin name."""
        return self.plugin_name
    
    def get_version(self) -> str:
        """Return plugin version."""
        return self.version
    
    def get_requirements(self) -> Dict[str, Any]:
        """Return plugin resource requirements."""
        return {
            'gpu_memory_mb': 1000,  # nnAudio GPU operations
            'cpu_cores': 2,         # librosa operations
            'ram_mb': 512,          # Feature caching
            'processing_time_estimate': 2.0
        }
    
    def can_process(self, file_metadata: Dict[str, Any]) -> bool:
        """Check if plugin can process the given file."""
        try:
            # Check if we have at least one backend available
            if not self._librosa_available:
                logger.warning("librosa unavailable - core functionality missing")
                return False
            
            # FeatureExtractorPlugin can process any audio file
            return True
            
        except Exception as e:
            logger.error(f"Error checking feature extraction capability: {e}")
            return False
    
    def can_process_audio(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[bool, str]:
        """Check if plugin can process specific audio data (for testing)."""
        try:
            if len(audio_data) == 0:
                return False, "Empty audio data"
            
            if sample_rate <= 0:
                return False, "Invalid sample rate"
            
            # Check available backends
            backends = []
            if self._nnaudio_available:
                backends.append("nnAudio")
            if self._librosa_available:
                backends.append("librosa")
            
            if not backends:
                return False, "No feature extraction backends available"
            
            return True, f"Ready with backends: {', '.join(backends)}"
            
        except Exception as e:
            logger.error(f"Error checking feature extraction capability: {e}")
            return False, f"Capability check failed: {str(e)}"
    
    def _check_dependencies(self):
        """Check availability of nnAudio and librosa."""
        # Check nnAudio
        try:
            import nnAudio
            self._nnaudio_available = True
            logger.info("✅ nnAudio available for GPU-accelerated spectrograms")
        except ImportError:
            self._nnaudio_available = False
            logger.info("❌ nnAudio unavailable, using librosa fallback")
        
        # Check librosa (required)
        try:
            import librosa
            self._librosa_available = True
            logger.info("✅ librosa available for chromagrams and onsets")
        except ImportError:
            self._librosa_available = False
            logger.error("❌ librosa unavailable - feature extraction will fail")
    
    def _initialize(self):
        """Initialize feature extraction backends."""
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Feature extractor initialized on {self._device}")
        
        # Initialize nnAudio transforms if available
        if self._nnaudio_available:
            try:
                import nnAudio.features
                # Pre-initialize transforms for common parameters
                self._init_nnaudio_transforms()
                device_type = "GPU" if torch.cuda.is_available() else "CPU"
                logger.info(f"✅ nnAudio transforms initialized for {device_type}")
            except Exception as e:
                logger.warning(f"nnAudio initialization failed: {e}")
                self._nnaudio_available = False
    
    def _init_nnaudio_transforms(self):
        """Initialize common nnAudio transforms."""
        try:
            import nnAudio.features as features
            
            # Common parameters based on plugin analysis
            sr = 22050  # Standard sample rate for most plugins
            
            # STFT transform for general spectrograms
            self._stft_transform = features.STFT(
                n_fft=2048,
                hop_length=512,
                window='hann',
                freq_scale='linear',
                sr=sr
            ).to(self._device)
            
            # Mel spectrogram for classification (PaSST compatibility)
            self._mel_transform = features.MelSpectrogram(
                n_fft=2048,
                hop_length=160,  # PaSST-compatible hop length
                n_mels=128,
                sr=sr
            ).to(self._device)
            
            logger.info("✅ nnAudio transforms initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize nnAudio transforms: {e}")
            self._nnaudio_available = False
    
    def process(self, audio_data: np.ndarray, sample_rate: int, 
               file_path: Optional[Path] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Extract all audio features using hybrid nnAudio + librosa approach.
        
        This is the main BasePlugin interface method.
        """
        # Convert to region info format for internal analyze method
        region_info = kwargs.get('region_info', {
            'start_time': 0.0,
            'end_time': len(audio_data) / sample_rate,
            'region_id': kwargs.get('region_id', 'default')
        })
        
        return self.analyze(audio_data, sample_rate, region_info)
    
    def analyze(self, audio_data: np.ndarray, sample_rate: int, 
                region_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract all audio features and cache them for downstream plugins.
        
        Args:
            audio_data: Audio samples (mono or stereo)
            sample_rate: Sample rate in Hz
            region_info: Region metadata
            
        Returns:
            Dict containing:
            - feature_cache_id: ID for accessing cached features
            - available_features: List of available feature types
            - processing_info: Backend info and timing
        """
        start_time = time.time()
        
        try:
            # Clear previous cache
            _GLOBAL_FEATURE_CACHE.clear()
            
            # Preprocess audio (shared across all plugins)
            processed_audio = self._preprocess_audio(audio_data, sample_rate)
            
            # Extract features using hybrid approach
            features = self._extract_all_features(processed_audio, sample_rate)
            
            # Cache features with unique ID
            cache_id = f"region_{region_info.get('start_time', 0):.3f}_{int(time.time())}"
            _GLOBAL_FEATURE_CACHE[cache_id] = features
            
            processing_time = time.time() - start_time
            
            # Determine which backends were used
            backends_used = []
            if self._nnaudio_available and torch.cuda.is_available():
                backends_used.append("nnAudio (GPU)")
            if self._librosa_available:
                backends_used.append("librosa")
            
            result = {
                'feature_cache_id': cache_id,
                'available_features': list(features.keys()),
                'backends_used': backends_used,
                'processing_time_ms': processing_time * 1000,
                'audio_duration_s': len(processed_audio) / sample_rate,
                'sample_rate': sample_rate,
                'success': True
            }
            
            logger.info(f"✅ Features extracted in {processing_time:.3f}s using {', '.join(backends_used)}")
            return result
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {
                'feature_cache_id': None,
                'available_features': [],
                'backends_used': [],
                'processing_time_ms': 0,
                'error': str(e),
                'success': False
            }
    
    def _preprocess_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Shared audio preprocessing for all plugins.
        
        Eliminates redundant preprocessing across plugins:
        - Mono conversion (was duplicated in all 3 plugins)
        - Normalization
        - Standard sample rate conversion
        """
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_mono = np.mean(audio_data, axis=1)
        else:
            audio_mono = audio_data.copy()
        
        # Normalize to [-1, 1] range
        if np.max(np.abs(audio_mono)) > 0:
            audio_mono = audio_mono / np.max(np.abs(audio_mono))
        
        # Resample to standard rate (22050 Hz) for consistency
        if sample_rate != 22050 and self._librosa_available:
            import librosa
            audio_mono = librosa.resample(audio_mono, orig_sr=sample_rate, target_sr=22050)
            sample_rate = 22050
        
        return audio_mono
    
    def _extract_all_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """
        Extract all features using hybrid nnAudio + librosa approach.
        
        Based on compatibility testing:
        - nnAudio: spectrograms, mel spectrograms (GPU-accelerated, 0.00% difference)
        - librosa: chromagrams, onset detection (API compatibility issues with nnAudio)
        """
        features = {}
        
        # 1. nnAudio spectrograms (GPU-accelerated when available, CPU fallback)
        if self._nnaudio_available:
            features.update(self._extract_nnaudio_features(audio_data, sample_rate))
        
        # 2. librosa features (chromagrams, onsets, fallback spectrograms)
        if self._librosa_available:
            features.update(self._extract_librosa_features(audio_data, sample_rate))
        
        return features
    
    def _extract_nnaudio_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract GPU-accelerated features using nnAudio."""
        features = {}
        
        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0).to(self._device)
            
            # STFT spectrogram (replaces scipy.signal.stft in ClassifierPlugin)
            if hasattr(self, '_stft_transform'):
                stft_result = self._stft_transform(audio_tensor)
                # Convert to magnitude spectrogram
                spectrogram = torch.abs(stft_result).squeeze(0).cpu().numpy()
                features['spectrogram_nnaudio'] = spectrogram
                logger.debug(f"nnAudio STFT: {spectrogram.shape}")
            
            # Mel spectrogram for PaSST (ClassifierPlugin)
            if hasattr(self, '_mel_transform'):
                mel_spec = self._mel_transform(audio_tensor)
                mel_spec_db = torch.log(mel_spec + 1e-8).squeeze(0).cpu().numpy()
                features['mel_spectrogram_nnaudio'] = mel_spec_db
                features['mel_spectrogram_torch'] = mel_spec.squeeze(0)  # Keep tensor for PaSST
                logger.debug(f"nnAudio mel spectrogram: {mel_spec_db.shape}")
            
        except Exception as e:
            logger.error(f"nnAudio feature extraction failed: {e}")
        
        return features
    
    def _extract_librosa_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Extract features using librosa (chromagrams, onsets, fallback spectrograms)."""
        features = {}
        
        try:
            import librosa
            
            # Chromagram for key detection (KeyFinderPlugin)
            # Note: nnAudio chromagram API incompatible (missing fmin parameter)
            chroma = librosa.feature.chroma_stft(
                y=audio_data, 
                sr=sample_rate, 
                hop_length=512, 
                n_fft=2048
            )
            features['chromagram_librosa'] = chroma
            logger.debug(f"librosa chromagram: {chroma.shape}")
            
            # Onset detection for tempo analysis (TempoMeterDetectorPlugin)
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, 
                sr=sample_rate, 
                hop_length=512
            )
            onset_times = librosa.frames_to_time(onset_frames, sr=sample_rate, hop_length=512)
            onset_strength = librosa.onset.onset_strength(
                y=audio_data, 
                sr=sample_rate, 
                hop_length=512
            )
            features['onsets_librosa'] = {
                'frames': onset_frames,
                'times': onset_times,
                'strength': onset_strength
            }
            logger.debug(f"librosa onsets: {len(onset_times)} detected")
            
            # MFCC for meter detection (TempoMeterDetectorPlugin)
            mfcc = librosa.feature.mfcc(
                y=audio_data, 
                sr=sample_rate, 
                n_mfcc=13, 
                n_mels=128,
                hop_length=512
            )
            features['mfcc_librosa'] = mfcc
            logger.debug(f"librosa MFCC: {mfcc.shape}")
            
            # Fallback STFT spectrogram (if nnAudio unavailable)
            if 'spectrogram_nnaudio' not in features:
                stft = librosa.stft(audio_data, hop_length=512, n_fft=2048)
                spectrogram = np.abs(stft)
                features['spectrogram_librosa'] = spectrogram
                logger.debug(f"librosa STFT fallback: {spectrogram.shape}")
            
            # Harmonic-Percussive Separation (TempoMeterDetectorPlugin Method 5)
            harmonic, percussive = librosa.effects.hpss(audio_data)
            features['hpss_librosa'] = {
                'harmonic': harmonic,
                'percussive': percussive
            }
            
        except Exception as e:
            logger.error(f"librosa feature extraction failed: {e}")
        
        return features
    
    def get_cached_features(self, cache_id: str, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        Retrieve cached features for downstream plugins.
        
        Args:
            cache_id: Cache ID from analyze() result
            feature_names: Specific features to retrieve (None = all)
            
        Returns:
            Dict of requested features or empty dict if cache miss
        """
        if cache_id not in _GLOBAL_FEATURE_CACHE:
            logger.warning(f"Cache miss for ID: {cache_id}")
            return {}
        
        cached_features = _GLOBAL_FEATURE_CACHE[cache_id]
        
        if feature_names is None:
            return cached_features
        
        # Return only requested features
        requested_features = {}
        for name in feature_names:
            if name in cached_features:
                requested_features[name] = cached_features[name]
            else:
                logger.warning(f"Requested feature '{name}' not available in cache")
        
        return requested_features
    
    def clear_cache(self):
        """Clear feature cache to free memory."""
        _GLOBAL_FEATURE_CACHE.clear()
        logger.info("Feature cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about current cache state."""
        cache_ids = list(_GLOBAL_FEATURE_CACHE.keys())
        total_features = sum(len(features) for features in _GLOBAL_FEATURE_CACHE.values())
        
        return {
            'cache_ids': cache_ids,
            'total_cached_regions': len(cache_ids),
            'total_features': total_features,
            'available_backends': {
                'nnAudio': self._nnaudio_available,
                'librosa': self._librosa_available,
                'gpu_available': torch.cuda.is_available()
            }
        }