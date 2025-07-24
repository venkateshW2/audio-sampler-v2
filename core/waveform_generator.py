"""
Waveform Data Generator for Audio Intelligence Sampler v2

Generates waveform visualization data compatible with Peaks.js format.
Produces downsampled audio data points for efficient client-side visualization.

Architecture principles:
- Never crash (graceful error handling)
- Efficient downsampling for web display
- Compatible with Peaks.js format
- Time-aligned with analysis results
"""

import logging
import numpy as np
import librosa
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class WaveformGenerator:
    """
    Generate waveform visualization data from audio files.
    
    Produces data compatible with Peaks.js and other waveform visualization libraries.
    Handles downsampling and peak detection for efficient client-side rendering.
    """
    
    def __init__(self, target_sample_rate: int = 22050):
        """
        Initialize waveform generator.
        
        Args:
            target_sample_rate: Sample rate for waveform data (default: 22050)
        """
        self.target_sample_rate = target_sample_rate
        self.logger = logging.getLogger(f"{__name__}.WaveformGenerator")
        
    def generate_waveform_data(self, audio_data: np.ndarray, sample_rate: int, 
                              overview_samples: int = 1000, 
                              zoom_samples: int = 8000) -> Dict[str, Any]:
        """
        Generate waveform visualization data from audio data.
        
        Args:
            audio_data: Raw audio data array
            sample_rate: Original sample rate of audio
            overview_samples: Number of samples for overview waveform
            zoom_samples: Number of samples for detailed zoom waveform
            
        Returns:
            Dict containing waveform data for visualization
        """
        try:
            self.logger.info(f"Generating waveform data: {len(audio_data)} samples @ {sample_rate}Hz")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=0)
                
            duration = len(audio_data) / sample_rate
            
            # Generate overview waveform (low resolution for full file view)
            overview_data = self._downsample_peaks(audio_data, overview_samples)
            
            # Generate zoom waveform (higher resolution for detailed view)
            zoom_data = self._downsample_peaks(audio_data, zoom_samples)
            
            # Calculate RMS values for better visualization
            overview_rms = self._calculate_rms_segments(audio_data, overview_samples)
            zoom_rms = self._calculate_rms_segments(audio_data, zoom_samples)
            
            # Create time axis
            overview_times = np.linspace(0, duration, overview_samples)
            zoom_times = np.linspace(0, duration, zoom_samples)
            
            waveform_data = {
                'duration': float(duration),
                'sample_rate': int(sample_rate),
                'overview': {
                    'data': overview_data.tolist(),
                    'rms': overview_rms.tolist(),
                    'times': overview_times.tolist(),
                    'samples': len(overview_data)
                },
                'zoom': {
                    'data': zoom_data.tolist(),
                    'rms': zoom_rms.tolist(), 
                    'times': zoom_times.tolist(),
                    'samples': len(zoom_data)
                },
                'peaks_js_format': {
                    # Format compatible with Peaks.js
                    'version': 2,
                    'channels': 1,
                    'sample_rate': sample_rate,
                    'samples_per_pixel': len(audio_data) // zoom_samples,
                    'bits': 16,
                    'length': len(audio_data),
                    'data': zoom_data.tolist()
                }
            }
            
            self.logger.info(f"✅ Waveform data generated: {overview_samples} overview + {zoom_samples} zoom samples")
            return waveform_data
            
        except Exception as e:
            self.logger.error(f"❌ Waveform generation failed: {e}")
            return {
                'error': str(e),
                'duration': 0,
                'sample_rate': sample_rate,
                'overview': {'data': [], 'rms': [], 'times': [], 'samples': 0},
                'zoom': {'data': [], 'rms': [], 'times': [], 'samples': 0}
            }
    
    def _downsample_peaks(self, audio_data: np.ndarray, target_samples: int) -> np.ndarray:
        """
        Downsample audio data while preserving peaks for visualization.
        
        Args:
            audio_data: Input audio data
            target_samples: Desired number of output samples
            
        Returns:
            Downsampled audio data preserving peaks
        """
        try:
            if len(audio_data) <= target_samples:
                return audio_data
                
            # Calculate window size for downsampling
            window_size = len(audio_data) // target_samples
            
            # Extract peaks (max absolute value) from each window
            peaks = []
            for i in range(target_samples):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(audio_data))
                
                if start_idx < len(audio_data):
                    window = audio_data[start_idx:end_idx]
                    # Take max absolute value to preserve peaks
                    peak = window[np.argmax(np.abs(window))]
                    peaks.append(peak)
                else:
                    peaks.append(0.0)
            
            return np.array(peaks)
            
        except Exception as e:
            self.logger.error(f"Peak downsampling failed: {e}")
            return np.zeros(target_samples)
    
    def _calculate_rms_segments(self, audio_data: np.ndarray, num_segments: int) -> np.ndarray:
        """
        Calculate RMS (Root Mean Square) values for audio segments.
        
        Args:
            audio_data: Input audio data
            num_segments: Number of RMS segments to calculate
            
        Returns:
            Array of RMS values
        """
        try:
            if len(audio_data) <= num_segments:
                return np.sqrt(np.mean(audio_data**2)) * np.ones(num_segments)
                
            window_size = len(audio_data) // num_segments
            rms_values = []
            
            for i in range(num_segments):
                start_idx = i * window_size
                end_idx = min((i + 1) * window_size, len(audio_data))
                
                if start_idx < len(audio_data):
                    window = audio_data[start_idx:end_idx]
                    rms = np.sqrt(np.mean(window**2))
                    rms_values.append(rms)
                else:
                    rms_values.append(0.0)
            
            return np.array(rms_values)
            
        except Exception as e:
            self.logger.error(f"RMS calculation failed: {e}")
            return np.zeros(num_segments)

    def generate_from_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Generate waveform data directly from audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict containing waveform visualization data
        """
        try:
            file_path = Path(file_path)
            self.logger.info(f"Loading audio file for waveform: {file_path}")
            
            # Load audio file
            audio_data, sample_rate = librosa.load(str(file_path), sr=None, mono=True)
            
            # Generate waveform data
            return self.generate_waveform_data(audio_data, sample_rate)
            
        except Exception as e:
            self.logger.error(f"Failed to generate waveform from file {file_path}: {e}")
            return {
                'error': str(e),
                'duration': 0,
                'sample_rate': 0,
                'overview': {'data': [], 'rms': [], 'times': [], 'samples': 0},
                'zoom': {'data': [], 'rms': [], 'times': [], 'samples': 0}
            }

# Singleton instance for use across the application
waveform_generator = WaveformGenerator()