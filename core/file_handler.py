"""
Audio Intelligence Sampler v2 - File Handler

Fast audio loading following sacred architecture:
- Primary: pedalboard (fastest, from sacred architecture)  
- Fallback: librosa (reliable, already required)

Architecture principles:
- Lazy imports (only load what's needed)
- Never crash on corrupted files
- Memory-efficient loading
- Simple and fast
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Union
import numpy as np


class FileHandler:
    """
    Fast audio file loading with pedalboard primary, librosa fallback.
    
    Simple and efficient - only loads libraries when actually needed.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize file handler with configuration."""
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Audio loading settings
        self.target_sr = self.config.get('target_sample_rate', 22050)
        self.mono = self.config.get('force_mono', True)
        self.normalize = self.config.get('normalize_audio', True)
        
        # Supported formats
        self.supported_formats = {
            '.wav', '.mp3', '.flac', '.aac', '.m4a', '.ogg', 
            '.wma', '.aiff', '.au'
        }
        
        self.logger.info("FileHandler initialized - pedalboard primary, librosa fallback")
    
    def load_audio(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load audio file: pedalboard first, librosa fallback.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict with success status, audio data, metadata, and errors
        """
        file_path = Path(file_path)
        
        result = {
            'success': False,
            'audio_data': None,
            'metadata': {},
            'errors': [],
            'loader_used': None
        }
        
        self.logger.info(f"Loading audio file: {file_path}")
        
        try:
            # Validate file
            if not self._validate_file(file_path, result):
                return result
            
            # Try pedalboard first (fastest)
            if self._try_pedalboard(file_path, result):
                return result
            
            # Fallback to librosa
            if self._try_librosa(file_path, result):
                return result
            
            # If we get here, both failed
            result['errors'].append("Both pedalboard and librosa failed to load file")
            
        except Exception as e:
            result['errors'].append(f"File loading error: {e}")
            
        return result
    
    def _validate_file(self, file_path: Path, result: Dict) -> bool:
        """Validate file and add basic metadata."""
        try:
            if not file_path.exists():
                result['errors'].append(f"File not found: {file_path}")
                return False
            
            if not file_path.is_file():
                result['errors'].append(f"Path is not a file: {file_path}")
                return False
            
            stat = file_path.stat()
            if stat.st_size == 0:
                result['errors'].append("File is empty")
                return False
            
            extension = file_path.suffix.lower()
            if extension not in self.supported_formats:
                result['errors'].append(f"Unsupported format: {extension}")
                return False
            
            # Add basic metadata
            result['metadata'] = {
                'filename': file_path.name,
                'extension': extension,
                'size_bytes': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024)
            }
            
            return True
            
        except Exception as e:
            result['errors'].append(f"File validation error: {e}")
            return False
    
    def _try_pedalboard(self, file_path: Path, result: Dict) -> bool:
        """Try loading with pedalboard (primary loader)."""
        try:
            # Lazy import - only load if we actually use it
            from pedalboard.io import AudioFile
            
            with AudioFile(str(file_path)) as f:
                # Load audio data
                audio_data = f.read(f.frames)
                
                # Convert to mono if needed and ensure 1D
                if len(audio_data.shape) > 1:
                    if self.mono:
                        audio_data = np.mean(audio_data, axis=0)
                    else:
                        audio_data = audio_data[0]  # Take first channel
                audio_data = audio_data.squeeze()
                
                # Process audio
                audio_data = self._process_audio(audio_data, f.samplerate)
                
                # Update result
                result.update({
                    'success': True,
                    'audio_data': audio_data,
                    'loader_used': 'pedalboard',
                    'metadata': {
                        **result['metadata'],
                        'original_sample_rate': f.samplerate,
                        'original_duration': f.duration,
                        'original_channels': f.num_channels,
                        'final_sample_rate': self.target_sr,
                        'final_duration': len(audio_data) / self.target_sr,
                        'samples': len(audio_data)
                    }
                })
                
                self.logger.info("Successfully loaded with pedalboard")
                return True
                
        except ImportError:
            self.logger.info("pedalboard not available, trying librosa")
            return False
        except Exception as e:
            self.logger.warning(f"pedalboard failed: {e}, trying librosa")
            return False
    
    def _try_librosa(self, file_path: Path, result: Dict) -> bool:
        """Try loading with librosa (fallback loader)."""
        try:
            # Lazy import - only load if we actually use it
            import librosa
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Load audio data
                audio_data, sample_rate = librosa.load(
                    str(file_path),
                    sr=None,  # Keep original sample rate
                    mono=self.mono
                )
                
                # Process audio
                audio_data = self._process_audio(audio_data, sample_rate)
                
                # Update result
                result.update({
                    'success': True,
                    'audio_data': audio_data,
                    'loader_used': 'librosa',
                    'metadata': {
                        **result['metadata'],
                        'original_sample_rate': sample_rate,
                        'original_duration': len(audio_data) / sample_rate,
                        'original_channels': 1 if self.mono else 'unknown',
                        'final_sample_rate': self.target_sr,
                        'final_duration': len(audio_data) / self.target_sr,
                        'samples': len(audio_data)
                    }
                })
                
                self.logger.info("Successfully loaded with librosa")
                return True
                
        except ImportError:
            result['errors'].append("librosa not available")
            return False
        except Exception as e:
            result['errors'].append(f"librosa failed: {e}")
            return False
    
    def _process_audio(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process loaded audio data (resample, normalize)."""
        # Resample if needed
        if sample_rate != self.target_sr:
            audio_data = self._resample_audio(audio_data, sample_rate)
        
        # Normalize if requested
        if self.normalize and len(audio_data) > 0:
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val
        
        return audio_data
    
    def _resample_audio(self, audio_data: np.ndarray, from_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=from_sr, target_sr=self.target_sr)
        except ImportError:
            # Simple fallback if librosa not available
            if from_sr > self.target_sr:
                step = from_sr // self.target_sr
                return audio_data[::step]
            return audio_data