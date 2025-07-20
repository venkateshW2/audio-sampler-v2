"""
Audio Intelligence Sampler v2 - File Handler

Fast audio loading + silence detection & region extraction:
- Primary: pedalboard (fastest, from sacred architecture)  
- Fallback: librosa (reliable, already required)
- Silence Detection: 10+ seconds silence = region boundary
- Region Extraction: filename-r1, filename-r2 naming

Architecture principles:
- Lazy imports (only load what's needed)
- Never crash on corrupted files
- Memory-efficient loading
- Region-based processing for long files
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, Any, Union, List
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
        
        # Region detection settings
        self.silence_threshold_seconds = self.config.get('silence_threshold_seconds', 10.0)
        self.silence_db_threshold = self.config.get('silence_db_threshold', -40.0)
        self.enable_region_detection = self.config.get('enable_region_detection', True)
        
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
    
    def load_audio_with_regions(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load audio file and extract regions based on silence detection.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict with success status, regions list, and metadata
        """
        file_path = Path(file_path)
        
        result = {
            'success': False,
            'file_metadata': {},
            'regions': [],
            'total_regions': 0,
            'errors': [],
            'silence_threshold_used': self.silence_threshold_seconds
        }
        
        try:
            # First load the audio normally
            audio_result = self.load_audio(file_path)
            
            if not audio_result['success']:
                result['errors'] = audio_result['errors']
                return result
            
            audio_data = audio_result['audio_data']
            metadata = audio_result['metadata']
            sample_rate = metadata['final_sample_rate']
            
            # Store file-level metadata
            result['file_metadata'] = metadata
            
            # Skip region detection if disabled or audio too short
            if not self.enable_region_detection:
                # Return single region covering entire file
                result.update({
                    'success': True,
                    'regions': [self._create_full_file_region(file_path, audio_data, sample_rate)],
                    'total_regions': 1
                })
                return result
            
            # Detect silence and extract regions
            regions = self._detect_regions(file_path, audio_data, sample_rate)
            
            result.update({
                'success': True,
                'regions': regions,
                'total_regions': len(regions)
            })
            
            self.logger.info(f"✅ Extracted {len(regions)} regions from {file_path.name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in region extraction: {e}")
            result['errors'].append(f"Region extraction failed: {str(e)}")
            return result
    
    def _detect_regions(self, file_path: Path, audio_data: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Detect content regions separated by silence."""
        try:
            # Convert to dB to detect silence
            audio_db = self._audio_to_db(audio_data)
            
            # Find silence/content segments
            is_silent = audio_db < self.silence_db_threshold
            
            # Find transitions between silence and content
            transitions = np.diff(is_silent.astype(int))
            
            # Get start and end points of content regions
            content_starts = np.where(transitions == -1)[0] + 1  # Silent to content
            content_ends = np.where(transitions == 1)[0] + 1     # Content to silent
            
            # Handle edge cases
            if not is_silent[0]:  # Starts with content
                content_starts = np.concatenate([[0], content_starts])
            if not is_silent[-1]:  # Ends with content
                content_ends = np.concatenate([content_ends, [len(audio_data)]])
            
            # Convert sample indices to time
            start_times = content_starts / sample_rate
            end_times = content_ends / sample_rate
            
            # Filter regions by minimum silence duration
            regions = []
            region_number = 1
            
            for start_sample, end_sample, start_time, end_time in zip(content_starts, content_ends, start_times, end_times):
                duration = end_time - start_time
                
                # Check if previous gap was long enough (silence threshold)
                if len(regions) > 0:
                    prev_end_time = regions[-1]['end_time']
                    silence_duration = start_time - prev_end_time
                    
                    # If silence gap is less than threshold, merge with previous region
                    if silence_duration < self.silence_threshold_seconds:
                        # Extend previous region
                        regions[-1]['end_time'] = end_time
                        regions[-1]['end_sample'] = end_sample
                        regions[-1]['duration'] = regions[-1]['end_time'] - regions[-1]['start_time']
                        regions[-1]['audio_data'] = audio_data[regions[-1]['start_sample']:end_sample]
                        continue
                
                # Create new region
                region = self._create_region(
                    file_path=file_path,
                    region_number=region_number,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    start_time=start_time,
                    end_time=end_time,
                    audio_data=audio_data[start_sample:end_sample],
                    sample_rate=sample_rate
                )
                
                regions.append(region)
                region_number += 1
            
            # If no regions found, return entire file as single region
            if len(regions) == 0:
                regions = [self._create_full_file_region(file_path, audio_data, sample_rate)]
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Error detecting regions: {e}")
            # Fallback: return entire file as single region
            return [self._create_full_file_region(file_path, audio_data, sample_rate)]
    
    def _audio_to_db(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio to decibel scale for silence detection."""
        try:
            # RMS energy calculation with small window
            window_size = 1024
            hop_length = 512
            
            # Pad audio to handle edge cases
            padded_audio = np.pad(audio_data, (window_size//2, window_size//2), mode='constant')
            
            # Calculate RMS energy
            rms_values = []
            for i in range(0, len(audio_data), hop_length):
                start_idx = i
                end_idx = min(i + window_size, len(padded_audio))
                window = padded_audio[start_idx:end_idx]
                
                rms = np.sqrt(np.mean(window**2))
                rms_values.append(rms)
            
            rms_array = np.array(rms_values)
            
            # Convert to dB (avoid log(0))
            rms_array = np.maximum(rms_array, 1e-8)
            db_values = 20 * np.log10(rms_array)
            
            # Interpolate back to original length
            if len(db_values) != len(audio_data):
                from scipy.interpolate import interp1d
                x_old = np.linspace(0, 1, len(db_values))
                x_new = np.linspace(0, 1, len(audio_data))
                interpolator = interp1d(x_old, db_values, kind='linear', fill_value='extrapolate')
                db_values = interpolator(x_new)
            
            return db_values
            
        except Exception as e:
            self.logger.error(f"Error converting audio to dB: {e}")
            # Fallback: use simple absolute value
            return 20 * np.log10(np.maximum(np.abs(audio_data), 1e-8))
    
    def _create_region(self, file_path: Path, region_number: int, 
                      start_sample: int, end_sample: int,
                      start_time: float, end_time: float,
                      audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Create region metadata dictionary."""
        
        # Generate region ID: filename-r1, filename-r2, etc.
        base_name = file_path.stem  # filename without extension
        region_id = f"{base_name}-r{region_number}"
        
        return {
            'region_id': region_id,
            'region_number': region_number,
            'parent_file': str(file_path),
            'start_time': float(start_time),
            'end_time': float(end_time),
            'duration': float(end_time - start_time),
            'start_sample': int(start_sample),
            'end_sample': int(end_sample),
            'sample_count': int(end_sample - start_sample),
            'audio_data': audio_data,
            'sample_rate': sample_rate,
            'region_type': 'content',
            'metadata': {
                'channels': 1 if len(audio_data.shape) == 1 else audio_data.shape[0],
                'format': file_path.suffix.lower(),
                'extracted_at': np.datetime64('now'),
                'silence_threshold_used': self.silence_threshold_seconds,
                'db_threshold_used': self.silence_db_threshold
            }
        }
    
    def _create_full_file_region(self, file_path: Path, audio_data: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Create single region covering entire file (when no silence detected)."""
        base_name = file_path.stem
        region_id = f"{base_name}-r1"
        
        duration = len(audio_data) / sample_rate
        
        return {
            'region_id': region_id,
            'region_number': 1,
            'parent_file': str(file_path),
            'start_time': 0.0,
            'end_time': duration,
            'duration': duration,
            'start_sample': 0,
            'end_sample': len(audio_data),
            'sample_count': len(audio_data),
            'audio_data': audio_data,
            'sample_rate': sample_rate,
            'region_type': 'full_file',
            'metadata': {
                'channels': 1 if len(audio_data.shape) == 1 else audio_data.shape[0],
                'format': file_path.suffix.lower(),
                'extracted_at': np.datetime64('now'),
                'silence_threshold_used': self.silence_threshold_seconds,
                'note': 'No silence regions detected - entire file as single region'
            }
        }