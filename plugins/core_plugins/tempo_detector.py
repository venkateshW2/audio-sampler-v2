"""
Audio Intelligence Sampler v2 - Tempo Detection Plugin

Uses BeatNet for state-of-the-art joint beat, downbeat, tempo, and meter tracking.
BeatNet is superior to madmom with CRNN + particle filtering for better accuracy.

Architecture principles:
- Resource-aware processing
- Comprehensive error handling  
- SACRED: Never crashes, always returns results
- GPU-optimized for GTX 1060
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch

from ..base_plugin import BasePlugin, PluginRequirements

logger = logging.getLogger(__name__)


class TempoDetectorPlugin(BasePlugin):
    """
    SOTA tempo, beat, and downbeat detection using BeatNet.
    
    BeatNet provides joint tracking of:
    - Beats and downbeats
    - Tempo (BPM) 
    - Meter/time signature
    - Confidence scores
    """
    
    def __init__(self):
        super().__init__()
        self.plugin_name = "tempo_detector"
        self.version = "2.0.0-beatnet"
        self._beatnet_model = None
        self._model_loaded = False
        
    def get_requirements(self) -> PluginRequirements:
        """Return resource requirements for tempo detection."""
        return PluginRequirements(
            gpu_memory_mb=512,  # BeatNet is GPU-efficient
            cpu_cores=1,
            ram_mb=256,
            processing_time_estimate=2.0
        )
    
    def can_process(self, audio_data: np.ndarray, sample_rate: int, 
                   resource_status: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if tempo detection can be performed."""
        try:
            # Check audio requirements
            if len(audio_data) == 0:
                return False, "Empty audio data"
            
            if len(audio_data) / sample_rate < 3.0:
                return False, "Audio too short for reliable tempo detection (min 3 seconds)"
            
            # Check resource availability
            gpu_available = resource_status.get('gpu', {}).get('available', False)
            memory_percent = resource_status.get('memory', {}).get('memory_percent', 100)
            
            if memory_percent > 85:
                return False, "Insufficient memory for tempo detection"
            
            # BeatNet works well on both GPU and CPU
            return True, "Ready for tempo detection"
            
        except Exception as e:
            logger.error(f"Error checking tempo detection capability: {e}")
            return False, f"Capability check failed: {str(e)}"
    
    def _load_beatnet_model(self) -> bool:
        """Load BeatNet model with error handling."""
        if self._model_loaded:
            return True
            
        try:
            logger.info("Loading BeatNet model...")
            
            # Import BeatNet
            from BeatNet.BeatNet import BeatNet
            
            # Initialize BeatNet with optimal settings for our use case
            # mode 1: offline mode (most accurate)
            # mode 2: online mode (causal, good for real-time)
            # mode 3: real-time mode
            self._beatnet_model = BeatNet(
                model=1,  # Offline mode for best accuracy
                mode='offline',
                inference_model='PF',  # Particle Filter for best results
                plot=[],  # No plotting to save resources
                thread=False  # Single-threaded for resource control
            )
            
            self._model_loaded = True
            logger.info("✅ BeatNet model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load BeatNet model: {e}")
            self._model_loaded = False
            return False
    
    def process(self, audio_data: np.ndarray, sample_rate: int, 
               file_path: Optional[Path] = None, 
               **kwargs) -> Dict[str, Any]:
        """
        Extract tempo, beats, and downbeats using BeatNet.
        
        Returns comprehensive tempo analysis with high accuracy.
        """
        try:
            # Load model if needed
            if not self._load_beatnet_model():
                return self._error_result("Failed to load BeatNet model")
            
            # Ensure audio is mono and correct sample rate
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # BeatNet expects specific sample rate (usually 22050)
            target_sr = 22050
            if sample_rate != target_sr:
                import librosa
                audio_data = librosa.resample(audio_data, 
                                            orig_sr=sample_rate, 
                                            target_sr=target_sr)
                sample_rate = target_sr
            
            logger.info(f"Processing {len(audio_data)/sample_rate:.1f}s audio for tempo detection")
            
            # Process with BeatNet
            output = self._beatnet_model.process(audio_data)
            
            # BeatNet returns: [beats, downbeats, tempo_curve, beats_confidence]
            beats = output[0] if len(output) > 0 else []
            downbeats = output[1] if len(output) > 1 else []
            tempo_curve = output[2] if len(output) > 2 else []
            
            # Extract tempo information
            tempo_analysis = self._analyze_tempo_results(
                beats, downbeats, tempo_curve, sample_rate
            )
            
            # Calculate beat intervals and consistency
            beat_analysis = self._analyze_beat_patterns(beats, downbeats)
            
            result = {
                'success': True,
                'tempo_bpm': tempo_analysis['primary_tempo'],
                'tempo_confidence': tempo_analysis['tempo_confidence'],
                'beats': {
                    'times': beats.tolist() if hasattr(beats, 'tolist') else beats,
                    'count': len(beats),
                    'intervals': beat_analysis['beat_intervals'],
                    'consistency': beat_analysis['consistency']
                },
                'downbeats': {
                    'times': downbeats.tolist() if hasattr(downbeats, 'tolist') else downbeats,
                    'count': len(downbeats),
                    'intervals': beat_analysis['downbeat_intervals']
                },
                'meter': {
                    'time_signature': beat_analysis['estimated_meter'],
                    'confidence': beat_analysis['meter_confidence']
                },
                'tempo_curve': {
                    'values': tempo_curve.tolist() if hasattr(tempo_curve, 'tolist') else tempo_curve,
                    'stability': tempo_analysis['tempo_stability']
                },
                'analysis_metadata': {
                    'method': 'BeatNet',
                    'model_version': '1.1.3',
                    'audio_duration': len(audio_data) / sample_rate,
                    'sample_rate': sample_rate,
                    'processing_mode': 'offline'
                }
            }
            
            logger.info(f"✅ Tempo detection: {tempo_analysis['primary_tempo']:.1f} BPM "
                       f"({tempo_analysis['tempo_confidence']:.2f} confidence)")
            
            return result
            
        except Exception as e:
            logger.error(f"Tempo detection error: {e}")
            return self._error_result(f"Tempo detection failed: {str(e)}")
    
    def _analyze_tempo_results(self, beats: List, downbeats: List, 
                              tempo_curve: List, sample_rate: int) -> Dict[str, Any]:
        """Analyze tempo detection results for comprehensive metrics."""
        try:
            if len(beats) < 2:
                return {
                    'primary_tempo': 0.0,
                    'tempo_confidence': 0.0,
                    'tempo_stability': 0.0
                }
            
            # Calculate beat intervals in seconds
            beat_times = np.array(beats)
            intervals = np.diff(beat_times)
            
            # Convert to BPM
            bpm_values = 60.0 / intervals
            
            # Remove outliers (likely detection errors)
            q75, q25 = np.percentile(bpm_values, [75, 25])
            iqr = q75 - q25
            lower_bound = q25 - 1.5 * iqr
            upper_bound = q75 + 1.5 * iqr
            clean_bpm = bpm_values[(bpm_values >= lower_bound) & (bpm_values <= upper_bound)]
            
            if len(clean_bpm) == 0:
                clean_bpm = bpm_values
            
            # Primary tempo (median for robustness)
            primary_tempo = float(np.median(clean_bpm))
            
            # Tempo confidence (inverse of coefficient of variation)
            tempo_std = np.std(clean_bpm)
            tempo_confidence = max(0.0, 1.0 - (tempo_std / primary_tempo))
            
            # Tempo stability (consistency over time)
            tempo_stability = 1.0 - (np.std(clean_bpm) / np.mean(clean_bpm))
            tempo_stability = max(0.0, min(1.0, tempo_stability))
            
            return {
                'primary_tempo': primary_tempo,
                'tempo_confidence': float(tempo_confidence),
                'tempo_stability': float(tempo_stability)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tempo results: {e}")
            return {
                'primary_tempo': 0.0,
                'tempo_confidence': 0.0,
                'tempo_stability': 0.0
            }
    
    def _analyze_beat_patterns(self, beats: List, downbeats: List) -> Dict[str, Any]:
        """Analyze beat and downbeat patterns for meter estimation."""
        try:
            result = {
                'beat_intervals': [],
                'downbeat_intervals': [],
                'consistency': 0.0,
                'estimated_meter': '4/4',
                'meter_confidence': 0.0
            }
            
            if len(beats) < 2:
                return result
            
            # Calculate beat intervals
            beat_times = np.array(beats)
            beat_intervals = np.diff(beat_times)
            result['beat_intervals'] = beat_intervals.tolist()
            
            # Beat consistency (lower standard deviation = more consistent)
            if len(beat_intervals) > 1:
                consistency = 1.0 - (np.std(beat_intervals) / np.mean(beat_intervals))
                result['consistency'] = max(0.0, min(1.0, float(consistency)))
            
            # Analyze downbeats for meter
            if len(downbeats) >= 2:
                downbeat_times = np.array(downbeats)
                downbeat_intervals = np.diff(downbeat_times)
                result['downbeat_intervals'] = downbeat_intervals.tolist()
                
                # Estimate meter from beat/downbeat ratio
                if len(beats) > 0 and len(downbeats) > 0:
                    avg_beat_interval = np.mean(beat_intervals)
                    avg_downbeat_interval = np.mean(downbeat_intervals)
                    
                    # Estimate beats per measure
                    beats_per_measure = avg_downbeat_interval / avg_beat_interval
                    
                    # Classify common meters
                    if 3.5 <= beats_per_measure <= 4.5:
                        result['estimated_meter'] = '4/4'
                        result['meter_confidence'] = 0.8
                    elif 2.5 <= beats_per_measure <= 3.5:
                        result['estimated_meter'] = '3/4'
                        result['meter_confidence'] = 0.7
                    elif 1.5 <= beats_per_measure <= 2.5:
                        result['estimated_meter'] = '2/4'
                        result['meter_confidence'] = 0.6
                    else:
                        result['estimated_meter'] = 'complex'
                        result['meter_confidence'] = 0.3
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing beat patterns: {e}")
            return {
                'beat_intervals': [],
                'downbeat_intervals': [],
                'consistency': 0.0,
                'estimated_meter': '4/4',
                'meter_confidence': 0.0
            }
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Return standardized error result."""
        return {
            'success': False,
            'error': error_message,
            'tempo_bpm': 0.0,
            'tempo_confidence': 0.0,
            'beats': {'times': [], 'count': 0, 'intervals': [], 'consistency': 0.0},
            'downbeats': {'times': [], 'count': 0, 'intervals': []},
            'meter': {'time_signature': 'unknown', 'confidence': 0.0},
            'tempo_curve': {'values': [], 'stability': 0.0},
            'analysis_metadata': {
                'method': 'BeatNet',
                'error': error_message
            }
        }
    
    def cleanup(self):
        """Clean up BeatNet resources."""
        if self._beatnet_model:
            # BeatNet cleanup (if available)
            try:
                del self._beatnet_model
                self._beatnet_model = None
                self._model_loaded = False
                logger.info("BeatNet model cleaned up")
            except Exception as e:
                logger.warning(f"Error during BeatNet cleanup: {e}")