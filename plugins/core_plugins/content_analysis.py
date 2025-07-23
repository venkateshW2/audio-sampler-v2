"""
Audio Intelligence Sampler v2 - Content Analysis Plugin

Timeline-based content change detection using FeatureExtractor's cached features.
Performs mathematical analysis on cached spectrograms/features to detect transitions
without duplicating feature extraction.

Use case: Generate timeline segments where PaSST should classify content
Example: Thunder intro → Music should create 2 segments for separate classification

Architecture principles:
- Uses FeatureExtractor's cached features (no duplicate librosa calls)
- Mathematical analysis of spectrograms, onsets, RMS for transition detection
- Creates timeline segments at transition points
- SACRED: Never crashes, always returns timeline analysis
"""

import logging
import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..base_plugin import BasePlugin, PluginRequirements

logger = logging.getLogger(__name__)


class ContentAnalysisPlugin(BasePlugin):
    """
    Timeline-based content analysis using FeatureExtractor's cached features.
    
    Analyzes cached spectrograms, onsets, RMS, and chromagrams to detect
    content transitions without re-extracting features.
    
    Returns timeline segments for PaSST to classify at specific positions.
    """
    
    def __init__(self):
        self.plugin_name = "content_analysis"
        self.version = "1.0.0-cached-features"
        super().__init__()
        
    def get_name(self) -> str:
        """Return plugin name."""
        return self.plugin_name
    
    def get_version(self) -> str:
        """Return plugin version."""
        return self.version
    
    def get_requirements(self) -> PluginRequirements:
        """Return resource requirements for mathematical analysis."""
        return PluginRequirements(
            gpu_memory_mb=0,  # Pure mathematical analysis
            cpu_cores=1,
            ram_mb=128,
            processing_time_estimate=0.5
        )
    
    def can_process(self, audio_data: np.ndarray, sample_rate: int, 
                   resource_status: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if content analysis can be performed."""
        try:
            if len(audio_data) == 0:
                return False, "Empty audio data"
            
            if len(audio_data) / sample_rate < 2.0:
                return False, "Audio too short for timeline analysis (min 2 seconds)"
            
            return True, "Ready for cached feature analysis"
            
        except Exception as e:
            logger.error(f"Error checking content analysis capability: {e}")
            return False, f"Capability check failed: {str(e)}"
    
    def process(self, audio_data: np.ndarray, sample_rate: int, 
               file_path: Optional[Path] = None, 
               feature_cache_id: Optional[str] = None,
               **kwargs) -> Dict[str, Any]:
        """
        Analyze cached features for timeline transitions.
        
        Uses FeatureExtractor's cached features to detect content changes
        and create timeline segments for intelligent classification.
        """
        start_time = time.time()
        
        try:
            # Get cached features from FeatureExtractor
            if not feature_cache_id:
                return self._error_result("No feature cache ID provided - FeatureExtractor must run first")
            
            cached_features = self._get_cached_features(feature_cache_id)
            if not cached_features:
                return self._error_result(f"No cached features found for ID: {feature_cache_id}")
            
            logger.info(f"Analyzing cached features for timeline transitions in {len(audio_data)/sample_rate:.1f}s audio")
            
            # Step 1: Analyze cached features for transitions
            transitions = self._detect_transitions_from_cache(cached_features, sample_rate)
            
            # Step 2: Create timeline segments for PaSST classification
            timeline_segments = self._create_classification_segments(transitions, len(audio_data), sample_rate)
            
            # Step 3: Analyze segment characteristics using cached data
            segment_analysis = self._analyze_segments_from_cache(timeline_segments, cached_features)
            
            processing_time = time.time() - start_time
            
            result = {
                'success': True,
                'timeline_segments': timeline_segments,
                'segment_analysis': segment_analysis, 
                'content_transitions': transitions,
                'total_segments': len(timeline_segments),
                'analysis_duration_s': len(audio_data) / sample_rate,
                'cached_features_used': list(cached_features.keys()),
                'transition_detection': {
                    'method': 'cached-feature-analysis',
                    'features_analyzed': len(cached_features),
                    'transitions_detected': len(transitions),
                    'segments_created': len(timeline_segments)
                },
                'processing_time_ms': int(processing_time * 1000),
                'analysis_metadata': {
                    'plugin_version': self.version,
                    'sample_rate': sample_rate,
                    'feature_cache_id': feature_cache_id,
                    'processing_time_seconds': processing_time
                }
            }
            
            logger.info(f"✅ Timeline analysis: {len(transitions)} transitions detected, "
                       f"{len(timeline_segments)} segments for classification ({processing_time:.2f}s)")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Content analysis error: {e}")
            result = self._error_result(f"Content analysis failed: {str(e)}")
            result['processing_time_ms'] = int(processing_time * 1000)
            return result
    
    def _get_cached_features(self, feature_cache_id: str) -> Dict[str, Any]:
        """Get cached features from FeatureExtractor."""
        try:
            from .feature_extractor import FeatureExtractorPlugin
            
            # Create temporary instance to access cache
            temp_fe = FeatureExtractorPlugin()
            cached_features = temp_fe.get_cached_features(feature_cache_id)
            
            if cached_features:
                logger.debug(f"Retrieved cached features: {list(cached_features.keys())}")
            else:
                logger.warning(f"No cached features found for ID: {feature_cache_id}")
            
            return cached_features
            
        except Exception as e:
            logger.error(f"Error accessing cached features: {e}")
            return {}
    
    def _detect_transitions_from_cache(self, cached_features: Dict[str, Any], sample_rate: int) -> List[Dict[str, Any]]:
        """Detect content transitions using mathematical analysis of cached features."""
        transitions = []
        
        try:
            # Method 1: Onset-based transition detection
            if 'onsets_librosa' in cached_features:
                onset_data = cached_features['onsets_librosa']
                onset_transitions = self._analyze_onset_transitions(onset_data, sample_rate)
                transitions.extend(onset_transitions)
                logger.debug(f"Onset analysis found {len(onset_transitions)} transitions")
            
            # Method 2: Spectrogram energy change detection
            if 'spectrogram_nnaudio' in cached_features or 'spectrogram_librosa' in cached_features:
                spectrogram = cached_features.get('spectrogram_nnaudio', cached_features.get('spectrogram_librosa'))
                if spectrogram is not None:
                    spectral_transitions = self._analyze_spectral_transitions(spectrogram, sample_rate)
                    transitions.extend(spectral_transitions)
                    logger.debug(f"Spectral analysis found {len(spectral_transitions)} transitions")
            
            # Method 3: Harmonic content change detection
            if 'hpss_librosa' in cached_features:
                hpss_data = cached_features['hpss_librosa']
                harmonic_transitions = self._analyze_harmonic_transitions(hpss_data, sample_rate)
                transitions.extend(harmonic_transitions)
                logger.debug(f"Harmonic analysis found {len(harmonic_transitions)} transitions")
            
            # Method 4: Chromagram-based musical content detection
            if 'chromagram_librosa' in cached_features:
                chromagram = cached_features['chromagram_librosa']
                chroma_transitions = self._analyze_chroma_transitions(chromagram, sample_rate)
                transitions.extend(chroma_transitions)
                logger.debug(f"Chroma analysis found {len(chroma_transitions)} transitions")
            
            # Remove duplicates and sort by time
            transitions = self._consolidate_transitions(transitions)
            
            # Debug: Log detected transitions with their times and types
            logger.info(f"Mathematical analysis detected {len(transitions)} content transitions")
            for i, transition in enumerate(transitions[:10]):  # Log first 10 transitions
                logger.info(f"  Transition {i+1}: {transition['time']:.1f}s - {transition['transition_type']} (confidence: {transition['confidence']:.2f})")
            
        except Exception as e:
            logger.error(f"Transition detection error: {e}")
            transitions = []
        
        return transitions
    
    def _analyze_onset_transitions(self, onset_data: Dict[str, Any], sample_rate: int) -> List[Dict[str, Any]]:
        """Analyze onset density changes to detect content transitions."""
        transitions = []
        
        try:
            onset_times = onset_data.get('times', np.array([]))
            onset_strength = onset_data.get('strength', np.array([]))
            
            if len(onset_times) == 0 or len(onset_strength) == 0:
                return []
            
            # Analyze onset density in time windows
            window_duration = 2.0  # 2-second windows
            hop_duration = 0.5     # 0.5-second hops
            
            max_time = onset_times[-1] if len(onset_times) > 0 else 10.0
            window_starts = np.arange(0, max_time - window_duration, hop_duration)
            
            onset_densities = []
            window_centers = []
            
            for start_time in window_starts:
                end_time = start_time + window_duration
                center_time = start_time + window_duration / 2
                
                # Count onsets in this window
                window_mask = (onset_times >= start_time) & (onset_times <= end_time)
                onset_count = np.sum(window_mask)
                onset_density = onset_count / window_duration
                
                onset_densities.append(onset_density)
                window_centers.append(center_time)
            
            onset_densities = np.array(onset_densities)
            window_centers = np.array(window_centers)
            
            # Detect significant changes in onset density
            if len(onset_densities) > 3:
                # Calculate density changes
                density_changes = np.abs(np.diff(onset_densities))
                change_threshold = np.mean(density_changes) + 2 * np.std(density_changes)
                
                # Find significant changes
                significant_changes = np.where(density_changes > change_threshold)[0]
                
                for change_idx in significant_changes:
                    if change_idx + 1 < len(window_centers):
                        transition_time = window_centers[change_idx + 1]
                        density_before = onset_densities[change_idx]
                        density_after = onset_densities[change_idx + 1]
                        
                        # Classify transition type
                        if density_after > density_before * 2:
                            transition_type = "rhythmic_onset"
                        elif density_after < density_before * 0.5:
                            transition_type = "rhythmic_decrease"
                        else:
                            transition_type = "rhythmic_change"
                        
                        transitions.append({
                            'time': float(transition_time),
                            'transition_score': float(density_changes[change_idx]),
                            'transition_type': transition_type,
                            'confidence': min(float(density_changes[change_idx] / change_threshold), 1.0),
                            'analysis_method': 'onset_density'
                        })
            
        except Exception as e:
            logger.error(f"Onset transition analysis error: {e}")
        
        return transitions
    
    def _analyze_spectral_transitions(self, spectrogram: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Analyze spectrogram energy changes to detect content transitions."""
        transitions = []
        
        try:
            if spectrogram is None or len(spectrogram.shape) != 2 or spectrogram.shape[1] < 10:
                return []
            
            # Calculate spectral energy over time
            spectral_energy = np.sum(spectrogram, axis=0)  # Sum across frequency bins
            
            # Smooth the energy curve to reduce noise
            from scipy.ndimage import gaussian_filter1d
            smoothed_energy = gaussian_filter1d(spectral_energy, sigma=2.0)
            
            # Calculate energy changes
            energy_changes = np.abs(np.diff(smoothed_energy))
            
            # Detect significant energy changes
            change_threshold = np.mean(energy_changes) + 2.5 * np.std(energy_changes)
            significant_changes = np.where(energy_changes > change_threshold)[0]
            
            # Convert frame indices to time
            hop_length = 512  # Standard hop length from FeatureExtractor
            
            for change_idx in significant_changes:
                if change_idx + 1 < len(spectral_energy):
                    transition_time = (change_idx + 1) * hop_length / sample_rate
                    
                    energy_before = smoothed_energy[change_idx]
                    energy_after = smoothed_energy[change_idx + 1]
                    
                    # Classify transition type
                    if energy_after > energy_before * 2:
                        transition_type = "energy_increase"
                    elif energy_after < energy_before * 0.5:
                        transition_type = "energy_decrease"
                    else:
                        transition_type = "energy_change"
                    
                    transitions.append({
                        'time': float(transition_time),
                        'transition_score': float(energy_changes[change_idx]),
                        'transition_type': transition_type,
                        'confidence': min(float(energy_changes[change_idx] / change_threshold), 1.0),
                        'analysis_method': 'spectral_energy'
                    })
            
        except Exception as e:
            logger.error(f"Spectral transition analysis error: {e}")
        
        return transitions
    
    def _analyze_harmonic_transitions(self, hpss_data: Dict[str, Any], sample_rate: int) -> List[Dict[str, Any]]:
        """Analyze harmonic vs percussive content changes."""
        transitions = []
        
        try:
            harmonic = hpss_data.get('harmonic', np.array([]))
            percussive = hpss_data.get('percussive', np.array([]))
            
            if len(harmonic) == 0 or len(percussive) == 0:
                return []
            
            # Calculate RMS energy for harmonic and percussive components in windows
            window_samples = int(2.0 * sample_rate)  # 2-second windows
            hop_samples = int(0.5 * sample_rate)     # 0.5-second hops
            
            window_starts = range(0, len(harmonic) - window_samples, hop_samples)
            
            harmonic_ratios = []
            window_times = []
            
            for start_idx in window_starts:
                end_idx = start_idx + window_samples
                center_time = (start_idx + window_samples // 2) / sample_rate
                
                # Calculate RMS energy for both components
                h_rms = np.sqrt(np.mean(harmonic[start_idx:end_idx] ** 2))
                p_rms = np.sqrt(np.mean(percussive[start_idx:end_idx] ** 2))
                
                # Calculate harmonic/percussive ratio
                if p_rms > 1e-8:
                    ratio = h_rms / p_rms
                else:
                    ratio = h_rms / 1e-8
                
                harmonic_ratios.append(ratio)
                window_times.append(center_time)
            
            harmonic_ratios = np.array(harmonic_ratios)
            window_times = np.array(window_times)
            
            # Detect significant changes in harmonic/percussive ratio
            if len(harmonic_ratios) > 3:
                ratio_changes = np.abs(np.diff(np.log(harmonic_ratios + 1e-8)))
                change_threshold = np.mean(ratio_changes) + 2 * np.std(ratio_changes)
                
                significant_changes = np.where(ratio_changes > change_threshold)[0]
                
                for change_idx in significant_changes:
                    if change_idx + 1 < len(window_times):
                        transition_time = window_times[change_idx + 1]
                        ratio_before = harmonic_ratios[change_idx]
                        ratio_after = harmonic_ratios[change_idx + 1]
                        
                        # Classify transition type
                        if ratio_after > ratio_before * 2:
                            transition_type = "music_onset"  # More harmonic
                        elif ratio_after < ratio_before * 0.5:
                            transition_type = "percussive_onset"  # More percussive
                        else:
                            transition_type = "harmonic_change"
                        
                        transitions.append({
                            'time': float(transition_time),
                            'transition_score': float(ratio_changes[change_idx]),
                            'transition_type': transition_type,
                            'confidence': min(float(ratio_changes[change_idx] / change_threshold), 1.0),
                            'analysis_method': 'harmonic_percussive'
                        })
            
        except Exception as e:
            logger.error(f"Harmonic transition analysis error: {e}")
        
        return transitions
    
    def _analyze_chroma_transitions(self, chromagram: np.ndarray, sample_rate: int) -> List[Dict[str, Any]]:
        """Analyze chromagram changes to detect musical content transitions."""
        transitions = []
        
        try:
            if chromagram is None or len(chromagram.shape) != 2 or chromagram.shape[1] < 10:
                return []
            
            # Calculate chroma stability over time (musical content indicator)
            chroma_stability = []
            window_size = 20  # frames (~1 second at hop_length=512)
            
            for i in range(window_size, chromagram.shape[1] - window_size):
                window = chromagram[:, i-window_size:i+window_size]
                # Calculate variance across time (low variance = stable musical content)
                stability = 1.0 / (np.mean(np.var(window, axis=1)) + 1e-8)
                chroma_stability.append(stability)
            
            chroma_stability = np.array(chroma_stability)
            
            # Detect changes in musical stability
            if len(chroma_stability) > 5:
                stability_changes = np.abs(np.diff(chroma_stability))
                change_threshold = np.mean(stability_changes) + 2 * np.std(stability_changes)
                
                significant_changes = np.where(stability_changes > change_threshold)[0]
                
                hop_length = 512  # From FeatureExtractor
                
                for change_idx in significant_changes:
                    # Convert to time (accounting for window offset)
                    frame_idx = change_idx + window_size + 1
                    transition_time = frame_idx * hop_length / sample_rate
                    
                    stability_before = chroma_stability[change_idx]
                    stability_after = chroma_stability[change_idx + 1] if change_idx + 1 < len(chroma_stability) else stability_before
                    
                    # Classify transition type
                    if stability_after > stability_before * 2:
                        transition_type = "musical_stabilization"
                    elif stability_after < stability_before * 0.5:
                        transition_type = "musical_destabilization"
                    else:
                        transition_type = "musical_change"
                    
                    transitions.append({
                        'time': float(transition_time),
                        'transition_score': float(stability_changes[change_idx]), 
                        'transition_type': transition_type,
                        'confidence': min(float(stability_changes[change_idx] / change_threshold), 1.0),
                        'analysis_method': 'chroma_stability'
                    })
            
        except Exception as e:
            logger.error(f"Chroma transition analysis error: {e}")
        
        return transitions
    
    def _consolidate_transitions(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate transitions and consolidate nearby ones."""
        if not transitions:
            return []
        
        # Sort by time
        transitions = sorted(transitions, key=lambda x: x['time'])
        
        consolidated = []
        min_separation = 2.0  # Minimum 2 seconds between transitions
        
        for transition in transitions:
            # Check if this transition is too close to the last one
            if not consolidated or transition['time'] - consolidated[-1]['time'] >= min_separation:
                consolidated.append(transition)
            else:
                # Merge with previous transition (keep higher confidence)
                if transition['confidence'] > consolidated[-1]['confidence']:
                    consolidated[-1] = transition
        
        return consolidated
    
    def _create_classification_segments(self, transitions: List[Dict[str, Any]], 
                                     audio_length: int, sample_rate: int) -> List[Dict[str, Any]]:
        """Create timeline segments for PaSST classification based on transitions."""
        segments = []
        
        try:
            audio_duration = audio_length / sample_rate
            
            # Create segment boundaries
            segment_starts = [0.0]  # Always start at beginning
            
            # Add transition points
            for transition in transitions:
                segment_starts.append(transition['time'])
            
            segment_starts.append(audio_duration)  # Always end at end
            
            # Remove duplicates and sort
            segment_starts = sorted(list(set(segment_starts)))
            
            # Create segments
            for i in range(len(segment_starts) - 1):
                start_time = segment_starts[i]
                end_time = segment_starts[i + 1]
                duration = end_time - start_time
                
                # Only create segments with minimum duration (1.5 seconds for short events like thunder)
                # PaSST can handle shorter segments, we just need enough audio for meaningful classification
                if duration >= 1.5:
                    segments.append({
                        'segment_id': f'timeline_seg_{i+1:02d}',
                        'start_time': float(start_time),
                        'end_time': float(end_time),
                        'duration': float(duration),
                        'start_sample': int(start_time * sample_rate),
                        'end_sample': int(end_time * sample_rate),
                        'segment_number': i + 1,
                        'classification_ready': True
                    })
            
            # Ensure we have at least one segment (entire audio)
            if not segments:
                segments.append({
                    'segment_id': 'timeline_seg_01',
                    'start_time': 0.0,
                    'end_time': audio_duration,
                    'duration': audio_duration,
                    'start_sample': 0,
                    'end_sample': audio_length,
                    'segment_number': 1,
                    'classification_ready': True
                })
            
            logger.info(f"Created {len(segments)} timeline segments ready for PaSST classification")
            for i, segment in enumerate(segments[:5]):  # Log first 5 segments
                logger.info(f"  Segment {segment['segment_id']}: {segment['start_time']:.1f}-{segment['end_time']:.1f}s ({segment['duration']:.1f}s)")
            
        except Exception as e:
            logger.error(f"Segment creation error: {e}")
            # Fallback: single segment
            segments = [{
                'segment_id': 'timeline_seg_01',
                'start_time': 0.0,
                'end_time': audio_length / sample_rate,
                'duration': audio_length / sample_rate,
                'start_sample': 0,
                'end_sample': audio_length,
                'segment_number': 1,
                'classification_ready': True
            }]
        
        return segments
    
    def _analyze_segments_from_cache(self, segments: List[Dict[str, Any]], 
                                   cached_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze segment characteristics using cached features."""
        segment_analysis = []
        
        try:
            for segment in segments:
                analysis = {
                    'segment_id': segment['segment_id'],
                    'content_prediction': 'unknown',
                    'classification_priority': 'normal',
                    'feature_summary': {}
                }
                
                # Basic content prediction based on available features
                if 'hpss_librosa' in cached_features:
                    analysis['content_prediction'] = 'mixed_content'
                    analysis['classification_priority'] = 'high'
                
                if 'chromagram_librosa' in cached_features:
                    # Presence of stable chromagram suggests musical content
                    analysis['feature_summary']['has_chromagram'] = True
                    
                if 'onsets_librosa' in cached_features:
                    onset_data = cached_features['onsets_librosa']
                    onset_count = len(onset_data.get('times', []))
                    analysis['feature_summary']['onset_count'] = onset_count
                    
                    if onset_count > 10:
                        analysis['content_prediction'] = 'rhythmic_content'
                
                segment_analysis.append(analysis)
                
        except Exception as e:
            logger.error(f"Segment analysis error: {e}")
            segment_analysis = [{'segment_id': seg['segment_id'], 'content_prediction': 'unknown'} for seg in segments]
        
        return segment_analysis
    
    def _error_result(self, error_message: str) -> Dict[str, Any]:
        """Return standardized error result."""
        return {
            'success': False,
            'error': error_message,
            'timeline_segments': [],
            'segment_analysis': [],
            'content_transitions': [],
            'total_segments': 0,
            'analysis_metadata': {
                'plugin_version': self.version,
                'error': error_message
            }
        }
    
    def cleanup(self):
        """Clean up content analysis resources."""
        try:
            logger.info("Content analysis plugin cleaned up")
        except Exception as e:
            logger.warning(f"Error during content analysis cleanup: {e}")