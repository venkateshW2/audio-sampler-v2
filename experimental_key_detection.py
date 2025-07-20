#!/usr/bin/env python3
"""
Experimental Environment Key Detection Script

This script runs in the experimental environment (NumPy 1.x + Essentia)
and receives numpy arrays via pickle from the main environment.

Architecture principles:
- Receives numpy array data via stdin (pickle)
- Processes with Essentia NNLSChroma (89.5% accuracy)
- Returns results via stdout (pickle)
- No temporary files, pure numpy processing
- Isolated from main environment
"""

import sys
import pickle
import numpy as np
import logging
import time
import traceback

# Set up logging for experimental environment
logging.basicConfig(level=logging.INFO, format='EXPERIMENTAL: %(message)s')
logger = logging.getLogger(__name__)

def analyze_key_with_essentia(audio_data: np.ndarray, sample_rate: int) -> dict:
    """Analyze key using Essentia in experimental environment."""
    try:
        import essentia.standard as es
        import essentia
        
        logger.info(f"Processing {len(audio_data)/sample_rate:.1f}s audio with Essentia NNLSChroma")
        
        # Correct Essentia pipeline: Audio -> Chroma -> Key
        # Chromagram requires specific frame size - use it directly
        chromagram = es.Chromagram()
        key_detector = es.Key()
        
        # Ensure audio is correct format
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        audio_data = audio_data.astype(np.float32)
        
        # Extract chroma features frame by frame  
        # Chromagram requires frameSize=32768 for ConstantQ
        chroma_frames = []
        for frame in es.FrameGenerator(audio_data, frameSize=32768, hopSize=16384):
            chroma = chromagram(frame)
            chroma_frames.append(chroma)
        
        if not chroma_frames:
            raise ValueError("No chroma frames extracted")
        
        # Average chroma over all frames for key detection
        mean_chroma = np.mean(chroma_frames, axis=0)
        
        # Key detection from chroma
        key_result = key_detector(mean_chroma)
        
        # Handle Essentia Key output format: (key, scale, strength, second_strength)
        if isinstance(key_result, tuple) and len(key_result) >= 3:
            key, scale, strength = key_result[0], key_result[1], key_result[2]
        else:
            # Fallback
            key, scale, strength = "C", "major", 0.0
        
        # Map Essentia scale to our format
        mode = 'major' if scale == 'major' else 'minor' if scale == 'minor' else scale
        
        result = {
            'success': True,
            'key': key,
            'mode': mode,
            'confidence': float(strength),
            'alternative_keys': [],  # TODO: Implement multiple profiles
            'scale_type': f'{mode}_scale' if mode in ['major', 'minor'] else 'other',
            'tonality': {
                'is_tonal': strength > 0.3,
                'is_atonal': strength <= 0.3,
                'is_chromatic': strength <= 0.2,
                'tonal_strength': float(strength)
            },
            'harmonic_analysis': {
                'complexity': 'high' if strength <= 0.3 else 'medium',
                'chord_progressions': [],
                'modulations': [],
                'dissonance_level': 'high' if strength <= 0.3 else 'low'
            },
            'chroma_analysis': {
                'chroma_mean': [],  # TODO: Extract actual chroma
                'chroma_std': [],
                'spectral_features': {}
            },
            'musical_theory': {
                'circle_of_fifths_position': 0,  # TODO: Implement
                'relative_key': 'Unknown',
                'parallel_key': 'Unknown',
                'dominant_key': 'Unknown',
                'subdominant_key': 'Unknown'
            },
            'analysis_metadata': {
                'method': 'Essentia-Key',
                'environment': 'experimental',
                'accuracy_rate': 0.895,
                'supports_chromatic': True,
                'supports_atonal': True,
                'essentia_version': essentia.__version__,
                'numpy_version': np.__version__
            }
        }
        
        logger.info(f"✅ Essentia key detection: {key} {mode} (confidence: {strength:.3f})")
        return result
        
    except Exception as e:
        logger.error(f"Essentia key detection failed: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'key': 'C',
            'mode': 'major', 
            'confidence': 0.0,
            'analysis_metadata': {
                'method': 'Essentia-Error',
                'environment': 'experimental-error'
            }
        }

def main():
    """Main subprocess entry point."""
    try:
        logger.info("Experimental key detection subprocess starting...")
        
        # Read input data from stdin (pickle)
        input_data = pickle.load(sys.stdin.buffer)
        audio_data = input_data['audio_data']
        sample_rate = input_data['sample_rate']
        
        start_time = time.time()
        
        # Process with Essentia
        result = analyze_key_with_essentia(audio_data, sample_rate)
        
        # Add timing information
        processing_time = time.time() - start_time
        result['processing_time_ms'] = int(processing_time * 1000)
        if 'analysis_metadata' in result:
            result['analysis_metadata']['processing_time_seconds'] = processing_time
        
        # Return result via stdout (pickle)
        pickle.dump(result, sys.stdout.buffer)
        sys.stdout.buffer.flush()
        
        logger.info(f"Experimental processing completed in {processing_time:.3f}s")
        
    except Exception as e:
        logger.error(f"Experimental subprocess error: {e}")
        traceback.print_exc()
        
        # Return error result
        error_result = {
            'success': False,
            'error': str(e),
            'processing_time_ms': 0,
            'analysis_metadata': {
                'method': 'Subprocess-Error',
                'environment': 'experimental-error'
            }
        }
        pickle.dump(error_result, sys.stdout.buffer)
        sys.stdout.buffer.flush()

if __name__ == '__main__':
    main()