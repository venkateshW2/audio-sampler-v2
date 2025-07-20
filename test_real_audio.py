#!/usr/bin/env python3
"""
Test with Real Audio Files

Test the region-based processing with actual audio files from test_audio folder.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import AudioSamplerEngine
from plugins.core_plugins.classifier import ClassifierPlugin
from plugins.core_plugins.tempo_meter_detector import TempoMeterDetectorPlugin
from plugins.core_plugins.key_finder import KeyFinderPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def test_real_audio_file(audio_path):
    """Test processing with a real audio file."""
    logger.info(f"🎵 Testing with real audio file: {audio_path}")
    
    # Initialize engine
    engine = AudioSamplerEngine({
        'file_handling': {
            'silence_threshold_seconds': 10.0,
            'enable_region_detection': True
        }
    })
    
    # Register ALL core plugins to show complete analysis results
    try:
        classifier = ClassifierPlugin()
        engine.register_plugin(classifier)
        logger.info("✅ ClassifierPlugin registered successfully")
        
        tempo_meter = TempoMeterDetectorPlugin()
        engine.register_plugin(tempo_meter)
        logger.info("✅ TempoMeterDetectorPlugin registered successfully")
        
        key_finder = KeyFinderPlugin()
        engine.register_plugin(key_finder)
        logger.info("✅ KeyFinderPlugin registered successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to register plugins: {e}")
        return False
    
    # Process the real audio file
    try:
        result = engine.process_file(audio_path)
        
        logger.info(f"🎯 Processing Results:")
        logger.info(f"  Success: {result['success']}")
        logger.info(f"  Total Regions: {result['total_regions']}")
        logger.info(f"  Successful Regions: {result['successful_regions']}")
        logger.info(f"  Processing Time: {result['duration_seconds']:.2f}s")
        
        if result['regions']:
            logger.info(f"📊 Regions Found:")
            for region in result['regions']:
                logger.info(f"    {region['region_id']}: {region['start_time']:.1f}s - {region['end_time']:.1f}s ({region['duration']:.1f}s)")
        
        if result['region_analyses']:
            logger.info(f"🔍 Analysis Results:")
            for analysis in result['region_analyses']:
                region_id = analysis['region_id']
                success = analysis['success']
                logger.info(f"    {region_id}: {'✅' if success else '❌'}")
                
                if 'plugin_results' in analysis:
                    for plugin_name, plugin_data in analysis['plugin_results'].items():
                        plugin_success = plugin_data['success']
                        processing_time = plugin_data['processing_time']
                        logger.info(f"      - {plugin_name}: {'✅' if plugin_success else '❌'} ({processing_time:.2f}s)")
                        
                        # Show plugin-specific results
                        if plugin_success and 'data' in plugin_data:
                            data = plugin_data['data']
                            
                            # Classification results
                            if plugin_name == 'classifier':
                                if 'top_prediction' in data:
                                    logger.info(f"        Classification: {data['top_prediction']} (confidence: {data.get('top_confidence', 0.0):.3f})")
                                if 'music_probability' in data:
                                    logger.info(f"        Music Probability: {data['music_probability']:.3f}")
                                if 'speech_probability' in data:
                                    logger.info(f"        Speech Probability: {data['speech_probability']:.3f}")
                            
                            # Tempo & Meter results
                            elif plugin_name == 'tempo_meter_detector':
                                if 'tempo_bpm' in data:
                                    logger.info(f"        Tempo: {data['tempo_bpm']:.1f} BPM (confidence: {data.get('tempo_confidence', 0.0):.3f})")
                                if 'meter' in data:
                                    logger.info(f"        Meter: {data['meter']} (confidence: {data.get('meter_confidence', 0.0):.3f})")
                                if 'downbeat_times' in data and len(data['downbeat_times']) > 0:
                                    logger.info(f"        Downbeats: {len(data['downbeat_times'])} detected")
                            
                            # Key detection results  
                            elif plugin_name == 'key_finder':
                                if 'key' in data:
                                    logger.info(f"        Key: {data['key']} (confidence: {data.get('key_confidence', 0.0):.3f})")
                                if 'mode' in data:
                                    logger.info(f"        Mode: {data['mode']}")
                                if 'scale' in data:
                                    logger.info(f"        Scale: {data['scale']}")
                            
                            # Generic fallback for any other data
                            elif 'classification' in data:
                                logger.info(f"        Classification: {data['classification']} (confidence: {data.get('confidence', 0.0):.3f})")
        
        return result['success']
        
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Test with various audio files from test_audio folder."""
    test_audio_dir = Path("test_audio")
    
    if not test_audio_dir.exists():
        logger.error("test_audio directory not found!")
        return 1
    
    # Get available audio files
    audio_files = list(test_audio_dir.glob("*.wav"))
    
    if not audio_files:
        logger.error("No .wav files found in test_audio directory!")
        return 1
    
    logger.info(f"Found {len(audio_files)} audio files to test:")
    for audio_file in audio_files:
        logger.info(f"  - {audio_file.name}")
    
    # Test with first audio file
    test_file = audio_files[0]
    logger.info(f"\n🚀 Testing with: {test_file}")
    logger.info("=" * 60)
    
    success = test_real_audio_file(test_file)
    
    logger.info("\n" + "=" * 60)
    if success:
        logger.info("🎉 Real audio test PASSED!")
        logger.info("✅ PaSST loading is now working correctly!")
    else:
        logger.error("❌ Real audio test FAILED!")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)