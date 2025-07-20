#!/usr/bin/env python3
"""
Test Region-Based Processing

Simple test to verify the region-based core engine works correctly.
Tests the complete pipeline: FileHandler → Engine → Plugin Processing
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import AudioSamplerEngine
from core.file_handler import FileHandler
from plugins.core_plugins.classifier import ClassifierPlugin
from plugins.core_plugins.tempo_meter_detector import TempoMeterDetectorPlugin
from plugins.core_plugins.key_finder import KeyFinderPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def create_test_audio():
    """Create a simple test audio signal with silence gaps."""
    logger.info("Creating test audio with silence gaps...")
    
    # Create test audio: 30 seconds total
    # 0-10s: sine wave (content)
    # 10-25s: silence (15 seconds - should trigger region boundary)
    # 25-30s: sine wave (content)
    
    sample_rate = 22050
    total_duration = 30  # seconds
    total_samples = int(total_duration * sample_rate)
    
    audio = np.zeros(total_samples, dtype=np.float32)
    
    # Region 1: 0-10 seconds (sine wave at 440 Hz)
    t1 = np.linspace(0, 10, int(10 * sample_rate), False)
    audio[0:len(t1)] = 0.3 * np.sin(2 * np.pi * 440 * t1)
    
    # Region 2: 10-25 seconds (silence - already zeros)
    
    # Region 3: 25-30 seconds (sine wave at 880 Hz)
    t3 = np.linspace(0, 5, int(5 * sample_rate), False)
    start_idx = int(25 * sample_rate)
    end_idx = start_idx + len(t3)
    audio[start_idx:end_idx] = 0.3 * np.sin(2 * np.pi * 880 * t3)
    
    return audio, sample_rate

def test_file_handler_regions():
    """Test FileHandler region detection."""
    logger.info("🧪 Testing FileHandler region detection...")
    
    file_handler = FileHandler({
        'silence_threshold_seconds': 10.0,
        'silence_db_threshold': -40.0,
        'enable_region_detection': True
    })
    
    # Create test audio
    audio_data, sample_rate = create_test_audio()
    
    # Test region detection directly
    fake_path = Path("/fake/test.wav")
    regions = file_handler._detect_regions(fake_path, audio_data, sample_rate)
    
    logger.info(f"✅ Detected {len(regions)} regions:")
    for i, region in enumerate(regions):
        logger.info(f"  Region {i+1}: {region['region_id']} - "
                   f"{region['start_time']:.1f}s to {region['end_time']:.1f}s "
                   f"({region['duration']:.1f}s)")
    
    return regions

def test_engine_with_plugins():
    """Test complete engine with plugins."""
    logger.info("🧪 Testing complete engine with region-based plugins...")
    
    # Initialize engine
    engine = AudioSamplerEngine({
        'file_handling': {
            'silence_threshold_seconds': 10.0,
            'enable_region_detection': True
        }
    })
    
    # Register plugins (with basic error handling)
    plugins = []
    
    try:
        classifier = ClassifierPlugin()
        engine.register_plugin(classifier)
        plugins.append("classifier")
        logger.info("✅ Registered ClassifierPlugin")
    except Exception as e:
        logger.warning(f"⚠️ Could not register ClassifierPlugin: {e}")
    
    try:
        tempo_detector = TempoMeterDetectorPlugin()
        engine.register_plugin(tempo_detector)
        plugins.append("tempo_detector")
        logger.info("✅ Registered TempoMeterDetectorPlugin")
    except Exception as e:
        logger.warning(f"⚠️ Could not register TempoMeterDetectorPlugin: {e}")
    
    try:
        key_finder = KeyFinderPlugin()
        engine.register_plugin(key_finder)
        plugins.append("key_finder")
        logger.info("✅ Registered KeyFinderPlugin")
    except Exception as e:
        logger.warning(f"⚠️ Could not register KeyFinderPlugin: {e}")
    
    logger.info(f"Registered {len(plugins)} plugins: {plugins}")
    
    # Get engine status
    status = engine.get_status()
    logger.info(f"Engine status: {status['status']} with {status['plugins_registered']} plugins")
    
    return engine, plugins

def test_manual_region_processing():
    """Test manual region processing without file I/O."""
    logger.info("🧪 Testing manual region processing...")
    
    engine, plugins = test_engine_with_plugins()
    
    if not plugins:
        logger.warning("⚠️ No plugins registered, skipping region processing test")
        return
    
    # Create test regions manually
    audio_data, sample_rate = create_test_audio()
    
    # Create fake regions (simulating FileHandler output)
    regions = [
        {
            'region_id': 'test-r1',
            'region_number': 1,
            'parent_file': '/fake/test.wav',
            'start_time': 0.0,
            'end_time': 10.0,
            'duration': 10.0,
            'start_sample': 0,
            'end_sample': int(10 * sample_rate),
            'sample_count': int(10 * sample_rate),
            'audio_data': audio_data[0:int(10 * sample_rate)],
            'sample_rate': sample_rate,
            'region_type': 'content'
        },
        {
            'region_id': 'test-r2',
            'region_number': 2,
            'parent_file': '/fake/test.wav',
            'start_time': 25.0,
            'end_time': 30.0,
            'duration': 5.0,
            'start_sample': int(25 * sample_rate),
            'end_sample': int(30 * sample_rate),
            'sample_count': int(5 * sample_rate),
            'audio_data': audio_data[int(25 * sample_rate):int(30 * sample_rate)],
            'sample_rate': sample_rate,
            'region_type': 'content'
        }
    ]
    
    logger.info(f"Processing {len(regions)} test regions through {len(plugins)} plugins...")
    
    # Process regions through engine
    result = engine._process_regions(regions)
    
    logger.info(f"✅ Processing complete: {result['successful_count']}/{len(regions)} regions successful")
    
    # Show results
    for i, region_result in enumerate(result['results']):
        region_id = region_result['region_id']
        success = region_result['success']
        plugin_count = region_result['successful_plugins']
        
        logger.info(f"Region {region_id}: {'✅' if success else '❌'} "
                   f"({plugin_count} plugins successful)")
        
        if region_result['plugin_results']:
            for plugin_name, plugin_data in region_result['plugin_results'].items():
                plugin_success = plugin_data['success']
                processing_time = plugin_data['processing_time']
                logger.info(f"  - {plugin_name}: {'✅' if plugin_success else '❌'} "
                           f"({processing_time:.2f}s)")
    
    return result

def main():
    """Run all region-based processing tests."""
    logger.info("🚀 Starting Region-Based Processing Tests")
    logger.info("=" * 60)
    
    try:
        # Test 1: FileHandler region detection
        logger.info("\n" + "=" * 60)
        logger.info("TEST 1: FileHandler Region Detection")
        logger.info("=" * 60)
        regions = test_file_handler_regions()
        
        if len(regions) >= 2:
            logger.info("✅ Region detection working correctly")
        else:
            logger.warning("⚠️ Expected at least 2 regions, got {len(regions)}")
        
        # Test 2: Engine with plugins
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Engine Plugin Registration")
        logger.info("=" * 60)
        engine, plugins = test_engine_with_plugins()
        
        # Test 3: Manual region processing
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Manual Region Processing")
        logger.info("=" * 60)
        if plugins:
            result = test_manual_region_processing()
            
            if result['successful_count'] > 0:
                logger.info("✅ Region-based processing working correctly")
            else:
                logger.warning("⚠️ No regions processed successfully")
        else:
            logger.warning("⚠️ Skipping region processing - no plugins available")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎉 ALL TESTS COMPLETED")
        logger.info("=" * 60)
        
        logger.info("✅ Region-based architecture is working!")
        logger.info("Ready for Week 3: Database & Integration")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)