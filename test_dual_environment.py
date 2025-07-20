#!/usr/bin/env python3
"""
Test Dual Environment Key Detection

Quick test of the dual environment architecture with Essentia subprocess bridge.
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from plugins.core_plugins.key_finder import KeyFinderPlugin

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_dual_environment():
    """Test the dual environment key detection."""
    
    print("=== Testing Dual Environment Key Detection ===")
    
    # Create key finder plugin
    key_finder = KeyFinderPlugin()
    
    # Create synthetic test audio (C major scale)
    import numpy as np
    
    sample_rate = 22050
    duration = 5.0  # 5 seconds
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # C major scale frequencies
    c_major_freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    
    # Generate audio with C major scale
    audio_data = np.zeros_like(t)
    for freq in c_major_freqs:
        audio_data += 0.1 * np.sin(2 * np.pi * freq * t)
    
    audio_data = audio_data.astype(np.float32)
    
    print(f"✅ Generated {duration}s synthetic C major audio")
    
    # Test key detection
    print("\n🧪 Testing key detection...")
    
    result = key_finder.process(audio_data, sample_rate)
    
    print(f"\n🎯 Results:")
    print(f"  Success: {result['success']}")
    print(f"  Key: {result['key']}")
    print(f"  Mode: {result['mode']}")
    print(f"  Confidence: {result['confidence']:.3f}")
    print(f"  Method: {result['analysis_metadata']['method']}")
    print(f"  Environment: {result['analysis_metadata']['environment']}")
    print(f"  Processing Time: {result['processing_time_ms']}ms")
    
    if result['analysis_metadata']['environment'] == 'experimental':
        print("\n✅ SUCCESS: Experimental environment (Essentia) used!")
        print("✅ Dual environment architecture working correctly")
    else:
        print(f"\n⚠️  Using fallback environment: {result['analysis_metadata']['environment']}")
        print("   (This is expected if experimental environment setup failed)")
    
    return result['success']

if __name__ == '__main__':
    success = test_dual_environment()
    if success:
        print("\n🎉 Dual environment test completed")
    else:
        print("\n❌ Test failed")
        sys.exit(1)