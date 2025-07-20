#!/usr/bin/env python3
"""
Test the subprocess bridge to experimental environment
"""

import numpy as np
import pickle
import subprocess
import sys
import os

def test_subprocess_bridge():
    """Test the complete subprocess bridge."""
    print("Testing subprocess bridge to experimental environment...")
    
    try:
        # Create test audio data
        sample_rate = 22050
        duration = 3.0  # 3 seconds
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Generate C major chord (C4, E4, G4)
        audio_data = (
            0.3 * np.sin(2 * np.pi * 261.63 * t) +  # C4
            0.3 * np.sin(2 * np.pi * 329.63 * t) +  # E4  
            0.3 * np.sin(2 * np.pi * 392.00 * t)    # G4
        ).astype(np.float32)
        
        print(f"✅ Generated {duration}s test audio (C major chord)")
        
        # Prepare input data
        input_data = {
            'audio_data': audio_data,
            'sample_rate': sample_rate
        }
        
        # Run subprocess
        experimental_python = "/mnt/2w12-data/audio-sampler-v2/miniconda/envs/audio-sampler-experimental/bin/python"
        script_path = "/mnt/2w12-data/audio-sampler-v2/experimental_key_detection.py"
        
        print("🚀 Starting experimental subprocess...")
        
        process = subprocess.Popen(
            [experimental_python, script_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Send input data
        input_bytes = pickle.dumps(input_data)
        stdout, stderr = process.communicate(input=input_bytes, timeout=30)
        
        if process.returncode != 0:
            print(f"❌ Subprocess failed with return code {process.returncode}")
            print(f"Stderr: {stderr.decode()}")
            return False
        
        # Parse result
        result = pickle.loads(stdout)
        
        print("✅ Subprocess completed successfully")
        print(f"Result: {result}")
        
        if result.get('success', False):
            key = result.get('key', 'unknown')
            mode = result.get('mode', 'unknown')
            confidence = result.get('confidence', 0.0)
            method = result.get('analysis_metadata', {}).get('method', 'unknown')
            
            print(f"🎵 Detected: {key} {mode} (confidence: {confidence:.3f})")
            print(f"📊 Method: {method}")
            print(f"⏱️  Processing time: {result.get('processing_time_ms', 0)}ms")
            
            return True
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Subprocess timeout")
        return False
    except Exception as e:
        print(f"❌ Bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("=== Subprocess Bridge Test ===")
    success = test_subprocess_bridge()
    
    if success:
        print("\n✅ Subprocess bridge test PASSED")
        print("✅ Experimental environment integration working")
    else:
        print("\n❌ Subprocess bridge test FAILED")
        sys.exit(1)