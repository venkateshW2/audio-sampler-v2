#!/usr/bin/env python3
"""
Test script to verify backend waveform data generation
"""

import requests
import json
from pathlib import Path

def test_waveform_generation():
    """Test that backend generates waveform data along with analysis"""
    
    print("🧪 Testing backend waveform data generation...")
    
    # Test file upload with waveform generation
    test_file = Path("test_audio/VO.wav")
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return False
        
    try:
        # Upload file for analysis
        with open(test_file, 'rb') as f:
            files = {'file': (test_file.name, f, 'audio/wav')}
            response = requests.post('http://localhost:8002/api/analyze/upload', files=files)
        
        if response.status_code != 200:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
        result = response.json()
        print("✅ Upload successful")
        
        # Check if analysis_result exists
        if 'analysis_result' not in result:
            print("❌ No analysis_result in response")
            return False
            
        analysis = result['analysis_result']
        
        # Check if waveform_data exists
        if 'waveform_data' not in analysis:
            print("❌ No waveform_data in analysis result")
            print(f"Available keys: {list(analysis.keys())}")
            return False
            
        waveform_data = analysis['waveform_data']
        print("✅ Waveform data found in response")
        
        # Check waveform data structure
        required_keys = ['duration', 'overview', 'zoom']
        missing_keys = [key for key in required_keys if key not in waveform_data]
        
        if missing_keys:
            print(f"❌ Missing waveform data keys: {missing_keys}")
            return False
            
        print(f"✅ Waveform data structure complete")
        
        # Check data content
        duration = waveform_data.get('duration', 0)
        overview_samples = waveform_data.get('overview', {}).get('samples', 0)
        zoom_samples = waveform_data.get('zoom', {}).get('samples', 0)
        
        print(f"📊 Waveform data stats:")
        print(f"  Duration: {duration:.2f}s")
        print(f"  Overview samples: {overview_samples}")
        print(f"  Zoom samples: {zoom_samples}")
        
        if duration > 0 and overview_samples > 0 and zoom_samples > 0:
            print("✅ Waveform data contains valid content")
            
            # Check if data arrays exist
            overview_data = waveform_data.get('overview', {}).get('data', [])
            zoom_data = waveform_data.get('zoom', {}).get('data', [])
            
            if len(overview_data) > 0 and len(zoom_data) > 0:
                print(f"✅ Waveform arrays populated: {len(overview_data)} overview, {len(zoom_data)} zoom points")
                return True
            else:
                print("❌ Waveform data arrays are empty")
                return False
        else:
            print("❌ Waveform data appears to be empty or invalid")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_waveform_generation()
    if success:
        print("\n🎉 Backend waveform generation test PASSED!")
    else:
        print("\n💥 Backend waveform generation test FAILED!")