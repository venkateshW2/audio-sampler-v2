#!/usr/bin/env python3
"""
Test Upload Endpoint

Test the file upload endpoint to debug the JSON parsing issue.
"""

import sys
import asyncio
import json
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_upload_endpoint():
    """Test the upload endpoint directly."""
    print("=== Testing Upload Endpoint ===")
    
    try:
        # Import modules
        from fastapi.testclient import TestClient
        from main import app, startup_event
        
        # Trigger startup manually
        await startup_event()
        print("✅ Startup completed")
        
        # Create test client
        client = TestClient(app)
        
        # Test file upload with a small test audio file
        test_file = "test_audio/VO.wav"
        if not Path(test_file).exists():
            print(f"❌ Test file {test_file} not found")
            return False
        
        print(f"🚀 Testing upload with {test_file}...")
        
        # Read test file
        with open(test_file, 'rb') as f:
            file_content = f.read()
        
        # Test upload endpoint
        response = client.post(
            "/api/analyze/upload",
            files={"file": ("test.wav", file_content, "audio/wav")}
        )
        
        print(f"Response status: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        
        if response.status_code == 200:
            try:
                result = response.json()
                print("✅ Upload successful!")
                print(f"   Analysis result: {result.get('message', 'No message')}")
                if 'analysis_result' in result:
                    analysis = result['analysis_result']
                    print(f"   Success: {analysis.get('success', False)}")
                    print(f"   Regions: {analysis.get('total_regions', 0)}")
                return True
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                print(f"Response text: {response.text[:500]}...")
                return False
        else:
            print(f"❌ Upload failed with status {response.status_code}")
            print(f"Response text: {response.text[:500]}...")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    # Install httpx first if needed
    try:
        import httpx
    except ImportError:
        print("Installing httpx for testing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "httpx"], check=True)
    
    success = asyncio.run(test_upload_endpoint())
    
    if success:
        print("\n✅ Upload endpoint test PASSED!")
    else:
        print("\n❌ Upload endpoint test FAILED")
        sys.exit(1)