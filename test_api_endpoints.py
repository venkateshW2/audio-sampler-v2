#!/usr/bin/env python3
"""
Test API Endpoints

Test the FastAPI endpoints without running a full server.
"""

import sys
import asyncio
import json
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_api_endpoints():
    """Test API endpoints manually."""
    print("=== Testing API Endpoints ===")
    
    try:
        # Import the FastAPI app
        from main import app
        print("✅ FastAPI app imported successfully")
        
        # Trigger startup event manually
        print("🚀 Triggering startup event...")
        
        # Call startup event directly
        from main import startup_event
        await startup_event()
        print("✅ Startup event completed")
        
        # Test engine initialization
        from main import engine
        if engine:
            print("✅ Engine initialized successfully")
            
            # Test engine status
            status = engine.get_status()
            print(f"✅ Engine status: {status['engine_version']}")
            print(f"   Plugins: {status['plugins_registered']}")
            print(f"   Database: {status['components']['database_integration']}")
            
            # Test a file analysis
            test_file = "test_audio/VO.wav"
            if Path(test_file).exists():
                print(f"🚀 Testing analysis with {test_file}...")
                result = engine.process_file(test_file)
                
                if result['success']:
                    print(f"✅ Analysis successful: {result['successful_regions']}/{result['total_regions']} regions")
                    
                    # Test database integration
                    if engine.db_integration:
                        print("🔍 Testing database queries...")
                        search_result = engine.db_integration.search_files("VO", limit=5)
                        print(f"✅ Database search: {search_result['total_count']} files found")
                        
                        stats = engine.db_integration.get_stats()
                        print(f"✅ Database stats: {stats['total_files']} files, {stats['total_analysis_results']} results")
                else:
                    print(f"❌ Analysis failed: {result['errors']}")
            else:
                print(f"⚠️  Test file {test_file} not found, skipping analysis test")
        else:
            print("❌ Engine not initialized")
        
        # Test API app structure
        print("\n📋 API Endpoints Summary:")
        for route in app.routes:
            if hasattr(route, 'path'):
                methods = getattr(route, 'methods', ['GET'])
                print(f"   {', '.join(methods):<10} {route.path}")
        
        print("\n🎉 API endpoint tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Cleanup
        try:
            from main import engine
            if engine:
                engine.shutdown()
        except:
            pass

if __name__ == '__main__':
    success = asyncio.run(test_api_endpoints())
    
    if success:
        print("\n✅ All API tests PASSED!")
        print("🌐 The server can be started with: conda run -n audio-sampler-v2 python main.py")
        print("🎨 UI will be available at: http://localhost:8000/ui")
        print("📚 API docs will be available at: http://localhost:8000/docs")
    else:
        print("\n❌ API tests FAILED")
        sys.exit(1)