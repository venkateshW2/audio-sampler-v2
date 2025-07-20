#!/usr/bin/env python3
"""
Test Engine Database Integration

Test that the core engine automatically stores analysis results in the database.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import AudioSamplerEngine
from plugins.core_plugins.classifier import ClassifierPlugin
from plugins.core_plugins.tempo_meter_detector import TempoMeterDetectorPlugin
from plugins.core_plugins.key_finder import KeyFinderPlugin
from database import DatabaseIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_engine_database_integration():
    """Test that engine stores results in database automatically."""
    print("=== Testing Engine Database Integration ===")
    
    # Remove test database if exists
    test_db_path = "test_engine_db.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    try:
        # Create engine with database configuration
        config = {
            'database': {
                'path': test_db_path
            }
        }
        
        engine = AudioSamplerEngine(config)
        print("✅ Engine created with database integration")
        
        # Register plugins
        engine.register_plugin(ClassifierPlugin())
        engine.register_plugin(TempoMeterDetectorPlugin()) 
        engine.register_plugin(KeyFinderPlugin())
        print("✅ Plugins registered")
        
        # Check engine status
        status = engine.get_status()
        if status['components']['database_integration']:
            print("✅ Database integration enabled")
        else:
            print("❌ Database integration not enabled")
            return False
        
        # Process a test file
        test_file = "test_audio/VO.wav"
        if not Path(test_file).exists():
            print(f"❌ Test file not found: {test_file}")
            return False
        
        print(f"🚀 Processing {test_file} with database storage...")
        result = engine.process_file(test_file)
        
        if result['success']:
            print(f"✅ File processed successfully: {result['successful_regions']}/{result['total_regions']} regions")
        else:
            print(f"❌ File processing failed: {result['errors']}")
            return False
        
        # Verify data was stored in database
        db_integration = DatabaseIntegration(test_db_path)
        stored_results = db_integration.get_file_results(test_file)
        
        if stored_results:
            print(f"✅ Results stored in database: {len(stored_results['regions'])} regions")
            print(f"   Plugins used: {stored_results['analysis_summary']['plugins_used']}")
            
            # Check specific analysis results
            for region in stored_results['regions']:
                plugin_count = len(region['analysis_results'])
                print(f"   Region {region['region_id']}: {plugin_count} plugin results")
        else:
            print("❌ No results found in database")
            return False
        
        # Test search functionality
        search_results = db_integration.search_files(query="VO")
        print(f"✅ Search found {search_results['total_count']} files")
        
        # Test stats
        stats = db_integration.get_stats()
        print(f"✅ Database stats: {stats['total_files']} files, {stats['total_regions']} regions, {stats['total_analysis_results']} results")
        
        # Clean up
        engine.shutdown()
        db_integration.close()
        
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Engine database integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing Engine Database Integration")
    
    success = test_engine_database_integration()
    
    if success:
        print("\n🎉 Engine database integration test PASSED!")
        print("✅ Core engine automatically stores analysis results")
        print("✅ Database integration working end-to-end")
    else:
        print("\n❌ Engine database integration test FAILED")
        sys.exit(1)