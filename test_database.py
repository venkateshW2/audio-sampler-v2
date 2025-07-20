#!/usr/bin/env python3
"""
Test Database Implementation

Test the SQLite database with region-based schema.
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from database import Database, DatabaseIntegration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_database_creation():
    """Test database creation and basic operations."""
    print("=== Testing Database Creation ===")
    
    # Remove test database if exists
    test_db_path = "test_audio_sampler.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    try:
        # Create database
        db = Database(test_db_path)
        print("✅ Database created successfully")
        
        # Test adding a file
        file_record = db.add_file(
            file_path="test_audio/VO.wav",
            size_bytes=1234567,
            duration_seconds=27.3,
            sample_rate=48000,
            channels=2,
            format="wav"
        )
        print(f"✅ File added: ID={file_record.id}")
        
        # Test adding regions
        regions_data = [
            {
                'region_id': 'VO-r1',
                'region_number': 1,
                'start_time': 0.8,
                'end_time': 28.2,
                'duration': 27.4,
                'region_type': 'content'
            }
        ]
        
        regions = db.add_regions(file_record.id, regions_data)
        print(f"✅ Regions added: {len(regions)} regions")
        
        # Get region ID from dictionary
        region_id = regions[0]['id']
        
        # Test adding analysis results
        analysis_result = db.add_analysis_result(
            region_id=region_id,
            file_id=file_record.id,
            plugin_name='classifier',
            plugin_version='2.0.0-passt-gtx1060',
            results={
                'success': True,
                'top_prediction': 'Speech synthesizer',
                'top_confidence': 0.467,
                'music_probability': 0.007,
                'speech_probability': 0.913
            },
            processing_time_ms=2770,
            success=True
        )
        print(f"✅ Analysis result added: ID={analysis_result.id}")
        
        # Test retrieval
        file_with_regions = db.get_file_with_regions(file_record.id)
        print(f"✅ Retrieved file with {len(file_with_regions.regions)} regions")
        
        # Test stats
        stats = db.get_database_stats()
        print(f"✅ Database stats: {stats}")
        
        # Clean up
        db.close()
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_database_integration():
    """Test database integration with engine results."""
    print("\n=== Testing Database Integration ===")
    
    # Remove test database if exists
    test_db_path = "test_integration.db"
    if os.path.exists(test_db_path):
        os.remove(test_db_path)
    
    try:
        # Create integration
        db_integration = DatabaseIntegration(test_db_path)
        print("✅ Database integration created")
        
        # Simulate engine results
        file_metadata = {
            'size_bytes': 1234567,
            'duration_seconds': 27.3,
            'sample_rate': 48000,
            'channels': 2,
            'format': 'wav'
        }
        
        regions_data = [
            {
                'region_id': 'VO-r1',
                'region_number': 1,
                'start_time': 0.8,
                'end_time': 28.2,
                'duration': 27.4,
                'region_type': 'content'
            }
        ]
        
        analysis_results = {
            'VO-r1': {
                'classifier': {
                    'success': True,
                    'top_prediction': 'Speech synthesizer',
                    'top_confidence': 0.467,
                    'music_probability': 0.007,
                    'speech_probability': 0.913,
                    'processing_time_ms': 2770,
                    'analysis_metadata': {'version': '2.0.0'}
                },
                'tempo_meter_detector': {
                    'success': True,
                    'tempo_bpm': 120.0,
                    'tempo_confidence': 0.8,
                    'meter': {'time_signature': '3/4', 'confidence': 0.8},
                    'processing_time_ms': 2840,
                    'analysis_metadata': {'version': '2.0.0'}
                },
                'key_finder': {
                    'success': True,
                    'key': 'C',
                    'mode': 'atonal',
                    'confidence': 0.231,
                    'processing_time_ms': 70,
                    'analysis_metadata': {'version': '6.0.0', 'method': 'LibROSA-chroma'}
                }
            }
        }
        
        # Store results
        file_id = db_integration.store_file_analysis(
            "test_audio/VO.wav",
            file_metadata,
            regions_data,
            analysis_results
        )
        print(f"✅ Stored complete analysis: File ID={file_id}")
        
        # Retrieve results
        stored_results = db_integration.get_file_results("test_audio/VO.wav")
        if stored_results:
            print(f"✅ Retrieved results: {len(stored_results['regions'])} regions")
            print(f"   Analysis summary: {stored_results['analysis_summary']['plugins_used']}")
        else:
            print("❌ Failed to retrieve results")
        
        # Test search
        search_results = db_integration.search_files(query="VO", limit=10)
        print(f"✅ Search results: {search_results['total_count']} files found")
        
        # Test stats
        stats = db_integration.get_stats()
        print(f"✅ Integration stats: {stats}")
        
        # Clean up
        db_integration.close()
        if os.path.exists(test_db_path):
            os.remove(test_db_path)
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == '__main__':
    print("Testing SQLite Database with Region-Based Schema")
    
    # Test basic database
    db_success = test_database_creation()
    
    # Test integration
    integration_success = test_database_integration()
    
    if db_success and integration_success:
        print("\n🎉 All database tests PASSED!")
        print("✅ SQLite database with region-based schema working")
    else:
        print("\n❌ Some database tests FAILED")
        sys.exit(1)