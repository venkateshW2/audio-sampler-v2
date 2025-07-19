#!/usr/bin/env python3
"""
Quick test of core Audio Intelligence Sampler v2 components.

Run this to verify the core system is working before adding plugins.
"""

import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import AudioSamplerEngine
from core.resource_manager import ResourceManager  
from core.file_handler import FileHandler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_resource_manager():
    """Test ResourceManager initialization and basic functionality."""
    logger.info("🔧 Testing ResourceManager...")
    
    try:
        rm = ResourceManager()
        status = rm.get_resource_status()
        
        logger.info(f"✅ ResourceManager OK - GPU: {status['gpu']['available']}")
        logger.info(f"   Memory: {status['memory']['memory_percent']:.1f}% used")
        logger.info(f"   CPU cores: {status['cpu']['cores']}")
        
        rm.cleanup()
        return True
        
    except Exception as e:
        logger.error(f"❌ ResourceManager failed: {e}")
        return False


def test_file_handler():
    """Test FileHandler initialization."""
    logger.info("📁 Testing FileHandler...")
    
    try:
        fh = FileHandler()
        logger.info("✅ FileHandler OK - pedalboard primary, librosa fallback")
        return True
        
    except Exception as e:
        logger.error(f"❌ FileHandler failed: {e}")
        return False


def test_engine():
    """Test AudioSamplerEngine initialization and basic functionality."""
    logger.info("🎵 Testing AudioSamplerEngine...")
    
    try:
        engine = AudioSamplerEngine()
        status = engine.get_status()
        
        logger.info(f"✅ Engine OK - Status: {status['status']}")
        logger.info(f"   Version: {status['engine_version']}")
        logger.info(f"   Components ready: {status['components']}")
        
        engine.shutdown()
        return True
        
    except Exception as e:
        logger.error(f"❌ Engine failed: {e}")
        return False


def test_fastapi_import():
    """Test that FastAPI components can be imported."""
    logger.info("🌐 Testing FastAPI imports...")
    
    try:
        import main
        import api.server
        logger.info("✅ FastAPI components OK")
        return True
        
    except Exception as e:
        logger.error(f"❌ FastAPI import failed: {e}")
        return False


def main():
    """Run all core tests."""
    logger.info("🚀 Audio Intelligence Sampler v2 - Core System Test")
    logger.info("=" * 60)
    
    tests = [
        ("Resource Manager", test_resource_manager),
        ("File Handler", test_file_handler), 
        ("Audio Engine", test_engine),
        ("FastAPI Components", test_fastapi_import)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"Test failed: {test_name}")
        except Exception as e:
            logger.error(f"Test error: {test_name} - {e}")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 ALL CORE TESTS PASSED!")
        logger.info("✅ Ready to start development server with: python main.py")
        logger.info("🌐 Access UI at: http://dell-g7-ip:8000")
        return True
    else:
        logger.error("❌ Some tests failed - check logs above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)