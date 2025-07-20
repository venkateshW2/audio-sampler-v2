#!/usr/bin/env python3
"""
Debug Engine Output Structure

Check what the engine actually returns to fix database integration.
"""

import sys
import json
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from core.engine import AudioSamplerEngine
from plugins.core_plugins.classifier import ClassifierPlugin
from plugins.core_plugins.key_finder import KeyFinderPlugin

def debug_engine_output():
    """Debug the actual engine output structure."""
    print("=== Debug Engine Output Structure ===")
    
    # Create engine without database to see raw output
    engine = AudioSamplerEngine()
    engine.register_plugin(ClassifierPlugin())
    engine.register_plugin(KeyFinderPlugin())
    
    # Process file
    result = engine.process_file("test_audio/VO.wav")
    
    # Print structure
    print("\n📊 Engine Result Structure:")
    print(json.dumps(result, indent=2, default=str))
    
    engine.shutdown()

if __name__ == '__main__':
    debug_engine_output()