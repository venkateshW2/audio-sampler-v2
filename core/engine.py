"""
Audio Intelligence Sampler v2 - Core Engine

This is the SACRED CORE ENGINE - NEVER MODIFY once stable.
All audio processing flows through this orchestrator.

Architecture principles:
- Never throws exceptions 
- Always returns results (partial if needed)
- Manages plugin lifecycle safely
- Handles all file I/O with fallbacks
- Comprehensive logging at every step
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class AudioSamplerEngine:
    """
    The core engine that orchestrates all audio processing.
    
    This class is SACRED - it never crashes and always returns results.
    All analysis features are handled by plugins that can fail safely.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the audio sampler engine.
        
        Args:
            config: Optional configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.plugins = []
        self.resource_manager = None
        self.file_handler = None
        self.start_time = datetime.now()
        
        self.logger.info("AudioSamplerEngine initializing...")
        
        try:
            self._initialize_components()
            self.logger.info("AudioSamplerEngine initialized successfully")
        except Exception as e:
            self.logger.error(f"Engine initialization error: {e}")
            # Engine continues to work even with initialization issues
    
    def _initialize_components(self):
        """Initialize core components with error handling."""
        try:
            # Import here to avoid circular imports
            from .resource_manager import ResourceManager
            from .file_handler import FileHandler
            
            self.resource_manager = ResourceManager(self.config.get('resources', {}))
            self.file_handler = FileHandler(self.config.get('file_handling', {}))
            
            self.logger.info("Core components initialized")
        except ImportError as e:
            self.logger.warning(f"Component not yet implemented: {e}")
        except Exception as e:
            self.logger.error(f"Component initialization error: {e}")
    
    def register_plugin(self, plugin):
        """Register a plugin with the engine.
        
        Args:
            plugin: Plugin instance following BasePlugin interface
        """
        try:
            plugin_name = getattr(plugin, 'get_name', lambda: 'Unknown')()
            self.logger.info(f"Registering plugin: {plugin_name}")
            self.plugins.append(plugin)
        except Exception as e:
            self.logger.error(f"Plugin registration failed: {e}")
            # Engine continues without this plugin
    
    def process_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a single audio file through the complete pipeline.
        
        This method NEVER raises exceptions. It always returns a result,
        even if partial or indicating failure.
        
        Args:
            file_path: Path to audio file to process
            
        Returns:
            Dict containing processing results, metadata, and status
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Initialize result structure
        result = {
            'file_path': str(file_path),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'duration_seconds': 0,
            'metadata': {},
            'analysis': {},
            'errors': [],
            'warnings': []
        }
        
        self.logger.info(f"Processing file: {file_path}")
        
        try:
            # Phase 1: File validation and loading
            file_result = self._load_file(file_path)
            if not file_result['success']:
                result['errors'].extend(file_result['errors'])
                result['duration_seconds'] = time.time() - start_time
                return result
            
            result['metadata'] = file_result['metadata']
            audio_data = file_result['audio_data']
            
            # Phase 2: Core analysis pipeline
            analysis_result = self._run_analysis_pipeline(audio_data, result['metadata'])
            result['analysis'] = analysis_result['results']
            result['errors'].extend(analysis_result['errors'])
            result['warnings'].extend(analysis_result['warnings'])
            
            # Phase 3: Success if we got here
            result['success'] = True
            self.logger.info(f"File processed successfully: {file_path}")
            
        except Exception as e:
            # This should never happen, but if it does, we handle it gracefully
            error_msg = f"Unexpected engine error: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            result['errors'].append(error_msg)
        
        finally:
            result['duration_seconds'] = time.time() - start_time
            self.logger.info(f"Processing completed in {result['duration_seconds']:.2f}s")
        
        return result
    
    def _load_file(self, file_path: Path) -> Dict[str, Any]:
        """Load audio file with comprehensive error handling.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict with success status, audio data, metadata, and errors
        """
        result = {
            'success': False,
            'audio_data': None,
            'metadata': {},
            'errors': []
        }
        
        try:
            # Basic file validation
            if not file_path.exists():
                result['errors'].append(f"File not found: {file_path}")
                return result
            
            if not file_path.is_file():
                result['errors'].append(f"Path is not a file: {file_path}")
                return result
            
            # Get basic file metadata
            result['metadata'] = {
                'filename': file_path.name,
                'size_bytes': file_path.stat().st_size,
                'extension': file_path.suffix.lower(),
                'modified_time': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
            }
            
            # Use file handler if available
            if self.file_handler:
                load_result = self.file_handler.load_audio(file_path)
                if load_result['success']:
                    result['audio_data'] = load_result['audio_data']
                    result['metadata'].update(load_result['metadata'])
                    result['success'] = True
                else:
                    result['errors'].extend(load_result['errors'])
            else:
                # Fallback: basic librosa loading
                result = self._fallback_audio_loading(file_path, result)
            
        except Exception as e:
            error_msg = f"File loading error: {e}"
            self.logger.error(error_msg)
            result['errors'].append(error_msg)
        
        return result
    
    def _fallback_audio_loading(self, file_path: Path, result: Dict) -> Dict:
        """Fallback audio loading using librosa directly."""
        try:
            import librosa
            import numpy as np
            
            self.logger.info("Using fallback librosa loading")
            
            # Load with librosa (safe defaults)
            audio_data, sample_rate = librosa.load(
                str(file_path), 
                sr=22050,  # Standard sample rate for analysis
                mono=True,  # Convert to mono
                duration=None  # Load full file
            )
            
            # Update metadata
            result['metadata'].update({
                'sample_rate': sample_rate,
                'duration_seconds': len(audio_data) / sample_rate,
                'channels': 1,  # mono
                'loader': 'librosa_fallback'
            })
            
            result['audio_data'] = audio_data
            result['success'] = True
            
        except Exception as e:
            error_msg = f"Fallback loading failed: {e}"
            self.logger.error(error_msg)
            result['errors'].append(error_msg)
        
        return result
    
    def _run_analysis_pipeline(self, audio_data, metadata: Dict) -> Dict[str, Any]:
        """Run all registered plugins on the audio data.
        
        Args:
            audio_data: Loaded audio data
            metadata: File metadata
            
        Returns:
            Dict with analysis results, errors, and warnings
        """
        pipeline_result = {
            'results': {},
            'errors': [],
            'warnings': []
        }
        
        if not self.plugins:
            pipeline_result['warnings'].append("No plugins registered")
            return pipeline_result
        
        self.logger.info(f"Running analysis pipeline with {len(self.plugins)} plugins")
        
        for plugin in self.plugins:
            try:
                plugin_name = getattr(plugin, 'get_name', lambda: 'Unknown')()
                self.logger.info(f"Running plugin: {plugin_name}")
                
                # Check if plugin can process this file
                can_process = getattr(plugin, 'can_process', lambda x: True)(metadata)
                if not can_process:
                    self.logger.info(f"Plugin {plugin_name} skipped (incompatible file)")
                    continue
                
                # Run plugin analysis
                plugin_start = time.time()
                plugin_result = plugin.process(audio_data, metadata)
                plugin_duration = time.time() - plugin_start
                
                # Store results
                pipeline_result['results'][plugin_name] = {
                    'data': plugin_result,
                    'processing_time': plugin_duration,
                    'timestamp': datetime.now().isoformat()
                }
                
                self.logger.info(f"Plugin {plugin_name} completed in {plugin_duration:.2f}s")
                
            except Exception as e:
                error_msg = f"Plugin {getattr(plugin, 'get_name', lambda: 'Unknown')()} failed: {e}"
                self.logger.error(error_msg)
                pipeline_result['errors'].append(error_msg)
                # Continue with other plugins
        
        return pipeline_result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current engine status and health information."""
        uptime = datetime.now() - self.start_time
        
        status = {
            'engine_version': '2.0.0-phase1',
            'status': 'healthy',
            'uptime_seconds': uptime.total_seconds(),
            'plugins_registered': len(self.plugins),
            'plugin_names': [getattr(p, 'get_name', lambda: 'Unknown')() for p in self.plugins],
            'components': {
                'resource_manager': self.resource_manager is not None,
                'file_handler': self.file_handler is not None
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    def shutdown(self):
        """Gracefully shutdown the engine and cleanup resources."""
        self.logger.info("AudioSamplerEngine shutting down...")
        
        try:
            if self.resource_manager:
                self.resource_manager.cleanup()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        self.logger.info("AudioSamplerEngine shutdown complete")