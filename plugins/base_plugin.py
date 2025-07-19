"""
Audio Intelligence Sampler v2 - Base Plugin Interface

Abstract base class that ALL plugins must inherit from.
This interface is SACRED - never change without updating all plugins.

Architecture principles:
- Plugins declare their resource requirements
- Plugins can fail without crashing core
- Standard input/output format for all plugins
- Version tracking for compatibility
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging


class BasePlugin(ABC):
    """
    Abstract base class for all audio analysis plugins.
    
    This interface ensures consistent behavior across all plugins
    and enables the core engine to manage them safely.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize plugin with optional configuration.
        
        Args:
            config: Optional plugin-specific configuration
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        
        # Initialize plugin
        try:
            self._initialize()
            self._initialized = True
            self.logger.info(f"Plugin {self.get_name()} initialized successfully")
        except Exception as e:
            self.logger.error(f"Plugin {self.get_name()} initialization failed: {e}")
            # Plugin continues but marks itself as not initialized
    
    @abstractmethod
    def get_name(self) -> str:
        """Get plugin name.
        
        Returns:
            Unique plugin name (used for identification)
        """
        pass
    
    @abstractmethod
    def get_version(self) -> str:
        """Get plugin version.
        
        Returns:
            Plugin version string (semantic versioning recommended)
        """
        pass
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """Get plugin resource requirements.
        
        Returns:
            Dict with resource requirements:
            {
                'cpu': bool,           # Requires CPU processing
                'gpu': bool,           # Requires GPU processing  
                'memory_mb': int,      # Estimated memory usage in MB
                'gpu_memory_mb': int,  # Estimated GPU memory in MB
                'libraries': list      # Required libraries
            }
        """
        pass
    
    @abstractmethod
    def process(self, audio_data, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio data and return analysis results.
        
        This is the main analysis method. It should:
        - Never raise exceptions (catch all internally)
        - Return meaningful results even on partial failure
        - Log errors/warnings appropriately
        
        Args:
            audio_data: Audio data (numpy array, typically mono 22kHz)
            metadata: File metadata dict
            
        Returns:
            Dict with analysis results:
            {
                'success': bool,           # Whether analysis succeeded
                'confidence': float,       # Confidence in results (0-1)
                'results': dict,          # Main analysis results
                'metadata': dict,         # Additional metadata
                'errors': list,           # List of errors encountered
                'warnings': list          # List of warnings
            }
        """
        pass
    
    @abstractmethod
    def can_process(self, file_metadata: Dict[str, Any]) -> bool:
        """Check if plugin can process the given file.
        
        Args:
            file_metadata: File metadata dict
            
        Returns:
            True if plugin can process this file type/format
        """
        pass
    
    def _initialize(self):
        """Initialize plugin-specific resources.
        
        Override this method to perform plugin initialization.
        Exceptions raised here will be caught and logged.
        """
        pass
    
    def is_initialized(self) -> bool:
        """Check if plugin initialized successfully.
        
        Returns:
            True if plugin is ready to process files
        """
        return self._initialized
    
    def get_info(self) -> Dict[str, Any]:
        """Get comprehensive plugin information.
        
        Returns:
            Dict with plugin details
        """
        return {
            'name': self.get_name(),
            'version': self.get_version(),
            'requirements': self.get_requirements(),
            'initialized': self.is_initialized(),
            'config': self.config
        }
    
    def validate_audio_data(self, audio_data, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input audio data and metadata.
        
        Args:
            audio_data: Audio data to validate
            metadata: Metadata to validate
            
        Returns:
            Dict with validation results:
            {
                'valid': bool,
                'errors': list,
                'warnings': list
            }
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check audio data
            if audio_data is None:
                validation['errors'].append("Audio data is None")
                validation['valid'] = False
                return validation
            
            if len(audio_data) == 0:
                validation['errors'].append("Audio data is empty")
                validation['valid'] = False
                return validation
            
            # Check for NaN or infinite values
            import numpy as np
            if np.any(np.isnan(audio_data)):
                validation['warnings'].append("Audio data contains NaN values")
            
            if np.any(np.isinf(audio_data)):
                validation['warnings'].append("Audio data contains infinite values")
            
            # Check metadata
            if not isinstance(metadata, dict):
                validation['errors'].append("Metadata must be a dictionary")
                validation['valid'] = False
            
            # Check for required metadata fields
            required_fields = ['final_sample_rate', 'final_duration']
            for field in required_fields:
                if field not in metadata:
                    validation['warnings'].append(f"Missing metadata field: {field}")
            
        except Exception as e:
            validation['errors'].append(f"Validation error: {e}")
            validation['valid'] = False
        
        return validation
    
    def safe_process(self, audio_data, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Safely process audio with comprehensive error handling.
        
        This method wraps the process() method with additional safety checks
        and ensures a valid result is always returned.
        
        Args:
            audio_data: Audio data to process
            metadata: File metadata
            
        Returns:
            Dict with processing results (guaranteed to be valid)
        """
        # Initialize safe result structure
        result = {
            'success': False,
            'confidence': 0.0,
            'results': {},
            'metadata': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Check if plugin is initialized
            if not self.is_initialized():
                result['errors'].append(f"Plugin {self.get_name()} not initialized")
                return result
            
            # Validate inputs
            validation = self.validate_audio_data(audio_data, metadata)
            if not validation['valid']:
                result['errors'].extend(validation['errors'])
                result['warnings'].extend(validation['warnings'])
                return result
            
            result['warnings'].extend(validation['warnings'])
            
            # Call the actual processing method
            plugin_result = self.process(audio_data, metadata)
            
            # Validate plugin result
            if isinstance(plugin_result, dict):
                result.update(plugin_result)
            else:
                result['errors'].append("Plugin returned invalid result format")
            
        except Exception as e:
            error_msg = f"Plugin {self.get_name()} processing error: {e}"
            self.logger.error(error_msg)
            result['errors'].append(error_msg)
        
        return result


class PluginRegistry:
    """Registry for managing available plugins."""
    
    def __init__(self):
        self.plugins = {}
        self.logger = logging.getLogger(__name__)
    
    def register(self, plugin: BasePlugin):
        """Register a plugin instance.
        
        Args:
            plugin: Plugin instance to register
        """
        try:
            name = plugin.get_name()
            if name in self.plugins:
                self.logger.warning(f"Plugin {name} already registered, overwriting")
            
            self.plugins[name] = plugin
            self.logger.info(f"Registered plugin: {name}")
            
        except Exception as e:
            self.logger.error(f"Plugin registration failed: {e}")
    
    def get_plugin(self, name: str) -> Optional[BasePlugin]:
        """Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(name)
    
    def get_all_plugins(self) -> Dict[str, BasePlugin]:
        """Get all registered plugins.
        
        Returns:
            Dict of plugin name -> plugin instance
        """
        return self.plugins.copy()
    
    def get_plugins_for_file(self, file_metadata: Dict[str, Any]) -> Dict[str, BasePlugin]:
        """Get plugins that can process a specific file.
        
        Args:
            file_metadata: File metadata dict
            
        Returns:
            Dict of plugin name -> plugin instance for compatible plugins
        """
        compatible = {}
        
        for name, plugin in self.plugins.items():
            try:
                if plugin.is_initialized() and plugin.can_process(file_metadata):
                    compatible[name] = plugin
            except Exception as e:
                self.logger.error(f"Error checking plugin {name} compatibility: {e}")
        
        return compatible