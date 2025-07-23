"""
Audio Intelligence Sampler v2 - Core Engine

This is the SACRED CORE ENGINE with REGION-BASED PROCESSING.
All audio processing flows through this orchestrator.

Architecture principles:
- Never throws exceptions 
- Always returns results (partial if needed)
- Manages plugin lifecycle safely
- Handles all file I/O with fallbacks + region detection
- Processes each region independently through all plugins
- Comprehensive logging at every step
"""

import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

# Import database integration
try:
    from database import DatabaseIntegration
except ImportError:
    DatabaseIntegration = None

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
        self.db_integration = None
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
            
            # Initialize database integration if available
            if DatabaseIntegration:
                db_path = self.config.get('database', {}).get('path', 'audio_sampler.db')
                self.db_integration = DatabaseIntegration(db_path)
                self.logger.info("Database integration initialized")
            else:
                self.logger.warning("Database integration not available")
            
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
        Process a single audio file through region-based pipeline.
        
        This method NEVER raises exceptions. It always returns a result,
        even if partial or indicating failure.
        
        Args:
            file_path: Path to audio file to process
            
        Returns:
            Dict containing file metadata, regions, and region analysis results
        """
        start_time = time.time()
        file_path = Path(file_path)
        
        # Initialize result structure (region-based)
        result = {
            'file_path': str(file_path),
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'duration_seconds': 0,
            'file_metadata': {},
            'regions': [],
            'region_analyses': [],
            'total_regions': 0,
            'successful_regions': 0,
            'failed_regions': 0,
            'errors': [],
            'warnings': []
        }
        
        self.logger.info(f"Processing file with region detection: {file_path}")
        
        try:
            # Phase 1: File loading with region detection
            region_result = self._load_file_with_regions(file_path)
            if not region_result['success']:
                result['errors'].extend(region_result['errors'])
                result['duration_seconds'] = time.time() - start_time
                return result
            
            result['file_metadata'] = region_result['file_metadata']
            result['regions'] = region_result['regions']
            result['total_regions'] = region_result['total_regions']
            
            # Phase 2: Process each region independently through all plugins
            region_analyses = self._process_regions(region_result['regions'])
            result['region_analyses'] = region_analyses['results']
            result['successful_regions'] = region_analyses['successful_count']
            result['failed_regions'] = region_analyses['failed_count']
            result['errors'].extend(region_analyses['errors'])
            result['warnings'].extend(region_analyses['warnings'])
            
            # Phase 3: Success if we processed at least one region
            result['success'] = result['successful_regions'] > 0
            
            # Phase 4: Store results in database if integration is available
            if self.db_integration and result['success']:
                try:
                    self._store_results_in_database(file_path, result)
                except Exception as e:
                    error_msg = f"Database storage failed: {e}"
                    self.logger.warning(error_msg)
                    result['warnings'].append(error_msg)
            
            if result['success']:
                self.logger.info(f"✅ File processed: {result['successful_regions']}/{result['total_regions']} regions successful")
            else:
                self.logger.warning(f"❌ File processing failed: 0/{result['total_regions']} regions successful")
            
        except Exception as e:
            # This should never happen, but if it does, we handle it gracefully
            error_msg = f"Unexpected engine error: {e}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            result['errors'].append(error_msg)
        
        finally:
            result['duration_seconds'] = time.time() - start_time
            self.logger.info(f"Region-based processing completed in {result['duration_seconds']:.2f}s")
        
        return result
    
    def _load_file_with_regions(self, file_path: Path) -> Dict[str, Any]:
        """Load audio file and extract regions with comprehensive error handling.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dict with success status, file metadata, regions, and errors
        """
        result = {
            'success': False,
            'file_metadata': {},
            'regions': [],
            'total_regions': 0,
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
            
            # Load audio with region detection using FileHandler
            if not self.file_handler:
                result['errors'].append("FileHandler not initialized")
                return result
            
            # Use the new region-based loading
            region_result = self.file_handler.load_audio_with_regions(file_path)
            
            if not region_result['success']:
                result['errors'].extend(region_result['errors'])
                return result
            
            # Extract regions and metadata
            result['file_metadata'] = region_result['file_metadata']
            result['regions'] = region_result['regions']
            result['total_regions'] = region_result['total_regions']
            result['success'] = True
            
            self.logger.info(f"✅ Loaded {result['total_regions']} regions from {file_path.name}")
            
        except Exception as e:
            error_msg = f"Region loading error: {e}"
            self.logger.error(error_msg)
            result['errors'].append(error_msg)
        
        return result
    
    def _process_regions(self, regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process each region independently through all registered plugins.
        
        Args:
            regions: List of region dictionaries with audio data and metadata
            
        Returns:
            Dict with region analysis results, counts, errors, and warnings
        """
        process_result = {
            'results': [],
            'successful_count': 0,
            'failed_count': 0,
            'errors': [],
            'warnings': []
        }
        
        if not self.plugins:
            process_result['warnings'].append("No plugins registered")
            return process_result
        
        if not regions:
            process_result['warnings'].append("No regions to process")
            return process_result
        
        self.logger.info(f"Processing {len(regions)} regions through {len(self.plugins)} plugins")
        
        # Process each region independently
        for region_idx, region in enumerate(regions):
            region_id = region.get('region_id', f'region-{region_idx+1}')
            
            try:
                self.logger.info(f"Processing region {region_id} ({region['duration']:.1f}s)")
                
                # Process this region through all plugins
                region_analysis = self._process_single_region(region)
                
                # Add region analysis to results
                process_result['results'].append(region_analysis)
                
                if region_analysis['success']:
                    process_result['successful_count'] += 1
                    self.logger.info(f"✅ Region {region_id} processed successfully")
                else:
                    process_result['failed_count'] += 1
                    self.logger.warning(f"⚠️ Region {region_id} processing failed")
                    process_result['errors'].extend(region_analysis['errors'])
                
            except Exception as e:
                error_msg = f"Region {region_id} processing error: {e}"
                self.logger.error(error_msg)
                process_result['errors'].append(error_msg)
                process_result['failed_count'] += 1
        
        total_regions = len(regions)
        self.logger.info(f"Region processing complete: {process_result['successful_count']}/{total_regions} successful")
        
        return process_result
    
    def _process_single_region(self, region: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single region through all plugins."""
        region_id = region.get('region_id', 'unknown')
        audio_data = region.get('audio_data')
        sample_rate = region.get('sample_rate', 22050)
        
        # Initialize region analysis result
        region_analysis = {
            'region_id': region_id,
            'region_metadata': {
                'start_time': region.get('start_time', 0.0),
                'end_time': region.get('end_time', 0.0),
                'duration': region.get('duration', 0.0),
                'region_type': region.get('region_type', 'content'),
                'parent_file': region.get('parent_file', 'unknown')
            },
            'plugin_results': {},
            'success': False,
            'successful_plugins': 0,
            'failed_plugins': 0,
            'errors': [],
            'warnings': []
        }
        
        if audio_data is None or len(audio_data) == 0:
            region_analysis['errors'].append("No audio data in region")
            return region_analysis
        
        # Get resource status for plugin capability checks
        resource_status = self.resource_manager.get_resource_status() if self.resource_manager else {}
        
        # Step 1: Check if FeatureExtractor is available and run it first
        feature_cache_id = None
        feature_extractor = None
        
        for plugin in self.plugins:
            if getattr(plugin, 'get_name', lambda: '')() == 'feature_extractor':
                feature_extractor = plugin
                break
        
        if feature_extractor:
            try:
                self.logger.info(f"Running FeatureExtractor for region {region_id}")
                
                region_info = {
                    'start_time': region.get('start_time', 0.0),
                    'end_time': region.get('end_time', 0.0),
                    'region_id': region_id
                }
                
                fe_start = time.time()
                fe_result = feature_extractor.analyze(audio_data, sample_rate, region_info)
                fe_duration = time.time() - fe_start
                
                # Store FeatureExtractor results
                region_analysis['plugin_results']['feature_extractor'] = {
                    'data': fe_result,
                    'processing_time': fe_duration,
                    'timestamp': datetime.now().isoformat(),
                    'success': fe_result.get('success', False)
                }
                
                if fe_result.get('success', False):
                    region_analysis['successful_plugins'] += 1
                    feature_cache_id = fe_result.get('feature_cache_id')
                    self.logger.info(f"✅ FeatureExtractor completed with cache ID: {feature_cache_id}")
                else:
                    region_analysis['failed_plugins'] += 1
                    if 'error' in fe_result:
                        region_analysis['errors'].append(f"FeatureExtractor: {fe_result['error']}")
                
            except Exception as e:
                error_msg = f"FeatureExtractor error: {e}"
                self.logger.error(error_msg)
                region_analysis['errors'].append(error_msg)
                region_analysis['failed_plugins'] += 1
        
        # Step 2: Run ContentAnalysisPlugin if available and FeatureExtractor succeeded
        content_analysis_plugin = None
        for plugin in self.plugins:
            if getattr(plugin, 'get_name', lambda: '')() == 'content_analysis':
                content_analysis_plugin = plugin
                break
        
        if content_analysis_plugin and feature_cache_id:
            try:
                self.logger.info(f"Running ContentAnalysisPlugin for region {region_id}")
                
                ca_start = time.time()
                ca_result = content_analysis_plugin.process(
                    audio_data=audio_data,
                    sample_rate=sample_rate,
                    feature_cache_id=feature_cache_id,
                    region_id=region_id
                )
                ca_duration = time.time() - ca_start
                
                # Store ContentAnalysisPlugin results
                region_analysis['plugin_results']['content_analysis'] = {
                    'data': ca_result,
                    'processing_time': ca_duration,
                    'timestamp': datetime.now().isoformat(),
                    'success': ca_result.get('success', False)
                }
                
                if ca_result.get('success', False):
                    region_analysis['successful_plugins'] += 1
                    
                    # Store content analysis results in feature cache for ClassifierPlugin
                    if feature_cache_id:
                        self._store_content_analysis_results(feature_cache_id, ca_result)
                    
                    self.logger.info(f"✅ ContentAnalysisPlugin completed: {ca_result.get('total_segments', 0)} segments")
                else:
                    region_analysis['failed_plugins'] += 1
                    if 'error' in ca_result:
                        region_analysis['errors'].append(f"ContentAnalysisPlugin: {ca_result['error']}")
                
            except Exception as e:
                error_msg = f"ContentAnalysisPlugin error: {e}"
                self.logger.error(error_msg)
                region_analysis['errors'].append(error_msg)
                region_analysis['failed_plugins'] += 1
        
        # Step 3: Process through remaining plugins (skip FeatureExtractor and ContentAnalysisPlugin)
        for plugin in self.plugins:
            try:
                plugin_name = getattr(plugin, 'get_name', lambda: 'Unknown')()
                
                # Skip FeatureExtractor and ContentAnalysisPlugin (already processed)
                if plugin_name in ['feature_extractor', 'content_analysis']:
                    continue
                
                # Check if plugin can process this region
                can_process, reason = getattr(plugin, 'can_process', lambda x, y, z: (True, "No check"))(
                    audio_data, sample_rate, resource_status
                )
                
                if not can_process:
                    region_analysis['warnings'].append(f"Plugin {plugin_name} skipped: {reason}")
                    continue
                
                # Run plugin analysis on region with shared features if available
                plugin_start = time.time()
                plugin_kwargs = {
                    'audio_data': audio_data,
                    'sample_rate': sample_rate,
                    'file_path': None,  # No file path for regions
                    'region_id': region_id,
                    'region_metadata': region_analysis['region_metadata']
                }
                
                # Add feature cache ID if available
                if feature_cache_id:
                    plugin_kwargs['feature_cache_id'] = feature_cache_id
                
                plugin_result = plugin.process(**plugin_kwargs)
                plugin_duration = time.time() - plugin_start
                
                # Store plugin results
                region_analysis['plugin_results'][plugin_name] = {
                    'data': plugin_result,
                    'processing_time': plugin_duration,
                    'timestamp': datetime.now().isoformat(),
                    'success': plugin_result.get('success', False)
                }
                
                if plugin_result.get('success', False):
                    region_analysis['successful_plugins'] += 1
                else:
                    region_analysis['failed_plugins'] += 1
                    if 'error' in plugin_result:
                        region_analysis['errors'].append(f"Plugin {plugin_name}: {plugin_result['error']}")
                
            except Exception as e:
                error_msg = f"Plugin {getattr(plugin, 'get_name', lambda: 'Unknown')()} error: {e}"
                region_analysis['errors'].append(error_msg)
                region_analysis['failed_plugins'] += 1
        
        # Region is successful if at least one plugin succeeded
        region_analysis['success'] = region_analysis['successful_plugins'] > 0
        
        return region_analysis
    
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
                'file_handler': self.file_handler is not None,
                'database_integration': self.db_integration is not None
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return status
    
    def _store_content_analysis_results(self, feature_cache_id: str, ca_result: Dict[str, Any]):
        """Store ContentAnalysisPlugin results in FeatureExtractor cache for ClassifierPlugin access."""
        try:
            # Import the global feature cache from FeatureExtractor
            from plugins.core_plugins.feature_extractor import _GLOBAL_FEATURE_CACHE
            
            if feature_cache_id in _GLOBAL_FEATURE_CACHE:
                # Add content analysis results to the cached features
                _GLOBAL_FEATURE_CACHE[feature_cache_id]['content_analysis_results'] = ca_result
                self.logger.debug(f"Stored ContentAnalysisPlugin results in cache {feature_cache_id}")
            else:
                self.logger.warning(f"Cache ID {feature_cache_id} not found for storing content analysis results")
                
        except Exception as e:
            self.logger.error(f"Error storing content analysis results: {e}")
    
    def _store_results_in_database(self, file_path: Path, result: Dict[str, Any]):
        """Store analysis results in database using DatabaseIntegration."""
        if not self.db_integration:
            return
        
        try:
            # Extract and map file metadata to database format
            engine_metadata = result.get('file_metadata', {})
            
            # Map engine metadata fields to database model fields
            file_metadata = {
                'size_bytes': engine_metadata.get('size_bytes'),
                'duration_seconds': engine_metadata.get('final_duration', engine_metadata.get('original_duration')),
                'sample_rate': engine_metadata.get('final_sample_rate', engine_metadata.get('original_sample_rate')),
                'channels': engine_metadata.get('original_channels'),
                'format': engine_metadata.get('extension', '').replace('.', '') if engine_metadata.get('extension') else None
            }
            
            # Remove None values
            file_metadata = {k: v for k, v in file_metadata.items() if v is not None}
            
            # Convert regions to database format
            regions_data = []
            for region in result.get('regions', []):
                regions_data.append({
                    'region_id': region['region_id'],
                    'region_number': region['region_number'],
                    'start_time': region['start_time'],
                    'end_time': region['end_time'],
                    'duration': region['duration'],
                    'region_type': region.get('region_type', 'content')
                })
            
            # Convert analysis results to database format
            analysis_results = {}
            for region_analysis in result.get('region_analyses', []):
                region_id = region_analysis['region_id']
                if region_id not in analysis_results:
                    analysis_results[region_id] = {}
                
                # Store each plugin result with proper data structure
                for plugin_name, plugin_result in region_analysis['plugin_results'].items():
                    # Extract the actual plugin data and metadata
                    plugin_data = plugin_result.get('data', {})
                    processing_time = plugin_result.get('processing_time', 0) * 1000  # Convert to ms
                    
                    # Create the database format with processing_time_ms and analysis_metadata
                    analysis_results[region_id][plugin_name] = {
                        **plugin_data,  # All the plugin results
                        'processing_time_ms': int(processing_time),
                        'analysis_metadata': plugin_data.get('analysis_metadata', {})
                    }
            
            # Store in database
            file_id = self.db_integration.store_file_analysis(
                str(file_path),
                file_metadata,
                regions_data,
                analysis_results
            )
            
            self.logger.info(f"✅ Stored analysis results in database (File ID: {file_id})")
            
        except Exception as e:
            self.logger.error(f"Failed to store results in database: {e}")
            raise
    
    def shutdown(self):
        """Gracefully shutdown the engine and cleanup resources."""
        self.logger.info("AudioSamplerEngine shutting down...")
        
        try:
            if self.resource_manager:
                self.resource_manager.cleanup()
            if self.db_integration:
                self.db_integration.close()
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
        
        self.logger.info("AudioSamplerEngine shutdown complete")