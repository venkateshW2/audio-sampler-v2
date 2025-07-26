"""
Audio Intelligence Sampler v2 - Database Integration

Integration between the core engine and database for storing analysis results.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .database import Database
from .models import File, Region, RegionAnalysis

logger = logging.getLogger(__name__)


class DatabaseIntegration:
    """
    Integration layer between core engine and database.
    
    Handles the conversion and storage of engine results into the database
    following the sacred region-based architecture.
    """
    
    def __init__(self, db_path: str = "audio_sampler.db"):
        """Initialize database integration."""
        self.db = Database(db_path)
        logger.info("Database integration initialized")
    
    def store_file_analysis(self, file_path: str, file_metadata: Dict[str, Any],
                           regions_data: List[Dict[str, Any]], 
                           analysis_results: Dict[str, Dict[str, Any]]) -> int:
        """
        Store complete file analysis in database.
        
        Args:
            file_path: Path to the audio file
            file_metadata: File metadata (duration, sample_rate, etc.)
            regions_data: List of region data from engine
            analysis_results: Plugin analysis results keyed by region_id
        
        Returns:
            int: File ID in database
        """
        try:
            # Add or get file record
            file_record = self.db.add_file(file_path, **file_metadata)
            
            # Update file status to processing
            self.db.update_file_status(file_record.id, 'processing')
            
            # Add regions
            regions = self.db.add_regions(file_record.id, regions_data)
            
            # Store analysis results for each region
            for region_dict in regions:
                region_id_key = region_dict['region_id']
                region_db_id = region_dict['id']
                region_file_id = region_dict['file_id']
                
                if region_id_key in analysis_results:
                    region_results = analysis_results[region_id_key]
                    self._store_region_analysis_by_id(region_db_id, region_file_id, region_results)
            
            # Update file status to completed
            self.db.update_file_status(file_record.id, 'completed', analysis_version='v2.0.0')
            
            logger.info(f"Stored complete file analysis: {file_path} (ID: {file_record.id})")
            return file_record.id
            
        except Exception as e:
            logger.error(f"Failed to store file analysis: {e}")
            # Update file status to failed
            if 'file_record' in locals():
                self.db.update_file_status(file_record.id, 'failed')
            raise
    
    def _store_region_analysis_by_id(self, region_id: int, file_id: int, analysis_results: Dict[str, Dict[str, Any]]):
        """Store analysis results for a single region by ID."""
        try:
            # Update region status to processing
            self.db.update_region_status(region_id, 'processing')
            
            # Store results from each plugin
            for plugin_name, plugin_result in analysis_results.items():
                if plugin_result.get('success', False):
                    # Extract processing time
                    processing_time_ms = plugin_result.get('processing_time_ms', 0)
                    
                    # Extract plugin version
                    plugin_version = plugin_result.get('analysis_metadata', {}).get('version', 'unknown')
                    
                    # Store analysis result
                    self.db.add_analysis_result(
                        region_id=region_id,
                        file_id=file_id,
                        plugin_name=plugin_name,
                        plugin_version=plugin_version,
                        results=plugin_result,
                        processing_time_ms=processing_time_ms,
                        success=True
                    )
                else:
                    # Store failed analysis
                    self.db.add_analysis_result(
                        region_id=region_id,
                        file_id=file_id,
                        plugin_name=plugin_name,
                        plugin_version='unknown',
                        results=plugin_result,
                        processing_time_ms=0,
                        success=False
                    )
            
            # Update region status to completed
            self.db.update_region_status(region_id, 'completed')
            
        except Exception as e:
            logger.error(f"Failed to store region analysis for region {region_id}: {e}")
            # Update region status to failed
            self.db.update_region_status(region_id, 'failed')
            raise
    
    def get_file_results(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get stored analysis results for a file.
        
        Args:
            file_path: Path to the audio file
        
        Returns:
            Dict containing file and analysis data, or None if not found
        """
        try:
            file_record = self.db.get_file_by_path(file_path)
            if not file_record:
                return None
            
            file_with_regions = self.db.get_file_with_regions(file_record.id)
            if not file_with_regions:
                return None
            
            # Convert to engine-compatible API format
            file_dict = file_with_regions.to_dict()
            result = {
                'file_id': file_with_regions.id,  # Include file ID at top level for audio serving
                'file_path': file_with_regions.path,  # Include file path at top level
                'file_metadata': file_dict,  # Match engine format
                'duration_seconds': file_dict.get('duration_seconds', 0),
                'success': True,
                'timestamp': file_dict.get('date_analyzed') or file_dict.get('date_added'),
                'regions': [],
                'region_analyses': [],  # Add engine-compatible format
                'analysis_summary': self._generate_analysis_summary(file_with_regions)
            }
            
            # Add regions with their analysis results in both formats
            for region in file_with_regions.regions:
                # Database format (existing)
                region_data = region.to_dict()
                region_data['analysis_results'] = {}
                
                # Engine format (new) - for timeline visualization compatibility
                region_analysis = {
                    'region_id': region.region_id,
                    'region_metadata': {
                        'start_time': region.start_time,
                        'end_time': region.end_time,
                        'duration': region.duration
                    },
                    'plugin_results': {}
                }
                
                for analysis in region.analysis_results:
                    analysis_dict = analysis.to_dict()
                    
                    # Database format
                    region_data['analysis_results'][analysis.plugin_name] = analysis_dict
                    
                    # Engine format - wrap in expected structure
                    plugin_data = analysis_dict.get('results', {})
                    region_analysis['plugin_results'][analysis.plugin_name] = {
                        'data': plugin_data,
                        'processing_time': (analysis_dict.get('processing_time_ms', 0) / 1000.0),  # Convert ms to seconds
                        'success': analysis_dict.get('success', True)
                    }
                
                result['regions'].append(region_data)
                result['region_analyses'].append(region_analysis)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get file results: {e}")
            return None
    
    def _generate_analysis_summary(self, file_record: File) -> Dict[str, Any]:
        """Generate analysis summary for a file."""
        try:
            summary = {
                'total_regions': len(file_record.regions),
                'completed_regions': 0,
                'failed_regions': 0,
                'plugins_used': set(),
                'total_processing_time_ms': 0,
                'dominant_classifications': [],
                'tempo_range': {'min': None, 'max': None, 'average': None},
                'key_distribution': {}
            }
            
            tempos = []
            classifications = []
            keys = []
            
            for region in file_record.regions:
                if region.status == 'completed':
                    summary['completed_regions'] += 1
                elif region.status == 'failed':
                    summary['failed_regions'] += 1
                
                for analysis in region.analysis_results:
                    if analysis.success:
                        summary['plugins_used'].add(analysis.plugin_name)
                        summary['total_processing_time_ms'] += analysis.processing_time_ms or 0
                        
                        # Extract specific metrics
                        results = analysis.results
                        
                        if analysis.plugin_name == 'tempo_meter_detector':
                            tempo = results.get('tempo_bpm')
                            if tempo and tempo > 0:
                                tempos.append(tempo)
                        
                        elif analysis.plugin_name == 'classifier':
                            top_prediction = results.get('top_prediction')
                            if top_prediction:
                                classifications.append(top_prediction)
                        
                        elif analysis.plugin_name == 'key_finder':
                            key = results.get('key')
                            mode = results.get('mode')
                            if key and mode:
                                key_str = f"{key} {mode}"
                                keys.append(key_str)
            
            # Calculate summaries
            summary['plugins_used'] = list(summary['plugins_used'])
            
            if tempos:
                summary['tempo_range'] = {
                    'min': min(tempos),
                    'max': max(tempos),
                    'average': sum(tempos) / len(tempos)
                }
            
            if classifications:
                from collections import Counter
                summary['dominant_classifications'] = Counter(classifications).most_common(3)
            
            if keys:
                from collections import Counter
                summary['key_distribution'] = dict(Counter(keys))
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate analysis summary: {e}")
            return {}
    
    def search_files(self, query: str = "", status: Optional[str] = None,
                    limit: int = 100, offset: int = 0) -> Dict[str, Any]:
        """Search files with pagination."""
        try:
            files, total_count = self.db.search_files(query, status, limit, offset)
            
            return {
                'files': [f.to_dict() for f in files],
                'total_count': total_count,
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total_count
            }
            
        except Exception as e:
            logger.error(f"Failed to search files: {e}")
            return {'files': [], 'total_count': 0, 'limit': limit, 'offset': offset, 'has_more': False}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return self.db.get_database_stats()
    
    def close(self):
        """Close database connection."""
        self.db.close()