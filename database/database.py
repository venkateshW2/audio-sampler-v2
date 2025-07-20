"""
Audio Intelligence Sampler v2 - Database Management

Database operations for region-based audio analysis storage.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from datetime import datetime

from .models import Base, File, Region, RegionAnalysis, FileSummary, Tag

logger = logging.getLogger(__name__)


class Database:
    """
    Database management for audio intelligence sampler.
    
    Handles SQLite database creation, sessions, and high-level operations
    following the sacred region-based architecture.
    """
    
    def __init__(self, db_path: str = "audio_sampler.db"):
        """Initialize database connection."""
        self.db_path = db_path
        self.engine = None
        self.SessionLocal = None
        self._initialize()
    
    def _initialize(self):
        """Initialize database engine and create tables."""
        try:
            # Create SQLite database with optimizations
            db_url = f"sqlite:///{self.db_path}"
            self.engine = create_engine(
                db_url,
                echo=False,  # Set to True for SQL debugging
                pool_pre_ping=True,
                connect_args={
                    "check_same_thread": False,  # Allow multi-threading
                    "timeout": 30,  # Connection timeout
                }
            )
            
            # Create session factory
            self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            
            # Apply optimizations and indexes
            self._apply_optimizations()
            
            logger.info(f"Database initialized: {self.db_path}")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _apply_optimizations(self):
        """Apply SQLite optimizations and create indexes."""
        try:
            with self.engine.connect() as conn:
                # SQLite performance optimizations
                conn.execute(text("PRAGMA journal_mode=WAL"))  # Write-Ahead Logging
                conn.execute(text("PRAGMA synchronous=NORMAL"))  # Faster writes
                conn.execute(text("PRAGMA cache_size=10000"))  # 10MB cache
                conn.execute(text("PRAGMA temp_store=MEMORY"))  # Use memory for temp
                conn.execute(text("PRAGMA mmap_size=268435456"))  # 256MB memory-mapped I/O
                
                # Create additional indexes for performance
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_files_status ON files(status)",
                    "CREATE INDEX IF NOT EXISTS idx_files_date_analyzed ON files(date_analyzed)",
                    "CREATE INDEX IF NOT EXISTS idx_regions_file_id ON regions(file_id)",
                    "CREATE INDEX IF NOT EXISTS idx_regions_status ON regions(status)",
                    "CREATE INDEX IF NOT EXISTS idx_region_analysis_plugin ON region_analysis(plugin_name)",
                    "CREATE INDEX IF NOT EXISTS idx_region_analysis_timestamp ON region_analysis(timestamp)",
                    "CREATE INDEX IF NOT EXISTS idx_tags_file_id ON tags(file_id)",
                ]
                
                for index_sql in indexes:
                    conn.execute(text(index_sql))
                
                conn.commit()
                
            logger.info("Database optimizations applied")
            
        except Exception as e:
            logger.warning(f"Failed to apply database optimizations: {e}")
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
    
    def add_file(self, file_path: str, **metadata) -> File:
        """
        Add a new file to the database.
        
        Args:
            file_path: Path to the audio file
            **metadata: Additional file metadata (size_bytes, duration_seconds, etc.)
        
        Returns:
            File: The created file record
        """
        with self.get_session() as session:
            try:
                # Check if file already exists
                existing_file = session.query(File).filter(File.path == file_path).first()
                if existing_file:
                    logger.info(f"File already exists in database: {file_path}")
                    return existing_file
                
                # Create new file record
                file_record = File(
                    path=file_path,
                    date_added=datetime.utcnow(),
                    status='pending',
                    **metadata
                )
                
                session.add(file_record)
                session.commit()
                session.refresh(file_record)
                
                logger.info(f"Added file to database: {file_path} (ID: {file_record.id})")
                return file_record
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to add file to database: {e}")
                raise
    
    def add_regions(self, file_id: int, regions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add regions for a file.
        
        Args:
            file_id: File ID
            regions_data: List of region data dictionaries
        
        Returns:
            List[Dict]: Created region records as dictionaries
        """
        with self.get_session() as session:
            try:
                regions = []
                for region_data in regions_data:
                    region = Region(
                        file_id=file_id,
                        date_created=datetime.utcnow(),
                        status='pending',
                        **region_data
                    )
                    regions.append(region)
                    session.add(region)
                
                session.commit()
                
                # Convert to dictionaries before returning
                result_regions = []
                for region in regions:
                    session.refresh(region)  # Ensure ID is populated
                    result_regions.append({
                        'id': region.id,
                        'file_id': region.file_id,
                        'region_id': region.region_id,
                        'region_number': region.region_number,
                        'start_time': region.start_time,
                        'end_time': region.end_time,
                        'duration': region.duration,
                        'region_type': region.region_type,
                        'status': region.status
                    })
                
                # Update file region count
                file_record = session.query(File).filter(File.id == file_id).first()
                if file_record:
                    file_record.region_count = len(regions)
                    session.commit()
                
                logger.info(f"Added {len(regions)} regions for file ID {file_id}")
                return result_regions
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to add regions: {e}")
                raise
    
    def add_analysis_result(self, region_id: int, file_id: int, 
                           plugin_name: str, plugin_version: str,
                           results: Dict[str, Any], processing_time_ms: int,
                           success: bool = True) -> RegionAnalysis:
        """
        Add analysis results for a region.
        
        Args:
            region_id: Region ID
            file_id: File ID (for easier queries)
            plugin_name: Name of the plugin
            plugin_version: Version of the plugin
            results: Analysis results dictionary
            processing_time_ms: Processing time in milliseconds
            success: Whether analysis was successful
        
        Returns:
            RegionAnalysis: Created analysis record
        """
        with self.get_session() as session:
            try:
                analysis = RegionAnalysis(
                    region_id=region_id,
                    file_id=file_id,
                    plugin_name=plugin_name,
                    plugin_version=plugin_version,
                    results=results,
                    processing_time_ms=processing_time_ms,
                    timestamp=datetime.utcnow(),
                    success=success
                )
                
                session.add(analysis)
                session.commit()
                session.refresh(analysis)
                
                logger.info(f"Added analysis result: {plugin_name} for region {region_id}")
                return analysis
                
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to add analysis result: {e}")
                raise
    
    def get_file_by_path(self, file_path: str) -> Optional[File]:
        """Get file by path."""
        with self.get_session() as session:
            return session.query(File).filter(File.path == file_path).first()
    
    def get_file_with_regions(self, file_id: int) -> Optional[File]:
        """Get file with all its regions and analysis results."""
        with self.get_session() as session:
            file_record = session.query(File).filter(File.id == file_id).first()
            if file_record:
                # Force load relationships
                _ = file_record.regions
                for region in file_record.regions:
                    _ = region.analysis_results
            return file_record
    
    def get_regions_by_file(self, file_id: int) -> List[Region]:
        """Get all regions for a file."""
        with self.get_session() as session:
            return session.query(Region).filter(Region.file_id == file_id).order_by(Region.region_number).all()
    
    def get_analysis_results(self, region_id: int, plugin_name: Optional[str] = None) -> List[RegionAnalysis]:
        """Get analysis results for a region, optionally filtered by plugin."""
        with self.get_session() as session:
            query = session.query(RegionAnalysis).filter(RegionAnalysis.region_id == region_id)
            if plugin_name:
                query = query.filter(RegionAnalysis.plugin_name == plugin_name)
            return query.order_by(RegionAnalysis.timestamp.desc()).all()
    
    def update_file_status(self, file_id: int, status: str, analysis_version: Optional[str] = None):
        """Update file processing status."""
        with self.get_session() as session:
            try:
                file_record = session.query(File).filter(File.id == file_id).first()
                if file_record:
                    file_record.status = status
                    if status == 'completed':
                        file_record.date_analyzed = datetime.utcnow()
                    if analysis_version:
                        file_record.analysis_version = analysis_version
                    session.commit()
                    logger.info(f"Updated file {file_id} status to {status}")
                    
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to update file status: {e}")
                raise
    
    def update_region_status(self, region_id: int, status: str):
        """Update region processing status."""
        with self.get_session() as session:
            try:
                region = session.query(Region).filter(Region.id == region_id).first()
                if region:
                    region.status = status
                    session.commit()
                    logger.info(f"Updated region {region_id} status to {status}")
                    
            except SQLAlchemyError as e:
                session.rollback()
                logger.error(f"Failed to update region status: {e}")
                raise
    
    def search_files(self, query: str = "", status: Optional[str] = None, 
                    limit: int = 100, offset: int = 0) -> Tuple[List[File], int]:
        """
        Search files with optional filters.
        
        Returns:
            Tuple[List[File], int]: (files, total_count)
        """
        with self.get_session() as session:
            base_query = session.query(File)
            
            # Apply filters
            if query:
                base_query = base_query.filter(File.path.contains(query))
            if status:
                base_query = base_query.filter(File.status == status)
            
            # Get total count
            total_count = base_query.count()
            
            # Apply pagination and get results
            files = base_query.order_by(File.date_added.desc()).offset(offset).limit(limit).all()
            
            return files, total_count
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self.get_session() as session:
            try:
                stats = {
                    'total_files': session.query(File).count(),
                    'total_regions': session.query(Region).count(),
                    'total_analysis_results': session.query(RegionAnalysis).count(),
                    'files_by_status': {},
                    'regions_by_status': {},
                    'plugins_used': []
                }
                
                # Files by status
                for status, count in session.query(File.status, session.query(File).filter(File.status == File.status).count()).distinct():
                    stats['files_by_status'][status] = count
                
                # Regions by status  
                for status, count in session.query(Region.status, session.query(Region).filter(Region.status == Region.status).count()).distinct():
                    stats['regions_by_status'][status] = count
                
                # Plugins used
                plugins = session.query(RegionAnalysis.plugin_name).distinct().all()
                stats['plugins_used'] = [p[0] for p in plugins]
                
                return stats
                
            except Exception as e:
                logger.error(f"Failed to get database stats: {e}")
                return {}
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")