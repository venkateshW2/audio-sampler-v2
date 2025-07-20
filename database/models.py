"""
Audio Intelligence Sampler v2 - Database Models

SQLAlchemy models for region-based audio analysis storage.
Implements the sacred architecture schema for files, regions, and analysis results.
"""

from sqlalchemy import create_engine, Column, Integer, String, Text, Float, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from datetime import datetime
from typing import Dict, Any, Optional

Base = declarative_base()


class File(Base):
    """
    Main files table - represents audio files being analyzed.
    
    Sacred architecture principle: Files contain regions, never process files directly.
    """
    __tablename__ = 'files'
    
    id = Column(Integer, primary_key=True)
    path = Column(Text, unique=True, nullable=False, index=True)
    size_bytes = Column(Integer)
    duration_seconds = Column(Float)
    sample_rate = Column(Integer)
    channels = Column(Integer)
    format = Column(String(10))
    date_added = Column(DateTime, default=datetime.utcnow)
    date_analyzed = Column(DateTime)
    analysis_version = Column(String(50))
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    region_count = Column(Integer, default=0)
    silence_threshold_seconds = Column(Float, default=10.0)
    
    # Relationships
    regions = relationship("Region", back_populates="file", cascade="all, delete-orphan")
    analysis_results = relationship("RegionAnalysis", back_populates="file")
    summary = relationship("FileSummary", back_populates="file", uselist=False)
    tags = relationship("Tag", back_populates="file", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<File(id={self.id}, path='{self.path}', regions={self.region_count})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'path': self.path,
            'size_bytes': self.size_bytes,
            'duration_seconds': self.duration_seconds,
            'sample_rate': self.sample_rate,
            'channels': self.channels,
            'format': self.format,
            'date_added': self.date_added.isoformat() if self.date_added else None,
            'date_analyzed': self.date_analyzed.isoformat() if self.date_analyzed else None,
            'analysis_version': self.analysis_version,
            'status': self.status,
            'region_count': self.region_count,
            'silence_threshold_seconds': self.silence_threshold_seconds
        }


class Region(Base):
    """
    Silence-detected regions within audio files.
    
    Sacred architecture principle: Regions are the fundamental unit of processing.
    Each region is analyzed independently.
    """
    __tablename__ = 'regions'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    region_id = Column(String(100), nullable=False, index=True)  # "filename-r1", "filename-r2"
    region_number = Column(Integer, nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    duration = Column(Float, nullable=False)
    region_type = Column(String(20), default='content')  # content, silence
    date_created = Column(DateTime, default=datetime.utcnow)
    status = Column(String(20), default='pending')  # pending, processing, completed, failed
    
    # Relationships
    file = relationship("File", back_populates="regions")
    analysis_results = relationship("RegionAnalysis", back_populates="region", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Region(id={self.id}, region_id='{self.region_id}', {self.start_time:.1f}-{self.end_time:.1f}s)>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'file_id': self.file_id,
            'region_id': self.region_id,
            'region_number': self.region_number,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'region_type': self.region_type,
            'date_created': self.date_created.isoformat() if self.date_created else None,
            'status': self.status
        }


class RegionAnalysis(Base):
    """
    Region-based analysis results from plugins.
    
    Sacred architecture principle: Store all plugin results as JSON to allow 
    flexible schema evolution without database migrations.
    """
    __tablename__ = 'region_analysis'
    
    id = Column(Integer, primary_key=True)
    region_id = Column(Integer, ForeignKey('regions.id'), nullable=False)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)  # For easier queries
    plugin_name = Column(String(100), nullable=False, index=True)
    plugin_version = Column(String(50))
    results = Column(JSON, nullable=False)  # Plugin-specific analysis results
    processing_time_ms = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    
    # Relationships
    region = relationship("Region", back_populates="analysis_results")
    file = relationship("File", back_populates="analysis_results")
    
    def __repr__(self):
        return f"<RegionAnalysis(id={self.id}, plugin='{self.plugin_name}', success={self.success})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'region_id': self.region_id,
            'file_id': self.file_id,
            'plugin_name': self.plugin_name,
            'plugin_version': self.plugin_version,
            'results': self.results,
            'processing_time_ms': self.processing_time_ms,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'success': self.success
        }


class FileSummary(Base):
    """
    File-level aggregated results (optional).
    
    Pre-computed summaries for faster queries across all regions in a file.
    """
    __tablename__ = 'file_summary'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False, unique=True)
    total_regions = Column(Integer)
    dominant_key = Column(String(20))
    average_tempo = Column(Float)
    classification_summary = Column(JSON)
    processing_summary = Column(JSON)
    
    # Relationships
    file = relationship("File", back_populates="summary")
    
    def __repr__(self):
        return f"<FileSummary(id={self.id}, file_id={self.file_id}, regions={self.total_regions})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'file_id': self.file_id,
            'total_regions': self.total_regions,
            'dominant_key': self.dominant_key,
            'average_tempo': self.average_tempo,
            'classification_summary': self.classification_summary,
            'processing_summary': self.processing_summary
        }


class Tag(Base):
    """
    User and automatic tags for files.
    
    Supports both user-added tags and confidence-based automatic tags.
    """
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)
    tag = Column(String(100), nullable=False, index=True)
    user_added = Column(Boolean, default=False)
    confidence = Column(Float)
    
    # Relationships
    file = relationship("File", back_populates="tags")
    
    def __repr__(self):
        return f"<Tag(id={self.id}, tag='{self.tag}', user_added={self.user_added})>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'file_id': self.file_id,
            'tag': self.tag,
            'user_added': self.user_added,
            'confidence': self.confidence
        }