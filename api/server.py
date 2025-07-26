"""
Audio Intelligence Sampler v2 - API Server

Extended API endpoints for the audio analysis system.
Complements main.py with additional functionality.

Architecture principles:
- Stateless API design
- Comprehensive error handling
- Detailed logging
- Resource-aware processing
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import logging
import tempfile
import os
from pathlib import Path
from core.json_utils import safe_json_response

# Response models for API documentation
class AnalysisResponse(BaseModel):
    message: str
    success: bool
    file_info: Dict[str, Any]
    regions_processed: int
    processing_time_seconds: float
    
    class Config:
        schema_extra = {
            "example": {
                "message": "File uploaded and analyzed successfully",
                "success": True,
                "file_info": {
                    "filename": "audio.wav",
                    "duration": 27.3,
                    "sample_rate": 22050
                },
                "regions_processed": 1,
                "processing_time_seconds": 13.5
            }
        }

logger = logging.getLogger(__name__)

def clean_response_data(data):
    """Remove large audio arrays from response data to prevent JSON size issues."""
    if isinstance(data, dict):
        cleaned = {}
        for key, value in data.items():
            if key == 'audio_data':
                # Replace large audio data with summary
                if isinstance(value, list):
                    cleaned[key] = f"[Audio data array with {len(value)} samples - excluded from response]"
                else:
                    cleaned[key] = value
            elif key == 'sample_count' and isinstance(value, int) and value > 1000:
                # Keep sample count but don't include the actual samples
                cleaned[key] = value
            else:
                cleaned[key] = clean_response_data(value)
        return cleaned
    elif isinstance(data, list):
        # If it's a large array, summarize it
        if len(data) > 1000 and all(isinstance(x, (int, float)) for x in data[:10]):
            return f"[Large numeric array with {len(data)} elements - excluded from response]"
        return [clean_response_data(item) for item in data]
    else:
        return data

# Create API router
router = APIRouter()


@router.post("/analyze/upload", response_model=None)
async def analyze_uploaded_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file.
    
    For Phase 1, this provides basic file upload capability.
    File is saved temporarily and processed through the engine.
    
    Returns detailed analysis results including plugin outputs.
    Note: Response may be large due to audio data arrays.
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('audio/'):
            # Also check by extension
            if not any(file.filename.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.m4a']):
                raise HTTPException(
                    status_code=400, 
                    detail="File must be an audio file (.wav, .mp3, .flac, .m4a)"
                )
        
        # Read uploaded file content
        content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Import engine here to avoid circular imports
            try:
                from main import engine
            except ImportError as import_err:
                logger.error(f"Failed to import engine: {import_err}")
                raise HTTPException(status_code=503, detail="Engine module not available")
            
            if not engine:
                logger.error("Engine is None - server may not be fully initialized")
                raise HTTPException(status_code=503, detail="Engine not initialized")
            
            logger.info(f"Processing uploaded file: {file.filename} ({len(content)} bytes)")
            
            # Check for duplicates based on original filename
            if engine.db_integration:
                try:
                    from database.models import File as FileModel
                    with engine.db_integration.db.get_session() as session:
                        existing_files = session.query(FileModel).filter(
                            FileModel.path.like(f"%/{file.filename}")
                        ).all()
                        
                        if existing_files:
                            logger.info(f"Found {len(existing_files)} existing files with name {file.filename}")
                            # Return the most recent analysis instead of re-processing
                            most_recent = max(existing_files, key=lambda x: x.date_analyzed or x.date_added)
                            logger.info(f"Returning existing analysis for file ID {most_recent.id}")
                            
                            # Get existing analysis results
                            existing_result = engine.db_integration.get_file_analysis(most_recent.id)
                            if existing_result:
                                return {
                                    "analysis_result": existing_result,
                                    "api_version": "2.0.0-phase1",
                                    "duplicate_detection": {
                                        "is_duplicate": True,
                                        "original_file_id": most_recent.id,
                                        "message": f"Returning existing analysis for {file.filename}"
                                    }
                                }
                except Exception as e:
                    logger.warning(f"Duplicate detection failed, proceeding with analysis: {e}")
            
            # Process the temporary file
            result = engine.process_file(tmp_path)
            
            if not result:
                raise HTTPException(status_code=500, detail="Engine returned empty result")
            
            # Add upload metadata and processed file info
            result['upload_metadata'] = {
                'original_filename': file.filename,
                'content_type': file.content_type,
                'file_size': len(content),
                'processed_file_path': tmp_path  # Include processed file path for frontend
            }
            
            logger.info(f"Upload analysis completed successfully for {file.filename}")
            
            # Clean response data to remove large audio arrays
            cleaned_result = clean_response_data(result)
            
            # Use safe JSON response to handle numpy objects
            response_data = {
                "analysis_result": cleaned_result,
                "message": "File uploaded and analyzed successfully"
            }
            
            # Convert to JSON-safe format
            safe_data = safe_json_response(response_data)
            
            return JSONResponse(content=safe_data)
            
        finally:
            # Store processed file path for later cleanup (don't delete immediately)
            # Frontend needs access to this file for correct audio display
            logger.info(f"Processed file available at: {tmp_path}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/processed_audio/{file_path:path}")
async def serve_processed_audio(file_path: str):
    """
    Serve processed audio files to frontend for playback.
    Supports both temporary uploaded files and permanent database files.
    """
    import os
    from fastapi.responses import FileResponse
    
    try:
        # Check if it's a temporary file (uploaded)
        if os.path.basename(file_path).startswith('tmp') and os.path.exists(file_path):
            return FileResponse(
                path=file_path, 
                media_type="audio/mpeg",
                filename=os.path.basename(file_path)
            )
        
        # Check if it's a database file path
        if os.path.exists(file_path):
            # Determine media type from file extension
            ext = os.path.splitext(file_path)[1].lower()
            media_type_map = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav', 
                '.flac': 'audio/flac',
                '.m4a': 'audio/mp4',
                '.ogg': 'audio/ogg'
            }
            media_type = media_type_map.get(ext, 'audio/mpeg')
            
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=os.path.basename(file_path)
            )
        
        raise HTTPException(status_code=404, detail="Audio file not found")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving processed audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve audio: {str(e)}")


@router.get("/audio/{file_id}")
async def serve_database_audio(file_id: int):
    """
    Serve audio files from database by file ID.
    This is the preferred method for database-loaded files.
    """
    import os
    from fastapi.responses import FileResponse
    
    try:
        # Import engine
        try:
            from main import engine
        except ImportError as import_err:
            logger.error(f"Failed to import engine: {import_err}")
            raise HTTPException(status_code=503, detail="Engine module not available")
        
        if not engine or not engine.db_integration:
            raise HTTPException(status_code=503, detail="Database not available")
        
        # Get file from database
        from database.models import File as FileModel
        with engine.db_integration.db.get_session() as session:
            file_record = session.query(FileModel).filter(FileModel.id == file_id).first()
            
            if not file_record:
                raise HTTPException(status_code=404, detail="File not found in database")
            
            file_path = file_record.path
            
            # Check if file exists on disk
            if not os.path.exists(file_path):
                raise HTTPException(status_code=404, detail="Audio file not found on disk")
            
            # Determine media type from file extension
            ext = os.path.splitext(file_path)[1].lower()
            media_type_map = {
                '.mp3': 'audio/mpeg',
                '.wav': 'audio/wav', 
                '.flac': 'audio/flac',
                '.m4a': 'audio/mp4',
                '.ogg': 'audio/ogg'
            }
            media_type = media_type_map.get(ext, 'audio/mpeg')
            
            return FileResponse(
                path=file_path,
                media_type=media_type,
                filename=os.path.basename(file_path)
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving database audio: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve audio: {str(e)}")


@router.get("/analyze/batch")
async def analyze_batch(file_paths: List[str]):
    """
    Analyze multiple files in batch.
    
    Note: This is a placeholder for Phase 2.
    Phase 1 focuses on single file processing.
    """
    return {
        "message": "Batch processing will be available in Phase 2",
        "phase": "Phase 1 - Single file processing only",
        "requested_files": len(file_paths)
    }


@router.get("/system/resources")
async def get_system_resources():
    """Get current system resource status."""
    try:
        try:
            from main import engine
        except ImportError as import_err:
            logger.error(f"Failed to import engine: {import_err}")
            raise HTTPException(status_code=503, detail="Engine module not available")
        
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        if not engine.resource_manager:
            raise HTTPException(status_code=503, detail="Resource manager not available")
        
        resource_status = engine.resource_manager.get_resource_status()
        
        response_data = {
            "resources": resource_status,
            "recommendations": _get_resource_recommendations(resource_status)
        }
        
        safe_data = safe_json_response(response_data)
        return JSONResponse(content=safe_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resource status error: {e}")
        raise HTTPException(status_code=500, detail="Resource status unavailable")


@router.post("/system/cleanup")
async def cleanup_resources():
    """Trigger system resource cleanup."""
    try:
        try:
            from main import engine
        except ImportError as import_err:
            logger.error(f"Failed to import engine: {import_err}")
            raise HTTPException(status_code=503, detail="Engine module not available")
        
        if not engine:
            raise HTTPException(status_code=503, detail="Engine not initialized")
        
        if not engine.resource_manager:
            raise HTTPException(status_code=503, detail="Resource manager not available")
        
        # Force cleanup
        engine.resource_manager.cleanup_resources(force=True)
        
        # Get status after cleanup
        resource_status = engine.resource_manager.get_resource_status()
        
        return {
            "message": "Resource cleanup completed",
            "resources_after_cleanup": resource_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup error: {e}")
        raise HTTPException(status_code=500, detail="Cleanup failed")


def _get_resource_recommendations(resource_status: Dict[str, Any]) -> List[str]:
    """Generate resource usage recommendations."""
    recommendations = []
    
    try:
        memory = resource_status.get('memory', {})
        memory_percent = memory.get('memory_percent', 0)
        
        if memory_percent > 90:
            recommendations.append("Critical: Memory usage very high, consider restarting")
        elif memory_percent > 80:
            recommendations.append("Warning: Memory usage high, cleanup recommended")
        elif memory_percent > 70:
            recommendations.append("Info: Memory usage moderate, monitor closely")
        
        gpu = resource_status.get('gpu', {})
        if gpu.get('available') and gpu.get('mutex_locked'):
            recommendations.append("GPU currently in use by another operation")
        
        if not recommendations:
            recommendations.append("System resources look healthy")
            
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        recommendations.append("Unable to generate recommendations")
    
    return recommendations