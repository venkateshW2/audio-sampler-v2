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
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            # Read and save uploaded file
            content = await file.read()
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
            
            # Process the temporary file
            result = engine.process_file(tmp_path)
            
            if not result:
                raise HTTPException(status_code=500, detail="Engine returned empty result")
            
            # Add upload metadata
            result['upload_metadata'] = {
                'original_filename': file.filename,
                'content_type': file.content_type,
                'file_size': len(content)
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
            # Clean up temporary file
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


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