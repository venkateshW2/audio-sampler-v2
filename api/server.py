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
from typing import Dict, Any, List, Optional
import logging
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Create API router
router = APIRouter()


@router.post("/analyze/upload")
async def analyze_uploaded_file(file: UploadFile = File(...)):
    """
    Analyze an uploaded audio file.
    
    For Phase 1, this provides basic file upload capability.
    File is saved temporarily and processed through the engine.
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
            from main import engine
            
            if not engine:
                raise HTTPException(status_code=503, detail="Engine not available")
            
            # Process the temporary file
            result = engine.process_file(tmp_path)
            
            # Add upload metadata
            result['upload_metadata'] = {
                'original_filename': file.filename,
                'content_type': file.content_type,
                'file_size': len(content)
            }
            
            return {
                "analysis_result": result,
                "message": "File uploaded and analyzed successfully"
            }
            
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
        from main import engine
        
        if not engine or not engine.resource_manager:
            raise HTTPException(status_code=503, detail="Resource manager not available")
        
        resource_status = engine.resource_manager.get_resource_status()
        
        return {
            "resources": resource_status,
            "recommendations": _get_resource_recommendations(resource_status)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Resource status error: {e}")
        raise HTTPException(status_code=500, detail="Resource status unavailable")


@router.post("/system/cleanup")
async def cleanup_resources():
    """Trigger system resource cleanup."""
    try:
        from main import engine
        
        if not engine or not engine.resource_manager:
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