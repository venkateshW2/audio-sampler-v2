"""
Audio Intelligence Sampler v2 - Main Application

FastAPI application entry point following sacred architecture.
Serves the core engine via REST API for Phase 1.

Architecture principles:
- API-first design (UI and processing separated)
- Network transparent (works local or remote)
- Never crash - graceful error handling
- Comprehensive logging
"""

import logging
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Dict, Any
import json

# Import core components
from core.engine import AudioSamplerEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Audio Intelligence Sampler v2",
    description="Professional audio analysis tool with PyTorch-based processing",
    version="2.0.0-phase1"
)

# Enable CORS for browser access from Mac
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine instance
engine = None


@app.on_event("startup")
async def startup_event():
    """Initialize the audio sampler engine on startup."""
    global engine
    
    logger.info("Audio Intelligence Sampler v2 starting up...")
    
    try:
        # Initialize engine with default config
        config = {
            'resources': {
                'gpu_memory_limit_gb': 5.0,  # GTX 1060 limit
                'memory_limit_percent': 80,
                'max_workers': 4
            },
            'file_handling': {
                'target_sample_rate': 22050,
                'force_mono': True,
                'normalize_audio': True
            },
            'database': {
                'path': 'audio_sampler.db'  # Enable database integration
            }
        }
        
        engine = AudioSamplerEngine(config)
        
        # Register core plugins
        from plugins.core_plugins.classifier import ClassifierPlugin
        from plugins.core_plugins.tempo_meter_detector import TempoMeterDetectorPlugin  
        from plugins.core_plugins.key_finder import KeyFinderPlugin
        
        engine.register_plugin(ClassifierPlugin())
        engine.register_plugin(TempoMeterDetectorPlugin())
        engine.register_plugin(KeyFinderPlugin())
        
        logger.info("Audio sampler engine initialized successfully with all core plugins")
        
    except Exception as e:
        logger.error(f"Engine initialization failed: {e}")
        # Continue anyway - engine will handle degraded mode


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global engine
    
    logger.info("Audio Intelligence Sampler v2 shutting down...")
    
    if engine:
        try:
            engine.shutdown()
        except Exception as e:
            logger.error(f"Engine shutdown error: {e}")


@app.get("/")
async def root():
    """Root endpoint - basic info."""
    return {
        "name": "Audio Intelligence Sampler v2",
        "version": "2.0.0-phase1",
        "status": "running",
        "phase": "Phase 1 - Core Backend Development",
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        status = engine.get_status()
        
        return {
            "status": "healthy",
            "timestamp": status["timestamp"],
            "engine": {
                "version": status["engine_version"],
                "uptime_seconds": status["uptime_seconds"],
                "plugins_registered": status["plugins_registered"],
                "components_ready": all(status["components"].values())
            },
            "system": {
                "gpu_available": status["components"]["resource_manager"],
                "file_handler_ready": status["components"]["file_handler"]
            }
        }
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


@app.get("/status")
async def get_status():
    """Detailed status information."""
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        return engine.get_status()
    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail="Status unavailable")


@app.post("/analyze")
async def analyze_file(request: Dict[str, Any]):
    """
    Analyze a single audio file.
    
    Request body:
    {
        "file_path": "/path/to/audio/file.wav"
    }
    """
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Validate request
        if "file_path" not in request:
            raise HTTPException(status_code=400, detail="file_path required")
        
        file_path = request["file_path"]
        
        # Process file
        result = engine.process_file(file_path)
        
        # Return result (engine never crashes, always returns something)
        return {
            "analysis_result": result,
            "api_version": "2.0.0-phase1"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.get("/plugins")
async def get_plugins():
    """Get information about registered plugins."""
    global engine
    
    if not engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        plugins_info = []
        for plugin in engine.plugins:
            try:
                plugin_info = {
                    "name": plugin.get_name(),
                    "version": plugin.get_version(),
                    "requirements": plugin.get_requirements(),
                    "initialized": plugin.is_initialized()
                }
                plugins_info.append(plugin_info)
            except Exception as e:
                plugins_info.append({
                    "name": "Unknown",
                    "error": str(e)
                })
        
        return {
            "plugins": plugins_info,
            "count": len(plugins_info)
        }
        
    except Exception as e:
        logger.error(f"Plugins info error: {e}")
        raise HTTPException(status_code=500, detail="Plugins info unavailable")


@app.get("/database/files")
async def search_files(query: str = "", status: str = None, limit: int = 100, offset: int = 0):
    """Search files in database with pagination."""
    global engine
    
    if not engine or not engine.db_integration:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        return engine.db_integration.search_files(query, status, limit, offset)
    except Exception as e:
        logger.error(f"Database search error: {e}")
        raise HTTPException(status_code=500, detail="Database search failed")


@app.get("/database/files/{file_path:path}")
async def get_file_results(file_path: str):
    """Get analysis results for a specific file."""
    global engine
    
    if not engine or not engine.db_integration:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        results = engine.db_integration.get_file_results(file_path)
        if not results:
            raise HTTPException(status_code=404, detail="File not found in database")
        return results
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Database file lookup error: {e}")
        raise HTTPException(status_code=500, detail="Database lookup failed")


@app.get("/database/stats")
async def get_database_stats():
    """Get database statistics."""
    global engine
    
    if not engine or not engine.db_integration:
        raise HTTPException(status_code=503, detail="Database not available")
    
    try:
        return engine.db_integration.get_stats()
    except Exception as e:
        logger.error(f"Database stats error: {e}")
        raise HTTPException(status_code=500, detail="Database stats unavailable")


# Include additional API routes
from api.server import router as api_router
app.include_router(api_router, prefix="/api", tags=["extended"])

# Mount static files for basic UI (Phase 1)
ui_path = Path(__file__).parent / "ui"
if ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(ui_path), html=True), name="ui")
    logger.info(f"UI mounted at /ui from {ui_path}")
else:
    logger.warning(f"UI directory not found at {ui_path}")


def main():
    """Main entry point for development server."""
    logger.info("Starting Audio Intelligence Sampler v2 development server...")
    
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",  # Bind to all interfaces for SSH access
        port=8000,
        reload=True,     # Auto-reload on code changes
        log_level="info"
    )


if __name__ == "__main__":
    main()