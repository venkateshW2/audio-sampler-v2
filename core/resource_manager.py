"""
Audio Intelligence Sampler v2 - Resource Manager

Prevents ALL resource conflicts and manages GPU/CPU/Memory safely.
Critical for GTX 1060 with 6GB VRAM - never exceed limits.

Architecture principles:
- Single GPU mutex (one GPU operation at a time)
- Memory monitoring with 80% limits
- CPU thread pool management  
- Automatic cleanup on resource pressure
- Never crash - graceful degradation always
"""

import logging
import threading
import time
import psutil
import gc
from typing import Dict, Optional, Any, Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime


class ResourceManager:
    """
    Manages all system resources to prevent conflicts and crashes.
    
    Critical for preventing GPU memory issues on GTX 1060 (6GB).
    Ensures stable operation under resource pressure.
    """
    
    def __init__(self, config: Dict = None):
        """Initialize resource manager with configuration.
        
        Args:
            config: Resource configuration dictionary
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # GPU management
        self.gpu_mutex = threading.Lock()
        self.gpu_available = False
        self.gpu_memory_limit = self.config.get('gpu_memory_limit_gb', 5.0)  # 5GB for GTX 1060
        
        # CPU management
        self.cpu_count = psutil.cpu_count()
        self.max_workers = self.config.get('max_workers', min(4, self.cpu_count))
        self.thread_pool = None
        
        # Memory management
        self.memory_limit_percent = self.config.get('memory_limit_percent', 80)
        self.cleanup_threshold_percent = self.config.get('cleanup_threshold_percent', 75)
        
        # Monitoring
        self.last_cleanup = time.time()
        self.cleanup_interval = self.config.get('cleanup_interval_seconds', 60)
        
        self.logger.info("ResourceManager initializing...")
        self._initialize_resources()
    
    def _initialize_resources(self):
        """Initialize and detect available resources."""
        try:
            # Detect GPU availability
            self._detect_gpu()
            
            # Initialize CPU thread pool
            self._initialize_thread_pool()
            
            # Log resource status
            self._log_resource_status()
            
            self.logger.info("ResourceManager initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Resource initialization error: {e}")
            # Continue with degraded functionality
    
    def _detect_gpu(self):
        """Detect GPU availability and capabilities."""
        try:
            import torch
            
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    device_name = torch.cuda.get_device_name(0)
                    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    
                    self.gpu_available = True
                    self.logger.info(f"GPU detected: {device_name} ({memory_gb:.1f}GB)")
                    
                    # Adjust memory limit based on actual GPU memory
                    if memory_gb < self.gpu_memory_limit:
                        self.gpu_memory_limit = memory_gb * 0.8  # 80% of available
                        self.logger.info(f"Adjusted GPU memory limit to {self.gpu_memory_limit:.1f}GB")
                else:
                    self.logger.warning("CUDA available but no GPU devices found")
            else:
                self.logger.info("CUDA not available - CPU-only mode")
                
        except ImportError:
            self.logger.warning("PyTorch not available - CPU-only mode")
        except Exception as e:
            self.logger.error(f"GPU detection error: {e}")
    
    def _initialize_thread_pool(self):
        """Initialize CPU thread pool for parallel processing."""
        try:
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.max_workers,
                thread_name_prefix="AudioSampler"
            )
            self.logger.info(f"Thread pool initialized with {self.max_workers} workers")
        except Exception as e:
            self.logger.error(f"Thread pool initialization error: {e}")
    
    def _log_resource_status(self):
        """Log current resource status."""
        memory = psutil.virtual_memory()
        
        self.logger.info("Resource Status:")
        self.logger.info(f"  CPU: {self.cpu_count} cores, {self.max_workers} workers")
        self.logger.info(f"  Memory: {memory.total / (1024**3):.1f}GB total, {memory.percent}% used")
        self.logger.info(f"  GPU: {'Available' if self.gpu_available else 'Not available'}")
        if self.gpu_available:
            self.logger.info(f"  GPU Memory Limit: {self.gpu_memory_limit:.1f}GB")
    
    def acquire_gpu(self, operation_name: str = "unknown") -> bool:
        """
        Acquire exclusive GPU access for an operation.
        
        Args:
            operation_name: Name of operation for logging
            
        Returns:
            True if GPU acquired, False if not available or failed
        """
        if not self.gpu_available:
            self.logger.debug(f"GPU not available for {operation_name}")
            return False
        
        try:
            acquired = self.gpu_mutex.acquire(blocking=False)
            if acquired:
                self.logger.debug(f"GPU acquired for {operation_name}")
                return True
            else:
                self.logger.debug(f"GPU busy, {operation_name} will use CPU")
                return False
        except Exception as e:
            self.logger.error(f"GPU acquisition error for {operation_name}: {e}")
            return False
    
    def release_gpu(self, operation_name: str = "unknown"):
        """
        Release GPU access and clean up GPU memory.
        
        Args:
            operation_name: Name of operation for logging
        """
        try:
            if self.gpu_available:
                # Clean up GPU memory
                self._cleanup_gpu_memory()
                self.logger.debug(f"GPU released by {operation_name}")
            
            self.gpu_mutex.release()
            
        except Exception as e:
            self.logger.error(f"GPU release error for {operation_name}: {e}")
    
    def _cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent accumulation."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        except Exception as e:
            self.logger.error(f"GPU memory cleanup error: {e}")
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check current memory usage and pressure.
        
        Returns:
            Dict with memory status and recommendations
        """
        memory = psutil.virtual_memory()
        
        status = {
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'pressure_level': 'low',
            'cleanup_recommended': False,
            'can_process': True
        }
        
        if memory.percent > self.memory_limit_percent:
            status['pressure_level'] = 'high'
            status['can_process'] = False
            status['cleanup_recommended'] = True
        elif memory.percent > self.cleanup_threshold_percent:
            status['pressure_level'] = 'medium'
            status['cleanup_recommended'] = True
        
        return status
    
    def cleanup_resources(self, force: bool = False):
        """Clean up system resources to free memory.
        
        Args:
            force: Force cleanup even if not due
        """
        current_time = time.time()
        
        if not force and (current_time - self.last_cleanup) < self.cleanup_interval:
            return
        
        try:
            self.logger.info("Cleaning up resources...")
            
            # Python garbage collection
            gc.collect()
            
            # GPU memory cleanup
            if self.gpu_available:
                self._cleanup_gpu_memory()
            
            self.last_cleanup = current_time
            
            # Log memory status after cleanup
            memory_status = self.check_memory_pressure()
            self.logger.info(f"Cleanup complete - Memory: {memory_status['memory_percent']:.1f}%")
            
        except Exception as e:
            self.logger.error(f"Resource cleanup error: {e}")
    
    def can_process_file(self, file_size_bytes: int) -> Dict[str, Any]:
        """Check if system can handle processing a file of given size.
        
        Args:
            file_size_bytes: Size of file to process
            
        Returns:
            Dict with processing capability assessment
        """
        memory_status = self.check_memory_pressure()
        file_size_gb = file_size_bytes / (1024**3)
        
        # Estimate memory needed (rough heuristic: 10x file size for audio processing)
        estimated_memory_gb = file_size_gb * 10
        
        result = {
            'can_process': True,
            'file_size_gb': file_size_gb,
            'estimated_memory_gb': estimated_memory_gb,
            'current_memory_percent': memory_status['memory_percent'],
            'memory_available_gb': memory_status['memory_available_gb'],
            'recommendations': []
        }
        
        if estimated_memory_gb > memory_status['memory_available_gb']:
            result['can_process'] = False
            result['recommendations'].append(f"File too large: needs ~{estimated_memory_gb:.1f}GB, only {memory_status['memory_available_gb']:.1f}GB available")
        
        if memory_status['pressure_level'] == 'high':
            result['can_process'] = False
            result['recommendations'].append("Memory pressure too high, cleanup needed")
        
        if file_size_gb > 1.0:  # Large file warning
            result['recommendations'].append("Large file - consider processing in chunks")
        
        return result
    
    def submit_cpu_task(self, func: Callable, *args, **kwargs):
        """Submit a CPU-intensive task to the thread pool.
        
        Args:
            func: Function to execute
            *args: Function arguments
            **kwargs: Function keyword arguments
            
        Returns:
            Future object or None if thread pool not available
        """
        try:
            if self.thread_pool:
                return self.thread_pool.submit(func, *args, **kwargs)
            else:
                self.logger.warning("Thread pool not available, running synchronously")
                return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Task submission error: {e}")
            return None
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource status.
        
        Returns:
            Dict with current resource utilization and status
        """
        memory_status = self.check_memory_pressure()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        status = {
            'timestamp': datetime.now().isoformat(),
            'cpu': {
                'cores': self.cpu_count,
                'usage_percent': cpu_percent,
                'thread_pool_workers': self.max_workers
            },
            'memory': memory_status,
            'gpu': {
                'available': self.gpu_available,
                'memory_limit_gb': self.gpu_memory_limit if self.gpu_available else 0,
                'mutex_locked': self.gpu_mutex.locked()
            }
        }
        
        if self.gpu_available:
            try:
                import torch
                status['gpu']['memory_allocated_gb'] = torch.cuda.memory_allocated() / (1024**3)
                status['gpu']['memory_reserved_gb'] = torch.cuda.memory_reserved() / (1024**3)
            except Exception:
                pass
        
        return status
    
    def cleanup(self):
        """Clean up and shutdown resource manager."""
        self.logger.info("ResourceManager shutting down...")
        
        try:
            # Cleanup resources one final time
            self.cleanup_resources(force=True)
            
            # Shutdown thread pool
            if self.thread_pool:
                self.thread_pool.shutdown(wait=True)
                self.logger.info("Thread pool shutdown complete")
            
            self.logger.info("ResourceManager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"ResourceManager shutdown error: {e}")


# Context manager for GPU operations
class GPUContext:
    """Context manager for safe GPU operations."""
    
    def __init__(self, resource_manager: ResourceManager, operation_name: str = "unknown"):
        self.resource_manager = resource_manager
        self.operation_name = operation_name
        self.gpu_acquired = False
    
    def __enter__(self):
        self.gpu_acquired = self.resource_manager.acquire_gpu(self.operation_name)
        return self.gpu_acquired
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.gpu_acquired:
            self.resource_manager.release_gpu(self.operation_name)