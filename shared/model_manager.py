"""
Model Manager for Delayed Loading and VRAM Management

This module provides intelligent model loading management for upscaling models:
- Delayed loading: Models load only when first needed
- VRAM management: Automatic cleanup when switching models
- Loading progress tracking
- Memory optimization for different model types

Key Features:
- Model state tracking (loaded, loading, unloaded)
- Automatic model unloading on switch
- VRAM cleanup with CUDA cache clearing
- Loading progress callbacks
- Support for different model backends (SeedVR2, GAN, RIFE)
"""

import gc
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple, Union
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class ModelState(Enum):
    """Model loading states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    ERROR = "error"


class ModelType(Enum):
    """Supported model types"""
    SEEDVR2 = "seedvr2"
    GAN = "gan"
    RIFE = "rife"


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    model_type: ModelType
    model_name: str
    model_id: str  # Unique identifier for this model instance
    state: ModelState = ModelState.UNLOADED
    loaded_at: Optional[float] = None
    last_used: Optional[float] = None
    memory_usage_mb: Optional[int] = None
    runner_instance: Optional[Any] = None
    cache_context: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class ModelManager:
    """
    Intelligent model manager for delayed loading and VRAM management.

    Features:
    - Delayed loading: Models load only when first used
    - Automatic unloading: Previous models unloaded when switching
    - VRAM cleanup: CUDA cache clearing and garbage collection
    - Loading progress: Callbacks for UI updates
    - Memory tracking: Monitor VRAM usage per model
    """

    # Configuration
    max_loaded_models: int = 1  # Maximum number of models to keep loaded
    auto_unload_delay: float = 300.0  # Seconds before auto-unloading unused models
    enable_progress_callbacks: bool = True

    # State tracking
    loaded_models: Dict[str, ModelInfo] = field(default_factory=dict)
    current_model_id: Optional[str] = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _logger: logging.Logger = field(init=False)

    def __post_init__(self):
        self._logger = logging.getLogger("ModelManager")
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _generate_model_id(self, model_type: ModelType, model_name: str, **kwargs) -> str:
        """Generate unique model ID based on type and parameters"""
        key_parts = [model_type.value, model_name]

        # Add relevant parameters to make ID unique
        if model_type == ModelType.SEEDVR2:
            key_parts.extend([
                kwargs.get('dit_model', ''),
                str(kwargs.get('resolution', '')),
                str(kwargs.get('batch_size', '')),
            ])
        elif model_type == ModelType.GAN:
            key_parts.extend([
                kwargs.get('model_path', ''),
                str(kwargs.get('scale', '')),
            ])
        elif model_type == ModelType.RIFE:
            key_parts.extend([
                kwargs.get('model_path', ''),
                str(kwargs.get('scale', '')),
            ])

        return "|".join(key_parts)

    def _get_memory_usage(self) -> int:
        """Get current GPU memory usage in MB"""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0

        try:
            return torch.cuda.memory_allocated() // (1024 * 1024)
        except Exception:
            return 0

    def _clear_vram(self):
        """Clear VRAM and run garbage collection"""
        if not TORCH_AVAILABLE:
            return

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            self._logger.warning(f"Failed to clear CUDA cache: {e}")

        # Force garbage collection
        gc.collect()

    def _unload_model(self, model_info: ModelInfo):
        """Unload a specific model and clean up resources"""
        try:
            # Clear runner instance
            if model_info.runner_instance:
                # Call cleanup if available
                if hasattr(model_info.runner_instance, 'cleanup'):
                    model_info.runner_instance.cleanup()
                model_info.runner_instance = None

            # Clear cache context
            if model_info.cache_context:
                model_info.cache_context.clear()
                model_info.cache_context = None

            # Update state
            model_info.state = ModelState.UNLOADED
            model_info.loaded_at = None
            model_info.memory_usage_mb = None

            self._logger.info(f"Unloaded model: {model_info.model_id}")

        except Exception as e:
            self._logger.error(f"Error unloading model {model_info.model_id}: {e}")

    def _should_unload_model(self, model_info: ModelInfo) -> bool:
        """Determine if a model should be unloaded"""
        if model_info.state != ModelState.LOADED:
            return False

        # Always keep current model
        if model_info.model_id == self.current_model_id:
            return False

        # Check if we've exceeded max loaded models
        loaded_count = sum(1 for m in self.loaded_models.values()
                          if m.state == ModelState.LOADED)
        if loaded_count > self.max_loaded_models:
            return True

        # Check auto-unload delay
        if model_info.last_used and time.time() - model_info.last_used > self.auto_unload_delay:
            return True

        return False

    def _cleanup_excess_models(self):
        """Unload models that should be unloaded"""
        models_to_unload = []
        for model_info in self.loaded_models.values():
            if self._should_unload_model(model_info):
                models_to_unload.append(model_info)

        for model_info in models_to_unload:
            self._unload_model(model_info)

        # Clear VRAM after unloading
        if models_to_unload:
            self._clear_vram()

    def is_model_loaded(self, model_type: ModelType, model_name: str, **kwargs) -> bool:
        """Check if a specific model is currently loaded"""
        model_id = self._generate_model_id(model_type, model_name, **kwargs)
        model_info = self.loaded_models.get(model_id)
        return model_info is not None and model_info.state == ModelState.LOADED

    def get_loaded_models_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all loaded models"""
        result = {}
        for model_id, model_info in self.loaded_models.items():
            result[model_id] = {
                "model_type": model_info.model_type.value,
                "model_name": model_info.model_name,
                "state": model_info.state.value,
                "loaded_at": model_info.loaded_at,
                "last_used": model_info.last_used,
                "memory_usage_mb": model_info.memory_usage_mb,
            }
        return result

    def preload_model(
        self,
        model_type: ModelType,
        model_name: str,
        load_callback: Callable,
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> bool:
        """
        Preload a model (called when user first selects it)

        Args:
            model_type: Type of model to load
            model_name: Name of the model
            load_callback: Function that actually loads the model
            progress_callback: Optional progress callback
            **kwargs: Additional parameters for model identification

        Returns:
            True if loaded successfully, False otherwise
        """
        model_id = self._generate_model_id(model_type, model_name, **kwargs)

        with self._lock:
            # Check if already loaded
            if model_id in self.loaded_models:
                model_info = self.loaded_models[model_id]
                if model_info.state == ModelState.LOADED:
                    model_info.last_used = time.time()
                    self.current_model_id = model_id
                    return True
                elif model_info.state == ModelState.LOADING:
                    # Wait for loading to complete
                    return False

            # Create new model info
            model_info = ModelInfo(
                model_type=model_type,
                model_name=model_name,
                model_id=model_id,
                state=ModelState.LOADING,
            )
            self.loaded_models[model_id] = model_info
            self.current_model_id = model_id

        try:
            # Update progress
            if progress_callback and self.enable_progress_callbacks:
                progress_callback(f"Loading {model_type.value} model: {model_name}")

            # Clear VRAM before loading new model
            self._clear_vram()

            # Load the model
            start_time = time.time()
            runner_instance, cache_context = load_callback()

            # Update model info
            with self._lock:
                model_info.state = ModelState.LOADED
                model_info.loaded_at = start_time
                model_info.last_used = time.time()
                model_info.runner_instance = runner_instance
                model_info.cache_context = cache_context
                model_info.memory_usage_mb = self._get_memory_usage()

            self._logger.info(f"Successfully loaded model: {model_id} ({model_info.memory_usage_mb}MB VRAM)")

            # Cleanup excess models
            self._cleanup_excess_models()

            if progress_callback and self.enable_progress_callbacks:
                progress_callback(f"Model loaded successfully: {model_name}")

            return True

        except Exception as e:
            error_msg = str(e)
            with self._lock:
                model_info.state = ModelState.ERROR
                model_info.error_message = error_msg

            self._logger.error(f"Failed to load model {model_id}: {error_msg}")

            if progress_callback and self.enable_progress_callbacks:
                progress_callback(f"Failed to load model: {error_msg}")

            return False

    def get_model_runner(
        self,
        model_type: ModelType,
        model_name: str,
        **kwargs
    ) -> Optional[Tuple[Any, Dict[str, Any]]]:
        """
        Get the runner instance for a loaded model

        Returns:
            Tuple of (runner_instance, cache_context) or None if not loaded
        """
        model_id = self._generate_model_id(model_type, model_name, **kwargs)
        model_info = self.loaded_models.get(model_id)

        if model_info and model_info.state == ModelState.LOADED:
            model_info.last_used = time.time()
            return model_info.runner_instance, model_info.cache_context

        return None

    def unload_all_models(self):
        """Unload all loaded models"""
        with self._lock:
            for model_info in self.loaded_models.values():
                if model_info.state == ModelState.LOADED:
                    self._unload_model(model_info)

            self.loaded_models.clear()
            self.current_model_id = None

        self._clear_vram()
        self._logger.info("All models unloaded")

    def switch_model(
        self,
        new_model_type: ModelType,
        new_model_name: str,
        load_callback: Callable,
        progress_callback: Optional[Callable[[str], None]] = None,
        **kwargs
    ) -> bool:
        """
        Switch to a different model, unloading the previous one if needed

        Returns:
            True if switch successful, False otherwise
        """
        new_model_id = self._generate_model_id(new_model_type, new_model_name, **kwargs)

        with self._lock:
            # If it's the same model, just update usage
            if new_model_id == self.current_model_id:
                model_info = self.loaded_models.get(new_model_id)
                if model_info:
                    model_info.last_used = time.time()
                return True

            # Check if new model is already loaded
            if new_model_id in self.loaded_models:
                new_model_info = self.loaded_models[new_model_id]
                if new_model_info.state == ModelState.LOADED:
                    self.current_model_id = new_model_id
                    new_model_info.last_used = time.time()
                    # Unload other models if needed
                    self._cleanup_excess_models()
                    return True

        # Load the new model
        return self.preload_model(
            new_model_type,
            new_model_name,
            load_callback,
            progress_callback,
            **kwargs
        )

    def get_vram_usage(self) -> Dict[str, Any]:
        """
        Get current VRAM usage information.
        """
        total_usage = 0
        model_breakdown = {}

        for model_id, model_info in self.loaded_models.items():
            if model_info.state == ModelState.LOADED and model_info.memory_usage_mb:
                total_usage += model_info.memory_usage_mb
                model_breakdown[model_id] = {
                    "model_name": model_info.model_name,
                    "model_type": model_info.model_type.value,
                    "memory_mb": model_info.memory_usage_mb,
                    "last_used": model_info.last_used
                }

        # Try to get actual GPU memory info
        gpu_info = {}
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    gpu_info[f"cuda:{i}"] = {
                        "allocated_mb": torch.cuda.memory_allocated(i) // (1024 * 1024),
                        "reserved_mb": torch.cuda.memory_reserved(i) // (1024 * 1024),
                        "total_mb": props.total_memory // (1024 * 1024),
                        "name": props.name
                    }
            except Exception:
                pass

        return {
            "total_tracked_usage_mb": total_usage,
            "model_breakdown": model_breakdown,
            "gpu_info": gpu_info,
            "loaded_model_count": len([m for m in self.loaded_models.values() if m.state == ModelState.LOADED])
        }

    def cleanup_idle_models(self, max_idle_time: float = 600.0):
        """
        Unload models that haven't been used for more than max_idle_time seconds.
        """
        current_time = time.time()
        models_to_unload = []

        for model_id, model_info in self.loaded_models.items():
            if (model_info.state == ModelState.LOADED and
                model_info.last_used and
                (current_time - model_info.last_used) > max_idle_time):
                models_to_unload.append(model_id)

        for model_id in models_to_unload:
            self.unload_model(model_id)

        return len(models_to_unload)

    def force_vram_cleanup(self):
        """
        Force cleanup of GPU memory (PyTorch CUDA cache).
        """
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                return True
            except Exception:
                pass
        return False


# Global instance
# Singleton instance with thread-safe initialization
_model_manager_instance: Optional[ModelManager] = None
_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """
    Get singleton ModelManager instance (thread-safe).
    
    This ensures only one ModelManager exists across the entire application,
    preventing duplicate model loading and VRAM waste.
    """
    global _model_manager_instance
    
    if _model_manager_instance is None:
        with _manager_lock:
            # Double-check locking pattern for thread safety
            if _model_manager_instance is None:
                _model_manager_instance = ModelManager()
    
    return _model_manager_instance
