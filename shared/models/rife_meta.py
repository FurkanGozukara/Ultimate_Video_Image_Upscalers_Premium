"""
RIFE Model Metadata Registry
Provides comprehensive model metadata with VRAM, compile, and multi-GPU constraints
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class RifeModel:
    """Comprehensive metadata for RIFE frame interpolation models"""
    name: str  # e.g., "rife-v4.17"
    version: str  # "4.6", "4.17", etc.
    variant: str  # "standard" or "anime"
    
    # Multi-GPU support
    supports_multi_gpu: bool = False  # RIFE is single-GPU optimized
    
    # Performance characteristics
    estimated_vram_gb: float = 4.0
    supports_fp16: bool = True
    supports_uhd: bool = True  # UHD mode for 4K+
    default_precision: str = "fp16"
    
    # Compilation support
    compile_compatible: bool = True  # RIFE supports torch.compile
    preferred_compile_backend: str = "inductor"
    
    # Processing capabilities
    supports_img_mode: bool = True  # Can interpolate between two images
    max_fps_multiplier: int = 8  # Maximum safe FPS multiplication
    
    # Resolution constraints
    max_resolution: int = 0  # 0 = no cap (but UHD mode recommended for 4K+)
    min_resolution: int = 64
    recommended_uhd_threshold: int = 2160  # Enable UHD mode for 4K+
    
    # Quality settings
    recommended_scale: float = 1.0  # Spatial scale (1.0 = no scaling)
    
    # Description
    notes: str = ""


def _get_rife_models() -> List[RifeModel]:
    """
    Define all known RIFE model configurations with comprehensive metadata.
    
    RIFE versions progress with improved temporal consistency and quality.
    Latest versions (v4.15+) offer best results.
    """
    models = [
        # v4.6 - Stable baseline
        RifeModel(
            name="rife-v4.6",
            version="4.6",
            variant="standard",
            estimated_vram_gb=3.0,
            notes="Stable baseline version. Good for general use."
        ),
        
        # v4.13-v4.15 - Progressive improvements
        RifeModel(
            name="rife-v4.13",
            version="4.13",
            variant="standard",
            estimated_vram_gb=3.5,
            notes="Improved temporal consistency over v4.6."
        ),
        RifeModel(
            name="rife-v4.14",
            version="4.14",
            variant="standard",
            estimated_vram_gb=3.5,
            notes="Enhanced motion estimation, fewer artifacts."
        ),
        RifeModel(
            name="rife-v4.15",
            version="4.15",
            variant="standard",
            estimated_vram_gb=4.0,
            notes="Significant quality improvements, recommended for most use cases."
        ),
        
        # v4.16-v4.17 - Latest stable
        RifeModel(
            name="rife-v4.16",
            version="4.16",
            variant="standard",
            estimated_vram_gb=4.0,
            notes="Latest quality improvements with better scene change handling."
        ),
        RifeModel(
            name="rife-v4.17",
            version="4.17",
            variant="standard",
            estimated_vram_gb=4.0,
            max_fps_multiplier=8,
            notes="Current recommended version. Best quality/performance balance."
        ),
        
        # v4.18 - Experimental/latest
        RifeModel(
            name="rife-v4.18",
            version="4.18",
            variant="standard",
            estimated_vram_gb=4.5,
            notes="Latest experimental version. May have cutting-edge features."
        ),
        
        # Anime-specific variant
        RifeModel(
            name="rife-anime",
            version="4.x",
            variant="anime",
            estimated_vram_gb=4.0,
            notes="Specialized for anime content. Better handles hard edges and stylized motion."
        ),
    ]
    
    return models


def get_rife_model_names(base_dir: Path = None) -> List[str]:
    """
    Get available RIFE model names.
    
    Scans RIFE/train_log directory for available models or returns metadata-based list.
    
    Args:
        base_dir: Base directory of the application (to find RIFE/train_log)
    
    Returns:
        List of model names (e.g., "rife-v4.6", "rife-v4.17", "rife-anime")
    """
    # Get models from metadata registry
    metadata_models = [m.name for m in _get_rife_models()]
    
    # Try to discover from train_log directory
    if base_dir:
        rife_dir = Path(base_dir) / "RIFE" / "train_log"
        if rife_dir.exists():
            discovered = []
            for item in rife_dir.iterdir():
                if item.is_dir() and not item.name.startswith("_") and not item.name.startswith("."):
                    # Verify it has model files
                    has_model = any(
                        f.suffix.lower() in (".pkl", ".pth")
                        for f in item.rglob("*")
                        if f.is_file()
                    )
                    if has_model:
                        discovered.append(item.name)
            
            if discovered:
                # Merge discovered with metadata models
                return sorted(list(set(metadata_models + discovered)))
    
    return sorted(metadata_models)


def get_rife_default_model() -> str:
    """Get default RIFE model identifier."""
    return "rife-v4.17"


def get_rife_metadata(model_name: str) -> Optional[RifeModel]:
    """
    Get comprehensive metadata for a RIFE model.
    
    Args:
        model_name: Model identifier (e.g., "rife-v4.17")
        
    Returns:
        RifeModel metadata or default metadata for unknown models
    """
    models_map = {m.name: m for m in _get_rife_models()}
    
    # Return exact match if found
    if model_name in models_map:
        return models_map[model_name]
    
    # Fallback for unknown models - conservative defaults
    return RifeModel(
        name=model_name,
        version="unknown",
        variant="standard",
        estimated_vram_gb=4.0,
        notes="Unknown RIFE model - using conservative defaults"
    )


def rife_model_map() -> Dict[str, RifeModel]:
    """Get mapping of model names to metadata."""
    return {m.name: m for m in _get_rife_models()}

