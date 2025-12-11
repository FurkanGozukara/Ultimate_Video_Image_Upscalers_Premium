"""
Shared model discovery and metadata functions.
"""

from pathlib import Path
from typing import List


def scan_gan_models(base_dir: Path) -> List[str]:
    """
    Scan for GAN models in Image_Upscale_Models folder.
    
    Args:
        base_dir: Base directory containing Image_Upscale_Models folder
        
    Returns:
        Sorted list of model filenames
    """
    models_dir = base_dir / "Image_Upscale_Models"
    if not models_dir.exists():
        return []
    
    choices = []
    for f in models_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".pth", ".safetensors"):
            choices.append(f.name)
    return sorted(choices)


# Re-export model metadata functions for convenience
from .seedvr2_meta import get_seedvr2_model_names, get_seedvr2_models, model_meta_map
from .flashvsr_meta import (
    get_flashvsr_model_names, 
    get_flashvsr_default_model,
    get_flashvsr_metadata,
    flashvsr_model_map
)
from .rife_meta import (
    get_rife_model_names, 
    get_rife_default_model,
    get_rife_metadata,
    rife_model_map
)


__all__ = [
    "scan_gan_models",
    "get_seedvr2_model_names",
    "get_seedvr2_models", 
    "model_meta_map",
    "get_flashvsr_model_names",
    "get_flashvsr_default_model",
    "get_flashvsr_metadata",
    "flashvsr_model_map",
    "get_rife_model_names",
    "get_rife_default_model",
    "get_rife_metadata",
    "rife_model_map"
]

