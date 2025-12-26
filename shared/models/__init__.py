"""
Shared model discovery and metadata functions.
"""

from pathlib import Path
from typing import List


GAN_MODEL_EXTS = (".pth", ".safetensors")


def _iter_gan_model_dirs(base_dir: Path) -> List[Path]:
    """
    Return model directories that may contain GAN / image upscaler weights.

    This project historically used `Image_Upscale_Models/`, but newer layouts store
    weights under `models/`. We support both for backwards compatibility.
    """
    dirs: List[Path] = []
    for folder_name in ("models", "Image_Upscale_Models"):
        d = base_dir / folder_name
        if d.exists() and d.is_dir():
            dirs.append(d)
    return dirs


def scan_gan_models(base_dir: Path) -> List[str]:
    """
    Scan for GAN / image upscaler model weights.
    
    Args:
        base_dir: App base directory
        
    Returns:
        Sorted list of model filenames
    """
    choices: set[str] = set()
    for models_dir in _iter_gan_model_dirs(base_dir):
        try:
            for f in models_dir.iterdir():
                if f.is_file() and f.suffix.lower() in GAN_MODEL_EXTS:
                    choices.add(f.name)
        except Exception:
            # Ignore unreadable dirs; keep scanning others.
            continue
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

