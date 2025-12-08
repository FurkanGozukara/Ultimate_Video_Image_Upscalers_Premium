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


__all__ = ["scan_gan_models"]

