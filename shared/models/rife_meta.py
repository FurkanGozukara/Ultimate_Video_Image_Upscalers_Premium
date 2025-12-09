"""
RIFE Model Metadata Registry
Provides model names and metadata for RIFE frame interpolation models
"""

from pathlib import Path
from typing import List


def get_rife_model_names(base_dir: Path = None) -> List[str]:
    """
    Get available RIFE model names.
    
    Scans RIFE/train_log directory for available models or returns default list.
    
    Args:
        base_dir: Base directory of the application (to find RIFE/train_log)
    
    Returns:
        List of model names (e.g., "rife-v4.6", "rife-v4.17", "rife-anime")
    """
    # Default known models
    default_models = [
        "rife-v4.6",
        "rife-v4.13", 
        "rife-v4.14",
        "rife-v4.15",
        "rife-v4.16",
        "rife-v4.17",
        "rife-v4.18",
        "rife-anime"
    ]
    
    # Try to discover from train_log directory
    if base_dir:
        rife_dir = Path(base_dir) / "RIFE" / "train_log"
        if rife_dir.exists():
            discovered = []
            for item in rife_dir.iterdir():
                if item.is_dir() and not item.name.startswith("_") and not item.name.startswith("."):
                    discovered.append(item.name)
            
            if discovered:
                return sorted(discovered)
    
    return default_models


def get_rife_default_model() -> str:
    """Get default RIFE model identifier."""
    return "rife-v4.17"

