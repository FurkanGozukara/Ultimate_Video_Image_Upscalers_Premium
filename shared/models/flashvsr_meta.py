"""
FlashVSR+ Model Metadata Registry
Provides model names and metadata for FlashVSR+ models
"""

from typing import List


def get_flashvsr_model_names() -> List[str]:
    """
    Get available FlashVSR+ model identifiers.
    
    FlashVSR+ models are identified by version + mode combination:
    - Versions: v10, v11
    - Modes: tiny, tiny-long, full
    - Scales: 2x, 4x
    
    Returns:
        List of model identifier strings (e.g., "v10_tiny_2x", "v11_full_4x")
    """
    versions = ["10", "11"]
    modes = ["tiny", "tiny-long", "full"]
    scales = ["2", "4"]
    
    model_names = []
    for version in versions:
        for mode in modes:
            for scale in scales:
                model_names.append(f"v{version}_{mode}_{scale}x")
    
    return sorted(model_names)


def get_flashvsr_default_model() -> str:
    """Get default FlashVSR+ model identifier."""
    return "v10_tiny_4x"

