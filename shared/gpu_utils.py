"""
GPU utility functions shared across all model services.

Provides consistent CUDA device handling for SeedVR2, GAN, RIFE, FlashVSR+.
"""
from typing import Optional


def expand_cuda_device_spec(cuda_spec: str) -> str:
    """
    Expand 'all' to actual device list.
    
    Supports:
    - "0" -> "0" (single GPU)
    - "0,1,2" -> "0,1,2" (explicit multi-GPU)
    - "all" -> "0,1,2,3" (auto-detect all GPUs)
    - Whitespace-tolerant: "0, 1, 2" -> "0,1,2"
    
    Args:
        cuda_spec: User-provided CUDA device specification
        
    Returns:
        Normalized device specification (comma-separated GPU IDs)
    """
    try:
        import torch
        
        spec_clean = str(cuda_spec).strip().lower()
        
        # Handle "all" keyword
        if spec_clean == "all" and torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            return ",".join(str(i) for i in range(device_count))
        
        # Normalize whitespace in device lists
        # "0, 1, 2" -> "0,1,2"
        devices = [d.strip() for d in str(cuda_spec).replace(" ", "").split(",") if d.strip()]
        return ",".join(devices)
        
    except Exception:
        # If torch unavailable or error, return as-is
        return cuda_spec


def validate_cuda_device_spec(cuda_spec: str) -> Optional[str]:
    """
    Validate CUDA device specification.
    
    Checks:
    - CUDA availability
    - Device ID validity (numeric and within range)
    - Multi-GPU compatibility warnings
    
    Args:
        cuda_spec: Device specification (already expanded via expand_cuda_device_spec)
        
    Returns:
        Error message if invalid, None if valid
    """
    try:
        import torch
        
        if not cuda_spec or cuda_spec.strip() == "":
            return None  # Empty is valid (CPU fallback)
        
        if not torch.cuda.is_available():
            return "CUDA is not available on this system, but CUDA devices were specified."
        
        # Parse device list
        devices = [d.strip() for d in str(cuda_spec).split(",") if d.strip()]
        device_count = torch.cuda.device_count()
        
        # Validate each device ID
        invalid = []
        for d in devices:
            if not d.isdigit():
                invalid.append(f"{d} (not numeric)")
            elif int(d) >= device_count:
                invalid.append(f"{d} (max: {device_count-1})")
        
        if invalid:
            return f"Invalid CUDA device ID(s): {', '.join(invalid)}. Available devices: 0-{device_count-1}"
        
        return None  # Valid
        
    except Exception as exc:
        return f"CUDA validation error: {exc}"


def clear_cuda_cache():
    """
    Clear CUDA cache to free VRAM.
    
    Safe to call even if CUDA unavailable (silently ignored).
    Used after subprocess completion or model switches.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for GPU operations to finish
    except Exception:
        pass  # Silently ignore if CUDA not available


def get_cuda_memory_info() -> str:
    """
    Get human-readable CUDA memory usage summary.
    
    Returns:
        Formatted string with memory stats for each GPU, or empty if unavailable
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return "CUDA not available"
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return "No CUDA devices found"
        
        lines = []
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
            reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
            total = props.total_memory / (1024 ** 3)  # GB
            
            lines.append(
                f"GPU {i} ({props.name}): "
                f"{allocated:.2f}GB allocated, "
                f"{reserved:.2f}GB reserved, "
                f"{total:.2f}GB total"
            )
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Memory info unavailable: {e}"


def check_gpu_compatibility(model_type: str) -> tuple[bool, str]:
    """
    Check if GPU acceleration is available and compatible.
    
    Args:
        model_type: Model type for specific warnings (seedvr2, gan, rife, flashvsr)
        
    Returns:
        (is_compatible, warning_message) tuple
    """
    try:
        import torch
        
        if not torch.cuda.is_available():
            return False, "CUDA not available - processing will use CPU (significantly slower)"
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False, "No CUDA devices detected - processing will use CPU"
        
        # Check compute capability for model-specific warnings
        props = torch.cuda.get_device_properties(0)
        compute_cap = (props.major, props.minor)
        
        warnings = []
        
        # Flash attention requires Ampere+ (8.0) for optimal performance
        if model_type in ("seedvr2", "flashvsr") and compute_cap[0] < 8:
            warnings.append(
                f"GPU compute capability {compute_cap[0]}.{compute_cap[1]} detected. "
                f"Flash attention works best on Ampere (8.0+) or newer GPUs."
            )
        
        # torch.compile requires recent GPU
        if compute_cap[0] < 7:
            warnings.append(
                f"GPU compute capability {compute_cap[0]}.{compute_cap[1]} is very old. "
                f"torch.compile may not work properly. Consider upgrading GPU."
            )
        
        if warnings:
            return True, "\n".join(warnings)
        
        return True, f"✅ {device_count} CUDA GPU(s) detected and compatible"
        
    except Exception as e:
        return False, f"GPU check failed: {e}"
