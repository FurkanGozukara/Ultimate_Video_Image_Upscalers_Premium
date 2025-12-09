"""
GPU Detection and Management Utilities

Provides comprehensive GPU detection with:
- GPU names and memory information
- CUDA availability checking
- Multi-GPU configuration validation
- VRAM usage monitoring
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import platform


@dataclass
class GPUInfo:
    """Information about a single GPU"""
    id: int
    name: str
    total_memory_gb: float
    available_memory_gb: float = 0.0
    is_available: bool = True
    compute_capability: Optional[Tuple[int, int]] = None


def get_gpu_info() -> List[GPUInfo]:
    """
    Get comprehensive information about all available GPUs.
    
    Returns:
        List of GPUInfo objects, empty list if CUDA not available
    """
    gpus = []
    
    try:
        import torch
        
        if not torch.cuda.is_available():
            return []
        
        device_count = torch.cuda.device_count()
        
        for i in range(device_count):
            try:
                # Get device properties
                props = torch.cuda.get_device_properties(i)
                name = props.name
                total_memory = props.total_memory / (1024 ** 3)  # Convert to GB
                
                # Get compute capability
                compute_cap = (props.major, props.minor)
                
                # Try to get available memory
                available_memory = 0.0
                try:
                    torch.cuda.set_device(i)
                    available_memory = (torch.cuda.mem_get_info()[0]) / (1024 ** 3)
                except Exception:
                    # If we can't get available memory, just use total
                    available_memory = total_memory
                
                gpu = GPUInfo(
                    id=i,
                    name=name,
                    total_memory_gb=total_memory,
                    available_memory_gb=available_memory,
                    is_available=True,
                    compute_capability=compute_cap
                )
                gpus.append(gpu)
                
            except Exception as e:
                # GPU exists but we couldn't get details
                gpus.append(GPUInfo(
                    id=i,
                    name=f"GPU {i} (details unavailable)",
                    total_memory_gb=0.0,
                    is_available=False
                ))
    
    except ImportError:
        # PyTorch not installed
        return []
    except Exception as e:
        # Other errors
        print(f"Error detecting GPUs: {e}")
        return []
    
    return gpus


def get_gpu_summary() -> str:
    """
    Get a human-readable summary of available GPUs.
    
    Returns:
        Formatted string with GPU information
    """
    gpus = get_gpu_info()
    
    if not gpus:
        return "No CUDA GPUs detected. CPU-only mode."
    
    lines = [f"**Detected {len(gpus)} GPU(s):**\n"]
    
    for gpu in gpus:
        if gpu.is_available:
            status_icon = "✅"
            memory_info = f"{gpu.available_memory_gb:.1f}GB available / {gpu.total_memory_gb:.1f}GB total"
            if gpu.compute_capability:
                cc_info = f" (Compute {gpu.compute_capability[0]}.{gpu.compute_capability[1]})"
            else:
                cc_info = ""
        else:
            status_icon = "⚠️"
            memory_info = "Unavailable"
            cc_info = ""
        
        lines.append(f"{status_icon} **GPU {gpu.id}:** {gpu.name} - {memory_info}{cc_info}")
    
    return "\n".join(lines)


def validate_gpu_ids(gpu_ids: str, allow_multi: bool = True) -> Tuple[bool, str, List[int]]:
    """
    Validate GPU ID string and return parsed IDs.
    
    Args:
        gpu_ids: Comma-separated GPU IDs (e.g., "0" or "0,1,2")
        allow_multi: Whether multi-GPU is allowed
        
    Returns:
        (is_valid, message, parsed_ids)
    """
    if not gpu_ids or not gpu_ids.strip():
        return True, "Using default GPU (0)", [0]
    
    try:
        # Parse IDs
        parsed = []
        for part in gpu_ids.split(","):
            part = part.strip()
            if part:
                gpu_id = int(part)
                if gpu_id < 0:
                    return False, f"Invalid GPU ID: {gpu_id} (must be >= 0)", []
                parsed.append(gpu_id)
        
        if not parsed:
            return True, "Using default GPU (0)", [0]
        
        # Check multi-GPU restriction
        if not allow_multi and len(parsed) > 1:
            return False, "Multi-GPU not supported for this model", []
        
        # Validate against available GPUs
        gpus = get_gpu_info()
        if gpus:
            available_ids = [g.id for g in gpus if g.is_available]
            max_id = max(available_ids) if available_ids else -1
            
            for gpu_id in parsed:
                if gpu_id > max_id:
                    return False, f"GPU {gpu_id} not available. Detected GPUs: 0-{max_id}", []
        
        # All valid
        if len(parsed) == 1:
            msg = f"Using GPU {parsed[0]}"
        else:
            msg = f"Using {len(parsed)} GPUs: {', '.join(map(str, parsed))}"
        
        return True, msg, parsed
        
    except ValueError as e:
        return False, f"Invalid GPU ID format: {gpu_ids}. Use comma-separated integers (e.g., '0,1,2')", []


def get_recommended_gpu() -> int:
    """
    Get the recommended GPU ID based on available memory.
    
    Returns:
        GPU ID with most available memory, or 0 if none detected
    """
    gpus = get_gpu_info()
    
    if not gpus:
        return 0
    
    # Find GPU with most available memory
    best_gpu = max(gpus, key=lambda g: g.available_memory_gb if g.is_available else -1)
    
    return best_gpu.id


def format_gpu_memory(bytes_value: int) -> str:
    """
    Format memory value in human-readable format.
    
    Args:
        bytes_value: Memory in bytes
        
    Returns:
        Formatted string (e.g., "8.5 GB")
    """
    gb = bytes_value / (1024 ** 3)
    if gb >= 1.0:
        return f"{gb:.1f} GB"
    else:
        mb = bytes_value / (1024 ** 2)
        return f"{mb:.0f} MB"


def check_vram_available(required_gb: float) -> Tuple[bool, str]:
    """
    Check if enough VRAM is available on any GPU.
    
    Args:
        required_gb: Required VRAM in gigabytes
        
    Returns:
        (is_available, message)
    """
    gpus = get_gpu_info()
    
    if not gpus:
        return False, f"No GPUs available. {required_gb:.1f}GB VRAM required."
    
    for gpu in gpus:
        if gpu.is_available and gpu.available_memory_gb >= required_gb:
            return True, f"GPU {gpu.id} has {gpu.available_memory_gb:.1f}GB available ({required_gb:.1f}GB required)"
    
    max_available = max((g.available_memory_gb for g in gpus if g.is_available), default=0)
    return False, f"Insufficient VRAM. Required: {required_gb:.1f}GB, Available: {max_available:.1f}GB"


def get_cuda_version() -> Optional[str]:
    """
    Get CUDA version string.
    
    Returns:
        CUDA version or None if not available
    """
    try:
        import torch
        if torch.cuda.is_available():
            return torch.version.cuda
    except Exception:
        pass
    return None


def is_apple_silicon() -> bool:
    """
    Check if running on Apple Silicon (M1/M2/M3).
    
    Returns:
        True if Apple Silicon detected
    """
    if platform.system() != "Darwin":
        return False
    
    try:
        import subprocess
        result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                              capture_output=True, text=True, timeout=1)
        return 'Apple' in result.stdout
    except Exception:
        return False


def get_device_recommendation(model_type: str = "default") -> str:
    """
    Get recommended device configuration based on hardware.
    
    Args:
        model_type: Type of model (affects VRAM requirements)
        
    Returns:
        Recommendation message
    """
    if is_apple_silicon():
        return "🍎 **Apple Silicon detected:** Use MPS backend for GPU acceleration. CUDA not available on macOS."
    
    gpus = get_gpu_info()
    
    if not gpus:
        return "⚠️ **No CUDA GPUs detected:** Processing will use CPU (significantly slower)."
    
    if len(gpus) == 1:
        gpu = gpus[0]
        if gpu.total_memory_gb >= 12:
            return f"✅ **Single GPU:** {gpu.name} with {gpu.total_memory_gb:.0f}GB VRAM - Excellent for all models."
        elif gpu.total_memory_gb >= 8:
            return f"✅ **Single GPU:** {gpu.name} with {gpu.total_memory_gb:.0f}GB VRAM - Good for most models. Use tiling for 4K+."
        else:
            return f"⚠️ **Single GPU:** {gpu.name} with {gpu.total_memory_gb:.0f}GB VRAM - Limited. Use BlockSwap and tiling."
    
    else:
        total_vram = sum(g.total_memory_gb for g in gpus if g.is_available)
        return f"✅ **Multi-GPU:** {len(gpus)} GPUs with {total_vram:.0f}GB total VRAM - Excellent for large batches and streaming."


def clear_cuda_cache():
    """
    Clear CUDA cache to free VRAM.
    Safe to call even if CUDA is not available.
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass

