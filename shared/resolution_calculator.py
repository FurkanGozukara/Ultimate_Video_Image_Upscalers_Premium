"""
Automatic Resolution Calculator with Aspect Ratio Awareness

Handles intelligent resolution calculation for upscaling with:
- Aspect ratio preservation
- Max target resolution constraints
- Downscale-then-upscale for fixed-scale models (GAN)
- Per-model resolution caching
- Smart rounding to model-compatible dimensions
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from .path_utils import get_media_dimensions, detect_input_type


@dataclass
class ResolutionConfig:
    """Configuration for resolution calculation"""
    input_width: int
    input_height: int
    target_resolution: int  # Target for shortest side
    max_resolution: int  # Maximum allowed (0 = unlimited)
    model_scale: Optional[int] = None  # For GAN models (2x, 4x, etc.)
    enable_max_target: bool = True
    auto_resolution: bool = True
    ratio_aware: bool = True
    allow_downscale: bool = True  # For GAN models to reach target
    round_to_multiple: int = 16  # Round to multiple of this (e.g., 16 for SeedVR2)


@dataclass
class ResolutionResult:
    """Result of resolution calculation"""
    output_width: int
    output_height: int
    input_resize_width: Optional[int] = None  # If downscaling needed first
    input_resize_height: Optional[int] = None
    effective_scale: float = 1.0
    needs_downscale_first: bool = False
    clamped_by_max: bool = False
    aspect_ratio: float = 1.0
    info_message: str = ""


def calculate_resolution(
    input_path: str,
    config: ResolutionConfig
) -> ResolutionResult:
    """
    Calculate optimal output resolution with aspect ratio awareness.
    
    Args:
        input_path: Path to input video or image
        config: Resolution configuration
        
    Returns:
        ResolutionResult with calculated dimensions and metadata
    """
    # Get input dimensions if not provided
    if not config.input_width or not config.input_height:
        try:
            width, height = get_media_dimensions(input_path)
            if not width or not height:
                raise ValueError("Could not determine input dimensions")
            config.input_width = width
            config.input_height = height
        except Exception as e:
            raise ValueError(f"Failed to get input dimensions: {e}")
    
    aspect_ratio = config.input_width / config.input_height
    short_side = min(config.input_width, config.input_height)
    long_side = max(config.input_width, config.input_height)
    is_landscape = config.input_width > config.input_height
    
    # Determine target based on settings
    if not config.auto_resolution:
        # Manual mode: use target_resolution directly
        target_short = config.target_resolution
    else:
        # Auto mode: calculate based on input and constraints
        target_short = config.target_resolution
    
    # Calculate long side maintaining aspect ratio
    if config.ratio_aware:
        target_long = int(target_short * (long_side / short_side))
    else:
        target_long = target_short
    
    # Apply max resolution constraint
    clamped = False
    if config.enable_max_target and config.max_resolution > 0:
        if max(target_short, target_long) > config.max_resolution:
            # Scale down to fit within max
            scale_factor = config.max_resolution / max(target_short, target_long)
            target_short = int(target_short * scale_factor)
            target_long = int(target_long * scale_factor)
            clamped = True
    
    # Round to compatible multiple
    target_short = round_to_multiple(target_short, config.round_to_multiple)
    target_long = round_to_multiple(target_long, config.round_to_multiple)
    
    # Set output dimensions based on orientation
    if is_landscape:
        output_width = target_long
        output_height = target_short
    else:
        output_width = target_short
        output_height = target_long
    
    # Calculate effective scale
    effective_scale = target_short / short_side
    
    # Handle fixed-scale models (GAN)
    input_resize_width = None
    input_resize_height = None
    needs_downscale = False
    
    if config.model_scale is not None and config.model_scale > 1:
        # GAN model with fixed scale
        # Calculate what input resolution we need to reach target with this scale
        required_input_short = target_short / config.model_scale
        required_input_long = target_long / config.model_scale
        
        required_input_short = round_to_multiple(int(required_input_short), config.round_to_multiple)
        required_input_long = round_to_multiple(int(required_input_long), config.round_to_multiple)
        
        # Check if we need to downscale input first
        if config.allow_downscale and (short_side > required_input_short):
            needs_downscale = True
            if is_landscape:
                input_resize_width = required_input_long
                input_resize_height = required_input_short
            else:
                input_resize_width = required_input_short
                input_resize_height = required_input_long
            
            # Recalculate output based on actual model scale
            output_width = input_resize_width * config.model_scale
            output_height = input_resize_height * config.model_scale
        else:
            # Just upscale directly with model scale
            output_width = round_to_multiple(config.input_width * config.model_scale, config.round_to_multiple)
            output_height = round_to_multiple(config.input_height * config.model_scale, config.round_to_multiple)
    
    # Generate info message
    info_parts = []
    info_parts.append(f"Input: {config.input_width}x{config.input_height}")
    
    if needs_downscale:
        info_parts.append(f"→ Downscale to: {input_resize_width}x{input_resize_height}")
        info_parts.append(f"→ {config.model_scale}x upscale")
    
    info_parts.append(f"Output: {output_width}x{output_height}")
    info_parts.append(f"(Effective: {effective_scale:.2f}x)")
    
    if clamped:
        info_parts.append(f"⚠️ Clamped by max resolution: {config.max_resolution}")
    
    info_message = " ".join(info_parts)
    
    return ResolutionResult(
        output_width=output_width,
        output_height=output_height,
        input_resize_width=input_resize_width,
        input_resize_height=input_resize_height,
        effective_scale=effective_scale,
        needs_downscale_first=needs_downscale,
        clamped_by_max=clamped,
        aspect_ratio=aspect_ratio,
        info_message=info_message
    )


def round_to_multiple(value: int, multiple: int) -> int:
    """Round value to nearest multiple"""
    return int(round(value / multiple) * multiple)


def calculate_chunk_count(
    video_path: str,
    chunk_size_sec: float,
    min_scene_len: float = 2.0
) -> Tuple[int, float, str]:
    """
    Estimate number of chunks for scene-based splitting.
    
    Returns:
        (estimated_chunks, total_duration, info_message)
    """
    from .path_utils import get_media_duration_seconds, get_media_fps
    
    try:
        duration = get_media_duration_seconds(video_path)
        fps = get_media_fps(video_path)
        
        if not duration or duration <= 0:
            return 0, 0, "⚠️ Could not determine video duration"
        
        # Simple estimation: total_duration / chunk_size
        if chunk_size_sec > 0:
            estimated_chunks = max(1, int(math.ceil(duration / chunk_size_sec)))
        else:
            # Scene-based detection: estimate conservatively
            # Assume average scene length is ~10 seconds for typical content
            avg_scene_length = 10.0
            estimated_chunks = max(1, int(math.ceil(duration / avg_scene_length)))
        
        # Format info message
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = int(duration % 60)
        
        if hours > 0:
            duration_str = f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = f"{seconds}s"
        
        fps_str = f"{fps:.2f} FPS" if fps else "unknown FPS"
        
        info = f"📊 Video: {duration_str} ({fps_str})\n"
        info += f"📦 Estimated chunks: {estimated_chunks}"
        
        if chunk_size_sec > 0:
            info += f" (~{chunk_size_sec:.0f}s each)"
        else:
            info += " (scene-based detection)"
        
        return estimated_chunks, duration, info
        
    except Exception as e:
        return 0, 0, f"⚠️ Error estimating chunks: {e}"


def calculate_disk_space_required(
    input_path: str,
    resolution_result: ResolutionResult,
    output_format: str = "mp4",
    duration: Optional[float] = None,
    safety_multiplier: float = 2.0
) -> Tuple[int, str]:
    """
    Estimate required disk space for output.
    
    Returns:
        (bytes_required, human_readable_string)
    """
    from .path_utils import get_media_duration_seconds, detect_input_type
    import os
    
    try:
        input_type = detect_input_type(input_path)
        
        if input_type == "video":
            if duration is None:
                duration = get_media_duration_seconds(input_path) or 10.0
            
            # Estimate bitrate based on resolution
            pixels = resolution_result.output_width * resolution_result.output_height
            
            # Very rough estimation (can vary wildly)
            # 1080p ~= 8-10 Mbps, 4K ~= 20-40 Mbps
            if pixels < 1920 * 1080:
                bitrate_mbps = 5
            elif pixels < 3840 * 2160:
                bitrate_mbps = 12
            else:
                bitrate_mbps = 30
            
            # Calculate file size
            bytes_required = int((bitrate_mbps * 1000000 / 8) * duration * safety_multiplier)
            
        else:  # Image
            # Rough estimate: uncompressed size * compression factor
            pixels = resolution_result.output_width * resolution_result.output_height
            bytes_per_pixel = 3  # RGB
            
            if output_format == "png":
                compression = 0.7  # PNG is lossless but compressed
            else:  # jpg, webp
                compression = 0.2  # Lossy compression
            
            bytes_required = int(pixels * bytes_per_pixel * compression * safety_multiplier)
        
        # Format human-readable
        if bytes_required < 1024:
            size_str = f"{bytes_required} B"
        elif bytes_required < 1024 ** 2:
            size_str = f"{bytes_required / 1024:.1f} KB"
        elif bytes_required < 1024 ** 3:
            size_str = f"{bytes_required / (1024 ** 2):.1f} MB"
        else:
            size_str = f"{bytes_required / (1024 ** 3):.2f} GB"
        
        return bytes_required, size_str
        
    except Exception as e:
        return 0, f"Unknown (error: {e})"


def get_available_disk_space(path: str) -> Tuple[int, str]:
    """
    Get available disk space at given path.
    
    Returns:
        (bytes_available, human_readable_string)
    """
    import shutil
    
    try:
        stat = shutil.disk_usage(path)
        available = stat.free
        
        # Format human-readable
        if available < 1024 ** 3:
            size_str = f"{available / (1024 ** 2):.0f} MB"
        else:
            size_str = f"{available / (1024 ** 3):.1f} GB"
        
        return available, size_str
        
    except Exception as e:
        return 0, f"Unknown (error: {e})"


# Resolution presets for common targets
RESOLUTION_PRESETS = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
    "1440p": 1440,
    "4K": 2160,
    "8K": 4320,
}

