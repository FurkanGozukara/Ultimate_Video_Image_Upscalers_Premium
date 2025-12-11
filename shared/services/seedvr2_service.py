"""
SeedVR2 Service Module - Complete Rewrite
Handles all SeedVR2 processing logic, presets, and callbacks
"""

import shutil
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.preset_auto_serializer import (
    auto_detect_inputs,
    serialize_component_values,
    deserialize_to_component_values,
    create_auto_order
)
from shared.runner import Runner, RunResult
from shared.path_utils import (
    normalize_path,
    ffmpeg_set_fps,
    get_media_dimensions,
    get_media_duration_seconds,
    detect_input_type,
)
from shared.chunking import chunk_and_process, check_resume_available
from shared.face_restore import restore_image, restore_video
from shared.models.seedvr2_meta import get_seedvr2_model_names, model_meta_map
from shared.logging_utils import RunLogger
from shared.comparison_unified import create_unified_comparison, build_comparison_selector
from shared.video_comparison import create_comparison_selector
from shared.model_manager import get_model_manager, ModelType
from shared.error_handling import (
    validate_input_path,
    validate_cuda_device as validate_cuda_spec,
    validate_batch_size,
    check_ffmpeg_available,
    check_disk_space,
    safe_execute,
    logger as error_logger,
)

# Constants --------------------------------------------------------------------
SEEDVR2_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
SEEDVR2_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


# Defaults and ordering --------------------------------------------------------
def _get_default_attention_mode() -> str:
    """
    Get default attention mode with actual runtime testing.
    
    DEFAULT PRIORITY: flash_attn > sdpa
    
    Attempts to use flash_attn if available and working, otherwise falls back to sdpa.
    This is more robust than just checking import - actually tests CUDA compatibility.
    
    Returns:
        "flash_attn" if available and CUDA compatible
        "sdpa" as fallback (always available in PyTorch 2.0+)
    """
    try:
        import flash_attn
        import torch
        
        # Check if CUDA is available (flash_attn requires CUDA)
        if not torch.cuda.is_available():
            return "sdpa"
        
        # Test flash_attn compatibility with actual CUDA device
        try:
            # Verify flash_attn has minimum required version
            version = getattr(flash_attn, '__version__', '')
            if not version:
                # No version attribute, try import test
                from flash_attn import flash_attn_func
                
            # Check GPU compute capability (flash_attn requires 8.0+ for optimal performance)
            # But will work on 7.5+ (Turing) with reduced performance
            device_props = torch.cuda.get_device_properties(0)
            compute_cap = (device_props.major, device_props.minor)
            
            # Flash attention works best on Ampere (8.0+) and Ada/Hopper (8.9+, 9.0+)
            # But also functional on Turing (7.5)
            if compute_cap[0] >= 7 and compute_cap[1] >= 5:
                # GPU supports flash attention
                return "flash_attn"
            else:
                # Older GPU, use sdpa
                return "sdpa"
                
        except (AttributeError, ImportError, RuntimeError):
            # Flash attention import succeeded but functionality test failed
            return "sdpa"
            
    except ImportError:
        # Flash attention not installed
        return "sdpa"
    except Exception:
        # Any other error, fall back safely to sdpa
        return "sdpa"


def seedvr2_defaults(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default SeedVR2 settings aligned with CLI defaults.
    Applies model-specific metadata when model_name is provided.
    """
    try:
        import torch
        cuda_default = "0" if torch.cuda.is_available() else ""
    except Exception:
        cuda_default = ""
    
    # Get model metadata if specific model is provided
    model_meta = None
    default_model = get_seedvr2_model_names()[0] if get_seedvr2_model_names() else "seedvr2_ema_3b_fp16.safetensors"
    target_model = model_name or default_model
    
    if target_model:
        from shared.models.seedvr2_meta import model_meta_map
        meta_map = model_meta_map()
        model_meta = meta_map.get(target_model)
    
    # Apply model-specific defaults if metadata available
    if model_meta:
        default_attention = model_meta.preferred_attention if hasattr(model_meta, 'preferred_attention') else _get_default_attention_mode()
        default_batch_size = model_meta.default_batch_size if hasattr(model_meta, 'default_batch_size') else 5
        compile_compatible = model_meta.compile_compatible if hasattr(model_meta, 'compile_compatible') else True
        max_res_cap = model_meta.max_resolution if hasattr(model_meta, 'max_resolution') else 0
    else:
        default_attention = _get_default_attention_mode()
        default_batch_size = 5
        compile_compatible = True
        max_res_cap = 0
    
    return {
        "input_path": "",
        "output_override": "",
        "output_format": "auto",
        "model_dir": "",
        "dit_model": target_model,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        # PySceneDetect chunking removed - now configured in Resolution tab
        # Legacy chunk_enable, scene_threshold, scene_min_len removed from ORDER
        "chunk_size": 0,  # SeedVR2 native chunking (frames per chunk, 0=disabled)
        "resolution": 1080,
        "max_resolution": max_res_cap,  # Apply model-specific cap
        "batch_size": default_batch_size,  # Apply model-specific default
        "uniform_batch_size": False,
        "seed": 42,
        "skip_first_frames": 0,
        "load_cap": 0,
        "prepend_frames": 0,
        "temporal_overlap": 0,
        "color_correction": "lab",
        "input_noise_scale": 0.0,
        "latent_noise_scale": 0.0,
        "cuda_device": cuda_default,
        "dit_offload_device": "none",
        "vae_offload_device": "none",
        "tensor_offload_device": "cpu",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "vae_encode_tiled": False,
        "vae_encode_tile_size": 1024,
        "vae_encode_tile_overlap": 128,
        "vae_decode_tiled": False,
        "vae_decode_tile_size": 1024,
        "vae_decode_tile_overlap": 128,
        "tile_debug": "false",
        "attention_mode": default_attention,  # Apply model-specific attention
        "compile_dit": False,  # Default off, will validate against model metadata
        "compile_vae": False,  # Default off, will validate against model metadata
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_fullgraph": False,
        "compile_dynamic": False,
        "compile_dynamo_cache_size_limit": 64,
        "compile_dynamo_recompile_limit": 128,
        "cache_dit": False,
        "cache_vae": False,
        "debug": False,
        "resume_chunking": False,
        "save_metadata": True,  # Per-run metadata toggle (respects global telemetry)
        "fps_override": 0,  # FPS override (0 = use source FPS, >0 = set specific FPS)
        "_compile_compatible": compile_compatible,  # Store for validation
    }


"""
📋 PRESET SERIALIZATION ORDER

This list defines the order of parameters for preset save/load.
MUST match inputs_list order in ui/seedvr2_tab.py.

⚠️ CURRENT LIMITATION - MANUAL SYNCHRONIZATION REQUIRED:
Adding a new control requires manual updates across 3 files:
1. Add default value to seedvr2_defaults() function (this file)
2. Append key name to SEEDVR2_ORDER below (this file, at end for backward compat)
3. Add Gradio component to ui/seedvr2_tab.py inputs_list (same position as SEEDVR2_ORDER)

✅ BACKWARD COMPATIBILITY:
Old presets auto-merge: missing keys use current defaults (no migration needed!)
The PresetManager.merge_config() preserves existing values for keys not in the preset.

⚠️ VALIDATION:
Runtime validation occurs in save_preset() callback - warns if len(inputs_list) != len(SEEDVR2_ORDER).
This catches integration bugs early but requires manual fixing.

💡 FUTURE IMPROVEMENT OPTIONS:
Option A: Auto-serialization from Gradio components (preset_auto_serializer module exists but unused)
  - Scan inputs_list component IDs/labels and auto-generate order
  - Eliminates manual SEEDVR2_ORDER maintenance
  - Requires component naming conventions

Option B: Schema-driven approach with dataclasses
  - Define settings schema as @dataclass
  - Auto-generate UI components from schema
  - Single source of truth for all tabs
  - Breaking change - requires refactoring all tabs

Option C: Keep current manual approach
  - Proven stable and predictable
  - Easy to debug when things break
  - Explicit control over serialization order
  - Trade-off: requires discipline to maintain sync

RECOMMENDATION: Option C (current manual approach) is acceptable for now given:
- Auto-merge backward compatibility works well
- Save callback validates sync at runtime
- Only ~50 controls per tab (manageable)
- Refactoring to A or B is significant breaking change

To validate sync, check: len(inputs_list) == len(SEEDVR2_ORDER)
"""

SEEDVR2_ORDER: List[str] = [
    # Input/Output
    "input_path",
    "output_override",
    "output_format",
    "model_dir",
    "dit_model",
    # Batch processing (internal, not exposed as batch UI in SeedVR2 - use directory input instead)
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    # PySceneDetect chunking - REMOVED LEGACY CONTROLS (use Resolution tab instead)
    # Former: "chunk_enable", "scene_threshold", "scene_min_len" - now managed by Resolution tab
    # SeedVR2 native streaming (CLI --chunk_size, works WITH PySceneDetect)
    "chunk_size",
    # Resolution and processing
    "resolution",
    "max_resolution",
    "batch_size",
    "uniform_batch_size",
    "seed",
    "skip_first_frames",
    "load_cap",
    "prepend_frames",
    "temporal_overlap",
    # Quality controls
    "color_correction",
    "input_noise_scale",
    "latent_noise_scale",
    # Device configuration
    "cuda_device",
    "dit_offload_device",
    "vae_offload_device",
    "tensor_offload_device",
    # BlockSwap memory optimization
    "blocks_to_swap",
    "swap_io_components",
    # VAE tiling
    "vae_encode_tiled",
    "vae_encode_tile_size",
    "vae_encode_tile_overlap",
    "vae_decode_tiled",
    "vae_decode_tile_size",
    "vae_decode_tile_overlap",
    "tile_debug",
    # Performance and compilation
    "attention_mode",
    "compile_dit",
    "compile_vae",
    "compile_backend",
    "compile_mode",
    "compile_fullgraph",
    "compile_dynamic",
    "compile_dynamo_cache_size_limit",
    "compile_dynamo_recompile_limit",
    # Model caching (batch processing optimization)
    "cache_dit",
    "cache_vae",
    # Debug and resume
    "debug",
    "resume_chunking",
    # Output & shared settings (from Output tab integration)
    "save_metadata",
    "fps_override",
]


# Guardrails -------------------------------------------------------------------
def _enforce_seedvr2_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply SeedVR2-specific validation rules and apply resolution tab settings if available."""
    cfg = cfg.copy()
    
    # Apply model-specific metadata constraints
    model_name = cfg.get("dit_model", "")
    if model_name:
        from shared.models.seedvr2_meta import model_meta_map
        meta_map = model_meta_map()
        model_meta = meta_map.get(model_name)
        
        if model_meta:
            # Disable compile if model doesn't support it (e.g., GGUF quantized models)
            compile_compatible = getattr(model_meta, 'compile_compatible', True)
            if not compile_compatible:
                if cfg.get("compile_dit") or cfg.get("compile_vae"):
                    error_logger.warning(f"Model {model_name} doesn't support torch.compile - disabling compile flags")
                    cfg["compile_dit"] = False
                    cfg["compile_vae"] = False
                    cfg["_compile_disabled_reason"] = f"Model {model_name} is not compile-compatible"
            
            # Enforce multi-GPU support check
            supports_multi_gpu = getattr(model_meta, 'supports_multi_gpu', True)
            cuda_device_str = str(cfg.get("cuda_device", ""))
            devices = [d.strip() for d in cuda_device_str.replace(" ", "").split(",") if d.strip() and d.strip().isdigit()]
            
            if not supports_multi_gpu and len(devices) > 1:
                error_logger.warning(f"Model {model_name} doesn't support multi-GPU - forcing single GPU (using first: {devices[0]})")
                cfg["cuda_device"] = devices[0]
                cfg["_multi_gpu_disabled_reason"] = f"Model {model_name} doesn't support multi-GPU"
            
            # Apply model max resolution cap if set
            model_max_res = getattr(model_meta, 'max_resolution', 0)
            if model_max_res > 0:
                current_res = cfg.get("resolution", 1080)
                if current_res > model_max_res:
                    error_logger.warning(f"Model {model_name} max resolution is {model_max_res}, clamping from {current_res}")
                    cfg["resolution"] = model_max_res
                    cfg["_resolution_clamped_reason"] = f"Model max resolution: {model_max_res}"
            
            # Set preferred attention mode if not explicitly set by user
            preferred_attention = getattr(model_meta, 'preferred_attention', None)
            if preferred_attention and cfg.get("attention_mode") == _get_default_attention_mode():
                cfg["attention_mode"] = preferred_attention

    # Apply resolution tab settings from shared state if available
    if state:
        seed_controls = state.get("seed_controls", {})
        
        # Only apply if values are set in resolution tab
        if "resolution_val" in seed_controls and seed_controls["resolution_val"]:
            cfg["resolution"] = int(seed_controls["resolution_val"])
        if "max_resolution_val" in seed_controls and seed_controls["max_resolution_val"]:
            cfg["max_resolution"] = int(seed_controls["max_resolution_val"])
        # PySceneDetect chunking now ONLY controlled by Resolution tab
        # No more chunk_enable - chunking triggers automatically when chunk_size_sec > 0
        if "chunk_overlap_sec" in seed_controls:
            cfg["chunk_overlap"] = float(seed_controls["chunk_overlap_sec"])

    # Batch size must be 4n+1 using centralized validation
    bs = int(cfg.get("batch_size", defaults["batch_size"]))
    is_valid, error_msg = validate_batch_size(bs, must_be_4n_plus_1=True)
    if not is_valid:
        error_logger.warning(f"Invalid batch size {bs}, correcting: {error_msg}")
        cfg["batch_size"] = max(5, (bs // 4) * 4 + 1)

    # VAE tiling constraints
    if cfg.get("vae_encode_tiled"):
        tile_size = cfg.get("vae_encode_tile_size", defaults["vae_encode_tile_size"])
        overlap = cfg.get("vae_encode_tile_overlap", 0)
        if overlap >= tile_size:
            cfg["vae_encode_tile_overlap"] = max(0, tile_size - 1)

    if cfg.get("vae_decode_tiled"):
        tile_size = cfg.get("vae_decode_tile_size", defaults["vae_decode_tile_size"])
        overlap = cfg.get("vae_decode_tile_overlap", 0)
        if overlap >= tile_size:
            cfg["vae_decode_tile_overlap"] = max(0, tile_size - 1)

    # BlockSwap requires dit_offload_device
    blockswap_enabled = cfg.get("blocks_to_swap", 0) > 0 or cfg.get("swap_io_components", False)
    if blockswap_enabled and str(cfg.get("dit_offload_device", "none")).lower() in ("none", ""):
        cfg["dit_offload_device"] = "cpu"

    # Multi-GPU constraints
    devices = [d.strip() for d in str(cfg.get("cuda_device", "")).split(",") if d.strip()]
    if len(devices) > 1:
        if cfg.get("cache_dit"):
            cfg["cache_dit"] = False
        if cfg.get("cache_vae"):
            cfg["cache_vae"] = False

    return cfg


# Helper functions -------------------------------------------------------------
def _validate_cuda_devices(cuda_spec: str) -> Optional[str]:
    """Validate CUDA device specification using centralized error handling."""
    is_valid, error_msg = validate_cuda_spec(cuda_spec)
    return error_msg if not is_valid else None


def _expand_cuda_spec(cuda_spec: str) -> str:
    """Expand 'all' to actual device list."""
    try:
        import torch
        
        if str(cuda_spec).strip().lower() == "all" and torch.cuda.is_available():
            return ",".join(str(i) for i in range(torch.cuda.device_count()))
    except Exception:
        pass
    return cuda_spec


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH using centralized error handling."""
    is_available, _ = check_ffmpeg_available()
    return is_available


def _resolve_input_path(file_upload: Optional[str], manual_path: str, batch_enable: bool, batch_input: str) -> Tuple[str, Optional[str]]:
    """
    Resolve the input path from various sources with priority order:
    1. Batch input (if batch enabled)
    2. File upload (highest priority for single files)
    3. Manual path entry (fallback)
    
    Returns:
        Tuple of (input_path, original_filename)
        - input_path: Actual file path to process
        - original_filename: User's original filename (if from upload), None otherwise
    """
    if batch_enable and batch_input and batch_input.strip():
        return batch_input.strip(), None
    
    # Check if file_upload is a FileData dict or path string
    original_filename = None
    if file_upload:
        # Gradio can pass FileData as dict with orig_name
        if isinstance(file_upload, dict) and 'orig_name' in file_upload:
            original_filename = file_upload.get('orig_name')
            input_path = file_upload.get('path', '')
        else:
            input_path = str(file_upload).strip()
        
        if input_path:
            return input_path, original_filename
    
    return manual_path.strip() if manual_path else "", None


def _list_media_files(folder: str, video_exts: set, image_exts: set) -> List[str]:
    """List media files in a folder."""
    try:
        p = Path(normalize_path(folder))
        if not p.exists() or not p.is_dir():
            return []
        items = []
        for f in sorted(p.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in video_exts or ext in image_exts:
                items.append(str(f))
        return items
    except Exception:
        return []


# Preset helpers ---------------------------------------------------------------
def _seedvr2_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    """
    Convert argument list to settings dictionary.
    
    Uses SEEDVR2_ORDER for backward compatibility, but could be auto-detected
    using auto_detect_inputs() from preset_auto_serializer for future-proofing.
    
    To add a new control:
    1. Add to seedvr2_defaults() with default value
    2. Add to SEEDVR2_ORDER list (append at end for backward compat)
    3. Add to inputs_list in seedvr2_tab.py (same order as SEEDVR2_ORDER)
    4. That's it! Auto-merging ensures old presets still work.
    """
    return dict(zip(SEEDVR2_ORDER, args))


def _validate_preset_completeness(components: List[gr.components.Component]) -> None:
    """
    Validate that components list matches SEEDVR2_ORDER length.
    Helps catch mismatches during development.
    
    This is a development-time helper that ensures inputs_list and SEEDVR2_ORDER
    stay in sync. Remove or disable in production.
    """
    if len(components) != len(SEEDVR2_ORDER):
        error_msg = (
            f"⚠️ PRESET MISMATCH: inputs_list has {len(components)} components "
            f"but SEEDVR2_ORDER has {len(SEEDVR2_ORDER)} keys.\n"
            f"This will cause preset save/load errors.\n"
            f"Expected keys: {SEEDVR2_ORDER}\n"
            f"Component count: {len(components)}"
        )
        error_logger.warning(error_msg)
        # Don't raise - just warn, as this is recoverable


def _apply_preset_to_values(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Apply preset values to current settings."""
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    merged = _enforce_seedvr2_guardrails(merged, defaults, state=None)  # No state during preset load
    return [merged[key] for key in SEEDVR2_ORDER]


# Core processing functions -----------------------------------------------------
def _process_single_file(
    runner: Runner,
    settings: Dict[str, Any],
    global_settings: Dict[str, Any],
    seed_controls: Dict[str, Any],
    face_apply: bool,
    face_strength: float,
    run_logger: RunLogger,
    output_dir: Path,
    preview_only: bool = False,
    progress_cb: Optional[Callable[[str], None]] = None
) -> Tuple[str, str, Optional[str], Optional[str], str, str, str]:
    """
    Process a single file with SeedVR2.
    Returns: (status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress)
    """
    local_logs = []
    output_video = None
    output_image = None
    chunk_info_msg = "No chunking performed."
    chunk_summary = "Single pass (no chunking)."
    chunk_progress_msg = ""
    status = "⚠️ Processing exited unexpectedly"

    try:
        # Handle first-frame preview mode
        if preview_only:
            input_type = detect_input_type(settings["input_path"])
            
            if input_type == "video":
                # Extract first frame
                from shared.frame_utils import extract_first_frame
                
                if progress_cb:
                    progress_cb("🎬 Extracting first frame for preview...\n")
                
                success, frame_path, error = extract_first_frame(
                    settings["input_path"],
                    format="png"
                )
                
                if not success or not frame_path:
                    return f"❌ Frame extraction failed: {error}", error, None, None, "Preview failed", "Preview failed"
                
                # Process the extracted frame as an image
                preview_settings = settings.copy()
                preview_settings["input_path"] = frame_path
                preview_settings["output_format"] = "png"
                preview_settings["load_cap"] = 1
                
                if progress_cb:
                    progress_cb("🎨 Upscaling first frame...\n")
                
                result = runner.run_seedvr2(
                    preview_settings,
                    on_progress=lambda x: progress_cb(x) if progress_cb else None,
                    preview_only=True
                )
                
                if result.output_path and Path(result.output_path).exists():
                    output_image = result.output_path
                    status = "✅ First-frame preview complete"
                    local_logs.append("Preview mode: Processed first frame only")
                    chunk_info_msg = "Preview: First frame extracted and upscaled"
                    chunk_summary = f"Preview output: {output_image}"
                    chunk_progress_msg = "Preview mode: 1/1 frames"
                else:
                    status = "❌ Preview upscaling failed"
                    local_logs.append(result.log)
                    chunk_progress_msg = "Preview failed"
                    
                return status, "\n".join(local_logs), None, output_image, chunk_info_msg, chunk_summary, chunk_progress_msg
                
            else:
                # For images, just process normally with load_cap=1
                settings["load_cap"] = 1
                settings["output_format"] = "png"

        # Model loading check
        model_manager = get_model_manager()
        dit_model = settings.get("dit_model", "")
        # Model loading check
        model_manager = get_model_manager()
        dit_model = settings.get("dit_model", "")

        if not model_manager.is_model_loaded(ModelType.SEEDVR2, dit_model, **settings):
            if progress_cb:
                progress_cb(f"Loading model: {dit_model}...\n")
            if not runner.ensure_seedvr2_model_loaded(settings, lambda x: progress_cb(x) if progress_cb else None):
                return "❌ Model load failed", "", None, None, "Model load failed", "Model load failed"
            if progress_cb:
                progress_cb("Model loaded successfully!\n")

            # 🎬 CHUNKING SYSTEM ARCHITECTURE - Two Complementary Methods:
        # 
        # METHOD 1: PySceneDetect Chunking (PREFERRED, UNIVERSAL)
        # --------------------------------------------------------
        # - Controlled by: Resolution & Scene Split tab → chunk_size_sec setting in shared state
        # - Settings: chunk_size_sec, chunk_overlap_sec, scene_threshold, min_scene_len
        # - How it works: Externally splits video into scenes using PySceneDetect,
        #   processes each scene separately, then concatenates with blending
        # - Works with: ALL models (SeedVR2, GAN, RIFE, FlashVSR+)
        # - Use when: Processing long videos, managing VRAM, or optimizing per-scene quality
        # 
        # METHOD 2: SeedVR2 Native Streaming (SEEDVR2-SPECIFIC OPTIMIZATION)
        # --------------------------------------------------------------------
        # - Controlled by: SeedVR2 tab → "Streaming Chunk Size (frames)" control
        # - Settings: chunk_size (frames), temporal_overlap
        # - How it works: CLI-internal memory-bounded processing, streams frames through model
        # - Works with: ONLY SeedVR2 (built into the CLI via --chunk_size flag)
        # - Use when: Processing VERY long videos where even scene chunks exceed VRAM
        # - Can combine: Both methods work together! PySceneDetect splits into scenes,
        #   then each scene uses native streaming internally for maximum efficiency.
        # 
        # Priority: PySceneDetect chunking happens FIRST (external), then native streaming
        # is applied within each chunk if enabled (settings["chunk_size"] > 0 passed to CLI).
        
        # Check if PySceneDetect chunking should be enabled (ONLY from Resolution tab now)
        # Legacy chunk_enable removed - chunking is now purely controlled by Resolution tab
        chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
        chunk_enabled_resolution_tab = chunk_size_sec > 0
        
        should_chunk = (
            chunk_enabled_resolution_tab
            and not preview_only
            and detect_input_type(settings["input_path"]) == "video"
        )

        if should_chunk:
            # Process with external PySceneDetect chunking
            # This splits video into scenes, processes each, then concatenates
            completed_chunks = 0
            total_chunks_estimate = 1
            chunk_progress_updates = []

            def chunk_progress_callback(progress_val, desc=""):
                nonlocal completed_chunks, total_chunks_estimate
                
                # Extract total chunks from description if available
                import re
                total_match = re.search(r'(\d+)/(\d+)', desc)
                if total_match:
                    completed_chunks = int(total_match.group(1))
                    total_chunks_estimate = int(total_match.group(2))
                elif "Completed chunk" in desc or "chunk" in desc.lower():
                    completed_chunks += 1
                
                chunk_progress_updates.append(f"[Chunk {completed_chunks}/{total_chunks_estimate}] {desc}")
                
                if progress_cb:
                    progress_cb(f"Chunk {completed_chunks}/{total_chunks_estimate}: {desc}\n")

            # Get ALL chunking params from Resolution tab (via seed_controls)
            # PySceneDetect parameters now managed centrally in Resolution tab
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls.get("min_scene_len", 2.0))
            
            rc, clog, final_out, chunk_count = chunk_and_process(
                runner,
                settings,
                scene_threshold=scene_threshold,
                min_scene_len=min_scene_len,
                temp_dir=Path(global_settings["temp_dir"]),
                on_progress=lambda msg: progress_cb(msg) if progress_cb else None,
                chunk_seconds=chunk_size_sec,
                chunk_overlap=float(seed_controls.get("chunk_overlap_sec", 0) or 0),
                per_chunk_cleanup=bool(seed_controls.get("per_chunk_cleanup", False)),
                resume_from_partial=bool(settings.get("resume_chunking", False)),
                allow_partial=True,
                global_output_dir=str(runner.output_dir) if hasattr(runner, "output_dir") else None,
                progress_tracker=chunk_progress_callback,
                process_func=None,  # Use default model_type routing
                model_type="seedvr2",  # Explicitly specify SeedVR2 processing
            )

            status = "✅ Chunked upscale complete" if rc == 0 else f"⚠️ Chunked upscale ended early ({rc})"
            output_path = final_out if final_out else None
            output_video = output_path if output_path and output_path.lower().endswith(".mp4") else None
            output_image = None
            local_logs.append(clog)
            
            # Enhanced summary showing both chunking methods if applicable
            native_streaming_info = ""
            if settings.get("chunk_size", 0) > 0:
                native_streaming_info = f" + native streaming ({settings['chunk_size']} frames/chunk)"
            chunk_summary = f"Processed {chunk_count} scene chunks{native_streaming_info}. Final: {output_path}"
            
            # Build detailed chunk info with live progress updates
            chunk_progress_detailed = "\n".join(chunk_progress_updates[-10:]) if chunk_progress_updates else "Chunking in progress..."
            chunk_info_msg = f"**Chunks Completed:** {completed_chunks}/{total_chunks_estimate}\n**Native Streaming:** {'Yes' if settings.get('chunk_size', 0) > 0 else 'No'}\n**Latest Progress:**\n{chunk_progress_detailed}"
            chunk_progress_msg = f"Chunks: {completed_chunks}/{total_chunks_estimate}"
            
            result = RunResult(rc, output_path, clog)
        else:
            # Process without external chunking
            # NOTE: SeedVR2 native streaming (--chunk_size in frames) is handled
            # directly by the CLI via settings["chunk_size"] parameter
            # This is independent of PySceneDetect chunking above
            result = runner.run_seedvr2(
                settings,
                on_progress=lambda x: progress_cb(x) if progress_cb else None,
                preview_only=preview_only
            )
            
            # Add informative message if native streaming is being used
            native_chunk_size = settings.get("chunk_size", 0)
            if native_chunk_size > 0:
                status = "✅ Upscale complete (SeedVR2 native streaming)" if result.returncode == 0 else f"⚠️ Upscale exited with code {result.returncode}"
                chunk_summary = f"SeedVR2 native streaming: {native_chunk_size} frames/chunk (CLI-internal, memory-efficient)"
                chunk_info_msg = f"Native streaming enabled: {native_chunk_size} frames/chunk\nTemporal overlap: {settings.get('temporal_overlap', 0)}\nMemory-bounded processing for long videos."
                chunk_progress_msg = "Native streaming: CLI-internal chunking"
            else:
                status = "✅ Upscale complete" if result.returncode == 0 else f"⚠️ Upscale exited with code {result.returncode}"
                chunk_summary = "Single pass (entire video loaded at once)"
                chunk_info_msg = "Single-pass processing (entire video in memory)"
                chunk_progress_msg = "Single pass: no chunking"

        # Extract output paths
        if result.output_path:
            output_video = result.output_path if result.output_path.lower().endswith(".mp4") else None
            output_image = result.output_path if not result.output_path.lower().endswith(".mp4") else None

            # Update state
            try:
                outp = Path(result.output_path)
                seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
            except Exception:
                pass

            # Log the run - use live output_dir from global settings
            live_output_dir = Path(global_settings.get("output_dir", output_dir))
            run_logger.write_summary(
                Path(result.output_path) if result.output_path else live_output_dir,
                {
                    "input": settings["input_path"],
                    "output": result.output_path,
                    "returncode": result.returncode,
                    "args": settings,
                    "face_global": face_apply,
                    "chunk_summary": chunk_summary,
                },
            )

        # Apply face restoration if enabled
        if face_apply and output_video and Path(output_video).exists():
            restored = restore_video(
                output_video,
                strength=face_strength,
                on_progress=lambda x: progress_cb(x) if progress_cb else None
            )
            if restored:
                local_logs.append(f"Face-restored video saved to {restored} (strength {face_strength})")
                output_video = restored

        if face_apply and output_image and Path(output_image).exists():
            restored_img = restore_image(output_image, strength=face_strength)
            if restored_img:
                local_logs.append(f"Face-restored image saved to {restored_img} (strength {face_strength})")
                output_image = restored_img

        # Apply FPS override if specified (check both tab setting and Output tab cache)
        fps_val = settings.get("fps_override", 0) or seed_controls.get("fps_override_val", 0)
        if fps_val and fps_val > 0 and output_video and Path(output_video).exists():
            try:
                adjusted = ffmpeg_set_fps(Path(output_video), float(fps_val))
                output_video = str(adjusted)
                local_logs.append(f"✅ FPS overridden to {fps_val}: {adjusted}")
            except Exception as e:
                local_logs.append(f"⚠️ FPS override failed: {str(e)}")

        # Generate comparison video if enabled
        comparison_mode = seed_controls.get("comparison_mode_val", "slider")
        if comparison_mode in ["side_by_side", "stacked"] and output_video and Path(output_video).exists():
            from shared.video_comparison_advanced import create_side_by_side_video, create_stacked_video
            
            input_video = settings["input_path"]
            if Path(input_video).exists() and Path(input_video).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                comparison_path = Path(output_video).parent / f"{Path(output_video).stem}_comparison.mp4"
                
                if comparison_mode == "side_by_side":
                    success, comp_path, err = create_side_by_side_video(
                        input_video, output_video, str(comparison_path)
                    )
                else:  # stacked
                    success, comp_path, err = create_stacked_video(
                        input_video, output_video, str(comparison_path)
                    )
                
                if success:
                    local_logs.append(f"✅ Comparison video created: {comp_path}")
                else:
                    local_logs.append(f"⚠️ Comparison video failed: {err}")

    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        local_logs.append(f"❌ {error_msg}")
        status = "❌ Processing failed"
        chunk_summary = "Failed"
        chunk_info_msg = f"Error: {error_msg}"
        chunk_progress_msg = "Error occurred"

    return status, "\n".join(local_logs), output_video, output_image, chunk_info_msg, chunk_summary, chunk_progress_msg


# Comparison initialization ---------------------------------------------------
def comparison_html_slider():
    """
    Initialize comparison note with helpful instructions.
    
    The actual video comparison slider is created dynamically during processing
    using create_video_comparison_html from shared.video_comparison_slider.
    """
    return gr.HTML.update(
        value="""
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
            <p style="color: #495057; font-size: 14px; margin: 0;">
                📊 <strong>Comparison View:</strong> Process a video or image to see before/after comparison.<br>
                Videos use interactive HTML5 slider with fullscreen support. Images use Gradio's ImageSlider.
            </p>
        </div>
        """
    )


# Core run/cancel/preset callbacks --------------------------------------------
def build_seedvr2_callbacks(
    preset_manager: PresetManager,
    runner: Runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    output_dir: Path,
    temp_dir: Path,
):
    """
    Build SeedVR2 callback functions for the UI.
    
    Returns a dict of callbacks that can be wired to Gradio components.
    The callbacks handle preset management, validation, and processing orchestration.
    
    Note: Component order validation is performed when inputs_list is passed to save_preset.
    If you add new controls, update both SEEDVR2_ORDER and inputs_list in seedvr2_tab.py.
    """
    defaults = seedvr2_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown."""
        presets = preset_manager.list_presets("seedvr2", model_name)
        last_used = preset_manager.get_last_used_name("seedvr2", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, model_name: str, *args):
        """
        Save a preset with automatic validation.
        
        Validates that args length matches SEEDVR2_ORDER to catch integration bugs early.
        """
        if not preset_name.strip():
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            # Validate component count matches ORDER
            if len(args) != len(SEEDVR2_ORDER):
                error_msg = (
                    f"⚠️ Preset schema mismatch: received {len(args)} values but expected {len(SEEDVR2_ORDER)}. "
                    f"This indicates inputs_list in seedvr2_tab.py is out of sync with SEEDVR2_ORDER. "
                    f"Preset saving aborted to prevent corruption."
                )
                error_logger.error(error_msg)
                return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ {error_msg[:200]}..."), *list(args)
            
            payload = _seedvr2_dict_from_args(list(args))
            validated_payload = _enforce_seedvr2_guardrails(payload, defaults, state=None)

            preset_manager.save_preset_safe("seedvr2", model_name, preset_name.strip(), validated_payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            # Reload the validated values to ensure UI consistency
            current_map = dict(zip(SEEDVR2_ORDER, list(args)))
            loaded_vals = _apply_preset_to_values(validated_payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """Load a preset."""
        try:
            preset = preset_manager.load_preset_safe("seedvr2", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("seedvr2", model_name, preset_name)
                preset = preset_manager.validate_preset_constraints(preset, "seedvr2", model_name)
                preset = _enforce_seedvr2_guardrails(preset, defaults, state=None)

            current_map = dict(zip(SEEDVR2_ORDER, current_values))
            values = _apply_preset_to_values(preset or {}, defaults, preset_manager, current=current_map)
            return values
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            return current_values

    def safe_defaults():
        """Get safe default values."""
        return [defaults[key] for key in SEEDVR2_ORDER]

    def check_resume_status(global_settings, output_format):
        """Check chunking resume status."""
        temp_dir_path = Path(global_settings["temp_dir"])
        available, message = check_resume_available(temp_dir_path, output_format or "mp4")
        if available:
            return gr.Markdown.update(value=f"✅ {message}", visible=True)
        else:
            return gr.Markdown.update(value=f"ℹ️ {message}", visible=True)

    def cancel():
        """
        Cancel current processing and properly compile any partial outputs if available.
        
        ENHANCED: Uses proper merge pipeline with scene overlap handling when possible,
        falls back to simple concat or latest-file recovery for edge cases.
        
        BATCH SUPPORT: Also handles partial batch outputs by collecting completed files.
        """
        canceled = runner.cancel()
        if not canceled:
            return gr.Markdown.update(value="No active process to cancel"), ""

        # Check multiple locations for partial outputs:
        # 1. External chunked processing (PySceneDetect chunks) - PREFERRED (uses proper merge)
        # 2. Batch processing outputs - collect completed files
        # 3. Single-pass outputs in temp directory - FALLBACK (best-effort copy)
        
        compiled_output = None
        merge_method = "none"
        temp_base = Path(global_settings["temp_dir"])
        temp_chunks_dir = temp_base / "chunks"
        
        # Try to find and properly merge chunk-based partial outputs
        if temp_chunks_dir.exists():
            try:
                from shared.chunking import detect_resume_state, concat_videos, concat_videos_with_blending
                from shared.path_utils import collision_safe_path, get_media_fps

                partial_video, completed_chunks = detect_resume_state(temp_chunks_dir, "mp4")
                partial_png, completed_png_chunks = detect_resume_state(temp_chunks_dir, "png")

                # Try to compile video chunks with PROPER BLENDING if overlap detected
                if completed_chunks and len(completed_chunks) > 0:
                    partial_target = collision_safe_path(temp_chunks_dir / "cancelled_partial.mp4")
                    
                    # Detect if chunks have overlap metadata (check for overlap markers in chunk names or metadata)
                    # For now, check if chunk count > 1 and use blending as safer default
                    use_blending = len(completed_chunks) > 1
                    
                    if use_blending:
                        # Use proper blending merge (same as main pipeline)
                        # Get FPS from first chunk
                        first_chunk_fps = get_media_fps(str(completed_chunks[0])) if completed_chunks else 30.0
                        
                        # Estimate overlap frames (assume 0.5s overlap by default for PySceneDetect chunks)
                        overlap_frames = int((first_chunk_fps or 30.0) * 0.5)
                        
                        success = concat_videos_with_blending(
                            chunk_paths=completed_chunks,
                            output_path=partial_target,
                            overlap_frames=overlap_frames,
                            fps=first_chunk_fps,
                            on_progress=None  # Silent merge during cancel
                        )
                        merge_method = "blended" if success else "simple"
                    else:
                        # Single chunk or no overlap - simple concat
                        success = concat_videos(completed_chunks, partial_target)
                        merge_method = "simple"
                    
                    if success and partial_target.exists():
                        # Copy to outputs folder with collision-safe naming
                        final_output = Path(output_dir) / f"cancelled_partial_upscaled.mp4"
                        final_output = collision_safe_path(final_output)
                        shutil.copy2(partial_target, final_output)
                        compiled_output = str(final_output)

                # Or compile PNG chunks (directory-based)
                elif completed_png_chunks and len(completed_png_chunks) > 0:
                    partial_target = collision_safe_path(temp_chunks_dir / "cancelled_partial_png")
                    partial_target.mkdir(parents=True, exist_ok=True)

                    for i, chunk_path in enumerate(completed_png_chunks, 1):
                        dest = partial_target / f"chunk_{i:04d}"
                        if Path(chunk_path).is_dir():
                            shutil.copytree(chunk_path, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(chunk_path, dest)

                    compiled_output = str(partial_target)
                    merge_method = "png_collection"

            except Exception as e:
                # Log error but continue to fallback
                error_logger.warning(f"Proper chunk merge failed during cancel: {e}")
        
        # Check for batch partial outputs (completed files in output_dir)
        if not compiled_output:
            try:
                # Look for recently created upscaled files (from batch jobs that completed before cancel)
                batch_outputs = []
                for ext in [".mp4", ".avi", ".mov", ".mkv", ".png", ".jpg", ".jpeg"]:
                    batch_outputs.extend(list(Path(output_dir).glob(f"*_upscaled{ext}")))
                
                # Filter to files modified in last 24 hours (current session)
                import time
                current_time = time.time()
                recent_outputs = [
                    f for f in batch_outputs 
                    if current_time - f.stat().st_mtime < 86400  # 24 hours
                ]
                
                if recent_outputs:
                    compiled_output = str(Path(output_dir))
                    merge_method = "batch_partial"
                    
            except Exception as e:
                error_logger.warning(f"Batch partial recovery failed: {e}")
        
        # FALLBACK: Check for single-pass partial outputs in temp directory
        if not compiled_output:
            try:
                # Look for any mp4/png files created in temp during processing
                temp_files = []
                for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    temp_files.extend(list(temp_base.glob(f"*{ext}")))
                for ext in [".png", ".jpg", ".jpeg"]:
                    temp_files.extend(list(temp_base.glob(f"*{ext}")))
                
                # Sort by modification time (most recent first)
                temp_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                
                if temp_files:
                    # Copy most recent partial output to outputs folder
                    most_recent = temp_files[0]
                    from shared.path_utils import collision_safe_path
                    
                    final_output = Path(output_dir) / f"cancelled_partial_{most_recent.name}"
                    final_output = collision_safe_path(final_output)
                    
                    if most_recent.is_file():
                        shutil.copy2(most_recent, final_output)
                        compiled_output = str(final_output)
                        merge_method = "latest_file"
                    elif most_recent.is_dir():
                        shutil.copytree(most_recent, final_output, dirs_exist_ok=True)
                        compiled_output = str(final_output)
                        merge_method = "latest_dir"
                        
            except Exception as e:
                error_logger.warning(f"Fallback recovery failed during cancel: {e}")

        if compiled_output:
            merge_info = {
                "blended": "Chunks merged with proper frame blending (scene overlap handled)",
                "simple": "Chunks concatenated (no overlap detected)",
                "png_collection": "PNG frames collected from chunks",
                "batch_partial": f"Batch processing cancelled - {len(recent_outputs)} completed files saved in output folder",
                "latest_file": "⚠️ Best-effort: Latest temp file copied (no proper merge)",
                "latest_dir": "⚠️ Best-effort: Latest temp directory copied (no proper merge)"
            }.get(merge_method, "Unknown merge method")
            
            return (
                gr.Markdown.update(value=f"⏹️ Cancelled - Partial output saved: {Path(compiled_output).name}\n**Merge method:** {merge_info}"),
                f"Partial results saved to: {compiled_output}\n\nMerge method: {merge_info}"
            )
        else:
            return (
                gr.Markdown.update(value="⏹️ Cancelled - No partial outputs found"),
                "Processing was cancelled. No recoverable partial outputs were found in temp directories."
            )

    def open_outputs_folder_seedvr2(state: Dict[str, Any]):
        """Open outputs folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import open_outputs_folder
        # Use live output_dir from global settings (may have changed since tab load)
        live_output_dir = str(global_settings.get("output_dir", output_dir))
        return open_outputs_folder(live_output_dir)
    
    def clear_temp_folder_seedvr2(confirm: bool):
        """Clear temp folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import clear_temp_folder
        # Use live temp_dir from global settings (may have changed since tab load)
        live_temp_dir = str(global_settings.get("temp_dir", temp_dir))
        return clear_temp_folder(live_temp_dir, confirm)

    def get_model_loading_status():
        """Get current model loading status for UI display."""
        try:
            model_manager = get_model_manager()
            loaded_models = model_manager.get_loaded_models_info()
            current_model = model_manager.current_model_id

            if not loaded_models:
                return "No models loaded"

            status_lines = []
            for model_id, info in loaded_models.items():
                state = info["state"]
                marker = "✅" if state == "loaded" else "⏳" if state == "loading" else "❌"
                current_marker = " ← current" if model_id == current_model else ""
                status_lines.append(f"{marker} {info['model_name']} ({state}){current_marker}")

            return "\n".join(status_lines)
        except Exception as e:
            return f"Error getting model status: {str(e)}"

    def _auto_res_from_input(input_path: str, state: Dict[str, Any]):
        """Auto-calculate resolution and chunk estimates."""
        seed_controls = state.get("seed_controls", {})
        model_name = seed_controls.get("current_model") or defaults.get("dit_model")
        model_cache = seed_controls.get("resolution_cache", {}).get(model_name, {})

        if not input_path:
            return (
                gr.Slider.update(),
                gr.Slider.update(),
                gr.Markdown.update(value="Provide an input to auto-calc resolution/chunks."),
                state
            )

        p = Path(normalize_path(input_path))
        if not p.exists():
            return (
                gr.Slider.update(),
                gr.Slider.update(),
                gr.Markdown.update(value="Input path not found; keeping current resolution."),
                state
            )

        auto_res = model_cache.get("auto_resolution", seed_controls.get("auto_resolution", True))
        enable_max = model_cache.get("enable_max_target", seed_controls.get("enable_max_target", True))
        ratio_down = model_cache.get("ratio_downscale", seed_controls.get("ratio_downscale", False))
        chunk_size = float(model_cache.get("chunk_size_sec", seed_controls.get("chunk_size_sec", 0) or 0))
        chunk_overlap = float(model_cache.get("chunk_overlap_sec", seed_controls.get("chunk_overlap_sec", 0) or 0))

        target_res = int(model_cache.get("resolution_val") or seed_controls.get("resolution_val") or defaults["resolution"])
        max_target_res = int(model_cache.get("max_resolution_val") or seed_controls.get("max_resolution_val") or defaults["max_resolution"])

        dims = get_media_dimensions(str(p))
        msg_lines = []
        new_res = target_res

        if auto_res and dims:
            w, h = dims
            short_side = min(w, h)
            computed = min(short_side, target_res or short_side)
            if ratio_down:
                computed = min(computed, target_res or computed)
            if enable_max and max_target_res and max_target_res > 0:
                computed = min(computed, max_target_res)
            new_res = int((computed // 16) * 16 or computed)
            state["seed_controls"]["resolution_val"] = new_res
            msg_lines.append(f"Auto-resolution: input {w}x{h} → target {new_res} (max {max_target_res})")
        else:
            msg_lines.append("Auto-resolution disabled; no change.")

        # Chunk estimate
        est_msg = ""
        if chunk_size > 0 and chunk_overlap < chunk_size:
            dur = get_media_duration_seconds(str(p)) if detect_input_type(str(p)) == "video" else None
            if dur:
                import math
                est_chunks = math.ceil(dur / max(0.001, chunk_size - chunk_overlap))
                est_msg = f"Chunk estimate: ~{est_chunks} chunks for {dur:.1f}s (size {chunk_size}s, overlap {chunk_overlap}s)."
            else:
                est_msg = f"Chunking: size {chunk_size}s, overlap {chunk_overlap}s (duration unknown)."
        if est_msg:
            msg_lines.append(est_msg)

        return (
            gr.Slider.update(value=new_res),
            gr.Slider.update(value=max_target_res),
            gr.Markdown.update(value="\n".join(msg_lines)),
            state
        )

    def run_action(uploaded_file, face_restore_run, *args, preview_only: bool = False, state: Dict[str, Any] = None, progress=None):
        """Main processing action with streaming support and gr.Progress integration."""
        try:
            state = state or {"seed_controls": {}, "operation_status": "ready"}
            state["operation_status"] = "running"
            seed_controls = state.get("seed_controls", {})

            # Parse settings
            settings = dict(zip(SEEDVR2_ORDER, list(args)))
            settings = _enforce_seedvr2_guardrails(settings, defaults, state=state)  # Pass state for resolution tab integration

            # Validate inputs - now extracts original filename from upload
            input_path, original_filename = _resolve_input_path(
                uploaded_file,
                settings["input_path"],
                settings["batch_enable"],
                settings["batch_input_path"]
            )
            settings["input_path"] = normalize_path(input_path)
            settings["_original_filename"] = original_filename  # Store for output naming
            state["seed_controls"]["last_input_path"] = settings["input_path"]
            state["seed_controls"]["_original_filename"] = original_filename

            if not settings["input_path"] or not Path(settings["input_path"]).exists():
                yield (
                    "❌ Input path missing or not found",
                    "",
                    "",
                    None,
                    None,
                    "No chunks",
                    "",
                    "",
                    gr.HTML.update(value="No comparison"),
                    gr.ImageSlider.update(value=None),
                    gr.HTML.update(value="", visible=False),
                    gr.Gallery.update(visible=False),
                    state
                )
                return

            # Expand "all" to device list if specified
            cuda_device_raw = settings.get("cuda_device", "")
            if cuda_device_raw:
                settings["cuda_device"] = _expand_cuda_spec(cuda_device_raw)
            
            # Validate CUDA devices
            cuda_warning = _validate_cuda_devices(settings.get("cuda_device", ""))
            if cuda_warning:
                yield (
                    f"⚠️ {cuda_warning}",
                    "",
                    "",
                    None,
                    None,
                    "No chunks",
                    "",
                    "",
                    gr.HTML.update(value="No comparison"),
                    gr.ImageSlider.update(value=None),
                    gr.HTML.update(value="", visible=False),
                    gr.Gallery.update(visible=False),
                    state
                )
                return

            # Check ffmpeg availability
            if not _ffmpeg_available():
                yield (
                    "❌ ffmpeg not found in PATH. Install ffmpeg and retry.",
                    "",
                    "",
                    None,
                    None,
                    "No chunks",
                    "",
                    "",
                    gr.HTML.update(value="No comparison"),
                    gr.ImageSlider.update(value=None),
                    gr.HTML.update(value="", visible=False),
                    gr.Gallery.update(visible=False),
                    state
                )
                return

            # Check disk space (require at least 5GB free)
            output_path = Path(global_settings.get("output_dir", output_dir))
            has_space, space_warning = check_disk_space(output_path, required_mb=5000)
            if not has_space:
                yield (
                    space_warning or "❌ Insufficient disk space",
                    f"Free up disk space before processing. Recommended: 5GB+ free",
                    "",
                    None,
                    None,
                    "No chunks",
                    "",
                    "",
                    gr.HTML.update(value="No comparison"),
                    gr.ImageSlider.update(value=None),
                    gr.HTML.update(value="", visible=False),
                    gr.Gallery.update(visible=False),
                    state
                )
                return
            elif space_warning:
                # Low space but might work - show warning
                yield (
                    f"⚠️ {space_warning}",
                    "Low disk space detected. Processing may fail if output is large.",
                    "",
                    None,
                    None,
                    "Disk space warning",
                    "",
                    "",
                    gr.HTML.update(value=""),
                    gr.ImageSlider.update(value=None),
                    gr.HTML.update(value="", visible=False),
                    gr.Gallery.update(visible=False),
                    state
                )

            # Setup processing parameters
            face_apply = bool(face_restore_run) or bool(global_settings.get("face_global", False))
            face_strength = float(global_settings.get("face_strength", 0.5))

            # Apply cached values from Resolution & Scene Split tab
            if seed_controls.get("resolution_val") is not None:
                settings["resolution"] = seed_controls["resolution_val"]
            if seed_controls.get("max_resolution_val") is not None:
                settings["max_resolution"] = seed_controls["max_resolution_val"]

            auto_res = seed_controls.get("auto_resolution", True)
            enable_max_target = seed_controls.get("enable_max_target", True)
            chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap_sec = float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            ratio_downscale = seed_controls.get("ratio_downscale", False)
            per_chunk_cleanup = seed_controls.get("per_chunk_cleanup", False)

            settings["chunk_size_sec"] = chunk_size_sec
            settings["chunk_overlap_sec"] = chunk_overlap_sec
            settings["per_chunk_cleanup"] = per_chunk_cleanup

            # Auto-resolution calculation
            media_dims = get_media_dimensions(settings["input_path"])
            if media_dims and auto_res:
                w, h = media_dims
                short_side = min(w, h)
                target_res = settings["resolution"]
                max_target_res = settings["max_resolution"]

                computed_res = min(short_side, target_res or short_side)
                if ratio_downscale:
                    computed_res = min(computed_res, target_res or computed_res)
                if enable_max_target and max_target_res and max_target_res > 0:
                    computed_res = min(computed_res, max_target_res)
                settings["resolution"] = int(computed_res // 16 * 16 or computed_res)

            # Apply output format from Comparison tab if set
            if seed_controls.get("output_format_val"):
                if settings.get("output_format") in (None, "auto"):
                    settings["output_format"] = seed_controls["output_format_val"]

            if settings["output_format"] == "auto":
                settings["output_format"] = None
            
            # Apply PNG padding/basename settings from Output tab if available
            # These are used for PNG sequence outputs and collision-safe path generation
            if seed_controls.get("png_padding_val") is not None:
                settings["png_padding"] = int(seed_controls["png_padding_val"])
            if seed_controls.get("png_keep_basename_val") is not None:
                settings["png_keep_basename"] = bool(seed_controls["png_keep_basename_val"])
            
            # Apply metadata and telemetry settings from Output tab
            if seed_controls.get("save_metadata_val") is not None:
                settings["save_metadata"] = bool(seed_controls["save_metadata_val"])
            if seed_controls.get("telemetry_enabled_val") is not None:
                # Note: telemetry is controlled globally via runner.set_telemetry()
                # This flag controls per-run metadata emission only
                settings["telemetry_enabled"] = bool(seed_controls["telemetry_enabled_val"])
            
            # Apply FPS override from Output tab ONLY if not explicitly set in SeedVR2 tab
            # SeedVR2 tab value takes precedence
            if seed_controls.get("fps_override_val") is not None and seed_controls["fps_override_val"] > 0:
                if settings.get("fps_override", 0) == 0:  # Only if not set in SeedVR2 tab
                    settings["fps_override"] = float(seed_controls["fps_override_val"])
            
            # Apply skip_first_frames and load_cap from Output tab ONLY if not explicitly set in SeedVR2 tab
            # SeedVR2 tab values take precedence over Output tab cached values
            if seed_controls.get("skip_first_frames_val") is not None:
                if settings.get("skip_first_frames", 0) == 0:  # Only if not set in SeedVR2 tab
                    settings["skip_first_frames"] = int(seed_controls["skip_first_frames_val"])
            if seed_controls.get("load_cap_val") is not None:
                if settings.get("load_cap", 0) == 0:  # Only if not set in SeedVR2 tab
                    settings["load_cap"] = int(seed_controls["load_cap_val"])

            # Batch processing
            if settings.get("batch_enable"):
                # Use the batch processor for multiple files
                from shared.batch_processor import BatchProcessor, BatchJob

                batch_input_path = Path(settings.get("batch_input_path", ""))
                batch_output_path = Path(settings.get("batch_output_path", ""))

                if not batch_input_path.exists():
                    yield (
                        "❌ Batch input path does not exist",
                        "",
                        "",
                        None,
                        None,
                        "No chunks",
                        "",
                        "",
                        gr.HTML.update(value="No comparison"),
                        gr.ImageSlider.update(value=None),
                        gr.HTML.update(value="", visible=False),
                        gr.Gallery.update(visible=False),
                        state
                    )
                    return

                # Collect all files to process
                supported_exts = SEEDVR2_VIDEO_EXTS | SEEDVR2_IMAGE_EXTS
                batch_files = []
                if batch_input_path.is_dir():
                    for ext in supported_exts:
                        batch_files.extend(batch_input_path.glob(f"**/*{ext}"))
                elif batch_input_path.suffix.lower() in supported_exts:
                    batch_files = [batch_input_path]

                # Detect if this is an all-image batch or mixed/video batch
                video_count = sum(1 for f in batch_files if Path(f).suffix.lower() in SEEDVR2_VIDEO_EXTS)
                image_count = sum(1 for f in batch_files if Path(f).suffix.lower() in SEEDVR2_IMAGE_EXTS)
                is_image_only_batch = image_count > 0 and video_count == 0

                if not batch_files:
                    yield (
                        "❌ No supported files found in batch input",
                        "",
                        "",
                        None,
                        None,
                        "No chunks",
                        "",
                        "",
                        gr.HTML.update(value="No comparison"),
                        gr.ImageSlider.update(value=None),
                        gr.HTML.update(value="", visible=False),
                        gr.Gallery.update(visible=False),
                        state
                    )
                    return

                # Create batch processor
                batch_processor = BatchProcessor(
                    output_dir=batch_output_path if batch_output_path.exists() else output_dir,
                    max_workers=1,  # Sequential processing for memory management
                    telemetry_enabled=global_settings.get("telemetry", True)
                )

                # Create batch jobs
                jobs = []
                for input_file in sorted(set(batch_files)):
                    # For image-only batches, disable per-file telemetry (will write consolidated metadata at end)
                    job_global_settings = global_settings.copy()
                    if is_image_only_batch:
                        job_global_settings["telemetry"] = False  # Disable per-file metadata for images
                    
                    job = BatchJob(
                        input_path=str(input_file),
                        metadata={
                            "settings": settings.copy(),
                            "global_settings": job_global_settings,
                            "face_apply": face_apply,
                            "face_strength": face_strength,
                            "seed_controls": seed_controls.copy(),
                            "is_image": Path(input_file).suffix.lower() in SEEDVR2_IMAGE_EXTS,
                        }
                    )
                    jobs.append(job)

                # Process batch with progress updates
                def batch_progress_callback(progress_data):
                    current_job = progress_data.get("current_job")
                    overall_progress = progress_data.get("overall_progress", 0)
                    completed_files = progress_data.get("completed_files", 0)
                    status_msg = f"Batch processing: {overall_progress:.1f}% complete"
                    if current_job:
                        status_msg += f" - Processing: {Path(current_job).name}"

                    # Update gr.Progress with actual progress
                    if progress:
                        progress(
                            overall_progress / 100.0,
                            desc=f"Batch: {completed_files}/{len(jobs)} files processed"
                        )

                    yield (
                        status_msg,
                        f"Processing {len(jobs)} files...",
                        "",
                        None,
                        None,
                        f"Batch: {completed_files}/{len(jobs)} completed",
                        "",
                        "",
                        gr.HTML.update(value="Batch processing in progress..."),
                        gr.ImageSlider.update(value=None),
                        gr.HTML.update(value="", visible=False),
                        gr.Gallery.update(visible=False),
                        state
                    )

                # Define processing function for each job
                def process_single_batch_job(job: BatchJob, progress_cb):
                    try:
                        job.status = "processing"
                        job.start_time = time.time()

                        # Process single file with current settings
                        single_settings = job.metadata["settings"].copy()
                        single_settings["input_path"] = job.input_path
                        single_settings["batch_enable"] = False  # Disable batch for individual processing
                        
                        # Generate unique output path for this batch item to prevent collisions
                        # Use collision_safe_path to ensure uniqueness
                        input_file = Path(job.input_path)
                        batch_output_folder = Path(batch_output_path) if batch_output_path.exists() else output_dir
                        
                        # Determine output format
                        out_fmt = single_settings.get("output_format", "auto")
                        if out_fmt == "auto":
                            out_fmt = "mp4" if input_file.suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"] else "png"
                        
                        # Create unique output path with collision safety
                        from shared.path_utils import collision_safe_path, collision_safe_dir
                        if out_fmt == "png":
                            # For PNG output, create a directory
                            output_name = f"{input_file.stem}_upscaled"
                            unique_output = collision_safe_dir(batch_output_folder / output_name)
                            single_settings["output_override"] = str(unique_output)
                        else:
                            # For video output, create a file
                            output_name = f"{input_file.stem}_upscaled.{out_fmt}"
                            unique_output = collision_safe_path(batch_output_folder / output_name)
                            single_settings["output_override"] = str(unique_output)

                        status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress = _process_single_file(
                            runner,
                            single_settings,
                            job.metadata["global_settings"],
                            job.metadata["seed_controls"],
                            job.metadata["face_apply"],
                            job.metadata["face_strength"],
                            run_logger,
                            output_dir,
                            False,  # not preview
                            progress_cb
                        )

                        if output_video or output_image:
                            job.output_path = output_video or output_image
                            job.status = "completed"
                        else:
                            job.status = "failed"
                            job.error_message = logs

                        job.end_time = time.time()

                    except Exception as e:
                        job.status = "failed"
                        job.error_message = str(e)
                        job.end_time = time.time()

                    return job

                # Run batch processing
                results = batch_processor.process_batch(
                    jobs=jobs,
                    process_func=process_single_batch_job,
                    progress_callback=batch_progress_callback
                )

                # Summarize results and collect output paths for gallery
                completed = sum(1 for r in results if r.status == "completed")
                failed = sum(1 for r in results if r.status == "failed")
                
                # Collect successful outputs for gallery
                batch_outputs = []
                for job in jobs:
                    if job.status == "completed" and job.output_path and Path(job.output_path).exists():
                        batch_outputs.append(str(job.output_path))

                summary_msg = f"Batch complete: {completed}/{len(jobs)} succeeded"
                if failed > 0:
                    summary_msg += f", {failed} failed"

                # Write consolidated metadata for ALL batches (images AND videos)
                # User requirement: "only have 1 metadata for batch image upscale and have individual metadata for videos upscale"
                # Implementation: 
                # - Image batches: Single consolidated metadata (no per-file metadata to avoid spam)
                # - Video batches: Individual per-video metadata (already written by _process_single_file) + batch summary
                # - Mixed batches: Both image and video summaries
                
                try:
                    metadata_dir = Path(batch_output_path) if batch_output_path.exists() else output_dir
                    
                    if is_image_only_batch:
                        # Image-only batch: Single consolidated metadata (no per-file metadata)
                        batch_metadata = {
                            "batch_type": "images",
                            "total_files": len(jobs),
                            "completed": completed,
                            "failed": failed,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "settings": settings,
                            "outputs": batch_outputs,
                            "failed_files": [
                                {"input": job.input_path, "error": job.error_message}
                                for job in jobs if job.status == "failed"
                            ]
                        }
                        metadata_path = metadata_dir / "batch_images_metadata.json"
                    else:
                        # Video batch or mixed batch: Batch summary + individual per-video metadata
                        # Note: Individual video metadata already written by _process_single_file → run_logger.write_summary
                        batch_metadata = {
                            "batch_type": "videos" if video_count > 0 and image_count == 0 else "mixed",
                            "total_files": len(jobs),
                            "video_files": video_count,
                            "image_files": image_count,
                            "completed": completed,
                            "failed": failed,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "settings": settings,
                            "outputs": batch_outputs,
                            "individual_metadata_note": "Each video has its own run_metadata.json in its output directory",
                            "failed_files": [
                                {"input": job.input_path, "error": job.error_message}
                                for job in jobs if job.status == "failed"
                            ]
                        }
                        metadata_path = metadata_dir / "batch_videos_summary.json"
                    
                    import json
                    with metadata_path.open("w", encoding="utf-8") as f:
                        json.dump(batch_metadata, f, indent=2)
                    
                except Exception as e:
                    # Don't fail batch on metadata error
                    pass

                # Update gr.Progress to 100%
                if progress:
                    progress(1.0, desc="Batch complete!")

                yield (
                    f"✅ {summary_msg}",
                    f"Batch processing finished. {len(batch_outputs)} files saved to output folder.",
                    "",
                    None,
                    None,
                    f"Batch: {completed} completed, {failed} failed",
                    "",
                    "",
                    gr.HTML.update(value=f"Batch processing complete. {len(batch_outputs)} files saved."),
                    gr.ImageSlider.update(value=None),
                    gr.HTML.update(value="", visible=False),
                    gr.Gallery.update(value=batch_outputs[:50], visible=True) if batch_outputs else gr.Gallery.update(visible=False),  # Show first 50
                    state
                )
                return

            # Single file processing with streaming updates
            processing_complete = False
            last_progress_update = 0
            chunk_info = "Initializing..."  # Initialize before use

            def progress_callback(message: str):
                nonlocal last_progress_update
                current_time = time.time()
                # Throttle updates to every 0.5 seconds to avoid UI spam
                if current_time - last_progress_update > 0.5:
                    last_progress_update = current_time
                    yield (
                        f"⚙️ Processing: {message}",
                        f"Progress: {message}",
                        "",
                        None,
                        None,
                        chunk_info or "Processing...",
                        "",
                        "",
                        gr.HTML.update(value=f'<div style="background: #f0f8ff; padding: 10px; border-radius: 5px;">{message}</div>'),
                        gr.ImageSlider.update(value=None),
                        gr.HTML.update(value="", visible=False),
                        gr.Gallery.update(visible=False),
                        state
                    )

            # Start processing with progress tracking
            yield (
                "⚙️ Starting processing...",
                "Initializing...",
                "",
                None,
                None,
                "Initializing...",
                "",
                "",
                gr.HTML.update(value="Starting processing..."),
                gr.ImageSlider.update(value=None),
                gr.HTML.update(value="", visible=False),
                gr.Gallery.update(visible=False),
                state
            )

            # Create a queue for progress updates
            progress_queue = queue.Queue()

            def processing_thread():
                try:
                    status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress = _process_single_file(
                        runner,
                        settings,
                        global_settings,
                        seed_controls,
                        face_apply,
                        face_strength,
                        run_logger,
                        output_dir,
                        preview_only,
                        lambda msg: progress_queue.put(("progress", msg))
                    )
                    progress_queue.put(("complete", (status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress)))
                except Exception as e:
                    progress_queue.put(("error", str(e)))

            # Start processing in background thread
            import threading
            proc_thread = threading.Thread(target=processing_thread, daemon=True)
            proc_thread.start()

            # Stream progress updates with gr.Progress integration and throttling
            chunk_count = 0
            total_chunks_estimate = 1
            last_progress_value = 0.0
            last_ui_update_time = 0
            ui_update_throttle = 0.5  # Only update UI every 0.5 seconds
            accumulated_messages = []
            
            while proc_thread.is_alive() or not progress_queue.empty():
                try:
                    update_type, data = progress_queue.get(timeout=0.1)
                    current_time = time.time()
                    
                    if update_type == "progress":
                        # Always update gr.Progress for responsiveness
                        if progress:
                            import re
                            
                            # Try to extract chunk progress (e.g., "chunk 5/10", "Completed 3 chunks")
                            chunk_match = re.search(r'(?:chunk|chunks|Completed)\s+(\d+)(?:/|/|\s+of\s+|\s+)(\d+)', data, re.IGNORECASE)
                            if chunk_match:
                                chunk_count = int(chunk_match.group(1))
                                total_chunks_estimate = int(chunk_match.group(2))
                                progress_value = chunk_count / total_chunks_estimate
                                progress(progress_value, desc=f"Processing chunk {chunk_count}/{total_chunks_estimate}")
                                last_progress_value = progress_value
                            # Try to extract percentage (e.g., "50%", "Progress: 75%")
                            elif '%' in data:
                                pct_match = re.search(r'(\d+(?:\.\d+)?)%', data)
                                if pct_match:
                                    progress_value = float(pct_match.group(1)) / 100.0
                                    progress(progress_value, desc=data[:100])
                                    last_progress_value = progress_value
                                else:
                                    progress(last_progress_value, desc=data[:100])
                            # Try to extract "N/M" pattern (e.g., "Processing 5/100 frames")
                            elif re.search(r'(\d+)/(\d+)', data):
                                nm_match = re.search(r'(\d+)/(\d+)', data)
                                if nm_match:
                                    current = int(nm_match.group(1))
                                    total = int(nm_match.group(2))
                                    if total > 0:
                                        progress_value = current / total
                                        progress(progress_value, desc=data[:100])
                                        last_progress_value = progress_value
                                    else:
                                        progress(last_progress_value, desc=data[:100])
                                else:
                                    progress(last_progress_value, desc=data[:100])
                            else:
                                # Generic progress update - use last known value
                                progress(last_progress_value, desc=data[:100] if data else "Processing...")
                        
                        # Accumulate messages for throttled UI updates
                        accumulated_messages.append(data)
                        
                        # Only update UI on CHUNK COMPLETION (not in-progress chunk messages)
                        # Requirement: "only update when last newer chunk is done"
                        is_chunk_completion = (
                            ("completed chunk" in data.lower() or 
                             "finished chunk" in data.lower() or
                             "chunk complete" in data.lower()) and
                            (chunk_count > 0)  # Valid chunk count extracted
                        )
                        
                        is_critical_event = (
                            "error" in data.lower() or
                            "failed" in data.lower() or
                            "✅" in data or "❌" in data or
                            "complete" in data.lower()
                        )
                        
                        # STRICT THROTTLING: Only yield on chunk completion or critical events
                        # Suppress intermediate frame-level progress to keep UI clean
                        should_update_ui = is_chunk_completion or is_critical_event
                        
                        if should_update_ui:
                            # Join recent messages (last 5)
                            recent_messages = accumulated_messages[-5:]
                            display_text = "\n".join(recent_messages[-3:])  # Show last 3 lines
                            
                            yield (
                                f"⚙️ Processing...",
                                f"Progress: {display_text}",
                                "",
                                None,
                                None,
                                chunk_info or "Processing...",
                                "",
                                "",
                                gr.HTML.update(value=f'<div style="background: #f0f8ff; padding: 10px; border-radius: 5px; white-space: pre-wrap;">{display_text}</div>'),
                                gr.ImageSlider.update(value=None),
                                gr.HTML.update(value="", visible=False),
                                gr.Gallery.update(visible=False),
                                state
                            )
                            last_ui_update_time = current_time
                            
                            # Clear old accumulated messages (keep last 10 for context)
                            if len(accumulated_messages) > 10:
                                accumulated_messages = accumulated_messages[-10:]
                    elif update_type == "complete":
                        status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress = data
                        processing_complete = True
                        if progress:
                            progress(1.0, desc="Complete!")
                        break
                    elif update_type == "error":
                        if progress:
                            progress(0, desc="Error occurred")
                        yield (
                            "❌ Processing failed",
                            f"Error: {data}",
                            "",
                            None,
                            None,
                            "Error occurred",
                            "",
                            "",
                            gr.HTML.update(value=f'<div style="background: #ffe6e6; padding: 10px; border-radius: 5px;">Error: {data}</div>'),
                            gr.ImageSlider.update(value=None),
                            gr.HTML.update(value="", visible=False),
                            gr.Gallery.update(visible=False),
                            state
                        )
                        return
                except queue.Empty:
                    continue

            if not processing_complete:
                yield (
                    "❌ Processing timed out",
                    "Processing did not complete within expected time",
                    "",
                    None,
                    None,
                    "Timeout",
                    "",
                    "",
                    gr.HTML.update(value="Processing timed out"),
                    gr.ImageSlider.update(value=None),
                    gr.HTML.update(value="", visible=False),
                    gr.Gallery.update(visible=False),
                    state
                )
                return

            # Create comparison based on mode from Output tab
            comparison_mode = seed_controls.get("comparison_mode_val", "native")
            
            if comparison_mode == "native":
                # Use gradio's native ImageSlider for images
                if output_image and Path(output_image).exists():
                    comparison_html = ""
                    # Check for pinned reference
                    pinned_ref = seed_controls.get("pinned_reference_path")
                    pin_enabled = seed_controls.get("pin_reference_val", False)
                    
                    image_slider_update = gr.ImageSlider.update(
                        value=(pinned_ref if (pin_enabled and pinned_ref) else settings.get("input_path"), output_image),
                        visible=True
                    )
                else:
                    # Check for pinned reference
                    pinned_ref = seed_controls.get("pinned_reference_path")
                    pin_enabled = seed_controls.get("pin_reference_val", False)
                    
                    comparison_html, image_slider_update = create_comparison_selector(
                        input_path=settings.get("input_path"),
                        output_path=output_video or output_image,
                        comparison_mode="slider",
                        pinned_reference_path=pinned_ref,
                        pin_enabled=pin_enabled
                    )
            else:
                # Use custom HTML comparisons for other modes
                pinned_ref = seed_controls.get("pinned_reference_path")
                pin_enabled = seed_controls.get("pin_reference_val", False)
                
                comparison_html, image_slider_update = create_comparison_selector(
                    input_path=settings.get("input_path"),
                    output_path=output_video or output_image,
                    comparison_mode=comparison_mode,
                    pinned_reference_path=pinned_ref,
                    pin_enabled=pin_enabled
                )

            # Build video comparison HTML for videos
            video_comparison_html_update = gr.HTML.update(value="", visible=False)
            if output_video and Path(output_video).exists():
                original_path = settings.get("input_path", "")
                if original_path and Path(original_path).exists():
                    # Use new video comparison slider
                    from shared.video_comparison_slider import create_video_comparison_html as create_vid_comp
                    
                    video_comp_html = create_vid_comp(
                        original_video=original_path,
                        upscaled_video=output_video,
                        height=600,
                        slider_position=50.0
                    )
                    video_comparison_html_update = gr.HTML.update(value=video_comp_html, visible=True)
            
            # If no HTML comparison, use ImageSlider for images
            if not comparison_html and output_image and not output_video:
                image_slider_update = gr.ImageSlider.update(
                    value=(settings.get("input_path"), output_image),
                    visible=True,
                )
            elif not image_slider_update:
                image_slider_update = gr.ImageSlider.update(value=None, visible=False)

            state["operation_status"] = "completed" if "✅" in status else "ready"
            yield (
                status,
                logs,
                "",  # progress_indicator
                output_video,
                output_image,
                chunk_info,
                "",  # resume_status
                chunk_progress,  # NOW POPULATED with actual chunk progress
                comparison_html if comparison_html else gr.HTML.update(value="", visible=False),
                image_slider_update,
                video_comparison_html_update,
                gr.Gallery.update(visible=False),  # Hide gallery for single file
                state
            )

        except Exception as e:
            error_msg = f"Critical error in SeedVR2 processing: {str(e)}"
            state["operation_status"] = "error"
            yield (
                "❌ Critical error",
                error_msg,
                "",
                None,
                None,
                "Error",
                "",
                "",
                gr.HTML.update(value="Error occurred"),
                gr.ImageSlider.update(value=None),
                gr.HTML.update(value="", visible=False),
                gr.Gallery.update(visible=False),
                state
            )

    return {
        "defaults": defaults,
        "order": SEEDVR2_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "check_resume_status": check_resume_status,
        "run_action": run_action,
        "cancel_action": cancel,
        "open_outputs_folder": open_outputs_folder_seedvr2,
        "clear_temp_folder": clear_temp_folder_seedvr2,
        "comparison_html_slider": comparison_html_slider,
        "auto_res_on_input": _auto_res_from_input,
        "get_model_loading_status": get_model_loading_status,
    }
