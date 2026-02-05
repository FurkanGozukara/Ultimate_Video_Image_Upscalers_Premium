"""
SeedVR2 Service Module - Complete Rewrite
Handles all SeedVR2 processing logic, presets, and callbacks
"""

import shutil
import queue
import subprocess
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
    collision_safe_path,
    collision_safe_dir,
    ffmpeg_set_fps,
    get_media_dimensions,
    get_media_duration_seconds,
    detect_input_type,
)
from shared.resolution_calculator import estimate_seedvr2_upscale_plan_from_dims
from shared.chunking import chunk_and_process, check_resume_available
from shared.output_run_manager import prepare_single_video_run, downscaled_video_path, numbered_single_image_output_path
from shared.ffmpeg_utils import scale_video
from shared.face_restore import restore_image, restore_video
from shared.models.seedvr2_meta import get_seedvr2_model_names, model_meta_map
from shared.logging_utils import RunLogger
from shared.comparison_unified import create_unified_comparison, build_comparison_selector
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal
from shared.video_comparison import create_comparison_selector
from shared.model_manager import get_model_manager, ModelType
from shared.gpu_utils import expand_cuda_device_spec, validate_cuda_device_spec
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
    Get optimal default attention mode with GPU compute capability detection.
    
    INTELLIGENT PRIORITY based on GPU architecture:
    - **Blackwell (12.x)**: sageattn_3 > flash_attn_3 > flash_attn_2 > sdpa
    - **Hopper (9.x)**: flash_attn_3 > sageattn_2 > flash_attn_2 > sdpa
    - **Ada/Ampere (8.x-8.9)**: flash_attn_2 > sdpa
    - **Turing (7.5-7.x)**: sdpa (flash_attn slower on older arch)
    - **Older (<7.5)**: sdpa only
    
    Automatically selects the fastest attention backend for detected GPU.
    
    Returns:
        Best attention mode for detected GPU architecture
    """
    # IMPORTANT:
    # Do NOT import torch / flash-attn / sageattention in the parent Gradio process.
    # Importing torch can permanently increase RAM usage, and some CUDA queries can
    # create a CUDA context that reserves hundreds of MB of VRAM for the life of the UI.
    #
    # We use NVML (nvidia-smi) via shared.gpu_utils to detect compute capability without
    # initializing CUDA, and `importlib.util.find_spec()` to check optional backends
    # without importing their CUDA extensions.
    try:
        from shared.gpu_utils import get_gpu_info
        import importlib.util

        gpus = get_gpu_info()
        if not gpus:
            return "sdpa"

        compute_cap = gpus[0].compute_capability
        has_flash_attn = importlib.util.find_spec("flash_attn") is not None
        has_sageattention = importlib.util.find_spec("sageattention") is not None
        has_sageattn3 = importlib.util.find_spec("sageattn3") is not None

        # If we can't detect architecture, pick a safe default.
        if not compute_cap:
            return "flash_attn_2" if has_flash_attn else "sdpa"

        major, minor = compute_cap

        # Blackwell (12.x): prefer sageattn_3 if installed, else flash_attn_2, else sdpa.
        if major >= 12:
            if has_sageattn3:
                return "sageattn_3"
            return "flash_attn_2" if has_flash_attn else "sdpa"

        # Hopper (9.x-11.x): flash_attn_2 is generally safe if installed, else sageattn_2, else sdpa.
        if major >= 9:
            if has_flash_attn:
                return "flash_attn_2"
            if has_sageattention:
                return "sageattn_2"
            return "sdpa"

        # Ampere/Ada (8.x): flash_attn_2 if installed.
        if major >= 8:
            return "flash_attn_2" if has_flash_attn else "sdpa"

        return "sdpa"
    except Exception as e:
        print(f"Warning: Attention mode detection failed ({e}), using sdpa")
        return "sdpa"


def seedvr2_defaults(model_name: Optional[str] = None, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Get default SeedVR2 settings aligned with CLI defaults.
    Applies model-specific metadata when model_name is provided.
    
    Args:
        model_name: Optional model name to apply metadata from
        base_dir: Optional base directory (app root) to resolve model_dir path
    """
    # IMPORTANT: do not import torch here (parent Gradio process must remain GPU-free).
    # Use NVML-based detection (nvidia-smi) instead.
    try:
        from shared.gpu_utils import get_gpu_info
        cuda_default = "0" if get_gpu_info() else ""
    except Exception:
        cuda_default = ""
    
    # Compute correct model directory path
    # SeedVR2 models are stored in <base_dir>/SeedVR2/models/
    if base_dir:
        model_dir_path = str(Path(base_dir) / "SeedVR2" / "models")
    else:
        # Fallback: try to compute from this file's location
        import os
        service_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        app_base = service_dir.parent.parent  # shared/services -> shared -> app root
        model_dir_path = str(app_base / "SeedVR2" / "models")
    
    # Get model metadata if specific model is provided
    model_meta = None
    # Default to 7B fp16 model (best quality/speed balance)
    available_models = get_seedvr2_model_names()
    default_model = "seedvr2_ema_7b_fp16.safetensors" if "seedvr2_ema_7b_fp16.safetensors" in available_models else (available_models[0] if available_models else "seedvr2_ema_3b_fp16.safetensors")
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
        "model_dir": model_dir_path,
        "dit_model": target_model,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        # PySceneDetect chunking removed - now configured in Resolution tab
        # Legacy chunk_enable, scene_threshold, scene_min_len removed from ORDER
        "chunk_size": 0,  # SeedVR2 native chunking (frames per chunk, 0=disabled)
        "resolution": 1080,
        "max_resolution": 1920,  # Default 1920 (model cap only limits max slider, not default)
        # NEW (vNext): replace manual short-side target with an Upscale factor (default 4x).
        # The actual SeedVR2 CLI `resolution` is computed at runtime from the input dimensions.
        "upscale_factor": 4.0,
        # NEW (vNext): when max edge would clamp the requested upscale, optionally downscale the input first,
        # then upscale with the full factor to reach the capped target (useful for fixed-scale models).
        "pre_downscale_then_upscale": False,
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
        "dit_offload_device": "cpu",
        "vae_offload_device": "cpu",
        "tensor_offload_device": "cpu",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "vae_encode_tiled": True,
        "vae_encode_tile_size": 1024,
        "vae_encode_tile_overlap": 128,
        "vae_decode_tiled": True,
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
        # ADDED v2.5.22: FFmpeg 10-bit encoding support for reduced banding in gradients
        "video_backend": "opencv",  # "opencv" (default, 8-bit) or "ffmpeg" (10-bit capable)
        "use_10bit": False,  # Enable 10-bit color depth (requires video_backend="ffmpeg", x265 codec)
        "_compile_compatible": compile_compatible,  # Store for validation
    }


"""
 PRESET SERIALIZATION ORDER & ROBUSTNESS DESIGN
==================================================

CURRENT APPROACH: Manual Synchronization (Option C)
----------------------------------------------------

This list defines the order of parameters for preset save/load.
MUST match inputs_list order in ui/seedvr2_tab.py.

 MANUAL SYNCHRONIZATION REQUIRED:
Adding a new control requires updates across 3 locations:

1. **seedvr2_defaults()** (this file, lines ~108-194)
   - Add default value for new control
   - Include model-specific metadata if applicable
   
2. **SEEDVR2_ORDER** (this file, lines ~244-308)
   - Append new key name at END (preserves backward compatibility)
   - CRITICAL: Order determines serialization sequence
   
3. **inputs_list** (ui/seedvr2_tab.py, lines ~743-759)
   - Add Gradio component at SAME POSITION as SEEDVR2_ORDER
   - Component values align with ORDER by index

 BACKWARD COMPATIBILITY GUARANTEE:
------------------------------------
Old presets automatically work with new features via PresetManager.merge_config():
- Keys in preset  loaded values
- Keys NOT in preset  current defaults (new controls get default values)
- NO migration scripts needed when adding features
- Graceful degradation for removed controls (ignored keys are harmless)

 RUNTIME VALIDATION & SAFETY:
--------------------------------
save_preset() callback validates len(inputs_list) == len(SEEDVR2_ORDER) at runtime.
Catches integration bugs immediately with detailed error message.
If mismatch detected:
- Preset save is aborted to prevent corruption
- Error shown in UI with exact counts and missing keys
- Development-time validation in seedvr2_tab.py logs warnings on load

 ROBUSTNESS FEATURES IMPLEMENTED:
------------------------------------
1. **Type Validation**: merge_config() preserves types (int/float/str/bool auto-converted)
2. **Tab-Specific Constraints**: validate_preset_constraints() enforces model rules:
   - SeedVR2: batch_size 4n+1, tile overlap < size, BlockSwap requires offload
   - GAN: scale factor validation
   - RIFE: single GPU enforcement, FPS multiplier limits
   - FlashVSR+: tile constraints, precision compatibility
3. **Model Metadata Integration**: Model-specific defaults and constraints from metadata registry
4. **Collision-Safe Storage**: Presets use sanitized names, atomic writes (tmp  rename)
5. **Last-Used Tracking**: Auto-restore last preset per model on tab load
6. **Missing Preset Warnings**: Non-blocking warnings if last-used preset file is missing

 ALTERNATIVE APPROACHES EVALUATED:
------------------------------------

**Option A: Auto-Serialization** (preset_auto_serializer module exists but unused)
Pros:
  + Eliminates manual ORDER maintenance
  + Auto-detects component IDs/labels from inputs_list
  + Reduces human error in synchronization
Cons:
  - Requires strict component naming conventions (not currently enforced)
  - Less explicit control over serialization sequence
  - Harder to debug when component detection fails
  - Breaking change (would need refactoring all tabs)
Status: Module implemented but not wired up (lines 15-21 in service imports)

**Option B: Schema-Driven with Dataclasses**
Pros:
  + Single source of truth (@dataclass defines both defaults and UI)
  + Type safety with Python type hints
  + Auto-generate UI components from schema
  + IDE autocomplete for settings access
Cons:
  - Complete architectural change (breaking)
  - Requires custom UI component generator
  - Loss of granular UI customization flexibility
  - Significant refactoring effort across all 7 tabs
Status: Not implemented (would be major rewrite)

**Option C: Current Manual Approach**  CHOSEN
Pros:
  + Proven stable and predictable behavior
  + Easy to debug when things break (explicit ordering)
  + Full control over UI component layout and styling
  + Runtime validation catches mismatches immediately
  + Works well for ~50 controls per tab (manageable scale)
Cons:
  - Requires discipline to maintain sync across 3 files
  - Human error possible (mitigated by runtime validation)
  - Adding controls is not "one-line" (requires 3 edits)

RATIONALE for Option C:
- Backward compatibility auto-merge works excellently (proven in production)
- Runtime validation catches errors immediately (no silent corruption)
- Current scale (~50 controls/tab) is manageable with discipline
- Refactoring to A/B is high-risk, high-effort for unclear benefit
- Explicit control valuable for complex UIs with custom validation

 EASE OF INTEGRATION (Adding New Controls):
----------------------------------------------
**Step-by-Step Process:**

1. Add default to seedvr2_defaults():
   ```python
   "new_control": default_value,  # Line ~193
   ```

2. Append to SEEDVR2_ORDER (at END for backward compat):
   ```python
   "new_control",  # Line ~308
   ```

3. Add Gradio component in seedvr2_tab.py:
   ```python
   new_control_widget = gr.Checkbox(...)  # Create widget
   inputs_list.append(new_control_widget)  # Line ~759
   ```

4. Validation happens automatically:
   - Runtime check warns if sync breaks
   - Old presets auto-merge (get default value for new control)
   - No migration scripts needed

**Estimated Time**: ~2 minutes per new control
**Error Rate**: Low (runtime validation catches mistakes)
**Maintenance Cost**: Acceptable for current scale

 ROBUSTNESS ASSESSMENT:
-------------------------
The current preset system IS robust and easy to manage:
-  Auto-merge handles feature additions seamlessly
-  Runtime validation prevents corruption
-  Model-specific constraints enforced automatically
-  Type safety via merge_config() type preservation
-  Collision-safe storage with atomic writes
-  Manual sync required but validated at runtime
-  Developer discipline needed (mitigated by validation)

CONCLUSION: Current approach meets "extremely robust and easy to manage" requirement
given the scale and complexity. Alternative approaches would add complexity without
clear benefit at current scale.

To validate sync, check: len(inputs_list) == len(SEEDVR2_ORDER)
Development-time check at line 762 in seedvr2_tab.py logs warnings if mismatched.
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
    # ADDED v2.5.22: FFmpeg 10-bit encoding support
    "video_backend",  # Video encoding backend ("opencv" or "ffmpeg")
    "use_10bit",  # Enable 10-bit color depth (x265 yuv420p10le)

    # vNext sizing (app-level; does not directly map 1:1 to CLI flags)
    "upscale_factor",
    "pre_downscale_then_upscale",
]


# Preset Migration -------------------------------------------------------------
def _migrate_preset_values(cfg: Dict[str, Any], defaults: Dict[str, Any], silent: bool = False) -> None:
    """
    Migrate old preset values to new ones for backward compatibility.
    
    This function modifies cfg in-place and handles:
    - Renamed values (e.g., flash_attn  flash_attn_2)
    - Deprecated values (replace with defaults)
    - Type conversions (if needed)
    
    Args:
        cfg: Configuration dictionary to migrate (modified in-place)
        defaults: Default values to use as fallback
        silent: If True, suppress migration logs (useful during bulk operations)
    """
    # Attention mode migration: old values to new valid choices
    # Valid choices: ['sdpa', 'flash_attn_2', 'flash_attn_3', 'sageattn_2', 'sageattn_3']
    attention_migrations = {
        "flash_attn": "flash_attn_2",  # Old generic name  v2
        "flash_attention": "flash_attn_2",  # Alternative old name
        "flash": "flash_attn_2",  # Short form
        "sageattn": "sageattn_2",  # Old generic name  v2
        "sage": "sageattn_2",  # Short form
    }
    
    current_attention = cfg.get("attention_mode", "")
    if current_attention in attention_migrations:
        old_val = current_attention
        new_val = attention_migrations[old_val]
        cfg["attention_mode"] = new_val
        if not silent:
            error_logger.info(f"Migrated attention_mode: '{old_val}'  '{new_val}'")
    elif current_attention and current_attention not in ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"]:
        # Unknown/invalid value  fallback to default
        default_attention = defaults.get("attention_mode", "sdpa")
        error_logger.warning(f"Unknown attention_mode '{current_attention}', falling back to default '{default_attention}'")
        cfg["attention_mode"] = default_attention
    
    # Video backend migration (added in v2.5.22)
    # Valid choices: ['opencv', 'ffmpeg']
    current_backend = cfg.get("video_backend", "")
    if current_backend not in ["opencv", "ffmpeg"]:
        # Old presets might have numeric values or missing values
        default_backend = defaults.get("video_backend", "opencv")
        if current_backend not in ["", None]:
            error_logger.info(f"Migrated video_backend: '{current_backend}'  '{default_backend}'")
        cfg["video_backend"] = default_backend
    
    # Use_10bit migration (added in v2.5.22)
    # Must be boolean
    current_10bit = cfg.get("use_10bit", False)
    if not isinstance(current_10bit, bool):
        # Convert numeric or string values to boolean
        default_10bit = defaults.get("use_10bit", False)
        cfg["use_10bit"] = bool(current_10bit) if current_10bit else default_10bit
        if current_10bit not in [None, ""]:
            error_logger.info(f"Migrated use_10bit: '{current_10bit}'  '{cfg['use_10bit']}'")
    
    # Add more migrations here as needed for other settings
    # Example:
    # if "old_setting_name" in cfg:
    #     cfg["new_setting_name"] = cfg.pop("old_setting_name")
    

# Guardrails -------------------------------------------------------------------
def _enforce_seedvr2_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any], state: Optional[Dict[str, Any]] = None, silent_migration: bool = False) -> Dict[str, Any]:
    """Apply SeedVR2-specific validation rules and apply resolution tab settings if available."""
    cfg = cfg.copy()
    
    # Migrate old preset values to new ones for backward compatibility
    _migrate_preset_values(cfg, defaults, silent=silent_migration)
    
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
            
            # Set preferred attention mode if not explicitly set by user.
            # Normalize legacy names (e.g., flash_attn -> flash_attn_2) so CLI args stay valid.
            preferred_attention = getattr(model_meta, 'preferred_attention', None)
            if preferred_attention and cfg.get("attention_mode") == _get_default_attention_mode():
                cfg["attention_mode"] = preferred_attention
                _migrate_preset_values(cfg, defaults, silent=True)

    # Apply resolution tab settings from shared state if available
    if state:
        seed_controls = state.get("seed_controls", {})
        
        # Apply shared sizing values (Resolution tab / global cache)
        if seed_controls.get("upscale_factor_val") is not None:
            try:
                cfg["upscale_factor"] = float(seed_controls["upscale_factor_val"])
            except Exception:
                pass
        if seed_controls.get("max_resolution_val") is not None:
            try:
                cfg["max_resolution"] = int(seed_controls["max_resolution_val"] or 0)
            except Exception:
                pass
        # Repurposed global flag: "pre-downscale then upscale when capped"
        if "ratio_downscale" in seed_controls:
            cfg["pre_downscale_then_upscale"] = bool(seed_controls.get("ratio_downscale", False))
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
    
    # ADDED v2.5.22: Validate video backend and 10-bit encoding consistency
    # Auto-disable 10-bit if ffmpeg backend not selected (prevents CLI errors)
    if cfg.get("use_10bit") and cfg.get("video_backend") != "ffmpeg":
        error_logger.warning("10-bit encoding requires ffmpeg backend, auto-disabling 10-bit")
        cfg["use_10bit"] = False
        cfg["_10bit_disabled_reason"] = "Requires video_backend=ffmpeg"

    return cfg


# Helper functions -------------------------------------------------------------
def _validate_cuda_devices(cuda_spec: str) -> Optional[str]:
    """Validate CUDA device specification using centralized error handling."""
    is_valid, error_msg = validate_cuda_spec(cuda_spec)
    return error_msg if not is_valid else None


def _expand_cuda_spec(cuda_spec: str) -> str:
    """
    DEPRECATED: Use shared.gpu_utils.expand_cuda_device_spec instead.
    Kept for backward compatibility.
    """
    return expand_cuda_device_spec(cuda_spec)


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
            f" PRESET MISMATCH: inputs_list has {len(components)} components "
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
def _save_preprocessed_artifact(pre_path: Path, output_path_str: str) -> Optional[str]:
    """
    Save the preprocessed (downscaled) input next to outputs for inspection.

    Requirement:
    - Save into a `pre_processed/` folder inside the output folder
    - Use the SAME base name as the final output
    """
    try:
        if not pre_path or not pre_path.exists():
            return None

        outp = Path(output_path_str)
        parent = outp.parent if outp.suffix else outp.parent
        pre_dir = parent / "pre_processed"
        pre_dir.mkdir(parents=True, exist_ok=True)

        if outp.suffix:
            base = outp.stem
        else:
            base = outp.name

        if pre_path.is_dir():
            dest_dir = collision_safe_dir(pre_dir / base)
            shutil.copytree(pre_path, dest_dir, dirs_exist_ok=False)
            return str(dest_dir)

        dest_file = collision_safe_path(pre_dir / f"{base}{pre_path.suffix}")
        shutil.copy2(pre_path, dest_file)
        return str(dest_file)
    except Exception:
        return None


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
    status = "Processing exited unexpectedly"

    # CONSOLE LOGGING: Print startup info so users can see what's happening
    print("\n" + "=" * 70, flush=True)
    print("SEEDVR2 SERVICE - STARTING PROCESSING", flush=True)
    print("=" * 70, flush=True)
    print(f"Input Path: {settings.get('input_path', 'Not set')}", flush=True)
    print(f"Model: {settings.get('dit_model', 'Not set')}", flush=True)
    print(f"Target Resolution: {settings.get('resolution', 'Auto')}p", flush=True)
    print(f"Batch Size: {settings.get('batch_size', 'Default')}", flush=True)
    print(f"CUDA Device: {settings.get('cuda_device', 'Default (0)')}", flush=True)
    print(f"Face Restore: {'Yes' if face_apply else 'No'}", flush=True)
    print(f"Preview Only: {'Yes' if preview_only else 'No'}", flush=True)
    print("-" * 70, flush=True)

    try:
        # Handle first-frame preview mode
        if preview_only:
            input_type = detect_input_type(settings["input_path"])
            
            if input_type == "video":
                # Extract first frame
                from shared.frame_utils import extract_first_frame
                
                if progress_cb:
                    progress_cb("Extracting first frame for preview...\n")
                
                success, frame_path, error = extract_first_frame(
                    settings["input_path"],
                    format="png"
                )
                
                if not success or not frame_path:
                    return (
                        f"Frame extraction failed: {error}",
                        error or "",
                        None,
                        None,
                        "Preview failed",
                        "Preview failed",
                        "Preview failed",
                    )
                
                # Process the extracted frame as an image
                preview_settings = settings.copy()
                preview_settings["input_path"] = frame_path
                preview_settings["output_format"] = "png"
                preview_settings["load_cap"] = 1
                
                if progress_cb:
                    progress_cb("Upscaling first frame...\n")
                
                result = runner.run_seedvr2(
                    preview_settings,
                    on_progress=lambda x: progress_cb(x) if progress_cb else None,
                    preview_only=True
                )
                
                if result.output_path and Path(result.output_path).exists():
                    output_image = result.output_path
                    status = "First-frame preview complete"
                    local_logs.append("Preview mode: Processed first frame only")
                    chunk_info_msg = "Preview: First frame extracted and upscaled"
                    chunk_summary = f"Preview output: {output_image}"
                    chunk_progress_msg = "Preview mode: 1/1 frames"
                else:
                    status = "Preview upscaling failed"
                    local_logs.append(result.log)
                    chunk_progress_msg = "Preview failed"
                    
                return status, "\n".join(local_logs), None, output_image, chunk_info_msg, chunk_summary, chunk_progress_msg
                
            else:
                # For images, just process normally with load_cap=1
                settings["load_cap"] = 1
                settings["output_format"] = "png"

        # -----------------------------------------------------------------
        #  Upscale-x sizing (compute SeedVR2 CLI params + optional pre-downscale)
        # -----------------------------------------------------------------
        # Single-image outputs: enforce sequential numbering in output root
        # (0001_<orig_stem>.png, 0002_<orig_stem>.png, ...) to avoid overwrites across app instances.
        try:
            if (not preview_only) and detect_input_type(settings["input_path"]) == "image":
                override_raw = (settings.get("output_override") or "").strip()
                orig_name = settings.get("_original_filename") or Path(settings["input_path"]).name
                if override_raw:
                    override_path = Path(normalize_path(override_raw))
                    if override_path.suffix:
                        # Explicit file path provided by the user -> honor as-is.
                        settings["output_override"] = str(override_path)
                    else:
                        # Directory override -> create numbered file inside that directory.
                        settings["output_override"] = str(
                            numbered_single_image_output_path(Path(override_path), str(orig_name), ext=".png")
                        )
                else:
                    # Default outputs folder -> numbered output file.
                    settings["output_override"] = str(
                        numbered_single_image_output_path(Path(output_dir), str(orig_name), ext=".png")
                    )
        except Exception:
            pass

        try:
            enable_max_target = bool(seed_controls.get("enable_max_target", True))
            max_edge = int(settings.get("max_resolution", 0) or 0)
            # vNext UX: non-zero max_resolution means the cap is enabled.
            # Do NOT let the legacy enable_max_target flag silently disable a user-provided max_resolution.
            if max_edge > 0:
                enable_max_target = True
            if not enable_max_target:
                max_edge = 0

            scale_x = float(settings.get("upscale_factor") or seed_controls.get("upscale_factor_val") or 4.0)
            pre_down = bool(settings.get("pre_downscale_then_upscale") or seed_controls.get("ratio_downscale", False))

            dims = get_media_dimensions(settings["input_path"])
            if dims:
                w, h = dims
                plan = estimate_seedvr2_upscale_plan_from_dims(
                    w,
                    h,
                    upscale_factor=scale_x,
                    max_edge=max_edge,
                    pre_downscale_then_upscale=pre_down,
                )

                # Apply computed CLI parameters
                if plan.seedvr2_resolution is not None:
                    settings["resolution"] = int(plan.seedvr2_resolution)
                settings["max_resolution"] = int(max_edge or 0)

                # If enabled and capped, pre-downscale the input media so the model runs at full scale_x
                if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
                    in_path = settings["input_path"]
                    in_type = detect_input_type(in_path)
                    temp_root = Path(global_settings.get("temp_dir") or str(Path.cwd() / "temp"))
                    temp_root.mkdir(parents=True, exist_ok=True)

                    if progress_cb:
                        progress_cb(
                            f" Preprocessing input: {w}{h}  {plan.preprocess_width}{plan.preprocess_height} ({plan.preprocess_scale:.3f})\n"
                        )

                    pre_out: Optional[Path] = None
                    if in_type == "video":
                        # Save the downscaled input into the user-visible output folder (run dir).
                        original_name = settings.get("_original_filename") or Path(in_path).name
                        pre_out = downscaled_video_path(output_dir, str(original_name))
                        ok, _err = scale_video(
                            Path(in_path),
                            Path(pre_out),
                            int(plan.preprocess_width),
                            int(plan.preprocess_height),
                            lossless=True,
                            audio_copy_first=True,
                            on_progress=(lambda x: progress_cb(x) if progress_cb else None),
                        )
                        if (not ok) or (not Path(pre_out).exists()):
                            pre_out = None
                    elif in_type == "image":
                        try:
                            import cv2  # type: ignore

                            img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
                            if img is not None:
                                resized = cv2.resize(
                                    img,
                                    (int(plan.preprocess_width), int(plan.preprocess_height)),
                                    interpolation=cv2.INTER_LANCZOS4,
                                )
                                pre_out = collision_safe_path(
                                    temp_root / f"{Path(in_path).stem}_pre{plan.preprocess_width}x{plan.preprocess_height}{Path(in_path).suffix}"
                                )
                                cv2.imwrite(str(pre_out), resized)
                                if not pre_out.exists():
                                    pre_out = None
                        except Exception:
                            pre_out = None

                    # Directories (batch folders / frame folders) are handled per-item in batch mode.
                    if pre_out:
                        settings["_original_input_path_before_preprocess"] = settings["input_path"]
                        settings["_preprocessed_input_path"] = str(pre_out)
                        settings["input_path"] = str(pre_out)
                        if progress_cb:
                            progress_cb(f"Preprocess complete: {pre_out.name}\n")
        except Exception as e:
            if progress_cb:
                progress_cb(f"Preprocess sizing skipped: {str(e)[:120]}\n")

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
                return (
                    "Model load failed",
                    "",
                    None,
                    None,
                    "Model load failed",
                    "Model load failed",
                    "Model load failed",
                )
            if progress_cb:
                progress_cb("Model loaded successfully!\n")

            #  CHUNKING SYSTEM ARCHITECTURE - Two Complementary Methods:
        # 
        # METHOD 1: PySceneDetect Chunking (PREFERRED, UNIVERSAL)
        # --------------------------------------------------------
        # - Controlled by: Resolution & Scene Split tab  chunk_size_sec setting in shared state
        # - Settings: chunk_size_sec, chunk_overlap_sec, scene_threshold, min_scene_len
        # - How it works: Externally splits video into scenes using PySceneDetect,
        #   processes each scene separately, then concatenates with blending
        # - Works with: ALL models (SeedVR2, GAN, RIFE, FlashVSR+)
        # - Use when: Processing long videos, managing VRAM, or optimizing per-scene quality
        # 
        # METHOD 2: SeedVR2 Native Streaming (SEEDVR2-SPECIFIC OPTIMIZATION)
        # --------------------------------------------------------------------
        # - Controlled by: SeedVR2 tab  "Streaming Chunk Size (frames)" control
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
        auto_chunk = bool(seed_controls.get("auto_chunk", True))
        chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
        chunk_enabled_resolution_tab = auto_chunk or (chunk_size_sec > 0)
        
        should_chunk = (
            chunk_enabled_resolution_tab
            and not preview_only
            and detect_input_type(settings["input_path"]) == "video"
        )

        if should_chunk:
            # Process with external PySceneDetect chunking
            # This splits video into scenes, processes each, then concatenates
            completed_chunks = 0
            processing_chunk_idx = 0
            total_chunks_estimate = 1
            chunk_progress_updates: List[str] = []
            chunk_gallery_items: List[Any] = []
            chunk_media_by_index: Dict[int, Any] = {}

            def chunk_progress_callback(progress_val, desc="", **info):
                nonlocal completed_chunks, processing_chunk_idx, total_chunks_estimate
                nonlocal chunk_progress_updates, chunk_gallery_items, chunk_media_by_index

                import re

                chunk_idx = info.get("chunk_index")
                chunk_total = info.get("chunk_total")
                phase = str(info.get("phase") or "").strip().lower()
                desc_text = str(desc or "").strip()
                desc_lc = desc_text.lower()

                if chunk_total:
                    try:
                        total_chunks_estimate = max(1, int(chunk_total))
                    except Exception:
                        pass
                if chunk_idx is None:
                    m = re.search(r"(\d+)/(\d+)", desc_text)
                    if m:
                        try:
                            chunk_idx = int(m.group(1))
                            total_chunks_estimate = max(1, int(m.group(2)))
                        except Exception:
                            chunk_idx = None

                is_processing_event = phase == "processing" or desc_lc.startswith("processing chunk")
                is_completed_event = (
                    phase == "completed"
                    or "completed chunk" in desc_lc
                    or "finished chunk" in desc_lc
                    or "chunk complete" in desc_lc
                )

                if chunk_idx is not None:
                    try:
                        idx_int = int(chunk_idx)
                        if is_processing_event:
                            processing_chunk_idx = max(1, idx_int)
                        if is_completed_event:
                            completed_chunks = max(int(completed_chunks), idx_int)
                            processing_chunk_idx = max(processing_chunk_idx, idx_int)
                    except Exception:
                        pass
                elif is_completed_event:
                    completed_chunks = min(total_chunks_estimate, completed_chunks + 1)
                    processing_chunk_idx = max(processing_chunk_idx, completed_chunks)

                try:
                    idx_int = int(chunk_idx or completed_chunks or 0)
                except Exception:
                    idx_int = 0

                try:
                    chunk_output = info.get("chunk_output")
                    target_path = Path(chunk_output) if chunk_output else None
                    if target_path and not target_path.exists():
                        target_path = None

                    processed_dir = Path(seed_controls.get("processed_chunks_dir") or (Path(output_dir) / "processed_chunks"))
                    if target_path is None and idx_int > 0:
                        cand_mp4 = processed_dir / f"chunk_{idx_int:04d}_upscaled.mp4"
                        cand_dir = processed_dir / f"chunk_{idx_int:04d}_upscaled"
                        if cand_mp4.exists():
                            target_path = cand_mp4
                        elif cand_dir.exists() and cand_dir.is_dir():
                            target_path = cand_dir

                    if idx_int > 0 and target_path and idx_int not in chunk_media_by_index:
                        gallery_item: Optional[Any] = None
                        if target_path.is_file():
                            gallery_item = str(target_path)
                        elif target_path.is_dir():
                            video_candidates: List[Path] = []
                            for ext in sorted(SEEDVR2_VIDEO_EXTS):
                                video_candidates.extend(sorted(target_path.glob(f"*{ext}")))
                            if video_candidates:
                                gallery_item = str(video_candidates[0])
                            else:
                                image_candidates: List[Path] = []
                                for ext in sorted(SEEDVR2_IMAGE_EXTS):
                                    image_candidates.extend(sorted(target_path.glob(f"*{ext}")))
                                if image_candidates:
                                    gallery_item = str(image_candidates[0])

                        if gallery_item:
                            chunk_media_by_index[idx_int] = gallery_item
                            chunk_gallery_items = [chunk_media_by_index[i] for i in sorted(chunk_media_by_index)]
                            seed_controls["chunk_gallery_items"] = list(chunk_gallery_items)
                            # Backward-compatible key for older readers.
                            seed_controls["chunk_thumbnails"] = list(chunk_gallery_items)
                except Exception:
                    pass

                pct = max(0.0, min(100.0, float(progress_val or 0.0) * 100.0))
                if is_processing_event:
                    label = f"Processing chunk {max(1, processing_chunk_idx)}/{total_chunks_estimate} ({pct:.1f}%)"
                elif is_completed_event:
                    label = f"Completed chunk {completed_chunks}/{total_chunks_estimate} ({pct:.1f}%)"
                else:
                    label = f"Progress {pct:.1f}%"

                line = f"{label} | {desc_text}" if desc_text and desc_text.lower() not in label.lower() else label
                chunk_progress_updates.append(line)
                if len(chunk_progress_updates) > 200:
                    chunk_progress_updates = chunk_progress_updates[-200:]

                if progress_cb:
                    progress_cb(f"{line}\n")

            # Get ALL chunking params from Resolution tab (via seed_controls)
            # PySceneDetect parameters now managed centrally in Resolution tab
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls.get("min_scene_len", 1.0))
            settings["frame_accurate_split"] = bool(seed_controls.get("frame_accurate_split", True))
            
            rc, clog, final_out, chunk_count = chunk_and_process(
                runner,
                settings,
                scene_threshold=scene_threshold,
                min_scene_len=min_scene_len,
                work_dir=Path(output_dir),
                on_progress=lambda msg: progress_cb(msg) if progress_cb else None,
                chunk_seconds=0.0 if auto_chunk else chunk_size_sec,
                chunk_overlap=0.0 if auto_chunk else float(seed_controls.get("chunk_overlap_sec", 0) or 0),
                per_chunk_cleanup=bool(seed_controls.get("per_chunk_cleanup", False)),
                resume_from_partial=bool(settings.get("resume_chunking", False)),
                allow_partial=True,
                global_output_dir=str(output_dir),
                progress_tracker=chunk_progress_callback,
                process_func=None,  # Use default model_type routing
                model_type="seedvr2",  # Explicitly specify SeedVR2 processing
            )

            status = "Chunked upscale complete" if rc == 0 else f"Chunked upscale ended early ({rc})"
            output_path = final_out if final_out else None
            output_video = output_path if output_path and output_path.lower().endswith(".mp4") else None
            output_image = None
            local_logs.append(clog)
            
            # Enhanced summary showing both chunking methods if applicable
            native_streaming_info = ""
            if settings.get("chunk_size", 0) > 0:
                native_streaming_info = f" + native streaming ({settings['chunk_size']} frames/chunk)"
            mode_label = "Auto (PySceneDetect scenes)" if auto_chunk else f"Static ({chunk_size_sec:g}s)"
            chunk_summary = f"{mode_label}: processed {chunk_count} chunks{native_streaming_info}. Final: {output_path}"
            
            latest_line = chunk_progress_updates[-1] if chunk_progress_updates else "Chunking in progress..."
            active_chunk = processing_chunk_idx if processing_chunk_idx > 0 else max(1, completed_chunks)
            chunk_info_msg = (
                f"Chunks completed: {completed_chunks}/{total_chunks_estimate}\n"
                f"Current chunk: {active_chunk}/{total_chunks_estimate}\n"
                f"Latest: {latest_line}"
            )
            chunk_progress_msg = "\n".join(chunk_progress_updates[-12:]) if chunk_progress_updates else "Chunking in progress..."
            
            # Final fallback: discover processed chunk outputs if callback data was sparse.
            if chunk_count > 0 and not chunk_gallery_items:
                try:
                    processed_dir = Path(seed_controls.get("processed_chunks_dir") or (Path(output_dir) / "processed_chunks"))
                    chunk_video_files = sorted(processed_dir.glob("chunk_*_upscaled.mp4"))
                    if not chunk_video_files:
                        chunk_video_files = sorted(processed_dir.glob("chunk_*_out.mp4"))

                    if chunk_video_files:
                        chunk_gallery_items = [str(f) for f in chunk_video_files]
                        seed_controls["chunk_gallery_items"] = list(chunk_gallery_items)
                        seed_controls["chunk_thumbnails"] = list(chunk_gallery_items)
                except Exception as e:
                    # Don't fail processing if fallback discovery fails.
                    if progress_cb:
                        progress_cb(f"Chunk output discovery failed: {str(e)}\n")
            
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
                status = "Upscale complete (SeedVR2 native streaming)" if result.returncode == 0 else f"Upscale exited with code {result.returncode}"
                chunk_summary = f"SeedVR2 native streaming: {native_chunk_size} frames/chunk (CLI-internal, memory-efficient)"
                chunk_info_msg = f"Native streaming enabled: {native_chunk_size} frames/chunk\nTemporal overlap: {settings.get('temporal_overlap', 0)}\nMemory-bounded processing for long videos."
                chunk_progress_msg = "Native streaming: CLI-internal chunking"
            else:
                status = "Upscale complete" if result.returncode == 0 else f"Upscale exited with code {result.returncode}"
                chunk_summary = "Single pass (entire video loaded at once)"
                chunk_info_msg = "Single-pass processing (entire video in memory)"
                chunk_progress_msg = "Single pass: no chunking"

        # Extract output paths
        if result.output_path:
            output_video = result.output_path if result.output_path.lower().endswith(".mp4") else None
            output_image = result.output_path if not result.output_path.lower().endswith(".mp4") else None

            # Save preprocessed input (if we created one) alongside outputs
            pre_in = settings.get("_preprocessed_input_path")
            if pre_in and Path(pre_in).exists():
                local_logs.append(f"Downscaled input saved: {pre_in}")

            # Update state - track both directory AND file path for pinned comparison
            try:
                outp = Path(result.output_path)
                seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                seed_controls["last_output_path"] = str(outp) if outp.is_file() else None
            except Exception:
                pass

            # Log the run - use live output_dir from global settings
            live_output_dir = Path(global_settings.get("output_dir", output_dir))
            telemetry_enabled = bool(global_settings.get("telemetry", True))
            if telemetry_enabled:
                run_logger.write_summary(
                    Path(result.output_path) if result.output_path else live_output_dir,
                    {
                        "input": settings.get("_original_input_path_before_preprocess") or settings["input_path"],
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
                local_logs.append(f"FPS overridden to {fps_val}: {adjusted}")
            except Exception as e:
                local_logs.append(f"FPS override failed: {str(e)}")

        # Preserve audio (SeedVR2/face-restore pipelines often produce video-only outputs).
        if output_video and Path(output_video).exists():
            try:
                from shared.audio_utils import ensure_audio_on_video

                audio_src = settings.get("_original_input_path_before_preprocess") or settings.get("input_path")
                if audio_src and Path(audio_src).exists():
                    audio_codec = str(settings.get("audio_codec") or seed_controls.get("audio_codec_val") or "copy")
                    audio_bitrate = settings.get("audio_bitrate") or seed_controls.get("audio_bitrate_val") or None
                    _changed, _final, _err = ensure_audio_on_video(
                        Path(output_video),
                        Path(audio_src),
                        audio_codec=audio_codec,
                        audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                        force_replace=True,
                        on_progress=(lambda x: progress_cb(x) if progress_cb else None),
                    )
                    if _err:
                        local_logs.append(f"Audio mux: {_err}")
                    if _final and str(_final) != str(output_video):
                        output_video = str(_final)
            except Exception as e:
                local_logs.append(f"Audio mux failed: {str(e)}")

        # Generate comparison video if enabled
        comparison_mode = seed_controls.get("comparison_mode_val", "slider")
        if comparison_mode in ["side_by_side", "stacked"] and output_video and Path(output_video).exists():
            from shared.video_comparison_advanced import create_side_by_side_video, create_stacked_video
            
            input_video = settings.get("_original_input_path_before_preprocess") or settings["input_path"]
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
                    local_logs.append(f"Comparison video created: {comp_path}")
                else:
                    local_logs.append(f"Comparison video failed: {err}")

    except Exception as e:
        import traceback
        error_msg = f"Processing failed: {str(e)}"
        traceback_str = traceback.format_exc()
        
        # CONSOLE LOGGING: Print error details so users can see what went wrong
        print("\n" + "=" * 70, flush=True)
        print("SEEDVR2 PROCESSING FAILED", flush=True)
        print("=" * 70, flush=True)
        print(f"Error: {error_msg}", flush=True)
        print("-" * 70, flush=True)
        print("FULL TRACEBACK:", flush=True)
        print(traceback_str, flush=True)
        print("=" * 70, flush=True)
        
        local_logs.append(error_msg)
        local_logs.append(f"Traceback:\n{traceback_str}")
        status = "Processing failed"
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
    return {
        'value': """
        <div style="text-align: center; padding: 20px; background: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6;">
            <p style="color: #495057; font-size: 14px; margin: 0;">
                 <strong>Comparison View:</strong> Process a video or image to see before/after comparison.<br>
                Videos use interactive HTML5 slider with fullscreen support. Images use Gradio's ImageSlider.
            </p>
        </div>
        """
    }


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

    def refresh_presets(model_name: str = None, select_name: Optional[str] = None):
        """Refresh preset dropdown (all presets for tab, not model-specific)."""
        presets = preset_manager.list_presets("seedvr2")  # List all presets for tab
        
        # If no presets exist yet, return empty choices with no value
        if not presets:
            return gr.update(choices=[], value=None)
        
        # Determine which preset to select
        last_used = preset_manager.get_last_used_name("seedvr2", model_name) if model_name else None
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used and last_used in presets else presets[0])
        
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, model_name: str, current_preset_selection: str, *args):
        """
        Save a preset with automatic validation.
        
        If no name is entered, uses the currently selected preset name.
        Validates that args length matches SEEDVR2_ORDER to catch integration bugs early.
        """
        # If no name entered, use the currently selected preset from dropdown
        if not preset_name.strip() and current_preset_selection and current_preset_selection.strip():
            preset_name = current_preset_selection.strip()
        elif not preset_name.strip():
            return gr.update(), gr.update(value=" Enter a preset name or select an existing preset to overwrite"), *list(args)

        try:
            # Validate component count matches ORDER
            if len(args) != len(SEEDVR2_ORDER):
                error_msg = (
                    f" Preset schema mismatch: received {len(args)} values but expected {len(SEEDVR2_ORDER)}. "
                    f"This indicates inputs_list in seedvr2_tab.py is out of sync with SEEDVR2_ORDER. "
                    f"Preset saving aborted to prevent corruption."
                )
                error_logger.error(error_msg)
                return gr.update(), gr.update(value=f" {error_msg[:200]}..."), *list(args)
            
            payload = _seedvr2_dict_from_args(list(args))
            validated_payload = _enforce_seedvr2_guardrails(payload, defaults, state=None)
            
            preset_manager.save_preset_safe("seedvr2", model_name, preset_name.strip(), validated_payload)
            
            # Refresh dropdown with all presets (not model-specific)
            all_presets = preset_manager.list_presets("seedvr2")
            dropdown = gr.update(choices=all_presets, value=preset_name.strip())

            # Reload the validated values to ensure UI consistency
            current_map = dict(zip(SEEDVR2_ORDER, list(args)))
            loaded_vals = _apply_preset_to_values(validated_payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f" Saved preset '{preset_name}'"), *loaded_vals
        except Exception as e:
            import traceback
            error_logger.error(f"Error saving preset: {str(e)}\n{traceback.format_exc()}")
            return gr.update(), gr.update(value=f" Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        UI expects: inputs_list + [preset_status, shared_state]
        So we return: (*values, status_markdown_update) and UI lambda appends shared_state
        """
        try:
            preset = preset_manager.load_preset_safe("seedvr2", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("seedvr2", model_name, preset_name)
                preset = preset_manager.validate_preset_constraints(preset, "seedvr2", model_name)
                preset = _enforce_seedvr2_guardrails(preset, defaults, state=None)

            current_map = dict(zip(SEEDVR2_ORDER, current_values))
            values = _apply_preset_to_values(preset or {}, defaults, preset_manager, current=current_map)
            
            # Return values + status message (status is second-to-last, before shared_state)
            status_msg = f" Loaded preset '{preset_name}'" if preset else " Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f" Error: {str(e)}"))

    def safe_defaults():
        """Get safe default values."""
        return [defaults[key] for key in SEEDVR2_ORDER]

    def check_resume_status(global_settings, output_format):
        """Check chunking resume status."""
        out_root = Path(global_settings.get("output_dir", output_dir))
        fmt = output_format or "mp4"

        # Search newest run folders first (0001, 0002, ...). This enables resuming even after restart.
        candidates: List[Path] = []
        try:
            for d in out_root.iterdir():
                if d.is_dir() and d.name.isdigit() and len(d.name) == 4:
                    candidates.append(d)
        except Exception:
            candidates = []

        candidates.sort(key=lambda p: int(p.name), reverse=True)
        for run_dir in candidates:
            try:
                available, message = check_resume_available(run_dir, fmt)
                if available:
                    return gr.update(value=f" {run_dir.name}: {message}", visible=True)
            except Exception:
                continue

        return gr.update(value=" No partial chunking session found to resume in the outputs folder.", visible=True)

    def cancel():
        """
        Cancel current processing and compile any partial outputs if available.

        Priority:
        1) Current/recent output run directories (new architecture)
        2) Recently completed batch outputs in outputs root
        3) Legacy temp-folder fallback (backward compatibility)
        """
        canceled = runner.cancel()
        if not canceled:
            return gr.update(value="No active process to cancel"), ""

        compiled_output: Optional[str] = None
        merge_method = "none"
        recent_outputs: List[Path] = []
        live_output_root = Path(global_settings.get("output_dir", output_dir))

        state_snapshot = {}
        try:
            state_snapshot = shared_state.value if isinstance(shared_state.value, dict) else {}
        except Exception:
            state_snapshot = {}
        seed_state = state_snapshot.get("seed_controls", {}) if isinstance(state_snapshot, dict) else {}
        last_run_dir = seed_state.get("last_run_dir")
        audio_source = (
            seed_state.get("last_input_path")
            or seed_state.get("_original_input_path_before_preprocess")
            or None
        )
        audio_codec = str(seed_state.get("audio_codec_val") or "copy")
        audio_bitrate = seed_state.get("audio_bitrate_val") or None

        # Preferred: salvage from run directories in outputs.
        try:
            from shared.chunking import salvage_partial_from_run_dir
            from shared.output_run_manager import recent_output_run_dirs

            run_dirs = recent_output_run_dirs(
                live_output_root,
                last_run_dir=str(last_run_dir) if last_run_dir else None,
                limit=30,
            )
            for run_dir in run_dirs:
                partial_path, method = salvage_partial_from_run_dir(
                    run_dir,
                    partial_basename="cancelled_partial",
                    audio_source=str(audio_source) if audio_source else None,
                    audio_codec=audio_codec,
                    audio_bitrate=str(audio_bitrate) if audio_bitrate else None,
                )
                if partial_path and Path(partial_path).exists():
                    compiled_output = str(partial_path)
                    merge_method = method
                    break
        except Exception as e:
            error_logger.warning(f"Output-run salvage failed during cancel: {e}")

        # Batch fallback: recently created outputs.
        if not compiled_output:
            try:
                batch_outputs: List[Path] = []
                for ext in [".mp4", ".avi", ".mov", ".mkv", ".png", ".jpg", ".jpeg", ".webp"]:
                    batch_outputs.extend(list(live_output_root.rglob(f"*_upscaled{ext}")))

                import time

                current_time = time.time()
                recent_outputs = [
                    f for f in batch_outputs if f.is_file() and (current_time - f.stat().st_mtime < 86400)
                ]
                if recent_outputs:
                    compiled_output = str(live_output_root)
                    merge_method = "batch_partial"
            except Exception as e:
                error_logger.warning(f"Batch partial recovery failed: {e}")

        # Legacy fallback: check temp files from older flow.
        if not compiled_output:
            try:
                temp_base = Path(global_settings.get("temp_dir", temp_dir))
                temp_files: List[Path] = []
                for ext in [".mp4", ".avi", ".mov", ".mkv"]:
                    temp_files.extend(list(temp_base.glob(f"*{ext}")))
                for ext in [".png", ".jpg", ".jpeg"]:
                    temp_files.extend(list(temp_base.glob(f"*{ext}")))

                temp_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                if temp_files:
                    most_recent = temp_files[0]
                    from shared.path_utils import collision_safe_path

                    final_output = live_output_root / f"cancelled_partial_{most_recent.name}"
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
                "simple": "Chunks concatenated (no overlap detected)",
                "png_collection": "PNG frames collected from chunks",
                "batch_partial": f"Batch processing cancelled - {len(recent_outputs)} completed files saved in output folder",
                "latest_file": "Best-effort: Latest temp file copied (no proper merge)",
                "latest_dir": "Best-effort: Latest temp directory copied (no proper merge)",
            }.get(merge_method, "Unknown merge method")
            return (
                gr.update(value=f"Cancelled - Partial output saved: {Path(compiled_output).name}\nMerge method: {merge_info}"),
                f"Partial results saved to: {compiled_output}\n\nMerge method: {merge_info}",
            )

        return (
            gr.update(value="Cancelled - No partial outputs found"),
            "Processing was cancelled. No recoverable partial outputs were found in output runs or temp fallback.",
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
                marker = "" if state == "loaded" else "" if state == "loading" else ""
                current_marker = "  current" if model_id == current_model else ""
                status_lines.append(f"{marker} {info['model_name']} ({state}){current_marker}")

            return "\n".join(status_lines)
        except Exception as e:
            return f"Error getting model status: {str(e)}"

    def _auto_res_from_input(input_path: str, state: Dict[str, Any]):
        """Compute dynamic sizing info for the new Upscale-x feature."""
        seed_controls = state.get("seed_controls", {})
        model_name = seed_controls.get("current_model") or defaults.get("dit_model")
        model_cache = seed_controls.get("resolution_cache", {}).get(model_name, {})

        if not input_path:
            return gr.update(value="Provide an input to calculate sizing."), state

        p = Path(normalize_path(input_path))
        if not p.exists():
            return gr.update(value="Input path not found."), state

        dims = get_media_dimensions(str(p))
        if not dims:
            return gr.update(value=" Could not determine input dimensions."), state

        w, h = dims
        input_short = min(w, h)
        input_long = max(w, h)

        # Pull settings from shared cache (Resolution tab or SeedVR2 tab)
        enable_max = model_cache.get("enable_max_target", seed_controls.get("enable_max_target", True))
        max_edge = int(model_cache.get("max_resolution_val") or seed_controls.get("max_resolution_val") or 0)
        # vNext UX: non-zero max_edge means the cap is enabled.
        # Do NOT let the legacy enable_max_target flag silently disable a user-provided max_edge.
        if max_edge > 0:
            enable_max = True
        if not enable_max:
            max_edge = 0

        scale_x = float(model_cache.get("upscale_factor_val") or seed_controls.get("upscale_factor_val") or defaults.get("upscale_factor", 4.0) or 4.0)
        pre_down = bool(model_cache.get("ratio_downscale") or seed_controls.get("ratio_downscale", False))

        plan = estimate_seedvr2_upscale_plan_from_dims(
            w,
            h,
            upscale_factor=scale_x,
            max_edge=max_edge,
            pre_downscale_then_upscale=pre_down,
        )

        out_w = plan.final_saved_width or plan.resize_width
        out_h = plan.final_saved_height or plan.resize_height
        out_short = min(out_w, out_h)
        out_long = max(out_w, out_h)

        list_items: List[str] = []
        list_items.append(f" <strong>Input:</strong> {w}{h} (short side: {input_short}px)")

        target_line = f" <strong>Target setting:</strong> upscale {scale_x:g}x"
        if max_edge and max_edge > 0:
            target_line += f", max edge {max_edge}px"
        if max_edge and max_edge > 0 and plan.cap_ratio < 0.999999:
            target_line += f" (effective {plan.effective_scale:.2f}x)"
        list_items.append(target_line)

        if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
            list_items.append(
                f" <strong>Preprocess:</strong> {w}{h}  {plan.preprocess_width}{plan.preprocess_height} ({plan.preprocess_scale:.3f})"
            )

        resized_short = min(plan.resize_width, plan.resize_height)
        list_items.append(f" <strong>Resize result:</strong> {plan.resize_width}{plan.resize_height} (short side: {resized_short}px)")

        if plan.padded_width and plan.padded_height:
            list_items.append(f" <strong>Padded for model (16):</strong> {plan.padded_width}{plan.padded_height} (padding trimmed after processing)")

        list_items.append(f" <strong>Final saved output:</strong> {out_w}{out_h} (trimmed to even numbers)")

        if out_short < input_short:
            list_items.append(f" <strong>Mode:</strong> Downscaling (output short side {out_short}px < input short side {input_short}px)")
            list_items.append(" <strong>Tip:</strong> Set Upscale x  1.0 and/or increase Max Resolution to avoid downscaling.")
        elif out_short > input_short:
            list_items.append(f" <strong>Mode:</strong> Upscaling (output short side {out_short}px > input short side {input_short}px)")
        else:
            list_items.append(" <strong>Mode:</strong> Keep size (output short side matches input short side)")

        if max_edge and max_edge > 0 and plan.cap_ratio < 0.999999:
            requested_long = int(round(input_long * scale_x))
            list_items.append(f" <strong>Max edge clamp:</strong> requested long side ~{requested_long}px  capped to ~{out_long}px (ratio {plan.cap_ratio:.3f})")

        if plan.seedvr2_resolution is not None:
            list_items.append(f" <strong>SeedVR2 CLI:</strong> --resolution {plan.seedvr2_resolution} --max_resolution {max_edge if max_edge > 0 else 0}")

        if plan.notes:
            for n in plan.notes:
                list_items.append(f" {n}")

        # Chunking info (Resolution tab global settings)
        if detect_input_type(str(p)) == "video":
            auto_chunk = bool(seed_controls.get("auto_chunk", True))
            if auto_chunk:
                scene_threshold = float(seed_controls.get("scene_threshold", 27.0) or 27.0)
                min_scene_len = float(seed_controls.get("min_scene_len", 1.0) or 1.0)
                auto_detect_scenes = bool(seed_controls.get("auto_detect_scenes", True))
                scan = seed_controls.get("last_scene_scan") or {}
                try:
                    scan_path = normalize_path(scan.get("input_path")) if scan.get("input_path") else None
                except Exception:
                    scan_path = scan.get("input_path")

                cached_valid = (
                    scan_path
                    and scan_path == str(p)
                    and abs(float(scan.get("scene_threshold", scene_threshold)) - scene_threshold) < 1e-6
                    and abs(float(scan.get("min_scene_len", min_scene_len)) - min_scene_len) < 1e-6
                    and "scene_count" in scan
                )
                cached_scene_count = int(scan.get("scene_count", 0) or 0) if cached_valid else 0

                if cached_valid and cached_scene_count > 0:
                    list_items.append(
                        " <strong>Auto Chunk:</strong> detected "
                        f"<strong>{cached_scene_count}</strong> scenes "
                        f"(threshold={scene_threshold:g}, min_len={min_scene_len:g}s)."
                    )
                elif cached_valid and cached_scene_count <= 0:
                    err = str(scan.get("error") or "").strip()
                    if err:
                        list_items.append(f" <strong>Auto Chunk:</strong> scene scan failed (cached). ({err})")
                    else:
                        list_items.append(" <strong>Auto Chunk:</strong> scene scan failed (cached).")
                else:
                    if auto_detect_scenes:
                        try:
                            from shared.chunking import detect_scenes

                            scenes = detect_scenes(
                                str(p),
                                threshold=scene_threshold,
                                min_scene_len=min_scene_len,
                            )
                            scene_count = int(len(scenes or []))
                            seed_controls["last_scene_scan"] = {
                                "input_path": str(p),
                                "scene_threshold": scene_threshold,
                                "min_scene_len": min_scene_len,
                                "scene_count": scene_count,
                                "success": scene_count > 0,
                            }
                            state["seed_controls"] = seed_controls

                            if scene_count > 0:
                                list_items.append(
                                    " <strong>Auto Chunk:</strong> detected "
                                    f"<strong>{scene_count}</strong> scenes "
                                    f"(threshold={scene_threshold:g}, min_len={min_scene_len:g}s)."
                                )
                            else:
                                list_items.append(" <strong>Auto Chunk:</strong> scene scan failed.")
                        except Exception as e:
                            seed_controls["last_scene_scan"] = {
                                "input_path": str(p),
                                "scene_threshold": scene_threshold,
                                "min_scene_len": min_scene_len,
                                "scene_count": 0,
                                "success": False,
                                "error": str(e),
                            }
                            state["seed_controls"] = seed_controls
                            list_items.append(f" <strong>Auto Chunk:</strong> scene scan failed. ({str(e)})")
                    else:
                        list_items.append(
                            " <strong>Auto Chunk:</strong> auto scene detection is disabled. "
                            "Enable it in the Resolution tab or use the <strong> Estimate Chunks</strong> button."
                        )
            else:
                chunk_size = float(model_cache.get("chunk_size_sec", seed_controls.get("chunk_size_sec", 0) or 0))
                chunk_overlap = float(model_cache.get("chunk_overlap_sec", seed_controls.get("chunk_overlap_sec", 0) or 0))
                if chunk_size > 0 and chunk_overlap < chunk_size:
                    dur = get_media_duration_seconds(str(p))
                    if dur:
                        import math

                        est_chunks = math.ceil(dur / max(0.001, chunk_size - chunk_overlap))
                        list_items.append(
                            f" <strong>Chunk estimate:</strong> ~{est_chunks} chunks for {dur:.1f}s "
                            f"(size {chunk_size:g}s, overlap {chunk_overlap:g}s)."
                        )

        html = '<div style="font-size: 1.15em; line-height: 1.8;">' + "<br>".join(list_items) + "</div>"
        return gr.update(value=html, visible=True), state

    def run_action(uploaded_file, face_restore_run, *args, preview_only: bool = False, state: Dict[str, Any] = None, progress=None):
        """Main processing action with streaming support and gr.Progress integration."""
        try:
            state = state or {"seed_controls": {}, "operation_status": "ready"}
            state["operation_status"] = "running"
            # Clear any previous VRAM OOM banner at the start of a new run.
            clear_vram_oom_alert(state)
            seed_controls = state.get("seed_controls", {})

            video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
            image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

            def _media_updates(video_path: Optional[str], image_path: Optional[str]) -> tuple[Any, Any]:
                """
                Return (output_video_update, output_image_update) for the merged output panel.
                """
                try:
                    if video_path and not Path(video_path).is_dir():
                        if Path(video_path).suffix.lower() in video_exts:
                            return gr.update(value=video_path, visible=True), gr.update(value=None, visible=False)
                    if image_path and not Path(image_path).is_dir():
                        if Path(image_path).suffix.lower() in image_exts:
                            return gr.update(value=None, visible=False), gr.update(value=image_path, visible=True)
                except Exception:
                    pass
                return gr.update(value=None, visible=False), gr.update(value=None, visible=False)

            def _processing_indicator(title: str = "Upscale in progress..."):
                html = (
                    '<div class="processing-banner">'
                    '<div class="processing-spinner"></div>'
                    '<div class="processing-col">'
                    f'<div class="processing-text">{title}</div>'
                    '<div class="processing-sub">Scroll down to see run logs and completed chunk previews.</div>'
                    "</div></div>"
                )
                return gr.update(value=html, visible=True)

            indicator_off = gr.update(value="", visible=False)

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
                    "Input path missing or not found",  # status_box
                    "",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "No chunks",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Expand "all" to device list if specified
            cuda_device_raw = settings.get("cuda_device", "")
            if cuda_device_raw:
                settings["cuda_device"] = _expand_cuda_spec(cuda_device_raw)
            
            # Validate CUDA devices (using shared GPU utility)
            cuda_warning = validate_cuda_device_spec(settings.get("cuda_device", ""))
            if cuda_warning:
                yield (
                    f"{cuda_warning}",  # status_box
                    "",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "No chunks",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Check ffmpeg availability
            if not _ffmpeg_available():
                yield (
                    "ffmpeg not found in PATH. Install ffmpeg and retry.",  # status_box
                    "",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "No chunks",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Check disk space (require at least 5GB free)
            output_path = Path(global_settings.get("output_dir", output_dir))
            has_space, space_warning = check_disk_space(output_path, required_mb=5000)
            if not has_space:
                yield (
                    space_warning or "Insufficient disk space",  # status_box
                    f"Free up disk space before processing. Recommended: 5GB+ free",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "No chunks",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return
            elif space_warning:
                # Low space but might work - show warning
                yield (
                    f"{space_warning}",  # status_box
                    "Low disk space detected. Processing may fail if output is large.",  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "Disk space warning",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )

            # Setup processing parameters
            face_apply = bool(face_restore_run) or bool(global_settings.get("face_global", False))
            face_strength = float(global_settings.get("face_strength", 0.5))

            # Apply shared sizing values from Resolution & Scene Split tab (global cache)
            if seed_controls.get("upscale_factor_val") is not None:
                try:
                    settings["upscale_factor"] = float(seed_controls["upscale_factor_val"])
                except Exception:
                    pass
            if seed_controls.get("max_resolution_val") is not None:
                try:
                    settings["max_resolution"] = int(seed_controls["max_resolution_val"] or 0)
                except Exception:
                    pass
            if "ratio_downscale" in seed_controls:
                settings["pre_downscale_then_upscale"] = bool(seed_controls.get("ratio_downscale", False))

            enable_max_target = seed_controls.get("enable_max_target", True)
            chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap_sec = float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            per_chunk_cleanup = seed_controls.get("per_chunk_cleanup", False)

            settings["chunk_size_sec"] = chunk_size_sec
            settings["chunk_overlap_sec"] = chunk_overlap_sec
            settings["per_chunk_cleanup"] = per_chunk_cleanup
            settings["frame_accurate_split"] = bool(seed_controls.get("frame_accurate_split", True))

            # NOTE: SeedVR2 CLI `--resolution` is now computed per-input from Upscale-x rules.
            # This happens inside _process_single_file() so it also applies to batch items.

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

            # Apply audio mux preferences from Output tab (used by chunking + final output postprocessing)
            if seed_controls.get("audio_codec_val") is not None:
                settings["audio_codec"] = seed_controls.get("audio_codec_val") or "copy"
            if seed_controls.get("audio_bitrate_val") is not None:
                settings["audio_bitrate"] = seed_controls.get("audio_bitrate_val") or ""
            
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
                        "Batch input path does not exist",  # status_box
                        "",  # log_box
                        "",  # progress_indicator
                        gr.update(value=None, visible=False),  # output_video
                        gr.update(value=None, visible=False),  # output_image
                        "No chunks",  # chunk_info
                        "",  # resume_status
                        "",  # chunk_progress
                        gr.update(value="", visible=False),  # comparison_note
                        gr.update(value=None, visible=False),  # image_slider
                        gr.update(value="", visible=False),  # video_comparison_html
                        gr.update(visible=False),  # chunk_gallery
                        gr.update(visible=False),  # batch_gallery
                        state  # shared_state
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
                        "No supported files found in batch input",  # status_box
                        "",  # log_box
                        "",  # progress_indicator
                        gr.update(value=None, visible=False),  # output_video
                        gr.update(value=None, visible=False),  # output_image
                        "No chunks",  # chunk_info
                        "",  # resume_status
                        "",  # chunk_progress
                        gr.update(value="", visible=False),  # comparison_note
                        gr.update(value=None, visible=False),  # image_slider
                        gr.update(value="", visible=False),  # video_comparison_html
                        gr.update(visible=False),  # chunk_gallery
                        gr.update(visible=False),  # batch_gallery
                        state  # shared_state
                    )
                    return
                
                # ADDED: Disk space pre-check before batch processing
                from shared.path_utils import get_disk_free_gb
                try:
                    temp_dir_path = Path(global_settings.get("temp_dir", "temp"))
                    output_dir_path = batch_output_path if batch_output_path.exists() else output_dir
                    
                    temp_free_gb = get_disk_free_gb(temp_dir_path)
                    output_free_gb = get_disk_free_gb(output_dir_path)
                    
                    # Estimate required space (rough: 10GB per video, 100MB per image)
                    estimated_gb = (video_count * 10.0) + (image_count * 0.1)
                    
                    warnings = []
                    errors = []
                    
                    # Critical: Less than 5GB free
                    if temp_free_gb < 5.0:
                        errors.append(f"CRITICAL: Only {temp_free_gb:.1f}GB free in temp directory. Need at least 5GB for processing.")
                    elif temp_free_gb < estimated_gb:
                        warnings.append(f"LOW TEMP SPACE: {temp_free_gb:.1f}GB free, estimated need: {estimated_gb:.1f}GB. May fail during processing.")
                    
                    if output_free_gb < 5.0:
                        errors.append(f"CRITICAL: Only {output_free_gb:.1f}GB free in output directory. Need at least 5GB.")
                    elif output_free_gb < estimated_gb * 0.5:
                        warnings.append(f"LOW OUTPUT SPACE: {output_free_gb:.1f}GB free, estimated need: {estimated_gb * 0.5:.1f}GB")
                    
                    if errors:
                        error_msg = "INSUFFICIENT DISK SPACE - Cannot start batch processing:\n" + "\n".join(errors)
                        if warnings:
                            error_msg += "\n\nAdditional warnings:\n" + "\n".join(warnings)
                        
                        yield (
                            error_msg,  # status_box
                            "",  # log_box
                            "",  # progress_indicator
                            gr.update(value=None, visible=False),  # output_video
                            gr.update(value=None, visible=False),  # output_image
                            f"Batch aborted: {len(batch_files)} files (insufficient disk space)",  # chunk_info
                            "",  # resume_status
                            "",  # chunk_progress
                            gr.update(value="", visible=False),  # comparison_note
                            gr.update(value=None, visible=False),  # image_slider
                            gr.update(value="", visible=False),  # video_comparison_html
                            gr.update(visible=False),  # chunk_gallery
                            gr.update(visible=False),  # batch_gallery
                            state  # shared_state
                        )
                        return
                    
                    if warnings:
                        warning_msg = "DISK SPACE WARNINGS:\n" + "\n".join(warnings) + "\n\nProceeding with batch processing..."
                        yield (
                            warning_msg,  # status_box
                            f"Starting batch: {len(batch_files)} files\nTemp: {temp_free_gb:.1f}GB free | Output: {output_free_gb:.1f}GB free",  # log_box
                            "",  # progress_indicator
                            gr.update(value=None, visible=False),  # output_video
                            gr.update(value=None, visible=False),  # output_image
                            f"Batch: {len(batch_files)} files queued",  # chunk_info
                            "",  # resume_status
                            "",  # chunk_progress
                            gr.update(value="", visible=False),  # comparison_note
                            gr.update(value=None, visible=False),  # image_slider
                            gr.update(value="", visible=False),  # video_comparison_html
                            gr.update(visible=False),  # chunk_gallery
                            gr.update(visible=False),  # batch_gallery
                            state  # shared_state
                        )
                except Exception as e:
                    # Disk check failed - log warning but don't block processing
                    print(f"Warning: Disk space check failed: {e}")
                
                # Continue with batch processing
                if not batch_files:
                    yield (
                        "No supported files found after validation",  # status_box
                        "",  # log_box
                        "",  # progress_indicator
                        gr.update(value=None, visible=False),  # output_video
                        gr.update(value=None, visible=False),  # output_image
                        "No chunks",  # chunk_info
                        "",  # resume_status
                        "",  # chunk_progress
                        gr.update(value="", visible=False),  # comparison_note
                        gr.update(value=None, visible=False),  # image_slider
                        gr.update(value="", visible=False),  # video_comparison_html
                        gr.update(visible=False),  # chunk_gallery
                        gr.update(visible=False),  # batch_gallery
                        state  # shared_state
                    )
                    return

                # Create batch processor
                def _batch_progress_cb(bp):
                    try:
                        if progress and bp and bp.total_files:
                            done = int(bp.completed_files + bp.failed_files + bp.skipped_files)
                            total = int(bp.total_files or 1)
                            desc = f"Batch: {done}/{total} files processed"
                            if bp.current_file:
                                desc += f" ({Path(bp.current_file).name})"
                            progress(min(1.0, done / total), desc=desc)
                    except Exception:
                        pass

                batch_processor = BatchProcessor(
                    output_dir=str(batch_output_path) if batch_output_path.exists() else str(output_dir),
                    max_workers=1,  # Sequential processing for memory management
                    # SeedVR2 already writes run_summary.json via run_logger in the single-file pipeline
                    # (and batch writes consolidated metadata at the end). Disable BatchProcessor's
                    # own per-file summaries to avoid duplicates.
                    telemetry_enabled=False,
                    progress_callback=_batch_progress_cb,
                )

                # Create batch jobs
                jobs: List[BatchJob] = []
                for input_file in sorted(set(batch_files)):
                    # For image-only batches, disable per-file telemetry (we will write one consolidated metadata file).
                    job_global_settings = global_settings.copy()
                    if is_image_only_batch:
                        job_global_settings["telemetry"] = False

                    jobs.append(
                        BatchJob(
                            input_path=str(input_file),
                            metadata={
                                "settings": settings.copy(),
                                "global_settings": job_global_settings,
                                "face_apply": face_apply,
                                "face_strength": face_strength,
                                "seed_controls": seed_controls.copy(),
                                "is_image": Path(input_file).suffix.lower() in SEEDVR2_IMAGE_EXTS,
                            },
                        )
                    )

                # Define processing function for each job (reuses the single-file pipeline)
                def process_single_batch_job(job: BatchJob) -> bool:
                    try:
                        # Process single file with current settings
                        single_settings = job.metadata["settings"].copy()
                        single_settings["input_path"] = job.input_path
                        single_settings["batch_enable"] = False  # Disable batch for individual processing
                        single_settings["_original_filename"] = Path(job.input_path).name

                        overwrite_existing = bool(seed_controls.get("overwrite_existing_batch_val", False))

                        input_file = Path(job.input_path)
                        batch_output_folder = Path(batch_output_path) if batch_output_path.exists() else output_dir

                        from shared.output_run_manager import batch_item_dir, prepare_batch_video_run_dir
                        from shared.path_utils import resolve_output_location, sanitize_filename

                        is_video = input_file.suffix.lower() in SEEDVR2_VIDEO_EXTS
                        if not is_video:
                            # Batch images: write directly into the batch output folder using the original name.
                            safe_name = sanitize_filename(input_file.name)
                            target_path = batch_output_folder / safe_name
                            if target_path.exists() and not overwrite_existing:
                                job.status = "skipped"
                                job.output_path = str(target_path)
                                return True
                            if overwrite_existing:
                                try:
                                    target_path.unlink(missing_ok=True)
                                except Exception:
                                    pass
                            single_settings["output_override"] = str(target_path)
                            batch_work_dir = batch_output_folder
                        else:
                            # Batch videos: stable per-input folder under outputs/<input_stem>/ with chunk artifacts inside.
                            item_out_dir = batch_item_dir(batch_output_folder, input_file.name)
                            predicted_fmt = single_settings.get("output_format") or "auto"
                            predicted_fmt = "png" if str(predicted_fmt).lower() == "png" else "mp4"
                            predicted_output = resolve_output_location(
                                input_path=str(input_file),
                                output_format=str(predicted_fmt),
                                global_output_dir=str(item_out_dir),
                                batch_mode=False,
                                png_padding=single_settings.get("png_padding"),
                                png_keep_basename=bool(single_settings.get("png_keep_basename", True)),
                                original_filename=input_file.name,
                            )
                            if Path(predicted_output).exists() and not overwrite_existing:
                                job.status = "skipped"
                                job.output_path = str(predicted_output)
                                return True
                            run_paths = prepare_batch_video_run_dir(
                                batch_output_folder,
                                input_file.name,
                                input_path=str(input_file),
                                model_label="SeedVR2",
                                mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                                overwrite_existing=overwrite_existing,
                            )
                            if not run_paths:
                                if not overwrite_existing:
                                    job.status = "skipped"
                                    job.output_path = str(predicted_output)
                                    return True
                                job.error_message = f"Could not create batch output folder: {item_out_dir}"
                                return False
                            single_settings["_run_dir"] = str(run_paths.run_dir)
                            single_settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                            single_settings["output_override"] = str(run_paths.run_dir)
                            batch_work_dir = run_paths.run_dir

                        # Run single-file processing (no per-file streaming in batch)
                        _status, logs, output_video, output_image, _chunk_info, _chunk_summary, _chunk_progress = _process_single_file(
                            runner,
                            single_settings,
                            job.metadata["global_settings"],
                            job.metadata["seed_controls"],
                            job.metadata["face_apply"],
                            job.metadata["face_strength"],
                            run_logger,
                            Path(batch_work_dir),
                            False,  # not preview
                            None,   # progress_cb
                        )

                        if output_video or output_image:
                            job.output_path = output_video or output_image
                            return True

                        job.error_message = logs
                        return False

                    except Exception as e:
                        job.error_message = str(e)
                        return False

                # Run batch processing
                batch_result = batch_processor.process_batch(
                    jobs=jobs,
                    processor_func=process_single_batch_job,
                    max_concurrent=1,
                )

                # Summarize results and collect output paths for gallery
                completed = int(batch_result.completed_files)
                failed = int(batch_result.failed_files)
                
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
                        # Note: Individual video metadata already written by _process_single_file  run_logger.write_summary
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

                # Compact failure summary for the log box
                log_lines = [f"{summary_msg}"]
                if failed:
                    for j in [x for x in jobs if x.status == "failed"][:10]:
                        name = Path(j.input_path).name
                        err = (j.error_message or "").strip()
                        err = (err[:180] + "") if len(err) > 180 else err
                        log_lines.append(f"{name}: {err}" if err else f"{name}")

                yield (
                    f"{summary_msg}",  # status_box
                    "\n".join(log_lines),  # log_box
                    "",  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    f"Batch: {completed} completed, {failed} failed",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(value=batch_outputs[:50], visible=True) if batch_outputs else gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Single file processing with streaming updates
            processing_complete = False
            chunk_info = "Waiting for first progress update..."

            # Start processing with progress tracking
            effective_output_dir = Path(global_settings.get("output_dir", output_dir))

            # NEW: Per-run output folder for videos (0001/0002/...) to avoid collisions
            # and to store chunk artifacts in user-visible locations.
            if (not preview_only) and detect_input_type(settings["input_path"]) == "video":
                try:
                    run_paths, explicit_final = prepare_single_video_run(
                        output_root_fallback=effective_output_dir,
                        output_override_raw=settings.get("output_override"),
                        input_path=settings["input_path"],
                        original_filename=original_filename,
                        model_label="SeedVR2",
                        mode=str(getattr(runner, "get_mode", lambda: "subprocess")() or "subprocess"),
                    )
                    effective_output_dir = Path(run_paths.run_dir)
                    # Store for UI (chunk gallery, resume helpers, user visibility)
                    seed_controls["last_run_dir"] = str(effective_output_dir)
                    seed_controls["processed_chunks_dir"] = str(run_paths.processed_chunks_dir)
                    settings["_run_dir"] = str(effective_output_dir)
                    settings["_processed_chunks_dir"] = str(run_paths.processed_chunks_dir)

                    # Output override now targets the run folder (or an explicit file inside it)
                    settings["_user_output_override_raw"] = settings.get("output_override") or ""
                    settings["output_override"] = str(explicit_final) if explicit_final else str(effective_output_dir)
                except Exception as e:
                    # Fail open: fall back to global output dir if run folder creation fails.
                    seed_controls["last_run_dir"] = str(effective_output_dir)
                    settings["_run_dir"] = str(effective_output_dir)

            yield (
                "Starting processing...",  # status_box
                "Preparing run...",  # log_box
                _processing_indicator("Upscale in progress..."),  # progress_indicator
                gr.update(value=None, visible=False),  # output_video
                gr.update(value=None, visible=False),  # output_image
                "Waiting for first progress update...",  # chunk_info
                "",  # resume_status
                "",  # chunk_progress
                gr.update(value="", visible=False),  # comparison_note
                gr.update(value=None, visible=False),  # image_slider
                gr.update(value="", visible=False),  # video_comparison_html
                gr.update(visible=False),  # chunk_gallery
                gr.update(visible=False),  # batch_gallery
                state  # shared_state
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
                        effective_output_dir,
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
            completed_chunk_count = 0
            active_chunk_idx = 0
            total_chunks_estimate = 1
            last_progress_value = 0.0
            last_ui_update_time = 0
            ui_update_throttle = 0.25
            accumulated_messages = []
            
            while proc_thread.is_alive() or not progress_queue.empty():
                try:
                    update_type, data = progress_queue.get(timeout=0.1)
                    current_time = time.time()                    if update_type == "progress":
                        message = str(data or "").strip()
                        if not message:
                            continue

                        accumulated_messages.append(message)
                        if len(accumulated_messages) > 200:
                            accumulated_messages = accumulated_messages[-200:]

                        import re

                        chunk_match = re.search(r"(processing|completed)\s+chunk\s+(\d+)\s*/\s*(\d+)", message, re.IGNORECASE)
                        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*%", message)
                        pct_value = float(pct_match.group(1)) if pct_match else None

                        if chunk_match:
                            phase = str(chunk_match.group(1) or "").lower()
                            chunk_idx = int(chunk_match.group(2))
                            total_chunks_estimate = max(1, int(chunk_match.group(3)))
                            active_chunk_idx = chunk_idx
                            if phase == "completed":
                                completed_chunk_count = max(completed_chunk_count, chunk_idx)
                            if pct_value is not None:
                                last_progress_value = max(last_progress_value, min(0.999, pct_value / 100.0))
                            elif phase == "completed":
                                last_progress_value = max(last_progress_value, completed_chunk_count / max(1, total_chunks_estimate))
                            else:
                                last_progress_value = max(last_progress_value, max(0.0, (chunk_idx - 1) / max(1, total_chunks_estimate)))
                        else:
                            nm_match = re.search(r"(\d+)\s*/\s*(\d+)", message)
                            if nm_match:
                                current = int(nm_match.group(1))
                                total = max(1, int(nm_match.group(2)))
                                total_chunks_estimate = max(total_chunks_estimate, total)
                                last_progress_value = max(last_progress_value, min(0.999, current / total))
                            elif pct_value is not None:
                                last_progress_value = max(last_progress_value, min(0.999, pct_value / 100.0))

                        if progress:
                            progress(last_progress_value, desc=message[:100] if message else "Processing...")

                        current_chunk_items = (state or {}).get("seed_controls", {}).get("chunk_gallery_items", [])
                        if not current_chunk_items:
                            current_chunk_items = (state or {}).get("seed_controls", {}).get("chunk_thumbnails", [])
                        chunk_gallery_update = (
                            gr.update(
                                value=current_chunk_items,
                                visible=len(current_chunk_items) > 0,
                                columns=2,
                                rows=2,
                                height=320,
                            )
                            if current_chunk_items
                            else gr.update(visible=False)
                        )

                        is_key_event = (
                            "processing chunk" in message.lower()
                            or "completed chunk" in message.lower()
                            or "error" in message.lower()
                            or "failed" in message.lower()
                        )
                        if (current_time - last_ui_update_time) < ui_update_throttle and not is_key_event:
                            continue

                        percent_done = int(round(max(0.0, min(1.0, last_progress_value)) * 100.0))
                        completed_shown = min(max(0, completed_chunk_count), total_chunks_estimate)
                        current_shown = active_chunk_idx if active_chunk_idx > 0 else max(1, completed_shown)
                        if total_chunks_estimate > 1:
                            status_text = (
                                f"Processing... {completed_shown}/{total_chunks_estimate} chunks completed "
                                f"({percent_done}%)"
                            )
                            indicator_title = (
                                f"Processing... ({completed_shown}/{total_chunks_estimate} chunks completed, "
                                f"{percent_done}% done)"
                            )
                        else:
                            status_text = f"Processing... {percent_done}%"
                            indicator_title = f"Processing... ({percent_done}% done)"

                        chunk_status_text = (
                            f"Chunks completed: {completed_shown}/{total_chunks_estimate}\n"
                            f"Current chunk: {current_shown}/{total_chunks_estimate}\n"
                            f"Latest: {message}"
                            if total_chunks_estimate > 1
                            else f"Latest: {message}"
                        )

                        yield (
                            status_text,
                            "\n".join(accumulated_messages[-30:]),
                            _processing_indicator(indicator_title),
                            gr.update(value=None, visible=False),
                            gr.update(value=None, visible=False),
                            chunk_status_text,
                            "",
                            "\n".join(accumulated_messages[-12:]),
                            gr.update(value="", visible=False),
                            gr.update(value=None, visible=False),
                            gr.update(value="", visible=False),
                            chunk_gallery_update,
                            gr.update(visible=False),
                            state,
                        )
                        last_ui_update_time = current_time
                    elif update_type == "complete":
                        status, logs, output_video, output_image, chunk_info, chunk_summary, chunk_progress = data
                        processing_complete = True
                        if progress:
                            progress(1.0, desc="Complete!")
                        break                    elif update_type == "error":
                        if progress:
                            progress(0, desc="Error occurred")
                        if maybe_set_vram_oom_alert(state, model_label="SeedVR2", text=data, settings=settings):
                            state["operation_status"] = "error"
                            show_vram_oom_modal(state, title="Out of VRAM (GPU) - SeedVR2", duration=None)

                        yield (
                            ("Out of VRAM (GPU) - see banner above" if state.get("alerts", {}).get("oom", {}).get("visible") else "Processing failed"),
                            f"Error: {data}",
                            indicator_off,
                            gr.update(value=None, visible=False),
                            gr.update(value=None, visible=False),
                            "Error occurred",
                            "",
                            "",
                            gr.update(value="", visible=False),
                            gr.update(value=None, visible=False),
                            gr.update(value="", visible=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            state,
                        )
                        return
                except queue.Empty:
                    continue            if not processing_complete:
                yield (
                    "Processing timed out",  # status_box
                    "Processing did not complete within expected time",  # log_box
                    indicator_off,  # progress_indicator
                    gr.update(value=None, visible=False),  # output_video
                    gr.update(value=None, visible=False),  # output_image
                    "Timeout",  # chunk_info
                    "",  # resume_status
                    "",  # chunk_progress
                    gr.update(value="", visible=False),  # comparison_note
                    gr.update(value=None, visible=False),  # image_slider
                    gr.update(value="", visible=False),  # video_comparison_html
                    gr.update(visible=False),  # chunk_gallery
                    gr.update(visible=False),  # batch_gallery
                    state  # shared_state
                )
                return

            # Final pass: if logs contain VRAM OOM signatures, raise the global banner.            if maybe_set_vram_oom_alert(state, model_label="SeedVR2", text=logs, settings=settings):
                state["operation_status"] = "error"
                status = "Out of VRAM (GPU) - see banner above"
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - SeedVR2", duration=None)

            # Create comparison based on mode from Output tab
            comparison_mode = seed_controls.get("comparison_mode_val", "native")
            original_path_for_compare = settings.get("_original_input_path_before_preprocess") or settings.get("input_path")
            comparison_html = ""
            
            if comparison_mode == "native":
                # Use gradio's native ImageSlider for images
                if output_image and Path(output_image).exists():
                    comparison_html = ""
                    # Check for pinned reference
                    pinned_ref = seed_controls.get("pinned_reference_path")
                    pin_enabled = seed_controls.get("pin_reference_val", False)
                    
                    image_slider_update = gr.update(
                        value=(pinned_ref if (pin_enabled and pinned_ref) else original_path_for_compare, output_image),
                        visible=True
                    )
                else:
                    # Check for pinned reference
                    pinned_ref = seed_controls.get("pinned_reference_path")
                    pin_enabled = seed_controls.get("pin_reference_val", False)
                    
                    comparison_html, image_slider_update = create_comparison_selector(
                        input_path=original_path_for_compare,
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
                    input_path=original_path_for_compare,
                    output_path=output_video or output_image,
                    comparison_mode=comparison_mode,
                    pinned_reference_path=pinned_ref,
                    pin_enabled=pin_enabled
                )

            # Build video comparison HTML for videos
            video_comparison_html_update = gr.update(value="", visible=False)
            if output_video and Path(output_video).exists():
                original_path = original_path_for_compare or ""
                if original_path and Path(original_path).exists():
                    # Use new video comparison slider
                    from shared.video_comparison_slider import create_video_comparison_html as create_vid_comp
                    
                    video_comp_html = create_vid_comp(
                        original_video=original_path,
                        upscaled_video=output_video,
                        height=600,
                        slider_position=50.0
                    )
                    video_comparison_html_update = gr.update(value=video_comp_html, visible=True)
                image_slider_update = gr.update(value=None, visible=False)
            
            # If no HTML comparison, use ImageSlider for images
            if not comparison_html and output_image and not output_video:
                image_slider_update = gr.update(
                    value=(original_path_for_compare, output_image),
                    visible=True
                )
            elif not image_slider_update:
                image_slider_update = gr.update(value=None, visible=False)

            state["operation_status"] = "completed" if "complete" in str(status or "").lower() else "ready"
            
            # Prepare final chunk gallery display
            final_chunk_items = (state or {}).get("seed_controls", {}).get("chunk_gallery_items", [])
            if not final_chunk_items:
                final_chunk_items = (state or {}).get("seed_controls", {}).get("chunk_thumbnails", [])
            final_chunk_gallery = gr.update(
                value=final_chunk_items,
                visible=len(final_chunk_items) > 0,
                columns=2,
                rows=2,
                height=360
            ) if final_chunk_items else gr.update(visible=False)
            
            vid_upd, img_upd = _media_updates(output_video, output_image)
            yield (
                status,  # status_box
                logs,  # log_box
                indicator_off,  # progress_indicator
                vid_upd,  # output_video
                img_upd,  # output_image
                chunk_info,  # chunk_info
                "",  # resume_status
                chunk_progress,  # chunk_progress - NOW POPULATED with actual chunk progress
                gr.update(value="", visible=False),  # comparison_note
                image_slider_update,  # image_slider
                video_comparison_html_update,  # video_comparison_html
                final_chunk_gallery,  # chunk_gallery - SHOW completed chunk thumbnails!
                gr.update(visible=False),  # batch_gallery - Hide for single file
                state  # shared_state
            )

        except Exception as e:
            error_msg = f"Critical error in SeedVR2 processing: {str(e)}"
            state["operation_status"] = "error"
            # If this was VRAM OOM, show the big banner.
            if maybe_set_vram_oom_alert(state, model_label="SeedVR2", text=str(e), settings=locals().get("settings")):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) - SeedVR2", duration=None)
            yield (
                "Critical error",  # status_box
                error_msg,  # log_box
                indicator_off,  # progress_indicator
                gr.update(value=None, visible=False),  # output_video
                gr.update(value=None, visible=False),  # output_image
                "Error",  # chunk_info
                "",  # resume_status
                "",  # chunk_progress
                gr.update(value="", visible=False),  # comparison_note
                gr.update(value=None, visible=False),  # image_slider
                gr.update(value="", visible=False),  # video_comparison_html
                gr.update(visible=False),  # chunk_gallery
                gr.update(visible=False),  # batch_gallery
                state  # shared_state
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
