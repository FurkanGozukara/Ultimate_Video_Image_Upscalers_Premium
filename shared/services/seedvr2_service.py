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
    
    Attempts to use flash_attn if available and working, otherwise falls back to sdpa.
    This is more robust than just checking import - actually tests CUDA compatibility.
    """
    try:
        import flash_attn
        import torch
        
        # Check if CUDA is available (flash_attn requires CUDA)
        if not torch.cuda.is_available():
            return "sdpa"
        
        # Actually test if flash_attn works on current GPU
        # Some GPUs support the import but not execution
        try:
            # Quick compatibility check without full model load
            _ = flash_attn.__version__
            return "flash_attn"
        except (AttributeError, RuntimeError):
            return "sdpa"
            
    except ImportError:
        return "sdpa"
    except Exception:
        # Any other error, fall back safely
        return "sdpa"


def seedvr2_defaults() -> Dict[str, Any]:
    """Get default SeedVR2 settings aligned with CLI defaults."""
    try:
        import torch
        cuda_default = "0" if torch.cuda.is_available() else ""
    except Exception:
        cuda_default = ""
    
    return {
        "input_path": "",
        "output_override": "",
        "output_format": "auto",
        "model_dir": "",
        "dit_model": get_seedvr2_model_names()[0] if get_seedvr2_model_names() else "seedvr2_ema_3b_fp16.safetensors",
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "chunk_enable": False,
        "scene_threshold": 27.0,
        "scene_min_len": 2.0,
        "chunk_size": 0,  # SeedVR2 native chunking (frames per chunk, 0=disabled)
        "resolution": 1080,
        "max_resolution": 0,
        "batch_size": 5,
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
        "attention_mode": _get_default_attention_mode(),
        "compile_dit": False,
        "compile_vae": False,
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
    }


SEEDVR2_ORDER: List[str] = [
    "input_path",
    "output_override",
    "output_format",
    "model_dir",
    "dit_model",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "chunk_enable",
    "scene_threshold",
    "scene_min_len",
    "chunk_size",
    "resolution",
    "max_resolution",
    "batch_size",
    "uniform_batch_size",
    "seed",
    "skip_first_frames",
    "load_cap",
    "prepend_frames",
    "temporal_overlap",
    "color_correction",
    "input_noise_scale",
    "latent_noise_scale",
    "cuda_device",
    "dit_offload_device",
    "vae_offload_device",
    "tensor_offload_device",
    "blocks_to_swap",
    "swap_io_components",
    "vae_encode_tiled",
    "vae_encode_tile_size",
    "vae_encode_tile_overlap",
    "vae_decode_tiled",
    "vae_decode_tile_size",
    "vae_decode_tile_overlap",
    "tile_debug",
    "attention_mode",
    "compile_dit",
    "compile_vae",
    "compile_backend",
    "compile_mode",
    "compile_fullgraph",
    "compile_dynamic",
    "compile_dynamo_cache_size_limit",
    "compile_dynamo_recompile_limit",
    "cache_dit",
    "cache_vae",
    "debug",
    "resume_chunking",
]


# Guardrails -------------------------------------------------------------------
def _enforce_seedvr2_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any], state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Apply SeedVR2-specific validation rules and apply resolution tab settings if available."""
    cfg = cfg.copy()

    # Apply resolution tab settings from shared state if available
    if state:
        seed_controls = state.get("seed_controls", {})
        
        # Only apply if values are set in resolution tab
        if "resolution_val" in seed_controls and seed_controls["resolution_val"]:
            cfg["resolution"] = int(seed_controls["resolution_val"])
        if "max_resolution_val" in seed_controls and seed_controls["max_resolution_val"]:
            cfg["max_resolution"] = int(seed_controls["max_resolution_val"])
        if "chunk_size_sec" in seed_controls and seed_controls["chunk_size_sec"] > 0:
            cfg["chunk_enable"] = True
            cfg["scene_threshold"] = 27.0  # Use scene detection if chunking enabled
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


def _resolve_input_path(file_upload: Optional[str], manual_path: str, batch_enable: bool, batch_input: str) -> str:
    """
    Resolve the input path from various sources with priority order:
    1. Batch input (if batch enabled)
    2. File upload (highest priority for single files)
    3. Manual path entry (fallback)
    """
    if batch_enable and batch_input and batch_input.strip():
        return batch_input.strip()
    if file_upload and str(file_upload).strip():
        return str(file_upload).strip()
    return manual_path.strip() if manual_path else ""


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
    """Convert argument list to settings dictionary."""
    return dict(zip(SEEDVR2_ORDER, args))


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
) -> Tuple[str, str, Optional[str], Optional[str], str, str]:
    """
    Process a single file with SeedVR2.
    Returns: (status, logs, output_video, output_image, chunk_info, chunk_summary)
    """
    local_logs = []
    output_video = None
    output_image = None
    chunk_info_msg = "No chunking performed."
    chunk_summary = "Single pass (no chunking)."
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
                else:
                    status = "❌ Preview upscaling failed"
                    local_logs.append(result.log)
                    
                return status, "\n".join(local_logs), None, output_image, chunk_info_msg, chunk_summary
                
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

        # Determine if chunking should be used
        should_chunk = (
            settings.get("chunk_enable", False)
            and not preview_only
            and detect_input_type(settings["input_path"]) == "video"
        )

        if should_chunk:
            # Process with chunking
            completed_chunks = 0

            def chunk_progress_callback(progress_val, desc=""):
                nonlocal completed_chunks
                if "Completed chunk" in desc:
                    completed_chunks += 1
                    if progress_cb:
                        progress_cb(f"Completed {completed_chunks} chunks\n")

            rc, clog, final_out, chunk_count = chunk_and_process(
                runner,
                settings,
                scene_threshold=settings.get("scene_threshold", 27.0),
                min_scene_len=settings.get("scene_min_len", 2.0),
                temp_dir=Path(global_settings["temp_dir"]),
                on_progress=lambda msg: None,
                chunk_seconds=float(settings.get("chunk_size_sec") or 0),
                chunk_overlap=float(settings.get("chunk_overlap_sec") or 0),
                per_chunk_cleanup=bool(settings.get("per_chunk_cleanup")),
                resume_from_partial=bool(settings.get("resume_chunking", False)),
                allow_partial=True,
                global_output_dir=str(runner.output_dir) if hasattr(runner, "output_dir") else None,
                progress_tracker=chunk_progress_callback,
            )

            status = "✅ Chunked upscale complete" if rc == 0 else f"⚠️ Chunked upscale ended early ({rc})"
            output_path = final_out if final_out else None
            output_video = output_path if output_path and output_path.lower().endswith(".mp4") else None
            output_image = None
            local_logs.append(clog)
            chunk_summary = f"Processed {chunk_count} chunks. Final: {output_path}"
            chunk_info_msg = f"Chunks: {chunk_count}\nOutput: {output_path}\n{clog}"
            result = RunResult(rc, output_path, clog)
        else:
            # Process without chunking
            result = runner.run_seedvr2(
                settings,
                on_progress=lambda x: progress_cb(x) if progress_cb else None,
                preview_only=preview_only
            )
            status = "✅ Upscale complete" if result.returncode == 0 else f"⚠️ Upscale exited with code {result.returncode}"

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

            # Log the run
            run_logger.write_summary(
                Path(result.output_path) if result.output_path else output_dir,
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

        # Apply FPS override if specified
        fps_val = seed_controls.get("fps_override_val")
        if fps_val and output_video and Path(output_video).exists():
            adjusted = ffmpeg_set_fps(Path(output_video), float(fps_val))
            output_video = str(adjusted)
            local_logs.append(f"FPS overridden to {fps_val}: {adjusted}")

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

    return status, "\n".join(local_logs), output_video, output_image, chunk_info_msg, chunk_summary


# Comparison fallback ----------------------------------------------------------
def comparison_html_slider():
    """Get comparison slider HTML fallback."""
    return gr.HTML.update(
        value="<p>Comparison fallback: use native slider where available; custom HTML slider assets can be loaded if deployed.</p>"
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
    """Build SeedVR2 callback functions for the UI."""
    defaults = seedvr2_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown."""
        presets = preset_manager.list_presets("seedvr2", model_name)
        last_used = preset_manager.get_last_used_name("seedvr2", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, model_name: str, *args):
        """Save a preset."""
        if not preset_name.strip():
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
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
        """Cancel current processing and compile any partial outputs if available."""
        canceled = runner.cancel()
        if not canceled:
            return gr.Markdown.update(value="No active process to cancel"), ""

        # Check for partial outputs that can be compiled
        temp_chunks_dir = Path(global_settings["temp_dir"]) / "chunks"
        if temp_chunks_dir.exists():
            try:
                from shared.chunking import detect_resume_state, concat_videos
                from shared.path_utils import collision_safe_path

                partial_video, completed_chunks = detect_resume_state(temp_chunks_dir, "mp4")
                partial_png, completed_png_chunks = detect_resume_state(temp_chunks_dir, "png")

                compiled_output = None

                # Try to compile video chunks
                if completed_chunks and len(completed_chunks) > 0:
                    partial_target = collision_safe_path(temp_chunks_dir / "cancelled_partial.mp4")
                    if concat_videos(completed_chunks, partial_target):
                        compiled_output = str(partial_target)
                        # Copy to outputs folder - use generic name since settings not in scope
                        final_output = Path(output_dir) / f"cancelled_partial_upscaled.mp4"
                        final_output = collision_safe_path(final_output)
                        shutil.copy2(partial_target, final_output)
                        compiled_output = str(final_output)

                # Or compile PNG chunks
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

                if compiled_output:
                    return gr.Markdown.update(value=f"⏹️ Cancelled - Partial output compiled: {Path(compiled_output).name}"), f"Partial results saved to: {compiled_output}"
                else:
                    return gr.Markdown.update(value="⏹️ Cancelled - No partial outputs to compile"), "Processing was cancelled before any chunks were completed"

            except Exception as e:
                return gr.Markdown.update(value=f"⏹️ Cancelled - Error compiling partials: {str(e)}"), "Cancellation successful but partial compilation failed"

        return gr.Markdown.update(value="⏹️ Processing cancelled"), "No partial outputs found to compile"

    def open_outputs_folder(state: Dict[str, Any]):
        """Open the outputs folder in file explorer."""
        try:
            import platform
            import subprocess
            
            out_dir = str(output_dir)
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", out_dir])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", out_dir])
            else:
                subprocess.Popen(["xdg-open", out_dir])
            return gr.Markdown.update(value=f"✅ Opened outputs folder: {out_dir}")
        except Exception as e:
            return gr.Markdown.update(value=f"❌ Failed to open outputs folder: {str(e)}")

    def clear_temp_folder(confirm: bool):
        """Clear temporary folder if confirmed."""
        if not confirm:
            return gr.Markdown.update(value="⚠️ Check 'Confirm delete temp' to clear temporary files")
        
        try:
            temp_path = Path(temp_dir)
            if temp_path.exists():
                shutil.rmtree(temp_path)
                temp_path.mkdir(parents=True, exist_ok=True)
                return gr.Markdown.update(value=f"✅ Cleared temp folder: {temp_path}")
            else:
                return gr.Markdown.update(value=f"ℹ️ Temp folder doesn't exist: {temp_path}")
        except Exception as e:
            return gr.Markdown.update(value=f"❌ Failed to clear temp folder: {str(e)}")

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

            # Validate inputs
            input_path = _resolve_input_path(
                uploaded_file,
                settings["input_path"],
                settings["batch_enable"],
                settings["batch_input_path"]
            )
            settings["input_path"] = normalize_path(input_path)
            state["seed_controls"]["last_input_path"] = settings["input_path"]

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
                    job = BatchJob(
                        input_path=str(input_file),
                        metadata={
                            "settings": settings.copy(),
                            "global_settings": global_settings,
                            "face_apply": face_apply,
                            "face_strength": face_strength,
                            "seed_controls": seed_controls.copy(),
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

                        status, logs, output_video, output_image, chunk_info, chunk_summary = _process_single_file(
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
                    status, logs, output_video, output_image, chunk_info, chunk_summary = _process_single_file(
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
                    progress_queue.put(("complete", (status, logs, output_video, output_image, chunk_info, chunk_summary)))
                except Exception as e:
                    progress_queue.put(("error", str(e)))

            # Start processing in background thread
            import threading
            proc_thread = threading.Thread(target=processing_thread, daemon=True)
            proc_thread.start()

            # Stream progress updates with gr.Progress integration
            chunk_count = 0
            total_chunks_estimate = 1
            last_progress_value = 0.0
            
            while proc_thread.is_alive() or not progress_queue.empty():
                try:
                    update_type, data = progress_queue.get(timeout=0.1)
                    if update_type == "progress":
                        # Update gr.Progress if available with intelligent parsing
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
                        
                        yield (
                            f"⚙️ Processing: {data}",
                            f"Progress: {data}",
                            "",
                            None,
                            None,
                            chunk_info or "Processing...",
                            "",
                            "",
                            gr.HTML.update(value=f'<div style="background: #f0f8ff; padding: 10px; border-radius: 5px;">{data}</div>'),
                            gr.ImageSlider.update(value=None),
                            gr.HTML.update(value="", visible=False),
                            gr.Gallery.update(visible=False),
                            state
                        )
                    elif update_type == "complete":
                        status, logs, output_video, output_image, chunk_info, chunk_summary = data
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
                "",  # chunk_progress
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
        "open_outputs_folder": open_outputs_folder,
        "clear_temp_folder": clear_temp_folder,
        "comparison_html_slider": comparison_html_slider,
        "auto_res_on_input": _auto_res_from_input,
        "get_model_loading_status": get_model_loading_status,
    }
