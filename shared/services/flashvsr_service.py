"""
FlashVSR+ Service Module
Handles FlashVSR+ processing logic, presets, and callbacks
"""

import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.flashvsr_runner import run_flashvsr, FlashVSRResult
from shared.path_utils import normalize_path
from shared.logging_utils import RunLogger
from shared.comparison_unified import create_unified_comparison
from shared.models.flashvsr_meta import get_flashvsr_metadata, get_flashvsr_default_model
from shared.gpu_utils import expand_cuda_device_spec, validate_cuda_device_spec
from shared.error_handling import logger as error_logger

# Cancel event for FlashVSR+ processing
_flashvsr_cancel_event = threading.Event()


def flashvsr_defaults(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default FlashVSR+ settings aligned with CLI defaults.
    Applies model-specific metadata when model_name is provided.
    """
    try:
        import torch
        cuda_default = "auto" if torch.cuda.is_available() else "cpu"
    except Exception:
        cuda_default = "cpu"
    
    # Get model metadata if specific model is provided
    default_model = model_name or get_flashvsr_default_model()
    model_meta = get_flashvsr_metadata(default_model)
    
    # Apply model-specific defaults if metadata available
    if model_meta:
        default_dtype = model_meta.default_dtype
        default_tile_size = model_meta.default_tile_size
        default_overlap = model_meta.default_overlap
        default_attention = model_meta.default_attention
        version = model_meta.version
        mode = model_meta.mode
        scale = model_meta.scale
    else:
        default_dtype = "bf16"
        default_tile_size = 256
        default_overlap = 24
        default_attention = "sage"
        version = "10"
        mode = "tiny"
        scale = 4
    
    return {
        "input_path": "",
        "output_override": "",
        "scale": scale,
        "version": version,
        "mode": mode,
        "tiled_vae": False,
        "tiled_dit": False,
        "tile_size": default_tile_size,
        "overlap": default_overlap,
        "unload_dit": False,
        "color_fix": True,
        "seed": 0,
        "dtype": default_dtype,
        "device": cuda_default,
        "fps": 30,
        "quality": 6,
        "attention": default_attention,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
    }


FLASHVSR_ORDER: List[str] = [
    "input_path",
    "output_override",
    "scale",
    "version",
    "mode",
    "tiled_vae",
    "tiled_dit",
    "tile_size",
    "overlap",
    "unload_dit",
    "color_fix",
    "seed",
    "dtype",
    "device",
    "fps",
    "quality",
    "attention",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
]


def _flashvsr_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(FLASHVSR_ORDER, args))


def _enforce_flashvsr_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply FlashVSR+-specific validation rules using metadata registry.
    
    Enforces:
    - Single GPU requirement (FlashVSR+ doesn't support multi-GPU well)
    - Valid CUDA device IDs (checks against available devices)
    - VRAM-based tile size recommendations
    - Resolution constraints from model metadata
    - Valid version/mode/scale combinations
    """
    cfg = cfg.copy()
    
    # Build model identifier and get metadata
    model_id = f"v{cfg.get('version', '10')}_{cfg.get('mode', 'tiny')}_{cfg.get('scale', 4)}x"
    model_meta = get_flashvsr_metadata(model_id)
    
    if model_meta:
        # Enforce single GPU (FlashVSR+ doesn't support multi-GPU)
        device_str = str(cfg.get("device", "auto"))
        if device_str not in ("auto", "cpu", ""):
            # Parse device specification
            devices = [d.strip() for d in device_str.replace(" ", "").split(",") if d.strip()]
            if len(devices) > 1:
                error_logger.warning(f"FlashVSR+ doesn't support multi-GPU - forcing single GPU (using first: {devices[0]})")
                cfg["device"] = devices[0]
                cfg["_multi_gpu_disabled_reason"] = "FlashVSR+ is single-GPU optimized"
            
            # Validate CUDA device ID is actually available
            try:
                import torch
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    device_id_str = cfg["device"].replace("cuda:", "")
                    if device_id_str.isdigit():
                        device_id = int(device_id_str)
                        if device_id >= device_count:
                            error_logger.warning(f"Device ID {device_id} not available (only {device_count} GPUs detected) - falling back to auto")
                            cfg["device"] = "auto"
                            cfg["_device_validation_warning"] = f"Requested GPU {device_id} not found, using auto-select"
            except Exception as e:
                error_logger.warning(f"Failed to validate device ID: {e}")
        
        # Apply model-specific defaults if not set
        if not cfg.get("dtype"):
            cfg["dtype"] = model_meta.default_dtype
        
        if cfg.get("tile_size", 0) == 0:
            cfg["tile_size"] = model_meta.default_tile_size
        
        if cfg.get("overlap", 0) == 0:
            cfg["overlap"] = model_meta.default_overlap
        
        if not cfg.get("attention"):
            cfg["attention"] = model_meta.default_attention
        
        # Validate tile overlap constraint
        if cfg.get("tiled_vae") or cfg.get("tiled_dit"):
            tile_size = int(cfg.get("tile_size", model_meta.default_tile_size))
            overlap = int(cfg.get("overlap", model_meta.default_overlap))
            
            if overlap >= tile_size:
                cfg["overlap"] = max(0, tile_size - 1)
                error_logger.warning(f"Tile overlap ({overlap}) >= tile size ({tile_size}), correcting to {cfg['overlap']}")
            
            if tile_size < 64:
                cfg["tile_size"] = model_meta.default_tile_size
                error_logger.warning(f"Tile size too small, resetting to {cfg['tile_size']}")
    
    return cfg


def _apply_flashvsr_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    # Apply guardrails to merged preset
    merged = _enforce_flashvsr_guardrails(merged, defaults)
    return [merged[k] for k in FLASHVSR_ORDER]


def build_flashvsr_callbacks(
    preset_manager: PresetManager,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    """Build FlashVSR+ callback functions for the UI."""
    defaults = flashvsr_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown."""
        presets = preset_manager.list_presets("flashvsr", model_name)
        last_used = preset_manager.get_last_used_name("flashvsr", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save a preset."""
        if not preset_name.strip():
            return gr.update(), gr.update(value="⚠️ Enter a preset name"), *list(args)

        try:
            payload = _flashvsr_dict_from_args(list(args))
            model_name = f"v{payload['version']}_{payload['mode']}"
            
            preset_manager.save_preset_safe("flashvsr", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(FLASHVSR_ORDER, list(args)))
            loaded_vals = _apply_flashvsr_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"✅ Saved preset '{preset_name}'"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"❌ Error: {str(e)}"), *list(args)

    def load_preset(preset_name: str, version: str, mode: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        UI expects: inputs_list + [preset_status]
        """
        try:
            model_name = f"v{version}_{mode}"
            preset = preset_manager.load_preset_safe("flashvsr", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("flashvsr", model_name, preset_name)

            current_map = dict(zip(FLASHVSR_ORDER, current_values))
            values = _apply_flashvsr_preset(preset or {}, defaults, preset_manager, current=current_map)
            
            # Return values + status message (status is LAST)
            status_msg = f"✅ Loaded preset '{preset_name}'" if preset else "ℹ️ Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f"❌ Error: {str(e)}"))

    def safe_defaults():
        """Get safe default values."""
        return [defaults[key] for key in FLASHVSR_ORDER]

    def run_action(upload, *args, preview_only: bool = False, state=None, progress=None):
        """Main processing action with gr.Progress integration and pre-flight checks."""
        try:
            state = state or {"seed_controls": {}}
            seed_controls = state.get("seed_controls", {})
            settings_dict = _flashvsr_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            
            # Apply FlashVSR+ guardrails (single GPU, tile validation, etc.)
            settings = _enforce_flashvsr_guardrails(settings, defaults)
            
            # Apply shared Resolution & Scene Split tab settings (like SeedVR2/GAN)
            if seed_controls.get("resolution_val") is not None:
                settings["target_resolution"] = seed_controls["resolution_val"]
            if seed_controls.get("max_resolution_val") is not None:
                settings["max_target_resolution"] = seed_controls["max_resolution_val"]
            
            # Apply Output tab cached settings
            if seed_controls.get("fps_override_val") is not None and seed_controls["fps_override_val"] > 0:
                settings["fps"] = seed_controls["fps_override_val"]
            if seed_controls.get("comparison_mode_val"):
                settings["_comparison_mode"] = seed_controls["comparison_mode_val"]
            if seed_controls.get("save_metadata_val") is not None:
                settings["save_metadata"] = seed_controls["save_metadata_val"]
            
            # Clear cancel event
            _flashvsr_cancel_event.clear()
            
            # Initialize progress
            if progress:
                progress(0, desc="Initializing FlashVSR+...")
            
            # PRE-FLIGHT CHECKS (mirrors SeedVR2/GAN for consistency)
            from shared.error_handling import check_ffmpeg_available, check_disk_space
            
            # Check ffmpeg availability
            ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
            if not ffmpeg_ok:
                yield (
                    "❌ ffmpeg not found in PATH",
                    ffmpeg_msg or "Install ffmpeg and add to PATH before processing",
                    None,
                    None,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            # Check disk space (require at least 5GB free)
            output_path_check = Path(global_settings.get("output_dir", output_dir))
            has_space, space_warning = check_disk_space(output_path_check, required_mb=5000)
            if not has_space:
                yield (
                    "❌ Insufficient disk space",
                    space_warning or "Free up at least 5GB disk space before processing",
                    None,
                    None,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            # Resolve input
            input_path = normalize_path(upload if upload else settings["input_path"])
            if not input_path or not Path(input_path).exists():
                yield (
                    "❌ Input path missing",
                    "",
                    None,
                    None,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            settings["input_path"] = input_path
            settings["global_output_dir"] = str(output_dir)
            
            # Run FlashVSR+ in thread with cancel support
            result_holder = {}
            progress_queue = queue.Queue()
            process_handle = {"proc": None}  # Store subprocess handle
            
            def processing_thread():
                try:
                    result = run_flashvsr(
                        settings,
                        base_dir,
                        on_progress=lambda msg: progress_queue.put(msg),
                        cancel_event=_flashvsr_cancel_event,
                        process_handle=process_handle
                    )
                    result_holder["result"] = result
                except Exception as e:
                    result_holder["error"] = str(e)
            
            thread = threading.Thread(target=processing_thread, daemon=True)
            thread.start()
            
            # Apply face restoration if globally enabled
            face_apply = global_settings.get("face_global", False)
            face_strength = global_settings.get("face_strength", 0.5)
            
            # Stream progress updates
            last_update = time.time()
            log_buffer = []
            
            while thread.is_alive() or not progress_queue.empty():
                # Check for cancellation
                if _flashvsr_cancel_event.is_set():
                    # Kill the subprocess if still running
                    if process_handle.get("proc"):
                        try:
                            import platform
                            proc = process_handle["proc"]
                            if platform.system() == "Windows":
                                proc.terminate()
                            else:
                                proc.kill()
                        except Exception:
                            pass
                    
                    if progress:
                        progress(0, desc="Cancelled")
                    
                    # Try to salvage partial outputs (mirrors SeedVR2/GAN behavior)
                    compiled_output = None
                    temp_base = Path(global_settings.get("temp_dir", temp_dir))
                    temp_chunks_dir = temp_base / "chunks"
                    
                    if temp_chunks_dir.exists():
                        try:
                            from shared.chunking import detect_resume_state, concat_videos
                            from shared.path_utils import collision_safe_path
                            import shutil
                            
                            # Check for completed video chunks
                            partial_video, completed_chunks = detect_resume_state(temp_chunks_dir, "mp4")
                            
                            if completed_chunks and len(completed_chunks) > 0:
                                partial_target = collision_safe_path(temp_chunks_dir / "cancelled_flashvsr_partial.mp4")
                                if concat_videos(completed_chunks, partial_target):
                                    final_output = Path(output_dir) / f"cancelled_flashvsr_partial_upscaled.mp4"
                                    final_output = collision_safe_path(final_output)
                                    shutil.copy2(partial_target, final_output)
                                    compiled_output = str(final_output)
                                    log_buffer.append(f"\n✅ Partial output salvaged: {final_output.name}")
                        except Exception as e:
                            log_buffer.append(f"\n⚠️ Could not salvage partials: {str(e)}")
                    
                    status_msg = "⏹️ Processing cancelled"
                    if compiled_output:
                        status_msg += f" - Partial output saved: {Path(compiled_output).name}"
                    
                    yield (
                        status_msg,
                        "\n".join(log_buffer[-50:]) + "\n\n[Cancelled by user]",
                        None,
                        None,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state
                    )
                    return
                
                try:
                    msg = progress_queue.get(timeout=0.1)
                    log_buffer.append(msg)
                    
                    # Update gr.Progress from messages
                    if progress and "%" in msg:
                        import re
                        match = re.search(r'(\d+)%', msg)
                        if match:
                            pct = int(match.group(1)) / 100.0
                            progress(pct, desc=msg[:100])
                    
                except queue.Empty:
                    pass
                
                # Yield updates every 0.5s
                now = time.time()
                if now - last_update > 0.5:
                    last_update = now
                    yield (
                        "⚙️ Processing with FlashVSR+...",
                        "\n".join(log_buffer[-50:]),
                        None,
                        None,
                        gr.update(visible=False),
                        gr.update(value="Processing...", visible=False),
                        state
                    )
            
            thread.join()
            
            # Get result
            if "error" in result_holder:
                if progress:
                    progress(0, desc="Error")
                yield (
                    "❌ Processing failed",
                    f"Error: {result_holder['error']}",
                    None,
                    None,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            result = result_holder.get("result")
            if not result:
                yield (
                    "❌ No result",
                    "Processing did not complete",
                    None,
                    None,
                    gr.update(visible=False),
                    gr.update(value="", visible=False),
                    state
                )
                return
            
            # Update progress to 100%
            if progress:
                progress(1.0, desc="FlashVSR+ complete!")
            
            # Apply face restoration if globally enabled
            output_path = result.output_path
            face_apply = global_settings.get("face_global", False)
            face_strength = global_settings.get("face_strength", 0.5)
            
            if face_apply and output_path and Path(output_path).exists():
                from shared.face_restore import restore_video, restore_image
                
                log_buffer.append(f"Applying face restoration (strength {face_strength})...")
                
                if Path(output_path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    # Video restoration
                    restored = restore_video(
                        output_path,
                        strength=face_strength,
                        on_progress=lambda x: log_buffer.append(x) if x else None
                    )
                    if restored and Path(restored).exists():
                        output_path = restored
                        log_buffer.append(f"✅ Face restoration complete: {restored}")
                else:
                    # Image restoration
                    restored = restore_image(output_path, strength=face_strength)
                    if restored and Path(restored).exists():
                        output_path = restored
                        log_buffer.append(f"✅ Face restoration complete: {restored}")
            
            # Create comparison
            html_comp, img_slider = create_unified_comparison(
                input_path=input_path,
                output_path=output_path,
                mode="slider" if output_path and output_path.endswith(".mp4") else "native"
            )
            
            # Track output path for pinned comparison feature
            if output_path:
                try:
                    outp = Path(output_path)
                    seed_controls = state.get("seed_controls", {})
                    seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                    seed_controls["last_output_path"] = str(outp) if outp.is_file() else None
                    state["seed_controls"] = seed_controls
                except Exception:
                    pass
            
            # Log run
            run_logger.write_summary(
                Path(output_path) if output_path else output_dir,
                {
                    "input": input_path,
                    "output": output_path,
                    "returncode": result.returncode,
                    "args": settings,
                    "face_apply": face_apply,
                    "face_strength": face_strength,
                    "pipeline": "flashvsr",
                }
            )
            
            status = "✅ FlashVSR+ upscaling complete" if result.returncode == 0 else f"⚠️ Exited with code {result.returncode}"
            
            yield (
                status,
                result.log,
                output_path if output_path and output_path.endswith(".mp4") else None,
                output_path if output_path and not output_path.endswith(".mp4") else None,
                img_slider if img_slider else gr.update(visible=False),
                html_comp if html_comp else gr.update(value="", visible=False),
                state
            )
            
        except Exception as e:
            if progress:
                progress(0, desc="Critical error")
            yield (
                "❌ Critical error",
                f"Error: {str(e)}",
                None,
                None,
                gr.update(visible=False),
                gr.update(value="", visible=False),
                state or {}
            )

    def cancel_action():
        """Cancel FlashVSR+ processing"""
        _flashvsr_cancel_event.set()
        return gr.update(value="⏹️ Cancellation requested - FlashVSR+ will stop at next checkpoint"), "Cancelling..."

    def open_outputs_folder_flashvsr():
        """Open outputs folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import open_outputs_folder
        return open_outputs_folder(str(output_dir))
    
    def clear_temp_folder_flashvsr(confirm: bool):
        """Clear temp folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import clear_temp_folder
        return clear_temp_folder(str(temp_dir), confirm)

    return {
        "defaults": defaults,
        "order": FLASHVSR_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "cancel_action": cancel_action,
        "open_outputs_folder": open_outputs_folder_flashvsr,
        "clear_temp_folder": clear_temp_folder_flashvsr,
    }

