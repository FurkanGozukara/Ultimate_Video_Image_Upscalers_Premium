"""
FlashVSR+ Service Module
Handles FlashVSR+ processing logic, presets, and callbacks
"""

import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.flashvsr_runner import run_flashvsr, FlashVSRResult
from shared.path_utils import (
    normalize_path,
    collision_safe_path,
    collision_safe_dir,
    get_media_dimensions,
    detect_input_type,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
)
from shared.logging_utils import RunLogger
from shared.comparison_unified import create_unified_comparison
from shared.models.flashvsr_meta import get_flashvsr_metadata, get_flashvsr_default_model
from shared.gpu_utils import expand_cuda_device_spec, validate_cuda_device_spec
from shared.error_handling import logger as error_logger
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from shared.oom_alert import clear_vram_oom_alert, maybe_set_vram_oom_alert, show_vram_oom_modal

# Cancel event for FlashVSR+ processing
_flashvsr_cancel_event = threading.Event()


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

        base = outp.stem if outp.suffix else outp.name

        if pre_path.is_dir():
            dest_dir = collision_safe_dir(pre_dir / base)
            shutil.copytree(pre_path, dest_dir, dirs_exist_ok=False)
            return str(dest_dir)

        dest_file = collision_safe_path(pre_dir / f"{base}{pre_path.suffix}")
        shutil.copy2(pre_path, dest_file)
        return str(dest_file)
    except Exception:
        return None


def flashvsr_defaults(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default FlashVSR+ settings aligned with CLI defaults.
    Applies model-specific metadata when model_name is provided.
    """
    # IMPORTANT: do not import torch in the parent Gradio process.
    # Use NVML-based detection (nvidia-smi) via shared.gpu_utils instead.
    try:
        from shared.gpu_utils import get_gpu_info
        cuda_default = "auto" if get_gpu_info() else "cpu"
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
        # vNext sizing (app-level)
        "use_resolution_tab": True,
        "upscale_factor": float(scale),
        "max_target_resolution": 0,  # Max edge cap (0 = no cap)
        "pre_downscale_then_upscale": False,
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
    # vNext sizing
    "use_resolution_tab",
    "upscale_factor",
    "max_target_resolution",
    "pre_downscale_then_upscale",
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
                from shared.gpu_utils import get_gpu_info

                device_count = len(get_gpu_info())
                device_id_str = str(cfg.get("device", "")).replace("cuda:", "").strip()
                if device_id_str.isdigit():
                    device_id = int(device_id_str)
                    if device_count == 0 or device_id >= device_count:
                        error_logger.warning(
                            f"Device ID {device_id} not available (detected {device_count} GPU(s)) - falling back to auto"
                        )
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
            # Clear any previous VRAM OOM banner at the start of a new run.
            clear_vram_oom_alert(state)
            seed_controls = state.get("seed_controls", {})
            settings_dict = _flashvsr_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            
            # Apply FlashVSR+ guardrails (single GPU, tile validation, etc.)
            settings = _enforce_flashvsr_guardrails(settings, defaults)
            use_global_resolution = bool(settings.get("use_resolution_tab", True))
            
            # Apply shared Resolution & Scene Split tab settings (vNext Upscale-x)
            if use_global_resolution:
                if seed_controls.get("upscale_factor_val") is not None:
                    try:
                        settings["upscale_factor"] = float(seed_controls["upscale_factor_val"])
                    except Exception:
                        pass
                if seed_controls.get("max_resolution_val") is not None:
                    settings["max_target_resolution"] = int(seed_controls["max_resolution_val"] or 0)
                # Repurposed global flag: "pre-downscale then upscale when capped"
                if "ratio_downscale" in seed_controls:
                    settings["pre_downscale_then_upscale"] = bool(seed_controls.get("ratio_downscale", False))
            
            # Apply Output tab cached settings
            if seed_controls.get("fps_override_val") is not None and seed_controls["fps_override_val"] > 0:
                settings["fps"] = seed_controls["fps_override_val"]
            if seed_controls.get("comparison_mode_val"):
                settings["_comparison_mode"] = seed_controls["comparison_mode_val"]
            if seed_controls.get("save_metadata_val") is not None:
                settings["save_metadata"] = seed_controls["save_metadata_val"]
            
            # Clear cancel event
            _flashvsr_cancel_event.clear()

            def _apply_vnext_preprocess(cfg: Dict[str, Any], src_input_path: str) -> None:
                """
                vNext preprocessing for fixed-scale FlashVSR models (2x/4x).

                If the requested upscale (and/or max-edge cap) implies an effective scale < model_scale,
                pre-downscale the input so the model still runs at its native scale without post-downscale.

                Supports:
                - Video files (ffmpeg scale)
                - Frame directories (resize images into a temp directory)
                """
                try:
                    dims = get_media_dimensions(src_input_path)
                    if not dims:
                        return
                    w, h = dims
                    model_scale = int(cfg.get("scale", 4) or 4)
                    requested_scale = float(cfg.get("upscale_factor") or float(model_scale))
                    max_edge = int(cfg.get("max_target_resolution", 0) or 0)

                    # Respect global enable_max_target when using Resolution tab
                    if use_global_resolution and not bool(seed_controls.get("enable_max_target", True)):
                        max_edge = 0

                    plan = estimate_fixed_scale_upscale_plan_from_dims(
                        int(w),
                        int(h),
                        requested_scale=requested_scale,
                        model_scale=model_scale,
                        max_edge=max_edge,
                        force_pre_downscale=True,
                    )

                    if plan.preprocess_scale >= 0.999999:
                        return

                    temp_root = Path(global_settings.get("temp_dir", temp_dir))
                    temp_root.mkdir(parents=True, exist_ok=True)

                    in_type = detect_input_type(src_input_path)

                    if in_type == "video":
                        pre_out = collision_safe_path(
                            temp_root
                            / f"{Path(src_input_path).stem}_pre{int(plan.preprocess_width)}x{int(plan.preprocess_height)}.mp4"
                        )
                        if progress:
                            progress(0, desc=f"Preprocessing input → {int(plan.preprocess_width)}×{int(plan.preprocess_height)}")
                        cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            src_input_path,
                            "-vf",
                            f"scale={int(plan.preprocess_width)}:{int(plan.preprocess_height)}:flags=lanczos",
                            str(pre_out),
                        ]
                        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if pre_out.exists():
                            cfg["_original_input_path_before_preprocess"] = src_input_path
                            cfg["_preprocessed_input_path"] = str(pre_out)
                            cfg["_effective_input_path"] = str(pre_out)
                        return

                    if in_type == "directory":
                        src_dir = Path(src_input_path)
                        img_files = [p for p in sorted(src_dir.iterdir()) if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
                        if not img_files:
                            return

                        pre_dir = collision_safe_dir(
                            temp_root
                            / f"{src_dir.name}_pre{int(plan.preprocess_width)}x{int(plan.preprocess_height)}"
                        )
                        pre_dir.mkdir(parents=True, exist_ok=True)

                        if progress:
                            progress(0, desc=f"Preprocessing frames → {int(plan.preprocess_width)}×{int(plan.preprocess_height)}")

                        # Prefer OpenCV for speed; fall back to Pillow.
                        try:
                            import cv2  # type: ignore
                            for f in img_files:
                                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                                if img is None:
                                    continue
                                resized = cv2.resize(
                                    img,
                                    (int(plan.preprocess_width), int(plan.preprocess_height)),
                                    interpolation=cv2.INTER_AREA,
                                )
                                cv2.imwrite(str(pre_dir / f.name), resized)
                        except Exception:
                            try:
                                from PIL import Image  # type: ignore
                                for f in img_files:
                                    with Image.open(f) as im:
                                        im2 = im.resize(
                                            (int(plan.preprocess_width), int(plan.preprocess_height)),
                                            resample=Image.LANCZOS,
                                        )
                                        im2.save(pre_dir / f.name)
                            except Exception:
                                return

                        if any(pre_dir.iterdir()):
                            cfg["_original_input_path_before_preprocess"] = src_input_path
                            cfg["_preprocessed_input_path"] = str(pre_dir)
                            cfg["_effective_input_path"] = str(pre_dir)
                        return
                except Exception:
                    return
            
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

            # -------------------------------------------------------------
            # ✅ Batch processing (folder of videos and/or frame directories)
            # -------------------------------------------------------------
            if bool(settings.get("batch_enable")):
                batch_in = normalize_path(settings.get("batch_input_path") or "")
                batch_out = normalize_path(settings.get("batch_output_path") or "") if settings.get("batch_output_path") else ""

                if not batch_in or not Path(batch_in).exists() or not Path(batch_in).is_dir():
                    yield (
                        "❌ Batch input folder missing/invalid",
                        "Provide a valid Batch Input Folder path.",
                        None,
                        None,
                        gr.update(visible=False),
                        gr.update(value="", visible=False),
                        state,
                    )
                    return

                if batch_out:
                    try:
                        Path(batch_out).mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass

                in_dir = Path(batch_in)
                items: List[Path] = [
                    p
                    for p in sorted(in_dir.iterdir())
                    if (p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS) or p.is_dir()
                ]
                if not items:
                    # If the batch folder itself is a frame directory, treat it as a single job.
                    items = [in_dir]

                face_apply = bool(global_settings.get("face_global", False))
                face_strength = float(global_settings.get("face_strength", 0.5))

                logs: List[str] = []
                outputs: List[str] = []
                last_input_path: Optional[str] = None
                last_output_path: Optional[str] = None

                if progress:
                    progress(0, desc=f"Batch: {len(items)} item(s) queued")

                for idx, item in enumerate(items, 1):
                    if _flashvsr_cancel_event.is_set():
                        yield (
                            "⏹️ Batch cancelled",
                            "\n".join(logs[-200:]) + "\n\n[Cancelled by user]",
                            None,
                            None,
                            gr.update(visible=False),
                            gr.update(value="", visible=False),
                            state,
                        )
                        return

                    item_path = str(item)
                    last_input_path = item_path

                    if progress:
                        progress((idx - 1) / max(1, len(items)), desc=f"Batch {idx}/{len(items)}: {Path(item_path).name}")

                    item_settings = settings.copy()
                    item_settings["batch_enable"] = False
                    item_settings["input_path"] = item_path
                    item_settings["_effective_input_path"] = item_path
                    item_settings["_original_filename"] = Path(item_path).name
                    item_settings["global_output_dir"] = str(output_dir)
                    if batch_out:
                        item_settings["output_override"] = batch_out

                    _apply_vnext_preprocess(item_settings, item_path)

                    result = run_flashvsr(
                        item_settings,
                        base_dir,
                        on_progress=None,
                        cancel_event=_flashvsr_cancel_event,
                        process_handle=None,
                    )

                    outp = result.output_path
                    if outp and Path(outp).exists():
                        # Optional face restoration (video-first)
                        if face_apply:
                            try:
                                from shared.face_restore import restore_video, restore_image
                                if Path(outp).suffix.lower() in [".mp4", ".avi", ".mov", ".mkv"]:
                                    restored = restore_video(
                                        outp,
                                        strength=face_strength,
                                        on_progress=lambda x: logs.append(x) if x else None,
                                    )
                                    if restored and Path(restored).exists():
                                        outp = restored
                                else:
                                    restored_img = restore_image(outp, strength=face_strength)
                                    if restored_img and Path(restored_img).exists():
                                        outp = restored_img
                            except Exception:
                                pass

                        # Save preprocessed input (if created) alongside outputs
                        pre_in = item_settings.get("_preprocessed_input_path")
                        if pre_in and outp:
                            saved_pre = _save_preprocessed_artifact(Path(pre_in), outp)
                            if saved_pre:
                                logs.append(f"🧩 Preprocessed input saved: {saved_pre}")

                        outputs.append(outp)
                        last_output_path = outp
                        logs.append(f"✅ [{idx}/{len(items)}] {Path(item_path).name} → {Path(outp).name}")
                    else:
                        if getattr(result, "returncode", 0) != 0:
                            maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=getattr(result, "log", ""), settings=item_settings)
                        logs.append(f"❌ [{idx}/{len(items)}] {Path(item_path).name} failed")

                    # Log run summary per-item
                    try:
                        run_logger.write_summary(
                            Path(outp) if outp else output_dir,
                            {
                                "input": item_path,
                                "output": outp,
                                "returncode": result.returncode,
                                "args": item_settings,
                                "face_apply": face_apply,
                                "face_strength": face_strength,
                                "pipeline": "flashvsr",
                                "batch": True,
                            },
                        )
                    except Exception:
                        pass

                if progress:
                    progress(1.0, desc=f"Batch complete ({len(outputs)}/{len(items)} succeeded)")

                # Track output path for pinned comparison feature (last output)
                if last_output_path:
                    try:
                        outp_path = Path(last_output_path)
                        seed_controls["last_output_dir"] = str(outp_path.parent if outp_path.is_file() else outp_path)
                        seed_controls["last_output_path"] = str(outp_path) if outp_path.is_file() else None
                        state["seed_controls"] = seed_controls
                    except Exception:
                        pass

                # Comparison for last item only
                html_comp = gr.update(value="", visible=False)
                img_slider = gr.update(visible=False)
                if last_input_path and last_output_path:
                    try:
                        h, sld = create_unified_comparison(
                            input_path=last_input_path,
                            output_path=last_output_path,
                            mode="slider" if last_output_path.endswith(".mp4") else "native",
                        )
                        html_comp = h if h else gr.update(value="", visible=False)
                        img_slider = sld if sld else gr.update(visible=False)
                    except Exception:
                        pass

                status = f"✅ FlashVSR+ batch complete ({len(outputs)}/{len(items)} succeeded, {len(items) - len(outputs)} failed)"
                yield (
                    status,
                    "\n".join(logs),
                    last_output_path if last_output_path and last_output_path.endswith(".mp4") else None,
                    last_output_path if last_output_path and not last_output_path.endswith(".mp4") else None,
                    img_slider if img_slider else gr.update(visible=False),
                    html_comp if html_comp else gr.update(value="", visible=False),
                    state,
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
            settings["_effective_input_path"] = input_path  # may be overridden by preprocessing
            settings["_original_filename"] = Path(input_path).name
            settings["global_output_dir"] = str(output_dir)
            _apply_vnext_preprocess(settings, input_path)
            
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
                if maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=result_holder.get("error", ""), settings=settings):
                    show_vram_oom_modal(state, title="Out of VRAM (GPU) — FlashVSR+", duration=None)
                yield (
                    ("🚫 Out of VRAM (GPU) — see banner above" if state.get("alerts", {}).get("oom", {}).get("visible") else "❌ Processing failed"),
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
                    restored_img = restore_image(output_path, strength=face_strength)
                    if restored_img and Path(restored_img).exists():
                        output_path = restored_img
                        log_buffer.append(f"✅ Face restoration complete: {restored_img}")

            # Save preprocessed input (if we created one) alongside outputs
            pre_in = settings.get("_preprocessed_input_path")
            if pre_in and output_path:
                saved_pre = _save_preprocessed_artifact(Path(pre_in), output_path)
                if saved_pre:
                    log_buffer.append(f"🧩 Preprocessed input saved: {saved_pre}")
            
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
            if result.returncode != 0 and maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=result.log, settings=settings):
                status = "🚫 Out of VRAM (GPU) — see banner above"
                show_vram_oom_modal(state, title="Out of VRAM (GPU) — FlashVSR+", duration=None)
            
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
            if maybe_set_vram_oom_alert(state, model_label="FlashVSR+", text=str(e), settings=locals().get("settings")):
                show_vram_oom_modal(state, title="Out of VRAM (GPU) — FlashVSR+", duration=None)
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

