"""
RIFE Service Module - Clean Implementation
Handles RIFE/FPS/Edit Videos processing logic, presets, and callbacks
"""

import shutil
import subprocess
import queue
import threading
import time
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import gradio as gr

from shared.preset_manager import PresetManager
from shared.runner import Runner
from shared.path_utils import normalize_path, ffmpeg_set_fps, get_media_dimensions
from shared.face_restore import restore_video
from shared.logging_utils import RunLogger
from shared.models.rife_meta import get_rife_metadata, get_rife_default_model
from shared.gpu_utils import expand_cuda_device_spec, validate_cuda_device_spec
from shared.error_handling import logger as error_logger


# Defaults and ordering --------------------------------------------------------
def rife_defaults(model_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Get default RIFE settings aligned with RIFE CLI.
    Applies model-specific metadata when model_name is provided.
    """
    try:
        import torch
        cuda_default = "0" if torch.cuda.is_available() else ""
    except Exception:
        cuda_default = ""
    
    # Get model metadata if specific model is provided
    default_model = model_name or get_rife_default_model()
    model_meta = get_rife_metadata(default_model)
    
    # Apply model-specific defaults if metadata available
    if model_meta:
        default_precision = model_meta.default_precision
        recommended_uhd = model_meta.recommended_uhd_threshold
    else:
        default_precision = "fp16"
        recommended_uhd = 2160
    
    return {
        "input_path": "",
        "rife_enabled": True,
        "output_override": "",
        "output_format": "mp4",
        "model_dir": "",
        "model": default_model,  # Fixed: was "rife_model", should match RIFE_ORDER
        "fps_multiplier": "x2",
        "fps_override": 0,  # Fixed: was "target_fps", should match RIFE_ORDER
        "scale": 1.0,
        "uhd_mode": False,
        "fp16_mode": default_precision == "fp16",  # Fixed: was "rife_precision", should be boolean
        "png_output": False,
        "no_audio": False,
        "show_ffmpeg": False,
        "montage": False,
        "img_mode": False,
        "skip_static_frames": False,
        "exp": 1,
        "multi": 2,
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "skip_first_frames": 0,
        "load_cap": 0,
        "cuda_device": cuda_default,
        # Video editing parameters
        "edit_mode": "none",
        "start_time": "",
        "end_time": "",
        "speed_factor": 1.0,
        "video_codec": "libx264",
        "output_quality": 23,
        "concat_videos": "",
        "_recommended_uhd_threshold": recommended_uhd,  # Store for validation
    }


"""
📋 RIFE PRESET ORDER - MUST match inputs_list in ui/rife_tab.py
Adding controls? Update rife_defaults(), RIFE_ORDER, and inputs_list in sync.
Current count: 32 components
"""

RIFE_ORDER: List[str] = [
    "input_path",
    "rife_enabled",
    "output_override",
    "output_format",
    "model_dir",
    "model",
    "fps_multiplier",
    "fps_override",
    "scale",
    "uhd_mode",
    "fp16_mode",
    "png_output",
    "no_audio",
    "show_ffmpeg",
    "montage",
    "img_mode",
    "skip_static_frames",
    "exp",
    "multi",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "skip_first_frames",
    "load_cap",
    "cuda_device",
    # Video editing parameters
    "edit_mode",
    "start_time",
    "end_time",
    "speed_factor",
    "video_codec",
    "output_quality",
    "concat_videos",
]


# Video editing functions -----------------------------------------------------

def _parse_time_to_seconds(time_str: str) -> float:
    """Parse time string in HH:MM:SS or seconds format to total seconds."""
    if not time_str or time_str.strip() == "":
        return 0.0

    time_str = time_str.strip()

    # Check if it's already a number (seconds)
    try:
        return float(time_str)
    except ValueError:
        pass

    # Parse HH:MM:SS format
    match = re.match(r'^(\d{1,2}):(\d{1,2}):(\d{1,2}(?:\.\d+)?)$', time_str)
    if match:
        hours, minutes, seconds = match.groups()
        return float(hours) * 3600 + float(minutes) * 60 + float(seconds)

    # Try MM:SS format
    match = re.match(r'^(\d{1,2}):(\d{1,2}(?:\.\d+)?)$', time_str)
    if match:
        minutes, seconds = match.groups()
        return float(minutes) * 60 + float(seconds)

    # If nothing matches, return 0
    return 0.0


def _trim_video(input_path: str, output_path: str, start_time: float, end_time: float,
                video_codec: str = "libx264", quality: int = 23, no_audio: bool = False) -> Tuple[bool, str]:
    """Trim video using FFmpeg."""
    try:
        cmd = ["ffmpeg", "-y"]

        # Input
        cmd.extend(["-i", input_path])

        # Time range
        if start_time > 0:
            cmd.extend(["-ss", str(start_time)])
        if end_time > 0:
            cmd.extend(["-to", str(end_time)])

        # Video codec and quality
        if video_codec == "libx264":
            cmd.extend(["-c:v", "libx264", "-crf", str(quality), "-preset", "fast"])
        elif video_codec == "libx265":
            cmd.extend(["-c:v", "libx265", "-crf", str(quality), "-preset", "fast"])
        elif video_codec == "libvpx-vp9":
            cmd.extend(["-c:v", "libvpx-vp9", "-crf", str(quality)])

        # Audio handling
        if no_audio:
            cmd.extend(["-an"])
        else:
            cmd.extend(["-c:a", "aac"])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return result.returncode == 0, result.stderr or result.stdout

    except Exception as e:
        return False, f"Trimming failed: {str(e)}"


def _change_video_speed(input_path: str, output_path: str, speed_factor: float,
                       video_codec: str = "libx264", quality: int = 23, no_audio: bool = False) -> Tuple[bool, str]:
    """Change video speed using FFmpeg."""
    try:
        if speed_factor <= 0:
            return False, "Speed factor must be greater than 0"

        # For speed changes, we need to adjust both video and audio
        # Video filter for speed change
        video_filter = f"setpts={1/speed_factor}*PTS"

        # Audio filter for speed change (if not removing audio)
        audio_filter = ""
        if not no_audio:
            audio_filter = f"atempo={speed_factor}"

        cmd = ["ffmpeg", "-y", "-i", input_path]

        # Apply filters
        if video_filter:
            cmd.extend(["-filter:v", video_filter])
        if audio_filter:
            cmd.extend(["-filter:a", audio_filter])

        # Video codec and quality
        if video_codec == "libx264":
            cmd.extend(["-c:v", "libx264", "-crf", str(quality), "-preset", "fast"])
        elif video_codec == "libx265":
            cmd.extend(["-c:v", "libx265", "-crf", str(quality), "-preset", "fast"])
        elif video_codec == "libvpx-vp9":
            cmd.extend(["-c:v", "libvpx-vp9", "-crf", str(quality)])

        # Audio handling
        if no_audio:
            cmd.extend(["-an"])
        elif audio_filter:
            cmd.extend(["-c:a", "aac"])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        return result.returncode == 0, result.stderr or result.stdout

    except Exception as e:
        return False, f"Speed change failed: {str(e)}"


def _concatenate_videos(video_paths: List[str], output_path: str,
                       video_codec: str = "libx264", quality: int = 23) -> Tuple[bool, str]:
    """Concatenate multiple videos using FFmpeg."""
    try:
        if len(video_paths) < 2:
            return False, "Need at least 2 videos to concatenate"

        # Create a temporary concat file
        concat_file = output_path + ".concat.txt"
        with open(concat_file, 'w', encoding='utf-8') as f:
            for video_path in video_paths:
                f.write(f"file '{video_path.replace(chr(39), chr(39) + chr(39))}'\n")

        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file]

        # Video codec and quality
        if video_codec == "libx264":
            cmd.extend(["-c:v", "libx264", "-crf", str(quality), "-preset", "fast"])
        elif video_codec == "libx265":
            cmd.extend(["-c:v", "libx265", "-crf", str(quality), "-preset", "fast"])
        elif video_codec == "libvpx-vp9":
            cmd.extend(["-c:v", "libvpx-vp9", "-crf", str(quality)])

        # Copy audio streams (they should be compatible)
        cmd.extend(["-c:a", "copy"])

        cmd.append(output_path)

        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        # Clean up concat file
        try:
            Path(concat_file).unlink(missing_ok=True)
        except:
            pass

        return result.returncode == 0, result.stderr or result.stdout

    except Exception as e:
        # Clean up concat file on error
        try:
            Path(concat_file).unlink(missing_ok=True)
        except:
            pass
        return False, f"Concatenation failed: {str(e)}"


def _apply_video_editing(input_path: str, output_path: str, settings: Dict[str, Any],
                        temp_dir: Path, progress_cb: Optional[Callable[[str], None]] = None) -> Tuple[bool, str, str]:
    """
    Apply video editing operations to the input video.
    Returns (success, log_message, final_output_path)
    """
    edit_mode = settings.get("edit_mode", "none")
    if edit_mode == "none":
        # No editing, just copy the file
        shutil.copy2(input_path, output_path)
        return True, "No video editing applied", output_path

    try:
        # Parse common parameters
        start_time = _parse_time_to_seconds(settings.get("start_time", ""))
        end_time = _parse_time_to_seconds(settings.get("end_time", ""))
        speed_factor = float(settings.get("speed_factor", 1.0))
        video_codec = settings.get("video_codec", "libx264")
        quality = int(settings.get("output_quality", 23))
        no_audio = bool(settings.get("no_audio", False))

        temp_output = temp_dir / f"temp_edit_{Path(output_path).name}"

        if edit_mode == "trim":
            if progress_cb:
                progress_cb("Trimming video...")
            success, log = _trim_video(
                input_path, str(temp_output), start_time, end_time,
                video_codec, quality, no_audio
            )
            if not success:
                return False, f"Trimming failed: {log}", ""

        elif edit_mode == "speed_change":
            if progress_cb:
                progress_cb(f"Changing video speed by factor {speed_factor}...")
            success, log = _change_video_speed(
                input_path, str(temp_output), speed_factor,
                video_codec, quality, no_audio
            )
            if not success:
                return False, f"Speed change failed: {log}", ""

        elif edit_mode == "concatenate":
            # For concatenation, we need additional video paths
            concat_videos_str = settings.get("concat_videos", "").strip()
            if not concat_videos_str:
                return False, "Concatenation mode requires additional video paths", ""

            # Parse the concatenation videos (comma-separated paths)
            concat_paths = [path.strip() for path in concat_videos_str.split(",") if path.strip()]
            if not concat_paths:
                return False, "No valid video paths found for concatenation", ""

            # Add the main input video as the first video
            all_paths = [input_path] + concat_paths

            if progress_cb:
                progress_cb(f"Concatenating {len(all_paths)} videos...")
            success, log = _concatenate_videos(all_paths, str(temp_output), video_codec, quality)
            if not success:
                return False, f"Concatenation failed: {log}", ""

        elif edit_mode == "effects":
            # Placeholder for future effects
            if progress_cb:
                progress_cb("Applying video effects...")
            # For now, just copy the file
            shutil.copy2(input_path, temp_output)
            success, log = True, "Basic effects applied (placeholder)"

        else:
            return False, f"Unknown edit mode: {edit_mode}", ""

        if success:
            # Move temp output to final output
            temp_output.replace(output_path)
            return True, f"Video editing completed: {edit_mode}", output_path
        else:
            return False, log, ""

    except Exception as e:
        return False, f"Video editing error: {str(e)}", ""


# Helper functions -------------------------------------------------------------
def _validate_cuda_devices(cuda_spec: str) -> Optional[str]:
    """Validate CUDA device specification."""
    try:
        import torch
        
        if not cuda_spec:
            return None
        if not torch.cuda.is_available():
            return "CUDA is not available on this system."
        
        devices = [d.strip() for d in str(cuda_spec).split(",") if d.strip() != ""]
        count = torch.cuda.device_count()
        invalid = [d for d in devices if (not d.isdigit()) or int(d) >= count]
        if invalid:
            return f"Invalid CUDA device id(s): {', '.join(invalid)}. Available: 0-{count-1}"
    except Exception as exc:
        return f"CUDA validation failed: {exc}"
    return None


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is available in PATH."""
    return shutil.which("ffmpeg") is not None


# Preset helpers ---------------------------------------------------------------
def _rife_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    """Convert argument list to settings dictionary."""
    return dict(zip(RIFE_ORDER, args))


def _enforce_rife_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply RIFE-specific validation rules using metadata registry.
    
    Enforces:
    - Single GPU requirement (RIFE doesn't support multi-GPU)
    - UHD mode recommendations for 4K+ content
    - FPS multiplier constraints from model metadata
    - Precision compatibility checks
    """
    cfg = cfg.copy()
    
    # Get model metadata
    model_name = cfg.get("rife_model", get_rife_default_model())
    model_meta = get_rife_metadata(model_name)
    
    if model_meta:
        # Enforce single GPU (RIFE is single-GPU optimized)
        gpu_device_str = str(cfg.get("gpu_device", ""))
        if gpu_device_str and gpu_device_str not in ("", "cpu"):
            devices = [d.strip() for d in gpu_device_str.replace(" ", "").split(",") if d.strip()]
            if len(devices) > 1:
                error_logger.warning(f"RIFE doesn't support multi-GPU - forcing single GPU (using first: {devices[0]})")
                cfg["gpu_device"] = devices[0]
                cfg["_multi_gpu_disabled_reason"] = "RIFE is single-GPU optimized"
        
        # Validate FPS multiplier against model limits
        fps_mult_str = str(cfg.get("fps_multiplier", "x2"))
        try:
            # Extract numeric multiplier (e.g., "x2" -> 2)
            mult_value = int(fps_mult_str.replace("x", "").strip())
            max_mult = model_meta.max_fps_multiplier
            
            if mult_value > max_mult:
                error_logger.warning(f"FPS multiplier {mult_value}x exceeds model limit {max_mult}x, clamping")
                cfg["fps_multiplier"] = f"x{max_mult}"
                cfg["_fps_mult_clamped_reason"] = f"Model {model_name} max: {max_mult}x"
        except (ValueError, AttributeError):
            pass  # Keep original value if parsing fails
        
        # Auto-enable UHD mode for high-resolution content
        # This is a recommendation check - actual enabling happens in processing
        if model_meta.supports_uhd:
            cfg["_uhd_threshold"] = model_meta.recommended_uhd_threshold
    
    # Validate scale factor (must be positive)
    try:
        scale = float(cfg.get("scale", 1.0))
        if scale <= 0:
            cfg["scale"] = 1.0
            error_logger.warning(f"Invalid scale {scale}, resetting to 1.0")
    except (ValueError, TypeError):
        cfg["scale"] = 1.0
    
    return cfg


def _apply_rife_preset(
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
    # Apply guardrails to merged preset
    merged = _enforce_rife_guardrails(merged, defaults)
    return [merged[key] for key in RIFE_ORDER]


# Core run/cancel/preset callbacks --------------------------------------------
def build_rife_callbacks(
    preset_manager: PresetManager,
    runner: Runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    output_dir: Path,
    temp_dir: Path,
    shared_state: gr.State,
):
    """Build RIFE callback functions for the UI."""
    defaults = rife_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        """Refresh preset dropdown."""
        presets = preset_manager.list_presets("rife", model_name)
        last_used = preset_manager.get_last_used_name("rife", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save preset with validation"""
        if not preset_name.strip():
            return gr.update(), gr.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            # Validate component count
            if len(args) != len(RIFE_ORDER):
                error_msg = f"⚠️ Preset mismatch: {len(args)} values vs {len(RIFE_ORDER)} expected. Check inputs_list in rife_tab.py"
                return gr.update(), gr.update(value=error_msg), *list(args)
            
            payload = _rife_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("rife", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(RIFE_ORDER, list(args)))
            loaded_vals = _apply_rife_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        UI expects: inputs_list + [preset_status]
        """
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("rife", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("rife", model_name, preset_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(RIFE_ORDER, current_values))
            values = _apply_rife_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
            
            # Return values + status message (status is LAST)
            status_msg = f"✅ Loaded preset '{preset_name}'" if preset else "ℹ️ Preset not found"
            return (*values, gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status
            return (*current_values, gr.update(value=f"❌ Error: {str(e)}"))

    def safe_defaults():
        """Get safe default values."""
        return [defaults[k] for k in RIFE_ORDER]

    def run_action(uploaded_file, img_folder, *args, state=None):
        """Main RIFE processing action with pre-flight checks."""
        try:
            state = state or {"seed_controls": {}, "operation_status": "ready"}
            state["operation_status"] = "running"
            seed_controls = state.get("seed_controls", {})
            
            settings_dict = _rife_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            
            # Apply RIFE guardrails (single GPU, FPS limits, etc.)
            settings = _enforce_rife_guardrails(settings, defaults)

            # PRE-FLIGHT CHECKS (mirrors SeedVR2/GAN for consistency)
            from shared.error_handling import check_ffmpeg_available, check_disk_space
            
            # Check ffmpeg availability
            ffmpeg_ok, ffmpeg_msg = check_ffmpeg_available()
            if not ffmpeg_ok:
                yield ("❌ ffmpeg not found in PATH", ffmpeg_msg or "Install ffmpeg and add to PATH before processing", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return
            
            # Check disk space (require at least 5GB free)
            output_path_check = Path(global_settings.get("output_dir", output_dir))
            has_space, space_warning = check_disk_space(output_path_check, required_mb=5000)
            if not has_space:
                yield ("❌ Insufficient disk space", space_warning or "Free up at least 5GB disk space before processing", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            input_path = normalize_path(uploaded_file if uploaded_file else img_folder)
            if not input_path or not Path(input_path).exists():
                yield ("❌ Input missing or not found", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            # Validate input type based on mode
            if settings.get("img_mode"):
                # In --img mode, require a frames folder or images
                if Path(input_path).is_file() and Path(input_path).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                    yield ("⚠️ --img mode expects frames folder or images, not a video file.", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return
            else:
                # In video mode, require a video file
                if Path(input_path).is_dir():
                    yield ("⚠️ Video mode expects a video file. Enable --img for frame folders.", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

            settings["input_path"] = input_path
            settings["output_override"] = settings.get("output_override") or None

            # Expand "all" to device list if specified (using shared GPU utility)
            cuda_device_raw = settings.get("cuda_device", "")
            if cuda_device_raw:
                settings["cuda_device"] = expand_cuda_device_spec(cuda_device_raw)

            # Validate CUDA devices (using shared GPU utility)
            cuda_warning = validate_cuda_device_spec(settings.get("cuda_device", ""))
            if cuda_warning:
                yield (f"⚠️ {cuda_warning}", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            # Check ffmpeg availability
            if not _ffmpeg_available():
                yield ("❌ ffmpeg not found in PATH. Install ffmpeg and retry.", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                return

            # Apply cached values from Resolution & Scene Split tab
            if seed_controls.get("resolution_val") is not None:
                # For RIFE, resolution affects downscaling before processing
                target_resolution = int(seed_controls["resolution_val"])
                max_resolution = int(seed_controls.get("max_resolution_val", 0) or 0)
                
                # Calculate scale factor needed to reach target resolution
                input_dims = get_media_dimensions(input_path)
                if input_dims:
                    input_w, input_h = input_dims
                    short_side = min(input_w, input_h)
                    
                    # Apply max resolution cap if enabled
                    effective_target = target_resolution
                    enable_max = seed_controls.get("enable_max_target", True)
                    if enable_max and max_resolution > 0:
                        effective_target = min(target_resolution, max_resolution)
                    
                    # Calculate scale needed to reach target
                    if short_side > 0 and effective_target > 0:
                        calculated_scale = effective_target / short_side
                        # Clamp to reasonable range for RIFE (0.5x to 4.0x)
                        calculated_scale = max(0.5, min(4.0, calculated_scale))
                        settings["scale"] = calculated_scale

            # Apply output format from Comparison tab if set
            cached_fmt = seed_controls.get("output_format_val")
            if settings.get("output_format") in (None, "auto") and cached_fmt:
                settings["output_format"] = cached_fmt
            
            # Pull PySceneDetect chunking settings from Resolution tab (universal chunking)
            chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
            chunk_overlap_sec = float(seed_controls.get("chunk_overlap_sec", 0) or 0)
            per_chunk_cleanup = seed_controls.get("per_chunk_cleanup", False)
            # PySceneDetect parameters now managed centrally in Resolution tab
            scene_threshold = float(seed_controls.get("scene_threshold", 27.0))
            min_scene_len = float(seed_controls.get("min_scene_len", 2.0))
            
            # Determine if PySceneDetect chunking should be used
            from shared.path_utils import detect_input_type as detect_type
            input_type_check = detect_type(input_path)
            should_use_chunking = (
                chunk_size_sec > 0 and
                input_type_check == "video" and
                not settings.get("batch_enable", False) and
                not settings.get("img_mode", False)  # Don't chunk image sequences
            )
            
            # If chunking enabled, use universal chunk_and_process for RIFE
            if should_use_chunking:
                from shared.chunking import chunk_and_process
                
                yield ("⚙️ Starting PySceneDetect chunking for RIFE processing...", 
                       "Initializing scene detection...", gr.update(value="Chunking...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)
                
                # Prepare settings for chunking
                settings["chunk_size_sec"] = chunk_size_sec
                settings["chunk_overlap_sec"] = chunk_overlap_sec
                settings["per_chunk_cleanup"] = per_chunk_cleanup
                
                def chunk_progress_cb(progress_val, desc=""):
                    yield (f"⚙️ Chunking: {desc}", f"Processing chunks... {desc}", gr.update(value=desc, visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)
                
                # Run chunked RIFE processing
                rc, clog, final_output, chunk_count = chunk_and_process(
                    runner=runner,
                    settings=settings,
                    scene_threshold=scene_threshold,
                    min_scene_len=min_scene_len,
                    temp_dir=temp_dir,
                    on_progress=lambda msg: None,
                    chunk_seconds=chunk_size_sec,
                    chunk_overlap=chunk_overlap_sec,
                    per_chunk_cleanup=per_chunk_cleanup,
                    allow_partial=True,
                    global_output_dir=str(output_dir),
                    resume_from_partial=False,
                    progress_tracker=chunk_progress_cb,
                    process_func=None,
                    model_type="rife",  # Route to runner.run_rife
                )
                
                status = "✅ RIFE chunked processing complete" if rc == 0 else f"⚠️ RIFE chunking failed (code {rc})"
                
                # Build comparison for chunked output
                video_comp_html_update = gr.update(value="", visible=False)
                image_slider_update = gr.update(value=None)
                
                if final_output and Path(final_output).exists():
                    if Path(final_output).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                        from shared.video_comparison_slider import create_video_comparison_html
                        video_comp_html_value = create_video_comparison_html(
                            original_video=settings["input_path"],
                            upscaled_video=final_output,
                            height=600,
                            slider_position=50.0
                        )
                        video_comp_html_update = gr.update(value=video_comp_html_value, visible=True)
                    elif not Path(final_output).is_dir():
                        image_slider_update = gr.update(value=(settings["input_path"], final_output), visible=True)
                
                meta_md = f"PySceneDetect chunking: {chunk_count} chunks processed\nOutput: {final_output}"
                
                yield (status, clog, gr.update(value="", visible=False), final_output if final_output and Path(final_output).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv') else None, image_slider_update, video_comp_html_update, state)
                return

            # Check for batch processing
            if settings.get("batch_enable"):
                # Use the batch processor for multiple files
                from shared.batch_processor import BatchProcessor, BatchJob

                batch_input_path = Path(settings.get("batch_input_path", ""))
                batch_output_path = Path(settings.get("batch_output_path", ""))

                if not batch_input_path.exists():
                    yield ("❌ Batch input path does not exist", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

                # Collect all video files for RIFE
                rife_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
                batch_files = []
                if batch_input_path.is_dir():
                    for ext in rife_exts:
                        batch_files.extend(batch_input_path.glob(f"**/*{ext}"))
                elif batch_input_path.suffix.lower() in rife_exts:
                    batch_files = [batch_input_path]

                if not batch_files:
                    yield ("❌ No supported video files found in batch input", "", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
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
                            "face_apply": bool(global_settings.get("face_global", False)),
                            "face_strength": float(global_settings.get("face_strength", 0.5)),
                        }
                    )
                    jobs.append(job)

                # Process batch with progress updates
                def batch_progress_callback(progress_data):
                    current_job = progress_data.get("current_job")
                    overall_progress = progress_data.get("overall_progress", 0)
                    status_msg = f"Batch RIFE processing: {overall_progress:.1f}% complete"
                    if current_job:
                        status_msg += f" - Processing: {Path(current_job).name}"

                    yield (status_msg, f"Processing {len(jobs)} videos...", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)

                # Define processing function for each job
                def process_single_rife_job(job: BatchJob, progress_cb):
                    try:
                        job.status = "processing"
                        job.start_time = time.time()

                        # Process single file with current settings
                        single_settings = job.metadata["settings"].copy()
                        single_settings["input_path"] = job.input_path
                        single_settings["batch_enable"] = False  # Disable batch for individual processing

                        result = runner.run_rife(single_settings, on_progress=lambda x: progress_cb(x) if progress_cb else None)

                        if result.output_path and Path(result.output_path).exists():
                            job.output_path = result.output_path
                            job.status = "completed"

                            # Apply face restoration if enabled
                            if job.metadata["face_apply"] and Path(job.output_path).exists():
                                restored = restore_video(
                                    job.output_path,
                                    strength=job.metadata["face_strength"],
                                    on_progress=lambda x: progress_cb(f"Applying face restoration...") if progress_cb else None
                                )
                                if restored:
                                    job.output_path = restored
                        else:
                            job.status = "failed"
                            job.error_message = result.log

                        job.end_time = time.time()

                    except Exception as e:
                        job.status = "failed"
                        job.error_message = str(e)
                        job.end_time = time.time()

                    return job

                # Run batch processing
                results = batch_processor.process_batch(
                    jobs=jobs,
                    process_func=process_single_rife_job,
                    progress_callback=batch_progress_callback
                )

                # Summarize results
                completed = sum(1 for r in results if r.status == "completed")
                failed = sum(1 for r in results if r.status == "failed")

                summary_msg = f"RIFE batch complete: {completed}/{len(jobs)} succeeded"
                if failed > 0:
                    summary_msg += f", {failed} failed"

                yield (f"✅ {summary_msg}", f"Batch processing finished. Check output folder for results.", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
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
                    yield (f"⚙️ Processing: {message}", f"Progress: {message}", gr.update(value=message, visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)

            # Start processing with progress tracking
            yield ("⚙️ Starting processing...", "Initializing...", gr.update(value="Initializing...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)

            # Determine processing workflow
            edit_mode = settings.get("edit_mode", "none")
            rife_enabled = settings.get("rife_enabled", False) or settings.get("fps_multiplier", 1.0) > 1.0 or settings.get("fps_override", 0.0) > 0.0

            current_input = settings["input_path"]
            final_output_path = None

            # Step 1: Apply video editing (if any)
            if edit_mode != "none":
                yield ("⚙️ Applying video editing...", "Processing video edits...", gr.update(value="Video editing in progress...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)

                edit_temp_output = temp_dir / f"edit_temp_{Path(current_input).stem}_{int(time.time())}.mp4"
                edit_success, edit_log, edited_path = _apply_video_editing(
                    current_input, str(edit_temp_output), settings, temp_dir,
                    lambda msg: progress_callback(f"Editing: {msg}")
                )

                if not edit_success:
                    yield (f"❌ Video editing failed: {edit_log}", "Edit failed", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

                current_input = edited_path
                yield ("✅ Video editing completed", "Edit completed successfully", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)

            # Step 2: Apply RIFE processing (if enabled)
            if rife_enabled:
                # Update settings to use the edited video as input
                rife_settings = settings.copy()
                rife_settings["input_path"] = current_input

                yield ("⚙️ Running RIFE frame interpolation...", "Starting RIFE processing...", gr.update(value="RIFE processing...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)

                # Create a queue for progress updates
                progress_queue = queue.Queue()

                def processing_thread():
                    try:
                        result = runner.run_rife(rife_settings, on_progress=lambda msg: progress_queue.put(("progress", msg)))
                        progress_queue.put(("complete", result))
                    except Exception as e:
                        progress_queue.put(("error", str(e)))

                # Start processing in background thread
                proc_thread = threading.Thread(target=processing_thread, daemon=True)
                proc_thread.start()

                # Stream progress updates
                while proc_thread.is_alive() or not progress_queue.empty():
                    try:
                        update_type, data = progress_queue.get(timeout=0.1)
                        if update_type == "progress":
                            yield (f"⚙️ RIFE Processing: {data}", f"Progress: {data}", gr.update(value=data, visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)
                        elif update_type == "complete":
                            result = data
                            processing_complete = True
                            break
                        elif update_type == "error":
                            yield ("❌ RIFE processing failed", f"Error: {data}", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                            return
                    except queue.Empty:
                        continue

                if not processing_complete:
                    yield ("❌ Processing timed out", "RIFE processing did not complete within expected time", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)
                    return

                status = "✅ RIFE complete" if result.returncode == 0 else f"⚠️ RIFE exited with code {result.returncode}"
                final_output_path = result.output_path
            else:
                # No RIFE processing, use the current input as final output
                final_output_path = current_input
                status = "✅ Processing complete" if edit_mode != "none" else "✅ File copied (no processing)"

            # Apply face restoration if enabled
            face_apply = bool(global_settings.get("face_global", False))
            if face_apply and final_output_path and Path(final_output_path).exists():
                yield ("⚙️ Applying face restoration...", "Face restoration in progress...", gr.update(value="Face restoration...", visible=True), None, gr.update(value=None), gr.update(value="", visible=False), state)
                face_strength = float(global_settings.get("face_strength", 0.5))
                restored = restore_video(final_output_path, strength=face_strength,
                                       on_progress=lambda x: progress_callback(f"Face restoration: {x}"))
                if restored:
                    final_output_path = restored
                    yield ("✅ Face restoration completed", "Face restoration done", gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)

            # Create metadata string
            processing_steps = []
            if edit_mode != "none":
                processing_steps.append(f"Edit: {edit_mode}")
            if rife_enabled:
                processing_steps.append(f"RIFE: {settings.get('fps_multiplier')}x")
            if face_apply:
                processing_steps.append(f"Face: {face_strength}")

            steps_str = ", ".join(processing_steps) if processing_steps else "None"

            meta_md = f"Input: {settings['input_path']}\nOutput: {final_output_path}\nProcessing: {steps_str}"

            # Log the run and update state with output path
            if final_output_path:
                # Track output path for pinned comparison feature
                try:
                    outp = Path(final_output_path)
                    seed_controls["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                    seed_controls["last_output_path"] = str(outp) if outp.is_file() else None
                except Exception:
                    pass
                
                run_logger.write_summary(
                    Path(final_output_path),
                    {
                        "input": settings["input_path"],
                        "output": final_output_path,
                        "returncode": result.returncode if rife_enabled else 0,
                        "args": settings,
                        "edit_mode": edit_mode,
                        "rife_enabled": rife_enabled,
                        "face_restoration": face_apply,
                    },
                )

            # Build comparison outputs (match UI expectations: 5 outputs total)
            comparison_mode = seed_controls.get("comparison_mode_val", "native")
            image_slider_update = gr.update(value=None)
            video_comparison_html_update = gr.update(value="", visible=False)
            
            if final_output_path and Path(final_output_path).exists():
                original_input = settings["input_path"]
                
                # Video comparison with custom HTML slider
                if Path(final_output_path).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                    if Path(original_input).exists():
                        from shared.video_comparison_slider import create_video_comparison_html
                        
                        video_comp_html_value = create_video_comparison_html(
                            original_video=original_input,
                            upscaled_video=final_output_path,
                            height=600,
                            slider_position=50.0
                        )
                        video_comparison_html_update = gr.update(value=video_comp_html_value, visible=True)
                
                # Image comparison with ImageSlider (for single-frame outputs or image mode)
                elif Path(final_output_path).suffix.lower() in ('.png', '.jpg', '.jpeg'):
                    if Path(original_input).exists():
                        image_slider_update = gr.update(
                            value=(original_input, final_output_path),
                            visible=True
                        )

            state["operation_status"] = "completed" if "✅" in status else "ready"
            
            # Return 7 outputs to match UI expectations: status, log, progress_indicator, output_video, image_slider, video_comparison_html, state
            yield (
                status,
                result.log if rife_enabled and 'result' in locals() else "",
                gr.update(value="", visible=False),  # progress_indicator (clear on completion)
                final_output_path if final_output_path and Path(final_output_path).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv') else None,
                image_slider_update,
                video_comparison_html_update,
                state
            )

        except Exception as e:
            error_msg = f"Critical error in RIFE processing: {str(e)}"
            state = state or {}
            state["operation_status"] = "error"
            yield ("❌ Critical error", error_msg, gr.update(value="", visible=False), None, gr.update(value=None), gr.update(value="", visible=False), state)

    def cancel():
        """Cancel current processing and compile any partial outputs if available."""
        canceled = runner.cancel()
        if not canceled:
            return gr.update(value="No active process to cancel"), ""

        # Try to salvage partial outputs (mirrors SeedVR2/GAN/FlashVSR+ behavior)
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
                partial_png, completed_png_chunks = detect_resume_state(temp_chunks_dir, "png")
                
                # Try to compile video chunks
                if completed_chunks and len(completed_chunks) > 0:
                    partial_target = collision_safe_path(temp_chunks_dir / "cancelled_rife_partial.mp4")
                    if concat_videos(completed_chunks, partial_target):
                        final_output = Path(output_dir) / f"cancelled_rife_partial_upscaled.mp4"
                        final_output = collision_safe_path(final_output)
                        shutil.copy2(partial_target, final_output)
                        compiled_output = str(final_output)
                
                # Or compile PNG chunks
                elif completed_png_chunks and len(completed_png_chunks) > 0:
                    from shared.path_utils import collision_safe_dir
                    partial_target = collision_safe_dir(temp_chunks_dir / "cancelled_rife_partial_png")
                    partial_target.mkdir(parents=True, exist_ok=True)
                    
                    for i, chunk_path in enumerate(completed_png_chunks, 1):
                        dest = partial_target / f"chunk_{i:04d}"
                        if Path(chunk_path).is_dir():
                            shutil.copytree(chunk_path, dest, dirs_exist_ok=True)
                        else:
                            shutil.copy2(chunk_path, dest)
                    
                    compiled_output = str(partial_target)
                
                if compiled_output:
                    return (
                        gr.update(value=f"⏹️ Cancelled - Partial RIFE output saved: {Path(compiled_output).name}"),
                        f"Partial results salvaged and saved to: {compiled_output}"
                    )
            except Exception as e:
                return (
                    gr.update(value=f"⏹️ Cancelled - Error salvaging partials: {str(e)}"),
                    "Processing cancelled but partial compilation failed"
                )
        
        return gr.update(value="⏹️ RIFE processing cancelled"), "Processing cancelled - no partial outputs found to salvage"

    def open_outputs_folder_rife():
        """Open outputs folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import open_outputs_folder
        return open_outputs_folder(str(output_dir))
    
    def clear_temp_folder_rife(confirm: bool):
        """Clear temp folder - delegates to shared utility (no code duplication)"""
        from shared.services.global_service import clear_temp_folder
        return clear_temp_folder(str(temp_dir), confirm)

    return {
        "defaults": defaults,
        "order": RIFE_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "cancel_action": cancel,
        "open_outputs_folder": open_outputs_folder_rife,
        "clear_temp_folder": clear_temp_folder_rife,
    }
