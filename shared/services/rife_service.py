"""
RIFE Service Module - Clean Implementation
Handles RIFE/FPS/Edit Videos processing logic, presets, and callbacks
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from shared.preset_manager import PresetManager
from shared.runner import Runner
from shared.path_utils import normalize_path, ffmpeg_set_fps, get_media_dimensions
from shared.face_restore import restore_video
from shared.logging_utils import RunLogger


# Defaults and ordering --------------------------------------------------------
def rife_defaults() -> Dict[str, Any]:
    """Get default RIFE settings."""
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
        "model": "rife",
        "fps_multiplier": 2.0,
        "fps_override": 0.0,
        "scale": 1.0,
        "uhd_mode": False,
        "fp16_mode": False,
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
    }


RIFE_ORDER: List[str] = [
    "input_path",
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
]


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
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save a preset."""
        if not preset_name.strip():
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _rife_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("rife", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(RIFE_ORDER, list(args)))
            loaded_vals = _apply_rife_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """Load a preset."""
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("rife", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("rife", model_name, preset_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(RIFE_ORDER, current_values))
            values = _apply_rife_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
            return values
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            return current_values

    def safe_defaults():
        """Get safe default values."""
        return [defaults[k] for k in RIFE_ORDER]

    def run_action(uploaded_file, img_folder, *args, state=None):
        """Main RIFE processing action."""
        try:
            state = state or {"seed_controls": {}, "operation_status": "ready"}
            state["operation_status"] = "running"
            seed_controls = state.get("seed_controls", {})
            
            settings_dict = _rife_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}

            input_path = normalize_path(uploaded_file if uploaded_file else img_folder)
            if not input_path or not Path(input_path).exists():
                yield ("❌ Input missing or not found", "", None, "No metadata")
                return

            # Validate input type based on mode
            if settings.get("img_mode"):
                # In --img mode, require a frames folder or images
                if Path(input_path).is_file() and Path(input_path).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                    yield ("⚠️ --img mode expects frames folder or images, not a video file.", "", None, "No metadata")
                    return
            else:
                # In video mode, require a video file
                if Path(input_path).is_dir():
                    yield ("⚠️ Video mode expects a video file. Enable --img for frame folders.", "", None, "No metadata")
                    return

            settings["input_path"] = input_path
            settings["output_override"] = settings.get("output_override") or None

            # Validate CUDA devices
            cuda_warning = _validate_cuda_devices(settings.get("cuda_device", ""))
            if cuda_warning:
                yield (f"⚠️ {cuda_warning}", "", None, "No metadata")
                return

            # Check ffmpeg availability
            if not _ffmpeg_available():
                yield ("❌ ffmpeg not found in PATH. Install ffmpeg and retry.", "", None, "No metadata")
                return

            # Apply cached values from Resolution & Scene Split tab
            if seed_controls.get("resolution_val") is not None:
                # For RIFE, resolution affects downscaling before processing
                pass

            # Apply output format from Comparison tab if set
            cached_fmt = seed_controls.get("output_format_val")
            if settings.get("output_format") in (None, "auto") and cached_fmt:
                settings["output_format"] = cached_fmt

            # Run RIFE processing
            result = runner.run_rife(settings, on_progress=lambda x: None)

            status = "✅ RIFE complete" if result.returncode == 0 else f"⚠️ RIFE exited with code {result.returncode}"
            output_path = result.output_path

            # Apply face restoration if enabled
            face_apply = bool(global_settings.get("face_global", False))
            if face_apply and output_path and Path(output_path).exists():
                face_strength = float(global_settings.get("face_strength", 0.5))
                restored = restore_video(output_path, strength=face_strength, on_progress=lambda x: None)
                if restored:
                    output_path = restored

            # Create metadata string
            meta_md = f"Input: {input_path}\nOutput: {output_path}\nFPS Multiplier: {settings.get('fps_multiplier')}"

            # Log the run
            if output_path:
                run_logger.write_summary(
                    Path(output_path),
                    {
                        "input": input_path,
                        "output": output_path,
                        "returncode": result.returncode,
                        "args": settings,
                    },
                )

            state["operation_status"] = "completed" if "✅" in status else "ready"
            yield (status, result.log or "", output_path, meta_md)

        except Exception as e:
            error_msg = f"Critical error in RIFE processing: {str(e)}"
            state = state or {}
            state["operation_status"] = "error"
            yield ("❌ Critical error", error_msg, None, "Error occurred")

    def cancel():
        """Cancel current processing."""
        canceled = runner.cancel()
        if canceled:
            return gr.Markdown.update(value="⏹️ Cancel requested"), ""
        return gr.Markdown.update(value="No active process"), ""

    def open_outputs_folder(state: Dict[str, Any]):
        """Open the outputs folder in file explorer."""
        try:
            import platform
            
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

    return {
        "defaults": defaults,
        "order": RIFE_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "cancel_action": cancel,
        "open_outputs_folder": open_outputs_folder,
        "clear_temp_folder": clear_temp_folder,
    }
