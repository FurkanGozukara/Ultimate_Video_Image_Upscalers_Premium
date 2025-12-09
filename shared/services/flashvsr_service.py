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

# Cancel event for FlashVSR+ processing
_flashvsr_cancel_event = threading.Event()


def flashvsr_defaults() -> Dict[str, Any]:
    """Get default FlashVSR+ settings aligned with CLI defaults."""
    try:
        import torch
        cuda_default = "auto" if torch.cuda.is_available() else "cpu"
    except Exception:
        cuda_default = "cpu"
    
    return {
        "input_path": "",
        "output_override": "",
        "scale": 4,
        "version": "10",
        "mode": "tiny",
        "tiled_vae": False,
        "tiled_dit": False,
        "tile_size": 256,
        "overlap": 24,
        "unload_dit": False,
        "color_fix": True,
        "seed": 0,
        "dtype": "bf16",
        "device": cuda_default,
        "fps": 30,
        "quality": 6,
        "attention": "sage",
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
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        """Save a preset."""
        if not preset_name.strip():
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name"), *list(args)

        try:
            payload = _flashvsr_dict_from_args(list(args))
            model_name = f"v{payload['version']}_{payload['mode']}"
            
            preset_manager.save_preset_safe("flashvsr", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(FLASHVSR_ORDER, list(args)))
            loaded_vals = _apply_flashvsr_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}'"), *loaded_vals
        except Exception as e:
            return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ Error: {str(e)}"), *list(args)

    def load_preset(preset_name: str, version: str, mode: str, current_values: List[Any]):
        """Load a preset."""
        try:
            model_name = f"v{version}_{mode}"
            preset = preset_manager.load_preset_safe("flashvsr", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("flashvsr", model_name, preset_name)

            current_map = dict(zip(FLASHVSR_ORDER, current_values))
            values = _apply_flashvsr_preset(preset or {}, defaults, preset_manager, current=current_map)
            return values
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            return current_values

    def safe_defaults():
        """Get safe default values."""
        return [defaults[key] for key in FLASHVSR_ORDER]

    def run_action(upload, *args, preview_only: bool = False, state=None, progress=None):
        """Main processing action with gr.Progress integration."""
        try:
            state = state or {"seed_controls": {}}
            settings_dict = _flashvsr_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            
            # Clear cancel event
            _flashvsr_cancel_event.clear()
            
            # Initialize progress
            if progress:
                progress(0, desc="Initializing FlashVSR+...")
            
            # Resolve input
            input_path = normalize_path(upload if upload else settings["input_path"])
            if not input_path or not Path(input_path).exists():
                yield (
                    "❌ Input path missing",
                    "",
                    None,
                    None,
                    gr.ImageSlider.update(visible=False),
                    gr.HTML.update(value="", visible=False),
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
                    
                    yield (
                        "⏹️ Processing cancelled",
                        "\n".join(log_buffer[-50:]) + "\n\n[Cancelled by user]",
                        None,
                        None,
                        gr.ImageSlider.update(visible=False),
                        gr.HTML.update(value="", visible=False),
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
                        gr.ImageSlider.update(visible=False),
                        gr.HTML.update(value="Processing...", visible=False),
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
                    gr.ImageSlider.update(visible=False),
                    gr.HTML.update(value="", visible=False),
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
                    gr.ImageSlider.update(visible=False),
                    gr.HTML.update(value="", visible=False),
                    state
                )
                return
            
            # Update progress to 100%
            if progress:
                progress(1.0, desc="FlashVSR+ complete!")
            
            # Create comparison
            html_comp, img_slider = create_unified_comparison(
                input_path=input_path,
                output_path=result.output_path,
                mode="slider" if result.output_path and result.output_path.endswith(".mp4") else "native"
            )
            
            # Log run
            run_logger.write_summary(
                Path(result.output_path) if result.output_path else output_dir,
                {
                    "input": input_path,
                    "output": result.output_path,
                    "returncode": result.returncode,
                    "args": settings,
                    "pipeline": "flashvsr",
                }
            )
            
            status = "✅ FlashVSR+ upscaling complete" if result.returncode == 0 else f"⚠️ Exited with code {result.returncode}"
            
            yield (
                status,
                result.log,
                result.output_path if result.output_path and result.output_path.endswith(".mp4") else None,
                result.output_path if result.output_path and not result.output_path.endswith(".mp4") else None,
                img_slider if img_slider else gr.ImageSlider.update(visible=False),
                html_comp if html_comp else gr.HTML.update(value="", visible=False),
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
                gr.ImageSlider.update(visible=False),
                gr.HTML.update(value="", visible=False),
                state or {}
            )

    def cancel_action():
        """Cancel FlashVSR+ processing"""
        _flashvsr_cancel_event.set()
        return gr.Markdown.update(value="⏹️ Cancellation requested - FlashVSR+ will stop at next checkpoint"), "Cancelling..."

    def open_outputs_folder():
        """Open outputs folder"""
        import platform
        import subprocess
        
        try:
            out_dir = str(output_dir)
            if platform.system() == "Windows":
                subprocess.Popen(["explorer", out_dir])
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", out_dir])
            else:
                subprocess.Popen(["xdg-open", out_dir])
            return gr.Markdown.update(value=f"✅ Opened: {out_dir}")
        except Exception as e:
            return gr.Markdown.update(value=f"❌ Error: {str(e)}")

    return {
        "defaults": defaults,
        "order": FLASHVSR_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "cancel_action": cancel_action,
        "open_outputs_folder": open_outputs_folder,
    }

