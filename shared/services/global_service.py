import platform
import subprocess
import os
from pathlib import Path
import gradio as gr

from shared.preset_manager import PresetManager


def save_global_settings(output_dir_val: str, temp_dir_val: str, telemetry_enabled: bool, face_global: bool, face_strength: float, runner, preset_manager: PresetManager, global_settings: dict, run_logger=None, state: dict = None):
    global_settings.update(
        {
            "output_dir": output_dir_val or global_settings.get("output_dir"),
            "temp_dir": temp_dir_val or global_settings.get("temp_dir"),
            "telemetry": telemetry_enabled,
            "face_global": bool(face_global),
            "face_strength": float(face_strength),
            "mode": runner.get_mode(),
        }
    )
    preset_manager.save_global_settings(global_settings)
    runner.temp_dir = Path(global_settings["temp_dir"])
    runner.output_dir = Path(global_settings["output_dir"])
    runner.set_telemetry(telemetry_enabled)
    if run_logger is not None:
        try:
            run_logger.enabled = telemetry_enabled
        except Exception:
            pass

    # Update state if provided
    if state:
        state["seed_controls"]["face_strength_val"] = float(face_strength)

    return gr.Markdown.update(value="✅ Global settings saved"), state or {}


def apply_mode_selection(mode_choice: str, confirm: bool, runner, preset_manager: PresetManager, global_settings: dict, state: dict = None):
    """
    Apply execution mode selection with proper confirmation and locking.
    
    Mode switching rules:
    - subprocess → in_app: Requires confirmation, locks until restart
    - in_app → subprocess: BLOCKED (requires app restart)
    - subprocess → subprocess: No-op
    - in_app → in_app: No-op
    """
    current = runner.get_mode()
    state = state or {}
    if "mode_state" not in state:
        state["mode_state"] = {"locked": False}
    mode_state = state.get("mode_state", {})

    # Load persisted lock state from global settings (survives restart)
    persisted_locked = global_settings.get("mode_locked", False)
    if persisted_locked:
        mode_state["locked"] = True
        state["mode_state"]["locked"] = True

    # Check if trying to switch FROM in-app TO subprocess (not allowed)
    if (mode_state.get("locked") or persisted_locked) and current == "in_app" and mode_choice == "subprocess":
        return (
            gr.Radio.update(value=current),  # Force radio back to in_app
            gr.Checkbox.update(value=False),  # Uncheck confirmation
            gr.Markdown.update(value=(
                "🔒 **In-app mode is active and locked**\n\n"
                "You cannot switch back to subprocess mode without restarting the application.\n\n"
                "**Why?** In-app mode keeps models loaded in VRAM. Switching back would require "
                "full memory cleanup which is only guaranteed by app restart.\n\n"
                "**To switch back:** Close the app and restart it. The default mode is subprocess."
            )),
            state
        )
    
    # Check if trying to switch TO in-app WITHOUT confirmation
    if mode_choice == "in_app" and not confirm:
        return (
            gr.Radio.update(value=current),  # Keep current mode
            gr.Checkbox.update(value=False),
            gr.Markdown.update(value=(
                "⚠️ **Confirmation required to switch to in-app mode**\n\n"
                "Please check the confirmation checkbox to proceed.\n\n"
                "**Important:** Once you switch to in-app mode, you cannot switch back without restarting the app.\n\n"
                "**In-app mode characteristics:**\n"
                "- ✅ Faster processing (models stay loaded)\n"
                "- ✅ No subprocess overhead\n"
                "- ⚠️ Higher VRAM usage (models persist)\n"
                "- ⚠️ Cannot cancel mid-run\n"
                "- ⚠️ Requires restart to switch back"
            )),
            state
        )
    
    # No change requested
    if mode_choice == current:
        return (
            gr.Radio.update(value=current),
            gr.Checkbox.update(value=False),
            gr.Markdown.update(value=f"ℹ️ Already in {current} mode. No change needed."),
            state
        )
    
    # Attempt mode switch
    try:
        runner.set_mode(mode_choice)
        actual_mode = runner.get_mode()
        
        # Verify mode was set correctly
        if actual_mode != mode_choice:
            return (
                gr.Radio.update(value=current),
                gr.Checkbox.update(value=False),
                gr.Markdown.update(value=f"❌ Mode switch failed: Expected {mode_choice}, got {actual_mode}"),
                state
            )
        
        # Save to global settings
        global_settings["mode"] = actual_mode
        
        # Persist lock state if switching to in-app
        if mode_choice == "in_app":
            state["mode_state"]["locked"] = True
            global_settings["mode_locked"] = True  # Persist to disk (survives restart)
        else:
            global_settings["mode_locked"] = False
        
        preset_manager.save_global_settings(global_settings)
            success_msg = (
                "✅ **Switched to in-app mode**\n\n"
                "Models will now persist in VRAM between runs for faster processing.\n\n"
                "**Reminder:** You must restart the app to switch back to subprocess mode.\n\n"
                "**Next steps:**\n"
                "1. Run your first upscale - model will load and stay in VRAM\n"
                "2. Subsequent runs will be faster (no reload)\n"
                "3. Monitor VRAM usage in GPU tools\n"
                "4. Use 'Clear CUDA Cache' buttons if VRAM fills up"
            )
        else:
            success_msg = f"✅ Switched to {actual_mode} mode"
        
        return (
            gr.Radio.update(value=actual_mode),
            gr.Checkbox.update(value=False),  # Uncheck after successful apply
            gr.Markdown.update(value=success_msg),
            state
        )
        
    except ValueError as e:
        # Invalid mode requested
        return (
            gr.Radio.update(value=current),
            gr.Checkbox.update(value=False),
            gr.Markdown.update(value=f"❌ Invalid mode: {str(e)}"),
            state
        )
    except Exception as exc:
        # Unexpected error
        return (
            gr.Radio.update(value=current),
            gr.Checkbox.update(value=False),
            gr.Markdown.update(value=f"❌ Mode change failed: {str(exc)}"),
            state
        )


def open_outputs_folder(path: str):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    try:
        if platform.system() == "Windows":
            os.startfile(path_obj)  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path_obj)])
        else:
            subprocess.run(["xdg-open", str(path_obj)])
        return gr.Markdown.update(value=f"Opened: {path_obj}")
    except Exception as exc:
        return gr.Markdown.update(value=f"⚠️ Could not open folder: {exc}")


def clear_temp_folder(path: str, confirm: bool = False):
    if not confirm:
        return gr.Markdown.update(value="⚠️ Enable 'Confirm delete' before clearing temp.")
    target = Path(path)
    if target.exists():
        for child in target.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            else:
                import shutil
                shutil.rmtree(child, ignore_errors=True)
    return gr.Markdown.update(value=f"Temp cleared at {target}")


