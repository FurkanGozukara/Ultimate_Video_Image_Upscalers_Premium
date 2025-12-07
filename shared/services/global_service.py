import platform
import subprocess
import os
from pathlib import Path
import gradio as gr

from shared.preset_manager import PresetManager


def save_global_settings(output_dir_val: str, temp_dir_val: str, telemetry_enabled: bool, face_global: bool, runner, preset_manager: PresetManager, global_settings: dict, run_logger=None):
    global_settings.update(
        {
            "output_dir": output_dir_val or global_settings.get("output_dir"),
            "temp_dir": temp_dir_val or global_settings.get("temp_dir"),
            "telemetry": telemetry_enabled,
            "face_global": bool(face_global),
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
    return gr.Markdown.update(value="✅ Global settings saved")


def apply_mode_selection(mode_choice: str, confirm: bool, runner, preset_manager: PresetManager, global_settings: dict, mode_state: dict):
    current = runner.get_mode()
    if mode_state.get("locked") and current == "in_app" and mode_choice == "subprocess":
        return (
            gr.Markdown.update(value="🔒 In-app mode active; restart to return to subprocess mode."),
            gr.Radio.update(value=current),
        )
    if mode_choice == "in_app" and not confirm:
        return (
            gr.Markdown.update(value="⚠️ Confirm to switch to in-app mode (requires restart to go back)."),
            gr.Radio.update(value=current),
        )
    try:
        runner.set_mode(mode_choice)
    except Exception as exc:
        return gr.Markdown.update(value=f"❌ Mode change failed: {exc}"), gr.Radio.update(value=current)
    global_settings["mode"] = runner.get_mode()
    preset_manager.save_global_settings(global_settings)
    if mode_choice == "in_app":
        mode_state["locked"] = True
    return gr.Markdown.update(value=f"✅ Mode set to {runner.get_mode()}"), gr.Radio.update(value=runner.get_mode())


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


def clear_temp_folder(path: str):
    target = Path(path)
    if target.exists():
        for child in target.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            else:
                import shutil
                shutil.rmtree(child, ignore_errors=True)
    return gr.Markdown.update(value=f"Temp cleared at {target}")


