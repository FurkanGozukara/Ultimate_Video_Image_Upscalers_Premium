import platform
import subprocess
import os
from pathlib import Path
import gradio as gr

from shared.preset_manager import PresetManager


def save_global_settings(output_dir_val: str, temp_dir_val: str, telemetry_enabled: bool, face_global: bool, face_strength: float, 
                         models_dir_val: str, hf_home_val: str, transformers_cache_val: str,
                         runner, preset_manager: PresetManager, global_settings: dict, run_logger=None, state: dict = None):
    """
    Save global settings including model cache paths.
    
    FIXED: Now persists model cache paths (MODELS_DIR, HF_HOME, TRANSFORMERS_CACHE) so user can
    override launcher BAT file settings from the UI. These paths control where AI models are downloaded.
    
    ⚠️ IMPORTANT: Changing model cache paths requires app restart to take effect (environment variables
    are set at app startup). Updated paths will be used on next launch.
    """
    # Preserve pinned reference from state if present
    seed_controls = state.get("seed_controls", {}) if state else {}
    pinned_ref = seed_controls.get("pinned_reference_path") or global_settings.get("pinned_reference_path")
    
    # FIXED: Persist model cache paths for next app launch
    # These override launcher BAT file settings if user configures them in UI
    # NOTE: Do NOT overwrite "mode" here - it's managed separately by apply_mode_selection
    # to preserve user's restart preference (e.g., if they selected subprocess while in in_app mode)
    global_settings.update(
        {
            "output_dir": output_dir_val or global_settings.get("output_dir"),
            "temp_dir": temp_dir_val or global_settings.get("temp_dir"),
            "telemetry": telemetry_enabled,
            "face_global": bool(face_global),
            "face_strength": float(face_strength),
            # "mode" is intentionally NOT updated here - managed by apply_mode_selection only
            "pinned_reference_path": pinned_ref,  # Persist pinned reference
            # FIXED: Save model cache paths (editable in UI)
            "models_dir": models_dir_val or global_settings.get("models_dir"),
            "hf_home": hf_home_val or global_settings.get("hf_home"),
            "transformers_cache": transformers_cache_val or global_settings.get("transformers_cache"),
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
    
    # Build status message with restart reminder if model cache paths changed
    status_lines = ["✅ Global settings saved"]
    
    # Check if model cache paths were changed
    original_models = global_settings.get("_original_models_dir")
    original_hf = global_settings.get("_original_hf_home")
    original_trans = global_settings.get("_original_transformers_cache")
    
    models_changed = (
        (models_dir_val and models_dir_val != original_models) or
        (hf_home_val and hf_home_val != original_hf) or
        (transformers_cache_val and transformers_cache_val != original_trans)
    )
    
    if models_changed:
        status_lines.append("")
        status_lines.append("⚠️ **Model cache paths changed - RESTART REQUIRED**")
        status_lines.append("")
        status_lines.append("**Action needed:**")
        status_lines.append("1. Close this application")
        status_lines.append("2. Restart to apply new model cache locations")
        status_lines.append("")
        status_lines.append("💡 Models will download to new location on next run.")
        status_lines.append("To keep existing models, manually copy them from old to new location.")

    return gr.update(value="\n".join(status_lines)), state or {}


def apply_mode_selection(mode_choice: str, confirm: bool, runner, preset_manager: PresetManager, global_settings: dict, state: dict = None):
    """
    Apply execution mode selection with proper confirmation and restart warnings.
    
    Mode switching rules:
    - subprocess → in_app: Applies immediately with confirmation, requires restart to revert
    - in_app → subprocess: Saves preference, requires app restart to take effect
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

    # Check if trying to switch FROM in-app TO subprocess (needs restart)
    if current == "in_app" and mode_choice == "subprocess":
        # Save the preference so it takes effect on restart
        global_settings["mode"] = "subprocess"
        global_settings["mode_locked"] = False  # Unlock on restart
        preset_manager.save_global_settings(global_settings)
        
        return (
            gr.update(value=mode_choice),  # Allow selection change in UI
            gr.update(value=False),  # Uncheck confirmation
            gr.update(value=(
                "✅ **Subprocess mode selected - RESTART REQUIRED**\n\n"
                "⚠️ **Your preference has been saved, but the change will NOT take effect until you restart the application.**\n\n"
                "**Current session:** Still running in in-app mode (models loaded in VRAM)\n"
                "**Next session:** Will start in subprocess mode (default, recommended)\n\n"
                "**Why restart is required:**\n"
                "In-app mode keeps models loaded in VRAM. Switching back requires full memory cleanup which is only guaranteed by app restart.\n\n"
                "**Action needed:**\n"
                "1. Close this application\n"
                "2. Restart to activate subprocess mode\n\n"
                "💡 Subprocess mode will be active on next launch."
            )),
            state
        )
    
    # Check if trying to switch TO in-app WITHOUT confirmation
    if mode_choice == "in_app" and not confirm:
        return (
            gr.update(value=current),  # Keep current mode
            gr.update(value=False),
            gr.update(value=(
                "⚠️ **Confirmation required to switch to in-app mode**\n\n"
                "Please check the confirmation checkbox to proceed.\n\n"
                "**Important:** In-app mode applies immediately but requires restart to switch back.\n\n"
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
            gr.update(value=current),
            gr.update(value=False),
            gr.update(value=f"ℹ️ Already in {current} mode. No change needed."),
            state
        )
    
    # Attempt mode switch (only for subprocess → in_app, which applies immediately)
    try:
        runner.set_mode(mode_choice)
        actual_mode = runner.get_mode()
        
        # Verify mode was set correctly
        if actual_mode != mode_choice:
            return (
                gr.update(value=current),
                gr.update(value=False),
                gr.update(value=f"❌ Mode switch failed: Expected {mode_choice}, got {actual_mode}"),
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
        
        if mode_choice == "in_app":
            success_msg = (
                "✅ **Switched to in-app mode - ACTIVE NOW**\n\n"
                "⚠️ **The mode change is now active. To switch back to subprocess mode, you must restart the application.**\n\n"
                "**⚠️ MODEL-SPECIFIC BEHAVIOR:**\n"
                "- **GAN Models:** ✅ Models may persist in VRAM (faster reruns possible)\n"
                "- **RIFE:** ✅ Models may persist in VRAM (faster reruns possible)\n"
                "- **SeedVR2:** ❌ NO benefit - models reload each run (use subprocess instead)\n"
                "- **FlashVSR+:** ❌ NO benefit - models reload each run (use subprocess instead)\n\n"
                "**⚠️ LIMITATIONS (ALL MODELS):**\n"
                "- ❌ **Cannot cancel** processing mid-run (no subprocess to kill)\n"
                "- ⚠️ **Memory leaks** possible without subprocess isolation\n"
                "- ⚠️ **Requires restart** to switch back to subprocess mode\n\n"
                "**💡 RECOMMENDATION:** Use subprocess mode for reliability and cancellation support.\n\n"
                "**If proceeding with in-app mode:**\n"
                "1. Only use for GAN/RIFE models (not SeedVR2/FlashVSR+)\n"
                "2. Monitor VRAM usage in GPU tools\n"
                "3. Use 'Clear CUDA Cache' if VRAM fills up\n"
                "4. Restart app if you encounter issues or want to switch back"
            )
        else:
            success_msg = f"✅ Switched to {actual_mode} mode - ACTIVE NOW"
        
        return (
            gr.update(value=actual_mode),
            gr.update(value=False),  # Uncheck after successful apply
            gr.update(value=success_msg),
            state
        )
        
    except ValueError as e:
        # Invalid mode requested
        return (
            gr.update(value=current),
            gr.update(value=False),
            gr.update(value=f"❌ Invalid mode: {str(e)}"),
            state
        )
    except Exception as exc:
        # Unexpected error
        return (
            gr.update(value=current),
            gr.update(value=False),
            gr.update(value=f"❌ Mode change failed: {str(exc)}"),
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
        return gr.update(value=f"Opened: {path_obj}")
    except Exception as exc:
        return gr.update(value=f"⚠️ Could not open folder: {exc}")


def clear_temp_folder(path: str, confirm: bool = False):
    if not confirm:
        return gr.update(value="⚠️ Enable 'Confirm delete' before clearing temp.")
    target = Path(path)
    if target.exists():
        for child in target.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            else:
                import shutil
                shutil.rmtree(child, ignore_errors=True)
    return gr.update(value=f"Temp cleared at {target}")


