import os
from pathlib import Path
from typing import Any, Dict

import gradio as gr

from shared.models import scan_gan_models
from shared.health import collect_health_report
from shared.logging_utils import RunLogger
from shared.path_utils import get_default_output_dir, get_default_temp_dir
from shared.preset_manager import PresetManager
from shared.runner import Runner
from shared.gradio_compat import check_gradio_version, check_required_features
from ui.seedvr2_tab import seedvr2_tab
from ui.resolution_tab import resolution_tab
from ui.output_tab import output_tab
from ui.face_tab import face_tab
from ui.rife_tab import rife_tab
from ui.gan_tab import gan_tab
from ui.flashvsr_tab import flashvsr_tab
from ui.health_tab import health_tab

BASE_DIR = Path(__file__).parent.resolve()
PRESET_DIR = BASE_DIR / "presets"
APP_TITLE = "SECourses Ultimate Video and Image Upscaler Pro V1.0 – https://www.patreon.com/posts/134405610"


# --------------------------------------------------------------------- #
# Global setup - Honor launcher BAT file environment variables
# --------------------------------------------------------------------- #
preset_manager = PresetManager(PRESET_DIR)

# Read TEMP/TMP from launcher BAT file if set, otherwise use defaults
# This ensures user-configured paths from Windows_Run_SECourses_Upscaler_Pro.bat are respected
launcher_temp = os.environ.get("TEMP") or os.environ.get("TMP")
launcher_output = None  # BAT doesn't set OUTPUT_DIR, but we check for future compatibility

# If BAT file set a custom temp that's NOT the system temp, use it
# This detects if user modified the BAT file's TEMP/TMP settings
system_temp = os.environ.get("SystemRoot", "C:\\Windows") + "\\Temp" if os.name == "nt" else "/tmp"
if launcher_temp and launcher_temp.lower() != system_temp.lower():
    default_temp = launcher_temp
else:
    default_temp = str(BASE_DIR / "temp")

GLOBAL_DEFAULTS = {
    "output_dir": launcher_output or str(BASE_DIR / "outputs"),
    "temp_dir": default_temp,
    "telemetry": True,
    "face_global": False,
    "face_strength": 0.5,
    "mode": "subprocess",
    "mode_locked": False,  # Persisted lock state for in-app mode
}
global_settings = preset_manager.load_global_settings(GLOBAL_DEFAULTS)

temp_dir = get_default_temp_dir(BASE_DIR, global_settings)
output_dir = get_default_output_dir(BASE_DIR, global_settings)
runner = Runner(
    BASE_DIR,
    temp_dir=temp_dir,
    output_dir=output_dir,
    telemetry_enabled=global_settings.get("telemetry", True),
)
# Restore execution mode from saved settings (default to subprocess)
saved_mode = global_settings.get("mode", "subprocess")
mode_locked = global_settings.get("mode_locked", False)
try:
    runner.set_mode(saved_mode)
except Exception:
    runner.set_mode("subprocess")
    global_settings["mode"] = "subprocess"
    global_settings["mode_locked"] = False
run_logger = RunLogger(enabled=global_settings.get("telemetry", True))



# --------------------------------------------------------------------- #
# UI construction
# --------------------------------------------------------------------- #
def main():
    # Initialize health check data
    try:
        initial_report = collect_health_report(temp_dir=temp_dir, output_dir=output_dir)
        warnings = []
        
        # Check Gradio compatibility FIRST (critical for UI)
        gradio_compatible, gradio_msg, gradio_features = check_gradio_version()
        if not gradio_compatible:
            warnings.append(f"⚠️ GRADIO: {gradio_msg}")
        
        required_features, features_msg = check_required_features()
        if not required_features:
            warnings.append(f"⚠️ GRADIO FEATURES: {features_msg}")
        
        # Add mode lock warning if applicable
        if mode_locked and saved_mode == "in_app":
            warnings.append("🔒 IN-APP MODE LOCKED: You are in in-app mode. To switch back to subprocess mode, restart the application.")
        
        for key, info in initial_report.items():
            if info.get("status") not in ("ok", "skipped"):
                warnings.append(f"{key}: {info.get('detail')}")
        # Surface CPU-only ffmpeg reminder and VS Build Tools guidance up front
        warnings.append("ffmpeg runs CPU-only; ensure ffmpeg is in PATH. CUDA ffmpeg is not used.")
        vs_info = initial_report.get("vs_build_tools")
        if vs_info and vs_info.get("status") != "ok":
            warnings.append("VS Build Tools not detected; torch.compile will be disabled on Windows until installed.")
        health_text = "\n".join(warnings) if warnings else "All health checks passed."
    except Exception:
        health_text = "Health check failed to initialize. Run Health Check tab for details."

    # Modern theme with enhanced styling
    modern_theme = gr.themes.Soft(
        primary_hue="indigo",
        font=["Inter", "Arial", "sans-serif"],
        font_mono=["JetBrains Mono", "Consolas", "monospace"]
    )
    css_overrides = """
    .gr-button { padding: 12px 16px; font-size: 16px; border-radius: 8px; }
    .gr-button-lg { padding: 16px 24px; font-size: 18px; }
    .gr-markdown, .gr-textbox label, .gr-number label, .gr-dropdown label { font-size: 16px; }
    .gr-tab { font-size: 16px; font-weight: 500; }
    .gr-form { gap: 16px; }
    .health-banner { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px; border-radius: 8px; margin: 8px 0; }
    .warning-text { color: #ff6b6b; font-weight: 500; }
    .success-text { color: #51cf66; font-weight: 500; }
    .info-text { color: #74c0fc; font-weight: 500; }
    .model-status { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 8px; padding: 12px; margin: 8px 0; font-family: 'Courier New', monospace; font-size: 14px; }
    """

    # Auto-apply last-used resolution presets on startup (for all models)
    # This ensures resolution/chunking settings are loaded automatically without manual "Apply"
    def load_default_resolution_settings():
        """Load last-used resolution settings for default models into shared state"""
        from shared.models import get_seedvr2_model_names, scan_gan_models, get_flashvsr_model_names
        
        # Get primary models for each pipeline
        seedvr2_models = get_seedvr2_model_names()
        gan_models = scan_gan_models(BASE_DIR)
        flashvsr_models = get_flashvsr_model_names()
        
        # Pick first model from each as default
        default_model = seedvr2_models[0] if seedvr2_models else (gan_models[0] if gan_models else (flashvsr_models[0] if flashvsr_models else "default"))
        
        # Load last-used resolution preset for default model
        last_used_res = preset_manager.load_last_used("resolution", default_model)
        
        # Extract resolution settings (fall back to defaults if no preset)
        from shared.services.resolution_service import resolution_defaults
        res_defaults = resolution_defaults([default_model])
        
        if last_used_res:
            res_settings = preset_manager.merge_config(res_defaults, last_used_res)
        else:
            res_settings = res_defaults
        
        return res_settings

    # Load resolution settings on startup
    startup_res_settings = load_default_resolution_settings()
    
    with gr.Blocks(title=APP_TITLE, theme=modern_theme, css=css_overrides) as demo:
        # Shared state for cross-tab communication
        # AUTO-POPULATED with last-used resolution settings on startup
        shared_state = gr.State({
            "health_banner": {"text": health_text},
            "seed_controls": {
                # AUTO-LOADED from Resolution tab last-used presets
                "resolution_val": startup_res_settings.get("target_resolution", 1080),
                "max_resolution_val": startup_res_settings.get("max_target_resolution", 0),
                "current_model": None,
                "last_input_path": "",
                "last_output_dir": "",
                "resolution_cache": {},
                "png_padding_val": 5,
                "png_keep_basename_val": True,
                "skip_first_frames_val": None,
                "load_cap_val": None,
                "fps_override_val": None,
                "output_format_val": None,
                "comparison_mode_val": "native",
                "pin_reference_val": False,
                "fullscreen_val": False,
                "face_strength_val": 0.5,
                # AUTO-LOADED chunking settings from Resolution tab
                "chunk_size_sec": startup_res_settings.get("chunk_size", 0),
                "chunk_overlap_sec": startup_res_settings.get("chunk_overlap", 0.5),
                "ratio_downscale": startup_res_settings.get("ratio_downscale_then_upscale", False),
                "enable_max_target": startup_res_settings.get("enable_max_target", True),
                "auto_resolution": startup_res_settings.get("auto_resolution", True),
                "per_chunk_cleanup": startup_res_settings.get("per_chunk_cleanup", False),
                "scene_threshold": startup_res_settings.get("scene_threshold", 27.0),
                "min_scene_len": startup_res_settings.get("min_scene_len", 2.0),
            },
            "operation_status": "ready"
        })

        # Health banner at the top
        health_banner = gr.Markdown(f'<div class="health-banner">{health_text}</div>')
        gr.Markdown(f"# {APP_TITLE}")

        # Global settings tab (simple controls)
        with gr.Tab("Global Settings"):
            with gr.Row():
                output_dir_box = gr.Textbox(
                    label="Default Outputs Folder",
                    value=global_settings["output_dir"],
                    info="Where processed files will be saved"
                )
                temp_dir_box = gr.Textbox(
                    label="Temp Folder",
                    value=global_settings["temp_dir"],
                    info="Temporary files during processing"
                )
            with gr.Row():
                telemetry_toggle = gr.Checkbox(
                    label="Save run metadata (local telemetry)",
                    value=global_settings.get("telemetry", True),
                    info="Save processing metadata for troubleshooting"
                )
                face_global_toggle = gr.Checkbox(
                    label="Apply Face Restoration globally",
                    value=global_settings.get("face_global", False),
                    info="Enable face restoration for all upscaling operations"
                )
            with gr.Row():
                face_strength_slider = gr.Slider(
                    label="Global Face Restoration Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=global_settings.get("face_strength", 0.5),
                    info="Strength of face restoration when globally enabled (0.0 = subtle, 1.0 = maximum)"
                )
            save_global = gr.Button("💾 Save Global Settings", variant="primary", size="lg")
            global_status = gr.Markdown("")

            # Execution mode controls
            gr.Markdown("### ⚙️ Execution Mode")
            gr.Markdown("""
            **Subprocess Mode** (Default & RECOMMENDED): Each processing run is a separate subprocess. Ensures 100% VRAM/RAM cleanup but model reloads each time.
            
            **⚠️ In-App Mode** (EXPERIMENTAL - MODEL-SPECIFIC SUPPORT): 
            - **Status**: This mode is **PARTIALLY IMPLEMENTED** with MODEL-SPECIFIC compatibility.
            
            - **MODEL COMPATIBILITY**:
              - ✅ **GAN Models**: Can work in in-app mode (no cancellation, but functional)
              - ✅ **RIFE**: Can work in in-app mode (no cancellation, but functional)
              - ❌ **SeedVR2**: Models reload each run (NO BENEFIT over subprocess)
              - ❌ **FlashVSR+**: Models reload each run (NO BENEFIT over subprocess)
            
            - **Current Limitations (ALL MODELS)**:
              - ❌ **CANNOT CANCEL**: No way to stop processing mid-run (no subprocess to kill)
              - ⚠️ **VS BUILD TOOLS**: Must be pre-activated on Windows for torch.compile
              - ⚠️ **MEMORY LEAKS**: No subprocess isolation; VRAM may not clean up properly
              - ⚠️ **REQUIRES APP RESTART** to return to subprocess mode
            
            - **SeedVR2-Specific Issues**:
              - ❌ **NO MODEL CACHING**: CLI design forces model reload each run (no speed benefit)
              - 💡 Requires CLI refactoring for persistent model caching
            
            - **Planned Features** (Future Work):
              - Persistent model caching for SeedVR2 (requires CLI changes)
              - Cancellation support via threading interrupts
              - Intelligent model swapping with VRAM management
            
            - **Current Recommendation**: 
              - **SeedVR2/FlashVSR+**: **USE SUBPROCESS MODE** (in-app provides no benefits)
              - **GAN/RIFE**: In-app may work but subprocess still recommended for stability
            
            💡 **Recommendation**: **Always use subprocess mode** for reliability, cancellation, and proper VRAM cleanup.
            """)
            mode_radio = gr.Radio(
                choices=["subprocess", "in_app"],
                value=saved_mode,  # Restore from saved settings
                label="Processing Mode",
                info="⚠️ Changing to in-app requires confirmation and persists until app restart",
                interactive=True
            )
            mode_confirm = gr.Checkbox(
                label="⚠️ I understand that in-app mode requires app restart to revert",
                value=False,
                visible=True,
                info="Enable this checkbox to confirm mode switch to in-app (cannot be undone without restart)"
            )
            apply_mode_btn = gr.Button("🔄 Apply Mode Change", variant="secondary", size="lg")

            # Wire up global settings events
            def save_global_settings(od, td, tel, face, state):
                from shared.services.global_service import save_global_settings
                return save_global_settings(od, td, tel, face, runner, preset_manager, global_settings, run_logger, state)

            def apply_mode_selection(mode_choice, confirm, state):
                from shared.services.global_service import apply_mode_selection
                return apply_mode_selection(mode_choice, confirm, runner, preset_manager, global_settings, state)

            # Add status display for mode changes
            mode_status = gr.Markdown("")

            save_global.click(
                fn=save_global_settings,
                inputs=[output_dir_box, temp_dir_box, telemetry_toggle, face_global_toggle, face_strength_slider, shared_state],
                outputs=[global_status, shared_state],
            )

            apply_mode_btn.click(
                fn=apply_mode_selection,
                inputs=[mode_radio, mode_confirm, shared_state],
                outputs=[mode_radio, mode_confirm, mode_status, shared_state],
            )

        # Self-contained tabs following SECourses pattern
        with gr.Tab("🎬 SeedVR2 (Video/Image)"):
            seedvr2_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        with gr.Tab("📐 Resolution & Scene Split"):
            resolution_tab(
                preset_manager=preset_manager,
                shared_state=shared_state,
                base_dir=BASE_DIR
            )

        with gr.Tab("🎭 Output & Comparison"):
            output_tab(
                preset_manager=preset_manager,
                shared_state=shared_state,
                base_dir=BASE_DIR
            )

        with gr.Tab("👤 Face Restoration"):
            face_tab(
                preset_manager=preset_manager,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR
            )

        with gr.Tab("⏱️ RIFE / FPS / Edit Videos"):
            rife_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        with gr.Tab("🖼️ Image-Based (GAN)"):
            gan_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        with gr.Tab("⚡ FlashVSR+ (Real-Time Diffusion)"):
            flashvsr_tab(
                preset_manager=preset_manager,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        with gr.Tab("🏥 Health Check"):
            health_tab(
                global_settings=global_settings,
                shared_state=shared_state,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        # Update health banner on load and changes
        def update_health_banner(state):
            """Update health banner with current state"""
            health_text = state.get("health_banner", {}).get("text", "System ready")
            return gr.Markdown.update(value=f'<div class="health-banner">{health_text}</div>')

        # Update on load
        demo.load(fn=update_health_banner, inputs=shared_state, outputs=health_banner)
        
        # Update when shared state changes (for dynamic updates from tabs)
        shared_state.change(fn=update_health_banner, inputs=shared_state, outputs=health_banner)

    demo.launch()


if __name__ == "__main__":
    main()

