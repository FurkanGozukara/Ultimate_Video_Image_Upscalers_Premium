import os
from pathlib import Path
from typing import Any, Dict

import gradio as gr

from shared.health import collect_health_report
from shared.logging_utils import RunLogger
from shared.path_utils import get_default_output_dir, get_default_temp_dir
from shared.preset_manager import PresetManager
from shared.runner import Runner
from ui.seedvr2_tab import seedvr2_tab
from ui.resolution_tab import resolution_tab
from ui.output_tab import output_tab
from ui.face_tab import face_tab
from ui.rife_tab import rife_tab
from ui.gan_tab import gan_tab
from ui.health_tab import health_tab

BASE_DIR = Path(__file__).parent.resolve()
PRESET_DIR = BASE_DIR / "presets"
APP_TITLE = "SECourses Ultimate Video and Image Upscaler Pro V1.0 – https://www.patreon.com/posts/134405610"


def _get_gan_model_names(base_dir: Path) -> list:
    models_dir = base_dir / "Image_Upscale_Models"
    if not models_dir.exists():
        return []
    choices = []
    for f in models_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".pth", ".safetensors"):
            choices.append(f.name)
    return sorted(choices)


# --------------------------------------------------------------------- #
# Global setup
# --------------------------------------------------------------------- #
preset_manager = PresetManager(PRESET_DIR)

GLOBAL_DEFAULTS = {
    "output_dir": str(BASE_DIR / "outputs"),
    "temp_dir": os.environ.get("TEMP") or str(BASE_DIR / "temp"),
    "telemetry": True,
    "face_global": False,
    "face_strength": 0.5,
    "mode": "subprocess",
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
runner.set_mode(global_settings.get("mode", "subprocess"))
run_logger = RunLogger(enabled=global_settings.get("telemetry", True))



# --------------------------------------------------------------------- #
# UI construction
# --------------------------------------------------------------------- #
def main():
    # Initialize health check data
    try:
        initial_report = collect_health_report(temp_dir=temp_dir, output_dir=output_dir)
        warnings = []
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

    with gr.Blocks(title=APP_TITLE, theme=modern_theme, css=css_overrides) as demo:
        # Shared state for cross-tab communication
        shared_state = gr.State({
            "health_banner": {"text": health_text},
            "seed_controls": {
                "resolution_val": None,
                "max_resolution_val": None,
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
                "chunk_size_sec": 0,
                "chunk_overlap_sec": 0,
                "ratio_downscale": False,
                "enable_max_target": True,
                "auto_resolution": True,
                "per_chunk_cleanup": False,
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
            save_global = gr.Button("💾 Save Global Settings", variant="primary", size="lg")
            global_status = gr.Markdown("")

            # Execution mode controls
            gr.Markdown("### ⚙️ Execution Mode")
            mode_radio = gr.Radio(
                choices=["subprocess", "in_app"],
                value=runner.get_mode(),
                label="Processing Mode",
                info="Subprocess: Clean memory, slower load. In-app: Faster but may leak memory."
            )
            mode_confirm = gr.Checkbox(
                label="⚠️ I understand switching to in-app locks until restart",
                value=False,
                visible=runner.get_mode() == "subprocess"
            )
            apply_mode_btn = gr.Button("🔄 Apply Mode Change", variant="secondary")

            # Wire up global settings events
            def save_global_settings(od, td, tel, face, state):
                from shared.services.global_service import save_global_settings
                return save_global_settings(od, td, tel, face, runner, preset_manager, global_settings, run_logger, state)

            def apply_mode_selection(mode_choice, confirm, state):
                from shared.services.global_service import apply_mode_selection
                return apply_mode_selection(mode_choice, confirm, runner, preset_manager, global_settings, state)

            save_global.click(
                fn=save_global_settings,
                inputs=[output_dir_box, temp_dir_box, telemetry_toggle, face_global_toggle, shared_state],
                outputs=[global_status, shared_state],
            )

            apply_mode_btn.click(
                fn=apply_mode_selection,
                inputs=[mode_radio, mode_confirm, shared_state],
                outputs=[mode_radio, mode_confirm, shared_state],
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
                shared_state=shared_state
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

        with gr.Tab("🏥 Health Check"):
            health_tab(
                global_settings=global_settings,
                shared_state=shared_state,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        # Update health banner on load
        def update_health_banner(state):
            return gr.Markdown.update(value=f'<div class="health-banner">{state["health_banner"]["text"]}</div>')

        demo.load(fn=update_health_banner, inputs=shared_state, outputs=health_banner)

    demo.launch()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
