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
        # Initialize shared state
        initial_state = initialize_state()
        shared_state = gr.State(initial_state)

        # Extract health banner text for display
        health_text = initial_state["health_banner"]["text"]
        banner = gr.Markdown(health_text)
        gr.Markdown(f"# {APP_TITLE}")

        with gr.Tab("Global"):
            with gr.Row():
                output_dir_box = gr.Textbox(label="Default Outputs Folder", value=global_settings["output_dir"])
                temp_dir_box = gr.Textbox(label="Temp Folder", value=global_settings["temp_dir"])
                telemetry_toggle = gr.Checkbox(label="Save run metadata (local)", value=global_settings.get("telemetry", True))
                face_global_toggle = gr.Checkbox(label="Apply Face Restoration globally", value=global_settings.get("face_global", False))
            save_global = gr.Button("Save Global Settings", variant="primary")
            global_status = gr.Markdown("")
            gr.Markdown("Outputs folder is controlled here (GUI). Launcher/batch defaults are ignored once saved.")
            save_global.click(
                fn=lambda od, td, tel, face, state: global_service.save_global_settings(od, td, tel, face, runner, preset_manager, global_settings, run_logger, state),
                inputs=[output_dir_box, temp_dir_box, telemetry_toggle, face_global_toggle, shared_state],
                outputs=[global_status, shared_state],
            )
            gr.Markdown("#### Execution Mode")
            mode_radio = gr.Radio(
                choices=["subprocess", "in_app"],
                value=runner.get_mode(),
                label="Mode",
                info="Subprocess (default) fully cleans memory; In-app keeps models loaded but may leak. Restart required to return to subprocess after switching.",
            )
            mode_confirm = gr.Checkbox(label="I understand switching to in-app locks until restart", value=False)
            apply_mode_btn = gr.Button("Apply Mode", variant="secondary")
            mode_status = gr.Markdown("")
            apply_mode_btn.click(
                fn=lambda mode_choice, confirm, state: global_service.apply_mode_selection(
                    mode_choice, confirm, runner, preset_manager, global_settings, state
                ),
                inputs=[mode_radio, mode_confirm, shared_state],
                outputs=[mode_status, mode_radio, shared_state],
            )

        with gr.Tab("SeedVR2"):
            seed_srv = seed_service.build_seedvr2_callbacks(
                preset_manager,
                runner,
                run_logger,
                global_settings,
                shared_state,
                output_dir,
                temp_dir,
            )
            seed_callbacks = {
                "order": seed_service.SEEDVR2_ORDER,
                "get_models": get_seedvr2_model_names,
                "refresh_presets": seed_srv["refresh_presets"],
                "save_preset": seed_srv["save_preset"],
                "load_preset": lambda preset, model, _defaults, vals: seed_srv["load_preset"](preset, model, list(vals)),
                "safe_defaults": seed_srv["safe_defaults"],
                "run_action": seed_srv["run_action"],
                "cancel_action": seed_srv["cancel_action"],
                "open_outputs_folder": lambda state: global_service.open_outputs_folder(
                    state["seed_controls"].get("last_output_dir") or global_settings["output_dir"]
                ),
                "clear_temp_folder": lambda confirm=False: global_service.clear_temp_folder(global_settings["temp_dir"], confirm),
            }
        seed_controls = build_seedvr2_tab_ui(
            seed_defaults,
            preset_manager,
            global_settings,
            shared_state,
            seed_srv["comparison_html_slider"],
            seed_callbacks,
        )

        # Add progress timer functionality
        progress_indicator = seed_controls.get("progress_indicator")

        def update_progress_indicator(state):
            operation_status = state.get("operation_status", "ready")
            if operation_status == "running":
                return gr.Markdown.update(value="🔄 Operation in progress...", visible=True)
            elif operation_status == "completed":
                return gr.Markdown.update(value="✅ Operation completed", visible=True)
            else:
                return gr.Markdown.update(visible=False)

        if progress_indicator:
            progress_timer.tick(
                fn=update_progress_indicator,
                inputs=shared_state,
                outputs=progress_indicator
            )

        with gr.Tab("Resolution & Scene Split"):
            res_srv = res_service.build_resolution_callbacks(preset_manager, shared_state, combined_models)
            res_last_name = preset_manager.get_last_used_name("resolution", res_srv["defaults"].get("model"))
            res_last = preset_manager.load_last_used("resolution", res_srv["defaults"].get("model"))
            if res_last_name and res_last is None:
                _append_warning(f"Last used Resolution preset '{res_last_name}' not found; loaded defaults.")
            res_defaults = preset_manager.merge_config(res_srv["defaults"], res_last or {})
        resolution_callbacks = {
            "order": res_service.RESOLUTION_ORDER,
            "models": lambda: combined_models,
            "refresh_presets": res_srv["refresh_presets"],
            "save_preset": res_srv["save_preset"],
            "load_preset": lambda preset, model, _defaults, vals: res_srv["load_preset"](preset, model, list(vals)),
            "safe_defaults": res_srv["safe_defaults"],
            "apply_to_seed": res_srv["apply_to_seed"],
            "chunk_estimate": res_srv["chunk_estimate"],
            "estimate_from_input": res_srv["estimate_from_input"],
            "cache_resolution": res_srv["cache_resolution"],
            "cache_resolution_flags": res_srv["cache_resolution_flags"],
            "seed_controls": seed_controls,
            "shared_state": shared_state,
        }
        build_resolution_tab_ui(
            res_defaults,
            preset_manager,
            shared_state,
            resolution_callbacks,
        )

        with gr.Tab("Output & Comparison"):
            out_srv = out_service.build_output_callbacks(preset_manager, shared_state, combined_models)
            out_last_name = preset_manager.get_last_used_name("output", out_srv["defaults"].get("model"))
            out_last = preset_manager.load_last_used("output", out_srv["defaults"].get("model"))
            if out_last_name and out_last is None:
                _append_warning(f"Last used Output preset '{out_last_name}' not found; loaded defaults.")
            out_defaults = preset_manager.merge_config(out_srv["defaults"], out_last or {})
            output_callbacks = {
                "order": out_service.OUTPUT_ORDER,
                "models": lambda: combined_models,
                "refresh_presets": out_srv["refresh_presets"],
                "save_preset": out_srv["save_preset"],
                "load_preset": lambda preset, model, _defaults, vals: out_srv["load_preset"](preset, model, list(vals)),
                "safe_defaults": out_srv["safe_defaults"],
                "seed_controls": seed_controls,
                "cache_output": out_srv["cache_output"],
                "cache_fps": out_srv["cache_fps"],
                "cache_comparison": out_srv["cache_comparison"],
                "cache_pin": out_srv["cache_pin"],
                "cache_fullscreen": out_srv["cache_fullscreen"],
                "cache_png_padding": out_srv["cache_png_padding"],
                "cache_png_basename": out_srv["cache_png_basename"],
                "cache_skip": out_srv["cache_skip"],
                "cache_cap": out_srv["cache_cap"],
                "shared_state": shared_state,
            }
            build_output_tab_ui(
                out_defaults,
                preset_manager,
                shared_state,
                output_callbacks,
            )

        with gr.Tab("Face Restoration"):
            face_srv = face_service.build_face_callbacks(preset_manager, global_settings, combined_models, shared_state)
            face_last_name = preset_manager.get_last_used_name("face", face_srv["defaults"].get("model"))
            face_last = preset_manager.load_last_used("face", face_srv["defaults"].get("model"))
            if face_last_name and face_last is None:
                _append_warning(f"Last used Face preset '{face_last_name}' not found; loaded defaults.")
            f_defaults = preset_manager.merge_config(face_srv["defaults"], face_last or {})
            face_callbacks = {
                "order": face_service.FACE_ORDER,
                "models": lambda: combined_models,
                "refresh_presets": face_srv["refresh_presets"],
                "save_preset": face_srv["save_preset"],
                "load_preset": lambda preset, model, _defaults, vals: face_srv["load_preset"](preset, model, list(vals)),
                "safe_defaults": face_srv["safe_defaults"],
                "set_face_global": face_srv["set_face_global"],
                "shared_state": shared_state,
            }
            build_face_tab_ui(
                f_defaults,
                preset_manager,
                global_settings,
                face_callbacks,
            )

        with gr.Tab("RIFE / FPS / Edit Videos"):
            rife_srv = rife_service.build_rife_callbacks(preset_manager, runner, run_logger, global_settings, output_dir, temp_dir, shared_state)
            rife_last_name = preset_manager.get_last_used_name("rife", rife_srv["defaults"].get("model"))
            rife_last = preset_manager.load_last_used("rife", rife_srv["defaults"].get("model"))
            if rife_last_name and rife_last is None:
                _append_warning(f"Last used RIFE preset '{rife_last_name}' not found; loaded defaults.")
            r_defaults = preset_manager.merge_config(rife_srv["defaults"], rife_last or {})
            rife_callbacks = {
                "order": rife_service.RIFE_ORDER,
                "refresh_presets": rife_srv["refresh_presets"],
                "save_preset": rife_srv["save_preset"],
                "load_preset": lambda preset, model, _defaults, vals: rife_srv["load_preset"](preset, model, list(vals)),
                "safe_defaults": rife_srv["safe_defaults"],
                "run_action": rife_srv["run_action"],
                "cancel_action": seed_srv["cancel_action"],
                "shared_state": shared_state,
            }
            build_rife_tab_ui(
                r_defaults,
                preset_manager,
                shared_state,
                rife_callbacks,
            )

        with gr.Tab("Image-Based (GAN)"):
            gan_srv = gan_service.build_gan_callbacks(preset_manager, run_logger, global_settings, shared_state, BASE_DIR, temp_dir, output_dir)
            gan_last_name = preset_manager.get_last_used_name("gan", gan_srv["defaults"].get("model"))
            gan_last = preset_manager.load_last_used("gan", gan_srv["defaults"].get("model"))
            if gan_last_name and gan_last is None:
                _append_warning(f"Last used GAN preset '{gan_last_name}' not found; loaded defaults.")
            g_defaults = preset_manager.merge_config(gan_srv["defaults"], gan_last or {})
            gan_callbacks = {
                "order": gan_service.GAN_ORDER,
                "models": gan_srv["model_scanner"],
                "refresh_presets": gan_srv["refresh_presets"],
                "save_preset": gan_srv["save_preset"],
                "load_preset": lambda preset, model, _defaults, vals: gan_srv["load_preset"](preset, model, list(vals)),
                "safe_defaults": gan_srv["safe_defaults"],
                "run_action": gan_srv["run_action"],
                "cancel_action": gan_srv["cancel_action"],
                "open_outputs_folder": lambda state: global_service.open_outputs_folder(
                    state["seed_controls"].get("last_output_dir") or global_settings["output_dir"]
                ),
                "clear_temp_folder": lambda confirm=False: global_service.clear_temp_folder(global_settings["temp_dir"], confirm),
                "shared_state": shared_state,
            }
            build_gan_tab_ui(
                g_defaults,
                preset_manager,
                shared_state,
                gan_callbacks,
            )

        health_srv = health_service.build_health_callbacks(global_settings, shared_state)
        with gr.Tab("Health Check"):
            health_btn = gr.Button("Run Health Check")
            health_report = gr.Markdown("Run to verify ffmpeg, CUDA, VS Build Tools, and disk/temp/output writability.")
            health_btn.click(fn=lambda state: health_srv["health_check_action"](state)[0], inputs=shared_state, outputs=[health_report, shared_state])

        def load_health_check(state):
            report_text, health_text, updated_state = health_srv["health_check_action"](state)
            return report_text, updated_state

        def load_health_banner(state):
            report_text, health_text, updated_state = health_srv["health_check_action"](state)
            return health_text, updated_state

        demo.load(fn=load_health_check, inputs=shared_state, outputs=[health_report, shared_state])
        demo.load(fn=load_health_banner, inputs=shared_state, outputs=[banner, shared_state])

    demo.launch()


if __name__ == "__main__":
    main()
