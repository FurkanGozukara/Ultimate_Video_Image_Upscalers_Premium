from typing import Any, Dict

import gradio as gr

from ui.shared_components import preset_section


def build_rife_tab(
    defaults: Dict[str, Any],
    preset_manager,
    health_banner: Dict[str, str],
    callbacks: Dict[str, Any],
):
    """
    RIFE / FPS tab UI builder.
    callbacks must provide:
      - order list
      - refresh_presets(model, select_name=None)
      - save_preset(name, *args)
      - load_preset(name, model, defaults, current_values)
      - safe_defaults()
      - run_action(uploaded_file, img_folder, *args)
      - cancel_action()
    """
    values = [defaults[k] for k in callbacks["order"]]

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### RIFE / FPS / Edit Videos")
            gr.Markdown(health_banner.get("text", ""))
            model_select = gr.Dropdown(label="Model", choices=["rife"], value=values[0])
            gpu_ids = gr.Textbox(label="CUDA device(s) (e.g., 0 or 0,1)", value="", placeholder="Leave empty for auto/CPU")
            fps_multiplier = gr.Slider(label="FPS Multiplier", minimum=0.25, maximum=4.0, step=0.05, value=values[1])
            scale = gr.Slider(label="Scale", minimum=0.25, maximum=4.0, step=0.05, value=values[2])
            uhd_half = gr.Checkbox(label="UHD shortcut (0.5x)", value=values[3])
            png_output = gr.Checkbox(label="Save PNG frames", value=values[4])
            no_audio = gr.Checkbox(label="No audio", value=values[5])
            show_ffmpeg = gr.Checkbox(label="Show ffmpeg output", value=values[6])
            montage = gr.Checkbox(label="Montage", value=values[7])
            img_mode = gr.Checkbox(label="Image sequence mode (--img)", value=values[8])
            output_override = gr.Textbox(label="Output override", value=values[9])
            model_dir = gr.Textbox(label="Model directory", value=values[10])
            warning = gr.Markdown("Skip flag is deprecated; cancel on Windows uses 'c' key in console. Legacy --skip is not exposed.")
            input_video = gr.File(label="Input Video (or leave empty for img folder)", type="filepath", file_types=["video"])
            input_img_folder = gr.Textbox(label="Image Frames Folder (for --img)", value="")
            skip_first_frames = gr.Number(label="Skip first frames", value=0, precision=0)
            load_cap = gr.Number(label="Load cap (0 = all)", value=0, precision=0)
            fps_override = gr.Number(label="FPS override (0=keep)", value=0, precision=0)
            output_format = gr.Dropdown(label="Output format", choices=["auto", "mp4", "png"], value="auto")
            face_restore_chk = gr.Checkbox(label="Apply Face Restoration after RIFE", value=False)
            status_box = gr.Markdown("Ready.")
            log_box = gr.Textbox(label="Run Log", value="", lines=12)
            output_video = gr.Video(label="Output Video", interactive=False)
            output_meta = gr.Markdown("Metadata will appear after run.")
            with gr.Row():
                run_btn = gr.Button("Run RIFE", variant="primary")
                cancel_confirm = gr.Checkbox(label="Confirm cancel", value=False)
                cancel_btn = gr.Button("Cancel", variant="stop")
        with gr.Column(scale=2):
            preset_dropdown, preset_name, save_preset_btn, load_preset_btn, preset_status, safe_defaults_btn = preset_section(
                "RIFE",
                preset_manager,
                values[0],
                preset_manager.list_presets("rife", values[0]),
                preset_manager.get_last_used_name("rife", values[0]),
                safe_defaults_label="Safe Defaults (RIFE)",
            )

    inputs_list = [
        model_select,
        fps_multiplier,
        scale,
        uhd_half,
        png_output,
        no_audio,
        show_ffmpeg,
        montage,
        img_mode,
        output_override,
        model_dir,
        skip_first_frames,
        load_cap,
        fps_override,
        output_format,
        gpu_ids,
    ]

    save_preset_btn.click(
        fn=callbacks["save_preset"],
        inputs=[preset_name] + inputs_list,
        outputs=[preset_dropdown, preset_status] + inputs_list,
    )
    load_preset_btn.click(
        fn=lambda preset, model, *vals: callbacks["load_preset"](preset, model, defaults, list(vals)),
        inputs=[preset_dropdown, model_select] + inputs_list,
        outputs=inputs_list,
    )
    safe_defaults_btn.click(fn=callbacks["safe_defaults"], outputs=inputs_list)

    run_btn.click(
        fn=callbacks["run_action"],
        inputs=[input_video, input_img_folder] + inputs_list + [face_restore_chk],
        outputs=[status_box, log_box, output_video, output_meta],
    )
    cancel_btn.click(
        fn=lambda ok: callbacks["cancel_action"]() if ok else (gr.Markdown.update(value="ℹ️ Enable 'Confirm cancel' to stop."), ""),
        inputs=[cancel_confirm],
        outputs=[status_box, log_box],
    )

