from pathlib import Path
from typing import Any, Dict

import gradio as gr

from ui.shared_components import comparison_help, preset_section


def build_gan_tab(
    defaults: Dict[str, Any],
    preset_manager,
    health_banner: Dict[str, str],
    seed_controls_cache: Dict[str, Any],
    callbacks: Dict[str, Any],
):
    """
    Image-based GAN tab UI builder.
    callbacks must provide:
      - order list
      - refresh_presets(model, select_name=None)
      - save_preset(name, *args)
      - load_preset(name, model, defaults, current_values)
      - safe_defaults()
      - run_action(upload, *args, preview_only=False)
      - cancel_action()
      - models(): list of available models
    """
    models = callbacks["models"]()
    values = [defaults[k] for k in callbacks["order"]]

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Image-Based GAN Upscaler")
            gr.Markdown(health_banner.get("text", ""))
            model_select = gr.Dropdown(label="Model", choices=models, value=values[0], info="Scanned from Image_Upscale_Models")
            backend = gr.Dropdown(label="Backend", choices=["realesrgan", "opencv"], value=values[2])
            scale = gr.Number(label="Scale (auto-detected from model)", value=values[1], precision=0, interactive=False, info="Automatically detected from selected model")
            input_file = gr.File(label="Upload image/video (optional)", type="filepath", file_types=["image", "video"])
            input_path = gr.Textbox(label="Input path (image, video, or frames folder)", value=values[3])
            gan_cuda = gr.Textbox(label="CUDA device(s) (e.g., 0 or 0,1)", value=values[4])
            batch_enable = gr.Checkbox(label="Enable batch (directory)", value=values[5])
            batch_input = gr.Textbox(label="Batch input folder", value=values[6])
            batch_output = gr.Textbox(label="Batch output folder override", value=values[7])
            output_format = gr.Dropdown(label="Output format", choices=["auto", "mp4", "png"], value=values[8])
            output_override = gr.Textbox(label="Output override", value=values[9])
            use_resolution = gr.Checkbox(label="Use Resolution & Scene Split settings", value=values[10])
            fps_override = gr.Number(label="FPS override (video)", value=values[11], precision=0)
            frames_per_batch = gr.Slider(label="Frames per batch (OpenCV backend)", minimum=0, maximum=240, step=1, value=values[12])
            face_restore_chk = gr.Checkbox(label="Apply Face Restoration", value=False)
            status_box = gr.Markdown("Ready.")
            log_box = gr.Textbox(label="Run Log", value="", lines=10)
            output_media = gr.Video(label="Output (video) or PNG folder path in log", interactive=False, show_download_button=True)
            comparison_note = gr.HTML("")
            image_slider = gr.ImageSlider(label="Image Comparison", interactive=False, visible=True, height=500)
            alpha_warn = gr.Markdown("⚠️ PNG inputs with alpha are preserved; MP4 output drops alpha. Choose PNG output to retain alpha.")
            with gr.Row():
                run_btn = gr.Button("Run GAN Upscale", variant="primary")
                cancel_confirm = gr.Checkbox(label="Confirm cancel", value=False)
                cancel_btn = gr.Button("Cancel", variant="stop")
            with gr.Row():
                open_outputs_btn = gr.Button("Open Outputs Folder")
                delete_confirm = gr.Checkbox(label="Confirm delete temp", value=False)
                delete_temp_btn = gr.Button("Delete Temp Folder")
        with gr.Column(scale=2):
            preset_dropdown, preset_name, save_preset_btn, load_preset_btn, preset_status, safe_defaults_btn = preset_section(
                "GAN",
                preset_manager,
                values[0],
                preset_manager.list_presets("gan", values[0]),
                preset_manager.get_last_used_name("gan", values[0]),
                safe_defaults_label="Safe Defaults (GAN)",
            )

    inputs_list = [
        model_select,
        scale,
        backend,
        input_path,
        gan_cuda,
        batch_enable,
        batch_input,
        batch_output,
        output_format,
        output_override,
        use_resolution,
        fps_override,
        frames_per_batch,
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
        inputs=[input_file] + inputs_list + [face_restore_chk],
        outputs=[status_box, log_box, output_media, comparison_note, image_slider],
    )
    cancel_btn.click(
        fn=lambda ok: callbacks["cancel_action"]() if ok else (gr.Markdown.update(value="ℹ️ Enable 'Confirm cancel' to stop."), ""),
        inputs=[cancel_confirm],
        outputs=[status_box, log_box],
    )

    open_outputs_btn.click(lambda: callbacks["open_outputs_folder"](), outputs=status_box)
    delete_temp_btn.click(lambda ok: callbacks["clear_temp_folder"](ok), inputs=[delete_confirm], outputs=status_box)

    # Update scale when model changes
    model_select.change(
        fn=lambda model: callbacks["get_model_scale"](model),
        inputs=[model_select],
        outputs=[scale]
    )

    return comparison_help()

