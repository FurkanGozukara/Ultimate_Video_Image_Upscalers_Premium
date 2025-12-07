from pathlib import Path
from typing import Any, Dict, List

import gradio as gr

from ui.shared_components import comparison_help, preset_section


def build_seedvr2_tab(
    defaults: Dict[str, Any],
    preset_manager,
    global_settings: Dict[str, Any],
    seed_controls_cache: Dict[str, Any],
    health_banner: Dict[str, str],
    comparison_html_slider,
    callbacks: Dict[str, Any],
):
    """
    SeedVR2 tab UI builder (modularized).
    callbacks must provide:
      - order: List[str]
      - get_models: Callable[[], List[str]]
      - refresh_presets(model, select_name=None)
      - save_preset(name, model, *args)
      - load_preset(name, model, defaults, current_values)
      - safe_defaults()
      - run_action(uploaded_file, face_restore_run, *args, preview_only=False)
      - cancel_action()
      - open_outputs_folder()
      - clear_temp_folder()
      - auto_res_on_input(path)
    """
    models = callbacks["get_models"]()
    values = [defaults[k] for k in callbacks["order"]]
    seed_controls_cache.setdefault("current_model", values[4] if len(values) > 4 else (models[0] if models else ""))

    # GPU hint
    try:
        import torch  # type: ignore

        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        gpu_hint = f"Detected CUDA GPUs: {cuda_count}" if cuda_count else "CUDA not available"
    except Exception:
        gpu_hint = "CUDA detection failed"

    def _cache_path_value(val: Any):
        seed_controls_cache["last_input_path"] = val if val else ""
        return gr.Markdown.update(value="Input cached for resolution/chunk estimates.")

    def _cache_upload(val: Any):
        seed_controls_cache["last_input_path"] = val if val else ""
        return val or "", gr.Markdown.update(value="Input cached for resolution/chunk estimates.")

    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Input / Controls")
            input_file = gr.File(label="Upload video or image (optional)", type="filepath", file_types=["video", "image"])
            input_path = gr.Textbox(label="Input Video or Frames Folder Path", value=values[0], placeholder="C:/path/to/video.mp4 or /path/to/frames")
            input_cache_msg = gr.Markdown("")
            auto_res_msg = gr.Markdown("")
            batch_enable = gr.Checkbox(label="Enable Batch Processing (use directory input)", value=values[5])
            batch_input = gr.Textbox(label="Batch Input Folder", value=values[6], placeholder="Folder containing videos or frames")
            batch_output = gr.Textbox(label="Batch Output Folder Override", value=values[7], placeholder="Optional override for batch outputs")
            gr.Markdown("#### Scene Split (PySceneDetect)")
            chunk_enable = gr.Checkbox(label="Enable scene-based chunking", value=values[8])
            scene_threshold = gr.Slider(label="Content Threshold", minimum=5, maximum=50, step=1, value=values[9])
            scene_min_len = gr.Slider(label="Min Scene Length (sec)", minimum=1, maximum=20, step=1, value=values[10])
            output_override = gr.Textbox(label="Output Override (single run)", value=values[1], placeholder="Leave empty for auto naming")
            output_format = gr.Dropdown(label="Output Format", choices=["auto", "mp4", "png"], value=values[2])
            model_dir = gr.Textbox(label="Model Directory (optional)", value=values[3])
            dit_model = gr.Dropdown(label="SeedVR2 Model", choices=models, value=values[4])
            model_cache_msg = gr.Markdown("")

            with gr.Row():
                resolution = gr.Slider(label="Target Resolution (short side)", minimum=256, maximum=4096, step=16, value=values[11])
                max_resolution = gr.Slider(label="Max Resolution (0 = no cap)", minimum=0, maximum=8192, step=16, value=values[12])

            batch_size = gr.Slider(label="Batch Size (4n+1)", minimum=1, maximum=201, step=4, value=values[13])
            uniform_batch_size = gr.Checkbox(label="Uniform Batch Size", value=values[14])
            seed = gr.Number(label="Seed", value=values[15], precision=0)

            with gr.Row():
                skip_first_frames = gr.Number(label="Skip First Frames", value=values[16], precision=0)
                load_cap = gr.Number(label="Load Cap (0 = all)", value=values[17], precision=0)
                prepend_frames = gr.Number(label="Prepend Frames", value=values[18], precision=0)
                temporal_overlap = gr.Number(label="Temporal Overlap", value=values[19], precision=0)

            color_correction = gr.Dropdown(
                label="Color Correction",
                choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                value=values[20],
            )
            input_noise_scale = gr.Slider(label="Input Noise Scale", minimum=0.0, maximum=1.0, step=0.01, value=values[21])
            latent_noise_scale = gr.Slider(label="Latent Noise Scale", minimum=0.0, maximum=1.0, step=0.01, value=values[22])

            gr.Markdown("#### Device & Offload")
            cuda_device = gr.Textbox(label="CUDA Devices (e.g., 0 or 0,1,2)", value=values[23], info=gpu_hint)
            dit_offload_device = gr.Textbox(label="DiT Offload Device", value=values[24], placeholder="none / cpu / GPU id")
            vae_offload_device = gr.Textbox(label="VAE Offload Device", value=values[25], placeholder="none / cpu / GPU id")
            tensor_offload_device = gr.Textbox(label="Tensor Offload Device", value=values[26], placeholder="cpu / none / GPU id")

            gr.Markdown("#### BlockSwap")
            blocks_to_swap = gr.Slider(label="Blocks to Swap", minimum=0, maximum=36, step=1, value=values[27])
            swap_io_components = gr.Checkbox(label="Swap I/O Components", value=values[28])

            gr.Markdown("#### VAE Tiling")
            vae_encode_tiled = gr.Checkbox(label="VAE Encode Tiled", value=values[29])
            vae_encode_tile_size = gr.Number(label="Encode Tile Size", value=values[30], precision=0)
            vae_encode_tile_overlap = gr.Number(label="Encode Tile Overlap", value=values[31], precision=0)
            vae_decode_tiled = gr.Checkbox(label="VAE Decode Tiled", value=values[32])
            vae_decode_tile_size = gr.Number(label="Decode Tile Size", value=values[33], precision=0)
            vae_decode_tile_overlap = gr.Number(label="Decode Tile Overlap", value=values[34], precision=0)
            tile_debug = gr.Dropdown(label="Tile Debug", choices=["false", "encode", "decode"], value=values[35])

            gr.Markdown("#### Performance & Compile")
            attention_mode = gr.Dropdown(label="Attention Backend", choices=["sdpa", "flash_attn"], value=values[36])
            compile_dit = gr.Checkbox(label="Compile DiT", value=values[37])
            compile_vae = gr.Checkbox(label="Compile VAE", value=values[38])
            compile_backend = gr.Dropdown(label="Compile Backend", choices=["inductor", "cudagraphs"], value=values[39])
            compile_mode = gr.Dropdown(
                label="Compile Mode",
                choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                value=values[40],
            )
            compile_fullgraph = gr.Checkbox(label="Compile Fullgraph", value=values[41])
            compile_dynamic = gr.Checkbox(label="Compile Dynamic Shapes", value=values[42])
            compile_dynamo_cache_size_limit = gr.Number(label="Compile Dynamo Cache Size Limit", value=values[43], precision=0)
            compile_dynamo_recompile_limit = gr.Number(label="Compile Dynamo Recompile Limit", value=values[44], precision=0)
            cache_dit = gr.Checkbox(label="Cache DiT (single GPU only)", value=values[45])
            cache_vae = gr.Checkbox(label="Cache VAE (single GPU only)", value=values[46])
            debug = gr.Checkbox(label="Debug Logging", value=values[47])
        with gr.Column(scale=2):
            gr.Markdown("### Output / Actions")
            gr.Markdown(health_banner.get("text", ""))
            status_box = gr.Markdown(value="Ready.")
            log_box = gr.Textbox(label="Run Log", value="", lines=16)
            output_video = gr.Video(label="Upscaled Video", interactive=False, show_download_button=True)
            output_image = gr.Image(label="Upscaled Image / Preview", interactive=False, show_download_button=True)
            image_slider = gr.ImageSlider(label="Image Comparison", interactive=False, visible=True, height=500)
            chunk_info = gr.Markdown("Last processed chunk will appear here.")
            alpha_warn = gr.Markdown("⚠️ PNG inputs with alpha are preserved; MP4 output drops alpha. Choose PNG output to retain alpha.")
            comparison_note = gr.HTML("")
            face_restore_chk = gr.Checkbox(label="Apply Face Restoration after upscale", value=global_settings.get("face_global", False))

            with gr.Row():
                upscale_btn = gr.Button("Upscale (subprocess)", variant="primary")
                cancel_confirm = gr.Checkbox(label="Confirm cancel", value=False)
                cancel_btn = gr.Button("Cancel", variant="stop")
                preview_btn = gr.Button("First-frame Preview")

            with gr.Row():
                open_outputs_btn = gr.Button("Open Outputs Folder")
                delete_confirm = gr.Checkbox(label="Confirm delete temp", value=False)
                delete_temp_btn = gr.Button("Delete Temp Folder")

            # Preset block
            preset_dropdown, preset_name, save_preset_btn, load_preset_btn, preset_status, safe_defaults_btn = preset_section(
                "SeedVR2",
                preset_manager,
                values[4],
                preset_manager.list_presets("seedvr2", defaults["dit_model"]),
                preset_manager.get_last_used_name("seedvr2", defaults["dit_model"]),
                safe_defaults_label="Safe Defaults (SeedVR2)",
            )

            gr.Markdown("#### Mode (info)")
            gr.Markdown(
                "Subprocess mode is active by default. Use the Global tab to switch to In-app mode (keeps models loaded, higher memory). Restart to return to subprocess."
            )
            gr.Markdown("Comparison: native slider will be used when available; HTML fallback when not.")

    inputs_list = [
        input_path,
        output_override,
        output_format,
        model_dir,
        dit_model,
        batch_enable,
        batch_input,
        batch_output,
        chunk_enable,
        scene_threshold,
        scene_min_len,
        resolution,
        max_resolution,
        batch_size,
        uniform_batch_size,
        seed,
        skip_first_frames,
        load_cap,
        prepend_frames,
        temporal_overlap,
        color_correction,
        input_noise_scale,
        latent_noise_scale,
        cuda_device,
        dit_offload_device,
        vae_offload_device,
        tensor_offload_device,
        blocks_to_swap,
        swap_io_components,
        vae_encode_tiled,
        vae_encode_tile_size,
        vae_encode_tile_overlap,
        vae_decode_tiled,
        vae_decode_tile_size,
        vae_decode_tile_overlap,
        tile_debug,
        attention_mode,
        compile_dit,
        compile_vae,
        compile_backend,
        compile_mode,
        compile_fullgraph,
        compile_dynamic,
        compile_dynamo_cache_size_limit,
        compile_dynamo_recompile_limit,
        cache_dit,
        cache_vae,
        debug,
    ]

    upscale_btn.click(
        fn=callbacks["run_action"],
        inputs=[input_file, face_restore_chk] + inputs_list,
        outputs=[status_box, log_box, output_video, output_image, chunk_info, comparison_note, image_slider],
    )
    preview_btn.click(
        fn=lambda *args: callbacks["run_action"](*args, preview_only=True),
        inputs=[input_file, face_restore_chk] + inputs_list,
        outputs=[status_box, log_box, output_video, output_image, chunk_info, comparison_note, image_slider],
    )
    cancel_btn.click(
        fn=lambda ok: callbacks["cancel_action"]() if ok else (gr.Markdown.update(value="ℹ️ Enable 'Confirm cancel' to stop."), ""),
        inputs=[cancel_confirm],
        outputs=[status_box, log_box],
    )
    open_outputs_btn.click(lambda: callbacks["open_outputs_folder"](), outputs=status_box)
    delete_temp_btn.click(lambda ok: callbacks["clear_temp_folder"](ok), inputs=[delete_confirm], outputs=status_box)

    # Presets wiring
    save_preset_btn.click(
        fn=callbacks["save_preset"],
        inputs=[preset_name, dit_model] + inputs_list,
        outputs=[preset_dropdown, preset_status] + inputs_list,
    )
    load_preset_btn.click(
        fn=lambda preset, model, *vals: callbacks["load_preset"](preset, model, defaults, list(vals)),
        inputs=[preset_dropdown, dit_model] + inputs_list,
        outputs=inputs_list,
    )
    safe_defaults_btn.click(fn=callbacks["safe_defaults"], outputs=inputs_list)

    # Sync upload to textbox
    input_file.upload(_cache_upload, inputs=input_file, outputs=[input_path, input_cache_msg])
    input_file.upload(
        fn=lambda fp: callbacks["auto_res_on_input"](fp if fp else ""),
        inputs=input_file,
        outputs=[resolution, max_resolution, auto_res_msg],
    )
    input_path.change(_cache_path_value, inputs=input_path, outputs=input_cache_msg)
    input_path.change(
        fn=lambda p: callbacks["auto_res_on_input"](p),
        inputs=input_path,
        outputs=[resolution, max_resolution, auto_res_msg],
    )

    def _set_model_cache(m):
        seed_controls_cache["current_model"] = m
        return gr.Markdown.update(value=f"Model cached for resolution/preset: {m}")

    dit_model.change(_set_model_cache, inputs=dit_model, outputs=model_cache_msg)

    # Comparison note
    comparison_note.update(comparison_html_slider())

    return {
        "resolution": resolution,
        "max_resolution": max_resolution,
        "output_format": output_format,
    }

