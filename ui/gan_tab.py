"""
Image-Based (GAN) Tab - Self-contained modular implementation
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.gan_service import (
    build_gan_callbacks, GAN_ORDER
)


def gan_tab(
    preset_manager,
    runner,
    run_logger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path
):
    """
    Self-contained Image-Based (GAN) tab.
    Handles GAN-based upscaling with fixed scale factors.
    """

    # Build service callbacks
    service = build_gan_callbacks(
        preset_manager, run_logger, global_settings, shared_state,
        base_dir, temp_dir, output_dir
    )

    # Get defaults and last used
    defaults = service["defaults"]
    last_used_name = preset_manager.get_last_used_name("gan", defaults.get("model", "default"))
    last_used = preset_manager.load_last_used("gan", defaults.get("model", "default"))

    # Handle last used preset loading with better error reporting
    if last_used_name:
        if last_used is None:
            def update_warning(state):
                existing = state["health_banner"]["text"]
                warning = f"⚠️ Last used GAN preset '{last_used_name}' not found or corrupted; loaded defaults."
                if existing:
                    state["health_banner"]["text"] = existing + "\n" + warning
                else:
                    state["health_banner"]["text"] = warning
                return state
            shared_state.value = update_warning(shared_state.value)
        else:
            def update_success(state):
                existing = state["health_banner"]["text"]
                success_msg = f"✅ Loaded last used GAN preset: '{last_used_name}'"
                if existing:
                    state["health_banner"]["text"] = existing + "\n" + success_msg
                else:
                    state["health_banner"]["text"] = success_msg
                return state
            shared_state.value = update_success(shared_state.value)

    merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    values = [merged_defaults[k] for k in GAN_ORDER]

    # Layout
    gr.Markdown("### 🖼️ Image-Based (GAN) Upscaling")
    gr.Markdown("*High-quality image upscaling using GAN models with fixed scale factors (2x, 4x, etc.)*")

    # Input section
    with gr.Accordion("📁 Input Configuration", open=True):
        input_file = gr.File(
            label="Upload Image or Video",
            type="filepath",
            file_types=["image", "video"],
            info="Upload single image or video (videos processed frame-by-frame)"
        )
        input_path = gr.Textbox(
            label="Image/Video Path",
            value=values[0],
            placeholder="C:/path/to/image.jpg or C:/path/to/video.mp4",
            info="Direct path to file or folder of images"
        )
        input_cache_msg = gr.Markdown("", visible=False)

    # Batch processing
    with gr.Accordion("📦 Batch Processing", open=False):
        batch_enable = gr.Checkbox(
            label="Enable Batch Processing",
            value=values[1],
            info="Process multiple files from directory"
        )
        batch_input = gr.Textbox(
            label="Batch Input Folder",
            value=values[2],
            placeholder="Folder containing images/videos",
            info="Directory with files to process"
        )
        batch_output = gr.Textbox(
            label="Batch Output Folder",
            value=values[3],
            placeholder="Output directory for batch results"
        )

    with gr.Tabs():
        # Model Selection
        with gr.TabItem("🤖 Model Selection"):
            gr.Markdown("#### GAN Model Configuration")

            with gr.Group():
                gan_model = gr.Dropdown(
                    label="GAN Model",
                    choices=service["model_scanner"](),
                    value=values[4],
                    info="Available pre-trained GAN models"
                )

                model_info = gr.Markdown("Select a model to see details...")

                def update_model_info(model_name):
                    if not model_name:
                        return "Select a model to see details..."

                    try:
                        from shared.gan_runner import get_gan_model_metadata
                        base_dir = base_dir  # Use the passed base_dir
                        metadata = get_gan_model_metadata(model_name, base_dir)

                        info_lines = [f"**{model_name}**"]
                        info_lines.append(f"- **Scale**: {metadata.scale}x")
                        info_lines.append(f"- **Architecture**: {metadata.architecture}")
                        if metadata.description and metadata.description != f"{model_name}":
                            info_lines.append(f"- **Description**: {metadata.description}")
                        if metadata.author and metadata.author != "unknown":
                            info_lines.append(f"- **Author**: {metadata.author}")
                        if metadata.tags:
                            info_lines.append(f"- **Tags**: {', '.join(metadata.tags)}")

                        return "\n".join(info_lines)
                    except Exception as e:
                        return f"**{model_name}**\n\nUnable to load metadata: {str(e)}"

                gan_model.change(
                    fn=update_model_info,
                    inputs=gan_model,
                    outputs=model_info
                )

        # Processing Settings
        with gr.TabItem("⚙️ Processing Settings"):
            gr.Markdown("#### Upscaling Parameters")

            with gr.Group():
                target_resolution = gr.Slider(
                    label="Target Resolution (longest side)",
                    minimum=512, maximum=4096, step=64,
                    value=values[5],
                    info="Desired output resolution (will be adjusted based on model scale)"
                )

                downscale_first = gr.Checkbox(
                    label="Downscale First if Needed",
                    value=values[6],
                    info="For fixed-scale models, downscale input to reach target via upscale chain"
                )

                auto_calculate_input = gr.Checkbox(
                    label="Auto-Calculate Input Resolution",
                    value=values[7],
                    info="Automatically determine best input size for target resolution"
                )

                tile_size = gr.Number(
                    label="Tile Size",
                    value=values[8],
                    precision=0,
                    info="Process image in tiles to reduce memory usage (0 = no tiling)"
                )

                overlap = gr.Number(
                    label="Tile Overlap",
                    value=values[9],
                    precision=0,
                    info="Overlap between tiles for seamless results"
                )

                batch_size = gr.Slider(
                    label="Batch Size (Frames per Iteration)",
                    minimum=1, maximum=16, step=1,
                    value=values[15],
                    info="Number of frames to process simultaneously (affects VRAM usage)"
                )

        # Quality & Performance
        with gr.TabItem("🎨 Quality & Performance"):
            gr.Markdown("#### Output Quality Settings")

            with gr.Group():
                denoising_strength = gr.Slider(
                    label="Denoising Strength",
                    minimum=0.0, maximum=1.0, step=0.05,
                    value=values[10],
                    info="Noise reduction applied during upscaling"
                )

                sharpening = gr.Slider(
                    label="Output Sharpening",
                    minimum=0.0, maximum=2.0, step=0.1,
                    value=values[11],
                    info="Sharpening applied to final result"
                )

                color_correction = gr.Checkbox(
                    label="Color Correction",
                    value=values[12],
                    info="Maintain color accuracy during upscaling"
                )

                gpu_acceleration = gr.Checkbox(
                    label="GPU Acceleration",
                    value=values[13],
                    info="Use GPU for faster processing"
                )

                gpu_device = gr.Textbox(
                    label="GPU Device",
                    value=values[14],
                    placeholder="0 or 0,1",
                    info="Specific GPU device(s) to use"
                )

        # Output Settings
        with gr.TabItem("📤 Output Settings"):
            gr.Markdown("#### File Output Configuration")

            with gr.Group():
                output_format_gan = gr.Dropdown(
                    label="Output Format",
                    choices=["auto", "png", "jpg", "webp"],
                    value=values[15],
                    info="'auto' matches input format"
                )

                output_quality_gan = gr.Slider(
                    label="Output Quality",
                    minimum=70, maximum=100, step=5,
                    value=values[16],
                    info="Quality for lossy formats (JPG/WebP)"
                )

                save_metadata = gr.Checkbox(
                    label="Save Processing Metadata",
                    value=values[17],
                    info="Embed processing information in output files"
                )

                create_subfolders = gr.Checkbox(
                    label="Create Subfolders by Model",
                    value=values[18],
                    info="Organize outputs in model-named subdirectories"
                )

    # Output section
    with gr.Accordion("🎯 Output & Results", open=True):
        gr.Markdown("#### Processing Results")

        status_box = gr.Markdown(value="Ready for processing.")
        progress_indicator = gr.Markdown(value="", visible=False)
        log_box = gr.Textbox(
            label="📋 Processing Log",
            value="",
            lines=10,
            show_copy_button=True
        )

        output_image = gr.Image(
            label="🖼️ Upscaled Image",
            interactive=False,
            show_download_button=True
        )
        output_video = gr.Video(
            label="🎬 Upscaled Video",
            interactive=False,
            show_download_button=True
        )

        # Enhanced comparison
        image_slider = gr.ImageSlider(
            label="🔍 Before/After Comparison",
            interactive=False,
            height=500,
            slider_position=50,
            max_height=600,
            buttons=["download", "fullscreen"]
        )

    # Last processed info
    last_processed = gr.Markdown("Batch processing results will appear here.")

    # Action buttons
    with gr.Row():
        upscale_btn = gr.Button(
            "🚀 Start Upscaling",
            variant="primary",
            size="lg"
        )
        cancel_confirm = gr.Checkbox(
            label="Confirm cancel",
            value=False,
            visible=False
        )
        cancel_btn = gr.Button(
            "⏹️ Cancel",
            variant="stop",
            visible=False
        )
        preview_btn = gr.Button(
            "👁️ Preview First Frame",
            size="lg"
        )

    # Utility buttons
    with gr.Row():
        open_outputs_btn = gr.Button("📂 Open Outputs Folder")
        clear_temp_btn = gr.Button("🗑️ Clear Temp Files")

    # Preset management
    with gr.Accordion("💾 Preset Management", open=True):
        preset_dropdown = gr.Dropdown(
            label="GAN Presets",
            choices=preset_manager.list_presets("gan", "default"),
            value=last_used_name or "",
        )

        with gr.Row():
            preset_name = gr.Textbox(
                label="Preset Name",
                placeholder="my_gan_preset"
            )
            save_preset_btn = gr.Button("💾 Save Preset", variant="secondary")

        with gr.Row():
            load_preset_btn = gr.Button("📂 Load Preset")
            safe_defaults_btn = gr.Button("🔄 Safe Defaults")

        preset_status = gr.Markdown("")

    # Model information
    with gr.Accordion("ℹ️ About GAN Upscaling", open=False):
        gr.Markdown("""
        #### GAN-Based Image Upscaling

        **Fixed Scale Factors:**
        - Models are trained for specific upscale ratios (2x, 4x, etc.)
        - To reach arbitrary resolutions, use "Downscale First" option
        - Example: For 4x model to reach 3x effective scale:
          1. Downscale input by 4/3 = 1.333x
          2. Apply 4x GAN model
          3. Result is 3x effective upscaling

        **Quality vs Speed:**
        - Newer/more complex models = higher quality but slower
        - Tiling reduces memory usage but may affect quality at boundaries
        - GPU acceleration is highly recommended

        **Use Cases:**
        - High-quality image enlargement
        - Video frame upscaling (frame-by-frame)
        - Batch processing of image collections
        """)

    # Collect all inputs
    inputs_list = [
        input_path, batch_enable, batch_input, batch_output, gan_model,
        target_resolution, downscale_first, auto_calculate_input, tile_size, overlap,
        denoising_strength, sharpening, color_correction, gpu_acceleration, gpu_device,
        output_format_gan, output_quality_gan, save_metadata, create_subfolders
    ]

    # Wire up event handlers

    # Input handling
    def cache_input(val, state):
        state["seed_controls"]["last_input_path"] = val if val else ""
        return val or "", gr.Markdown.update(value="✅ Input cached for processing.", visible=True), state

    input_file.upload(
        fn=lambda val, state: cache_input(val, state),
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state]
    )

    input_path.change(
        fn=lambda val, state: (gr.Markdown.update(value="✅ Input path updated.", visible=True), state),
        inputs=[input_path, shared_state],
        outputs=[input_cache_msg, shared_state]
    )

    # Main processing
    upscale_btn.click(
        fn=lambda *args: service["run_action"](*args[:-1], preview_only=False, state=args[-1]),
        inputs=inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_image, output_video,
            last_processed, image_slider, shared_state
        ]
    )

    preview_btn.click(
        fn=lambda *args: service["run_action"](*args[:-1], preview_only=True, state=args[-1]),
        inputs=inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_image, output_video,
            last_processed, image_slider, shared_state
        ]
    )

    cancel_btn.click(
        fn=lambda ok, state: (service["cancel_action"](), state) if ok else (gr.Markdown.update(value="⚠️ Enable 'Confirm cancel' to stop."), "", state),
        inputs=[cancel_confirm, shared_state],
        outputs=[status_box, log_box, shared_state]
    )

    # Utility functions
    open_outputs_btn.click(
        fn=lambda state: (service["open_outputs_folder"](state), state),
        inputs=shared_state,
        outputs=[status_box, shared_state]
    )

    clear_temp_btn.click(
        fn=lambda: service["clear_temp_folder"](False),
        outputs=status_box
    )

    # Preset management
    save_preset_btn.click(
        fn=lambda name, *vals: service["save_preset"](name, "default", list(vals)),
        inputs=[preset_name] + inputs_list,
        outputs=[preset_dropdown, preset_status]
    )

    load_preset_btn.click(
        fn=lambda preset, *vals: service["load_preset"](preset, "default", list(vals)),
        inputs=[preset_dropdown] + inputs_list,
        outputs=inputs_list + [preset_status]
    )

    safe_defaults_btn.click(
        fn=service["safe_defaults"],
        outputs=inputs_list
    )
