"""
Image-Based (GAN) Tab - Self-contained modular implementation
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.gan_service import (
    build_gan_callbacks, GAN_ORDER
)
from shared.video_comparison_slider import create_video_comparison_html
from shared.ui_validators import validate_resolution


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
        preset_manager, runner, run_logger, global_settings, shared_state,
        base_dir, temp_dir, output_dir
    )

    # Get defaults and last used
    # FIXED: Use startup cache and check ACTUAL current model, not just "default"
    defaults = service["defaults"]
    default_model = defaults.get("model", "default")
    
    # Try to load from startup cache first (already loaded in secourses_app.py)
    gan_cache = shared_state.value.get("seed_controls", {}).get("gan_cache", {})
    cached_for_model = gan_cache.get(default_model)
    
    if cached_for_model:
        # Use pre-loaded settings from startup
        merged_defaults = cached_for_model
        last_used_name = preset_manager.get_last_used_name("gan", default_model)
        
        def update_success(state):
            existing = state["health_banner"]["text"]
            success_msg = f"✅ GAN: Restored last-used preset for {default_model}"
            state["health_banner"]["text"] = existing + "\n" + success_msg if existing else success_msg
            return state
        shared_state.value = update_success(shared_state.value)
    else:
        # Fallback: Load on-demand if cache missed
        last_used_name = preset_manager.get_last_used_name("gan", default_model)
        last_used = preset_manager.load_last_used("gan", default_model)
        
        if last_used_name and last_used is None:
            def update_warning(state):
                existing = state["health_banner"]["text"]
                warning = f"⚠️ Last used GAN preset '{last_used_name}' for model '{default_model}' not found; loaded defaults."
                state["health_banner"]["text"] = existing + "\n" + warning if existing else warning
                return state
            shared_state.value = update_warning(shared_state.value)
        
        merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    
    values = [merged_defaults[k] for k in GAN_ORDER]

    # GPU availability check (like SeedVR2 tab)
    import platform
    cuda_available = False
    cuda_count = 0
    gpu_hint = "CUDA detection in progress..."
    
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        
        if cuda_available and cuda_count > 0:
            gpu_hint = f"✅ Detected {cuda_count} CUDA GPU(s) - GPU acceleration available"
        else:
            gpu_hint = "⚠️ CUDA not available - GPU acceleration disabled. Processing will use CPU (significantly slower)"
    except Exception as e:
        gpu_hint = f"❌ CUDA detection failed: {str(e)}"
        cuda_available = False

    # Layout: Two-column design (left=controls, right=output)
    gr.Markdown("### 🖼️ Image-Based (GAN) Upscaling")
    gr.Markdown("*High-quality image upscaling using GAN models with fixed scale factors (2x, 4x, etc.)*")
    
    # Import shared layout helpers
    from ui.shared_layouts import create_gpu_warning_banner
    
    # Show GPU warning if not available
    create_gpu_warning_banner(cuda_available, gpu_hint, "GAN")

    # Two-column layout
    with gr.Row():
        # ===== LEFT COLUMN: Input & Controls =====
        with gr.Column(scale=3):
            gr.Markdown("### 📥 Input / Controls")
            
            # Input section
            with gr.Accordion("📁 Input Configuration", open=True):
                input_file = gr.File(
                    label="Upload Image or Video",
                    type="filepath",
                    file_types=["image", "video"]
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

            # Model and processing settings
            with gr.Tabs():
                # Model Selection
                with gr.TabItem("🤖 Model Selection"):
                    gr.Markdown("#### GAN Model Configuration")

                    with gr.Group():
                        gan_model = gr.Dropdown(
                            label="GAN Model",
                            choices=service["model_scanner"](),
                            value=values[4],
                            info="Pre-trained GAN models with fixed scale factors (2x, 4x, etc.). Models auto-detected from Image_Upscale_Models folder."
                        )

                        model_info = gr.Markdown("Select a model to see details...")

                        def update_model_info(model_name):
                            if not model_name:
                                return "Select a model to see details..."

                            try:
                                from shared.gan_runner import get_gan_model_metadata
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
                        target_res_warning = gr.Markdown("", visible=False)

                        downscale_first = gr.Checkbox(
                            label="Downscale First if Needed",
                            value=values[6],
                            info="GAN models have fixed scales (2x/4x). Enable to downscale input first, then upscale to reach arbitrary target resolutions. E.g., 4x model → 3x effective by downscaling 133% first."
                        )

                        auto_calculate_input = gr.Checkbox(
                            label="Auto-Calculate Input Resolution",
                            value=values[7],
                            info="Automatically calculate optimal input resolution based on target output and model scale. Recommended ON for best results with Resolution & Scene Split tab."
                        )

                        use_resolution_tab = gr.Checkbox(
                            label="🔗 Use Resolution & Scene Split Tab Settings",
                            value=values[8],
                            info="Apply target resolution, max resolution, and downscale-then-upscale settings from Resolution tab. Enables universal resolution control across all models. Recommended ON."
                        )

                        tile_size = gr.Number(
                            label="Tile Size",
                            value=values[9],
                            precision=0,
                            info="Process image in tiles to reduce VRAM usage. 0 = process whole image. Try 512 for 8GB GPUs, 1024 for 12GB+. May cause subtle seams."
                        )

                        overlap = gr.Number(
                            label="Tile Overlap",
                            value=values[10],
                            precision=0,
                            info="Pixels of overlap between tiles to prevent seam artifacts. Higher = smoother but slower. Try 32-128. Must be less than tile size."
                        )

                    batch_size = gr.Slider(
                        label="Batch Size (Frames per Iteration)",
                        minimum=1, maximum=16, step=1,
                        value=values[16],
                        info="Frames processed simultaneously for videos. Higher = faster but more VRAM. 1 = safest, 4-8 = balanced, 16 = max speed if VRAM allows."
                    )

                # Quality & Performance
                with gr.TabItem("🎨 Quality & Performance"):
                    gr.Markdown("#### Output Quality Settings")

                    with gr.Group():
                        denoising_strength = gr.Slider(
                            label="Denoising Strength",
                            minimum=0.0, maximum=1.0, step=0.05,
                            value=values[11],
                            info="Reduce noise/compression artifacts. 0 = no denoising, 1 = maximum. Try 0.3-0.7 for compressed videos. May reduce fine detail at high values."
                        )

                        sharpening = gr.Slider(
                            label="Output Sharpening",
                            minimum=0.0, maximum=2.0, step=0.1,
                            value=values[12],
                            info="Post-process sharpening. 0 = none, 1 = moderate, 2 = strong. Over-sharpening causes halos. Try 0.5-1.0 for balanced results."
                        )

                        color_correction = gr.Checkbox(
                            label="Color Correction",
                            value=values[13],
                            info="Maintain color accuracy by matching output colors to input. Prevents color shifts. Recommended ON for most content."
                        )

                        gpu_acceleration = gr.Checkbox(
                            label="GPU Acceleration",
                            value=values[14] if cuda_available else False,  # Force False if no CUDA
                            info=f"{gpu_hint} | Use GPU for processing. HIGHLY RECOMMENDED for speed. CPU fallback is 10-100x slower.",
                            interactive=cuda_available  # Disable if no CUDA
                        )

                        gpu_device = gr.Textbox(
                            label="GPU Device",
                            value=values[15] if cuda_available else "",  # Clear if no CUDA
                            placeholder="0 or all" if cuda_available else "CUDA not available",
                            info=f"GPU device ID(s). {cuda_count} GPU(s) detected. Single ID (0) for one GPU, 'all' for all available. Multi-GPU support model-dependent.",
                            interactive=cuda_available  # Disable if no CUDA
                        )
                        # FIXED: Live CUDA validation feedback for GAN tab
                        gpu_device_warning = gr.Markdown("", visible=False)

                # Output Settings
                with gr.TabItem("📤 Output Settings"):
                    gr.Markdown("#### File Output Configuration")

                    with gr.Group():
                        output_format_gan = gr.Dropdown(
                            label="Output Format",
                            choices=["auto", "png", "jpg", "webp"],
                            value=values[17],
                            info="'auto' matches input format"
                        )

                        output_quality_gan = gr.Slider(
                            label="Output Quality",
                            minimum=70, maximum=100, step=5,
                            value=values[18],
                            info="Quality for lossy formats (JPG/WebP)"
                        )

                        save_metadata = gr.Checkbox(
                            label="Save Processing Metadata",
                            value=values[19],
                            info="Embed processing information in output files"
                        )

                        create_subfolders = gr.Checkbox(
                            label="Create Subfolders by Model",
                            value=values[20],
                            info="Organize outputs in model-named subdirectories"
                        )
        
        # ===== RIGHT COLUMN: Output & Actions =====
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Output / Actions")
            
            # Status and progress
            status_box = gr.Markdown(value="Ready for processing.")
            progress_indicator = gr.Markdown(value="", visible=False)
            log_box = gr.Textbox(
                label="📋 Processing Log",
                value="",
                lines=10,
                buttons=["copy"]
            )

            # Output displays
            output_image = gr.Image(
                label="🖼️ Upscaled Image",
                interactive=False,
                buttons=["download"]
            )
            output_video = gr.Video(
                label="🎬 Upscaled Video",
                interactive=False,
                buttons=["download"]
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
            
            # Video Comparison with custom HTML5 slider
            video_comparison_html = gr.HTML(
                label="🎬 Video Comparison Slider",
                value="",
                visible=False
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
                cancel_btn = gr.Button(
                    "⏹️ Cancel",
                    variant="stop"
                )
                preview_btn = gr.Button(
                    "👁️ Preview First Frame",
                    size="lg"
                )
            
            cancel_confirm = gr.Checkbox(
                label="⚠️ Confirm cancel (required for safety)",
                value=False,
                info="Enable this checkbox to confirm cancellation of processing"
            )

            # Utility buttons
            with gr.Row():
                open_outputs_btn = gr.Button("📂 Open Outputs Folder")
                clear_temp_btn = gr.Button("🗑️ Clear Temp Files")

            # Preset management (in right column)
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

    # Model information (outside columns, full width)
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

    # ============================================================================
    # 📋 GAN PRESET INPUT LIST - MUST match GAN_ORDER in gan_service.py
    # Adding controls? Update gan_defaults(), GAN_ORDER, and this list in sync.
    # Current count: 21 components
    # ============================================================================
    
    inputs_list = [
        input_path, batch_enable, batch_input, batch_output, gan_model,
        target_resolution, downscale_first, auto_calculate_input, use_resolution_tab, tile_size, overlap,
        denoising_strength, sharpening, color_correction, gpu_acceleration, gpu_device,
        batch_size, output_format_gan, output_quality_gan, save_metadata, create_subfolders
    ]
    
    # Development validation
    if len(inputs_list) != len(GAN_ORDER):
        import logging
        logging.getLogger("GANTab").error(
            f"❌ inputs_list ({len(inputs_list)}) != GAN_ORDER ({len(GAN_ORDER)})"
        )

    # Wire up event handlers
    
    # FIXED: Live CUDA device validation for GAN tab
    def validate_cuda_device_live_gan(cuda_device_val):
        """Live CUDA validation for GAN models (enforces single GPU)"""
        if not cuda_device_val or not cuda_device_val.strip():
            return gr.update(value="", visible=False)
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                return gr.update(value="⚠️ CUDA not available. GPU acceleration disabled.", visible=True)
            
            device_str = str(cuda_device_val).strip()
            device_count = torch.cuda.device_count()
            
            if device_str.lower() == "all":
                return gr.update(value=f"⚠️ GAN models use single GPU. 'all' will use GPU 0 (of {device_count} available)", visible=True)
            
            devices = [d.strip() for d in device_str.replace(" ", "").split(",") if d.strip()]
            
            invalid_devices = []
            valid_devices = []
            
            for device in devices:
                if not device.isdigit():
                    invalid_devices.append(device)
                else:
                    device_id = int(device)
                    if device_id >= device_count:
                        invalid_devices.append(f"{device} (max: {device_count-1})")
                    else:
                        valid_devices.append(device_id)
            
            if invalid_devices:
                return gr.update(
                    value=f"❌ Invalid device ID(s): {', '.join(invalid_devices)}. Available: 0-{device_count-1}",
                    visible=True
                )
            
            if len(valid_devices) > 1:
                return gr.update(
                    value=f"⚠️ GAN models use single GPU. Will use GPU {valid_devices[0]} (ignoring others)",
                    visible=True
                )
            elif len(valid_devices) == 1:
                return gr.update(
                    value=f"✅ Using GPU {valid_devices[0]}",
                    visible=True
                )
            
            return gr.update(value="", visible=False)
        except Exception as e:
            return gr.update(value=f"⚠️ Validation error: {str(e)}", visible=True)
    
    # Wire up live CUDA validation
    gpu_device.change(
        fn=validate_cuda_device_live_gan,
        inputs=gpu_device,
        outputs=gpu_device_warning
    )

    # Input handling
    def cache_input(val, state):
        state["seed_controls"]["last_input_path"] = val if val else ""
        return val or "", gr.update(value="✅ Input cached for processing.", visible=True), state

    input_file.upload(
        fn=lambda val, state: cache_input(val, state),
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state]
    )

    input_path.change(
        fn=lambda val, state: (gr.update(value="✅ Input path updated.", visible=True), state),
        inputs=[input_path, shared_state],
        outputs=[input_cache_msg, shared_state]
    )

    # Batch results gallery
    batch_gallery = gr.Gallery(
        label="📦 Batch Results",
        visible=False,
        columns=4,
        rows=2,
        height="auto",
        object_fit="contain",
        buttons=["download"]
    )

    # Main processing with gr.Progress - include input_file upload
    upscale_btn.click(
        fn=lambda upload, *args, progress=gr.Progress(): service["run_action"](upload, *args[:-1], preview_only=False, state=args[-1], progress=progress),
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_image, output_video,
            last_processed, image_slider, video_comparison_html, batch_gallery, shared_state
        ]
    )

    preview_btn.click(
        fn=lambda upload, *args, progress=gr.Progress(): service["run_action"](upload, *args[:-1], preview_only=True, state=args[-1], progress=progress),
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_image, output_video,
            last_processed, image_slider, video_comparison_html, batch_gallery, shared_state
        ]
    )
    
    # Add resolution validation
    def validate_target_res_ui(val):
        is_valid, message, corrected = validate_resolution(val, must_be_multiple_of=64)
        if not is_valid:
            return corrected, gr.update(value=f"<span style='color: orange;'>{message}</span>", visible=True)
        return val, gr.update(value="", visible=False)
    
    target_resolution.change(
        fn=validate_target_res_ui,
        inputs=target_resolution,
        outputs=[target_resolution, target_res_warning]
    )

    cancel_btn.click(
        fn=lambda ok, state: (service["cancel_action"](), state) if ok else (gr.update(value="⚠️ Enable 'Confirm cancel' to stop."), "", state),
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

    # Preset management - use selected model context
    # First, add model selector for preset context
    with gr.Accordion("📌 Preset Model Context", open=False):
        gr.Markdown("*Presets are saved per-model. Select which model's presets to save/load:*")
        preset_model_selector = gr.Dropdown(
            label="Preset Context Model",
            choices=service["model_scanner"](),
            value=values[4],  # gan_model default
            info="Presets are saved/loaded for this specific model"
        )
    
    # Wire up preset operations with model context - FIXED: Flatten args properly
    save_preset_btn.click(
        fn=lambda name, model, *vals: service["save_preset"](name, model, *vals),
        inputs=[preset_name, preset_model_selector] + inputs_list,
        outputs=[preset_dropdown, preset_status] + inputs_list  # Captures reapplied validated values
    )

    load_preset_btn.click(
        fn=lambda preset, model, *vals: service["load_preset"](preset, model, list(vals)),
        inputs=[preset_dropdown, preset_model_selector] + inputs_list,
        outputs=inputs_list + [preset_status]  # Status is last (matches service return order)
    )

    safe_defaults_btn.click(
        fn=service["safe_defaults"],
        outputs=inputs_list
    )
    
    # Auto-sync preset model selector with main model selection
    gan_model.change(
        fn=lambda m: gr.update(value=m),
        inputs=gan_model,
        outputs=preset_model_selector
    )
    
    # Refresh presets when preset model selector changes
    def refresh_presets_for_model(model):
        presets = preset_manager.list_presets("gan", model)
        return gr.update(choices=presets, value="")
    
    preset_model_selector.change(
        fn=refresh_presets_for_model,
        inputs=preset_model_selector,
        outputs=preset_dropdown
    )
