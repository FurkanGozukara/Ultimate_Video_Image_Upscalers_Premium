"""
Image-Based (GAN) Tab - Self-contained modular implementation
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.gan_service import (
    build_gan_callbacks, GAN_ORDER
)
from shared.video_comparison_slider import create_video_comparison_html

from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from shared.universal_preset import dict_to_values
from shared.path_utils import get_media_dimensions, normalize_path
from shared.resolution_calculator import estimate_fixed_scale_upscale_plan_from_dims
from ui.media_preview import preview_updates


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

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    gan_settings = seed_controls.get("gan_settings", {})
    current_preset_name = seed_controls.get("current_preset_name")
    models_list = seed_controls.get("available_models", ["default"])
    
    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in gan_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    values = [merged_defaults[k] for k in GAN_ORDER]
    
    if current_preset_name:
        def update_status(state):
            existing = state["health_banner"]["text"]
            msg = f"✅ GAN: Using universal preset '{current_preset_name}'"
            state["health_banner"]["text"] = existing + "\n" + msg if existing else msg
            return state
        shared_state.value = update_status(shared_state.value)

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
                with gr.Row():
                    input_file = gr.File(
                        label="Upload Image or Video",
                        type="filepath",
                        file_types=["image", "video"]
                    )
                    with gr.Column():
                        input_image_preview = gr.Image(
                            label="📸 Input Preview (Image)",
                            type="filepath",
                            interactive=False,
                            height=250,
                            visible=False
                        )
                        input_video_preview = gr.Video(
                            label="🎬 Input Preview (Video)",
                            interactive=False,
                            height=250,
                            visible=False
                        )
                input_path = gr.Textbox(
                    label="Image/Video Path",
                    value=values[0],
                    placeholder="C:/path/to/image.jpg or C:/path/to/video.mp4",
                    info="Direct path to file or folder of images"
                )
                input_cache_msg = gr.Markdown("", visible=False)
                sizing_info = gr.Markdown("", visible=False, elem_classes=["resolution-info"])
                input_detection_result = gr.Markdown("", visible=False)

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
                            info="Pre-trained GAN models with fixed scale factors (2x, 4x, etc.). Models auto-detected from ./models (or legacy Image_Upscale_Models)."
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
                        # Legacy controls (kept for old presets, no longer used by vNext sizing)
                        target_resolution = gr.Slider(
                            label="(Legacy) Target Resolution (longest side) [internal]",
                            minimum=512, maximum=4096, step=64,
                            value=values[5],
                            visible=False,
                            interactive=False,
                        )
                        target_res_warning = gr.Markdown("", visible=False)

                        downscale_first = gr.Checkbox(
                            label="(Legacy) Downscale First if Needed [internal]",
                            value=values[6],
                            visible=False,
                            interactive=False,
                        )

                        auto_calculate_input = gr.Checkbox(
                            label="(Legacy) Auto-Calculate Input Resolution [internal]",
                            value=values[7],
                            visible=False,
                            interactive=False,
                        )

                        # NEW sizing controls (Upscale-x)
                        upscale_factor = gr.Number(
                            label="Upscale x (any factor)",
                            value=values[21] if len(values) > 21 else 4.0,
                            precision=2,
                            info="Target scale factor relative to input. For fixed-scale GAN models, input is pre-downscaled so one model pass reaches the target."
                        )

                        with gr.Row():
                            max_resolution = gr.Slider(
                                label="Max Resolution (max edge, 0 = no cap)",
                                minimum=0, maximum=8192, step=16,
                                value=values[22] if len(values) > 22 else 0,
                                info="Caps the LONG side (max(width,height)) of the target."
                            )
                            pre_downscale_then_upscale = gr.Checkbox(
                                label="⬇️➡️⬆️ Pre-downscale then upscale (auto when needed)",
                                value=values[23] if len(values) > 23 else False,
                                info="For fixed-scale GAN models this is applied automatically when needed to satisfy Upscale-x / Max Resolution without post-resize."
                            )

                        use_resolution_tab = gr.Checkbox(
                            label="🔗 Use Resolution & Scene Split Tab Settings",
                            value=values[8],
                            info="Apply Upscale-x, Max Resolution, and Pre-downscale settings from the Resolution tab. Recommended ON."
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

            # UNIVERSAL PRESET MANAGEMENT
            (
                preset_dropdown,
                preset_name_input,
                save_preset_btn,
                load_preset_btn,
                preset_status,
                reset_defaults_btn,
                delete_preset_btn,
                preset_callbacks,
            ) = universal_preset_section(
                preset_manager=preset_manager,
                shared_state=shared_state,
                tab_name="gan",
                inputs_list=[],  # Will be set after inputs_list is defined
                base_dir=base_dir,
                models_list=models_list,
                open_accordion=True,
            )

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
    # Current count: 24 components (includes vNext sizing)
    # ============================================================================
    
    inputs_list = [
        input_path, batch_enable, batch_input, batch_output, gan_model,
        target_resolution, downscale_first, auto_calculate_input, use_resolution_tab, tile_size, overlap,
        denoising_strength, sharpening, color_correction, gpu_acceleration, gpu_device,
        batch_size, output_format_gan, output_quality_gan, save_metadata, create_subfolders,
        # vNext sizing
        upscale_factor, max_resolution, pre_downscale_then_upscale
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
    def _build_input_detection_md(path_val: str) -> gr.update:
        from shared.input_detector import detect_input
        if not path_val or not str(path_val).strip():
            # Hide when empty (clearing input should clear this panel).
            return gr.update(value="", visible=False)
        try:
            info = detect_input(path_val)
            if not info.is_valid:
                return gr.update(value=f"❌ **Invalid Input**\n\n{info.error_message}", visible=True)
            parts = [f"✅ **Input Detected: {info.input_type.upper()}**"]
            if info.input_type == "frame_sequence":
                parts.append(f"&nbsp;&nbsp;📁 Pattern: `{info.frame_pattern}`")
                parts.append(f"&nbsp;&nbsp;🎞️ Frames: {info.frame_start}-{info.frame_end}")
                if info.missing_frames:
                    parts.append(f"&nbsp;&nbsp;⚠️ Missing: {len(info.missing_frames)}")
            elif info.input_type == "directory":
                parts.append(f"&nbsp;&nbsp;📂 Files: {info.total_files}")
            elif info.input_type in ["video", "image"]:
                parts.append(f"&nbsp;&nbsp;📄 Format: **{info.format.upper()}**")
            return gr.update(value=" ".join(parts), visible=True)
        except Exception as e:
            return gr.update(value=f"❌ **Detection Error**\n\n{str(e)}", visible=True)

    def _resolve_dims_for_preview(path_val: str):
        """Return (w,h) and a representative file path for directories (first media file)."""
        if not path_val:
            return None, None, None
        p = Path(normalize_path(path_val))
        if not p.exists():
            return None, None, None
        if p.is_file():
            dims = get_media_dimensions(str(p))
            return (dims[0], dims[1], str(p)) if dims else (None, None, str(p))
        # Directory: pick first media file
        exts = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".mp4", ".mov", ".mkv", ".avi", ".webm")
        items = [x for x in sorted(p.iterdir()) if x.is_file() and x.suffix.lower() in exts]
        if not items:
            return None, None, None
        dims = get_media_dimensions(str(items[0]))
        return (dims[0], dims[1], str(items[0])) if dims else (None, None, str(items[0]))

    def _build_sizing_info(
        input_path_val: str,
        model_name: str,
        use_global: bool,
        local_scale_x: float,
        local_max_edge: int,
        local_pre_down: bool,
        state,
    ) -> gr.update:
        if not input_path_val or not str(input_path_val).strip():
            return gr.update(visible=False)
        # Determine model scale
        try:
            from shared.gan_runner import get_gan_model_metadata
            meta = get_gan_model_metadata(model_name, base_dir)
            model_scale = int(meta.scale or 4)
        except Exception:
            model_scale = 4

        seed_controls = (state or {}).get("seed_controls", {})
        scale_x = float(seed_controls.get("upscale_factor_val", 4.0) or 4.0) if use_global else float(local_scale_x or 4.0)
        max_edge = int(seed_controls.get("max_resolution_val", 0) or 0) if use_global else int(local_max_edge or 0)
        pre_down = bool(seed_controls.get("ratio_downscale", False)) if use_global else bool(local_pre_down)

        enable_max = bool(seed_controls.get("enable_max_target", True)) if use_global else True
        if not enable_max:
            max_edge = 0

        w, h, rep = _resolve_dims_for_preview(input_path_val)
        if not w or not h:
            return gr.update(value="⚠️ Could not determine input dimensions for sizing preview.", visible=True)

        # Fixed-scale planning (GAN)
        plan = estimate_fixed_scale_upscale_plan_from_dims(
            int(w),
            int(h),
            requested_scale=float(scale_x),
            model_scale=int(model_scale),
            max_edge=int(max_edge or 0),
            force_pre_downscale=True,
        )

        out_w = plan.final_saved_width or plan.resize_width
        out_h = plan.final_saved_height or plan.resize_height
        input_short = min(plan.input_width, plan.input_height)
        out_short = min(int(out_w), int(out_h))

        items = []
        items.append(f"📐 <strong>Input:</strong> {plan.input_width}×{plan.input_height} (short side: {input_short}px)")

        t = f"🎯 <strong>Target setting:</strong> upscale {scale_x:g}x"
        if max_edge and max_edge > 0:
            t += f", max edge {max_edge}px (effective {plan.effective_scale:.2f}x)"
        items.append(t)

        # For fixed-scale models, pre-downscale is mandatory when the effective scale < model_scale.
        if plan.pre_downscale_then_upscale and plan.preprocess_scale < 0.999999:
            items.append(
                f"🧩 <strong>Preprocess:</strong> {plan.input_width}×{plan.input_height} → {plan.preprocess_width}×{plan.preprocess_height} (×{plan.preprocess_scale:.3f})"
            )

        items.append(f"🧱 <strong>Model pass:</strong> fixed {model_scale}x GAN upscale")
        items.append(f"✅ <strong>Final saved output:</strong> {int(out_w)}×{int(out_h)}")

        if out_short < input_short:
            items.append(f"📉 <strong>Mode:</strong> Downscaling (output short side {out_short}px < input short side {input_short}px)")
        elif out_short > input_short:
            items.append(f"📈 <strong>Mode:</strong> Upscaling (output short side {out_short}px > input short side {input_short}px)")
        else:
            items.append("➡️ <strong>Mode:</strong> Keep size (output short side matches input short side)")

        if plan.notes:
            for n in plan.notes:
                items.append(f"ℹ️ {n}")

        html = '<div style="font-size: 1.15em; line-height: 1.8;">' + "<br>".join(items) + "</div>"
        return gr.update(value=html, visible=True)

    def cache_input(val, model_val, use_global, scale_x, max_edge, pre_down, state):
        state["seed_controls"]["last_input_path"] = val if val else ""
        det = _build_input_detection_md(val or "")
        info = _build_sizing_info(val or "", model_val, bool(use_global), scale_x, max_edge, pre_down, state)
        img_prev, vid_prev = preview_updates(val)
        return (
            val or "",
            gr.update(value="✅ Input cached for processing.", visible=True),
            img_prev,
            vid_prev,
            det,
            info,
            state,
        )

    input_file.upload(
        fn=cache_input,
        inputs=[input_file, gan_model, use_resolution_tab, upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )

    # When the user clicks the “X” to clear the upload, clear the path + hide panels.
    def clear_on_upload_clear(file_path, state):
        if file_path:
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), state
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = ""
        except Exception:
            pass
        img_prev, vid_prev = preview_updates(None)
        return (
            "",  # input_path
            gr.update(value="", visible=False),  # input_cache_msg
            img_prev,
            vid_prev,
            gr.update(value="", visible=False),  # input_detection_result
            gr.update(value="", visible=False),  # sizing_info
            state,
        )

    input_file.change(
        fn=clear_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state],
    )

    def update_from_path(val, model_val, use_global, scale_x, max_edge, pre_down, state):
        det = _build_input_detection_md(val or "")
        info = _build_sizing_info(val or "", model_val, bool(use_global), scale_x, max_edge, pre_down, state)
        img_prev, vid_prev = preview_updates(val)
        return (
            gr.update(value="✅ Input path updated.", visible=True),
            img_prev,
            vid_prev,
            det,
            info,
            state,
        )

    input_path.change(
        fn=update_from_path,
        inputs=[input_path, gan_model, use_resolution_tab, upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[input_cache_msg, input_image_preview, input_video_preview, input_detection_result, sizing_info, shared_state]
    )

    # Refresh sizing info when settings change
    def refresh_sizing(scale_x, max_edge, pre_down, use_global, model_val, path_val, state):
        info = _build_sizing_info(path_val or "", model_val, bool(use_global), scale_x, max_edge, pre_down, state)
        return info, state

    for comp in [upscale_factor, max_resolution, pre_downscale_then_upscale, use_resolution_tab, gan_model]:
        comp.change(
            fn=refresh_sizing,
            inputs=[upscale_factor, max_resolution, pre_downscale_then_upscale, use_resolution_tab, gan_model, input_path, shared_state],
            outputs=[sizing_info, shared_state],
            trigger_mode="always_last",
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
    
    # NOTE: Legacy target_resolution is hidden; sizing is now driven by Upscale-x.

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

    # UNIVERSAL PRESET EVENT WIRING
    wire_universal_preset_events(
        preset_dropdown=preset_dropdown,
        preset_name_input=preset_name_input,
        save_btn=save_preset_btn,
        load_btn=load_preset_btn,
        preset_status=preset_status,
        reset_btn=reset_defaults_btn,
        delete_btn=delete_preset_btn,
        callbacks=preset_callbacks,
        inputs_list=inputs_list,
        shared_state=shared_state,
        tab_name="gan",
    )
