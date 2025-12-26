"""
RIFE / FPS / Edit Videos Tab - Self-contained modular implementation
UPDATED: Now uses Universal Preset System
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.rife_service import (
    build_rife_callbacks, RIFE_ORDER
)
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
)
from shared.universal_preset import dict_to_values
from ui.media_preview import preview_updates


def rife_tab(
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
    Self-contained RIFE / FPS / Edit Videos tab.
    Handles frame interpolation, FPS changes, and video editing.
    """

    # Build service callbacks
    service = build_rife_callbacks(
        preset_manager, runner, run_logger, global_settings,
        output_dir, temp_dir, shared_state
    )

    # Get defaults
    defaults = service["defaults"]
    
    # UNIVERSAL PRESET: Load from shared_state
    seed_controls = shared_state.value.get("seed_controls", {})
    rife_settings = seed_controls.get("rife_settings", {})
    current_preset_name = seed_controls.get("current_preset_name")
    models_list = seed_controls.get("available_models", ["default"])
    
    # Merge with defaults
    merged_defaults = defaults.copy()
    for key, value in rife_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    values = [merged_defaults[k] for k in RIFE_ORDER]
    
    if current_preset_name:
        def update_status(state):
            existing = state["health_banner"]["text"]
            msg = f"✅ RIFE: Using universal preset '{current_preset_name}'"
            state["health_banner"]["text"] = existing + "\n" + msg if existing else msg
            return state
        shared_state.value = update_status(shared_state.value)

    # GPU availability check (like SeedVR2/GAN tabs)
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
    gr.Markdown("### ⏱️ RIFE / FPS / Edit Videos")
    gr.Markdown("*Frame interpolation, FPS adjustment, and video editing tools*")
    
    # Import shared layout helpers
    from ui.shared_layouts import create_gpu_warning_banner
    
    # Show GPU warning if not available
    create_gpu_warning_banner(cuda_available, gpu_hint, "RIFE")

    # Two-column layout
    with gr.Row():
        # ===== LEFT COLUMN: Input & Controls =====
        with gr.Column(scale=3):
            gr.Markdown("### 📥 Input / Controls")
            
            # Input section
            with gr.Accordion("📁 Input Configuration", open=True):
                with gr.Row():
                    input_file = gr.File(
                        label="Upload Video or Image",
                        type="filepath",
                        file_types=["video", "image"]
                    )
                    with gr.Column():
                        input_image_preview = gr.Image(
                            label="📸 Input Preview (Image)",
                            type="filepath",
                            interactive=False,
                            height=220,
                            visible=False,
                        )
                        input_video_preview = gr.Video(
                            label="🎬 Input Preview (Video)",
                            interactive=False,
                            height=220,
                            visible=False,
                        )
                input_path = gr.Textbox(
                    label="Input Path",
                    value=values[0],
                    placeholder="C:/path/to/video.mp4 or C:/path/to/images/",
                    info="Direct path to video file or image folder"
                )
                input_cache_msg = gr.Markdown("", visible=False)
                
                # Batch processing controls
                batch_enable = gr.Checkbox(
                    label="Enable Batch Processing",
                    value=values[19],
                    info="Process multiple files from directory"
                )
                batch_input = gr.Textbox(
                    label="Batch Input Folder",
                    value=values[20],
                    placeholder="Folder containing videos",
                    info="Directory with files to process in batch mode"
                )
                batch_output = gr.Textbox(
                    label="Batch Output Folder Override",
                    value=values[21],
                    placeholder="Optional override for batch outputs",
                    info="Custom output directory for batch results"
                )

            # Processing settings
            with gr.Tabs():
                # Frame Interpolation (RIFE)
                with gr.TabItem("🎬 Frame Interpolation"):
                    gr.Markdown("#### RIFE - Real-Time Intermediate Flow Estimation")

                    # Output controls at top (more important than RIFE toggle for workflow)
                    with gr.Group():
                        gr.Markdown("#### 📁 Output Configuration")
                        
                        output_override = gr.Textbox(
                            label="Output Override (custom path)",
                            value=values[2],
                            placeholder="Leave empty for auto naming",
                            info="Specify custom output path. Auto-naming creates files in output folder."
                        )
                        
                        output_format_rife = gr.Dropdown(
                            label="Output Format",
                            choices=["auto", "mp4", "avi", "mov", "webm"],
                            value=values[3],
                            info="Container format for output video"
                        )
                        
                        png_output = gr.Checkbox(
                            label="Export as PNG Sequence",
                            value=values[11],
                            info="Save output as numbered PNG frames instead of video file. Useful for further editing."
                        )
                    
                    with gr.Group():
                        gr.Markdown("#### ⏱️ RIFE Interpolation")
                        
                        rife_enabled = gr.Checkbox(
                            label="Enable Frame Interpolation",
                            value=values[1],
                            info="Use RIFE AI model to generate smooth intermediate frames. Creates slow-motion or higher FPS videos. Essential for fluid motion."
                        )

                        def _discover_rife_models():
                            """Dynamically discover available RIFE models."""
                            # Default fallback models
                            default_models = ["rife-v4.6", "rife-v4.13", "rife-v4.14", "rife-v4.15", "rife-v4.16", "rife-v4.17", "rife-anime"]
                            
                            # Try to scan train_log for actual models
                            rife_dir = base_dir / "RIFE" / "train_log"
                            discovered_models = []
                            if rife_dir.exists():
                                for item in rife_dir.iterdir():
                                    if item.is_dir() and not item.name.startswith("_"):
                                        discovered_models.append(item.name)
                            
                            # Return discovered models if found, else default list
                            return discovered_models if discovered_models else default_models
                        
                        model_dir = gr.Textbox(
                            label="Model Directory Override",
                            value=values[4],
                            placeholder="Leave empty for default (RIFE/train_log)",
                            info="Custom path to RIFE model directory. Only needed if models are in non-standard location."
                        )
                        
                        rife_model = gr.Dropdown(
                            label="RIFE Model",
                            choices=_discover_rife_models(),
                            value=values[5],
                            info="RIFE model version. v4.6 = fastest. v4.15+ = best quality. 'anime' optimized for animation. Newer versions slower but smoother."
                        )
                        
                        # Model info display with metadata
                        rife_model_info = gr.Markdown("")
                        
                        def update_rife_model_info(model_name_val):
                            """Display RIFE model metadata information"""
                            from shared.models.rife_meta import get_rife_metadata
                            
                            metadata = get_rife_metadata(model_name_val)
                            
                            if metadata:
                                info_lines = [
                                    f"**📊 Model: {metadata.name}**",
                                    f"**Version:** {metadata.version} | **Variant:** {metadata.variant.title()}",
                                    f"**VRAM Estimate:** ~{metadata.estimated_vram_gb:.1f}GB",
                                    f"**Multi-GPU:** {'❌ Not supported (single GPU only)' if not metadata.supports_multi_gpu else '✅ Supported'}",
                                    f"**Max FPS Multiplier:** {metadata.max_fps_multiplier}x",
                                    f"**UHD Mode:** {'✅ Supported (recommended for 4K+)' if metadata.supports_uhd else '❌ Not available'}",
                                ]
                                if metadata.notes:
                                    info_lines.append(f"\n💡 {metadata.notes}")
                                
                                return gr.update(value="\n".join(info_lines), visible=True)
                            else:
                                return gr.update(value="Model metadata not available", visible=False)
                        
                        # Wire up model info update
                        rife_model.change(
                            fn=update_rife_model_info,
                            inputs=rife_model,
                            outputs=rife_model_info
                        )

                        fps_multiplier = gr.Dropdown(
                            label="FPS Multiplier",
                            choices=["x1", "x2", "x4", "x8"],
                            value=values[6],
                            info="Multiply original FPS. x2 = double smoothness (30→60fps). x4 = 4x smoother. x8 = extreme slow-mo. Higher = more processing time."
                        )
                        
                        target_fps = gr.Number(
                            label="Target FPS Override",
                            value=values[7],
                            precision=1,
                            info="Desired output frame rate. 0 = use multiplier instead. 60 = smooth 60fps. 120 = ultra-smooth. Higher FPS = larger file size."
                        )
                        
                        scale = gr.Slider(
                            label="Spatial Scale Factor",
                            minimum=0.5, maximum=4.0, step=0.1,
                            value=values[8],
                            info="Scale video resolution. 1.0 = original size, 2.0 = double resolution. Can combine with interpolation. >1.0 significantly increases processing time."
                        )
                        
                        uhd_mode = gr.Checkbox(
                            label="UHD Mode (4K+ Processing)",
                            value=values[9] if cuda_available else False,  # Force False if no CUDA
                            info=f"{gpu_hint} | Enable optimizations for 4K/8K videos. Uses more memory. Enable for 3840x2160+ inputs.",
                            interactive=cuda_available  # Disable if no CUDA
                        )

                        rife_precision = gr.Dropdown(
                            label="Precision",
                            choices=["fp16", "fp32"],
                            value=values[10] if cuda_available else "fp32",  # Force fp32 if no CUDA
                            info=f"fp16 = half precision, 2x faster, less VRAM. fp32 = full precision. {'(fp16 requires GPU)' if not cuda_available else 'Use fp16 for speed.'}",
                            interactive=cuda_available  # Disable if no CUDA (CPU uses fp32 only)
                        )
                        
                        montage = gr.Checkbox(
                            label="📊 Create Montage (Side-by-Side Comparison)",
                            value=values[14],
                            info="Generate side-by-side comparison video showing original vs interpolated. Useful for quality checking."
                        )
                        
                        img_mode = gr.Checkbox(
                            label="🖼️ Image Sequence Mode",
                            value=values[15],
                            info="Process image sequence instead of video. Automatically enabled when input is folder of images."
                        )
                        
                        skip_static_frames = gr.Checkbox(
                            label="Skip Static Frames (Auto-Detect)",
                            value=values[16],
                            info="Automatically skip static/duplicate frames. Saves processing time for videos with static scenes. May miss subtle motion."
                        )
                        
                        exp = gr.Number(
                            label="Temporal Recursion Depth",
                            value=values[17],
                            precision=0,
                            info="Exponential frame generation depth. 1 = direct interpolation, 2+ = recursive. Higher = smoother but exponentially slower. Use 1 for most cases."
                        )

                        rife_gpu = gr.Textbox(
                            label="GPU Device (Single GPU Only)",
                            value=values[24] if cuda_available else "",
                            placeholder="0" if cuda_available else "CPU only (no CUDA)",
                            info=f"{gpu_hint}\n⚠️ RIFE uses SINGLE GPU only. Multi-GPU not supported. Enter single GPU ID (e.g., 0, 1, 2). Leave empty for default (GPU 0).",
                            interactive=cuda_available
                        )
                        # FIXED: Live CUDA validation feedback for RIFE tab
                        rife_gpu_warning = gr.Markdown("", visible=False)

                # Video Editing
                with gr.TabItem("✂️ Video Editing"):
                    gr.Markdown("#### Video Trimming & Effects")

                    with gr.Group():
                        edit_mode = gr.Dropdown(
                            label="Edit Mode",
                            choices=["none", "trim", "concatenate", "speed_change", "effects"],
                            value=values[25],
                            info="Type of video editing to perform"
                        )

                        start_time = gr.Textbox(
                            label="Start Time (HH:MM:SS or seconds)",
                            value=values[26],
                            placeholder="00:00:30 or 30",
                            info="Where to start the edit"
                        )

                        end_time = gr.Textbox(
                            label="End Time (HH:MM:SS or seconds)",
                            value=values[27],
                            placeholder="00:01:30 or 90",
                            info="Where to end the edit"
                        )

                        speed_factor = gr.Slider(
                            label="Speed Factor",
                            minimum=0.25, maximum=4.0, step=0.25,
                            value=values[28],
                            info="1.0 = normal speed, 2.0 = 2x faster, 0.5 = 2x slower"
                        )

                        concat_videos = gr.Textbox(
                            label="Additional Videos for Concatenation",
                            value=values[31],  # Fixed: concat_videos is at index 31 in RIFE_ORDER
                            placeholder="C:/path/to/video1.mp4, C:/path/to/video2.mp4",
                            info="Comma-separated list of video files to concatenate with the main input",
                            lines=2
                        )

                # Frame Control & Advanced
                with gr.TabItem("🎞️ Frame Control"):
                    gr.Markdown("#### Advanced Frame Processing")

                    with gr.Group():
                        skip_first_frames = gr.Number(
                            label="Skip First Frames",
                            value=values[22],
                            precision=0,
                            info="Skip N frames from start of video. Useful to skip intros/logos. 0 = process from beginning."
                        )

                        load_cap = gr.Number(
                            label="Frame Load Cap (0 = all)",
                            value=values[23],
                            precision=0,
                            info="Process only first N frames. Useful for quick tests. 0 = process entire video. Combine with skip for specific range."
                        )

                # Output Settings
                with gr.TabItem("📤 Output Settings"):
                    gr.Markdown("#### Video Export Configuration")

                    with gr.Group():
                        video_codec_rife = gr.Dropdown(
                            label="Video Codec",
                            choices=["libx264", "libx265", "libvpx-vp9"],
                            value=values[29],
                            info="Compression codec"
                        )

                        output_quality_rife = gr.Slider(
                            label="Quality (CRF)",
                            minimum=0, maximum=51, step=1,
                            value=values[30],
                            info="Lower = higher quality, larger file"
                        )

                        no_audio = gr.Checkbox(
                            label="Remove Audio",
                            value=values[12],
                            info="Strip audio track from output"
                        )

                        show_ffmpeg_output = gr.Checkbox(
                            label="Show FFmpeg Output",
                            value=values[13],
                            info="Display detailed processing logs"
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
            output_video = gr.Video(
                label="🎬 Processed Video",
                interactive=False,
                buttons=["download"]
            )
            
            # Comparison outputs (matching SeedVR2/GAN tabs)
            image_slider = gr.ImageSlider(
                label="🔍 Before/After Comparison",
                interactive=False,
                height=500,
                slider_position=50,
                max_height=600,
                buttons=["download", "fullscreen"]
            )
            
            video_comparison_html = gr.HTML(
                label="🎬 Video Comparison Slider",
                value="",
                visible=False
            )

            # Action buttons
            with gr.Row():
                process_btn = gr.Button(
                    "🚀 Process Video",
                    variant="primary",
                    size="lg"
                )
                cancel_btn = gr.Button(
                    "⏹️ Cancel",
                    variant="stop",
                    size="lg"
                )
            
            cancel_confirm = gr.Checkbox(
                label="⚠️ Confirm cancel (required for safety)",
                value=False,
                info="Enable this checkbox to confirm cancellation"
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
                tab_name="rife",
                inputs_list=[],
                base_dir=base_dir,
                models_list=models_list,
                open_accordion=True,
            )

    # Info section (outside columns, full width)
    with gr.Accordion("ℹ️ About RIFE & FPS", open=False):
        gr.Markdown("""
        #### RIFE (Real-Time Intermediate Flow Estimation)

        **What it does:**
        - Generates smooth intermediate frames between existing frames
        - Converts 30fps video to 60fps, 120fps, etc.
        - Creates natural motion without stuttering

        **Use cases:**
        - Smooth slow-motion video
        - Fix stuttering from low frame rate sources
        - Enhance video playback quality

        **Performance notes:**
        - Processing time increases with multiplier
        - Higher quality models are slower
        - GPU acceleration highly recommended

        #### Video Editing Features

        **Trimming:** Cut specific time ranges
        **Speed Change:** Slow down or speed up video
        **Effects:** Apply various video filters
        **Format Conversion:** Change codecs/containers
        """)

    # Collect all inputs matching RIFE_ORDER exactly
    # IMPORTANT: Order must match RIFE_ORDER in shared/services/rife_service.py
    # ============================================================================
    # 📋 RIFE PRESET INPUT LIST - MUST match RIFE_ORDER in rife_service.py
    # Adding controls? Update rife_defaults(), RIFE_ORDER, and this list in sync.
    # Current count: 32 components
    # ============================================================================
    
    inputs_list = [
        input_path,           # 0: input_path
        rife_enabled,         # 1: rife_enabled  
        output_override,      # 2: output_override (NOW SEPARATE!)
        output_format_rife,   # 3: output_format
        model_dir,            # 4: model_dir (NOW EXPOSED!)
        rife_model,           # 5: model
        fps_multiplier,       # 6: fps_multiplier
        target_fps,           # 7: fps_override
        scale,                # 8: scale (NOW EXPOSED!)
        uhd_mode,             # 9: uhd_mode (NOW EXPOSED!)
        rife_precision,       # 10: fp16_mode
        png_output,           # 11: png_output (NOW EXPOSED!)
        no_audio,             # 12: no_audio
        show_ffmpeg_output,   # 13: show_ffmpeg
        montage,              # 14: montage
        img_mode,             # 15: img_mode
        skip_static_frames,   # 16: skip_static_frames (NOW EXPOSED!)
        exp,                  # 17: exp (NOW EXPOSED!)
        gr.State(2),          # 18: multi (internal - fps_multiplier handles this)
        batch_enable,         # 19: batch_enable (NOW EXPOSED!)
        batch_input,          # 20: batch_input_path (NOW EXPOSED!)
        batch_output,         # 21: batch_output_path (NOW EXPOSED!)
        skip_first_frames,    # 22: skip_first_frames (NOW EXPOSED!)
        load_cap,             # 23: load_cap (NOW EXPOSED!)
        rife_gpu,             # 24: cuda_device
        edit_mode,            # 25: edit_mode
        start_time,           # 26: start_time
        end_time,             # 27: end_time
        speed_factor,         # 28: speed_factor
        video_codec_rife,     # 29: video_codec
        output_quality_rife,  # 30: output_quality
        concat_videos,        # 31: concat_videos
    ]
    
    # Development validation
    if len(inputs_list) != len(RIFE_ORDER):
        import logging
        logging.getLogger("RIFETab").error(
            f"❌ inputs_list ({len(inputs_list)}) != RIFE_ORDER ({len(RIFE_ORDER)})"
        )

    # Wire up event handlers
    
    # FIXED: Live CUDA device validation for RIFE tab (enforces single GPU)
    def validate_cuda_device_live_rife(cuda_device_val):
        """Live CUDA validation for RIFE (enforces single GPU)"""
        if not cuda_device_val or not cuda_device_val.strip():
            return gr.update(value="", visible=False)
        
        try:
            import torch
            
            if not torch.cuda.is_available():
                return gr.update(value="⚠️ CUDA not available. CPU mode will be used (very slow).", visible=True)
            
            device_str = str(cuda_device_val).strip()
            device_count = torch.cuda.device_count()
            
            if device_str.lower() == "all":
                return gr.update(value=f"⚠️ RIFE uses single GPU only. Will use GPU 0 (ignoring 'all')", visible=True)
            
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
                    value=f"⚠️ RIFE single GPU only. Will use GPU {valid_devices[0]} (ignoring {', '.join(map(str, valid_devices[1:]))})",
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
    rife_gpu.change(
        fn=validate_cuda_device_live_rife,
        inputs=rife_gpu,
        outputs=rife_gpu_warning
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

    # Input preview (image + video)
    input_file.change(
        fn=lambda p: preview_updates(p),
        inputs=[input_file],
        outputs=[input_image_preview, input_video_preview],
    )

    # If user clears the upload (clicks “X”), clear the textbox + hide the cached message.
    def clear_on_upload_clear(file_path, state):
        if file_path:
            return gr.update(), gr.update(), state
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = ""
        except Exception:
            pass
        return "", gr.update(value="", visible=False), state

    input_file.change(
        fn=clear_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state],
    )

    input_path.change(
        fn=lambda val, state: (gr.update(value="✅ Input path updated.", visible=True), state),
        inputs=[input_path, shared_state],
        outputs=[input_cache_msg, shared_state]
    )

    input_path.change(
        fn=lambda p: preview_updates(p),
        inputs=[input_path],
        outputs=[input_image_preview, input_video_preview],
    )

    # Main processing
    process_btn.click(
        fn=lambda *args: service["run_action"](*args[:-1], state=args[-1]),
        inputs=inputs_list + [shared_state],
        outputs=[status_box, log_box, progress_indicator, output_video, image_slider, video_comparison_html, shared_state]
    )

    cancel_btn.click(
        fn=lambda ok, state: (service["cancel_action"](), state) if ok else (gr.update(value="⚠️ Enable 'Confirm cancel' to stop."), "", state),
        inputs=[cancel_confirm, shared_state],
        outputs=[status_box, log_box, shared_state]
    )

    # Utility functions
    open_outputs_btn.click(
        fn=lambda: service["open_outputs_folder"](),
        outputs=status_box
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
        tab_name="rife",
    )
