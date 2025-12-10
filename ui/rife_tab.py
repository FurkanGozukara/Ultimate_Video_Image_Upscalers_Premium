"""
RIFE / FPS / Edit Videos Tab - Self-contained modular implementation
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.rife_service import (
    build_rife_callbacks, RIFE_ORDER
)


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

    # Get defaults and last used
    defaults = service["defaults"]
    last_used_name = preset_manager.get_last_used_name("rife", defaults.get("model", "default"))
    last_used = preset_manager.load_last_used("rife", defaults.get("model", "default"))
    if last_used_name and last_used is None:
        def update_warning(state):
            existing = state["health_banner"]["text"]
            warning = f"Last used RIFE preset '{last_used_name}' not found; loaded defaults."
            if existing:
                state["health_banner"]["text"] = existing + "\n" + warning
            else:
                state["health_banner"]["text"] = warning
            return state
        shared_state.value = update_warning(shared_state.value)

    merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    values = [merged_defaults[k] for k in RIFE_ORDER]

    # Layout
    gr.Markdown("### ⏱️ RIFE / FPS / Edit Videos")
    gr.Markdown("*Frame interpolation, FPS adjustment, and video editing tools*")

    # Input section
    with gr.Accordion("📁 Input Configuration", open=True):
        input_file = gr.File(
            label="Upload Video or Image",
            type="filepath",
            file_types=["video", "image"],
            info="Select video file or image sequence for processing"
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
                    value=values[9],
                    info="Enable optimizations for 4K/8K videos. Uses more memory but handles high resolutions better. Enable for 3840x2160+ inputs."
                )

                rife_precision = gr.Dropdown(
                    label="Precision",
                    choices=["fp16", "fp32"],
                    value=values[10],
                    info="fp16 = half precision, 2x faster, uses less VRAM, minimal quality loss. fp32 = full precision, slower, more accurate. Use fp16 unless quality-critical."
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
                    label="GPU Device",
                    value=values[24],
                    placeholder="0 or all",
                    info="GPU to use for RIFE interpolation. Single GPU recommended. Use 'all' for all available GPUs (support limited). Leave empty for default GPU (0)."
                )

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
                    value=values[33],  # Updated index for concat_videos
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

        output_video = gr.Video(
            label="🎬 Processed Video",
            interactive=False,
            show_download_button=True
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

    # Preset management
    with gr.Accordion("💾 Preset Management", open=True):
        preset_dropdown = gr.Dropdown(
            label="RIFE Presets",
            choices=preset_manager.list_presets("rife", "default"),
            value=last_used_name or "",
        )

        with gr.Row():
            preset_name = gr.Textbox(
                label="Preset Name",
                placeholder="my_rife_preset"
            )
            save_preset_btn = gr.Button("💾 Save Preset", variant="secondary")

        with gr.Row():
            load_preset_btn = gr.Button("📂 Load Preset")
            safe_defaults_btn = gr.Button("🔄 Safe Defaults")

        preset_status = gr.Markdown("")

    # Info section
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
    process_btn.click(
        fn=lambda *args: service["run_action"](*args[:-1], state=args[-1]),
        inputs=inputs_list + [shared_state],
        outputs=[status_box, log_box, progress_indicator, output_video, shared_state]
    )

    cancel_btn.click(
        fn=lambda ok, state: (service["cancel_action"](), state) if ok else (gr.Markdown.update(value="⚠️ Enable 'Confirm cancel' to stop."), "", state),
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
