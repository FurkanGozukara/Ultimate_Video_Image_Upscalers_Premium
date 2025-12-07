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
            label="Upload Video",
            type="filepath",
            file_types=["video"],
            info="Select video file for processing"
        )
        input_path = gr.Textbox(
            label="Video Path",
            value=values[0],
            placeholder="C:/path/to/video.mp4",
            info="Direct path to video file"
        )
        input_cache_msg = gr.Markdown("", visible=False)

    with gr.Tabs():
        # Frame Interpolation (RIFE)
        with gr.TabItem("🎬 Frame Interpolation"):
            gr.Markdown("#### RIFE - Real-Time Intermediate Flow Estimation")

            with gr.Group():
                rife_enabled = gr.Checkbox(
                    label="Enable Frame Interpolation",
                    value=values[1],
                    info="Use RIFE to generate intermediate frames"
                )

                rife_model = gr.Dropdown(
                    label="RIFE Model",
                    choices=["rife-v4.6", "rife-v4.13", "rife-v4.14", "rife-v4.15", "rife-v4.16", "rife-anime"],
                    value=values[2],
                    info="Newer models are slower but higher quality"
                )

                target_fps = gr.Number(
                    label="Target FPS",
                    value=values[3],
                    precision=1,
                    info="Desired output frame rate"
                )

                fps_multiplier = gr.Dropdown(
                    label="FPS Multiplier",
                    choices=["x2", "x4", "x8"],
                    value=values[4],
                    info="How many times to multiply original FPS"
                )

                rife_precision = gr.Dropdown(
                    label="Precision",
                    choices=["fp16", "fp32"],
                    value=values[5],
                    info="fp16 is faster, fp32 is more accurate"
                )

                rife_gpu = gr.Textbox(
                    label="GPU Device",
                    value=values[6],
                    placeholder="0 or 0,1",
                    info="GPU(s) to use for RIFE processing"
                )

        # Video Editing
        with gr.TabItem("✂️ Video Editing"):
            gr.Markdown("#### Video Trimming & Effects")

            with gr.Group():
                edit_mode = gr.Dropdown(
                    label="Edit Mode",
                    choices=["trim", "concatenate", "speed_change", "effects"],
                    value=values[7],
                    info="Type of video editing to perform"
                )

                start_time = gr.Textbox(
                    label="Start Time (HH:MM:SS or seconds)",
                    value=values[8],
                    placeholder="00:00:30 or 30",
                    info="Where to start the edit"
                )

                end_time = gr.Textbox(
                    label="End Time (HH:MM:SS or seconds)",
                    value=values[9],
                    placeholder="00:01:30 or 90",
                    info="Where to end the edit"
                )

                speed_factor = gr.Slider(
                    label="Speed Factor",
                    minimum=0.25, maximum=4.0, step=0.25,
                    value=values[10],
                    info="1.0 = normal speed, 2.0 = 2x faster, 0.5 = 2x slower"
                )

        # Output Settings
        with gr.TabItem("📤 Output Settings"):
            gr.Markdown("#### Video Export Configuration")

            with gr.Group():
                output_format_rife = gr.Dropdown(
                    label="Output Format",
                    choices=["mp4", "avi", "mov", "webm"],
                    value=values[11],
                    info="Container format for output video"
                )

                video_codec_rife = gr.Dropdown(
                    label="Video Codec",
                    choices=["libx264", "libx265", "libvpx-vp9"],
                    value=values[12],
                    info="Compression codec"
                )

                output_quality_rife = gr.Slider(
                    label="Quality (CRF)",
                    minimum=0, maximum=51, step=1,
                    value=values[13],
                    info="Lower = higher quality, larger file"
                )

                no_audio = gr.Checkbox(
                    label="Remove Audio",
                    value=values[14],
                    info="Strip audio track from output"
                )

                show_ffmpeg_output = gr.Checkbox(
                    label="Show FFmpeg Output",
                    value=values[15],
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

    # Collect all inputs
    inputs_list = [
        input_path, rife_enabled, rife_model, target_fps, fps_multiplier,
        rife_precision, rife_gpu, edit_mode, start_time, end_time,
        speed_factor, output_format_rife, video_codec_rife, output_quality_rife,
        no_audio, show_ffmpeg_output
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
