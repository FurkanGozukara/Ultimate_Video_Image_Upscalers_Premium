"""
Output & Comparison Tab - Self-contained modular implementation
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.output_service import (
    build_output_callbacks, OUTPUT_ORDER
)
from shared.models import (
    get_seedvr2_model_names,
    get_flashvsr_model_names,
    get_rife_model_names,
    scan_gan_models
)
from shared.video_codec_options import (
    get_codec_choices,
    get_pixel_format_choices,
    get_codec_info,
    get_pixel_format_info,
    ENCODING_PRESETS,
    AUDIO_CODECS
)


def output_tab(preset_manager, shared_state: gr.State, base_dir: Path, global_settings: Dict[str, Any] = None):
    """
    Self-contained Output & Comparison tab.
    Handles output format and comparison settings shared across ALL upscaler models.
    """

    # Get available models from ALL pipelines (SeedVR2, GAN, FlashVSR+, RIFE)
    seedvr2_models = get_seedvr2_model_names()
    gan_models = scan_gan_models(base_dir)
    flashvsr_models = get_flashvsr_model_names()
    rife_models = get_rife_model_names(base_dir)
    
    # Combine and deduplicate all models for per-model preset support
    combined_models = sorted(list({
        *seedvr2_models,
        *gan_models,
        *flashvsr_models,
        *rife_models
    }))

    # Build service callbacks with global_settings for pinned reference persistence
    service = build_output_callbacks(preset_manager, shared_state, combined_models, global_settings)

    # Get defaults and last used
    current_model = combined_models[0] if combined_models else "default"
    defaults = service["defaults"]
    last_used_name = preset_manager.get_last_used_name("output", current_model)
    last_used = preset_manager.load_last_used("output", current_model)
    if last_used_name and last_used is None:
        def update_warning(state):
            existing = state["health_banner"]["text"]
            warning = f"Last used Output preset '{last_used_name}' not found; loaded defaults."
            if existing:
                state["health_banner"]["text"] = existing + "\n" + warning
            else:
                state["health_banner"]["text"] = warning
            return state
        shared_state.value = update_warning(shared_state.value)

    merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    values = [merged_defaults[k] for k in OUTPUT_ORDER]

    # Layout
    gr.Markdown("### 🎭 Output & Comparison Settings")
    gr.Markdown("*Configure output formats, FPS handling, and comparison display options shared across all upscaler models*")

    # Model selector for presets
    model_selector = gr.Dropdown(
        label="Model Context (for presets)",
        choices=combined_models,
        value=current_model,
        info="Settings are saved/loaded per model type"
    )

    with gr.Tabs():
        # Output Format Settings
        with gr.TabItem("📁 Output Format"):
            gr.Markdown("#### File Output Configuration")

            with gr.Group():
                output_format = gr.Dropdown(
                    label="Output Format",
                    choices=["auto", "mp4", "png"],
                    value=values[0],
                    info="'auto' uses mp4 for videos, png for images. Explicit format overrides auto-detection."
                )

                png_sequence_enabled = gr.Checkbox(
                    label="Enable PNG Sequence Output",
                    value=values[1],
                    info="Save as numbered PNG frames instead of video (useful for further processing)"
                )

                png_padding = gr.Slider(
                    label="PNG Frame Number Padding",
                    minimum=1, maximum=10, step=1,
                    value=values[2],
                    info="Number of digits for frame numbers (e.g., 5 = 00001.png, 6 = 000001.png).\n\n"
                         "⚠️ **CRITICAL MODEL-SPECIFIC LIMITATION:**\n"
                         "• **SeedVR2**: ❌ HARDCODED to 6 digits in CLI (line 728 of inference_cli.py)\n"
                         "  → This slider has NO EFFECT on SeedVR2 PNG outputs\n"
                         "  → SeedVR2 will ALWAYS use 6-digit padding regardless of this setting\n"
                         "• **GAN/RIFE/FlashVSR**: ✅ Fully respects this setting\n\n"
                         "💡 **Recommendation**: Keep at 6 for consistency. If using SeedVR2 + other models,\n"
                         "setting to 6 ensures all outputs match. Custom padding only works for GAN/RIFE/FlashVSR.",
                    interactive=True
                )

                png_keep_basename = gr.Checkbox(
                    label="Keep Original Basename in PNG Names",
                    value=values[3],
                    info="Preserve input filename as base for PNG frames (e.g., 'video.mp4' → 'video_00001.png').\n\n"
                         "⚠️ **MODEL-SPECIFIC BEHAVIOR:**\n"
                         "• **SeedVR2**: ✅ Always preserves input basename (CLI design)\n"
                         "  → This checkbox has NO EFFECT on SeedVR2 outputs\n"
                         "  → SeedVR2 will ALWAYS keep basename regardless of this setting\n"
                         "• **GAN/RIFE/FlashVSR**: ✅ Fully respects this setting\n\n"
                         "💡 **Note**: All models use collision-safe naming (_0001, _0002, etc.) to prevent overwrites."
                )

        # Video Settings
        with gr.TabItem("🎬 Video Output"):
            gr.Markdown("#### Video Encoding & FPS")

            with gr.Group():
                fps_override = gr.Number(
                    label="FPS Override",
                    value=values[4],
                    precision=2,
                    info="Override output FPS (0 = use source FPS)"
                )
                
                gr.Markdown("---\n**Codec Selection**")

                video_codec = gr.Dropdown(
                    label="Video Codec",
                    choices=get_codec_choices(),
                    value=values[5] if values[5] in get_codec_choices() else "h264",
                    info="Choose encoding codec based on your use case"
                )
                
                codec_info_display = gr.Markdown("")
                
                pixel_format = gr.Dropdown(
                    label="Pixel Format",
                    choices=["yuv420p", "yuv422p", "yuv444p", "yuv420p10le", "yuv444p10le", "rgb24"],
                    value=values[11] if len(values) > 11 else "yuv420p",
                    info="Color subsampling and bit depth. yuv420p = best compatibility"
                )
                
                pixel_format_info = gr.Markdown("")

                video_quality = gr.Slider(
                    label="Video Quality (CRF - lower is better)",
                    minimum=0, maximum=51, step=1,
                    value=values[6],
                    info="0 = lossless (huge files), 18 = visually lossless, 23 = high quality, 28 = medium, 35+ = low quality"
                )

                video_preset = gr.Dropdown(
                    label="Encoding Preset",
                    choices=ENCODING_PRESETS,
                    value=values[7],
                    info="ultrafast = fastest encoding, veryslow = best compression. medium = balanced"
                )
                
                gr.Markdown("---\n**Audio Options**")
                
                audio_codec = gr.Dropdown(
                    label="Audio Codec",
                    choices=list(AUDIO_CODECS.keys()),
                    value=values[12] if len(values) > 12 else "copy",
                    info="Audio encoding: copy = no re-encode (fastest), aac = compatible, flac = lossless, none = remove audio"
                )
                
                audio_bitrate = gr.Textbox(
                    label="Audio Bitrate (optional)",
                    value=values[13] if len(values) > 13 else "",
                    placeholder="192k, 320k, etc.",
                    info="Only used when re-encoding audio (not for 'copy')"
                )

                two_pass_encoding = gr.Checkbox(
                    label="Two-Pass Encoding",
                    value=values[8],
                    info="Slower but better quality/filesize ratio. Recommended for archival."
                )
                
                # Quick preset buttons
                gr.Markdown("---\n**Quick Presets**")
                with gr.Row():
                    preset_youtube = gr.Button("🎬 YouTube", size="sm")
                    preset_archival = gr.Button("💾 Archival", size="sm")
                    preset_editing = gr.Button("✂️ Editing", size="sm")
                    preset_web = gr.Button("🌐 Web", size="sm")

        # Frame Handling
        with gr.TabItem("🎭 Frame Processing"):
            gr.Markdown("#### Frame Trimming & Timing")

            with gr.Group():
                skip_first_frames = gr.Number(
                    label="Skip First Frames",
                    value=values[9],
                    precision=0,
                    info="Number of frames to skip from start"
                )

                load_cap = gr.Number(
                    label="Frame Load Cap",
                    value=values[10],
                    precision=0,
                    info="Maximum frames to process (0 = all frames)"
                )

                temporal_padding = gr.Number(
                    label="Temporal Padding",
                    value=values[14],
                    precision=0,
                    info="Extra frames for temporal processing"
                )

                frame_interpolation = gr.Checkbox(
                    label="Enable Frame Interpolation",
                    value=values[15],
                    info="Smooth motion between frames"
                )

        # Comparison Settings
        with gr.TabItem("🔍 Comparison Display"):
            gr.Markdown("#### Comparison Viewer Configuration")

            with gr.Group():
                comparison_mode = gr.Dropdown(
                    label="Comparison Mode",
                    choices=["native", "slider", "side_by_side", "overlay"],
                    value=values[16],
                    info="How to display before/after comparison"
                )

                pin_reference = gr.Checkbox(
                    label="Pin Reference Image",
                    value=values[17],
                    info="Keep original as fixed reference when changing settings"
                )

                fullscreen_enabled = gr.Checkbox(
                    label="Enable Fullscreen Comparison",
                    value=values[18],
                    info="Allow fullscreen viewing of comparisons"
                )

                comparison_zoom = gr.Slider(
                    label="Default Zoom Level",
                    minimum=25, maximum=400, step=25,
                    value=values[19],
                    info="Default zoom percentage for comparison viewer"
                )

                show_difference = gr.Checkbox(
                    label="Show Difference Overlay",
                    value=values[20],
                    info="Highlight differences between original and upscaled"
                )

        # Metadata & Logging
        with gr.TabItem("📊 Metadata & Logging"):
            gr.Markdown("#### Output Metadata & Telemetry")

            with gr.Group():
                save_metadata = gr.Checkbox(
                    label="Save Processing Metadata",
                    value=values[21],
                    info="Embed processing info in output files"
                )

                metadata_format = gr.Dropdown(
                    label="Metadata Format",
                    choices=["json", "xml", "exif", "none"],
                    value=values[22],
                    info="Format for embedded metadata"
                )

                telemetry_enabled = gr.Checkbox(
                    label="Enable Run Telemetry",
                    value=values[23],
                    info="Log processing stats for troubleshooting"
                )

                log_level = gr.Dropdown(
                    label="Log Verbosity",
                    choices=["error", "warning", "info", "debug"],
                    value=values[24],
                    info="Detail level for processing logs"
                )

    # Pin Reference Feature
    with gr.Accordion("📌 Pin Reference Frame", open=False):
        gr.Markdown("#### Pin a reference frame for iterative comparison")
        gr.Markdown("*Useful when testing different settings - keep the original pinned while comparing multiple upscaled versions*")
        
        pinned_reference_display = gr.Image(
            label="Pinned Reference",
            interactive=False,
            height=200
        )
        
        with gr.Row():
            pin_btn = gr.Button("📌 Pin Current Output", variant="secondary")
            unpin_btn = gr.Button("❌ Unpin Reference")
        
        pin_status = gr.Markdown("")

    # Apply to Pipeline
    gr.Markdown("#### 🔗 Apply to Pipeline")
    apply_to_pipeline_btn = gr.Button(
        "✅ Apply Output Settings to All Upscalers",
        variant="primary",
        size="lg"
    )
    apply_status = gr.Markdown("")

    # Preset management
    with gr.Accordion("💾 Preset Management", open=True):
        gr.Markdown("#### Save/Load Output Settings")

        preset_dropdown = gr.Dropdown(
            label="Output Presets",
            choices=preset_manager.list_presets("output", current_model),
            value=last_used_name or "",
        )

        with gr.Row():
            preset_name = gr.Textbox(
                label="Preset Name",
                placeholder="my_output_preset"
            )
            save_preset_btn = gr.Button("💾 Save Preset", variant="secondary")

        with gr.Row():
            load_preset_btn = gr.Button("📂 Load Preset")
            safe_defaults_btn = gr.Button("🔄 Safe Defaults")

        preset_status = gr.Markdown("")

    # Cache management buttons
    with gr.Accordion("🔄 Cache Management", open=False):
        gr.Markdown("#### Update Cached Values")

        with gr.Row():
            cache_fps_btn = gr.Button("📹 Cache FPS", size="sm")
            cache_comparison_btn = gr.Button("🔍 Cache Comparison", size="sm")

        with gr.Row():
            cache_png_btn = gr.Button("🖼️ Cache PNG Settings", size="sm")
            cache_skip_btn = gr.Button("⏭️ Cache Skip/Cap", size="sm")

        cache_status = gr.Markdown("")

    # Collect inputs for callbacks
    inputs_list = [
        output_format, png_sequence_enabled, png_padding, png_keep_basename,
        fps_override, video_codec, video_quality, video_preset, two_pass_encoding,
        skip_first_frames, load_cap, pixel_format, audio_codec, audio_bitrate,
        temporal_padding, frame_interpolation, comparison_mode, pin_reference,
        fullscreen_enabled, comparison_zoom, show_difference,
        save_metadata, metadata_format, telemetry_enabled, log_level
    ]

    # Wire up callbacks
    def refresh_presets(model):
        presets = preset_manager.list_presets("output", model)
        return gr.update(choices=presets, value="")

    model_selector.change(
        fn=refresh_presets,
        inputs=model_selector,
        outputs=preset_dropdown
    )

    save_preset_btn.click(
        fn=lambda name, model, *vals: service["save_preset"](name, model, list(vals)),
        inputs=[preset_name, model_selector] + inputs_list,
        outputs=[preset_dropdown, preset_status] + inputs_list  # FIXED: Capture reapplied validated values
    )

    load_preset_btn.click(
        fn=lambda preset, model, *vals: service["load_preset"](preset, model, list(vals)),
        inputs=[preset_dropdown, model_selector] + inputs_list,
        outputs=inputs_list + [preset_status]  # Include all controls including output_format
    )

    safe_defaults_btn.click(
        fn=lambda: service["safe_defaults"](),
        outputs=inputs_list  # Reset all controls including output_format
    )

    # Apply to pipeline
    apply_to_pipeline_btn.click(
        fn=lambda *args: service["apply_to_pipeline"](*args),
        inputs=inputs_list + [shared_state],
        outputs=[apply_status, shared_state]
    )

    # Pin reference handlers
    def pin_current_output(state):
        """Pin the current output as reference"""
        # Get last output path from state
        last_output = state.get("seed_controls", {}).get("last_output_path", "")
        return service["pin_reference_frame"](last_output, state)
    
    pin_btn.click(
        fn=pin_current_output,
        inputs=shared_state,
        outputs=[pin_status, shared_state]
    )
    
    unpin_btn.click(
        fn=lambda state: service["unpin_reference"](state),
        inputs=shared_state,
        outputs=[pin_status, shared_state]
    )

    # Cache management - wire individual values correctly
    def cache_fps_wrapper(fps_val, state):
        return service["cache_fps"](fps_val, state)
    
    def cache_comparison_wrapper(comp_mode, state):
        return service["cache_comparison"](comp_mode, state)
    
    def cache_png_wrapper(padding, basename, state):
        msg1, state1 = service["cache_png_padding"](padding, state)
        msg2, state2 = service["cache_png_basename"](basename, state1)
        return gr.update(value=msg1 + "\n" + msg2), state2
    
    def cache_skip_wrapper(skip_val, cap_val, state):
        msg1, state1 = service["cache_skip"](skip_val, state)
        msg2, state2 = service["cache_cap"](cap_val, state1)
        return gr.update(value=msg1 + "\n" + msg2), state2
    
    cache_fps_btn.click(
        fn=cache_fps_wrapper,
        inputs=[fps_override, shared_state],
        outputs=[cache_status, shared_state]
    )

    cache_comparison_btn.click(
        fn=cache_comparison_wrapper,
        inputs=[comparison_mode, shared_state],
        outputs=[cache_status, shared_state]
    )

    cache_png_btn.click(
        fn=cache_png_wrapper,
        inputs=[png_padding, png_keep_basename, shared_state],
        outputs=[cache_status, shared_state]
    )

    cache_skip_btn.click(
        fn=cache_skip_wrapper,
        inputs=[skip_first_frames, load_cap, shared_state],
        outputs=[cache_status, shared_state]
    )
    
    # Codec info updates
    video_codec.change(
        fn=service["update_codec_info"],
        inputs=video_codec,
        outputs=codec_info_display
    )
    
    pixel_format.change(
        fn=service["update_pixel_format_info"],
        inputs=pixel_format,
        outputs=pixel_format_info
    )
    
    # Quick codec preset buttons
    preset_youtube.click(
        fn=lambda *vals: service["apply_codec_preset"]("youtube", list(vals)),
        inputs=inputs_list,
        outputs=inputs_list
    )
    
    preset_archival.click(
        fn=lambda *vals: service["apply_codec_preset"]("archival", list(vals)),
        inputs=inputs_list,
        outputs=inputs_list
    )
    
    preset_editing.click(
        fn=lambda *vals: service["apply_codec_preset"]("editing", list(vals)),
        inputs=inputs_list,
        outputs=inputs_list
    )
    
    preset_web.click(
        fn=lambda *vals: service["apply_codec_preset"]("web", list(vals)),
        inputs=inputs_list,
        outputs=inputs_list
    )
