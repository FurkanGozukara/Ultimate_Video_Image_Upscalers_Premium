"""
Output & Comparison Tab - Self-contained modular implementation
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.output_service import (
    build_output_callbacks, OUTPUT_ORDER
)
from shared.models.seedvr2_meta import get_seedvr2_model_names


def _get_gan_model_names(base_dir: Path) -> list:
    """Get GAN model names from Image_Upscale_Models folder"""
    models_dir = base_dir / "Image_Upscale_Models"
    if not models_dir.exists():
        return []
    choices = []
    for f in models_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".pth", ".safetensors"):
            choices.append(f.name)
    return sorted(choices)


def output_tab(preset_manager, shared_state: gr.State, base_dir: Path):
    """
    Self-contained Output & Comparison tab.
    Handles output format and comparison settings shared across models.
    """

    # Get available models
    seedvr2_models = get_seedvr2_model_names()
    gan_models = _get_gan_model_names(base_dir)
    combined_models = sorted(list({*seedvr2_models, *gan_models}))

    # Build service callbacks
    service = build_output_callbacks(preset_manager, shared_state, combined_models)

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
                    info="Digits in frame filenames (e.g., 0001.png, 00001.png)"
                )

                png_keep_basename = gr.Checkbox(
                    label="Keep Original Basename in PNG Names",
                    value=values[3],
                    info="Include original filename in PNG sequence names"
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

                video_codec = gr.Dropdown(
                    label="Video Codec",
                    choices=["libx264", "libx265", "libvpx-vp9", "auto"],
                    value=values[5],
                    info="'auto' selects based on quality preferences"
                )

                video_quality = gr.Slider(
                    label="Video Quality (CRF)",
                    minimum=0, maximum=51, step=1,
                    value=values[6],
                    info="Lower = higher quality, higher file size (libx264/libx265)"
                )

                video_preset = gr.Dropdown(
                    label="Encoding Preset",
                    choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow", "slower", "veryslow"],
                    value=values[7],
                    info="Encoding speed vs compression efficiency"
                )

                two_pass_encoding = gr.Checkbox(
                    label="Two-Pass Encoding",
                    value=values[8],
                    info="Slower but better quality for target file sizes"
                )

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
                    value=values[11],
                    precision=0,
                    info="Extra frames for temporal processing"
                )

                frame_interpolation = gr.Checkbox(
                    label="Enable Frame Interpolation",
                    value=values[12],
                    info="Smooth motion between frames"
                )

        # Comparison Settings
        with gr.TabItem("🔍 Comparison Display"):
            gr.Markdown("#### Comparison Viewer Configuration")

            with gr.Group():
                comparison_mode = gr.Dropdown(
                    label="Comparison Mode",
                    choices=["native", "slider", "side_by_side", "overlay"],
                    value=values[13],
                    info="How to display before/after comparison"
                )

                pin_reference = gr.Checkbox(
                    label="Pin Reference Image",
                    value=values[14],
                    info="Keep original as fixed reference when changing settings"
                )

                fullscreen_enabled = gr.Checkbox(
                    label="Enable Fullscreen Comparison",
                    value=values[15],
                    info="Allow fullscreen viewing of comparisons"
                )

                comparison_zoom = gr.Slider(
                    label="Default Zoom Level",
                    minimum=25, maximum=400, step=25,
                    value=values[16],
                    info="Default zoom percentage for comparison viewer"
                )

                show_difference = gr.Checkbox(
                    label="Show Difference Overlay",
                    value=values[17],
                    info="Highlight differences between original and upscaled"
                )

        # Metadata & Logging
        with gr.TabItem("📊 Metadata & Logging"):
            gr.Markdown("#### Output Metadata & Telemetry")

            with gr.Group():
                save_metadata = gr.Checkbox(
                    label="Save Processing Metadata",
                    value=values[18],
                    info="Embed processing info in output files"
                )

                metadata_format = gr.Dropdown(
                    label="Metadata Format",
                    choices=["json", "xml", "exif", "none"],
                    value=values[19],
                    info="Format for embedded metadata"
                )

                telemetry_enabled = gr.Checkbox(
                    label="Enable Run Telemetry",
                    value=values[20],
                    info="Log processing stats for troubleshooting"
                )

                log_level = gr.Dropdown(
                    label="Log Verbosity",
                    choices=["error", "warning", "info", "debug"],
                    value=values[21],
                    info="Detail level for processing logs"
                )

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
        model_selector, output_format, png_sequence_enabled, png_padding, png_keep_basename,
        fps_override, video_codec, video_quality, video_preset, two_pass_encoding,
        skip_first_frames, load_cap, temporal_padding, frame_interpolation,
        comparison_mode, pin_reference, fullscreen_enabled, comparison_zoom, show_difference,
        save_metadata, metadata_format, telemetry_enabled, log_level
    ]

    # Wire up callbacks
    def refresh_presets(model):
        presets = preset_manager.list_presets("output", model)
        return gr.Dropdown.update(choices=presets, value="")

    model_selector.change(
        fn=refresh_presets,
        inputs=model_selector,
        outputs=preset_dropdown
    )

    save_preset_btn.click(
        fn=lambda name, model, *vals: service["save_preset"](name, model, list(vals)),
        inputs=[preset_name, model_selector] + inputs_list,
        outputs=[preset_dropdown, preset_status]
    )

    load_preset_btn.click(
        fn=lambda preset, model, *vals: service["load_preset"](preset, model, list(vals)),
        inputs=[preset_dropdown, model_selector] + inputs_list,
        outputs=inputs_list[1:] + [preset_status]  # Skip model_selector
    )

    safe_defaults_btn.click(
        fn=lambda model: service["safe_defaults"](model),
        inputs=model_selector,
        outputs=inputs_list[1:]  # Skip model_selector
    )

    # Cache management
    cache_fps_btn.click(
        fn=lambda *vals: service["cache_fps"](list(vals)),
        inputs=inputs_list,
        outputs=[cache_status, shared_state]
    )

    cache_comparison_btn.click(
        fn=lambda *vals: service["cache_comparison"](list(vals)),
        inputs=inputs_list,
        outputs=[cache_status, shared_state]
    )

    cache_png_btn.click(
        fn=lambda *vals: service["cache_png_padding"](list(vals)) + "\n" + service["cache_png_basename"](list(vals)),
        inputs=inputs_list,
        outputs=cache_status
    )

    cache_skip_btn.click(
        fn=lambda *vals: service["cache_skip"](list(vals)) + "\n" + service["cache_cap"](list(vals)),
        inputs=inputs_list,
        outputs=cache_status
    )
