"""
Resolution & Scene Split Tab - Self-contained modular implementation
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.resolution_service import (
    build_resolution_callbacks, RESOLUTION_ORDER
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


def resolution_tab(preset_manager, shared_state: gr.State, base_dir: Path):
    """
    Self-contained Resolution & Scene Split tab.
    Handles resolution settings shared across all upscaler models.
    """

    # Get available models from both SeedVR2 and GAN
    seedvr2_models = get_seedvr2_model_names()
    gan_models = _get_gan_model_names(base_dir)
    combined_models = sorted(list({*seedvr2_models, *gan_models}))

    # Build service callbacks
    service = build_resolution_callbacks(preset_manager, shared_state, combined_models)

    # Get defaults and last used for current model (use first available)
    current_model = combined_models[0] if combined_models else "default"
    defaults = service["defaults"]
    last_used_name = preset_manager.get_last_used_name("resolution", current_model)
    last_used = preset_manager.load_last_used("resolution", current_model)
    if last_used_name and last_used is None:
        def update_warning(state):
            existing = state["health_banner"]["text"]
            warning = f"Last used Resolution preset '{last_used_name}' not found; loaded defaults."
            if existing:
                state["health_banner"]["text"] = existing + "\n" + warning
            else:
                state["health_banner"]["text"] = warning
            return state
        shared_state.value = update_warning(shared_state.value)

    merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    values = [merged_defaults[k] for k in RESOLUTION_ORDER]

    # Layout
    gr.Markdown("### 📐 Resolution & Scene Split Settings")
    gr.Markdown("*Configure resolution, aspect ratio handling, and scene-based chunking shared across all upscaler models*")

    # Model selector for presets
    model_selector = gr.Dropdown(
        label="Model Context (for presets)",
        choices=combined_models,
        value=current_model,
        info="Settings are saved/loaded per model type"
    )

    with gr.Tabs():
        # Resolution Settings Tab
        with gr.TabItem("🎯 Resolution Settings"):
            gr.Markdown("#### Target Resolution Configuration")

            with gr.Group():
                enable_max_target = gr.Checkbox(
                    label="🎯 Enable Max Target Resolution",
                    value=values[1],
                    info="Automatically calculate optimal resolution based on input and constraints"
                )

                auto_resolution = gr.Checkbox(
                    label="🔄 Auto-Resolution (Aspect Ratio Aware)",
                    value=values[2],
                    info="Automatically adjust resolution to maintain aspect ratio"
                )

                with gr.Row():
                    target_width = gr.Number(
                        label="Target Width",
                        value=values[3],
                        precision=0,
                        info="Desired output width in pixels"
                    )
                    target_height = gr.Number(
                        label="Target Height",
                        value=values[4],
                        precision=0,
                        info="Desired output height in pixels"
                    )

                max_resolution = gr.Slider(
                    label="Maximum Resolution Cap",
                    minimum=0, maximum=8192, step=16,
                    value=values[5],
                    info="0 = no limit, otherwise caps the maximum dimension"
                )

                ratio_downscale = gr.Checkbox(
                    label="🔀 Ratio Upscale → Downscale → Upscale",
                    value=values[6],
                    info="For fixed-ratio models: upscale to target, then downscale to fit constraints, then final upscale"
                )

        # Scene Split Settings Tab
        with gr.TabItem("🎬 Scene Detection & Chunking"):
            gr.Markdown("#### PySceneDetect Configuration")

            with gr.Group():
                scene_detection_enabled = gr.Checkbox(
                    label="Enable Scene Detection",
                    value=values[7],
                    info="Split videos into scenes for processing"
                )

                scene_threshold = gr.Slider(
                    label="Content Threshold",
                    minimum=1, maximum=100, step=1,
                    value=values[8],
                    info="Sensitivity for scene change detection (higher = fewer scenes)"
                )

                min_scene_length = gr.Slider(
                    label="Minimum Scene Length (seconds)",
                    minimum=0.5, maximum=30.0, step=0.5,
                    value=values[9],
                    info="Minimum duration for a valid scene"
                )

                scene_overlap = gr.Slider(
                    label="Scene Overlap (seconds)",
                    minimum=0.0, maximum=5.0, step=0.1,
                    value=values[10],
                    info="Overlap between consecutive scenes to avoid artifacts"
                )

                fade_detection = gr.Checkbox(
                    label="Enable Fade Detection",
                    value=values[11],
                    info="Detect and handle fade transitions"
                )

        # Advanced Settings Tab
        with gr.TabItem("⚙️ Advanced Settings"):
            gr.Markdown("#### Processing Optimizations")

            with gr.Group():
                chunk_size_preference = gr.Dropdown(
                    label="Preferred Chunk Size Strategy",
                    choices=["auto", "fixed_duration", "fixed_scenes", "memory_based"],
                    value=values[12],
                    info="How to determine optimal chunk sizes"
                )

                max_chunk_duration = gr.Slider(
                    label="Maximum Chunk Duration (seconds)",
                    minimum=10, maximum=600, step=10,
                    value=values[13],
                    info="Longest allowed chunk duration"
                )

                memory_aware_chunking = gr.Checkbox(
                    label="Memory-Aware Chunking",
                    value=values[14],
                    info="Adjust chunk sizes based on available VRAM"
                )

                parallel_processing = gr.Checkbox(
                    label="Enable Parallel Chunk Processing",
                    value=values[15],
                    info="Process multiple chunks simultaneously (uses more resources)"
                )

                per_chunk_cleanup = gr.Checkbox(
                    label="Clean Temp Files Per Chunk",
                    value=values[16],
                    info="Delete intermediate files immediately after each chunk (saves disk space)"
                )

    # Estimation and preview
    with gr.Accordion("📊 Chunk Estimation", open=False):
        gr.Markdown("#### Estimate Processing Requirements")

        with gr.Row():
            estimate_input = gr.Textbox(
                label="Input Path (for estimation)",
                placeholder="Path to video file",
                info="Enter a video path to estimate chunk count and processing time"
            )
            estimate_btn = gr.Button("🔍 Estimate Chunks", size="sm")

        estimation_result = gr.Markdown("", visible=False)

        estimate_btn.click(
            fn=lambda path, *args: service["estimate_from_input"](path, list(args)),
            inputs=[estimate_input] + [enable_max_target, auto_resolution, target_width, target_height,
                                     max_resolution, ratio_downscale, scene_detection_enabled, scene_threshold,
                                     min_scene_length, scene_overlap, fade_detection],
            outputs=estimation_result
        )

    # Preset management
    with gr.Accordion("💾 Preset Management", open=True):
        gr.Markdown("#### Save/Load Resolution Settings")

        preset_dropdown = gr.Dropdown(
            label="Resolution Presets",
            choices=preset_manager.list_presets("resolution", current_model),
            value=last_used_name or "",
        )

        with gr.Row():
            preset_name = gr.Textbox(
                label="Preset Name",
                placeholder="my_resolution_preset"
            )
            save_preset_btn = gr.Button("💾 Save Preset", variant="secondary")

        with gr.Row():
            load_preset_btn = gr.Button("📂 Load Preset")
            safe_defaults_btn = gr.Button("🔄 Safe Defaults")

        preset_status = gr.Markdown("")

    # Apply to SeedVR2 button
    gr.Markdown("#### 🔗 Integration")
    apply_to_seed_btn = gr.Button(
        "🔄 Apply Settings to SeedVR2 Tab",
        variant="primary",
        info="Copy current resolution settings to the SeedVR2 processing tab"
    )
    apply_status = gr.Markdown("")

    # Collect inputs for callbacks
    inputs_list = [
        model_selector, enable_max_target, auto_resolution, target_width, target_height,
        max_resolution, ratio_downscale, scene_detection_enabled, scene_threshold,
        min_scene_length, scene_overlap, fade_detection, chunk_size_preference,
        max_chunk_duration, memory_aware_chunking, parallel_processing, per_chunk_cleanup
    ]

    # Wire up callbacks
    def refresh_presets(model):
        presets = preset_manager.list_presets("resolution", model)
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

    apply_to_seed_btn.click(
        fn=lambda *args: service["apply_to_seed"](list(args)),
        inputs=inputs_list,
        outputs=[apply_status, shared_state]
    )

    # Update estimation result visibility
    def update_estimation_visibility(result):
        return gr.Markdown.update(value=result, visible=bool(result))

    estimation_result.update(update_estimation_visibility)
