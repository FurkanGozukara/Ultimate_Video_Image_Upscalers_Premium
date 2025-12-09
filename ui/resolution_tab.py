"""
Resolution & Scene Split Tab - Complete Implementation with Auto-Calculation
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.resolution_service import (
    build_resolution_callbacks, RESOLUTION_ORDER
)
from shared.models import (
    get_seedvr2_model_names,
    scan_gan_models,
    get_flashvsr_model_names,
    get_rife_model_names
)


def resolution_tab(preset_manager, shared_state: gr.State, base_dir: Path):
    """
    Self-contained Resolution & Scene Split tab with auto-calculation features.
    
    This tab's settings apply to ALL upscaler models (SeedVR2, GAN, FlashVSR+, RIFE).
    Settings are saved per-model for flexibility.
    """

    # Get available models from ALL pipelines
    seedvr2_models = get_seedvr2_model_names()
    gan_models = scan_gan_models(base_dir)
    flashvsr_models = get_flashvsr_model_names()
    rife_models = get_rife_model_names(base_dir)
    
    # Combine and deduplicate
    combined_models = sorted(list({
        *seedvr2_models, 
        *gan_models,
        *flashvsr_models,
        *rife_models
    }))

    # Build service callbacks
    service = build_resolution_callbacks(preset_manager, shared_state, combined_models)

    # Get defaults and last used
    current_model = combined_models[0] if combined_models else "default"
    defaults = service["defaults"]
    last_used_name = preset_manager.get_last_used_name("resolution", current_model)
    last_used = preset_manager.load_last_used("resolution", current_model)
    
    if last_used_name and last_used is None:
        def update_warning(state):
            existing = state["health_banner"]["text"]
            warning = f"⚠️ Last used Resolution preset '{last_used_name}' not found; loaded defaults."
            if existing:
                state["health_banner"]["text"] = existing + "\n" + warning
            else:
                state["health_banner"]["text"] = warning
            return state
        shared_state.value = update_warning(shared_state.value)

    merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    values = [merged_defaults[k] for k in RESOLUTION_ORDER]

    # Header
    gr.Markdown("### 📐 Resolution & Scene Split Settings")
    gr.Markdown("""
    *Configure resolution, aspect ratio handling, and **universal PySceneDetect chunking** for all upscaler models*
    
    **🎬 PySceneDetect Chunking**: Intelligent scene-based video splitting that works with **ALL models** (SeedVR2, GAN, RIFE, FlashVSR+).
    This is the **PREFERRED chunking method** for long videos, managing VRAM, and optimizing quality per scene.
    """)

    # Model selector
    model_selector = gr.Dropdown(
        label="Model Context (for presets)",
        choices=combined_models,
        value=values[0],
        info="Settings are saved/loaded per model"
    )

    with gr.Row():
        # Left Column: Settings
        with gr.Column(scale=2):
            gr.Markdown("#### 🎯 Resolution Settings")
            
            with gr.Group():
                auto_resolution = gr.Checkbox(
                    label="🔄 Auto-Resolution (Aspect Ratio Aware)",
                    value=values[1],
                    info="Automatically calculate optimal resolution maintaining aspect ratio"
                )

                enable_max_target = gr.Checkbox(
                    label="🎯 Enable Max Target Resolution",
                    value=values[2],
                    info="Apply maximum resolution cap to prevent excessive upscaling"
                )

                target_resolution = gr.Slider(
                    label="Target Resolution (short side)",
                    minimum=256, maximum=4096, step=16,
                    value=values[3],
                    info="Target for shortest side (e.g., 1080 for 1080p)"
                )

                max_target_resolution = gr.Slider(
                    label="Max Target Resolution",
                    minimum=0, maximum=8192, step=16,
                    value=values[4],
                    info="Maximum allowed resolution (0 = unlimited)"
                )

                ratio_downscale_then_upscale = gr.Checkbox(
                    label="🔀 Enable Downscale-Then-Upscale (for fixed-scale GAN models)",
                    value=values[7],
                    info="For 2x/4x models: downscale input first to reach arbitrary target resolutions"
                )

            gr.Markdown("#### 🎬 Scene Splitting & Chunking")
            
            with gr.Group():
                chunk_size = gr.Slider(
                    label="Chunk Size (seconds, 0=disabled)",
                    minimum=0, maximum=600, step=10,
                    value=values[5],
                    info="Split long videos into chunks. 0=off, 60=1min chunks, 300=5min chunks"
                )

                chunk_overlap = gr.Slider(
                    label="Chunk Overlap (seconds)",
                    minimum=0.0, maximum=5.0, step=0.1,
                    value=values[6],
                    info="Overlap between chunks for smooth transitions. 0.5-2s recommended"
                )

                per_chunk_cleanup = gr.Checkbox(
                    label="Clean Temp Files Per Chunk",
                    value=values[8],
                    info="Delete temporary files after each chunk (saves disk space)"
                )

        # Right Column: Auto-Calculation & Preview
        with gr.Column(scale=1):
            gr.Markdown("#### 🔍 Auto-Calculate & Preview")
            
            # Input path for calculation
            calc_input_path = gr.Textbox(
                label="Input Path (for estimation)",
                placeholder="Paste input video/image path or use SeedVR2 tab input",
                info="Path to calculate resolution for"
            )
            
            calc_model_scale = gr.Number(
                label="Model Scale (2, 4, etc. - leave 0 for auto)",
                value=0,
                precision=0,
                info="For GAN models with fixed scale. 0=auto-detect, 2=2x model, 4=4x model"
            )
            
            with gr.Row():
                calc_resolution_btn = gr.Button("🎯 Calculate Resolution", variant="primary")
                calc_chunks_btn = gr.Button("📊 Estimate Chunks", variant="secondary")
            
            # Results display
            calc_result = gr.Markdown("", visible=False)
            
            # Disk space warning
            disk_space_warning = gr.Markdown("", visible=False)
            
            # Quick actions
            gr.Markdown("#### ⚡ Quick Actions")
            
            with gr.Row():
                use_seedvr2_input_btn = gr.Button("📥 Use SeedVR2 Input", size="sm")
                refresh_calc_btn = gr.Button("🔄 Refresh", size="sm")

    # Preset management
    with gr.Accordion("💾 Preset Management", open=True):
        gr.Markdown("#### Save/Load Resolution Presets")
        
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

    # Apply to pipeline
    gr.Markdown("#### 🔗 Apply to Pipeline")
    apply_to_seed_btn = gr.Button(
        "✅ Apply Resolution Settings to All Upscalers",
        variant="primary",
        size="lg"
    )
    apply_status = gr.Markdown("")

    # Collect inputs
    inputs_list = [
        model_selector, auto_resolution, enable_max_target, target_resolution,
        max_target_resolution, chunk_size, chunk_overlap, ratio_downscale_then_upscale,
        per_chunk_cleanup
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
        fn=lambda name, *vals: service["save_preset"](name, *vals),
        inputs=[preset_name] + inputs_list,
        outputs=[preset_dropdown, preset_status] + inputs_list
    )

    load_preset_btn.click(
        fn=lambda preset, model, *vals: service["load_preset"](preset, model, list(vals)),
        inputs=[preset_dropdown, model_selector] + inputs_list,
        outputs=inputs_list
    )

    safe_defaults_btn.click(
        fn=service["safe_defaults"],
        outputs=inputs_list
    )

    apply_to_seed_btn.click(
        fn=lambda *args: service["apply_to_seed"](*args),
        inputs=inputs_list + [shared_state],
        outputs=[apply_status, shared_state]
    )

    # Auto-calculation callbacks
    def calculate_resolution_wrapper(input_path, target_res, max_res, enable_max, auto_mode, 
                                    ratio_aware, model_scale, state):
        """Wrapper for auto-resolution calculation"""
        if not input_path or not input_path.strip():
            # Try to get from shared state
            input_path = state.get("seed_controls", {}).get("last_input_path", "")
            if not input_path:
                return "⚠️ No input path provided. Upload in SeedVR2 tab or enter path above.", state, gr.Markdown.update(visible=False)
        
        # Call service function
        info, updated_state = service["calculate_auto_resolution"](
            input_path, 
            int(target_res), 
            int(max_res),
            enable_max,
            auto_mode,
            ratio_aware,
            int(model_scale) if model_scale > 0 else None,
            state
        )
        
        return gr.Markdown.update(value=info, visible=True), updated_state, gr.Markdown.update(visible=False)

    def calculate_chunks_wrapper(input_path, chunk_size, chunk_overlap, state):
        """Wrapper for chunk estimation"""
        if not input_path or not input_path.strip():
            input_path = state.get("seed_controls", {}).get("last_input_path", "")
            if not input_path:
                return "⚠️ No input path provided", state
        
        info, updated_state = service["calculate_chunk_estimate"](
            input_path,
            float(chunk_size),
            float(chunk_overlap),
            state
        )
        
        # Check for disk space warnings in the info
        if "⚠️" in info and "disk space" in info.lower():
            disk_warning = "⚠️ **DISK SPACE WARNING**: Insufficient space detected!"
            return gr.Markdown.update(value=info, visible=True), updated_state, gr.Markdown.update(value=disk_warning, visible=True)
        
        return gr.Markdown.update(value=info, visible=True), updated_state, gr.Markdown.update(visible=False)

    calc_resolution_btn.click(
        fn=calculate_resolution_wrapper,
        inputs=[calc_input_path, target_resolution, max_target_resolution, 
                enable_max_target, auto_resolution, ratio_downscale_then_upscale, 
                calc_model_scale, shared_state],
        outputs=[calc_result, shared_state, disk_space_warning]
    )

    calc_chunks_btn.click(
        fn=calculate_chunks_wrapper,
        inputs=[calc_input_path, chunk_size, chunk_overlap, shared_state],
        outputs=[calc_result, shared_state, disk_space_warning]
    )

    # Use SeedVR2 input button
    def use_seedvr2_input(state):
        input_path = state.get("seed_controls", {}).get("last_input_path", "")
        if input_path:
            return gr.Textbox.update(value=input_path), gr.Markdown.update(value=f"✅ Using input from SeedVR2 tab: {input_path}", visible=True)
        else:
            return gr.Textbox.update(), gr.Markdown.update(value="⚠️ No input set in SeedVR2 tab yet", visible=True)

    use_seedvr2_input_btn.click(
        fn=use_seedvr2_input,
        inputs=shared_state,
        outputs=[calc_input_path, calc_result]
    )

    # Auto-update calculation when settings change
    for component in [target_resolution, max_target_resolution, auto_resolution, enable_max_target, ratio_downscale_then_upscale]:
        component.change(
            fn=lambda *args: gr.Markdown.update(value="ℹ️ Settings changed. Click 'Calculate Resolution' to update.", visible=True),
            outputs=calc_result
        )
