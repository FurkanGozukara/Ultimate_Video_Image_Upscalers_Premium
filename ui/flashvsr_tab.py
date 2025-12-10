"""
FlashVSR+ Tab - Self-contained modular implementation
Real-time diffusion-based streaming video super-resolution
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.flashvsr_service import (
    build_flashvsr_callbacks, FLASHVSR_ORDER
)


def flashvsr_tab(
    preset_manager,
    run_logger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path
):
    """
    Self-contained FlashVSR+ tab following SECourses modular pattern.
    """

    # Build service callbacks
    service = build_flashvsr_callbacks(
        preset_manager, run_logger, global_settings, shared_state,
        base_dir, temp_dir, output_dir
    )

    # Get defaults and last used
    defaults = service["defaults"]
    last_used_name = preset_manager.get_last_used_name("flashvsr", "v10_tiny")
    last_used = preset_manager.load_last_used("flashvsr", "v10_tiny")

    if last_used_name and last_used is None:
        def update_warning(state):
            existing = state["health_banner"]["text"]
            warning = f"⚠️ Last used FlashVSR+ preset '{last_used_name}' not found; loaded defaults."
            state["health_banner"]["text"] = existing + "\n" + warning if existing else warning
            return state
        shared_state.value = update_warning(shared_state.value)

    merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    values = [merged_defaults[k] for k in FLASHVSR_ORDER]

    # Layout
    gr.Markdown("### ⚡ FlashVSR+ - Real-Time Diffusion Video Super-Resolution")
    gr.Markdown("*High-quality real-time video upscaling with diffusion models*")

    with gr.Row():
        # Left Column: Input & Settings
        with gr.Column(scale=3):
            gr.Markdown("#### 📁 Input")
            
            input_file = gr.File(
                label="Upload Video or Image Folder",
                type="filepath",
                file_types=["video"],
                info="Select video file or folder containing image sequence"
            )
            input_path = gr.Textbox(
                label="Input Path",
                value=values[0],
                placeholder="C:/path/to/video.mp4 or C:/path/to/frames/",
                info="Video file or image sequence folder"
            )
            
            # Model Configuration
            gr.Markdown("#### 🤖 Model Configuration")
            
            with gr.Row():
                scale = gr.Dropdown(
                    label="Upscale Factor",
                    choices=["2", "4"],
                    value=str(values[2]),
                    info="2x or 4x upscaling"
                )
                version = gr.Dropdown(
                    label="Model Version",
                    choices=["10", "11"],
                    value=values[3],
                    info="v10 = faster, v11 = higher quality"
                )
                mode = gr.Dropdown(
                    label="Pipeline Mode",
                    choices=["tiny", "tiny-long", "full"],
                    value=values[4],
                    info="tiny = fastest, full = best quality, tiny-long = balanced"
                )
            
            # Processing Settings
            gr.Markdown("#### ⚙️ Processing Settings")
            
            with gr.Group():
                dtype = gr.Dropdown(
                    label="Precision",
                    choices=["fp16", "bf16"],
                    value=values[12],
                    info="bf16 = faster, more stable. fp16 = broader compatibility"
                )
                
                device = gr.Textbox(
                    label="Device",
                    value=values[13],
                    placeholder="auto, cuda:0, cpu",
                    info="auto = automatic selection, cuda:0 = specific GPU, cpu = CPU mode"
                )
                
                seed = gr.Number(
                    label="Random Seed",
                    value=values[11],
                    precision=0,
                    info="Seed for reproducibility. 0 = random"
                )
                
                attention = gr.Dropdown(
                    label="Attention Mode",
                    choices=["sage", "block"],
                    value=values[16],
                    info="sage = default, block = alternative implementation"
                )
            
            # Memory Optimization
            gr.Markdown("#### 💾 Memory Optimization (Tiling)")
            
            with gr.Group():
                tiled_vae = gr.Checkbox(
                    label="Enable VAE Tiling",
                    value=values[5],
                    info="Reduce VRAM usage during VAE encoding/decoding. Essential for high resolutions."
                )
                
                tiled_dit = gr.Checkbox(
                    label="Enable DiT Tiling",
                    value=values[6],
                    info="Reduce VRAM usage during diffusion inference. Enables processing larger videos."
                )
                
                tile_size = gr.Slider(
                    label="Tile Size",
                    minimum=128, maximum=512, step=32,
                    value=values[7],
                    info="Size of each tile. Larger = faster but more VRAM"
                )
                
                overlap = gr.Slider(
                    label="Tile Overlap",
                    minimum=8, maximum=64, step=8,
                    value=values[8],
                    info="Overlap between tiles to reduce seams. Higher = smoother"
                )
                
                unload_dit = gr.Checkbox(
                    label="Unload DiT Before Decoding",
                    value=values[9],
                    info="Free VRAM before VAE decoding. Slower but uses less memory."
                )
            
            # Quality Settings
            gr.Markdown("#### 🎨 Quality Settings")
            
            with gr.Group():
                color_fix = gr.Checkbox(
                    label="Color Correction",
                    value=values[10],
                    info="Maintain color accuracy. Recommended ON."
                )
                
                fps_flashvsr = gr.Number(
                    label="Output FPS (image sequences only)",
                    value=values[14],
                    precision=0,
                    info="Frame rate for image sequence outputs. Ignored for video inputs."
                )
                
                quality = gr.Slider(
                    label="Video Quality",
                    minimum=1, maximum=10, step=1,
                    value=values[15],
                    info="Output quality. 1 = lowest, 10 = highest. 6 is recommended."
                )
        
        # Right Column: Output & Controls
        with gr.Column(scale=2):
            gr.Markdown("#### 🎯 Output & Actions")
            
            status_box = gr.Markdown(value="Ready.")
            progress_indicator = gr.Markdown(value="", visible=True)
            
            log_box = gr.Textbox(
                label="📋 Processing Log",
                value="",
                lines=12,
                show_copy_button=True
            )
            
            # Output displays
            output_video = gr.Video(
                label="🎬 Upscaled Video",
                interactive=False,
                show_download_button=True
            )
            output_image = gr.Image(
                label="🖼️ Output Image",
                interactive=False,
                show_download_button=True
            )
            
            # Comparison
            image_slider = gr.ImageSlider(
                label="🔍 Comparison",
                interactive=False,
                height=500
            )
            
            video_comparison_html = gr.HTML(
                label="🎬 Video Comparison",
                value="",
                visible=False
            )
            
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
            
            cancel_confirm = gr.Checkbox(
                label="⚠️ Confirm cancel (required for safety)",
                value=False,
                info="Enable this checkbox to confirm cancellation of processing"
            )
            
            # Utility buttons
            with gr.Row():
                open_outputs_btn = gr.Button("📂 Open Outputs")
                clear_temp_btn = gr.Button("🗑️ Clear Temp")
            
            # Preset Management
            with gr.Accordion("💾 Preset Management", open=True):
                preset_dropdown = gr.Dropdown(
                    label="FlashVSR+ Presets",
                    choices=preset_manager.list_presets("flashvsr", "v10_tiny"),
                    value=last_used_name or ""
                )
                
                with gr.Row():
                    preset_name = gr.Textbox(
                        label="Preset Name",
                        placeholder="my_flashvsr_preset"
                    )
                    save_preset_btn = gr.Button("💾 Save", variant="secondary")
                
                with gr.Row():
                    load_preset_btn = gr.Button("📂 Load")
                    safe_defaults_btn = gr.Button("🔄 Reset to Defaults")
                
                preset_status = gr.Markdown("")
            
            # Info
            gr.Markdown("""
            #### ℹ️ About FlashVSR+
            
            **Real-time Diffusion Video SR:**
            - Streaming processing for memory efficiency
            - Multiple pipeline modes for speed/quality tradeoff
            - Automatic model download from HuggingFace
            
            **Recommended Settings:**
            - Mode: `tiny` for real-time, `full` for best quality
            - Enable tiling for high-res or limited VRAM
            - Use color fix for accurate colors
            """)
    
    # Batch processing (optional)
    with gr.Accordion("📦 Batch Processing", open=False):
        batch_enable = gr.Checkbox(
            label="Enable Batch",
            value=values[17],
            info="Process multiple files"
        )
        batch_input = gr.Textbox(
            label="Batch Input Folder",
            value=values[18],
            placeholder="Folder with videos"
        )
        batch_output = gr.Textbox(
            label="Batch Output Folder",
            value=values[19],
            placeholder="Output directory"
        )
    
    # Collect inputs
    inputs_list = [
        input_path, gr.State(values[1]), scale, version, mode,
        tiled_vae, tiled_dit, tile_size, overlap, unload_dit,
        color_fix, seed, dtype, device, fps_flashvsr,
        quality, attention, batch_enable, batch_input, batch_output
    ]
    
    # Wire up events
    def cache_input(val, state):
        return val or "", gr.Markdown.update(value="✅ Input cached", visible=True), state
    
    input_file.upload(
        fn=cache_input,
        inputs=[input_file, shared_state],
        outputs=[input_path, preset_status, shared_state]
    )
    
    # Main processing
    upscale_btn.click(
        fn=lambda *args, progress=gr.Progress(): service["run_action"](args[0], *args[1:-1], preview_only=False, state=args[-1], progress=progress),
        inputs=[input_file] + inputs_list + [shared_state],
        outputs=[status_box, log_box, output_video, output_image, image_slider, video_comparison_html, shared_state]
    )
    
    cancel_btn.click(
        fn=lambda ok: service["cancel_action"]() if ok else (gr.Markdown.update(value="⚠️ Enable 'Confirm cancel' to stop."), ""),
        inputs=[cancel_confirm],
        outputs=[status_box, log_box]
    )
    
    open_outputs_btn.click(
        fn=service["open_outputs_folder"],
        outputs=status_box
    )
    
    # Preset management
    save_preset_btn.click(
        fn=lambda name, *vals: service["save_preset"](name, *vals),
        inputs=[preset_name] + inputs_list,
        outputs=[preset_dropdown, preset_status] + inputs_list
    )
    
    load_preset_btn.click(
        fn=lambda preset, ver, mod, *vals: service["load_preset"](preset, ver, mod, list(vals)),
        inputs=[preset_dropdown, version, mode] + inputs_list,
        outputs=inputs_list
    )
    
    safe_defaults_btn.click(
        fn=service["safe_defaults"],
        outputs=inputs_list
    )

