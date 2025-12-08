"""
SeedVR2 Tab - Self-contained modular implementation
Following SECourses_Musubi_Trainer pattern
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any

from shared.services.seedvr2_service import (
    seedvr2_defaults, SEEDVR2_ORDER, build_seedvr2_callbacks
)
from shared.models.seedvr2_meta import get_seedvr2_model_names
from ui.shared_components import preset_section


def seedvr2_tab(
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
    Self-contained SeedVR2 tab following SECourses modular pattern.
    All logic, callbacks, and state management internal to this function.
    """
    # Get defaults and last used preset
    defaults = seedvr2_defaults()
    last_used_name = preset_manager.get_last_used_name("seedvr2", defaults.get("dit_model"))
    last_used = preset_manager.load_last_used("seedvr2", defaults.get("dit_model"))

    # Handle last used preset loading with better error reporting
    if last_used_name:
        if last_used is None:
            # Update health banner with warning
            def update_warning(state):
                existing = state["health_banner"]["text"]
                warning = f"⚠️ Last used SeedVR2 preset '{last_used_name}' not found or corrupted; loaded defaults."
                if existing:
                    state["health_banner"]["text"] = existing + "\n" + warning
                else:
                    state["health_banner"]["text"] = warning
                return state
            shared_state.value = update_warning(shared_state.value)
        else:
            # Successfully loaded last used preset
            def update_success(state):
                existing = state["health_banner"]["text"]
                success_msg = f"✅ Loaded last used SeedVR2 preset: '{last_used_name}'"
                if existing:
                    state["health_banner"]["text"] = existing + "\n" + success_msg
                else:
                    state["health_banner"]["text"] = success_msg
                return state
            shared_state.value = update_success(shared_state.value)

    merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    values = [merged_defaults[k] for k in SEEDVR2_ORDER]

    # Build service callbacks
    service = build_seedvr2_callbacks(
        preset_manager, runner, run_logger, global_settings,
        shared_state, output_dir, temp_dir
    )

    # GPU hint and macOS detection
    import platform
    is_macos = platform.system() == "Darwin"

    try:
        import torch
        cuda_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if is_macos:
            gpu_hint = "macOS detected - CUDA device selection disabled (not supported by SeedVR2 CLI)"
        else:
            gpu_hint = f"Detected CUDA GPUs: {cuda_count}" if cuda_count else "CUDA not available"
    except Exception:
        gpu_hint = "CUDA detection failed"

    # Layout: Two-column design
    with gr.Row():
        # Left Column: Input Controls
        with gr.Column(scale=3):
            gr.Markdown("### 🎬 Input / Controls")

            # Input section with enhanced detection
            with gr.Group():
                gr.Markdown("#### 📁 Enhanced Input: Video Files & Frame Folders")
                gr.Markdown("*Auto-detects whether your input is a single video file or a folder containing frame sequences*")

                input_file = gr.File(
                    label="Upload video or image (optional)",
                    type="filepath",
                    file_types=["video", "image"]
                )
                input_path = gr.Textbox(
                    label="Input Video or Frames Folder Path",
                    value=values[0],
                    placeholder="C:/path/to/video.mp4 or C:/path/to/frames/",
                    info="Enter path to either a video file (mp4, avi, mov, etc.) or folder containing image frames (jpg, png, tiff, etc.). Automatically detected - works on Windows and Linux."
                )
                input_cache_msg = gr.Markdown("", visible=False)
                auto_res_msg = gr.Markdown("", visible=False)

            # Batch processing controls (above Last Processed Chunk as requested)
            with gr.Accordion("📦 Batch Processing", open=False):
                batch_enable = gr.Checkbox(
                    label="Enable Batch Processing (use directory input)",
                    value=values[5]
                )
                batch_input = gr.Textbox(
                    label="Batch Input Folder",
                    value=values[6],
                    placeholder="Folder containing videos or frames",
                    info="Process multiple files in batch mode"
                )
                batch_output = gr.Textbox(
                    label="Batch Output Folder Override",
                    value=values[7],
                    placeholder="Optional override for batch outputs"
                )

            # Scene splitting controls
            with gr.Accordion("🎬 Scene Split (PySceneDetect)", open=False):
                chunk_enable = gr.Checkbox(
                    label="Enable scene-based chunking",
                    value=values[8],
                    info="Split video into scenes and process separately. Prevents VRAM overflow on long videos. Automatically detects scene changes."
                )
                scene_threshold = gr.Slider(
                    label="Content Threshold",
                    minimum=5, maximum=50, step=1,
                    value=values[9],
                    info="Sensitivity for scene detection. Lower = more cuts (more sensitive). Higher = fewer cuts. Default 27 works well for most content."
                )
                scene_min_len = gr.Slider(
                    label="Min Scene Length (sec)",
                    minimum=1, maximum=20, step=1,
                    value=values[10],
                    info="Minimum duration for a scene. Prevents too many short chunks. 2-5 seconds recommended for smooth processing."
                )
                resume_chunking = gr.Checkbox(
                    label="Resume from partial chunks",
                    value=values[48],
                    info="Resume interrupted chunking from existing partial outputs. Useful for recovering from crashes or cancellations."
                )
                check_resume_btn = gr.Button("🔍 Check Resume Status", size="sm")

            # Output controls
            output_override = gr.Textbox(
                label="Output Override (single run)",
                value=values[1],
                placeholder="Leave empty for auto naming",
                info="Specify custom output path. Auto-naming creates '_upscaled' files in output folder. Supports both file paths and directories."
            )
            output_format = gr.Dropdown(
                label="Output Format",
                choices=["auto", "mp4", "png"],
                value=values[2],
                info="'auto' chooses based on input type. 'mp4' for video output. 'png' exports frame sequence. Note: MP4 drops alpha channels."
            )

            # Model selection
            model_dir = gr.Textbox(
                label="Model Directory (optional)",
                value=values[3],
                info="Override default model directory. Leave empty to use default ./SeedVR2/models location."
            )
            dit_model = gr.Dropdown(
                label="SeedVR2 Model",
                choices=get_seedvr2_model_names(),
                value=values[4],
                info="3B models are faster, 7B models higher quality. 'sharp' variants enhance edges. fp16 recommended for best speed/quality balance."
            )
            model_cache_msg = gr.Markdown("", visible=False)

            # Model loading status with periodic updates
            model_status = gr.Markdown("### 🔧 Model Status\nNo models loaded", elem_classes="model-status")
            
            # Timer for periodic model status updates
            model_status_timer = gr.Timer(value=2.0, active=False)  # Update every 2 seconds when active

            # Resolution controls with auto-resolution
            with gr.Row():
                resolution = gr.Slider(
                    label="Target Resolution (short side)",
                    minimum=256, maximum=4096, step=16,
                    value=values[11],
                    info="Target resolution for shortest side. 1080 = 1080p, 2160 = 4K. Higher = better quality but slower. Must be multiple of 16."
                )
                max_resolution = gr.Slider(
                    label="Max Resolution (0 = no cap)",
                    minimum=0, maximum=8192, step=16,
                    value=values[12],
                    info="Maximum resolution cap for safety. Prevents accidental 8K+ upscaling. 0 = unlimited. Set to 4096 for 4K max, 2160 for 1080p max."
                )

            # Core processing parameters
            batch_size = gr.Slider(
                label="Batch Size (must be 4n+1: 5, 9, 13, 17...)",
                minimum=5, maximum=201, step=4,
                value=values[13],
                info="SeedVR2 requires batch size to follow 4n+1 formula (5, 9, 13, 17, 21...)"
            )
            uniform_batch_size = gr.Checkbox(
                label="Uniform Batch Size",
                value=values[14],
                info="Force all batches to same size by padding. Improves compilation efficiency but may use more memory. Recommended ON with torch.compile."
            )
            seed = gr.Number(
                label="Seed",
                value=values[15],
                precision=0,
                info="Random seed for reproducible results. Same seed + settings = identical output. -1 or 0 = random. Try 42 for consistent testing."
            )

            # Frame controls
            with gr.Row():
                skip_first_frames = gr.Number(
                    label="Skip First Frames",
                    value=values[16],
                    precision=0,
                    info="Skip N frames from start of video. Useful to skip intros/logos. 0 = process from beginning."
                )
                load_cap = gr.Number(
                    label="Load Cap (0 = all)",
                    value=values[17],
                    precision=0,
                    info="Process only first N frames. Useful for quick tests. 0 = process entire video. Combine with skip for specific range."
                )
                prepend_frames = gr.Number(
                    label="Prepend Frames",
                    value=values[18],
                    precision=0,
                    info="Prepend N copies of first frame for temporal stability. Helps reduce artifacts at video start. Try 2-4."
                )
                temporal_overlap = gr.Number(
                    label="Temporal Overlap",
                    value=values[19],
                    precision=0,
                    info="Overlap frames between processing batches. Improves temporal consistency. Higher = smoother but slower. Try 1-3."
                )

            # Color correction
            color_correction = gr.Dropdown(
                label="Color Correction",
                choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                value=values[20],
                info="Method for maintaining color accuracy. 'lab' is default and robust. 'wavelet' preserves details better. 'none' for creative control."
            )

            # Noise controls
            input_noise_scale = gr.Slider(
                label="Input Noise Scale",
                minimum=0.0, maximum=1.0, step=0.01,
                value=values[21],
                info="Add noise to input before encoding. Can help with smooth gradients but may reduce sharpness. 0.0 = no noise. Try 0.0-0.1."
            )
            latent_noise_scale = gr.Slider(
                label="Latent Noise Scale",
                minimum=0.0, maximum=1.0, step=0.01,
                value=values[22],
                info="Add noise in latent space during diffusion. Can improve detail generation. 0.0 = no noise. Typical: 0.0-0.05."
            )

            # Device configuration
            gr.Markdown("#### 💻 Device & Offload")
            cuda_device = gr.Textbox(
                label="CUDA Devices (e.g., 0 or 0,1,2)",
                value=values[23] if not is_macos else "",
                info=gpu_hint,
                visible=not is_macos
            )
            dit_offload_device = gr.Textbox(
                label="DiT Offload Device",
                value=values[24],
                placeholder="none / cpu / GPU id",
                info="Where to offload DiT model when not in use. 'cpu' saves VRAM, 'none' keeps on GPU. Required for BlockSwap."
            )
            vae_offload_device = gr.Textbox(
                label="VAE Offload Device",
                value=values[25],
                placeholder="none / cpu / GPU id",
                info="Where to offload VAE model when not in use. 'cpu' saves VRAM, 'none' keeps on GPU for faster processing."
            )
            tensor_offload_device = gr.Textbox(
                label="Tensor Offload Device",
                value=values[26],
                placeholder="cpu / none / GPU id",
                info="Where to offload intermediate tensors. 'cpu' is recommended for memory management between processing phases."
            )

            # BlockSwap configuration
            gr.Markdown("#### 🔄 BlockSwap")
            blocks_to_swap = gr.Slider(
                label="Blocks to Swap",
                minimum=0, maximum=36, step=1,
                value=values[27],
                info="Number of DiT blocks to swap to CPU. Higher values save more VRAM but slow processing. Try 20-30 for 8GB GPUs."
            )
            swap_io_components = gr.Checkbox(
                label="Swap I/O Components",
                value=values[28],
                info="Swap input/output layers to CPU. Enable for maximum VRAM savings on limited GPUs. Requires DiT offload to CPU."
            )

            # VAE Tiling
            gr.Markdown("#### 🧩 VAE Tiling")
            vae_encode_tiled = gr.Checkbox(
                label="VAE Encode Tiled",
                value=values[29],
                info="Process VAE encoding in tiles to reduce VRAM usage. Essential for 4K+ resolutions on GPUs with <16GB VRAM."
            )
            vae_encode_tile_size = gr.Number(
                label="Encode Tile Size",
                value=values[30],
                precision=0,
                info="Size of each tile during encoding. Larger = faster but more VRAM. Try 512-1024 for 8-12GB GPUs."
            )
            vae_encode_tile_overlap = gr.Number(
                label="Encode Tile Overlap",
                value=values[31],
                precision=0,
                info="Overlap between tiles to avoid seam artifacts. Must be less than tile size. Try 64-256."
            )
            vae_decode_tiled = gr.Checkbox(
                label="VAE Decode Tiled",
                value=values[32],
                info="Process VAE decoding in tiles. Recommended for high resolutions. Can use larger tiles than encoding."
            )
            vae_decode_tile_size = gr.Number(
                label="Decode Tile Size",
                value=values[33],
                precision=0,
                info="Size of each tile during decoding. Can be larger than encode tiles. Try 1024-2048."
            )
            vae_decode_tile_overlap = gr.Number(
                label="Decode Tile Overlap",
                value=values[34],
                precision=0,
                info="Overlap during decoding. Higher overlap = smoother seams but slower. Must be < tile size."
            )
            tile_debug = gr.Dropdown(
                label="Tile Debug",
                choices=["false", "encode", "decode"],
                value=values[35],
                info="Debug tiling process. 'false' = normal operation. Use 'encode'/'decode' to save intermediate tiles for troubleshooting."
            )

            # Performance & Compile
            gr.Markdown("#### ⚡ Performance & Compile")
            attention_mode = gr.Dropdown(
                label="Attention Backend",
                choices=["sdpa", "flash_attn"],
                value=values[36],
                info="flash_attn is faster but requires installation. Auto-falls back to sdpa if unavailable. sdpa is more compatible."
            )
            compile_dit = gr.Checkbox(
                label="Compile DiT",
                value=values[37],
                info="Use torch.compile for DiT model. 2-3x faster after warmup. Requires VS Build Tools on Windows. First run is slow."
            )
            compile_vae = gr.Checkbox(
                label="Compile VAE",
                value=values[38],
                info="Use torch.compile for VAE model. Significant speedup for decoding. Requires VS Build Tools on Windows."
            )
            compile_backend = gr.Dropdown(
                label="Compile Backend",
                choices=["inductor", "cudagraphs"],
                value=values[39],
                info="'inductor' is default and most compatible. 'cudagraphs' may be faster but less flexible. Use inductor unless you know what you're doing."
            )
            compile_mode = gr.Dropdown(
                label="Compile Mode",
                choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                value=values[40],
                info="'default' balanced. 'reduce-overhead' faster warmup. 'max-autotune' best performance but slow compilation. Start with default."
            )
            compile_fullgraph = gr.Checkbox(
                label="Compile Fullgraph",
                value=values[41],
                info="Compile entire model graph at once. May fail on complex models. Leave unchecked unless you need maximum performance."
            )
            compile_dynamic = gr.Checkbox(
                label="Compile Dynamic Shapes",
                value=values[42],
                info="Support varying input shapes with compilation. Slower compilation but more flexible. Enable if processing mixed resolutions."
            )
            compile_dynamo_cache_size_limit = gr.Number(
                label="Compile Dynamo Cache Size Limit",
                value=values[43],
                precision=0,
                info="Max cached compiled graphs. Higher = more memory but fewer recompilations. Default 64 is good for most cases."
            )
            compile_dynamo_recompile_limit = gr.Number(
                label="Compile Dynamo Recompile Limit",
                value=values[44],
                precision=0,
                info="Max recompilations allowed. Prevents infinite recompile loops. Default 128 is safe. Lower if compilation is slow."
            )
            cache_dit = gr.Checkbox(
                label="Cache DiT (single GPU only)",
                value=values[45],
                info="Keep DiT model in CUDA graphs cache for maximum speed. Only works with single GPU. Significant speedup for repeated processing."
            )
            cache_vae = gr.Checkbox(
                label="Cache VAE (single GPU only)",
                value=values[46],
                info="Keep VAE model in CUDA graphs cache. Single GPU only. Faster encoding/decoding at cost of higher baseline VRAM usage."
            )
            debug = gr.Checkbox(
                label="Debug Logging",
                value=values[47],
                info="Enable detailed debug output. Useful for troubleshooting but creates verbose logs. Enable if encountering errors."
            )

        # Right Column: Output & Actions
        with gr.Column(scale=2):
            gr.Markdown("### 🎯 Output / Actions")

            # Status and progress
            health_display = gr.Markdown(value="", visible=False)
            status_box = gr.Markdown(value="Ready.")
            
            # Progress tracking
            progress_indicator = gr.Markdown(value="", visible=True)
            eta_display = gr.Markdown(value="", visible=True)
            
            log_box = gr.Textbox(
                label="📋 Run Log",
                value="",
                lines=12,
                show_copy_button=True
            )

            # Output display with enhanced comparison
            output_video = gr.Video(
                label="🎬 Upscaled Video",
                interactive=False,
                show_download_button=True
            )
            output_image = gr.Image(
                label="🖼️ Upscaled Image / Preview",
                interactive=False,
                show_download_button=True
            )
            
            # Gallery for batch results
            batch_gallery = gr.Gallery(
                label="📦 Batch Results",
                visible=False,
                columns=3,
                rows=2,
                height="auto",
                object_fit="contain",
                show_download_button=True
            )

            # Enhanced ImageSlider with latest Gradio features
            image_slider = gr.ImageSlider(
                label="🔍 Image Comparison",
                interactive=False,
                height=500,
                slider_position=50,
                max_height=600,
                buttons=["download", "fullscreen"]
            )

            # Last processed chunk info
            chunk_info = gr.Markdown("Last processed chunk will appear here.")
            resume_status = gr.Markdown("", visible=True)
            chunk_progress = gr.Markdown("", visible=True)

            # Warnings and info
            alpha_warn = gr.Markdown(
                '<span class="warning-text">⚠️ PNG inputs with alpha are preserved; MP4 output drops alpha. Choose PNG output to retain alpha.</span>',
                visible=False
            )
            fps_warn = gr.Markdown(
                '<span class="warning-text">⚠️ Input video has no FPS metadata. Output will use 30 FPS default. Override FPS if needed.</span>',
                visible=False
            )
            comparison_note = gr.HTML("")

            # Face restoration toggle
            face_restore_chk = gr.Checkbox(
                label="👤 Apply Face Restoration after upscale",
                value=global_settings.get("face_global", False)
            )

            # Action buttons
            with gr.Row():
                upscale_btn = gr.Button(
                    "🚀 Upscale (subprocess)",
                    variant="primary",
                    size="lg"
                )
                cancel_confirm = gr.Checkbox(
                    label="Confirm cancel",
                    value=False
                )
                cancel_btn = gr.Button(
                    "⏹️ Cancel",
                    variant="stop"
                )
                preview_btn = gr.Button(
                    "👁️ First-frame Preview",
                    size="lg"
                )

            # Utility buttons
            with gr.Row():
                open_outputs_btn = gr.Button("📂 Open Outputs Folder")
                delete_confirm = gr.Checkbox(
                    label="Confirm delete temp",
                    value=False,
                    visible=False
                )
                delete_temp_btn = gr.Button("🗑️ Delete Temp Folder")

            # Preset management
            preset_dropdown, preset_name, save_preset_btn, load_preset_btn, preset_status, safe_defaults_btn = preset_section(
                "SeedVR2",
                preset_manager,
                values[4],  # dit_model
                preset_manager.list_presets("seedvr2", defaults["dit_model"]),
                last_used_name or "",
                safe_defaults_label="🔄 Safe Defaults (SeedVR2)"
            )

            # Mode information
            gr.Markdown("#### ℹ️ Mode Info")
            gr.Markdown(
                "Subprocess mode is active by default. Use Global Settings tab to switch to In-app mode (keeps models loaded, higher memory). Restart required to return to subprocess after switching."
            )
            gr.Markdown("Comparison: Enhanced ImageSlider with fullscreen and download support.")

    # Collect all input components for preset/callback management
    inputs_list = [
        input_path, output_override, output_format, model_dir, dit_model,
        batch_enable, batch_input, batch_output, chunk_enable, scene_threshold,
        scene_min_len, resolution, max_resolution, batch_size, uniform_batch_size,
        seed, skip_first_frames, load_cap, prepend_frames, temporal_overlap,
        color_correction, input_noise_scale, latent_noise_scale, cuda_device,
        dit_offload_device, vae_offload_device, tensor_offload_device, blocks_to_swap,
        swap_io_components, vae_encode_tiled, vae_encode_tile_size, vae_encode_tile_overlap,
        vae_decode_tiled, vae_decode_tile_size, vae_decode_tile_overlap, tile_debug,
        attention_mode, compile_dit, compile_vae, compile_backend, compile_mode,
        compile_fullgraph, compile_dynamic, compile_dynamo_cache_size_limit,
        compile_dynamo_recompile_limit, cache_dit, cache_vae, debug, resume_chunking
    ]

    # Update model status on tab load
    def initialize_model_status():
        try:
            from shared.model_manager import get_model_manager
            model_manager = get_model_manager()
            status_text = service.get("get_model_loading_status", lambda: "No models loaded")()
            return gr.Markdown.update(value=f"### 🔧 Model Status\n{status_text}")
        except Exception:
            return gr.Markdown.update(value="### 🔧 Model Status\nStatus unavailable")

    # Wire up all the event handlers

    # Input handling
    def cache_path_value(val, state):
        state["seed_controls"]["last_input_path"] = val if val else ""
        return gr.Markdown.update(value="✅ Input cached for resolution/chunk estimates.", visible=True), state

    def cache_upload(val, state):
        state["seed_controls"]["last_input_path"] = val if val else ""
        return val or "", gr.Markdown.update(value="✅ Input cached for resolution/chunk estimates.", visible=True), state

    input_file.upload(
        fn=lambda val, state: cache_upload(val, state),
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state]
    )

    input_path.change(
        fn=lambda val, state: cache_path_value(val, state),
        inputs=[input_path, shared_state],
        outputs=[input_cache_msg, shared_state]
    )

    # Model caching and status updates
    def cache_model(m, state):
        state["seed_controls"]["current_model"] = m
        # Trigger model loading when model changes
        try:
            from shared.model_manager import get_model_manager
            model_manager = get_model_manager()
            status_text = service.get("get_model_loading_status", lambda: "Model status unavailable")()
            return gr.Markdown.update(value=f"✅ Model selected: {m}", visible=True), gr.Markdown.update(value=f"### 🔧 Model Status\n{status_text}")
        except Exception as e:
            return gr.Markdown.update(value=f"✅ Model selected: {m}", visible=True), gr.Markdown.update(value=f"### 🔧 Model Status\nError: {str(e)}")

    dit_model.change(
        fn=lambda m, state: cache_model(m, state),
        inputs=[dit_model, shared_state],
        outputs=[model_cache_msg, model_status]
    )

    # Update model status periodically
    def update_model_status():
        try:
            from shared.model_manager import get_model_manager
            model_manager = get_model_manager()
            status_text = service.get("get_model_loading_status", lambda: "Model status unavailable")()
            return gr.Markdown.update(value=f"### 🔧 Model Status\n{status_text}")
        except Exception:
            return gr.Markdown.update(value="### 🔧 Model Status\nStatus unavailable")

    # Add a refresh button for model status
    with gr.Row():
        refresh_model_status_btn = gr.Button("🔄 Refresh Model Status", size="sm", variant="secondary")
        toggle_auto_refresh = gr.Checkbox(label="Auto-refresh (2s)", value=False, scale=0)
    
    refresh_model_status_btn.click(
        fn=update_model_status,
        outputs=model_status
    )
    
    # Toggle timer on/off
    toggle_auto_refresh.change(
        fn=lambda enabled: gr.Timer(value=2.0, active=enabled),
        inputs=toggle_auto_refresh,
        outputs=model_status_timer
    )
    
    # Timer tick updates model status
    model_status_timer.tick(
        fn=update_model_status,
        outputs=model_status
    )

    # Resume status checking
    check_resume_btn.click(
        fn=lambda fmt, gs: service["check_resume_status"](gs, fmt),
        inputs=[output_format, global_settings],
        outputs=resume_status
    )

    # Main action buttons with gr.Progress
    upscale_btn.click(
        fn=lambda *args, progress=gr.Progress(): service["run_action"](*args[:-1], preview_only=False, state=args[-1], progress=progress),
        inputs=[input_file, face_restore_chk] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_video, output_image,
            chunk_info, resume_status, chunk_progress, comparison_note, image_slider, batch_gallery, shared_state
        ]
    )

    preview_btn.click(
        fn=lambda *args, progress=gr.Progress(): service["run_action"](*args[:-1], preview_only=True, state=args[-1], progress=progress),
        inputs=[input_file, face_restore_chk] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_video, output_image,
            chunk_info, resume_status, chunk_progress, comparison_note, image_slider, batch_gallery, shared_state
        ]
    )

    cancel_btn.click(
        fn=lambda ok, state: (service["cancel_action"](), state) if ok else (gr.Markdown.update(value="⚠️ Enable 'Confirm cancel' to stop."), "", state),
        inputs=[cancel_confirm, shared_state],
        outputs=[status_box, log_box, shared_state]
    )

    # Utility buttons
    open_outputs_btn.click(
        fn=lambda state: (service["open_outputs_folder"](state), state),
        inputs=[shared_state],
        outputs=[status_box, shared_state]
    )

    delete_temp_btn.click(
        fn=lambda ok, state: (service["clear_temp_folder"](ok), state),
        inputs=[delete_confirm, shared_state],
        outputs=[status_box, shared_state]
    )

    # Preset management
    save_preset_btn.click(
        fn=lambda *args: service["save_preset"](*args[:-1]) + (args[-1],),
        inputs=[preset_name, dit_model] + inputs_list + [shared_state],
        outputs=[preset_dropdown, preset_status] + inputs_list + [shared_state]
    )

    load_preset_btn.click(
        fn=lambda preset, model, *vals: service["load_preset"](preset, model, defaults, list(vals[:-1])) + (vals[-1],),
        inputs=[preset_dropdown, dit_model] + inputs_list + [shared_state],
        outputs=inputs_list + [shared_state]
    )

    safe_defaults_btn.click(
        fn=service["safe_defaults"],
        outputs=inputs_list
    )

    # Update health display from shared state
    def update_health_display(state):
        health_text = state.get("health_banner", {}).get("text", "")
        return gr.Markdown.update(value=health_text)

    shared_state.change(
        fn=update_health_display,
        inputs=shared_state,
        outputs=health_display
    )

    # Output format change handler for alpha warnings
    def update_alpha_warning(format_choice, input_path):
        if format_choice == "mp4":
            # Check if input might have alpha (PNG files or certain video formats)
            has_potential_alpha = False
            if input_path and input_path.strip():
                path_lower = input_path.lower()
                has_potential_alpha = (path_lower.endswith('.png') or
                                     path_lower.endswith('.tiff') or
                                     path_lower.endswith('.tif') or
                                     'png' in path_lower or
                                     'alpha' in path_lower)
            return gr.Markdown.update(visible=True)
        return gr.Markdown.update(visible=False)

    output_format.change(
        fn=update_alpha_warning,
        inputs=[output_format, input_path],
        outputs=alpha_warn
    )

    # Input path change should also trigger alpha warning update
    input_path.change(
        fn=lambda path, fmt: update_alpha_warning(fmt, path),
        inputs=[input_path, output_format],
        outputs=alpha_warn
    )

    # FPS metadata checking
    def check_fps_metadata(input_path_val):
        if not input_path_val or not input_path_val.strip():
            return gr.Markdown.update(visible=False)

        from shared.path_utils import get_media_fps, detect_input_type
        try:
            input_type = detect_input_type(input_path_val)
            if input_type == "video":
                fps = get_media_fps(input_path_val)
                if fps is None or fps <= 0:
                    return gr.Markdown.update(visible=True)
        except Exception:
            return gr.Markdown.update(visible=True)

        return gr.Markdown.update(visible=False)

    input_path.change(
        fn=check_fps_metadata,
        inputs=[input_path],
        outputs=fps_warn
    )

    # Initialize comparison slider and model status
    comparison_note.update(service["comparison_html_slider"]())
    model_status.update(initialize_model_status())
