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
from shared.video_comparison_slider import create_video_comparison_html
from shared.ui_validators import validate_batch_size_seedvr2, validate_resolution


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
    # FIXED: Use startup cache for current model instead of re-loading
    defaults = seedvr2_defaults()
    default_model = defaults.get("dit_model")
    
    # Try to load from startup cache first (already loaded in secourses_app.py)
    seedvr2_cache = shared_state.value.get("seed_controls", {}).get("seedvr2_cache", {})
    cached_for_model = seedvr2_cache.get(default_model)
    
    if cached_for_model:
        # Use pre-loaded settings from startup
        merged_defaults = cached_for_model
        last_used_name = preset_manager.get_last_used_name("seedvr2", default_model)
        
        def update_success(state):
            existing = state["health_banner"]["text"]
            success_msg = f"✅ SeedVR2: Restored last-used preset for {default_model}"
            state["health_banner"]["text"] = existing + "\n" + success_msg if existing else success_msg
            return state
        shared_state.value = update_success(shared_state.value)
    else:
        # Fallback: Load on-demand if cache missed
        last_used_name = preset_manager.get_last_used_name("seedvr2", default_model)
        last_used = preset_manager.load_last_used("seedvr2", default_model)
        
        if last_used_name and last_used is None:
            def update_warning(state):
                existing = state["health_banner"]["text"]
                warning = f"⚠️ Last used SeedVR2 preset '{last_used_name}' not found or corrupted; loaded defaults."
                state["health_banner"]["text"] = existing + "\n" + warning if existing else warning
                return state
            shared_state.value = update_warning(shared_state.value)
        
        merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    
    values = [merged_defaults[k] for k in SEEDVR2_ORDER]

    # Build service callbacks
    service = build_seedvr2_callbacks(
        preset_manager, runner, run_logger, global_settings,
        shared_state, output_dir, temp_dir
    )

    # GPU hint and macOS detection with early validation + ffmpeg check
    import platform
    is_macos = platform.system() == "Darwin"
    cuda_available = False
    cuda_count = 0
    gpu_hint = "CUDA detection in progress..."
    compile_available = False
    ffmpeg_available = False

    # Check ffmpeg availability up front
    try:
        from shared.error_handling import check_ffmpeg_available
        ffmpeg_available, ffmpeg_msg = check_ffmpeg_available()
        if not ffmpeg_available:
            gpu_hint = f"❌ CRITICAL: {ffmpeg_msg}\nffmpeg is REQUIRED for video processing. Install ffmpeg and restart."
    except Exception:
        ffmpeg_available = False
        gpu_hint = "❌ CRITICAL: ffmpeg not found. Install ffmpeg to enable video processing."

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        cuda_count = torch.cuda.device_count() if cuda_available else 0
        
        if is_macos:
            gpu_hint = "macOS detected - CUDA device selection disabled (not supported by SeedVR2 CLI)"
            compile_available = False  # macOS doesn't support torch.compile with MPS well
        else:
            if cuda_available and cuda_count > 0:
                gpu_hint = f"✅ Detected {cuda_count} CUDA GPU(s) - GPU acceleration available"
                # Check if VS Build Tools available for compile
                from shared.health import is_vs_build_tools_available
                compile_available = is_vs_build_tools_available()
                if not compile_available:
                    gpu_hint += "\n⚠️ VS Build Tools not detected - torch.compile will be disabled"
            else:
                gpu_hint = "⚠️ CUDA not available - GPU-only features disabled. Processing will use CPU (significantly slower)"
                compile_available = False
        
        # Append ffmpeg status to GPU hint
        if not ffmpeg_available:
            gpu_hint = f"❌ CRITICAL: ffmpeg not found\n{gpu_hint}"
        
    except Exception as e:
        gpu_hint = f"❌ CUDA detection failed: {str(e)}"
        cuda_available = False
        compile_available = False

    # Layout: Two-column design
    with gr.Row():
        # Left Column: Input Controls
        with gr.Column(scale=3):
            gr.Markdown("### 🎬 Input / Controls")

            # Input section with enhanced detection (file upload only, path textbox moved below)
            with gr.Group():
                gr.Markdown("#### 📁 Enhanced Input: Video Files & Frame Folders")
                gr.Markdown("*Auto-detects whether your input is a single video file or a folder containing frame sequences*")

                input_file = gr.File(
                    label="Upload video or image (optional)",
                    type="filepath",
                    file_types=["video", "image"]
                )
                
                # Auto-detection results
                input_detection_result = gr.Markdown("", visible=False)
                
                # Detect input button
                detect_input_btn = gr.Button("🔍 Detect Input Type", size="sm", variant="secondary")
                
                input_cache_msg = gr.Markdown("", visible=False)
                auto_res_msg = gr.Markdown("", visible=False)

            # Scene splitting controls - CLEANED UP VERSION
            with gr.Accordion("🎬 SeedVR2 Native Streaming (Advanced)", open=False):
                gr.Markdown("""
                ### 🚀 SeedVR2 Native Streaming Mode
                
                **What is this?** SeedVR2-specific CLI-internal memory-bounded processing that streams frames through the model.
                
                **When to use:**
                - VERY long videos (>10 minutes)
                - Limited VRAM scenarios
                - Can combine with PySceneDetect (Resolution tab) for maximum efficiency
                
                **How it works:**
                - PySceneDetect (Resolution tab) splits video into scenes FIRST
                - Then this native streaming processes each scene in frame chunks
                - Double-chunking for ultimate memory efficiency
                
                💡 **For most users**: Configure chunking in **Resolution & Scene Split** tab only.
                Use this only if PySceneDetect chunking alone isn't enough for your VRAM.
                """)
                
                chunk_size_frames = gr.Number(
                    label="Streaming Chunk Size (frames, 0=disabled)",
                    value=values[8] if len(values) > 8 and isinstance(values[8], (int, float)) else 0,  # Now index 8 (removed 3 legacy controls)
                    precision=0,
                    info="Process video in N-frame chunks via CLI --chunk_size. Recommended: 300-1000 frames/chunk for videos >10 min. 0 = disabled (use PySceneDetect only). Works TOGETHER with PySceneDetect chunking."
                )
                
                # Automatic chunk estimation display
                chunk_estimate_display = gr.Markdown(
                    "",
                    visible=False,
                    label="📊 Chunk Estimation"
                )
                
                resume_chunking = gr.Checkbox(
                    label="Resume from partial chunks",
                    value=values[46] if len(values) > 46 else False,  # Now index 46 (was 49, removed 3 items)
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
            
            # Model management buttons with clarification
            gr.Markdown("""
            **ℹ️ Model Unloading (Subprocess Mode):**
            - In subprocess mode (current), models are automatically unloaded after each run
            - These buttons force CUDA cache clearing for manual VRAM management
            - **In-app mode** (when implemented) will cache models between runs - then these buttons will actually unload persistent models
            """)
            with gr.Row():
                unload_model_btn = gr.Button("🧹 Clear CUDA Cache", variant="secondary", size="sm")
                unload_all_models_btn = gr.Button("🧹 Clear All CUDA Caches", variant="stop", size="sm")
            model_unload_status = gr.Markdown("", visible=False)
            
            # Timer for periodic model status updates
            model_status_timer = gr.Timer(value=2.0, active=False)  # Update every 2 seconds when active

            # Resolution controls with auto-resolution
            with gr.Row():
                resolution = gr.Slider(
                    label="Target Resolution (short side)",
                    minimum=256, maximum=4096, step=16,
                    value=values[9],  # Was 12, now 9 (removed 3 legacy items)
                    info="Target resolution for shortest side. 1080 = 1080p, 2160 = 4K. Higher = better quality but slower. Must be multiple of 16."
                )
                max_resolution = gr.Slider(
                    label="Max Resolution (0 = no cap)",
                    minimum=0, maximum=8192, step=16,
                    value=values[10],  # Was 13, now 10
                    info="Maximum resolution cap for safety. Prevents accidental 8K+ upscaling. 0 = unlimited. Set to 4096 for 4K max, 2160 for 1080p max."
                )
            resolution_warning = gr.Markdown("", visible=False)

            # Core processing parameters
            batch_size = gr.Slider(
                label="Batch Size (must be 4n+1: 5, 9, 13, 17...)",
                minimum=5, maximum=201, step=4,
                value=values[11],  # Was 14, now 11 (shift -3)
                info="SeedVR2 requires batch size to follow 4n+1 formula (5, 9, 13, 17, 21...)"
            )
            batch_size_warning = gr.Markdown("", visible=False)
            uniform_batch_size = gr.Checkbox(
                label="Uniform Batch Size",
                value=values[12],  # Was 15, now 12
                info="Force all batches to same size by padding. Improves compilation efficiency but may use more memory. Recommended ON with torch.compile."
            )
            seed = gr.Number(
                label="Seed",
                value=values[13],  # Was 16, now 13
                precision=0,
                info="Random seed for reproducible results. Same seed + settings = identical output. -1 or 0 = random. Try 42 for consistent testing."
            )

            # Frame controls
            with gr.Row():
                skip_first_frames = gr.Number(
                    label="Skip First Frames",
                    value=values[14],  # Was 17, now 14
                    precision=0,
                    info="Skip N frames from start of video. Useful to skip intros/logos. 0 = process from beginning."
                )
                load_cap = gr.Number(
                    label="Load Cap (0 = all)",
                    value=values[15],  # Was 18, now 15
                    precision=0,
                    info="Process only first N frames. Useful for quick tests. 0 = process entire video. Combine with skip for specific range."
                )
                prepend_frames = gr.Number(
                    label="Prepend Frames",
                    value=values[16],  # Was 19, now 16
                    precision=0,
                    info="Prepend N copies of first frame for temporal stability. Helps reduce artifacts at video start. Try 2-4."
                )
                temporal_overlap = gr.Number(
                    label="Temporal Overlap",
                    value=values[17],  # Was 20, now 17
                    precision=0,
                    info="Overlap frames between processing batches. Improves temporal consistency. Higher = smoother but slower. Try 1-3."
                )

            # Color correction
            color_correction = gr.Dropdown(
                label="Color Correction",
                choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                value=values[18],  # Was 21, now 18
                info="Method for maintaining color accuracy. 'lab' is default and robust. 'wavelet' preserves details better. 'none' for creative control."
            )

            # Noise controls
            input_noise_scale = gr.Slider(
                label="Input Noise Scale",
                minimum=0.0, maximum=1.0, step=0.01,
                value=values[19],  # Was 22, now 19
                info="Add noise to input before encoding. Can help with smooth gradients but may reduce sharpness. 0.0 = no noise. Try 0.0-0.1."
            )
            latent_noise_scale = gr.Slider(
                label="Latent Noise Scale",
                minimum=0.0, maximum=1.0, step=0.01,
                value=values[20],  # Was 23, now 20
                info="Add noise in latent space during diffusion. Can improve detail generation. 0.0 = no noise. Typical: 0.0-0.05."
            )

            # Device configuration
            gr.Markdown("#### 💻 Device & Offload")
            
            # Show GPU availability warning if CUDA not available
            if not cuda_available and not is_macos:
                gr.Markdown(
                    f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
                    f'<strong>⚠️ GPU Acceleration Unavailable</strong><br>'
                    f'{gpu_hint}<br><br>'
                    f'GPU-only controls are disabled. Processing will use CPU (10-100x slower).'
                    f'</div>',
                    elem_classes="warning-text"
                )
            
            cuda_device = gr.Textbox(
                label="CUDA Devices (e.g., 0 or 0,1,2 or 'all')",
                value=values[21] if (cuda_available and not is_macos) else "",
                info=f"{gpu_hint} | Single GPU: 0, 1, etc. | Multi-GPU: 0,1,2 or 'all' to use all available GPUs. Single GPU recommended for caching.",
                placeholder="0 or all",
                visible=not is_macos,
                interactive=cuda_available  # Disable if CUDA not available
            )
            dit_offload_device = gr.Textbox(
                label="DiT Offload Device",
                value=values[22],  # Was 25, now 22
                placeholder="none / cpu / GPU id",
                info="Where to offload DiT model when not in use. 'cpu' saves VRAM, 'none' keeps on GPU. Required for BlockSwap."
            )
            vae_offload_device = gr.Textbox(
                label="VAE Offload Device",
                value=values[23],  # Was 26, now 23
                placeholder="none / cpu / GPU id",
                info="Where to offload VAE model when not in use. 'cpu' saves VRAM, 'none' keeps on GPU for faster processing."
            )
            tensor_offload_device = gr.Textbox(
                label="Tensor Offload Device",
                value=values[24],  # Was 27, now 24
                placeholder="cpu / none / GPU id",
                info="Where to offload intermediate tensors. 'cpu' is recommended for memory management between processing phases."
            )

            # BlockSwap configuration
            gr.Markdown("#### 🔄 BlockSwap")
            blocks_to_swap = gr.Slider(
                label="Blocks to Swap",
                minimum=0, maximum=36, step=1,
                value=values[25],  # Was 28, now 25
                info="Number of DiT blocks to swap to CPU. Higher values save more VRAM but slow processing. Try 20-30 for 8GB GPUs."
            )
            swap_io_components = gr.Checkbox(
                label="Swap I/O Components",
                value=values[26],  # Was 29, now 26
                info="Swap input/output layers to CPU. Enable for maximum VRAM savings on limited GPUs. Requires DiT offload to CPU."
            )

            # VAE Tiling
            gr.Markdown("#### 🧩 VAE Tiling")
            vae_encode_tiled = gr.Checkbox(
                label="VAE Encode Tiled",
                value=values[27],  # Was 30, now 27 (shift -3)
                info="Process VAE encoding in tiles to reduce VRAM usage. Essential for 4K+ resolutions on GPUs with <16GB VRAM."
            )
            vae_encode_tile_size = gr.Number(
                label="Encode Tile Size",
                value=values[28],  # Was 31, now 28
                precision=0,
                info="Size of each tile during encoding. Larger = faster but more VRAM. Try 512-1024 for 8-12GB GPUs."
            )
            vae_encode_tile_overlap = gr.Number(
                label="Encode Tile Overlap",
                value=values[29],  # Was 32, now 29
                precision=0,
                info="Overlap between tiles to avoid seam artifacts. Must be less than tile size. Try 64-256."
            )
            vae_decode_tiled = gr.Checkbox(
                label="VAE Decode Tiled",
                value=values[30],  # Was 33, now 30
                info="Process VAE decoding in tiles. Recommended for high resolutions. Can use larger tiles than encoding."
            )
            vae_decode_tile_size = gr.Number(
                label="Decode Tile Size",
                value=values[31],  # Was 34, now 31
                precision=0,
                info="Size of each tile during decoding. Can be larger than encode tiles. Try 1024-2048."
            )
            vae_decode_tile_overlap = gr.Number(
                label="Decode Tile Overlap",
                value=values[32],  # Was 35, now 32
                precision=0,
                info="Overlap during decoding. Higher overlap = smoother seams but slower. Must be < tile size."
            )
            
            # Tile validation warnings
            tile_encode_warning = gr.Markdown("", visible=False)
            tile_decode_warning = gr.Markdown("", visible=False)
            
            tile_debug = gr.Dropdown(
                label="Tile Debug",
                choices=["false", "encode", "decode"],
                value=values[33],  # Was 36, now 33
                info="Debug tiling process. 'false' = normal operation. Use 'encode'/'decode' to save intermediate tiles for troubleshooting."
            )

            # Performance & Compile
            gr.Markdown("#### ⚡ Performance & Compile")
            
            # Show compile availability warning if not available
            if not compile_available:
                gr.Markdown(
                    f'<div style="background: #fff3cd; padding: 12px; border-radius: 8px; border: 1px solid #ffc107;">'
                    f'<strong>⚠️ Torch.compile Unavailable</strong><br>'
                    f'Compile options are disabled. Install Visual Studio Build Tools (Windows) or ensure CUDA is available.<br>'
                    f'Compilation provides 2-3x speedup but is not required for processing.'
                    f'</div>',
                    elem_classes="warning-text"
                )
            
            attention_mode = gr.Dropdown(
                label="Attention Backend",
                choices=["sdpa", "flash_attn"],
                value=values[34],  # Was 37, now 34
                info="flash_attn is faster but requires installation. Auto-falls back to sdpa if unavailable. sdpa is more compatible.",
                interactive=cuda_available  # Disable if no CUDA (attention backends need GPU)
            )
            compile_dit = gr.Checkbox(
                label="Compile DiT",
                value=values[35] if compile_available else False,  # Force False if compile unavailable
                info="Use torch.compile for DiT model. 2-3x faster after warmup. Requires VS Build Tools on Windows. First run is slow.",
                interactive=compile_available  # Disable if compile not available
            )
            compile_vae = gr.Checkbox(
                label="Compile VAE",
                value=values[36] if compile_available else False,  # Force False if compile unavailable
                info="Use torch.compile for VAE model. Significant speedup for decoding. Requires VS Build Tools on Windows.",
                interactive=compile_available  # Disable if compile not available
            )
            compile_backend = gr.Dropdown(
                label="Compile Backend",
                choices=["inductor", "cudagraphs"],
                value=values[37],  # Was 40, now 37
                info="'inductor' is default and most compatible. 'cudagraphs' may be faster but less flexible. Use inductor unless you know what you're doing.",
                interactive=compile_available
            )
            compile_mode = gr.Dropdown(
                label="Compile Mode",
                choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                value=values[38],  # Was 41, now 38
                info="'default' balanced. 'reduce-overhead' faster warmup. 'max-autotune' best performance but slow compilation. Start with default.",
                interactive=compile_available
            )
            compile_fullgraph = gr.Checkbox(
                label="Compile Fullgraph",
                value=values[39] if compile_available else False,
                info="Compile entire model graph at once. May fail on complex models. Leave unchecked unless you need maximum performance.",
                interactive=compile_available
            )
            compile_dynamic = gr.Checkbox(
                label="Compile Dynamic Shapes",
                value=values[40] if compile_available else False,
                info="Support varying input shapes with compilation. Slower compilation but more flexible. Enable if processing mixed resolutions.",
                interactive=compile_available
            )
            compile_dynamo_cache_size_limit = gr.Number(
                label="Compile Dynamo Cache Size Limit",
                value=values[41],  # Was 44, now 41
                precision=0,
                info="Max cached compiled graphs. Higher = more memory but fewer recompilations. Default 64 is good for most cases.",
                interactive=compile_available
            )
            compile_dynamo_recompile_limit = gr.Number(
                label="Compile Dynamo Recompile Limit",
                value=values[42],  # Was 45, now 42
                precision=0,
                info="Max recompilations allowed. Prevents infinite recompile loops. Default 128 is safe. Lower if compilation is slow.",
                interactive=compile_available
            )
            cache_dit = gr.Checkbox(
                label="Cache DiT (single GPU only)",
                value=values[43] if (cuda_available and compile_available) else False,
                info="Keep DiT model in CUDA graphs cache for maximum speed. Only works with single GPU. Significant speedup for repeated processing.",
                interactive=(cuda_available and compile_available)
            )
            cache_vae = gr.Checkbox(
                label="Cache VAE (single GPU only)",
                value=values[44] if (cuda_available and compile_available) else False,
                info="Keep VAE model in CUDA graphs cache. Single GPU only. Faster encoding/decoding at cost of higher baseline VRAM usage.",
                interactive=(cuda_available and compile_available)
            )
            
            # Cache warning for multi-GPU
            cache_warning = gr.Markdown("", visible=False)
            
            debug = gr.Checkbox(
                label="Debug Logging",
                value=values[45],  # Was 48, now 45
                info="Enable detailed debug output. Useful for troubleshooting but creates verbose logs. Enable if encountering errors."
            )

            # Output & Metadata controls (shared settings from Output tab)
            gr.Markdown("---")
            gr.Markdown("#### 📤 Output & Metadata")
            
            save_metadata = gr.Checkbox(
                label="Save Processing Metadata",
                value=values[46] if len(values) > 46 else True,
                info="Save run metadata (settings, timestamps, status) to output folder. Respects global telemetry toggle."
            )
            
            fps_override = gr.Number(
                label="FPS Override (0 = use source FPS)",
                value=values[47] if len(values) > 47 else 0,
                precision=2,
                info="Override output video FPS. 0 = preserve original FPS, 30 = force 30fps, 60 = force 60fps. Also sourced from Output tab if set."
            )

            # Input path textbox - MOVED HERE: after all controls, at bottom of left column as requested
            gr.Markdown("---")
            gr.Markdown("#### 📁 Direct Input Path (Alternative to Upload)")
            input_path = gr.Textbox(
                label="Input Video or Frames Folder Path",
                value=values[0],
                placeholder="C:/path/to/video.mp4 or C:/path/to/frames/",
                info="Enter path to either a video file (mp4, avi, mov, etc.) or folder containing image frames (jpg, png, tiff, etc.). Automatically detected - works on Windows and Linux."
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
            
            # Video Comparison with custom HTML5 slider
            video_comparison_html = gr.HTML(
                label="🎬 Video Comparison Slider",
                value="",
                visible=False
            )

            # Utility buttons - MOVED HERE: directly under output panel as requested
            with gr.Row():
                open_outputs_btn = gr.Button("📂 Open Outputs Folder", size="lg")
                delete_temp_btn = gr.Button("🗑️ Delete Temp Folder", size="lg")
            
            delete_confirm = gr.Checkbox(
                label="⚠️ Confirm delete temp folder (required for safety)",
                value=False,
                info="Enable this to confirm deletion of temporary files"
            )

            # Batch processing controls - MOVED HERE: right column above Last Processed Chunk as requested
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

            # Action buttons with ffmpeg gating
            with gr.Row():
                upscale_btn = gr.Button(
                    "🚀 Upscale" if ffmpeg_available else "❌ Upscale (ffmpeg required)",
                    variant="primary" if ffmpeg_available else "stop",
                    size="lg",
                    interactive=ffmpeg_available
                )
                cancel_confirm = gr.Checkbox(
                    label="⚠️ Confirm cancel (subprocess mode only)",
                    value=False,
                    info="Cancel only works in subprocess mode. Check Global Settings to verify mode."
                )
                cancel_btn = gr.Button(
                    "⏹️ Cancel (subprocess only)",
                    variant="stop",
                    scale=1
                )
                preview_btn = gr.Button(
                    "👁️ First-frame Preview" if ffmpeg_available else "❌ Preview (ffmpeg required)",
                    size="lg",
                    interactive=ffmpeg_available
                )
            
            if not ffmpeg_available:
                gr.Markdown("""
                <div style="background: #ffebee; padding: 12px; border-radius: 8px; border: 2px solid #f44336;">
                    <strong>⚠️ ffmpeg NOT FOUND</strong><br>
                    Video processing requires ffmpeg to be installed and available in your system PATH.<br>
                    <br>
                    <strong>Installation:</strong><br>
                    • Windows: Download from <a href="https://ffmpeg.org/download.html" target="_blank">ffmpeg.org</a> and add to PATH<br>
                    • Linux: <code>sudo apt install ffmpeg</code> or <code>sudo yum install ffmpeg</code><br>
                    • macOS: <code>brew install ffmpeg</code><br>
                    <br>
                    After installation, restart the application.
                </div>
                """)

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
            gr.Markdown("#### ℹ️ Processing Mode")
            gr.Markdown(
                "**Current Mode:** Check Global Settings tab to view/change execution mode\n\n"
                "**Available Modes:**\n"
                "- **Subprocess (Default & RECOMMENDED):** Each run is isolated with full VRAM cleanup and **cancellation support**\n"
                "  - ✅ **Automatic vcvars wrapper** on Windows (enables torch.compile when VS Build Tools installed)\n"
                "  - ✅ **Full cancellation support** with proper cleanup\n"
                "  - ✅ **100% VRAM/RAM cleanup** after each run\n"
                "- **⚠️ In-app (EXPERIMENTAL - NOT RECOMMENDED FOR SEEDVR2):**\n"
                "  - **SeedVR2 Limitation**: Models reload each run (CLI design) - **NO speed benefit**\n"
                "  - **Cannot cancel** mid-process (no subprocess to kill)\n"
                "  - **vcvars must be pre-activated** before app start (cannot activate after Python loaded)\n"
                "  - **May work for GAN/RIFE** but SeedVR2 gains nothing from in-app mode\n\n"
                "**⚠️ IMPORTANT:** Cancel button only works in **subprocess mode**. In-app mode runs cannot be cancelled mid-process.\n\n"
                "**💡 SeedVR2 Recommendation:** **Always use subprocess mode**. In-app provides no benefits for SeedVR2 due to CLI architecture (models reload each run). Switch modes in Global Settings tab."
            )
            gr.Markdown("**Comparison:** Enhanced ImageSlider with fullscreen and download support for images. Custom HTML5 slider for videos.")

    # ============================================================================
    # 📋 PRESET INPUT LIST - CRITICAL SYNCHRONIZATION POINT
    # ============================================================================
    # This list MUST match SEEDVR2_ORDER in shared/services/seedvr2_service.py
    # 
    # ⚠️ WHEN ADDING NEW CONTROLS:
    # 1. Add default value to seedvr2_defaults() in seedvr2_service.py
    # 2. Append key to SEEDVR2_ORDER in seedvr2_service.py
    # 3. Add component to this inputs_list AT THE SAME POSITION
    # 4. Save callback will auto-validate and warn if counts mismatch
    #
    # ✅ BACKWARD COMPATIBILITY:
    # Old presets automatically get new defaults via merge_config() - no migration needed!
    #
    # Current count: len(SEEDVR2_ORDER) = 49, len(inputs_list) must also = 49 (added save_metadata, fps_override)
    # ============================================================================
    
    inputs_list = [
        input_path, output_override, output_format, model_dir, dit_model,
        batch_enable, batch_input, batch_output,
        # Legacy PySceneDetect controls removed - use Resolution tab instead
        chunk_size_frames,  # SeedVR2 native streaming only
        resolution, max_resolution, batch_size, uniform_batch_size,
        seed, skip_first_frames, load_cap, prepend_frames, temporal_overlap,
        color_correction, input_noise_scale, latent_noise_scale, cuda_device,
        dit_offload_device, vae_offload_device, tensor_offload_device, blocks_to_swap,
        swap_io_components, vae_encode_tiled, vae_encode_tile_size, vae_encode_tile_overlap,
        vae_decode_tiled, vae_decode_tile_size, vae_decode_tile_overlap, tile_debug,
        attention_mode, compile_dit, compile_vae, compile_backend, compile_mode,
        compile_fullgraph, compile_dynamic, compile_dynamo_cache_size_limit,
        compile_dynamo_recompile_limit, cache_dit, cache_vae, debug, resume_chunking,
        # Output & shared settings (integrated from Output tab)
        save_metadata, fps_override
    ]
    
    # Validate synchronization at tab initialization (development-time check)
    if len(inputs_list) != len(SEEDVR2_ORDER):
        import logging
        logger = logging.getLogger("SeedVR2Tab")
        logger.error(
            f"❌ CRITICAL: inputs_list ({len(inputs_list)}) and SEEDVR2_ORDER ({len(SEEDVR2_ORDER)}) "
            f"are OUT OF SYNC! Presets will not work correctly. "
            f"Expected {len(SEEDVR2_ORDER)} components."
        )

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

    # Input handling with auto-resolution calculation
    def cache_path_value(val, state):
        """Cache input path and optionally auto-calculate resolution"""
        state["seed_controls"]["last_input_path"] = val if val else ""
        
        # Check if auto-resolution is enabled in seed_controls
        auto_res_enabled = state.get("seed_controls", {}).get("auto_resolution", False)
        
        if auto_res_enabled and val and val.strip():
            # Auto-calculate resolution if enabled
            try:
                # Use the service's auto-resolution function
                res_slider, max_res_slider, calc_msg, updated_state = service["auto_res_on_input"](val, state)
                return (
                    gr.Markdown.update(value="✅ Input cached. Auto-resolution calculated.", visible=True),
                    updated_state,
                    res_slider,
                    max_res_slider,
                    calc_msg
                )
            except Exception as e:
                # Fallback if auto-calc fails
                return (
                    gr.Markdown.update(value=f"✅ Input cached. Auto-calc warning: {str(e)[:50]}", visible=True),
                    state,
                    gr.Slider.update(),
                    gr.Slider.update(),
                    gr.Markdown.update(visible=False)
                )
        else:
            # Just cache without auto-calc
            return (
                gr.Markdown.update(value="✅ Input cached for resolution/chunk estimates.", visible=True),
                state,
                gr.Slider.update(),
                gr.Slider.update(),
                gr.Markdown.update(visible=False)
            )

    def cache_upload(val, state):
        """Cache uploaded file and optionally auto-calculate resolution"""
        state["seed_controls"]["last_input_path"] = val if val else ""
        
        # Check if auto-resolution is enabled
        auto_res_enabled = state.get("seed_controls", {}).get("auto_resolution", False)
        
        if auto_res_enabled and val:
            try:
                # Auto-calculate resolution if enabled
                res_slider, max_res_slider, calc_msg, updated_state = service["auto_res_on_input"](val, state)
                return (
                    val or "",
                    gr.Markdown.update(value="✅ File uploaded. Auto-resolution calculated.", visible=True),
                    updated_state,
                    res_slider,
                    max_res_slider,
                    calc_msg
                )
            except Exception:
                return (
                    val or "",
                    gr.Markdown.update(value="✅ File uploaded and cached.", visible=True),
                    state,
                    gr.Slider.update(),
                    gr.Slider.update(),
                    gr.Markdown.update(visible=False)
                )
        else:
            return (
                val or "",
                gr.Markdown.update(value="✅ File uploaded and cached.", visible=True),
                state,
                gr.Slider.update(),
                gr.Slider.update(),
                gr.Markdown.update(visible=False)
            )

    # Wire up input events with resolution auto-calculation
    input_file.upload(
        fn=lambda val, state: cache_upload(val, state),
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state, resolution, max_resolution, auto_res_msg]
    )

    input_path.change(
        fn=lambda val, state: cache_path_value(val, state),
        inputs=[input_path, shared_state],
        outputs=[input_cache_msg, shared_state, resolution, max_resolution, auto_res_msg]
    )

    # Model caching and status updates with preset reload + dynamic UI updates
    def cache_model_and_reload_preset(m, state, *current_vals):
        """
        Cache model selection, reload preset, and update UI based on model metadata.
        
        Dynamically disables incompatible options (e.g., compile for GGUF models).
        """
        state["seed_controls"]["current_model"] = m
        
        # Get model metadata to check compatibility
        from shared.models.seedvr2_meta import model_meta_map
        meta_map = model_meta_map()
        model_meta = meta_map.get(m)
        
        # Determine if compile should be disabled for this model
        compile_supported = True
        multi_gpu_supported = True
        compile_warning = ""
        multi_gpu_warning = ""
        
        if model_meta:
            compile_supported = getattr(model_meta, 'compile_compatible', True)
            multi_gpu_supported = getattr(model_meta, 'supports_multi_gpu', True)
            
            if not compile_supported:
                compile_warning = f"⚠️ Model '{m}' doesn't support torch.compile (e.g., GGUF quantized models). Compile options will be auto-disabled at runtime."
            
            if not multi_gpu_supported:
                multi_gpu_warning = f"⚠️ Model '{m}' is single-GPU only. Multi-GPU device specs will be reduced to first GPU."
        
        # Load last-used preset for this model
        last_used_preset = preset_manager.load_last_used("seedvr2", m)
        
        # Get model status
        try:
            from shared.model_manager import get_model_manager
            model_manager = get_model_manager()
            status_text = service.get("get_model_loading_status", lambda: "Model status unavailable")()
            
            # Add metadata warnings to status
            if compile_warning or multi_gpu_warning:
                warnings = []
                if compile_warning:
                    warnings.append(compile_warning)
                if multi_gpu_warning:
                    warnings.append(multi_gpu_warning)
                status_text += "\n\n" + "\n".join(warnings)
            
            model_status_update = gr.Markdown.update(value=f"### 🔧 Model Status\n{status_text}")
        except Exception as e:
            model_status_update = gr.Markdown.update(value=f"### 🔧 Model Status\nError: {str(e)}")
        
        # If last-used preset exists, merge with current values
        if last_used_preset:
            current_dict = dict(zip(SEEDVR2_ORDER, current_vals))
            merged = preset_manager.merge_config(current_dict, last_used_preset)
            
            # Apply model constraints to merged preset
            if not compile_supported:
                merged["compile_dit"] = False
                merged["compile_vae"] = False
            
            new_vals = [merged[k] for k in SEEDVR2_ORDER]
            cache_msg = gr.Markdown.update(
                value=f"✅ Model '{m}' selected - loaded last-used preset\n{compile_warning}\n{multi_gpu_warning}", 
                visible=True
            )
            return [cache_msg, model_status_update] + new_vals
        else:
            # No last-used preset, keep current values but apply constraints
            current_dict = dict(zip(SEEDVR2_ORDER, current_vals))
            if not compile_supported:
                current_dict["compile_dit"] = False
                current_dict["compile_vae"] = False
            
            new_vals = [current_dict[k] for k in SEEDVR2_ORDER]
            
            cache_msg = gr.Markdown.update(
                value=f"✅ Model '{m}' selected (no saved preset)\n{compile_warning}\n{multi_gpu_warning}", 
                visible=True
            )
            return [cache_msg, model_status_update] + new_vals

    dit_model.change(
        fn=cache_model_and_reload_preset,
        inputs=[dit_model, shared_state] + inputs_list,
        outputs=[model_cache_msg, model_status] + inputs_list
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
    
    # Model unloading callbacks (CUDA cache management in subprocess mode)
    def unload_current_model():
        """Clear CUDA cache (subprocess mode) or unload model (in-app mode when implemented)"""
        from shared.model_manager import get_model_manager
        
        try:
            # In subprocess mode, just clear CUDA cache manually
            # In in-app mode (when implemented), this will actually unload the model
            try:
                from shared.gpu_utils import clear_cuda_cache
                clear_cuda_cache()
                msg = "✅ CUDA cache cleared. Note: In subprocess mode, models reload each run automatically."
            except Exception:
                msg = "⚠️ CUDA cache clear attempted (may not be available)"
            
            model_manager = get_model_manager()
            model_manager.unload_all_models()  # Update tracking state
            
            return gr.Markdown.update(value=msg, visible=True), service["get_model_loading_status"]()
        except Exception as e:
            return gr.Markdown.update(value=f"❌ Error: {str(e)}", visible=True), service["get_model_loading_status"]()
    
    def unload_all_models():
        """Clear all CUDA caches (subprocess mode) or unload all models (in-app mode when implemented)"""
        from shared.model_manager import get_model_manager
        
        try:
            # Force aggressive CUDA cache clearing
            try:
                from shared.gpu_utils import clear_cuda_cache
                clear_cuda_cache()
                msg = "✅ All CUDA caches cleared. Subprocess mode: Models reload each run."
            except Exception:
                msg = "⚠️ CUDA cache clear attempted"
            
            model_manager = get_model_manager()
            model_manager.unload_all_models()
            
            return gr.Markdown.update(value=msg, visible=True), service["get_model_loading_status"]()
        except Exception as e:
            return gr.Markdown.update(value=f"❌ Error: {str(e)}", visible=True), service["get_model_loading_status"]()
    
    unload_model_btn.click(
        fn=unload_current_model,
        outputs=[model_unload_status, model_status]
    )
    
    unload_all_models_btn.click(
        fn=unload_all_models,
        outputs=[model_unload_status, model_status]
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
            chunk_info, resume_status, chunk_progress, comparison_note, image_slider, video_comparison_html, batch_gallery, shared_state
        ]
    )

    preview_btn.click(
        fn=lambda *args, progress=gr.Progress(): service["run_action"](*args[:-1], preview_only=True, state=args[-1], progress=progress),
        inputs=[input_file, face_restore_chk] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_video, output_image,
            chunk_info, resume_status, chunk_progress, comparison_note, image_slider, video_comparison_html, batch_gallery, shared_state
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

    # Preset management - FIXED: Proper argument passing and output alignment
    save_preset_btn.click(
        fn=lambda name, model, *vals: service["save_preset"](name, model, *vals[:-1]) + (vals[-1],),
        inputs=[preset_name, dit_model] + inputs_list + [shared_state],
        outputs=[preset_dropdown, preset_status] + inputs_list + [shared_state]
    )

    load_preset_btn.click(
        fn=lambda preset, model, *vals: service["load_preset"](preset, model, list(vals[:-1])) + (vals[-1],),
        inputs=[preset_dropdown, dit_model] + inputs_list + [shared_state],
        outputs=inputs_list + [preset_status, shared_state]  # FIXED: status is now second-to-last
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

    # Tile validation helpers
    def validate_tile_encode(tile_size, overlap):
        if tile_size > 0 and overlap >= tile_size:
            return gr.Markdown.update(value=f"⚠️ Encode tile overlap ({overlap}) must be < tile size ({tile_size}). Will be auto-corrected.", visible=True)
        return gr.Markdown.update(value="", visible=False)
    
    def validate_tile_decode(tile_size, overlap):
        if tile_size > 0 and overlap >= tile_size:
            return gr.Markdown.update(value=f"⚠️ Decode tile overlap ({overlap}) must be < tile size ({tile_size}). Will be auto-corrected.", visible=True)
        return gr.Markdown.update(value="", visible=False)
    
    # Wire up tile validation
    vae_encode_tile_overlap.change(
        fn=validate_tile_encode,
        inputs=[vae_encode_tile_size, vae_encode_tile_overlap],
        outputs=tile_encode_warning
    )
    vae_encode_tile_size.change(
        fn=validate_tile_encode,
        inputs=[vae_encode_tile_size, vae_encode_tile_overlap],
        outputs=tile_encode_warning
    )
    vae_decode_tile_overlap.change(
        fn=validate_tile_decode,
        inputs=[vae_decode_tile_size, vae_decode_tile_overlap],
        outputs=tile_decode_warning
    )
    vae_decode_tile_size.change(
        fn=validate_tile_decode,
        inputs=[vae_decode_tile_size, vae_decode_tile_overlap],
        outputs=tile_decode_warning
    )
    
    # Cache + GPU validation
    def validate_cache_gpu(cache_dit_val, cache_vae_val, cuda_device_val):
        if not cuda_device_val:
            return gr.Markdown.update(value="", visible=False)
        devices = [d.strip() for d in str(cuda_device_val).split(",") if d.strip()]
        if len(devices) > 1 and (cache_dit_val or cache_vae_val):
            return gr.Markdown.update(value="⚠️ Model caching (cache_dit/cache_vae) only works with single GPU. Multi-GPU detected - caching will be auto-disabled.", visible=True)
        return gr.Markdown.update(value="", visible=False)
    
    # Wire up cache validation
    cache_dit.change(
        fn=validate_cache_gpu,
        inputs=[cache_dit, cache_vae, cuda_device],
        outputs=cache_warning
    )
    cache_vae.change(
        fn=validate_cache_gpu,
        inputs=[cache_dit, cache_vae, cuda_device],
        outputs=cache_warning
    )
    cuda_device.change(
        fn=validate_cache_gpu,
        inputs=[cache_dit, cache_vae, cuda_device],
        outputs=cache_warning
    )
    
    # Input detection callback
    def detect_input_type(input_path_val):
        """Detect and display input type information"""
        from shared.input_detector import detect_input
        
        if not input_path_val or not input_path_val.strip():
            return gr.Markdown.update(value="⚠️ No input path provided", visible=True)
        
        try:
            input_info = detect_input(input_path_val)
            
            if not input_info.is_valid:
                return gr.Markdown.update(
                    value=f"❌ **Invalid Input**\n\n{input_info.error_message}",
                    visible=True
                )
            
            # Build info message
            info_lines = [f"### ✅ Input Detected: {input_info.input_type.upper()}\n"]
            info_lines.append(input_info.info_message)
            
            if input_info.input_type == "frame_sequence":
                info_lines.append(f"\n**Frame Pattern:** `{input_info.frame_pattern}`")
                info_lines.append(f"**Frame Range:** {input_info.frame_start} - {input_info.frame_end}")
                if input_info.missing_frames:
                    info_lines.append(f"⚠️ **Missing Frames:** {len(input_info.missing_frames)}")
            elif input_info.input_type == "directory":
                info_lines.append(f"\n**Total Files:** {input_info.total_files}")
                info_lines.append("\n💡 **Tip:** Enable batch processing for multiple files")
            elif input_info.input_type in ["video", "image"]:
                info_lines.append(f"\n**Format:** {input_info.format.upper()}")
            
            result_md = "\n".join(info_lines)
            return gr.Markdown.update(value=result_md, visible=True)
            
        except Exception as e:
            return gr.Markdown.update(
                value=f"❌ **Detection Error**\n\n{str(e)}",
                visible=True
            )
    
    detect_input_btn.click(
        fn=detect_input_type,
        inputs=input_path,
        outputs=input_detection_result
    )
    
    # Auto-detect on input path change with debouncing using Timer
    # Create a timer that triggers detection after 1 second of no changes
    detection_timer = gr.Timer(value=1.0, active=False)
    
    def start_detection_timer(path, state):
        """Restart timer on path change to debounce detection"""
        return gr.Timer(value=1.0, active=True), state
    
    def trigger_detection(path):
        """Called by timer - does actual detection and stops timer"""
        result = detect_input_type(path)
        return result, gr.Timer(value=1.0, active=False)
    
    # Start timer on path change
    input_path.change(
        fn=start_detection_timer,
        inputs=[input_path, shared_state],
        outputs=[detection_timer, shared_state]
    )
    
    # Timer triggers detection and stops itself
    detection_timer.tick(
        fn=trigger_detection,
        inputs=input_path,
        outputs=[input_detection_result, detection_timer]
    )

    # Add batch size validation
    def validate_batch_size_ui(val):
        is_valid, message, corrected = validate_batch_size_seedvr2(val)
        if not is_valid:
            return corrected, gr.Markdown.update(value=f"<span style='color: orange;'>{message}</span>", visible=True)
        return val, gr.Markdown.update(value="", visible=False)
    
    batch_size.change(
        fn=validate_batch_size_ui,
        inputs=batch_size,
        outputs=[batch_size, batch_size_warning]
    )
    
    # Add resolution validation
    def validate_resolution_ui(val):
        is_valid, message, corrected = validate_resolution(val, must_be_multiple_of=16)
        if not is_valid:
            return corrected, gr.Markdown.update(value=f"<span style='color: orange;'>{message}</span>", visible=True)
        return val, gr.Markdown.update(value="", visible=False)
    
    resolution.change(
        fn=validate_resolution_ui,
        inputs=resolution,
        outputs=[resolution, resolution_warning]
    )

    # Automatic chunk estimation when input or chunk settings change
    def auto_estimate_chunks(input_path_val, state):
        """Automatically estimate chunks based on Resolution tab settings and current input"""
        if not input_path_val or not input_path_val.strip():
            return gr.Markdown.update(value="", visible=False), state
        
        # Get chunk settings from Resolution tab (via shared state)
        seed_controls = state.get("seed_controls", {})
        chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
        
        if chunk_size_sec <= 0:
            # PySceneDetect chunking disabled
            return gr.Markdown.update(value="", visible=False), state
        
        # Calculate chunk estimate
        try:
            from shared.path_utils import get_media_duration_seconds, detect_input_type
            
            input_type = detect_input_type(input_path_val)
            if input_type != "video":
                return gr.Markdown.update(value="", visible=False), state
            
            duration = get_media_duration_seconds(input_path_val)
            if not duration or duration <= 0:
                return gr.Markdown.update(value="⚠️ Could not detect video duration for chunk estimation", visible=True), state
            
            chunk_overlap_sec = float(seed_controls.get("chunk_overlap_sec", 0.5) or 0.5)
            
            # Estimate number of chunks
            import math
            effective_chunk_size = max(0.001, chunk_size_sec - chunk_overlap_sec)
            estimated_chunks = math.ceil(duration / effective_chunk_size)
            
            # Build info message
            info_lines = [
                "### 📊 PySceneDetect Chunking Estimate",
                f"**Video Duration:** {duration:.1f}s ({duration/60:.1f} min)",
                f"**Chunk Size:** {chunk_size_sec}s with {chunk_overlap_sec}s overlap",
                f"**Estimated Chunks:** ~{estimated_chunks} scenes/chunks",
                f"**Processing:** Each chunk processed independently, then merged with frame blending",
                "",
                "💡 *Actual chunk count may vary based on scene detection (cuts, transitions)*"
            ]
            
            return gr.Markdown.update(value="\n".join(info_lines), visible=True), state
            
        except Exception as e:
            return gr.Markdown.update(value=f"⚠️ Estimation error: {str(e)[:100]}", visible=True), state
    
    # Wire up automatic chunk estimation
    input_path.change(
        fn=auto_estimate_chunks,
        inputs=[input_path, shared_state],
        outputs=[chunk_estimate_display, shared_state]
    )
    
    # Also update when shared state changes (Resolution tab updates chunk settings)
    shared_state.change(
        fn=auto_estimate_chunks,
        inputs=[input_path, shared_state],
        outputs=[chunk_estimate_display, shared_state]
    )

    # Initialize comparison slider and model status
    comparison_note.update(service["comparison_html_slider"]())
    model_status.update(initialize_model_status())
