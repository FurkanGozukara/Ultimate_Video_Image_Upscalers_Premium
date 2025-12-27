"""
SeedVR2 Tab - Self-contained modular implementation
Following SECourses_Musubi_Trainer pattern

UPDATED: Now uses Universal Preset System for all preset operations.
"""

import gradio as gr
from pathlib import Path
from typing import Dict, Any, List

from shared.services.seedvr2_service import (
    seedvr2_defaults, SEEDVR2_ORDER, build_seedvr2_callbacks
)
from shared.models.seedvr2_meta import get_seedvr2_model_names
from shared.video_comparison_slider import create_video_comparison_html
from shared.ui_validators import validate_batch_size_seedvr2
from ui.universal_preset_section import (
    universal_preset_section,
    wire_universal_preset_events,
    get_tab_values_from_state,
)
from shared.universal_preset import dict_to_values
from ui.media_preview import preview_updates


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
    
    UPDATED: Uses Universal Preset System - settings loaded from shared_state.
    """
    # Get defaults
    defaults = seedvr2_defaults()
    default_model = defaults.get("dit_model")
    
    # Import migration and guardrails for preset loading
    from shared.services.seedvr2_service import _enforce_seedvr2_guardrails
    
    # =========================================================================
    # UNIVERSAL PRESET: Load values from shared_state (populated on startup)
    # =========================================================================
    seed_controls = shared_state.value.get("seed_controls", {})
    seedvr2_settings = seed_controls.get("seedvr2_settings", {})
    current_preset_name = seed_controls.get("current_preset_name")
    models_list = seed_controls.get("available_models", ["default"])
    
    # Merge with defaults to ensure all keys exist
    merged_defaults = defaults.copy()
    for key, value in seedvr2_settings.items():
        if value is not None:
            merged_defaults[key] = value
    
    # Apply guardrails
    merged_defaults = _enforce_seedvr2_guardrails(merged_defaults, defaults, state=None, silent_migration=True)
    
    # Ensure ALL keys from SEEDVR2_ORDER exist with valid values
    for key in SEEDVR2_ORDER:
        if key not in merged_defaults or merged_defaults[key] is None:
            merged_defaults[key] = defaults[key]
    
    values = [merged_defaults[k] for k in SEEDVR2_ORDER]
    
    # Log preset status
    if current_preset_name:
        def update_status(state):
            existing = state["health_banner"]["text"]
            msg = f"✅ SeedVR2: Using universal preset '{current_preset_name}'"
            state["health_banner"]["text"] = existing + "\n" + msg if existing else msg
            return state
        shared_state.value = update_status(shared_state.value)

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
        # IMPORTANT: Do NOT import torch here. Using torch.cuda in the parent Gradio
        # process can create a persistent CUDA context (~500MB VRAM) that never goes
        # away until the UI exits. We use NVML (`nvidia-smi`) via shared.gpu_utils instead.
        from shared.gpu_utils import get_gpu_info

        gpus = [] if is_macos else get_gpu_info()
        cuda_count = len(gpus)
        cuda_available = (not is_macos) and cuda_count > 0

        if is_macos:
            gpu_hint = "macOS detected - CUDA device selection disabled (not supported by SeedVR2 CLI)"
            compile_available = False
        else:
            if cuda_available:
                gpu_hint = f"✅ Detected {cuda_count} CUDA GPU(s) - GPU acceleration available"
                # Check if VS Build Tools available for compile
                from shared.health import is_vs_build_tools_available
                compile_available = is_vs_build_tools_available()
                if not compile_available:
                    gpu_hint += "\n⚠️ VS Build Tools not detected - torch.compile will be disabled"
            else:
                gpu_hint = "⚠️ CUDA not detected (nvidia-smi unavailable or no NVIDIA GPU) - GPU-only features disabled. Processing will use CPU (significantly slower)"
                compile_available = False

        # Append ffmpeg status to GPU hint
        if not ffmpeg_available:
            gpu_hint = f"❌ CRITICAL: ffmpeg not found\n{gpu_hint}"

    except Exception as e:
        gpu_hint = f"❌ CUDA detection failed: {str(e)}"
        cuda_available = False
        cuda_count = 0
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

                with gr.Row():
                    input_file = gr.File(
                        label="Upload video or image (optional)",
                        type="filepath",
                        file_types=["video", "image"]
                    )
                    with gr.Column():
                        input_image_preview = gr.Image(
                            label="📸 Input Preview (Image)",
                            type="filepath",
                            interactive=False,
                            height=250,
                            visible=False,
                        )
                        input_video_preview = gr.Video(
                            label="🎬 Input Preview (Video)",
                            interactive=False,
                            height=250,
                            visible=False,
                        )
                
                # Status messages (compact display, auto-detected on upload/path change)
                input_cache_msg = gr.Markdown("", visible=False)
                
                # Resolution calculation display (compact, single line)
                auto_res_msg = gr.Markdown("", visible=False, elem_classes=["resolution-info"])
                
                # Auto-detection results (auto-triggered, no manual button needed)
                input_detection_result = gr.Markdown("", visible=False)

            # NOTE: 🎬 SeedVR2 Native Streaming (Advanced) moved to right column (above ℹ️ Processing Mode).

            # Output controls (Output Override / Output Format) moved to the right column
            # above the utility buttons for quick access before/after runs.

            # Model selection is placed above Batch Size (see below).

            # -----------------------------------------------------------------
            # ✅ New sizing controls (Upscale-x)
            # -----------------------------------------------------------------
            # Keep the legacy short-side resolution value for backward compatibility
            # with old presets, but hide it from the UI. The runtime `resolution`
            # passed to SeedVR2 CLI is computed from `upscale_factor` + input size.
            resolution = gr.Number(
                label="(Legacy) Target Resolution (short side) [internal]",
                value=values[9],  # Kept for old presets; NOT used for sizing anymore
                precision=0,
                visible=False,
                interactive=False,
            )

            # NOTE: Upscale-x sizing controls (Upscale x / Max Resolution / Pre-downscale)
            # are defined in the right column directly above 📋 Run Log for quick access.

            # Core processing parameters
            # SeedVR2 Model selection (placed just above Batch Size)
            available_models = get_seedvr2_model_names()
            
            # Ensure we always have at least one model choice (fallback to placeholder)
            if not available_models:
                available_models = ["seedvr2_ema_7b_fp16.safetensors"]  # Placeholder if no models found
                dit_model_value = available_models[0]
            else:
                dit_model_value = values[4] if len(values) > 4 else available_models[0]
                if dit_model_value not in available_models:
                    dit_model_value = available_models[0]  # Fallback to first available model
            
            dit_model = gr.Dropdown(
                label="SeedVR2 Model",
                choices=available_models,
                value=dit_model_value,
                info="3B models are faster, 7B models higher quality. 'sharp' variants enhance edges. fp16 recommended for best speed/quality balance.",
            )
            model_cache_msg = gr.Markdown("", visible=False)

            with gr.Row():
                batch_size = gr.Slider(
                    label="Batch Size (must be 4n+1: 5, 9, 13, 17...)",
                    minimum=5,
                    maximum=201,
                    step=4,
                    value=values[11],  # Was 14, now 11 (shift -3)
                    info="SeedVR2 requires batch size to follow 4n+1 formula (5, 9, 13, 17, 21...)",
                    scale=3,
                )
                uniform_batch_size = gr.Checkbox(
                    label="Uniform Batch Size",
                    value=values[12],  # Was 15, now 12
                    info="Force all batches to same size by padding. Improves compilation efficiency but may use more memory. Recommended ON with torch.compile.",
                    scale=1,
                )
            batch_size_warning = gr.Markdown("", visible=False)

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

            # Color correction - validate value
            color_correction_value = values[18] if len(values) > 18 else "lab"
            if color_correction_value not in ["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"]:
                color_correction_value = "lab"
            
            with gr.Row():
                seed = gr.Number(
                    label="Seed",
                    value=values[13],  # Was 16, now 13
                    precision=0,
                    info="Random seed for reproducible results. Same seed + settings = identical output. -1 or 0 = random. Try 42 for consistent testing.",
                    scale=1,
                )
                color_correction = gr.Dropdown(
                    label="Color Correction",
                    choices=["lab", "wavelet", "wavelet_adaptive", "hsv", "adain", "none"],
                    value=color_correction_value,
                    info="Method for maintaining color accuracy. 'lab' is default and robust. 'wavelet' preserves details better. 'none' for creative control.",
                    scale=2,
                )

            # Noise controls
            with gr.Row():
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
            
            with gr.Row():
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
            # FIXED: Live CUDA validation feedback (validates device IDs immediately on change)
            cuda_device_warning = gr.Markdown("", visible=False)
            with gr.Row():
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
            with gr.Row():
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
            with gr.Row():
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
            with gr.Row():
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
            
            # Validate tile_debug value
            tile_debug_value = values[33] if len(values) > 33 else "false"
            if tile_debug_value not in ["false", "encode", "decode"]:
                tile_debug_value = "false"
            
            tile_debug = gr.Dropdown(
                label="Tile Debug",
                choices=["false", "encode", "decode"],
                value=tile_debug_value,
                info="Debug tiling process. 'false' = normal operation. Use 'encode'/'decode' to save intermediate tiles for troubleshooting."
            )

            # Performance & Compile (moved here: directly above Output & Metadata)
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
            
            # Validate attention_mode value before using it
            attention_value = values[34] if len(values) > 34 else "sdpa"
            if attention_value not in ["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"]:
                attention_value = "sdpa"  # Fallback to default if invalid
            
            # Layout: put Attention + Debug on same row (requested)
            with gr.Row():
                attention_mode = gr.Dropdown(
                    label="Attention Backend",
                    choices=["sdpa", "flash_attn_2", "flash_attn_3", "sageattn_2", "sageattn_3"],
                    value=attention_value,
                    info="sdpa (default, most compatible) | flash_attn_2/3 (faster, requires flash-attn install) | sageattn_2/3 (optimized, requires sageattention install). Auto-falls back to sdpa if unavailable.",
                    interactive=cuda_available,  # Disable if no CUDA (attention backends need GPU)
                    scale=3,
                )
                debug = gr.Checkbox(
                    label="Debug Logging",
                    value=values[45],  # Was 48, now 45
                    info="Enable detailed debug output. Useful for troubleshooting but creates verbose logs. Enable if encountering errors.",
                    scale=1,
                )

            # Compile DiT / Compile VAE / Compile Backend in same row (requested)
            # Validate compile_backend value
            compile_backend_value = values[37] if len(values) > 37 else "inductor"
            if compile_backend_value not in ["inductor", "cudagraphs"]:
                compile_backend_value = "inductor"
            
            with gr.Row():
                compile_dit = gr.Checkbox(
                    label="Compile DiT",
                    value=values[35] if compile_available else False,  # Force False if compile unavailable
                    info="Use torch.compile for DiT model. 2-3x faster after warmup. Requires VS Build Tools on Windows. First run is slow.",
                    interactive=compile_available,  # Disable if compile not available
                    scale=1,
                )
                compile_vae = gr.Checkbox(
                    label="Compile VAE",
                    value=values[36] if compile_available else False,  # Force False if compile unavailable
                    info="Use torch.compile for VAE model. Significant speedup for decoding. Requires VS Build Tools on Windows.",
                    interactive=compile_available,  # Disable if compile not available
                    scale=1,
                )
                compile_backend = gr.Dropdown(
                    label="Compile Backend",
                    choices=["inductor", "cudagraphs"],
                    value=compile_backend_value,
                    info="'inductor' is default and most compatible. 'cudagraphs' may be faster but less flexible. Use inductor unless you know what you're doing.",
                    interactive=compile_available,
                    scale=2,
                )
            
            # Validate compile_mode value
            compile_mode_value = values[38] if len(values) > 38 else "default"
            if compile_mode_value not in ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]:
                compile_mode_value = "default"
            
            # Compile Mode / Fullgraph / Dynamic in same row (requested)
            with gr.Row():
                compile_mode = gr.Dropdown(
                    label="Compile Mode",
                    choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
                    value=compile_mode_value,
                    info="'default' balanced. 'reduce-overhead' faster warmup. 'max-autotune' best performance but slow compilation. Start with default.",
                    interactive=compile_available,
                    scale=2,
                )
                compile_fullgraph = gr.Checkbox(
                    label="Compile Fullgraph",
                    value=values[39] if compile_available else False,
                    info="Compile entire model graph at once. May fail on complex models. Leave unchecked unless you need maximum performance.",
                    interactive=compile_available,
                    scale=1,
                )
                compile_dynamic = gr.Checkbox(
                    label="Compile Dynamic Shapes",
                    value=values[40] if compile_available else False,
                    info="Support varying input shapes with compilation. Slower compilation but more flexible. Enable if processing mixed resolutions.",
                    interactive=compile_available,
                    scale=1,
                )

            # Dynamo cache limits + caching toggles in same row (requested)
            with gr.Row():
                compile_dynamo_cache_size_limit = gr.Number(
                    label="Compile Dynamo Cache Size Limit",
                    value=values[41],  # Was 44, now 41
                    precision=0,
                    info="Max cached compiled graphs. Higher = more memory but fewer recompilations. Default 64 is good for most cases.",
                    interactive=compile_available,
                    scale=2,
                )
                compile_dynamo_recompile_limit = gr.Number(
                    label="Compile Dynamo Recompile Limit",
                    value=values[42],  # Was 45, now 42
                    precision=0,
                    info="Max recompilations allowed. Prevents infinite recompile loops. Default 128 is safe. Lower if compilation is slow.",
                    interactive=compile_available,
                    scale=2,
                )
                cache_dit = gr.Checkbox(
                    label="Cache DiT (single GPU only)",
                    value=values[43] if (cuda_available and compile_available) else False,
                    info="Keep DiT model in CUDA graphs cache for maximum speed. Only works with single GPU. Significant speedup for repeated processing.",
                    interactive=(cuda_available and compile_available),
                    scale=1,
                )
                cache_vae = gr.Checkbox(
                    label="Cache VAE (single GPU only)",
                    value=values[44] if (cuda_available and compile_available) else False,
                    info="Keep VAE model in CUDA graphs cache. Single GPU only. Faster encoding/decoding at cost of higher baseline VRAM usage.",
                    interactive=(cuda_available and compile_available),
                    scale=1,
                )
            
            # Cache warning for multi-GPU
            cache_warning = gr.Markdown("", visible=False)

            # Output & Metadata controls (shared settings from Output tab)
            gr.Markdown("---")
            gr.Markdown("#### 📤 Output & Metadata")
            
            with gr.Row():
                save_metadata = gr.Checkbox(
                    label="Save Processing Metadata",
                    value=values[47] if len(values) > 47 else True,
                    info="Save run metadata (settings, timestamps, status) to output folder. Respects global telemetry toggle.",
                    scale=1,
                )
                
                fps_override = gr.Number(
                    label="FPS Override (0 = use source FPS)",
                    value=values[48] if len(values) > 48 else 0,
                    precision=2,
                    info="Override output video FPS. 0 = preserve original FPS, 30 = force 30fps, 60 = force 60fps. Also sourced from Output tab if set.",
                    scale=2,
                )
            
            # ADDED v2.5.22: FFmpeg 10-bit encoding for reduced banding
            gr.Markdown("#### 🎨 Video Encoding (v2.5.22+)")
            gr.Markdown("""
            **NEW: 10-bit Color Depth Support**
            
            SeedVR2 v2.5.22 adds FFmpeg backend with 10-bit encoding (x265 yuv420p10le).
            Benefits over default 8-bit OpenCV:
            - **Reduced banding** in smooth gradients (skies, shadows)
            - **Better color accuracy** for high-quality sources
            - **Smaller file size** at same quality (x265 compression)
            
            ⚠️ Note: 10-bit requires `ffmpeg` backend (disables OpenCV).
            """)
            
            with gr.Row():
                # Validate video_backend value before using it
                backend_value = values[49] if len(values) > 49 else "opencv"
                if backend_value not in ["opencv", "ffmpeg"]:
                    backend_value = "opencv"  # Fallback to default if invalid
                
                video_backend = gr.Dropdown(
                    label="Video Backend",
                    choices=["opencv", "ffmpeg"],
                    value=backend_value,
                    info="opencv: Fast 8-bit encoding (libx264) | ffmpeg: Subprocess-based, enables 10-bit support"
                )
                
                # Validate use_10bit value before using it
                use_10bit_value = values[50] if len(values) > 50 else False
                if not isinstance(use_10bit_value, bool):
                    use_10bit_value = False  # Fallback to default if invalid
                
                use_10bit = gr.Checkbox(
                    label="Enable 10-bit Encoding",
                    value=use_10bit_value,
                    info="Use x265 with yuv420p10le for 10-bit color depth. Requires 'ffmpeg' backend. Reduces banding in gradients."
                )
            
            video_backend_warning = gr.Markdown("", visible=False)

            # Input path textbox (direct path alternative to upload)
            gr.Markdown("---")
            gr.Markdown("#### 📁 Direct Input Path (Alternative to Upload)")
            input_path = gr.Textbox(
                label="Input Video or Frames Folder Path",
                value=values[0],
                placeholder="C:/path/to/video.mp4 or C:/path/to/frames/",
                info="Enter path to either a video file (mp4, avi, mov, etc.) or folder containing image frames (jpg, png, tiff, etc.). Automatically detected - works on Windows and Linux."
            )

            # Model directory override (optional). SeedVR2 Model selection is above Batch Size.
            model_dir = gr.Textbox(
                label="Model Directory (optional)",
                value=values[3],
                info="Override default model directory. Leave empty to use default ./SeedVR2/models location."
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
            
            # Upscale factor + action buttons (placed directly above 📋 Run Log)
            with gr.Group():
                _upscale_factor_default = merged_defaults.get("upscale_factor", 4.0) or 4.0
                try:
                    _upscale_factor_default = float(_upscale_factor_default)
                except (TypeError, ValueError):
                    _upscale_factor_default = 4.0
                _upscale_factor_default = min(9.9, max(1.0, _upscale_factor_default))

                _max_resolution_default = values[10]
                try:
                    _max_resolution_default = int(_max_resolution_default)
                except (TypeError, ValueError):
                    _max_resolution_default = 0
                _max_resolution_default = min(8192, max(0, _max_resolution_default))

                with gr.Row():
                    upscale_factor = gr.Slider(
                        label="Upscale x (any factor)",
                        minimum=1.0,
                        maximum=9.9,
                        step=0.1,
                        value=_upscale_factor_default,
                        info="e.g., 4.0 = 4x. Target size is computed from input, then capped by Max Resolution (max edge).",
                        scale=2,
                    )

                    max_resolution = gr.Slider(
                        label="Max Resolution (max edge, 0 = no cap)",
                        minimum=0,
                        maximum=8192,
                        step=16,
                        value=_max_resolution_default,
                        info="Caps the LONG side (max(width,height)) of the target. 0 = unlimited.",
                        scale=2,
                    )

                    pre_downscale_then_upscale = gr.Checkbox(
                        label="⬇️➡️⬆️ Pre-downscale then upscale (when capped)",
                        value=bool(merged_defaults.get("pre_downscale_then_upscale", False)),
                        info="If max edge would reduce effective scale, downscale input first so the model still runs at the full Upscale x.",
                        scale=1,
                    )

                # 2-row action layout:
                # Row 1: Upscale + Preview
                with gr.Row():
                    upscale_btn = gr.Button(
                        "🚀 Upscale" if ffmpeg_available else "❌ Upscale (ffmpeg required)",
                        variant="primary" if ffmpeg_available else "stop",
                        size="lg",
                        interactive=ffmpeg_available,
                    )
                    preview_btn = gr.Button(
                        "👁️ First-frame Preview" if ffmpeg_available else "❌ Preview (ffmpeg required)",
                        size="lg",
                        interactive=ffmpeg_available,
                    )

                # Row 2: Confirm cancel + Cancel
                with gr.Row():
                    cancel_confirm = gr.Checkbox(
                        label="⚠️ Confirm cancel (subprocess mode only)",
                        value=False,
                        info="Cancel only works in subprocess mode. Check Global Settings to verify mode.",
                        scale=3,
                    )
                    cancel_btn = gr.Button(
                        "⏹️ Cancel (subprocess only)",
                        variant="stop",
                        size="sm",
                        scale=0,
                    )
            
            log_box = gr.Textbox(
                label="📋 Run Log",
                value="",
                lines=12,
                buttons=["copy"]
            )

            # Output display with enhanced comparison
            output_video = gr.Video(
                label="🎬 Upscaled Video",
                interactive=False,
                buttons=["download"]
            )
            output_image = gr.Image(
                label="🖼️ Upscaled Image / Preview",
                interactive=False,
                buttons=["download"]
            )
            
            # Gallery for batch results
            batch_gallery = gr.Gallery(
                label="📦 Batch Results",
                visible=False,
                columns=3,
                rows=2,
                height="auto",
                object_fit="contain",
                buttons=["download"]
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

            # Output controls (single run) - placed above utility buttons
            # Validate output_format value
            output_format_value = values[2] if len(values) > 2 else "auto"
            if output_format_value not in ["auto", "mp4", "png"]:
                output_format_value = "auto"

            with gr.Row():
                output_override = gr.Textbox(
                    label="Output Override (single run)",
                    value=values[1],
                    placeholder="Leave empty for auto naming",
                    info="Specify custom output path. Auto-naming creates '_upscaled' files in output folder. Supports both file paths and directories.",
                    scale=3,
                )
                output_format = gr.Dropdown(
                    label="Output Format",
                    choices=["auto", "mp4", "png"],
                    value=output_format_value,
                    info="'auto' chooses based on input type. 'mp4' for video output. 'png' exports frame sequence. Note: MP4 drops alpha channels.",
                    scale=1,
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
            
            # Chunk thumbnail gallery - Shows completed chunks as they finish
            with gr.Accordion("🎬 Completed Chunks Gallery", open=False):
                gr.Markdown("*Thumbnails of completed chunks appear here during processing*")
                chunk_gallery = gr.Gallery(
                    label="Completed Chunks (updates as processing progresses)",
                    visible=False,
                    columns=4,
                    rows=2,
                    height="auto",
                    object_fit="contain",
                    buttons=[],
                    allow_preview=True
                )

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

            # =========================================================================
            # UNIVERSAL PRESET MANAGEMENT - Same UI across ALL tabs
            # =========================================================================
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
                tab_name="seedvr2",
                inputs_list=[],  # Will be set after inputs_list is defined
                base_dir=base_dir,
                models_list=models_list,
                open_accordion=True,
            )

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
                unload_model_btn = gr.Button("🧹 Clear CUDA Cache", variant="secondary", size="lg")
                unload_all_models_btn = gr.Button("🧹 Clear All CUDA Caches", variant="stop", size="lg")
            model_unload_status = gr.Markdown("", visible=False)

            # Timer for periodic model status updates
            model_status_timer = gr.Timer(value=2.0, active=False)  # Update every 2 seconds when active

            # 🎬 SeedVR2 Native Streaming (Advanced) - moved here (must be immediately above ℹ️ Processing Mode)
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
                check_resume_btn = gr.Button("🔍 Check Resume Status", size="lg")

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
    # Current count: len(SEEDVR2_ORDER) = 53, len(inputs_list) must also = 53
    # (includes: save_metadata, fps_override, video_backend, use_10bit)
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
        save_metadata, fps_override,
        # ADDED v2.5.22: FFmpeg 10-bit encoding support
        video_backend, use_10bit,
        # vNext sizing
        upscale_factor, pre_downscale_then_upscale
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
            return gr.update(value=f"### 🔧 Model Status\n{status_text}")
        except Exception:
            return gr.update(value="### 🔧 Model Status\nStatus unavailable")

    # Wire up all the event handlers

    # Input handling with sizing info updates (Upscale-x)
    def cache_path_value(val, state):
        """Cache input path and refresh sizing info panel."""
        state["seed_controls"]["last_input_path"] = val if val else ""
        if val and str(val).strip():
            try:
                calc_msg, updated_state = service["auto_res_on_input"](val, state)
                if isinstance(calc_msg, dict):
                    calc_msg["visible"] = True
                else:
                    calc_msg = gr.update(value=str(calc_msg), visible=True)
                return gr.update(visible=False), updated_state, calc_msg
            except Exception as e:
                return gr.update(value=f"✅ Input cached (info error: {str(e)[:80]})", visible=True), state, gr.update(visible=False)
        # If input is empty, hide panels (clearing input should clear this info).
        return gr.update(value="", visible=False), state, gr.update(value="", visible=False)

    def cache_upload(val, state):
        """Cache uploaded file path and refresh sizing info panel."""
        state["seed_controls"]["last_input_path"] = val if val else ""
        if val:
            try:
                calc_msg, updated_state = service["auto_res_on_input"](val, state)
                if isinstance(calc_msg, dict):
                    calc_msg["visible"] = True
                else:
                    calc_msg = gr.update(value=str(calc_msg), visible=True)
                return val or "", gr.update(visible=False), updated_state, calc_msg
            except Exception as e:
                return val or "", gr.update(value=f"✅ File uploaded (info error: {str(e)[:80]})", visible=True), state, gr.update(visible=False)
        # If upload is cleared, hide panels (don’t show a persistent message).
        return val or "", gr.update(value="", visible=False), state, gr.update(value="", visible=False)

    # Wire up input events with resolution auto-calculation
    input_file.upload(
        fn=lambda val, state: cache_upload(val, state),
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, shared_state, auto_res_msg]
    )

    input_path.change(
        fn=lambda val, state: cache_path_value(val, state),
        inputs=[input_path, shared_state],
        outputs=[input_cache_msg, shared_state, auto_res_msg]
    )
    
    # Recalculate sizing info when any sizing control changes
    def recalculate_sizing_info(scale_x, max_res_val, pre_down, state):
        try:
            scale_f = float(scale_x) if scale_x is not None else None
            max_i = int(max_res_val) if max_res_val is not None else None
        except (ValueError, TypeError):
            return gr.update(), state

        if scale_f is None or scale_f <= 0:
            return gr.update(), state
        if max_i is None or max_i < 0 or max_i > 8192:
            return gr.update(), state

        # Cache globally (used by other tabs/services)
        state["seed_controls"]["upscale_factor_val"] = scale_f
        state["seed_controls"]["max_resolution_val"] = max_i
        state["seed_controls"]["ratio_downscale"] = bool(pre_down)
        # vNext UX: a non-zero max_resolution means "cap enabled". Ensure the legacy flag can't block it.
        if max_i and max_i > 0:
            state["seed_controls"]["enable_max_target"] = True

        input_path_val = state.get("seed_controls", {}).get("last_input_path", "")
        if input_path_val:
            try:
                calc_msg, updated_state = service["auto_res_on_input"](input_path_val, state)
                if isinstance(calc_msg, dict):
                    calc_msg["visible"] = True
                else:
                    calc_msg = gr.update(value=str(calc_msg), visible=True)
                return calc_msg, updated_state
            except Exception:
                return gr.update(visible=False), state
        return gr.update(visible=False), state

    upscale_factor.change(
        fn=recalculate_sizing_info,
        inputs=[upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[auto_res_msg, shared_state],
        trigger_mode="always_last",
    )

    pre_downscale_then_upscale.change(
        fn=recalculate_sizing_info,
        inputs=[upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[auto_res_msg, shared_state],
        trigger_mode="always_last",
    )

    max_resolution.release(
        fn=recalculate_sizing_info,
        inputs=[upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[auto_res_msg, shared_state],
        preprocess=False,
        trigger_mode="always_last",
    )

    # Also respond to typed value changes (release may not fire when user edits the number field directly)
    max_resolution.change(
        fn=recalculate_sizing_info,
        inputs=[upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[auto_res_msg, shared_state],
        trigger_mode="always_last",
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
            
            model_status_update = gr.update(value=f"### 🔧 Model Status\n{status_text}")
        except Exception as e:
            model_status_update = gr.update(value=f"### 🔧 Model Status\nError: {str(e)}")
        
        # If last-used preset exists, merge with current values
        if last_used_preset:
            current_dict = dict(zip(SEEDVR2_ORDER, current_vals))
            merged = preset_manager.merge_config(current_dict, last_used_preset)
            
            # Apply model constraints to merged preset
            if not compile_supported:
                merged["compile_dit"] = False
                merged["compile_vae"] = False
            
            new_vals = [merged[k] for k in SEEDVR2_ORDER]
            cache_msg = gr.update(
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
            
            cache_msg = gr.update(
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
            return gr.update(value=f"### 🔧 Model Status\n{status_text}")
        except Exception:
            return gr.update(value="### 🔧 Model Status\nStatus unavailable")

    # Add a refresh button for model status
    with gr.Row():
        refresh_model_status_btn = gr.Button("🔄 Refresh Model Status", size="lg", variant="secondary")
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
            
            return gr.update(value=msg, visible=True), service["get_model_loading_status"]()
        except Exception as e:
            return gr.update(value=f"❌ Error: {str(e)}", visible=True), service["get_model_loading_status"]()
    
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
            
            return gr.update(value=msg, visible=True), service["get_model_loading_status"]()
        except Exception as e:
            return gr.update(value=f"❌ Error: {str(e)}", visible=True), service["get_model_loading_status"]()
    
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
        fn=lambda fmt: service["check_resume_status"](global_settings, fmt),
        inputs=[output_format],
        outputs=resume_status
    )

    # Wrapper functions for generator support in Gradio 6.2.0
    def run_upscale_wrapper(*args, progress=gr.Progress()):
        """Wrapper to properly handle generator function for Gradio 6.2.0"""
        # Call the generator function and yield from it
        yield from service["run_action"](*args[:-1], preview_only=False, state=args[-1], progress=progress)
    
    def run_preview_wrapper(*args, progress=gr.Progress()):
        """Wrapper to properly handle generator function for Gradio 6.2.0"""
        # Call the generator function and yield from it
        yield from service["run_action"](*args[:-1], preview_only=True, state=args[-1], progress=progress)
    
    # Main action buttons with gr.Progress
    upscale_btn.click(
        fn=run_upscale_wrapper,
        inputs=[input_file, face_restore_chk] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_video, output_image,
            chunk_info, resume_status, chunk_progress, comparison_note, image_slider, video_comparison_html, chunk_gallery, batch_gallery, shared_state
        ]
    )

    preview_btn.click(
        fn=run_preview_wrapper,
        inputs=[input_file, face_restore_chk] + inputs_list + [shared_state],
        outputs=[
            status_box, log_box, progress_indicator, output_video, output_image,
            chunk_info, resume_status, chunk_progress, comparison_note, image_slider, video_comparison_html, chunk_gallery, batch_gallery, shared_state
        ]
    )

    # Cancel button with confirmation requirement
    def handle_cancel_with_confirmation(confirm_checked, state):
        """Handle cancel with explicit confirmation check"""
        if not confirm_checked:
            return (
                gr.update(value="⚠️ Cancellation not confirmed. Enable 'Confirm cancel' checkbox and click again."),
                gr.update(value="Cancellation requires confirmation. Please enable the checkbox above."),
                state
            )
        
        # User confirmed - proceed with cancellation
        cancel_result = service["cancel_action"]()
        return (
            gr.update(value="⏹️ Cancellation requested. Subprocess will be terminated."),
            gr.update(value="Cancellation signal sent. Process terminating..."),
            state
        )
    
    cancel_btn.click(
        fn=handle_cancel_with_confirmation,
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

    # =========================================================================
    # UNIVERSAL PRESET EVENT WIRING
    # =========================================================================
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
        tab_name="seedvr2",
    )
    
    # Recalculate sizing info after preset changes (upscale/max/pre-downscale may have changed)
    def recalc_after_preset_change(scale_x, max_res_val, pre_down, state):
        """Recalculate sizing info after preset load/reset"""
        return recalculate_sizing_info(scale_x, max_res_val, pre_down, state)
    
    # Trigger recalculation when resolution values change from preset loading
    load_preset_btn.click(
        fn=recalc_after_preset_change,
        inputs=[upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[auto_res_msg, shared_state]
    ).then(
        fn=None,  # Just trigger the update
        inputs=None,
        outputs=None
    )
    
    reset_defaults_btn.click(
        fn=recalc_after_preset_change,
        inputs=[upscale_factor, max_resolution, pre_downscale_then_upscale, shared_state],
        outputs=[auto_res_msg, shared_state]
    ).then(
        fn=None,
        inputs=None,
        outputs=None
    )

    # Update health display from shared state
    def update_health_display(state):
        health_text = state.get("health_banner", {}).get("text", "")
        return gr.update(value=health_text)

    shared_state.change(
        fn=update_health_display,
        inputs=shared_state,
        outputs=health_display
    )
    
    # Input preview (image + video) for Gradio 6.2.0
    def update_input_previews(path_val):
        return preview_updates(path_val)

    input_file.change(
        fn=update_input_previews,
        inputs=[input_file],
        outputs=[input_image_preview, input_video_preview],
    )

    # When the user clicks the “X” to clear the upload, also clear derived info panels + textbox path.
    def clear_input_panels_on_upload_clear(file_path, state):
        # Only act when cleared.
        if file_path:
            return gr.update(), gr.update(), gr.update(), gr.update(), state
        try:
            state = state or {}
            state.setdefault("seed_controls", {})
            state["seed_controls"]["last_input_path"] = ""
        except Exception:
            pass
        return (
            "",  # input_path
            gr.update(value="", visible=False),  # input_cache_msg
            gr.update(value="", visible=False),  # auto_res_msg
            gr.update(value="", visible=False),  # input_detection_result
            state,
        )

    input_file.change(
        fn=clear_input_panels_on_upload_clear,
        inputs=[input_file, shared_state],
        outputs=[input_path, input_cache_msg, auto_res_msg, input_detection_result, shared_state],
    )

    # Also preview when user types/pastes a path (not only uploads)
    input_path.change(
        fn=update_input_previews,
        inputs=[input_path],
        outputs=[input_image_preview, input_video_preview],
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
            return gr.update(visible=True)
        return gr.update(visible=False)

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
            return gr.update(visible=False)

        from shared.path_utils import get_media_fps, detect_input_type
        try:
            input_type = detect_input_type(input_path_val)
            if input_type == "video":
                fps = get_media_fps(input_path_val)
                if fps is None or fps <= 0:
                    return gr.update(visible=True)
        except Exception:
            return gr.update(visible=True)

        return gr.update(visible=False)

    input_path.change(
        fn=check_fps_metadata,
        inputs=[input_path],
        outputs=fps_warn
    )

    # Tile validation helpers
    def validate_tile_encode(tile_size, overlap):
        if tile_size > 0 and overlap >= tile_size:
            return gr.update(value=f"⚠️ Encode tile overlap ({overlap}) must be < tile size ({tile_size}). Will be auto-corrected.", visible=True)
        return gr.update(value="", visible=False)
    
    def validate_tile_decode(tile_size, overlap):
        if tile_size > 0 and overlap >= tile_size:
            return gr.update(value=f"⚠️ Decode tile overlap ({overlap}) must be < tile size ({tile_size}). Will be auto-corrected.", visible=True)
        return gr.update(value="", visible=False)
    
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
    
    # Validate 10-bit encoding requirements
    def validate_10bit_settings(backend, use_10bit_val):
        """Validate that 10-bit encoding has required backend"""
        if use_10bit_val and backend != "ffmpeg":
            return gr.update(
                value="⚠️ 10-bit encoding requires 'ffmpeg' backend. Please select 'ffmpeg' from Video Backend dropdown or disable 10-bit encoding.",
                visible=True
            )
        elif use_10bit_val and backend == "ffmpeg":
            return gr.update(
                value="✅ 10-bit encoding enabled: Using x265 codec with yuv420p10le pixel format for reduced banding in gradients.",
                visible=True
            )
        return gr.update(value="", visible=False)
    
    # Wire up 10-bit validation
    video_backend.change(
        fn=validate_10bit_settings,
        inputs=[video_backend, use_10bit],
        outputs=video_backend_warning
    )
    use_10bit.change(
        fn=validate_10bit_settings,
        inputs=[video_backend, use_10bit],
        outputs=video_backend_warning
    )
    
    # FIXED: Live CUDA device validation with comprehensive checks
    cuda_device_warning = gr.Markdown("", visible=False)
    
    def validate_cuda_device_live(cuda_device_val):
        """
        Live CUDA device validation - warns user immediately on input change.
        
        Checks:
        - Device ID validity (must be numeric)
        - Device availability (checks against torch.cuda.device_count())
        - Multi-GPU compatibility warnings
        """
        if not cuda_device_val or not cuda_device_val.strip():
            return gr.update(value="", visible=False)
        
        # IMPORTANT: No torch import here (keeps parent process VRAM at 0).
        try:
            if not cuda_available or cuda_count <= 0:
                return gr.update(value="⚠️ CUDA not detected on this system. GPU acceleration disabled.", visible=True)

            # Parse device spec (handle "all" and comma-separated IDs)
            device_str = str(cuda_device_val).strip()
            if device_str.lower() == "all":
                return gr.update(value=f"✅ Using all {cuda_count} available GPU(s)", visible=True)

            devices = [d.strip() for d in device_str.replace(" ", "").split(",") if d.strip()]

            invalid_devices: List[str] = []
            valid_devices: List[int] = []

            for device in devices:
                if not device.isdigit():
                    invalid_devices.append(device)
                    continue
                device_id = int(device)
                if device_id >= cuda_count:
                    invalid_devices.append(f"{device} (max: {cuda_count-1})")
                else:
                    valid_devices.append(device_id)

            if invalid_devices:
                return gr.update(
                    value=f"❌ Invalid CUDA device ID(s): {', '.join(invalid_devices)}. Available devices: 0-{cuda_count-1}",
                    visible=True,
                )

            if len(valid_devices) > 1:
                return gr.update(
                    value=(
                        f"✅ Multi-GPU: Using {len(valid_devices)} GPUs ({', '.join(map(str, valid_devices))})\n"
                        f"⚠️ Note: Multi-GPU disables model caching (cache_dit/cache_vae auto-disabled)"
                    ),
                    visible=True,
                )
            if len(valid_devices) == 1:
                return gr.update(value=f"✅ Single GPU: Using GPU {valid_devices[0]}", visible=True)

            return gr.update(value="", visible=False)
        except Exception as e:
            return gr.update(value=f"⚠️ Validation error: {str(e)}", visible=True)
    
    # Cache + GPU validation (enhanced with comprehensive checks)
    def validate_cache_gpu(cache_dit_val, cache_vae_val, cuda_device_val):
        if not cuda_device_val:
            return gr.update(value="", visible=False)
        devices = [d.strip() for d in str(cuda_device_val).split(",") if d.strip()]
        if len(devices) > 1 and (cache_dit_val or cache_vae_val):
            return gr.update(value="⚠️ Model caching (cache_dit/cache_vae) only works with single GPU. Multi-GPU detected - caching will be auto-disabled.", visible=True)
        return gr.update(value="", visible=False)
    
    # Wire up LIVE CUDA device validation (validates on every change)
    cuda_device.change(
        fn=validate_cuda_device_live,
        inputs=cuda_device,
        outputs=cuda_device_warning
    )
    
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
            # Hide when empty (clearing input should clear this panel).
            return gr.update(value="", visible=False)
        
        try:
            input_info = detect_input(input_path_val)
            
            if not input_info.is_valid:
                return gr.update(
                    value=f"❌ **Invalid Input**\n\n{input_info.error_message}",
                    visible=True
                )
            
            # Build compact info message
            info_parts = [f"✅ **Input Detected: {input_info.input_type.upper()}**"]
            
            if input_info.input_type == "frame_sequence":
                info_parts.append(f"&nbsp;&nbsp;📁 Pattern: `{input_info.frame_pattern}`")
                info_parts.append(f"&nbsp;&nbsp;🎞️ Frames: {input_info.frame_start}-{input_info.frame_end}")
                if input_info.missing_frames:
                    info_parts.append(f"&nbsp;&nbsp;⚠️ Missing: {len(input_info.missing_frames)}")
            elif input_info.input_type == "directory":
                info_parts.append(f"&nbsp;&nbsp;📂 Files: {input_info.total_files}")
            elif input_info.input_type in ["video", "image"]:
                info_parts.append(f"&nbsp;&nbsp;📄 Format: **{input_info.format.upper()}**")
            
            # Single line format for compact display
            result_md = " ".join(info_parts)
            return gr.update(value=result_md, visible=True)
            
        except Exception as e:
            return gr.update(
                value=f"❌ **Detection Error**\n\n{str(e)}",
                visible=True
            )
    
    # Auto-detect on input path change with debouncing using Timer
    # No manual button needed - detection happens automatically
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
            return corrected, gr.update(value=f"<span style='color: orange;'>{message}</span>", visible=True)
        return val, gr.update(value="", visible=False)
    
    batch_size.change(
        fn=validate_batch_size_ui,
        inputs=batch_size,
        outputs=[batch_size, batch_size_warning]
    )
    
    # NOTE: Legacy `resolution` control is hidden; sizing is now driven by Upscale-x.

    # Automatic chunk estimation when input or chunk settings change
    def auto_estimate_chunks(input_path_val, state):
        """Automatically estimate chunks based on Resolution tab settings and current input"""
        if not input_path_val or not input_path_val.strip():
            return gr.update(value="", visible=False), state
        
        # Get chunk settings from Resolution tab (via shared state)
        seed_controls = state.get("seed_controls", {})
        chunk_size_sec = float(seed_controls.get("chunk_size_sec", 0) or 0)
        
        if chunk_size_sec <= 0:
            # PySceneDetect chunking disabled
            return gr.update(value="", visible=False), state
        
        # Calculate chunk estimate
        try:
            from shared.path_utils import get_media_duration_seconds, detect_input_type
            
            input_type = detect_input_type(input_path_val)
            if input_type != "video":
                return gr.update(value="", visible=False), state
            
            duration = get_media_duration_seconds(input_path_val)
            if not duration or duration <= 0:
                return gr.update(value="⚠️ Could not detect video duration for chunk estimation", visible=True), state
            
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
            
            return gr.update(value="\n".join(info_lines), visible=True), state
            
        except Exception as e:
            return gr.update(value=f"⚠️ Estimation error: {str(e)[:100]}", visible=True), state
    
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

    # Note: In Gradio 6.2.0, component.update() is removed
    # Components are initialized with their default values in constructors above
