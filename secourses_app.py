import os
from pathlib import Path
from typing import Any, Dict

import gradio as gr

from shared.models import scan_gan_models
from shared.health import collect_health_report
from shared.logging_utils import RunLogger
from shared.path_utils import get_default_output_dir, get_default_temp_dir
from shared.preset_manager import PresetManager
from shared.runner import Runner
from shared.gradio_compat import check_gradio_version, check_required_features
from ui.seedvr2_tab import seedvr2_tab
from ui.resolution_tab import resolution_tab
from ui.output_tab import output_tab
from ui.face_tab import face_tab
from ui.rife_tab import rife_tab
from ui.gan_tab import gan_tab
from ui.flashvsr_tab import flashvsr_tab
from ui.health_tab import health_tab

BASE_DIR = Path(__file__).parent.resolve()
PRESET_DIR = BASE_DIR / "presets"
APP_TITLE = "SECourses Ultimate Video and Image Upscaler Pro V1.0 – https://www.patreon.com/posts/134405610"


# --------------------------------------------------------------------- #
# Global setup - Honor launcher BAT file environment variables
# --------------------------------------------------------------------- #
preset_manager = PresetManager(PRESET_DIR)

# FIXED: Read ALL launcher BAT settings (TEMP/TMP + model cache paths)
# This ensures user-configured paths from Windows_Run_SECourses_Upscaler_Pro.bat are respected
launcher_temp = os.environ.get("TEMP") or os.environ.get("TMP")
launcher_output = None  # BAT doesn't set OUTPUT_DIR, but we check for future compatibility

# FIXED: Also read model cache paths set by launcher (MODELS_DIR, HF_HOME, etc.)
# These are used by HuggingFace/Transformers for model downloads and caching
launcher_models_dir = os.environ.get("MODELS_DIR")
launcher_hf_home = os.environ.get("HF_HOME")
launcher_transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
launcher_hf_datasets = os.environ.get("HF_DATASETS_CACHE")

# Validate and propagate model cache paths if set by launcher
# This ensures models download to the correct location
if launcher_models_dir and Path(launcher_models_dir).exists():
    # User set MODELS_DIR in launcher - ensure HF libraries use it
    if not launcher_hf_home:
        os.environ["HF_HOME"] = launcher_models_dir
    if not launcher_transformers_cache:
        os.environ["TRANSFORMERS_CACHE"] = launcher_models_dir
    if not launcher_hf_datasets:
        os.environ["HF_DATASETS_CACHE"] = launcher_models_dir

# If BAT file set a custom temp that's NOT the system temp, use it
# This detects if user modified the BAT file's TEMP/TMP settings
system_temp = os.environ.get("SystemRoot", "C:\\Windows") + "\\Temp" if os.name == "nt" else "/tmp"
if launcher_temp and launcher_temp.lower() != system_temp.lower():
    default_temp = launcher_temp
else:
    default_temp = str(BASE_DIR / "temp")

GLOBAL_DEFAULTS = {
    "output_dir": launcher_output or str(BASE_DIR / "outputs"),
    "temp_dir": default_temp,
    "telemetry": True,
    "face_global": False,
    "face_strength": 0.5,
    "mode": "subprocess",
    "mode_locked": False,  # Persisted lock state for in-app mode
    "pinned_reference_path": None,  # Global pinned reference for iterative comparison
    # FIXED: Store model cache paths - editable in UI, persisted across restarts
    "models_dir": launcher_models_dir or str(BASE_DIR / "models"),
    "hf_home": launcher_hf_home or os.environ.get("HF_HOME") or str(BASE_DIR / "models"),
    "transformers_cache": launcher_transformers_cache or os.environ.get("TRANSFORMERS_CACHE") or str(BASE_DIR / "models"),
    # Store originals for change detection (helps warn user about restart requirement)
    "_original_models_dir": launcher_models_dir,
    "_original_hf_home": launcher_hf_home,
    "_original_transformers_cache": launcher_transformers_cache,
}
global_settings = preset_manager.load_global_settings(GLOBAL_DEFAULTS)

# FIXED: Apply saved model cache paths to environment for current session
# If user previously saved custom paths, honor them immediately (though full effect requires restart)
if global_settings.get("models_dir"):
    os.environ["MODELS_DIR"] = global_settings["models_dir"]
if global_settings.get("hf_home"):
    os.environ["HF_HOME"] = global_settings["hf_home"]
if global_settings.get("transformers_cache"):
    os.environ["TRANSFORMERS_CACHE"] = global_settings["transformers_cache"]

temp_dir = get_default_temp_dir(BASE_DIR, global_settings)
output_dir = get_default_output_dir(BASE_DIR, global_settings)
runner = Runner(
    BASE_DIR,
    temp_dir=temp_dir,
    output_dir=output_dir,
    telemetry_enabled=global_settings.get("telemetry", True),
)
# Restore execution mode from saved settings (default to subprocess)
saved_mode = global_settings.get("mode", "subprocess")
mode_locked = global_settings.get("mode_locked", False)
try:
    runner.set_mode(saved_mode)
except Exception:
    runner.set_mode("subprocess")
    global_settings["mode"] = "subprocess"
    global_settings["mode_locked"] = False
run_logger = RunLogger(enabled=global_settings.get("telemetry", True))



# --------------------------------------------------------------------- #
# UI construction
# --------------------------------------------------------------------- #
def main():
    # Initialize health check data
    try:
        initial_report = collect_health_report(temp_dir=temp_dir, output_dir=output_dir)
        warnings = []
        
        # Check Gradio compatibility FIRST (critical for UI)
        gradio_compatible, gradio_msg, gradio_features = check_gradio_version()
        if not gradio_compatible:
            warnings.append(f"⚠️ GRADIO: {gradio_msg}")
        
        required_features, features_msg = check_required_features()
        if not required_features:
            warnings.append(f"⚠️ GRADIO FEATURES: {features_msg}")
        
        # Add mode lock warning if applicable
        if mode_locked and saved_mode == "in_app":
            warnings.append("🔒 IN-APP MODE LOCKED: You are in in-app mode. To switch back to subprocess mode, restart the application.")
        
        for key, info in initial_report.items():
            if info.get("status") not in ("ok", "skipped"):
                warnings.append(f"{key}: {info.get('detail')}")
        # Surface CPU-only ffmpeg reminder and VS Build Tools guidance up front
        warnings.append("ffmpeg runs CPU-only; ensure ffmpeg is in PATH. CUDA ffmpeg is not used.")
        vs_info = initial_report.get("vs_build_tools")
        if vs_info and vs_info.get("status") != "ok":
            warnings.append("VS Build Tools not detected; torch.compile will be disabled on Windows until installed.")
        health_text = "\n".join(warnings) if warnings else "All health checks passed."
    except Exception:
        health_text = "Health check failed to initialize. Run Health Check tab for details."

    # Ultra-modern theme with maximum readability (Gradio 6.2.0)
    # Using Soft theme + Google Fonts for best readability
    modern_theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),  # Most readable UI font
        font_mono=gr.themes.GoogleFont("JetBrains Mono")  # Best code font
    ).set(
        # Enhanced readability settings
        body_text_size="16px",
        body_text_weight="400",
        button_large_text_size="18px",
        button_large_text_weight="600",
        button_large_padding="16px 28px",
        button_border_width="2px",
        button_primary_shadow="0 2px 8px rgba(0,0,0,0.1)",
        button_primary_shadow_hover="0 4px 12px rgba(0,0,0,0.15)",
        input_border_width="2px",
        input_shadow="0 1px 3px rgba(0,0,0,0.05)",
        block_label_text_size="16px",
        block_label_text_weight="600",
        block_title_text_size="18px",
        block_title_text_weight="700",
    )

    # Auto-apply last-used presets on startup for ALL tabs
    # This ensures settings are restored automatically when app starts
    def load_all_startup_presets():
        """
        Load last-used presets for ALL tabs and models into shared state.
        
        COMPLETE IMPLEMENTATION: Now loads SeedVR2, GAN, RIFE, FlashVSR+, Face presets on startup.
        
        Returns tuple: (resolution_settings, resolution_cache, output_settings, output_cache,
                        seedvr2_cache, gan_cache, rife_cache, flashvsr_cache, face_cache)
        """
        from shared.models import get_seedvr2_model_names, scan_gan_models, get_flashvsr_model_names, get_rife_model_names
        from shared.services.resolution_service import resolution_defaults
        from shared.services.output_service import output_defaults
        from shared.services.seedvr2_service import seedvr2_defaults
        from shared.services.gan_service import gan_defaults
        from shared.services.rife_service import rife_defaults
        from shared.services.flashvsr_service import flashvsr_defaults
        from shared.services.face_service import face_defaults
        
        # Get all models for each pipeline
        seedvr2_models = get_seedvr2_model_names()
        gan_models = scan_gan_models(BASE_DIR)
        flashvsr_models = get_flashvsr_model_names()
        rife_models = get_rife_model_names(BASE_DIR)
        
        all_models = []
        all_models.extend(seedvr2_models)
        all_models.extend(gan_models)
        all_models.extend(flashvsr_models)
        all_models.extend(rife_models)
        
        if not all_models:
            all_models = ["default"]
        
        # Pick first available model as primary default for each pipeline
        primary_seedvr2 = seedvr2_models[0] if seedvr2_models else "default"
        primary_gan = gan_models[0] if gan_models else "default"
        primary_rife = rife_models[0] if rife_models else "default"
        primary_flashvsr = flashvsr_models[0] if flashvsr_models else "v10_tiny_4x"
        primary_face = all_models[0]
        
        # === RESOLUTION TAB ===
        primary_res_last_used = preset_manager.load_last_used("resolution", all_models[0])
        res_defaults = resolution_defaults([all_models[0]])
        primary_res_settings = preset_manager.merge_config(res_defaults, primary_res_last_used) if primary_res_last_used else res_defaults
        
        resolution_cache = {}
        for model in all_models:
            model_last_used = preset_manager.load_last_used("resolution", model)
            model_defaults = resolution_defaults([model])
            resolution_cache[model] = preset_manager.merge_config(model_defaults, model_last_used) if model_last_used else model_defaults
        
        # === OUTPUT TAB ===
        primary_output_last_used = preset_manager.load_last_used("output", all_models[0])
        out_defaults = output_defaults(all_models)
        primary_output_settings = preset_manager.merge_config(out_defaults, primary_output_last_used) if primary_output_last_used else out_defaults
        
        output_cache = {}
        for model in all_models:
            model_output_last_used = preset_manager.load_last_used("output", model)
            model_out_defaults = output_defaults([model])
            output_cache[model] = preset_manager.merge_config(model_out_defaults, model_output_last_used) if model_output_last_used else model_out_defaults
        
        # === SEEDVR2 TAB ===
        seedvr2_cache = {}
        for model in seedvr2_models:
            model_last_used = preset_manager.load_last_used("seedvr2", model)
            model_defaults = seedvr2_defaults(model)
            seedvr2_cache[model] = preset_manager.merge_config(model_defaults, model_last_used) if model_last_used else model_defaults
        
        # === GAN TAB ===
        gan_cache = {}
        for model in gan_models:
            model_last_used = preset_manager.load_last_used("gan", model)
            model_defaults = gan_defaults(BASE_DIR)
            gan_cache[model] = preset_manager.merge_config(model_defaults, model_last_used) if model_last_used else model_defaults
        
        # === RIFE TAB ===
        rife_cache = {}
        for model in rife_models:
            model_last_used = preset_manager.load_last_used("rife", model)
            model_defaults = rife_defaults(model)
            rife_cache[model] = preset_manager.merge_config(model_defaults, model_last_used) if model_last_used else model_defaults
        
        # === FLASHVSR TAB ===
        flashvsr_cache = {}
        for model in flashvsr_models:
            model_last_used = preset_manager.load_last_used("flashvsr", model)
            model_defaults = flashvsr_defaults(model)
            flashvsr_cache[model] = preset_manager.merge_config(model_defaults, model_last_used) if model_last_used else model_defaults
        
        # === FACE TAB ===
        face_cache = {}
        for model in all_models:  # Load face presets for ALL models (no limit)
            model_last_used = preset_manager.load_last_used("face", model)
            model_defaults = face_defaults(all_models)
            face_cache[model] = preset_manager.merge_config(model_defaults, model_last_used) if model_last_used else model_defaults
        
        return (primary_res_settings, resolution_cache, primary_output_settings, output_cache,
                seedvr2_cache, gan_cache, rife_cache, flashvsr_cache, face_cache)

    # Load all startup presets for ALL tabs
    (startup_res_settings, startup_res_cache, startup_output_settings, startup_output_cache,
     startup_seedvr2_cache, startup_gan_cache, startup_rife_cache, startup_flashvsr_cache, startup_face_cache) = load_all_startup_presets()
    
    with gr.Blocks(title=APP_TITLE, theme=modern_theme) as demo:
        # Shared state for cross-tab communication
        # AUTO-POPULATED with last-used presets for ALL tabs on startup (Resolution, Output, SeedVR2, GAN, RIFE, FlashVSR+, Face)
        shared_state = gr.State({
            "health_banner": {"text": health_text},
            "seed_controls": {
                # AUTO-LOADED from Resolution tab last-used presets (primary model UI values)
                "resolution_val": startup_res_settings.get("target_resolution", 1080),
                "max_resolution_val": startup_res_settings.get("max_target_resolution", 0),
                "current_model": None,
                "last_input_path": "",
                "last_output_dir": "",
                "last_output_path": None,  # FIXED: Added for pinned comparison feature
                # AUTO-LOADED per-model caches for ALL tabs (restored at startup)
                "resolution_cache": startup_res_cache,
                "output_cache": startup_output_cache,
                "seedvr2_cache": startup_seedvr2_cache,  # NEW: SeedVR2 presets per model
                "gan_cache": startup_gan_cache,  # NEW: GAN presets per model
                "rife_cache": startup_rife_cache,  # NEW: RIFE presets per model
                "flashvsr_cache": startup_flashvsr_cache,  # NEW: FlashVSR+ presets per model
                "face_cache": startup_face_cache,  # NEW: Face presets per model
                # AUTO-LOADED from Output tab last-used presets (primary model UI values)
                "png_padding_val": startup_output_settings.get("png_padding", 6),
                "png_keep_basename_val": startup_output_settings.get("png_keep_basename", True),
                "skip_first_frames_val": startup_output_settings.get("skip_first_frames", 0),
                "load_cap_val": startup_output_settings.get("load_cap", 0),
                "fps_override_val": startup_output_settings.get("fps_override", 0),
                "output_format_val": startup_output_settings.get("output_format", "auto"),
                "comparison_mode_val": startup_output_settings.get("comparison_mode", "slider"),
                "pin_reference_val": startup_output_settings.get("pin_reference", False),
                "fullscreen_val": startup_output_settings.get("fullscreen_enabled", True),
                "save_metadata_val": startup_output_settings.get("save_metadata", True),
                "face_strength_val": global_settings.get("face_strength", 0.5),
                # AUTO-LOADED chunking settings from Resolution tab (primary model)
                "chunk_size_sec": startup_res_settings.get("chunk_size", 0),
                "chunk_overlap_sec": startup_res_settings.get("chunk_overlap", 0.5),
                "ratio_downscale": startup_res_settings.get("ratio_downscale_then_upscale", False),
                "enable_max_target": startup_res_settings.get("enable_max_target", True),
                "auto_resolution": startup_res_settings.get("auto_resolution", True),
                "per_chunk_cleanup": startup_res_settings.get("per_chunk_cleanup", False),
                "scene_threshold": startup_res_settings.get("scene_threshold", 27.0),
                "min_scene_len": startup_res_settings.get("min_scene_len", 2.0),
                # RESTORE pinned reference from global settings (persists across restarts)
                "pinned_reference_path": global_settings.get("pinned_reference_path"),
            },
            "operation_status": "ready"
        })

        # Health banner at the top
        health_banner = gr.Markdown(f'<div class="health-banner">{health_text}</div>')
        gr.Markdown(f"# {APP_TITLE}")

        # Global settings tab (simple controls)
        with gr.Tab("Global Settings"):
            gr.Markdown("### 📁 Output & Temp Directories")
            with gr.Row():
                output_dir_box = gr.Textbox(
                    label="Default Outputs Folder",
                    value=global_settings["output_dir"],
                    info="Where processed files will be saved"
                )
                temp_dir_box = gr.Textbox(
                    label="Temp Folder",
                    value=global_settings["temp_dir"],
                    info="Temporary files during processing"
                )
            
            gr.Markdown("### 🎭 Face Restoration")
            with gr.Row():
                telemetry_toggle = gr.Checkbox(
                    label="Save run metadata (local telemetry)",
                    value=global_settings.get("telemetry", True),
                    info="Save processing metadata for troubleshooting"
                )
                face_global_toggle = gr.Checkbox(
                    label="Apply Face Restoration globally",
                    value=global_settings.get("face_global", False),
                    info="Enable face restoration for all upscaling operations (SeedVR2, GAN, RIFE, FlashVSR+)"
                )
            with gr.Row():
                face_strength_slider = gr.Slider(
                    label="Global Face Restoration Strength",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=global_settings.get("face_strength", 0.5),
                    info="Strength of face restoration when globally enabled (0.0 = subtle, 1.0 = maximum)"
                )
            
            # FIXED: Make model cache paths EDITABLE and persistable (not read-only)
            gr.Markdown("### 📦 Model Cache Paths")
            gr.Markdown("""
            **Configure where AI models are downloaded and cached.**
            
            These paths control where models (SeedVR2, FlashVSR+, Real-ESRGAN, etc.) are stored.
            If left empty, defaults from launcher BAT file or system defaults will be used.
            
            ⚠️ **IMPORTANT**: Changing these paths will NOT move existing models. You must:
            1. Save new paths here
            2. Restart the application
            3. Models will re-download to new location (or manually copy from old location)
            """)
            
            with gr.Row():
                models_dir_box = gr.Textbox(
                    label="Models Directory (MODELS_DIR)",
                    value=global_settings.get("models_dir", ""),
                    placeholder=str(BASE_DIR / "models"),
                    info="Base directory for all model weights. Leave empty to use launcher default or './models'."
                )
            
            with gr.Row():
                hf_home_box = gr.Textbox(
                    label="HuggingFace Home (HF_HOME)",
                    value=global_settings.get("hf_home", ""),
                    placeholder=str(BASE_DIR / "models"),
                    info="HuggingFace cache directory. Usually same as Models Directory. Leave empty to use MODELS_DIR."
                )
                transformers_cache_box = gr.Textbox(
                    label="Transformers Cache (TRANSFORMERS_CACHE)",
                    value=global_settings.get("transformers_cache", ""),
                    placeholder=str(BASE_DIR / "models"),
                    info="Transformers library cache. Usually same as HF_HOME. Leave empty to use MODELS_DIR."
                )
            
            gr.Markdown("""
            💡 **Current launcher settings** (from `Windows_Run_SECourses_Upscaler_Pro.bat`):
            - Models Dir: `{}`
            - HF Home: `{}`
            - Transformers: `{}`
            
            You can override these in the UI above, or edit the launcher BAT file for permanent changes.
            """.format(
                launcher_models_dir or "Not set (using defaults)",
                launcher_hf_home or "Not set (using MODELS_DIR)",
                launcher_transformers_cache or "Not set (using MODELS_DIR)"
            ))
            
            save_global = gr.Button("💾 Save Global Settings", variant="primary", size="lg")
            global_status = gr.Markdown("")

            # Execution mode controls
            gr.Markdown("### ⚙️ Execution Mode")
            gr.Markdown("""
            ## 🟢 Subprocess Mode (Default & **STRONGLY RECOMMENDED**)
            
            **What it does:**
            - Each processing run is a **completely isolated subprocess**
            - Models load fresh, process, then exit with **guaranteed cleanup**
            - **Works perfectly for ALL models** (SeedVR2, GAN, RIFE, FlashVSR+)
            
            **Benefits:**
            - ✅ **100% VRAM/RAM cleanup** after each run (guaranteed by process termination)
            - ✅ **Full cancellation support** - kill subprocess at any time, even mid-processing
            - ✅ **Automatic vcvars wrapper** on Windows (enables torch.compile without manual setup)
            - ✅ **Process isolation** - prevents memory leaks, CUDA errors don't crash app
            - ✅ **Proven stability** - production-ready, no known issues
            - ✅ **Cross-platform** - works identically on Windows and Linux
            
            **Performance:**
            - Adds ~5-10s model loading overhead per run (one-time cost)
            - For long videos (>1 min), overhead is negligible (<1% of total time)
            - For batch processing, overhead amortized across many files
            
            ---
            
            ## 🔴 In-App Mode (**EXPERIMENTAL - DO NOT USE**)
            
            **⚠️ CRITICAL: This mode is a NON-FUNCTIONAL PLACEHOLDER**
            
            **Status**: Partially implemented framework with **ZERO BENEFITS** and **CRITICAL LIMITATIONS**
            
            ### Why In-App Mode Doesn't Work:
            
            **1. ❌ NO MODEL PERSISTENCE (Core Issue)**
            - **Current Reality**: Models reload EVERY RUN (identical to subprocess)
            - **Expected**: Models stay in VRAM between runs for speed
            - **Why It Fails**: 
              - SeedVR2/FlashVSR+ use CLI architecture → models reload by design
              - GAN/RIFE runners don't implement persistent caching yet
              - ModelManager tracks state but can't force CLI to keep models loaded
            - **Result**: **ZERO SPEED BENEFIT** - same loading time as subprocess mode
            
            **2. ❌ NO CANCELLATION SUPPORT**
            - **Current Reality**: Cannot stop processing once started
            - **Why**: No subprocess to kill, threading-based cancel not implemented
            - **Impact**: Must wait for completion or force-quit entire app
            
            **3. ❌ NO VCVARS WRAPPER (Windows torch.compile broken)**
            - **Current Reality**: C++ toolchain not activated for in-app runs
            - **Why**: vcvarsall.bat must run BEFORE Python starts (can't activate mid-app)
            - **Impact**: torch.compile fails cryptically on Windows
            - **Workaround**: Must activate vcvars BEFORE launching app (manual, error-prone)
            
            **4. ⚠️ MEMORY LEAKS & VRAM ACCUMULATION**
            - **Current Reality**: No subprocess isolation, VRAM may not fully clear
            - **Why**: Python GC doesn't guarantee CUDA memory release
            - **Impact**: VRAM usage creeps up across runs, eventual OOM crashes
            
            **5. ⚠️ MODE LOCK (Cannot Switch Back)**
            - **Current Reality**: Switching to in-app locks mode until app restart
            - **Why**: Prevents unsafe mid-session mode switching
            - **Impact**: Must restart app to return to subprocess mode
            
            ### What Would Be Required for Functional In-App Mode:
            
            **Major Architecture Changes Needed:**
            1. Refactor SeedVR2/FlashVSR+ CLIs to expose `load_model()` and `infer()` separately
            2. Implement ModelManager.persistent_load() with actual VRAM caching
            3. Add intelligent model swapping (auto-unload when user switches models)
            4. Implement threading.Event-based cancellation throughout runners
            5. Pre-activate vcvars on Windows at app startup (before torch import)
            6. Add CUDA memory profiling and automatic leak detection
            7. Implement forced GC + cache clearing between runs with verification
            
            **Estimated Effort**: ~40-60 hours of development + extensive testing
            **Risk**: High (CUDA/memory management complexity)
            **Benefit**: Marginal (~5-10% speed for short videos, 0% for long videos)
            
            ### Current Recommendation:
            
            🚫 **DO NOT USE IN-APP MODE** - It is:
            - ❌ Non-functional (models reload anyway)
            - ❌ Slower (no cancellation = wasted time on errors)
            - ❌ Dangerous (memory leaks, torch.compile failures)
            - ❌ Locked-in (requires restart to escape)
            
            ✅ **ALWAYS USE SUBPROCESS MODE** - It is:
            - ✅ Production-ready and battle-tested
            - ✅ Reliable cleanup and cancellation
            - ✅ Works for all models without exceptions
            - ✅ Recommended by developers
            
            💡 **In-app mode exists ONLY as a code framework for potential future optimization.**
            It should be considered **disabled** for all practical purposes.
            """)
            mode_radio = gr.Radio(
                choices=["subprocess", "in_app"],
                value=saved_mode,  # Restore from saved settings
                label="Processing Mode",
                info="⚠️ Changing to in-app requires confirmation and persists until app restart",
                interactive=True
            )
            mode_confirm = gr.Checkbox(
                label="⚠️ I understand that in-app mode requires app restart to revert",
                value=False,
                visible=True,
                info="Enable this checkbox to confirm mode switch to in-app (cannot be undone without restart)"
            )
            apply_mode_btn = gr.Button("🔄 Apply Mode Change", variant="secondary", size="lg")

            # Wire up global settings events
            def save_global_settings(od, td, tel, face, face_str, models_dir, hf_home, trans_cache, state):
                from shared.services.global_service import save_global_settings
                return save_global_settings(od, td, tel, face, face_str, models_dir, hf_home, trans_cache, runner, preset_manager, global_settings, run_logger, state)

            def apply_mode_selection(mode_choice, confirm, state):
                from shared.services.global_service import apply_mode_selection
                return apply_mode_selection(mode_choice, confirm, runner, preset_manager, global_settings, state)

            # Add status display for mode changes
            mode_status = gr.Markdown("")

            save_global.click(
                fn=save_global_settings,
                inputs=[output_dir_box, temp_dir_box, telemetry_toggle, face_global_toggle, face_strength_slider, 
                       models_dir_box, hf_home_box, transformers_cache_box, shared_state],
                outputs=[global_status, shared_state],
            )

            apply_mode_btn.click(
                fn=apply_mode_selection,
                inputs=[mode_radio, mode_confirm, shared_state],
                outputs=[mode_radio, mode_confirm, mode_status, shared_state],
            )

        # Self-contained tabs following SECourses pattern
        with gr.Tab("🎬 SeedVR2 (Video/Image)"):
            seedvr2_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        with gr.Tab("📐 Resolution & Scene Split"):
            resolution_tab(
                preset_manager=preset_manager,
                shared_state=shared_state,
                base_dir=BASE_DIR
            )

        with gr.Tab("🎭 Output & Comparison"):
            output_tab(
                preset_manager=preset_manager,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                global_settings=global_settings
            )

        with gr.Tab("👤 Face Restoration"):
            face_tab(
                preset_manager=preset_manager,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR
            )

        with gr.Tab("⏱️ RIFE / FPS / Edit Videos"):
            rife_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        with gr.Tab("🖼️ Image-Based (GAN)"):
            gan_tab(
                preset_manager=preset_manager,
                runner=runner,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        with gr.Tab("⚡ FlashVSR+ (Real-Time Diffusion)"):
            flashvsr_tab(
                preset_manager=preset_manager,
                run_logger=run_logger,
                global_settings=global_settings,
                shared_state=shared_state,
                base_dir=BASE_DIR,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        with gr.Tab("🏥 Health Check"):
            health_tab(
                global_settings=global_settings,
                shared_state=shared_state,
                temp_dir=temp_dir,
                output_dir=output_dir
            )

        # Update health banner on load and changes
        def update_health_banner(state):
            """Update health banner with current state"""
            health_text = state.get("health_banner", {}).get("text", "System ready")
            return gr.update(value=f'<div class="health-banner">{health_text}</div>')

        # Update on load
        demo.load(fn=update_health_banner, inputs=shared_state, outputs=health_banner)
        
        # Update when shared state changes (for dynamic updates from tabs)
        shared_state.change(fn=update_health_banner, inputs=shared_state, outputs=health_banner)

    demo.launch(inbrowser=True)


if __name__ == "__main__":
    main()

