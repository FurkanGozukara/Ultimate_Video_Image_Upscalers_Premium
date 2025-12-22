"""
Face Restoration Tab - Self-contained modular implementation
"""

import gradio as gr
from typing import Dict, Any

from shared.services.face_service import (
    build_face_callbacks, FACE_ORDER
)
from shared.models.seedvr2_meta import get_seedvr2_model_names


def _get_gan_model_names(base_dir) -> list:
    """Get GAN model names from Image_Upscale_Models folder"""
    from pathlib import Path
    models_dir = Path(base_dir) / "Image_Upscale_Models"
    if not models_dir.exists():
        return []
    choices = []
    for f in models_dir.iterdir():
        if f.is_file() and f.suffix.lower() in (".pth", ".safetensors"):
            choices.append(f.name)
    return sorted(choices)


def face_tab(preset_manager, global_settings: Dict[str, Any], shared_state: gr.State, base_dir=None):
    """
    Self-contained Face Restoration tab.
    Handles face restoration settings shared across models.
    
    Args:
        base_dir: Base directory for the application (required for GAN model discovery)
    """
    
    # Derive base_dir from global settings if not provided
    if base_dir is None:
        from pathlib import Path
        output_dir_path = Path(global_settings.get("output_dir", "./outputs"))
        # Navigate up from outputs to project root
        if output_dir_path.name == "outputs":
            base_dir = output_dir_path.parent
        else:
            # Fallback: try to find Image_Upscale_Models in parent directories
            current = output_dir_path
            for _ in range(3):  # Search up to 3 levels
                if (current / "Image_Upscale_Models").exists():
                    base_dir = current
                    break
                current = current.parent
            else:
                # Last resort: use current working directory
                base_dir = Path.cwd()

    # Get available models
    seedvr2_models = get_seedvr2_model_names()
    gan_models = _get_gan_model_names(base_dir)
    combined_models = sorted(list({*seedvr2_models, *gan_models}))

    # Build service callbacks
    service = build_face_callbacks(preset_manager, global_settings, combined_models, shared_state)

    # Get defaults and last used
    current_model = combined_models[0] if combined_models else "default"
    defaults = service["defaults"]
    last_used_name = preset_manager.get_last_used_name("face", current_model)
    last_used = preset_manager.load_last_used("face", current_model)
    if last_used_name and last_used is None:
        def update_warning(state):
            existing = state["health_banner"]["text"]
            warning = f"Last used Face preset '{last_used_name}' not found; loaded defaults."
            if existing:
                state["health_banner"]["text"] = existing + "\n" + warning
            else:
                state["health_banner"]["text"] = warning
            return state
        shared_state.value = update_warning(shared_state.value)

    merged_defaults = preset_manager.merge_config(defaults, last_used or {})
    values = [merged_defaults.get(k, defaults.get(k, "")) for k in FACE_ORDER]

    # Layout
    gr.Markdown("### 👤 Face Restoration Settings")
    gr.Markdown("*Configure face enhancement applied after upscaling across all models*")

    # Global face restoration toggle
    with gr.Group():
        gr.Markdown("#### 🌐 Global Face Restoration")
        global_face_enabled = gr.Checkbox(
            label="⚡ Enable Globally (All Upscalers)",
            value=global_settings.get("face_global", False),
            info="When enabled, face restoration is automatically applied to ALL video and image upscaling operations"
        )

        with gr.Group():
            gr.Markdown("""
            **ℹ️ Global Mode Info:**
            - Face restoration will be applied to SeedVR2, GAN, and RIFE processing
            - Settings below control the enhancement strength and behavior
            - Individual model presets can override global settings
            - Processing time increases with face restoration enabled
            """)

        apply_global_btn = gr.Button(
            "💾 Save Global Face Setting",
            variant="secondary",
            size="lg"
        )
        global_status = gr.Markdown("")

    # Model selector for presets
    model_selector = gr.Dropdown(
        label="Model Context (for presets)",
        choices=combined_models,
        value=current_model,
        info="Settings are saved/loaded per model type"
    )

    with gr.Tabs():
        # Model Selection
        with gr.TabItem("🤖 Model Selection"):
            gr.Markdown("#### Face Restoration Model")
            
            # Get available backends
            from shared.face_restore import get_available_backends
            available_backends = get_available_backends()
            
            if not available_backends:
                gr.Markdown("""
                **⚠️ No Face Restoration Models Available**
                
                Please install one of the following:
                - **GFPGAN**: `pip install gfpgan`
                - **CodeFormer**: Coming soon
                
                Until installed, face restoration will be skipped.
                """)
                backend_choices = ["auto"]
            else:
                backend_choices = ["auto"] + available_backends
            
            with gr.Group():
                # Note: face_backend is NOT in FACE_ORDER - it's for display only
                # The actual restoration model is controlled by restoration_model in the Restoration tab
                face_backend = gr.Dropdown(
                    label="Face Restoration Backend (Info Only)",
                    choices=backend_choices,
                    value="auto",
                    info="'auto' uses first available backend. GFPGAN is recommended. This is informational - configure in Face Restoration tab.",
                    interactive=False
                )
                
                backend_info = gr.Markdown("")
                
                def update_backend_info(backend):
                    if backend == "auto":
                        if available_backends:
                            return f"**Auto**: Will use {available_backends[0]} (first available)"
                        else:
                            return "**Auto**: No backends available - face restoration will be skipped"
                    elif backend == "gfpgan":
                        return """
                        **GFPGAN v1.3**
                        - High quality face restoration
                        - Works on photos and videos
                        - GPU accelerated
                        - Best for realistic faces
                        """
                    elif backend == "codeformer":
                        return """
                        **CodeFormer**
                        - Advanced face restoration
                        - Better for heavily degraded faces
                        - Controllable fidelity
                        - Coming soon
                        """
                    return ""
                
                face_backend.change(
                    fn=update_backend_info,
                    inputs=face_backend,
                    outputs=backend_info
                )
        
        # Face Detection Settings
        with gr.TabItem("🔍 Face Detection"):
            gr.Markdown("#### Face Detection Configuration")

            with gr.Group():
                face_detector = gr.Dropdown(
                    label="Face Detection Model",
                    choices=["retinaface", "yunet", "opencv", "dlib"],
                    value=values[1],  # FACE_ORDER index 1 = face_detector
                    info="Algorithm for detecting faces in images/videos"
                )

                detection_confidence = gr.Slider(
                    label="Detection Confidence Threshold",
                    minimum=0.1, maximum=1.0, step=0.05,
                    value=values[2],  # FACE_ORDER index 2 = detection_confidence
                    info="Minimum confidence for face detection (higher = fewer false positives)"
                )

                min_face_size = gr.Number(
                    label="Minimum Face Size (pixels)",
                    value=values[3],  # FACE_ORDER index 3 = min_face_size
                    precision=0,
                    info="Skip faces smaller than this size"
                )

                max_faces = gr.Number(
                    label="Maximum Faces Per Frame",
                    value=values[4],  # FACE_ORDER index 4 = max_faces
                    precision=0,
                    info="Limit faces processed per frame (0 = unlimited)"
                )

        # Restoration Settings
        with gr.TabItem("✨ Face Restoration"):
            gr.Markdown("#### Enhancement Parameters")

            with gr.Group():
                restoration_model = gr.Dropdown(
                    label="Restoration Model",
                    choices=["gfpgan", "restoreformer", "codeformer", "auto"],
                    value=values[5],  # FACE_ORDER index 5 = restoration_model
                    info="'auto' selects best model based on input quality"
                )

                face_strength = gr.Slider(
                    label="Face Enhancement Strength",
                    minimum=0.0, maximum=1.0, step=0.05,
                    value=values[6],  # FACE_ORDER index 6 = face_strength
                    info="How strongly to apply face restoration (0 = no change, 1 = maximum enhancement)"
                )

                restore_blindly = gr.Checkbox(
                    label="Restore All Faces Blindly",
                    value=values[7],  # FACE_ORDER index 7 = restore_blindly
                    info="Apply restoration to all detected faces without quality checks"
                )

                upscale_faces = gr.Checkbox(
                    label="Upscale Face Region First",
                    value=values[8],  # FACE_ORDER index 8 = upscale_faces
                    info="Pre-upscale face areas before restoration"
                )

        # Advanced Settings
        with gr.TabItem("⚙️ Advanced Settings"):
            gr.Markdown("#### Fine-tuning & Performance")

            with gr.Group():
                face_padding = gr.Slider(
                    label="Face Padding Ratio",
                    minimum=0.1, maximum=1.0, step=0.1,
                    value=values[9],  # FACE_ORDER index 9 = face_padding
                    info="Extra area around face for context"
                )

                face_landmarks = gr.Checkbox(
                    label="Use Face Landmarks",
                    value=values[10],  # FACE_ORDER index 10 = use_landmarks (key name differs)
                    info="Guide restoration using facial feature detection"
                )

                color_correction = gr.Checkbox(
                    label="Apply Color Correction",
                    value=values[11],  # FACE_ORDER index 11 = color_correction
                    info="Match face colors to surrounding skin tone"
                )

                gpu_acceleration = gr.Checkbox(
                    label="Enable GPU Acceleration",
                    value=values[12],  # FACE_ORDER index 12 = gpu_acceleration
                    info="Use GPU for face processing (faster but uses more VRAM)"
                )

                batch_face_processing = gr.Checkbox(
                    label="Batch Face Processing",
                    value=values[13],  # FACE_ORDER index 13 = batch_faces (key name differs)
                    info="Process multiple faces simultaneously"
                )

        # Quality & Output
        with gr.TabItem("🎨 Quality & Output"):
            gr.Markdown("#### Output Quality Settings")

            with gr.Group():
                output_quality = gr.Slider(
                    label="Output Quality",
                    minimum=0.1, maximum=1.0, step=0.05,
                    value=values[14],  # FACE_ORDER index 14 = output_quality
                    info="Restoration output quality (higher = better but slower)"
                )

                preserve_original = gr.Checkbox(
                    label="Preserve Original When Better",
                    value=values[15],  # FACE_ORDER index 15 = preserve_original
                    info="Keep original face if restoration doesn't improve quality"
                )

                artifact_reduction = gr.Checkbox(
                    label="Artifact Reduction",
                    value=values[16],  # FACE_ORDER index 16 = artifact_reduction
                    info="Apply additional filtering to reduce processing artifacts"
                )

                save_face_masks = gr.Checkbox(
                    label="Save Face Masks",
                    value=values[17],  # FACE_ORDER index 17 = save_face_masks
                    info="Save debug masks showing detected face regions"
                )

    # Preset management
    with gr.Accordion("💾 Preset Management", open=True):
        gr.Markdown("#### Save/Load Face Settings")

        preset_dropdown = gr.Dropdown(
            label="Face Presets",
            choices=preset_manager.list_presets("face", current_model),
            value=last_used_name or "",
        )

        with gr.Row():
            preset_name = gr.Textbox(
                label="Preset Name",
                placeholder="my_face_preset"
            )
            save_preset_btn = gr.Button("💾 Save Preset", variant="secondary")

        with gr.Row():
            load_preset_btn = gr.Button("📂 Load Preset")
            safe_defaults_btn = gr.Button("🔄 Safe Defaults")

        preset_status = gr.Markdown("")

    # Performance info
    with gr.Accordion("📊 Performance Impact", open=False):
        gr.Markdown("""
        #### Face Restoration Performance Notes

        **Processing Time Increase:**
        - Single face: ~2-3x slower
        - Multiple faces: ~3-5x slower
        - HD video: Significantly longer

        **Memory Usage:**
        - GPU: Additional 2-4GB VRAM
        - CPU: Additional 4-8GB RAM

        **Best Practices:**
        - Enable only when needed
        - Use global mode for batch processing
        - Lower strength for subtle improvements
        - Use GPU acceleration when available
        """)

    # Collect inputs for callbacks
    inputs_list = [
        model_selector, face_detector, detection_confidence, min_face_size, max_faces,
        restoration_model, face_strength, restore_blindly, upscale_faces,
        face_padding, face_landmarks, color_correction, gpu_acceleration, batch_face_processing,
        output_quality, preserve_original, artifact_reduction, save_face_masks
    ]

    # Wire up callbacks
    def refresh_presets(model):
        presets = preset_manager.list_presets("face", model)
        return gr.update(choices=presets, value="")

    model_selector.change(
        fn=refresh_presets,
        inputs=model_selector,
        outputs=preset_dropdown
    )

    # Global face toggle
    apply_global_btn.click(
        fn=lambda enabled, state: service["set_face_global"](enabled, state),
        inputs=[global_face_enabled, shared_state],
        outputs=[global_status, shared_state]
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
