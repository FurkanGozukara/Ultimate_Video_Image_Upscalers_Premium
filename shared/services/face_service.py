from typing import Any, Dict, List, Optional
import gradio as gr

from shared.preset_manager import PresetManager


def face_defaults(models: List[str]) -> Dict[str, Any]:
    """Get default face restoration settings"""
    from shared.face_restore import get_available_backends
    available = get_available_backends()
    default_backend = available[0] if available else "auto"
    
    return {
        "model": models[0] if models else "",
        "face_detector": "retinaface",
        "detection_confidence": 0.7,
        "min_face_size": 64,
        "max_faces": 0,
        "restoration_model": "auto",
        "face_strength": 0.5,
        "restore_blindly": False,
        "upscale_faces": False,
        "face_padding": 0.3,
        "use_landmarks": True,
        "color_correction": True,
        "gpu_acceleration": True,
        "batch_faces": True,
        "output_quality": 0.8,
        "preserve_original": True,
        "artifact_reduction": False,
        "save_face_masks": False,
        "backend": default_backend,  # Keep for backward compatibility with old presets
    }


FACE_ORDER: List[str] = [
    # IMPORTANT: This order MUST match inputs_list in ui/face_tab.py exactly!
    # The UI creates components in this exact sequence (lines 344-349 in face_tab.py)
    "model",                  # 0: model_selector
    "face_detector",          # 1: face_detector (Face Detection tab)
    "detection_confidence",   # 2: detection_confidence
    "min_face_size",          # 3: min_face_size
    "max_faces",              # 4: max_faces
    "restoration_model",      # 5: restoration_model (Restoration tab)
    "face_strength",          # 6: face_strength (Restoration tab)
    "restore_blindly",        # 7: restore_blindly
    "upscale_faces",          # 8: upscale_faces
    "face_padding",           # 9: face_padding (Advanced tab)
    "use_landmarks",          # 10: face_landmarks (UI name differs)
    "color_correction",       # 11: color_correction
    "gpu_acceleration",       # 12: gpu_acceleration
    "batch_faces",            # 13: batch_face_processing (UI name differs)
    "output_quality",         # 14: output_quality (Quality tab)
    "preserve_original",      # 15: preserve_original
    "artifact_reduction",     # 16: artifact_reduction
    "save_face_masks",        # 17: save_face_masks
]


def _face_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(FACE_ORDER, args))


def _apply_face_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    return [merged[k] for k in FACE_ORDER]


def build_face_callbacks(
    preset_manager: PresetManager,
    global_settings: Dict[str, Any],
    models: List[str],
    shared_state: gr.State = None,
):
    defaults = face_defaults(models)

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("face", model_name)
        last_used = preset_manager.get_last_used_name("face", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name.strip():
            return gr.update(), gr.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _face_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("face", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(FACE_ORDER, list(args)))
            loaded_vals = _apply_face_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.update(), gr.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        """
        Load a preset.
        
        FIXED: Now returns (*values, status_message) to match UI output expectations.
        Note: Face tab has special logic - skips model_selector in outputs.
        """
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("face", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("face", model_name, preset_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(FACE_ORDER, current_values))
            values = _apply_face_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)

            try:
                global_settings["face_strength"] = float(values[2])
                preset_manager.save_global_settings(global_settings)
            except Exception:
                pass

            # Return values + status message (status is LAST)
            # Face tab skips model_selector in outputs, so return values[1:] + status
            status_msg = f"✅ Loaded preset '{preset_name}'" if preset else "ℹ️ Preset not found"
            return (*values[1:], gr.update(value=status_msg))
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            # Return current values + error status (skip first value for model_selector)
            return (*current_values[1:], gr.update(value=f"❌ Error: {str(e)}"))

    def safe_defaults():
        return [defaults[k] for k in FACE_ORDER]

    def set_face_global(val, state=None):
        global_settings["face_global"] = bool(val)
        preset_manager.save_global_settings(global_settings)
        if state:
            state["seed_controls"]["face_strength_val"] = global_settings.get("face_strength", 0.5)
        return gr.update(value="✅ Global face restoration updated"), state or {}

    def cache_strength(strength_val: float, state=None):
        try:
            strength_num = float(strength_val)
        except Exception:
            strength_num = defaults["strength"]
        strength_num = max(0.0, min(1.0, strength_num))
        global_settings["face_strength"] = strength_num
        preset_manager.save_global_settings(global_settings)
        if state:
            state["seed_controls"]["face_strength_val"] = strength_num
        return gr.update(value=f"✅ Face strength set to {strength_num}"), state or {}

    return {
        "defaults": defaults,
        "order": FACE_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "set_face_global": lambda *args: set_face_global(*args[:-1], args[-1]) if len(args) > 1 else set_face_global(args[0]),
        "cache_strength": lambda *args: cache_strength(*args[:-1], args[-1]) if len(args) > 1 else cache_strength(args[0]),
    }


