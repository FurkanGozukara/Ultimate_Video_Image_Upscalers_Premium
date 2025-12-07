from typing import Any, Dict, List, Optional
import gradio as gr

from shared.preset_manager import PresetManager


def face_defaults(models: List[str]) -> Dict[str, Any]:
    return {
        "model": models[0] if models else "",
        "enable_face_restore": False,
        "strength": 0.5,
        "apply_globally": False,
    }


FACE_ORDER: List[str] = [
    "model",
    "enable_face_restore",
    "strength",
    "apply_globally",
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
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name.strip():
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _face_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("face", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(FACE_ORDER, list(args)))
            loaded_vals = _apply_face_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
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

            return values
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            return current_values

    def safe_defaults():
        return [defaults[k] for k in FACE_ORDER]

    def set_face_global(val, state=None):
        global_settings["face_global"] = bool(val)
        preset_manager.save_global_settings(global_settings)
        if state:
            state["seed_controls"]["face_strength_val"] = global_settings.get("face_strength", 0.5)
        return gr.Markdown.update(value="✅ Global face restoration updated"), state or {}

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
        return gr.Markdown.update(value=f"✅ Face strength set to {strength_num}"), state or {}

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


