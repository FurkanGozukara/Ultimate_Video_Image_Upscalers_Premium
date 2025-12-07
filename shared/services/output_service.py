from typing import Any, Dict, List, Optional
import gradio as gr

from shared.preset_manager import PresetManager


def output_defaults(models: List[str]) -> Dict[str, Any]:
    return {
        "model": models[0] if models else "",
        "output_format": "auto",
        "fps_override": 0,
        "comparison_mode": "native",
        "pin_reference": False,
        "fullscreen": True,
        "png_padding": 5,
        "png_keep_basename": True,
        "skip_first_frames": 0,
        "load_cap": 0,
    }


OUTPUT_ORDER: List[str] = [
    "model",
    "output_format",
    "fps_override",
    "comparison_mode",
    "pin_reference",
    "fullscreen",
    "png_padding",
    "png_keep_basename",
    "skip_first_frames",
    "load_cap",
]


def _output_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(OUTPUT_ORDER, args))


def _apply_output_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    return [merged[k] for k in OUTPUT_ORDER]


def build_output_callbacks(
    preset_manager: PresetManager,
    shared_state: gr.State,
    models: List[str],
):
    defaults = output_defaults(models)

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("output", model_name)
        last_used = preset_manager.get_last_used_name("output", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name:
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)
        payload = _output_dict_from_args(list(args))
        model_name = payload["model"]
        preset_manager.save_preset("output", model_name, preset_name, payload)
        dropdown = refresh_presets(model_name, select_name=preset_name)
        current_map = dict(zip(OUTPUT_ORDER, list(args)))
        loaded_vals = _apply_output_preset(payload, defaults, preset_manager, current=current_map)
        return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        model_name = model_name or defaults["model"]
        preset = preset_manager.load_preset("output", model_name, preset_name)
        if preset:
            preset_manager.set_last_used("output", model_name, preset_name)
        defaults_with_model = defaults.copy()
        defaults_with_model["model"] = model_name
        current_map = dict(zip(OUTPUT_ORDER, current_values))
        values = _apply_output_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
        return values

    def safe_defaults():
        return [defaults[k] for k in OUTPUT_ORDER]

    # Cache helpers used by tab_output UI
    def cache_output(fmt, state):
        state["seed_controls"]["output_format_val"] = fmt
        return gr.Markdown.update(value="Output format cached for runs."), state

    def cache_fps(fps_val, state):
        state["seed_controls"]["fps_override_val"] = fps_val
        return gr.Markdown.update(value="FPS override cached for runs."), state

    def cache_comparison(mode, state):
        state["seed_controls"]["comparison_mode_val"] = mode
        return gr.Markdown.update(value="Comparison mode cached for runs."), state

    def cache_pin(val, state):
        state["seed_controls"]["pin_reference_val"] = bool(val)
        return gr.Markdown.update(value="Pin reference preference cached."), state

    def cache_fullscreen(val, state):
        state["seed_controls"]["fullscreen_val"] = bool(val)
        return gr.Markdown.update(value="Fullscreen preference cached."), state

    def cache_png_padding(val, state):
        try:
            state["seed_controls"]["png_padding_val"] = max(1, int(val))
        except Exception:
            state["seed_controls"]["png_padding_val"] = defaults["png_padding"]
        return gr.Markdown.update(value="PNG padding cached for runs."), state

    def cache_png_basename(val, state):
        state["seed_controls"]["png_keep_basename_val"] = bool(val)
        return gr.Markdown.update(value="PNG base-name preference cached."), state

    def cache_skip(val, state):
        try:
            state["seed_controls"]["skip_first_frames_val"] = max(0, int(val))
        except Exception:
            state["seed_controls"]["skip_first_frames_val"] = 0
        return gr.Markdown.update(value="Skip-first-frames cached for runs."), state

    def cache_cap(val, state):
        try:
            state["seed_controls"]["load_cap_val"] = max(0, int(val))
        except Exception:
            state["seed_controls"]["load_cap_val"] = 0
        return gr.Markdown.update(value="Load-cap cached for runs."), state

    return {
        "defaults": defaults,
        "order": OUTPUT_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "cache_output": lambda *args: cache_output(*args[:-1], args[-1]),
        "cache_fps": lambda *args: cache_fps(*args[:-1], args[-1]),
        "cache_comparison": lambda *args: cache_comparison(*args[:-1], args[-1]),
        "cache_pin": lambda *args: cache_pin(*args[:-1], args[-1]),
        "cache_fullscreen": lambda *args: cache_fullscreen(*args[:-1], args[-1]),
        "cache_png_padding": lambda *args: cache_png_padding(*args[:-1], args[-1]),
        "cache_png_basename": lambda *args: cache_png_basename(*args[:-1], args[-1]),
        "cache_skip": lambda *args: cache_skip(*args[:-1], args[-1]),
        "cache_cap": lambda *args: cache_cap(*args[:-1], args[-1]),
    }


