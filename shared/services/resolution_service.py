from typing import Any, Dict, List, Optional
import gradio as gr

from shared.preset_manager import PresetManager
from shared.path_utils import normalize_path, get_media_duration_seconds


def resolution_defaults(models: List[str]) -> Dict[str, Any]:
    return {
        "model": models[0] if models else "",
        "auto_resolution": True,
        "enable_max_target": True,
        "target_resolution": 1080,
        "max_target_resolution": 1920,
        "chunk_size": 0,
        "chunk_overlap": 0,
        "ratio_downscale_then_upscale": False,
        "per_chunk_cleanup": False,
    }


RESOLUTION_ORDER: List[str] = [
    "model",
    "auto_resolution",
    "enable_max_target",
    "target_resolution",
    "max_target_resolution",
    "chunk_size",
    "chunk_overlap",
    "ratio_downscale_then_upscale",
    "per_chunk_cleanup",
]


def _res_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(RESOLUTION_ORDER, args))


def _apply_resolution_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    if merged.get("chunk_size", 0) and merged.get("chunk_overlap", 0) >= merged.get("chunk_size", 0):
        merged["chunk_overlap"] = max(0, merged.get("chunk_size", 0) - 1)
    return [merged[k] for k in RESOLUTION_ORDER]


def chunk_estimate(chunk_size: float, chunk_overlap: float):
    if chunk_size <= 0:
        return gr.Markdown.update(value="Chunking disabled.")
    if chunk_overlap >= chunk_size:
        return gr.Markdown.update(value="⚠️ Chunk overlap must be smaller than chunk size.")
    return gr.Markdown.update(
        value=f"Chunking enabled: size={chunk_size}s, overlap={chunk_overlap}s. Estimated chunks = ceil(duration / (size-overlap))."
    )


def build_resolution_callbacks(
    preset_manager: PresetManager,
    shared_state: gr.State,
    models: List[str],
):
    defaults = resolution_defaults(models)

    def _ensure_model_cache(model: str, state: Dict[str, Any]) -> Dict[str, Any]:
        cache_root = state["seed_controls"].setdefault("resolution_cache", {})
        model_key = model or defaults["model"]
        return cache_root.setdefault(model_key, {})

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("resolution", model_name)
        last_used = preset_manager.get_last_used_name("resolution", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name.strip():
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _res_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("resolution", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(RESOLUTION_ORDER, list(args)))
            loaded_vals = _apply_resolution_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("resolution", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("resolution", model_name, preset_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(RESOLUTION_ORDER, current_values))
            values = _apply_resolution_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
            return values
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            return current_values

    def safe_defaults():
        return [defaults[k] for k in RESOLUTION_ORDER]

    def apply_to_seed(*args):
        """Apply resolution settings to SeedVR2 pipeline via shared state"""
        state = args[-1]
        values = list(args[:-1])
        
        # Extract resolution settings from inputs
        settings_dict = _res_dict_from_args(values)
        
        # Update shared state with resolution settings for SeedVR2
        seed_controls = state.get("seed_controls", {})
        
        # Cache resolution values for SeedVR2 to use
        seed_controls["resolution_val"] = settings_dict.get("target_resolution", 1080)
        seed_controls["max_resolution_val"] = settings_dict.get("max_target_resolution", 0)
        seed_controls["enable_max_target"] = settings_dict.get("enable_max_target", True)
        seed_controls["auto_resolution"] = settings_dict.get("auto_resolution", True)
        seed_controls["chunk_size_sec"] = settings_dict.get("chunk_size", 0)
        seed_controls["chunk_overlap_sec"] = settings_dict.get("chunk_overlap", 0)
        seed_controls["ratio_downscale"] = settings_dict.get("ratio_downscale_then_upscale", False)
        seed_controls["per_chunk_cleanup"] = settings_dict.get("per_chunk_cleanup", False)
        
        state["seed_controls"] = seed_controls
        
        status_msg = f"✅ Applied resolution settings to pipeline:\n"
        status_msg += f"- Target Resolution: {seed_controls['resolution_val']}px\n"
        status_msg += f"- Max Resolution: {seed_controls['max_resolution_val']}px\n"
        status_msg += f"- Auto-Resolution: {seed_controls['auto_resolution']}\n"
        if seed_controls['chunk_size_sec'] > 0:
            status_msg += f"- Chunking: {seed_controls['chunk_size_sec']}s (overlap: {seed_controls['chunk_overlap_sec']}s)\n"
        
        return gr.Markdown.update(value=status_msg), state

    def estimate_from_input(size, ov):
        path = seed_controls_cache.get("last_input_path")
        if not path:
            return gr.Markdown.update(value="Provide input path (upload or textbox) to estimate chunks.")
        path = normalize_path(path)
        dur = get_media_duration_seconds(path) if path else None
        if not dur:
            return chunk_estimate(size, ov)
        if size <= 0 or ov >= size:
            return chunk_estimate(size, ov)
        import math

        est = math.ceil(dur / max(0.001, size - ov))
        return gr.Markdown.update(value=f"Duration ~{dur:.1f}s → est. {est} chunks (size {size}s, overlap {ov}s).")

    def cache_resolution(t_res, m_res, model, state):
        model_cache = _ensure_model_cache(model, state)
        model_cache["resolution_val"] = t_res
        model_cache["max_resolution_val"] = m_res
        return gr.Markdown.update(value=f"Resolution cached for {model}."), state

    def cache_resolution_flags(auto_res, enable_max, chunk_sz, chunk_ov, ratio_down, per_cleanup, model, state):
        model_cache = _ensure_model_cache(model, state)
        model_cache["auto_resolution"] = auto_res
        model_cache["enable_max_target"] = enable_max
        model_cache["chunk_size_sec"] = float(chunk_sz or 0)
        model_cache["chunk_overlap_sec"] = float(chunk_ov or 0)
        model_cache["ratio_downscale"] = ratio_down
        model_cache["per_chunk_cleanup"] = per_cleanup
        return gr.Markdown.update(value=f"Resolution options cached for {model}."), state

    return {
        "defaults": defaults,
        "order": RESOLUTION_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "apply_to_seed": apply_to_seed,
        "chunk_estimate": chunk_estimate,
        "estimate_from_input": estimate_from_input,
        "cache_resolution": lambda *args: cache_resolution(*args[:-1], args[-1]),
        "cache_resolution_flags": lambda *args: cache_resolution_flags(*args[:-1], args[-1]),
    }


