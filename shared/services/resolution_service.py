from typing import Any, Dict, List, Optional, Tuple
import gradio as gr

from shared.preset_manager import PresetManager
from shared.path_utils import normalize_path, get_media_duration_seconds
from shared.resolution_calculator import (
    calculate_resolution, calculate_chunk_count, calculate_disk_space_required,
    get_available_disk_space, ResolutionConfig, ResolutionResult
)
from shared.input_detector import detect_input, validate_batch_directory


def resolution_defaults(models: List[str]) -> Dict[str, Any]:
    return {
        "model": models[0] if models else "",
        "auto_resolution": True,
        "enable_max_target": True,
        "target_resolution": 1080,
        "max_target_resolution": 0,  # 0 = unlimited
        "chunk_size": 0,
        "chunk_overlap": 0.5,
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


def _get_aspect_ratio_str(aspect_ratio: float) -> Tuple[int, int]:
    """Convert aspect ratio to simplified fraction (e.g., 1.777 -> 16:9)"""
    common_ratios = {
        (16, 9): 1.778,
        (4, 3): 1.333,
        (21, 9): 2.333,
        (1, 1): 1.0,
        (3, 2): 1.5,
        (5, 4): 1.25,
    }
    
    # Find closest common ratio
    min_diff = float('inf')
    best_ratio = (16, 9)
    
    for (w, h), ratio in common_ratios.items():
        diff = abs(aspect_ratio - ratio)
        if diff < min_diff:
            min_diff = diff
            best_ratio = (w, h)
    
    # If very close to common ratio, use it
    if min_diff < 0.01:
        return best_ratio
    
    # Otherwise, calculate from actual ratio
    from math import gcd
    w = int(aspect_ratio * 1000)
    h = 1000
    divisor = gcd(w, h)
    return (w // divisor, h // divisor)


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

    def calculate_auto_resolution(input_path: str, target_res: int, max_res: int, 
                                  enable_max: bool, auto_mode: bool, ratio_aware: bool,
                                  model_scale: Optional[int], state: Dict) -> Tuple[str, Dict]:
        """
        Auto-calculate optimal resolution based on input and settings.
        
        Returns:
            (info_message, updated_state)
        """
        if not input_path or not input_path.strip():
            return "⚠️ No input path provided", state
        
        try:
            # Detect input
            input_info = detect_input(input_path)
            if not input_info.is_valid:
                return f"⚠️ {input_info.error_message}", state
            
            # Create resolution config
            config = ResolutionConfig(
                input_width=0,  # Will be detected
                input_height=0,
                target_resolution=target_res,
                max_resolution=max_res if enable_max else 0,
                model_scale=model_scale,
                enable_max_target=enable_max,
                auto_resolution=auto_mode,
                ratio_aware=ratio_aware,
                allow_downscale=True
            )
            
            # Calculate resolution
            result = calculate_resolution(input_path, config)
            
            # Update state with calculated values
            seed_controls = state.get("seed_controls", {})
            seed_controls["calculated_output_width"] = result.output_width
            seed_controls["calculated_output_height"] = result.output_height
            seed_controls["needs_downscale_first"] = result.needs_downscale_first
            seed_controls["input_resize_width"] = result.input_resize_width
            seed_controls["input_resize_height"] = result.input_resize_height
            state["seed_controls"] = seed_controls
            
            # Build info message
            info = f"### Auto-Resolution Result\n\n"
            info += f"{result.info_message}\n\n"
            
            if result.needs_downscale_first:
                info += f"**Note:** Input will be downscaled first to match model scale factor.\n\n"
            
            if result.clamped_by_max:
                info += f"⚠️ **Resolution was clamped by max target setting.**\n\n"
            
            # Add aspect ratio info
            ar_w, ar_h = _get_aspect_ratio_str(result.aspect_ratio)
            info += f"**Aspect Ratio:** {ar_w}:{ar_h} ({result.aspect_ratio:.3f})\n"
            
            return info, state
            
        except Exception as e:
            return f"❌ Error calculating resolution: {str(e)}", state
    
    def calculate_chunk_estimate(input_path: str, chunk_size: float, chunk_overlap: float, state: Dict) -> Tuple[str, Dict]:
        """
        Estimate number of chunks and processing info.
        
        Returns:
            (info_message, updated_state)
        """
        if not input_path or not input_path.strip():
            return "⚠️ No input path provided", state
        
        if chunk_size <= 0:
            return "ℹ️ Chunking disabled (chunk size = 0)", state
        
        try:
            # Get chunk estimate
            chunk_count, duration, info = calculate_chunk_count(input_path, chunk_size, 2.0)
            
            if chunk_count == 0:
                return info, state
            
            # Add disk space estimate (rough)
            config = ResolutionConfig(
                input_width=0,
                input_height=0,
                target_resolution=state.get("seed_controls", {}).get("resolution_val", 1080),
                max_resolution=0
            )
            
            try:
                result = calculate_resolution(input_path, config)
                space_required, space_str = calculate_disk_space_required(
                    input_path, result, "mp4", duration
                )
                
                # Get available space
                output_dir = state.get("seed_controls", {}).get("last_output_dir", ".")
                avail_bytes, avail_str = get_available_disk_space(output_dir)
                
                info += f"\n\n### Disk Space\n"
                info += f"Estimated required: **{space_str}**\n"
                info += f"Available: **{avail_str}**\n"
                
                if space_required > 0 and avail_bytes > 0:
                    if space_required > avail_bytes * 0.9:  # Using >90% of available
                        info += f"\n⚠️ **Warning:** Insufficient disk space!"
                    
            except Exception:
                pass
            
            # Update state
            seed_controls = state.get("seed_controls", {})
            seed_controls["estimated_chunk_count"] = chunk_count
            state["seed_controls"] = seed_controls
            
            return info, state
            
        except Exception as e:
            return f"❌ Error estimating chunks: {str(e)}", state
    
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
        "calculate_auto_resolution": calculate_auto_resolution,
        "calculate_chunk_estimate": calculate_chunk_estimate,
    }


