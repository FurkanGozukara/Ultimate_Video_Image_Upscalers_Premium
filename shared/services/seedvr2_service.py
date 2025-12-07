import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import shutil
import platform
import subprocess
import gradio as gr

from shared.preset_manager import PresetManager
from shared.runner import Runner, RunResult
from shared.path_utils import (
    normalize_path,
    ffmpeg_set_fps,
    get_media_dimensions,
    get_media_duration_seconds,
    detect_input_type,
)
from shared.chunking import chunk_and_process, check_resume_available
from shared.face_restore import restore_image, restore_video
from shared.models.seedvr2_meta import get_seedvr2_model_names, model_meta_map
from shared.logging_utils import RunLogger
from shared.video_comparison import build_video_comparison, build_image_comparison
from shared.model_manager import get_model_manager, ModelType

# Constants --------------------------------------------------------------------
SEEDVR2_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}
SEEDVR2_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

# Defaults and ordering --------------------------------------------------------
def seedvr2_defaults() -> Dict[str, Any]:
    try:
        import torch  # type: ignore
        cuda_default = "0" if torch.cuda.is_available() else ""
    except Exception:
        cuda_default = ""
    return {
        "input_path": "",
        "output_override": "",
        "output_format": "auto",
        "model_dir": "",
        "dit_model": get_seedvr2_model_names()[0],
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "chunk_enable": False,
        "scene_threshold": 27.0,
        "scene_min_len": 2.0,
        "resolution": 1080,
        "max_resolution": 0,
        "batch_size": 5,
        "uniform_batch_size": False,
        "seed": 42,
        "skip_first_frames": 0,
        "load_cap": 0,
        "prepend_frames": 0,
        "temporal_overlap": 0,
        "color_correction": "lab",
        "input_noise_scale": 0.0,
        "latent_noise_scale": 0.0,
        "cuda_device": cuda_default,
        "dit_offload_device": "none",
        "vae_offload_device": "none",
        "tensor_offload_device": "cpu",
        "blocks_to_swap": 0,
        "swap_io_components": False,
        "vae_encode_tiled": False,
        "vae_encode_tile_size": 1024,
        "vae_encode_tile_overlap": 128,
        "vae_decode_tiled": False,
        "vae_decode_tile_size": 1024,
        "vae_decode_tile_overlap": 128,
        "tile_debug": "false",
        "attention_mode": "flash_attn",
        "compile_dit": False,
        "compile_vae": False,
        "compile_backend": "inductor",
        "compile_mode": "default",
        "compile_fullgraph": False,
        "compile_dynamic": False,
        "compile_dynamo_cache_size_limit": 64,
        "compile_dynamo_recompile_limit": 128,
        "cache_dit": False,
        "cache_vae": False,
        "debug": False,
        "resume_chunking": False,
    }


SEEDVR2_ORDER: List[str] = [
    "input_path",
    "output_override",
    "output_format",
    "model_dir",
    "dit_model",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "chunk_enable",
    "scene_threshold",
    "scene_min_len",
    "resolution",
    "max_resolution",
    "batch_size",
    "uniform_batch_size",
    "seed",
    "skip_first_frames",
    "load_cap",
    "prepend_frames",
    "temporal_overlap",
    "color_correction",
    "input_noise_scale",
    "latent_noise_scale",
    "cuda_device",
    "dit_offload_device",
    "vae_offload_device",
    "tensor_offload_device",
    "blocks_to_swap",
    "swap_io_components",
    "vae_encode_tiled",
    "vae_encode_tile_size",
    "vae_encode_tile_overlap",
    "vae_decode_tiled",
    "vae_decode_tile_size",
    "vae_decode_tile_overlap",
    "tile_debug",
    "attention_mode",
    "compile_dit",
    "compile_vae",
    "compile_backend",
    "compile_mode",
    "compile_fullgraph",
    "compile_dynamic",
    "compile_dynamo_cache_size_limit",
    "compile_dynamo_recompile_limit",
    "cache_dit",
        "cache_vae",
        "debug",
        "resume_chunking",
    ]


# Guardrails -------------------------------------------------------------------
def _enforce_seedvr2_guardrails(cfg: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    cfg = cfg.copy()
    bs = int(cfg.get("batch_size", defaults["batch_size"]))
    if bs % 4 != 1:
        cfg["batch_size"] = max(1, (bs // 4) * 4 + 1)
    if cfg.get("vae_encode_tiled"):
        if cfg.get("vae_encode_tile_overlap", 0) >= cfg.get("vae_encode_tile_size", defaults["vae_encode_tile_size"]):
            cfg["vae_encode_tile_overlap"] = max(0, cfg.get("vae_encode_tile_size", defaults["vae_encode_tile_size"]) - 1)
    if cfg.get("vae_decode_tiled"):
        if cfg.get("vae_decode_tile_overlap", 0) >= cfg.get("vae_decode_tile_size", defaults["vae_decode_tile_size"]):
            cfg["vae_decode_tile_overlap"] = max(0, cfg.get("vae_decode_tile_size", defaults["vae_decode_tile_size"]) - 1)
    blockswap_enabled = cfg.get("blocks_to_swap", 0) > 0 or cfg.get("swap_io_components")
    if blockswap_enabled and str(cfg.get("dit_offload_device", "none")).lower() in ("none", ""):
        cfg["dit_offload_device"] = "cpu"
    devices = [d.strip() for d in str(cfg.get("cuda_device", "")).split(",") if d.strip()]
    if len(devices) > 1:
        if cfg.get("cache_dit"):
            cfg["cache_dit"] = False
        if cfg.get("cache_vae"):
            cfg["cache_vae"] = False
    return cfg


# Preset helpers ---------------------------------------------------------------
def _seedvr2_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(SEEDVR2_ORDER, args))


def _apply_preset_to_values(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    merged = _enforce_seedvr2_guardrails(merged, defaults)
    return [merged[key] for key in SEEDVR2_ORDER]


# Validation helpers -----------------------------------------------------------
def _validate_cuda_devices(cuda_spec: str) -> Optional[str]:
    try:
        import torch  # type: ignore

        if not cuda_spec:
            return None
        if not torch.cuda.is_available():
            return "CUDA is not available on this system, but CUDA devices were specified."
        devices = [d.strip() for d in str(cuda_spec).split(",") if d.strip() != ""]
        count = torch.cuda.device_count()
        invalid = [d for d in devices if (not d.isdigit()) or int(d) >= count]
        if invalid:
            return f"Invalid CUDA device id(s): {', '.join(invalid)}. Available: 0-{count-1}"
    except Exception as exc:
        return f"CUDA validation failed: {exc}"
    return None


def _expand_cuda_spec(cuda_spec: str) -> str:
    try:
        import torch  # type: ignore

        if str(cuda_spec).strip().lower() == "all" and torch.cuda.is_available():
            return ",".join(str(i) for i in range(torch.cuda.device_count()))
    except Exception:
        pass
    return cuda_spec


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _list_media_files(folder: str, video_exts: set, image_exts: set) -> List[str]:
    try:
        p = Path(normalize_path(folder))
        if not p.exists() or not p.is_dir():
            return []
        items = []
        for f in sorted(p.iterdir()):
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext in video_exts or ext in image_exts:
                items.append(str(f))
        return items
    except Exception:
        return []


# Comparison fallback ----------------------------------------------------------
def comparison_html_slider():
    return gr.HTML.update(
        value="<p>Comparison fallback: use native slider where available; custom HTML slider assets can be loaded if deployed.</p>"
    )


# Core run/cancel/preset callbacks --------------------------------------------
def build_seedvr2_callbacks(
    preset_manager: PresetManager,
    runner: Runner,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    output_dir: Path,
    temp_dir: Path,
):
    defaults = seedvr2_defaults()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("seedvr2", model_name)
        last_used = preset_manager.get_last_used_name("seedvr2", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, model_name: str, *args):
        if not preset_name.strip():
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _seedvr2_dict_from_args(list(args))
            # Validate the payload before saving
            validated_payload = _enforce_seedvr2_guardrails(payload, defaults)

            preset_manager.save_preset_safe("seedvr2", model_name, preset_name.strip(), validated_payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            # Reload the validated values to ensure UI consistency
            current_map = dict(zip(SEEDVR2_ORDER, list(args)))
            loaded_vals = _apply_preset_to_values(validated_payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        try:
            preset = preset_manager.load_preset_safe("seedvr2", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("seedvr2", model_name, preset_name)
                # Apply comprehensive validation to loaded preset
                preset = preset_manager.validate_preset_constraints(preset, "seedvr2", model_name)
                preset = _enforce_seedvr2_guardrails(preset, defaults)

            current_map = dict(zip(SEEDVR2_ORDER, current_values))
            values = _apply_preset_to_values(preset or {}, defaults, preset_manager, current=current_map)
            return values
        except Exception as e:
            # On error, return current values unchanged
            print(f"Error loading preset {preset_name}: {e}")
            return current_values

    def safe_defaults():
        return [defaults[key] for key in SEEDVR2_ORDER]

    def check_resume_status(global_settings, output_format):
        """Check if chunking resume is available and return status message."""
        temp_dir = Path(global_settings["temp_dir"])
        available, message = check_resume_available(temp_dir, output_format or "mp4")
        if available:
            return gr.Markdown.update(value=f"✅ {message}", visible=True)
        else:
            return gr.Markdown.update(value=f"ℹ️ {message}", visible=True)

    def cancel():
        canceled = runner.cancel()
        if canceled:
            return gr.Markdown.update(value="⏹️ Cancel requested"), ""
        return gr.Markdown.update(value="No active process"), ""

    def get_model_loading_status():
        """Get current model loading status for UI display"""
        model_manager = get_model_manager()
        loaded_models = model_manager.get_loaded_models_info()
        current_model = model_manager.current_model_id

        if not loaded_models:
            return "No models loaded"

        status_lines = []
        for model_id, info in loaded_models.items():
            state = info["state"]
            marker = "✅" if state == "loaded" else "⏳" if state == "loading" else "❌"
            current_marker = " ← current" if model_id == current_model else ""
            status_lines.append(f"{marker} {info['model_name']} ({state}){current_marker}")

        return "\n".join(status_lines)

    def _auto_res_from_input(input_path: str, state: Dict[str, Any]):
        """
        Recalculate target/max resolution and chunk estimate when input changes.
        """
        seed_controls = state.get("seed_controls", {})
        model_name = seed_controls.get("current_model") or defaults.get("dit_model")
        model_cache = seed_controls.get("resolution_cache", {}).get(model_name, {})
        if not input_path:
            return (
                gr.Slider.update(),
                gr.Slider.update(),
                gr.Markdown.update(value="Provide an input to auto-calc resolution/chunks."),
                state
            )
        p = Path(normalize_path(input_path))
        if not p.exists():
            return (
                gr.Slider.update(),
                gr.Slider.update(),
                gr.Markdown.update(value="Input path not found; keeping current resolution."),
                state
            )
        auto_res = model_cache.get("auto_resolution", seed_controls.get("auto_resolution", True))
        enable_max = model_cache.get("enable_max_target", seed_controls.get("enable_max_target", True))
        ratio_down = model_cache.get("ratio_downscale", seed_controls.get("ratio_downscale", False))
        chunk_size = float(model_cache.get("chunk_size_sec", seed_controls.get("chunk_size_sec", 0) or 0))
        chunk_overlap = float(model_cache.get("chunk_overlap_sec", seed_controls.get("chunk_overlap_sec", 0) or 0))
        target_res = int(model_cache.get("resolution_val") or seed_controls.get("resolution_val") or defaults["resolution"])
        max_target_res = int(model_cache.get("max_resolution_val") or seed_controls.get("max_resolution_val") or defaults["max_resolution"])

        dims = get_media_dimensions(str(p))
        msg_lines = []
        new_res = target_res
        if auto_res and dims:
            w, h = dims
            short_side = min(w, h)
            computed = min(short_side, target_res or short_side)
            if ratio_down:
                computed = min(computed, target_res or computed)
            if enable_max and max_target_res and max_target_res > 0:
                computed = min(computed, max_target_res)
            new_res = int((computed // 16) * 16 or computed)
            state["seed_controls"]["resolution_val"] = new_res
            msg_lines.append(f"Auto-resolution: input {w}x{h} → target {new_res} (max {max_target_res})")
        else:
            msg_lines.append("Auto-resolution disabled; no change.")

        # Chunk estimate
        est_msg = ""
        if chunk_size > 0 and chunk_overlap < chunk_size:
            dur = get_media_duration_seconds(str(p)) if detect_input_type(str(p)) == "video" else None
            if dur:
                import math

                est_chunks = math.ceil(dur / max(0.001, chunk_size - chunk_overlap))
                est_msg = f"Chunk estimate: ~{est_chunks} chunks for {dur:.1f}s (size {chunk_size}s, overlap {chunk_overlap}s)."
            else:
                est_msg = f"Chunking: size {chunk_size}s, overlap {chunk_overlap}s (duration unknown)."
        if est_msg:
            msg_lines.append(est_msg)

        return (
            gr.Slider.update(value=new_res),
            gr.Slider.update(value=max_target_res),
            gr.Markdown.update(value="\n".join(msg_lines)),
            state
        )

    def _resolve_input_path(file_upload: Optional[str], manual_path: str, batch_enable: bool, batch_input: str) -> str:
        if batch_enable and batch_input:
            return batch_input
        if file_upload:
            return str(file_upload)
        return manual_path

    def run_action(uploaded_file, face_restore_run, *args, preview_only: bool = False, state: Dict[str, Any] = None):
        try:
            state = state or {"seed_controls": {}, "operation_status": "ready"}
            state["operation_status"] = "running"
            seed_controls = state.get("seed_controls", {})
            settings_dict = _seedvr2_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            guardrail_msgs: List[str] = []
            settings["cuda_device"] = _expand_cuda_spec(settings.get("cuda_device", ""))
            state["seed_controls"]["current_model"] = settings.get("dit_model", defaults.get("dit_model"))
            model_cache = seed_controls.get("resolution_cache", {}).get(settings["dit_model"], {})
            meta = model_meta_map().get(settings.get("dit_model"))
            if meta:
                if settings.get("batch_size") == defaults["batch_size"]:
                    settings["batch_size"] = meta.default_batch_size
                if meta.max_resolution > 0:
                    settings["max_resolution"] = min(
                        meta.max_resolution,
                        settings.get("max_resolution", meta.max_resolution) or meta.max_resolution,
                    )
                if settings.get("attention_mode") == defaults["attention_mode"] and meta.preferred_attention:
                    settings["attention_mode"] = meta.preferred_attention
                if not meta.compile_compatible and (settings.get("compile_dit") or settings.get("compile_vae")):
                    settings["compile_dit"] = False
                    settings["compile_vae"] = False
                    guardrail_msgs.append(
                        f"{settings.get('dit_model')} is not compile-compatible; compile_dit/compile_vae disabled."
                    )
            before_guardrails = settings.copy()
            settings = _enforce_seedvr2_guardrails(settings, defaults)

        if settings.get("attention_mode") == "flash_attn":
            try:
                import flash_attn  # type: ignore  # noqa: F401
            except Exception as exc:
                settings["attention_mode"] = "sdpa"
                guardrail_msgs.append(f"flash_attn unavailable ({exc}); falling back to sdpa.")
        if settings.get("batch_size") != before_guardrails.get("batch_size"):
            guardrail_msgs.append(
                f"Batch size adjusted to {settings['batch_size']} (must follow 4n+1 rule)."
            )
        if settings.get("vae_encode_tiled") and settings.get("vae_encode_tile_overlap") != before_guardrails.get(
            "vae_encode_tile_overlap"
        ):
            guardrail_msgs.append(
                f"VAE encode overlap capped to {settings['vae_encode_tile_overlap']} (< tile size)."
            )
        if settings.get("vae_decode_tiled") and settings.get("vae_decode_tile_overlap") != before_guardrails.get(
            "vae_decode_tile_overlap"
        ):
            guardrail_msgs.append(
                f"VAE decode overlap capped to {settings['vae_decode_tile_overlap']} (< tile size)."
            )
        blockswap_enabled = settings.get("blocks_to_swap", 0) > 0 or settings.get("swap_io_components")
        if blockswap_enabled and before_guardrails.get("dit_offload_device", "").lower() in ("", "none") and str(
            settings.get("dit_offload_device", "")
        ).lower() == "cpu":
            guardrail_msgs.append("BlockSwap enabled without offload; dit_offload_device set to cpu.")

        settings["cuda_device"] = _expand_cuda_spec(settings.get("cuda_device", ""))

        input_path = _resolve_input_path(uploaded_file, settings["input_path"], settings["batch_enable"], settings["batch_input_path"])
        settings["input_path"] = normalize_path(input_path)
        state["seed_controls"]["last_input_path"] = settings["input_path"]

        if not settings["input_path"] or not Path(settings["input_path"]).exists():
            return ("❌ Input path missing or not found", "", None, None, "No chunks", gr.HTML.update(value="No comparison"), gr.ImageSlider.update(value=None), state)

        if settings.get("batch_enable") and settings.get("batch_output_path"):
            settings["output_override"] = settings["batch_output_path"]

        settings["batch_mode"] = bool(settings.get("batch_enable"))

        # Apply global output hints for PNG padding / skip-cap if user set them in Output tab
        settings["png_padding"] = seed_controls.get("png_padding_val", 5)
        settings["png_keep_basename"] = seed_controls.get("png_keep_basename_val", True)
        if settings.get("skip_first_frames", defaults["skip_first_frames"]) == defaults["skip_first_frames"]:
            if seed_controls.get("skip_first_frames_val") is not None:
                settings["skip_first_frames"] = seed_controls.get("skip_first_frames_val")
        if settings.get("load_cap", defaults["load_cap"]) == defaults["load_cap"]:
            if seed_controls.get("load_cap_val") is not None:
                settings["load_cap"] = seed_controls.get("load_cap_val")

        if settings["output_format"] == "auto":
            settings["output_format"] = None

        if not _ffmpeg_available():
            return ("❌ ffmpeg not found in PATH. Install ffmpeg and retry.", "", None, None, "No chunks", gr.HTML.update(value="No comparison"), gr.ImageSlider.update(value=None), state)

        face_apply = bool(face_restore_run) or bool(global_settings.get("face_global", False))
        face_strength = float(global_settings.get("face_strength", 0.5))
        settings["face_restore_global"] = face_apply

        # Apply Resolution tab cached values when available
        if seed_controls.get("resolution_val") is not None:
            settings["resolution"] = seed_controls["resolution_val"]
        if seed_controls.get("max_resolution_val") is not None:
            settings["max_resolution"] = seed_controls["max_resolution_val"]
        auto_res = model_cache.get("auto_resolution", seed_controls.get("auto_resolution", True))
        enable_max_target = model_cache.get("enable_max_target", seed_controls.get("enable_max_target", True))
        chunk_size_sec = float(model_cache.get("chunk_size_sec", seed_controls.get("chunk_size_sec", 0) or 0))
        chunk_overlap_sec = float(model_cache.get("chunk_overlap_sec", seed_controls.get("chunk_overlap_sec", 0) or 0))
        if chunk_size_sec > 0 and chunk_overlap_sec >= chunk_size_sec:
            chunk_overlap_sec = max(0.0, chunk_size_sec - 1.0)
        ratio_downscale = model_cache.get("ratio_downscale", seed_controls.get("ratio_downscale", False))
        per_chunk_cleanup = model_cache.get("per_chunk_cleanup", seed_controls.get("per_chunk_cleanup", False))
        target_res = model_cache.get("resolution_val", seed_controls.get("resolution_val", settings["resolution"]))
        max_target_res = model_cache.get("max_resolution_val", seed_controls.get("max_resolution_val", settings["max_resolution"]))

        media_dims = get_media_dimensions(settings["input_path"])
        if media_dims and auto_res:
            w, h = media_dims
            short_side = min(w, h)
            computed_res = min(short_side, target_res or short_side)
            if ratio_downscale:
                computed_res = min(computed_res, target_res or computed_res)
            if enable_max_target and max_target_res and max_target_res > 0:
                computed_res = min(computed_res, max_target_res)
            settings["resolution"] = int(computed_res // 16 * 16 or computed_res)
            settings["max_resolution"] = max_target_res or settings["max_resolution"]
        if seed_controls.get("output_format_val"):
            if settings.get("output_format") in (None, "auto"):
                settings["output_format"] = seed_controls["output_format_val"]

        settings["chunk_size_sec"] = chunk_size_sec
        settings["chunk_overlap_sec"] = chunk_overlap_sec
        settings["per_chunk_cleanup"] = per_chunk_cleanup

        if settings["batch_size"] % 4 != 1:
            settings["batch_size"] = max(1, int(settings["batch_size"] // 4) * 4 + 1)

        blockswap_enabled = settings["blocks_to_swap"] > 0 or settings["swap_io_components"]
        if blockswap_enabled and str(settings.get("dit_offload_device", "")).lower() in ("none", ""):
            settings["dit_offload_device"] = "cpu"

        if settings["vae_encode_tiled"] and settings["vae_encode_tile_overlap"] >= settings["vae_encode_tile_size"]:
            settings["vae_encode_tile_overlap"] = max(0, settings["vae_encode_tile_size"] - 1)
        if settings["vae_decode_tiled"] and settings["vae_decode_tile_overlap"] >= settings["vae_decode_tile_size"]:
            settings["vae_decode_tile_overlap"] = max(0, settings["vae_decode_tile_size"] - 1)

        cuda_warning = _validate_cuda_devices(settings.get("cuda_device", ""))
        if cuda_warning:
            return (f"⚠️ {cuda_warning}", "", None, None, "No chunks", gr.HTML.update(value="No comparison"), gr.ImageSlider.update(value=None), state)
        devices_list = [d.strip() for d in str(settings.get("cuda_device", "")).split(",") if d.strip()]
        if meta and not meta.supports_multi_gpu and len(devices_list) > 1:
            return (
                f"⚠️ {settings.get('dit_model')} supports single GPU only. Select one CUDA device.",
                "",
                None,
                None,
                "No chunks",
                gr.HTML.update(value="No comparison"),
                gr.ImageSlider.update(value=None),
                state
            )
        if len(devices_list) > 1 and (settings.get("cache_dit") or settings.get("cache_vae")):
            return ("⚠️ Model caching requires a single GPU selection. Disable caching or choose one GPU.", "", None, None, "No chunks", gr.HTML.update(value="No comparison"), gr.ImageSlider.update(value=None), state)

        def _process_single(single_settings: Dict[str, Any], progress_cb: Optional[Callable[[str], None]] = None):
            try:
                local_logs: List[str] = []
                if guardrail_msgs:
                    local_logs.extend([f"⚠️ {m}" for m in guardrail_msgs])
                local_media_dims = get_media_dimensions(single_settings["input_path"]) if auto_res else None
                if auto_res and local_media_dims:
                    local_logs.append(
                        f"Auto-resolution applied: input {local_media_dims[0]}x{local_media_dims[1]}, target {single_settings['resolution']}, max {single_settings['max_resolution']}"
                    )
                if single_settings.get("chunk_enable"):
                    local_logs.append(
                        f"Chunking enabled: size={single_settings.get('chunk_size_sec', 0)}s overlap={single_settings.get('chunk_overlap_sec', 0)}s cleanup={single_settings.get('per_chunk_cleanup')}"
                    )

                def on_progress(line: str):
                    line = line.rstrip()
                    local_logs.append(line)
                    if progress_cb:
                        progress_cb(line)

                result: RunResult
                output_video: Optional[str] = None
                output_image: Optional[str] = None
                chunk_info_msg = "No chunking performed."
                chunk_summary = "Single pass (no chunking)."
                status = "⚠️ Upscale exited unexpectedly"

            # Handle model loading for first run
            model_manager = get_model_manager()
            dit_model = single_settings.get("dit_model", "")

            # Check if we need to switch/load model
            if not model_manager.is_model_loaded(ModelType.SEEDVR2, dit_model, **single_settings):
                on_progress(f"Loading model: {dit_model}...\n")

                # Try to load the model
                if not runner.ensure_seedvr2_model_loaded(single_settings, on_progress):
                    error_msg = f"❌ Failed to load model: {dit_model}"
                    return (error_msg, "", None, None, "Model load failed", gr.HTML.update(value="No comparison"), gr.ImageSlider.update(value=None), state)

                on_progress("Model loaded successfully!\n")

            # Skip chunking for preview-only runs; process just the first frame instead
            should_chunk = (
                single_settings.get("chunk_enable")
                and not preview_only
                and detect_input_type(single_settings["input_path"]) == "video"
            )

            try:
                if should_chunk:
                    # Create a progress callback that only updates on chunk completion
                    completed_chunks = 0
                    def chunk_progress_callback(progress_val, desc=""):
                        nonlocal completed_chunks
                        if "Completed chunk" in desc:
                            completed_chunks += 1
                            on_progress(f"Completed {completed_chunks} chunks\n")

                    rc, clog, final_out, chunk_count = chunk_and_process(
                        runner,
                        single_settings,
                        scene_threshold=single_settings.get("scene_threshold", 27.0),
                        min_scene_len=single_settings.get("scene_min_len", 2.0),
                        temp_dir=Path(global_settings["temp_dir"]),
                        on_progress=lambda msg: None,  # Suppress individual chunk messages
                        chunk_seconds=float(single_settings.get("chunk_size_sec") or 0),
                        chunk_overlap=float(single_settings.get("chunk_overlap_sec") or 0),
                        per_chunk_cleanup=bool(single_settings.get("per_chunk_cleanup")),
                        resume_from_partial=bool(single_settings.get("resume_chunking", False)),
                        allow_partial=True,
                        global_output_dir=str(runner.output_dir) if hasattr(runner, "output_dir") else None,
                        progress_tracker=chunk_progress_callback,
                    )
                    status = "✅ Chunked upscale complete" if rc == 0 else f"⚠️ Chunked upscale ended early ({rc})"
                    output_path = final_out if final_out else None
                    output_video = output_path if output_path and output_path.lower().endswith(".mp4") else None
                    output_image = None
                    local_logs.append(clog)
                    chunk_summary = f"Processed {chunk_count} chunks. Final: {output_path}"
                    chunk_info_msg = f"Chunks: {chunk_count}\nOutput: {output_path}\n{clog}"
                    result = RunResult(rc, output_path, clog)
                else:
                    result = runner.run_seedvr2(single_settings, on_progress=on_progress, preview_only=preview_only)
                    chunk_summary = "Single pass (no chunking)."
            except Exception as e:
                error_msg = f"Processing failed: {str(e)}"
                local_logs.append(f"❌ {error_msg}")
                on_progress(f"❌ {error_msg}\n")
                status = "❌ Processing failed"
                output_path = None
                output_video = None
                output_image = None
                chunk_summary = "Failed"
                chunk_info_msg = f"Error: {error_msg}"
                result = RunResult(1, None, "\n".join(local_logs))

            status = "✅ Upscale complete" if result.returncode == 0 else f"⚠️ Upscale exited with code {result.returncode}"
            except Exception as e:
                error_msg = f"Unexpected error during processing: {str(e)}"
                local_logs.append(f"❌ {error_msg}")
                on_progress(f"❌ {error_msg}\n")
                status = "❌ Unexpected error"
                output_path = None
                output_video = None
                output_image = None
                chunk_summary = "Error"
                chunk_info_msg = f"Error: {error_msg}"
                result = RunResult(1, None, "\n".join(local_logs))
            output_video = result.output_path if result.output_path and str(result.output_path).lower().endswith(".mp4") else None
            output_image = result.output_path if result.output_path and not str(result.output_path).lower().endswith(".mp4") else None

            current_out_dir = runner.output_dir if hasattr(runner, "output_dir") else output_dir
        if result.output_path:
            try:
                outp = Path(result.output_path)
                state["seed_controls"]["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
            except Exception:
                pass
            run_logger.write_summary(
                Path(result.output_path) if result.output_path else current_out_dir,
                {
                    "input": single_settings["input_path"],
                    "output": result.output_path,
                    "returncode": result.returncode,
                    "args": single_settings,
                    "face_global": face_apply,
                    "chunk_summary": chunk_summary,
                },
            )

            if face_apply and output_video and Path(output_video).exists():
                restored = restore_video(output_video, strength=face_strength, on_progress=on_progress)
                if restored:
                    local_logs.append(f"Face-restored video saved to {restored} (strength {face_strength})")
                    output_video = restored
            if face_apply and output_image and Path(output_image).exists():
                restored_img = restore_image(output_image, strength=face_strength)
                if restored_img:
                    local_logs.append(f"Face-restored image saved to {restored_img} (strength {face_strength})")
                    output_image = restored_img

            fps_val = seed_controls.get("fps_override_val")
            if fps_val and output_video and Path(output_video).exists():
                adjusted = ffmpeg_set_fps(Path(output_video), float(fps_val))
                output_video = str(adjusted)
                local_logs.append(f"FPS overridden to {fps_val}: {adjusted}")

            comparison_html = ""
            image_slider_update = gr.ImageSlider.update(value=None)
            comparison_mode = seed_controls.get("comparison_mode_val", "native")
            pin_pref = bool(seed_controls.get("pin_reference_val", False))
            fs_pref = bool(seed_controls.get("fullscreen_val", False))
            if output_video:
                use_fallback = comparison_mode in ("html_slider", "fallback")
                # Native video comparison via ImageSlider not supported; choose HTML slider vs fallback assets
                comparison_html = build_video_comparison(
                    single_settings.get("input_path"),
                    output_video,
                    pin_reference=pin_pref,
                    start_fullscreen=fs_pref,
                    use_fallback_assets=use_fallback,
                )
            elif output_image:
                image_slider_update = gr.ImageSlider.update(
                    value=(single_settings.get("input_path"), output_image),
                    visible=True,
                )
                if comparison_mode in ("html_slider", "fallback"):
                    comparison_html = build_image_comparison(
                        single_settings.get("input_path"),
                        output_image,
                        pin_reference=pin_pref,
                    )

            return (
                status,
                "\n".join(local_logs),
                output_video,
                output_image,
                chunk_info_msg,
                comparison_html,
                image_slider_update,
                state
            )

        if settings.get("batch_enable") and Path(settings["input_path"]).is_dir() and not preview_only:
            files = _list_media_files(settings["input_path"], SEEDVR2_VIDEO_EXTS, SEEDVR2_IMAGE_EXTS)
            if not files:
                return (
                    "❌ No media files found in batch folder",
                    "",
                    None,
                    None,
                    "No chunks",
                    gr.HTML.update(value="No comparison"),
                    gr.ImageSlider.update(value=None),
                    state
                )
            batch_logs = []
            batch_outputs: List[str] = []
            last_video = None
            last_image = None
            last_chunk_info = "Batch processed."
            last_cmp = ""
            last_slider = gr.ImageSlider.update(value=None)
            for fp in files:
                single = settings.copy()
                single["input_path"] = normalize_path(fp)
                single["batch_enable"] = False
                single["batch_mode"] = False
                status, lg, ov, oi, cinfo, cmp_html, slider_upd = _process_single(single)
                batch_logs.append(f"[{Path(fp).name}] {status}\n{lg}")
                last_video = ov or last_video
                last_image = oi or last_image
                if ov:
                    batch_outputs.append(ov)
                if oi:
                    batch_outputs.append(oi)
                last_chunk_info = cinfo
                last_cmp = cmp_html
                last_slider = slider_upd

            current_out_dir = runner.output_dir if hasattr(runner, "output_dir") else output_dir
            if batch_outputs:
                try:
                    last_out = Path(batch_outputs[-1])
                    seed_controls_cache["last_output_dir"] = str(last_out.parent if last_out.is_file() else last_out)
                except Exception:
                    pass
            run_logger.write_summary(
                Path(last_video or last_image or current_out_dir),
                {
                    "batch": True,
                    "inputs": files,
                    "outputs": batch_outputs or [last_video, last_image],
                    "returncode": 0 if (last_video or last_image) else 1,
                    "args": settings,
                    "chunk_summary": last_chunk_info,
                },
            )
            return (
                "✅ Batch complete" if last_video or last_image else "⚠️ Batch finished with issues",
                "\n\n".join(batch_logs),
                last_video,
                last_image,
                last_chunk_info,
                gr.HTML.update(value=last_cmp),
                last_slider,
                state
            )

        # Streaming: queue progress lines while a worker thread runs the job
        progress_q: "queue.Queue[str]" = queue.Queue()
        result_holder: Dict[str, Any] = {}
        last_chunk_line = "Processing..."
        pending_update = False
        last_emit = time.time()

        def progress_cb(line: str):
            nonlocal last_chunk_line
            if "chunk" in line.lower():
                last_chunk_line = line.strip()
            progress_q.put(line)

        def worker():
            try:
                result_holder["payload"] = _process_single(settings, progress_cb=progress_cb)
            except Exception as exc:
                result_holder["payload"] = (f"❌ Failed: {exc}", "", None, None, "No chunks", gr.HTML.update(value="No comparison"), gr.ImageSlider.update(value=None))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

        log_acc: List[str] = []
        # Stream logs while processing; emit on chunk boundaries or sparse heartbeat
        while t.is_alive() or not progress_q.empty():
            try:
                line = progress_q.get(timeout=0.5)
                log_acc.append(line)
                if "chunk" in line.lower():
                    last_chunk_line = line.strip()
                    pending_update = True
            except queue.Empty:
                pass
            now = time.time()
            if pending_update or (now - last_emit > 5 and log_acc):
                last_emit = now
                pending_update = False
                yield (
                    gr.Markdown.update(value="⏳ Running SeedVR2..."),
                    "\n".join(log_acc[-400:]),
                    None,
                    None,
                    last_chunk_line,
                    gr.HTML.update(value=""),
                    gr.ImageSlider.update(value=None),
                    state
                )
        t.join()

        status, log_text, output_video, output_image, chunk_info_msg, comparison_html, image_slider_update = result_holder.get(
            "payload",
            ("❌ Failed", "\n".join(log_acc), None, None, "No chunks", gr.HTML.update(value="No comparison"), gr.ImageSlider.update(value=None), state),
        )
        # Final yield with results and accumulated logs
        final_log = log_text if log_text else "\n".join(log_acc)
        state["operation_status"] = "completed" if "✅" in status else "ready"
        yield (
            status,
            final_log,
            output_video,
            output_image,
            chunk_info_msg if chunk_info_msg else last_chunk_line,
            gr.HTML.update(value=comparison_html),
            image_slider_update,
            state
        )
            except Exception as e:
                error_msg = f"Critical error in SeedVR2 processing: {str(e)}"
                state["operation_status"] = "error"
                yield (
                    "❌ Critical error",
                    error_msg,
                    None,
                    None,
                    "Error",
                    gr.HTML.update(value="Error occurred"),
                    gr.ImageSlider.update(value=None),
                    state
                )

    return {
        "defaults": defaults,
        "order": SEEDVR2_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "check_resume_status": check_resume_status,
        "run_action": lambda *args: run_action(*args[:-1], state=args[-1]) if len(args) > 1 else run_action(*args),
        "cancel_action": lambda state=None: cancel(),
        "comparison_html_slider": comparison_html_slider,
        "auto_res_on_input": lambda path, state=None: _auto_res_from_input(path, state or {"seed_controls": {}}),
    }


