import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, Tuple
import gradio as gr

from shared.preset_manager import PresetManager
from shared.path_utils import (
    normalize_path,
    collision_safe_dir,
    collision_safe_path,
    ffmpeg_set_fps,
    get_media_dimensions,
)
from shared.face_restore import restore_image, restore_video
from shared.logging_utils import RunLogger
from shared.realesrgan_runner import run_realesrgan
from shared.gan_runner_complete import GanRunner, GanResult
from shared.gan_runner import get_gan_model_metadata
from shared.video_comparison import build_video_comparison, build_image_comparison
from shared.video_comparison_slider import create_video_comparison_html


GAN_MODEL_EXTS = {".pth", ".safetensors"}
GAN_META_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_key(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _parse_scale_from_name(name: str) -> int:
    """Legacy fallback scale detection - now superseded by metadata system"""
    import re
    lowered = name.lower()
    m = re.search(r"(\d+)x", lowered)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 4


def _calculate_input_resolution_for_target(input_dims: Tuple[int, int], target_resolution: int, model_scale: int, enable_max: bool = True, max_resolution: int = 0) -> Tuple[int, int]:
    """
    Calculate optimal input resolution for GAN models with fixed scale factors.

    For fixed-scale GAN models (2x, 4x), we need to determine what input resolution
    will produce output closest to the desired target resolution.
    """
    if not input_dims or target_resolution <= 0:
        return input_dims

    input_w, input_h = input_dims
    input_short = min(input_w, input_h)

    # Calculate what input resolution would give us the target when multiplied by scale
    ideal_input_short = target_resolution / model_scale

    # Apply max resolution constraint if enabled
    if enable_max and max_resolution > 0:
        max_input_for_max_res = max_resolution / model_scale
        ideal_input_short = min(ideal_input_short, max_input_for_max_res)

    # Don't upscale beyond original resolution (would be wasteful)
    ideal_input_short = min(ideal_input_short, input_short)

    # Calculate new dimensions maintaining aspect ratio
    aspect_ratio = input_w / input_h
    if input_w <= input_h:  # Portrait or square
        new_w = ideal_input_short * aspect_ratio
        new_h = ideal_input_short
    else:  # Landscape
        new_w = ideal_input_short
        new_h = ideal_input_short / aspect_ratio

    # Round to even numbers (often required by models)
    new_w = int(new_w // 2 * 2)
    new_h = int(new_h // 2 * 2)

    return (max(64, new_w), max(64, new_h))  # Minimum 64px


def _load_gan_catalog(base_dir: Path):
    if GAN_META_CACHE:
        return
    data_dir = base_dir / "open-model-database" / "data" / "models"
    if not data_dir.exists():
        return
    for jf in data_dir.glob("*.json"):
        try:
            import json

            with jf.open("r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("name") or jf.stem
            scale = data.get("scale") or _parse_scale_from_name(name)
            GAN_META_CACHE[_normalize_key(name)] = {"name": name, "scale": scale}
        except Exception:
            continue


def _get_gan_meta(filename: str, base_dir: Path) -> Dict[str, Any]:
    """Legacy metadata function - now uses comprehensive registry"""
    from shared.gan_runner import get_gan_model_metadata
    metadata = get_gan_model_metadata(filename, base_dir)
    return {
        "scale": metadata.scale,
        "canonical": metadata.name,
        "supports_multi_gpu": False,
        "architecture": metadata.architecture,
        "description": metadata.description,
        "author": metadata.author,
        "tags": metadata.tags
    }


def _is_realesrgan_builtin(name: str) -> bool:
    key = _normalize_key(name)
    builtins = [
        "realesrganx4plus",
        "realesrganx4plusanime6b",
        "realesrnetx4plus",
        "realesrganx2plus",
        "realesranimevideov3",
        "realesrgeneralx4v3",
    ]
    return key in builtins


def _scan_gan_models(base_dir: Path) -> List[str]:
    """Scan for GAN models with comprehensive metadata"""
    models_dir = base_dir / "Image_Upscale_Models"
    if not models_dir.exists():
        return []

    models = []
    for f in models_dir.iterdir():
        if f.is_file() and f.suffix.lower() in GAN_MODEL_EXTS:
            models.append(f.name)
    
    # Trigger cache reload for metadata
    from shared.gan_runner import reload_gan_models_cache
    reload_gan_models_cache(base_dir)
    
    return sorted(models)


def gan_defaults(base_dir: Path) -> Dict[str, Any]:
    models = _scan_gan_models(base_dir)
    default_model = models[0] if models else ""

    return {
        "input_path": "",
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "model": default_model,
        "target_resolution": 1920,
        "downscale_first": True,
        "auto_calculate_input": True,
        "tile_size": 0,
        "overlap": 32,
        "denoising_strength": 0.0,
        "sharpening": 0.0,
        "color_correction": True,
        "gpu_acceleration": True,
        "gpu_device": "0",
        "batch_size": 1,
        "output_format": "auto",
        "output_quality": 95,
        "save_metadata": True,
        "create_subfolders": False,
    }


GAN_ORDER: List[str] = [
    "input_path",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "model",
    "target_resolution",
    "downscale_first",
    "auto_calculate_input",
    "tile_size",
    "overlap",
    "denoising_strength",
    "sharpening",
    "color_correction",
    "gpu_acceleration",
    "gpu_device",
    "batch_size",
    "output_format",
    "output_quality",
    "save_metadata",
    "create_subfolders",
]


def _gan_dict_from_args(args: List[Any]) -> Dict[str, Any]:
    return dict(zip(GAN_ORDER, args))


def _apply_gan_preset(
    preset: Dict[str, Any],
    defaults: Dict[str, Any],
    preset_manager: PresetManager,
    current: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    base = defaults.copy()
    if current:
        base.update(current)
    merged = preset_manager.merge_config(base, preset)
    return [merged[k] for k in GAN_ORDER]


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


def build_gan_callbacks(
    preset_manager: PresetManager,
    run_logger: RunLogger,
    global_settings: Dict[str, Any],
    shared_state: gr.State,
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    defaults = gan_defaults(base_dir)
    cancel_event = threading.Event()

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("gan", model_name)
        last_used = preset_manager.get_last_used_name("gan", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name.strip():
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)

        try:
            payload = _gan_dict_from_args(list(args))
            model_name = payload["model"]
            preset_manager.save_preset_safe("gan", model_name, preset_name.strip(), payload)
            dropdown = refresh_presets(model_name, select_name=preset_name.strip())

            current_map = dict(zip(GAN_ORDER, list(args)))
            loaded_vals = _apply_gan_preset(payload, defaults, preset_manager, current=current_map)

            return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals
        except Exception as e:
            return gr.Dropdown.update(), gr.Markdown.update(value=f"❌ Error saving preset: {str(e)}"), *list(args)

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        try:
            model_name = model_name or defaults["model"]
            preset = preset_manager.load_preset_safe("gan", model_name, preset_name)
            if preset:
                preset_manager.set_last_used("gan", model_name, preset_name)
                # Apply validation constraints
                preset = preset_manager.validate_preset_constraints(preset, "gan", model_name)

            defaults_with_model = defaults.copy()
            defaults_with_model["model"] = model_name
            current_map = dict(zip(GAN_ORDER, current_values))
            values = _apply_gan_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
            return values
        except Exception as e:
            print(f"Error loading preset {preset_name}: {e}")
            return current_values

    def safe_defaults():
        return [defaults[k] for k in GAN_ORDER]

    def prepare_single(single_path: str) -> Dict[str, Any]:
        s = settings.copy()
        s["input_path"] = normalize_path(single_path)
        s["base_dir"] = str(base_dir)  # Pass base directory for metadata lookup
        meta = _get_gan_meta(s.get("model", ""), base_dir)
        s["scale"] = meta.get("scale", s.get("scale", 4))  # Use metadata scale, default to 4x
        s["supports_multi_gpu"] = meta.get("supports_multi_gpu", False)
        s["model_name"] = meta.get("canonical", s.get("model", ""))
        # PNG padding (from Output tab cache if present)
        s["png_padding"] = int(seed_controls.get("png_padding_val", 5))
        return s

    def maybe_downscale(s):
            """
            Dynamic resolution adjustment for fixed-scale GAN models.
            When using Resolution & Scene Split settings, calculate optimal input resolution
            that will produce output closest to target resolution after model scaling.
            """
            if not s.get("use_resolution_tab"):
                return s

            # Get model scale factor
            model_scale = s.get("scale", 4)
            if model_scale <= 1:  # Not a scaling model
                return s

            # Get resolution settings from cache
            model_cache = seed_controls.get("resolution_cache", {}).get(s.get("model"), {})
            target_resolution = model_cache.get("resolution_val") or seed_controls.get("resolution_val", 0)
            max_resolution = model_cache.get("max_resolution_val") or seed_controls.get("max_resolution_val", 0)
            enable_max = model_cache.get("enable_max_target", seed_controls.get("enable_max_target", True))
            auto_resolution = model_cache.get("auto_resolution", seed_controls.get("auto_resolution", True))

            if not target_resolution and not auto_resolution:
                return s

            # Get input dimensions
            dims = get_media_dimensions(s["input_path"])
            if not dims:
                return s

            # Calculate optimal input resolution for target output
            optimal_input_dims = _calculate_input_resolution_for_target(
                dims, target_resolution, model_scale, enable_max, max_resolution
            )

            input_w, input_h = dims
            optimal_w, optimal_h = optimal_input_dims

            # Skip if already at optimal resolution (within tolerance)
            tolerance = 32  # Allow some tolerance for rounding
            if abs(input_w - optimal_w) <= tolerance and abs(input_h - optimal_h) <= tolerance:
                return s

            # Create temporary adjusted file
            tmp_path = Path(current_temp_dir) / f"gan_input_adjust_{Path(s['input_path']).stem}.mp4"
            input_is_video = Path(s["input_path"]).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")

            if input_is_video:
                # Video resolution adjustment with ffmpeg
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        s["input_path"],
                        "-vf",
                        f"scale={optimal_w}:{optimal_h}",
                        str(tmp_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                if tmp_path.exists():
                    s["input_path"] = str(tmp_path)
                    s["resolution_adjusted"] = True
            else:
                # Image resolution adjustment with OpenCV
                try:
                    import cv2
                    img = cv2.imread(s["input_path"], cv2.IMREAD_UNCHANGED)
                    if img is not None:
                        adjusted = cv2.resize(img, (optimal_w, optimal_h), interpolation=cv2.INTER_LANCZOS4)
                        tmp_path = tmp_path.with_suffix(Path(s["input_path"]).suffix)
                        cv2.imwrite(str(tmp_path), adjusted)
                        if tmp_path.exists():
                            s["input_path"] = str(tmp_path)
                            s["resolution_adjusted"] = True
                except Exception:
                    pass

            return s

    def run_action(upload, *args, preview_only: bool = False, state=None):
        # Streaming: run in background thread, stream log lines if available
        progress_q: "queue.Queue[str]" = queue.Queue()
        result_holder: Dict[str, Any] = {}

        # Define worker functions (moved outside try block)
        def worker_single(prepped_settings):
            status, lg, outp, cmp_html, slider_upd = run_single(prepped_settings, progress_cb=progress_q.put)
            result_holder["payload"] = (status, lg, outp, cmp_html, slider_upd)

        def worker_batch(batch_items):
            """Process batch using BatchProcessor - no duplicate code"""
            from shared.batch_processor import BatchProcessor, BatchJob
            
            outputs = []
            logs = []
            last_cmp = ""
            last_slider = gr.ImageSlider.update(value=None)
            
            # Create batch jobs
            jobs = []
            for item in batch_items:
                job = BatchJob(
                    input_path=str(item),
                    metadata={"settings": settings.copy()}
                )
                jobs.append(job)
            
            # Define processor function that reuses run_single
            def process_job(job: BatchJob) -> bool:
                if cancel_event.is_set():
                    return False
                
                try:
                    ps = maybe_downscale(prepare_single(job.input_path))
                    status, lg, outp, cmp_html, slider_upd = run_single(ps, progress_cb=progress_q.put)
                    
                    # Store results in job
                    job.output_path = outp
                    job.metadata["log"] = lg
                    job.metadata["comparison"] = cmp_html
                    job.metadata["slider"] = slider_upd
                    job.status = "completed" if outp else "failed"
                    
                    return bool(outp)
                except Exception as e:
                    job.status = "failed"
                    job.error_message = str(e)
                    return False
            
            # Use BatchProcessor for controlled execution
            batch_processor = BatchProcessor(max_workers=1)  # Sequential for GPU
            batch_result = batch_processor.process_batch(
                jobs=jobs,
                processor_func=process_job,
                max_concurrent=1
            )
            
            # Collect results from jobs
            for job in jobs:
                if job.status == "completed" and job.output_path:
                    outputs.append(job.output_path)
                if "log" in job.metadata:
                    logs.append(job.metadata["log"])
                if "comparison" in job.metadata and job.metadata["comparison"]:
                    last_cmp = job.metadata["comparison"]
                if "slider" in job.metadata:
                    last_slider = job.metadata["slider"]
            
            result_holder["payload"] = (
                f"✅ Batch complete: {len(outputs)}/{len(batch_items)} processed ({batch_result.failed_files} failed)",
                "\n\n".join(logs),
                outputs[0] if outputs else None,
                last_cmp,
                last_slider,
            )

        try:
            state = state or {"seed_controls": {}}
            seed_controls = state.get("seed_controls", {})
            cancel_event.clear()
            settings_dict = _gan_dict_from_args(list(args))
            settings = {**defaults, **settings_dict}
            settings["output_override"] = settings.get("output_override")
            settings["cuda_device"] = settings.get("cuda_device", "")

            # Pull latest global paths in case user changed them in Global tab
            current_output_dir = Path(global_settings.get("output_dir", output_dir))
            current_temp_dir = Path(global_settings.get("temp_dir", temp_dir))

            inp = normalize_path(upload if upload else settings["input_path"])
            if settings.get("batch_enable"):
                if not inp or not Path(inp).exists() or not Path(inp).is_dir():
                    return ("❌ Batch input folder missing", "", None, "", gr.ImageSlider.update(value=None), state)
            else:
                if not inp or not Path(inp).exists():
                    return ("❌ Input missing", "", None, "", gr.ImageSlider.update(value=None), state)

            settings["input_path"] = inp

            cuda_warn = _validate_cuda_devices(settings.get("cuda_device", ""))
            if cuda_warn:
                return (f"⚠️ {cuda_warn}", "", None, "", state)
            devices = [d.strip() for d in str(settings.get("cuda_device") or "").split(",") if d.strip()]
            if len(devices) > 1:
                return (
                    "⚠️ GAN backends currently use a single GPU; select one CUDA device.",
                    "",
                    None,
                    "",
                    gr.ImageSlider.update(value=None),
                    state
                )

            face_apply = bool(args[-1])
            face_apply = face_apply or global_settings.get("face_global", False)
            face_strength = float(global_settings.get("face_strength", 0.5))
            backend_val = settings.get("backend", "realesrgan")
            # Pull shared output/comparison preferences
            if (not settings.get("output_format")) or settings.get("output_format") == "auto":
                cached_fmt = seed_controls.get("output_format_val")
                if cached_fmt:
                    settings["output_format"] = cached_fmt
            if (not settings.get("fps_override")) or float(settings.get("fps_override") or 0) == 0:
                cached_fps = seed_controls.get("fps_override_val")
                if cached_fps:
                    settings["fps_override"] = cached_fps
            cmp_mode = seed_controls.get("comparison_mode_val", "native")
            pin_pref = bool(seed_controls.get("pin_reference_val", False))
            fs_pref = bool(seed_controls.get("fullscreen_val", False))

            def relocate_output(path_str: Optional[str]) -> Optional[str]:
                if not path_str:
                    return None
                target_root = settings.get("output_override")
                if not target_root:
                    return path_str
                src = Path(path_str)
                target_root_path = Path(normalize_path(target_root))
                target_root_path.mkdir(parents=True, exist_ok=True)
                if src.is_dir():
                    dest = collision_safe_dir(target_root_path / src.name)
                    shutil.copytree(src, dest)
                    return str(dest)
                else:
                    dest = collision_safe_path(target_root_path / src.name)
                    shutil.copyfile(src, dest)
                    return str(dest)

            # Define run_single function (moved outside try block)
            def run_single(prepped_settings: Dict[str, Any], progress_cb: Optional[Callable[[str], None]] = None):
                if cancel_event.is_set():
                    return ("⏹️ Canceled", "\n".join(["Canceled before start"]), None, "", gr.ImageSlider.update(value=None), state)
                header_log = [
                    f"Model: {prepped_settings['model_name']}",
                    f"Backend: {backend_val}",
                    f"Scale: {prepped_settings['scale']}x",
                    f"GPU: {prepped_settings.get('cuda_device') or 'auto/CPU'} (single GPU enforced)",
                    f"Input: {prepped_settings['input_path']}",
                ]
                if progress_cb:
                    for line in header_log:
                        progress_cb(line)
                # Run backend and collect logs
                try:
                    # Use new unified GAN runner
                    model_path_check = base_dir / "Image_Upscale_Models" / prepped_settings["model_name"]
                    if not model_path_check.exists() and not _is_realesrgan_builtin(prepped_settings["model_name"]):
                        return ("❌ Model weights not found", "\n".join(header_log + ["Missing model file."]), None, "", gr.ImageSlider.update(value=None), state)
                    
                    # Add face restoration settings
                    prepped_settings["face_restore"] = face_apply
                    prepped_settings["face_strength"] = face_strength
                    
                    # Use new complete GAN runner
                    gan_runner = GanRunner(base_dir)
                    output_path_target = current_output_dir / f"{Path(prepped_settings['input_path']).stem}_upscaled.png"
                    gan_result = gan_runner.run_gan_processing(
                        input_path=prepped_settings["input_path"],
                        model_name=prepped_settings["model"],
                        output_path=str(output_path_target),
                        settings=prepped_settings,
                        on_progress=progress_cb if progress_cb else None
                    )
                    
                    # Convert GanResult to expected result format
                    result = type('obj', (object,), {
                        'returncode': gan_result.returncode,
                        'output_path': gan_result.output_path,
                        'log': gan_result.log
                    })()
                except Exception as exc:  # surface ffmpeg or other runtime issues
                    err_msg = f"❌ GAN upscale failed: {exc}"
                    if progress_cb:
                        progress_cb(err_msg)
                    return (err_msg, "\n".join(header_log + [str(exc)]), None, "", gr.ImageSlider.update(value=None), state)
                if cancel_event.is_set():
                    status = "⏹️ Canceled"
                else:
                    status = "✅ GAN upscale complete" if result.returncode == 0 else f"⚠️ GAN upscale failed"
                log_body = result.log or ""
                full_log = "\n".join(header_log + [log_body])
                if progress_cb:
                    progress_cb(status)
                relocated = relocate_output(result.output_path)
                if relocated or result.output_path:
                    try:
                        outp = Path(relocated or result.output_path)
                        state["seed_controls"]["last_output_dir"] = str(outp.parent if outp.is_file() else outp)
                    except Exception:
                        pass
                run_logger.write_summary(
                    Path(relocated) if relocated else output_dir,
                    {
                        "input": prepped_settings.get("input_path"),
                        "output": relocated or result.output_path,
                        "returncode": result.returncode,
                        "args": prepped_settings,
                        "face_apply": face_apply,
                        "pipeline": "gan",
                    },
                )
                cmp_html = ""
                slider_update = gr.ImageSlider.update(value=None)
                if relocated or result.output_path:
                    src = prepped_settings.get("input_path")
                    outp = relocated or result.output_path
                    if outp and Path(outp).exists():
                        if Path(outp).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                            use_fallback = cmp_mode == "fallback"
                            cmp_html = build_video_comparison(
                                src,
                                outp,
                                pin_reference=pin_pref,
                                start_fullscreen=fs_pref,
                                use_fallback_assets=use_fallback or cmp_mode == "html_slider",
                            )
                        elif Path(outp).is_dir():
                            cmp_html = f"<p>PNG frames saved to {outp}</p>"
                        else:
                            slider_update = gr.ImageSlider.update(value=(src, outp), visible=True)
                            cmp_html = build_image_comparison(src, outp, pin_reference=pin_pref)
                return status, full_log, relocated if relocated else (result.output_path if result.output_path else None), cmp_html, slider_update

            # Kick off worker thread
            if settings.get("batch_enable"):
                folder = Path(settings["input_path"])
                # Check if this is a frame folder (contains only images, treated as a sequence)
                image_files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")]
                video_files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")]

                if image_files and len(image_files) > 1 and not video_files:
                    # Treat as frame folder - process as single unit
                    items = [folder]
                else:
                    # Regular batch processing - individual files
                    items = image_files + video_files

                if not items:
                    return ("❌ No media files or frame folders found in batch folder", "", None, "", gr.ImageSlider.update(value=None), state)
                t = threading.Thread(target=worker_batch, args=(items,), daemon=True)
            else:
                prepped = maybe_downscale(prepare_single(settings["input_path"]))
                t = threading.Thread(target=worker_single, args=(prepped,), daemon=True)
            t.start()

            last_yield = time.time()
            while t.is_alive() or not progress_q.empty():
                try:
                    line = progress_q.get(timeout=0.2)
                    if line:
                        result_holder.setdefault("live_logs", []).append(line)
                except queue.Empty:
                    pass
                now = time.time()
                if now - last_yield > 0.5:
                    last_yield = now
                    live_logs = result_holder.get("live_logs", [])
                    yield (
                        gr.Markdown.update(value="⏳ Running GAN upscale..."),
                        "\n".join(live_logs[-400:]),
                        None,
                        None,
                        gr.ImageSlider.update(value=None),
                        state
                    )
                time.sleep(0.1)
            t.join()

            status, lg, outp, cmp_html, slider_upd = result_holder.get(
                "payload",
                ("❌ Failed", "", None, "", gr.ImageSlider.update(value=None)),
            )
            live_logs = result_holder.get("live_logs", [])
            merged_logs = lg if lg else "\n".join(live_logs)
            
            # Build video comparison for videos
            video_comp_html_update = gr.HTML.update(value="", visible=False)
            if outp and Path(outp).exists() and Path(outp).suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                original_path = settings.get("input_path", "")
                if original_path and Path(original_path).exists():
                    video_comp_html_value = create_video_comparison_html(
                        original_video=original_path,
                        upscaled_video=outp,
                        height=600,
                        slider_position=50.0
                    )
                    video_comp_html_update = gr.HTML.update(value=video_comp_html_value, visible=True)
            
            yield (
                status,
                merged_logs,
                gr.Markdown.update(value="", visible=False),
                outp if outp and not Path(outp).is_dir() and Path(outp).suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp') else None,
                outp if outp and str(outp).endswith(".mp4") else None,
                f"Output: {outp}" if outp else "No output",
                slider_upd,
                video_comp_html_update,
                gr.Gallery.update(visible=False),
                state
            )
        except Exception as e:
            error_msg = f"Critical error in GAN processing: {str(e)}"
            yield (
                "❌ Critical error",
                error_msg,
                gr.Markdown.update(value="", visible=False),
                None,
                None,
                "Error",
                gr.ImageSlider.update(value=None),
                gr.HTML.update(value="", visible=False),
                gr.Gallery.update(visible=False),
                state or {}
            )

    def cancel_action(state=None):
        cancel_event.set()
        return gr.Markdown.update(value="⏹️ Cancel requested"), "", state or {}

    def get_model_scale(model_name: str) -> int:
        """Get the scale factor for a specific model"""
        if not model_name:
            return 4
        try:
            from shared.gan_runner import get_gan_model_metadata
            meta = get_gan_model_metadata(model_name, Path(defaults["base_dir"]))
            return meta.scale
        except Exception:
            return _parse_scale_from_name(model_name)

    return {
        "defaults": defaults,
        "order": GAN_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": lambda *args: run_action(*args[:-1], args[-1]) if len(args) > 1 else run_action(*args),
        "model_scanner": lambda: _scan_gan_models(base_dir),
        "cancel_action": lambda *args: cancel_action(args[0] if args else None),
        "get_model_scale": get_model_scale,
    }


