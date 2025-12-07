import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
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
from shared.gan_runner import run_gan_upscale
from shared.video_comparison import build_video_comparison, build_image_comparison


GAN_MODEL_EXTS = {".pth", ".safetensors"}
GAN_META_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_key(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _parse_scale_from_name(name: str) -> int:
    import re
    lowered = name.lower()
    m = re.search(r"(\d+)x", lowered)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return 4


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
    _load_gan_catalog(base_dir)
    stem = Path(filename).stem
    norm = _normalize_key(stem)
    meta = GAN_META_CACHE.get(norm, {})
    scale = meta.get("scale") or _parse_scale_from_name(stem)
    canonical = meta.get("name", stem)
    return {"scale": scale, "canonical": canonical}


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
    models_dir = base_dir / "Image_Upscale_Models"
    if not models_dir.exists():
        return []
    files = []
    for f in models_dir.iterdir():
        if f.is_file() and f.suffix.lower() in GAN_MODEL_EXTS:
            files.append(f.name)
    return sorted(files)


def gan_defaults(base_dir: Path) -> Dict[str, Any]:
    models = _scan_gan_models(base_dir)
    default_model = models[0] if models else ""
    meta = _get_gan_meta(default_model, base_dir) if default_model else {"scale": 2}
    return {
        "model": default_model,
        "scale": meta.get("scale", 2),
        "backend": "realesrgan",
        "input_path": "",
        "cuda_device": "0",
        "batch_enable": False,
        "batch_input_path": "",
        "batch_output_path": "",
        "output_format": "auto",
        "output_override": "",
        "use_resolution_tab": True,
        "fps_override": 0,
    }


GAN_ORDER: List[str] = [
    "model",
    "scale",
    "backend",
    "input_path",
    "cuda_device",
    "batch_enable",
    "batch_input_path",
    "batch_output_path",
    "output_format",
    "output_override",
    "use_resolution_tab",
    "fps_override",
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
    seed_controls_cache: Dict[str, Any],
    base_dir: Path,
    temp_dir: Path,
    output_dir: Path,
):
    defaults = gan_defaults(base_dir)

    def refresh_presets(model_name: str, select_name: Optional[str] = None):
        presets = preset_manager.list_presets("gan", model_name)
        last_used = preset_manager.get_last_used_name("gan", model_name)
        preferred = select_name if select_name in presets else None
        value = preferred or (last_used if last_used in presets else (presets[-1] if presets else None))
        return gr.Dropdown.update(choices=presets, value=value)

    def save_preset(preset_name: str, *args):
        if not preset_name:
            return gr.Dropdown.update(), gr.Markdown.update(value="⚠️ Enter a preset name before saving"), *list(args)
        payload = _gan_dict_from_args(list(args))
        model_name = payload["model"]
        preset_manager.save_preset("gan", model_name, preset_name, payload)
        dropdown = refresh_presets(model_name, select_name=preset_name)
        current_map = dict(zip(GAN_ORDER, list(args)))
        loaded_vals = _apply_gan_preset(payload, defaults, preset_manager, current=current_map)
        return dropdown, gr.Markdown.update(value=f"✅ Saved preset '{preset_name}' for {model_name}"), *loaded_vals

    def load_preset(preset_name: str, model_name: str, current_values: List[Any]):
        model_name = model_name or defaults["model"]
        preset = preset_manager.load_preset("gan", model_name, preset_name)
        if preset:
            preset_manager.set_last_used("gan", model_name, preset_name)
        defaults_with_model = defaults.copy()
        defaults_with_model["model"] = model_name
        current_map = dict(zip(GAN_ORDER, current_values))
        values = _apply_gan_preset(preset or {}, defaults_with_model, preset_manager, current=current_map)
        return values

    def safe_defaults():
        return [defaults[k] for k in GAN_ORDER]

    def run_action(upload, *args, preview_only: bool = False):
        settings_dict = _gan_dict_from_args(list(args))
        settings = {**defaults, **settings_dict}
        settings["output_override"] = settings.get("output_override")
        settings["cuda_device"] = settings.get("cuda_device", "")

        inp = normalize_path(upload if upload else settings["input_path"])
        if settings.get("batch_enable"):
            if not inp or not Path(inp).exists() or not Path(inp).is_dir():
                return ("❌ Batch input folder missing", "", None, "", gr.ImageSlider.update(value=None))
        else:
            if not inp or not Path(inp).exists():
                return ("❌ Input missing", "", None, "", gr.ImageSlider.update(value=None))

        settings["input_path"] = inp

        cuda_warn = _validate_cuda_devices(settings.get("cuda_device", ""))
        if cuda_warn:
            return (f"⚠️ {cuda_warn}", "", None, "")

        face_apply = bool(args[-1])
        face_apply = face_apply or global_settings.get("face_global", False)
        backend_val = settings.get("backend", "realesrgan")

        def prepare_single(single_path: str) -> Dict[str, Any]:
            s = settings.copy()
            s["input_path"] = normalize_path(single_path)
            meta = _get_gan_meta(s.get("model", ""), base_dir)
            s["scale"] = meta.get("scale", s.get("scale", 2))
            s["model_name"] = meta.get("canonical", s.get("model", ""))
            return s

        def maybe_downscale(s):
            if s.get("use_resolution_tab") and seed_controls_cache.get("resolution_val"):
                target_short = seed_controls_cache.get("resolution_val")
                dims = get_media_dimensions(s["input_path"])
                if dims and target_short:
                    short_side = min(dims)
                    desired_input_short = target_short / max(1, s["scale"])
                    if desired_input_short < short_side:
                        tmp_path = Path(temp_dir) / f"gan_downscale_{Path(s['input_path']).stem}.mp4"
                        if Path(s["input_path"]).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
                            subprocess.run(
                                [
                                    "ffmpeg",
                                    "-y",
                                    "-i",
                                    s["input_path"],
                                    "-vf",
                                    f"scale=-2:{int(desired_input_short)}",
                                    str(tmp_path),
                                ],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                            if tmp_path.exists():
                                s["input_path"] = str(tmp_path)
                        else:
                            try:
                                import cv2

                                img = cv2.imread(s["input_path"], cv2.IMREAD_UNCHANGED)
                                if img is not None:
                                    h, w = img.shape[:2]
                                    scale_factor = desired_input_short / float(short_side)
                                    new_w = int(w * scale_factor)
                                    new_h = int(h * scale_factor)
                                    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                    tmp_img = Path(temp_dir) / f"gan_downscale_{Path(s['input_path']).name}"
                                    cv2.imwrite(str(tmp_img), resized)
                                    s["input_path"] = str(tmp_img)
                            except Exception:
                                pass
            return s

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

        def run_single(prepped_settings: Dict[str, Any], progress_cb: Optional[Callable[[str], None]] = None):
            header_log = [
                f"Model: {prepped_settings['model_name']}",
                f"Backend: {backend_val}",
                f"Scale: {prepped_settings['scale']}x",
                f"Input: {prepped_settings['input_path']}",
            ]
            if progress_cb:
                for line in header_log:
                    progress_cb(line)
            # Run backend and collect logs
            try:
                if backend_val == "realesrgan":
                    prepped_settings["model"] = prepped_settings["model_name"].split(".")[0]
                    prepped_settings["model_path"] = str((base_dir / "Image_Upscale_Models" / prepped_settings.get("model_name")).resolve())
                    if not Path(prepped_settings["model_path"]).exists() and not _is_realesrgan_builtin(prepped_settings["model_name"]):
                        return ("❌ Model weights not found", "\n".join(header_log + ["Missing model file."]), None, "", gr.ImageSlider.update(value=None))
                    result = run_realesrgan(prepped_settings, apply_face=face_apply)
                else:
                    result = run_gan_upscale(prepped_settings, apply_face=face_apply)
            except Exception as exc:  # surface ffmpeg or other runtime issues
                err_msg = f"❌ GAN upscale failed: {exc}"
                if progress_cb:
                    progress_cb(err_msg)
                return (err_msg, "\n".join(header_log + [str(exc)]), None, "", gr.ImageSlider.update(value=None))
            status = "✅ GAN upscale complete" if result.returncode == 0 else f"⚠️ GAN upscale failed"
            log_body = result.log or ""
            full_log = "\n".join(header_log + [log_body])
            if progress_cb:
                progress_cb(status)
            relocated = relocate_output(result.output_path)
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
                        cmp_html = build_video_comparison(src, outp)
                    elif Path(outp).is_dir():
                        cmp_html = f"<p>PNG frames saved to {outp}</p>"
                    else:
                        slider_update = gr.ImageSlider.update(value=(src, outp), visible=True)
                        cmp_html = build_image_comparison(src, outp)
            return status, full_log, relocated if relocated else (result.output_path if result.output_path else None), cmp_html, slider_update

        # Streaming: run in background thread, stream log lines if available
        progress_q: "queue.Queue[str]" = queue.Queue()
        result_holder: Dict[str, Any] = {}

        def worker_single(prepped_settings):
            status, lg, outp, cmp_html, slider_upd = run_single(prepped_settings, progress_cb=progress_q.put)
            result_holder["payload"] = (status, lg, outp, cmp_html, slider_upd)

        def worker_batch(batch_items):
            outputs = []
            logs = []
            last_cmp = ""
            last_slider = gr.ImageSlider.update(value=None)
            for item in batch_items:
                ps = maybe_downscale(prepare_single(str(item)))
                status, lg, outp, cmp_html, slider_upd = run_single(ps, progress_cb=progress_q.put)
                logs.append(lg)
                if outp:
                    outputs.append(outp)
                if cmp_html:
                    last_cmp = cmp_html
                last_slider = slider_upd
                progress_q.put(f"Processed {len(outputs)}/{len(batch_items)} items")
            final_out = outputs[-1] if outputs else None
            result_holder["payload"] = (
                "✅ Batch complete" if outputs else "⚠️ Batch finished with errors",
                "\n\n".join(logs),
                final_out,
                last_cmp,
                last_slider,
            )

        # Kick off worker thread
        if settings.get("batch_enable"):
            folder = Path(settings["input_path"])
            items = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")]
            if not items:
                return ("❌ No media files found in batch folder", "", None, "", gr.ImageSlider.update(value=None))
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
                    "",
                    gr.ImageSlider.update(value=None),
                )
            time.sleep(0.1)
        t.join()

        status, lg, outp, cmp_html, slider_upd = result_holder.get(
            "payload",
            ("❌ Failed", "", None, "", gr.ImageSlider.update(value=None)),
        )
        live_logs = result_holder.get("live_logs", [])
        merged_logs = lg if lg else "\n".join(live_logs)
        yield (
            status,
            merged_logs,
            outp if outp and str(outp).endswith(".mp4") else outp,
            cmp_html,
            slider_upd,
        )

    return {
        "defaults": defaults,
        "order": GAN_ORDER,
        "refresh_presets": refresh_presets,
        "save_preset": save_preset,
        "load_preset": load_preset,
        "safe_defaults": safe_defaults,
        "run_action": run_action,
        "model_scanner": lambda: _scan_gan_models(base_dir),
    }


